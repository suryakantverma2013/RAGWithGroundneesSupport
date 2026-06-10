"""
ingest_pdf.py — PDF → Milvus hybrid-RAG ingestion
===================================================
Parses a PDF with Docling, OCRs images with Tesseract (when installed),
generates OpenAI dense embeddings, and writes chunks to a
MilvusDocumentStore that also builds a BM25 sparse index via
Milvus's built-in BM25 function.

Index design
------------
  Dense  (HNSW / COSINE)           — OpenAI text-embedding-3-large, 3 072-dim
  Sparse (SPARSE_INVERTED_INDEX)   — Milvus BM25BuiltInFunction, server-side

Chunk strategy
--------------
  Text   → HybridChunker (structure-aware, ≤ MAX_CHUNK_TOKENS tokens)
  Tables → atomic Markdown chunk (never split mid-row)
  Images → Tesseract OCR → text chunk (skipped if pytesseract not installed)

Metadata stored per chunk
--------------------------
  document_name   source file name (stem, no extension)
  source_path     absolute path of the PDF on disk
  chunk_type      "text" | "table" | "image_ocr"
  embedding_time  ISO-8601 UTC timestamp of this ingestion run
  page_count      number of distinct PDF pages this chunk spans
  page_numbers    sorted list of 1-based page numbers

The document ID (primary key in Milvus) is a deterministic SHA-256 hash of
(document_name, chunk_index, content prefix).  Before writing, any existing
chunks with the same document_name are deleted, so re-ingesting the same PDF
replaces its chunks cleanly — no duplicates, and other documents in the
collection are untouched.  (Milvus does not enforce primary-key uniqueness,
so deterministic IDs alone are not enough.)

Usage
-----
    python ingest_pdf.py <pdf_path> <collection_name> [options]

    # Append to an existing collection (default):
    python ingest_pdf.py report.pdf my_docs

    # Drop & recreate the collection before ingesting:
    python ingest_pdf.py report.pdf my_docs --drop-old

    # Scanned (image-only) PDF — enable full-page OCR:
    python ingest_pdf.py scan.pdf my_docs --page-ocr

    # Logging (see logging_setup.py): default is INFO to a size-rotated
    # logs/ingest_pdf.log (ns timestamps).  Log to console / troubleshoot:
    python ingest_pdf.py report.pdf my_docs --log-dest console --log-level DEBUG

Environment variables
---------------------
    OPENAI_API_KEY        required (or --openai-api-key flag)
    MILVUS_URI            optional, default http://localhost:19530
    LANGFUSE_SECRET_KEY   optional — enables Langfuse cost/token tracing
    LANGFUSE_PUBLIC_KEY   optional — required alongside LANGFUSE_SECRET_KEY
    LANGFUSE_HOST         optional, default http://localhost:3000

Optional system dependency
--------------------------
    Tesseract OCR binary (for image text extraction)
      Windows : https://github.com/UB-Mannheim/tesseract/wiki
      macOS   : brew install tesseract
      Linux   : apt-get install tesseract-ocr
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Windows: force UTF-8 stdout ───────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── .env: load OPENAI_API_KEY / LANGFUSE_* / MILVUS_URI from the project root ─
# Must run before any os.environ reads below.  Existing shell vars win
# (load_dotenv does not override already-set variables).
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Haystack's OpenAI components default to a 30 s request timeout (OPENAI_TIMEOUT
# env var) — too tight for large embedding batches.  A value set in .env or the
# shell still wins.
os.environ.setdefault("OPENAI_TIMEOUT", "120")

# ── logging: shared setup (ns timestamps, rotating file under logs/, noise
# capping).  Must run before the heavy imports below so import-time messages
# (Tesseract probe, Langfuse activation) are captured.  --log-dest/--log-level
# are pre-parsed from sys.argv here; main()'s parser re-declares them for
# --help.  See logging_setup.py.
from logging_setup import add_logging_args, setup_logging_from_argv

setup_logging_from_argv("ingest_pdf.log")
log = logging.getLogger(__name__)

# ── optional: Tesseract OCR ───────────────────────────────────────────────────
_OCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image as _PILImage  # noqa: F401
    _OCR_AVAILABLE = True
    log.info("Tesseract OCR available — image text will be indexed")
except ImportError:
    log.warning(
        "pytesseract / Pillow not installed — image text will NOT be indexed.\n"
        "  Install: uv add pytesseract Pillow\n"
        "  Then install the Tesseract binary (see module docstring)."
    )

# ── optional: Langfuse tracing ────────────────────────────────────────────────
# Shared setup (env defaults + custom span handler) lives in langfuse_tracing.py.
# Activated automatically when LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY are set.
# Must run BEFORE the haystack imports below so content tracing is configured.
from langfuse_tracing import enable_langfuse

_LANGFUSE_ENABLED = enable_langfuse("ingest_pdf")

# ── docling ───────────────────────────────────────────────────────────────────
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

# ── tiktoken (token-accurate chunking for the OpenAI embedding model) ─────────
import tiktoken

# ── haystack + milvus ─────────────────────────────────────────────────────────
from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.function import BM25BuiltInFunction

# ─────────────────────────────────────────────────────────────────────────────
# Tuneable constants
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"   # 3 072-dim, 8 191-token limit
MAX_CHUNK_TOKENS = 512    # comfortably within the 8 191-token ceiling
EMBED_BATCH_SIZE = 32     # chunks per OpenAI API call
MIN_OCR_CHARS = 30        # discard OCR strings shorter than this (noise)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _page_numbers(doc_items: list | None) -> set[int]:
    """Collect the set of PDF page numbers referenced by a list of DocItems."""
    pages: set[int] = set()
    for item in doc_items or []:
        for prov in getattr(item, "prov", None) or []:
            pno = getattr(prov, "page_no", None)
            if pno is not None:
                pages.add(pno)
    return pages


def _chunk_id(doc_name: str, idx: int, text: str) -> str:
    """
    Deterministic chunk ID — SHA-256 of (doc_name, idx, content prefix).
    Enables idempotent re-ingestion: re-running on the same PDF overwrites
    existing chunks instead of creating duplicates.
    """
    raw = f"{doc_name}::{idx}::{text[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: PDF → DoclingDocument
# ─────────────────────────────────────────────────────────────────────────────
def _build_converter(page_ocr: bool = False) -> DocumentConverter:
    opts = PdfPipelineOptions()
    # Full-page OCR is only needed for scanned PDFs; born-digital PDFs already
    # carry a text layer, and the OCR pass is very slow on CPU.  Enable with
    # --page-ocr.  (Independent of the Tesseract *figure* OCR below.)
    opts.do_ocr = page_ocr
    opts.generate_picture_images = _OCR_AVAILABLE  # only extract images if we'll OCR
    opts.images_scale = 2.0  # 2× resolution improves OCR accuracy on small text
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2a: Text + Table chunks
# HybridChunker is structure-aware:
#   - Respects headings and paragraph boundaries
#   - Tables are kept as atomic Markdown chunks (never split mid-row)
#   - Adjacent small chunks are merged up to MAX_CHUNK_TOKENS
# ─────────────────────────────────────────────────────────────────────────────
def _text_table_chunks(
    dl_doc,
    *,
    doc_name: str,
    source_path: str,
    embedding_time: str,
) -> list[Document]:
    # OpenAITokenizer wraps tiktoken so chunk-size limits are computed with the
    # exact same tokenizer as the target embedding model.  (docling-core ≥ 2.x
    # requires a BaseTokenizer instance here; max_tokens lives on the tokenizer.)
    chunker = HybridChunker(
        tokenizer=OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model(EMBEDDING_MODEL),
            max_tokens=MAX_CHUNK_TOKENS,
        ),
        merge_peers=True,
    )

    docs: list[Document] = []
    for idx, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
        text = (chunk.text or "").strip()
        if not text:
            continue

        doc_items = getattr(getattr(chunk, "meta", None), "doc_items", None)
        pages = _page_numbers(doc_items)

        chunk_type = "text"
        try:
            from docling.datamodel.document import TableItem
            if any(isinstance(it, TableItem) for it in (doc_items or [])):
                chunk_type = "table"
        except ImportError:
            pass

        docs.append(
            Document(
                id=_chunk_id(doc_name, idx, text),
                content=text,
                meta={
                    "document_name": doc_name,
                    "source_path": source_path,
                    "chunk_type": chunk_type,
                    "embedding_time": embedding_time,
                    "page_count": len(pages) if pages else 1,
                    "page_numbers": sorted(pages),
                },
            )
        )
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2b: Image OCR chunks
# Each PictureItem is OCR'd; results with >= MIN_OCR_CHARS become their own chunk.
# Diagrams/photos with no embedded text are silently discarded.
# ─────────────────────────────────────────────────────────────────────────────
def _image_ocr_chunks(
    dl_doc,
    *,
    doc_name: str,
    source_path: str,
    embedding_time: str,
) -> list[Document]:
    if not _OCR_AVAILABLE:
        return []

    try:
        from docling.datamodel.document import PictureItem
    except ImportError:
        log.warning("Cannot import PictureItem from docling — image OCR skipped")
        return []

    docs: list[Document] = []
    img_idx = 0

    for item, _ in dl_doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        img_idx += 1

        pil_img = None
        try:
            img_obj = getattr(item, "image", None)
            pil_img = getattr(img_obj, "pil_image", None)
        except Exception:
            pass

        if pil_img is None:
            log.debug("Image %d: no pixel data — skipped", img_idx)
            continue

        try:
            ocr_text = pytesseract.image_to_string(pil_img, lang="eng").strip()
        except Exception as exc:
            log.warning("Image %d: OCR error — %s", img_idx, exc)
            continue

        if len(ocr_text) < MIN_OCR_CHARS:
            log.debug(
                "Image %d: OCR produced %d chars (< %d) — discarded",
                img_idx, len(ocr_text), MIN_OCR_CHARS,
            )
            continue

        pages = _page_numbers([item])
        log.info("Image %d: %d OCR chars indexed (pages %s)", img_idx, len(ocr_text), sorted(pages))

        content = f"[Image OCR — figure {img_idx}]\n{ocr_text}"
        docs.append(
            Document(
                id=_chunk_id(doc_name, 100_000 + img_idx, content),
                content=content,
                meta={
                    "document_name": doc_name,
                    "source_path": source_path,
                    "chunk_type": "image_ocr",
                    "embedding_time": embedding_time,
                    "page_count": len(pages) if pages else 1,
                    "page_numbers": sorted(pages),
                    "image_index": img_idx,
                },
            )
        )
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: MilvusDocumentStore
#   vector_field        → dense HNSW index (OpenAI embeddings)
#   sparse_vector_field → BM25 SPARSE_INVERTED_INDEX (Milvus server-side)
#   BM25BuiltInFunction → tokenises text_field → populates sparse_vector_field
#
# NOTE: BM25BuiltInFunction requires Milvus Standalone or Distributed.
#       It is NOT available in Milvus Lite.
# ─────────────────────────────────────────────────────────────────────────────
def _build_store(
    milvus_uri: str,
    collection_name: str,
    drop_old: bool,
) -> MilvusDocumentStore:
    return MilvusDocumentStore(
        connection_args={"uri": milvus_uri},
        collection_name=collection_name,
        vector_field="vector",
        sparse_vector_field="sparse_vector",
        text_field="text",
        builtin_function=[
            BM25BuiltInFunction(
                input_field_names="text",
                output_field_names="sparse_vector",
                # Keep this analyzer identical to Milvus_Collection_With_Fields.py
                # and haystack_milvus_hybrid_rag.py so all three agree on
                # tokenisation (applies only when THIS call creates the
                # collection; when attaching to an existing one it is a no-op).
                analyzer_params={
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        {"type": "stop", "stop_words": ["a", "an", "the", "is", "of"]},
                    ],
                },
                enable_match=True,
            )
        ],
        consistency_level="Bounded",
        drop_old=drop_old,
    )


def _delete_existing_chunks(store: MilvusDocumentStore, doc_name: str) -> None:
    """
    Delete previously ingested chunks of *doc_name* so re-ingestion replaces
    them instead of duplicating.  Milvus does not enforce primary-key
    uniqueness and milvus-haystack ignores DuplicatePolicy.OVERWRITE, so this
    pre-delete is what makes re-runs idempotent.  Other documents in the
    collection are untouched.
    """
    col = getattr(store, "col", None)
    if col is None:  # collection doesn't exist yet — nothing to clean up
        return
    res = col.delete(expr=f'document_name == "{doc_name}"')
    deleted = getattr(res, "delete_count", 0) or 0
    if deleted:
        log.info("Re-ingestion: deleted %d existing chunks of '%s'", deleted, doc_name)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Indexing pipeline
#   OpenAIDocumentEmbedder → adds .embedding to each Document
#   DocumentWriter         → upserts into MilvusDocumentStore
#                            (BM25 sparse index built server-side on insert)
# ─────────────────────────────────────────────────────────────────────────────
def _build_pipeline(store: MilvusDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            model=EMBEDDING_MODEL,
            batch_size=EMBED_BATCH_SIZE,
            progress_bar=True,
        ),
    )
    pipeline.add_component(
        "writer",
        DocumentWriter(
            document_store=store,
            # Milvus only supports NONE; idempotency comes from the pre-delete
            # of this document's chunks in _delete_existing_chunks().
            policy=DuplicatePolicy.NONE,
        ),
    )
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def ingest(
    pdf_path: str | Path,
    collection_name: str,
    milvus_uri: str = "http://localhost:19530",
    drop_old: bool = False,
    page_ocr: bool = False,
) -> int:
    """
    Parse *pdf_path*, chunk it, embed with OpenAI, and write to Milvus.

    Parameters
    ----------
    pdf_path        : path to the PDF file
    collection_name : target Milvus collection (created if it doesn't exist)
    milvus_uri      : Milvus connection URI
    drop_old        : if True, drop and recreate the collection before ingesting
    page_ocr        : if True, run full-page OCR (needed only for scanned PDFs)

    Returns
    -------
    Number of chunks written to the collection.

    Raises
    ------
    FileNotFoundError : if *pdf_path* does not exist
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_name = pdf_path.stem
    embedding_time = datetime.now(timezone.utc).isoformat()

    # 1 ── Parse ──────────────────────────────────────────────────────────────
    log.info("── Parsing %s (page_ocr=%s) ──", pdf_path.name, page_ocr)
    dl_doc = _build_converter(page_ocr).convert(str(pdf_path)).document
    page_total = len(getattr(dl_doc, "pages", {}) or {})
    log.info("Parsed: %d pages", page_total)

    # 2 ── Chunk ──────────────────────────────────────────────────────────────
    log.info("── Chunking text + tables (max %d tokens/chunk) ──", MAX_CHUNK_TOKENS)
    text_chunks = _text_table_chunks(
        dl_doc,
        doc_name=doc_name,
        source_path=str(pdf_path),
        embedding_time=embedding_time,
    )
    table_count = sum(1 for d in text_chunks if d.meta.get("chunk_type") == "table")
    log.info(
        "Produced %d chunks  (%d tables, %d text)",
        len(text_chunks), table_count, len(text_chunks) - table_count,
    )

    log.info("── OCR-ing images ──")
    img_chunks = _image_ocr_chunks(
        dl_doc,
        doc_name=doc_name,
        source_path=str(pdf_path),
        embedding_time=embedding_time,
    )
    log.info("Produced %d image-OCR chunks", len(img_chunks))

    all_docs = text_chunks + img_chunks
    if not all_docs:
        log.warning("No content extracted — nothing indexed.")
        return 0

    log.info("Total chunks to embed + index: %d", len(all_docs))

    # 3 ── Connect to Milvus ──────────────────────────────────────────────────
    log.info("── Milvus: %s / collection='%s' (drop_old=%s) ──",
             milvus_uri, collection_name, drop_old)
    store = _build_store(milvus_uri, collection_name, drop_old)
    if not drop_old:
        _delete_existing_chunks(store, doc_name)

    # 4 ── Embed + write ──────────────────────────────────────────────────────
    log.info("── Embedding with %s + writing to Milvus ──", EMBEDDING_MODEL)
    result = _build_pipeline(store).run({"embedder": {"documents": all_docs}})
    written = result["writer"]["documents_written"]
    log.info(
        "Done.  %d chunks written.  Collection total: %d documents.",
        written,
        store.count_documents(),
    )
    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into Milvus with dense (OpenAI) + sparse (BM25) hybrid indexing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf_path", help="Full path to the PDF file")
    parser.add_argument("collection_name", help="Target Milvus collection name")
    parser.add_argument(
        "--milvus-uri",
        default=os.environ.get("MILVUS_URI", "http://localhost:19530"),
        metavar="URI",
        help="Milvus endpoint  (env: MILVUS_URI)",
    )
    parser.add_argument(
        "--drop-old",
        action="store_true",
        help="Drop and recreate the collection before ingesting",
    )
    parser.add_argument(
        "--page-ocr",
        action="store_true",
        help="Run full-page OCR — only needed for scanned PDFs (slow on CPU)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,  # resolved from env after parsing — never echo the key in --help
        metavar="KEY",
        help="OpenAI API key  (env: OPENAI_API_KEY)",
    )
    # --log-dest / --log-level — already applied at import time by
    # setup_logging_from_argv(); declared here so they appear in --help.
    add_logging_args(parser)
    args = parser.parse_args()

    args.openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not args.openai_api_key:
        parser.error(
            "OpenAI API key required.  Set the OPENAI_API_KEY env var or use --openai-api-key."
        )
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    try:
        ingest(
            pdf_path=args.pdf_path,
            collection_name=args.collection_name,
            milvus_uri=args.milvus_uri,
            drop_old=args.drop_old,
            page_ocr=args.page_ocr,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception:
        log.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
