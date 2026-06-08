"""
Milvus_Collection_With_Fields.py — Step 1: create the hybrid collection
========================================================================
Creates the Milvus collection that the rest of the pipeline attaches to:

    ingest_pdf.py                  → populates it  (dense embeddings + BM25)
    haystack_milvus_hybrid_rag.py  → queries it

The schema is built directly with the native pymilvus client so the hybrid
(dense + BM25) layout is explicit and inspectable.  Crucially, the field names,
types, and analyzer settings MATCH exactly what Haystack's MilvusDocumentStore
expects, so ingest_pdf.py can attach to this collection (drop_old=False) and
write to it without a schema conflict:

    id             VARCHAR  primary key, auto_id=False
                            ← Haystack writes string Document IDs (SHA-256 hex)
    text           VARCHAR  analyzer enabled — BM25 reads from this field
    vector         FLOAT_VECTOR(1536)  OpenAI text-embedding-3-small (dense)
    sparse_vector  SPARSE_FLOAT_VECTOR  BM25 output (populated server-side)
    + dynamic field          all Document.meta keys (document_name,
                             page_numbers, chunk_type, …) — no need to predeclare

A BM25 Function tokenises `text` into `sparse_vector`.  Indexes are HNSW/COSINE
(dense) and SPARSE_INVERTED_INDEX/BM25 (sparse).

Requirements
------------
Milvus Standalone or Distributed — BM25 is NOT available in Milvus Lite.

Usage
-----
    uv run python Milvus_Collection_With_Fields.py            # create if absent
    uv run python Milvus_Collection_With_Fields.py --drop-old # recreate

Re-running is a no-op when the collection already exists (it is NOT dropped
unless --drop-old is given), so it's safe to call before every ingestion.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import DataType, Function, FunctionType, MilvusClient

# MILVUS_URI may live in the project-root .env (shared with the other scripts).
load_dotenv(Path(__file__).parent / ".env")

COLLECTION_NAME = "hybrid_rag_docs"
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small

# Canonical BM25 analyzer — MUST stay identical to the analyzer_params passed to
# BM25BuiltInFunction in ingest_pdf.py and haystack_milvus_hybrid_rag.py so all
# three scripts agree on how `text` is tokenised.  (Haystack applies these to
# the text field as enable_analyzer / enable_match / analyzer_params.)
ANALYZER_PARAMS = {
    "tokenizer": "standard",
    "filter": [
        "lowercase",
        {"type": "stop", "stop_words": ["a", "an", "the", "is", "of"]},
    ],
}


def create_collection(uri: str, drop_old: bool) -> None:
    client = MilvusClient(uri=uri)

    if client.has_collection(COLLECTION_NAME):
        if not drop_old:
            print(
                f"Collection '{COLLECTION_NAME}' already exists — leaving it as is.\n"
                f"Pass --drop-old to recreate it, or proceed to ingestion:\n"
                f"    uv run python ingest_pdf.py <pdf_path> {COLLECTION_NAME}"
            )
            return
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'.")

    # 1. Schema
    #    auto_id=False         — Haystack supplies the (string) primary key.
    #    enable_dynamic_field  — Document.meta is stored without predeclaring keys.
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)

    # Primary key — VARCHAR because Haystack Document IDs are strings (SHA-256 hex).
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=65535)

    # Raw text — BM25 reads from here, so the analyzer must be enabled.  These
    # three kwargs mirror what Haystack's BM25BuiltInFunction sets on the field.
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,            # required for BM25 tokenisation
        enable_match=True,               # keyword-match inverted index
        analyzer_params=ANALYZER_PARAMS,
    )

    # Dense vector — named 'vector' to match Haystack's vector_field="vector".
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    # Sparse vector — BM25 writes here; never insert values manually.
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # 2. BM25 Function — tokenise `text` and populate `sparse_vector`.
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    )

    # 3. Indexes
    index_params = client.prepare_index_params()

    # Dense HNSW / COSINE (OpenAI embeddings are well-suited to cosine similarity).
    index_params.add_index(
        field_name="vector",
        index_name="dense_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 32, "efConstruction": 200},
    )

    # Sparse BM25 — metric_type BM25 is valid because the BM25 Function is attached.
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "bm25_k1": 1.2,   # term-frequency saturation (default 1.2)
            "bm25_b": 0.75,   # document-length normalisation (default 0.75)
        },
    )

    # 4. Create the collection.
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(
        f"Collection '{COLLECTION_NAME}' created "
        f"(dense HNSW/COSINE + BM25 sparse, dynamic meta field).\n"
        f"Next: uv run python ingest_pdf.py <pdf_path> {COLLECTION_NAME}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the hybrid (dense + BM25) Milvus collection that "
        "ingest_pdf.py populates and haystack_milvus_hybrid_rag.py queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--milvus-uri",
        default=os.environ.get("MILVUS_URI", "http://localhost:19530"),
        metavar="URI",
        help="Milvus endpoint (env: MILVUS_URI)",
    )
    parser.add_argument(
        "--drop-old",
        action="store_true",
        help="Drop and recreate the collection if it already exists",
    )
    args = parser.parse_args()
    create_collection(args.milvus_uri, args.drop_old)


if __name__ == "__main__":
    main()
