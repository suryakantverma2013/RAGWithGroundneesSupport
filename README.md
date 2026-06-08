# RAG with Groundedness Support

A **hybrid Retrieval-Augmented Generation** pipeline built on **Haystack 2.x** and
**Milvus 2.5**, with **query-quality-driven branching** and a **grounded retrieval
feedback loop**.

Given a PDF, it parses and chunks the document (text, tables, and image OCR),
embeds the chunks with OpenAI, and indexes them in Milvus using **both** a dense
vector index (HNSW / cosine) **and** a sparse **BM25** index. At query time it:

1. **Analyzes the query** with an LLM — classifies it as `simple` / `vague` /
   `complex`, then *refines* a vague query or *decomposes* a complex one into
   focused sub-queries.
2. **Retrieves** with `MilvusHybridRetriever` (dense ANN + BM25, fused with RRF)
   once per sub-query and merges the results.
3. Runs a **bounded feedback loop**: scores the retrieved context for relevance
   and, if it's too low, re-analyzes the query (with the failure as a hint) and
   retries — stopping on a relevance threshold, attempt cap, plateau, or stable
   doc set so it always terminates with an answer.
4. **Generates** a grounded answer and **evaluates** it for *faithfulness* and
   *context relevance*.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
   PDF ──► ingest_pdf.py │ Docling parse → chunk (text/table/OCR)       │
                         │   → OpenAI embed → MilvusDocumentStore        │──► Milvus
                         │   (dense HNSW + BM25 sparse, server-side)     │   collection
                         └─────────────────────────────────────────────┘  "hybrid_rag_docs"
                                                                                  │
                                                                                  ▼
                         ┌─────────────────────────────────────────────────────────────┐
  question ──► haystack_ │  query analyzer (refine / decompose)                          │
              milvus_    │     → hybrid retrieval (per sub-query, merged)                │──► answer
              hybrid_    │     → relevance feedback loop (retry if low)                  │  + metrics
              rag.py     │     → generate → faithfulness + context-relevance eval        │
                         └─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.14** | Pinned in `.python-version`; `uv` will fetch it automatically. |
| **[uv](https://docs.astral.sh/uv/)** | Recommended for dependency + Python management (`uv.lock` is committed). `pip` works too. |
| **Docker** | To run **Milvus Standalone** (BM25 needs Standalone/Distributed — **not** Milvus Lite). |
| **OpenAI API key** | Used for embeddings, generation, and the LLM evaluators. |
| **Tesseract OCR** *(optional)* | Only needed to index text inside images. |
| **Langfuse** *(optional)* | LLM cost/token tracing — see [`LANGFUSE_SETUP.md`](LANGFUSE_SETUP.md). |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/suryakantverma2013/RAGWithGroundneesSupport.git
cd RAGWithGroundneesSupport
```

### 2. Install dependencies

**With uv (recommended)** — creates the virtualenv, fetches Python 3.14, and
installs everything from `uv.lock`:

```bash
uv sync
```

**With pip** (alternative):

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -e .
```

### 3. Start Milvus (Standalone, via Docker)

```bash
docker run -d --name milvus-standalone ^
    -p 19530:19530 -p 9091:9091 ^
    milvusdb/milvus:v2.5.0 milvus run standalone
```

> On PowerShell use a backtick `` ` `` for line continuation, or just put it on
> one line. On Linux/macOS use `\`.

### 4. Configure secrets (`.env`)

Create a `.env` file in the project root. **It is git-ignored — never commit it.**

```dotenv
# Required
OPENAI_API_KEY=sk-proj-...

# Optional (defaults shown)
MILVUS_URI=http://localhost:19530

# Optional — enable Langfuse tracing (all three must be set)
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

All scripts call `load_dotenv()` at startup, so these are picked up
automatically. Already-set shell variables take precedence over `.env`.

### 5. *(Optional)* Tesseract OCR — to index image text

`ingest_pdf.py` will OCR images **only** if the Tesseract binary is installed.
Without it, image text is skipped (text and tables still index fine).

- **Windows:** https://github.com/UB-Mannheim/tesseract/wiki
- **macOS:** `brew install tesseract`
- **Linux:** `apt-get install tesseract-ocr`

### 6. *(Optional)* Langfuse — LLM observability

Follow [`LANGFUSE_SETUP.md`](LANGFUSE_SETUP.md) to start Langfuse locally with
`docker-compose.langfuse.yml` and generate the `pk-lf-` / `sk-lf-` keys, then add
them to `.env`. Tracing is fully opt-in — the scripts run identically without it.

---

## Running the scripts

A clean **three-step linear sequence**: create the collection, ingest a PDF,
then query it.

```
   1. Milvus_Collection_With_Fields.py  ──►  creates the "hybrid_rag_docs" schema
                              │
   2. ingest_pdf.py  ─────────┴──►  populates it (dense embeddings + BM25)
                              │
   3. haystack_milvus_hybrid_rag.py  ──►  queries it + evaluates groundedness
```

> The query script hard-codes the collection name **`hybrid_rag_docs`**, so all
> three steps use that collection for the demo to work out of the box.

### Step 1 — `Milvus_Collection_With_Fields.py` (create the collection)

Builds the hybrid Milvus collection schema **directly with the native `pymilvus`
client**, so the layout is explicit and inspectable: a string `VARCHAR` primary
key (`auto_id=False`), an analyzer-enabled `text` field, a dense
`vector FLOAT_VECTOR(1536)`, a `sparse_vector SPARSE_FLOAT_VECTOR`, a dynamic
field for metadata, and a BM25 `Function` that auto-populates the sparse field.
The field names, types, and analyzer settings **match exactly** what Haystack's
`MilvusDocumentStore` expects, so Step 2 attaches to this collection and writes
to it without a schema conflict.

```bash
uv run python Milvus_Collection_With_Fields.py            # create if absent
uv run python Milvus_Collection_With_Fields.py --drop-old # recreate from scratch
```

Re-running is a safe no-op when the collection already exists (it is not dropped
unless `--drop-old` is given).

> **Optional:** `ingest_pdf.py` (Step 2) will auto-create the collection with the
> identical schema if you skip Step 1. Running Step 1 first just gives you
> explicit control over (and a clear view of) the schema and indexes.

### Step 2 — `ingest_pdf.py` (build the index)

Parses a PDF, chunks it (structure-aware text + atomic tables + optional image
OCR), embeds with `text-embedding-3-small`, and writes to Milvus with both the
dense and BM25 sparse indexes. Re-running on the same PDF **replaces** that
document's chunks (idempotent), leaving other documents untouched.

```bash
# Ingest into the collection the query script expects:
uv run python ingest_pdf.py "C:\path\to\paper.pdf" hybrid_rag_docs
```

Common options:

```bash
# Drop & recreate the collection before ingesting:
uv run python ingest_pdf.py paper.pdf hybrid_rag_docs --drop-old

# Scanned (image-only) PDF — enable full-page OCR (slow on CPU):
uv run python ingest_pdf.py scan.pdf hybrid_rag_docs --page-ocr

# Troubleshoot — stream logs to the console at DEBUG:
uv run python ingest_pdf.py paper.pdf hybrid_rag_docs --log-dest console --log-level DEBUG
```

| Argument / flag | Meaning |
|---|---|
| `pdf_path` *(positional)* | Full path to the PDF. |
| `collection_name` *(positional)* | Target Milvus collection (use `hybrid_rag_docs`). |
| `--drop-old` | Drop and recreate the collection first. |
| `--page-ocr` | Full-page OCR for scanned PDFs (needs Tesseract). |
| `--milvus-uri` | Override `MILVUS_URI` (default `http://localhost:19530`). |
| `--openai-api-key` | Override `OPENAI_API_KEY`. |
| `--log-dest` / `--log-level` | `file` (default, under `logs/`) or `console`; `INFO`/`DEBUG`/`WARN`. |

### Step 3 — `haystack_milvus_hybrid_rag.py` (query + evaluate)

Runs the query-analyzer → hybrid-retrieval → feedback-loop → generation pipeline
against the populated collection, and prints a **RUN SUMMARY** with each answer
plus faithfulness / context-relevance metrics.

```bash
# Ask your own question(s) — each positional arg is a separate query:
uv run python haystack_milvus_hybrid_rag.py "What is the 'aha moment' in DeepSeek-R1?"

uv run python haystack_milvus_hybrid_rag.py ^
    "Compare DeepSeek-R1-Zero and DeepSeek-R1, and explain the aha moment"

# No arguments → runs a built-in DeepSeek-R1 demo question set:
uv run python haystack_milvus_hybrid_rag.py

# See the analyzer + feedback loop in detail:
uv run python haystack_milvus_hybrid_rag.py --log-dest console --log-level INFO "your question"
```

| Argument / flag | Meaning |
|---|---|
| `QUERY` *(positional, repeatable)* | Question(s) to ask. None → built-in demo set. |
| `--log-dest` / `--log-level` | `file` (default) or `console`; `INFO`/`DEBUG`/`WARN`. |

Example console output for one query:

```
Attempt 1/2 — query analysis: simple — 1 query(ies): ['aha moment in DeepSeek-R1']
Attempt 1 retrieved 4 docs — context_relevance=1.000
Retrieval loop ended after 1 attempt(s) — reason=threshold_met, context_relevance=1.000
[Q1] What is the 'aha moment' in DeepSeek-R1?
Query analysis: simple — searched 1 query(ies) in 1 attempt(s) (stop: threshold_met): [...]
Answer: ...
Metrics: faithfulness=1.000  context_relevance=1.000  docs_retrieved=4  avg_retrieval_score=0.0163
```

---

## How the query-quality branching & feedback loop work

The retrieval loop in `ask()` is bounded by independent stop conditions and
**always terminates with an answer** (from the best-scoring attempt). The
defaults live as module constants near the top of `haystack_milvus_hybrid_rag.py`:

| Constant | Default | Role |
|---|---|---|
| `RELEVANCE_THRESHOLD` | `0.7` | Success exit — stop once context relevance ≥ this. |
| `MAX_RETRIEVAL_ATTEMPTS` | `2` | Hard ceiling — original try + (N−1) refined retries. |
| `MIN_GAIN` | `0.05` | Plateau exit — stop if a retry doesn't improve by this much. |
| *(stable doc set)* | — | Stop if a retry returns the same chunks. |
| `RETRIEVER_TOP_K` | `4` | Docs retrieved per sub-query. |
| `MAX_CONTEXT_DOCS` | `8` | Cap on the merged context after fusing sub-queries. |
| `MAX_SUBQUERIES` | `4` | Cap on sub-queries a complex decomposition may request. |

Most queries pass on the first attempt and never loop; the guards exist so a hard
question can't run up unbounded cost or latency.

---

## Project layout

| File | Purpose |
|---|---|
| `haystack_milvus_hybrid_rag.py` | **Query pipeline** — analyzer, hybrid retrieval, feedback loop, generation, evaluation. |
| `ingest_pdf.py` | **Ingestion** — PDF → chunks → embeddings → Milvus (dense + BM25). |
| `Milvus_Collection_With_Fields.py` | **Step 1** — create the hybrid (dense + BM25) collection with native `pymilvus`. |
| `langfuse_tracing.py` | Shared, opt-in Langfuse tracing setup. |
| `logging_setup.py` | Shared logging — ns timestamps, rotating files under `logs/`, noise capping. |
| `docker-compose.langfuse.yml` | Local Langfuse stack. |
| `LANGFUSE_SETUP.md` | Step-by-step Langfuse deployment guide. |
| `pyproject.toml` / `uv.lock` | Dependencies and pinned lockfile. |

---

## Troubleshooting

- **`Collection 'hybrid_rag_docs' is empty or does not exist`** — run `ingest_pdf.py`
  into the `hybrid_rag_docs` collection first (Step 2).
- **BM25 / `BM25BuiltInFunction` errors** — you're likely on **Milvus Lite**.
  BM25 requires Milvus **Standalone** or **Distributed** (Step 3 of Setup).
- **`OPENAI_API_KEY is not set`** — add it to `.env` in the project root.
- **OpenAI timeouts during evaluation** — the evaluators are slow; the scripts
  raise `OPENAI_TIMEOUT` to 120s by default. Override via `.env` if needed.
- **Image text not indexed** — install the Tesseract binary (Setup step 5) and
  re-ingest; born-digital text and tables don't need it.
- **Detailed logs** — add `--log-dest console --log-level DEBUG` to any script, or
  read the rotating file under `logs/`.

---

## Notes

- The collection name `hybrid_rag_docs` is hard-coded in the query script; the
  built-in demo questions target the **DeepSeek-R1** paper, so ingest that PDF (or
  edit the questions / your own queries) to match your corpus.
- `.env`, `.venv/`, and `logs/` are git-ignored.
