"""
Hybrid RAG Pipeline — Haystack 2.29 + Milvus 2.5  (query-only)
================================================================
Runs hybrid retrieval + RAG against an existing Milvus collection.

  Query Analyzer — an LLM classifies each query as simple / vague / complex,
                   then REFINES vague queries or DECOMPOSES complex ones into
                   focused sub-queries before retrieval (query-quality-driven
                   branching).
  Feedback Loop  — the merged context is scored for relevance; if it's too low
                   the query is re-analyzed (with the failure as a hint) and
                   re-retrieved, bounded by a threshold / attempt cap / plateau
                   / stable-doc-set so it always terminates with an answer.
  RAG Pipeline   — MilvusHybridRetriever (dense + BM25 fused with RRF) is run
                   once per (sub-)query; the retrieved docs are merged + deduped
                   → PromptBuilder → OpenAIChatGenerator (one answer per query)

Each question is answered in three tiers (see answer_question):

  Tier 0  Semantic cache — a SEPARATE 'qa_cache' Milvus collection holds prior
          LLM-fallback answers, keyed by the question's dense embedding (COSINE).
          A new question is matched against it (top-1, high-similarity gate); an
          LLM then confirms the cached answer actually fits before it is reused,
          so a near-paraphrase with different intent is not served the wrong
          answer.  A hit short-circuits retrieval entirely.
  Tier 1  Grounded RAG — the hybrid-retrieval + feedback loop above.  Used when
          the corpus can answer the question (context relevance ≥ threshold).
  Tier 2  LLM fallback — when retrieval cannot find relevant-enough context, the
          question is answered from the model's own (parametric) knowledge with
          NO corpus context.  The answer is flagged source="model_knowledge"
          (it is ungrounded by construction, so faithfulness is not scored) and
          written back to 'qa_cache' so a future similar question hits Tier 0.

The 'qa_cache' collection is kept PHYSICALLY SEPARATE from the document
collection on purpose: cached LLM answers are unverified, and blending them into
document retrieval would let a hallucination be retrieved and cited as if it
were source material.  ingest_pdf.py owns 'hybrid_rag_docs' and never sees the
cache; this script owns 'qa_cache' and never writes to 'hybrid_rag_docs'.

Ingestion is handled separately by ingest_pdf.py, which populates the
collection with chunked, embedded PDF content:

    uv run python ingest_pdf.py <pdf_path> hybrid_rag_docs

Haystack 2.x differences from LangChain worth knowing:
  • Pipelines are explicit graph objects; components are connected with .connect()
  • MilvusHybridRetriever handles both dense ANN + BM25 sparse in one call
  • PromptBuilder uses Jinja2 templates; ChatMessage.from_user() wraps the result
  • No LCEL-style chaining — each component's output dict feeds the next via .run()

Requirements
------------
Milvus: Standalone or Distributed (BM25BuiltInFunction is NOT available in
        Milvus Lite).

Python: 3.12 or 3.13 recommended. pymilvus supports 3.14; haystack-ai 2.29 is
        a pure-Python wheel (py3-none-any) so it installs on 3.14, but some
        transitive C-extension deps (grpcio, numpy) may lack 3.14 wheels yet.

Installation
------------
pip install \
    "haystack-ai>=2.29.0" \
    "milvus-haystack" \
    "pymilvus>=2.5.0" \
    "openai"

Start Milvus Standalone with Docker:
    docker run -d --name milvus-standalone \
        -p 19530:19530 -p 9091:9091 \
        milvusdb/milvus:v2.5.0 milvus run standalone

Usage
-----
    uv run python haystack_milvus_hybrid_rag.py [options] ["a question" ...]

    Positional QUERY args are the questions to ask.  Pass one or more; with
    none, a built-in DeepSeek-R1 demo set runs.  Each query is first sent
    through the LLM query analyzer (refine vague / decompose complex) before
    hybrid retrieval.  Example:

        uv run python haystack_milvus_hybrid_rag.py \
            "Compare DeepSeek-R1-Zero and DeepSeek-R1, and explain the aha moment"

    --log-dest  file|console   file = size-rotated logs/hybrid_rag_query.log
                               (default; ns timestamps — see logging_setup.py)
    --log-level LEVEL          DEBUG (troubleshooting) | INFO (default) | WARN

Per-question details (retrieved-doc scores, answers, metrics) are logged at
DEBUG as they happen; a consolidated RUN SUMMARY with every answer and all
metrics is logged at INFO at the end of the run.  The RUN SUMMARY is always
echoed to the console, even when --log-dest file routes everything else to
the log file.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

# Windows terminals default to cp1252 which can't encode Haystack's emoji output.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load OPENAI_API_KEY / MILVUS_URI from the project-root .env file.
# Existing shell vars win (load_dotenv does not override already-set variables).
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── CLI + logging ────────────────────────────────────────────────────────────
# This script's only CLI options are the two logging flags.  Logging must be
# configured before enable_langfuse() and the haystack imports below so their
# import-time messages land in the log.  Shared setup (ns timestamps, rotating
# file under logs/, third-party noise capping) lives in logging_setup.py.
from logging_setup import add_logging_args, echo_to_console, setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid RAG (dense + BM25 / RRF) demo queries against an "
        "existing Milvus collection, with LLM groundedness evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "queries",
        nargs="*",
        metavar="QUERY",
        help="Question(s) to ask. Pass one or more; with none, a built-in "
        "DeepSeek-R1 demo set runs. Each is analyzed (refined/decomposed) "
        "before retrieval.",
    )
    add_logging_args(parser)
    return parser.parse_args()


ARGS = _parse_args()
setup_logging("hybrid_rag_query.log", dest=ARGS.log_dest, level=ARGS.log_level)
log = logging.getLogger(__name__)
# The end-of-run summary goes to the log file AND the console, even with
# --log-dest file (echo_to_console is a no-op in console mode).  Pinned to
# INFO so the summary — the run's primary output — still appears when
# --log-level WARN suppresses the rest (propagated records bypass the root
# logger's level; only handler levels apply).
summary_log = echo_to_console(f"{__name__}.summary")
summary_log.setLevel(logging.INFO)

# Langfuse tracing (optional) — shared setup in langfuse_tracing.py.  Activates
# when LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY are set, and MUST be configured
# before the haystack imports below.  Each pipeline.run() (one per question, plus
# the BM25 demo) becomes a "hybrid_rag_query" trace with retriever + LLM spans.
from langfuse_tracing import enable_langfuse

LANGFUSE_ENABLED = enable_langfuse("hybrid_rag_query")

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.evaluators import ContextRelevanceEvaluator, FaithfulnessEvaluator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.function import BM25BuiltInFunction
from milvus_haystack.milvus_embedding_retriever import (
    MilvusEmbeddingRetriever,
    MilvusHybridRetriever,
)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

if not os.environ.get("OPENAI_API_KEY"):
    log.error("OPENAI_API_KEY is not set — add it to the .env file in the project root.")
    sys.exit(1)

# Haystack's OpenAI components default to a 30 s request timeout (OPENAI_TIMEOUT
# env var).  The LLM evaluators routinely take 20-30 s per call (gpt-5-mini
# reasoning), so one slow call times out, exhausts retries, and kills the run.
# Raise the default; a value set in .env or the shell still wins.
os.environ.setdefault("OPENAI_TIMEOUT", "120")

MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "hybrid_rag_docs"

# Dense embedding model — MUST match what ingest_pdf.py wrote into the collection
# (text-embedding-3-large → 3072-dim) and the dim declared in
# Milvus_Collection_With_Fields.py.  Used by the document retriever AND both
# qa_cache embedders (lookup + write) so they always share one vector space.
EMBEDDING_MODEL = "text-embedding-3-large"

# For Zilliz Cloud (fully-managed Milvus):
# MILVUS_URI   = "https://<cluster>.api.gcp-us-west1.zillizcloud.com"
# MILVUS_TOKEN = os.getenv("ZILLIZ_CLOUD_API_KEY", "")

# ---------------------------------------------------------------------------
# 2. Document Store  (attach-only — never drops or writes; ingest_pdf.py owns
#    the collection schema and its content)
#    - vector_field        : stores OpenAI dense embeddings
#    - sparse_vector_field : stores BM25 TF sparse vectors (server-side)
#    - builtin_function    : tells Milvus to auto-tokenise the 'text' field
#                            and store sparse output in 'sparse_vector'
#    - analyzer_params     : optional — customise tokenisation per your domain
# ---------------------------------------------------------------------------

document_store = MilvusDocumentStore(
    connection_args={"uri": MILVUS_URI},
    collection_name=COLLECTION_NAME,
    vector_field="vector",               # dense embedding field
    sparse_vector_field="sparse_vector", # BM25 sparse output field
    text_field="text",                   # raw text field Milvus tokenises
    builtin_function=[
        BM25BuiltInFunction(
            input_field_names="text",
            output_field_names="sparse_vector",
            analyzer_params={            # optional — customise tokeniser
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
    drop_old=False,  # NEVER drop — the collection is populated by ingest_pdf.py
)

doc_count = document_store.count_documents()
if doc_count == 0:
    log.error(
        "Collection '%s' is empty or does not exist.\n"
        "Ingest a PDF first:\n"
        "    uv run python ingest_pdf.py <pdf_path> %s",
        COLLECTION_NAME, COLLECTION_NAME,
    )
    sys.exit(1)
log.info("Collection '%s' ready — %d chunks available", COLLECTION_NAME, doc_count)
log.info("Langfuse tracing: %s", "enabled" if LANGFUSE_ENABLED else "disabled")

# ---------------------------------------------------------------------------
# 2b. Q&A semantic-cache store  (a SEPARATE collection — see module docstring)
#     Holds prior Tier-2 LLM-fallback answers so a future similar question can
#     be served without going to the LLM again.  Deliberately kept apart from
#     the document collection so unverified cached answers can never leak into
#     document retrieval and be cited as source material.
#
#     - dense-only: no BM25BuiltInFunction — cache matching is by question
#       embedding similarity (paraphrase-robust), not keyword overlap.
#     - COSINE index: makes the retriever's score an interpretable similarity
#       in [0, 1] (milvus-haystack maps it as (cosine + 1) / 2, so 1.0 = the
#       same question, 0.5 = orthogonal) — that's what CACHE_SIMILARITY_THRESHOLD
#       is compared against below.
#     - Strong consistency: read-your-writes within a single run, so a fallback
#       answer written for one question is visible to a later similar one in the
#       SAME run.  The cache is tiny, so the cost is negligible.
#     A doc stores the QUESTION as its content (that's what we embed + match on);
#     the answer rides along in meta["answer"].  The collection is created lazily
#     on the first write — until then lookups find nothing (handled gracefully).
# ---------------------------------------------------------------------------

QA_CACHE_COLLECTION = "qa_cache"

qa_cache_store = MilvusDocumentStore(
    connection_args={"uri": MILVUS_URI},
    collection_name=QA_CACHE_COLLECTION,
    vector_field="vector",
    text_field="text",
    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
    consistency_level="Strong",
    drop_old=False,  # cache persists across runs — never drop it
)

try:
    cache_count = qa_cache_store.count_documents()
except Exception:
    # Collection doesn't exist yet (no fallback has ever been cached).  It will
    # be created on the first write_documents() in store_in_cache().
    cache_count = 0
log.info("Q&A cache '%s' ready — %d cached answer(s)", QA_CACHE_COLLECTION, cache_count)

# ---------------------------------------------------------------------------
# 3. RAG Prompt Template
#    Jinja2 syntax — Haystack's PromptBuilder renders this at query time.
#    'documents' is a list[Document] from the retriever.
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say so clearly — do not
invent facts.

Context:
{% for doc in documents %}
[{{ loop.index }}] (document: {{ doc.meta.get('document_name', 'unknown') }}, \
pages: {{ doc.meta.get('page_numbers', []) }}, \
type: {{ doc.meta.get('chunk_type', 'text') }})
{{ doc.content }}
{% endfor %}

Question: {{ question }}
Answer:
""".strip()

# ---------------------------------------------------------------------------
# 3b. Query Analyzer Prompt Template
#     The analyzer LLM inspects the raw user question and decides how to search:
#       • simple  — clear & specific; search as-is (lightly normalised)
#       • vague   — underspecified/ambiguous; REFINE into one sharper query
#       • complex — several distinct sub-questions; DECOMPOSE into 2-4 queries
#     It returns strict JSON so we can branch deterministically in Python.
#     response_format=json_object (set on the generator below) guarantees the
#     reply parses; this template still spells out the shape for the model.
# ---------------------------------------------------------------------------

QUERY_ANALYZER_TEMPLATE = """
You are a query-analysis assistant for a document retrieval system. Inspect the
user's question and prepare the best search query (or queries) for a hybrid
(dense + BM25) retriever over a technical document collection.

Classify the question into exactly one of:
  - "simple":  clear and specific; answerable from a single retrieval.
               Return the question as one query, lightly normalised.
  - "vague":   underspecified, ambiguous, or poorly worded.
               REFINE it into ONE clearer, more specific search query
               (add likely-intended terms; resolve obvious ambiguity).
  - "complex": asks several distinct things, or requires comparing/combining
               multiple facts. DECOMPOSE it into 2-4 focused, self-contained
               sub-queries that together cover the question.

Rules:
  - "simple" and "vague"  -> exactly ONE query in "queries".
  - "complex"             -> 2 to 4 queries in "queries".
  - Each query must be a standalone search string (no pronouns like "it").
  - Do NOT answer the question. Only produce search queries.

Respond with ONLY a JSON object, no prose, in this exact shape:
{
  "classification": "simple" | "vague" | "complex",
  "reasoning": "<one short sentence>",
  "queries": ["<query 1>", "<query 2>", ...]
}
{% if feedback %}
A PREVIOUS retrieval attempt for this question did not find relevant enough
documents. Feedback: {{ feedback }}
Produce DIFFERENT queries this time — do not repeat the earlier ones. Broaden
or rephrase, try alternative terminology, or split the question differently so
the retriever surfaces better-matching passages.
{% endif %}
Question: {{ question }}
""".strip()

# ---------------------------------------------------------------------------
# 3c. Cache-gate Prompt Template  (Tier 0)
#     A high vector-similarity score only says two questions are CLOSE, not that
#     they mean the same thing ("advantages of X" vs "disadvantages of X" embed
#     very near each other).  Before reusing a cached answer we ask an LLM to
#     confirm the cached answer genuinely and fully answers the NEW question.
#     Returns strict JSON so the decision is deterministic in Python.
# ---------------------------------------------------------------------------

CACHE_GATE_TEMPLATE = """
You are validating a cache hit for a question-answering system. We retrieved a
previously-answered question whose wording is similar to a new question. Decide
whether the CACHED ANSWER below fully and correctly answers the NEW QUESTION.

Say "applies": true ONLY if the cached answer addresses exactly what the new
question asks. If the new question asks about a different aspect, entity, scope,
polarity (e.g. advantages vs disadvantages), or level of detail — even if the
wording is close — say "applies": false.

Cached question: {{ cached_question }}
Cached answer: {{ cached_answer }}

New question: {{ question }}

Respond with ONLY a JSON object, no prose, in this exact shape:
{
  "applies": true | false,
  "reason": "<one short sentence>"
}
""".strip()

# ---------------------------------------------------------------------------
# 3d. Fallback Prompt Template  (Tier 2 — answer from the model's own knowledge)
#     Used only when hybrid retrieval cannot surface relevant-enough context.
#     There is intentionally NO corpus context here: the answer comes from the
#     model's parametric knowledge, so it is ungrounded by construction (we mark
#     it source="model_knowledge" and skip faithfulness scoring).  We still ask
#     the model to flag uncertainty rather than invent specifics.
# ---------------------------------------------------------------------------

FALLBACK_PROMPT_TEMPLATE = """
You are a helpful assistant. The document collection did not contain enough
information to answer the question below, so answer it from your own general
knowledge as accurately as you can. If you are unsure or the question needs
information you don't reliably have, say so plainly rather than inventing
specifics.

Question: {{ question }}
Answer:
""".strip()

# ---------------------------------------------------------------------------
# 4. Pipelines
#
#    The flow now branches on query quality, so it no longer fits a single
#    static graph (decomposition fans one question out to N retrievals that
#    must be merged before a single answer is generated).  It is split into
#    three small pipelines, each wrapped so Langfuse traces it:
#
#    raw question (str)
#         │
#    [query_analyzer_pipeline]  PromptBuilder → OpenAIChatGenerator (JSON)
#         │ classification + 1..N search queries
#         │   for each (sub-)query ↓
#    [retrieval_pipeline]       OpenAITextEmbedder → MilvusHybridRetriever
#         │ documents per query  →  merged + deduped in Python
#         │
#    [generation_pipeline]      PromptBuilder → OpenAIChatGenerator
#         │ final answer (grounded on the ORIGINAL question + merged context)
# ---------------------------------------------------------------------------

# How many docs the merged context is capped at after fusing per-(sub-)query
# results.  A single query retrieves top_k=4; a decomposed query can surface
# more distinct evidence, so we allow a larger merged set before generation.
RETRIEVER_TOP_K = 4
MAX_CONTEXT_DOCS = 8
# Hard cap on sub-queries the analyzer may request, as a cost/latency guard.
MAX_SUBQUERIES = 4

# ── Tier 0 semantic cache ────────────────────────────────────────────────────
# Minimum mapped-COSINE similarity (see qa_cache_store: (cosine + 1) / 2, so this
# is in [0, 1] with 1.0 = identical question) for a cached answer to even be
# CONSIDERED.  Deliberately strict — it's only a cheap pre-filter; the LLM cache
# gate (CACHE_GATE_TEMPLATE) makes the final reuse decision, so this just keeps
# us from spending a gate call on questions that aren't plausibly the same.
CACHE_SIMILARITY_THRESHOLD = 0.90

# ── Retrieval feedback loop ──────────────────────────────────────────────────
# After retrieving, ask() scores the merged context with the ContextRelevance
# evaluator and, if it's too low, re-analyzes the query (with the failure as a
# hint) and retrieves again.  The loop is bounded by several independent stop
# conditions so it always terminates with an answer (see ask() for the logic):
#   • RELEVANCE_THRESHOLD   — success exit: stop once relevance ≥ this.
#   • MAX_RETRIEVAL_ATTEMPTS— hard ceiling: original try + (N-1) retries.
#   • MIN_GAIN              — plateau exit: stop if a retry doesn't beat the
#                             previous score by at least this much.
#   • (stable doc set)      — stop if a retry returns the same chunks.
# Most queries should pass on the first attempt and never loop; the guards
# exist so a hard question can't run up unbounded cost/latency.
RELEVANCE_THRESHOLD = 0.7
MAX_RETRIEVAL_ATTEMPTS = 2
MIN_GAIN = 0.05

# ── 4a. Query analyzer (refine vague / decompose complex) ────────────────────
# response_format=json_object forces the generator to emit parseable JSON, so
# analyze_query() can branch deterministically instead of regex-scraping prose.
query_analyzer_pipeline = Pipeline()
query_analyzer_pipeline.add_component(
    "prompt_builder",
    PromptBuilder(template=QUERY_ANALYZER_TEMPLATE, required_variables=["question"]),
)
query_analyzer_pipeline.add_component(
    "llm",
    OpenAIChatGenerator(
        model="gpt-4o-mini",
        generation_kwargs={"response_format": {"type": "json_object"}},
    ),
)
query_analyzer_pipeline.connect("prompt_builder.prompt", "llm.messages")

# ── 4b. Retrieval (dense ANN + BM25 sparse, fused with RRF) — run per query ──
retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component(
    "text_embedder",
    OpenAITextEmbedder(model=EMBEDDING_MODEL),
)
retrieval_pipeline.add_component(
    "retriever",
    MilvusHybridRetriever(
        document_store=document_store,
        top_k=RETRIEVER_TOP_K,
        # reranker=WeightedRanker(0.4, 0.6),  # uncomment to favour BM25
        # default reranker is RRFRanker() — no import needed
    ),
)
retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

# ── 4c. Generation (one answer from the merged context) ──────────────────────
generation_pipeline = Pipeline()
generation_pipeline.add_component(
    "prompt_builder",
    PromptBuilder(template=RAG_PROMPT_TEMPLATE, required_variables=["question"]),
)
generation_pipeline.add_component(
    "llm",
    OpenAIChatGenerator(model="gpt-4o-mini"),
)
generation_pipeline.connect("prompt_builder.prompt", "llm.messages")

# ── 4e. Cache lookup (Tier 0) — dense top-1 over the qa_cache collection ──────
# Same embedding model as the document retriever so questions land in a
# comparable space.  top_k=1: we only ever consider the single closest cached
# question; the gate decides whether to actually reuse it.
cache_lookup_pipeline = Pipeline()
cache_lookup_pipeline.add_component(
    "text_embedder",
    OpenAITextEmbedder(model=EMBEDDING_MODEL),
)
cache_lookup_pipeline.add_component(
    "retriever",
    MilvusEmbeddingRetriever(document_store=qa_cache_store, top_k=1),
)
cache_lookup_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

# ── 4f. Cache gate (Tier 0) — LLM confirms a cached answer fits the new query ─
# response_format=json_object so check_cache_gate() can branch deterministically.
cache_gate_pipeline = Pipeline()
cache_gate_pipeline.add_component(
    "prompt_builder",
    PromptBuilder(
        template=CACHE_GATE_TEMPLATE,
        required_variables=["question", "cached_question", "cached_answer"],
    ),
)
cache_gate_pipeline.add_component(
    "llm",
    OpenAIChatGenerator(
        model="gpt-4o-mini",
        generation_kwargs={"response_format": {"type": "json_object"}},
    ),
)
cache_gate_pipeline.connect("prompt_builder.prompt", "llm.messages")

# ── 4g. Fallback generation (Tier 2) — answer from the model's own knowledge ──
fallback_pipeline = Pipeline()
fallback_pipeline.add_component(
    "prompt_builder",
    PromptBuilder(template=FALLBACK_PROMPT_TEMPLATE, required_variables=["question"]),
)
fallback_pipeline.add_component(
    "llm",
    OpenAIChatGenerator(model="gpt-4o-mini"),
)
fallback_pipeline.connect("prompt_builder.prompt", "llm.messages")

# ── 4h. Cache write (Tier 2 → Tier 0) — embed the question, store the answer ──
# Mirrors ingest_pdf.py's indexing path: OpenAIDocumentEmbedder adds the dense
# vector (over the QUESTION, which is the doc content), DocumentWriter upserts
# into qa_cache.  This is what creates the qa_cache collection on first use.
cache_write_pipeline = Pipeline()
cache_write_pipeline.add_component(
    "embedder",
    OpenAIDocumentEmbedder(model=EMBEDDING_MODEL, progress_bar=False),
)
cache_write_pipeline.add_component(
    "writer",
    DocumentWriter(document_store=qa_cache_store, policy=DuplicatePolicy.NONE),
)
cache_write_pipeline.connect("embedder.documents", "writer.documents")

log.info(
    "Pipelines wired (cache lookup → query analyzer → hybrid retrieval → "
    "generation; LLM fallback + cache write)"
)
log.debug("Analyzer graph:\n%s", query_analyzer_pipeline)
log.debug("Retrieval graph:\n%s", retrieval_pipeline)
log.debug("Generation graph:\n%s", generation_pipeline)
log.debug("Cache lookup graph:\n%s", cache_lookup_pipeline)

# ── 4d. Quality evaluators (also drive the retrieval feedback loop) ──────────
#   FaithfulnessEvaluator     — LLM checks whether every claim in the answer is
#                               supported by the retrieved context.
#                               Score = faithful statements / total statements
#   ContextRelevanceEvaluator — LLM checks whether each retrieved document is
#                               relevant to the question.
#                               Score = relevant docs / total docs retrieved
# Both use OpenAI (OPENAI_API_KEY) and are instantiated once at module level.
# ContextRelevance is dual-purpose: ask() uses it INSIDE the retrieval loop as
# the control signal that decides whether to retry, and evaluate_groundedness
# reuses that same score for the final report (so it's computed once per query).
#
# Each evaluator is wrapped in a single-component Pipeline: Haystack only traces
# pipeline executions, so bare component.run() calls would be invisible to
# Langfuse.  Wrapped, every evaluation becomes its own trace with the internal
# gpt-4o-mini token usage and cost attached.
#
# progress_bar=False: the evaluators' tqdm bars write straight to stderr,
# bypassing logging — without them the console stays clean in file mode (just
# the breadcrumb + RUN SUMMARY).  Per-question progress is logged at INFO.
faithfulness_pipeline = Pipeline()
faithfulness_pipeline.add_component("faithfulness", FaithfulnessEvaluator(progress_bar=False))

context_relevance_pipeline = Pipeline()
context_relevance_pipeline.add_component(
    "context_relevance", ContextRelevanceEvaluator(progress_bar=False)
)

# ---------------------------------------------------------------------------
# 5. Query helper
# ---------------------------------------------------------------------------

def analyze_query(question: str, feedback: str | None = None) -> dict:
    """
    Classify the question and produce the search query/queries to run.

    *feedback*, when given, tells the analyzer a prior retrieval attempt found
    irrelevant docs so it produces DIFFERENT queries (used by ask()'s loop).

    Returns a dict:
      classification — "simple" | "vague" | "complex"
      queries        — list[str] of 1..MAX_SUBQUERIES search strings
      reasoning      — short rationale from the analyzer LLM (may be "")

    Always returns at least one query.  Any failure (LLM error, malformed
    JSON, empty result) degrades gracefully to treating the original question
    as a single "simple" query, so analysis never blocks retrieval.
    """
    try:
        result = query_analyzer_pipeline.run(
            {"prompt_builder": {"question": question, "feedback": feedback}}
        )
        raw = result["llm"]["replies"][0].text
        data = json.loads(raw)
        classification = data.get("classification", "simple")
        reasoning = data.get("reasoning", "")
        queries = [
            q.strip()
            for q in data.get("queries", [])
            if isinstance(q, str) and q.strip()
        ]
    except Exception:
        log.exception("Query analysis failed — falling back to the raw question")
        return {"classification": "simple", "queries": [question], "reasoning": "analyzer error"}

    if not queries:
        # Model returned no usable queries — fall back rather than search nothing.
        log.warning("Analyzer returned no queries — using the raw question")
        return {"classification": "simple", "queries": [question], "reasoning": reasoning}

    # Guard against an over-eager decomposition blowing up cost/latency.
    if len(queries) > MAX_SUBQUERIES:
        log.warning("Analyzer returned %d queries — capping at %d", len(queries), MAX_SUBQUERIES)
        queries = queries[:MAX_SUBQUERIES]

    return {"classification": classification, "queries": queries, "reasoning": reasoning}


def retrieve_for_queries(queries: list[str]) -> list[Document]:
    """
    Run hybrid retrieval once per (sub-)query and merge the results.

    retriever is the leaf component, so its documents come back without
    include_outputs_from.  Returns the deduped, score-ranked context capped at
    MAX_CONTEXT_DOCS (see merge_documents).
    """
    doc_lists: list[list[Document]] = []
    for sub_query in queries:
        retrieval = retrieval_pipeline.run(
            {
                "text_embedder": {"text": sub_query},
                "retriever":     {"query_text": sub_query},
            }
        )
        sub_docs = retrieval["retriever"]["documents"]
        log.debug("  sub-query %r → %d docs", sub_query, len(sub_docs))
        doc_lists.append(sub_docs)
    return merge_documents(doc_lists, MAX_CONTEXT_DOCS)


def score_context_relevance(question: str, docs: list[Document]) -> float:
    """
    Context-relevance score (0–1) for a retrieved set — the loop's control
    signal.  Returns 0.0 for an empty set (nothing relevant was retrieved).
    """
    if not docs:
        return 0.0
    contexts = [[doc.content for doc in docs if doc.content]]
    cr_result = context_relevance_pipeline.run(
        {"context_relevance": {"questions": [question], "contexts": contexts}}
    )["context_relevance"]
    # The evaluator can return None (e.g. no extractable statements); treat as 0.
    return cr_result.get("score") or 0.0


def merge_documents(doc_lists: list[list[Document]], max_docs: int) -> list[Document]:
    """
    Fuse per-(sub-)query retrieval results into one deduped context.

    Documents are deduped by Document.id; when the same chunk is retrieved by
    more than one sub-query we keep the instance with the highest hybrid (RRF)
    score.  The survivors are sorted by score (desc) and truncated to max_docs.

    Note: RRF scores are not strictly comparable across different queries, so
    this ranking is a pragmatic heuristic — good enough to surface the
    strongest evidence first; Context Relevance (section 5b) measures the
    actual quality of the merged set.
    """
    by_id: dict[str, Document] = {}
    for docs in doc_lists:
        for doc in docs:
            existing = by_id.get(doc.id)
            if existing is None or (doc.score or 0.0) > (existing.score or 0.0):
                by_id[doc.id] = doc
    merged = sorted(by_id.values(), key=lambda d: d.score or 0.0, reverse=True)
    return merged[:max_docs]


def retrieve_best_context(question: str) -> dict:
    """
    Run the bounded retrieval feedback loop and return the best attempt's
    context — WITHOUT generating an answer.  Split out from ask() so the
    three-tier orchestrator (answer_question) can inspect the relevance score
    and decide between a grounded answer and an LLM fallback before paying for
    a generation call.

    Each attempt:
      1. analyze_query()         — refine a vague query / decompose a complex
                                   one (on retries, with the failure as a hint)
      2. retrieve_for_queries()  — hybrid retrieval per (sub-)query, merged
      3. score_context_relevance — grade the merged context (the control signal)

    The loop keeps the best-scoring attempt and stops as soon as ANY holds:
      • relevance ≥ RELEVANCE_THRESHOLD   (success)
      • MAX_RETRIEVAL_ATTEMPTS reached    (hard ceiling)
      • a retry returns the same chunks   (stable doc set — retrying won't help)
      • a retry gains ≤ MIN_GAIN          (plateau — converged)

    Returns the winning attempt dict: classification, queries, reasoning, docs,
    score (context_relevance), attempts, stop_reason.  It always returns a set
    (possibly weak) — the caller decides whether it's strong enough to ground on.
    """
    best: dict | None = None        # winning attempt: {classification, queries, reasoning, docs, score}
    feedback: str | None = None     # hint fed to the next analyze_query()
    prev_score: float | None = None
    prev_doc_ids: set[str] | None = None
    stop_reason = "max_attempts"    # overwritten if an earlier exit fires
    attempt = 0

    for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        analysis = analyze_query(question, feedback=feedback)
        log.info(
            "Attempt %d/%d — query analysis: %s — %d query(ies): %s",
            attempt, MAX_RETRIEVAL_ATTEMPTS, analysis["classification"],
            len(analysis["queries"]), analysis["queries"],
        )
        if analysis["reasoning"]:
            log.debug("Analyzer reasoning: %s", analysis["reasoning"])

        docs = retrieve_for_queries(analysis["queries"])
        score = score_context_relevance(question, docs)
        log.info(
            "Attempt %d retrieved %d docs (merged from %d query(ies)) — context_relevance=%.3f",
            attempt, len(docs), len(analysis["queries"]), score,
        )

        # Keep the strongest attempt so we can always answer from the best set.
        if best is None or score > best["score"]:
            best = {
                "classification": analysis["classification"],
                "queries":        analysis["queries"],
                "reasoning":      analysis["reasoning"],
                "docs":           docs,
                "score":          score,
            }

        # (1) Success — relevant enough, stop.
        if score >= RELEVANCE_THRESHOLD:
            stop_reason = "threshold_met"
            break

        doc_ids = {d.id for d in docs}

        # (4) Stable doc set — a retry surfaced the same chunks; retrying again
        #     won't help (checked before plateau as it's the more specific reason).
        if prev_doc_ids is not None and doc_ids == prev_doc_ids:
            stop_reason = "stable_docs"
            break

        # (3) Plateau — the retry didn't improve enough; we've converged.
        if prev_score is not None and score <= prev_score + MIN_GAIN:
            stop_reason = "plateau"
            break

        # Below threshold and attempts remain → set up the next retry.  The
        # feedback must change the analyzer's output, else attempt N+1 repeats
        # attempt N and the stable-doc check fires immediately, wasting a call.
        prev_score = score
        prev_doc_ids = doc_ids
        feedback = (
            f"The previous queries {analysis['queries']} scored only {score:.2f} "
            f"on context relevance (target ≥ {RELEVANCE_THRESHOLD:.2f}); the "
            f"retrieved passages were largely off-topic."
        )
        log.info("Relevance %.3f < %.2f — refining the query and retrying", score, RELEVANCE_THRESHOLD)

    # best is never None: the loop runs at least once (MAX_RETRIEVAL_ATTEMPTS ≥ 1).
    log.info(
        "Retrieval loop ended after %d attempt(s) — reason=%s, context_relevance=%.3f",
        attempt, stop_reason, best["score"],
    )
    for i, doc in enumerate(best["docs"], start=1):
        src   = doc.meta.get("document_name", "?")
        pages = doc.meta.get("page_numbers", [])
        doc_score = f"{doc.score:.4f}" if doc.score is not None else "n/a"
        preview = (doc.content or "")[:80].replace("\n", " ")
        log.debug("  [%d] score=%s (%s p.%s) %s...", i, doc_score, src, pages, preview)

    best["attempts"] = attempt
    best["stop_reason"] = stop_reason
    return best


def generate_grounded_answer(question: str, docs: list[Document]) -> str:
    """Generate one answer to *question* grounded ONLY on *docs* (Tier 1)."""
    generation = generation_pipeline.run(
        {"prompt_builder": {"question": question, "documents": docs}}
    )
    llm_replies: list[ChatMessage] = generation["llm"]["replies"]
    answer = llm_replies[0].text if llm_replies else "(no reply)"
    log.info("Grounded answer generated (%d chars)", len(answer))
    log.debug("Answer:\n%s", answer)
    return answer


def ask(question: str) -> tuple[str, list[Document], dict]:
    """
    Grounded RAG answer (Tier 1 only): retrieve the best context, then generate
    one answer from it.  Kept as a thin convenience wrapper around
    retrieve_best_context() + generate_grounded_answer(); the full three-tier
    flow (cache → RAG → LLM fallback) lives in answer_question().

    Returns (answer, best_docs, analysis); analysis carries the winning
    classification/queries plus loop telemetry and the final context_relevance.
    """
    log.info("Question: %s", question)
    best = retrieve_best_context(question)
    answer = generate_grounded_answer(question, best["docs"])
    analysis_result = {
        "classification":    best["classification"],
        "queries":           best["queries"],
        "reasoning":         best["reasoning"],
        "attempts":          best["attempts"],
        "stop_reason":       best["stop_reason"],
        "context_relevance": best["score"],
        "source":            "rag",
    }
    return answer, best["docs"], analysis_result


# ---------------------------------------------------------------------------
# 5a. Three-tier answering: semantic cache → grounded RAG → LLM fallback
# ---------------------------------------------------------------------------

def check_cache_gate(question: str, cached_question: str, cached_answer: str) -> bool:
    """
    LLM gate for a cache near-hit: does *cached_answer* genuinely answer
    *question*?  A high embedding similarity only means the questions are close
    in wording, not intent — this guards against serving the wrong cached answer
    for a near-paraphrase (e.g. advantages vs disadvantages).  Any failure is
    treated as "does not apply" so we fall through to a real answer.
    """
    try:
        result = cache_gate_pipeline.run(
            {
                "prompt_builder": {
                    "question": question,
                    "cached_question": cached_question,
                    "cached_answer": cached_answer,
                }
            }
        )
        data = json.loads(result["llm"]["replies"][0].text)
        return bool(data.get("applies", False))
    except Exception:
        log.exception("Cache gate failed — treating the candidate as a miss")
        return False


def lookup_cache(question: str) -> dict | None:
    """
    Tier 0 — try to answer *question* from the qa_cache collection.

    Embeds the question, finds the single nearest cached question, and reuses
    its answer ONLY if (a) similarity ≥ CACHE_SIMILARITY_THRESHOLD and (b) the
    LLM gate confirms the cached answer fits.  Returns
    {answer, cached_question, similarity} on a confirmed hit, else None.

    Never raises: if the cache collection doesn't exist yet (no fallback has
    ever been written) or anything errors, it degrades to a miss so the grounded
    path runs normally.
    """
    try:
        result = cache_lookup_pipeline.run({"text_embedder": {"text": question}})
        docs = result["retriever"]["documents"]
    except Exception:
        log.debug("Cache lookup skipped (collection not ready or error)", exc_info=True)
        return None

    if not docs:
        return None

    top = docs[0]
    similarity = top.score or 0.0
    cached_question = top.meta.get("question") or top.content or ""
    cached_answer = top.meta.get("answer")

    if not cached_answer or similarity < CACHE_SIMILARITY_THRESHOLD:
        log.debug(
            "Cache miss — best similarity %.3f < %.2f", similarity, CACHE_SIMILARITY_THRESHOLD
        )
        return None

    if not check_cache_gate(question, cached_question, cached_answer):
        log.info("Cache near-hit (sim=%.3f) rejected by gate: %r", similarity, cached_question)
        return None

    log.info("Cache HIT (sim=%.3f) — reusing the answer cached for %r", similarity, cached_question)
    return {"answer": cached_answer, "cached_question": cached_question, "similarity": similarity}


def generate_fallback_answer(question: str) -> str:
    """Tier 2 — answer *question* from the model's own knowledge (no context)."""
    generation = fallback_pipeline.run({"prompt_builder": {"question": question}})
    replies: list[ChatMessage] = generation["llm"]["replies"]
    answer = replies[0].text if replies else "(no reply)"
    log.info("Fallback (model-knowledge) answer generated (%d chars)", len(answer))
    log.debug("Fallback answer:\n%s", answer)
    return answer


def store_in_cache(question: str, answer: str) -> None:
    """
    Persist a Tier-2 fallback (question, answer) into qa_cache so a future
    similar question hits Tier 0.  This is what lazily creates the qa_cache
    collection on first use.  Never raises — a cache-write failure must not lose
    the answer we just produced.
    """
    try:
        doc = Document(
            content=question,  # embedded + matched on; the answer rides in meta
            meta={
                "question": question,
                "answer": answer,
                "source": "llm_cache",
                "cached_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        cache_write_pipeline.run({"embedder": {"documents": [doc]}})
        log.info("Cached the fallback answer for future similar questions")
    except Exception:
        log.exception("Failed to write the fallback answer to qa_cache — continuing")


def answer_question(question: str) -> tuple[str, list[Document], dict]:
    """
    Answer a question through the three tiers (see module docstring):

      Tier 0  cache          — reuse a confirmed prior LLM-fallback answer
      Tier 1  rag            — grounded answer when the corpus is relevant enough
      Tier 2  model_knowledge— LLM fallback (no context) when it isn't, cached

    Returns (answer, docs, analysis).  analysis["source"] records which tier
    produced the answer so downstream evaluation/reporting can treat grounded
    and ungrounded answers correctly (faithfulness is only meaningful for "rag").
    For a cache hit, docs is empty; for a fallback, docs is the best (weak)
    retrieved set, kept for transparency but not used to ground the answer.
    """
    log.info("Question: %s", question)

    # ── Tier 0 — semantic cache ──────────────────────────────────────────────
    cached = lookup_cache(question)
    if cached is not None:
        analysis = {
            "source":            "cache",
            "classification":    "cached",
            "queries":           [],
            "reasoning":         f"reused answer cached for {cached['cached_question']!r}",
            "attempts":          0,
            "stop_reason":       "cache_hit",
            "context_relevance": None,
            "cache_similarity":  cached["similarity"],
            "cached_question":   cached["cached_question"],
        }
        return cached["answer"], [], analysis

    # ── Tier 1 — grounded retrieval ──────────────────────────────────────────
    best = retrieve_best_context(question)
    analysis = {
        "classification":    best["classification"],
        "queries":           best["queries"],
        "reasoning":         best["reasoning"],
        "attempts":          best["attempts"],
        "stop_reason":       best["stop_reason"],
        "context_relevance": best["score"],
    }

    if best["score"] >= RELEVANCE_THRESHOLD:
        analysis["source"] = "rag"
        answer = generate_grounded_answer(question, best["docs"])
        return answer, best["docs"], analysis

    # ── Tier 2 — corpus can't answer it → LLM fallback, then cache it ────────
    log.info(
        "Context relevance %.3f < %.2f — the corpus can't answer this; "
        "falling back to the LLM (no context) and caching the result",
        best["score"], RELEVANCE_THRESHOLD,
    )
    analysis["source"] = "model_knowledge"
    answer = generate_fallback_answer(question)
    store_in_cache(question, answer)
    return answer, best["docs"], analysis


# ---------------------------------------------------------------------------
# 5b. Groundedness evaluation
#     Combines the faithfulness check with the context-relevance score from the
#     retrieval loop into the per-question metrics dict.  The evaluator pipelines
#     themselves are defined in section 4d (context relevance is shared with the
#     loop); see there for how/why they're wrapped for Langfuse tracing.
# ---------------------------------------------------------------------------

def evaluate_groundedness(
    question: str,
    answer: str,
    retrieved_docs: list[Document],
    context_relevance: float | None = None,
) -> dict:
    """
    Run faithfulness evaluation for a single Q&A pair and assemble the metrics.

    *context_relevance*, when provided, is the score ask()'s retrieval loop
    already computed for this doc set — reused here so we don't pay for a second
    identical ContextRelevance call.  When None (e.g. called standalone), it's
    computed on the spot.

    Returns a dict with:
      faithfulness        — 0–1; how well the answer is supported by context
      context_relevance   — 0–1; how relevant the retrieved docs are
      num_docs_retrieved  — integer count of retrieved documents
      avg_retrieval_score — mean hybrid (RRF) score across retrieved docs
    """
    contexts = [[doc.content for doc in retrieved_docs if doc.content]]

    scored_docs = [d.score for d in retrieved_docs if d.score is not None]
    avg_retrieval_score = sum(scored_docs) / len(scored_docs) if scored_docs else 0.0

    # LLM-based faithfulness: is the answer supported by the retrieved context?
    f_result = faithfulness_pipeline.run(
        {
            "faithfulness": {
                "questions": [question],
                "contexts": contexts,
                "predicted_answers": [answer],
            }
        }
    )["faithfulness"]

    # Reuse the loop's context-relevance score; only recompute if not supplied.
    if context_relevance is None:
        context_relevance = score_context_relevance(question, retrieved_docs)

    metrics = {
        "faithfulness":        f_result["score"],
        "context_relevance":   context_relevance,
        "num_docs_retrieved":  len(retrieved_docs),
        "avg_retrieval_score": avg_retrieval_score,
    }

    # Per-question metrics at DEBUG — the end-of-run summary consolidates them.
    log.info("Evaluation done (faithfulness + context relevance)")
    log.debug(
        "[metrics] faithfulness=%.3f  context_relevance=%.3f  "
        "docs_retrieved=%d  avg_retrieval_score=%.4f",
        metrics["faithfulness"], metrics["context_relevance"],
        metrics["num_docs_retrieved"], metrics["avg_retrieval_score"],
    )

    return metrics


# ---------------------------------------------------------------------------
# 6. End-of-run summary
#    All answers and all metrics consolidated in one block at the END of the
#    run (per-question details are only at DEBUG while the run progresses).
# ---------------------------------------------------------------------------

def log_run_summary(results: list[dict]) -> None:
    """
    Emit the consolidated RUN SUMMARY at INFO.

    Uses summary_log, which echoes to the console even when --log-dest file
    routes the rest of the run to the log file.

    *results* is a list of {"question": str, "answer": str, "metrics": dict}
    in run order (metrics dict as returned by evaluate_groundedness(), or
    None when evaluation failed for that question).
    """
    n = len(results)
    sep = "=" * 60

    if n == 0:
        summary_log.warning("No queries completed — nothing to summarize.")
        return

    summary_log.info("%s\nRUN SUMMARY  (%d queries)\n%s", sep, n, sep)

    # Answers + per-question metrics
    for i, r in enumerate(results, start=1):
        m = r["metrics"]
        analysis = r.get("analysis") or {}
        source = analysis.get("source", "rag")

        # The per-question header describes how the answer was produced — which
        # of the three tiers fired (see answer_question).
        if source == "cache":
            query_line = (
                f"Source: CACHE — reused a prior LLM-fallback answer "
                f"(similarity={analysis.get('cache_similarity', 0.0):.3f}) cached for: "
                f"{analysis.get('cached_question', '?')!r}"
            )
        elif source == "model_knowledge":
            cr = analysis.get("context_relevance")
            cr_txt = f"{cr:.3f}" if cr is not None else "n/a"
            query_line = (
                f"Source: MODEL KNOWLEDGE — LLM fallback (no corpus context; "
                f"best context_relevance={cr_txt} < {RELEVANCE_THRESHOLD:.2f}); "
                f"answer cached for future similar questions"
            )
        else:  # "rag"
            cls = analysis.get("classification", "n/a")
            sub_queries = analysis.get("queries", [])
            attempts = analysis.get("attempts", 1)
            stop_reason = analysis.get("stop_reason", "n/a")
            query_line = (
                f"Source: RAG — query analysis: {cls} — searched {len(sub_queries)} "
                f"query(ies) in {attempts} attempt(s) (stop: {stop_reason}): {sub_queries}"
            )

        if m is None:
            # Not all None metrics are failures: cache hits and model-knowledge
            # fallbacks are ungrounded by design, so groundedness is N/A.
            metrics_note = (
                "(not applicable — answer is not grounded in the corpus)"
                if source != "rag"
                else "(evaluation failed — see errors earlier in the run)"
            )
            summary_log.info(
                "[Q%d] %s\n%s\nAnswer:\n%s\nMetrics: %s",
                i, r["question"], query_line, r["answer"], metrics_note,
            )
            continue
        summary_log.info(
            "[Q%d] %s\n"
            "%s\n"
            "Answer:\n%s\n"
            "Metrics: faithfulness=%.3f  context_relevance=%.3f  "
            "docs_retrieved=%d  avg_retrieval_score=%.4f",
            i, r["question"], query_line, r["answer"],
            m["faithfulness"], m["context_relevance"],
            m["num_docs_retrieved"], m["avg_retrieval_score"],
        )

    # Aggregated metrics table — over successfully evaluated queries only
    all_metrics = [r["metrics"] for r in results if r["metrics"] is not None]
    k = len(all_metrics)
    if k == 0:
        summary_log.warning(
            "No grounded (RAG) answers to score — metrics table skipped "
            "(all answers came from cache or LLM fallback)."
        )
        return
    if k < n:
        summary_log.warning(
            "Metrics table covers %d of %d queries — the rest were answered from "
            "cache or LLM fallback (ungrounded), or failed evaluation.", k, n
        )
    avg_faithfulness      = sum(m["faithfulness"]        for m in all_metrics) / k
    avg_context_relevance = sum(m["context_relevance"]   for m in all_metrics) / k
    avg_retrieval_score   = sum(m["avg_retrieval_score"] for m in all_metrics) / k
    avg_docs_retrieved    = sum(m["num_docs_retrieved"]  for m in all_metrics) / k

    faithfulness_scores = ", ".join(f"{m['faithfulness']:.3f}"        for m in all_metrics)
    relevance_scores    = ", ".join(f"{m['context_relevance']:.3f}"   for m in all_metrics)
    retrieval_scores    = ", ".join(f"{m['avg_retrieval_score']:.4f}" for m in all_metrics)
    docs_counts         = ", ".join(str(m["num_docs_retrieved"])      for m in all_metrics)

    summary_log.info(
        "EVALUATION METRICS  (%d queries)\n"
        "  %-35s %7s  %s\n"
        "  %s %s  %s\n"
        "  %-35s %7.3f  %s\n"
        "  %-35s %7.3f  %s\n"
        "  %-35s %7.4f  %s\n"
        "  %-35s %7.1f  %s\n"
        "%s\n"
        "Score interpretation:\n"
        "  Faithfulness      1.0 = all answer claims are supported by retrieved context\n"
        "  Context Relevance 1.0 = all retrieved documents are relevant to the question\n"
        "  Hybrid Retrieval Score = RRF rank fusion, NOT a 0-1 relevance score:\n"
        "    each doc scores 1/(60+dense_rank) + 1/(60+bm25_rank), so the maximum\n"
        "    is ~0.033 (rank 1 in both lists) and ~0.016 means top-ranked in one\n"
        "    list only.  Retrieval QUALITY is measured by Context Relevance above.",
        k,
        "Metric", "Avg", "Per-query scores",
        "-" * 35, "-" * 7, "-" * 30,
        "Faithfulness", avg_faithfulness, faithfulness_scores,
        "Context Relevance", avg_context_relevance, relevance_scores,
        "Avg Hybrid Retrieval Score", avg_retrieval_score, retrieval_scores,
        "Docs Retrieved", avg_docs_retrieved, docs_counts,
        sep,
    )


# ---------------------------------------------------------------------------
# 7. Demo queries  +  groundedness evaluation
#    Edit these to match whatever you've ingested into the collection.
#    The defaults target the DeepSeek-R1 paper (2501.12948).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Questions come from the CLI (positional QUERY args).  With none given,
    # fall back to the built-in demo set targeting the DeepSeek-R1 paper.
    DEMO_QUESTIONS = [
        "What is DeepSeek-R1-Zero and how does it differ from DeepSeek-R1?",
        "What reinforcement learning algorithm is used to train DeepSeek-R1, and how does it work?",
        "What is the 'aha moment' described in the paper?",
        "How well do the distilled smaller models perform compared to the full model?",
    ]
    questions = ARGS.queries if ARGS.queries else DEMO_QUESTIONS
    if not ARGS.queries:
        log.info("No QUERY args given — running the built-in DeepSeek-R1 demo set")

    results: list[dict] = []
    for i, q in enumerate(questions, start=1):
        log.info("── Query %d/%d ──", i, len(questions))
        try:
            answer, docs, analysis = answer_question(q)
        except Exception:
            log.exception("Query %d/%d failed — skipping it", i, len(questions))
            continue
        # Groundedness only makes sense for a grounded (Tier-1 "rag") answer.
        # Cache hits and Tier-2 LLM fallbacks have no corpus context to be
        # faithful to, so they are reported as "not applicable" rather than
        # scored.  An evaluation failure (e.g. an OpenAI timeout in the
        # gpt-5-mini evaluators) must not lose the whole run — keep the answer,
        # mark metrics as unavailable, and carry on.
        if analysis["source"] != "rag":
            log.info(
                "Query %d/%d answered via '%s' — skipping groundedness evaluation",
                i, len(questions), analysis["source"],
            )
            metrics = None
        else:
            try:
                metrics = evaluate_groundedness(
                    q, answer, docs, context_relevance=analysis["context_relevance"]
                )
            except Exception:
                log.exception(
                    "Evaluation for query %d/%d failed — keeping the answer without metrics",
                    i, len(questions),
                )
                metrics = None
        results.append(
            {"question": q, "answer": answer, "metrics": metrics, "analysis": analysis}
        )

    log_run_summary(results)

    # -----------------------------------------------------------------------
    # 8. Standalone BM25-only retrieval (no dense embedding needed)
    #    Useful for keyword-heavy queries, exact-match lookups, or debugging
    #    which documents BM25 alone would rank top.
    # -----------------------------------------------------------------------
    from haystack import Pipeline
    from milvus_haystack import MilvusSparseEmbeddingRetriever

    bm25_pipeline = Pipeline()
    bm25_pipeline.add_component(
        "bm25_retriever",
        MilvusSparseEmbeddingRetriever(document_store=document_store, top_k=3),
    )

    log.info("── BM25-only retrieval example ──")
    bm25_question = "GRPO group relative policy optimization reward"
    bm25_result = bm25_pipeline.run(
        {"bm25_retriever": {"query_text": bm25_question}}
    )
    log.info("Query: %s", bm25_question)
    for i, doc in enumerate(bm25_result["bm25_retriever"]["documents"], start=1):
        score = f"{doc.score:.4f}" if doc.score is not None else "n/a"
        log.info("  [%d] score=%s — %s...", i, score, (doc.content or "")[:100])
