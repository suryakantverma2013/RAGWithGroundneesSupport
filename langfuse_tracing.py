"""
langfuse_tracing.py — shared Langfuse tracing setup for the Haystack pipelines
================================================================================
Used by:
    ingest_pdf.py                 → trace name "ingest_pdf"
    haystack_milvus_hybrid_rag.py → trace name "hybrid_rag_query"

Tracing activates automatically when LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY
are set.  LANGFUSE_HOST defaults to the local Docker deployment
(http://localhost:3000, see LANGFUSE_SETUP.md).

Why a custom span handler is needed
-----------------------------------
* OpenAI embedders are not in langfuse-haystack's generator list, so by
  default they produce plain SPANs with no model / token-usage / cost fields.
* Langfuse tokenizes any input/output content attached to a GENERATION and
  lets that estimate OVERRIDE explicitly provided usage numbers.  Embedded
  document payloads also blow the per-event size limit (215 docs x 1536-dim
  embeddings ~ 7 MB), triggering truncation that destroys usage metadata.

The handler below records embedders as GENERATION spans carrying the exact
OpenAI-reported token usage, withholds the raw payload (a compact summary
goes into span metadata, which Langfuse never tokenizes), and compacts the
DocumentWriter's input so no event exceeds the size limit.

IMPORTANT: call enable_langfuse() BEFORE importing haystack in the caller —
HAYSTACK_CONTENT_TRACING_ENABLED must be set before haystack's tracer loads.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def enable_langfuse(trace_name: str) -> bool:
    """
    Enable Haystack → Langfuse tracing for this process.

    Parameters
    ----------
    trace_name : name shown for each trace in the Langfuse UI
                 (one trace per pipeline.run() call).

    Returns
    -------
    True if tracing was enabled, False otherwise (missing keys or package).
    """
    if not (os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY")):
        log.info("Langfuse tracing off — set LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY to enable.")
        return False

    # The Langfuse SDK reads LANGFUSE_HOST from the environment; default to the
    # local Docker deployment instead of cloud.langfuse.com.
    os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")
    # Must be set BEFORE haystack is imported, or span content (inputs/outputs,
    # token usage) is not captured.
    os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")

    try:
        from langfuse import Langfuse
        from haystack_integrations.tracing.langfuse import LangfuseTracer
        from haystack_integrations.tracing.langfuse.tracer import (
            DefaultSpanHandler,
            LangfuseSpan,
            SpanContext,
        )
        from haystack import tracing
    except ImportError:
        log.warning(
            "langfuse-haystack package not installed — tracing disabled.\n"
            "  Install: uv add langfuse-haystack"
        )
        return False

    class _SummarizingSpan(LangfuseSpan):
        """
        Caches component input/output locally (handle() reads it) but does
        NOT ship the raw payload to Langfuse.  Two reasons:
          - documents with 1536-dim embeddings exceed Langfuse's event
            size limit and would be truncated anyway;
          - on GENERATION spans, Langfuse tokenizes any input/output
            content it receives and lets that count OVERRIDE provided
            usage numbers — so the only way to record exact OpenAI token
            counts is to send no tokenizable content at all.
        With send_compact=True a documents-list is replaced by a count
        (safe for plain spans, which are never tokenized).
        """

        def __init__(self, span, *, send_compact: bool) -> None:
            super().__init__(span)
            self._send_compact = send_compact

        @staticmethod
        def _compact(value):
            if isinstance(value, dict) and isinstance(value.get("documents"), list):
                value = {**value, "documents": f"<{len(value['documents'])} documents>"}
            return value

        def set_content_tag(self, key: str, value) -> None:
            self._data[key] = value  # full payload stays available to handle()
            if self._send_compact:
                if key.endswith(".input"):
                    self._span.update(input=self._compact(value))
                elif key.endswith(".output"):
                    self._span.update(output=self._compact(value))

    class _EmbedderAwareSpanHandler(DefaultSpanHandler):
        """
        OpenAI embedders and LLM-based evaluators are not in langfuse-haystack's
        generator list, so by default they get plain SPANs with no
        model/usage/cost.  This handler records both as GENERATION spans
        carrying the exact token usage reported by the OpenAI API (see
        _SummarizingSpan for why the raw payload must be withheld).
        """

        _EMBEDDERS = ("OpenAIDocumentEmbedder", "OpenAITextEmbedder")
        # LLM-as-judge evaluators: run gpt-4o-mini internally and surface its
        # per-call meta (model + usage) in their output under "meta".
        _EVALUATORS = ("FaithfulnessEvaluator", "ContextRelevanceEvaluator", "LLMEvaluator")

        def create_span(self, context: SpanContext) -> LangfuseSpan:
            parent = context.parent_span
            if parent and context.component_type in (*self._EMBEDDERS, *self._EVALUATORS):
                return _SummarizingSpan(
                    parent.raw_span().generation(name=context.name),
                    send_compact=False,  # no content → provided usage wins
                )
            if parent and context.component_type == "DocumentWriter":
                return _SummarizingSpan(
                    parent.raw_span().span(name=context.name),
                    send_compact=True,  # avoid event-size truncation warnings
                )
            return super().create_span(context)

        def handle(self, span: LangfuseSpan, component_type: str | None) -> None:
            super().handle(span, component_type)
            output = span.get_data().get("haystack.component.output") or {}

            if component_type in self._EMBEDDERS:
                meta = output.get("meta") or {}
                usage = meta.get("usage") or {}
                lf_usage = None
                if usage.get("prompt_tokens"):
                    lf_usage = {
                        "input": usage["prompt_tokens"],
                        "total": usage.get("total_tokens", usage["prompt_tokens"]),
                        "unit": "TOKENS",
                    }
                span.raw_span().update(
                    model=meta.get("model"),
                    usage=lf_usage,
                    # metadata is never tokenized — safe home for the summary
                    metadata={
                        "embedding_meta": meta,
                        "documents_embedded": len(output.get("documents") or []),
                    },
                )

            elif component_type in self._EVALUATORS:
                # Evaluator output "meta" is a list of per-LLM-call meta dicts.
                metas = output.get("meta") or []
                model = next((m.get("model") for m in metas if m.get("model")), None)
                p_tokens = sum((m.get("usage") or {}).get("prompt_tokens", 0) for m in metas)
                c_tokens = sum((m.get("usage") or {}).get("completion_tokens", 0) for m in metas)
                lf_usage = None
                if p_tokens or c_tokens:
                    lf_usage = {
                        "input": p_tokens,
                        "output": c_tokens,
                        "total": p_tokens + c_tokens,
                        "unit": "TOKENS",
                    }
                scores = {k: v for k, v in output.items() if k != "meta"}
                span.raw_span().update(
                    model=model,
                    usage=lf_usage,
                    metadata={"evaluator_result": scores, "llm_meta": metas},
                )

    tracing.enable_tracing(
        LangfuseTracer(
            tracer=Langfuse(),
            name=trace_name,
            span_handler=_EmbedderAwareSpanHandler(),
        )
    )
    log.info("Langfuse tracing enabled (%s) → %s", trace_name, os.environ["LANGFUSE_HOST"])
    return True
