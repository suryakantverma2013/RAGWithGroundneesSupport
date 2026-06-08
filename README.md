
Good — latest is **Haystack 2.29.0** (May 2026) and **milvus-haystack** (Dec 2025). Now I have everything needed to write accurate, up-to-date code.**Installation:**

---
```bash
pip install "haystack-ai>=2.29.0" milvus-haystack "pymilvus>=2.5.0" openai
```
---

### How this differs from the LangChain version

The structure reflects Haystack 2.x's explicit pipeline philosophy, which has some meaningful differences worth calling out.

**Two separate pipelines, not one chain.** Haystack separates indexing from retrieval by design. The indexing pipeline runs once (or on schedule), the RAG pipeline runs per query. In LangChain, both happened implicitly inside `Milvus.from_documents()`.

**Explicit component wiring.** Every connection between components is declared with `pipeline.connect("source.output", "target.input")`. The component graph is inspectable and serialisable to YAML — important for the production reproducibility that Haystack prioritises over LangChain.

**`MilvusHybridRetriever` receives two inputs simultaneously.** The `query_embedding` (from `OpenAITextEmbedder`) drives the dense ANN search, while `query_text` (passed directly) drives the BM25 sparse search. The retriever fuses both result sets with RRF internally — no `EnsembleRetriever` needed as in LangChain.

**`PromptBuilder` uses Jinja2 templates**, not Python format strings. The template receives the `documents` list directly, so you can iterate over metadata in the template itself rather than in a helper function.

**Section 9** shows a standalone `MilvusSparseEmbeddingRetriever` pipeline — the BM25-only path — which is handy for debugging keyword recall independently of the dense vector path before combining them.

### Generate the required fields by running the below python code ###
F:\UdemyCourses_Study_Material\BM25IndexingUsingApacheSolr\MilvusWithHaystack\Milvus_Collection_With_Fields.py


