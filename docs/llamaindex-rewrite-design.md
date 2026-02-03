# QMD TypeScript → Python (LlamaIndex) Rewrite Design

## 1. Executive Summary

This document describes a high-level design to rewrite the QMD TypeScript RAG + hybrid search system into Python using **LlamaIndex** as the core framework. The goal is feature parity for ingestion, indexing, hybrid retrieval, query expansion, reranking, and evaluation while **avoiding any custom re-implementation** of LlamaIndex capabilities. The proposed design uses LlamaIndex abstractions for ingestion pipelines, retrievers, fusion/hybrid retrieval, postprocessors, response synthesis, and (if needed) agent/workflow tooling. All custom Python code is limited to **configuration, orchestration, and integration** with existing utilities in the target Python repo.

---

## 2. TypeScript Capability Inventory (ML/RAG/LLM Core)

> Scope: ingestion, indexing/storage, retrieval, post-processing, prompting/response, conversational/agents, evaluation. Non-ML utilities omitted.

### 2.1 Ingestion & Transformations

**Chunking (char-based + token-based)**
- **Files**: `src/store.ts` (`chunkDocument`, `chunkDocumentByTokens`).
- **Interfaces**: `chunkDocument(content, maxChars, overlapChars)` and `chunkDocumentByTokens(content, maxTokens, overlapTokens)`.
- **Behavior**:
  - Default: 800 tokens with 15% overlap. Token-based chunking uses the LlamaCpp tokenizer for accurate counts. Char-based fallback approximates 4 chars/token. Both prefer paragraph/sentence/line boundaries when slicing. Chunk metadata includes `pos` (character offset) and `tokens` for token-based chunking.【F:src/store.ts†L1166-L1363】
- **Data flow**: raw document text → chunks with positions/tokens → embedding pipeline.

**Embedding formatting**
- **Files**: `src/llm.ts` (`formatQueryForEmbedding`, `formatDocForEmbedding`).
- **Behavior**: Nomic-style prefix formatting. Query formatted as `task: search result | query: ...`; docs as `title: ... | text: ...`.【F:src/llm.ts†L22-L51】

**Embedding pipeline (batch + fallback)**
- **Files**: `src/qmd.ts` (`vectorIndex` command).
- **Behavior**: chunks documents by tokens, formats each chunk for embedding, batch embeds in groups of 32, inserts embeddings into SQLite vector table, and falls back to single-item embeddings on batch errors. First chunk determines vector dimension and builds vector table if needed.【F:src/qmd.ts†L1492-L1665】
- **Runtime dependencies**: local GGUF embedding model via node-llama-cpp (`embeddinggemma` default).【F:src/llm.ts†L170-L205】

### 2.2 Indexing & Storage

**Document store + content store + cache**
- **Files**: `src/store.ts` (`initializeDatabase`, `insertContent`, `insertDocument`, `llm_cache`).
- **Behavior**: SQLite tables for content (`content`), documents (`documents`), embeddings (`content_vectors`), and a cache table (`llm_cache`) keyed by a hash of query+doc for LLM reranking and query expansion caching.【F:src/store.ts†L480-L540】【F:src/store.ts†L1074-L1097】

**Lexical index (BM25)**
- **Files**: `src/store.ts` (`documents_fts`, `searchFTS`).
- **Behavior**: SQLite FTS5 virtual table with `bm25` scoring. FTS triggers keep index updated. Search builds a simple AND-constrained prefix query and converts negative BM25 into a [0..1] score (higher better).【F:src/store.ts†L517-L621】【F:src/store.ts†L1837-L1903】

**Vector index (sqlite-vec)**
- **Files**: `src/store.ts` (`ensureVecTableInternal`, `searchVec`, `insertEmbedding`).
- **Behavior**: virtual table `vectors_vec` (sqlite-vec) with cosine distance. Vector search runs a **two-step query**: retrieve nearest vectors, then join document metadata separately to avoid sqlite-vec JOIN hang. Vector results are scored as `1 - cosine_distance` with per-doc dedupe by best chunk distance.【F:src/store.ts†L565-L637】【F:src/store.ts†L1905-L2018】

### 2.3 Retrieval (Lexical, Vector, Hybrid)

**Lexical retrieval**
- **Files**: `src/store.ts` (`searchFTS`).
- **Inputs/outputs**: query string → list of `SearchResult` with metadata, body, context, scores (BM25).【F:src/store.ts†L1846-L1903】

**Vector retrieval**
- **Files**: `src/store.ts` (`searchVec`).
- **Inputs/outputs**: query string → embed query → retrieve vector matches → list of `SearchResult` with chunk positions and vector scores.【F:src/store.ts†L1905-L2018】
- **Runtime dependency**: LlamaCpp embedding model for query embedding.【F:src/store.ts†L2025-L2042】

**Query expansion (lex/vec/hyde)**
- **Files**: `src/llm.ts` (`expandQuery`), `src/qmd.ts` (`expandQueryStructured`).
- **Behavior**:
  - Uses LlamaCpp generate model with a grammar to emit lines tagged as `lex`, `vec`, `hyde`.
  - Filters expansions that don’t include query terms; produces fallback expansions if generation fails.
  - Query expansion is used in vector search and hybrid search to increase recall.【F:src/llm.ts†L791-L861】【F:src/qmd.ts†L2036-L2072】

**Hybrid retrieval with RRF fusion**
- **Files**: `src/qmd.ts` (`querySearch`), `src/store.ts` (`reciprocalRankFusion`).
- **Behavior**:
  - Runs expanded lexical (FTS) + vector searches in parallel.
  - Applies **Reciprocal Rank Fusion**; original query results are weighted 2x.
  - Produces fused candidate list for reranking, with a hard rerank limit (40 docs).【F:src/qmd.ts†L2101-L2219】【F:src/store.ts†L2140-L2188】

### 2.4 Post-processing & Reranking

**Chunk selection for reranking**
- **Files**: `src/qmd.ts` (`querySearch`).
- **Behavior**: For each candidate doc, chunk body; pick the chunk with best keyword hits and rerank only that chunk (one chunk/doc). Tracks chunk position for snippet generation.【F:src/qmd.ts†L2198-L2238】

**Reranking**
- **Files**: `src/store.ts` (`rerank`), `src/llm.ts` (`rerank`).
- **Behavior**: LlamaCpp reranker returns relevance scores; results cached in `llm_cache` keyed by query+file+model. Rerank results blend with RRF rank using position-aware weights to protect top lexical hits.【F:src/store.ts†L2080-L2109】【F:src/qmd.ts†L2239-L2259】

**Snippet extraction / presentation**
- **Files**: `src/store.ts` (`extractSnippet`).
- **Behavior**: uses chunk position hint to search locally for best line; returns diff-style header with line numbers for excerpts.【F:src/store.ts†L2512-L2568】

### 2.5 Prompting & Response Synthesis

- **Observed**: No explicit LLM response synthesis beyond query expansion and reranking. Search results are formatted and returned as snippets or full docs; there is no generation of final natural-language answers in the TS system (CLI/MCP return search results).【F:src/qmd.ts†L1919-L2034】【F:src/mcp.ts†L360-L437】

### 2.6 Conversational RAG / Memory

- **Observed**: No conversation memory or chat history management in core logic.

### 2.7 Agents / Tools

**MCP tools exposure**
- **Files**: `src/mcp.ts`.
- **Behavior**: registers MCP tools for search, vector search, hybrid query (RRF+rerank), and document retrieval. Tool outputs include snippets and metadata; tool surface is for external agent orchestration, but QMD itself does not implement agent loops.【F:src/mcp.ts†L300-L437】

### 2.8 Evaluation (ML/RAG Behavior Only)

**Search quality evaluation**
- **Files**: `src/eval.test.ts`.
- **Behavior**: Golden queries with expected documents; separate metrics for BM25, vector, and hybrid; asserts Hit@K thresholds. Embeddings generated with `chunkDocumentByTokens` and `formatDocForEmbedding` before vector tests.【F:src/eval.test.ts†L1-L261】

---

## 3. Conceptual Architecture (TS System)

### 3.1 Narrative

1. **Ingestion**: markdown content is stored in SQLite (`content` + `documents` tables). For embeddings, documents are chunked (token-based preferred), formatted with embedding-specific prefixes, embedded by LlamaCpp, and persisted in `content_vectors` + `vectors_vec`.
2. **Retrieval**:
   - **Lexical**: FTS5 BM25 query, scored and returned with context/snippet.
   - **Vector**: embed query → sqlite-vec ANN search → dedupe by best chunk per document.
3. **Hybrid**: query expansion emits lexical/vector/HyDE queries; results are fused by RRF, then reranked with a LlamaCpp reranker on the best chunk per document. Final ranking blends RRF position scores with rerank scores.
4. **Output**: results are returned as snippet-based citations or full text (CLI/MCP). There is no final LLM response synthesis.

### 3.2 Component Diagram (ASCII)

```
[Docs] -> [Chunking] -> [Embeddings] -> [Vector Store]
    \-> [Content Store + FTS]

Query -> [Query Expansion] -> [FTS Retriever] ----\
         [Query Expansion] -> [Vector Retriever] --+--> [RRF Fusion] -> [Chunk Selection] -> [Reranker] -> [Final Ranked Results]

Final Ranked Results -> [Snippet Extraction/Formatting] -> Output
```

---

## 4. Capability → LlamaIndex Mapping Table

**References to LlamaIndex docs (verified via web search):**
- Querying & query engines: https://developers.llamaindex.ai/python/framework/understanding/rag/querying/
- Response synthesizers: https://developers.llamaindex.ai/python/framework/module_guides/querying/response_synthesizers/
- Ingestion pipeline: https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/
- Node parsers: https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/
- Retriever API: https://developers.llamaindex.ai/python/framework/module_guides/querying/retriever/
- Reciprocal Rerank Fusion retriever: https://developers.llamaindex.ai/python/framework/integrations/retrievers/reciprocal_rerank_fusion/
- Function-calling agent workflow: https://developers.llamaindex.ai/python/examples/workflow/function_calling_agent/

| TS capability | Evidence in TS repo | LlamaIndex abstraction(s) | Config knobs to match TS | Gaps / differences | Custom glue required (allowed) |
| --- | --- | --- | --- | --- | --- |
| Token/char chunking with overlap | `chunkDocument`, `chunkDocumentByTokens` in `src/store.ts`.【F:src/store.ts†L1166-L1363】 | `NodeParser` modules (e.g., `SentenceSplitter`, `TokenTextSplitter`) + `IngestionPipeline` | `chunk_size`, `chunk_overlap`, `separator`/`split_by`, token-based vs char-based | TS prefers paragraph/sentence/line boundaries; may need best-fit LlamaIndex splitter settings | Configure a `NodeParser` in pipeline; choose token splitter with overlap; pass token counting/metadata. |
| Embedding formatting (query/doc prefixes) | `formatQueryForEmbedding`, `formatDocForEmbedding` in `src/llm.ts`.【F:src/llm.ts†L22-L51】 | `TransformComponent` in `IngestionPipeline` + query transform on retriever | Prefix strings for queries/docs; title injection | LlamaIndex doesn’t know custom prefixes by default | Minimal transform wrapper to apply prefix before embedding. |
| Embedding batch + persistence | `vectorIndex` in `src/qmd.ts`.【F:src/qmd.ts†L1492-L1665】 | `IngestionPipeline` + `VectorStoreIndex.from_documents` (or `from_nodes`) | Batch size, embedding model, storage backend | Ensure same batching behavior; ensure embedding dimensions | Orchestrate pipeline + vector store init; use LlamaIndex storage context. |
| Lexical BM25 search (FTS5) | `searchFTS` in `src/store.ts`.【F:src/store.ts†L1846-L1903】 | `BM25Retriever` or equivalent lexical retriever | `top_k`, BM25 params, filters | Different tokenizer/scoring from SQLite FTS5 | Configure BM25 retriever; accept minor scoring differences; keep filters. |
| Vector retrieval | `searchVec` in `src/store.ts`.【F:src/store.ts†L1905-L2018】 | `VectorIndexRetriever` | `top_k`, similarity metric | sqlite-vec query semantics differ from vector store | Select vector store backend with cosine similarity; ensure score normalization. |
| Query expansion (lex/vec/hyde) | `expandQuery` in `src/llm.ts`, `expandQueryStructured` in `src/qmd.ts`.【F:src/llm.ts†L791-L861】【F:src/qmd.ts†L2036-L2072】 | LlamaIndex `QueryTransform` / `QueryPipeline` components | number of expansions, include lexical vs vector, grammar-based output | LlamaIndex may not include grammar-structured expansion out-of-the-box | Minimal custom QueryTransform to generate expanded queries (if LlamaIndex doesn’t provide). |
| Hybrid retrieval + RRF fusion | `querySearch`, `reciprocalRankFusion`.【F:src/qmd.ts†L2101-L2219】【F:src/store.ts†L2140-L2188】 | `ReciprocalRerankFusionRetriever` | weights for original query, `k`, per-retriever `top_k` | Score blending differences | Configure fusion retriever with weights; avoid manual RRF implementation. |
| Reranking | `rerank` in `src/store.ts`, `src/llm.ts`.【F:src/store.ts†L2080-L2109】【F:src/llm.ts†L874-L916】 | LlamaIndex `NodePostprocessor` / `LLMRerank` | reranker model, top_n | LlamaIndex reranker may differ from node-llama-cpp | Use LlamaIndex reranker integration; map model config. |
| Best-chunk selection for rerank | `querySearch` chunk heuristic in `src/qmd.ts`.【F:src/qmd.ts†L2198-L2238】 | `NodePostprocessor` + metadata/score-based selection | `top_k`, scoring heuristic | LlamaIndex might rerank nodes directly | Use LlamaIndex node postprocessors; avoid reimplementing rerank model. |
| Result formatting/snippets | `extractSnippet` in `src/store.ts`.【F:src/store.ts†L2512-L2568】 | Response synthesizer / node postprocessor (optional) | snippet length, line numbers | LlamaIndex doesn’t provide snippet formatting by default | Keep snippet extraction as a non-LLM utility (allowed). |
| MCP tools (search/query/get) | `src/mcp.ts` tool registration. 【F:src/mcp.ts†L300-L437】 | LlamaIndex tools or workflows (if agents used) | tool schema, output formatting | Python repo may already have tooling framework | Implement tool handlers that call LlamaIndex pipelines. |
| Search evaluation | `src/eval.test.ts` metrics and thresholds. 【F:src/eval.test.ts†L1-L261】 | LlamaIndex eval / custom harness | hit@k thresholds, golden queries | Use existing tests or LlamaIndex eval packages | Keep golden query eval as custom tests. |

---

## 5. Proposed Python Architecture & Module Boundaries

> **Adapt this design to the existing Python repo structure.** If a matching module already exists, integrate there rather than creating new ones.

### 5.1 Design Principles

- Prefer **LlamaIndex-native constructs**; do **not** wrap or re-implement LlamaIndex features.
- If a TS component maps to a LlamaIndex component, **use the LlamaIndex component**.
- Custom code is limited to configuration, composition, integration, and domain logic (e.g., prefix formatting, SQLite metadata extraction, MCP wiring).
- Build for testability: deterministic retrieval configs, injectable LLM/embedding providers, prompt snapshot tests.

### 5.2 Suggested Modules (adapt to Python repo)

- `ingestion/`
  - Build `IngestionPipeline` with node parsers, transforms, and embedding (LlamaIndex).
  - Custom transform for query/doc prefix formatting.

- `indexing/`
  - Create/load `VectorStoreIndex` + `StorageContext` + docstore.
  - Bind to existing DB utilities if needed for metadata storage.

- `retrieval/`
  - LlamaIndex retrievers: BM25 retriever + vector retriever.
  - Hybrid fusion using `ReciprocalRerankFusionRetriever`.
  - Node postprocessors / reranker configuration.

- `querying/`
  - `RetrieverQueryEngine` (or equivalent) wired with retrievers + response synthesizer.
  - Response synthesis settings (refine/compact/tree if needed).

- `agents/` (only if TS usage indicates agent loops)
  - Use LlamaIndex workflow/agent primitives for MCP tools or function-calling.

- `domain/`
  - Glue for existing DB/session/config utilities.
  - Keep snippet extraction, context lookup, and path normalization here (non-LLM utilities already in target repo).

---

## 6. End-to-End Dataflows

### 6.1 Ingestion Flow

1. Read documents from filesystem or existing content store.
2. Apply `IngestionPipeline` with `NodeParser` (token-based chunking, overlap).
3. Apply custom transform to prefix text for embedding (query/doc format).
4. Use LlamaIndex embedding model to embed nodes.
5. Persist nodes/embeddings in `VectorStoreIndex` + docstore.
6. Persist metadata (hash, title, collection, position) via existing Python repo utilities.

### 6.2 Query Flow (RAG)

1. Input query → optional query transform (expansion / HyDE).
2. Use retriever(s) to fetch nodes (vector, lexical).
3. Apply node postprocessors (rerank, filters).
4. Feed nodes to `ResponseSynthesizer` to generate final answer (if user expects LLM response), else return ranked nodes + snippets.

### 6.3 Hybrid Flow (BM25 + Vector + Fusion)

1. Query expansion (lex + vec + hyde) via LlamaIndex query transform.
2. Execute BM25 retriever and vector retriever.
3. Fuse results via `ReciprocalRerankFusionRetriever` with weights favoring original query.
4. Apply reranker postprocessor (`LLMRerank` or equivalent).
5. Return ranked results or synthesize response.

### 6.4 Agent/Tool Flow (if needed)

1. Tool schema definitions align with existing MCP tools.
2. LlamaIndex agent/workflow executes tools: search, vector search, hybrid query, get doc.
3. Tool outputs format into snippets with line numbers (existing utility), returned to caller.

---

## 7. Configuration Spec (ML/RAG Only)

> This is a conceptual schema; integrate with the existing Python config system.

- `chunking`
  - `chunk_size_tokens: int` (default 800)
  - `chunk_overlap_tokens: int` (default 120)
  - `splitter: str` (e.g., sentence/token)
- `embedding`
  - `provider: str` (local/remote)
  - `model_name: str`
  - `batch_size: int` (default 32)
- `vector_store`
  - `backend: str` (e.g., FAISS, SQLite, Postgres)
  - `similarity_metric: str` (cosine)
- `bm25`
  - `top_k: int`
  - `tokenizer: str`
- `hybrid_fusion`
  - `strategy: str` (`rrf`)
  - `weights: list[float]`
  - `rrf_k: int`
- `reranker`
  - `enabled: bool`
  - `model_name: str`
  - `top_n: int`
- `query_expansion`
  - `enabled: bool`
  - `include_lexical: bool`
  - `max_expansions: int`
- `response_synthesis`
  - `mode: str` (compact/refine/tree/summarize)
  - `templates: {}` (prompt templates)
- `safety`
  - `redaction_enabled: bool`
  - `filters: list[str]`
- `evaluation`
  - `golden_queries_path: str`
  - `hit_k_thresholds: {}`

---

## 8. Compatibility Targets (Parity with TS)

- **Retrieval parity**: match `top_k`, filters, and fusion weights; use RRF fusion settings equivalent to TS (original queries weighted 2x, `k=60`).
- **Query expansion parity**: support lex/vec/hyde expansions; keep a fallback to original query if expansion fails.
- **Reranking parity**: use LlamaIndex reranker and limit rerank to ~40 candidates; use postprocessors to select one chunk per doc if needed.
- **Prompt parity**: port embedding prefixes and query expansion prompt exactly (as config or template files).
- **Response format parity**: keep snippet extraction with line numbers and docid-style references if required by downstream agents.
- **Performance parity**: batch embeddings; enable caching via LlamaIndex or existing repo utilities where possible.

---

## 9. Implementation Plan (for the Integrating Agent)

### Phase 1 — Vertical Slice (Hybrid Search for One Corpus)
- Implement ingestion pipeline with token-based chunking + embeddings.
- Build vector index + BM25 retriever.
- Implement RRF fusion retriever.
- **DoD**: Given a small corpus, hybrid search returns ranked results for a query.

### Phase 2 — Reranking + Query Expansion
- Add query expansion transform (lex/vec/hyde).
- Add LlamaIndex reranker postprocessor.
- **DoD**: Hybrid search uses expanded queries, reranks top candidates, and returns stable ordering.

### Phase 3 — Tool/Workflow Integration (if needed)
- Map MCP tool endpoints to LlamaIndex query engines.
- Add response formatting/snippets.
- **DoD**: MCP tools return structured results in same schema as TS.

### Phase 4 — Parity Backfill
- Port embedding prefixes, chunk overlap rules, and scoring weights.
- Add context injection (path-based context) if needed.
- **DoD**: outputs are functionally similar to TS on the same dataset.

### Phase 5 — Eval + Regression
- Port golden queries and thresholds from TS tests.
- Add prompt snapshot tests.
- **DoD**: test suite passes; retrieval quality meets TS thresholds.

---

## 10. Tests & Evaluation Strategy (RAG-Specific)

- **Golden query set**: reuse `eval.test.ts` queries and expected docs; implement Hit@K tests in Python.
- **Retrieval unit tests**: verify metadata filters, collection constraints, and fusion ordering.
- **Prompt snapshots**: capture embedding/query expansion prompts and compare snapshots.
- **Safety tests** (if applicable): validate any redaction or filtering stages.
- **Regression harness**: run hybrid search on a static corpus; compare ranked lists over time.

---

## 11. Risks, Gaps, and “Won’t Build” Items

### Risks / Gaps
- **Scoring drift**: LlamaIndex retrievers and BM25 may score differently than SQLite FTS5; expect small ranking differences.
- **Reranker model differences**: LlamaIndex reranker may not match node-llama-cpp scores; calibrate by adjusting weights/top_n.
- **Vector store behavior**: sqlite-vec search behavior (two-step query) differs from standard vector stores; test for parity.

### Explicit “Won’t Build” (because LlamaIndex provides it)
- Custom vector index or ANN search logic.
- Custom BM25 implementation or FTS query parser.
- Custom query engines or response synthesizers.
- Custom reranking models or fusion implementations.

