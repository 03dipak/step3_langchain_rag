# Step 3: LangChain RAG Rebuild — Study Notes

## The big question this step answers
Step 2 proved the pipeline works by hand. Step 3 asks: **what does LangChain actually do for me, and what does it hide?** The answer comes from rebuilding the *same* project two ways and comparing them.

## The one idea to internalize: Adapter over the registry ⭐
LangChain's `PromptTemplate` is a dumb string template. It does **not** know about versions, statuses, or eval scores. So:
```
Your Registry (source of truth)     LangChain (adapter/execution)
Content: template            ──→    PromptTemplate.from_template()
Policy: model, temperature   ──→    ChatOpenAI(model=..., temperature=...)
Evidence: eval scores, runs  ──→    (stays in the registry)
```
Every later step (4-7) imports the same registry. LangChain is just another consumer — via `LangChainPromptAdapter.build_chain()`.

## Where LangChain genuinely saved us
- **Splitting:** `RecursiveCharacterTextSplitter` = separator-aware chunking in ~5 lines (vs ~70-line hand chunker).
- **Vector store + persistence:** `Chroma` = on-disk cosine store + `persistence_directory` (vs numpy store + `.npz` + manual save/load).
- **LLM swapping:** `ChatOpenAI`/`ChatGroq`/`ChatGemini` — change one class, not client boilerplate.
- **LCEL:** composing `context | prompt | llm` reads declaratively.

## Where LangChain cost us / hides mechanics
- **Scores are hidden.** `similarity_search()` returns `Document`s with no score. To keep our `min_score` gate + `{text, metadata, score}` shape we had to reach into `vectorstore._collection.query(...)` and convert `distance → similarity (1 - distance)`. That's a private-API reacharound.
- **Chunk boundaries differ.** `RecursiveCharacterTextSplitter` splits on separators (`\n\n`, `\n`, `. `, space) with overlap; Step 2's chunker split on words/char count. Same corpus → *different* chunk set → small eval score deltas. Not a bug, but it changes retrieval.
- **Prompt versioning is absent.** You'd have to build the registry anyway (this is why the adapter exists).
- **Framework-internals debugging.** `langchain.debug=True` dumps lots of noise; finding "your" line is harder than reading your own code.

## Gotchas to remember
1. **`ChatOpenAI` uses `openai_api_base`/`openai_api_key`, not `base_url`.** LangChain 1.x exposes the base URL/key under legacy alias model fields. The OpenAI-compatible Qwen gateway still works — just with the old names.
2. **Embedding interface:** to use our own qwen3 model, implement LangChain's `Embeddings` (`embed_documents` + `embed_query`), not `HuggingFaceEmbeddings`. Keep `embed()` for docs and `query_embed()` for queries (instruction-aware) so docs/queries land in the same space.
3. **Chroma score recovery:** `1.0 - distance` = cosine similarity (Chroma's default distance metric for cosine is `1 - cosine_similarity`). Verify by comparing a known chunk to itself → score ~1.0.
4. **Evaluator needs basenames:** `metadata["source"]` must be `Path(file).name`, or `context_precision` mismatches the golden `source`.
5. **DeepEval in Streamlit:** run it as a **subprocess**; import-time `signal.signal()` needs the main thread.
6. **deepeval vs qwen3-embed / click:** pin `deepeval==2.9.3` to coexist cleanly with qwen3-embed (same as Step 2).
7. **`python -c` live tests don't load `.env`:** `python-dotenv` is a dependency but **isn't wired into the package** — a bare `uv run python -c "..."` for live LLM calls fails with `OpenAIError: Missing credentials` inside `ChatOpenAI.__init__` (the generator eagerly calls `build_chain`). Source `.env` into the shell first: `set -a && . ./.env && set +a`. Offline unit tests mock `LangChainPromptAdapter.build_chain` so they never need keys.

## The A/B eval read
Run the same golden set on both. Expected: roughly equal (same model + data), with small deltas from chunk boundaries and Chroma cosine. Any large gap = investigate the splitter or store, not the metrics (metrics are identical constants in this comparison).

## The four metrics (unchanged from Step 2 — memorize)
| Metric | Measures | Low score → fix |
|--------|----------|-----------------|
| Context Recall | retrieved coverage of gold | bigger chunks / higher top_k |
| Context Precision | how many retrieved are relevant | tighter min_score / metadata filter |
| Faithfulness | answer grounded in context | strengthen grounding prompt |
| Answer Relevance | answer addresses question | fix retrieval / query rewrite |

## Decision matrix (fill with your numbers in Task 19)
Hand-written when you need control/debugging/versioning; LangChain for speed/prototyping/provider-switching; **hybrid** (registry + LangChain adapter) for production — you get the framework's traction with your registry's versioning.

## Workflow
```
write pyproject (uv add) → splitter → embeddings → Chroma → retriever → reranker → registry+adapter → generator → pipeline → eval → app → CLI → tests → integration → DeepEval → compare vs Step 2
```
Each file: write it → run ruff/mypy/pytest → **have the mentor review it** → next.

## Embedding model (unchanged decision from Step 2)
`qwen3_embed` → `n24q02m/Qwen3-Embedding-0.6B-ONNX` (multilingual, 1024-dim, instruction-aware query path). Same model in Step 2 and Step 3 so the framework — not the embeddings — is what differs.

## Reranker (decision revised for Step 3)
**In Step 3 — yes (new).** Step 2 had **no** reranker, so it's the one component this step adds (**Task 7**): a **cross-encoder second stage** (`sentence-transformers` `cross-encoder/ms-marco-MiniLM-L-6-v2`) over the already-retrieved Step-2 dicts. It over-fetches Chroma candidates, re-scores, drops to `top_n`. This makes Step 2 vs Step 3 not a strict 1:1 — it's a 1:1 **plus a measure of what a real cross-encoder adds**, so any eval lift is attributed to the rerank stage. New dep to keep in mind: `sentence-transformers` (pulls torch + transformers; a second model + slower `ask()`). We deliberately did **not** use LangChain's `ContextualCompressionRetriever` + `CrossEncoderReranker`: that API needs a `Document`-based `BaseRetriever` and its import path isn't in the modern `langchain` split — two reasons it conflicts with our dict-shaped retriever. Multi-source reranking still belongs to Step 4 onward.

## Interfaces & tests (Tasks 17–18, mentor rulings)
- **Dense-only this step.** Hybrid dense+sparse / BM25 / RRF fusion is **deferred to Step 4**. Keep the retrieval path single-mode so eval scores stay interpretable; don't bolt hybrid flags into the CLI/pipeline now.
- **CLI (Task 17):** lazy import per command (so `--help` loads no models); **never stash CLI flags on a cached `get_settings()`** — pass `top_k`/`min_score` as args to `pipeline.ask()`. Extract a `_load_pipeline()` + `_fmt_chunk()` helper so `search`/`ask`/`eval` don't duplicate setup (that was the top duplication finding on the Side Project CLI). `eval` should reuse Task 12, not be a stub.
- **Rollback + security (architected):** prompt rollback is an **app-layer** operation (`registry.rollback()`, not git/HF); security is an **action-boundary `ADMIN_TOKEN`** gate, never a secret in a prompt string. Git = DR/audit backup only. Full design in `doc/ARCHITECTURE_ROLLBACK_SECURITY.md`; shared signed store + RBAC deferred to Step 7.
- **Output contract (`output_schema`) on each version:** each version owns *how the LLM answers* (`format`, `shape`, `length_policy`, `citation_policy`, `refusal_string`, `display`) — versioned prompt policy, **not** UI presentation. Editing it on an **approved** version forces a drop to `testing` (no silent change) and records a history diff; rollback restores the full output behavior of the target version. The generator reads `refusal_string` from the approved schema (no hardcode). See `docs/TASK_10`, `TASK_08`, and `ARCHITECTURE_ROLLBACK_SECURITY.md` §5.
- **Integration tests (Task 18):** mark `integration` and deselect by default (`addopts = "-m 'not integration'"`). **Load the embedding model once via a module-scoped fixture** — `Qwen3Embeddings` caches per-instance, so a per-test embedder would reload the ONNX model N times (the "hang my system" risk). Real embeddings + real Chroma; stub only the LLM chain in `ask()` so no API key is needed.

## Pipeline `ask()` (Task 9) — mentoring notes
- **Interface contract:** `Pipeline` exposes the exact Step 2 shape `{answer, sources, prompt_key, rendered_hash}` so Task 12/16 eval runs unchanged. `rendered_hash` = sha256 over the **approved** prompt record + question (deterministic). The reranker (Task 7) is inserted **transparently** behind the retriever via `RerankableRetriever` — the over-fetch factor (`top_k * 2`) lives only inside `RerankableRetriever.retrieve()`, and `ask()` calls the same `retriever.retrieve(...)` interface either way.
- **Gotcha (design tension):** `Pipeline.__init__` builds `LangChainGenerator` eagerly (spec-required), but `build_chain` needs an **already-approved** `RAG_ANSWER` prompt — which only exists after `registry.load()` in `load_documents()`. So a fresh `Pipeline()` throws unless an approved registry file is already on disk. Tests dodge this by patching `LangChainPromptAdapter.build_chain`; prod must ship a seeded registry (see `TASK_10`). Flag for Task 19 gap analysis: consider deferring the LLM build or letting generator read the registry lazily.
- **Tracing:** `ask()` wraps retrieve/generate with `trace_retrieve`/`trace_generate` (Task 11) — transparent no-ops without `LANGSMITH_API_KEY`, so the shape is identical with or without tracing.
- **Refusal path:** zero retrieved results → `generator.refusal_response(question)` + empty `sources`, still carrying `prompt_key`/`rendered_hash`. The hash must be computed **before** the no-results branch (spec has a minor ordering slip — lead ruling: compute first so refusal results are still deterministic).
- **Persistence:** `load_documents` idempotently reopens Chroma `persist_directory` (skip re-index) unless `force_rebuild=True`. `index.npz` arg kept only for Step 2 compat.

## Adapter testing (Task 10) — LCEL mock boundary gotcha
- **`build_chain` unit tests:** a plain `MagicMock` LLM is coerced by LCEL into a `RunnableLambda` that **calls the mock as a callable** (`fake(...)`, not `fake.invoke(...)`). So to capture what the LLM actually receives, set `fake.side_effect` and read the passed `ChatPromptValue` (`.messages[0].content` = rendered System message). Mocking `fake.invoke.side_effect` silently does nothing. `prompts.build_llm` is patched at module import site (`langchain_rag.prompts.build_llm`) so `build_chain` builds offline.
- **`build_llm` body (lines 27-31) is intentionally untested** (real `ChatOpenAI` construction would touch network/env) — that's why `prompts.py` sits at 91%, not 100%. Same deliberate mock boundary as `embeddings.py` download path.
