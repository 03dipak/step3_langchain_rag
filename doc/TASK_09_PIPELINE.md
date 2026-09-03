# Task 9: Implement Pipeline (LCEL, same `ask()` interface as Step 2)

## Objective
Create `src/langchain_rag/pipeline.py` that wires the LangChain components together (splitter → Chroma → reranker → registry adapter → LCEL chain) and exposes **the exact same public interface as Step 2** (`load_documents`, `ask`, `get_stats`, plus `retriever`/`generator`/`registry`) — so the evaluator (Task 12) and DeepEval (Task 16) run unchanged.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/pipeline.py`

## The contract (kept identical to Step 2)
```python
class Pipeline:
    def __init__(self, index_path="index.npz", prompt_key=None) -> None:
        self.store = LangChainStore()          # Chroma
        self.splitter = LangChainSplitter()
        self.retriever: Retriever | None = None   # wrapped in Reranker (Task 7)
        self.generator = LangChainGenerator(registry, prompt_key)
        self.registry = PromptRegistry()
        self.prompt_key = prompt_key

    def load_documents(self, data_dir="data/documents", force_rebuild=False) -> None: ...
    def ask(self, question, top_k=3, min_score=0.5) -> dict: ...
    def get_stats(self) -> dict: ...
```

> ⭐ **Step 3 difference:** Step 2 persisted the raw vector store to `index.npz`. Step 3 uses Chroma's `persist_directory="chroma_db"` for the same purpose. `index.npz` is kept as a compatibility arg but the *actual* persistence is Chroma.

### Method: `load_documents`
**Logic:**
1. `registry.load()` (restore approved prompts/evidence).
2. If `index_path` exists **and** not `force_rebuild` → reopen the Chroma store (`persist_directory`) and skip indexing (idempotent).
3. Else → `store.clear()`, then:
   - `chunks = splitter.load_directory(data_dir)` → `[{text, metadata}]`
   - `store.add_documents(chunks)` (embeds via the shared `Qwen3Embeddings`)
4. Create `Retriever(store)`, then wrap it in a `RerankableRetriever(Retriever(store), Reranker(top_n=top_k))` (Task 7). The base retriever over-fetches **`top_k * 2`** candidates, the cross-encoder re-scores and drops to `top_n = top_k` (over-fetch factor lives only inside `RerankableRetriever.retrieve()`); `ask()` still calls `retriever.retrieve(question, top_k, min_score)` on the transparently-swapped object (interface unchanged).
> If `chroma_db` exists, reopening is cheap (no re-embedding). Decide your rebuild flag: `force_rebuild=True` clears and re-indexes.

### Method: `ask` — eval-ready result (unchanged shape) ⭐
```python
def ask(self, question, top_k=3, min_score=0.5) -> dict:
```
1. If `retriever is None` → raise `ValueError("Retriever not initialized. Call load_documents() first.")`
2. `prompt = self.registry.get(self.prompt_key or "RAG_ANSWER")` → `prompt_key = prompt["key"]` (approved version)
3. `results = self.retriever.retrieve(question, top_k=top_k, min_score=min_score)` — via the reranker (Task 7): the base retriever returns **`top_k * 2`** candidates, cross-encoder re-scores, returns `top_k` (collect timing). Wrap with the tracer (Task 11):
   ```python
   from langchain_rag.tracer import trace_retrieve
   results = trace_retrieve(
       lambda: self.retriever.retrieve(question, top_k=top_k, min_score=min_score),
       question, top_k,
   )
   ```
4. If no results → return `{answer: generator.refusal_response(question), sources: [], prompt_key, rendered_hash}`
5. `rendered_hash = sha256(json.dumps({**prompt, "question": question}, sort_keys=True).hexdigest())` (deterministic — same as Step 2)
6. `output = generator.generate(question, results)` (LCEL chain), wrapped with the tracer (Task 11):
   ```python
   from langchain_rag.tracer import trace_generate
   output = trace_generate(lambda: self.generator.generate(question, results), question, results)
   ```
7. `registry.log_run(key=prompt_key, rendered_hash=rendered_hash, retrieved_doc_ids=[r["metadata"]["source"] for r in results], output=output, latency_ms=..., token_usage={...})`
8. Return `{answer: output, sources: results, prompt_key, rendered_hash}`

> The tracer wrappers are **transparent no-ops** when no `LANGSMITH_API_KEY` is set (Task 11) — same return value, zero overhead — so `ask()` behaves identically with or without tracing.

> The shape is bit-for-bit compatible with Step 2 — this is what lets Task 12/16 compare the two frameworks on the same metrics.

### `get_stats`
Return `{index_exists, num_chunks, retriever_initialized, prompt_key}` (`num_chunks` = `store.count()`).

## Integration with other tasks
- Uses `splitter.py` (Task 3), `vectorstore.py` (Task 5), `retriever.py` (Task 6), `reranker.py` (Task 7), `generator.py` (Task 8 via the adapter), `prompt_registry.py` (Task 10).
- Wraps retrieve/generate with tracer (Task 11) once it exists.
- `registry.get(prompt_key)["key"]` and `registry.log_run(...)` come from Task 10.
- `REQUIRED_EVAL_KEYS=("accuracy",)` — Task 12's `evaluate_with_registry` must record `accuracy`.

## Completion Criteria
- [ ] `pipeline.py` created; `load_documents` defaults to `data/documents`
- [ ] `ask()` returns `{answer, sources, prompt_key, rendered_hash}`
- [ ] `ask()` uses the **approved** prompt from the registry + LCEL generator
- [ ] `retrieved_doc_ids` from `r["metadata"]["source"]`
- [ ] `rendered_hash` deterministic (SHA-256 over prompt dict + question)
- [ ] Persists via Chroma `persist_directory` (idempotent reopen)
- [ ] Works offline (no LangSmith key, no LLM) without crashing when no args hit the API

## Mocking (learner note) — mock collaborators for orchestration tests
`Pipeline` sequences splitter → store → retriever → reranker → generator. Test orchestration, not the models:
- Mock `splitter.load_directory` → canned chunks.
- Mock `store` (`add_documents`, `count`, `search`) or provide a real Chroma on `tmp_path` with a **mocked embedding** returning random vectors.
- Mock `generator.generate` (or its `chain`) to avoid the LLM.
- Mock `registry` to avoid file I/O, or use a real one on `tmp_path`.
- Mock `retriever.retrieve` for the "no results" refusal path.
- Mock the reranker (Task 7) to return a fixed ordering without loading a cross-encoder.
Everything offline, fast, deterministic — same rule as Step 2.

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/pipeline.py
uv run mypy src/langchain_rag/pipeline.py
uv run pytest tests/test_pipeline.py -v
uv run pytest tests/test_pipeline.py -v --cov=langchain_rag.pipeline --cov-report=term-missing
uv run ruff check tests/test_pipeline.py
uv run mypy tests/test_pipeline.py
```

## Report Back
When done, tell me:
1. Stats output (chunk_count, retriever_initialized)
2. First 150 chars of an `ask()` answer (or a mocked one)
3. Number of sources returned
4. `prompt_key` and `rendered_hash` (first 12 chars)
5. Paste your `pipeline.py` for review — I want to see how you wired the LCEL `ask()` and where Chroma persistence lives
