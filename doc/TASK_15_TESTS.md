# Task 15: Create Tests

## Objective
Create the `tests/` suite covering every LangChain module **offline** — no LLM calls, no embedding downloads, no Chroma network churn. This proves the LangChain rebuild is correct *and* locked in.

## Files (one per module, mirroring Step 2)
```
tests/
├── test_splitter.py
├── test_embeddings.py
├── test_vectorstore.py
├── test_retriever.py
├── test_generator.py
├── test_prompt_registry.py
├── test_pipeline.py
├── test_tracer.py
├── test_eval.py
└── test_deepeval_suite.py
```

## Offline principle (mentor note)
- Mock the **boundary to the outside world**: embedding model (`Qwen3Embeddings._get_model`), LLM/chain (`build_llm`/`generator.chain`), and any Chroma persistence you don't want (use `tmp_path` + mocked embeddings, or mock the store).
- Pure logic (splitter, registry, evaluator metrics) is tested **for real**.
- The DeepEval suite mocks the Groq judge + pipeline, exactly like Step 2.

## What each test file covers

### `test_splitter.py`
- `split_text` returns multiple chunks; chunk sizes ≤ `chunk_size`
- `load_and_split` returns `{text, metadata}` with `source` = basename
- `load_directory` aggregates and handles a missing dir (`[]`)
- Setting `separators=[""]` still works (empty-char fallback)

### `test_embeddings.py`
- `embed_documents` returns `list[list[float]]` dim 1024 (via a `FakeModel` returning real numpy)
- `embed_query` uses the **instruction-aware** `query_embed` path (spy)
- Model is a singleton; `EMBED_MODEL` env override (monkeypatch)
- Patch `TextEmbedding` where imported (`langchain_rag.embeddings`)

### `test_vectorstore.py`
- `add_documents` + `count` on a real Chroma `tmp_path` persist_dir with a **mocked embedding** (random vectors)
- `search` returns `{text, metadata, score}`; `min_score` gate drops below-threshold hits
- `clear` empties the collection

### `test_retriever.py`
- `retrieve` delegates to `store.search` (`assert_called_once_with("q", top_k=3, min_score=0.5)`)
- Returns Step 2-shaped dicts

### `test_generator.py`
- `generate` runs the chain and returns `.content`; `refusal_response` string
- Mock `build_chain`/`chain.invoke` (patch where imported)
- No real API call fires

### `test_prompt_registry.py`
- Full lifecycle: register → promote (draft→testing→approved→retired), rollback, evidence gate (`_activate` raises without `accuracy`), `record_eval_scores`, `log_run` increments `run_count`, `compare_versions` sort, crash-safe `save`/`load` via `tmp_path`, immutable no-overwrite guard, `get_status_history`. Aim ~100% coverage.

### `test_pipeline.py`
- `load_documents` idempotent re-open; rebuild on `force_rebuild`
- `ask` returns `{answer, sources, prompt_key, rendered_hash}`; refusal path when no results
- `rendered_hash` deterministic; `retrieved_doc_ids` from `metadata["source"]`
- `get_stats` shape
- Mock splitter/store/retriever/generator/registry (offline orchestration)

### `test_tracer.py`
- `tracing_enabled` for missing/placeholder/real key; `trace_retrieve`/`trace_generate` passthrough; `setup_tracing` sets env only when enabled (100%)

### `test_eval.py`
- All 4 metrics + delegations + `evaluate_single`/`run_full_eval`/`evaluate_with_registry` (same as Step 2, ~30 tests)

### `test_deepeval_suite.py`
- `GroqJudge._coerce_result`, `_retry_delay` (parses Groq "try again in Ns"), `generate`/`a_generate` retry-on-429 / success / exhaustion; `save_results` payload; `load_golden` skips blanks; `build_test_cases` raises without retriever (mock judge + pipeline) — offline

## Test tooling config (`pyproject.toml`)
Add to `pyproject.toml` so tests can import the source + eval packages and measure coverage:
```toml
[tool.pytest.ini_options]
pythonpath = ["src", "eval"]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["langchain_rag", "deepeval_suite"]

[tool.coverage.report]
skip_covered = false
show_missing = true
```
> You can add these with `uv add --dev pytest-asyncio` if `asyncio_mode` triggers a need — but check whether `deepeval_suite` needs it (Step 2 used `asyncio_mode = "auto"`). Add `pytest-asyncio` to the dev group if needed.

## Run Tests
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run pytest tests/ -v
uv run pytest tests/ -v --cov --cov-report=term-missing
uv run ruff check src tests eval app.py
uv run mypy src/langchain_rag tests eval app.py
```

## Completion Criteria
- [ ] All 10 test files created
- [ ] All tests pass **offline** (no LLM, no embed download, no LangSmith)
- [ ] Coverage reported (`.pytest.ini_options` + coverage config)
- [ ] `ruff` + `mypy` clean (strict per pyproject)
- [ ] Test count noted for the Step 2 vs Step 3 comparison

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src tests eval app.py
uv run mypy src/langchain_rag tests eval app.py
uv run pytest tests/ -v
uv run pytest tests/ -v --cov --cov-report=term-missing
```

## Report Back
When done, tell me:
1. Number of tests, passed/failed
2. Coverage % (total + lowest module)
3. `ruff` + `mypy` status
4. Which module was hardest to keep offline (likely vectorstore or generator/chain) — good discussion material for `notes.md`
