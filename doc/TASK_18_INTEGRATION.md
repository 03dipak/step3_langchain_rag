# Task 18: Integration Tests (real Chroma + real embedding) ⭐ NEW

## Objective
Add an **integration suite** that exercises the full Step 3 stack **for real** — real `Qwen3Embeddings`, a real on-disk (or `tmp_path`) Chroma store, real splitter, real reranker/chain where feasible — end-to-end, marked `integration` so it is **excluded from the default unit-test run** (no model download / no slow loads in CI or everyday `pytest`).

## File to Create
`/home/dipak/agentic/step3_langchain_rag/tests/integration/test_index_integration.py`

## The pattern (mirrors the well-reviewed Side Project integration test)

### 1. Mark it
```python
import pytest
pytestmark = pytest.mark.integration
```
- Configure in `pyproject.toml` so integration is **not** run by default (same as the Side Project):
```toml
[tool.pytest.ini_options]
addopts = "-q -m 'not integration'"
markers = [
    "integration: real ONNX/embedding + Chroma roundtrip (run with -m integration)",
]
```
> ⭐ Existing unit tests must **not** require the `integration` marker to be absent-but-present; the `addopts` deselects it. Run them explicitly with `uv run pytest -m integration`.

### 2. Load the embedding model ONCE (avoid the 3×-load hang) ⭐
This was the single most important fix on the Side Project. `Qwen3Embeddings` loads the ONNX model **per instance** (`_ensure_loaded` caches on the instance, not the process). So share **one module-scoped embedder fixture**:
```python
@pytest.fixture(scope="module")
def embedder() -> Qwen3Embeddings:
    return Qwen3Embeddings()          # bge/qwen ONNX loaded exactly once for all tests
```
Every test that needs embeddings takes `embedder`, never constructs its own. The store fixture is also module-scoped over a `tmp_path` Chroma `persist_directory` (or in-memory) so it too is created once.

### 3. What it covers (real, offline after first model download)
- `test_embedder_dense_dim`: embed one probe → `len(vec) == 1024` (qwen3-embed), not a literal that silently drifts.
- `test_end_to_end_embed_upsert_search`: real embed → `store.add_documents` → `store.count()` → `search` returns the right chunk with a `{text, metadata, score}` dict, `score` ≈ 1.0 for a near-identical re-embed.
- `test_pipeline_load_idempotent`: `load_documents()` on a tmp corpus, call again → no duplicate chunks; `force_rebuild=True` re-indexes.
- `test_pipeline_ask_real`: run `ask()` end-to-end — **but stub the LLM/chain** (mock `generator.generate`) so no API key is needed; assert the `{answer, sources, prompt_key, rendered_hash}` shape and that `sources` are real, `metadata["source"]` is a basename. (This keeps integration fast & hermetic while still using real embeddings + Chroma.)

## Why a shared embedder fixture is non-negotiable (mentor note)
Loading `bge`/`qwen`-size ONNX models once per test = 3× redundant RAM + load latency in one run — the exact "hangs my system" failure you flagged. Module-scoped fixture = **1 load, all tests share it**. Carry this pattern forward anywhere real models are used in tests.

## Completion Criteria
- [ ] `tests/integration/test_index_integration.py` with `pytestmark = integration`
- [ ] `[tool.pytest.ini_options]` with `addopts = "-m 'not integration'"` + marker registered
- [ ] **One module-scoped `embedder` fixture** shared by all tests (no per-test model loads)
- [ ] Real embeddings + real Chroma; `ask()` stubs only the LLM chain
- [ ] Default `uv run pytest` runs **unit** tests only (integration deselected)
- [ ] `uv run pytest -m integration` passes offline (embeddings cached after first download)
- [ ] `ruff` + `mypy` clean

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run pytest tests/ -v                     # unit only (integration deselected)
uv run pytest tests/integration -m integration -v   # real roundtrip (downloads model once)
uv run ruff check tests/integration/
uv run mypy tests/integration/
```

## Report Back
When done, tell me:
1. Confirm default `uv run pytest` deselects integration (count: unit only)
2. `uv run pytest -m integration` result (3–4 tests pass)
3. Embedding dim asserted (should be 1024 for qwen3-embed)
4. Confirmed the embedder loads **once** (log shows a single model-load line for the module)
5. Paste your integration test for review
