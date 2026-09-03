# Task 4: Implement Embeddings (LangChain `Embeddings` Adapter over qwen3-embed)

## Objective
Create `src/langchain_rag/embeddings.py` — a custom LangChain `Embeddings` adapter backed by **the exact same `qwen3-embed` model as Step 2** (Qwen3-Embedding-0.6B, 1024-dim). This is the first real "adapter" moment: LangChain owns the contract (`embed_documents` / `embed_query`), qwen3-embed owns the math.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/embeddings.py`

## Why NOT `HuggingFaceEmbeddings`
Step 2's README proposed `HuggingFaceEmbeddings(model_name="bge-base-en-v1.5")`, but we decided to keep the **same qwen3 embed as Step 2** so the A/B is apples-to-apples. Instead of cramming qwen3 into a HF wrapper, we implement LangChain's **`Embeddings` interface** directly — that's the proper abstraction boundary:
```python
from langchain_core.embeddings import Embeddings

class Qwen3Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```
Chroma (Task 5) accepts any object implementing `Embeddings` — it calls `embed_documents` for indexing and `embed_query` for the query at search time.

## What You Need To Do

### Class: `Qwen3Embeddings(Embeddings)`
Mirror Step 2's `Embedder` semantics but under the LangChain interface:
```python
import os
from langchain_core.embeddings import Embeddings

class Qwen3Embeddings(Embeddings):
    _model = None  # singleton — load once, reuse

    def _get_model(self):
        if Qwen3Embeddings._model is None:
            from qwen3_embed import TextEmbedding
            Qwen3Embeddings._model = TextEmbedding(model_name=self._model_name())
        return Qwen3Embeddings._model

    @staticmethod
    def _model_name() -> str:
        return os.getenv("EMBED_MODEL", "n24q02m/Qwen3-Embedding-0.6B-ONNX")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # model.embed(texts); convert each to list[float]

    def embed_query(self, text: str) -> list[float]:
        # model.query_embed(text)  ← instruction-aware, first result as list[float]
```

### Step 4.1: `embed_documents`
Call `model.embed(texts)` → iterable of vectors → `[[float(x) for x in vec] for vec in ...]`. The float conversion keeps strict mypy happy (matches Step 2).

### Step 4.2: `embed_query` — instruction-aware ⭐
Use `model.query_embed(text)` (the Qwen3 query-instruction path) and take the first vector as `list[float]`. Documents embed with `embed()`, queries with `query_embed()`; skipping the query prefix costs ~1–5% retrieval quality (the "misuse the query prefix" bug class from Step 1).

### Step 4.3: singleton
The model loads **once** (class attribute `_model`) — the Chroma index and every query reuse it. Model name configurable via `EMBED_MODEL`.

## Testing
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run python -c "
from langchain_rag.embeddings import Qwen3Embeddings
e = Qwen3Embeddings()
print('docs:', len(e.embed_documents(['What is Python?','How does ML work?'])), 'dim:', len(e.embed_documents(['x'])[0]))
print('query dim:', len(e.embed_query('What is gradient descent?')))
"
```
**Expected:** 2×1024 + 1×1024 (first run downloads the ONNX model).

## Completion Criteria
- [ ] `embeddings.py` created with `Qwen3Embeddings(Embeddings)`
- [ ] `embed_documents()` returns `list[list[float]]`, dim 1024
- [ ] `embed_query()` uses `query_embed` (instruction-aware), returns `list[float]`
- [ ] Model is a singleton
- [ ] `ruff` + `mypy --strict` pass
- [ ] Chroma (Task 5) accepts this object as its `embedding_function` (interface fits)

## Mocking (learner note) — mock the model boundary ⭐
`qwen3_embed` is an ONNX model (possible download) = **external boundary** → mock in tests to stay offline:
- `mocker.patch.object(Qwen3Embeddings, "_get_model", return_value=FakeModel())`
- `FakeModel` = hand-written fake returning real numpy arrays (keeps the `[float(x) for x in ...]` conversion honest)
- patch `TextEmbedding` **where it's imported** (`langchain_rag.embeddings`), not `qwen3_embed.TextEmbedding`
- `monkeypatch.setenv("EMBED_MODEL", ...)` to test env override

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/embeddings.py
uv run mypy src/langchain_rag/embeddings.py
uv run pytest tests/test_embeddings.py -v
uv run pytest tests/test_embeddings.py -v --cov=langchain_rag.embeddings --cov-report=term-missing
uv run ruff check tests/test_embeddings.py
uv run mypy tests/test_embeddings.py
```

## Report Back
When done, tell me:
1. Embedding dimension
2. Whether the model loaded once (no re-download on 2nd call)
3. Paste your `embeddings.py` for review — specifically confirm it satisfies LangChain's `Embeddings` interface (both methods, correct types)
