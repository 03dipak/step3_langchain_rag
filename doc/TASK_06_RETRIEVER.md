# Task 6: Implement Retriever (Chroma retriever, Step 2-shaped results)

## Objective
Create `src/langchain_rag/retriever.py` — takes a query string and returns top-k relevant chunks from the `LangChainStore` (Chroma), in the **same Step 2 shape** so the evaluator and DeepEval suite work unchanged.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/retriever.py`

## The comparison (Step 2 vs Step 3)
Step 2's `Retriever` embedded the query by hand then searched the numpy store. Here the retriever delegates to Chroma via our `LangChainStore.search(...)`, which already returns `{text, metadata, score}`:

```python
class Retriever:
    def __init__(self, store: LangChainStore) -> None:
        self.store = store

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[dict]:
        return self.store.search(query, top_k=top_k, min_score=min_score)
```

> ⭐ **Framework note:** LangChain offers `vectorstore.as_retriever()` which returns a `BaseRetriever` with `.invoke()`. We deliberately wrap our own `Retriever` returning Step 2-shaped dicts instead, because:
> - `as_retriever().invoke()` yields `Document`s (no `score`), which would break our `min_score` gate and the evaluator's `metadata["source"]` access.
> - Keeping our own thin retriever preserves the **exact contract** Step 2's eval/DeepEval code expects.
> This is a deliberate, documented deviation — note it in `notes.md`. (`as_retriever()` is still available on the store if you want to inspect it.)

## What You Need To Do

### Step 6.1: Implement `Retriever`
A thin, typed class wrapping `LangChainStore`. Constructor takes the store; `retrieve(query, top_k, min_score)` calls `store.search(...)` and returns the result.

### Step 6.2: Same vector space (mentor note)
The retriever must use the **same embedding model** through the store's `Qwen3Embeddings`. Docs were indexed with `embed_documents`; the search embeds the query with `embed_query` (instruction-aware) inside `LangChainStore.search`. Same model + same query-embed path = results land in the same vector space. Don't embed separately in the retriever — reuse `store.search`.

## Completion Criteria
- [ ] `retriever.py` created with `Retriever`
- [ ] `retrieve()` returns Step 2-shaped dicts `[{text, metadata, score}]`
- [ ] Reuses `store.search` (same embedding path as indexing)
- [ ] `top_k` + `min_score` honored

## Mocking (learner note) — mock the store boundary ⭐
`Retriever` is pure orchestration (query → `store.search`). The store (Chroma + embedding) is the heavy/external boundary, so tests mock it:
```python
store = mocker.MagicMock()
store.search.return_value = [{"text":"c","metadata":{"source":"s.txt"},"score":0.9}]
res = Retriever(store).retrieve("q", top_k=3, min_score=0.5)
store.search.assert_called_once_with("q", top_k=3, min_score=0.5)
```
This mirrors Step 2's "patch where imported, use return_value chain" rule — except here we mock the **store object** we inject, not a module-level name.

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/retriever.py
uv run mypy src/langchain_rag/retriever.py
uv run pytest tests/test_retriever.py -v
uv run pytest tests/test_retriever.py -v --cov=langchain_rag.retriever --cov-report=term-missing
uv run ruff check tests/test_retriever.py
uv run mypy tests/test_retriever.py
```

## Report Back
When done, tell me:
1. Number of results returned for a sample query
2. Keys present in each result
3. Did you try `store._db.as_retriever().invoke(...)` and note what shape it returns? (framework-internals lesson)
4. Paste your `retriever.py` for review
