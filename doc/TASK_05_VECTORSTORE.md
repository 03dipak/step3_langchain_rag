# Task 5: Implement Vector Store (LangChain `Chroma`)

## Objective
Create `src/langchain_rag/vectorstore.py` — a `Chroma`-backed vector store that **replaces** Step 2's hand-written numpy cosine store. It persists embeddings to disk and exposes a retriever, but we wrap it so results come back in the **same `{text, metadata, score}` dict shape** Step 2 produced (the rest of the pipeline/evaluator depends on that shape).

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/vectorstore.py`

## The comparison (Step 2 numpy vs Step 3 Chroma)
Step 2's store was ~113 lines of hand-written cosine similarity, `np.vstack`, `.save`/`.load` to `.npz`. Chroma does all of it:
```python
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="rag_docs",
    embedding_function=my_embeddings,      # our Qwen3Embeddings (Task 4)
    persist_directory="chroma_db",
)
vectorstore.add_documents(docs)
hits = vectorstore.similarity_search(query, k=3)   # returns Documents
```
- **Gain:** on-disk persistence, built-in similarity (cosine default), metadata filtering, no manual vector math.
- **Lose:** you can no longer "see" the raw similarity scores without reaching into `vectorstore._collection.query(...)` for distances.

## What You Need To Do

### Class: `LangChainStore`
```python
class LangChainStore:
    def __init__(self, persist_dir="chroma_db", embedding=None, collection_name="rag_docs") -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = embedding or Qwen3Embeddings()
        self._db = None   # Chroma handle (lazy)

    def _db_instance(self) -> Chroma: ...          # lazy-create/reuse Chroma
    def add_documents(self, chunk_dicts) -> None:  # [{"text","metadata"}] -> Chroma
    def count(self) -> int: ...
    def search(self, query, top_k=3, min_score=0.0) -> list[dict]: ...
    def clear(self) -> None: ...
```

### Step 5.1: `add_documents`
Convert our `[{text, metadata}]` dicts into LangChain `Document(page_content=..., metadata=...)` objects and pass to `vectorstore.add_documents(docs)`.

### Step 5.2: `search` — reach into Chroma for scores ⭐
`similarity_search` hides the score. To keep the Step 2 output shape (`score` present) you use the underlying collection:
```python
query_vec = self.embeddings.embed_query(query)
collection = self._db_instance()._collection
hits = collection.query(
    query_embeddings=[query_vec],
    n_results=top_k,
    include=["documents", "metadatas", "distances"],
)
# documents[0], metadatas[0], distances[0]; similarity = 1.0 - distance
```
Build `[{"text":..., "metadata":..., "score": similarity}]`, **filtering by `min_score`** and capping `top_k`. Break out of the loop on the first below-threshold hit.

> ⭐ **Framework lesson:** this is where LangChain's abstraction "hides mechanics." Reaching into `_collection` to recover scores is the price of keeping our `min_score` gate + eval shape. Note this tradeoff in `notes.md` — it's exactly the kind of place where the framework's convenience and your need for transparency collide.

### Step 5.3: Persistence
`persist_directory="chroma_db"` gives on-disk persistence for free. `clear()` deletes the collection (used for rebuilds).

## Testing
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run python -c "
from langchain_rag.vectorstore import LangChainStore
s = LangChainStore(persist_dir='/tmp/lc_chroma_test')
s.clear()
s.add_documents([{'text':'Python is high level','metadata':{'source':'python_basics.txt'}},
                 {'text':'Gradient descent optimizes','metadata':{'source':'machine_learning.txt'}}])
print('count:', s.count())
res = s.search('python', top_k=2, min_score=0.0)
print('n:', len(res), 'keys:', sorted(res[0].keys()))
"
```
**Expected:** n=2, each with `text`, `metadata`, `score`.

## Completion Criteria
- [ ] `vectorstore.py` created with `LangChainStore`
- [ ] `add_documents()` stores chunk dicts
- [ ] `search()` returns `[{text, metadata, score}]` (Step 2 shape) with `min_score` gate
- [ ] `count()` works
- [ ] Persists to `chroma_db` (reopens without re-adding)
- [ ] `clear()` empties the collection

## Mocking (learner note)
Chroma is a real on-disk vector DB. For offline unit tests, either (a) use an in-memory/temp `persist_directory` (`tmp_path`) **without** embedding (mock `self.embeddings.embed_query` to return a random vector), or (b) mock the whole store boundary where the pipeline uses it. Prefer real Chroma on `tmp_path` where feasible — it's fast and needs no network.

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/vectorstore.py
uv run mypy src/langchain_rag/vectorstore.py
uv run pytest tests/test_vectorstore.py -v
uv run pytest tests/test_vectorstore.py -v --cov=langchain_rag.vectorstore --cov-report=term-missing
uv run ruff check tests/test_vectorstore.py
uv run mypy tests/test_vectorstore.py
```

## Report Back
When done, tell me:
1. `count()` output
2. `search()` top-k and score values (confirm `score` in each dict)
3. Whether reopening from `chroma_db` (no re-add) worked
4. The `_collection` call — what `distances` → similarity formula did you use? This is a key framework-internals spot to have reviewed
5. Paste your `vectorstore.py` for review
