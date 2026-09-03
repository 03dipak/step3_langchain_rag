# Task 7: Implement Reranker (Cross-Encoder second stage) ⭐ NEW

## Objective
Insert a **rerank stage** between the Chroma retriever (Task 6) and the generator (Task 8). After candidate retrieval returns `top_k` chunks, a cross-encoder **re-scores and re-sorts** them so the most relevant chunks float to the top before they reach the LLM. This is an **extra stage in Step 3's pipeline only** — Step 2 has no reranker — and we keep the same 4 eval metrics so we can measure **what reranking buys** against Step 2.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/reranker.py`

## ⭐ Architect correction: rerank as a second stage over dicts (read this)
The original plan used LangChain's `ContextualCompressionRetriever` + `CrossEncoderReranker`. **Verified: this does not fit this stack.**
1. The installed `langchain` is the **modern split** (1.3.x) — it has **no `langchain.retrievers`** module at all; `ContextualCompressionRetriever`/`CrossEncoderReranker` live in the classic `langchain` + `langchain_community` packages we'd rather not pull in.
2. `ContextualCompressionRetriever` requires a **`BaseRetriever` yielding `Document`s**, but our `Retriever` (Task 6) returns **Step-2 dicts** `{text, metadata, score}` and the pipeline calls `retriever.retrieve(question, top_k, min_score)`. Two incompatible contracts.

**Decision (mentor/architect):** drop `ContextualCompressionRetriever`. Implement `Reranker` as a **pure second stage** that re-scores the already-retrieved Step-2 dicts with a cross-encoder **directly**, preserving the `{text, metadata, score}` shape end-to-end.

```
Retriever.retrieve(...) → top_k dicts {text, metadata, score}
        ↓
Reranker.rerank(candidates: list[dict], question) → re-sorted {text, metadata, score, rerank_score}
        ↓ (drop to top_n)
pipeline.ask() passes re-ranked dicts to the generator
```

This keeps the whole downstream contract (evaluator, DeepEval, `ask()` shape) untouched. **No new large framework package needed** — just the cross-encoder runtime.

## Dependency to add
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv add sentence-transformers
```
> ⚠️ Do **NOT** add `langchain-community` — it's not needed for this design. `sentence-transformers` pulls `torch` + `transformers` (adds ~1–2 GB RAM residency when loaded). That's the cost of a real cross-encoder. (fastembed's cross-encoder is used by the older side-project; not here.)

## What You Need To Do

### `Reranker` class
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, top_n: int = 3) -> None:
        self.top_n = top_n
        self._model = None           # lazy singleton

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._model

    def rerank(self, question: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        # Build (question, text) pairs; model.predict(pairs) → scores
        # Re-sort candidates by cross-encoder score (desc), cap to top_n
        # Attach "rerank_score" and keep "score" (the original retrieval score)
```

### Step 7.1: lazy singleton model
`_get_model()` loads once (instance attribute), first call downloads `ms-marco-MiniLM-L-6-v2` (~23MB), cached after. **Degrade offline:** if the model can't load (no net + not cached), fall back to returning candidates **unscored/original order** rather than crashing — so offline tests and eval still work (same principle as the embeddings singleton, Task 4).

### Step 7.2: `rerank`
1. `pairs = [(question, c["text"]) for c in candidates]`
2. `scores = self._get_model().predict(pairs)` → `list[float]`
3. Attach `rerank_score = float(s)`, keep original `score`.
4. Sort by `rerank_score` desc, `candidates[:top_n]`, return dicts with `rerank_score` present.

> **Preserve `metadata["source"]` (basename)** — the evaluator/DeepEval rely on it. Don't reshape.

### Step 7.3: Wire into the pipeline (Task 9)
In `pipeline.load_documents()`: after building the `Retriever`, wrap it so `ask()` transparently reranks:
```python
class RerankableRetriever:
    """Over-fetch at 2x, cross-encode rerank, drop to the caller's top_n.

    This is the ONE place the over-fetch factor lives — no per-caller guessing.
    """
    def __init__(self, base, reranker):
        self.base, self.reranker = base, reranker

    def retrieve(self, question, top_k=3, min_score=0.0):
        cands = self.base.retrieve(question, top_k=top_k * 2, min_score=min_score)  # over-fetch 2x
        return self.reranker.rerank(question, cands, top_n=top_k)                   # drop to caller's top_k
```
`pipeline.retriever = RerankableRetriever(Retriever(store), Reranker(top_n=top_k))`. `ask()` interface unchanged.
> **Over-fetch decision (mentor/lead):** the base retriever pulls **`top_k * 2`** candidates, then the cross-encoder re-scores and keeps **`top_n = top_k`** (the caller's requested count). This factor lives **only** inside `RerankableRetriever.retrieve()`, so `search`/`ask`/eval all behave identically and there's no per-caller inconsistency. If the candidate pool is smaller than `top_k`, rerank just returns what it has (reranker caps with `candidates[:top_n]`, naturally bounded).

## Why this is Step-3-specific (mentor note, feeds Task 19)
Reranking is the **first component Step 2 didn't have**, so it's a genuine "what can we add cheaply?" data point: a real cross-encoder in a few lines. Cost: a second model download + torch runtime (slow `ask()`, more RAM) — weigh in the Task 19 gap analysis. Note also: Step 2's side-project used **dense-similarity as a cross-encoder proxy**; here we use a **true cross-encoder** — that's the upgrade this stage brings.

## Completion Criteria
- [ ] `reranker.py` created with `Reranker` (lazy single model, signal `rerank_score`)
- [ ] `rerank()` re-sorts candidates cross-encoder-desc and caps `top_n`
- [ ] Returns `[{text, metadata, score, rerank_score}]`; `metadata["source"]` = basename
- [ ] Offline fallback: no crash when the model can't load (returns candidates unchanged)
- [ ] `RerankableRetriever` wired in `pipeline.retriever`; `ask()` interface unchanged
- [ ] `sentence-transformers` added; **`langchain-community` NOT added**
- [ ] Offline unit tests (mocked cross-encoder) pass + 100% coverage

## Testing
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run python -c "
from langchain_rag.reranker import Reranker
r = Reranker(top_n=2)
out = r.rerank('how does x work?', [{'text':'x works by y','metadata':{'source':'a.txt'},'score':0.9},{'text':'unrelated','metadata':{'source':'b.txt'},'score':0.5}])
for o in out: print(o['rerank_score'], o['metadata']['source'])
"
```

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/reranker.py tests/test_reranker.py
uv run mypy src/langchain_rag/reranker.py tests/test_reranker.py
uv run pytest tests/test_reranker.py -v --cov=langchain_rag.reranker --cov-report=term-missing
uv run ruff check src tests eval app.py
uv run mypy src/langchain_rag tests eval app.py
```

## Report Back
When done, tell me:
1. Candidate pool size vs `top_n` you used
2. A sample re-ranked `(text[:60], metadata["source"], rerank_score)` from one `ask()`
3. How the offline fallback behaved without the model/cache
4. `rerank_score` surfaced in stats
5. Paste your `reranker.py` for review
