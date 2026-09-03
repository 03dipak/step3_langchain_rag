# Task 12: Implement Evaluator (4 Metrics) — same as Step 2

## Objective
Create `eval/evaluator.py` — the **exact same 4 keyword metrics** and eval helpers as Step 2. Because we kept the documents, embedding model, golden dataset, and `ask()` interface identical, these produce directly comparable scores.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/eval/evaluator.py`

## The Four Metrics (unchanged from Step 2)
| Metric | Question it answers | Target |
|--------|--------------------|--------|
| Context Recall | Did we retrieve the right chunks (coverage of the gold answer)? | ≥ 0.70 |
| Context Precision | Of the retrieved chunks, how many are actually relevant? | ≥ 0.60 |
| Faithfulness | Are the answer's claims supported by the context? | ≥ 0.80 |
| Answer Relevance | Does the answer address the question? | ≥ 0.70 |

## What You Need To Do (write fresh, same behavior as Step 2)
Functions:
- `extract_keywords(text) -> list[str]` — lowercase, drop stopwords/short/non-alpha tokens
- `context_recall(gold_answer, retrieved_chunks) -> float` — keyword coverage of gold in retrieved context
- `context_precision(question, retrieved_chunks, gold_source) -> float` — fraction of chunks whose `metadata["source"]` == gold_source
- `faithful_fallback(answer, context_chunks) -> float` — offline keyword grounding
- `faithfulness(answer, context_chunks) -> float` — wraps fallback
- `relevance_fallback(question, answer) -> float` — keyword overlap
- `answer_relevance(question, answer) -> float` — wraps fallback
- `evaluate_single(question, gold_answer, gold_source, pipeline, **kwargs) -> dict`
- `run_full_eval(pipeline, golden_path, **kwargs) -> dict` — `{results, summary, timestamp}`
- `evaluate_with_registry(pipeline, golden_path, registry, prompt_key, **kwargs) -> dict` — adds `"accuracy" = context_recall` and calls `registry.record_eval_scores(...)`

> `metadata["source"]` must be the basename (you enforced this in the splitter, Task 3) or `context_precision` will mismatch the golden `source` — the classic "detail that breaks eval" trap.

## Why same metrics (mentor note)
The whole point of Step 3 is a **framework gap analysis on identical metrics**. Same data in → same metric functions → any difference in scores is LangChain's splitter/retriever/chain vs the hand-written ones. Do **not** change the metric definitions here.

## Testing (offline, no LLM/embedding)
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run pytest tests/test_eval.py -v
uv run pytest tests/test_eval.py -v --cov=evaluator --cov-report=term-missing
```
Mirror Step 2's `test_eval.py` coverage (~30 tests): extract_keywords, context_recall, context_precision, faithful_fallback, faithfulness delegation, relevance_fallback, answer_relevance delegation, evaluate_single (pipeline.ask called once / empty sources), run_full_eval (averages / empty / preserves results), evaluate_with_registry (records accuracy).

## Completion Criteria
- [ ] `evaluator.py` created with all metric + helper functions
- [ ] `evaluate_single` returns dict with 4 metrics + answer + sources + prompt_key
- [ ] `run_full_eval` averages; `evaluate_with_registry` records scores (incl. `accuracy`)
- [ ] Metrics return float in [0, 1]
- [ ] 100% coverage offline (`--cov=evaluator`)

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check eval/evaluator.py tests/test_eval.py
uv run mypy eval/evaluator.py tests/test_eval.py
uv run pytest tests/test_eval.py -v --cov=evaluator --cov-report=term-missing
uv run pytest tests/ -v
```

## Report Back
When done, tell me:
1. An example `context_recall` value
2. An example `context_precision` value
3. Confirmation offline tests run without LLM/embedding
4. Paste your `evaluator.py` for review
