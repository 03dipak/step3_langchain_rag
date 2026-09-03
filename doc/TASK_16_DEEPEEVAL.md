# Task 16: DeepEval LLM-Judged Evaluation Suite (Groq judge) — reused from Step 2

## Objective
Add the **same** DeepEval LLM-judged layer as Step 2 (`eval/deepeval_suite/`) — scoring contextual recall, contextual precision, and faithfulness with a **free-tier Groq judge** (`openai/gpt-oss-120b`). Because the DeepEval code and data are identical to Step 2, the scores are directly comparable (framework gap analysis on LLM-judged metrics too).

## Why DeepEval
The keyword evaluator (Task 12) is fast, deterministic, offline — but it's keyword-coverage heuristics. DeepEval's metrics call an **LLM judge** that reasons about the answer, expected answer, and retrieved context — far more accurate, at the cost of API calls.

## Files Added
| File | Purpose |
|------|---------|
| `eval/deepeval_suite/__init__.py` | Package marker |
| `eval/deepeval_suite/judge.py` | `GroqJudge` — `DeepEvalBaseLLM` adapter over `langchain_groq.ChatGroq` |
| `eval/deepeval_suite/evaluate.py` | Builds `LLMTestCase`s from `golden.jsonl`, runs DeepEval metrics |

## Dependencies (already in pyproject from Task 1)
`deepeval==2.9.3`, `langchain-core`, `langchain-groq`.

## `judge.py` — GroqJudge (write fresh, same behavior as Step 2)
A `DeepEvalBaseLLM` subclass wrapping **`langchain_groq.ChatGroq`**:
- **Default model:** `openai/gpt-oss-120b` (free tier — the only freely-accessible model on this Groq account; `llama-3.3-70b-versatile` returns 404).
- **Configurable** via `JUDGE_MODEL` env or `model` arg.
- **Rate-limit resilience:** free Groq tier is TPM-limited (~8000 tokens/min) and DeepEval fires metric calls in parallel. `GroqJudge` (a) throttles concurrent judge calls through a shared asyncio semaphore (`JUDGE_MAX_CONCURRENT`, default **1** = serialized so TPM refills) and (b) retries `RateLimitError` up to `MAX_RETRIES=10`, sleeping the **exact retry-after seconds parsed from the error message** (e.g. "try again in 6.975s").
- **Schema-aware output:** DeepEval 2.x passes a pydantic `schema` to `generate`/`a_generate`. The judge parses the model's JSON into it (`model_validate_json`), stripping fenced ```json``` blocks. Without this, DeepEval errors with `AttributeError: 'str' object has no attribute 'claims'`.

## `evaluate.py` — Runner (implements the same interface as Step 2)
Builds one `LLMTestCase` per golden entry:
- `input` = `entry["question"]`
- `actual_output` = pipeline-generated answer (`pipeline.generator.generate`)
- `expected_output` = gold `answer`
- `retrieval_context` = retrieved chunk texts (`pipeline.retriever.retrieve`, `top_k=k`)

Metrics: `ContextualRecallMetric`, `ContextualPrecisionMetric`, `FaithfulnessMetric`, `threshold=0.7`, `include_reason=...`.

CLI:
```bash
uv run python eval/deepeval_suite/evaluate.py --k 3 --limit 2 --no-reason
```
Flags: `--k` (chunks/query), `--limit N` (first N entries — free tier can't fit 20×3 in one window; use ~4), `--no-reason` (skip reasoning strings, ~halves tokens). Persists to `eval/results/deepeval_<timestamp>.json` (engine, judge, config, per-metric pass rate).

> **Reuse insight:** this `judge.py`/`evaluate.py` is conceptually the same as Step 2's. You can write it fresh or adapt — the point is the **interface** (`GroqJudge`, `evaluate.py` entrypoint) stays the same so Step 2 vs Step 3 DeepEval scores line up.

## Usage from Streamlit
Eval Dashboard → "DeepEval (LLM-judged via Groq)" expander. Because Streamlit runs in a non-main thread and DeepEval's `TraceManager` calls `signal.signal()` at import (main-thread only), the app launches `evaluate.py` as a **subprocess** and renders captured stdout + the persisted summary gauges — never imports the suite inline (same as Step 2).

## Verifying
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check eval/deepeval_suite/ app.py
uv run mypy eval/deepeval_suite/
uv run python eval/deepeval_suite/evaluate.py --k 3 --limit 1 --no-reason   # live smoke test
uv run pytest tests/test_deepeval_suite.py -v                              # offline unit tests
```

## Completion Criteria
- [ ] `eval/deepeval_suite/` created with `judge.py` + `evaluate.py`
- [ ] `GroqJudge` parses 429 retry-after and throttles concurrent calls
- [ ] `evaluate.py` builds `LLMTestCase`s from `golden.jsonl` and runs the 3 metrics
- [ ] Results persisted to `eval/results/deepeval_*.json`
- [ ] Offline unit tests pass (judge + evaluate logic, mocked Groq)
- [ ] Keyword evaluator (Task 12) still passes its tests

## Report Back
When done, tell me:
1. The 3 DeepEval scores for one case (Contextual Recall / Precision / Faithfulness)
2. The judge model name shown in the run
3. Whether the offline keyword evaluator still passes
4. Paste `judge.py` + `evaluate.py` for review
