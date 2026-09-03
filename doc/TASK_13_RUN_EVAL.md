# Task 13: Run the Evaluation Suite

## Objective
Run the full evaluation over `golden.jsonl`, record scores to the shared registry, and save timestamped results to `eval/results/` — verifying the **Step 2 vs Step 3 comparison** on identical metrics.

## How it runs (same as Step 2)
- **Keyword metrics:** from the Streamlit Eval Dashboard "Run Evaluation" button, which calls `evaluate_with_registry(pipeline, "eval/golden.jsonl", registry, prompt_key)` and saves to `eval/results/eval_<timestamp>.json`.
- **DeepEval (LLM-judged):** `uv run python eval/deepeval_suite/evaluate.py --k 3 --limit N --no-reason` (Task 16).

## The A/B read (mentor note)
Run the same golden set on **Step 2** and **Step 3**, then compare the four averages:
```
Step 2 (hand-written):   recall 0.64 | precision 0.70 | faithfulness 0.84 | relevance 0.78
Step 3 (LangChain):      recall 0.70 | precision 0.66 | faithfulness 0.86 | relevance 0.79
```
- Roughly equal scores → LangChain's `RecursiveCharacterTextSplitter`/Chroma behave like your hand-written code for this corpus. 
- A large recall gap (LangChain higher/lower) → the splitter's separator strategy retrieves different chunk boundaries than your word-count chunker; investigate in Task 19.
- A precision gap → Chroma's default cosine over your qwen embeddings vs your numpy cosine. Expect them close (both cosine), but small diffs are normal.

Thus the saved `eval/results/*.json` files are the raw material for `TASK_19_COMPARISON.md`.

## Entry Points (current)
- App → Eval Dashboard → "Run Evaluation" (keyword, saves `eval_*.json`)
- DeepEval → `eval/deepeval_suite/evaluate.py` (Groq judge, saves `deepeval_*.json`)

## Quick manual run of the keyword evaluator
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run python -c "
from langchain_rag.pipeline import Pipeline
from langchain_rag.prompt_registry import PromptRegistry
from evaluator import run_full_eval

p = Pipeline(); p.load_documents()
r = p.registry
res = run_full_eval(p, 'eval/golden.jsonl')
import json
print(json.dumps(res['summary'], indent=2))
"
```

## Completion Criteria
- [ ] Evaluation runs over `golden.jsonl` (keyword via app or the snippet above)
- [ ] Records scores to the shared registry (with `accuracy`)
- [ ] Saves timestamped JSON to `eval/results/`
- [ ] DeepEval runner works (Task 16) — optional here
- [ ] Captures a Step 2 vs Step 3 score table for `TASK_19_COMPARISON.md`

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check eval/deepeval_suite/ app.py
uv run mypy eval/deepeval_suite/
uv run pytest -v
```

## Report Back
When done, tell me:
1. The four summary scores (Step 3)
2. Path of the saved results file
3. Any metric below target — and, if you have Step 2's numbers handy, the Step 2 vs Step 3 delta
4. Any issues encountered
