# Task 20: Multi-Level LLM Evaluation Suite (L1/L2/L3) ⭐ NEW

## Objective
Port the CampusX `rag-eval-deepeval` reference eval suite into this repo's **GroqJudge** architecture, evaluating the Step 3 RAG stack at **three levels** with per-component golden files on real `qwen/qwen2.5-7b`-driven pipelines (judging via free-tier Groq). This directly closes gaps **A–F** of `doc/GAP_ANALYSIS_EVAL_FRAMEWORK.md`. Regression/**CI-CD**/MLflow evals are **explicitly deferred to the next step**.

Build:
- **L1 — Component:** `eval_retriever.py` (Contextual Recall + Precision), `eval_generator.py` (Faithfulness + Answer Relevancy)
- **L2 — Pipeline (RAG Triad):** `eval_rag_pipeline.py` (Contextual Relevancy + Faithfulness + Answer Relevancy), live `pipeline.ask()`
- **L3 — Application + Ops:** `eval_application.py` (GEval correctness/completeness/style), `eval_latency.py` (stage-split SLOs), `eval_cost.py` (usage_metadata cost projection)
- A shared `evals/harness.py` (golden loading + result summarization)

> The reference project lives at `/home/dipak/agentic/rag-eval-deepeval-main` (goldens, `evals/run_suite.py`, `metric_registry.py`, safety/reliability/ops evals). Only the component/pipeline/application/ops layers are ported here; safety/reliability/reranker suites land in the next step.

## Key Architectural Decisions (approved)
1. **Reuse `GroqJudge()`** for every metric (`from deepeval_suite.judge import GroqJudge`); DeepEval metrics take the judge **instance** (`model=GroqJudge()`), never a string.
2. **Per-component golden files** from the existing `golden.jsonl` (20 queries), matching reference schemas:
   - `retriever_goldens.json` → `[{id, query, ideal_answer}]`
   - `faithfulness_dataset.json` → `[{id, query, ideal_context, source}]` (`ideal_context` = `list[str]`)
   - `correctness_goldens.json` → `[{id, question, ideal_answer}]`
3. **Generator context contract:** `LangChainGenerator.generate()` takes `context_chunks: list[dict]`, so golden `ideal_context` strings are wrapped as `[{"text": s}]`.
4. **Cost eval reads `usage_metadata` directly off `pipeline.generator.chain.invoke(...)`** (the chain ends at `| llm`, no `StrOutputParser` → returns an `AIMessage` with `usage_metadata`). No shim in `generate()`. Generator `generate()` is **untouched**.
5. **L3 GEval** metrics are reference-free where possible: `_build_style()` uses `[INPUT, ACTUAL_OUTPUT]`; correctness/completeness compare against `expected_output`; `Rubric`, `strict_mode=False`, threshold `0.7`.

## Files Created
```
evals/
├── __init__.py
├── harness.py            # load_goldens, summarize_by_metric, print_summary (defensive across .test_results/.metrics_data/.metrics)
├── eval_retriever.py     # L1: Contextual Recall + Precision on pipeline.retriever.retrieve()
├── eval_generator.py     # L1: Faithfulness + Answer Relevancy via generator.generate()
├── eval_rag_pipeline.py  # L2: RAG Triad via pipeline.ask()
├── eval_application.py   # L3: GEval correctness/completeness/style
├── eval_latency.py       # L3: retrieval vs generation stage latency, P95 SLO
└── eval_cost.py          # L3: usage_metadata cost projection + budget verdict
eval/goldens/
├── retriever_goldens.json        (20)
├── faithfulness_dataset.json     (20)
└── correctness_goldens.json      (20)
```

## Latent bug fixed in `src` (required to run any live suite)
`Pipeline()`/`LangChainGenerator()` eagerly call `build_chain()` → `registry.get(approved_only=True)` against an **empty** `PromptRegistry()`, which **failed at construction** ("No approved version found"). The existing `eval/deepeval_suite/evaluate.py` was broken by this too.
**Fix:** in `LangChainGenerator.__init__`, when no registry is passed, `self.registry.load()` the default seed before building the chain (`src/langchain_rag/generator.py:24-31`). Bare `Pipeline()` now constructs with the approved seed (then requires the live LLM key at model build / run time). Callers that pass an explicit registry are untouched.

## Completion Criteria
- [ ] All 7 eval scripts + `harness.py` written; `eval/goldens/*.json` generated (20 each)
- [ ] L1, L2, L3 coverage per the A–F roadmap (regression/CI-CD explicitly deferred)
- [ ] `ruff check evals/` + `mypy evals/` clean
- [ ] Latent `Pipeline()` construction bug fixed with full regression green
- [ ] Scripts run via `python -m evals.eval_xxx`; `sys.path.insert(0, "src")` where needed
- [ ] Deterministic offline paths smoke-tested (goldens load, retrieval runs, chain shape verified)

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check evals/ src/langchain_rag/generator.py
uv run mypy evals/
uv run pytest -q -p no:cacheprovider          # full unit suite, no regression
uv run python -m evals.eval_retriever --limit 1 --no-reason      # live Groq (needs .env keys)
```

## Report Back
1. Central path each level maps to (retriever / generator / pipeline.ask / app / ops)
2. The LLM-judge scores your suite produced (L1 recall/precision, generator faithfulness/relevance, L2 RAG Triad, L3 GEval + latency P95 + cost)
3. Which gaps A–F are now closed vs deferred to next step
4. One concrete finding from the latent `Pipeline()` bug you fixed
