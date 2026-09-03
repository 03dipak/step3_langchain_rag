# Gap Analysis: "Proving It Works in Production" — 3-Level Eval Framework vs. What This Repo Actually Ships

## Objective
Take the marketing post's claims (3-Level DeepEval evaluation framework + MLflow regression testing + CI/CD gates + online drift observability) and do a **Business-Analyst-style gap analysis** against the *actual*, *verified* state of `step3_langchain_rag`. This is not a task to implement — it is an **honest scorecard** of what is true today versus what is aspirational, so we know exactly what remains to be built before "production reliability" is real.

> **Mentor note:** Building a RAG app is easy; proving it works in production is the hard part. This document exists to make the *"proving it"* layer explicit — the post is strong framing, but most of the production stack is **not built in this repo yet**. Filling that gap is exactly the 3-level + regression + observability work ahead.

## The Post's Claims (the "as-advertised" contract)

> 🔹 **Level 1 — Component-Level:** Retriever (Context Precision & Recall), Generator (Faithfulness & Citation Accuracy)
> 🔹 **Level 2 — Pipeline-Level (RAG Triad):** Context Relevance ↔ Faithfulness ↔ Answer Relevance
> 🔹 **Level 3 — System & Ops-Level:** Correctness, Completeness, Latency, Cost & PII/Safety
> Plus: **Regression Testing with MLflow + CI/CD gates** to compare iterations against a baseline, and **Online Observability** for live drift monitoring.

## The Repo's Actual State (verified against source)

| Post claim | Repo reality today | Evidence (file:line) | Gap |
|---|---|---|---|
| L1 retriever: Context Precision & Recall | ✅ Exists, but **pipeline-level** (retriever+generator fused), **not isolated** | `eval/deepeval_suite/evaluate.py:109-111` runs all 3 on the same `LLMTestCase` | 🟡 not component-isolated |
| L1 generator: Faithfulness | ✅ Present | `FaithfulnessMetric` in `evaluate.py:111`; keyword `faithfulness` in `eval/evaluator.py:62` | ✅ |
| L1 generator: **Citation Accuracy** | ❌ **Not present** — only a prompt-`citation_policy`, no targeted metric | `prompts.py` schema rules | 🔴 missing |
| L2 Triad: Context Relevance / Faithfulness | ✅ Present (as Recall/Precision + Faithfulness) | `evaluate.py:109-111` | ✅ |
| L2 Triad: **Answer Relevance** | 🟡 **Keyword only** (`answer_relevance`), **no DeepEval `AnswerRelevancyMetric`** | `eval/evaluator.py:76-78`; metric list in `evaluate.py:108-112` | 🔴 missing DeepEval metric |
| L3 Correctness | ⚠️ **Alias, not real** — `accuracy = context_recall` | `eval/evaluator.py:146` | 🔴 misleading branding |
| L3 Completeness | ❌ Not covered | — | 🔴 missing |
| L3 Latency | ⚠️ Field **exists** in registry (`latency_ms`) but **DeepEval run doesn't measure it** | registry `log_run` / `eval_scores` | 🟡 unmeasured |
| L3 Cost | ❌ Not covered (no token-cost aggregation) | — | 🔴 missing |
| L3 PII/Safety | ❌ Not covered (**metric-layer gap**; prompt `safety_policy` kept flat by earlier ruling) | — | 🔴 missing |
| Regression testing (MLflow + baseline) | ❌ **No MLflow, no baseline, no regression harness** | no `.github/workflows/`, no mlflow anywhere | 🔴 missing |
| CI/CD gates "deploy only if score ≥ baseline" | ❌ **No CI/CD at all** | no `.github` | 🔴 missing |
| Online observability / drift | 🟡 **LangSmith tracing only** (opt-in spans on retrieve/generate), **no live eval or drift monitor** | `src/langchain_rag/tracer.py` | 🟡 partial |
| Reranker (the Step 3 novelty) | ✅ Present — over-fetch ×2 + cross-encoder rerank | `src/langchain_rag/reranker.py` + `RerankableRetriever` in `pipeline.py:25-43` | ✅ (the one real addition) |

## Summary Scorecard

| Layer | Metric | Status |
|---|---|---|
| L1 | Retriever Context Precision / Recall | 🟡 pipeline-level, not isolated |
| L1 | Generator Faithfulness | ✅ |
| L1 | Generator Citation Accuracy | 🔴 |
| L2 | Answer Relevance (DeepEval) | 🔴 |
| L2 | Triad completeness | 🟡 2 of 3 |
| L3 | Correctness | 🔴 (aliased) |
| L3 | Completeness | 🔴 |
| L3 | Latency | 🟡 |
| L3 | Cost | 🔴 |
| L3 | PII/Safety | 🔴 |
| — | Regression baseline (MLflow) | 🔴 |
| — | CI/CD gates | 🔴 |
| — | Online drift observability | 🟡 |

**Verdict from the BA desk:** the post reads as a complete production story, but **the verified repo covers ~½ of Level 1, ⅔ of Level 2, and almost none of Level 3 + regression/CI-CD/observability.** The single genuinely new runtime artifact in `step3_langchain_rag` is the **reranker**. Everything else is momentum or aspiration relative to what is built.

## Root-Cause of the Gaps

1. **Monolithic eval harness.** `evaluate.py` builds one `LLMTestCase` per golden that carries *both* retriever and generator output — so component-level isolation (retriever judged without generation, generator fed golden context without retrieval) is structurally impossible today.
2. **"Correctness" rebranded recall.** `evaluate_with_registry` labels `context_recall` as `accuracy`, conflating retrieval coverage with answer correctness.
3. **No measurement boundary.** Nothing instruments latency or token cost during a DeepEval run, and nothing streams per-`ask()` evals to a store — so ops-level and drift numbers cannot exist.
4. **No VCS/CI harness.** Without `.github/`, there is no place for a regression gate to live; and without a baseline store (MLflow or JSON), there is nothing to compare "iterations of the last release stopping a gate."

## Recommended Remediation Roadmap (BA hand-off to engineering)

| Phase | Scope | Uniqueness | Est. relative effort |
|---|---|---|---|
| **A** | Component-level isolation (L1): per-component golden sets + `eval_retriever.py` + `eval_generator.py` on **GroqJudge** | fix monolithic harness | Small |
| **B** | Add DeepEval `AnswerRelevancyMetric` at both component and pipeline level (completes the Triad) | close L2 | Small |
| **C** | L3 metrics: real `Correctness`/`G-Eval`, `Completeness`/`G-Eval`, latency+cost instrumentation, PII/Safety metrics | close L3 | Medium |
| **D** | Regression baseline: MLflow (or JSON baseline) storing evals; `compare_against_baseline()` fail-on-regression | regression testing | Medium |
| **E** | CI/CD gates: `.github/workflows/` — keyword evals gate PRs, DeepEval nightly + pre-release | deployment safety | Medium |
| **F** | Online observability: stream per-`ask()` latency + scores; rolling-window drift monitor vs baseline | live damping | Medium |

> **Tradeoff call for the lead:** Phases A–B are cheap and directly match the "component level" scripts; C–F are the "production-grade" layer and should be staged rather than one mega-pass. MLflow (D) vs a JSON-baseline registry is a dependency decision — the post names MLflow, but a JSON baseline needs zero new infra.

## Files Added by This Analysis
| File | Purpose |
|------|---------|
| `doc/GAP_ANALYSIS_EVAL_FRAMEWORK.md` | This document — the BA gap scorecard |

## Report Back
1. Confirm the scorecard read (which gaps you disagree with and why).
2. Tell me which phases to schedule next (A+B now, then C–F incrementally — recommended).
3. Decide D: **MLflow** vs **JSON baseline** for the regression gate.
4. Confirm judge layer for A–B: **GroqJudge** (this repo) vs the post's HTTP `gpt-4o-mini`.