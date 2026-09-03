# RAG Learning Journey - High-Level Scope (Step 3)

The overall RAG journey is scoped across **7 milestones**. We work through them
sequentially, and **enhance the approach / record decisions in the `doc/*.md`
files** as we go (each milestone gets task docs + notes).

## The 7 Milestones

| # | Milestone | Scope | Where |
|---|-----------|-------|-------|
| 1 | **#HandWrittenRAG** | Zero frameworks — manual chunking, vector embeddings, cosine-similarity mechanics by hand. | `step1_basic_rag` |
| 2 | **#Evaluation & #Tracing** | Measuring retrieval precision and answer quality using LangSmith (+ eval metrics / DeepEval). | `step2_rag_eval` |
| 3 | **#FrameworkComparison** | Deep-dive gap analysis comparing custom scripts to LangChain abstractions. | `step3_langchain_rag` *(THIS step)* |
| 4 | **#MultiSource #Routing** | Structuring dynamic query routing across disparate knowledge bases. | `step4_multi_source_rag` |
| 5 | **#AgenticRAG** | Building state-machine loops (Corrective / Self-RAG) with LangGraph. | `step5_agentic_rag` |
| 6 | **#MultiAgent Systems** | Setting up specialized nodes with targeted roles, tool access, and trace monitoring. | `step6_multi_agent` |
| 7 | **#Production & #Guardrails** | Hardening the pipeline with rate limits, fallback behavior, cost controls, and deployment. | `step7_deploy` |

## Where the steps live

```
1 HandWrittenRAG     ✅ step1_basic_rag
2 Eval & Tracing     ✅ step2_rag_eval
3 FrameworkComparison ← step3_langchain_rag  (THIS step)
4 MultiSource routing
5 Agentic RAG (LangGraph)
6 MultiAgent systems
7 Production / guardrails
```

## Why Step 3 exists (the point of the milestone)

Step 2 built the pipeline **by hand** so you felt the mechanics (chunking, numpy
cosine similarity, prompt formatting, versioning). Step 3 rebuilds the *same*
project with **LangChain** so you can quantify the tradeoff:

- **What you gain:** less boilerplate (splitters, Chroma, LCEL chains, built-in
  retriever/vector-store abstractions), easier provider switching, built-in
  streaming/tracing hooks.
- **What you lose:** transparency into the mechanics, and — critically —
  LangChain has **no** concept of a versioned prompt lifecycle with eval
  evidence. That's why the shared PromptRegistry + `LangChainPromptAdapter`
  (Task 10) is the heart of this step.

## Cross-cutting concerns

- **PromptRegistry threads through all 7 steps** — every step records which
  prompt version + eval score produced each result. In Step 3, LangChain's
  `PromptTemplate` is just an **adapter** over this registry (never the source
  of truth).
- **Chat-log logging is in scope (as in Step 2):** user chat traffic is
  persisted to `data/chat_logs/`. The app appends every exchange to
  `data/chat_logs/chat_log_<YYYY_MM_DD>.jsonl` (JSONL, gitignored) — a durable,
  local log independent of LangSmith.
- **Production-eval gap:** the marketing "3-Level Eval Framework + MLflow
  regression + CI/CD + drift observability" is largely aspirational here. See
  `GAP_ANALYSIS_EVAL_FRAMEWORK.md` for the honest scorecard and the A–F
  remediation roadmap (component-level isolation → AnswerRelevancy → L3
  ops/metrics → baseline regression → CI gates → drift monitoring).
- **Task 20 closes gaps A–E** (component L1, pipeline L2, application L3 GEval,
  ops latency/cost) via the `evals/` multilevel suite on GroqJudge. **Gap F
  (baseline regression / CI gates) + MLflow + drift are deferred to the next
  step.** A latent `Pipeline()` construction bug (registry never seeded) was
  fixed here in `generator.py`; see `TASK_20_EVALS_IMPLEMENTATION.md`.

## How we work

1. Pick the current milestone's task doc in its `doc/` folder.
2. Implement, then **enhance the approach and write the findings into the MD docs**.
3. Run the verify commands (ruff, mypy, pytest) with coverage.
4. Record any new decision/note in `doc/notes.md`.
5. **The developer writes the code; the mentor reviews it file-by-file before moving on.**

## Reranker decision (revised for Step 3)

- **In Step 3 — yes (new).** Reranking was **not** in Step 2, so it's the one
  component this step adds (**Task 7**). It's a **cross-encoder second stage**
  over the already-retrieved Step-2 dicts (`sentence-transformers`
  `cross-encoder/ms-marco-MiniLM-L-6-v2`), sitting between the Chroma
  retriever and the generator: over-fetch candidates, re-score, drop to
  `top_n`. (Dropped LangChain's `ContextualCompressionRetriever` — it needs a
  `Document`-based `BaseRetriever`, which conflicts with our dict-shaped
  retriever, and its import path doesn't resolve in the modern `langchain`
  split.)
- **Effect on the comparison:** Step 2 (no reranker) vs Step 3 (reranker) is no
  longer a strict 1:1 — it's a 1:1 **plus a measure of what LangChain can add
  cheaply**. We keep the same 4 eval metrics, so any score lift is attributed
  to the rerank stage. Weigh that in the Task 19 gap analysis.
- **From Step 4 onwards — yes, at scale.** Step 4 (Multi-Source Routing) is the
  natural home for cross-source reranking (retrieve top-k from *multiple* KBs,
  then re-rank across sources before generating).

## Step 3 scope (mentor ruling)

- **Dense-only.** Hybrid dense+sparse / BM25 / RRF fusion is **deferred to
  Step 4** — it's a retrieval-strategy concern, not a core-pipeline one.
- **Two interface/verification layers added:** a **Typer CLI** (ingest / search
  / ask / eval, Task 17) and **integration tests** (real embeddings + Chroma,
  Task 18). Both reuse the same `Pipeline.ask()` and, in tests, a
  **shared-loader fixture** so models load once.

