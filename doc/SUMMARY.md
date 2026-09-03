# Step 3: LangChain RAG Rebuild - Task Summary

## Your Role
**Team Lead (Mentor)** → Assigning tasks, reviewing code
**Developer (You)** → Implementing code

## Project Overview
Rebuild the **exact same** Step 2 RAG project (same documents, same embedding model, same golden dataset, same 4 eval metrics, same DeepEval judge, same pipeline interface) but **using LangChain's abstractions** instead of the hand-written ones. This is primarily a **framework gap analysis** — Step 3 vs Step 2 differ in internals, so the eval scores are directly comparable and we learn where LangChain helps and where it gets in the way. **Plus one new component Step 2 lacked: a LangChain reranker** (Task 7), so we also measure what the framework can add cheaply.

> Why this matters: Step 2 proved it *works* by hand. Step 3 rebuilds it with LangChain so you can *feel* what the framework abstracts away — and decide when to use which.

> **Scope note (mentor ruling):** this step is **dense-only**. Hybrid dense+sparse search / BM25 / RRF fusion is **deferred to Step 4** (it's a retrieval-strategy concern, not a core-pipeline one). Step 3 also adds a **Typer CLI** (Task 17) and **integration tests** (Task 18) as interface/verification layers, using the **same** `Pipeline.ask()` and a **shared-loader** test pattern.

## Core idea: the "adapter" pattern ⭐
LangChain's `PromptTemplate`/chat models are an **adapter over our shared PromptRegistry**, not the source of truth. The registry (threaded through the milestones) still owns Content / Policy / Output (`output_schema`) / Evidence; LangChain just executes. See `TASK_10_PROMPT_REGISTRY.md`.

## The A/B contract (kept identical to Step 2)
| Aspect | Same as Step 2? | Notes |
|--------|----------------|-------|
| Documents (`data/documents/*.txt`) | ✅ Yes | 4 txt files, identical |
| Golden dataset (`eval/golden.jsonl`) | ✅ Yes | 20 Q&A pairs |
| Embedding model | ✅ Yes | `qwen3-embed` (Qwen3-Embedding-0.6B), 1024-dim |
| LLM | ✅ Yes | Same Qwen via `LLM_BASE_URL`/`LLM_MODEL` (LangChain `ChatOpenAI`) |
| Eval metrics | ✅ Yes | Context Recall / Precision, Faithfulness, Answer Relevance |
| DeepEval judge | ✅ Yes | Groq `openai/gpt-oss-120b` (free) |
| `pipeline.ask()` interface | ✅ Yes | returns `{answer, sources, prompt_key, rendered_hash}` |
| Framework | ❌ **Changes** | Hand-written → LangChain (splitter, Chroma, LCEL chains) |
| Reranker | ⭐ **New (Step 3 only)** | Cross-encoder second stage over Step-2 dicts (Task 7) |

## Task Order (Follow Sequentially)
Full task list below (regenerator renumbered after inserting the reranker). The reranker (7) sits between retrieval (6) and the generator (8).

| Task | File | Description | Replaces (Step 2) |
|------|------|-------------|--------------------|
| 1 | `TASK_01_SETUP.md` | Project setup via `uv init`/`uv add`, LangChain stack | `TASK_01_SETUP.md` |
| 2 | `TASK_02_EVAL_DATA.md` | Review golden dataset + register prompt V1 | `TASK_02_EVAL_DATA.md` |
| 3 | `TASK_03_SPLITTER.md` | LangChain `RecursiveCharacterTextSplitter` | `TASK_03_CHUNKER.md` |
| 4 | `TASK_04_EMBEDDINGS.md` | LangChain `Embeddings` adapter over qwen3-embed | `TASK_04_EMBEDDER.md` |
| 5 | `TASK_05_VECTORSTORE.md` | LangChain `Chroma` vector store | `TASK_05_STORE.md` |
| 6 | `TASK_06_RETRIEVER.md` | Chroma retriever (Step 2-shaped results) | `TASK_06_RETRIEVER.md` |
| 7 | `TASK_07_RERANKER.md` | ⭐ Reranker (cross-encoder second stage) | — (NEW) |
| 8 | `TASK_08_GENERATOR.md` | LangChain chain generator (`ChatOpenAI` + LCEL) | `TASK_07_GENERATOR.md` |
| 9 | `TASK_09_PIPELINE.md` | LCEL pipeline (w/ rerank stage), same `ask()` interface | `TASK_08_PIPELINE.md` |
| 10 | `TASK_10_PROMPT_REGISTRY.md` | Shared registry + LangChain adapter ⭐ | `TASK_09_PROMPT_REGISTRY.md` |
| 11 | `TASK_11_TRACER.md` | LangSmith tracing | `TASK_10_TRACER.md` |
| 12 | `TASK_12_EVALUATOR.md` | 4 keyword metrics (same as Step 2) | `TASK_11_EVALUATOR.md` |
| 13 | `TASK_13_RUN_EVAL.md` | Run eval, save results | `TASK_12_RUN_EVAL.md` |
| 14 | `TASK_14_APP.md` | Streamlit app (Chat / Eval / Traces) | `TASK_13_APP.md` |
| 15 | `TASK_15_TESTS.md` | Unit tests | `TASK_14_TESTS.md` |
| 16 | `TASK_16_DEEPEEVAL.md` | DeepEval LLM-judged suite (Groq) | `TASK_15_DEEPEEVAL.md` |
| 17 | `TASK_17_CLI.md` | Typer CLI (ingest / search / ask / eval / prompt / admin rollback) — dense-only ⭐ NEW | — (NEW) |
| 18 | `TASK_18_INTEGRATION.md` | Integration tests (real embed + Chroma, shared loader) ⭐ NEW | — (NEW) |
| 19 | `TASK_19_COMPARISON.md` | Step 2 vs Step 3 gap analysis (incl. rerank lift) ⭐ NEW | — (NEW) |
| 20 | `TASK_20_EVALS_IMPLEMENTATION.md` | Multi-level LLM eval suite: L1 (retriever/generator) + L2 (RAG Triad) + L3 (GEval/ops) on GroqJudge ⭐ NEW | — (NEW) |
| — | `GAP_ANALYSIS_EVAL_FRAMEWORK.md` | **BA analysis** (not a task): the 3-Level Eval Framework + MLflow/CI-CD/observability post vs. what this repo actually ships | — (analysis) |

> High-level journey scope (7 milestones): see `ROADMAP.md`.

## Dependency Graph
```
Task 1 (Setup: uv init + uv add LangChain stack + reranker deps)
    ↓
Task 2 (Eval Data + Prompt V1)
    ↓
Task 3 (Splitter) ← independent
    ↓
Task 4 (Embeddings) ← independent
    ↓
Task 5 (VectorStore/Chroma) ← Task 4
    ↓
Task 6 (Retriever) ← Task 4, Task 5
    ↓
Task 7 (Reranker) ← Task 6        ⭐ NEW
    ↓
Task 8 (Generator) ← Task 2 (prompt), Task 10 (adapter)
    ↓
Task 10 (Prompt Registry + Adapter) ← independent
    ↓
Task 9 (Pipeline) ← Task 3..8 (incl. reranker), Task 11, Task 10
    ↓
Task 11 (Tracer) ← independent
    ↓
Task 12 (Evaluator) ← Task 9, Task 10
    ↓
Task 13 (Run Eval) ← Task 12
    ↓
Task 14 (App) ← Task 9, Task 12
    ↓
Task 15 (Tests) ← All above
Task 16 (DeepEval) ← Task 9
Task 17 (CLI) ← Task 9, Task 12, Task 14        ⭐ NEW
Task 18 (Integration Tests) ← Task 3, Task 4, Task 5, Task 9   ⭐ NEW
Task 19 (Comparison) ← All above
Task 20 (Eval Suite L1/L2/L3) ← Task 9, Task 12, Task 16, Task 18   ⭐ NEW
```

## Files to Create
```
step3_langchain_rag/
├── data/
│   ├── documents/              ← same 4 source docs as Step 2
│   └── chat_logs/              ← app chat logs (*.jsonl, gitignored)
├── eval/
│   ├── golden.jsonl            ← same 20 Q&A pairs
│   ├── deepeval_suite/         ← LLM-judged eval (Groq)
│   │   ├── judge.py
│   │   └── evaluate.py
│   ├── goldens/                ← ⭐ per-component golden files (Task 20)
│   │   ├── retriever_goldens.json
│   │   ├── faithfulness_dataset.json
│   │   └── correctness_goldens.json
│   └── results/                ← timestamped eval results
├── evals/                      ← ⭐ Task-20 multi-level suite (L1/L2/L3 + ops)
│   ├── harness.py
│   ├── eval_retriever.py       ← L1
│   ├── eval_generator.py       ← L1
│   ├── eval_rag_pipeline.py    ← L2 (RAG Triad)
│   ├── eval_application.py     ← L3 (GEval)
│   ├── eval_latency.py         ← L3 ops
│   └── eval_cost.py            ← L3 ops
├── src/langchain_rag/
│   ├── __init__.py
│   ├── splitter.py             ← RecursiveCharacterTextSplitter
│   ├── embeddings.py           ← qwen3-embed via LangChain Embeddings
│   ├── vectorstore.py          ← LangChain Chroma
│   ├── retriever.py            ← Chroma retriever (Step 2-shaped dicts)
│   ├── reranker.py             ← ⭐ cross-encoder second stage (dict-based)
│   ├── prompts.py              ← LangChainPromptAdapter ⭐
│   ├── generator.py            ← LangChain LCEL generator
│   ├── prompt_registry.py      ← shared cross-cutting registry (adapted)
│   ├── pipeline.py             ← LCEL pipeline (same interface, rerank stage)
│   ├── tracer.py               ← LangSmith
│   └── evaluator.py            ← 4 keyword metrics (same as Step 2)
│   └── cli.py                  ← ⭐ Typer CLI (ingest/search/ask/eval/prompt/rollback)
├── tests/
│   ├── test_splitter.py
│   ├── test_embeddings.py
│   ├── test_vectorstore.py
│   ├── test_retriever.py
│   ├── test_reranker.py        ← ⭐ NEW
│   ├── test_generator.py
│   ├── test_prompt_registry.py
│   ├── test_pipeline.py
│   ├── test_tracer.py
│   ├── test_eval.py
│   ├── test_deepeval_suite.py
│   └── integration/
│       └── test_index_integration.py   ← ⭐ NEW (marker: integration)
├── app.py                      ← Streamlit UI
├── pyproject.toml              ← created by uv (add sentence-transformers, typer)
├── uv.lock                     ← created by uv
├── .env                        ← add LLM keys (Task 1)
├── .env.example
├── .gitignore
└── doc/
    ├── SUMMARY.md
    ├── ROADMAP.md
    ├── TASK_01..20.md
    └── notes.md
```

## How to Work
1. Open task file (e.g., `TASK_01_SETUP.md`)
2. Follow steps exactly
3. Run the lint/type/test commands shown
4. Show me (`Mentor`) your code → I **review it file-by-file** before you move on
5. Report back with results
6. Wait for next task assignment

## Progress Tracking
- ⬜ Not started
- 🔄 In progress
- ✅ Completed
- ❌ Blocked

## Current Status
**Next Task:** Task 20 - Multi-Level LLM Eval Suite (L1/L2/L3)
**Blocked:** None
