# Step 3 — LangChain RAG (Framework Comparison)

A complete, **dense-only** Retrieval-Augmented Generation (RAG) pipeline rebuilt on
**LangChain** — the same documents, embedding model, golden dataset, evaluation
metrics, and query interface as the hand-written pipeline (Step 2), so the two can
be compared apples-to-apples.

> **In one line:** prove *what a framework earns you and what it hides* by
> rebuilding the same RAG twice — by hand (Step 2) and with LangChain (this step).

---

## 1. For Developers — Install & Run in 2 minutes

This project uses **[uv](https://docs.astral.sh/uv/)** (modern, fast). You do **not**
need pip, venv, or a `requirements.txt`.

### Prerequisites
- Python **3.12**
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- An OpenAI-compatible LLM endpoint (e.g. Groq or a local Qwen gateway) — see `.env`

### 1.1 Install
```bash
cd step3_langchain_rag
uv sync                     # installs all run + dev deps from uv.lock
```

### 1.2 Configure `.env`
```bash
cp .env.example .env        # then fill in keys
```
Minimal keys (same LLM as Step 2, for a fair comparison):
```
LLM_BASE_URL=<your-openai-compatible-url/v1>
LLM_API_KEY=sk-...
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
EMBED_MODEL=n24q02m/Qwen3-Embedding-0.6B-ONNX
GROQ_API_KEY=gsk_...        # used by the DeepEval LLM-judge (free tier)
# Optional:
# LANGSMITH_API_KEY=ls_...  # LangSmith tracing
```

### 1.3 Quick start
```bash
uv run langchain-rag --help          # CLI: ingest / search / ask / eval / prompt / rollback
uv run langchain-rag ingest          # chunk + embed + index data/documents
uv run langchain-rag ask "how does gradient descent work?" --top-k 2
uv run langchain-rag prompt current  # read-only: what prompt is live right now
uv run langchain-rag prompt list     # read-only: all versions + rollback targets
uv run streamlit run app.py          # Web UI (Chat / Eval / Traces)
# admin, authenticated rollback (fail-closed on ADMIN_TOKEN):
uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0 --dry-run   # preview only
ADMIN_TOKEN=... uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0  # apply
```
> **CLI == Task 17** (`ingest` / `search` / `ask` / `eval` / `prompt` / `rollback`).
> The `prompt current` / `prompt list` commands are **read-only** (no token needed)
> and answer "what's live / what can I roll back to" before any destructive action.
> `rollback` is fail-closed (requires `ADMIN_TOKEN` or `--token`), supports a
> non-destructive `--dry-run` that shows the output-contract diff, and prompts for
> confirmation (`--yes` to bypass). The Streamlit app and direct module calls work
> identically.

### 1.4 The first run downloads models
- **Embedding:** `Qwen3-Embedding-0.6B` (~60–110 MB ONNX, cached after first run).
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~23 MB, cached). Added in Task 7.
After the first download, everything runs **offline** (no more network).

---

## 2. For Business Analysts — Scope, ROI, Deliverables

### What this project does
Takes a folder of lecture notes / documents, indexes them into a searchable store,
and answers questions *only from those documents* — with retrievable sources and
measurable answer quality.

### What we built (deliverables)
| Deliverable | What it is |
|-------------|-----------|
| **RAG pipeline** | Split → Embed → Store → Retrieve → Rerank → Generate |
| **Prompt Registry** | Versioned, reviewable prompt lifecycle with eval evidence (audit trail) |
| **Evaluation harness** | 4 quality scores per query + an LLM-judged DeepEval suite |
| **CLI** | `ingest` / `search` / `ask` / `eval` + read-only `prompt` (current/list) + admin `rollback` (fail-closed, dry-run, confirm) from the terminal |
| **Web app** (Streamlit) | Chat UI + Eval Dashboard + Trace viewer |
| **Tests** | Unit + integration suites (run offline) |

### Scope (in / out)
- ✅ **In scope:** dense retrieval, cross-encoder reranking, prompt versioning,
  keyword + LLM-judged evaluation, CLI, web app, tests.
- ❌ **Out of scope (deferred to Step 4):** hybrid dense+sparse search (BM25),
  RRF fusion, multi-source routing.
- 🎯 **North star:** same data, same model, same metrics as Step 2 ⇒ the **only**
  difference is the framework, so score changes are attributable.

### Why it matters (the business question)
We are deciding **when to use a framework vs. write it by hand.** This step produces
a **measured answer**: code-volume reduction, debugging cost, flexibility — and
whether those help or hurt eval scores. Not a gut call, a scorecard.

---

## 3. For Solution Architects — Architecture & Key Decisions

### Pipeline flow (dense-only)
```
data/documents/*.txt
   │  RecursiveCharacterTextSplitter (Chroma metadata: source=basename)
   ▼
[{text, metadata}]
   │  Qwen3Embeddings.embed_documents()   (Task 4 — custom LangChain Embeddings)
   ▼
Chroma (persisted, cosine default)          LangChainStore.search() → {text, metadata, score}
   │                                         (scores recovered via _collection.query, 1−distance)
   ▼
Retriever.retrieve(question, top_k, min_score) → top_k dicts
   │
   ▼
Reranker.rerank(question, candidates)       (Task 7 — cross-encoder 2nd stage)
   │                                            ms-marco-MiniLM-L-6-v2, re-sort → top_n
   ▼
LangChainGenerator.generate(question, reranked)   (Task 8 — LCEL chain)
   │                                            ChatOpenAI → same Qwen as Step 2
   ▼
{ answer, sources, prompt_key, rendered_hash }    ← identical shape to Step 2
```

### The core design insight ⭐
**LangChain's `PromptTemplate` is an *adapter over a shared PromptRegistry* — not
the source of truth.** The registry owns Content (template), Policy (model/
temperature), and Evidence (eval scores, run log). LangChain only *executes* the
approved prompt via `LangChainPromptAdapter.build_chain(...)`.

```
Your Registry (source of truth)     LangChain (adapter)
──────────────────────────          ──────────────────
template, model, temp      ──→      PromptTemplate + ChatOpenAI
output_schema (rules)      ──→      injected into the rendered prompt
eval scores, run log       ──→      (stays in the registry)
```
Version a prompt in the registry → the LCEL chain changes with **zero code edits**. Each version also carries an **`output_schema`** (format/length/citation/refusal rules) so changing *how the answer is produced* is versioned and rolled back with the prompt — see `doc/ARCHITECTURE_ROLLBACK_SECURITY.md` §5.

### Key architecture decisions (recorded in `doc/notes.md`)
| Decision | Rationale |
|----------|-----------|
| **Same embedding as Step 2** (qwen3-embed, 1024-dim) | Apples-to-apples A/B — not `HuggingFaceEmbeddings`/bge |
| **Custom `Qwen3Embeddings(Embeddings)` adapter** | qwen3 isn't a HF wrapper; proper abstraction boundary; query uses instruction-aware `query_embed` |
| **Score recovery via Chroma `_collection`** | `similarity_search` hides scores; we need `min_score` + Step-2 shape |
| **Reranker = direct cross-encoder, not `ContextualCompressionRetriever`** | That API needs a `Document`-based `BaseRetriever`; our retriever is dict-shaped, and the import path isn't in the modern `langchain` split |
| **Registry + adapter for prompts** | LangChain has no versioned prompt lifecycle with eval evidence |
| **`output_schema` on each version** | The output *contract* (format/length/citations/refusal) is versioned policy → gets eval + rollback; editing it on an approved version forces re-eval (no silent change) |
| **Dense-only** | Hybrid fusion is a Step 4 concern; deferred deliberately |

---

## 4. For Code Reviewers & Testers — Quality Gates

### Toolchain
| Tool | Command | What it checks |
|------|---------|----------------|
| **ruff** (lint) | `uv run ruff check src tests eval app.py` | style + correctness lint |
| **mypy** (types) | `uv run mypy --strict src/langchain_rag` | strict static typing |
| **pytest** (unit) | `uv run pytest -v` | unit tests (offline, no model download) |
| **pytest -m integration** | `uv run pytest -m integration` | real embedding + Chroma roundtrip |
| **coverage** | `uv run pytest --cov --cov-report=term-missing` | code coverage per module |

> **Dev/test convention:** unit tests never hit the network, never load models,
> never call the LLM (they mock the external boundary). Integration tests are the
> only ones that load real models — and they share a **module-scoped embedder
> fixture** so a model loads **once**, not once per test (that's what keeps the
> suite fast and non-hanging).

### How to review code
Follow the numbered tasks in `doc/` in order — **the developer writes each file and
the mentor (lead) reviews it file-by-file** before moving on:
1. `doc/SUMMARY.md` — the 19-task plan, dependency graph, files to create.
2. `doc/TASK_0N_*.md` — per-file spec (objective, contract, mock strategy, verify).
3. `doc/notes.md` — architecture decisions, gotchas, and known tradeoffs.
4. `doc/ARCHITECTURE_ROLLBACK_SECURITY.md` — system-design spec for prompt rollback + action-boundary auth (Task 17 `rollback`).

### Definition of Done (each task)
- ✅ `ruff` clean
- ✅ `mypy --strict` clean
- ✅ `pytest` for that module passes (offline)
- ✅ Mentor code review passed
- ✅ Findings recorded back in `doc/notes.md` / the task file

---

## 5. Project Layout
```
step3_langchain_rag/
├── doc/                      # 19 task specs + SUMMARY + ROADMAP + notes
├── data/
│   ├── documents/            # 4 source docs (same as Step 2)
│   └── chat_logs/            # app chat logs (gitignored)
├── eval/
│   ├── golden.jsonl          # 20 Q&A pairs (same as Step 2)
│   ├── deepeval_suite/       # Groq LLM-judged eval (Task 16)
│   └── results/              # timestamped eval results
├── src/langchain_rag/
│   ├── splitter.py           # RecursiveCharacterTextSplitter
│   ├── embeddings.py         # Qwen3Embeddings(Embeddings)
│   ├── vectorstore.py        # LangChain Chroma store
│   ├── retriever.py          # dict-shaped retriever
│   ├── reranker.py           # cross-encoder 2nd stage
│   ├── prompts.py            # LangChainPromptAdapter + build_llm
│   ├── generator.py          # LangChain LCEL generator
│   ├── prompt_registry.py    # versioned prompt lifecycle (shared)
│   ├── prompt_registry.json  # seeded registry: RAG_ANSWER V1.0.0/1.1.0(retired)/1.2.0(live)
│   ├── pipeline.py           # wires it all together; pipeline.ask()
│   ├── tracer.py             # LangSmith (optional)
│   ├── evaluator.py          # 4 keyword metrics (same as Step 2)
│   └── cli.py                # ingest / search / ask / eval / prompt / rollback
├── tests/                    # unit + integration
├── app.py                    # Streamlit UI
└── pyproject.toml            # uv project (deps, scripts, dev group)
```

---

## 6. Roadmap & This Step's Place
```
1  Hand-Written RAG      ✅ step1_basic_rag
2  Eval & Tracing        ✅ step2_rag_eval
3  Framework Comparison  ◀  THIS repo (LangChain rebuild)
4  Multi-Source Routing     (hybrid fusion starts here)
5  Agentic RAG (LangGraph)
6  Multi-Agent systems
7  Production / guardrails
```

See `doc/ROADMAP.md` for the full journey and how this step sets up Step 4+.
