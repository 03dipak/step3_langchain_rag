# Step 3 — LangChain RAG (Framework Comparison)

A complete, **dense-only** Retrieval-Augmented Generation (RAG) pipeline rebuilt on
**LangChain** — the same documents, embedding model, golden dataset, evaluation metrics, and
query interface as the hand-written pipeline (Step 2), so the two can be compared
apples-to-apples.

> **In one line:** prove *what a framework earns you and what it hides* by rebuilding the same
> RAG twice — by hand (Step 2) and with LangChain (this step).

---

## 1. What this is (for a Business Analyst / Product person)

**The problem:** we're deciding *when to use a framework versus writing it by hand*. That's a
business decision, not a gut call — so this step produces a **measured answer** (code volume,
debugging cost, flexibility, and whether they help or hurt eval scores).

**What this project delivers:**

| Deliverable | What it is |
|---|---|
| **RAG pipeline** | Split → Embed → Store → Retrieve → Rerank → Generate |
| **Prompt Registry** | Versioned, reviewable prompt lifecycle with eval evidence (audit trail) |
| **Evaluation harness** | 4 quality scores per query + a DeepEval (LLM-judged) suite |
| **CLI** | `ingest` / `search` / `ask` / `eval` + read-only `prompt` (current/list) + admin `rollback` (fail-closed, dry-run, confirm) |
| **Web app** (Streamlit) | Chat UI + Eval Dashboard + Trace viewer |
| **Tests** | Unit + integration suites (offline) |

**Scope (in / out):**
- ✅ **In scope:** dense retrieval, cross-encoder reranking, prompt versioning, keyword +
  LLM-judged evaluation, CLI, web app, tests.
- ❌ **Out of scope (deferred to Step 4):** hybrid dense+sparse (BM25), RRF fusion, multi-source
  routing.
- 🎯 **North star:** same data, same model, same metrics as Step 2 ⇒ the **only** difference is
  the framework, so score changes are attributable.

---

## 2. How it works (for a Solution Architect)

```
data/documents/*.txt
   │  RecursiveCharacterTextSplitter (Chroma metadata: source=basename)
   ▼
[{text, metadata}]
   │  Qwen3Embeddings.embed_documents()   (custom LangChain Embeddings)
   ▼
Chroma (persisted, cosine default)          LangChainStore.search() → {text, metadata, score}
   │                                         (scores recovered via _collection.query, 1−distance)
   ▼
Retriever.retrieve(question, top_k, min_score) → top_k dicts
   │
   ▼
Reranker.rerank(question, candidates)       (cross-encoder 2nd stage)
   │                                            ms-marco-MiniLM-L-6-v2, re-sort → top_n
   ▼
LangChainGenerator.generate(question, reranked)   (LCEL chain)
   │                                            ChatOpenAI → same Qwen as Step 2
   ▼
{ answer, sources, prompt_key, rendered_hash }    ← identical shape to Step 2
```

**The core design insight ⭐**
LangChain's `PromptTemplate` is an **adapter over a shared PromptRegistry — not the source of
truth.** The registry owns Content (template), Policy (model/temperature), and Evidence (eval
scores, run log). LangChain only *executes* the approved prompt via `LangChainPromptAdapter.build_chain(...)`.

```
Your Registry (source of truth)     LangChain (adapter)
──────────────────────────          ──────────────────
template, model, temp      ──→      PromptTemplate + ChatOpenAI
output_schema (rules)      ──→      injected into the rendered prompt
eval scores, run log       ──→      (stays in the registry)
```

Version a prompt in the registry → the LCEL chain changes with **zero code edits**. Each version
carries an **`output_schema`** (format/length/citation/refusal rules) so *how the answer is
produced* is versioned and rolled back with the prompt.

**Key architecture decisions** (recorded in `doc/notes.md`):

| Decision | Rationale |
|---|---|
| **Same embedding as Step 2** (qwen3-embed, 1024-dim) | Apples-to-apples A/B — not `HuggingFaceEmbeddings`/bge |
| **Custom `Qwen3Embeddings(Embeddings)` adapter** | qwen3 isn't a HF wrapper; query uses instruction-aware `query_embed` |
| **Score recovery via Chroma `_collection`** | `similarity_search` hides scores; we need `min_score` + Step-2 shape |
| **Reranker = direct cross-encoder, not `ContextualCompressionRetriever`** | dict-shaped retriever; modern `langchain` split path |
| **Registry + adapter for prompts** | LangChain has no versioned, evidence-linked prompt lifecycle |
| **`output_schema` on each version** | Output contract is versioned policy → eval + rollback; editing an approved version forces re-eval |
| **Dense-only** | Hybrid fusion is deliberately deferred to Step 4 |

---

## 3. Tech stack

| Area | Tool |
|---|---|
| Language | Python ≥ 3.12, `uv` package manager |
| Embeddings | `Qwen3-Embedding-0.6B` (ONNX, local) |
| Vector store | ChromaDB (cosine, persisted) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM / judge | OpenAI-compatible endpoint (`LLM_BASE_URL`) + Groq LLM-judge |
| Framework | LangChain, LangGraph-ready, Typer CLI, Streamlit |
| Evaluation | DeepEval 2.9.3 |

---

## 4. Install & run (for a Junior Developer)

Prerequisite: [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Clone
git clone git@github.com:03dipak/step3_langchain_rag.git
cd step3_langchain_rag

# 2. Create env + install dependencies (run + dev)
uv sync

# 3. Configure secrets
cp .env.example .env
#  → edit .env: set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, EMBED_MODEL, GROQ_API_KEY
```

> `.env` is git-ignored (holds API keys). NEVER commit it.

**What you need in `.env`:**

| Key | Required? | Purpose |
|---|---|---|
| `LLM_BASE_URL` | yes | OpenAI-compatible LLM endpoint (`<…>/v1`) |
| `LLM_API_KEY` | yes | LLM key |
| `LLM_MODEL` | yes | model id (same Qwen as Step 2, for fair comparison) |
| `EMBED_MODEL` | yes | embedding model id (auto-downloaded on first run) |
| `GROQ_API_KEY` | yes | DeepEval LLM-judge (free tier) |
| `LANGSMITH_API_KEY` | optional | LangSmith tracing |

### Run it

```bash
# CLI (ingest / search / ask / eval / prompt / rollback)
uv run langchain-rag --help
uv run langchain-rag ingest                       # chunk + embed + index data/documents
uv run langchain-rag ask "how does gradient descent work?" --top-k 2
uv run langchain-rag prompt current               # read-only: what prompt is live
uv run langchain-rag prompt list                  # read-only: versions + rollback targets
uv run streamlit run app.py                       # Web UI (Chat / Eval / Traces)

# admin, authenticated rollback (fail-closed on ADMIN_TOKEN)
uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0 --dry-run   # preview only
ADMIN_TOKEN=... uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0  # apply
```

**First run downloads models** (~60–110 MB ONNX embedder + ~23 MB reranker, cached). After that
everything runs **offline**.

---

## 5. Evaluation & quality gates

```bash
# Keyword eval (same 4 metrics as Step 2)
uv run langchain-rag eval

# DeepEval (Groq LLM-judge) suite
uv run python -m eval.deepeval_suite.eval_retriever
uv run python -m eval.deepeval_suite.eval_generator
# ...full suite in eval/deepeval_suite/

# Quality gates
uv run ruff check src tests eval app.py           # lint
uv run mypy --strict src/langchain_rag            # strict static typing
uv run pytest -v                                  # unit tests (offline, no model download)
uv run pytest -m integration                      # real embedding + Chroma roundtrip
uv run pytest --cov --cov-report=term-missing      # coverage per module
```

> **Dev/test convention:** unit tests never hit the network, never load models, never call the
> LLM (they mock the boundary). Integration tests are the only ones that load real models — and
> they share a **module-scoped embedder fixture** so a model loads **once**, not per test.

**What's checked per task (Definition of Done):** `ruff` clean, `mypy --strict` clean,
`pytest` passes offline, mentor review passed, findings recorded back in `doc/notes.md`.

---

## 6. Project layout

```
step3_langchain_rag/
├── doc/                      # 19 task specs + SUMMARY + ROADMAP + notes
├── data/
│   ├── documents/            # 4 source docs (same as Step 2)
│   └── chat_logs/            # app chat logs (gitignored)
├── eval/
│   ├── golden.jsonl          # 20 Q&A pairs (same as Step 2)
│   ├── deepeval_suite/       # Groq LLM-judged eval
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

## 7. Where to go next

```
1  Hand-Written RAG      ✅ step1_basic_rag
2  Eval & Tracing        ✅ step2_rag_eval
3  Framework Comparison  ◀  THIS repo (LangChain rebuild)
4  Multi-Source Routing     step4_multi_source_rag   (hybrid fusion starts here)
5  Agentic RAG (LangGraph)  step5_agentic_rag
6  Multi-Agent systems      step6_multi_agent
7  Production / guardrails  step7_deploy
```

**More:** roadmap & notes in `doc/ROADMAP.md`, `doc/SUMMARY.md`, and per-file task specs in `doc/TASK_*.md`.