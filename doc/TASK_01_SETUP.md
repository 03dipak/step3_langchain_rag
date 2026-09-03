# Task 1: Project Setup (uv + LangChain Stack)

## Objective
Set up the Step 3 project using **`uv`** as the project/package manager (the recommended modern way), install the LangChain stack, and configure `.env` with the same working LLM keys as Step 2 (so Step 2 vs Step 3 is a fair, like-for-like comparison).

## What You Need To Do

### Step 1.1: Scaffold the project with `uv init`
A bare Step 3 folder existed (README only). Since you asked to scaffold properly (rather than copy a `pyproject.toml`), the project was created with `uv`:
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv init --package --name langchain-rag --python 3.12
```
This auto-creates:
- `pyproject.toml` (packaging, `[project.scripts]`)
- `uv.lock` (dependency lockfile)
- `.python-version` (`3.12`)
- `src/langchain_rag/` package dir + `__init__.py`

> Mentor note: `--package` gives you a `src/`-layout, `--python 3.12` bakes the interpreter into `.python-version` (uv reads this so `uv run` always uses 3.12). This is the modern uv workflow — no manual `requirements.txt`.

### Step 1.2: Add runtime dependencies with `uv add`
Add the LangChain "modern split" stack + the evaluation/UI deps:
```bash
uv add langchain langchain-core langchain-text-splitters \
       langchain-chroma langchain-openai langchain-groq \
       langchain-huggingface chromadb \
       qwen3-embed streamlit python-dotenv langsmith \
       deepeval openai numpy sentence-transformers typer
```
Then pin DeepEval to the same version Step 2 uses (coexists with qwen3-embed):
```bash
uv add "deepeval==2.9.3"
```

**What each is for:**
| Package | Role in Step 3 |
|---------|----------------|
| `langchain` / `langchain-core` | Core framework + LCEL abstractions |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` (replaces hand chunker) |
| `langchain-chroma` / `chromadb` | `Chroma` vector store (replaces numpy store) |
| `langchain-openai` | `ChatOpenAI` → the **same Qwen** as Step 2, OpenAI-compatible |
| `langchain-groq` | `ChatGroq` → DeepEval **judge** |
| `langchain-huggingface` | HF embedding utils (reference; we use a custom qwen adapter) |
| `qwen3-embed` | **Same** embedding model as Step 2 (Qwen3-Embedding-0.6B) |
| `deepeval` | LLM-judged eval suite (reused from Step 2) |
| `streamlit` / `python-dotenv` / `numpy` / `openai` / `langsmith` | UI, env, math, client, tracing |

### Step 1.3: Add dev dependencies
```bash
uv add --dev mypy pyright pytest pytest-cov pytest-mock ruff
```

### Step 1.4: Verify install
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run python -c "import langchain, langchain_core, langchain_text_splitters, langchain_chroma, langchain_openai, langchain_groq, chromadb, qwen3_embed, deepeval, streamlit; print('Setup OK')"
```
**Expected resolved versions (approx):** deepeval **2.9.3**, langchain 1.3.x, langchain-core 1.6.x, langchain-chroma 1.1.x, langchain-openai 1.6.x, qwen3-embed 1.10.x.

### Step 1.5: Configure `.env` with working keys
Step 3's `.env` currently has `GROQ_API_KEY` + `GEMINI_API_KEY` only. To use the **same Qwen** as Step 2 (fair comparison), add the OpenAI-compatible endpoint keys:
```
GROQ_API_KEY=gsk_...                  # already present (DeepEval judge)
LLM_BASE_URL=<same OpenAI-compatible URL as Step 2 /v1>
LLM_API_KEY=sk-...                    # same key as Step 2
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
EMBED_MODEL=n24q02m/Qwen3-Embedding-0.6B-ONNX

LANGSMITH_API_KEY=ls_your_key_here    # optional: real LangSmith PAT
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=step3-langchain-rag
```
> ⚠️ `.env` is git-ignored. Never put real keys into a tracked file.

### Step 1.6: Seed the shared PromptRegistry + golden data
The `eval/golden.jsonl` (20 Q&A pairs) and `data/documents/*.txt` are **already identical to Step 2** (this is required for the A/B comparison). The shared `PromptRegistry` is imported from `src/langchain_rag/prompt_registry.py` (adapted from Step 2 — Task 10). Nothing to copy; you will write it by hand following Task 10.

### Step 1.7: Create `py.typed` marker for mypy
```bash
touch src/langchain_rag/py.typed
```
This empty file tells mypy the package supports typing (PEP 561). Without it, mypy skips analysis of your code and throws `import-untyped` errors.

## Completion Criteria
- [ ] `pyproject.toml` + `uv.lock` created by `uv init` (not hand-copied)
- [ ] LangChain stack installs cleanly (deepeval 2.9.3 pins correctly with qwen3-embed)
- [ ] `.env` has `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL` (same Qwen as Step 2)
- [ ] `py.typed` marker exists in `src/langchain_rag/`
- [ ] Test command passes
- [ ] (Optional) Real LangSmith key added

## Pytest Config

Add this to `pyproject.toml` (after `[build-system]`, before `[dependency-groups]`):

```toml
[tool.pytest.ini_options]
pythonpath = ["src", "eval"]
asyncio_mode = "auto"
addopts = "-m 'not integration'"
```

Without this, `uv run pytest` will fail on `import langchain_rag.splitter` because pytest won't know to look in `src/`.

## Mocking (learner note)
Set up `pytest-mock` (`mocker` fixture) + pytest's `monkeypatch`/`tmp_path` now — every later task's offline tests depend on them, exactly as in Step 2.

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv pip list | grep -iE "^(langchain|langchain-core|langchain-chroma|langchain-openai|deepeval|qwen3)" | sort
uv run ruff --version && uv run mypy --version && uv run pytest --version
```

## Report Back
When done, tell me:
1. Output of `uv pip list` (the LangChain stack + deepeval/qwen3 versions)
2. Which keys are in `.env` (names only, not the values)
3. Any install conflicts (esp. deepeval vs qwen3/click)
