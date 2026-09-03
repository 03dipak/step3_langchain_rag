# Task 17: Implement CLI (Typer ingest / search / ask / eval / prompt / rollback) ⭐ NEW

## Objective
Add a **terminal interface** (`src/langchain_rag/cli.py`) alongside the Streamlit app (Task 14) — a quick way to ingest, search, and ask **without launching the web app**. Useful for debugging retrieval/rerank/answers at the command line and for scripted eval. **Dense-only** (no hybrid — that's Step 4). It drives the *same* `Pipeline.ask()` so there is exactly one code path for querying. It also exposes an **admin `rollback`** command (authenticated) for production prompt-version fallback.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/cli.py`

## Add the dependency
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv add typer
```
And register the script in `pyproject.toml`:
```toml
[project.scripts]
langchain-rag = "langchain_rag.cli:main"
```
Then it runs as `uv run langchain-rag ...` (or `langchain-rag ...` once installed).

## Commands (all dense-only, mirror the Side Project's CLI shape)

### `ingest`
```bash
uv run langchain-rag ingest [--data-dir data/documents] [--rebuild] [--dry-run PREVIEW]
```
- `--rebuild`: `pipeline.load_documents(force_rebuild=True)` (drop + re-index Chroma).
- `--dry-run`: parse/chunk only and dump the preview JSON, **no index write** (use the splitter directly).
- Echo per-file chunk counts + total; exit 1 if no documents found.

### `search`
```bash
uv run langchain-rag search "question" [--top-k 3] [--min-score 0.3]
```
- Load the pipeline (`load_documents()`), call the retriever directly, print top `k` chunks with `[i] module/... score=.4f` + 200-char preview.
- Note: `search` uses the **retriever/reranker** only (no LLM) — the cheap way to debug what gets retrieved.

### `ask`
```bash
uv run langchain-rag ask "question" [--top-k 3] [--min-score 0.3]
```
- `pipeline.ask(question, top_k, min_score)` → print `answer`, then `Sources:` with rerank/similarity score + source preview.
- Print the `prompt_key` + `rendered_hash` (first 12 chars) under the answer for traceability.

### `eval`
```bash
uv run langchain-rag eval [--golden eval/golden.jsonl]
```
- `evaluate_with_registry(pipeline, golden, registry, prompt_key)` → print the 4-metric summary + save `eval/results/eval_<ts>.json`. (Reuses Task 12/13 logic; no stub.)

### `rollback` (admin, authenticated) ⭐
```bash
uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0 [--reason "1.1.0 regressed"] [--token <ADMIN_TOKEN>] [--dry-run] [--yes]
```
- **Auth gate:** require the `ADMIN_TOKEN` env var (or `--token` for scripting). **Fail closed** — reject if neither matches. The secret lives at the **action boundary only**, never in any prompt string.
- **Read-only `--dry-run`:** show the target version, its eval evidence, the **output-contract diff** (`output_schema` keys that change) and which version will be retired — **without executing** anything.
- **Interactive confirmation:** prints a human-readable summary then asks `Proceed? [y/N]`. `--yes` bypasses it for scripting/CI (same pattern as `rm -i` vs `rm -f`). This is a UX layer on top of the unchanged fail-closed auth — no new security surface.
- **Behavior** (all registry primitives already exist):
  1. `registry.load()`
  2. read-only checks: `get(approved_only=True)` (the current live version) + `get(version=to)` (the target), then verify the target is `retired`
  3. `registry.rollback(prompt_id, to_version, reason)` — verified: re-approves the target, auto-retires the current approved, and requires the target to be a **previously released (retired)** version with **eval evidence** (`accuracy`).
  4. `registry.save()` (atomic temp-file + `os.replace`).
- **Output:** a human-readable summary (live → target, reason, target eval acc, output-contract changes, what moves to retired), then the confirmation prompt, then on success `Done. New approved version: <key>`.

### `prompt` (read-only visibility — answer "what's live / what are my options?" BEFORE rollback) ⭐
```bash
uv run langchain-rag prompt current --prompt RAG_ANSWER
uv run langchain-rag prompt list --prompt RAG_ANSWER
```
- **`current`** — the active approved version, its `output_schema` (the output contract), eval accuracy, runs logged, and *how long it's been live* (from the approve history timestamp). The most client-reassuring command: "what am I running right now."
- **`list`** — every version with status (`approved` / `retired` / `testing`), eval accuracy, created-at, and a `<-- rollback target` marker on retired (rollback-eligible) versions. This turns `--to <version>` from a blind guess into a *safe, informed* choice: the client sees only real, eligible targets instead of typing a version number blind.
- **Rationale:** `rollback()` already *enforces* "must be retired + must have eval evidence", but the client only discovers that by trial-and-error via a `ValueError`. Surfacing eligible targets up front makes a hidden constraint a visible affordance. Both commands are **read-only** (no `ADMIN_TOKEN` required).

## Execute the security/rollback dependency first
```bash
cd /home/dipak/agentic/step3_langchain_rag
# the rollback command needs the admin token in the environment
echo 'ADMIN_TOKEN=<your-admin-token>' >> .env   # local only; never commit
```

## What to look out for (from the Side Project CLI)
- **Lazy imports** inside each command (keeps `--help` fast; the app's import of `cli` must not trigger model loads). Structure as in the side project: `from ... import X` *inside* the command body.
- **`settings` singleton gotcha:** don't mutate a cached `get_settings()` with per-invocation `top_k`/`min_score` — those are **function args** passed straight to `pipeline.ask(top_k=..., min_score=...)`. Never stash CLI flags on the shared settings object.
- **DRY between commands:** `_load_pipeline()` / `_fmt_chunk()` helpers so `search`/`ask`/`eval` don't repeat the same 30 lines (this was the top duplication finding in the side-project CLI review).
- **Read-only visibility is token-free:** `prompt current` / `prompt list` must work on the checked-out registry with *no* `ADMIN_TOKEN` — only the destructive `rollback` commands gate on auth. Reuse `_load_registry()` so `prompt` and `rollback` share one load path.
- **`--dry-run` diff before any execution:** compute the before/after `output_schema` keys ($\Delta$ between the live approved version and the target) *before* calling `registry.rollback()`, so it prints the contract change (`format`, `length_policy`, `citation_policy`, `refusal_string`) without mutating state. The interactive `Proceed? [y/N]` (and `--yes`) sits on top of the unchanged fail-closed auth gate.

## Completion Criteria
- [ ] `cli.py` with `ingest` / `search` / `ask` / `eval` / `prompt` / `rollback`
- [ ] `[project.scripts]` entry point registered
- [ ] Lazy imports per command (no model load on `--help`)
- [ ] Dense-only; no hybrid flags
- [ ] No mutation of a shared cached settings object
- [ ] `ask` uses `pipeline.ask()` (single query path with the app)
- [ ] `eval` reuses Task 12 logic (not a stub)
- [ ] `rollback` requires `ADMIN_TOKEN` (fail closed); no secret in any prompt string
- [ ] `rollback` targets a previously-released (retired) version with eval evidence
- [ ] `prompt current` / `prompt list` are read-only and surface the live version + eligible rollback targets
- [ ] `rollback --dry-run` shows the output-contract diff without executing
- [ ] `rollback` confirms interactively (`Proceed? [y/N]`) and honors `--yes`
- [ ] Exit codes: non-zero on failure ("no results"/no docs/unauthenticated/cancelled)
- [ ] `ruff` + `mypy` clean

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run langchain-rag --help
uv run langchain-rag ingest --dry-run preview.json
uv run langchain-rag search "how does gradient descent work?" --top-k 2
uv run langchain-rag ask "how does gradient descent work?" --top-k 2 --min-score 0.3
uv run langchain-rag eval --golden eval/golden.jsonl
# read-only visibility (no token needed)
uv run langchain-rag prompt current --prompt RAG_ANSWER
uv run langchain-rag prompt list --prompt RAG_ANSWER
# admin: preview the rollback WITHOUT applying it
uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0 --reason "regression" --dry-run
# admin: actually roll back (requires ADMIN_TOKEN; prompts unless --yes)
uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0 --reason "regression" --token "$ADMIN_TOKEN"
uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0 --token "$ADMIN_TOKEN" --yes
# negative test: must fail without a valid token
uv run langchain-rag rollback --prompt RAG_ANSWER --to 1.0.0 || echo "rejected (expected)"
uv run ruff check src/langchain_rag/cli.py
uv run mypy src/langchain_rag/cli.py
uv run pytest tests/ -v
```

## Report Back
When done, tell me:
1. `--help` output showing the 6 commands (`ingest search ask eval prompt rollback`)
2. One `search` result (top chunk + score)
3. One `ask` answer (first 150 chars) + sources + prompt_key
4. The saved eval summary from `eval`
5. `prompt current` + `prompt list` output (what's live, the rollback-eligible targets)
6. A `rollback --dry-run` (output-contract diff, nothing applied) + a successful `rollback` (or confirm) ++ the unauthenticated rejection, with the printed summary
7. Paste your `cli.py` for review (I want to see the `_load_pipeline()` helper + how `eval` reuses Task 12 + the `rollback` auth gate, `--dry-run` diff, and confirmation)
