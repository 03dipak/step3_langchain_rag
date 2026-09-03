# Solution Architecture — Prompt Version Rollback & Security (Step 3)

> **Audience:** Solution Architects / System Designers.
> **Decision under review:** how to roll a prompt back in production (1.0.0 → 1.1.0 → back to 1.0.0), whether any "HF"/HuggingFace or git involvement is needed, and where security belongs.

## TL;DR — the answer in one table

| Concern | Tool / Mechanism | Is it *the* answer? |
|---------|------------------|---------------------|
| Live rollback (the switch) | `PromptRegistry.rollback()` + authenticated CLI command | ✅ Yes — the actual mechanism |
| Audit trail | `registry.history` entries (in `prompt_registry.json`) | ✅ Yes — built-in, timestamped, non-repudiable |
| DR / backup / blame | committed `prompt_registry.json` in the git repo | 🟡 Backup/DR only — **not** the live switch |
| Embeddings | `qwen3-embed` (custom adapter, **not** HF) | ❌ Irrelevant to rollback |
| Auth / integrity | `ADMIN_TOKEN` at the **action boundary**; signed config **never** in a prompt | 🔒 Recommended (this doc) |

**Core principle: the registry is the source of truth for the *active* prompt; git is the source of truth for *history/DR*; security lives at the action boundary, not in the prompt string.**

---

## 1. The live rollback is an application operation — not git, not HF

`PromptRegistry` already implements rollback (`src/langchain_rag/prompt_registry.py:233`):

| Step | Call | Effect |
|------|------|--------|
| Release 1.0.0 | `register` → `promote` → `record_eval_scores(..."accuracy")` → `promote` | 1.0.0 = **approved** |
| Release 1.1.0 | same path → `promote` | 1.1.0 = **approved**; `_activate` auto-**retires** 1.0.0 |
| **Rollback to 1.0.0** | `rollback(prompt_id, "1.0.0", reason)` | 1.0.0 = **approved** again; `_activate` retires 1.1.0 |

Because `_activate` (lines 53–80) is the **single atomic** approve+retire rule that both release and rollback share, there is exactly one code path and no way to have two approved prompts at once. `get(approved_only=True)` returns `_current_approved` (newest approved, line 143), so after rollback it serves 1.0.0.

**Why not git as the switch?** `git revert` requires a redeploy per node and mixes code with runtime config. In a multi-instance deployment each node's local `prompt_registry.json` would **drift** — git solves audit, not consistency. Git is the **DR + blame layer** (the JSON is committed and version-controlled); the rollback command is the **runtime switch**.

**Why not HF?** HuggingFace is about models/embeddings. Rollback is a *data-policy* operation on the registry — no model re-download, no HF, no `langchain-huggingface` involvement. (We use `qwen3-embed` here anyway.)

---

## 2. Security: forbid secrets in prompts; gate the *action*

### Why NOT put a password inside the prompt
- A prompt is sent to the LLM (and to provider servers / traces / logs) — a secret there **leaks** by design.
- A prompt is model content, not an auth boundary; any caller with the `prompt_id` can read it.
- It doesn't prevent an internal actor from approving a tampered prompt.

### What to do instead (the layered model)

```
              ┌───────────────────────────────────────────────┐
              │   ADMIN (who can flip the approved prompt)     │
              │   ┌───────────────────────────────────────────┐│
   Subject ──>│   │  ACTION GATE  (authn + authz)             ││
              │   │  - ADMIN_TOKEN env / --token  → identity  ││
              │   │  - role allow-list             → "admin"  ││
              │   └───────────────────────────────────────────┘│
              │        │  (authenticated, authorized)          │
              │        ▼                                        │
              │   PromptRegistry.rollback()  ──> save()        │
              │   Immutable versions + full history audit       │
              └───────────────────────────────────────────────┘
                          (source of truth: prompt_registry.json)
```

1. **Authn at the action layer** — a `rollback` requires `ADMIN_TOKEN` (env) or `--token`; **fail closed**. Streamlit/API wrappers add a role check; full RBAC/IAM is deferred to Step 7 (Production guardrails).
2. **Audit (non-repudiation)** — every status change is appended to `history` (lines 117–126): `key`, `from_status`, `to_status`, `timestamp`, `reason`. The rollback command prints a human-readable before/after summary (live → target, output-contract diff, what moves to retired) + the new approved key, so the operator has clean on-screen evidence.
3. **Integrity (optional hardening, Step 7)** — sign/hash the active registry blob and resolve the *active prompt* from a shared store so tampering is detectable and all nodes agree (see §4).

### Client trust: make the rollback *visible and informed* before it happens
A blind `rollback --to <version>` forces the client to type a version and hope. The CLI therefore answers three questions **before** the rollback executes — closing the gap between "the mechanism is correct" and "the client can see and trust the mechanism":

| Layer | Command | Answers | Auth required? |
|-------|---------|---------|----------------|
| **What's live now?** | `prompt current` | active version, `output_schema`, eval acc, runs logged, time-live | ❌ read-only |
| **What can I roll back to?** | `prompt list` | all versions + status + acc + `<-- rollback target` marker on eligible | ❌ read-only |
| **What will change?** | `rollback --dry-run` | target eval evidence + `output_schema` diff + who gets retired, **without executing** | ✅ (token) |
| **Did I mean it?** | `Proceed? [y/N]` + `--yes` | confirmation gate before the switch | ✅ (token) |
| **The switch** | `rollback ... [--yes]` | re-approve target, retire current, audit | ✅ (token) |

Why this matters architecturally: `rollback()` already **enforces** "must be retired + must have eval evidence" — but a client only discovers that by trial-and-error via a `ValueError`. Surfacing eligible targets up front turns a hidden constraint into a visible affordance. And because the docs flag "output contract, not just the template string" as the subtle trap, `--dry-run` shows the exact `output_schema` keys that change (`format`, `length_policy`, `citation_policy`, `refusal_string`) so the cost of the change is unambiguous **before committing**. The confirmation + summary is purely a UX layer on top — it adds **zero** new security surface; fail-closed `ADMIN_TOKEN` auth is unchanged.

---

## 3. Guardrails already enforced by the registry (surfaced in CLI output)

| Guardrail | Where | Behavior if violated |
|-----------|-------|----------------------|
| Only previously-released versions | `rollback` checks `status == "retired"` | `ValueError` — you cannot roll back to a never-released or never-approved draft |
| Eval evidence required | `_has_required_evidence` needs `accuracy`, enforced on promote | Cannot approve a version without measured accuracy |
| No scoring a retired version | `record_eval_scores` rejects retired | After rollback, the **newly retired** 1.1.0 cannot be re-scored — intended |
| Atomic persistence | `save()` = temp file + `os.replace` + `fsync` | Crash-safe: no partial writes, no torn registry |
| **Visible targets (NEW)** | `prompt list` / `rollback --dry-run` surface retired-eligible versions + the `output_schema` diff *before* any change | Client never guesses a version; the rollback's real effect is shown up front (read-only, or dry-run with token) |

> **In practice (validated):** `prompt current` → shows `Live: RAG_ANSWER_V1.1.0` with its markdown output contract + `Eval acc: 0.8500`; `prompt list` → marks `1.0.0 retired ... <-- rollback target`; `rollback --dry-run` → prints `Output-contract changes: citation_policy, format, length_policy, refusal_string` and applies **nothing**; `rollback --yes` → `Done. New approved version: RAG_ANSWER_V1.0.0`, after which `prompt current` shows the restored 1.0.0 text contract. A no-token call rejects with exit 1 (fail closed), and answering `n` at `Proceed? [y/N]` cancels (exit 1).

---

## 4. Recommended sequencing for Step 3 vs Step 7

| Tier | Concern | Step 3 (this step) | Step 7 (production/guardrails) |
|------|---------|--------------------|--------------------------------|
| T1 | Live switch | `rollback` CLI command on local `prompt_registry.json` ✅ | Same, but against shared store |
| T2 | Consistency (multi-node) | N/A (single process) | **Shared versioned store** (Postgres/config-DB); resolve active prompt at read time; signed active pointer |
| T3 | Trust | `ADMIN_TOKEN` action gate ✅ | Full RBAC/IAM + config signing + audit export |

**Recommendation:** keep Step 3 on local JSON + `ADMIN_TOKEN`. This step's goal is *framework comparison*, not distributed config. Defer shared-store + RBAC + signing to Step 7 (documented here so the upgrade path is explicit).

---

## 5. Output contract: what rollback actually reverts (junior explainer) NEW

Rollback is not just "swap the template string back." A prompt version is a **whole behavior contract** = template + policy + **output rules**. When you roll back 1.1.0 → 1.0.0, you must also restore *how the answer is produced and presented*, otherwise the rollback is incomplete.

### The `output_schema` (part of every version record)
```
output_schema: {
  format:          "markdown",                  # how the LLM renders answer text
  shape:           {answer: str, sources: [...]},  # machine return the PIPELINE guarantees
  length_policy:   "4-6 sentences",             # verbosity rule
  citation_policy: "inline [source-N] markers", # how sources are cited (for faithfulness)
  refusal_string:  "I don't have enough context...",  # versioned no-context fallback
  display:         "answer, then Sources list", # guidance for consumers (non-visual)
}
```

### Mental model — two kinds of "display" (don't mix them up)
| Kind | Owned by | Versioned? | Example |
|------|----------|-----------|---------|
| **Output contract** (what/how the LLM answers) | `output_schema` in the registry | ✅ Yes — bumped, re-eval'd, rolled back | length, citations, format, refusal |
| **Presentation** (how the UI paints it) | `app.py` / CLI | ❌ No — plain app code | colors, widgets, terminal styling |

A junior's #1 mistake is putting presentation (CSS/Streamlit) into the registry. **Only the contract goes in the registry.**

### The two QA guardrails for output
1. **G1 — No silent output change.** `set_output_schema()` on an **approved** version forces it back to **`testing`** → it must be re-scored (re-eval) before it can be re-approved. Editing output rules is a *user-visible behavior change*, so it's gated exactly like a release. A history entry records `{from_schema → to_schema, reason}` for QA.
2. **G2 — Rollback restores full behavior.** Because the generator reads `refusal_string` (and the adapter injects `format`/`length`/`citation` rules) from the **approved version's `output_schema`**, rolling back 1.1.0 → 1.0.0 restores 1.0.0's *entire* output behavior, not just the template. This is what makes a rollback trustworthy.

```
   rollback 1.1.0 → 1.0.0
   ──────────────────────────────────────────────
   Registry: 1.0.0 becomes "approved" (get() serves it)
   Generator: refusal_response() now reads 1.0.0's refusal_string
   Adapter:   build_chain() injects 1.0.0's format/length/citation rules
   Result:    users see 1.0.0's full output behavior again ✅
```

---

## 6. Decision log

| # | Decision | Status |
|---|----------|--------|
| D1 | Rollback lives at the **application layer** (registry + CLI), not git or HF | ✅ Adopted |
| D2 | Security = **action-boundary auth** (`ADMIN_TOKEN`), never in a prompt string | ✅ Adopted |
| D3 | Git repo = **DR/audit backup** of `prompt_registry.json`, not the live switch | ✅ Adopted |
| D4 | Shared signed store + RBAC/IAM deferred to **Step 7** | ✅ Adopted (documented upgrade path) |
| D5 | The **output contract** (`output_schema`) is versioned prompt policy → rides on each version, gets eval + rollback | ✅ Adopted |
| D6 | Editing `output_schema` on an approved version forces it back to `testing` (no silent change) | ✅ Adopted |
| D7 | Output = **text/markdown** answer + machine `sources` structure (not strict-LLM JSON) | ✅ Adopted (chat UX + parse-safety) |
| D8 | Expose **read-only visibility** (`prompt current` / `prompt list`) so clients see what's live and the rollback-eligible targets up front | ✅ Adopted (no auth required) |
| D9 | `rollback --dry-run` shows the `output_schema` diff **without executing** | ✅ Adopted (client sees the output-contract cost pre-commit) |
| D10 | `rollback` gates on an interactive `Proceed? [y/N]` confirmation, with `--yes` for scripting/CI | ✅ Adopted (UX layer, zero new security surface) |
| D11 | The `prompt`/`dry-run`/confirm layers convert a hidden validation constraint into a visible affordance; `ADMIN_TOKEN` fail-closed auth unchanged | ✅ Adopted |
