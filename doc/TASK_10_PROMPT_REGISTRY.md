# Task 10: Shared PromptRegistry + LangChain Adapter ⭐ (heart of Step 3)

## Objective
Create `src/langchain_rag/prompt_registry.py` (the **shared, cross-cutting** registry adapted from Step 2) and `src/langchain_rag/prompts.py` (the `LangChainPromptAdapter` that bridges the registry to LangChain's `PromptTemplate` + chat models). This is the single most important insight of Step 3.

## Why this matters (mentor note)
LangChain has **no concept** of a versioned prompt lifecycle with eval evidence — `PromptTemplate` is a dead-simple string template. So:
- **The registry is the source of truth** for Content (template), Policy (model/temperature), and Evidence (eval scores, run log).
- **LangChain is just the execution layer.** `PromptTemplate` + the chat model are built *from* the registry.
- This is the ⭐ insight: **Steps 1–7 all import the same registry; only the consumer changes.** LangChain is just another consumer (via the adapter).

```
Your Registry (source of truth)          LangChain (adapter)
─────────────────────────────            ──────────────────
Content: template string        ──→      PromptTemplate.from_template()
Policy: model, temperature      ──→      ChatOpenAI(model=..., temperature=...)
Output: output_schema (rules)   ──→      injected into the rendered prompt
Evidence: eval scores, logs     ──→      (not in LangChain — stays in registry)
```

### What is `output_schema`? (junior explainer) ⭐ NEW
Every prompt version should also answer: **"what is the LLM *allowed/required* to produce?"** A prompt that just says `"Answer from context"` gives you unpredictable answers — sometimes a JSON blob, sometimes a wall of text, sometimes with/without citations. That makes evaluation (`faithfulness`, `answer_relevancy`) fragile and makes rollback meaningless (you can't tell if the *output format* regressed).

So each version record carries an **`output_schema`** — a small spec of the output rules the LLM must honor:

| Field | Meaning (plain words) |
|-------|------------------------|
| `format` | `text` / `markdown` / `json` — how the *answer text* should be rendered |
| `shape` | the machine return the **pipeline** guarantees (e.g. `{answer: str, sources: [...]}`) — NOT what we ask the LLM to emit as JSON |
| `length_policy` | verbosity, e.g. `"4-6 sentences"` |
| `citation_policy` | how to cite sources, e.g. inline `[source-N]` markers matching retrieved doc ids |
| `refusal_string` | the no-context fallback message (so it is **versioned**, not hard-coded) |
| `display` | guidance for consumers (answer, then a Sources list) — **non-visual**; real UI styling stays in the app |

**The golden rule (architect/QA):** the *output contract* is **versioned prompt policy** → it lives in the registry, gets **eval + rollback** treatment. The *visual* layout (colors, widgets) is **unversioned app code** in `app.py` — never in the registry.

**Why this is QA-critical:** an output-rule change (e.g. `length_policy` 3→6 sentences, or format text→markdown) changes what users see and what the judge scores → it must be a **1.1.0 change → re-eval → rollback if it regresses**, exactly like a template change. That's why `output_schema` rides on the *version*, not on the whole prompt_id.

## Files to Create
- `/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/prompt_registry.py`
- `/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/prompts.py`

## Part A: `prompt_registry.py` (write fresh, adapted from Step 2)

Re-write the `PromptRegistry` by hand (do **not** copy Step 2's file) — internalize its behavior. Same methods as Step 2:

```python
class PromptRegistry:
    NEXT_STATUS = {"draft": "testing", "testing": "approved", "approved": "retired"}
    REQUIRED_EVAL_KEYS = ("accuracy",)

    def _timestamp(self) -> str: ...
    def _version_key(self, version) -> tuple[int,int,int]: ...
    def _format_version(self, key) -> str: ...
    def _has_required_evidence(self, record) -> bool: ...
    def _activate(self, key, reason, action) -> dict: ...
    def save(self, path="prompt_registry.json") -> None: ...   # atomic tmp + os.replace
    def load(self, path="prompt_registry.json") -> bool: ...
    def _record_history(self, key, from_status, to_status, reason) -> None: ...
    def _increment_version(self, version) -> str: ...
    def _next_minor(self, version) -> str: ...
    def _current_approved(self, prompt_id) -> dict | None: ...
    def register(self, prompt_id, template, input_variables, model=..., temperature=0.1, change_note="", output_schema=None) -> str: ...
    def set_output_schema(self, key, schema, reason="") -> None: ...   # NEW
    def get(self, prompt_id, *, version=None, approved_only=True) -> dict: ...
    def promote(self, key, reason="") -> dict: ...
    def rollback(self, prompt_id, to_version, reason="") -> dict: ...
    def list_versions(self, prompt_id) -> list: ...
    def get_status_history(self, key) -> list: ...
    def record_eval_scores(self, key, scores, eval_run_id) -> None: ...
    def log_run(self, key, rendered_hash, retrieved_doc_ids, output, latency_ms, token_usage, error=None) -> None: ...
    def compare_versions(self, prompt_id) -> list: ...
```

Key behaviors to reproduce (the quality gaps fixed in Step 2):
1. **`_activate()`** — the ONE place that grants `approved` status; shared by `promote()` and `rollback()` (no duplicated retire-then-approve logic).
2. **Evidence gate** — `_activate()` calls `_has_required_evidence()` and raises `ValueError` if `required (accuracy)` eval keys are missing before promotion to approved.
3. **`record_eval_scores`** — sets `{**scores, evaluated_at, eval_run_id}`; raises if key missing or prompt retired.
4. **`log_run`** — increments `run_count`, updates `last_run_at`, appends a run entry.
5. **Crash-safe save** — write temp file + `os.replace()` (atomic).
6. **`register`** — never overwrites; minor bump from approved, patch bump otherwise.
7. **`output_schema`** — (a) stored on the version when provided, else filled with a sane default; (b) **the no-silent-change gate (NEW, ⭐ QA guard): `set_output_schema()` on an `approved` version forces it back to `testing`** so it must be re-scored before it can be re-approved. A schema edit is a *behavior* change, so it must be gated exactly like a release.
8. **Schema-change history (NEW)** — `set_output_schema` appends a `history` entry of the form `{from_schema, to_schema, reason}` so QA can see exactly what output behavior changed on any rollback/update.

Default `model` for this step: `"Qwen/Qwen2.5-7B-Instruct-AWQ"` (the Step 2 Qwen), `temperature` default `0.1`.

Default `output_schema` (used when `register(...)` is called with `output_schema=None`):
```python
DEFAULT_OUTPUT_SCHEMA = {
    "format": "markdown",
    "shape": {"answer": "str", "sources": "list[dict]"},
    "length_policy": "4-6 sentences",
    "citation_policy": "inline [source-N] markers; N = index into retrieved sources",
    "refusal_string": "I don't have enough context to answer: {question}",
    "display": "answer text, then 'Sources:' list with title + score",
}
```

## Part B: `prompts.py` — `LangChainPromptAdapter` (+ `build_llm`)

```python
import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_rag.prompt_registry import PromptRegistry


def build_llm(model: str, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model, temperature=temperature,
        openai_api_base=os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1",
        openai_api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"),
        max_tokens=300,
    )


class LangChainPromptAdapter:
    """Bridges the shared PromptRegistry to LangChain's PromptTemplate + LLM.
    Registry owns the lifecycle; LangChain handles execution."""

    def __init__(self, registry: PromptRegistry) -> None:
        self.registry = registry

    def get_prompt_template(self, prompt_id: str) -> PromptTemplate:
        # return PromptTemplate(template=record["template"],
        #                      input_variables=record["input_variables"])

    def get_llm(self, prompt_id: str):
        # record = registry.get(prompt_id, approved_only=True)
        # return build_llm(record["model"], record["temperature"])

    def build_chain(self, prompt_id: str):
        # Build LCEL runnable: format context + question into the registry
        # template, then call the LLM. e.g. ChatPromptTemplate + RunnablePassthrough.
```

### Step 9.1: `build_llm`
Returns `ChatOpenAI` reading the env LLM keys (same Qwen as Step 2). This is the helper Task 8 needs.

### Step 9.2: `get_prompt_template`
Gets the **approved** version from the registry and wraps its `template`/`input_variables` in a `PromptTemplate`. Note: the registry template may reference `{context}` — decide how context gets injected (see `build_chain`).

### Step 9.3: `build_chain` — LCEL composition
The recommended design: the registry template is the **system** prompt; the context and question are mapped in via `RunnablePassthrough` + a formatter. Example shape:
```python
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", record["template"]),          # e.g. "Answer from context: {context}. {question}"
    ("human", "{question}"),
])
llm = build_llm(record["model"], record["temperature"])
chain = {
    "context": RunnablePassthrough() | (lambda x: format_context(x["sources"])),
    "question": RunnablePassthrough() | (lambda x: x["question"]),
} | prompt | llm
```
> ⭐ You have freedom here — the point is the **template + policy come from the registry**, never hard-coded in LangChain. The adapter proves: swap registry versions → chain changes without code edits.

## Usage example
```python
registry = PromptRegistry()
key = registry.register("RAG_ANSWER",
    "Answer ONLY from context: {context}\nQuestion: {question}",
    ["context", "question"], change_note="Initial V1",
    output_schema={
        "format": "markdown",
        "shape": {"answer": "str", "sources": "list[dict]"},
        "length_policy": "4-6 sentences",
        "citation_policy": "inline [source-N] markers",
        "refusal_string": "I don't have enough context to answer: {question}",
        "display": "answer text, then 'Sources:' list",
    })
registry.promote(key)                       # draft -> testing
registry.record_eval_scores(key, {"accuracy": 0.9, "faithfulness": 0.9}, "eval_1")
registry.promote(key)                       # testing -> approved

# ⭐ QA gate: editing the schema on an APPROVED version drops it back to testing
# (see the "no silent output change" rule) — forces a re-eval before re-approval.
registry.set_output_schema(key, {"length_policy": "3-4 sentences"}, reason="tighter length")
# key now has status "testing" again; history records the schema diff.

adapter = LangChainPromptAdapter(registry)
chain = adapter.build_chain("RAG_ANSWER")

# runtime evidence still tracked by the registry:
registry.log_run(key, rendered_hash="abc", retrieved_doc_ids=["python_basics.txt"],
                 output="Python is high-level...", latency_ms=120, token_usage={})
```

## Completion Criteria
- [ ] `prompt_registry.py` written fresh with all Step 2 methods + quality gates
- [ ] `prompts.py` has `build_llm` + `LangChainPromptAdapter`
- [ ] `build_chain` returns an LCEL runnable driven by the registry
- [ ] Registry (not LangChain) owns Content / Policy / Output / Evidence
- [ ] `register()` stores `output_schema` (defaults if `None`)
- [ ] `set_output_schema()` on an approved version forces it back to `testing` (no silent change) and records a schema-diff history entry
- [ ] `record_eval_scores` / `log_run` work; `_activate` gates on evidence
- [ ] Tests pass offline (no API), `ruff`/`mypy` clean

## Tests to add (⭐ the QA gate)
In `tests/test_prompt_registry.py`, besides the existing cases, add:
- `test_register_default_output_schema` — `register(..., output_schema=None)` yields the default schema.
- `test_set_output_schema_on_approved_demotes_to_testing` — after `promote`→approved, calling `set_output_schema` sets status back to `testing`.
- `test_set_output_schema_records_history_diff` — history contains `{change: "output_schema", from_schema, to_schema, reason}`.
- `test_rollback_restores_served_schema` — pushing 1.0.0→1.1.0 (different schema) then `rollback` to 1.0.0 makes `get()` serve 1.0.0's schema.

## Mocking (learner note)
- `PromptRegistry` is **pure logic** — test it for real (no mocks), using `tmp_path` for persistence, exactly like Step 2 (aim ~100% coverage).
- `LangChainPromptAdapter.build_chain` touches `ChatOpenAI` (would fire a network read on import?) — so in tests, **mock `build_llm`** to return a fake chat model object, or assert on the prompt/schema shape without invoking a real model. Patch where imported (`langchain_rag.prompts.build_llm`).
- `build_chain` should **inject the schema rules (`format`, `length_policy`, `citation_policy`) into the rendered prompt** — assert the prompt string contains them, mock the LLM.

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/prompt_registry.py src/langchain_rag/prompts.py
uv run mypy src/langchain_rag/prompt_registry.py src/langchain_rag/prompts.py
uv run pytest tests/test_prompt_registry.py -v
uv run pytest tests/test_prompt_registry.py -v --cov=langchain_rag.prompt_registry --cov-report=term-missing
uv run ruff check tests/test_prompt_registry.py
uv run mypy tests/test_prompt_registry.py
```

## Report Back
When done, tell me:
1. `compare_versions()` ordering example
2. `run_count` after a couple of `log_run()` calls
3. Confirmation the evidence gate (`_activate`) blocks promotion without `accuracy`
4. **One `prompt_registry.json` record showing `output_schema`** (paste the JSON)
5. **The schema-edit gate:** status of a version after `set_output_schema` on an approved version (should be `testing`)
6. Show me your `build_chain` — I want to review how you composed the LCEL chain (this is the Step 3 crown jewel)
7. Paste `prompt_registry.py` + `prompts.py` for review

## Next steps
After Task 10, Task 9 (pipeline) wires how these pieces — plus splitter/vectorstore/reranker — run a full `ask()` through LCEL.
