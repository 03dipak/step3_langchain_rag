# Task 11: Implement Tracer (LangSmith) ⭐ NEW

## Objective
Create `src/langchain_rag/tracer.py` — a thin LangSmith integration that wraps pipeline steps so every `ask()` is traceable, and **degrades gracefully** when no real `LANGSMITH_API_KEY` is set (silent no-op → offline tests & non-LangSmith environments still work). This mirrors Step 2's `tracer.py`.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/tracer.py`

## What You Need To Do

### Functions (same contract as Step 2)
```python
LANGSMITH_PLACEHOLDER = "ls_your_key_here"

def tracing_enabled() -> bool: ...
    # LANGSMITH_API_KEY set AND != placeholder

def run_with_trace(name, fn, *args, **kwargs): ...
    # disabled -> fn(*args, **kwargs); enabled -> langsmith.traceable(name=name)(fn)(*args, **kwargs)

def trace_retrieve(fn, question, top_k): ...
def trace_generate(fn, question, context_chunks): ...
def setup_tracing(project="step3-langchain-rag") -> None: ...
    # only when enabled: set LANGCHAIN_TRACING_V2=true, LANGCHAIN_PROJECT
```

### Step 10.1-10.4
- `tracing_enabled()` — checks the env var is not missing/placeholder.
- `run_with_trace()` — the core; **lazy-imports** `langsmith.traceable` inside the `if enabled` branch so the module imports even without langsmith (keeps it light).
- `trace_retrieve`/`trace_generate` — thin wrappers delegating to `run_with_trace`.
- `setup_tracing()` — sets `LANGCHAIN_TRACING_V2` + `LANGCHAIN_PROJECT=step3-langchain-rag` when a real key is present.

### Step 10.5: Wire into pipeline
Update `pipeline.ask()` to wrap retrieve/generate:
```python
from langchain_rag.tracer import trace_retrieve, trace_generate

results = trace_retrieve(lambda: self.retriever.retrieve(question, top_k, min_score), question, top_k)
output  = trace_generate(lambda: self.generator.generate(question, results), question, results)
```
These are transparent no-ops when disabled (same return value, zero overhead).

## Completion Criteria
- [ ] `tracer.py` created
- [ ] `tracing_enabled()` correct for missing/placeholder/real key
- [ ] `trace_retrieve`/`trace_generate` return the wrapped result unchanged
- [ ] `setup_tracing()` sets env only when enabled
- [ ] No crash / no LangSmith warning with placeholder key
- [ ] `pipeline.py` imports from `tracer.py` directly
- [ ] 100% test coverage (`--cov=langchain_rag.tracer`)

## Mocking (learner note)
Mock `tracing_enabled` to `False`/`True` to control the gate without a real key. The `langsmith.traceable` import is lazy (inside `run_with_trace` when enabled), so the module imports fine offline.

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/tracer.py src/langchain_rag/pipeline.py tests/test_tracer.py
uv run mypy src/langchain_rag/tracer.py src/langchain_rag/pipeline.py tests/test_tracer.py
uv run pytest tests/test_tracer.py -v --cov=langchain_rag.tracer --cov-report=term-missing
uv run pytest tests/ -v
```

## Report Back
When done, tell me:
1. `tracing_enabled()` result with the placeholder key
2. Result returned by `trace_retrieve`
3. Whether you saw a real LangSmith trace (optional, only if you added a key)
4. Paste your `tracer.py` for review
