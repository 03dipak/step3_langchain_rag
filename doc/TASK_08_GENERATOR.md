# Task 8: Implement Generator (LangChain LCEL + registry adapter)

## Objective
Create `src/langchain_rag/generator.py` — turns retrieved context + question into an answer using **LangChain's `ChatOpenAI`** pointed at the **same OpenAI-compatible Qwen endpoint as Step 2** (via `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`), composed with an LCEL chain. This is where the `LangChainPromptAdapter` (Task 10) starts paying off.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/src/langchain_rag/generator.py`

## The comparison (Step 2 openai SDK vs Step 3 LangChain)
Step 2 built messages by hand and called `client.chat.completions.create(...)`. Step 3 uses LangChain chat models + LCEL:
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
    temperature=0.0,
    openai_api_base=os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1",
    openai_api_key=os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY"),
    max_tokens=300,
)
```
> ⭐ **Why `openai_api_base`/`openai_api_key` and not `base_url`?** In LangChain 1.x, `ChatOpenAI` exposes the base URL/key via the `openai_api_base`/`openai_api_key` model fields (legacy aliases), not `base_url`. This is a common gotcha — the OpenAI-compatible gateway still works, just under the older field names. Document in `notes.md`.

## What You Need To Do

### Step 7.1: `LangChainGenerator`
Same public surface as Step 2's generator so the evaluator/app are unchanged:
```python
class LangChainGenerator:
    def __init__(self, registry, prompt_id="RAG_ANSWER") -> None:
        self.registry = registry
        self.adapter = LangChainPromptAdapter(registry)
        self.chain = self.adapter.build_chain(prompt_id)   # LCEL: prompt | llm

    def generate(self, question, context_chunks) -> str: ...
        # chain.invoke({"sources": context_chunks, "question": question})
        # return .content (AIMessage) or str(result)

    def refusal_response(self, question) -> str: ...
```

### Step 7.2: Build the LLM (helper, used by the adapter)
Write a `build_llm(model, temperature)` helper (probably in `prompts.py`, used by `LangChainPromptAdapter`) that returns a `ChatOpenAI` reading env as above. Task 8 depends on this existing — do Task 10's `prompts.py` first, or at least the `build_llm` part.

### Step 7.3: `generate`
Run the LCEL chain over `{"sources": context_chunks, "question": question}`. The chain (from the adapter) formats the registry template with the context and question, calls the LLM, and returns an `AIMessage` — extract `.content` and `.strip()`.

### Step 7.4: `refusal_response` (versioned, from `output_schema`) NEW
Instead of hard-coding the string, **read it from the approved version's `output_schema.refusal_string`**:
```python
def refusal_response(self, question) -> str:
    record = self.registry.get(self.prompt_id, approved_only=True)
    return record["output_schema"].get(
        "refusal_string", "I don't have enough context to answer: {question}"
    ).format(question=question)
```
Why: the no-context fallback is part of the *output concept* — it should be **versioned and rolled back** with the prompt, not a magic string buried in code. When you rollback 1.1.0→1.0.0, the refusal text reverts too. (This is the output-contract design from Task 10 / the architecture doc.)

> The `citation_policy` / `length_policy` / `format` rules are injected by the adapter's `build_chain` (Task 10) into the rendered prompt — `generate()` itself just runs the chain and returns the answer text.

## Testing
> ⚠️ **Live test needs your `.env` keys in the shell.** `python -c` does **not**
> auto-load `.env` (python-dotenv is a dependency but isn't wired into the package
> yet). Source `.env` into the process env first (`set -a && . ./.env && set +a`),
> otherwise `ChatOpenAI` raises `OpenAIError: Missing credentials` inside
> `LangChainGenerator.__init__` (it eagerly calls `build_chain` → `build_llm`).

```bash
cd /home/dipak/agentic/step3_langchain_rag
set -a && . ./.env && set +a   # load LLM keys into the shell
uv run python -c "
from langchain_rag.generator import LangChainGenerator
from langchain_rag.prompt_registry import PromptRegistry

r = PromptRegistry()
key = r.register('RAG_ANSWER', 'Answer from context only: {context}\nQuestion: {question}', ['context','question'], change_note='V1')
r.promote(key); r.record_eval_scores(key, {'accuracy': 0.9}, 'e1'); r.promote(key)  # -> approved
g = LangChainGenerator(r)
chunks = [{'text':'Python is high-level.','metadata':{'source':'python_basics.txt'},'score':0.9}]
print('--- live (needs .env LLM keys) ---')
print(g.generate('What is Python?', chunks)[:200])
print(g.refusal_response('hi'))
"
```
**Expected:** a numbered-context LCEL answer (with keys), and the refusal string.

## Completion Criteria
- [ ] `generator.py` created with `LangChainGenerator`
- [ ] `generate()` runs an LCEL chain and returns a string
- [ ] LLM is `ChatOpenAI` reading `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL` (fallback Groq)
- [ ] `refusal_response()` reads the **approved version's `output_schema.refusal_string`** (not hard-coded)
- [ ] `ruff` + `mypy --strict` pass

## Mocking (learner note) — mock the LLM/chain ⭐
`generate()` calls a live LLM = the biggest external boundary. In offline tests, mock the adapter/chain:

```python
# patch where the name is imported
mock_chain = mocker.MagicMock()
mock_chain.invoke.return_value = type("AIMessage", (), {"content": "ok"})()
# inject a generator whose chain is the mock
```
Or mock `LangChainPromptAdapter.build_chain` to return a fake that returns a canned `AIMessage`. Key rule stays: **patch where it's looked up**, use the `return_value` chain for `.invoke()`, and don't let any real API call fire offline.

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check src/langchain_rag/generator.py
uv run mypy src/langchain_rag/generator.py
uv run pytest tests/test_generator.py -v
uv run pytest tests/test_generator.py -v --cov=langchain_rag.generator --cov-report=term-missing
uv run ruff check tests/test_generator.py
uv run mypy tests/test_generator.py
```

## Report Back
When done, tell me:
1. First 200 chars of a live answer (or note if keys invalid)
2. `refusal_response()` output
3. The exact `ChatOpenAI` field names that worked (`openai_api_base` vs `base_url`) — confirm the gotcha
4. Paste your `generator.py` (and the `build_llm` helper) for review
