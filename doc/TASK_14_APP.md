# Task 14: Implement Streamlit App (Chat + Eval Dashboard + Traces)

## Objective
Create `app.py` — a Streamlit UI with **three tabs** mirroring Step 2: **Chat**, **Eval Dashboard**, **Traces**. It drives the LangChain `Pipeline` through the same interface.

## File to Create
`/home/dipak/agentic/step3_langchain_rag/app.py`

## What To Implement (mirrors Step 2)

### Tab 1: Chat
- Sidebar `Settings`: `top_k` slider (1–10), `min_score` slider (0.0–1.0, **default 0.3**)
- Sidebar `Index Stats`: chunk count (`pipeline.get_stats()["num_chunks"]`) + index-ready metric
- Sidebar `Clear Chat` button
- Chat history via `st.session_state.messages`
- `st.chat_input` → `pipeline.ask(prompt, top_k, min_score)`
- Expandable "Sources": `**[i]** (Score: {score:.2f})` + 200-char preview
- `prompt_key` caption under the answer
- **Chat-log persistence (independent of LangSmith):** append each exchange to `data/chat_logs/chat_log_<YYYY_MM_DD>.jsonl` (gitignored)

### Tab 2: Eval Dashboard
- **"Run Evaluation"** button → `evaluate_with_registry(pipeline, "eval/golden.jsonl", registry, prompt_key)` → saves `eval/results/eval_<ts>.json`
- Metrics summary: 4 `st.metric` gauges
- Per-question `st.dataframe`
- Failed-questions banner (below-threshold)
- Compare-two-runs dropdowns (`st.selectbox` with `cmp_a`/`cmp_b` keys)
- **DeepEval expander** (Task 16): `top_k`, case limit, "skip reasoning" checkbox, **Run DeepEval** button that launches `eval/deepeval_suite/evaluate.py` as a **subprocess** (DeepEval's import-time `signal.signal()` needs a main thread; Streamlit runs in a non-main thread — so don't import it inline). Render persisted `deepeval_*.json` summary gauges.

### Tab 3: Traces
- LangSmith link: `https://smith.langchain.com/project/step3-langchain-rag`
- Note traces only appear with a real `LANGSMITH_API_KEY`
- Explain per-`ask()` tracing

### Caching
```python
@st.cache_resource
def get_pipeline() -> Pipeline:
    p = Pipeline()
    p.load_documents()
    return p
```

## Implementation notes (carried from Step 2)
- `_save_results`/`_load_results`/`_load_deepeval_results`/`_chat_log_path`/`_write_chat_log` helpers
- `evaluate_with_registry` includes `"accuracy"` so the registry evidence gate can later approve the version
- golden fields `answer`/`source` or `gold_answer`/`gold_source` via `.get(...)`

## Verify
```bash
cd /home/dipak/agentic/step3_langchain_rag
uv run ruff check app.py
uv run mypy app.py
uv run python -c "import ast; ast.parse(open('app.py').read()); print('app.py OK')"
uv run pytest tests/ -v
```

## Completion Criteria
- [ ] `app.py` with 3 tabs
- [ ] Chat works (think, sources, prompt_key, chat log)
- [ ] Eval Dashboard runs eval + gauges + per-question table + compare
- [ ] DeepEval expander runs as subprocess and shows results
- [ ] Traces tab shows LangSmith link
- [ ] Pipeline cached (no rebuild per click)

## Report Back
When done, tell me:
1. All 3 tabs render
2. Eval Dashboard shows metrics + table
3. One chat exchange produced a chat-log line in `data/chat_logs/`
4. Paste your `app.py` for review
