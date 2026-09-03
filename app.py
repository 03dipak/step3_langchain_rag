import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from evaluator import evaluate_with_registry

from langchain_rag.pipeline import Pipeline
from langchain_rag.prompt_registry import PromptRegistry


@st.cache_resource
def get_pipeline() -> Pipeline:
    p = Pipeline()
    p.load_documents()
    return p


def _save_results(results: dict[str, Any]) -> Path:
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"eval_{ts}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    return path


def _load_results() -> list[dict[str, Any]]:
    results_dir = Path("eval/results")
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("eval_*.json"), reverse=True)
    return [{"path": f, "data": json.loads(f.read_text())} for f in files]


def _load_deepeval_results() -> list[dict[str, Any]]:
    results_dir = Path("eval/results")
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("deepeval_*.json"), reverse=True)
    return [{"path": f, "data": json.loads(f.read_text())} for f in files]


def _chat_log_path() -> Path:
    log_dir = Path("data/chat_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"chat_log_{datetime.now(UTC).strftime('%Y_%m_%d')}.jsonl"


def _write_chat_log(entry: dict[str, Any]) -> None:
    """Append one user/assistant exchange to today's JSONL chat log."""
    with _chat_log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


st.set_page_config(page_title="RAG Eval Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Evaluation Assistant")
st.caption("Ask questions about the documents, run evaluation, and inspect traces.")

pipeline = get_pipeline()
registry = PromptRegistry()
registry.load()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k", 1, 10, 3)
    min_score = st.slider("Min relevance score", 0.0, 1.0, 0.3, 0.05)

    st.header("Index Stats")
    stats = pipeline.get_stats()
    st.metric("Chunks", stats["num_chunks"])
    st.metric("Index", "Ready" if stats["index_exists"] else "Missing")

    if st.button("Clear Chat"):
        st.session_state.messages = []

tab1, tab2, tab3 = st.tabs(["Chat", "Eval Dashboard", "Traces"])

# --- Tab 1: Chat ---
with tab1:
    st.header("Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(msg["sources"]):
                        st.write(f"**[{i+1}]** (Score: {src.get('score'):.2f})")
                        st.write(src.get("text", "")[:200] + "...")

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.ask(prompt, top_k=top_k, min_score=min_score)
                st.markdown(result["answer"])

            sources = result["sources"]
            if sources:
                with st.expander("Sources"):
                    for i, src in enumerate(sources):
                        st.write(f"**[{i+1}]** (Score: {src.get('score'):.2f})")
                        st.write(src.get("text", "")[:200] + "...")
            else:
                st.write("No sources retrieved.")
            st.caption(f"Prompt key: `{result.get('prompt_key')}`")

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": sources}
        )

        _write_chat_log(
            {
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "user_message": prompt,
                "agent_message": result["answer"],
                "prompt_key": result.get("prompt_key"),
                "config": {"top_k": top_k, "min_score": min_score},
                "num_sources": len(sources),
            }
        )

# --- Tab 2: Eval Dashboard ---
with tab2:
    st.header("Eval Dashboard")

    if st.button("Run Evaluation"):
        prompt_key = pipeline.registry.get("RAG_ANSWER")["key"]
        results = evaluate_with_registry(
            pipeline, "eval/golden.jsonl", registry, prompt_key
        )
        saved_path = _save_results(results)
        st.success(f"Evaluation complete. Results saved to `{saved_path}`")

    with st.expander("DeepEval (LLM-judged via Groq)"):
        st.write(
            "Runs DeepEval Contextual Recall / Precision / Faithfulness using the "
            "free Groq judge (`openai/gpt-oss-120b`). LLM-judged, so it calls the "
            "judge for every case and metric."
        )
        deep_k = st.slider("deepeval top_k", 1, 6, 3, key="deep_k")
        deep_limit = st.number_input(
            "Limit cases (0 = all)", min_value=0, max_value=100, value=0, key="deep_limit"
        )
        deep_no_reason = st.checkbox(
            "Skip per-score reasoning (faster, lower token use)", value=True, key="deep_no_reason"
        )
        st.caption(
            "Free Groq tier is ~8000 tokens/min; a 20-case run needs many judge "
            "calls. Use a small limit and/or skip reasoning to stay under quota."
        )
        if st.button("Run DeepEval"):
            import subprocess
            import sys

            st.write("Running DeepEval in a separate process (Streamlit threads can't import it)…")
            args = [
                sys.executable,
                "eval/deepeval_suite/evaluate.py",
                "--k",
                str(deep_k),
            ]
            if deep_limit:
                args += ["--limit", str(deep_limit)]
            if deep_no_reason:
                args.append("--no-reason")
            proc = subprocess.run(args, capture_output=True, text=True, check=False)
            st.code(proc.stdout, language="text")
            if proc.returncode != 0:
                st.error(proc.stderr)
            else:
                st.success("DeepEval evaluation complete.")

    st.subheader("DeepEval Results")
    deeplogs = _load_deepeval_results()
    if deeplogs:
        latest = deeplogs[0]["data"]
        summary = latest.get("summary", {})
        cols = st.columns(3)
        for col, key in zip(cols, ("Contextual Recall", "Contextual Precision", "Faithfulness")):
            entry = summary.get(key) or {}
            total = entry.get("total", 0)
            rate = entry.get("pass_rate", 0)
            col.metric(key, f"{rate:.0%}", help=f"{entry.get('pass',0)}/{total} passed")
        st.caption(
            f"Judge: `{latest.get('judge_model')}` · "
            f"saved `{Path(deeplogs[0]['path']).name}`"
        )
    else:
        st.info("No DeepEval results yet.")

    st.subheader("Metrics Summary")
    loaded = _load_results()
    if loaded:
        latest = loaded[0]["data"]
        summary = latest.get("summary", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Context Recall", f"{summary.get('context_recall', 0):.2f}")
        col2.metric("Context Precision", f"{summary.get('context_precision', 0):.2f}")
        col3.metric("Faithfulness", f"{summary.get('faithfulness', 0):.2f}")
        col4.metric("Answer Relevance", f"{summary.get('answer_relevance', 0):.2f}")
    else:
        st.info("Run an evaluation to see metrics here.")

    st.subheader("Per-Question Results")
    if loaded:
        latest = loaded[0]["data"]
        rows = []
        for r in latest.get("results", []):
            row = {
                "question": r["question"],
                "answer": r["answer"][:80],
                "context_recall": round(r["context_recall"], 3),
                "context_precision": round(r["context_precision"], 3),
                "faithfulness": round(r["faithfulness"], 3),
                "answer_relevance": round(r["answer_relevance"], 3),
            }
            rows.append(row)
        if rows:
            st.dataframe(rows, width="stretch")
            failed = [
                r for r in latest["results"]
                if r["context_recall"] < 0.70
                or r["context_precision"] < 0.60
                or r["faithfulness"] < 0.80
                or r["answer_relevance"] < 0.70
            ]
            if failed:
                st.error(f"{len(failed)} question(s) below target thresholds")
                for f in failed:
                    st.write(f"- {f['question']}")
    else:
        st.info("Run an evaluation to see per-question results.")

    st.subheader("Compare Two Runs")
    files = _load_results()
    if len(files) >= 2:
        labels = [f"{f['path'].name}" for f in files]
        cmp_a = st.selectbox("Run A", range(len(files)), key="cmp_a")
        cmp_b = st.selectbox("Run B", range(len(files)), key="cmp_b")
        run_a = files[cmp_a]["data"]
        run_b = files[cmp_b]["data"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**{files[cmp_a]['path'].name}**")
            sa = run_a.get("summary", {})
            st.metric("Context Recall", f"{sa.get('context_recall', 0):.2f}")
            st.metric("Context Precision", f"{sa.get('context_precision', 0):.2f}")
            st.metric("Faithfulness", f"{sa.get('faithfulness', 0):.2f}")
            st.metric("Answer Relevance", f"{sa.get('answer_relevance', 0):.2f}")
        with col_b:
            st.write(f"**{files[cmp_b]['path'].name}**")
            sb = run_b.get("summary", {})
            st.metric("Context Recall", f"{sb.get('context_recall', 0):.2f}")
            st.metric("Context Precision", f"{sb.get('context_precision', 0):.2f}")
            st.metric("Faithfulness", f"{sb.get('faithfulness', 0):.2f}")
            st.metric("Answer Relevance", f"{sb.get('answer_relevance', 0):.2f}")
    else:
        st.info("Need at least 2 saved runs to compare.")

# --- Tab 3: Traces ---
with tab3:
    st.header("Traces")
    st.write("LangSmith traces are available when a real `LANGSMITH_API_KEY` is set.")
    project = "step3-langchain-rag"
    st.write(f"Project: `{project}`")
    st.write(
        f"[Open in LangSmith](https://smith.langchain.com/project/{project})"
    )
    st.write(
        "Every `pipeline.ask()` call is automatically traced when tracing is enabled. "
        "Use the **Traces** tab in LangSmith to inspect retrieval and generation spans."
    )
