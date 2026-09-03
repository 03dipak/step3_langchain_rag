"""DeepEval-based evaluation suite for the step3 LangChain RAG pipeline.

Judge-free metrics aside, this uses DeepEval's LLM-judged contextual
precision/recall and faithfulness metrics with a Groq judge (free tier).

Run:  uv run python eval/deepeval_suite/evaluate.py [--k N] [--limit N] [--no-reason]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepeval import evaluate
from deepeval.metrics import (  # type: ignore[attr-defined]
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase  # type: ignore[attr-defined]

from deepeval_suite.judge import GroqJudge
from langchain_rag.pipeline import Pipeline

GOLDEN_PATH = Path("eval/golden.jsonl")
RESULTS_DIR = Path("eval/results")
THRESHOLD = 0.7


def save_results(results: Any, k: int, limit: int | None, no_reason: bool) -> Path:
    """Persist DeepEval results to eval/results/deepeval_<timestamp>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary: dict[str, list[float]] = {}
    for tr in getattr(results, "test_results", []) or []:
        for m in getattr(tr, "metrics_data", []) or []:
            name = m.name
            summary.setdefault(name, []).append(float(m.score))
    pass_rates = {
        name: (sum(s >= THRESHOLD for s in scores), len(scores))
        for name, scores in summary.items()
    }
    payload = {
        "engine": "deepeval",
        "judge_model": "openai/gpt-oss-120b",
        "config": {"k": k, "limit": limit, "no_reason": no_reason, "threshold": THRESHOLD},
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": {
            name: {"pass": p, "total": t, "pass_rate": round(p / t, 4) if t else 0.0}
            for name, (p, t) in pass_rates.items()
        },
    }
    path = RESULTS_DIR / f"deepeval_{ts}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_golden(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    """Load golden.jsonl into a list of dicts."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def build_test_cases(
    entries: list[dict[str, Any]], pipeline: Pipeline, k: int = 3
) -> list[LLMTestCase]:
    """Run retrieval + generation on each golden entry and wrap in LLMTestCase."""
    if pipeline.retriever is None:
        raise ValueError("Pipeline retriever not initialized.")

    cases = []
    for entry in entries:
        question = entry["question"]
        results = pipeline.retriever.retrieve(question, top_k=k)
        retrieval_context = [r["text"] for r in results]
        answer = pipeline.generator.generate(question, results)
        cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=entry.get("answer", entry.get("gold_answer", "")),
                retrieval_context=retrieval_context,
            )
        )
    return cases


def run_eval(
    entries: list[dict[str, Any]],
    pipeline: Pipeline,
    k: int = 3,
    judge: Any | None = None,
    include_reason: bool = True,
) -> Any:
    """Run DeepEval metrics and print/return the results object."""
    judge = judge or GroqJudge()
    metrics = [
        ContextualRecallMetric(model=judge, threshold=THRESHOLD, include_reason=include_reason),
        ContextualPrecisionMetric(model=judge, threshold=THRESHOLD, include_reason=include_reason),
        FaithfulnessMetric(model=judge, threshold=THRESHOLD, include_reason=include_reason),
    ]
    test_cases = build_test_cases(entries, pipeline, k=k)
    print(f"Judging {len(test_cases)} cases with {judge.get_model_name()}...")
    return evaluate(test_cases, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="chunks retrieved per query")
    parser.add_argument("--limit", type=int, default=None, help="only first N golden entries")
    parser.add_argument(
        "--no-reason",
        action="store_true",
        help="skip per-score reasoning to cut token usage (free-tier TPM friendly)",
    )
    args = parser.parse_args()

    entries = load_golden()
    if args.limit:
        entries = entries[: args.limit]

    pipeline = Pipeline()
    pipeline.load_documents()
    results = run_eval(entries, pipeline, k=args.k, include_reason=not args.no_reason)
    path = save_results(results, k=args.k, limit=args.limit, no_reason=args.no_reason)
    print(f"Saved DeepEval results to {path}")


if __name__ == "__main__":
    main()
