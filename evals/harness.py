"""Shared harness for the component/pipeline/application eval suite.

Adapted from the CampusX rag-eval-deepeval reference project for
step3_langchain_rag's GroqJudge architecture.

Usage from eval scripts::

    from evals.harness import load_goldens, summarize_by_metric, print_summary
"""

from __future__ import annotations

import json
from typing import Any


def load_goldens(path: str) -> list[dict[str, Any]]:
    """Read a golden JSON file (list of dicts)."""
    with open(path) as f:
        return json.load(f)


def summarize_by_metric(result: Any) -> dict[str, Any]:
    """DeepEval ``EvaluationResult`` → per-metric summary.

    Returns ``{metric_name: {n, pass_rate, avg_score, min_score, max_score}}``.

    The extractor is defensive across DeepEval versions: test results may live
    on ``.test_results`` (or be a bare list), and per-test metrics on
    ``.metrics_data`` (newer) or ``.metrics`` (older). Pass/fail uses each
    metric's own ``success`` flag so it stays correct whether higher or lower
    score is "good".
    """
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets: dict[str, dict[str, Any]] = {}
    for tr in test_results:
        metrics = (
            getattr(tr, "metrics_data", None)
            or getattr(tr, "metrics", None)
            or []
        )
        for m in metrics:
            name = getattr(m, "name", "unknown")
            b = buckets.setdefault(
                name, {"scores": [], "passed": 0, "total": 0}
            )
            b["total"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                b["scores"].append(score)
            if getattr(m, "success", False):
                b["passed"] += 1

    summary: dict[str, dict[str, Any]] = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "pass_rate": (100 * b["passed"] / b["total"]) if b["total"] else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary


def print_summary(title: str, summary: dict[str, Any]) -> None:
    """Readable per-metric recap, printed under DeepEval's own report."""
    print("\n" + "=" * 60)
    print(f"{title}  (per-metric summary)")
    print("=" * 60)
    for name, s in summary.items():
        avg = (
            f"{s['avg_score']:.2f}"
            if s["avg_score"] == s["avg_score"]
            else "nan"
        )
        print(
            f"  {name:<26} pass_rate={s['pass_rate']:5.0f}%  "
            f"avg={avg}  n={s['n']}"
        )
    print("=" * 60)
