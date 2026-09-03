"""Application-level quality evaluation: G-Eval (Correctness / Completeness / Style).

Uses DeepEval's GEval with explicit evaluation steps, rubrics, and
probability-weighted scoring (strict_mode=False) for deterministic,
repeatable judge scores.

    Correctness  — reference-based, judges TRUTH (not coverage or length)
    Completeness  — reference-based, judges COVERAGE (not correctness)
    Style         — reference-free, judges TONE ONLY

Usage::

    python -m evals.eval_application
    python -m evals.eval_application --limit 5 --no-reason
"""

from __future__ import annotations

import argparse

from deepeval import evaluate  # type: ignore[attr-defined]
from deepeval.metrics import GEval  # type: ignore[attr-defined]
from deepeval.metrics.g_eval import Rubric  # type: ignore[attr-defined]
from deepeval.test_case import (  # type: ignore[attr-defined]
    LLMTestCase,
    LLMTestCaseParams,
)
from dotenv import load_dotenv

from evals.harness import load_goldens, print_summary, summarize_by_metric
from langchain_rag.pipeline import Pipeline

load_dotenv()

GOLDEN_PATH = "eval/goldens/correctness_goldens.json"
THRESHOLD = 0.7


def _build_correctness() -> GEval:
    """Correctness: is the answer factually accurate relative to the golden answer?"""
    return GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            (
                "A claim is wrong only if it CONTRADICTS the expected output or is "
                "factually false. Judge truth, not completeness."
            ),
            (
                "A factually accurate answer must score at least 0.9 even if it is "
                "shorter or covers fewer points than the expected output."
            ),
            (
                "Do NOT deduct for brevity, missing elaboration, or omitted points — "
                "omissions are not errors here."
            ),
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "All stated claims are factually correct and consistent. "
                    "No contradictions. Brevity is fine."
                ),
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Mostly correct but one minor inaccuracy.",
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "Contains a clear factual error or a claim that contradicts "
                    "the expected output."
                ),
            ),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        strict_mode=False,
    )


def _build_completeness() -> GEval:
    """Completeness: does the answer cover all key points from the golden answer?"""
    return GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            (
                "Penalize the actual output for each key point from the expected "
                "output that it omits or only partially covers."
            ),
            (
                "Judge coverage only. Do NOT lower the score because a covered point "
                "is stated incorrectly — factual correctness is judged separately."
            ),
            (
                "Do NOT penalize the actual output for adding extra information "
                "beyond the expected output."
            ),
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "Addresses essentially all key points in the expected output."
                ),
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="Covers the main key points but misses one or more.",
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "Misses several key points; only partially covers the expected output."
                ),
            ),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        strict_mode=False,
    )


def _build_style() -> GEval:
    """Style: reference-free tone check (no expected_output needed)."""
    return GEval(
        name="Style",
        evaluation_steps=[
            (
                "Judge only the teaching style and tone of the actual output, not "
                "whether it is factually correct or complete."
            ),
            (
                "Reward an intuitive, explanatory tone: plain language, the idea "
                "explained before any formula or jargon, and technical terms briefly "
                "unpacked when used."
            ),
            (
                "Reward a direct, conversational register written in prose, as a "
                "teacher explains it out loud, rather than a dry, formal, or "
                "bullet-list tone."
            ),
            (
                "An analogy or concrete example is a BONUS when the concept is "
                "abstract, but a clear, direct, well-explained answer is fully "
                "acceptable and must NOT be penalized for not having one."
            ),
            (
                "Penalize answers that are stiff, bureaucratic, structured as a "
                "bare list with no explanation, or that use unexplained jargon."
            ),
            (
                "Do NOT reward or penalize based on correctness, completeness, or "
                "length — only on style and tone."
            ),
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "Clearly in a teaching voice: intuitive, conversational prose "
                    "that explains before it formalizes."
                ),
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "Clear, conversational, and well-explained in prose. Fully "
                    "acceptable even without an analogy or example."
                ),
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places.",
            ),
            Rubric(
                score_range=(0, 3),
                expected_outcome=(
                    "Dry, stiff, bare-list, jargon-heavy, or robotic; does not "
                    "read like a teaching explanation."
                ),
            ),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=THRESHOLD,
        strict_mode=False,
    )


def run(pipeline: Pipeline, *, limit: int | None = None) -> dict:  # type: ignore[type-arg]
    """Run correctness + completeness + style on live pipeline.ask() output."""
    goldens = load_goldens(GOLDEN_PATH)
    if limit:
        goldens = goldens[:limit]

    test_cases = []
    for g in goldens:
        out = pipeline.ask(g["question"])

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=out["answer"],
                expected_output=g["ideal_answer"],
            )
        )

    from typing import Any as _Any

    metrics: list[_Any] = [
        _build_correctness(),
        _build_completeness(),
        _build_style(),
    ]
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.load_documents()
    summary = run(pipeline, limit=args.limit)
    print_summary("application", summary)


if __name__ == "__main__":
    main()
