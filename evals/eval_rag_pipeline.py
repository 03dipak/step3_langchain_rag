"""Pipeline-level evaluation: the RAG Triad.

Tests the INTEGRATED pipeline — retriever feeds the generator, eval scores
all three pairwise relationships:

    Context Relevance  — is the retrieved context relevant to the question?
    Faithfulness       — is the generated answer grounded in the context?
    Answer Relevance   — does the generated answer address the question?

Usage::

    python -m evals.eval_rag_pipeline
    python -m evals.eval_rag_pipeline --limit 5 --no-reason
"""

from __future__ import annotations

import argparse

from deepeval import evaluate  # type: ignore[attr-defined]
from deepeval.metrics import (  # type: ignore[attr-defined]
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase  # type: ignore[attr-defined]
from dotenv import load_dotenv

from evals.harness import load_goldens, print_summary, summarize_by_metric
from langchain_rag.pipeline import Pipeline

load_dotenv()

GOLDEN_PATH = "eval/goldens/faithfulness_dataset.json"
THRESHOLD = 0.7


def run(pipeline: Pipeline, *, limit: int | None = None, include_reason: bool = True) -> dict:  # type: ignore[type-arg]
    """Run the RAG triad on live pipeline.ask() output."""
    from deepeval_suite.judge import GroqJudge

    judge = GroqJudge()

    goldens = load_goldens(GOLDEN_PATH)
    if limit:
        goldens = goldens[:limit]

    test_cases = []
    for g in goldens:
        out = pipeline.ask(g["query"])

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=out["answer"],
                retrieval_context=[s["text"] for s in out["sources"]],
            )
        )

    metrics = [
        ContextualRelevancyMetric(
            threshold=THRESHOLD, model=judge, include_reason=include_reason
        ),
        FaithfulnessMetric(
            threshold=THRESHOLD, model=judge, include_reason=include_reason
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD, model=judge, include_reason=include_reason
        ),
    ]

    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-reason", action="store_true")
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.load_documents()
    summary = run(pipeline, limit=args.limit, include_reason=not args.no_reason)
    print_summary("rag_pipeline", summary)


if __name__ == "__main__":
    main()
