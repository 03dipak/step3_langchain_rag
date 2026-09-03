"""Component-level evaluation of the RETRIEVABLE, in isolation.

Contextual Recall: what portion of all truly relevant chunks did the retriever find?
Contextual Precision: what portion of retrieved chunks are actually useful, weighted by rank?

Uses the real store + RerankableRetriever from the pipeline (no LLM generator).

Usage::

    python -m evals.eval_retriever
    python -m evals.eval_retriever --limit 5 --no-reason
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from deepeval import evaluate  # type: ignore[attr-defined]
from deepeval.metrics import (  # type: ignore[attr-defined]
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase  # type: ignore[attr-defined]
from dotenv import load_dotenv

from evals.harness import load_goldens, print_summary, summarize_by_metric
from langchain_rag.pipeline import Pipeline

load_dotenv()

GOLDEN_PATH = "eval/goldens/retriever_goldens.json"
THRESHOLD = 0.7


def run(pipeline: Pipeline, *, limit: int | None = None, include_reason: bool = True) -> dict:  # type: ignore[type-arg]
    """Run contextual recall + precision on the pipeline's rerankable retriever."""
    from deepeval_suite.judge import GroqJudge

    judge = GroqJudge()

    goldens = load_goldens(GOLDEN_PATH)
    if limit:
        goldens = goldens[:limit]

    assert pipeline.retriever is not None, (
        "Pipeline retriever not initialized — call load_documents() first."
    )

    test_cases = []
    for g in goldens:
        results = pipeline.retriever.retrieve(g["query"])
        retrieval_context = [r["text"] for r in results]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD, model=judge, include_reason=include_reason
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD, model=judge, include_reason=include_reason
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "RerankableRetriever",
            "embedding_model": "qwen3-embed",
            "top_k": 3,
            "judge_model": judge.get_model_name(),
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-reason", action="store_true")
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.load_documents()
    summary = run(pipeline, limit=args.limit, include_reason=not args.no_reason)
    print_summary("retriever", summary)


if __name__ == "__main__":
    main()
