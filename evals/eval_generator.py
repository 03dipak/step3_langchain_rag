"""Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the golden context it was given? (Did the generator make things up?)

Answer Relevancy: does the generated answer actually address the user's question?

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks from
the faithfulness dataset), NOT the retriever's output.  So a low score is purely
the generator's fault — the context was already correct.

Usage::

    python -m evals.eval_generator
    python -m evals.eval_generator --limit 5 --no-reason
"""

from __future__ import annotations

import argparse

from deepeval import evaluate  # type: ignore[attr-defined]
from deepeval.metrics import (  # type: ignore[attr-defined]
    AnswerRelevancyMetric,
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
    """Run faithfulness + answer relevancy on the generator with golden context."""
    from deepeval_suite.judge import GroqJudge

    judge = GroqJudge()

    goldens = load_goldens(GOLDEN_PATH)
    if limit:
        goldens = goldens[:limit]

    test_cases = []
    for g in goldens:
        # ideal_context is a list of strings (full source doc content).
        # Wrap each string as a dict so generator.generate() gets list[dict].
        ideal_context_strs: list[str] = g["ideal_context"]
        context_chunks = [{"text": s} for s in ideal_context_strs]

        answer = pipeline.generator.generate(g["query"], context_chunks)

        # retrieval_context for faithfulness check = the golden strings themselves
        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,
                retrieval_context=ideal_context_strs,
            )
        )

    metrics = [
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
    print_summary("generator", summary)


if __name__ == "__main__":
    main()
