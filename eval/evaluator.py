from datetime import UTC, datetime
from typing import Any

STOPWORDS = frozenset(
    [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "as", "was", "are", "be",
        "has", "have", "had", "been", "will", "would", "could", "should",
        "may", "might", "shall", "can", "do", "did", "not", "no", "so",
        "if", "then", "than", "too", "very", "just", "about", "above",
        "after", "again", "all", "also", "am", "any", "because", "before",
        "below", "between", "both", "each", "few", "further", "here",
        "how", "into", "more", "most", "other", "our", "out", "over",
        "own", "same", "some", "such", "there", "these", "they", "this",
        "those", "through", "under", "until", "up", "we", "what", "when",
        "where", "which", "while", "who", "whom", "why", "you", "your",
        "he", "she", "his", "her", "its", "me", "my", "mine", "myself",
        "himself", "herself", "itself", "ourselves", "themselves",
    ]
)


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text (lowercased, stopwords removed)."""
    return [
        w
        for w in text.lower().split()
        if w.isalpha() and len(w) > 2 and w not in STOPWORDS
    ]


def context_recall(gold_answer: str, retrieved_chunks: list[str]) -> float:
    """Fraction of gold_answer keywords found in the retrieved context."""
    gold_keywords = extract_keywords(gold_answer)
    context_text = " ".join(retrieved_chunks).lower()
    if not gold_keywords:
        return 0.0
    covered = sum(1 for kw in gold_keywords if kw in context_text)
    return covered / len(gold_keywords)


def context_precision(
    question: str, retrieved_chunks: list[dict[str, Any]], gold_source: str
) -> float:
    """Fraction of retrieved chunks whose source matches the gold source."""
    if not retrieved_chunks:
        return 0.0
    relevant = sum(1 for c in retrieved_chunks if c["metadata"]["source"] == gold_source)
    return relevant / len(retrieved_chunks)


def faithful_fallback(answer: str, context_chunks: list[str]) -> float:
    """Offline fallback for faithfulness: fraction of answer keywords in context."""
    answer_keywords = extract_keywords(answer)
    context_text = " ".join(context_chunks).lower()
    if not answer_keywords:
        return 0.0
    supported = sum(1 for kw in answer_keywords if kw in context_text)
    return supported / len(answer_keywords)


def faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Check if answer claims is grounded in context (offline fallback)."""
    return faithful_fallback(answer, context_chunks)


def relevance_fallback(question: str, answer: str) -> float:
    """Offline fallback for answer_relevance: keyword overlap between question and answer."""
    q_keywords = set(extract_keywords(question))
    a_keywords = set(extract_keywords(answer))
    if not q_keywords:
        return 0.0
    return len(q_keywords & a_keywords) / len(q_keywords)


def answer_relevance(question: str, answer: str) -> float:
    """Check if the answer addresses the question (offline fallback)."""
    return relevance_fallback(question, answer)


def evaluate_single(
    question: str,
    gold_answer: str,
    gold_source: str,
    pipeline: Any,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    """Run pipeline.ask() and compute all four metrics."""
    result = pipeline.ask(question, **pipeline_kwargs)
    return {
        "question": question,
        "context_recall": context_recall(gold_answer, [c["text"] for c in result["sources"]]),
        "context_precision": context_precision(question, result["sources"], gold_source),
        "faithfulness": faithfulness(result["answer"], [c["text"] for c in result["sources"]]),
        "answer_relevance": answer_relevance(question, result["answer"]),
        "answer": result["answer"],
        "sources": result["sources"],
        "prompt_key": result.get("prompt_key"),
    }


def run_full_eval(
    pipeline: Any, golden_path: str, **pipeline_kwargs: Any
) -> dict[str, Any]:
    """Evaluate every entry in golden.jsonl and return averaged metrics."""
    import json

    with open(golden_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    results = [
        evaluate_single(
            line["question"],
            line.get("gold_answer", line.get("answer")),
            line.get("gold_source", line.get("source")),
            pipeline,
            **pipeline_kwargs,
        )
        for line in lines
    ]

    n = len(results)
    summary: dict[str, float] = {}
    if n > 0:
        for key in ("context_recall", "context_precision", "faithfulness", "answer_relevance"):
            summary[key] = sum(r[key] for r in results) / n

    return {
        "results": results,
        "summary": summary,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def evaluate_with_registry(
    pipeline: Any,
    golden_path: str,
    registry: Any,
    prompt_key: str,
    **pipeline_kwargs: Any,
) -> dict[str, Any]:
    """Run full eval and record scores via the registry."""
    results = run_full_eval(pipeline, golden_path, **pipeline_kwargs)
    summary = dict(results["summary"])
    summary["accuracy"] = summary["context_recall"]
    registry.record_eval_scores(
        prompt_key, summary, eval_run_id=f"eval_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    )
    return results
