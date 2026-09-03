"""Operational eval: COST.

Cost is not measured, it is DERIVED: cost = tokens x price. So the real work is
getting an honest token count, then multiplying by the current per-token rate.

Why cost is a legitimately OFFLINE metric:
  - tokens are near-deterministic (same retrieved context + temperature=0 => near
    same counts every run), so cost barely moves run-to-run. That stability lets
    you estimate unit economics BEFORE launch.

How tokens are captured: this repo's generator chain ends at `| llm` (no
StrOutputParser), so invoking `generator.chain` returns an AIMessage carrying
`usage_metadata` — no extra reconstruction needed.

Usage::

    python -m evals.eval_cost
"""

from __future__ import annotations

from dotenv import load_dotenv

from langchain_rag.pipeline import Pipeline

load_dotenv()

QUESTIONS = [
    "What is the difference between supervised and unsupervised learning?",
    "What is overfitting in machine learning?",
    "What is reranking in RAG?",
    "What is cosine similarity?",
]

REPEATS = 3

# --- Pricing for the OpenAI-compatible gateway (GROQ). Prices change; verify ---
# --- these constants against the provider's pricing page before trusting a budget.
PRICE_INPUT_PER_1M = 0.15  # cache-miss input
PRICE_CACHED_INPUT_PER_1M = 0.075  # cached (repeated prefix) input
PRICE_OUTPUT_PER_1M = 0.60  # output (4x input)

QUERIES_PER_DAY = 2000
USD_TO_INR = 88.0

COST_BUDGET_PER_QUERY_USD = 0.0015


def measure_tokens(pipeline: Pipeline, question: str) -> dict[str, float]:
    """Run one generation over real retrieved context; read token usage off the AIMessage."""
    assert pipeline.retriever is not None
    results = pipeline.retriever.retrieve(question)

    msg = pipeline.generator.chain.invoke(
        {"sources": results, "question": question}
    )
    usage = getattr(msg, "usage_metadata", None) or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    details = usage.get("input_token_details") or {}
    cached_tokens = details.get("cache_read", 0) or 0

    return {
        "input": float(input_tokens),
        "output": float(output_tokens),
        "cached": float(cached_tokens),
    }


def cost_usd(
    input_tokens: float, output_tokens: float, cached_tokens: float
) -> dict[str, float]:
    uncached_input = max(input_tokens - cached_tokens, 0)
    c_in = uncached_input / 1_000_000 * PRICE_INPUT_PER_1M
    c_cached = cached_tokens / 1_000_000 * PRICE_CACHED_INPUT_PER_1M
    c_out = output_tokens / 1_000_000 * PRICE_OUTPUT_PER_1M
    return {
        "input": c_in,
        "cached": c_cached,
        "output": c_out,
        "total": c_in + c_cached + c_out,
    }


def benchmark(pipeline: Pipeline) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    print("Measuring token usage...")
    for question in QUESTIONS:
        for _ in range(REPEATS):
            tok = measure_tokens(pipeline, question)
            cost = cost_usd(tok["input"], tok["output"], tok["cached"])
            rows.append({**tok, **{f"cost_{k}": v for k, v in cost.items()}})
    return rows


def _avg(rows: list[dict[str, float]], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def report(rows: list[dict[str, float]]) -> None:
    n = len(rows)
    avg_in = _avg(rows, "input")
    avg_out = _avg(rows, "output")
    avg_cached = _avg(rows, "cached")
    avg_cost = _avg(rows, "cost_total")
    min_cost = min(r["cost_total"] for r in rows)
    max_cost = max(r["cost_total"] for r in rows)

    avg_cost_out = _avg(rows, "cost_output")
    out_share = 100 * avg_cost_out / avg_cost if avg_cost else 0

    print("\n" + "=" * 70)
    print(
        f"COST (${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out)"
    )
    print("=" * 70)
    print(f"samples                : {n}")
    print(f"avg input tokens       : {avg_in:8.0f}   ({avg_cached:.0f} cached)")
    print(f"avg output tokens      : {avg_out:8.0f}")
    print("-" * 70)
    print(f"avg cost / query       : ${avg_cost:.6f}   (Rs {avg_cost * USD_TO_INR:.4f})")
    print(
        f"   min / max           : ${min_cost:.6f} / ${max_cost:.6f}   "
        f"<- tight range = cost is stable, unlike latency"
    )
    print(
        f"   input vs output     : {100 - out_share:.0f}% input / {out_share:.0f}% output"
    )
    print("-" * 70)

    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print(f"projection @ {QUERIES_PER_DAY}/day :")
    print(f"   per day             : ${daily:8.2f}   (Rs {daily * USD_TO_INR:8.2f})")
    print(f"   per month           : ${monthly:8.2f}   (Rs {monthly * USD_TO_INR:8.2f})")
    print("=" * 70)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(
        f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  "
        f"${avg_cost:.6f}   [{verdict}]"
    )
    print("=" * 70)


def main() -> None:
    pipeline = Pipeline()
    pipeline.load_documents()
    report(benchmark(pipeline))


if __name__ == "__main__":
    main()
