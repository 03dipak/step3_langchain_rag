"""Operational eval: LATENCY (stage-split).

Unlike quality evals, latency needs no golden dataset and no LLM judge. It is a
deterministic measurement: run the pipeline N times, collect a distribution, and
report percentiles against a budget (SLO).

This repo's generator has NO streaming twin, so TTFT is out of scope here; we
report end-to-end plus a retrieval-vs-generation stage split.

Key ideas encoded below:
  - perf_counter, not time()          (right clock for elapsed time)
  - many samples -> percentiles       (p95/p99 tail, not the misleading mean)
  - discard warmup                    (cold start poisons the stats)
  - decompose the pipeline            (retrieval + generation)
  - log answer length                 (latency couples to output length)
  - single-user only                  (load testing is a separate exercise)

Usage::

    python -m evals.eval_latency
"""

from __future__ import annotations

import math
import time

from dotenv import load_dotenv

from langchain_rag.pipeline import Pipeline

load_dotenv()

QUESTIONS = [
    "What is the difference between supervised and unsupervised learning?",
    "What is overfitting in machine learning?",
    "What is reranking in RAG?",
    "How should API errors be formatted?",
]

REPEATS = 5
WARMUP_RUNS = 2

SLO_P95_MS = 3000


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile — matches numpy.percentile."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def run_stages(pipeline: Pipeline, question: str) -> tuple[str, dict[str, float]]:
    """Time retrieval and generation separately over live pipeline components."""
    t0 = time.perf_counter()
    assert pipeline.retriever is not None
    results = pipeline.retriever.retrieve(question)
    t1 = time.perf_counter()
    answer = pipeline.generator.generate(question, results)
    t2 = time.perf_counter()

    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
    }


def benchmark(pipeline: Pipeline) -> dict[str, list[float]]:
    print(f"Warming up ({WARMUP_RUNS} runs, discarded)...")
    for i in range(WARMUP_RUNS):
        run_stages(pipeline, QUESTIONS[i % len(QUESTIONS)])

    total_ms, retrieval_ms, generation_ms, answer_lengths = [], [], [], []

    print("Measuring...")
    for question in QUESTIONS:
        for _ in range(REPEATS):
            start = time.perf_counter()
            answer, stage = run_stages(pipeline, question)
            elapsed_ms = (time.perf_counter() - start) * 1000

            total_ms.append(elapsed_ms)
            retrieval_ms.append(stage["retrieval"])
            generation_ms.append(stage["generation"])
            answer_lengths.append(float(len(answer or "")))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "answer_len": answer_lengths,
    }


def _summarize(samples: list[float]) -> dict[str, float]:
    clean = [s for s in samples if not math.isnan(s)]
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean) if clean else float("nan"),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean) if clean else float("nan"),
        "max": max(clean) if clean else float("nan"),
    }


def _print_row(label: str, s: dict[str, float]) -> None:
    print(
        f"{label:<12} | n={s['n']:<3} mean={s['mean']:7.1f}  "
        f"p50={s['p50']:7.1f}  p95={s['p95']:7.1f}  "
        f"p99={s['p99']:7.1f}  min={s['min']:7.1f}  max={s['max']:7.1f}"
    )


def _slo_line(label: str, p95: float, budget: float) -> None:
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(
        f"SLO: {label:<22} p95 <= {budget:>5} ms  ->  p95 = {p95:7.0f} ms  "
        f"[{verdict}]"
    )


def report(results: dict[str, list[float]]) -> None:
    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)
    print(
        f"{'stage':<12} | {'samples':<5} {'mean':>11} {'p50':>11} "
        f"{'p95':>11} {'p99':>11} {'min':>11} {'max':>11}"
    )
    print("-" * 78)

    total = _summarize(results["total"])
    _print_row("end-to-end", total)
    _print_row("retrieval", _summarize(results["retrieval"]))
    _print_row("generation", _summarize(results["generation"]))

    avg_len = sum(results["answer_len"]) / len(results["answer_len"]) if results["answer_len"] else 0
    print("-" * 78)
    print(
        f"avg answer length: {avg_len:.0f} chars "
        f"(latency scales with output length — keep in mind when comparing configs)"
    )
    print("=" * 78)
    _slo_line("full answer", total["p95"], SLO_P95_MS)
    print("=" * 78)


def main() -> None:
    pipeline = Pipeline()
    pipeline.load_documents()
    report(benchmark(pipeline))


if __name__ == "__main__":
    main()
