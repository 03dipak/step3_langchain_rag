"""Unit tests for the DeepEval suite (eval/deepeval_suite/).

These test the pure logic and failures paths of the judge and results
persistence offline. The live Groq judge + pipeline are mocked so no API calls
or embedding downloads happen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import deepeval_suite.evaluate as eval_module
import pytest
from deepeval_suite.evaluate import save_results
from deepeval_suite.judge import MAX_RETRIES, GroqJudge
from dotenv import load_dotenv

# GroqJudge() constructs a ChatGroq which needs an API key present at build
# time (before tests mock judge.model). Step 3 does not auto-load .env (unlike
# Step 2's generator, which called load_dotenv), so load it here to mirror the
# Step 2 runtime where construct succeeds and the judge is then mocked offline.
load_dotenv()


# ---- _coerce_result ----
def test_coerce_result_none_schema_returns_raw() -> None:
    assert GroqJudge._coerce_result("hello", None) == "hello"


def test_coerce_result_plain_json_parsed_to_schema() -> None:
    class FakeSchema:
        @staticmethod
        def model_validate_json(text: str) -> Any:
            return json.loads(text)

    out = GroqJudge._coerce_result('{"a": 1}', FakeSchema)
    assert out == {"a": 1}


def test_coerce_result_strips_fenced_json() -> None:
    class FakeSchema:
        @staticmethod
        def model_validate_json(text: str) -> str:
            return text

    out = GroqJudge._coerce_result('```json\n{"a": 1}\n```', FakeSchema)
    assert json.loads(out) == {"a": 1}


# ---- _retry_delay ----
def test_retry_delay_parses_groq_message() -> None:
    err = Exception(
        "Rate limit reached ... Limit 8000, Used 7635, "
        "Requested 1371. Please try again in 7.545s."
    )
    assert GroqJudge._retry_delay(err) == pytest.approx(8.045)


def test_retry_delay_fallback_when_no_seconds() -> None:
    assert GroqJudge._retry_delay(Exception("rate limited")) == 8.0


# ---- generate (sync) with retry ----
def test_generate_retries_on_rate_limit_then_succeeds(
    mocker: Any, monkeypatch: Any
) -> None:
    from groq import RateLimitError

    judge = GroqJudge()
    sleeps: list[float] = []

    class Res:
        content = "ok"

    invoke = mocker.MagicMock(
        side_effect=[
            RateLimitError(
                "try again in 2.0s",
                response=mocker.MagicMock(status_code=429),
                body={},
            ),
            Res(),
        ]
    )
    judge.model = mocker.MagicMock(invoke=invoke)
    monkeypatch.setattr("deepeval_suite.judge.time.sleep", sleeps.append)
    assert judge.generate("p") == "ok"
    assert invoke.call_count == 2
    assert sleeps and sleeps[0] == pytest.approx(2.5)


def test_generate_raises_after_max_retries(
    mocker: Any, monkeypatch: Any
) -> None:
    from groq import RateLimitError

    judge = GroqJudge()
    judge.model = mocker.MagicMock(
        invoke=mocker.MagicMock(
            side_effect=RateLimitError(
                "x",
                response=mocker.MagicMock(status_code=429),
                body={},
            )
        )
    )
    monkeypatch.setattr("deepeval_suite.judge.time.sleep", lambda _: None)
    with pytest.raises(RateLimitError):
        judge.generate("p")


def test_generate_json_error_retries(mocker: Any, monkeypatch: Any) -> None:
    class FakeSchema:
        @staticmethod
        def model_validate_json(text: str) -> str:
            raise ValueError("bad json")

    class Res:
        content = "oops"

    judge = GroqJudge()
    judge.model = mocker.MagicMock(invoke=mocker.MagicMock(return_value=Res()))
    sleeps: list[float] = []
    monkeypatch.setattr("deepeval_suite.judge.time.sleep", sleeps.append)
    with pytest.raises(ValueError):
        judge.generate("p", schema=FakeSchema)
    assert len(sleeps) == MAX_RETRIES - 1


# ---- a_generate (async) ----
@pytest.mark.asyncio
async def test_a_generate_succeeds(mocker: Any, monkeypatch: Any) -> None:
    class Res:
        content = "async-ok"

    judge = GroqJudge()
    judge.model = mocker.MagicMock(
        ainvoke=mocker.AsyncMock(return_value=Res())
    )
    monkeypatch.setattr("deepeval_suite.judge.asyncio.sleep", _noop_await)
    assert await judge.a_generate("p") == "async-ok"


@pytest.mark.asyncio
async def test_a_generate_retries_rate_limit(mocker: Any, monkeypatch: Any) -> None:
    from groq import RateLimitError

    class Res:
        content = "ok"

    judge = GroqJudge()
    judge.model = mocker.MagicMock(
        ainvoke=mocker.AsyncMock(
            side_effect=[
                RateLimitError(
                    "try again in 3.0s",
                    response=mocker.MagicMock(status_code=429),
                    body={},
                ),
                Res(),
            ]
        )
    )
    sleeps: list[float] = []

    async def _sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("deepeval_suite.judge.asyncio.sleep", _sleep)
    assert await judge.a_generate("p") == "ok"
    assert sleeps[0] == pytest.approx(3.5)


async def _noop_await(*args: Any) -> None:
    return None


# ---- misc judge ----
def test_get_model_name_default() -> None:
    assert GroqJudge().get_model_name() == "openai/gpt-oss-120b"


# ---- evaluate.save_results ----
def test_save_results_writes_summary(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)

    class Metric:
        def __init__(self, name: str, score: float) -> None:
            self.name = name
            self.score = score

    class TestResult:
        def __init__(self, metrics: list[Metric]) -> None:
            self.metrics_data = metrics

    class Results:
        test_results: ClassVar[list[Any]] = [
            TestResult(
                [
                    Metric("Contextual Recall", 1.0),
                    Metric("Contextual Precision", 1.0),
                    Metric("Faithfulness", 0.6),
                ]
            ),
            TestResult(
                [
                    Metric("Contextual Recall", 1.0),
                    Metric("Contextual Precision", 0.5),
                    Metric("Faithfulness", 1.0),
                ]
            ),
        ]

    path = save_results(Results(), k=3, limit=2, no_reason=True)
    payload = json.loads(path.read_text())
    assert payload["engine"] == "deepeval"
    assert payload["config"] == {
        "k": 3,
        "limit": 2,
        "no_reason": True,
        "threshold": 0.7,
    }
    summary = payload["summary"]
    assert summary["Contextual Recall"] == {"pass": 2, "total": 2, "pass_rate": 1.0}
    assert summary["Contextual Precision"]["pass"] == 1
    assert summary["Faithfulness"]["pass"] == 1


def test_save_results_empty_metrics(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(eval_module, "RESULTS_DIR", tmp_path)

    class NoMetrics:
        metrics_data: ClassVar[list[Any]] = []

    class Results:
        test_results: ClassVar[list[Any]] = [NoMetrics()]

    payload = json.loads(save_results(Results(), k=1, limit=None, no_reason=False).read_text())
    assert payload["summary"] == {}


# ---- evaluate.load_golden ----
def test_load_golden_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "golden.jsonl"
    p.write_text('{"question": "q1", "answer": "a1"}\n\n{"question": "q2"}\n')
    assert eval_module.load_golden(p) == [
        {"question": "q1", "answer": "a1"},
        {"question": "q2"},
    ]


# ---- evaluate.build_test_cases ----
def test_build_test_cases_raises_without_retriever() -> None:
    class NoRetriever:
        retriever: Any = None

    with pytest.raises(ValueError):
        eval_module.build_test_cases(
            [{"question": "q", "answer": "a"}], NoRetriever()  # type: ignore[arg-type]
        )


def test_build_test_cases_builds_llm_cases(mocker: Any) -> None:
    retriever = mocker.MagicMock()
    retriever.retrieve.return_value = [
        {"text": "chunk", "metadata": {"source": "s.txt"}, "score": 0.9}
    ]
    generator = mocker.MagicMock()
    generator.generate.return_value = "generated answer"

    class FakePipeline:
        retriever: Any
        generator: Any

    fp = FakePipeline()
    fp.retriever = retriever
    fp.generator = generator

    entries = [{"question": "what", "answer": "gold"}]
    cases = eval_module.build_test_cases(entries, fp, k=3)  # type: ignore[arg-type]
    assert len(cases) == 1
    assert cases[0].input == "what"
    assert cases[0].actual_output == "generated answer"
    assert cases[0].expected_output == "gold"
    assert cases[0].retrieval_context == ["chunk"]
    retriever.retrieve.assert_called_once_with("what", top_k=3)
