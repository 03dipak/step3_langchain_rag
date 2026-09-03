"""Tests for LangSmith tracer — including the enabled paths (lines 27-29, 46-48).

Controls the gate by mocking ``tracing_enabled`` because we don't want a real
key; the real-key branches exercise the lazy ``langsmith.traceable`` import and
the ``os.environ.setdefault`` writes.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from pytest_mock import MockerFixture

import langchain_rag.tracer as tracer_mod
from langchain_rag.tracer import (
    LANGSMITH_PLACEHOLDER,
    run_with_trace,
    setup_tracing,
    trace_generate,
    trace_retrieve,
    tracing_enabled,
)


def _fake_langsmith(monkeypatch: Any, wrapped: Any) -> Any:
    """Inject a fake ``langsmith`` module so the lazy import resolves offline."""
    mod: Any = types.ModuleType("langsmith")
    mod.traceable = lambda name=None: (lambda f: wrapped)
    monkeypatch.setitem(sys.modules, "langsmith", mod)
    return mod


class TestTracingEnabled:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    def test_false_when_not_set(self) -> None:
        assert tracing_enabled() is False

    def test_false_when_placeholder(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", LANGSMITH_PLACEHOLDER)
        assert tracing_enabled() is False

    def test_true_with_real_key(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_abc123")
        assert tracing_enabled() is True


class TestRunWithTrace:
    def test_calls_fn_directly_when_disabled(self, mocker: MockerFixture) -> None:
        """Disabled -> fn run directly, args forwarded, result unchanged."""
        mocker.patch.object(tracer_mod, "tracing_enabled", return_value=False)
        fn = mocker.MagicMock(return_value="result")

        out = run_with_trace("retrieve", fn, "arg1", kw="x")

        fn.assert_called_once_with("arg1", kw="x")
        assert out == "result"


class TestRunWithTraceEnabled:
    def test_traceable_wraps_fn_when_enabled(
        self, mocker: MockerFixture, monkeypatch: Any
    ) -> None:
        """Enabled -> langsmith.traceable(name)(fn) is called and invoked."""
        mocker.patch.object(tracer_mod, "tracing_enabled", return_value=True)
        wrapped = mocker.MagicMock(return_value="traced")
        recorded: list[Any] = []
        mod: Any = types.ModuleType("langsmith")

        def _traceable(*, name: str | None = None) -> Any:
            def _wrap(f: Any) -> Any:
                recorded.append(name)
                return wrapped

            return _wrap

        mod.traceable = _traceable
        monkeypatch.setitem(sys.modules, "langsmith", mod)

        out = run_with_trace("gen", mocker.MagicMock(), 1, 2)

        assert recorded == ["gen"]
        wrapped.assert_called_once_with(1, 2)
        assert out == "traced"


class TestTraceRetrieve:
    def test_disabled_returns_result_and_calls_fn(self, mocker: MockerFixture) -> None:
        mocker.patch.object(tracer_mod, "tracing_enabled", return_value=False)
        fn = mocker.MagicMock(return_value=["c1", "c2"])

        out = trace_retrieve(fn, "What is Python?", 3)

        fn.assert_called_once()
        assert out == ["c1", "c2"]

    def test_enabled_returns_result(
        self, mocker: MockerFixture, monkeypatch: Any
    ) -> None:
        mocker.patch.object(tracer_mod, "tracing_enabled", return_value=True)
        wrapped = mocker.MagicMock(return_value=["c1"])
        _fake_langsmith(monkeypatch, wrapped)
        fn = mocker.MagicMock()

        assert trace_retrieve(fn, "Q", 3) == ["c1"]


class TestTraceGenerate:
    def test_disabled_returns_result(self, mocker: MockerFixture) -> None:
        mocker.patch.object(tracer_mod, "tracing_enabled", return_value=False)
        fn = mocker.MagicMock(return_value="the answer")

        assert trace_generate(fn, "Q", [{}]) == "the answer"

    def test_enabled_returns_result(
        self, mocker: MockerFixture, monkeypatch: Any
    ) -> None:
        mocker.patch.object(tracer_mod, "tracing_enabled", return_value=True)
        wrapped = mocker.MagicMock(return_value="gen")
        _fake_langsmith(monkeypatch, wrapped)

        assert trace_generate(mocker.MagicMock(), "Q", [{}]) == "gen"


class TestSetupTracing:
    def test_sets_defaults_when_enabled(self, mocker: MockerFixture) -> None:
        """Enabled -> setdefault LANGCHAIN_TRACING_V2 and LANGCHAIN_PROJECT."""
        mocker.patch.object(tracer_mod, "tracing_enabled", return_value=True)
        mocker.patch.dict("os.environ", {"LANGSMITH_API_KEY": "lsv2_abc123"}, clear=True)
        mock_setdefault = mocker.patch("os.environ.setdefault")

        setup_tracing()
        mock_setdefault.assert_any_call("LANGCHAIN_TRACING_V2", "true")
        mock_setdefault.assert_any_call("LANGCHAIN_PROJECT", "step3-langchain-rag")

    def test_no_op_when_disabled(self, mocker: MockerFixture) -> None:
        mocker.patch.object(tracer_mod, "tracing_enabled", return_value=False)
        mocker.patch.dict("os.environ", {}, clear=True)
        mock_setdefault = mocker.patch("os.environ.setdefault")

        setup_tracing()
        mock_setdefault.assert_not_called()