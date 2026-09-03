"""Tests for LangChainGenerator — mock the chain/LLM boundary, not internals."""

from __future__ import annotations

from typing import Any

from pytest_mock import MockerFixture

from langchain_rag.generator import LangChainGenerator
from langchain_rag.prompt_registry import PromptRegistry


def _approved_registry(refusal: str | None = None) -> tuple[PromptRegistry, str]:
    """Build a registry with an approved RAG_ANSWER version."""
    registry = PromptRegistry()
    key = registry.register(
        "RAG_ANSWER",
        "Answer from context only: {question}",
        ["question"],
        change_note="V1",
        output_schema=(
            {"refusal_string": refusal} if refusal else None
        ),
    )
    registry.promote(key)
    registry.record_eval_scores(key, {"accuracy": 0.9}, "e1")
    registry.promote(key)
    return registry, key


def _patch_build_chain(mocker: MockerFixture) -> Any:
    from langchain_rag.prompts import LangChainPromptAdapter

    chain = mocker.MagicMock()
    mocker.patch.object(LangChainPromptAdapter, "build_chain", return_value=chain)
    return chain


def test_generate_returns_content(mocker: MockerFixture) -> None:
    """generate extracts .content from an AIMessage and strips it."""
    registry, _ = _approved_registry()
    chain = _patch_build_chain(mocker)
    chain.invoke.return_value = type("AIMessage", (), {"content": "  hello  "})()

    gen = LangChainGenerator(registry)
    result = gen.generate("q", [{"text": "ctx", "metadata": {}, "score": 0.9}])

    assert result == "hello"


def test_generate_falls_back_to_str(mocker: MockerFixture) -> None:
    """generate returns str(result) when result has no .content."""
    registry, _ = _approved_registry()
    chain = _patch_build_chain(mocker)
    chain.invoke.return_value = "  plain string answer  "

    gen = LangChainGenerator(registry)
    result = gen.generate("q", [])

    assert result == "plain string answer"


def test_generate_calls_chain_with_input(mocker: MockerFixture) -> None:
    """generate passes {sources, question} to chain.invoke and strips .content."""
    registry, _ = _approved_registry()
    chain = _patch_build_chain(mocker)
    chain.invoke.return_value = type("AIMessage", (), {"content": "ok"})()

    chunks = [{"text": "ctx", "metadata": {"source": "a.txt"}, "score": 0.9}]
    LangChainGenerator(registry).generate("q", chunks)

    chain.invoke.assert_called_once_with({"sources": chunks, "question": "q"})


def test_refusal_response_reads_schema(mocker: MockerFixture) -> None:
    """refusal_response returns the approved version's refusal_string."""
    _patch_build_chain(mocker)
    registry, _ = _approved_registry(refusal="No context for: {question}")
    gen = LangChainGenerator(registry)

    assert gen.refusal_response("hi") == "No context for: hi"


def test_refusal_response_default(mocker: MockerFixture) -> None:
    """refusal_response falls back to the default when no refusal_string set."""
    _patch_build_chain(mocker)
    registry, _ = _approved_registry()
    gen = LangChainGenerator(registry)

    assert (
        gen.refusal_response("hi")
        == "I don't have enough context to answer: hi"
    )
