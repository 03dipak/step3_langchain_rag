"""Tests for LangChainPromptAdapter — the registry -> LangChain bridge.

The registry owns Content / Policy / Output / Evidence; LangChain is just the
execution layer. We mock ``build_llm`` (the one boundary that could touch the
network) and assert the adapter drives everything from the registry, including
injecting the version's ``output_schema`` rules into the rendered prompt.
"""

from __future__ import annotations

from typing import Any, cast

from pytest_mock import MockerFixture

from langchain_rag.prompt_registry import PromptRegistry
from langchain_rag.prompts import LangChainPromptAdapter

SCHEMA = {
    "format": "markdown",
    "shape": {"answer": "str", "sources": "list[dict]"},
    "length_policy": "4-6 sentences",
    "citation_policy": "inline [source-N] markers",
    "refusal_string": "I don't have enough context to answer: {question}",
    "display": "answer text, then a Sources list",
}
TEMPLATE = "Answer ONLY from context: {context}\nQuestion: {question}"


def _approved_registry() -> PromptRegistry:
    registry = PromptRegistry()
    key = registry.register(
        "RAG_ANSWER",
        TEMPLATE,
        ["context", "question"],
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        temperature=0.1,
        output_schema=SCHEMA,
    )
    registry.promote(key)
    registry.record_eval_scores(key, {"accuracy": 0.9}, "e1")
    registry.promote(key)
    return registry


def _mock_build_llm(mocker: MockerFixture) -> Any:
    """Return a fake LLM instead of ChatOpenAI (no network / no API key)."""
    fake = mocker.MagicMock()
    import langchain_rag.prompts as prompts_mod

    mocker.patch.object(prompts_mod, "build_llm", return_value=fake)
    return fake


def test_get_prompt_template_from_registry(mocker: MockerFixture) -> None:
    """get_prompt_template reads the approved template + input_variables."""
    _mock_build_llm(mocker)
    adapter = LangChainPromptAdapter(_approved_registry())

    tmpl = adapter.get_prompt_template("RAG_ANSWER")

    assert tmpl.template == TEMPLATE
    assert tmpl.input_variables == ["context", "question"]


def test_get_llm_uses_registry_policy(mocker: MockerFixture) -> None:
    """get_llm builds a model from the registry's model + temperature."""
    fake = _mock_build_llm(mocker)
    adapter = LangChainPromptAdapter(_approved_registry())

    llm = adapter.get_llm("RAG_ANSWER")

    assert llm is fake
    import langchain_rag.prompts as prompts_mod

    cast(Any, prompts_mod.build_llm).assert_called_once_with(
        "Qwen/Qwen2.5-7B-Instruct-AWQ", 0.1
    )


def test_build_prompt_injects_schema_rules(mocker: MockerFixture) -> None:
    """build_prompt injects format/length/citation rules into the system prompt."""
    _mock_build_llm(mocker)
    adapter = LangChainPromptAdapter(_approved_registry())

    prompt = adapter.build_prompt("RAG_ANSWER")
    system = prompt.format_messages(question="q", context="ctx")[0].content

    assert "Answer ONLY from context: ctx\nQuestion: q" in system
    assert "Output format: markdown" in system
    assert "Length: 4-6 sentences" in system
    assert "Citations: inline [source-N] markers" in system


def test_build_chain_driven_by_registry(mocker: MockerFixture) -> None:
    """build_chain sends the registry template + schema rules to the LLM.

    The mocked LLM (a plain callable mock, coerced by LCEL) receives a
    ChatPromptValue; capture it and assert the rendered System message carries
    the template + output rules over the formatted context.
    """
    from langchain_core.prompt_values import ChatPromptValue

    fake = _mock_build_llm(mocker)
    seen: list[ChatPromptValue] = []

    def _echo(messages: ChatPromptValue) -> str:
        seen.append(messages)
        return "LLM_OUT"

    fake.side_effect = _echo
    adapter = LangChainPromptAdapter(_approved_registry())

    chain = adapter.build_chain("RAG_ANSWER")
    out = chain.invoke(
        {
            "sources": [
                {"text": "chunk one", "metadata": {"source": "a.txt"}},
                {"text": "chunk two", "metadata": {"source": "b.txt"}},
            ],
            "question": "What is Python?",
        }
    )

    assert out == "LLM_OUT"
    system = seen[0].messages[0].content
    human = seen[0].messages[1].content
    assert "Answer ONLY from context: [1] chunk one\n[2] chunk two\nQuestion: What is Python?" in system
    assert "Output format: markdown" in system
    assert "Citations: inline [source-N] markers" in system
    assert human == "What is Python?"


def test_format_context_numbered_sources() -> None:
    """_format_context renders sources as [N] text lines."""
    sources = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

    result = LangChainPromptAdapter._format_context(sources)

    assert result == "[1] a\n[2] b\n[3] c"