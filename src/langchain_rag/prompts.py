"""Registry -> LangChain adapter.

LangChain's ``PromptTemplate``/chat models should be an *adapter* over the
shared PromptRegistry, not the source of truth. The registry owns Content
(template), Policy (model/temperature) and Evidence (eval scores); LangChain
owns execution. This module turns a registered prompt into a ready-to-invoke
LCEL ``Runnable`` (``prompt | llm``).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_rag.prompt_registry import PromptRegistry


def build_llm(model: str, temperature: float = 0.0) -> Any:
    """Build a LangChain chat model from a registered prompt's Policy fields.

    Defaults to ``ChatOpenAI`` pointed at the OpenAI-compatible gateway
    (``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL``) so Step 3 uses the exact
    same LLM as Step 2 -- the framework, not the model, is what varies.
    """
    from langchain_openai import ChatOpenAI

    base = os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1"
    api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_base=base,
        openai_api_key=api_key,
        max_tokens=300,
    )


class LangChainPromptAdapter:
    """Bridges the shared PromptRegistry to LangChain ``PromptTemplate`` + LLM.

    The registry owns the lifecycle; LangChain handles execution.
    """

    def __init__(self, registry: PromptRegistry):
        self.registry = registry

    def get_prompt_template(self, prompt_id: str, **kwargs: Any) -> PromptTemplate:
        """Get the approved prompt from the registry, wrap in a PromptTemplate."""
        record = self.registry.get(prompt_id, approved_only=True)
        return PromptTemplate(
            template=record["template"],
            input_variables=record["input_variables"],
        )

    def get_llm(self, prompt_id: str) -> Any:
        """Get a LangChain chat model configured from the registry's Policy."""
        record = self.registry.get(prompt_id, approved_only=True)
        return build_llm(record["model"], record["temperature"])

    def build_prompt(self, prompt_id: str) -> ChatPromptTemplate:
        """Build a chat prompt that inserts context/question into the registry template."""
        record = self.registry.get(prompt_id, approved_only=True)
        system = self._inject_schema_rules(record)
        return ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

    def build_chain(self, prompt_id: str) -> Any:
        """Build an LCEL runnable: format context into the template, then call the LLM."""
        from langchain_core.runnables import RunnablePassthrough

        record = self.registry.get(prompt_id, approved_only=True)
        system = self._inject_schema_rules(record)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        llm = build_llm(record["model"], record["temperature"])
        return {
            "context": RunnablePassthrough()
            | (lambda x: self._format_context(x["sources"])),
            "question": RunnablePassthrough() | (lambda x: x["question"]),
        } | prompt | llm

    def _inject_schema_rules(self, record: dict[str, Any]) -> str:
        """Append the version's output_schema rules to the system prompt.

        This is how output behavior (format, length, citation) becomes
        **versioned prompt policy** — the rules ride on the version and
        change when the version changes (rollback restores full output behavior).
        """
        schema = record.get("output_schema") or {}
        parts = [record["template"]]
        rules = []
        if schema.get("format"):
            rules.append(f"Output format: {schema['format']}")
        if schema.get("length_policy"):
            rules.append(f"Length: {schema['length_policy']}")
        if schema.get("citation_policy"):
            rules.append(f"Citations: {schema['citation_policy']}")
        if rules:
            parts.append("Output rules:\n" + "\n".join(f"- {r}" for r in rules))
        return "\n\n".join(parts)

    @staticmethod
    def _format_context(sources: list[dict[str, Any]]) -> str:
        return "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources, start=1))
