"""LangChain LLM generator.

Replaces Step 2's raw ``openai`` SDK call with a LangChain chain built from the
registry-backed adapter. The public surface (``generate`` / ``refusal_response``)
stays identical to Step 2's generator so the evaluator and app work unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_rag.prompt_registry import PromptRegistry
from langchain_rag.prompts import LangChainPromptAdapter


class LangChainGenerator:
    """Generate answers with a LangChain chain driven by the PromptRegistry."""

    def __init__(
        self,
        registry: PromptRegistry | None = None,
        prompt_id: str = "RAG_ANSWER",
    ) -> None:
        self.registry = registry or PromptRegistry()
        # When no registry is passed in, seed it from the default registry file so
        # the eagerly-built chain finds an approved prompt. Without this, a bare
        # Pipeline()/LangChainGenerator() fails at construction with "No approved
        # version found" because a fresh PromptRegistry() starts empty.
        if registry is None:
            self.registry.load()
        self.prompt_id = prompt_id
        self.adapter = LangChainPromptAdapter(self.registry)
        self.chain = self.adapter.build_chain(prompt_id)

    def generate(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        """Run the LCEL chain over question + retrieved chunks."""
        result = self.chain.invoke({"sources": context_chunks, "question": question})
        if hasattr(result, "content"):
            return str(result.content).strip()
        return str(result).strip()

    def refusal_response(self, question: str) -> str:
        """Polite refusal when no context was retrieved.

        Reads the ``refusal_string`` from the approved version's ``output_schema``
        so the fallback text is **versioned**: rollback 1.1.0 -> 1.0.0 restores the
        refusal wording of 1.0.0 end-to-end.
        """
        record = self.registry.get(self.prompt_id, approved_only=True)
        schema = record.get("output_schema") or PromptRegistry.DEFAULT_OUTPUT_SCHEMA
        template = schema.get(
            "refusal_string",
            "I don't have enough context to answer: {question}",
        )
        return template.format(question=question)
