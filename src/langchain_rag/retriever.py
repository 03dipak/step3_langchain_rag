
"""Chroma-backed retriever returning Step 2-shaped results.

Deliberately a thin wrapper over ``LangChainStore.search`` instead of LangChain's
``vectorstore.as_retriever()``: that API returns ``Document`` objects with no
``score``, which would break our ``min_score`` gate and the evaluator's
``metadata["source"]`` access. Keeping our own retriever preserves the exact
``{text, metadata, score}`` contract Step 2's eval/DeepEval code expects.
"""

from __future__ import annotations

from typing import Any

from langchain_rag.vectorstore import LangChainStore


class Retriever:
    def __init__(self, store: LangChainStore) -> None:
        self.store = store

    def retrieve(
        self, query: str, top_k: int = 3, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Return top-k relevant chunks from the store in Step 2 dict shape."""
        return self.store.search(query, top_k=top_k, min_score=min_score)