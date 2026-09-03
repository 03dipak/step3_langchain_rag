"""LangChain ``Embeddings`` adapter backed by the shared ``qwen3_embed`` model.

This is the key "adapter" insight in action: LangChain's ``Embeddings`` interface
owns the contract (``embed_documents`` / ``embed_query``), while the actual
embedding backend is the exact same ``qwen3-embed`` model that Step 2 used by
hand. Because the model is identical, Step 2 vs Step 3 retrieval scores are a
valid A/B comparison of the *framework*, not of the embeddings.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.embeddings import Embeddings


class Qwen3Embeddings(Embeddings):
    """Adapts the qwen3_embed model to LangChain's ``Embeddings`` interface."""

    _model: Any = None

    def _get_model(self) -> Any:
        if Qwen3Embeddings._model is None:
            from qwen3_embed import TextEmbedding

            Qwen3Embeddings._model = TextEmbedding(model_name=self._model_name())
        return Qwen3Embeddings._model

    @staticmethod
    def _model_name() -> str:
        return os.getenv("EMBED_MODEL", "n24q02m/Qwen3-Embedding-0.6B-ONNX")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.embed(texts)
        return [[float(x) for x in embedding] for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        embeddings = list(model.query_embed(text))
        return [float(x) for x in embeddings[0]]
