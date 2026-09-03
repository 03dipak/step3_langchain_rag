"""Cross-encoder second-stage reranker over Step 2-shaped dicts.

Step 2 had no reranker; Step 3 adds a real cross-encoder stage between the
Chroma retriever and the LLM generator so we can measure what reranking buys.
We deliberately use a raw ``sentence-transformers.CrossEncoder`` directly
on the already-retrieved ``{text, metadata, score}`` dicts (not LangChain's
``ContextualCompressionRetriever``), which needs a ``BaseRetriever`` yielding
``Document`` objects and isn't in the modern ``langchain`` split. This keeps
the whole downstream contract (evaluator, DeepEval, ``ask()`` shape) untouched.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, top_n: int = 3) -> None:
        self.top_n = top_n
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder | None:
        """Lazy-load the cross-encoder once; None if it can't load (offline)."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:  # noqa: BLE001 - offline fallback: any load error degrades to None
                return None
        return self._model

    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Re-score and re-sort candidates by cross-encoder score (desc).

        Attaches ``rerank_score`` and keeps the original ``score`` and
        ``metadata["source"]``. Caps output to ``top_n`` (defaults to the
        instance's ``top_n``). If the model can't load (offline / no cache),
        returns the candidates unchanged (original order, no crash).
        """
        if not candidates:
            return []
        limit = top_n if top_n is not None else self.top_n

        model = self._get_model()
        if model is None:
            return candidates

        pairs = [(question, c["text"]) for c in candidates]
        scores = model.predict(pairs)

        ranked = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:limit]
