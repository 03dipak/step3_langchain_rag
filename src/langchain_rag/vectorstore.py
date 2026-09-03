"""LangChain Chroma vector-store adapter.

Chroma (via ``langchain_chroma``) replaces Step 2's hand-written numpy cosine
store. It stores the qwen3 embeddings on disk and exposes a retriever. We wrap
it so the pipeline can ask for results in the same ``{text, metadata, score}``
dict shape that the rest of the app (and the offline evaluator) expects.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_rag.embeddings import Qwen3Embeddings


@lru_cache(maxsize=1)
def _chroma_instances() -> dict[str, Chroma]:
    """Cache Chroma handles keyed by persist dir (avoid reopening on hot reload)."""
    return {}


class LangChainStore:
    """Thin wrapper over a LangChain Chroma collection."""

    def __init__(
        self,
        persist_dir: str | Path = "chroma_db",
        embedding: Any | None = None,
        collection_name: str = "rag_docs",
    ) -> None:
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.embeddings: Any = embedding or Qwen3Embeddings()
        self._db: Chroma | None = None

    def _db_instance(self) -> Chroma:
        if self._db is None:
            self._db = _chroma_instances().get(self.persist_dir)
        if self._db is None:
            self._db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
                collection_metadata={"hnsw:space": "cosine"},
            )
            _chroma_instances()[self.persist_dir] = self._db
        return self._db

    def add_documents(self, chunk_dicts: list[dict[str, Any]]) -> None:
        """Add chunk dicts {text, metadata} to the collection."""
        if not chunk_dicts:
            return
        docs = [
            Document(page_content=c["text"], metadata=c.get("metadata") or {})
            for c in chunk_dicts
        ]
        self._db_instance().add_documents(docs)

    def count(self) -> int:
        return self._db_instance()._collection.count()  # type: ignore[attr-defined]

    def search(
        self, query: str, top_k: int = 3, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Return top-k results as {text, metadata, score} dicts (Step 2 shape)."""
        query_vec = self.embeddings.embed_query(query)
        collection = self._db_instance()._collection  # type: ignore[attr-defined]
        if collection.count() == 0:
            return []
        top_k = min(top_k, collection.count())
        hits = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        results: list[dict[str, Any]] = []
        documents = (hits.get("documents") or [[]])[0]
        metadatas = (hits.get("metadatas") or [[]])[0]
        distances = (hits.get("distances") or [[]])[0]
        for text, metadata, distance in zip(documents, metadatas, distances):
            similarity = float(1.0 - distance)
            if similarity < min_score:
                break
            results.append({
                "text": text,
                "metadata": metadata or {},
                "score": similarity,
            })
        return results

    def clear(self) -> None:
        try:
            self._db_instance().delete_collection()
        except Exception:
            pass
        self._db = None
        _chroma_instances().pop(self.persist_dir, None)
