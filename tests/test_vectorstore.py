"""Tests for LangChainStore — real Chroma on tmp_path, no network."""

from __future__ import annotations

from typing import Any

from langchain_rag.vectorstore import LangChainStore


def _make_store(tmp_path: Any, fake_embeddings: object) -> LangChainStore:
    """Helper: create a LangChainStore on tmp_path with fake embeddings."""
    return LangChainStore(
        persist_dir=str(tmp_path / "chroma_db"),
        embedding=fake_embeddings,
        collection_name="test_docs",
    )


def test_count_returns_zero_empty(tmp_path: object, fake_embeddings: object) -> None:
    """Empty store → count() == 0."""
    store = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    assert store.count() == 0


def test_add_documents_stores_chunks(tmp_path: object, fake_embeddings: object) -> None:
    """add_documents() → count() > 0."""
    store = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    chunks = [
        {"text": "Python is high level", "metadata": {"source": "python.txt"}},
        {"text": "Gradient descent optimizes", "metadata": {"source": "ml.txt"}},
    ]

    store.add_documents(chunks)

    assert store.count() == 2


def test_add_documents_empty_list(tmp_path: object, fake_embeddings: object) -> None:
    """add_documents([]) → no crash, count stays 0."""
    store = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    store.add_documents([])
    assert store.count() == 0


def test_search_returns_step2_shape(tmp_path: object, fake_embeddings: object) -> None:
    """search() returns [{text, metadata, score}] dicts."""
    store = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    store.add_documents([
        {"text": "Python basics", "metadata": {"source": "python.txt"}},
        {"text": "Machine learning", "metadata": {"source": "ml.txt"}},
    ])

    results = store.search("python", top_k=2, min_score=0.0)

    assert len(results) == 2
    for r in results:
        assert "text" in r
        assert "metadata" in r
        assert "score" in r
        assert isinstance(r["score"], float)


def test_search_empty_store_returns_empty(tmp_path: object, fake_embeddings: object) -> None:
    """Empty store → search() == []."""
    store = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    results = store.search("anything", top_k=3, min_score=0.0)
    assert results == []


def test_min_score_filters_results(tmp_path: object, fake_embeddings: object) -> None:
    """High min_score → fewer results returned."""
    store = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    store.add_documents([
        {"text": "Python basics", "metadata": {"source": "python.txt"}},
        {"text": "Machine learning", "metadata": {"source": "ml.txt"}},
    ])

    all_results = store.search("python", top_k=10, min_score=0.0)
    filtered = store.search("python", top_k=10, min_score=0.99)

    assert len(filtered) <= len(all_results)


def test_clear_empties_collection(tmp_path: object, fake_embeddings: object) -> None:
    """clear() → count() == 0."""
    store = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    store.add_documents([
        {"text": "To be cleared", "metadata": {"source": "temp.txt"}},
    ])
    assert store.count() == 1

    store.clear()

    assert store.count() == 0


def test_persistence_reopens_without_readd(tmp_path: object, fake_embeddings: object) -> None:
    """New LangChainStore on same dir → count persists."""
    store1 = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    store1.add_documents([
        {"text": "Persistent doc", "metadata": {"source": "persist.txt"}},
    ])
    assert store1.count() == 1

    store2 = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    assert store2.count() == 1


def test_search_top_k_limits_results(tmp_path: object, fake_embeddings: object) -> None:
    """top_k=1 returns only 1 result."""
    store = _make_store(tmp_path, fake_embeddings)  # type: ignore[arg-type]
    store.add_documents([
        {"text": "Doc one", "metadata": {"source": "a.txt"}},
        {"text": "Doc two", "metadata": {"source": "b.txt"}},
        {"text": "Doc three", "metadata": {"source": "c.txt"}},
    ])

    results = store.search("query", top_k=1, min_score=0.0)

    assert len(results) == 1
