"""Tests for Retriever — mock the store boundary, not the retriever internals."""

from __future__ import annotations

from pytest_mock import MockerFixture

from langchain_rag.retriever import Retriever

_RESULT = {"text": "chunk", "metadata": {"source": "s.txt"}, "score": 0.9}
_EMPTY: list[dict[str, object]] = []


def test_retrieve_returns_store_results(mocker: MockerFixture) -> None:
    """Retrieve returns exactly what the store's search returns."""
    store = mocker.MagicMock()
    store.search.return_value = [_RESULT]

    results = Retriever(store).retrieve("q", top_k=3, min_score=0.5)

    assert results == [_RESULT]


def test_retrieve_forwards_args(mocker: MockerFixture) -> None:
    """store.search is called with the exact query, top_k, min_score."""
    store = mocker.MagicMock()
    store.search.return_value = [_RESULT]

    Retriever(store).retrieve("q", top_k=3, min_score=0.5)

    store.search.assert_called_once_with("q", top_k=3, min_score=0.5)


def test_retrieve_defaults(mocker: MockerFixture) -> None:
    """Defaults top_k=3, min_score=0.0 when not specified."""
    store = mocker.MagicMock()
    store.search.return_value = [_RESULT]

    Retriever(store).retrieve("q")

    store.search.assert_called_once_with("q", top_k=3, min_score=0.0)


def test_retrieve_empty_store(mocker: MockerFixture) -> None:
    """Empty store → retrieve returns []."""
    store = mocker.MagicMock()
    store.search.return_value = _EMPTY

    results = Retriever(store).retrieve("q")

    assert results == []
