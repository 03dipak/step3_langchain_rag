"""Tests for Reranker — mock the cross-encoder boundary, not the rerank logic."""

from __future__ import annotations

from typing import Any

from pytest_mock import MockerFixture

from langchain_rag.reranker import Reranker

_CANDIDATES: list[dict[str, Any]] = [
    {"text": "gradient descent updates weights", "metadata": {"source": "ml.txt"}, "score": 0.9},
    {"text": "python is high level", "metadata": {"source": "python.txt"}, "score": 0.8},
    {"text": "unrelated content", "metadata": {"source": "api.txt"}, "score": 0.5},
]


def _patch_model(mocker: MockerFixture, scores: list[float]):
    """Patch _get_model to return a fake cross-encoder yielding given scores."""
    fake = mocker.MagicMock()
    fake.predict.return_value = scores
    return mocker.patch.object(Reranker, "_get_model", return_value=fake)


def test_rerank_attaches_rerank_scores(mocker: MockerFixture) -> None:
    """Each output dict has rerank_score, keeps original score + metadata."""
    _patch_model(mocker, [1.0, 0.5, 0.2])

    out = Reranker(top_n=3).rerank("q", _CANDIDATES)

    assert len(out) == 3
    for row in out:
        assert "rerank_score" in row
        assert "score" in row
        assert row["metadata"]["source"] is not None
        assert row["rerank_score"] in (1.0, 0.5, 0.2)


def test_rerank_sorts_desc(mocker: MockerFixture) -> None:
    """Higher cross-encoder score comes first."""
    _patch_model(mocker, [0.2, 1.0, 0.5])

    out = Reranker(top_n=3).rerank("q", _CANDIDATES)

    assert out[0]["rerank_score"] == 1.0
    assert out[1]["rerank_score"] == 0.5
    assert out[2]["rerank_score"] == 0.2


def test_rerank_caps_top_n(mocker: MockerFixture) -> None:
    """Returns at most top_n results."""
    _patch_model(mocker, [1.0, 0.5, 0.2])

    out = Reranker(top_n=2).rerank("q", _CANDIDATES)

    assert len(out) == 2


def test_rerank_default_top_n(mocker: MockerFixture) -> None:
    """Uses instance top_n when none passed."""
    _patch_model(mocker, [1.0, 0.5, 0.2])

    out = Reranker(top_n=2).rerank("q", _CANDIDATES)

    assert len(out) == 2


def test_rerank_preserves_metadata(mocker: MockerFixture) -> None:
    """metadata["source"] basename survives reranking untouched."""
    _patch_model(mocker, [0.2, 1.0, 0.5])

    out = Reranker(top_n=3).rerank("q", _CANDIDATES)

    sources = {row["metadata"]["source"] for row in out}
    assert sources == {"ml.txt", "python.txt", "api.txt"}


def test_rerank_offline_fallback(mocker: MockerFixture) -> None:
    """Model can't load → candidates returned unchanged (no crash)."""
    mocker.patch.object(Reranker, "_get_model", return_value=None)

    out = Reranker(top_n=2).rerank("q", _CANDIDATES)

    assert out == _CANDIDATES
    assert all("rerank_score" not in row for row in out)


def test_rerank_empty_candidates(mocker: MockerFixture) -> None:
    """Empty candidates → empty list, no model call."""
    _patch_model(mocker, [])

    out = Reranker(top_n=3).rerank("q", [])

    assert out == []


def test_get_model_singleton(mocker: MockerFixture) -> None:
    """_get_model loads the model once and caches it (lazy singleton)."""
    fake = mocker.MagicMock()
    mocker.patch("sentence_transformers.CrossEncoder", return_value=fake)

    r = Reranker()
    first = r._get_model()
    second = r._get_model()

    assert first is fake
    assert second is fake


def test_get_model_offline_returns_none(mocker: MockerFixture) -> None:
    """If CrossEncoder construction raises, _get_model returns None (no crash)."""
    mocker.patch(
        "sentence_transformers.CrossEncoder",
        side_effect=Exception("no network / no cache"),
    )

    assert Reranker()._get_model() is None
