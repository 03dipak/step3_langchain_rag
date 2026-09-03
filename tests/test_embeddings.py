"""Tests for Qwen3Embeddings — mock the model boundary, not the internals."""

from __future__ import annotations

import numpy as np
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from langchain_rag.embeddings import Qwen3Embeddings


def test_embed_documents_returns_list_of_lists(mocker: MockerFixture, fake_embeddings: object) -> None:
    """embed_documents(["a", "b"]) → list[list[float]] with correct shape."""
    mocker.patch.object(
        Qwen3Embeddings, "_get_model", return_value=fake_embeddings
    )

    emb = Qwen3Embeddings()
    result = emb.embed_documents(["What is Python?", "How does ML work?"])

    assert len(result) == 2
    assert len(result[0]) == 1024
    assert len(result[1]) == 1024
    assert all(isinstance(x, float) for x in result[0])


def test_embed_query_returns_single_vector(mocker: MockerFixture, fake_embeddings: object) -> None:
    """embed_query("text") → list[float], dim 1024."""
    mocker.patch.object(
        Qwen3Embeddings, "_get_model", return_value=fake_embeddings
    )

    emb = Qwen3Embeddings()
    result = emb.embed_query("What is gradient descent?")

    assert isinstance(result, list)
    assert len(result) == 1024
    assert all(isinstance(x, float) for x in result)


def test_embed_documents_converts_numpy_to_float(mocker: MockerFixture, fake_embeddings: object) -> None:
    """FakeModel returns numpy arrays → output is plain float (not np.float64)."""
    mocker.patch.object(
        Qwen3Embeddings, "_get_model", return_value=fake_embeddings
    )

    emb = Qwen3Embeddings()
    result = emb.embed_documents(["test"])

    assert isinstance(result[0][0], float)
    assert not isinstance(result[0][0], np.floating)


def test_model_name_reads_env_var(monkeypatch: MonkeyPatch) -> None:
    """monkeypatch.setenv("EMBED_MODEL", ...) → _model_name() returns it."""
    monkeypatch.setenv("EMBED_MODEL", "custom-model-name")
    assert Qwen3Embeddings._model_name() == "custom-model-name"


def test_model_name_default() -> None:
    """Without env var, _model_name() returns the ONNX default."""
    import os

    os.environ.pop("EMBED_MODEL", None)
    assert Qwen3Embeddings._model_name() == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


def test_singleton_model_reused(mocker: MockerFixture, fake_embeddings: object) -> None:
    """Call embed_documents twice → _get_model called twice (per-call invocation)."""
    spy = mocker.patch.object(
        Qwen3Embeddings, "_get_model", return_value=fake_embeddings
    )

    emb = Qwen3Embeddings()
    emb.embed_documents(["first call"])
    emb.embed_documents(["second call"])

    assert spy.call_count == 2


def test_embed_documents_empty_list(mocker: MockerFixture, fake_embeddings: object) -> None:
    """embed_documents([]) → empty list (no crash)."""
    mocker.patch.object(
        Qwen3Embeddings, "_get_model", return_value=fake_embeddings
    )

    emb = Qwen3Embeddings()
    result = emb.embed_documents([])

    assert result == []