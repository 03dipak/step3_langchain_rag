"""Tests for Pipeline — mock collaborators, test orchestration."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pytest_mock import MockerFixture

from langchain_rag.pipeline import Pipeline, RerankableRetriever
from langchain_rag.prompt_registry import PromptRegistry

# Accessing mocks through the pipeline's typed-ish attributes confuses mypy; an
# "Any pipeline" lets us call .return_value / .assert_called freely.
APipeline = Any


def _approved_registry() -> PromptRegistry:
    """Registry with an approved RAG_ANSWER version."""
    r = PromptRegistry()
    key = r.register(
        "RAG_ANSWER",
        "Answer from context: {context}\nQuestion: {question}",
        ["context", "question"],
        change_note="V1",
    )
    r.promote(key)
    r.record_eval_scores(key, {"accuracy": 0.9}, "e1")
    r.promote(key)
    return r


def _make_pipeline(mocker: MockerFixture) -> APipeline:
    # Pipeline() builds a LangChainGenerator which eagerly calls build_chain();
    # that needs an already-approved prompt in the registry. Patch build_chain
    # so construction succeeds offline, then swap in a mocked generator below.
    mocker.patch(
        "langchain_rag.prompts.LangChainPromptAdapter.build_chain", return_value=None
    )
    p = cast(APipeline, Pipeline())
    p.registry = _approved_registry()
    # Replace the real collaborators with mocks.
    p.store = mocker.MagicMock()
    p.store.persist_dir = "chroma_db"
    p.store.count.return_value = 2
    p.splitter = mocker.MagicMock()
    p.splitter.load_directory.return_value = [
        {"text": "c1", "metadata": {"source": "a.txt"}},
        {"text": "c2", "metadata": {"source": "b.txt"}},
    ]
    p.generator = mocker.MagicMock()
    p.generator.refusal_response.return_value = "refusal text"
    p.generator.generate.return_value = "generated answer"
    return p


def _with_retriever(p: APipeline, mocker: MockerFixture, results: list[dict]) -> Any:
    """Install a mocked retriever on the pipeline; return the mock."""
    retriever = mocker.MagicMock()
    retriever.retrieve.return_value = results
    p.retriever = retriever
    return retriever


def test_ask_without_load_raises(mocker: MockerFixture) -> None:
    """ask() before load_documents raises ValueError."""
    p = _make_pipeline(mocker)
    p.retriever = None
    with pytest.raises(ValueError):
        p.ask("q")


def test_ask_returns_step2_shape(mocker: MockerFixture) -> None:
    """ask() returns {answer, sources, prompt_key, rendered_hash}."""
    p = _make_pipeline(mocker)
    _with_retriever(
        p, mocker, [{"text": "c1", "metadata": {"source": "a.txt"}, "score": 0.9}]
    )

    result = p.ask("q", top_k=3, min_score=0.5)

    assert set(result.keys()) == {"answer", "sources", "prompt_key", "rendered_hash"}
    assert result["answer"] == "generated answer"
    assert result["sources"][0]["metadata"]["source"] == "a.txt"
    assert result["prompt_key"].startswith("RAG_ANSWER_V")
    assert len(result["rendered_hash"]) == 64  # sha256 hex


def test_ask_uses_approved_prompt_key(mocker: MockerFixture) -> None:
    """ask() uses the approved version's key."""
    p = _make_pipeline(mocker)
    _with_retriever(
        p, mocker, [{"text": "c1", "metadata": {"source": "a.txt"}, "score": 0.9}]
    )

    result = p.ask("q", top_k=3, min_score=0.5)

    assert result["prompt_key"] == "RAG_ANSWER_V1.0.0"


def test_ask_no_results_returns_refusal(mocker: MockerFixture) -> None:
    """ask() with no retrieved results returns the refusal answer + empty sources."""
    p = _make_pipeline(mocker)
    _with_retriever(p, mocker, [])

    result = p.ask("q", top_k=3, min_score=0.5)

    assert result["answer"] == "refusal text"
    assert result["sources"] == []
    p.generator.generate.assert_not_called()


def test_ask_call_args_forwards_top_k_min_score(mocker: MockerFixture) -> None:
    """ask() forwards top_k/min_score to the retriever."""
    p = _make_pipeline(mocker)
    retriever = _with_retriever(
        p, mocker, [{"text": "c1", "metadata": {"source": "a.txt"}, "score": 0.9}]
    )

    p.ask("q", top_k=5, min_score=0.7)

    retriever.retrieve.assert_called_once_with("q", top_k=5, min_score=0.7)


def test_genuine_reranker_retrieve_overfetches(mocker: MockerFixture) -> None:
    """RerankableRetriever.retrieve over-fetches 2x then caps to top_n."""
    base = mocker.MagicMock()
    base.retrieve.return_value = [
        {"text": f"c{i}", "metadata": {"source": f"{i}.txt"}, "score": 0.9}
        for i in range(4)
    ]
    reranker = mocker.MagicMock()
    reranker.rerank.side_effect = lambda q, c, top_n=None: c[:top_n]

    rr = RerankableRetriever(base, reranker)
    result = rr.retrieve("q", top_k=3, min_score=0.5)

    base.retrieve.assert_called_once_with("q", top_k=6, min_score=0.5)
    reranker.rerank.assert_called_once()
    assert len(result) == 3


def test_get_stats(mocker: MockerFixture) -> None:
    """get_stats reflects store count and retriever init state."""
    p = _make_pipeline(mocker)
    _with_retriever(p, mocker, [])

    stats = p.get_stats()

    assert stats["num_chunks"] == 2
    assert stats["retriever_initialized"] is True
    assert stats["prompt_key"] is None


def test_load_documents_wires_rerankable_retriever(mocker: MockerFixture) -> None:
    """load_documents adds chunks and installs a RerankableRetriever."""
    p = _make_pipeline(mocker)
    p.load_documents(data_dir="data/documents", force_rebuild=True)

    p.splitter.load_directory.assert_called_once_with("data/documents")
    p.store.add_documents.assert_called_once()
    assert isinstance(p.retriever, RerankableRetriever)


def test_load_documents_reopen_skips_indexing(
    mocker: MockerFixture, tmp_path
) -> None:
    """idempotent reopen: existing chroma_db skips clear/load/add.

    RerankableRetriever is still wired over the reopened store.
    """
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (chroma_dir / "chroma.sqlite3").touch()

    p = _make_pipeline(mocker)
    p.store.persist_dir = str(chroma_dir)

    p.load_documents(data_dir="data/documents", force_rebuild=False)

    p.store.clear.assert_not_called()
    p.splitter.load_directory.assert_not_called()
    p.store.add_documents.assert_not_called()
    assert isinstance(p.retriever, RerankableRetriever)
