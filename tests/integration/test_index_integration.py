"""Integration tests — real Qwen3 embeddings + real Chroma.

Marked ``integration`` so they are deselected from the default ``pytest`` run
(no model download / no slow loads in CI). Run explicitly with
``uv run pytest -m integration``. The ONNX embedding model loads **once** via a
module-scoped fixture; the LLM chain is stubbed so no API key is needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from langchain_rag.embeddings import Qwen3Embeddings
from langchain_rag.pipeline import Pipeline
from langchain_rag.prompt_registry import PromptRegistry
from langchain_rag.vectorstore import LangChainStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder() -> Qwen3Embeddings:
    """One module-scoped embedder — ONNX qwen3 model loads exactly once."""
    return Qwen3Embeddings()


@pytest.fixture()
def store(tmp_path, embedder: Qwen3Embeddings) -> LangChainStore:
    """A real Chroma store over a temp persist dir, sharing the module embedder."""
    return LangChainStore(persist_dir=tmp_path / "chroma", embedding=embedder)


def test_embedder_dense_dim(embedder: Qwen3Embeddings) -> None:
    """qwen3-embed produces 1024-dim vectors (assert dimension, not a literal)."""
    vec = embedder.embed_query("gradient descent")
    assert len(vec) == 1024


def test_end_to_end_embed_upsert_search(
    store: LangChainStore, embedder: Qwen3Embeddings
) -> None:
    """Real embed -> upsert -> count -> search returns the right chunk + score."""
    chunk: dict[str, Any] = {
        "text": "Gradient descent minimizes a loss function by iterating "
        "opposite to the gradient.",
        "metadata": {"source": "ml_notes.txt", "index": 0},
    }
    store.add_documents([chunk])
    assert store.count() == 1

    results = store.search(
        chunk["text"], top_k=1, min_score=0.0
    )
    assert len(results) == 1
    hit = results[0]
    assert set(hit) == {"text", "metadata", "score"}
    assert hit["text"] == chunk["text"]
    assert hit["metadata"]["source"] == "ml_notes.txt"
    # A near-identical re-embed scores high under cosine space (1 - distance).
    assert 0.7 <= hit["score"] <= 1.0


def _write_corpus(tmp_path) -> Any:
    """Seed a couple of .txt docs in a temp corpus dir."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(
        "Neural networks learn by backpropagation of error gradients.",
        encoding="utf-8",
    )
    (docs / "b.txt").write_text(
        "Attention mechanisms let transformers weigh token relevance.",
        encoding="utf-8",
    )
    return docs


def _approved_registry() -> PromptRegistry:
    r = PromptRegistry()
    key = r.register(
        "RAG_ANSWER",
        "Answer from context: {context}\nQuestion: {question}",
        ["context", "question"],
    )
    r.promote(key)
    r.record_eval_scores(key, {"accuracy": 0.9}, "e1")
    r.promote(key)
    return r


def _make_real_pipeline(tmp_path, monkeypatch) -> Any:
    """Real pipeline on a temp Chroma dir; stub only the LLM generator."""
    # Pipeline() builds a LangChainGenerator that eagerly calls build_chain(),
    # which would require a live LLM key. Stub that boundary so construction
    # is hermetic; we still exercise real embeddings + Chroma below.
    monkeypatch.setattr(
        "langchain_rag.prompts.LangChainPromptAdapter.build_chain", lambda self, pid: None
    )
    p = Pipeline()
    p.registry = _approved_registry()
    # The generator was constructed with the *old* empty registry; point it at
    # the approved one so refusal_response reads real data (retrieval runs for real).
    p.generator.registry = p.registry
    p.store = LangChainStore(persist_dir=tmp_path / "chroma")
    # Stub only the generation step; retrieval stays real.
    monkeypatch.setattr(p.generator, "generate", lambda *a, **k: "real retrieval answer")
    return p


def test_pipeline_load_idempotent(tmp_path, monkeypatch) -> None:
    """load_documents twice -> no duplicate chunks; force_rebuild re-indexes."""
    docs = _write_corpus(tmp_path)
    p = _make_real_pipeline(tmp_path, monkeypatch)

    p.load_documents(data_dir=str(docs))
    first = p.store.count()

    p.load_documents(data_dir=str(docs))
    second = p.store.count()
    assert second == first  # idempotent: no duplicate chunks on re-load

    p.load_documents(data_dir=str(docs), force_rebuild=True)
    assert p.store.count() == first  # rebuilt index has the same chunk count


def test_pipeline_ask_real(tmp_path, monkeypatch) -> None:
    """ask() end-to-end with real embeddings/Chroma, stubbed LLM generate."""
    docs = _write_corpus(tmp_path)
    p = _make_real_pipeline(tmp_path, monkeypatch)
    p.load_documents(data_dir=str(docs))

    result = p.ask("attention mechanisms transformer")
    assert set(result) == {"answer", "sources", "prompt_key", "rendered_hash"}
    assert result["answer"] == "real retrieval answer"
    assert result["sources"], "expected real retrieved sources"
    src = result["sources"][0]
    assert src["metadata"]["source"] in {"a.txt", "b.txt"}
    assert "score" in src
