import numpy as np
import pytest

from langchain_rag.embeddings import Qwen3Embeddings
from langchain_rag.prompt_registry import PromptRegistry
from langchain_rag.vectorstore import _chroma_instances


class FakeEmbeddingModel:
    """Mimics qwen3_embed.TextEmbedding — no download, no network.

    Implements both interfaces:
    - ``embed`` / ``query_embed`` — qwen3's raw API, used when passed as the
      model returned by ``Qwen3Embeddings._get_model`` (test_embeddings).
    - ``embed_documents`` / ``embed_query`` — LangChain's ``Embeddings``
      interface, used when passed to Chroma as the ``embedding_function``
      (test_vectorstore).
    """

    DIM = 1024

    @staticmethod
    def _unit_vector() -> np.ndarray:
        vec = np.random.rand(FakeEmbeddingModel.DIM)
        return vec / np.linalg.norm(vec)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._unit_vector() for _ in texts]

    def query_embed(self, text: str) -> list[np.ndarray]:
        return [self._unit_vector()]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._unit_vector().tolist() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._unit_vector().tolist()


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingModel:
    """Fake embedding model for offline tests."""
    return FakeEmbeddingModel()


@pytest.fixture
def empty_registry():
    """Fresh in-memory registry — no disk, no prior state."""
    return PromptRegistry()


@pytest.fixture
def tmp_registry(tmp_path):
    """Registry with a temp-file save path for persistence tests."""
    def _save(registry):
        registry.save(tmp_path / "registry.json")
    return _save


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset Qwen3Embeddings class-level _model singleton between tests."""
    Qwen3Embeddings._model = None
    yield
    Qwen3Embeddings._model = None


@pytest.fixture(autouse=True)
def _clear_chroma_cache():
    """Clear the shared Chroma handle cache between tests to avoid stale handles.

    ``_chroma_instances()`` caches ``Chroma`` handles keyed by persist dir. Without
    clearing, a handle opened in one test (backed by a deleted ``tmp_path`` dir)
    would be reused by a later test, causing stale-state or file-not-found errors.
    """
    _chroma_instances().clear()
    yield
    _chroma_instances().clear()