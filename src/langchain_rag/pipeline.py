"""LCEL pipeline wiring the Step 3 LangChain components together.

Exposes the **same public interface as Step 2** (``load_documents``, ``ask``,
``get_stats``, plus ``retriever``/``generator``/``registry``) so the evaluator
(Task 12) and DeepEval (Task 16) run unchanged. The reranker (Task 7) is
inserted transparently behind the retriever via ``RerankableRetriever`` — the
``ask()`` interface never changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from langchain_rag.generator import LangChainGenerator
from langchain_rag.prompt_registry import PromptRegistry
from langchain_rag.reranker import Reranker
from langchain_rag.retriever import Retriever
from langchain_rag.splitter import LangChainSplitter
from langchain_rag.vectorstore import LangChainStore


class RerankableRetriever:
    """Over-fetch at 2x, cross-encode rerank, drop to the caller's top_n.

    This is the ONE place the over-fetch factor lives — no per-caller guessing.
    """

    def __init__(self, base: Retriever, reranker: Reranker) -> None:
        self.base = base
        self.reranker = reranker

    def retrieve(
        self, question: str, top_k: int = 3, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        candidates = self.base.retrieve(
            question, top_k=top_k * 2, min_score=min_score
        )
        return self.reranker.rerank(question, candidates, top_n=top_k)


class Pipeline:
    def __init__(
        self,
        index_path: str = "index.npz",
        prompt_key: str | None = None,
    ) -> None:
        self.index_path = index_path
        self.prompt_key = prompt_key
        self.store = LangChainStore()
        self.splitter = LangChainSplitter()
        self.retriever: Retriever | RerankableRetriever | None = None
        self.registry = PromptRegistry()
        self.generator = LangChainGenerator(self.registry, self.prompt_key or "RAG_ANSWER")

    def load_documents(
        self, data_dir: str = "data/documents", force_rebuild: bool = False
    ) -> None:
        """Index documents into Chroma (idempotent) and wire the rerankable retriever."""
        self.registry.load()

        persist_dir = self.store.persist_dir
        index_exists = os.path.isdir(persist_dir) and bool(
            os.listdir(persist_dir)
        )

        if index_exists and not force_rebuild:
            chunks: list[dict[str, Any]] = []
        else:
            self.store.clear()
            chunks = self.splitter.load_directory(data_dir)
            self.store.add_documents(chunks)

        base = Retriever(self.store)
        self.retriever = RerankableRetriever(
            base, Reranker(top_n=self._default_top_k())
        )

    def _default_top_k(self) -> int:
        return 3

    def ask(
        self, question: str, top_k: int = 3, min_score: float = 0.5
    ) -> dict[str, Any]:
        """Answer a question, returning the Step 2-shaped result dict."""
        if self.retriever is None:
            raise ValueError(
                "Retriever not initialized. Call load_documents() first."
            )
        retriever = self.retriever

        prompt = self.registry.get(self.prompt_key or "RAG_ANSWER")
        prompt_key = prompt["key"]

        from langchain_rag.tracer import trace_retrieve

        results = trace_retrieve(
            lambda: retriever.retrieve(
                question, top_k=top_k, min_score=min_score
            ),
            question,
            top_k,
        )

        rendered_hash = hashlib.sha256(
            json.dumps({**prompt, "question": question}, sort_keys=True).encode()
        ).hexdigest()

        if not results:
            return {
                "answer": self.generator.refusal_response(question),
                "sources": [],
                "prompt_key": prompt_key,
                "rendered_hash": rendered_hash,
            }

        from langchain_rag.tracer import trace_generate

        output = trace_generate(
            lambda: self.generator.generate(question, results),
            question,
            results,
        )

        self.registry.log_run(
            key=prompt_key,
            rendered_hash=rendered_hash,
            retrieved_doc_ids=[r["metadata"]["source"] for r in results],
            output=output,
            latency_ms=0,
            token_usage={},
        )

        return {
            "answer": output,
            "sources": results,
            "prompt_key": prompt_key,
            "rendered_hash": rendered_hash,
        }

    def get_stats(self) -> dict[str, Any]:
        """Return build/runtime stats."""
        persist_dir = self.store.persist_dir
        index_exists = os.path.isdir(persist_dir) and bool(os.listdir(persist_dir))
        num_chunks = self.store.count() if self.retriever is not None else 0
        return {
            "index_exists": index_exists,
            "num_chunks": num_chunks,
            "retriever_initialized": self.retriever is not None,
            "prompt_key": self.prompt_key,
        }
