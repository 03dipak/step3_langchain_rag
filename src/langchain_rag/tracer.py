"""LangSmith tracing helpers.

LangSmith tracing is opt-in: it only wires up when a real ``LANGSMITH_API_KEY``
is present (not the placeholder). When enabled, retrieval and generation steps
are wrapped with ``@traceable`` so every ``ask()`` produces spans in LangSmith.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

LANGSMITH_PLACEHOLDER = "ls_your_key_here"


def tracing_enabled() -> bool:
    """True only if LANGSMITH_API_KEY is set and not the placeholder."""
    key = os.getenv("LANGSMITH_API_KEY", "")
    return bool(key) and key != LANGSMITH_PLACEHOLDER


def run_with_trace(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Wrap a callable with LangSmith tracing if enabled, else call directly."""
    if not tracing_enabled():
        return fn(*args, **kwargs)
    from langsmith import traceable

    return traceable(name=name)(fn)(*args, **kwargs)


def trace_retrieve(fn: Callable[..., Any], question: str, top_k: int) -> Any:
    """Trace a retrieval step."""
    return run_with_trace("retrieve", fn)


def trace_generate(
    fn: Callable[..., Any], question: str, context_chunks: list[dict[str, Any]]
) -> Any:
    """Trace a generation step."""
    return run_with_trace("generate", fn)


def setup_tracing(project: str = "step3-langchain-rag") -> None:
    """Set LANGCHAIN_* env defaults if a real key is present."""
    if tracing_enabled():
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
