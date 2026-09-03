"""Judge-model adapter for DeepEval metrics.

GroqJudge is the default: free tier, fast, reliable for judging. Free-tier TPM
limits are tight and DeepEval fires metric calls in parallel, so every call
waits out 429s with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from deepeval.models import DeepEvalBaseLLM  # type: ignore[attr-defined]
from groq import RateLimitError
from langchain_groq import ChatGroq
from pydantic import SecretStr

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 10

# DeepEval fires metric calls in parallel; the free Groq tier is TPM-limited.
# A shared semaphore throttles concurrent judge calls so we don't blow the
# tokens-per-minute budget all at once. Free tier refills usable tokens only
# ~1000/7s, so serializing (1) is the reliable default; the env var lets users
# trade throughput for parallelism on higher tiers.
_MAX_CONCURRENT = int(os.getenv("JUDGE_MAX_CONCURRENT", "1"))
_concurrency = asyncio.Semaphore(_MAX_CONCURRENT)

_RETRY_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


class GroqJudge(DeepEvalBaseLLM):
    """Groq-hosted LLM used as the judge for DeepEval metrics.

    Free-tier TPM limits are tight and DeepEval fires metric calls in parallel,
    so every call is throttled through a shared semaphore and waits out 429s
    using the retry-after time parsed from the error message.
    """

    def __init__(self, model: str | None = None, temperature: float = 0) -> None:
        self._name: str = model or os.getenv("JUDGE_MODEL") or DEFAULT_GROQ_MODEL
        self.model = ChatGroq(
            model=self._name,
            temperature=temperature,
            api_key=SecretStr(os.getenv("GROQ_API_KEY") or ""),
        )

    def load_model(self):  # type: ignore[no-untyped-def]
        return self.model

    @staticmethod
    def _coerce_result(raw: str, schema: Any) -> Any:
        """If a pydantic schema is requested, parse the model's JSON into it."""
        if schema is None:
            return raw
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json")
        return schema.model_validate_json(text)

    @staticmethod
    def _retry_delay(exc: BaseException) -> float:
        """Extract the retry-after seconds from a Groq 429 message, else fall back."""
        msg = str(exc)
        m = _RETRY_RE.search(msg)
        if m:
            return float(m.group(1)) + 0.5
        return 8.0

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        schema = kwargs.get("schema")
        delay = 2.0
        for attempt in range(MAX_RETRIES):
            try:
                raw = str(self.model.invoke(prompt).content)
                return self._coerce_result(raw, schema)
            except RateLimitError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(self._retry_delay(e))
            except (json.JSONDecodeError, ValueError):
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError("unreachable")

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        schema = kwargs.get("schema")
        delay = 2.0
        for attempt in range(MAX_RETRIES):
            async with _concurrency:
                try:
                    raw = str((await self.model.ainvoke(prompt)).content)
                    return self._coerce_result(raw, schema)
                except RateLimitError as e:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(self._retry_delay(e))
                except (json.JSONDecodeError, ValueError):
                    if attempt == MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
        raise RuntimeError("unreachable")

    def get_model_name(self) -> str:
        return self._name
