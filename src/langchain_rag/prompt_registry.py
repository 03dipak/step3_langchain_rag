"""Versioned prompt registry with immutable versions, rollback, and evidence.

This is the **shared, cross-cutting** component that threads through Steps 1-7.
It is intentionally framework-agnostic: it owns *what* the prompt is (template,
policy, status, eval evidence) and never imports LangChain. LangChain's
``PromptTemplate`` is merely an *adapter* over this registry (see ``prompts.py``).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_REGISTRY_PATH = Path(__file__).parent / "prompt_registry.json"


class PromptRegistry:
    """Versioned prompt registry with immutable versions, rollback, and evidence."""

    NEXT_STATUS: dict[str, str] = {
        "draft": "testing",
        "testing": "approved",
        "approved": "retired",
    }

    # Minimum eval keys required before a version may be promoted to "approved".
    REQUIRED_EVAL_KEYS: tuple[str, ...] = ("accuracy",)

    # Default output contract used when register() is called with output_schema=None.
    DEFAULT_OUTPUT_SCHEMA: dict[str, Any] = {
        "format": "markdown",
        "shape": {"answer": "str", "sources": "list[dict]"},
        "length_policy": "4-6 sentences",
        "citation_policy": "inline [source-N] markers; N = index into retrieved sources",
        "refusal_string": "I don't have enough context to answer: {question}",
        "display": "answer text, then 'Sources:' list with title + score",
    }

    def __init__(self) -> None:
        self.prompts: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []

    def _timestamp(self) -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _version_key(version: str) -> tuple[int, int, int]:
        parts = version.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    @staticmethod
    def _format_version(key: tuple[int, int, int]) -> str:
        return f"{key[0]}.{key[1]}.{key[2]}"

    def _has_required_evidence(self, record: dict[str, Any]) -> bool:
        scores = record.get("eval_scores") or {}
        return all(k in scores for k in self.REQUIRED_EVAL_KEYS)

    def _activate(self, key: str, reason: str, action: str) -> dict[str, Any]:
        """Make ``key`` the single approved version; retire the previously approved one.

        Release and rollback share this single atomic operation, so there is
        exactly one implementation of the approve + retire rule.
        """
        record = self.prompts[key]

        if not self._has_required_evidence(record):
            missing = [k for k in self.REQUIRED_EVAL_KEYS
                       if k not in (record.get("eval_scores") or {})]
            raise ValueError(
                f"Cannot approve '{key}': missing required eval evidence {missing}. "
                f"Call record_eval_scores() first."
            )

        from_status = record["status"]
        current = self._current_approved(record["prompt_id"])
        if current is not None and current["key"] != key:
            current["status"] = "retired"
            self._record_history(
                current["key"], "approved", "retired",
                reason or f"{action} supersedes {current['version']}",
            )
        if record["status"] != "approved":
            record["status"] = "approved"
            self._record_history(key, from_status, "approved", reason or action)
        return record

    def save(self, path: str | Path = _DEFAULT_REGISTRY_PATH) -> None:
        """Persist registry state atomically (temp file + os.replace)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"prompts": self.prompts, "history": self.history}

        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def load(self, path: str | Path = _DEFAULT_REGISTRY_PATH) -> bool:
        """Restore registry state from disk. Returns False if missing/corrupt."""
        if not Path(path).exists():
            return False
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            self.prompts = payload["prompts"]
            self.history = payload["history"]
        except (OSError, ValueError, KeyError):
            return False
        # Backward-compat: older registry files lack output_schema. Fill defaults
        # so every version still carries an output contract (guarantees G1/G2).
        for record in self.prompts.values():
            if "output_schema" not in record:
                record["output_schema"] = dict(self.DEFAULT_OUTPUT_SCHEMA)
        return True

    def _record_history(
        self, key: str, from_status: str | None, to_status: str, reason: str
    ) -> None:
        self.history.append({
            "key": key,
            "from_status": from_status,
            "to_status": to_status,
            "timestamp": self._timestamp(),
            "reason": reason,
        })

    def _increment_version(self, version: str) -> str:
        major, minor, patch = version.split(".")
        return f"{major}.{minor}.{int(patch) + 1}"

    def _next_minor(self, version: str) -> str:
        parts = version.split(".")
        return f"{parts[0]}.{int(parts[1]) + 1}.0"

    def _current_approved(self, prompt_id: str) -> dict[str, Any] | None:
        approved = [
            r for r in self.prompts.values()
            if r["prompt_id"] == prompt_id and r["status"] == "approved"
        ]
        if not approved:
            return None
        return max(approved, key=lambda r: self._version_key(r["version"]))

    def register(
        self,
        prompt_id: str,
        template: str,
        input_variables: list[str],
        model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        temperature: float = 0.1,
        change_note: str = "",
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        """Register a new version. Never overwrites. Auto-increments version.

        ``output_schema`` is the version's **output contract** (format, shape,
        length/citation/refusal rules). When omitted, a sensible default is used.
        """
        versions = [r for r in self.prompts.values() if r["prompt_id"] == prompt_id]
        current_approved = self._current_approved(prompt_id)

        if not versions:
            version = "1.0.0"
        elif current_approved:
            version = self._next_minor(current_approved["version"])
        else:
            last = max(self._version_key(r["version"]) for r in versions)
            version = self._increment_version(self._format_version(last))

        key = f"{prompt_id}_V{version}"
        if key in self.prompts:
            raise ValueError(f"Version {key} already exists (immutable).")

        record: dict[str, Any] = {
            "key": key,
            "prompt_id": prompt_id,
            "version": version,
            "template": template,
            "input_variables": input_variables,
            "model": model,
            "temperature": temperature,
            "status": "draft",
            "eval_scores": {},
            "run_count": 0,
            "created_at": self._timestamp(),
            "change_note": change_note,
            "parent_version": current_approved["version"] if current_approved else None,
            "output_schema": dict(output_schema or self.DEFAULT_OUTPUT_SCHEMA),
        }
        self.prompts[key] = record
        self._record_history(key, None, "draft", change_note or "Initial registration")
        return key

    def set_output_schema(self, key: str, schema: dict[str, Any], reason: str = "") -> None:
        """Replace a version's output_schema.

        Gating (QA rule — no silent output change in production):
        - On a **draft/testing** version, the schema is simply updated.
        - On an **approved** version, the schema is updated AND the version is
          sent back to ``testing`` (the approve->testing gate), so it must be
          re-scored (re-eval) before it can be re-approved.
        - Retired versions are rejected (schema of a shipped-then-retired prompt
          is frozen).

        A history entry records the {from_schema, to_schema, reason} diff so QA
        can see exactly what user-visible output behavior changed on any update
        or rollback.
        """
        record = self.prompts[key]
        if record["status"] == "retired":
            raise ValueError(
                f"Cannot change output_schema of retired version '{key}' (frozen)."
            )

        from_schema = record.get("output_schema")
        collapsed_from = self._collapsed_schema(from_schema)
        collapsed_to = self._collapsed_schema(schema)

        record["output_schema"] = dict(schema)

        if record["status"] == "approved":
            record["status"] = "testing"
            self._record_schema_history(key, collapsed_from, collapsed_to,
                                        reason or f"Schema change demotes {key} to testing")
            self._record_history(key, "approved", "testing",
                                 reason or f"Schema change demotes {key} to testing")
        else:
            self._record_schema_history(
                key, collapsed_from, collapsed_to,
                reason or f"output_schema updated on {key}",
            )
            self._record_history(
                key, None, None, reason or f"output_schema updated on {key}"
            )

    @staticmethod
    def _collapsed_schema(schema: dict[str, Any] | None) -> str:
        """Stable, compact summary of a schema for audit diffs."""
        if not schema:
            return "{}"
        return json.dumps(schema, sort_keys=True, separators=(",", ":"))

    def _record_schema_history(
        self, key: str, from_schema: str, to_schema: str, reason: str
    ) -> None:
        self.history.append({
            "key": key,
            "change": "output_schema",
            "from_schema": from_schema,
            "to_schema": to_schema,
            "timestamp": self._timestamp(),
            "reason": reason,
        })

    def get(
        self,
        prompt_id: str,
        *,
        version: str | None = None,
        approved_only: bool = True,
    ) -> dict[str, Any]:
        """Get a prompt version."""
        if version is not None:
            key = f"{prompt_id}_V{version}"
            if key not in self.prompts:
                raise ValueError(f"Version {key} does not exist.")
            return self.prompts[key]

        if approved_only:
            record = self._current_approved(prompt_id)
            if record is None:
                raise ValueError(f"No approved version found for '{prompt_id}'.")
            return record

        records = [r for r in self.prompts.values() if r["prompt_id"] == prompt_id]
        if not records:
            raise ValueError(f"No versions found for '{prompt_id}'.")
        return records[-1]

    def promote(self, key: str, reason: str = "") -> dict[str, Any]:
        """Promote a prompt to the next status in the lifecycle."""
        if key not in self.prompts:
            raise ValueError(f"Version {key} does not exist.")

        record = self.prompts[key]
        current_status = record["status"]
        if current_status == "retired":
            raise ValueError(f"Cannot promote '{key}' - it is RETIRED (terminal state).")

        next_status = self.NEXT_STATUS[current_status]
        if next_status == "approved":
            return self._activate(key, reason, f"Promotion of {key}")

        record["status"] = next_status
        self._record_history(key, current_status, next_status,
                             reason or f"Promoted to {next_status}")
        return record

    def rollback(self, prompt_id: str, to_version: str, reason: str = "") -> dict[str, Any]:
        """Rollback to a previously released (retired) version."""
        target_key = f"{prompt_id}_V{to_version}"
        if target_key not in self.prompts:
            raise ValueError(f"Target version {to_version} does not exist.")

        target = self.prompts[target_key]
        if target["status"] != "retired":
            raise ValueError(
                f"Can only roll back to a previously released (retired) version, "
                f"got {to_version} ({target['status']})."
            )
        return self._activate(target_key, reason or f"Rolled back to {to_version}", "Rollback")

    def list_versions(self, prompt_id: str) -> list[dict[str, Any]]:
        records = [r for r in self.prompts.values() if r["prompt_id"] == prompt_id]
        return sorted(records, key=lambda r: self._version_key(r["version"]))

    def get_status_history(self, key: str) -> list[dict[str, Any]]:
        return [h for h in self.history if h["key"] == key]

    def record_eval_scores(self, key: str, scores: dict[str, float], eval_run_id: str) -> None:
        if key not in self.prompts:
            raise ValueError(f"Version {key} does not exist.")
        record = self.prompts[key]
        if record["status"] == "retired":
            raise ValueError("Cannot record eval for retired prompt.")
        record["eval_scores"] = {
            **scores,
            "evaluated_at": self._timestamp(),
            "eval_run_id": eval_run_id,
        }

    def log_run(
        self,
        key: str,
        rendered_hash: str,
        retrieved_doc_ids: list[str],
        output: str,
        latency_ms: int,
        token_usage: dict[str, int],
        error: str | None = None,
    ) -> None:
        """Record a production run; increments run_count, updates last_run_at."""
        if key not in self.prompts:
            raise ValueError(f"Version {key} does not exist.")
        record = self.prompts[key]
        record["run_count"] = record.get("run_count", 0) + 1
        record["last_run_at"] = self._timestamp()
        record.setdefault("runs", []).append({
            "rendered_hash": rendered_hash,
            "retrieved_doc_ids": retrieved_doc_ids,
            "output": output,
            "latency_ms": latency_ms,
            "token_usage": token_usage,
            "error": error,
            "timestamp": self._timestamp(),
        })

    def compare_versions(self, prompt_id: str) -> list[dict[str, Any]]:
        """Compare all eval'd versions, sorted by faithfulness (best first)."""
        versions = self.list_versions(prompt_id)
        evaluated = [v for v in versions if v.get("eval_scores")]
        return sorted(
            evaluated,
            key=lambda v: v["eval_scores"].get("faithfulness", 0.0),
            reverse=True,
        )
