"""Tests for PromptRegistry — pure logic, no mocks, tmp_path for persistence.

Covers the full version lifecycle, the evidence gate, rollback, persistence,
and the output_schema QA gate (the "no silent output change" rule). Aim: 100%.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from langchain_rag.prompt_registry import PromptRegistry

TEMPLATE = "Answer from context: {context}\nQuestion: {question}"


def _register(registry: PromptRegistry, prompt_id: str = "RAG_ANSWER") -> str:
    return registry.register(
        prompt_id,
        TEMPLATE,
        ["context", "question"],
        change_note="V1",
    )


def _approve(registry: PromptRegistry, key: str) -> PromptRegistry:
    registry.promote(key)
    registry.record_eval_scores(key, {"accuracy": 0.9}, "e1")
    registry.promote(key)
    return registry


def test_register_creates_draft(empty_registry: PromptRegistry) -> None:
    """register creates a 1.0.0 draft with default field values."""
    key = _register(empty_registry)
    record = empty_registry.prompts[key]

    assert record["status"] == "draft"
    assert record["version"] == "1.0.0"
    assert record["prompt_id"] == "RAG_ANSWER"
    assert record["run_count"] == 0


def test_register_immutable_no_overwrite(empty_registry: PromptRegistry) -> None:
    """register never overwrites an existing version; each call makes a new key."""
    key1 = _register(empty_registry)
    key2 = empty_registry.register("RAG_ANSWER", TEMPLATE, ["c", "q"])

    assert key1 != key2
    assert key1 in empty_registry.prompts
    assert key2 in empty_registry.prompts
    # both versions coexist (1.0.0 and 1.0.0-patch-bump) — none overwritten
    assert empty_registry.prompts[key1]["version"] != empty_registry.prompts[key2]["version"]


def test_register_bumps_minor_from_approved(empty_registry: PromptRegistry) -> None:
    """Registering after an approved version bumps minor (1.0.0 -> 1.1.0)."""
    key1 = _register(empty_registry)
    _approve(empty_registry, key1)

    key2 = empty_registry.register(
        "RAG_ANSWER", TEMPLATE, ["c", "q"], change_note="V2"
    )

    assert empty_registry.prompts[key2]["version"] == "1.1.0"


def test_register_default_output_schema(empty_registry: PromptRegistry) -> None:
    """register(..., output_schema=None) yields the default schema."""
    key = _register(empty_registry)
    assert (
        empty_registry.prompts[key]["output_schema"]
        == PromptRegistry.DEFAULT_OUTPUT_SCHEMA
    )


def test_promote_lifecycle(empty_registry: PromptRegistry) -> None:
    """promote walks draft -> testing -> approved."""
    key = _register(empty_registry)
    empty_registry.promote(key)
    assert empty_registry.prompts[key]["status"] == "testing"
    empty_registry.record_eval_scores(key, {"accuracy": 0.9}, "e1")
    empty_registry.promote(key)
    assert empty_registry.prompts[key]["status"] == "approved"


def test_promote_gate_blocks_without_evidence(empty_registry: PromptRegistry) -> None:
    """Can't promote to approved without required eval evidence."""
    key = _register(empty_registry)
    empty_registry.promote(key)  # -> testing
    with pytest.raises(ValueError):
        empty_registry.promote(key)  # -> approved blocked (no accuracy)


def test_promote_retired_terminal(empty_registry: PromptRegistry) -> None:
    """Cannot promote a retired version."""
    key = _register(empty_registry)
    _approve(empty_registry, key)
    key2 = empty_registry.register("RAG_ANSWER", TEMPLATE, ["c", "q"])
    _approve(empty_registry, key2)  # auto-retires key (1.0.0)

    with pytest.raises(ValueError):
        empty_registry.promote(key)


def test_get_approved_only(empty_registry: PromptRegistry) -> None:
    """get(approved_only=True) serves the approved version."""
    key = _register(empty_registry)
    _approve(empty_registry, key)
    assert empty_registry.get("RAG_ANSWER")["key"] == key


def test_get_no_approved_raises(empty_registry: PromptRegistry) -> None:
    """get with no approved version raises ValueError."""
    _register(empty_registry)
    with pytest.raises(ValueError):
        empty_registry.get("RAG_ANSWER")


def test_get_by_version(empty_registry: PromptRegistry) -> None:
    """get(version=...) returns the specific version."""
    key = _register(empty_registry)
    assert empty_registry.get("RAG_ANSWER", version="1.0.0")["key"] == key


def test_get_missing_version_raises(empty_registry: PromptRegistry) -> None:
    """get with a nonexistent version raises."""
    _register(empty_registry)
    with pytest.raises(ValueError):
        empty_registry.get("RAG_ANSWER", version="9.9.9")


def test_rollback_returns_to_prior(empty_registry: PromptRegistry) -> None:
    """rollback makes the retired 1.0.0 approved again and serves it."""
    key1 = _register(empty_registry)
    _approve(empty_registry, key1)
    key2 = empty_registry.register("RAG_ANSWER", TEMPLATE, ["c", "q"])
    _approve(empty_registry, key2)  # 1.1.0 approved, 1.0.0 retired

    empty_registry.rollback("RAG_ANSWER", "1.0.0", reason="regression")

    assert empty_registry.prompts[key1]["status"] == "approved"
    assert empty_registry.prompts[key2]["status"] == "retired"
    assert empty_registry.get("RAG_ANSWER")["version"] == "1.0.0"


def test_rollback_to_unreleased_raises(empty_registry: PromptRegistry) -> None:
    """Can only roll back to a previously released (retired) version."""
    _register(empty_registry)  # draft only
    with pytest.raises(ValueError):
        empty_registry.rollback("RAG_ANSWER", "1.0.0")


def test_rollback_missing_version_raises(empty_registry: PromptRegistry) -> None:
    """Rollback to a nonexistent version raises."""
    _register(empty_registry)
    with pytest.raises(ValueError):
        empty_registry.rollback("RAG_ANSWER", "9.9.9")


def test_set_output_schema_on_approved_demotes_to_testing(
    empty_registry: PromptRegistry,
) -> None:
    """⭐ QA gate: editing schema on an approved version drops it to testing."""
    key = _register(empty_registry)
    _approve(empty_registry, key)
    assert empty_registry.prompts[key]["status"] == "approved"

    empty_registry.set_output_schema(
        key, {"length_policy": "3-4 sentences"}, reason="tighter length"
    )

    assert empty_registry.prompts[key]["status"] == "testing"


def test_set_output_schema_records_history_diff(
    empty_registry: PromptRegistry,
) -> None:
    """set_output_schema records a {from_schema, to_schema, reason} history entry."""
    key = _register(empty_registry)
    _approve(empty_registry, key)

    empty_registry.set_output_schema(key, {"length_policy": "3-4"}, reason="tighten")

    entry = next(
        h for h in empty_registry.history if h.get("change") == "output_schema"
    )
    assert entry["change"] == "output_schema"
    assert "from_schema" in entry
    assert "to_schema" in entry
    assert entry["reason"] == "tighten"


def test_set_output_schema_on_retired_raises(empty_registry: PromptRegistry) -> None:
    """Retired versions' schema is frozen — set_output_schema raises."""
    key1 = _register(empty_registry)
    _approve(empty_registry, key1)
    key2 = empty_registry.register("RAG_ANSWER", TEMPLATE, ["c", "q"])
    _approve(empty_registry, key2)  # retires key1

    with pytest.raises(ValueError):
        empty_registry.set_output_schema(key1, {"format": "text"})


def test_rollback_restores_served_schema(empty_registry: PromptRegistry) -> None:
    """Rollback to 1.0.0 makes get() serve 1.0.0's schema (output contract)."""
    key1 = _register(empty_registry)
    empty_registry.promote(key1)
    empty_registry.record_eval_scores(key1, {"accuracy": 0.9}, "e1")
    empty_registry.promote(key1)  # 1.0.0 approved with default schema

    key2 = empty_registry.register(
        "RAG_ANSWER",
        TEMPLATE,
        ["c", "q"],
        output_schema={"refusal_string": "custom refusal {question}"},
    )
    empty_registry.promote(key2)
    empty_registry.record_eval_scores(key2, {"accuracy": 0.95}, "e2")
    empty_registry.promote(key2)  # 1.1.0 approved, 1.0.0 retired

    empty_registry.rollback("RAG_ANSWER", "1.0.0", reason="refusal regressed")

    served = empty_registry.get("RAG_ANSWER")
    assert served["version"] == "1.0.0"
    assert served["output_schema"] == PromptRegistry.DEFAULT_OUTPUT_SCHEMA


def test_record_eval_scores_stores_metadata(empty_registry: PromptRegistry) -> None:
    """record_eval_scores sets {scores, evaluated_at, eval_run_id}."""
    key = _register(empty_registry)
    empty_registry.record_eval_scores(key, {"accuracy": 0.9, "faithfulness": 0.8}, "e1")
    scores = empty_registry.prompts[key]["eval_scores"]

    assert scores["accuracy"] == 0.9
    assert scores["faithfulness"] == 0.8
    assert scores["eval_run_id"] == "e1"
    assert "evaluated_at" in scores


def test_record_eval_scores_retired_raises(empty_registry: PromptRegistry) -> None:
    """Cannot record eval for a retired prompt."""
    key1 = _register(empty_registry)
    _approve(empty_registry, key1)
    key2 = empty_registry.register("RAG_ANSWER", TEMPLATE, ["c", "q"])
    _approve(empty_registry, key2)  # retires key1

    with pytest.raises(ValueError):
        empty_registry.record_eval_scores(key1, {"accuracy": 1.0}, "e3")


def test_log_run_increments_and_appends(empty_registry: PromptRegistry) -> None:
    """log_run bumps run_count and appends a run entry."""
    key = _register(empty_registry)
    empty_registry.log_run(
        key, "hash1", ["a.txt"], "out1", 120, {"prompt_tokens": 10}
    )
    empty_registry.log_run(
        key, "hash2", ["b.txt"], "out2", 200, {"prompt_tokens": 12}
    )

    record = empty_registry.prompts[key]
    assert record["run_count"] == 2
    assert "last_run_at" in record
    assert len(record["runs"]) == 2
    assert record["runs"][0]["latency_ms"] == 120


def test_compare_versions_orders_by_faithfulness(empty_registry: PromptRegistry) -> None:
    """compare_versions sorts evaluated versions by faithfulness desc."""
    key1 = _register(empty_registry)
    empty_registry.record_eval_scores(key1, {"faithfulness": 0.7}, "e1")
    key2 = empty_registry.register("RAG_ANSWER", TEMPLATE, ["c", "q"])
    empty_registry.record_eval_scores(key2, {"faithfulness": 0.9}, "e2")

    ordered = empty_registry.compare_versions("RAG_ANSWER")

    assert ordered[0]["key"] == key2
    assert ordered[1]["key"] == key1


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """save writes atomically; a fresh registry loads it back."""
    path = tmp_path / "registry.json"
    registry = PromptRegistry()
    key = _register(registry)
    _approve(registry, key)

    registry.save(path)

    loaded = PromptRegistry()
    assert loaded.load(path) is True
    assert loaded.get("RAG_ANSWER")["version"] == "1.0.0"
    assert loaded.prompts[key]["template"] == TEMPLATE


def test_load_missing_file_returns_false(tmp_path: Path) -> None:
    """load with no file returns False."""
    assert PromptRegistry().load(tmp_path / "nope.json") is False


def test_get_status_history(empty_registry: PromptRegistry) -> None:
    """get_status_history returns entries for a given key."""
    key = _register(empty_registry)
    empty_registry.promote(key)
    history = empty_registry.get_status_history(key)

    assert history[0]["to_status"] == "draft"
    assert history[1]["to_status"] == "testing"


def test_list_versions_sorted(empty_registry: PromptRegistry) -> None:
    """list_versions returns all versions sorted ascending by version."""
    key1 = _register(empty_registry)
    key2 = empty_registry.register("RAG_ANSWER", TEMPLATE, ["c", "q"])

    versions = empty_registry.list_versions("RAG_ANSWER")

    assert [v["key"] for v in versions] == [key1, key2]


def test_set_output_schema_else_branch(empty_registry: PromptRegistry) -> None:
    """set_output_schema on a non-approved (testing) version doesn't demote status."""
    key = _register(empty_registry)
    empty_registry.promote(key)  # -> testing

    empty_registry.set_output_schema(key, {"format": "text"})

    assert empty_registry.prompts[key]["status"] == "testing"
    assert empty_registry.prompts[key]["output_schema"]["format"] == "text"


def test_get_no_versions_raises(empty_registry: PromptRegistry) -> None:
    """get(approved_only=False) with no versions raises ValueError."""
    with pytest.raises(ValueError):
        empty_registry.get("RAG_ANSWER", approved_only=False)


def test_promote_missing_key_raises(empty_registry: PromptRegistry) -> None:
    """promote with a nonexistent key raises."""
    with pytest.raises(ValueError):
        empty_registry.promote("NOPE_V1.0.0")


def test_record_eval_scores_missing_key_raises(empty_registry: PromptRegistry) -> None:
    """record_eval_scores with a nonexistent key raises."""
    with pytest.raises(ValueError):
        empty_registry.record_eval_scores("NOPE_V1.0.0", {"accuracy": 1.0}, "e1")


def test_log_run_missing_key_raises(empty_registry: PromptRegistry) -> None:
    """log_run with a nonexistent key raises."""
    with pytest.raises(ValueError):
        empty_registry.log_run("NOPE_V1.0.0", "h", [], "o", 1, {})


def test_load_corrupt_returns_false(tmp_path: Path) -> None:
    """load with an unparseable file returns False (crash-safe)."""
    path = tmp_path / "registry.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    assert PromptRegistry().load(path) is False


def test_load_missing_payload_keys_returns_false(tmp_path: Path) -> None:
    """load with missing 'prompts'/'history' keys returns False."""
    path = tmp_path / "registry.json"
    path.write_text('{"foo": 1}', encoding="utf-8")
    assert PromptRegistry().load(path) is False


def test_register_no_versions_branch(empty_registry: PromptRegistry) -> None:
    """register on an empty registry for a new prompt_id starts at 1.0.0."""
    empty_registry.register("OTHER", TEMPLATE, ["c", "q"])
    assert empty_registry.prompts["OTHER_V1.0.0"]["parent_version"] is None


def test_load_backwards_compat_fills_schema(tmp_path: Path) -> None:
    """Loading a legacy registry (no output_schema) fills the default schema."""
    path = tmp_path / "registry.json"
    path.write_text(
        '{"prompts": {"RAG_ANSWER_V1.0.0": {"key": "RAG_ANSWER_V1.0.0", '
        '"prompt_id": "RAG_ANSWER", "version": "1.0.0", "template": "t", '
        '"input_variables": [], "model": "m", "temperature": 0.1, '
        '"status": "approved", "eval_scores": {}, "run_count": 0}}, '
        '"history": []}',
        encoding="utf-8",
    )

    registry = PromptRegistry()
    assert registry.load(path) is True
    assert (
        registry.prompts["RAG_ANSWER_V1.0.0"]["output_schema"]
        == PromptRegistry.DEFAULT_OUTPUT_SCHEMA
    )


def test_collapsed_schema_empty() -> None:
    """_collapsed_schema(None) returns '{}' for empty audit diffs."""
    assert PromptRegistry._collapsed_schema(None) == "{}"


def test_register_patch_bump_from_unapproved(empty_registry: PromptRegistry) -> None:
    """Registering after an unapproved (only draft/testing) version patch-bumps."""
    key1 = _register(empty_registry)  # 1.0.0, draft
    assert empty_registry.prompts[key1]["status"] == "draft"

    key2 = empty_registry.register("RAG_ANSWER", TEMPLATE, ["c", "q"])
    # no approved version exists -> patch bump 1.0.0 -> 1.0.1
    assert empty_registry.prompts[key2]["version"] == "1.0.1"
