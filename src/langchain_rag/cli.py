"""Command-line interface for the LangChain RAG pipeline (Task 17).

Dense-only CLI alongside the Streamlit app. Drives the same
``Pipeline.ask()`` and exposes an authenticated admin ``rollback``.
Lazy imports per command keep ``--help`` fast and avoid model loads.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="langchain-rag",
    help="LangChain RAG: ingest / search / ask / eval / prompt / admin rollback (dense-only).",
    no_args_is_help=True,
)
prompt_app = typer.Typer(
    name="prompt",
    help="Inspect live prompt versions and rollback targets (read-only).",
    no_args_is_help=True,
)
app.add_typer(prompt_app)


def _registry_path() -> Path:
    from langchain_rag import prompt_registry as pr

    return pr._DEFAULT_REGISTRY_PATH


def _load_pipeline() -> Any:
    from langchain_rag.pipeline import Pipeline

    p = Pipeline()
    p.load_documents()
    return p


def _fmt_chunk(i: int, src: dict[str, Any]) -> str:
    score = src.get("score")
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
    source = (src.get("metadata") or {}).get("source", "?")
    text = (src.get("text") or "")[:200].replace("\n", " ")
    return f"[{i}] {source} score={score_str}\n    {text}"


def _load_registry() -> Any:
    from langchain_rag.prompt_registry import PromptRegistry

    registry = PromptRegistry()
    if not registry.load():
        typer.echo(f"No registry found at {_registry_path()}", err=True)
        raise typer.Exit(1)
    return registry


def _accuracy(record: dict[str, Any]) -> float | None:
    scores = record.get("eval_scores") or {}
    acc = scores.get("accuracy")
    if isinstance(acc, (int, float)):
        return float(acc)
    return None


def _schema_names(schema: dict[str, Any] | None) -> list[str]:
    """Return the output-contract keys present on a version (for diff display)."""
    return sorted((schema or {}).keys())


def _schema_diff(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> list[str]:
    b = before or {}
    a = after or {}
    changed = [k for k in set(b) | set(a) if b.get(k) != a.get(k)]
    return sorted(changed)


@prompt_app.command("current")
def prompt_current(
    prompt: str = typer.Option("RAG_ANSWER", "--prompt", help="Prompt id"),
) -> None:
    """Show the active approved version, its output contract and eval scores."""
    registry = _load_registry()
    try:
        record = registry.get(prompt, approved_only=True)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    accuracy = _accuracy(record)
    schema = record.get("output_schema") or {}
    live_since = _approved_timestamp(registry, record["key"])

    typer.echo(f"Prompt:      {prompt}")
    typer.echo(f"Live:        {record['key']}  (approved since {live_since or 'unknown'})")
    typer.echo(f"Model:       {record.get('model')}  temp={record.get('temperature')}")
    typer.echo(
        f"Eval acc:    {f'{accuracy:.4f}' if accuracy is not None else 'n/a'}"
    )
    typer.echo(f"Runs logged: {record.get('run_count')}")
    typer.echo("Output contract:")
    for name in _schema_names(schema):
        typer.echo(f"  - {name}: {json.dumps(schema[name], default=str)}")


@prompt_app.command("list")
def prompt_list(
    prompt: str = typer.Option("RAG_ANSWER", "--prompt", help="Prompt id"),
) -> None:
    """List every version with status, eval accuracy and created-at."""
    registry = _load_registry()
    versions = registry.list_versions(prompt)
    if not versions:
        typer.echo(f"No versions found for '{prompt}'.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Versions for {prompt} (rollback targets = RETIRED with eval):")
    for record in versions:
        status = record["status"]
        eligible = "  <-- rollback target" if status == "retired" else ""
        acc = _accuracy(record)
        acc_str = f"{acc:.4f}" if acc is not None else "    -"
        typer.echo(
            f"  {record['version']:<8} {status:<10} acc={acc_str} "
            f"created={record.get('created_at','')[:19]}{eligible}"
        )


def _approved_timestamp(registry: Any, key: str) -> str | None:
    for entry in reversed(registry.history):
        if entry.get("key") == key and entry.get("to_status") == "approved":
            return (entry.get("timestamp") or "")[:19]
    return None


def _admin_token_cmd(token: str | None) -> str | None:
    env_token = os.environ.get("ADMIN_TOKEN")
    if token:
        return token
    if env_token:
        return env_token
    return None


@app.command()
def ingest(
    data_dir: str = typer.Option("data/documents", help="Directory of .txt files"),
    rebuild: bool = typer.Option(False, help="Drop and re-index Chroma"),
    dry_run: str = typer.Option(
        None, help="Parse/chunk only and write preview JSON, no index write"
    ),
) -> None:
    """Ingest documents into the index (or dump a chunk preview)."""
    from langchain_rag.pipeline import Pipeline
    from langchain_rag.splitter import LangChainSplitter

    splitter = LangChainSplitter()
    dir_path = Path(data_dir)
    txt_files = sorted(dir_path.glob("*.txt"))

    if dry_run:
        preview: list[dict[str, Any]] = []
        for file_path in txt_files:
            chunks = splitter.load_and_split(file_path)
            preview.append(
                {"file": file_path.name, "count": len(chunks),
                 "preview": [c["text"][:200] for c in chunks[:3]]}
            )
        out = Path(dry_run)
        out.write_text(json.dumps(preview, indent=2, ensure_ascii=False))
        total = sum(p["count"] for p in preview)
        typer.echo(f"Dry-run preview written to {out}: {total} chunks across {len(preview)} files.")
        if total == 0:
            raise typer.Exit(1)
        return

    p = Pipeline()
    p.load_documents(data_dir=data_dir, force_rebuild=rebuild)
    stats = p.get_stats()
    typer.echo(f"Ingested. chunks={stats.get('num_chunks')} index={'Ready' if stats.get('index_exists') else 'Missing'}")
    if stats.get("num_chunks", 0) == 0:
        typer.echo("No documents found.", err=True)
        raise typer.Exit(1)


@app.command()
def search(
    question: str,
    top_k: int = typer.Option(3, "--top-k", help="Number of chunks"),
    min_score: float = typer.Option(0.3, "--min-score", help="Min similarity"),
) -> None:
    """Retrieve relevant chunks (retriever/reranker only, no LLM)."""
    p = _load_pipeline()
    if p.retriever is None:
        typer.echo("Retriever not initialized.", err=True)
        raise typer.Exit(1)
    results = p.retriever.retrieve(question, top_k=top_k, min_score=min_score)
    if not results:
        typer.echo("No results retrieved.", err=True)
        raise typer.Exit(1)
    for i, src in enumerate(results):
        typer.echo(_fmt_chunk(i, src))


@app.command()
def ask(
    question: str,
    top_k: int = typer.Option(3, "--top-k", help="Number of chunks"),
    min_score: float = typer.Option(0.3, "--min-score", help="Min similarity"),
) -> None:
    """Ask the pipeline and print the answer plus sources."""
    p = _load_pipeline()
    result = p.ask(question, top_k=top_k, min_score=min_score)
    typer.echo(result["answer"])
    typer.echo("")
    typer.echo("Sources:")
    sources = result["sources"]
    if sources:
        for i, src in enumerate(sources):
            typer.echo(_fmt_chunk(i, src))
    else:
        typer.echo("No sources retrieved.")
    typer.echo("")
    typer.echo(
        f"prompt_key: {result.get('prompt_key')}  "
        f"rendered_hash: {(result.get('rendered_hash') or '')[:12]}"
    )


@app.command(name="eval")
def eval_(
    golden: str = typer.Option("eval/golden.jsonl", "--golden", help="Golden JSONL path"),
) -> None:
    """Run the 4-metric keyword evaluation over the golden set."""
    p = _load_pipeline()
    from evaluator import evaluate_with_registry

    prompt_key = p.registry.get(p.prompt_key or "RAG_ANSWER")["key"]
    results = evaluate_with_registry(p, golden, p.registry, prompt_key)

    summary = results.get("summary", {})
    for key in ("context_recall", "context_precision", "faithfulness", "answer_relevance"):
        typer.echo(f"{key}: {summary.get(key, 0.0):.4f}")

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"eval_{ts}.json"
    path.write_text(json.dumps(results, indent=2, default=str))
    typer.echo(f"Saved to {path}")


@app.command()
def rollback(
    prompt: str = typer.Option(..., "--prompt", help="Prompt id, e.g. RAG_ANSWER"),
    to: str = typer.Option(..., "--to", help="Target version, e.g. 1.0.0"),
    reason: str = typer.Option("", "--reason", help="Rollback reason"),
    token: str | None = typer.Option(None, "--token", help="Admin token (or ADMIN_TOKEN env)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the change without applying it"),
    yes: bool = typer.Option(False, "--yes", help="Skip the interactive confirmation"),
) -> None:
    """Admin: roll back a live prompt to a previously-released version.

    Read-only with ``--dry-run`` (shows the output-contract diff and target
    without applying anything). Fails closed unless ADMIN_TOKEN or --token
    is provided. Prompts for confirmation unless --yes.
    """
    resolved = _admin_token_cmd(token)
    if not resolved:
        typer.echo(
            "Unauthenticated: set ADMIN_TOKEN env var or pass --token.",
            err=True,
        )
        raise typer.Exit(1)

    registry = _load_registry()

    try:
        current = registry.get(prompt, approved_only=True)
        target = registry.get(prompt, version=to)
    except ValueError as exc:
        typer.echo(f"Rollback rejected: {exc}", err=True)
        raise typer.Exit(1)

    if target["status"] != "retired":
        typer.echo(
            f"Rollback rejected: can only target a previously RELEASED (retired) "
            f"version; {to} is {target['status']}.",
            err=True,
        )
        raise typer.Exit(1)

    schema_diff = _schema_diff(current.get("output_schema"), target.get("output_schema"))
    acc = _accuracy(target)
    acc_str = f"{acc:.4f}" if acc is not None else "n/a"

    typer.echo(f"⚠ Rolling back {prompt}: {current['version']} (approved) → {to} (was retired)")
    if reason:
        typer.echo(f"  Reason: {reason}")
    typer.echo(f"  Eval acc of target: {acc_str}")
    if schema_diff:
        typer.echo(f"  Output-contract changes: {', '.join(schema_diff)}")
    else:
        typer.echo("  Output-contract: no schema changes")
    typer.echo(f"  {current['version']} will move to: retired")

    if dry_run:
        typer.echo("Dry run (no changes applied).")
        return

    if not yes:
        typer.echo()
        typer.echo("Proceed? [y/N]", nl=False)
        answer = typer.prompt("", default="n", show_default=False).strip().lower()
        if answer not in ("y", "yes"):
            typer.echo("Cancelled.")
            raise typer.Exit(1)

    try:
        record = registry.rollback(prompt, to, reason)
    except ValueError as exc:
        typer.echo(f"Rollback rejected: {exc}", err=True)
        raise typer.Exit(1)

    registry.save()
    typer.echo(f"Done. New approved version: {record['key']}")


def main() -> None:
    app()
