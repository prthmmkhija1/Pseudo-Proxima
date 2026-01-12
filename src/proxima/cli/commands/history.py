"""
History CLI commands.

Browse and manage execution history.
"""

import typer

from proxima.data.store import get_store

app = typer.Typer(help="Browse and manage execution history.")


@app.callback(invoke_without_command=True)
def history_callback(ctx: typer.Context) -> None:
    """Show recent execution history."""
    if ctx.invoked_subcommand is None:
        list_history(limit=10)


@app.command("list")
def list_history(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum entries to show"),
    session_id: str | None = typer.Option(
        None, "--session", "-s", help="Filter by session"
    ),
    backend: str | None = typer.Option(
        None, "--backend", "-b", help="Filter by backend"
    ),
) -> None:
    """List execution history entries."""
    store = get_store()
    results = store.list_results(
        session_id=session_id,
        backend_name=backend,
        limit=limit,
    )

    if not results:
        typer.echo("No execution history found.")
        return

    typer.echo(
        f"\n{'ID':<36} {'Backend':<12} {'Qubits':<8} {'Shots':<8} {'Time':<12} {'Timestamp'}"
    )
    typer.echo("-" * 100)

    for result in results:
        timestamp = (
            result.timestamp.strftime("%Y-%m-%d %H:%M") if result.timestamp else "N/A"
        )
        typer.echo(
            f"{result.id:<36} {result.backend_name:<12} {result.qubit_count:<8} "
            f"{result.shots:<8} {result.execution_time_ms:>8.1f}ms  {timestamp}"
        )


@app.command("show")
def show_result(
    result_id: str = typer.Argument(..., help="Result ID to display"),
) -> None:
    """Show details of a specific execution result."""
    store = get_store()
    result = store.get_result(result_id)

    if not result:
        typer.echo(f"Result not found: {result_id}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Result: {result.id}")
    typer.echo(f"{'='*60}")
    typer.echo(f"Backend:        {result.backend_name}")
    typer.echo(f"Session ID:     {result.session_id}")
    typer.echo(f"Circuit Name:   {result.circuit_name or 'N/A'}")
    typer.echo(f"Qubits:         {result.qubit_count}")
    typer.echo(f"Shots:          {result.shots}")
    typer.echo(f"Execution Time: {result.execution_time_ms:.2f}ms")
    typer.echo(f"Memory Used:    {result.memory_used_mb:.2f}MB")
    typer.echo(f"Timestamp:      {result.timestamp}")

    if result.counts:
        typer.echo("\nMeasurement Counts:")
        for state, count in sorted(result.counts.items()):
            typer.echo(f"  {state}: {count}")

    if result.metadata:
        typer.echo("\nMetadata:")
        for key, value in result.metadata.items():
            typer.echo(f"  {key}: {value}")


@app.command("delete")
def delete_result(
    result_id: str = typer.Argument(..., help="Result ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a specific execution result."""
    store = get_store()
    result = store.get_result(result_id)

    if not result:
        typer.echo(f"Result not found: {result_id}", err=True)
        raise typer.Exit(1)

    if not confirm:
        typer.confirm(f"Delete result {result_id}?", abort=True)

    if store.delete_result(result_id):
        typer.echo(f"Deleted result: {result_id}")
    else:
        typer.echo("Failed to delete result.", err=True)
        raise typer.Exit(1)


@app.command("clear")
def clear_history(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    session_id: str | None = typer.Option(
        None, "--session", "-s", help="Clear only this session"
    ),
) -> None:
    """Clear execution history."""
    store = get_store()

    if session_id:
        if not confirm:
            typer.confirm(f"Clear all results for session {session_id}?", abort=True)

        results = store.list_results(session_id=session_id, limit=1000)
        deleted = 0
        for result in results:
            if store.delete_result(result.id):
                deleted += 1
        typer.echo(f"Deleted {deleted} results from session {session_id}")
    else:
        if not confirm:
            typer.confirm("Clear ALL execution history?", abort=True)

        results = store.list_results(limit=10000)
        deleted = 0
        for result in results:
            if store.delete_result(result.id):
                deleted += 1
        typer.echo(f"Deleted {deleted} results")


@app.command("export")
def export_history(
    output_path: str = typer.Argument(..., help="Output file path (CSV or JSON)"),
    session_id: str | None = typer.Option(
        None, "--session", "-s", help="Export only this session"
    ),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Output format (csv|json)"
    ),
) -> None:
    """Export execution history to file."""
    import csv
    import json
    from pathlib import Path

    store = get_store()
    results = store.list_results(session_id=session_id, limit=10000)

    if not results:
        typer.echo("No results to export.")
        return

    output = Path(output_path)

    if format.lower() == "json":
        data = [
            {
                "id": r.id,
                "session_id": r.session_id,
                "backend_name": r.backend_name,
                "circuit_name": r.circuit_name,
                "qubit_count": r.qubit_count,
                "shots": r.shots,
                "execution_time_ms": r.execution_time_ms,
                "memory_used_mb": r.memory_used_mb,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "counts": r.counts,
                "metadata": r.metadata,
            }
            for r in results
        ]
        output.write_text(json.dumps(data, indent=2, default=str))
    else:
        with output.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "session_id",
                    "backend_name",
                    "circuit_name",
                    "qubit_count",
                    "shots",
                    "execution_time_ms",
                    "memory_used_mb",
                    "timestamp",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.id,
                        r.session_id,
                        r.backend_name,
                        r.circuit_name,
                        r.qubit_count,
                        r.shots,
                        r.execution_time_ms,
                        r.memory_used_mb,
                        r.timestamp.isoformat() if r.timestamp else "",
                    ]
                )

    typer.echo(f"Exported {len(results)} results to {output_path}")
