"""
Session CLI commands.

Manage execution sessions.
"""

from datetime import datetime

import typer

from proxima.data.store import StoredSession, get_store

app = typer.Typer(help="Manage execution sessions.")


@app.callback(invoke_without_command=True)
def session_callback(ctx: typer.Context) -> None:
    """Show active session or list sessions."""
    if ctx.invoked_subcommand is None:
        list_sessions(limit=10)


@app.command("list")
def list_sessions(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum sessions to show"),
) -> None:
    """List all sessions."""
    store = get_store()
    sessions = store.list_sessions(limit=limit)

    if not sessions:
        typer.echo("No sessions found.")
        return

    typer.echo(f"\n{'ID':<36} {'Name':<20} {'Results':<10} {'Created'}")
    typer.echo("-" * 90)

    for session in sessions:
        created = (
            session.created_at.strftime("%Y-%m-%d %H:%M")
            if session.created_at
            else "N/A"
        )
        name = session.name or "(unnamed)"
        typer.echo(f"{session.id:<36} {name:<20} {session.result_count:<10} {created}")


@app.command("new")
def new_session(
    name: str | None = typer.Option(None, "--name", "-n", help="Session name"),
    agent_file: str | None = typer.Option(
        None, "--agent", "-a", help="Associated agent file"
    ),
) -> None:
    """Create a new session."""
    store = get_store()

    session = StoredSession(
        name=name,
        agent_file=agent_file,
    )
    session_id = store.create_session(session)

    typer.echo(f"Created session: {session_id}")
    if name:
        typer.echo(f"Name: {name}")


@app.command("show")
def show_session(
    session_id: str = typer.Argument(..., help="Session ID to display"),
) -> None:
    """Show details of a specific session."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        typer.echo(f"Session not found: {session_id}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Session: {session.id}")
    typer.echo(f"{'='*60}")
    typer.echo(f"Name:         {session.name or '(unnamed)'}")
    typer.echo(f"Agent File:   {session.agent_file or 'N/A'}")
    typer.echo(f"Result Count: {session.result_count}")
    typer.echo(f"Created:      {session.created_at}")
    typer.echo(f"Updated:      {session.updated_at}")

    if session.metadata:
        typer.echo("\nMetadata:")
        for key, value in session.metadata.items():
            typer.echo(f"  {key}: {value}")

    # Show recent results
    results = store.list_results(session_id=session_id, limit=5)
    if results:
        typer.echo("\nRecent Results:")
        for result in results:
            typer.echo(
                f"  - {result.id[:8]}... ({result.backend_name}, {result.qubit_count} qubits)"
            )


@app.command("delete")
def delete_session(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a session and all its results."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        typer.echo(f"Session not found: {session_id}", err=True)
        raise typer.Exit(1)

    if not confirm:
        typer.confirm(
            f"Delete session '{session.name or session_id}' and all {session.result_count} results?",
            abort=True,
        )

    if store.delete_session(session_id):
        typer.echo(f"Deleted session: {session_id}")
    else:
        typer.echo("Failed to delete session.", err=True)
        raise typer.Exit(1)


@app.command("rename")
def rename_session(
    session_id: str = typer.Argument(..., help="Session ID to rename"),
    name: str = typer.Argument(..., help="New session name"),
) -> None:
    """Rename a session."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        typer.echo(f"Session not found: {session_id}", err=True)
        raise typer.Exit(1)

    old_name = session.name or "(unnamed)"

    # Update the session name
    session.name = name
    session.updated_at = datetime.utcnow()

    # Persist the update
    if store.update_session(session):
        typer.echo(f"Renamed session from '{old_name}' to '{name}'")
    else:
        typer.echo("Failed to rename session.", err=True)
        raise typer.Exit(1)


@app.command("resume")
def resume_session(
    session_id: str = typer.Argument(..., help="Session ID to resume"),
) -> None:
    """Resume a previous session."""
    store = get_store()
    session = store.get_session(session_id)

    if not session:
        typer.echo(f"Session not found: {session_id}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Resuming session: {session.name or session_id}")
    typer.echo(f"Results in session: {session.result_count}")

    if session.agent_file:
        typer.echo(f"Agent file: {session.agent_file}")
        typer.echo("\nRun 'proxima agent run <file>' to continue execution")
    else:
        typer.echo("\nSession resumed. New executions will be added to this session.")
