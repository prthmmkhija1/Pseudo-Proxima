"""Compare command implementation."""

from __future__ import annotations

import typer

from proxima.utils.logging import get_logger

app = typer.Typer(name="compare", help="Run multi-backend comparison (stub)")


@app.command()
def main(
    ctx: typer.Context,
    backends: list[str] = typer.Option(
        ["cirq", "qiskit"], "--backend", "-b", help="Backends to compare"
    ),
    objective: str = typer.Argument("demo", help="Objective to compare"),
):
    """Stub comparison command for Phase 1 scaffold."""

    ctx.ensure_object(dict)
    logger = get_logger("cli.compare")
    logger.info("compare.start", objective=objective, backends=backends)
    typer.echo(f"Comparing backends {backends} for objective '{objective}' (stub)")
