"""Run command implementation."""

from __future__ import annotations

from typing import Any

import typer

from proxima.core.executor import Executor
from proxima.core.planner import Planner
from proxima.core.state import ExecutionStateMachine
from proxima.utils.logging import get_logger


app = typer.Typer(name="run", help="Execute a simulation")


@app.command()
def main(
    ctx: typer.Context,
    objective: str = typer.Argument("demo", help="Objective or plan target"),
    backend: str = typer.Option(None, "--backend", "-b", help="Backend override"),
):
    """Plan (via model) and execute a run. Execution is stubbed for Phase 1."""

    ctx.ensure_object(dict)
    settings = ctx.obj.get("settings")
    dry_run = ctx.obj.get("dry_run", False)

    logger = get_logger("cli.run")
    logger.info(
        "run.start", objective=objective, backend=backend or settings.backends.default_backend
    )

    fsm = ExecutionStateMachine()
    planner = Planner(fsm)
    executor = Executor(fsm)

    plan = planner.plan(objective)
    result: Any = {"status": "skipped", "reason": "dry-run"} if dry_run else executor.run(plan)

    logger.info("run.finish", state=fsm.state, result=result)
    typer.echo(f"State: {fsm.state}, Result: {result}")
