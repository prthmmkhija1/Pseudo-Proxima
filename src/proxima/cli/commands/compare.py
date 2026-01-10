"""Compare command implementation - Multi-backend comparison workflows.

This module provides:
- Multi-backend execution comparison
- Performance metrics analysis
- Side-by-side result display
- Export comparison reports
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer

from proxima.backends.registry import backend_registry
from proxima.cli.formatters import TableFormatter, echo_output
from proxima.cli.prompts import prompt_multi_select
from proxima.cli.workflows import (
    CompareOptions,
    CompareWorkflow,
    WorkflowContext,
)
from proxima.core.executor import Executor
from proxima.core.planner import Planner
from proxima.core.state import ExecutionStateMachine
from proxima.utils.logging import get_logger

app = typer.Typer(name="compare", help="Compare execution across backends")


@app.command()
def main(
    ctx: typer.Context,
    objective: str = typer.Argument(..., help="Objective to compare"),
    backends: list[str] = typer.Option(
        None, "--backend", "-b", help="Backends to compare (can specify multiple)"
    ),
    all_backends: bool = typer.Option(
        False, "--all", "-a", help="Compare across all available backends"
    ),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Run comparisons in parallel"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Save comparison report to file"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress display"),
):
    """Compare execution results across multiple backends.

    Examples:
        proxima compare "bell state" --backend aer_simulator --backend statevector
        proxima compare "grover search" --all
        proxima compare demo --output comparison.json
    """
    ctx.ensure_object(dict)
    settings = ctx.obj.get("settings")
    dry_run = ctx.obj.get("dry_run", False)
    quiet = ctx.obj.get("quiet", False)
    output_format = ctx.obj.get("output_format", "text")

    logger = get_logger("cli.compare")

    # Determine backends to compare
    if all_backends:
        available = backend_registry.list_available()
        backend_list = [b.name for b in available]
    elif backends:
        backend_list = list(backends)
    else:
        # Interactive selection
        available = backend_registry.list_available()
        if not available:
            typer.echo("No backends available for comparison", err=True)
            raise typer.Exit(1)

        backend_list = prompt_multi_select(
            "Select backends to compare:",
            options=[b.name for b in available],
            default=[settings.backends.default_backend],
        )

    if len(backend_list) < 2:
        typer.echo("Need at least 2 backends for comparison", err=True)
        raise typer.Exit(1)

    logger.info("compare.start", objective=objective, backends=backend_list)

    if not quiet:
        typer.echo(f"Comparing: {objective}")
        typer.echo(f"Backends: {', '.join(backend_list)}")

    # Create workflow context
    workflow_ctx = WorkflowContext(
        settings=settings,
        dry_run=dry_run,
        verbose=ctx.obj.get("verbose", 0),
        quiet=quiet,
        output_format=output_format,
        no_progress=no_progress,
    )

    # Dry-run mode
    if dry_run:
        plan = {
            "objective": objective,
            "backends": backend_list,
            "parallel": parallel,
            "mode": "dry-run",
        }
        echo_output(ctx, plan, format=output_format)
        return

    # Run comparison
    options = CompareOptions(
        objective=objective,
        backends=backend_list,
        parallel=parallel,
    )
    workflow = CompareWorkflow(workflow_ctx, options)
    result = workflow.run()

    if not result.success:
        typer.echo(f"Comparison failed: {result.error}", err=True)
        raise typer.Exit(1)

    # Process results
    comparison_data = result.output
    results = comparison_data.get("results", {})
    comparison = comparison_data.get("comparison", {})

    # Display results
    if output_format == "json":
        echo_output(ctx, comparison_data, format="json")
    elif output_format == "table":
        _display_table(results, comparison)
    else:
        _display_text(results, comparison, quiet)

    # Save to file if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(comparison_data, indent=2, default=str),
            encoding="utf-8",
        )
        typer.echo(f"\nReport saved to {output_file}")

    logger.info("compare.complete", backends=backend_list)


def _display_table(results: dict[str, Any], comparison: dict[str, Any]) -> None:
    """Display comparison as table."""
    rows = []
    for backend, data in results.items():
        rows.append(
            {
                "backend": backend,
                "time_s": f"{data.get('execution_time', 0):.3f}",
                "state": data.get("state", "unknown"),
                "speedup": f"{comparison.get('speedup', {}).get(backend, 1.0):.2f}x",
            }
        )

    formatter = TableFormatter()
    typer.echo(formatter.format(rows, title="Backend Comparison"))

    fastest = comparison.get("fastest_backend")
    if fastest:
        typer.echo(f"\nFastest: {fastest}")


def _display_text(
    results: dict[str, Any],
    comparison: dict[str, Any],
    quiet: bool,
) -> None:
    """Display comparison as text."""
    if not quiet:
        typer.echo("\n--- Results ---")

    for backend, data in results.items():
        exec_time = data.get("execution_time", 0)
        state = data.get("state", "unknown")
        speedup = comparison.get("speedup", {}).get(backend, 1.0)

        typer.echo(f"\n{backend}:")
        typer.echo(f"  Time: {exec_time:.3f}s")
        typer.echo(f"  State: {state}")
        typer.echo(f"  Speedup: {speedup:.2f}x")

    fastest = comparison.get("fastest_backend")
    if fastest and not quiet:
        typer.echo(f"\n[OK] Fastest backend: {fastest}")


@app.command("report")
def report_cmd(
    ctx: typer.Context,
    input_file: Path = typer.Argument(..., help="Comparison results file"),
    format: str = typer.Option("text", "--format", "-f", help="Output format"),
):
    """Generate a comparison report from saved results.

    Examples:
        proxima compare report comparison.json
        proxima compare report comparison.json --format table
    """
    if not input_file.exists():
        typer.echo(f"File not found: {input_file}", err=True)
        raise typer.Exit(1)

    try:
        data = json.loads(input_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON: {e}", err=True)
        raise typer.Exit(1)

    results = data.get("results", {})
    comparison = data.get("comparison", {})

    if format == "table":
        _display_table(results, comparison)
    elif format == "json":
        typer.echo(json.dumps(data, indent=2))
    else:
        _display_text(results, comparison, quiet=False)


@app.command("quick")
def quick_cmd(
    ctx: typer.Context,
    objective: str = typer.Argument("demo", help="Objective to compare"),
):
    """Quick comparison between default and fastest backends.

    Examples:
        proxima compare quick "bell state"
    """
    ctx.ensure_object(dict)
    settings = ctx.obj.get("settings")
    quiet = ctx.obj.get("quiet", False)

    logger = get_logger("cli.compare.quick")

    # Get two backends: default and one other
    default_backend = settings.backends.default_backend
    available = backend_registry.list_available()
    other_backends = [b.name for b in available if b.name != default_backend]

    if not other_backends:
        typer.echo("Only one backend available, cannot compare", err=True)
        raise typer.Exit(1)

    compare_backend = other_backends[0]
    backend_list = [default_backend, compare_backend]

    if not quiet:
        typer.echo(f"Quick comparison: {default_backend} vs {compare_backend}")

    results = {}
    for backend in backend_list:
        fsm = ExecutionStateMachine()
        planner = Planner(fsm)
        executor = Executor(fsm)

        start = time.time()
        plan = planner.plan(objective)
        executor.run(plan)
        elapsed = time.time() - start

        results[backend] = {
            "time": elapsed,
            "state": fsm.state,
        }

    # Determine winner
    times = {b: d["time"] for b, d in results.items()}
    fastest = min(times, key=times.get)
    slowest = max(times, key=times.get)
    speedup = times[slowest] / times[fastest] if times[fastest] > 0 else 1.0

    typer.echo(f"\n{fastest} is {speedup:.2f}x faster than {slowest}")
    typer.echo(f"  {fastest}: {times[fastest]:.3f}s")
    typer.echo(f"  {slowest}: {times[slowest]:.3f}s")

    logger.info("compare.quick.complete", fastest=fastest, speedup=speedup)
