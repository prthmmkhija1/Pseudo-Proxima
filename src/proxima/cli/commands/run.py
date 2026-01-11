"""Run command implementation - Full workflow execution.

This module provides the complete run command with:
- Progress tracking and display
- Consent management
- Multi-backend support
- Output formatting
- Dry-run mode
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from proxima.cli.formatters import echo_output
from proxima.cli.progress import step_context
from proxima.cli.prompts import context_consent
from proxima.cli.workflows import (
    ValidateOptions,
    ValidationWorkflow,
    WorkflowContext,
)
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
    shots: int = typer.Option(None, "--shots", "-s", help="Number of shots"),
    timeout: float = typer.Option(None, "--timeout", "-t", help="Execution timeout in seconds"),
    validate: bool = typer.Option(False, "--validate", "-v", help="Validate config before run"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress display"),
    save_results: bool = typer.Option(True, "--save/--no-save", help="Save results to history"),
):
    """Plan (via model) and execute a run.

    Examples:
        proxima run "create bell state"
        proxima run "quantum teleportation" --backend aer_simulator
        proxima run demo --shots 1000 --validate
    """
    ctx.ensure_object(dict)
    settings = ctx.obj.get("settings")
    dry_run = ctx.obj.get("dry_run", False)
    force = ctx.obj.get("force", False)
    quiet = ctx.obj.get("quiet", False)
    output_format = ctx.obj.get("output_format", "text")
    verbose = ctx.obj.get("verbose", 0)

    logger = get_logger("cli.run")
    effective_backend = backend or settings.backends.default_backend

    if verbose >= 2:
        typer.echo(f"[DEBUG] Starting run with backend: {effective_backend}")

    logger.info(
        "run.start",
        objective=objective,
        backend=effective_backend,
        dry_run=dry_run,
    )

    # Create workflow context
    workflow_ctx = WorkflowContext(
        settings=settings,
        dry_run=dry_run,
        force=force,
        verbose=ctx.obj.get("verbose", 0),
        quiet=quiet,
        output_format=output_format,
        no_progress=no_progress,
    )

    # Validation step (optional)
    if validate:
        if not quiet:
            typer.echo("Validating configuration...")

        validate_options = ValidateOptions(backend=effective_backend)
        validation = ValidationWorkflow(workflow_ctx, validate_options)
        validation_result = validation.run()

        if not validation_result.success:
            typer.echo(f"Validation failed: {validation_result.error}", err=True)
            raise typer.Exit(1)

        if validation_result.output:
            issues = validation_result.output.get("issues", [])
            if issues:
                typer.echo(f"Found {len(issues)} issue(s):")
                for issue in issues:
                    severity = issue.get("severity", "info")
                    msg = issue.get("message", "")
                    icon = "X" if severity == "error" else "!" if severity == "warning" else "i"
                    typer.echo(f"  [{icon}] {msg}")

                if not force and any(i.get("severity") == "error" for i in issues):
                    typer.echo("Fix errors before running (use --force to override)")
                    raise typer.Exit(1)

    # Consent for remote LLM (if applicable)
    if settings.llm.provider.lower() in ("openai", "anthropic", "remote"):
        if not context_consent(
            ctx,
            title="Remote LLM Usage",
            description="This operation may send data to a remote LLM API.",
            details=[
                f"Provider: {settings.llm.provider}",
                f"Model: {settings.llm.model_name}",
            ],
            implications=[
                "Your objective will be processed by external servers",
                "API costs may apply",
            ],
        ):
            typer.echo("Operation cancelled: consent required for remote LLM")
            raise typer.Exit(0)

    # Run workflow with step progress
    steps = [
        "Initialize execution",
        "Plan objective",
        "Validate plan",
        "Execute simulation",
        "Process results",
    ]

    if dry_run:
        steps = ["Initialize execution", "Plan objective (dry-run)"]

    # Import runner before entering progress context
    runner_func: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    runner_error: str | None = None
    try:
        from proxima.core.runner import quantum_runner

        runner_func = quantum_runner
        logger.info("runner.loaded", status="success")
    except Exception as e:
        runner_error = str(e)
        logger.warning("runner.load_failed", error=str(e))

    if not quiet and runner_error:
        typer.echo(f"[!] Runner load failed: {runner_error}")

    try:
        with step_context(
            steps, title=f"Running: {objective}", no_progress=no_progress or quiet
        ) as progress:
            # Step 1: Initialize
            fsm = ExecutionStateMachine()
            planner = Planner(fsm)
            executor = Executor(fsm, runner=runner_func)
            progress.advance()

            # Step 2: Plan
            plan = planner.plan(objective)
            # Add runtime configuration to plan
            plan["backend"] = effective_backend
            plan["shots"] = shots or 1024
            plan["objective"] = objective
            if timeout is not None:
                plan["timeout_seconds"] = timeout
            progress.advance()

            if dry_run:
                # Dry-run: output plan and exit
                result = {
                    "status": "dry-run",
                    "objective": objective,
                    "backend": effective_backend,
                    "plan": plan,
                    "shots": shots,
                }
                echo_output(ctx, result, format=output_format)
                logger.info("run.dry_run", plan=plan)
                return

            # Step 3: Validate plan
            if plan:
                progress.advance()
            else:
                progress.advance(error="No plan generated")

            # Step 4: Execute
            result = executor.run(plan)
            progress.advance()

            # Step 5: Process results
            execution_result = {
                "status": "completed",
                "objective": objective,
                "backend": effective_backend,
                "state": fsm.state,
                "result": result,
            }

            # Save to history (if enabled)
            if save_results:
                try:
                    from datetime import datetime

                    from proxima.data.history import ExecutionResult, execution_history

                    history_entry = ExecutionResult(
                        id=f"run_{int(datetime.now().timestamp())}",
                        backend_name=effective_backend,
                        circuit_name=objective[:50],
                        qubit_count=1,
                        shots=shots or 1024,
                        execution_time_ms=0,
                        counts=result if isinstance(result, dict) else {},
                    )
                    execution_history.add_result(history_entry)
                except Exception as e:
                    logger.warning("run.save_failed", error=str(e))

            progress.advance()

        # Output final result
        logger.info("run.finish", state=fsm.state, result=result)

        if output_format == "json":
            echo_output(ctx, execution_result, format="json")
        elif not quiet:
            typer.echo(f"\n✅ State: {fsm.state}")
            typer.echo(f"⚡ Backend: {effective_backend}")

            if isinstance(result, dict):
                if result.get("status") == "success":
                    typer.echo(
                        f"🎯 Circuit: {result.get('circuit_type', 'unknown')} ({result.get('qubits', 0)} qubits)"
                    )
                    typer.echo(f"⏱️  Execution time: {result.get('execution_time_ms', 0):.2f} ms")
                    typer.echo(f"🔢 Shots: {result.get('shots', 0)}")
                    typer.echo("\n📊 Measurement Results:")

                    counts = result.get("counts", {})
                    for state, data in list(counts.items())[:10]:  # Show top 10
                        count = data.get("count", 0)
                        percentage = data.get("percentage", 0)
                        bar_length = int(percentage / 2)  # Scale to max 50 chars
                        bar = "█" * bar_length
                        typer.echo(f"  |{state}⟩: {percentage:6.2f}% {bar} ({count})")

                    if len(counts) > 10:
                        typer.echo(f"  ... and {len(counts) - 10} more states")
                elif result.get("status") == "error":
                    typer.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
                else:
                    typer.echo(f"  Results: {len(result)} items")

    except KeyboardInterrupt:
        typer.echo("\n[!] Execution cancelled by user")
        raise typer.Exit(130)
    except Exception as e:
        logger.error("run.error", error=str(e))
        typer.echo(f"\n[X] Execution failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("validate")
def validate_cmd(
    ctx: typer.Context,
    backend: str = typer.Option(None, "--backend", "-b", help="Backend to validate"),
    config_path: Path = typer.Option(None, "--config", "-c", help="Config file to validate"),
    strict: bool = typer.Option(False, "--strict", help="Enable strict validation"),
):
    """Validate configuration before execution."""
    ctx.ensure_object(dict)
    settings = ctx.obj.get("settings")
    output_format = ctx.obj.get("output_format", "text")

    workflow_ctx = WorkflowContext(
        settings=settings,
        output_format=output_format,
    )

    options = ValidateOptions(
        config_path=config_path,
        backend=backend,
        strict=strict,
    )

    validation = ValidationWorkflow(workflow_ctx, options)
    result = validation.run()

    if result.success and result.output:
        issues = result.output.get("issues", [])
        valid = result.output.get("valid", True)

        if output_format == "json":
            echo_output(ctx, result.output, format="json")
        else:
            if valid:
                typer.echo("[OK] Configuration is valid")
            else:
                typer.echo("[X] Configuration has issues:")

            for issue in issues:
                severity = issue.get("severity", "info")
                path = issue.get("path", "")
                msg = issue.get("message", "")
                suggestion = issue.get("suggestion", "")

                icon = "X" if severity == "error" else "!" if severity == "warning" else "i"
                typer.echo(f"  [{icon}] [{path}] {msg}")
                if suggestion:
                    typer.echo(f"      Hint: {suggestion}")

        if not valid:
            raise typer.Exit(1)
    else:
        typer.echo(f"Validation error: {result.error}", err=True)
        raise typer.Exit(1)


@app.command("plan")
def plan_cmd(
    ctx: typer.Context,
    objective: str = typer.Argument(..., help="Objective to plan"),
    backend: str = typer.Option(None, "--backend", "-b", help="Target backend"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Save plan to file"),
):
    """Generate an execution plan without running."""
    ctx.ensure_object(dict)
    settings = ctx.obj.get("settings")
    output_format = ctx.obj.get("output_format", "text")
    quiet = ctx.obj.get("quiet", False)

    logger = get_logger("cli.run.plan")

    fsm = ExecutionStateMachine()
    planner = Planner(fsm)

    if not quiet:
        typer.echo(f"Planning: {objective}")

    plan = planner.plan(objective)

    result = {
        "objective": objective,
        "backend": backend or settings.backends.default_backend,
        "plan": plan,
        "state": fsm.state,
    }

    if output_file:
        import json

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(result, indent=2, default=str))
        typer.echo(f"Plan saved to {output_file}")
    else:
        echo_output(ctx, result, format=output_format)

    logger.info("run.plan.complete", objective=objective)
