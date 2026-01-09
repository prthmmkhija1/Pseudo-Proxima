"""Backends command implementation."""

from __future__ import annotations

import typer

from proxima.backends.registry import BackendStatus, backend_registry

app = typer.Typer(name="backends", help="Backend management")


def _format_capabilities(status: BackendStatus) -> str:
    if not status.capabilities:
        return "n/a"
    caps = status.capabilities
    sims = ", ".join(sim.value for sim in caps.simulator_types)
    flags = []
    if caps.supports_noise:
        flags.append("noise")
    if caps.supports_gpu:
        flags.append("gpu")
    if caps.supports_batching:
        flags.append("batching")
    flag_str = f"; features: {', '.join(flags)}" if flags else ""
    return f"simulators: {sims}; max_qubits: {caps.max_qubits}{flag_str}"


@app.command()
def list() -> None:  # noqa: A001 - intentional command name
    """List backend availability and status."""

    statuses = backend_registry.list_statuses()
    if not statuses:
        typer.echo("No backends registered")
        return

    for status in statuses:
        if status.available:
            version = status.version or "unknown"
            typer.echo(f"- {status.name}: available (version {version})")
        else:
            reason = status.reason or "unavailable"
            typer.echo(f"- {status.name}: unavailable ({reason})")


@app.command()
def info(name: str = typer.Argument(..., help="Backend name")) -> None:
    """Show backend details and capabilities."""

    try:
        status = backend_registry.get_status(name)
    except KeyError:
        typer.echo(f"Backend '{name}' not registered")
        raise typer.Exit(code=1)

    availability = "yes" if status.available else "no"
    version = status.version or "unknown"
    cap_str = _format_capabilities(status)
    reason = status.reason or ""

    typer.echo(f"Name: {status.name}")
    typer.echo(f"Available: {availability}")
    typer.echo(f"Version: {version}")
    typer.echo(f"Capabilities: {cap_str}")
    if not status.available and reason:
        typer.echo(f"Reason: {reason}")


@app.command()
def test(name: str = typer.Argument(..., help="Backend name")) -> None:
    """Check backend availability."""

    try:
        status = backend_registry.get_status(name)
    except KeyError:
        typer.echo(f"Backend '{name}' not registered")
        raise typer.Exit(code=1)

    if status.available:
        typer.echo(f"Backend '{name}' is available")
    else:
        reason = status.reason or "unavailable"
        typer.echo(f"Backend '{name}' is unavailable: {reason}")
        raise typer.Exit(code=1)
