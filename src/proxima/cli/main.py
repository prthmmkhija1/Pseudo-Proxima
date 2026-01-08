"""
Main CLI application definition.
"""

import typer
from pathlib import Path
from typing import Optional

from proxima.cli.commands import config as config_commands
from proxima.cli.commands import run as run_commands
from proxima.cli.commands import backends as backends_commands
from proxima.cli.commands import compare as compare_commands
from proxima.config.settings import config_service
from proxima.cli import utils as cli_utils
from proxima.utils.logging import configure_from_settings

app = typer.Typer(
    name="proxima",
    help="Proxima: Intelligent Quantum Simulation Orchestration Framework",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Select backend (lret|cirq|qiskit|auto)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output format (text|json|rich)"
    ),
    color: Optional[bool] = typer.Option(
        None, "--color/--no-color", help="Enable or disable color output"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (stackable)"
    ),
    quiet: int = typer.Option(
        0, "--quiet", "-q", count=True, help="Decrease verbosity (stackable)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, do not execute"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip consent prompts"),
):
    """Global options and configuration bootstrap."""

    cli_overrides = {"general": {}, "backends": {}, "consent": {}}

    if backend:
        cli_overrides["backends"]["default_backend"] = backend
    if output:
        cli_overrides["general"]["output_format"] = output
    if color is not None:
        cli_overrides["general"]["color_enabled"] = color

    base_settings = config_service.load()
    base_level = base_settings.general.verbosity
    effective_level = cli_utils.compute_verbosity(base_level, verbose, quiet)
    cli_overrides["general"]["verbosity"] = effective_level

    if force:
        cli_overrides["consent"]["auto_approve_local_llm"] = True
        cli_overrides["consent"]["auto_approve_remote_llm"] = True

    settings = cli_utils.load_settings_with_cli_overrides(
        config_path=config,
        cli_overrides=cli_overrides,
    )

    configure_from_settings(settings)
    ctx.obj = {
        "settings": settings,
        "dry_run": dry_run,
    }

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def version():
    """Show version information."""
    from proxima import __version__

    typer.echo(f"Proxima version {__version__}")


@app.command()
def init():
    """Initialize Proxima configuration."""
    path = config_service.save(config_service.load(), scope="user")
    typer.echo(f"Initialized user configuration at {path}")


app.add_typer(config_commands.app, name="config")
app.add_typer(run_commands.app, name="run")
app.add_typer(backends_commands.app, name="backends")
app.add_typer(compare_commands.app, name="compare")

if __name__ == "__main__":
    app()
