"""Config command implementation."""

from __future__ import annotations

import json
from typing import Any

import typer
import yaml

from proxima.config.settings import config_service

app = typer.Typer(name="config", help="Configuration management")


def _parse_cli_value(raw: str) -> Any:
    # Try JSON first to catch numbers/bools/null/quoted strings
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered in {"none", "null"}:
            return None
        return raw


@app.command()
def show(format: str = typer.Option("yaml", help="Output format: yaml or json")) -> None:
    """Show the effective configuration after all layers are merged."""

    settings = config_service.load()
    data = settings.model_dump()

    if format.lower() == "json":
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(yaml.safe_dump(data, sort_keys=False))


@app.command()
def get(key: str = typer.Argument(..., help="Dot path (e.g., general.verbosity)")) -> None:
    """Get a configuration value by key path."""

    try:
        value = config_service.get_value(key)
    except KeyError:
        raise typer.Exit(code=1)
    typer.echo(value)


@app.command()
def set(
    key: str = typer.Argument(..., help="Dot path (e.g., general.verbosity)"),
    value: str = typer.Argument(..., help="Value to set (JSON or plain text)"),
    scope: str = typer.Option("user", case_sensitive=False, help="Scope: user or project"),
) -> None:
    """Set a configuration value in the selected scope."""

    scope_value = scope.lower()
    if scope_value not in {"user", "project"}:
        raise typer.BadParameter("Scope must be 'user' or 'project'")

    parsed_value = _parse_cli_value(value)

    target = (
        config_service.user_config_path
        if scope_value == "user"
        else config_service.project_config_path
    )
    previous = target.read_text(encoding="utf-8") if target.exists() else None

    try:
        config_service.set_value(key, parsed_value, scope=scope_value)  # type: ignore[arg-type]
        # Validate final configuration
        config_service.load()
        typer.echo(f"Set {key} in {scope_value} config ({target})")
    except Exception as exc:  # noqa: BLE001 - surface validation errors
        if previous is None and target.exists():
            target.unlink(missing_ok=True)
        elif previous is not None:
            target.write_text(previous, encoding="utf-8")
        raise typer.Exit(code=1) from exc


@app.command()
def reset(
    scope: str = typer.Option("user", case_sensitive=False, help="Scope: user or project")
) -> None:
    """Remove scoped config file to fall back to lower-priority sources."""

    scope_value = scope.lower()
    if scope_value not in {"user", "project"}:
        raise typer.BadParameter("Scope must be 'user' or 'project'")

    config_service.reset(scope=scope_value)  # type: ignore[arg-type]
    typer.echo(f"Reset {scope_value} config")
