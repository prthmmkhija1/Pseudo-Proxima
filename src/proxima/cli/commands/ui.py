"""
TUI CLI commands.

Launch the Terminal User Interface.
"""

import typer

app = typer.Typer(help="Launch the Terminal User Interface.")


@app.callback(invoke_without_command=True)
def ui_callback(ctx: typer.Context) -> None:
    """Launch the TUI."""
    if ctx.invoked_subcommand is None:
        launch_tui()


@app.command("launch")
def launch(
    theme: str = typer.Option("dark", "--theme", "-t", help="UI theme (dark|light)"),
    screen: str = typer.Option("dashboard", "--screen", "-s", help="Initial screen"),
) -> None:
    """Launch the Proxima TUI."""
    launch_tui(theme, screen)


def launch_tui(theme: str = "dark", screen: str = "dashboard") -> None:
    """Launch the Proxima TUI with the specified theme and screen."""
    try:
        from proxima.tui.app import ProximaApp
    except ImportError as e:
        typer.echo("TUI dependencies not installed.", err=True)
        typer.echo("Install with: pip install proxima-agent[tui]", err=True)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Validate theme option
    valid_themes = ["dark", "light"]
    if theme.lower() not in valid_themes:
        typer.echo(
            f"Invalid theme '{theme}'. Valid options: {', '.join(valid_themes)}",
            err=True,
        )
        raise typer.Exit(1)

    # Validate screen option
    valid_screens = ["dashboard", "execution", "configuration", "results", "backends"]
    if screen.lower() not in valid_screens:
        typer.echo(
            f"Invalid screen '{screen}'. Valid options: {', '.join(valid_screens)}",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Launching Proxima TUI (theme={theme}, screen={screen})...")

    try:
        app = ProximaApp(theme=theme, initial_screen=screen)
        app.run()
    except Exception as e:
        typer.echo(f"TUI error: {e}", err=True)
        raise typer.Exit(1)


@app.command("check")
def check_tui() -> None:
    """Check if TUI dependencies are available."""
    dependencies = {
        "textual": "TUI framework",
        "rich": "Rich text rendering",
    }

    all_ok = True
    for pkg, desc in dependencies.items():
        try:
            __import__(pkg)
            typer.echo(f"✓ {pkg}: {desc}")
        except ImportError:
            typer.echo(f"✗ {pkg}: {desc} (not installed)")
            all_ok = False

    if all_ok:
        typer.echo("\n✓ All TUI dependencies are available")
        typer.echo("Run 'proxima ui' to launch")
    else:
        typer.echo("\n✗ Some dependencies are missing")
        typer.echo("Install with: pip install proxima-agent[tui]")
        raise typer.Exit(1)
