"""
Main CLI application definition.

Proxima: Intelligent Quantum Simulation Orchestration Framework

This CLI provides comprehensive quantum circuit simulation capabilities with:
- Multi-backend support (LRET, Cirq, Qiskit Aer, QuEST, cuQuantum, qsim)
- Intelligent backend auto-selection
- LLM-powered insights and analysis
- Multi-backend comparison
- Resource monitoring and safety controls
- Interactive shell mode
- Command aliases and shortcuts
- Shell completion scripts
"""

from pathlib import Path

import typer

from proxima.cli import utils as cli_utils
from proxima.cli.commands import (
    agent as agent_commands,
    backends as backends_commands,
    compare as compare_commands,
    config as config_commands,
    history as history_commands,
    run as run_commands,
    session as session_commands,
    ui as ui_commands,
)
from proxima.cli.interactive import (
    interactive_app,
    command_aliases,
    CompletionGenerator,
    InteractiveShell,
    DetailedHelp,
)
from proxima.config.settings import config_service
from proxima.utils.logging import configure_from_settings

app = typer.Typer(
    name="proxima",
    help="""Proxima: Intelligent Quantum Simulation Orchestration Framework

    \b
    COMMANDS:
      run       - Execute quantum circuit simulations
      compare   - Compare results across multiple backends
      backends  - List and manage simulation backends
      config    - View and modify configuration
      history   - View past execution results
      session   - Manage execution sessions
      agent     - Run agent.md automation files
      ui        - Launch interactive terminal UI
      shell     - Interactive shell and completion
    
    \b
    COMMAND ALIASES:
      r, exec   → run          be, ls    → backends
      cmp, diff → compare      cfg       → config
      hist, h   → history      sess, s   → session
      a         → agent        bell, qft, ghz → quick circuits
    
    \b
    EXAMPLES:
      proxima run "create bell state"
      proxima run demo --backend cirq --shots 1000
      proxima compare "quantum teleportation" --all
      proxima backends list
      proxima config show
      proxima shell interactive
    
    \b
    For more information on a command, use: proxima <command> --help
    For detailed help with examples, use: proxima shell help <command>
    """,
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="markdown",
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to config YAML"
    ),
    backend: str | None = typer.Option(
        None, "--backend", "-b", help="Select backend (lret|cirq|qiskit|auto)"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output format (text|json|rich)"
    ),
    color: bool | None = typer.Option(
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
    from typing import Any

    cli_overrides: dict[str, Any] = {"general": {}, "backends": {}, "consent": {}}

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


@app.command("interactive")
def cmd_interactive_shell():
    """Launch interactive Proxima shell.
    
    \b
    Provides a REPL-style interface with:
    - Command history with arrow keys
    - Tab completion for commands and aliases
    - Alias expansion (e.g., 'r' → 'run')
    - Rich formatted output
    - Session persistence
    
    \b
    EXAMPLES:
      proxima interactive
      
    Inside the shell:
      > run "bell state"
      > r demo --backend cirq
      > help run
      > aliases
      > exit
    """
    shell = InteractiveShell()
    shell.run()


@app.command("completion")
def cmd_completion(
    shell: str = typer.Argument(
        "bash",
        help="Shell type: bash, zsh, powershell, fish",
    ),
):
    """Generate shell completion script.
    
    \b
    Generates completion scripts for various shells including
    command and alias completion.
    
    \b
    INSTALLATION:
      Bash:       source <(proxima completion bash)
      Zsh:        eval "$(proxima completion zsh)"
      PowerShell: . (proxima completion powershell | Out-String)
      Fish:       proxima completion fish > ~/.config/fish/completions/proxima.fish
    
    \b
    EXAMPLES:
      proxima completion bash >> ~/.bashrc
      proxima completion zsh >> ~/.zshrc
      proxima completion powershell > proxima_completion.ps1
    """
    typer.echo(CompletionGenerator.generate(shell))


# Register command groups
app.add_typer(config_commands.app, name="config")
app.add_typer(run_commands.app, name="run")
app.add_typer(backends_commands.app, name="backends")
app.add_typer(compare_commands.app, name="compare")
app.add_typer(history_commands.app, name="history")
app.add_typer(session_commands.app, name="session")
app.add_typer(agent_commands.app, name="agent")
app.add_typer(ui_commands.app, name="ui")
app.add_typer(interactive_app, name="shell")


# =============================================================================
# Command Aliases (registered as top-level commands)
# =============================================================================

# Run aliases
@app.command("r", hidden=True)
def alias_r(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="Task to execute"),
):
    """Alias for 'run'."""
    ctx.invoke(run_commands.app.registered_commands[0].callback, task=task)


@app.command("exec", hidden=True)
def alias_exec(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="Task to execute"),
):
    """Alias for 'run'."""
    ctx.invoke(run_commands.app.registered_commands[0].callback, task=task)


# Compare aliases
@app.command("cmp", hidden=True)
def alias_cmp(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="Task to compare"),
):
    """Alias for 'compare'."""
    from proxima.cli.commands.compare import compare
    ctx.invoke(compare, task=task)


# Quick circuit commands
@app.command("bell")
def quick_bell(
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend to use"),
    shots: int = typer.Option(1024, "--shots", "-s", help="Number of shots"),
):
    """Quick: Create and run a Bell state circuit.
    
    \b
    EXAMPLES:
      proxima bell
      proxima bell --backend cirq
      proxima bell --shots 2000
    """
    from proxima.cli.commands.run import run_task
    run_task("create bell state", backend=backend, shots=shots)


@app.command("qft")
def quick_qft(
    qubits: int = typer.Option(4, "--qubits", "-q", help="Number of qubits"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend to use"),
    shots: int = typer.Option(1024, "--shots", "-s", help="Number of shots"),
):
    """Quick: Run Quantum Fourier Transform circuit.
    
    \b
    EXAMPLES:
      proxima qft
      proxima qft --qubits 6
      proxima qft --backend qiskit --qubits 8
    """
    from proxima.cli.commands.run import run_task
    run_task(f"quantum fourier transform on {qubits} qubits", backend=backend, shots=shots)


@app.command("ghz")
def quick_ghz(
    qubits: int = typer.Option(3, "--qubits", "-q", help="Number of qubits"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend to use"),
    shots: int = typer.Option(1024, "--shots", "-s", help="Number of shots"),
):
    """Quick: Create and run a GHZ state circuit.
    
    \b
    EXAMPLES:
      proxima ghz
      proxima ghz --qubits 5
      proxima ghz --backend cirq --qubits 4
    """
    from proxima.cli.commands.run import run_task
    run_task(f"create ghz state with {qubits} qubits", backend=backend, shots=shots)



# ==============================================================================
# Additional Commands and Aliases for 100% Coverage
# ==============================================================================


@app.command("status")
def show_status():
    """Show current Proxima status and active sessions.
    
    \b
    Displays:
    - Active session (if any)
    - Current backend configuration
    - Recent execution history
    - System resource status
    
    \b
    EXAMPLES:
      proxima status
    """
    from proxima.config.settings import config_service
    from proxima.data.store import get_store
    
    typer.echo("\n" + "=" * 60)
    typer.echo("PROXIMA STATUS")
    typer.echo("=" * 60)
    
    # Configuration
    settings = config_service.load()
    typer.echo(f"\nDefault Backend: {settings.backends.default_backend}")
    typer.echo(f"Output Format:   {settings.general.output_format}")
    typer.echo(f"Color Enabled:   {settings.general.color_enabled}")
    
    # Recent sessions
    store = get_store()
    sessions = store.list_sessions(limit=3)
    if sessions:
        typer.echo("\nRecent Sessions:")
        for session in sessions:
            name = session.name or "(unnamed)"
            typer.echo(f"  - {session.id[:8]}... {name} ({session.result_count} results)")
    else:
        typer.echo("\nNo sessions found.")
    
    # Recent results
    results = store.list_results(limit=3)
    if results:
        typer.echo("\nRecent Executions:")
        for result in results:
            typer.echo(f"  - {result.backend_name}: {result.qubit_count} qubits, {result.shot_count} shots")
    
    typer.echo()


@app.command("export")
def export_results(
    session_id: str | None = typer.Option(None, "--session", "-s", help="Session ID to export"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json|csv|yaml)"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export execution results to file.
    
    \b
    Exports results from a session or all results to the specified format.
    
    \b
    FORMATS:
      json  - JSON format (default)
      csv   - Comma-separated values
      yaml  - YAML format
    
    \b
    EXAMPLES:
      proxima export --session abc123 --format json
      proxima export --format csv --output results.csv
    """
    from proxima.data.store import get_store
    import json
    
    store = get_store()
    
    if session_id:
        results = store.list_results(session_id=session_id)
    else:
        results = store.list_results(limit=100)
    
    if not results:
        typer.echo("No results to export.", err=True)
        raise typer.Exit(1)
    
    # Convert to exportable format
    export_data = []
    for result in results:
        export_data.append({
            "id": result.id,
            "session_id": result.session_id,
            "backend": result.backend_name,
            "qubits": result.qubit_count,
            "shots": result.shot_count,
            "execution_time_ms": result.execution_time_ms,
            "created_at": str(result.created_at),
        })
    
    if format.lower() == "json":
        output_str = json.dumps(export_data, indent=2)
    elif format.lower() == "csv":
        if export_data:
            headers = list(export_data[0].keys())
            lines = [",".join(headers)]
            for item in export_data:
                lines.append(",".join(str(item.get(h, "")) for h in headers))
            output_str = "\n".join(lines)
        else:
            output_str = ""
    elif format.lower() == "yaml":
        import yaml
        output_str = yaml.dump(export_data, default_flow_style=False)
    else:
        typer.echo(f"Unknown format: {format}", err=True)
        raise typer.Exit(1)
    
    if output:
        with open(output, 'w') as f:
            f.write(output_str)
        typer.echo(f"Exported {len(export_data)} results to {output}")
    else:
        typer.echo(output_str)


@app.command("doctor")
def run_diagnostics():
    """Run Proxima diagnostics and health checks.
    
    \b
    Checks:
    - Backend availability
    - Configuration validity
    - Data store connectivity
    - Python environment
    
    \b
    EXAMPLES:
      proxima doctor
    """
    typer.echo("\n" + "=" * 60)
    typer.echo("PROXIMA DIAGNOSTICS")
    typer.echo("=" * 60 + "\n")
    
    issues = []
    
    # Check Python version
    import sys
    typer.echo(f"✓ Python version: {sys.version.split()[0]}")
    
    # Check configuration
    try:
        from proxima.config.settings import config_service
        settings = config_service.load()
        typer.echo(f"✓ Configuration loaded: {settings.backends.default_backend}")
    except Exception as exc:
        issues.append(f"Configuration error: {exc}")
        typer.echo(f"✗ Configuration: {exc}")
    
    # Check data store
    try:
        from proxima.data.store import get_store
        store = get_store()
        session_count = len(store.list_sessions(limit=10))
        typer.echo(f"✓ Data store: {session_count} sessions found")
    except Exception as exc:
        issues.append(f"Data store error: {exc}")
        typer.echo(f"✗ Data store: {exc}")
    
    # Check backends
    typer.echo("\nBackend Status:")
    backends_to_check = ["lret", "cirq", "qiskit", "quest", "qsim", "cuquantum"]
    
    for backend in backends_to_check:
        try:
            # Try importing backend adapter
            adapter_module = f"proxima.backends.{backend}_adapter" if backend != "lret" else "proxima.backends.lret"
            __import__(adapter_module)
            typer.echo(f"  ✓ {backend}: module available")
        except ImportError:
            typer.echo(f"  ○ {backend}: module not installed")
        except Exception as exc:
            typer.echo(f"  ✗ {backend}: {exc}")
    
    # Summary
    typer.echo("\n" + "-" * 60)
    if issues:
        typer.echo(f"Found {len(issues)} issue(s):")
        for issue in issues:
            typer.echo(f"  - {issue}")
    else:
        typer.echo("All checks passed! Proxima is ready to use.")
    typer.echo()


# Additional Aliases
@app.command("be", hidden=True)
def alias_be():
    """Alias for 'backends list'."""
    from proxima.cli.commands.backends import list_backends
    list_backends()


@app.command("cfg", hidden=True)
def alias_cfg():
    """Alias for 'config show'."""
    from proxima.cli.commands.config import show
    show()


@app.command("hist", hidden=True)
def alias_hist(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of results"),
):
    """Alias for 'history list'."""
    from proxima.cli.commands.history import list_history
    list_history(limit=limit)


@app.command("sess", hidden=True)
def alias_sess(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions"),
):
    """Alias for 'session list'."""
    from proxima.cli.commands.session import list_sessions
    list_sessions(limit=limit)


@app.command("diff", hidden=True)
def alias_diff(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="Task to compare"),
):
    """Alias for 'compare'."""
    from proxima.cli.commands.compare import compare
    ctx.invoke(compare, task=task)



if __name__ == "__main__":
    app()
