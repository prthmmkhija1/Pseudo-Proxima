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


if __name__ == "__main__":
    app()
