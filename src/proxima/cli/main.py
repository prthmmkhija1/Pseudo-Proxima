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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import typer

from proxima.cli import utils as cli_utils
from proxima.cli.commands import (
    agent as agent_commands,
    benchmark as benchmark_commands,
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


# =============================================================================
# Command Alias System
# =============================================================================


class AliasConflictError(Exception):
    """Raised when an alias conflicts with an existing command or alias."""

    def __init__(self, alias: str, existing: str, message: Optional[str] = None):
        self.alias = alias
        self.existing = existing
        super().__init__(
            message or f"Alias '{alias}' conflicts with existing command/alias '{existing}'"
        )


@dataclass
class AliasDefinition:
    """Definition of a command alias.

    Attributes:
        alias: Short alias name.
        target: Full command this aliases to.
        description: Description of the alias.
        hidden: Whether to hide from help.
        category: Category for grouping.
        arguments: Default arguments to pass.
    """

    alias: str
    target: str
    description: str = ""
    hidden: bool = True
    category: str = "general"
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alias": self.alias,
            "target": self.target,
            "description": self.description,
            "hidden": self.hidden,
            "category": self.category,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AliasDefinition":
        """Create from dictionary."""
        return cls(
            alias=data["alias"],
            target=data["target"],
            description=data.get("description", ""),
            hidden=data.get("hidden", True),
            category=data.get("category", "general"),
            arguments=data.get("arguments", {}),
        )


class AliasRegistry:
    """Registry for command aliases.

    Manages registration, lookup, and conflict detection
    for command aliases.

    Example:
        >>> registry = AliasRegistry()
        >>> registry.register("r", "run", "Short for run command")
        >>> registry.expand("r demo") == "run demo"
        True
    """

    # Built-in aliases (cannot be overwritten without force)
    BUILTIN_ALIASES: Dict[str, str] = {
        # Run aliases
        "r": "run",
        "exec": "run",
        # Compare aliases
        "cmp": "compare",
        "diff": "compare",
        # Backends aliases
        "be": "backends list",
        "ls": "backends list",
        # Config aliases
        "cfg": "config show",
        # History aliases
        "hist": "history list",
        "h": "history",
        # Session aliases
        "sess": "session list",
        "s": "session",
        # Agent aliases
        "a": "agent run",
        # Quick circuit shortcuts
        "bell": "run 'create bell state'",
        "qft": "run 'quantum fourier transform'",
        "ghz": "run 'create ghz state'",
    }

    def __init__(self) -> None:
        """Initialize the registry."""
        self._aliases: Dict[str, AliasDefinition] = {}
        self._categories: Dict[str, List[str]] = {}
        self._reserved: set = set()
        self._load_builtins()

    def _load_builtins(self) -> None:
        """Load built-in aliases."""
        builtin_categories = {
            "run": ["r", "exec"],
            "compare": ["cmp", "diff"],
            "backends": ["be", "ls"],
            "config": ["cfg"],
            "history": ["hist", "h"],
            "session": ["sess", "s"],
            "agent": ["a"],
            "quick": ["bell", "qft", "ghz"],
        }

        for alias, target in self.BUILTIN_ALIASES.items():
            category = "general"
            for cat, aliases in builtin_categories.items():
                if alias in aliases:
                    category = cat
                    break

            self._aliases[alias] = AliasDefinition(
                alias=alias,
                target=target,
                description=f"Alias for '{target}'",
                category=category,
            )

            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(alias)

    def register(
        self,
        alias: str,
        target: str,
        description: str = "",
        hidden: bool = True,
        category: str = "custom",
        arguments: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """Register a new alias.

        Args:
            alias: Alias name.
            target: Target command.
            description: Alias description.
            hidden: Whether to hide from help.
            category: Category for grouping.
            arguments: Default arguments.
            overwrite: Whether to overwrite existing.

        Raises:
            AliasConflictError: If alias conflicts and overwrite is False.
        """
        # Check for conflicts
        if alias in self._reserved:
            raise AliasConflictError(alias, alias, f"'{alias}' is a reserved command")

        if alias in self._aliases and not overwrite:
            existing = self._aliases[alias]
            if alias in self.BUILTIN_ALIASES:
                raise AliasConflictError(
                    alias, existing.target,
                    f"Cannot overwrite built-in alias '{alias}' → '{existing.target}'"
                )
            raise AliasConflictError(alias, existing.target)

        # Register alias
        definition = AliasDefinition(
            alias=alias,
            target=target,
            description=description or f"Alias for '{target}'",
            hidden=hidden,
            category=category,
            arguments=arguments or {},
        )

        self._aliases[alias] = definition

        if category not in self._categories:
            self._categories[category] = []
        if alias not in self._categories[category]:
            self._categories[category].append(alias)

    def unregister(self, alias: str) -> bool:
        """Unregister an alias.

        Args:
            alias: Alias to remove.

        Returns:
            True if alias was removed.

        Raises:
            ValueError: If trying to remove built-in alias.
        """
        if alias in self.BUILTIN_ALIASES:
            raise ValueError(f"Cannot remove built-in alias '{alias}'")

        if alias in self._aliases:
            definition = self._aliases.pop(alias)
            if definition.category in self._categories:
                self._categories[definition.category].remove(alias)
            return True
        return False

    def get(self, alias: str) -> Optional[AliasDefinition]:
        """Get alias definition."""
        return self._aliases.get(alias)

    def expand(self, command_line: str) -> str:
        """Expand alias in a command line.

        Args:
            command_line: Full command line to expand.

        Returns:
            Command line with alias expanded.

        Example:
            >>> expand("r demo --backend cirq")
            "run demo --backend cirq"
        """
        parts = command_line.strip().split(maxsplit=1)
        if not parts:
            return command_line

        first_word = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if first_word in self._aliases:
            definition = self._aliases[first_word]
            expanded = definition.target

            # Add default arguments
            if definition.arguments:
                for key, value in definition.arguments.items():
                    if f"--{key}" not in rest:
                        expanded += f" --{key} {value}"

            if rest:
                expanded += f" {rest}"

            return expanded

        return command_line

    def is_alias(self, word: str) -> bool:
        """Check if a word is a registered alias."""
        return word in self._aliases

    def list_aliases(
        self,
        category: Optional[str] = None,
        include_hidden: bool = False,
    ) -> List[AliasDefinition]:
        """List registered aliases.

        Args:
            category: Filter by category.
            include_hidden: Include hidden aliases.

        Returns:
            List of alias definitions.
        """
        aliases = list(self._aliases.values())

        if category:
            aliases = [a for a in aliases if a.category == category]

        if not include_hidden:
            aliases = [a for a in aliases if not a.hidden]

        return sorted(aliases, key=lambda a: a.alias)

    def list_categories(self) -> List[str]:
        """Get list of alias categories."""
        return sorted(self._categories.keys())

    def get_by_category(self) -> Dict[str, List[AliasDefinition]]:
        """Get aliases grouped by category."""
        result: Dict[str, List[AliasDefinition]] = {}
        for category, alias_names in self._categories.items():
            result[category] = [
                self._aliases[name]
                for name in alias_names
                if name in self._aliases
            ]
        return result

    def add_reserved(self, *commands: str) -> None:
        """Mark commands as reserved (cannot be used as aliases)."""
        self._reserved.update(commands)

    def check_conflicts(
        self,
        alias: str,
    ) -> Optional[str]:
        """Check if alias would conflict.

        Args:
            alias: Alias to check.

        Returns:
            Conflicting command/alias name if conflict exists, None otherwise.
        """
        if alias in self._reserved:
            return alias
        if alias in self._aliases:
            return self._aliases[alias].target
        return None

    def format_help(self) -> str:
        """Format alias help text."""
        lines = ["COMMAND ALIASES:", ""]

        by_category = self.get_by_category()
        for category in sorted(by_category.keys()):
            aliases = by_category[category]
            if not aliases:
                continue

            lines.append(f"  {category.upper()}:")
            for alias_def in aliases:
                lines.append(
                    f"    {alias_def.alias:10} → {alias_def.target}"
                )
            lines.append("")

        return "\n".join(lines)

    def load_from_config(self, config_path: Path) -> int:
        """Load custom aliases from config file.

        Args:
            config_path: Path to YAML config file.

        Returns:
            Number of aliases loaded.
        """
        import yaml

        if not config_path.exists():
            return 0

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        aliases_data = data.get("aliases", {})
        count = 0

        for alias, value in aliases_data.items():
            try:
                if isinstance(value, str):
                    self.register(alias, value, category="custom")
                elif isinstance(value, dict):
                    self.register(
                        alias=alias,
                        target=value.get("target", ""),
                        description=value.get("description", ""),
                        hidden=value.get("hidden", True),
                        category=value.get("category", "custom"),
                        arguments=value.get("arguments", {}),
                        overwrite=value.get("overwrite", False),
                    )
                count += 1
            except AliasConflictError:
                continue

        return count

    def save_to_config(self, config_path: Path) -> None:
        """Save custom aliases to config file.

        Args:
            config_path: Path to YAML config file.
        """
        import yaml

        # Only save non-builtin aliases
        custom_aliases = {
            alias: defn.to_dict()
            for alias, defn in self._aliases.items()
            if alias not in self.BUILTIN_ALIASES
        }

        data = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        data["aliases"] = custom_aliases

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)


# Global alias registry
alias_registry = AliasRegistry()

# Mark existing commands as reserved
alias_registry.add_reserved(
    "run", "compare", "backends", "config", "history",
    "session", "benchmark", "agent", "ui", "shell",
    "version", "init", "interactive", "completion",
    "status", "export", "doctor", "help",
)

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
app.add_typer(benchmark_commands.app, name="benchmark")
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
# Alias Management Commands
# ==============================================================================


@app.command("aliases")
def list_aliases_cmd(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Filter by category"
    ),
    all_aliases: bool = typer.Option(
        False, "--all", "-a", help="Show all aliases including hidden"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON"
    ),
):
    """List all registered command aliases.

    \b
    Shows all command shortcuts and their target commands,
    organized by category.

    \b
    CATEGORIES:
      run      - Run command shortcuts (r, exec)
      compare  - Compare command shortcuts (cmp, diff)
      backends - Backend command shortcuts (be, ls)
      config   - Configuration shortcuts (cfg)
      history  - History shortcuts (hist, h)
      session  - Session shortcuts (sess, s)
      quick    - Quick circuit commands (bell, qft, ghz)
      custom   - User-defined aliases

    \b
    EXAMPLES:
      proxima aliases
      proxima aliases --category run
      proxima aliases --all
      proxima aliases --json
    """
    import json

    if json_output:
        aliases = alias_registry.list_aliases(category=category, include_hidden=all_aliases)
        data = {
            "count": len(aliases),
            "aliases": [a.to_dict() for a in aliases],
        }
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo("\n" + "=" * 60)
    typer.echo("COMMAND ALIASES")
    typer.echo("=" * 60)

    by_category = alias_registry.get_by_category()

    if category:
        if category not in by_category:
            typer.echo(f"\nUnknown category: {category}", err=True)
            typer.echo(f"Available: {', '.join(alias_registry.list_categories())}")
            raise typer.Exit(1)
        by_category = {category: by_category[category]}

    for cat in sorted(by_category.keys()):
        aliases = by_category[cat]
        if not aliases:
            continue

        visible = [a for a in aliases if not a.hidden or all_aliases]
        if not visible:
            continue

        typer.echo(f"\n{cat.upper()}:")
        for alias_def in visible:
            marker = " (hidden)" if alias_def.hidden else ""
            typer.echo(f"  {alias_def.alias:12} → {alias_def.target}{marker}")

    typer.echo()


@app.command("alias")
def manage_alias(
    action: str = typer.Argument(
        ..., help="Action: add, remove, show"
    ),
    name: Optional[str] = typer.Argument(
        None, help="Alias name"
    ),
    target: Optional[str] = typer.Argument(
        None, help="Target command (for add action)"
    ),
    description: str = typer.Option(
        "", "--desc", "-d", help="Alias description"
    ),
    category: str = typer.Option(
        "custom", "--category", "-c", help="Alias category"
    ),
):
    """Manage command aliases.

    \b
    ACTIONS:
      add     - Add a new alias
      remove  - Remove a custom alias
      show    - Show details of an alias

    \b
    EXAMPLES:
      proxima alias add q quick           # Add alias 'q' for 'quick'
      proxima alias add rb "run bell" -d "Run bell state"
      proxima alias remove q              # Remove alias 'q'
      proxima alias show r                # Show details of alias 'r'
    """
    action = action.lower()

    if action == "add":
        if not name or not target:
            typer.echo("Usage: proxima alias add <name> <target>", err=True)
            raise typer.Exit(1)

        try:
            alias_registry.register(
                alias=name,
                target=target,
                description=description,
                category=category,
            )
            typer.echo(f"✓ Added alias: {name} → {target}")
        except AliasConflictError as e:
            typer.echo(f"✗ Error: {e}", err=True)
            raise typer.Exit(1)

    elif action == "remove":
        if not name:
            typer.echo("Usage: proxima alias remove <name>", err=True)
            raise typer.Exit(1)

        try:
            if alias_registry.unregister(name):
                typer.echo(f"✓ Removed alias: {name}")
            else:
                typer.echo(f"Alias not found: {name}", err=True)
                raise typer.Exit(1)
        except ValueError as e:
            typer.echo(f"✗ Error: {e}", err=True)
            raise typer.Exit(1)

    elif action == "show":
        if not name:
            typer.echo("Usage: proxima alias show <name>", err=True)
            raise typer.Exit(1)

        definition = alias_registry.get(name)
        if not definition:
            typer.echo(f"Alias not found: {name}", err=True)
            raise typer.Exit(1)

        typer.echo(f"\nAlias: {definition.alias}")
        typer.echo(f"Target: {definition.target}")
        typer.echo(f"Description: {definition.description}")
        typer.echo(f"Category: {definition.category}")
        typer.echo(f"Hidden: {definition.hidden}")
        if definition.arguments:
            typer.echo(f"Default Args: {definition.arguments}")
        typer.echo()

    else:
        typer.echo(f"Unknown action: {action}", err=True)
        typer.echo("Available actions: add, remove, show")
        raise typer.Exit(1)


@app.command("expand")
def expand_alias_cmd(
    command_line: str = typer.Argument(..., help="Command line to expand"),
):
    """Expand aliases in a command line.

    \b
    Shows how aliases would be expanded without executing.

    \b
    EXAMPLES:
      proxima expand "r demo --backend cirq"
      proxima expand "cmp bell --all"
    """
    expanded = alias_registry.expand(command_line)

    if expanded == command_line:
        typer.echo(f"No expansion: {command_line}")
    else:
        typer.echo(f"Original: {command_line}")
        typer.echo(f"Expanded: {expanded}")


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


# ==============================================================================
# Rich Help Command with Examples
# ==============================================================================


@app.command("help")
def show_help(
    command: Optional[str] = typer.Argument(
        None, help="Command to show help for"
    ),
    all_examples: bool = typer.Option(
        False, "--all", "-a", help="Show all examples"
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", "-t", help="Filter examples by tags (comma-separated)"
    ),
    search: Optional[str] = typer.Option(
        None, "--search", "-s", help="Search examples"
    ),
):
    """Show detailed help with examples.

    \b
    Shows rich help content with practical examples for commands.
    More detailed than the standard --help flag.

    \b
    FEATURES:
    - Detailed command descriptions
    - Multiple practical examples per command
    - Related commands and documentation links
    - Example search functionality
    - Tag-based filtering

    \b
    EXAMPLES:
      proxima help                  # List all commands with help
      proxima help run              # Show help for 'run' command
      proxima help run --all        # Show all examples for 'run'
      proxima help --search bell    # Search examples for 'bell'
      proxima help run --tags basic # Show only basic examples
    """
    from proxima.cli.commands import help_registry

    # Search mode
    if search:
        typer.echo(f"\nSearching examples for: {search}\n")
        results = help_registry.search_examples(search)

        if not results:
            typer.echo("No matching examples found.")
            return

        typer.echo(f"Found {len(results)} matching example(s):\n")
        for cmd, example in results:
            typer.echo(f"[{cmd}]")
            typer.echo(example.format())
            typer.echo()
        return

    # List all commands
    if not command:
        typer.echo("\n" + "=" * 60)
        typer.echo("PROXIMA COMMAND HELP")
        typer.echo("=" * 60)
        typer.echo("\nAvailable commands with detailed help:\n")

        for cmd in help_registry.list_commands():
            help_content = help_registry.get(cmd)
            if help_content:
                typer.echo(f"  {cmd:15} {help_content.summary}")

        typer.echo("\n" + "-" * 60)
        typer.echo("Use 'proxima help <command>' for detailed help with examples.")
        typer.echo("Use 'proxima help --search <query>' to search examples.")
        typer.echo()
        return

    # Show help for specific command
    tag_list = tags.split(",") if tags else None

    if tag_list:
        # Filter by tags
        examples = help_registry.get_examples(command, tags=tag_list)
        if not examples:
            typer.echo(f"No examples with tags [{tags}] for '{command}'")
            return

        typer.echo(f"\nExamples for '{command}' with tags [{tags}]:\n")
        for example in examples:
            typer.echo(example.format())
            typer.echo()
        return

    # Full help
    help_text = help_registry.format_command_help(command, include_all=all_examples)
    typer.echo("\n" + help_text)


@app.command("examples")
def show_examples(
    command: Optional[str] = typer.Argument(
        None, help="Command to show examples for"
    ),
    count: int = typer.Option(
        5, "--count", "-n", help="Number of examples"
    ),
):
    """Show usage examples for commands.

    \b
    Quick way to see practical examples for any command.

    \b
    EXAMPLES:
      proxima examples           # Show random examples
      proxima examples run       # Show examples for 'run'
      proxima examples run -n 10 # Show 10 examples for 'run'
    """
    from proxima.cli.commands import help_registry

    if command:
        examples = help_registry.get_examples(command, max_examples=count)
        if not examples:
            typer.echo(f"No examples available for '{command}'")
            return

        typer.echo(f"\nExamples for '{command}':\n")
        for example in examples:
            typer.echo(example.format())
            typer.echo()
    else:
        # Show examples from multiple commands
        typer.echo("\nExample commands:\n")
        shown = 0
        for cmd in help_registry.list_commands():
            examples = help_registry.get_examples(cmd, max_examples=1)
            for example in examples:
                typer.echo(f"[{cmd}]")
                typer.echo(example.format())
                typer.echo()
                shown += 1
                if shown >= count:
                    return


if __name__ == "__main__":
    app()
