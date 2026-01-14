"""
Interactive CLI mode and enhanced features.

Provides:
- InteractiveShell: REPL-style interactive shell
- CommandAliases: Short command aliases and shortcuts
- CompletionGenerator: Shell completion script generation
- DetailedHelp: Rich help with examples

Step 11: Feature enhancements for CLI interface.
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False
    try:
        import pyreadline3 as readline  # type: ignore
        HAS_READLINE = True
    except ImportError:
        readline = None  # type: ignore

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()


# =============================================================================
# Command Aliases & Shortcuts (Feature - CLI)
# =============================================================================


@dataclass
class CommandAlias:
    """A command alias definition."""
    
    alias: str
    command: str
    description: str
    examples: list[str] = field(default_factory=list)


class CommandAliases:
    """Manages command aliases and shortcuts.
    
    Features:
    - Short aliases for common commands
    - Custom user-defined aliases
    - Alias expansion
    - Alias listing and help
    """
    
    # Built-in aliases
    BUILTIN_ALIASES: dict[str, CommandAlias] = {
        # Run shortcuts
        "r": CommandAlias(
            alias="r",
            command="run",
            description="Execute a quantum circuit (alias for 'run')",
            examples=["proxima r 'bell state'", "proxima r demo --backend cirq"],
        ),
        "exec": CommandAlias(
            alias="exec",
            command="run",
            description="Execute a quantum circuit (alias for 'run')",
            examples=["proxima exec 'teleportation'"],
        ),
        # Compare shortcuts
        "cmp": CommandAlias(
            alias="cmp",
            command="compare",
            description="Compare across backends (alias for 'compare')",
            examples=["proxima cmp 'bell state' --all"],
        ),
        "diff": CommandAlias(
            alias="diff",
            command="compare",
            description="Compare across backends (alias for 'compare')",
            examples=["proxima diff 'qft'"],
        ),
        # Backend shortcuts
        "be": CommandAlias(
            alias="be",
            command="backends",
            description="Backend management (alias for 'backends')",
            examples=["proxima be list", "proxima be status cirq"],
        ),
        "ls": CommandAlias(
            alias="ls",
            command="backends list",
            description="List available backends",
            examples=["proxima ls"],
        ),
        # Config shortcuts
        "cfg": CommandAlias(
            alias="cfg",
            command="config",
            description="Configuration management (alias for 'config')",
            examples=["proxima cfg show", "proxima cfg set backend cirq"],
        ),
        # History shortcuts
        "hist": CommandAlias(
            alias="hist",
            command="history",
            description="View execution history (alias for 'history')",
            examples=["proxima hist", "proxima hist last 10"],
        ),
        "h": CommandAlias(
            alias="h",
            command="history",
            description="View execution history (alias for 'history')",
            examples=["proxima h"],
        ),
        # Session shortcuts
        "sess": CommandAlias(
            alias="sess",
            command="session",
            description="Session management (alias for 'session')",
            examples=["proxima sess list", "proxima sess new my-session"],
        ),
        "s": CommandAlias(
            alias="s",
            command="session",
            description="Session management (alias for 'session')",
            examples=["proxima s list"],
        ),
        # Agent shortcuts
        "a": CommandAlias(
            alias="a",
            command="agent",
            description="Run agent.md files (alias for 'agent')",
            examples=["proxima a run proxima_agent.md"],
        ),
        # Quick actions
        "bell": CommandAlias(
            alias="bell",
            command="run 'create bell state'",
            description="Create and run a Bell state circuit",
            examples=["proxima bell", "proxima bell --backend qiskit"],
        ),
        "qft": CommandAlias(
            alias="qft",
            command="run 'quantum fourier transform'",
            description="Run QFT circuit",
            examples=["proxima qft", "proxima qft --qubits 4"],
        ),
        "ghz": CommandAlias(
            alias="ghz",
            command="run 'create ghz state'",
            description="Create and run a GHZ state circuit",
            examples=["proxima ghz", "proxima ghz --qubits 5"],
        ),
    }
    
    def __init__(self) -> None:
        """Initialize command aliases."""
        self._aliases: dict[str, CommandAlias] = dict(self.BUILTIN_ALIASES)
        self._custom_aliases: dict[str, CommandAlias] = {}
    
    def add_alias(
        self,
        alias: str,
        command: str,
        description: str = "",
        examples: list[str] | None = None,
    ) -> None:
        """Add a custom alias.
        
        Args:
            alias: The alias string
            command: The command to expand to
            description: Optional description
            examples: Optional usage examples
        """
        self._custom_aliases[alias] = CommandAlias(
            alias=alias,
            command=command,
            description=description or f"Custom alias for '{command}'",
            examples=examples or [],
        )
        self._aliases[alias] = self._custom_aliases[alias]
    
    def remove_alias(self, alias: str) -> bool:
        """Remove a custom alias.
        
        Args:
            alias: Alias to remove
            
        Returns:
            True if removed
        """
        if alias in self._custom_aliases:
            del self._custom_aliases[alias]
            del self._aliases[alias]
            return True
        return False
    
    def expand(self, alias: str) -> str | None:
        """Expand an alias to its full command.
        
        Args:
            alias: The alias to expand
            
        Returns:
            Expanded command or None if not found
        """
        if alias in self._aliases:
            return self._aliases[alias].command
        return None
    
    def expand_argv(self, argv: list[str]) -> list[str]:
        """Expand aliases in argument vector.
        
        Args:
            argv: Command line arguments
            
        Returns:
            Arguments with aliases expanded
        """
        if not argv:
            return argv
        
        # Check if first arg (command) is an alias
        first_arg = argv[0]
        if first_arg in self._aliases:
            expanded = self._aliases[first_arg].command.split()
            return expanded + argv[1:]
        
        return argv
    
    def get_alias(self, alias: str) -> CommandAlias | None:
        """Get alias definition."""
        return self._aliases.get(alias)
    
    def list_aliases(self) -> list[CommandAlias]:
        """List all available aliases."""
        return list(self._aliases.values())
    
    def print_aliases(self) -> None:
        """Print all aliases in a formatted table."""
        table = Table(title="Command Aliases", show_header=True)
        table.add_column("Alias", style="cyan", no_wrap=True)
        table.add_column("Command", style="green")
        table.add_column("Description")
        
        for alias in sorted(self._aliases.values(), key=lambda x: x.alias):
            table.add_row(alias.alias, alias.command, alias.description)
        
        console.print(table)


# Global aliases instance
command_aliases = CommandAliases()


# =============================================================================
# Shell Completion Scripts (Feature - CLI)
# =============================================================================


class CompletionGenerator:
    """Generates shell completion scripts.
    
    Features:
    - Bash completion script
    - Zsh completion script
    - PowerShell completion script
    - Fish completion script
    - Installation instructions
    """
    
    BASH_COMPLETION = '''
# Proxima CLI Bash completion
# Add to ~/.bashrc: source <(proxima completion bash)

_proxima_completion() {
    local cur prev words cword
    _init_completion || return

    local commands="run compare backends config history session agent ui version init"
    local aliases="r exec cmp diff be ls cfg hist h sess s a bell qft ghz"
    local global_opts="--config --backend --output --color --verbose --quiet --dry-run --force --help"
    
    case "${prev}" in
        proxima)
            COMPREPLY=( $(compgen -W "${commands} ${aliases}" -- "${cur}") )
            return 0
            ;;
        --backend|-b)
            COMPREPLY=( $(compgen -W "lret cirq qiskit quest cuquantum qsim auto" -- "${cur}") )
            return 0
            ;;
        --output|-o)
            COMPREPLY=( $(compgen -W "text json rich" -- "${cur}") )
            return 0
            ;;
        --config|-c)
            COMPREPLY=( $(compgen -f -X '!*.yaml' -- "${cur}") $(compgen -f -X '!*.yml' -- "${cur}") )
            return 0
            ;;
        run|r|exec)
            COMPREPLY=( $(compgen -W "--backend --shots --qubits --depth --seed" -- "${cur}") )
            return 0
            ;;
        compare|cmp|diff)
            COMPREPLY=( $(compgen -W "--all --backends --shots --format" -- "${cur}") )
            return 0
            ;;
        backends|be)
            COMPREPLY=( $(compgen -W "list status info select benchmark" -- "${cur}") )
            return 0
            ;;
        config|cfg)
            COMPREPLY=( $(compgen -W "show set get reset edit" -- "${cur}") )
            return 0
            ;;
        history|hist|h)
            COMPREPLY=( $(compgen -W "list show export clear last" -- "${cur}") )
            return 0
            ;;
        session|sess|s)
            COMPREPLY=( $(compgen -W "list new switch delete export" -- "${cur}") )
            return 0
            ;;
        agent|a)
            COMPREPLY=( $(compgen -W "run validate list" -- "${cur}") )
            return 0
            ;;
    esac
    
    if [[ "${cur}" == -* ]]; then
        COMPREPLY=( $(compgen -W "${global_opts}" -- "${cur}") )
        return 0
    fi
}

complete -F _proxima_completion proxima
'''
    
    ZSH_COMPLETION = '''
#compdef proxima
# Proxima CLI Zsh completion
# Add to ~/.zshrc: eval "$(proxima completion zsh)"

_proxima() {
    local -a commands aliases global_opts
    
    commands=(
        'run:Execute quantum circuit simulations'
        'compare:Compare results across multiple backends'
        'backends:List and manage simulation backends'
        'config:View and modify configuration'
        'history:View past execution results'
        'session:Manage execution sessions'
        'agent:Run agent.md automation files'
        'ui:Launch interactive terminal UI'
        'version:Show version information'
        'init:Initialize Proxima configuration'
    )
    
    aliases=(
        'r:Run (alias)'
        'exec:Execute (alias)'
        'cmp:Compare (alias)'
        'diff:Compare (alias)'
        'be:Backends (alias)'
        'ls:List backends (alias)'
        'cfg:Config (alias)'
        'hist:History (alias)'
        'h:History (alias)'
        'sess:Session (alias)'
        's:Session (alias)'
        'a:Agent (alias)'
        'bell:Run Bell state'
        'qft:Run QFT'
        'ghz:Run GHZ state'
    )
    
    global_opts=(
        '--config[Path to config YAML]:file:_files -g "*.y(a|)ml"'
        '--backend[Select backend]:backend:(lret cirq qiskit quest cuquantum qsim auto)'
        '--output[Output format]:format:(text json rich)'
        '--color[Enable color output]'
        '--no-color[Disable color output]'
        '--verbose[Increase verbosity]'
        '--quiet[Decrease verbosity]'
        '--dry-run[Plan only, do not execute]'
        '--force[Skip consent prompts]'
        '--help[Show help]'
    )
    
    _arguments -C \\
        $global_opts \\
        '1:command:->command' \\
        '*::arg:->args'
    
    case "$state" in
        command)
            _describe -t commands 'proxima commands' commands
            _describe -t aliases 'proxima aliases' aliases
            ;;
        args)
            case "$words[1]" in
                run|r|exec)
                    _arguments \\
                        '--backend[Backend to use]:backend:(lret cirq qiskit quest cuquantum qsim auto)' \\
                        '--shots[Number of shots]:shots:' \\
                        '--qubits[Number of qubits]:qubits:' \\
                        '--depth[Circuit depth]:depth:' \\
                        '--seed[Random seed]:seed:'
                    ;;
                compare|cmp|diff)
                    _arguments \\
                        '--all[Compare all backends]' \\
                        '--backends[Specific backends]:backends:' \\
                        '--shots[Number of shots]:shots:' \\
                        '--format[Output format]:format:(table json csv)'
                    ;;
                backends|be)
                    _values 'subcommand' list status info select benchmark
                    ;;
                config|cfg)
                    _values 'subcommand' show set get reset edit
                    ;;
            esac
            ;;
    esac
}

_proxima
'''
    
    POWERSHELL_COMPLETION = '''
# Proxima CLI PowerShell completion
# Add to $PROFILE: . (proxima completion powershell | Out-String)

Register-ArgumentCompleter -Native -CommandName proxima -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    
    $commands = @(
        'run', 'compare', 'backends', 'config', 'history', 
        'session', 'agent', 'ui', 'version', 'init'
    )
    $aliases = @(
        'r', 'exec', 'cmp', 'diff', 'be', 'ls', 'cfg', 
        'hist', 'h', 'sess', 's', 'a', 'bell', 'qft', 'ghz'
    )
    $backends = @('lret', 'cirq', 'qiskit', 'quest', 'cuquantum', 'qsim', 'auto')
    $outputs = @('text', 'json', 'rich')
    
    $elements = $commandAst.CommandElements
    $lastWord = if ($elements.Count -gt 1) { $elements[-2].Value } else { '' }
    
    switch ($lastWord) {
        'proxima' {
            ($commands + $aliases) | Where-Object { $_ -like "$wordToComplete*" } | 
                ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
        }
        { $_ -in @('--backend', '-b') } {
            $backends | Where-Object { $_ -like "$wordToComplete*" } | 
                ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
        }
        { $_ -in @('--output', '-o') } {
            $outputs | Where-Object { $_ -like "$wordToComplete*" } | 
                ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
        }
        { $_ -in @('run', 'r', 'exec') } {
            @('--backend', '--shots', '--qubits', '--depth', '--seed') | 
                Where-Object { $_ -like "$wordToComplete*" } | 
                ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
        }
        default {
            if ($wordToComplete -like '-*') {
                @('--config', '--backend', '--output', '--color', '--verbose', '--quiet', '--dry-run', '--force', '--help') | 
                    Where-Object { $_ -like "$wordToComplete*" } | 
                    ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
            } else {
                ($commands + $aliases) | Where-Object { $_ -like "$wordToComplete*" } | 
                    ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
            }
        }
    }
}
'''
    
    FISH_COMPLETION = '''
# Proxima CLI Fish completion
# Save to ~/.config/fish/completions/proxima.fish

# Disable file completion
complete -c proxima -f

# Commands
complete -c proxima -n "__fish_use_subcommand" -a "run" -d "Execute quantum circuit simulations"
complete -c proxima -n "__fish_use_subcommand" -a "compare" -d "Compare results across backends"
complete -c proxima -n "__fish_use_subcommand" -a "backends" -d "List and manage backends"
complete -c proxima -n "__fish_use_subcommand" -a "config" -d "View and modify configuration"
complete -c proxima -n "__fish_use_subcommand" -a "history" -d "View past execution results"
complete -c proxima -n "__fish_use_subcommand" -a "session" -d "Manage execution sessions"
complete -c proxima -n "__fish_use_subcommand" -a "agent" -d "Run agent.md files"
complete -c proxima -n "__fish_use_subcommand" -a "ui" -d "Launch terminal UI"
complete -c proxima -n "__fish_use_subcommand" -a "version" -d "Show version"
complete -c proxima -n "__fish_use_subcommand" -a "init" -d "Initialize configuration"

# Aliases
complete -c proxima -n "__fish_use_subcommand" -a "r" -d "Run (alias)"
complete -c proxima -n "__fish_use_subcommand" -a "cmp" -d "Compare (alias)"
complete -c proxima -n "__fish_use_subcommand" -a "be" -d "Backends (alias)"
complete -c proxima -n "__fish_use_subcommand" -a "cfg" -d "Config (alias)"
complete -c proxima -n "__fish_use_subcommand" -a "hist" -d "History (alias)"
complete -c proxima -n "__fish_use_subcommand" -a "sess" -d "Session (alias)"
complete -c proxima -n "__fish_use_subcommand" -a "bell" -d "Run Bell state"
complete -c proxima -n "__fish_use_subcommand" -a "qft" -d "Run QFT"
complete -c proxima -n "__fish_use_subcommand" -a "ghz" -d "Run GHZ state"

# Global options
complete -c proxima -s c -l config -d "Path to config YAML" -r
complete -c proxima -s b -l backend -d "Select backend" -xa "lret cirq qiskit quest cuquantum qsim auto"
complete -c proxima -s o -l output -d "Output format" -xa "text json rich"
complete -c proxima -l color -d "Enable color output"
complete -c proxima -l no-color -d "Disable color output"
complete -c proxima -s v -l verbose -d "Increase verbosity"
complete -c proxima -s q -l quiet -d "Decrease verbosity"
complete -c proxima -l dry-run -d "Plan only"
complete -c proxima -s f -l force -d "Skip consent prompts"
complete -c proxima -s h -l help -d "Show help"

# Run subcommand options
complete -c proxima -n "__fish_seen_subcommand_from run r exec" -l shots -d "Number of shots" -r
complete -c proxima -n "__fish_seen_subcommand_from run r exec" -l qubits -d "Number of qubits" -r
complete -c proxima -n "__fish_seen_subcommand_from run r exec" -l depth -d "Circuit depth" -r
complete -c proxima -n "__fish_seen_subcommand_from run r exec" -l seed -d "Random seed" -r
'''
    
    @classmethod
    def generate(cls, shell: str) -> str:
        """Generate completion script for a shell.
        
        Args:
            shell: Shell type (bash, zsh, powershell, fish)
            
        Returns:
            Completion script content
        """
        scripts = {
            "bash": cls.BASH_COMPLETION,
            "zsh": cls.ZSH_COMPLETION,
            "powershell": cls.POWERSHELL_COMPLETION,
            "pwsh": cls.POWERSHELL_COMPLETION,
            "fish": cls.FISH_COMPLETION,
        }
        
        return scripts.get(shell.lower(), cls.BASH_COMPLETION)
    
    @classmethod
    def install_instructions(cls, shell: str) -> str:
        """Get installation instructions for a shell.
        
        Args:
            shell: Shell type
            
        Returns:
            Installation instructions
        """
        instructions = {
            "bash": """
# Add to ~/.bashrc:
source <(proxima completion bash)

# Or save to a file:
proxima completion bash > /etc/bash_completion.d/proxima
""",
            "zsh": """
# Add to ~/.zshrc:
eval "$(proxima completion zsh)"

# Or save to completions directory:
proxima completion zsh > ~/.zsh/completions/_proxima
""",
            "powershell": """
# Add to your PowerShell profile ($PROFILE):
. (proxima completion powershell | Out-String)

# Or save to a file and source it:
proxima completion powershell > ~/proxima_completion.ps1
. ~/proxima_completion.ps1
""",
            "fish": """
# Save to completions directory:
proxima completion fish > ~/.config/fish/completions/proxima.fish
""",
        }
        
        return instructions.get(shell.lower(), instructions["bash"])


# =============================================================================
# Detailed Help with Examples (Feature - CLI)
# =============================================================================


class DetailedHelp:
    """Provides detailed help with examples for commands.
    
    Features:
    - Rich formatted help output
    - Command examples with explanations
    - Related commands suggestions
    - Common patterns and recipes
    """
    
    COMMAND_HELP: dict[str, dict[str, Any]] = {
        "run": {
            "description": "Execute quantum circuit simulations with various backends",
            "usage": "proxima run [TASK] [OPTIONS]",
            "examples": [
                {
                    "command": 'proxima run "create bell state"',
                    "description": "Create and run a Bell state circuit",
                },
                {
                    "command": 'proxima run demo --backend cirq --shots 1000',
                    "description": "Run demo circuit with Cirq backend, 1000 shots",
                },
                {
                    "command": 'proxima run "qft on 4 qubits" --backend qiskit',
                    "description": "Run 4-qubit QFT on Qiskit Aer",
                },
                {
                    "command": 'proxima run teleportation.qasm --output json',
                    "description": "Run circuit from QASM file, output as JSON",
                },
            ],
            "options": [
                ("--backend, -b", "Backend to use (lret|cirq|qiskit|auto)", "auto"),
                ("--shots", "Number of measurement shots", "1024"),
                ("--qubits", "Number of qubits", "auto-detect"),
                ("--depth", "Maximum circuit depth", "unlimited"),
                ("--seed", "Random seed for reproducibility", "random"),
                ("--output, -o", "Output format (text|json|rich)", "rich"),
            ],
            "related": ["compare", "backends", "history"],
            "tips": [
                "Use --dry-run to see the execution plan without running",
                "Use --force to skip consent prompts for LLM usage",
                "Task can be natural language, circuit name, or file path",
            ],
        },
        "compare": {
            "description": "Compare simulation results across multiple backends",
            "usage": "proxima compare [TASK] [OPTIONS]",
            "examples": [
                {
                    "command": 'proxima compare "bell state" --all',
                    "description": "Compare Bell state across all available backends",
                },
                {
                    "command": 'proxima compare demo --backends cirq,qiskit',
                    "description": "Compare demo circuit on Cirq and Qiskit only",
                },
                {
                    "command": 'proxima compare circuit.qasm --format csv --output results.csv',
                    "description": "Compare and export results to CSV",
                },
            ],
            "options": [
                ("--all", "Compare on all available backends", "false"),
                ("--backends", "Specific backends to compare", "auto-select"),
                ("--shots", "Number of shots per backend", "1024"),
                ("--format", "Output format (table|json|csv)", "table"),
                ("--output", "Output file path", "stdout"),
            ],
            "related": ["run", "backends", "history"],
            "tips": [
                "Results include fidelity metrics and timing comparisons",
                "Use --format csv for spreadsheet analysis",
            ],
        },
        "backends": {
            "description": "List and manage simulation backends",
            "usage": "proxima backends [SUBCOMMAND] [OPTIONS]",
            "examples": [
                {
                    "command": "proxima backends list",
                    "description": "List all available backends",
                },
                {
                    "command": "proxima backends status cirq",
                    "description": "Show detailed status for Cirq backend",
                },
                {
                    "command": "proxima backends benchmark --qubits 10",
                    "description": "Benchmark all backends with 10 qubits",
                },
                {
                    "command": "proxima backends select",
                    "description": "Interactive backend selection wizard",
                },
            ],
            "subcommands": [
                ("list", "List all available backends"),
                ("status", "Show status for a specific backend"),
                ("info", "Show detailed information about a backend"),
                ("select", "Interactive backend selection"),
                ("benchmark", "Run performance benchmarks"),
            ],
            "related": ["run", "compare", "config"],
        },
        "config": {
            "description": "View and modify Proxima configuration",
            "usage": "proxima config [SUBCOMMAND] [OPTIONS]",
            "examples": [
                {
                    "command": "proxima config show",
                    "description": "Show current configuration",
                },
                {
                    "command": "proxima config set backends.default cirq",
                    "description": "Set default backend to Cirq",
                },
                {
                    "command": "proxima config get consent.auto_approve_local_llm",
                    "description": "Get a specific config value",
                },
                {
                    "command": "proxima config edit",
                    "description": "Open config file in editor",
                },
            ],
            "subcommands": [
                ("show", "Display current configuration"),
                ("set", "Set a configuration value"),
                ("get", "Get a configuration value"),
                ("reset", "Reset configuration to defaults"),
                ("edit", "Open configuration in editor"),
            ],
            "related": ["init", "backends"],
        },
    }
    
    @classmethod
    def get_help(cls, command: str) -> dict[str, Any] | None:
        """Get detailed help for a command.
        
        Args:
            command: Command name
            
        Returns:
            Help information dict or None
        """
        return cls.COMMAND_HELP.get(command)
    
    @classmethod
    def print_help(cls, command: str) -> None:
        """Print detailed help for a command.
        
        Args:
            command: Command name
        """
        help_info = cls.get_help(command)
        
        if help_info is None:
            console.print(f"[red]No detailed help available for '{command}'[/red]")
            return
        
        # Title
        console.print(Panel(
            f"[bold cyan]proxima {command}[/bold cyan]\n\n{help_info['description']}",
            title="Command Help",
            border_style="blue",
        ))
        
        # Usage
        console.print(f"\n[bold]USAGE:[/bold]")
        console.print(f"  {help_info['usage']}")
        
        # Examples
        if "examples" in help_info:
            console.print(f"\n[bold]EXAMPLES:[/bold]")
            for ex in help_info["examples"]:
                console.print(f"\n  [cyan]{ex['command']}[/cyan]")
                console.print(f"    {ex['description']}")
        
        # Options
        if "options" in help_info:
            console.print(f"\n[bold]OPTIONS:[/bold]")
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="green")
            table.add_column()
            table.add_column(style="dim")
            
            for opt, desc, default in help_info["options"]:
                table.add_row(opt, desc, f"[default: {default}]")
            
            console.print(table)
        
        # Subcommands
        if "subcommands" in help_info:
            console.print(f"\n[bold]SUBCOMMANDS:[/bold]")
            for sub, desc in help_info["subcommands"]:
                console.print(f"  [green]{sub:12}[/green] {desc}")
        
        # Tips
        if "tips" in help_info:
            console.print(f"\n[bold]TIPS:[/bold]")
            for tip in help_info["tips"]:
                console.print(f"  • {tip}")
        
        # Related commands
        if "related" in help_info:
            related = ", ".join(f"[cyan]{cmd}[/cyan]" for cmd in help_info["related"])
            console.print(f"\n[bold]SEE ALSO:[/bold] {related}")


# =============================================================================
# Interactive Shell (Feature - CLI)
# =============================================================================


class InteractiveShell:
    """Interactive REPL-style shell for Proxima.
    
    Features:
    - Command history with readline
    - Tab completion for commands
    - Alias expansion
    - Rich formatted output
    - Help integration
    - Session persistence
    """
    
    PROMPT = "[bold cyan]proxima[/bold cyan]> "
    BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   [bold cyan]Proxima Interactive Shell[/bold cyan]                                        ║
║   Quantum Simulation Orchestration Framework                         ║
║                                                                      ║
║   Commands: run, compare, backends, config, history, session, agent  ║
║   Type 'help' for commands, 'help <cmd>' for details, 'exit' to quit ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    
    def __init__(
        self,
        history_file: Path | None = None,
        aliases: CommandAliases | None = None,
    ) -> None:
        """Initialize interactive shell.
        
        Args:
            history_file: Path to command history file
            aliases: Command aliases instance
        """
        self._history_file = history_file or Path.home() / ".proxima_history"
        self._aliases = aliases or command_aliases
        self._running = False
        self._commands: dict[str, Callable[[list[str]], None]] = {
            "help": self._cmd_help,
            "aliases": self._cmd_aliases,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "clear": self._cmd_clear,
            "history": self._cmd_history,
        }
        
        # Setup readline
        self._setup_readline()
    
    def _setup_readline(self) -> None:
        """Setup readline with history and completion."""
        if not HAS_READLINE or readline is None:
            return
            
        try:
            # Load history
            if self._history_file.exists():
                readline.read_history_file(str(self._history_file))
            
            # Set history length
            readline.set_history_length(1000)
            
            # Setup tab completion
            readline.set_completer(self._complete)
            readline.parse_and_bind("tab: complete")
            
        except Exception:
            pass  # Readline not available on all platforms
    
    def _save_history(self) -> None:
        """Save command history."""
        if not HAS_READLINE or readline is None:
            return
            
        try:
            readline.write_history_file(str(self._history_file))
        except Exception:
            pass
    
    def _complete(self, text: str, state: int) -> str | None:
        """Tab completion function for readline.
        
        Args:
            text: Current text being completed
            state: Completion state
            
        Returns:
            Completion suggestion or None
        """
        commands = [
            "run", "compare", "backends", "config", "history",
            "session", "agent", "ui", "help", "aliases", "exit", "clear",
        ]
        aliases = list(self._aliases._aliases.keys())
        all_completions = commands + aliases
        
        matches = [c for c in all_completions if c.startswith(text)]
        
        if state < len(matches):
            return matches[state]
        return None
    
    def run(self) -> None:
        """Run the interactive shell."""
        console.print(self.BANNER)
        
        self._running = True
        
        while self._running:
            try:
                # Get input
                line = console.input(self.PROMPT).strip()
                
                if not line:
                    continue
                
                # Parse input
                try:
                    args = shlex.split(line)
                except ValueError as e:
                    console.print(f"[red]Parse error: {e}[/red]")
                    continue
                
                # Execute
                self._execute(args)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' or 'quit' to leave[/yellow]")
            except EOFError:
                self._cmd_exit([])
        
        # Save history on exit
        self._save_history()
    
    def _execute(self, args: list[str]) -> None:
        """Execute a command.
        
        Args:
            args: Command arguments
        """
        if not args:
            return
        
        command = args[0].lower()
        
        # Check for built-in shell commands
        if command in self._commands:
            self._commands[command](args[1:])
            return
        
        # Expand aliases
        expanded = self._aliases.expand_argv(args)
        
        # Execute through main CLI
        try:
            from proxima.cli.main import app
            
            # Prepend 'proxima' for the CLI
            sys.argv = ["proxima"] + expanded
            app(standalone_mode=False)
            
        except SystemExit:
            pass  # Normal exit from typer
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    def _cmd_help(self, args: list[str]) -> None:
        """Show help."""
        if args:
            DetailedHelp.print_help(args[0])
        else:
            console.print("""
[bold]Available Commands:[/bold]

  [cyan]run[/cyan]        Execute quantum circuit simulations
  [cyan]compare[/cyan]    Compare results across multiple backends
  [cyan]backends[/cyan]   List and manage simulation backends
  [cyan]config[/cyan]     View and modify configuration
  [cyan]history[/cyan]    View past execution results
  [cyan]session[/cyan]    Manage execution sessions
  [cyan]agent[/cyan]      Run agent.md automation files
  [cyan]ui[/cyan]         Launch interactive terminal UI

[bold]Shell Commands:[/bold]

  [cyan]help[/cyan] [cmd]  Show help (detailed if command specified)
  [cyan]aliases[/cyan]     Show command aliases
  [cyan]history[/cyan]     Show command history
  [cyan]clear[/cyan]       Clear the screen
  [cyan]exit[/cyan]        Exit the shell

Type [cyan]help <command>[/cyan] for detailed help with examples.
""")
    
    def _cmd_aliases(self, args: list[str]) -> None:
        """Show command aliases."""
        self._aliases.print_aliases()
    
    def _cmd_exit(self, args: list[str]) -> None:
        """Exit the shell."""
        console.print("[dim]Goodbye![/dim]")
        self._running = False
    
    def _cmd_clear(self, args: list[str]) -> None:
        """Clear the screen."""
        console.clear()
    
    def _cmd_history(self, args: list[str]) -> None:
        """Show command history."""
        if not HAS_READLINE or readline is None:
            console.print("[dim]History not available (readline not installed)[/dim]")
            return
            
        try:
            history_length = readline.get_current_history_length()
            console.print(f"[bold]Command History ({history_length} entries):[/bold]\n")
            
            start = max(1, history_length - 20) if not args else 1
            for i in range(start, history_length + 1):
                item = readline.get_history_item(i)
                if item:
                    console.print(f"  {i:4}  {item}")
        except Exception:
            console.print("[dim]History not available[/dim]")


# =============================================================================
# CLI Commands for Interactive Features
# =============================================================================


interactive_app = typer.Typer(
    name="shell",
    help="Interactive shell and completion commands",
)


@interactive_app.command("interactive")
def cmd_interactive():
    """Launch interactive Proxima shell.
    
    Provides a REPL-style interface with:
    - Command history with arrow keys
    - Tab completion for commands
    - Alias expansion
    - Rich formatted output
    
    Example:
        proxima shell interactive
    """
    shell = InteractiveShell()
    shell.run()


@interactive_app.command("completion")
def cmd_completion(
    shell: str = typer.Argument(
        ...,
        help="Shell type: bash, zsh, powershell, fish",
    ),
    install: bool = typer.Option(
        False,
        "--install",
        "-i",
        help="Show installation instructions",
    ),
):
    """Generate shell completion scripts.
    
    Generates completion scripts for various shells.
    
    Examples:
        proxima shell completion bash
        proxima shell completion zsh --install
        proxima shell completion powershell > proxima_completion.ps1
    """
    if install:
        typer.echo(CompletionGenerator.install_instructions(shell))
    else:
        typer.echo(CompletionGenerator.generate(shell))


@interactive_app.command("aliases")
def cmd_aliases(
    add: str | None = typer.Option(
        None,
        "--add",
        "-a",
        help="Add alias in format 'alias=command'",
    ),
    remove: str | None = typer.Option(
        None,
        "--remove",
        "-r",
        help="Remove an alias",
    ),
):
    """Manage command aliases.
    
    View, add, or remove command aliases.
    
    Examples:
        proxima shell aliases
        proxima shell aliases --add "bell=run 'bell state'"
        proxima shell aliases --remove bell
    """
    if add:
        if "=" not in add:
            typer.echo("Error: Use format 'alias=command'")
            raise typer.Exit(1)
        alias, command = add.split("=", 1)
        command_aliases.add_alias(alias.strip(), command.strip())
        typer.echo(f"Added alias: {alias} -> {command}")
    elif remove:
        if command_aliases.remove_alias(remove):
            typer.echo(f"Removed alias: {remove}")
        else:
            typer.echo(f"Alias not found: {remove}")
    else:
        command_aliases.print_aliases()


@interactive_app.command("help")
def cmd_detailed_help(
    command: str = typer.Argument(
        ...,
        help="Command to get help for",
    ),
):
    """Show detailed help with examples.
    
    Provides comprehensive help including examples, options,
    and related commands.
    
    Examples:
        proxima shell help run
        proxima shell help compare
        proxima shell help backends
    """
    DetailedHelp.print_help(command)
