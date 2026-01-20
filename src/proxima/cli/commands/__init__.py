"""
CLI commands module.

Command submodules are imported directly by main.py to avoid circular imports.
This module also provides a rich help examples system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# Rich Help Examples System
# =============================================================================


@dataclass
class CommandExample:
    """A single command example with description.

    Attributes:
        command: The example command to run.
        description: What the example demonstrates.
        output: Optional expected output snippet.
        notes: Additional notes or tips.
        tags: Tags for categorization.
    """

    command: str
    description: str
    output: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def format(self, include_output: bool = False) -> str:
        """Format the example for display."""
        lines = [f"  $ {self.command}"]
        lines.append(f"    # {self.description}")
        if self.notes:
            lines.append(f"    Note: {self.notes}")
        if include_output and self.output:
            lines.append(f"    Output: {self.output}")
        return "\n".join(lines)


@dataclass
class CommandHelp:
    """Rich help content for a command.

    Attributes:
        name: Command name.
        summary: One-line summary.
        description: Full description.
        examples: List of examples.
        related: Related commands.
        see_also: External documentation links.
    """

    name: str
    summary: str
    description: str
    examples: List[CommandExample] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)

    def format_examples(self, max_examples: int = 5) -> str:
        """Format examples for display."""
        if not self.examples:
            return "No examples available."

        lines = ["EXAMPLES:"]
        for example in self.examples[:max_examples]:
            lines.append(example.format())
            lines.append("")

        if len(self.examples) > max_examples:
            lines.append(f"  ... and {len(self.examples) - max_examples} more examples")
            lines.append("  Use 'proxima help <command> --all' to see all examples")

        return "\n".join(lines)

    def format_full(self) -> str:
        """Format full help content."""
        lines = [
            "=" * 60,
            f"COMMAND: {self.name}",
            "=" * 60,
            "",
            self.summary,
            "",
            "DESCRIPTION:",
            f"  {self.description}",
            "",
            self.format_examples(),
        ]

        if self.related:
            lines.extend([
                "",
                "RELATED COMMANDS:",
                "  " + ", ".join(self.related),
            ])

        if self.see_also:
            lines.extend([
                "",
                "SEE ALSO:",
            ])
            for link in self.see_also:
                lines.append(f"  - {link}")

        lines.append("")
        return "\n".join(lines)


class HelpExamplesRegistry:
    """Registry for command help examples.

    Provides centralized storage and retrieval of rich
    help content for CLI commands.

    Example:
        >>> registry = HelpExamplesRegistry()
        >>> registry.get_examples("run")
        [CommandExample(...), ...]
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._commands: Dict[str, CommandHelp] = {}
        self._load_builtin_help()

    def _load_builtin_help(self) -> None:
        """Load built-in help content."""
        # Run command help
        self.register(CommandHelp(
            name="run",
            summary="Execute quantum circuit simulations",
            description=(
                "The run command plans and executes quantum circuit simulations. "
                "It uses LLM-powered planning to translate natural language objectives "
                "into executable circuits, then runs them on the selected backend."
            ),
            examples=[
                CommandExample(
                    command='proxima run "create bell state"',
                    description="Create and run a Bell state circuit",
                    tags=["basic", "bell"],
                ),
                CommandExample(
                    command='proxima run "quantum teleportation" --backend cirq',
                    description="Run teleportation circuit on Cirq backend",
                    tags=["intermediate", "teleportation"],
                ),
                CommandExample(
                    command="proxima run demo --shots 2000",
                    description="Run demo circuit with custom shot count",
                    tags=["basic", "demo"],
                ),
                CommandExample(
                    command="proxima run demo --backend qiskit --validate",
                    description="Run with validation on Qiskit backend",
                    tags=["qiskit", "validation"],
                ),
                CommandExample(
                    command='proxima run "qft on 5 qubits" --timeout 30',
                    description="Run QFT with 30-second timeout",
                    tags=["qft", "timeout"],
                ),
                CommandExample(
                    command="proxima run demo --benchmark --benchmark-runs 5",
                    description="Run with benchmarking enabled (5 runs)",
                    tags=["benchmark"],
                ),
                CommandExample(
                    command="proxima run demo --no-progress --no-save",
                    description="Run silently without saving results",
                    tags=["quiet", "advanced"],
                ),
            ],
            related=["compare", "backends", "history"],
            see_also=["docs/user-guide/running-simulations.md"],
        ))

        # Compare command help
        self.register(CommandHelp(
            name="compare",
            summary="Compare results across multiple backends",
            description=(
                "The compare command runs the same circuit on multiple backends "
                "and compares the results. Useful for validating correctness, "
                "benchmarking performance, and identifying backend differences."
            ),
            examples=[
                CommandExample(
                    command='proxima compare "bell state"',
                    description="Compare Bell state across default backends",
                    tags=["basic", "bell"],
                ),
                CommandExample(
                    command='proxima compare "bell state" --all',
                    description="Compare on all available backends",
                    tags=["comprehensive"],
                ),
                CommandExample(
                    command='proxima compare "grover" --backends lret,cirq,qiskit',
                    description="Compare on specific backends only",
                    tags=["selective"],
                ),
                CommandExample(
                    command="proxima compare demo --format json",
                    description="Output comparison results as JSON",
                    tags=["json", "export"],
                ),
                CommandExample(
                    command="proxima compare demo --save comparison_results.json",
                    description="Save comparison results to file",
                    tags=["save", "export"],
                ),
                CommandExample(
                    command="proxima compare demo --shots 5000 --runs 3",
                    description="Compare with custom shots and multiple runs",
                    tags=["benchmark"],
                ),
            ],
            related=["run", "backends", "benchmark"],
            see_also=["docs/user-guide/comparing-backends.md"],
        ))

        # Backends command help
        self.register(CommandHelp(
            name="backends",
            summary="List and manage simulation backends",
            description=(
                "The backends command lists available quantum simulation backends, "
                "shows their capabilities, and allows configuration. Backends include "
                "LRET, Cirq, Qiskit Aer, QuEST, qsim, and cuQuantum."
            ),
            examples=[
                CommandExample(
                    command="proxima backends list",
                    description="List all available backends",
                    tags=["basic", "list"],
                ),
                CommandExample(
                    command="proxima backends list --available",
                    description="List only installed/available backends",
                    tags=["available"],
                ),
                CommandExample(
                    command="proxima backends info lret",
                    description="Show detailed info about LRET backend",
                    tags=["info"],
                ),
                CommandExample(
                    command="proxima backends info cirq --capabilities",
                    description="Show Cirq backend capabilities",
                    tags=["info", "capabilities"],
                ),
                CommandExample(
                    command="proxima backends set-default qiskit",
                    description="Set Qiskit as the default backend",
                    tags=["config"],
                ),
                CommandExample(
                    command="proxima backends test --all",
                    description="Test all backends for availability",
                    tags=["test", "diagnostics"],
                ),
            ],
            related=["run", "compare", "config"],
            see_also=["docs/backends/backend-selection.md"],
        ))

        # Config command help
        self.register(CommandHelp(
            name="config",
            summary="View and modify configuration",
            description=(
                "The config command manages Proxima configuration settings. "
                "Configuration can be set at user, project, or session level."
            ),
            examples=[
                CommandExample(
                    command="proxima config show",
                    description="Show current configuration",
                    tags=["basic", "view"],
                ),
                CommandExample(
                    command="proxima config show --format yaml",
                    description="Show config in YAML format",
                    tags=["yaml", "view"],
                ),
                CommandExample(
                    command="proxima config set backends.default_backend cirq",
                    description="Set default backend to Cirq",
                    tags=["set", "backends"],
                ),
                CommandExample(
                    command="proxima config set general.output_format json",
                    description="Set output format to JSON",
                    tags=["set", "output"],
                ),
                CommandExample(
                    command="proxima config get backends.default_backend",
                    description="Get a specific configuration value",
                    tags=["get"],
                ),
                CommandExample(
                    command="proxima config reset",
                    description="Reset configuration to defaults",
                    tags=["reset"],
                ),
                CommandExample(
                    command="proxima config path",
                    description="Show configuration file paths",
                    tags=["paths"],
                ),
            ],
            related=["backends", "init"],
            see_also=["docs/getting-started/configuration.md"],
        ))

        # History command help
        self.register(CommandHelp(
            name="history",
            summary="View past execution results",
            description=(
                "The history command displays past simulation results. "
                "Results are stored locally and can be filtered, exported, "
                "or compared."
            ),
            examples=[
                CommandExample(
                    command="proxima history list",
                    description="List recent execution history",
                    tags=["basic", "list"],
                ),
                CommandExample(
                    command="proxima history list --limit 50",
                    description="List last 50 results",
                    tags=["limit"],
                ),
                CommandExample(
                    command="proxima history show abc123",
                    description="Show details of a specific result",
                    tags=["show", "details"],
                ),
                CommandExample(
                    command="proxima history export --format csv -o results.csv",
                    description="Export history to CSV file",
                    tags=["export", "csv"],
                ),
                CommandExample(
                    command="proxima history clear --before 2024-01-01",
                    description="Clear history before a date",
                    tags=["clear", "maintenance"],
                ),
                CommandExample(
                    command="proxima history stats",
                    description="Show execution statistics",
                    tags=["stats", "analytics"],
                ),
            ],
            related=["run", "session", "export"],
            see_also=["docs/user-guide/benchmarking.md"],
        ))

        # Session command help
        self.register(CommandHelp(
            name="session",
            summary="Manage execution sessions",
            description=(
                "The session command manages execution sessions. Sessions group "
                "related runs together and can be named, resumed, and exported."
            ),
            examples=[
                CommandExample(
                    command="proxima session list",
                    description="List all sessions",
                    tags=["basic", "list"],
                ),
                CommandExample(
                    command='proxima session create --name "Bell experiments"',
                    description="Create a named session",
                    tags=["create"],
                ),
                CommandExample(
                    command="proxima session use abc123",
                    description="Switch to a specific session",
                    tags=["switch"],
                ),
                CommandExample(
                    command="proxima session info",
                    description="Show current session details",
                    tags=["info"],
                ),
                CommandExample(
                    command="proxima session export --format json -o session.json",
                    description="Export session results",
                    tags=["export"],
                ),
                CommandExample(
                    command="proxima session close",
                    description="Close current session",
                    tags=["close"],
                ),
            ],
            related=["run", "history"],
        ))

        # Benchmark command help
        self.register(CommandHelp(
            name="benchmark",
            summary="Run performance benchmarks",
            description=(
                "The benchmark command runs performance benchmarks across backends "
                "with configurable parameters like shots, qubits, and iterations."
            ),
            examples=[
                CommandExample(
                    command="proxima benchmark run --suite standard",
                    description="Run standard benchmark suite",
                    tags=["basic", "suite"],
                ),
                CommandExample(
                    command="proxima benchmark run --circuit bell --runs 10",
                    description="Benchmark Bell circuit with 10 runs",
                    tags=["circuit", "runs"],
                ),
                CommandExample(
                    command="proxima benchmark run --suite comprehensive --backends all",
                    description="Run comprehensive benchmarks on all backends",
                    tags=["comprehensive"],
                ),
                CommandExample(
                    command="proxima benchmark compare lret cirq --metric execution_time",
                    description="Compare specific metric between backends",
                    tags=["compare", "metric"],
                ),
                CommandExample(
                    command="proxima benchmark report --format html -o report.html",
                    description="Generate HTML benchmark report",
                    tags=["report", "export"],
                ),
                CommandExample(
                    command="proxima benchmark history --last 5",
                    description="Show last 5 benchmark runs",
                    tags=["history"],
                ),
            ],
            related=["run", "compare", "backends"],
            see_also=["docs/user-guide/benchmarking.md"],
        ))

        # Agent command help
        self.register(CommandHelp(
            name="agent",
            summary="Run agent.md automation files",
            description=(
                "The agent command executes automation scripts written in the "
                "agent.md format. Agent files can contain multi-step workflows "
                "with LLM-powered planning and decision making."
            ),
            examples=[
                CommandExample(
                    command="proxima agent run agent.md",
                    description="Run an agent file",
                    tags=["basic", "run"],
                ),
                CommandExample(
                    command="proxima agent run workflow.md --dry-run",
                    description="Preview agent execution without running",
                    tags=["preview", "dry-run"],
                ),
                CommandExample(
                    command="proxima agent run agent.md --verbose",
                    description="Run with verbose output",
                    tags=["verbose"],
                ),
                CommandExample(
                    command="proxima agent validate agent.md",
                    description="Validate agent file syntax",
                    tags=["validate"],
                ),
                CommandExample(
                    command="proxima agent list",
                    description="List available agent files",
                    tags=["list"],
                ),
            ],
            related=["run"],
            see_also=["docs/user-guide/agent-files.md"],
        ))

        # UI command help
        self.register(CommandHelp(
            name="ui",
            summary="Launch interactive terminal UI",
            description=(
                "The ui command launches an interactive terminal user interface "
                "for browsing results, managing sessions, and monitoring executions."
            ),
            examples=[
                CommandExample(
                    command="proxima ui",
                    description="Launch the terminal UI",
                    tags=["basic"],
                ),
                CommandExample(
                    command="proxima ui --theme dark",
                    description="Launch with dark theme",
                    tags=["theme"],
                ),
                CommandExample(
                    command="proxima ui --session abc123",
                    description="Launch UI with specific session",
                    tags=["session"],
                ),
            ],
            related=["interactive", "history"],
        ))

    def register(
        self,
        help_content: CommandHelp,
        overwrite: bool = False,
    ) -> None:
        """Register help content for a command.

        Args:
            help_content: Help content to register.
            overwrite: Whether to overwrite existing.
        """
        if help_content.name in self._commands and not overwrite:
            return
        self._commands[help_content.name] = help_content

    def get(self, command: str) -> Optional[CommandHelp]:
        """Get help content for a command."""
        return self._commands.get(command)

    def get_examples(
        self,
        command: str,
        tags: Optional[List[str]] = None,
        max_examples: int = 10,
    ) -> List[CommandExample]:
        """Get examples for a command.

        Args:
            command: Command name.
            tags: Filter by tags.
            max_examples: Maximum examples to return.

        Returns:
            List of matching examples.
        """
        help_content = self._commands.get(command)
        if not help_content:
            return []

        examples = help_content.examples

        if tags:
            examples = [
                e for e in examples
                if any(t in e.tags for t in tags)
            ]

        return examples[:max_examples]

    def list_commands(self) -> List[str]:
        """List all commands with help content."""
        return sorted(self._commands.keys())

    def search_examples(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[tuple[str, CommandExample]]:
        """Search examples across all commands.

        Args:
            query: Search query.
            max_results: Maximum results.

        Returns:
            List of (command, example) tuples.
        """
        query_lower = query.lower()
        results = []

        for command, help_content in self._commands.items():
            for example in help_content.examples:
                if (
                    query_lower in example.command.lower() or
                    query_lower in example.description.lower() or
                    any(query_lower in t for t in example.tags)
                ):
                    results.append((command, example))
                    if len(results) >= max_results:
                        return results

        return results

    def format_command_help(
        self,
        command: str,
        include_all: bool = False,
    ) -> str:
        """Format help for a specific command.

        Args:
            command: Command name.
            include_all: Include all examples.

        Returns:
            Formatted help text.
        """
        help_content = self._commands.get(command)
        if not help_content:
            return f"No help available for '{command}'"

        if include_all:
            return help_content.format_full()

        # Abbreviated version
        lines = [
            f"COMMAND: {help_content.name}",
            "",
            help_content.summary,
            "",
            help_content.format_examples(max_examples=3),
        ]

        if help_content.related:
            lines.extend([
                "",
                f"Related: {', '.join(help_content.related)}",
            ])

        return "\n".join(lines)


# Global help registry
help_registry = HelpExamplesRegistry()


__all__ = [
    "init",
    "config",
    "run",
    "compare",
    "backends",
    "benchmark",
    "version",
    "history",
    "session",
    "agent",
    "ui",
    "CommandExample",
    "CommandHelp",
    "HelpExamplesRegistry",
    "help_registry",
]
