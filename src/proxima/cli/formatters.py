"""CLI Output Formatters - Text, JSON, Table, and Rich output formatting.

This module provides:
- OutputFormatter: Base class for output formatting
- TextFormatter: Plain text output
- JsonFormatter: JSON output
- TableFormatter: Table-based output (using rich tables)
- RichFormatter: Rich console output with colors/styles
- format_output: Auto-select formatter based on format type
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import Any, TextIO

import typer

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ========== Format Types ==========


class OutputFormat(str, Enum):
    """Supported output formats."""

    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    TABLE = "table"
    RICH = "rich"
    CSV = "csv"


# ========== Base Formatter ==========


class OutputFormatter(ABC):
    """Base class for output formatters."""

    def __init__(
        self,
        stream: TextIO | None = None,
        no_color: bool = False,
    ) -> None:
        self.stream = stream or sys.stdout
        self.no_color = no_color

    @abstractmethod
    def format(self, data: Any, **kwargs) -> str:
        """Format data as string."""
        pass

    def write(self, data: Any, **kwargs) -> None:
        """Format and write data to stream."""
        output = self.format(data, **kwargs)
        self.stream.write(output)
        if not output.endswith("\n"):
            self.stream.write("\n")
        self.stream.flush()


# ========== Text Formatter ==========


class TextFormatter(OutputFormatter):
    """Plain text output formatter."""

    def __init__(
        self,
        stream: TextIO | None = None,
        indent: int = 2,
        no_color: bool = False,
    ) -> None:
        super().__init__(stream, no_color)
        self.indent = indent

    def format(self, data: Any, **kwargs) -> str:
        """Format data as plain text."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return self._format_dict(data, level=0)
        elif isinstance(data, (list, tuple)):
            return self._format_list(data, level=0)
        else:
            return str(data)

    def _format_dict(self, data: dict, level: int = 0) -> str:
        """Format dictionary as text."""
        lines = []
        prefix = " " * (self.indent * level)

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, level + 1))
            elif isinstance(value, (list, tuple)):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_list(value, level + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")

        return "\n".join(lines)

    def _format_list(self, data: list | tuple, level: int = 0) -> str:
        """Format list as text."""
        lines = []
        prefix = " " * (self.indent * level)

        for item in data:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(self._format_dict(item, level + 1))
            elif isinstance(item, (list, tuple)):
                lines.append(f"{prefix}-")
                lines.append(self._format_list(item, level + 1))
            else:
                lines.append(f"{prefix}- {item}")

        return "\n".join(lines)


# ========== JSON Formatter ==========


class JsonFormatter(OutputFormatter):
    """JSON output formatter."""

    def __init__(
        self,
        stream: TextIO | None = None,
        indent: int = 2,
        no_color: bool = False,
    ) -> None:
        super().__init__(stream, no_color)
        self.json_indent = indent

    def format(self, data: Any, **kwargs) -> str:
        """Format data as JSON."""
        compact = kwargs.get("compact", False)

        return json.dumps(
            data,
            indent=None if compact else self.json_indent,
            default=self._json_serializer,
            sort_keys=kwargs.get("sort_keys", False),
        )

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return str(obj)


# ========== YAML Formatter ==========


class YamlFormatter(OutputFormatter):
    """YAML output formatter."""

    def format(self, data: Any, **kwargs) -> str:
        """Format data as YAML."""
        try:
            import yaml

            return yaml.safe_dump(
                data,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        except ImportError:
            # Fall back to JSON if YAML not available
            return JsonFormatter(self.stream, no_color=self.no_color).format(data)


# ========== CSV Formatter ==========


class CsvFormatter(OutputFormatter):
    """CSV output formatter (for list of dicts)."""

    def format(self, data: Any, **kwargs) -> str:
        """Format data as CSV."""
        import csv

        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else [{"value": data}]

        if not data:
            return ""

        output = StringIO()
        fieldnames = self._extract_fieldnames(data)
        writer = csv.DictWriter(output, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            if isinstance(row, dict):
                # Flatten nested values
                flat_row = {k: self._flatten_value(v) for k, v in row.items()}
                writer.writerow(flat_row)

        return output.getvalue()

    def _extract_fieldnames(self, data: list) -> list[str]:
        """Extract column names from data."""
        fields = set()
        for row in data:
            if isinstance(row, dict):
                fields.update(row.keys())
        return sorted(fields)

    def _flatten_value(self, value: Any) -> str:
        """Flatten complex values to strings."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value) if value is not None else ""


# ========== Table Formatter ==========


class TableFormatter(OutputFormatter):
    """Table output formatter (uses rich.Table if available)."""

    def __init__(
        self,
        stream: TextIO | None = None,
        no_color: bool = False,
        show_header: bool = True,
        box_style: str = "rounded",
    ) -> None:
        super().__init__(stream, no_color)
        self.show_header = show_header
        self.box_style = box_style

    def format(self, data: Any, **kwargs) -> str:
        """Format data as table."""
        if RICH_AVAILABLE and not self.no_color:
            return self._format_rich_table(data, **kwargs)
        else:
            return self._format_ascii_table(data, **kwargs)

    def _format_rich_table(self, data: Any, **kwargs) -> str:
        """Format using Rich table."""
        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else [{"value": data}]

        if not data:
            return "(empty)"

        # Get box style
        from rich import box as rich_box

        box_styles = {
            "rounded": rich_box.ROUNDED,
            "simple": rich_box.SIMPLE,
            "minimal": rich_box.MINIMAL,
            "double": rich_box.DOUBLE,
            "ascii": rich_box.ASCII,
        }
        box = box_styles.get(self.box_style, rich_box.ROUNDED)

        table = Table(
            box=box,
            show_header=self.show_header,
            title=kwargs.get("title"),
        )

        # Add columns from first row
        if data and isinstance(data[0], dict):
            for key in data[0].keys():
                table.add_column(str(key).replace("_", " ").title())

        # Add rows
        for row in data:
            if isinstance(row, dict):
                values = [self._format_cell(v) for v in row.values()]
                table.add_row(*values)

        # Render to string
        console = Console(file=StringIO(), force_terminal=not self.no_color)
        console.print(table)
        return console.file.getvalue()

    def _format_ascii_table(self, data: Any, **kwargs) -> str:
        """Format as ASCII table (fallback)."""
        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else [{"value": data}]

        if not data:
            return "(empty)"

        # Get column widths
        columns = list(data[0].keys()) if isinstance(data[0], dict) else ["value"]
        widths = {col: len(str(col)) for col in columns}

        for row in data:
            if isinstance(row, dict):
                for col in columns:
                    widths[col] = max(widths[col], len(str(row.get(col, ""))))

        # Build table
        lines = []

        # Header
        if self.show_header:
            header = " | ".join(str(col).ljust(widths[col]) for col in columns)
            separator = "-+-".join("-" * widths[col] for col in columns)
            lines.append(header)
            lines.append(separator)

        # Rows
        for row in data:
            if isinstance(row, dict):
                line = " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
                lines.append(line)

        return "\n".join(lines)

    def _format_cell(self, value: Any) -> str:
        """Format a cell value."""
        if value is None:
            return ""
        elif isinstance(value, bool):
            return "✓" if value else "✗"
        elif isinstance(value, (dict, list)):
            return json.dumps(value)
        else:
            return str(value)


# ========== Rich Formatter ==========


class RichFormatter(OutputFormatter):
    """Rich console output with colors and styles."""

    def __init__(
        self,
        stream: TextIO | None = None,
        no_color: bool = False,
        theme: str = "default",
    ) -> None:
        super().__init__(stream, no_color)
        self.theme = theme
        if RICH_AVAILABLE:
            self.console = Console(
                file=stream or sys.stdout,
                force_terminal=not no_color,
                color_system=None if no_color else "auto",
            )
        else:
            self.console = None

    def format(self, data: Any, **kwargs) -> str:
        """Format data with Rich styling."""
        if not RICH_AVAILABLE or self.no_color:
            return TextFormatter(self.stream, no_color=self.no_color).format(data)

        output = StringIO()
        console = Console(file=output, force_terminal=True)

        style = kwargs.get("style", "default")

        if isinstance(data, str):
            text = Text(data, style=style if style != "default" else None)
            console.print(text)
        elif isinstance(data, dict):
            self._print_dict_tree(console, data, kwargs.get("title", "Result"), style=style)
        elif isinstance(data, (list, tuple)):
            self._print_list(console, data, kwargs.get("title", "Items"), style=style)
        else:
            console.print(str(data), style=style if style != "default" else None)

        return output.getvalue()

    def _print_dict_tree(
        self,
        console: Console,
        data: dict,
        title: str = "Data",
        style: str = "default",
    ) -> None:
        """Print dictionary as tree."""
        tree = Tree(f"[bold]{title}[/bold]")
        self._add_dict_to_tree(tree, data)
        console.print(tree)

    def _add_dict_to_tree(self, tree: Tree, data: dict) -> None:
        """Recursively add dict to tree."""
        for key, value in data.items():
            if isinstance(value, dict):
                branch = tree.add(f"[cyan]{key}[/cyan]")
                self._add_dict_to_tree(branch, value)
            elif isinstance(value, (list, tuple)):
                branch = tree.add(f"[cyan]{key}[/cyan]")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        item_branch = branch.add(f"[dim][{i}][/dim]")
                        self._add_dict_to_tree(item_branch, item)
                    else:
                        branch.add(f"[dim][{i}][/dim] {item}")
            else:
                tree.add(f"[cyan]{key}[/cyan]: {value}")

    def _print_list(
        self,
        console: Console,
        data: list | tuple,
        title: str = "Items",
        style: str = "default",
    ) -> None:
        """Print list with styling."""
        tree = Tree(f"[bold]{title}[/bold]")
        for i, item in enumerate(data):
            if isinstance(item, dict):
                branch = tree.add(f"[dim][{i}][/dim]")
                self._add_dict_to_tree(branch, item)
            else:
                tree.add(f"[dim][{i}][/dim] {item}")
        console.print(tree)

    def success(self, message: str) -> None:
        """Print success message."""
        if self.console:
            self.console.print(f"[green]✓[/green] {message}")
        else:
            typer.echo(f"✓ {message}")

    def error(self, message: str) -> None:
        """Print error message."""
        if self.console:
            self.console.print(f"[red]✗[/red] {message}")
        else:
            typer.echo(f"✗ {message}", err=True)

    def warning(self, message: str) -> None:
        """Print warning message."""
        if self.console:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
        else:
            typer.echo(f"⚠ {message}")

    def info(self, message: str) -> None:
        """Print info message."""
        if self.console:
            self.console.print(f"[blue]ℹ[/blue] {message}")
        else:
            typer.echo(f"ℹ {message}")

    def panel(self, content: str, title: str = "") -> None:
        """Print content in a panel."""
        if self.console:
            self.console.print(Panel(content, title=title))
        else:
            typer.echo(f"=== {title} ===")
            typer.echo(content)
            typer.echo("=" * (len(title) + 8))

    def code(self, code: str, language: str = "python") -> None:
        """Print syntax-highlighted code."""
        if self.console:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            typer.echo(code)


# ========== Formatter Factory ==========


def get_formatter(
    format: str | OutputFormat,
    stream: TextIO | None = None,
    no_color: bool = False,
    **kwargs,
) -> OutputFormatter:
    """Get formatter for specified format type."""
    if isinstance(format, str):
        try:
            format = OutputFormat(format.lower())
        except ValueError:
            format = OutputFormat.TEXT

    formatters = {
        OutputFormat.TEXT: TextFormatter,
        OutputFormat.JSON: JsonFormatter,
        OutputFormat.YAML: YamlFormatter,
        OutputFormat.TABLE: TableFormatter,
        OutputFormat.CSV: CsvFormatter,
        OutputFormat.RICH: RichFormatter,
    }

    formatter_class = formatters.get(format, TextFormatter)
    return formatter_class(stream=stream, no_color=no_color, **kwargs)


def format_output(
    data: Any,
    format: str | OutputFormat = OutputFormat.TEXT,
    no_color: bool = False,
    **kwargs,
) -> str:
    """Format data using specified formatter."""
    formatter = get_formatter(format, no_color=no_color, **kwargs)
    return formatter.format(data, **kwargs)


# ========== Output Helpers ==========


@dataclass
class OutputConfig:
    """Configuration for output formatting."""

    format: OutputFormat = OutputFormat.TEXT
    no_color: bool = False
    quiet: bool = False
    verbose: int = 0
    stream: TextIO | None = None

    @classmethod
    def from_context(cls, ctx: typer.Context) -> OutputConfig:
        """Create from Typer context."""
        obj = ctx.obj or {}
        format_str = obj.get("output_format", "text")
        try:
            format = OutputFormat(format_str.lower())
        except ValueError:
            format = OutputFormat.TEXT

        return cls(
            format=format,
            no_color=obj.get("no_color", False),
            quiet=obj.get("quiet", False),
            verbose=obj.get("verbose", 0),
        )


def echo_output(
    ctx: typer.Context,
    data: Any,
    format: str | None = None,
    **kwargs,
) -> None:
    """Echo formatted output based on context settings."""
    config = OutputConfig.from_context(ctx)

    if config.quiet:
        return

    output_format = OutputFormat(format) if format else config.format
    formatter = get_formatter(output_format, no_color=config.no_color)
    formatter.write(data, **kwargs)


def echo_result(
    ctx: typer.Context,
    result: Any,
    success_message: str = "Done",
    error_message: str = "Failed",
) -> None:
    """Echo workflow result with appropriate styling."""
    config = OutputConfig.from_context(ctx)

    if config.quiet:
        return

    formatter = RichFormatter(no_color=config.no_color)

    if hasattr(result, "success"):
        if result.success:
            formatter.success(success_message)
            if config.verbose > 0 and hasattr(result, "output"):
                echo_output(ctx, result.output)
        else:
            formatter.error(f"{error_message}: {getattr(result, 'error', 'unknown')}")
    else:
        echo_output(ctx, result)
