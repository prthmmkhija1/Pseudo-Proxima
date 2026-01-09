"""Custom widgets for Proxima TUI."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static,
    Label,
    Button,
    DataTable,
    ProgressBar as TextualProgressBar,
    RichLog,
    Markdown,
    Input,
    Switch,
)
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group


class StatusLevel(Enum):
    """Status levels for display."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


@dataclass
class StatusItem:
    """A status item for display."""
    label: str
    value: str
    level: StatusLevel = StatusLevel.INFO
    timestamp: Optional[float] = None


class StatusPanel(Static):
    """Panel showing system status information."""
    
    DEFAULT_CSS = """
    StatusPanel {
        border: solid $primary;
        padding: 1;
        margin: 1;
        height: auto;
    }
    
    StatusPanel .status-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    StatusPanel .status-ok {
        color: $success;
    }
    
    StatusPanel .status-warning {
        color: $warning;
    }
    
    StatusPanel .status-error {
        color: $error;
    }
    
    StatusPanel .status-info {
        color: $text;
    }
    """
    
    title: reactive[str] = reactive("System Status")
    items: reactive[list] = reactive(list)
    
    def __init__(
        self,
        title: str = "System Status",
        items: Optional[List[StatusItem]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.title = title
        self._items = items or []
    
    def compose(self) -> ComposeResult:
        yield Label(self.title, classes="status-title")
        yield Container(id="status-items")
    
    def on_mount(self) -> None:
        self._refresh_items()
    
    def _refresh_items(self) -> None:
        """Refresh the status items display."""
        container = self.query_one("#status-items", Container)
        container.remove_children()
        
        for item in self._items:
            level_class = f"status-{item.level.value}"
            text = f"  {item.label}: {item.value}"
            if item.timestamp:
                ago = self._format_time_ago(item.timestamp)
                text += f" ({ago})"
            container.mount(Label(text, classes=level_class))
    
    def _format_time_ago(self, timestamp: float) -> str:
        """Format a timestamp as 'X ago'."""
        diff = time.time() - timestamp
        if diff < 60:
            return f"{int(diff)}s ago"
        elif diff < 3600:
            return f"{int(diff / 60)}m ago"
        elif diff < 86400:
            return f"{int(diff / 3600)}h ago"
        else:
            return f"{int(diff / 86400)}d ago"
    
    def update_items(self, items: List[StatusItem]) -> None:
        """Update status items."""
        self._items = items
        self._refresh_items()
    
    def add_item(self, item: StatusItem) -> None:
        """Add a status item."""
        self._items.append(item)
        self._refresh_items()


class LogViewer(RichLog):
    """Enhanced log viewer with filtering and search."""
    
    DEFAULT_CSS = """
    LogViewer {
        border: solid $primary;
        padding: 1;
        margin: 1;
        height: 100%;
        background: $surface;
    }
    """
    
    def __init__(
        self,
        max_lines: int = 1000,
        auto_scroll: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            max_lines=max_lines,
            auto_scroll=auto_scroll,
            highlight=True,
            markup=True,
            **kwargs,
        )
        self._log_entries: List[Dict[str, Any]] = []
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.write(f"[dim]{timestamp}[/dim] [blue]INFO[/blue] {message}")
        self._log_entries.append({"level": "info", "message": message, "time": time.time()})
    
    def log_success(self, message: str) -> None:
        """Log a success message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.write(f"[dim]{timestamp}[/dim] [green]SUCCESS[/green] {message}")
        self._log_entries.append({"level": "success", "message": message, "time": time.time()})
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.write(f"[dim]{timestamp}[/dim] [yellow]WARNING[/yellow] {message}")
        self._log_entries.append({"level": "warning", "message": message, "time": time.time()})
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.write(f"[dim]{timestamp}[/dim] [red]ERROR[/red] {message}")
        self._log_entries.append({"level": "error", "message": message, "time": time.time()})
    
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.write(f"[dim]{timestamp}[/dim] [dim]DEBUG[/dim] {message}")
        self._log_entries.append({"level": "debug", "message": message, "time": time.time()})
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get all log entries."""
        return self._log_entries.copy()


class ProgressBar(Static):
    """Enhanced progress bar with status text."""
    
    DEFAULT_CSS = """
    ProgressBar {
        height: 3;
        margin: 1;
    }
    
    ProgressBar .progress-label {
        text-align: center;
    }
    
    ProgressBar .progress-bar {
        margin-top: 1;
    }
    """
    
    progress: reactive[float] = reactive(0.0)
    status: reactive[str] = reactive("")
    
    def __init__(
        self,
        total: float = 100.0,
        label: str = "Progress",
        show_percentage: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._total = total
        self._label = label
        self._show_percentage = show_percentage
        self._current = 0.0
    
    def compose(self) -> ComposeResult:
        yield Label(self._format_label(), classes="progress-label", id="progress-label")
        yield TextualProgressBar(total=self._total, classes="progress-bar", id="progress-bar")
    
    def _format_label(self) -> str:
        """Format the progress label."""
        text = self._label
        if self.status:
            text += f" - {self.status}"
        if self._show_percentage and self._total > 0:
            pct = (self._current / self._total) * 100
            text += f" ({pct:.1f}%)"
        return text
    
    def update_progress(self, current: float, status: str = "") -> None:
        """Update progress value and status."""
        self._current = current
        self.status = status
        
        bar = self.query_one("#progress-bar", TextualProgressBar)
        bar.update(progress=current)
        
        label = self.query_one("#progress-label", Label)
        label.update(self._format_label())
    
    def advance(self, amount: float = 1.0, status: str = "") -> None:
        """Advance progress by amount."""
        self.update_progress(self._current + amount, status)
    
    def reset(self) -> None:
        """Reset progress to zero."""
        self.update_progress(0.0, "")


class BackendStatus(Enum):
    """Backend connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CHECKING = "checking"


@dataclass
class BackendInfo:
    """Information about a backend."""
    name: str
    backend_type: str
    status: BackendStatus = BackendStatus.DISCONNECTED
    last_used: Optional[float] = None
    total_executions: int = 0
    avg_latency_ms: Optional[float] = None
    error_message: Optional[str] = None


class BackendCard(Static):
    """Card displaying backend status and info."""
    
    DEFAULT_CSS = """
    BackendCard {
        border: solid $primary;
        padding: 1;
        margin: 1;
        height: auto;
        min-width: 30;
    }
    
    BackendCard.connected {
        border: solid $success;
    }
    
    BackendCard.disconnected {
        border: solid $surface-lighten-2;
    }
    
    BackendCard.error {
        border: solid $error;
    }
    
    BackendCard .backend-name {
        text-style: bold;
        margin-bottom: 1;
    }
    
    BackendCard .backend-type {
        color: $text-muted;
    }
    
    BackendCard .backend-stats {
        margin-top: 1;
        color: $text-muted;
    }
    """
    
    class Selected(Message):
        """Message when backend card is selected."""
        def __init__(self, backend: BackendInfo) -> None:
            super().__init__()
            self.backend = backend
    
    def __init__(self, backend: BackendInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backend = backend
        self.add_class(backend.status.value)
    
    def compose(self) -> ComposeResult:
        status_icon = {
            BackendStatus.CONNECTED: "🟢",
            BackendStatus.DISCONNECTED: "⚪",
            BackendStatus.ERROR: "🔴",
            BackendStatus.CHECKING: "🟡",
        }.get(self.backend.status, "⚪")
        
        yield Label(f"{status_icon} {self.backend.name}", classes="backend-name")
        yield Label(f"Type: {self.backend.backend_type}", classes="backend-type")
        
        if self.backend.total_executions > 0:
            stats = f"Executions: {self.backend.total_executions}"
            if self.backend.avg_latency_ms:
                stats += f" | Avg: {self.backend.avg_latency_ms:.1f}ms"
            yield Label(stats, classes="backend-stats")
        
        if self.backend.error_message:
            yield Label(f"⚠ {self.backend.error_message}", classes="backend-error")
    
    def on_click(self) -> None:
        """Handle click on backend card."""
        self.post_message(self.Selected(self.backend))
    
    def update_status(self, status: BackendStatus) -> None:
        """Update backend status."""
        self.remove_class(self.backend.status.value)
        self.backend.status = status
        self.add_class(status.value)
        self.refresh()


class ResultsTable(DataTable):
    """Table for displaying execution results."""
    
    DEFAULT_CSS = """
    ResultsTable {
        height: 100%;
        margin: 1;
    }
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results: List[Dict[str, Any]] = []
    
    def on_mount(self) -> None:
        """Initialize table columns."""
        self.add_column("ID", key="id")
        self.add_column("Backend", key="backend")
        self.add_column("Status", key="status")
        self.add_column("Duration", key="duration")
        self.add_column("Timestamp", key="timestamp")
    
    def load_results(self, results: List[Dict[str, Any]]) -> None:
        """Load results into the table."""
        self._results = results
        self.clear()
        
        for result in results:
            status = result.get("status", "unknown")
            status_display = {
                "success": "[green]✓ Success[/green]",
                "failed": "[red]✗ Failed[/red]",
                "pending": "[yellow]⏳ Pending[/yellow]",
            }.get(status, status)
            
            duration = result.get("duration_ms")
            duration_str = f"{duration:.1f}ms" if duration else "-"
            
            timestamp = result.get("timestamp")
            if timestamp:
                ts_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
            else:
                ts_str = "-"
            
            self.add_row(
                result.get("id", "-"),
                result.get("backend", "-"),
                status_display,
                duration_str,
                ts_str,
                key=str(result.get("id", "")),
            )
    
    def get_selected_result(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected result."""
        row_key = self.cursor_row
        if row_key is not None and row_key < len(self._results):
            return self._results[row_key]
        return None


class HelpModal(Static):
    """Modal dialog showing keyboard shortcuts and help."""
    
    DEFAULT_CSS = """
    HelpModal {
        layer: modal;
        align: center middle;
        width: 60;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 2;
    }
    
    HelpModal .help-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 2;
    }
    
    HelpModal .help-section {
        margin-bottom: 1;
        text-style: bold;
    }
    
    HelpModal .help-shortcut {
        margin-left: 2;
    }
    
    HelpModal .help-close {
        margin-top: 2;
        text-align: center;
        color: $text-muted;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Label("⌨️ Keyboard Shortcuts", classes="help-title")
        
        # Navigation
        yield Label("Navigation", classes="help-section")
        yield Label("  [bold]1-5[/bold]  Switch screens", classes="help-shortcut")
        yield Label("  [bold]Tab[/bold]  Next element", classes="help-shortcut")
        yield Label("  [bold]↑/↓[/bold]  Navigate lists", classes="help-shortcut")
        yield Label("  [bold]Enter[/bold]  Select/Confirm", classes="help-shortcut")
        
        # Actions
        yield Label("Actions", classes="help-section")
        yield Label("  [bold]r[/bold]  Refresh data", classes="help-shortcut")
        yield Label("  [bold]e[/bold]  Execute task", classes="help-shortcut")
        yield Label("  [bold]s[/bold]  Stop execution", classes="help-shortcut")
        yield Label("  [bold]c[/bold]  Clear logs", classes="help-shortcut")
        
        # General
        yield Label("General", classes="help-section")
        yield Label("  [bold]?[/bold]  Toggle this help", classes="help-shortcut")
        yield Label("  [bold]q[/bold]  Quit application", classes="help-shortcut")
        yield Label("  [bold]Esc[/bold]  Close modal/Cancel", classes="help-shortcut")
        
        yield Label("Press Esc or ? to close", classes="help-close")
    
    def on_key(self, event) -> None:
        """Handle key press to close modal."""
        if event.key in ("escape", "question_mark"):
            self.remove()


class ConfigInput(Static):
    """Configuration input field with label."""
    
    DEFAULT_CSS = """
    ConfigInput {
        height: 3;
        margin-bottom: 1;
    }
    
    ConfigInput .config-label {
        margin-right: 1;
    }
    
    ConfigInput Horizontal {
        height: 3;
    }
    """
    
    class Changed(Message):
        """Message when config value changes."""
        def __init__(self, key: str, value: str) -> None:
            super().__init__()
            self.key = key
            self.value = value
    
    def __init__(
        self,
        key: str,
        label: str,
        value: str = "",
        placeholder: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._key = key
        self._label = label
        self._value = value
        self._placeholder = placeholder
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(f"{self._label}:", classes="config-label")
            yield Input(
                value=self._value,
                placeholder=self._placeholder,
                id=f"input-{self._key}",
            )
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        self.post_message(self.Changed(self._key, event.value))


class ConfigToggle(Static):
    """Configuration toggle with label."""
    
    DEFAULT_CSS = """
    ConfigToggle {
        height: 3;
        margin-bottom: 1;
    }
    
    ConfigToggle .toggle-label {
        margin-right: 1;
    }
    
    ConfigToggle Horizontal {
        height: 3;
    }
    """
    
    class Changed(Message):
        """Message when toggle value changes."""
        def __init__(self, key: str, value: bool) -> None:
            super().__init__()
            self.key = key
            self.value = value
    
    def __init__(
        self,
        key: str,
        label: str,
        value: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._key = key
        self._label = label
        self._value = value
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(f"{self._label}:", classes="toggle-label")
            yield Switch(value=self._value, id=f"switch-{self._key}")
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        self.post_message(self.Changed(self._key, event.value))


class ExecutionCard(Static):
    """Card showing a recent execution."""
    
    DEFAULT_CSS = """
    ExecutionCard {
        border: solid $primary;
        padding: 1;
        margin: 1;
        height: auto;
    }
    
    ExecutionCard.success {
        border: solid $success;
    }
    
    ExecutionCard.failed {
        border: solid $error;
    }
    
    ExecutionCard .exec-title {
        text-style: bold;
    }
    
    ExecutionCard .exec-details {
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        execution_id: str,
        backend: str,
        status: str,
        duration_ms: float,
        timestamp: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._id = execution_id
        self._backend = backend
        self._status = status
        self._duration = duration_ms
        self._timestamp = timestamp
        self.add_class(status)
    
    def compose(self) -> ComposeResult:
        status_icon = "✓" if self._status == "success" else "✗"
        yield Label(f"{status_icon} Execution {self._id}", classes="exec-title")
        
        ts_str = datetime.fromtimestamp(self._timestamp).strftime("%H:%M:%S")
        yield Label(f"Backend: {self._backend} | Duration: {self._duration:.1f}ms | {ts_str}", classes="exec-details")
