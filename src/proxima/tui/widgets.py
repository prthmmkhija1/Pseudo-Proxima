"""TUI Widgets for Proxima.

Step 6.1: Rich widget library including:
- StatusPanel: System status display
- LogViewer: Real-time log display
- ProgressBar: Custom progress bar
- BackendCard: Backend status card
- ResultsTable: Execution results table
- HelpModal: Keyboard shortcuts help
- MetricDisplay: Metric visualization
- Timer: Execution timer display
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    ProgressBar as TextualProgressBar,
    Static,
    Switch,
)

# ========== Status Level Enum ==========


class StatusLevel(Enum):
    """Status level for items."""

    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StatusItem:
    """A status item for display in StatusPanel."""

    label: str
    value: str
    level: StatusLevel = StatusLevel.INFO
    timestamp: float | None = None


# ========== Status Indicators ==========


class StatusIndicator(Static):
    """Status indicator with icon and color."""

    DEFAULT_CSS = """
    StatusIndicator {
        width: auto;
        height: 1;
        padding: 0 1;
    }

    StatusIndicator.success { color: $success; }
    StatusIndicator.error { color: $error; }
    StatusIndicator.warning { color: $warning; }
    StatusIndicator.info { color: $info; }
    StatusIndicator.pending { color: $text-muted; }
    StatusIndicator.running { color: $accent; }
    """

    ICONS = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "ℹ",
        "pending": "○",
        "running": "◉",
        "connected": "●",
        "disconnected": "○",
    }

    status = reactive("pending")

    def __init__(self, status: str = "pending", label: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self.status = status

    def watch_status(self, new_status: str) -> None:
        for cls in list(self.classes):
            if cls in self.ICONS:
                self.remove_class(cls)
        self.add_class(new_status)
        self._update_content()

    def _update_content(self) -> None:
        icon = self.ICONS.get(self.status, "○")
        self.update(f"{icon} {self._label}" if self._label else icon)


# ========== Status Panel ==========


class StatusPanel(Static):
    """Panel showing system status summary."""

    DEFAULT_CSS = """
    StatusPanel {
        height: auto;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    StatusPanel .status-title { text-style: bold; margin-bottom: 1; }
    StatusPanel .status-row { height: 1; margin: 0; }
    StatusPanel .status-label { width: 20; }
    """

    def __init__(
        self, title: str = "System Status", items: list[StatusItem] | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._items: list[StatusItem] = items or []
        self._status_items: dict[str, tuple[str, str, str]] = {}
        # Convert items to internal format
        for item in self._items:
            self._status_items[item.label] = (item.label, item.value, item.level.value)

    def compose(self) -> ComposeResult:
        if self._title:
            yield Label(f"📊 {self._title}", classes="status-title")
        yield Container(id="status-items")

    def on_mount(self) -> None:
        """Mount initial items."""
        self._refresh_items()

    def set_status(self, key: str, label: str, value: str, status: str = "info") -> None:
        self._status_items[key] = (label, value, status)
        self._refresh_items()

    def add_item(self, item: StatusItem) -> None:
        """Add a status item."""
        self._items.append(item)
        self._status_items[item.label] = (item.label, item.value, item.level.value)
        self._refresh_items()

    def _refresh_items(self) -> None:
        try:
            container = self.query_one("#status-items", Container)
            container.remove_children()
            for _key, (label, value, status) in self._status_items.items():
                row = Horizontal(classes="status-row")
                row.mount(Label(f"{label}:", classes="status-label"))
                row.mount(StatusIndicator(status=status, label=value))
                container.mount(row)
        except Exception:
            pass  # Widget not mounted yet


# ========== Log Viewer ==========


@dataclass
class LogEntry:
    """A log entry."""

    timestamp: float
    level: str
    message: str
    component: str = ""
    metadata: dict[str, Any] | None = None

    def format_timestamp(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3]

    def format_level(self) -> str:
        return self.level.upper()


class LogViewer(Static):
    """Scrollable log viewer with filtering."""

    DEFAULT_CSS = """
    LogViewer {
        height: 100%;
        border: solid $primary;
        background: $background;
    }
    LogViewer .log-header { height: 3; padding: 1; background: $surface; }
    LogViewer .log-content { height: 1fr; padding: 0 1; }
    LogViewer .log-entry { height: 1; }
    LogViewer .log-timestamp { color: $text-muted; width: 12; }
    LogViewer .log-level { width: 8; text-align: center; }
    LogViewer .log-level-debug { color: $text-muted; }
    LogViewer .log-level-info { color: $info; }
    LogViewer .log-level-warning { color: $warning; }
    LogViewer .log-level-error { color: $error; }
    LogViewer .log-level-success { color: $success; }
    LogViewer .log-level-critical { color: $error; text-style: bold; }
    LogViewer .log-message { width: 1fr; }
    LogViewer .log-footer { height: 3; padding: 1; background: $surface; }
    """

    BINDINGS = [
        Binding("c", "clear_logs", "Clear"),
        Binding("f", "toggle_filter", "Filter"),
    ]

    def __init__(self, max_entries: int = 1000, auto_scroll: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries: list[LogEntry] = []
        self._max_entries = max_entries
        self._auto_scroll = auto_scroll
        self._filter_level: str | None = None
        self._filter_text: str = ""

    def compose(self) -> ComposeResult:
        with Container(classes="log-header"):
            yield Label("📋 Logs")
        with ScrollableContainer(classes="log-content", id="log-content"):
            pass
        with Horizontal(classes="log-footer"):
            yield Button("Clear", id="btn-clear")
            yield Input(placeholder="Filter...", id="filter-input")

    def add_entry(self, entry: LogEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        self._render_entry(entry)

    def log(self, message: str, level: str = "info", component: str = "") -> None:
        self.add_entry(
            LogEntry(timestamp=time.time(), level=level, message=message, component=component)
        )

    def log_debug(self, message: str, component: str = "") -> None:
        """Log a debug-level message."""
        self.log(message, level="debug", component=component)

    def log_info(self, message: str, component: str = "") -> None:
        """Log an info-level message."""
        self.log(message, level="info", component=component)

    def log_warning(self, message: str, component: str = "") -> None:
        """Log a warning-level message."""
        self.log(message, level="warning", component=component)

    def log_error(self, message: str, component: str = "") -> None:
        """Log an error-level message."""
        self.log(message, level="error", component=component)

    def log_success(self, message: str, component: str = "") -> None:
        """Log a success-level message."""
        self.log(message, level="success", component=component)

    def log_critical(self, message: str, component: str = "") -> None:
        """Log a critical-level message."""
        self.log(message, level="critical", component=component)

    def scroll_to_end(self) -> None:
        """Scroll the log view to the end (latest entry).

        This is a public API for external callers to ensure
        the most recent log entries are visible.
        """
        try:
            content = self.query_one("#log-content", ScrollableContainer)
            content.scroll_end()
        except Exception:
            pass  # Widget not mounted yet

    def clear(self) -> None:
        """Clear all log entries. Convenience alias for action_clear_logs."""
        self.action_clear_logs()

    def get_entries(self) -> list:
        """Return a copy of all log entries for export."""
        return list(self._entries)

    def export_to_text(self) -> str:
        """Export all log entries to formatted text."""
        lines = []
        for entry in self._entries:
            ts = entry.format_timestamp()
            level = entry.format_level()
            comp = f"[{entry.component}] " if entry.component else ""
            lines.append(f"{ts} [{level}] {comp}{entry.message}")
        return "\n".join(lines)

    def _render_entry(self, entry: LogEntry) -> None:
        if self._filter_level and entry.level != self._filter_level:
            return
        if self._filter_text and self._filter_text.lower() not in entry.message.lower():
            return
        content = self.query_one("#log-content", ScrollableContainer)
        with content.batch():
            row = Horizontal(classes="log-entry")
            row.mount(Label(entry.format_timestamp(), classes="log-timestamp"))
            row.mount(
                Label(f"[{entry.format_level()}]", classes=f"log-level log-level-{entry.level}")
            )
            row.mount(Label(entry.message, classes="log-message"))
            content.mount(row)
        if self._auto_scroll:
            content.scroll_end()

    def action_clear_logs(self) -> None:
        self._entries.clear()
        self.query_one("#log-content", ScrollableContainer).remove_children()

    def action_toggle_filter(self) -> None:
        self.query_one("#filter-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter-input":
            self._filter_text = event.value
            self._refresh_display()

    def _refresh_display(self) -> None:
        content = self.query_one("#log-content", ScrollableContainer)
        content.remove_children()
        for entry in self._entries:
            self._render_entry(entry)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-clear":
            self.action_clear_logs()


# ========== Backend Card ==========


class BackendStatus(Enum):
    """Backend connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"


@dataclass
class BackendInfo:
    """Information about a backend."""

    name: str
    backend_type: str
    status: BackendStatus = BackendStatus.DISCONNECTED
    last_used: float | None = None
    total_executions: int = 0
    avg_latency_ms: float | None = None
    error_message: str | None = None


class BackendCard(Static):
    """Card displaying backend information."""

    DEFAULT_CSS = """
    BackendCard {
        border: solid $surface-light;
        padding: 1;
        margin: 1;
        height: auto;
    }
    BackendCard:hover { border: solid $primary; }
    BackendCard:focus { border: heavy $primary; }
    BackendCard.connected { border: solid $success; }
    BackendCard.disconnected { border: solid $text-muted; }
    BackendCard.error { border: solid $error; }
    BackendCard .backend-name { text-style: bold; }
    BackendCard .backend-type { color: $text-muted; }
    BackendCard .backend-stats { margin-top: 1; color: $text-muted; }
    """

    can_focus = True

    class Selected(Message):
        def __init__(self, backend: BackendInfo) -> None:
            super().__init__()
            self.backend = backend

    def __init__(self, backend: BackendInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self._backend = backend
        self.add_class(backend.status.value)

    def compose(self) -> ComposeResult:
        icons = {
            BackendStatus.CONNECTED: "🟢",
            BackendStatus.DISCONNECTED: "⚪",
            BackendStatus.ERROR: "🔴",
            BackendStatus.CONNECTING: "🟡",
        }
        yield Label(
            f"{icons.get(self._backend.status, '⚪')} {self._backend.name}", classes="backend-name"
        )
        yield Label(f"Type: {self._backend.backend_type}", classes="backend-type")
        if self._backend.total_executions > 0:
            stats = f"Executions: {self._backend.total_executions}"
            if self._backend.avg_latency_ms:
                stats += f" | Avg: {self._backend.avg_latency_ms:.1f}ms"
            yield Label(stats, classes="backend-stats")
        if self._backend.error_message:
            yield Label(f"Error: {self._backend.error_message}", classes="backend-error")

    def on_click(self) -> None:
        self.post_message(self.Selected(self._backend))

    def on_key(self, event) -> None:
        if event.key == "enter":
            self.post_message(self.Selected(self._backend))


# ========== Results Table ==========


class ResultsTable(Static):
    """Table displaying execution results."""

    DEFAULT_CSS = """
    ResultsTable { height: 100%; }
    ResultsTable DataTable { height: 100%; }
    """

    class ResultSelected(Message):
        def __init__(self, result: dict[str, Any]) -> None:
            super().__init__()
            self.result = result

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        table = DataTable(id="results-data-table")
        table.add_columns("ID", "Backend", "Status", "Duration", "Time")
        yield table

    def load_results(self, results: list[dict[str, Any]]) -> None:
        self._results = results
        table = self.query_one("#results-data-table", DataTable)
        table.clear()
        for result in results:
            icon = "✓" if result.get("status") == "success" else "✗"
            ts = datetime.fromtimestamp(result.get("timestamp", time.time())).strftime("%H:%M:%S")
            table.add_row(
                result.get("id", ""),
                result.get("backend", ""),
                f"{icon} {result.get('status', '')}",
                f"{result.get('duration_ms', 0):.1f}ms",
                ts,
            )

    def get_selected_result(self) -> dict[str, Any] | None:
        table = self.query_one("#results-data-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self._results):
            return self._results[table.cursor_row]
        return None


# ========== Execution Timer ==========


class ExecutionTimer(Static):
    """Timer display for execution duration."""

    DEFAULT_CSS = """
    ExecutionTimer { height: 3; padding: 1; border: solid $primary; text-align: center; }
    ExecutionTimer.running { border: solid $accent; }
    ExecutionTimer.completed { border: solid $success; }
    ExecutionTimer.failed { border: solid $error; }
    ExecutionTimer .timer-value { text-style: bold; text-align: center; }
    ExecutionTimer .timer-label { color: $text-muted; text-align: center; }
    """

    elapsed = reactive(0.0)
    running = reactive(False)

    def __init__(self, label: str = "Elapsed Time", **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._start_time: float | None = None
        self._update_timer: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield Label("00:00.000", classes="timer-value", id="timer-value")
        yield Label(self._label, classes="timer-label")

    def start(self) -> None:
        self._start_time = time.time()
        self.running = True
        self.add_class("running")
        self._update_timer = asyncio.create_task(self._update_loop())

    def stop(self) -> None:
        self.running = False
        self.remove_class("running")
        if self._update_timer:
            self._update_timer.cancel()
            self._update_timer = None

    def reset(self) -> None:
        self.stop()
        self.elapsed = 0.0
        self._start_time = None
        self._update_display()

    async def _update_loop(self) -> None:
        while self.running:
            if self._start_time:
                self.elapsed = time.time() - self._start_time
                self._update_display()
            await asyncio.sleep(0.05)

    def _update_display(self) -> None:
        m, s, ms = int(self.elapsed // 60), int(self.elapsed % 60), int((self.elapsed % 1) * 1000)
        self.query_one("#timer-value", Label).update(f"{m:02d}:{s:02d}.{ms:03d}")


# ========== Metric Display ==========


class MetricDisplay(Static):
    """Display for a single metric with optional trend."""

    DEFAULT_CSS = """
    MetricDisplay { height: auto; min-width: 15; padding: 1; border: solid $surface-light; text-align: center; }
    MetricDisplay .metric-value { text-style: bold; text-align: center; }
    MetricDisplay .metric-label { color: $text-muted; text-align: center; }
    MetricDisplay .trend-up { color: $success; }
    MetricDisplay .trend-down { color: $error; }
    MetricDisplay .trend-neutral { color: $text-muted; }
    """

    value = reactive("0")

    def __init__(
        self, label: str, value: str = "0", unit: str = "", trend: float | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._label, self._unit, self._trend = label, unit, trend
        self.value = value

    def compose(self) -> ComposeResult:
        yield Label(f"{self.value}{self._unit}", classes="metric-value", id="metric-value")
        yield Label(self._label, classes="metric-label")
        if self._trend is not None:
            cls = (
                "trend-up"
                if self._trend > 0
                else ("trend-down" if self._trend < 0 else "trend-neutral")
            )
            icon = "↑" if self._trend > 0 else ("↓" if self._trend < 0 else "→")
            yield Label(f"{icon} {abs(self._trend):.1f}%", classes=f"metric-trend {cls}")

    def watch_value(self, new_value: str) -> None:
        try:
            self.query_one("#metric-value", Label).update(f"{new_value}{self._unit}")
        except Exception:
            pass


# ========== Progress Widget ==========


class ExecutionProgress(Static):
    """Progress display for execution with stages."""

    DEFAULT_CSS = """
    ExecutionProgress { height: auto; padding: 1; border: solid $primary; }
    ExecutionProgress .progress-header { height: 1; margin-bottom: 1; }
    ExecutionProgress .progress-stage { color: $text-muted; }
    ExecutionProgress .progress-stats { margin-top: 1; color: $text-muted; }
    """

    progress = reactive(0.0)
    stage = reactive("")

    def __init__(self, title: str = "Execution Progress", **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._start_time: float | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="progress-header"):
            yield Label(self._title)
            yield Label("", classes="progress-stage", id="stage-label")
        yield TextualProgressBar(total=100, id="exec-progress-bar")
        yield Label("", classes="progress-stats", id="stats-label")

    def start(self) -> None:
        self._start_time = time.time()
        self.progress = 0.0
        self.stage = "Starting..."

    def update_progress(self, progress: float, stage: str = "") -> None:
        self.progress = progress
        if stage:
            self.stage = stage
        self.query_one("#exec-progress-bar", TextualProgressBar).update(progress=progress)
        self.query_one("#stage-label", Label).update(f"[{stage}]" if stage else "")
        if self._start_time:
            elapsed = time.time() - self._start_time
            eta = (elapsed / progress * (100 - progress)) if progress > 0 else 0
            stats = f"Elapsed: {elapsed:.1f}s" + (
                f" | ETA: {eta:.1f}s" if progress < 100 and eta > 0 else ""
            )
            self.query_one("#stats-label", Label).update(stats)


# ========== Help Modal ==========


class HelpModal(Static):
    """Modal showing keyboard shortcuts help."""

    DEFAULT_CSS = """
    HelpModal { align: center middle; width: auto; height: auto; max-height: 80%; border: thick $primary; background: $surface; padding: 2; }
    HelpModal .help-title { text-align: center; text-style: bold; margin-bottom: 2; }
    HelpModal .help-section { margin-bottom: 1; text-style: bold; }
    HelpModal .help-shortcut { margin-left: 2; }
    HelpModal .help-close { margin-top: 2; text-align: center; color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        yield Label("⌨️ Keyboard Shortcuts", classes="help-title")
        yield Label("Navigation", classes="help-section")
        yield Label("  [bold]1-5[/bold]  Switch screens", classes="help-shortcut")
        yield Label("  [bold]Tab[/bold]  Next element", classes="help-shortcut")
        yield Label("  [bold]↑/↓[/bold]  Navigate lists", classes="help-shortcut")
        yield Label("  [bold]Enter[/bold]  Select/Confirm", classes="help-shortcut")
        yield Label("Actions", classes="help-section")
        yield Label("  [bold]r[/bold]  Refresh data", classes="help-shortcut")
        yield Label("  [bold]e[/bold]  Execute task", classes="help-shortcut")
        yield Label("  [bold]s[/bold]  Stop execution", classes="help-shortcut")
        yield Label("  [bold]c[/bold]  Clear logs", classes="help-shortcut")
        yield Label("General", classes="help-section")
        yield Label("  [bold]?[/bold]  Toggle this help", classes="help-shortcut")
        yield Label("  [bold]q[/bold]  Quit application", classes="help-shortcut")
        yield Label("  [bold]Esc[/bold]  Close modal/Cancel", classes="help-shortcut")
        yield Label("Press Esc or ? to close", classes="help-close")

    def on_key(self, event) -> None:
        if event.key in ("escape", "question_mark"):
            self.remove()


# ========== Config Widgets ==========


class ConfigInput(Static):
    """Configuration input field with label."""

    DEFAULT_CSS = """
    ConfigInput { height: 3; margin-bottom: 1; }
    ConfigInput .config-label { margin-right: 1; }
    ConfigInput Horizontal { height: 3; }
    """

    class Changed(Message):
        def __init__(self, key: str, value: str) -> None:
            super().__init__()
            self.key, self.value = key, value

    def __init__(
        self, key: str, label: str, value: str = "", placeholder: str = "", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._key, self._label, self._value, self._placeholder = key, label, value, placeholder

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(f"{self._label}:", classes="config-label")
            yield Input(value=self._value, placeholder=self._placeholder, id=f"input-{self._key}")

    def on_input_changed(self, event: Input.Changed) -> None:
        self.post_message(self.Changed(self._key, event.value))


class ConfigToggle(Static):
    """Configuration toggle with label."""

    DEFAULT_CSS = """
    ConfigToggle { height: 3; margin-bottom: 1; }
    ConfigToggle .toggle-label { margin-right: 1; }
    ConfigToggle Horizontal { height: 3; }
    """

    class Changed(Message):
        def __init__(self, key: str, value: bool) -> None:
            super().__init__()
            self.key, self.value = key, value

    def __init__(self, key: str, label: str, value: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._key, self._label, self._value = key, label, value

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(f"{self._label}:", classes="toggle-label")
            yield Switch(value=self._value, id=f"switch-{self._key}")

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self.post_message(self.Changed(self._key, event.value))


# ========== Execution Card ==========


class ExecutionCard(Static):
    """Card showing a recent execution."""

    DEFAULT_CSS = """
    ExecutionCard { border: solid $primary; padding: 1; margin: 1; height: auto; }
    ExecutionCard.success { border: solid $success; }
    ExecutionCard.failed { border: solid $error; }
    ExecutionCard .exec-title { text-style: bold; }
    ExecutionCard .exec-details { color: $text-muted; }
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
        self._id, self._backend, self._status, self._duration, self._timestamp = (
            execution_id,
            backend,
            status,
            duration_ms,
            timestamp,
        )
        self.add_class(status)

    def compose(self) -> ComposeResult:
        icon = "✓" if self._status == "success" else "✗"
        yield Label(f"{icon} Execution {self._id}", classes="exec-title")
        ts = datetime.fromtimestamp(self._timestamp).strftime("%H:%M:%S")
        yield Label(
            f"Backend: {self._backend} | Duration: {self._duration:.1f}ms | {ts}",
            classes="exec-details",
        )


# ========== ProgressBar Wrapper ==========


class ProgressBar(Static):
    """Custom progress bar wrapper with label support."""

    DEFAULT_CSS = """
    ProgressBar { height: auto; padding: 1; }
    ProgressBar .progress-label { margin-bottom: 1; }
    ProgressBar .progress-value { color: $text-muted; margin-top: 1; }
    """

    progress = reactive(0.0)

    def __init__(
        self, label: str = "", total: float = 100.0, show_percentage: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._label, self._total, self._show_percentage = label, total, show_percentage

    def compose(self) -> ComposeResult:
        if self._label:
            yield Label(self._label, classes="progress-label")
        yield TextualProgressBar(total=self._total, id="inner-progress")
        if self._show_percentage:
            yield Label("0%", classes="progress-value", id="progress-value")

    def update_progress(self, value: float) -> None:
        self.progress = value
        self.query_one("#inner-progress", TextualProgressBar).update(progress=value)
        if self._show_percentage:
            self.query_one("#progress-value", Label).update(f"{(value / self._total) * 100:.0f}%")
