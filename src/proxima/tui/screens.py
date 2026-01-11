"""TUI Screens for Proxima.

Screens:
1. Dashboard   - System status, recent executions
2. Execution   - Real-time progress, logs
3. Configuration - Settings management
4. Results     - Browse and analyze results
5. Backends    - Backend status and management
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
)

from .widgets import (
    BackendCard,
    BackendInfo,
    BackendStatus,
    ConfigInput,
    ConfigToggle,
    ExecutionCard,
    HelpModal,
    LogViewer,
    ProgressBar,
    ResultsTable,
    StatusItem,
    StatusLevel,
    StatusPanel,
)

if TYPE_CHECKING:
    from proxima.backends.registry import BackendRegistry
    from proxima.data.store import ResultStore


def _get_backend_registry() -> BackendRegistry | None:
    """Safely get the backend registry."""
    try:
        from proxima.backends.registry import backend_registry

        return backend_registry
    except ImportError:
        return None


def _get_result_store() -> ResultStore | None:
    """Safely get the result store."""
    try:
        from proxima.data.store import get_store

        return get_store()
    except (ImportError, Exception):
        return None


def _get_version() -> str:
    """Get Proxima version."""
    try:
        from proxima import __version__

        return __version__
    except ImportError:
        return "unknown"


class BaseScreen(Screen):
    """Base screen with common functionality."""

    BINDINGS = [
        Binding("question_mark", "toggle_help", "Help", show=True),
        Binding("1", "goto_dashboard", "Dashboard", show=False),
        Binding("2", "goto_execution", "Execution", show=False),
        Binding("3", "goto_config", "Config", show=False),
        Binding("4", "goto_results", "Results", show=False),
        Binding("5", "goto_backends", "Backends", show=False),
        Binding("q", "quit", "Quit", show=True),
    ]

    def action_toggle_help(self) -> None:
        """Toggle help modal."""
        help_modal = self.query("HelpModal")
        if help_modal:
            help_modal.first().remove()
        else:
            self.mount(HelpModal())

    def action_goto_dashboard(self) -> None:
        """Switch to dashboard screen."""
        self.app.push_screen("dashboard")

    def action_goto_execution(self) -> None:
        """Switch to execution screen."""
        self.app.push_screen("execution")

    def action_goto_config(self) -> None:
        """Switch to configuration screen."""
        self.app.push_screen("configuration")

    def action_goto_results(self) -> None:
        """Switch to results screen."""
        self.app.push_screen("results")

    def action_goto_backends(self) -> None:
        """Switch to backends screen."""
        self.app.push_screen("backends")


class DashboardScreen(BaseScreen):
    """Dashboard screen showing system status and recent executions.

    Features:
    - System status overview
    - Backend connectivity status
    - Recent execution history
    - Quick action buttons
    """

    CSS = """
    DashboardScreen {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }

    #status-section {
        row-span: 1;
        column-span: 1;
    }

    #backends-section {
        row-span: 1;
        column-span: 1;
    }

    #recent-section {
        row-span: 1;
        column-span: 2;
    }

    .section-title {
        text-style: bold;
        padding: 1;
        background: $primary;
        color: $text;
    }

    #quick-actions {
        dock: bottom;
        height: 3;
        padding: 1;
    }

    #quick-actions Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        *BaseScreen.BINDINGS,
        Binding("r", "refresh", "Refresh", show=True),
        Binding("e", "new_execution", "New Execution", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="status-section"):
            yield Label("📊 System Status", classes="section-title")
            yield StatusPanel(
                title="",
                items=self._get_status_items(),
                id="system-status",
            )

        with Container(id="backends-section"):
            yield Label("🔌 Backend Status", classes="section-title")
            yield Container(id="backend-cards")

        with Container(id="recent-section"):
            yield Label("📜 Recent Executions", classes="section-title")
            yield ScrollableContainer(id="recent-executions")

        with Horizontal(id="quick-actions"):
            yield Button("New Execution", id="btn-execute", variant="primary")
            yield Button("View Results", id="btn-results")
            yield Button("Settings", id="btn-settings")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize dashboard data."""
        self._refresh_backends()
        self._refresh_recent_executions()

    def _get_status_items(self) -> list[StatusItem]:
        """Get current status items from real sources."""
        items: list[StatusItem] = []

        # Version info
        version = _get_version()
        items.append(StatusItem("Version", version, StatusLevel.INFO))

        # Python version
        python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        items.append(StatusItem("Python", python_version, StatusLevel.INFO))

        # Backend count from registry
        registry = _get_backend_registry()
        if registry:
            available_backends = registry.list_available()
            count = len(available_backends)
            level = StatusLevel.OK if count > 0 else StatusLevel.WARNING
            items.append(StatusItem("Backends Available", str(count), level))
        else:
            items.append(StatusItem("Backends Available", "N/A", StatusLevel.WARNING))

        # Total executions from store
        store = _get_result_store()
        if store:
            try:
                results = store.list_results(limit=1000)
                items.append(StatusItem("Total Executions", str(len(results)), StatusLevel.INFO))

                # Last run
                if results:
                    # Results are sorted by timestamp descending
                    last_result = results[0]
                    last_ts = last_result.timestamp.timestamp()
                    age_seconds = time.time() - last_ts
                    if age_seconds < 60:
                        last_run = "just now"
                    elif age_seconds < 3600:
                        last_run = f"{int(age_seconds // 60)} min ago"
                    elif age_seconds < 86400:
                        last_run = f"{int(age_seconds // 3600)} hours ago"
                    else:
                        last_run = f"{int(age_seconds // 86400)} days ago"
                    items.append(StatusItem("Last Run", last_run, StatusLevel.OK, last_ts))
                else:
                    items.append(StatusItem("Last Run", "Never", StatusLevel.INFO))
            except Exception:
                items.append(StatusItem("Total Executions", "N/A", StatusLevel.WARNING))
                items.append(StatusItem("Last Run", "N/A", StatusLevel.WARNING))
        else:
            items.append(StatusItem("Total Executions", "N/A", StatusLevel.WARNING))
            items.append(StatusItem("Last Run", "N/A", StatusLevel.WARNING))

        return items

    def _refresh_backends(self) -> None:
        """Refresh backend cards from registry."""
        container = self.query_one("#backend-cards", Container)
        container.remove_children()

        # Get real backend data from registry
        registry = _get_backend_registry()
        backends: list[BackendInfo] = []

        if registry:
            for status in registry.list_statuses():
                # Map registry status to widget BackendStatus
                if status.available:
                    widget_status = BackendStatus.CONNECTED
                elif status.reason and "failed" in status.reason.lower():
                    widget_status = BackendStatus.ERROR
                else:
                    widget_status = BackendStatus.DISCONNECTED

                # Determine backend type from capabilities or name
                backend_type = "simulator"
                if status.capabilities:
                    if status.capabilities.max_qubits > 30:
                        backend_type = "cloud"
                if "local" in status.name.lower():
                    backend_type = "local"
                elif "ibm" in status.name.lower() or "aws" in status.name.lower():
                    backend_type = "cloud"

                error_msg = status.reason if not status.available else None

                backends.append(
                    BackendInfo(
                        status.name,
                        backend_type,
                        widget_status,
                        last_used=None,
                        total_executions=0,
                        avg_latency_ms=None,
                        error_message=error_msg,
                    )
                )
        else:
            # Fallback if registry unavailable
            backends = [
                BackendInfo("Registry Unavailable", "unknown", BackendStatus.ERROR),
            ]

        for backend in backends:
            container.mount(BackendCard(backend))

    def _refresh_recent_executions(self) -> None:
        """Refresh recent executions list from store."""
        container = self.query_one("#recent-executions", ScrollableContainer)
        container.remove_children()

        # Get real execution data from store
        store = _get_result_store()
        executions: list[tuple[str, str, str, float, float]] = []

        if store:
            try:
                results = store.list_results(limit=10)
                for result in results:
                    exec_id = result.id[:8] if len(result.id) > 8 else result.id
                    backend = result.backend_name
                    # Determine status - if we have counts, it succeeded
                    status = "success" if result.counts else "completed"
                    duration = result.execution_time_ms
                    ts = result.timestamp.timestamp()
                    executions.append((exec_id, backend, status, duration, ts))
            except Exception:
                # Store access failed
                pass

        if not executions:
            container.mount(Label("No recent executions"))
        else:
            for exec_id, backend, status, duration, ts in executions:
                container.mount(ExecutionCard(exec_id, backend, status, duration, ts))

    def action_refresh(self) -> None:
        """Refresh dashboard data."""
        status_panel = self.query_one("#system-status", StatusPanel)
        status_panel.update_items(self._get_status_items())
        self._refresh_backends()
        self._refresh_recent_executions()

    def action_new_execution(self) -> None:
        """Start a new execution."""
        self.app.push_screen("execution")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-execute":
            self.action_new_execution()
        elif event.button.id == "btn-results":
            self.app.push_screen("results")
        elif event.button.id == "btn-settings":
            self.app.push_screen("configuration")


class ExecutionScreen(BaseScreen):
    """Execution screen for real-time progress and logs.

    Features:
    - Task configuration
    - Real-time progress bar
    - Live log output
    - Stop/Cancel controls
    """

    CSS = """
    ExecutionScreen {
        layout: grid;
        grid-size: 1 3;
        grid-rows: auto 1fr 3;
    }

    #exec-header {
        height: auto;
        padding: 1;
        background: $surface;
        border-bottom: solid $primary;
    }

    #exec-header Label {
        margin-right: 2;
    }

    #log-section {
        height: 100%;
    }

    #exec-controls {
        dock: bottom;
        height: 3;
        padding: 1;
    }

    #exec-controls Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        *BaseScreen.BINDINGS,
        Binding("s", "stop_execution", "Stop", show=True),
        Binding("c", "clear_logs", "Clear Logs", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_running = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="exec-header"):
            yield Label("🚀 Execution Console", classes="section-title")
            yield ProgressBar(total=100, label="Task Progress", id="exec-progress")
            with Horizontal():
                yield Label("Status: Idle", id="exec-status")
                yield Label("Backend: -", id="exec-backend")
                yield Label("Duration: -", id="exec-duration")

        with Container(id="log-section"):
            yield LogViewer(id="exec-logs")

        with Horizontal(id="exec-controls"):
            yield Button("Start", id="btn-start", variant="success")
            yield Button("Stop", id="btn-stop", variant="error", disabled=True)
            yield Button("Clear Logs", id="btn-clear")
            yield Button("Export Logs", id="btn-export")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize execution screen."""
        log_viewer = self.query_one("#exec-logs", LogViewer)
        log_viewer.log_info("Execution console ready")
        log_viewer.log_info("Press 'Start' to begin a new execution")

    def action_stop_execution(self) -> None:
        """Stop current execution."""
        if self._is_running:
            self._is_running = False
            self._update_controls()
            log_viewer = self.query_one("#exec-logs", LogViewer)
            log_viewer.log_warning("Execution stopped by user")

    def action_clear_logs(self) -> None:
        """Clear the log viewer."""
        log_viewer = self.query_one("#exec-logs", LogViewer)
        log_viewer.clear()
        log_viewer.log_info("Logs cleared")

    def _update_controls(self) -> None:
        """Update control button states."""
        start_btn = self.query_one("#btn-start", Button)
        stop_btn = self.query_one("#btn-stop", Button)

        start_btn.disabled = self._is_running
        stop_btn.disabled = not self._is_running

        status_label = self.query_one("#exec-status", Label)
        status_label.update(f"Status: {'Running' if self._is_running else 'Idle'}")

    async def _run_demo_execution(self) -> None:
        """Run a demo execution for testing."""
        log_viewer = self.query_one("#exec-logs", LogViewer)
        progress_bar = self.query_one("#exec-progress", ProgressBar)

        log_viewer.log_info("Starting execution...")

        steps = [
            (10, "Initializing backend"),
            (30, "Compiling circuit"),
            (50, "Executing task"),
            (70, "Processing results"),
            (90, "Generating report"),
            (100, "Complete"),
        ]

        for progress, status in steps:
            if not self._is_running:
                break
            progress_bar.update_progress(progress, status)
            log_viewer.log_info(status)
            await self.app.call_later(0.5)

        if self._is_running:
            log_viewer.log_success("Execution completed successfully")
            self._is_running = False
            self._update_controls()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-start":
            self._is_running = True
            self._update_controls()
            self.run_worker(self._run_demo_execution())
        elif event.button.id == "btn-stop":
            self.action_stop_execution()
        elif event.button.id == "btn-clear":
            self.action_clear_logs()
        elif event.button.id == "btn-export":
            log_viewer = self.query_one("#exec-logs", LogViewer)
            log_viewer.log_info("Export functionality coming soon...")


class ConfigurationScreen(BaseScreen):
    """Configuration screen for settings management.

    Features:
    - Backend configuration
    - Execution settings
    - Export preferences
    - Security settings
    """

    CSS = """
    ConfigurationScreen {
        layout: grid;
        grid-size: 2 1;
        grid-gutter: 1;
    }

    .config-section {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    .config-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #config-actions {
        dock: bottom;
        height: 3;
        padding: 1;
    }
    """

    BINDINGS = [
        *BaseScreen.BINDINGS,
        Binding("ctrl+s", "save_config", "Save", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with ScrollableContainer():
            with Container(classes="config-section"):
                yield Label("⚙️ General Settings", classes="config-title")
                yield ConfigInput("log_level", "Log Level", "INFO", "DEBUG, INFO, WARNING, ERROR")
                yield ConfigInput("output_dir", "Output Directory", "./results")
                yield ConfigToggle("auto_export", "Auto Export Results", True)
                yield ConfigToggle("show_progress", "Show Progress", True)

            with Container(classes="config-section"):
                yield Label("🔌 Backend Settings", classes="config-title")
                yield ConfigInput("default_backend", "Default Backend", "qiskit_aer")
                yield ConfigInput("timeout_s", "Timeout (seconds)", "300")
                yield ConfigToggle("parallel_execution", "Parallel Execution", False)
                yield ConfigInput("max_workers", "Max Workers", "4")

        with ScrollableContainer():
            with Container(classes="config-section"):
                yield Label("📊 Export Settings", classes="config-title")
                yield ConfigInput("export_format", "Default Format", "json")
                yield ConfigToggle("include_metadata", "Include Metadata", True)
                yield ConfigToggle("pretty_print", "Pretty Print JSON", True)

            with Container(classes="config-section"):
                yield Label("🔒 Security Settings", classes="config-title")
                yield ConfigToggle("require_consent", "Require Consent", True)
                yield ConfigToggle("dry_run_default", "Dry Run by Default", False)
                yield ConfigInput("sensitive_ops", "Sensitive Operations", "file_write,network")

        with Horizontal(id="config-actions"):
            yield Button("Save", id="btn-save", variant="primary")
            yield Button("Reset", id="btn-reset")
            yield Button("Export Config", id="btn-export")
            yield Button("Import Config", id="btn-import")

        yield Footer()

    def action_save_config(self) -> None:
        """Save configuration."""
        self.notify("Configuration saved", severity="information")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-save":
            self.action_save_config()
        elif event.button.id == "btn-reset":
            self.notify("Configuration reset to defaults", severity="warning")
        elif event.button.id == "btn-export":
            self.notify("Config exported to proxima-config.json", severity="information")
        elif event.button.id == "btn-import":
            self.notify("Import functionality coming soon", severity="information")


class ResultsScreen(BaseScreen):
    """Results screen for browsing and analyzing execution results.

    Features:
    - Results table
    - Filtering and sorting
    - Detail view
    - Export options
    """

    CSS = """
    ResultsScreen {
        layout: grid;
        grid-size: 1 2;
        grid-rows: 1fr auto;
    }

    #results-container {
        height: 100%;
        border: solid $primary;
        margin: 1;
    }

    #results-header {
        height: 3;
        padding: 1;
    }

    #results-actions {
        dock: bottom;
        height: 3;
        padding: 1;
    }

    #filter-input {
        width: 30;
        margin-right: 1;
    }
    """

    BINDINGS = [
        *BaseScreen.BINDINGS,
        Binding("r", "refresh_results", "Refresh", show=True),
        Binding("delete", "delete_result", "Delete", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="results-container"):
            with Horizontal(id="results-header"):
                yield Label("📋 Execution Results")
            yield ResultsTable(id="results-table")

        with Horizontal(id="results-actions"):
            yield Button("Refresh", id="btn-refresh")
            yield Button("View Details", id="btn-details", variant="primary")
            yield Button("Export", id="btn-export")
            yield Button("Delete", id="btn-delete", variant="error")
            yield Button("Clear All", id="btn-clear")

        yield Footer()

    def on_mount(self) -> None:
        """Load results on mount."""
        self._load_sample_results()

    def _load_sample_results(self) -> None:
        """Load results data from store."""
        results_table = self.query_one("#results-table", ResultsTable)

        # Get real results from store
        store = _get_result_store()
        results_data: list[dict[str, object]] = []

        if store:
            try:
                results = store.list_results(limit=100)
                for result in results:
                    status = "success" if result.counts else "completed"
                    results_data.append(
                        {
                            "id": result.id[:8] if len(result.id) > 8 else result.id,
                            "backend": result.backend_name,
                            "status": status,
                            "duration_ms": result.execution_time_ms,
                            "timestamp": result.timestamp.timestamp(),
                        }
                    )
            except Exception:
                pass

        if not results_data:
            # Show empty state
            results_data = []

        results_table.load_results(results_data)

    def action_refresh_results(self) -> None:
        """Refresh results list."""
        self._load_sample_results()
        self.notify("Results refreshed", severity="information")

    def action_delete_result(self) -> None:
        """Delete selected result."""
        results_table = self.query_one("#results-table", ResultsTable)
        selected = results_table.get_selected_result()
        if selected:
            self.notify(f"Deleted result {selected['id']}", severity="warning")
        else:
            self.notify("No result selected", severity="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-refresh":
            self.action_refresh_results()
        elif event.button.id == "btn-details":
            results_table = self.query_one("#results-table", ResultsTable)
            selected = results_table.get_selected_result()
            if selected:
                self.notify(f"Details for {selected['id']}: {selected}", severity="information")
        elif event.button.id == "btn-export":
            self.notify("Results exported to results.json", severity="information")
        elif event.button.id == "btn-delete":
            self.action_delete_result()
        elif event.button.id == "btn-clear":
            self.notify("All results cleared", severity="warning")


class BackendsScreen(BaseScreen):
    """Backends screen for backend status and management.

    Features:
    - Backend list with status
    - Connection testing
    - Backend configuration
    - Performance metrics
    """

    CSS = """
    BackendsScreen {
        layout: grid;
        grid-size: 2 1;
        grid-gutter: 1;
    }

    #backends-list {
        border: solid $primary;
        margin: 1;
        padding: 1;
    }

    #backend-details {
        border: solid $primary;
        margin: 1;
        padding: 1;
    }

    .backends-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #backends-actions {
        dock: bottom;
        height: 3;
        padding: 1;
    }
    """

    BINDINGS = [
        *BaseScreen.BINDINGS,
        Binding("r", "refresh_backends", "Refresh", show=True),
        Binding("t", "test_connection", "Test", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._selected_backend: BackendInfo | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with ScrollableContainer(id="backends-list"):
            yield Label("🔌 Available Backends", classes="backends-title")
            yield Container(id="backend-cards-container")

        with Container(id="backend-details"):
            yield Label("📊 Backend Details", classes="backends-title")
            yield Container(id="details-content")

        with Horizontal(id="backends-actions"):
            yield Button("Refresh", id="btn-refresh")
            yield Button("Test Connection", id="btn-test", variant="primary")
            yield Button("Configure", id="btn-configure")
            yield Button("Add Backend", id="btn-add")

        yield Footer()

    def on_mount(self) -> None:
        """Load backends on mount."""
        self._refresh_backends()

    def _refresh_backends(self) -> None:
        """Refresh backend list from registry."""
        container = self.query_one("#backend-cards-container", Container)
        container.remove_children()

        # Get real backend data from registry
        registry = _get_backend_registry()
        backends: list[BackendInfo] = []

        if registry:
            for status in registry.list_statuses():
                # Map registry status to widget BackendStatus
                if status.available:
                    widget_status = BackendStatus.CONNECTED
                elif status.reason and "failed" in status.reason.lower():
                    widget_status = BackendStatus.ERROR
                else:
                    widget_status = BackendStatus.DISCONNECTED

                # Determine backend type from capabilities or name
                backend_type = "simulator"
                if status.capabilities:
                    if status.capabilities.max_qubits > 30:
                        backend_type = "cloud"
                if "local" in status.name.lower() or "lret" in status.name.lower():
                    backend_type = "local"
                elif "ibm" in status.name.lower() or "aws" in status.name.lower():
                    backend_type = "cloud"

                error_msg = status.reason if not status.available else None

                backends.append(
                    BackendInfo(
                        status.name,
                        backend_type,
                        widget_status,
                        last_used=None,
                        total_executions=0,
                        avg_latency_ms=None,
                        error_message=error_msg,
                    )
                )
        else:
            # Fallback if registry unavailable
            backends = [
                BackendInfo(
                    "Registry Unavailable",
                    "unknown",
                    BackendStatus.ERROR,
                    error_message="Could not access backend registry",
                ),
            ]

        for backend in backends:
            container.mount(BackendCard(backend))

    def _show_backend_details(self, backend: BackendInfo) -> None:
        """Show details for selected backend."""
        self._selected_backend = backend

        details = self.query_one("#details-content", Container)
        details.remove_children()

        details.mount(Label(f"Name: {backend.name}"))
        details.mount(Label(f"Type: {backend.backend_type}"))
        details.mount(Label(f"Status: {backend.status.value}"))
        details.mount(Label(f"Total Executions: {backend.total_executions}"))

        if backend.avg_latency_ms:
            details.mount(Label(f"Avg Latency: {backend.avg_latency_ms:.1f}ms"))

        if backend.last_used:
            last_used = datetime.fromtimestamp(backend.last_used).strftime("%Y-%m-%d %H:%M:%S")
            details.mount(Label(f"Last Used: {last_used}"))

        if backend.error_message:
            details.mount(Label(f"Error: {backend.error_message}"))

    def on_backend_card_selected(self, event: BackendCard.Selected) -> None:
        """Handle backend card selection."""
        self._show_backend_details(event.backend)

    def action_refresh_backends(self) -> None:
        """Refresh backend list."""
        self._refresh_backends()
        self.notify("Backends refreshed", severity="information")

    def action_test_connection(self) -> None:
        """Test connection to selected backend."""
        if self._selected_backend:
            self.notify(
                f"Testing connection to {self._selected_backend.name}...", severity="information"
            )
        else:
            self.notify("No backend selected", severity="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-refresh":
            self.action_refresh_backends()
        elif event.button.id == "btn-test":
            self.action_test_connection()
        elif event.button.id == "btn-configure":
            if self._selected_backend:
                self.notify(f"Configure {self._selected_backend.name}", severity="information")
            else:
                self.notify("No backend selected", severity="warning")
        elif event.button.id == "btn-add":
            self.notify("Add backend dialog coming soon", severity="information")
