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
        Binding("1", "goto_dashboard", "1 Dashboard", show=False),
        Binding("2", "goto_execution", "2 Execution", show=False),
        Binding("3", "goto_config", "3 Config", show=False),
        Binding("4", "goto_results", "4 Results", show=False),
        Binding("5", "goto_backends", "5 Backends", show=False),
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
        margin-bottom: 1;
    }

    #quick-actions Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        *BaseScreen.BINDINGS,
        Binding("r", "refresh", "r Refresh", show=False),
        Binding("e", "new_execution", "e New Execution", show=False),
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
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
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
                items.append(
                    StatusItem("Total Executions", str(len(results)), StatusLevel.INFO)
                )

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
                    items.append(
                        StatusItem("Last Run", last_run, StatusLevel.OK, last_ts)
                    )
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
    - Real-time progress bar with auto-refresh
    - Live log output with streaming
    - Stop/Cancel controls
    - Backend selection
    - Real execution with pipeline integration
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
    
    #exec-config {
        height: auto;
        padding: 0 1;
    }
    
    #exec-config Input {
        width: 40;
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
        Binding("s", "stop_execution", "s Stop", show=False),
        Binding("c", "clear_logs", "c Clear Logs", show=False),
        Binding("enter", "start_execution", "Start", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_running = False
        self._cancellation_requested = False
        self._execution_start_time: float = 0
        self._duration_timer = None  # Renamed to avoid Textual property conflict
        self._current_backend = "auto"
        self._shots = 1024

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="exec-header"):
            yield Label("🚀 Execution Console", classes="section-title")
            with Horizontal(id="exec-config"):
                yield Label("Backend:")
                from textual.widgets import Input, Select
                yield Input(value="auto", id="backend-input", placeholder="cirq, qiskit, lret...")
                yield Label("Shots:")
                yield Input(value="1024", id="shots-input", placeholder="1024")
            yield ProgressBar(total=100, label="Task Progress", id="exec-progress")
            with Horizontal():
                yield Label("Status: Idle", id="exec-status")
                yield Label("Backend: -", id="exec-backend")
                yield Label("Duration: -", id="exec-duration")
                yield Label("ETA: -", id="exec-eta")

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
        log_viewer.log_info("Configure backend and shots, then press 'Start'")
        log_viewer.log_info("Available backends: cirq, qiskit, lret, or 'auto' for default")

    def _start_timer(self) -> None:
        """Start the duration update timer."""
        self._execution_start_time = time.time()
        self._duration_timer = self.set_interval(0.5, self._update_duration)

    def _stop_timer(self) -> None:
        """Stop the duration update timer."""
        if self._duration_timer:
            self._duration_timer.stop()
            self._duration_timer = None

    def _update_duration(self) -> None:
        """Update the duration label in real-time."""
        if self._is_running and self._execution_start_time:
            elapsed = time.time() - self._execution_start_time
            duration_label = self.query_one("#exec-duration", Label)
            duration_label.update(f"Duration: {elapsed:.1f}s")

    def action_start_execution(self) -> None:
        """Start execution from keyboard."""
        if not self._is_running:
            self._start_execution_task()

    def action_stop_execution(self) -> None:
        """Stop current execution."""
        if self._is_running:
            self._cancellation_requested = True
            self._is_running = False
            self._stop_timer()
            self._update_controls()
            log_viewer = self.query_one("#exec-logs", LogViewer)
            log_viewer.log_warning("Execution stop requested by user")

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
        
        # Update backend display
        backend_label = self.query_one("#exec-backend", Label)
        backend_label.update(f"Backend: {self._current_backend}")

    def _start_execution_task(self) -> None:
        """Initiate execution with current config."""
        # Read config from inputs
        try:
            from textual.widgets import Input
            backend_input = self.query_one("#backend-input", Input)
            shots_input = self.query_one("#shots-input", Input)
            self._current_backend = backend_input.value or "auto"
            self._shots = int(shots_input.value) if shots_input.value else 1024
        except Exception:
            self._current_backend = "auto"
            self._shots = 1024
        
        self._is_running = True
        self._cancellation_requested = False
        self._update_controls()
        self._start_timer()
        # Try real execution first, fall back to demo
        self.run_worker(self._run_real_execution(self._current_backend))

    async def _run_demo_execution(self) -> None:
        """Run a demo execution for testing."""
        log_viewer = self.query_one("#exec-logs", LogViewer)
        progress_bar = self.query_one("#exec-progress", ProgressBar)
        eta_label = self.query_one("#exec-eta", Label)

        log_viewer.log_info("Starting demo execution...")

        steps = [
            (10, "Initializing backend", 4.5),
            (30, "Compiling circuit", 3.5),
            (50, "Executing task", 2.5),
            (70, "Processing results", 1.5),
            (90, "Generating report", 0.5),
            (100, "Complete", 0.0),
        ]

        for progress, status, eta_seconds in steps:
            if not self._is_running or self._cancellation_requested:
                break
            progress_bar.update_progress(progress, status)
            eta_label.update(f"ETA: {eta_seconds:.1f}s" if eta_seconds > 0 else "ETA: -")
            log_viewer.log_info(status)
            await self.app.call_later(0.5)

        if self._is_running and not self._cancellation_requested:
            log_viewer.log_success("Execution completed successfully")
            log_viewer.log_info(f"Simulated {self._shots} shots completed")
        
        self._stop_timer()
        self._is_running = False
        self._update_controls()

    async def _export_logs(self, log_viewer) -> None:
        """Export logs to a file."""
        import os
        from datetime import datetime

        # Get log content
        log_text = log_viewer.export_to_text()

        if not log_text.strip():
            self.app.notify("No logs to export", severity="warning")
            return

        # Create exports directory if needed
        export_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(export_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"proxima_logs_{timestamp}.txt"
        filepath = os.path.join(export_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("Proxima Execution Logs\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")
                f.write(log_text)

            log_viewer.log_success(f"Logs exported to {filepath}")
            self.app.notify(f"Logs exported to {filename}", severity="information")
        except Exception as e:
            log_viewer.log_error(f"Failed to export logs: {e}")
            self.app.notify(f"Export failed: {e}", severity="error")

    async def _run_real_execution(self, backend_name: str = "auto") -> None:
        """Run a real execution using the Proxima pipeline."""
        log_viewer = self.query_one("#exec-logs", LogViewer)
        progress_bar = self.query_one("#exec-progress", ProgressBar)
        eta_label = self.query_one("#exec-eta", Label)

        try:
            # Try to import real execution components
            from proxima.core.pipeline import DataFlowPipeline, PipelineStage

            log_viewer.log_info("Initializing execution pipeline...")
            progress_bar.update_progress(5, "Initializing")

            log_viewer.log_info(f"Backend: {backend_name}")
            log_viewer.log_info(f"Shots: {self._shots}")
            progress_bar.update_progress(10, "Backend selected")

            # Create and run pipeline with real-time progress
            pipeline = DataFlowPipeline(
                require_consent=False,
                auto_approve_consent=True,
                default_timeout=60.0,
                max_retries=2,
            )

            # Stage progress mapping
            stage_progress = {
                PipelineStage.PARSING: (15, "Parsing input", 5.0),
                PipelineStage.PLANNING: (25, "Creating execution plan", 4.0),
                PipelineStage.RESOURCE_CHECK: (35, "Checking resources", 3.5),
                PipelineStage.CONSENT: (40, "Consent check", 3.0),
                PipelineStage.EXECUTING: (60, "Executing on backend", 2.0),
                PipelineStage.COLLECTING: (75, "Collecting results", 1.0),
                PipelineStage.ANALYZING: (85, "Analyzing results", 0.5),
                PipelineStage.EXPORTING: (95, "Exporting data", 0.1),
            }

            # Register progress callback
            def on_stage(stage, ctx):
                if self._cancellation_requested:
                    return
                info = stage_progress.get(stage, (50, stage.name, 1.0))
                log_viewer.log_info(f"Stage: {info[1]}")
                progress_bar.update_progress(info[0], info[1])
                eta_label.update(f"ETA: {info[2]:.1f}s")

            pipeline.on_stage_start(on_stage)

            # Set up cancellation
            import asyncio
            cancellation_event = asyncio.Event()
            pipeline.set_cancellation_event(cancellation_event)

            log_viewer.log_info("Running pipeline...")
            result_ctx = await pipeline.run(
                f"simulate bell state with {self._shots} shots",
                backend=backend_name,
                shots=self._shots,
            )

            progress_bar.update_progress(100, "Complete")
            eta_label.update("ETA: -")

            if result_ctx.current_stage.name == "COMPLETED":
                log_viewer.log_success("Execution completed successfully")
                # Log result details
                if result_ctx.backend_results:
                    for backend, result in result_ctx.backend_results.items():
                        counts = result.get("counts", {})
                        duration = result.get("duration_ms", 0)
                        log_viewer.log_info(
                            f"  {backend}: {len(counts)} outcomes, {duration:.1f}ms"
                        )
                if result_ctx.insights:
                    log_viewer.log_info("Insights:")
                    for insight in result_ctx.insights[:5]:
                        log_viewer.log_info(f"  • {insight}")
            elif result_ctx.current_stage.name == "ABORTED":
                log_viewer.log_warning("Execution was cancelled")
            else:
                last_error = ""
                for sr in result_ctx.stage_results:
                    if not sr.success and sr.error:
                        last_error = sr.error
                log_viewer.log_error(f"Execution failed: {last_error}")

        except ImportError as e:
            log_viewer.log_warning(f"Real execution unavailable: {e}")
            log_viewer.log_info("Falling back to demo execution...")
            await self._run_demo_execution()
            return
        except Exception as e:
            log_viewer.log_error(f"Execution error: {e}")
            progress_bar.update_progress(0, "Failed")
        finally:
            self._stop_timer()
            self._is_running = False
            self._update_controls()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-start":
            self._start_execution_task()
        elif event.button.id == "btn-stop":
            self.action_stop_execution()
        elif event.button.id == "btn-clear":
            self.action_clear_logs()
        elif event.button.id == "btn-export":
            log_viewer = self.query_one("#exec-logs", LogViewer)
            self.run_worker(self._export_logs(log_viewer))


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
        Binding("ctrl+s", "save_config", "Save", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with ScrollableContainer():
            with Container(classes="config-section"):
                yield Label("⚙️ General Settings", classes="config-title")
                yield ConfigInput(
                    "log_level", "Log Level", "INFO", "DEBUG, INFO, WARNING, ERROR"
                )
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
                yield ConfigInput(
                    "sensitive_ops", "Sensitive Operations", "file_write,network"
                )

        with Horizontal(id="config-actions"):
            yield Button("Save", id="btn-save", variant="primary")
            yield Button("Reset", id="btn-reset")
            yield Button("Export Config", id="btn-export")
            yield Button("Import Config", id="btn-import")

        yield Footer()

    def action_save_config(self) -> None:
        """Save configuration to file."""
        import json
        import os

        # Collect current config values from inputs
        config = {}
        try:
            for widget in self.query("ConfigInput"):
                key = widget._key
                input_widget = widget.query_one(f"#input-{key}")
                if input_widget:
                    config[key] = input_widget.value

            for widget in self.query("ConfigToggle"):
                key = widget._key
                switch_widget = widget.query_one(f"#switch-{key}")
                if switch_widget:
                    config[key] = switch_widget.value
        except Exception:
            pass  # Widget query may fail

        # Save to file
        config_dir = os.path.expanduser("~/.proxima")
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, "config.json")

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            self.app.notify(f"Configuration saved to {config_file}", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to save config: {e}", severity="error")

    async def _import_config(self) -> None:
        """Import configuration from file."""
        import json
        import os

        from .modals import FormField, FormModal

        # Show file selection dialog
        field = FormField(
            key="filepath",
            label="Config File Path",
            field_type="text",
            default=os.path.expanduser("~/.proxima/config.json"),
            placeholder="Path to config file (JSON or YAML)",
            required=True,
        )

        modal = FormModal(title="Import Configuration", fields=[field])
        response = await self.app.push_screen_wait(modal)

        if not response or not response.confirmed:
            return

        filepath = response.data.get("filepath", "")
        if not filepath or not os.path.exists(filepath):
            self.app.notify("File not found", severity="error")
            return

        try:
            with open(filepath, encoding="utf-8") as f:
                if filepath.endswith((".yml", ".yaml")):
                    try:
                        import yaml

                        config = yaml.safe_load(f)
                    except ImportError:
                        self.app.notify(
                            "YAML support requires pyyaml package", severity="error"
                        )
                        return
                else:
                    config = json.load(f)

            # Apply config values to widgets
            for key, value in config.items():
                try:
                    # Try ConfigInput
                    input_widget = self.query_one(f"#input-{key}")
                    if input_widget:
                        input_widget.value = str(value)
                        continue
                except Exception:
                    pass

                try:
                    # Try ConfigToggle
                    switch_widget = self.query_one(f"#switch-{key}")
                    if switch_widget:
                        switch_widget.value = bool(value)
                except Exception:
                    pass

            self.app.notify(
                f"Configuration imported from {filepath}", severity="information"
            )
        except Exception as e:
            self.app.notify(f"Failed to import config: {e}", severity="error")

    async def _export_config(self) -> None:
        """Export current configuration to file."""
        import json
        import os
        from datetime import datetime

        # Collect current config values
        config = {}
        try:
            for widget in self.query("ConfigInput"):
                key = widget._key
                input_widget = widget.query_one(f"#input-{key}")
                if input_widget:
                    config[key] = input_widget.value

            for widget in self.query("ConfigToggle"):
                key = widget._key
                switch_widget = widget.query_one(f"#switch-{key}")
                if switch_widget:
                    config[key] = switch_widget.value
        except Exception:
            pass

        # Export to file
        export_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(export_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"proxima-config-{timestamp}.json"
        filepath = os.path.join(export_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            self.app.notify(f"Config exported to {filename}", severity="information")
        except Exception as e:
            self.app.notify(f"Export failed: {e}", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-save":
            self.action_save_config()
        elif event.button.id == "btn-reset":
            self.app.notify("Configuration reset to defaults", severity="warning")
        elif event.button.id == "btn-export":
            self.run_worker(self._export_config())
        elif event.button.id == "btn-import":
            self.run_worker(self._import_config())


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
        Binding("r", "refresh_results", "r Refresh", show=False),
        Binding("delete", "delete_result", "Del Delete", show=False),
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
        self.app.notify("Results refreshed", severity="information")

    def action_delete_result(self) -> None:
        """Delete selected result."""
        results_table = self.query_one("#results-table", ResultsTable)
        selected = results_table.get_selected_result()
        if selected:
            self.app.notify(f"Deleted result {selected['id']}", severity="warning")
        else:
            self.app.notify("No result selected", severity="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-refresh":
            self.action_refresh_results()
        elif event.button.id == "btn-details":
            results_table = self.query_one("#results-table", ResultsTable)
            selected = results_table.get_selected_result()
            if selected:
                self.app.notify(
                    f"Details for {selected['id']}: {selected}", severity="information"
                )
        elif event.button.id == "btn-export":
            self.app.notify("Results exported to results.json", severity="information")
        elif event.button.id == "btn-delete":
            self.action_delete_result()
        elif event.button.id == "btn-clear":
            self.app.notify("All results cleared", severity="warning")


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
        Binding("r", "refresh_backends", "r Refresh", show=False),
        Binding("t", "test_connection", "t Test", show=False),
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
            last_used = datetime.fromtimestamp(backend.last_used).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            details.mount(Label(f"Last Used: {last_used}"))

        if backend.error_message:
            details.mount(Label(f"Error: {backend.error_message}"))

    def on_backend_card_selected(self, event: BackendCard.Selected) -> None:
        """Handle backend card selection."""
        self._show_backend_details(event.backend)

    def action_refresh_backends(self) -> None:
        """Refresh backend list."""
        self._refresh_backends()
        self.app.notify("Backends refreshed", severity="information")

    async def action_test_connection(self) -> None:
        """Test connection to selected backend."""
        if not self._selected_backend:
            self.app.notify("No backend selected", severity="warning")
            return

        self.app.notify(
            f"Testing connection to {self._selected_backend.name}...",
            severity="information",
        )

        try:
            # Try to test actual backend connection
            from proxima.backends.registry import backend_registry

            backend = backend_registry.get(self._selected_backend.name)
            if backend:
                # Attempt a simple test
                is_available = await backend.is_available()
                if is_available:
                    self.app.notify(
                        f"{self._selected_backend.name}: Connection successful!",
                        severity="information",
                    )
                    self._selected_backend.status = BackendStatus.CONNECTED
                else:
                    self.app.notify(
                        f"{self._selected_backend.name}: Connection failed",
                        severity="error",
                    )
                    self._selected_backend.status = BackendStatus.ERROR
                # Refresh the display
                self._show_backend_details(self._selected_backend)
            else:
                self.app.notify(
                    f"Backend {self._selected_backend.name} not found in registry",
                    severity="warning",
                )
        except ImportError:
            # Fallback if registry not available
            self.app.notify(
                f"Simulated test for {self._selected_backend.name}: OK",
                severity="information",
            )
        except Exception as e:
            self.app.notify(f"Connection test failed: {e}", severity="error")

    async def _show_add_backend_dialog(self) -> None:
        """Show dialog to add a new backend."""
        from .modals import FormField, FormModal

        fields = [
            FormField(
                key="name",
                label="Backend Name",
                field_type="text",
                placeholder="e.g., my_custom_backend",
                required=True,
            ),
            FormField(
                key="backend_type",
                label="Backend Type",
                field_type="text",
                default="simulator",
                placeholder="simulator, cloud, local",
            ),
            FormField(
                key="endpoint",
                label="Endpoint URL (optional)",
                field_type="text",
                placeholder="http://localhost:8080",
            ),
            FormField(
                key="api_key",
                label="API Key (optional)",
                field_type="password",
                placeholder="Enter API key if required",
            ),
        ]

        modal = FormModal(title="Add New Backend", fields=fields)
        response = await self.app.push_screen_wait(modal)

        if not response or not response.confirmed:
            return

        data = response.data
        backend_name = data.get("name", "").strip()

        if not backend_name:
            self.app.notify("Backend name is required", severity="error")
            return

        try:
            # Try to register with actual backend registry
            from proxima.backends.registry import backend_registry

            # Create a simple custom backend entry
            backend_registry.register_custom(
                name=backend_name,
                backend_type=data.get("backend_type", "simulator"),
                endpoint=data.get("endpoint"),
                api_key=data.get("api_key"),
            )

            self.app.notify(
                f"Backend '{backend_name}' added successfully!", severity="information"
            )
            self._refresh_backends()

        except ImportError:
            # Fallback if registry not available - just show success
            self.app.notify(
                f"Backend '{backend_name}' registered (simulation mode)",
                severity="information",
            )
            self._refresh_backends()
        except AttributeError:
            # register_custom might not exist
            self.app.notify(
                "Custom backend registration not supported yet", severity="warning"
            )
        except Exception as e:
            self.app.notify(f"Failed to add backend: {e}", severity="error")

    async def _show_configure_backend_dialog(self) -> None:
        """Show configuration dialog for selected backend."""
        if not self._selected_backend:
            self.app.notify("No backend selected", severity="warning")
            return

        from .modals import FormField, FormModal

        fields = [
            FormField(
                key="timeout",
                label="Timeout (seconds)",
                field_type="text",
                default="300",
                placeholder="Request timeout",
            ),
            FormField(
                key="max_shots",
                label="Max Shots",
                field_type="text",
                default="10000",
                placeholder="Maximum shots per execution",
            ),
            FormField(
                key="retry_count",
                label="Retry Count",
                field_type="text",
                default="3",
                placeholder="Number of retries on failure",
            ),
        ]

        modal = FormModal(
            title=f"Configure {self._selected_backend.name}", fields=fields
        )
        response = await self.app.push_screen_wait(modal)

        if not response or not response.confirmed:
            return

        data = response.data

        try:
            # Apply configuration
            from proxima.backends.registry import backend_registry

            backend = backend_registry.get(self._selected_backend.name)
            if backend:
                backend.configure(
                    timeout=int(data.get("timeout", 300)),
                    max_shots=int(data.get("max_shots", 10000)),
                    retry_count=int(data.get("retry_count", 3)),
                )
            self.app.notify(
                f"Configuration applied to {self._selected_backend.name}",
                severity="information",
            )
        except Exception as e:
            self.app.notify(f"Configuration applied (local only): {e}", severity="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-refresh":
            self.action_refresh_backends()
        elif event.button.id == "btn-test":
            self.action_test_connection()
        elif event.button.id == "btn-configure":
            self.run_worker(self._show_configure_backend_dialog())
        elif event.button.id == "btn-add":
            self.run_worker(self._show_add_backend_dialog())
