"""Execution screen for Proxima TUI.

Live execution monitoring with progress and controls.
"""

from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, RichLog
from rich.text import Text
from rich.panel import Panel

from .base import BaseScreen
from ..styles.theme import get_theme
from ..components.progress import ProgressBar, StageTimeline

# Import controller for execution management
try:
    from ..controllers import ExecutionController
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False


class ExecutionScreen(BaseScreen):
    """Execution monitoring screen.
    
    Shows:
    - Current execution info
    - Progress bar
    - Stage timeline
    - Execution controls
    - Log viewer
    """
    
    SCREEN_NAME = "execution"
    SCREEN_TITLE = "Execution Monitor"
    
    BINDINGS = [
        ("p", "pause_execution", "Pause"),
        ("r", "resume_execution", "Resume"),
        ("a", "abort_execution", "Abort"),
        ("z", "rollback", "Rollback"),
        ("l", "toggle_log", "Toggle Log"),
    ]
    
    DEFAULT_CSS = """
    ExecutionScreen .execution-panel {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    ExecutionScreen .execution-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    ExecutionScreen .execution-info {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    ExecutionScreen .progress-section {
        margin: 1 0;
    }
    
    ExecutionScreen .timeline-section {
        margin: 1 0;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
    }
    
    ExecutionScreen .controls-section {
        height: auto;
        layout: horizontal;
        margin: 1 0;
    }
    
    ExecutionScreen .control-button {
        margin-right: 1;
        min-width: 12;
    }
    
    ExecutionScreen .log-section {
        height: 1fr;
        border: solid $primary-darken-2;
        background: $surface-darken-2;
    }
    
    ExecutionScreen .log-section.-hidden {
        display: none;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the execution screen."""
        super().__init__(**kwargs)
        self._log_visible = True
        self._update_timer = None
        
        # Initialize execution controller
        self._controller = None
        if CONTROLLER_AVAILABLE:
            try:
                self._controller = ExecutionController(self.state)
            except Exception as e:
                # Controller init failed, will use fallback mode
                pass
    
    def on_mount(self) -> None:
        """Set up progress update timer on mount."""
        # Start periodic progress updates
        self._update_timer = self.set_interval(0.5, self._update_progress)
    
    def on_unmount(self) -> None:
        """Clean up timer on unmount."""
        if self._update_timer:
            self._update_timer.stop()
    
    def _update_progress(self) -> None:
        """Update progress display from controller or state.
        
        Handles real-time progress updates from the execution controller,
        including elapsed time tracking and simulated progress in demo mode.
        """
        import time
        
        # Track elapsed time when running
        if self.state.execution_status == "RUNNING" and not self.state.is_paused:
            if not hasattr(self, '_start_time') or self._start_time is None:
                self._start_time = time.time()
            self.state.elapsed_ms = (time.time() - self._start_time) * 1000
            
            # Simulate progress in demo mode (when no real backend is running)
            if not self._controller or not getattr(self._controller, '_core_controller', None):
                # Auto-advance progress for demo purposes
                if self.state.progress_percent < 100:
                    self.state.progress_percent = min(100.0, self.state.progress_percent + 0.5)
                    # Update stage based on progress
                    new_stage_index = int(self.state.progress_percent / 100 * self.state.total_stages)
                    new_stage_index = min(new_stage_index, self.state.total_stages - 1)
                    if new_stage_index != self.state.stage_index:
                        self.state.stage_index = new_stage_index
                        if self.state.all_stages and new_stage_index < len(self.state.all_stages):
                            self.state.current_stage = self.state.all_stages[new_stage_index].name
                    # Estimate ETA
                    if self.state.progress_percent > 0:
                        total_estimated = self.state.elapsed_ms / (self.state.progress_percent / 100)
                        self.state.eta_ms = max(0, total_estimated - self.state.elapsed_ms)
        elif self.state.execution_status == "IDLE":
            # Reset start time when idle
            self._start_time = None
        
        if self._controller:
            try:
                # Get status from controller
                status = self._controller.get_status()
                if status:
                    self.state.progress_percent = status.get('progress', self.state.progress_percent)
                    self.state.current_stage = status.get('stage', self.state.current_stage)
                    self.state.is_running = status.get('is_running', self.state.is_running)
                    self.state.is_paused = status.get('is_paused', self.state.is_paused)
                    self.state.elapsed_ms = status.get('elapsed_ms', self.state.elapsed_ms)
                    self.state.eta_ms = status.get('eta_ms', self.state.eta_ms)
            except Exception:
                pass
        
        # Update progress bar display
        try:
            progress_bar = self.query_one(ProgressBar)
            progress_bar.progress = self.state.progress_percent
            progress_bar.stage_name = f"Stage {self.state.stage_index + 1}/{self.state.total_stages}: {self.state.current_stage}"
            progress_bar.eta_text = f"Elapsed: {self.state.get_formatted_elapsed()}  |  ETA: {self.state.get_formatted_eta()}"
            progress_bar.refresh()
        except Exception:
            pass
        
        # Update stage timeline
        try:
            timeline = self.query_one(StageTimeline)
            timeline.current_index = self.state.stage_index
            timeline.refresh()
        except Exception:
            pass
        
        # Update info panel if available
        try:
            info_panel = self.query_one(ExecutionInfoPanel)
            info_panel.state = self.state
            info_panel.refresh()
        except Exception:
            pass
    
    def compose_main(self):
        """Compose the execution screen content."""
        with Vertical(classes="main-content"):
            # Execution panel
            with Container(classes="execution-panel"):
                yield Static(
                    "Execution Monitor",
                    classes="execution-title",
                )
                yield ExecutionInfoPanel(self.state)
                
                # Progress bar
                with Vertical(classes="progress-section"):
                    yield ProgressBar(
                        progress=self.state.progress_percent,
                        stage_name=f"Stage {self.state.stage_index + 1}/{self.state.total_stages}: {self.state.current_stage}",
                        eta_text=f"Elapsed: {self.state.get_formatted_elapsed()}  |  ETA: {self.state.get_formatted_eta()}",
                    )
                
                # Stage timeline
                with Vertical(classes="timeline-section"):
                    yield StageTimeline(
                        stages=self.state.all_stages,
                        current_index=self.state.stage_index,
                        total_elapsed_ms=self.state.elapsed_ms,
                        total_eta_ms=self.state.eta_ms,
                    )
            
            # Controls
            with Horizontal(classes="controls-section"):
                yield Button(
                    "[P] Pause",
                    id="btn-pause",
                    classes="control-button",
                    variant="warning",
                )
                yield Button(
                    "[R] Resume",
                    id="btn-resume",
                    classes="control-button",
                    variant="success",
                    disabled=True,
                )
                yield Button(
                    "[A] Abort",
                    id="btn-abort",
                    classes="control-button",
                    variant="error",
                )
                yield Button(
                    "[Z] Rollback",
                    id="btn-rollback",
                    classes="control-button",
                    disabled=not self.state.rollback_available,
                )
                yield Button(
                    "[L] Toggle Log",
                    id="btn-toggle-log",
                    classes="control-button",
                )
            
            # Log viewer
            yield ExecutionLog(classes="log-section")
    
    def action_pause_execution(self) -> None:
        """Pause the current execution."""
        if self._controller:
            try:
                # Check if pause is possible
                if not self._controller.can_pause:
                    self.notify("Cannot pause - execution not running", severity="warning")
                    return
                    
                result = self._controller.pause()
                if result:
                    self.notify("⏸ Execution paused", severity="success")
                    self._update_control_buttons("paused")
                    self._log_action("Execution paused by user")
                    # Sync state
                    self.state.execution_status = "PAUSED"
                    self._update_status_display()
                else:
                    self.notify("Could not pause execution", severity="warning")
            except Exception as e:
                self.notify(f"Pause failed: {e}", severity="error")
        else:
            # Fallback: Update UI state directly
            self.state.is_paused = True
            self.state.execution_status = "PAUSED"
            self._update_control_buttons("paused")
            self._log_action("Execution paused (local mode)")
            self._update_status_display()
            self.notify("⏸ Execution paused", severity="success")
    
    def action_resume_execution(self) -> None:
        """Resume the paused execution."""
        if self._controller:
            try:
                # Check if resume is possible
                if not self._controller.can_resume:
                    self.notify("Cannot resume - execution not paused", severity="warning")
                    return
                    
                result = self._controller.resume()
                if result:
                    self.notify("▶ Execution resumed", severity="success")
                    self._update_control_buttons("running")
                    self._log_action("Execution resumed by user")
                    # Sync state
                    self.state.execution_status = "RUNNING"
                    self._update_status_display()
                else:
                    self.notify("Could not resume execution", severity="warning")
            except Exception as e:
                self.notify(f"Resume failed: {e}", severity="error")
        else:
            # Fallback: Update UI state directly
            self.state.is_paused = False
            self.state.execution_status = "RUNNING"
            self._update_control_buttons("running")
            self._log_action("Execution resumed (local mode)")
            self._update_status_display()
            self.notify("▶ Execution resumed", severity="success")
    
    def action_abort_execution(self) -> None:
        """Abort the current execution."""
        if self._controller:
            try:
                # Check if abort is possible
                if not self._controller.can_abort:
                    self.notify("Cannot abort - no active execution", severity="warning")
                    return
                    
                result = self._controller.abort()
                if result:
                    self.notify("⏹ Execution aborted", severity="warning")
                    self._update_control_buttons("stopped")
                    self._log_action("Execution aborted by user")
                    # Sync state
                    self.state.execution_status = "ABORTED"
                    self._update_status_display()
                else:
                    self.notify("Could not abort execution", severity="warning")
            except Exception as e:
                self.notify(f"Abort failed: {e}", severity="error")
        else:
            # Fallback: Update UI state directly
            self.state.is_running = False
            self.state.is_paused = False
            self.state.execution_status = "ABORTED"
            self._update_control_buttons("stopped")
            self._log_action("Execution aborted (local mode)")
            self._update_status_display()
            self.notify("⏹ Execution aborted", severity="warning")
    
    def action_rollback(self) -> None:
        """Rollback to last checkpoint."""
        if self._controller:
            try:
                # Check if rollback is possible
                if not self._controller.can_rollback:
                    self.notify("No checkpoint available for rollback", severity="warning")
                    return
                    
                result = self._controller.rollback()
                if result:
                    self.notify("↩ Rolled back to last checkpoint", severity="success")
                    self._log_action("Rolled back to last checkpoint")
                    # Sync state - get current status from controller
                    status = self._controller.get_status()
                    self.state.stage_index = status.get('stage_index', 0)
                    self.state.progress_percent = status.get('progress', 0)
                    self._update_control_buttons("running")
                    self._update_status_display()
                else:
                    self.notify("Rollback failed", severity="warning")
            except Exception as e:
                self.notify(f"Rollback failed: {e}", severity="error")
        else:
            # Fallback: Check local state for rollback
            if self.state.rollback_available:
                self._log_action("Rolled back to last checkpoint (local mode)")
                self._update_status_display()
                self.notify("↩ Rolled back to last checkpoint", severity="success")
            else:
                self.notify("No checkpoint available for rollback", severity="warning")
    
    def _update_status_display(self) -> None:
        """Update the status display after state changes."""
        try:
            # Refresh the execution info panel
            info_panel = self.query_one(ExecutionInfoPanel)
            info_panel.refresh()
        except Exception:
            pass
    
    def _log_action(self, message: str) -> None:
        """Log an action to the execution log."""
        try:
            log = self.query_one(ExecutionLog)
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            theme = get_theme()
            text = Text()
            text.append(f"[{timestamp}] ", style=theme.fg_subtle)
            text.append("ACTION  ", style=f"bold {theme.primary}")
            text.append(message, style=theme.fg_base)
            log.write(text)
        except Exception:
            pass  # Log may not exist
    
    def _update_control_buttons(self, state: str) -> None:
        """Update control button states based on execution state."""
        try:
            pause_btn = self.query_one("#btn-pause", Button) if self.query("#btn-pause") else None
            resume_btn = self.query_one("#btn-resume", Button) if self.query("#btn-resume") else None
            abort_btn = self.query_one("#btn-abort", Button) if self.query("#btn-abort") else None
            
            if state == "running":
                if pause_btn:
                    pause_btn.disabled = False
                if resume_btn:
                    resume_btn.disabled = True
                if abort_btn:
                    abort_btn.disabled = False
            elif state == "paused":
                if pause_btn:
                    pause_btn.disabled = True
                if resume_btn:
                    resume_btn.disabled = False
                if abort_btn:
                    abort_btn.disabled = False
            elif state == "stopped":
                if pause_btn:
                    pause_btn.disabled = True
                if resume_btn:
                    resume_btn.disabled = True
                if abort_btn:
                    abort_btn.disabled = True
        except Exception:
            pass  # Buttons may not exist in all screen modes

    def action_toggle_log(self) -> None:
        """Toggle the log panel visibility."""
        self._log_visible = not self._log_visible
        log_section = self.query_one(".log-section")
        log_section.set_class(not self._log_visible, "-hidden")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-pause":
            self.action_pause_execution()
        elif button_id == "btn-resume":
            self.action_resume_execution()
        elif button_id == "btn-abort":
            self.action_abort_execution()
        elif button_id == "btn-rollback":
            self.action_rollback()
        elif button_id == "btn-toggle-log":
            self.action_toggle_log()


class ExecutionInfoPanel(Static):
    """Panel showing current execution information."""
    
    def __init__(self, state, **kwargs):
        """Initialize the info panel."""
        super().__init__(**kwargs)
        self._state = state
    
    def render(self) -> Text:
        """Render the execution info."""
        theme = get_theme()
        text = Text()
        
        if self._state.current_task:
            # Task info
            text.append("Task: ", style=theme.fg_muted)
            text.append(self._state.current_task, style=f"bold {theme.fg_base}")
            text.append("\n")
            
            # Task ID
            if self._state.current_task_id:
                text.append("ID: ", style=theme.fg_muted)
                text.append(self._state.current_task_id, style=theme.fg_subtle)
                text.append("\n")
            
            # Backend info
            text.append("Backend: ", style=theme.fg_muted)
            text.append(
                f"{self._state.current_backend or 'N/A'} ({self._state.current_simulator or 'N/A'})",
                style=theme.fg_base,
            )
            text.append(" â€¢ ", style=theme.border)
            text.append(f"{self._state.qubits} qubits", style=theme.fg_base)
            text.append(" â€¢ ", style=theme.border)
            text.append(f"{self._state.shots} shots", style=theme.fg_base)
        else:
            text.append("No active execution", style=theme.fg_subtle)
            text.append("\n\n")
            text.append("Start a simulation from the Dashboard or Command Palette (Ctrl+P)",
                       style=theme.fg_muted)
        
        return text


class ExecutionLog(RichLog):
    """Log viewer for execution output."""
    
    DEFAULT_CSS = """
    ExecutionLog {
        padding: 1;
    }
    """
    
    def on_mount(self) -> None:
        """Set up the log."""
        self.border_title = "Log"
        
        # Add sample log entries
        self._add_sample_logs()
    
    def _add_sample_logs(self) -> None:
        """Add sample log entries for demo."""
        theme = get_theme()
        
        logs = [
            ("14:30:22", "INFO", "Starting simulation with Cirq backend"),
            ("14:30:23", "INFO", "Initialized StateVector simulator"),
            ("14:30:24", "INFO", "Stage 1/5: Planning - completed"),
            ("14:30:26", "INFO", "Stage 2/5: Backend Init - completed"),
            ("14:30:27", "INFO", "Stage 3/5: Simulation - started"),
        ]
        
        for timestamp, level, message in logs:
            level_color = {
                "INFO": theme.info,
                "WARNING": theme.warning,
                "ERROR": theme.error,
                "DEBUG": theme.fg_subtle,
            }.get(level, theme.fg_muted)
            
            text = Text()
            text.append(f"[{timestamp}] ", style=theme.fg_subtle)
            text.append(f"{level:<8}", style=f"bold {level_color}")
            text.append(message, style=theme.fg_base)
            
            self.write(text)
