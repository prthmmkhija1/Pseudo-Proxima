"""Deployment Progress Dialog.

Modal dialog showing detailed deployment progress with stage tracking.
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, Any, List
import asyncio

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import Static, Button, ProgressBar, Label, RichLog
from textual.screen import ModalScreen
from textual.reactive import reactive


class DeploymentStageWidget(Static):
    """Widget showing a single deployment stage."""
    
    DEFAULT_CSS = """
    DeploymentStageWidget {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    
    DeploymentStageWidget.pending {
        color: $text-muted;
    }
    
    DeploymentStageWidget.in-progress {
        color: $warning;
        text-style: bold;
    }
    
    DeploymentStageWidget.completed {
        color: $success;
    }
    
    DeploymentStageWidget.failed {
        color: $error;
    }
    """
    
    status = reactive("pending")
    
    def __init__(
        self,
        stage_name: str,
        description: str,
        **kwargs
    ):
        """Initialize stage widget.
        
        Args:
            stage_name: Name of the deployment stage
            description: Description of what this stage does
        """
        super().__init__(**kwargs)
        self.stage_name = stage_name
        self.description = description
    
    def render(self) -> str:
        """Render the stage display."""
        icons = {
            "pending": "○",
            "in-progress": "◐",
            "completed": "✓",
            "failed": "✗",
        }
        icon = icons.get(self.status, "○")
        
        return f"{icon} {self.stage_name}: {self.description}"
    
    def watch_status(self, status: str) -> None:
        """Watch status changes to update CSS class."""
        self.remove_class("pending", "in-progress", "completed", "failed")
        self.add_class(status)


class DeploymentProgressDialog(ModalScreen[bool]):
    """Dialog showing detailed deployment progress.
    
    Shows stage-by-stage progress with logs and status updates.
    """
    
    DEFAULT_CSS = """
    DeploymentProgressDialog {
        align: center middle;
    }
    
    DeploymentProgressDialog .dialog-container {
        width: 80;
        height: 40;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    DeploymentProgressDialog .dialog-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    DeploymentProgressDialog .stages-section {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
        margin: 1 0;
    }
    
    DeploymentProgressDialog .log-section {
        height: 1fr;
        border: solid $primary-darken-3;
        margin: 1 0;
    }
    
    DeploymentProgressDialog .progress-section {
        height: auto;
        padding: 1;
    }
    
    DeploymentProgressDialog .current-file {
        color: $text-muted;
        text-align: center;
    }
    
    DeploymentProgressDialog .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    DeploymentProgressDialog .action-button {
        margin: 0 1;
        min-width: 12;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    current_stage = reactive("Initializing")
    current_file = reactive("")
    progress = reactive(0.0)
    is_complete = reactive(False)
    has_error = reactive(False)
    
    def __init__(
        self,
        backend_name: str,
        display_name: str,
        on_cancel: Optional[Callable[[], None]] = None
    ):
        """Initialize deployment progress dialog.
        
        Args:
            backend_name: Internal backend name
            display_name: User-friendly backend name
            on_cancel: Callback when user cancels
        """
        super().__init__()
        self.backend_name = backend_name
        self.display_name = display_name
        self._on_cancel = on_cancel
        self._stages: Dict[str, DeploymentStageWidget] = {}
    
    def compose(self) -> ComposeResult:
        """Create dialog widgets."""
        with Center():
            with Vertical(classes="dialog-container"):
                # Title
                yield Static(
                    f"🚀 Deploying {self.display_name}",
                    classes="dialog-title"
                )
                
                # Stages section
                with Vertical(classes="stages-section"):
                    stages = [
                        ("initializing", "Preparing deployment"),
                        ("validating", "Validating generated code"),
                        ("backup", "Creating backup"),
                        ("writing", "Writing files"),
                        ("registry", "Updating registry"),
                        ("verify", "Running verification"),
                        ("complete", "Finalizing"),
                    ]
                    
                    for stage_id, description in stages:
                        widget = DeploymentStageWidget(
                            stage_name=stage_id.title(),
                            description=description,
                            id=f"stage_{stage_id}"
                        )
                        self._stages[stage_id] = widget
                        yield widget
                
                # Progress section
                with Vertical(classes="progress-section"):
                    yield ProgressBar(total=100, show_eta=False, id="progress_bar")
                    yield Label("", id="current_file_label", classes="current-file")
                
                # Log section
                yield RichLog(id="deployment_log", classes="log-section")
                
                # Buttons
                with Horizontal(classes="button-container"):
                    yield Button(
                        "Cancel",
                        id="btn_cancel",
                        variant="default",
                        classes="action-button"
                    )
                    yield Button(
                        "Close",
                        id="btn_close",
                        variant="primary",
                        classes="action-button",
                        disabled=True
                    )
    
    def update_stage(self, stage: str, status: str = "in-progress") -> None:
        """Update a stage's status.
        
        Args:
            stage: Stage identifier
            status: New status (pending, in-progress, completed, failed)
        """
        if stage in self._stages:
            self._stages[stage].status = status
    
    def mark_stage_complete(self, stage: str) -> None:
        """Mark a stage as completed."""
        self.update_stage(stage, "completed")
    
    def mark_stage_failed(self, stage: str) -> None:
        """Mark a stage as failed."""
        self.update_stage(stage, "failed")
    
    def log(self, message: str, style: str = "") -> None:
        """Add a message to the deployment log.
        
        Args:
            message: Message to log
            style: Rich style for the message
        """
        try:
            log = self.query_one("#deployment_log", RichLog)
            if style:
                log.write(f"[{style}]{message}[/{style}]")
            else:
                log.write(message)
        except Exception:
            pass
    
    def watch_progress(self, progress: float) -> None:
        """Watch progress changes."""
        try:
            bar = self.query_one("#progress_bar", ProgressBar)
            bar.update(progress=int(progress * 100))
        except Exception:
            pass
    
    def watch_current_file(self, current_file: str) -> None:
        """Watch current file changes."""
        try:
            label = self.query_one("#current_file_label", Label)
            if current_file:
                label.update(f"Writing: {current_file}")
            else:
                label.update("")
        except Exception:
            pass
    
    def watch_is_complete(self, is_complete: bool) -> None:
        """Watch completion status."""
        if is_complete:
            try:
                cancel_btn = self.query_one("#btn_cancel", Button)
                cancel_btn.disabled = True
                
                close_btn = self.query_one("#btn_close", Button)
                close_btn.disabled = False
            except Exception:
                pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn_cancel":
            if self._on_cancel:
                self._on_cancel()
            self.dismiss(False)
        
        elif event.button.id == "btn_close":
            self.dismiss(not self.has_error)
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        if not self.is_complete:
            if self._on_cancel:
                self._on_cancel()
            self.dismiss(False)
        else:
            self.dismiss(not self.has_error)
    
    def complete_success(self) -> None:
        """Mark deployment as successfully complete."""
        self.is_complete = True
        self.has_error = False
        self.current_stage = "Complete"
        self.progress = 1.0
        self.log("✓ Deployment completed successfully!", "bold green")
        
        # Mark all stages complete
        for stage in self._stages.values():
            if stage.status == "in-progress":
                stage.status = "completed"
    
    def complete_failure(self, error: str) -> None:
        """Mark deployment as failed.
        
        Args:
            error: Error message
        """
        self.is_complete = True
        self.has_error = True
        self.current_stage = "Failed"
        self.log(f"✗ Deployment failed: {error}", "bold red")
        
        # Mark current stage as failed
        for stage in self._stages.values():
            if stage.status == "in-progress":
                stage.status = "failed"
                break


class QuickDeployDialog(ModalScreen[bool]):
    """Simplified deployment dialog for quick deployments."""
    
    DEFAULT_CSS = """
    QuickDeployDialog {
        align: center middle;
    }
    
    QuickDeployDialog .dialog-container {
        width: 60;
        height: 15;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    QuickDeployDialog .dialog-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
    }
    
    QuickDeployDialog .status-message {
        text-align: center;
        padding: 1;
    }
    
    QuickDeployDialog .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
    }
    """
    
    status_message = reactive("Deploying...")
    
    def __init__(self, backend_name: str):
        """Initialize quick deploy dialog.
        
        Args:
            backend_name: Name of the backend being deployed
        """
        super().__init__()
        self.backend_name = backend_name
    
    def compose(self) -> ComposeResult:
        """Create dialog widgets."""
        with Center():
            with Vertical(classes="dialog-container"):
                yield Static(
                    f"🚀 Deploying {self.backend_name}",
                    classes="dialog-title"
                )
                
                yield ProgressBar(total=100, id="progress")
                
                yield Static("", id="status", classes="status-message")
                
                with Horizontal(classes="button-container"):
                    yield Button(
                        "Cancel",
                        id="btn_cancel",
                        variant="default"
                    )
    
    def watch_status_message(self, message: str) -> None:
        """Watch status message changes."""
        try:
            status = self.query_one("#status", Static)
            status.update(message)
        except Exception:
            pass
    
    def set_progress(self, value: float) -> None:
        """Set progress bar value (0.0 to 1.0)."""
        try:
            bar = self.query_one("#progress", ProgressBar)
            bar.update(progress=int(value * 100))
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn_cancel":
            self.dismiss(False)
