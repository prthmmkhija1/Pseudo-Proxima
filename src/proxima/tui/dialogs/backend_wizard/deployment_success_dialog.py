"""Deployment Success Dialog.

Modal dialog shown after successful backend deployment.
Provides next steps and links to documentation.
"""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center
from textual.widgets import Static, Button
from textual.screen import ModalScreen


class DeploymentSuccessDialog(ModalScreen[bool]):
    """Dialog shown after successful backend deployment.
    
    Displays success message, created files, and next steps
    for using the new backend.
    """
    
    DEFAULT_CSS = """
    DeploymentSuccessDialog {
        align: center middle;
    }
    
    DeploymentSuccessDialog .dialog-container {
        width: 70;
        height: auto;
        max-height: 80%;
        border: double $success;
        background: $surface;
        padding: 1 2;
    }
    
    DeploymentSuccessDialog .success-icon {
        width: 100%;
        text-align: center;
        color: $success;
        text-style: bold;
        padding: 1 0;
    }
    
    DeploymentSuccessDialog .success-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $success;
        padding: 0 0 1 0;
    }
    
    DeploymentSuccessDialog .success-message {
        text-align: center;
        padding: 1;
        color: $text;
    }
    
    DeploymentSuccessDialog .files-section {
        padding: 1;
        margin: 1 0;
        background: $surface-darken-1;
        border: solid $success-darken-2;
    }
    
    DeploymentSuccessDialog .section-title {
        text-style: bold;
        color: $success;
        margin-bottom: 1;
    }
    
    DeploymentSuccessDialog .file-item {
        color: $text-muted;
        padding-left: 2;
    }
    
    DeploymentSuccessDialog .next-steps {
        padding: 1;
        margin: 1 0;
        background: $primary-darken-3;
        border: solid $primary;
    }
    
    DeploymentSuccessDialog .code-hint {
        background: $surface-darken-2;
        padding: 1;
        margin: 1 0;
        font-family: monospace;
    }
    
    DeploymentSuccessDialog .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
        margin-top: 1;
    }
    
    DeploymentSuccessDialog .action-button {
        margin: 0 1;
        min-width: 16;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("enter", "close", "Close"),
    ]
    
    def __init__(
        self,
        backend_name: str,
        display_name: Optional[str] = None,
        created_files: Optional[list] = None,
        output_path: Optional[str] = None
    ):
        """Initialize the success dialog.
        
        Args:
            backend_name: Internal name of the backend
            display_name: User-friendly name
            created_files: List of created file paths
            output_path: Base output directory
        """
        super().__init__()
        self.backend_name = backend_name
        self.display_name = display_name or backend_name.replace("_", " ").title()
        self.created_files = created_files or []
        self.output_path = output_path
    
    def compose(self) -> ComposeResult:
        """Create dialog widgets."""
        with Center():
            with Vertical(classes="dialog-container"):
                # Success icon
                yield Static("✓", classes="success-icon")
                
                # Title
                yield Static(
                    f"Backend Deployed Successfully!",
                    classes="success-title"
                )
                
                # Success message
                yield Static(
                    f"'{self.display_name}' has been created and registered.\n"
                    f"It is now available in the backend selection menu.",
                    classes="success-message"
                )
                
                # Created files section
                if self.created_files:
                    with Vertical(classes="files-section"):
                        yield Static("📁 Created Files:", classes="section-title")
                        for file_path in self.created_files[:5]:  # Show first 5
                            yield Static(f"  ✓ {file_path}", classes="file-item")
                        if len(self.created_files) > 5:
                            yield Static(
                                f"  ... and {len(self.created_files) - 5} more",
                                classes="file-item"
                            )
                
                # Next steps
                with Vertical(classes="next-steps"):
                    yield Static("📋 Next Steps:", classes="section-title")
                    yield Static(
                        "1. Review the generated code\n"
                        "2. Implement the TODO sections\n"
                        "3. Run the tests to verify\n"
                        "4. Update the documentation"
                    )
                
                # Usage hint
                yield Static(
                    f"Usage:\n"
                    f"from proxima.backends.{self.backend_name} import get_adapter\n"
                    f"adapter = get_adapter()\n"
                    f"adapter.initialize()",
                    classes="code-hint"
                )
                
                # Buttons
                with Horizontal(classes="button-container"):
                    yield Button(
                        "📝 View Files",
                        id="btn_view_files",
                        variant="default",
                        classes="action-button"
                    )
                    yield Button(
                        "✓ Done",
                        id="btn_done",
                        variant="success",
                        classes="action-button"
                    )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn_done":
            self.dismiss(True)
        
        elif event.button.id == "btn_view_files":
            # Open file explorer or show file list
            if self.output_path:
                self.notify(
                    f"Files located at: {self.output_path}",
                    severity="information",
                    timeout=5
                )
    
    def action_close(self) -> None:
        """Handle escape/enter key."""
        self.dismiss(True)


class DeploymentFailureDialog(ModalScreen[bool]):
    """Dialog shown when backend deployment fails.
    
    Displays error information and suggestions for fixing.
    """
    
    DEFAULT_CSS = """
    DeploymentFailureDialog {
        align: center middle;
    }
    
    DeploymentFailureDialog .dialog-container {
        width: 70;
        height: auto;
        max-height: 80%;
        border: double $error;
        background: $surface;
        padding: 1 2;
    }
    
    DeploymentFailureDialog .error-icon {
        width: 100%;
        text-align: center;
        color: $error;
        text-style: bold;
        padding: 1 0;
    }
    
    DeploymentFailureDialog .error-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $error;
        padding: 0 0 1 0;
    }
    
    DeploymentFailureDialog .error-message {
        background: $error-darken-3;
        padding: 1;
        margin: 1 0;
        border: solid $error;
    }
    
    DeploymentFailureDialog .suggestions {
        padding: 1;
        margin: 1 0;
        background: $surface-darken-1;
    }
    
    DeploymentFailureDialog .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    DeploymentFailureDialog .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
    }
    """
    
    def __init__(self, error_message: str, backend_name: Optional[str] = None):
        """Initialize failure dialog.
        
        Args:
            error_message: Error message to display
            backend_name: Name of the backend that failed
        """
        super().__init__()
        self.error_message = error_message
        self.backend_name = backend_name
    
    def compose(self) -> ComposeResult:
        """Create dialog widgets."""
        with Center():
            with Vertical(classes="dialog-container"):
                yield Static("✗", classes="error-icon")
                
                yield Static(
                    "Deployment Failed",
                    classes="error-title"
                )
                
                yield Static(
                    f"Error: {self.error_message}",
                    classes="error-message"
                )
                
                with Vertical(classes="suggestions"):
                    yield Static("💡 Suggestions:", classes="section-title")
                    yield Static(
                        "• Check file permissions\n"
                        "• Verify the output path exists\n"
                        "• Review the generated code for errors\n"
                        "• Check the logs for more details"
                    )
                
                with Horizontal(classes="button-container"):
                    yield Button(
                        "← Go Back",
                        id="btn_back",
                        variant="default"
                    )
                    yield Button(
                        "Retry",
                        id="btn_retry",
                        variant="warning"
                    )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn_back":
            self.dismiss(False)
        elif event.button.id == "btn_retry":
            self.dismiss(True)  # Signal to retry
