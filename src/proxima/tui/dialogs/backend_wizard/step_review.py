"""Step 7: Review & Finalize.

Final review of all backend configuration and code
before saving to the workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import Static, Button, Collapsible, TextArea
from textual.screen import ModalScreen

from .wizard_state import BackendWizardState


class ReviewStepScreen(ModalScreen[dict]):
    """
    Step 7: Final review and save screen.
    
    Shows a summary of all configuration and allows
    the user to save the backend to their project.
    """
    
    DEFAULT_CSS = """
    ReviewStepScreen {
        align: center middle;
    }
    
    ReviewStepScreen .wizard-container {
        width: 95;
        height: auto;
        max-height: 95%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    ReviewStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    ReviewStepScreen .form-container {
        height: auto;
        max-height: 75%;
        padding: 1;
    }
    
    ReviewStepScreen .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }
    
    ReviewStepScreen .field-hint {
        color: $text-muted;
        margin: 0 0 1 0;
    }
    
    ReviewStepScreen .review-section {
        padding: 1;
        margin: 0 0 1 0;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
    }
    
    ReviewStepScreen .review-item {
        padding: 0 1;
    }
    
    ReviewStepScreen .review-label {
        color: $text-muted;
        width: 20;
    }
    
    ReviewStepScreen .review-value {
        color: $text;
    }
    
    ReviewStepScreen .code-preview {
        height: 15;
        margin: 1 0;
        border: solid $primary-darken-3;
    }
    
    ReviewStepScreen .success-banner {
        background: $success-darken-3;
        padding: 1;
        margin: 1 0;
        border: solid $success;
        text-align: center;
        color: $success;
    }
    
    ReviewStepScreen .file-path-box {
        background: $primary-darken-3;
        padding: 1;
        margin: 1 0;
        border: solid $primary;
    }
    
    ReviewStepScreen .progress-section {
        margin: 1 0;
        padding: 1 0;
        border-top: solid $primary-darken-3;
    }
    
    ReviewStepScreen .progress-text {
        color: $text-muted;
    }
    
    ReviewStepScreen .progress-bar {
        color: $success;
    }
    
    ReviewStepScreen .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    ReviewStepScreen .nav-button {
        margin: 0 1;
        min-width: 14;
    }
    
    ReviewStepScreen .save-button {
        min-width: 20;
    }
    
    ReviewStepScreen Collapsible {
        margin: 0 0 1 0;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, state: BackendWizardState):
        """
        Initialize the review screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
        self.saved = False
    
    def compose(self) -> ComposeResult:
        """Compose the review screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "📋 Add Custom Backend - Review & Save",
                    classes="wizard-title"
                )
                
                yield Static(
                    "Review your backend configuration before saving:",
                    classes="field-hint"
                )
                
                with ScrollableContainer(classes="form-container"):
                    # Basic Information Section
                    with Collapsible(title="Basic Information", collapsed=False):
                        with Vertical(classes="review-section"):
                            yield Static(
                                self._format_review_item(
                                    "Backend Name",
                                    self.state.backend_name
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Display Name",
                                    self.state.display_name
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Version",
                                    self.state.version
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Type",
                                    self.state.backend_type.value if self.state.backend_type else "N/A"
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Library",
                                    self.state.library_name or "None"
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Author",
                                    self.state.author or "Unknown"
                                ),
                                classes="review-item"
                            )
                    
                    # Capabilities Section
                    with Collapsible(title="Capabilities", collapsed=False):
                        with Vertical(classes="review-section"):
                            yield Static(
                                self._format_review_item(
                                    "Max Qubits",
                                    str(self.state.max_qubits)
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Simulator Types",
                                    ", ".join(self.state.simulator_types) or "state_vector"
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Noise Support",
                                    "Yes" if self.state.supports_noise else "No"
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "GPU Support",
                                    "Yes" if self.state.supports_gpu else "No"
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Batching",
                                    "Yes" if self.state.supports_batching else "No"
                                ),
                                classes="review-item"
                            )
                    
                    # Gate Mapping Section
                    with Collapsible(title="Gate Mapping", collapsed=True):
                        with Vertical(classes="review-section"):
                            yield Static(
                                self._format_review_item(
                                    "Mode",
                                    self.state.gate_mapping_mode or "automatic"
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Template",
                                    self.state.gate_template or "standard"
                                ),
                                classes="review-item"
                            )
                            yield Static(
                                self._format_review_item(
                                    "Gates Mapped",
                                    str(len(self.state.gate_mappings))
                                ),
                                classes="review-item"
                            )
                    
                    # Test Results Section
                    with Collapsible(title="Test Results", collapsed=True):
                        with Vertical(classes="review-section"):
                            if self.state.test_results:
                                passed = self.state.test_results.get("passed", 0)
                                total = self.state.test_results.get("total", 0)
                                yield Static(
                                    self._format_review_item(
                                        "Tests Passed",
                                        f"{passed}/{total}"
                                    ),
                                    classes="review-item"
                                )
                            else:
                                yield Static(
                                    "No tests run yet",
                                    classes="review-item"
                                )
                    
                    # Code Preview Section
                    with Collapsible(title="Generated Code Preview", collapsed=True):
                        yield TextArea(
                            self.state.generated_code or "# No code generated",
                            id="code_preview",
                            classes="code-preview",
                            language="python",
                            read_only=True,
                            show_line_numbers=True
                        )
                    
                    # File output path
                    yield Static(
                        "Output Location:",
                        classes="section-title"
                    )
                    
                    file_path = self._get_output_path()
                    yield Static(
                        f"📁 {file_path}\n\n"
                        "The backend file will be created at this location.\n"
                        "You can import it with:\n"
                        f"  from proxima.backends.contrib.{self.state.backend_name} import get_backend",
                        classes="file-path-box",
                        id="file_path_display"
                    )
                
                # Progress indicator
                with Vertical(classes="progress-section"):
                    yield Static(
                        "Progress: Step 7 of 7",
                        classes="progress-text"
                    )
                    yield Static(
                        "████████ 100%",
                        classes="progress-bar"
                    )
                
                # Navigation buttons
                with Horizontal(classes="button-container"):
                    yield Button(
                        "← Back",
                        id="btn_back",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Cancel",
                        id="btn_cancel",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "💾 Save Backend",
                        id="btn_save",
                        variant="success",
                        classes="nav-button save-button"
                    )
    
    def _format_review_item(self, label: str, value: str) -> str:
        """Format a review item for display."""
        return f"{label}: {value}"
    
    def _get_output_path(self) -> str:
        """Get the output file path for the backend."""
        return f"src/proxima/backends/contrib/{self.state.backend_name}_backend.py"
    
    def _save_backend(self) -> bool:
        """
        Save the backend code to file.
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Get the workspace root (we'll use relative path)
            output_path = Path(self._get_output_path())
            
            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the code
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self.state.generated_code or "")
            
            return True
        except Exception as e:
            self.log.error(f"Failed to save backend: {e}")
            return False
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_back":
            self.state.current_step = 6
            self.dismiss({"action": "back", "state": self.state})
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_save":
            # Save the backend
            success = self._save_backend()
            
            if success:
                self.saved = True
                self.notify(
                    f"✅ Backend saved: {self._get_output_path()}",
                    severity="information",
                    timeout=5
                )
                
                # Return completion result
                self.dismiss({
                    "action": "complete",
                    "state": self.state,
                    "output_path": self._get_output_path()
                })
            else:
                self.notify(
                    "❌ Failed to save backend. Check permissions.",
                    severity="error",
                    timeout=5
                )
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        if self.saved:
            self.dismiss({
                "action": "complete",
                "state": self.state,
                "output_path": self._get_output_path()
            })
        else:
            self.dismiss({"action": "cancel"})
