"""Step 2: Basic Information Input.

Collects basic metadata about the backend including:
- Backend name (internal identifier)
- Display name (shown in UI)
- Version
- Description
- Library name (for Python backends)
- Author/maintainer
"""

from __future__ import annotations

import re
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import Static, Button, Input, Label
from textual.screen import ModalScreen
from textual.validation import Validator, ValidationResult

from .wizard_state import BackendWizardState


class BackendNameValidator(Validator):
    """Validator for backend name format."""
    
    def validate(self, value: str) -> ValidationResult:
        """
        Validate that the backend name is a valid Python identifier.
        
        Requirements:
        - Must start with a lowercase letter
        - Can only contain lowercase letters, numbers, and underscores
        - Cannot be a Python keyword
        """
        if not value:
            return self.failure("Backend name is required")
        
        if not re.match(r'^[a-z][a-z0-9_]*$', value):
            return self.failure(
                "Must start with lowercase letter, "
                "contain only lowercase letters, numbers, and underscores"
            )
        
        # Check for Python keywords
        import keyword
        if keyword.iskeyword(value):
            return self.failure(f"'{value}' is a Python keyword")
        
        return self.success()


class VersionValidator(Validator):
    """Validator for semantic version format."""
    
    def validate(self, value: str) -> ValidationResult:
        """
        Validate that the version follows semantic versioning.
        
        Format: X.Y.Z where X, Y, Z are non-negative integers
        """
        if not value:
            return self.failure("Version is required")
        
        if not re.match(r'^\d+\.\d+\.\d+$', value):
            return self.failure("Must be in format: X.Y.Z (e.g., 1.0.0)")
        
        return self.success()


class BasicInfoStepScreen(ModalScreen[dict]):
    """
    Step 2: Basic information input screen.
    
    Collects essential metadata about the backend that will be
    used to generate the backend code and configuration.
    """
    
    DEFAULT_CSS = """
    BasicInfoStepScreen {
        align: center middle;
    }
    
    BasicInfoStepScreen .wizard-container {
        width: 85;
        height: auto;
        max-height: 95%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    BasicInfoStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    BasicInfoStepScreen .form-container {
        height: auto;
        padding: 1;
    }
    
    BasicInfoStepScreen .form-field {
        width: 100%;
        height: auto;
        margin: 1 0;
    }
    
    BasicInfoStepScreen .field-label {
        color: $text;
        margin-bottom: 0;
        text-style: bold;
    }
    
    BasicInfoStepScreen .field-input {
        width: 100%;
        margin: 0;
    }
    
    BasicInfoStepScreen .field-hint {
        color: $text-muted;
        margin-top: 0;
        margin-left: 2;
    }
    
    BasicInfoStepScreen .required-marker {
        color: $error;
    }
    
    BasicInfoStepScreen .section-divider {
        width: 100%;
        height: 1;
        border-top: solid $primary-darken-3;
        margin: 1 0;
    }
    
    BasicInfoStepScreen .progress-section {
        margin: 1 0;
        padding: 1 0;
        border-top: solid $primary-darken-3;
    }
    
    BasicInfoStepScreen .progress-text {
        color: $text-muted;
    }
    
    BasicInfoStepScreen .progress-bar {
        color: $primary;
    }
    
    BasicInfoStepScreen .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    BasicInfoStepScreen .nav-button {
        margin: 0 1;
        min-width: 14;
    }
    
    BasicInfoStepScreen Input.-valid {
        border: tall $success;
    }
    
    BasicInfoStepScreen Input.-invalid {
        border: tall $error;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, state: BackendWizardState):
        """
        Initialize the basic info screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
    
    def compose(self) -> ComposeResult:
        """Compose the basic info screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "📝 Add Custom Backend - Basic Information",
                    classes="wizard-title"
                )
                
                yield Static(
                    "Please provide basic information about your backend:",
                    classes="field-hint"
                )
                
                with ScrollableContainer(classes="form-container"):
                    # Backend Name
                    with Vertical(classes="form-field"):
                        yield Label(
                            "Backend Name (internal identifier) *",
                            classes="field-label"
                        )
                        yield Input(
                            placeholder="my_quantum_backend",
                            value=self.state.backend_name,
                            validators=[BackendNameValidator()],
                            id="input_backend_name",
                            classes="field-input"
                        )
                        yield Static(
                            "ℹ️ Must be lowercase, no spaces (use underscores)",
                            classes="field-hint"
                        )
                    
                    # Display Name
                    with Vertical(classes="form-field"):
                        yield Label(
                            "Display Name (shown in UI) *",
                            classes="field-label"
                        )
                        yield Input(
                            placeholder="My Quantum Backend",
                            value=self.state.display_name,
                            id="input_display_name",
                            classes="field-input"
                        )
                    
                    # Version
                    with Vertical(classes="form-field"):
                        yield Label(
                            "Version *",
                            classes="field-label"
                        )
                        yield Input(
                            placeholder="1.0.0",
                            value=self.state.version or "1.0.0",
                            validators=[VersionValidator()],
                            id="input_version",
                            classes="field-input"
                        )
                    
                    # Description
                    with Vertical(classes="form-field"):
                        yield Label(
                            "Description",
                            classes="field-label"
                        )
                        yield Input(
                            placeholder="A custom quantum simulator backend for...",
                            value=self.state.description,
                            id="input_description",
                            classes="field-input"
                        )
                    
                    yield Static(classes="section-divider")
                    
                    # Library Name (conditional on backend type)
                    if self.state.backend_type in ["python_library", None]:
                        with Vertical(classes="form-field"):
                            yield Label(
                                "Python Library/Module Name",
                                classes="field-label"
                            )
                            yield Input(
                                placeholder="my_quantum_lib",
                                value=self.state.library_name,
                                id="input_library_name",
                                classes="field-input"
                            )
                            yield Static(
                                "ℹ️ The Python package to import (e.g., 'qiskit', 'cirq')",
                                classes="field-hint"
                            )
                    
                    # Author
                    with Vertical(classes="form-field"):
                        yield Label(
                            "Author/Maintainer (optional)",
                            classes="field-label"
                        )
                        yield Input(
                            placeholder="Your Name <email@example.com>",
                            value=self.state.author,
                            id="input_author",
                            classes="field-input"
                        )
                
                # Progress indicator
                with Vertical(classes="progress-section"):
                    yield Static(
                        "Progress: Step 2 of 7",
                        classes="progress-text"
                    )
                    yield Static(
                        "███░░░░ 29%",
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
                        "Next: Capabilities →",
                        id="btn_next",
                        variant="primary",
                        classes="nav-button"
                    )
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input field changes."""
        input_id = event.input.id
        value = event.value
        
        if input_id == "input_backend_name":
            # Store lowercase version
            self.state.backend_name = value.lower().strip()
            
            # Auto-generate display name if empty
            display_input = self.query_one("#input_display_name", Input)
            if not display_input.value:
                auto_display = value.replace('_', ' ').title()
                display_input.value = auto_display
                self.state.display_name = auto_display
        
        elif input_id == "input_display_name":
            self.state.display_name = value.strip()
        
        elif input_id == "input_version":
            self.state.version = value.strip()
        
        elif input_id == "input_description":
            self.state.description = value.strip()
        
        elif input_id == "input_library_name":
            self.state.library_name = value.strip()
        
        elif input_id == "input_author":
            self.state.author = value.strip()
    
    def _validate_form(self) -> tuple[bool, str]:
        """
        Validate all form fields.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if not self.state.backend_name:
            return False, "Backend name is required"
        
        if not self.state.display_name:
            return False, "Display name is required"
        
        if not self.state.version:
            return False, "Version is required"
        
        # Validate backend name format
        backend_name_input = self.query_one("#input_backend_name", Input)
        if not backend_name_input.is_valid:
            return False, "Invalid backend name format"
        
        # Validate version format
        version_input = self.query_one("#input_version", Input)
        if not version_input.is_valid:
            return False, "Invalid version format (use X.Y.Z)"
        
        return True, ""
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_back":
            self.state.current_step = 1
            self.dismiss({"action": "back", "state": self.state})
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_next":
            is_valid, error_msg = self._validate_form()
            
            if not is_valid:
                self.notify(error_msg, severity="warning")
                return
            
            self.state.current_step = 3
            self.dismiss({"action": "next", "state": self.state})
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss({"action": "cancel"})
