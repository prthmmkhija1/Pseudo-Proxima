"""Step 1: Welcome Screen.

Welcome screen for the backend addition wizard.
Introduces the wizard and lets users select the backend type.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center
from textual.widgets import Static, Button, RadioButton, RadioSet
from textual.screen import ModalScreen

from .wizard_state import BackendWizardState


class WelcomeStepScreen(ModalScreen[dict]):
    """
    Step 1: Welcome and backend type selection.
    
    This screen introduces the wizard and allows users to select
    the type of backend they want to create:
    - Python Library: Import existing Python quantum simulator
    - Command Line Tool: Execute external simulator via CLI
    - API Server: Connect to remote quantum API
    - Custom: Fully custom implementation
    """
    
    DEFAULT_CSS = """
    WelcomeStepScreen {
        align: center middle;
    }
    
    WelcomeStepScreen .wizard-container {
        width: 80;
        height: auto;
        max-height: 90%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    WelcomeStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    WelcomeStepScreen .welcome-text {
        width: 100%;
        margin: 1 0;
        color: $text;
        padding: 1;
    }
    
    WelcomeStepScreen .section-divider {
        width: 100%;
        height: 1;
        border-top: solid $primary-darken-3;
        margin: 1 0;
    }
    
    WelcomeStepScreen .backend-type-section {
        padding: 1;
    }
    
    WelcomeStepScreen .backend-type-option {
        margin: 1 0;
        padding: 0 2;
    }
    
    WelcomeStepScreen .option-description {
        color: $text-muted;
        margin-left: 4;
        padding: 0 0 1 0;
    }
    
    WelcomeStepScreen .progress-section {
        margin: 1 0;
        padding: 1;
        border-top: solid $primary-darken-3;
    }
    
    WelcomeStepScreen .progress-text {
        color: $text-muted;
    }
    
    WelcomeStepScreen .progress-bar {
        color: $primary;
    }
    
    WelcomeStepScreen .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    WelcomeStepScreen .nav-button {
        margin: 0 1;
        min-width: 16;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, state: BackendWizardState):
        """
        Initialize the welcome screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
    
    def compose(self) -> ComposeResult:
        """Compose the welcome screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "🚀 Add Custom Backend - Welcome",
                    classes="wizard-title"
                )
                
                # Welcome message
                yield Static(
                    "Welcome to the Custom Backend Addition Wizard!\n\n"
                    "This wizard will guide you through creating a new quantum\n"
                    "simulator backend for Proxima in 7 easy steps.\n\n"
                    "✨ No coding required - just answer a few questions!",
                    classes="welcome-text"
                )
                
                yield Static(classes="section-divider")
                
                # Backend type selection
                yield Static(
                    "📦 Select your backend type:",
                    classes="welcome-text"
                )
                
                with Vertical(classes="backend-type-section"):
                    with RadioSet(id="backend_type_radio"):
                        # Python Library option
                        with Vertical(classes="backend-type-option"):
                            yield RadioButton(
                                "Python Library",
                                value=True,
                                id="type_python"
                            )
                            yield Static(
                                "Import and use an existing Python quantum simulator\n"
                                "Example: pyQuEST, ProjectQ, QuTiP",
                                classes="option-description"
                            )
                        
                        # Command Line option
                        with Vertical(classes="backend-type-option"):
                            yield RadioButton(
                                "Command Line Tool",
                                id="type_cli"
                            )
                            yield Static(
                                "Execute external quantum simulator via command line\n"
                                "Example: QuEST binary, custom C++ simulator",
                                classes="option-description"
                            )
                        
                        # API Server option
                        with Vertical(classes="backend-type-option"):
                            yield RadioButton(
                                "API Server",
                                id="type_api"
                            )
                            yield Static(
                                "Connect to a remote quantum simulator API\n"
                                "Example: IBM Quantum Cloud, AWS Braket",
                                classes="option-description"
                            )
                        
                        # Custom option
                        with Vertical(classes="backend-type-option"):
                            yield RadioButton(
                                "Custom Implementation",
                                id="type_custom"
                            )
                            yield Static(
                                "Fully custom backend with manual code entry",
                                classes="option-description"
                            )
                
                # Progress indicator
                with Vertical(classes="progress-section"):
                    yield Static(
                        "Progress: Step 1 of 7",
                        classes="progress-text"
                    )
                    yield Static(
                        "█░░░░░░ 14%",
                        classes="progress-bar"
                    )
                
                # Navigation buttons
                with Horizontal(classes="button-container"):
                    yield Button(
                        "Cancel",
                        id="btn_cancel",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Next: Basic Info →",
                        id="btn_next",
                        variant="primary",
                        classes="nav-button"
                    )
    
    def on_mount(self) -> None:
        """Handle screen mount."""
        # Pre-select if state has a value
        if self.state.backend_type:
            self._select_backend_type(self.state.backend_type)
    
    def _select_backend_type(self, backend_type: str) -> None:
        """Select a backend type in the radio set."""
        type_map = {
            "python_library": "type_python",
            "command_line": "type_cli",
            "api_server": "type_api",
            "custom": "type_custom",
        }
        
        radio_id = type_map.get(backend_type)
        if radio_id:
            try:
                radio = self.query_one(f"#{radio_id}", RadioButton)
                radio.value = True
            except Exception:
                pass
    
    def _get_selected_backend_type(self) -> str | None:
        """Get the currently selected backend type."""
        try:
            radio_set = self.query_one("#backend_type_radio", RadioSet)
            pressed = radio_set.pressed_button
            
            if pressed:
                type_map = {
                    "type_python": "python_library",
                    "type_cli": "command_line",
                    "type_api": "api_server",
                    "type_custom": "custom",
                }
                return type_map.get(pressed.id)
        except Exception:
            pass
        
        return None
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle backend type selection change."""
        backend_type = self._get_selected_backend_type()
        if backend_type:
            self.state.backend_type = backend_type
            self.state.can_proceed = True
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_next":
            backend_type = self._get_selected_backend_type()
            
            if not backend_type:
                self.notify(
                    "Please select a backend type to continue",
                    severity="warning"
                )
                return
            
            self.state.backend_type = backend_type
            self.state.current_step = 2
            self.dismiss({"action": "next", "state": self.state})
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss({"action": "cancel"})
