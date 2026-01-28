"""Step 3: Capabilities Configuration.

Configure what the backend can do:
- Simulator types (state vector, density matrix, etc.)
- Maximum qubits supported
- Additional features (noise, GPU, batching, etc.)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import Static, Button, Input, Checkbox, Select, Label
from textual.screen import ModalScreen

from .wizard_state import BackendWizardState


class CapabilitiesStepScreen(ModalScreen[dict]):
    """
    Step 3: Capabilities configuration screen.
    
    Allows users to specify what their backend can do, including
    simulation types, qubit limits, and additional features.
    """
    
    DEFAULT_CSS = """
    CapabilitiesStepScreen {
        align: center middle;
    }
    
    CapabilitiesStepScreen .wizard-container {
        width: 85;
        height: auto;
        max-height: 95%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    CapabilitiesStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    CapabilitiesStepScreen .form-container {
        height: auto;
        padding: 1;
    }
    
    CapabilitiesStepScreen .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }
    
    CapabilitiesStepScreen .checkbox-group {
        margin: 0 2;
        padding: 1;
    }
    
    CapabilitiesStepScreen .capability-checkbox {
        margin: 0 0 0 2;
    }
    
    CapabilitiesStepScreen .form-field {
        width: 100%;
        height: auto;
        margin: 1 0;
    }
    
    CapabilitiesStepScreen .field-label {
        color: $text;
        margin-bottom: 0;
    }
    
    CapabilitiesStepScreen .field-input {
        width: 100%;
        margin: 0;
    }
    
    CapabilitiesStepScreen .field-hint {
        color: $text-muted;
        margin-top: 0;
        margin-left: 2;
    }
    
    CapabilitiesStepScreen .section-divider {
        width: 100%;
        height: 1;
        border-top: solid $primary-darken-3;
        margin: 1 0;
    }
    
    CapabilitiesStepScreen .progress-section {
        margin: 1 0;
        padding: 1 0;
        border-top: solid $primary-darken-3;
    }
    
    CapabilitiesStepScreen .progress-text {
        color: $text-muted;
    }
    
    CapabilitiesStepScreen .progress-bar {
        color: $primary;
    }
    
    CapabilitiesStepScreen .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    CapabilitiesStepScreen .nav-button {
        margin: 0 1;
        min-width: 14;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, state: BackendWizardState):
        """
        Initialize the capabilities screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
    
    def compose(self) -> ComposeResult:
        """Compose the capabilities screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "⚡ Add Custom Backend - Capabilities",
                    classes="wizard-title"
                )
                
                yield Static(
                    "Configure what your backend can do:",
                    classes="field-hint"
                )
                
                with ScrollableContainer(classes="form-container"):
                    # Simulator Types
                    yield Static(
                        "Simulator Types (select all that apply):",
                        classes="section-title"
                    )
                    
                    with Vertical(classes="checkbox-group"):
                        yield Checkbox(
                            "State Vector Simulation",
                            value="state_vector" in self.state.simulator_types,
                            id="cb_state_vector",
                            classes="capability-checkbox"
                        )
                        yield Static(
                            "Full quantum state representation",
                            classes="field-hint"
                        )
                        
                        yield Checkbox(
                            "Density Matrix Simulation",
                            value="density_matrix" in self.state.simulator_types,
                            id="cb_density_matrix",
                            classes="capability-checkbox"
                        )
                        yield Static(
                            "Mixed state and noise simulation",
                            classes="field-hint"
                        )
                        
                        yield Checkbox(
                            "Tensor Network Simulation",
                            value="tensor_network" in self.state.simulator_types,
                            id="cb_tensor_network",
                            classes="capability-checkbox"
                        )
                        yield Static(
                            "Efficient for specific circuit structures",
                            classes="field-hint"
                        )
                        
                        yield Checkbox(
                            "Custom Simulation Type",
                            value="custom" in self.state.simulator_types,
                            id="cb_custom_sim",
                            classes="capability-checkbox"
                        )
                    
                    yield Static(classes="section-divider")
                    
                    # Max Qubits
                    with Vertical(classes="form-field"):
                        yield Label(
                            "Maximum Qubits Supported:",
                            classes="field-label"
                        )
                        yield Input(
                            placeholder="20",
                            value=str(self.state.max_qubits),
                            type="integer",
                            id="input_max_qubits",
                            classes="field-input"
                        )
                        yield Static(
                            "ℹ️ Typical range: 10-30 for CPU, 30-50 for GPU",
                            classes="field-hint"
                        )
                    
                    yield Static(classes="section-divider")
                    
                    # Additional Features
                    yield Static(
                        "Additional Features:",
                        classes="section-title"
                    )
                    
                    with Vertical(classes="checkbox-group"):
                        yield Checkbox(
                            "Noise Model Support",
                            value=self.state.supports_noise,
                            id="cb_noise",
                            classes="capability-checkbox"
                        )
                        yield Static(
                            "Simulate realistic quantum noise",
                            classes="field-hint"
                        )
                        
                        yield Checkbox(
                            "GPU Acceleration",
                            value=self.state.supports_gpu,
                            id="cb_gpu",
                            classes="capability-checkbox"
                        )
                        yield Static(
                            "CUDA/OpenCL acceleration support",
                            classes="field-hint"
                        )
                        
                        yield Checkbox(
                            "Batch Execution",
                            value=self.state.supports_batching,
                            id="cb_batching",
                            classes="capability-checkbox"
                        )
                        yield Static(
                            "Execute multiple circuits in parallel",
                            classes="field-hint"
                        )
                        
                        yield Checkbox(
                            "Parameter Binding",
                            value=self.state.supports_parameter_binding,
                            id="cb_parameter_binding",
                            classes="capability-checkbox"
                        )
                        yield Static(
                            "Bind parameters at runtime",
                            classes="field-hint"
                        )
                        
                        yield Checkbox(
                            "Custom Gate Definitions",
                            value=self.state.supports_custom_gates,
                            id="cb_custom_gates",
                            classes="capability-checkbox"
                        )
                        yield Static(
                            "Define new quantum gates",
                            classes="field-hint"
                        )
                
                # Progress indicator
                with Vertical(classes="progress-section"):
                    yield Static(
                        "Progress: Step 3 of 7",
                        classes="progress-text"
                    )
                    yield Static(
                        "████░░░ 43%",
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
                        "Next: Gate Mapping →",
                        id="btn_next",
                        variant="primary",
                        classes="nav-button"
                    )
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes."""
        checkbox_id = event.checkbox.id
        is_checked = event.value
        
        # Simulator types
        sim_type_map = {
            "cb_state_vector": "state_vector",
            "cb_density_matrix": "density_matrix",
            "cb_tensor_network": "tensor_network",
            "cb_custom_sim": "custom",
        }
        
        if checkbox_id in sim_type_map:
            sim_type = sim_type_map[checkbox_id]
            if is_checked and sim_type not in self.state.simulator_types:
                self.state.simulator_types.append(sim_type)
            elif not is_checked and sim_type in self.state.simulator_types:
                self.state.simulator_types.remove(sim_type)
        
        # Features
        elif checkbox_id == "cb_noise":
            self.state.supports_noise = is_checked
        elif checkbox_id == "cb_gpu":
            self.state.supports_gpu = is_checked
        elif checkbox_id == "cb_batching":
            self.state.supports_batching = is_checked
        elif checkbox_id == "cb_parameter_binding":
            self.state.supports_parameter_binding = is_checked
        elif checkbox_id == "cb_custom_gates":
            self.state.supports_custom_gates = is_checked
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input field changes."""
        if event.input.id == "input_max_qubits":
            try:
                value = int(event.value) if event.value else 20
                self.state.max_qubits = max(1, min(100, value))
            except ValueError:
                self.state.max_qubits = 20
    
    def _validate_capabilities(self) -> tuple[bool, str]:
        """
        Validate capabilities configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.state.simulator_types:
            return False, "Please select at least one simulator type"
        
        if self.state.max_qubits < 1:
            return False, "Maximum qubits must be at least 1"
        
        return True, ""
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_back":
            self.state.current_step = 2
            self.dismiss({"action": "back", "state": self.state})
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_next":
            is_valid, error_msg = self._validate_capabilities()
            
            if not is_valid:
                self.notify(error_msg, severity="warning")
                return
            
            self.state.current_step = 4
            self.dismiss({"action": "next", "state": self.state})
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss({"action": "cancel"})
