"""Step 4: Gate Mapping Configuration.

Configure how Proxima gates map to the backend's gates:
- Automatic (use standard gate names)
- Template (use common backend templates like Qiskit, Cirq)
- Manual (define custom mappings)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import Static, Button, RadioButton, RadioSet, Select, DataTable
from textual.screen import ModalScreen

from .wizard_state import BackendWizardState


# Gate templates for common quantum frameworks
GATE_TEMPLATES = {
    "qiskit": {
        "name": "Qiskit-style",
        "gates": {
            "H": "h",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "S": "s",
            "T": "t",
            "Rx": "rx",
            "Ry": "ry",
            "Rz": "rz",
            "CNOT": "cx",
            "CZ": "cz",
            "SWAP": "swap",
            "TOFFOLI": "ccx",
        }
    },
    "cirq": {
        "name": "Cirq-style",
        "gates": {
            "H": "H",
            "X": "X",
            "Y": "Y",
            "Z": "Z",
            "S": "S",
            "T": "T",
            "Rx": "rx",
            "Ry": "ry",
            "Rz": "rz",
            "CNOT": "CNOT",
            "CZ": "CZ",
            "SWAP": "SWAP",
            "TOFFOLI": "TOFFOLI",
        }
    },
    "pennylane": {
        "name": "PennyLane-style",
        "gates": {
            "H": "Hadamard",
            "X": "PauliX",
            "Y": "PauliY",
            "Z": "PauliZ",
            "S": "S",
            "T": "T",
            "Rx": "RX",
            "Ry": "RY",
            "Rz": "RZ",
            "CNOT": "CNOT",
            "CZ": "CZ",
            "SWAP": "SWAP",
            "TOFFOLI": "Toffoli",
        }
    },
    "standard": {
        "name": "Standard (most simulators)",
        "gates": {
            "H": "H",
            "X": "X",
            "Y": "Y",
            "Z": "Z",
            "S": "S",
            "T": "T",
            "Rx": "Rx",
            "Ry": "Ry",
            "Rz": "Rz",
            "CNOT": "CNOT",
            "CZ": "CZ",
            "SWAP": "SWAP",
            "TOFFOLI": "TOFFOLI",
        }
    }
}


class GateMappingStepScreen(ModalScreen[dict]):
    """
    Step 4: Gate mapping configuration screen.
    
    Allows users to configure how Proxima gates map to their
    backend's gate implementation.
    """
    
    DEFAULT_CSS = """
    GateMappingStepScreen {
        align: center middle;
    }
    
    GateMappingStepScreen .wizard-container {
        width: 85;
        height: auto;
        max-height: 95%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    GateMappingStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    GateMappingStepScreen .form-container {
        height: auto;
        padding: 1;
    }
    
    GateMappingStepScreen .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }
    
    GateMappingStepScreen .radio-option {
        margin: 1 0;
        padding: 0 2;
    }
    
    GateMappingStepScreen .option-description {
        color: $text-muted;
        margin-left: 4;
        padding: 0 0 1 0;
    }
    
    GateMappingStepScreen .field-hint {
        color: $text-muted;
        margin: 1 0;
    }
    
    GateMappingStepScreen .template-select {
        margin: 1 2;
        width: 50%;
    }
    
    GateMappingStepScreen .gate-table {
        height: 12;
        margin: 1 2;
        border: solid $primary-darken-3;
    }
    
    GateMappingStepScreen .section-divider {
        width: 100%;
        height: 1;
        border-top: solid $primary-darken-3;
        margin: 1 0;
    }
    
    GateMappingStepScreen .progress-section {
        margin: 1 0;
        padding: 1 0;
        border-top: solid $primary-darken-3;
    }
    
    GateMappingStepScreen .progress-text {
        color: $text-muted;
    }
    
    GateMappingStepScreen .progress-bar {
        color: $primary;
    }
    
    GateMappingStepScreen .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    GateMappingStepScreen .nav-button {
        margin: 0 1;
        min-width: 14;
    }
    
    GateMappingStepScreen .info-box {
        background: $primary-darken-3;
        padding: 1;
        margin: 1 0;
        border: solid $primary;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, state: BackendWizardState):
        """
        Initialize the gate mapping screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
        self.current_template = "standard"
    
    def compose(self) -> ComposeResult:
        """Compose the gate mapping screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "🔗 Add Custom Backend - Gate Mapping",
                    classes="wizard-title"
                )
                
                yield Static(
                    "Configure how Proxima gates map to your backend:",
                    classes="field-hint"
                )
                
                with ScrollableContainer(classes="form-container"):
                    # Mapping mode selection
                    yield Static(
                        "Gate Mapping Mode:",
                        classes="section-title"
                    )
                    
                    with RadioSet(id="mapping_mode_radio"):
                        # Automatic option
                        with Vertical(classes="radio-option"):
                            yield RadioButton(
                                "Automatic (recommended)",
                                value=True,
                                id="mode_automatic"
                            )
                            yield Static(
                                "Use standard gate names (H, X, Y, Z, CNOT, etc.)",
                                classes="option-description"
                            )
                        
                        # Template option
                        with Vertical(classes="radio-option"):
                            yield RadioButton(
                                "Use Template",
                                id="mode_template"
                            )
                            yield Static(
                                "Select from common backend templates (Qiskit, Cirq, etc.)",
                                classes="option-description"
                            )
                        
                        # Manual option
                        with Vertical(classes="radio-option"):
                            yield RadioButton(
                                "Manual Mapping",
                                id="mode_manual"
                            )
                            yield Static(
                                "Define custom gate mappings (advanced)",
                                classes="option-description"
                            )
                    
                    yield Static(classes="section-divider")
                    
                    # Template selection (shown when template mode selected)
                    yield Static(
                        "Select Template:",
                        classes="section-title",
                        id="template_label"
                    )
                    
                    yield Select(
                        options=[
                            (t["name"], key)
                            for key, t in GATE_TEMPLATES.items()
                        ],
                        value="standard",
                        id="template_select",
                        classes="template-select"
                    )
                    
                    # Gate mapping preview
                    yield Static(
                        "Gate Mapping Preview:",
                        classes="section-title"
                    )
                    
                    yield DataTable(
                        id="gate_table",
                        classes="gate-table"
                    )
                    
                    # Info box
                    yield Static(
                        "ℹ️ Standard gates will be automatically mapped:\n"
                        "  • Single-qubit: H, X, Y, Z, S, T, Rx, Ry, Rz\n"
                        "  • Two-qubit: CNOT, CZ, SWAP\n"
                        "  • Three-qubit: TOFFOLI, FREDKIN\n\n"
                        "You can customize individual gate mappings in the code\n"
                        "template step if needed.",
                        classes="info-box"
                    )
                
                # Progress indicator
                with Vertical(classes="progress-section"):
                    yield Static(
                        "Progress: Step 4 of 7",
                        classes="progress-text"
                    )
                    yield Static(
                        "█████░░ 57%",
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
                        "Next: Code Template →",
                        id="btn_next",
                        variant="primary",
                        classes="nav-button"
                    )
    
    def on_mount(self) -> None:
        """Initialize the gate mapping table."""
        self._setup_gate_table()
        self._update_gate_table("standard")
    
    def _setup_gate_table(self) -> None:
        """Set up the gate mapping table structure."""
        table = self.query_one("#gate_table", DataTable)
        table.add_columns("Proxima Gate", "Backend Gate", "Category")
    
    def _update_gate_table(self, template_key: str) -> None:
        """Update the gate table with mappings from a template."""
        table = self.query_one("#gate_table", DataTable)
        table.clear()
        
        template = GATE_TEMPLATES.get(template_key, GATE_TEMPLATES["standard"])
        
        # Categorize gates
        single_qubit = ["H", "X", "Y", "Z", "S", "T", "Rx", "Ry", "Rz"]
        two_qubit = ["CNOT", "CZ", "SWAP"]
        three_qubit = ["TOFFOLI"]
        
        for proxima_gate, backend_gate in template["gates"].items():
            if proxima_gate in single_qubit:
                category = "Single-qubit"
            elif proxima_gate in two_qubit:
                category = "Two-qubit"
            elif proxima_gate in three_qubit:
                category = "Three-qubit"
            else:
                category = "Other"
            
            table.add_row(proxima_gate, backend_gate, category)
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle mapping mode selection change."""
        mode_map = {
            "mode_automatic": "automatic",
            "mode_template": "template",
            "mode_manual": "manual",
        }
        
        if event.pressed:
            mode = mode_map.get(event.pressed.id, "automatic")
            self.state.gate_mapping_mode = mode
            
            # Show/hide template selector based on mode
            template_label = self.query_one("#template_label", Static)
            template_select = self.query_one("#template_select", Select)
            
            if mode == "template":
                template_label.display = True
                template_select.display = True
            else:
                template_label.display = mode == "automatic"
                template_select.display = mode == "automatic"
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle template selection change."""
        if event.select.id == "template_select":
            self.state.gate_template = event.value
            self._update_gate_table(event.value)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_back":
            self.state.current_step = 3
            self.dismiss({"action": "back", "state": self.state})
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_next":
            # Store selected gate mappings
            template_key = self.state.gate_template or "standard"
            template = GATE_TEMPLATES.get(template_key, GATE_TEMPLATES["standard"])
            
            # Convert to GateMapping objects
            from .wizard_state import GateMapping
            for proxima_gate, backend_gate in template["gates"].items():
                self.state.gate_mappings[proxima_gate] = GateMapping(
                    proxima_gate=proxima_gate,
                    backend_gate=backend_gate
                )
            
            self.state.current_step = 5
            self.dismiss({"action": "next", "state": self.state})
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss({"action": "cancel"})
