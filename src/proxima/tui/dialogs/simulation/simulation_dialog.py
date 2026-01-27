"""Simulation dialog for Proxima TUI.

Dialog for configuring and starting a quantum simulation.
"""

from dataclasses import dataclass
from typing import Optional, List

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, Button, Select, Switch
from textual import on

from ...styles.theme import get_theme


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    
    description: str = ""
    backend: str = "auto"
    qubits: int = 2
    shots: int = 1024
    use_llm: bool = False
    circuit_type: str = "bell"


class SimulationDialog(ModalScreen):
    """Dialog for configuring and starting a simulation.
    
    Allows users to set up:
    - Description/task
    - Backend selection
    - Number of qubits
    - Number of shots
    - Circuit type
    """
    
    DEFAULT_CSS = """
    SimulationDialog {
        align: center middle;
    }
    
    SimulationDialog > .dialog-container {
        width: 70;
        max-height: 80%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    
    SimulationDialog .dialog-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }
    
    SimulationDialog .form-section {
        margin: 1 0;
    }
    
    SimulationDialog .form-row {
        layout: horizontal;
        height: auto;
        margin: 0 0 1 0;
    }
    
    SimulationDialog .form-label {
        width: 20;
        padding-right: 1;
        color: $text-muted;
    }
    
    SimulationDialog .form-input {
        width: 1fr;
    }
    
    SimulationDialog .buttons {
        layout: horizontal;
        height: auto;
        margin-top: 1;
        align: center middle;
        padding-top: 1;
        border-top: solid $primary-darken-3;
    }
    
    SimulationDialog .btn {
        margin: 0 1;
        min-width: 15;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(
        self,
        default_config: Optional[SimulationConfig] = None,
        **kwargs,
    ):
        """Initialize the simulation dialog.
        
        Args:
            default_config: Default configuration values
        """
        super().__init__(**kwargs)
        self.config = default_config or SimulationConfig()
    
    def compose(self):
        """Compose the dialog layout."""
        with Vertical(classes="dialog-container"):
            yield Static("🚀 New Simulation", classes="dialog-title")
            
            with Vertical(classes="form-section"):
                # Description
                with Horizontal(classes="form-row"):
                    yield Static("Description:", classes="form-label")
                    yield Input(
                        value=self.config.description,
                        placeholder="e.g., 'Create a Bell state' or 'Run Grover search'",
                        id="input-description",
                        classes="form-input",
                    )
                
                # Circuit type
                with Horizontal(classes="form-row"):
                    yield Static("Circuit Type:", classes="form-label")
                    yield Select(
                        [
                            ("Bell State", "bell"),
                            ("GHZ State", "ghz"),
                            ("Quantum Teleportation", "teleport"),
                            ("Grover Search", "grover"),
                            ("Quantum Fourier Transform", "qft"),
                            ("VQE Ansatz", "vqe"),
                            ("Custom (LLM)", "custom"),
                        ],
                        value=self.config.circuit_type,
                        id="select-circuit",
                        classes="form-input",
                    )
                
                # Backend
                with Horizontal(classes="form-row"):
                    yield Static("Backend:", classes="form-label")
                    yield Select(
                        [
                            ("Auto (Recommended)", "auto"),
                            ("LRET", "lret"),
                            ("Cirq", "cirq"),
                            ("Qiskit Aer", "qiskit_aer"),
                            ("QuEST", "quest"),
                            ("qsim", "qsim"),
                            ("cuQuantum (GPU)", "cuquantum"),
                        ],
                        value=self.config.backend,
                        id="select-backend",
                        classes="form-input",
                    )
                
                # Qubits
                with Horizontal(classes="form-row"):
                    yield Static("Qubits:", classes="form-label")
                    yield Input(
                        value=str(self.config.qubits),
                        placeholder="2",
                        id="input-qubits",
                        classes="form-input",
                    )
                
                # Shots
                with Horizontal(classes="form-row"):
                    yield Static("Shots:", classes="form-label")
                    yield Input(
                        value=str(self.config.shots),
                        placeholder="1024",
                        id="input-shots",
                        classes="form-input",
                    )
                
                # Use LLM
                with Horizontal(classes="form-row"):
                    yield Static("Use LLM Assistance:", classes="form-label")
                    yield Switch(value=self.config.use_llm, id="switch-llm")
            
            with Horizontal(classes="buttons"):
                yield Button(
                    "🚀 Run Simulation",
                    id="btn-run",
                    classes="btn",
                    variant="primary",
                )
                yield Button(
                    "Cancel",
                    id="btn-cancel",
                    classes="btn",
                )
    
    def on_mount(self) -> None:
        """Focus the description input on mount."""
        self.query_one("#input-description", Input).focus()
    
    @on(Input.Submitted)
    def submit_on_enter(self, event: Input.Submitted) -> None:
        """Submit on Enter key in input fields."""
        self._submit()
    
    def action_cancel(self) -> None:
        """Cancel the dialog."""
        self.dismiss(None)
    
    def _validate_inputs(self) -> Optional[str]:
        """Validate input values.
        
        Returns:
            Error message if validation fails, None if valid
        """
        try:
            qubits = int(self.query_one("#input-qubits", Input).value)
            if qubits < 1 or qubits > 30:
                return "Qubits must be between 1 and 30"
        except ValueError:
            return "Invalid qubits value (must be a number)"
        
        try:
            shots = int(self.query_one("#input-shots", Input).value)
            if shots < 1:
                return "Shots must be at least 1"
        except ValueError:
            return "Invalid shots value (must be a number)"
        
        return None
    
    def _submit(self) -> None:
        """Submit the simulation configuration."""
        error = self._validate_inputs()
        if error:
            self.notify(error, severity="error")
            return
        
        # Build config
        config = SimulationConfig(
            description=self.query_one("#input-description", Input).value,
            circuit_type=self.query_one("#select-circuit", Select).value,
            backend=self.query_one("#select-backend", Select).value,
            qubits=int(self.query_one("#input-qubits", Input).value),
            shots=int(self.query_one("#input-shots", Input).value),
            use_llm=self.query_one("#switch-llm", Switch).value,
        )
        
        self.dismiss(config)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-run":
            self._submit()
        else:
            self.action_cancel()
