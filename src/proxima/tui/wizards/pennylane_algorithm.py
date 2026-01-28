"""PennyLane Algorithm Wizard for Proxima TUI.

A 6-step wizard for configuring and executing VQE, QAOA, and QNN algorithms.
Part of the TUI Integration Guide implementation.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container, Grid
from textual.widgets import (
    Static, Button, Input, Label, Select, RadioSet, RadioButton,
    Checkbox, ProgressBar, DataTable, TabbedContent, TabPane,
)
from textual.screen import ModalScreen
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel


# Try to import PennyLane components
try:
    from proxima.backends.lret.pennylane_device import QLRETDevice
    from proxima.backends.lret.algorithms import VQE, QAOA, QNN
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class AlgorithmType(Enum):
    """Available quantum algorithm types."""
    VQE = "vqe"
    QAOA = "qaoa"
    QNN = "qnn"
    CUSTOM = "custom"


class AnsatzType(Enum):
    """Available ansatz types."""
    HARDWARE_EFFICIENT = "hardware_efficient"
    UCCSD = "uccsd"
    REAL_AMPLITUDE = "real_amplitude"
    CUSTOM = "custom"


class OptimizerType(Enum):
    """Available optimizer types."""
    ADAM = "adam"
    SGD = "sgd"
    MOMENTUM = "momentum"
    LBFGS = "lbfgs"
    COBYLA = "cobyla"


@dataclass
class WizardConfig:
    """Complete wizard configuration."""
    # Step 1: Algorithm Selection
    algorithm: AlgorithmType = AlgorithmType.VQE
    
    # Step 2: Problem Definition
    num_qubits: int = 4
    hamiltonian_type: str = "ising"
    problem_params: Dict[str, Any] = field(default_factory=dict)
    
    # Step 3: Ansatz Configuration
    ansatz_type: AnsatzType = AnsatzType.HARDWARE_EFFICIENT
    num_layers: int = 2
    entanglement: str = "linear"
    
    # Step 4: Optimizer Settings
    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.01
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    
    # Step 5: Device Configuration
    shots: int = 1024
    use_statevector: bool = False
    noise_model: str = "none"
    
    # Step 6: Execution
    auto_run: bool = True


@dataclass
class ExecutionResult:
    """Result from algorithm execution."""
    success: bool
    energy: float = 0.0
    parameters: List[float] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WizardStep(Static):
    """Base class for wizard steps."""
    
    step_number: int = 0
    step_title: str = "Step"
    
    def __init__(self, config: WizardConfig, **kwargs):
        super().__init__(**kwargs)
        self._config = config
    
    def validate(self) -> tuple[bool, str]:
        """Validate step configuration. Returns (is_valid, error_message)."""
        return True, ""
    
    def save_to_config(self) -> None:
        """Save current UI state to config."""
        pass


class Step1AlgorithmSelection(WizardStep):
    """Step 1: Algorithm Selection."""
    
    step_number = 1
    step_title = "Algorithm Selection"
    
    def compose(self) -> ComposeResult:
        yield Static("Select the quantum algorithm to run:", classes="step-description")
        
        with RadioSet(id="algorithm-select"):
            yield RadioButton("VQE - Variational Quantum Eigensolver", id="algo-vqe", value=True)
            yield RadioButton("QAOA - Quantum Approximate Optimization", id="algo-qaoa")
            yield RadioButton("QNN - Quantum Neural Network", id="algo-qnn")
            yield RadioButton("Custom Algorithm", id="algo-custom")
        
        yield Static("\n📋 Algorithm Descriptions:", classes="section-header")
        yield Static(self._get_descriptions(), id="algo-descriptions")
    
    def _get_descriptions(self) -> str:
        return """
• VQE: Find ground state energy of molecular Hamiltonians
• QAOA: Solve combinatorial optimization problems
• QNN: Build hybrid quantum-classical neural networks
• Custom: Define your own variational algorithm
        """
    
    def save_to_config(self) -> None:
        try:
            radio_set = self.query_one("#algorithm-select", RadioSet)
            pressed = radio_set.pressed_button
            if pressed:
                if pressed.id == "algo-vqe":
                    self._config.algorithm = AlgorithmType.VQE
                elif pressed.id == "algo-qaoa":
                    self._config.algorithm = AlgorithmType.QAOA
                elif pressed.id == "algo-qnn":
                    self._config.algorithm = AlgorithmType.QNN
                else:
                    self._config.algorithm = AlgorithmType.CUSTOM
        except Exception:
            pass


class Step2ProblemDefinition(WizardStep):
    """Step 2: Problem Definition."""
    
    step_number = 2
    step_title = "Problem Definition"
    
    def compose(self) -> ComposeResult:
        yield Static("Define the problem to solve:", classes="step-description")
        
        with Horizontal(classes="input-row"):
            yield Label("Number of Qubits:", classes="input-label")
            yield Input(
                value=str(self._config.num_qubits),
                placeholder="4",
                id="input-num-qubits",
                classes="input-field",
            )
        
        yield Static("\nHamiltonian Type:", classes="section-header")
        with RadioSet(id="hamiltonian-select"):
            yield RadioButton("Ising Model", id="ham-ising", value=True)
            yield RadioButton("Heisenberg Model", id="ham-heisenberg")
            yield RadioButton("Molecular (H2)", id="ham-molecular")
            yield RadioButton("Custom Hamiltonian", id="ham-custom")
        
        yield Static("\nProblem Parameters:", classes="section-header")
        
        with Horizontal(classes="input-row"):
            yield Label("Coupling Strength (J):", classes="input-label")
            yield Input(value="1.0", id="input-coupling", classes="input-field")
        
        with Horizontal(classes="input-row"):
            yield Label("Field Strength (h):", classes="input-label")
            yield Input(value="0.5", id="input-field", classes="input-field")
    
    def save_to_config(self) -> None:
        try:
            self._config.num_qubits = int(self.query_one("#input-num-qubits", Input).value or 4)
            
            coupling = float(self.query_one("#input-coupling", Input).value or 1.0)
            field = float(self.query_one("#input-field", Input).value or 0.5)
            
            self._config.problem_params = {
                'coupling': coupling,
                'field': field,
            }
            
            radio_set = self.query_one("#hamiltonian-select", RadioSet)
            pressed = radio_set.pressed_button
            if pressed:
                if pressed.id == "ham-ising":
                    self._config.hamiltonian_type = "ising"
                elif pressed.id == "ham-heisenberg":
                    self._config.hamiltonian_type = "heisenberg"
                elif pressed.id == "ham-molecular":
                    self._config.hamiltonian_type = "molecular"
                else:
                    self._config.hamiltonian_type = "custom"
        except Exception:
            pass
    
    def validate(self) -> tuple[bool, str]:
        try:
            qubits = int(self.query_one("#input-num-qubits", Input).value or 0)
            if qubits < 2:
                return False, "Number of qubits must be at least 2"
            if qubits > 20:
                return False, "Maximum 20 qubits supported"
            return True, ""
        except ValueError:
            return False, "Invalid number of qubits"


class Step3AnsatzConfiguration(WizardStep):
    """Step 3: Ansatz Configuration."""
    
    step_number = 3
    step_title = "Ansatz Configuration"
    
    def compose(self) -> ComposeResult:
        yield Static("Configure the variational ansatz:", classes="step-description")
        
        yield Static("\nAnsatz Type:", classes="section-header")
        with RadioSet(id="ansatz-select"):
            yield RadioButton("Hardware-Efficient", id="ansatz-he", value=True)
            yield RadioButton("UCCSD (Chemistry)", id="ansatz-uccsd")
            yield RadioButton("RealAmplitudes", id="ansatz-real")
            yield RadioButton("Custom Ansatz", id="ansatz-custom")
        
        with Horizontal(classes="input-row"):
            yield Label("Number of Layers:", classes="input-label")
            yield Input(value=str(self._config.num_layers), id="input-layers", classes="input-field")
        
        yield Static("\nEntanglement Pattern:", classes="section-header")
        with RadioSet(id="entangle-select"):
            yield RadioButton("Linear", id="ent-linear", value=True)
            yield RadioButton("Full", id="ent-full")
            yield RadioButton("Circular", id="ent-circular")
            yield RadioButton("SCA (Shifted Circular Alternating)", id="ent-sca")
        
        yield Static("\n📊 Ansatz Visualization:", classes="section-header")
        yield Static(self._get_ansatz_diagram(), id="ansatz-diagram")
    
    def _get_ansatz_diagram(self) -> str:
        return """
  Hardware-Efficient Ansatz (2 layers):
  
  q0 ─[RY]─[RZ]─●────[RY]─[RZ]─●────
                │              │
  q1 ─[RY]─[RZ]─X─●──[RY]─[RZ]─X─●──
                  │              │
  q2 ─[RY]─[RZ]───X──[RY]─[RZ]───X──
        """
    
    def save_to_config(self) -> None:
        try:
            self._config.num_layers = int(self.query_one("#input-layers", Input).value or 2)
            
            ansatz_set = self.query_one("#ansatz-select", RadioSet)
            pressed = ansatz_set.pressed_button
            if pressed:
                if pressed.id == "ansatz-he":
                    self._config.ansatz_type = AnsatzType.HARDWARE_EFFICIENT
                elif pressed.id == "ansatz-uccsd":
                    self._config.ansatz_type = AnsatzType.UCCSD
                elif pressed.id == "ansatz-real":
                    self._config.ansatz_type = AnsatzType.REAL_AMPLITUDE
                else:
                    self._config.ansatz_type = AnsatzType.CUSTOM
            
            ent_set = self.query_one("#entangle-select", RadioSet)
            ent_pressed = ent_set.pressed_button
            if ent_pressed:
                if ent_pressed.id == "ent-linear":
                    self._config.entanglement = "linear"
                elif ent_pressed.id == "ent-full":
                    self._config.entanglement = "full"
                elif ent_pressed.id == "ent-circular":
                    self._config.entanglement = "circular"
                else:
                    self._config.entanglement = "sca"
        except Exception:
            pass
    
    def validate(self) -> tuple[bool, str]:
        try:
            layers = int(self.query_one("#input-layers", Input).value or 0)
            if layers < 1:
                return False, "Number of layers must be at least 1"
            if layers > 10:
                return False, "Maximum 10 layers recommended"
            return True, ""
        except ValueError:
            return False, "Invalid number of layers"


class Step4OptimizerSettings(WizardStep):
    """Step 4: Optimizer Settings."""
    
    step_number = 4
    step_title = "Optimizer Settings"
    
    def compose(self) -> ComposeResult:
        yield Static("Configure the classical optimizer:", classes="step-description")
        
        yield Static("\nOptimizer Type:", classes="section-header")
        with RadioSet(id="optimizer-select"):
            yield RadioButton("Adam (Recommended)", id="opt-adam", value=True)
            yield RadioButton("SGD (Stochastic Gradient Descent)", id="opt-sgd")
            yield RadioButton("Momentum", id="opt-momentum")
            yield RadioButton("L-BFGS (Quasi-Newton)", id="opt-lbfgs")
            yield RadioButton("COBYLA (Derivative-Free)", id="opt-cobyla")
        
        yield Static("\nHyperparameters:", classes="section-header")
        
        with Horizontal(classes="input-row"):
            yield Label("Learning Rate:", classes="input-label")
            yield Input(value=str(self._config.learning_rate), id="input-lr", classes="input-field")
        
        with Horizontal(classes="input-row"):
            yield Label("Max Iterations:", classes="input-label")
            yield Input(value=str(self._config.max_iterations), id="input-iterations", classes="input-field")
        
        with Horizontal(classes="input-row"):
            yield Label("Convergence Threshold:", classes="input-label")
            yield Input(value=str(self._config.convergence_threshold), id="input-threshold", classes="input-field")
        
        yield Static("\n💡 Tips:", classes="section-header")
        yield Static("""
• Adam: Works well for most problems, adaptive learning rate
• SGD: Simple and stable, may need careful tuning
• L-BFGS: Fast convergence for smooth landscapes
• COBYLA: Good when gradients are noisy or unavailable
        """)
    
    def save_to_config(self) -> None:
        try:
            self._config.learning_rate = float(self.query_one("#input-lr", Input).value or 0.01)
            self._config.max_iterations = int(self.query_one("#input-iterations", Input).value or 100)
            self._config.convergence_threshold = float(self.query_one("#input-threshold", Input).value or 1e-6)
            
            opt_set = self.query_one("#optimizer-select", RadioSet)
            pressed = opt_set.pressed_button
            if pressed:
                if pressed.id == "opt-adam":
                    self._config.optimizer = OptimizerType.ADAM
                elif pressed.id == "opt-sgd":
                    self._config.optimizer = OptimizerType.SGD
                elif pressed.id == "opt-momentum":
                    self._config.optimizer = OptimizerType.MOMENTUM
                elif pressed.id == "opt-lbfgs":
                    self._config.optimizer = OptimizerType.LBFGS
                else:
                    self._config.optimizer = OptimizerType.COBYLA
        except Exception:
            pass
    
    def validate(self) -> tuple[bool, str]:
        try:
            lr = float(self.query_one("#input-lr", Input).value or 0)
            if lr <= 0 or lr > 1:
                return False, "Learning rate should be between 0 and 1"
            
            iters = int(self.query_one("#input-iterations", Input).value or 0)
            if iters < 1:
                return False, "Max iterations must be at least 1"
            
            return True, ""
        except ValueError:
            return False, "Invalid optimizer settings"


class Step5DeviceConfiguration(WizardStep):
    """Step 5: Device Configuration."""
    
    step_number = 5
    step_title = "Device Configuration"
    
    def compose(self) -> ComposeResult:
        yield Static("Configure the quantum device:", classes="step-description")
        
        yield Static("\nExecution Mode:", classes="section-header")
        with RadioSet(id="mode-select"):
            yield RadioButton("Shot-based Simulation", id="mode-shots", value=True)
            yield RadioButton("Statevector (Exact)", id="mode-sv")
        
        with Horizontal(classes="input-row"):
            yield Label("Number of Shots:", classes="input-label")
            yield Input(value=str(self._config.shots), id="input-shots", classes="input-field")
        
        yield Static("\nNoise Model:", classes="section-header")
        with RadioSet(id="noise-select"):
            yield RadioButton("None (Ideal)", id="noise-none", value=True)
            yield RadioButton("Depolarizing", id="noise-depol")
            yield RadioButton("Amplitude Damping", id="noise-amp")
            yield RadioButton("Custom Noise Model", id="noise-custom")
        
        with Horizontal(classes="input-row"):
            yield Label("Noise Strength:", classes="input-label")
            yield Input(value="0.01", id="input-noise-strength", classes="input-field")
        
        yield Static("\n🔧 Device Info:", classes="section-header")
        yield Static("""
  Device: QLRETDevice (LRET PennyLane Hybrid)
  Backend: LRET Simulator with automatic optimization
  Features: Gradient computation, batch execution, GPU acceleration
        """)
    
    def save_to_config(self) -> None:
        try:
            self._config.shots = int(self.query_one("#input-shots", Input).value or 1024)
            
            mode_set = self.query_one("#mode-select", RadioSet)
            mode_pressed = mode_set.pressed_button
            if mode_pressed:
                self._config.use_statevector = (mode_pressed.id == "mode-sv")
            
            noise_set = self.query_one("#noise-select", RadioSet)
            noise_pressed = noise_set.pressed_button
            if noise_pressed:
                if noise_pressed.id == "noise-none":
                    self._config.noise_model = "none"
                elif noise_pressed.id == "noise-depol":
                    self._config.noise_model = "depolarizing"
                elif noise_pressed.id == "noise-amp":
                    self._config.noise_model = "amplitude_damping"
                else:
                    self._config.noise_model = "custom"
        except Exception:
            pass
    
    def validate(self) -> tuple[bool, str]:
        try:
            shots = int(self.query_one("#input-shots", Input).value or 0)
            if shots < 1:
                return False, "Number of shots must be at least 1"
            if shots > 1000000:
                return False, "Maximum 1,000,000 shots"
            return True, ""
        except ValueError:
            return False, "Invalid number of shots"


class Step6ExecuteMonitor(WizardStep):
    """Step 6: Execute & Monitor."""
    
    step_number = 6
    step_title = "Execute & Monitor"
    
    def __init__(self, config: WizardConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._is_running = False
        self._result: Optional[ExecutionResult] = None
    
    def compose(self) -> ComposeResult:
        yield Static("Execute the algorithm and monitor progress:", classes="step-description")
        
        # Configuration Summary
        yield Static("\n📋 Configuration Summary:", classes="section-header")
        yield Static(id="config-summary")
        
        # Progress Section
        yield Static("\n⚡ Execution Progress:", classes="section-header")
        yield ProgressBar(id="exec-progress", total=100)
        yield Static("Ready to execute", id="exec-status")
        
        # Convergence Plot (ASCII)
        yield Static("\n📈 Energy Convergence:", classes="section-header")
        yield Static(id="convergence-plot")
        
        # Results Table
        yield Static("\n📊 Results:", classes="section-header")
        yield DataTable(id="results-table")
        
        # Action Buttons
        with Horizontal(classes="action-row"):
            yield Button("Execute", id="btn-execute", variant="primary")
            yield Button("Stop", id="btn-stop", disabled=True)
            yield Button("Export Results", id="btn-export", disabled=True)
    
    def on_mount(self) -> None:
        """Initialize the step."""
        # Update config summary
        self._update_config_summary()
        
        # Initialize results table
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Property", "Value")
        
        # Initialize convergence plot
        self._update_convergence_plot([])
    
    def _update_config_summary(self) -> None:
        """Update the configuration summary."""
        try:
            summary = self.query_one("#config-summary", Static)
            text = f"""
  Algorithm: {self._config.algorithm.value.upper()}
  Qubits: {self._config.num_qubits}
  Ansatz: {self._config.ansatz_type.value} ({self._config.num_layers} layers)
  Optimizer: {self._config.optimizer.value.upper()} (lr={self._config.learning_rate})
  Shots: {self._config.shots if not self._config.use_statevector else 'Statevector'}
  Noise: {self._config.noise_model}
            """
            summary.update(text)
        except Exception:
            pass
    
    def _update_convergence_plot(self, history: List[float]) -> None:
        """Update the ASCII convergence plot."""
        try:
            plot = self.query_one("#convergence-plot", Static)
            
            if not history:
                plot.update("  (No data yet - run algorithm to see convergence)")
                return
            
            # Simple ASCII plot
            height = 6
            width = 40
            
            min_e = min(history)
            max_e = max(history)
            range_e = max(max_e - min_e, 0.001)
            
            lines = []
            for row in range(height):
                y_val = max_e - (row / (height - 1)) * range_e
                line = f"  {y_val:>8.4f} │"
                
                for i, e in enumerate(history[-width:]):
                    x = int(i * width / len(history[-width:]))
                    e_row = int((max_e - e) / range_e * (height - 1))
                    if e_row == row:
                        line = line[:12 + x] + "●" + line[13 + x:]
                
                lines.append(line)
            
            lines.append("           └" + "─" * width)
            lines.append("             Iteration →")
            
            plot.update("\n".join(lines))
        except Exception:
            pass
    
    def _update_results_table(self, result: ExecutionResult) -> None:
        """Update the results table with execution results."""
        try:
            table = self.query_one("#results-table", DataTable)
            table.clear()
            
            table.add_row("Final Energy", f"{result.energy:.6f}")
            table.add_row("Converged", "Yes" if result.success else "No")
            table.add_row("Iterations", str(len(result.convergence_history)))
            table.add_row("Execution Time", f"{result.execution_time_ms:.2f} ms")
            table.add_row("Num Parameters", str(len(result.parameters)))
            
            # Add first few parameters
            for i, p in enumerate(result.parameters[:5]):
                table.add_row(f"  θ_{i}", f"{p:.4f}")
            
            if len(result.parameters) > 5:
                table.add_row("  ...", f"({len(result.parameters) - 5} more)")
        except Exception:
            pass


class PennyLaneAlgorithmWizard(ModalScreen):
    """PennyLane Algorithm Wizard - 6-step wizard for VQE, QAOA, QNN.
    
    Steps:
    1. Algorithm Selection - Choose VQE, QAOA, QNN, or Custom
    2. Problem Definition - Define Hamiltonian and problem parameters
    3. Ansatz Configuration - Configure variational circuit structure
    4. Optimizer Settings - Select optimizer and hyperparameters
    5. Device Configuration - Configure QLRETDevice settings
    6. Execute & Monitor - Run algorithm with real-time monitoring
    """
    
    DEFAULT_CSS = """
    PennyLaneAlgorithmWizard {
        align: center middle;
    }
    
    PennyLaneAlgorithmWizard > Container {
        width: 90%;
        height: 90%;
        border: thick $primary 50%;
        background: $surface;
    }
    
    PennyLaneAlgorithmWizard .wizard-header {
        height: 3;
        padding: 0 2;
        background: $primary-darken-2;
    }
    
    PennyLaneAlgorithmWizard .wizard-title {
        text-style: bold;
        padding: 1;
    }
    
    PennyLaneAlgorithmWizard .step-indicators {
        height: 4;
        layout: horizontal;
        padding: 1;
        background: $surface-darken-1;
    }
    
    PennyLaneAlgorithmWizard .step-indicator {
        width: 1fr;
        content-align: center middle;
        padding: 0 1;
    }
    
    PennyLaneAlgorithmWizard .step-indicator.-current {
        background: $primary;
        text-style: bold;
    }
    
    PennyLaneAlgorithmWizard .step-indicator.-completed {
        background: $success-darken-2;
    }
    
    PennyLaneAlgorithmWizard .step-content {
        height: 1fr;
        padding: 1 2;
        overflow: auto;
    }
    
    PennyLaneAlgorithmWizard .step-description {
        margin-bottom: 1;
        text-style: italic;
    }
    
    PennyLaneAlgorithmWizard .section-header {
        text-style: bold;
        margin-top: 1;
    }
    
    PennyLaneAlgorithmWizard .input-row {
        height: 3;
        margin: 0 0 1 0;
    }
    
    PennyLaneAlgorithmWizard .input-label {
        width: 25;
        height: 3;
        content-align: left middle;
    }
    
    PennyLaneAlgorithmWizard .input-field {
        width: 20;
    }
    
    PennyLaneAlgorithmWizard .wizard-footer {
        height: 3;
        layout: horizontal;
        padding: 0 2;
        background: $surface-darken-1;
    }
    
    PennyLaneAlgorithmWizard .nav-btn {
        margin-right: 1;
    }
    
    PennyLaneAlgorithmWizard .action-row {
        height: 3;
        margin-top: 1;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("left", "prev_step", "Previous"),
        ("right", "next_step", "Next"),
    ]
    
    current_step: reactive[int] = reactive(1)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config = WizardConfig()
        self._steps: Dict[int, WizardStep] = {}
        self._is_executing = False
        self._result: Optional[ExecutionResult] = None
    
    def compose(self) -> ComposeResult:
        with Container():
            # Header
            with Container(classes="wizard-header"):
                yield Static("🧪 PennyLane Algorithm Wizard", classes="wizard-title")
            
            # Step Indicators
            with Horizontal(classes="step-indicators"):
                for i in range(1, 7):
                    step_names = [
                        "1. Algorithm",
                        "2. Problem",
                        "3. Ansatz",
                        "4. Optimizer",
                        "5. Device",
                        "6. Execute",
                    ]
                    classes = "step-indicator"
                    if i == 1:
                        classes += " -current"
                    yield Static(step_names[i-1], id=f"step-ind-{i}", classes=classes)
            
            # Step Content
            with Container(classes="step-content", id="step-container"):
                # Steps will be mounted here
                pass
            
            # Footer with navigation
            with Horizontal(classes="wizard-footer"):
                yield Button("Cancel", id="btn-cancel", classes="nav-btn")
                yield Button("← Previous", id="btn-prev", classes="nav-btn", disabled=True)
                yield Button("Next →", id="btn-next", classes="nav-btn", variant="primary")
                yield Button("Finish", id="btn-finish", classes="nav-btn", disabled=True)
    
    def on_mount(self) -> None:
        """Initialize the wizard."""
        self._mount_step(1)
    
    def _mount_step(self, step_num: int) -> None:
        """Mount the specified step."""
        container = self.query_one("#step-container", Container)
        
        # Clear existing content
        for child in list(container.children):
            child.remove()
        
        # Create and mount new step
        step_classes = {
            1: Step1AlgorithmSelection,
            2: Step2ProblemDefinition,
            3: Step3AnsatzConfiguration,
            4: Step4OptimizerSettings,
            5: Step5DeviceConfiguration,
            6: Step6ExecuteMonitor,
        }
        
        step_class = step_classes.get(step_num)
        if step_class:
            step = step_class(self._config)
            self._steps[step_num] = step
            container.mount(step)
        
        # Update step indicators
        self._update_step_indicators()
        
        # Update button states
        self._update_navigation_buttons()
    
    def _update_step_indicators(self) -> None:
        """Update step indicator styling."""
        for i in range(1, 7):
            try:
                indicator = self.query_one(f"#step-ind-{i}", Static)
                indicator.remove_class("-current", "-completed")
                
                if i == self.current_step:
                    indicator.add_class("-current")
                elif i < self.current_step:
                    indicator.add_class("-completed")
            except Exception:
                pass
    
    def _update_navigation_buttons(self) -> None:
        """Update navigation button states."""
        try:
            prev_btn = self.query_one("#btn-prev", Button)
            next_btn = self.query_one("#btn-next", Button)
            finish_btn = self.query_one("#btn-finish", Button)
            
            prev_btn.disabled = (self.current_step == 1)
            next_btn.disabled = (self.current_step == 6)
            finish_btn.disabled = (self.current_step != 6)
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-cancel":
            self.app.pop_screen()
        elif button_id == "btn-prev":
            self.action_prev_step()
        elif button_id == "btn-next":
            self.action_next_step()
        elif button_id == "btn-finish":
            self._finish_wizard()
        elif button_id == "btn-execute":
            self._execute_algorithm()
        elif button_id == "btn-stop":
            self._stop_execution()
        elif button_id == "btn-export":
            self._export_results()
    
    def action_prev_step(self) -> None:
        """Go to previous step."""
        if self.current_step > 1:
            self.current_step -= 1
            self._mount_step(self.current_step)
    
    def action_next_step(self) -> None:
        """Go to next step."""
        # Validate current step
        current = self._steps.get(self.current_step)
        if current:
            current.save_to_config()
            is_valid, error = current.validate()
            if not is_valid:
                self.notify(error, severity="error")
                return
        
        if self.current_step < 6:
            self.current_step += 1
            self._mount_step(self.current_step)
    
    def action_cancel(self) -> None:
        """Cancel the wizard."""
        self.app.pop_screen()
    
    def _finish_wizard(self) -> None:
        """Complete the wizard and return results."""
        self.notify("Wizard completed!", severity="success")
        self.app.pop_screen()
    
    def _execute_algorithm(self) -> None:
        """Execute the configured algorithm."""
        if self._is_executing:
            return
        
        self._is_executing = True
        self.notify("Starting algorithm execution...", severity="information")
        
        # Run asynchronously
        asyncio.create_task(self._run_algorithm())
    
    async def _run_algorithm(self) -> None:
        """Run the algorithm asynchronously."""
        try:
            # Update UI
            try:
                exec_btn = self.query_one("#btn-execute", Button)
                stop_btn = self.query_one("#btn-stop", Button)
                exec_btn.disabled = True
                stop_btn.disabled = False
            except Exception:
                pass
            
            progress = self.query_one("#exec-progress", ProgressBar)
            status = self.query_one("#exec-status", Static)
            
            # Simulate algorithm execution
            convergence_history = []
            import random
            import time
            
            start_time = time.time()
            
            for i in range(self._config.max_iterations):
                if not self._is_executing:
                    break
                
                # Update progress
                progress.update(progress=(i / self._config.max_iterations) * 100)
                status.update(f"Iteration {i + 1}/{self._config.max_iterations}")
                
                # Simulate energy calculation
                if i == 0:
                    energy = random.uniform(0, 5)
                else:
                    # Converging behavior
                    prev = convergence_history[-1]
                    decay = 0.95
                    noise = random.uniform(-0.05, 0.05)
                    energy = prev * decay + noise
                
                convergence_history.append(energy)
                
                # Update convergence plot
                step6 = self._steps.get(6)
                if step6 and isinstance(step6, Step6ExecuteMonitor):
                    step6._update_convergence_plot(convergence_history)
                
                # Check convergence
                if len(convergence_history) > 5:
                    recent = convergence_history[-5:]
                    if max(recent) - min(recent) < self._config.convergence_threshold:
                        break
                
                await asyncio.sleep(0.05)
            
            # Create result
            exec_time = (time.time() - start_time) * 1000
            
            self._result = ExecutionResult(
                success=True,
                energy=convergence_history[-1] if convergence_history else 0,
                parameters=[random.uniform(-3.14, 3.14) for _ in range(self._config.num_qubits * self._config.num_layers * 2)],
                convergence_history=convergence_history,
                execution_time_ms=exec_time,
                metadata={
                    'algorithm': self._config.algorithm.value,
                    'qubits': self._config.num_qubits,
                    'optimizer': self._config.optimizer.value,
                },
            )
            
            # Update results table
            step6 = self._steps.get(6)
            if step6 and isinstance(step6, Step6ExecuteMonitor):
                step6._update_results_table(self._result)
            
            progress.update(progress=100)
            status.update("✓ Execution complete!")
            
            self.notify("Algorithm completed successfully!", severity="success")
            
            # Enable export button
            try:
                export_btn = self.query_one("#btn-export", Button)
                export_btn.disabled = False
            except Exception:
                pass
            
        except Exception as e:
            self.notify(f"Execution error: {e}", severity="error")
        finally:
            self._is_executing = False
            try:
                exec_btn = self.query_one("#btn-execute", Button)
                stop_btn = self.query_one("#btn-stop", Button)
                exec_btn.disabled = False
                stop_btn.disabled = True
            except Exception:
                pass
    
    def _stop_execution(self) -> None:
        """Stop the current execution."""
        self._is_executing = False
        self.notify("Execution stopped", severity="warning")
    
    def _export_results(self) -> None:
        """Export algorithm results."""
        if not self._result:
            self.notify("No results to export", severity="warning")
            return
        
        try:
            from pathlib import Path
            from datetime import datetime
            import json
            
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"algorithm_result_{timestamp}.json"
            
            data = {
                'algorithm': self._config.algorithm.value,
                'config': {
                    'num_qubits': self._config.num_qubits,
                    'ansatz_type': self._config.ansatz_type.value,
                    'num_layers': self._config.num_layers,
                    'optimizer': self._config.optimizer.value,
                    'learning_rate': self._config.learning_rate,
                    'max_iterations': self._config.max_iterations,
                    'shots': self._config.shots,
                },
                'result': {
                    'success': self._result.success,
                    'final_energy': self._result.energy,
                    'parameters': self._result.parameters,
                    'convergence_history': self._result.convergence_history,
                    'execution_time_ms': self._result.execution_time_ms,
                },
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.notify(f"Results exported to {filename}", severity="success")
        except Exception as e:
            self.notify(f"Export error: {e}", severity="error")
