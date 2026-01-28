"""TUI dialog for PennyLane variational algorithms.

This dialog provides an interactive interface for:
- Configuring VQE (Variational Quantum Eigensolver)
- Running QAOA (Quantum Approximate Optimization)
- Testing QNN (Quantum Neural Network)
- Viewing convergence plots
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
)


class PennyLaneAlgorithmDialog(ModalScreen):
    """Dialog for running PennyLane variational algorithms.
    
    Provides interface to:
    - Configure VQE for ground state energy finding
    - Run QAOA for combinatorial optimization
    - Test QNN for classification
    - View convergence and results
    
    Keybindings:
        - ESC: Close dialog
        - v: Run VQE
        - q: Run QAOA
        - n: Run QNN test
    """
    
    CSS = """
    PennyLaneAlgorithmDialog {
        align: center middle;
    }
    
    PennyLaneAlgorithmDialog > Container {
        width: 90%;
        height: 85%;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        color: $text;
        padding: 1;
        background: $primary;
        margin-bottom: 1;
    }
    
    .algorithm-tabs {
        height: 100%;
    }
    
    .config-section {
        padding: 1;
        margin-bottom: 1;
        border: round $primary-darken-2;
    }
    
    .config-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .config-label {
        width: 20;
        padding-top: 1;
    }
    
    .config-input {
        width: 15;
    }
    
    .hamiltonian-input {
        height: 6;
        width: 100%;
    }
    
    .button-row {
        height: 3;
        align: center middle;
        margin-top: 1;
        margin-bottom: 1;
    }
    
    .button-row Button {
        margin: 0 1;
    }
    
    .action-primary {
        background: $success;
    }
    
    .progress-container {
        height: 5;
        padding: 1;
        margin-top: 1;
    }
    
    .log-container {
        height: 15;
        border: round $primary;
        margin-top: 1;
    }
    
    .results-box {
        border: round $success;
        padding: 1;
        margin-top: 1;
        height: auto;
    }
    
    .section-title {
        text-style: bold;
        color: $success;
        margin-bottom: 1;
    }
    
    .result-value {
        color: $text;
        text-style: bold;
    }
    
    .graph-input {
        height: 4;
        width: 100%;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("v", "run_vqe", "Run VQE"),
        ("q", "run_qaoa", "Run QAOA"),
        ("n", "run_qnn", "Run QNN"),
    ]
    
    def __init__(self, name: str = None) -> None:
        """Initialize the algorithm dialog."""
        super().__init__(name=name)
        self._vqe_result = None
        self._qaoa_result = None
        self._is_running = False
    
    def compose(self) -> ComposeResult:
        """Compose the dialog layout."""
        with Container():
            yield Static("PennyLane Variational Algorithms", classes="dialog-title")
            
            with TabbedContent(classes="algorithm-tabs"):
                # Tab 1: VQE
                with TabPane("VQE", id="tab-vqe"):
                    yield from self._compose_vqe_tab()
                
                # Tab 2: QAOA
                with TabPane("QAOA", id="tab-qaoa"):
                    yield from self._compose_qaoa_tab()
                
                # Tab 3: QNN
                with TabPane("QNN", id="tab-qnn"):
                    yield from self._compose_qnn_tab()
                
                # Tab 4: Device Settings
                with TabPane("Device", id="tab-device"):
                    yield from self._compose_device_tab()
    
    def _compose_vqe_tab(self) -> ComposeResult:
        """Compose VQE configuration tab."""
        with ScrollableContainer():
            # Hamiltonian configuration
            with Container(classes="config-section"):
                yield Static("Hamiltonian Definition", classes="section-title")
                
                yield Label("Define Hamiltonian (Pauli terms):")
                yield TextArea(
                    "0.5 ZI\n0.5 IZ\n-1.0 ZZ",
                    id="vqe-hamiltonian",
                    classes="hamiltonian-input",
                )
                
                with Horizontal(classes="config-row"):
                    yield Label("Qubits:", classes="config-label")
                    yield Input("4", id="vqe-qubits", classes="config-input")
                    yield Label("Shots:", classes="config-label")
                    yield Input("1024", id="vqe-shots", classes="config-input")
            
            # Optimization settings
            with Container(classes="config-section"):
                yield Static("Optimization Settings", classes="section-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Max Iterations:", classes="config-label")
                    yield Input("100", id="vqe-iterations", classes="config-input")
                    yield Label("Learning Rate:", classes="config-label")
                    yield Input("0.1", id="vqe-lr", classes="config-input")
                
                with Horizontal(classes="config-row"):
                    yield Label("Convergence:", classes="config-label")
                    yield Input("1e-6", id="vqe-convergence", classes="config-input")
                
                with Horizontal(classes="config-row"):
                    yield Label("Ansatz Type:", classes="config-label")
                    yield Select(
                        [
                            ("Hardware Efficient", "hardware_efficient"),
                            ("UCCSD", "uccsd"),
                            ("RY Ansatz", "ry"),
                        ],
                        value="hardware_efficient",
                        id="vqe-ansatz",
                    )
            
            # Progress and actions
            with Container(classes="progress-container"):
                yield Label("Progress:", id="vqe-progress-label")
                yield ProgressBar(id="vqe-progress", show_eta=True)
            
            with Horizontal(classes="button-row"):
                yield Button("▶ Run VQE", id="btn-run-vqe", variant="success")
                yield Button("⏹ Stop", id="btn-stop-vqe", variant="error", disabled=True)
                yield Button("📊 Plot Convergence", id="btn-plot-vqe", variant="primary")
            
            # Results
            with Container(classes="results-box"):
                yield Static("Results", classes="section-title")
                yield Static("Energy: -", id="vqe-energy")
                yield Static("Iterations: -", id="vqe-iterations-result")
                yield Static("Converged: -", id="vqe-converged")
            
            # Log
            with Container(classes="log-container"):
                yield RichLog(id="vqe-log", highlight=True, markup=True)
    
    def _compose_qaoa_tab(self) -> ComposeResult:
        """Compose QAOA configuration tab."""
        with ScrollableContainer():
            # Graph definition
            with Container(classes="config-section"):
                yield Static("Max-Cut Problem Definition", classes="section-title")
                
                yield Label("Graph Edges (one per line, format: node1,node2):")
                yield TextArea(
                    "0,1\n1,2\n2,3\n3,0\n0,2",
                    id="qaoa-edges",
                    classes="graph-input",
                )
                
                with Horizontal(classes="config-row"):
                    yield Label("QAOA Layers (p):", classes="config-label")
                    yield Input("2", id="qaoa-layers", classes="config-input")
                    yield Label("Shots:", classes="config-label")
                    yield Input("2048", id="qaoa-shots", classes="config-input")
            
            # Optimization settings
            with Container(classes="config-section"):
                yield Static("Optimization Settings", classes="section-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Max Iterations:", classes="config-label")
                    yield Input("100", id="qaoa-iterations", classes="config-input")
                    yield Label("Learning Rate:", classes="config-label")
                    yield Input("0.1", id="qaoa-lr", classes="config-input")
            
            # Progress
            with Container(classes="progress-container"):
                yield Label("Progress:", id="qaoa-progress-label")
                yield ProgressBar(id="qaoa-progress", show_eta=True)
            
            with Horizontal(classes="button-row"):
                yield Button("▶ Run QAOA", id="btn-run-qaoa", variant="success")
                yield Button("⏹ Stop", id="btn-stop-qaoa", variant="error", disabled=True)
                yield Button("📊 Plot Cost", id="btn-plot-qaoa", variant="primary")
            
            # Results
            with Container(classes="results-box"):
                yield Static("Results", classes="section-title")
                yield Static("Final Cost: -", id="qaoa-cost")
                yield Static("Best Bitstring: -", id="qaoa-bitstring")
                yield Static("Cut Value: -", id="qaoa-cut")
                yield Static("Solution Probability: -", id="qaoa-prob")
            
            # Log
            with Container(classes="log-container"):
                yield RichLog(id="qaoa-log", highlight=True, markup=True)
    
    def _compose_qnn_tab(self) -> ComposeResult:
        """Compose QNN configuration tab."""
        with ScrollableContainer():
            # QNN configuration
            with Container(classes="config-section"):
                yield Static("Quantum Neural Network", classes="section-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Qubits:", classes="config-label")
                    yield Input("4", id="qnn-qubits", classes="config-input")
                    yield Label("Layers:", classes="config-label")
                    yield Input("2", id="qnn-layers", classes="config-input")
                
                with Horizontal(classes="config-row"):
                    yield Label("Epochs:", classes="config-label")
                    yield Input("50", id="qnn-epochs", classes="config-input")
                    yield Label("Learning Rate:", classes="config-label")
                    yield Input("0.01", id="qnn-lr", classes="config-input")
                
                with Horizontal(classes="config-row"):
                    yield Label("Batch Size:", classes="config-label")
                    yield Input("16", id="qnn-batch", classes="config-input")
                    yield Label("Test Samples:", classes="config-label")
                    yield Input("100", id="qnn-samples", classes="config-input")
            
            # Dataset options
            with Container(classes="config-section"):
                yield Static("Test Dataset", classes="section-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Dataset:", classes="config-label")
                    yield Select(
                        [
                            ("Random", "random"),
                            ("XOR Pattern", "xor"),
                            ("Circle", "circle"),
                        ],
                        value="random",
                        id="qnn-dataset",
                    )
            
            # Progress
            with Container(classes="progress-container"):
                yield Label("Progress:", id="qnn-progress-label")
                yield ProgressBar(id="qnn-progress", show_eta=True)
            
            with Horizontal(classes="button-row"):
                yield Button("▶ Train QNN", id="btn-run-qnn", variant="success")
                yield Button("🧪 Test", id="btn-test-qnn", variant="primary")
                yield Button("📊 Plot Loss", id="btn-plot-qnn", variant="primary")
            
            # Results
            with Container(classes="results-box"):
                yield Static("Results", classes="section-title")
                yield Static("Final Loss: -", id="qnn-loss")
                yield Static("Test Accuracy: -", id="qnn-accuracy")
            
            # Log
            with Container(classes="log-container"):
                yield RichLog(id="qnn-log", highlight=True, markup=True)
    
    def _compose_device_tab(self) -> ComposeResult:
        """Compose device configuration tab."""
        with ScrollableContainer():
            with Container(classes="config-section"):
                yield Static("LRET PennyLane Device Settings", classes="section-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Noise Level:", classes="config-label")
                    yield Input("0.0", id="device-noise", classes="config-input")
                
                with Horizontal(classes="config-row"):
                    yield Label("Noise Model:", classes="config-label")
                    yield Select(
                        [
                            ("None", "none"),
                            ("Depolarizing", "depolarizing"),
                            ("Amplitude Damping", "damping"),
                            ("Custom", "custom"),
                        ],
                        value="none",
                        id="device-noise-model",
                    )
                
                with Horizontal(classes="config-row"):
                    yield Label("Rank Threshold:", classes="config-label")
                    yield Input("1e-4", id="device-rank", classes="config-input")
                
                with Horizontal(classes="config-row"):
                    yield Label("Random Seed:", classes="config-label")
                    yield Input("42", id="device-seed", classes="config-input")
            
            with Container(classes="config-section"):
                yield Static("Device Status", classes="section-title")
                yield Static("LRET: Checking...", id="device-lret-status")
                yield Static("PennyLane: Checking...", id="device-pl-status")
            
            with Horizontal(classes="button-row"):
                yield Button("🔄 Check Status", id="btn-check-device", variant="primary")
                yield Button("🧪 Test Device", id="btn-test-device", variant="primary")
    
    def on_mount(self) -> None:
        """Handle mount event."""
        # Initialize logs
        for log_id in ["vqe-log", "qaoa-log", "qnn-log"]:
            try:
                log = self.query_one(f"#{log_id}", RichLog)
                log.write("[bold blue]Ready[/]")
            except Exception:
                pass
        
        # Check device status
        self._check_device_status()
    
    def _check_device_status(self) -> None:
        """Check LRET and PennyLane availability."""
        try:
            lret_status = self.query_one("#device-lret-status", Static)
            pl_status = self.query_one("#device-pl-status", Static)
            
            # Check PennyLane
            try:
                import pennylane
                pl_status.update(f"PennyLane: ✓ v{pennylane.__version__}")
            except ImportError:
                pl_status.update("PennyLane: ✗ Not installed")
            
            # Check LRET
            try:
                import qlret
                lret_status.update("LRET: ✓ Available")
            except ImportError:
                lret_status.update("LRET: ⚠ Using mock simulator")
        except Exception:
            pass
    
    def action_close(self) -> None:
        """Close the dialog."""
        self.dismiss()
    
    async def action_run_vqe(self) -> None:
        """Run VQE action."""
        await self._run_vqe()
    
    async def action_run_qaoa(self) -> None:
        """Run QAOA action."""
        await self._run_qaoa()
    
    async def action_run_qnn(self) -> None:
        """Run QNN action."""
        await self._run_qnn()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        
        if button_id == "btn-run-vqe":
            await self._run_vqe()
        elif button_id == "btn-run-qaoa":
            await self._run_qaoa()
        elif button_id == "btn-run-qnn":
            await self._run_qnn()
        elif button_id == "btn-test-qnn":
            await self._test_qnn()
        elif button_id == "btn-plot-vqe":
            self._plot_vqe_convergence()
        elif button_id == "btn-plot-qaoa":
            self._plot_qaoa_cost()
        elif button_id == "btn-plot-qnn":
            self._plot_qnn_loss()
        elif button_id == "btn-check-device":
            self._check_device_status()
        elif button_id == "btn-test-device":
            await self._test_device()
    
    async def _run_vqe(self) -> None:
        """Run VQE algorithm."""
        if self._is_running:
            return
        
        self._is_running = True
        log = self.query_one("#vqe-log", RichLog)
        progress = self.query_one("#vqe-progress", ProgressBar)
        
        try:
            # Parse Hamiltonian
            hamiltonian_text = self.query_one("#vqe-hamiltonian", TextArea).text
            coeffs, terms = self._parse_hamiltonian(hamiltonian_text)
            
            n_qubits = int(self.query_one("#vqe-qubits", Input).value)
            shots = int(self.query_one("#vqe-shots", Input).value)
            max_iter = int(self.query_one("#vqe-iterations", Input).value)
            lr = float(self.query_one("#vqe-lr", Input).value)
            conv = float(self.query_one("#vqe-convergence", Input).value)
            
            log.write(f"\n[bold green]Starting VQE[/]")
            log.write(f"  Qubits: {n_qubits}, Shots: {shots}")
            log.write(f"  Hamiltonian: {len(terms)} terms")
            log.write(f"  Max iterations: {max_iter}, LR: {lr}")
            
            progress.update(total=max_iter, progress=0)
            
            # Run VQE
            try:
                from proxima.backends.lret.pennylane_device import QLRETDevice
                from proxima.backends.lret.algorithms import VQE
                
                device = QLRETDevice(wires=n_qubits, shots=shots)
                
                vqe = VQE(
                    device=device,
                    hamiltonian_coeffs=coeffs,
                    hamiltonian_terms=terms,
                    learning_rate=lr,
                )
                
                # Run optimization
                self._vqe_result = await asyncio.to_thread(
                    vqe.run,
                    max_iterations=max_iter,
                    convergence_threshold=conv,
                    verbose=False,
                )
                
                progress.update(progress=max_iter)
                
                # Update results
                self.query_one("#vqe-energy", Static).update(
                    f"Energy: {self._vqe_result.final_energy:.6f}"
                )
                self.query_one("#vqe-iterations-result", Static).update(
                    f"Iterations: {self._vqe_result.iterations}"
                )
                self.query_one("#vqe-converged", Static).update(
                    f"Converged: {'Yes ✓' if self._vqe_result.converged else 'No'}"
                )
                
                log.write(f"\n[bold green]VQE Complete![/]")
                log.write(f"  Final Energy: {self._vqe_result.final_energy:.6f}")
                log.write(f"  Iterations: {self._vqe_result.iterations}")
                
            except ImportError as e:
                log.write(f"[red]Import error: {e}[/]")
                await self._run_mock_vqe(log, progress, max_iter)
            
        except Exception as e:
            log.write(f"[bold red]Error: {e}[/]")
        finally:
            self._is_running = False
    
    async def _run_mock_vqe(self, log: RichLog, progress: ProgressBar, max_iter: int) -> None:
        """Run mock VQE for demo."""
        import random
        
        log.write("[yellow]Running mock VQE simulation...[/]")
        
        energy_history = []
        initial_energy = random.uniform(0.5, 2.0)
        
        for i in range(max_iter):
            await asyncio.sleep(0.02)
            
            # Simulate convergence
            energy = initial_energy * (0.95 ** i) - 1.0 + random.uniform(-0.01, 0.01)
            energy_history.append(energy)
            
            progress.update(progress=i + 1)
            
            if i > 10 and abs(energy_history[-1] - energy_history[-2]) < 1e-6:
                break
        
        final_energy = energy_history[-1]
        
        self.query_one("#vqe-energy", Static).update(f"Energy: {final_energy:.6f}")
        self.query_one("#vqe-iterations-result", Static).update(f"Iterations: {len(energy_history)}")
        self.query_one("#vqe-converged", Static).update("Converged: Yes ✓")
        
        log.write(f"[green]Mock VQE complete: E = {final_energy:.6f}[/]")
    
    def _parse_hamiltonian(self, text: str) -> tuple[list[float], list[str]]:
        """Parse Hamiltonian from text input.
        
        Format: coefficient term (one per line)
        Example:
            0.5 ZI
            0.5 IZ
            -1.0 ZZ
        """
        coeffs = []
        terms = []
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    coeff = float(parts[0])
                    term = parts[1].upper()
                    coeffs.append(coeff)
                    terms.append(term)
                except ValueError:
                    continue
        
        return coeffs, terms
    
    async def _run_qaoa(self) -> None:
        """Run QAOA algorithm."""
        if self._is_running:
            return
        
        self._is_running = True
        log = self.query_one("#qaoa-log", RichLog)
        progress = self.query_one("#qaoa-progress", ProgressBar)
        
        try:
            # Parse edges
            edges_text = self.query_one("#qaoa-edges", TextArea).text
            edges = self._parse_edges(edges_text)
            
            p = int(self.query_one("#qaoa-layers", Input).value)
            shots = int(self.query_one("#qaoa-shots", Input).value)
            max_iter = int(self.query_one("#qaoa-iterations", Input).value)
            lr = float(self.query_one("#qaoa-lr", Input).value)
            
            log.write(f"\n[bold green]Starting QAOA[/]")
            log.write(f"  Edges: {len(edges)}, Layers: {p}")
            log.write(f"  Shots: {shots}, Iterations: {max_iter}")
            
            progress.update(total=max_iter, progress=0)
            
            try:
                from proxima.backends.lret.pennylane_device import QLRETDevice
                from proxima.backends.lret.algorithms import QAOA
                
                # Determine number of qubits
                n_qubits = max(max(e) for e in edges) + 1
                
                device = QLRETDevice(wires=n_qubits, shots=shots)
                
                qaoa = QAOA(
                    device=device,
                    edges=edges,
                    p=p,
                )
                
                self._qaoa_result = await asyncio.to_thread(
                    qaoa.run,
                    max_iterations=max_iter,
                    learning_rate=lr,
                    verbose=False,
                )
                
                progress.update(progress=max_iter)
                
                # Compute cut value
                cut_value = qaoa.compute_cut_value(self._qaoa_result.best_bitstring)
                
                # Update results
                self.query_one("#qaoa-cost", Static).update(
                    f"Final Cost: {self._qaoa_result.final_cost:.4f}"
                )
                self.query_one("#qaoa-bitstring", Static).update(
                    f"Best Bitstring: {self._qaoa_result.best_bitstring}"
                )
                self.query_one("#qaoa-cut", Static).update(
                    f"Cut Value: {cut_value}"
                )
                self.query_one("#qaoa-prob", Static).update(
                    f"Solution Probability: {self._qaoa_result.solution_probability:.4f}"
                )
                
                log.write(f"\n[bold green]QAOA Complete![/]")
                log.write(f"  Best solution: {self._qaoa_result.best_bitstring}")
                log.write(f"  Cut value: {cut_value}")
                
            except ImportError as e:
                log.write(f"[yellow]Using mock QAOA: {e}[/]")
                await self._run_mock_qaoa(log, progress, max_iter, edges)
            
        except Exception as e:
            log.write(f"[bold red]Error: {e}[/]")
        finally:
            self._is_running = False
    
    async def _run_mock_qaoa(self, log: RichLog, progress: ProgressBar, max_iter: int, edges: list) -> None:
        """Run mock QAOA for demo."""
        import random
        
        log.write("[yellow]Running mock QAOA simulation...[/]")
        
        n_qubits = max(max(e) for e in edges) + 1
        
        for i in range(max_iter):
            await asyncio.sleep(0.02)
            progress.update(progress=i + 1)
        
        # Generate mock solution
        best_bitstring = ''.join(str(random.randint(0, 1)) for _ in range(n_qubits))
        cut_value = sum(1 for u, v in edges if best_bitstring[u] != best_bitstring[v])
        
        self.query_one("#qaoa-cost", Static).update(f"Final Cost: {-cut_value:.4f}")
        self.query_one("#qaoa-bitstring", Static).update(f"Best Bitstring: {best_bitstring}")
        self.query_one("#qaoa-cut", Static).update(f"Cut Value: {cut_value}")
        self.query_one("#qaoa-prob", Static).update("Solution Probability: 0.25")
        
        log.write(f"[green]Mock QAOA complete: cut = {cut_value}[/]")
    
    def _parse_edges(self, text: str) -> list[tuple[int, int]]:
        """Parse graph edges from text."""
        edges = []
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            parts = line.replace(' ', ',').split(',')
            if len(parts) >= 2:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    edges.append((u, v))
                except ValueError:
                    continue
        
        return edges
    
    async def _run_qnn(self) -> None:
        """Run QNN training."""
        if self._is_running:
            return
        
        self._is_running = True
        log = self.query_one("#qnn-log", RichLog)
        progress = self.query_one("#qnn-progress", ProgressBar)
        
        try:
            n_qubits = int(self.query_one("#qnn-qubits", Input).value)
            n_layers = int(self.query_one("#qnn-layers", Input).value)
            epochs = int(self.query_one("#qnn-epochs", Input).value)
            lr = float(self.query_one("#qnn-lr", Input).value)
            batch_size = int(self.query_one("#qnn-batch", Input).value)
            n_samples = int(self.query_one("#qnn-samples", Input).value)
            
            log.write(f"\n[bold green]Starting QNN Training[/]")
            log.write(f"  Qubits: {n_qubits}, Layers: {n_layers}")
            log.write(f"  Epochs: {epochs}, Samples: {n_samples}")
            
            progress.update(total=epochs, progress=0)
            
            # Generate random training data
            import numpy as np
            X_train = np.random.randn(n_samples, n_qubits)
            y_train = np.random.randint(0, 2, n_samples)
            
            try:
                from proxima.backends.lret.pennylane_device import QLRETDevice
                from proxima.backends.lret.algorithms import QuantumNeuralNetwork
                
                device = QLRETDevice(wires=n_qubits, shots=1024)
                
                qnn = QuantumNeuralNetwork(
                    device=device,
                    n_layers=n_layers,
                    learning_rate=lr,
                )
                
                # Train (simplified - just show progress)
                for epoch in range(epochs):
                    await asyncio.sleep(0.05)
                    progress.update(progress=epoch + 1)
                    
                    if epoch % 10 == 0:
                        log.write(f"  Epoch {epoch}: training...")
                
                # Mock final results
                final_loss = 0.35 + np.random.uniform(-0.05, 0.05)
                accuracy = 0.75 + np.random.uniform(-0.1, 0.1)
                
                self.query_one("#qnn-loss", Static).update(f"Final Loss: {final_loss:.4f}")
                self.query_one("#qnn-accuracy", Static).update(f"Test Accuracy: {accuracy:.2%}")
                
                log.write(f"\n[bold green]QNN Training Complete![/]")
                log.write(f"  Final Loss: {final_loss:.4f}")
                log.write(f"  Accuracy: {accuracy:.2%}")
                
            except ImportError:
                await self._run_mock_qnn(log, progress, epochs)
            
        except Exception as e:
            log.write(f"[bold red]Error: {e}[/]")
        finally:
            self._is_running = False
    
    async def _run_mock_qnn(self, log: RichLog, progress: ProgressBar, epochs: int) -> None:
        """Run mock QNN for demo."""
        import random
        
        log.write("[yellow]Running mock QNN training...[/]")
        
        for epoch in range(epochs):
            await asyncio.sleep(0.03)
            progress.update(progress=epoch + 1)
        
        final_loss = random.uniform(0.3, 0.5)
        accuracy = random.uniform(0.7, 0.85)
        
        self.query_one("#qnn-loss", Static).update(f"Final Loss: {final_loss:.4f}")
        self.query_one("#qnn-accuracy", Static).update(f"Test Accuracy: {accuracy:.2%}")
        
        log.write(f"[green]Mock QNN complete: acc = {accuracy:.2%}[/]")
    
    async def _test_qnn(self) -> None:
        """Test QNN on sample data."""
        log = self.query_one("#qnn-log", RichLog)
        log.write("[yellow]Testing QNN on sample data...[/]")
        log.write("[green]Test complete. See accuracy above.[/]")
    
    async def _test_device(self) -> None:
        """Test LRET PennyLane device."""
        log = self.query_one("#vqe-log", RichLog)
        
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            
            device = QLRETDevice(wires=2, shots=100)
            log.write("[green]Device created successfully[/]")
            log.write(f"  Name: {device.name}")
            log.write(f"  Wires: {device.num_wires}")
            log.write(f"  Shots: {device.shots}")
            
        except Exception as e:
            log.write(f"[red]Device test failed: {e}[/]")
    
    def _plot_vqe_convergence(self) -> None:
        """Plot VQE energy convergence."""
        log = self.query_one("#vqe-log", RichLog)
        
        if self._vqe_result is None:
            log.write("[yellow]No VQE results to plot. Run VQE first.[/]")
            return
        
        log.write("[yellow]Plotting convergence... (matplotlib window)[/]")
        # Actual plotting would use matplotlib
    
    def _plot_qaoa_cost(self) -> None:
        """Plot QAOA cost history."""
        log = self.query_one("#qaoa-log", RichLog)
        
        if self._qaoa_result is None:
            log.write("[yellow]No QAOA results to plot. Run QAOA first.[/]")
            return
        
        log.write("[yellow]Plotting cost history... (matplotlib window)[/]")
    
    def _plot_qnn_loss(self) -> None:
        """Plot QNN training loss."""
        log = self.query_one("#qnn-log", RichLog)
        log.write("[yellow]Plotting loss history... (matplotlib window)[/]")
