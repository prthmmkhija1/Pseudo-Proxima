"""LRET Phase 7 Unified Configuration Dialog.

This dialog provides a TUI interface for configuring and testing
the LRET Phase 7 unified multi-framework adapter.

Features:
- Enable/disable frameworks (Cirq, PennyLane, Qiskit)
- Configure framework preference order
- Gate fusion settings (mode: row, column, hybrid)
- GPU acceleration settings
- Test execution across multiple frameworks
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import asyncio

from textual.app import ComposeResult
from textual.containers import (
    Container,
    Horizontal,
    Vertical,
    ScrollableContainer,
)
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    OptionList,
    ProgressBar,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    DataTable,
)
from textual.screen import ModalScreen
from textual.message import Message
from textual.reactive import reactive

try:
    from proxima.backends.lret.phase7_unified import (
        LRETPhase7UnifiedAdapter,
        Phase7Config,
        Framework,
        FusionMode,
    )
    PHASE7_AVAILABLE = True
except ImportError:
    PHASE7_AVAILABLE = False


DEFAULT_CSS = """
Phase7Dialog {
    align: center middle;
}

Phase7Dialog > Container {
    width: 90;
    height: 40;
    border: thick $primary;
    background: $surface;
    padding: 1 2;
}

Phase7Dialog .dialog-title {
    text-align: center;
    text-style: bold;
    color: $text;
    padding-bottom: 1;
}

Phase7Dialog .section-title {
    text-style: bold;
    color: $primary;
    padding: 1 0;
}

Phase7Dialog .framework-list {
    height: 8;
    border: round $primary;
    padding: 1;
}

Phase7Dialog .config-row {
    height: 3;
    padding: 0 1;
}

Phase7Dialog .status-panel {
    height: 6;
    border: round $secondary;
    padding: 1;
}

Phase7Dialog .results-panel {
    height: auto;
    max-height: 12;
    border: round $accent;
    padding: 1;
    overflow-y: auto;
}

Phase7Dialog .button-row {
    height: 3;
    align: center middle;
    padding-top: 1;
}

Phase7Dialog Button {
    margin: 0 1;
}

Phase7Dialog .success {
    color: $success;
}

Phase7Dialog .error {
    color: $error;
}

Phase7Dialog .warning {
    color: $warning;
}
"""


@dataclass
class FrameworkStatus:
    """Status information for a framework."""
    name: str
    enabled: bool
    available: bool
    version: str
    preference_rank: int


class Phase7Dialog(ModalScreen[Dict[str, Any]]):
    """Modal dialog for LRET Phase 7 unified configuration.
    
    Provides tabs for:
    - Frameworks: Enable/disable and set preference
    - Optimization: Gate fusion and optimization level
    - GPU: GPU acceleration settings
    - Test: Run test executions
    """
    
    CSS = DEFAULT_CSS
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("f5", "test_all", "Test All"),
    ]
    
    # Reactive state
    config: reactive[Phase7Config] = reactive(Phase7Config, init=False)
    is_testing: reactive[bool] = reactive(False)
    test_progress: reactive[float] = reactive(0.0)
    
    def __init__(
        self,
        initial_config: Optional[Phase7Config] = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize dialog.
        
        Args:
            initial_config: Initial configuration to use
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.config = initial_config or Phase7Config()
        self._adapter: Optional[LRETPhase7UnifiedAdapter] = None
        self._test_results: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose dialog layout."""
        with Container():
            yield Label("⚡ LRET Phase 7 Unified Configuration", classes="dialog-title")
            
            with TabbedContent():
                # Frameworks tab
                with TabPane("Frameworks", id="tab-frameworks"):
                    yield from self._compose_frameworks_tab()
                
                # Optimization tab
                with TabPane("Optimization", id="tab-optimization"):
                    yield from self._compose_optimization_tab()
                
                # GPU tab
                with TabPane("GPU", id="tab-gpu"):
                    yield from self._compose_gpu_tab()
                
                # Test tab
                with TabPane("Test", id="tab-test"):
                    yield from self._compose_test_tab()
            
            # Button row
            with Horizontal(classes="button-row"):
                yield Button("Apply", variant="primary", id="btn-apply")
                yield Button("Save & Close", variant="success", id="btn-save")
                yield Button("Cancel", variant="error", id="btn-cancel")
    
    def _compose_frameworks_tab(self) -> ComposeResult:
        """Compose frameworks configuration tab."""
        yield Label("Enabled Frameworks", classes="section-title")
        
        with Vertical(classes="framework-list"):
            yield Checkbox(
                "Cirq",
                value='cirq' in self.config.enabled_frameworks,
                id="chk-cirq"
            )
            yield Checkbox(
                "PennyLane",
                value='pennylane' in self.config.enabled_frameworks,
                id="chk-pennylane"
            )
            yield Checkbox(
                "Qiskit",
                value='qiskit' in self.config.enabled_frameworks,
                id="chk-qiskit"
            )
        
        yield Label("Backend Preference Order", classes="section-title")
        yield Static(
            "Drag to reorder (first = highest priority)",
            classes="help-text"
        )
        
        with Vertical(classes="framework-list", id="preference-list"):
            for i, fw in enumerate(self.config.backend_preference):
                yield Horizontal(
                    Label(f"{i+1}. {fw.capitalize()}", id=f"pref-{fw}"),
                    Button("▲", id=f"up-{fw}", classes="small"),
                    Button("▼", id=f"down-{fw}", classes="small"),
                    classes="config-row"
                )
        
        yield Label("Framework Status", classes="section-title")
        yield Static(id="framework-status", classes="status-panel")
    
    def _compose_optimization_tab(self) -> ComposeResult:
        """Compose optimization settings tab."""
        yield Label("Gate Fusion", classes="section-title")
        
        with Horizontal(classes="config-row"):
            yield Label("Enable Gate Fusion:")
            yield Switch(value=self.config.gate_fusion, id="switch-fusion")
        
        yield Label("Fusion Mode", classes="section-title")
        
        with RadioSet(id="radio-fusion-mode"):
            yield RadioButton(
                "Row (time-slice fusion)",
                value=self.config.fusion_mode == 'row',
                id="radio-row"
            )
            yield RadioButton(
                "Column (single-qubit fusion)",
                value=self.config.fusion_mode == 'column',
                id="radio-column"
            )
            yield RadioButton(
                "Hybrid (both strategies)",
                value=self.config.fusion_mode == 'hybrid',
                id="radio-hybrid"
            )
        
        yield Label("Optimization Level", classes="section-title")
        
        with Horizontal(classes="config-row"):
            yield Label("Level (0-2):")
            yield Select(
                [
                    ("0 - None", 0),
                    ("1 - Basic", 1),
                    ("2 - Full", 2),
                ],
                value=self.config.optimization_level,
                id="select-opt-level"
            )
        
        yield Label("Optimization Benefits", classes="section-title")
        yield Static(
            """• Row fusion: Combines adjacent gates in same time slice
• Column fusion: Merges sequential single-qubit gates
• Hybrid: Applies both for maximum optimization

Higher optimization levels may increase compilation time
but improve execution performance.""",
            classes="status-panel"
        )
    
    def _compose_gpu_tab(self) -> ComposeResult:
        """Compose GPU settings tab."""
        yield Label("GPU Acceleration", classes="section-title")
        
        with Horizontal(classes="config-row"):
            yield Label("Enable GPU:")
            yield Switch(value=self.config.gpu_enabled, id="switch-gpu")
        
        with Horizontal(classes="config-row"):
            yield Label("GPU Device ID:")
            yield Input(
                str(self.config.gpu_device_id),
                type="integer",
                id="input-gpu-id"
            )
        
        yield Label("cuQuantum Integration", classes="section-title")
        
        with Horizontal(classes="config-row"):
            yield Label("Enable cuQuantum:")
            yield Switch(value=self.config.cuquantum_enabled, id="switch-cuquantum")
        
        yield Label("GPU Status", classes="section-title")
        yield Static(id="gpu-status", classes="status-panel")
        
        yield Button(
            "Check GPU Availability",
            id="btn-check-gpu",
            variant="primary"
        )
    
    def _compose_test_tab(self) -> ComposeResult:
        """Compose test execution tab."""
        yield Label("Test Configuration", classes="section-title")
        
        with Horizontal(classes="config-row"):
            yield Label("Test Qubits:")
            yield Input("4", type="integer", id="input-test-qubits")
        
        with Horizontal(classes="config-row"):
            yield Label("Test Shots:")
            yield Input("1024", type="integer", id="input-test-shots")
        
        with Horizontal(classes="config-row"):
            yield Label("Test Framework:")
            yield Select(
                [
                    ("Auto", "auto"),
                    ("Cirq", "cirq"),
                    ("PennyLane", "pennylane"),
                    ("Qiskit", "qiskit"),
                ],
                value="auto",
                id="select-test-framework"
            )
        
        yield Label("Test Progress", classes="section-title")
        yield ProgressBar(total=100, id="test-progress")
        yield Static("Ready to test", id="test-status")
        
        with Horizontal(classes="button-row"):
            yield Button("Run Test", variant="primary", id="btn-run-test")
            yield Button("Test All Frameworks", variant="success", id="btn-test-all")
        
        yield Label("Test Results", classes="section-title")
        yield DataTable(id="test-results")
    
    async def on_mount(self) -> None:
        """Handle mount event."""
        # Initialize adapter
        if PHASE7_AVAILABLE:
            self._adapter = LRETPhase7UnifiedAdapter(self.config)
            self._adapter.connect()
            await self._update_framework_status()
            await self._update_gpu_status()
        
        # Set up test results table
        table = self.query_one("#test-results", DataTable)
        table.add_columns("Framework", "Status", "Time (ms)", "Gates", "Shots")
    
    async def _update_framework_status(self) -> None:
        """Update framework status display."""
        if not self._adapter:
            return
        
        status_widget = self.query_one("#framework-status", Static)
        
        status_lines = []
        for name in ['cirq', 'pennylane', 'qiskit']:
            enabled = name in self.config.enabled_frameworks
            available = name in self._adapter.available_frameworks
            
            if available:
                icon = "✅"
                state = "Available"
            elif enabled:
                icon = "⚠️"
                state = "Enabled (not installed)"
            else:
                icon = "❌"
                state = "Disabled"
            
            status_lines.append(f"{icon} {name.capitalize()}: {state}")
        
        status_widget.update("\n".join(status_lines))
    
    async def _update_gpu_status(self) -> None:
        """Update GPU status display."""
        status_widget = self.query_one("#gpu-status", Static)
        
        # Check GPU availability
        gpu_available = False
        gpu_info = "No GPU detected"
        
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info = result.stdout.strip()
                gpu_available = True
        except Exception:
            pass
        
        # Check cuQuantum
        cuquantum_available = False
        try:
            import cuquantum
            cuquantum_available = True
        except ImportError:
            pass
        
        status_lines = [
            f"GPU: {'✅ ' + gpu_info if gpu_available else '❌ Not available'}",
            f"cuQuantum: {'✅ Installed' if cuquantum_available else '❌ Not installed'}",
        ]
        
        if gpu_available:
            status_lines.append(f"Selected Device: {self.config.gpu_device_id}")
        
        status_widget.update("\n".join(status_lines))
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        
        if button_id == "btn-cancel":
            self.dismiss(None)
        
        elif button_id == "btn-apply":
            self._apply_config()
            self.notify("Configuration applied")
        
        elif button_id == "btn-save":
            self._apply_config()
            self.dismiss(self._get_config_dict())
        
        elif button_id == "btn-check-gpu":
            await self._update_gpu_status()
        
        elif button_id == "btn-run-test":
            await self._run_single_test()
        
        elif button_id == "btn-test-all":
            await self._run_all_tests()
        
        elif button_id and button_id.startswith("up-"):
            framework = button_id[3:]
            self._move_preference(framework, -1)
        
        elif button_id and button_id.startswith("down-"):
            framework = button_id[5:]
            self._move_preference(framework, 1)
    
    def _apply_config(self) -> None:
        """Apply current UI state to config."""
        # Get enabled frameworks
        enabled = []
        if self.query_one("#chk-cirq", Checkbox).value:
            enabled.append('cirq')
        if self.query_one("#chk-pennylane", Checkbox).value:
            enabled.append('pennylane')
        if self.query_one("#chk-qiskit", Checkbox).value:
            enabled.append('qiskit')
        
        self.config.enabled_frameworks = enabled
        
        # Get fusion settings
        self.config.gate_fusion = self.query_one("#switch-fusion", Switch).value
        
        radio_set = self.query_one("#radio-fusion-mode", RadioSet)
        if radio_set.pressed_button:
            mode_id = radio_set.pressed_button.id
            if mode_id == "radio-row":
                self.config.fusion_mode = 'row'
            elif mode_id == "radio-column":
                self.config.fusion_mode = 'column'
            else:
                self.config.fusion_mode = 'hybrid'
        
        # Get optimization level
        opt_select = self.query_one("#select-opt-level", Select)
        if opt_select.value is not None:
            self.config.optimization_level = int(opt_select.value)
        
        # Get GPU settings
        self.config.gpu_enabled = self.query_one("#switch-gpu", Switch).value
        
        gpu_id_input = self.query_one("#input-gpu-id", Input)
        try:
            self.config.gpu_device_id = int(gpu_id_input.value)
        except ValueError:
            pass
        
        self.config.cuquantum_enabled = self.query_one("#switch-cuquantum", Switch).value
        
        # Reinitialize adapter with new config
        if self._adapter:
            self._adapter.disconnect()
            self._adapter = LRETPhase7UnifiedAdapter(self.config)
            self._adapter.connect()
    
    def _get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return {
            'enabled_frameworks': self.config.enabled_frameworks,
            'backend_preference': self.config.backend_preference,
            'gate_fusion': self.config.gate_fusion,
            'fusion_mode': self.config.fusion_mode,
            'gpu_enabled': self.config.gpu_enabled,
            'gpu_device_id': self.config.gpu_device_id,
            'cuquantum_enabled': self.config.cuquantum_enabled,
            'optimization_level': self.config.optimization_level,
        }
    
    def _move_preference(self, framework: str, direction: int) -> None:
        """Move framework in preference order.
        
        Args:
            framework: Framework name
            direction: -1 for up, 1 for down
        """
        pref = list(self.config.backend_preference)
        
        try:
            idx = pref.index(framework)
            new_idx = idx + direction
            
            if 0 <= new_idx < len(pref):
                pref[idx], pref[new_idx] = pref[new_idx], pref[idx]
                self.config.backend_preference = pref
                self._refresh_preference_list()
        except ValueError:
            pass
    
    def _refresh_preference_list(self) -> None:
        """Refresh preference list display."""
        for i, fw in enumerate(self.config.backend_preference):
            label = self.query_one(f"#pref-{fw}", Label)
            if label:
                label.update(f"{i+1}. {fw.capitalize()}")
    
    async def _run_single_test(self) -> None:
        """Run single framework test."""
        if not self._adapter:
            self.notify("Adapter not initialized", severity="error")
            return
        
        self.is_testing = True
        status = self.query_one("#test-status", Static)
        progress = self.query_one("#test-progress", ProgressBar)
        
        try:
            # Get test parameters
            qubits = int(self.query_one("#input-test-qubits", Input).value)
            shots = int(self.query_one("#input-test-shots", Input).value)
            framework_select = self.query_one("#select-test-framework", Select)
            framework = str(framework_select.value) if framework_select.value else "auto"
            
            status.update(f"Testing with {framework}...")
            progress.update(progress=30)
            
            # Create test circuit
            circuit = self._create_test_circuit(qubits, framework)
            
            progress.update(progress=50)
            
            # Execute
            result = self._adapter.execute(circuit, options={
                'framework': framework,
                'shots': shots,
                'optimize': self.config.gate_fusion,
            })
            
            progress.update(progress=100)
            
            # Add result to table
            table = self.query_one("#test-results", DataTable)
            metrics = self._adapter.metrics
            
            table.add_row(
                framework if framework != 'auto' else result.metadata.get('framework', 'auto'),
                "✅ Success" if result.data else "⚠️ Empty",
                f"{result.execution_time_ms:.2f}",
                str(metrics.gate_count if metrics else 0),
                str(result.shot_count or 0),
            )
            
            status.update("Test completed successfully")
            self.notify("Test completed", severity="information")
            
        except Exception as e:
            status.update(f"Test failed: {e}")
            self.notify(f"Test failed: {e}", severity="error")
        
        finally:
            self.is_testing = False
    
    async def _run_all_tests(self) -> None:
        """Run tests on all enabled frameworks."""
        if not self._adapter:
            self.notify("Adapter not initialized", severity="error")
            return
        
        self.is_testing = True
        status = self.query_one("#test-status", Static)
        progress = self.query_one("#test-progress", ProgressBar)
        table = self.query_one("#test-results", DataTable)
        
        # Clear previous results
        table.clear()
        
        frameworks = self._adapter.available_frameworks
        total = len(frameworks)
        
        try:
            qubits = int(self.query_one("#input-test-qubits", Input).value)
            shots = int(self.query_one("#input-test-shots", Input).value)
            
            for i, fw in enumerate(frameworks):
                status.update(f"Testing {fw} ({i+1}/{total})...")
                progress.update(progress=((i + 1) / total) * 100)
                
                try:
                    circuit = self._create_test_circuit(qubits, fw)
                    
                    result = self._adapter.execute(circuit, options={
                        'framework': fw,
                        'shots': shots,
                        'optimize': self.config.gate_fusion,
                    })
                    
                    metrics = self._adapter.metrics
                    
                    table.add_row(
                        fw,
                        "✅ Success",
                        f"{result.execution_time_ms:.2f}",
                        str(metrics.gate_count if metrics else 0),
                        str(result.shot_count or 0),
                    )
                    
                except Exception as e:
                    table.add_row(fw, f"❌ {str(e)[:20]}", "-", "-", "-")
                
                # Small delay between tests
                await asyncio.sleep(0.1)
            
            status.update(f"All tests completed ({total} frameworks)")
            self.notify(f"Tested {total} frameworks", severity="information")
            
        except Exception as e:
            status.update(f"Tests failed: {e}")
            self.notify(f"Tests failed: {e}", severity="error")
        
        finally:
            self.is_testing = False
    
    def _create_test_circuit(self, qubits: int, framework: str) -> Any:
        """Create test circuit for specified framework.
        
        Args:
            qubits: Number of qubits
            framework: Target framework
            
        Returns:
            Test circuit
        """
        if framework == 'cirq' or framework == 'auto':
            try:
                import cirq
                q = cirq.LineQubit.range(qubits)
                circuit = cirq.Circuit([
                    cirq.H.on_each(*q),
                    *[cirq.CNOT(q[i], q[i+1]) for i in range(qubits - 1)],
                    cirq.measure(*q, key='m')
                ])
                return circuit
            except ImportError:
                pass
        
        if framework == 'qiskit':
            try:
                from qiskit import QuantumCircuit
                circuit = QuantumCircuit(qubits, qubits)
                for i in range(qubits):
                    circuit.h(i)
                for i in range(qubits - 1):
                    circuit.cx(i, i + 1)
                circuit.measure_all()
                return circuit
            except ImportError:
                pass
        
        # Return mock circuit for testing
        return MockCircuit(qubits)
    
    def action_cancel(self) -> None:
        """Cancel dialog."""
        self.dismiss(None)
    
    async def action_test_all(self) -> None:
        """Test all frameworks shortcut."""
        await self._run_all_tests()


class MockCircuit:
    """Mock circuit for testing when frameworks not available."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.gates: List[str] = []
        
        # Create simple circuit
        for i in range(num_qubits):
            self.gates.append(f"H({i})")
        for i in range(num_qubits - 1):
            self.gates.append(f"CNOT({i},{i+1})")
    
    def all_qubits(self):
        """Return all qubits (Cirq-like interface)."""
        return list(range(self.num_qubits))
    
    def all_operations(self):
        """Return all operations (Cirq-like interface)."""
        return self.gates
    
    def __len__(self):
        return len(self.gates)


# Convenience function for launching dialog
async def show_phase7_dialog(
    app: Any,
    initial_config: Optional[Phase7Config] = None
) -> Optional[Dict[str, Any]]:
    """Show Phase 7 configuration dialog.
    
    Args:
        app: Parent Textual app
        initial_config: Initial configuration
        
    Returns:
        Configuration dictionary or None if cancelled
    """
    dialog = Phase7Dialog(initial_config=initial_config)
    return await app.push_screen(dialog)
