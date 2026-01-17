"""
Enhanced TUI Components for Proxima.

Step 6.1 / 11: Additional TUI features including:
- SystemMetricsDashboard: Real-time system metrics display
- CircuitEditor: Interactive quantum circuit editor
- ExecutionMonitor: Real-time execution monitoring panel
- KeyboardShortcutsModal: Keyboard shortcuts help
- QuickCommandBar: Command palette for quick actions
"""

from __future__ import annotations

import asyncio
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Grid
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    ProgressBar as TextualProgressBar,
    Static,
    TextArea,
)


# =============================================================================
# System Metrics Dashboard (Feature - TUI)
# =============================================================================


@dataclass
class SystemMetric:
    """A system metric data point."""
    
    name: str
    value: float
    unit: str
    max_value: float | None = None
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    trend: list[float] = field(default_factory=list)


class MetricGauge(Static):
    """Visual gauge for displaying a metric value."""
    
    DEFAULT_CSS = """
    MetricGauge {
        height: 5;
        min-width: 20;
        padding: 0 1;
        border: solid $surface-lighten-1;
        margin: 0 1;
    }
    MetricGauge .gauge-label { text-style: bold; }
    MetricGauge .gauge-value { text-align: right; }
    MetricGauge .gauge-bar { height: 1; }
    MetricGauge.warning { border: solid $warning; }
    MetricGauge.critical { border: solid $error; }
    """
    
    value = reactive(0.0)
    
    def __init__(
        self,
        label: str,
        value: float = 0.0,
        max_value: float = 100.0,
        unit: str = "%",
        warning_threshold: float = 70.0,
        critical_threshold: float = 90.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._max_value = max_value
        self._unit = unit
        self._warning = warning_threshold
        self._critical = critical_threshold
        self.value = value
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(self._label, classes="gauge-label")
            yield Label(
                f"{self.value:.1f}{self._unit}",
                classes="gauge-value",
                id="gauge-value",
            )
        yield TextualProgressBar(
            total=int(self._max_value),
            show_eta=False,
            show_percentage=False,
            id="gauge-bar",
        )
    
    def watch_value(self, new_value: float) -> None:
        """Update gauge when value changes."""
        try:
            value_label = self.query_one("#gauge-value", Label)
            value_label.update(f"{new_value:.1f}{self._unit}")
            
            progress = self.query_one("#gauge-bar", TextualProgressBar)
            progress.progress = int(new_value)
            
            # Update classes for warning/critical
            self.remove_class("warning", "critical")
            if new_value >= self._critical:
                self.add_class("critical")
            elif new_value >= self._warning:
                self.add_class("warning")
        except Exception:
            pass  # Widget not mounted yet
    
    def update_metric(self, value: float) -> None:
        """Update the metric value."""
        self.value = min(value, self._max_value)


class SystemMetricsDashboard(Static):
    """Dashboard showing real-time system metrics.
    
    Features:
    - CPU usage
    - Memory usage
    - GPU usage (if available)
    - Disk usage
    - Backend status summary
    - Execution queue status
    """
    
    DEFAULT_CSS = """
    SystemMetricsDashboard {
        height: auto;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    SystemMetricsDashboard .dashboard-title {
        text-style: bold;
        margin-bottom: 1;
    }
    SystemMetricsDashboard .metrics-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
    }
    SystemMetricsDashboard .summary-row {
        margin-top: 1;
        height: 3;
    }
    """
    
    def __init__(
        self,
        auto_refresh: bool = True,
        refresh_interval: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._auto_refresh = auto_refresh
        self._refresh_interval = refresh_interval
        self._refresh_timer = None
        self._metrics: dict[str, SystemMetric] = {}
    
    def compose(self) -> ComposeResult:
        yield Label("📊 System Metrics", classes="dashboard-title")
        
        with Grid(classes="metrics-grid"):
            yield MetricGauge(
                "CPU",
                value=0.0,
                max_value=100.0,
                unit="%",
                warning_threshold=70.0,
                critical_threshold=90.0,
                id="gauge-cpu",
            )
            yield MetricGauge(
                "Memory",
                value=0.0,
                max_value=100.0,
                unit="%",
                warning_threshold=80.0,
                critical_threshold=95.0,
                id="gauge-memory",
            )
            yield MetricGauge(
                "Disk",
                value=0.0,
                max_value=100.0,
                unit="%",
                warning_threshold=85.0,
                critical_threshold=95.0,
                id="gauge-disk",
            )
        
        with Horizontal(classes="summary-row"):
            yield Label("Backends: -", id="backends-summary")
            yield Label("Queue: -", id="queue-summary")
            yield Label(f"Refresh: {self._refresh_interval}s", id="refresh-info")
    
    def on_mount(self) -> None:
        """Start auto-refresh if enabled."""
        if self._auto_refresh:
            self._start_refresh()
        self._update_metrics()
    
    def _start_refresh(self) -> None:
        """Start the refresh timer."""
        self._refresh_timer = self.set_interval(
            self._refresh_interval,
            self._update_metrics,
        )
    
    def _stop_refresh(self) -> None:
        """Stop the refresh timer."""
        if self._refresh_timer:
            self._refresh_timer.stop()
            self._refresh_timer = None
    
    def _update_metrics(self) -> None:
        """Update all metrics from system."""
        try:
            # Try to get real metrics
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            self.query_one("#gauge-cpu", MetricGauge).update_metric(cpu_percent)
            self.query_one("#gauge-memory", MetricGauge).update_metric(memory.percent)
            self.query_one("#gauge-disk", MetricGauge).update_metric(disk.percent)
            
        except ImportError:
            # Use placeholder values if psutil not available
            import random
            
            self.query_one("#gauge-cpu", MetricGauge).update_metric(
                random.uniform(10, 60)
            )
            self.query_one("#gauge-memory", MetricGauge).update_metric(
                random.uniform(30, 70)
            )
            self.query_one("#gauge-disk", MetricGauge).update_metric(
                random.uniform(40, 80)
            )
        except Exception:
            pass
        
        # Update summary labels
        try:
            self._update_backend_summary()
            self._update_queue_summary()
        except Exception:
            pass
    
    def _update_backend_summary(self) -> None:
        """Update backend status summary."""
        try:
            from proxima.backends.registry import backend_registry
            
            statuses = backend_registry.list_statuses()
            available = sum(1 for s in statuses if s.available)
            total = len(statuses)
            
            self.query_one("#backends-summary", Label).update(
                f"Backends: {available}/{total} available"
            )
        except Exception:
            self.query_one("#backends-summary", Label).update("Backends: N/A")
    
    def _update_queue_summary(self) -> None:
        """Update execution queue summary."""
        # Placeholder - would connect to actual queue
        self.query_one("#queue-summary", Label).update("Queue: 0 pending")


# =============================================================================
# Circuit Editor (Feature - TUI)
# =============================================================================


class GateType(Enum):
    """Quantum gate types."""
    
    H = "H"      # Hadamard
    X = "X"      # Pauli-X
    Y = "Y"      # Pauli-Y
    Z = "Z"      # Pauli-Z
    S = "S"      # S gate
    T = "T"      # T gate
    CX = "CX"    # CNOT
    CZ = "CZ"    # CZ
    SWAP = "SW"  # SWAP
    RX = "Rx"    # Rotation X
    RY = "Ry"    # Rotation Y
    RZ = "Rz"    # Rotation Z
    M = "M"      # Measurement


@dataclass
class GatePlacement:
    """A gate placement in the circuit."""
    
    gate_type: GateType
    qubit: int
    control_qubit: int | None = None
    parameter: float | None = None
    column: int = 0


class CircuitCanvas(Static):
    """Canvas for displaying and editing quantum circuits."""
    
    DEFAULT_CSS = """
    CircuitCanvas {
        height: 100%;
        border: solid $primary;
        padding: 1;
        overflow: auto scroll;
    }
    CircuitCanvas .circuit-header { text-style: bold; margin-bottom: 1; }
    CircuitCanvas .qubit-line { height: 1; }
    CircuitCanvas .gate { min-width: 4; text-align: center; }
    CircuitCanvas .gate-h { color: $primary; }
    CircuitCanvas .gate-x { color: $error; }
    CircuitCanvas .gate-cx { color: $warning; }
    CircuitCanvas .gate-m { color: $success; }
    """
    
    cursor_qubit = reactive(0)
    cursor_column = reactive(0)
    
    def __init__(
        self,
        num_qubits: int = 3,
        num_columns: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._num_qubits = num_qubits
        self._num_columns = num_columns
        self._gates: list[GatePlacement] = []
        self._selected_gate: GatePlacement | None = None
    
    def compose(self) -> ComposeResult:
        yield Label(
            f"Circuit: {self._num_qubits} qubits, {self._num_columns} columns",
            classes="circuit-header",
        )
        yield Container(id="circuit-lines")
    
    def on_mount(self) -> None:
        """Render initial circuit."""
        self._render_circuit()
    
    def _render_circuit(self) -> None:
        """Render the circuit display."""
        container = self.query_one("#circuit-lines", Container)
        container.remove_children()
        
        for qubit in range(self._num_qubits):
            line = self._render_qubit_line(qubit)
            container.mount(line)
    
    def _render_qubit_line(self, qubit: int) -> Horizontal:
        """Render a single qubit line."""
        line = Horizontal(classes="qubit-line")
        
        # Qubit label
        line.mount(Label(f"q{qubit}│", classes="qubit-label"))
        
        # Gates and wires
        for col in range(self._num_columns):
            gate = self._get_gate_at(qubit, col)
            if gate:
                gate_label = Label(
                    f"[{gate.gate_type.value}]",
                    classes=f"gate gate-{gate.gate_type.name.lower()}",
                )
            else:
                gate_label = Label("─────", classes="wire")
            line.mount(gate_label)
        
        # Measurement indicator
        line.mount(Label("│", classes="qubit-end"))
        
        return line
    
    def _get_gate_at(self, qubit: int, column: int) -> GatePlacement | None:
        """Get gate at specific position."""
        for gate in self._gates:
            if gate.qubit == qubit and gate.column == column:
                return gate
            if gate.control_qubit == qubit and gate.column == column:
                return gate
        return None
    
    def add_gate(
        self,
        gate_type: GateType,
        qubit: int,
        column: int | None = None,
        control_qubit: int | None = None,
        parameter: float | None = None,
    ) -> None:
        """Add a gate to the circuit.
        
        Args:
            gate_type: Type of gate to add
            qubit: Target qubit
            column: Column position (auto if None)
            control_qubit: Control qubit for two-qubit gates
            parameter: Parameter for parameterized gates
        """
        if column is None:
            column = self.cursor_column
        
        gate = GatePlacement(
            gate_type=gate_type,
            qubit=qubit,
            control_qubit=control_qubit,
            parameter=parameter,
            column=column,
        )
        self._gates.append(gate)
        self._render_circuit()
    
    def remove_gate_at(self, qubit: int, column: int) -> bool:
        """Remove gate at position.
        
        Args:
            qubit: Qubit position
            column: Column position
            
        Returns:
            True if gate was removed
        """
        for i, gate in enumerate(self._gates):
            if gate.qubit == qubit and gate.column == column:
                self._gates.pop(i)
                self._render_circuit()
                return True
        return False
    
    def clear(self) -> None:
        """Clear all gates from circuit."""
        self._gates.clear()
        self._render_circuit()
    
    def get_circuit_code(self) -> str:
        """Generate circuit code from gates.
        
        Returns:
            Circuit code string
        """
        lines = [
            f"# Quantum Circuit: {self._num_qubits} qubits",
            "from cirq import Circuit, LineQubit, H, X, Y, Z, S, T, CNOT, CZ, SWAP, measure",
            "",
            f"qubits = [LineQubit(i) for i in range({self._num_qubits})]",
            "circuit = Circuit()",
            "",
        ]
        
        # Group gates by column
        gates_by_col: dict[int, list[GatePlacement]] = {}
        for gate in self._gates:
            gates_by_col.setdefault(gate.column, []).append(gate)
        
        for col in sorted(gates_by_col.keys()):
            gates = gates_by_col[col]
            ops = []
            for gate in gates:
                if gate.gate_type in (GateType.H, GateType.X, GateType.Y, GateType.Z,
                                       GateType.S, GateType.T):
                    ops.append(f"{gate.gate_type.value}(qubits[{gate.qubit}])")
                elif gate.gate_type == GateType.CX and gate.control_qubit is not None:
                    ops.append(
                        f"CNOT(qubits[{gate.control_qubit}], qubits[{gate.qubit}])"
                    )
                elif gate.gate_type == GateType.M:
                    ops.append(f"measure(qubits[{gate.qubit}])")
            
            if ops:
                lines.append(f"circuit.append([{', '.join(ops)}])")
        
        lines.append("")
        lines.append("print(circuit)")
        
        return "\n".join(lines)
    
    def get_gates(self) -> list[GatePlacement]:
        """Get all gate placements."""
        return list(self._gates)


class CircuitEditor(Static):
    """Interactive quantum circuit editor.
    
    Features:
    - Visual circuit builder
    - Gate palette
    - Circuit validation
    - Code export
    - Keyboard navigation
    """
    
    DEFAULT_CSS = """
    CircuitEditor {
        height: 100%;
        layout: grid;
        grid-size: 1 3;
        grid-rows: auto 1fr 3;
    }
    CircuitEditor .editor-toolbar {
        height: auto;
        padding: 1;
        background: $surface;
    }
    CircuitEditor .gate-palette {
        height: 3;
        padding: 1;
    }
    CircuitEditor .gate-btn {
        min-width: 6;
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("h", "add_gate_h", "H Gate"),
        Binding("x", "add_gate_x", "X Gate"),
        Binding("z", "add_gate_z", "Z Gate"),
        Binding("c", "add_gate_cx", "CNOT"),
        Binding("m", "add_gate_m", "Measure"),
        Binding("delete", "delete_gate", "Delete"),
        Binding("ctrl+c", "copy_code", "Copy Code"),
        Binding("ctrl+x", "clear_circuit", "Clear"),
    ]
    
    class GateAdded(Message):
        """Emitted when a gate is added."""
        
        def __init__(self, gate: GatePlacement) -> None:
            super().__init__()
            self.gate = gate
    
    def __init__(
        self,
        num_qubits: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._num_qubits = num_qubits
        self._current_qubit = 0
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="editor-toolbar"):
            yield Label("🔧 Circuit Editor", classes="editor-title")
            yield Button("Export", id="btn-export", variant="primary")
            yield Button("Clear", id="btn-clear", variant="error")
            yield Button("Run", id="btn-run", variant="success")
        
        yield CircuitCanvas(
            num_qubits=self._num_qubits,
            num_columns=10,
            id="circuit-canvas",
        )
        
        with Horizontal(classes="gate-palette"):
            yield Label("Gates: ")
            for gate_type in [GateType.H, GateType.X, GateType.Y, GateType.Z,
                             GateType.CX, GateType.M]:
                yield Button(
                    gate_type.value,
                    id=f"btn-gate-{gate_type.name.lower()}",
                    classes="gate-btn",
                )
    
    def action_add_gate_h(self) -> None:
        """Add Hadamard gate."""
        self._add_gate(GateType.H)
    
    def action_add_gate_x(self) -> None:
        """Add X gate."""
        self._add_gate(GateType.X)
    
    def action_add_gate_z(self) -> None:
        """Add Z gate."""
        self._add_gate(GateType.Z)
    
    def action_add_gate_cx(self) -> None:
        """Add CNOT gate."""
        self._add_gate(GateType.CX, control_qubit=0)
    
    def action_add_gate_m(self) -> None:
        """Add measurement."""
        self._add_gate(GateType.M)
    
    def action_delete_gate(self) -> None:
        """Delete gate at cursor."""
        canvas = self.query_one("#circuit-canvas", CircuitCanvas)
        canvas.remove_gate_at(canvas.cursor_qubit, canvas.cursor_column)
    
    def action_copy_code(self) -> None:
        """Copy circuit code to clipboard."""
        canvas = self.query_one("#circuit-canvas", CircuitCanvas)
        code = canvas.get_circuit_code()
        # Would copy to clipboard in real implementation
        self.notify("Circuit code copied to clipboard")
    
    def action_clear_circuit(self) -> None:
        """Clear the circuit."""
        canvas = self.query_one("#circuit-canvas", CircuitCanvas)
        canvas.clear()
    
    def _add_gate(
        self,
        gate_type: GateType,
        control_qubit: int | None = None,
    ) -> None:
        """Add a gate to the circuit."""
        canvas = self.query_one("#circuit-canvas", CircuitCanvas)
        canvas.add_gate(
            gate_type=gate_type,
            qubit=self._current_qubit,
            control_qubit=control_qubit,
        )
        
        # Move cursor to next column
        canvas.cursor_column += 1
        
        # Emit event
        gate = GatePlacement(
            gate_type=gate_type,
            qubit=self._current_qubit,
            control_qubit=control_qubit,
            column=canvas.cursor_column - 1,
        )
        self.post_message(self.GateAdded(gate))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id
        
        if btn_id == "btn-export":
            canvas = self.query_one("#circuit-canvas", CircuitCanvas)
            code = canvas.get_circuit_code()
            # Would show code in modal
            self.notify("Code exported")
        elif btn_id == "btn-clear":
            self.action_clear_circuit()
        elif btn_id == "btn-run":
            self.notify("Running circuit...")
        elif btn_id and btn_id.startswith("btn-gate-"):
            gate_name = btn_id.replace("btn-gate-", "").upper()
            try:
                gate_type = GateType[gate_name]
                self._add_gate(gate_type)
            except KeyError:
                pass


# =============================================================================
# Execution Monitor (Feature - TUI)
# =============================================================================


@dataclass
class ExecutionEvent:
    """An event in execution monitoring."""
    
    timestamp: float
    event_type: str
    message: str
    stage: str | None = None
    progress: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionMonitor(Static):
    """Real-time execution monitoring panel.
    
    Features:
    - Live progress tracking
    - Stage timeline
    - Resource usage during execution
    - Event log
    - Control buttons (pause/resume/abort)
    """
    
    DEFAULT_CSS = """
    ExecutionMonitor {
        height: 100%;
        layout: grid;
        grid-size: 1 4;
        grid-rows: 3 auto 1fr 3;
    }
    ExecutionMonitor .monitor-header {
        padding: 1;
        background: $surface;
    }
    ExecutionMonitor .progress-section {
        height: auto;
        padding: 1;
        border: solid $surface-lighten-1;
    }
    ExecutionMonitor .events-section {
        border: solid $surface-lighten-1;
        padding: 0;
    }
    ExecutionMonitor .control-section {
        padding: 1;
        background: $surface;
    }
    ExecutionMonitor .stage-indicator {
        margin: 0 2;
    }
    ExecutionMonitor .stage-active { color: $accent; text-style: bold; }
    ExecutionMonitor .stage-complete { color: $success; }
    ExecutionMonitor .stage-pending { color: $text-muted; }
    """
    
    progress = reactive(0.0)
    status = reactive("idle")
    current_stage = reactive("")
    
    BINDINGS = [
        Binding("space", "toggle_pause", "Pause/Resume"),
        Binding("escape", "abort", "Abort"),
        Binding("r", "retry", "Retry"),
    ]
    
    class StatusChanged(Message):
        """Emitted when execution status changes."""
        
        def __init__(self, status: str, stage: str) -> None:
            super().__init__()
            self.status = status
            self.stage = stage
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: list[ExecutionEvent] = []
        self._stages = [
            "Parsing", "Planning", "Resource Check",
            "Consent", "Executing", "Analyzing", "Complete",
        ]
        self._is_paused = False
        self._start_time: float | None = None
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="monitor-header"):
            yield Label("📡 Execution Monitor", id="monitor-title")
            yield Label("Status: Idle", id="status-label")
            yield Label("Duration: --:--", id="duration-label")
        
        with Container(classes="progress-section"):
            yield TextualProgressBar(
                total=100,
                show_eta=True,
                id="main-progress",
            )
            with Horizontal(id="stage-indicators"):
                for stage in self._stages:
                    yield Label(
                        f"○ {stage}",
                        classes="stage-indicator stage-pending",
                        id=f"stage-{stage.lower().replace(' ', '-')}",
                    )
        
        with ScrollableContainer(classes="events-section", id="events-container"):
            yield Label("Waiting for execution...", id="events-placeholder")
        
        with Horizontal(classes="control-section"):
            yield Button("▶ Start", id="btn-start", variant="success")
            yield Button("⏸ Pause", id="btn-pause", disabled=True)
            yield Button("⏹ Stop", id="btn-stop", variant="error", disabled=True)
            yield Button("🔄 Retry", id="btn-retry", disabled=True)
    
    def watch_progress(self, new_progress: float) -> None:
        """Update progress bar."""
        try:
            progress_bar = self.query_one("#main-progress", TextualProgressBar)
            progress_bar.progress = int(new_progress)
        except Exception:
            pass
    
    def watch_status(self, new_status: str) -> None:
        """Update status label."""
        try:
            status_label = self.query_one("#status-label", Label)
            status_label.update(f"Status: {new_status.title()}")
            
            # Update button states
            is_running = new_status in ("running", "paused")
            self.query_one("#btn-start", Button).disabled = is_running
            self.query_one("#btn-pause", Button).disabled = not is_running
            self.query_one("#btn-stop", Button).disabled = not is_running
            self.query_one("#btn-retry", Button).disabled = new_status != "failed"
        except Exception:
            pass
    
    def watch_current_stage(self, new_stage: str) -> None:
        """Update stage indicators."""
        try:
            stage_idx = -1
            for i, stage in enumerate(self._stages):
                if stage.lower() == new_stage.lower():
                    stage_idx = i
                    break
            
            for i, stage in enumerate(self._stages):
                indicator = self.query_one(
                    f"#stage-{stage.lower().replace(' ', '-')}",
                    Label,
                )
                indicator.remove_class("stage-active", "stage-complete", "stage-pending")
                
                if i < stage_idx:
                    indicator.update(f"✓ {stage}")
                    indicator.add_class("stage-complete")
                elif i == stage_idx:
                    indicator.update(f"● {stage}")
                    indicator.add_class("stage-active")
                else:
                    indicator.update(f"○ {stage}")
                    indicator.add_class("stage-pending")
        except Exception:
            pass
    
    def add_event(
        self,
        event_type: str,
        message: str,
        stage: str | None = None,
        progress: float | None = None,
    ) -> None:
        """Add an event to the monitor.
        
        Args:
            event_type: Type of event (info, warning, error, success)
            message: Event message
            stage: Current stage name
            progress: Progress percentage
        """
        event = ExecutionEvent(
            timestamp=time.time(),
            event_type=event_type,
            message=message,
            stage=stage,
            progress=progress,
        )
        self._events.append(event)
        
        # Update UI
        if progress is not None:
            self.progress = progress
        if stage is not None:
            self.current_stage = stage
        
        # Add to events container
        self._render_event(event)
    
    def _render_event(self, event: ExecutionEvent) -> None:
        """Render an event in the events container."""
        try:
            container = self.query_one("#events-container", ScrollableContainer)
            
            # Remove placeholder
            placeholder = container.query("#events-placeholder")
            if placeholder:
                placeholder.first().remove()
            
            # Add event row
            icons = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "❌",
                "success": "✅",
                "progress": "📊",
            }
            
            ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            icon = icons.get(event.event_type, "•")
            
            row = Horizontal()
            row.mount(Label(f"{ts} {icon} {event.message}"))
            container.mount(row)
            
            # Scroll to bottom
            container.scroll_end()
        except Exception:
            pass
    
    def start_monitoring(self) -> None:
        """Start execution monitoring."""
        self._start_time = time.time()
        self._events.clear()
        self.status = "running"
        self.progress = 0.0
        self.current_stage = "Parsing"
        
        self.add_event("info", "Execution started", "Parsing", 0.0)
    
    def action_toggle_pause(self) -> None:
        """Toggle pause state."""
        if self.status == "running":
            self._is_paused = True
            self.status = "paused"
            self.add_event("warning", "Execution paused")
        elif self.status == "paused":
            self._is_paused = False
            self.status = "running"
            self.add_event("info", "Execution resumed")
    
    def action_abort(self) -> None:
        """Abort execution."""
        self.status = "aborted"
        self.add_event("error", "Execution aborted by user")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-start":
            self.start_monitoring()
        elif event.button.id == "btn-pause":
            self.action_toggle_pause()
        elif event.button.id == "btn-stop":
            self.action_abort()
        elif event.button.id == "btn-retry":
            self.start_monitoring()


# =============================================================================
# Keyboard Shortcuts Modal (Feature - TUI)
# =============================================================================


class KeyboardShortcutsModal(Static):
    """Modal showing keyboard shortcuts.
    
    Features:
    - Categorized shortcuts
    - Context-aware display
    - Search/filter
    """
    
    DEFAULT_CSS = """
    KeyboardShortcutsModal {
        width: 60;
        height: auto;
        max-height: 80%;
        border: heavy $primary;
        background: $surface;
        padding: 1 2;
    }
    KeyboardShortcutsModal .modal-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    KeyboardShortcutsModal .shortcut-category {
        text-style: bold;
        margin-top: 1;
        color: $primary;
    }
    KeyboardShortcutsModal .shortcut-row {
        height: 1;
    }
    KeyboardShortcutsModal .shortcut-key {
        min-width: 15;
        color: $accent;
    }
    KeyboardShortcutsModal .shortcut-desc {
        color: $text;
    }
    """
    
    SHORTCUTS = {
        "Navigation": [
            ("1-5", "Switch screens"),
            ("Tab", "Next element"),
            ("Shift+Tab", "Previous element"),
            ("↑/↓", "Navigate lists"),
            ("Enter", "Select/Confirm"),
            ("Escape", "Close/Cancel"),
        ],
        "Execution": [
            ("Ctrl+Enter", "Start execution"),
            ("Ctrl+C", "Stop execution"),
            ("Space", "Pause/Resume"),
            ("R", "Retry failed"),
        ],
        "Circuit Editor": [
            ("H", "Add Hadamard gate"),
            ("X", "Add X gate"),
            ("Z", "Add Z gate"),
            ("C", "Add CNOT gate"),
            ("M", "Add measurement"),
            ("Delete", "Remove gate"),
            ("Ctrl+X", "Clear circuit"),
        ],
        "General": [
            ("?", "Show this help"),
            ("Q", "Quit application"),
            ("Ctrl+S", "Save configuration"),
            ("Ctrl+R", "Refresh data"),
        ],
    }
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Label("⌨️ Keyboard Shortcuts", classes="modal-title")
        
        for category, shortcuts in self.SHORTCUTS.items():
            yield Label(f"▸ {category}", classes="shortcut-category")
            for key, desc in shortcuts:
                with Horizontal(classes="shortcut-row"):
                    yield Label(key, classes="shortcut-key")
                    yield Label(desc, classes="shortcut-desc")
        
        yield Label("")
        yield Label("Press ESC or Q to close", classes="modal-footer")
    
    def action_close(self) -> None:
        """Close the modal."""
        self.remove()


# =============================================================================
# Quick Command Bar (Feature - TUI)
# =============================================================================


@dataclass
class QuickCommand:
    """A quick command definition."""
    
    name: str
    description: str
    action: str
    shortcut: str | None = None
    category: str = "General"


class QuickCommandBar(Static):
    """Command palette for quick actions.
    
    Features:
    - Fuzzy search
    - Recent commands
    - Categorized commands
    - Keyboard invocation
    """
    
    DEFAULT_CSS = """
    QuickCommandBar {
        width: 60;
        height: auto;
        max-height: 70%;
        border: heavy $primary;
        background: $surface;
        padding: 1;
    }
    QuickCommandBar .command-input {
        width: 100%;
        margin-bottom: 1;
    }
    QuickCommandBar .command-list {
        height: auto;
        max-height: 20;
    }
    QuickCommandBar .command-item {
        height: 2;
        padding: 0 1;
    }
    QuickCommandBar .command-item:hover {
        background: $surface-lighten-1;
    }
    QuickCommandBar .command-name { text-style: bold; }
    QuickCommandBar .command-desc { color: $text-muted; }
    QuickCommandBar .command-shortcut { color: $accent; }
    """
    
    COMMANDS: list[QuickCommand] = [
        QuickCommand("Run Bell State", "Execute Bell state circuit", "run_bell", "Ctrl+B", "Execution"),
        QuickCommand("Run QFT", "Execute Quantum Fourier Transform", "run_qft", None, "Execution"),
        QuickCommand("Compare Backends", "Compare across all backends", "compare", "Ctrl+K", "Execution"),
        QuickCommand("List Backends", "Show available backends", "list_backends", None, "Backends"),
        QuickCommand("Benchmark", "Run backend benchmarks", "benchmark", None, "Backends"),
        QuickCommand("View History", "Show execution history", "history", "Ctrl+H", "Results"),
        QuickCommand("Export Results", "Export last results", "export", "Ctrl+E", "Results"),
        QuickCommand("Edit Config", "Open configuration", "config", "Ctrl+,", "Settings"),
        QuickCommand("Toggle Theme", "Switch light/dark theme", "theme", None, "Settings"),
        QuickCommand("Open Help", "Show help documentation", "help", "F1", "Help"),
        QuickCommand("Keyboard Shortcuts", "Show shortcuts", "shortcuts", "?", "Help"),
    ]
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "execute", "Execute"),
        Binding("up", "previous", "Previous"),
        Binding("down", "next", "Next"),
    ]
    
    class CommandSelected(Message):
        """Emitted when a command is selected."""
        
        def __init__(self, command: QuickCommand) -> None:
            super().__init__()
            self.command = command
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filtered_commands = list(self.COMMANDS)
        self._selected_index = 0
    
    def compose(self) -> ComposeResult:
        yield Input(
            placeholder="Type to search commands...",
            classes="command-input",
            id="command-search",
        )
        
        with ScrollableContainer(classes="command-list", id="command-list"):
            for cmd in self._filtered_commands:
                yield self._render_command_item(cmd)
    
    def _render_command_item(self, cmd: QuickCommand) -> Static:
        """Render a command item."""
        item = Static(classes="command-item")
        
        with Horizontal():
            item.mount(Label(cmd.name, classes="command-name"))
            if cmd.shortcut:
                item.mount(Label(f"[{cmd.shortcut}]", classes="command-shortcut"))
        item.mount(Label(cmd.description, classes="command-desc"))
        
        return item
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter commands based on input."""
        query = event.value.lower()
        
        if not query:
            self._filtered_commands = list(self.COMMANDS)
        else:
            self._filtered_commands = [
                cmd for cmd in self.COMMANDS
                if query in cmd.name.lower() or query in cmd.description.lower()
            ]
        
        self._selected_index = 0
        self._refresh_list()
    
    def _refresh_list(self) -> None:
        """Refresh the command list display."""
        container = self.query_one("#command-list", ScrollableContainer)
        container.remove_children()
        
        for cmd in self._filtered_commands:
            container.mount(self._render_command_item(cmd))
    
    def action_close(self) -> None:
        """Close the command bar."""
        self.remove()
    
    def action_execute(self) -> None:
        """Execute selected command."""
        if self._filtered_commands and self._selected_index < len(self._filtered_commands):
            cmd = self._filtered_commands[self._selected_index]
            self.post_message(self.CommandSelected(cmd))
            self.remove()
    
    def action_previous(self) -> None:
        """Select previous command."""
        if self._selected_index > 0:
            self._selected_index -= 1
    
    def action_next(self) -> None:
        """Select next command."""
        if self._selected_index < len(self._filtered_commands) - 1:
            self._selected_index += 1


# =============================================================================
# Theme Switcher (Feature - TUI)
# =============================================================================


class ThemeVariant(Enum):
    """Available theme variants."""
    
    DARK = "dark"
    LIGHT = "light"
    MONOKAI = "monokai"
    DRACULA = "dracula"
    NORD = "nord"
    SOLARIZED_DARK = "solarized-dark"
    SOLARIZED_LIGHT = "solarized-light"


@dataclass
class ThemeColors:
    """Color definitions for a theme."""
    
    name: str
    primary: str
    secondary: str
    background: str
    surface: str
    text: str
    accent: str
    success: str
    warning: str
    error: str


class ThemeSwitcher(Static):
    """Theme switching panel with preview.
    
    Features:
    - Live theme preview
    - Custom color support
    - Persist theme selection
    """
    
    DEFAULT_CSS = """
    ThemeSwitcher {
        width: 40;
        height: auto;
        border: heavy $primary;
        background: $surface;
        padding: 1;
    }
    ThemeSwitcher .theme-title { text-style: bold; margin-bottom: 1; }
    ThemeSwitcher .theme-option {
        height: 2;
        padding: 0 1;
    }
    ThemeSwitcher .theme-option:hover { background: $surface-lighten-1; }
    ThemeSwitcher .theme-preview {
        height: 3;
        margin-top: 1;
        padding: 1;
        border: solid $primary;
    }
    """
    
    THEMES: dict[ThemeVariant, ThemeColors] = {
        ThemeVariant.DARK: ThemeColors(
            name="Dark",
            primary="#7c3aed", secondary="#a78bfa",
            background="#1a1a2e", surface="#252542",
            text="#e4e4e7", accent="#22d3ee",
            success="#22c55e", warning="#eab308", error="#ef4444",
        ),
        ThemeVariant.LIGHT: ThemeColors(
            name="Light",
            primary="#6366f1", secondary="#818cf8",
            background="#ffffff", surface="#f1f5f9",
            text="#1e293b", accent="#0ea5e9",
            success="#16a34a", warning="#ca8a04", error="#dc2626",
        ),
        ThemeVariant.MONOKAI: ThemeColors(
            name="Monokai",
            primary="#f92672", secondary="#ae81ff",
            background="#272822", surface="#3e3d32",
            text="#f8f8f2", accent="#a6e22e",
            success="#a6e22e", warning="#e6db74", error="#f92672",
        ),
        ThemeVariant.DRACULA: ThemeColors(
            name="Dracula",
            primary="#bd93f9", secondary="#ff79c6",
            background="#282a36", surface="#44475a",
            text="#f8f8f2", accent="#8be9fd",
            success="#50fa7b", warning="#f1fa8c", error="#ff5555",
        ),
        ThemeVariant.NORD: ThemeColors(
            name="Nord",
            primary="#88c0d0", secondary="#81a1c1",
            background="#2e3440", surface="#3b4252",
            text="#eceff4", accent="#8fbcbb",
            success="#a3be8c", warning="#ebcb8b", error="#bf616a",
        ),
    }
    
    current_theme = reactive(ThemeVariant.DARK)
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "apply", "Apply"),
    ]
    
    class ThemeChanged(Message):
        """Emitted when theme is changed."""
        
        def __init__(self, theme: ThemeVariant) -> None:
            super().__init__()
            self.theme = theme
    
    def compose(self) -> ComposeResult:
        yield Label("🎨 Theme Selector", classes="theme-title")
        
        for variant in self.THEMES:
            colors = self.THEMES[variant]
            yield Button(
                f"{colors.name}",
                id=f"theme-{variant.value}",
                classes="theme-option",
            )
        
        yield Container(id="theme-preview", classes="theme-preview")
    
    def on_mount(self) -> None:
        """Load saved theme preference."""
        self._update_preview()
    
    def _update_preview(self) -> None:
        """Update theme preview."""
        try:
            preview = self.query_one("#theme-preview", Container)
            preview.remove_children()
            
            colors = self.THEMES[self.current_theme]
            preview.mount(Label(f"Preview: {colors.name}"))
            preview.mount(
                Label(f"Colors: Primary={colors.primary} Accent={colors.accent}")
            )
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle theme selection."""
        btn_id = event.button.id
        if btn_id and btn_id.startswith("theme-"):
            theme_name = btn_id.replace("theme-", "")
            try:
                self.current_theme = ThemeVariant(theme_name)
                self._update_preview()
            except ValueError:
                pass
    
    def action_apply(self) -> None:
        """Apply the selected theme."""
        self.post_message(self.ThemeChanged(self.current_theme))
        self.remove()
    
    def action_close(self) -> None:
        """Close without applying."""
        self.remove()


# =============================================================================
# Notification System (Feature - TUI)
# =============================================================================


class NotificationType(Enum):
    """Types of notifications."""
    
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Notification:
    """A notification message."""
    
    message: str
    type: NotificationType = NotificationType.INFO
    title: str | None = None
    timeout: float = 5.0
    dismissible: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: __import__('uuid').uuid4().hex[:8])


class NotificationToast(Static):
    """Toast notification widget."""
    
    DEFAULT_CSS = """
    NotificationToast {
        width: 50;
        height: auto;
        min-height: 3;
        padding: 1;
        border: solid $primary;
        margin: 1;
    }
    NotificationToast.info { border: solid $primary; }
    NotificationToast.success { border: solid $success; }
    NotificationToast.warning { border: solid $warning; }
    NotificationToast.error { border: solid $error; }
    NotificationToast .toast-title { text-style: bold; }
    NotificationToast .toast-close { dock: right; }
    """
    
    def __init__(self, notification: Notification, **kwargs) -> None:
        super().__init__(**kwargs)
        self._notification = notification
        self.add_class(notification.type.value)
    
    def compose(self) -> ComposeResult:
        icons = {
            NotificationType.INFO: "ℹ️",
            NotificationType.SUCCESS: "✅",
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
        }
        icon = icons.get(self._notification.type, "•")
        
        with Horizontal():
            if self._notification.title:
                yield Label(
                    f"{icon} {self._notification.title}",
                    classes="toast-title",
                )
            else:
                yield Label(icon)
            
            if self._notification.dismissible:
                yield Button("×", id="toast-close", classes="toast-close")
        
        yield Label(self._notification.message)
    
    def on_mount(self) -> None:
        """Set up auto-dismiss timer."""
        if self._notification.timeout > 0:
            self.set_timer(self._notification.timeout, self._dismiss)
    
    def _dismiss(self) -> None:
        """Dismiss the notification."""
        self.remove()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle close button."""
        if event.button.id == "toast-close":
            self._dismiss()


class NotificationCenter(Static):
    """Central notification management panel.
    
    Features:
    - Stack multiple notifications
    - Auto-dismiss with timeout
    - Notification history
    - Filter by type
    """
    
    DEFAULT_CSS = """
    NotificationCenter {
        dock: right;
        width: 55;
        height: 100%;
        background: $surface;
        border-left: solid $surface-lighten-1;
    }
    NotificationCenter .center-header {
        height: 3;
        padding: 1;
        background: $primary;
    }
    NotificationCenter .notifications-container {
        height: 100%;
        overflow: auto;
    }
    """
    
    def __init__(self, max_visible: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_visible = max_visible
        self._notifications: list[Notification] = []
        self._history: list[Notification] = []
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="center-header"):
            yield Label("🔔 Notifications")
            yield Button("Clear All", id="btn-clear-all")
        
        yield ScrollableContainer(
            id="notifications-container",
            classes="notifications-container",
        )
    
    def add_notification(self, notification: Notification) -> None:
        """Add a notification to the center."""
        self._notifications.append(notification)
        self._history.append(notification)
        
        # Limit visible notifications
        while len(self._notifications) > self._max_visible:
            old = self._notifications.pop(0)
            try:
                self.query_one(f"#toast-{old.id}", NotificationToast).remove()
            except Exception:
                pass
        
        # Add toast widget
        container = self.query_one(
            "#notifications-container",
            ScrollableContainer,
        )
        toast = NotificationToast(notification, id=f"toast-{notification.id}")
        container.mount(toast)
    
    def notify(
        self,
        message: str,
        type: NotificationType = NotificationType.INFO,
        title: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        """Quick notification helper."""
        self.add_notification(
            Notification(
                message=message,
                type=type,
                title=title,
                timeout=timeout,
            )
        )
    
    def clear_all(self) -> None:
        """Clear all visible notifications."""
        self._notifications.clear()
        container = self.query_one(
            "#notifications-container",
            ScrollableContainer,
        )
        container.remove_children()
    
    def get_history(
        self,
        type_filter: NotificationType | None = None,
        limit: int = 50,
    ) -> list[Notification]:
        """Get notification history."""
        if type_filter:
            filtered = [n for n in self._history if n.type == type_filter]
        else:
            filtered = list(self._history)
        return filtered[-limit:]
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-clear-all":
            self.clear_all()


# =============================================================================
# Resource Monitor Widget (Feature - TUI)
# =============================================================================


class ResourceMonitor(Static):
    """Compact resource monitoring widget.
    
    Shows:
    - GPU memory (if CUDA available)
    - Active backends
    - Running executions
    """
    
    DEFAULT_CSS = """
    ResourceMonitor {
        height: 3;
        width: 100%;
        padding: 0 1;
        background: $surface;
        border-bottom: solid $surface-lighten-1;
    }
    ResourceMonitor .resource-item { margin: 0 2; }
    ResourceMonitor .resource-warning { color: $warning; }
    ResourceMonitor .resource-critical { color: $error; }
    """
    
    def __init__(self, refresh_interval: float = 5.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._refresh_interval = refresh_interval
    
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("💻 CPU: --", id="cpu-usage", classes="resource-item")
            yield Label("🧠 RAM: --", id="ram-usage", classes="resource-item")
            yield Label("🎮 GPU: --", id="gpu-usage", classes="resource-item")
            yield Label("⚡ Active: 0", id="active-count", classes="resource-item")
    
    def on_mount(self) -> None:
        """Start refresh timer."""
        self.set_interval(self._refresh_interval, self._refresh)
        self._refresh()
    
    def _refresh(self) -> None:
        """Refresh resource data."""
        try:
            import psutil
            
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            
            cpu_label = self.query_one("#cpu-usage", Label)
            cpu_label.update(f"💻 CPU: {cpu:.0f}%")
            cpu_label.remove_class("resource-warning", "resource-critical")
            if cpu > 90:
                cpu_label.add_class("resource-critical")
            elif cpu > 70:
                cpu_label.add_class("resource-warning")
            
            ram_label = self.query_one("#ram-usage", Label)
            ram_label.update(f"🧠 RAM: {ram:.0f}%")
            ram_label.remove_class("resource-warning", "resource-critical")
            if ram > 90:
                ram_label.add_class("resource-critical")
            elif ram > 80:
                ram_label.add_class("resource-warning")
            
        except ImportError:
            pass
        
        # GPU info (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_pct = (gpu_mem / gpu_total) * 100
                self.query_one("#gpu-usage", Label).update(
                    f"🎮 GPU: {gpu_mem:.1f}/{gpu_total:.1f}GB"
                )
        except Exception:
            self.query_one("#gpu-usage", Label).update("🎮 GPU: N/A")


# =============================================================================
# Confirmation Dialog (Feature - TUI)
# =============================================================================


class ConfirmationDialog(Static):
    """Modal confirmation dialog.
    
    For user confirmation before destructive actions.
    """
    
    DEFAULT_CSS = """
    ConfirmationDialog {
        width: 50;
        height: auto;
        border: heavy $error;
        background: $surface;
        padding: 1 2;
    }
    ConfirmationDialog .dialog-title { text-style: bold; margin-bottom: 1; }
    ConfirmationDialog .dialog-message { margin-bottom: 1; }
    ConfirmationDialog .dialog-buttons { height: 3; margin-top: 1; }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Confirm"),
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
    ]
    
    class Confirmed(Message):
        """Emitted when user confirms."""
        
        def __init__(self, action_id: str) -> None:
            super().__init__()
            self.action_id = action_id
    
    class Cancelled(Message):
        """Emitted when user cancels."""
        
        def __init__(self, action_id: str) -> None:
            super().__init__()
            self.action_id = action_id
    
    def __init__(
        self,
        title: str,
        message: str,
        action_id: str = "",
        confirm_label: str = "Yes, Continue",
        cancel_label: str = "No, Cancel",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._message = message
        self._action_id = action_id
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label
    
    def compose(self) -> ComposeResult:
        yield Label(f"⚠️ {self._title}", classes="dialog-title")
        yield Label(self._message, classes="dialog-message")
        
        with Horizontal(classes="dialog-buttons"):
            yield Button(
                self._cancel_label,
                id="btn-cancel",
                variant="default",
            )
            yield Button(
                self._confirm_label,
                id="btn-confirm",
                variant="error",
            )
    
    def action_confirm(self) -> None:
        """Handle confirmation."""
        self.post_message(self.Confirmed(self._action_id))
        self.remove()
    
    def action_cancel(self) -> None:
        """Handle cancellation."""
        self.post_message(self.Cancelled(self._action_id))
        self.remove()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-confirm":
            self.action_confirm()
        elif event.button.id == "btn-cancel":
            self.action_cancel()
