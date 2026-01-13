# Backend Development Guide

Complete guide for developing and integrating new quantum simulation backends into Proxima.

## Table of Contents

1. [Overview](#overview)
2. [Backend Architecture](#backend-architecture)
3. [Implementing a New Backend](#implementing-a-new-backend)
4. [Backend Adapters](#backend-adapters)
5. [Testing Backends](#testing-backends)
6. [Backend Registration](#backend-registration)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Proxima supports multiple quantum simulation backends through a unified adapter interface. This guide covers how to implement adapters for new backends such as:

| Backend | Type | Description |
|---------|------|-------------|
| **Cirq** | State Vector, Density Matrix | Google's quantum computing framework |
| **Qiskit Aer** | State Vector, Density Matrix, Noise | IBM's quantum simulator |
| **LRET** | State Vector | Lightweight Research & Education Toolkit |
| **QuEST** | State Vector, Density Matrix, GPU | High-performance simulator |
| **qsim** | State Vector, GPU | Google's optimized simulator |
| **cuQuantum** | Tensor Network, GPU | NVIDIA's GPU-accelerated library |

---

## Backend Architecture

### Adapter Pattern

All backends implement the common adapter interface:

```
┌─────────────────────────────────────────────────┐
│              Backend Adapter Interface           │
├─────────────────────────────────────────────────┤
│  + name: str                                     │
│  + backend_type: str                             │
│  + capabilities: BackendCapability               │
│  + connect() -> bool                             │
│  + disconnect() -> None                          │
│  + is_connected() -> bool                        │
│  + execute(circuit, options) -> BackendResult    │
│  + validate_circuit(circuit) -> ValidationResult │
└─────────────────────────────────────────────────┘
           △                △                △
           │                │                │
    ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐
    │ CirqAdapter │  │QiskitAdapter│  │ LRETAdapter │
    └─────────────┘  └─────────────┘  └─────────────┘
```

### Capability System

Backends declare their capabilities for intelligent selection:

```python
class BackendCapability(Flag):
    STATE_VECTOR = auto()      # Can return state vectors
    DENSITY_MATRIX = auto()    # Can return density matrices
    MEASUREMENT = auto()       # Can perform measurements
    NOISE_MODEL = auto()       # Supports noise simulation
    GPU_ACCELERATION = auto()  # Has GPU support
    CUSTOM_GATES = auto()      # Supports custom gates
    PARAMETERIZED = auto()     # Supports parameterized gates
```

---

## Implementing a New Backend

### Step 1: Create the Adapter Class

Create a new file in `src/proxima/backends/`:

```python
# src/proxima/backends/my_backend_adapter.py
"""Adapter for MyBackend quantum simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from proxima.backends.base import (
    BackendAdapter,
    BackendCapability,
    BackendResult,
    BackendStatus,
    ValidationResult,
)
from proxima.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MyBackendConfig:
    """Configuration for MyBackend adapter."""
    
    simulator_type: str = "state_vector"
    max_qubits: int = 30
    precision: str = "double"
    seed: int | None = None


class MyBackendAdapter(BackendAdapter):
    """Adapter for MyBackend quantum simulator.
    
    This adapter wraps the MyBackend library to provide
    quantum simulation capabilities through Proxima's
    unified interface.
    
    Example:
        >>> adapter = MyBackendAdapter(config=MyBackendConfig())
        >>> await adapter.connect()
        >>> result = await adapter.execute(circuit, options)
        >>> await adapter.disconnect()
    """
    
    def __init__(self, config: MyBackendConfig | None = None) -> None:
        """Initialize the adapter.
        
        Args:
            config: Backend configuration. Uses defaults if not provided.
        """
        self._config = config or MyBackendConfig()
        self._simulator = None
        self._connected = False
        
    @property
    def name(self) -> str:
        """Backend name."""
        return "my_backend"
    
    @property
    def backend_type(self) -> str:
        """Backend type."""
        return "simulator"
    
    @property
    def capabilities(self) -> BackendCapability:
        """Backend capabilities."""
        caps = BackendCapability.STATE_VECTOR | BackendCapability.MEASUREMENT
        if self._config.simulator_type == "density_matrix":
            caps |= BackendCapability.DENSITY_MATRIX
        return caps
    
    async def connect(self) -> bool:
        """Connect to the backend.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Import the backend library
            import my_backend
            
            # Initialize simulator
            self._simulator = my_backend.Simulator(
                precision=self._config.precision,
                seed=self._config.seed,
            )
            self._connected = True
            
            logger.info(
                "backend.connected",
                backend=self.name,
                config=self._config.__dict__,
            )
            return True
            
        except ImportError:
            logger.error(
                "backend.import_error",
                backend=self.name,
                message="my_backend library not installed",
            )
            return False
        except Exception as e:
            logger.exception(
                "backend.connection_error",
                backend=self.name,
                error=str(e),
            )
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the backend."""
        if self._simulator is not None:
            # Cleanup resources
            self._simulator = None
        self._connected = False
        
        logger.info("backend.disconnected", backend=self.name)
    
    def is_connected(self) -> bool:
        """Check if connected.
        
        Returns:
            True if connected, False otherwise.
        """
        return self._connected and self._simulator is not None
    
    async def execute(
        self,
        circuit: Any,
        options: dict[str, Any] | None = None,
    ) -> BackendResult:
        """Execute a circuit on the backend.
        
        Args:
            circuit: The quantum circuit to execute.
            options: Execution options (shots, seed, etc.).
            
        Returns:
            BackendResult with execution results.
            
        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected():
            raise RuntimeError("Backend not connected")
        
        options = options or {}
        start_time = time.time()
        
        try:
            # Convert circuit if needed
            native_circuit = self._convert_circuit(circuit)
            
            # Execute
            if options.get("shots"):
                result = self._simulator.run(
                    native_circuit,
                    shots=options["shots"],
                )
                result_data = {"counts": result.counts}
            else:
                result = self._simulator.simulate(native_circuit)
                result_data = {"state_vector": result.state_vector}
            
            execution_time = (time.time() - start_time) * 1000
            
            return BackendResult(
                backend_name=self.name,
                success=True,
                execution_time_ms=execution_time,
                memory_peak_mb=self._get_memory_usage(),
                result_data=result_data,
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            logger.exception(
                "backend.execution_error",
                backend=self.name,
                error=str(e),
            )
            
            return BackendResult(
                backend_name=self.name,
                success=False,
                execution_time_ms=execution_time,
                memory_peak_mb=0.0,
                error=str(e),
            )
    
    async def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate a circuit for this backend.
        
        Args:
            circuit: The circuit to validate.
            
        Returns:
            ValidationResult with any issues found.
        """
        issues = []
        
        # Check qubit count
        num_qubits = self._get_qubit_count(circuit)
        if num_qubits > self._config.max_qubits:
            issues.append(
                f"Circuit has {num_qubits} qubits, "
                f"max supported is {self._config.max_qubits}"
            )
        
        # Check gate support
        unsupported = self._check_gate_support(circuit)
        if unsupported:
            issues.append(f"Unsupported gates: {unsupported}")
        
        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
        )
    
    def _convert_circuit(self, circuit: Any) -> Any:
        """Convert circuit to native format."""
        # Implement circuit conversion
        return circuit
    
    def _get_qubit_count(self, circuit: Any) -> int:
        """Get number of qubits in circuit."""
        return circuit.num_qubits
    
    def _check_gate_support(self, circuit: Any) -> list[str]:
        """Check for unsupported gates."""
        return []
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
```

### Step 2: Implement Circuit Conversion

Each backend has its own circuit representation. Implement converters:

```python
# src/proxima/backends/converters/my_backend_converter.py
"""Circuit converter for MyBackend."""

from typing import Any


class MyBackendCircuitConverter:
    """Converts circuits to MyBackend format."""
    
    GATE_MAP = {
        "H": "hadamard",
        "X": "pauli_x",
        "Y": "pauli_y",
        "Z": "pauli_z",
        "CNOT": "cx",
        "CZ": "cz",
        "RX": "rx",
        "RY": "ry",
        "RZ": "rz",
    }
    
    def convert(self, circuit: Any) -> Any:
        """Convert a circuit to MyBackend format.
        
        Args:
            circuit: Input circuit (Cirq, Qiskit, or generic).
            
        Returns:
            MyBackend circuit object.
        """
        import my_backend
        
        num_qubits = circuit.num_qubits
        native = my_backend.Circuit(num_qubits)
        
        for gate in circuit.gates:
            native_gate = self.GATE_MAP.get(gate.name)
            if native_gate is None:
                raise ValueError(f"Unsupported gate: {gate.name}")
            
            if gate.params:
                native.add_gate(native_gate, gate.qubits, *gate.params)
            else:
                native.add_gate(native_gate, gate.qubits)
        
        return native
```

### Step 3: Add Configuration

Add backend configuration to the config schema:

```python
# In src/proxima/config/backends.py

@dataclass
class MyBackendSettings:
    """Settings for MyBackend."""
    
    enabled: bool = True
    simulator_type: str = "state_vector"
    max_qubits: int = 30
    precision: str = "double"
    timeout_seconds: float = 300.0
```

---

## Backend Adapters

### Cirq Adapter

```python
from proxima.backends.cirq_adapter import CirqAdapter, CirqConfig

adapter = CirqAdapter(CirqConfig(
    simulator_type="state_vector",  # or "density_matrix"
    noise_model=None,
    seed=42,
))

await adapter.connect()
result = await adapter.execute(circuit, {"shots": 1000})
```

### Qiskit Aer Adapter

```python
from proxima.backends.qiskit_adapter import QiskitAerAdapter, QiskitConfig

adapter = QiskitAerAdapter(QiskitConfig(
    method="statevector",  # or "density_matrix", "automatic"
    shots=1024,
    noise_model=None,
    coupling_map=None,
))

await adapter.connect()
result = await adapter.execute(circuit)
```

### LRET Adapter

```python
from proxima.backends.lret_adapter import LRETAdapter, LRETConfig

adapter = LRETAdapter(LRETConfig(
    precision="double",
    max_qubits=24,
))

await adapter.connect()
result = await adapter.execute(circuit)
```

### QuEST Adapter

```python
from proxima.backends.quest_adapter import QuESTAdapter, QuESTConfig

adapter = QuESTAdapter(QuESTConfig(
    use_gpu=True,
    num_threads=8,
    precision="double",
))

await adapter.connect()
result = await adapter.execute(circuit)
```

### qsim Adapter

```python
from proxima.backends.qsim_adapter import QsimAdapter, QsimConfig

adapter = QsimAdapter(QsimConfig(
    use_gpu=True,
    gpu_mode=0,  # CUDA device ID
    num_threads=16,
))

await adapter.connect()
result = await adapter.execute(circuit)
```

### cuQuantum Adapter

```python
from proxima.backends.cuquantum_adapter import CuQuantumAdapter, CuQuantumConfig

adapter = CuQuantumAdapter(CuQuantumConfig(
    network_type="mps",  # Matrix Product State
    precision="complex128",
    gpu_id=0,
))

await adapter.connect()
result = await adapter.execute(circuit)
```

---

## Testing Backends

### Unit Tests

Create unit tests in `tests/unit/`:

```python
# tests/unit/test_my_backend.py
"""Unit tests for MyBackend adapter."""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestMyBackendAdapter:
    """Tests for MyBackendAdapter."""
    
    @pytest.fixture
    def mock_backend_lib(self):
        """Mock the backend library."""
        mock = MagicMock()
        mock.Simulator = MagicMock()
        return mock
    
    @pytest.mark.backend
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_backend_lib):
        """Test successful connection."""
        with patch.dict("sys.modules", {"my_backend": mock_backend_lib}):
            from proxima.backends.my_backend_adapter import MyBackendAdapter
            
            adapter = MyBackendAdapter()
            result = await adapter.connect()
            
            assert result is True
            assert adapter.is_connected()
    
    @pytest.mark.backend
    @pytest.mark.asyncio
    async def test_execute_state_vector(self, mock_backend_lib):
        """Test state vector execution."""
        mock_result = MagicMock()
        mock_result.state_vector = [1, 0, 0, 0]
        mock_backend_lib.Simulator.return_value.simulate.return_value = mock_result
        
        with patch.dict("sys.modules", {"my_backend": mock_backend_lib}):
            from proxima.backends.my_backend_adapter import MyBackendAdapter
            
            adapter = MyBackendAdapter()
            await adapter.connect()
            
            result = await adapter.execute(MagicMock(num_qubits=2))
            
            assert result.success is True
            assert "state_vector" in result.result_data
```

### Integration Tests

Create integration tests in `tests/integration/`:

```python
# tests/integration/test_my_backend_integration.py
"""Integration tests for MyBackend."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backend_with_pipeline():
    """Test backend integration with pipeline."""
    from proxima.data.pipeline import Pipeline, PipelineConfig, Stage
    from proxima.backends.my_backend_adapter import MyBackendAdapter
    
    adapter = MyBackendAdapter()
    
    async def execute_stage(ctx, circuit):
        await adapter.connect()
        result = await adapter.execute(circuit)
        await adapter.disconnect()
        return result
    
    pipeline = Pipeline(PipelineConfig(name="backend_test"))
    pipeline.add_stage(Stage(
        stage_id="execute",
        name="Execute",
        handler=execute_stage,
    ))
    
    result = await pipeline.execute(initial_input=mock_circuit)
    assert result.is_success
```

---

## Backend Registration

Register your backend in the backend registry:

```python
# src/proxima/backends/__init__.py

from .my_backend_adapter import MyBackendAdapter, MyBackendConfig

BACKEND_REGISTRY = {
    "cirq": CirqAdapter,
    "qiskit_aer": QiskitAerAdapter,
    "lret": LRETAdapter,
    "quest": QuESTAdapter,
    "qsim": QsimAdapter,
    "cuquantum": CuQuantumAdapter,
    "my_backend": MyBackendAdapter,  # Add your backend
}

__all__ = [
    # ... existing exports
    "MyBackendAdapter",
    "MyBackendConfig",
]
```

---

## Performance Optimization

### Memory Management

```python
async def execute(self, circuit, options):
    try:
        # Pre-allocate if possible
        state_size = 2 ** circuit.num_qubits
        
        # Use appropriate precision
        if self._config.precision == "single":
            dtype = np.complex64
        else:
            dtype = np.complex128
        
        result = self._simulator.simulate(circuit, dtype=dtype)
        return self._build_result(result)
        
    finally:
        # Cleanup large arrays
        if hasattr(self, '_temp_buffer'):
            del self._temp_buffer
        gc.collect()
```

### GPU Optimization

```python
async def execute_gpu(self, circuit, options):
    # Transfer to GPU
    gpu_circuit = self._transfer_to_gpu(circuit)
    
    try:
        # Execute on GPU
        result = self._gpu_simulator.run(gpu_circuit)
        
        # Transfer back only what's needed
        if options.get("return_state"):
            return self._transfer_from_gpu(result.state)
        else:
            return result.counts
            
    finally:
        # Free GPU memory
        self._gpu_simulator.reset()
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ImportError` | Backend library not installed | Install with `pip install my-backend` |
| `OutOfMemoryError` | Circuit too large | Reduce qubit count or use GPU |
| Slow execution | Not using GPU | Enable GPU with `use_gpu=True` |
| Incorrect results | Precision issues | Use `precision="double"` |

### Debug Logging

Enable debug logging for backend troubleshooting:

```python
import logging

logging.getLogger("proxima.backends").setLevel(logging.DEBUG)
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

await adapter.execute(circuit)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```
