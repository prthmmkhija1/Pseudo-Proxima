# Adding Backends

This guide explains how to add new quantum simulation backends to Proxima.

## Overview

Proxima uses a plugin-based backend architecture. Each backend implements the `BaseBackendAdapter` interface and registers with the backend registry.

## Backend Architecture

```
proxima/
 backends/
    __init__.py
    base.py              # BaseBackendAdapter interface
    registry.py          # Backend registration
    lret/                # LRET backend
       __init__.py
       adapter.py
       normalizer.py
    cirq_backend/        # Cirq backend
       __init__.py
       adapter.py
       normalizer.py
    ...
```

## Quick Start

### 1. Create Backend Directory

```bash
mkdir proxima/backends/my_backend
touch proxima/backends/my_backend/__init__.py
touch proxima/backends/my_backend/adapter.py
touch proxima/backends/my_backend/normalizer.py
```

### 2. Implement Adapter

```python
# proxima/backends/my_backend/adapter.py

from proxima.backends.base import BaseBackendAdapter
from proxima.core.circuit import Circuit
from proxima.core.result import ExecutionResult

class MyBackendAdapter(BaseBackendAdapter):
    """Adapter for MyBackend quantum simulator."""
    
    name = "my_backend"
    version = "1.0.0"
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self._simulator = None
    
    def initialize(self) -> None:
        """Initialize the backend."""
        # Import and initialize your backend
        import my_backend_lib
        self._simulator = my_backend_lib.Simulator()
    
    def execute(
        self,
        circuit: Circuit,
        shots: int = 1024,
        **options
    ) -> ExecutionResult:
        """Execute a circuit and return results."""
        # Convert circuit to backend format
        native_circuit = self._convert_circuit(circuit)
        
        # Execute
        raw_result = self._simulator.run(native_circuit, shots=shots)
        
        # Convert to Proxima result
        return self._convert_result(raw_result)
    
    def validate_circuit(self, circuit: Circuit) -> bool:
        """Validate circuit is executable on this backend."""
        return True
    
    def estimate_resources(self, circuit: Circuit) -> dict:
        """Estimate resources needed for circuit."""
        return {
            "memory_bytes": 2 ** circuit.qubit_count * 16,
            "estimated_time_ms": circuit.gate_count * 0.1
        }
    
    def supports_simulator(self, simulator_type: str) -> bool:
        """Check if simulator type is supported."""
        return simulator_type in ["state_vector", "density_matrix"]
    
    def cleanup(self) -> None:
        """Clean up backend resources."""
        if self._simulator:
            self._simulator.close()
            self._simulator = None
```

### 3. Implement Normalizer

```python
# proxima/backends/my_backend/normalizer.py

from proxima.backends.base import BaseResultNormalizer
from proxima.core.result import ExecutionResult

class MyBackendNormalizer(BaseResultNormalizer):
    """Normalize results from MyBackend."""
    
    def normalize(self, raw_result: dict) -> ExecutionResult:
        """Convert backend-specific result to Proxima format."""
        counts = {}
        
        # Convert measurement results
        for state, count in raw_result.get("measurements", {}).items():
            # Ensure consistent bit ordering (big-endian)
            normalized_state = self._normalize_state(state)
            counts[normalized_state] = count
        
        return ExecutionResult(
            counts=counts,
            shots=sum(counts.values()),
            metadata=raw_result.get("metadata", {})
        )
    
    def _normalize_state(self, state: str) -> str:
        """Normalize state string representation."""
        # Remove any prefix/suffix
        state = state.strip("| <>")
        # Ensure binary format
        return format(int(state, 2) if state.isdigit() else 0, f"0{self.qubit_count}b")
```

### 4. Register Backend

```python
# proxima/backends/my_backend/__init__.py

from .adapter import MyBackendAdapter
from .normalizer import MyBackendNormalizer

__all__ = ["MyBackendAdapter", "MyBackendNormalizer"]
```

```python
# proxima/backends/registry.py (add to existing)

from proxima.backends.my_backend import MyBackendAdapter

BACKEND_REGISTRY = {
    "lret": LRETAdapter,
    "cirq": CirqAdapter,
    "qiskit": QiskitAdapter,
    "quest": QuESTAdapter,
    "qsim": QsimAdapter,
    "cuquantum": CuQuantumAdapter,
    "my_backend": MyBackendAdapter,  # Add your backend
}
```

## BaseBackendAdapter Interface

### Required Methods

```python
class BaseBackendAdapter(ABC):
    """Abstract base class for backend adapters."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Backend version."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend."""
        pass
    
    @abstractmethod
    def execute(
        self,
        circuit: Circuit,
        shots: int = 1024,
        **options
    ) -> ExecutionResult:
        """Execute a quantum circuit."""
        pass
    
    @abstractmethod
    def validate_circuit(self, circuit: Circuit) -> bool:
        """Check if circuit can be executed."""
        pass
    
    @abstractmethod
    def estimate_resources(self, circuit: Circuit) -> dict:
        """Estimate execution resources."""
        pass
    
    @abstractmethod
    def supports_simulator(self, simulator_type: str) -> bool:
        """Check simulator type support."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release backend resources."""
        pass
```

### Optional Methods

```python
class BaseBackendAdapter(ABC):
    # ... required methods ...
    
    def get_state_vector(self, circuit: Circuit) -> np.ndarray:
        """Return final state vector (optional)."""
        raise NotImplementedError("State vector not supported")
    
    def get_density_matrix(self, circuit: Circuit) -> np.ndarray:
        """Return density matrix (optional)."""
        raise NotImplementedError("Density matrix not supported")
    
    def apply_noise(self, circuit: Circuit, noise_model: dict) -> Circuit:
        """Apply noise model to circuit (optional)."""
        raise NotImplementedError("Noise not supported")
    
    def transpile(self, circuit: Circuit, **options) -> Circuit:
        """Optimize circuit for backend (optional)."""
        return circuit  # No-op by default
    
    def get_capabilities(self) -> dict:
        """Return backend capabilities."""
        return {
            "max_qubits": 30,
            "supported_gates": ["H", "X", "Y", "Z", "CNOT", "CZ"],
            "noise_support": False,
            "gpu_support": False
        }
```

## Circuit Conversion

### Gate Mapping

```python
GATE_MAP = {
    "H": my_backend.Hadamard,
    "X": my_backend.PauliX,
    "Y": my_backend.PauliY,
    "Z": my_backend.PauliZ,
    "CNOT": my_backend.CNOT,
    "CZ": my_backend.CZ,
    "RX": my_backend.RotationX,
    "RY": my_backend.RotationY,
    "RZ": my_backend.RotationZ,
}

def _convert_circuit(self, circuit: Circuit):
    """Convert Proxima circuit to backend format."""
    native_circuit = my_backend.Circuit(circuit.qubit_count)
    
    for gate in circuit.gates:
        native_gate_class = GATE_MAP.get(gate.name)
        if native_gate_class is None:
            raise ValueError(f"Unsupported gate: {gate.name}")
        
        if gate.is_controlled:
            native_gate = native_gate_class(
                control=gate.control,
                target=gate.target,
                **gate.parameters
            )
        else:
            native_gate = native_gate_class(
                targets=gate.targets,
                **gate.parameters
            )
        
        native_circuit.add_gate(native_gate)
    
    return native_circuit
```

### Result Conversion

```python
def _convert_result(self, raw_result) -> ExecutionResult:
    """Convert backend result to Proxima format."""
    normalizer = MyBackendNormalizer()
    
    return ExecutionResult(
        backend=self.name,
        simulator_type=self._config.get("simulator", "state_vector"),
        counts=normalizer.normalize(raw_result),
        execution_time_ms=raw_result.timing,
        qubit_count=raw_result.num_qubits,
        shot_count=raw_result.shots,
        metadata={
            "backend_version": self.version,
            "raw_result": raw_result.to_dict()
        }
    )
```

## Configuration

### Backend Configuration Schema

```python
# Define configuration schema
CONFIG_SCHEMA = {
    "simulator": {
        "type": "string",
        "enum": ["state_vector", "density_matrix"],
        "default": "state_vector"
    },
    "precision": {
        "type": "string",
        "enum": ["single", "double"],
        "default": "double"
    },
    "gpu": {
        "type": "boolean",
        "default": False
    },
    "threads": {
        "type": "integer",
        "minimum": 1,
        "default": 4
    }
}
```

### Using Configuration

```python
class MyBackendAdapter(BaseBackendAdapter):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self._config = config or {}
        
        # Apply configuration
        self.precision = self._config.get("precision", "double")
        self.use_gpu = self._config.get("gpu", False)
        self.threads = self._config.get("threads", 4)
```

## Testing

### Unit Tests

```python
# tests/backends/test_my_backend.py

import pytest
from proxima.backends.my_backend import MyBackendAdapter
from proxima.core.circuit import Circuit

class TestMyBackendAdapter:
    @pytest.fixture
    def adapter(self):
        adapter = MyBackendAdapter()
        adapter.initialize()
        yield adapter
        adapter.cleanup()
    
    def test_initialize(self, adapter):
        assert adapter._simulator is not None
    
    def test_execute_bell_state(self, adapter):
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        result = adapter.execute(circuit, shots=1000)
        
        assert result.shot_count == 1000
        assert "00" in result.counts or "11" in result.counts
    
    def test_supports_simulator(self, adapter):
        assert adapter.supports_simulator("state_vector") is True
        assert adapter.supports_simulator("tensor_network") is False
    
    def test_estimate_resources(self, adapter):
        circuit = Circuit(10)
        resources = adapter.estimate_resources(circuit)
        
        assert "memory_bytes" in resources
        assert resources["memory_bytes"] > 0
```

### Integration Tests

```python
# tests/integration/test_my_backend_integration.py

import pytest
from proxima.core.execution import execute_circuit
from proxima.core.circuit import Circuit

@pytest.mark.integration
class TestMyBackendIntegration:
    def test_full_execution_flow(self):
        circuit = Circuit(3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.cnot(1, 2)
        
        result = execute_circuit(
            circuit=circuit,
            backend="my_backend",
            shots=1000
        )
        
        assert result.success
        assert len(result.counts) > 0
```

## Checklist

Before submitting your backend:

### Implementation
- [ ] `BaseBackendAdapter` fully implemented
- [ ] All required methods work correctly
- [ ] Circuit conversion handles all standard gates
- [ ] Result normalization is consistent
- [ ] Error handling is comprehensive

### Testing
- [ ] Unit tests cover all methods
- [ ] Integration tests verify full flow
- [ ] Edge cases are tested
- [ ] Test coverage > 80%

### Documentation
- [ ] Docstrings for all public methods
- [ ] Backend capabilities documented
- [ ] Configuration options documented
- [ ] Usage examples provided

### Configuration
- [ ] Config schema defined
- [ ] Default values are sensible
- [ ] Invalid config is rejected

### Performance
- [ ] Memory usage is reasonable
- [ ] Execution time is competitive
- [ ] Resource estimation is accurate

## Examples

### LRET Backend (Simple)

See `proxima/backends/lret/adapter.py` for a simple state vector simulator.

### Cirq Backend (Intermediate)

See `proxima/backends/cirq_backend/adapter.py` for noise model support.

### cuQuantum Backend (Advanced)

See `proxima/backends/cuquantum/adapter.py` for GPU acceleration.

## Troubleshooting

### Import Errors

```python
# Make import optional
try:
    import my_backend_lib
    HAS_MY_BACKEND = True
except ImportError:
    HAS_MY_BACKEND = False

class MyBackendAdapter(BaseBackendAdapter):
    def initialize(self):
        if not HAS_MY_BACKEND:
            raise ImportError("my_backend_lib not installed")
```

### Dependency Conflicts

```python
# Use lazy imports
def _get_simulator(self):
    if self._simulator is None:
        import my_backend_lib
        self._simulator = my_backend_lib.Simulator()
    return self._simulator
```

### Memory Issues

```python
def cleanup(self):
    """Release all resources."""
    if self._simulator:
        self._simulator.clear_cache()
        self._simulator = None
    import gc
    gc.collect()
```
