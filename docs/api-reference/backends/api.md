# Backend API Reference

Complete API documentation for the `proxima.backends` module.

## Overview

The backends module provides a unified interface for quantum circuit simulation across multiple frameworks:

- **Cirq**: Google's quantum computing framework
- **Qiskit Aer**: IBM's high-performance quantum simulator
- **LRET**: Lightweight Reference Execution Target
- **cuQuantum**: NVIDIA GPU-accelerated simulation
- **qsim**: Google's optimized state vector simulator
- **QuEST**: Quantum Exact Simulation Toolkit

---

## Base Types

### BackendAdapter

Abstract base class for all backend adapters.

```python
from proxima.backends.base import BackendAdapter

class BackendAdapter(ABC):
    """Abstract base class for quantum backend adapters."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this backend."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is installed and usable."""
        pass
    
    @abstractmethod
    def run(
        self,
        circuit: Any,
        shots: int = 1000,
        **options
    ) -> ExecutionResult:
        """
        Execute a quantum circuit.
        
        Args:
            circuit: Circuit in native or universal format
            shots: Number of measurement shots
            **options: Backend-specific options
            
        Returns:
            ExecutionResult with counts and metadata
            
        Raises:
            BackendError: If execution fails
        """
        pass
    
    @abstractmethod
    def get_statevector(
        self,
        circuit: Any,
        **options
    ) -> np.ndarray:
        """
        Get the statevector after circuit execution.
        
        Args:
            circuit: Circuit to simulate
            **options: Backend-specific options
            
        Returns:
            Complex numpy array of shape (2^n,)
        """
        pass
    
    def get_capabilities(self) -> BackendCapabilities:
        """Return backend capabilities."""
        pass
    
    def get_version(self) -> str:
        """Return backend library version."""
        pass
    
    def validate_circuit(self, circuit: Any) -> list[str]:
        """
        Validate a circuit for this backend.
        
        Returns:
            List of validation errors (empty if valid)
        """
        pass
```

### ExecutionResult

Result from circuit execution.

```python
from proxima.backends.base import ExecutionResult

@dataclass
class ExecutionResult:
    """Result from quantum circuit execution."""
    
    counts: dict[str, int]
    """Measurement counts by bitstring."""
    
    backend: str
    """Name of backend that executed the circuit."""
    
    shots: int
    """Number of shots executed."""
    
    execution_time: float
    """Execution time in seconds."""
    
    statevector: np.ndarray | None = None
    """Optional statevector (if requested)."""
    
    density_matrix: np.ndarray | None = None
    """Optional density matrix (if requested)."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional execution metadata."""
    
    def probabilities(self) -> dict[str, float]:
        """Convert counts to probabilities."""
        total = sum(self.counts.values())
        return {k: v / total for k, v in self.counts.items()}
    
    def most_likely_state(self) -> str:
        """Return the most frequently measured state."""
        return max(self.counts, key=self.counts.get)
    
    def entropy(self) -> float:
        """Calculate Shannon entropy of the distribution."""
        import math
        probs = self.probabilities()
        return -sum(p * math.log2(p) for p in probs.values() if p > 0)
```

### BackendCapabilities

Capabilities and features of a backend.

```python
from proxima.backends.base import BackendCapabilities

@dataclass
class BackendCapabilities:
    """Backend capability flags and limits."""
    
    max_qubits: int
    """Maximum number of qubits supported."""
    
    supports_statevector: bool = True
    """Supports statevector simulation."""
    
    supports_density_matrix: bool = False
    """Supports density matrix simulation."""
    
    supports_gpu: bool = False
    """Supports GPU acceleration."""
    
    supports_distributed: bool = False
    """Supports distributed simulation."""
    
    supports_noise: bool = False
    """Supports noise models."""
    
    native_gates: list[str] = field(default_factory=list)
    """List of natively supported gates."""
    
    noise_models: list[str] = field(default_factory=list)
    """Supported noise model types."""
    
    precision: str = "double"
    """Numerical precision (single, double)."""
```

---

## Backend Registry

### BackendRegistry

Central registry for backend management.

```python
from proxima.backends.registry import BackendRegistry, backend_registry

class BackendRegistry:
    """Registry for quantum backend adapters with hot-reload support."""
    
    def register(
        self, 
        adapter_class: type[BackendAdapter],
        priority: int = 0,
    ) -> None:
        """
        Register a backend adapter.
        
        Args:
            adapter_class: Adapter class (not instance)
            priority: Selection priority (higher = preferred)
        """
    
    def get(self, name: str) -> BackendAdapter | None:
        """
        Get a backend adapter by name.
        
        Args:
            name: Backend identifier
            
        Returns:
            Adapter instance or None if not found
        """
    
    def list_available(self) -> list[str]:
        """List names of available backends."""
    
    def list_all(self) -> list[str]:
        """List all registered backend names."""
    
    def get_default(self) -> BackendAdapter:
        """
        Get the default backend (highest priority available).
        
        Returns:
            Default backend adapter
            
        Raises:
            NoBackendError: If no backends available
        """
    
    def get_status(self, name: str) -> BackendStatus:
        """
        Get detailed status of a backend.
        
        Args:
            name: Backend identifier
            
        Returns:
            BackendStatus with availability and details
        """
    
    def refresh(self) -> None:
        """Refresh backend availability status (hot-reload)."""


@dataclass
class BackendStatus:
    """Detailed status of a backend."""
    
    name: str
    available: bool
    version: str | None
    error: str | None
    capabilities: BackendCapabilities | None
    last_checked: datetime


# Global registry instance
backend_registry = BackendRegistry()
```

---

## Backend Adapters

### CirqAdapter

Google Cirq backend adapter.

```python
from proxima.backends.cirq_adapter import CirqAdapter

class CirqAdapter(BackendAdapter):
    """Google Cirq quantum simulator adapter."""
    
    name = "cirq"
    
    def __init__(
        self,
        simulator_type: str = "simulator",
        seed: int | None = None,
        dtype: type = np.complex128,
    ):
        """
        Initialize Cirq adapter.
        
        Args:
            simulator_type: 'simulator', 'density_matrix', or 'clifford'
            seed: Random seed for reproducibility
            dtype: Numerical dtype (np.complex64 or np.complex128)
        """
    
    @property
    def is_available(self) -> bool:
        """Check if Cirq is installed."""
    
    def run(
        self,
        circuit: cirq.Circuit | str,
        shots: int = 1000,
        **options
    ) -> ExecutionResult:
        """
        Execute circuit on Cirq simulator.
        
        Args:
            circuit: Cirq Circuit or OpenQASM string
            shots: Number of measurement shots
            **options:
                - seed: Random seed
                - noise_model: Noise model to apply
                - initial_state: Initial state vector
                
        Returns:
            ExecutionResult with measurement counts
        """
    
    def get_statevector(
        self,
        circuit: cirq.Circuit | str,
        **options
    ) -> np.ndarray:
        """Get final statevector."""
    
    def get_density_matrix(
        self,
        circuit: cirq.Circuit | str,
        **options
    ) -> np.ndarray:
        """Get final density matrix."""
    
    def from_qasm(self, qasm: str) -> cirq.Circuit:
        """Convert OpenQASM to Cirq circuit."""
    
    def to_qasm(self, circuit: cirq.Circuit) -> str:
        """Convert Cirq circuit to OpenQASM."""
```

### QiskitAerAdapter

IBM Qiskit Aer backend adapter.

```python
from proxima.backends.qiskit_adapter import QiskitAerAdapter

class QiskitAerAdapter(BackendAdapter):
    """IBM Qiskit Aer quantum simulator adapter."""
    
    name = "qiskit_aer"
    
    def __init__(
        self,
        backend_type: str = "aer_simulator",
        method: str = "automatic",
        seed: int | None = None,
        precision: str = "double",
    ):
        """
        Initialize Qiskit Aer adapter.
        
        Args:
            backend_type: 'aer_simulator', 'statevector_simulator', etc.
            method: Simulation method ('automatic', 'statevector', 
                    'density_matrix', 'matrix_product_state', etc.)
            seed: Random seed
            precision: 'single' or 'double'
        """
    
    def run(
        self,
        circuit: QuantumCircuit | str,
        shots: int = 1000,
        **options
    ) -> ExecutionResult:
        """
        Execute circuit on Qiskit Aer.
        
        Args:
            circuit: Qiskit QuantumCircuit or OpenQASM
            shots: Number of measurement shots
            **options:
                - noise_model: Qiskit NoiseModel
                - coupling_map: Device coupling map
                - basis_gates: Basis gate set
                - optimization_level: Transpiler optimization (0-3)
                
        Returns:
            ExecutionResult with measurement counts
        """
    
    def get_statevector(
        self,
        circuit: QuantumCircuit | str,
        **options
    ) -> np.ndarray:
        """Get final statevector."""
    
    def apply_noise_model(
        self,
        model_type: str,
        **params
    ) -> None:
        """
        Apply a noise model for subsequent executions.
        
        Args:
            model_type: 'depolarizing', 'thermal', 'readout', etc.
            **params: Model-specific parameters
        """
    
    def create_noise_model(
        self,
        error_rates: dict[str, float],
    ) -> "NoiseModel":
        """Create a custom noise model."""
```

### LRETAdapter

Lightweight Reference Execution Target adapter.

```python
from proxima.backends.lret import LRETAdapter

class LRETAdapter(BackendAdapter):
    """Lightweight Reference Execution Target adapter."""
    
    name = "lret"
    
    def __init__(
        self,
        precision: str = "double",
        seed: int | None = None,
        max_qubits: int = 20,
    ):
        """
        Initialize LRET adapter.
        
        Args:
            precision: 'single' or 'double'
            seed: Random seed
            max_qubits: Maximum qubit limit
        """
    
    def run(
        self,
        circuit: Any,
        shots: int = 1000,
        **options
    ) -> ExecutionResult:
        """
        Execute circuit on LRET simulator.
        
        Args:
            circuit: Circuit in universal format or QASM
            shots: Number of measurement shots
            **options:
                - validate: Validate circuit before execution
                - optimize: Apply circuit optimizations
                
        Returns:
            ExecutionResult with measurement counts
        """
    
    def get_statevector(
        self,
        circuit: Any,
        **options
    ) -> np.ndarray:
        """Get final statevector."""
```

### cuQuantumAdapter

NVIDIA cuQuantum GPU-accelerated adapter.

```python
from proxima.backends.cuquantum_adapter import cuQuantumAdapter

class cuQuantumAdapter(BackendAdapter):
    """NVIDIA cuQuantum GPU-accelerated simulator adapter."""
    
    name = "cuquantum"
    
    def __init__(
        self,
        device_id: int = 0,
        precision: str = "double",
        use_tensor_network: bool = False,
    ):
        """
        Initialize cuQuantum adapter.
        
        Args:
            device_id: CUDA device ID
            precision: 'single' or 'double'
            use_tensor_network: Use tensor network contraction
        """
    
    @property
    def is_available(self) -> bool:
        """Check if cuQuantum and CUDA are available."""
    
    def run(
        self,
        circuit: Any,
        shots: int = 1000,
        **options
    ) -> ExecutionResult:
        """
        Execute circuit with GPU acceleration.
        
        Args:
            circuit: Circuit to execute
            shots: Number of shots
            **options:
                - max_bond_dimension: For tensor network
                - cutoff: Singular value cutoff
                
        Returns:
            ExecutionResult with GPU-accelerated execution
        """
    
    def get_gpu_info(self) -> dict[str, Any]:
        """Get GPU device information."""
```

### qsimAdapter

Google qsim optimized simulator adapter.

```python
from proxima.backends.qsim_adapter import qsimAdapter

class qsimAdapter(BackendAdapter):
    """Google qsim optimized state vector simulator adapter."""
    
    name = "qsim"
    
    def __init__(
        self,
        num_threads: int = 0,  # 0 = auto
        use_gpu: bool = False,
        max_fused_gate_size: int = 2,
    ):
        """
        Initialize qsim adapter.
        
        Args:
            num_threads: Number of CPU threads (0 = auto)
            use_gpu: Use GPU acceleration (requires qsim-gpu)
            max_fused_gate_size: Maximum fused gate size
        """
    
    def run(
        self,
        circuit: cirq.Circuit,
        shots: int = 1000,
        **options
    ) -> ExecutionResult:
        """
        Execute circuit on qsim.
        
        Args:
            circuit: Cirq circuit
            shots: Number of shots
            **options:
                - verbosity: Logging verbosity (0-3)
                
        Returns:
            ExecutionResult with optimized execution
        """
```

### QuESTAdapter

Quantum Exact Simulation Toolkit adapter.

```python
from proxima.backends.quest_adapter import QuESTAdapter

class QuESTAdapter(BackendAdapter):
    """QuEST (Quantum Exact Simulation Toolkit) adapter."""
    
    name = "quest"
    
    def __init__(
        self,
        precision: str = "double",
        use_distributed: bool = False,
        num_ranks: int = 1,
    ):
        """
        Initialize QuEST adapter.
        
        Args:
            precision: 'single' or 'double'
            use_distributed: Enable MPI distribution
            num_ranks: Number of MPI ranks
        """
    
    @property
    def is_available(self) -> bool:
        """Check if QuEST is available."""
    
    def run(
        self,
        circuit: Any,
        shots: int = 1000,
        **options
    ) -> ExecutionResult:
        """
        Execute circuit on QuEST.
        
        Args:
            circuit: Circuit to execute
            shots: Number of shots
            **options:
                - density_matrix_mode: Use density matrix
                
        Returns:
            ExecutionResult from QuEST execution
        """
    
    def get_statevector(
        self,
        circuit: Any,
        **options
    ) -> np.ndarray:
        """Get final statevector."""
```

---

## Circuit Conversion

### CircuitConverter

Utility for converting between circuit formats.

```python
from proxima.backends.converter import CircuitConverter

class CircuitConverter:
    """Convert circuits between different formats."""
    
    @staticmethod
    def to_cirq(circuit: Any) -> cirq.Circuit:
        """Convert any circuit to Cirq format."""
    
    @staticmethod
    def to_qiskit(circuit: Any) -> QuantumCircuit:
        """Convert any circuit to Qiskit format."""
    
    @staticmethod
    def to_qasm(circuit: Any) -> str:
        """Convert any circuit to OpenQASM 2.0."""
    
    @staticmethod
    def to_qasm3(circuit: Any) -> str:
        """Convert any circuit to OpenQASM 3.0."""
    
    @staticmethod
    def from_qasm(qasm: str) -> Any:
        """Parse OpenQASM to native format."""
    
    @staticmethod
    def detect_format(circuit: Any) -> str:
        """Detect circuit format ('cirq', 'qiskit', 'qasm', etc.)."""
```

---

## Error Handling

### Exception Classes

```python
from proxima.backends.base import (
    BackendError,
    BackendNotAvailableError,
    CircuitValidationError,
    ExecutionError,
    NoBackendError,
)

class BackendError(Exception):
    """Base exception for backend errors."""
    pass

class BackendNotAvailableError(BackendError):
    """Backend is not installed or unavailable."""
    backend_name: str

class CircuitValidationError(BackendError):
    """Circuit failed validation for this backend."""
    validation_errors: list[str]

class ExecutionError(BackendError):
    """Error during circuit execution."""
    backend_name: str
    original_error: Exception | None

class NoBackendError(BackendError):
    """No backends are available."""
    pass
```

---

## Usage Examples

### Basic Execution

```python
from proxima.backends.registry import backend_registry

# Get default backend
backend = backend_registry.get_default()

# Define circuit (QASM)
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

# Execute
result = backend.run(qasm, shots=1000)
print(f"Counts: {result.counts}")
print(f"Execution time: {result.execution_time:.3f}s")
```

### Multi-Backend Comparison

```python
from proxima.backends.registry import backend_registry

circuit = "bell.qasm"  # Bell state circuit
shots = 1000

results = {}
for name in backend_registry.list_available():
    backend = backend_registry.get(name)
    results[name] = backend.run(circuit, shots=shots)

# Compare
for name, result in results.items():
    print(f"{name}: {result.counts}")
```

### GPU Acceleration

```python
from proxima.backends.cuquantum_adapter import cuQuantumAdapter

# Check availability
if cuQuantumAdapter().is_available:
    adapter = cuQuantumAdapter(device_id=0)
    
    # Large circuit
    result = adapter.run(large_circuit, shots=10000)
    print(f"GPU execution: {result.execution_time:.3f}s")
else:
    print("cuQuantum not available, falling back to CPU")
```

### Noise Simulation

```python
from proxima.backends.qiskit_adapter import QiskitAerAdapter

adapter = QiskitAerAdapter()

# Apply depolarizing noise
adapter.apply_noise_model(
    "depolarizing",
    single_qubit_error=0.001,
    two_qubit_error=0.01,
)

# Execute with noise
result = adapter.run(circuit, shots=10000)
print(f"Noisy counts: {result.counts}")
```
