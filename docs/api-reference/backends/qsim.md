# qsim Backend API Reference

> **Module:** `proxima.backends.qsim_adapter`  
> **Version:** 1.0  
> **Backend:** Google qsim (Schrödinger Full State-Vector Simulator)

---

## Overview

The qsim backend adapter provides access to Google's high-performance quantum circuit simulator through Proxima's unified backend interface. qsim is optimized for CPU execution with AVX2/AVX-512 vectorization and OpenMP parallelization.

---

## Classes

### QsimAdapter

Main adapter class for qsim integration.

```python
from proxima.backends.qsim_adapter import QsimAdapter

adapter = QsimAdapter()
```

#### Constructor

```python
QsimAdapter(
    config: QsimConfig | None = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `QsimConfig` | `None` | Optional configuration object |

**Raises:**
- `QsimInstallationError`: If qsimcirq is not available
- `QsimCPUError`: If CPU lacks required features

---

#### Methods

##### get_name

```python
def get_name(self) -> str
```

Returns the backend identifier.

**Returns:** `str` - Always returns `"qsim"`

---

##### get_version

```python
def get_version(self) -> str
```

Returns the qsim library version.

**Returns:** `str` - Version string (e.g., `"0.17.0"`)

---

##### is_available

```python
def is_available(self) -> bool
```

Check if qsim backend is available for use.

**Returns:** `bool` - `True` if qsimcirq is installed and CPU is compatible

---

##### get_capabilities

```python
def get_capabilities(self) -> Capabilities
```

Get backend capabilities and features.

**Returns:** `Capabilities` object with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `supports_density_matrix` | `bool` | Usually `False` |
| `supports_statevector` | `bool` | Always `True` |
| `supports_gpu` | `bool` | `False` (CPU-only) |
| `max_qubits` | `int` | Maximum supported qubits |
| `supported_gates` | `list[str]` | List of supported gates |
| `gate_fusion` | `bool` | Gate fusion support |
| `vectorization` | `str` | AVX mode (AVX2/AVX-512) |

---

##### validate_circuit

```python
def validate_circuit(
    self,
    circuit: Any
) -> ValidationResult
```

Validate a circuit for qsim execution.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `circuit` | `Any` | Circuit to validate |

**Returns:** `ValidationResult` with:
- `valid`: Whether circuit is valid
- `errors`: List of validation errors
- `warnings`: List of validation warnings
- `unsupported_gates`: Gates not supported by qsim
- `suggested_decomposition`: Suggested gate decompositions

---

##### estimate_resources

```python
def estimate_resources(
    self,
    circuit: Any
) -> ResourceEstimate
```

Estimate computational resources required.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `circuit` | `Any` | Circuit to estimate |

**Returns:** `ResourceEstimate` with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `memory_bytes` | `int` | Estimated memory usage |
| `estimated_time_seconds` | `float` | Estimated execution time |
| `recommended_threads` | `int` | Optimal thread count |
| `warnings` | `list[str]` | Resource warnings |

---

##### execute

```python
def execute(
    self,
    circuit: Any,
    options: dict[str, Any] | None = None
) -> ExecutionResult
```

Execute a quantum circuit on qsim.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuit` | `Any` | Required | Circuit to execute |
| `options` | `dict` | `None` | Execution options |

**Execution Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `shots` | `int` | `None` | Number of measurement shots |
| `num_threads` | `int` | `auto` | OpenMP thread count |
| `gate_fusion` | `bool` | `True` | Enable gate fusion |
| `fusion_strategy` | `str` | `"balanced"` | Fusion strategy |
| `verbosity` | `int` | `0` | Logging verbosity (0-3) |
| `timeout` | `int` | `None` | Execution timeout |
| `seed` | `int` | `None` | Random seed |

**Returns:** `ExecutionResult` with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | `bool` | Execution success status |
| `counts` | `dict[str, int]` | Measurement counts |
| `statevector` | `np.ndarray` | Final state vector |
| `probabilities` | `np.ndarray` | State probabilities |
| `metadata` | `dict` | Execution metadata |

**Raises:**
- `QsimCircuitError`: Invalid circuit
- `QsimMemoryError`: Insufficient memory
- `QsimRuntimeError`: Execution failure

---

##### execute_cirq

```python
def execute_cirq(
    self,
    cirq_circuit: cirq.Circuit,
    options: dict[str, Any] | None = None
) -> ExecutionResult
```

Execute a Cirq circuit directly.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cirq_circuit` | `cirq.Circuit` | Cirq circuit to execute |
| `options` | `dict` | Execution options |

**Returns:** `ExecutionResult`

---

##### get_qsim_simulator

```python
def get_qsim_simulator(
    self,
    options: dict[str, Any] | None = None
) -> qsimcirq.QSimSimulator
```

Get the underlying qsim simulator instance.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `options` | `dict` | Simulator options |

**Returns:** `qsimcirq.QSimSimulator` instance

---

##### optimize_for_qsim

```python
def optimize_for_qsim(
    self,
    circuit: Any
) -> Any
```

Optimize a circuit for qsim execution.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `circuit` | `Any` | Circuit to optimize |

**Returns:** Optimized circuit

**Optimizations Applied:**
- Gate decomposition to native gates
- Gate fusion preprocessing
- Measurement ordering

---

##### get_cpu_info

```python
def get_cpu_info(self) -> QsimCPUInfo
```

Get CPU information relevant to qsim.

**Returns:** `QsimCPUInfo` object

---

##### tune_threads

```python
def tune_threads(
    self,
    circuit: Any,
    max_threads: int | None = None
) -> ThreadTuningResult
```

Auto-tune thread count for optimal performance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuit` | `Any` | Required | Circuit for tuning |
| `max_threads` | `int` | `None` | Maximum threads to test |

**Returns:** `ThreadTuningResult` with optimal configuration

---

##### compare_with_cirq

```python
def compare_with_cirq(
    self,
    circuit: Any,
    tolerance: float = 1e-6
) -> ComparisonResult
```

Compare qsim results with Cirq simulator.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuit` | `Any` | Required | Circuit to compare |
| `tolerance` | `float` | `1e-6` | Comparison tolerance |

**Returns:** `ComparisonResult` with agreement metrics

---

### QsimConfig

Configuration class for qsim adapter.

```python
from proxima.backends.qsim_adapter import QsimConfig

config = QsimConfig(
    num_threads=8,
    gate_fusion=True,
    fusion_strategy="aggressive"
)
```

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_threads` | `int | None` | `None` | Thread count (auto if None) |
| `gate_fusion` | `bool` | `True` | Enable gate fusion |
| `fusion_strategy` | `str` | `"balanced"` | Fusion strategy |
| `verbosity` | `int` | `0` | Logging verbosity |
| `max_qubits` | `int` | `30` | Maximum qubit limit |
| `seed` | `int | None` | `None` | Random seed |
| `fallback_enabled` | `bool` | `True` | Enable fallback |

#### Methods

##### from_dict

```python
@classmethod
def from_dict(
    cls,
    config: dict[str, Any]
) -> QsimConfig
```

Create configuration from dictionary.

---

##### validate

```python
def validate(self) -> tuple[bool, list[str]]
```

Validate configuration settings.

**Returns:** Tuple of (valid, error_messages)

---

### QsimCPUInfo

CPU information relevant to qsim execution.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `physical_cores` | `int` | Physical CPU cores |
| `logical_cores` | `int` | Logical cores (with HT) |
| `has_avx2` | `bool` | AVX2 support |
| `has_avx512` | `bool` | AVX-512 support |
| `vectorization_mode` | `str` | Best vectorization mode |
| `cache_size_l3_mb` | `int` | L3 cache size in MB |
| `numa_nodes` | `int` | Number of NUMA nodes |

---

### ThreadTuningResult

Result of thread tuning operation.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `optimal_threads` | `int` | Recommended thread count |
| `speedup_vs_single` | `float` | Speedup over single thread |
| `efficiency` | `float` | Parallel efficiency (0-1) |
| `thread_times` | `dict[int, float]` | Time for each thread count |

---

## Enums

### FusionStrategy

```python
from proxima.backends.qsim_adapter import FusionStrategy

class FusionStrategy(str, Enum):
    NONE = "none"             # No fusion
    CONSERVATIVE = "conservative"  # Minimal fusion
    BALANCED = "balanced"     # Default balanced
    AGGRESSIVE = "aggressive" # Maximum fusion
```

---

### VectorizationMode

```python
from proxima.backends.qsim_adapter import VectorizationMode

class VectorizationMode(str, Enum):
    BASIC = "basic"       # No vectorization
    AVX2 = "avx2"         # AVX2 (256-bit)
    AVX512 = "avx512"     # AVX-512 (512-bit)
```

---

## Exceptions

### QsimInstallationError

Raised when qsim/qsimcirq is not properly installed.

```python
from proxima.backends.qsim_adapter import QsimInstallationError

try:
    adapter = QsimAdapter()
except QsimInstallationError as e:
    print(f"qsim not available: {e}")
    print(e.installation_help)
```

**Attributes:**
- `installation_help`: Installation instructions

---

### QsimCPUError

Raised when CPU lacks required features.

```python
from proxima.backends.qsim_adapter import QsimCPUError

try:
    adapter = QsimAdapter()
except QsimCPUError as e:
    print(f"CPU error: {e}")
    print(f"Required: AVX2, Found: {e.available_features}")
```

**Attributes:**
- `required_features`: Required CPU features
- `available_features`: Available features

---

### QsimCircuitError

Raised for circuit validation errors.

```python
from proxima.backends.qsim_adapter import QsimCircuitError

try:
    result = adapter.execute(invalid_circuit)
except QsimCircuitError as e:
    print(f"Circuit error: {e}")
    print(f"Unsupported gates: {e.unsupported_gates}")
    print(f"Suggested decomposition: {e.decomposition_hint}")
```

**Attributes:**
- `unsupported_gates`: List of unsupported gates
- `decomposition_hint`: Suggestion for decomposition

---

### QsimMemoryError

Raised when memory requirements exceed available resources.

```python
from proxima.backends.qsim_adapter import QsimMemoryError

try:
    result = adapter.execute(large_circuit)
except QsimMemoryError as e:
    print(f"Memory error: {e}")
    print(f"Required: {e.required_bytes / 1e9:.2f} GB")
    print(f"Available: {e.available_bytes / 1e9:.2f} GB")
```

**Attributes:**
- `required_bytes`: Memory required
- `available_bytes`: Memory available

---

### QsimRuntimeError

Raised for runtime execution errors.

```python
from proxima.backends.qsim_adapter import QsimRuntimeError

try:
    result = adapter.execute(circuit)
except QsimRuntimeError as e:
    print(f"Runtime error: {e}")
```

---

## Utility Functions

### check_qsim_availability

```python
def check_qsim_availability() -> tuple[bool, str]
```

Check if qsim is available.

**Returns:** Tuple of (available, message)

```python
from proxima.backends.qsim_adapter import check_qsim_availability

available, message = check_qsim_availability()
if available:
    print(f"qsim ready: {message}")
else:
    print(f"qsim unavailable: {message}")
```

---

### check_cpu_features

```python
def check_cpu_features() -> dict[str, bool]
```

Check CPU features required by qsim.

**Returns:** Dictionary of feature availability

```python
from proxima.backends.qsim_adapter import check_cpu_features

features = check_cpu_features()
print(f"AVX2: {features['avx2']}")
print(f"AVX-512: {features['avx512']}")
```

---

### get_optimal_threads

```python
def get_optimal_threads(
    num_qubits: int,
    max_threads: int | None = None
) -> int
```

Get optimal thread count for qubit count.

**Returns:** Recommended thread count

---

### convert_qiskit_to_cirq

```python
def convert_qiskit_to_cirq(
    qiskit_circuit: Any
) -> cirq.Circuit
```

Convert Qiskit circuit to Cirq format.

**Returns:** Equivalent Cirq circuit

---

## Examples

### Basic Execution

```python
from proxima.backends.qsim_adapter import QsimAdapter
from proxima.circuit import Circuit

# Create adapter
adapter = QsimAdapter()

# Check CPU features
cpu_info = adapter.get_cpu_info()
print(f"CPU cores: {cpu_info.physical_cores}")
print(f"AVX2: {cpu_info.has_avx2}")
print(f"AVX-512: {cpu_info.has_avx512}")

# Create circuit
circuit = Circuit(num_qubits=20)
circuit.h(range(20))

# Execute
result = adapter.execute(
    circuit,
    options={'shots': 1000}
)

print(f"Counts: {len(result.counts)} unique states")
```

### Gate Fusion Strategies

```python
# Aggressive fusion for maximum performance
result = adapter.execute(
    circuit,
    options={
        'gate_fusion': True,
        'fusion_strategy': 'aggressive'
    }
)

# Disable fusion for debugging
result = adapter.execute(
    circuit,
    options={
        'gate_fusion': False,
        'verbosity': 2
    }
)
```

### Thread Configuration

```python
# Auto-tune threads
tuning = adapter.tune_threads(circuit)
print(f"Optimal threads: {tuning.optimal_threads}")
print(f"Speedup: {tuning.speedup_vs_single:.2f}x")

# Execute with optimal threads
result = adapter.execute(
    circuit,
    options={'num_threads': tuning.optimal_threads}
)

# Or use specific thread count
result = adapter.execute(
    circuit,
    options={'num_threads': 8}
)
```

### Cirq Integration

```python
import cirq

# Create Cirq circuit
qubits = cirq.LineQubit.range(10)
cirq_circuit = cirq.Circuit([
    cirq.H.on_each(*qubits),
    cirq.CNOT(qubits[i], qubits[i+1]) for i in range(9)
])

# Execute directly
result = adapter.execute_cirq(
    cirq_circuit,
    options={'shots': 1000}
)

# Or use native qsim simulator
qsim_sim = adapter.get_qsim_simulator()
result = qsim_sim.run(cirq_circuit, repetitions=1000)
```

### Circuit Optimization

```python
# Optimize circuit for qsim
optimized_circuit = adapter.optimize_for_qsim(circuit)

# Execute optimized circuit
result = adapter.execute(optimized_circuit)

# Compare with Cirq
comparison = adapter.compare_with_cirq(circuit)
print(f"Agreement: {comparison.agreement:.6f}")
print(f"Max difference: {comparison.max_difference:.2e}")
```

### Error Handling with Fallback

```python
from proxima.backends.qsim_adapter import (
    QsimAdapter,
    QsimMemoryError,
    QsimCircuitError
)
from proxima.backends import get_default_backend

try:
    adapter = QsimAdapter()
    result = adapter.execute(circuit)
except QsimCircuitError as e:
    print(f"Unsupported gates: {e.unsupported_gates}")
    # Decompose and retry
    decomposed = adapter.optimize_for_qsim(circuit)
    result = adapter.execute(decomposed)
except QsimMemoryError as e:
    print(f"Memory exceeded, using fallback...")
    adapter = get_default_backend()
    result = adapter.execute(circuit)
```

### Batch Execution

```python
# Execute multiple circuits
results = []
for circuit in circuits:
    result = adapter.execute(
        circuit,
        options={'shots': 1000}
    )
    results.append(result)

# Or use batch mode for parameter sweeps
for params in parameter_sets:
    parameterized_circuit = circuit.bind_parameters(params)
    result = adapter.execute(parameterized_circuit)
    results.append((params, result))
```

---

## Supported Gates

### Native Gates (Optimal Performance)

| Gate | Description |
|------|-------------|
| `H` | Hadamard |
| `X`, `Y`, `Z` | Pauli gates |
| `RX`, `RY`, `RZ` | Rotation gates |
| `CX` / `CNOT` | Controlled-X |
| `CZ` | Controlled-Z |
| `SWAP` | SWAP gate |
| `ISWAP` | iSWAP gate |
| `T`, `S` | T and S gates |
| `Measure` | Measurement |

### Decomposed Gates

| Gate | Decomposition |
|------|---------------|
| `Toffoli` | 6 CNOT + T gates |
| `CCZ` | 6 CNOT + T gates |
| `CSWAP` | 2 CNOT + Toffoli |
| `U3` | RZ, RY, RZ sequence |

---

## See Also

- [qsim Installation Guide](../backends/qsim-installation.md)
- [qsim Usage Guide](../backends/qsim-usage.md)
- [Backend Selection](../backends/backend-selection.md)
