# QuEST Backend API Reference

> **Module:** `proxima.backends.quest_adapter`  
> **Version:** 1.0  
> **Backend:** QuEST (Quantum Exact Simulation Toolkit)

---

## Overview

The QuEST backend adapter provides access to the high-performance QuEST quantum circuit simulator through Proxima's unified backend interface. QuEST supports both state vector and density matrix simulations with optional GPU acceleration.

---

## Classes

### QuestBackendAdapter

Main adapter class for QuEST integration.

```python
from proxima.backends.quest_adapter import QuestBackendAdapter

adapter = QuestBackendAdapter()
```

#### Constructor

```python
QuestBackendAdapter(
    config: QuestConfig | None = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `QuestConfig` | `None` | Optional configuration object |

**Raises:**
- `QuestInstallationError`: If pyquest module is not available

---

#### Methods

##### get_name

```python
def get_name(self) -> str
```

Returns the backend identifier.

**Returns:** `str` - Always returns `"quest"`

---

##### get_version

```python
def get_version(self) -> str
```

Returns the QuEST library version.

**Returns:** `str` - Version string (e.g., `"0.9.0"`)

---

##### is_available

```python
def is_available(self) -> bool
```

Check if QuEST backend is available for use.

**Returns:** `bool` - `True` if pyquest is installed and working

---

##### get_capabilities

```python
def get_capabilities(self) -> Capabilities
```

Get backend capabilities and features.

**Returns:** `Capabilities` object with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `supports_density_matrix` | `bool` | Always `True` |
| `supports_statevector` | `bool` | Always `True` |
| `supports_gpu` | `bool` | GPU availability |
| `max_qubits` | `int` | Maximum supported qubits |
| `supported_gates` | `list[str]` | List of supported gate names |
| `precision_modes` | `list[str]` | Available precision modes |

---

##### validate_circuit

```python
def validate_circuit(
    self,
    circuit: Any
) -> ValidationResult
```

Validate a circuit for QuEST execution.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `circuit` | `Any` | Circuit to validate |

**Returns:** `ValidationResult` with:
- `valid`: Whether circuit is valid
- `errors`: List of validation errors
- `warnings`: List of validation warnings
- `unsupported_gates`: Gates not supported by QuEST

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
| `gpu_memory_bytes` | `int` | Estimated GPU memory (if applicable) |
| `estimated_time_seconds` | `float` | Estimated execution time |
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

Execute a quantum circuit on QuEST.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuit` | `Any` | Required | Circuit to execute |
| `options` | `dict` | `None` | Execution options |

**Execution Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `simulation_type` | `str` | `"state_vector"` | `"state_vector"` or `"density_matrix"` |
| `precision` | `str` | `"double"` | `"single"`, `"double"`, or `"quad"` |
| `shots` | `int` | `None` | Number of measurement shots |
| `use_gpu` | `bool` | `False` | Enable GPU acceleration |
| `num_threads` | `int` | `auto` | OpenMP thread count |
| `truncation_threshold` | `float` | `1e-10` | Rank truncation threshold |
| `timeout` | `int` | `None` | Execution timeout in seconds |

**Returns:** `ExecutionResult` with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | `bool` | Execution success status |
| `counts` | `dict[str, int]` | Measurement counts (if shots specified) |
| `statevector` | `np.ndarray` | Final state vector (if requested) |
| `density_matrix` | `np.ndarray` | Final density matrix (if DM mode) |
| `metadata` | `dict` | Execution metadata |

**Raises:**
- `QuestCircuitError`: Invalid circuit
- `QuestMemoryError`: Insufficient memory
- `QuestRuntimeError`: Execution failure

---

##### tune_threads

```python
def tune_threads(
    self,
    circuit: Any,
    max_threads: int | None = None
) -> ThreadTuningResult
```

Automatically tune OpenMP thread count for optimal performance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuit` | `Any` | Required | Circuit for tuning |
| `max_threads` | `int` | `None` | Maximum threads to test |

**Returns:** `ThreadTuningResult` with optimal thread configuration

---

##### supports_simulator

```python
def supports_simulator(
    self,
    sim_type: SimulatorType
) -> bool
```

Check if a simulation type is supported.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sim_type` | `SimulatorType` | Simulation type to check |

**Returns:** `bool` - Whether the simulation type is supported

---

##### get_hardware_info

```python
def get_hardware_info(self) -> QuestHardwareInfo | None
```

Get hardware information detected by QuEST.

**Returns:** `QuestHardwareInfo` or `None` if unavailable

---

### QuestConfig

Configuration class for QuEST adapter.

```python
from proxima.backends.quest_adapter import QuestConfig

config = QuestConfig(
    precision="double",
    use_gpu=True,
    truncation_threshold=1e-6
)
```

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `precision` | `str` | `"double"` | Precision mode |
| `use_gpu` | `bool` | `False` | Enable GPU by default |
| `num_threads` | `int | None` | `None` | Default thread count |
| `truncation_threshold` | `float` | `1e-10` | DM truncation threshold |
| `max_qubits` | `int` | `30` | Maximum qubit limit |
| `gpu_device_id` | `int` | `0` | GPU device to use |

#### Methods

##### from_dict

```python
@classmethod
def from_dict(
    cls,
    config: dict[str, Any]
) -> QuestConfig
```

Create configuration from dictionary.

---

### QuestHardwareInfo

Hardware information detected by QuEST.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `cpu_cores` | `int` | Number of CPU cores |
| `gpu_available` | `bool` | GPU availability |
| `gpu_name` | `str | None` | GPU device name |
| `gpu_memory_mb` | `int | None` | GPU memory in MB |
| `mpi_available` | `bool` | MPI availability |
| `mpi_ranks` | `int` | Number of MPI ranks |

---

## Enums

### QuestPrecision

```python
from proxima.backends.quest_adapter import QuestPrecision

class QuestPrecision(str, Enum):
    SINGLE = "single"     # float32, 8 bytes per complex
    DOUBLE = "double"     # float64, 16 bytes per complex
    QUAD = "quad"         # float128, 32 bytes per complex
```

---

## Exceptions

### QuestInstallationError

Raised when QuEST/pyquest is not properly installed.

```python
from proxima.backends.quest_adapter import QuestInstallationError

try:
    adapter = QuestBackendAdapter()
except QuestInstallationError as e:
    print(f"QuEST not available: {e}")
    print(e.installation_help)
```

**Attributes:**
- `installation_help`: Instructions for installing QuEST

---

### QuestGPUError

Raised for GPU-related errors.

```python
from proxima.backends.quest_adapter import QuestGPUError

try:
    result = adapter.execute(circuit, options={'use_gpu': True})
except QuestGPUError as e:
    print(f"GPU error: {e}")
    # Fall back to CPU
    result = adapter.execute(circuit, options={'use_gpu': False})
```

---

### QuestMemoryError

Raised when memory requirements exceed available resources.

```python
from proxima.backends.quest_adapter import QuestMemoryError

try:
    result = adapter.execute(large_circuit)
except QuestMemoryError as e:
    print(f"Memory error: {e}")
    print(f"Required: {e.required_bytes / 1e9:.2f} GB")
    print(f"Available: {e.available_bytes / 1e9:.2f} GB")
```

**Attributes:**
- `required_bytes`: Memory required
- `available_bytes`: Memory available

---

### QuestCircuitError

Raised for circuit validation errors.

```python
from proxima.backends.quest_adapter import QuestCircuitError

try:
    result = adapter.execute(invalid_circuit)
except QuestCircuitError as e:
    print(f"Circuit error: {e}")
    print(f"Invalid gates: {e.invalid_gates}")
```

---

### QuestRuntimeError

Raised for runtime execution errors.

```python
from proxima.backends.quest_adapter import QuestRuntimeError

try:
    result = adapter.execute(circuit)
except QuestRuntimeError as e:
    print(f"Runtime error: {e}")
```

---

## Utility Functions

### check_quest_availability

```python
def check_quest_availability() -> tuple[bool, str]
```

Check if QuEST is available.

**Returns:** Tuple of (available, message)

```python
from proxima.backends.quest_adapter import check_quest_availability

available, message = check_quest_availability()
if available:
    print(f"QuEST ready: {message}")
else:
    print(f"QuEST unavailable: {message}")
```

---

### get_quest_installation_help

```python
def get_quest_installation_help() -> str
```

Get installation instructions for QuEST.

**Returns:** Markdown-formatted installation guide

---

## Examples

### Basic Execution

```python
from proxima.backends.quest_adapter import QuestBackendAdapter
from proxima.circuit import Circuit

# Create adapter
adapter = QuestBackendAdapter()

# Create circuit
circuit = Circuit(num_qubits=10)
circuit.h(0)
for i in range(9):
    circuit.cnot(i, i + 1)

# Execute with state vector
result = adapter.execute(
    circuit,
    options={'shots': 1000}
)

print(f"Counts: {result.counts}")
```

### Density Matrix Simulation

```python
# Execute in density matrix mode
result = adapter.execute(
    circuit,
    options={
        'simulation_type': 'density_matrix',
        'truncation_threshold': 1e-6
    }
)

dm = result.density_matrix
print(f"Density matrix shape: {dm.shape}")
print(f"Trace: {np.trace(dm)}")
print(f"Purity: {np.trace(dm @ dm)}")
```

### GPU Execution

```python
# Check GPU availability
caps = adapter.get_capabilities()
if caps.supports_gpu:
    result = adapter.execute(
        circuit,
        options={
            'use_gpu': True,
            'precision': 'single'  # Faster on GPU
        }
    )
    print(f"GPU execution: {result.metadata['execution_mode']}")
```

### Thread Tuning

```python
# Auto-tune thread count
tuning = adapter.tune_threads(circuit)
print(f"Optimal threads: {tuning.optimal_threads}")
print(f"Expected speedup: {tuning.speedup_vs_single:.2f}x")

# Execute with optimal threads
result = adapter.execute(
    circuit,
    options={'num_threads': tuning.optimal_threads}
)
```

---

## See Also

- [QuEST Installation Guide](../backends/quest-installation.md)
- [QuEST Usage Guide](../backends/quest-usage.md)
- [Backend Selection](../backends/backend-selection.md)
