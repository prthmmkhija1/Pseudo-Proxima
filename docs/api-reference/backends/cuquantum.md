# cuQuantum Backend API Reference

> **Module:** `proxima.backends.cuquantum_adapter`  
> **Version:** 1.0  
> **Backend:** NVIDIA cuQuantum SDK

---

## Overview

The cuQuantum backend adapter provides GPU-accelerated quantum circuit simulation using NVIDIA's cuQuantum SDK. It leverages cuStateVec for state vector simulations and supports multi-GPU configurations.

---

## Classes

### CuQuantumAdapter

Main adapter class for cuQuantum integration.

```python
from proxima.backends.cuquantum_adapter import CuQuantumAdapter

adapter = CuQuantumAdapter()
```

#### Constructor

```python
CuQuantumAdapter(
    config: CuQuantumConfig | None = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `CuQuantumConfig` | `None` | Optional configuration object |

**Raises:**
- `CuQuantumInstallationError`: If cuQuantum is not available
- `CuQuantumGPUError`: If no compatible GPU is found

---

#### Methods

##### get_name

```python
def get_name(self) -> str
```

Returns the backend identifier.

**Returns:** `str` - Always returns `"cuquantum"`

---

##### get_version

```python
def get_version(self) -> str
```

Returns the cuQuantum SDK version.

**Returns:** `str` - Version string (e.g., `"23.10.0"`)

---

##### is_available

```python
def is_available(self) -> bool
```

Check if cuQuantum backend is available for use.

**Returns:** `bool` - `True` if cuQuantum is installed and GPU is available

---

##### get_capabilities

```python
def get_capabilities(self) -> Capabilities
```

Get backend capabilities and features.

**Returns:** `Capabilities` object with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `supports_density_matrix` | `bool` | Density matrix support |
| `supports_statevector` | `bool` | Always `True` |
| `supports_gpu` | `bool` | Always `True` |
| `supports_multi_gpu` | `bool` | Multi-GPU capability |
| `max_qubits` | `int` | Maximum supported qubits |
| `supported_gates` | `list[str]` | List of supported gates |
| `precision_modes` | `list[str]` | Available precisions |
| `tensor_cores` | `bool` | Tensor core availability |

---

##### validate_circuit

```python
def validate_circuit(
    self,
    circuit: Any
) -> ValidationResult
```

Validate a circuit for cuQuantum execution.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `circuit` | `Any` | Circuit to validate |

**Returns:** `ValidationResult` with validation status and errors

---

##### estimate_resources

```python
def estimate_resources(
    self,
    circuit: Any
) -> ResourceEstimate
```

Estimate GPU resources required.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `circuit` | `Any` | Circuit to estimate |

**Returns:** `ResourceEstimate` with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `memory_bytes` | `int` | System memory estimate |
| `gpu_memory_bytes` | `int` | GPU memory required |
| `estimated_time_seconds` | `float` | Estimated execution time |
| `gpus_required` | `int` | Number of GPUs recommended |
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

Execute a quantum circuit on cuQuantum.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuit` | `Any` | Required | Circuit to execute |
| `options` | `dict` | `None` | Execution options |

**Execution Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `precision` | `str` | `"single"` | `"single"` or `"double"` |
| `shots` | `int` | `None` | Number of measurement shots |
| `gpu_id` | `int` | `0` | Primary GPU device |
| `multi_gpu` | `bool` | `False` | Enable multi-GPU |
| `gpu_ids` | `list[int]` | `None` | Specific GPUs to use |
| `use_memory_pool` | `bool` | `True` | Use memory pool |
| `async_execution` | `bool` | `False` | Async execution mode |
| `stream` | `CudaStream` | `None` | CUDA stream to use |
| `timeout` | `int` | `None` | Execution timeout |

**Returns:** `ExecutionResult` with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | `bool` | Execution success status |
| `counts` | `dict[str, int]` | Measurement counts |
| `statevector` | `np.ndarray` | Final state vector |
| `probabilities` | `np.ndarray` | State probabilities |
| `metadata` | `dict` | Execution metadata |

**Raises:**
- `CuQuantumCircuitError`: Invalid circuit
- `CuQuantumMemoryError`: GPU memory exceeded
- `CuQuantumRuntimeError`: Execution failure

---

##### execute_async

```python
async def execute_async(
    self,
    circuit: Any,
    options: dict[str, Any] | None = None
) -> ExecutionResult
```

Asynchronously execute a circuit.

**Parameters:** Same as `execute()`

**Returns:** `ExecutionResult` (await for result)

```python
import asyncio

result = await adapter.execute_async(circuit, options)
```

---

##### execute_batch

```python
def execute_batch(
    self,
    circuits: list[Any],
    options: dict[str, Any] | None = None
) -> list[ExecutionResult]
```

Execute multiple circuits in batch.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `circuits` | `list[Any]` | Circuits to execute |
| `options` | `dict` | Shared execution options |

**Returns:** `list[ExecutionResult]` - Results for each circuit

---

##### get_gpu_info

```python
def get_gpu_info(self) -> GPUDeviceInfo | list[GPUDeviceInfo]
```

Get GPU device information.

**Returns:** `GPUDeviceInfo` or list for multi-GPU

---

##### warmup

```python
def warmup(
    self,
    num_qubits: int = 10
) -> None
```

Warm up GPU with test execution.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_qubits` | `int` | `10` | Qubits for warmup circuit |

---

##### create_memory_pool

```python
def create_memory_pool(
    self,
    initial_size_mb: int = 1024,
    max_size_mb: int | None = None
) -> MemoryPool
```

Create a GPU memory pool.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_size_mb` | `int` | `1024` | Initial pool size |
| `max_size_mb` | `int` | `None` | Maximum pool size |

**Returns:** `MemoryPool` handle

---

##### get_memory_stats

```python
def get_memory_stats(self) -> MemoryStats
```

Get GPU memory statistics.

**Returns:** `MemoryStats` with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `total_bytes` | `int` | Total GPU memory |
| `free_bytes` | `int` | Free GPU memory |
| `used_bytes` | `int` | Used GPU memory |
| `pool_allocated` | `int` | Pool allocated |
| `pool_reserved` | `int` | Pool reserved |

---

##### clear_memory

```python
def clear_memory(
    self,
    force_gc: bool = False
) -> int
```

Clear GPU memory caches.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_gc` | `bool` | `False` | Force garbage collection |

**Returns:** `int` - Bytes freed

---

##### enable_profiling

```python
def enable_profiling(
    self,
    detailed: bool = False
) -> None
```

Enable execution profiling.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detailed` | `bool` | `False` | Enable detailed profiling |

---

##### get_profiling_results

```python
def get_profiling_results(self) -> ProfilingResults
```

Get profiling data from recent executions.

**Returns:** `ProfilingResults` with timing and memory data

---

### CuQuantumConfig

Configuration class for cuQuantum adapter.

```python
from proxima.backends.cuquantum_adapter import CuQuantumConfig

config = CuQuantumConfig(
    precision="single",
    gpu_id=0,
    use_memory_pool=True
)
```

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `precision` | `str` | `"single"` | Default precision |
| `gpu_id` | `int` | `0` | Default GPU device |
| `multi_gpu` | `bool` | `False` | Enable multi-GPU |
| `gpu_ids` | `list[int]` | `None` | Specific GPUs |
| `use_memory_pool` | `bool` | `True` | Enable memory pool |
| `memory_pool_size_mb` | `int` | `1024` | Pool initial size |
| `enable_tensor_cores` | `bool` | `True` | Use tensor cores |
| `max_qubits` | `int` | `32` | Maximum qubit limit |
| `fallback_to_cpu` | `bool` | `True` | CPU fallback on error |

#### Methods

##### from_dict

```python
@classmethod
def from_dict(
    cls,
    config: dict[str, Any]
) -> CuQuantumConfig
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

### GPUDeviceInfo

GPU device information.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `device_id` | `int` | GPU device ID |
| `name` | `str` | GPU device name |
| `compute_capability` | `tuple[int, int]` | Compute capability |
| `memory_total_mb` | `int` | Total memory in MB |
| `memory_free_mb` | `int` | Free memory in MB |
| `supports_tensor_cores` | `bool` | Tensor core support |
| `cuda_version` | `str` | CUDA version |

---

### MemoryPool

GPU memory pool handle.

#### Methods

```python
def allocate(self, size_bytes: int) -> int  # Returns pointer
def deallocate(self, ptr: int) -> None
def get_stats(self) -> dict
def clear(self) -> int  # Returns bytes freed
def close(self) -> None
```

---

## Enums

### CuQuantumPrecision

```python
from proxima.backends.cuquantum_adapter import CuQuantumPrecision

class CuQuantumPrecision(str, Enum):
    SINGLE = "single"   # float32, faster
    DOUBLE = "double"   # float64, more accurate
```

---

### MultiGPUStrategy

```python
from proxima.backends.cuquantum_adapter import MultiGPUStrategy

class MultiGPUStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"     # Distribute evenly
    MEMORY_BASED = "memory_based"   # Use GPUs with most memory
    PERFORMANCE = "performance"     # Use fastest GPUs
```

---

## Exceptions

### CuQuantumInstallationError

Raised when cuQuantum is not properly installed.

```python
from proxima.backends.cuquantum_adapter import CuQuantumInstallationError

try:
    adapter = CuQuantumAdapter()
except CuQuantumInstallationError as e:
    print(f"cuQuantum not available: {e}")
    print(e.installation_help)
```

**Attributes:**
- `installation_help`: Installation instructions

---

### CuQuantumGPUError

Raised for GPU-related errors.

```python
from proxima.backends.cuquantum_adapter import CuQuantumGPUError

try:
    adapter = CuQuantumAdapter()
except CuQuantumGPUError as e:
    print(f"GPU error: {e}")
    print(f"Minimum compute capability: {e.min_compute_capability}")
```

**Attributes:**
- `min_compute_capability`: Minimum required (7.0)
- `actual_compute_capability`: Device compute capability

---

### CuQuantumMemoryError

Raised when GPU memory is exceeded.

```python
from proxima.backends.cuquantum_adapter import CuQuantumMemoryError

try:
    result = adapter.execute(large_circuit)
except CuQuantumMemoryError as e:
    print(f"GPU OOM: {e}")
    print(f"Required: {e.required_bytes / 1e9:.2f} GB")
    print(f"Available: {e.available_bytes / 1e9:.2f} GB")
```

**Attributes:**
- `required_bytes`: Memory required
- `available_bytes`: Memory available

---

### CuQuantumDriverError

Raised for CUDA driver errors.

```python
from proxima.backends.cuquantum_adapter import CuQuantumDriverError

try:
    result = adapter.execute(circuit)
except CuQuantumDriverError as e:
    print(f"Driver error: {e}")
    print(f"Error code: {e.cuda_error_code}")
```

**Attributes:**
- `cuda_error_code`: CUDA error code

---

## Utility Functions

### check_cuquantum_availability

```python
def check_cuquantum_availability() -> tuple[bool, str]
```

Check if cuQuantum is available.

**Returns:** Tuple of (available, message)

```python
from proxima.backends.cuquantum_adapter import check_cuquantum_availability

available, message = check_cuquantum_availability()
if available:
    print(f"cuQuantum ready: {message}")
else:
    print(f"cuQuantum unavailable: {message}")
```

---

### get_available_gpus

```python
def get_available_gpus() -> list[GPUDeviceInfo]
```

Get list of available compatible GPUs.

**Returns:** List of `GPUDeviceInfo` objects

---

### check_gpu_compute_capability

```python
def check_gpu_compute_capability(
    device_id: int = 0
) -> tuple[bool, tuple[int, int]]
```

Check if GPU meets compute capability requirements.

**Returns:** Tuple of (meets_requirements, (major, minor))

---

## Examples

### Basic GPU Execution

```python
from proxima.backends.cuquantum_adapter import CuQuantumAdapter
from proxima.circuit import Circuit

# Create adapter
adapter = CuQuantumAdapter()

# Check GPU
gpu_info = adapter.get_gpu_info()
print(f"GPU: {gpu_info.name} ({gpu_info.memory_total_mb} MB)")

# Create circuit
circuit = Circuit(num_qubits=20)
circuit.h(range(20))

# Warm up GPU
adapter.warmup()

# Execute
result = adapter.execute(
    circuit,
    options={'shots': 1000}
)

print(f"Counts: {len(result.counts)} unique states")
```

### Multi-GPU Execution

```python
from proxima.backends.cuquantum_adapter import CuQuantumAdapter, CuQuantumConfig

# Configure multi-GPU
config = CuQuantumConfig(
    multi_gpu=True,
    gpu_ids=[0, 1],
    precision="single"
)

adapter = CuQuantumAdapter(config=config)

# Execute on multiple GPUs
result = adapter.execute(
    large_circuit,
    options={'multi_gpu': True}
)

print(f"GPUs used: {result.metadata['gpus_used']}")
```

### Memory Pool Usage

```python
# Create memory pool
pool = adapter.create_memory_pool(
    initial_size_mb=2048,
    max_size_mb=8192
)

# Execute multiple circuits
for circuit in circuits:
    result = adapter.execute(
        circuit,
        options={'use_memory_pool': True}
    )

# Check memory stats
stats = adapter.get_memory_stats()
print(f"Pool usage: {stats.pool_allocated / 1e6:.1f} MB")

# Cleanup
pool.close()
```

### Async Batch Execution

```python
import asyncio

async def run_batch():
    # Execute circuits asynchronously
    tasks = [
        adapter.execute_async(circuit, {'shots': 1000})
        for circuit in circuits
    ]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(run_batch())
```

### Error Handling with Fallback

```python
from proxima.backends.cuquantum_adapter import (
    CuQuantumAdapter,
    CuQuantumMemoryError
)
from proxima.backends import get_default_backend

try:
    adapter = CuQuantumAdapter()
    result = adapter.execute(circuit)
except CuQuantumMemoryError:
    print("GPU OOM, falling back to CPU...")
    adapter = get_default_backend()
    result = adapter.execute(circuit)
```

---

## See Also

- [cuQuantum Installation Guide](../backends/cuquantum-installation.md)
- [cuQuantum Usage Guide](../backends/cuquantum-usage.md)
- [Backend Selection](../backends/backend-selection.md)
