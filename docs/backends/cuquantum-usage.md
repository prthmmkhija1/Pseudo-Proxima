# cuQuantum Backend Usage Guide

> **Version:** 1.0  
> **Last Updated:** January 13, 2026  
> **Backend:** cuQuantum (NVIDIA GPU-Accelerated Quantum Simulation)

---

## Overview

This guide covers practical usage of the cuQuantum backend in Proxima for GPU-accelerated quantum circuit simulation. cuQuantum provides significant speedups for large circuits and is the recommended backend when NVIDIA GPUs are available.

---

## Quick Start

### Basic Usage

```python
from proxima.backends import get_backend
from proxima.circuit import Circuit

# Get the cuQuantum backend
backend = get_backend('cuquantum')

# Check availability
if backend.is_available():
    print(f"cuQuantum ready with {backend.get_capabilities().gpu_memory_mb} MB GPU memory")
else:
    print("cuQuantum not available, falling back to CPU")

# Create a simple circuit
circuit = Circuit(num_qubits=20)
circuit.h(0)
for i in range(19):
    circuit.cnot(i, i + 1)

# Execute on GPU
result = backend.execute(circuit, options={'shots': 1000})
print(f"Measurement counts: {result.counts}")
```

### Checking GPU Information

```python
from proxima.backends import get_backend

backend = get_backend('cuquantum')

# Get GPU device information
gpu_info = backend.get_gpu_info(device_id=0)
print(f"GPU: {gpu_info.name}")
print(f"Total Memory: {gpu_info.total_memory_mb} MB")
print(f"Free Memory: {gpu_info.free_memory_mb} MB")
print(f"Compute Capability: {gpu_info.compute_capability}")

# Get all GPUs
all_gpus = backend.get_all_gpu_info()
for gpu in all_gpus:
    print(f"  GPU {gpu.device_id}: {gpu.name}")
```

---

## Execution Modes

### State Vector Simulation

The default mode for exact quantum state simulation:

```python
result = backend.execute(
    circuit,
    options={
        'method': 'statevector',
        'output': 'statevector'  # Return full state vector
    }
)

# Access the state vector
statevector = result.statevector
print(f"State vector shape: {statevector.shape}")
print(f"State vector norm: {np.linalg.norm(statevector)}")
```

### Shot-Based Sampling

For measurement-based results:

```python
result = backend.execute(
    circuit,
    options={
        'shots': 10000,
        'output': 'counts'
    }
)

# Access measurement counts
for bitstring, count in result.counts.items():
    probability = count / 10000
    print(f"  {bitstring}: {count} ({probability:.2%})")
```

### Probability Distribution

Get probabilities without sampling:

```python
result = backend.execute(
    circuit,
    options={
        'output': 'probabilities'
    }
)

# Access probability distribution
for state, prob in result.probabilities.items():
    if prob > 0.01:  # Only show significant probabilities
        print(f"  |{state}>: {prob:.4f}")
```

---

## GPU Memory Management

### Estimating Memory Requirements

```python
# Check memory requirements before execution
estimate = backend.estimate_resources(circuit)

print(f"State vector memory: {estimate.statevector_memory_bytes / 1e9:.2f} GB")
print(f"Workspace memory: {estimate.workspace_memory_bytes / 1e9:.2f} GB")
print(f"Total GPU memory needed: {estimate.total_gpu_memory_bytes / 1e9:.2f} GB")

if estimate.exceeds_available:
    print("WARNING: Circuit may exceed GPU memory!")
    print(f"Available: {estimate.available_memory_bytes / 1e9:.2f} GB")
```

### Memory Pool Management

For batch execution or repeated simulations:

```python
# Create a memory pool for efficient allocation
backend.create_memory_pool(size_mb=8000)

# Run multiple circuits
for circuit in circuits:
    result = backend.execute(circuit)

# Check pool statistics
stats = backend.get_memory_pool_stats()
print(f"Pool size: {stats['pool_size_mb']} MB")
print(f"Peak usage: {stats['peak_usage_mb']} MB")
print(f"Allocations: {stats['allocation_count']}")

# Clean up pool when done
backend.clear_memory_pool()
```

### Handling Out of Memory

```python
try:
    result = backend.execute(large_circuit)
except MemoryError as e:
    print(f"GPU out of memory: {e}")
    
    # Option 1: Use single precision
    result = backend.execute(
        large_circuit,
        options={'precision': 'single'}
    )
    
    # Option 2: Fall back to CPU
    cpu_backend = get_backend('qiskit')
    result = cpu_backend.execute(large_circuit)
```

---

## Precision Modes

### Double Precision (Default)

Maximum numerical accuracy:

```python
result = backend.execute(
    circuit,
    options={
        'precision': 'double',  # complex128 (default)
    }
)
# Memory: 16 bytes per amplitude
```

### Single Precision

Reduced memory, faster execution:

```python
result = backend.execute(
    circuit,
    options={
        'precision': 'single',  # complex64
    }
)
# Memory: 8 bytes per amplitude (50% reduction)
```

### Comparing Precision Modes

```python
# Compare accuracy between precision modes
double_result = backend.execute(circuit, options={'precision': 'double'})
single_result = backend.execute(circuit, options={'precision': 'single'})

# Calculate fidelity
fidelity = np.abs(np.dot(
    double_result.statevector.conj(),
    single_result.statevector.astype(np.complex128)
)) ** 2

print(f"Fidelity (single vs double): {fidelity:.10f}")
```

---

## Multi-GPU Execution

### Enabling Multi-GPU

```python
# Check multi-GPU configuration
config = backend.get_multi_gpu_config()
print(f"Available GPUs: {config['gpu_ids']}")
print(f"NVLink available: {config['nvlink_available']}")

# Execute with multiple GPUs
result = backend.execute(
    very_large_circuit,  # 30+ qubits
    options={
        'multi_gpu': True,
        'gpu_ids': [0, 1],  # Use GPUs 0 and 1
    }
)

print(f"GPUs used: {result.metadata['gpus_used']}")
```

### GPU Selection Strategies

```python
# Let the backend choose the best GPU
best_gpu = backend.select_best_gpu(num_qubits=28)
print(f"Selected GPU: {best_gpu}")

# Force specific GPU
result = backend.execute(
    circuit,
    options={
        'device_id': 1,  # Use GPU 1
    }
)
```

---

## Async and Batch Execution

### Asynchronous Execution

```python
# Start async execution
task = backend.execute_async(circuit)
print(f"Task ID: {task.task_id}")
print(f"Status: {task.status}")

# Do other work...

# Check result
result = backend.get_async_result(task.task_id)
if result.status == 'completed':
    print(f"Result: {result.result.counts}")
```

### Batch Execution

```python
# Prepare batch of circuits
circuits = [create_random_circuit(20) for _ in range(100)]

# Execute all circuits
batch_result = backend.execute_batch(
    circuits,
    options={
        'parallel_streams': 4,  # Use 4 CUDA streams
    }
)

print(f"Batch size: {batch_result.metadata['batch_size']}")
print(f"Total time: {batch_result.metadata['total_time_ms']:.2f} ms")
print(f"Average per circuit: {batch_result.metadata['avg_time_per_circuit_ms']:.2f} ms")
```

### Parameter Sweep

```python
# Variational circuit with parameters
base_circuit = create_variational_circuit(num_qubits=10, num_layers=5)

# Define parameter sets
parameter_sets = [
    [0.1 * i for _ in range(15)]  # 15 parameters
    for i in range(100)
]

# Execute parameter sweep efficiently
sweep_result = backend.execute_parameter_sweep(
    base_circuit,
    parameter_sets
)

# Analyze results
for i, result in enumerate(sweep_result.results):
    energy = result.expectation_value
    print(f"Parameters {i}: Energy = {energy:.4f}")
```

---

## Performance Optimization

### GPU Warmup

```python
# Warm up GPU before performance-critical operations
warmup_time = backend.warm_up_gpu(num_qubits=15)
print(f"Warmup completed in {warmup_time:.2f} seconds")
```

### Kernel Caching

```python
# Enable kernel caching for repeated circuits
result = backend.execute(
    circuit,
    options={
        'cache_kernels': True,
    }
)

# Subsequent executions of similar circuits will be faster
```

### Profiling Execution

```python
result = backend.execute(
    circuit,
    options={
        'profile': True,
    }
)

# Access profiling data
profile = result.metadata
print(f"Gate application time: {profile['gate_application_time_ms']:.2f} ms")
print(f"Memory transfer time: {profile['memory_transfer_time_ms']:.2f} ms")
print(f"Kernel launch overhead: {profile['kernel_launch_overhead_ms']:.2f} ms")
print(f"Total GPU time: {profile['total_gpu_time_ms']:.2f} ms")
```

---

## Error Handling

### Graceful Fallback

```python
from proxima.backends import get_backend

def execute_with_fallback(circuit, preferred_backend='cuquantum'):
    """Execute circuit with automatic fallback to CPU."""
    
    backend = get_backend(preferred_backend)
    
    if not backend.is_available():
        print(f"{preferred_backend} not available, using CPU")
        backend = get_backend('qiskit')
    
    try:
        return backend.execute(circuit)
    except MemoryError:
        print("GPU out of memory, falling back to CPU")
        cpu_backend = get_backend('qiskit')
        return cpu_backend.execute(circuit)

# Use the function
result = execute_with_fallback(large_circuit)
```

### Handling CUDA Errors

```python
try:
    result = backend.execute(circuit)
except RuntimeError as e:
    if "CUDA" in str(e):
        print(f"CUDA error: {e}")
        # Possible solutions:
        # 1. Reduce circuit size
        # 2. Clear GPU memory
        # 3. Restart Python session
    raise
```

---

## Integration Examples

### With Proxima Compare

```python
from proxima.data.compare import compare_backends

# Compare cuQuantum with other backends
results = compare_backends(
    circuit,
    backends=['cuquantum', 'qiskit', 'cirq'],
    shots=1000
)

for backend_name, result in results.items():
    print(f"{backend_name}:")
    print(f"  Time: {result.execution_time_ms:.2f} ms")
    print(f"  Counts: {result.counts}")
```

### With Proxima Pipeline

```python
from proxima.data.pipeline import Pipeline, stage

@stage(name="gpu_simulation")
def run_on_gpu(circuit, context):
    backend = get_backend('cuquantum')
    return backend.execute(circuit, options={'shots': 1000})

# Build pipeline with GPU execution
pipeline = Pipeline()
pipeline.add_stage(run_on_gpu)
result = pipeline.run(my_circuit)
```

### CLI Usage

```bash
# Run simulation with cuQuantum backend
proxima run circuit.json --backend cuquantum --shots 1000

# Compare backends including cuQuantum
proxima compare circuit.json --backends cuquantum,qiskit,cirq

# Check GPU status
proxima config --show-gpu
```

---

## Configuration Options

### All Execution Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `method` | str | "statevector" | Simulation method |
| `precision` | str | "double" | "single" or "double" |
| `shots` | int | None | Number of measurement shots |
| `output` | str | "counts" | Output format: counts, statevector, probabilities |
| `device_id` | int | 0 | GPU device to use |
| `multi_gpu` | bool | False | Enable multi-GPU execution |
| `gpu_ids` | list | None | Specific GPUs to use |
| `blocking` | bool | True | Wait for completion |
| `profile` | bool | False | Enable profiling |
| `cache_kernels` | bool | False | Cache CUDA kernels |
| `timeout` | int | None | Execution timeout (seconds) |

### Environment Configuration

```python
# Configure via environment variables
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PROXIMA_CUQUANTUM_PRECISION'] = 'single'

# Or via Proxima config
from proxima.config import Config
config = Config()
config.set('cuquantum.default_precision', 'single')
config.set('cuquantum.device_id', 0)
config.save()
```

---

## Benchmarking

### Performance Comparison

```python
import time

def benchmark_execution(backend, circuit, iterations=10):
    """Benchmark circuit execution."""
    times = []
    
    # Warmup
    backend.execute(circuit)
    
    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        backend.execute(circuit)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': min(times),
        'max_ms': max(times),
    }

# Compare GPU vs CPU
gpu_stats = benchmark_execution(get_backend('cuquantum'), circuit)
cpu_stats = benchmark_execution(get_backend('qiskit'), circuit)

print(f"GPU: {gpu_stats['mean_ms']:.2f} ± {gpu_stats['std_ms']:.2f} ms")
print(f"CPU: {cpu_stats['mean_ms']:.2f} ± {cpu_stats['std_ms']:.2f} ms")
print(f"Speedup: {cpu_stats['mean_ms'] / gpu_stats['mean_ms']:.1f}x")
```

---

## See Also

- [cuQuantum Installation Guide](cuquantum-installation.md) - Installation instructions
- [Backend Selection Guide](backend-selection.md) - Choosing the right backend
- [QuEST Usage Guide](quest-usage.md) - Alternative high-performance backend
- [qsim Usage Guide](qsim-usage.md) - CPU-optimized alternative
