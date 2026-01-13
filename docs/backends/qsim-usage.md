# qsim Backend Usage Guide

> **Version:** 1.0  
> **Last Updated:** January 13, 2026  
> **Backend:** qsim (Google's High-Performance Quantum Simulator)

---

## Overview

This guide covers practical usage of the qsim backend in Proxima for high-performance CPU-based quantum circuit simulation. qsim provides excellent performance through AVX vectorization and OpenMP parallelization, making it ideal for state vector simulations when GPU is unavailable.

---

## Quick Start

### Basic Usage

```python
from proxima.backends import get_backend
from proxima.circuit import Circuit

# Get the qsim backend
backend = get_backend('qsim')

# Check availability
if backend.is_available():
    caps = backend.get_capabilities()
    print(f"qsim ready with {caps.max_threads} threads, AVX2: {caps.avx2_enabled}")
else:
    print("qsim not available")

# Create a circuit
circuit = Circuit(num_qubits=20)
circuit.h(0)
for i in range(19):
    circuit.cnot(i, i + 1)

# Execute
result = backend.execute(circuit, options={'shots': 1000})
print(f"Measurement counts: {result.counts}")
```

### Checking CPU Features

```python
from proxima.backends import get_backend

backend = get_backend('qsim')

# Get CPU information
cpu_info = backend.get_cpu_info()
print(f"CPU: {cpu_info.brand}")
print(f"Cores: {cpu_info.cores} ({cpu_info.threads} threads)")
print(f"AVX2: {'✅' if cpu_info.avx2_supported else '❌'}")
print(f"AVX512: {'✅' if cpu_info.avx512_supported else '❌'}")
print(f"L3 Cache: {cpu_info.cache_l3_mb} MB")
```

---

## Execution Modes

### State Vector Simulation

The default mode for exact quantum state simulation:

```python
result = backend.execute(
    circuit,
    options={
        'output': 'statevector'
    }
)

# Access the state vector
statevector = result.statevector
print(f"State vector shape: {statevector.shape}")
print(f"State vector norm: {np.linalg.norm(statevector)}")

# Get specific amplitudes
print(f"|00...0> amplitude: {statevector[0]}")
print(f"|11...1> amplitude: {statevector[-1]}")
```

### Shot-Based Sampling

For measurement-based results:

```python
result = backend.execute(
    circuit,
    options={
        'shots': 10000
    }
)

# Access measurement counts
for bitstring, count in sorted(result.counts.items(), key=lambda x: -x[1])[:10]:
    probability = count / 10000
    print(f"  {bitstring}: {count} ({probability:.2%})")
```

### Probability Distribution

Get probabilities without sampling noise:

```python
result = backend.execute(
    circuit,
    options={
        'output': 'probabilities'
    }
)

# Access probability distribution
total_prob = 0
for state, prob in result.probabilities.items():
    if prob > 0.01:
        print(f"  |{state}>: {prob:.4f}")
        total_prob += prob

print(f"Sum of shown probabilities: {total_prob:.4f}")
```

---

## Thread Configuration

### Setting Thread Count

```python
# Method 1: Via execution options
result = backend.execute(
    circuit,
    options={
        'num_threads': 16
    }
)

# Method 2: Via environment variable
import os
os.environ['OMP_NUM_THREADS'] = '16'

# Method 3: Auto-select optimal threads
result = backend.execute(
    circuit,
    options={
        'num_threads': 'auto'  # Let qsim choose
    }
)

print(f"Threads used: {result.metadata['threads_used']}")
```

### Thread Optimization

```python
from proxima.backends import get_backend
import multiprocessing

backend = get_backend('qsim')

# Get system info
total_cores = multiprocessing.cpu_count()
print(f"Total CPU threads: {total_cores}")

# Optimal thread count (leave some for system)
optimal_threads = max(1, total_cores - 2)
print(f"Recommended threads: {optimal_threads}")

# Execute with optimal threads
result = backend.execute(
    circuit,
    options={'num_threads': optimal_threads}
)
```

### Thread Scaling Analysis

```python
def analyze_thread_scaling(circuit, max_threads=None):
    """Analyze how performance scales with thread count."""
    import time
    
    if max_threads is None:
        max_threads = multiprocessing.cpu_count()
    
    results = []
    
    for threads in [1, 2, 4, 8, 16, max_threads]:
        if threads > max_threads:
            continue
            
        times = []
        for _ in range(5):
            start = time.perf_counter()
            backend.execute(circuit, options={'num_threads': threads})
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        speedup = results[0]['time'] / avg_time if results else 1.0
        efficiency = speedup / threads
        
        results.append({
            'threads': threads,
            'time': avg_time,
            'speedup': speedup,
            'efficiency': efficiency
        })
        
        print(f"{threads} threads: {avg_time:.2f} ms, "
              f"speedup: {speedup:.2f}x, efficiency: {efficiency:.1%}")
    
    return results

# Analyze scaling
scaling = analyze_thread_scaling(large_circuit)
```

---

## Gate Fusion

### Enabling Gate Fusion

Gate fusion combines adjacent gates for better performance:

```python
# Enable gate fusion (default)
result = backend.execute(
    circuit,
    options={
        'gate_fusion': True
    }
)

print(f"Original gates: {result.metadata['original_gates']}")
print(f"Fused gates: {result.metadata['fused_gates']}")
print(f"Fusion ratio: {result.metadata['fusion_ratio']:.2%}")
```

### Fusion Strategies

```python
# Aggressive fusion - maximum performance
result = backend.execute(
    circuit,
    options={
        'fusion_strategy': 'aggressive'
    }
)

# Balanced fusion - good performance, moderate compilation time
result = backend.execute(
    circuit,
    options={
        'fusion_strategy': 'balanced'
    }
)

# No fusion - fastest compilation, slower execution
result = backend.execute(
    circuit,
    options={
        'gate_fusion': False
    }
)
```

### When to Disable Fusion

```python
# Disable fusion for very simple circuits (fusion overhead not worth it)
simple_circuit = Circuit(num_qubits=5)
simple_circuit.h(0)
simple_circuit.cnot(0, 1)

result = backend.execute(
    simple_circuit,
    options={
        'gate_fusion': False  # Skip fusion for simple circuits
    }
)
```

---

## Cirq Integration

### Using Cirq Circuits Directly

```python
import cirq
from proxima.backends import get_backend

# Create Cirq circuit
qubits = cirq.LineQubit.range(10)
cirq_circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.measure(*qubits, key='result')
)

# Execute with Proxima qsim backend
backend = get_backend('qsim')
result = backend.execute(cirq_circuit, options={'shots': 1000})

print(f"Counts: {result.counts}")
```

### Converting from Qiskit

```python
from qiskit import QuantumCircuit
from proxima.backends import get_backend

# Create Qiskit circuit
qiskit_circuit = QuantumCircuit(5)
qiskit_circuit.h(0)
qiskit_circuit.cx(0, 1)
qiskit_circuit.cx(1, 2)

# Convert to Cirq (qsim backend handles this automatically)
backend = get_backend('qsim')
cirq_circuit = backend.convert_from_qiskit(qiskit_circuit)

# Execute
result = backend.execute(cirq_circuit)
```

### Native qsim Simulator Access

```python
import cirq
import qsimcirq

# Create circuit
qubits = cirq.LineQubit.range(5)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT.on_each(zip(qubits[:-1], qubits[1:]))
)

# Create qsim simulator directly
options = qsimcirq.QSimOptions(
    max_fused_gate_size=4,
    cpu_threads=8,
    verbosity=0
)
simulator = qsimcirq.QSimSimulator(options)

# Run simulation
result = simulator.simulate(circuit)
print(f"Final state: {result.final_state_vector}")
```

---

## Gate Support

### Supported Gates

```python
caps = backend.get_capabilities()
print("Supported gates:")
for gate in caps.supported_gates:
    print(f"  - {gate}")

# Common supported gates:
# H, X, Y, Z, S, T, Rx, Ry, Rz, 
# CNOT, CZ, CY, SWAP, ISWAP,
# CCX (Toffoli), CCZ, CSWAP
```

### Handling Unsupported Gates

```python
# Create circuit with potentially unsupported gate
circuit = Circuit(num_qubits=3)
circuit.add_gate('custom_gate', [0], params=[0.5])

# Validate before execution
validation = backend.validate_circuit(circuit)

if not validation.valid:
    print(f"Unsupported gates: {validation.unsupported_gates}")
    
    # Option 1: Decompose unsupported gates
    decomposed = backend.decompose_unsupported_gates(circuit)
    result = backend.execute(decomposed)
    
    # Option 2: Use different backend
    from proxima.backends import get_backend
    cirq_backend = get_backend('cirq')
    result = cirq_backend.execute(circuit)
```

### Mid-Circuit Measurement Limitation

```python
# qsim has limited mid-circuit measurement support
circuit = Circuit(num_qubits=3)
circuit.h(0)
circuit.measure(0)  # Mid-circuit measurement
circuit.cnot(0, 1)  # Depends on measurement

validation = backend.validate_circuit(circuit)
if not validation.valid:
    print("Mid-circuit measurement not fully supported")
    print("Consider using Cirq backend for this circuit")
```

---

## Performance Optimization

### Circuit Optimization

```python
# Optimize circuit for qsim
optimized = backend.optimize_for_qsim(
    circuit,
    options={
        'gate_ordering': True,     # Reorder gates for better fusion
        'remove_identities': True, # Remove identity gates
        'merge_rotations': True    # Merge consecutive rotations
    }
)

print(f"Gates before: {circuit.gate_count}")
print(f"Gates after: {optimized.gate_count_after}")
print(f"Optimizations: {optimized.optimizations_applied}")
```

### Memory Optimization

```python
# Estimate memory requirements
estimate = backend.estimate_resources(large_circuit)
print(f"Memory required: {estimate.memory_bytes / 1e9:.2f} GB")

# Check available memory
import psutil
available = psutil.virtual_memory().available / 1e9
print(f"Available RAM: {available:.2f} GB")

if estimate.memory_bytes > available * 0.8:
    print("WARNING: May run out of memory!")
    print("Consider reducing qubit count or using memory-mapped execution")
```

### Verbosity Configuration

```python
# Enable verbose output for debugging
result = backend.execute(
    circuit,
    options={
        'verbosity': 2  # 0=silent, 1=info, 2=debug
    }
)
```

---

## Batch Execution

### Execute Multiple Circuits

```python
# Prepare batch of circuits
circuits = []
for i in range(100):
    circuit = Circuit(num_qubits=15)
    circuit.h(0)
    circuit.rz(i * 0.01, 0)  # Different angle each circuit
    circuit.cnot(0, 1)
    circuits.append(circuit)

# Execute batch
batch_result = backend.execute_batch(circuits)

print(f"Batch size: {len(batch_result.results)}")
print(f"Total time: {batch_result.total_time_ms:.2f} ms")
print(f"Avg per circuit: {batch_result.avg_time_per_circuit_ms:.2f} ms")
```

### Parallel Batch Execution

```python
# Execute with parallel processing
batch_result = backend.execute_batch(
    circuits,
    options={
        'parallel': True,
        'batch_size': 10  # Process 10 circuits at a time
    }
)
```

---

## Error Handling

### Memory Errors

```python
try:
    result = backend.execute(huge_circuit)
except MemoryError as e:
    print(f"Out of memory: {e}")
    
    # Options:
    # 1. Reduce qubit count
    # 2. Use a machine with more RAM
    # 3. Consider approximate simulation methods
```

### Timeout Handling

```python
try:
    result = backend.execute(
        circuit,
        options={'timeout': 60}  # 60 second timeout
    )
except TimeoutError:
    print("Execution timed out")
    
    # Reduce threads for better resource sharing
    result = backend.execute(
        circuit,
        options={'num_threads': 4, 'timeout': 120}
    )
```

### Graceful Degradation

```python
def execute_with_fallback(circuit):
    """Execute with automatic fallback."""
    backends = ['qsim', 'cirq', 'qiskit']
    
    for backend_name in backends:
        try:
            backend = get_backend(backend_name)
            if backend.is_available():
                return backend.execute(circuit)
        except Exception as e:
            print(f"{backend_name} failed: {e}")
            continue
    
    raise RuntimeError("All backends failed")

result = execute_with_fallback(circuit)
```

---

## Resource Estimation

### Memory Estimation

```python
# Estimate resources before execution
for n_qubits in [15, 20, 25, 30]:
    circuit = Circuit(num_qubits=n_qubits)
    estimate = backend.estimate_resources(circuit)
    
    memory_gb = estimate.memory_bytes / 1e9
    print(f"{n_qubits} qubits: {memory_gb:.2f} GB")
```

### Time Estimation

```python
estimate = backend.estimate_resources(circuit)

print(f"Estimated time: {estimate.estimated_time_ms:.2f} ms")
print(f"Confidence: {estimate.confidence}")

if estimate.warnings:
    for warning in estimate.warnings:
        print(f"⚠️ {warning}")
```

---

## Comparison with Other Backends

### Performance Comparison

```python
from proxima.data.compare import compare_backends

# Compare qsim with other CPU backends
results = compare_backends(
    circuit,
    backends=['qsim', 'cirq', 'qiskit'],
    shots=1000
)

for name, result in results.items():
    print(f"{name}:")
    print(f"  Time: {result.execution_time_ms:.2f} ms")
    print(f"  Memory: {result.memory_bytes / 1e6:.1f} MB")
```

### Accuracy Comparison

```python
# Compare state vectors from different backends
qsim_result = get_backend('qsim').execute(circuit, options={'output': 'statevector'})
cirq_result = get_backend('cirq').execute(circuit, options={'output': 'statevector'})

# Calculate fidelity
fidelity = np.abs(np.dot(
    qsim_result.statevector.conj(),
    cirq_result.statevector
)) ** 2

print(f"Fidelity between qsim and cirq: {fidelity:.10f}")
```

---

## CLI Usage

```bash
# Run simulation with qsim backend
proxima run circuit.json --backend qsim --shots 1000

# Run with specific thread count
proxima run circuit.json --backend qsim --threads 8

# Compare qsim with other backends
proxima compare circuit.json --backends qsim,cirq,qiskit

# Benchmark qsim performance
proxima benchmark --backend qsim --qubits 10,15,20,25
```

---

## Configuration Options

### All Execution Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_threads` | int/str | "auto" | Number of OpenMP threads |
| `gate_fusion` | bool | True | Enable gate fusion |
| `fusion_strategy` | str | "balanced" | Fusion strategy |
| `verbosity` | int | 0 | Logging verbosity |
| `shots` | int | None | Number of measurement shots |
| `output` | str | "counts" | Output format |
| `timeout` | int | None | Execution timeout (seconds) |

### Environment Configuration

```python
# Set defaults via config
from proxima.config import Config
config = Config()
config.set('qsim.default_threads', 16)
config.set('qsim.fusion_strategy', 'aggressive')
config.save()
```

---

## See Also

- [qsim Installation Guide](qsim-installation.md) - Installation instructions
- [Backend Selection Guide](backend-selection.md) - Choosing the right backend
- [cuQuantum Usage Guide](cuquantum-usage.md) - GPU alternative
- [Google qsim GitHub](https://github.com/quantumlib/qsim)
