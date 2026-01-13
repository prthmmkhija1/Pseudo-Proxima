# Backend Migration Guide

This guide covers migrating between different quantum simulation backends in Proxima.

## Overview

Proxima supports multiple backends with varying capabilities. This guide helps you:

- Choose the right backend for your use case
- Migrate circuits between backends
- Handle backend-specific features

## Backend Comparison

| Backend | Max Qubits | GPU Support | Density Matrix | Best For |
|---------|------------|-------------|----------------|----------|
| Cirq | 30 | No | Yes | General use, Google hardware |
| Qiskit Aer | 32 | Yes | Yes | IBM hardware, noise models |
| LRET | 40 | No | No | Research, tensor networks |
| QuEST | 40 | Yes | Yes | High-performance, distributed |
| cuQuantum | 32+ | Yes (required) | Yes | NVIDIA GPU acceleration |
| qsim | 40 | Yes | No | Large circuits, Google TPU |

## Migrating from Cirq

### To Qiskit Aer

```python
# Cirq circuit
import cirq

q0, q1 = cirq.LineQubit.range(2)
cirq_circuit = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
])

# Convert to OpenQASM (intermediate format)
qasm = cirq.contrib.qasm.circuit_to_qasm_output(cirq_circuit)

# Use with Qiskit Aer backend
from proxima.backends import get_backend

backend = get_backend("qiskit_aer")
result = backend.run_qasm(qasm.code, shots=1000)
```

### To QuEST

```python
# Use Proxima's unified interface
from proxima.backends import get_backend
from proxima.core.circuit import Circuit

# Define circuit in Proxima format
circuit = Circuit(num_qubits=2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Run on QuEST
backend = get_backend("quest", num_threads=4)
result = backend.run(circuit, shots=1000)
```

## Migrating from Qiskit

### To Cirq

```python
# Qiskit circuit
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Convert via OpenQASM
qasm = qc.qasm()

# Use with Cirq backend
from proxima.backends import get_backend

backend = get_backend("cirq")
result = backend.run_qasm(qasm, shots=1000)
```

### To cuQuantum (GPU)

```python
# Qiskit circuit
from qiskit import QuantumCircuit

qc = QuantumCircuit(20)
for i in range(19):
    qc.h(i)
    qc.cx(i, i+1)
qc.measure_all()

# Convert and run on GPU
from proxima.backends import get_backend

backend = get_backend("cuquantum", device="cuda:0")
result = backend.run_qasm(qc.qasm(), shots=1000)
```

## Circuit Format Conversion

### OpenQASM as Intermediate

OpenQASM 2.0 serves as a universal intermediate format:

```python
from proxima.core.conversion import convert_circuit

# From any supported format
qasm = convert_circuit(circuit, source_format="cirq", target_format="openqasm")

# To any backend
result = backend.run_qasm(qasm, shots=1000)
```

### Native Format Preservation

For best performance, use native formats when possible:

```python
from proxima.backends import get_backend

# Cirq native
cirq_backend = get_backend("cirq")
result = cirq_backend.run_native(cirq_circuit, shots=1000)

# Qiskit native
qiskit_backend = get_backend("qiskit_aer")
result = qiskit_backend.run_native(qiskit_circuit, shots=1000)
```

## Handling Backend-Specific Features

### Noise Models

Not all backends support the same noise models:

```python
from proxima.backends import get_backend

# Qiskit Aer - Full noise model support
backend = get_backend("qiskit_aer")
backend.set_noise_model(
    depolarizing_error=0.01,
    readout_error=0.02
)

# Cirq - Different noise API
backend = get_backend("cirq")
backend.set_noise(
    gate_noise=cirq.depolarize(0.01),
    measurement_noise=cirq.bit_flip(0.02)
)

# QuEST - Density matrix simulation
backend = get_backend("quest")
backend.enable_density_matrix()  # Required for noise
```

### GPU Acceleration

```python
from proxima.backends import get_backend

# Check GPU availability
backend = get_backend("cuquantum")
if backend.is_available():
    # Configure GPU
    backend.configure(
        device="cuda:0",
        precision="single"  # Faster, less memory
    )
    result = backend.run(circuit, shots=1000)
else:
    # Fallback to CPU
    backend = get_backend("qiskit_aer")
    result = backend.run(circuit, shots=1000)
```

### Distributed Simulation

```python
from proxima.backends import get_backend

# QuEST distributed mode
backend = get_backend("quest")
backend.configure(
    distributed=True,
    num_nodes=4,
    ranks_per_node=1
)

# Large circuit simulation
result = backend.run(circuit_40_qubits, shots=1000)
```

## Migration Checklist

When migrating between backends:

- [ ] Check qubit count limits
- [ ] Verify gate support (native gates differ)
- [ ] Test noise model compatibility
- [ ] Compare performance characteristics
- [ ] Validate result consistency
- [ ] Update configuration files

## Performance Considerations

### Gate Translation Overhead

Some gates may not be native to all backends:

```python
# Cirq: iSWAP is native
# Qiskit: SWAP + phases needed

# Use backend-aware compilation
from proxima.core.compiler import compile_for_backend

compiled = compile_for_backend(circuit, target="qiskit_aer")
```

### Memory Usage

Different backends have different memory profiles:

| Backend | Memory per Qubit (approx) |
|---------|---------------------------|
| Cirq | 16 bytes (statevector) |
| Qiskit Aer | 16 bytes (statevector) |
| QuEST | 16 bytes (distributed) |
| cuQuantum | 8 bytes (GPU single) |

### Batch Execution

```python
# Some backends optimize batch execution
from proxima.backends import get_backend

backend = get_backend("qiskit_aer")
circuits = [circuit1, circuit2, circuit3]

# Single batch (faster)
results = backend.run_batch(circuits, shots=1000)

# vs. Sequential (slower)
results = [backend.run(c, shots=1000) for c in circuits]
```

## Troubleshooting

### Gate Not Supported

```python
# Error: Gate 'iSWAP' not supported by backend
from proxima.core.decomposition import decompose_gate

# Decompose to universal gates
decomposed = decompose_gate("iswap", target_backend="qiskit_aer")
```

### Qubit Limit Exceeded

```python
# Error: Circuit requires 35 qubits, backend supports 32
# Solution 1: Use higher-capacity backend
backend = get_backend("quest")  # Supports 40+ qubits

# Solution 2: Use circuit cutting
from proxima.core.cutting import cut_circuit

subcircuits = cut_circuit(large_circuit, max_qubits=30)
```

### Result Format Differences

```python
# Normalize results across backends
from proxima.core.results import normalize_counts

# Different backends may use different bit ordering
normalized = normalize_counts(
    result.counts,
    source_backend="qiskit_aer",
    target_format="little_endian"
)
```

## See Also

- [Backend Reference](../backends/README.md)
- [Circuit Compilation](../core/compilation.md)
- [Performance Tuning](../performance/tuning.md)
