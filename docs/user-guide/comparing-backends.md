# Comparing Backends

This guide explains how to compare quantum simulation backends in Proxima to identify the optimal backend for your workloads.

## Overview

Proxima supports 6 quantum simulation backends, each with different strengths:

| Backend | Type | Best For | GPU Support |
|---------|------|----------|-------------|
| LRET | State Vector | Educational, small circuits | No |
| Cirq | State Vector/Density Matrix | Noise simulation, optimization | No |
| Qiskit | State Vector/Density Matrix | Transpilation, hardware prep | No |
| QuEST | State Vector/Density Matrix | Large circuits, HPC | Yes |
| qsim | State Vector | High-performance, SIMD | No |
| cuQuantum | State Vector | GPU acceleration, very large circuits | Yes |

## Quick Comparison

```bash
# Compare all available backends
proxima compare circuit.json

# Compare specific backends
proxima compare circuit.json --backends cirq,qiskit,quest
```

## Comparison Modes

### Accuracy Comparison

Compare the numerical accuracy of backends:

```bash
proxima compare circuit.json --mode accuracy --reference qiskit
```

Output:
```
Backend Accuracy Comparison (Reference: qiskit)
================================================
Backend    | State Fidelity | Count Fidelity | Max Error
-----------|----------------|----------------|----------
cirq       | 0.999999       | 0.998          | 1.2e-6
quest      | 0.999998       | 0.997          | 2.1e-6
qsim       | 0.999997       | 0.996          | 3.5e-6
cuquantum  | 0.999996       | 0.995          | 4.2e-6
lret       | 0.999995       | 0.994          | 5.8e-6
```

### Performance Comparison

Compare execution speed and resource usage:

```bash
proxima compare circuit.json --mode performance --shots 10000
```

Output:
```
Backend Performance Comparison
==============================
Backend    | Exec Time (ms) | Memory (MB) | Throughput (shots/s)
-----------|----------------|-------------|---------------------
cuquantum  | 45.2          | 2048        | 221,238
qsim       | 78.3          | 512         | 127,713
quest      | 124.5         | 1024        | 80,321
cirq       | 234.7         | 256         | 42,607
qiskit     | 312.1         | 384         | 32,041
lret       | 456.8         | 128         | 21,891
```

### Full Comparison

Run both accuracy and performance comparison:

```bash
proxima compare circuit.json --mode full
```

## Comparison Metrics

### Execution Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `execution_time` | Total execution time | milliseconds |
| `setup_time` | Backend initialization time | milliseconds |
| `gate_time` | Gate application time | milliseconds |
| `measurement_time` | Measurement extraction time | milliseconds |

### Memory Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `peak_memory` | Maximum memory used | bytes |
| `average_memory` | Average memory during execution | bytes |
| `state_size` | Size of state vector | bytes |

### Accuracy Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `state_fidelity` | Fidelity between state vectors | 0.0 - 1.0 |
| `count_fidelity` | Statistical distance between measurement distributions | 0.0 - 1.0 |
| `max_amplitude_error` | Maximum amplitude difference | 0.0+ |
| `mean_amplitude_error` | Mean amplitude difference | 0.0+ |

### Throughput Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| `shots_per_second` | Measurement throughput | shots/s |
| `gates_per_second` | Gate application throughput | gates/s |
| `circuits_per_second` | Circuit execution throughput | circuits/s |

## Comparison Options

### Shots Configuration

```bash
# Compare with specific shot counts
proxima compare circuit.json --shots 1000,10000,100000
```

### Warm-up Runs

```bash
# Exclude warm-up runs from timing
proxima compare circuit.json --warmup-runs 3
```

### Repeated Runs

```bash
# Average across multiple runs
proxima compare circuit.json --runs 10
```

### Reference Backend

```bash
# Use specific backend as reference
proxima compare circuit.json --reference qiskit
```

### Circuit Variations

```bash
# Compare across different qubit counts
proxima compare circuit.json --vary-qubits 5,10,15,20
```

## Reports

### Console Report

Default output to terminal:

```bash
proxima compare circuit.json
```

### JSON Report

```bash
proxima compare circuit.json --output comparison.json --format json
```

```json
{
  "circuit": "circuit.json",
  "timestamp": "2026-01-16T10:30:00Z",
  "backends": ["cirq", "qiskit", "quest"],
  "results": {
    "cirq": {
      "execution_time_ms": 234.7,
      "peak_memory_bytes": 268435456,
      "shots_per_second": 42607,
      "state_fidelity": 0.999999
    },
    "qiskit": {
      "execution_time_ms": 312.1,
      "peak_memory_bytes": 402653184,
      "shots_per_second": 32041,
      "state_fidelity": 1.0
    },
    "quest": {
      "execution_time_ms": 124.5,
      "peak_memory_bytes": 1073741824,
      "shots_per_second": 80321,
      "state_fidelity": 0.999998
    }
  },
  "summary": {
    "fastest": "quest",
    "most_accurate": "qiskit",
    "most_memory_efficient": "cirq"
  }
}
```

### CSV Report

```bash
proxima compare circuit.json --output comparison.csv --format csv
```

### HTML Report

```bash
proxima compare circuit.json --output comparison.html --format html
```

Generates an interactive HTML report with charts and tables.

### Markdown Report

```bash
proxima compare circuit.json --output comparison.md --format markdown
```

## Visualization

### Performance Charts

```bash
# Generate performance charts
proxima compare circuit.json --visualize performance
```

Generates:
- Execution time bar chart
- Memory usage bar chart
- Throughput comparison

### Scaling Charts

```bash
# Compare scaling behavior
proxima compare circuit.json --visualize scaling --vary-qubits 5,10,15,20,25
```

Generates:
- Execution time vs. qubit count
- Memory vs. qubit count
- Throughput vs. qubit count

### Accuracy Charts

```bash
# Generate accuracy heatmap
proxima compare circuit.json --visualize accuracy
```

## Multi-Circuit Comparison

### Batch Comparison

```bash
# Compare across multiple circuits
proxima compare circuits/*.json --output batch_comparison.json
```

### Circuit Categories

```bash
# Compare by circuit type
proxima compare --category "random,structured,variational" --qubits 10
```

## Programmatic Comparison

### Python API

```python
from proxima.core.execution import compare_backends
from proxima.core.circuit import Circuit

# Load circuit
circuit = Circuit.from_file("circuit.json")

# Compare backends
results = compare_backends(
    circuit=circuit,
    backends=["cirq", "qiskit", "quest"],
    shots=1024,
    runs=5,
    reference="qiskit"
)

# Access results
print(f"Fastest: {results.summary.fastest}")
print(f"Most accurate: {results.summary.most_accurate}")

# Export report
results.to_json("comparison.json")
results.to_html("comparison.html")
```

### REST API

```bash
POST /api/v1/compare
{
  "circuit": {...},
  "backends": ["cirq", "qiskit"],
  "shots": 1024,
  "mode": "full"
}
```

Response:
```json
{
  "comparison_id": "cmp_abc123",
  "status": "completed",
  "results": {...}
}
```

## Best Practices

### Fair Comparison

1. **Use warm-up runs**: First execution often includes initialization overhead
   ```bash
   proxima compare circuit.json --warmup-runs 3
   ```

2. **Multiple runs**: Average across runs to reduce variance
   ```bash
   proxima compare circuit.json --runs 10
   ```

3. **Same seed**: Use consistent random seed
   ```bash
   proxima compare circuit.json --seed 42
   ```

4. **Disable other processes**: Ensure consistent system load

### Choosing the Right Backend

**For small circuits (< 15 qubits):**
- Use Cirq or Qiskit for best accuracy and features
- LRET for educational purposes

**For medium circuits (15-25 qubits):**
- Use qsim for CPU-only systems
- Use QuEST for multi-node execution
- Use cuQuantum if GPU available

**For large circuits (> 25 qubits):**
- Use cuQuantum with high-memory GPU
- Use QuEST with distributed computing

**For noise simulation:**
- Use Cirq for comprehensive noise models
- Use Qiskit for hardware noise profiles

**For variational algorithms:**
- Use Qiskit for gradient computation
- Use cuQuantum for fast parameter updates

## Troubleshooting

### Inconsistent Results

```bash
# Increase shot count for statistical convergence
proxima compare circuit.json --shots 100000

# Use state vector comparison instead of counts
proxima compare circuit.json --mode accuracy --compare-state
```

### Large Variance

```bash
# Increase number of runs
proxima compare circuit.json --runs 20

# Exclude outliers
proxima compare circuit.json --exclude-outliers
```

### Memory Issues

```bash
# Limit concurrent backends
proxima compare circuit.json --sequential

# Reduce state output
proxima compare circuit.json --no-state-output
```
