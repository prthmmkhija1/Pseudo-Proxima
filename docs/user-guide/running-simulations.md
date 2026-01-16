# Running Simulations

This guide covers how to execute quantum circuits using Proxima across different backends.

## Quick Start

```bash
# Run a circuit with auto-selected backend
proxima run circuit.json

# Run with specific backend
proxima run circuit.json --backend cirq

# Run with specific shots
proxima run circuit.json --backend qiskit --shots 1024
```

## Circuit Input Formats

Proxima accepts circuits in multiple formats:

### JSON Format

```json
{
  "name": "bell_state",
  "qubits": 2,
  "gates": [
    {"type": "H", "targets": [0]},
    {"type": "CNOT", "control": 0, "target": 1}
  ],
  "measurements": [0, 1]
}
```

### OpenQASM 2.0

```bash
proxima run circuit.qasm --format qasm
```

### Cirq JSON

```bash
proxima run cirq_circuit.json --format cirq
```

### Qiskit Pickle

```bash
proxima run qiskit_circuit.pkl --format qiskit
```

## Backend Selection

### Auto Selection

When no backend is specified, Proxima automatically selects the optimal backend:

```bash
proxima run circuit.json
# Output: Selected backend: cirq (Reason: Best match for 5-qubit circuit with noise model)
```

### Manual Selection

```bash
# Use specific backend
proxima run circuit.json --backend lret
proxima run circuit.json --backend cirq
proxima run circuit.json --backend qiskit
proxima run circuit.json --backend quest
proxima run circuit.json --backend qsim
proxima run circuit.json --backend cuquantum
```

### Backend Priority

Set preferred backends for auto-selection:

```bash
proxima run circuit.json --prefer quest,cirq,qiskit
```

## Execution Options

### Basic Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--backend` | `-b` | Specify backend | auto |
| `--shots` | `-s` | Number of shots | 1024 |
| `--seed` | | Random seed | None |
| `--timeout-ms` | `-t` | Execution timeout | 300000 |
| `--dry-run` | | Validate without executing | false |

### Simulator Type

```bash
# State vector simulation
proxima run circuit.json --simulator statevector

# Density matrix simulation
proxima run circuit.json --simulator density_matrix
```

### Noise Models

```bash
# Apply depolarizing noise
proxima run circuit.json --noise depolarizing --noise-strength 0.01

# Apply custom noise model
proxima run circuit.json --noise-file noise_model.json
```

### Resource Limits

```bash
# Set memory limit
proxima run circuit.json --max-memory 8GB

# Set qubit limit
proxima run circuit.json --max-qubits 25
```

## Execution Modes

### Single Execution

```bash
proxima run circuit.json --backend cirq
```

### Batch Execution

```bash
# Run multiple circuits
proxima run circuits/*.json --backend qiskit

# With parallel execution
proxima run circuits/*.json --parallel --workers 4
```

### Interactive Mode

```bash
# Start interactive session
proxima run --interactive

# In session:
> load circuit.json
> set backend cirq
> run
> show results
```

## Execution Control

### Pause and Resume

```bash
# Start long-running execution
proxima run large_circuit.json --background

# Pause execution
proxima control pause

# Resume execution
proxima control resume

# Check status
proxima control status
```

### Abort and Rollback

```bash
# Abort current execution
proxima control abort

# Rollback to previous state
proxima control rollback
```

### Checkpoints

```bash
# Enable checkpointing
proxima run circuit.json --checkpoint --checkpoint-interval 60

# Resume from checkpoint
proxima run circuit.json --resume-checkpoint
```

## Resource Monitoring

### Memory Monitoring

```bash
# Enable memory monitoring
proxima run circuit.json --monitor-memory

# Set warning threshold
proxima run circuit.json --memory-warn 8GB
```

### Progress Tracking

```bash
# Show detailed progress
proxima run circuit.json --progress

# Output:
# [] 50% | Stage: Gate Application | Elapsed: 2.3s
```

## Consent and Safety

### Explicit Consent

When `require_consent` is enabled:

```bash
proxima run circuit.json
# Prompt: This will execute a 20-qubit circuit requiring ~16GB memory. Continue? [y/N]
```

### Force Execution

```bash
# Skip consent prompts (use with caution)
proxima run circuit.json --force
```

### Resource Warnings

```bash
proxima run large_circuit.json
# Warning: Estimated memory usage (12GB) exceeds warning threshold (8GB)
# Warning: Consider using a GPU-accelerated backend
```

## Results

### View Results

```bash
# Show results summary
proxima results show

# Show specific result
proxima results show --id abc123

# Show as JSON
proxima results show --format json
```

### Result Structure

```json
{
  "id": "abc123",
  "backend": "cirq",
  "simulator_type": "state_vector",
  "execution_time_ms": 234.5,
  "qubit_count": 5,
  "shot_count": 1024,
  "result_type": "counts",
  "data": {
    "counts": {
      "00000": 512,
      "11111": 512
    }
  },
  "metadata": {
    "timestamp": "2026-01-16T10:30:00Z",
    "version": "0.3.0"
  }
}
```

### State Vector Results

```bash
proxima run circuit.json --simulator statevector --output-format amplitudes
```

### Density Matrix Results

```bash
proxima run circuit.json --simulator density_matrix --output-format matrix
```

## Export Results

```bash
# Export to JSON
proxima export --format json --output results.json

# Export to CSV
proxima export --format csv --output results.csv

# Export to Excel
proxima export --format xlsx --output results.xlsx

# Export with insights
proxima export --format html --include-insights
```

## Backend-Specific Options

### LRET Options

```bash
proxima run circuit.json --backend lret --normalize-results
```

### Cirq Options

```bash
proxima run circuit.json --backend cirq \
  --simulator density_matrix \
  --noise depolarizing \
  --optimization-level 2
```

### Qiskit Options

```bash
proxima run circuit.json --backend qiskit \
  --transpile \
  --optimization-level 3 \
  --coupling-map linear
```

### QuEST Options

```bash
proxima run circuit.json --backend quest \
  --precision double \
  --gpu \
  --openmp-threads 8
```

### qsim Options

```bash
proxima run circuit.json --backend qsim \
  --avx avx512 \
  --gate-fusion aggressive \
  --threads 16
```

### cuQuantum Options

```bash
proxima run circuit.json --backend cuquantum \
  --device 0 \
  --memory-limit 8GB
```

## Examples

### Bell State

```bash
proxima run examples/bell.json --shots 1000
# Expected output: ~50% |00, ~50% |11
```

### GHZ State

```bash
proxima run examples/ghz.json --qubits 5 --shots 1000
# Expected output: ~50% |00000, ~50% |11111
```

### Quantum Fourier Transform

```bash
proxima run examples/qft.json --backend quest --qubits 10
```

### Variational Circuit

```bash
proxima run examples/vqe.json --backend qiskit --parameters params.json
```

## Troubleshooting

### Common Issues

**Backend not available:**
```bash
# Check available backends
proxima backends list

# Test backend connectivity
proxima backends test cirq
```

**Memory exceeded:**
```bash
# Reduce qubit count or use GPU backend
proxima run circuit.json --backend cuquantum
```

**Timeout exceeded:**
```bash
# Increase timeout
proxima run circuit.json --timeout-ms 600000
```

**Invalid circuit:**
```bash
# Validate circuit before execution
proxima run circuit.json --dry-run --verbose
```
