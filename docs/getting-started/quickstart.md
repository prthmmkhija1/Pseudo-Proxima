# Quickstart Guide

Welcome to Proxima! This guide will help you get up and running with quantum circuit simulation in minutes.

## Prerequisites

Before starting, ensure you have:

- Python 3.11 or higher installed
- pip package manager (comes with Python)
- Optional: CUDA-capable GPU for accelerated simulations

## Step 1: Initialize Your Project

Create a new Proxima project with sensible defaults:

```bash
# Initialize with the default template
proxima init --template default

# Or initialize with minimal setup
proxima init --template minimal

# For advanced users - include all backends
proxima init --template full
```

This creates the following structure:

```
my-project/
 proxima.yaml        # Configuration file
 circuits/           # Your quantum circuits
    examples/       # Example circuits
 results/            # Execution results
 exports/            # Exported reports
```

## Step 2: Configure Your Backend

Set up your preferred quantum simulation backend:

### Using the Local LRET Backend (Recommended for Getting Started)

```bash
# Set the default backend
proxima config set backends.default lret

# Configure execution timeout (in seconds)
proxima config set backends.timeout_s 300

# View current configuration
proxima config show
```

### Available Backends

| Backend | Command | Best For |
|---------|---------|----------|
| LRET | `proxima config set backends.default lret` | General purpose, fast |
| Cirq | `proxima config set backends.default cirq` | Google ecosystem |
| Qiskit Aer | `proxima config set backends.default qiskit` | IBM ecosystem |
| QuEST | `proxima config set backends.default quest` | High-performance |
| qsim | `proxima config set backends.default qsim` | CPU-optimized |
| cuQuantum | `proxima config set backends.default cuquantum` | GPU-accelerated |

## Step 3: Run Your First Circuit

### Create a Simple Bell State Circuit

Create a file `circuits/bell.json`:

```json
{
  "name": "Bell State",
  "qubits": 2,
  "gates": [
    {"gate": "H", "target": 0},
    {"gate": "CNOT", "control": 0, "target": 1}
  ],
  "measurements": [0, 1]
}
```

### Execute the Circuit

```bash
# Run with default backend
proxima run circuits/bell.json

# Run with a specific backend
proxima run circuits/bell.json --backend cirq

# Run with custom shots
proxima run circuits/bell.json --shots 10000

# Run with verbose output
proxima run circuits/bell.json --verbose
```

### Expected Output

```

                   Execution Results                      

  Circuit: Bell State                                    
  Backend: lret                                          
  Shots: 1000                                            
  Execution Time: 0.145s                                 

  Measurement Results:                                   
    |00: 498 (49.8%)                                   
    |11: 502 (50.2%)                                   

   Results saved to: results/bell_2025-01-16.json      

```

## Step 4: Compare Multiple Backends

One of Proxima's key features is the ability to compare results across different backends:

```bash
# Compare two backends
proxima compare circuits/bell.json --backends lret,cirq

# Compare all available backends
proxima compare circuits/bell.json --backends all

# Compare with detailed statistics
proxima compare circuits/bell.json --backends lret,cirq,qiskit --detailed

# Save comparison report
proxima compare circuits/bell.json --backends lret,cirq --output comparison.json
```

### Comparison Output

```

              Multi-Backend Comparison                    

  Circuit: Bell State                                    
  Backends: lret, cirq, qiskit                           

 Backend   Time(ms)  Fidelity  Distribution          

 lret      145.3     0.998     |00:49.8% |11:50.2%
 cirq      152.1     0.997     |00:50.1% |11:49.9%
 qiskit    168.4     0.996     |00:49.5% |11:50.5%

   All backends agree within statistical error         

```

## Step 5: View and Export Results

### View Previous Results

```bash
# List all results
proxima results list

# Show specific result details
proxima results show result-id-12345

# Filter results by date
proxima results list --since 2025-01-01

# Filter by backend
proxima results list --backend cirq
```

### Export Results

```bash
# Export to JSON
proxima export --format json --output reports/results.json

# Export to CSV
proxima export --format csv --output reports/results.csv

# Export to Excel with charts
proxima export --format xlsx --output reports/results.xlsx

# Export with LLM-generated insights
proxima export --format json --output reports/results.json --with-insights
```

## Step 6: Using the Interactive TUI

For a richer experience, launch the Terminal User Interface:

```bash
# Launch the TUI
proxima tui

# Or with a specific session
proxima tui --session my-experiment
```

### TUI Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `r` | Run circuit |
| `c` | Compare backends |
| `h` | View history |
| `e` | Export results |
| `q` | Quit |

## Common Commands Reference

```bash
# Help
proxima --help
proxima run --help

# Configuration
proxima config show
proxima config set <key> <value>
proxima config get <key>

# Backend management
proxima backends list
proxima backends info <backend>
proxima backends test <backend>

# Session management
proxima session list
proxima session load <name>
proxima session save <name>

# LLM integration
proxima llm plan circuits/complex.json
proxima llm summarize results/latest.json
```

## Next Steps

Now that you have Proxima running:

1. **Explore Example Circuits**: Check the `circuits/examples/` directory
2. **Configure LLM Integration**: See [Using LLM](using-llm.md)
3. **Create Agent Files**: Learn about [Agent Files](agent-files.md)
4. **Advanced Configuration**: Read [Configuration Guide](configuration.md)
5. **Add Custom Backends**: See [Adding Backends](../developer-guide/adding-backends.md)

## Troubleshooting

### Common Issues

**Backend not found:**
```bash
# Check available backends
proxima backends list

# Install missing backend dependencies
pip install proxima[qiskit]  # For Qiskit support
pip install proxima[all]     # For all backends
```

**Memory errors:**
```bash
# Reduce shot count
proxima run circuit.json --shots 1000

# Use a memory-efficient backend
proxima run circuit.json --backend lret
```

**Configuration issues:**
```bash
# Reset to defaults
proxima config reset

# Validate configuration
proxima config validate
```

For more help, see our [Troubleshooting Guide](advanced-topics.md#troubleshooting) or [open an issue](https://github.com/proxima-project/proxima/issues).
