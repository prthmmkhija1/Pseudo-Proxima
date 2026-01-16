# Proxima Agent Documentation

Welcome to the official Proxima documentation. Proxima is an advanced multi-backend quantum execution orchestrator with LLM-assisted planning, intelligent backend selection, and comprehensive result analysis capabilities.

## What is Proxima?

Proxima is designed to simplify quantum circuit execution across multiple simulation backends while providing:

- **Multi-Backend Orchestration**: Execute quantum circuits on LRET, Cirq, Qiskit Aer, QuEST, qsim, and cuQuantum backends
- **Intelligent Backend Selection**: Automatic recommendation of optimal backends based on circuit characteristics
- **LLM-Assisted Workflows**: Integration with OpenAI, Anthropic, Ollama, and LM Studio for planning and insights
- **Comprehensive Comparison**: Side-by-side execution with identical parameters across multiple backends
- **Rich Export Options**: Generate reports in JSON, CSV, XLSX, and HTML formats
- **Safety & Control**: Full execution control with pause, resume, abort, and rollback capabilities
- **Terminal & Web Interfaces**: Feature-rich CLI, TUI, and REST API

## Key Features

### Backend Support

| Backend | Type | Simulation Modes | Hardware Support |
|---------|------|------------------|------------------|
| **LRET** | Framework | Custom | CPU |
| **Cirq** | Google | StateVector, DensityMatrix | CPU |
| **Qiskit Aer** | IBM | StateVector, DensityMatrix | CPU, GPU |
| **QuEST** | C++ Library | StateVector, DensityMatrix | CPU, GPU, MPI |
| **qsim** | Google | StateVector | CPU (AVX optimized) |
| **cuQuantum** | NVIDIA | StateVector | NVIDIA GPU |

### Execution Control

- **Timer & Transparency**: Real-time progress tracking with stage information
- **Pause/Resume**: Suspend and continue long-running simulations
- **Abort/Rollback**: Safely cancel executions with state recovery
- **Resource Monitoring**: Memory and CPU usage tracking with warnings

### Intelligence Layer

- **Auto-Selection**: Intelligent backend recommendation with explanations
- **LLM Integration**: Natural language planning and result interpretation
- **Insight Generation**: Human-readable analysis of quantum results

## Quick Links

### Getting Started
- [Installation](getting-started/installation.md) - Install Proxima and dependencies
- [Quickstart](getting-started/quickstart.md) - Run your first simulation
- [Configuration](getting-started/configuration.md) - Configure backends and settings

### User Guide
- [Running Simulations](user-guide/running-simulations.md) - Execute quantum circuits
- [Comparing Backends](user-guide/comparing-backends.md) - Multi-backend comparison
- [CLI Reference](user-guide/cli-reference.md) - Complete CLI documentation
- [Using LLM](user-guide/using-llm.md) - LLM integration guide
- [Agent Files](user-guide/agent-files.md) - Working with proxima_agent.md

### Developer Guide
- [Architecture](developer-guide/architecture.md) - System design overview
- [Adding Backends](developer-guide/adding-backends.md) - Create custom backends
- [Backend Development](developer-guide/backend-development.md) - Backend implementation details
- [Testing](developer-guide/testing.md) - Test suite documentation
- [Contributing](developer-guide/contributing.md) - Contribution guidelines

### Backends
- [Backend Selection](backends/backend-selection.md) - How auto-selection works
- [QuEST Usage](backends/quest-usage.md) - QuEST backend guide
- [cuQuantum Usage](backends/cuquantum-usage.md) - GPU acceleration guide
- [qsim Usage](backends/qsim-usage.md) - High-performance CPU simulation

### API Reference
- [API Overview](api-reference/index.md) - REST API documentation
- [Backend API](api-reference/backends/index.md) - Backend adapter interfaces

## System Requirements

### Minimum Requirements
- Python 3.11+
- 4 GB RAM
- 2 CPU cores

### Recommended Requirements
- Python 3.12+
- 16 GB RAM
- 8+ CPU cores
- NVIDIA GPU (for cuQuantum)

### Optional Dependencies
- CUDA Toolkit 12.0+ (for cuQuantum)
- OpenMP (for parallel backends)
- MPI (for distributed QuEST)

## Support

- **Issues**: Open a ticket in the GitHub repository
- **Discussions**: Join the community chat
- **Documentation**: Check the [FAQ](user-guide/advanced-topics.md)
- **Licensing**: See LICENSE in the repository

## Version

Current Version: **0.3.0**

See [Migration Guides](migration/README.md) for upgrade instructions.
