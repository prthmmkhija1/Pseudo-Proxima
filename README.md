# Proxima: Intelligent Quantum Simulation Orchestration Framework

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proxima is an intelligent quantum simulation orchestration framework that provides a unified interface for running quantum simulations across multiple backends with advanced features like automatic backend selection, resource monitoring, and intelligent result interpretation.

## Features

- **Multi-Backend Support**: LRET, Cirq (DensityMatrix + StateVector), Qiskit Aer (DensityMatrix + StateVector)
- **Intelligent Backend Selection**: Automatic selection with explanations
- **Execution Control**: Start, Abort, Pause, Resume, Rollback
- **Resource Awareness**: Memory and CPU monitoring with fail-safe mechanisms
- **Explicit Consent**: User confirmation for critical operations
- **LLM Integration**: Support for local and remote AI models
- **Result Interpretation**: Human-readable insights and analytics
- **Multi-Backend Comparison**: Run identical simulations across backends
- **Execution Transparency**: Real-time progress and timing display

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/proxima-project/proxima.git
cd proxima

# Install dependencies
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"
```

### Dependencies Only

```bash
pip install -e .
```

### Development Setup

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Initialize configuration
proxima init

# Show version
proxima version

# List available backends
proxima backends list

# Run a simulation (to be implemented)
proxima run --backend cirq simulation.py
```

## Project Structure

```
proxima/
├── src/proxima/          # Main package
│   ├── cli/              # Command-line interface
│   ├── core/             # Core domain logic
│   ├── backends/         # Backend adapters
│   ├── intelligence/     # AI/ML components
│   ├── resources/        # Resource management
│   ├── data/             # Data handling
│   ├── config/           # Configuration
│   └── utils/            # Utilities
├── tests/                # Test suites
├── configs/              # Configuration files
└── docs/                 # Documentation
```

## Configuration

Proxima supports multiple configuration sources (in priority order):

1. Command-line arguments
2. Environment variables (PROXIMA\_\*)
3. User config file (~/.proxima/config.yaml)
4. Project config file (./proxima.yaml)
5. Default values

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

## Architecture

Proxima follows a layered modular architecture:

1. **Presentation Layer**: CLI, TUI (future), Web API (future)
2. **Orchestration Layer**: Planner, Executor, State Manager
3. **Intelligence Layer**: LLM Router, Backend Selector, Insight Engine
4. **Resources & Safety Layer**: Memory Monitor, Consent Manager, Execution Control
5. **Backend Abstraction Layer**: Unified adapter interface
6. **Data & Output Layer**: Result storage, comparison, export

## Roadmap

- **Phase 1** (Weeks 1-4): Foundation & Core Infrastructure
- **Phase 2** (Weeks 5-9): Backend Integration
- **Phase 3** (Weeks 10-14): Intelligence Features
- **Phase 4** (Weeks 15-18): Safety & Resource Management
- **Phase 5** (Weeks 19-23): Advanced Features
- **Phase 6** (Weeks 24-27): Production Ready

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

## Credits

Architectural inspiration from:

- [OpenCode AI](https://github.com/opencode-ai/opencode)
- [Crush (Charmbracelet)](https://github.com/charmbracelet/crush)

Proxima is an independent implementation, not a fork or derivative.
