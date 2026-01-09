# Architecture

## Components
- Core: planning, execution orchestration, consent checks
- Backends: adapters for local, Qiskit, IBM, and mocks
- LLM: providers for planning and summarization
- Export/Reporting: JSON/CSV/HTML exporters, comparison metrics
- Interfaces: CLI and Textual TUI

## Data flow
1. Parse circuit and options
2. Plan execution (strategy, resources, consent)
3. Execute across backends
4. Collect results and metrics
5. Generate insights and exports

## Extensibility
- Backends implement a shared adapter interface
- LLM providers are swappable via config
- Exporters are pluggable per format
