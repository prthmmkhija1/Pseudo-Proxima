# CLI Reference

This document provides a comprehensive reference for all Proxima CLI commands.

## Global Options

These options are available for all commands:

```bash
proxima [OPTIONS] COMMAND [ARGS]...
```

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-V` | Show version and exit |
| `--help` | `-h` | Show help message and exit |
| `--verbose` | `-v` | Enable verbose output |
| `--quiet` | `-q` | Suppress non-essential output |
| `--config` | `-c` | Path to config file |
| `--no-color` | | Disable colored output |
| `--json` | | Output in JSON format |

## Commands Overview

| Command | Description |
|---------|-------------|
| `run` | Execute a quantum circuit |
| `compare` | Compare execution across backends |
| `backends` | Manage quantum backends |
| `config` | View and modify configuration |
| `results` | View and manage execution results |
| `export` | Export results to various formats |
| `session` | Manage sessions |
| `llm` | LLM integration commands |
| `tui` | Launch terminal user interface |
| `agent` | Agent file operations |
| `init` | Initialize a new project |

---

## `proxima run`

Execute a quantum circuit on a specified backend.

### Usage

```bash
proxima run [OPTIONS] CIRCUIT
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `CIRCUIT` | Yes | Path to circuit file (JSON/QASM) |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backend` | `-b` | `auto` | Backend to use (lret, cirq, qiskit, quest, qsim, cuquantum) |
| `--shots` | `-s` | `1000` | Number of measurement shots |
| `--timeout` | `-t` | `300` | Execution timeout in seconds |
| `--output` | `-o` | auto | Output file path for results |
| `--no-save` | | | Don't save results to history |
| `--dry-run` | | | Show execution plan without running |
| `--watch` | `-w` | | Watch file for changes and re-run |
| `--parameters` | `-p` | | Parameter values as JSON |

### Examples

```bash
# Basic execution
proxima run circuits/bell.json

# Use specific backend with more shots
proxima run circuits/vqe.json --backend cirq --shots 10000

# Execute with custom parameters
proxima run circuits/parametric.json -p '{"theta": 0.5, "phi": 1.2}'

# Dry run to see execution plan
proxima run circuits/complex.json --dry-run

# Watch mode for iterative development
proxima run circuits/test.json --watch
```

---

## `proxima compare`

Compare circuit execution across multiple backends.

### Usage

```bash
proxima compare [OPTIONS] CIRCUIT
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `CIRCUIT` | Yes | Path to circuit file |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--backends` | `-b` | `all` | Comma-separated list of backends |
| `--shots` | `-s` | `1000` | Number of shots per backend |
| `--output` | `-o` | auto | Output file for comparison report |
| `--detailed` | `-d` | | Include detailed statistics |
| `--parallel` | | | Run backends in parallel |
| `--timeout` | `-t` | `600` | Total timeout in seconds |
| `--format` | `-f` | `table` | Output format (table, json, csv) |

### Examples

```bash
# Compare all available backends
proxima compare circuits/bell.json

# Compare specific backends
proxima compare circuits/bell.json --backends lret,cirq,qiskit

# Detailed comparison with parallel execution
proxima compare circuits/complex.json -b all --detailed --parallel

# Export comparison as JSON
proxima compare circuits/bell.json -f json -o comparison.json
```

---

## `proxima backends`

Manage and inspect quantum backends.

### Subcommands

#### `proxima backends list`

List all available backends.

```bash
proxima backends list [OPTIONS]

Options:
  --available    Show only available backends
  --all          Show all backends including unavailable
  --format       Output format (table, json)
```

#### `proxima backends info`

Show detailed information about a backend.

```bash
proxima backends info BACKEND

Arguments:
  BACKEND    Backend name (lret, cirq, qiskit, quest, qsim, cuquantum)
```

#### `proxima backends test`

Test a backend's functionality.

```bash
proxima backends test BACKEND [OPTIONS]

Options:
  --circuit      Path to test circuit
  --shots        Number of test shots
  --verbose      Show detailed test output
```

#### `proxima backends benchmark`

Run performance benchmarks.

```bash
proxima backends benchmark [OPTIONS]

Options:
  --backends     Backends to benchmark
  --qubits       Qubit range (e.g., "2-10")
  --output       Output file for results
```

### Examples

```bash
# List all backends with status
proxima backends list

# Get detailed info about QuEST
proxima backends info quest

# Test Cirq backend
proxima backends test cirq --verbose

# Benchmark all GPU backends
proxima backends benchmark --backends cuquantum,quest
```

---

## `proxima config`

View and modify configuration.

### Subcommands

#### `proxima config show`

Display current configuration.

```bash
proxima config show [SECTION]

Arguments:
  SECTION    Optional section (backends, llm, resources, etc.)
```

#### `proxima config get`

Get a specific configuration value.

```bash
proxima config get KEY
```

#### `proxima config set`

Set a configuration value.

```bash
proxima config set KEY VALUE
```

#### `proxima config reset`

Reset configuration to defaults.

```bash
proxima config reset [--section SECTION]
```

#### `proxima config validate`

Validate current configuration.

```bash
proxima config validate
```

#### `proxima config edit`

Open configuration in editor.

```bash
proxima config edit
```

### Examples

```bash
# Show all configuration
proxima config show

# Show backend configuration
proxima config show backends

# Get specific value
proxima config get llm.provider

# Set values
proxima config set backends.default cirq
proxima config set llm.model gpt-4
proxima config set resources.memory_warn_threshold_mb 8192

# Reset to defaults
proxima config reset

# Validate configuration
proxima config validate
```

---

## `proxima results`

View and manage execution results.

### Subcommands

#### `proxima results list`

List execution results.

```bash
proxima results list [OPTIONS]

Options:
  --limit        Maximum results to show (default: 20)
  --since        Show results since date
  --until        Show results until date
  --backend      Filter by backend
  --circuit      Filter by circuit name
  --format       Output format (table, json)
```

#### `proxima results show`

Show details of a specific result.

```bash
proxima results show RESULT_ID [OPTIONS]

Options:
  --format       Output format (table, json, full)
```

#### `proxima results delete`

Delete a result.

```bash
proxima results delete RESULT_ID [--force]
```

#### `proxima results clear`

Clear all results.

```bash
proxima results clear [--before DATE] [--force]
```

### Examples

```bash
# List recent results
proxima results list

# Filter by backend
proxima results list --backend cirq --limit 10

# Show specific result
proxima results show res-abc123

# Delete old results
proxima results clear --before 2025-01-01
```

---

## `proxima export`

Export results to various formats.

### Usage

```bash
proxima export [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--format` | `-f` | `json` | Export format (json, csv, xlsx) |
| `--output` | `-o` | Required | Output file path |
| `--results` | `-r` | all | Result IDs to export (comma-separated) |
| `--since` | | | Export results since date |
| `--until` | | | Export results until date |
| `--with-insights` | | | Include LLM-generated insights |
| `--template` | | | Use export template |

### Examples

```bash
# Export all results to JSON
proxima export -f json -o results.json

# Export to Excel with insights
proxima export -f xlsx -o report.xlsx --with-insights

# Export specific results
proxima export -f csv -o subset.csv -r res-001,res-002,res-003

# Export recent results
proxima export -f json -o recent.json --since 2025-01-15
```

---

## `proxima session`

Manage execution sessions.

### Subcommands

#### `proxima session list`

List all sessions.

```bash
proxima session list
```

#### `proxima session new`

Create a new session.

```bash
proxima session new [NAME]
```

#### `proxima session load`

Load a saved session.

```bash
proxima session load NAME
```

#### `proxima session save`

Save current session.

```bash
proxima session save [NAME]
```

#### `proxima session delete`

Delete a session.

```bash
proxima session delete NAME [--force]
```

### Examples

```bash
# Create new session
proxima session new my-experiment

# Save current session
proxima session save

# Load previous session
proxima session load my-experiment

# List all sessions
proxima session list
```

---

## `proxima llm`

LLM integration commands.

### Subcommands

#### `proxima llm plan`

Generate execution plan using LLM.

```bash
proxima llm plan CIRCUIT [OPTIONS]

Options:
  --max-time         Maximum execution time
  --prefer-gpu       Prefer GPU backends
  --interactive      Interactive planning mode
```

#### `proxima llm summarize`

Summarize results using LLM.

```bash
proxima llm summarize RESULT_FILE [OPTIONS]

Options:
  --focus            Focus area (comparison, statistics, insights)
  --output           Output file for summary
```

#### `proxima llm test`

Test LLM connection.

```bash
proxima llm test
```

#### `proxima llm usage`

Show LLM usage statistics.

```bash
proxima llm usage [--show-cost]
```

### Examples

```bash
# Generate execution plan
proxima llm plan circuits/complex.json

# Summarize results
proxima llm summarize results/latest.json

# Test LLM connection
proxima llm test
```

---

## `proxima tui`

Launch the terminal user interface.

### Usage

```bash
proxima tui [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--session` | `-s` | Load specific session |
| `--theme` | | UI theme (dark, light) |
| `--no-mouse` | | Disable mouse support |

### Examples

```bash
# Launch TUI
proxima tui

# Launch with specific session
proxima tui --session my-experiment
```

---

## `proxima agent`

Agent file operations.

### Subcommands

#### `proxima agent run`

Execute an agent file.

```bash
proxima agent run AGENT_FILE [OPTIONS]

Options:
  --dry-run         Show plan without executing
  --interactive     Confirm each step
```

#### `proxima agent validate`

Validate an agent file.

```bash
proxima agent validate AGENT_FILE
```

#### `proxima agent create`

Create a new agent file from template.

```bash
proxima agent create OUTPUT_FILE [--template TEMPLATE]
```

### Examples

```bash
# Run agent file
proxima agent run my_experiment.agent.md

# Validate agent file
proxima agent validate my_experiment.agent.md

# Create new agent file
proxima agent create new_experiment.agent.md
```

---

## `proxima init`

Initialize a new Proxima project.

### Usage

```bash
proxima init [OPTIONS] [DIRECTORY]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `DIRECTORY` | `.` | Project directory |

### Options

| Option | Description |
|--------|-------------|
| `--template` | Project template (default, minimal, full) |
| `--name` | Project name |
| `--no-examples` | Don't create example files |

### Examples

```bash
# Initialize in current directory
proxima init

# Initialize with full template
proxima init --template full

# Initialize in new directory
proxima init my-project --template default
```

---

## Environment Variables

Proxima respects the following environment variables:

| Variable | Description |
|----------|-------------|
| `PROXIMA_CONFIG` | Path to configuration file |
| `PROXIMA_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PROXIMA_NO_COLOR` | Disable colored output |
| `OPENAI_API_KEY` | OpenAI API key for LLM |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM |
| `OLLAMA_HOST` | Ollama server URL |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Configuration error |
| `3` | Backend error |
| `4` | Timeout error |
| `5` | User abort |

---

## See Also

- [Quickstart Guide](../getting-started/quickstart.md)
- [Configuration](../getting-started/configuration.md)
- [Using LLM](using-llm.md)
