# CLI Reference

Complete command-line interface documentation for Proxima.

## Overview

Proxima provides a comprehensive CLI for quantum circuit simulation, backend comparison, and workflow management.

```bash
proxima [OPTIONS] COMMAND [ARGS]
```

---

## Global Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to configuration YAML | `proxima.yaml` |
| `--backend` | `-b` | Select simulation backend | `auto` |
| `--output` | `-o` | Output format (json, table, yaml) | `table` |
| `--verbose` | `-v` | Increase verbosity (repeat for more) | 0 |
| `--quiet` | `-q` | Decrease verbosity | - |
| `--dry-run` | | Show plan without executing | - |
| `--force` | `-f` | Skip confirmation prompts | - |
| `--no-color` | | Disable colored output | - |
| `--help` | `-h` | Show help message | - |
| `--version` | | Show version information | - |

---

## Commands

### proxima init

Initialize Proxima configuration in the current directory.

```bash
proxima init [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--template` | Configuration template (minimal, standard, full) | `standard` |
| `--backends` | Backends to enable (comma-separated) | `auto` |
| `--llm` | Enable LLM features | `false` |
| `--force` | Overwrite existing config | `false` |

**Examples:**

```bash
# Basic initialization
proxima init

# Initialize with specific backends
proxima init --backends cirq,qiskit_aer,lret

# Full configuration with LLM support
proxima init --template full --llm
```

---

### proxima run

Execute a quantum circuit simulation.

```bash
proxima run [OPTIONS] [CIRCUIT]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `CIRCUIT` | Circuit file path, QASM string, or circuit name |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | Backend to use | `auto` |
| `--shots` | Number of measurement shots | `1000` |
| `--qubits` | Number of qubits | from circuit |
| `--seed` | Random seed for reproducibility | - |
| `--timeout` | Execution timeout (seconds) | `300` |
| `--save` | Save results to file | - |
| `--format` | Output format (json, csv, yaml) | `json` |
| `--statevector` | Return statevector instead of counts | `false` |
| `--density-matrix` | Return density matrix | `false` |
| `--noise` | Apply noise model | - |

**Examples:**

```bash
# Run a QASM file
proxima run circuit.qasm --backend cirq --shots 2000

# Run a built-in circuit
proxima run bell --shots 1000

# Run with noise model
proxima run ghz.qasm --noise depolarizing --backend qiskit_aer

# Save results
proxima run circuit.qasm --save results.json --format json
```

---

### proxima compare

Compare circuit execution across multiple backends.

```bash
proxima compare [OPTIONS] [CIRCUIT]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `CIRCUIT` | Circuit file or name to compare |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--backends` | Comma-separated backend list | all available |
| `--shots` | Number of shots per backend | `1000` |
| `--metrics` | Metrics to calculate | tvd,fidelity |
| `--reference` | Reference backend for comparison | first listed |
| `--parallel` | Run backends in parallel | `true` |
| `--timeout` | Per-backend timeout (seconds) | `60` |
| `--save` | Save comparison results | - |
| `--format` | Output format | `table` |
| `--llm` | Generate LLM insights | `false` |

**Examples:**

```bash
# Compare across all backends
proxima compare bell.qasm

# Compare specific backends
proxima compare ghz.qasm --backends cirq,qiskit_aer,lret

# With LLM analysis
proxima compare circuit.qasm --llm --backends cirq,qiskit_aer

# Save comparison report
proxima compare circuit.qasm --save report.json --format json
```

**Output:**

```
╭─────────────────────────────────────────────────────────────────╮
│                    Backend Comparison Results                    │
├─────────────┬───────────┬───────────┬───────────┬───────────────┤
│ Backend     │ Time (ms) │ 00        │ 11        │ TVD vs Cirq   │
├─────────────┼───────────┼───────────┼───────────┼───────────────┤
│ cirq        │ 12.3      │ 0.498     │ 0.502     │ -             │
│ qiskit_aer  │ 15.7      │ 0.495     │ 0.505     │ 0.007         │
│ lret        │ 8.2       │ 0.501     │ 0.499     │ 0.004         │
╰─────────────┴───────────┴───────────┴───────────┴───────────────╯
```

---

### proxima backends

Manage and inspect quantum backends.

```bash
proxima backends [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### backends list

List available backends.

```bash
proxima backends list [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--all` | Include unavailable backends |
| `--format` | Output format (table, json, yaml) |

**Output:**

```
╭──────────────────────────────────────────────────────────────────╮
│                       Available Backends                          │
├─────────────┬───────────┬──────────┬──────────┬─────────────────┤
│ Backend     │ Status    │ GPU      │ Max Qubits │ Version        │
├─────────────┼───────────┼──────────┼──────────┼─────────────────┤
│ cirq        │ ✓ Ready   │ No       │ 25        │ 1.3.0          │
│ qiskit_aer  │ ✓ Ready   │ No       │ 30        │ 0.14.0         │
│ lret        │ ✓ Ready   │ No       │ 20        │ 1.0.0          │
│ cuquantum   │ ✗ N/A     │ Yes      │ 32+       │ not installed  │
│ quest       │ ✗ N/A     │ No       │ 40+       │ not installed  │
╰─────────────┴───────────┴──────────┴──────────┴─────────────────╯
```

#### backends info

Get detailed information about a backend.

```bash
proxima backends info [BACKEND]
```

**Output:**

```
╭────────────────────────────────────────────────────────────────╮
│                    Backend: cirq                                │
├────────────────────┬───────────────────────────────────────────┤
│ Property           │ Value                                      │
├────────────────────┼───────────────────────────────────────────┤
│ Version            │ 1.3.0                                      │
│ Status             │ Ready                                      │
│ Max Qubits         │ 25 (estimated)                            │
│ GPU Support        │ No                                         │
│ Density Matrix     │ Yes                                        │
│ Statevector        │ Yes                                        │
│ Noise Models       │ depolarizing, amplitude_damping, ...      │
│ Native Gates       │ H, X, Y, Z, CNOT, CZ, RX, RY, RZ, ...     │
╰────────────────────┴───────────────────────────────────────────╯
```

#### backends test

Test backend functionality.

```bash
proxima backends test [BACKEND] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--quick` | Quick sanity test only |
| `--full` | Run comprehensive test suite |

---

### proxima history

View and manage execution history.

```bash
proxima history [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### history list

List recent executions.

```bash
proxima history list [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--limit` | Maximum entries to show | `20` |
| `--since` | Filter by date (e.g., "1 week") | - |
| `--backend` | Filter by backend | - |
| `--status` | Filter by status | - |
| `--format` | Output format | `table` |

#### history show

Show details of a specific execution.

```bash
proxima history show [EXECUTION_ID]
```

#### history export

Export execution history.

```bash
proxima history export [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--format` | Export format (json, csv, xlsx) |
| `--output` | Output file path |
| `--since` | Filter by date |

#### history clear

Clear execution history.

```bash
proxima history clear [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--before` | Clear entries before date |
| `--all` | Clear all history |
| `--force` | Skip confirmation |

---

### proxima session

Manage execution sessions with checkpoints.

```bash
proxima session [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### session list

List active and recent sessions.

```bash
proxima session list [OPTIONS]
```

#### session resume

Resume a suspended session.

```bash
proxima session resume [SESSION_ID]
```

#### session checkpoint

Create a checkpoint in the current session.

```bash
proxima session checkpoint [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name` | Checkpoint name |
| `--description` | Checkpoint description |

#### session restore

Restore from a checkpoint.

```bash
proxima session restore [CHECKPOINT_ID]
```

---

### proxima agent

Work with agent files for automated workflows.

```bash
proxima agent [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### agent run

Execute an agent file.

```bash
proxima agent run [AGENT_FILE] [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Show plan without executing | `false` |
| `--continue-on-error` | Continue after task failures | `false` |
| `--parallel` | Execute independent tasks in parallel | `true` |
| `--timeout` | Overall timeout (seconds) | - |

**Example Agent File (workflow.agent):**

```yaml
name: Bell State Analysis
version: "1.0"

tasks:
  - id: simulate
    type: simulate
    circuit: bell
    backends: [cirq, qiskit_aer, lret]
    shots: 10000
    
  - id: compare
    type: compare
    depends_on: [simulate]
    metrics: [tvd, fidelity, entropy]
    
  - id: report
    type: export
    depends_on: [compare]
    format: markdown
    output: report.md
```

```bash
# Run agent file
proxima agent run workflow.agent

# Dry run to see plan
proxima agent run workflow.agent --dry-run
```

#### agent validate

Validate an agent file.

```bash
proxima agent validate [AGENT_FILE]
```

#### agent create

Create a new agent file from template.

```bash
proxima agent create [OUTPUT_FILE] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--template` | Template type (simple, comparison, benchmark) |

---

### proxima config

Configuration management.

```bash
proxima config [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### config show

Show current configuration.

```bash
proxima config show [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--format` | Output format (yaml, json) |
| `--section` | Show specific section only |

#### config set

Set a configuration value.

```bash
proxima config set [KEY] [VALUE]
```

**Examples:**

```bash
proxima config set backends.default cirq
proxima config set shots 2000
proxima config set llm.provider openai
```

#### config get

Get a configuration value.

```bash
proxima config get [KEY]
```

#### config validate

Validate configuration file.

```bash
proxima config validate [CONFIG_FILE]
```

---

### proxima export

Export results in various formats.

```bash
proxima export [EXECUTION_ID] [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--format` | Export format | `json` |
| `--output` | Output file path | stdout |
| `--include-metadata` | Include execution metadata | `true` |
| `--include-circuit` | Include circuit definition | `false` |

**Supported Formats:**

| Format | Extension | Description |
|--------|-----------|-------------|
| JSON | `.json` | Structured JSON with full data |
| CSV | `.csv` | Tabular data (counts only) |
| YAML | `.yaml` | YAML format |
| Markdown | `.md` | Human-readable report |
| HTML | `.html` | Interactive HTML report |
| Excel | `.xlsx` | Excel workbook with sheets |
| LaTeX | `.tex` | LaTeX table format |
| QASM | `.qasm` | OpenQASM circuit export |

**Examples:**

```bash
# Export to JSON
proxima export abc123 --format json --output results.json

# Export to Markdown report
proxima export abc123 --format markdown --output report.md

# Export to Excel with metadata
proxima export abc123 --format xlsx --output results.xlsx --include-metadata
```

---

### proxima benchmark

Run performance benchmarks.

```bash
proxima benchmark [SUITE] [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `SUITE` | Benchmark suite (quick, standard, comprehensive, gpu) |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--backends` | Backends to benchmark | all available |
| `--circuits` | Specific circuits to test | from suite |
| `--iterations` | Iterations per test | `5` |
| `--warmup` | Warmup iterations | `2` |
| `--output` | Save results to file | - |
| `--compare-baseline` | Compare to baseline | - |

**Examples:**

```bash
# Quick benchmark
proxima benchmark quick

# Standard benchmark on specific backends
proxima benchmark standard --backends cirq,qiskit_aer

# Comprehensive with baseline comparison
proxima benchmark comprehensive --compare-baseline baseline.json
```

---

### proxima ui

Launch the Terminal User Interface.

```bash
proxima ui [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--theme` | UI theme (dark, light, high-contrast) | `dark` |
| `--session` | Resume specific session | - |

**Features:**

- Interactive circuit builder
- Real-time execution monitoring
- Backend comparison dashboard
- Result visualization
- History browser

---

### proxima llm

LLM-related commands.

```bash
proxima llm [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### llm providers

List available LLM providers.

```bash
proxima llm providers [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--all` | Include unavailable providers |
| `--check` | Check API connectivity |

#### llm consent

Manage LLM usage consent.

```bash
proxima llm consent [grant|revoke|status]
```

#### llm configure

Configure LLM provider settings.

```bash
proxima llm configure [PROVIDER]
```

#### llm explain

Get LLM explanation of results.

```bash
proxima llm explain [EXECUTION_ID] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--level` | Detail level (basic, intermediate, advanced, expert) |
| `--provider` | Specific provider to use |

---

### proxima plugin

Plugin management.

```bash
proxima plugin [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**

#### plugin list

List installed plugins.

```bash
proxima plugin list [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--type` | Filter by type (exporter, analyzer, hook, backend) |
| `--enabled` | Show only enabled plugins |

#### plugin install

Install a plugin.

```bash
proxima plugin install [PLUGIN_NAME] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--source` | Installation source (pypi, git, local) |
| `--version` | Specific version |

#### plugin enable/disable

Enable or disable a plugin.

```bash
proxima plugin enable [PLUGIN_NAME]
proxima plugin disable [PLUGIN_NAME]
```

#### plugin info

Show plugin details.

```bash
proxima plugin info [PLUGIN_NAME]
```

---

### proxima version

Show version information.

```bash
proxima version [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--full` | Show full version details including dependencies |
| `--check-update` | Check for available updates |

**Output:**

```
Proxima v0.3.0
Python 3.11.5
Platform: Windows-10-10.0.22621-SP0

Backends:
  cirq: 1.3.0
  qiskit: 1.0.2
  qiskit_aer: 0.14.0

Plugins: 8 installed, 8 enabled
```

---

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | Backend error |
| 5 | Timeout exceeded |
| 6 | Consent denied |
| 10 | Keyboard interrupt |

---

## Shell Completion

Enable shell completion for faster CLI usage.

### Bash

```bash
# Add to ~/.bashrc
eval "$(_PROXIMA_COMPLETE=bash_source proxima)"
```

### Zsh

```zsh
# Add to ~/.zshrc
eval "$(_PROXIMA_COMPLETE=zsh_source proxima)"
```

### PowerShell

```powershell
# Add to $PROFILE
Register-ArgumentCompleter -Native -CommandName proxima -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $env:_PROXIMA_COMPLETE = "powershell_complete"
    proxima | ForEach-Object { [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }
    Remove-Item env:_PROXIMA_COMPLETE
}
```

---

## Configuration File Reference

Default configuration file location: `./proxima.yaml` or `~/.proxima/config.yaml`

```yaml
# Proxima Configuration

# Default settings
defaults:
  backend: auto           # auto, cirq, qiskit_aer, lret, etc.
  shots: 1000
  timeout: 300
  output_format: table

# Backend configuration
backends:
  cirq:
    enabled: true
    priority: 1
  qiskit_aer:
    enabled: true
    priority: 2
    options:
      precision: double
  lret:
    enabled: true
    priority: 3

# LLM configuration
llm:
  enabled: false
  default_provider: openai
  consent_required: true
  providers:
    openai:
      model: gpt-4-turbo-preview
    anthropic:
      model: claude-3-opus-20240229

# Logging
logging:
  level: INFO
  file: ~/.proxima/logs/proxima.log
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Storage
storage:
  history_path: ~/.proxima/history
  cache_path: ~/.proxima/cache
  max_history_entries: 1000

# UI settings
ui:
  theme: dark
  show_progress: true
  table_style: rounded
```
