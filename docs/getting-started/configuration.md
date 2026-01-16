# Configuration

This guide covers all configuration options for Proxima, including file locations, environment variables, and runtime settings.

## Configuration Hierarchy

Proxima uses a layered configuration system with the following priority (highest to lowest):

1. **Command-line arguments** (e.g., `--backend cirq`)
2. **Environment variables** (e.g., `PROXIMA_BACKEND=cirq`)
3. **User config file** (`~/.proxima/config.yaml`)
4. **Project config file** (`./proxima.yaml`)
5. **Default values**

## Configuration File Location

Proxima looks for configuration files in the following order:

```bash
# 1. Current directory
./proxima.yaml

# 2. User home directory
~/.proxima/config.yaml

# 3. System-wide (Linux/macOS)
/etc/proxima/config.yaml

# 4. Environment variable override
$PROXIMA_CONFIG
```

## Configuration File Structure

```yaml
# proxima.yaml - Complete configuration reference
version: "1.0"

# General settings
general:
  verbosity: info          # debug | info | warning | error
  output_format: rich      # text | json | rich
  color_enabled: true
  output_dir: "./reports"
  log_file: "./logs/proxima.log"

# Backend configuration
backends:
  default: auto            # auto | lret | cirq | qiskit | quest | qsim | cuquantum
  timeout_seconds: 300
  parallel_execution: true
  max_workers: 4
  
  # Backend-specific settings
  lret:
    enabled: true
    normalize_results: true
    
  cirq:
    enabled: true
    simulator_type: density_matrix  # state_vector | density_matrix
    noise_model: null
    
  qiskit:
    enabled: true
    simulator_type: aer_simulator
    optimization_level: 1
    transpile: true
    
  quest:
    enabled: true
    precision: double      # single | double | quad
    gpu_enabled: auto
    openmp_threads: auto
    truncation_threshold: 1e-4
    
  qsim:
    enabled: true
    use_avx: auto
    gate_fusion: aggressive
    threads: auto
    
  cuquantum:
    enabled: true
    device_id: 0
    memory_limit: auto
    fallback_to_cpu: true

# Execution settings
execution:
  parallel: true
  dry_run: false
  require_consent: true
  auto_approve_local: false
  checkpoint_interval: 60
  max_execution_time: 3600
  
  # Resource limits
  resources:
    memory_warn_threshold_mb: 8192
    memory_critical_threshold_mb: 14336
    max_qubits: 30

# LLM integration
llm:
  provider: none           # none | openai | anthropic | ollama | lm_studio
  model: gpt-4
  local_endpoint: "http://localhost:11434"
  require_consent: true
  auto_approve_local: true
  max_tokens: 4096
  temperature: 0.7
  
  # Provider-specific settings
  openai:
    api_key_env: OPENAI_API_KEY
    organization: null
    
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    
  ollama:
    endpoint: "http://localhost:11434"
    model: llama2
    
  lm_studio:
    endpoint: "http://localhost:1234"

# Export settings
export:
  format: json             # json | csv | xlsx | html
  include_metadata: true
  include_raw_results: false
  pretty_print: true
  compression: false
  
  # Comparison reports
  comparison:
    include_charts: true
    include_recommendations: true
    fidelity_threshold: 0.99

# Session management
session:
  persist: true
  auto_save: true
  save_interval: 30
  history_limit: 100
  
# Consent and audit
consent:
  auto_approve_local_llm: false
  auto_approve_remote_llm: false
  remember_decisions: true
  audit_all_decisions: true
  require_explicit_consent: true
```

## Environment Variables

All configuration options can be overridden using environment variables with the `PROXIMA_` prefix:

| Variable | Description | Example |
|----------|-------------|---------|
| `PROXIMA_CONFIG` | Path to config file | `/path/to/config.yaml` |
| `PROXIMA_LOG_LEVEL` | Logging verbosity | `debug`, `info`, `warning`, `error` |
| `PROXIMA_OUTPUT_DIR` | Output directory | `./reports` |
| `PROXIMA_BACKEND` | Default backend | `cirq`, `qiskit`, `quest` |
| `PROXIMA_PARALLEL` | Enable parallel execution | `true`, `false` |
| `PROXIMA_DRY_RUN` | Enable dry-run mode | `true`, `false` |
| `PROXIMA_LLM_PROVIDER` | LLM provider | `openai`, `anthropic`, `ollama` |
| `PROXIMA_REQUIRE_CONSENT` | Require user consent | `true`, `false` |

### API Keys

Store API keys securely using environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Or use a .env file (never commit to version control)
echo "OPENAI_API_KEY=sk-..." >> ~/.proxima/.env
```

## CLI Configuration Commands

### View Current Configuration

```bash
# Show all settings
proxima config show

# Show specific section
proxima config show --section backends

# Show as JSON
proxima config show --format json
```

### Modify Configuration

```bash
# Set a value
proxima config set backends.default cirq

# Set nested value
proxima config set llm.provider openai

# Set from environment
proxima config set-env PROXIMA_BACKEND cirq
```

### Reset Configuration

```bash
# Reset to defaults
proxima config reset

# Reset specific section
proxima config reset --section backends

# Backup before reset
proxima config reset --backup
```

### Validate Configuration

```bash
# Check configuration validity
proxima config validate

# Validate with verbose output
proxima config validate --verbose
```

## Backend-Specific Configuration

### LRET Configuration

```yaml
backends:
  lret:
    enabled: true
    normalize_results: true
    batch_size: 100
    api_verification: true
```

### Cirq Configuration

```yaml
backends:
  cirq:
    enabled: true
    simulator_type: density_matrix
    noise_model: null
    optimization_level: 2
    batch_execution: true
```

### Qiskit Aer Configuration

```yaml
backends:
  qiskit:
    enabled: true
    simulator_type: aer_simulator
    method: statevector        # statevector | density_matrix | automatic
    device: CPU                # CPU | GPU
    optimization_level: 1
    transpile: true
    coupling_map: null
```

### QuEST Configuration

```yaml
backends:
  quest:
    enabled: true
    precision: double          # single | double | quad
    gpu_enabled: auto          # auto | true | false
    openmp_threads: auto       # auto or integer
    mpi_enabled: false
    truncation_threshold: 1e-4
    max_qubits: 30
```

### qsim Configuration

```yaml
backends:
  qsim:
    enabled: true
    use_avx: auto              # auto | avx2 | avx512 | none
    gate_fusion: aggressive    # none | light | aggressive
    threads: auto              # auto or integer
    memory_mapped: false
    verbosity: 0
```

### cuQuantum Configuration

```yaml
backends:
  cuquantum:
    enabled: true
    device_id: 0
    memory_limit: auto
    workspace_size: 1073741824  # 1 GB
    fallback_to_cpu: true
    multi_gpu: false
```

## Validation Tips

1. **Timeouts**: Keep timeouts reasonable (30-300 seconds for local, longer for remote)
2. **Memory**: Set appropriate thresholds based on available system RAM
3. **Consent**: Enable `require_consent` when handling sensitive data
4. **Logging**: Use `debug` level only when troubleshooting
5. **Parallel**: Disable parallel execution if backends conflict
6. **GPU**: Set `gpu_enabled: false` if CUDA is not available

## Troubleshooting Configuration

### Common Issues

**Config file not found:**
```bash
# Check current config path
proxima config path

# Initialize default config
proxima init --config
```

**Invalid YAML syntax:**
```bash
# Validate config file
proxima config validate --verbose
```

**Environment variable not recognized:**
```bash
# List all recognized variables
proxima config env-vars

# Check specific variable
echo $PROXIMA_BACKEND
```

## Example Configurations

### Minimal Configuration

```yaml
version: "1.0"
backends:
  default: auto
execution:
  require_consent: false
```

### GPU-Optimized Configuration

```yaml
version: "1.0"
backends:
  default: cuquantum
  cuquantum:
    enabled: true
    device_id: 0
    memory_limit: 8192
  quest:
    gpu_enabled: true
execution:
  parallel: false  # GPU backends often conflict
```

### CI/CD Configuration

```yaml
version: "1.0"
general:
  verbosity: warning
  color_enabled: false
backends:
  default: cirq
  timeout_seconds: 60
execution:
  require_consent: false
  dry_run: false
export:
  format: json
  pretty_print: false
```
