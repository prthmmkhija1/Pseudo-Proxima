# Agent Files

This guide covers the `proxima_agent.md` file format, which enables LLM-based interaction with Proxima simulations.

## Overview

Agent files are Markdown-based instruction files that define how the Proxima agent processes natural language queries about quantum simulations. They provide:

- **Task definitions**: What the agent should accomplish
- **Execution context**: Backend preferences, constraints
- **Output specifications**: How results should be formatted
- **Safety rules**: Consent requirements and resource limits

## File Structure

```markdown
# proxima_agent.md

## Context
[Background information for the agent]

## Task
[What the agent should accomplish]

## Constraints
[Execution limits and requirements]

## Output
[Expected output format]
```

## Quick Start

### Creating an Agent File

```bash
# Initialize agent file in current directory
proxima agent init

# Initialize with template
proxima agent init --template simulation
```

### Running with Agent

```bash
# Execute with agent file
proxima agent run "Run a 5-qubit GHZ circuit"

# Use specific agent file
proxima agent run "Compare backends" --agent custom_agent.md
```

## Agent File Format

### Full Example

```markdown
# Quantum Simulation Agent

## Context

You are a quantum simulation assistant using the Proxima framework.
Available backends: LRET, Cirq, Qiskit, QuEST, qsim, cuQuantum.

Current workspace: /projects/quantum-research
Default backend: cirq

## Task

Execute quantum circuits as requested by the user. Select the optimal
backend based on circuit characteristics:

- Small circuits (< 10 qubits): Use LRET or Cirq
- Medium circuits (10-20 qubits): Use Qiskit or QuEST
- Large circuits (> 20 qubits): Use cuQuantum if GPU available

## Constraints

### Resource Limits
- max_qubits: 25
- max_memory_gb: 16
- timeout_seconds: 300

### Safety
- require_consent: true
- allow_file_writes: false
- sandbox_mode: true

### Backend Preferences
- preferred: ["cirq", "qiskit"]
- excluded: []

## Output

Format results as follows:

### Execution Summary
- Backend used
- Execution time
- Qubit count

### Results
- Measurement counts (top 10)
- State fidelity (if applicable)

### Recommendations
- Optimization suggestions
- Alternative backends

## Examples

User: "Create a Bell state"
Action: Execute 2-qubit circuit with H and CNOT gates

User: "Compare Cirq and Qiskit"
Action: Run same circuit on both backends, compare results
```

## Section Reference

### Context Section

Defines the agent's knowledge base and environment:

```markdown
## Context

### Role
You are a quantum simulation assistant specializing in circuit optimization.

### Environment
- Framework: Proxima v0.3.0
- Available backends: Cirq, Qiskit, QuEST, qsim, cuQuantum
- Python version: 3.10

### Knowledge
- Quantum gates: H, X, Y, Z, CNOT, CZ, SWAP, T, S, RX, RY, RZ
- Noise models: Depolarizing, amplitude damping, phase damping
- Optimization: Gate fusion, circuit simplification
```

### Task Section

Defines what the agent should accomplish:

```markdown
## Task

### Primary Objectives
1. Execute quantum circuits based on user requests
2. Select optimal backends automatically
3. Provide clear explanations of results

### Secondary Objectives
1. Suggest circuit optimizations
2. Warn about potential issues
3. Compare backend performance when relevant

### Decision Rules
- If circuit has noise: prefer Cirq
- If circuit needs transpilation: use Qiskit
- If GPU available and > 15 qubits: use cuQuantum
```

### Constraints Section

Defines execution limits and safety rules:

```markdown
## Constraints

### Hard Limits
- max_qubits: 30
- max_shots: 1000000
- max_memory_gb: 32
- timeout_seconds: 600

### Soft Limits (warnings)
- warn_qubits: 20
- warn_memory_gb: 16
- warn_execution_ms: 60000

### Safety Rules
- require_consent: true
- consent_threshold_qubits: 15
- consent_threshold_memory_gb: 8
- allow_dangerous_operations: false
- sandbox_mode: true

### Denied Operations
- File system writes outside workspace
- Network requests to external URLs
- Execution of arbitrary code
```

### Output Section

Defines how results should be formatted:

```markdown
## Output

### Format
- style: markdown
- verbosity: detailed
- include_code: true

### Sections

#### Summary
Include: backend, qubits, shots, execution time, success status

#### Results
Include: measurement counts, state vector (if small), probabilities

#### Visualization
Include: histogram for counts, Bloch sphere for single qubit

#### Code
Include: Python code to reproduce the simulation

### Examples
Include example input/output pairs for common operations
```

## Advanced Features

### Variables

Use variables for dynamic configuration:

```markdown
## Context

Current date: {{date}}
User: {{user_name}}
Workspace: {{workspace_path}}

## Constraints

max_qubits: {{max_qubits | default: 25}}
```

### Conditional Sections

```markdown
## Task

{{#if gpu_available}}
Prefer GPU-accelerated backends (cuQuantum) for large circuits.
{{else}}
Use CPU backends (qsim, QuEST) for best performance.
{{/if}}
```

### Includes

```markdown
## Context

{{include: common/quantum_gates.md}}
{{include: common/noise_models.md}}
```

## Integration with LLM Router

### Router Configuration

The agent file works with the LLM router for intelligent query processing:

```yaml
# proxima.yaml
intelligence:
  llm_router:
    enabled: true
    agent_file: proxima_agent.md
    models:
      - name: gpt-4
        priority: 1
        max_tokens: 4096
      - name: claude-3
        priority: 2
        max_tokens: 4096
    fallback: local
```

### Query Processing Pipeline

1. **Parse Query**: Extract intent and parameters
2. **Load Agent**: Read proxima_agent.md
3. **Apply Constraints**: Validate against limits
4. **Select Backend**: Choose optimal backend
5. **Execute**: Run simulation
6. **Format Output**: Apply output rules

### Example Flow

```
User Query: "Run a 10-qubit QFT with noise"
    |
    v
+-------------------+
| Parse Intent      |
| - Operation: QFT  |
| - Qubits: 10      |
| - Noise: yes      |
+-------------------+
    |
    v
+-------------------+
| Check Constraints |
| - 10 < 25        |
| - Noise OK       |
+-------------------+
    |
    v
+-------------------+
| Select Backend    |
| -> Cirq (noise)   |
+-------------------+
    |
    v
+-------------------+
| Execute & Return  |
+-------------------+
```

## CLI Commands

### Initialize

```bash
# Create default agent file
proxima agent init

# Create from template
proxima agent init --template research

# Create in specific location
proxima agent init --output /path/to/agent.md
```

### Validate

```bash
# Validate agent file syntax
proxima agent validate

# Validate specific file
proxima agent validate custom_agent.md
```

### Run

```bash
# Run query with agent
proxima agent run "Create a 5-qubit GHZ state"

# Run with specific agent file
proxima agent run "Query" --agent custom_agent.md

# Run in interactive mode
proxima agent run --interactive
```

### Show

```bash
# Show current agent configuration
proxima agent show

# Show parsed constraints
proxima agent show --section constraints
```

## Templates

### Simulation Template

```bash
proxima agent init --template simulation
```

Optimized for running quantum simulations with automatic backend selection.

### Research Template

```bash
proxima agent init --template research
```

Includes detailed output formatting and comparison capabilities.

### Education Template

```bash
proxima agent init --template education
```

Focused on explanations and step-by-step execution.

### CI/CD Template

```bash
proxima agent init --template cicd
```

Minimal output, strict resource limits, no consent prompts.

## Best Practices

### Agent File Organization

```
project/
 proxima_agent.md          # Main agent file
 agents/
    simulation.md         # Simulation-focused agent
    comparison.md         # Comparison-focused agent
    research.md           # Research-focused agent
 includes/
     gates.md              # Gate definitions
     constraints.md        # Shared constraints
     output.md             # Output templates
```

### Security Considerations

1. **Enable sandbox mode** for untrusted queries
2. **Set resource limits** to prevent abuse
3. **Require consent** for expensive operations
4. **Restrict file access** to workspace only

### Performance Tips

1. **Cache agent parsing** for repeated queries
2. **Use includes** for shared configuration
3. **Set appropriate timeouts** for your use case
4. **Prefer specific backends** over auto-selection for known workloads

## Troubleshooting

### Common Issues

**Agent file not found:**
```bash
# Check current directory
proxima agent show --path

# Specify explicit path
proxima agent run "Query" --agent /path/to/agent.md
```

**Constraint violations:**
```bash
# Show current constraints
proxima agent show --section constraints

# Override for single run
proxima agent run "Query" --override max_qubits=30
```

**Invalid syntax:**
```bash
# Validate agent file
proxima agent validate --verbose
```

**LLM connection issues:**
```bash
# Test LLM connectivity
proxima agent test-connection

# Use fallback mode
proxima agent run "Query" --fallback local
```
