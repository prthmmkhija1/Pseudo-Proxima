# How to Build Proxima: A Strategic Guide to Designing an AI Agent for Quantum Simulations

> **Document Version:** 1.0  
> **Last Updated:** January 7, 2026  
> **Status:** High-Level Design Document

---

## Table of Contents

1. [Introduction](#introduction)
2. [Strategic System Sketch](#strategic-system-sketch)
3. [Phased Roadmap](#phased-roadmap)
4. [Phase-by-Phase Implementation Guide](#phase-by-phase-implementation-guide)
5. [Phase Summaries & Usage Guidance](#phase-summaries--usage-guidance)

---

## Introduction

### What is Proxima?

Proxima is an intelligent AI agent designed to orchestrate quantum simulations across multiple backends. It provides a unified interface for selecting, executing, comparing, and interpreting results from various quantum computing frameworks.

### Design Philosophy

Proxima draws architectural and UX inspiration from:

- **OpenCode AI** ([GitHub](https://github.com/opencode-ai/opencode)): For its intelligent code assistance patterns and agent-driven workflows
- **Crush by Charmbracelet** ([GitHub](https://github.com/charmbracelet/crush)): For its elegant terminal UI paradigms and user experience design

However, Proxima is built as a completely independent, extensible system with its own identity.

### Supported Quantum Backends

| Backend        | Simulator Types                         | Repository                                                                             |
| -------------- | --------------------------------------- | -------------------------------------------------------------------------------------- |
| **LRET**       | Framework Integration                   | [kunal5556/LRET](https://github.com/kunal5556/LRET/tree/feature/framework-integration) |
| **Cirq**       | Density Matrix, State Vector            | [quantumlib/Cirq](https://github.com/quantumlib/Cirq)                                  |
| **Qiskit Aer** | Density Matrix, State Vector            | [Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer)                              |
| **Extensible** | Custom backends via plugin architecture | —                                                                                      |

---

## Strategic System Sketch

### Overall Architecture Overview

Proxima follows a **layered modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                        │
│         (CLI / Future TUI via Bubble Tea / Future Web UI)           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT ORCHESTRATION LAYER                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │   Planner   │  │   Executor   │  │   State Machine Manager    │  │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              proxima_agent.md Interpreter                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         INTELLIGENCE LAYER                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  LLM Router      │  │  Backend Selector │  │  Insight Engine  │   │
│  │  (Local/Remote)  │  │  (Auto-Selection) │  │  (Analysis)      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RESOURCE MANAGEMENT LAYER                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Memory Monitor  │  │  Execution Timer │  │  Consent Manager │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Execution Control (Start/Abort/Pause/Resume)    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKEND ABSTRACTION LAYER                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │    LRET    │  │    Cirq    │  │ Qiskit Aer │  │  Custom    │     │
│  │  Adapter   │  │  Adapter   │  │  Adapter   │  │  Adapter   │     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA & OUTPUT LAYER                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Result Store    │  │  Comparison      │  │  Export Engine   │   │
│  │  (JSON/SQLite)   │  │  Aggregator      │  │  (CSV/XLSX)      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Core Components Detailed

#### 1. User Interface Layer

**Purpose:** Provides the interaction surface for users to communicate with Proxima.

**Components:**

- **CLI Interface:** Primary interface using libraries like Cobra (Go) or Click/Typer (Python)
- **TUI Interface (Future):** Rich terminal UI using Bubble Tea (Go) or Rich/Textual (Python)
- **Web UI (Future):** React or Svelte-based dashboard

**Data Flow:**

- User commands enter here
- Execution status, timers, and progress displayed here
- All consent prompts surface through this layer

---

#### 2. Agent Orchestration Layer

**Purpose:** The brain of Proxima—plans, coordinates, and manages task execution.

**Components:**

| Component                        | Responsibility                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| **Planner**                      | Decomposes user requests into executable stages; creates task DAGs                   |
| **Executor**                     | Runs tasks according to plan; manages parallel/sequential execution                  |
| **State Machine Manager**        | Tracks execution states (IDLE, PLANNING, RUNNING, PAUSED, ABORTED, COMPLETED, ERROR) |
| **proxima_agent.md Interpreter** | Parses and executes instructions from agent definition files                         |

**Key Behaviors:**

- Explicit planning before execution
- Transparent stage transitions
- Traceable execution history
- Support for future agent.md-based automation

---

#### 3. Intelligence Layer

**Purpose:** Provides smart decision-making capabilities.

**Components:**

| Component            | Function                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **LLM Router**       | Routes requests to appropriate LLM (local vs. remote); manages API keys; enforces consent |
| **Backend Selector** | Analyzes workload characteristics and recommends optimal backend                          |
| **Insight Engine**   | Transforms raw simulation data into human-readable analytical insights                    |

**LLM Router Decision Tree:**

```
User Request → Is LLM needed?
                    │
            ┌───────┴───────┐
            ▼               ▼
           Yes              No
            │                │
    Local LLM available?    Proceed
            │
    ┌───────┴───────┐
    ▼               ▼
   Yes              No
    │                │
User consent?    Use Remote API
    │            (with consent)
    ▼
Use Local LLM
```

---

#### 4. Resource Management Layer

**Purpose:** Ensures safe, transparent, and controllable execution.

**Components:**

- **Memory Monitor:** Continuously tracks RAM usage using psutil or similar; triggers warnings at configurable thresholds
- **Execution Timer:** Tracks wall-clock time per stage and total execution; displays elapsed time in real-time
- **Consent Manager:** Gates all potentially risky or resource-intensive operations behind explicit user approval
- **Execution Controller:** Implements Start, Abort, Pause, Resume operations with proper state persistence

**Fail-Safe Decision Flow:**

```
Before Execution:
  │
  ├─→ Check available memory
  │     └─→ Below threshold? → Warn user → Require consent
  │
  ├─→ Check hardware compatibility
  │     └─→ Incompatible? → Explain risks → Offer "force execute"
  │
  └─→ All checks pass → Proceed with execution
```

---

#### 5. Backend Abstraction Layer

**Purpose:** Provides a unified interface to diverse quantum simulation backends.

**Design Pattern:** Adapter Pattern with Plugin Architecture

**Each Adapter Must Implement:**

- `initialize()` — Set up the backend
- `validate_circuit(circuit)` — Check circuit compatibility
- `execute(circuit, options)` — Run the simulation
- `get_capabilities()` — Report supported features (noise models, qubit limits, etc.)
- `get_resource_requirements()` — Estimate memory/compute needs

**Backend Selection Intelligence:**

```
User Query Analysis:
  │
  ├─→ Extract circuit size (qubit count)
  ├─→ Identify noise requirements
  ├─→ Detect density matrix vs. state vector needs
  ├─→ Check hardware constraints
  │
  └─→ Score each backend → Select highest score → Explain selection
```

---

#### 6. Data & Output Layer

**Purpose:** Manages simulation results, comparisons, and exports.

**Components:**

- **Result Store:** Persists simulation outcomes in structured format (JSON or SQLite)
- **Comparison Aggregator:** Aligns results from multiple backends for side-by-side analysis
- **Export Engine:** Generates CSV, XLSX, and formatted reports

---

### Control & Data Flow Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CONTROL FLOW                                │
│                                                                      │
│  User Command → Planning → Resource Check → Consent → Execute →      │
│  Monitor → Complete/Abort/Pause → Report Results                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                  │
│                                                                      │
│  User Input → Parsed Request → Execution Plan → Backend Calls →      │
│  Raw Results → Insight Processing → Formatted Output → User          │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Feature Interconnection Map

| Feature                  | Depends On                           | Feeds Into                       |
| ------------------------ | ------------------------------------ | -------------------------------- |
| Execution Timer          | State Machine                        | UI Layer, Logs                   |
| Backend Selection        | Intelligence Layer, Backend Adapters | Executor                         |
| Fail-Safe                | Memory Monitor, Consent Manager      | Executor (gates execution)       |
| Execution Control        | State Machine                        | All execution paths              |
| Result Interpretation    | Insight Engine, LLM Router           | Export Engine, UI                |
| Multi-Backend Comparison | All Adapters, Comparison Aggregator  | Insight Engine                   |
| LLM Integration          | LLM Router, Consent Manager          | Insight Engine, Backend Selector |
| agent.md Compatibility   | Interpreter, Planner                 | Full orchestration               |

---

## Phased Roadmap

### Overview

The development of Proxima is divided into **six phases**, each building upon the previous to create a fully functional quantum simulation agent.

```
Phase 1: Foundation & Core Infrastructure
    │
    ▼
Phase 2: Backend Integration & Abstraction
    │
    ▼
Phase 3: Intelligence & Decision Systems
    │
    ▼
Phase 4: Safety, Control & Transparency
    │
    ▼
Phase 5: Advanced Features & agent.md Support
    │
    ▼
Phase 6: UI Enhancement & Production Hardening
```

---

### Phase Summary Table

| Phase | Name                 | Duration  | Key Deliverables                                                  |
| ----- | -------------------- | --------- | ----------------------------------------------------------------- |
| 1     | Foundation           | 3-4 weeks | Project structure, CLI scaffold, configuration system             |
| 2     | Backend Integration  | 4-5 weeks | LRET, Cirq, Qiskit adapters; unified interface                    |
| 3     | Intelligence Systems | 4-5 weeks | LLM integration, backend auto-selection, insight engine           |
| 4     | Safety & Control     | 3-4 weeks | Memory monitoring, execution control, consent system              |
| 5     | Advanced Features    | 4-5 weeks | Multi-backend comparison, agent.md interpreter, pipeline planning |
| 6     | UI & Production      | 3-4 weeks | TUI, documentation, testing, deployment                           |

**Total Estimated Duration:** 21-27 weeks

---

## Phase-by-Phase Implementation Guide

---

### Phase 1: Foundation & Core Infrastructure

**Objective:** Establish the project skeleton, development environment, and foundational systems.

---

#### Step 1.1: Project Initialization

**Language Selection Decision:**

- **Option A (Recommended for Performance):** Go
  - Use Go modules for dependency management
  - Leverage goroutines for concurrent execution
  - Utilize Cobra for CLI framework
- **Option B (Recommended for Quantum Ecosystem):** Python
  - Use Poetry or PDM for dependency management
  - Leverage asyncio for concurrent execution
  - Utilize Typer or Click for CLI framework

**Directory Structure to Create:**

```
proxima/
├── cmd/                    # CLI entry points
├── internal/               # Private application code
│   ├── agent/              # Orchestration logic
│   ├── backends/           # Backend adapters
│   ├── intelligence/       # LLM and decision systems
│   ├── resources/          # Resource management
│   ├── ui/                 # User interface components
│   └── utils/              # Shared utilities
├── pkg/                    # Public libraries (if needed)
├── configs/                # Configuration files
├── docs/                   # Documentation
├── tests/                  # Test suites
├── proxima_agent.md        # Agent definition file (for Phase 5)
└── README.md
```

---

#### Step 1.2: Configuration System

**Tools to Use:**

- **Go:** Viper for configuration management
- **Python:** Pydantic Settings or Dynaconf

**Configuration Hierarchy to Implement:**

1. Default values (hardcoded)
2. Configuration file (`~/.proxima/config.yaml`)
3. Environment variables (prefixed with `PROXIMA_`)
4. Command-line flags (highest priority)

**Configuration Categories:**

- General settings (verbosity, output format)
- Backend configurations (API endpoints, credentials)
- LLM settings (API keys, model preferences, local LLM paths)
- Resource thresholds (memory limits, timeout values)
- Consent preferences (auto-approve certain actions)

---

#### Step 1.3: Logging & Telemetry Foundation

**Logging Framework:**

- **Go:** Zap or Zerolog for structured logging
- **Python:** Structlog or Loguru

**Log Levels to Implement:**

- DEBUG: Detailed execution information
- INFO: General operational messages
- WARN: Resource warnings, consent prompts
- ERROR: Failures and exceptions

**Telemetry Considerations:**

- Execution timing metrics
- Backend usage statistics
- Error frequency tracking
- All telemetry must be opt-in with explicit consent

---

#### Step 1.4: CLI Scaffold

**Commands to Implement:**

| Command            | Description                        |
| ------------------ | ---------------------------------- |
| `proxima init`     | Initialize configuration           |
| `proxima config`   | View/modify settings               |
| `proxima run`      | Execute a simulation (placeholder) |
| `proxima backends` | List available backends            |
| `proxima version`  | Display version information        |

**CLI Flags to Support:**

- `--verbose` / `-v`: Increase output verbosity
- `--config`: Specify configuration file path
- `--backend`: Explicitly select backend
- `--dry-run`: Show plan without executing
- `--force`: Skip consent prompts (with warnings)

---

#### Step 1.5: State Machine Foundation

**States to Define:**

```
IDLE → PLANNING → READY → RUNNING → COMPLETED
                    │        │
                    │        ├──→ PAUSED → RUNNING (resume)
                    │        │
                    │        └──→ ABORTED
                    │
                    └──→ ERROR
```

**State Transition Rules:**

- Only valid transitions allowed
- Each transition logged with timestamp
- State persisted for recovery

**Implementation Approach:**

- **Go:** Use a finite state machine library or implement custom
- **Python:** Use transitions library or implement custom

---

### Phase 2: Backend Integration & Abstraction

**Objective:** Create adapters for quantum backends with a unified interface.

---

#### Step 2.1: Define Backend Interface Contract

**Abstract Interface Methods:**

| Method                               | Purpose                        |
| ------------------------------------ | ------------------------------ |
| `get_name()`                         | Returns backend identifier     |
| `get_version()`                      | Returns backend version        |
| `get_capabilities()`                 | Returns feature dictionary     |
| `get_resource_requirements(circuit)` | Estimates memory/compute needs |
| `validate(circuit)`                  | Checks circuit compatibility   |
| `execute(circuit, options)`          | Runs simulation                |
| `get_result_schema()`                | Returns expected output format |

**Capability Flags:**

- `supports_density_matrix`: Boolean
- `supports_state_vector`: Boolean
- `supports_noise_models`: Boolean
- `max_qubits`: Integer
- `supports_gpu`: Boolean

---

#### Step 2.2: LRET Adapter Implementation

**Integration Steps:**

1. Add LRET framework-integration branch as dependency
2. Create adapter class implementing the interface
3. Map LRET's native API to unified interface
4. Implement circuit translation layer (if needed)
5. Handle LRET-specific error types
6. Write adapter-specific unit tests

**LRET-Specific Considerations:**

- Framework integration branch features
- Custom simulation modes
- Result format normalization

---

#### Step 2.3: Cirq Adapter Implementation

**Integration Steps:**

1. Add Cirq as dependency via pip/poetry
2. Create adapter class with dual simulator support
3. Implement Density Matrix simulator path
4. Implement State Vector simulator path
5. Add simulator selection logic based on use case
6. Normalize Cirq result objects to common format

**Cirq Simulator Selection Logic:**

- Use Density Matrix for: noise simulation, mixed states, smaller circuits
- Use State Vector for: pure states, larger circuits (memory permitting)

---

#### Step 2.4: Qiskit Aer Adapter Implementation

**Integration Steps:**

1. Add qiskit-aer as dependency
2. Create adapter class with dual simulator support
3. Implement Density Matrix simulator path via AerSimulator
4. Implement State Vector simulator path via AerSimulator
5. Support Qiskit's transpilation pipeline
6. Handle Qiskit-specific job patterns

**Qiskit-Specific Features:**

- Noise model integration
- Backend options configuration
- Shot-based vs. statevector execution

---

#### Step 2.5: Backend Registry & Discovery

**Registry Responsibilities:**

- Maintain list of available backends
- Lazy-load adapters on demand
- Report backend health status
- Support dynamic backend registration

**Discovery Mechanism:**

1. Scan for installed quantum packages
2. Verify each backend is functional
3. Cache capabilities for quick access
4. Re-scan on user request

---

#### Step 2.6: Unified Result Format

**Common Result Schema:**

```
{
  "backend": "string",
  "simulator_type": "density_matrix | state_vector",
  "execution_time_ms": "number",
  "qubit_count": "number",
  "result_type": "counts | statevector | density_matrix",
  "data": { ... },
  "metadata": { ... }
}
```

**Result Normalization:**

- Convert backend-specific formats to common schema
- Preserve original data in metadata
- Standardize probability representations

---

### Phase 3: Intelligence & Decision Systems

**Objective:** Integrate LLM capabilities and intelligent decision-making.

---

#### Step 3.1: LLM Router Architecture

**Components to Build:**

1. **Provider Registry:** Tracks available LLM providers
2. **Local LLM Detector:** Finds locally installed models
3. **API Key Manager:** Securely stores and retrieves keys
4. **Request Router:** Directs queries to appropriate provider
5. **Consent Gate:** Enforces user approval before LLM calls

**Supported Provider Types:**

| Type              | Examples                     | Detection Method                |
| ----------------- | ---------------------------- | ------------------------------- |
| Remote API        | OpenAI, Anthropic, Google    | API key presence                |
| Local Inference   | Ollama, LM Studio, llama.cpp | Process detection, socket check |
| Local Model Files | GGUF, GGML files             | File system scan                |

---

#### Step 3.2: Local LLM Integration

**Detection Strategies:**

1. Check for running Ollama service (default port 11434)
2. Check for LM Studio server (configurable port)
3. Scan configured directories for model files
4. Verify model compatibility and readiness

**Local LLM Interface:**

- Use OpenAI-compatible API format when available
- Fall back to native API for specific runtimes
- Support model selection from available local models

**User Distinction Requirements:**

- Clear labeling: "[LOCAL LLM]" vs "[REMOTE API]"
- Separate consent prompts for each type
- Log which provider handled each request

---

#### Step 3.3: Remote API Integration

**Providers to Support:**

1. **OpenAI:** GPT-4, GPT-4-turbo models
2. **Anthropic:** Claude models
3. **Google:** Gemini models
4. **Custom:** User-defined OpenAI-compatible endpoints

**API Key Management:**

- Store encrypted in system keychain (keyring library)
- Support environment variable fallback
- Never log or display API keys
- Validate keys on startup (optional)

---

#### Step 3.4: Consent Management System

**Consent Types:**

| Action                        | Consent Level                      |
| ----------------------------- | ---------------------------------- |
| Use local LLM                 | Explicit per-session or persistent |
| Use remote API                | Explicit per-session or persistent |
| Modify backend logic          | Always explicit                    |
| Force execute (low resources) | Always explicit                    |
| Execute untrusted agent.md    | Always explicit                    |

**Consent Storage:**

- Session consents: In-memory only
- Persistent consents: Encrypted configuration file
- Audit log: Record all consent decisions

**Consent Prompt Format:**

```
╭─────────────────────────────────────────────────────╮
│ CONSENT REQUIRED                                    │
├─────────────────────────────────────────────────────┤
│ Action: Use remote LLM (OpenAI GPT-4)               │
│ Reason: Analyze simulation results                  │
│ Data sent: Summary statistics (no raw data)         │
├─────────────────────────────────────────────────────┤
│ [Y] Approve this time                               │
│ [A] Always approve this action                      │
│ [N] Deny                                            │
╰─────────────────────────────────────────────────────╯
```

---

#### Step 3.5: Backend Auto-Selection Intelligence

**Selection Algorithm:**

1. **Parse User Query:** Extract circuit characteristics
2. **Analyze Requirements:**
   - Qubit count
   - Gate types used
   - Noise requirements
   - Output type needed (counts vs. statevector)
3. **Score Backends:** Rate each backend on:
   - Feature compatibility (required features supported)
   - Performance (historical execution times)
   - Resource fit (memory requirements vs. available)
4. **Select Best:** Choose highest-scoring backend
5. **Explain Selection:** Generate human-readable justification

**Explanation Template:**

```
Selected backend: Qiskit Aer (State Vector Simulator)
Reason: Your circuit has 12 qubits with no noise requirements.
        State vector simulation provides exact amplitudes.
        Estimated memory: 128 MB (available: 8 GB)
        Estimated time: ~2 seconds
```

---

#### Step 3.6: Insight Engine

**Purpose:** Transform raw simulation data into actionable insights.

**Analysis Capabilities:**

1. **Statistical Analysis:**

   - Probability distributions
   - Entropy calculations
   - Fidelity metrics

2. **Comparative Analysis:**

   - Backend result differences
   - Performance comparisons
   - Accuracy assessments

3. **Visualization Recommendations:**
   - Suggest appropriate chart types
   - Highlight significant patterns
   - Flag anomalies

**LLM-Assisted Interpretation:**

- Feed summarized results to LLM
- Request natural language explanations
- Generate decision recommendations
- Always with user consent

---

### Phase 4: Safety, Control & Transparency

**Objective:** Implement resource monitoring, execution control, and transparency features.

---

#### Step 4.1: Memory Monitoring System

**Implementation Using:**

- **Python:** psutil library
- **Go:** gopsutil library

**Monitoring Metrics:**

| Metric         | Description                | Threshold                     |
| -------------- | -------------------------- | ----------------------------- |
| Available RAM  | Free memory for allocation | Configurable (default: 500MB) |
| Process Memory | Proxima's own usage        | Configurable (default: 2GB)   |
| Swap Usage     | Virtual memory pressure    | Warning at 50%                |

**Monitoring Workflow:**

1. Check resources before execution starts
2. Continuous monitoring during execution
3. Trigger warnings at configurable thresholds
4. Pause execution if critical levels reached
5. Resume only with user consent

---

#### Step 4.2: Execution Timer & Progress Tracking

**Timer Components:**

1. **Global Timer:** Total execution time
2. **Stage Timer:** Per-stage elapsed time
3. **ETA Calculator:** Estimated time remaining

**Display Format:**

```
╭─ Execution Status ──────────────────────────────────╮
│ Overall: ██████████░░░░░░░░░░ 50% │ 2m 34s elapsed  │
├─────────────────────────────────────────────────────┤
│ ✓ Planning          │ 0.3s                          │
│ ✓ Backend Init      │ 1.2s                          │
│ → Executing Circuit │ 2m 32s (running...)           │
│ ○ Result Analysis   │ pending                       │
│ ○ Insight Generation│ pending                       │
╰─────────────────────────────────────────────────────╯
```

**Progress Events:**

- Stage start/complete
- Percentage updates (for long operations)
- Warning events
- Error events

---

#### Step 4.3: Execution Control Implementation

**Control Operations:**

| Operation  | Implementation                                       |
| ---------- | ---------------------------------------------------- |
| **Start**  | Initialize resources, begin execution pipeline       |
| **Abort**  | Immediate termination, cleanup resources, log reason |
| **Pause**  | Suspend at next safe checkpoint, preserve state      |
| **Resume** | Restore state, continue from checkpoint              |

**Pause/Resume Mechanism:**

1. Define safe checkpoint locations in execution flow
2. At checkpoints, check for pause signal
3. If paused, serialize current state to disk
4. On resume, deserialize and continue

**Abort Cleanup Checklist:**

- Release backend resources
- Close file handles
- Save partial results (if any)
- Log abort reason and state
- Return to IDLE state

---

#### Step 4.4: State Visibility & Traceability

**State Transition Logging:**

```
[2026-01-07 10:23:45] State: IDLE → PLANNING
[2026-01-07 10:23:45] State: PLANNING → READY
[2026-01-07 10:23:46] State: READY → RUNNING
[2026-01-07 10:25:12] State: RUNNING → PAUSED (user request)
[2026-01-07 10:26:30] State: PAUSED → RUNNING (resumed)
[2026-01-07 10:28:45] State: RUNNING → COMPLETED
```

**Execution History:**

- Persist last N executions
- Include: start time, duration, backend, status, result summary
- Queryable via CLI command

---

#### Step 4.5: Hardware Compatibility Checks

**Checks to Perform:**

1. **GPU Availability:** For GPU-accelerated backends
2. **CUDA Version:** If GPU backend selected
3. **CPU Features:** AVX support for certain optimizations
4. **Memory Architecture:** 32-bit vs 64-bit limitations

**Incompatibility Handling:**

```
╭─ Hardware Warning ──────────────────────────────────╮
│ The selected backend (Qiskit Aer GPU) requires      │
│ CUDA 11.0+, but CUDA 10.2 was detected.             │
├─────────────────────────────────────────────────────┤
│ Risks of forcing execution:                         │
│ • Simulation may fail partway through               │
│ • Results may be incorrect                          │
│ • System instability possible                       │
├─────────────────────────────────────────────────────┤
│ [F] Force execute anyway (not recommended)          │
│ [S] Switch to CPU backend (recommended)             │
│ [C] Cancel                                          │
╰─────────────────────────────────────────────────────╯
```

---

### Phase 5: Advanced Features & agent.md Support

**Objective:** Implement multi-backend comparison, pipeline planning, and agent.md support.

---

#### Step 5.1: Multi-Backend Comparison Framework

**Comparison Workflow:**

1. User specifies multiple backends or selects "compare all"
2. Proxima validates circuit compatibility with each backend
3. Execute same simulation on each backend (parallel if possible)
4. Collect and normalize results
5. Generate comparative analysis

**Parallel Execution Strategy:**

- Use process pools for CPU-bound backends
- Respect memory constraints (may need sequential execution)
- Track individual backend timings

**Comparison Report Structure:**

```
╭─ Multi-Backend Comparison Report ───────────────────╮
│ Circuit: 8-qubit Grover's Algorithm                 │
│ Backends Compared: 3                                │
├─────────────────────────────────────────────────────┤
│ Backend          │ Time    │ Memory  │ Status       │
├──────────────────┼─────────┼─────────┼──────────────┤
│ LRET             │ 1.23s   │ 256 MB  │ ✓ Success    │
│ Cirq (SV)        │ 0.89s   │ 128 MB  │ ✓ Success    │
│ Qiskit Aer (SV)  │ 0.95s   │ 130 MB  │ ✓ Success    │
├─────────────────────────────────────────────────────┤
│ Result Consistency: 99.97% agreement                │
│ Fastest: Cirq (State Vector)                        │
│ Most Memory Efficient: Cirq (State Vector)          │
╰─────────────────────────────────────────────────────╯
```

---

#### Step 5.2: Pipeline Planning System

**Planning Stages:**

1. **Request Parsing:** Understand what user wants
2. **Requirement Analysis:** Determine needed resources and backends
3. **Dependency Resolution:** Order tasks by dependencies
4. **Resource Allocation:** Assign backends and compute resources
5. **Execution Plan Generation:** Create detailed step-by-step plan

**Plan Representation (DAG):**

```
Parse Input → Validate Circuit → Select Backend(s)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              Execute on         Execute on          Execute on
                LRET              Cirq               Qiskit Aer
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                              Aggregate Results
                                        │
                                        ▼
                              Generate Insights
                                        │
                                        ▼
                              Export Report
```

**Plan Display Before Execution:**

```
╭─ Execution Plan ────────────────────────────────────╮
│ The following steps will be executed:               │
├─────────────────────────────────────────────────────┤
│ 1. Parse and validate input circuit                 │
│ 2. Initialize backends: LRET, Cirq                  │
│ 3. Execute simulation on LRET                       │
│ 4. Execute simulation on Cirq (parallel)            │
│ 5. Compare and aggregate results                    │
│ 6. Generate insight report                          │
│ 7. Export to CSV                                    │
├─────────────────────────────────────────────────────┤
│ Estimated total time: 3-5 minutes                   │
│ Estimated memory peak: 512 MB                       │
├─────────────────────────────────────────────────────┤
│ [P] Proceed   [M] Modify plan   [C] Cancel          │
╰─────────────────────────────────────────────────────╯
```

---

#### Step 5.3: proxima_agent.md Interpreter

**File Format Specification:**

```markdown
# proxima_agent.md

## Metadata

- version: 1.0
- author: user
- created: 2026-01-07

## Configuration

- default_backend: auto
- parallel_execution: true
- insight_level: detailed

## Tasks

### Task: Run Quantum Simulation

- circuit_file: grover_8qubit.qasm
- backends: [cirq, qiskit_aer]
- compare_results: true
- export_format: xlsx

### Task: Analyze Results

- use_llm: local_preferred
- analysis_type: statistical
- generate_recommendations: true
```

**Interpreter Workflow:**

1. Parse markdown file using markdown parser
2. Extract metadata and configuration
3. Build task list from Task sections
4. Validate each task against capabilities
5. Generate execution plan
6. Request user consent for sensitive operations
7. Execute plan

**Security Considerations:**

- Validate file integrity before parsing
- Sandbox any file path references
- Require explicit consent for LLM usage
- Log all interpreted actions

---

#### Step 5.4: Result Export & Interpretation

**Export Formats:**

| Format | Use Case                         | Library                 |
| ------ | -------------------------------- | ----------------------- |
| CSV    | Data analysis, spreadsheets      | csv (stdlib)            |
| XLSX   | Rich formatting, multiple sheets | openpyxl                |
| JSON   | Programmatic access              | json (stdlib)           |
| PDF    | Reports (future)                 | reportlab or weasyprint |

**Insight Generation Pipeline:**

1. Load raw results
2. Apply statistical analysis
3. Identify patterns and anomalies
4. Generate natural language summary (with LLM if consented)
5. Create visualizations (matplotlib, plotly)
6. Compile into structured report

**Human-Readable Insight Example:**

```
╭─ Simulation Insights ───────────────────────────────╮
│                                                     │
│ Summary:                                            │
│ The Grover's algorithm simulation successfully      │
│ amplified the target state |101⟩ to 97.3%           │
│ probability after 2 iterations, consistent with     │
│ theoretical predictions.                            │
│                                                     │
│ Key Findings:                                       │
│ • Target state probability: 97.3%                   │
│ • Theoretical optimum: 97.8%                        │
│ • Fidelity: 99.5%                                   │
│ • All non-target states suppressed below 1%         │
│                                                     │
│ Recommendations:                                    │
│ • Results are valid for algorithm verification      │
│ • Consider noise simulation for hardware estimates  │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

---

#### Step 5.5: Additional Features (Inspired by OpenCode AI & Crush)

**Feature: Multi-Model Support**

- **Value:** Users can leverage different LLMs for different tasks
- **Implementation:** Model router with task-to-model mapping
- **Use Case:** Use fast model for quick analysis, powerful model for complex interpretation

**Feature: Flexible LLM Switching**

- **Value:** Switch between local and remote LLMs based on task sensitivity
- **Implementation:** Runtime model switching via command or config
- **Use Case:** Use local for sensitive data, remote for general queries

**Feature: Session Persistence**

- **Value:** Resume previous work after restart
- **Implementation:** Serialize session state to disk
- **Use Case:** Long-running simulations, computer restarts

**Feature: Plugin System**

- **Value:** Extend Proxima without modifying core
- **Implementation:** Plugin discovery and loading mechanism
- **Use Case:** Custom backends, custom analysis modules

**Feature: Batch Execution**

- **Value:** Run multiple simulations unattended
- **Implementation:** Queue-based execution with notifications
- **Use Case:** Parameter sweeps, overnight runs

**Feature: Result Caching**

- **Value:** Avoid redundant computations
- **Implementation:** Content-addressed cache with configurable TTL
- **Use Case:** Repeated simulations, iterative development

---

### Phase 6: UI Enhancement & Production Hardening

**Objective:** Polish the user experience and prepare for production deployment.

---

#### Step 6.1: Terminal UI (TUI) Implementation

**Framework Options:**

- **Go:** Bubble Tea (bubbletea) with Lip Gloss for styling
- **Python:** Textual or Rich

**TUI Components to Build:**

1. **Dashboard View:** Overview of system status, recent executions
2. **Execution View:** Real-time progress, logs, timers
3. **Configuration View:** Interactive settings management
4. **Results Browser:** Navigate and inspect past results
5. **Backend Manager:** View and configure backends

**Design Principles (Inspired by Crush):**

- Minimal, clean aesthetic
- Responsive to terminal size
- Keyboard-first navigation
- Contextual help available
- Consistent color theming

---

#### Step 6.2: Error Handling & Recovery

**Error Categories:**

| Category       | Handling Strategy                    |
| -------------- | ------------------------------------ |
| User Error     | Clear message, suggest correction    |
| Backend Error  | Retry logic, fallback backend option |
| Resource Error | Pause, warn, wait for consent        |
| System Error   | Log, notify, graceful degradation    |

**Recovery Mechanisms:**

- Checkpoint-based recovery for long operations
- Automatic retry with exponential backoff
- Fallback backend selection on failure
- Partial result preservation

---

#### Step 6.3: Testing Strategy

**Test Levels:**

1. **Unit Tests:** Individual functions and methods
2. **Integration Tests:** Component interactions
3. **Backend Tests:** Adapter functionality with mock backends
4. **End-to-End Tests:** Full workflow scenarios
5. **Performance Tests:** Resource usage, execution time benchmarks

**Testing Frameworks:**

- **Python:** pytest with pytest-asyncio, pytest-mock
- **Go:** testing package with testify

**CI/CD Pipeline:**

- Run tests on every pull request
- Linting and formatting checks
- Security scanning
- Build artifacts for releases

---

#### Step 6.4: Documentation

**Documentation Types:**

1. **User Guide:** How to use Proxima
2. **API Reference:** For programmatic usage
3. **Developer Guide:** How to extend Proxima
4. **Backend Guide:** How to add new backends
5. **agent.md Reference:** File format specification

**Documentation Tools:**

- **Python:** Sphinx or MkDocs
- **Go:** godoc with Hugo or MkDocs

---

#### Step 6.5: Packaging & Distribution

**Distribution Methods:**

1. **PyPI Package** (if Python): `pip install proxima-agent`
2. **Homebrew Tap** (macOS): `brew install proxima`
3. **Binary Releases** (all platforms): GitHub Releases
4. **Container Image**: Docker Hub or GitHub Container Registry

**Packaging Checklist:**

- Version management (semantic versioning)
- Changelog maintenance
- License file inclusion
- Dependency locking

---

## Phase Summaries & Usage Guidance

---

### Phase 1 Summary: Foundation

**Features Implemented:**

- Project structure and development environment
- Configuration system with hierarchical loading
- Basic CLI with core commands
- Logging infrastructure
- State machine foundation

**New Capabilities:**

- Initialize Proxima with custom configuration
- View and modify settings
- Basic command structure ready for extension

**Usage After Phase 1:**

```
# Initialize Proxima
proxima init

# View configuration
proxima config show

# Set a configuration value
proxima config set verbosity debug

# Check version
proxima version
```

---

### Phase 2 Summary: Backend Integration

**Features Implemented:**

- LRET adapter with full simulation support
- Cirq adapter (Density Matrix + State Vector)
- Qiskit Aer adapter (Density Matrix + State Vector)
- Backend registry and discovery
- Unified result format

**New Capabilities:**

- List available backends and their capabilities
- Run simulations on any supported backend
- Receive normalized results regardless of backend

**Usage After Phase 2:**

```
# List available backends
proxima backends list

# Show backend details
proxima backends info cirq

# Run a simulation on specific backend
proxima run --backend qiskit_aer --circuit circuit.qasm

# Run with automatic backend selection (placeholder)
proxima run --circuit circuit.qasm
```

---

### Phase 3 Summary: Intelligence Systems

**Features Implemented:**

- LLM Router with local/remote support
- API key management for remote LLMs
- Local LLM detection and integration
- Backend auto-selection intelligence
- Insight engine for result interpretation
- Consent management system

**New Capabilities:**

- Automatic backend selection with explanation
- LLM-assisted result analysis (with consent)
- Switch between local and remote LLMs
- Human-readable insights from simulation data

**Usage After Phase 3:**

```
# Run with auto-selection (now functional)
proxima run --circuit circuit.qasm
# Output: Selected backend: Cirq (State Vector)
#         Reason: 8-qubit pure state simulation...

# Get insights on results
proxima analyze --results results.json --use-llm local

# Configure LLM settings
proxima config set llm.provider openai
proxima config set llm.local_path /path/to/ollama
```

---

### Phase 4 Summary: Safety & Control

**Features Implemented:**

- Memory monitoring with configurable thresholds
- Execution timer with stage tracking
- Start/Abort/Pause/Resume operations
- State visibility and transition logging
- Hardware compatibility checks
- Force execute with explicit consent

**New Capabilities:**

- Real-time execution progress display
- Pause long-running simulations
- Resume from checkpoints
- Clear warnings before risky operations
- Full execution history

**Usage After Phase 4:**

```
# Run with real-time progress
proxima run --circuit circuit.qasm
# Shows: ██████░░░░ 60% | 1m 23s elapsed

# Pause execution (during run, press Ctrl+P)
# Resume execution
proxima resume --session latest

# Abort execution (during run, press Ctrl+C)
# View execution history
proxima history

# Force execute despite warnings
proxima run --circuit large_circuit.qasm --force
```

---

### Phase 5 Summary: Advanced Features

**Features Implemented:**

- Multi-backend comparison framework
- Pipeline planning with DAG visualization
- proxima_agent.md interpreter
- Enhanced result export (CSV, XLSX)
- Detailed insight generation
- Plugin system foundation
- Session persistence
- Result caching

**New Capabilities:**

- Compare same simulation across multiple backends
- Plan execution before running
- Automate workflows via agent.md files
- Export rich reports
- Resume sessions after restart

**Usage After Phase 5:**

```
# Compare across backends
proxima compare --circuit circuit.qasm --backends cirq,qiskit_aer,lret

# Show execution plan
proxima plan --circuit circuit.qasm --compare-all

# Execute from agent.md file
proxima agent run proxima_agent.md

# Export results
proxima export --session latest --format xlsx --output report.xlsx

# Resume previous session
proxima session resume
```

---

### Phase 6 Summary: UI & Production

**Features Implemented:**

- Full Terminal UI (TUI)
- Comprehensive error handling
- Complete test coverage
- Full documentation
- Multiple distribution packages

**New Capabilities:**

- Interactive dashboard
- Visual execution monitoring
- In-app configuration
- Results browser

**Usage After Phase 6:**

```
# Launch interactive TUI
proxima ui

# TUI provides:
# - Dashboard with system overview
# - Real-time execution monitoring
# - Configuration management
# - Results browsing and analysis
# - Backend management

# Standard CLI remains available
proxima run --circuit circuit.qasm
```

---

## Appendix

### A. Technology Stack Summary

| Component        | Python Option               | Go Option              |
| ---------------- | --------------------------- | ---------------------- |
| CLI Framework    | Typer, Click                | Cobra                  |
| TUI Framework    | Textual, Rich               | Bubble Tea             |
| Configuration    | Pydantic Settings, Dynaconf | Viper                  |
| Logging          | Structlog, Loguru           | Zap, Zerolog           |
| Async            | asyncio                     | goroutines             |
| Testing          | pytest                      | testing + testify      |
| State Machine    | transitions                 | custom or statemachine |
| HTTP Client      | httpx, aiohttp              | net/http               |
| Resource Monitor | psutil                      | gopsutil               |
| Keyring          | keyring                     | go-keyring             |

### B. Quantum Libraries Reference

| Library    | Purpose                      | Documentation         |
| ---------- | ---------------------------- | --------------------- |
| Cirq       | Google's quantum framework   | quantumai.google/cirq |
| Qiskit     | IBM's quantum SDK            | qiskit.org            |
| Qiskit Aer | High-performance simulators  | qiskit.org/aer        |
| LRET       | Custom framework integration | GitHub repository     |

### C. LLM Integration Reference

| Provider  | API Style         | Local Option |
| --------- | ----------------- | ------------ |
| OpenAI    | REST API          | —            |
| Anthropic | REST API          | —            |
| Ollama    | OpenAI-compatible | Yes          |
| LM Studio | OpenAI-compatible | Yes          |
| llama.cpp | Native API        | Yes          |

---

## Final Notes

This document provides a comprehensive blueprint for building Proxima. Each phase builds upon the previous, allowing for incremental development and testing.

**Key Success Factors:**

1. **Modularity:** Keep components loosely coupled
2. **Transparency:** Never hide what Proxima is doing
3. **Consent:** Always ask before sensitive operations
4. **Extensibility:** Design for future backends and features
5. **User Focus:** Prioritize clear communication and usability

**Next Steps:**

1. Review this document thoroughly
2. Set up development environment
3. Begin Phase 1 implementation
4. Iterate based on testing and feedback

---

_Document generated for the Proxima AI Agent project. This is a living document—update as the project evolves._
