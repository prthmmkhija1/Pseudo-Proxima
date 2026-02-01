# Agent Functionality Implementation Guide

**Proxima Quantum Computing Platform - AI Agent System**  
**Version:** 1.0  
**Date:** February 1, 2026  
**Target Implementation:** GPT-5.1 Codex Max, GPT-5.2 Codex Max, Claude Opus 4.5

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1: Real-Time Execution & Results Monitoring](#phase-1-real-time-execution--results-monitoring)
4. [Phase 2: Agent UI/UX Enhancements](#phase-2-agent-uiux-enhancements)
5. [Phase 3: Terminal Integration & Multi-Process Management](#phase-3-terminal-integration--multi-process-management)
6. [Phase 4: Backend Building & Compilation System](#phase-4-backend-building--compilation-system)
7. [Phase 5: File System Operations & Administrative Access](#phase-5-file-system-operations--administrative-access)
8. [Phase 6: Natural Language Planning & Execution](#phase-6-natural-language-planning--execution)
9. [Phase 7: Git Operations Integration](#phase-7-git-operations-integration)
10. [Phase 8: Backend Code Modification with Safety](#phase-8-backend-code-modification-with-safety)
11. [Phase 9: Agent Statistics & Telemetry System](#phase-9-agent-statistics--telemetry-system)
12. [Phase 10: Integration & Testing](#phase-10-integration--testing)

---

## Executive Summary

This guide provides a comprehensive, phase-by-phase implementation plan for transforming Proxima's AI agent system into a production-grade, real-time monitoring and execution platform. The system will support three AI agents (AI Assistant, AI Analysis, AI Thinking) with full terminal access, backend compilation, file system manipulation, git operations, and code modification capabilities.

**Key Technologies:**
- **UI Framework:** Textual (Python TUI framework)
- **LLM Integration:** Multi-provider support (OpenAI, Anthropic, Google AI, xAI, Ollama, etc.)
- **Terminal Management:** asyncio-based subprocess management with PTY (pseudo-terminal) support
- **Real-time Updates:** Event-driven architecture with reactive data streams
- **Safety Systems:** Checkpoint-based rollback, consent management, audit logging

**Reference Implementation:** Charm's Crush AI Agent (https://github.com/charmbracelet/crush)

---

## Architecture Overview

### System Components

1. **Agent Controller Layer**
   - Centralized orchestration of all agent operations
   - Tool registry and execution dispatcher
   - Session management with persistent state
   - Consent and safety management

2. **Terminal Executor Layer**
   - Cross-platform shell abstraction (PowerShell, bash, zsh)
   - Asynchronous process spawning with PTY
   - Real-time output streaming with line buffering
   - Session pooling for resource efficiency

3. **UI Presentation Layer**
   - Textual-based TUI with reactive widgets
   - Split-pane layouts with resizable panels
   - Real-time log viewers with auto-scroll
   - Progress indicators and status displays

4. **Safety & Rollback Layer**
   - File system checkpoints before modifications
   - Operation approval workflows
   - Undo/redo stack management
   - Audit trail logging

### Data Flow Architecture

```
User Input → LLM Provider → Tool Selection → Agent Controller → 
Terminal Executor → Real-time Output Stream → UI Update → 
Safety Layer (if needed) → Result Storage → Stats Update
```

---

## Phase 1: Real-Time Execution & Results Monitoring

**Objective:** Transform the Execution and Results tabs into hardcore real-time monitoring systems that display live output from all running processes.

### Step 1.1: Implement Real-Time Event Bus System

**Technical Approach:**
- Create an event bus using Python's `asyncio.Queue` for non-blocking message passing
- Design event types: `PROCESS_STARTED`, `OUTPUT_LINE`, `ERROR_LINE`, `PROCESS_COMPLETED`, `PROGRESS_UPDATE`, `STAGE_CHANGED`
- Implement event listeners that can subscribe to specific event types
- Use weak references to prevent memory leaks from orphaned listeners

**Libraries:**
- `asyncio` - Core async primitives
- `typing` - Type hints for event contracts
- `dataclasses` - Event data structure definitions
- `weakref` - Weak reference management

**Key Design Decisions:**
- Events should be immutable dataclass instances
- Each event carries a timestamp, source ID, and payload
- Event bus maintains a circular buffer (max 10,000 events) for replay
- Events are dispatched on the asyncio event loop

### Step 1.2: Refactor Execution Screen for Real-Time Updates

**Technical Approach:**
- Replace polling-based updates with reactive updates triggered by events
- Implement a `RichLog` widget from Textual that auto-scrolls with new content
- Use Textual's `reactive` decorators for automatic UI updates when data changes
- Create a connection between the event bus and the UI update cycle
- Implement debouncing for high-frequency updates (max 60 FPS)

**Textual Components:**
- `RichLog` - Rich text log viewer with syntax highlighting
- `ProgressBar` - Indeterminate and determinate progress bars
- `DataTable` - Tabular data for process lists
- `Static` with `reactive` - For dynamic text updates

**Update Strategy:**
- Batch UI updates every 16ms (60 FPS) to prevent UI blocking
- Use Textual's `call_later` for deferred updates
- Implement virtual scrolling for logs exceeding 10,000 lines
- Store full log in memory but only render visible window

### Step 1.3: Implement Process Output Streaming

**Technical Approach:**
- Use `asyncio.create_subprocess_exec` for non-blocking process spawning
- Set `stdout=asyncio.subprocess.PIPE` and `stderr=asyncio.subprocess.PIPE`
- Create separate async tasks for reading stdout and stderr streams
- Implement line-based buffering using `StreamReader.readline()`
- Detect ANSI escape codes and strip or interpret them for rendering

**Stream Handling:**
- Use `asyncio.StreamReader` for non-blocking line reading
- Implement timeout on readline to prevent hung processes
- Detect process termination via returncode checking
- Capture both stdout and stderr separately but merge in UI

**ANSI Code Handling:**
- Use `rich.console.Console.render_str()` for ANSI interpretation
- Strip codes if terminal doesn't support colors
- Map ANSI colors to Textual theme colors

### Step 1.4: Create Multi-Process Monitor Widget

**Technical Approach:**
- Design a split-pane widget showing multiple terminal outputs simultaneously
- Implement a `MultiTerminalView` container with dynamic columns
- Allow up to 4 concurrent terminal views in 2x2 grid layout
- Each terminal view has independent scrolling and search
- Highlight active terminal with colored border

**Textual Layout:**
- Use `Horizontal` and `Vertical` containers for grid layout
- Implement `ScrollableContainer` for each terminal pane
- Use `TabbedContent` as alternative to grid for many terminals
- Implement terminal switcher with keyboard shortcuts (Alt+1-9)

**State Management:**
- Each terminal has unique ID and associated metadata
- Store terminal state: command, start time, status, output buffer
- Implement terminal history (last 20 terminals) for quick access
- Auto-close completed terminals after configurable timeout

### Step 1.5: Enhance Results Tab with Real-Time Data

**Technical Approach:**
- Refactor results tab to subscribe to execution completion events
- Display results as they arrive, not after all processes complete
- Implement streaming results viewer for long-running benchmarks
- Show partial results with "in-progress" indicators

**Data Structures:**
- Use `list` with `append()` for incremental result accumulation
- Store results in SQLite database for persistence and queries
- Index by execution ID, timestamp, backend name
- Support JSON export of results at any time

**UI Components:**
- `DataTable` for tabular results with sortable columns
- `PlotextPlot` from textual-plotext for real-time graphs
- `ProgressBar` for per-result completion status
- Collapsible sections for detailed result inspection

---

## Phase 2: Agent UI/UX Enhancements

**Objective:** Improve the agent UI to be more professional, user-friendly, and efficient with word wrapping, resizable panels, clean layouts, and toggle-able real-time stats.

### Step 2.1: Implement Word Wrapping for Agent Responses

**Technical Approach:**
- Configure `RichLog` widget to use word wrapping by default
- Set `wrap=True` parameter when creating log widgets
- Use Rich's `Text` object with automatic wrapping
- Implement soft wrapping that respects word boundaries

**Rich Text Configuration:**
- Use `rich.console.Console` with `soft_wrap=True`
- Set maximum width to container width minus padding
- Use `overflow="fold"` to handle long unbreakable strings
- Remove horizontal scrollbar from wrapped containers

**Code-Specific Handling:**
- Use `Syntax` widget from Rich for code blocks with language detection
- Enable line numbers for code snippets
- Use horizontal scrolling ONLY for code blocks (user expects this)
- Wrap natural language responses but not code

### Step 2.2: Add Resizable Panel Sliders

**Technical Approach:**
- Implement custom `ResizableContainer` widget extending Textual's `Container`
- Add drag handles (vertical bars) between panels
- Detect mouse drag events on handles
- Recalculate panel widths dynamically during drag
- Store resize preferences in user settings JSON

**Textual Mouse Handling:**
- Override `on_mouse_down`, `on_mouse_move`, `on_mouse_up` methods
- Detect if click occurred on resize handle (5px width region)
- Update width percentages as mouse moves
- Implement minimum width constraints (20% each panel)

**Keyboard Alternative:**
- Bind `Ctrl+[` and `Ctrl+]` to decrease/increase agent panel width
- Step size: 5% width change per keypress
- Show temporary tooltip with current width percentage

**Settings Persistence:**
- Store panel width ratios in `~/.proxima/tui_settings.json`
- Load saved widths on screen initialization
- Provide "Reset Layout" button to restore defaults

### Step 2.3: Redesign Agent UI for Professional Look

**Technical Approach:**
- Create custom CSS theme specifically for agent panels
- Use consistent spacing: 1-unit padding, 2-unit margins
- Implement color hierarchy: primary (headings), accent (active), muted (metadata)
- Add subtle borders with rounded corners (if terminal supports)
- Use icons from Unicode or Nerd Fonts for visual clarity

**Theme Elements:**
- **Header:** Bold, large text with icon, solid background color
- **Message Bubbles:** Distinct user vs AI styling (border vs background)
- **Tool Execution:** Collapsible sections with status icons
- **Statistics:** Card-based layout with icons and colored numbers
- **Buttons:** Consistent variants (primary, success, warning, error)

**Layout Structure:**
```
┌─ Agent Panel ──────────────────────────────────┐
│ [Icon] AI Assistant          [≡ Menu] [✕ Hide] │ ← Header
├────────────────────────────────────────────────┤
│ ┌─ Stats (collapsible) ────────────────────┐  │
│ │ Provider: OpenAI | Model: gpt-4         │  │
│ │ Messages: 42 | Tokens: 15.2K            │  │
│ └──────────────────────────────────────────┘  │
├────────────────────────────────────────────────┤
│                                                │
│ 👤 User: Build LRET backend                   │ ← Message
│ ─────────────────────────────────────────────  │
│ 🤖 AI: I'll build the LRET Cirq backend...    │
│                                                │
│   ┌─ Tool: build_backend ───────────────┐     │ ← Tool
│   │ ✓ Completed in 45s                  │     │
│   │ Backend: lret_cirq                  │     │
│   └─────────────────────────────────────┘     │
│                                                │
│ [Scroll for more messages...]                 │
├────────────────────────────────────────────────┤
│ [Input field: Ask AI for help...]        [⮕]  │ ← Input
└────────────────────────────────────────────────┘
```

**Textual CSS Best Practices:**
- Use CSS variables for colors defined in theme
- Employ relative units (fr, auto) over fixed pixels
- Implement hover states for interactive elements
- Add focus indicators for keyboard navigation

### Step 2.4: Implement Toggle-able Real-Time Stats Panel

**Technical Approach:**
- Create collapsible stats widget with expand/collapse animation
- Bind keyboard shortcut (Ctrl+T) to toggle visibility
- Add small button in agent header to toggle stats
- Animate collapse using Textual's height transition
- When visible, update stats every 500ms

**Stats Layout:**
- Use grid layout with label-value pairs
- Position stats at top of agent panel, below header
- Use different panel for stats vs messages to prevent overlap
- Stats panel should push message content down, not overlay

**Stats to Display:**
- **LLM Stats:** Provider, model, temperature, max tokens
- **Session Stats:** Messages sent, tokens used, requests made
- **Performance:** Avg response time, uptime, errors
- **Agent Stats:** Tools executed, files modified, commands run
- **Terminal Stats:** Active terminals, completed processes

**Real-Time Update Mechanism:**
- Create reactive properties for all stat values
- Update properties on every event (tool execution, message, etc.)
- Textual automatically re-renders when reactive properties change
- Use `set_interval()` to refresh time-based stats (uptime, elapsed time)

### Step 2.5: Ensure Stats Don't Overlap Other Content

**Technical Approach:**
- Use proper Textual layout constraints (no absolute positioning)
- Stats panel is part of normal document flow
- When stats expand, increase panel height
- When stats collapse, decrease panel height
- Use `Vertical` container to stack stats → messages → input

**Layout Strategy:**
```
Vertical Container (height: 100%)
  ├─ Header (height: 3)
  ├─ Stats Panel (height: auto, max: 10)
  │   └─ Collapsible(visible=True/False)
  ├─ Messages Area (height: 1fr)  ← Takes remaining space
  │   └─ ScrollableContainer
  └─ Input Section (height: auto, min: 5)
```

**Z-Index Management:**
- Don't use overlays for stats
- Use proper container nesting instead
- Only modal dialogs should overlay content
- Stats are part of the normal widget tree

---

## Phase 3: Terminal Integration & Multi-Process Management

**Objective:** Implement robust terminal execution with support for multiple concurrent processes, real-time output streaming, and cross-platform compatibility.

### Step 3.1: Design Terminal Executor Core

**Technical Approach:**
- Create `TerminalExecutor` class that abstracts shell differences
- Detect OS using `platform.system()` (Windows, Linux, Darwin)
- On Windows: Use PowerShell Core (pwsh) or fallback to PowerShell 5.1
- On Unix: Use bash, zsh, or sh based on availability
- Store default shell and working directory per executor instance

**Shell Detection Logic:**
- Windows: Check for `pwsh.exe` in PATH, else use `powershell.exe`
- macOS: Use `zsh` (default since Catalina), fallback to `bash`
- Linux: Check $SHELL environment variable, fallback to `bash`
- Verify shell exists using `shutil.which()` before use

**Working Directory Management:**
- Maintain stack of working directories (pushd/popd semantics)
- Every command execution records current working directory
- Support `cd` command to change working directory for session
- Validate directory exists before changing

**Environment Variables:**
- Inherit parent process environment
- Allow per-session environment variable overrides
- Support setting environment variables in command string
- Persist environment across commands in same session

### Step 3.2: Implement Asynchronous Process Execution

**Technical Approach:**
- Use `asyncio.create_subprocess_exec()` for process spawning
- Pass command through shell with `-c` flag for proper parsing
- Create async tasks for stdout and stderr reading
- Use `asyncio.gather()` to wait for both streams
- Capture return code when process completes

**Subprocess Configuration:**
```python
# Pseudocode structure
process = await asyncio.create_subprocess_exec(
    shell_executable,
    shell_flag,  # '-c' for bash/zsh, '-Command' for PowerShell
    command_string,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    cwd=working_directory,
    env=environment_dict,
    start_new_session=True  # For process group management
)
```

**Stream Reading Strategy:**
- Read line-by-line using `stream.readline()`
- Decode bytes to UTF-8 (handle errors with 'replace' mode)
- Call callback function for each line received
- Continue reading until stream reaches EOF
- Handle partial lines at end of stream

**Timeout Handling:**
- Implement configurable timeout per command (default: 300s)
- Use `asyncio.wait_for()` to enforce timeout
- Send SIGTERM on timeout, then SIGKILL after grace period
- Return timeout error in execution result

### Step 3.3: Create Session Management System

**Technical Approach:**
- Design `AgentSession` dataclass to store session state
- Include fields: id, name, created_at, working_dir, environment, command_history
- Maintain session pool (max 10 concurrent sessions)
- Implement session factory for creating/destroying sessions
- Support session persistence to disk for restoration

**Session Lifecycle:**
1. **Creation:** Generate unique ID, set initial working directory
2. **Active:** Execute commands, update state, append to history
3. **Idle:** Session unused for 5 minutes, can be reclaimed
4. **Destroyed:** Clean up resources, save history to disk

**Session Storage:**
- Store session data in `~/.proxima/agent_sessions/{session_id}.json`
- Include: all commands executed, working directory changes, environment
- Limit stored history to last 100 commands per session
- Allow session import to continue previous work

**Concurrency Control:**
- Use `asyncio.Semaphore` to limit concurrent executions (max 10)
- Queue commands when all slots occupied
- Priority queue: user commands > background tasks > cleanup tasks

### Step 3.4: Implement Multi-Terminal Monitoring

**Technical Approach:**
- Create `MultiTerminalMonitor` class to track all active processes
- Maintain dictionary mapping terminal ID to terminal state
- Each state includes: command, status, output buffer, timestamps
- Implement event emitter to notify UI of changes
- Support filtering and searching across all terminals

**Terminal State Machine:**
- **PENDING:** Command queued but not started
- **STARTING:** Process spawning in progress
- **RUNNING:** Process actively executing
- **COMPLETED:** Process finished successfully (return code 0)
- **FAILED:** Process exited with non-zero return code
- **TIMEOUT:** Process killed due to timeout
- **CANCELLED:** User manually stopped process

**Output Buffering Strategy:**
- Maintain circular buffer per terminal (max 10,000 lines)
- Store each line with metadata: timestamp, stream (stdout/stderr), line number
- Implement efficient line retrieval for UI rendering
- Support streaming buffer to file for very long outputs

**Event System:**
- Emit events for all state changes: `STARTED`, `OUTPUT_RECEIVED`, `COMPLETED`
- Include terminal ID and relevant data in each event
- UI components subscribe to events and update accordingly
- Debounce rapid events (multiple lines in quick succession)

### Step 3.5: Add Cross-Platform Command Normalization

**Technical Approach:**
- Create command translation layer for common operations
- Map generic commands to platform-specific equivalents
- Handle path separators (forward slash vs backslash)
- Convert environment variable syntax ($VAR vs %VAR% vs $env:VAR)

**Command Mappings:**
- List directory: `ls` → `dir` (CMD), `Get-ChildItem` (PowerShell), `ls` (bash/zsh)
- Copy file: `cp` → `copy` (CMD), `Copy-Item` (PowerShell), `cp` (bash/zsh)
- Remove file: `rm` → `del` (CMD), `Remove-Item` (PowerShell), `rm` (bash/zsh)
- Create directory: `mkdir` → `mkdir` (all), but use `-Force` flag in PowerShell

**Path Handling:**
- Normalize all paths using `pathlib.Path`
- Convert to OS-specific format before passing to shell
- Handle relative paths relative to current working directory
- Support home directory expansion (~/ → C:\Users\username\ on Windows)

**Environment Variables:**
- Detect variable references in command string
- Convert to platform-specific syntax before execution
- On Windows PowerShell: Use `$env:VAR_NAME`
- On Unix shells: Use `$VAR_NAME` or `${VAR_NAME}`

---

## Phase 4: Backend Building & Compilation System

**Objective:** Enable AI agents to build and compile quantum simulator backends through terminal commands with proper error handling and progress reporting.

### Step 4.1: Define Backend Build Profiles

**Technical Approach:**
- Create configuration file (YAML or JSON) listing all supported backends
- Each backend profile includes: name, build commands, dependencies, working directory
- Store profiles in `configs/backend_build_profiles.yaml`
- Load profiles on agent initialization

**Backend Profile Structure:**
```yaml
backends:
  lret_cirq:
    name: "LRET Cirq Scalability"
    directory: "src/proxima/backends/lret_cirq"
    build_steps:
      - command: "pip install -r requirements.txt"
        description: "Install dependencies"
      - command: "python setup.py build_ext --inplace"
        description: "Compile Cython extensions"
      - command: "python -m pytest tests/"
        description: "Run tests"
    dependencies: ["python>=3.9", "cirq>=1.0", "cython"]
    platform: "all"
    gpu_required: false
```

**Supported Backends:**
1. **LRET Cirq Scalability:** CPU-based Cirq backend with scaling optimizations
2. **LRET PennyLane Hybrid:** Hybrid quantum-classical backend using PennyLane
3. **LRET Phase 7 Unified:** Unified multi-backend interface
4. **Cirq:** Google's quantum computing framework
5. **Qiskit Aer:** IBM's high-performance simulator
6. **QuEST:** Quantum Exact Simulation Toolkit
7. **qsim:** Google's quantum circuit simulator
8. **cuQuantum:** NVIDIA's GPU-accelerated quantum simulator

### Step 4.2: Implement Build Command Executor

**Technical Approach:**
- Create `BackendBuilder` class that uses `TerminalExecutor`
- Validate backend name against known profiles
- Load build profile and resolve paths
- Execute each build step sequentially
- Stream output to UI in real-time

**Build Process Flow:**
1. **Pre-build Validation:**
   - Check if backend directory exists
   - Verify required dependencies are available
   - Ensure sufficient disk space for build artifacts
   
2. **Dependency Installation:**
   - Parse requirements file or dependencies list
   - Install using pip, conda, or system package manager
   - Handle virtual environment activation if needed

3. **Compilation:**
   - Execute build commands in correct order
   - Set appropriate environment variables (CC, CFLAGS, etc.)
   - Handle platform-specific compilation flags
   - Compile Cython, C++, CUDA code as needed

4. **Post-build Verification:**
   - Run test suite to verify build
   - Check for expected output files (shared libraries, Python modules)
   - Report build statistics (time, artifact size)

**Error Handling:**
- Detect compilation errors from stderr
- Parse error messages to extract file:line information
- Provide helpful suggestions based on common error patterns
- Support build retry with clean workspace

### Step 4.3: Add GPU Backend Support

**Technical Approach:**
- Detect GPU availability using appropriate APIs
- For CUDA: Use `nvidia-smi` command to check GPU presence
- For ROCm: Use `rocm-smi` command for AMD GPUs
- Only offer GPU backends if hardware detected
- Set CUDA/ROCm environment variables before compilation

**GPU Detection:**
- **NVIDIA:** Run `nvidia-smi --query-gpu=name,memory.total --format=csv`
- **AMD:** Run `rocm-smi --showproductname`
- Parse output to determine GPU model and memory
- Store GPU info in agent context for reference

**CUDA Compilation:**
- Set `CUDA_HOME` environment variable to CUDA toolkit path
- Add CUDA bin and lib directories to PATH/LD_LIBRARY_PATH
- Specify compute capability flags for nvcc compiler
- Link against cuBLAS, cuSPARSE for numerical operations

**cuQuantum Integration:**
- Check for cuQuantum library installation
- Set `CUQUANTUM_ROOT` environment variable
- Use cuStateVec for state vector simulation
- Use cuTensorNet for tensor network contraction

### Step 4.4: Implement Build Progress Tracking

**Technical Approach:**
- Parse build output for progress indicators
- Detect compilation steps (compiling X.cpp, linking Y.so)
- Estimate total build time based on previous builds
- Update progress bar in UI as build progresses

**Progress Estimation:**
- Count total build steps from profile
- Assign weight to each step (dependency install: 20%, compile: 60%, test: 20%)
- Update progress percentage as each step completes
- Show indeterminate progress during long-running steps

**Output Parsing:**
- Use regex to detect compilation messages: `Compiling (\w+\.cpp)`
- Detect link messages: `Linking (\w+\.so)`
- Detect test execution: `Running test: (\w+)`
- Extract warnings and errors for summary report

**Build Caching:**
- Check if backend already built (check for output files)
- Offer to skip rebuild if no source changes detected
- Use file modification times to determine if rebuild needed
- Implement `--force-rebuild` flag to override cache

### Step 4.5: Add Build Artifact Management

**Technical Approach:**
- Store build artifacts in dedicated directory (`build/`)
- Organize by backend name and timestamp
- Keep last 3 successful builds for rollback
- Clean old builds automatically to save space

**Artifact Storage Structure:**
```
build/
├── lret_cirq/
│   ├── 20260201_143022/  # Latest build
│   │   ├── lib/
│   │   ├── logs/
│   │   └── manifest.json
│   ├── 20260131_092011/  # Previous build
│   └── 20260130_151405/
├── cirq/
└── qiskit/
```

**Build Manifest:**
- JSON file with build metadata: timestamp, duration, success status
- List of all generated files with checksums
- Compilation flags used
- Test results summary

---

## Phase 5: File System Operations & Administrative Access

**Objective:** Grant AI agents secure access to the local file system with proper permissions, safety checks, and administrative privilege handling.

### Step 5.1: Implement Safe File Operations

**Technical Approach:**
- Create `FileSystemOperations` class with methods for read, write, list, delete
- Implement path validation to prevent directory traversal attacks
- Use `pathlib.Path.resolve()` to normalize and validate paths
- Check file permissions before operations
- Implement operation whitelist and blacklist

**Path Validation:**
- Resolve all paths to absolute paths
- Ensure path is within allowed directories (project root, temp directory)
- Reject paths containing `..` after normalization
- Block access to system directories (/etc, /sys, C:\Windows\System32)
- Create allowed path list in configuration

**File Reading:**
- Implement configurable max file size (default: 10MB)
- Use encoding detection for text files (chardet library)
- Handle binary files gracefully (show hex preview or metadata)
- Support reading file ranges (lines X-Y)
- Stream large files instead of loading entirely in memory

**File Writing:**
- Create backup before overwriting existing files
- Use atomic write operations (write to temp, then rename)
- Set appropriate file permissions after creation
- Verify write success by reading back content checksum

**Directory Operations:**
- List directory with metadata: size, modification time, permissions
- Support recursive directory listing with max depth
- Implement file search with glob patterns
- Exclude hidden files by default (show with flag)

### Step 5.2: Add Administrative Privilege Handling

**Technical Approach:**
- Detect current privilege level on startup
- On Windows: Use `ctypes.windll.shell32.IsUserAnAdmin()`
- On Unix: Check if effective UID is 0 (root)
- Prompt user for elevation when needed
- Provide non-privileged alternatives where possible

**Elevation Mechanisms:**
- **Windows:** Use `ShellExecute` with `runas` verb to launch elevated process
- **macOS/Linux:** Use `sudo` command with password prompt
- **UAC Bypass:** Store credentials temporarily in secure memory (not on disk)
- **Alternative:** Request one-time admin approval for sensitive operations

**Privilege Checking:**
- Before privileged operations, check if elevation needed
- Show clear warning to user about privileged operation
- Require explicit consent (not just "allow all")
- Log all privileged operations to audit log

**Privileged Operations:**
- Installing system packages (apt, yum, brew)
- Writing to system directories
- Modifying system services
- Changing file ownership or permissions
- Binding to privileged ports (<1024)

### Step 5.3: Implement File System Monitoring

**Technical Approach:**
- Use `watchdog` library for cross-platform file system monitoring
- Monitor project directories for changes
- Detect file creation, modification, deletion, move events
- Notify agent when relevant files change
- Trigger automatic actions based on events (e.g., rebuild on source change)

**Watchdog Configuration:**
- Create `FileSystemWatcher` class wrapping watchdog's Observer
- Set up recursive monitoring on project root
- Filter events by file patterns (ignore .git, __pycache__, etc.)
- Implement rate limiting to handle bursts of events

**Event Handling:**
- On file created: Update internal file index
- On file modified: Invalidate cached file contents
- On file deleted: Remove from index, notify if important file
- On file moved: Update references in project database

**Smart Rebuild Triggers:**
- Detect changes to source files (.py, .cpp, .cu)
- Automatically suggest rebuild when backend sources change
- Show diff of what changed since last build
- Offer to run tests after automatic rebuild

### Step 5.4: Add File Content Search and Analysis

**Technical Approach:**
- Implement code search using `ripgrep` (rg) for speed
- Support regex and literal string search
- Search across multiple files simultaneously
- Show context around matches (3 lines before/after)
- Highlight matches in UI with syntax highlighting

**Search Features:**
- **Full-text search:** Search all files in project
- **Code-aware search:** Respect language syntax (skip comments, strings)
- **Semantic search:** Use embeddings to find similar code (optional)
- **File type filtering:** Search only .py, .cpp, etc.
- **Case sensitivity:** Toggle case-sensitive/insensitive search

**Code Analysis:**
- Parse Python files with `ast` module for structure
- Extract function/class definitions
- Identify imports and dependencies
- Calculate code metrics (lines of code, complexity)
- Detect code smells and anti-patterns

### Step 5.5: Implement Secure Temporary File Management

**Technical Approach:**
- Use Python's `tempfile` module for secure temp file creation
- Create agent-specific temp directory on startup
- Clean up temp files on agent shutdown
- Implement auto-cleanup for temp files older than 24 hours

**Temp File Usage:**
- Store intermediate build artifacts
- Cache downloaded files
- Store large command outputs
- Create scratch space for file modifications

**Security Considerations:**
- Set restrictive permissions on temp files (owner read/write only)
- Use unpredictable filenames to prevent race conditions
- Ensure temp directory is on encrypted volume if available
- Sanitize filenames to prevent injection attacks

---

## Phase 6: Natural Language Planning & Execution

**Objective:** Enable agents to understand natural language requests, create execution plans, and autonomously execute multi-step tasks.

### Step 6.1: Design Task Planner System

**Technical Approach:**
- Create `TaskPlanner` class that uses LLM for plan generation
- Given user request, generate structured execution plan
- Plan consists of sequential or parallel steps
- Each step specifies tool to use and arguments
- Validate plan feasibility before execution

**Plan Generation Process:**
1. **Intent Recognition:** Classify user request into categories (build, analyze, modify)
2. **Dependency Analysis:** Identify prerequisites and order of operations
3. **Tool Selection:** Choose appropriate tools for each step
4. **Argument Extraction:** Extract values for tool parameters from request
5. **Validation:** Ensure all required parameters present and valid

**Plan Structure:**
```json
{
  "plan_id": "plan_20260201_143022",
  "description": "Build LRET Cirq backend and run tests",
  "steps": [
    {
      "step_id": 1,
      "tool": "execute_command",
      "arguments": {"command": "pip install cirq"},
      "description": "Install Cirq dependency",
      "depends_on": [],
      "estimated_duration": 30
    },
    {
      "step_id": 2,
      "tool": "build_backend",
      "arguments": {"backend_name": "lret_cirq"},
      "description": "Build LRET Cirq backend",
      "depends_on": [1],
      "estimated_duration": 120
    }
  ],
  "parallel_groups": [[1], [2]],
  "estimated_total_duration": 150
}
```

**LLM Prompt Engineering:**
- Provide LLM with tool descriptions and examples
- Use few-shot learning with example plans
- Request structured JSON output
- Validate JSON against schema
- Re-prompt if plan is invalid or incomplete

### Step 6.2: Implement Plan Execution Engine

**Technical Approach:**
- Create `PlanExecutor` class to run generated plans
- Execute steps in dependency order
- Support parallel execution of independent steps
- Monitor progress and update UI
- Handle step failures gracefully

**Execution Strategy:**
- Build dependency graph from `depends_on` relationships
- Use topological sort to determine execution order
- Execute steps without dependencies first
- Once step completes, trigger dependent steps
- Use `asyncio.gather()` for parallel execution

**Progress Tracking:**
- Calculate overall progress as percentage of completed steps
- Weight steps by estimated duration for accurate progress
- Update UI with current step and status
- Show step-by-step breakdown in collapsible section

**Error Handling:**
- On step failure, mark step as failed
- Determine if subsequent steps can still execute
- Offer to skip failed step and continue
- Provide option to retry failed step with modifications

**Plan Adaptation:**
- If step fails, ask LLM to revise plan
- Allow user to modify plan mid-execution
- Support inserting new steps between existing steps
- Enable plan pause, resume, and abort

### Step 6.3: Add Natural Language Command Parser

**Technical Approach:**
- Create command parser that extracts intent and parameters
- Use LLM to interpret ambiguous requests
- Support common command patterns (build X, run Y, show Z)
- Fall back to LLM if pattern matching fails

**Pattern Matching:**
- Regex patterns for common commands: `^build (.+)$`, `^run (.+)$`
- Extract backend name, script name, file path from patterns
- Validate extracted values against known entities
- Provide suggestions if entity not recognized

**LLM Fallback:**
- If no pattern matches, send request to LLM
- Ask LLM to categorize as: tool execution, question, clarification needed
- Extract tool name and arguments from LLM response
- Confirm with user before executing ambiguous commands

**Example Mappings:**
```
"build lret cirq" → build_backend(backend_name="lret_cirq")
"show git status" → git_status(path=".")
"run tests for cirq" → execute_command(command="python -m pytest tests/test_cirq.py")
"what files changed?" → git_status(path=".") + interpret_results
```

**Clarification Handling:**
- If request ambiguous, ask follow-up questions
- Provide multiple choice options when applicable
- Remember context from previous messages
- Support multi-turn dialogue for complex requests

### Step 6.4: Implement Script Execution Framework

**Technical Approach:**
- Support executing scripts in multiple languages: Python, Bash, PowerShell, JavaScript
- Auto-detect script language from file extension or shebang
- Set up appropriate interpreter and environment
- Pass arguments to scripts
- Capture and parse script output

**Language Support:**
- **Python:** Use same Python interpreter as Proxima (`sys.executable`)
- **Bash:** Use bash shell on Unix, Git Bash or WSL on Windows
- **PowerShell:** Use pwsh or powershell
- **JavaScript:** Use Node.js if available
- **Lua:** Use Lua interpreter if available (for embedded scripts)

**Script Execution:**
1. **Validation:** Check if script file exists and is readable
2. **Permission Check:** Ensure script has execute permission (chmod +x on Unix)
3. **Interpreter Selection:** Choose interpreter based on extension/shebang
4. **Environment Setup:** Set working directory, environment variables
5. **Execution:** Run script with `TerminalExecutor`
6. **Result Capture:** Collect stdout, stderr, return code

**Argument Passing:**
- Pass arguments as command-line parameters
- Support both positional and named arguments
- Handle spaces and special characters in arguments
- Support reading arguments from files (for large argument lists)

**Script Output Handling:**
- Parse JSON output for structured results
- Detect and extract specific patterns (e.g., test results)
- Stream output to UI in real-time
- Store full output in execution history

### Step 6.5: Add Context-Aware Command Suggestions

**Technical Approach:**
- Track user's recent commands and intents
- Suggest relevant follow-up commands
- Use LLM to generate contextual suggestions
- Learn from user's command patterns over time

**Suggestion Generation:**
- After executing command, analyze result
- Suggest natural next steps (e.g., after build, suggest run tests)
- Show suggestions as clickable buttons or autocomplete
- Rank suggestions by relevance and frequency

**Contextual Examples:**
```
After "build lret_cirq":
  - Suggest: "Run tests for LRET Cirq"
  - Suggest: "Show build logs"
  - Suggest: "Check compilation warnings"

After "git status" shows changes:
  - Suggest: "Commit changes"
  - Suggest: "Show diff"
  - Suggest: "Discard changes"
```

**Learning System:**
- Store command sequences in database
- Identify common patterns (A → B → C)
- Increase suggestion rank for frequently used sequences
- Personalize suggestions per user

---

## Phase 7: Git Operations Integration

**Objective:** Enable agents to perform full git operations including clone, pull, push, commit, branch management, and conflict resolution.

### Step 7.1: Implement Core Git Commands

**Technical Approach:**
- Create `GitOperations` class wrapping git CLI commands
- Execute git commands using `TerminalExecutor`
- Parse git output to structured format
- Detect and handle git errors gracefully

**Git Command Execution:**
- Use `git` command-line tool (ensure it's in PATH)
- Pass `--porcelain` flag for machine-readable output when available
- Set GIT_TERMINAL_PROMPT=0 to disable interactive prompts
- Use `--no-pager` to avoid less/more invocation

**Supported Commands:**
1. **clone:** `git clone <url> [destination]`
2. **pull:** `git pull [remote] [branch]`
3. **push:** `git push [remote] [branch]`
4. **commit:** `git commit -m "message"`
5. **status:** `git status --porcelain`
6. **add:** `git add <files>`
7. **branch:** `git branch`, `git branch <name>`
8. **checkout:** `git checkout <branch>`, `git checkout -b <new_branch>`
9. **merge:** `git merge <branch>`
10. **diff:** `git diff [commit1] [commit2]`
11. **log:** `git log --oneline --graph`
12. **remote:** `git remote add/remove/show`
13. **stash:** `git stash push/pop/list`
14. **reset:** `git reset [--hard/--soft] [commit]`
15. **rebase:** `git rebase <branch>`

### Step 7.2: Add Git Status Parser

**Technical Approach:**
- Parse `git status --porcelain` output
- Extract file statuses: modified, added, deleted, untracked
- Detect merge conflicts
- Show human-readable status summary in UI

**Porcelain Format Parsing:**
```
Porcelain output format:
XY PATH
X  PATH -> NEWPATH

Where X is status in index, Y is status in working tree:
' ' = unmodified
M = modified
A = added
D = deleted
R = renamed
C = copied
U = updated but unmerged
? = untracked
! = ignored
```

**Status Object:**
```python
@dataclass
class GitFileStatus:
    path: str
    status: str  # "modified", "added", "deleted", "untracked", "conflict"
    staged: bool
    index_status: str  # Raw X from XY
    worktree_status: str  # Raw Y from XY
```

**UI Presentation:**
- Group files by status (modified, added, deleted, untracked)
- Color-code each status: red (deleted), green (added), yellow (modified)
- Show summary: "5 files changed, 12 insertions(+), 3 deletions(-)"
- Provide quick action buttons: Stage All, Commit, Discard

### Step 7.3: Implement Git Diff Visualization

**Technical Approach:**
- Execute `git diff` command with `--no-color` or `--color=never`
- Parse unified diff format
- Display side-by-side diff in UI using custom widget
- Highlight added (green) and removed (red) lines

**Diff Parsing:**
- Use `difflib` or regex to parse unified diff output
- Extract hunks (sections of changes)
- Parse line numbers and change types (+/-)
- Store as structured diff object

**Diff Viewer Widget:**
- Create `DiffViewer` Textual widget
- Support three modes: unified, split, inline
- Use syntax highlighting for code files
- Allow navigation between hunks (next/prev buttons)

**Advanced Diff Features:**
- Show word-level diffs within lines (not just line-level)
- Fold unchanged context (show only changed sections)
- Support viewing diffs for specific commits
- Compare branches or tags

### Step 7.4: Add Commit & Push Workflow

**Technical Approach:**
- Implement staged commit workflow
- Support amending previous commits
- Push to remote with progress indication
- Handle authentication (SSH keys, HTTPS credentials)

**Commit Process:**
1. **Stage Files:** Use `git add` for selected files or `git add -A` for all
2. **Verify Staged:** Show list of staged files for confirmation
3. **Compose Message:** Prompt for commit message (with template)
4. **Validate Message:** Ensure message follows conventions (e.g., length limits)
5. **Commit:** Execute `git commit -m "message"`
6. **Confirm:** Show commit hash and summary

**Commit Message Assistance:**
- Provide template based on changes (e.g., "Fix bug in X", "Add feature Y")
- Use LLM to generate commit message from diff
- Enforce commit message conventions (Conventional Commits)
- Store commit message history for reuse

**Push Operation:**
- Detect current branch and tracking remote
- Show push preview: commits to be pushed
- Execute `git push origin <branch>`
- Handle errors: non-fast-forward, rejected push
- Suggest solutions: pull first, force push (with warning)

**Authentication Handling:**
- For SSH: Ensure SSH agent has key loaded
- For HTTPS: Use git credential helper
- Support personal access tokens (PAT)
- Store credentials securely in OS keychain (Windows Credential Manager, macOS Keychain, libsecret on Linux)

### Step 7.5: Implement Branch Management

**Technical Approach:**
- List all branches (local and remote)
- Create new branches
- Switch between branches
- Delete branches safely
- Visualize branch history

**Branch Listing:**
- Execute `git branch -a --format="%(refname:short)"`
- Parse output to extract branch names
- Mark current branch with indicator
- Show remote-tracking branches separately

**Branch Switching:**
- Use `git checkout <branch>` or `git switch <branch>` (Git 2.23+)
- Check for uncommitted changes before switch
- Offer to stash changes if present
- Restore working directory after switch

**Branch Creation:**
- Create from current HEAD: `git branch <name>`
- Create from specific commit: `git branch <name> <commit>`
- Create and switch: `git checkout -b <name>`
- Push new branch to remote: `git push -u origin <name>`

**Branch Visualization:**
- Use `git log --graph --oneline --all --decorate`
- Parse output to tree structure
- Display as ASCII graph in TUI
- Use Unicode box-drawing characters for clean appearance

### Step 7.6: Add Merge Conflict Resolution

**Technical Approach:**
- Detect merge conflicts from git status
- Extract conflict markers from files
- Provide UI for manual resolution
- Support automatic resolution strategies

**Conflict Detection:**
- Parse `git status --porcelain` for "U" status
- List files with conflicts
- Extract conflict sections from files

**Conflict Marker Parsing:**
```
<<<<<<< HEAD
Current branch content
=======
Incoming branch content
>>>>>>> incoming_branch
```

**Resolution UI:**
- Show three-way merge view: base, ours, theirs
- Provide buttons: Accept Ours, Accept Theirs, Edit Manually
- Highlight differences between versions
- Save resolved file and mark as resolved: `git add <file>`

**Automatic Resolution:**
- For simple conflicts (one side only changed), auto-resolve
- Use merge strategies: ours, theirs, union
- Implement smart resolution for common patterns (imports, config)

---

## Phase 8: Backend Code Modification with Safety

**Objective:** Enable agents to modify quantum backend code with full safety mechanisms including consent, backup, undo/redo, and rollback.

### Step 8.1: Implement File Backup System

**Technical Approach:**
- Before any modification, create backup of original file
- Store backups in `.proxima/backups/` directory
- Include timestamp and operation ID in backup filename
- Implement automatic cleanup of old backups (keep last 50)

**Backup Structure:**
```
.proxima/backups/
├── 20260201_143022_abc123/
│   ├── manifest.json
│   ├── lret_cirq/
│   │   └── simulator.py
│   └── lret_cirq/
│       └── optimizer.py
```

**Manifest File:**
```json
{
  "operation_id": "abc123",
  "timestamp": "2026-02-01T14:30:22Z",
  "operation_type": "modify_backend_code",
  "files": [
    {
      "path": "src/proxima/backends/lret_cirq/simulator.py",
      "checksum": "sha256:abcdef...",
      "size": 12345
    }
  ],
  "description": "Optimize state vector allocation"
}
```

**Backup Creation:**
- Use `shutil.copy2()` to preserve metadata (mtime, permissions)
- Calculate SHA256 checksum of original file
- Store relative path from project root
- Create manifest with operation metadata

### Step 8.2: Design Modification Preview System

**Technical Approach:**
- Generate diff preview before applying changes
- Show side-by-side comparison of old vs new code
- Require user approval before actual modification
- Support "dry run" mode that only shows changes

**Preview Generation:**
1. **Parse Request:** Extract file path and modification type (replace, insert, delete)
2. **Read Current Content:** Load file into memory
3. **Apply Modification (in memory):** Generate new content
4. **Generate Diff:** Use `difflib.unified_diff()` to create diff
5. **Display Preview:** Show diff in UI with syntax highlighting

**Modification Types:**
- **Replace:** Find specific text and replace with new text
- **Insert:** Add lines at specific line number
- **Delete:** Remove specific lines or text blocks
- **Append:** Add content to end of file
- **Prepend:** Add content to beginning of file

**Preview Display:**
- Use `DiffViewer` widget from earlier phase
- Show file path and modification summary
- Display lines added (green) and removed (red)
- Show context (5 lines before/after changes)
- Calculate impact: lines changed, new/old file size

### Step 8.3: Implement Consent Management System

**Technical Approach:**
- Create `ConsentManager` class to handle approval requests
- Support multiple consent scopes: once, session, always
- Track approval history in database
- Implement consent revocation

**Consent Types:**
1. **Once:** Approve this specific operation only
2. **Session:** Approve all similar operations in current session
3. **Always:** Approve automatically in future (stored in settings)
4. **Deny:** Reject operation

**Consent Request Structure:**
```python
@dataclass
class ConsentRequest:
    request_id: str
    operation: str  # "modify_backend_code", "execute_command", etc.
    description: str  # Human-readable description
    risk_level: str  # "low", "medium", "high", "critical"
    details: Dict[str, Any]  # Operation-specific details
    timestamp: datetime
```

**Consent Dialog:**
- Show modal dialog blocking UI
- Display operation description and details
- Show risk level with appropriate color coding
- Provide buttons: Approve, Approve Always, Deny
- Include timeout (auto-deny after 5 minutes)

**Consent Storage:**
- Store "always" approvals in `~/.proxima/consent_rules.json`
- Include: operation pattern, granted_at timestamp, granted_by user
- Support wildcard patterns (e.g., "execute_command:git *")
- Provide UI to review and revoke stored consents

### Step 8.4: Add Checkpoint-Based Rollback

**Technical Approach:**
- Create checkpoint before each operation
- Checkpoint includes: all affected files, operation metadata, timestamp
- Maintain checkpoint stack (last 20 operations)
- Support rolling back to any checkpoint

**Checkpoint Creation:**
```python
@dataclass
class Checkpoint:
    id: str
    timestamp: datetime
    operation: str
    description: str
    files: List[FileSnapshot]  # Path, content, checksum
    metadata: Dict[str, Any]
```

**FileSnapshot:**
- Store full file content (for small files < 1MB)
- Store diff only (for large files)
- Include: path, content/diff, checksum, size, mtime

**Rollback Process:**
1. **Select Checkpoint:** Choose checkpoint to restore
2. **Validate Files:** Ensure target files haven't changed since backup
3. **Prompt Confirmation:** Show what will be restored
4. **Restore Files:** Write backup content back to original locations
5. **Verify Restoration:** Check checksums match backup
6. **Update Checkpoint Stack:** Mark newer checkpoints as invalidated

**Rollback Safety:**
- Detect if files modified after checkpoint creation
- Warn user if rollback will lose recent changes
- Offer to create snapshot of current state before rollback
- Implement "rollback of rollback" to undo mistaken rollbacks

### Step 8.5: Implement Undo/Redo Stack

**Technical Approach:**
- Maintain separate undo and redo stacks
- Push operation to undo stack on each modification
- Support sequential undo/redo
- Integrate with checkpoint system

**Undo Stack Structure:**
```python
undo_stack: List[Checkpoint] = []
redo_stack: List[Checkpoint] = []
```

**Undo Operation:**
1. **Pop from Undo Stack:** Get most recent operation
2. **Create Redo Checkpoint:** Save current state to redo stack
3. **Restore Files:** Apply undo checkpoint
4. **Update UI:** Show undo confirmation

**Redo Operation:**
1. **Pop from Redo Stack:** Get last undone operation
2. **Create Undo Checkpoint:** Save current state to undo stack
3. **Restore Files:** Apply redo checkpoint
4. **Update UI:** Show redo confirmation

**Stack Management:**
- Limit stack size to 20 operations each
- Clear redo stack when new modification made after undo
- Provide "Clear History" option to free memory
- Support jumping to specific point in history

**Keyboard Shortcuts:**
- Ctrl+Z: Undo
- Ctrl+Y or Ctrl+Shift+Z: Redo
- Alt+Z: Show undo history
- Bind to undo/redo buttons in UI

### Step 8.6: Add Code Modification Intelligence

**Technical Approach:**
- Use LLM to understand code structure
- Parse code with AST (Abstract Syntax Tree) for Python
- Use Tree-sitter for multi-language parsing
- Intelligent suggestion for modification location

**AST-Based Modification:**
- Parse Python file with `ast` module
- Locate specific function/class by name
- Insert, modify, or delete code at precise AST node location
- Regenerate source code from modified AST

**Tree-sitter Integration:**
- Use Tree-sitter for C++, CUDA, JavaScript parsing
- Build language-specific grammars
- Query syntax nodes with Tree-sitter queries
- Support incremental parsing for large files

**Smart Code Insertion:**
- Detect appropriate indentation level
- Match surrounding code style (spaces vs tabs)
- Auto-import required modules
- Suggest variable names matching conventions

**Modification Templates:**
- Provide templates for common modifications
- "Add new method to class"
- "Add parameter to function"
- "Insert logging statement"
- "Add type annotations"

---

## Phase 9: Agent Statistics & Telemetry System

**Objective:** Implement comprehensive real-time statistics tracking for all agent operations with toggle-able display and minimal performance overhead.

### Step 9.1: Design Telemetry Data Model

**Technical Approach:**
- Create `AgentTelemetry` class to collect all metrics
- Use reactive properties for real-time updates
- Store metrics in time-series structure for historical analysis
- Implement efficient in-memory storage with periodic persistence

**Metrics Categories:**

**1. LLM Metrics:**
- Provider name (OpenAI, Anthropic, etc.)
- Model name (gpt-4, claude-3, etc.)
- Temperature, max tokens settings
- Prompt tokens used (cumulative)
- Completion tokens used (cumulative)
- Total tokens used
- Requests sent
- Requests failed
- Average response time (ms)
- Token rate (tokens/second)
- Cost estimate (if pricing available)

**2. Agent Performance:**
- Uptime (seconds since start)
- Messages processed
- Tools executed (by type)
- Commands executed
- Files modified
- Errors encountered
- Success rate (%)
- Average tool execution time

**3. Terminal Statistics:**
- Active terminals
- Completed processes
- Failed processes
- Total output lines captured
- Current terminal count
- Peak terminal count

**4. Git Operations:**
- Commits made
- Branches created
- Files staged
- Pushes/pulls executed
- Merge conflicts resolved

**5. File Operations:**
- Files read
- Files written
- Files deleted
- Bytes read
- Bytes written
- Search operations

**6. Backend Building:**
- Builds initiated
- Builds succeeded
- Builds failed
- Total build time (seconds)
- Average build time

### Step 9.2: Implement Real-Time Metric Collection

**Technical Approach:**
- Instrument all agent operations to emit metric events
- Use decorator pattern to wrap functions with telemetry
- Collect metrics asynchronously to avoid blocking
- Aggregate metrics in background thread

**Instrumentation Decorator:**
```python
# Pseudocode
@collect_metrics(category="tool_execution")
async def execute_tool(tool_name: str, args: Dict):
    start = time.time()
    try:
        result = await _execute_tool_impl(tool_name, args)
        metrics.record_success(tool_name, time.time() - start)
        return result
    except Exception as e:
        metrics.record_failure(tool_name, str(e))
        raise
```

**Metric Emission:**
- Every operation emits start and end events
- Include: operation type, timestamp, duration, result
- Send events to centralized metric collector
- Collector aggregates and stores metrics

**Aggregation Strategy:**
- Update cumulative counters (total tokens, total requests)
- Maintain moving averages for rates (messages/minute)
- Store recent values in circular buffer (last 100 operations)
- Calculate percentiles (p50, p90, p99) for latencies

### Step 9.3: Create Statistics Display Widget

**Technical Approach:**
- Design `StatsPanel` Textual widget
- Use card-based layout with categories
- Update display every 500ms
- Implement smooth transitions for changing values

**UI Layout:**
```
┌─ Agent Statistics ──────────────────────────┐
│ 🤖 LLM                                      │
│   Provider: OpenAI          Model: gpt-4    │
│   Tokens: 15,234           Requests: 42     │
│   Avg Response: 1.2s       Cost: $0.23      │
├─────────────────────────────────────────────┤
│ ⚙️ Performance                               │
│   Uptime: 2h 15m           Messages: 156    │
│   Tools Run: 89            Success: 97.8%   │
│   Errors: 2                Avg Time: 850ms  │
├─────────────────────────────────────────────┤
│ 🖥️ Terminal                                 │
│   Active: 3                Completed: 47    │
│   Output Lines: 12,405     Processes: 50    │
└─────────────────────────────────────────────┘
```

**Value Formatting:**
- Large numbers: Use K/M/B suffixes (15234 → 15.2K)
- Time durations: Show in appropriate units (seconds, minutes, hours)
- Percentages: Show to 1 decimal place
- Currencies: Format with currency symbol and 2 decimals
- Rates: Show per-minute or per-hour as appropriate

**Real-Time Updates:**
- Subscribe to metric update events
- Update reactive properties when metrics change
- Textual automatically re-renders changed properties
- Use color changes to indicate significant events (spike in errors → red)

### Step 9.4: Implement Toggle Functionality

**Technical Approach:**
- Add keyboard shortcut (Ctrl+I) to toggle stats visibility
- Add button in agent header to toggle
- Animate expand/collapse with CSS transitions
- Store preference in user settings

**Toggle Implementation:**
- Use Textual's `display` property to show/hide
- When hidden: `stats_panel.display = False`
- When shown: `stats_panel.display = True`
- Bind to reactive property for automatic UI update

**Animation:**
- Use CSS transition for height change
- Animate from 0 to full height over 200ms
- Use easing function for smooth motion
- Fade in/out opacity simultaneously

**Persistence:**
- Store visibility preference in settings JSON
- Restore state on agent restart
- Per-agent setting (AI Assistant, AI Analysis, AI Thinking)

**Compact Mode:**
- Offer compact mode showing only key metrics
- Single row with most important stats
- Full mode shows all categories
- Toggle between compact and full with separate shortcut

### Step 9.5: Add Historical Metrics & Graphing

**Technical Approach:**
- Store time-series data in SQLite database
- Implement rolling window (keep last 7 days)
- Create graphs for key metrics over time
- Use Plotext or textual-plotext for terminal-based graphs

**Time-Series Storage:**
```sql
CREATE TABLE metrics (
    timestamp INTEGER PRIMARY KEY,
    metric_name TEXT,
    value REAL,
    metadata TEXT  -- JSON for additional context
);

CREATE INDEX idx_metric_time ON metrics(metric_name, timestamp);
```

**Graph Types:**
- **Line Graph:** Tokens used over time
- **Bar Graph:** Tools executed by type
- **Scatter Plot:** Response time vs token count
- **Histogram:** Distribution of command execution times

**Plotext Integration:**
- Use `plotext.plot()` for line graphs
- Render in Textual `Static` widget
- Update graph every 5 seconds
- Support zoom and pan with keyboard

**Export Features:**
- Export metrics to CSV for external analysis
- Generate HTML report with graphs
- Support Prometheus format for monitoring tools
- Create JSON API endpoint for dashboard integration

### Step 9.6: Implement Performance Monitoring & Alerts

**Technical Approach:**
- Monitor agent performance metrics
- Detect anomalies (response time spike, error rate increase)
- Show alerts in UI when issues detected
- Log alerts for later review

**Anomaly Detection:**
- Calculate baseline metrics (average, stddev)
- Flag values exceeding threshold (mean + 3*stddev)
- Use sliding window for dynamic thresholds
- Implement cooldown to prevent alert spam

**Alert Types:**
- **High Latency:** Response time > 5s
- **Error Spike:** Error rate > 5% in last 10 requests
- **Token Limit:** Approaching token limit (> 80% of max)
- **Memory Usage:** Process memory > 1GB
- **Disk Space:** Available disk < 1GB

**Alert Display:**
- Show notification banner in UI
- Play sound alert (if enabled)
- Log to alert history
- Provide "Dismiss" and "View Details" buttons

**Performance Optimization:**
- If high latency detected, suggest reducing token limit
- If many errors, suggest checking LLM connectivity
- If memory high, offer to clear caches
- Auto-clear old logs to free disk space

---

## Phase 10: Integration & Testing

**Objective:** Integrate all components, ensure seamless interaction, and comprehensively test the entire agent system.

### Step 10.1: Component Integration

**Technical Approach:**
- Wire all modules together through `AgentController`
- Ensure proper dependency injection
- Test inter-component communication
- Validate event flow from source to sink

**Integration Points:**
1. **UI → Agent Controller:** User actions trigger agent operations
2. **Agent Controller → LLM Router:** Route requests to appropriate LLM provider
3. **Agent Controller → Terminal Executor:** Execute commands and capture output
4. **Terminal Executor → Multi-Terminal Monitor:** Stream output to monitoring system
5. **Agent Controller → Safety Manager:** Request consent for dangerous operations
6. **Safety Manager → UI:** Display consent dialogs
7. **Agent Controller → Telemetry System:** Record all operations for statistics
8. **Telemetry System → Stats Widget:** Display real-time metrics

**Dependency Management:**
- Use constructor injection for all dependencies
- Avoid circular dependencies (use interfaces/protocols)
- Implement factory pattern for complex object creation
- Use singleton pattern for shared resources (LLM router, terminal pool)

**Event Flow Validation:**
- Trace event path from generation to consumption
- Ensure no events lost or duplicated
- Verify event ordering (causality)
- Test event bus under high load

### Step 10.2: Unit Testing

**Technical Approach:**
- Write unit tests for each module using pytest
- Mock external dependencies (LLM APIs, file system, subprocess)
- Achieve >80% code coverage
- Use parameterized tests for multiple scenarios

**Test Structure:**
```
tests/
├── agent/
│   ├── test_terminal_executor.py
│   ├── test_session_manager.py
│   ├── test_tools.py
│   ├── test_safety.py
│   ├── test_git_operations.py
│   ├── test_backend_modifier.py
│   ├── test_multi_terminal_monitor.py
│   └── test_agent_controller.py
├── tui/
│   ├── test_agent_widgets.py
│   └── test_agent_ai_assistant.py
└── fixtures/
    ├── mock_llm.py
    ├── mock_subprocess.py
    └── sample_files/
```

**Mocking Strategy:**
- Mock `asyncio.create_subprocess_exec` for terminal tests
- Mock LLM responses with predefined outputs
- Use `pytest-mock` for flexible mocking
- Create fixtures for common test data

**Coverage Tools:**
- Use `pytest-cov` for coverage reports
- Generate HTML coverage report for review
- Identify untested code paths
- Add tests for edge cases

### Step 10.3: Integration Testing

**Technical Approach:**
- Test complete workflows end-to-end
- Use test fixtures for realistic scenarios
- Validate UI updates in response to operations
- Test error handling and recovery

**Test Scenarios:**

**1. Backend Build Workflow:**
- User requests "Build LRET Cirq backend"
- Agent parses request → creates plan → requests consent
- User approves → build starts → progress updates in UI
- Build completes → results shown → stats updated
- Verify: Output captured, progress accurate, stats correct

**2. Git Clone & Modify Workflow:**
- User requests "Clone repo from GitHub and modify X"
- Agent clones repo → checks out branch → modifies file
- Shows diff preview → requests consent → applies changes
- Commits changes → pushes to remote
- Verify: Repo cloned, file modified correctly, commit created

**3. Multi-Terminal Monitoring:**
- Start 4 concurrent backend builds
- Verify all terminals show output
- Verify output correctly attributed to each terminal
- Check performance under load
- Verify stats show 4 active terminals

**4. Error Recovery:**
- Trigger operation that fails
- Verify error captured and displayed
- Check rollback functionality
- Ensure agent remains operational after error

**Test Automation:**
- Use `pytest-bdd` for behavior-driven tests
- Write Gherkin scenarios for user stories
- Automate UI testing with Textual's `pilot` API
- Run integration tests in CI/CD pipeline

### Step 10.4: Performance Testing

**Technical Approach:**
- Test agent under high load
- Measure response times and throughput
- Identify bottlenecks
- Optimize hot paths

**Performance Scenarios:**

**1. High-Frequency Operations:**
- Execute 100 commands in rapid succession
- Measure average execution time
- Check for memory leaks
- Verify UI remains responsive

**2. Large Output Handling:**
- Execute command generating 100K lines of output
- Measure memory usage
- Verify scrolling performance
- Check log truncation works

**3. Concurrent Operations:**
- Run 10 tools simultaneously
- Measure resource utilization (CPU, memory)
- Verify no deadlocks or race conditions
- Check event bus handles concurrent events

**4. Long-Running Operations:**
- Start operation that runs for 10 minutes
- Verify progress updates continue
- Check UI doesn't freeze
- Test cancellation works

**Profiling:**
- Use `cProfile` for Python profiling
- Identify slow functions
- Use `line_profiler` for line-level analysis
- Optimize critical paths (event dispatch, UI updates)

**Benchmarks:**
- Set performance targets: < 100ms tool execution overhead
- Measure against targets
- Track performance over time
- Prevent performance regressions in CI

### Step 10.5: User Acceptance Testing

**Technical Approach:**
- Create test scenarios based on user stories
- Conduct usability testing with real users
- Gather feedback on UI/UX
- Iterate based on findings

**Test Users:**
- Quantum computing researchers (primary users)
- Backend developers
- Novice users (for accessibility)
- Power users (for advanced features)

**Usability Metrics:**
- Task completion rate
- Time to complete tasks
- Error rate
- User satisfaction (survey)
- Net Promoter Score (NPS)

**Feedback Collection:**
- In-app feedback button
- User surveys after testing sessions
- Analytics on feature usage
- Bug reports via GitHub issues

**Iteration Process:**
1. Identify pain points from feedback
2. Prioritize issues by impact and frequency
3. Implement fixes and improvements
4. Re-test with users
5. Repeat until satisfaction threshold met

### Step 10.6: Documentation & Deployment

**Technical Approach:**
- Write comprehensive documentation
- Create video tutorials
- Set up automated deployment
- Monitor production usage

**Documentation:**
- **User Guide:** How to use agent features
- **Developer Guide:** Architecture and extension points
- **API Reference:** Tool descriptions and parameters
- **Troubleshooting:** Common issues and solutions
- **FAQ:** Frequently asked questions

**Tutorials:**
- Getting started with AI agent
- Building backends with agent
- Using git operations
- Modifying backend code safely
- Advanced scripting and automation

**Deployment:**
- Package agent module as part of Proxima
- Distribute via pip: `pip install proxima-quantum`
- Create standalone TUI executable for non-Python users
- Set up auto-update mechanism

**Monitoring:**
- Implement telemetry opt-in for usage analytics
- Track feature adoption rates
- Monitor error rates in production
- Collect performance metrics

**Support:**
- Set up community forum
- Create support ticket system
- Provide email support for critical issues
- Maintain changelog for updates

---

## Implementation Timeline

**Total Estimated Duration:** 12-16 weeks

### Phase Breakdown:

- **Phase 1 (Real-Time Execution):** 2 weeks
- **Phase 2 (UI Enhancements):** 2 weeks
- **Phase 3 (Terminal Integration):** 2 weeks
- **Phase 4 (Backend Building):** 1.5 weeks
- **Phase 5 (File System Operations):** 1.5 weeks
- **Phase 6 (Natural Language Planning):** 2 weeks
- **Phase 7 (Git Operations):** 1.5 weeks
- **Phase 8 (Code Modification):** 2 weeks
- **Phase 9 (Statistics System):** 1.5 weeks
- **Phase 10 (Integration & Testing):** 3 weeks

### Milestones:

1. **M1 (Week 4):** Real-time execution and UI enhancements complete
2. **M2 (Week 8):** Terminal integration and backend building working
3. **M3 (Week 12):** All core features implemented
4. **M4 (Week 16):** Testing complete, ready for deployment

---

## Key Technologies & Libraries

### Python Libraries:
- **Textual** (v0.47+): TUI framework
- **Rich** (v13.7+): Terminal text formatting
- **asyncio**: Asynchronous I/O
- **aiohttp**: Async HTTP client
- **pytest** (v7.4+): Testing framework
- **SQLite**: Embedded database
- **watchdog**: File system monitoring
- **gitpython** (optional): Git operations (alternative to CLI)
- **tree-sitter**: Code parsing
- **chardet**: Encoding detection

### External Tools:
- **git**: Version control
- **ripgrep (rg)**: Fast code search
- **PowerShell Core (pwsh)**: Windows shell
- **bash/zsh**: Unix shells

### LLM Providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Google AI (Gemini)
- xAI (Grok)
- Ollama (local models)
- LM Studio (local models)

### Backend Build Tools:
- **pip**: Python package manager
- **cmake**: C++ build system
- **nvcc**: CUDA compiler
- **g++/clang++**: C++ compilers

---

## Success Criteria

The implementation is considered successful when:

1. ✅ All three AI agents (Assistant, Analysis, Thinking) functional
2. ✅ Real-time execution and results monitoring working
3. ✅ Word wrapping and resizable panels implemented
4. ✅ Clean, professional UI with toggle-able stats
5. ✅ Backend building supports all target simulators
6. ✅ File system operations work securely
7. ✅ Natural language command parsing functional
8. ✅ Admin access handling works on all platforms
9. ✅ Script execution supports multiple languages
10. ✅ Multi-terminal monitoring shows real-time output
11. ✅ Git operations complete and reliable
12. ✅ Backend code modification with safety features working
13. ✅ All statistics display in real-time
14. ✅ Test coverage >80%
15. ✅ Performance meets targets (<100ms overhead)
16. ✅ User acceptance testing passed
17. ✅ Documentation complete

---

## Risk Mitigation

### Technical Risks:

**Risk:** LLM API rate limits or failures  
**Mitigation:** Implement retry logic, caching, fallback providers

**Risk:** Terminal command incompatibility across platforms  
**Mitigation:** Extensive testing on Windows, macOS, Linux; command normalization

**Risk:** File system race conditions  
**Mitigation:** Use file locking, atomic operations, careful ordering

**Risk:** Security vulnerabilities in code modification  
**Mitigation:** Sandbox execution, consent system, audit logging

**Risk:** Performance degradation with many concurrent operations  
**Mitigation:** Resource pooling, rate limiting, async design

### User Experience Risks:

**Risk:** UI complexity overwhelming users  
**Mitigation:** Progressive disclosure, good defaults, tutorials

**Risk:** Consent fatigue (too many approval requests)  
**Mitigation:** Smart defaults, "always allow" option, operation grouping

**Risk:** Confusing error messages  
**Mitigation:** Clear, actionable error messages; suggested fixes

---

## Future Enhancements

Beyond initial implementation:

1. **AI Model Fine-Tuning:** Train custom models on Proxima domain
2. **Collaborative Editing:** Multiple users on same agent session
3. **Cloud Sync:** Sync sessions and history across devices
4. **Mobile Companion App:** Monitor executions from mobile
5. **Voice Commands:** Speak commands to agent
6. **IDE Integration:** VS Code extension for Proxima agent
7. **Jupyter Integration:** Use agent in Jupyter notebooks
8. **CI/CD Integration:** Agent as part of build pipeline
9. **Kubernetes Support:** Deploy backends to K8s clusters
10. **Advanced Analytics:** ML-based performance analysis

---

## Conclusion

This comprehensive guide provides all necessary technical details for implementing a production-grade AI agent system for Proxima. The system will transform the quantum computing platform with real-time monitoring, intelligent automation, and a professional user experience.

The phase-by-phase approach ensures steady progress with clear milestones and deliverables. Each phase builds upon previous work, culminating in a fully integrated, tested, and documented system.

Implementation by GPT-5.1 Codex Max, GPT-5.2 Codex Max, or Claude Opus 4.5 following this guide should result in a robust, feature-complete agent system that meets all specified requirements.

---

**Document Version:** 1.0  
**Last Updated:** February 1, 2026  
**Authors:** Proxima Development Team  
**Status:** Ready for Implementation
