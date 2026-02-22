# Agent Functionality Implementation P2

## Comprehensive Phase-by-Phase Implementation Guide

---

## Document Purpose and Core Objective

This document provides a complete, phase-by-phase implementation guide for Proxima's AI agent system. The goal is **not** to make the AI assistant itself dynamically changing in structure or logic. Instead, the assistant must support **dynamically operating integrated models**. Once a model is integrated (via API or local LLM such as Ollama), it must be able to:

- Understand user requests expressed in **any natural language form**
- Dynamically interpret the user's intent
- Correctly map that intent to the capabilities of the integrated model
- Execute the desired task reliably, regardless of how the request is phrased

The assistant's architecture remains stable. The integrated model operates dynamically through natural language understanding and intent-driven execution.

---

## Current System Baseline

### Access Points
- **AI Assistant**: Accessible by pressing the **6** key in Proxima TUI
- **Execution Tab**: Accessible by pressing the **2** key in Proxima TUI
- **Result Tab**: Accessible by pressing the **3** key in Proxima TUI
- The assistant becomes active after a model is successfully integrated through API or local LLM

### Existing Files and Their Roles

**Intent Recognition Layer:**
- `src/proxima/agent/dynamic_tools/robust_nl_processor.py` (1,416 lines): Hybrid rule-based + context-aware natural language processor. Contains `IntentType` enum with 27 intent types, `ExtractedEntity` dataclass, `Intent` dataclass, `SessionContext` for cross-message state tracking, and `RobustNLProcessor` class with pattern-based entity extraction. Currently handles: navigate, list directory, git checkout/clone/pull/status, file create/read/write/delete/copy/move, run command/script, and query operations.
- `src/proxima/agent/dynamic_tools/intent_classifier.py` (630 lines): LLM-reasoning-based intent classifier. Contains a **separate** `IntentType` enum (conflicting with the one in `robust_nl_processor.py`), `ClassifiedIntent` dataclass, and `IntentClassifier` class that uses LLM reasoning instead of keyword matching. Intent types here are broader categories: `FILE_OPERATION`, `DIRECTORY_OPERATION`, `GIT_OPERATION`, `TERMINAL_OPERATION`, `SEARCH_OPERATION`, `ANALYSIS_OPERATION`, `CONFIGURATION`, `INFORMATION_QUERY`, `CONVERSATION`, `MULTI_STEP`, `UNKNOWN`.

**Tool System Layer:**
- `src/proxima/agent/dynamic_tools/tool_interface.py` (984 lines): Abstract base classes and contracts. Defines `ToolInterface`, `BaseTool` (abstract subclass providing standard property-based metadata), `ToolDefinition`, `ToolParameter`, `ToolResult`, `ToolCategory` (29 members: 7 top-level — FILE_SYSTEM, GIT, GITHUB, TERMINAL, BACKEND, ANALYSIS, SYSTEM — plus 22 sub-categories like FILE_READ, FILE_WRITE, GIT_BRANCH, etc.), `PermissionLevel` (READ_ONLY, READ_WRITE, EXECUTE, SYSTEM, NETWORK, ADMIN, FULL_ADMIN), `RiskLevel` (NONE, LOW, MEDIUM, HIGH, CRITICAL), `ExecutionStatus`, `ParameterType` (11 values including STRING, PATH, URL, BRANCH_NAME). Uses Pydantic for validation.
- `src/proxima/agent/dynamic_tools/tool_registry.py` (667 lines): Central registry for dynamic tool discovery. Provides `ToolRegistry` singleton with `RegisteredTool` tracking, semantic search via `ToolSearchResult`, and `@register_tool` decorator for auto-registration. Tracks usage counts and success rates.
- `src/proxima/agent/dynamic_tools/tool_orchestrator.py` (669 lines): Multi-tool execution coordinator. Contains `ExecutionStep`, `ExecutionPlan`, `ExecutionStepStatus`, `ExecutionPlanStatus` dataclasses. Supports dependency resolution, parallel execution via `ThreadPoolExecutor`, and error handling with retry logic.
- `src/proxima/agent/dynamic_tools/execution_context.py` (514 lines): State carrier across tool executions. Contains `GitState`, `TerminalSession`, `FileSystemState` dataclasses. Supports snapshots for rollback capability and serialization for persistence.
- `src/proxima/agent/dynamic_tools/llm_integration.py` (656 lines): Provider-agnostic LLM bridge. Supports OpenAI, Anthropic, Google, Ollama, Azure, Cohere, and Generic via `LLMProvider` enum (7 values). Contains `ToolCall` and `ToolCallResult` dataclasses, `LLMToolConfig`, and `LLMToolIntegration` class. Parses tool calls from various LLM response formats. **Note:** This `LLMProvider` enum is a local enum for tool integration config, separate from the 18+ provider classes in `src/proxima/intelligence/llm_router.py` which handles actual API communication with OpenAI, Anthropic, Ollama, LMStudio, LlamaCpp, Together, Groq, Mistral, Azure, Cohere, Perplexity, GoogleGemini, XAI, DeepSeek, Fireworks, HuggingFace, Replicate, and OpenRouter.

**Tool Implementations:**
- `src/proxima/agent/dynamic_tools/tools/filesystem_tools.py` (909 lines): `ReadFileTool`, `WriteFileTool`, `ListDirectoryTool`, `SearchFilesTool`, `DeleteFileTool`, `MoveFileTool`, `FileInfoTool`. All registered via `@register_tool` decorator. **Note:** `CreateDirectoryTool` does NOT currently exist — directory creation must go through `RunCommandTool` with `mkdir`, or a new tool must be created.
- `src/proxima/agent/dynamic_tools/tools/git_tools.py` (849 lines): `GitStatusTool`, `GitCommitTool`, `GitBranchTool`, `GitLogTool`, `GitDiffTool`, `GitAddTool`. Uses `subprocess` for git command execution with 30-second timeouts. Note: there is no `GitCheckoutTool` or `GitStashTool` — checkout is handled via `GitBranchTool` (action="switch"), and stash goes through `RunCommandTool`.
- `src/proxima/agent/dynamic_tools/tools/terminal_tools.py` (464 lines): `RunCommandTool`, `GetWorkingDirectoryTool`, `ChangeDirectoryTool`, `EnvironmentVariableTool`. Uses PowerShell on Windows, user's shell on Unix. Has `get_shell_info()` for cross-platform shell detection.

**Agent Infrastructure:**
- `src/proxima/agent/task_planner.py` (1,097 lines): LLM-based plan generation from natural language. Contains `TaskCategory` enum (BUILD, ANALYZE, MODIFY, EXECUTE, QUERY, GIT, FILE, SYSTEM, UNKNOWN — 9 values), `StepStatus` enum (PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED, CANCELLED), `PlanStatus` enum (DRAFT, VALIDATED, EXECUTING, COMPLETED, FAILED, PAUSED, CANCELLED), `PlanStep` dataclass with dependency tracking, `ExecutionPlan` dataclass with status tracking, `IntentRecognitionResult` dataclass, and validation/feasibility checking.
- `src/proxima/agent/plan_executor.py` (661 lines): Dependency-ordered execution engine. Supports sequential and parallel modes via `ExecutionMode` enum, plus dry-run mode. Contains `StepResult` and `ExecutionResult` dataclasses with duration tracking.
- `src/proxima/agent/script_executor.py` (758 lines): Multi-language script execution. Supports Python, Bash, PowerShell, JavaScript, Lua, Shell, and Unknown via `ScriptLanguage` enum (7 values). Also has `ScriptSource` enum (FILE, INLINE, STDIN). Auto-detects language from extension and shebang. Contains `ScriptInfo`, `InterpreterInfo`, and `ScriptResult` dataclasses. Has `InterpreterRegistry` class for interpreter discovery and `ScriptExecutor` class.
- `src/proxima/agent/safety.py` (877 lines): Consent and safety system. Defines `ConsentType` (COMMAND_EXECUTION, FILE_MODIFICATION, GIT_OPERATION, ADMIN_ACCESS, NETWORK_ACCESS, BACKEND_MODIFICATION, SYSTEM_CHANGE, BULK_OPERATION), `ConsentDecision` (APPROVED, DENIED, APPROVED_ONCE, APPROVED_SESSION, APPROVED_ALWAYS), `ConsentRequest` dataclass with risk levels (low/medium/high/critical), `ConsentResponse` dataclass, `OperationCheckpoint` dataclass. Also contains `RollbackManager` (checkpoint creation, undo, redo, rollback) and `SafetyManager` (consent workflow, blocked command detection via `is_blocked()`, safe operation checking via `is_safe_operation()`, consent requirement checking via `requires_consent()`, audit logging, session-scoped approvals, and built-in `BLOCKED_PATTERNS` and `SAFE_OPERATIONS` lists).
- `src/proxima/agent/admin_privilege_handler.py` (726 lines): Privilege detection and elevation. Contains `PrivilegeLevel` (STANDARD, ELEVATED, SYSTEM, UNKNOWN — 4 values), `ElevationMethod` (UAC, SUDO, PKEXEC, RUNAS, NONE — 5 values), `OperationCategory` (FILE_SYSTEM, PACKAGE_INSTALL, SERVICE_CONTROL, NETWORK, REGISTRY, PERMISSION, SYSTEM_CONFIG), `PrivilegeInfo` dataclass (with `level`, `is_admin`, `username`, `user_id`, `groups`, `elevation_available`, `elevation_method`, `platform`), `PrivilegedOperation` dataclass, `ElevationResult` dataclass.
- `src/proxima/agent/checkpoint_manager.py` (761 lines): Undo/redo/rollback system. Contains `FileState` (tracks file content and checksums), `Checkpoint` dataclass (with checkpoint ID, timestamp, operation, file states, metadata, rollback status).
- `src/proxima/agent/backend_modifier.py` (800 lines): Safe code modification for quantum backends. Contains `ModificationType` (REPLACE, INSERT, DELETE, APPEND, PREPEND), `CodeChange` dataclass (at line 46) with `get_diff()` method (at line 76, uses `difflib.unified_diff`), `ModificationResult` dataclass. Creates checkpoints via `RollbackManager` before every modification.
- `src/proxima/agent/modification_preview.py` (652 lines): Diff visualization system. Contains `ModificationScope`, `DiffLineType`, `DiffLine`, `DiffHunk`, `ModificationPreview` dataclasses, and `ModificationPreviewGenerator` class. Supports side-by-side diff display with syntax highlighting. Also has a standalone `generate_preview()` convenience function at line 640.
- `src/proxima/agent/backend_builder.py` (1,146 lines): Backend build pipeline. Contains `BuildStatus` enum (PENDING, VALIDATING, INSTALLING_DEPS, BUILDING, TESTING, VERIFYING, COMPLETED, FAILED, CANCELLED), `DependencyCheck` dataclass, `ValidationResult` dataclass, `BuildStepResult` dataclass, `BuildResult` dataclass, `BuildProfileLoader` (loads from `configs/backend_build_profiles.yaml`), `BackendBuilder` class with async build method, GPU detection integration, progress callbacks, and validation. **Note:** `BuildArtifactManager` is imported from a separate module `src/proxima/agent/build_artifact_manager.py`, NOT defined in this file.
- `src/proxima/agent/multi_terminal.py` (1,625 lines): Multi-terminal session management. Contains `TerminalState` enum (PENDING, STARTING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED), `TerminalEventType`, `TerminalEvent`, circular output buffer (10,000 lines), `SessionManager` for 10 concurrent sessions, cross-platform command normalization, event-based state notifications.
- `src/proxima/agent/dynamic_tools/deployment_monitoring.py` (2,999 lines): Deployment and dependency monitoring. Contains `DependencyManager` class (line 1004) focused on vulnerability scanning, license compliance checking, dependency pinning, and update strategy management. Note: this is a different concern from the `ProjectDependencyManager` created in Phase 5 — the existing `DependencyManager` handles security/compliance auditing, while `ProjectDependencyManager` handles project dependency detection, installation, and error-driven auto-fix.
- `src/proxima/agent/dynamic_tools/error_detection.py` (1,263 lines): Error classification and analysis. Contains `ErrorSeverity` enum (6 values: DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL), `ErrorCategory` enum (20 values: FILESYSTEM, NETWORK, AUTHENTICATION, PERMISSION, RESOURCE, TIMEOUT, VALIDATION, CONFIGURATION, DEPENDENCY, GIT, GITHUB, TERMINAL, BUILD, RUNTIME, MEMORY, DISK, SYNTAX, LOGIC, CONCURRENCY, UNKNOWN), `ErrorState` enum (8 values), `ErrorContext`, `ErrorPattern`, `AnalyzedError` dataclasses, and `ErrorClassifier` class with `classify(error: Exception, context: Optional[ErrorContext])` and `classify_with_llm()` (async) methods.
- `src/proxima/agent/terminal_state_machine.py`: Terminal lifecycle state machine for managing terminal process states. Contains `TerminalProcessState` enum (PENDING, STARTING, RUNNING, PAUSED, COMPLETED, FAILED, TIMEOUT, CANCELLED) and `TerminalStateMachine` class. **Note:** This file is in `src/proxima/agent/`, NOT in `src/proxima/agent/dynamic_tools/`.
- `src/proxima/agent/dynamic_tools/entity_extractor.py` (813 lines): LLM-based entity extraction system (separate from the regex-based extraction in `robust_nl_processor.py`). Contains `EntityType` enum (FILE_PATH, DIRECTORY_PATH, GIT_BRANCH, GIT_COMMIT, etc.) and `EntityExtractor` class. Uses LLM reasoning and context-aware inference instead of hardcoded regex patterns. **Note:** This module's `ExtractedEntity` is different from the one in `robust_nl_processor.py` — avoid name collision when importing both.

**TUI Layer:**
- `src/proxima/tui/screens/agent_ai_assistant.py` (7,746 lines): Main AI assistant screen. Contains 5-phase response pipeline in `_generate_response()`: Phase 0 (`_try_direct_backend_operation`) for pattern-matched backend/clone/build requests; Phase 1 (`_try_robust_nl_execution`) using `RobustNLProcessor`; Phase 2 (`_analyze_and_execute_with_llm`) for LLM-based intent extraction; Phase 3 (`_try_execute_agent_command`) for keyword-based agent commands; Phase 4 (`_generate_llm_response`) for general chat. Contains ~145 methods covering all operations.
- `src/proxima/tui/screens/execution.py` (1,913 lines): Execution monitoring screen with progress bars, stage timeline, log viewer, and AI thinking panel. Accessible via key **2**.
- `src/proxima/tui/screens/results.py` (761 lines): Results browser with probability visualization and export engine. Accessible via key **3**.
- `src/proxima/tui/widgets/agent_widgets.py`: Enhanced UI widgets including `MultiTerminalView`.

**Configuration:**
- `configs/backend_build_profiles.yaml` (496 lines): Build profiles for 9 simulator backends (lret_cirq_scalability, lret_pennylane_hybrid, lret_phase_7_unified, cirq, qiskit, quest, qsim, cuquantum, qsim_cuda) plus 5 build preset profiles (minimal, standard, full, gpu, lret_all). Each backend profile contains: name, description, directory, platform, gpu_required, dependencies (python_version, packages, system_packages), build_steps (ordered commands with timeouts), verification (test_import, test_command, expected_files), and error_patterns.

### Known Gaps in Current System
1. **Two conflicting IntentType enums** — `robust_nl_processor.py` and `intent_classifier.py` each define their own `IntentType` enum with different values and semantics.
2. **No backend-specific intents** — The `IntentType` enum in `robust_nl_processor.py` has no entries for `BACKEND_BUILD`, `BACKEND_CONFIGURE`, `BACKEND_TEST`, `BACKEND_MODIFY`, or `BACKEND_LIST`. (Note: `BACKEND_RUN` is covered by `RUN_COMMAND` with backend context; `BACKEND_INSTALL` is covered by `INSTALL_DEPENDENCY` with backend context — no separate intents needed.)
3. **No simulation or analysis intents** — No `ANALYZE_RESULTS`, `EXPORT_RESULTS`. (Note: `SIMULATION_RUN` is covered by `RUN_SCRIPT` or `RUN_COMMAND` with script entities — no separate intent needed.)
4. **No dependency management intents** — No `INSTALL_DEPENDENCY`, `CONFIGURE_ENVIRONMENT`.
5. **Monolithic agent screen** — `agent_ai_assistant.py` at 7,747 lines contains all logic in a single class.
6. **No streaming response support** — Responses are generated as complete blocks, not streamed token-by-token.
7. **No agentic loop** — Current pipeline is single-shot (one intent → one execution), not multi-turn (LLM reasons → executes tool → sees result → reasons again).
8. **Multi-terminal monitoring not connected to NL intent system** — The `MultiTerminalMonitor` exists but is not accessible through natural language commands.
9. **Results tab bridge is limited** — Only `_export_to_results_tab()` exists; no structured pipeline from terminal output to analyzed results.
10. **No execution tab integration from agent** — The Execution Tab (key 2) is not programmatically driven by the agent's terminal operations.

---

## Required Agent Functionalities

The agent must be able to perform the following **dynamically**, based on natural-language user requests:

### 1. Natural-Language Planning and Execution
- Interpret user requests in natural language
- Plan the required steps
- Execute scripts and commands dynamically

### 2. Local File System Access
- Access the local file system dynamically
- Read, write, move, and modify files and directories
- Operate through terminal commands

### 3. Script Execution
- Execute scripts through the terminal
- Access any directory as needed
- Dynamically run scripts based on user requests

### 4. Dependency Management
- Install required dependencies
- Configure environments
- Handle compatibility issues
- Ensure all backends run correctly
- Detect and fix build or compilation errors

### 5. GitHub Repository Operations
Through the terminal, the agent must be able to:
- Clone repositories
- Pull updates
- Push changes

All actions must be dynamic and based on user intent.

### 6. Backend Build, Compilation, and Modification
The agent must be able to build and compile simulator backends via terminal, including:
- LRET cirq scalability
- LRET pennylane hybrid
- LRET phase 7 unified
- Cirq
- Qiskit Aer
- QuEST
- qsim
- cuQuantum
- Other similar backends

The agent must be able to modify code of these or any backends on user request.

**Safety requirements:**
- Explicit user consent before modifications
- Fail-safe mechanisms
- Undo capability
- Redo capability
- Rollback support
- All operations performed through terminal

### 7. Administrative Access
- Request and obtain administrative privileges when required
- Follow safe escalation procedures

### 8. Multi-Terminal Monitoring
The agent must:
- Monitor multiple terminal sessions simultaneously
- Provide live, real-time execution output
- Display execution data in the **Execution Tab** (key **2**)
- Provide comprehensive analysis of results in the **Result Tab** (key **3**)
- All operations must be dynamic and automated

### 9. Complex Natural-Language Task Execution
The agent must dynamically process and execute complex multi-step instructions expressed in any natural language form. See Phase 12 for detailed worked examples (Examples A, B, C).

### 10. Session Context Handling
The agent must:
- Maintain proper session context across the entire conversation lifecycle
- Retain context from imported chats (e.g., loaded from exported JSON files in `exports/`)
- Use session history appropriately during execution — past actions, resolved entities, active terminals, installed packages, and tool results must persist and influence subsequent operations
- Support auto-summarization when the context window approaches its limit, preserving critical state (todo list, modified files, current strategy, exact next steps) so resumed sessions lose no essential information
- Store and retrieve session state to/from persistent storage (JSON files) so sessions survive application restart
- Track per-session metadata: title, message count, token usage, cost, summary message pointer, and active todos

### 11. Additional Agent Capabilities (Crush-Inspired)
The agent must incorporate the following capabilities, adapted from the Crush AI agent architecture (charmbracelet/crush) to Proxima's Python/Textual stack:

**Sub-Agent Delegation:**
- Spawn lightweight read-only sub-agents for search, context gathering, and web research
- Sub-agents operate with a restricted tool set (view, grep, glob, ls, fetch) and cannot modify files
- Sub-agents auto-approve all permission requests within their scoped session
- Parent agent receives sub-agent results as tool responses and continues reasoning

**Web Search and Agentic Fetch:**
- Provide a web search/fetch tool that retrieves and analyzes web content
- Use a sub-agent with a smaller/cheaper model to perform web content analysis
- Save large fetched content (>50KB) to temporary files and return summaries

**Tool Permission System:**
- Every tool execution that modifies state (write, edit, bash execute, delete) requires user consent
- Support three consent responses: Allow (once), Allow for Session, Deny
- Maintain a configurable allowlist of tools that skip permission prompts (e.g., `view`, `ls`, `grep`)
- Support a YOLO mode (`--yolo` flag equivalent) that auto-approves all permissions

**Dangerous Command Blocking:**
- Maintain a blocklist of dangerous commands that are never executed, even with YOLO mode
- Blocklist includes: `sudo rm -rf /`, `mkfs`, `dd if=`, `:(){:|:&};:`, `chmod -R 777 /`, `shutdown`, `reboot`, `halt`, `init 0`, `init 6`, and similar destructive patterns
- When a blocked command is detected, return an error explaining why and suggest a safer alternative

**Structured Todo/Task Tracking:**
- Provide a `todos` tool the agent can call to create and manage a structured task list
- Each todo has: content (imperative form), status (pending/in_progress/completed), and active_form (present continuous for display)
- Enforce exactly one task in_progress at a time
- Persist todos in the session object so they survive context summarization
- Display todo progress in the TUI (e.g., pill widget showing "3/7" completed)

**Dual-Model Architecture:**
- Support configuring a large model (for complex reasoning, code generation) and a small model (for title generation, summarization, sub-agent tasks)
- Default: use the same model for both if only one is configured
- Sub-agents and auto-summarization always prefer the small model for cost efficiency

### 12. Strict Compliance
All instructions in this guide must be followed **strictly and completely** with zero gaps, no vague guidance, and no incomplete instructions.

---

## Phase 1: Unified Intent Taxonomy and Canonical Enum

### Objective
Consolidate the two conflicting `IntentType` enums into a single, comprehensive intent taxonomy that covers every capability the agent needs: natural-language planning, file system access, script execution, dependency management, and GitHub operations.

### Step 1.1: Extend the Canonical IntentType Enum in robust_nl_processor.py

Open the file `src/proxima/agent/dynamic_tools/robust_nl_processor.py`. Locate the `IntentType` enum class at line 31. Currently it has 27 entries. Extend it to cover all required capability domains. Add the following new members to the enum, grouped by domain:

**Terminal and script execution intents (for Functionalities 1, 3, 5):**
- `TERMINAL_MONITOR` — user wants to see active terminal sessions or their output
- `TERMINAL_KILL` — user wants to stop a running terminal process
- `TERMINAL_OUTPUT` — user wants to see the output of a specific terminal
- `TERMINAL_LIST` — user wants to list all active and recent terminals

**Dependency management intents (for Functionality 4):**
- `INSTALL_DEPENDENCY` — user wants to install a Python package, system package, or requirements file
- `CONFIGURE_ENVIRONMENT` — user wants to set up a virtual environment, configure PATH, or set environment variables
- `CHECK_DEPENDENCY` — user wants to verify whether a dependency is installed or check compatibility

**GitHub extended intents (for Functionality 5):**
- `GIT_MERGE` — user wants to merge branches
- `GIT_REBASE` — user wants to rebase a branch
- `GIT_STASH` — user wants to stash or pop changes
- `GIT_LOG` — user wants to view commit history
- `GIT_DIFF` — user wants to see diffs
- `GIT_CONFLICT_RESOLVE` — user wants to resolve merge conflicts

**Search and analysis intents:**
- `SEARCH_FILE` — user wants to search file contents by pattern or text
- `ANALYZE_RESULTS` — user wants the agent to analyze execution output
- `EXPORT_RESULTS` — user wants to export results to a file or the Result Tab

**Plan and execution intents (for Functionality 1):**
- `PLAN_EXECUTION` — user wants the agent to create and present a multi-step plan before executing
- `UNDO_OPERATION` — user wants to undo the last operation
- `REDO_OPERATION` — user wants to redo an undone operation

**Backend build and modification intents (for Functionality 6):**
- `BACKEND_BUILD` — user wants to build or compile a quantum simulator backend
- `BACKEND_CONFIGURE` — user wants to configure Proxima to use a particular backend
- `BACKEND_TEST` — user wants to test or verify a built backend
- `BACKEND_MODIFY` — user wants to modify the source code of a backend
- `BACKEND_LIST` — user wants to list available backends or build profiles

**System intents (for Functionality 7):**
- `SYSTEM_INFO` — user wants system information (Python version, OS, GPU, disk space)
- `ADMIN_ELEVATE` — user wants to run something with elevated privileges

**Web and research intents (for Functionality 11):**
- `WEB_SEARCH` — user wants to search the web or fetch content from a URL

Keep all existing 27 entries unchanged. Only append the new entries. The total enum should have 54 members (27 existing + 27 new).

### Step 1.2: Deprecate the IntentType Enum in intent_classifier.py

Open the file `src/proxima/agent/dynamic_tools/intent_classifier.py`. At the top of the file (line 40), there is a separate `IntentType` enum with values like `FILE_OPERATION`, `GIT_OPERATION`, etc. These are broad categories, not specific intents.

Add an import at the top of `intent_classifier.py` that imports the canonical `IntentType` from `robust_nl_processor`:
```
from .robust_nl_processor import IntentType as CanonicalIntentType
```

Rename the existing `IntentType` in `intent_classifier.py` to `IntentCategory` to avoid collision. This class represents broad categories (file, git, terminal, etc.), which is a different abstraction level from the specific intents in the canonical enum.

Update all internal references within `intent_classifier.py` from `IntentType` to `IntentCategory`.

Add a mapping dictionary at module level in `intent_classifier.py` that maps each `IntentCategory` to a list of `CanonicalIntentType` values. For example, `IntentCategory.FILE_OPERATION` maps to `[CanonicalIntentType.CREATE_FILE, CanonicalIntentType.READ_FILE, CanonicalIntentType.WRITE_FILE, CanonicalIntentType.DELETE_FILE, CanonicalIntentType.COPY_FILE, CanonicalIntentType.MOVE_FILE, CanonicalIntentType.SEARCH_FILE]`. This allows the classifier to narrow from broad category to specific intent.

### Step 1.3: Update All Import Sites

Search the entire codebase for any file that imports `IntentType` from `intent_classifier`. Update those imports to either:
- Import `IntentCategory` (if they use the broad categories), or
- Import `IntentType` from `robust_nl_processor` (if they need specific intents)

Files confirmed affected:
- `src/proxima/agent/dynamic_tools/__init__.py` — **CRITICAL**: This file imports `IntentType` from BOTH `intent_classifier` (line 135) and `robust_nl_processor` (line 638), causing a silent shadowing issue where the `robust_nl_processor` import overwrites the `intent_classifier` one. After renaming, import `IntentCategory` from `intent_classifier` and `IntentType` from `robust_nl_processor`. Export both under their distinct names.

Files NOT affected (verified — they do NOT import `IntentType` from `intent_classifier`):
- `src/proxima/agent/dynamic_tools/llm_integration.py` — does not reference `IntentType`
- `src/proxima/agent/dynamic_tools/tool_orchestrator.py` — does not reference `IntentType`
- `src/proxima/tui/screens/agent_ai_assistant.py` — imports `IntentType` from `robust_nl_processor` (not from `intent_classifier`)

Run a workspace-wide search for `from .intent_classifier import IntentType` and `from ..dynamic_tools.intent_classifier import IntentType` to confirm no other files are affected.

### Step 1.4: Expand Keyword Mappings for New Intents

Open `robust_nl_processor.py` and locate the `_build_intent_keywords()` method. This method returns a dictionary mapping each `IntentType` to a list of trigger phrases. Add entries for every new intent type:

**INSTALL_DEPENDENCY keywords:**
`install`, `pip install`, `install package`, `install dependency`, `install dependencies`, `install requirements`, `pip install -r`, `install module`, `add package`, `install lib`, `install library`, `npm install`, `conda install`, `apt install`, `brew install`

**CONFIGURE_ENVIRONMENT keywords:**
`configure environment`, `setup environment`, `create venv`, `create virtual environment`, `activate venv`, `set env var`, `set environment variable`, `setup python`, `configure path`, `set path`

**CHECK_DEPENDENCY keywords:**
`check dependency`, `check dependencies`, `is installed`, `verify package`, `check if`, `dependency check`, `pip show`, `pip list`, `check version`, `verify installation`

**TERMINAL_MONITOR keywords:**
`monitor terminals`, `show terminals`, `terminal status`, `what is running`, `active processes`, `background jobs`, `running terminals`, `watch terminals`, `show processes`

**TERMINAL_OUTPUT keywords:**
`show output`, `terminal output`, `what did it print`, `show log`, `show terminal`, `output of`, `see output`, `print output`, `display output`

**TERMINAL_KILL keywords:**
`kill terminal`, `stop terminal`, `kill process`, `stop process`, `cancel process`, `terminate`, `abort process`, `end process`, `stop running`

**TERMINAL_LIST keywords:**
`list terminals`, `all terminals`, `terminal list`, `show all terminals`, `how many terminals`

**GIT_MERGE keywords:**
`merge`, `git merge`, `merge branch`, `merge into`, `merge from`

**GIT_REBASE keywords:**
`rebase`, `git rebase`, `rebase on`, `rebase onto`, `rebase from`

**GIT_STASH keywords:**
`stash`, `git stash`, `stash changes`, `pop stash`, `stash pop`, `stash list`, `stash drop`

**GIT_LOG keywords:**
`git log`, `show commits`, `commit history`, `log`, `show log`, `recent commits`, `last commits`, `show history`

**GIT_DIFF keywords:**
`git diff`, `show diff`, `what changed`, `show changes`, `diff with`, `compare`, `see changes`, `see diff`

**SEARCH_FILE keywords:**
`search`, `find in file`, `grep`, `search for`, `look for`, `find text`, `search content`, `search files`

**ANALYZE_RESULTS keywords:**
`analyze`, `analyze results`, `analysis`, `evaluate`, `assess`, `examine results`, `interpret results`, `summarize results`

**EXPORT_RESULTS keywords:**
`export`, `export results`, `save results`, `download results`, `export to`, `save to file`, `write results`

**PLAN_EXECUTION keywords:**
`plan`, `create plan`, `make a plan`, `execution plan`, `step by step`, `steps to`, `plan to`, `plan how`, `plan the`, `what steps`, `how should I`

**UNDO_OPERATION keywords:**
`undo`, `undo that`, `revert`, `revert that`, `take back`, `undo last`, `undo change`, `reverse that`

**REDO_OPERATION keywords:**
`redo`, `redo that`, `redo last`, `do again`, `redo change`, `apply again`

**SYSTEM_INFO keywords:**
`system info`, `system information`, `python version`, `os info`, `gpu info`, `disk space`, `memory`, `cpu info`, `what system`

**ADMIN_ELEVATE keywords:**
`admin`, `administrator`, `sudo`, `run as admin`, `elevate`, `root`, `admin access`, `elevated`, `admin privileges`

**BACKEND_BUILD keywords:**
`build`, `compile`, `build backend`, `compile backend`, `build and compile`, `make`, `cmake`, `setup build`, `build lret`, `build cirq`, `build qiskit`, `build quest`, `build qsim`, `build cuquantum`, `build pennylane`, `compile lret`, `compile cirq`

**BACKEND_CONFIGURE keywords:**
`configure`, `configure backend`, `configure proxima`, `use backend`, `set backend`, `switch backend`, `set as default`, `activate backend`, `configure proxima to use`, `use it`, `set it as backend`

**BACKEND_TEST keywords:**
`test backend`, `verify backend`, `test build`, `run backend tests`, `check backend`, `validate backend`, `test it`, `verify build`, `test the build`

**BACKEND_MODIFY keywords:**
`modify backend`, `change backend code`, `edit backend`, `modify code`, `change source`, `edit source`, `patch backend`, `update backend code`, `modify the backend`, `change the code`, `edit the source`

**BACKEND_LIST keywords:**
`list backends`, `available backends`, `show backends`, `what backends`, `backend list`, `which backends`, `supported backends`, `show build profiles`

**GIT_CONFLICT_RESOLVE keywords:**
`resolve conflict`, `fix conflict`, `merge conflict`, `conflict resolution`, `resolve merge`, `accept theirs`, `accept ours`, `abort merge`, `resolve git conflict`

**WEB_SEARCH keywords:**
`search the web`, `web search`, `google`, `look up online`, `search online`, `fetch url`, `fetch page`, `open url`, `browse`, `search internet`, `find online`

> **⚠️ Keyword Collision Warning:** Several keywords above are intentionally generic to capture natural phrasing, but they risk false matches. During implementation, these MUST be disambiguated using multi-keyword scoring (requiring 2+ keywords to match) or entity context:
> - `build`, `compile`, `make` — only trigger `BACKEND_BUILD` when combined with a backend name entity (e.g., "build cirq", "compile lret"). Standalone "build" without a backend name should fall through to `RUN_COMMAND`.
> - `configure` — only triggers `BACKEND_CONFIGURE` when combined with "backend", "proxima", or a backend name. Standalone "configure" should fall through to `CONFIGURE_ENVIRONMENT`.
> - `search` — only triggers `SEARCH_FILE` when combined with file/content context ("search for X in files"). Generic "search" with a URL context should trigger `WEB_SEARCH`.
> - `log`, `show log` — triggers `GIT_LOG` only when in a git context or combined with "git", "commits". With terminal context ("show terminal log"), triggers `TERMINAL_OUTPUT`.
> - `export` — triggers `EXPORT_RESULTS` only when combined with "results", "data", "analysis". "Export file" should be a file operation.
> - `install` — triggers `INSTALL_DEPENDENCY` in most cases, but "install backend" should trigger `BACKEND_BUILD`.
> - `test` — triggers `BACKEND_TEST` only when combined with "backend". "run tests" or "test the code" should trigger `RUN_COMMAND`.

### Step 1.5: Validate Enum Completeness

After adding all new intent types and keywords, verify that every capability described in the requirements maps to at least one `IntentType`:

| Functionality | IntentTypes |
|---|---|
| 1. NL Planning & Execution | `PLAN_EXECUTION`, `MULTI_STEP`, `RUN_COMMAND`, `RUN_SCRIPT` |
| 2. File System Access | `CREATE_FILE`, `READ_FILE`, `WRITE_FILE`, `DELETE_FILE`, `COPY_FILE`, `MOVE_FILE`, `NAVIGATE_DIRECTORY`, `LIST_DIRECTORY`, `CREATE_DIRECTORY` (requires new tool or `RunCommandTool` with `mkdir` — `CreateDirectoryTool` does not currently exist), `DELETE_DIRECTORY`, `SEARCH_FILE`, `SHOW_CURRENT_DIR` |
| 3. Script Execution | `RUN_SCRIPT`, `RUN_COMMAND`, `NAVIGATE_DIRECTORY` |
| 4. Dependency Management | `INSTALL_DEPENDENCY`, `CONFIGURE_ENVIRONMENT`, `CHECK_DEPENDENCY` |
| 5. GitHub Operations | `GIT_CLONE`, `GIT_PULL`, `GIT_PUSH`, `GIT_COMMIT`, `GIT_ADD`, `GIT_BRANCH`, `GIT_CHECKOUT`, `GIT_FETCH`, `GIT_STATUS`, `GIT_MERGE`, `GIT_REBASE`, `GIT_STASH`, `GIT_LOG`, `GIT_DIFF`, `GIT_CONFLICT_RESOLVE` |
| 6. Backend Build/Compile/Modify | `BACKEND_BUILD`, `BACKEND_CONFIGURE`, `BACKEND_TEST`, `BACKEND_MODIFY`, `BACKEND_LIST` |
| 7. Administrative Access | `ADMIN_ELEVATE`, `SYSTEM_INFO` |
| 8. Multi-Terminal Monitoring | `TERMINAL_MONITOR`, `TERMINAL_OUTPUT`, `TERMINAL_KILL`, `TERMINAL_LIST` |
| 9. Complex NL Tasks | `MULTI_STEP`, `PLAN_EXECUTION` (combined with all above) |
| 10. Query/Status | `QUERY_LOCATION`, `QUERY_STATUS` (existing intents for "where is X" and "what happened" queries) |
| 11. Web/Research | `WEB_SEARCH` (for web content retrieval and research tasks) |

If any functionality lacks a corresponding intent, add it now.

> **Implementation Note — Missing tools to create:** `CreateDirectoryTool` does not exist in `filesystem_tools.py`. Either create it as a new `BaseTool` subclass (recommended — follow the pattern of `ListDirectoryTool`), or map `CREATE_DIRECTORY` to `RunCommandTool` with `mkdir` as the command. `COPY_DIRECTORY` has no tool — map to `RunCommandTool` with `xcopy`/`cp -r`.

---

## Phase 2: Enhanced Entity Extraction and Context Resolution

### Objective
Ensure the NL processor can extract every entity type needed to fulfill the five required functionalities, and can resolve pronouns and contextual references like "it", "that", "the repo" to concrete values from session history.

### Step 2.1: Add New Entity Extraction Patterns

Open `robust_nl_processor.py` and locate the `_compile_patterns()` method and the `extract_entities()` method. Currently, the system extracts: paths, branches, URLs, filenames, and commands. Add extraction patterns for:

**Package names:**
Add a regex pattern that captures package names appearing after dependency-related keywords. The pattern should match text after `pip install`, `install`, `add package`, followed by one or more package specifiers. Each specifier may include version constraints (e.g., `numpy>=1.21.0`). Extract each as `ExtractedEntity(entity_type='package', value='numpy>=1.21.0', confidence=0.9, source='regex')`.

**Script file paths:**
Add a pattern that detects file paths ending in common script extensions: `.py`, `.sh`, `.ps1`, `.bat`, `.js`, `.lua`. These should be extracted as `ExtractedEntity(entity_type='script_path', ...)`. This is more specific than the generic `path` entity and helps the intent recognizer distinguish between "navigate to X" and "run X".

**Environment names:**
Add a pattern for virtual environment references: text like `venv`, `.venv`, `env`, `myenv`, or text after `activate` or `create venv named`. Extract as `ExtractedEntity(entity_type='environment', ...)`.

**Line range patterns:**
Add a pattern for line references in file reading/editing: "lines 10-50", "line 23", "from line 10 to 20". Extract as `ExtractedEntity(entity_type='line_range', value='10-50', ...)`.

**Process identifiers:**
Add a pattern for terminal/process references: "terminal 1", "process 3", "PID 12345", "the build terminal", "first terminal". Extract as `ExtractedEntity(entity_type='process_id', ...)`.

**Quoted content extraction:**
Add a pattern that captures text inside double quotes or single quotes when preceded by prepositions. Classify based on context:
- After `in`, `to`, `at`, `from`, `into`, `under` → `entity_type='path'`
- After `named`, `called`, `with name` → `entity_type='name'`
- After `run`, `execute`, `with command` → `entity_type='command'`
- After `install`, `add` → `entity_type='package'`

### Step 2.2: Enhance SessionContext for Stateful Conversations

Open `robust_nl_processor.py` and locate the `SessionContext` dataclass at line 105. Currently it tracks: `current_directory`, `last_mentioned_paths`, `last_mentioned_branches`, `last_mentioned_urls`, `last_operation`, `operation_history`, `variables`, `cloned_repos`, `last_cloned_repo`, `last_cloned_url`.

Add these new fields:

- `active_terminals: Dict[str, Dict[str, Any]]` — dictionary mapping terminal IDs to their info (command, state, last output lines, PID). Default: empty dict.
- `installed_packages: List[str]` — packages installed during this session. Default: empty list.
- `active_environments: Dict[str, str]` — active virtual environments (name → path). Default: empty dict.
- `last_operation_result: Optional[str]` — the text result of the last executed operation, enabling "it" and "that" resolution for results. Default: None.
- `last_script_executed: Optional[str]` — path of the last script that was run. Default: None.
- `last_mentioned_packages: List[str]` — recently referenced package names (max 10). Default: empty list.
- `conversation_history: List[Tuple[str, str]]` — last 20 pairs of (user_message, intent_type_name) for context inference. Default: empty list.
- `working_directory_stack: List[str]` — directory stack for pushd/popd semantics, enabling "go back" navigation. Default: empty list.
- `last_built_backend: Optional[str]` — name/path of the last backend that was built. Default: None.
- `backend_checkpoints: Dict[str, str]` — maps backend name to the most recent checkpoint ID, enabling targeted rollback per backend. Default: empty dict.
- `last_modified_files: List[str]` — list of files modified in the last backend code modification operation. Default: empty list.

Add a method `add_package(self, package: str)` that inserts into `last_mentioned_packages` (same pattern as existing `add_path`).

Add a method `add_conversation_entry(self, message: str, intent_type: str)` that appends to `conversation_history` and trims to 20 entries.

Add a method `push_directory(self, path: str)` that appends the current directory to `working_directory_stack` and updates `current_directory`.

Add a method `pop_directory(self) -> Optional[str]` that pops from `working_directory_stack` and updates `current_directory`.

### Step 2.3: Implement Pronoun and Reference Resolution

Add a new method `resolve_reference(self, text: str) -> Optional[str]` to the `SessionContext` class. This method resolves contextual references to concrete values:

**Resolution rules (check in order, using surrounding verb context for disambiguation):**
1. If `text` is `"it"` or `"that"`:
   - If the surrounding verb context is action-oriented (`run it`, `execute it`, `build it`, `test it`, `compile it`) → return `self.last_script_executed` or `self.last_cloned_repo` (whichever is more recent), since the user wants to ACT ON something
   - If the surrounding verb context is query-oriented (`show it`, `what is it`, `print it`, `display it`) → return `self.last_operation_result`, since the user wants to SEE a result
   - If no verb context can be determined → return `self.last_operation_result` as fallback
2. If `text` is `"the result"`, `"the output"` → return `self.last_operation_result`
3. If `text` is `"the repo"`, `"the repository"`, `"that repo"` → return `self.last_cloned_repo` (the path) or `self.last_cloned_url`
4. If `text` is `"there"`, `"that directory"`, `"that folder"` → return `self.last_mentioned_paths[0]` if available
5. If `text` is `"that branch"`, `"the branch"` → return `self.last_mentioned_branches[0]` if available
6. If `text` is `"the script"`, `"that script"` → return `self.last_script_executed` if available
7. If `text` is `"that backend"`, `"the backend"` → return `self.last_built_backend` if available
8. If `text` is `"back"`, `"previous directory"` → return result of `self.pop_directory()`
9. Return `None` if no resolution found

The `resolve_reference()` method signature should be `resolve_reference(self, text: str, verb_context: Optional[str] = None) -> Optional[str]` to accept the surrounding verb for disambiguation. The caller in `recognize_intent()` should pass the verb that precedes the pronoun.

### Step 2.4: Integrate Reference Resolution into Intent Recognition

In the `recognize_intent()` method of `RobustNLProcessor`, after entity extraction is complete but before returning the `Intent`, add a post-processing pass that:

1. Iterates through all extracted entities
2. For any entity whose `value` matches a pronoun pattern (it, that, there, the repo, etc.), calls `self._session_context.resolve_reference(entity.value)`
3. If resolution succeeds, replaces the entity's value with the resolved concrete value and sets `entity.source = 'context'`
4. If resolution fails, keeps the original entity but lowers its confidence to 0.3

This ensures that messages like "run it" after a script mention, or "go there" after a directory mention, resolve correctly.

### Step 2.5: Update Context After Every Operation

In `agent_ai_assistant.py`, after every successful tool execution (in each of the 5 phases of `_generate_response`), call `self._session_context.update_from_intent(intent)` and also:
- After script execution: set `self._session_context.last_script_executed = script_path`
- After any operation: set `self._session_context.last_operation_result = result_text`
- After directory navigation: call `self._session_context.push_directory(new_path)`
- After package installation: call `self._session_context.add_package(package_name)` for each installed package
- Always: call `self._session_context.add_conversation_entry(message, intent.intent_type.name)`

---

## Phase 3: Natural-Language Planning and Execution Pipeline

### Objective
Enable the integrated model to interpret any natural language user request, dynamically plan the required steps, and execute scripts and commands — all without hardcoded command mapping.

### Step 3.1: Restructure the Intent Recognition Pipeline

Open `robust_nl_processor.py` and restructure the `recognize_intent()` method into a clearly layered pipeline. Each layer is a fallback if the previous one fails or produces low confidence:

**Layer 1 — High-Priority Pattern Matching (no LLM needed):**
Check for intents that can be unambiguously determined from text patterns alone:
- Messages containing a URL (http/https) combined with "clone" → `GIT_CLONE`
- Messages containing a script file path ending in `.py`, `.sh`, `.ps1`, `.bat`, `.js` combined with "run" or "execute" → `RUN_SCRIPT`
- Messages starting with `cd `, `ls `, `dir `, `pwd`, `mkdir` → the corresponding directory/navigation intent
- Messages containing `pip install`, `npm install`, `conda install`, `apt install` → `INSTALL_DEPENDENCY`
- Messages that are pure queries ("where is", "what is", "did it work", "what happened") → `QUERY_STATUS` or `QUERY_LOCATION`

Use the existing `_is_query_intent()` and `_is_clone_intent()` methods. Add `_is_dependency_intent()` and `_is_script_intent()` following the same pattern.

**Layer 2 — Multi-Step Detection:**
Check for multi-step separators: "then", "after that", "next", "and then", numbered lists (1. ... 2. ... 3. ...), semicolons separating distinct commands. If detected, split the message into sub-steps. Classify each sub-step using Layer 1 and Layer 3. Return `IntentType.MULTI_STEP` with sub-intents stored in a new field on the `Intent` dataclass called `sub_intents: List[Intent]`.

Add the `sub_intents` field to the `Intent` dataclass:
- `sub_intents: List['Intent'] = field(default_factory=list)`

**Layer 3 — Keyword Scoring:**
The existing keyword-based scoring system from `_build_intent_keywords()` (expanded in Phase 1). Score every intent type against the message. Use the longest-matching-keyword-wins strategy. Normalize confidence scores to 0.0–1.0 range. Return the highest-scoring intent if its confidence exceeds 0.5.

**Layer 4 — LLM-Assisted Classification (optional, model-agnostic):**
If Layers 1-3 produce confidence below 0.5, AND an LLM router is available (checked via a flag `self._llm_available`), send a classification request to the integrated model. Structure the prompt as a **multiple-choice question** (not JSON extraction) to work with any model, including small local models:

The prompt format should be:
```
"The user said: '{message}'
Which action best matches? Pick ONE number:
1. {top_candidate_1.name}: {description}
2. {top_candidate_2.name}: {description}
3. {top_candidate_3.name}: {description}
4. {top_candidate_4.name}: {description}
5. None of the above
Answer with just the number:"
```

The top 4 candidates come from Layer 3's sorted scores. Parse the response by looking for a single digit (1-5) in the first line. This avoids JSON parsing failures entirely. If the model returns "5" or an unparseable response, fall through to Layer 5.

**Layer 5 — Context-Based Inference:**
If all above fail, use `SessionContext` to infer intent from conversation flow:
- If last operation was `GIT_CLONE` and user says "build it" → infer `BACKEND_BUILD` if the cloned repo matches a known backend name, otherwise infer `RUN_COMMAND` with a generic build command for the cloned repository
- If user says "run it" → infer `RUN_SCRIPT` using `last_script_executed` or `RUN_COMMAND` using recent context
- If user says "go back" → infer `NAVIGATE_DIRECTORY` with `pop_directory()` result
- If user says "install the dependencies" after cloning → infer `INSTALL_DEPENDENCY` with path context from cloned repo
- Default to `UNKNOWN` only if all 5 layers fail

### Step 3.2: Integrate ExecutionPlanner for Multi-Step Plans

When the recognized intent is `PLAN_EXECUTION` or `MULTI_STEP`, the system must construct and present an execution plan before running anything.

Connect the `RobustNLProcessor`'s multi-step detection to the existing `TaskPlanner` class in `src/proxima/agent/task_planner.py`. The flow:

> **Important — Two `ExecutionPlan` classes exist in the codebase:**
> - `src/proxima/agent/task_planner.py` line 117: `ExecutionPlan` with `PlanStep` objects, `PlanStatus`, and high-level task tracking (title, description, category, steps with dependencies). This is the one to use for multi-step plan presentation and execution.
> - `src/proxima/agent/dynamic_tools/tool_orchestrator.py` line 96: `ExecutionPlan` with `ExecutionStep` objects and `ExecutionPlanStatus`, designed for lower-level tool orchestration with parallel execution.
>
> Use the `task_planner.ExecutionPlan` for the agent's planning pipeline (user-facing plans). Use the `tool_orchestrator.ExecutionPlan` only if delegating to the `ToolOrchestrator` for parallel tool execution. Import them with aliases to avoid collision: `from ..task_planner import ExecutionPlan as TaskExecutionPlan` and `from .tool_orchestrator import ExecutionPlan as ToolExecutionPlan`.

1. `RobustNLProcessor.recognize_intent()` detects `MULTI_STEP` and fills `intent.sub_intents` with ordered sub-intents
2. Create a method `_create_plan_from_intents(self, sub_intents: List[Intent]) -> ExecutionPlan` in the agent assistant that converts the list of `Intent` objects into a `TaskPlanner.ExecutionPlan`:
   - For each `Intent` in `sub_intents`, create a `PlanStep` with:
     - `tool` = the tool name corresponding to the intent type (use the mapping from the `IntentToolBridge` — defined in Step 3.3)
     - `arguments` = extracted entities converted to tool parameters
     - `description` = a human-readable description of what the step does
     - `depends_on` = list of previous step IDs (sequential dependency by default)
3. Present the plan to the user in the chat as a numbered list with descriptions
4. Ask for confirmation: "Shall I execute this plan? (yes/no/modify)"
5. On "yes": pass the plan to the `PlanExecutor` in `src/proxima/agent/plan_executor.py`
6. On "no": cancel and wait for new input
7. On "modify": show the plan in an editable format and re-parse

### Step 3.3: Create a Tool Dispatch Bridge

Create a new file `src/proxima/agent/dynamic_tools/intent_tool_bridge.py`. This module maps every canonical `IntentType` to the appropriate tool executor. It bridges the gap between intent recognition (from `RobustNLProcessor`) and tool execution (from the existing `ToolRegistry`).

The `IntentToolBridge` class must contain:

**A static mapping dictionary `INTENT_TO_TOOL`** that maps each `IntentType` to a tool name string. For example:
- `IntentType.CREATE_FILE` → `"write_file"`
- `IntentType.WRITE_FILE` → `"write_file"`
- `IntentType.READ_FILE` → `"read_file"`
- `IntentType.LIST_DIRECTORY` → `"list_directory"`
- `IntentType.NAVIGATE_DIRECTORY` → `"change_directory"`
- `IntentType.RUN_COMMAND` → `"run_command"`
- `IntentType.RUN_SCRIPT` → `"run_command"` (with script path as the command)
- `IntentType.GIT_CLONE` → `"run_command"` (with `git clone` as the command)
- `IntentType.GIT_PULL` → `"run_command"` (with `git pull`)
- `IntentType.GIT_PUSH` → `"run_command"` (with `git push`)
- `IntentType.GIT_COMMIT` → `"git_commit"` (from `git_tools.py`)
- `IntentType.GIT_STATUS` → `"git_status"` (from `git_tools.py`)
- `IntentType.GIT_CHECKOUT` → `"git_branch"` (from `git_tools.py` — `GitBranchTool` handles checkout/switch via its `action` param)
- `IntentType.GIT_BRANCH` → `"git_branch"` (from `git_tools.py`)
- `IntentType.GIT_LOG` → `"git_log"` (from `git_tools.py`)
- `IntentType.GIT_DIFF` → `"git_diff"` (from `git_tools.py`)
- `IntentType.GIT_ADD` → `"git_add"` (from `git_tools.py`'s `GitAddTool`)
- `IntentType.GIT_STASH` → `"run_command"` (with `git stash` as the command — no dedicated stash tool exists)
- `IntentType.INSTALL_DEPENDENCY` → `"run_command"` (with pip/npm/conda command)
- `IntentType.SEARCH_FILE` → `"search_files"` (from `filesystem_tools.py`)
- `IntentType.ANALYZE_RESULTS` → custom handler (see Phase 9, Step 9.5)
- `IntentType.EXPORT_RESULTS` → custom handler (see Phase 9, Step 9.4)
- `IntentType.PLAN_EXECUTION` → custom handler (uses TaskPlanner)
- `IntentType.UNDO_OPERATION` → custom handler (uses CheckpointManager)
- `IntentType.REDO_OPERATION` → custom handler (uses CheckpointManager)
- `IntentType.ADMIN_ELEVATE` → custom handler (uses AdminPrivilegeHandler)
- `IntentType.BACKEND_BUILD` → custom handler (uses BackendBuilder from `backend_builder.py`, see Phase 6)
- `IntentType.BACKEND_CONFIGURE` → custom handler (writes to `configs/default.yaml`, see Phase 6, Step 6.5)
- `IntentType.BACKEND_TEST` → custom handler (runs verification from YAML profile, see Phase 6, Step 6.4)
- `IntentType.BACKEND_MODIFY` → custom handler (uses BackendModifier + CheckpointManager with consent, see Phase 6, Step 6.2)
- `IntentType.BACKEND_LIST` → custom handler (reads `backend_build_profiles.yaml`, see Phase 6, Step 6.6)
- `IntentType.SHOW_CURRENT_DIR` → `"get_working_directory"` (from `terminal_tools.py`)
- `IntentType.COPY_DIRECTORY` → `"run_command"` (with `xcopy`/`cp -r` command)
- `IntentType.GIT_FETCH` → `"run_command"` (with `git fetch` as the command)
- `IntentType.GIT_MERGE` → `"run_command"` (with `git merge {branch}` as the command)
- `IntentType.GIT_REBASE` → `"run_command"` (with `git rebase {branch}` as the command)
- `IntentType.GIT_CONFLICT_RESOLVE` → custom handler (analyzes conflict files, presents options)
- `IntentType.QUERY_LOCATION` → custom handler (searches for entity in file system, reports path)
- `IntentType.QUERY_STATUS` → custom handler (checks SessionContext for last operation result, reports status)
- `IntentType.SYSTEM_INFO` → `"run_command"` (with platform-appropriate system info commands)
- `IntentType.CHECK_DEPENDENCY` → custom handler (uses ProjectDependencyManager)
- `IntentType.CONFIGURE_ENVIRONMENT` → `"run_command"` (with venv/env var commands)
- `IntentType.TERMINAL_MONITOR` → custom handler (uses TerminalOrchestrator.get_all_terminals)
- `IntentType.TERMINAL_OUTPUT` → custom handler (uses TerminalOrchestrator.get_output)
- `IntentType.TERMINAL_KILL` → custom handler (uses TerminalOrchestrator.kill_terminal)
- `IntentType.TERMINAL_LIST` → custom handler (uses TerminalOrchestrator.get_all_terminals)
- `IntentType.DELETE_DIRECTORY` → `"run_command"` (with `rmdir`/`rm -rf` — requires consent)
- `IntentType.CREATE_DIRECTORY` → `"run_command"` (with `mkdir` command — or new `CreateDirectoryTool` if created)
- `IntentType.WEB_SEARCH` → custom handler (uses `AgenticFetchTool` from Phase 16.2 — map to a no-op or "not yet available" stub until Phase 16 is implemented)

**A method `build_tool_arguments(intent: Intent) -> Dict[str, Any]`** that converts the extracted entities from an `Intent` into the parameter dictionary expected by the target tool. For example:
- For `GIT_CLONE`: extract `url` entity → `{"command": f"git clone {url} {target_path}"}`, extract `branch` entity → append `--branch {branch}`, extract `path` entity → set as clone destination
- For `RUN_SCRIPT`: extract `script_path` entity → build the appropriate interpreter command (Python for `.py`, PowerShell for `.ps1`, etc.) using `ScriptExecutor.detect_language()`
- For `INSTALL_DEPENDENCY`: extract `package` entities → `{"command": f"pip install {' '.join(packages)}"}`
- For file operations: extract `path` and `content` entities into the tool's expected parameter names

**A method `dispatch(intent: Intent, context: ExecutionContext) -> ToolResult`** that:
1. Looks up the tool name from `INTENT_TO_TOOL`
2. Builds arguments via `build_tool_arguments()`
3. Gets the tool instance from `ToolRegistry.get_tool(tool_name)`
4. Calls `tool.execute(arguments, context)`
5. Returns the `ToolResult`
6. For custom handlers (ANALYZE_RESULTS, PLAN_EXECUTION, UNDO, REDO, ADMIN), delegates to the appropriate module directly

### Step 3.4: Integrate the Bridge into agent_ai_assistant.py

In `agent_ai_assistant.py`, modify the `_try_robust_nl_execution()` method to use the new `IntentToolBridge`:

1. After `RobustNLProcessor.recognize_intent()` returns an `Intent` with confidence > 0.5:
2. If the intent type is `MULTI_STEP` or `PLAN_EXECUTION`, call the plan creation flow from Step 3.2
3. Otherwise, call `IntentToolBridge.dispatch(intent, context)` to execute
4. Display the `ToolResult` in the chat
5. Update `SessionContext` with the result

This replaces the current approach where `_try_robust_nl_execution()` calls individual `_execute_*` methods directly on `RobustNLProcessor`. The bridge centralizes all dispatch logic.

### Step 3.5: Handle Complex Multi-Step Natural Language Requests

This step specifically addresses the requirement to handle requests like:

**Example:**
"Clone https://github.com/kunal5556/LRET into C:\Users\dell\Pictures\Screenshots, switch to cirq-scalability-comparison branch, install dependencies, and run the tests"

The pipeline for this:
1. `RobustNLProcessor.recognize_intent()` detects `MULTI_STEP` (multiple operations chained with commas and "and")
2. The multi-step parser splits this into 4 sub-intents:
   - Sub-intent 1: `GIT_CLONE` with entities: url=`https://github.com/kunal5556/LRET`, path=`C:\Users\dell\Pictures\Screenshots`
   - Sub-intent 2: `GIT_CHECKOUT` with entities: branch=`cirq-scalability-comparison`
   - Sub-intent 3: `INSTALL_DEPENDENCY` with no specific packages (system should auto-detect `requirements.txt` in the cloned repo)
   - Sub-intent 4: `RUN_COMMAND` with command=`pytest` or auto-detected test command
3. The `_create_plan_from_intents()` method builds an `ExecutionPlan` with 4 sequential steps
4. Plan is presented to the user for confirmation
5. On confirmation, `PlanExecutor` executes each step in order
6. After each step, `SessionContext` is updated so subsequent steps can resolve references:
   - After step 1: `last_cloned_repo` is set to the clone path
   - After step 2: working directory context is updated
   - After step 3: `installed_packages` is updated
7. Results are displayed incrementally in the chat

For the auto-detection in step 3 (install dependencies without specifying packages): after cloning, the `IntentToolBridge.build_tool_arguments()` for `INSTALL_DEPENDENCY` must check the cloned repository's directory for `requirements.txt`, `setup.py`, `pyproject.toml`, or `package.json`. If `requirements.txt` exists, the command becomes `pip install -r requirements.txt`. If `setup.py` exists, the command becomes `pip install -e .`. This logic goes into a helper method `_detect_dependency_file(repo_path: str) -> Optional[str]` in `intent_tool_bridge.py`.

---

## Phase 4: Local File System Access and Script Execution

### Objective
Enable the integrated model to fully interact with the local file system — read, write, create, delete, move, search — and execute any script from any directory, all through natural language requests processed via the terminal.

### Step 4.1: Enhance File System Tool Executors via the Bridge

The existing tools in `src/proxima/agent/dynamic_tools/tools/filesystem_tools.py` already implement `ReadFileTool`, `WriteFileTool`, `ListDirectoryTool`, `DeleteFileTool`, `MoveFileTool`, `SearchFilesTool`, `FileInfoTool`. These tools already use the `@register_tool` decorator and are discoverable via `ToolRegistry`. **Note:** `CreateDirectoryTool` does NOT exist yet — see Step 4.2 below for creation instructions, or dispatch to `RunCommandTool` with `mkdir`.

In `IntentToolBridge.build_tool_arguments()`, add argument construction logic for each file intent:

**For `READ_FILE`:**
- Extract `path` entity from intent → map to `file_path` parameter
- Extract `line_range` entity (if present) → map to `start_line` and `end_line` parameters
- If no path entity exists, check `SessionContext.last_mentioned_paths[0]` as fallback

**For `WRITE_FILE` / `CREATE_FILE`:**
- Extract `path` entity → `file_path`
- Extract `content` entity → `content`
- If content is not in the intent entities, and the user said something like "create an empty file", set `content` to empty string

**For `DELETE_FILE`:**
- Extract `path` entity → `file_path`
- Before dispatching, trigger a consent request via `ConsentRequest` from `safety.py` with `consent_type=ConsentType.FILE_MODIFICATION` and `risk_level='medium'`

**For `COPY_FILE`:**
- Extract two path entities. The first mentioned is the source, the second is the destination. Map to `source_path` and `destination_path`.

**For `MOVE_FILE`:**
- Same as COPY_FILE but for move operation. Also requires consent.

**For `SEARCH_FILE`:**
- Extract a text pattern entity and optional path entity → `pattern` and `directory` parameters

**For `NAVIGATE_DIRECTORY` (implemented as ChangeDirectoryTool):**
- Extract `path` entity → `directory`
- If the entity resolves to a relative path, prefix with `SessionContext.current_directory`
- After execution, call `SessionContext.push_directory(new_path)`

**For `CREATE_DIRECTORY`:**
- Extract `path` entity → the directory path to create
- Since `CreateDirectoryTool` does not currently exist in `filesystem_tools.py`, dispatch to `RunCommandTool` with command `mkdir -p {path}` (Unix) or `New-Item -ItemType Directory -Path "{path}" -Force` (Windows)
- **Recommended**: Create a dedicated `CreateDirectoryTool` in `filesystem_tools.py` following the `ListDirectoryTool` pattern, using `os.makedirs(path, exist_ok=True)`. Register via `@register_tool`.

**For `LIST_DIRECTORY`:**
- Extract `path` entity → `directory`
- If no path, use `SessionContext.current_directory`

### Step 4.2: Implement Path Resolution Logic

Add a method `resolve_path(raw_path: str, context: SessionContext) -> str` to `IntentToolBridge`. This method:

1. Expands `~` to the user's home directory using `os.path.expanduser()`
2. Expands environment variables using `os.path.expandvars()` (handles `%USERPROFILE%` on Windows, `$HOME` on Unix)
3. If the path is relative (does not start with `/` or `X:\`), joins it with `context.current_directory` using `os.path.join()`
4. Normalizes the path using `os.path.normpath()` to resolve `..` and `.` components
5. Returns the absolute, normalized path

All file and directory tool argument builders must call `resolve_path()` before passing paths to tools.

### Step 4.3: Enhance Script Execution

The existing `ScriptExecutor` in `src/proxima/agent/script_executor.py` already supports Python, Bash, PowerShell, JavaScript, and Lua with auto-detection. Integrate it with the intent bridge:

In `IntentToolBridge`, for `RUN_SCRIPT` intents:

1. Extract the `script_path` entity from the intent
2. Call `resolve_path(script_path, context)` to get the absolute path
3. Call `ScriptExecutor.detect_language(script_path)` to determine the interpreter
4. Build the execution command. Use the `InterpreterRegistry` class (from `script_executor.py`) to look up interpreters — call `InterpreterRegistry.get_interpreter(ScriptLanguage.PYTHON)` (not `ScriptExecutor.get_interpreter`). The mapping:
   - Python: `InterpreterRegistry.get_interpreter(ScriptLanguage.PYTHON).path` + the script path
   - PowerShell: `powershell -NoProfile -File` + the script path
   - Bash: use the detected shell + the script path
   - JavaScript: `node` + the script path
5. Check if the script's directory contains a virtual environment (`venv/`, `.venv/`, `env/`). If so, prepend the activation command:
   - Windows: `& "{venv_path}\Scripts\Activate.ps1"; `
   - Unix: `source {venv_path}/bin/activate && `
6. Build the full command string and dispatch to `RunCommandTool`
7. Set the working directory to the script's parent directory

For `RUN_COMMAND` intents:
1. Extract the `command` entity from the intent
2. If the command references a specific directory, set cwd accordingly
3. Dispatch to `RunCommandTool` with `command` and `working_directory` parameters

### Step 4.4: Integrate Terminal Output Capture with SessionContext

After every `RUN_COMMAND` or `RUN_SCRIPT` execution:

1. Capture the `ToolResult.output` text
2. Store the full output in `SessionContext.last_operation_result`
3. If the output exceeds 5,000 characters, store a truncated version (first 2,000 + last 2,000 with "[...truncated...]" in the middle) in the session context, but keep the full output available via a reference
4. Display the output in the chat window via `_show_ai_message()`
5. If the command was a script execution, update `SessionContext.last_script_executed`

### Step 4.5: Add Safety Checks for Destructive Operations

In `IntentToolBridge.dispatch()`, before executing any intent classified as destructive, check consent:

**Operations requiring consent (using the existing `ConsentRequest` from `safety.py`):**
- `DELETE_FILE` — `ConsentType.FILE_MODIFICATION`, risk_level `medium`
- `DELETE_DIRECTORY` — `ConsentType.FILE_MODIFICATION`, risk_level `high`
- `WRITE_FILE` (when overwriting existing file) — `ConsentType.FILE_MODIFICATION`, risk_level `low`
- `MOVE_FILE` — `ConsentType.FILE_MODIFICATION`, risk_level `low`
- `RUN_COMMAND` (when command matches dangerous patterns) — `ConsentType.COMMAND_EXECUTION`, risk_level varies
- `ADMIN_ELEVATE` — `ConsentType.ADMIN_ACCESS`, risk_level `critical`

**Dangerous command patterns to detect:**
- `rm -rf`, `del /s /q`, `format`, `mkfs` — risk_level `critical`, always block with explanation
- `rm`, `del`, `rmdir` — risk_level `high`, require consent
- `sudo`, `runas` — risk_level `high`, require consent
- `chmod 777`, permission changes — risk_level `medium`, require consent

**Safe commands that skip consent:**
`ls`, `dir`, `pwd`, `cd`, `cat`, `head`, `tail`, `echo`, `type`, `Get-ChildItem`, `Get-Content`, `Get-Location`, `git status`, `git log`, `git diff`, `git branch`, `pip list`, `pip show`, `python --version`, `node --version`, `conda list`

Store these lists as class-level constants in `IntentToolBridge`: `SAFE_COMMANDS`, `DANGEROUS_PATTERNS`, `BLOCKED_PATTERNS`.

The consent flow:
1. `IntentToolBridge.dispatch()` checks if the operation needs consent
2. If yes, creates a `ConsentRequest` with the operation description, risk level, and details
3. Posts the consent request to the TUI via a callback (passed to the bridge during initialization)
4. The TUI displays the consent dialog: "⚠️ This operation will [description]. Risk: [level]. Proceed? (yes/no)"
5. If the user approves (`ConsentDecision.APPROVED` or `APPROVED_SESSION`), execution continues
6. If denied, the operation is cancelled and the user is informed

---

## Phase 5: Dependency Management System

### Objective
Enable the agent to install required dependencies, configure environments, handle compatibility issues, ensure backends run correctly, and detect and fix build or compilation errors — all dynamically from natural language requests.

### Step 5.1: Create Dependency Detection Logic

Add a new file `src/proxima/agent/dependency_manager.py`. This module provides dependency detection and installation capabilities.

> **Important:** A class named `DependencyManager` already exists in `src/proxima/agent/dynamic_tools/deployment_monitoring.py` (line 1004). That class handles vulnerability scanning, license compliance, and dependency pinning — a different concern. The new class below is named `ProjectDependencyManager` to avoid collision. It focuses on project dependency detection (scanning `requirements.txt`, `setup.py`, etc.), installation, and build error auto-fix.

The `ProjectDependencyManager` class must contain:

**A method `detect_project_dependencies(project_path: str) -> Dict[str, Any]`** that scans a directory for dependency specification files and returns a structured description:
1. Check for `requirements.txt` → parse it line by line, extracting package names and version constraints
2. Check for `setup.py` → extract `install_requires` from the contents (regex-based extraction of the list)
3. Check for `pyproject.toml` → parse using Python's `tomllib` (stdlib since Python 3.11). **For Python 3.10 compatibility**, use a conditional import: `try: import tomllib; except ModuleNotFoundError: import tomli as tomllib` and add `tomli` to `requirements-build.txt` as a fallback dependency. Extract `[project.dependencies]` and `[project.optional-dependencies]`.
4. Check for `setup.cfg` → parse using `configparser`, extract `install_requires` from `[options]`
5. Check for `package.json` → parse as JSON, extract `dependencies` and `devDependencies`
6. Check for `Pipfile` → parse using `tomllib`, extract `[packages]`
7. Check for `environment.yml` → parse using `pyyaml`, extract `dependencies`
8. Return a dict with keys: `python_packages`, `node_packages`, `system_packages`, `source_file`, `detected_manager` (pip/conda/npm/yarn)

**A method `install_dependencies(project_path: str, packages: Optional[List[str]] = None) -> Tuple[bool, str]`** that:
1. If `packages` is provided explicitly, use those
2. If not, call `detect_project_dependencies()` to find them
3. Determine the package manager: if `Pipfile` exists use `pipenv install`; if `environment.yml` exists use `conda env create -f`; if `package.json` exists use `npm install`; otherwise use `pip install`
4. Build the install command string
5. Execute via `subprocess.Popen` with stdout/stderr capture
6. Parse the output for errors
7. Return (success, output_text)

**A method `check_installed(package_name: str) -> Tuple[bool, Optional[str]]`** that:
1. Run `pip show {package_name}` via subprocess
2. If return code 0, parse the version from output → return (True, version)
3. If return code non-zero → return (False, None)

**A method `detect_and_fix_errors(error_output: str, project_path: str) -> Optional[str]`** that:
1. Parse the error output for common patterns:
   - `ModuleNotFoundError: No module named 'X'` → suggest and auto-install package X
   - `error: Microsoft Visual C++ 14.0 or greater is required` → suggest installing Visual Studio Build Tools
   - `CMake Error` → suggest installing cmake (`pip install cmake`)
   - `fatal error: 'X.h' file not found` → suggest installing the system library
   - `CUDA error` or `nvcc not found` → suggest installing CUDA toolkit
   - `Permission denied` → suggest running with admin privileges
2. If a fix is identified, return the fix command string
3. If no fix found, return None

### Step 5.2: Register Dependency Intent Executors

In `IntentToolBridge`, add handling for the three dependency-related intents:

**For `INSTALL_DEPENDENCY`:**
1. Extract `package` entities from the intent
2. If specific packages are named, build `pip install package1 package2 ...` command
3. If no packages named but a path context exists (e.g., user said "install dependencies" after cloning), call `ProjectDependencyManager.detect_project_dependencies(context_path)` and build the install command from detected dependencies
4. Execute via `RunCommandTool`
5. After execution, parse the output. If errors found, call `ProjectDependencyManager.detect_and_fix_errors()` and present the fix suggestion to the user. If the user approves, auto-execute the fix.

**For `CONFIGURE_ENVIRONMENT`:**
1. Determine what the user wants: create venv, activate venv, set env vars
2. For "create venv": build command `python -m venv {env_name}` where env_name defaults to `.venv`
3. For "activate venv": build the activation command appropriate to the OS
4. For "set env var": build `$env:VAR_NAME = "value"` on Windows or `export VAR_NAME=value` on Unix
5. Execute via `RunCommandTool`
6. Update `SessionContext.active_environments`

**For `CHECK_DEPENDENCY`:**
1. Extract `package` entities
2. For each package, call `ProjectDependencyManager.check_installed(package)`
3. Format results as a table showing package name, installed status, and version
4. Display in chat

### Step 5.3: Integrate Error Detection into Script and Build Execution

In `IntentToolBridge.dispatch()`, after executing any `RUN_COMMAND` or `RUN_SCRIPT` that fails (ToolResult.success is False):

1. Pass the error output to `ProjectDependencyManager.detect_and_fix_errors(error_output, working_dir)`
2. If a fix is found:
   - Display to the user: "❌ Execution failed. Detected issue: [description]. Suggested fix: [fix_command]. Shall I apply it?"
   - If user approves, execute the fix command
   - After fixing, automatically retry the original command
   - If retry succeeds, report success
   - If retry fails again, report both the original error and the fix attempt
3. If no fix found, display the error output and suggest the user check the error manually

This creates an automatic error-detection-and-fix loop that handles the most common build and dependency issues dynamically.

### Step 5.4: Backend-Specific Dependency Handling

For backend build operations (detected by keywords like "build", "compile" combined with backend names), the system must check the backend build profile in `configs/backend_build_profiles.yaml`:

1. When the user requests building a backend, the bridge should load the backend's profile from the YAML using `BuildProfileLoader` from `backend_builder.py`
2. The profile's `dependencies.packages` list specifies required Python packages
3. Before building, check each required package via `ProjectDependencyManager.check_installed()`
4. If any packages are missing, present the list to the user and offer to install them
5. If the user approves, install all missing packages via `pip install`
6. Then proceed with the build steps from the profile

This ensures that all dependency issues are resolved before the build begins, preventing mid-build failures.

---

## Phase 6: Backend Build, Compilation, and Code Modification

### Objective
Enable the integrated model to build, compile, test, and modify quantum simulator backends dynamically via natural language user requests. All operations execute through the terminal. Backend modifications require explicit user consent and support full undo, redo, and rollback for safety.

### Supported Backends
The following backends must be buildable and modifiable through the agent:
1. **LRET cirq scalability** — profile: `lret_cirq_scalability` in `backend_build_profiles.yaml`
2. **LRET pennylane hybrid** — profile: `lret_pennylane_hybrid`
3. **LRET phase 7 unified** — profile: `lret_phase_7_unified`
4. **Cirq** — profile: `cirq`
5. **Qiskit Aer** — profile: `qiskit`
6. **QuEST** — profile: `quest`
7. **qsim** — profile: `qsim`
8. **cuQuantum** — profile: `cuquantum`
9. **qsim CUDA** — profile: `qsim_cuda`
10. **Any additional backends** — the system must support generic build workflows for backends not in the YAML profiles

### Step 6.1: Backend Build via YAML Profiles and Terminal

When the user requests building a backend (intent `BACKEND_BUILD`), the `IntentToolBridge` dispatches to a custom handler that orchestrates the entire build pipeline:

**Step 1 — Backend Identification:**
1. Extract the backend name from the intent entities. The entity extractor (Phase 2) should recognize backend names from the user's message: "build lret cirq", "compile the qiskit backend", "build cirq-scalability-comparison", etc.
2. Normalize the extracted name to match a profile key in `configs/backend_build_profiles.yaml`. Build a normalization mapping as a constant in `IntentToolBridge`: `{"lret cirq": "lret_cirq_scalability", "lret pennylane": "lret_pennylane_hybrid", "lret phase 7": "lret_phase_7_unified", "cirq": "cirq", "qiskit": "qiskit", "qiskit aer": "qiskit", "quest": "quest", "qsim": "qsim", "qsim cuda": "qsim_cuda", "cuquantum": "cuquantum", "cuquantum sim": "cuquantum"}`.
3. If the backend matches a known profile, load the profile using `BuildProfileLoader` from `src/proxima/agent/backend_builder.py`.
4. If no profile matches, treat as a generic backend: look for `setup.py`, `pyproject.toml`, or `Makefile` in the backend directory and construct a default build sequence.

**Step 2 — Dependency Pre-Check:**
1. Load the profile's `dependencies.packages` list
2. For each package, call `ProjectDependencyManager.check_installed(package)` (from Phase 5)
3. If any packages are missing, present the missing list to the user:
   ```
   "⚠️ Missing dependencies for {backend_name}:
   - cirq>=1.0.0 (not installed)
   - numpy>=1.21.0 (installed: 1.24.0 ✓)
   
   Install missing packages? (yes/no)"
   ```
4. On approval, install missing packages via `pip install`
5. Check `dependencies.system_packages` — if any system packages are required (e.g., `cmake`, `gcc`, CUDA toolkit), verify their availability by running `where {package}` (Windows) or `which {package}` (Unix). If missing, inform the user with installation instructions.

**Step 3 — Build Execution:**
1. Read the profile's `build_steps` — this is an ordered list where each step has: `step_id`, `command`, `description`, `timeout` (seconds), `retry` (count)
2. For each build step:
   - Display in chat: "⚙️ Step {n}/{total}: {description}"
   - Spawn a terminal via `TerminalOrchestrator.spawn_terminal(name="{backend_name} build step {n}", command=step.command, cwd=backend_directory)`
   - Stream output to the Execution Tab (key **2**) in real time
   - Wait for completion via `TerminalOrchestrator.wait_for_completion(terminal_id, timeout=step.timeout)`
   - If the step fails and `retry > 0`: retry up to `step.retry` times with a 5-second delay between retries
   - If the step fails after all retries: stop the build, report which step failed, show the last 50 lines of output, and offer to continue with the remaining steps or abort
3. After all steps complete successfully, report:
   ```
   "✅ Backend {backend_name} built successfully
   Build steps completed: {n}/{total}
   Total time: {duration}s"
   ```

**Step 4 — Build Verification:**
1. Run the profile's `verification.test_command` (e.g., `python -c "import cirq; print(cirq.__version__)"`)
2. Check that `verification.expected_files` exist in the build directory
3. If verification passes: confirm to the user
4. If verification fails: report which checks failed, suggest remediation

**For generic backends (no YAML profile):**
1. Check if the directory contains `setup.py` → run `pip install -e .`
2. Check if it contains `pyproject.toml` with `[build-system]` → run `pip install -e .`
3. Check if it contains `Makefile` → run `make`
4. Check if it contains `CMakeLists.txt` → run `mkdir build && cd build && cmake .. && cmake --build .`
5. If none of these exist, ask the user for the build command

### Step 6.2: Backend Code Modification with Safety

When the user requests modifying backend code (intent `BACKEND_MODIFY`), strict safety procedures apply. This uses the existing `BackendModifier` class from `src/proxima/agent/backend_modifier.py` and `CheckpointManager` from `src/proxima/agent/checkpoint_manager.py`.

**Modification Flow:**

**Step 1 — Consent:**
Before any code modification, create a `ConsentRequest` from `safety.py`:
- `consent_type`: `ConsentType.BACKEND_MODIFICATION`
- `risk_level`: `high` (all backend code modifications are high-risk by default)
- `description`: detailed description of what will be changed, including file path, modification type, and summary of the change
- `reversible`: `True` (all backend modifications create checkpoints)
- Display in chat:
  ```
  "🔴 Backend Code Modification Request
  
  File: {file_path}
  Change: {modification_description}
  Risk: HIGH
  
  This will modify backend source code. A checkpoint will be created for rollback.
  
  Proceed? (yes/no)"
  ```
- Wait for user approval. Only proceed on explicit `APPROVED` or `APPROVED_ONCE`.

**Step 2 — Checkpoint Creation:**
Before any modification:
1. Call `CheckpointManager.create_checkpoint(operation="backend_modify", description="{backend_name}: {change_description}", files=[file_path])` from `checkpoint_manager.py`
2. The checkpoint captures: file path, full file content, SHA256 checksum, file metadata (size, permissions, mtime), and a backup copy at a timestamped path under `~/.proxima/checkpoints/`
3. Store the checkpoint ID in `SessionContext.backend_checkpoints[backend_name]`
4. Store affected file paths in `SessionContext.last_modified_files`

**Step 3 — Modification Preview:**
Before applying the change, show a diff preview using `ModificationPreviewGenerator` from `modification_preview.py` (which returns a `ModificationPreview` data object):
1. Build a `CodeChange` object with the proposed modification:
   - `file_path`: the file being modified
   - `modification_type`: one of `ModificationType.REPLACE`, `INSERT`, `DELETE`, `APPEND`, `PREPEND`
   - `old_content`: the current content (or specific lines)
   - `new_content`: the proposed replacement
2. Generate a diff via `CodeChange.get_diff()` (method at line 76 of `backend_modifier.py`, on the `CodeChange` dataclass defined at line 46, uses `difflib.unified_diff`) — this produces a unified diff string
3. Use `ModificationPreviewGenerator.generate_preview()` (defined at line 262 of `modification_preview.py`, signature: `generate_preview(self, file_path, old_content, new_content, modification_type="modify", description="") -> ModificationPreview`) to generate and format the diff with Rich syntax highlighting:
   - Green for additions, red for deletions, gray for context lines
   - Line numbers on both sides
4. Display the formatted diff in the chat:
   ```
   "📝 Proposed modification to {file_path}:
   
   {formatted_diff}
   
   Apply this change? (yes/no)"
   ```
5. Wait for a second explicit approval before applying

**Step 4 — Apply Modification:**
1. Call `BackendModifier.apply_change(code_change)` from `backend_modifier.py`
2. The modifier writes the new content to the file
3. Verify the modification by reading the file back and comparing checksums
4. Return a `ModificationResult` with `success`, `diff`, and the checkpoint ID

**Step 5 — Post-Modification Verification:**
1. After the modification, optionally run the backend's test command to verify the change didn't break anything
2. If tests fail: automatically offer to rollback:
   ```
   "❌ Backend tests failed after modification.
   
   Rollback to the checkpoint? (yes/no)"
   ```
3. If the user approves rollback: execute Step 6.3 rollback procedure

### Step 6.3: Undo, Redo, and Rollback System

The undo/redo/rollback system leverages the existing `CheckpointManager` from `checkpoint_manager.py` and `RollbackManager` from `safety.py`.

**Undo (intent `UNDO_OPERATION`):**
1. Retrieve the most recent checkpoint from `CheckpointManager.list_checkpoints(limit=1)[0]` (the `checkpoint_manager.py` class does not have a `get_latest_checkpoint()` method — use `list_checkpoints()` which returns checkpoints newest-first)
2. Display what will be undone:
   ```
   "↩️ Undo last operation: {checkpoint.operation}
   Description: {checkpoint.description}
   Files affected: {list of files}
   Created: {checkpoint.timestamp}
   
   Proceed with undo? (yes/no)"
   ```
3. On approval, for each `FileState` in the checkpoint:
   - Read the stored original content from `file_state.backup_path`
   - Write it back to the original `file_state.path`
   - Verify the restored file's checksum matches `file_state.checksum`
4. Mark the checkpoint as `rolled_back = True`
5. Push the current state (before undo) as a new "redo checkpoint" so the change can be re-applied
6. Report success:
   ```
   "✅ Undo complete. {n} file(s) restored to their previous state."
   ```

**Redo (intent `REDO_OPERATION`):**
1. Check if there is a redo checkpoint available (created by the undo step above)
2. If available, restore the state from the redo checkpoint using the same procedure as undo
3. If no redo available, inform the user: "No operation to redo."

**Rollback (targeted to a specific checkpoint):**
1. If the user specifies which operation to rollback (e.g., "rollback the backend modification"), look up `SessionContext.backend_checkpoints[backend_name]` to find the checkpoint ID
2. If the user says "rollback" without specifics, use the most recent checkpoint
3. Call `CheckpointManager.rollback_to(checkpoint_id)`:
   - Restore all files in the checkpoint's `FileState` list
   - Verify all checksums
   - Mark the checkpoint as rolled back
4. If any file restoration fails (e.g., file was deleted or permissions changed):
   - Report which files could not be restored
   - Attempt to restore from the backup path directly
   - If backup is also gone, report failure and suggest manual recovery

**Rollback Chain:**
Multiple operations may have been performed since a checkpoint. The system must support rolling back to any checkpoint, not just the most recent one:
1. `CheckpointManager` maintains an ordered list of all checkpoints
2. When the user asks "rollback the build" or "undo everything since the clone", find the appropriate checkpoint by matching the `operation` or `description` field
3. Apply all intermediate rollbacks in reverse order (most recent first) to reach the target checkpoint state

### Step 6.4: Backend Testing and Verification

When the user requests testing a backend (intent `BACKEND_TEST`):

1. Determine which backend to test:
   - If specified in the message, use the named backend
   - If not specified, use `SessionContext.last_built_backend`
   - If neither available, ask the user

2. Load the backend's YAML profile. Use the `verification` section:
   - `test_import`: a Python import statement to verify the module loads (e.g., `import cirq`)
   - `test_command`: a shell command to run the test suite (e.g., `python -m pytest tests/ -v`)
   - `expected_files`: list of files that should exist after a successful build

3. Execute the verification steps in order:
   - **Import test**: Run `python -c "{test_import}"` via terminal. If it raises `ImportError`, the backend is not properly installed.
   - **Test command**: Spawn a monitored terminal for the test suite. Stream output to the Execution Tab. Parse test output for pass/fail counts.
   - **File existence**: Check each expected file with `os.path.exists()`.

4. Collect all results and display:
   ```
   "🧪 Backend Test Results: {backend_name}
   
   Import test: ✅ Passed / ❌ Failed
   Test suite: ✅ 42/42 passed / ❌ 3 failures
   Expected files: ✅ All present / ❌ Missing: {list}
   
   Overall: ✅ PASS / ❌ FAIL"
   ```

5. Push the test results to the Result Tab (key **3**) as a structured result entry.

### Step 6.5: Backend Configuration for Proxima

When the user requests configuring Proxima to use a backend (intent `BACKEND_CONFIGURE`):

1. Determine the backend name and its installation path
2. Read the current `configs/default.yaml` file
3. Update the backend configuration section:
   - Set the active backend name
   - Set the backend module path
   - Set any backend-specific parameters from the YAML profile
4. Write the updated configuration back to `configs/default.yaml`
5. Create a checkpoint before writing (uses `ConsentType.FILE_MODIFICATION`, risk_level `medium`)
6. Verify the configuration by attempting to import the backend module
7. Report success:
   ```
   "✅ Proxima configured to use {backend_name}
   Configuration file: configs/default.yaml
   Backend path: {installation_path}"
   ```

### Step 6.6: Register All Backend Intents in the Bridge

Ensure all `BACKEND_*` intents are registered in `IntentToolBridge`:

| Intent | Handler | Description |
|---|---|---|
| `BACKEND_BUILD` | Custom: loads YAML profile, executes build steps via `TerminalOrchestrator` | Build/compile a backend |
| `BACKEND_CONFIGURE` | Custom: updates `configs/default.yaml` | Configure Proxima to use a backend |
| `BACKEND_TEST` | Custom: runs verification from YAML profile | Test a built backend |
| `BACKEND_MODIFY` | Custom: uses `BackendModifier` + `CheckpointManager` with consent flow | Modify backend source code |
| `BACKEND_LIST` | Custom: reads `backend_build_profiles.yaml`, lists available profiles | List available backends |

For `BACKEND_LIST`, the handler simply reads `configs/backend_build_profiles.yaml`, extracts all profile names and descriptions, and formats them as a table:
```
"📋 Available Backend Profiles:

| # | Name | Description | GPU Required |
|---|---|---|---|
| 1 | lret_cirq_scalability | LRET Cirq Scalability Backend | No |
| 2 | lret_pennylane_hybrid | LRET PennyLane Hybrid Backend | No |
| 3 | lret_phase_7_unified | LRET Phase 7 Unified Backend | No |
| 4 | cirq | Google Cirq Backend | No |
| 5 | qiskit | IBM Qiskit Aer Backend | No |
| 6 | quest | QuEST Backend | No |
| 7 | qsim | Google qsim Backend | No |
| 8 | cuquantum | NVIDIA cuQuantum Backend | Yes |
| 9 | qsim_cuda | Google qsim CUDA Backend | Yes |"
```

---

## Phase 7: Administrative Access and Safe Privilege Escalation

### Objective
Enable the agent to detect when administrative privileges are required, request them through safe escalation procedures, execute admin-level operations with explicit user consent, and return to standard privileges immediately after.

### Step 7.1: Detect When Admin Access Is Required

The `AdminPrivilegeHandler` from `src/proxima/agent/admin_privilege_handler.py` already defines `OperationCategory` (FILE_SYSTEM, PACKAGE_INSTALL, SERVICE_CONTROL, NETWORK, REGISTRY, PERMISSION, SYSTEM_CONFIG) and `PrivilegeLevel` (STANDARD, ELEVATED, SYSTEM).

In the `IntentToolBridge.dispatch()` method, before executing certain operations, check whether elevated privileges are required:

**Auto-detection triggers (check the command/operation against these patterns):**
1. **Package installation to system-level paths**: `pip install` without `--user` flag when the Python environment is system-wide (not a venv). Detected by checking if the pip target path requires write access.
2. **System service control**: commands containing `sc`, `systemctl`, `service`, `net start/stop`
3. **Writing to protected directories**: file write operations targeting `C:\Windows\`, `C:\Program Files\`, `/usr/`, `/etc/`, `/opt/` (system directories)
4. **Network configuration**: commands with `netsh`, `iptables`, `ufw`, `firewall-cmd`
5. **Registry access** (Windows): commands with `reg add`, `reg delete`, `regedit`
6. **Permission changes**: `chmod`, `chown`, `icacls`, `attrib`, `takeown`
7. **CUDA/GPU operations**: installing CUDA toolkit, modifying GPU drivers, running `nvidia-smi` configuration

When any of these patterns are detected:
1. Call `AdminPrivilegeHandler.get_privilege_info().level` to get the current `PrivilegeLevel` (returns a `PrivilegeInfo` dataclass with `.level`, `.elevation_method`, and other fields)
2. If already `ELEVATED` or `SYSTEM`, proceed directly
3. If `STANDARD`, trigger the escalation flow in Step 7.2

### Step 7.2: Safe Escalation Procedure

When elevation is needed and the current level is `STANDARD`:

**Step 1 — Inform and request consent:**
Display in chat:
```
"🔐 Administrative Access Required

Operation: {operation_description}
Category: {operation_category.name}
Reason: {why_admin_is_needed}
Risk: HIGH

This operation requires elevated privileges. On Windows, a UAC prompt will appear.
On Linux/macOS, you will be prompted for your password.

Proceed with privilege escalation? (yes/no)"
```

Create a `ConsentRequest` with `consent_type=ConsentType.ADMIN_ACCESS`, `risk_level='critical'`. Only proceed on explicit approval.

**Step 2 — Determine escalation method:**
Use `AdminPrivilegeHandler.get_privilege_info().elevation_method` which detects the platform:
- **Windows**: `ElevationMethod.UAC` — use PowerShell `Start-Process -Verb RunAs` to launch an elevated process
- **Linux/macOS with sudo**: `ElevationMethod.SUDO` — prepend `sudo` to the command
- **Linux with pkexec**: `ElevationMethod.PKEXEC` — use `pkexec` (Polkit) if sudo is unavailable
- **Windows RunAs**: `ElevationMethod.RUNAS` — use `runas /user:Administrator` as fallback

**Step 3 — Execute elevated command:**
1. Build the elevated command:
   - Windows: `powershell Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -Command {original_command}'`
   - Linux: `sudo {original_command}`
2. Spawn the command via `TerminalOrchestrator` in a monitored terminal
3. The system UAC dialog or sudo password prompt will appear to the user — the agent does NOT handle password input, the user interacts with the OS prompt directly
4. Wait for completion and capture output

**Step 4 — Return to standard privileges:**
After the elevated command completes:
1. The elevated process terminates naturally (it was a single command)
2. Log the admin operation to `~/.proxima/logs/admin_operations.log` with timestamp, command, user, and result
3. Update `SessionContext` with the operation result
4. **Do not** retain elevated privileges for subsequent operations — each admin operation requires its own escalation

### Step 7.3: Admin Operations Execution

For specific admin operation categories, provide tailored handling in the `IntentToolBridge`:

**PACKAGE_INSTALL (system-level):**
1. If the user's Python environment is a venv, no admin needed — install directly
2. If system Python: suggest creating a venv first ("Would you like me to create a virtual environment instead?")
3. If the user insists on system-level install: escalate and run `sudo pip install` or elevated `pip install`

**SERVICE_CONTROL:**
1. On Windows: use `sc.exe` or `net start/stop` with elevation
2. On Linux: use `systemctl` with sudo
3. Always show the current service status before and after the operation

**PERMISSION_CHANGES:**
1. Display current permissions of the target file/directory
2. Show what the new permissions will be
3. Require consent
4. Execute with elevation

**CUDA/GPU SETUP:**
1. If installing CUDA toolkit: build the full command sequence (download, install, set PATH)
2. Require elevation for system-level installation
3. After installation, verify with `nvidia-smi` and `nvcc --version`

### Step 7.4: Security Constraints and Audit Logging

Implement hard security constraints that the agent must NEVER violate, regardless of user request:

**Hard restrictions (cannot be overridden):**
1. Never store or transmit passwords — if admin requires a password, rely on the OS prompt (UAC, sudo)
2. Never disable system security features (Windows Defender, SELinux, firewall) unless explicitly requested with critical-level consent and a clear warning
3. Never modify boot records, partition tables, or system firmware
4. Never execute commands from untrusted remote sources without inspection
5. Maximum of 3 consecutive admin escalations per session — after 3, require the user to restart the session with a warning

**Audit logging:**
Every admin operation must be logged to `~/.proxima/logs/admin_audit.log`:
- Timestamp (ISO 8601 format)
- User identity (OS username)
- Operation description
- Command executed
- Consent decision (APPROVED/DENIED)
- Result (SUCCESS/FAILURE)
- Files affected (if any)

The audit log is append-only and should not be modifiable by the agent itself.

---

## Phase 8: GitHub Repository Operations

### Objective
Enable the integrated model to perform all GitHub operations — clone, pull, push, and branch management — dynamically based on natural language user intent, executed through the terminal.

### Step 8.1: Enhance Git Clone Handling

In `IntentToolBridge.build_tool_arguments()` for `GIT_CLONE`:

1. Extract the `url` entity — this is the GitHub repository URL
2. Extract the `path` entity — this is the target directory where the repo should be cloned. If not specified, default to a standard location: `os.path.join(os.path.expanduser('~'), '.proxima', 'backends', repo_name)` where `repo_name` is extracted from the URL (last path component minus `.git`)
3. Extract the `branch` entity — if present, append `--branch {branch}` to the git clone command
4. Build the full command: `git clone {url} {target_path}` (with optional `--branch {branch}`)
5. After successful execution:
   - Update `SessionContext.cloned_repos[url] = target_path`
   - Update `SessionContext.last_cloned_repo = target_path`
   - Update `SessionContext.last_cloned_url = url`
   - Update `SessionContext.current_directory = target_path`

**URL Extraction must handle these formats:**
- Full HTTPS: `https://github.com/user/repo`
- Full HTTPS with .git: `https://github.com/user/repo.git`
- SSH: `git@github.com:user/repo.git`
- Short form: `github.com/user/repo`
- Just owner/repo: `kunal5556/LRET` (should be prefixed with `https://github.com/`)

Add these patterns to the entity extraction in `robust_nl_processor.py`:
- `https?://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?`
- `git@github\.com:[\w\-\.]+/[\w\-\.]+(?:\.git)?`
- `github\.com/[\w\-\.]+/[\w\-\.]+`
- For the short form `owner/repo`, only match when preceded by clone-related context words

### Step 8.2: Enhance Git Pull and Push

**For `GIT_PULL`:**
1. Determine the working directory: use `SessionContext.current_directory` or the extracted path entity
2. Verify it's a git repository: check for `.git` directory or run `git rev-parse --is-inside-work-tree`
3. Build command: `git pull` (default remote and branch from tracking). If a specific remote or branch is mentioned in entities, use `git pull {remote} {branch}`
4. Execute and report results

**For `GIT_PUSH`:**
1. Same directory handling as pull
2. Build command: `git push`. If remote/branch specified, use `git push {remote} {branch}`
3. If push fails with "no upstream branch", automatically suggest and execute `git push --set-upstream origin {current_branch}`
4. If push fails with authentication error, suggest setting up credentials
5. Requires consent: `ConsentType.GIT_OPERATION`, risk_level `medium`

### Step 8.3: Enhanced Branch and Checkout Operations

**For `GIT_CHECKOUT`:**
1. Extract `branch` entity from intent
2. Validate the branch name: must be at least 2 characters, start with a letter, not match common English words (the, and, for, with, this, that, from, clone, build, configure, into, after, before, next, then)
3. Build command: `git checkout {branch}`. If the branch doesn't exist locally, try `git checkout -b {branch} origin/{branch}`
4. After successful checkout, update `SessionContext.last_mentioned_branches`

**For `GIT_BRANCH`:**
1. Determine sub-operation from the user's message:
   - "create branch X" → `git checkout -b X`
   - "delete branch X" → `git branch -d X` (requires consent, risk_level `medium`)
   - "list branches" → `git branch -a`
   - "rename branch" → `git branch -m old_name new_name`
2. Execute and display the result

**For `GIT_MERGE`:**
1. Extract `branch` entity (the branch to merge FROM)
2. Build command: `git merge {branch}`
3. If merge conflict occurs (exit code non-zero and output contains "CONFLICT"), report the conflicting files and ask the user how to proceed
4. Requires consent: risk_level `medium`

### Step 8.4: Git Commit Workflow

**For `GIT_COMMIT`:**
1. Extract commit message from the intent. Look for quoted text or content after "with message" / "message" / "saying"
2. If no message provided, check if an LLM is available. If so, run `git diff --staged` and send the diff to the LLM with the prompt: "Generate a concise commit message for these changes:" — use the LLM's response as the commit message
3. If nothing is staged, check if the user said "commit all" or "commit everything" → run `git add -A` first
4. Build command: `git commit -m "{message}"`
5. Execute and report the result (commit hash, files changed)

**For `GIT_ADD`:**
1. Extract file path entities. If the user said "add everything" or "stage all" → use `git add -A`
2. If specific files mentioned → `git add {file1} {file2} ...`
3. If pattern mentioned (e.g., "add all python files") → `git add "*.py"`
4. Show what was staged by running `git status --short` after add

### Step 8.5: Register All Git Intents in the Bridge

Ensure that every `IntentType.GIT_*` maps to an appropriate executor in `IntentToolBridge.INTENT_TO_TOOL`. For operations that existing tool classes handle (GitStatusTool, GitCommitTool, etc. from `git_tools.py`), map to those tools. For operations not covered by existing tools (push, pull, clone, checkout, merge, rebase), map to `RunCommandTool` with constructed git commands.

---

## Phase 9: Multi-Terminal Monitoring and Result Display

### Objective
Enable the agent to spawn, monitor, and display output from multiple simultaneous terminal sessions, providing live real-time execution output in the Execution Tab (key 2) and comprehensive results analysis in the Result Tab (key 3).

### Step 9.1: Create a Terminal Orchestrator

Create a new file `src/proxima/agent/terminal_orchestrator.py`. This module wraps the existing `MultiTerminalMonitor` (from `multi_terminal.py`) and `TerminalStateMachine` to provide a unified interface for the agent:

**The `TerminalOrchestrator` class contains:**

**`spawn_terminal(name: str, command: str, cwd: str, background: bool = False) -> str`:**
1. Generate a unique terminal ID using `uuid.uuid4().hex[:8]`
2. Create a `subprocess.Popen` object with `stdout=subprocess.PIPE`, `stderr=subprocess.PIPE`, `stdin=subprocess.PIPE`, `shell=True`
3. On Windows, use PowerShell: prepend `powershell -NoProfile -Command` to the command
4. Register the process with the `MultiTerminalMonitor` from `multi_terminal.py` for state tracking
5. Register with `TerminalStateMachine` from `src/proxima/agent/terminal_state_machine.py` (**Note:** this file is in `src/proxima/agent/`, NOT in `dynamic_tools/`) for lifecycle management
6. Start a background thread that reads stdout/stderr line by line and stores each line in a circular buffer (from `multi_terminal.py`'s `CircularOutputBuffer`, capacity 10,000 lines)
7. Store the terminal info in `SessionContext.active_terminals[terminal_id]`
8. Emit a `TerminalEvent` with `event_type=STARTED`
9. Return the `terminal_id`

**`get_output(terminal_id: str, tail_lines: int = 50) -> str`:**
1. Look up the terminal's circular buffer
2. Return the last `tail_lines` lines as a single string
3. If the terminal doesn't exist, return an error message

**`get_all_terminals() -> List[Dict[str, Any]]`:**
1. Return a list of dictionaries, one per terminal, containing: `id`, `name`, `command`, `state` (from `TerminalState` enum), `pid`, `start_time`, `elapsed_seconds`, `output_lines_count`, `last_5_lines`
2. Include both active and recently completed terminals (completed within the last 10 minutes)

**`kill_terminal(terminal_id: str) -> bool`:**
1. Look up the process by terminal_id
2. On Windows: call `process.terminate()`, wait 5 seconds, then `process.kill()` if still running
3. On Unix: send `SIGTERM`, wait 5 seconds, then `SIGKILL`
4. Update the terminal state to `CANCELLED`
5. Emit a `TerminalEvent` with `event_type=CANCELLED`
6. Return True on success

**`wait_for_completion(terminal_id: str, timeout: int = 600) -> Tuple[bool, str]`:**
1. Poll the process state every 0.5 seconds
2. If completed within timeout: return (process_returncode == 0, full_output)
3. If timeout exceeded: kill the process, return (False, "Timed out after {timeout}s")

**`subscribe_output(terminal_id: str, callback: Callable[[str], None])`:**
1. Register a callback that is invoked for each new line of output
2. The background reader thread calls this callback in addition to storing in the circular buffer
3. This is used for live streaming to the TUI

### Step 9.2: Implement Terminal Intent Executors

In `IntentToolBridge`, add handling for the four terminal monitoring intents:

**For `TERMINAL_MONITOR`:**
1. Call `TerminalOrchestrator.get_all_terminals()`
2. Format the result as a table:
   ```
   | # | Name | Command | State | Duration | Output Lines |
   ```
3. For running terminals, include the last 3 lines of output
4. Display in chat

**For `TERMINAL_OUTPUT`:**
1. Determine which terminal the user is asking about:
   - If they said "terminal 1" or "first terminal" → use the first terminal in the list
   - If they said "the build terminal" → search by name/command containing "build"
   - If they said "show output" with no qualifier → use the most recent terminal
2. Call `TerminalOrchestrator.get_output(terminal_id, tail_lines=100)`
3. Display the output in the chat with syntax highlighting where possible

**For `TERMINAL_KILL`:**
1. Determine which terminal to kill (same resolution as TERMINAL_OUTPUT)
2. Show consent dialog: "Stop terminal '{name}' running '{command}'? (yes/no)"
3. On approval, call `TerminalOrchestrator.kill_terminal(terminal_id)`
4. Report success/failure

**For `TERMINAL_LIST`:**
1. Call `TerminalOrchestrator.get_all_terminals()`
2. Display as a compact list showing name, state, and elapsed time for each

### Step 9.3: Connect Terminal Output to the Execution Tab

The Execution Tab (`src/proxima/tui/screens/execution.py`, accessible via key **2**) already has a `RichLog` widget for displaying execution output and a progress bar.

To bridge agent terminal sessions to the Execution Tab:

1. In `TerminalOrchestrator.spawn_terminal()`, after creating the process, post a Textual message to the `ExecutionScreen` informing it of a new terminal. Use Textual's `post_message()` or a custom `Message` subclass (e.g., `AgentTerminalStarted(terminal_id, name, command)`)

2. In the `ExecutionScreen` class, handle the `AgentTerminalStarted` message:
   - Update the execution info display (stage, command name)
   - Start receiving output lines via `TerminalOrchestrator.subscribe_output()`
   - Each received line is appended to the `RichLog` widget using `self.query_one(RichLog).write(line)`
   - Update the progress bar based on heuristics (lines of output) or known step counts

3. When the terminal completes, post `AgentTerminalCompleted(terminal_id, success, output)`:
   - Update the execution status display
   - Show final output summary
   - If the terminal was a build operation, show build success/failure status

4. For simultaneous terminals, use Textual's `TabbedContent` widget to create one tab per active terminal in the Execution Tab, each with its own `RichLog`. The tab label shows the terminal name and a state indicator (🟢 running, ✅ completed, ❌ failed).

### Step 9.4: Connect Terminal Results to the Result Tab

The Result Tab (`src/proxima/tui/screens/results.py`, accessible via key **3**) shows results with probability visualization and export options.

When a terminal completes and produces output that should be analyzed:

1. In `TerminalOrchestrator`, after a terminal reaches `COMPLETED` state, check if the terminal was spawned by the agent for a task that produces results (build, test, simulation, script execution)

2. If results are expected, parse the terminal output into a structured `ResultEntry`:
   - `title`: derived from the command (e.g., "Build lret_cirq_scalability", "Run pennylane_4q_50e_25s_10n.py")
   - `timestamp`: completion time
   - `status`: success/failure based on return code
   - `output_summary`: first 500 characters of output
   - `metrics`: extract key-value pairs from the output using regex patterns for common formats:
     - `"Time: 12.5s"` → `{"execution_time": 12.5}`
     - `"Tests passed: 42/42"` → `{"tests_passed": 42, "tests_total": 42}`
     - `"Fidelity: 0.995"` → `{"fidelity": 0.995}`
     - JSON output → parse directly
   - `raw_output`: full output text

3. Send the `ResultEntry` to the Result Tab. Access the `ResultsScreen` instance via Textual's screen management (`self.app.get_screen('results')`) and call its method to add a new result entry.

4. If the existing `ResultsScreen` doesn't have a method to accept programmatic result entries, add a method `add_agent_result(self, result: Dict[str, Any])` that:
   - Creates a new entry in the results list
   - Populates the detail view when selected
   - Makes the result available for export via the existing `ExportEngine`

### Step 9.5: Implement LLM-Based Result Analysis

When the user requests analysis (intent `ANALYZE_RESULTS`) or when the agent completes a complex task:

1. Collect the raw output from the terminal(s) involved
2. If an LLM is available, send the output to the model with a structured analysis prompt:
   ```
   "Analyze the following execution output and provide:
   1. Summary: What was executed and what happened
   2. Status: Success or failure with specific details
   3. Key Metrics: Extract any numerical results (timing, accuracy, counts)
   4. Issues: Any warnings or errors detected
   5. Recommendations: Next steps or improvements
   
   Output to analyze:
   {output_text}"
   ```
3. Parse the LLM's analysis response
4. Format the analysis with Rich markup (bold headings, colored status, tables for metrics)
5. Display in the chat window
6. Also push a structured version to the Result Tab

If no LLM is available, perform basic analysis using regex pattern matching:
- Count error lines (lines containing "error", "Error", "ERROR", "FAIL")
- Count warning lines
- Extract timing information
- Determine overall pass/fail
- Format as a simple report

---

## Phase 10: Agentic Loop and Streaming Response Architecture

### Objective
Replace the current single-shot response pipeline with a proper agentic loop where the integrated model can reason multi-turn: execute a tool, see the result, decide what to do next, execute another tool, and so on — until the task is complete. This is the Crush-inspired architecture.

### Step 10.1: Create the Agent Loop Module

Create a new file `src/proxima/agent/dynamic_tools/agent_loop.py`. This module implements the core agentic loop that replaces the current 5-phase `_generate_response()` cascade.

**The `AgentLoop` class contains:**

**Constructor `__init__(self, nl_processor, tool_bridge, llm_router, llm_tool_integration, session_context, ui_callback)`:**
- `nl_processor`: the `RobustNLProcessor` instance
- `tool_bridge`: the `IntentToolBridge` instance
- `llm_router`: the `LLMRouter` instance from `src/proxima/intelligence/llm_router.py` (may be None if no LLM is integrated). Provides `route()` and `route_with_fallback()` methods for sending requests to the configured LLM provider. **Note:** `LLMRouter` does NOT have a `stream_send()` method. Streaming is done at the provider level: get the provider via `_pick_provider()`, then call `provider.stream_send(request, api_key, callback)`. Not all providers implement streaming — see Step 10.4 for the detection pattern.
- `llm_tool_integration`: the `LLMToolIntegration` instance from `llm_integration.py` (may be None). Provides `parse_tool_calls()` for extracting structured tool calls from LLM responses.
- `session_context`: the `SessionContext` instance
- `ui_callback`: a callable that takes a message string and displays it in the TUI chat

**Method `process_message(message: str) -> str`:**
This is the main entry point, called from `agent_ai_assistant.py` when the user sends a message.

> **Important — Synchronous baseline:** The existing `_generate_response()` method in `agent_ai_assistant.py` is **synchronous** (regular `def`, not `async def`). The `AgentLoop.process_message()` should also be synchronous for initial implementation. Streaming token display (Step 10.4) requires using Textual's `call_from_thread()` or `Worker` API for background execution — NOT converting the entire method to `async`. See Step 10.4 for the specific Textual integration.

The loop works as follows:

**Turn 1 — Intent Recognition:**
1. Call `nl_processor.recognize_intent(message)` to get an `Intent`
2. If intent confidence >= 0.5 and intent is NOT `UNKNOWN`:
   - Execute the intent directly via `tool_bridge.dispatch(intent, context)`
   - Store the result
   - If the intent is `MULTI_STEP`, execute all sub-intents sequentially via `PlanExecutor`, collecting results from each step
   - Go to **Result Evaluation**
3. If intent confidence is between 0.2 and 0.5, or if the intent is complex:
   - Go to **LLM-Assisted Execution**
4. If intent confidence < 0.2:
   - If LLM available: go to **LLM-Assisted Execution**
   - If LLM not available: respond with "I'm not sure what you want me to do. Could you rephrase?"

**LLM-Assisted Execution (multi-turn loop):**
1. Build a system prompt using `SystemPromptBuilder` (Step 10.2)
2. Build the conversation messages array: system prompt + last 10 conversation history entries + current user message
3. Send to the LLM via the `LLMRouter` instance (from `src/proxima/intelligence/llm_router.py`; its `route()` method handles provider selection, consent, and API key management). Use `stream=False` for reliability with smaller models.
4. Parse the LLM response for tool calls. **Use this priority order** (most reliable first):
   - **Priority 1 — Structured function calling**: If the `LLMProvider` supports function calling (OpenAI, Anthropic, Google), use the structured `ToolCall` parsing already in `LLMToolIntegration.parse_tool_calls()` from `llm_integration.py`. This is the most reliable method.
   - **Priority 2 — JSON tool-call blocks**: If the model returns a JSON code block like `{"tool": "...", "arguments": {...}}`, parse it directly.
   - **Priority 3 — Text pattern fallback** (for models like Ollama `llama2-uncensored` that may not support function calling): Detect patterns like:
     - "I'll run: `{command}`" → extract command, execute via `RunCommandTool`
     - "Let me execute: `{command}`" → same
     - "Running `{command}`..." → same
   - If a tool call is detected:
     - Execute the tool
     - Feed the result back to the LLM as a new message: "Tool result: {result}"
     - Continue the loop (goto step 3 with updated conversation)
   - If no tool call detected (the response is a final answer):
     - Return the response text
5. **Loop limit**: Maximum 10 iterations per user message. After 10 iterations, summarize what was accomplished and what remains, then stop. (Lower than typical agentic systems because Proxima's tools are coarser-grained — each iteration does more work.)
6. **Error in loop**: If a tool execution fails, include the error in the next LLM message so it can reason about alternatives.

**Result Evaluation:**
1. After execution (either direct or LLM-assisted), check if the result indicates:
   - Success → display result, update context
   - Failure with fixable error → call `ProjectDependencyManager.detect_and_fix_errors()`, apply fix if user approves, retry
   - Failure without fix → display error, suggest alternatives
2. Update `SessionContext` with the result
3. If the original request was a multi-step plan, check if there are remaining steps and continue execution

### Step 10.2: Create the System Prompt Builder

Create a new file `src/proxima/agent/dynamic_tools/system_prompt_builder.py`. This module constructs the system prompt sent to the integrated LLM.

**The `SystemPromptBuilder` class contains:**

**Method `build(context: SessionContext, capabilities: List[str]) -> str`:**

Construct a system prompt with these sections (in order):

**Section 1 — Role:**
"You are Proxima's AI agent, a terminal-based assistant for quantum computing simulation. You execute tasks on the user's local machine through terminal commands. You have access to the file system, git, package managers, and build tools."

**Section 2 — Current State:**
- "Current directory: {context.current_directory}"
- "Operating system: {platform.system()} {platform.release()}"
- "Python version: {sys.version}"
- "Active terminals: {count}" (list names and states if any)
- "Last operation: {context.last_operation.intent_type.name if context.last_operation else 'none'}"
- "Recent directories: {', '.join(context.last_mentioned_paths[:3])}"

**Section 3 — Capabilities:**
"You can perform these operations by describing what you want to do:
{numbered list of capability descriptions generated from the registered tools}"

The capabilities list is generated by iterating through `ToolRegistry.get_all_tools()` and using each tool's `description` property. Group by `ToolCategory` for clarity.

**Section 4 — Execution Instructions:**
"To execute a command, describe it naturally. For example:
- 'Run git clone https://github.com/user/repo'
- 'Install numpy and scipy'
- 'List files in the current directory'
- 'Read the contents of config.yaml'

For multi-step tasks, describe all steps. I will create a plan and ask for your confirmation before executing.

When you need to run a terminal command, state it clearly with backticks: `command here`

Always report results after execution. If something fails, suggest a fix."

**Section 5 — Safety Rules:**
"Never execute destructive operations without user confirmation. Always create backups before modifying files. Report errors clearly with suggested fixes."

The total prompt must stay under 2,000 tokens to leave room for conversation history and responses. Dynamically truncate the state section if it gets too long—prioritize capabilities and instructions over state details.

### Step 10.3: Refactor _generate_response in agent_ai_assistant.py

Replace the current 5-phase `_generate_response()` method in `agent_ai_assistant.py` with a call to the `AgentLoop`:

1. In the `_initialize_components()` method (or equivalent initialization), instantiate the `AgentLoop`:
   ```python
   self._agent_loop = AgentLoop(
       nl_processor=self._robust_nl_processor,
       tool_bridge=self._intent_tool_bridge,
       llm_router=self._llm_router,              # LLMRouter from intelligence/llm_router.py
       llm_tool_integration=self._llm_tool_integration,  # LLMToolIntegration or None
       session_context=self._session_context,
       ui_callback=self._show_ai_message
   )
   ```

2. In `_generate_response(message)`:
   - Replace the current PHASE 0 through PHASE 4 cascade with: `result = self._agent_loop.process_message(message)`
   - Display the result via `self._show_ai_message(result)`
   - Update stats and save session (keep existing post-response logic)

   > **Migration note:** The current 5-phase cascade (`_try_direct_backend_operation` → `_try_robust_nl_execution` → `_analyze_and_execute_with_llm` → `_try_execute_agent_command` → `_generate_llm_response`) is deeply integrated into 7,747 lines of code. **Recommended approach:** Rather than deleting all five phases at once, keep the old methods as private fallback methods (rename to `_legacy_try_direct_backend_operation`, etc.) and route through `AgentLoop.process_message()` first. If the agent loop fails or returns no result, fall back to the legacy path. Remove the legacy methods only after the agent loop is verified to handle all cases. This reduces the risk of breaking the entire assistant during migration.

3. The `AgentLoop.process_message()` handles all of:
   - Simple commands (high-confidence intent → direct dispatch)
   - Complex multi-step tasks (plan → confirm → execute)
   - LLM-assisted reasoning (when intent is unclear)
   - Error recovery (fix-and-retry)
   - Context updating (after every operation)

This refactoring eliminates the need for the current Phase 0 (`_try_direct_backend_operation`) and Phase 3 (`_try_execute_agent_command`) which were workarounds for the single-shot architecture. The agentic loop handles all these cases naturally.

### Step 10.4: Implement Streaming Token Display

If the integrated LLM supports streaming (Ollama and most API providers do), enable token-by-token display:

1. In `AgentLoop`, when the LLM is called in the multi-turn loop:
   - Check if the provider actually implements streaming: `stream_send()` is defined as a method on the base `_BaseProvider` class in `llm_router.py` (line 213), and `LLMProvider` Protocol (line 132) declares it. Since `hasattr(provider, 'stream_send')` will **always** return `True` (all providers inherit from `_BaseProvider`), check whether the provider's class actually overrides `stream_send` — use this pattern: `provider_instance = self._llm_router._pick_provider(request)` and check `type(provider_instance).stream_send is not _BaseProvider.stream_send`. Alternatively, maintain a list of known streaming-capable provider names (verified from `llm_router.py` source — only providers that override `stream_send`): `STREAMING_PROVIDERS = {'ollama', 'openai', 'anthropic', 'google_gemini', 'deepseek', 'groq', 'cohere', 'xai', 'lm_studio', 'openrouter'}`.
   - **Note:** `mistral`, `together`, and `huggingface` providers exist in `llm_router.py` but do NOT override `stream_send` — they use the base stub which falls back to non-streaming. Do not include them in `STREAMING_PROVIDERS`.
   - **Note:** `stream_send()` exists on individual provider subclasses (e.g., `OllamaProvider.stream_send`, `OpenAIProvider.stream_send`, `OpenRouterProvider.stream_send`), NOT on `LLMRouter` itself. The `LLMRouter` only has `route()` and `route_with_fallback()`. To stream, you must get the provider instance directly: `provider = self._llm_router._pick_provider(request)`, then call `provider.stream_send(request, api_key, callback)`.
   - Create a generator that yields tokens via the `callback` parameter of `stream_send(request, api_key, callback)` — the callback receives each token string
   - Pass tokens to the UI callback incrementally

2. In `agent_ai_assistant.py`, update `_show_ai_message()` to support incremental display:
   - Create a `StreamingMessageWidget` or modify the existing `WordWrappedRichLog` to accept incremental appends
   - When streaming starts, create a new message entry in the chat
   - As tokens arrive, append them to the same message entry
   - **Textual integration**: Since `_generate_response()` is synchronous, use Textual's `Worker` API (`self.run_worker()`) to run the streaming loop in a background thread. Use `self.call_from_thread(self._append_token, token)` to safely update the UI from the worker thread. Alternatively, use `app.call_later()` for deferred UI updates.
   - Re-render the Rich text after each sentence boundary (. ! ? newline) to avoid excessive re-rendering
   - When streaming completes, finalize the message

3. During streaming, if a tool call is detected in the partial response:
   - Pause the streaming display
   - Show a "⚙️ Executing: {tool_name}..." indicator
   - Execute the tool
   - Show a brief result summary
   - Resume streaming with the next LLM response

### Step 10.5: Implement Context Window Management

For conversations that run long (many tool executions accumulating results), manage the context window to avoid exceeding the model's token limit:

1. Track the approximate token count of the conversation history. Use a simple heuristic: 1 token ≈ 4 characters (for English text).

2. Before each LLM call, check if the **current conversation history** (not cumulative usage) exceeds 80% of the model's context window. Calculate the current size by summing the approximate token count of all messages that will be sent in this request:
   - System prompt tokens
   - Conversation history messages tokens
   - Current user message tokens
   - Expected response buffer (~1,000 tokens)
   
   Model context windows:
   - Ollama models: varies by model (llama2: 4,096; llama3: 8,192; mistral: 32,768; codellama: 16,384). Query via Ollama API `/api/show` for the specific model's `context_length`.
   - GPT-4 / Claude: 128,000+ tokens (effectively unlimited for this use case)
   - Other models: check the `TokenPricing.context_window` field (default 128,000) from `llm_router.py` line 6735

3. If the limit is approaching:
   - Summarize the oldest conversation entries: take the oldest 10 entries, send them to the LLM with "Summarize this conversation so far in 200 words", replace the 10 entries with the summary
   - Alternatively, for non-LLM mode: keep only the last 5 entries plus the SessionContext state

4. Tool execution results are especially long. Truncate them in the conversation history to 500 characters each, but keep the full result available in SessionContext for reference.

---

## Phase 11: Error Handling, Recovery, and Safety

### Objective
Ensure every operation has proper error handling, meaningful error messages, recovery suggestions, and appropriate safety mechanisms for all five functionality domains.

### Step 11.1: Create an Agent Error Handler

Add a new file `src/proxima/agent/agent_error_handler.py`. This module provides agent-specific error classification and recovery strategies.

> **Important:** An `ErrorClassifier` class already exists in `src/proxima/agent/dynamic_tools/error_detection.py` with its own `classify(error: Exception, context: Optional[ErrorContext])` method and the `ErrorCategory` enum (20 values: FILESYSTEM, NETWORK, AUTHENTICATION, PERMISSION, RESOURCE, TIMEOUT, VALIDATION, CONFIGURATION, DEPENDENCY, GIT, GITHUB, TERMINAL, BUILD, RUNTIME, MEMORY, DISK, SYNTAX, LOGIC, CONCURRENCY, UNKNOWN). The new `AgentErrorHandler` class below **wraps** the existing `ErrorClassifier` for classifying raw terminal output (strings + exit codes) and adds recovery strategy logic. Import and reuse `ErrorCategory` from `error_detection.py` — do NOT redefine it.

**The `AgentErrorHandler` class maps terminal output to existing `ErrorCategory` values with recovery strategies:**

- `ErrorCategory.PERMISSION`: File access denied, admin needed, git auth failure
  - Recovery: suggest running with admin privileges or fixing permissions
- `ErrorCategory.AUTHENTICATION`: Git credential failure, API key invalid, SSH key missing
  - Recovery: suggest configuring credentials or SSH keys
- `ErrorCategory.FILESYSTEM`: File, directory, package, branch, command not found
  - Recovery: list similar files/directories, suggest correct spelling
- `ErrorCategory.NETWORK`: DNS failure, timeout, connection refused, SSL error
  - Recovery: check internet, retry with longer timeout
- `ErrorCategory.DEPENDENCY`: Module not found, version conflict, missing system library
  - Recovery: auto-detect and suggest install command (delegates to `ProjectDependencyManager`)
- `ErrorCategory.BUILD`: Compilation error, cmake error, make error
  - Recovery: suggest installing build tools, check logs for specific error
- `ErrorCategory.SYNTAX`: Invalid code produced by modification
  - Recovery: undo the modification via CheckpointManager, show the syntax error
- `ErrorCategory.TIMEOUT`: Command exceeded time limit
  - Recovery: retry with longer timeout or run in background
- `ErrorCategory.RESOURCE` / `ErrorCategory.MEMORY` / `ErrorCategory.DISK`: Disk full, out of memory, GPU not available
  - Recovery: suggest freeing resources, point to resource monitor
- `ErrorCategory.GIT`: Merge conflict, detached HEAD, dirty working tree
  - Recovery: specific git commands to resolve the situation
- `ErrorCategory.RUNTIME`: Unexpected runtime errors, assertion failures, unhandled exceptions
  - Recovery: show stack trace, suggest debugging approach
- `ErrorCategory.VALIDATION`: Invalid input, schema mismatch, config validation failure
  - Recovery: show expected format, suggest corrections
- `ErrorCategory.CONFIGURATION`: Missing config file, invalid settings, misconfigured tool
  - Recovery: suggest creating/fixing config file
- `ErrorCategory.TERMINAL`: Shell error, command not found, process crash
  - Recovery: verify shell availability, suggest alternative commands
- `ErrorCategory.LOGIC` / `ErrorCategory.CONCURRENCY`: Race conditions, deadlocks, logic errors
  - Recovery: suggest code review or debugging
- (User cancellation): No `ErrorCategory` equivalent — handle separately as a control flow event, not an error

**Method `classify_output(error_output: str, exit_code: int) -> Tuple[ErrorCategory, str, Optional[str]]`:**
Returns: (category, human_readable_message, suggested_fix_command)

Note: This method is named `classify_output` (not `classify`) to avoid confusion with the existing `ErrorClassifier.classify()` which takes an `Exception` object.

Uses regex pattern matching on the error output:
- `"Permission denied"` or `"Access is denied"` → `ErrorCategory.PERMISSION`
- `"Authentication failed"` or `"Invalid credentials"` or `"could not read Username"` → `ErrorCategory.AUTHENTICATION`
- `"No such file"` or `"not found"` or `"not recognized"` → `ErrorCategory.FILESYSTEM`
- `"Could not resolve host"` or `"Connection refused"` or `"timed out"` → `ErrorCategory.NETWORK`
- `"ModuleNotFoundError"` or `"No module named"` → `ErrorCategory.DEPENDENCY`
- `"error: "` combined with `"cmake"` or `"make"` or `"build"` → `ErrorCategory.BUILD`
- `"SyntaxError"` or `"IndentationError"` → `ErrorCategory.SYNTAX`
- `"Timed out"` or exit code 124 → `ErrorCategory.TIMEOUT`
- `"No space left"` or `"MemoryError"` → `ErrorCategory.RESOURCE`
- `"CONFLICT"` or `"fatal: "` in git context → `ErrorCategory.GIT`

### Step 11.2: Wrap Tool Execution with Error Handling

In `IntentToolBridge.dispatch()`, wrap every tool execution in a try-except block that:

1. Catches all exceptions (`Exception` and subclasses)
2. Calls `AgentErrorHandler.classify_output(str(e), getattr(e, 'returncode', 1))` to categorize
3. Formats a user-friendly message:
   ```
   "❌ {operation} failed: {human_readable_message}
   
   📋 Details: {first 500 chars of error output}
   
   💡 Suggested fix: {suggested_fix_command or 'No automatic fix available'}"
   ```
4. Logs the full exception traceback to `~/.proxima/logs/agent.log` using Python's `logging` module with `logger.exception()`
5. Returns a `ToolResult(success=False, output=formatted_message)` instead of raising

### Step 11.3: Implement Automatic Retry Logic

In `IntentToolBridge.dispatch()`, after a failed execution:

**For network operations (git clone, pip install):**
- Retry up to 3 times with exponential backoff: wait 2 seconds, then 4 seconds, then 8 seconds between retries
- On each retry, display "🔄 Retry {n}/3..."

**For dependency operations:**
- Retry once after attempting the suggested fix (if available)
- The fix is executed automatically only if risk_level is `low`; for higher risk, ask the user first

**For build operations:**
- Retry once with verbose output enabled (add `--verbose` flag) to get more diagnostic information

**For file operations:**
- No retry (failures are deterministic)
- Immediately report the error

**For timeout:**
- Retry once with doubled timeout
- If still times out, suggest running in background

### Step 11.4: Integrate Error Handler into the Agent Loop

In `AgentLoop.process_message()`:

1. After each tool execution that fails, pass the error through `AgentErrorHandler`
2. If the classification suggests a fix:
   - If an LLM is available, include the error and fix suggestion in the next LLM prompt: "The {operation} failed with error: {error}. Suggested fix: {fix}. Should I apply the fix?"
   - The LLM can then decide: apply the fix and retry, try an alternative approach, or report failure
3. If no LLM is available:
   - If risk_level of the fix is `low`, apply automatically
   - If risk_level is `medium` or higher, present to the user and ask for consent
4. Track retry counts per operation to prevent infinite retry loops (max 3 retries per operation)

### Step 11.5: Implement Comprehensive Consent System Integration

Connect the existing `ConsentRequest` / `ConsentDecision` system from `safety.py` to the agent loop and TUI:

1. In `IntentToolBridge`, before dispatching any operation categorized as requiring consent (from the lists in Phase 4, Step 4.5):
   - Create a `ConsentRequest` with the operation description and risk level
   - Call the consent callback (passed during bridge initialization)
   - Wait for the `ConsentDecision`

2. In `agent_ai_assistant.py`, implement the consent callback:
   - Display a consent dialog in the chat: "⚠️ {risk_emoji} **{operation_name}**\n{description}\nRisk: {risk_level}\n\nProceed? Type 'yes' or 'no'"
   - Risk emojis: low=🟡, medium=🟠, high=🔴, critical=⛔
   - Wait for the user's next message. If it starts with 'y' → `APPROVED`. If 'n' → `DENIED`.
   - Support `APPROVED_SESSION` for "yes, always" responses (applies to all operations of the same type for this session)

3. Track consent history in `SessionContext`:
   - `session_consents: Dict[str, ConsentDecision]` mapping consent_type → decision
   - If a consent_type has `APPROVED_SESSION`, skip the consent dialog for subsequent operations of that type

---

## Phase 12: Complete Walkthrough of Complex Task Execution

### Objective
Verify the entire system works end-to-end by walking through the exact complex examples from the requirements, detailing exactly what each component does at each step.

### Step 12.1: Example A — Clone, Branch, Navigate, Run, Analyze

User types in the AI assistant (key 6):
> "Clone https://github.com/kunal5556/LRET into C:\Users\dell\Pictures\Camera Roll, switch to pennylane-documentation-benchmarking branch, go to benchmarks/pennylane, run pennylane_4q_50e_25s_10n.py, then analyze the results and show in Result Tab"

**System processing:**

1. **RobustNLProcessor.recognize_intent()** fires Layer 2 (Multi-Step Detection). Detects comma separators and "then" keyword. Splits into 5 sub-intents:
   - Sub-intent 1: `GIT_CLONE` (entities: url=`https://github.com/kunal5556/LRET`, path=`C:\Users\dell\Pictures\Camera Roll`)
   - Sub-intent 2: `GIT_CHECKOUT` (entities: branch=`pennylane-documentation-benchmarking`)
   - Sub-intent 3: `NAVIGATE_DIRECTORY` (entities: path=`benchmarks/pennylane`)
   - Sub-intent 4: `RUN_SCRIPT` (entities: script_path=`pennylane_4q_50e_25s_10n.py`)
   - Sub-intent 5: `ANALYZE_RESULTS` + `EXPORT_RESULTS` (no specific entities)

2. **AgentLoop.process_message()** detects MULTI_STEP. Calls `_create_plan_from_intents()` to build an `ExecutionPlan`:
   ```
   Step 1: Clone repository (git clone https://github.com/kunal5556/LRET "C:\Users\dell\Pictures\Camera Roll\LRET")
   Step 2: Switch branch (git checkout pennylane-documentation-benchmarking) [depends on: Step 1]
   Step 3: Navigate to benchmarks/pennylane [depends on: Step 2]
   Step 4: Run pennylane_4q_50e_25s_10n.py [depends on: Step 3]
   Step 5: Analyze results and display in Result Tab [depends on: Step 4]
   ```

3. **Plan presented to user** in chat:
   ```
   📋 Execution Plan (5 steps):
   1. Clone https://github.com/kunal5556/LRET → C:\Users\dell\Pictures\Camera Roll\LRET
   2. Switch to branch: pennylane-documentation-benchmarking
   3. Navigate to: benchmarks/pennylane
   4. Execute: pennylane_4q_50e_25s_10n.py
   5. Analyze results → Result Tab
   
   Proceed? (yes/no)
   ```

4. **User types "yes"**. `PlanExecutor` begins sequential execution:

   **Step 1:** `IntentToolBridge` builds command `git clone https://github.com/kunal5556/LRET "C:\Users\dell\Pictures\Camera Roll\LRET"`. Spawns terminal via `TerminalOrchestrator`. Output streams to Execution Tab. On success: `SessionContext.last_cloned_repo = "C:\Users\dell\Pictures\Camera Roll\LRET"`, `SessionContext.current_directory` updated.

   **Step 2:** `IntentToolBridge` builds command `git checkout pennylane-documentation-benchmarking` with cwd = cloned repo path. Executes. On success: `SessionContext.last_mentioned_branches` updated.

   **Step 3:** `IntentToolBridge` resolves `benchmarks/pennylane` relative to current directory (the cloned repo). Calls `SessionContext.push_directory()`. Updates `current_directory`.

   **Step 4:** `IntentToolBridge` detects `.py` extension. Builds command `python pennylane_4q_50e_25s_10n.py` with cwd = `benchmarks/pennylane`. Spawns a monitored terminal. Output streams to Execution Tab in real time. On completion: full output captured.

   **Step 5:** Collects the script output. If LLM available: sends to LLM with analysis prompt. Parses analysis. If no LLM: performs regex-based metric extraction. Pushes structured result to Result Tab via `ResultsScreen.add_agent_result()`. Displays analysis in chat.

5. **Final report** in chat:
   ```
   ✅ All 5 steps completed successfully
   📊 Results have been sent to the Result Tab (press 3 to view)
   ```

### Step 12.2: Example B — Local Path, Branch, Build, Configure

User types:
> "The LRET repo is at C:\Users\dell\Pictures\Screenshots\LRET. Checkout the cirq-scalability-comparison branch, build and compile it, configure Proxima to use it"

**System processing:**

1. **RobustNLProcessor** detects multi-step with entities:
   - Extracts local path: `C:\Users\dell\Pictures\Screenshots\LRET`
   - Extracts branch: `cirq-scalability-comparison`
   - Detects: checkout, build, configure operations

2. **Execution plan:**
   ```
   Step 1: git checkout cirq-scalability-comparison (in C:\Users\dell\Pictures\Screenshots\LRET)
   Step 2: Detect and install dependencies
   Step 3: Build the backend (follow profile from backend_build_profiles.yaml if available, else pip install -e .)
   Step 4: Configure Proxima to use the backend (update configs/default.yaml)
   ```

3. For Step 2: `ProjectDependencyManager.detect_project_dependencies()` scans the repo for `requirements.txt` / `setup.py` / `pyproject.toml`. Installs any missing packages.

4. For Step 3: `BackendBuilder` is invoked if the backend matches a known profile name. The bridge checks if `cirq-scalability-comparison` maps to the `lret_cirq_scalability` profile in `backend_build_profiles.yaml`. If yes, follows the profile's build steps. If no profile matches, falls back to generic build: `pip install -e .` followed by `python -m pytest tests/ -v`.

5. For Step 4: Calls the existing `_configure_backend_for_proxima()` method from `agent_ai_assistant.py`. This writes a backend entry to `configs/default.yaml`.

### Step 12.3: Example C — Full 7-Step Pipeline

User types:
> "1. Clone https://github.com/kunal5556/LRET into C:\Users\dell\Pictures\Screenshots
> 2. Switch to cirq-scalability-comparison branch
> 3. Install dependencies
> 4. Compile the backend
> 5. Test it
> 6. Configure Proxima to use it"

**System processing:**

1. **RobustNLProcessor** Layer 2 detects numbered list format (digits followed by periods). Splits into 6 sub-intents.

2. **Plan created and confirmed.**

3. **Execution** — each step uses context from the previous:
   - Step 1: `git clone ... "C:\Users\dell\Pictures\Screenshots\LRET"` → updates `last_cloned_repo`
   - Step 2: `git checkout cirq-scalability-comparison` → uses `last_cloned_repo` as cwd
   - Step 3: `ProjectDependencyManager.detect_project_dependencies()` → auto-detects and installs
   - Step 4: Build via `BackendBuilder` or `pip install -e .` + `python setup.py build_ext --inplace`
   - Step 5: `python -m pytest tests/ -v` → runs tests, captures output
   - Step 6: Updates `configs/default.yaml` with the new backend

4. Each step's output flows to the Execution Tab (key 2). Final test results and configuration status flow to the Result Tab (key 3).

---

## Phase 13: Testing and Validation

### Objective
Verify that every component works correctly, every intent is recognized, every operation executes properly, and there are no gaps.

### Step 13.1: Intent Recognition Test Cases

Create a test file `tests/test_intent_recognition.py` that verifies the `RobustNLProcessor` correctly recognizes intents from diverse phrasings. For each intent type, test at least 5 different natural language phrasings:

**INSTALL_DEPENDENCY (at least 5 tests):**
- "install numpy"
- "pip install scipy pandas"
- "add the requests library"
- "I need to install the dependencies"
- "install requirements from requirements.txt"

**RUN_SCRIPT (at least 5 tests):**
- "run pennylane_4q_50e_25s_10n.py"
- "execute the benchmark script"
- "python test_backend.py"
- "run the tests"
- "execute benchmark.sh"

**GIT_CLONE (at least 5 tests):**
- "clone https://github.com/kunal5556/LRET"
- "git clone the LRET repository"
- "clone kunal5556/LRET into my Pictures folder"
- "download the repo from github.com/kunal5556/LRET"
- "get the LRET repository from GitHub"

**NAVIGATE_DIRECTORY (at least 5 tests):**
- "go to src/proxima"
- "cd benchmarks/pennylane"
- "navigate to C:\Users\dell\Pictures"
- "change directory to the benchmarks folder"
- "go back" (should resolve from directory stack)

**PLAN_EXECUTION (at least 5 tests):**
- "plan how to build the LRET backend"
- "what steps do I need to clone and build the repo?"
- "create a plan for setting up cirq"
- "step by step, how should I compile this?"
- "make a plan to install and test qiskit"

**MULTI_STEP (at least 5 tests):**
- "clone the repo, install deps, then build it"
- "1. clone 2. checkout branch 3. build 4. test"
- "first clone LRET, after that switch to the cirq branch, then compile"
- "clone and build the LRET backend"
- "navigate to the project, install requirements, and run the tests"

**TERMINAL_MONITOR (at least 5 tests):**
- "what terminals are running?"
- "show me active processes"
- "monitor the build"
- "how many terminals are active?"
- "what is currently executing?"

**CHECK_DEPENDENCY (at least 5 tests):**
- "is numpy installed?"
- "check if cirq is available"
- "verify that qiskit is installed"
- "what version of scipy do I have?"
- "do I have cmake?"

### Step 13.2: Multi-Step Parsing Test Cases

Test the multi-step parser specifically with complex chained requests:

- Comma-separated: "clone X, checkout Y, build, test"
- "Then" separated: "clone X then build it then test it"
- Numbered list: "1. clone X 2. build 3. configure"
- "And" with verbs: "clone X and build it and test it"
- Mixed: "first clone X, then 1. install deps 2. build 3. test, and finally configure Proxima"

Verify that each test produces the correct number of sub-intents with the correct types and entities.

### Step 13.3: Entity Extraction Test Cases

Test entity extraction specifically:

**URL extraction:**
- "clone https://github.com/kunal5556/LRET" → url = `https://github.com/kunal5556/LRET`
- "clone github.com/user/repo" → url = `https://github.com/user/repo` (auto-prefixed)
- "clone git@github.com:user/repo.git" → url = `git@github.com:user/repo.git`

**Path extraction (Windows):**
- "into C:\Users\dell\Pictures\Camera Roll" → path = `C:\Users\dell\Pictures\Camera Roll`
- "at D:\projects\quantum" → path = `D:\projects\quantum`
- "from ~\Documents\repos" → path expanded to full user home path

**Branch extraction:**
- "switch to pennylane-documentation-benchmarking" → branch = `pennylane-documentation-benchmarking`
- "the cirq-scalability-comparison branch" → branch = `cirq-scalability-comparison`
- "checkout main" → branch = `main`
- Verify that common words ("the", "and", "from", "into") are NOT extracted as branches

**Package extraction:**
- "install numpy scipy pandas" → packages = [`numpy`, `scipy`, `pandas`]
- "pip install cirq>=1.0.0" → packages = [`cirq>=1.0.0`]

### Step 13.4: End-to-End Integration Tests

Create `tests/test_e2e_agent.py` with integration tests that exercise the full pipeline:

**Test 1 — Simple command:** Send "list files in the current directory" through `AgentLoop.process_message()`. Verify that `ListDirectoryTool` is invoked and returns a directory listing.

**Test 2 — Git clone:** Send "clone https://github.com/kunal5556/LRET into /tmp/test-clone". Verify that git clone command is executed (mock subprocess), SessionContext is updated with cloned_repos.

**Test 3 — Multi-step plan:** Send "clone the repo, install dependencies, build it". Verify that a plan with 3 steps is created, presented, and (after mock confirmation) executed in order.

**Test 4 — Error recovery:** Send a command that fails with a dependency error. Verify that `AgentErrorHandler` correctly identifies the error, `ProjectDependencyManager` suggests a fix, and the fix is presented to the user.

**Test 5 — Context resolution:** Send "clone https://github.com/kunal5556/LRET", then send "build it". Verify that "it" resolves to the cloned repository path.

### Step 13.5: Performance Validation

Verify performance meets these targets:

- Intent recognition (Layers 1-3, no LLM) must complete in under 100 milliseconds. Measure using `time.perf_counter()` around `recognize_intent()` calls.
- Tool dispatch (from intent to tool execution start) must begin within 50 milliseconds.
- Entity extraction must handle messages up to 5,000 characters without degradation.
- Multi-step parsing must handle up to 20 sub-steps without degradation.
- Context window management must not lose critical session state during summarization.

---

## Phase 14: Core Integration and Wiring

### Objective
Wire the core agent components together and ensure all modules communicate correctly. (Phases 15 and 16 add session persistence and Crush-inspired capabilities on top of this foundation.)

### Step 14.1: Module Registration and Initialization

In `agent_ai_assistant.py`, update the initialization sequence to instantiate all new components in the correct order:

1. `RobustNLProcessor` — already exists, ensure it has the expanded IntentType enum and keyword mappings from Phase 1
2. `ProjectDependencyManager` — new (distinct from the existing `DependencyManager` in `deployment_monitoring.py`), instantiate and store as `self._dependency_manager`
3. `TerminalOrchestrator` — new, wraps existing `MultiTerminalMonitor`, store as `self._terminal_orchestrator`
4. `IntentToolBridge` — new, takes `ToolRegistry`, `ProjectDependencyManager`, `TerminalOrchestrator`, `CheckpointManager`, `AdminPrivilegeHandler`; store as `self._intent_tool_bridge`
5. `SystemPromptBuilder` — new, store as `self._system_prompt_builder`
6. `AgentErrorHandler` — new (wraps existing `ErrorClassifier` from `error_detection.py`), store as `self._error_handler`
7. `AgentLoop` — new, takes all of the above; store as `self._agent_loop`

Pass the consent callback to `IntentToolBridge` during initialization. The callback is a method on the agent assistant screen that displays consent dialogs.

### Step 14.2: Verify Import Chain

Ensure no circular imports exist. The dependency chain should be:
```
tool_interface.py ← tool_registry.py ← tools/*.py
                  ← execution_context.py
                  ← tool_orchestrator.py ← llm_integration.py
robust_nl_processor.py (standalone, no imports from above)
intent_tool_bridge.py ← robust_nl_processor.py, tool_registry.py, tools/*.py
dependency_manager.py (uses subprocess, tomllib/tomli, pyyaml; contains ProjectDependencyManager)
terminal_orchestrator.py ← multi_terminal.py, terminal_state_machine.py
agent_error_handler.py ← error_detection.py (uses existing ErrorCategory and ErrorClassifier)
system_prompt_builder.py ← tool_registry.py, execution_context.py
agent_loop.py ← all of the above
agent_ai_assistant.py ← agent_loop.py (single entry point)
```

If circular imports threaten, use lazy imports (import inside methods rather than at module top).

### Step 14.3: Wire Execution Tab Messages

Define custom Textual `Message` classes for communication between the agent and the Execution Tab:

1. `AgentTerminalStarted(terminal_id: str, name: str, command: str)` — posted when `TerminalOrchestrator.spawn_terminal()` creates a new process
2. `AgentTerminalOutput(terminal_id: str, line: str)` — posted for each output line
3. `AgentTerminalCompleted(terminal_id: str, success: bool, return_code: int)` — posted when process exits
4. `AgentPlanStarted(plan_id: str, steps: List[Dict])` — posted when plan execution begins
5. `AgentPlanStepCompleted(plan_id: str, step_id: int, success: bool)` — posted after each completed step

These messages are defined in a shared file `src/proxima/tui/messages/agent_messages.py`. **Note:** This directory does not currently exist and must be created. Both `agent_ai_assistant.py` and `execution.py` import from this file.

In `ExecutionScreen`, add handlers:
- `on_agent_terminal_started`: create a new tab or update the main log with "[Starting: {command}]"
- `on_agent_terminal_output`: append line to the log widget
- `on_agent_terminal_completed`: show completion status, update progress bar
- `on_agent_plan_started`: show plan overview in the stage timeline widget
- `on_agent_plan_step_completed`: update the timeline to show progress

### Step 14.4: Wire Result Tab Messages

Define additional messages for the Result Tab:

1. `AgentResultReady(result: Dict[str, Any])` — posted when analysis is complete

In `ResultsScreen`, add a handler:
- `on_agent_result_ready`: parse the result dict, add to the results list, update the display
- The result dict contains: `title`, `timestamp`, `status`, `metrics` (dict of key-value pairs), `output_summary`, `analysis` (if LLM was used)

### Step 14.5: Final Validation Checklist

Run through each required functionality and verify:

| # | Functionality | Verification |
|---|---|---|
| 1 | NL Planning & Execution | Send "plan how to build the LRET backend" → plan is created, presented, and executable |
| 2 | File System Access | Send "create a file test.txt with content hello" → file is created. Send "read test.txt" → content shown. Send "delete test.txt" → consent dialog → file deleted |
| 3 | Script Execution | Send "run tests/test_backend.py" → script executes, output shown in chat and Execution Tab |
| 4 | Dependency Management | Send "install numpy scipy" → packages install. Send "is cirq installed?" → status shown. Clone a repo and send "install dependencies" → auto-detected and installed |
| 5 | GitHub Operations | Send "clone https://github.com/kunal5556/LRET" → cloned. Send "pull latest" → pulled. Send "push changes" → consent, pushed |
| 6 | Backend Build/Compile/Modify | Send "build the cirq backend" → dependencies pre-checked, build steps executed in terminal, verification passed. Send "modify the qiskit backend" → consent requested, checkpoint created, diff previewed, change applied. Send "undo that" → rollback from checkpoint |
| 7 | Admin Access | Send "install numpy globally" → admin access auto-detected, consent dialog shown, UAC/sudo prompt appears, package installed with elevation. Audit log entry created |
| 8 | Streaming to Execution Tab | During any terminal operation → output appears in real-time on key-2 screen |
| 9 | Results to Result Tab | After script completes → results appear on key-3 screen with metrics and analysis |
| 10 | Error Recovery | Send a command that fails → error classified, fix suggested, retry offered |
| 11 | Context Resolution | After clone: "build it" → builds the cloned repo. After navigate: "run the script here" → runs in current dir |
| 12 | Multi-step Complex | Full Example A/B/C from requirements → all steps execute correctly |
| 13 | Session Persistence | Close and reopen Proxima → previous session loads with full context, todos, and history intact |
| 14 | Sub-Agent Delegation | Send "search the web for quantum error correction benchmarks" → sub-agent spawns, searches, returns summary to parent agent |
| 15 | Todo Tracking | During multi-step tasks → agent creates todos, tracks progress, displays pill widget in TUI |
| 16 | Permission System | Send "delete all files in /tmp" → permission dialog appears with Allow/Allow Session/Deny options |

---

## Phase 15: Session Context Handling and Persistence

### Objective
Implement comprehensive session management so the agent maintains proper context across the entire conversation lifecycle, retains context from imported chats, uses session history during execution, and supports auto-summarization when the context window approaches its limit. This directly fulfills Required Functionality #10.

### Step 15.1: Create the Session Manager Module

Create a new file `src/proxima/agent/session_manager.py`. This module handles session persistence, loading, switching, and metadata tracking.

**The `SessionState` dataclass contains:**
```python
@dataclass
class SessionState:
    session_id: str                          # UUID for this session
    title: str                               # Auto-generated or user-set session title
    created_at: float                        # Unix timestamp of session creation
    updated_at: float                        # Unix timestamp of last activity
    message_count: int                       # Total messages in session (user + assistant)
    prompt_tokens: int                       # Cumulative prompt tokens used
    completion_tokens: int                   # Cumulative completion tokens used
    cost: float                              # Estimated cost (if using paid API)
    summary_message_id: Optional[str]        # ID of the summary message (None if not summarized)
    todos: List[TodoItem]                    # Active todo list for this session
    context: SessionContext                  # The full SessionContext from Phase 2
    parent_session_id: Optional[str]         # For sub-agent sessions — links to parent
    is_sub_agent_session: bool               # True if this is a sub-agent task session
    messages: List[SessionMessage]           # Complete message history
```

**The `SessionMessage` dataclass contains:**
```python
@dataclass
class SessionMessage:
    message_id: str                          # UUID for this message
    role: str                                # 'user', 'assistant', or 'tool'
    content: str                             # Text content of the message
    timestamp: float                         # When the message was created
    model: Optional[str]                     # Which model generated this (for assistant messages)
    tool_calls: List[Dict[str, Any]]         # Tool calls made in this message (for assistant)
    tool_results: List[Dict[str, Any]]       # Tool results (for tool role messages)
    is_summary: bool                         # True if this is an auto-generated summary message
    metadata: Dict[str, Any]                 # Additional metadata (token count, etc.)
```

**The `TodoItem` dataclass contains:**
```python
@dataclass
class TodoItem:
    content: str                             # What needs to be done (imperative form, e.g., "Run tests")
    status: str                              # "pending", "in_progress", or "completed"
    active_form: str                         # Present continuous form (e.g., "Running tests")
```

**The `AgentSessionManager` class contains:**

> **Note:** Named `AgentSessionManager` (not `SessionManager`) to avoid collision with the existing `SessionManager` in `multi_terminal.py` (L888) which manages terminal sessions.

**Constructor `__init__(self, storage_dir: str = ".proxima/sessions")`:**
- `storage_dir`: directory for session JSON files (relative to workspace root)
- `_sessions: Dict[str, SessionState]` — in-memory cache of loaded sessions
- `_current_session_id: Optional[str]` — the active session
- Create the storage directory if it does not exist

**Method `create_session(title: str = "Untitled Session") -> SessionState`:**
1. Generate a new UUID for `session_id`
2. Create a `SessionState` with empty message list, fresh `SessionContext`, empty todos
3. Set `created_at` and `updated_at` to `time.time()`
4. Add to `_sessions` cache
5. Set as current session
6. Persist to disk immediately via `_save_session(session_id)`
7. Return the new session

**Method `load_session(session_id: str) -> SessionState`:**
1. Check if already in `_sessions` cache — return if so
2. Load from `{storage_dir}/{session_id}.json`
3. Deserialize all fields including `SessionContext` reconstruction
4. Populate `_sessions` cache
5. Return the session

**Method `switch_session(session_id: str) -> SessionState`:**
1. Save the current session if one is active
2. Load the requested session (from cache or disk)
3. Set as `_current_session_id`
4. Restore the `SessionContext` from the loaded session
5. Return the session

**Method `list_sessions() -> List[Dict[str, Any]]`:**
1. Scan `{storage_dir}/*.json` files
2. For each, load only metadata (id, title, updated_at, message_count) without full message history
3. Return sorted by `updated_at` descending (most recent first)

**Method `delete_session(session_id: str) -> None`:**
1. Remove from `_sessions` cache
2. Delete `{storage_dir}/{session_id}.json`
3. If this was the current session, clear `_current_session_id`

**Method `import_session(file_path: str) -> SessionState`:**
1. Read the JSON file from `file_path` (supports the `exports/ai_conversation_*.json` format)
2. Parse the conversation into `SessionMessage` objects:
   - Map the export format's message structure to `SessionMessage` fields
   - Reconstruct `SessionContext` from the imported conversation (extract mentioned paths, packages, repos, etc.)
3. Create a new `SessionState` with the imported messages
4. Set the title from the export file metadata or auto-generate from the first message
5. Persist and return
6. **Critical**: The imported context must be retained — all paths, repos, packages, branches mentioned in the imported conversation must populate the `SessionContext` so subsequent agent operations can reference them

**Method `add_message(message: SessionMessage) -> None`:**
1. Append to `current_session.messages`
2. Increment `message_count`
3. Update `updated_at`
4. Auto-save every 5 messages (batch persistence to reduce I/O)

**Method `get_current_session() -> Optional[SessionState]`:**
Return `_sessions.get(_current_session_id)` or `None`.

**Method `_save_session(session_id: str) -> None`:**
1. Serialize the full `SessionState` to JSON
2. Write atomically: write to `{session_id}.tmp`, then rename to `{session_id}.json`
3. This prevents corruption from interrupted writes

> **Serialization Note:** `SessionContext` contains an `Intent` object (`last_operation`) which uses `IntentType` enum values and `ExtractedEntity` dataclasses. These are NOT directly JSON-serializable. Implement a custom serialization strategy:
> - `IntentType` → serialize as its `.name` string (e.g., `"GIT_CLONE"`), deserialize via `IntentType[name]`
> - `ExtractedEntity` → serialize as a dict with `entity_type`, `value`, `confidence`, `source` keys
> - `Intent` → serialize as a dict with `intent_type` (name string), `confidence`, `entities` (list of dicts), `original_message`
> - `operation_history` → serialize as a list of Intent dicts (keep only the most recent 20 for size control)
> - Use a helper pair: `session_context_to_dict(ctx: SessionContext) -> dict` and `session_context_from_dict(data: dict) -> SessionContext`

**Method `_generate_title(messages: List[SessionMessage]) -> str`:**
1. If an LLM is available, send the first user message with prompt: "Generate a concise title (max 8 words) for this conversation: {first_message}"
2. If no LLM available, use the first 8 words of the first user message
3. Strip think tags from the response if present

### Step 15.2: Implement Auto-Summarization

Add summarization capability to the `AgentSessionManager` that triggers automatically when the context window approaches its limit. This follows the Crush pattern of replacing old messages with a summary message while preserving critical state.

**Add to `AgentSessionManager`:**

**Constants:**
```python
LARGE_CONTEXT_WINDOW_THRESHOLD = 200_000   # tokens — models with context > 200K
LARGE_CONTEXT_WINDOW_BUFFER = 20_000       # tokens — buffer for large context models
SMALL_CONTEXT_WINDOW_RATIO = 0.2           # summarize when 20% of context remains
DEFAULT_CONTEXT_WINDOW = 4_096             # default for unknown models (e.g., Ollama llama2)
```

**Method `should_summarize(model_context_window: int) -> bool`:**
1. Calculate the **current conversation size** by estimating tokens for all messages that would be sent to the LLM in the next request. Use `self.get_messages_for_llm()` and estimate: `current_tokens = sum(len(msg.content) // 4 for msg in messages)` (1 token ≈ 4 chars heuristic). Do NOT use `session.prompt_tokens + session.completion_tokens` — those are cumulative lifetime values, not the current conversation size.
2. Calculate remaining: `remaining = model_context_window - current_tokens`
3. If `model_context_window > LARGE_CONTEXT_WINDOW_THRESHOLD`:
   - Return `True` if `remaining <= LARGE_CONTEXT_WINDOW_BUFFER`
4. Else:
   - Return `True` if `remaining <= model_context_window * SMALL_CONTEXT_WINDOW_RATIO`

**Method `summarize_session(llm_router, session_id: str) -> SessionMessage`:**
1. Get all messages for the session
2. If `session.summary_message_id` is set, start from that message onward (don't re-summarize old summaries)
3. Build a summarization prompt using the `summary.md` template:

```
You are summarizing a conversation to preserve context for continuing work later.

**Critical**: This summary will be the ONLY context available when the conversation resumes. Assume all previous messages will be lost. Be thorough.

**Required sections**:

## Current State
- What is the current state of the project/task?
- What files have been modified?
- What is working and what isn't?

## Files & Changes
- List all files that were created, modified, or deleted
- Note the purpose of each change

## Technical Context
- What languages, frameworks, tools are being used?
- What environment setup has been done?
- What dependencies were installed?

## Strategy & Approach
- What approach was taken?
- What alternatives were considered?

## Exact Next Steps
- What specific actions should be taken next?
- In what order?
```

4. If the session has active todos, append them to the prompt:
```
## Current Todo List
- [pending] Task 1 description
- [in_progress] Task 2 description
- [completed] Task 3 description

Include these tasks and their statuses in your summary.
Instruct the resuming assistant to use the todos tool to continue tracking progress on these tasks.
```

5. Send the summarization prompt with all conversation messages to the LLM
6. Create a new `SessionMessage` with `role='assistant'`, `is_summary=True`
7. Update `session.summary_message_id` to point to this message
8. Reset `session.prompt_tokens` to 0 (the summary replaces all prior context)
9. `session.completion_tokens` = token count of the summary
10. Save the session

**Method `get_messages_for_llm(session_id: str) -> List[SessionMessage]`:**
1. If `session.summary_message_id` is set:
   - Find the summary message index
   - Return messages from summary onward (treat summary as a user message to the LLM)
2. Else:
   - Return all messages
3. Truncate tool results in returned messages to 500 characters each (keep full results in the stored session)

### Step 15.3: Integrate Session Manager into the Agent Loop

Modify `agent_loop.py` (from Phase 10) to use the `AgentSessionManager`:

1. **In `AgentLoop.__init__`**, accept an additional parameter `session_manager: AgentSessionManager`

2. **At the start of `process_message()`:**
   - If no current session exists, create one: `self.session_manager.create_session()`
   - Add the user message: `self.session_manager.add_message(SessionMessage(role='user', content=message, ...))`

3. **After each assistant response:**
   - Add the assistant message: `self.session_manager.add_message(SessionMessage(role='assistant', content=response, ...))`
   - Update token counts on the session from the LLM response usage metadata

4. **After each tool execution:**
   - Add a tool message: `self.session_manager.add_message(SessionMessage(role='tool', content=result, tool_results=[...]))`

5. **Before each LLM call in the agentic loop:**
   - Check `self.session_manager.should_summarize(model_context_window)`
   - If True:
     - Call `self.session_manager.summarize_session(self.llm_router, session_id)`
     - Display a brief notification in the TUI: "Session summarized to preserve context"
     - If the agent was mid-task (tool calls pending), re-queue the current task with prompt: "The previous session was interrupted because it got too long. The initial user request was: `{original_prompt}`"
   - Use `self.session_manager.get_messages_for_llm()` to get the conversation history for the LLM call

6. **On agent initialization (app startup):**
   - Load the most recent session if one exists
   - Restore `SessionContext` from the loaded session
   - Display session title and message count in the TUI

### Step 15.4: Add Session UI Controls to TUI

In `agent_ai_assistant.py`, add session management commands that are accessible via the chat interface:

**New Session:** User types `/new` or presses `Ctrl+N`:
1. Save the current session
2. Clear the chat history display
3. Create a new session via `AgentSessionManager.create_session()`
4. Reset the `SessionContext`
5. Display "New session started"

**Switch Session:** User types `/sessions` or presses `Ctrl+S`:
1. Call `AgentSessionManager.list_sessions()` to get available sessions
2. Display a selection dialog (similar to Textual's `OptionList` widget) showing:
   - Session title
   - Last activity time (humanized, e.g., "2 hours ago")
   - Message count
3. On selection, call `AgentSessionManager.switch_session(session_id)`
4. Load and display the session's message history in the chat
5. Restore the `SessionContext`

**Import Session:** User types `/import <path>`:
1. Call `AgentSessionManager.import_session(path)`
2. Load and display the imported conversation
3. Display "Imported session: {title} ({message_count} messages)"

**Summarize Session:** User types `/summarize`:
1. Call `AgentSessionManager.summarize_session()` explicitly
2. Display the generated summary in the chat
3. Show "Session summarized — older messages compressed"

**Delete Session:** User types `/delete`:
1. Show confirmation dialog: "Delete current session? This cannot be undone."
2. On confirm: `AgentSessionManager.delete_session(current_session_id)`
3. Create a new empty session

### Step 15.5: Session Title Auto-Generation

When a new session receives its first user message:

1. In `AgentSessionManager.add_message()`, check if this is the first user message (message_count was 0)
2. If so, asynchronously generate a title:
   - Prefer the small model (if dual-model is configured) for cost efficiency
   - Prompt: "Generate a concise title (max 8 words) for this conversation: {first_message}"
   - Set max output tokens to 40
   - Strip any `<think>` tags from the response
   - Update `session.title` with the generated title
3. If no LLM is available, use the first 8 words of the first user message, truncated with "..."
4. Update the TUI title bar to display the session title

### Step 15.6: Session Manager Tests

Create `tests/test_session_manager.py` with the following test cases:

1. **Session persistence:** Create a session, add messages, save it. Load it back and verify all messages, metadata, and timestamps are preserved.
2. **Session listing and switching:** Create 3 sessions, list them, switch between them. Verify the correct session is active after each switch.
3. **Session import:** Create a mock JSON file in `exports/` format, import it via `AgentSessionManager.import_session()`. Verify the imported messages appear correctly.
4. **Session deletion:** Create a session, delete it, verify it no longer appears in `list_sessions()` and its file is removed from disk.
5. **Auto-summarization trigger:** Mock having a session with messages that exceed 80% of a small context window. Verify `_should_summarize()` returns `True` and `_summarize_old_messages()` produces a valid summary prefix.
6. **Title auto-generation:** Add a first message to a new session. Verify that a title is generated (mock the LLM response) and stored on the session.
7. **Context window calculation:** Verify `get_messages_for_llm()` returns messages that fit within the model's context window, with older messages summarized.

---

## Phase 16: Additional Agent Capabilities (Crush-Inspired)

### Objective
Incorporate select capabilities from the Crush AI agent architecture that enhance Proxima's agent with sub-agent delegation, web research, structured task tracking, a tool permission system, dangerous command blocking, and dual-model support. This directly fulfills Required Functionality #11.

### Step 16.1: Implement the Sub-Agent System

Create a new file `src/proxima/agent/sub_agent.py`. This module enables the main agent to spawn lightweight sub-agents for search, context gathering, and web research tasks.

**The `SubAgentConfig` dataclass contains:**
```python
@dataclass
class SubAgentConfig:
    name: str                                # Descriptive name (e.g., "Search Agent", "Fetch Analysis")
    allowed_tools: List[str]                 # Restricted tool set (read-only tools only)
    model_preference: str                    # "small" or "large" — which model to use
    max_iterations: int                      # Maximum agentic loop iterations (default: 10)
    auto_approve_permissions: bool           # Always True for sub-agents
    parent_session_id: str                   # Links to the parent session for UI display
    timeout_seconds: int                     # Maximum wall-clock time (default: 120)
```

**Default read-only tool set for sub-agents:**
```python
SUB_AGENT_READ_ONLY_TOOLS = [
    "ReadFileTool",          # View file contents
    "ListDirectoryTool",     # List directories
    "SearchFilesTool",       # Grep/search in files
    "FileInfoTool",          # Get file metadata
    "GitStatusTool",         # Check git status
    "GitLogTool",            # View git history
    "GitDiffTool",           # View git diffs
    "RunCommandTool",        # ONLY for read-only commands (ls, cat, head, tail, find, grep)
]
```

**The `SubAgent` class contains:**

**Constructor `__init__(self, config: SubAgentConfig, nl_processor, tool_registry, llm_router, session_manager)`:**
- Creates a filtered `IntentToolBridge` that only has access to `config.allowed_tools`
- Creates a new `SessionContext` (isolated from parent)
- Creates a sub-agent session via `session_manager.create_sub_session(parent_id, title)`

**Method `run(prompt: str) -> str`:**
1. Create a system prompt specific to sub-agents:
   ```
   You are a research sub-agent. Your task is to find and return information.
   You can ONLY read files, search, and browse — you CANNOT modify anything.
   Be concise and return only the relevant information requested.
   ```
2. Run a simplified agentic loop (same structure as `AgentLoop.process_message()` but with:
   - Restricted tool set (only read-only tools)
   - Smaller iteration limit (`config.max_iterations`)
   - Auto-approved permissions
   - Timeout enforcement
3. Return the final response text

**Integration with `AgentLoop`:**

Add a new method to `AgentLoop`:

**Method `_spawn_sub_agent(prompt: str, name: str = "Research Agent") -> str`:**
1. Create a `SubAgentConfig` with read-only tools and small model preference
2. Instantiate `SubAgent`
3. Show a TUI indicator: "🔍 Sub-agent: {name}..."
4. Run the sub-agent with the prompt
5. Show completion: "Sub-agent completed"
6. Return the result

The main agent can spawn sub-agents when:
- The user asks to search for something across the codebase ("find all files that import numpy")
- The intent classifier identifies a research/search task that doesn't require modification
- The web fetch tool needs to analyze fetched content (agentic fetch pattern)
- A complex task requires gathering context from multiple files before making changes

### Step 16.2: Implement the Agentic Fetch Tool

Create a new file `src/proxima/agent/dynamic_tools/tools/web_tools.py`. This module provides web search and content fetching capabilities.

**The `WebFetchTool` class (registered as `web_fetch`):**

> **Note:** All new tools must inherit from `BaseTool` (not `ToolInterface` directly) and use `@property` methods for metadata, matching the convention used by all existing tools (`ReadFileTool`, `GitStatusTool`, etc.).

```python
@register_tool
class WebFetchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch content from a URL and return the text"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SYSTEM

    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.NETWORK  # Requires user consent — network access

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="url", param_type=ParameterType.URL, description="The URL to fetch", required=True),
            ToolParameter(name="timeout", param_type=ParameterType.INTEGER, description="Timeout in seconds (max 60)", required=False, default=30),
        ]
```

**Implementation of `execute()`:**
1. Validate URL starts with `http://` or `https://`
2. Make an HTTP GET request using `urllib.request` (stdlib, no external dependency)
3. Set User-Agent header: `"Proxima/1.0"`
4. Enforce timeout from parameter (max 60 seconds)
5. Read response body (max 1MB)
6. If content is HTML, extract text using a simple tag-stripping approach:
   - Remove `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>` blocks
   - Strip remaining HTML tags
   - Collapse whitespace
   - Truncate to 50,000 characters
7. If content exceeds 50KB (`LARGE_CONTENT_THRESHOLD`):
   - Save full content to a temporary file in `.proxima/fetch_cache/`
   - Return a truncated version (first 5,000 chars) with note: "Full content saved to: {temp_path}"
8. Return the cleaned text content

**The `AgenticFetchTool` class (registered as `agentic_fetch`):**

This tool spawns a sub-agent to analyze web content, similar to Crush's `agentic_fetch`:

```python
@register_tool
class AgenticFetchTool(BaseTool):
    @property
    def name(self) -> str:
        return "agentic_fetch"

    @property
    def description(self) -> str:
        return "Search the web and/or fetch a URL, then analyze the content with an AI sub-agent"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SYSTEM

    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.NETWORK  # Network access required

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.LOW

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="url", param_type=ParameterType.URL, description="URL to fetch (optional — if omitted, searches the web)", required=False),
            ToolParameter(name="prompt", param_type=ParameterType.STRING, description="What information to find or extract", required=True),
        ]
```

**Implementation of `execute()`:**
1. If `url` is provided: fetch the URL using `WebFetchTool`
2. If `url` is not provided: this is a web search request — use a search engine API or fallback
3. Create a sub-agent (via `SubAgent`) with:
   - Tools: `web_fetch` (for following links), `ReadFileTool` (for cached content), `SearchFilesTool` (for grep in fetched content)
   - Model: prefer small model for cost efficiency
   - Prompt: combine the fetched content with the user's prompt:
     ```
     The user wants to know: {prompt}

     Here is the web content from {url}:
     ---
     {fetched_content}
     ---

     Analyze this content and provide a clear, concise answer to the user's question.
     If the content doesn't contain the answer, say so clearly.
     ```
4. Auto-approve all permissions within the sub-agent session
5. Return the sub-agent's analysis

### Step 16.3: Implement the Tool Permission System

Create a new file `src/proxima/agent/tool_permissions.py`. This module provides a Crush-inspired permission layer that **wraps and extends** the existing `SafetyManager` from `safety.py`. The existing `SafetyManager` already has `is_blocked()`, `is_safe_operation()`, `requires_consent()`, `request_consent()`, `respond_to_consent()`, and `wait_for_consent()` methods, plus built-in `BLOCKED_PATTERNS` and `SAFE_OPERATIONS` lists. The `ToolPermissionManager` adds: YOLO mode, an extended blocked-commands list, the three-option consent flow (Allow Once / Allow for Session / Deny), and configurable allowed-tools from YAML.

> **Important:** Do NOT duplicate the blocked command patterns or consent workflow that already exists in `SafetyManager`. Instead, compose `SafetyManager` inside `ToolPermissionManager` and delegate to it.

**The `ToolPermissionManager` class contains:**

**Constructor `__init__(self, safety_manager: SafetyManager, config: Dict[str, Any])`:**
- `_safety: SafetyManager` — the existing SafetyManager instance (delegate to it for blocking, consent)
- `allowed_tools: List[str]` — tools that never require permission (loaded from config YAML, extends SafetyManager's `SAFE_OPERATIONS`)
- `session_permissions: Dict[str, List[str]]` — session-scoped auto-approvals (session_id → list of tool:action keys)
- `skip_all: bool` — YOLO mode flag (auto-approve everything except blocked commands)
- `_extended_blocked: List[str]` — additional blocked patterns beyond SafetyManager's built-in list
- `pending_requests: Dict[str, threading.Event]` — active permission requests awaiting user response (**Note:** uses `threading.Event` NOT `asyncio.Event`, since the agent loop is synchronous)

**Default `allowed_tools` (extends SafetyManager's SAFE_OPERATIONS):**
```python
DEFAULT_ALLOWED_TOOLS = [
    "ReadFileTool",
    "ListDirectoryTool",
    "SearchFilesTool",
    "FileInfoTool",
    "GetWorkingDirectoryTool",
    "GitStatusTool",
    "GitLogTool",
    "GitDiffTool",
]
```

**Extended `blocked_commands` list (supplements SafetyManager's BLOCKED_PATTERNS):**
```python
EXTENDED_BLOCKED_COMMANDS = [
    "rm -rf /*", "rm -rf ~",
    "mkfs.", "dd if=", "dd of=/dev/",
    "chmod -R 000 /",
    "poweroff",
    "init 0", "init 6",
    "> /dev/sda", "cat /dev/zero >",
    "mv / /dev/null",
    "wget -O- | sh", "curl | sh", "curl | bash",     # Piped execution from web
    "python -c 'import os; os.remove",                # Destructive Python one-liners
    "del /F /S /Q C:\\",                               # Windows destructive
]
# Note: "rm -rf /", "format", ":(){:|:&};:", "shutdown", "reboot", "del /f /s /q C:\\"
# are already in SafetyManager.BLOCKED_PATTERNS — no duplication needed
```

**Method `check_permission(session_id: str, tool_name: str, action: str, params: Dict) -> PermissionResult`:**

The `PermissionResult` is an enum: `ALLOWED`, `DENIED`, `NEEDS_CONSENT`.

1. If the tool is `RunCommandTool` and the command matches ANY blocked pattern (check both `self._safety.is_blocked(command)` and `self._extended_blocked`) → return `DENIED` (cannot be overridden, even in YOLO mode)
2. If `skip_all` is True → return `ALLOWED` (YOLO mode, skips all other checks)
3. Check if tool_name is in `allowed_tools` → return `ALLOWED`
4. Check if `self._safety.is_safe_operation(tool_name)` → return `ALLOWED`
5. Check if `{tool_name}:{action}` is in `session_permissions[session_id]` → return `ALLOWED`
6. Check if the command is a safe read-only command (matches `SAFE_READ_COMMANDS = ["ls", "cat", "head", "tail", "find", "grep", "wc", "file", "stat", "echo", "pwd", "whoami", "uname", "date", "hostname"]`) → return `ALLOWED`
7. Otherwise → return `NEEDS_CONSENT`

**Method `request_consent(session_id: str, tool_name: str, action: str, description: str, params: Dict) -> bool`:**
1. Delegate to `self._safety.request_consent()` to create a `ConsentRequest`:
   ```
   Tool: {tool_name}
   Action: {action}
   Description: {description}
   Parameters: {formatted_params}
   ```
2. Display a consent dialog in the TUI with three options:
   - **Allow** (once) — grants permission for this single execution (`ConsentDecision.APPROVED_ONCE`)
   - **Allow for Session** — adds `{tool_name}:{action}` to `session_permissions[session_id]`, auto-approves future identical requests (`ConsentDecision.APPROVED_SESSION`)
   - **Deny** — blocks this execution, returns False (`ConsentDecision.DENIED`)
3. Wait for user response via `threading.Event` (synchronous wait)
4. Return True if allowed, False if denied

**Integration with `IntentToolBridge`:**

In `IntentToolBridge.dispatch()` (from Phase 3), before executing any tool:

1. Call `tool_permissions.check_permission(session_id, tool.name, action, params)`
2. If `ALLOWED` → proceed with execution
3. If `DENIED` → return error message explaining why the command is blocked and suggesting a safer alternative
4. If `NEEDS_CONSENT` → call `tool_permissions.request_consent(...)`:
   - If user approves → proceed
   - If user denies → return "Operation cancelled by user"

### Step 16.4: Implement the Todos Tool

Create a new file `src/proxima/agent/dynamic_tools/tools/todos_tool.py`. This provides a structured todo list that the agent can use during complex multi-step tasks.

**The `TodosTool` class (registered via `@register_tool`):**
```python
@register_tool
class TodosTool(BaseTool):
    @property
    def name(self) -> str:
        return "todos"

    @property
    def description(self) -> str:
        return """Creates and manages a structured task list for tracking progress on complex, multi-step coding tasks.
    Use this tool proactively for:
    - Complex multi-step tasks requiring 3+ distinct steps
    - After receiving new instructions to capture requirements
    - When starting work on a task (mark as in_progress BEFORE beginning)
    - After completing a task (mark completed immediately)
    Do NOT use for single, trivial tasks."""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SYSTEM

    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_ONLY  # No consent needed — it's just tracking

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.NONE

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="todos",
                param_type=ParameterType.ARRAY,
                description="The complete updated todo list. Each item is a JSON object with 'content' (imperative string), 'status' ('pending'|'in_progress'|'completed'), and 'active_form' (present continuous string).",
                required=True,
                inference_hint="Array of {content, status, active_form} objects. At most one item may be in_progress at a time."
            ),
        ]
```

**Implementation of `execute()`:**
1. Get the current session from context
2. Parse the `todos` parameter into `List[TodoItem]`
3. Validate:
   - Each todo has content, status, and active_form
   - Status is one of: "pending", "in_progress", "completed"
   - At most one todo is "in_progress" at a time
4. Determine what changed compared to the session's existing todos:
   - `just_completed`: todos that changed from non-completed to completed
   - `just_started`: the todo that changed to in_progress (if any)
   - `is_new`: True if the session had no previous todos
5. Update `session.todos` with the new list
6. Save the session
7. Return a structured response:
   ```
   Todo list updated successfully.
   Status: {pending_count} pending, {in_progress_count} in progress, {completed_count} completed
   ```
8. Include metadata in the response for the TUI to render:
   ```python
   metadata = {
       "is_new": is_new,
       "todos": todos,
       "just_completed": just_completed,
       "just_started": just_started,
       "completed": completed_count,
       "total": len(todos),
   }
   ```

**TUI Integration:**

In `agent_ai_assistant.py`, add a todo progress indicator:

1. When the todos tool is invoked, update a `TodoPillWidget` in the TUI header area:
   - Display format: "To-Do 3/7" (completed/total)
   - If a task is in-progress, append the active_form: "To-Do 3/7 · Running tests"
   - Use a spinner animation while a task is in-progress and the agent is busy
2. The pill widget is always visible when todos exist, collapsed when all are completed
3. When the user clicks/selects the pill, expand to show the full todo list:
   - ✓ Completed tasks (dimmed)
   - → In-progress task (highlighted with spinner)
   - ○ Pending tasks

**Agent Behavior with Todos:**

Update `SystemPromptBuilder` (Phase 10) to include todo instructions in the system prompt:

```
## Task Tracking

You have a `todos` tool for managing structured task lists. Use it when:
- Starting work on a complex task (create the plan)
- Beginning each step (mark as in_progress before starting)
- Completing each step (mark as completed immediately after)
- The user asks for multiple things at once

Rules:
- Exactly ONE task in_progress at any time
- Mark tasks completed IMMEDIATELY after finishing
- Never print todo lists in your response — the user sees them in the UI
- Include both 'content' (imperative) and 'active_form' (present continuous) for each task
```

### Step 16.5: Implement Dual-Model Support

Extend the LLM integration to support configuring separate models for different purposes.

**Create a new file `src/proxima/agent/dual_model_router.py`:**

> **Note:** This is a NEW file in the `agent/` directory (NOT in `llm_integration.py`), because `DualModelRouter` wraps `LLMRouter` from `src/proxima/intelligence/llm_router.py` — a different package. Placing it in `llm_integration.py` would create an awkward cross-package dependency.

Add a `ModelRole` enum:
```python
class ModelRole(Enum):
    LARGE = "large"    # Complex reasoning, code generation, main agent
    SMALL = "small"    # Title generation, summarization, sub-agents
```

**The `DualModelRouter` class:**

> **Note:** `LLMRouter.__init__` takes `(settings: Settings | None = None, consent_prompt: Callable[[str], bool] | None = None)`. The `DualModelRouter` creates two routers by cloning the base `Settings` and overriding the provider/model fields for each role.

```python
class DualModelRouter:
    def __init__(self, settings: Settings, consent_prompt: Callable[[str], bool] | None = None):
        self._consent_prompt = consent_prompt
        # Large model router uses the base settings (settings.llm.provider + settings.llm.model)
        self._large = LLMRouter(settings=settings, consent_prompt=consent_prompt)
        # Small model router — read 'agent.models.small' from configs/default.yaml.
        # Access via settings object: settings has an 'agent' section once default.yaml is updated.
        # The settings object structure mirrors the YAML: settings.agent['models']['small']
        agent_cfg = getattr(settings, 'agent', None) or {}
        models_cfg = agent_cfg.get('models', {}) if isinstance(agent_cfg, dict) else {}
        small_cfg = models_cfg.get('small', None)
        if small_cfg and isinstance(small_cfg, dict):
            small_settings = settings.model_copy(deep=True)
            small_settings.llm.provider = small_cfg.get('provider', settings.llm.provider)
            small_settings.llm.model = small_cfg.get('model', settings.llm.model)
            self._small = LLMRouter(settings=small_settings, consent_prompt=consent_prompt)
        else:
            self._small = self._large  # Use same model for both if no small model configured

    def get_router(self, role: ModelRole = ModelRole.LARGE) -> LLMRouter:
        if role == ModelRole.SMALL:
            return self._small
        return self._large

    def set_models(self, settings: Settings, small_settings: Settings | None = None):
        self._large = LLMRouter(settings=settings, consent_prompt=self._consent_prompt)
        if small_settings:
            self._small = LLMRouter(settings=small_settings, consent_prompt=self._consent_prompt)
        else:
            self._small = self._large
```

**Usage pattern:**
- `AgentLoop` uses `ModelRole.LARGE` for the main reasoning loop
- `SubAgent` uses `ModelRole.SMALL` for search and context tasks
- `AgentSessionManager.summarize_session()` uses `ModelRole.SMALL` (summaries are cost-sensitive; the small model is sufficient for summarization since the conversation content provides all needed context)
- `AgentSessionManager._generate_title()` uses `ModelRole.SMALL` (titles are simple)
- `AgenticFetchTool` uses `ModelRole.SMALL` for web content analysis

**Configuration in `configs/default.yaml`:**
```yaml
agent:
  models:
    large:
      provider: "ollama"
      model: "llama2-uncensored"
      context_window: 4096
      max_tokens: 2048
    small:
      provider: "ollama"
      model: "llama2-uncensored"  # Same model if only one available
      context_window: 4096
      max_tokens: 1024
  permissions:
    allowed_tools:
      - ReadFileTool
      - ListDirectoryTool
      - SearchFilesTool
      - FileInfoTool
      - GetWorkingDirectoryTool
      - GitStatusTool
      - GitLogTool
      - GitDiffTool
    yolo_mode: false
  session:
    storage_dir: ".proxima/sessions"
    auto_summarize: true
    max_sessions: 50
```

### Step 16.6: Integrate Everything into the Agent Loop

Update `agent_loop.py` to incorporate all Phase 16 capabilities:

1. **In `AgentLoop.__init__`**, accept additional parameters:
   - `sub_agent_factory: Callable` — factory function for creating sub-agents
   - `tool_permissions: ToolPermissionManager` — the permission manager
   - `dual_model_router: DualModelRouter` — for model selection

2. **In `process_message()`**, before tool execution:
   - Check permissions via `tool_permissions.check_permission()`
   - Handle `DENIED` gracefully with explanation
   - Handle `NEEDS_CONSENT` with UI dialog

3. **In `process_message()`**, when a research/search task is identified:
   - Spawn a sub-agent via `_spawn_sub_agent()` instead of executing inline
   - Sub-agent results are fed back into the main loop as tool results

4. **In `process_message()`**, for complex tasks:
   - Create a todo list via `TodosTool` at the start of planning
   - Update todo status as each step begins and completes
   - Include todos in conversation context so the LLM tracks progress

5. **In `process_message()`**, use `dual_model_router.get_router(ModelRole.LARGE)` for the main reasoning loop

### Step 16.7: Tool Permissions Tests

Create `tests/test_tool_permissions.py` with the following test cases:

1. **Allowlist:** Configure an allowlist of tool names. Verify that allowed tools return `ALLOWED` from `check_permission()` and non-listed tools return `NEEDS_CONSENT`.
2. **Blocklist:** Configure a blocklist with glob patterns (e.g., `"rm -rf *"`). Verify that blocked commands return `DENIED` with explanation.
3. **Session-scoped auto-approval:** Grant consent for a tool. Verify that subsequent calls for the same tool within the session return `ALLOWED` without re-prompting. Verify the approval does NOT persist after session reset.
4. **Consent flow:** Trigger a `NEEDS_CONSENT` result, simulate user approval, verify the tool is then auto-approved for the rest of the session.
5. **SafetyManager delegation:** Verify that `ToolPermissionManager` correctly delegates risk assessment to `SafetyManager.is_blocked()` and `SafetyManager.requires_consent()`.
6. **Dangerous command blocking:** Verify that `SafetyManager.BLOCKED_PATTERNS` patterns (e.g., `rm -rf /`) are caught and blocked before tool execution.

---

### New Files to Create

| File | Phase | Description |
|---|---|---|
| `src/proxima/agent/dynamic_tools/intent_tool_bridge.py` | 3 | Central dispatcher mapping intents to tool executions |
| `src/proxima/agent/dependency_manager.py` | 5 | ProjectDependencyManager for detecting and installing project dependencies |
| `src/proxima/agent/terminal_orchestrator.py` | 9 | Unified terminal spawning, monitoring, and output management |
| `src/proxima/agent/dynamic_tools/system_prompt_builder.py` | 10 | Constructs LLM system prompts with current state and capabilities |
| `src/proxima/agent/dynamic_tools/agent_loop.py` | 10 | Core agentic loop replacing the single-shot response pipeline |
| `src/proxima/agent/agent_error_handler.py` | 11 | Agent-specific error classification wrapping existing ErrorClassifier |
| `src/proxima/tui/messages/agent_messages.py` | 14 | Custom Textual messages for inter-screen communication |
| `tests/test_intent_recognition.py` | 13 | Intent recognition test suite |
| `tests/test_e2e_agent.py` | 13 | End-to-end integration tests |
| `src/proxima/agent/session_manager.py` | 15 | Session persistence, loading, switching, import, auto-summarization |
| `src/proxima/agent/sub_agent.py` | 16 | Sub-agent delegation for read-only research/search tasks |
| `src/proxima/agent/dynamic_tools/tools/web_tools.py` | 16 | WebFetchTool and AgenticFetchTool for web content retrieval |
| `src/proxima/agent/tool_permissions.py` | 16 | Per-tool consent system with session-scoped auto-approval |
| `src/proxima/agent/dynamic_tools/tools/todos_tool.py` | 16 | Structured todo/task tracking tool |
| `tests/test_session_manager.py` | 15 | Session persistence, import, summarization tests |
| `tests/test_tool_permissions.py` | 16 | Permission system tests (allowlist, blocklist, consent flow) |
| `src/proxima/agent/dual_model_router.py` | 16 | DualModelRouter wrapping LLMRouter for large/small model selection |

### Existing Files to Modify

| File | Phases | Changes |
|---|---|---|
| `src/proxima/agent/dynamic_tools/robust_nl_processor.py` | 1, 2, 3 | Expand IntentType enum (+27 entries, bringing total from 27 to 54), add keyword mappings, add entity extraction patterns (packages, scripts, envs, backends, process IDs), enhance SessionContext with new fields (incl. backend checkpoint tracking), add resolve_reference() method, add sub_intents to Intent dataclass, restructure recognize_intent() into 5-layer pipeline |
| `src/proxima/agent/dynamic_tools/intent_classifier.py` | 1 | Rename IntentType to IntentCategory, add mapping to canonical IntentType |
| `src/proxima/agent/dynamic_tools/__init__.py` | 1, 14, 16 | Export IntentToolBridge, AgentLoop, SystemPromptBuilder, TodosTool, WebFetchTool, AgenticFetchTool |
| `src/proxima/tui/screens/agent_ai_assistant.py` | 3, 10, 14, 15, 16 | Replace 5-phase _generate_response with AgentLoop.process_message(), instantiate new components, add consent callback, support streaming display, add session management commands (/new, /sessions, /import, /summarize, /delete), add TodoPillWidget, integrate ToolPermissionManager |
| `src/proxima/tui/screens/execution.py` | 9 | Add handlers for AgentTerminalStarted/Output/Completed and AgentPlanStarted/StepCompleted messages |
| `src/proxima/tui/screens/results.py` | 9 | Add handler for AgentResultReady message, add_agent_result() method |
| `src/proxima/agent/dynamic_tools/tools/filesystem_tools.py` | 4 | No changes to tool implementations (already complete) |
| `src/proxima/agent/dynamic_tools/tools/git_tools.py` | 8 | No changes to tool implementations (already complete) |
| `src/proxima/agent/dynamic_tools/tools/terminal_tools.py` | 4 | No changes to tool implementations (already complete) |
| `src/proxima/agent/dynamic_tools/agent_loop.py` | 15, 16 | (Created in Phase 10 as a New File.) Accept AgentSessionManager, SubAgent factory, ToolPermissionManager, DualModelRouter; add auto-summarization checks, permission checks, sub-agent spawning, todos integration |
| `src/proxima/agent/dynamic_tools/llm_integration.py` | — | No changes needed (DualModelRouter and ModelRole are in separate `dual_model_router.py`) |
| `src/proxima/agent/backend_builder.py` | 6 | Called by IntentToolBridge for BACKEND_BUILD; uses BuildProfileLoader to load YAML profiles and execute build steps |
| `src/proxima/agent/backend_modifier.py` | 6 | Called by IntentToolBridge for BACKEND_MODIFY; generates CodeChange objects, applies modifications with diff preview |
| `src/proxima/agent/checkpoint_manager.py` | 6 | Used by BACKEND_MODIFY handler for checkpoint creation, undo, redo, and rollback |
| `src/proxima/agent/modification_preview.py` | 6 | Used by BACKEND_MODIFY handler for Rich-formatted diff display before applying changes |
| `src/proxima/agent/safety.py` | 6, 7, 11 | ConsentType.BACKEND_MODIFICATION for backend changes, ConsentType.ADMIN_ACCESS for admin escalation, RollbackManager for undo/redo |
| `src/proxima/agent/admin_privilege_handler.py` | 7 | Used by IntentToolBridge for ADMIN_ELEVATE; provides platform-specific elevation methods and audit logging |
| `configs/backend_build_profiles.yaml` | 6 | Read by BACKEND_LIST and BACKEND_BUILD handlers to load build profiles for 9 backends |
| `configs/default.yaml` | 16 | Add agent.models (large/small), agent.permissions (allowed_tools, yolo_mode), agent.session (storage_dir, auto_summarize, max_sessions) configuration sections |

### Key Libraries and Frameworks

| Library | Usage |
|---|---|
| `subprocess` (stdlib) | Terminal command execution, git operations, package installation |
| `threading` (stdlib) | Background terminal output reading, synchronous consent wait via `threading.Event`, Worker-based streaming display |
| `os`, `shutil`, `pathlib` (stdlib) | File system operations, path resolution |
| `re` (stdlib) | Pattern matching for intent recognition, entity extraction, error classification |
| `json` (stdlib) | Configuration parsing, result serialization, session persistence |
| `tomllib` (stdlib since Python 3.11) | pyproject.toml parsing for dependency detection |
| `uuid` (stdlib) | Terminal ID generation, session ID generation |
| `logging` (stdlib) | Error logging to file |
| `time`, `datetime` (stdlib) | Timing, timestamps, backoff calculations |
| `platform` (stdlib) | OS detection for cross-platform command normalization |
| `urllib.request` (stdlib) | HTTP requests for WebFetchTool (no external dependency) |
| `html.parser` (stdlib) | HTML tag stripping for web content extraction |
| `pydantic` | Tool parameter validation (already used in tool_interface.py) |
| `pyyaml` | Backend build profile loading (already used in backend_builder.py) |
| `rich` | Terminal output formatting, syntax highlighting, diff display |
| `textual` | TUI framework — screens, widgets, messages, containers, Workers for background tasks |

---

## Implementation Order and Dependencies

```
Phase 1 (Unified Intent Taxonomy)
    ↓
Phase 2 (Entity Extraction & Context)
    ↓
Phase 3 (NL Planning & Execution Pipeline)  ← Core functionality 1
    ↓
Phase 4 (File System & Script Execution)    ← Core functionalities 2, 3
    ↓
Phase 5 (Dependency Management)             ← Core functionality 4
    ↓
Phase 6 (Backend Build/Compile/Modify)      ← Core functionality 6 (consent, undo, redo, rollback)
    ↓
Phase 7 (Admin Access & Escalation)         ← Core functionality 7 (safe privilege escalation)
    ↓
Phase 8 (GitHub Operations)                 ← Core functionality 5
    ↓
Phase 9 (Multi-Terminal & Result Display)   ← Core functionality 8 — Execution Tab + Result Tab
    ↓
Phase 10 (Agentic Loop & Streaming)         ← Integrates everything
    ↓
Phase 11 (Error Handling & Safety)          ← Cross-cutting concern
    ↓
Phase 12 (Complex Task Walkthrough)         ← Core functionality 9 — Validation of Examples A, B, C
    ↓
Phase 13 (Testing)                          ← Comprehensive test suite
    ↓
Phase 14 (Core Integration)               ← Wiring all components together
    ↓
Phase 15 (Session Context Handling)         ← Core functionality 10 — persistence, import, auto-summarization
    ↓
Phase 16 (Crush-Inspired Capabilities)      ← Core functionality 11 — sub-agents, permissions, todos, dual-model
```

**Parallel opportunities:**
- Phases 4, 5, 6, 7, 8 can be implemented in parallel after Phase 3 is complete (they all use the IntentToolBridge independently)
- Phase 6 (Backend Build) depends on Phase 5 (Dependency Management) for pre-build dependency checks, so Phase 5 should ideally complete first
- Phase 7 (Admin Access) is independent and can be developed alongside any Phase after 3
- Phase 11 (Error Handling) can be incrementally developed alongside Phases 4-10
- Phase 9 (Multi-Terminal) can start after Phase 4 since it depends on terminal execution infrastructure
- Phase 15 (Session Context) can start after Phase 10 (requires AgentLoop) and Phase 2 (requires SessionContext)
- Phase 16 (Crush-Inspired) can start after Phase 15's AgentSessionManager is ready (sub-agents need session creation); the TodosTool and PermissionManager steps can begin in parallel with Phase 15

**Sequential requirements:**
- Phase 1 must complete before Phase 2 (enum must be stable)
- Phase 2 must complete before Phase 3 (entity extraction needed for argument building)
- Phase 3 must complete before Phases 4-8 (IntentToolBridge is the execution backbone)
- Phase 6 should follow Phase 5 (backend builds use dependency management)
- Phase 10 must come after Phases 3-9 (integrates all components)
- Phase 14 must complete before Phase 15 (core wiring must be stable before session persistence layer)
- Phase 15 must substantially complete before Phase 16 (AgentSessionManager needed for sub-agent TaskSessions, dual-model routing)
- Phase 16 is the final phase — strict compliance validation (functionality 12) should be verified across all 16 phases at the end

---

*End of Agent Functionality Implementation P2 Guide*
