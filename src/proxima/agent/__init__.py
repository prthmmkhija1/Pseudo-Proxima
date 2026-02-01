"""Proxima Agent Module.

This module provides powerful AI agent capabilities including:
- Terminal execution (build, compile, run scripts)
- File system access and manipulation
- Git operations (clone, pull, push)
- Backend code modification with safety features
- Multi-terminal monitoring with real-time output
- Natural language task planning and execution

Phase 3 Additions:
- MultiTerminalMonitor: Track multiple concurrent processes
- SessionManager: Manage agent sessions with persistence
- CommandNormalizer: Cross-platform command translation
- TerminalStateMachine: Process state tracking with metrics

Phase 4 Additions:
- BackendBuilder: Build backends with progress tracking
- GPUDetector: Detect NVIDIA/AMD/Apple GPUs
- BuildArtifactManager: Manage build artifacts and versions
- BuildProgressTracker: Track build progress with callbacks

Phase 5 Additions:
- FileSystemOperations: Safe file operations with path validation
- AdminPrivilegeHandler: Handle elevated permissions (UAC/sudo)
- FileSystemWatcher: Monitor file changes with watchdog
- FileContentSearch: Fast search with ripgrep and Python AST analysis
- TempFileManager: Secure temporary file management

Phase 6 Additions:
- TaskPlanner: Natural language to execution plan conversion
- PlanExecutor: Dependency-ordered plan execution engine
- NLCommandParser: Natural language command parsing
- ScriptExecutor: Multi-language script execution framework
- CommandSuggestions: Context-aware command suggestions

Phase 7 Additions:
- GitStatusParser: Enhanced git status parsing with categorization
- GitDiffParser: Unified diff parsing with hunk extraction
- GitCommitWorkflow: Commit workflow with message validation
- GitBranchManager: Branch operations with safety checks
- GitConflictResolver: Merge conflict detection and resolution

Phase 8 Additions:
- FileBackupSystem: Manifest-based file backups with checksums
- ModificationPreviewGenerator: Diff preview with impact analysis
- ConsentManager: Enhanced consent with scopes and persistence
- CheckpointManager: Undo/redo stack with checkpoint validation
- CodeIntelligence: AST-based code analysis and modification

Phase 9 Additions:
- AgentTelemetry: Comprehensive telemetry data model
- MetricCollector: Instrumentation and metric collection
- StatsPanel: Real-time statistics display widget
- HistoricalMetricsStore: SQLite-based time-series storage
- AlertManager: Anomaly detection and alerting system

Inspired by Charm's Crush AI agent architecture.
"""

from .terminal_executor import TerminalExecutor, TerminalSession, TerminalOutput
from .session_manager import AgentSessionManager, TerminalPool
from .tools import AgentTools, ToolDefinition, ToolResult
from .safety import (
    SafetyManager, 
    ConsentRequest, 
    ConsentResponse,
    ConsentType,
    ConsentDecision,
    RollbackManager,
    OperationCheckpoint,
)
from .git_operations import GitOperations, GitResult
from .backend_modifier import BackendModifier, CodeChange, ModificationResult
from .multi_terminal_monitor import MultiTerminalMonitor, TerminalEvent

# Phase 3: Enhanced multi-terminal and session management
from .multi_terminal import (
    MultiTerminalMonitor as Phase3Monitor,
    TerminalInfo,
    TerminalState,
    TerminalEvent as Phase3Event,
    TerminalEventType,
    CircularOutputBuffer,
    OutputLine,
    SessionManager,
    AgentSession,
    SessionState,
    CommandHistoryEntry,
    CommandNormalizer,
    CommandQueue,
    CommandPriority,
    QueuedCommand,
    get_multi_terminal_monitor,
    get_session_manager,
    get_command_normalizer,
)
from .terminal_state_machine import (
    TerminalStateMachine,
    TerminalProcessState,
    TerminalStateEvent,
    ProcessMetrics,
    StateContext,
    EventDebouncer,
    VALID_TRANSITIONS,
    get_terminal_state_machine,
)

# Phase 4: Backend Building & Compilation System
from .gpu_detector import (
    GPUDetector,
    GPUInfo,
    GPUEnvironment,
    GPUVendor,
    GPUCapability,
    get_gpu_detector,
    detect_gpus,
    has_cuda,
    has_gpu,
)
from .build_artifact_manager import (
    BuildArtifactManager,
    BuildManifest,
    ArtifactInfo,
    ArtifactType,
    BuildVersion,
    generate_build_id,
)
from .build_progress_tracker import (
    BuildProgressTracker,
    BuildProgress,
    BuildStep,
    BuildPhase,
    BuildStepStatus,
    ProgressPattern,
    ProgressCallback,
    MultiBackendProgressTracker,
)
from .backend_builder import (
    BackendBuilder,
    BuildProfileLoader,
    BuildResult,
    BuildStepResult,
    BuildStatus,
    ValidationResult,
    DependencyCheck,
    get_backend_builder,
    build_backend,
)

# Phase 5: File System Operations & Administrative Access
from .file_system_operations import (
    FileSystemOperations,
    PathValidator,
    FileInfo,
    FileOperationResult,
    SearchMatch,
    get_file_system_operations,
)
from .admin_privilege_handler import (
    AdminPrivilegeHandler,
    PrivilegeInfo,
    PrivilegedOperation,
    ElevationResult,
    PrivilegeLevel,
    OperationCategory,
    get_admin_privilege_handler,
    is_admin,
    requires_elevation,
)
from .file_system_watcher import (
    FileSystemWatcher,
    FileEvent,
    WatchConfig,
    RebuildTrigger,
    EventDebouncer as FileEventDebouncer,
    FileEventType,
    get_file_system_watcher,
)
from .file_content_search import (
    FileContentSearch,
    PythonCodeAnalyzer,
    SearchResult,
    CodeElement,
    CodeMetrics,
    SearchEngine,
    SearchSummary,
    get_file_content_search,
)
from .temp_file_manager import (
    TempFileManager,
    TempFileInfo,
    TempFileType,
    CleanupStats,
    get_temp_file_manager,
    create_temp_file,
    create_temp_directory,
)

# Phase 6: Natural Language Planning & Execution
from .task_planner import (
    TaskPlanner,
    ExecutionPlan,
    PlanStep,
    PlanStatus,
    StepStatus,
    TaskCategory,
    IntentRecognitionResult,
    get_task_planner,
)
from .plan_executor import (
    PlanExecutor,
    AdaptivePlanExecutor,
    ToolExecutor,
    ExecutionResult,
    StepResult,
    ExecutionMode,
    ExecutorCallbacks,
    get_plan_executor,
    execute_plan,
)
from .nl_command_parser import (
    NLCommandParser,
    ParsedCommand,
    CommandType,
    CommandPattern,
    ConversationContext,
    ExtractedEntity,
    ConfidenceLevel,
    get_nl_command_parser,
    parse_command,
)
from .script_executor import (
    ScriptExecutor,
    ScriptResult,
    ScriptInfo,
    ScriptLanguage,
    ScriptSource,
    InterpreterInfo,
    InterpreterRegistry,
    get_script_executor,
    execute_script,
    execute_code,
)
from .command_suggestions import (
    CommandSuggestions,
    Suggestion,
    ExecutionContext,
    PatternLearner,
    CommandSequence,
    get_command_suggestions,
    suggest_next,
)

# Phase 7: Git Operations Integration
from .git_status_parser import (
    GitStatusParser,
    GitFileStatus as ParsedFileStatus,
    FileStatusCode,
    BranchStatus,
    RepositoryStatus,
    ConflictMarker,
    ConflictFile as StatusConflictFile,
    MergeConflictType,
    get_git_status_parser,
    parse_git_status,
)
from .git_diff_parser import (
    GitDiffParser,
    DiffLine,
    DiffHunk,
    FileDiff,
    DiffResult,
    LineType,
    WordDiff,
    get_git_diff_parser,
    parse_git_diff,
)
from .git_commit_workflow import (
    GitCommitWorkflow,
    CommitMessageValidator,
    CommitMessageGenerator,
    CommitType,
    CommitMessageParts,
    ValidationResult as CommitValidationResult,
    StagedFile,
    CommitPreview,
    PushPreview,
    get_commit_workflow,
)
from .git_branch_manager import (
    GitBranchManager,
    BranchInfo,
    BranchType,
    BranchListResult,
    BranchComparisonResult,
    MergePreview,
    get_branch_manager,
)
from .git_conflict_resolver import (
    GitConflictResolver,
    ConflictParser,
    ConflictType,
    ResolutionStrategy,
    ConflictSection,
    ConflictFile,
    ConflictResolutionResult,
    MergeStatus,
    get_conflict_resolver,
    parse_conflict_markers,
)

# Phase 8: Backend Code Modification with Safety
from .file_backup_system import (
    FileBackupSystem,
    FileSnapshot,
    BackupManifest,
    RestoreResult,
    BackupType,
    get_file_backup_system,
    create_backup,
    restore_backup,
)
from .modification_preview import (
    ModificationPreviewGenerator,
    ModificationPreview,
    ImpactAnalysis,
    DiffHunk as ModDiffHunk,
    DiffLine as ModDiffLine,
    ModificationScope,
    DiffLineType,
    get_preview_generator,
    generate_preview,
)
from .consent_manager import (
    ConsentManager,
    ConsentRequest as EnhancedConsentRequest,
    ConsentResponse as EnhancedConsentResponse,
    ConsentRule,
    ConsentScope,
    RiskLevel as ConsentRiskLevel,
    ConsentAuditEntry,
    get_consent_manager,
    request_consent,
)
from .checkpoint_manager import (
    CheckpointManager,
    Checkpoint,
    FileState,
    RollbackResult,
    UndoRedoState,
    get_checkpoint_manager,
    create_checkpoint,
    undo,
    redo,
)
from .code_intelligence import (
    CodeIntelligence,
    CodeAnalysisResult,
    Symbol,
    SymbolType,
    CodeLocation,
    ImportInfo,
    ModificationTemplate,
    get_code_intelligence,
)

# Phase 9: Agent Statistics & Telemetry System
from .telemetry import (
    AgentTelemetry,
    TelemetrySnapshot,
    LLMMetrics,
    PerformanceMetrics,
    TerminalMetrics,
    GitMetrics,
    FileMetrics,
    BuildMetrics,
    MetricCategory,
    MetricType,
    CircularBuffer as TelemetryBuffer,
    format_number,
    format_bytes,
    format_duration,
    format_percentage,
    format_currency,
    get_telemetry,
    reset_telemetry,
)
from .metric_collector import (
    MetricCollector,
    MetricEmitter,
    OperationMetric,
    AggregatedMetric,
    OperationType as MetricOperationType,
    get_metric_collector,
    set_metric_collector,
    track_llm_request,
    track_tool,
    track_command,
    track_file_operation,
    track_git_operation,
    track_build,
)
from .stats_panel import (
    StatsPanel,
    StatsCard,
    StatItem,
    CompactStatsBar,
    create_stats_panel,
    create_compact_stats_bar,
)
from .historical_metrics import (
    HistoricalMetricsStore,
    TimeSeriesPoint,
    TimeSeriesStats,
    AggregatedSeries,
    get_historical_store,
    record_metric,
)
from .alert_manager import (
    AlertManager,
    Alert,
    AlertThreshold,
    AlertSeverity,
    AlertType,
    AlertState,
    AlertStats,
    AnomalyDetector,
    get_alert_manager,
    check_metric,
    create_alert,
)

from .agent_controller import AgentController

__all__ = [
    # Terminal execution
    "TerminalExecutor",
    "TerminalSession",
    "TerminalOutput",
    # Session management
    "AgentSessionManager",
    "TerminalPool",
    # Tools
    "AgentTools",
    "ToolDefinition",
    "ToolResult",
    # Safety
    "SafetyManager",
    "ConsentRequest",
    "ConsentResponse",
    "ConsentType",
    "ConsentDecision",
    "RollbackManager",
    "OperationCheckpoint",
    # Git
    "GitOperations",
    "GitResult",
    # Backend modification
    "BackendModifier",
    "CodeChange",
    "ModificationResult",
    # Multi-terminal (legacy)
    "MultiTerminalMonitor",
    "TerminalEvent",
    # Controller
    "AgentController",
    
    # Phase 3: Enhanced multi-terminal
    "Phase3Monitor",
    "TerminalInfo",
    "TerminalState",
    "Phase3Event",
    "TerminalEventType",
    "CircularOutputBuffer",
    "OutputLine",
    
    # Phase 3: Session management
    "SessionManager",
    "AgentSession",
    "SessionState",
    "CommandHistoryEntry",
    
    # Phase 3: Command normalization
    "CommandNormalizer",
    "CommandQueue",
    "CommandPriority",
    "QueuedCommand",
    
    # Phase 3: State machine
    "TerminalStateMachine",
    "TerminalProcessState",
    "TerminalStateEvent",
    "ProcessMetrics",
    "StateContext",
    "EventDebouncer",
    "VALID_TRANSITIONS",
    
    # Phase 3: Global accessors
    "get_multi_terminal_monitor",
    "get_session_manager",
    "get_command_normalizer",
    "get_terminal_state_machine",
    
    # Phase 4: GPU Detection
    "GPUDetector",
    "GPUInfo",
    "GPUEnvironment",
    "GPUVendor",
    "GPUCapability",
    "get_gpu_detector",
    "detect_gpus",
    "has_cuda",
    "has_gpu",
    
    # Phase 4: Build Artifact Management
    "BuildArtifactManager",
    "BuildManifest",
    "ArtifactInfo",
    "ArtifactType",
    "BuildVersion",
    "generate_build_id",
    
    # Phase 4: Build Progress Tracking
    "BuildProgressTracker",
    "BuildProgress",
    "BuildStep",
    "BuildPhase",
    "BuildStepStatus",
    "ProgressPattern",
    "ProgressCallback",
    "MultiBackendProgressTracker",
    
    # Phase 4: Backend Builder
    "BackendBuilder",
    "BuildProfileLoader",
    "BuildResult",
    "BuildStepResult",
    "BuildStatus",
    "ValidationResult",
    "DependencyCheck",
    "get_backend_builder",
    "build_backend",
    
    # Phase 5: File System Operations
    "FileSystemOperations",
    "PathValidator",
    "FileInfo",
    "FileOperationResult",
    "SearchMatch",
    "get_file_system_operations",
    
    # Phase 5: Administrative Privileges
    "AdminPrivilegeHandler",
    "PrivilegeInfo",
    "PrivilegedOperation",
    "ElevationResult",
    "PrivilegeLevel",
    "OperationCategory",
    "get_admin_privilege_handler",
    "is_admin",
    "requires_elevation",
    
    # Phase 5: File System Watcher
    "FileSystemWatcher",
    "FileEvent",
    "WatchConfig",
    "RebuildTrigger",
    "FileEventDebouncer",
    "FileEventType",
    "get_file_system_watcher",
    
    # Phase 5: File Content Search
    "FileContentSearch",
    "PythonCodeAnalyzer",
    "SearchResult",
    "CodeElement",
    "CodeMetrics",
    "SearchEngine",
    "SearchSummary",
    "get_file_content_search",
    
    # Phase 5: Temp File Manager
    "TempFileManager",
    "TempFileInfo",
    "TempFileType",
    "CleanupStats",
    "get_temp_file_manager",
    "create_temp_file",
    "create_temp_directory",
    
    # Phase 6: Task Planning
    "TaskPlanner",
    "ExecutionPlan",
    "PlanStep",
    "PlanStatus",
    "StepStatus",
    "TaskCategory",
    "IntentRecognitionResult",
    "get_task_planner",
    
    # Phase 6: Plan Execution
    "PlanExecutor",
    "AdaptivePlanExecutor",
    "ToolExecutor",
    "ExecutionResult",
    "StepResult",
    "ExecutionMode",
    "ExecutorCallbacks",
    "get_plan_executor",
    "execute_plan",
    
    # Phase 6: Natural Language Command Parsing
    "NLCommandParser",
    "ParsedCommand",
    "CommandType",
    "CommandPattern",
    "ConversationContext",
    "ExtractedEntity",
    "ConfidenceLevel",
    "get_nl_command_parser",
    "parse_command",
    
    # Phase 6: Script Execution
    "ScriptExecutor",
    "ScriptResult",
    "ScriptInfo",
    "ScriptLanguage",
    "ScriptSource",
    "InterpreterInfo",
    "InterpreterRegistry",
    "get_script_executor",
    "execute_script",
    "execute_code",
    
    # Phase 6: Command Suggestions
    "CommandSuggestions",
    "Suggestion",
    "ExecutionContext",
    "PatternLearner",
    "CommandSequence",
    "get_command_suggestions",
    "suggest_next",
    
    # Phase 7: Git Status Parser
    "GitStatusParser",
    "ParsedFileStatus",
    "FileStatusCode",
    "BranchStatus",
    "RepositoryStatus",
    "ConflictMarker",
    "StatusConflictFile",
    "MergeConflictType",
    "get_git_status_parser",
    "parse_git_status",
    
    # Phase 7: Git Diff Parser
    "GitDiffParser",
    "DiffLine",
    "DiffHunk",
    "FileDiff",
    "DiffResult",
    "LineType",
    "WordDiff",
    "get_git_diff_parser",
    "parse_git_diff",
    
    # Phase 7: Git Commit Workflow
    "GitCommitWorkflow",
    "CommitMessageValidator",
    "CommitMessageGenerator",
    "CommitType",
    "CommitMessageParts",
    "CommitValidationResult",
    "StagedFile",
    "CommitPreview",
    "PushPreview",
    "get_commit_workflow",
    
    # Phase 7: Git Branch Manager
    "GitBranchManager",
    "BranchInfo",
    "BranchType",
    "BranchListResult",
    "BranchComparisonResult",
    "MergePreview",
    "get_branch_manager",
    
    # Phase 7: Git Conflict Resolver
    "GitConflictResolver",
    "ConflictParser",
    "ConflictType",
    "ResolutionStrategy",
    "ConflictSection",
    "ConflictFile",
    "ConflictResolutionResult",
    "MergeStatus",
    "get_conflict_resolver",
    "parse_conflict_markers",
    
    # Phase 8: File Backup System
    "FileBackupSystem",
    "FileSnapshot",
    "BackupManifest",
    "RestoreResult",
    "BackupType",
    "get_file_backup_system",
    "create_backup",
    "restore_backup",
    
    # Phase 8: Modification Preview
    "ModificationPreviewGenerator",
    "ModificationPreview",
    "ImpactAnalysis",
    "ModDiffHunk",
    "ModDiffLine",
    "ModificationScope",
    "DiffLineType",
    "get_preview_generator",
    "generate_preview",
    
    # Phase 8: Consent Manager
    "ConsentManager",
    "EnhancedConsentRequest",
    "EnhancedConsentResponse",
    "ConsentRule",
    "ConsentScope",
    "ConsentRiskLevel",
    "ConsentAuditEntry",
    "get_consent_manager",
    "request_consent",
    
    # Phase 8: Checkpoint Manager
    "CheckpointManager",
    "Checkpoint",
    "FileState",
    "RollbackResult",
    "UndoRedoState",
    "get_checkpoint_manager",
    "create_checkpoint",
    "undo",
    "redo",
    
    # Phase 8: Code Intelligence
    "CodeIntelligence",
    "CodeAnalysisResult",
    "Symbol",
    "SymbolType",
    "CodeLocation",
    "ImportInfo",
    "ModificationTemplate",
    "get_code_intelligence",
    
    # Phase 9: Telemetry Data Model
    "AgentTelemetry",
    "TelemetrySnapshot",
    "LLMMetrics",
    "PerformanceMetrics",
    "TerminalMetrics",
    "GitMetrics",
    "FileMetrics",
    "BuildMetrics",
    "MetricCategory",
    "MetricType",
    "TelemetryBuffer",
    "format_number",
    "format_bytes",
    "format_duration",
    "format_percentage",
    "format_currency",
    "get_telemetry",
    "reset_telemetry",
    
    # Phase 9: Metric Collector
    "MetricCollector",
    "MetricEmitter",
    "OperationMetric",
    "AggregatedMetric",
    "MetricOperationType",
    "get_metric_collector",
    "set_metric_collector",
    "track_llm_request",
    "track_tool",
    "track_command",
    "track_file_operation",
    "track_git_operation",
    "track_build",
    
    # Phase 9: Stats Panel Widget
    "StatsPanel",
    "StatsCard",
    "StatItem",
    "CompactStatsBar",
    "create_stats_panel",
    "create_compact_stats_bar",
    
    # Phase 9: Historical Metrics
    "HistoricalMetricsStore",
    "TimeSeriesPoint",
    "TimeSeriesStats",
    "AggregatedSeries",
    "get_historical_store",
    "record_metric",
    
    # Phase 9: Alert Manager
    "AlertManager",
    "Alert",
    "AlertThreshold",
    "AlertSeverity",
    "AlertType",
    "AlertState",
    "AlertStats",
    "AnomalyDetector",
    "get_alert_manager",
    "check_metric",
    "create_alert",
]
