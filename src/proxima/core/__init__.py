"""Core domain logic module - includes Step 5.2: Agent.md Interpreter and Real-Time Event System."""

from .agent_interpreter import (
    AgentConfiguration,
    AgentFile,
    # Classes
    AgentFileParser,
    AgentInterpreter,
    AgentMetadata,
    DefaultTaskExecutor,
    ExecutionReport,
    TaskDefinition,
    TaskExecutor,
    TaskResult,
    TaskStatus,
    # Enums
    TaskType,
    # Data classes
    ValidationIssue,
    ValidationSeverity,
    # Convenience function
    run_agent_file,
)

# Phase 1: Real-Time Event Bus System
from .event_bus import (
    Event,
    EventBus,
    EventType,
    Subscription,
    get_event_bus,
    reset_event_bus,
    emit_process_started,
    emit_output_line,
    emit_process_completed,
    emit_progress_update,
    emit_result_available,
)

# Phase 1: Process Output Streaming
from .process_streaming import (
    ProcessState,
    ProcessInfo,
    ProcessOutputStream,
    ProcessExecutor,
    StreamingProcessRunner,
    run_command,
)
from .executor import Executor
from .pipeline import (
    AnalysisHandler,
    CollectionHandler,
    ConsentHandler,
    # Main pipeline class
    DataFlowPipeline,
    ExecutionHandler,
    ExportHandler,
    # Handler implementations
    ParseHandler,
    PipelineContext,
    # Handler base class
    PipelineHandler,
    # Enums
    PipelineStage,
    PlanHandler,
    ResourceCheckHandler,
    # Data classes
    StageResult,
    compare_backends,
    # Convenience functions
    run_simulation,
)
from .planner import Planner
from .runner import quantum_runner
from .session import (
    # Data classes
    Checkpoint,
    ExecutionRecord,
    # Classes
    Session,
    SessionManager,
    # Enums
    SessionStatus,
)
from .state import (
    # Enums
    ExecutionState,
    # Classes
    ExecutionStateMachine,
)

__all__ = [
    # ============ event_bus (Phase 1) ============
    "Event",
    "EventBus",
    "EventType",
    "Subscription",
    "get_event_bus",
    "reset_event_bus",
    "emit_process_started",
    "emit_output_line",
    "emit_process_completed",
    "emit_progress_update",
    "emit_result_available",
    # ============ process_streaming (Phase 1) ============
    "ProcessState",
    "ProcessInfo",
    "ProcessOutputStream",
    "ProcessExecutor",
    "StreamingProcessRunner",
    "run_command",
    # ============ agent_interpreter ============
    # Enums
    "TaskType",
    "TaskStatus",
    "ValidationSeverity",
    # Data classes
    "ValidationIssue",
    "AgentMetadata",
    "AgentConfiguration",
    "TaskDefinition",
    "TaskResult",
    "AgentFile",
    "ExecutionReport",
    # Classes
    "AgentFileParser",
    "TaskExecutor",
    "DefaultTaskExecutor",
    "AgentInterpreter",
    # Convenience function
    "run_agent_file",
    # ============ executor ============
    "Executor",
    # ============ pipeline ============
    # Enums
    "PipelineStage",
    # Data classes
    "StageResult",
    "PipelineContext",
    # Handler base class
    "PipelineHandler",
    # Handler implementations
    "ParseHandler",
    "PlanHandler",
    "ResourceCheckHandler",
    "ConsentHandler",
    "ExecutionHandler",
    "CollectionHandler",
    "AnalysisHandler",
    "ExportHandler",
    # Main pipeline class
    "DataFlowPipeline",
    # Convenience functions
    "run_simulation",
    "compare_backends",
    # ============ planner ============
    "Planner",
    # ============ runner ============
    "quantum_runner",
    # ============ session ============
    # Enums
    "SessionStatus",
    # Data classes
    "Checkpoint",
    "ExecutionRecord",
    # Classes
    "Session",
    "SessionManager",
    # ============ state ============
    # Enums
    "ExecutionState",
    # Classes
    "ExecutionStateMachine",
]
