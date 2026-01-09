"""Core domain logic module - includes Step 5.2: Agent.md Interpreter."""

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

__all__ = [
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
]
