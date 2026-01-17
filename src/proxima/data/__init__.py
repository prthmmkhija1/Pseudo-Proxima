"""Data handling module - Step 5.1, 5.3 & 5.6: Multi-Backend Comparison, Export Engine & Data Pipeline."""

from .compare import (
    # Data classes
    BackendResult,
    ComparisonMetrics,
    ComparisonReport,
    # Enums
    ComparisonStatus,
    # Classes
    ExecutionPlanner,
    ExecutionStrategy,
    MultiBackendComparator,
    ResultAnalyzer,
    # Convenience function
    compare_backends,
)
from .metrics import (
    # Enums
    BenchmarkStatus,
    # Data classes
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkComparison,
)
from .benchmark_registry import BenchmarkRegistry
from .export import (
    # Exporter classes
    BaseExporter,
    CSVExporter,
    # Main engine
    ExportEngine,
    # Enums
    ExportFormat,
    # Data classes
    ExportOptions,
    ExportResult,
    HTMLExporter,
    JSONExporter,
    MarkdownExporter,
    ReportData,
    XLSXExporter,
    YAMLExporter,
    # Convenience functions
    export_to_csv,
    export_to_html,
    export_to_json,
    export_to_markdown,
    export_to_string,
    export_to_xlsx,
    export_to_yaml,
)
from .pipeline import (
    # Enums
    StageStatus,
    PipelineStatus,
    RetryStrategy,
    CancellationReason,
    # Configuration classes
    RetryConfig,
    TimeoutConfig,
    PipelineConfig,
    # Result classes
    StageResult,
    PipelineResult,
    # Context
    PipelineContext,
    # Exceptions
    PipelineException,
    StageTimeoutException,
    PipelineTimeoutException,
    PipelineCancelledException,
    StageExecutionException,
    DependencyFailedException,
    # Core classes
    Stage,
    Pipeline,
    PipelineBuilder,
    # Decorators
    stage,
    with_retry,
    # Convenience functions
    run_pipeline,
    create_stage,
)
from .session import (
    # Enums
    SessionStatus,
    # Data classes
    ExecutionContext,
    SessionState,
    UserPreferences,
    WorkflowStep,
    # Main class
    SessionPersistence,
    # Global functions
    get_session_persistence,
    reset_session_persistence,
)

__all__ = [
    # ===== Benchmarking Metrics =====
    # Enums
    "BenchmarkStatus",
    # Data classes
    "BenchmarkMetrics",
    "BenchmarkResult",
    "BenchmarkComparison",
    "BenchmarkRegistry",
    # ===== Step 5.1: Multi-Backend Comparison =====
    # Enums
    "ComparisonStatus",
    "ExecutionStrategy",
    # Data classes
    "BackendResult",
    "ComparisonMetrics",
    "ComparisonReport",
    # Classes
    "ExecutionPlanner",
    "ResultAnalyzer",
    "MultiBackendComparator",
    # Convenience function
    "compare_backends",
    # ===== Step 5.3: Export Engine =====
    # Enums
    "ExportFormat",
    # Data classes
    "ExportOptions",
    "ExportResult",
    "ReportData",
    # Exporter classes
    "BaseExporter",
    "JSONExporter",
    "CSVExporter",
    "XLSXExporter",
    "HTMLExporter",
    "MarkdownExporter",
    "YAMLExporter",
    # Main engine
    "ExportEngine",
    # Convenience functions
    "export_to_json",
    "export_to_csv",
    "export_to_xlsx",
    "export_to_html",
    "export_to_markdown",
    "export_to_yaml",
    "export_to_string",
    # ===== Step 5.6: Data Pipeline =====
    # Enums
    "StageStatus",
    "PipelineStatus",
    "RetryStrategy",
    "CancellationReason",
    # Configuration classes
    "RetryConfig",
    "TimeoutConfig",
    "PipelineConfig",
    # Result classes
    "StageResult",
    "PipelineResult",
    # Context
    "PipelineContext",
    # Exceptions
    "PipelineException",
    "StageTimeoutException",
    "PipelineTimeoutException",
    "PipelineCancelledException",
    "StageExecutionException",
    "DependencyFailedException",
    # Core classes
    "Stage",
    "Pipeline",
    "PipelineBuilder",
    # Decorators
    "stage",
    "with_retry",
    # Convenience functions
    "run_pipeline",
    "create_stage",
    # ===== Session Persistence =====
    # Enums
    "SessionStatus",
    # Data classes
    "ExecutionContext",
    "SessionState",
    "UserPreferences",
    "WorkflowStep",
    # Main class
    "SessionPersistence",
    # Global functions
    "get_session_persistence",
    "reset_session_persistence",
]
