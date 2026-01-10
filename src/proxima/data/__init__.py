"""Data handling module - Step 5.1 & 5.3: Multi-Backend Comparison & Export Engine."""

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

__all__ = [
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
]
