"""Data handling module - Step 5.1 & 5.3: Multi-Backend Comparison & Export Engine."""

from .compare import (
    # Enums
    ComparisonStatus,
    ExecutionStrategy,
    # Data classes
    BackendResult,
    ComparisonMetrics,
    ComparisonReport,
    # Classes
    ExecutionPlanner,
    ResultAnalyzer,
    MultiBackendComparator,
    # Convenience function
    compare_backends,
)

from .export import (
    # Enums
    ExportFormat,
    # Data classes
    ExportOptions,
    ExportResult,
    ReportData,
    # Exporter classes
    BaseExporter,
    JSONExporter,
    CSVExporter,
    XLSXExporter,
    HTMLExporter,
    # Main engine
    ExportEngine,
    # Convenience functions
    export_to_json,
    export_to_csv,
    export_to_xlsx,
    export_to_html,
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
    # Main engine
    "ExportEngine",
    # Convenience functions
    "export_to_json",
    "export_to_csv",
    "export_to_xlsx",
    "export_to_html",
]
