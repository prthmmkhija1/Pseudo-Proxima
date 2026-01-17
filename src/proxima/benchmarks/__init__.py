"""Benchmarking utilities for Proxima.

Includes benchmark runner, configuration, circuit analysis, statistics,
profiling, visualization, suites, scheduling, and Phase 10 optimizations.
"""

from .circuit_analyzer import CircuitAnalyzer
from .comparator import BackendComparator
from .profiler import BackendProfile, BackendProfiler
from .runner import BenchmarkRunner, BenchmarkRunnerConfig
from .scheduler import BenchmarkScheduler
from .statistics import StatisticsCalculator, TrendResult
from .suite import BenchmarkSuite, SuiteResults
from .visualization import VisualizationDataBuilder

# Phase 10.1: Performance Optimizations
from .optimizations import (
    AdaptiveSampler,
    AdaptiveSamplerConfig,
    ConnectionPool,
    IncrementalStats,
    LazyResultLoader,
    ParallelStatsCalculator,
    PreparedStatements,
    ResultSummary,
    StatisticsCache,
    StreamingResultIterator,
)

# Phase 10.2: Error Handling and Robustness
from .error_handling import (
    BackendValidator,
    BenchmarkError,
    BenchmarkErrorCodes,
    CircuitValidator,
    CrashRecoveryManager,
    DatabaseRecovery,
    ErrorHandler,
    ErrorSeverity,
    RecoveryState,
    ResultValidator,
    RetryConfig,
    RetryError,
    SchemaValidator,
    ValidationResult,
    is_retryable,
    safe_database_operation,
    with_retry,
)

# Phase 10.3: User Experience Enhancements
from .ux_enhancements import (
    BenchmarkProgress,
    BenchmarkProgressConfig,
    InteractiveMode,
    ProgressStyle,
    RealTimeMetricsDisplay,
    ResultFormatter,
    SmartDefaults,
    Sparkline,
    WatchMode,
)

__all__ = [
    # Core benchmarking
    "BenchmarkRunner",
    "BenchmarkRunnerConfig",
    "CircuitAnalyzer",
    "BackendComparator",
    "StatisticsCalculator",
    "TrendResult",
    "BackendProfiler",
    "BackendProfile",
    "VisualizationDataBuilder",
    "BenchmarkSuite",
    "SuiteResults",
    "BenchmarkScheduler",
    # Phase 10.1: Performance Optimizations
    "AdaptiveSampler",
    "AdaptiveSamplerConfig",
    "ConnectionPool",
    "IncrementalStats",
    "LazyResultLoader",
    "ParallelStatsCalculator",
    "PreparedStatements",
    "ResultSummary",
    "StatisticsCache",
    "StreamingResultIterator",
    # Phase 10.2: Error Handling
    "BackendValidator",
    "BenchmarkError",
    "BenchmarkErrorCodes",
    "CircuitValidator",
    "CrashRecoveryManager",
    "DatabaseRecovery",
    "ErrorHandler",
    "ErrorSeverity",
    "RecoveryState",
    "ResultValidator",
    "RetryConfig",
    "RetryError",
    "SchemaValidator",
    "ValidationResult",
    "is_retryable",
    "safe_database_operation",
    "with_retry",
    # Phase 10.3: UX Enhancements
    "BenchmarkProgress",
    "BenchmarkProgressConfig",
    "InteractiveMode",
    "ProgressStyle",
    "RealTimeMetricsDisplay",
    "ResultFormatter",
    "SmartDefaults",
    "Sparkline",
    "WatchMode",
]
