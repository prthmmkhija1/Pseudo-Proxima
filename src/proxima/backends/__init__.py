"""Backend adapters module - comprehensive quantum backend integration."""

from proxima.backends.base import (
    BaseBackendAdapter,
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
    ValidationResult,
)
from proxima.backends.conversion import (
    CircuitFormat,
    CircuitInfo,
    ConversionResult,
    convert_circuit,
    detect_circuit_format,
    extract_circuit_info,
    validate_for_backend,
)
from proxima.backends.exceptions import (
    BackendError,
    BackendErrorCode,
    BackendNotInstalledError,
    BackendTimeoutError,
    CircuitValidationError,
    ExecutionError,
    MemoryExceededError,
    QubitLimitExceededError,
    UnsupportedOperationError,
    wrap_backend_exception,
)
from proxima.backends.execution import (
    BatchConfig,
    BatchResult,
    RetryConfig,
    RetryResult,
    TimeoutConfig,
    execute_async,
    execute_batch,
    execute_batch_async,
    execute_with_retry,
    execute_with_timeout,
    with_retry,
)
from proxima.backends.health import (
    HealthCheckResult,
    HealthStatus,
    check_backend_health,
    check_backends_health_async,
    generate_health_report,
    run_simple_execution_test,
)
from proxima.backends.normalization import (
    compare_probabilities,
    normalize_counts,
    normalize_density_matrix,
    normalize_result,
    normalize_statevector,
    probabilities_from_density_matrix,
    probabilities_from_statevector,
)
from proxima.backends.registry import BackendRegistry, BackendStatus, backend_registry

__all__ = [
    # Base types
    "BaseBackendAdapter",
    "Capabilities",
    "ExecutionResult",
    "ResourceEstimate",
    "ResultType",
    "SimulatorType",
    "ValidationResult",
    # Exceptions
    "BackendError",
    "BackendErrorCode",
    "BackendNotInstalledError",
    "BackendTimeoutError",
    "CircuitValidationError",
    "ExecutionError",
    "MemoryExceededError",
    "QubitLimitExceededError",
    "UnsupportedOperationError",
    "wrap_backend_exception",
    # Execution utilities
    "BatchConfig",
    "BatchResult",
    "RetryConfig",
    "RetryResult",
    "TimeoutConfig",
    "execute_async",
    "execute_batch",
    "execute_batch_async",
    "execute_with_retry",
    "execute_with_timeout",
    "with_retry",
    # Circuit conversion
    "CircuitFormat",
    "CircuitInfo",
    "ConversionResult",
    "convert_circuit",
    "detect_circuit_format",
    "extract_circuit_info",
    "validate_for_backend",
    # Health checks
    "HealthCheckResult",
    "HealthStatus",
    "check_backend_health",
    "check_backends_health_async",
    "generate_health_report",
    "run_simple_execution_test",
    # Registry
    "BackendRegistry",
    "BackendStatus",
    "backend_registry",
    # Normalization
    "compare_probabilities",
    "normalize_counts",
    "normalize_density_matrix",
    "normalize_result",
    "normalize_statevector",
    "probabilities_from_density_matrix",
    "probabilities_from_statevector",
]
