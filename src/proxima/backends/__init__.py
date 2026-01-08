"""Backend adapters module."""

from proxima.backends.base import (
    BaseBackendAdapter,
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
    ValidationResult,
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
