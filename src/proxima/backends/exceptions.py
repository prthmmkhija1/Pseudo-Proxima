"""Backend-specific exceptions for comprehensive error handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BackendErrorCode(str, Enum):
    """Categorized error codes for backend operations."""

    # Connection/availability errors
    NOT_INSTALLED = "not_installed"
    IMPORT_FAILED = "import_failed"
    CONNECTION_FAILED = "connection_failed"
    TIMEOUT = "timeout"

    # Validation errors
    INVALID_CIRCUIT = "invalid_circuit"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    QUBIT_LIMIT_EXCEEDED = "qubit_limit_exceeded"
    INVALID_OPTIONS = "invalid_options"

    # Resource errors
    MEMORY_EXCEEDED = "memory_exceeded"
    RESOURCE_EXHAUSTED = "resource_exhausted"

    # Execution errors
    EXECUTION_FAILED = "execution_failed"
    SIMULATION_ERROR = "simulation_error"
    RESULT_EXTRACTION_FAILED = "result_extraction_failed"

    # Internal errors
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


@dataclass
class BackendError(Exception):
    """Base exception for all backend-related errors.

    Provides structured error information including error code,
    message, recovery suggestions, and original exception details.
    """

    code: BackendErrorCode
    message: str
    backend_name: str = ""
    recoverable: bool = False
    retry_after_seconds: float | None = None
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    original_exception: Exception | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [f"[{self.code.value}]"]
        if self.backend_name:
            parts.append(f"({self.backend_name})")
        parts.append(self.message)
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"BackendError(code={self.code!r}, message={self.message!r}, "
            f"backend_name={self.backend_name!r}, recoverable={self.recoverable})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert error to serializable dictionary."""
        return {
            "code": self.code.value,
            "message": self.message,
            "backend_name": self.backend_name,
            "recoverable": self.recoverable,
            "retry_after_seconds": self.retry_after_seconds,
            "suggestions": self.suggestions,
            "details": self.details,
            "original_exception": str(self.original_exception) if self.original_exception else None,
        }


class BackendNotInstalledError(BackendError):
    """Raised when a backend's dependencies are not installed."""

    def __init__(
        self,
        backend_name: str,
        missing_packages: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        packages = missing_packages or [backend_name]
        suggestions = [
            f"Install the required package(s): pip install {' '.join(packages)}",
            "Check that you have the correct Python environment activated",
        ]
        super().__init__(
            code=BackendErrorCode.NOT_INSTALLED,
            message=f"Backend '{backend_name}' requires package(s): {', '.join(packages)}",
            backend_name=backend_name,
            recoverable=False,
            suggestions=suggestions,
            details={"missing_packages": packages},
            **kwargs,
        )


class BackendTimeoutError(BackendError):
    """Raised when a backend operation times out."""

    def __init__(
        self,
        backend_name: str,
        timeout_seconds: float,
        operation: str = "execution",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            code=BackendErrorCode.TIMEOUT,
            message=f"Backend '{backend_name}' timed out after {timeout_seconds}s during {operation}",
            backend_name=backend_name,
            recoverable=True,
            retry_after_seconds=timeout_seconds,
            suggestions=[
                "Increase the timeout limit",
                "Try a simpler circuit with fewer qubits/gates",
                "Check system resource availability",
            ],
            details={"timeout_seconds": timeout_seconds, "operation": operation},
            **kwargs,
        )


class CircuitValidationError(BackendError):
    """Raised when circuit validation fails."""

    def __init__(
        self,
        backend_name: str,
        reason: str,
        circuit_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            code=BackendErrorCode.INVALID_CIRCUIT,
            message=f"Circuit validation failed: {reason}",
            backend_name=backend_name,
            recoverable=False,
            suggestions=[
                "Ensure the circuit type matches the backend (Cirq, Qiskit, etc.)",
                "Check that all gates are supported by this backend",
                "Verify qubit count is within backend limits",
            ],
            details={"reason": reason, "circuit_info": circuit_info or {}},
            **kwargs,
        )


class QubitLimitExceededError(BackendError):
    """Raised when circuit exceeds backend's qubit limit."""

    def __init__(
        self,
        backend_name: str,
        requested_qubits: int,
        max_qubits: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            code=BackendErrorCode.QUBIT_LIMIT_EXCEEDED,
            message=(
                f"Circuit requires {requested_qubits} qubits, "
                f"but backend '{backend_name}' supports max {max_qubits}"
            ),
            backend_name=backend_name,
            recoverable=False,
            suggestions=[
                f"Reduce circuit to {max_qubits} qubits or fewer",
                "Use a backend with higher qubit capacity",
                "Consider circuit partitioning techniques",
            ],
            details={
                "requested_qubits": requested_qubits,
                "max_qubits": max_qubits,
            },
            **kwargs,
        )


class UnsupportedOperationError(BackendError):
    """Raised when an operation is not supported by the backend."""

    def __init__(
        self,
        backend_name: str,
        operation: str,
        supported_operations: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            code=BackendErrorCode.UNSUPPORTED_OPERATION,
            message=f"Operation '{operation}' not supported by backend '{backend_name}'",
            backend_name=backend_name,
            recoverable=False,
            suggestions=[
                "Use an alternative backend that supports this operation",
                "Try an equivalent operation that is supported",
            ],
            details={
                "operation": operation,
                "supported_operations": supported_operations or [],
            },
            **kwargs,
        )


class ExecutionError(BackendError):
    """Raised when circuit execution fails."""

    def __init__(
        self,
        backend_name: str,
        reason: str,
        stage: str = "execution",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            code=BackendErrorCode.EXECUTION_FAILED,
            message=f"Execution failed at {stage}: {reason}",
            backend_name=backend_name,
            recoverable=True,
            suggestions=[
                "Check circuit for unsupported operations",
                "Verify system has sufficient memory",
                "Try with fewer shots or a simpler circuit",
            ],
            details={"reason": reason, "stage": stage},
            **kwargs,
        )


class MemoryExceededError(BackendError):
    """Raised when execution would exceed memory limits."""

    def __init__(
        self,
        backend_name: str,
        required_mb: float,
        available_mb: float | None = None,
        **kwargs: Any,
    ) -> None:
        message = f"Circuit requires ~{required_mb:.1f}MB memory"
        if available_mb is not None:
            message += f", but only ~{available_mb:.1f}MB available"
        super().__init__(
            code=BackendErrorCode.MEMORY_EXCEEDED,
            message=message,
            backend_name=backend_name,
            recoverable=False,
            suggestions=[
                "Reduce qubit count (memory scales as 2^n)",
                "Use density matrix simulator for mixed states",
                "Close other applications to free memory",
            ],
            details={
                "required_mb": required_mb,
                "available_mb": available_mb,
            },
            **kwargs,
        )


def wrap_backend_exception(
    exc: Exception,
    backend_name: str,
    operation: str = "execution",
) -> BackendError:
    """Wrap a generic exception as a BackendError.

    Args:
        exc: The original exception
        backend_name: Name of the backend
        operation: What operation was being performed

    Returns:
        A BackendError wrapping the original exception
    """
    # Check for known exception types
    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    if "memory" in exc_msg or "alloc" in exc_msg:
        return BackendError(
            code=BackendErrorCode.MEMORY_EXCEEDED,
            message=f"Memory error during {operation}: {exc}",
            backend_name=backend_name,
            recoverable=False,
            original_exception=exc,
        )

    if "timeout" in exc_msg or "timed out" in exc_msg:
        return BackendError(
            code=BackendErrorCode.TIMEOUT,
            message=f"Timeout during {operation}: {exc}",
            backend_name=backend_name,
            recoverable=True,
            original_exception=exc,
        )

    if "import" in exc_type or "module" in exc_msg:
        return BackendError(
            code=BackendErrorCode.IMPORT_FAILED,
            message=f"Import error: {exc}",
            backend_name=backend_name,
            recoverable=False,
            original_exception=exc,
        )

    # Default to execution failed
    return ExecutionError(
        backend_name=backend_name,
        reason=str(exc),
        stage=operation,
        original_exception=exc,
    )
