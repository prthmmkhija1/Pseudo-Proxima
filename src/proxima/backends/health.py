"""Backend health check and connection testing utilities."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from proxima.backends.base import BaseBackendAdapter, ExecutionResult, SimulatorType


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a backend health check."""

    backend_name: str
    status: HealthStatus
    available: bool
    latency_ms: float | None = None
    version: str | None = None
    message: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConnectionTestResult:
    """Result of a connection/execution test."""

    backend_name: str
    success: bool
    execution_time_ms: float | None = None
    result_valid: bool = False
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def check_backend_health(
    adapter: BaseBackendAdapter,
    *,
    run_execution_test: bool = False,
    timeout_seconds: float = 30.0,
) -> HealthCheckResult:
    """Perform comprehensive health check on a backend.

    Args:
        adapter: The backend adapter to check
        run_execution_test: Whether to run a simple circuit execution
        timeout_seconds: Timeout for checks

    Returns:
        HealthCheckResult with detailed status
    """
    backend_name = adapter.get_name()
    checks: dict[str, bool] = {}
    metadata: dict[str, Any] = {}
    start_time = time.perf_counter()

    # Check 1: Availability
    try:
        available = adapter.is_available()
        checks["availability"] = available
    except Exception as exc:
        checks["availability"] = False
        metadata["availability_error"] = str(exc)
        return HealthCheckResult(
            backend_name=backend_name,
            status=HealthStatus.UNHEALTHY,
            available=False,
            message=f"Availability check failed: {exc}",
            checks=checks,
            metadata=metadata,
        )

    if not available:
        return HealthCheckResult(
            backend_name=backend_name,
            status=HealthStatus.UNHEALTHY,
            available=False,
            message="Backend is not available (dependencies not installed)",
            checks=checks,
            metadata=metadata,
        )

    # Check 2: Version retrieval
    try:
        version = adapter.get_version()
        checks["version"] = version not in ("unknown", "unavailable")
        metadata["version"] = version
    except Exception as exc:
        checks["version"] = False
        metadata["version_error"] = str(exc)
        version = None

    # Check 3: Capabilities retrieval
    try:
        capabilities = adapter.get_capabilities()
        checks["capabilities"] = capabilities is not None
        metadata["max_qubits"] = capabilities.max_qubits
        metadata["simulator_types"] = [st.value for st in capabilities.simulator_types]
        metadata["supports_noise"] = capabilities.supports_noise
    except Exception as exc:
        checks["capabilities"] = False
        metadata["capabilities_error"] = str(exc)

    # Check 4: Execution test (optional)
    if run_execution_test:
        exec_result = run_simple_execution_test(adapter, timeout_seconds)
        checks["execution"] = exec_result.success
        if exec_result.execution_time_ms:
            metadata["test_execution_ms"] = exec_result.execution_time_ms
        if exec_result.error:
            metadata["execution_error"] = exec_result.error

    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Determine overall status
    failed_checks = [k for k, v in checks.items() if not v]
    if not failed_checks:
        status = HealthStatus.HEALTHY
        message = "All checks passed"
    elif len(failed_checks) == 1 and failed_checks[0] in ("version", "execution"):
        status = HealthStatus.DEGRADED
        message = f"Degraded: {failed_checks[0]} check failed"
    else:
        status = HealthStatus.UNHEALTHY
        message = f"Failed checks: {', '.join(failed_checks)}"

    return HealthCheckResult(
        backend_name=backend_name,
        status=status,
        available=available,
        latency_ms=latency_ms,
        version=version,
        message=message,
        checks=checks,
        metadata=metadata,
    )


def run_simple_execution_test(
    adapter: BaseBackendAdapter,
    timeout_seconds: float = 30.0,
) -> ConnectionTestResult:
    """Run a simple circuit execution to test the backend.

    Creates a minimal 1-qubit circuit and verifies execution works.

    Args:
        adapter: The backend adapter to test
        timeout_seconds: Maximum time for the test

    Returns:
        ConnectionTestResult with test outcome
    """
    backend_name = adapter.get_name()
    details: dict[str, Any] = {}

    try:
        # Create appropriate test circuit based on backend
        circuit = create_test_circuit(backend_name)
        if circuit is None:
            return ConnectionTestResult(
                backend_name=backend_name,
                success=False,
                error="Could not create test circuit for this backend",
            )

        details["circuit_type"] = type(circuit).__name__

        # Validate circuit first
        validation = adapter.validate_circuit(circuit)
        if not validation.valid:
            return ConnectionTestResult(
                backend_name=backend_name,
                success=False,
                error=f"Circuit validation failed: {validation.message}",
                details=details,
            )

        # Execute circuit
        start_time = time.perf_counter()
        result = adapter.execute(circuit, {"simulator_type": SimulatorType.STATE_VECTOR})
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Verify result
        result_valid = verify_test_result(result)
        details["result_type"] = result.result_type.value
        details["qubit_count"] = result.qubit_count

        return ConnectionTestResult(
            backend_name=backend_name,
            success=True,
            execution_time_ms=execution_time_ms,
            result_valid=result_valid,
            details=details,
        )

    except Exception as exc:
        return ConnectionTestResult(
            backend_name=backend_name,
            success=False,
            error=str(exc),
            details=details,
        )


def create_test_circuit(backend_name: str) -> Any:
    """Create a minimal test circuit for the given backend.

    Args:
        backend_name: Name of the backend

    Returns:
        A simple 1-qubit circuit or None if creation fails
    """
    if backend_name == "cirq":
        try:
            import cirq

            q = cirq.LineQubit(0)
            return cirq.Circuit([cirq.H(q)])
        except ImportError:
            return None

    elif backend_name == "qiskit":
        try:
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(1)
            qc.h(0)
            return qc
        except ImportError:
            return None

    elif backend_name == "lret":
        # LRET accepts dictionary-based circuit format
        # This creates a simple Hadamard gate test circuit
        return {"type": "test", "qubits": 1, "gates": [{"name": "H", "qubit": 0}]}

    return None


def verify_test_result(result: ExecutionResult) -> bool:
    """Verify that a test execution result is valid.

    For a 1-qubit H gate, we expect:
    - Statevector with 2 amplitudes
    - Each amplitude should be approximately 1/sqrt(2)

    Args:
        result: The execution result to verify

    Returns:
        True if result appears valid
    """
    try:
        if result.qubit_count != 1:
            return False

        data = result.data
        if "statevector" in data:
            sv = data["statevector"]
            if hasattr(sv, "__len__") and len(sv) == 2:
                # Check amplitudes are roughly equal magnitude
                import numpy as np

                probs = np.abs(sv) ** 2
                return bool(np.allclose(probs, [0.5, 0.5], atol=0.1))

        if "density_matrix" in data:
            dm = data["density_matrix"]
            if hasattr(dm, "shape") and dm.shape == (2, 2):
                import numpy as np

                # Diagonal should be [0.5, 0.5] for |+> state
                diag = np.real(np.diag(dm))
                return bool(np.allclose(diag, [0.5, 0.5], atol=0.1))

        # For counts, check distribution
        if "counts" in data:
            counts = data["counts"]
            total = sum(counts.values())
            if total > 0:
                # Both states should appear with roughly equal probability
                probs = {k: v / total for k, v in counts.items()}
                if len(probs) == 2:
                    return all(0.3 < p < 0.7 for p in probs.values())

        return True  # Accept if we can't verify

    except Exception:
        return False


async def check_backends_health_async(
    adapters: list[BaseBackendAdapter],
    *,
    run_execution_tests: bool = False,
) -> list[HealthCheckResult]:
    """Check health of multiple backends concurrently.

    Args:
        adapters: List of backend adapters to check
        run_execution_tests: Whether to run execution tests

    Returns:
        List of HealthCheckResult for each backend
    """
    loop = asyncio.get_event_loop()

    async def check_one(adapter: BaseBackendAdapter) -> HealthCheckResult:
        return await loop.run_in_executor(
            None,
            lambda: check_backend_health(adapter, run_execution_test=run_execution_tests),
        )

    tasks = [check_one(adapter) for adapter in adapters]
    return await asyncio.gather(*tasks)


def generate_health_report(results: list[HealthCheckResult]) -> dict[str, Any]:
    """Generate a summary health report from multiple check results.

    Args:
        results: List of health check results

    Returns:
        Summary report dictionary
    """
    healthy_count = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
    degraded_count = sum(1 for r in results if r.status == HealthStatus.DEGRADED)
    unhealthy_count = sum(1 for r in results if r.status == HealthStatus.UNHEALTHY)
    available_count = sum(1 for r in results if r.available)

    overall_status = HealthStatus.HEALTHY
    if unhealthy_count > 0:
        overall_status = (
            HealthStatus.UNHEALTHY if unhealthy_count == len(results) else HealthStatus.DEGRADED
        )
    elif degraded_count > 0:
        overall_status = HealthStatus.DEGRADED

    return {
        "overall_status": overall_status.value,
        "total_backends": len(results),
        "available": available_count,
        "healthy": healthy_count,
        "degraded": degraded_count,
        "unhealthy": unhealthy_count,
        "backends": {
            r.backend_name: {
                "status": r.status.value,
                "available": r.available,
                "version": r.version,
                "latency_ms": r.latency_ms,
                "message": r.message,
            }
            for r in results
        },
        "timestamp": time.time(),
    }
