"""Tests for backend edge cases and new functionality."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from proxima.backends.base import (
    Capabilities,
    SimulatorType,
)
from proxima.backends.conversion import (
    CircuitFormat,
    CircuitInfo,
    detect_circuit_format,
    extract_circuit_info,
)
from proxima.backends.exceptions import (
    BackendError,
    BackendErrorCode,
    BackendNotInstalledError,
    BackendTimeoutError,
    CircuitValidationError,
    MemoryExceededError,
    QubitLimitExceededError,
    wrap_backend_exception,
)
from proxima.backends.execution import (
    BatchConfig,
    RetryConfig,
    calculate_backoff_delay,
    execute_with_retry,
    execute_with_timeout,
    should_retry,
)
from proxima.backends.health import (
    HealthCheckResult,
    HealthStatus,
    check_backend_health,
    generate_health_report,
)

# =============================================================================
# Exception Tests
# =============================================================================


class TestBackendExceptions:
    """Test backend exception classes and error handling."""

    def test_backend_error_basic(self) -> None:
        """Test basic BackendError creation."""
        error = BackendError(
            code=BackendErrorCode.EXECUTION_FAILED,
            message="Test error message",
            backend_name="test_backend",
        )
        assert error.code == BackendErrorCode.EXECUTION_FAILED
        assert error.message == "Test error message"
        assert error.backend_name == "test_backend"
        assert not error.recoverable

    def test_backend_error_to_dict(self) -> None:
        """Test BackendError serialization."""
        error = BackendError(
            code=BackendErrorCode.TIMEOUT,
            message="Operation timed out",
            backend_name="cirq",
            recoverable=True,
            retry_after_seconds=5.0,
            suggestions=["Increase timeout"],
        )
        data = error.to_dict()
        assert data["code"] == "timeout"
        assert data["recoverable"] is True
        assert data["retry_after_seconds"] == 5.0
        assert "Increase timeout" in data["suggestions"]

    def test_backend_not_installed_error(self) -> None:
        """Test BackendNotInstalledError with suggestions."""
        error = BackendNotInstalledError("qiskit", ["qiskit", "qiskit-aer"])
        assert error.code == BackendErrorCode.NOT_INSTALLED
        assert "qiskit" in error.message
        assert "qiskit-aer" in error.message
        assert len(error.suggestions) > 0
        assert not error.recoverable

    def test_backend_timeout_error(self) -> None:
        """Test BackendTimeoutError."""
        error = BackendTimeoutError("cirq", 30.0, "simulation")
        assert error.code == BackendErrorCode.TIMEOUT
        assert "30" in error.message
        assert error.recoverable
        assert error.retry_after_seconds == 30.0

    def test_circuit_validation_error(self) -> None:
        """Test CircuitValidationError."""
        error = CircuitValidationError(
            backend_name="qiskit",
            reason="Invalid gate sequence",
            circuit_info={"qubits": 5},
        )
        assert error.code == BackendErrorCode.INVALID_CIRCUIT
        assert "Invalid gate sequence" in error.message
        assert error.details["circuit_info"]["qubits"] == 5

    def test_qubit_limit_exceeded_error(self) -> None:
        """Test QubitLimitExceededError."""
        error = QubitLimitExceededError(
            backend_name="cirq",
            requested_qubits=35,
            max_qubits=30,
        )
        assert error.code == BackendErrorCode.QUBIT_LIMIT_EXCEEDED
        assert "35" in error.message
        assert "30" in error.message
        assert error.details["requested_qubits"] == 35
        assert error.details["max_qubits"] == 30

    def test_memory_exceeded_error(self) -> None:
        """Test MemoryExceededError."""
        error = MemoryExceededError("qiskit", 16000.0, 8000.0)
        assert error.code == BackendErrorCode.MEMORY_EXCEEDED
        assert "16000" in error.message
        assert "8000" in error.message
        assert not error.recoverable

    def test_wrap_backend_exception_memory(self) -> None:
        """Test wrapping memory-related exceptions."""
        exc = MemoryError("Unable to allocate array")
        wrapped = wrap_backend_exception(exc, "cirq", "simulation")
        assert wrapped.code == BackendErrorCode.MEMORY_EXCEEDED

    def test_wrap_backend_exception_timeout(self) -> None:
        """Test wrapping timeout-related exceptions."""
        exc = TimeoutError("Operation timed out after 30s")
        wrapped = wrap_backend_exception(exc, "qiskit", "execution")
        assert wrapped.code == BackendErrorCode.TIMEOUT

    def test_wrap_backend_exception_generic(self) -> None:
        """Test wrapping generic exceptions."""
        exc = ValueError("Something went wrong")
        wrapped = wrap_backend_exception(exc, "test", "operation")
        assert wrapped.code == BackendErrorCode.EXECUTION_FAILED
        assert wrapped.original_exception == exc


# =============================================================================
# Execution Utility Tests
# =============================================================================


class TestExecutionUtilities:
    """Test timeout, retry, and batch execution utilities."""

    def test_execute_with_timeout_success(self) -> None:
        """Test successful execution within timeout."""

        def fast_function() -> str:
            return "success"

        result = execute_with_timeout(fast_function, timeout_seconds=5.0)
        assert result == "success"

    def test_execute_with_timeout_timeout(self) -> None:
        """Test timeout behavior."""

        def slow_function() -> None:
            time.sleep(2)

        with pytest.raises(BackendTimeoutError):
            execute_with_timeout(
                slow_function, timeout_seconds=0.1, backend_name="test"
            )

    def test_retry_config_defaults(self) -> None:
        """Test RetryConfig default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay_seconds == 1.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_calculate_backoff_delay(self) -> None:
        """Test exponential backoff calculation."""
        config = RetryConfig(initial_delay_seconds=1.0, jitter=False)

        delay_1 = calculate_backoff_delay(1, config)
        delay_2 = calculate_backoff_delay(2, config)
        delay_3 = calculate_backoff_delay(3, config)

        assert delay_1 == 1.0
        assert delay_2 == 2.0
        assert delay_3 == 4.0

    def test_calculate_backoff_respects_max(self) -> None:
        """Test backoff respects max delay."""
        config = RetryConfig(
            initial_delay_seconds=10.0,
            max_delay_seconds=15.0,
            jitter=False,
        )

        delay = calculate_backoff_delay(10, config)
        assert delay == 15.0

    def test_should_retry_respects_max_retries(self) -> None:
        """Test retry logic respects max attempts."""
        config = RetryConfig(max_retries=3)
        error = BackendError(
            code=BackendErrorCode.EXECUTION_FAILED,
            message="Test",
            recoverable=True,
        )

        assert should_retry(error, config, 1) is True
        assert should_retry(error, config, 2) is True
        assert should_retry(error, config, 3) is False

    def test_should_retry_checks_recoverable(self) -> None:
        """Test retry only occurs for recoverable errors."""
        config = RetryConfig(max_retries=3)

        recoverable = BackendError(
            code=BackendErrorCode.TIMEOUT,
            message="Timeout",
            recoverable=True,
        )
        non_recoverable = BackendError(
            code=BackendErrorCode.NOT_INSTALLED,
            message="Not installed",
            recoverable=False,
        )

        assert should_retry(recoverable, config, 1) is True
        assert should_retry(non_recoverable, config, 1) is False

    def test_execute_with_retry_success(self) -> None:
        """Test retry succeeds on first attempt."""
        call_count = 0

        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = execute_with_retry(succeed)
        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1

    def test_execute_with_retry_eventual_success(self) -> None:
        """Test retry succeeds after failures."""
        call_count = 0

        def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise BackendError(
                    code=BackendErrorCode.TIMEOUT,
                    message="Retry please",
                    recoverable=True,
                )
            return "success"

        config = RetryConfig(max_retries=5, initial_delay_seconds=0.01)
        result = execute_with_retry(fail_twice, config=config)
        assert result.success is True
        assert result.attempts == 3

    def test_execute_with_retry_all_failures(self) -> None:
        """Test retry exhausts attempts."""

        def always_fail() -> None:
            raise BackendError(
                code=BackendErrorCode.TIMEOUT,
                message="Always fail",
                recoverable=True,
            )

        config = RetryConfig(max_retries=3, initial_delay_seconds=0.01)
        result = execute_with_retry(always_fail, config=config)
        assert result.success is False
        assert result.attempts == 3
        assert result.last_exception is not None


# =============================================================================
# Circuit Conversion Tests
# =============================================================================


class TestCircuitConversion:
    """Test circuit format detection and conversion utilities."""

    def test_detect_format_unknown(self) -> None:
        """Test format detection for unknown types."""
        assert detect_circuit_format(None) == CircuitFormat.UNKNOWN
        assert detect_circuit_format("random string") == CircuitFormat.UNKNOWN
        assert detect_circuit_format(12345) == CircuitFormat.UNKNOWN

    def test_detect_format_openqasm2(self) -> None:
        """Test OpenQASM 2.0 detection."""
        qasm = "OPENQASM 2.0;\ninclude qelib1.inc;\nqreg q[2];"
        assert detect_circuit_format(qasm) == CircuitFormat.OPENQASM2

    def test_detect_format_openqasm3(self) -> None:
        """Test OpenQASM 3.0 detection."""
        qasm = "OPENQASM 3;\nqubit[2] q;"
        assert detect_circuit_format(qasm) == CircuitFormat.OPENQASM3

    def test_extract_circuit_info_qasm(self) -> None:
        """Test info extraction from OpenQASM."""
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
measure q -> c;"""

        info = extract_circuit_info(qasm)
        assert info.format == CircuitFormat.OPENQASM2
        assert info.num_qubits == 3
        assert info.num_classical_bits == 3
        assert info.has_measurements is True

    def test_circuit_info_dataclass(self) -> None:
        """Test CircuitInfo dataclass fields."""
        info = CircuitInfo(
            format=CircuitFormat.QISKIT,
            num_qubits=5,
            num_classical_bits=5,
            depth=10,
            gate_count=20,
            gate_types={"h": 5, "cx": 10, "measure": 5},
            has_measurements=True,
        )
        assert info.num_qubits == 5
        assert info.depth == 10
        assert "h" in info.gate_types


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthChecks:
    """Test backend health check functionality."""

    def test_health_status_enum(self) -> None:
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_health_check_result_dataclass(self) -> None:
        """Test HealthCheckResult dataclass."""
        result = HealthCheckResult(
            backend_name="cirq",
            status=HealthStatus.HEALTHY,
            available=True,
            latency_ms=50.0,
            version="1.2.0",
            message="All checks passed",
        )
        assert result.backend_name == "cirq"
        assert result.status == HealthStatus.HEALTHY
        assert result.available is True
        assert result.latency_ms == 50.0

    def test_check_backend_health_unavailable(self) -> None:
        """Test health check for unavailable backend."""
        mock_adapter = MagicMock()
        mock_adapter.get_name.return_value = "test_backend"
        mock_adapter.is_available.return_value = False

        result = check_backend_health(mock_adapter)
        assert result.status == HealthStatus.UNHEALTHY
        assert result.available is False

    def test_check_backend_health_available(self) -> None:
        """Test health check for available backend."""
        mock_adapter = MagicMock()
        mock_adapter.get_name.return_value = "test_backend"
        mock_adapter.is_available.return_value = True
        mock_adapter.get_version.return_value = "1.0.0"
        mock_adapter.get_capabilities.return_value = Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR],
            max_qubits=30,
        )

        result = check_backend_health(mock_adapter, run_execution_test=False)
        assert result.status == HealthStatus.HEALTHY
        assert result.available is True
        assert result.version == "1.0.0"

    def test_generate_health_report(self) -> None:
        """Test health report generation."""
        results = [
            HealthCheckResult(
                backend_name="cirq",
                status=HealthStatus.HEALTHY,
                available=True,
            ),
            HealthCheckResult(
                backend_name="qiskit",
                status=HealthStatus.HEALTHY,
                available=True,
            ),
            HealthCheckResult(
                backend_name="lret",
                status=HealthStatus.UNHEALTHY,
                available=False,
            ),
        ]

        report = generate_health_report(results)
        assert report["total_backends"] == 3
        assert report["healthy"] == 2
        assert report["unhealthy"] == 1
        assert report["available"] == 2
        assert "cirq" in report["backends"]
        assert "qiskit" in report["backends"]

    def test_generate_health_report_all_healthy(self) -> None:
        """Test report when all backends healthy."""
        results = [
            HealthCheckResult(
                backend_name="test1",
                status=HealthStatus.HEALTHY,
                available=True,
            ),
            HealthCheckResult(
                backend_name="test2",
                status=HealthStatus.HEALTHY,
                available=True,
            ),
        ]

        report = generate_health_report(results)
        assert report["overall_status"] == "healthy"

    def test_generate_health_report_degraded(self) -> None:
        """Test report with degraded status."""
        results = [
            HealthCheckResult(
                backend_name="test1",
                status=HealthStatus.HEALTHY,
                available=True,
            ),
            HealthCheckResult(
                backend_name="test2",
                status=HealthStatus.DEGRADED,
                available=True,
            ),
        ]

        report = generate_health_report(results)
        assert report["overall_status"] == "degraded"


# =============================================================================
# Integration Tests with Mock Backends
# =============================================================================


class TestBackendIntegration:
    """Integration tests for backend adapters with mocked dependencies."""

    def test_cirq_adapter_not_installed(self) -> None:
        """Test CirqBackendAdapter when cirq not installed."""
        from proxima.backends.cirq_adapter import CirqBackendAdapter

        adapter = CirqBackendAdapter()

        with patch.object(adapter, "is_available", return_value=False):
            with pytest.raises(BackendNotInstalledError):
                adapter.execute(MagicMock())

    def test_qiskit_adapter_not_installed(self) -> None:
        """Test QiskitBackendAdapter when qiskit not installed."""
        from proxima.backends.qiskit_adapter import QiskitBackendAdapter

        adapter = QiskitBackendAdapter()

        with patch.object(adapter, "is_available", return_value=False):
            with pytest.raises(BackendNotInstalledError):
                adapter.execute(MagicMock())

    def test_lret_adapter_validation(self) -> None:
        """Test LRET adapter circuit validation."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()

        # Test None circuit
        result = adapter.validate_circuit(None)
        assert result.valid is False
        assert "None" in result.message

        # Test dict circuit with required keys
        result = adapter.validate_circuit({"gates": [], "qubits": 2})
        assert result.valid is True

    def test_registry_discovery(self) -> None:
        """Test backend registry discovery."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        statuses = registry.list_statuses()
        assert len(statuses) >= 3  # LRET, Cirq, Qiskit

        # All should have names
        for status in statuses:
            assert status.name in ["lret", "cirq", "qiskit"]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_circuit_validation(self) -> None:
        """Test validation of empty circuits."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()

        # Empty dict should be invalid
        result = adapter.validate_circuit({})
        assert result.valid is False

    def test_large_qubit_count(self) -> None:
        """Test handling of large qubit circuits."""
        error = QubitLimitExceededError("test", 100, 30)
        assert error.details["requested_qubits"] == 100
        assert "Reduce circuit" in error.suggestions[0]

    def test_error_string_representation(self) -> None:
        """Test error __str__ and __repr__."""
        error = BackendError(
            code=BackendErrorCode.TIMEOUT,
            message="Timed out",
            backend_name="test",
        )
        str_repr = str(error)
        assert "timeout" in str_repr.lower()
        assert "test" in str_repr

        repr_repr = repr(error)
        assert "BackendError" in repr_repr
        assert "TIMEOUT" in repr_repr

    def test_retry_with_on_retry_callback(self) -> None:
        """Test retry with callback notification."""
        callback_calls: list[tuple[int, Exception, float]] = []

        def on_retry(attempt: int, exc: Exception, delay: float) -> None:
            callback_calls.append((attempt, exc, delay))

        call_count = 0

        def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BackendError(
                    code=BackendErrorCode.TIMEOUT,
                    message="Retry",
                    recoverable=True,
                )
            return "success"

        config = RetryConfig(max_retries=3, initial_delay_seconds=0.01)
        result = execute_with_retry(fail_once, config=config, on_retry=on_retry)

        assert result.success is True
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 1  # First attempt failed

    def test_batch_config_defaults(self) -> None:
        """Test BatchConfig default values."""
        config = BatchConfig()
        assert config.max_batch_size == 10
        assert config.parallel is True
        assert config.max_workers == 4
        assert config.fail_fast is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
