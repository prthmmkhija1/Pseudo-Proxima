"""Tests for Phase 10.2: Error Handling and Robustness."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proxima.benchmarks.error_handling import (
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


class TestBenchmarkError:
    """Tests for BenchmarkError data class."""

    def test_error_creation(self) -> None:
        """BenchmarkError can be created with required fields."""
        error = BenchmarkError(
            code="E001",
            message="Test error",
            severity=ErrorSeverity.ERROR,
        )

        assert error.code == "E001"
        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.ERROR
        assert error.recoverable is True

    def test_to_dict(self) -> None:
        """BenchmarkError serializes to dictionary."""
        error = BenchmarkError(
            code="E001",
            message="Test error",
            severity=ErrorSeverity.WARNING,
            context={"backend": "lret"},
        )

        result = error.to_dict()

        assert result["code"] == "E001"
        assert result["severity"] == "WARNING"
        assert result["context"]["backend"] == "lret"
        assert "timestamp" in result

    def test_to_log_record(self) -> None:
        """BenchmarkError creates valid JSON log record."""
        error = BenchmarkError(
            code="E001",
            message="Test error",
            severity=ErrorSeverity.ERROR,
        )

        log_record = error.to_log_record()
        parsed = json.loads(log_record)

        assert parsed["code"] == "E001"
        assert parsed["message"] == "Test error"


class TestErrorHandler:
    """Tests for ErrorHandler."""

    def test_handle_exception(self) -> None:
        """ErrorHandler captures and logs exceptions."""
        handler = ErrorHandler()

        try:
            raise ValueError("Test exception")
        except Exception as e:
            error = handler.handle(
                e,
                code=BenchmarkErrorCodes.VALIDATION_FAILED,
                context={"test": True},
            )

        assert error.code == BenchmarkErrorCodes.VALIDATION_FAILED
        assert "Test exception" in error.message
        assert error.context["test"] is True
        assert error.traceback_str is not None

    def test_get_errors_filtered(self) -> None:
        """ErrorHandler filters errors by severity."""
        handler = ErrorHandler()

        handler.handle(ValueError("warning"), severity=ErrorSeverity.WARNING)
        handler.handle(ValueError("error"), severity=ErrorSeverity.ERROR)
        handler.handle(ValueError("critical"), severity=ErrorSeverity.CRITICAL)

        warnings = handler.get_errors(severity=ErrorSeverity.WARNING)
        assert len(warnings) == 1

        errors = handler.get_errors(severity=ErrorSeverity.ERROR)
        assert len(errors) == 1

    def test_has_critical_errors(self) -> None:
        """ErrorHandler detects critical errors."""
        handler = ErrorHandler()

        assert handler.has_critical_errors is False

        handler.handle(ValueError("error"), severity=ErrorSeverity.ERROR)
        assert handler.has_critical_errors is False

        handler.handle(ValueError("critical"), severity=ErrorSeverity.CRITICAL)
        assert handler.has_critical_errors is True

    def test_clear_errors(self) -> None:
        """ErrorHandler can clear recorded errors."""
        handler = ErrorHandler()

        handler.handle(ValueError("error1"))
        handler.handle(ValueError("error2"))

        assert len(handler.get_errors()) == 2

        handler.clear_errors()
        assert len(handler.get_errors()) == 0


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_valid_result(self) -> None:
        """Valid result returns True in boolean context."""
        result = ValidationResult(valid=True)
        assert result
        assert bool(result) is True

    def test_invalid_result(self) -> None:
        """Invalid result returns False in boolean context."""
        result = ValidationResult(valid=False, errors=["Error 1"])
        assert not result
        assert bool(result) is False

    def test_warnings(self) -> None:
        """Result can have warnings without being invalid."""
        result = ValidationResult(valid=True, warnings=["Warning 1"])
        assert result
        assert len(result.warnings) == 1


class TestCircuitValidator:
    """Tests for CircuitValidator."""

    def test_none_circuit_invalid(self) -> None:
        """None circuit is invalid."""
        result = CircuitValidator.validate(None)

        assert not result.valid
        assert "cannot be None" in result.errors[0]

    def test_circuit_with_qubit_count(self) -> None:
        """Circuit with num_qubits attribute is validated."""
        mock_circuit = MagicMock()
        mock_circuit.num_qubits = 10

        result = CircuitValidator.validate(mock_circuit)

        assert result.valid

    def test_circuit_exceeds_max_qubits(self) -> None:
        """Circuit exceeding max qubits is invalid."""
        mock_circuit = MagicMock()
        mock_circuit.num_qubits = 60  # Exceeds MAX_QUBITS (50)

        result = CircuitValidator.validate(mock_circuit)

        assert not result.valid
        assert "exceeds maximum" in result.errors[0]

    def test_large_circuit_warning(self) -> None:
        """Large circuit generates warning but is valid."""
        mock_circuit = MagicMock()
        mock_circuit.num_qubits = 35  # > 30 but < 50

        result = CircuitValidator.validate(mock_circuit)

        assert result.valid
        assert len(result.warnings) > 0
        assert "significant memory" in result.warnings[0]


class TestBackendValidator:
    """Tests for BackendValidator."""

    def test_empty_backend_name_invalid(self) -> None:
        """Empty backend name is invalid."""
        result = BackendValidator.validate("", MagicMock())

        assert not result.valid
        assert "cannot be empty" in result.errors[0]

    def test_backend_not_found(self) -> None:
        """Backend not in registry is invalid."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        result = BackendValidator.validate("unknown", mock_registry)

        assert not result.valid
        assert "not found" in result.errors[0]

    def test_backend_unavailable(self) -> None:
        """Unavailable backend is invalid."""
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = False

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_backend

        result = BackendValidator.validate("lret", mock_registry)

        assert not result.valid
        assert "not available" in result.errors[0]

    def test_available_backend_valid(self) -> None:
        """Available backend is valid."""
        mock_backend = MagicMock()
        mock_backend.is_available.return_value = True

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_backend

        result = BackendValidator.validate("lret", mock_registry)

        assert result.valid


class TestResultValidator:
    """Tests for ResultValidator."""

    def test_none_result_invalid(self) -> None:
        """None result is invalid."""
        result = ResultValidator.validate(None)

        assert not result.valid

    def test_valid_metrics(self) -> None:
        """Result with valid metrics is valid."""
        mock_result = MagicMock()
        mock_result.metrics.execution_time_ms = 100.0
        mock_result.metrics.memory_peak_mb = 512.0
        mock_result.metrics.success_rate_percent = 100.0
        mock_result.metrics.cpu_usage_percent = 50.0

        result = ResultValidator.validate(mock_result)

        assert result.valid

    def test_negative_time_invalid(self) -> None:
        """Negative execution time is invalid."""
        mock_result = MagicMock()
        mock_result.metrics.execution_time_ms = -10.0
        mock_result.metrics.memory_peak_mb = 100.0
        mock_result.metrics.success_rate_percent = 100.0
        mock_result.metrics.cpu_usage_percent = 50.0

        result = ResultValidator.validate(mock_result)

        assert not result.valid
        assert "Negative execution time" in result.errors[0]

    def test_invalid_percentage(self) -> None:
        """Percentage outside 0-100 is invalid."""
        mock_result = MagicMock()
        mock_result.metrics.execution_time_ms = 100.0
        mock_result.metrics.memory_peak_mb = 100.0
        mock_result.metrics.success_rate_percent = 150.0  # Invalid
        mock_result.metrics.cpu_usage_percent = 50.0

        result = ResultValidator.validate(mock_result)

        assert not result.valid


class TestRetryLogic:
    """Tests for retry decorator and helpers."""

    def test_is_retryable_database_lock(self) -> None:
        """Database lock errors are retryable."""
        exc = sqlite3.OperationalError("database is locked")
        assert is_retryable(exc) is True

    def test_is_retryable_timeout(self) -> None:
        """Timeout errors are retryable."""
        exc = Exception("connection timeout")
        assert is_retryable(exc) is True

    def test_is_not_retryable_value_error(self) -> None:
        """ValueError is not retryable."""
        exc = ValueError("invalid circuit")
        assert is_retryable(exc) is False

    def test_with_retry_succeeds(self) -> None:
        """Decorated function succeeds on first try."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3))
        def success_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()

        assert result == "success"
        assert call_count == 1

    def test_with_retry_succeeds_after_failures(self) -> None:
        """Decorated function retries on transient failures."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, initial_delay=0.01))
        def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count == 3

    def test_with_retry_exhausted(self) -> None:
        """Decorated function raises RetryError when exhausted."""
        @with_retry(RetryConfig(max_attempts=2, initial_delay=0.01))
        def always_fails() -> str:
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(RetryError) as exc_info:
            always_fails()

        assert exc_info.value.attempts == 2

    def test_with_retry_no_retry_on_deterministic_error(self) -> None:
        """Decorated function doesn't retry deterministic errors."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3))
        def deterministic_failure() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid input")

        with pytest.raises(ValueError):
            deterministic_failure()

        assert call_count == 1


class TestCrashRecoveryManager:
    """Tests for crash recovery."""

    def test_start_suite_creates_checkpoint(self) -> None:
        """Starting a suite creates checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecoveryManager(
                checkpoint_dir=Path(tmpdir),
                checkpoint_interval=1,
            )

            recovery.start_suite("test_suite", total_benchmarks=10)

            assert recovery._current_state is not None
            assert recovery._current_state.suite_name == "test_suite"
            assert recovery._current_state.total_benchmarks == 10

    def test_checkpoint_saves_state(self) -> None:
        """Checkpointing saves state to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecoveryManager(
                checkpoint_dir=Path(tmpdir),
                checkpoint_interval=1,
            )

            recovery.start_suite("test_suite", total_benchmarks=5)

            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"id": "test"}

            recovery.checkpoint(mock_result)

            # State should be updated
            assert recovery._current_state.completed_benchmarks == 1
            assert len(recovery._current_state.results_so_far) == 1

    def test_complete_suite_removes_checkpoint(self) -> None:
        """Completing a suite removes checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecoveryManager(
                checkpoint_dir=Path(tmpdir),
                checkpoint_interval=1,
            )

            recovery.start_suite("test_suite", total_benchmarks=1)

            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"id": "test"}
            recovery.checkpoint(mock_result)

            checkpoint_path = recovery._current_state.checkpoint_path
            recovery.complete_suite()

            assert recovery._current_state is None

    def test_find_incomplete_suites(self) -> None:
        """Manager finds incomplete suite checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a checkpoint file manually
            checkpoint = {
                "suite_name": "incomplete",
                "total_benchmarks": 10,
                "completed_benchmarks": 5,
                "results_so_far": [],
                "timestamp": datetime.utcnow().isoformat(),
            }
            checkpoint_path = Path(tmpdir) / "incomplete_123.json"
            with checkpoint_path.open("w") as f:
                json.dump(checkpoint, f)

            recovery = CrashRecoveryManager(checkpoint_dir=Path(tmpdir))
            incomplete = recovery.find_incomplete_suites()

            assert len(incomplete) == 1
            assert incomplete[0].suite_name == "incomplete"
            assert incomplete[0].completed_benchmarks == 5


class TestDatabaseRecovery:
    """Tests for database recovery utilities."""

    def test_check_integrity_valid_db(self) -> None:
        """Integrity check passes for valid database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            conn = sqlite3.connect(f.name)
            conn.execute("CREATE TABLE test (id INTEGER);")
            conn.close()

            result = DatabaseRecovery.check_integrity(Path(f.name))

            assert result.valid

    def test_check_integrity_missing_db(self) -> None:
        """Integrity check fails for missing database."""
        result = DatabaseRecovery.check_integrity(Path("/nonexistent/path.db"))

        assert not result.valid
        assert "not found" in result.errors[0]

    def test_create_backup(self) -> None:
        """Database backup is created successfully."""
        import gc
        import sys
        
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER);")
            conn.execute("INSERT INTO test VALUES (1);")
            conn.commit()  # Ensure data is committed before backup
            conn.close()

            backup_path = DatabaseRecovery.create_backup(
                db_path,
                backup_dir=Path(tmpdir) / "backups",
            )

            assert backup_path is not None
            assert backup_path.exists()

            # Verify backup content
            backup_conn = sqlite3.connect(backup_path)
            cursor = backup_conn.execute("SELECT * FROM test")
            row = cursor.fetchone()
            assert row is not None, "Backup database should have data"
            assert row[0] == 1
            cursor.close()
            backup_conn.close()
        finally:
            # Force garbage collection and cleanup on Windows
            gc.collect()
            # Allow time for Windows to release file handles
            if sys.platform == 'win32':
                import time
                time.sleep(0.1)
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors on Windows


class TestSchemaValidator:
    """Tests for database schema validation."""

    def test_fresh_database_valid(self) -> None:
        """Fresh database with no schema table is valid."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            conn = sqlite3.connect(f.name)

            result = SchemaValidator.validate(conn)

            assert result.valid
            conn.close()

    def test_outdated_schema_warning(self) -> None:
        """Outdated schema version generates warning."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            conn = sqlite3.connect(f.name)
            conn.execute("CREATE TABLE schema_version (version INTEGER);")
            conn.execute("INSERT INTO schema_version (version) VALUES (0);")
            conn.commit()

            result = SchemaValidator.validate(conn)

            assert result.valid  # Still valid, just outdated
            assert len(result.warnings) > 0
            assert "outdated" in result.warnings[0].lower()
            conn.close()
