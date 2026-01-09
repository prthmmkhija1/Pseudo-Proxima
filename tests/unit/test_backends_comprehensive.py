"""
Comprehensive Unit Tests for Backend System

Tests for:
- Backend data classes (SimulatorType, ResultType, Capabilities, etc.)
- BaseBackendAdapter interface
- BackendRegistry functionality
- Backend result handling
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from proxima.backends.base import (
    BaseBackendAdapter,
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
    ValidationResult,
)


# =============================================================================
# SIMULATOR TYPE TESTS
# =============================================================================


class TestSimulatorType:
    """Tests for SimulatorType enum."""

    @pytest.mark.unit
    @pytest.mark.backend
    def test_all_simulator_types_defined(self):
        """Verify all required simulator types are defined."""
        expected = ["STATE_VECTOR", "DENSITY_MATRIX", "CUSTOM"]
        for type_name in expected:
            assert hasattr(SimulatorType, type_name)

    @pytest.mark.unit
    @pytest.mark.backend
    def test_simulator_type_values(self):
        """Test simulator type string values."""
        assert SimulatorType.STATE_VECTOR.value == "state_vector"
        assert SimulatorType.DENSITY_MATRIX.value == "density_matrix"
        assert SimulatorType.CUSTOM.value == "custom"

    @pytest.mark.unit
    @pytest.mark.backend
    def test_simulator_type_is_string_enum(self):
        """SimulatorType should be a string enum."""
        assert isinstance(SimulatorType.STATE_VECTOR, str)
        assert SimulatorType.STATE_VECTOR == "state_vector"


# =============================================================================
# RESULT TYPE TESTS
# =============================================================================


class TestResultType:
    """Tests for ResultType enum."""

    @pytest.mark.unit
    @pytest.mark.backend
    def test_all_result_types_defined(self):
        """Verify all required result types are defined."""
        expected = ["COUNTS", "STATEVECTOR", "DENSITY_MATRIX"]
        for type_name in expected:
            assert hasattr(ResultType, type_name)

    @pytest.mark.unit
    @pytest.mark.backend
    def test_result_type_values(self):
        """Test result type string values."""
        assert ResultType.COUNTS.value == "counts"
        assert ResultType.STATEVECTOR.value == "statevector"
        assert ResultType.DENSITY_MATRIX.value == "density_matrix"


# =============================================================================
# CAPABILITIES TESTS
# =============================================================================


class TestCapabilities:
    """Tests for Capabilities dataclass."""

    @pytest.mark.unit
    @pytest.mark.backend
    def test_capabilities_creation(self):
        """Test basic Capabilities creation."""
        caps = Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR],
            max_qubits=25,
        )
        assert caps.max_qubits == 25
        assert SimulatorType.STATE_VECTOR in caps.simulator_types

    @pytest.mark.unit
    @pytest.mark.backend
    def test_capabilities_defaults(self):
        """Test Capabilities default values."""
        caps = Capabilities(
            simulator_types=[],
            max_qubits=10,
        )
        assert caps.supports_noise is False
        assert caps.supports_gpu is False
        assert caps.supports_batching is False
        assert caps.custom_features == {}

    @pytest.mark.unit
    @pytest.mark.backend
    def test_capabilities_with_all_options(self):
        """Test Capabilities with all options set."""
        caps = Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=30,
            supports_noise=True,
            supports_gpu=True,
            supports_batching=True,
            custom_features={"sparse_mode": True, "threads": 8},
        )
        assert caps.supports_noise is True
        assert caps.supports_gpu is True
        assert caps.supports_batching is True
        assert caps.custom_features["sparse_mode"] is True

    @pytest.mark.unit
    @pytest.mark.backend
    def test_capabilities_serializable(self):
        """Test Capabilities can be serialized."""
        caps = Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR],
            max_qubits=20,
        )
        data = asdict(caps)
        assert "simulator_types" in data
        assert "max_qubits" in data


# =============================================================================
# VALIDATION RESULT TESTS
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    @pytest.mark.unit
    @pytest.mark.backend
    def test_valid_result(self):
        """Test successful validation result."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.message is None
        assert result.details == {}

    @pytest.mark.unit
    @pytest.mark.backend
    def test_invalid_result_with_message(self):
        """Test failed validation with message."""
        result = ValidationResult(
            valid=False,
            message="Circuit too large for backend",
            details={"max_qubits": 25, "requested_qubits": 30},
        )
        assert result.valid is False
        assert "too large" in result.message
        assert result.details["max_qubits"] == 25

    @pytest.mark.unit
    @pytest.mark.backend
    def test_validation_result_as_bool(self):
        """Test ValidationResult can be used in boolean context."""
        valid = ValidationResult(valid=True)
        invalid = ValidationResult(valid=False)
        
        # Note: dataclass doesn't define __bool__ by default
        assert valid.valid
        assert not invalid.valid


# =============================================================================
# RESOURCE ESTIMATE TESTS
# =============================================================================


class TestResourceEstimate:
    """Tests for ResourceEstimate dataclass."""

    @pytest.mark.unit
    @pytest.mark.backend
    def test_resource_estimate_defaults(self):
        """Test ResourceEstimate default values."""
        estimate = ResourceEstimate()
        assert estimate.memory_mb is None
        assert estimate.time_ms is None
        assert estimate.metadata == {}

    @pytest.mark.unit
    @pytest.mark.backend
    def test_resource_estimate_with_values(self):
        """Test ResourceEstimate with all values."""
        estimate = ResourceEstimate(
            memory_mb=512.5,
            time_ms=1500.0,
            metadata={"simulator": "state_vector", "qubits": 20},
        )
        assert estimate.memory_mb == 512.5
        assert estimate.time_ms == 1500.0
        assert estimate.metadata["qubits"] == 20

    @pytest.mark.unit
    @pytest.mark.backend
    def test_resource_estimate_large_values(self):
        """Test ResourceEstimate with large circuit values."""
        estimate = ResourceEstimate(
            memory_mb=16384.0,  # 16 GB
            time_ms=3600000.0,  # 1 hour
            metadata={"qubits": 40, "gates": 1000000},
        )
        assert estimate.memory_mb > 10000
        assert estimate.time_ms > 1000000


# =============================================================================
# EXECUTION RESULT TESTS
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    @pytest.mark.unit
    @pytest.mark.backend
    def test_execution_result_creation(self):
        """Test basic ExecutionResult creation."""
        result = ExecutionResult(
            backend="cirq",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=150.5,
            qubit_count=5,
            shot_count=1024,
            result_type=ResultType.COUNTS,
        )
        assert result.backend == "cirq"
        assert result.qubit_count == 5
        assert result.shot_count == 1024

    @pytest.mark.unit
    @pytest.mark.backend
    def test_execution_result_with_counts(self):
        """Test ExecutionResult with measurement counts."""
        counts = {"00": 512, "11": 512}
        result = ExecutionResult(
            backend="qiskit-aer",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=200.0,
            qubit_count=2,
            shot_count=1024,
            result_type=ResultType.COUNTS,
            data={"counts": counts},
        )
        assert result.data["counts"]["00"] == 512
        assert result.data["counts"]["11"] == 512

    @pytest.mark.unit
    @pytest.mark.backend
    def test_execution_result_without_shots(self):
        """Test ExecutionResult without shots (statevector simulation)."""
        result = ExecutionResult(
            backend="cirq",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=50.0,
            qubit_count=3,
            shot_count=None,
            result_type=ResultType.STATEVECTOR,
            data={"statevector": [0.707, 0, 0, 0.707]},
        )
        assert result.shot_count is None
        assert result.result_type == ResultType.STATEVECTOR

    @pytest.mark.unit
    @pytest.mark.backend
    def test_execution_result_with_metadata(self):
        """Test ExecutionResult with metadata."""
        result = ExecutionResult(
            backend="lret",
            simulator_type=SimulatorType.CUSTOM,
            execution_time_ms=100.0,
            qubit_count=4,
            shot_count=1000,
            result_type=ResultType.COUNTS,
            metadata={
                "git_commit": "abc123",
                "optimization_level": 2,
                "noise_model": None,
            },
        )
        assert result.metadata["git_commit"] == "abc123"
        assert result.metadata["optimization_level"] == 2

    @pytest.mark.unit
    @pytest.mark.backend
    def test_execution_result_serializable(self):
        """Test ExecutionResult can be serialized."""
        result = ExecutionResult(
            backend="test",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=100.0,
            qubit_count=2,
            shot_count=1000,
            result_type=ResultType.COUNTS,
        )
        data = asdict(result)
        assert "backend" in data
        assert "execution_time_ms" in data


# =============================================================================
# MOCK BACKEND ADAPTER FOR TESTING
# =============================================================================


class MockBackendAdapter(BaseBackendAdapter):
    """Mock backend adapter for testing abstract interface."""

    def __init__(
        self,
        name: str = "mock",
        version: str = "1.0.0",
        available: bool = True,
        max_qubits: int = 25,
    ):
        self._name = name
        self._version = version
        self._available = available
        self._max_qubits = max_qubits

    def get_name(self) -> str:
        return self._name

    def get_version(self) -> str:
        return self._version

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=self._max_qubits,
            supports_noise=True,
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        if circuit is None:
            return ValidationResult(valid=False, message="Circuit is None")
        return ValidationResult(valid=True)

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        return ResourceEstimate(memory_mb=256.0, time_ms=100.0)

    def execute(self, circuit: Any, options: dict[str, Any] | None = None) -> ExecutionResult:
        shots = (options or {}).get("shots", 1024)
        return ExecutionResult(
            backend=self._name,
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=50.0,
            qubit_count=2,
            shot_count=shots,
            result_type=ResultType.COUNTS,
            data={"counts": {"00": 512, "11": 512}},
        )

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        return sim_type in [SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX]

    def is_available(self) -> bool:
        return self._available


# =============================================================================
# BASE BACKEND ADAPTER TESTS
# =============================================================================


class TestBaseBackendAdapter:
    """Tests for BaseBackendAdapter interface."""

    @pytest.fixture
    def mock_adapter(self) -> MockBackendAdapter:
        return MockBackendAdapter()

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_name(self, mock_adapter):
        """Test get_name returns correct name."""
        assert mock_adapter.get_name() == "mock"

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_version(self, mock_adapter):
        """Test get_version returns correct version."""
        assert mock_adapter.get_version() == "1.0.0"

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_capabilities(self, mock_adapter):
        """Test get_capabilities returns Capabilities object."""
        caps = mock_adapter.get_capabilities()
        assert isinstance(caps, Capabilities)
        assert caps.max_qubits == 25

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_validate_circuit_valid(self, mock_adapter):
        """Test validate_circuit with valid circuit."""
        result = mock_adapter.validate_circuit({"gates": []})
        assert result.valid is True

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_validate_circuit_invalid(self, mock_adapter):
        """Test validate_circuit with invalid circuit."""
        result = mock_adapter.validate_circuit(None)
        assert result.valid is False
        assert "None" in result.message

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_estimate_resources(self, mock_adapter):
        """Test estimate_resources returns ResourceEstimate."""
        estimate = mock_adapter.estimate_resources({"gates": []})
        assert isinstance(estimate, ResourceEstimate)
        assert estimate.memory_mb == 256.0

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_execute(self, mock_adapter):
        """Test execute returns ExecutionResult."""
        result = mock_adapter.execute({"gates": []})
        assert isinstance(result, ExecutionResult)
        assert result.backend == "mock"
        assert result.data["counts"]["00"] == 512

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_execute_with_options(self, mock_adapter):
        """Test execute respects options."""
        result = mock_adapter.execute({"gates": []}, options={"shots": 2048})
        assert result.shot_count == 2048

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_supports_simulator(self, mock_adapter):
        """Test supports_simulator for various types."""
        assert mock_adapter.supports_simulator(SimulatorType.STATE_VECTOR) is True
        assert mock_adapter.supports_simulator(SimulatorType.DENSITY_MATRIX) is True
        assert mock_adapter.supports_simulator(SimulatorType.CUSTOM) is False

    @pytest.mark.unit
    @pytest.mark.backend
    def test_adapter_is_available(self, mock_adapter):
        """Test is_available returns correct status."""
        assert mock_adapter.is_available() is True
        
        unavailable_adapter = MockBackendAdapter(available=False)
        assert unavailable_adapter.is_available() is False


# =============================================================================
# BACKEND REGISTRY TESTS
# =============================================================================


class TestBackendRegistry:
    """Tests for BackendRegistry functionality."""

    @pytest.mark.unit
    @pytest.mark.backend
    def test_registry_creation(self):
        """Test registry can be created."""
        from proxima.backends.registry import BackendRegistry
        
        registry = BackendRegistry()
        assert registry is not None

    @pytest.mark.unit
    @pytest.mark.backend
    def test_registry_register_adapter(self):
        """Test registering a custom adapter."""
        from proxima.backends.registry import BackendRegistry
        
        registry = BackendRegistry()
        adapter = MockBackendAdapter(name="custom-test", version="2.0.0")
        
        registry.register(adapter)
        
        # Should be able to get the status
        assert "custom-test" in registry._statuses
        assert registry._statuses["custom-test"].available is True

    @pytest.mark.unit
    @pytest.mark.backend
    def test_registry_discover_backends(self):
        """Test backend discovery."""
        from proxima.backends.registry import BackendRegistry
        
        registry = BackendRegistry()
        registry.discover()
        
        # Should discover at least LRET (which doesn't need external deps)
        assert len(registry._statuses) > 0
        assert "lret" in registry._statuses

    @pytest.mark.unit
    @pytest.mark.backend
    def test_backend_status_fields(self):
        """Test BackendStatus dataclass fields."""
        from proxima.backends.registry import BackendStatus
        
        status = BackendStatus(
            name="test",
            available=True,
            adapter=None,
            capabilities=None,
            version="1.0",
            reason=None,
        )
        
        assert status.name == "test"
        assert status.available is True
        assert status.version == "1.0"

    @pytest.mark.unit
    @pytest.mark.backend
    def test_unavailable_backend_has_reason(self):
        """Test unavailable backends have a reason."""
        from proxima.backends.registry import BackendStatus
        
        status = BackendStatus(
            name="missing-backend",
            available=False,
            adapter=None,
            capabilities=None,
            version=None,
            reason="dependency not installed",
        )
        
        assert status.available is False
        assert "not installed" in status.reason


# =============================================================================
# BACKEND COMPARISON TESTS
# =============================================================================


class TestBackendComparison:
    """Tests for comparing backend results."""

    @pytest.mark.unit
    @pytest.mark.backend
    def test_compare_identical_results(self):
        """Test comparing identical results from different backends."""
        result1 = ExecutionResult(
            backend="cirq",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=100.0,
            qubit_count=2,
            shot_count=1024,
            result_type=ResultType.COUNTS,
            data={"counts": {"00": 512, "11": 512}},
        )
        
        result2 = ExecutionResult(
            backend="qiskit-aer",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=120.0,
            qubit_count=2,
            shot_count=1024,
            result_type=ResultType.COUNTS,
            data={"counts": {"00": 510, "11": 514}},  # Slightly different due to randomness
        )
        
        # Both have similar distributions
        total1 = sum(result1.data["counts"].values())
        total2 = sum(result2.data["counts"].values())
        assert total1 == total2

    @pytest.mark.unit
    @pytest.mark.backend
    def test_backend_performance_comparison(self):
        """Test comparing backend performance."""
        fast_result = ExecutionResult(
            backend="fast-backend",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=50.0,
            qubit_count=2,
            shot_count=1024,
            result_type=ResultType.COUNTS,
        )
        
        slow_result = ExecutionResult(
            backend="slow-backend",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=500.0,
            qubit_count=2,
            shot_count=1024,
            result_type=ResultType.COUNTS,
        )
        
        assert fast_result.execution_time_ms < slow_result.execution_time_ms
        speedup = slow_result.execution_time_ms / fast_result.execution_time_ms
        assert speedup == 10.0
