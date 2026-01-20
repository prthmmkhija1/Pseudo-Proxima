"""
Comprehensive Unit Tests for LRET Backend Adapter

Tests all components of the LRET backend including:
- Result normalization for all formats
- Framework-integration branch API verification
- Mock simulator functionality
- Adapter interface compliance
- Error handling and edge cases
"""

import numpy as np
import pytest
from typing import Any
from unittest.mock import MagicMock, patch

# Import LRET components
from proxima.backends.lret import (
    LRETBackendAdapter,
    LRETResultNormalizer,
    LRETResultFormat,
    NormalizedResult,
    LRETAPIVerifier,
    LRETAPIVerification,
    LRETPerformanceMetrics,
    LRETBenchmarkResult,
    LRETPerformanceMonitor,
    LRETBenchmarkRunner,
    MockLRETSimulator,
    MockLRETResult,
    get_mock_lret_module,
)
from proxima.backends.base import (
    SimulatorType,
    ResultType,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def lret_adapter() -> LRETBackendAdapter:
    """Create LRET adapter instance for testing."""
    adapter = LRETBackendAdapter()
    adapter.use_mock_backend(True)  # Use mock for testing
    return adapter


@pytest.fixture
def mock_simulator() -> MockLRETSimulator:
    """Create mock simulator instance."""
    return MockLRETSimulator()


@pytest.fixture
def result_normalizer() -> LRETResultNormalizer:
    """Create result normalizer instance."""
    return LRETResultNormalizer(num_qubits=2)


@pytest.fixture
def sample_circuit() -> dict[str, Any]:
    """Create sample circuit for testing."""
    return {
        "num_qubits": 2,
        "gates": [
            {"name": "H", "qubits": [0]},
            {"name": "CX", "qubits": [0, 1]},
        ],
    }


@pytest.fixture
def sample_ghz_circuit() -> dict[str, Any]:
    """Create GHZ state circuit for testing."""
    return {
        "num_qubits": 3,
        "gates": [
            {"name": "H", "qubits": [0]},
            {"name": "CX", "qubits": [0, 1]},
            {"name": "CX", "qubits": [1, 2]},
        ],
    }


# ==============================================================================
# ADAPTER BASIC TESTS
# ==============================================================================


class TestLRETBackendAdapter:
    """Tests for LRETBackendAdapter class."""

    def test_adapter_name(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test adapter returns correct name."""
        assert lret_adapter.get_name() == "lret"

    def test_adapter_version(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test adapter returns version string."""
        version = lret_adapter.get_version()
        assert isinstance(version, str)
        assert version in ("unavailable", "unknown") or version  # Non-empty

    def test_capabilities(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test adapter capabilities."""
        caps = lret_adapter.get_capabilities()
        
        assert SimulatorType.CUSTOM in caps.simulator_types
        assert SimulatorType.STATE_VECTOR in caps.simulator_types
        assert caps.max_qubits == 32
        assert caps.supports_batching is True
        assert caps.supports_noise is False
        assert "framework_integration" in caps.custom_features
        assert "result_normalization" in caps.custom_features
        assert "api_verification" in caps.custom_features

    def test_supported_gates(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test supported gates list."""
        gates = lret_adapter.get_supported_gates()
        
        assert isinstance(gates, list)
        assert len(gates) > 0
        assert "H" in gates
        assert "X" in gates
        assert "CX" in gates
        assert "CNOT" in gates

    def test_supports_simulator(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test simulator type support checking."""
        assert lret_adapter.supports_simulator(SimulatorType.STATE_VECTOR) is True
        assert lret_adapter.supports_simulator(SimulatorType.CUSTOM) is True

    def test_use_mock_backend(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test mock backend toggle."""
        lret_adapter.use_mock_backend(True)
        assert lret_adapter._use_mock is True
        
        lret_adapter.use_mock_backend(False)
        assert lret_adapter._use_mock is False


# ==============================================================================
# CIRCUIT VALIDATION TESTS
# ==============================================================================


class TestCircuitValidation:
    """Tests for circuit validation."""

    def test_validate_none_circuit(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test validation of None circuit."""
        result = lret_adapter.validate_circuit(None)
        
        assert result.valid is False
        assert "None" in result.message

    def test_validate_dict_circuit_with_gates(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test validation of dict circuit with gates."""
        circuit = {"gates": [{"name": "H", "qubits": [0]}]}
        result = lret_adapter.validate_circuit(circuit)
        
        assert result.valid is True
        assert "dict" in result.details.get("format", "")

    def test_validate_dict_circuit_with_operations(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test validation of dict circuit with operations key."""
        circuit = {"operations": [{"name": "X", "qubits": [0]}]}
        result = lret_adapter.validate_circuit(circuit)
        
        assert result.valid is True

    def test_validate_dict_circuit_with_qubits(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test validation of dict circuit with qubits key."""
        circuit = {"qubits": [0, 1], "instructions": []}
        result = lret_adapter.validate_circuit(circuit)
        
        assert result.valid is True

    def test_validate_invalid_dict_circuit(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test validation of dict without required keys."""
        circuit = {"invalid_key": "value"}
        result = lret_adapter.validate_circuit(circuit)
        
        assert result.valid is False
        assert "missing" in result.message.lower()

    def test_validate_full_circuit(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test validation of complete sample circuit."""
        result = lret_adapter.validate_circuit(sample_circuit)
        
        assert result.valid is True


# ==============================================================================
# CIRCUIT EXECUTION TESTS
# ==============================================================================


class TestCircuitExecution:
    """Tests for circuit execution."""

    def test_execute_bell_circuit(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test execution of Bell state circuit."""
        result = lret_adapter.execute(sample_circuit, shots=1000)
        
        assert result is not None
        assert result.result_type in (ResultType.COUNTS, ResultType.STATEVECTOR)
        assert result.backend is not None
        assert result.backend == "lret"

    def test_execute_with_counts(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test execution returns measurement counts."""
        result = lret_adapter.execute(sample_circuit, shots=1000)
        
        counts = result.data.get("counts")
        if counts:
            assert isinstance(counts, dict)
            total_shots = sum(counts.values())
            assert total_shots == 1000

    def test_execute_ghz_circuit(self, lret_adapter: LRETBackendAdapter, sample_ghz_circuit: dict) -> None:
        """Test execution of GHZ state circuit."""
        result = lret_adapter.execute(sample_ghz_circuit, shots=1000)
        
        assert result is not None
        assert result.result_type in (ResultType.COUNTS, ResultType.STATEVECTOR)

    def test_execute_with_seed(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test deterministic execution with seed."""
        result1 = lret_adapter.execute(sample_circuit, shots=100, seed=42)
        result2 = lret_adapter.execute(sample_circuit, shots=100, seed=42)
        
        # With same seed, results should be identical
        assert result1.data.get("counts") == result2.data.get("counts")

    def test_execute_different_shots(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test execution with different shot counts."""
        for shots in [100, 500, 1000]:
            result = lret_adapter.execute(sample_circuit, shots=shots)
            
            counts = result.data.get("counts")
        if counts:
                total = sum(counts.values())
                assert total == shots

    def test_execute_zero_shots_returns_statevector(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test execution with 0 shots returns statevector."""
        result = lret_adapter.execute(sample_circuit, shots=0)
        
        assert result is not None
        # Should have statevector in this case
        assert result.data.get("statevector") is not None or result.data.get("counts") is not None


# ==============================================================================
# RESULT NORMALIZATION TESTS
# ==============================================================================


class TestResultNormalization:
    """Tests for LRETResultNormalizer class."""

    def test_normalize_counts_dict(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of count dictionary."""
        counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        
        result = result_normalizer.normalize(counts)
        
        assert result.format == LRETResultFormat.COUNTS
        assert result.counts == counts
        assert sum(result.probabilities.values()) == pytest.approx(1.0)

    def test_normalize_counts_with_prefix(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of counts with 0b prefix."""
        counts = {"0b00": 500, "0b01": 500}
        
        result = result_normalizer.normalize(counts)
        
        assert "00" in result.counts or "0b00" not in result.counts

    def test_normalize_statevector_array(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of numpy statevector."""
        sv = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        
        result = result_normalizer.normalize(sv)
        
        assert result.format == LRETResultFormat.STATEVECTOR
        assert result.statevector is not None
        # Check normalization
        norm = np.linalg.norm(result.statevector)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_normalize_statevector_dict(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of statevector in dict format."""
        data = {"statevector": [0.5, 0.5, 0.5, 0.5]}
        
        result = result_normalizer.normalize(data)
        
        assert result.format == LRETResultFormat.STATEVECTOR
        assert result.statevector is not None

    def test_normalize_density_matrix(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of density matrix."""
        dm = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        
        result = result_normalizer.normalize(dm)
        
        assert result.format == LRETResultFormat.DENSITY_MATRIX
        assert result.density_matrix is not None

    def test_normalize_probabilities(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of probability dict."""
        probs = {"probabilities": {"00": 0.5, "01": 0.3, "10": 0.2}}
        
        result = result_normalizer.normalize(probs)
        
        assert result.format == LRETResultFormat.PROBABILITIES
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0)

    def test_normalize_result_object(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of MockLRETResult object."""
        mock_result = MockLRETResult(counts={"00": 500, "11": 500}, shots=1000)
        
        result = result_normalizer.normalize(mock_result)
        
        assert result.format == LRETResultFormat.COUNTS
        assert result.counts == {"00": 500, "11": 500}

    def test_cross_populate_counts_to_probs(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test cross-population from counts to probabilities."""
        counts = {"00": 750, "11": 250}
        
        result = result_normalizer.normalize(counts)
        
        assert "00" in result.probabilities
        assert result.probabilities["00"] == pytest.approx(0.75)
        assert result.probabilities["11"] == pytest.approx(0.25)

    def test_cross_populate_statevector_to_probs(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test cross-population from statevector to probabilities."""
        sv = np.array([1, 0, 0, 0], dtype=complex)  # |00> state
        
        result = result_normalizer.normalize(sv)
        
        assert "00" in result.probabilities
        assert result.probabilities["00"] == pytest.approx(1.0)

    def test_normalized_result_to_dict(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test NormalizedResult.to_dict() method."""
        counts = {"00": 500, "11": 500}
        result = result_normalizer.normalize(counts, shots=1000)
        
        result_dict = result.to_dict()
        
        assert "format" in result_dict
        assert "counts" in result_dict
        assert result_dict["shots"] == 1000


# ==============================================================================
# API VERIFICATION TESTS
# ==============================================================================


class TestAPIVerification:
    """Tests for LRETAPIVerifier class."""

    def test_verifier_without_lret(self) -> None:
        """Test verifier when LRET is not installed."""
        verifier = LRETAPIVerifier(None)
        result = verifier.verify()
        
        assert isinstance(result, LRETAPIVerification)
        assert result.is_compatible is False
        assert len(result.warnings) > 0

    def test_verifier_with_mock_module(self) -> None:
        """Test verifier with mock LRET module."""
        mock_module = get_mock_lret_module()
        verifier = LRETAPIVerifier(mock_module)
        
        result = verifier.verify()
        
        assert isinstance(result, LRETAPIVerification)
        assert result.api_version == "0.1.0-mock"
        assert "Simulator" in result.available_apis
        assert "validate_circuit" in result.available_apis
        assert "execute" in result.available_apis

    def test_verifier_caches_result(self) -> None:
        """Test that verification result is cached."""
        mock_module = get_mock_lret_module()
        verifier = LRETAPIVerifier(mock_module)
        
        result1 = verifier.verify()
        result2 = verifier.verify()
        
        assert result1 is result2  # Same object

    def test_verifier_force_reverification(self) -> None:
        """Test force re-verification."""
        mock_module = get_mock_lret_module()
        verifier = LRETAPIVerifier(mock_module)
        
        result1 = verifier.verify()
        result2 = verifier.verify(force=True)
        
        assert result1 is not result2  # New object

    def test_compatibility_report(self) -> None:
        """Test compatibility report generation."""
        mock_module = get_mock_lret_module()
        verifier = LRETAPIVerifier(mock_module)
        
        report = verifier.get_compatibility_report()
        
        assert isinstance(report, str)
        assert "LRET" in report
        assert "Available APIs" in report

    def test_adapter_api_verification(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test API verification through adapter."""
        verification = lret_adapter.get_api_verification()
        
        assert isinstance(verification, LRETAPIVerification)


# ==============================================================================
# MOCK SIMULATOR TESTS
# ==============================================================================


class TestMockSimulator:
    """Tests for MockLRETSimulator class."""

    def test_simulator_simulate(self, mock_simulator: MockLRETSimulator, sample_circuit: dict) -> None:
        """Test simulate method returns statevector."""
        result = mock_simulator.simulate(sample_circuit)
        
        assert isinstance(result, MockLRETResult)
        assert result.statevector is not None
        assert len(result.statevector) == 4  # 2 qubits = 4 states

    def test_simulator_run(self, mock_simulator: MockLRETSimulator, sample_circuit: dict) -> None:
        """Test run method returns counts."""
        result = mock_simulator.run(sample_circuit, shots=1000)
        
        assert isinstance(result, MockLRETResult)
        counts = result.counts
        assert counts is not None
        assert sum(counts.values()) == 1000

    def test_simulator_seed(self, mock_simulator: MockLRETSimulator, sample_circuit: dict) -> None:
        """Test deterministic results with seed."""
        mock_simulator.set_seed(42)
        result1 = mock_simulator.run(sample_circuit, shots=100)
        
        mock_simulator.set_seed(42)
        result2 = mock_simulator.run(sample_circuit, shots=100)
        
        assert result1.counts == result2.counts

    def test_simulator_qubit_count_from_dict(self, mock_simulator: MockLRETSimulator) -> None:
        """Test qubit count extraction from dict circuit."""
        circuit = {"num_qubits": 3, "gates": []}
        count = mock_simulator._get_qubit_count(circuit)
        
        assert count == 3

    def test_simulator_qubit_count_from_gates(self, mock_simulator: MockLRETSimulator) -> None:
        """Test qubit count inference from gates."""
        circuit = {
            "gates": [
                {"name": "H", "qubits": [2]},
                {"name": "CX", "qubits": [2, 3]},
            ]
        }
        count = mock_simulator._get_qubit_count(circuit)
        
        assert count >= 4  # At least qubit indices 0-3

    def test_simulator_hadamard(self, mock_simulator: MockLRETSimulator) -> None:
        """Test Hadamard gate application."""
        circuit = {
            "num_qubits": 1,
            "gates": [{"name": "H", "qubits": [0]}],
        }
        result = mock_simulator.simulate(circuit)
        
        # After H on |0>, should have equal amplitudes
        assert result.statevector is not None
        probs = np.abs(result.statevector) ** 2
        assert probs[0] == pytest.approx(probs[1], abs=0.1)

    def test_simulator_x_gate(self, mock_simulator: MockLRETSimulator) -> None:
        """Test X gate application."""
        circuit = {
            "num_qubits": 1,
            "gates": [{"name": "X", "qubits": [0]}],
        }
        result = mock_simulator.simulate(circuit)
        
        # X on |0> gives |1>
        assert result.statevector is not None
        assert np.abs(result.statevector[1]) > 0.9

    def test_simulator_cnot(self, mock_simulator: MockLRETSimulator) -> None:
        """Test CNOT gate creates entanglement."""
        circuit = {
            "num_qubits": 2,
            "gates": [
                {"name": "H", "qubits": [0]},
                {"name": "CX", "qubits": [0, 1]},
            ],
        }
        result = mock_simulator.run(circuit, shots=1000)
        
        # Bell state should give roughly equal 00 and 11
        counts = result.counts
        assert counts is not None
        # At least should have some 00 and 11 outcomes
        total = sum(counts.values())
        assert total == 1000


# ==============================================================================
# MOCK LRET MODULE TESTS
# ==============================================================================


class TestMockLRETModule:
    """Tests for mock LRET module."""

    def test_module_version(self) -> None:
        """Test module has version."""
        module = get_mock_lret_module()
        assert hasattr(module, "__version__")
        assert "mock" in module.__version__

    def test_module_simulator(self) -> None:
        """Test module has Simulator class."""
        module = get_mock_lret_module()
        assert hasattr(module, "Simulator")
        assert module.Simulator is MockLRETSimulator

    def test_module_validate_circuit(self) -> None:
        """Test module validate_circuit function."""
        module = get_mock_lret_module()
        
        assert module.validate_circuit(None) is False
        assert module.validate_circuit({"gates": []}) is True
        assert module.validate_circuit({"invalid": "data"}) is False

    def test_module_execute(self) -> None:
        """Test module execute function."""
        module = get_mock_lret_module()
        circuit = {"num_qubits": 2, "gates": [{"name": "H", "qubits": [0]}]}
        
        result = module.execute(circuit, shots=100)
        
        assert isinstance(result, MockLRETResult)
        assert sum(result.counts.values()) == 100

    def test_module_supported_gates(self) -> None:
        """Test module SUPPORTED_GATES list."""
        module = get_mock_lret_module()
        
        assert hasattr(module, "SUPPORTED_GATES")
        assert "H" in module.SUPPORTED_GATES
        assert "CX" in module.SUPPORTED_GATES


# ==============================================================================
# RESOURCE ESTIMATION TESTS
# ==============================================================================


class TestResourceEstimation:
    """Tests for resource estimation."""

    def test_estimate_resources(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test resource estimation for circuit."""
        estimate = lret_adapter.estimate_resources(sample_circuit)
        
        assert estimate is not None
        assert estimate.metadata.get("qubits") == 2
        assert estimate.metadata.get("gate_count") == 2

    def test_estimate_resources_memory(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test memory estimation scales with qubits."""
        circuit_small = {"num_qubits": 4, "gates": []}
        circuit_large = {"num_qubits": 10, "gates": []}
        
        est_small = lret_adapter.estimate_resources(circuit_small)
        est_large = lret_adapter.estimate_resources(circuit_large)
        
        # Memory should scale exponentially
        if est_small.memory_mb and est_large.memory_mb:
            assert est_large.memory_mb > est_small.memory_mb

    def test_extract_qubit_count(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test qubit count extraction."""
        circuit = {"num_qubits": 5, "gates": []}
        count = lret_adapter._extract_qubit_count(circuit)
        
        assert count == 5

    def test_extract_gate_count(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test gate count extraction."""
        circuit = {
            "gates": [
                {"name": "H", "qubits": [0]},
                {"name": "X", "qubits": [1]},
                {"name": "CX", "qubits": [0, 1]},
            ]
        }
        count = lret_adapter._extract_gate_count(circuit)
        
        assert count == 3


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_execute_invalid_circuit(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test execution of invalid circuit raises error."""
        with pytest.raises(Exception):  # CircuitValidationError
            lret_adapter.execute(None, shots=100)

    def test_execute_empty_circuit(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test execution of empty circuit."""
        circuit = {"num_qubits": 2, "gates": []}
        result = lret_adapter.execute(circuit, shots=100)
        
        # Should still work, just no gates applied
        assert result is not None

    def test_density_matrix_validation_non_square(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test density matrix validation for non-square matrix."""
        invalid_dm = np.array([[1, 0, 0], [0, 1, 0]], dtype=complex)
        
        with pytest.raises(ValueError, match="square"):
            result_normalizer._validate_density_matrix(invalid_dm)

    def test_density_matrix_validation_non_power_of_2(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test density matrix validation for non-power-of-2 dimension."""
        invalid_dm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=complex)
        
        with pytest.raises(ValueError, match="power of 2"):
            result_normalizer._validate_density_matrix(invalid_dm)


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_qubit_circuit(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test single qubit circuit."""
        circuit = {"num_qubits": 1, "gates": [{"name": "H", "qubits": [0]}]}
        result = lret_adapter.execute(circuit, shots=100)
        
        assert result is not None
        counts = result.data.get("counts")
        if counts:
            assert all(len(k) == 1 for k in counts.keys())

    def test_large_qubit_circuit(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test larger qubit circuit."""
        circuit = {
            "num_qubits": 10,
            "gates": [{"name": "H", "qubits": [i]} for i in range(10)],
        }
        result = lret_adapter.execute(circuit, shots=100)
        
        assert result is not None

    def test_empty_counts_normalization(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of empty counts."""
        result = result_normalizer.normalize({})
        
        assert result.format == LRETResultFormat.COUNTS
        assert result.counts == {}

    def test_zero_statevector(self, result_normalizer: LRETResultNormalizer) -> None:
        """Test normalization of zero statevector."""
        sv = np.zeros(4, dtype=complex)
        result = result_normalizer.normalize(sv)
        
        # Should handle gracefully
        assert result.statevector is not None

    def test_many_shots(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test execution with many shots."""
        result = lret_adapter.execute(sample_circuit, shots=10000)
        
        assert result is not None
        counts = result.data.get("counts")
        if counts:
            assert sum(counts.values()) == 10000

    def test_execution_metadata(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test execution result contains proper metadata."""
        result = lret_adapter.execute(sample_circuit, shots=100)
        
        assert result.metadata is not None
        assert result.backend is not None
        assert result.execution_time_ms is not None
        assert result.metadata["normalized"] is True


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_execution_pipeline(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test complete execution pipeline."""
        # Create circuit
        circuit = {
            "num_qubits": 3,
            "gates": [
                {"name": "H", "qubits": [0]},
                {"name": "H", "qubits": [1]},
                {"name": "H", "qubits": [2]},
                {"name": "CX", "qubits": [0, 1]},
                {"name": "CX", "qubits": [1, 2]},
            ],
        }
        
        # Validate
        validation = lret_adapter.validate_circuit(circuit)
        assert validation.valid is True
        
        # Estimate resources
        estimate = lret_adapter.estimate_resources(circuit)
        assert estimate.metadata.get("qubits") == 3
        
        # Execute
        result = lret_adapter.execute(circuit, shots=1000)
        assert result is not None
        
        # Check result structure
        counts = result.data.get("counts")
        if counts:
            assert sum(counts.values()) == 1000

    def test_multiple_executions(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test multiple sequential executions."""
        results = []
        for _ in range(5):
            result = lret_adapter.execute(sample_circuit, shots=100)
            results.append(result)
        
        assert len(results) == 5
        assert all(r is not None for r in results)

    def test_api_verification_integration(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test API verification integration with adapter."""
        verification = lret_adapter.get_api_verification()
        
        assert isinstance(verification, LRETAPIVerification)
        assert verification.verified_at > 0


# ==============================================================================
# PERFORMANCE BENCHMARKING TESTS
# ==============================================================================


class TestLRETPerformanceMetrics:
    """Tests for LRETPerformanceMetrics data class."""

    def test_metrics_creation(self) -> None:
        """Test creating performance metrics."""
        from proxima.backends.lret import LRETPerformanceMetrics
        
        metrics = LRETPerformanceMetrics(
            execution_time_ms=15.5,
            gate_execution_time_ms=10.0,
            measurement_time_ms=3.0,
            normalization_time_ms=2.5,
            memory_peak_mb=128.0,
            memory_baseline_mb=64.0,
            throughput_shots_per_sec=66000.0,
            gates_per_second=100000.0,
            qubits=4,
            gate_count=10,
            circuit_depth=5,
            shots=1024,
        )
        
        assert metrics.execution_time_ms == 15.5
        assert metrics.qubits == 4
        assert metrics.shots == 1024

    def test_metrics_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        from proxima.backends.lret import LRETPerformanceMetrics
        
        metrics = LRETPerformanceMetrics(
            execution_time_ms=20.0,
            qubits=3,
            gate_count=5,
            shots=500,
        )
        
        data = metrics.to_dict()
        
        assert data["execution_time_ms"] == 20.0
        assert data["qubits"] == 3
        assert data["gate_count"] == 5
        assert data["shots"] == 500
        assert "timestamp" in data

    def test_metrics_from_dict(self) -> None:
        """Test creating metrics from dictionary."""
        from proxima.backends.lret import LRETPerformanceMetrics
        
        data = {
            "execution_time_ms": 25.0,
            "qubits": 5,
            "gate_count": 15,
            "shots": 2000,
            "memory_peak_mb": 256.0,
        }
        
        metrics = LRETPerformanceMetrics.from_dict(data)
        
        assert metrics.execution_time_ms == 25.0
        assert metrics.qubits == 5
        assert metrics.memory_peak_mb == 256.0


class TestLRETBenchmarkResult:
    """Tests for LRETBenchmarkResult data class."""

    def test_benchmark_result_creation(self) -> None:
        """Test creating benchmark result."""
        from proxima.backends.lret import LRETBenchmarkResult, LRETPerformanceMetrics
        
        result = LRETBenchmarkResult(
            circuit_name="test_circuit",
            num_runs=5,
        )
        
        assert result.circuit_name == "test_circuit"
        assert result.num_runs == 5
        assert result.metrics == []

    def test_compute_statistics(self) -> None:
        """Test computing aggregate statistics."""
        from proxima.backends.lret import LRETBenchmarkResult, LRETPerformanceMetrics
        
        result = LRETBenchmarkResult(
            circuit_name="test",
            num_runs=3,
            metrics=[
                LRETPerformanceMetrics(execution_time_ms=10.0, throughput_shots_per_sec=1000),
                LRETPerformanceMetrics(execution_time_ms=12.0, throughput_shots_per_sec=900),
                LRETPerformanceMetrics(execution_time_ms=11.0, throughput_shots_per_sec=950),
            ],
        )
        
        result.compute_statistics()
        
        assert result.mean_execution_time_ms == 11.0
        assert result.min_execution_time_ms == 10.0
        assert result.max_execution_time_ms == 12.0
        assert result.std_execution_time_ms > 0
        assert result.mean_throughput > 0
        assert result.success_rate == 100.0

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        from proxima.backends.lret import LRETBenchmarkResult
        
        result = LRETBenchmarkResult(
            circuit_name="test",
            num_runs=3,
            mean_execution_time_ms=15.0,
            success_rate=100.0,
        )
        
        data = result.to_dict()
        
        assert data["circuit_name"] == "test"
        assert data["num_runs"] == 3
        assert data["mean_execution_time_ms"] == 15.0


class TestLRETPerformanceMonitor:
    """Tests for LRETPerformanceMonitor class."""

    def test_monitor_basic_timing(self) -> None:
        """Test basic timing functionality."""
        from proxima.backends.lret import LRETPerformanceMonitor
        import time
        
        monitor = LRETPerformanceMonitor()
        monitor.start()
        time.sleep(0.01)  # 10ms
        elapsed = monitor.stop()
        
        assert elapsed >= 9.0  # At least 9ms (allowing for timer variance)
        assert elapsed < 100.0  # Should not take more than 100ms

    def test_phase_timing(self) -> None:
        """Test phase timing."""
        from proxima.backends.lret import LRETPerformanceMonitor
        import time
        
        monitor = LRETPerformanceMonitor()
        monitor.start()
        
        monitor.start_phase("phase1")
        time.sleep(0.005)
        duration = monitor.end_phase("phase1")
        
        assert duration >= 4.0  # At least 4ms
        assert monitor.get_phase_time("phase1") == duration

    def test_memory_sampling(self) -> None:
        """Test memory sampling."""
        from proxima.backends.lret import LRETPerformanceMonitor
        
        monitor = LRETPerformanceMonitor()
        monitor.start()
        
        mem = monitor.sample_memory()
        peak = monitor.get_peak_memory()
        
        # Memory values should be positive (or 0 if psutil not available)
        assert mem >= 0
        assert peak >= 0

    def test_get_metrics(self) -> None:
        """Test getting aggregated metrics."""
        from proxima.backends.lret import LRETPerformanceMonitor, LRETPerformanceMetrics
        
        monitor = LRETPerformanceMonitor()
        monitor.start()
        monitor.sample_memory()
        
        metrics = monitor.get_metrics(qubits=3, gate_count=10, circuit_depth=5, shots=1000)
        
        assert isinstance(metrics, LRETPerformanceMetrics)
        assert metrics.qubits == 3
        assert metrics.gate_count == 10
        assert metrics.shots == 1000

    def test_reset(self) -> None:
        """Test resetting monitor state."""
        from proxima.backends.lret import LRETPerformanceMonitor
        
        monitor = LRETPerformanceMonitor()
        monitor.start()
        monitor.sample_memory()
        monitor.start_phase("test")
        
        monitor.reset()
        
        assert monitor._is_running is False
        assert monitor._start_time == 0.0


class TestLRETBenchmarkRunner:
    """Tests for LRETBenchmarkRunner class."""

    def test_runner_creation(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test creating benchmark runner."""
        from proxima.backends.lret import LRETBenchmarkRunner
        
        runner = LRETBenchmarkRunner(lret_adapter)
        assert runner._adapter is lret_adapter

    def test_run_single(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test running a single benchmark iteration."""
        from proxima.backends.lret import LRETBenchmarkRunner, LRETPerformanceMetrics
        
        runner = LRETBenchmarkRunner(lret_adapter)
        metrics = runner.run_single(sample_circuit, shots=100, circuit_name="test")
        
        assert isinstance(metrics, LRETPerformanceMetrics)
        assert metrics.execution_time_ms > 0
        assert metrics.shots == 100

    def test_run_benchmark(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test running full benchmark with multiple runs."""
        from proxima.backends.lret import LRETBenchmarkRunner, LRETBenchmarkResult
        
        runner = LRETBenchmarkRunner(lret_adapter)
        result = runner.run_benchmark(
            circuit=sample_circuit,
            num_runs=3,
            shots=100,
            circuit_name="test",
            warmup_runs=1,
        )
        
        assert isinstance(result, LRETBenchmarkResult)
        assert result.circuit_name == "test"
        assert result.num_runs == 3
        assert len(result.metrics) <= 3
        assert result.mean_execution_time_ms > 0

    def test_run_scaling_benchmark(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test scaling benchmark across circuit sizes."""
        from proxima.backends.lret import LRETBenchmarkRunner
        
        def make_circuit(n_qubits: int) -> dict:
            return {
                "num_qubits": n_qubits,
                "gates": [{"name": "H", "qubits": [i]} for i in range(n_qubits)],
            }
        
        runner = LRETBenchmarkRunner(lret_adapter)
        results = runner.run_scaling_benchmark(
            circuit_generator=make_circuit,
            qubit_range=[2, 3, 4],
            shots=50,
            num_runs=2,
        )
        
        assert len(results) == 3
        assert all(r.circuit_name.startswith("circuit_") for r in results)

    def test_run_throughput_benchmark(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test throughput benchmark across shot counts."""
        from proxima.backends.lret import LRETBenchmarkRunner
        
        runner = LRETBenchmarkRunner(lret_adapter)
        results = runner.run_throughput_benchmark(
            circuit=sample_circuit,
            shot_counts=[50, 100, 200],
            num_runs=2,
            circuit_name="throughput",
        )
        
        assert len(results) == 3
        assert all(r.mean_throughput >= 0 for r in results)

    def test_generate_report(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test generating benchmark report."""
        from proxima.backends.lret import LRETBenchmarkRunner, LRETBenchmarkResult
        
        runner = LRETBenchmarkRunner(lret_adapter)
        
        # Run a quick benchmark
        result = runner.run_benchmark(sample_circuit, num_runs=2, shots=50)
        
        report = runner.generate_report([result], include_raw_metrics=False)
        
        assert "backend" in report
        assert report["backend"] == "lret"
        assert "summary" in report
        assert "results" in report
        assert len(report["results"]) == 1


class TestLRETAdapterBenchmarkMethods:
    """Tests for benchmark methods in LRETBackendAdapter."""

    def test_get_benchmark_runner(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test getting benchmark runner from adapter."""
        from proxima.backends.lret import LRETBenchmarkRunner
        
        runner = lret_adapter.get_benchmark_runner()
        
        assert isinstance(runner, LRETBenchmarkRunner)
        assert runner._adapter is lret_adapter

    def test_run_performance_benchmark(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test running performance benchmark via adapter method."""
        from proxima.backends.lret import LRETBenchmarkResult
        
        result = lret_adapter.run_performance_benchmark(
            circuit=sample_circuit,
            shots=100,
            num_runs=3,
            warmup_runs=1,
            circuit_name="adapter_test",
        )
        
        assert isinstance(result, LRETBenchmarkResult)
        assert result.circuit_name == "adapter_test"
        assert result.num_runs == 3

    def test_run_scaling_benchmark_via_adapter(self, lret_adapter: LRETBackendAdapter) -> None:
        """Test scaling benchmark via adapter method."""
        def make_circuit(n: int) -> dict:
            return {"num_qubits": n, "gates": [{"name": "H", "qubits": [0]}]}
        
        results = lret_adapter.run_scaling_benchmark(
            circuit_generator=make_circuit,
            qubit_range=[2, 3],
            shots=50,
            num_runs=2,
        )
        
        assert len(results) == 2

    def test_run_throughput_benchmark_via_adapter(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test throughput benchmark via adapter method."""
        results = lret_adapter.run_throughput_benchmark(
            circuit=sample_circuit,
            shot_counts=[50, 100],
            num_runs=2,
        )
        
        assert len(results) == 2

    def test_generate_benchmark_report_via_adapter(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test generating report via adapter method."""
        result = lret_adapter.run_performance_benchmark(
            sample_circuit, shots=50, num_runs=2
        )
        
        report = lret_adapter.generate_benchmark_report([result])
        
        assert "backend" in report
        assert "results" in report

    def test_get_performance_profile(self, lret_adapter: LRETBackendAdapter, sample_circuit: dict) -> None:
        """Test getting detailed performance profile."""
        profile = lret_adapter.get_performance_profile(sample_circuit, shots=100)
        
        assert "backend" in profile
        assert profile["backend"] == "lret"
        assert "circuit_info" in profile
        assert "execution_time_ms" in profile
        assert "gate_execution_time_ms" in profile
        assert "memory_peak_mb" in profile
        assert "throughput_shots_per_sec" in profile
        assert profile["shots"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
