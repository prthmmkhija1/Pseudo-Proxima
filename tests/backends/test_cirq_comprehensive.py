"""Comprehensive tests for Cirq Backend Adapter.

Tests all features including:
- Noise model integration verification
- DensityMatrix mode comprehensive testing
- Batch execution support
- Performance optimization for large circuits
"""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import adapter and supporting classes
from proxima.backends.cirq_adapter import (
    BatchExecutionConfig,
    BatchExecutionResult,
    BatchExecutor,
    CircuitOptimizer,
    CirqBackendAdapter,
    DensityMatrixTestResult,
    DensityMatrixTester,
    NoiseModelVerification,
    NoiseModelVerifier,
    NoiseType,
    PerformanceConfig,
    PerformanceMonitor,
)
from proxima.backends.base import (
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
    ValidationResult,
)
from proxima.backends.exceptions import (
    BackendNotInstalledError,
    CircuitValidationError,
    QubitLimitExceededError,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def adapter() -> CirqBackendAdapter:
    """Create a CirqBackendAdapter instance."""
    return CirqBackendAdapter()


@pytest.fixture
def cirq_available() -> bool:
    """Check if Cirq is available."""
    try:
        import cirq
        return True
    except ImportError:
        return False


@pytest.fixture
def simple_circuit():
    """Create a simple 2-qubit Bell state circuit."""
    try:
        import cirq
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="result"),
        ])
        return circuit
    except ImportError:
        pytest.skip("Cirq not installed")


@pytest.fixture
def parameterized_circuit():
    """Create a parameterized variational circuit."""
    try:
        import cirq
        import sympy
        
        q0 = cirq.LineQubit(0)
        theta = sympy.Symbol("theta")
        phi = sympy.Symbol("phi")
        
        circuit = cirq.Circuit([
            cirq.rx(theta).on(q0),
            cirq.rz(phi).on(q0),
        ])
        return circuit, {"theta": np.pi / 4, "phi": np.pi / 2}
    except ImportError:
        pytest.skip("Cirq not installed")


@pytest.fixture
def large_circuit():
    """Create a larger circuit for performance testing."""
    try:
        import cirq
        
        n_qubits = 10
        qubits = cirq.LineQubit.range(n_qubits)
        
        moments = []
        for _ in range(20):  # 20 layers
            # Single-qubit layer
            moments.append(cirq.Moment([cirq.H(q) for q in qubits]))
            # Two-qubit layer
            for i in range(0, n_qubits - 1, 2):
                moments.append(cirq.Moment([cirq.CNOT(qubits[i], qubits[i + 1])]))
        
        circuit = cirq.Circuit(moments)
        return circuit
    except ImportError:
        pytest.skip("Cirq not installed")


@pytest.fixture
def mock_cirq():
    """Create a mock Cirq module for unit testing."""
    mock = MagicMock()
    mock.__version__ = "1.2.3"
    mock.Circuit = MagicMock
    mock.Simulator = MagicMock
    mock.DensityMatrixSimulator = MagicMock
    return mock


# ==============================================================================
# BASIC ADAPTER TESTS
# ==============================================================================


class TestCirqBackendAdapter:
    """Test basic CirqBackendAdapter functionality."""

    def test_get_name(self, adapter: CirqBackendAdapter) -> None:
        """Test get_name returns 'cirq'."""
        assert adapter.get_name() == "cirq"

    def test_get_version(self, adapter: CirqBackendAdapter, cirq_available: bool) -> None:
        """Test get_version returns a version string."""
        version = adapter.get_version()
        if cirq_available:
            assert isinstance(version, str)
            assert version != "unavailable"
        else:
            assert version in ("unknown", "unavailable")

    def test_is_available(self, adapter: CirqBackendAdapter, cirq_available: bool) -> None:
        """Test is_available matches actual Cirq availability."""
        assert adapter.is_available() == cirq_available

    def test_get_capabilities(self, adapter: CirqBackendAdapter) -> None:
        """Test get_capabilities returns proper Capabilities object."""
        caps = adapter.get_capabilities()
        
        assert isinstance(caps, Capabilities)
        assert SimulatorType.STATE_VECTOR in caps.simulator_types
        assert SimulatorType.DENSITY_MATRIX in caps.simulator_types
        assert caps.max_qubits == 30
        assert caps.supports_noise is True
        assert caps.supports_batching is True
        
        # Check custom features
        assert caps.custom_features.get("batch_execution") is True
        assert caps.custom_features.get("noise_verification") is True
        assert caps.custom_features.get("density_matrix_testing") is True
        assert caps.custom_features.get("performance_optimization") is True

    def test_supports_simulator(self, adapter: CirqBackendAdapter) -> None:
        """Test supports_simulator method."""
        assert adapter.supports_simulator(SimulatorType.STATE_VECTOR) is True
        assert adapter.supports_simulator(SimulatorType.DENSITY_MATRIX) is True
        assert adapter.supports_simulator(SimulatorType.TENSOR_NETWORK) is False

    def test_get_supported_gates(self, adapter: CirqBackendAdapter) -> None:
        """Test get_supported_gates returns a list of gates."""
        gates = adapter.get_supported_gates()
        
        assert isinstance(gates, list)
        assert "H" in gates
        assert "X" in gates
        assert "CNOT" in gates
        assert "CZ" in gates


# ==============================================================================
# NOISE MODEL VERIFICATION TESTS
# ==============================================================================


class TestNoiseModelVerifier:
    """Test NoiseModelVerifier functionality."""

    def test_verifier_initialization(self) -> None:
        """Test NoiseModelVerifier can be initialized."""
        verifier = NoiseModelVerifier()
        assert verifier is not None

    def test_verify_none_model(self) -> None:
        """Test verification of None noise model."""
        verifier = NoiseModelVerifier()
        result = verifier.verify(None)
        
        assert isinstance(result, NoiseModelVerification)
        assert result.is_valid is False
        assert "None" in result.errors[0]

    def test_verify_probability_valid(self) -> None:
        """Test probability verification with valid values."""
        verifier = NoiseModelVerifier()
        
        is_valid, msg = verifier.verify_probability(NoiseType.DEPOLARIZING, 0.01)
        assert is_valid is True
        
        is_valid, msg = verifier.verify_probability(NoiseType.BIT_FLIP, 0.5)
        assert is_valid is True

    def test_verify_probability_invalid(self) -> None:
        """Test probability verification with invalid values."""
        verifier = NoiseModelVerifier()
        
        is_valid, msg = verifier.verify_probability(NoiseType.DEPOLARIZING, -0.1)
        assert is_valid is False
        assert "below minimum" in msg
        
        is_valid, msg = verifier.verify_probability(NoiseType.DEPOLARIZING, 0.8)
        assert is_valid is False
        assert "exceeds maximum" in msg

    def test_get_channel_description(self) -> None:
        """Test getting channel descriptions."""
        verifier = NoiseModelVerifier()
        
        desc = verifier.get_channel_description(None)
        assert desc == "No noise model"

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_verify_actual_noise_model(self, adapter: CirqBackendAdapter) -> None:
        """Test verification of actual Cirq noise model."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        noise_model = adapter.create_noise_model("depolarizing", 0.01)
        verification = adapter.verify_noise_model(noise_model, 0.01)
        
        assert verification.is_valid is True
        assert verification.error_probability == 0.01


class TestNoiseModelIntegration:
    """Integration tests for noise model features."""

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_create_noise_models(self, adapter: CirqBackendAdapter) -> None:
        """Test creation of various noise models."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        # Test all supported noise types
        noise_types = ["depolarizing", "bit_flip", "amplitude_damping", "phase_damping"]
        
        for noise_type in noise_types:
            noise_model = adapter.create_noise_model(noise_type, 0.01)
            assert noise_model is not None
            
            # Verify each noise model
            verification = adapter.verify_noise_model(noise_model)
            assert verification.is_valid is True

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_noisy_execution(
        self,
        adapter: CirqBackendAdapter,
        simple_circuit,
    ) -> None:
        """Test circuit execution with noise model."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        noise_model = adapter.create_noise_model("depolarizing", 0.01)
        
        result = adapter.execute(
            simple_circuit,
            {"shots": 100, "noise_model": noise_model},
        )
        
        assert result is not None
        assert result.result_type == ResultType.COUNTS
        assert result.metadata.get("noisy") is True


# ==============================================================================
# DENSITY MATRIX TESTING TESTS
# ==============================================================================


class TestDensityMatrixTester:
    """Test DensityMatrixTester functionality."""

    def test_tester_initialization(self) -> None:
        """Test DensityMatrixTester can be initialized."""
        tester = DensityMatrixTester()
        assert tester is not None

    def test_pure_state_validation(self) -> None:
        """Test validation of a pure state density matrix."""
        tester = DensityMatrixTester()
        
        # |0><0| state
        dm = np.array([[1, 0], [0, 0]], dtype=complex)
        results = tester.run_all_tests(dm)
        
        assert len(results) == 5
        assert all(r.passed for r in results)

    def test_maximally_mixed_state(self) -> None:
        """Test validation of maximally mixed state."""
        tester = DensityMatrixTester()
        
        # Maximally mixed 2-qubit state
        dm = np.eye(4, dtype=complex) / 4
        results = tester.run_all_tests(dm)
        
        assert all(r.passed for r in results)

    def test_trace_preservation(self) -> None:
        """Test trace preservation test."""
        tester = DensityMatrixTester()
        
        dm = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        result = tester.test_trace_preservation(dm, expected_trace=1.0)
        
        assert result.passed == True
        assert abs(result.actual - 1.0) < 1e-10

    def test_hermiticity(self) -> None:
        """Test Hermiticity test."""
        tester = DensityMatrixTester()
        
        # Hermitian matrix
        dm = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex)
        result = tester.test_hermiticity(dm)
        assert result.passed == True
        
        # Non-Hermitian matrix
        dm_bad = np.array([[0.5, 0.5], [0.3, 0.5]], dtype=complex)
        result_bad = tester.test_hermiticity(dm_bad)
        assert result_bad.passed == False

    def test_positivity(self) -> None:
        """Test positivity test."""
        tester = DensityMatrixTester()
        
        # Valid density matrix
        dm = np.array([[0.7, 0.3], [0.3, 0.3]], dtype=complex)
        result = tester.test_positivity(dm)
        assert result.passed == True

    def test_purity(self) -> None:
        """Test purity calculation."""
        tester = DensityMatrixTester()
        
        # Pure state (purity = 1)
        dm_pure = np.array([[1, 0], [0, 0]], dtype=complex)
        result_pure = tester.test_purity(dm_pure)
        assert abs(result_pure.actual - 1.0) < 1e-10
        
        # Maximally mixed (purity = 0.5 for 2-dim)
        dm_mixed = np.eye(2, dtype=complex) / 2
        result_mixed = tester.test_purity(dm_mixed)
        assert abs(result_mixed.actual - 0.5) < 1e-10

    def test_generate_report(self) -> None:
        """Test report generation."""
        tester = DensityMatrixTester()
        
        dm = np.array([[1, 0], [0, 0]], dtype=complex)
        tester.run_all_tests(dm)
        
        report = tester.generate_report()
        assert "Density Matrix Validation Report" in report
        assert "PASS" in report


class TestDensityMatrixIntegration:
    """Integration tests for density matrix features."""

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_density_matrix_execution(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test density matrix simulation mode."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.H(q)])
        
        result = adapter.execute(
            circuit,
            {"simulator_type": SimulatorType.DENSITY_MATRIX},
        )
        
        assert result.result_type == ResultType.DENSITY_MATRIX
        assert "density_matrix" in result.data
        
        dm = result.data["density_matrix"]
        
        # Validate the density matrix
        test_results = adapter.validate_density_matrix(dm)
        assert all(r.passed for r in test_results)

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_noisy_density_matrix(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test density matrix with noise model."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.H(q)])
        
        noise_model = adapter.create_noise_model("depolarizing", 0.1)
        
        result = adapter.execute(
            circuit,
            {"noise_model": noise_model},
        )
        
        assert result.result_type == ResultType.DENSITY_MATRIX
        
        dm = result.data["density_matrix"]
        
        # Should still be a valid density matrix
        test_results = adapter.validate_density_matrix(dm)
        assert all(r.passed for r in test_results)
        
        # But purity should be less than 1 due to noise
        tester = adapter.get_density_matrix_tester()
        purity_result = tester.test_purity(dm)
        assert purity_result.actual < 1.0


# ==============================================================================
# BATCH EXECUTION TESTS
# ==============================================================================


class TestBatchExecutor:
    """Test BatchExecutor functionality."""

    def test_batch_config_defaults(self) -> None:
        """Test BatchExecutionConfig default values."""
        config = BatchExecutionConfig()
        
        assert config.max_batch_size == 100
        assert config.parallelize is False
        assert config.continue_on_error is True

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_batch_execution(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test batch execution of multiple circuits."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        
        # Create multiple circuits with measurements
        circuits = []
        for i in range(5):
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit([
                cirq.rx(np.pi / (i + 1)).on(q),
                cirq.measure(q, key='m'),
            ])
            circuits.append(circuit)
        
        result = adapter.execute_batch(circuits, {"shots": 100})
        
        assert isinstance(result, BatchExecutionResult)
        assert result.total_circuits == 5
        assert result.successful == 5
        assert result.failed == 0
        assert len(result.results) == 5

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_parameter_sweep(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test parameter sweep execution."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        import sympy
        
        q = cirq.LineQubit(0)
        theta = sympy.Symbol("theta")
        circuit = cirq.Circuit([cirq.rx(theta).on(q)])
        
        param_sets = [{"theta": np.pi * i / 4} for i in range(5)]
        
        result = adapter.execute_parameter_sweep(
            circuit,
            param_sets,
            {"shots": 100},
        )
        
        assert result.total_circuits == 5
        assert result.successful == 5

    def test_batch_with_errors(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test batch execution handles errors gracefully."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        # Create batch with invalid circuit
        circuits = [None, None, None]  # All invalid
        
        config = BatchExecutionConfig(continue_on_error=True)
        result = adapter.execute_batch(circuits, {}, config)
        
        assert result.failed == 3
        assert result.successful == 0
        assert len(result.errors) == 3

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_batch_timing(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test batch execution timing metrics."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        
        circuits = []
        for _ in range(10):
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.H(q)])
            circuits.append(circuit)
        
        result = adapter.execute_batch(circuits, {"shots": 10})
        
        assert result.total_execution_time_ms > 0
        assert result.average_time_per_circuit_ms > 0
        assert result.average_time_per_circuit_ms < result.total_execution_time_ms


# ==============================================================================
# PERFORMANCE OPTIMIZATION TESTS
# ==============================================================================


class TestCircuitOptimizer:
    """Test CircuitOptimizer functionality."""

    def test_optimizer_initialization(self) -> None:
        """Test CircuitOptimizer can be initialized."""
        optimizer = CircuitOptimizer()
        assert optimizer is not None

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_complexity_estimation(
        self,
        adapter: CirqBackendAdapter,
        large_circuit,
    ) -> None:
        """Test circuit complexity estimation."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        complexity = adapter.estimate_circuit_complexity(large_circuit)
        
        assert "num_qubits" in complexity
        assert "num_moments" in complexity
        assert "total_gates" in complexity
        assert "two_qubit_gates" in complexity
        assert "statevector_memory_mb" in complexity
        assert "is_large_circuit" in complexity
        assert "recommended_optimization_level" in complexity

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_circuit_optimization_levels(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test different optimization levels."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        
        # Create circuit with empty moments
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([
            cirq.Moment([]),  # Empty moment
            cirq.H(q),
            cirq.Moment([]),  # Empty moment
            cirq.X(q),
        ])
        
        original_depth = len(circuit)
        
        # Level 0: No optimization
        opt0 = adapter.optimize_circuit(circuit, level=0)
        assert len(opt0) == original_depth
        
        # Level 1: Basic optimization
        opt1 = adapter.optimize_circuit(circuit, level=1)
        assert len(opt1) <= original_depth


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""

    def test_monitor_initialization(self) -> None:
        """Test PerformanceMonitor can be initialized."""
        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_record_execution(self) -> None:
        """Test recording execution metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record_execution(10.0, 5)
        monitor.record_execution(15.0, 5)
        monitor.record_execution(12.0, 5)
        
        stats = monitor.get_statistics()
        
        assert stats["total_executions"] == 3
        assert abs(stats["mean_execution_time_ms"] - 12.33) < 0.1
        assert stats["min_execution_time_ms"] == 10.0
        assert stats["max_execution_time_ms"] == 15.0
        assert stats["total_time_ms"] == 37.0

    def test_reset(self) -> None:
        """Test resetting monitor."""
        monitor = PerformanceMonitor()
        
        monitor.record_execution(10.0, 5)
        monitor.reset()
        
        stats = monitor.get_statistics()
        assert "error" in stats

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_adapter_monitors_performance(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test that adapter monitors execution performance."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        
        # Reset monitor
        adapter.get_performance_monitor().reset()
        
        # Execute several circuits
        for _ in range(5):
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit([cirq.H(q)])
            adapter.execute(circuit, {"shots": 10})
        
        stats = adapter.get_performance_monitor().get_statistics()
        assert stats["total_executions"] == 5


class TestPerformanceConfig:
    """Test PerformanceConfig functionality."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        config = PerformanceConfig()
        
        assert config.enable_caching is True
        assert config.cache_size == 100
        assert config.lazy_evaluation is True
        assert config.large_circuit_threshold == 20
        assert config.auto_optimize is True

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_auto_optimization(
        self,
        adapter: CirqBackendAdapter,
        large_circuit,
    ) -> None:
        """Test automatic optimization for large circuits."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        # Enable auto-optimization
        config = PerformanceConfig(auto_optimize=True)
        adapter.set_performance_config(config)
        
        # Execute large circuit
        result = adapter.execute(large_circuit, {"shots": 10})
        
        # Should have been optimized
        assert result.metadata.get("optimized") is True


# ==============================================================================
# CIRCUIT VALIDATION TESTS
# ==============================================================================


class TestCircuitValidation:
    """Test circuit validation functionality."""

    def test_validate_circuit_not_installed(self) -> None:
        """Test validation when Cirq is not installed."""
        adapter = CirqBackendAdapter()
        
        with patch.object(adapter, "is_available", return_value=False):
            result = adapter.validate_circuit("not a circuit")
        
        assert result.valid is False
        assert "not installed" in result.message

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_validate_invalid_circuit(self, adapter: CirqBackendAdapter) -> None:
        """Test validation of invalid circuit."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        result = adapter.validate_circuit("not a circuit")
        assert result.valid is False

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_validate_valid_circuit(
        self,
        adapter: CirqBackendAdapter,
        simple_circuit,
    ) -> None:
        """Test validation of valid circuit."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        result = adapter.validate_circuit(simple_circuit)
        assert result.valid is True


# ==============================================================================
# RESOURCE ESTIMATION TESTS
# ==============================================================================


class TestResourceEstimation:
    """Test resource estimation functionality."""

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_estimate_resources(
        self,
        adapter: CirqBackendAdapter,
        simple_circuit,
    ) -> None:
        """Test resource estimation."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        estimate = adapter.estimate_resources(simple_circuit)
        
        assert isinstance(estimate, ResourceEstimate)
        assert estimate.memory_mb is not None
        assert estimate.time_ms is not None
        assert "qubits" in estimate.metadata
        assert "gate_count" in estimate.metadata
        assert "depth" in estimate.metadata
        assert "two_qubit_gates" in estimate.metadata


# ==============================================================================
# EXECUTION TESTS
# ==============================================================================


class TestExecution:
    """Test circuit execution functionality."""

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_statevector_execution(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test statevector simulation."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.H(q)])
        
        result = adapter.execute(circuit)
        
        assert result.result_type == ResultType.STATEVECTOR
        assert "statevector" in result.data

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_sampling_execution(
        self,
        adapter: CirqBackendAdapter,
        simple_circuit,
    ) -> None:
        """Test sampling execution."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        result = adapter.execute(simple_circuit, {"shots": 100})
        
        assert result.result_type == ResultType.COUNTS
        assert "counts" in result.data
        assert sum(result.data["counts"].values()) == 100

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_parameterized_execution(
        self,
        adapter: CirqBackendAdapter,
        parameterized_circuit,
    ) -> None:
        """Test parameterized circuit execution."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        circuit, params = parameterized_circuit
        
        result = adapter.execute(circuit, {"params": params, "shots": 100})
        
        assert result is not None
        assert result.metadata.get("parameterized") is True


# ==============================================================================
# EXPECTATION VALUE TESTS
# ==============================================================================


class TestExpectationValues:
    """Test expectation value computation."""

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_compute_expectation(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test expectation value computation."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.H(q)])
        
        # Expectation of Z should be ~0 for |+> state
        exp_z = adapter.compute_expectation(circuit, cirq.Z(q))
        assert abs(exp_z) < 1e-10
        
        # Expectation of X should be 1 for |+> state
        exp_x = adapter.compute_expectation(circuit, cirq.X(q))
        assert abs(exp_x - 1.0) < 1e-10


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_backend_not_installed_error(self) -> None:
        """Test BackendNotInstalledError is raised appropriately."""
        adapter = CirqBackendAdapter()
        
        with patch.object(adapter, "is_available", return_value=False):
            with pytest.raises(BackendNotInstalledError):
                adapter.execute("circuit", {})

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_circuit_validation_error(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test CircuitValidationError is raised for invalid circuits."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        with pytest.raises(CircuitValidationError):
            adapter.execute("not a circuit", {})

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_unsupported_simulator_type(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test ValueError for unsupported simulator type."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit([cirq.H(q)])
        
        with pytest.raises(ValueError):
            adapter.execute(circuit, {"simulator_type": SimulatorType.TENSOR_NETWORK})

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_invalid_noise_type(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test ValueError for invalid noise type."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        with pytest.raises(ValueError):
            adapter.create_noise_model("invalid_noise", 0.01)


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_full_workflow(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test a complete workflow using multiple features."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        import sympy
        
        # 1. Create parameterized circuit
        q0, q1 = cirq.LineQubit.range(2)
        theta = sympy.Symbol("theta")
        
        circuit = cirq.Circuit([
            cirq.H(q0),
            cirq.rx(theta).on(q1),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="m"),
        ])
        
        # 2. Validate circuit
        validation = adapter.validate_circuit(circuit)
        assert validation.valid is True
        
        # 3. Estimate resources
        resources = adapter.estimate_resources(circuit)
        assert resources.memory_mb is not None
        
        # 4. Create noise model and verify it
        noise_model = adapter.create_noise_model("depolarizing", 0.01)
        verification = adapter.verify_noise_model(noise_model)
        assert verification.is_valid is True
        
        # 5. Execute parameter sweep
        param_sets = [{"theta": np.pi * i / 4} for i in range(4)]
        batch_result = adapter.execute_parameter_sweep(
            circuit,
            param_sets,
            {"shots": 100, "noise_model": noise_model},
        )
        
        assert batch_result.successful == 4
        
        # 6. Check performance metrics
        stats = adapter.get_performance_monitor().get_statistics()
        assert stats["total_executions"] > 0

    @pytest.mark.skipif(
        not pytest.importorskip("cirq", reason="Cirq not installed"),
        reason="Cirq required",
    )
    def test_vqe_like_workflow(
        self,
        adapter: CirqBackendAdapter,
    ) -> None:
        """Test a VQE-like variational algorithm workflow."""
        if not adapter.is_available():
            pytest.skip("Cirq not installed")
        
        import cirq
        import sympy
        
        # Create variational ansatz
        q = cirq.LineQubit(0)
        theta = sympy.Symbol("theta")
        ansatz = cirq.Circuit([
            cirq.rx(theta).on(q),
        ])
        
        # Observable to measure
        observable = cirq.Z(q)
        
        # Optimize parameters (simplified)
        def objective(angle: float) -> float:
            return adapter.compute_expectation(
                ansatz,
                observable,
                params={"theta": angle},
            )
        
        # Simple grid search
        best_angle = 0
        best_energy = float("inf")
        
        for angle in np.linspace(0, 2 * np.pi, 20):
            energy = objective(angle)
            if energy < best_energy:
                best_energy = energy
                best_angle = angle
        
        # Should find minimum near theta = pi (Z expectation = -1)
        assert abs(best_angle - np.pi) < 0.5
        assert best_energy < -0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
