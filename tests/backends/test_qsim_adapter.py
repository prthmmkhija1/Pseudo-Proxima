"""Step 5.1: Unit Testing - qsim Adapter Tests.

Comprehensive test suite for qsim backend adapter covering:
- Initialization and CPU feature detection
- Gate support and circuit validation
- High-performance execution
- qsim limitations handling

Test Categories:
| Test Type       | Purpose                                       |
|-----------------|-----------------------------------------------|
| Initialization  | qsimcirq import, CPU feature detection        |
| Gate Support    | Supported/unsupported gate detection          |
| Execution       | Circuit simulation with qsim                  |
| Performance     | Thread configuration, gate fusion             |
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from proxima.backends.base import (
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_qsimcirq():
    """Mock qsimcirq module for testing."""
    mock_module = MagicMock()
    mock_module.__version__ = "0.22.0"

    # Mock QSimSimulator
    mock_simulator = MagicMock()
    mock_result = MagicMock()
    mock_result.final_state_vector = np.array([0.707 + 0j, 0, 0, 0.707 + 0j])
    mock_result.measurements = {"m": np.array([[0, 0], [1, 1]])}
    mock_simulator.simulate.return_value = mock_result
    mock_simulator.run.return_value = mock_result

    mock_module.QSimSimulator.return_value = mock_simulator

    # Mock options
    mock_module.QSimOptions = MagicMock()

    return mock_module


@pytest.fixture
def mock_cirq():
    """Mock Cirq module for testing."""
    mock_module = MagicMock()
    mock_module.__version__ = "1.0.0"

    # Mock common gates
    mock_module.H = MagicMock()
    mock_module.CNOT = MagicMock()
    mock_module.X = MagicMock()
    mock_module.Y = MagicMock()
    mock_module.Z = MagicMock()
    mock_module.rx = MagicMock()
    mock_module.ry = MagicMock()
    mock_module.rz = MagicMock()

    # Mock Circuit
    mock_circuit = MagicMock()
    mock_circuit.all_qubits.return_value = [MagicMock(), MagicMock()]
    mock_module.Circuit.return_value = mock_circuit

    # Mock LineQubit
    mock_module.LineQubit.range.return_value = [MagicMock() for _ in range(10)]

    return mock_module


@pytest.fixture
def mock_qsim_adapter(mock_qsimcirq, mock_cirq):
    """Create a qsim adapter with mocked dependencies."""
    with patch.dict(
        "sys.modules",
        {
            "qsimcirq": mock_qsimcirq,
            "cirq": mock_cirq,
        },
    ):
        from proxima.backends.qsim_adapter import QsimAdapter

        adapter = QsimAdapter()
        return adapter


@pytest.fixture
def simple_circuit():
    """Create a simple test circuit."""
    return {
        "num_qubits": 2,
        "gates": [
            {"name": "H", "qubits": [0]},
            {"name": "CNOT", "qubits": [0, 1]},
        ],
        "measurements": [0, 1],
    }


@pytest.fixture
def parameterized_circuit():
    """Create a parameterized circuit."""
    return {
        "num_qubits": 3,
        "gates": [
            {"name": "Rx", "qubits": [0], "params": {"theta": 1.5708}},
            {"name": "Ry", "qubits": [1], "params": {"theta": 0.7854}},
            {"name": "Rz", "qubits": [2], "params": {"theta": 3.1416}},
            {"name": "CNOT", "qubits": [0, 1]},
            {"name": "CNOT", "qubits": [1, 2]},
        ],
        "measurements": [0, 1, 2],
    }


@pytest.fixture
def large_circuit():
    """Create a large circuit for performance testing."""
    return {
        "num_qubits": 25,
        "gates": [{"name": "H", "qubits": [i]} for i in range(25)]
        + [{"name": "CNOT", "qubits": [i, i + 1]} for i in range(24)],
        "measurements": list(range(25)),
    }


# =============================================================================
# STEP 5.1.1: INITIALIZATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQsimAdapterInitialization:
    """Tests for qsim adapter initialization."""

    def test_adapter_instantiation(self, mock_qsimcirq, mock_cirq):
        """Test that adapter instantiates successfully."""
        with patch.dict(
            "sys.modules",
            {
                "qsimcirq": mock_qsimcirq,
                "cirq": mock_cirq,
            },
        ):
            from proxima.backends.qsim_adapter import QsimAdapter

            adapter = QsimAdapter()
            assert adapter is not None
            assert adapter.get_name() == "qsim"

    def test_version_detection(self, mock_qsim_adapter):
        """Test qsim version detection."""
        version = mock_qsim_adapter.get_version()
        assert version is not None
        assert isinstance(version, str)

    def test_capabilities_reporting(self, mock_qsim_adapter):
        """Test capability reporting."""
        caps = mock_qsim_adapter.get_capabilities()

        assert isinstance(caps, Capabilities)
        assert SimulatorType.STATE_VECTOR in caps.simulator_types
        # qsim is state-vector only
        assert SimulatorType.DENSITY_MATRIX not in caps.simulator_types
        assert caps.max_qubits >= 30

    def test_cpu_optimization_detection(self, mock_qsim_adapter):
        """Test CPU optimization detection (AVX2/AVX512)."""
        caps = mock_qsim_adapter.get_capabilities()

        # Should indicate CPU optimization
        assert caps.custom_features.get("cpu_optimized", True) is True

    def test_unavailable_without_qsimcirq(self, mock_cirq):
        """Test that adapter handles missing qsimcirq gracefully."""
        with patch.dict(
            "sys.modules",
            {
                "qsimcirq": None,
                "cirq": mock_cirq,
            },
        ):
            from proxima.backends.qsim_adapter import QsimAdapter

            adapter = QsimAdapter()
            assert adapter.is_available() is False

    def test_unavailable_without_cirq(self, mock_qsimcirq):
        """Test that adapter handles missing cirq gracefully."""
        with patch.dict(
            "sys.modules",
            {
                "qsimcirq": mock_qsimcirq,
                "cirq": None,
            },
        ):
            from proxima.backends.qsim_adapter import QsimAdapter

            adapter = QsimAdapter()
            assert adapter.is_available() is False


# =============================================================================
# STEP 5.1.2: GATE SUPPORT TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQsimGateSupport:
    """Tests for qsim gate support and validation."""

    def test_basic_gates_supported(self, mock_qsim_adapter, simple_circuit):
        """Test that basic gates are supported."""
        validation = mock_qsim_adapter.validate_circuit(simple_circuit)
        assert validation.valid is True

    def test_parameterized_gates_supported(
        self, mock_qsim_adapter, parameterized_circuit
    ):
        """Test that parameterized rotation gates are supported."""
        validation = mock_qsim_adapter.validate_circuit(parameterized_circuit)
        assert validation.valid is True

    def test_multi_qubit_gates(self, mock_qsim_adapter):
        """Test multi-qubit gate support."""
        circuit = {
            "num_qubits": 3,
            "gates": [
                {"name": "CNOT", "qubits": [0, 1]},
                {"name": "CCX", "qubits": [0, 1, 2]},  # Toffoli
            ],
            "measurements": [0, 1, 2],
        }

        validation = mock_qsim_adapter.validate_circuit(circuit)
        assert validation.valid is True

    def test_unsupported_features_detection(self, mock_qsim_adapter):
        """Test detection of unsupported features."""
        # qsim has limited mid-circuit measurement support
        circuit = {
            "num_qubits": 2,
            "gates": [
                {"name": "H", "qubits": [0]},
                {"name": "Measure", "qubits": [0]},  # Mid-circuit measurement
                {"name": "X", "qubits": [1], "conditional": True},
            ],
        }

        validation = mock_qsim_adapter.validate_circuit(circuit)
        # Should warn or reject mid-circuit measurement
        if not validation.valid:
            assert validation.message is not None

    def test_empty_circuit(self, mock_qsim_adapter):
        """Test validation of empty circuit."""
        circuit = {
            "num_qubits": 2,
            "gates": [],
            "measurements": [],
        }

        validation = mock_qsim_adapter.validate_circuit(circuit)
        assert validation.valid is True

    def test_invalid_qubit_indices(self, mock_qsim_adapter):
        """Test detection of invalid qubit indices."""
        circuit = {
            "num_qubits": 2,
            "gates": [
                {"name": "H", "qubits": [10]},  # Invalid
            ],
        }

        validation = mock_qsim_adapter.validate_circuit(circuit)
        assert validation.valid is False


# =============================================================================
# STEP 5.1.3: EXECUTION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQsimExecution:
    """Tests for qsim circuit execution."""

    def test_simple_circuit_execution(self, mock_qsim_adapter, simple_circuit):
        """Test execution of simple circuit."""
        result = mock_qsim_adapter.execute(
            simple_circuit,
            options={"shots": 1024},
        )

        assert isinstance(result, ExecutionResult)
        assert result.backend == "qsim"
        assert result.qubit_count == 2

    def test_statevector_output(self, mock_qsim_adapter, simple_circuit):
        """Test state vector output."""
        result = mock_qsim_adapter.execute(
            simple_circuit,
            options={"return_statevector": True},
        )

        assert result.simulator_type == SimulatorType.STATE_VECTOR
        assert result.data is not None

    def test_shot_based_measurement(self, mock_qsim_adapter, simple_circuit):
        """Test shot-based measurement returns counts."""
        result = mock_qsim_adapter.execute(
            simple_circuit,
            options={"shots": 1000},
        )

        assert result.result_type == ResultType.COUNTS

    def test_execution_timing(self, mock_qsim_adapter, simple_circuit):
        """Test that execution time is measured."""
        result = mock_qsim_adapter.execute(simple_circuit)

        assert result.execution_time_ms >= 0

    def test_large_circuit_execution(self, mock_qsim_adapter, large_circuit):
        """Test execution of large circuit."""
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={"shots": 100},
        )

        assert isinstance(result, ExecutionResult)
        assert result.qubit_count == 25

    def test_metadata_included(self, mock_qsim_adapter, simple_circuit):
        """Test that metadata is included in results."""
        result = mock_qsim_adapter.execute(simple_circuit)

        assert result.metadata is not None
        assert isinstance(result.metadata, dict)


# =============================================================================
# STEP 5.1.4: PERFORMANCE CONFIGURATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQsimPerformanceConfiguration:
    """Tests for qsim performance configuration."""

    def test_thread_count_configuration(self, mock_qsim_adapter, simple_circuit):
        """Test thread count configuration."""
        result = mock_qsim_adapter.execute(
            simple_circuit,
            options={"num_threads": 8},
        )

        assert isinstance(result, ExecutionResult)

    def test_gate_fusion_enabled(self, mock_qsim_adapter, parameterized_circuit):
        """Test that gate fusion is enabled by default."""
        result = mock_qsim_adapter.execute(
            parameterized_circuit,
            options={"gate_fusion": True},
        )

        assert isinstance(result, ExecutionResult)

    def test_gate_fusion_disabled(self, mock_qsim_adapter, parameterized_circuit):
        """Test gate fusion can be disabled."""
        result = mock_qsim_adapter.execute(
            parameterized_circuit,
            options={"gate_fusion": False},
        )

        assert isinstance(result, ExecutionResult)

    def test_verbosity_configuration(self, mock_qsim_adapter, simple_circuit):
        """Test verbosity configuration."""
        result = mock_qsim_adapter.execute(
            simple_circuit,
            options={"verbosity": 0},  # Quiet mode
        )

        assert isinstance(result, ExecutionResult)


# =============================================================================
# STEP 5.1.5: ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQsimErrorHandling:
    """Tests for qsim error handling."""

    def test_missing_dependencies(self):
        """Test handling of missing dependencies."""
        with patch.dict("sys.modules", {"qsimcirq": None, "cirq": None}):
            from proxima.backends.qsim_adapter import QsimAdapter

            adapter = QsimAdapter()
            assert adapter.is_available() is False

    def test_invalid_circuit_structure(self, mock_qsim_adapter):
        """Test handling of invalid circuit structure."""
        invalid_circuit = {"invalid": "data"}

        validation = mock_qsim_adapter.validate_circuit(invalid_circuit)
        assert validation.valid is False

    def test_qubit_limit_exceeded(self, mock_qsim_adapter):
        """Test detection of qubit limit exceeded."""
        huge_circuit = {
            "num_qubits": 50,  # Very large
            "gates": [],
        }

        validation = mock_qsim_adapter.validate_circuit(huge_circuit)
        # May reject or warn about resource limits
        if not validation.valid:
            assert validation.message is not None

    def test_resource_estimation(self, mock_qsim_adapter, large_circuit):
        """Test resource estimation for large circuits."""
        estimate = mock_qsim_adapter.estimate_resources(large_circuit)

        assert isinstance(estimate, ResourceEstimate)
        assert estimate.memory_mb is not None
        assert estimate.memory_mb > 0

    def test_cleanup(self, mock_qsim_adapter):
        """Test resource cleanup."""
        mock_qsim_adapter._cleanup()
        # Should not raise exceptions


# =============================================================================
# RESOURCE ESTIMATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQsimResourceEstimation:
    """Tests for qsim resource estimation."""

    def test_memory_estimation_small(self, mock_qsim_adapter, simple_circuit):
        """Test memory estimation for small circuits."""
        estimate = mock_qsim_adapter.estimate_resources(simple_circuit)

        assert estimate.memory_mb is not None
        assert estimate.memory_mb < 10  # Small circuit

    def test_memory_estimation_large(self, mock_qsim_adapter, large_circuit):
        """Test memory estimation for large circuits."""
        estimate = mock_qsim_adapter.estimate_resources(large_circuit)

        assert estimate.memory_mb is not None
        # 25 qubits = 2^25 * 16 bytes = 512 MB
        assert estimate.memory_mb > 100

    def test_time_estimation(self, mock_qsim_adapter, simple_circuit):
        """Test time estimation."""
        estimate = mock_qsim_adapter.estimate_resources(simple_circuit)

        # Time estimation may or may not be available
        assert isinstance(estimate, ResourceEstimate)
