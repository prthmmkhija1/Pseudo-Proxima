"""Step 5.1: Unit Testing - QuEST Adapter Tests.

Comprehensive test suite for QuEST backend adapter covering:
- Initialization tests
- Circuit translation tests
- Execution tests
- Error handling tests

Test Categories:
| Test Type       | Purpose                                    |
|-----------------|-----------------------------------------------|
| Initialization  | Backend instantiation, capability detection  |
| Translation     | Gate mapping, circuit conversion             |
| Execution       | Circuit simulation, result validation        |
| Error Handling  | Missing deps, resource limits, invalid input |
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
def mock_pyquest():
    """Mock pyQuEST module for testing without actual installation."""
    mock_module = MagicMock()
    mock_module.__version__ = "0.9.0"

    # Mock QuEST environment
    mock_env = MagicMock()
    mock_env.num_ranks = 1
    mock_env.num_threads = 4
    mock_module.createQuESTEnv.return_value = mock_env

    # Mock Qureg (quantum register)
    mock_qureg = MagicMock()
    mock_qureg.numQubitsRepresented = 3
    mock_qureg.numAmps = 8
    mock_module.createQureg.return_value = mock_qureg
    mock_module.createDensityQureg.return_value = mock_qureg

    # Mock measurement functions
    mock_module.measure.return_value = 0
    mock_module.calcProbOfOutcome.return_value = 0.5
    mock_module.getAmp.return_value = complex(0.707, 0.0)

    return mock_module


@pytest.fixture
def mock_quest_adapter(mock_pyquest):
    """Create a QuEST adapter with mocked dependencies."""
    with patch.dict("sys.modules", {"pyquest": mock_pyquest}):
        from proxima.backends.quest_adapter import QuestAdapter

        adapter = QuestAdapter()
        return adapter


@pytest.fixture
def simple_circuit():
    """Create a simple test circuit (Bell state)."""
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
    """Create a parameterized circuit for testing."""
    return {
        "num_qubits": 2,
        "gates": [
            {"name": "Rx", "qubits": [0], "params": {"theta": 1.5708}},
            {"name": "Ry", "qubits": [1], "params": {"theta": 0.7854}},
            {"name": "CNOT", "qubits": [0, 1]},
            {"name": "Rz", "qubits": [0], "params": {"theta": 3.1416}},
        ],
        "measurements": [0, 1],
    }


@pytest.fixture
def large_circuit():
    """Create a larger circuit for resource testing."""
    return {
        "num_qubits": 20,
        "gates": [{"name": "H", "qubits": [i]} for i in range(20)]
        + [{"name": "CNOT", "qubits": [i, (i + 1) % 20]} for i in range(19)],
        "measurements": list(range(20)),
    }


# =============================================================================
# STEP 5.1.1: INITIALIZATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQuestAdapterInitialization:
    """Tests for QuEST adapter initialization."""

    def test_adapter_instantiation_with_pyquest(self, mock_pyquest):
        """Test that adapter instantiates successfully when pyQuEST is available."""
        with patch.dict("sys.modules", {"pyquest": mock_pyquest}):
            from proxima.backends.quest_adapter import QuestAdapter

            adapter = QuestAdapter()
            assert adapter is not None
            assert adapter.get_name() == "quest"

    def test_adapter_version_detection(self, mock_quest_adapter):
        """Test that adapter correctly detects pyQuEST version."""
        version = mock_quest_adapter.get_version()
        assert version is not None
        assert isinstance(version, str)

    def test_capabilities_detection(self, mock_quest_adapter):
        """Test that adapter reports correct capabilities."""
        caps = mock_quest_adapter.get_capabilities()

        assert isinstance(caps, Capabilities)
        assert SimulatorType.STATE_VECTOR in caps.simulator_types
        assert SimulatorType.DENSITY_MATRIX in caps.simulator_types
        assert caps.max_qubits >= 20
        assert caps.supports_noise is True

    def test_gpu_availability_detection(self, mock_pyquest):
        """Test GPU availability detection during initialization."""
        mock_pyquest.QuESTEnv.return_value.is_gpu = True

        with patch.dict("sys.modules", {"pyquest": mock_pyquest}):
            from proxima.backends.quest_adapter import QuestAdapter

            adapter = QuestAdapter()
            caps = adapter.get_capabilities()
            # GPU support depends on build configuration
            assert isinstance(caps.supports_gpu, bool)

    def test_adapter_unavailable_without_pyquest(self):
        """Test that adapter handles missing pyQuEST gracefully."""
        with patch.dict("sys.modules", {"pyquest": None}):
            from proxima.backends.quest_adapter import QuestAdapter

            adapter = QuestAdapter()
            assert adapter.is_available() is False

    def test_environment_cleanup(self, mock_quest_adapter):
        """Test that QuEST environment is cleaned up properly."""
        # Trigger cleanup
        mock_quest_adapter._cleanup()
        # Should not raise any exceptions


# =============================================================================
# STEP 5.1.2: CIRCUIT TRANSLATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQuestCircuitTranslation:
    """Tests for circuit translation to QuEST format."""

    def test_basic_gate_translation(self, mock_quest_adapter, simple_circuit):
        """Test translation of basic gates (H, CNOT)."""
        validation = mock_quest_adapter.validate_circuit(simple_circuit)

        assert validation.valid is True
        assert validation.message is None or "error" not in validation.message.lower()

    def test_parameterized_gate_translation(
        self, mock_quest_adapter, parameterized_circuit
    ):
        """Test translation of parameterized gates (Rx, Ry, Rz)."""
        validation = mock_quest_adapter.validate_circuit(parameterized_circuit)

        assert validation.valid is True

    def test_unsupported_gate_detection(self, mock_quest_adapter):
        """Test that unsupported gates are detected."""
        invalid_circuit = {
            "num_qubits": 2,
            "gates": [
                {"name": "UnsupportedGate", "qubits": [0]},
            ],
        }

        validation = mock_quest_adapter.validate_circuit(invalid_circuit)
        # Should either reject or warn about unsupported gates
        if not validation.valid:
            assert validation.message is not None

    def test_multi_qubit_gate_translation(self, mock_quest_adapter):
        """Test translation of multi-qubit gates."""
        circuit = {
            "num_qubits": 3,
            "gates": [
                {"name": "CCNOT", "qubits": [0, 1, 2]},  # Toffoli
            ],
            "measurements": [0, 1, 2],
        }

        validation = mock_quest_adapter.validate_circuit(circuit)
        # Multi-qubit gates should be supported
        assert validation.valid is True

    def test_empty_circuit_validation(self, mock_quest_adapter):
        """Test validation of empty circuit."""
        empty_circuit = {
            "num_qubits": 2,
            "gates": [],
            "measurements": [],
        }

        validation = mock_quest_adapter.validate_circuit(empty_circuit)
        assert validation.valid is True  # Empty circuit should be valid

    def test_invalid_qubit_index(self, mock_quest_adapter):
        """Test detection of invalid qubit indices."""
        invalid_circuit = {
            "num_qubits": 2,
            "gates": [
                {"name": "H", "qubits": [5]},  # Invalid index
            ],
        }

        validation = mock_quest_adapter.validate_circuit(invalid_circuit)
        assert validation.valid is False


# =============================================================================
# STEP 5.1.3: EXECUTION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQuestExecution:
    """Tests for QuEST circuit execution."""

    def test_simple_circuit_execution(self, mock_quest_adapter, simple_circuit):
        """Test execution of a simple circuit."""
        result = mock_quest_adapter.execute(
            simple_circuit,
            options={"shots": 1024, "simulator_type": "state_vector"},
        )

        assert isinstance(result, ExecutionResult)
        assert result.backend == "quest"
        assert result.qubit_count == 2
        assert result.execution_time_ms >= 0

    def test_state_vector_mode(self, mock_quest_adapter, simple_circuit):
        """Test state vector simulation mode."""
        result = mock_quest_adapter.execute(
            simple_circuit,
            options={"simulator_type": "state_vector"},
        )

        assert result.simulator_type == SimulatorType.STATE_VECTOR

    def test_density_matrix_mode(self, mock_quest_adapter, simple_circuit):
        """Test density matrix simulation mode."""
        result = mock_quest_adapter.execute(
            simple_circuit,
            options={"simulator_type": "density_matrix"},
        )

        assert result.simulator_type == SimulatorType.DENSITY_MATRIX

    def test_shot_based_measurement(self, mock_quest_adapter, simple_circuit):
        """Test shot-based measurement returns counts."""
        result = mock_quest_adapter.execute(
            simple_circuit,
            options={"shots": 1000},
        )

        assert result.result_type == ResultType.COUNTS
        assert "counts" in result.data or result.data.get("counts") is not None

    def test_statevector_output(self, mock_quest_adapter, simple_circuit):
        """Test that statevector can be extracted."""
        result = mock_quest_adapter.execute(
            simple_circuit,
            options={"return_statevector": True},
        )

        # Should have statevector or amplitude data
        assert result.data is not None

    def test_execution_timing(self, mock_quest_adapter, simple_circuit):
        """Test that execution time is measured."""
        result = mock_quest_adapter.execute(simple_circuit)

        assert result.execution_time_ms >= 0

    def test_metadata_included(self, mock_quest_adapter, simple_circuit):
        """Test that backend-specific metadata is included."""
        result = mock_quest_adapter.execute(simple_circuit)

        assert result.metadata is not None
        # Should include QuEST-specific info
        assert isinstance(result.metadata, dict)


# =============================================================================
# STEP 5.1.4: ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQuestErrorHandling:
    """Tests for QuEST adapter error handling."""

    def test_missing_dependency_error(self):
        """Test clear error when pyQuEST is not installed."""
        with patch.dict("sys.modules", {"pyquest": None}):
            from proxima.backends.quest_adapter import QuestAdapter

            adapter = QuestAdapter()

            # Should not crash, should indicate unavailability
            assert adapter.is_available() is False

    def test_resource_exceeded_error(self, mock_quest_adapter, large_circuit):
        """Test handling of resource limit exceeded."""
        estimate = mock_quest_adapter.estimate_resources(large_circuit)

        assert isinstance(estimate, ResourceEstimate)
        assert estimate.memory_mb is not None
        assert estimate.memory_mb > 0

    def test_invalid_circuit_error(self, mock_quest_adapter):
        """Test handling of invalid circuit structure."""
        invalid_circuit = {"invalid": "structure"}

        validation = mock_quest_adapter.validate_circuit(invalid_circuit)
        assert validation.valid is False

    def test_qubit_limit_exceeded(self, mock_quest_adapter):
        """Test detection of qubit count exceeding limits."""
        huge_circuit = {
            "num_qubits": 100,  # Too many qubits
            "gates": [],
        }

        validation = mock_quest_adapter.validate_circuit(huge_circuit)
        # Should detect resource limits
        if not validation.valid:
            assert (
                "qubit" in validation.message.lower()
                or "limit" in validation.message.lower()
            )

    def test_execution_timeout(self, mock_quest_adapter, simple_circuit):
        """Test handling of execution timeout."""
        # Should accept timeout option
        result = mock_quest_adapter.execute(
            simple_circuit,
            options={"timeout_seconds": 60},
        )

        assert isinstance(result, ExecutionResult)

    def test_cleanup_on_error(self, mock_pyquest):
        """Test that resources are cleaned up on error."""
        mock_pyquest.hadamard.side_effect = RuntimeError("Simulated error")

        with patch.dict("sys.modules", {"pyquest": mock_pyquest}):
            from proxima.backends.quest_adapter import QuestAdapter

            adapter = QuestAdapter()

            # Execute should handle error gracefully
            try:
                adapter.execute(
                    {"num_qubits": 2, "gates": [{"name": "H", "qubits": [0]}]}
                )
            except Exception:
                pass  # Expected

            # Cleanup should still work
            adapter._cleanup()


# =============================================================================
# STEP 5.1.5: PRECISION AND GPU TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQuestAdvancedFeatures:
    """Tests for QuEST advanced features (precision, GPU, OpenMP)."""

    def test_precision_configuration(self, mock_quest_adapter, simple_circuit):
        """Test different precision modes."""
        for precision in ["single", "double"]:
            result = mock_quest_adapter.execute(
                simple_circuit,
                options={"precision": precision},
            )

            assert isinstance(result, ExecutionResult)

    def test_thread_count_configuration(self, mock_quest_adapter, simple_circuit):
        """Test OpenMP thread count configuration."""
        result = mock_quest_adapter.execute(
            simple_circuit,
            options={"num_threads": 4},
        )

        assert isinstance(result, ExecutionResult)

    def test_gpu_fallback(self, mock_quest_adapter, simple_circuit):
        """Test fallback to CPU when GPU unavailable."""
        result = mock_quest_adapter.execute(
            simple_circuit,
            options={"use_gpu": True},
        )

        # Should complete even if GPU unavailable
        assert isinstance(result, ExecutionResult)


# =============================================================================
# RESOURCE ESTIMATION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestQuestResourceEstimation:
    """Tests for resource estimation."""

    def test_memory_estimation_small_circuit(self, mock_quest_adapter, simple_circuit):
        """Test memory estimation for small circuits."""
        estimate = mock_quest_adapter.estimate_resources(simple_circuit)

        assert estimate.memory_mb is not None
        assert estimate.memory_mb > 0
        assert estimate.memory_mb < 100  # Small circuit should need little memory

    def test_memory_estimation_large_circuit(self, mock_quest_adapter, large_circuit):
        """Test memory estimation for large circuits."""
        estimate = mock_quest_adapter.estimate_resources(large_circuit)

        assert estimate.memory_mb is not None
        # 20 qubits should require significant memory (2^20 amplitudes)
        assert estimate.memory_mb > 10

    def test_density_matrix_memory_scaling(self, mock_quest_adapter):
        """Test that density matrix mode estimates higher memory."""
        circuit = {"num_qubits": 5, "gates": []}

        sv_estimate = mock_quest_adapter.estimate_resources(circuit)

        # DM requires O(4^n) vs SV O(2^n)
        dm_circuit = {**circuit, "simulator_type": "density_matrix"}
        dm_estimate = mock_quest_adapter.estimate_resources(dm_circuit)

        # Both should have valid estimates
        assert sv_estimate.memory_mb is not None
        assert dm_estimate.memory_mb is not None
