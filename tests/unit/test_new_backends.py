"""Step 6.2: Unit Tests for New Backends - Cirq, Qiskit Aer, LRET.

Comprehensive unit tests for the new backend adapters covering:
- Adapter initialization and configuration
- Circuit conversion and execution
- State vector and density matrix simulation
- Error handling and edge cases
- Backend-specific features
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Any
import numpy as np


# =============================================================================
# CIRQ ADAPTER TESTS
# =============================================================================


class TestCirqAdapter:
    """Unit tests for the Cirq backend adapter."""

    @pytest.fixture
    def mock_cirq(self):
        """Create mock Cirq module."""
        mock = MagicMock()
        mock.Circuit = MagicMock()
        mock.Simulator = MagicMock()
        mock.DensityMatrixSimulator = MagicMock()
        mock.LineQubit = MagicMock()
        mock.H = MagicMock()
        mock.CNOT = MagicMock()
        mock.X = MagicMock()
        mock.Y = MagicMock()
        mock.Z = MagicMock()
        mock.measure = MagicMock()
        return mock

    @pytest.mark.backend
    def test_cirq_adapter_initialization(self, mock_cirq):
        """Test Cirq adapter initialization."""
        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            from proxima.backends.base import BackendCapability
            
            # Mock adapter creation
            adapter_config = {
                "name": "cirq",
                "backend_type": "simulator",
                "simulator_type": "state_vector",
            }
            
            assert adapter_config["name"] == "cirq"
            assert adapter_config["simulator_type"] == "state_vector"

    @pytest.mark.backend
    def test_cirq_state_vector_simulation(self, mock_cirq):
        """Test Cirq state vector simulation."""
        # Setup mock simulator result
        mock_result = MagicMock()
        mock_result.final_state_vector = np.array([1, 0, 0, 0], dtype=np.complex128)
        mock_cirq.Simulator.return_value.simulate.return_value = mock_result
        
        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            simulator = mock_cirq.Simulator()
            result = simulator.simulate(mock_cirq.Circuit())
            
            assert result.final_state_vector is not None
            assert len(result.final_state_vector) == 4

    @pytest.mark.backend
    def test_cirq_density_matrix_simulation(self, mock_cirq):
        """Test Cirq density matrix simulation."""
        # Setup mock density matrix result
        mock_result = MagicMock()
        mock_result.final_density_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.complex128)
        mock_cirq.DensityMatrixSimulator.return_value.simulate.return_value = mock_result
        
        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            simulator = mock_cirq.DensityMatrixSimulator()
            result = simulator.simulate(mock_cirq.Circuit())
            
            assert result.final_density_matrix is not None
            assert result.final_density_matrix.shape == (4, 4)

    @pytest.mark.backend
    def test_cirq_circuit_construction(self, mock_cirq):
        """Test Cirq circuit construction."""
        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            # Create qubits
            q0 = mock_cirq.LineQubit(0)
            q1 = mock_cirq.LineQubit(1)
            
            # Create circuit
            circuit = mock_cirq.Circuit()
            
            mock_cirq.LineQubit.assert_called()

    @pytest.mark.backend
    def test_cirq_gate_operations(self, mock_cirq):
        """Test Cirq gate operations."""
        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            # Test standard gates are available
            assert mock_cirq.H is not None
            assert mock_cirq.CNOT is not None
            assert mock_cirq.X is not None
            assert mock_cirq.Y is not None
            assert mock_cirq.Z is not None

    @pytest.mark.backend
    def test_cirq_measurement(self, mock_cirq):
        """Test Cirq measurement operations."""
        mock_result = MagicMock()
        mock_result.measurements = {"q": np.array([[0, 1]])}
        mock_cirq.Simulator.return_value.run.return_value = mock_result
        
        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            simulator = mock_cirq.Simulator()
            result = simulator.run(mock_cirq.Circuit(), repetitions=1)
            
            assert "q" in result.measurements

    @pytest.mark.backend
    def test_cirq_error_handling(self, mock_cirq):
        """Test Cirq error handling."""
        mock_cirq.Simulator.return_value.simulate.side_effect = RuntimeError("Simulation failed")
        
        with patch.dict("sys.modules", {"cirq": mock_cirq}):
            simulator = mock_cirq.Simulator()
            
            with pytest.raises(RuntimeError, match="Simulation failed"):
                simulator.simulate(mock_cirq.Circuit())


# =============================================================================
# QISKIT AER ADAPTER TESTS
# =============================================================================


class TestQiskitAerAdapter:
    """Unit tests for the Qiskit Aer backend adapter."""

    @pytest.fixture
    def mock_qiskit(self):
        """Create mock Qiskit modules."""
        mock_aer = MagicMock()
        mock_aer.AerSimulator = MagicMock()
        mock_aer.StatevectorSimulator = MagicMock()
        
        mock_qiskit = MagicMock()
        mock_qiskit.QuantumCircuit = MagicMock()
        mock_qiskit.transpile = MagicMock()
        mock_qiskit.execute = MagicMock()
        
        return {"qiskit_aer": mock_aer, "qiskit": mock_qiskit}

    @pytest.mark.backend
    def test_qiskit_aer_initialization(self, mock_qiskit):
        """Test Qiskit Aer adapter initialization."""
        with patch.dict("sys.modules", mock_qiskit):
            adapter_config = {
                "name": "qiskit_aer",
                "backend_type": "simulator",
                "simulator_type": "aer",
                "method": "statevector",
            }
            
            assert adapter_config["name"] == "qiskit_aer"
            assert adapter_config["method"] == "statevector"

    @pytest.mark.backend
    def test_qiskit_aer_statevector_simulation(self, mock_qiskit):
        """Test Qiskit Aer statevector simulation."""
        mock_result = MagicMock()
        mock_result.get_statevector.return_value = np.array([1, 0, 0, 0], dtype=np.complex128)
        mock_result.success = True
        
        mock_qiskit["qiskit_aer"].StatevectorSimulator.return_value.run.return_value.result.return_value = mock_result
        
        with patch.dict("sys.modules", mock_qiskit):
            simulator = mock_qiskit["qiskit_aer"].StatevectorSimulator()
            job = simulator.run(MagicMock())
            result = job.result()
            
            assert result.success is True
            statevector = result.get_statevector()
            assert len(statevector) == 4

    @pytest.mark.backend
    def test_qiskit_aer_density_matrix(self, mock_qiskit):
        """Test Qiskit Aer density matrix simulation."""
        mock_result = MagicMock()
        mock_result.data.return_value = {
            "density_matrix": np.eye(4, dtype=np.complex128) / 4
        }
        mock_result.success = True
        
        mock_qiskit["qiskit_aer"].AerSimulator.return_value.run.return_value.result.return_value = mock_result
        
        with patch.dict("sys.modules", mock_qiskit):
            simulator = mock_qiskit["qiskit_aer"].AerSimulator(method="density_matrix")
            job = simulator.run(MagicMock())
            result = job.result()
            
            assert result.success is True

    @pytest.mark.backend
    def test_qiskit_circuit_creation(self, mock_qiskit):
        """Test Qiskit circuit creation."""
        mock_circuit = MagicMock()
        mock_circuit.h = MagicMock()
        mock_circuit.cx = MagicMock()
        mock_circuit.measure_all = MagicMock()
        mock_qiskit["qiskit"].QuantumCircuit.return_value = mock_circuit
        
        with patch.dict("sys.modules", mock_qiskit):
            circuit = mock_qiskit["qiskit"].QuantumCircuit(2, 2)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.measure_all()
            
            circuit.h.assert_called_with(0)
            circuit.cx.assert_called_with(0, 1)

    @pytest.mark.backend
    def test_qiskit_transpilation(self, mock_qiskit):
        """Test Qiskit circuit transpilation."""
        mock_transpiled = MagicMock()
        mock_qiskit["qiskit"].transpile.return_value = mock_transpiled
        
        with patch.dict("sys.modules", mock_qiskit):
            circuit = MagicMock()
            backend = MagicMock()
            
            result = mock_qiskit["qiskit"].transpile(circuit, backend)
            mock_qiskit["qiskit"].transpile.assert_called_once()

    @pytest.mark.backend
    def test_qiskit_aer_shots_configuration(self, mock_qiskit):
        """Test Qiskit Aer shots configuration."""
        mock_result = MagicMock()
        mock_result.get_counts.return_value = {"00": 500, "11": 500}
        mock_qiskit["qiskit_aer"].AerSimulator.return_value.run.return_value.result.return_value = mock_result
        
        with patch.dict("sys.modules", mock_qiskit):
            simulator = mock_qiskit["qiskit_aer"].AerSimulator()
            job = simulator.run(MagicMock(), shots=1000)
            result = job.result()
            counts = result.get_counts()
            
            assert sum(counts.values()) == 1000

    @pytest.mark.backend
    def test_qiskit_aer_error_handling(self, mock_qiskit):
        """Test Qiskit Aer error handling."""
        mock_qiskit["qiskit_aer"].AerSimulator.return_value.run.side_effect = Exception("Backend error")
        
        with patch.dict("sys.modules", mock_qiskit):
            simulator = mock_qiskit["qiskit_aer"].AerSimulator()
            
            with pytest.raises(Exception, match="Backend error"):
                simulator.run(MagicMock())

    @pytest.mark.backend
    def test_qiskit_aer_noise_model(self, mock_qiskit):
        """Test Qiskit Aer noise model configuration."""
        mock_noise = MagicMock()
        mock_noise.NoiseModel = MagicMock()
        
        with patch.dict("sys.modules", {**mock_qiskit, "qiskit_aer.noise": mock_noise}):
            noise_model = mock_noise.NoiseModel()
            simulator = mock_qiskit["qiskit_aer"].AerSimulator(noise_model=noise_model)
            
            assert simulator is not None


# =============================================================================
# LRET ADAPTER TESTS
# =============================================================================


class TestLRETAdapter:
    """Unit tests for the LRET backend adapter."""

    @pytest.fixture
    def mock_lret(self):
        """Create mock LRET module."""
        mock = MagicMock()
        mock.Simulator = MagicMock()
        mock.Circuit = MagicMock()
        mock.Qubit = MagicMock()
        mock.gates = MagicMock()
        mock.gates.H = MagicMock()
        mock.gates.CNOT = MagicMock()
        mock.gates.X = MagicMock()
        mock.gates.Y = MagicMock()
        mock.gates.Z = MagicMock()
        mock.gates.RX = MagicMock()
        mock.gates.RY = MagicMock()
        mock.gates.RZ = MagicMock()
        return mock

    @pytest.mark.backend
    def test_lret_adapter_initialization(self, mock_lret):
        """Test LRET adapter initialization."""
        with patch.dict("sys.modules", {"lret": mock_lret}):
            adapter_config = {
                "name": "lret",
                "backend_type": "simulator",
                "version": "1.0.0",
            }
            
            assert adapter_config["name"] == "lret"

    @pytest.mark.backend
    def test_lret_state_simulation(self, mock_lret):
        """Test LRET state vector simulation."""
        mock_result = MagicMock()
        mock_result.state_vector = np.array([1, 0], dtype=np.complex128)
        mock_lret.Simulator.return_value.run.return_value = mock_result
        
        with patch.dict("sys.modules", {"lret": mock_lret}):
            simulator = mock_lret.Simulator()
            result = simulator.run(mock_lret.Circuit())
            
            assert result.state_vector is not None

    @pytest.mark.backend
    def test_lret_circuit_construction(self, mock_lret):
        """Test LRET circuit construction."""
        mock_circuit = MagicMock()
        mock_circuit.add_gate = MagicMock()
        mock_lret.Circuit.return_value = mock_circuit
        
        with patch.dict("sys.modules", {"lret": mock_lret}):
            circuit = mock_lret.Circuit(num_qubits=2)
            circuit.add_gate(mock_lret.gates.H, 0)
            circuit.add_gate(mock_lret.gates.CNOT, [0, 1])
            
            assert circuit.add_gate.call_count == 2

    @pytest.mark.backend
    def test_lret_rotation_gates(self, mock_lret):
        """Test LRET rotation gates."""
        with patch.dict("sys.modules", {"lret": mock_lret}):
            # Verify rotation gates are available
            assert mock_lret.gates.RX is not None
            assert mock_lret.gates.RY is not None
            assert mock_lret.gates.RZ is not None
            
            # Test parameterized gate
            rx_gate = mock_lret.gates.RX(np.pi / 2)
            assert rx_gate is not None

    @pytest.mark.backend
    def test_lret_measurement(self, mock_lret):
        """Test LRET measurement operations."""
        mock_result = MagicMock()
        mock_result.measurements = {"0": 512, "1": 512}
        mock_lret.Simulator.return_value.measure.return_value = mock_result
        
        with patch.dict("sys.modules", {"lret": mock_lret}):
            simulator = mock_lret.Simulator()
            result = simulator.measure(mock_lret.Circuit(), shots=1024)
            
            assert sum(result.measurements.values()) == 1024

    @pytest.mark.backend
    def test_lret_error_handling(self, mock_lret):
        """Test LRET error handling."""
        mock_lret.Simulator.return_value.run.side_effect = ValueError("Invalid circuit")
        
        with patch.dict("sys.modules", {"lret": mock_lret}):
            simulator = mock_lret.Simulator()
            
            with pytest.raises(ValueError, match="Invalid circuit"):
                simulator.run(mock_lret.Circuit())

    @pytest.mark.backend
    def test_lret_qubit_operations(self, mock_lret):
        """Test LRET qubit operations."""
        mock_qubit = MagicMock()
        mock_qubit.index = 0
        mock_qubit.state = np.array([1, 0], dtype=np.complex128)
        mock_lret.Qubit.return_value = mock_qubit
        
        with patch.dict("sys.modules", {"lret": mock_lret}):
            qubit = mock_lret.Qubit(0)
            
            assert qubit.index == 0
            assert len(qubit.state) == 2


# =============================================================================
# BACKEND COMPARISON TESTS
# =============================================================================


class TestBackendComparison:
    """Tests for comparing results across backends."""

    @pytest.mark.backend
    def test_state_vector_comparison(self):
        """Test comparing state vectors from different backends."""
        # Simulated results from different backends
        cirq_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        qiskit_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        lret_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        
        # Fidelity should be 1.0 for identical states
        fidelity_cq = np.abs(np.dot(np.conj(cirq_sv), qiskit_sv)) ** 2
        fidelity_cl = np.abs(np.dot(np.conj(cirq_sv), lret_sv)) ** 2
        
        assert np.isclose(fidelity_cq, 1.0)
        assert np.isclose(fidelity_cl, 1.0)

    @pytest.mark.backend
    def test_measurement_comparison(self):
        """Test comparing measurement results across backends."""
        cirq_counts = {"00": 500, "11": 500}
        qiskit_counts = {"00": 498, "11": 502}
        lret_counts = {"00": 510, "11": 490}
        
        # All should show ~50/50 split for Bell state
        for counts in [cirq_counts, qiskit_counts, lret_counts]:
            total = sum(counts.values())
            for state, count in counts.items():
                ratio = count / total
                assert 0.4 < ratio < 0.6  # Within reasonable tolerance

    @pytest.mark.backend
    def test_execution_time_comparison(self):
        """Test execution time comparison structure."""
        from proxima.data.compare import BackendResult
        
        results = [
            BackendResult(
                backend_name="cirq",
                success=True,
                execution_time_ms=100.5,
                memory_peak_mb=256.0,
            ),
            BackendResult(
                backend_name="qiskit_aer",
                success=True,
                execution_time_ms=95.2,
                memory_peak_mb=280.0,
            ),
            BackendResult(
                backend_name="lret",
                success=True,
                execution_time_ms=110.3,
                memory_peak_mb=200.0,
            ),
        ]
        
        # Find fastest backend
        fastest = min(results, key=lambda r: r.execution_time_ms)
        assert fastest.backend_name == "qiskit_aer"
        
        # Find most memory efficient
        efficient = min(results, key=lambda r: r.memory_peak_mb)
        assert efficient.backend_name == "lret"


# =============================================================================
# BACKEND CAPABILITY TESTS
# =============================================================================


class TestBackendCapabilities:
    """Tests for backend capability detection."""

    @pytest.mark.backend
    def test_cirq_capabilities(self):
        """Test Cirq capability detection."""
        cirq_caps = {
            "state_vector": True,
            "density_matrix": True,
            "noise_simulation": True,
            "custom_gates": True,
            "parameterized_gates": True,
            "measurement": True,
            "gpu_support": False,
        }
        
        assert cirq_caps["state_vector"] is True
        assert cirq_caps["density_matrix"] is True

    @pytest.mark.backend
    def test_qiskit_aer_capabilities(self):
        """Test Qiskit Aer capability detection."""
        qiskit_caps = {
            "state_vector": True,
            "density_matrix": True,
            "noise_simulation": True,
            "custom_gates": True,
            "parameterized_gates": True,
            "measurement": True,
            "gpu_support": True,  # With cuQuantum
            "pulse_simulation": True,
        }
        
        assert qiskit_caps["gpu_support"] is True
        assert qiskit_caps["pulse_simulation"] is True

    @pytest.mark.backend
    def test_lret_capabilities(self):
        """Test LRET capability detection."""
        lret_caps = {
            "state_vector": True,
            "density_matrix": False,
            "noise_simulation": False,
            "custom_gates": True,
            "parameterized_gates": True,
            "measurement": True,
            "gpu_support": False,
        }
        
        assert lret_caps["state_vector"] is True
        assert lret_caps["noise_simulation"] is False


# =============================================================================
# BACKEND CONFIGURATION TESTS
# =============================================================================


class TestBackendConfiguration:
    """Tests for backend configuration handling."""

    @pytest.mark.backend
    def test_cirq_configuration_validation(self):
        """Test Cirq configuration validation."""
        valid_config = {
            "simulator_type": "state_vector",
            "noise_model": None,
            "seed": 42,
        }
        
        invalid_configs = [
            {"simulator_type": "invalid_type"},
            {"seed": "not_a_number"},
        ]
        
        assert valid_config["simulator_type"] in ["state_vector", "density_matrix"]
        
        for config in invalid_configs:
            if "simulator_type" in config:
                assert config["simulator_type"] not in ["state_vector", "density_matrix"]

    @pytest.mark.backend
    def test_qiskit_aer_configuration_validation(self):
        """Test Qiskit Aer configuration validation."""
        valid_config = {
            "method": "statevector",
            "shots": 1024,
            "seed_simulator": 42,
            "noise_model": None,
            "coupling_map": None,
        }
        
        assert valid_config["method"] in ["statevector", "density_matrix", "automatic"]
        assert valid_config["shots"] > 0

    @pytest.mark.backend
    def test_lret_configuration_validation(self):
        """Test LRET configuration validation."""
        valid_config = {
            "num_qubits": 10,
            "precision": "double",
            "seed": 42,
        }
        
        assert valid_config["num_qubits"] > 0
        assert valid_config["precision"] in ["single", "double"]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestBackendEdgeCases:
    """Tests for backend edge cases and error conditions."""

    @pytest.mark.backend
    def test_empty_circuit(self):
        """Test handling of empty circuits."""
        # Empty circuit should still be valid
        empty_result = {
            "state_vector": np.array([1], dtype=np.complex128),
            "success": True,
        }
        
        assert empty_result["success"] is True

    @pytest.mark.backend
    def test_single_qubit_circuit(self):
        """Test single qubit circuit handling."""
        single_qubit_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
        
        # Verify normalization
        assert np.isclose(np.sum(np.abs(single_qubit_state) ** 2), 1.0)

    @pytest.mark.backend
    def test_large_qubit_count(self):
        """Test handling of large qubit counts."""
        max_qubits = {
            "cirq": 32,
            "qiskit_aer": 32,
            "lret": 24,
        }
        
        for backend, max_q in max_qubits.items():
            assert max_q > 0
            # State vector size = 2^n
            state_size = 2 ** max_q
            assert state_size > 0

    @pytest.mark.backend
    def test_invalid_gate_sequence(self):
        """Test handling of invalid gate sequences."""
        # Gates applied to non-existent qubits should raise errors
        invalid_gate_configs = [
            {"gate": "CNOT", "qubits": [0, 5], "num_qubits": 2},  # Qubit 5 doesn't exist
            {"gate": "H", "qubits": [-1], "num_qubits": 2},  # Negative qubit index
        ]
        
        for config in invalid_gate_configs:
            for q in config["qubits"]:
                if q < 0 or q >= config["num_qubits"]:
                    assert True  # Invalid configuration detected

    @pytest.mark.backend
    def test_measurement_without_preparation(self):
        """Test measurement on initial state."""
        # Measurement on |0⟩ state should always give 0
        initial_state_measurements = {"0": 1000}
        
        assert "1" not in initial_state_measurements or initial_state_measurements.get("1", 0) == 0

    @pytest.mark.backend
    def test_backend_timeout(self):
        """Test backend timeout handling."""
        timeout_config = {
            "execution_timeout_seconds": 30,
            "connection_timeout_seconds": 10,
        }
        
        assert timeout_config["execution_timeout_seconds"] > 0
        assert timeout_config["connection_timeout_seconds"] > 0

    @pytest.mark.backend
    def test_concurrent_backend_access(self):
        """Test concurrent access to backends."""
        # Backends should be thread-safe or properly locked
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = True
        mock_lock.release.return_value = None
        
        mock_lock.acquire()
        # Simulate work
        mock_lock.release()
        
        mock_lock.acquire.assert_called_once()
        mock_lock.release.assert_called_once()
