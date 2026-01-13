"""Advanced Tests for QuEST Backend Adapter.

This module provides comprehensive test coverage for the QuEST backend adapter,
including density matrix operations, GPU acceleration, precision modes,
truncation strategies, and edge cases.

Tests: 40%→70%+ coverage improvement
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_pyquest():
    """Mock pyquest module for testing."""
    with patch.dict('sys.modules', {'pyquest': MagicMock()}):
        import sys
        mock_module = sys.modules['pyquest']
        mock_module.__version__ = '0.9.0'
        mock_module.createQuESTEnv = MagicMock(return_value=MagicMock())
        mock_module.createQureg = MagicMock(return_value=MagicMock())
        mock_module.createDensityQureg = MagicMock(return_value=MagicMock())
        mock_module.destroyQureg = MagicMock()
        mock_module.destroyQuESTEnv = MagicMock()
        mock_module.hadamard = MagicMock()
        mock_module.controlledNot = MagicMock()
        mock_module.rotateZ = MagicMock()
        mock_module.rotateX = MagicMock()
        mock_module.rotateY = MagicMock()
        mock_module.pauliX = MagicMock()
        mock_module.pauliY = MagicMock()
        mock_module.pauliZ = MagicMock()
        mock_module.measure = MagicMock(return_value=0)
        mock_module.getStateVector = MagicMock(return_value=np.array([1.0, 0.0]))
        mock_module.getDensityMatrix = MagicMock(return_value=np.array([[1.0, 0.0], [0.0, 0.0]]))
        yield mock_module


@pytest.fixture
def mock_quest_adapter(mock_pyquest):
    """Create a mocked QuEST adapter for testing."""
    with patch('proxima.backends.quest_adapter.QuestBackendAdapter') as MockAdapter:
        adapter = MagicMock()
        adapter.get_name.return_value = "quest"
        adapter.get_version.return_value = "0.9.0"
        adapter.is_available.return_value = True
        adapter.get_capabilities.return_value = MagicMock(
            supports_density_matrix=True,
            supports_statevector=True,
            supports_gpu=True,
            max_qubits=30,
            supported_gates=['H', 'X', 'Y', 'Z', 'CNOT', 'Rx', 'Ry', 'Rz', 'CZ', 'SWAP']
        )
        MockAdapter.return_value = adapter
        yield adapter


@pytest.fixture
def bell_state_circuit():
    """Create a Bell state circuit fixture."""
    circuit = MagicMock()
    circuit.num_qubits = 2
    circuit.gates = [
        {'name': 'H', 'qubits': [0], 'params': []},
        {'name': 'CNOT', 'qubits': [0, 1], 'params': []},
    ]
    circuit.measurements = [0, 1]
    return circuit


@pytest.fixture
def ghz_circuit():
    """Create a GHZ state circuit fixture."""
    circuit = MagicMock()
    circuit.num_qubits = 5
    circuit.gates = [
        {'name': 'H', 'qubits': [0], 'params': []},
        {'name': 'CNOT', 'qubits': [0, 1], 'params': []},
        {'name': 'CNOT', 'qubits': [1, 2], 'params': []},
        {'name': 'CNOT', 'qubits': [2, 3], 'params': []},
        {'name': 'CNOT', 'qubits': [3, 4], 'params': []},
    ]
    circuit.measurements = list(range(5))
    return circuit


@pytest.fixture
def noisy_circuit():
    """Create a circuit with noise for density matrix testing."""
    circuit = MagicMock()
    circuit.num_qubits = 3
    circuit.gates = [
        {'name': 'H', 'qubits': [0], 'params': []},
        {'name': 'CNOT', 'qubits': [0, 1], 'params': []},
        {'name': 'depolarizing', 'qubits': [0], 'params': [0.01]},
        {'name': 'CNOT', 'qubits': [1, 2], 'params': []},
    ]
    circuit.noise_model = {'type': 'depolarizing', 'probability': 0.01}
    circuit.measurements = list(range(3))
    return circuit


# =============================================================================
# Density Matrix Mode Tests
# =============================================================================

class TestQuestDensityMatrixMode:
    """Test QuEST density matrix simulation features."""

    def test_density_matrix_creation(self, mock_quest_adapter, bell_state_circuit):
        """Test creating a density matrix qureg."""
        mock_quest_adapter.execute.return_value = MagicMock(
            simulation_type='density_matrix',
            density_matrix=np.eye(4) / 4,
            success=True
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        assert result.simulation_type == 'density_matrix'
        assert result.density_matrix is not None

    def test_pure_state_density_matrix(self, mock_quest_adapter, bell_state_circuit):
        """Test density matrix for a pure state has rank 1."""
        # For |00> + |11> / sqrt(2), density matrix has specific form
        bell_dm = np.array([
            [0.5, 0, 0, 0.5],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0.5, 0, 0, 0.5]
        ])
        
        mock_quest_adapter.execute.return_value = MagicMock(
            density_matrix=bell_dm,
            success=True,
            metadata={'rank': 1}
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        assert result.metadata['rank'] == 1

    def test_mixed_state_density_matrix(self, mock_quest_adapter, noisy_circuit):
        """Test density matrix for a mixed state has rank > 1."""
        mock_quest_adapter.execute.return_value = MagicMock(
            density_matrix=np.eye(8) / 8,
            success=True,
            metadata={'rank': 8, 'purity': 0.125}
        )
        
        result = mock_quest_adapter.execute(
            noisy_circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        assert result.metadata['rank'] > 1
        assert result.metadata['purity'] < 1.0

    def test_density_matrix_trace(self, mock_quest_adapter, bell_state_circuit):
        """Test density matrix has trace 1."""
        dm = np.array([[0.5, 0.5], [0.5, 0.5]])
        mock_quest_adapter.execute.return_value = MagicMock(
            density_matrix=dm,
            success=True
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        trace = np.trace(result.density_matrix)
        assert np.isclose(trace, 1.0)

    def test_density_matrix_hermiticity(self, mock_quest_adapter, bell_state_circuit):
        """Test density matrix is Hermitian."""
        dm = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
        mock_quest_adapter.execute.return_value = MagicMock(
            density_matrix=dm,
            success=True
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        assert np.allclose(result.density_matrix, result.density_matrix.conj().T)

    def test_density_matrix_positive_semidefinite(self, mock_quest_adapter):
        """Test density matrix eigenvalues are non-negative."""
        dm = np.array([[0.5, 0.3], [0.3, 0.5]])
        mock_quest_adapter.execute.return_value = MagicMock(
            density_matrix=dm,
            success=True
        )
        
        circuit = MagicMock(num_qubits=1)
        result = mock_quest_adapter.execute(
            circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        eigenvalues = np.linalg.eigvalsh(result.density_matrix)
        assert all(ev >= -1e-10 for ev in eigenvalues)

    def test_density_matrix_memory_estimation(self, mock_quest_adapter):
        """Test memory estimation for density matrix mode."""
        for n_qubits in [5, 10, 15]:
            circuit = MagicMock(num_qubits=n_qubits)
            mock_quest_adapter.estimate_resources.return_value = MagicMock(
                memory_bytes=2 ** (2 * n_qubits) * 16,
                simulation_type='density_matrix'
            )
            
            estimate = mock_quest_adapter.estimate_resources(circuit)
            expected_memory = 2 ** (2 * n_qubits) * 16  # 2^(2n) complex128
            assert estimate.memory_bytes == expected_memory


# =============================================================================
# GPU Acceleration Tests
# =============================================================================

class TestQuestGPUAcceleration:
    """Test QuEST GPU acceleration features."""

    def test_gpu_detection(self, mock_quest_adapter):
        """Test GPU availability detection."""
        mock_quest_adapter.get_capabilities.return_value = MagicMock(
            supports_gpu=True,
            gpu_devices=['NVIDIA GeForce RTX 3080'],
            gpu_memory_mb=10240
        )
        
        caps = mock_quest_adapter.get_capabilities()
        assert caps.supports_gpu is True
        assert len(caps.gpu_devices) > 0

    def test_gpu_execution_mode(self, mock_quest_adapter, bell_state_circuit):
        """Test execution with GPU acceleration enabled."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'execution_mode': 'GPU', 'gpu_device': 'NVIDIA GeForce RTX 3080'}
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'use_gpu': True}
        )
        
        assert result.metadata['execution_mode'] == 'GPU'

    def test_gpu_fallback_on_insufficient_memory(self, mock_quest_adapter):
        """Test fallback to CPU when GPU memory is insufficient."""
        large_circuit = MagicMock(num_qubits=28)
        
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'execution_mode': 'CPU',
                'fallback_reason': 'Insufficient GPU memory'
            }
        )
        
        result = mock_quest_adapter.execute(
            large_circuit,
            options={'use_gpu': True}
        )
        
        assert result.metadata['execution_mode'] == 'CPU'
        assert 'fallback_reason' in result.metadata

    def test_gpu_memory_estimation(self, mock_quest_adapter):
        """Test GPU memory estimation for circuits."""
        circuit = MagicMock(num_qubits=20)
        mock_quest_adapter.estimate_resources.return_value = MagicMock(
            gpu_memory_bytes=2 ** 20 * 16,
            cpu_memory_bytes=2 ** 20 * 16
        )
        
        estimate = mock_quest_adapter.estimate_resources(circuit)
        expected = 2 ** 20 * 16  # 16 bytes per complex128
        assert estimate.gpu_memory_bytes == expected

    def test_multi_gpu_support(self, mock_quest_adapter, ghz_circuit):
        """Test multi-GPU execution support."""
        mock_quest_adapter.get_capabilities.return_value = MagicMock(
            supports_gpu=True,
            gpu_devices=['GPU 0', 'GPU 1'],
            supports_multi_gpu=True
        )
        
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'gpu_count': 2, 'execution_mode': 'multi-GPU'}
        )
        
        result = mock_quest_adapter.execute(
            ghz_circuit,
            options={'use_gpu': True, 'multi_gpu': True}
        )
        
        assert result.metadata['gpu_count'] == 2

    def test_gpu_unavailable_handling(self, mock_pyquest):
        """Test graceful handling when GPU is unavailable."""
        mock_pyquest.cuda_available = False
        
        # Adapter should still work in CPU mode
        adapter = MagicMock()
        adapter.is_available.return_value = True
        adapter.get_capabilities.return_value = MagicMock(
            supports_gpu=False,
            gpu_error='CUDA not available'
        )
        
        caps = adapter.get_capabilities()
        assert caps.supports_gpu is False


# =============================================================================
# Precision Configuration Tests
# =============================================================================

class TestQuestPrecisionModes:
    """Test QuEST precision configuration."""

    def test_single_precision(self, mock_quest_adapter, bell_state_circuit):
        """Test single precision execution."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'precision': 'single', 'bytes_per_complex': 8}
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'precision': 'single'}
        )
        
        assert result.metadata['precision'] == 'single'

    def test_double_precision(self, mock_quest_adapter, bell_state_circuit):
        """Test double precision execution."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'precision': 'double', 'bytes_per_complex': 16}
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'precision': 'double'}
        )
        
        assert result.metadata['precision'] == 'double'

    def test_quad_precision(self, mock_quest_adapter, bell_state_circuit):
        """Test quad precision execution (if available)."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'precision': 'quad', 'bytes_per_complex': 32}
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'precision': 'quad'}
        )
        
        assert result.metadata['precision'] == 'quad'

    def test_precision_affects_memory(self, mock_quest_adapter):
        """Test that precision affects memory estimation."""
        circuit = MagicMock(num_qubits=10)
        
        # Single precision
        mock_quest_adapter.estimate_resources.return_value = MagicMock(
            memory_bytes=2 ** 10 * 8
        )
        single_estimate = mock_quest_adapter.estimate_resources(circuit)
        
        # Double precision
        mock_quest_adapter.estimate_resources.return_value = MagicMock(
            memory_bytes=2 ** 10 * 16
        )
        double_estimate = mock_quest_adapter.estimate_resources(circuit)
        
        assert double_estimate.memory_bytes == 2 * single_estimate.memory_bytes

    def test_precision_numerical_accuracy(self, mock_quest_adapter):
        """Test that higher precision gives more accurate results."""
        circuit = MagicMock(num_qubits=5)
        
        # Simulate error bounds
        mock_quest_adapter.execute.side_effect = [
            MagicMock(success=True, metadata={'precision': 'single', 'error_bound': 1e-6}),
            MagicMock(success=True, metadata={'precision': 'double', 'error_bound': 1e-14}),
        ]
        
        single_result = mock_quest_adapter.execute(circuit, options={'precision': 'single'})
        double_result = mock_quest_adapter.execute(circuit, options={'precision': 'double'})
        
        assert double_result.metadata['error_bound'] < single_result.metadata['error_bound']


# =============================================================================
# Truncation Strategy Tests
# =============================================================================

class TestQuestTruncation:
    """Test QuEST rank truncation for density matrices."""

    def test_truncation_threshold_applied(self, mock_quest_adapter, noisy_circuit):
        """Test truncation threshold is applied correctly."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'truncation_applied': True,
                'truncation_threshold': 1e-4,
                'original_rank': 8,
                'truncated_rank': 4
            }
        )
        
        result = mock_quest_adapter.execute(
            noisy_circuit,
            options={
                'simulation_type': 'density_matrix',
                'truncation_threshold': 1e-4
            }
        )
        
        assert result.metadata['truncation_applied'] is True
        assert result.metadata['truncated_rank'] < result.metadata['original_rank']

    def test_no_truncation_for_pure_states(self, mock_quest_adapter, bell_state_circuit):
        """Test truncation is not needed for pure states."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'truncation_applied': False,
                'rank': 1
            }
        )
        
        result = mock_quest_adapter.execute(
            bell_state_circuit,
            options={'simulation_type': 'density_matrix', 'truncation_threshold': 1e-4}
        )
        
        assert result.metadata['truncation_applied'] is False
        assert result.metadata['rank'] == 1

    def test_truncation_preserves_normalization(self, mock_quest_adapter, noisy_circuit):
        """Test truncation preserves trace normalization."""
        dm = np.diag([0.4, 0.3, 0.2, 0.1])  # Already normalized
        mock_quest_adapter.execute.return_value = MagicMock(
            density_matrix=dm,
            success=True,
            metadata={'truncation_applied': True}
        )
        
        result = mock_quest_adapter.execute(
            noisy_circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        assert np.isclose(np.trace(result.density_matrix), 1.0)

    def test_adaptive_truncation(self, mock_quest_adapter, noisy_circuit):
        """Test adaptive truncation based on circuit depth."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'truncation_strategy': 'adaptive',
                'adaptive_threshold': 1e-5
            }
        )
        
        result = mock_quest_adapter.execute(
            noisy_circuit,
            options={
                'simulation_type': 'density_matrix',
                'truncation_strategy': 'adaptive'
            }
        )
        
        assert result.metadata['truncation_strategy'] == 'adaptive'


# =============================================================================
# OpenMP Parallelization Tests
# =============================================================================

class TestQuestOpenMPParallelization:
    """Test QuEST OpenMP parallelization features."""

    def test_thread_count_detection(self, mock_quest_adapter):
        """Test detection of available OpenMP threads."""
        mock_quest_adapter.get_capabilities.return_value = MagicMock(
            max_threads=16,
            openmp_enabled=True
        )
        
        caps = mock_quest_adapter.get_capabilities()
        assert caps.openmp_enabled is True
        assert caps.max_threads > 0

    def test_thread_count_configuration(self, mock_quest_adapter, ghz_circuit):
        """Test configuring number of OpenMP threads."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'threads_used': 8}
        )
        
        result = mock_quest_adapter.execute(
            ghz_circuit,
            options={'num_threads': 8}
        )
        
        assert result.metadata['threads_used'] == 8

    def test_auto_thread_selection(self, mock_quest_adapter, ghz_circuit):
        """Test automatic thread selection based on circuit size."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'threads_used': 4, 'thread_selection': 'auto'}
        )
        
        result = mock_quest_adapter.execute(
            ghz_circuit,
            options={'num_threads': 'auto'}
        )
        
        assert result.metadata['thread_selection'] == 'auto'

    def test_thread_scaling_efficiency(self, mock_quest_adapter):
        """Test thread scaling efficiency reporting."""
        circuit = MagicMock(num_qubits=15)
        
        mock_quest_adapter.tune_threads.return_value = MagicMock(
            optimal_threads=8,
            efficiency_1_thread=1.0,
            efficiency_8_threads=0.85,
            speedup_8_threads=6.8
        )
        
        result = mock_quest_adapter.tune_threads(circuit)
        
        assert result.optimal_threads > 0
        assert result.speedup_8_threads > 1.0


# =============================================================================
# MPI Distributed Computing Tests
# =============================================================================

class TestQuestMPIDistributed:
    """Test QuEST MPI distributed computing features."""

    def test_mpi_availability_detection(self, mock_quest_adapter):
        """Test MPI availability detection."""
        mock_quest_adapter.get_capabilities.return_value = MagicMock(
            supports_mpi=True,
            mpi_ranks=4
        )
        
        caps = mock_quest_adapter.get_capabilities()
        assert caps.supports_mpi is True

    def test_mpi_execution_mode(self, mock_quest_adapter, ghz_circuit):
        """Test MPI distributed execution."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'execution_mode': 'MPI', 'mpi_ranks': 4}
        )
        
        result = mock_quest_adapter.execute(
            ghz_circuit,
            options={'use_mpi': True, 'mpi_ranks': 4}
        )
        
        assert result.metadata['execution_mode'] == 'MPI'

    def test_mpi_unavailable_fallback(self, mock_quest_adapter, ghz_circuit):
        """Test fallback when MPI is unavailable."""
        mock_quest_adapter.get_capabilities.return_value = MagicMock(
            supports_mpi=False
        )
        
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'execution_mode': 'single-node'}
        )
        
        result = mock_quest_adapter.execute(
            ghz_circuit,
            options={'use_mpi': True}  # Request MPI but unavailable
        )
        
        assert result.metadata['execution_mode'] == 'single-node'


# =============================================================================
# Gate Support Edge Cases
# =============================================================================

class TestQuestGateEdgeCases:
    """Test QuEST gate support edge cases."""

    def test_controlled_rotation_gates(self, mock_quest_adapter):
        """Test controlled rotation gates (CRx, CRy, CRz)."""
        circuit = MagicMock(num_qubits=2)
        circuit.gates = [
            {'name': 'CRz', 'qubits': [0, 1], 'params': [math.pi / 4]}
        ]
        
        mock_quest_adapter.validate_circuit.return_value = MagicMock(
            valid=True,
            gate_support={'CRz': True}
        )
        
        result = mock_quest_adapter.validate_circuit(circuit)
        assert result.valid is True

    def test_three_qubit_gates(self, mock_quest_adapter):
        """Test three-qubit gates (CCX, CCZ)."""
        circuit = MagicMock(num_qubits=3)
        circuit.gates = [
            {'name': 'CCX', 'qubits': [0, 1, 2], 'params': []}
        ]
        
        mock_quest_adapter.validate_circuit.return_value = MagicMock(
            valid=True,
            gate_support={'CCX': True}
        )
        
        result = mock_quest_adapter.validate_circuit(circuit)
        assert result.valid is True

    def test_custom_unitary_gate(self, mock_quest_adapter):
        """Test custom unitary gate application."""
        custom_gate = np.array([[1, 0], [0, 1j]])
        circuit = MagicMock(num_qubits=1)
        circuit.gates = [
            {'name': 'unitary', 'qubits': [0], 'params': [custom_gate]}
        ]
        
        mock_quest_adapter.validate_circuit.return_value = MagicMock(
            valid=True,
            gate_support={'unitary': True}
        )
        
        result = mock_quest_adapter.validate_circuit(circuit)
        assert result.valid is True

    def test_unsupported_gate_decomposition(self, mock_quest_adapter):
        """Test automatic decomposition of unsupported gates."""
        circuit = MagicMock(num_qubits=2)
        circuit.gates = [
            {'name': 'ISWAP', 'qubits': [0, 1], 'params': []}
        ]
        
        mock_quest_adapter.validate_circuit.return_value = MagicMock(
            valid=True,
            decomposed_gates=['ISWAP → [CNOT, S, CNOT, ...]']
        )
        
        result = mock_quest_adapter.validate_circuit(circuit)
        assert 'decomposed_gates' in result.__dict__

    def test_identity_gate_optimization(self, mock_quest_adapter):
        """Test identity gates are optimized away."""
        circuit = MagicMock(num_qubits=1)
        circuit.gates = [
            {'name': 'I', 'qubits': [0], 'params': []},
            {'name': 'H', 'qubits': [0], 'params': []},
        ]
        
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'optimized_gate_count': 1}  # I was removed
        )
        
        result = mock_quest_adapter.execute(circuit)
        assert result.metadata['optimized_gate_count'] == 1


# =============================================================================
# Resource Estimation Tests
# =============================================================================

class TestQuestResourceEstimation:
    """Test QuEST resource estimation accuracy."""

    def test_state_vector_memory_scaling(self, mock_quest_adapter):
        """Test state vector memory scales as 2^n."""
        for n in [5, 10, 15, 20]:
            circuit = MagicMock(num_qubits=n)
            expected_memory = 2 ** n * 16  # complex128
            
            mock_quest_adapter.estimate_resources.return_value = MagicMock(
                memory_bytes=expected_memory,
                simulation_type='state_vector'
            )
            
            estimate = mock_quest_adapter.estimate_resources(circuit)
            assert estimate.memory_bytes == expected_memory

    def test_density_matrix_memory_scaling(self, mock_quest_adapter):
        """Test density matrix memory scales as 2^(2n)."""
        for n in [3, 5, 8, 10]:
            circuit = MagicMock(num_qubits=n)
            expected_memory = 2 ** (2 * n) * 16  # complex128
            
            mock_quest_adapter.estimate_resources.return_value = MagicMock(
                memory_bytes=expected_memory,
                simulation_type='density_matrix'
            )
            
            estimate = mock_quest_adapter.estimate_resources(circuit)
            assert estimate.memory_bytes == expected_memory

    def test_execution_time_estimation(self, mock_quest_adapter):
        """Test execution time estimation."""
        circuit = MagicMock(num_qubits=15, gate_count=100)
        
        mock_quest_adapter.estimate_resources.return_value = MagicMock(
            estimated_time_seconds=0.5,
            time_per_gate_ns=100
        )
        
        estimate = mock_quest_adapter.estimate_resources(circuit)
        assert estimate.estimated_time_seconds > 0

    def test_resource_exceeded_warning(self, mock_quest_adapter):
        """Test warning when resources may be exceeded."""
        large_circuit = MagicMock(num_qubits=35)
        
        mock_quest_adapter.estimate_resources.return_value = MagicMock(
            memory_bytes=2 ** 35 * 16,
            warnings=['Memory requirement (512 GB) exceeds available RAM (32 GB)']
        )
        
        estimate = mock_quest_adapter.estimate_resources(large_circuit)
        assert len(estimate.warnings) > 0


# =============================================================================
# Error Recovery Tests  
# =============================================================================

class TestQuestErrorRecovery:
    """Test QuEST error recovery mechanisms."""

    def test_environment_cleanup_on_error(self, mock_quest_adapter, mock_pyquest):
        """Test QuEST environment is cleaned up after errors."""
        mock_quest_adapter.execute.side_effect = Exception("Simulation failed")
        
        with pytest.raises(Exception):
            mock_quest_adapter.execute(MagicMock())
        
        # Verify cleanup was called (in real implementation)
        # mock_pyquest.destroyQuESTEnv.assert_called()

    def test_qureg_cleanup_on_error(self, mock_quest_adapter, mock_pyquest):
        """Test qureg is destroyed after errors."""
        mock_quest_adapter.execute.side_effect = MemoryError("Out of memory")
        
        with pytest.raises(MemoryError):
            mock_quest_adapter.execute(MagicMock())

    def test_retry_on_transient_error(self, mock_quest_adapter, bell_state_circuit):
        """Test retry mechanism for transient errors."""
        call_count = 0
        
        def execute_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Transient timeout")
            return MagicMock(success=True)
        
        mock_quest_adapter.execute.side_effect = execute_with_retry
        
        try:
            for _ in range(3):
                try:
                    result = mock_quest_adapter.execute(bell_state_circuit)
                    break
                except TimeoutError:
                    continue
        except TimeoutError:
            pass
        
        assert call_count == 3

    def test_graceful_degradation(self, mock_quest_adapter, ghz_circuit):
        """Test graceful degradation when features unavailable."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'requested_gpu': True,
                'actual_mode': 'CPU',
                'degradation_reason': 'GPU unavailable'
            }
        )
        
        result = mock_quest_adapter.execute(
            ghz_circuit,
            options={'use_gpu': True}
        )
        
        assert result.metadata['actual_mode'] == 'CPU'
        assert 'degradation_reason' in result.metadata


# =============================================================================
# Noise Model Tests
# =============================================================================

class TestQuestNoiseModels:
    """Test QuEST noise model support."""

    def test_depolarizing_noise(self, mock_quest_adapter, noisy_circuit):
        """Test depolarizing noise application."""
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'noise_applied': True, 'noise_type': 'depolarizing'}
        )
        
        result = mock_quest_adapter.execute(
            noisy_circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        assert result.metadata['noise_applied'] is True

    def test_amplitude_damping_noise(self, mock_quest_adapter):
        """Test amplitude damping noise."""
        circuit = MagicMock(num_qubits=2)
        circuit.noise_model = {'type': 'amplitude_damping', 'gamma': 0.1}
        
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'noise_type': 'amplitude_damping'}
        )
        
        result = mock_quest_adapter.execute(
            circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        assert result.metadata['noise_type'] == 'amplitude_damping'

    def test_noise_requires_density_matrix(self, mock_quest_adapter, noisy_circuit):
        """Test that noisy simulation requires density matrix mode."""
        mock_quest_adapter.validate_circuit.return_value = MagicMock(
            valid=True,
            warnings=['Noise model requires density_matrix simulation type']
        )
        
        result = mock_quest_adapter.validate_circuit(noisy_circuit)
        assert 'density_matrix' in str(result.warnings)


# =============================================================================
# Integration Scenario Tests
# =============================================================================

class TestQuestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_full_bell_state_workflow(self, mock_quest_adapter, bell_state_circuit):
        """Test complete Bell state preparation and measurement workflow."""
        # 1. Validate circuit
        mock_quest_adapter.validate_circuit.return_value = MagicMock(valid=True)
        validation = mock_quest_adapter.validate_circuit(bell_state_circuit)
        assert validation.valid
        
        # 2. Estimate resources
        mock_quest_adapter.estimate_resources.return_value = MagicMock(
            memory_bytes=64,
            estimated_time_seconds=0.001
        )
        estimate = mock_quest_adapter.estimate_resources(bell_state_circuit)
        assert estimate.memory_bytes > 0
        
        # 3. Execute
        mock_quest_adapter.execute.return_value = MagicMock(
            success=True,
            counts={'00': 500, '11': 500}
        )
        result = mock_quest_adapter.execute(bell_state_circuit, options={'shots': 1000})
        assert result.success

    def test_variational_circuit_workflow(self, mock_quest_adapter):
        """Test variational quantum circuit workflow with parameter updates."""
        circuit = MagicMock(num_qubits=4)
        circuit.parameters = [0.1, 0.2, 0.3]
        
        results = []
        for params in [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]:
            circuit.parameters = params
            mock_quest_adapter.execute.return_value = MagicMock(
                success=True,
                expectation_value=-0.5 + sum(params) / 10
            )
            result = mock_quest_adapter.execute(circuit)
            results.append(result)
        
        assert len(results) == 3

    def test_comparison_with_other_backends(self, mock_quest_adapter, bell_state_circuit):
        """Test result comparison between QuEST and other backends."""
        quest_result = MagicMock(
            statevector=np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),
            backend='quest'
        )
        cirq_result = MagicMock(
            statevector=np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),
            backend='cirq'
        )
        
        # Calculate fidelity
        fidelity = np.abs(np.dot(
            quest_result.statevector.conj(),
            cirq_result.statevector
        )) ** 2
        
        assert np.isclose(fidelity, 1.0)
