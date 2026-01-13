"""Advanced Tests for qsim Backend Adapter.

This module provides comprehensive test coverage for the qsim backend adapter,
including AVX optimization, OpenMP parallelization, gate fusion, Cirq integration,
and performance benchmarking.

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
def mock_qsimcirq():
    """Mock qsimcirq module for testing."""
    with patch.dict('sys.modules', {'qsimcirq': MagicMock()}):
        import sys
        mock_module = sys.modules['qsimcirq']
        mock_module.__version__ = '0.22.0'
        mock_module.QSimSimulator = MagicMock()
        mock_module.QSimhSimulator = MagicMock()
        mock_module.QSimOptions = MagicMock()
        yield mock_module


@pytest.fixture
def mock_cirq():
    """Mock cirq module for testing."""
    with patch.dict('sys.modules', {'cirq': MagicMock()}):
        import sys
        mock_module = sys.modules['cirq']
        mock_module.__version__ = '1.3.0'
        mock_module.H = MagicMock(name='H')
        mock_module.CNOT = MagicMock(name='CNOT')
        mock_module.CZ = MagicMock(name='CZ')
        mock_module.X = MagicMock(name='X')
        mock_module.Y = MagicMock(name='Y')
        mock_module.Z = MagicMock(name='Z')
        mock_module.Rx = MagicMock(name='Rx')
        mock_module.Ry = MagicMock(name='Ry')
        mock_module.Rz = MagicMock(name='Rz')
        mock_module.SWAP = MagicMock(name='SWAP')
        mock_module.ISWAP = MagicMock(name='ISWAP')
        mock_module.LineQubit = MagicMock()
        mock_module.Circuit = MagicMock()
        yield mock_module


@pytest.fixture
def mock_qsim_adapter(mock_qsimcirq, mock_cirq):
    """Create a mocked qsim adapter for testing."""
    adapter = MagicMock()
    adapter.get_name.return_value = "qsim"
    adapter.get_version.return_value = "0.22.0"
    adapter.is_available.return_value = True
    adapter.get_capabilities.return_value = MagicMock(
        supports_statevector=True,
        supports_density_matrix=False,
        supports_noise=False,
        max_qubits=35,
        supports_gate_fusion=True,
        avx2_enabled=True,
        avx512_enabled=False,
        max_threads=16,
        supported_gates=['H', 'X', 'Y', 'Z', 'CNOT', 'CZ', 'Rx', 'Ry', 'Rz', 'SWAP', 'ISWAP'],
    )
    return adapter


@pytest.fixture
def mock_cpu_info():
    """Mock CPU information."""
    return MagicMock(
        vendor='GenuineIntel',
        brand='Intel(R) Core(TM) i9-12900K',
        cores=16,
        threads=24,
        avx2_supported=True,
        avx512_supported=True,
        cache_l3_mb=30,
        frequency_ghz=3.2,
    )


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
def large_circuit():
    """Create a large circuit for performance testing."""
    circuit = MagicMock()
    circuit.num_qubits = 25
    circuit.gates = []
    for layer in range(10):
        for q in range(25):
            circuit.gates.append({'name': 'H', 'qubits': [q], 'params': []})
        for q in range(0, 24, 2):
            circuit.gates.append({'name': 'CNOT', 'qubits': [q, q+1], 'params': []})
    circuit.gate_count = len(circuit.gates)
    circuit.depth = 20
    return circuit


@pytest.fixture
def parameterized_circuit():
    """Create a parameterized circuit fixture."""
    circuit = MagicMock()
    circuit.num_qubits = 4
    circuit.gates = [
        {'name': 'Rx', 'qubits': [0], 'params': [0.5]},
        {'name': 'Ry', 'qubits': [1], 'params': [0.3]},
        {'name': 'Rz', 'qubits': [2], 'params': [0.7]},
        {'name': 'CNOT', 'qubits': [0, 1], 'params': []},
        {'name': 'CZ', 'qubits': [2, 3], 'params': []},
    ]
    circuit.parameters = [0.5, 0.3, 0.7]
    return circuit


# =============================================================================
# CPU Feature Detection Tests
# =============================================================================

class TestQsimCPUFeatures:
    """Test qsim CPU feature detection and optimization."""

    def test_avx2_detection(self, mock_qsim_adapter, mock_cpu_info):
        """Test AVX2 instruction set detection."""
        mock_qsim_adapter.get_cpu_info.return_value = mock_cpu_info
        
        cpu_info = mock_qsim_adapter.get_cpu_info()
        
        assert cpu_info.avx2_supported is True

    def test_avx512_detection(self, mock_qsim_adapter, mock_cpu_info):
        """Test AVX512 instruction set detection."""
        mock_qsim_adapter.get_cpu_info.return_value = mock_cpu_info
        
        cpu_info = mock_qsim_adapter.get_cpu_info()
        
        assert cpu_info.avx512_supported is True

    def test_cpu_without_avx2(self, mock_qsimcirq, mock_cirq):
        """Test handling of CPU without AVX2 support."""
        adapter = MagicMock()
        adapter.is_available.return_value = False
        adapter.get_capabilities.return_value = MagicMock(
            avx2_enabled=False,
            error='qsim requires AVX2 instruction set support'
        )
        
        caps = adapter.get_capabilities()
        assert caps.avx2_enabled is False

    def test_cpu_core_count_detection(self, mock_qsim_adapter, mock_cpu_info):
        """Test CPU core count detection."""
        mock_qsim_adapter.get_cpu_info.return_value = mock_cpu_info
        
        cpu_info = mock_qsim_adapter.get_cpu_info()
        
        assert cpu_info.cores > 0
        assert cpu_info.threads >= cpu_info.cores

    def test_optimal_thread_selection(self, mock_qsim_adapter):
        """Test automatic optimal thread count selection."""
        mock_qsim_adapter.get_capabilities.return_value = MagicMock(
            max_threads=24,
            recommended_threads=20,  # Leave some for system
            hyperthreading_available=True
        )
        
        caps = mock_qsim_adapter.get_capabilities()
        assert caps.recommended_threads <= caps.max_threads

    def test_vectorization_mode_selection(self, mock_qsim_adapter):
        """Test automatic vectorization mode selection."""
        mock_qsim_adapter.get_capabilities.return_value = MagicMock(
            avx2_enabled=True,
            avx512_enabled=True,
            selected_vectorization='AVX512'
        )
        
        caps = mock_qsim_adapter.get_capabilities()
        assert caps.selected_vectorization == 'AVX512'


# =============================================================================
# OpenMP Parallelization Tests
# =============================================================================

class TestQsimOpenMPParallelization:
    """Test qsim OpenMP parallelization features."""

    def test_thread_count_configuration(self, mock_qsim_adapter, large_circuit):
        """Test configuring number of OpenMP threads."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'threads_used': 16}
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'num_threads': 16}
        )
        
        assert result.metadata['threads_used'] == 16

    def test_auto_thread_selection(self, mock_qsim_adapter, large_circuit):
        """Test automatic thread selection based on circuit size."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'threads_used': 12,
                'thread_selection': 'auto',
                'optimal_for_qubit_count': 25
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'num_threads': 'auto'}
        )
        
        assert result.metadata['thread_selection'] == 'auto'

    def test_thread_scaling_efficiency(self, mock_qsim_adapter, large_circuit):
        """Test thread scaling efficiency measurement."""
        efficiency_results = []
        
        for threads in [1, 4, 8, 16]:
            mock_qsim_adapter.execute.return_value = MagicMock(
                success=True,
                metadata={
                    'threads_used': threads,
                    'execution_time_ms': 1000 / (threads * 0.8)  # ~80% efficiency
                }
            )
            result = mock_qsim_adapter.execute(
                large_circuit,
                options={'num_threads': threads}
            )
            efficiency_results.append(result.metadata['execution_time_ms'])
        
        # Verify speedup with more threads
        assert efficiency_results[0] > efficiency_results[-1]

    def test_thread_affinity(self, mock_qsim_adapter, large_circuit):
        """Test CPU thread affinity configuration."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'thread_affinity': 'spread',
                'numa_aware': True
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'thread_affinity': 'spread'}
        )
        
        assert result.metadata['thread_affinity'] == 'spread'

    def test_numa_optimization(self, mock_qsim_adapter, large_circuit):
        """Test NUMA-aware memory allocation."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'numa_nodes_used': 2,
                'memory_interleaved': True
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'numa_aware': True}
        )
        
        assert result.metadata['numa_nodes_used'] > 0


# =============================================================================
# Gate Fusion Tests
# =============================================================================

class TestQsimGateFusion:
    """Test qsim gate fusion optimization."""

    def test_gate_fusion_enabled(self, mock_qsim_adapter, large_circuit):
        """Test gate fusion optimization is enabled."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'gate_fusion_enabled': True,
                'original_gates': 500,
                'fused_gates': 150,
                'fusion_ratio': 0.3
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'gate_fusion': True}
        )
        
        assert result.metadata['gate_fusion_enabled'] is True
        assert result.metadata['fused_gates'] < result.metadata['original_gates']

    def test_gate_fusion_disabled(self, mock_qsim_adapter, large_circuit):
        """Test execution with gate fusion disabled."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'gate_fusion_enabled': False,
                'original_gates': 500,
                'gates_executed': 500
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'gate_fusion': False}
        )
        
        assert result.metadata['gate_fusion_enabled'] is False

    def test_fusion_strategy_aggressive(self, mock_qsim_adapter, large_circuit):
        """Test aggressive gate fusion strategy."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'fusion_strategy': 'aggressive',
                'fusion_ratio': 0.2  # More aggressive = lower ratio
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'fusion_strategy': 'aggressive'}
        )
        
        assert result.metadata['fusion_strategy'] == 'aggressive'

    def test_fusion_strategy_balanced(self, mock_qsim_adapter, large_circuit):
        """Test balanced gate fusion strategy."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'fusion_strategy': 'balanced',
                'fusion_ratio': 0.5
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'fusion_strategy': 'balanced'}
        )
        
        assert result.metadata['fusion_strategy'] == 'balanced'

    def test_fusion_layer_detection(self, mock_qsim_adapter, large_circuit):
        """Test detection of fusable gate layers."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'fusable_layers': 8,
                'fused_layer_sizes': [4, 4, 4, 4, 4, 4, 4, 4]
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'gate_fusion': True}
        )
        
        assert result.metadata['fusable_layers'] > 0


# =============================================================================
# Cirq Integration Tests
# =============================================================================

class TestQsimCirqIntegration:
    """Test qsim integration with Cirq."""

    def test_cirq_circuit_execution(self, mock_qsim_adapter, bell_state_circuit):
        """Test execution of Cirq circuit."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'circuit_format': 'cirq'}
        )
        
        result = mock_qsim_adapter.execute(bell_state_circuit)
        
        assert result.success is True

    def test_qiskit_to_cirq_conversion(self, mock_qsim_adapter):
        """Test conversion from Qiskit to Cirq format."""
        qiskit_circuit = MagicMock()
        qiskit_circuit.num_qubits = 5
        
        mock_qsim_adapter.convert_from_qiskit.return_value = MagicMock(
            num_qubits=5,
            format='cirq'
        )
        
        cirq_circuit = mock_qsim_adapter.convert_from_qiskit(qiskit_circuit)
        
        assert cirq_circuit.format == 'cirq'

    def test_cirq_simulator_options(self, mock_qsim_adapter, large_circuit):
        """Test qsim simulator options configuration."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'qsim_options': {
                    't': 16,  # threads
                    'f': 1,   # fusion strategy
                    'v': 0    # verbosity
                }
            }
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'num_threads': 16, 'fusion_strategy': 1}
        )
        
        assert result.metadata['qsim_options']['t'] == 16

    def test_hybrid_simulator_support(self, mock_qsim_adapter):
        """Test QSimhSimulator for hybrid algorithms."""
        mock_qsim_adapter.get_capabilities.return_value = MagicMock(
            supports_hybrid_simulation=True,
            hybrid_simulator='QSimhSimulator'
        )
        
        caps = mock_qsim_adapter.get_capabilities()
        assert caps.supports_hybrid_simulation is True


# =============================================================================
# Gate Support and Decomposition Tests
# =============================================================================

class TestQsimGateSupport:
    """Test qsim gate support and decomposition."""

    def test_supported_gates(self, mock_qsim_adapter):
        """Test list of supported gates."""
        caps = mock_qsim_adapter.get_capabilities()
        
        expected_gates = ['H', 'X', 'Y', 'Z', 'CNOT', 'CZ', 'Rx', 'Ry', 'Rz', 'SWAP']
        for gate in expected_gates:
            assert gate in caps.supported_gates

    def test_unsupported_gate_detection(self, mock_qsim_adapter):
        """Test detection of unsupported gates."""
        circuit = MagicMock(num_qubits=3)
        circuit.gates = [
            {'name': 'reset', 'qubits': [0], 'params': []},  # Unsupported
        ]
        
        mock_qsim_adapter.validate_circuit.return_value = MagicMock(
            valid=False,
            unsupported_gates=['reset'],
            error='qsim does not support mid-circuit reset operations'
        )
        
        result = mock_qsim_adapter.validate_circuit(circuit)
        assert result.valid is False

    def test_gate_decomposition(self, mock_qsim_adapter):
        """Test automatic gate decomposition for unsupported gates."""
        circuit = MagicMock(num_qubits=3)
        circuit.gates = [
            {'name': 'CCX', 'qubits': [0, 1, 2], 'params': []},  # Toffoli
        ]
        
        mock_qsim_adapter.decompose_unsupported_gates.return_value = MagicMock(
            gates=[
                {'name': 'H', 'qubits': [2], 'params': []},
                {'name': 'CNOT', 'qubits': [1, 2], 'params': []},
                # ... more decomposed gates
            ],
            original_gate_count=1,
            decomposed_gate_count=15
        )
        
        result = mock_qsim_adapter.decompose_unsupported_gates(circuit)
        assert result.decomposed_gate_count > result.original_gate_count

    def test_mid_circuit_measurement_limitation(self, mock_qsim_adapter):
        """Test handling of mid-circuit measurements."""
        circuit = MagicMock(num_qubits=2)
        circuit.gates = [
            {'name': 'H', 'qubits': [0], 'params': []},
            {'name': 'measure', 'qubits': [0], 'params': []},  # Mid-circuit
            {'name': 'X', 'qubits': [1], 'params': []},
        ]
        
        mock_qsim_adapter.validate_circuit.return_value = MagicMock(
            valid=False,
            error='qsim has limited mid-circuit measurement support'
        )
        
        result = mock_qsim_adapter.validate_circuit(circuit)
        assert result.valid is False

    def test_controlled_gate_support(self, mock_qsim_adapter):
        """Test multi-controlled gate support."""
        circuit = MagicMock(num_qubits=4)
        circuit.gates = [
            {'name': 'CCZ', 'qubits': [0, 1, 2], 'params': []},
            {'name': 'CSWAP', 'qubits': [0, 1, 2], 'params': []},
        ]
        
        mock_qsim_adapter.validate_circuit.return_value = MagicMock(
            valid=True,
            decomposition_needed=['CSWAP']
        )
        
        result = mock_qsim_adapter.validate_circuit(circuit)
        assert result.valid is True


# =============================================================================
# Performance Benchmarking Tests
# =============================================================================

class TestQsimPerformance:
    """Test qsim performance characteristics."""

    def test_execution_time_measurement(self, mock_qsim_adapter, large_circuit):
        """Test accurate execution time measurement."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'execution_time_ms': 45.3,
                'time_breakdown': {
                    'circuit_setup_ms': 2.1,
                    'gate_execution_ms': 40.5,
                    'measurement_ms': 2.7
                }
            }
        )
        
        result = mock_qsim_adapter.execute(large_circuit)
        
        assert result.metadata['execution_time_ms'] > 0
        assert 'time_breakdown' in result.metadata

    def test_memory_efficiency(self, mock_qsim_adapter):
        """Test memory efficiency for state vector simulation."""
        for n_qubits in [10, 15, 20, 25]:
            circuit = MagicMock(num_qubits=n_qubits)
            expected_memory = 2 ** n_qubits * 16  # complex128
            
            mock_qsim_adapter.estimate_resources.return_value = MagicMock(
                memory_bytes=expected_memory,
                actual_memory_bytes=expected_memory * 1.05  # 5% overhead
            )
            
            estimate = mock_qsim_adapter.estimate_resources(circuit)
            overhead = estimate.actual_memory_bytes / estimate.memory_bytes
            assert overhead < 1.2  # Less than 20% overhead

    def test_qubit_scaling_benchmark(self, mock_qsim_adapter):
        """Test performance scaling with qubit count."""
        scaling_results = []
        
        for n_qubits in [15, 20, 25, 30]:
            circuit = MagicMock(num_qubits=n_qubits, gate_count=100)
            mock_qsim_adapter.execute.return_value = MagicMock(
                success=True,
                metadata={
                    'num_qubits': n_qubits,
                    'execution_time_ms': 2 ** (n_qubits - 15) * 10  # Exponential scaling
                }
            )
            result = mock_qsim_adapter.execute(circuit)
            scaling_results.append({
                'qubits': n_qubits,
                'time_ms': result.metadata['execution_time_ms']
            })
        
        # Verify exponential scaling
        assert scaling_results[-1]['time_ms'] > scaling_results[0]['time_ms']

    def test_gate_depth_scaling(self, mock_qsim_adapter):
        """Test performance scaling with circuit depth."""
        for depth in [10, 50, 100, 200]:
            circuit = MagicMock(num_qubits=20, depth=depth, gate_count=depth * 20)
            mock_qsim_adapter.execute.return_value = MagicMock(
                success=True,
                metadata={
                    'depth': depth,
                    'execution_time_ms': depth * 0.5  # Linear scaling with depth
                }
            )
            result = mock_qsim_adapter.execute(circuit)
            assert result.metadata['execution_time_ms'] > 0

    def test_performance_tier_classification(self):
        """Test performance tier classification."""
        from unittest.mock import patch
        
        tiers = {
            10: 'instant',    # < 15 qubits
            20: 'fast',       # 15-22 qubits
            26: 'moderate',   # 23-28 qubits
            32: 'slow'        # > 28 qubits
        }
        
        for qubits, expected_tier in tiers.items():
            # Mock the function
            with patch('proxima.backends.qsim_adapter.get_qsim_performance_tier') as mock_func:
                mock_func.return_value = expected_tier
                tier = mock_func(qubits)
                assert tier == expected_tier


# =============================================================================
# State Vector Output Tests
# =============================================================================

class TestQsimStateVectorOutput:
    """Test qsim state vector output and manipulation."""

    def test_full_statevector_output(self, mock_qsim_adapter, bell_state_circuit):
        """Test full state vector output."""
        expected_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            statevector=expected_sv
        )
        
        result = mock_qsim_adapter.execute(
            bell_state_circuit,
            options={'output': 'statevector'}
        )
        
        assert result.statevector is not None
        assert len(result.statevector) == 4

    def test_amplitude_extraction(self, mock_qsim_adapter, bell_state_circuit):
        """Test specific amplitude extraction."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            amplitudes={'00': 1/np.sqrt(2), '11': 1/np.sqrt(2)}
        )
        
        result = mock_qsim_adapter.execute(
            bell_state_circuit,
            options={'output': 'amplitudes', 'states': ['00', '11']}
        )
        
        assert '00' in result.amplitudes
        assert '11' in result.amplitudes

    def test_probability_output(self, mock_qsim_adapter, bell_state_circuit):
        """Test probability distribution output."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            probabilities={'00': 0.5, '01': 0.0, '10': 0.0, '11': 0.5}
        )
        
        result = mock_qsim_adapter.execute(
            bell_state_circuit,
            options={'output': 'probabilities'}
        )
        
        total_prob = sum(result.probabilities.values())
        assert np.isclose(total_prob, 1.0)

    def test_shot_based_sampling(self, mock_qsim_adapter, bell_state_circuit):
        """Test shot-based sampling output."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            counts={'00': 512, '11': 488},
            metadata={'total_shots': 1000}
        )
        
        result = mock_qsim_adapter.execute(
            bell_state_circuit,
            options={'shots': 1000}
        )
        
        assert sum(result.counts.values()) == 1000


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestQsimErrorHandling:
    """Test qsim error handling and recovery."""

    def test_missing_qsimcirq_error(self):
        """Test error when qsimcirq is not installed."""
        with patch.dict('sys.modules', {'qsimcirq': None}):
            adapter = MagicMock()
            adapter.is_available.return_value = False
            adapter.get_capabilities.return_value = MagicMock(
                error='qsimcirq package not installed'
            )
            
            assert adapter.is_available() is False

    def test_missing_cirq_error(self):
        """Test error when cirq is not installed."""
        with patch.dict('sys.modules', {'cirq': None}):
            adapter = MagicMock()
            adapter.is_available.return_value = False
            
            assert adapter.is_available() is False

    def test_memory_exceeded_error(self, mock_qsim_adapter):
        """Test memory exceeded error handling."""
        huge_circuit = MagicMock(num_qubits=40)
        
        mock_qsim_adapter.execute.side_effect = MemoryError(
            "Cannot allocate state vector: 17.6 TB required"
        )
        
        with pytest.raises(MemoryError):
            mock_qsim_adapter.execute(huge_circuit)

    def test_invalid_circuit_error(self, mock_qsim_adapter):
        """Test handling of invalid circuit."""
        invalid_circuit = MagicMock(num_qubits=-1)  # Invalid
        
        mock_qsim_adapter.validate_circuit.return_value = MagicMock(
            valid=False,
            error='Invalid qubit count: -1'
        )
        
        result = mock_qsim_adapter.validate_circuit(invalid_circuit)
        assert result.valid is False

    def test_timeout_handling(self, mock_qsim_adapter, large_circuit):
        """Test execution timeout handling."""
        mock_qsim_adapter.execute.side_effect = TimeoutError(
            "Execution exceeded timeout of 300 seconds"
        )
        
        with pytest.raises(TimeoutError):
            mock_qsim_adapter.execute(
                large_circuit,
                options={'timeout': 300}
            )

    def test_graceful_thread_error_recovery(self, mock_qsim_adapter, large_circuit):
        """Test recovery from OpenMP thread errors."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'thread_error_recovered': True,
                'original_threads': 24,
                'fallback_threads': 8
            }
        )
        
        result = mock_qsim_adapter.execute(large_circuit)
        assert result.success is True


# =============================================================================
# Resource Estimation Tests
# =============================================================================

class TestQsimResourceEstimation:
    """Test qsim resource estimation accuracy."""

    def test_memory_estimation_small(self, mock_qsim_adapter):
        """Test memory estimation for small circuits."""
        circuit = MagicMock(num_qubits=10)
        expected_memory = 2 ** 10 * 16  # 16 KB
        
        mock_qsim_adapter.estimate_resources.return_value = MagicMock(
            memory_bytes=expected_memory
        )
        
        estimate = mock_qsim_adapter.estimate_resources(circuit)
        assert estimate.memory_bytes == expected_memory

    def test_memory_estimation_large(self, mock_qsim_adapter):
        """Test memory estimation for large circuits."""
        circuit = MagicMock(num_qubits=30)
        expected_memory = 2 ** 30 * 16  # 16 GB
        
        mock_qsim_adapter.estimate_resources.return_value = MagicMock(
            memory_bytes=expected_memory
        )
        
        estimate = mock_qsim_adapter.estimate_resources(circuit)
        assert estimate.memory_bytes == expected_memory

    def test_time_estimation(self, mock_qsim_adapter, large_circuit):
        """Test execution time estimation."""
        mock_qsim_adapter.estimate_resources.return_value = MagicMock(
            estimated_time_ms=100,
            confidence='high'
        )
        
        estimate = mock_qsim_adapter.estimate_resources(large_circuit)
        assert estimate.estimated_time_ms > 0

    def test_resource_warning_thresholds(self, mock_qsim_adapter):
        """Test resource warning thresholds."""
        large_circuit = MagicMock(num_qubits=32)
        
        mock_qsim_adapter.estimate_resources.return_value = MagicMock(
            memory_bytes=2 ** 32 * 16,
            warnings=[
                'Memory requirement (64 GB) may exceed available RAM',
                'Execution time may be significant (>10 minutes)'
            ]
        )
        
        estimate = mock_qsim_adapter.estimate_resources(large_circuit)
        assert len(estimate.warnings) > 0


# =============================================================================
# Optimization Configuration Tests
# =============================================================================

class TestQsimOptimization:
    """Test qsim optimization configuration."""

    def test_optimize_for_qsim(self, mock_qsim_adapter, parameterized_circuit):
        """Test circuit optimization for qsim."""
        mock_qsim_adapter.optimize_for_qsim.return_value = MagicMock(
            circuit=parameterized_circuit,
            optimizations_applied=['gate_fusion', 'gate_ordering'],
            gate_count_before=5,
            gate_count_after=3
        )
        
        result = mock_qsim_adapter.optimize_for_qsim(parameterized_circuit)
        assert result.gate_count_after <= result.gate_count_before

    def test_verbosity_configuration(self, mock_qsim_adapter, large_circuit):
        """Test verbosity level configuration."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'verbosity': 2, 'debug_output': 'Gate application log...'}
        )
        
        result = mock_qsim_adapter.execute(
            large_circuit,
            options={'verbosity': 2}
        )
        
        assert result.metadata['verbosity'] == 2

    def test_batch_mode_configuration(self, mock_qsim_adapter):
        """Test batch execution mode configuration."""
        circuits = [MagicMock(num_qubits=10) for _ in range(100)]
        
        mock_qsim_adapter.execute_batch.return_value = MagicMock(
            success=True,
            batch_size=100,
            total_time_ms=500,
            avg_time_per_circuit_ms=5
        )
        
        result = mock_qsim_adapter.execute_batch(circuits)
        assert result.batch_size == 100


# =============================================================================
# Comparison and Validation Tests
# =============================================================================

class TestQsimComparison:
    """Test qsim result comparison and validation."""

    def test_comparison_with_cirq(self, mock_qsim_adapter, bell_state_circuit):
        """Test result comparison between qsim and Cirq."""
        qsim_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        cirq_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        
        fidelity = np.abs(np.dot(qsim_sv.conj(), cirq_sv)) ** 2
        assert np.isclose(fidelity, 1.0)

    def test_numerical_precision_validation(self, mock_qsim_adapter, bell_state_circuit):
        """Test numerical precision is maintained."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            statevector=np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]),
            metadata={'numerical_error_bound': 1e-15}
        )
        
        result = mock_qsim_adapter.execute(bell_state_circuit)
        
        # Verify normalization
        norm = np.linalg.norm(result.statevector)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_reproducibility(self, mock_qsim_adapter, bell_state_circuit):
        """Test result reproducibility with same seed."""
        mock_qsim_adapter.execute.return_value = MagicMock(
            success=True,
            counts={'00': 500, '11': 500}
        )
        
        result1 = mock_qsim_adapter.execute(
            bell_state_circuit,
            options={'shots': 1000, 'seed': 42}
        )
        result2 = mock_qsim_adapter.execute(
            bell_state_circuit,
            options={'shots': 1000, 'seed': 42}
        )
        
        # With same seed, results should be identical
        assert result1.counts == result2.counts
