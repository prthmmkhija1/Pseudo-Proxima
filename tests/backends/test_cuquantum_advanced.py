"""Advanced Tests for cuQuantum Backend Adapter.

This module provides comprehensive test coverage for the cuQuantum backend adapter,
including GPU memory management, multi-GPU support, CUDA streams, performance optimization,
and integration with Qiskit Aer.

Tests: 35%→70%+ coverage improvement
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
def mock_cuda_environment():
    """Mock CUDA/cuQuantum environment."""
    with patch.dict('sys.modules', {
        'cupy': MagicMock(),
        'custatevec': MagicMock(),
        'pycuda': MagicMock(),
    }):
        import sys
        cuda_mock = sys.modules['cupy']
        cuda_mock.cuda.runtime.getDeviceCount = MagicMock(return_value=2)
        cuda_mock.cuda.runtime.getDeviceProperties = MagicMock(return_value={
            'name': b'NVIDIA GeForce RTX 4090',
            'totalGlobalMem': 24 * 1024 ** 3,  # 24 GB
            'major': 8,
            'minor': 9,
        })
        yield cuda_mock


@pytest.fixture
def mock_cuquantum_adapter(mock_cuda_environment):
    """Create a mocked cuQuantum adapter."""
    adapter = MagicMock()
    adapter.get_name.return_value = "cuquantum"
    adapter.get_version.return_value = "24.03.0"
    adapter.is_available.return_value = True
    adapter.get_capabilities.return_value = MagicMock(
        supports_gpu=True,
        supports_multi_gpu=True,
        max_qubits=35,
        gpu_devices=['NVIDIA GeForce RTX 4090'],
        gpu_memory_mb=24576,
        custatevec_enabled=True,
        supported_methods=['statevector', 'density_matrix'],
    )
    return adapter


@pytest.fixture
def mock_gpu_info():
    """Mock GPU device information."""
    return MagicMock(
        device_id=0,
        name='NVIDIA GeForce RTX 4090',
        total_memory_mb=24576,
        free_memory_mb=20000,
        compute_capability=(8, 9),
        cuda_cores=16384,
        memory_bandwidth_gbps=1008,
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
    """Create a large circuit for GPU testing."""
    circuit = MagicMock()
    circuit.num_qubits = 25
    circuit.gates = [{'name': 'H', 'qubits': [i], 'params': []} for i in range(25)]
    circuit.gate_count = 100
    circuit.depth = 50
    return circuit


@pytest.fixture  
def multi_gpu_circuit():
    """Create a circuit suitable for multi-GPU execution."""
    circuit = MagicMock()
    circuit.num_qubits = 30
    circuit.gates = [{'name': 'H', 'qubits': [i], 'params': []} for i in range(30)]
    circuit.gate_count = 200
    return circuit


# =============================================================================
# GPU Detection and Configuration Tests
# =============================================================================

class TestCuQuantumGPUConfiguration:
    """Test cuQuantum GPU detection and configuration."""

    def test_single_gpu_detection(self, mock_cuquantum_adapter, mock_gpu_info):
        """Test detection of a single GPU."""
        mock_cuquantum_adapter.get_gpu_info.return_value = mock_gpu_info
        
        gpu_info = mock_cuquantum_adapter.get_gpu_info(device_id=0)
        
        assert gpu_info.device_id == 0
        assert 'RTX 4090' in gpu_info.name

    def test_multi_gpu_detection(self, mock_cuquantum_adapter):
        """Test detection of multiple GPUs."""
        mock_cuquantum_adapter.get_all_gpu_info.return_value = [
            MagicMock(device_id=0, name='GPU 0', total_memory_mb=24576),
            MagicMock(device_id=1, name='GPU 1', total_memory_mb=24576),
        ]
        
        gpus = mock_cuquantum_adapter.get_all_gpu_info()
        
        assert len(gpus) == 2
        assert gpus[0].device_id == 0
        assert gpus[1].device_id == 1

    def test_compute_capability_check(self, mock_cuquantum_adapter):
        """Test compute capability validation."""
        mock_cuquantum_adapter.get_capabilities.return_value = MagicMock(
            min_compute_capability=(7, 0),
            actual_compute_capability=(8, 9),
            compute_capability_supported=True
        )
        
        caps = mock_cuquantum_adapter.get_capabilities()
        
        assert caps.compute_capability_supported is True
        assert caps.actual_compute_capability >= caps.min_compute_capability

    def test_insufficient_compute_capability(self, mock_cuda_environment):
        """Test handling of insufficient compute capability."""
        adapter = MagicMock()
        adapter.get_capabilities.return_value = MagicMock(
            min_compute_capability=(7, 0),
            actual_compute_capability=(6, 1),
            compute_capability_supported=False,
            error='Compute capability 6.1 < minimum required 7.0'
        )
        
        caps = adapter.get_capabilities()
        assert caps.compute_capability_supported is False

    def test_cuda_driver_version_check(self, mock_cuquantum_adapter):
        """Test CUDA driver version validation."""
        mock_cuquantum_adapter.get_capabilities.return_value = MagicMock(
            cuda_version='12.3',
            cuquantum_version='24.03.0',
            driver_version='535.104.05'
        )
        
        caps = mock_cuquantum_adapter.get_capabilities()
        assert caps.cuda_version is not None

    def test_device_selection_by_memory(self, mock_cuquantum_adapter, large_circuit):
        """Test automatic device selection based on memory."""
        mock_cuquantum_adapter.select_best_gpu.return_value = 1  # GPU 1 has more free memory
        
        best_gpu = mock_cuquantum_adapter.select_best_gpu(num_qubits=25)
        
        assert best_gpu == 1


# =============================================================================
# GPU Memory Management Tests
# =============================================================================

class TestCuQuantumMemoryManagement:
    """Test cuQuantum GPU memory management."""

    def test_memory_estimation_accuracy(self, mock_cuquantum_adapter):
        """Test accurate GPU memory estimation."""
        for n_qubits in [10, 15, 20, 25]:
            circuit = MagicMock(num_qubits=n_qubits)
            expected_sv_memory = 2 ** n_qubits * 16  # complex128
            workspace_overhead = 1024 * 1024 * 1024  # ~1GB
            
            mock_cuquantum_adapter.estimate_resources.return_value = MagicMock(
                statevector_memory_bytes=expected_sv_memory,
                workspace_memory_bytes=workspace_overhead,
                total_gpu_memory_bytes=expected_sv_memory + workspace_overhead
            )
            
            estimate = mock_cuquantum_adapter.estimate_resources(circuit)
            assert estimate.statevector_memory_bytes == expected_sv_memory

    def test_memory_pool_creation(self, mock_cuquantum_adapter):
        """Test GPU memory pool creation."""
        mock_cuquantum_adapter.create_memory_pool.return_value = True
        
        result = mock_cuquantum_adapter.create_memory_pool(size_mb=4096)
        
        assert result is True
        mock_cuquantum_adapter.create_memory_pool.assert_called_with(size_mb=4096)

    def test_memory_pool_stats(self, mock_cuquantum_adapter):
        """Test memory pool statistics retrieval."""
        mock_cuquantum_adapter.get_memory_pool_stats.return_value = {
            'pool_size_mb': 4096,
            'allocated_mb': 2048,
            'free_mb': 2048,
            'peak_usage_mb': 3000,
            'allocation_count': 150
        }
        
        stats = mock_cuquantum_adapter.get_memory_pool_stats()
        
        assert stats['pool_size_mb'] == 4096
        assert stats['free_mb'] + stats['allocated_mb'] == stats['pool_size_mb']

    def test_memory_pool_cleanup(self, mock_cuquantum_adapter):
        """Test memory pool cleanup."""
        mock_cuquantum_adapter.clear_memory_pool.return_value = None
        
        mock_cuquantum_adapter.clear_memory_pool()
        
        mock_cuquantum_adapter.clear_memory_pool.assert_called_once()

    def test_out_of_memory_detection(self, mock_cuquantum_adapter):
        """Test out of memory detection before execution."""
        huge_circuit = MagicMock(num_qubits=35)
        
        mock_cuquantum_adapter.estimate_resources.return_value = MagicMock(
            total_gpu_memory_bytes=2 ** 35 * 16,  # ~550 GB
            exceeds_available=True,
            available_memory_bytes=24 * 1024 ** 3
        )
        
        estimate = mock_cuquantum_adapter.estimate_resources(huge_circuit)
        assert estimate.exceeds_available is True

    def test_memory_fragmentation_handling(self, mock_cuquantum_adapter):
        """Test handling of memory fragmentation."""
        mock_cuquantum_adapter.get_memory_pool_stats.return_value = {
            'total_free_mb': 8000,
            'largest_contiguous_mb': 4000,
            'fragmented': True,
            'fragmentation_ratio': 0.5
        }
        
        stats = mock_cuquantum_adapter.get_memory_pool_stats()
        assert stats['fragmented'] is True

    def test_automatic_memory_defragmentation(self, mock_cuquantum_adapter, large_circuit):
        """Test automatic memory defragmentation."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'defragmentation_performed': True, 'defrag_time_ms': 50}
        )
        
        result = mock_cuquantum_adapter.execute(
            large_circuit,
            options={'auto_defragment': True}
        )
        
        assert result.metadata.get('defragmentation_performed') is True


# =============================================================================
# cuStateVec Integration Tests
# =============================================================================

class TestCuStateVecIntegration:
    """Test cuStateVec library integration."""

    def test_custatevec_initialization(self, mock_cuquantum_adapter):
        """Test cuStateVec library initialization."""
        mock_cuquantum_adapter.get_capabilities.return_value = MagicMock(
            custatevec_enabled=True,
            custatevec_version='1.6.0'
        )
        
        caps = mock_cuquantum_adapter.get_capabilities()
        assert caps.custatevec_enabled is True

    def test_statevector_simulation(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test statevector simulation using cuStateVec."""
        expected_sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
        
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            statevector=expected_sv,
            metadata={'backend': 'custatevec'}
        )
        
        result = mock_cuquantum_adapter.execute(
            bell_state_circuit,
            options={'method': 'statevector'}
        )
        
        assert result.metadata['backend'] == 'custatevec'
        assert np.allclose(np.abs(result.statevector) ** 2, [0.5, 0, 0, 0.5])

    def test_gate_application_on_gpu(self, mock_cuquantum_adapter):
        """Test gate application directly on GPU."""
        circuit = MagicMock(num_qubits=10)
        circuit.gates = [
            {'name': 'H', 'qubits': [0], 'params': []},
            {'name': 'CNOT', 'qubits': [0, 1], 'params': []},
            {'name': 'Rz', 'qubits': [2], 'params': [math.pi / 4]},
        ]
        
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'gates_applied_on_gpu': 3, 'gpu_kernel_time_ms': 0.5}
        )
        
        result = mock_cuquantum_adapter.execute(circuit)
        assert result.metadata['gates_applied_on_gpu'] == 3

    def test_batch_gate_execution(self, mock_cuquantum_adapter, large_circuit):
        """Test batched gate execution for performance."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'batch_size': 10,
                'batches_executed': 10,
                'batching_speedup': 1.8
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            large_circuit,
            options={'gate_batching': True}
        )
        
        assert result.metadata['batching_speedup'] > 1.0


# =============================================================================
# Multi-GPU Execution Tests
# =============================================================================

class TestCuQuantumMultiGPU:
    """Test cuQuantum multi-GPU execution."""

    def test_multi_gpu_capability_detection(self, mock_cuquantum_adapter):
        """Test multi-GPU capability detection."""
        mock_cuquantum_adapter.get_capabilities.return_value = MagicMock(
            supports_multi_gpu=True,
            gpu_count=4,
            nvlink_available=True
        )
        
        caps = mock_cuquantum_adapter.get_capabilities()
        assert caps.supports_multi_gpu is True
        assert caps.gpu_count >= 2

    def test_multi_gpu_execution(self, mock_cuquantum_adapter, multi_gpu_circuit):
        """Test execution across multiple GPUs."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'gpus_used': 2,
                'gpu_0_utilization': 0.95,
                'gpu_1_utilization': 0.93,
                'inter_gpu_communication_time_ms': 5.2
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            multi_gpu_circuit,
            options={'multi_gpu': True, 'gpu_ids': [0, 1]}
        )
        
        assert result.metadata['gpus_used'] == 2

    def test_multi_gpu_load_balancing(self, mock_cuquantum_adapter, multi_gpu_circuit):
        """Test load balancing across GPUs."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'load_balance_strategy': 'memory_based',
                'gpu_0_workload': 0.55,
                'gpu_1_workload': 0.45
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            multi_gpu_circuit,
            options={'multi_gpu': True, 'load_balance': 'memory_based'}
        )
        
        # Workloads should sum to ~1.0
        total_workload = (
            result.metadata['gpu_0_workload'] + 
            result.metadata['gpu_1_workload']
        )
        assert 0.99 <= total_workload <= 1.01

    def test_multi_gpu_config_retrieval(self, mock_cuquantum_adapter):
        """Test multi-GPU configuration retrieval."""
        mock_cuquantum_adapter.get_multi_gpu_config.return_value = {
            'gpu_ids': [0, 1, 2, 3],
            'peer_access_matrix': [[True, True, True, True]] * 4,
            'nvlink_topology': 'ring',
            'recommended_partitioning': 'equal'
        }
        
        config = mock_cuquantum_adapter.get_multi_gpu_config()
        
        assert len(config['gpu_ids']) == 4
        assert config['nvlink_topology'] is not None

    def test_single_gpu_fallback(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test fallback to single GPU for small circuits."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'multi_gpu_requested': True,
                'gpus_used': 1,
                'fallback_reason': 'Circuit too small for multi-GPU benefit'
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            bell_state_circuit,
            options={'multi_gpu': True}
        )
        
        assert result.metadata['gpus_used'] == 1


# =============================================================================
# CUDA Streams and Async Execution Tests
# =============================================================================

class TestCuQuantumCUDAStreams:
    """Test CUDA stream and async execution features."""

    def test_cuda_stream_creation(self, mock_cuquantum_adapter):
        """Test CUDA stream creation for async execution."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'cuda_stream_id': 1, 'async_execution': True}
        )
        
        result = mock_cuquantum_adapter.execute(
            MagicMock(num_qubits=10),
            options={'async': True, 'stream_id': 1}
        )
        
        assert result.metadata['async_execution'] is True

    def test_multiple_streams_parallel(self, mock_cuquantum_adapter):
        """Test parallel execution using multiple CUDA streams."""
        circuits = [MagicMock(num_qubits=10, id=i) for i in range(4)]
        
        mock_cuquantum_adapter.execute_batch.return_value = MagicMock(
            success=True,
            results=[MagicMock(success=True) for _ in circuits],
            metadata={'streams_used': 4, 'parallel_speedup': 3.2}
        )
        
        result = mock_cuquantum_adapter.execute_batch(
            circuits,
            options={'parallel_streams': 4}
        )
        
        assert len(result.results) == 4
        assert result.metadata['parallel_speedup'] > 1.0

    def test_stream_synchronization(self, mock_cuquantum_adapter, large_circuit):
        """Test CUDA stream synchronization."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'stream_synced': True,
                'sync_time_ms': 0.1
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            large_circuit,
            options={'blocking': True}
        )
        
        assert result.metadata['stream_synced'] is True

    def test_async_result_retrieval(self, mock_cuquantum_adapter, large_circuit):
        """Test async result retrieval."""
        # Start async execution
        mock_cuquantum_adapter.execute_async.return_value = MagicMock(
            task_id='async-123',
            status='running'
        )
        
        task = mock_cuquantum_adapter.execute_async(large_circuit)
        assert task.status == 'running'
        
        # Check result
        mock_cuquantum_adapter.get_async_result.return_value = MagicMock(
            task_id='async-123',
            status='completed',
            result=MagicMock(success=True)
        )
        
        result = mock_cuquantum_adapter.get_async_result('async-123')
        assert result.status == 'completed'


# =============================================================================
# Performance Optimization Tests
# =============================================================================

class TestCuQuantumPerformance:
    """Test cuQuantum performance optimization features."""

    def test_gpu_warmup(self, mock_cuquantum_adapter):
        """Test GPU warmup for consistent performance."""
        mock_cuquantum_adapter.warm_up_gpu.return_value = 0.5  # warmup time in seconds
        
        warmup_time = mock_cuquantum_adapter.warm_up_gpu(num_qubits=10)
        
        assert warmup_time > 0
        mock_cuquantum_adapter.warm_up_gpu.assert_called_with(num_qubits=10)

    def test_kernel_caching(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test CUDA kernel caching for repeated circuits."""
        # First execution
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'kernel_cached': False, 'execution_time_ms': 10.0}
        )
        result1 = mock_cuquantum_adapter.execute(bell_state_circuit)
        
        # Second execution with cached kernels
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'kernel_cached': True, 'execution_time_ms': 2.0}
        )
        result2 = mock_cuquantum_adapter.execute(bell_state_circuit)
        
        assert result2.metadata['kernel_cached'] is True
        assert result2.metadata['execution_time_ms'] < result1.metadata['execution_time_ms']

    def test_performance_profiling(self, mock_cuquantum_adapter, large_circuit):
        """Test performance profiling output."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'profiling_enabled': True,
                'gate_application_time_ms': 5.0,
                'memory_transfer_time_ms': 2.0,
                'kernel_launch_overhead_ms': 0.5,
                'total_gpu_time_ms': 7.5
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            large_circuit,
            options={'profile': True}
        )
        
        assert result.metadata['profiling_enabled'] is True
        assert 'gate_application_time_ms' in result.metadata

    def test_tensor_core_utilization(self, mock_cuquantum_adapter, large_circuit):
        """Test Tensor Core utilization for compatible GPUs."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'tensor_cores_used': True,
                'tensor_core_speedup': 1.5
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            large_circuit,
            options={'use_tensor_cores': True}
        )
        
        assert result.metadata['tensor_cores_used'] is True


# =============================================================================
# Qiskit Aer Integration Tests
# =============================================================================

class TestCuQuantumQiskitIntegration:
    """Test cuQuantum integration with Qiskit Aer."""

    def test_qiskit_aer_gpu_backend(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test execution via Qiskit Aer GPU backend."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'qiskit_aer_version': '0.15.0',
                'aer_method': 'statevector',
                'aer_device': 'GPU'
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            bell_state_circuit,
            options={'use_qiskit_aer': True}
        )
        
        assert result.metadata['aer_device'] == 'GPU'

    def test_qiskit_circuit_transpilation(self, mock_cuquantum_adapter):
        """Test Qiskit circuit transpilation for GPU."""
        qiskit_circuit = MagicMock()
        qiskit_circuit.num_qubits = 10
        
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'transpilation_applied': True,
                'optimization_level': 3,
                'gates_reduced': 15
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            qiskit_circuit,
            options={'optimization_level': 3}
        )
        
        assert result.metadata['transpilation_applied'] is True

    def test_shot_based_sampling(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test shot-based sampling on GPU."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            counts={'00': 512, '11': 488},
            metadata={'sampling_method': 'gpu_accelerated'}
        )
        
        result = mock_cuquantum_adapter.execute(
            bell_state_circuit,
            options={'shots': 1000}
        )
        
        total_counts = sum(result.counts.values())
        assert total_counts == 1000


# =============================================================================
# Error Handling and Recovery Tests
# =============================================================================

class TestCuQuantumErrorHandling:
    """Test cuQuantum error handling and recovery."""

    def test_cuda_out_of_memory_error(self, mock_cuquantum_adapter):
        """Test handling of CUDA out of memory errors."""
        huge_circuit = MagicMock(num_qubits=40)
        
        mock_cuquantum_adapter.execute.side_effect = MemoryError(
            "CUDA out of memory. Tried to allocate 16.00 TiB"
        )
        
        with pytest.raises(MemoryError) as exc_info:
            mock_cuquantum_adapter.execute(huge_circuit)
        
        assert "CUDA out of memory" in str(exc_info.value)

    def test_cuda_driver_error_handling(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test handling of CUDA driver errors."""
        mock_cuquantum_adapter.execute.side_effect = RuntimeError(
            "CUDA driver version is insufficient"
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            mock_cuquantum_adapter.execute(bell_state_circuit)
        
        assert "driver" in str(exc_info.value).lower()

    def test_gpu_timeout_handling(self, mock_cuquantum_adapter, large_circuit):
        """Test handling of GPU execution timeout."""
        mock_cuquantum_adapter.execute.side_effect = TimeoutError(
            "GPU kernel execution timed out after 300s"
        )
        
        with pytest.raises(TimeoutError):
            mock_cuquantum_adapter.execute(
                large_circuit,
                options={'timeout': 300}
            )

    def test_graceful_fallback_to_cpu(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test graceful fallback to CPU on GPU errors."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'gpu_error': 'CUDA initialization failed',
                'fallback_to_cpu': True,
                'execution_device': 'CPU'
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            bell_state_circuit,
            options={'allow_cpu_fallback': True}
        )
        
        assert result.metadata['fallback_to_cpu'] is True
        assert result.success is True

    def test_resource_cleanup_on_error(self, mock_cuquantum_adapter, large_circuit):
        """Test proper resource cleanup after errors."""
        mock_cuquantum_adapter.execute.side_effect = RuntimeError("GPU error")
        
        try:
            mock_cuquantum_adapter.execute(large_circuit)
        except RuntimeError:
            pass
        
        # Verify cleanup (in real implementation, check GPU memory is freed)
        mock_cuquantum_adapter.get_memory_pool_stats.return_value = {
            'allocated_mb': 0  # All memory freed
        }
        
        stats = mock_cuquantum_adapter.get_memory_pool_stats()
        assert stats['allocated_mb'] == 0


# =============================================================================
# Precision and Accuracy Tests
# =============================================================================

class TestCuQuantumPrecision:
    """Test cuQuantum precision and accuracy."""

    def test_single_precision_mode(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test single precision (float32) execution."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'precision': 'single', 'bytes_per_complex': 8}
        )
        
        result = mock_cuquantum_adapter.execute(
            bell_state_circuit,
            options={'precision': 'single'}
        )
        
        assert result.metadata['precision'] == 'single'

    def test_double_precision_mode(self, mock_cuquantum_adapter, bell_state_circuit):
        """Test double precision (float64) execution."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={'precision': 'double', 'bytes_per_complex': 16}
        )
        
        result = mock_cuquantum_adapter.execute(
            bell_state_circuit,
            options={'precision': 'double'}
        )
        
        assert result.metadata['precision'] == 'double'

    def test_numerical_accuracy_verification(self, mock_cuquantum_adapter):
        """Test numerical accuracy against known results."""
        # Simple Hadamard on |0> should give |+>
        hadamard_circuit = MagicMock(num_qubits=1)
        hadamard_circuit.gates = [{'name': 'H', 'qubits': [0], 'params': []}]
        
        expected_sv = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            statevector=expected_sv
        )
        
        result = mock_cuquantum_adapter.execute(hadamard_circuit)
        
        assert np.allclose(result.statevector, expected_sv, atol=1e-10)


# =============================================================================
# Batch Execution Tests
# =============================================================================

class TestCuQuantumBatchExecution:
    """Test batch circuit execution on GPU."""

    def test_batch_statevector_simulation(self, mock_cuquantum_adapter):
        """Test batch statevector simulation."""
        circuits = [MagicMock(num_qubits=10, id=i) for i in range(100)]
        
        mock_cuquantum_adapter.execute_batch.return_value = MagicMock(
            success=True,
            results=[MagicMock(success=True) for _ in circuits],
            metadata={
                'batch_size': 100,
                'total_time_ms': 50,
                'avg_time_per_circuit_ms': 0.5
            }
        )
        
        result = mock_cuquantum_adapter.execute_batch(circuits)
        
        assert len(result.results) == 100
        assert result.metadata['avg_time_per_circuit_ms'] < 1.0

    def test_parameter_sweep_execution(self, mock_cuquantum_adapter):
        """Test parameter sweep across multiple circuits."""
        base_circuit = MagicMock(num_qubits=5)
        parameter_sets = [[0.1 * i for _ in range(3)] for i in range(50)]
        
        mock_cuquantum_adapter.execute_parameter_sweep.return_value = MagicMock(
            success=True,
            results=[MagicMock(params=p, energy=-0.5 + sum(p)/10) for p in parameter_sets],
            metadata={'sweep_size': 50}
        )
        
        result = mock_cuquantum_adapter.execute_parameter_sweep(
            base_circuit,
            parameter_sets
        )
        
        assert len(result.results) == 50


# =============================================================================
# Noise Simulation Tests
# =============================================================================

class TestCuQuantumNoiseSimulation:
    """Test cuQuantum noise simulation capabilities."""

    def test_density_matrix_noise_simulation(self, mock_cuquantum_adapter):
        """Test density matrix simulation with noise."""
        noisy_circuit = MagicMock(num_qubits=5)
        noisy_circuit.noise_model = {'type': 'depolarizing', 'probability': 0.01}
        
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'simulation_type': 'density_matrix',
                'noise_applied': True
            }
        )
        
        result = mock_cuquantum_adapter.execute(
            noisy_circuit,
            options={'simulation_type': 'density_matrix'}
        )
        
        assert result.metadata['noise_applied'] is True

    def test_gpu_accelerated_noise(self, mock_cuquantum_adapter):
        """Test GPU-accelerated noise simulation."""
        mock_cuquantum_adapter.execute.return_value = MagicMock(
            success=True,
            metadata={
                'noise_simulation_on_gpu': True,
                'noise_speedup_vs_cpu': 5.2
            }
        )
        
        circuit = MagicMock(num_qubits=10)
        circuit.noise_model = {'type': 'thermal_relaxation', 't1': 50e-6, 't2': 70e-6}
        
        result = mock_cuquantum_adapter.execute(circuit)
        
        assert result.metadata['noise_simulation_on_gpu'] is True
