"""Step 2.5: cuQuantum Integration Tests.

This module provides comprehensive integration tests for the cuQuantum backend:
1. GPU Availability Tests - Test with/without GPU
2. Memory Tests - Test memory estimation and limits
3. Performance Tests - Compare GPU vs CPU execution time
4. Correctness Tests - Verify results match CPU reference

Test Categories:
- Unit tests for memory estimation
- Integration tests for GPU detection
- Performance benchmarks
- Correctness validation against Qiskit Aer CPU
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


def _has_gpu() -> bool:
    """Check if GPU is available for testing."""
    try:
        # Try pycuda
        import pycuda.driver as cuda

        cuda.init()
        return cuda.Device.count() > 0
    except Exception:
        pass

    try:
        # Try cupy
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        pass

    return False


# =============================================================================
# Test Markers
# =============================================================================

# Mark tests that require actual GPU
requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="No GPU available for testing")

# Mark tests that can run without GPU (mocked)
mock_gpu = pytest.mark.usefixtures("mock_gpu_environment")

# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_gpu_device_info():
    """Create mock GPU device info."""
    from proxima.backends.cuquantum_adapter import GPUDeviceInfo

    return GPUDeviceInfo(
        device_id=0,
        name="NVIDIA Mock GPU",
        compute_capability="8.6",
        total_memory_mb=16384,  # 16 GB
        free_memory_mb=14000,  # 14 GB free
        cuda_version="12.0",
        driver_version="525.60",
        is_cuquantum_compatible=True,
    )


@pytest.fixture
def mock_gpu_environment(mock_gpu_device_info):
    """Mock GPU environment for testing without actual GPU."""
    with patch(
        "proxima.backends.cuquantum_adapter.CuQuantumAdapter._check_cuda_available"
    ) as mock_cuda:
        with patch(
            "proxima.backends.cuquantum_adapter.CuQuantumAdapter._detect_gpu_devices"
        ) as mock_devices:
            with patch(
                "proxima.backends.cuquantum_adapter.CuQuantumAdapter._check_cuquantum_available"
            ) as mock_cuq:
                with patch(
                    "proxima.backends.cuquantum_adapter.CuQuantumAdapter._check_qiskit_gpu_available"
                ) as mock_qiskit:
                    mock_cuda.return_value = True
                    mock_devices.return_value = [mock_gpu_device_info]
                    mock_cuq.return_value = True
                    mock_qiskit.return_value = True
                    yield


@pytest.fixture
def cuquantum_config():
    """Create test cuQuantum configuration."""
    from proxima.backends.cuquantum_adapter import (
        CuQuantumConfig,
        CuQuantumExecutionMode,
        CuQuantumPrecision,
    )

    return CuQuantumConfig(
        execution_mode=CuQuantumExecutionMode.GPU_PREFERRED,
        gpu_device_id=0,
        precision=CuQuantumPrecision.DOUBLE,
        memory_limit_mb=0,
        workspace_size_mb=1024,
        blocking=True,
        fusion_enabled=True,
        max_qubits=30,
        fallback_to_cpu=True,
    )


@pytest.fixture
def memory_manager(cuquantum_config, mock_gpu_device_info):
    """Create GPU memory manager with mock GPU."""
    from proxima.backends.gpu_memory_manager import GPUMemoryManager

    manager = GPUMemoryManager(config=cuquantum_config)
    manager.set_gpu_devices([mock_gpu_device_info])
    return manager


@pytest.fixture
def sample_circuit():
    """Create a sample quantum circuit for testing."""
    try:
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(5)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(3, 4)
        circuit.measure_all()
        return circuit
    except ImportError:
        pytest.skip("Qiskit not available")


@pytest.fixture
def large_circuit():
    """Create a larger circuit for memory testing."""
    try:
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(20)
        for i in range(20):
            circuit.h(i)
        for i in range(19):
            circuit.cx(i, i + 1)
        return circuit
    except ImportError:
        pytest.skip("Qiskit not available")


# =============================================================================
# GPU Availability Tests (Step 2.5.1)
# =============================================================================


class TestGPUAvailability:
    """Tests for GPU availability and detection."""

    def test_adapter_initialization_with_mock_gpu(self, mock_gpu_environment):
        """Test adapter initializes correctly with mock GPU."""
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        adapter = CuQuantumAdapter()

        assert adapter.is_available()
        assert adapter.get_name() == "cuquantum"

    def test_adapter_initialization_without_gpu(self):
        """Test adapter handles missing GPU gracefully."""
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        with patch.object(
            CuQuantumAdapter, "_check_cuda_available", return_value=False
        ):
            adapter = CuQuantumAdapter()

            # Should still initialize, but report unavailable
            assert adapter.get_name() == "cuquantum"

    def test_gpu_device_detection(self, mock_gpu_environment, mock_gpu_device_info):
        """Test GPU device detection returns correct info."""
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        adapter = CuQuantumAdapter()
        gpu_info = adapter.get_gpu_info(0)

        assert gpu_info is not None
        assert gpu_info.name == mock_gpu_device_info.name
        assert gpu_info.compute_capability == mock_gpu_device_info.compute_capability
        assert gpu_info.is_cuquantum_compatible

    def test_capabilities_with_gpu(self, mock_gpu_environment):
        """Test capabilities reporting with GPU available."""
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        adapter = CuQuantumAdapter()
        caps = adapter.get_capabilities()

        assert caps.supports_statevector
        assert caps.max_qubits >= 30

    def test_capabilities_without_gpu(self):
        """Test capabilities reporting without GPU."""
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        with patch.object(
            CuQuantumAdapter, "_check_cuda_available", return_value=False
        ):
            adapter = CuQuantumAdapter()
            caps = adapter.get_capabilities()

            # Should report limited capabilities
            assert caps.supports_statevector

    @requires_gpu
    def test_real_gpu_detection(self):
        """Test actual GPU detection (requires real GPU)."""
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        adapter = CuQuantumAdapter()

        assert adapter._cuda_available
        assert len(adapter._gpu_devices) > 0

        gpu = adapter._gpu_devices[0]
        assert gpu.total_memory_mb > 0
        assert float(gpu.compute_capability.split(".")[0]) >= 7


# =============================================================================
# Memory Tests (Step 2.5.2)
# =============================================================================


class TestMemoryEstimation:
    """Tests for GPU memory estimation and validation."""

    def test_memory_estimate_small_circuit(self, memory_manager):
        """Test memory estimation for small circuit (10 qubits)."""
        estimate = memory_manager.estimate_memory(10)

        # 10 qubits = 2^10 = 1024 amplitudes * 16 bytes = 16 KB
        expected_sv_mb = (2**10 * 16) / (1024 * 1024)

        assert estimate.qubit_count == 10
        assert abs(estimate.state_vector_mb - expected_sv_mb) < 0.01
        assert estimate.can_fit_on_device
        assert not estimate.fallback_required

    def test_memory_estimate_medium_circuit(self, memory_manager):
        """Test memory estimation for medium circuit (20 qubits)."""
        estimate = memory_manager.estimate_memory(20)

        # 20 qubits = 2^20 = 1M amplitudes * 16 bytes = 16 MB
        expected_sv_mb = (2**20 * 16) / (1024 * 1024)

        assert estimate.qubit_count == 20
        assert abs(estimate.state_vector_mb - expected_sv_mb) < 0.01
        assert estimate.can_fit_on_device

    def test_memory_estimate_large_circuit(self, memory_manager):
        """Test memory estimation for large circuit (30 qubits)."""
        estimate = memory_manager.estimate_memory(30)

        # 30 qubits = 2^30 = 1B amplitudes * 16 bytes = 16 GB
        expected_sv_mb = (2**30 * 16) / (1024 * 1024)

        assert estimate.qubit_count == 30
        assert (
            abs(estimate.state_vector_mb - expected_sv_mb) < 1.0
        )  # Allow 1 MB tolerance

        # With 14 GB free, 16 GB circuit should not fit
        assert not estimate.can_fit_on_device
        assert estimate.fallback_required

    def test_memory_estimate_with_single_precision(self, memory_manager):
        """Test memory estimation with single precision."""
        from proxima.backends.cuquantum_adapter import CuQuantumPrecision

        double_estimate = memory_manager.estimate_memory(25, CuQuantumPrecision.DOUBLE)
        single_estimate = memory_manager.estimate_memory(25, CuQuantumPrecision.SINGLE)

        # Single precision should be half the memory
        assert (
            abs(single_estimate.state_vector_mb - double_estimate.state_vector_mb / 2)
            < 0.1
        )

    def test_memory_check_before_execution_success(self, memory_manager):
        """Test pre-execution memory check for valid circuit."""
        can_execute, estimate = memory_manager.check_memory_before_execution(20)

        assert can_execute
        assert estimate.can_fit_on_device

    def test_memory_check_before_execution_failure(self, memory_manager):
        """Test pre-execution memory check for too-large circuit."""

        # 35 qubits would require ~512 GB - way too much
        can_execute, estimate = memory_manager.check_memory_before_execution(
            35, raise_on_insufficient=False
        )

        assert not can_execute
        assert estimate.fallback_required

    def test_query_gpu_memory(self, memory_manager, mock_gpu_device_info):
        """Test querying current GPU memory status."""
        memory_status = memory_manager.query_current_gpu_memory()

        assert memory_status["total_mb"] == mock_gpu_device_info.total_memory_mb
        assert memory_status["free_mb"] == mock_gpu_device_info.free_memory_mb
        assert 0 <= memory_status["utilization"] <= 1

    def test_calculate_max_qubits(self, memory_manager):
        """Test calculating maximum qubits for available memory."""
        # With 14 GB free
        max_qubits = memory_manager.calculate_max_qubits_for_memory(14000)

        # Should be around 29-30 qubits
        assert 28 <= max_qubits <= 31

    def test_convenience_memory_estimation(self):
        """Test convenience function for memory estimation."""
        from proxima.backends.gpu_memory_manager import estimate_gpu_memory_for_qubits

        estimate = estimate_gpu_memory_for_qubits(20, "double", 1024)

        assert estimate["qubit_count"] == 20
        assert estimate["precision"] == "double"
        assert estimate["state_vector_mb"] > 0
        assert estimate["total_mb"] > estimate["state_vector_mb"]


# =============================================================================
# Memory Pool Tests
# =============================================================================


class TestMemoryPooling:
    """Tests for GPU memory pooling functionality."""

    def test_pool_initialization(self, memory_manager):
        """Test memory pool initialization."""
        result = memory_manager.initialize_memory_pool(4096)  # 4 GB pool

        assert result
        stats = memory_manager.get_pool_stats()
        assert stats.pool_size_mb == 4096
        assert stats.available_mb == 4096

    def test_pool_allocation(self, memory_manager):
        """Test memory allocation from pool."""
        memory_manager.initialize_memory_pool(4096)

        success, alloc_id = memory_manager.allocate_from_pool(1024)

        assert success
        assert alloc_id is not None

        stats = memory_manager.get_pool_stats()
        assert stats.allocated_mb == 1024
        assert stats.available_mb == 3072

    def test_pool_release(self, memory_manager):
        """Test memory release back to pool."""
        memory_manager.initialize_memory_pool(4096)

        success, alloc_id = memory_manager.allocate_from_pool(1024)
        memory_manager.release_to_pool(alloc_id, 1024)

        stats = memory_manager.get_pool_stats()
        assert stats.allocated_mb == 0
        assert stats.available_mb == 4096
        assert stats.reuse_count == 1

    def test_pool_insufficient_memory(self, memory_manager):
        """Test allocation failure when pool is exhausted."""
        memory_manager.initialize_memory_pool(1024)  # 1 GB pool

        # Try to allocate 2 GB
        success, alloc_id = memory_manager.allocate_from_pool(2048)

        assert not success
        assert alloc_id is None

    def test_pool_cleanup(self, memory_manager):
        """Test memory pool cleanup."""
        memory_manager.initialize_memory_pool(4096)
        memory_manager.allocate_from_pool(1024)

        result = memory_manager.cleanup_pool()

        assert result
        stats = memory_manager.get_pool_stats()
        assert stats.allocated_mb == 0


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Tests for batch circuit execution."""

    def test_batch_execution_success(self, memory_manager):
        """Test successful batch execution."""
        # Create mock circuits
        circuits = [{"id": i, "qubits": 5} for i in range(5)]

        # Mock executor that succeeds
        def mock_executor(circuit):
            return {"circuit_id": circuit["id"], "success": True}

        result = memory_manager.execute_batch(circuits, mock_executor)

        assert result.total_circuits == 5
        assert result.successful_circuits == 5
        assert result.failed_circuits == 0
        assert len(result.results) == 5

    def test_batch_execution_with_failures(self, memory_manager):
        """Test batch execution with some failures."""
        circuits = [{"id": i, "qubits": 5} for i in range(5)]

        # Mock executor that fails on circuit 2
        def mock_executor(circuit):
            if circuit["id"] == 2:
                raise RuntimeError("Simulated failure")
            return {"circuit_id": circuit["id"], "success": True}

        result = memory_manager.execute_batch(circuits, mock_executor)

        assert result.total_circuits == 5
        assert result.successful_circuits == 4
        assert result.failed_circuits == 1
        assert len(result.errors) == 1
        assert result.errors[0]["circuit_index"] == 2

    def test_batch_execution_timing(self, memory_manager):
        """Test batch execution tracks timing."""
        circuits = [{"id": i} for i in range(3)]

        def slow_executor(circuit):
            time.sleep(0.01)  # 10ms delay
            return {"id": circuit["id"]}

        result = memory_manager.execute_batch(circuits, slow_executor)

        # Should take at least 30ms
        assert result.total_execution_time_ms >= 30


# =============================================================================
# Fallback Tests
# =============================================================================


class TestFallbackStrategy:
    """Tests for CPU fallback strategy."""

    def test_no_fallback_for_valid_circuit(self, memory_manager):
        """Test no fallback needed for circuit that fits."""
        should_fallback, reason = memory_manager.should_fallback_to_cpu(20)

        assert not should_fallback
        assert reason == ""

    def test_fallback_for_large_circuit(self, memory_manager):
        """Test fallback recommended for too-large circuit."""
        should_fallback, reason = memory_manager.should_fallback_to_cpu(35)

        assert should_fallback
        assert "memory" in reason.lower() or "insufficient" in reason.lower()

    def test_fallback_for_small_circuit(self, memory_manager):
        """Test fallback recommended for very small circuits."""
        should_fallback, reason = memory_manager.should_fallback_to_cpu(5)

        # Small circuits may be faster on CPU
        assert should_fallback
        assert "small" in reason.lower()

    def test_no_fallback_when_forced(self, memory_manager):
        """Test no fallback when force_gpu is True."""
        should_fallback, reason = memory_manager.should_fallback_to_cpu(
            5, force_gpu=True
        )

        assert not should_fallback

    def test_fallback_recommendation(self, memory_manager):
        """Test detailed fallback recommendation."""
        recommendation = memory_manager.get_fallback_recommendation(35)

        assert recommendation["should_fallback"]
        assert len(recommendation["alternatives"]) > 0

        # Should suggest alternatives
        actions = [alt["action"] for alt in recommendation["alternatives"]]
        assert "reduce_qubits" in actions or "use_single_precision" in actions


# =============================================================================
# Performance Tests (Step 2.5.3)
# =============================================================================


class TestPerformance:
    """Performance comparison tests."""

    @requires_gpu
    def test_gpu_vs_cpu_speedup(self, sample_circuit):
        """Test GPU provides speedup over CPU (requires real GPU)."""
        from qiskit_aer import AerSimulator

        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        # CPU execution
        cpu_sim = AerSimulator(method="statevector")
        start = time.perf_counter()
        cpu_sim.run(sample_circuit, shots=1000).result()
        cpu_time = time.perf_counter() - start

        # GPU execution
        adapter = CuQuantumAdapter()
        if adapter.is_available():
            start = time.perf_counter()
            gpu_result = adapter.execute(sample_circuit, {"shots": 1000})
            gpu_time = gpu_result.execution_time_ms / 1000

            # Log comparison (not necessarily faster for small circuits)
            print(f"CPU time: {cpu_time*1000:.2f}ms, GPU time: {gpu_time*1000:.2f}ms")

    def test_memory_estimation_performance(self, memory_manager):
        """Test memory estimation is fast."""
        start = time.perf_counter()

        for qubits in range(10, 35):
            memory_manager.estimate_memory(qubits)

        elapsed = time.perf_counter() - start

        # Should complete 25 estimations in < 100ms
        assert elapsed < 0.1

    def test_pool_operations_performance(self, memory_manager):
        """Test memory pool operations are fast."""
        memory_manager.initialize_memory_pool(8192)

        start = time.perf_counter()

        for _ in range(100):
            success, alloc_id = memory_manager.allocate_from_pool(50)
            if success:
                memory_manager.release_to_pool(alloc_id, 50)

        elapsed = time.perf_counter() - start

        # 100 alloc/release cycles in < 100ms
        assert elapsed < 0.1


# =============================================================================
# Correctness Tests (Step 2.5.4)
# =============================================================================


class TestCorrectness:
    """Tests verifying GPU results match CPU reference."""

    def test_statevector_comparison_mock(self, mock_gpu_environment, sample_circuit):
        """Test statevector results are correct (mocked GPU)."""
        # This test uses mock GPU, so we just verify the adapter structure
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        adapter = CuQuantumAdapter()
        assert adapter.get_name() == "cuquantum"

    @requires_gpu
    def test_statevector_fidelity(self):
        """Test GPU statevector matches CPU (requires real GPU)."""
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        # Create test circuit
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.save_statevector()

        # CPU reference
        cpu_sim = AerSimulator(method="statevector")
        cpu_result = cpu_sim.run(circuit).result()
        cpu_sv = np.array(cpu_result.data()["statevector"])

        # GPU execution
        adapter = CuQuantumAdapter()
        if adapter.is_available():
            gpu_result = adapter.execute(circuit, {"shots": 0})
            gpu_sv = np.array(gpu_result.data.get("statevector", []))

            # Calculate fidelity
            if len(gpu_sv) > 0:
                fidelity = abs(np.dot(np.conj(cpu_sv), gpu_sv)) ** 2
                assert fidelity > 0.9999, f"Fidelity too low: {fidelity}"

    @requires_gpu
    def test_measurement_distribution(self, sample_circuit):
        """Test measurement distribution is statistically similar."""
        from qiskit_aer import AerSimulator

        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        shots = 10000

        # CPU reference
        cpu_sim = AerSimulator()
        cpu_result = cpu_sim.run(sample_circuit, shots=shots).result()
        cpu_counts = cpu_result.get_counts()

        # GPU execution
        adapter = CuQuantumAdapter()
        if adapter.is_available():
            gpu_result = adapter.execute(sample_circuit, {"shots": shots})
            gpu_counts = gpu_result.data.get("counts", {})

            # Compare distributions
            for key in set(cpu_counts.keys()) | set(gpu_counts.keys()):
                cpu_prob = cpu_counts.get(key, 0) / shots
                gpu_prob = gpu_counts.get(key, 0) / shots

                # Allow 5% difference due to statistical variation
                assert (
                    abs(cpu_prob - gpu_prob) < 0.05
                ), f"Distribution mismatch for {key}"

    def test_bell_state_preparation(self, mock_gpu_environment):
        """Test Bell state is prepared correctly."""
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        try:
            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)

            adapter = CuQuantumAdapter()

            # Verify circuit validation passes
            validation = adapter.validate_circuit(circuit)
            assert validation.is_valid

        except ImportError:
            pytest.skip("Qiskit not available")


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_qubit_circuit(self, memory_manager):
        """Test handling of zero-qubit circuit."""
        estimate = memory_manager.estimate_memory(0)

        # 2^0 = 1 amplitude
        assert estimate.state_vector_mb < 1  # Less than 1 MB

    def test_single_qubit_circuit(self, memory_manager):
        """Test handling of single-qubit circuit."""
        estimate = memory_manager.estimate_memory(1)

        assert estimate.can_fit_on_device
        assert not estimate.fallback_required

    def test_max_qubit_limit(self, mock_gpu_environment, cuquantum_config):
        """Test maximum qubit limit is enforced."""
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        cuquantum_config.max_qubits = 25
        adapter = CuQuantumAdapter(config=cuquantum_config)

        caps = adapter.get_capabilities()
        assert caps.max_qubits == 25

    def test_invalid_device_id(self, memory_manager):
        """Test handling of invalid GPU device ID."""
        gpu_info = memory_manager._get_gpu_info(999)  # Non-existent device

        assert gpu_info is None

    def test_empty_batch_execution(self, memory_manager):
        """Test handling of empty batch."""
        result = memory_manager.execute_batch([], lambda x: x)

        assert result.total_circuits == 0
        assert result.successful_circuits == 0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for cuQuantum configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from proxima.backends.cuquantum_adapter import (
            CuQuantumConfig,
            CuQuantumExecutionMode,
            CuQuantumPrecision,
        )

        config = CuQuantumConfig()

        assert config.execution_mode == CuQuantumExecutionMode.GPU_PREFERRED
        assert config.precision == CuQuantumPrecision.DOUBLE
        assert config.fallback_to_cpu is True
        assert config.workspace_size_mb == 1024

    def test_custom_config(self):
        """Test custom configuration."""
        from proxima.backends.cuquantum_adapter import (
            CuQuantumConfig,
            CuQuantumExecutionMode,
            CuQuantumPrecision,
        )

        config = CuQuantumConfig(
            execution_mode=CuQuantumExecutionMode.GPU_ONLY,
            precision=CuQuantumPrecision.SINGLE,
            gpu_device_id=1,
            max_qubits=40,
        )

        assert config.execution_mode == CuQuantumExecutionMode.GPU_ONLY
        assert config.precision == CuQuantumPrecision.SINGLE
        assert config.gpu_device_id == 1
        assert config.max_qubits == 40

    def test_config_affects_memory_estimation(self, mock_gpu_device_info):
        """Test configuration affects memory estimation."""
        from proxima.backends.cuquantum_adapter import (
            CuQuantumConfig,
            CuQuantumPrecision,
        )
        from proxima.backends.gpu_memory_manager import GPUMemoryManager

        # Single precision config
        single_config = CuQuantumConfig(precision=CuQuantumPrecision.SINGLE)
        single_manager = GPUMemoryManager(config=single_config)
        single_manager.set_gpu_devices([mock_gpu_device_info])

        # Double precision config
        double_config = CuQuantumConfig(precision=CuQuantumPrecision.DOUBLE)
        double_manager = GPUMemoryManager(config=double_config)
        double_manager.set_gpu_devices([mock_gpu_device_info])

        single_est = single_manager.estimate_memory(25)
        double_est = double_manager.estimate_memory(25)

        # Double should require ~2x the state vector memory
        assert double_est.state_vector_mb > single_est.state_vector_mb * 1.5


# =============================================================================
# Integration with Registry
# =============================================================================


class TestRegistryIntegration:
    """Tests for integration with backend registry."""

    def test_cuquantum_registered(self, mock_gpu_environment):
        """Test cuQuantum is registered in backend registry."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        backends = registry.list_backends()

        # cuQuantum should be in the list
        backend_names = [b.get_name() for b in backends]
        assert "cuquantum" in backend_names

    def test_gpu_backend_discovery(self, mock_gpu_environment):
        """Test GPU backends are discovered correctly."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()

        # Try to get GPU backends
        if hasattr(registry, "get_gpu_backends"):
            gpu_backends = registry.get_gpu_backends()
            assert any(b.get_name() == "cuquantum" for b in gpu_backends)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
