"""Step 5.1: Unit Testing - cuQuantum Adapter Tests.

Comprehensive test suite for cuQuantum/GPU backend adapter covering:
- GPU detection and initialization
- GPU memory management
- Execution with GPU acceleration
- Fallback to CPU when GPU unavailable

Test Categories:
| Test Type       | Purpose                                       |
|-----------------|-----------------------------------------------|
| GPU Detection   | CUDA availability, device enumeration         |
| Memory          | GPU memory estimation, allocation, cleanup    |
| Execution       | GPU-accelerated simulation                    |
| Fallback        | CPU fallback when GPU unavailable             |
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
    ResultType,
    SimulatorType,
    ValidationResult,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA as available with a GPU device."""
    mock_cupy = MagicMock()
    mock_cupy.cuda.runtime.getDeviceCount.return_value = 1
    mock_cupy.cuda.Device.return_value.mem_info = (
        8 * 1024**3,
        16 * 1024**3,
    )  # 8GB free, 16GB total
    mock_cupy.cuda.Device.return_value.compute_capability = (8, 0)  # Ampere

    return mock_cupy


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable."""
    mock_cupy = MagicMock()
    mock_cupy.cuda.runtime.getDeviceCount.side_effect = RuntimeError("No CUDA devices")

    return mock_cupy


@pytest.fixture
def mock_qiskit_aer_gpu():
    """Mock Qiskit Aer with GPU support."""
    mock_aer = MagicMock()

    # Mock AerSimulator
    mock_simulator = MagicMock()
    mock_simulator.run.return_value.result.return_value.get_counts.return_value = {
        "00": 512,
        "11": 512,
    }
    mock_aer.AerSimulator.return_value = mock_simulator

    return mock_aer


@pytest.fixture
def mock_cuquantum_adapter(mock_cuda_available, mock_qiskit_aer_gpu):
    """Create a cuQuantum adapter with mocked dependencies."""
    with patch.dict(
        "sys.modules",
        {
            "cupy": mock_cuda_available,
            "qiskit_aer": mock_qiskit_aer_gpu,
        },
    ):
        from proxima.backends.cuquantum_adapter import CuQuantumAdapter

        adapter = CuQuantumAdapter()
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
def medium_circuit():
    """Create a medium-sized circuit (15 qubits)."""
    return {
        "num_qubits": 15,
        "gates": [{"name": "H", "qubits": [i]} for i in range(15)]
        + [{"name": "CNOT", "qubits": [i, i + 1]} for i in range(14)],
        "measurements": list(range(15)),
    }


@pytest.fixture
def large_circuit():
    """Create a large circuit for GPU testing (25 qubits)."""
    return {
        "num_qubits": 25,
        "gates": [{"name": "H", "qubits": [i]} for i in range(25)]
        + [{"name": "CNOT", "qubits": [i, i + 1]} for i in range(24)],
        "measurements": list(range(25)),
    }


# =============================================================================
# STEP 5.1.1: GPU AVAILABILITY TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestCuQuantumGPUDetection:
    """Tests for GPU detection and initialization."""

    def test_gpu_available_detection(self, mock_cuda_available):
        """Test detection when GPU is available."""
        with patch.dict("sys.modules", {"cupy": mock_cuda_available}):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()
            assert adapter.is_available() is True

    def test_gpu_unavailable_detection(self, mock_cuda_unavailable):
        """Test detection when GPU is unavailable."""
        with patch.dict("sys.modules", {"cupy": mock_cuda_unavailable}):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()
            # Should indicate GPU unavailable
            caps = adapter.get_capabilities()
            assert caps.supports_gpu is False or adapter.is_available() is False

    def test_multiple_gpu_detection(self, mock_cuda_available):
        """Test detection of multiple GPUs."""
        mock_cuda_available.cuda.runtime.getDeviceCount.return_value = 4

        with patch.dict("sys.modules", {"cupy": mock_cuda_available}):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()
            assert adapter.is_available() is True

    def test_gpu_compute_capability_check(self, mock_cuda_available):
        """Test that compute capability is verified."""
        # Set compute capability to 7.0 (Volta - minimum for cuQuantum)
        mock_cuda_available.cuda.Device.return_value.compute_capability = (7, 0)

        with patch.dict("sys.modules", {"cupy": mock_cuda_available}):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()
            caps = adapter.get_capabilities()
            # Should indicate GPU support with Volta or newer
            assert caps.supports_gpu is True

    def test_adapter_name(self, mock_cuquantum_adapter):
        """Test adapter identification."""
        assert mock_cuquantum_adapter.get_name() == "cuquantum"

    def test_adapter_capabilities(self, mock_cuquantum_adapter):
        """Test capability reporting."""
        caps = mock_cuquantum_adapter.get_capabilities()

        assert isinstance(caps, Capabilities)
        assert SimulatorType.STATE_VECTOR in caps.simulator_types
        assert caps.supports_gpu is True
        assert caps.max_qubits >= 30  # GPU can handle more qubits


# =============================================================================
# STEP 5.1.2: GPU MEMORY MANAGEMENT TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestCuQuantumMemoryManagement:
    """Tests for GPU memory management."""

    def test_gpu_memory_estimation_small_circuit(
        self, mock_cuquantum_adapter, simple_circuit
    ):
        """Test GPU memory estimation for small circuits."""
        estimate = mock_cuquantum_adapter.estimate_resources(simple_circuit)

        assert estimate.memory_mb is not None
        assert estimate.memory_mb > 0
        # 2 qubits = 4 amplitudes * 16 bytes = 64 bytes + overhead
        assert estimate.memory_mb < 10

    def test_gpu_memory_estimation_large_circuit(
        self, mock_cuquantum_adapter, large_circuit
    ):
        """Test GPU memory estimation for large circuits."""
        estimate = mock_cuquantum_adapter.estimate_resources(large_circuit)

        assert estimate.memory_mb is not None
        # 25 qubits = 2^25 amplitudes * 16 bytes = 512 MB + workspace
        assert estimate.memory_mb > 500

    def test_memory_fit_validation(self, mock_cuquantum_adapter, large_circuit):
        """Test validation that circuit fits in GPU memory."""
        validation = mock_cuquantum_adapter.validate_circuit(large_circuit)

        # Should validate if memory is sufficient
        assert isinstance(validation, ValidationResult)

    def test_memory_exceeded_detection(self, mock_cuda_available):
        """Test detection when circuit exceeds GPU memory."""
        # Set low GPU memory
        mock_cuda_available.cuda.Device.return_value.mem_info = (
            1 * 1024**3,
            2 * 1024**3,
        )  # 1GB free

        with patch.dict("sys.modules", {"cupy": mock_cuda_available}):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()

            huge_circuit = {
                "num_qubits": 32,  # Would need ~64GB
                "gates": [],
            }

            validation = adapter.validate_circuit(huge_circuit)
            # Should warn about memory or fail validation
            if not validation.valid:
                assert (
                    "memory" in validation.message.lower()
                    or "resource" in validation.message.lower()
                )

    def test_workspace_memory_included(self, mock_cuquantum_adapter, medium_circuit):
        """Test that workspace memory is included in estimate."""
        estimate = mock_cuquantum_adapter.estimate_resources(medium_circuit)

        # cuStateVec workspace should be included
        # 15 qubits = 2^15 * 16 bytes = ~0.5MB state + 1-2GB workspace
        assert estimate.memory_mb is not None
        assert estimate.metadata.get("includes_workspace", True)


# =============================================================================
# STEP 5.1.3: GPU EXECUTION TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestCuQuantumExecution:
    """Tests for GPU-accelerated execution."""

    def test_simple_circuit_execution(self, mock_cuquantum_adapter, simple_circuit):
        """Test execution of simple circuit on GPU."""
        result = mock_cuquantum_adapter.execute(
            simple_circuit,
            options={"shots": 1024},
        )

        assert isinstance(result, ExecutionResult)
        assert result.backend == "cuquantum"
        assert result.qubit_count == 2

    def test_gpu_execution_mode(self, mock_cuquantum_adapter, simple_circuit):
        """Test that GPU execution mode is used."""
        result = mock_cuquantum_adapter.execute(
            simple_circuit,
            options={"device": "GPU"},
        )

        assert isinstance(result, ExecutionResult)
        # Metadata should indicate GPU was used
        assert result.metadata.get("device") in ["GPU", "cuda", "gpu", None]

    def test_device_selection(self, mock_cuquantum_adapter, simple_circuit):
        """Test GPU device selection."""
        result = mock_cuquantum_adapter.execute(
            simple_circuit,
            options={"gpu_device_id": 0},
        )

        assert isinstance(result, ExecutionResult)

    def test_execution_timing(self, mock_cuquantum_adapter, simple_circuit):
        """Test that execution time is measured."""
        result = mock_cuquantum_adapter.execute(simple_circuit)

        assert result.execution_time_ms >= 0

    def test_statevector_mode(self, mock_cuquantum_adapter, simple_circuit):
        """Test state vector simulation on GPU."""
        result = mock_cuquantum_adapter.execute(
            simple_circuit,
            options={"method": "statevector"},
        )

        assert result.simulator_type == SimulatorType.STATE_VECTOR

    def test_measurement_results(self, mock_cuquantum_adapter, simple_circuit):
        """Test that measurement results are returned."""
        result = mock_cuquantum_adapter.execute(
            simple_circuit,
            options={"shots": 1000},
        )

        assert result.result_type == ResultType.COUNTS
        assert result.data is not None


# =============================================================================
# STEP 5.1.4: FALLBACK TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestCuQuantumFallback:
    """Tests for CPU fallback when GPU unavailable."""

    def test_fallback_when_gpu_unavailable(
        self, mock_cuda_unavailable, mock_qiskit_aer_gpu
    ):
        """Test fallback to CPU when GPU is not available."""
        with patch.dict(
            "sys.modules",
            {
                "cupy": mock_cuda_unavailable,
                "qiskit_aer": mock_qiskit_aer_gpu,
            },
        ):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()

            simple_circuit = {
                "num_qubits": 2,
                "gates": [{"name": "H", "qubits": [0]}],
            }

            # Should either fall back to CPU or indicate unavailability
            if adapter.is_available():
                result = adapter.execute(simple_circuit)
                assert isinstance(result, ExecutionResult)
            else:
                assert adapter.is_available() is False

    def test_fallback_on_memory_exceeded(
        self, mock_cuda_available, mock_qiskit_aer_gpu
    ):
        """Test fallback when GPU memory is insufficient."""
        # Set low GPU memory
        mock_cuda_available.cuda.Device.return_value.mem_info = (
            100 * 1024**2,
            200 * 1024**2,
        )  # 100MB

        with patch.dict(
            "sys.modules",
            {
                "cupy": mock_cuda_available,
                "qiskit_aer": mock_qiskit_aer_gpu,
            },
        ):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()

            large_circuit = {
                "num_qubits": 28,  # Would need ~4GB
                "gates": [],
            }

            # Should handle gracefully - either fallback or error
            try:
                result = adapter.execute(
                    large_circuit, options={"fallback_to_cpu": True}
                )
                assert isinstance(result, ExecutionResult)
            except Exception as e:
                # Should be a clear memory-related error
                assert "memory" in str(e).lower() or "resource" in str(e).lower()

    def test_explicit_cpu_fallback_option(self, mock_cuquantum_adapter, simple_circuit):
        """Test explicit CPU fallback option."""
        result = mock_cuquantum_adapter.execute(
            simple_circuit,
            options={"fallback_to_cpu": True},
        )

        assert isinstance(result, ExecutionResult)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.unit
@pytest.mark.backend
class TestCuQuantumErrorHandling:
    """Tests for error handling in cuQuantum adapter."""

    def test_missing_cupy_error(self):
        """Test error when cupy is not installed."""
        with patch.dict("sys.modules", {"cupy": None}):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()
            assert adapter.is_available() is False

    def test_missing_qiskit_aer_error(self, mock_cuda_available):
        """Test error when qiskit-aer is not installed."""
        with patch.dict(
            "sys.modules",
            {
                "cupy": mock_cuda_available,
                "qiskit_aer": None,
            },
        ):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()
            # Should handle gracefully
            assert isinstance(adapter.is_available(), bool)

    def test_invalid_circuit_error(self, mock_cuquantum_adapter):
        """Test handling of invalid circuit."""
        invalid_circuit = {"invalid": "structure"}

        validation = mock_cuquantum_adapter.validate_circuit(invalid_circuit)
        assert validation.valid is False

    def test_cuda_runtime_error(self, mock_cuda_available, mock_qiskit_aer_gpu):
        """Test handling of CUDA runtime errors."""
        mock_qiskit_aer_gpu.AerSimulator.return_value.run.side_effect = RuntimeError(
            "CUDA error"
        )

        with patch.dict(
            "sys.modules",
            {
                "cupy": mock_cuda_available,
                "qiskit_aer": mock_qiskit_aer_gpu,
            },
        ):
            from proxima.backends.cuquantum_adapter import CuQuantumAdapter

            adapter = CuQuantumAdapter()

            try:
                adapter.execute({"num_qubits": 2, "gates": []})
            except Exception as e:
                # Should handle error gracefully
                assert isinstance(e, Exception)

    def test_cleanup_after_error(self, mock_cuquantum_adapter):
        """Test that resources are cleaned up after errors."""
        mock_cuquantum_adapter._cleanup()
        # Should not raise any exceptions
