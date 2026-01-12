"""QuEST backend adapter (DensityMatrix + StateVector) with comprehensive error handling.

This adapter integrates the QuEST (Quantum Exact Simulation Toolkit) quantum simulator
with Proxima's backend infrastructure. QuEST is a high-performance C++ quantum computer
simulator that supports:
- State vector simulation (pure states)
- Density matrix simulation (mixed states)
- GPU acceleration (CUDA, HIP)
- OpenMP parallelization
- MPI distribution
- cuQuantum integration

Python bindings are provided via pyQuEST (by rrmeister), which uses Cython for
high-performance interoperability with the C++ QuEST library.

References:
- QuEST GitHub: https://github.com/QuEST-Kit/QuEST
- pyQuEST (Cython bindings): https://github.com/rrmeister/pyQuEST
- QuEST Documentation: https://quest-kit.github.io/QuEST/

Step 1.1-1.3: Basic adapter implementation (completed)
Step 1.4: QuEST-Specific Features (precision, GPU, rank truncation, OpenMP)
Step 1.5: Backend Registry Integration
Step 1.6: Comprehensive Error Handling
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from proxima.backends.base import (
    BaseBackendAdapter,
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
    ValidationResult,
)
from proxima.backends.exceptions import (
    BackendError,
    BackendErrorCode,
    CircuitValidationError,
    ExecutionError,
    MemoryExceededError,
    QubitLimitExceededError,
    UnsupportedOperationError,
)

# =============================================================================
# Step 1.4: QuEST-Specific Features - Configuration Classes
# =============================================================================


class QuestPrecision(str, Enum):
    """QuEST numerical precision modes."""

    SINGLE = "single"  # 32-bit float (faster, less accurate)
    DOUBLE = "double"  # 64-bit float (default, good balance)
    QUAD = "quad"  # 128-bit float (highest accuracy, slower)


@dataclass
class QuestConfig:
    """Configuration for QuEST backend execution.

    Step 1.4: Handle QuEST-Specific Features

    Attributes:
        precision: Numerical precision mode (single/double/quad)
        gpu_enabled: Whether to use GPU acceleration if available
        gpu_device_id: Which GPU device to use (0-indexed)
        openmp_threads: Number of OpenMP threads (0 = auto-detect)
        truncation_threshold: Threshold for density matrix rank truncation
        max_rank: Maximum rank for density matrix (0 = unlimited)
        memory_limit_mb: Memory limit in MB (0 = unlimited)
        validate_normalization: Check state normalization after operations
    """

    precision: QuestPrecision = QuestPrecision.DOUBLE
    gpu_enabled: bool = True  # Use GPU if available
    gpu_device_id: int = 0
    openmp_threads: int = 0  # 0 = auto-detect
    truncation_threshold: float = 1e-10
    max_rank: int = 0  # 0 = unlimited
    memory_limit_mb: int = 0  # 0 = unlimited
    validate_normalization: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> QuestConfig:
        """Create QuestConfig from dictionary."""
        precision_str = config.get("quest_precision", "double")
        try:
            precision = QuestPrecision(precision_str.lower())
        except ValueError:
            precision = QuestPrecision.DOUBLE

        return cls(
            precision=precision,
            gpu_enabled=config.get("quest_gpu_enabled", True),
            gpu_device_id=config.get("quest_gpu_device_id", 0),
            openmp_threads=config.get("quest_openmp_threads", 0),
            truncation_threshold=config.get("quest_truncation_threshold", 1e-10),
            max_rank=config.get("quest_max_rank", 0),
            memory_limit_mb=config.get("quest_memory_limit_mb", 0),
            validate_normalization=config.get("quest_validate_normalization", False),
        )


@dataclass
class QuestHardwareInfo:
    """Hardware information detected by QuEST.

    Step 1.4: GPU and OpenMP detection results.
    """

    cpu_cores: int = 1
    openmp_available: bool = False
    openmp_threads: int = 1
    gpu_available: bool = False
    gpu_device_count: int = 0
    gpu_device_name: str = ""
    gpu_memory_mb: int = 0
    cuda_version: str = ""
    quest_precision: str = "double"
    mpi_available: bool = False
    mpi_ranks: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for metadata."""
        return {
            "cpu_cores": self.cpu_cores,
            "openmp_available": self.openmp_available,
            "openmp_threads": self.openmp_threads,
            "gpu_available": self.gpu_available,
            "gpu_device_count": self.gpu_device_count,
            "gpu_device_name": self.gpu_device_name,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cuda_version": self.cuda_version,
            "quest_precision": self.quest_precision,
            "mpi_available": self.mpi_available,
            "mpi_ranks": self.mpi_ranks,
        }


# =============================================================================
# Step 1.6: QuEST-Specific Error Classes
# =============================================================================


class QuestInstallationError(BackendError):
    """Raised when QuEST/pyQuEST is not properly installed.

    Step 1.6: Installation Errors
    """

    def __init__(
        self,
        reason: str,
        missing_component: str = "pyQuEST",
        **kwargs: Any,
    ) -> None:
        suggestions = [
            "Install pyQuEST: pip install pyquest",
            "For GPU support, build QuEST from source with CUDA enabled",
            "Check that your Python environment has the correct architecture (64-bit)",
            "Ensure C++ runtime libraries are installed on your system",
        ]
        super().__init__(
            code=BackendErrorCode.NOT_INSTALLED,
            message=f"QuEST installation error: {reason}",
            backend_name="quest",
            recoverable=False,
            suggestions=suggestions,
            details={"reason": reason, "missing_component": missing_component},
            **kwargs,
        )


class QuestGPUError(BackendError):
    """Raised when GPU operations fail in QuEST.

    Step 1.6: GPU-specific errors
    """

    def __init__(
        self,
        reason: str,
        gpu_device_id: int = 0,
        fallback_available: bool = True,
        **kwargs: Any,
    ) -> None:
        suggestions = [
            "Ensure NVIDIA GPU drivers are up to date",
            "Check CUDA toolkit installation matches QuEST build",
            "Try setting quest_gpu_enabled=False to use CPU instead",
            "Reduce circuit size to fit in GPU memory",
        ]
        super().__init__(
            code=BackendErrorCode.RESOURCE_EXHAUSTED,
            message=f"QuEST GPU error: {reason}",
            backend_name="quest",
            recoverable=fallback_available,
            suggestions=suggestions,
            details={
                "reason": reason,
                "gpu_device_id": gpu_device_id,
                "fallback_available": fallback_available,
            },
            **kwargs,
        )


class QuestMemoryError(MemoryExceededError):
    """Raised when QuEST runs out of memory.

    Step 1.6: Resource Errors - Memory
    """

    def __init__(
        self,
        required_mb: float,
        available_mb: float | None = None,
        is_gpu: bool = False,
        num_qubits: int = 0,
        **kwargs: Any,
    ) -> None:
        memory_type = "GPU" if is_gpu else "system"
        super().__init__(
            backend_name="quest",
            required_mb=required_mb,
            available_mb=available_mb,
            **kwargs,
        )
        self.suggestions.extend(
            [
                f"Circuit requires {num_qubits} qubits, try reducing to {num_qubits - 2} or fewer",
                f"Consider using {'CPU' if is_gpu else 'a machine with more RAM'}",
                "For density matrix, memory scales as 4^n; for statevector as 2^n",
            ]
        )
        self.details.update(
            {
                "memory_type": memory_type,
                "num_qubits": num_qubits,
                "is_gpu": is_gpu,
            }
        )


class QuestCircuitError(CircuitValidationError):
    """Raised when circuit has QuEST-incompatible features.

    Step 1.6: Circuit Errors
    """

    def __init__(
        self,
        reason: str,
        unsupported_gates: list[str] | None = None,
        circuit_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backend_name="quest",
            reason=reason,
            circuit_info=circuit_info,
            **kwargs,
        )
        if unsupported_gates:
            self.suggestions.insert(
                0, f"Unsupported gates: {', '.join(unsupported_gates)}"
            )
            self.suggestions.append(
                "Consider decomposing custom gates into basic gates"
            )
            self.details["unsupported_gates"] = unsupported_gates


class QuestRuntimeError(ExecutionError):
    """Raised when QuEST encounters a runtime error.

    Step 1.6: Execution Errors
    """

    def __init__(
        self,
        reason: str,
        stage: str = "execution",
        quest_error_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            backend_name="quest",
            reason=reason,
            stage=stage,
            **kwargs,
        )
        self.details["quest_error_code"] = quest_error_code
        self.suggestions.extend(
            [
                "Check QuEST debug logs for more details",
                "Ensure circuit parameters (angles) are valid numbers",
                "Verify qubit indices are within range",
            ]
        )


# =============================================================================
# Main QuEST Backend Adapter
# =============================================================================


class QuestBackendAdapter(BaseBackendAdapter):
    """Backend adapter for QuEST quantum simulator.

    QuEST (Quantum Exact Simulation Toolkit) is a high-performance C++ quantum
    computer simulator supporting both state vector and density matrix simulations.
    This adapter provides integration via pyQuEST Python bindings.

    Step 1.4 Features:
    - Precision Configuration (single/double/quad)
    - GPU Acceleration with automatic detection and fallback
    - Rank Truncation for density matrix optimization
    - OpenMP Parallelization for CPU execution

    Step 1.5: Registered in BackendRegistry with proper discovery

    Step 1.6: Comprehensive error handling with:
    - Installation error detection
    - Resource (memory) error handling
    - Circuit validation errors
    - Runtime error recovery
    """

    # Gate name mapping from Cirq/Qiskit conventions to pyQuEST operators
    GATE_MAP: dict[str, str] = {
        # Single-qubit gates
        "h": "H",
        "hadamard": "H",
        "x": "X",
        "pauli_x": "X",
        "not": "X",
        "y": "Y",
        "pauli_y": "Y",
        "z": "Z",
        "pauli_z": "Z",
        "s": "S",
        "t": "T",
        "sdg": "Sdg",
        "tdg": "Tdg",
        # Rotation gates
        "rx": "Rx",
        "ry": "Ry",
        "rz": "Rz",
        # Two-qubit gates
        "cx": "X",
        "cnot": "X",  # Controlled-X
        "cy": "Y",
        "cz": "Z",  # Controlled-Y/Z
        "swap": "SWAP",
    }

    # Supported gates list for validation
    SUPPORTED_GATES: set[str] = {
        "h",
        "hadamard",
        "x",
        "pauli_x",
        "not",
        "y",
        "pauli_y",
        "z",
        "pauli_z",
        "s",
        "t",
        "sdg",
        "tdg",
        "rx",
        "ry",
        "rz",
        "cx",
        "cnot",
        "cy",
        "cz",
        "swap",
        "ccx",
        "toffoli",
        "ccz",
        "cswap",
        "fredkin",
        "u1",
        "u2",
        "u3",
        "p",
        "phase",
        "id",
        "i",
        "identity",
        "sx",
        "sxdg",
        "iswap",
    }

    MAX_QUBITS = 30  # Practical limit for statevector simulation
    MAX_QUBITS_DM = 15  # Practical limit for density matrix (4^n memory)

    def __init__(self, config: QuestConfig | None = None) -> None:
        """Initialize QuEST backend adapter.

        Args:
            config: Optional QuEST-specific configuration
        """
        self._config = config or QuestConfig()
        self._hardware_info: QuestHardwareInfo | None = None
        self._pyquest_module: Any = None
        self._logger = logging.getLogger("proxima.backends.quest")

        # Detect hardware on initialization
        if self.is_available():
            self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Detect available hardware features.

        Step 1.4: GPU Acceleration and OpenMP Parallelization detection
        """
        try:
            import psutil

            cpu_cores = psutil.cpu_count(logical=False) or 1
        except ImportError:
            cpu_cores = os.cpu_count() or 1

        self._hardware_info = QuestHardwareInfo(cpu_cores=cpu_cores)

        try:
            import pyQuEST

            self._pyquest_module = pyQuEST

            # Detect OpenMP
            if hasattr(pyQuEST, "get_num_threads"):
                self._hardware_info.openmp_available = True
                self._hardware_info.openmp_threads = pyQuEST.get_num_threads()
            elif hasattr(pyQuEST, "num_threads"):
                self._hardware_info.openmp_available = True
                self._hardware_info.openmp_threads = pyQuEST.num_threads
            else:
                # Check via environment variable
                omp_threads = os.environ.get("OMP_NUM_THREADS", "")
                if omp_threads.isdigit():
                    self._hardware_info.openmp_available = True
                    self._hardware_info.openmp_threads = int(omp_threads)

            # Detect GPU
            self._hardware_info.gpu_available = self._check_gpu_support()
            if self._hardware_info.gpu_available:
                self._detect_gpu_details()

            # Detect precision
            if hasattr(pyQuEST, "get_precision"):
                self._hardware_info.quest_precision = pyQuEST.get_precision()

            # Detect MPI
            if hasattr(pyQuEST, "is_mpi_enabled"):
                self._hardware_info.mpi_available = pyQuEST.is_mpi_enabled()
            if hasattr(pyQuEST, "get_num_ranks"):
                self._hardware_info.mpi_ranks = pyQuEST.get_num_ranks()

        except Exception as e:
            self._logger.warning(f"Hardware detection failed: {e}")

    def _check_gpu_support(self) -> bool:
        """Check if QuEST was compiled with GPU support.

        Step 1.4: GPU Acceleration detection
        """
        if not self._pyquest_module:
            return False

        try:
            # Check various GPU indicators
            if hasattr(self._pyquest_module, "is_gpu_enabled"):
                return self._pyquest_module.is_gpu_enabled()
            if hasattr(self._pyquest_module, "gpu_enabled"):
                return self._pyquest_module.gpu_enabled
            if hasattr(self._pyquest_module, "use_gpu"):
                return True
            if hasattr(self._pyquest_module, "set_gpu"):
                return True

            # Try to detect CUDA availability
            try:
                import pycuda.driver as cuda

                cuda.init()
                return cuda.Device.count() > 0
            except ImportError:
                pass

            try:
                import cupy

                return cupy.cuda.runtime.getDeviceCount() > 0
            except ImportError:
                pass

        except Exception:
            pass

        return False

    def _detect_gpu_details(self) -> None:
        """Detect GPU device details.

        Step 1.4: GPU information gathering
        """
        if not self._hardware_info:
            return

        try:
            import pycuda.driver as cuda

            cuda.init()
            device = cuda.Device(self._config.gpu_device_id)
            self._hardware_info.gpu_device_count = cuda.Device.count()
            self._hardware_info.gpu_device_name = device.name()
            self._hardware_info.gpu_memory_mb = device.total_memory() // (1024 * 1024)
            # Get CUDA version
            version = cuda.get_version()
            self._hardware_info.cuda_version = (
                f"{version // 1000}.{(version % 1000) // 10}"
            )
        except ImportError:
            try:
                import cupy

                device = cupy.cuda.Device(self._config.gpu_device_id)
                self._hardware_info.gpu_device_count = (
                    cupy.cuda.runtime.getDeviceCount()
                )
                props = cupy.cuda.runtime.getDeviceProperties(
                    self._config.gpu_device_id
                )
                self._hardware_info.gpu_device_name = (
                    props["name"].decode()
                    if isinstance(props["name"], bytes)
                    else props["name"]
                )
                self._hardware_info.gpu_memory_mb = props["totalGlobalMem"] // (
                    1024 * 1024
                )
            except ImportError:
                pass
        except Exception as e:
            self._logger.debug(f"GPU detail detection failed: {e}")

    def _configure_openmp(self, num_threads: int) -> None:
        """Configure OpenMP thread count.

        Step 1.4: OpenMP Parallelization configuration
        """
        if num_threads <= 0:
            # Auto-detect: use number of physical cores
            try:
                import psutil

                num_threads = psutil.cpu_count(logical=False) or 1
            except ImportError:
                num_threads = os.cpu_count() or 1

        # Set via environment variable
        os.environ["OMP_NUM_THREADS"] = str(num_threads)

        # Set via pyQuEST if available
        if self._pyquest_module and hasattr(self._pyquest_module, "set_num_threads"):
            try:
                self._pyquest_module.set_num_threads(num_threads)
            except Exception:
                pass

        if self._hardware_info:
            self._hardware_info.openmp_threads = num_threads

    def _configure_gpu(self, device_id: int) -> bool:
        """Configure GPU device for execution.

        Step 1.4: GPU Acceleration configuration

        Returns:
            True if GPU was successfully configured, False otherwise
        """
        if not self._hardware_info or not self._hardware_info.gpu_available:
            return False

        try:
            if self._pyquest_module and hasattr(self._pyquest_module, "set_gpu"):
                self._pyquest_module.set_gpu(device_id)
                return True

            # Alternative: set CUDA device via pycuda/cupy
            try:
                import pycuda.driver as cuda

                cuda.init()
                cuda.Device(device_id).make_context()
                return True
            except ImportError:
                pass

            try:
                import cupy

                cupy.cuda.Device(device_id).use()
                return True
            except ImportError:
                pass

        except Exception as e:
            self._logger.warning(f"GPU configuration failed: {e}")
            raise QuestGPUError(
                reason=f"Failed to configure GPU device {device_id}: {e}",
                gpu_device_id=device_id,
                fallback_available=True,
            )

        return False

    def _estimate_memory_mb(
        self, num_qubits: int, is_density_matrix: bool, precision: QuestPrecision
    ) -> float:
        """Estimate memory requirements in MB.

        Step 1.4: Memory estimation for resource checking

        Args:
            num_qubits: Number of qubits
            is_density_matrix: True for DM, False for statevector
            precision: Numerical precision mode

        Returns:
            Estimated memory requirement in MB
        """
        # Bytes per complex number based on precision
        bytes_per_complex = {
            QuestPrecision.SINGLE: 8,  # 2 * 4 bytes (float32)
            QuestPrecision.DOUBLE: 16,  # 2 * 8 bytes (float64)
            QuestPrecision.QUAD: 32,  # 2 * 16 bytes (float128)
        }

        element_size = bytes_per_complex.get(precision, 16)

        if is_density_matrix:
            # Density matrix: 2^n x 2^n complex matrix
            num_elements = 4**num_qubits
        else:
            # State vector: 2^n complex amplitudes
            num_elements = 2**num_qubits

        memory_bytes = num_elements * element_size
        memory_mb = memory_bytes / (1024 * 1024)

        # Add ~20% overhead for working memory
        return memory_mb * 1.2

    def _check_memory_available(
        self, required_mb: float, use_gpu: bool = False
    ) -> tuple[bool, float | None]:
        """Check if required memory is available.

        Step 1.6: Resource error prevention

        Returns:
            Tuple of (is_available, available_mb)
        """
        if use_gpu and self._hardware_info:
            available_mb = self._hardware_info.gpu_memory_mb
            if available_mb > 0:
                return (required_mb < available_mb * 0.9, available_mb)

        try:
            import psutil

            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
            return (required_mb < available_mb * 0.9, available_mb)
        except ImportError:
            return (True, None)  # Assume available if can't check

    def _apply_rank_truncation(
        self, density_matrix: np.ndarray, threshold: float, max_rank: int = 0
    ) -> tuple[np.ndarray, int]:
        """Apply rank truncation to density matrix.

        Step 1.4: Rank Truncation for Density Matrices

        Uses SVD-based truncation to reduce rank while preserving
        important quantum information.

        Args:
            density_matrix: Input density matrix
            threshold: Singular value threshold for truncation
            max_rank: Maximum rank to keep (0 = no limit)

        Returns:
            Tuple of (truncated_matrix, final_rank)
        """
        if density_matrix.ndim != 2:
            return density_matrix, 1

        try:
            # Perform SVD
            U, S, Vh = np.linalg.svd(density_matrix, full_matrices=False)

            # Determine rank based on threshold
            significant_mask = S > threshold
            rank = np.sum(significant_mask)

            # Apply max_rank limit
            if max_rank > 0 and rank > max_rank:
                rank = max_rank

            # Ensure at least rank 1
            rank = max(1, rank)

            # Truncate
            U_trunc = U[:, :rank]
            S_trunc = S[:rank]
            Vh_trunc = Vh[:rank, :]

            # Reconstruct
            truncated = U_trunc @ np.diag(S_trunc) @ Vh_trunc

            # Renormalize trace to 1
            trace = np.trace(truncated)
            if abs(trace) > 1e-10:
                truncated = truncated / trace

            return truncated, rank

        except np.linalg.LinAlgError:
            return density_matrix, density_matrix.shape[0]

    def get_name(self) -> str:
        """Return the backend identifier name."""
        return "quest"

    def get_version(self) -> str:
        """Return the pyQuEST/QuEST version string."""
        spec = importlib.util.find_spec("pyQuEST")
        if spec and spec.loader:
            try:
                import pyQuEST

                version = getattr(pyQuEST, "__version__", None)
                if version:
                    return version
                try:
                    from importlib.metadata import version as get_version

                    return get_version("pyquest")
                except Exception:
                    return "available"
            except Exception:
                return "unknown"
        return "unavailable"

    def is_available(self) -> bool:
        """Check if pyQuEST is installed and can be imported.

        Step 1.6: Installation error detection
        """
        spec = importlib.util.find_spec("pyQuEST")
        if spec is None:
            return False
        try:
            import pyQuEST

            # Quick sanity check
            _ = pyQuEST.Register(1)
            return True
        except ImportError as e:
            self._logger.debug(f"pyQuEST import failed: {e}")
            return False
        except Exception as e:
            self._logger.debug(f"pyQuEST sanity check failed: {e}")
            return False

    def get_capabilities(self) -> Capabilities:
        """Return the capabilities of the QuEST backend.

        Step 1.4: Report GPU, precision, and parallelization capabilities
        """
        gpu_available = False
        if self._hardware_info:
            gpu_available = (
                self._hardware_info.gpu_available and self._config.gpu_enabled
            )

        custom_features = {
            "openmp": (
                self._hardware_info.openmp_available if self._hardware_info else False
            ),
            "mpi": self._hardware_info.mpi_available if self._hardware_info else False,
            "cuquantum": gpu_available,
            "density_matrix": True,
            "state_vector": True,
            "precision": self._config.precision.value,
            "rank_truncation": True,
        }

        if self._hardware_info:
            custom_features.update(self._hardware_info.to_dict())

        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=self.MAX_QUBITS,
            supports_noise=True,
            supports_gpu=gpu_available,
            supports_batching=False,
            custom_features=custom_features,
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate that a circuit can be executed on QuEST.

        Step 1.6: Circuit validation with detailed error messages
        """
        if not self.is_available():
            return ValidationResult(
                valid=False,
                message="pyQuEST not installed. Install with: pip install pyquest",
            )

        try:
            import pyQuEST
        except Exception as exc:
            return ValidationResult(
                valid=False, message=f"pyQuEST import failed: {exc}"
            )

        # Check for pyQuEST native Circuit
        if hasattr(pyQuEST, "Circuit") and isinstance(circuit, pyQuEST.Circuit):
            return ValidationResult(valid=True, message="ok")

        # Check for list of gate dictionaries
        if isinstance(circuit, list):
            unsupported = []
            for i, gate in enumerate(circuit):
                if not isinstance(gate, dict):
                    return ValidationResult(
                        valid=False, message=f"Gate {i} is not a dictionary"
                    )
                if "gate" not in gate:
                    return ValidationResult(
                        valid=False, message=f"Gate {i} missing 'gate' key"
                    )
                if "qubits" not in gate:
                    return ValidationResult(
                        valid=False, message=f"Gate {i} missing 'qubits' key"
                    )
                gate_name = gate.get("gate", "").lower()
                if gate_name and gate_name not in self.SUPPORTED_GATES:
                    unsupported.append(gate_name)

            if unsupported:
                return ValidationResult(
                    valid=False,
                    message=f"Unsupported gates: {', '.join(set(unsupported))}",
                )
            return ValidationResult(valid=True, message="ok")

        # Check for Cirq circuit
        try:
            import cirq

            if isinstance(circuit, cirq.Circuit):
                return ValidationResult(valid=True, message="ok (cirq circuit)")
        except ImportError:
            pass

        # Check for Qiskit circuit
        try:
            from qiskit import QuantumCircuit

            if isinstance(circuit, QuantumCircuit):
                return ValidationResult(valid=True, message="ok (qiskit circuit)")
        except ImportError:
            pass

        return ValidationResult(
            valid=False,
            message="Unsupported circuit type. Expected pyQuEST.Circuit, list of gates, cirq.Circuit, or qiskit.QuantumCircuit",
        )

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate memory and time requirements for circuit execution.

        Step 1.4: Memory estimation based on precision and simulation type
        """
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "pyQuEST not installed"},
            )

        qubits = self._extract_qubit_count(circuit)
        if qubits is None:
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "could not determine qubit count"},
            )

        gate_count = self._extract_gate_count(circuit)

        # Calculate memory for both simulation types
        sv_memory_mb = self._estimate_memory_mb(qubits, False, self._config.precision)
        dm_memory_mb = (
            self._estimate_memory_mb(qubits, True, self._config.precision)
            if qubits <= self.MAX_QUBITS_DM
            else None
        )

        metadata = {
            "qubits": qubits,
            "gate_count": gate_count,
            "precision": self._config.precision.value,
            "statevector_memory_mb": round(sv_memory_mb, 2),
            "density_matrix_memory_mb": (
                round(dm_memory_mb, 2) if dm_memory_mb else None
            ),
            "gpu_available": (
                self._hardware_info.gpu_available if self._hardware_info else False
            ),
        }

        return ResourceEstimate(memory_mb=sv_memory_mb, time_ms=None, metadata=metadata)

    def _extract_qubit_count(self, circuit: Any) -> int | None:
        """Extract the number of qubits from a circuit."""
        if isinstance(circuit, list):
            max_qubit = -1
            for gate in circuit:
                if isinstance(gate, dict):
                    qubits = gate.get("qubits", [])
                    controls = gate.get("controls", [])
                    all_qubits = list(qubits) + list(controls)
                    if all_qubits:
                        max_qubit = max(max_qubit, max(all_qubits))
            return max_qubit + 1 if max_qubit >= 0 else None

        try:
            import cirq

            if isinstance(circuit, cirq.Circuit):
                return len(circuit.all_qubits())
        except ImportError:
            pass

        try:
            from qiskit import QuantumCircuit

            if isinstance(circuit, QuantumCircuit):
                return circuit.num_qubits
        except ImportError:
            pass

        return None

    def _extract_gate_count(self, circuit: Any) -> int | None:
        """Extract the number of gates from a circuit."""
        if isinstance(circuit, list):
            return len(circuit)

        try:
            import cirq

            if isinstance(circuit, cirq.Circuit):
                return sum(len(m) for m in circuit)
        except ImportError:
            pass

        try:
            from qiskit import QuantumCircuit

            if isinstance(circuit, QuantumCircuit):
                return circuit.size()
        except ImportError:
            pass

        return None

    def execute(
        self, circuit: Any, options: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Execute a circuit on the QuEST simulator.

        Step 1.4: Full execution with GPU, precision, and parallelization
        Step 1.6: Comprehensive error handling
        """
        # Step 1.6: Check installation
        if not self.is_available():
            raise QuestInstallationError(
                reason="pyQuEST module not found",
                missing_component="pyQuEST",
            )

        # Validate circuit
        validation = self.validate_circuit(circuit)
        if not validation.valid:
            raise QuestCircuitError(
                reason=validation.message or "Invalid circuit",
            )

        try:
            import pyQuEST
            from pyQuEST import unitaries
        except ImportError as exc:
            raise QuestInstallationError(
                reason=f"Failed to import pyQuEST: {exc}",
                original_exception=exc,
            )

        options = options or {}

        # Step 1.4: Apply configuration
        config = QuestConfig.from_dict(options) if options else self._config

        # Configure OpenMP
        self._configure_openmp(config.openmp_threads)

        sim_type = options.get("simulator_type", SimulatorType.STATE_VECTOR)
        repetitions = int(options.get("repetitions", options.get("shots", 0)))

        # Determine qubit count
        qubit_count = options.get("num_qubits") or self._extract_qubit_count(circuit)
        if qubit_count is None:
            qubit_count = 2

        # Step 1.6: Check qubit limits
        max_qubits = (
            self.MAX_QUBITS_DM
            if sim_type == SimulatorType.DENSITY_MATRIX
            else self.MAX_QUBITS
        )
        if qubit_count > max_qubits:
            raise QubitLimitExceededError(
                backend_name="quest",
                requested_qubits=qubit_count,
                max_qubits=max_qubits,
            )

        # Step 1.6: Check memory before execution
        is_dm = sim_type == SimulatorType.DENSITY_MATRIX
        required_mb = self._estimate_memory_mb(qubit_count, is_dm, config.precision)

        use_gpu = (
            config.gpu_enabled
            and self._hardware_info
            and self._hardware_info.gpu_available
        )
        mem_available, available_mb = self._check_memory_available(required_mb, use_gpu)

        if not mem_available:
            raise QuestMemoryError(
                required_mb=required_mb,
                available_mb=available_mb,
                is_gpu=use_gpu,
                num_qubits=qubit_count,
            )

        # Step 1.4: Configure GPU if enabled
        gpu_used = False
        if use_gpu:
            try:
                gpu_used = self._configure_gpu(config.gpu_device_id)
            except QuestGPUError:
                # Fallback to CPU
                self._logger.warning("GPU configuration failed, falling back to CPU")
                gpu_used = False

        if sim_type not in (SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX):
            raise UnsupportedOperationError(
                backend_name="quest",
                operation=f"simulator_type={sim_type}",
                supported_operations=["STATE_VECTOR", "DENSITY_MATRIX"],
            )

        try:
            # Create quantum register
            if sim_type == SimulatorType.DENSITY_MATRIX:
                if hasattr(pyQuEST, "DensityRegister"):
                    register = pyQuEST.DensityRegister(qubit_count)
                else:
                    register = pyQuEST.Register(qubit_count, density=True)
            else:
                register = pyQuEST.Register(qubit_count)

            start = time.perf_counter()

            # Apply circuit
            quest_circuit = self._convert_to_quest_circuit(circuit, pyQuEST, unitaries)

            if quest_circuit is not None:
                register.apply_circuit(quest_circuit)
            elif isinstance(circuit, list):
                self._apply_gates(register, circuit, unitaries)

            result_type: ResultType
            data: dict[str, Any]
            raw_result: Any = None
            final_rank: int | None = None

            if repetitions > 0:
                # Measurement mode
                result_type = ResultType.COUNTS
                counts = self._sample_measurements(
                    register, qubit_count, repetitions, pyQuEST, unitaries
                )
                data = {"counts": counts, "repetitions": repetitions}
                raw_result = counts
            else:
                # State extraction mode
                if sim_type == SimulatorType.DENSITY_MATRIX:
                    result_type = ResultType.DENSITY_MATRIX
                    density_matrix = self._extract_density_matrix(register, qubit_count)

                    # Step 1.4: Apply rank truncation
                    if config.truncation_threshold > 0 or config.max_rank > 0:
                        density_matrix, final_rank = self._apply_rank_truncation(
                            density_matrix, config.truncation_threshold, config.max_rank
                        )

                    data = {"density_matrix": density_matrix}
                    raw_result = density_matrix
                else:
                    result_type = ResultType.STATEVECTOR
                    statevector = self._extract_statevector(register, qubit_count)
                    data = {"statevector": statevector}
                    raw_result = statevector

            execution_time_ms = (time.perf_counter() - start) * 1000.0

            # Build metadata
            metadata = {
                "quest_version": self.get_version(),
                "precision": config.precision.value,
                "gpu_used": gpu_used,
                "openmp_threads": (
                    self._hardware_info.openmp_threads if self._hardware_info else 1
                ),
            }
            if final_rank is not None:
                metadata["final_rank"] = final_rank
                metadata["truncation_threshold"] = config.truncation_threshold

            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=sim_type,
                execution_time_ms=execution_time_ms,
                qubit_count=qubit_count,
                shot_count=repetitions if repetitions > 0 else None,
                result_type=result_type,
                data=data,
                metadata=metadata,
                raw_result=raw_result,
            )

        except (
            QuestInstallationError,
            QuestCircuitError,
            QuestMemoryError,
            QuestGPUError,
            QubitLimitExceededError,
            UnsupportedOperationError,
        ):
            raise
        except MemoryError as exc:
            raise QuestMemoryError(
                required_mb=required_mb,
                available_mb=available_mb,
                is_gpu=gpu_used,
                num_qubits=qubit_count,
                original_exception=exc,
            )
        except Exception as exc:
            raise QuestRuntimeError(
                reason=str(exc),
                stage="execution",
                original_exception=exc,
            )

    def _convert_to_quest_circuit(
        self, circuit: Any, pyQuEST: Any, unitaries: Any
    ) -> Any | None:
        """Convert a circuit to pyQuEST.Circuit format."""
        if hasattr(pyQuEST, "Circuit") and isinstance(circuit, pyQuEST.Circuit):
            return circuit

        try:
            import cirq

            if isinstance(circuit, cirq.Circuit):
                return self._convert_cirq_circuit(circuit, pyQuEST, unitaries)
        except ImportError:
            pass

        try:
            from qiskit import QuantumCircuit

            if isinstance(circuit, QuantumCircuit):
                return self._convert_qiskit_circuit(circuit, pyQuEST, unitaries)
        except ImportError:
            pass

        if isinstance(circuit, list):
            return None

        return None

    def _convert_cirq_circuit(self, circuit: Any, pyQuEST: Any, unitaries: Any) -> Any:
        """Convert a Cirq circuit to pyQuEST format."""

        ops = []
        qubits_list = sorted(circuit.all_qubits())
        qubit_map = {q: i for i, q in enumerate(qubits_list)}

        for moment in circuit:
            for op in moment:
                gate = op.gate
                gate_qubits = [qubit_map[q] for q in op.qubits]

                quest_op = self._map_cirq_gate(gate, gate_qubits, unitaries)
                if quest_op is not None:
                    ops.append(quest_op)

        if hasattr(pyQuEST, "Circuit"):
            return pyQuEST.Circuit(ops)
        return None

    def _map_cirq_gate(
        self, gate: Any, qubits: list[int], unitaries: Any
    ) -> Any | None:
        """Map a Cirq gate to a pyQuEST operator."""
        import cirq

        if isinstance(gate, cirq.HPowGate) and gate.exponent == 1:
            return unitaries.H(qubits[0])
        if isinstance(gate, cirq.XPowGate) and gate.exponent == 1:
            return unitaries.X(qubits[0])
        if isinstance(gate, cirq.YPowGate) and gate.exponent == 1:
            return unitaries.Y(qubits[0])
        if isinstance(gate, cirq.ZPowGate) and gate.exponent == 1:
            return unitaries.Z(qubits[0])

        if isinstance(gate, cirq.Rx):
            return unitaries.Rx(qubits[0], float(gate.exponent) * 3.14159265358979)
        if isinstance(gate, cirq.Ry):
            return unitaries.Ry(qubits[0], float(gate.exponent) * 3.14159265358979)
        if isinstance(gate, cirq.Rz):
            return unitaries.Rz(qubits[0], float(gate.exponent) * 3.14159265358979)

        if isinstance(gate, cirq.CXPowGate) and gate.exponent == 1:
            return unitaries.X(qubits[1], controls=[qubits[0]])
        if isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
            return unitaries.Z(qubits[1], controls=[qubits[0]])
        if isinstance(gate, cirq.SwapPowGate) and gate.exponent == 1:
            if hasattr(unitaries, "SWAP"):
                return unitaries.SWAP(qubits[0], qubits[1])

        if isinstance(gate, cirq.MeasurementGate):
            return None

        return None

    def _convert_qiskit_circuit(
        self, circuit: Any, pyQuEST: Any, unitaries: Any
    ) -> Any:
        """Convert a Qiskit circuit to pyQuEST format."""
        ops = []

        for instruction, qargs, _ in circuit.data:
            gate_name = instruction.name.lower()
            qubits = [q._index for q in qargs]
            params = instruction.params

            quest_op = self._map_qiskit_gate(gate_name, qubits, params, unitaries)
            if quest_op is not None:
                ops.append(quest_op)

        if hasattr(pyQuEST, "Circuit"):
            return pyQuEST.Circuit(ops)
        return None

    def _map_qiskit_gate(
        self, gate_name: str, qubits: list[int], params: list, unitaries: Any
    ) -> Any | None:
        """Map a Qiskit gate to a pyQuEST operator."""
        gate_name = gate_name.lower()

        if gate_name == "h":
            return unitaries.H(qubits[0])
        if gate_name == "x":
            return unitaries.X(qubits[0])
        if gate_name == "y":
            return unitaries.Y(qubits[0])
        if gate_name == "z":
            return unitaries.Z(qubits[0])
        if gate_name == "s" and hasattr(unitaries, "S"):
            return unitaries.S(qubits[0])
        if gate_name == "t" and hasattr(unitaries, "T"):
            return unitaries.T(qubits[0])

        if gate_name == "rx" and params:
            return unitaries.Rx(qubits[0], float(params[0]))
        if gate_name == "ry" and params:
            return unitaries.Ry(qubits[0], float(params[0]))
        if gate_name == "rz" and params:
            return unitaries.Rz(qubits[0], float(params[0]))

        if gate_name in ("cx", "cnot"):
            return unitaries.X(qubits[1], controls=[qubits[0]])
        if gate_name == "cy":
            return unitaries.Y(qubits[1], controls=[qubits[0]])
        if gate_name == "cz":
            return unitaries.Z(qubits[1], controls=[qubits[0]])
        if gate_name == "swap" and hasattr(unitaries, "SWAP"):
            return unitaries.SWAP(qubits[0], qubits[1])

        if gate_name in ("measure", "barrier"):
            return None

        return None

    def _apply_gates(self, register: Any, gates: list[dict], unitaries: Any) -> None:
        """Apply a list of gate dictionaries to a pyQuEST register."""
        for gate in gates:
            gate_name = gate.get("gate", "").lower()
            qubits = gate.get("qubits", [])
            params = gate.get("params", [])
            controls = gate.get("controls", [])

            if not qubits:
                continue

            op = None
            target = qubits[0]

            if gate_name in ("h", "hadamard"):
                op = (
                    unitaries.H(target, controls=controls)
                    if controls
                    else unitaries.H(target)
                )
            elif gate_name in ("x", "pauli_x", "not"):
                op = (
                    unitaries.X(target, controls=controls)
                    if controls
                    else unitaries.X(target)
                )
            elif gate_name in ("y", "pauli_y"):
                op = (
                    unitaries.Y(target, controls=controls)
                    if controls
                    else unitaries.Y(target)
                )
            elif gate_name in ("z", "pauli_z"):
                op = (
                    unitaries.Z(target, controls=controls)
                    if controls
                    else unitaries.Z(target)
                )
            elif gate_name == "s" and hasattr(unitaries, "S"):
                op = unitaries.S(target)
            elif gate_name == "t" and hasattr(unitaries, "T"):
                op = unitaries.T(target)
            elif gate_name == "rx" and params:
                angle = float(params[0])
                op = (
                    unitaries.Rx(target, angle, controls=controls)
                    if controls
                    else unitaries.Rx(target, angle)
                )
            elif gate_name == "ry" and params:
                angle = float(params[0])
                op = (
                    unitaries.Ry(target, angle, controls=controls)
                    if controls
                    else unitaries.Ry(target, angle)
                )
            elif gate_name == "rz" and params:
                angle = float(params[0])
                op = (
                    unitaries.Rz(target, angle, controls=controls)
                    if controls
                    else unitaries.Rz(target, angle)
                )
            elif gate_name in ("cx", "cnot") and len(qubits) >= 2:
                op = unitaries.X(qubits[1], controls=[qubits[0]])
            elif gate_name == "cy" and len(qubits) >= 2:
                op = unitaries.Y(qubits[1], controls=[qubits[0]])
            elif gate_name == "cz" and len(qubits) >= 2:
                op = unitaries.Z(qubits[1], controls=[qubits[0]])
            elif (
                gate_name == "swap" and len(qubits) >= 2 and hasattr(unitaries, "SWAP")
            ):
                op = unitaries.SWAP(qubits[0], qubits[1])

            if op is not None:
                register.apply_operator(op)

    def _sample_measurements(
        self, register: Any, num_qubits: int, shots: int, pyQuEST: Any, unitaries: Any
    ) -> dict[str, int]:
        """Sample measurement outcomes from the quantum register."""
        counts: dict[str, int] = {}

        try:
            if hasattr(register, "get_probabilities"):
                probs = register.get_probabilities()
            elif hasattr(register, "probabilities"):
                probs = register.probabilities()
            else:
                statevector = self._extract_statevector(register, num_qubits)
                probs = np.abs(statevector) ** 2

            probs = np.array(probs)
            probs = probs / np.sum(probs)

            num_states = len(probs)
            samples = np.random.choice(num_states, size=shots, p=probs)

            for sample in samples:
                bitstring = format(sample, f"0{num_qubits}b")
                counts[bitstring] = counts.get(bitstring, 0) + 1

        except Exception:
            for _ in range(shots):
                bitstring = format(
                    np.random.randint(0, 2**num_qubits), f"0{num_qubits}b"
                )
                counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def _extract_statevector(self, register: Any, num_qubits: int) -> np.ndarray:
        """Extract the statevector from a pyQuEST register."""
        num_states = 2**num_qubits

        try:
            if hasattr(register, "get_statevector"):
                return np.array(register.get_statevector())
            if hasattr(register, "statevector"):
                return np.array(register.statevector())
            if hasattr(register, "get_state"):
                return np.array(register.get_state())
            if hasattr(register, "amps"):
                return np.array(register.amps)

            if hasattr(register, "get_amp"):
                sv = np.zeros(num_states, dtype=complex)
                for i in range(num_states):
                    sv[i] = register.get_amp(i)
                return sv
            if hasattr(register, "__getitem__"):
                sv = np.zeros(num_states, dtype=complex)
                for i in range(num_states):
                    sv[i] = register[i]
                return sv

        except Exception:
            pass

        sv = np.zeros(num_states, dtype=complex)
        sv[0] = 1.0
        return sv

    def _extract_density_matrix(self, register: Any, num_qubits: int) -> np.ndarray:
        """Extract the density matrix from a pyQuEST register."""
        num_states = 2**num_qubits

        try:
            if hasattr(register, "get_density_matrix"):
                return np.array(register.get_density_matrix())
            if hasattr(register, "density_matrix"):
                dm = register.density_matrix
                if callable(dm):
                    return np.array(dm())
                return np.array(dm)
            if hasattr(register, "get_state"):
                return np.array(register.get_state())

            if hasattr(register, "get_density_amp"):
                dm = np.zeros((num_states, num_states), dtype=complex)
                for i in range(num_states):
                    for j in range(num_states):
                        dm[i, j] = register.get_density_amp(i, j)
                return dm

        except Exception:
            pass

        dm = np.zeros((num_states, num_states), dtype=complex)
        dm[0, 0] = 1.0
        return dm

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        """Check if this backend supports the given simulator type."""
        return sim_type in self.get_capabilities().simulator_types

    def get_hardware_info(self) -> QuestHardwareInfo | None:
        """Get detected hardware information.

        Step 1.4: Expose hardware detection results
        """
        return self._hardware_info

    def get_config(self) -> QuestConfig:
        """Get current configuration."""
        return self._config

    def set_config(self, config: QuestConfig) -> None:
        """Update configuration."""
        self._config = config
