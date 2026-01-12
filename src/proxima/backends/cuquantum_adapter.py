"""cuQuantum backend adapter for GPU-accelerated state vector simulations.

This adapter provides GPU-accelerated quantum circuit simulation using NVIDIA's
cuQuantum SDK through Qiskit Aer. cuQuantum is NVIDIA's library for high-performance
quantum computing simulation on NVIDIA GPUs.

Step 2.1: cuQuantum Architecture Understanding
==============================================
cuQuantum consists of two main libraries:
1. cuStateVec: State vector simulation on GPU
2. cuTensorNet: Tensor network simulation on GPU

Key Integration Points:
- cuQuantum integrates with Qiskit Aer via device='GPU' option
- Requires NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA Toolkit 12.0+ and cuQuantum SDK required
- Memory requirements: 2^n * 16 bytes for n-qubit state vector

Performance Characteristics:
- GPU acceleration provides 10-100x speedup for large circuits
- Best for circuits with 20+ qubits where GPU memory allows
- Overhead for small circuits may negate benefits

References:
- cuQuantum SDK: https://github.com/NVIDIA/cuQuantum
- Qiskit Aer GPU: https://qiskit.github.io/qiskit-aer/

Step 2.2: Extended Qiskit Aer for GPU
Step 2.3: cuQuantum Configuration Helper
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

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
    MemoryExceededError,
    wrap_backend_exception,
)

# =============================================================================
# Step 2.1: cuQuantum Architecture - Configuration Enums and Dataclasses
# =============================================================================


class CuQuantumExecutionMode(str, Enum):
    """Execution modes for cuQuantum."""

    GPU_ONLY = "gpu_only"  # Force GPU execution, fail if unavailable
    GPU_PREFERRED = "gpu_preferred"  # Prefer GPU, fallback to CPU if needed
    AUTO = "auto"  # Auto-detect based on circuit size


class CuQuantumPrecision(str, Enum):
    """Numerical precision for cuQuantum simulations."""

    SINGLE = "single"  # complex64, 8 bytes per amplitude
    DOUBLE = "double"  # complex128, 16 bytes per amplitude


@dataclass
class GPUDeviceInfo:
    """Information about an NVIDIA GPU device.

    Attributes:
        device_id: GPU device index (0-based)
        name: GPU device name (e.g., "NVIDIA RTX 4090")
        compute_capability: Compute capability version (e.g., "8.9")
        total_memory_mb: Total GPU memory in MB
        free_memory_mb: Currently available GPU memory in MB
        cuda_version: CUDA runtime version string
        driver_version: NVIDIA driver version string
        is_cuquantum_compatible: Whether device supports cuQuantum (CC 7.0+)
    """

    device_id: int = 0
    name: str = "Unknown GPU"
    compute_capability: str = "0.0"
    total_memory_mb: int = 0
    free_memory_mb: int = 0
    cuda_version: str = "unknown"
    driver_version: str = "unknown"
    is_cuquantum_compatible: bool = False


@dataclass
class CuQuantumConfig:
    """Configuration options for cuQuantum execution.

    Step 2.2: GPU Configuration Options from the guide.

    Attributes:
        execution_mode: GPU execution mode (gpu_only, gpu_preferred, auto)
        gpu_device_id: Which GPU to use (0-indexed)
        precision: Numerical precision (single or double)
        memory_limit_mb: GPU memory limit in MB (0 = unlimited)
        workspace_size_mb: cuStateVec scratch memory in MB
        blocking: Whether to wait for GPU completion
        fusion_enabled: Enable gate fusion optimization
        max_qubits: Maximum qubits for GPU simulation
        fallback_to_cpu: Allow CPU fallback if GPU fails
    """

    execution_mode: CuQuantumExecutionMode = CuQuantumExecutionMode.GPU_PREFERRED
    gpu_device_id: int = 0
    precision: CuQuantumPrecision = CuQuantumPrecision.DOUBLE
    memory_limit_mb: int = 0  # 0 = unlimited
    workspace_size_mb: int = 1024  # 1GB default workspace
    blocking: bool = True
    fusion_enabled: bool = True
    max_qubits: int = 35
    fallback_to_cpu: bool = True


# =============================================================================
# Step 2.3: cuQuantum-Specific Error Classes
# =============================================================================


class CuQuantumError(BackendError):
    """Base exception for cuQuantum-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: BackendErrorCode = BackendErrorCode.EXECUTION_FAILED,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            backend_name="cuquantum",
            error_code=error_code,
            details=details or {},
        )


class CuQuantumInstallationError(CuQuantumError):
    """Raised when cuQuantum/CUDA dependencies are missing."""

    def __init__(self, missing_component: str, install_hint: str = ""):
        details = {
            "missing_component": missing_component,
            "install_hint": install_hint or self._get_install_hint(missing_component),
        }
        super().__init__(
            message=f"cuQuantum dependency missing: {missing_component}. {details['install_hint']}",
            error_code=BackendErrorCode.NOT_INSTALLED,
            details=details,
        )

    @staticmethod
    def _get_install_hint(component: str) -> str:
        """Get installation hint for missing component."""
        hints = {
            "cuda": "Install CUDA Toolkit 12.0+ from https://developer.nvidia.com/cuda-downloads",
            "cuquantum": "Install cuQuantum SDK: pip install cuquantum-cu12",
            "qiskit-aer-gpu": "Install Qiskit Aer GPU: pip install qiskit-aer-gpu",
            "pycuda": "Install PyCUDA: pip install pycuda",
            "cupy": "Install CuPy: pip install cupy-cuda12x",
            "nvidia-driver": "Install NVIDIA driver 525.60+ from https://www.nvidia.com/drivers",
        }
        return hints.get(component.lower(), f"Please install {component}")


class CuQuantumGPUError(CuQuantumError):
    """Raised when GPU device errors occur."""

    def __init__(
        self, message: str, device_id: int = 0, gpu_info: GPUDeviceInfo | None = None
    ):
        details = {
            "device_id": device_id,
            "gpu_info": gpu_info.__dict__ if gpu_info else None,
        }
        super().__init__(
            message=f"GPU error on device {device_id}: {message}",
            error_code=BackendErrorCode.HARDWARE_UNAVAILABLE,
            details=details,
        )


class CuQuantumMemoryError(MemoryExceededError):
    """Raised when GPU memory is insufficient for simulation."""

    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        qubit_count: int,
        device_id: int = 0,
    ):
        super().__init__(
            backend_name="cuquantum",
            required_mb=required_mb,
            available_mb=available_mb,
            circuit_info={"qubit_count": qubit_count, "device_id": device_id},
        )
        self.device_id = device_id


# =============================================================================
# Step 2.2 & 2.3: CuQuantumAdapter Implementation
# =============================================================================


class CuQuantumAdapter(BaseBackendAdapter):
    """cuQuantum backend adapter for GPU-accelerated quantum simulation.

    This adapter extends Qiskit Aer functionality to leverage NVIDIA GPUs
    for high-performance state vector simulation via cuQuantum SDK.

    Key Features:
    - GPU-accelerated state vector simulation
    - Automatic GPU detection and configuration
    - Memory management and estimation
    - Fallback to CPU when GPU unavailable
    - Support for circuits up to 35+ qubits (GPU memory dependent)

    Example:
        >>> adapter = CuQuantumAdapter()
        >>> if adapter.is_available():
        ...     result = adapter.execute(circuit, {"shots": 1000})
    """

    def __init__(self, config: CuQuantumConfig | None = None):
        """Initialize cuQuantum adapter.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or CuQuantumConfig()
        self._logger = logging.getLogger(__name__)

        # GPU detection results (lazy initialization)
        self._gpu_available: bool | None = None
        self._gpu_devices: list[GPUDeviceInfo] = []
        self._cuda_available: bool | None = None
        self._cuquantum_available: bool | None = None
        self._qiskit_gpu_available: bool | None = None

        # Initialize GPU detection
        self._detect_gpu_environment()

    # =========================================================================
    # GPU Detection Methods (Step 2.2)
    # =========================================================================

    def _detect_gpu_environment(self) -> None:
        """Detect GPU availability and capabilities.

        This method checks for:
        1. NVIDIA GPU hardware
        2. CUDA toolkit installation
        3. cuQuantum SDK availability
        4. Qiskit Aer GPU support
        """
        self._logger.debug("Detecting GPU environment for cuQuantum...")

        # Check CUDA availability
        self._cuda_available = self._check_cuda_available()

        if self._cuda_available:
            # Detect GPU devices
            self._gpu_devices = self._detect_gpu_devices()
            self._gpu_available = len(self._gpu_devices) > 0

            # Check cuQuantum SDK
            self._cuquantum_available = self._check_cuquantum_available()

            # Check Qiskit Aer GPU
            self._qiskit_gpu_available = self._check_qiskit_gpu_available()
        else:
            self._gpu_available = False
            self._cuquantum_available = False
            self._qiskit_gpu_available = False

        self._logger.info(
            f"GPU detection complete: cuda={self._cuda_available}, "
            f"gpu_count={len(self._gpu_devices)}, "
            f"cuquantum={self._cuquantum_available}, "
            f"qiskit_gpu={self._qiskit_gpu_available}"
        )

    def _check_cuda_available(self) -> bool:
        """Check if CUDA toolkit is available."""
        # Method 1: Try pycuda
        try:
            import pycuda.driver as cuda

            cuda.init()
            return cuda.Device.count() > 0
        except Exception:
            pass

        # Method 2: Try cupy
        try:
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
            return True
        except Exception:
            pass

        # Method 3: Check environment variable
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            return True

        # Method 4: Try importing cuda module from numba
        try:
            from numba import cuda as numba_cuda

            return numba_cuda.is_available()
        except Exception:
            pass

        return False

    def _detect_gpu_devices(self) -> list[GPUDeviceInfo]:
        """Detect available NVIDIA GPU devices."""
        devices: list[GPUDeviceInfo] = []

        # Try pycuda first
        try:
            import pycuda.driver as cuda

            cuda.init()

            for i in range(cuda.Device.count()):
                device = cuda.Device(i)
                cc = device.compute_capability()
                cc_str = f"{cc[0]}.{cc[1]}"

                # Get memory info
                total_mem = device.total_memory() // (1024 * 1024)

                # Get free memory (requires context)
                try:
                    context = device.make_context()
                    free_mem, _ = cuda.mem_get_info()
                    free_mem = free_mem // (1024 * 1024)
                    context.pop()
                except Exception:
                    free_mem = total_mem  # Assume all available

                devices.append(
                    GPUDeviceInfo(
                        device_id=i,
                        name=device.name(),
                        compute_capability=cc_str,
                        total_memory_mb=total_mem,
                        free_memory_mb=free_mem,
                        cuda_version=".".join(map(str, cuda.get_version())),
                        driver_version=cuda.get_driver_version(),
                        is_cuquantum_compatible=(cc[0] >= 7),
                    )
                )
            return devices
        except Exception:
            pass

        # Try cupy as fallback
        try:
            import cupy as cp

            for i in range(cp.cuda.runtime.getDeviceCount()):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    cc_str = f"{props['major']}.{props['minor']}"
                    total_mem = props["totalGlobalMem"] // (1024 * 1024)

                    # Get free memory
                    try:
                        free_mem = cp.cuda.runtime.memGetInfo()[0] // (1024 * 1024)
                    except Exception:
                        free_mem = total_mem

                    devices.append(
                        GPUDeviceInfo(
                            device_id=i,
                            name=(
                                props["name"].decode()
                                if isinstance(props["name"], bytes)
                                else str(props["name"])
                            ),
                            compute_capability=cc_str,
                            total_memory_mb=total_mem,
                            free_memory_mb=free_mem,
                            cuda_version=f"{cp.cuda.runtime.runtimeGetVersion() // 1000}.{(cp.cuda.runtime.runtimeGetVersion() % 1000) // 10}",
                            driver_version=str(cp.cuda.runtime.driverGetVersion()),
                            is_cuquantum_compatible=(props["major"] >= 7),
                        )
                    )
            return devices
        except Exception:
            pass

        return devices

    def _check_cuquantum_available(self) -> bool:
        """Check if cuQuantum SDK is available."""
        # Check for custatevec
        if importlib.util.find_spec("cuquantum") is not None:
            try:
                return True
            except Exception:
                pass

        # Check for custatevec directly
        if importlib.util.find_spec("custatevec") is not None:
            return True

        return False

    def _check_qiskit_gpu_available(self) -> bool:
        """Check if Qiskit Aer GPU support is available."""
        if importlib.util.find_spec("qiskit_aer") is None:
            return False

        try:
            from qiskit_aer import AerSimulator

            # Try to create GPU simulator
            AerSimulator(method="statevector", device="GPU")
            return True
        except Exception:
            pass

        return False

    def get_gpu_info(self, device_id: int | None = None) -> GPUDeviceInfo | None:
        """Get information about a specific GPU device.

        Args:
            device_id: GPU device index. Uses configured device if None.

        Returns:
            GPUDeviceInfo or None if device not found.
        """
        if device_id is None:
            device_id = self._config.gpu_device_id

        for device in self._gpu_devices:
            if device.device_id == device_id:
                return device

        return None

    def get_all_gpu_info(self) -> list[GPUDeviceInfo]:
        """Get information about all detected GPU devices."""
        return self._gpu_devices.copy()

    # =========================================================================
    # BaseBackendAdapter Implementation
    # =========================================================================

    def get_name(self) -> str:
        """Return backend identifier."""
        return "cuquantum"

    def get_version(self) -> str:
        """Return cuQuantum/Qiskit Aer version string."""
        versions = []

        # Get cuQuantum version
        try:
            import cuquantum

            versions.append(f"cuquantum={getattr(cuquantum, '__version__', 'unknown')}")
        except Exception:
            pass

        # Get Qiskit Aer version
        try:
            import qiskit_aer

            versions.append(
                f"qiskit-aer={getattr(qiskit_aer, '__version__', 'unknown')}"
            )
        except Exception:
            pass

        # Get CUDA version
        if self._gpu_devices:
            versions.append(f"cuda={self._gpu_devices[0].cuda_version}")

        return ", ".join(versions) if versions else "unavailable"

    def is_available(self) -> bool:
        """Check if cuQuantum backend is available.

        Returns True if:
        - GPU hardware is detected
        - CUDA is available
        - Either cuQuantum SDK or Qiskit Aer GPU is available
        """
        return (
            self._gpu_available is True
            and self._cuda_available is True
            and (self._cuquantum_available or self._qiskit_gpu_available)
        )

    def get_capabilities(self) -> Capabilities:
        """Return GPU-specific capabilities.

        cuQuantum is optimized for state vector simulation only.
        Density matrix and noise are not supported on GPU via cuQuantum.
        """
        # Determine max qubits based on GPU memory
        max_qubits = self._calculate_max_qubits()

        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR],
            max_qubits=max_qubits,
            supports_noise=False,  # cuQuantum via Aer doesn't support noise on GPU
            supports_gpu=True,
            supports_batching=False,
            custom_features={
                "gpu_accelerated": True,
                "cuquantum_sdk": self._cuquantum_available,
                "qiskit_aer_gpu": self._qiskit_gpu_available,
                "precision": self._config.precision.value,
                "fusion_enabled": self._config.fusion_enabled,
                "gpu_devices": len(self._gpu_devices),
                "compute_capability": (
                    self._gpu_devices[0].compute_capability
                    if self._gpu_devices
                    else None
                ),
            },
        )

    def _calculate_max_qubits(self) -> int:
        """Calculate maximum supported qubits based on GPU memory."""
        if not self._gpu_devices:
            return self._config.max_qubits

        # Get configured device
        device = self.get_gpu_info(self._config.gpu_device_id)
        if not device:
            device = self._gpu_devices[0]

        # Calculate max qubits from available memory
        # State vector size: 2^n * 16 bytes (complex128) or 2^n * 8 bytes (complex64)
        bytes_per_amplitude = (
            16 if self._config.precision == CuQuantumPrecision.DOUBLE else 8
        )

        # Reserve memory for workspace and overhead (20% of total)
        usable_memory_bytes = device.free_memory_mb * 1024 * 1024 * 0.8

        # max_qubits where 2^n * bytes_per_amplitude <= usable_memory
        import math

        max_qubits = int(math.log2(usable_memory_bytes / bytes_per_amplitude))

        # Cap at configured maximum
        return min(max_qubits, self._config.max_qubits)

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate circuit for GPU execution.

        Checks:
        1. GPU/cuQuantum availability
        2. Circuit is valid Qiskit QuantumCircuit
        3. Qubit count within GPU memory limits
        4. No unsupported features (mid-circuit measurement, etc.)
        """
        # Check backend availability
        if not self.is_available():
            reasons = []
            if not self._cuda_available:
                reasons.append("CUDA not available")
            if not self._gpu_available:
                reasons.append("No GPU detected")
            if not (self._cuquantum_available or self._qiskit_gpu_available):
                reasons.append("Neither cuQuantum SDK nor Qiskit Aer GPU available")

            return ValidationResult(
                valid=False,
                message=f"cuQuantum backend not available: {'; '.join(reasons)}",
                details={"reasons": reasons},
            )

        # Import Qiskit
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            return ValidationResult(
                valid=False,
                message="Qiskit not installed",
            )

        # Check circuit type
        if not isinstance(circuit, QuantumCircuit):
            return ValidationResult(
                valid=False,
                message=f"Expected qiskit.QuantumCircuit, got {type(circuit).__name__}",
            )

        # Check qubit count
        max_qubits = self._calculate_max_qubits()
        if circuit.num_qubits > max_qubits:
            required_memory = self._estimate_memory_mb(circuit.num_qubits)
            device = self.get_gpu_info()
            available_memory = device.free_memory_mb if device else 0

            return ValidationResult(
                valid=False,
                message=f"Circuit requires {circuit.num_qubits} qubits but GPU supports max {max_qubits}",
                details={
                    "requested_qubits": circuit.num_qubits,
                    "max_qubits": max_qubits,
                    "required_memory_mb": required_memory,
                    "available_memory_mb": available_memory,
                },
            )

        # Check for unsupported operations
        unsupported = []
        for instruction in circuit.data:
            op_name = instruction.operation.name
            # Mid-circuit measurements not well supported on GPU
            if op_name == "measure" and instruction != circuit.data[-1]:
                unsupported.append("mid-circuit measurement")
            # Reset operations limited
            if op_name == "reset":
                unsupported.append("reset operation")

        if unsupported:
            return ValidationResult(
                valid=False,
                message=f"Circuit contains unsupported GPU operations: {', '.join(set(unsupported))}",
                details={"unsupported_operations": list(set(unsupported))},
            )

        return ValidationResult(valid=True, message="Circuit valid for GPU execution")

    def _estimate_memory_mb(self, num_qubits: int) -> float:
        """Estimate GPU memory required for state vector simulation.

        Formula: (2^n * bytes_per_amplitude + workspace) / (1024^2)
        """
        bytes_per_amplitude = (
            16 if self._config.precision == CuQuantumPrecision.DOUBLE else 8
        )

        # State vector size
        sv_size = (2**num_qubits) * bytes_per_amplitude

        # Add workspace
        workspace = self._config.workspace_size_mb * 1024 * 1024

        # Add overhead (10%)
        total = (sv_size + workspace) * 1.1

        return total / (1024 * 1024)

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate GPU resources for circuit execution."""
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "cuQuantum not available"},
            )

        try:
            from qiskit import QuantumCircuit
        except ImportError:
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "Qiskit not installed"},
            )

        if not isinstance(circuit, QuantumCircuit):
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "Not a QuantumCircuit"},
            )

        num_qubits = circuit.num_qubits
        depth = circuit.depth() or 0
        gate_count = circuit.size()

        # Memory estimation
        memory_mb = self._estimate_memory_mb(num_qubits)

        # Time estimation (rough heuristic based on gate count and depth)
        # GPU is ~10-100x faster than CPU for large circuits
        base_time_ms = gate_count * 0.001  # 1 microsecond per gate baseline
        if num_qubits > 25:
            base_time_ms *= 2 ** (num_qubits - 25)  # Exponential scaling

        device = self.get_gpu_info()

        return ResourceEstimate(
            memory_mb=memory_mb,
            time_ms=base_time_ms,
            metadata={
                "qubits": num_qubits,
                "depth": depth,
                "gate_count": gate_count,
                "precision": self._config.precision.value,
                "gpu_device": device.name if device else "unknown",
                "gpu_memory_available_mb": device.free_memory_mb if device else 0,
            },
        )

    def execute(
        self, circuit: Any, options: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Execute circuit on GPU using cuQuantum/Qiskit Aer.

        Args:
            circuit: Qiskit QuantumCircuit to execute
            options: Execution options including:
                - shots: Number of measurement shots (0 for statevector)
                - simulator_type: Must be STATE_VECTOR for cuQuantum
                - gpu_device_id: Override default GPU device
                - precision: Override default precision

        Returns:
            ExecutionResult with GPU execution metrics

        Raises:
            CuQuantumInstallationError: If dependencies missing
            CuQuantumGPUError: If GPU execution fails
            CuQuantumMemoryError: If GPU memory insufficient
        """
        if not self.is_available():
            if self._config.fallback_to_cpu:
                self._logger.warning("GPU not available, falling back to CPU execution")
                return self._execute_on_cpu(circuit, options)
            raise CuQuantumInstallationError(
                "cuda",
                "GPU not available and fallback disabled",
            )

        # Validate circuit
        validation = self.validate_circuit(circuit)
        if not validation.valid:
            if self._config.fallback_to_cpu:
                self._logger.warning(
                    f"Circuit validation failed: {validation.message}. Falling back to CPU."
                )
                return self._execute_on_cpu(circuit, options)
            raise CircuitValidationError(
                backend_name="cuquantum",
                reason=validation.message or "Validation failed",
                circuit_info=validation.details,
            )

        options = options or {}
        shots = int(options.get("shots", options.get("repetitions", 0)))

        # Execute on GPU
        try:
            return self._execute_on_gpu(circuit, shots, options)
        except CuQuantumMemoryError:
            if self._config.fallback_to_cpu:
                self._logger.warning("GPU out of memory, falling back to CPU")
                return self._execute_on_cpu(circuit, options)
            raise
        except Exception as exc:
            if self._config.fallback_to_cpu:
                self._logger.warning(
                    f"GPU execution failed: {exc}. Falling back to CPU."
                )
                return self._execute_on_cpu(circuit, options)
            raise wrap_backend_exception(exc, "cuquantum", "gpu_execution")

    def _execute_on_gpu(
        self,
        circuit: Any,
        shots: int,
        options: dict[str, Any],
    ) -> ExecutionResult:
        """Execute circuit on GPU via Qiskit Aer.

        Step 2.2: GPU Execution Path
        """
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        qubit_count = circuit.num_qubits

        # Check memory before execution
        required_memory = self._estimate_memory_mb(qubit_count)
        device = self.get_gpu_info(
            options.get("gpu_device_id", self._config.gpu_device_id)
        )

        if device and required_memory > device.free_memory_mb:
            raise CuQuantumMemoryError(
                required_mb=required_memory,
                available_mb=device.free_memory_mb,
                qubit_count=qubit_count,
                device_id=device.device_id,
            )

        # Configure GPU simulator
        sim_options = {
            "method": "statevector",
            "device": "GPU",
            "precision": self._config.precision.value,
            "blocking_enable": self._config.blocking,
            "fusion_enable": self._config.fusion_enabled,
        }

        if self._config.workspace_size_mb > 0:
            sim_options["cuStateVec_enable"] = True

        try:
            simulator = AerSimulator(**sim_options)
        except Exception as exc:
            raise CuQuantumGPUError(
                f"Failed to initialize GPU simulator: {exc}",
                device_id=self._config.gpu_device_id,
                gpu_info=device,
            )

        # Transpile for GPU
        t_circuit = transpile(circuit, simulator)

        # Prepare circuit
        exec_circuit = t_circuit.copy()
        if shots == 0:
            exec_circuit.save_statevector()

        # Execute
        start = time.perf_counter()
        try:
            result = simulator.run(
                exec_circuit,
                shots=shots if shots > 0 else None,
            ).result()
        except Exception as exc:
            error_str = str(exc).lower()
            if "memory" in error_str or "cuda" in error_str:
                raise CuQuantumMemoryError(
                    required_mb=required_memory,
                    available_mb=device.free_memory_mb if device else 0,
                    qubit_count=qubit_count,
                    device_id=self._config.gpu_device_id,
                )
            raise CuQuantumGPUError(
                f"GPU execution failed: {exc}",
                device_id=self._config.gpu_device_id,
                gpu_info=device,
            )

        execution_time_ms = (time.perf_counter() - start) * 1000.0

        # Process results
        if shots > 0:
            counts = result.get_counts(exec_circuit)
            data = {"counts": counts, "shots": shots}
            result_type = ResultType.COUNTS
        else:
            result_data = result.data(exec_circuit)
            statevector = result_data.get("statevector")
            data = {"statevector": statevector}
            result_type = ResultType.STATEVECTOR

        return ExecutionResult(
            backend=self.get_name(),
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=execution_time_ms,
            qubit_count=qubit_count,
            shot_count=shots if shots > 0 else None,
            result_type=result_type,
            data=data,
            metadata={
                "gpu_device": device.name if device else "unknown",
                "gpu_device_id": self._config.gpu_device_id,
                "precision": self._config.precision.value,
                "fusion_enabled": self._config.fusion_enabled,
                "cuquantum_version": self.get_version(),
                "execution_mode": "gpu",
            },
            raw_result=result,
        )

    def _execute_on_cpu(
        self,
        circuit: Any,
        options: dict[str, Any] | None,
    ) -> ExecutionResult:
        """Fallback CPU execution via Qiskit Aer.

        Used when GPU is unavailable or memory insufficient.
        """
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        options = options or {}
        shots = int(options.get("shots", options.get("repetitions", 0)))
        qubit_count = circuit.num_qubits

        simulator = AerSimulator(method="statevector")
        t_circuit = transpile(circuit, simulator)

        exec_circuit = t_circuit.copy()
        if shots == 0:
            exec_circuit.save_statevector()

        start = time.perf_counter()
        result = simulator.run(
            exec_circuit, shots=shots if shots > 0 else None
        ).result()
        execution_time_ms = (time.perf_counter() - start) * 1000.0

        if shots > 0:
            counts = result.get_counts(exec_circuit)
            data = {"counts": counts, "shots": shots}
            result_type = ResultType.COUNTS
        else:
            result_data = result.data(exec_circuit)
            statevector = result_data.get("statevector")
            data = {"statevector": statevector}
            result_type = ResultType.STATEVECTOR

        return ExecutionResult(
            backend=self.get_name(),
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=execution_time_ms,
            qubit_count=qubit_count,
            shot_count=shots if shots > 0 else None,
            result_type=result_type,
            data=data,
            metadata={
                "execution_mode": "cpu_fallback",
                "fallback_reason": "GPU unavailable or insufficient memory",
            },
            raw_result=result,
        )

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        """Check if simulator type is supported.

        cuQuantum only supports state vector simulation.
        """
        return sim_type == SimulatorType.STATE_VECTOR
