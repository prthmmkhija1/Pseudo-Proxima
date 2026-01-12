"""qsim backend adapter for high-performance CPU state vector simulations.

Phase 3: qsim Integration
=========================

Step 3.1: qsim Architecture Understanding
-----------------------------------------
qsim is Google's high-performance quantum circuit simulator optimized for CPU.
Key features:
- Highly optimized for Intel/AMD CPUs with AVX2/AVX512 vectorization
- Automatic gate fusion for performance optimization
- OpenMP parallelization across CPU cores
- Seamless Cirq integration via qsimcirq package

qsim Components:
- qsimcirq.QSimSimulator: Main simulator class
- QSimOptions: Configuration for simulation parameters
- Gate fusion: Automatically combines sequential gates for efficiency

Performance Characteristics:
- 10-100x faster than standard Cirq simulator for large circuits
- Best for circuits with 20+ qubits on multi-core CPUs
- AVX2/AVX512 provides 2-4x speedup over scalar operations
- OpenMP scales well up to 16-32 cores

References:
- qsim: https://github.com/quantumlib/qsim
- qsimcirq: https://pypi.org/project/qsimcirq/

Step 3.2: qsim Adapter Implementation
Step 3.3: Performance Optimization
Step 3.4: Limitation Handling
"""

from __future__ import annotations

import importlib.util
import logging
import math
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
    BackendNotInstalledError,
    CircuitValidationError,
    MemoryExceededError,
    QubitLimitExceededError,
    wrap_backend_exception,
)

# =============================================================================
# Step 3.1: qsim Configuration Enums and Dataclasses
# =============================================================================


class QsimVectorization(str, Enum):
    """CPU vectorization instruction set support."""

    SCALAR = "scalar"  # No SIMD vectorization
    SSE = "sse"  # SSE instructions (128-bit)
    AVX = "avx"  # AVX instructions (256-bit)
    AVX2 = "avx2"  # AVX2 instructions (256-bit, more ops)
    AVX512 = "avx512"  # AVX-512 instructions (512-bit)


class QsimGateFusion(str, Enum):
    """Gate fusion optimization levels."""

    OFF = "off"  # No gate fusion
    LOW = "low"  # Conservative fusion
    MEDIUM = "medium"  # Balanced fusion (default)
    HIGH = "high"  # Aggressive fusion


@dataclass
class QsimCPUInfo:
    """Information about CPU capabilities for qsim.

    Attributes:
        core_count: Number of physical CPU cores
        thread_count: Number of logical threads (with hyperthreading)
        vectorization: Best available vectorization instruction set
        has_avx2: Whether AVX2 is supported
        has_avx512: Whether AVX-512 is supported
        cache_size_kb: L3 cache size in KB (if detectable)
        cpu_name: CPU model name
    """

    core_count: int = 1
    thread_count: int = 1
    vectorization: QsimVectorization = QsimVectorization.SCALAR
    has_avx2: bool = False
    has_avx512: bool = False
    cache_size_kb: int = 0
    cpu_name: str = "Unknown CPU"


@dataclass
class QsimConfig:
    """Configuration options for qsim execution.

    Step 3.3: Performance configuration options.

    Attributes:
        num_threads: Number of OpenMP threads (0 = auto-detect)
        gate_fusion: Gate fusion optimization level
        max_fused_gate_size: Maximum gates to fuse together
        verbosity: Logging verbosity (0 = silent, 1 = info, 2 = debug)
        use_gpu: Whether to use GPU (requires qsim-gpu, usually False)
        max_qubits: Maximum supported qubits
        seed: Random seed for reproducibility (None = random)
    """

    num_threads: int = 0  # 0 = auto-detect
    gate_fusion: QsimGateFusion = QsimGateFusion.MEDIUM
    max_fused_gate_size: int = 4
    verbosity: int = 0
    use_gpu: bool = False
    max_qubits: int = 35
    seed: int | None = None


# =============================================================================
# Step 3.4: qsim-Specific Error Classes
# =============================================================================


class QsimError(BackendError):
    """Base exception for qsim-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: BackendErrorCode = BackendErrorCode.EXECUTION_FAILED,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            backend_name="qsim",
            error_code=error_code,
            details=details or {},
        )


class QsimInstallationError(QsimError):
    """Raised when qsimcirq is not properly installed."""

    def __init__(self, missing_component: str, install_hint: str = ""):
        details = {
            "missing_component": missing_component,
            "install_hint": install_hint or self._get_install_hint(missing_component),
        }
        super().__init__(
            message=f"qsim dependency missing: {missing_component}. {details['install_hint']}",
            error_code=BackendErrorCode.NOT_INSTALLED,
            details=details,
        )

    @staticmethod
    def _get_install_hint(component: str) -> str:
        """Get installation hint for missing component."""
        hints = {
            "qsimcirq": "Install qsimcirq: pip install qsimcirq",
            "cirq": "Install Cirq: pip install cirq",
            "openmp": "OpenMP should be installed with your compiler",
        }
        return hints.get(component.lower(), f"Please install {component}")


class QsimGateError(QsimError):
    """Raised when circuit contains unsupported gates."""

    def __init__(self, unsupported_gates: list[str], suggestions: str = ""):
        details = {
            "unsupported_gates": unsupported_gates,
            "suggestions": suggestions,
        }
        super().__init__(
            message=f"Circuit contains unsupported gates: {', '.join(unsupported_gates)}. {suggestions}",
            error_code=BackendErrorCode.CIRCUIT_INVALID,
            details=details,
        )


class QsimMemoryError(MemoryExceededError):
    """Raised when circuit requires too much memory for qsim."""

    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        qubit_count: int,
    ):
        super().__init__(
            backend_name="qsim",
            required_mb=required_mb,
            available_mb=available_mb,
            circuit_info={"qubit_count": qubit_count},
        )


# =============================================================================
# Step 3.2 & 3.3: QsimAdapter Implementation
# =============================================================================


class QsimAdapter(BaseBackendAdapter):
    """qsim backend adapter for high-performance CPU quantum simulation.

    This adapter provides access to Google's qsim simulator through the qsimcirq
    Python interface. qsim is optimized for multi-core CPUs with AVX2/AVX512
    support and provides significant speedups over standard simulators.

    Key Features:
    - AVX2/AVX512 vectorization for fast state vector operations
    - OpenMP parallelization across CPU cores
    - Automatic gate fusion for reduced gate overhead
    - Seamless integration with Cirq circuits
    - Support for circuits up to 35+ qubits (memory dependent)

    Example:
        >>> adapter = QsimAdapter()
        >>> if adapter.is_available():
        ...     result = adapter.execute(circuit, {"shots": 1000})

    Note:
        qsim only supports state vector simulation. Density matrix
        simulation is not available. Use CirqBackendAdapter for
        density matrix simulations.
    """

    def __init__(self, config: QsimConfig | None = None):
        """Initialize qsim adapter.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or QsimConfig()
        self._logger = logging.getLogger(__name__)

        # CPU detection results (lazy initialization)
        self._cpu_info: QsimCPUInfo | None = None
        self._qsimcirq_available: bool | None = None
        self._cirq_available: bool | None = None

        # Detect environment
        self._detect_cpu_capabilities()
        self._check_dependencies()

    # =========================================================================
    # Step 3.1: CPU Detection and Environment Setup
    # =========================================================================

    def _detect_cpu_capabilities(self) -> None:
        """Detect CPU capabilities for optimal qsim configuration.

        Detects:
        - Number of CPU cores and threads
        - AVX2/AVX512 support for vectorization
        - CPU model and cache information
        """
        self._logger.debug("Detecting CPU capabilities for qsim...")

        # Get core count
        try:
            import psutil

            core_count = psutil.cpu_count(logical=False) or 1
            thread_count = psutil.cpu_count(logical=True) or 1
        except ImportError:
            core_count = os.cpu_count() or 1
            thread_count = core_count

        # Detect vectorization support
        vectorization = QsimVectorization.SCALAR
        has_avx2 = False
        has_avx512 = False
        cpu_name = "Unknown CPU"

        # Try to detect CPU features
        try:
            # Method 1: Use cpuinfo package if available
            import cpuinfo

            info = cpuinfo.get_cpu_info()
            cpu_name = info.get("brand_raw", "Unknown CPU")
            flags = info.get("flags", [])

            if "avx512f" in flags or "avx512" in flags:
                vectorization = QsimVectorization.AVX512
                has_avx512 = True
                has_avx2 = True
            elif "avx2" in flags:
                vectorization = QsimVectorization.AVX2
                has_avx2 = True
            elif "avx" in flags:
                vectorization = QsimVectorization.AVX
            elif "sse4_2" in flags or "sse" in flags:
                vectorization = QsimVectorization.SSE

        except ImportError:
            # Method 2: Infer from platform
            import platform

            cpu_name = platform.processor() or "Unknown CPU"

            # Most modern Intel/AMD CPUs have AVX2
            if any(x in cpu_name.lower() for x in ["intel", "amd", "ryzen", "core"]):
                vectorization = QsimVectorization.AVX2
                has_avx2 = True

        self._cpu_info = QsimCPUInfo(
            core_count=core_count,
            thread_count=thread_count,
            vectorization=vectorization,
            has_avx2=has_avx2,
            has_avx512=has_avx512,
            cache_size_kb=0,  # Hard to detect portably
            cpu_name=cpu_name,
        )

        self._logger.info(
            f"CPU detection complete: {cpu_name}, "
            f"{core_count} cores, {thread_count} threads, "
            f"vectorization={vectorization.value}"
        )

    def _check_dependencies(self) -> None:
        """Check if qsimcirq and cirq are available."""
        self._cirq_available = importlib.util.find_spec("cirq") is not None
        self._qsimcirq_available = importlib.util.find_spec("qsimcirq") is not None

        self._logger.info(
            f"Dependency check: cirq={self._cirq_available}, "
            f"qsimcirq={self._qsimcirq_available}"
        )

    def get_cpu_info(self) -> QsimCPUInfo:
        """Get detected CPU information.

        Returns:
            QsimCPUInfo with CPU capabilities
        """
        if self._cpu_info is None:
            self._detect_cpu_capabilities()
        return self._cpu_info  # type: ignore

    # =========================================================================
    # Step 3.3: Thread Configuration for Performance
    # =========================================================================

    def _get_optimal_thread_count(self, qubit_count: int) -> int:
        """Calculate optimal thread count based on circuit size.

        Step 3.3: Performance tuning based on circuit size.

        Args:
            qubit_count: Number of qubits in the circuit

        Returns:
            Optimal number of threads to use
        """
        if self._config.num_threads > 0:
            return self._config.num_threads

        cpu_info = self.get_cpu_info()
        max_threads = cpu_info.thread_count

        # Tuning based on circuit size
        if qubit_count < 15:
            # Small circuits: limited parallelism benefit
            # Use fewer threads to reduce overhead
            return min(4, max_threads)
        elif qubit_count < 20:
            # Medium circuits: moderate parallelism
            return min(8, max_threads)
        elif qubit_count < 25:
            # Large circuits: good parallelism
            return min(16, max_threads)
        else:
            # Very large circuits: use all available threads
            return max_threads

    def _get_fusion_settings(self, qubit_count: int, depth: int) -> dict[str, Any]:
        """Get gate fusion settings based on circuit characteristics.

        Args:
            qubit_count: Number of qubits
            depth: Circuit depth

        Returns:
            Dictionary with fusion configuration
        """
        fusion_level = self._config.gate_fusion
        max_fused = self._config.max_fused_gate_size

        # Adjust fusion based on circuit size
        if fusion_level == QsimGateFusion.OFF:
            return {"f": 1}  # Minimal fusion

        elif fusion_level == QsimGateFusion.LOW:
            return {"f": min(2, max_fused)}

        elif fusion_level == QsimGateFusion.MEDIUM:
            # Default balanced fusion
            if qubit_count < 15:
                return {"f": min(2, max_fused)}
            else:
                return {"f": min(4, max_fused)}

        else:  # HIGH
            # Aggressive fusion for large circuits
            if depth > 100:
                return {"f": min(6, max_fused)}
            else:
                return {"f": min(4, max_fused)}

    # =========================================================================
    # BaseBackendAdapter Implementation
    # =========================================================================

    def get_name(self) -> str:
        """Return backend identifier."""
        return "qsim"

    def get_version(self) -> str:
        """Return qsimcirq version string."""
        if not self._qsimcirq_available:
            return "unavailable"

        try:
            import qsimcirq

            version = getattr(qsimcirq, "__version__", "unknown")

            # Also report Cirq version
            try:
                import cirq

                cirq_version = getattr(cirq, "__version__", "unknown")
                return f"qsimcirq={version}, cirq={cirq_version}"
            except ImportError:
                return f"qsimcirq={version}"

        except Exception:
            return "unknown"

    def is_available(self) -> bool:
        """Check if qsim backend is available.

        Returns True if:
        - qsimcirq package is installed
        - Cirq package is installed
        """
        return self._qsimcirq_available is True and self._cirq_available is True

    def get_capabilities(self) -> Capabilities:
        """Return qsim-specific capabilities.

        qsim is optimized for state vector simulation only.
        It does not support density matrix or noise simulation.
        """
        cpu_info = self.get_cpu_info()

        # Calculate max qubits based on available memory
        max_qubits = self._calculate_max_qubits()

        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR],  # SV only
            max_qubits=max_qubits,
            supports_noise=False,  # qsim doesn't support noise natively
            supports_gpu=False,  # GPU version is separate package
            supports_batching=True,
            custom_features={
                "cpu_optimized": True,
                "vectorization": cpu_info.vectorization.value,
                "has_avx2": cpu_info.has_avx2,
                "has_avx512": cpu_info.has_avx512,
                "openmp_threads": cpu_info.thread_count,
                "gate_fusion": True,
                "cpu_cores": cpu_info.core_count,
            },
        )

    def _calculate_max_qubits(self) -> int:
        """Calculate maximum supported qubits based on available memory."""
        try:
            import psutil

            available_memory_bytes = psutil.virtual_memory().available

            # State vector size: 2^n * 16 bytes (complex128)
            # Leave 2GB for system and overhead
            usable_bytes = available_memory_bytes - (2 * 1024 * 1024 * 1024)

            if usable_bytes <= 0:
                return 20  # Minimum reasonable value

            # Calculate max qubits: 2^n * 16 <= usable_bytes
            max_qubits = int(math.log2(usable_bytes / 16))

            # Cap at configured maximum
            return min(max_qubits, self._config.max_qubits)

        except ImportError:
            # Without psutil, use conservative estimate
            return min(30, self._config.max_qubits)

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate circuit for qsim execution.

        Step 3.4: Check for qsim limitations.

        Checks:
        1. qsimcirq availability
        2. Circuit is valid Cirq circuit
        3. No unsupported features (mid-circuit measurement, classical control)
        4. Qubit count within memory limits
        """
        # Check backend availability
        if not self.is_available():
            reasons = []
            if not self._cirq_available:
                reasons.append("Cirq not installed")
            if not self._qsimcirq_available:
                reasons.append("qsimcirq not installed")

            return ValidationResult(
                valid=False,
                message=f"qsim backend not available: {'; '.join(reasons)}",
                details={"reasons": reasons},
            )

        # Import Cirq for type checking
        try:
            import cirq
        except ImportError:
            return ValidationResult(
                valid=False,
                message="Cirq not installed",
            )

        # Check circuit type
        if not isinstance(circuit, cirq.Circuit):
            return ValidationResult(
                valid=False,
                message=f"Expected cirq.Circuit, got {type(circuit).__name__}",
            )

        # Check qubit count
        qubit_count = len(circuit.all_qubits())
        max_qubits = self._calculate_max_qubits()

        if qubit_count > max_qubits:
            return ValidationResult(
                valid=False,
                message=f"Circuit has {qubit_count} qubits, maximum supported is {max_qubits}",
                details={
                    "qubit_count": qubit_count,
                    "max_qubits": max_qubits,
                },
            )

        # Step 3.4: Check for unsupported features
        unsupported_gates = []
        has_mid_circuit_measurement = False

        moments = list(circuit)
        for i, moment in enumerate(moments):
            for op in moment:
                # Check for mid-circuit measurements (not at the end)
                if isinstance(op.gate, cirq.MeasurementGate):
                    if i < len(moments) - 1:
                        # Check if there are non-measurement ops after this
                        for later_moment in moments[i + 1 :]:
                            for later_op in later_moment:
                                if not isinstance(later_op.gate, cirq.MeasurementGate):
                                    has_mid_circuit_measurement = True
                                    break

                # Check for classical control (not well supported)
                if hasattr(op, "classical_controls") and op.classical_controls:
                    unsupported_gates.append(f"ClassicallyControlled({op.gate})")

        if has_mid_circuit_measurement:
            return ValidationResult(
                valid=False,
                message="qsim has limited support for mid-circuit measurements",
                details={
                    "suggestion": "Move measurements to end of circuit or use Cirq simulator"
                },
            )

        if unsupported_gates:
            return ValidationResult(
                valid=False,
                message=f"Circuit contains unsupported features: {', '.join(unsupported_gates[:5])}",
                details={
                    "unsupported_gates": unsupported_gates,
                    "suggestion": "Use Cirq simulator for classical control flow",
                },
            )

        return ValidationResult(
            valid=True,
            message="Circuit is valid for qsim execution",
            details={
                "qubit_count": qubit_count,
                "depth": len(circuit),
                "gate_count": sum(len(m) for m in circuit),
            },
        )

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate resources required for qsim execution.

        Args:
            circuit: Cirq circuit to estimate

        Returns:
            ResourceEstimate with memory and timing estimates
        """
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "qsim not available"},
            )

        try:
            import cirq
        except ImportError:
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "cirq import failed"},
            )

        if not isinstance(circuit, cirq.Circuit):
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "not a cirq.Circuit"},
            )

        qubit_count = len(circuit.all_qubits())
        gate_count = sum(len(m) for m in circuit)
        depth = len(circuit)

        # State vector memory: 2^n * 16 bytes (complex128)
        state_vector_bytes = (2**qubit_count) * 16
        state_vector_mb = state_vector_bytes / (1024 * 1024)

        # Add overhead for workspace (~20%)
        total_memory_mb = state_vector_mb * 1.2

        # Estimate time based on circuit characteristics
        # This is a rough estimate based on empirical observations
        cpu_info = self.get_cpu_info()

        # Base time per gate (microseconds) - varies by vectorization
        if cpu_info.has_avx512:
            us_per_gate = 0.1
        elif cpu_info.has_avx2:
            us_per_gate = 0.2
        else:
            us_per_gate = 0.5

        # Adjust for qubit count (larger state vectors are slower)
        qubit_factor = max(1, qubit_count / 10)

        estimated_time_ms = (gate_count * us_per_gate * qubit_factor) / 1000

        return ResourceEstimate(
            memory_mb=total_memory_mb if qubit_count <= 30 else None,
            time_ms=estimated_time_ms if qubit_count <= 25 else None,
            metadata={
                "qubits": qubit_count,
                "gate_count": gate_count,
                "depth": depth,
                "state_vector_mb": state_vector_mb,
                "vectorization": cpu_info.vectorization.value,
                "recommended_threads": self._get_optimal_thread_count(qubit_count),
            },
        )

    def execute(
        self, circuit: Any, options: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Execute circuit using qsim.

        Args:
            circuit: Cirq circuit to execute
            options: Execution options
                - shots/repetitions: Number of measurement shots (0 = statevector)
                - seed: Random seed for reproducibility
                - num_threads: Override thread count

        Returns:
            ExecutionResult with simulation results
        """
        if not self.is_available():
            raise BackendNotInstalledError("qsim", ["qsimcirq", "cirq"])

        # Validate circuit
        validation = self.validate_circuit(circuit)
        if not validation.valid:
            raise CircuitValidationError(
                backend_name="qsim",
                reason=validation.message or "Invalid circuit",
            )

        try:
            import cirq  # noqa: F401 - required by qsimcirq
            import qsimcirq
        except ImportError as exc:
            raise BackendNotInstalledError(
                "qsim", ["qsimcirq", "cirq"], original_exception=exc
            )

        options = options or {}
        repetitions = int(options.get("repetitions", options.get("shots", 0)))
        options.get("seed", self._config.seed)

        # Get circuit info
        qubit_count = len(circuit.all_qubits())
        depth = len(circuit)

        # Check qubit limits
        max_qubits = self._calculate_max_qubits()
        if qubit_count > max_qubits:
            raise QubitLimitExceededError(
                backend_name="qsim",
                requested_qubits=qubit_count,
                max_qubits=max_qubits,
            )

        try:
            # Step 3.3: Configure qsim for optimal performance
            num_threads = options.get("num_threads") or self._get_optimal_thread_count(
                qubit_count
            )
            fusion_settings = self._get_fusion_settings(qubit_count, depth)

            # Create qsim options
            qsim_options = qsimcirq.QSimOptions(
                max_fused_gate_size=fusion_settings.get("f", 4),
                cpu_threads=num_threads,
                verbosity=self._config.verbosity,
            )

            # Create simulator
            simulator = qsimcirq.QSimSimulator(qsim_options)

            self._logger.debug(
                f"Executing on qsim: {qubit_count} qubits, {depth} depth, "
                f"{num_threads} threads, fusion={fusion_settings}"
            )

            start = time.perf_counter()
            result_type: ResultType
            data: dict[str, Any]
            raw_result: Any

            if repetitions > 0:
                # Shot-based measurement
                raw_result = simulator.run(circuit, repetitions=repetitions)
                result_type = ResultType.COUNTS

                # Process measurement results
                counts: dict[str, int] = {}
                measurement_keys = list(raw_result.measurements.keys())

                if measurement_keys:
                    for key in measurement_keys:
                        histogram = raw_result.histogram(key=key)
                        for state_int, count in histogram.items():
                            n_bits = raw_result.measurements[key].shape[1]
                            bitstring = format(state_int, f"0{n_bits}b")
                            counts[bitstring] = counts.get(bitstring, 0) + count

                data = {"counts": counts, "repetitions": repetitions}

            else:
                # State vector simulation
                raw_result = simulator.simulate(circuit)
                statevector = raw_result.final_state_vector

                result_type = ResultType.STATEVECTOR
                data = {"statevector": statevector}

            execution_time_ms = (time.perf_counter() - start) * 1000.0

            cpu_info = self.get_cpu_info()

            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=SimulatorType.STATE_VECTOR,
                execution_time_ms=execution_time_ms,
                qubit_count=qubit_count,
                shot_count=repetitions if repetitions > 0 else None,
                result_type=result_type,
                data=data,
                metadata={
                    "qsim_version": self.get_version(),
                    "num_threads": num_threads,
                    "gate_fusion": fusion_settings,
                    "vectorization": cpu_info.vectorization.value,
                    "depth": depth,
                    "gate_count": sum(len(m) for m in circuit),
                },
                raw_result=raw_result,
            )

        except (
            BackendNotInstalledError,
            CircuitValidationError,
            QubitLimitExceededError,
        ):
            raise
        except MemoryError as exc:
            # Handle out of memory
            raise QsimMemoryError(
                required_mb=self.estimate_resources(circuit).memory_mb or 0,
                available_mb=0,
                qubit_count=qubit_count,
            ) from exc
        except Exception as exc:
            raise wrap_backend_exception(exc, "qsim", "execution")

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        """Check if simulator type is supported.

        qsim only supports state vector simulation.
        """
        return sim_type == SimulatorType.STATE_VECTOR


# =============================================================================
# Step 3.5: Helper Functions for Registry Integration
# =============================================================================


def get_qsim_config() -> dict[str, Any]:
    """Get default qsim configuration for defaults.py."""
    return {
        "enabled": True,
        "num_threads": 0,  # Auto-detect
        "gate_fusion": "medium",
        "max_fused_gate_size": 4,
        "verbosity": 0,
        "max_qubits": 35,
    }


def check_qsim_available() -> tuple[bool, str]:
    """Check if qsim is available and return status message.

    Returns:
        Tuple of (is_available, status_message)
    """
    cirq_available = importlib.util.find_spec("cirq") is not None
    qsimcirq_available = importlib.util.find_spec("qsimcirq") is not None

    if not cirq_available:
        return False, "Cirq not installed (pip install cirq)"
    if not qsimcirq_available:
        return False, "qsimcirq not installed (pip install qsimcirq)"

    return True, "qsim available and ready"


def get_qsim_performance_tier(qubit_count: int) -> str:
    """Determine performance tier for qsim based on qubit count.

    Args:
        qubit_count: Number of qubits

    Returns:
        Performance tier: "optimal", "good", "acceptable", or "slow"
    """
    if qubit_count <= 20:
        return "optimal"
    elif qubit_count <= 25:
        return "good"
    elif qubit_count <= 30:
        return "acceptable"
    else:
        return "slow"
