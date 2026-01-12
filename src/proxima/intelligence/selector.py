"""
Backend Selector with LLM-Enhanced Selection.

Intelligent backend auto-selection based on circuit characteristics,
backend capabilities, and optional LLM-assisted recommendations.

Step 4.1: Updated for Phase 4 - Unified Backend Selection Enhancement
- Added GPU-aware selection
- Added new backends: QuEST, cuQuantum, qsim
- Enhanced priority-based selection
- Memory-based and performance-based selection
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from proxima.core.circuit import Circuit

logger = logging.getLogger(__name__)


# ==============================================================================
# Enums
# ==============================================================================


class SelectionStrategy(Enum):
    """Strategy for backend selection."""

    PERFORMANCE = auto()  # Prioritize speed
    ACCURACY = auto()  # Prioritize simulation fidelity
    MEMORY = auto()  # Prioritize low memory usage
    BALANCED = auto()  # Balance all factors
    LLM_ASSISTED = auto()  # Use LLM for complex decisions
    GPU_PREFERRED = auto()  # Step 4.1: Prefer GPU backends
    CPU_OPTIMIZED = auto()  # Step 4.1: Prefer CPU-optimized backends


class BackendType(Enum):
    """Type of simulation backend."""

    NUMPY = "numpy"
    CUPY = "cupy"
    QISKIT = "qiskit"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    QUEST = "quest"  # Step 4.1: QuEST backend
    CUQUANTUM = "cuquantum"  # Step 4.1: cuQuantum backend
    QSIM = "qsim"  # Step 4.1: qsim backend
    LRET = "lret"  # LRET backend
    CUSTOM = "custom"


class SimulationType(Enum):
    """Type of quantum simulation."""

    STATE_VECTOR = "state_vector"
    DENSITY_MATRIX = "density_matrix"
    NOISY = "noisy"


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class CircuitCharacteristics:
    """Analyzed characteristics of a quantum circuit."""

    qubit_count: int
    gate_count: int
    depth: int
    gate_types: set[str] = field(default_factory=set)
    entanglement_density: float = 0.0  # 0-1 ratio of entangling gates
    measurement_count: int = 0
    has_custom_gates: bool = False
    has_parameterized_gates: bool = False
    has_noise_model: bool = False  # Step 4.1: Track noise model
    has_mid_circuit_measurement: bool = False  # Step 4.1: Track MCM
    simulation_type: SimulationType = SimulationType.STATE_VECTOR  # Step 4.1
    estimated_memory_mb: float = 0.0

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> CircuitCharacteristics:
        """Analyze a circuit and extract its characteristics."""
        gate_types = set()
        entangling_gates = 0
        parameterized = False
        custom = False
        measurements = 0
        mid_circuit_measurement = False
        last_non_measure_idx = -1

        for idx, gate in enumerate(circuit.gates):
            gate_types.add(gate.name)

            # Check for entangling gates (multi-qubit)
            if len(gate.qubits) > 1:
                entangling_gates += 1

            # Check for measurements
            if gate.name.lower() in ("measure", "m", "measurement"):
                measurements += 1
                if last_non_measure_idx >= 0 and idx > last_non_measure_idx + 1:
                    mid_circuit_measurement = True
            else:
                last_non_measure_idx = idx

            # Check for parameterized gates
            if hasattr(gate, "params") and gate.params:
                parameterized = True

            # Check for custom gates
            if gate.name.lower() not in (
                "h",
                "x",
                "y",
                "z",
                "s",
                "t",
                "cx",
                "cnot",
                "cz",
                "swap",
                "rx",
                "ry",
                "rz",
                "u1",
                "u2",
                "u3",
                "i",
                "id",
                "identity",
                "measure",
                "m",
                "barrier",
                "ccx",
                "ccz",
                "toffoli",
            ):
                custom = True

        gate_count = len(circuit.gates)
        density = entangling_gates / gate_count if gate_count > 0 else 0.0

        # Estimate memory: state vector = 2^n complex numbers, 16 bytes each
        estimated_memory = (2**circuit.num_qubits) * 16 / (1024 * 1024)

        return cls(
            qubit_count=circuit.num_qubits,
            gate_count=gate_count,
            depth=circuit.depth,
            gate_types=gate_types,
            entanglement_density=density,
            measurement_count=measurements,
            has_custom_gates=custom,
            has_parameterized_gates=parameterized,
            has_mid_circuit_measurement=mid_circuit_measurement,
            estimated_memory_mb=estimated_memory,
        )


@dataclass
class BackendCapabilities:
    """Capabilities and characteristics of a simulation backend."""

    name: str
    backend_type: BackendType
    max_qubits: int = 30
    supports_gpu: bool = False
    supports_distributed: bool = False
    supports_noise: bool = False
    supports_custom_gates: bool = True
    supports_density_matrix: bool = False  # Step 4.1
    supports_state_vector: bool = True  # Step 4.1
    supports_mid_circuit_measurement: bool = True  # Step 4.1
    is_cpu_optimized: bool = False  # Step 4.1: AVX/vectorization
    supported_gate_types: set[str] = field(default_factory=set)
    performance_score: float = 1.0  # 0-1, higher is faster
    memory_efficiency: float = 1.0  # 0-1, higher uses less memory
    accuracy_score: float = 1.0  # 0-1, higher is more accurate


@dataclass
class SelectionScore:
    """Detailed scoring for a backend selection decision."""

    backend_name: str
    total_score: float
    feature_score: float  # Can handle all required features
    performance_score: float  # Speed rating
    memory_score: float  # Memory efficiency
    history_score: float  # Past success rate
    compatibility_score: float  # Gate/feature compatibility
    gpu_score: float = 0.0  # Step 4.1: GPU capability score

    # Individual component details
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """Result of backend selection with explanation."""

    selected_backend: str
    confidence: float  # 0-1, how confident in the selection
    scores: list[SelectionScore]  # All backends scored
    explanation: str  # Human-readable explanation
    reasoning_steps: list[str]  # Step-by-step reasoning
    warnings: list[str] = field(default_factory=list)  # Any concerns
    alternatives: list[str] = field(default_factory=list)  # Other good options
    llm_recommendation: str | None = None  # If LLM was used
    gpu_available: bool = False  # Step 4.1: Was GPU detected

    @property
    def backend(self) -> str:
        """Alias for selected_backend for backwards compatibility."""
        return self.selected_backend


# ==============================================================================
# GPU Detection - Step 4.1
# ==============================================================================


class GPUDetector:
    """Detects GPU availability and capabilities."""

    def __init__(self) -> None:
        self._gpu_available: bool | None = None
        self._gpu_memory_mb: float = 0.0
        self._gpu_name: str = ""

    def is_gpu_available(self) -> bool:
        """Check if a compatible GPU is available."""
        if self._gpu_available is not None:
            return self._gpu_available

        self._gpu_available = False

        # Try CUDA detection via cupy
        try:
            import cupy

            device_count = cupy.cuda.runtime.getDeviceCount()
            if device_count > 0:
                self._gpu_available = True
                # Get GPU info
                device = cupy.cuda.Device(0)
                self._gpu_memory_mb = device.mem_info[1] / (1024 * 1024)
                self._gpu_name = cupy.cuda.runtime.getDeviceProperties(0)[
                    "name"
                ].decode()
                logger.info(
                    f"GPU detected: {self._gpu_name} ({self._gpu_memory_mb:.0f} MB)"
                )
        except Exception:
            pass

        # Try pycuda as fallback
        if not self._gpu_available:
            try:
                import pycuda.driver as cuda

                self._gpu_available = True
                self._gpu_memory_mb = cuda.mem_get_info()[1] / (1024 * 1024)
                self._gpu_name = cuda.Device(0).name()
            except Exception:
                pass

        # Try torch CUDA
        if not self._gpu_available:
            try:
                import torch

                if torch.cuda.is_available():
                    self._gpu_available = True
                    self._gpu_memory_mb = torch.cuda.get_device_properties(
                        0
                    ).total_memory / (1024 * 1024)
                    self._gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass

        return self._gpu_available

    def get_gpu_memory_mb(self) -> float:
        """Get available GPU memory in MB."""
        self.is_gpu_available()  # Ensure detection ran
        return self._gpu_memory_mb

    def can_fit_circuit(self, qubit_count: int, precision: str = "double") -> bool:
        """Check if circuit can fit in GPU memory."""
        if not self.is_gpu_available():
            return False

        bytes_per_amplitude = 16 if precision == "double" else 8
        required_mb = (
            (2**qubit_count) * bytes_per_amplitude + 1024 * 1024 * 1024
        ) / (1024 * 1024)
        return required_mb < self._gpu_memory_mb * 0.8  # 80% threshold


# ==============================================================================
# Backend Registry - Step 4.1: Enhanced with new backends
# ==============================================================================


class BackendRegistry:
    """Registry of available backends and their capabilities."""

    def __init__(self) -> None:
        self._backends: dict[str, BackendCapabilities] = {}
        self._register_default_backends()

    def _register_default_backends(self) -> None:
        """Register the default simulation backends including new ones."""

        # LRET backend - custom rank-reduction
        self.register(
            BackendCapabilities(
                name="lret",
                backend_type=BackendType.LRET,
                max_qubits=15,
                supports_gpu=False,
                supports_noise=True,
                supports_density_matrix=True,
                supports_state_vector=True,
                performance_score=0.5,
                memory_efficiency=0.9,  # Rank-reduced
                accuracy_score=0.95,
            )
        )

        # Cirq backend - Google's framework
        self.register(
            BackendCapabilities(
                name="cirq",
                backend_type=BackendType.CIRQ,
                max_qubits=20,
                supports_gpu=False,
                supports_noise=True,
                supports_density_matrix=True,
                supports_state_vector=True,
                performance_score=0.75,
                memory_efficiency=0.65,
                accuracy_score=0.98,
            )
        )

        # Qiskit Aer backend - feature rich
        self.register(
            BackendCapabilities(
                name="qiskit",
                backend_type=BackendType.QISKIT,
                max_qubits=30,
                supports_gpu=False,
                supports_noise=True,
                supports_distributed=True,
                supports_density_matrix=True,
                supports_state_vector=True,
                performance_score=0.7,
                memory_efficiency=0.6,
                accuracy_score=0.98,
            )
        )

        # ==================================================================
        # Step 4.1: New backends registration
        # ==================================================================

        # QuEST backend - high-performance C++ with GPU option
        self.register(
            BackendCapabilities(
                name="quest",
                backend_type=BackendType.QUEST,
                max_qubits=30,
                supports_gpu=True,  # GPU supported
                supports_noise=True,
                supports_density_matrix=True,  # Native DM support
                supports_state_vector=True,
                is_cpu_optimized=True,  # OpenMP optimized
                performance_score=0.9,
                memory_efficiency=0.7,
                accuracy_score=0.99,
            )
        )

        # cuQuantum backend - NVIDIA GPU-accelerated
        self.register(
            BackendCapabilities(
                name="cuquantum",
                backend_type=BackendType.CUQUANTUM,
                max_qubits=35,  # GPU allows larger circuits
                supports_gpu=True,
                supports_noise=False,  # State vector only
                supports_density_matrix=False,
                supports_state_vector=True,
                supports_mid_circuit_measurement=False,
                performance_score=0.98,  # Fastest for GPU
                memory_efficiency=0.5,  # GPU memory limited
                accuracy_score=0.99,
            )
        )

        # qsim backend - Google's CPU-optimized simulator
        self.register(
            BackendCapabilities(
                name="qsim",
                backend_type=BackendType.QSIM,
                max_qubits=35,  # Very efficient for large circuits
                supports_gpu=False,
                supports_noise=False,  # State vector only
                supports_density_matrix=False,
                supports_state_vector=True,
                supports_mid_circuit_measurement=False,
                is_cpu_optimized=True,  # AVX2/AVX512 optimized
                performance_score=0.95,  # Fastest CPU simulator
                memory_efficiency=0.6,
                accuracy_score=0.99,
            )
        )

        # NumPy backend - always available, good for small circuits
        self.register(
            BackendCapabilities(
                name="numpy",
                backend_type=BackendType.NUMPY,
                max_qubits=25,
                supports_gpu=False,
                supports_noise=True,
                supports_density_matrix=True,
                supports_state_vector=True,
                performance_score=0.6,
                memory_efficiency=0.7,
                accuracy_score=1.0,
            )
        )

        # CuPy backend - GPU accelerated NumPy
        self.register(
            BackendCapabilities(
                name="cupy",
                backend_type=BackendType.CUPY,
                max_qubits=30,
                supports_gpu=True,
                supports_noise=True,
                supports_density_matrix=True,
                supports_state_vector=True,
                performance_score=0.95,
                memory_efficiency=0.5,  # GPU memory limited
                accuracy_score=0.99,  # Floating point on GPU
            )
        )

        # PennyLane backend - ML focused
        self.register(
            BackendCapabilities(
                name="pennylane",
                backend_type=BackendType.PENNYLANE,
                max_qubits=28,
                supports_gpu=True,
                supports_noise=True,
                supports_density_matrix=True,
                supports_state_vector=True,
                performance_score=0.8,
                memory_efficiency=0.6,
                accuracy_score=0.97,
            )
        )

    def register(self, backend: BackendCapabilities) -> None:
        """Register a backend."""
        self._backends[backend.name] = backend

    def get(self, name: str) -> BackendCapabilities | None:
        """Get a backend by name."""
        return self._backends.get(name)

    def list_all(self) -> list[BackendCapabilities]:
        """List all registered backends."""
        return list(self._backends.values())

    def list_compatible(
        self,
        characteristics: CircuitCharacteristics,
        check_runtime: bool = True,
    ) -> list[BackendCapabilities]:
        """List backends compatible with given circuit characteristics."""
        compatible = []

        for backend in self._backends.values():
            # Check qubit limit
            if characteristics.qubit_count > backend.max_qubits:
                continue

            # Check custom gate support
            if characteristics.has_custom_gates and not backend.supports_custom_gates:
                continue

            # Step 4.1: Check simulation type compatibility
            if characteristics.simulation_type == SimulationType.DENSITY_MATRIX:
                if not backend.supports_density_matrix:
                    continue

            # Step 4.1: Check mid-circuit measurement support
            if characteristics.has_mid_circuit_measurement:
                if not backend.supports_mid_circuit_measurement:
                    continue

            # Step 4.1: Check noise model support
            if characteristics.has_noise_model and not backend.supports_noise:
                continue

            # Runtime availability check
            if check_runtime and not self._is_backend_available(backend.name):
                continue

            compatible.append(backend)

        return compatible

    def _is_backend_available(self, backend_name: str) -> bool:
        """Check if a backend is actually available at runtime."""
        try:
            if backend_name == "numpy":
                import numpy  # noqa: F401

                return True
            elif backend_name == "cupy":
                try:
                    import cupy  # noqa: F401

                    cupy.cuda.runtime.getDeviceCount()
                    return True
                except Exception:
                    return False
            elif backend_name == "qiskit":
                try:
                    from qiskit import QuantumCircuit  # noqa: F401

                    return True
                except ImportError:
                    return False
            elif backend_name == "cirq":
                try:
                    import cirq  # noqa: F401

                    return True
                except ImportError:
                    return False
            elif backend_name == "pennylane":
                try:
                    import pennylane  # noqa: F401

                    return True
                except ImportError:
                    return False
            # Step 4.1: New backend availability checks
            elif backend_name == "quest":
                try:
                    import pyquest  # noqa: F401

                    return True
                except ImportError:
                    return False
            elif backend_name == "cuquantum":
                try:
                    from qiskit_aer import AerSimulator  # noqa: F401

                    # Check if GPU method is available
                    AerSimulator(method="statevector", device="GPU")
                    return True
                except Exception:
                    return False
            elif backend_name == "qsim":
                try:
                    import qsimcirq  # noqa: F401

                    return True
                except ImportError:
                    return False
            elif backend_name == "lret":
                try:
                    from lret import LRET  # noqa: F401

                    return True
                except ImportError:
                    return False
            else:
                # Unknown backends are assumed available
                return True
        except Exception:
            return False

    def get_available_backends(self) -> list[BackendCapabilities]:
        """Get list of backends that are actually available at runtime."""
        return [
            b for b in self._backends.values() if self._is_backend_available(b.name)
        ]

    # ==========================================================================
    # Step 4.1: GPU-aware backend selection helpers
    # ==========================================================================

    def get_gpu_backends(self) -> list[BackendCapabilities]:
        """Return list of GPU-enabled backends."""
        return [b for b in self._backends.values() if b.supports_gpu]

    def get_cpu_optimized_backends(self) -> list[BackendCapabilities]:
        """Return list of CPU-optimized backends."""
        return [b for b in self._backends.values() if b.is_cpu_optimized]

    def get_density_matrix_backends(self) -> list[BackendCapabilities]:
        """Return list of backends supporting density matrix simulation."""
        return [b for b in self._backends.values() if b.supports_density_matrix]


# ==============================================================================
# Selection Scoring - Step 4.1: Enhanced scoring
# ==============================================================================


class SelectionScorer:
    """Scores backends for selection based on various factors."""

    # Scoring weights by strategy - Step 4.1: Updated with GPU weights
    STRATEGY_WEIGHTS = {
        SelectionStrategy.PERFORMANCE: {
            "feature": 0.15,
            "performance": 0.45,
            "memory": 0.1,
            "history": 0.1,
            "compatibility": 0.1,
            "gpu": 0.1,
        },
        SelectionStrategy.ACCURACY: {
            "feature": 0.3,
            "performance": 0.1,
            "memory": 0.1,
            "history": 0.2,
            "compatibility": 0.3,
            "gpu": 0.0,
        },
        SelectionStrategy.MEMORY: {
            "feature": 0.2,
            "performance": 0.1,
            "memory": 0.5,
            "history": 0.1,
            "compatibility": 0.1,
            "gpu": 0.0,
        },
        SelectionStrategy.BALANCED: {
            "feature": 0.2,
            "performance": 0.2,
            "memory": 0.15,
            "history": 0.15,
            "compatibility": 0.15,
            "gpu": 0.15,
        },
        SelectionStrategy.LLM_ASSISTED: {
            "feature": 0.2,
            "performance": 0.2,
            "memory": 0.2,
            "history": 0.2,
            "compatibility": 0.2,
            "gpu": 0.0,
        },
        # Step 4.1: New GPU-preferred strategy
        SelectionStrategy.GPU_PREFERRED: {
            "feature": 0.1,
            "performance": 0.3,
            "memory": 0.1,
            "history": 0.1,
            "compatibility": 0.1,
            "gpu": 0.3,
        },
        # Step 4.1: New CPU-optimized strategy
        SelectionStrategy.CPU_OPTIMIZED: {
            "feature": 0.15,
            "performance": 0.4,
            "memory": 0.15,
            "history": 0.15,
            "compatibility": 0.15,
            "gpu": 0.0,
        },
    }

    def __init__(
        self,
        history_provider: Callable[[str], float] | None = None,
        gpu_detector: GPUDetector | None = None,
    ) -> None:
        """
        Initialize scorer.

        Args:
            history_provider: Optional callback that returns historical
                success rate (0-1) for a given backend name.
            gpu_detector: Optional GPU detector instance.
        """
        self._history_provider = history_provider
        self._gpu_detector = gpu_detector or GPUDetector()

    def score(
        self,
        backend: BackendCapabilities,
        characteristics: CircuitCharacteristics,
        strategy: SelectionStrategy,
    ) -> SelectionScore:
        """Score a backend for the given circuit and strategy."""
        weights = self.STRATEGY_WEIGHTS[strategy]

        # Feature score - does it have required capabilities?
        feature_score = self._score_features(backend, characteristics)

        # Performance score - how fast is it?
        performance_score = self._score_performance(backend, characteristics)

        # Memory score - can it handle the memory requirements?
        memory_score = self._score_memory(backend, characteristics)

        # History score - past success rate
        history_score = self._score_history(backend.name)

        # Compatibility score - gate type support
        compatibility_score = self._score_compatibility(backend, characteristics)

        # Step 4.1: GPU score - GPU availability and suitability
        gpu_score = self._score_gpu(backend, characteristics)

        # Calculate weighted total
        total = (
            weights["feature"] * feature_score
            + weights["performance"] * performance_score
            + weights["memory"] * memory_score
            + weights["history"] * history_score
            + weights["compatibility"] * compatibility_score
            + weights["gpu"] * gpu_score
        )

        return SelectionScore(
            backend_name=backend.name,
            total_score=total,
            feature_score=feature_score,
            performance_score=performance_score,
            memory_score=memory_score,
            history_score=history_score,
            compatibility_score=compatibility_score,
            gpu_score=gpu_score,
            details={
                "qubit_headroom": backend.max_qubits - characteristics.qubit_count,
                "gpu_available": backend.supports_gpu,
                "noise_support": backend.supports_noise,
                "density_matrix_support": backend.supports_density_matrix,
                "cpu_optimized": backend.is_cpu_optimized,
            },
        )

    def _score_features(
        self,
        backend: BackendCapabilities,
        characteristics: CircuitCharacteristics,
    ) -> float:
        """Score based on feature support."""
        score = 1.0

        # Penalty for not supporting custom gates when needed
        if characteristics.has_custom_gates and not backend.supports_custom_gates:
            score -= 0.5

        # Step 4.1: Check simulation type support
        if characteristics.simulation_type == SimulationType.DENSITY_MATRIX:
            if not backend.supports_density_matrix:
                score -= 0.8
            else:
                score += 0.1

        # Step 4.1: Check mid-circuit measurement
        if characteristics.has_mid_circuit_measurement:
            if not backend.supports_mid_circuit_measurement:
                score -= 0.6

        # Bonus for GPU support with large circuits
        if characteristics.qubit_count > 20 and backend.supports_gpu:
            score += 0.2

        # Bonus for noise support with measurements
        if characteristics.measurement_count > 0 and backend.supports_noise:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_performance(
        self,
        backend: BackendCapabilities,
        characteristics: CircuitCharacteristics,
    ) -> float:
        """Score based on expected performance."""
        base_score = backend.performance_score

        # Step 4.1: Boost for CPU-optimized backends on large circuits
        if characteristics.qubit_count > 20 and backend.is_cpu_optimized:
            base_score += 0.1

        # Step 4.1: Boost for GPU backends on very large circuits
        if characteristics.qubit_count > 25 and backend.supports_gpu:
            if self._gpu_detector.is_gpu_available():
                base_score += 0.15

        # Penalty for large circuits on low-performance backends
        if characteristics.qubit_count > 25:
            base_score *= 0.9 if backend.performance_score < 0.7 else 1.0

        return max(0.0, min(1.0, base_score))

    def _score_memory(
        self,
        backend: BackendCapabilities,
        characteristics: CircuitCharacteristics,
    ) -> float:
        """Score based on memory efficiency."""
        base_score = backend.memory_efficiency

        # Get system memory
        try:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            available_memory = 8192  # Assume 8GB

        # Check if circuit fits
        required_memory = characteristics.estimated_memory_mb
        if required_memory > available_memory * 0.8:
            base_score *= 0.5  # Heavy penalty if tight on memory

        # Step 4.1: GPU memory check for GPU backends
        if backend.supports_gpu and self._gpu_detector.is_gpu_available():
            if not self._gpu_detector.can_fit_circuit(characteristics.qubit_count):
                base_score *= 0.7  # Penalty if GPU memory insufficient

        return max(0.0, min(1.0, base_score))

    def _score_history(self, backend_name: str) -> float:
        """Score based on historical success rate."""
        if self._history_provider:
            return self._history_provider(backend_name)
        return 0.8  # Default neutral score

    def _score_compatibility(
        self,
        backend: BackendCapabilities,
        characteristics: CircuitCharacteristics,
    ) -> float:
        """Score based on gate type compatibility."""
        score = backend.accuracy_score

        # Check qubit headroom
        headroom = backend.max_qubits - characteristics.qubit_count
        if headroom < 2:
            score *= 0.9  # Slight penalty for being close to limit
        elif headroom > 10:
            score += 0.05  # Small bonus for plenty of headroom

        return max(0.0, min(1.0, score))

    def _score_gpu(
        self,
        backend: BackendCapabilities,
        characteristics: CircuitCharacteristics,
    ) -> float:
        """Step 4.1: Score based on GPU availability and suitability."""
        if not backend.supports_gpu:
            return 0.3  # Neutral score for non-GPU backends

        if not self._gpu_detector.is_gpu_available():
            return 0.1  # Low score if GPU backend but no GPU available

        # GPU is available and backend supports it
        score = 0.9

        # Bonus for large circuits that benefit from GPU
        if characteristics.qubit_count > 20:
            score += 0.1

        # Check if circuit fits in GPU memory
        if not self._gpu_detector.can_fit_circuit(characteristics.qubit_count):
            score *= 0.5  # Penalty if won't fit

        return max(0.0, min(1.0, score))


# ==============================================================================
# Explanation Generator
# ==============================================================================


class SelectionExplainer:
    """Generates human-readable explanations for backend selection."""

    def generate(
        self,
        result: SelectionResult,
        characteristics: CircuitCharacteristics,
        strategy: SelectionStrategy,
    ) -> str:
        """Generate explanation for why a backend was selected."""
        parts = []

        # Main selection statement
        parts.append(
            f"Selected '{result.selected_backend}' backend "
            f"(confidence: {result.confidence:.0%})"
        )

        # Strategy explanation
        strategy_explanations = {
            SelectionStrategy.PERFORMANCE: "prioritizing execution speed",
            SelectionStrategy.ACCURACY: "prioritizing simulation accuracy",
            SelectionStrategy.MEMORY: "prioritizing memory efficiency",
            SelectionStrategy.BALANCED: "balancing all factors",
            SelectionStrategy.LLM_ASSISTED: "using AI-assisted analysis",
            SelectionStrategy.GPU_PREFERRED: "prioritizing GPU acceleration",
            SelectionStrategy.CPU_OPTIMIZED: "prioritizing CPU optimization",
        }
        parts.append(f"Strategy: {strategy_explanations.get(strategy, 'balanced')}")

        # Circuit characteristics summary
        parts.append(
            f"Circuit: {characteristics.qubit_count} qubits, "
            f"{characteristics.gate_count} gates, depth {characteristics.depth}"
        )

        # Step 4.1: GPU status
        if result.gpu_available:
            parts.append("GPU: Available and considered in selection")
        else:
            parts.append("GPU: Not available, using CPU-based selection")

        # Memory estimate
        if characteristics.estimated_memory_mb > 100:
            parts.append(
                f"Estimated memory: {characteristics.estimated_memory_mb:.1f} MB"
            )

        return ". ".join(parts) + "."

    def generate_reasoning_steps(
        self,
        characteristics: CircuitCharacteristics,
        scores: list[SelectionScore],
        selected: str,
        strategy: SelectionStrategy,
    ) -> list[str]:
        """Generate step-by-step reasoning for selection."""
        steps = []

        # Step 1: Circuit analysis
        steps.append(
            f"1. Analyzed circuit: {characteristics.qubit_count} qubits, "
            f"{characteristics.gate_count} gates"
        )

        # Step 2: Strategy
        steps.append(f"2. Applied {strategy.name.lower().replace('_', ' ')} strategy")

        # Step 3: Compatible backends
        compatible_count = len([s for s in scores if s.total_score > 0.3])
        steps.append(f"3. Found {compatible_count} compatible backends")

        # Step 4: Scoring
        top_3 = sorted(scores, key=lambda s: s.total_score, reverse=True)[:3]
        score_summary = ", ".join(
            f"{s.backend_name}: {s.total_score:.2f}" for s in top_3
        )
        steps.append(f"4. Top scores: {score_summary}")

        # Step 5: Selection
        winner = next((s for s in scores if s.backend_name == selected), None)
        if winner:
            steps.append(
                f"5. Selected '{selected}' with score {winner.total_score:.2f} "
                f"(feature: {winner.feature_score:.2f}, "
                f"performance: {winner.performance_score:.2f}, "
                f"gpu: {winner.gpu_score:.2f})"
            )

        return steps


# ==============================================================================
# Backend Priority - Step 4.1
# ==============================================================================


class BackendPrioritySelector:
    """Step 4.1: Priority-based backend selection for specific use cases."""

    # Priority lists for different simulation scenarios
    PRIORITIES = {
        "state_vector_gpu": ["cuquantum", "quest", "qsim", "cirq", "qiskit"],
        "state_vector_cpu": ["qsim", "quest", "cirq", "qiskit", "numpy"],
        "density_matrix": ["quest", "cirq", "qiskit", "lret"],
        "noisy_circuit": ["quest", "qiskit", "cirq"],
        "small_circuit": ["numpy", "cirq", "qiskit"],  # < 15 qubits
        "large_circuit_gpu": ["cuquantum", "quest"],  # > 25 qubits with GPU
        "large_circuit_cpu": ["qsim", "quest"],  # > 25 qubits CPU only
    }

    def __init__(
        self,
        registry: BackendRegistry,
        gpu_detector: GPUDetector | None = None,
    ) -> None:
        self._registry = registry
        self._gpu_detector = gpu_detector or GPUDetector()

    def get_priority_list(
        self,
        characteristics: CircuitCharacteristics,
        prefer_gpu: bool = True,
    ) -> list[str]:
        """Get prioritized backend list based on circuit characteristics."""

        # Determine primary use case
        if characteristics.simulation_type == SimulationType.DENSITY_MATRIX:
            return self.PRIORITIES["density_matrix"]

        if characteristics.has_noise_model:
            return self.PRIORITIES["noisy_circuit"]

        # State vector simulation
        if characteristics.qubit_count < 15:
            return self.PRIORITIES["small_circuit"]

        if characteristics.qubit_count > 25:
            if prefer_gpu and self._gpu_detector.is_gpu_available():
                return self.PRIORITIES["large_circuit_gpu"]
            return self.PRIORITIES["large_circuit_cpu"]

        # Medium circuits
        if prefer_gpu and self._gpu_detector.is_gpu_available():
            return self.PRIORITIES["state_vector_gpu"]
        return self.PRIORITIES["state_vector_cpu"]

    def select_best_available(
        self,
        characteristics: CircuitCharacteristics,
        prefer_gpu: bool = True,
    ) -> str | None:
        """Select best available backend from priority list."""
        priority_list = self.get_priority_list(characteristics, prefer_gpu)

        for backend_name in priority_list:
            backend = self._registry.get(backend_name)
            if backend and self._registry._is_backend_available(backend_name):
                # Verify compatibility
                if backend.max_qubits >= characteristics.qubit_count:
                    if characteristics.simulation_type == SimulationType.DENSITY_MATRIX:
                        if backend.supports_density_matrix:
                            return backend_name
                    else:
                        return backend_name

        return None


# ==============================================================================
# Main Backend Selector - Step 4.1: Enhanced
# ==============================================================================


class BackendSelector:
    """Main backend selection engine with GPU awareness."""

    def __init__(
        self,
        registry: BackendRegistry | None = None,
        scorer: SelectionScorer | None = None,
        explainer: SelectionExplainer | None = None,
        gpu_detector: GPUDetector | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self._gpu_detector = gpu_detector or GPUDetector()
        self._registry = registry or BackendRegistry()
        self._scorer = scorer or SelectionScorer(gpu_detector=self._gpu_detector)
        self._explainer = explainer or SelectionExplainer()
        self._priority_selector = BackendPrioritySelector(
            self._registry, self._gpu_detector
        )
        self._llm_client = llm_client

    def select(
        self,
        circuit: Circuit,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
    ) -> SelectionResult:
        """Select the best backend for executing a circuit."""
        # Analyze circuit
        characteristics = CircuitCharacteristics.from_circuit(circuit)

        # Check GPU availability
        gpu_available = self._gpu_detector.is_gpu_available()

        # Step 4.1: Adjust strategy based on GPU availability
        if strategy == SelectionStrategy.GPU_PREFERRED and not gpu_available:
            logger.warning("GPU preferred but not available, falling back to balanced")
            strategy = SelectionStrategy.BALANCED

        # Get compatible backends
        compatible = self._registry.list_compatible(characteristics)

        if not compatible:
            return SelectionResult(
                selected_backend="numpy",  # Fallback
                confidence=0.1,
                scores=[],
                explanation="No compatible backends found, using numpy fallback",
                reasoning_steps=["No backends compatible with circuit requirements"],
                warnings=["Using fallback backend - results may be limited"],
                gpu_available=gpu_available,
            )

        # Score all compatible backends
        scores = [
            self._scorer.score(backend, characteristics, strategy)
            for backend in compatible
        ]

        # Sort by total score
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Generate warnings
        warnings = self._generate_warnings(characteristics, scores[0])

        # Get alternatives
        alternatives = [s.backend_name for s in scores[1:4] if s.total_score > 0.5]

        # Generate explanation and reasoning
        result = SelectionResult(
            selected_backend=scores[0].backend_name,
            confidence=scores[0].total_score,
            scores=scores,
            explanation="",
            reasoning_steps=[],
            warnings=warnings,
            alternatives=alternatives,
            gpu_available=gpu_available,
        )

        result.explanation = self._explainer.generate(result, characteristics, strategy)
        result.reasoning_steps = self._explainer.generate_reasoning_steps(
            characteristics, scores, scores[0].backend_name, strategy
        )

        return result

    def select_by_priority(
        self,
        circuit: Circuit,
        prefer_gpu: bool = True,
    ) -> SelectionResult:
        """Step 4.1: Select backend using priority-based selection."""
        characteristics = CircuitCharacteristics.from_circuit(circuit)
        gpu_available = self._gpu_detector.is_gpu_available()

        selected = self._priority_selector.select_best_available(
            characteristics, prefer_gpu and gpu_available
        )

        if not selected:
            return self.select(circuit, SelectionStrategy.BALANCED)

        return SelectionResult(
            selected_backend=selected,
            confidence=0.9,
            scores=[],
            explanation=f"Selected '{selected}' based on priority for "
            f"{characteristics.simulation_type.value} simulation",
            reasoning_steps=[
                f"Used priority-based selection for {characteristics.qubit_count}-qubit circuit",
                f"GPU available: {gpu_available}, prefer_gpu: {prefer_gpu}",
                f"Selected: {selected}",
            ],
            gpu_available=gpu_available,
        )

    def _generate_warnings(
        self,
        characteristics: CircuitCharacteristics,
        top_score: SelectionScore,
    ) -> list[str]:
        """Generate warnings about potential issues."""
        warnings = []

        if characteristics.qubit_count > 25:
            warnings.append(
                f"Large circuit ({characteristics.qubit_count} qubits) - "
                "simulation may be slow"
            )

        if characteristics.estimated_memory_mb > 1000:
            warnings.append(
                f"High memory requirement ({characteristics.estimated_memory_mb:.0f} MB)"
            )

        if top_score.total_score < 0.5:
            warnings.append("Low confidence in selection - consider alternatives")

        if characteristics.has_custom_gates:
            warnings.append("Custom gates may have limited backend support")

        # Step 4.1: GPU-specific warnings
        if (
            characteristics.qubit_count > 25
            and not self._gpu_detector.is_gpu_available()
        ):
            warnings.append(
                "Large circuit without GPU - consider using GPU for better performance"
            )

        return warnings

    def _build_llm_prompt(
        self,
        characteristics: CircuitCharacteristics,
        scores: list[SelectionScore],
    ) -> str:
        """Build prompt for LLM recommendation."""
        return f"""You are a quantum computing expert. Given this circuit analysis,
recommend the best simulation backend and explain why.

Circuit characteristics:
- Qubits: {characteristics.qubit_count}
- Gates: {characteristics.gate_count}
- Depth: {characteristics.depth}
- Simulation type: {characteristics.simulation_type.value}
- Entanglement density: {characteristics.entanglement_density:.2f}
- Custom gates: {characteristics.has_custom_gates}
- Parameterized: {characteristics.has_parameterized_gates}
- Memory estimate: {characteristics.estimated_memory_mb:.1f} MB
- GPU available: {self._gpu_detector.is_gpu_available()}

Backend scores:
{chr(10).join(f'- {s.backend_name}: {s.total_score:.2f}' for s in scores[:5])}

Provide a brief recommendation (2-3 sentences) on which backend to use and why."""

    def register_backend(self, backend: BackendCapabilities) -> None:
        """Register a custom backend."""
        self._registry.register(backend)

    def list_backends(self) -> list[str]:
        """List all registered backend names."""
        return [b.name for b in self._registry.list_all()]

    def get_gpu_status(self) -> dict[str, Any]:
        """Step 4.1: Get GPU status information."""
        return {
            "available": self._gpu_detector.is_gpu_available(),
            "memory_mb": self._gpu_detector.get_gpu_memory_mb(),
            "gpu_backends": [b.name for b in self._registry.get_gpu_backends()],
        }


# ==============================================================================
# Convenience Functions
# ==============================================================================


def select_backend(
    circuit: Circuit,
    strategy: str = "balanced",
    prefer_gpu: bool = True,
) -> tuple[str, str]:
    """
    Quick backend selection.

    Args:
        circuit: Circuit to simulate.
        strategy: One of 'performance', 'accuracy', 'memory', 'balanced',
                  'gpu_preferred', 'cpu_optimized'.
        prefer_gpu: Whether to prefer GPU backends when available.

    Returns:
        Tuple of (backend_name, explanation).
    """
    strategy_map = {
        "performance": SelectionStrategy.PERFORMANCE,
        "accuracy": SelectionStrategy.ACCURACY,
        "memory": SelectionStrategy.MEMORY,
        "balanced": SelectionStrategy.BALANCED,
        "gpu_preferred": SelectionStrategy.GPU_PREFERRED,
        "cpu_optimized": SelectionStrategy.CPU_OPTIMIZED,
    }

    selector = BackendSelector()

    if strategy == "priority":
        result = selector.select_by_priority(circuit, prefer_gpu)
    else:
        result = selector.select(
            circuit, strategy_map.get(strategy, SelectionStrategy.BALANCED)
        )

    return result.selected_backend, result.explanation


def analyze_circuit(circuit: Circuit) -> CircuitCharacteristics:
    """Analyze a circuit and return its characteristics."""
    return CircuitCharacteristics.from_circuit(circuit)


def is_gpu_available() -> bool:
    """Step 4.1: Check if GPU is available for quantum simulation."""
    return GPUDetector().is_gpu_available()


def get_recommended_backend(
    qubit_count: int,
    simulation_type: str = "state_vector",
    prefer_gpu: bool = True,
) -> str:
    """Step 4.1: Get recommended backend for given requirements."""
    registry = BackendRegistry()
    gpu_detector = GPUDetector()
    priority_selector = BackendPrioritySelector(registry, gpu_detector)

    # Create mock characteristics
    characteristics = CircuitCharacteristics(
        qubit_count=qubit_count,
        gate_count=qubit_count * 10,  # Estimate
        depth=qubit_count * 2,
        simulation_type=(
            SimulationType(simulation_type)
            if isinstance(simulation_type, str)
            else simulation_type
        ),
    )

    result = priority_selector.select_best_available(characteristics, prefer_gpu)
    return result or "numpy"


# Backwards compatibility alias
SelectionInput = CircuitCharacteristics
