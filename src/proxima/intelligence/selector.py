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

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
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


# =============================================================================
# Performance History Database (Feature - Backend Selector)
# =============================================================================


@dataclass
class PerformanceRecord:
    """Record of a single backend execution."""
    
    timestamp: float
    backend: str
    qubit_count: int
    gate_count: int
    depth: int
    execution_time_ms: float
    memory_peak_mb: float
    success: bool
    error: str | None = None
    simulation_type: str = "state_vector"
    gpu_used: bool = False


@dataclass
class BackendStatistics:
    """Aggregated statistics for a backend."""
    
    backend: str
    total_executions: int
    successful_executions: int
    success_rate: float
    avg_execution_time_ms: float
    median_execution_time_ms: float
    p95_execution_time_ms: float
    avg_memory_mb: float
    max_memory_mb: float
    avg_qubits: float
    max_qubits: int
    last_used: float
    reliability_score: float  # 0-1 based on success rate and consistency


class PerformanceHistoryDatabase:
    """Database for tracking backend performance history.
    
    Features:
    - SQLite-backed persistent storage
    - Performance trend analysis
    - Backend reliability scoring
    - Query by circuit characteristics
    - Automatic cleanup of old records
    """
    
    def __init__(self, db_path: str | None = None) -> None:
        """Initialize database.
        
        Args:
            db_path: Path to SQLite database file
        """
        import sqlite3
        
        self._db_path = db_path or str(Path.home() / ".proxima" / "performance_history.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._conn: sqlite3.Connection | None = None
        self._init_db()
    
    def _get_conn(self) -> "sqlite3.Connection":
        """Get database connection."""
        import sqlite3
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS performance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                backend TEXT NOT NULL,
                qubit_count INTEGER NOT NULL,
                gate_count INTEGER NOT NULL,
                depth INTEGER NOT NULL,
                execution_time_ms REAL NOT NULL,
                memory_peak_mb REAL NOT NULL,
                success INTEGER NOT NULL,
                error TEXT,
                simulation_type TEXT DEFAULT 'state_vector',
                gpu_used INTEGER DEFAULT 0
            );
            
            CREATE INDEX IF NOT EXISTS idx_backend ON performance_records(backend);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_records(timestamp);
            CREATE INDEX IF NOT EXISTS idx_qubits ON performance_records(qubit_count);
            
            CREATE TABLE IF NOT EXISTS backend_stats_cache (
                backend TEXT PRIMARY KEY,
                stats_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        conn.commit()
    
    def record_execution(
        self,
        backend: str,
        qubit_count: int,
        gate_count: int,
        depth: int,
        execution_time_ms: float,
        memory_peak_mb: float,
        success: bool,
        error: str | None = None,
        simulation_type: str = "state_vector",
        gpu_used: bool = False,
    ) -> PerformanceRecord:
        """Record a backend execution.
        
        Args:
            backend: Backend name
            qubit_count: Number of qubits
            gate_count: Number of gates
            depth: Circuit depth
            execution_time_ms: Execution time in milliseconds
            memory_peak_mb: Peak memory usage in MB
            success: Whether execution succeeded
            error: Error message if failed
            simulation_type: Type of simulation
            gpu_used: Whether GPU was used
            
        Returns:
            Created PerformanceRecord
        """
        import time as time_module
        
        timestamp = time_module.time()
        
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO performance_records 
            (timestamp, backend, qubit_count, gate_count, depth, execution_time_ms,
             memory_peak_mb, success, error, simulation_type, gpu_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, backend, qubit_count, gate_count, depth, execution_time_ms,
             memory_peak_mb, int(success), error, simulation_type, int(gpu_used))
        )
        conn.commit()
        
        # Invalidate stats cache
        conn.execute("DELETE FROM backend_stats_cache WHERE backend = ?", (backend,))
        conn.commit()
        
        return PerformanceRecord(
            timestamp=timestamp,
            backend=backend,
            qubit_count=qubit_count,
            gate_count=gate_count,
            depth=depth,
            execution_time_ms=execution_time_ms,
            memory_peak_mb=memory_peak_mb,
            success=success,
            error=error,
            simulation_type=simulation_type,
            gpu_used=gpu_used,
        )
    
    def get_backend_statistics(
        self,
        backend: str,
        days: int = 30,
    ) -> BackendStatistics | None:
        """Get aggregated statistics for a backend.
        
        Args:
            backend: Backend name
            days: Number of days to include
            
        Returns:
            BackendStatistics or None if no data
        """
        import time as time_module
        
        conn = self._get_conn()
        cutoff = time_module.time() - (days * 24 * 3600)
        
        # Get all records
        cursor = conn.execute(
            """
            SELECT * FROM performance_records
            WHERE backend = ? AND timestamp > ?
            ORDER BY timestamp DESC
            """,
            (backend, cutoff)
        )
        rows = cursor.fetchall()
        
        if not rows:
            return None
        
        # Calculate statistics
        times = [r["execution_time_ms"] for r in rows]
        memories = [r["memory_peak_mb"] for r in rows]
        qubits = [r["qubit_count"] for r in rows]
        successes = [r["success"] for r in rows]
        
        import statistics
        
        success_rate = sum(successes) / len(successes)
        
        # Calculate reliability score
        # Based on success rate and consistency (lower variance is better)
        try:
            variance = statistics.variance(times) if len(times) > 1 else 0
            consistency = 1 / (1 + variance / 1000)  # Normalize
        except Exception:
            consistency = 0.5
        
        reliability = 0.7 * success_rate + 0.3 * consistency
        
        return BackendStatistics(
            backend=backend,
            total_executions=len(rows),
            successful_executions=sum(successes),
            success_rate=success_rate,
            avg_execution_time_ms=statistics.mean(times),
            median_execution_time_ms=statistics.median(times),
            p95_execution_time_ms=sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0],
            avg_memory_mb=statistics.mean(memories),
            max_memory_mb=max(memories),
            avg_qubits=statistics.mean(qubits),
            max_qubits=max(qubits),
            last_used=max(r["timestamp"] for r in rows),
            reliability_score=reliability,
        )
    
    def get_history_score(self, backend: str, days: int = 30) -> float:
        """Get historical success score for backend selection.
        
        Args:
            backend: Backend name
            days: Number of days to consider
            
        Returns:
            Score between 0 and 1
        """
        stats = self.get_backend_statistics(backend, days)
        if not stats:
            return 0.8  # Default neutral score
        
        return stats.reliability_score
    
    def get_best_backend_for_circuit(
        self,
        qubit_count: int,
        gate_count: int,
        simulation_type: str = "state_vector",
    ) -> str | None:
        """Get historically best backend for similar circuits.
        
        Args:
            qubit_count: Number of qubits
            gate_count: Number of gates
            simulation_type: Simulation type
            
        Returns:
            Best backend name or None
        """
        import time as time_module
        
        conn = self._get_conn()
        cutoff = time_module.time() - (30 * 24 * 3600)
        
        # Find similar circuits (within 20% qubit/gate range)
        qubit_min = int(qubit_count * 0.8)
        qubit_max = int(qubit_count * 1.2)
        gate_min = int(gate_count * 0.8)
        gate_max = int(gate_count * 1.2)
        
        cursor = conn.execute(
            """
            SELECT backend, 
                   AVG(execution_time_ms) as avg_time,
                   COUNT(*) as count,
                   SUM(success) * 1.0 / COUNT(*) as success_rate
            FROM performance_records
            WHERE timestamp > ?
              AND qubit_count BETWEEN ? AND ?
              AND gate_count BETWEEN ? AND ?
              AND simulation_type = ?
              AND success = 1
            GROUP BY backend
            HAVING count >= 3
            ORDER BY success_rate DESC, avg_time ASC
            LIMIT 1
            """,
            (cutoff, qubit_min, qubit_max, gate_min, gate_max, simulation_type)
        )
        row = cursor.fetchone()
        
        return row["backend"] if row else None
    
    def estimate_execution_time(
        self,
        backend: str,
        qubit_count: int,
        gate_count: int,
    ) -> float | None:
        """Estimate execution time based on history.
        
        Uses linear regression on historical data.
        
        Args:
            backend: Backend name
            qubit_count: Number of qubits
            gate_count: Number of gates
            
        Returns:
            Estimated time in ms or None if insufficient data
        """
        import time as time_module
        
        conn = self._get_conn()
        cutoff = time_module.time() - (30 * 24 * 3600)
        
        cursor = conn.execute(
            """
            SELECT qubit_count, gate_count, execution_time_ms
            FROM performance_records
            WHERE backend = ? AND timestamp > ? AND success = 1
            ORDER BY timestamp DESC
            LIMIT 100
            """,
            (backend, cutoff)
        )
        rows = cursor.fetchall()
        
        if len(rows) < 10:
            return None
        
        # Simple estimation using average scaling
        # Time ≈ base + k1 * 2^qubits + k2 * gates
        import statistics
        
        # Calculate average time per complexity unit
        times = [r["execution_time_ms"] for r in rows]
        complexities = [2 ** r["qubit_count"] + r["gate_count"] for r in rows]
        
        time_per_complexity = statistics.mean(
            t / c for t, c in zip(times, complexities) if c > 0
        )
        
        target_complexity = 2 ** qubit_count + gate_count
        return time_per_complexity * target_complexity
    
    def cleanup_old_records(self, days: int = 90) -> int:
        """Remove records older than specified days.
        
        Args:
            days: Age threshold
            
        Returns:
            Number of records deleted
        """
        import time as time_module
        
        conn = self._get_conn()
        cutoff = time_module.time() - (days * 24 * 3600)
        
        cursor = conn.execute(
            "DELETE FROM performance_records WHERE timestamp < ?",
            (cutoff,)
        )
        deleted = cursor.rowcount
        conn.commit()
        
        # Clear cache
        conn.execute("DELETE FROM backend_stats_cache")
        conn.commit()
        
        return deleted
    
    def export_history(self, output_path: str, format: str = "json") -> None:
        """Export performance history.
        
        Args:
            output_path: Output file path
            format: 'json' or 'csv'
        """
        conn = self._get_conn()
        cursor = conn.execute("SELECT * FROM performance_records ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        if format == "json":
            data = [dict(row) for row in rows]
            Path(output_path).write_text(json.dumps(data, indent=2))
        elif format == "csv":
            headers = ["timestamp", "backend", "qubit_count", "gate_count", "depth",
                      "execution_time_ms", "memory_peak_mb", "success", "error",
                      "simulation_type", "gpu_used"]
            lines = [",".join(headers)]
            for row in rows:
                lines.append(",".join(str(row[h]) for h in headers))
            Path(output_path).write_text("\n".join(lines))


# =============================================================================
# Enhanced Memory Estimation (Feature - Backend Selector)
# =============================================================================


@dataclass
class MemoryEstimate:
    """Detailed memory estimate for a simulation."""
    
    state_vector_mb: float
    intermediate_mb: float
    overhead_mb: float
    total_mb: float
    peak_mb: float
    fits_in_ram: bool
    fits_in_gpu: bool
    recommended_precision: str  # 'single' or 'double'
    warnings: list[str]


class MemoryEstimator:
    """Estimates memory requirements for quantum circuit simulation.
    
    Features:
    - Per-backend memory models
    - Precision-aware estimation
    - GPU memory consideration
    - Safety margins
    - Recommendations
    """
    
    # Memory overhead factors per backend
    BACKEND_OVERHEAD: dict[str, float] = {
        "numpy": 1.2,      # Some overhead for Python
        "cupy": 1.1,       # Efficient GPU
        "qiskit": 1.5,     # Qiskit has more overhead
        "cirq": 1.4,
        "quest": 1.1,      # Efficient C++
        "cuquantum": 1.05, # Very efficient
        "qsim": 1.15,
        "pennylane": 1.3,
        "lret": 0.8,       # Rank-reduced is more efficient
    }
    
    # Intermediate memory factors (for gate application)
    INTERMEDIATE_FACTORS: dict[str, float] = {
        "numpy": 2.0,      # Needs copy for operations
        "cupy": 1.5,
        "qiskit": 2.5,
        "cirq": 2.0,
        "quest": 1.2,
        "cuquantum": 1.1,
        "qsim": 1.3,
        "pennylane": 2.0,
        "lret": 0.5,       # Much less due to rank reduction
    }
    
    def __init__(self, gpu_detector: GPUDetector | None = None) -> None:
        """Initialize estimator."""
        self._gpu_detector = gpu_detector or GPUDetector()
    
    def estimate(
        self,
        backend: str,
        qubit_count: int,
        gate_count: int,
        simulation_type: str = "state_vector",
        precision: str = "double",
    ) -> MemoryEstimate:
        """Estimate memory requirements.
        
        Args:
            backend: Backend name
            qubit_count: Number of qubits
            gate_count: Number of gates
            simulation_type: 'state_vector' or 'density_matrix'
            precision: 'single' or 'double'
            
        Returns:
            MemoryEstimate with all details
        """
        warnings: list[str] = []
        
        # Base state vector size
        bytes_per_amplitude = 16 if precision == "double" else 8
        
        if simulation_type == "density_matrix":
            # Density matrix: 2^n x 2^n complex matrix
            num_elements = (2 ** qubit_count) ** 2
        else:
            # State vector: 2^n complex amplitudes
            num_elements = 2 ** qubit_count
        
        state_vector_bytes = num_elements * bytes_per_amplitude
        state_vector_mb = state_vector_bytes / (1024 * 1024)
        
        # Backend-specific overhead
        overhead_factor = self.BACKEND_OVERHEAD.get(backend, 1.5)
        overhead_mb = state_vector_mb * (overhead_factor - 1)
        
        # Intermediate memory for gate operations
        intermediate_factor = self.INTERMEDIATE_FACTORS.get(backend, 2.0)
        intermediate_mb = state_vector_mb * (intermediate_factor - 1)
        
        # Total and peak
        total_mb = state_vector_mb + overhead_mb
        peak_mb = state_vector_mb + overhead_mb + intermediate_mb
        
        # Check system RAM
        try:
            available_ram = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            available_ram = 8192
        
        fits_in_ram = peak_mb < available_ram * 0.8
        
        # Check GPU memory
        fits_in_gpu = False
        if self._gpu_detector.is_gpu_available():
            gpu_mem = self._gpu_detector.get_gpu_memory_mb()
            fits_in_gpu = peak_mb < gpu_mem * 0.8
        
        # Generate warnings
        if peak_mb > available_ram * 0.5:
            warnings.append(f"High memory usage: {peak_mb:.0f} MB (50%+ of available RAM)")
        
        if qubit_count > 25 and precision == "double":
            warnings.append("Consider using single precision for large circuits")
        
        if simulation_type == "density_matrix" and qubit_count > 15:
            warnings.append("Density matrix simulation is memory-intensive for >15 qubits")
        
        # Recommend precision
        if qubit_count > 25:
            recommended_precision = "single"
        elif simulation_type == "density_matrix" and qubit_count > 12:
            recommended_precision = "single"
        else:
            recommended_precision = "double"
        
        return MemoryEstimate(
            state_vector_mb=state_vector_mb,
            intermediate_mb=intermediate_mb,
            overhead_mb=overhead_mb,
            total_mb=total_mb,
            peak_mb=peak_mb,
            fits_in_ram=fits_in_ram,
            fits_in_gpu=fits_in_gpu,
            recommended_precision=recommended_precision,
            warnings=warnings,
        )
    
    def estimate_all_backends(
        self,
        qubit_count: int,
        gate_count: int,
        simulation_type: str = "state_vector",
    ) -> dict[str, MemoryEstimate]:
        """Estimate memory for all backends.
        
        Args:
            qubit_count: Number of qubits
            gate_count: Number of gates
            simulation_type: Simulation type
            
        Returns:
            Dict mapping backend name to estimate
        """
        backends = list(self.BACKEND_OVERHEAD.keys())
        return {
            backend: self.estimate(backend, qubit_count, gate_count, simulation_type)
            for backend in backends
        }
    
    def get_best_backend_by_memory(
        self,
        qubit_count: int,
        simulation_type: str = "state_vector",
        require_gpu: bool = False,
    ) -> str | None:
        """Get backend with best memory efficiency.
        
        Args:
            qubit_count: Number of qubits
            simulation_type: Simulation type
            require_gpu: Whether GPU is required
            
        Returns:
            Best backend name or None
        """
        estimates = self.estimate_all_backends(qubit_count, 100, simulation_type)
        
        # Filter by requirements
        candidates = []
        for backend, estimate in estimates.items():
            if not estimate.fits_in_ram:
                continue
            if require_gpu and not estimate.fits_in_gpu:
                continue
            candidates.append((backend, estimate))
        
        if not candidates:
            return None
        
        # Sort by peak memory
        candidates.sort(key=lambda x: x[1].peak_mb)
        return candidates[0][0]


# =============================================================================
# Comprehensive Explanation Generation (Feature - Backend Selector)
# =============================================================================


@dataclass
class DetailedExplanation:
    """Comprehensive explanation of backend selection."""
    
    summary: str
    circuit_analysis: str
    strategy_rationale: str
    backend_comparison: str
    scoring_breakdown: str
    memory_analysis: str
    gpu_analysis: str
    history_analysis: str
    recommendations: list[str]
    warnings: list[str]
    alternatives_analysis: str
    confidence_explanation: str


class ComprehensiveExplainer:
    """Generates detailed explanations for backend selection.
    
    Features:
    - Multi-section explanations
    - Visual score breakdowns
    - Memory and GPU analysis
    - Historical context
    - Actionable recommendations
    """
    
    def __init__(
        self,
        history_db: PerformanceHistoryDatabase | None = None,
        memory_estimator: MemoryEstimator | None = None,
    ) -> None:
        """Initialize explainer."""
        self._history_db = history_db
        self._memory_estimator = memory_estimator or MemoryEstimator()
    
    def generate_detailed_explanation(
        self,
        result: SelectionResult,
        characteristics: CircuitCharacteristics,
        strategy: SelectionStrategy,
    ) -> DetailedExplanation:
        """Generate comprehensive explanation.
        
        Args:
            result: Selection result
            characteristics: Circuit characteristics
            strategy: Selection strategy used
            
        Returns:
            DetailedExplanation with all sections
        """
        # Summary
        summary = self._generate_summary(result, characteristics)
        
        # Circuit analysis
        circuit_analysis = self._generate_circuit_analysis(characteristics)
        
        # Strategy rationale
        strategy_rationale = self._generate_strategy_rationale(strategy, characteristics)
        
        # Backend comparison
        backend_comparison = self._generate_backend_comparison(result.scores)
        
        # Scoring breakdown
        scoring_breakdown = self._generate_scoring_breakdown(result.scores, result.selected_backend)
        
        # Memory analysis
        memory_analysis = self._generate_memory_analysis(result.selected_backend, characteristics)
        
        # GPU analysis
        gpu_analysis = self._generate_gpu_analysis(result, characteristics)
        
        # History analysis
        history_analysis = self._generate_history_analysis(result.selected_backend)
        
        # Recommendations
        recommendations = self._generate_recommendations(result, characteristics)
        
        # Alternatives analysis
        alternatives_analysis = self._generate_alternatives_analysis(result)
        
        # Confidence explanation
        confidence_explanation = self._generate_confidence_explanation(result)
        
        return DetailedExplanation(
            summary=summary,
            circuit_analysis=circuit_analysis,
            strategy_rationale=strategy_rationale,
            backend_comparison=backend_comparison,
            scoring_breakdown=scoring_breakdown,
            memory_analysis=memory_analysis,
            gpu_analysis=gpu_analysis,
            history_analysis=history_analysis,
            recommendations=recommendations,
            warnings=result.warnings,
            alternatives_analysis=alternatives_analysis,
            confidence_explanation=confidence_explanation,
        )
    
    def _generate_summary(
        self,
        result: SelectionResult,
        characteristics: CircuitCharacteristics,
    ) -> str:
        """Generate summary section."""
        return (
            f"Selected **{result.selected_backend}** backend with "
            f"{result.confidence:.0%} confidence for a {characteristics.qubit_count}-qubit, "
            f"{characteristics.gate_count}-gate circuit. "
            f"{'GPU acceleration is available.' if result.gpu_available else 'Running on CPU.'}"
        )
    
    def _generate_circuit_analysis(
        self,
        characteristics: CircuitCharacteristics,
    ) -> str:
        """Generate circuit analysis section."""
        lines = [
            "**Circuit Characteristics:**",
            f"• Qubits: {characteristics.qubit_count}",
            f"• Gates: {characteristics.gate_count}",
            f"• Depth: {characteristics.depth}",
            f"• Entanglement density: {characteristics.entanglement_density:.1%}",
            f"• Simulation type: {characteristics.simulation_type.value}",
        ]
        
        if characteristics.has_custom_gates:
            lines.append("• Contains custom gates (may limit backend options)")
        
        if characteristics.has_parameterized_gates:
            lines.append("• Contains parameterized gates")
        
        if characteristics.has_mid_circuit_measurement:
            lines.append("• Contains mid-circuit measurements (limits some backends)")
        
        if characteristics.has_noise_model:
            lines.append("• Uses noise model")
        
        lines.append(f"• Estimated base memory: {characteristics.estimated_memory_mb:.1f} MB")
        
        return "\n".join(lines)
    
    def _generate_strategy_rationale(
        self,
        strategy: SelectionStrategy,
        characteristics: CircuitCharacteristics,
    ) -> str:
        """Generate strategy rationale section."""
        rationales = {
            SelectionStrategy.PERFORMANCE: (
                "**Performance Strategy**: Prioritizing execution speed. "
                "Best for iterative development, quick experiments, or when time is critical."
            ),
            SelectionStrategy.ACCURACY: (
                "**Accuracy Strategy**: Prioritizing simulation fidelity. "
                "Best for final validation, publication results, or error-sensitive work."
            ),
            SelectionStrategy.MEMORY: (
                "**Memory Strategy**: Prioritizing memory efficiency. "
                "Best for large circuits or systems with limited RAM."
            ),
            SelectionStrategy.BALANCED: (
                "**Balanced Strategy**: Weighing all factors equally. "
                "Good general-purpose selection for most use cases."
            ),
            SelectionStrategy.GPU_PREFERRED: (
                "**GPU-Preferred Strategy**: Prioritizing GPU acceleration. "
                "Best for large circuits where GPU parallelism provides speedup."
            ),
            SelectionStrategy.CPU_OPTIMIZED: (
                "**CPU-Optimized Strategy**: Prioritizing CPU-optimized backends. "
                "Best when GPU is unavailable or for smaller circuits."
            ),
            SelectionStrategy.LLM_ASSISTED: (
                "**LLM-Assisted Strategy**: Using AI to analyze circuit and recommend backend. "
                "Best for complex decisions with many competing factors."
            ),
        }
        
        base = rationales.get(strategy, "**Custom Strategy**")
        
        # Add context
        if characteristics.qubit_count > 25:
            base += f" Note: Large circuit ({characteristics.qubit_count} qubits) benefits from GPU or optimized backends."
        
        return base
    
    def _generate_backend_comparison(
        self,
        scores: list[SelectionScore],
    ) -> str:
        """Generate backend comparison table."""
        if not scores:
            return "No backends were scored."
        
        lines = ["**Backend Comparison:**", ""]
        lines.append("| Backend | Total | Feature | Perf | Memory | GPU | History |")
        lines.append("|---------|-------|---------|------|--------|-----|---------|")
        
        for score in sorted(scores, key=lambda s: -s.total_score)[:6]:
            lines.append(
                f"| {score.backend_name} | "
                f"{score.total_score:.2f} | "
                f"{score.feature_score:.2f} | "
                f"{score.performance_score:.2f} | "
                f"{score.memory_score:.2f} | "
                f"{score.gpu_score:.2f} | "
                f"{score.history_score:.2f} |"
            )
        
        return "\n".join(lines)
    
    def _generate_scoring_breakdown(
        self,
        scores: list[SelectionScore],
        selected: str,
    ) -> str:
        """Generate detailed scoring breakdown for selected backend."""
        score = next((s for s in scores if s.backend_name == selected), None)
        if not score:
            return "No scoring data available."
        
        lines = [
            f"**Scoring Breakdown for {selected}:**",
            "",
            f"• **Total Score**: {score.total_score:.3f}",
            f"• **Feature Score**: {score.feature_score:.3f} - Measures capability match",
            f"• **Performance Score**: {score.performance_score:.3f} - Expected execution speed",
            f"• **Memory Score**: {score.memory_score:.3f} - Memory efficiency",
            f"• **GPU Score**: {score.gpu_score:.3f} - GPU utilization potential",
            f"• **History Score**: {score.history_score:.3f} - Past reliability",
            f"• **Compatibility Score**: {score.compatibility_score:.3f} - Gate/feature support",
        ]
        
        if score.details:
            lines.append("")
            lines.append("**Additional Details:**")
            for key, value in score.details.items():
                lines.append(f"• {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines)
    
    def _generate_memory_analysis(
        self,
        backend: str,
        characteristics: CircuitCharacteristics,
    ) -> str:
        """Generate memory analysis section."""
        estimate = self._memory_estimator.estimate(
            backend,
            characteristics.qubit_count,
            characteristics.gate_count,
            characteristics.simulation_type.value,
        )
        
        lines = [
            "**Memory Analysis:**",
            "",
            f"• State vector: {estimate.state_vector_mb:.1f} MB",
            f"• Intermediate buffers: {estimate.intermediate_mb:.1f} MB",
            f"• Backend overhead: {estimate.overhead_mb:.1f} MB",
            f"• **Total required**: {estimate.total_mb:.1f} MB",
            f"• **Peak usage**: {estimate.peak_mb:.1f} MB",
            "",
            f"• Fits in RAM: {'✓' if estimate.fits_in_ram else '✗'}",
            f"• Fits in GPU: {'✓' if estimate.fits_in_gpu else '✗ (or N/A)'}",
            f"• Recommended precision: {estimate.recommended_precision}",
        ]
        
        if estimate.warnings:
            lines.append("")
            lines.append("**Memory Warnings:**")
            for warning in estimate.warnings:
                lines.append(f"• ⚠️ {warning}")
        
        return "\n".join(lines)
    
    def _generate_gpu_analysis(
        self,
        result: SelectionResult,
        characteristics: CircuitCharacteristics,
    ) -> str:
        """Generate GPU analysis section."""
        lines = ["**GPU Analysis:**", ""]
        
        if result.gpu_available:
            lines.append("• GPU: ✓ Available")
            
            selected_score = next(
                (s for s in result.scores if s.backend_name == result.selected_backend),
                None
            )
            
            if selected_score:
                if selected_score.details.get("gpu_available"):
                    lines.append(f"• Selected backend ({result.selected_backend}) supports GPU")
                else:
                    lines.append(f"• Selected backend ({result.selected_backend}) is CPU-only")
            
            gpu_backends = [s.backend_name for s in result.scores if s.details.get("gpu_available")]
            if gpu_backends:
                lines.append(f"• Available GPU backends: {', '.join(gpu_backends)}")
            
            # GPU benefit analysis
            if characteristics.qubit_count > 20:
                lines.append("• GPU recommended for this circuit size (>20 qubits)")
            else:
                lines.append("• GPU may not provide significant speedup for small circuits")
        else:
            lines.append("• GPU: ✗ Not available")
            lines.append("• Consider using CPU-optimized backends (qsim, quest)")
            
            if characteristics.qubit_count > 25:
                lines.append("• ⚠️ Large circuit would benefit from GPU acceleration")
        
        return "\n".join(lines)
    
    def _generate_history_analysis(self, backend: str) -> str:
        """Generate history analysis section."""
        if not self._history_db:
            return "**History Analysis:** No historical data available."
        
        stats = self._history_db.get_backend_statistics(backend)
        
        if not stats:
            return f"**History Analysis:** No historical data for {backend}."
        
        lines = [
            f"**Historical Performance of {backend}:**",
            "",
            f"• Total executions: {stats.total_executions}",
            f"• Success rate: {stats.success_rate:.1%}",
            f"• Reliability score: {stats.reliability_score:.2f}",
            "",
            f"• Avg execution time: {stats.avg_execution_time_ms:.1f} ms",
            f"• Median execution time: {stats.median_execution_time_ms:.1f} ms",
            f"• 95th percentile: {stats.p95_execution_time_ms:.1f} ms",
            "",
            f"• Avg memory usage: {stats.avg_memory_mb:.1f} MB",
            f"• Max memory usage: {stats.max_memory_mb:.1f} MB",
            f"• Avg circuit size: {stats.avg_qubits:.1f} qubits",
        ]
        
        return "\n".join(lines)
    
    def _generate_recommendations(
        self,
        result: SelectionResult,
        characteristics: CircuitCharacteristics,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Performance recommendations
        if characteristics.qubit_count > 20 and not result.gpu_available:
            recommendations.append(
                "Consider enabling GPU acceleration for faster simulation of large circuits"
            )
        
        if characteristics.qubit_count > 25:
            recommendations.append(
                "For circuits >25 qubits, consider using cuQuantum or qsim for best performance"
            )
        
        # Memory recommendations
        if characteristics.estimated_memory_mb > 1000:
            recommendations.append(
                "High memory requirement - consider single precision or chunked simulation"
            )
        
        # Accuracy recommendations
        if characteristics.has_noise_model:
            recommendations.append(
                "For noisy simulations, verify backend noise model matches your requirements"
            )
        
        # Backend-specific recommendations
        if result.selected_backend == "numpy" and characteristics.qubit_count > 15:
            recommendations.append(
                "NumPy backend selected - consider installing qsim or qiskit for better performance"
            )
        
        if result.confidence < 0.7:
            recommendations.append(
                "Low confidence selection - consider trying alternative backends listed below"
            )
        
        return recommendations
    
    def _generate_alternatives_analysis(self, result: SelectionResult) -> str:
        """Generate alternatives analysis section."""
        if not result.alternatives:
            return "**Alternatives:** No competitive alternatives identified."
        
        lines = ["**Alternative Backends:**", ""]
        
        for alt in result.alternatives[:3]:
            alt_score = next((s for s in result.scores if s.backend_name == alt), None)
            if alt_score:
                lines.append(
                    f"• **{alt}** (score: {alt_score.total_score:.2f}): "
                    f"Perf={alt_score.performance_score:.2f}, "
                    f"Memory={alt_score.memory_score:.2f}, "
                    f"GPU={alt_score.gpu_score:.2f}"
                )
            else:
                lines.append(f"• **{alt}**")
        
        return "\n".join(lines)
    
    def _generate_confidence_explanation(self, result: SelectionResult) -> str:
        """Generate confidence explanation section."""
        confidence = result.confidence
        
        if confidence >= 0.9:
            assessment = "**Very High Confidence**: Clear best choice with strong scores across all metrics."
        elif confidence >= 0.75:
            assessment = "**High Confidence**: Good match with minor trade-offs."
        elif confidence >= 0.6:
            assessment = "**Moderate Confidence**: Reasonable choice but alternatives may be worth considering."
        elif confidence >= 0.4:
            assessment = "**Low Confidence**: No ideal match found. Consider alternatives or different strategy."
        else:
            assessment = "**Very Low Confidence**: Poor match. Review circuit requirements and available backends."
        
        lines = [
            f"**Confidence Assessment ({confidence:.0%}):**",
            "",
            assessment,
        ]
        
        if result.warnings:
            lines.append("")
            lines.append("Factors affecting confidence:")
            for warning in result.warnings[:3]:
                lines.append(f"• {warning}")
        
        return "\n".join(lines)
    
    def format_as_markdown(self, explanation: DetailedExplanation) -> str:
        """Format detailed explanation as Markdown document.
        
        Args:
            explanation: DetailedExplanation to format
            
        Returns:
            Formatted Markdown string
        """
        sections = [
            "# Backend Selection Report",
            "",
            "## Summary",
            explanation.summary,
            "",
            "## Circuit Analysis",
            explanation.circuit_analysis,
            "",
            "## Strategy",
            explanation.strategy_rationale,
            "",
            "## Backend Comparison",
            explanation.backend_comparison,
            "",
            "## Selected Backend Details",
            explanation.scoring_breakdown,
            "",
            "## Memory Analysis",
            explanation.memory_analysis,
            "",
            "## GPU Analysis",
            explanation.gpu_analysis,
            "",
            "## Historical Performance",
            explanation.history_analysis,
            "",
            "## Confidence",
            explanation.confidence_explanation,
            "",
            "## Alternatives",
            explanation.alternatives_analysis,
        ]
        
        if explanation.recommendations:
            sections.extend([
                "",
                "## Recommendations",
                "",
            ])
            for rec in explanation.recommendations:
                sections.append(f"- {rec}")
        
        if explanation.warnings:
            sections.extend([
                "",
                "## Warnings",
                "",
            ])
            for warning in explanation.warnings:
                sections.append(f"- ⚠️ {warning}")
        
        return "\n".join(sections)
    
    def format_as_text(self, explanation: DetailedExplanation) -> str:
        """Format detailed explanation as plain text.
        
        Args:
            explanation: DetailedExplanation to format
            
        Returns:
            Formatted text string
        """
        # Simple text format without Markdown
        text = self.format_as_markdown(explanation)
        
        # Remove Markdown formatting
        import re
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Remove headers
        text = re.sub(r'\|[^\n]+\|', '', text)  # Remove tables
        
        return text


# =============================================================================
# Integration: Enhanced Backend Selector
# =============================================================================


class EnhancedBackendSelector(BackendSelector):
    """Backend selector with all enhanced features.
    
    Integrates:
    - Performance history database
    - Advanced memory estimation
    - Comprehensive explanations
    - GPU-aware selection for all backends
    """
    
    def __init__(
        self,
        history_db: PerformanceHistoryDatabase | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize enhanced selector."""
        super().__init__(**kwargs)
        
        self._history_db = history_db or PerformanceHistoryDatabase()
        self._memory_estimator = MemoryEstimator(self._gpu_detector)
        self._comprehensive_explainer = ComprehensiveExplainer(
            self._history_db,
            self._memory_estimator,
        )
        
        # Update scorer to use history database
        self._scorer = SelectionScorer(
            history_provider=self._history_db.get_history_score,
            gpu_detector=self._gpu_detector,
        )
    
    def select_with_detailed_explanation(
        self,
        circuit: "Circuit",
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
    ) -> tuple[SelectionResult, DetailedExplanation]:
        """Select backend with comprehensive explanation.
        
        Args:
            circuit: Circuit to simulate
            strategy: Selection strategy
            
        Returns:
            Tuple of (SelectionResult, DetailedExplanation)
        """
        result = self.select(circuit, strategy)
        characteristics = CircuitCharacteristics.from_circuit(circuit)
        
        explanation = self._comprehensive_explainer.generate_detailed_explanation(
            result, characteristics, strategy
        )
        
        return result, explanation
    
    def record_execution_result(
        self,
        backend: str,
        circuit: "Circuit",
        execution_time_ms: float,
        memory_peak_mb: float,
        success: bool,
        error: str | None = None,
        gpu_used: bool = False,
    ) -> None:
        """Record execution result for history.
        
        Args:
            backend: Backend used
            circuit: Circuit executed
            execution_time_ms: Execution time
            memory_peak_mb: Peak memory
            success: Whether successful
            error: Error message if failed
            gpu_used: Whether GPU was used
        """
        characteristics = CircuitCharacteristics.from_circuit(circuit)
        
        self._history_db.record_execution(
            backend=backend,
            qubit_count=characteristics.qubit_count,
            gate_count=characteristics.gate_count,
            depth=characteristics.depth,
            execution_time_ms=execution_time_ms,
            memory_peak_mb=memory_peak_mb,
            success=success,
            error=error,
            simulation_type=characteristics.simulation_type.value,
            gpu_used=gpu_used,
        )
    
    def get_memory_estimate(
        self,
        circuit: "Circuit",
        backend: str | None = None,
    ) -> MemoryEstimate | dict[str, MemoryEstimate]:
        """Get memory estimate for circuit.
        
        Args:
            circuit: Circuit to estimate
            backend: Specific backend (or None for all)
            
        Returns:
            MemoryEstimate or dict of estimates
        """
        characteristics = CircuitCharacteristics.from_circuit(circuit)
        
        if backend:
            return self._memory_estimator.estimate(
                backend,
                characteristics.qubit_count,
                characteristics.gate_count,
                characteristics.simulation_type.value,
            )
        
        return self._memory_estimator.estimate_all_backends(
            characteristics.qubit_count,
            characteristics.gate_count,
            characteristics.simulation_type.value,
        )
    
    def export_history_report(
        self,
        output_path: str,
        format: str = "json",
    ) -> None:
        """Export performance history report.
        
        Args:
            output_path: Output file path
            format: 'json' or 'csv'
        """
        self._history_db.export_history(output_path, format)
    
    def get_backend_statistics(self, backend: str) -> BackendStatistics | None:
        """Get statistics for a backend.
        
        Args:
            backend: Backend name
            
        Returns:
            BackendStatistics or None
        """
        return self._history_db.get_backend_statistics(backend)
