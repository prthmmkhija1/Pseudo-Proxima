"""Abstract base adapter and shared models for quantum backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from typing import Any


class SimulatorType(str, Enum):
    STATE_VECTOR = "state_vector"
    DENSITY_MATRIX = "density_matrix"
    TENSOR_NETWORK = "tensor_network"
    CUSTOM = "custom"


class ResultType(str, Enum):
    COUNTS = "counts"
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"


class BackendCapability(Flag):
    """Capabilities that a backend may support."""
    
    NONE = 0
    STATE_VECTOR = auto()
    DENSITY_MATRIX = auto()
    TENSOR_NETWORK = auto()
    NOISE_MODEL = auto()
    GPU_ACCELERATION = auto()
    BATCH_EXECUTION = auto()
    PARAMETER_BINDING = auto()
    CUSTOM_GATES = auto()


@dataclass
class Capabilities:
    simulator_types: list[SimulatorType]
    max_qubits: int
    supports_noise: bool = False
    supports_gpu: bool = False
    supports_batching: bool = False
    custom_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    valid: bool
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceEstimate:
    memory_mb: float | None = None
    time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    backend: str
    simulator_type: SimulatorType
    execution_time_ms: float
    qubit_count: int
    shot_count: int | None
    result_type: ResultType
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_result: Any = None


class BaseBackendAdapter(ABC):
    """Contract for all backend adapters."""

    supports_benchmarking: bool = True

    @abstractmethod
    def get_name(self) -> str:
        """Return backend identifier."""

    @abstractmethod
    def get_version(self) -> str:
        """Return backend version string."""

    @abstractmethod
    def get_capabilities(self) -> Capabilities:
        """Return supported capabilities."""

    @abstractmethod
    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate circuit compatibility with the backend."""

    @abstractmethod
    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate resources for execution."""

    @abstractmethod
    def execute(
        self, circuit: Any, options: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """Execute a circuit and return results."""

    @abstractmethod
    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        """Return whether the simulator type is supported."""

    def is_available(self) -> bool:
        """Return whether the backend is available on this system."""

        return True

    # ------------------------------------------------------------------
    # Benchmarking lifecycle hooks
    # ------------------------------------------------------------------
    def prepare_for_benchmark(self, circuit: Any | None = None, shots: int | None = None) -> None:
        """Optional pre-execution setup for benchmarking.

        Default implementation is a no-op; adapters can override to reset
        caches, warm up simulators, or pre-allocate resources.
        """

    def cleanup_after_benchmark(self) -> None:
        """Optional post-execution cleanup for benchmarking.

        Default implementation is a no-op; adapters can override to release
        resources or clear internal state between benchmark runs.
        """
