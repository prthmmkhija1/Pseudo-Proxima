"""
Backend Selector with LLM-Enhanced Selection.

Intelligent backend auto-selection based on circuit characteristics,
backend capabilities, and optional LLM-assisted recommendations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from proxima.core.circuit import Circuit

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class SelectionStrategy(Enum):
    """Strategy for backend selection."""

    PERFORMANCE = auto()  # Prioritize speed
    ACCURACY = auto()  # Prioritize simulation fidelity
    MEMORY = auto()  # Prioritize low memory usage
    BALANCED = auto()  # Balance all factors
    LLM_ASSISTED = auto()  # Use LLM for complex decisions


class BackendType(Enum):
    """Type of simulation backend."""

    NUMPY = "numpy"
    CUPY = "cupy"
    QISKIT = "qiskit"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    CUSTOM = "custom"


# =============================================================================
# Data Classes
# =============================================================================


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
    estimated_memory_mb: float = 0.0

    @classmethod
    def from_circuit(cls, circuit: Circuit) -> CircuitCharacteristics:
        """Analyze a circuit and extract its characteristics."""
        gate_types = set()
        entangling_gates = 0
        parameterized = False
        custom = False
        measurements = 0

        for gate in circuit.gates:
            gate_types.add(gate.name)

            # Check for entangling gates (multi-qubit)
            if len(gate.qubits) > 1:
                entangling_gates += 1

            # Check for measurements
            if gate.name.lower() in ("measure", "m", "measurement"):
                measurements += 1

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

    @property
    def backend(self) -> str:
        """Alias for selected_backend for backwards compatibility."""
        return self.selected_backend


# =============================================================================
# Backend Registry
# =============================================================================


class BackendRegistry:
    """Registry of available backends and their capabilities."""

    def __init__(self) -> None:
        self._backends: dict[str, BackendCapabilities] = {}
        self._register_default_backends()

    def _register_default_backends(self) -> None:
        """Register the default simulation backends."""
        # NumPy backend - always available, good for small circuits
        self.register(
            BackendCapabilities(
                name="numpy",
                backend_type=BackendType.NUMPY,
                max_qubits=25,
                supports_gpu=False,
                supports_noise=True,
                performance_score=0.6,
                memory_efficiency=0.7,
                accuracy_score=1.0,
            )
        )

        # CuPy backend - GPU accelerated
        self.register(
            BackendCapabilities(
                name="cupy",
                backend_type=BackendType.CUPY,
                max_qubits=30,
                supports_gpu=True,
                supports_noise=True,
                performance_score=0.95,
                memory_efficiency=0.5,  # GPU memory limited
                accuracy_score=0.99,  # Floating point on GPU
            )
        )

        # Qiskit backend - feature rich
        self.register(
            BackendCapabilities(
                name="qiskit",
                backend_type=BackendType.QISKIT,
                max_qubits=32,
                supports_gpu=False,
                supports_noise=True,
                supports_distributed=True,
                performance_score=0.7,
                memory_efficiency=0.6,
                accuracy_score=0.98,
            )
        )

        # Cirq backend - Google's framework
        self.register(
            BackendCapabilities(
                name="cirq",
                backend_type=BackendType.CIRQ,
                max_qubits=30,
                supports_gpu=False,
                supports_noise=True,
                performance_score=0.75,
                memory_efficiency=0.65,
                accuracy_score=0.98,
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
        """List backends compatible with given circuit characteristics.

        Args:
            characteristics: Circuit characteristics to check against.
            check_runtime: If True, verify backends are actually available at runtime.
        """
        compatible = []

        for backend in self._backends.values():
            # Check qubit limit
            if characteristics.qubit_count > backend.max_qubits:
                continue

            # Check custom gate support
            if characteristics.has_custom_gates and not backend.supports_custom_gates:
                continue

            # Runtime availability check
            if check_runtime and not self._is_backend_available(backend.name):
                continue

            compatible.append(backend)

        return compatible

    def _is_backend_available(self, backend_name: str) -> bool:
        """Check if a backend is actually available at runtime.

        Performs import checks and basic validation to ensure the backend
        can actually be used.
        """
        try:
            if backend_name == "numpy":
                import numpy  # noqa: F401
                return True
            elif backend_name == "cupy":
                try:
                    import cupy  # noqa: F401
                    # Also check if GPU is available
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
            else:
                # Unknown backends are assumed available
                return True
        except Exception:
            return False

    def get_available_backends(self) -> list[BackendCapabilities]:
        """Get list of backends that are actually available at runtime."""
        return [b for b in self._backends.values() if self._is_backend_available(b.name)]


# =============================================================================
# Selection Scoring
# =============================================================================


class SelectionScorer:
    """Scores backends for selection based on various factors."""

    # Scoring weights by strategy
    STRATEGY_WEIGHTS = {
        SelectionStrategy.PERFORMANCE: {
            "feature": 0.2,
            "performance": 0.5,
            "memory": 0.1,
            "history": 0.1,
            "compatibility": 0.1,
        },
        SelectionStrategy.ACCURACY: {
            "feature": 0.3,
            "performance": 0.1,
            "memory": 0.1,
            "history": 0.2,
            "compatibility": 0.3,
        },
        SelectionStrategy.MEMORY: {
            "feature": 0.2,
            "performance": 0.1,
            "memory": 0.5,
            "history": 0.1,
            "compatibility": 0.1,
        },
        SelectionStrategy.BALANCED: {
            "feature": 0.25,
            "performance": 0.25,
            "memory": 0.2,
            "history": 0.15,
            "compatibility": 0.15,
        },
        SelectionStrategy.LLM_ASSISTED: {
            "feature": 0.2,
            "performance": 0.2,
            "memory": 0.2,
            "history": 0.2,
            "compatibility": 0.2,
        },
    }

    def __init__(
        self,
        history_provider: Callable[[str], float] | None = None,
    ) -> None:
        """
        Initialize scorer.

        Args:
            history_provider: Optional callback that returns historical
                success rate (0-1) for a given backend name.
        """
        self._history_provider = history_provider

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

        # Calculate weighted total
        total = (
            weights["feature"] * feature_score
            + weights["performance"] * performance_score
            + weights["memory"] * memory_score
            + weights["history"] * history_score
            + weights["compatibility"] * compatibility_score
        )

        return SelectionScore(
            backend_name=backend.name,
            total_score=total,
            feature_score=feature_score,
            performance_score=performance_score,
            memory_score=memory_score,
            history_score=history_score,
            compatibility_score=compatibility_score,
            details={
                "qubit_headroom": backend.max_qubits - characteristics.qubit_count,
                "gpu_available": backend.supports_gpu,
                "noise_support": backend.supports_noise,
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

        # Bonus for GPU support with large circuits
        if characteristics.qubit_count > 15 and backend.supports_gpu:
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

        # Large circuits benefit from GPU
        if characteristics.qubit_count > 18 and backend.supports_gpu:
            base_score += 0.15

        # Deep circuits may slow down some backends
        if characteristics.depth > 100:
            if not backend.supports_gpu:
                base_score -= 0.1

        return max(0.0, min(1.0, base_score))

    def _score_memory(
        self,
        backend: BackendCapabilities,
        characteristics: CircuitCharacteristics,
    ) -> float:
        """Score based on memory efficiency."""
        base_score = backend.memory_efficiency

        # Check if estimated memory is manageable
        if characteristics.estimated_memory_mb > 1000:  # > 1GB
            if not backend.supports_distributed:
                base_score -= 0.2

        # Very large circuits need efficient backends
        if characteristics.qubit_count > 25:
            if backend.memory_efficiency < 0.6:
                base_score -= 0.2

        return max(0.0, min(1.0, base_score))

    def _score_history(self, backend_name: str) -> float:
        """Score based on historical success rate."""
        if self._history_provider is None:
            return 0.5  # Neutral if no history

        try:
            return self._history_provider(backend_name)
        except Exception:
            return 0.5

    def _score_compatibility(
        self,
        backend: BackendCapabilities,
        characteristics: CircuitCharacteristics,
    ) -> float:
        """Score based on gate/feature compatibility."""
        score = backend.accuracy_score

        # If backend has specific supported gates, check overlap
        if backend.supported_gate_types:
            if characteristics.gate_types:
                overlap = len(characteristics.gate_types & backend.supported_gate_types)
                coverage = overlap / len(characteristics.gate_types)
                score *= coverage

        return score


# =============================================================================
# Explanation Generator
# =============================================================================


class ExplanationGenerator:
    """Generates human-readable explanations for selection decisions."""

    def generate(
        self,
        result: SelectionResult,
        characteristics: CircuitCharacteristics,
        strategy: SelectionStrategy,
    ) -> str:
        """Generate a detailed explanation."""
        lines = []

        # Header
        lines.append(f"Selected backend: **{result.selected_backend}**")
        lines.append(f"Confidence: {result.confidence:.1%}")
        lines.append("")

        # Circuit summary
        lines.append("**Circuit Analysis:**")
        lines.append(f"- Qubits: {characteristics.qubit_count}")
        lines.append(f"- Gates: {characteristics.gate_count}")
        lines.append(f"- Depth: {characteristics.depth}")
        lines.append(f"- Estimated memory: {characteristics.estimated_memory_mb:.1f} MB")
        if characteristics.has_custom_gates:
            lines.append("- Uses custom gates")
        if characteristics.has_parameterized_gates:
            lines.append("- Uses parameterized gates")
        lines.append("")

        # Strategy info
        lines.append(f"**Strategy:** {strategy.name.replace('_', ' ').title()}")
        lines.append("")

        # Top scores
        lines.append("**Backend Scores:**")
        for score in sorted(result.scores, key=lambda s: s.total_score, reverse=True)[:3]:
            lines.append(
                f"- {score.backend_name}: {score.total_score:.2f} "
                f"(perf={score.performance_score:.2f}, mem={score.memory_score:.2f})"
            )
        lines.append("")

        # Reasoning
        if result.reasoning_steps:
            lines.append("**Reasoning:**")
            for step in result.reasoning_steps:
                lines.append(f"- {step}")
            lines.append("")

        # Warnings
        if result.warnings:
            lines.append("**⚠️ Warnings:**")
            for warning in result.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        # Alternatives
        if result.alternatives:
            lines.append("**Alternatives:**")
            lines.append(f"- {', '.join(result.alternatives)}")

        return "\n".join(lines)

    def generate_reasoning_steps(
        self,
        characteristics: CircuitCharacteristics,
        scores: list[SelectionScore],
        selected: str,
        strategy: SelectionStrategy,
    ) -> list[str]:
        """Generate step-by-step reasoning."""
        steps = []

        # Step 1: Analyze circuit
        steps.append(
            f"Analyzed circuit: {characteristics.qubit_count} qubits, "
            f"{characteristics.gate_count} gates, depth {characteristics.depth}"
        )

        # Step 2: Strategy consideration
        if strategy == SelectionStrategy.PERFORMANCE:
            steps.append("Prioritizing performance as requested")
        elif strategy == SelectionStrategy.MEMORY:
            steps.append("Prioritizing memory efficiency as requested")
        elif strategy == SelectionStrategy.ACCURACY:
            steps.append("Prioritizing accuracy as requested")
        else:
            steps.append("Using balanced scoring across all factors")

        # Step 3: Memory consideration
        if characteristics.estimated_memory_mb > 100:
            steps.append(
                f"Large state vector ({characteristics.estimated_memory_mb:.0f} MB) "
                "favors GPU-enabled backends"
            )

        # Step 4: Winner selection
        winner_score = next(s for s in scores if s.backend_name == selected)
        runner_up = sorted(
            [s for s in scores if s.backend_name != selected],
            key=lambda s: s.total_score,
            reverse=True,
        )

        if runner_up:
            margin = winner_score.total_score - runner_up[0].total_score
            steps.append(
                f"Selected '{selected}' with score {winner_score.total_score:.2f}, "
                f"margin of {margin:.2f} over '{runner_up[0].backend_name}'"
            )
        else:
            steps.append(f"Selected '{selected}' as the only compatible backend")

        return steps


# =============================================================================
# Main Selector
# =============================================================================


class BackendSelector:
    """
    Intelligent backend selector with scoring and explanations.

    Example usage:
        selector = BackendSelector()
        result = selector.select(circuit)
        print(result.explanation)
    """

    def __init__(
        self,
        registry: BackendRegistry | None = None,
        history_provider: Callable[[str], float] | None = None,
        llm_recommender: Callable[[str], str] | None = None,
    ) -> None:
        """
        Initialize the selector.

        Args:
            registry: Backend registry, or use default.
            history_provider: Callback returning historical success rate.
            llm_recommender: Optional LLM callback for complex decisions.
        """
        self._registry = registry or BackendRegistry()
        self._scorer = SelectionScorer(history_provider)
        self._explainer = ExplanationGenerator()
        self._llm_recommender = llm_recommender

    def select(
        self,
        circuit: Circuit,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        prefer_local: bool = True,
    ) -> SelectionResult:
        """
        Select the best backend for a circuit.

        Args:
            circuit: The circuit to simulate.
            strategy: Selection strategy to use.
            prefer_local: Prefer backends that run locally (no cloud).

        Returns:
            SelectionResult with the selection and explanation.
        """
        # Analyze circuit
        characteristics = CircuitCharacteristics.from_circuit(circuit)

        # Get compatible backends
        compatible = self._registry.list_compatible(characteristics)

        if not compatible:
            return SelectionResult(
                selected_backend="numpy",  # Fallback
                confidence=0.0,
                scores=[],
                explanation="No compatible backends found, using numpy as fallback",
                reasoning_steps=["No backends support this circuit configuration"],
                warnings=["Using fallback backend - may not work correctly"],
            )

        # Score all compatible backends
        scores = [self._scorer.score(backend, characteristics, strategy) for backend in compatible]

        # Sort by total score
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Select winner
        selected = scores[0].backend_name
        confidence = scores[0].total_score

        # Determine alternatives
        alternatives = [s.backend_name for s in scores[1:4] if s.total_score > 0.5]

        # Generate warnings
        warnings = self._generate_warnings(characteristics, scores[0])

        # Generate reasoning
        reasoning = self._explainer.generate_reasoning_steps(
            characteristics, scores, selected, strategy
        )

        # Optional LLM recommendation for complex cases
        llm_recommendation = None
        if strategy == SelectionStrategy.LLM_ASSISTED and self._llm_recommender is not None:
            try:
                prompt = self._build_llm_prompt(characteristics, scores)
                llm_recommendation = self._llm_recommender(prompt)
            except Exception as e:
                logger.warning(f"LLM recommendation failed: {e}")

        # Build result
        result = SelectionResult(
            selected_backend=selected,
            confidence=confidence,
            scores=scores,
            explanation="",  # Will be set below
            reasoning_steps=reasoning,
            warnings=warnings,
            alternatives=alternatives,
            llm_recommendation=llm_recommendation,
        )

        # Generate explanation
        result.explanation = self._explainer.generate(result, characteristics, strategy)

        return result

    def select_from_characteristics(
        self,
        characteristics: CircuitCharacteristics,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
    ) -> SelectionResult:
        """
        Select backend from pre-analyzed characteristics.

        Useful when you've already analyzed the circuit or want to
        test with specific characteristics.
        """
        compatible = self._registry.list_compatible(characteristics)

        if not compatible:
            return SelectionResult(
                selected_backend="numpy",
                confidence=0.0,
                scores=[],
                explanation="No compatible backends found",
                reasoning_steps=[],
            )

        scores = [self._scorer.score(backend, characteristics, strategy) for backend in compatible]
        scores.sort(key=lambda s: s.total_score, reverse=True)

        return SelectionResult(
            selected_backend=scores[0].backend_name,
            confidence=scores[0].total_score,
            scores=scores,
            explanation=self._explainer.generate(
                SelectionResult(
                    selected_backend=scores[0].backend_name,
                    confidence=scores[0].total_score,
                    scores=scores,
                    explanation="",
                    reasoning_steps=[],
                ),
                characteristics,
                strategy,
            ),
            reasoning_steps=self._explainer.generate_reasoning_steps(
                characteristics, scores, scores[0].backend_name, strategy
            ),
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
                f"Large circuit ({characteristics.qubit_count} qubits) - " "simulation may be slow"
            )

        if characteristics.estimated_memory_mb > 1000:
            warnings.append(
                f"High memory requirement ({characteristics.estimated_memory_mb:.0f} MB)"
            )

        if top_score.total_score < 0.5:
            warnings.append("Low confidence in selection - consider alternatives")

        if characteristics.has_custom_gates:
            warnings.append("Custom gates may have limited backend support")

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
- Entanglement density: {characteristics.entanglement_density:.2f}
- Custom gates: {characteristics.has_custom_gates}
- Parameterized: {characteristics.has_parameterized_gates}
- Memory estimate: {characteristics.estimated_memory_mb:.1f} MB

Backend scores:
{chr(10).join(f'- {s.backend_name}: {s.total_score:.2f}' for s in scores[:5])}

Provide a brief recommendation (2-3 sentences) on which backend to use and why."""

    def register_backend(self, backend: BackendCapabilities) -> None:
        """Register a custom backend."""
        self._registry.register(backend)

    def list_backends(self) -> list[str]:
        """List all registered backend names."""
        return [b.name for b in self._registry.list_all()]


# =============================================================================
# Convenience Functions
# =============================================================================


def select_backend(
    circuit: Circuit,
    strategy: str = "balanced",
) -> tuple[str, str]:
    """
    Quick backend selection.

    Args:
        circuit: Circuit to simulate.
        strategy: One of 'performance', 'accuracy', 'memory', 'balanced'.

    Returns:
        Tuple of (backend_name, explanation).
    """
    strategy_map = {
        "performance": SelectionStrategy.PERFORMANCE,
        "accuracy": SelectionStrategy.ACCURACY,
        "memory": SelectionStrategy.MEMORY,
        "balanced": SelectionStrategy.BALANCED,
    }

    selector = BackendSelector()
    result = selector.select(circuit, strategy_map.get(strategy, SelectionStrategy.BALANCED))

    return result.selected_backend, result.explanation


def analyze_circuit(circuit: Circuit) -> CircuitCharacteristics:
    """Analyze a circuit and return its characteristics."""
    return CircuitCharacteristics.from_circuit(circuit)


# Backwards compatibility alias
SelectionInput = CircuitCharacteristics
