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
import time
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


# ==============================================================================
# DEEPER CIRCUIT ANALYSIS - Complexity, Patterns, Optimization
# ==============================================================================


class CircuitPatternType(Enum):
    """Recognized quantum algorithm patterns."""
    
    QFT = "qft"  # Quantum Fourier Transform
    INVERSE_QFT = "inverse_qft"
    QPE = "qpe"  # Quantum Phase Estimation
    GROVER = "grover"  # Grover's Search
    VQE = "vqe"  # Variational Quantum Eigensolver
    QAOA = "qaoa"  # Quantum Approximate Optimization
    QUANTUM_WALK = "quantum_walk"
    ERROR_CORRECTION = "error_correction"
    BELL_STATE = "bell_state"
    GHZ_STATE = "ghz_state"
    AMPLITUDE_ESTIMATION = "amplitude_estimation"
    SWAP_TEST = "swap_test"
    TROTTERIZATION = "trotterization"
    UNKNOWN = "unknown"


@dataclass
class PatternMatch:
    """Result of pattern detection in a circuit."""
    
    pattern_type: CircuitPatternType
    confidence: float  # 0-1 confidence in the match
    gate_indices: list[int]  # Indices of gates involved
    description: str
    optimization_hints: list[str] = field(default_factory=list)


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for a circuit."""
    
    # Basic metrics
    computational_complexity: float  # O(2^n) normalized
    gate_complexity: float  # Based on gate types and depth
    entanglement_complexity: float  # Based on entanglement structure
    
    # Advanced metrics
    t_count: int  # Number of T gates (expensive for fault-tolerant)
    cnot_count: int  # Number of CNOT gates
    two_qubit_gate_ratio: float  # Ratio of 2-qubit gates
    circuit_width_depth_ratio: float  # Parallelism indicator
    
    # Decomposition estimates
    estimated_clifford_count: int
    estimated_non_clifford_count: int
    
    # Resource estimates
    estimated_classical_simulation_time_ms: float
    memory_bandwidth_requirement_mb_s: float
    
    # Complexity class
    complexity_class: str  # "low", "medium", "high", "extreme"
    complexity_score: float  # 0-100 overall score
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "computational_complexity": self.computational_complexity,
            "gate_complexity": self.gate_complexity,
            "entanglement_complexity": self.entanglement_complexity,
            "t_count": self.t_count,
            "cnot_count": self.cnot_count,
            "two_qubit_gate_ratio": self.two_qubit_gate_ratio,
            "circuit_width_depth_ratio": self.circuit_width_depth_ratio,
            "estimated_clifford_count": self.estimated_clifford_count,
            "estimated_non_clifford_count": self.estimated_non_clifford_count,
            "estimated_classical_simulation_time_ms": self.estimated_classical_simulation_time_ms,
            "memory_bandwidth_requirement_mb_s": self.memory_bandwidth_requirement_mb_s,
            "complexity_class": self.complexity_class,
            "complexity_score": self.complexity_score,
        }


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation for a circuit."""
    
    category: str  # "gate_fusion", "layout", "backend", "circuit_structure"
    priority: int  # 1-5, higher is more important
    title: str
    description: str
    expected_improvement: str  # e.g., "10-20% faster"
    implementation_hints: list[str] = field(default_factory=list)


@dataclass  
class DeepCircuitAnalysis:
    """Comprehensive circuit analysis result."""
    
    characteristics: CircuitCharacteristics
    complexity: ComplexityMetrics
    detected_patterns: list[PatternMatch]
    recommendations: list[OptimizationRecommendation]
    backend_suitability: dict[str, float]  # backend_name -> score 0-1
    analysis_timestamp: float = field(default_factory=time.time)
    analysis_duration_ms: float = 0.0
    
    def get_best_backend(self) -> str | None:
        """Get the most suitable backend based on analysis."""
        if not self.backend_suitability:
            return None
        return max(self.backend_suitability, key=self.backend_suitability.get)
    
    def get_top_recommendations(self, n: int = 3) -> list[OptimizationRecommendation]:
        """Get top N recommendations by priority."""
        sorted_recs = sorted(self.recommendations, key=lambda r: r.priority, reverse=True)
        return sorted_recs[:n]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "complexity": self.complexity.to_dict(),
            "detected_patterns": [
                {
                    "type": p.pattern_type.value,
                    "confidence": p.confidence,
                    "description": p.description,
                }
                for p in self.detected_patterns
            ],
            "recommendations": [
                {
                    "category": r.category,
                    "priority": r.priority,
                    "title": r.title,
                    "description": r.description,
                }
                for r in self.recommendations
            ],
            "backend_suitability": self.backend_suitability,
            "analysis_duration_ms": self.analysis_duration_ms,
        }


class GatePatternDetector:
    """Detects common quantum algorithm patterns in circuits."""
    
    # Gate signatures for pattern detection
    QFT_SIGNATURE = {"h", "cp", "cphase", "crz", "swap"}
    GROVER_SIGNATURE = {"h", "x", "z", "cz", "ccz", "mcx"}
    VQE_SIGNATURE = {"rx", "ry", "rz", "cx", "cnot"}
    QAOA_SIGNATURE = {"rx", "rz", "rzz", "cx"}
    
    def __init__(self) -> None:
        self._pattern_cache: dict[int, list[PatternMatch]] = {}
    
    def detect_patterns(
        self,
        circuit: Circuit,
        characteristics: CircuitCharacteristics,
    ) -> list[PatternMatch]:
        """Detect all patterns in a circuit."""
        circuit_hash = hash((
            characteristics.qubit_count,
            characteristics.gate_count,
            tuple(sorted(characteristics.gate_types)),
        ))
        
        if circuit_hash in self._pattern_cache:
            return self._pattern_cache[circuit_hash]
        
        patterns: list[PatternMatch] = []
        
        # Detect specific patterns
        patterns.extend(self._detect_qft_pattern(circuit, characteristics))
        patterns.extend(self._detect_grover_pattern(circuit, characteristics))
        patterns.extend(self._detect_vqe_pattern(circuit, characteristics))
        patterns.extend(self._detect_bell_ghz_pattern(circuit, characteristics))
        patterns.extend(self._detect_qaoa_pattern(circuit, characteristics))
        patterns.extend(self._detect_swap_test_pattern(circuit, characteristics))
        patterns.extend(self._detect_trotterization_pattern(circuit, characteristics))
        
        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Cache result
        self._pattern_cache[circuit_hash] = patterns
        
        # Limit cache size
        if len(self._pattern_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self._pattern_cache.keys())[:500]
            for key in keys_to_remove:
                del self._pattern_cache[key]
        
        return patterns
    
    def _detect_qft_pattern(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> list[PatternMatch]:
        """Detect QFT pattern: H gates followed by controlled phase rotations."""
        patterns = []
        
        # QFT signature: H + controlled rotations + SWAPs
        qft_gates = chars.gate_types & self.QFT_SIGNATURE
        if len(qft_gates) >= 2 and "h" in chars.gate_types:
            # Count Hadamards and controlled phases
            h_count = sum(1 for g in circuit.gates if g.name.lower() == "h")
            cp_count = sum(
                1 for g in circuit.gates 
                if g.name.lower() in ("cp", "cphase", "crz", "cu1")
            )
            
            # QFT has n Hadamards and n(n-1)/2 controlled phases for n qubits
            expected_h = chars.qubit_count
            expected_cp = chars.qubit_count * (chars.qubit_count - 1) // 2
            
            # Calculate confidence based on match
            h_match = min(h_count / max(expected_h, 1), 1.0)
            cp_match = min(cp_count / max(expected_cp, 1), 1.0) if expected_cp > 0 else 0.5
            confidence = (h_match * 0.4 + cp_match * 0.6) if cp_count > 0 else h_match * 0.3
            
            if confidence > 0.4:
                patterns.append(PatternMatch(
                    pattern_type=CircuitPatternType.QFT,
                    confidence=confidence,
                    gate_indices=[],  # Could populate with actual indices
                    description=f"Quantum Fourier Transform pattern detected (H={h_count}, CP={cp_count})",
                    optimization_hints=[
                        "Consider using optimized QFT implementation if available",
                        "QFT benefits from gate fusion optimizations",
                        "GPU backends excel at QFT due to parallelism",
                    ],
                ))
        
        return patterns
    
    def _detect_grover_pattern(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> list[PatternMatch]:
        """Detect Grover's search pattern: H + Oracle + Diffusion."""
        patterns = []
        
        grover_gates = chars.gate_types & self.GROVER_SIGNATURE
        if len(grover_gates) >= 3:
            # Look for repeated H-oracle-diffusion structure
            h_count = sum(1 for g in circuit.gates if g.name.lower() == "h")
            multi_control_count = sum(
                1 for g in circuit.gates 
                if g.name.lower() in ("ccx", "ccz", "mcx", "mct", "toffoli")
            )
            
            # Grover has ~sqrt(N) iterations, each with initial H and diffusion
            # Check if we have enough structure
            if h_count >= chars.qubit_count * 2 and multi_control_count > 0:
                iterations = multi_control_count // 2  # Rough estimate
                confidence = min(0.3 + 0.1 * min(iterations, 5), 0.85)
                
                patterns.append(PatternMatch(
                    pattern_type=CircuitPatternType.GROVER,
                    confidence=confidence,
                    gate_indices=[],
                    description=f"Grover's search pattern (~{iterations} iterations estimated)",
                    optimization_hints=[
                        "Grover benefits from multi-controlled gate optimization",
                        "Consider cuQuantum for large search spaces",
                        "Memory usage scales exponentially with qubits",
                    ],
                ))
        
        return patterns
    
    def _detect_vqe_pattern(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> list[PatternMatch]:
        """Detect VQE pattern: parameterized rotation layers + entangling layers."""
        patterns = []
        
        if not chars.has_parameterized_gates:
            return patterns
        
        vqe_gates = chars.gate_types & self.VQE_SIGNATURE
        if len(vqe_gates) >= 2:
            # Count parameterized rotations
            rotation_count = sum(
                1 for g in circuit.gates 
                if g.name.lower() in ("rx", "ry", "rz") and hasattr(g, "params") and g.params
            )
            cnot_count = sum(
                1 for g in circuit.gates 
                if g.name.lower() in ("cx", "cnot")
            )
            
            if rotation_count >= chars.qubit_count and cnot_count >= chars.qubit_count - 1:
                # Estimate number of layers
                layers = rotation_count // chars.qubit_count
                confidence = min(0.5 + 0.05 * min(layers, 8), 0.9)
                
                patterns.append(PatternMatch(
                    pattern_type=CircuitPatternType.VQE,
                    confidence=confidence,
                    gate_indices=[],
                    description=f"VQE ansatz pattern (~{layers} layers, {rotation_count} rotations)",
                    optimization_hints=[
                        "VQE benefits from gradient computation support",
                        "Consider batched parameter sweeps",
                        "Adjoint differentiation can speed up gradients",
                    ],
                ))
        
        return patterns
    
    def _detect_bell_ghz_pattern(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> list[PatternMatch]:
        """Detect Bell state and GHZ state preparation."""
        patterns = []
        
        gates = [g.name.lower() for g in circuit.gates]
        
        # Bell state: H followed by CNOT on 2 qubits
        if chars.qubit_count == 2 and chars.gate_count <= 5:
            if "h" in gates and any(g in gates for g in ("cx", "cnot")):
                h_idx = gates.index("h")
                cnot_indices = [i for i, g in enumerate(gates) if g in ("cx", "cnot")]
                
                if cnot_indices and any(i > h_idx for i in cnot_indices):
                    patterns.append(PatternMatch(
                        pattern_type=CircuitPatternType.BELL_STATE,
                        confidence=0.95,
                        gate_indices=[h_idx] + cnot_indices,
                        description="Bell state preparation (H + CNOT)",
                        optimization_hints=[
                            "Bell states are trivial for all backends",
                            "Useful for entanglement verification",
                        ],
                    ))
        
        # GHZ state: H followed by chain of CNOTs
        if chars.qubit_count >= 3 and "h" in gates:
            cnot_count = sum(1 for g in gates if g in ("cx", "cnot"))
            
            if cnot_count == chars.qubit_count - 1 and chars.gate_count <= chars.qubit_count * 2:
                patterns.append(PatternMatch(
                    pattern_type=CircuitPatternType.GHZ_STATE,
                    confidence=0.85,
                    gate_indices=[],
                    description=f"GHZ state preparation ({chars.qubit_count} qubits)",
                    optimization_hints=[
                        "GHZ preparation is linear-depth",
                        "All backends handle this efficiently",
                    ],
                ))
        
        return patterns
    
    def _detect_qaoa_pattern(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> list[PatternMatch]:
        """Detect QAOA pattern: RX/RZ mixers + problem Hamiltonian."""
        patterns = []
        
        if not chars.has_parameterized_gates:
            return patterns
        
        qaoa_gates = chars.gate_types & self.QAOA_SIGNATURE
        if len(qaoa_gates) >= 2 and "rx" in chars.gate_types:
            rz_count = sum(1 for g in circuit.gates if g.name.lower() == "rz")
            rx_count = sum(1 for g in circuit.gates if g.name.lower() == "rx")
            
            # QAOA has balanced RX (mixer) and RZ (problem) gates
            if rx_count >= chars.qubit_count and rz_count >= chars.qubit_count:
                layers = min(rx_count, rz_count) // chars.qubit_count
                confidence = min(0.4 + 0.1 * min(layers, 5), 0.85)
                
                patterns.append(PatternMatch(
                    pattern_type=CircuitPatternType.QAOA,
                    confidence=confidence,
                    gate_indices=[],
                    description=f"QAOA pattern (~{layers} layers)",
                    optimization_hints=[
                        "QAOA benefits from parameter optimization",
                        "Consider noise-aware backends for NISQ execution",
                    ],
                ))
        
        return patterns
    
    def _detect_swap_test_pattern(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> list[PatternMatch]:
        """Detect SWAP test for state comparison."""
        patterns = []
        
        if chars.qubit_count >= 3 and chars.qubit_count % 2 == 1:
            # SWAP test uses odd number of qubits (1 ancilla + 2n for comparison)
            if "h" in chars.gate_types and "cswap" in chars.gate_types:
                patterns.append(PatternMatch(
                    pattern_type=CircuitPatternType.SWAP_TEST,
                    confidence=0.8,
                    gate_indices=[],
                    description="SWAP test for state comparison",
                    optimization_hints=[
                        "SWAP test is efficient on all backends",
                    ],
                ))
        
        return patterns
    
    def _detect_trotterization_pattern(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> list[PatternMatch]:
        """Detect Trotterization for Hamiltonian simulation."""
        patterns = []
        
        # Trotterization has repeated layers of rotation gates
        if chars.gate_count > chars.qubit_count * 5:
            rotation_gates = {"rx", "ry", "rz", "rxx", "ryy", "rzz"}
            rotation_count = sum(
                1 for g in circuit.gates 
                if g.name.lower() in rotation_gates
            )
            
            # High ratio of rotations suggests Trotterization
            if rotation_count / chars.gate_count > 0.6:
                steps = rotation_count // (chars.qubit_count * 2)  # Rough estimate
                patterns.append(PatternMatch(
                    pattern_type=CircuitPatternType.TROTTERIZATION,
                    confidence=0.7,
                    gate_indices=[],
                    description=f"Trotterization pattern (~{steps} steps)",
                    optimization_hints=[
                        "Trotterization benefits from gate fusion",
                        "Consider higher-order Trotter formulas",
                        "GPU backends excel at repeated structure",
                    ],
                ))
        
        return patterns


class CircuitComplexityAnalyzer:
    """Analyzes computational complexity of quantum circuits."""
    
    # Gate costs (approximate in terms of error rate and runtime)
    GATE_COSTS = {
        "h": 1.0,
        "x": 1.0, "y": 1.0, "z": 1.0,
        "s": 1.0, "t": 2.0, "tdg": 2.0,
        "rx": 1.5, "ry": 1.5, "rz": 1.0,
        "cx": 10.0, "cnot": 10.0, "cz": 10.0,
        "swap": 30.0,
        "ccx": 50.0, "toffoli": 50.0, "ccz": 50.0,
        "u1": 1.0, "u2": 1.5, "u3": 2.0,
    }
    
    # Clifford gates (cheap for error correction)
    CLIFFORD_GATES = {"h", "s", "cx", "cnot", "cz", "x", "y", "z", "swap"}
    
    def __init__(self) -> None:
        self._analysis_cache: dict[int, ComplexityMetrics] = {}
    
    def analyze(
        self,
        circuit: Circuit,
        characteristics: CircuitCharacteristics,
    ) -> ComplexityMetrics:
        """Perform comprehensive complexity analysis."""
        circuit_hash = hash((
            characteristics.qubit_count,
            characteristics.gate_count,
            characteristics.depth,
            tuple(sorted(characteristics.gate_types)),
        ))
        
        if circuit_hash in self._analysis_cache:
            return self._analysis_cache[circuit_hash]
        
        # Count specific gates
        t_count = sum(
            1 for g in circuit.gates 
            if g.name.lower() in ("t", "tdg")
        )
        cnot_count = sum(
            1 for g in circuit.gates 
            if g.name.lower() in ("cx", "cnot")
        )
        two_qubit_count = sum(
            1 for g in circuit.gates 
            if len(g.qubits) == 2
        )
        
        # Clifford vs non-Clifford
        clifford_count = sum(
            1 for g in circuit.gates 
            if g.name.lower() in self.CLIFFORD_GATES
        )
        non_clifford_count = characteristics.gate_count - clifford_count
        
        # Compute ratios
        two_qubit_ratio = two_qubit_count / max(characteristics.gate_count, 1)
        width_depth_ratio = characteristics.qubit_count / max(characteristics.depth, 1)
        
        # Compute complexity scores
        computational_complexity = self._compute_computational_complexity(characteristics)
        gate_complexity = self._compute_gate_complexity(circuit, characteristics)
        entanglement_complexity = self._compute_entanglement_complexity(
            characteristics, two_qubit_count
        )
        
        # Estimate simulation time
        simulation_time_ms = self._estimate_simulation_time(
            characteristics, gate_complexity
        )
        
        # Memory bandwidth requirement
        memory_bandwidth = self._estimate_memory_bandwidth(characteristics)
        
        # Overall complexity score and class
        complexity_score = (
            computational_complexity * 0.4 +
            gate_complexity * 0.3 +
            entanglement_complexity * 0.3
        ) * 100
        
        if complexity_score < 20:
            complexity_class = "low"
        elif complexity_score < 50:
            complexity_class = "medium"
        elif complexity_score < 80:
            complexity_class = "high"
        else:
            complexity_class = "extreme"
        
        metrics = ComplexityMetrics(
            computational_complexity=computational_complexity,
            gate_complexity=gate_complexity,
            entanglement_complexity=entanglement_complexity,
            t_count=t_count,
            cnot_count=cnot_count,
            two_qubit_gate_ratio=two_qubit_ratio,
            circuit_width_depth_ratio=width_depth_ratio,
            estimated_clifford_count=clifford_count,
            estimated_non_clifford_count=non_clifford_count,
            estimated_classical_simulation_time_ms=simulation_time_ms,
            memory_bandwidth_requirement_mb_s=memory_bandwidth,
            complexity_class=complexity_class,
            complexity_score=complexity_score,
        )
        
        # Cache result
        self._analysis_cache[circuit_hash] = metrics
        
        # Limit cache size
        if len(self._analysis_cache) > 500:
            keys_to_remove = list(self._analysis_cache.keys())[:250]
            for key in keys_to_remove:
                del self._analysis_cache[key]
        
        return metrics
    
    def _compute_computational_complexity(
        self,
        chars: CircuitCharacteristics,
    ) -> float:
        """Compute normalized computational complexity O(2^n)."""
        # State vector simulation scales as O(2^n * depth)
        # Normalize to 0-1 scale
        n = chars.qubit_count
        
        if n <= 10:
            return n / 30  # Very tractable
        elif n <= 20:
            return 0.33 + (n - 10) / 30  # Tractable with good hardware
        elif n <= 30:
            return 0.66 + (n - 20) / 30  # Challenging
        else:
            return min(1.0, 0.9 + (n - 30) / 100)  # Extreme
    
    def _compute_gate_complexity(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> float:
        """Compute gate-based complexity score."""
        total_cost = 0.0
        
        for gate in circuit.gates:
            gate_name = gate.name.lower()
            cost = self.GATE_COSTS.get(gate_name, 5.0)  # Default cost for unknown
            total_cost += cost
        
        # Normalize by circuit size
        avg_cost = total_cost / max(chars.gate_count, 1)
        depth_factor = min(chars.depth / 100, 1.0)
        
        # Combine average cost and depth
        normalized = (avg_cost / 50 + depth_factor) / 2
        return min(normalized, 1.0)
    
    def _compute_entanglement_complexity(
        self,
        chars: CircuitCharacteristics,
        two_qubit_count: int,
    ) -> float:
        """Compute entanglement-based complexity."""
        # High entanglement means harder classical simulation
        density = chars.entanglement_density
        
        # Factor in absolute count
        count_factor = min(two_qubit_count / 100, 1.0)
        
        # Combine density and count
        return (density * 0.6 + count_factor * 0.4)
    
    def _estimate_simulation_time(
        self,
        chars: CircuitCharacteristics,
        gate_complexity: float,
    ) -> float:
        """Estimate classical simulation time in milliseconds."""
        # Rough estimate: base time * 2^n * depth * complexity
        base_time_per_gate_us = 0.1  # 0.1 microseconds per gate operation
        
        state_size = 2 ** chars.qubit_count
        time_us = base_time_per_gate_us * state_size * chars.gate_count * (1 + gate_complexity)
        
        return time_us / 1000  # Convert to ms
    
    def _estimate_memory_bandwidth(
        self,
        chars: CircuitCharacteristics,
    ) -> float:
        """Estimate memory bandwidth requirement in MB/s."""
        # State vector access patterns
        state_size_mb = chars.estimated_memory_mb
        
        # Each gate accesses portions of state vector
        # Two-qubit gates access more memory
        accesses_per_gate = 2 * chars.entanglement_density + 1
        
        # Assume 1ms gate time
        bandwidth = state_size_mb * chars.gate_count * accesses_per_gate / (chars.depth * 0.001)
        
        return min(bandwidth, 1e6)  # Cap at 1 TB/s


class OptimizationRecommendationEngine:
    """Generates optimization recommendations based on circuit analysis."""
    
    def __init__(self) -> None:
        self._backend_profiles: dict[str, dict[str, Any]] = {
            "qsim": {
                "strengths": ["large circuits", "parallelism", "gate fusion"],
                "max_efficient_qubits": 32,
                "optimal_patterns": [CircuitPatternType.QFT, CircuitPatternType.GROVER],
            },
            "quest": {
                "strengths": ["distributed", "memory efficiency", "MPI support"],
                "max_efficient_qubits": 40,
                "optimal_patterns": [CircuitPatternType.VQE, CircuitPatternType.QAOA],
            },
            "cuquantum": {
                "strengths": ["GPU acceleration", "tensor networks", "large circuits"],
                "max_efficient_qubits": 35,
                "optimal_patterns": [CircuitPatternType.TROTTERIZATION],
            },
            "cirq": {
                "strengths": ["flexibility", "noise modeling", "NISQ"],
                "max_efficient_qubits": 25,
                "optimal_patterns": [CircuitPatternType.VQE, CircuitPatternType.QAOA],
            },
            "lret": {
                "strengths": ["tensor networks", "low-entanglement", "MPS"],
                "max_efficient_qubits": 100,
                "optimal_patterns": [CircuitPatternType.TROTTERIZATION],
            },
        }
    
    def generate_recommendations(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        patterns: list[PatternMatch],
    ) -> list[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations: list[OptimizationRecommendation] = []
        
        # Gate fusion recommendations
        recommendations.extend(
            self._gate_fusion_recommendations(chars, complexity)
        )
        
        # Backend-specific recommendations
        recommendations.extend(
            self._backend_recommendations(chars, complexity, patterns)
        )
        
        # Circuit structure recommendations
        recommendations.extend(
            self._structure_recommendations(chars, complexity)
        )
        
        # Memory optimization recommendations
        recommendations.extend(
            self._memory_recommendations(chars, complexity)
        )
        
        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations
    
    def compute_backend_suitability(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        patterns: list[PatternMatch],
        available_backends: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute suitability score for each backend."""
        if available_backends is None:
            available_backends = list(self._backend_profiles.keys())
        
        scores: dict[str, float] = {}
        
        for backend in available_backends:
            profile = self._backend_profiles.get(backend, {})
            score = 0.5  # Base score
            
            # Qubit efficiency
            max_qubits = profile.get("max_efficient_qubits", 25)
            if chars.qubit_count <= max_qubits:
                score += 0.2
            elif chars.qubit_count <= max_qubits * 1.2:
                score += 0.1
            else:
                score -= 0.2
            
            # Pattern match bonus
            optimal_patterns = profile.get("optimal_patterns", [])
            for pattern in patterns:
                if pattern.pattern_type in optimal_patterns:
                    score += 0.15 * pattern.confidence
            
            # Complexity match
            if complexity.complexity_class == "extreme":
                if backend in ("quest", "cuquantum"):
                    score += 0.1
            elif complexity.complexity_class == "low":
                if backend == "cirq":
                    score += 0.1
            
            # Entanglement consideration for LRET
            if backend == "lret" and chars.entanglement_density < 0.3:
                score += 0.25  # LRET excels at low entanglement
            elif backend == "lret" and chars.entanglement_density > 0.7:
                score -= 0.3  # LRET struggles with high entanglement
            
            scores[backend] = min(max(score, 0.0), 1.0)
        
        return scores
    
    def _gate_fusion_recommendations(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> list[OptimizationRecommendation]:
        """Generate gate fusion recommendations."""
        recs = []
        
        if chars.depth > 50 and complexity.two_qubit_gate_ratio > 0.3:
            recs.append(OptimizationRecommendation(
                category="gate_fusion",
                priority=4,
                title="Enable Two-Qubit Gate Fusion",
                description=(
                    f"Circuit has {complexity.cnot_count} CNOT gates. "
                    "Fusing adjacent two-qubit gates can reduce depth."
                ),
                expected_improvement="10-30% depth reduction",
                implementation_hints=[
                    "Use qsim's gate fusion option",
                    "Consider cuQuantum's automatic fusion",
                ],
            ))
        
        if complexity.t_count > 10:
            recs.append(OptimizationRecommendation(
                category="gate_fusion",
                priority=3,
                title="T-Gate Optimization",
                description=(
                    f"Circuit has {complexity.t_count} T gates. "
                    "T gates are expensive for fault-tolerant computation."
                ),
                expected_improvement="Reduced magic state usage",
                implementation_hints=[
                    "Consider T-gate synthesis optimization",
                    "Look for T-gate cancellation opportunities",
                ],
            ))
        
        return recs
    
    def _backend_recommendations(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        patterns: list[PatternMatch],
    ) -> list[OptimizationRecommendation]:
        """Generate backend-specific recommendations."""
        recs = []
        
        if chars.qubit_count > 25:
            recs.append(OptimizationRecommendation(
                category="backend",
                priority=5,
                title="Use High-Performance Backend",
                description=(
                    f"Circuit has {chars.qubit_count} qubits. "
                    "Consider qsim, cuQuantum, or QuEST for efficiency."
                ),
                expected_improvement="2-10x faster simulation",
                implementation_hints=[
                    "qsim: Best for CPU parallelism",
                    "cuQuantum: Best if GPU available",
                    "QuEST: Best for distributed computing",
                ],
            ))
        
        if chars.entanglement_density < 0.2 and chars.qubit_count > 30:
            recs.append(OptimizationRecommendation(
                category="backend",
                priority=5,
                title="Consider LRET/MPS Backend",
                description=(
                    f"Low entanglement density ({chars.entanglement_density:.1%}) "
                    "makes tensor network methods efficient."
                ),
                expected_improvement="Exponential speedup possible",
                implementation_hints=[
                    "LRET can handle 100+ qubits for low entanglement",
                    "Set appropriate bond dimension",
                ],
            ))
        
        # Pattern-specific recommendations
        for pattern in patterns:
            if pattern.pattern_type == CircuitPatternType.VQE:
                recs.append(OptimizationRecommendation(
                    category="backend",
                    priority=4,
                    title="VQE-Optimized Backend",
                    description="VQE detected. Use backend with gradient support.",
                    expected_improvement="Faster parameter optimization",
                    implementation_hints=[
                        "cirq supports adjoint differentiation",
                        "Consider batched parameter evaluation",
                    ],
                ))
                break
        
        return recs
    
    def _structure_recommendations(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> list[OptimizationRecommendation]:
        """Generate circuit structure recommendations."""
        recs = []
        
        if complexity.circuit_width_depth_ratio < 0.5:
            recs.append(OptimizationRecommendation(
                category="circuit_structure",
                priority=3,
                title="Increase Circuit Parallelism",
                description=(
                    f"Width/depth ratio is {complexity.circuit_width_depth_ratio:.2f}. "
                    "Consider reordering gates for more parallelism."
                ),
                expected_improvement="Better hardware utilization",
                implementation_hints=[
                    "Identify independent gate layers",
                    "Use circuit optimization passes",
                ],
            ))
        
        if chars.has_mid_circuit_measurement:
            recs.append(OptimizationRecommendation(
                category="circuit_structure",
                priority=4,
                title="Mid-Circuit Measurement Handling",
                description="Circuit contains mid-circuit measurements.",
                expected_improvement="Correct execution with state collapse",
                implementation_hints=[
                    "Ensure backend supports MCM",
                    "Consider deferred measurement if possible",
                ],
            ))
        
        return recs
    
    def _memory_recommendations(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> list[OptimizationRecommendation]:
        """Generate memory optimization recommendations."""
        recs = []
        
        if chars.estimated_memory_mb > 1024:  # > 1GB
            recs.append(OptimizationRecommendation(
                category="memory",
                priority=5,
                title="High Memory Requirements",
                description=(
                    f"Estimated memory: {chars.estimated_memory_mb:.0f} MB. "
                    "Ensure sufficient system resources."
                ),
                expected_improvement="Avoid out-of-memory errors",
                implementation_hints=[
                    "Use memory-efficient backends",
                    "Consider distributed simulation",
                    "Close other applications",
                ],
            ))
        
        if complexity.memory_bandwidth_requirement_mb_s > 10000:
            recs.append(OptimizationRecommendation(
                category="memory",
                priority=3,
                title="High Memory Bandwidth",
                description="Circuit requires high memory bandwidth.",
                expected_improvement="Faster execution with proper hardware",
                implementation_hints=[
                    "GPU backends have higher bandwidth",
                    "Consider cache-efficient gate ordering",
                ],
            ))
        
        return recs


class DeepCircuitAnalyzer:
    """Main class for deep circuit analysis combining all analyzers."""
    
    def __init__(self) -> None:
        self._pattern_detector = GatePatternDetector()
        self._complexity_analyzer = CircuitComplexityAnalyzer()
        self._recommendation_engine = OptimizationRecommendationEngine()
    
    def analyze(
        self,
        circuit: Circuit,
        available_backends: list[str] | None = None,
    ) -> DeepCircuitAnalysis:
        """Perform comprehensive circuit analysis."""
        start_time = time.time()
        
        # Get basic characteristics
        characteristics = CircuitCharacteristics.from_circuit(circuit)
        
        # Detect patterns
        patterns = self._pattern_detector.detect_patterns(circuit, characteristics)
        
        # Analyze complexity
        complexity = self._complexity_analyzer.analyze(circuit, characteristics)
        
        # Generate recommendations
        recommendations = self._recommendation_engine.generate_recommendations(
            characteristics, complexity, patterns
        )
        
        # Compute backend suitability
        backend_suitability = self._recommendation_engine.compute_backend_suitability(
            characteristics, complexity, patterns, available_backends
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        return DeepCircuitAnalysis(
            characteristics=characteristics,
            complexity=complexity,
            detected_patterns=patterns,
            recommendations=recommendations,
            backend_suitability=backend_suitability,
            analysis_duration_ms=duration_ms,
        )
    
    def quick_analyze(
        self,
        circuit: Circuit,
    ) -> tuple[str, float, list[str]]:
        """Quick analysis returning best backend, confidence, and top hints."""
        analysis = self.analyze(circuit)
        
        best_backend = analysis.get_best_backend() or "cirq"
        confidence = analysis.backend_suitability.get(best_backend, 0.5)
        
        hints = [
            r.title for r in analysis.get_top_recommendations(3)
        ]
        
        return best_backend, confidence, hints


# ==============================================================================
# ADVANCED CIRCUIT COMPLEXITY ANALYSIS (5% Gap Coverage)
# Theoretical Bounds, Quantum Volume, Parallelizability, Synthesis Analysis
# ==============================================================================


class ComplexityBound(Enum):
    """Types of complexity bounds."""
    
    EXACT = "exact"  # Known exact complexity
    UPPER = "upper"  # Upper bound
    LOWER = "lower"  # Lower bound
    ASYMPTOTIC = "asymptotic"  # Big-O notation


@dataclass
class TheoreticalComplexityBounds:
    """Theoretical complexity bounds for a circuit."""
    
    # Time complexity
    time_complexity_class: str  # e.g., "O(2^n)", "O(n^2)"
    time_bound_type: ComplexityBound
    
    # Space complexity
    space_complexity_class: str  # e.g., "O(2^n)"
    space_bound_type: ComplexityBound
    
    # Circuit complexity measures
    circuit_depth_optimal: bool  # Is depth provably optimal?
    gate_count_optimal: bool  # Is gate count provably optimal?
    
    # Known bounds
    t_depth_lower_bound: int | None  # Lower bound on T-depth
    cnot_count_lower_bound: int | None  # Lower bound on CNOT count
    
    # Hardness indicators
    is_bqp_complete: bool = False  # BQP-complete problems
    is_classically_hard: bool = False  # Classically intractable
    classical_simulation_method: str | None = None  # If simulable, how
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_complexity_class": self.time_complexity_class,
            "time_bound_type": self.time_bound_type.value,
            "space_complexity_class": self.space_complexity_class,
            "space_bound_type": self.space_bound_type.value,
            "circuit_depth_optimal": self.circuit_depth_optimal,
            "gate_count_optimal": self.gate_count_optimal,
            "t_depth_lower_bound": self.t_depth_lower_bound,
            "cnot_count_lower_bound": self.cnot_count_lower_bound,
            "is_bqp_complete": self.is_bqp_complete,
            "is_classically_hard": self.is_classically_hard,
            "classical_simulation_method": self.classical_simulation_method,
        }


@dataclass
class GateSynthesisAnalysis:
    """Analysis of gate synthesis requirements and costs."""
    
    # T-gate analysis (critical for fault-tolerant QC)
    t_count: int
    t_depth: int  # Depth considering only T gates
    t_gates_per_layer: list[int]  # T gates per depth layer
    
    # Clifford analysis
    clifford_count: int
    clifford_depth: int
    
    # Rotation gates
    rotation_count: int
    rotation_synthesis_cost: int  # Estimated T gates after synthesis
    rotation_precision_bits: int  # Required precision
    
    # Synthesis estimates for fault-tolerant
    estimated_logical_t_gates: int
    estimated_magic_state_factories: int
    estimated_logical_qubits: int
    
    # Decomposition analysis
    needs_decomposition: bool
    decomposition_overhead: float  # Multiplier on gate count
    custom_gates_to_decompose: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "t_count": self.t_count,
            "t_depth": self.t_depth,
            "t_gates_per_layer": self.t_gates_per_layer,
            "clifford_count": self.clifford_count,
            "clifford_depth": self.clifford_depth,
            "rotation_count": self.rotation_count,
            "rotation_synthesis_cost": self.rotation_synthesis_cost,
            "rotation_precision_bits": self.rotation_precision_bits,
            "estimated_logical_t_gates": self.estimated_logical_t_gates,
            "estimated_magic_state_factories": self.estimated_magic_state_factories,
            "estimated_logical_qubits": self.estimated_logical_qubits,
            "needs_decomposition": self.needs_decomposition,
            "decomposition_overhead": self.decomposition_overhead,
            "custom_gates_to_decompose": self.custom_gates_to_decompose,
        }


@dataclass
class QuantumVolumeEstimate:
    """Quantum volume estimation for the circuit."""
    
    # Circuit dimensions
    effective_qubits: int  # Qubits actively used
    effective_depth: int  # Depth of non-trivial operations
    
    # Quantum volume proxy
    log_quantum_volume: float  # log2(QV)
    quantum_volume_class: str  # "trivial", "moderate", "high", "extreme"
    
    # Heavy output probability estimates
    estimated_success_probability: float  # Without noise
    estimated_noisy_success: float  # With typical noise
    
    # Noise resilience
    critical_depth: int  # Depth where noise dominates
    coherence_limited: bool  # Whether coherence time is limiting
    gate_error_limited: bool  # Whether gate errors dominate
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "effective_qubits": self.effective_qubits,
            "effective_depth": self.effective_depth,
            "log_quantum_volume": self.log_quantum_volume,
            "quantum_volume_class": self.quantum_volume_class,
            "estimated_success_probability": self.estimated_success_probability,
            "estimated_noisy_success": self.estimated_noisy_success,
            "critical_depth": self.critical_depth,
            "coherence_limited": self.coherence_limited,
            "gate_error_limited": self.gate_error_limited,
        }


@dataclass
class ParallelizabilityMetrics:
    """Metrics for parallel execution potential."""
    
    # Layer analysis
    total_layers: int
    parallelizable_layers: int
    max_gates_per_layer: int
    average_gates_per_layer: float
    
    # Parallelism scores
    circuit_parallelism: float  # 0-1, higher = more parallel
    qubit_utilization: float  # 0-1, avg fraction of qubits active
    critical_path_length: int  # Longest dependency chain
    
    # DAG analysis
    dag_width: int  # Max parallel operations
    dag_height: int  # Critical path
    gate_dependencies: int  # Number of gate dependencies
    
    # Speedup estimates
    theoretical_gpu_speedup: float  # Max speedup with GPU
    theoretical_distributed_speedup: float  # Max with distribution
    amdahl_parallel_fraction: float  # Parallelizable fraction
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_layers": self.total_layers,
            "parallelizable_layers": self.parallelizable_layers,
            "max_gates_per_layer": self.max_gates_per_layer,
            "average_gates_per_layer": self.average_gates_per_layer,
            "circuit_parallelism": self.circuit_parallelism,
            "qubit_utilization": self.qubit_utilization,
            "critical_path_length": self.critical_path_length,
            "dag_width": self.dag_width,
            "dag_height": self.dag_height,
            "gate_dependencies": self.gate_dependencies,
            "theoretical_gpu_speedup": self.theoretical_gpu_speedup,
            "theoretical_distributed_speedup": self.theoretical_distributed_speedup,
            "amdahl_parallel_fraction": self.amdahl_parallel_fraction,
        }


@dataclass
class ClassicalSimulationFeasibility:
    """Analysis of classical simulation feasibility."""
    
    # Feasibility assessment
    is_efficiently_simulable: bool
    simulation_method: str  # "state_vector", "tensor_network", "clifford", "stabilizer"
    
    # Resource estimates
    estimated_time_seconds: float
    estimated_memory_gb: float
    estimated_flops: float
    
    # Limiting factors
    primary_bottleneck: str  # "memory", "time", "entanglement", "none"
    bottleneck_description: str
    
    # Optimization opportunities
    can_use_tensor_network: bool
    tensor_network_speedup: float
    can_use_stabilizer_simulation: bool
    stabilizer_fraction: float  # Fraction of gates that are Clifford
    
    # Specialized simulators
    recommended_simulator: str
    simulator_specific_optimizations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_efficiently_simulable": self.is_efficiently_simulable,
            "simulation_method": self.simulation_method,
            "estimated_time_seconds": self.estimated_time_seconds,
            "estimated_memory_gb": self.estimated_memory_gb,
            "estimated_flops": self.estimated_flops,
            "primary_bottleneck": self.primary_bottleneck,
            "bottleneck_description": self.bottleneck_description,
            "can_use_tensor_network": self.can_use_tensor_network,
            "tensor_network_speedup": self.tensor_network_speedup,
            "can_use_stabilizer_simulation": self.can_use_stabilizer_simulation,
            "stabilizer_fraction": self.stabilizer_fraction,
            "recommended_simulator": self.recommended_simulator,
            "simulator_specific_optimizations": self.simulator_specific_optimizations,
        }


@dataclass
class EntanglementStructure:
    """Analysis of entanglement structure in the circuit."""
    
    # Qubit connectivity
    qubit_graph_edges: int  # Edges in qubit interaction graph
    graph_diameter: int  # Max shortest path between qubits
    is_planar: bool  # Can be mapped to 2D grid
    connectivity_type: str  # "linear", "ring", "all-to-all", "grid", "custom"
    
    # Entanglement metrics
    max_schmidt_rank: int  # Upper bound on entanglement
    estimated_bond_dimension: int  # For MPS simulation
    entanglement_entropy_estimate: float  # Average bipartite entropy
    
    # Structure patterns
    is_layered: bool  # Has clear layer structure
    has_localized_entanglement: bool  # Entanglement is spatially limited
    entanglement_light_cone: int  # Qubits that can be entangled
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "qubit_graph_edges": self.qubit_graph_edges,
            "graph_diameter": self.graph_diameter,
            "is_planar": self.is_planar,
            "connectivity_type": self.connectivity_type,
            "max_schmidt_rank": self.max_schmidt_rank,
            "estimated_bond_dimension": self.estimated_bond_dimension,
            "entanglement_entropy_estimate": self.entanglement_entropy_estimate,
            "is_layered": self.is_layered,
            "has_localized_entanglement": self.has_localized_entanglement,
            "entanglement_light_cone": self.entanglement_light_cone,
        }


@dataclass
class AdvancedCircuitAnalysis:
    """Complete advanced circuit complexity analysis."""
    
    # Basic analysis
    characteristics: CircuitCharacteristics
    complexity: ComplexityMetrics
    
    # Advanced analysis
    theoretical_bounds: TheoreticalComplexityBounds
    synthesis_analysis: GateSynthesisAnalysis
    quantum_volume: QuantumVolumeEstimate
    parallelizability: ParallelizabilityMetrics
    simulation_feasibility: ClassicalSimulationFeasibility
    entanglement_structure: EntanglementStructure
    
    # Timestamps
    analysis_timestamp: float = field(default_factory=time.time)
    analysis_duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "complexity": self.complexity.to_dict(),
            "theoretical_bounds": self.theoretical_bounds.to_dict(),
            "synthesis_analysis": self.synthesis_analysis.to_dict(),
            "quantum_volume": self.quantum_volume.to_dict(),
            "parallelizability": self.parallelizability.to_dict(),
            "simulation_feasibility": self.simulation_feasibility.to_dict(),
            "entanglement_structure": self.entanglement_structure.to_dict(),
            "analysis_duration_ms": self.analysis_duration_ms,
        }
    
    def get_summary(self) -> dict[str, Any]:
        """Get a concise summary of the analysis."""
        return {
            "qubits": self.characteristics.qubit_count,
            "depth": self.characteristics.depth,
            "gate_count": self.characteristics.gate_count,
            "complexity_class": self.complexity.complexity_class,
            "is_classically_hard": self.theoretical_bounds.is_classically_hard,
            "quantum_volume_class": self.quantum_volume.quantum_volume_class,
            "parallelism_score": self.parallelizability.circuit_parallelism,
            "simulation_method": self.simulation_feasibility.simulation_method,
            "estimated_time_seconds": self.simulation_feasibility.estimated_time_seconds,
        }


class TheoreticalComplexityAnalyzer:
    """Analyzes theoretical complexity bounds for circuits."""
    
    # Known circuit families and their complexities
    KNOWN_COMPLEXITIES: dict[str, tuple[str, str]] = {
        "qft": ("O(n^2)", "O(2^n)"),  # time, space for full state
        "grover": ("O(sqrt(N))", "O(2^n)"),
        "vqe": ("O(depth)", "O(2^n)"),
        "qaoa": ("O(depth)", "O(2^n)"),
        "trotterization": ("O(t * steps)", "O(2^n)"),
    }
    
    def __init__(self) -> None:
        self._cache: dict[int, TheoreticalComplexityBounds] = {}
    
    def analyze(
        self,
        circuit: Circuit,
        characteristics: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        detected_patterns: list[PatternMatch] | None = None,
    ) -> TheoreticalComplexityBounds:
        """Analyze theoretical complexity bounds."""
        # Generate cache key
        cache_key = hash((
            characteristics.qubit_count,
            characteristics.gate_count,
            characteristics.depth,
            complexity.t_count,
        ))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        n = characteristics.qubit_count
        
        # Determine complexity class based on circuit structure
        time_class, time_bound = self._determine_time_complexity(
            circuit, characteristics, complexity, detected_patterns
        )
        space_class, space_bound = self._determine_space_complexity(
            characteristics, complexity
        )
        
        # Calculate optimality
        depth_optimal = self._is_depth_optimal(circuit, characteristics)
        gate_optimal = self._is_gate_count_optimal(circuit, characteristics, complexity)
        
        # Lower bounds
        t_lower = self._t_depth_lower_bound(characteristics, complexity)
        cnot_lower = self._cnot_lower_bound(characteristics, complexity)
        
        # Hardness analysis
        is_hard = self._is_classically_hard(characteristics, complexity)
        simulation_method = self._get_simulation_method(characteristics, complexity)
        
        bounds = TheoreticalComplexityBounds(
            time_complexity_class=time_class,
            time_bound_type=time_bound,
            space_complexity_class=space_class,
            space_bound_type=space_bound,
            circuit_depth_optimal=depth_optimal,
            gate_count_optimal=gate_optimal,
            t_depth_lower_bound=t_lower,
            cnot_count_lower_bound=cnot_lower,
            is_bqp_complete=n >= 50 and complexity.t_count > 0,  # Rough heuristic
            is_classically_hard=is_hard,
            classical_simulation_method=simulation_method,
        )
        
        self._cache[cache_key] = bounds
        return bounds
    
    def _determine_time_complexity(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        patterns: list[PatternMatch] | None,
    ) -> tuple[str, ComplexityBound]:
        """Determine time complexity class."""
        n = chars.qubit_count
        
        # Check for known patterns
        if patterns:
            for pattern in patterns:
                if pattern.pattern_type.value in self.KNOWN_COMPLEXITIES:
                    time_class, _ = self.KNOWN_COMPLEXITIES[pattern.pattern_type.value]
                    return time_class, ComplexityBound.ASYMPTOTIC
        
        # Generic analysis
        if complexity.estimated_clifford_count == chars.gate_count:
            # Pure Clifford circuit - polynomial simulation
            return f"O(n^2 * depth)", ComplexityBound.UPPER
        
        if n <= 10:
            return f"O(2^{n})", ComplexityBound.EXACT
        elif n <= 20:
            return f"O(2^n)", ComplexityBound.ASYMPTOTIC
        else:
            return f"O(2^n)", ComplexityBound.LOWER
    
    def _determine_space_complexity(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> tuple[str, ComplexityBound]:
        """Determine space complexity class."""
        n = chars.qubit_count
        
        # State vector simulation
        if chars.simulation_type == SimulationType.STATE_VECTOR:
            return f"O(2^{n})", ComplexityBound.EXACT
        
        # Density matrix
        if chars.simulation_type == SimulationType.DENSITY_MATRIX:
            return f"O(4^{n})", ComplexityBound.EXACT
        
        return f"O(2^n)", ComplexityBound.ASYMPTOTIC
    
    def _is_depth_optimal(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> bool:
        """Check if circuit depth is near-optimal."""
        # Heuristic: depth is optimal if close to theoretical minimum
        # Min depth for n qubits with full connectivity is O(n) for QFT
        min_possible = max(1, chars.gate_count // chars.qubit_count)
        return chars.depth <= min_possible * 1.5
    
    def _is_gate_count_optimal(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> bool:
        """Check if gate count is near-optimal."""
        # Very rough heuristic
        n = chars.qubit_count
        depth = chars.depth
        
        # Theoretical minimum for a depth-d circuit on n qubits
        min_gates = depth  # At least one gate per layer
        max_efficient = n * depth * 2  # Reasonable upper bound
        
        return chars.gate_count <= max_efficient
    
    def _t_depth_lower_bound(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> int | None:
        """Calculate lower bound on T-depth."""
        if complexity.t_count == 0:
            return 0
        
        # T-depth is at least ceil(T-count / n) with perfect parallelism
        return max(1, (complexity.t_count + chars.qubit_count - 1) // chars.qubit_count)
    
    def _cnot_lower_bound(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> int | None:
        """Calculate lower bound on CNOT count."""
        # For n-qubit fully entangled state, need at least n-1 CNOTs
        if chars.entanglement_density > 0.5:
            return chars.qubit_count - 1
        return None
    
    def _is_classically_hard(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> bool:
        """Determine if circuit is classically hard to simulate."""
        # High entanglement + many non-Clifford gates = classically hard
        if chars.qubit_count < 30:
            return False  # Can still brute-force
        
        if complexity.estimated_non_clifford_count == 0:
            return False  # Clifford is efficient
        
        if chars.entanglement_density < 0.2:
            return False  # MPS may work
        
        return True
    
    def _get_simulation_method(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> str | None:
        """Get recommended classical simulation method if feasible."""
        if complexity.estimated_non_clifford_count == 0:
            return "stabilizer"
        
        if chars.entanglement_density < 0.2:
            return "tensor_network"
        
        if chars.qubit_count <= 30:
            return "state_vector"
        
        if chars.qubit_count <= 40 and complexity.t_count < 50:
            return "t_expansion"
        
        return None


class GateSynthesisAnalyzer:
    """Analyzes gate synthesis requirements for fault-tolerant computation."""
    
    # T-gate costs for common rotations (Solovay-Kitaev)
    ROTATION_T_COSTS = {
        8: 10,   # 8-bit precision
        16: 25,  # 16-bit precision
        32: 55,  # 32-bit precision
        64: 120, # 64-bit precision
    }
    
    # Non-Clifford gates
    NON_CLIFFORD_GATES = {"t", "tdg", "rx", "ry", "rz", "u1", "u2", "u3", "p", "phase"}
    CLIFFORD_GATES = {"h", "s", "sdg", "x", "y", "z", "cx", "cnot", "cz", "swap"}
    
    def __init__(self) -> None:
        self._cache: dict[int, GateSynthesisAnalysis] = {}
    
    def analyze(
        self,
        circuit: Circuit,
        characteristics: CircuitCharacteristics,
        precision_bits: int = 16,
    ) -> GateSynthesisAnalysis:
        """Analyze gate synthesis requirements."""
        cache_key = hash((
            characteristics.qubit_count,
            characteristics.gate_count,
            precision_bits,
        ))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Count gate types
        t_count, clifford_count, rotation_count = 0, 0, 0
        custom_gates: list[str] = []
        t_per_layer: dict[int, int] = {}
        current_layer = 0
        
        for idx, gate in enumerate(circuit.gates):
            name = gate.name.lower()
            
            if name in ("t", "tdg"):
                t_count += 1
                t_per_layer[current_layer] = t_per_layer.get(current_layer, 0) + 1
            elif name in self.CLIFFORD_GATES:
                clifford_count += 1
            elif name in ("rx", "ry", "rz", "u1", "u2", "u3", "p", "phase"):
                rotation_count += 1
            elif name not in ("measure", "barrier", "i", "id"):
                if name not in custom_gates:
                    custom_gates.append(name)
            
            # Approximate layer assignment
            current_layer = idx // max(1, characteristics.qubit_count // 2)
        
        # Calculate T-depth
        t_depth = len([l for l in t_per_layer.values() if l > 0])
        t_gates_per_layer = list(t_per_layer.values())
        
        # Rotation synthesis cost
        synthesis_cost = rotation_count * self.ROTATION_T_COSTS.get(precision_bits, 50)
        
        # Clifford depth
        clifford_depth = max(1, clifford_count // characteristics.qubit_count)
        
        # Fault-tolerant estimates
        total_logical_t = t_count + synthesis_cost
        magic_factories = max(1, total_logical_t // 100)  # Rough estimate
        logical_qubits = characteristics.qubit_count * 2  # With ancillas
        
        # Decomposition needs
        needs_decomp = len(custom_gates) > 0 or rotation_count > 0
        decomp_overhead = 1.0 + 0.5 * len(custom_gates) + 0.2 * (rotation_count > 0)
        
        analysis = GateSynthesisAnalysis(
            t_count=t_count,
            t_depth=t_depth,
            t_gates_per_layer=t_gates_per_layer,
            clifford_count=clifford_count,
            clifford_depth=clifford_depth,
            rotation_count=rotation_count,
            rotation_synthesis_cost=synthesis_cost,
            rotation_precision_bits=precision_bits,
            estimated_logical_t_gates=total_logical_t,
            estimated_magic_state_factories=magic_factories,
            estimated_logical_qubits=logical_qubits,
            needs_decomposition=needs_decomp,
            decomposition_overhead=decomp_overhead,
            custom_gates_to_decompose=custom_gates,
        )
        
        self._cache[cache_key] = analysis
        return analysis


class QuantumVolumeAnalyzer:
    """Analyzes quantum volume and related metrics."""
    
    # Typical gate error rates for classification
    GATE_ERROR_1Q = 0.001  # 0.1%
    GATE_ERROR_2Q = 0.01   # 1%
    
    def __init__(self) -> None:
        self._cache: dict[int, QuantumVolumeEstimate] = {}
    
    def analyze(
        self,
        circuit: Circuit,
        characteristics: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> QuantumVolumeEstimate:
        """Analyze quantum volume and related metrics."""
        cache_key = hash((
            characteristics.qubit_count,
            characteristics.depth,
            complexity.cnot_count,
        ))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Effective dimensions
        effective_qubits = self._count_effective_qubits(circuit, characteristics)
        effective_depth = self._calculate_effective_depth(circuit, characteristics)
        
        # Quantum volume proxy (log2)
        # QV ~ min(n, depth) for square circuits
        log_qv = min(effective_qubits, effective_depth)
        
        # Classify
        if log_qv <= 5:
            qv_class = "trivial"
        elif log_qv <= 10:
            qv_class = "moderate"
        elif log_qv <= 20:
            qv_class = "high"
        else:
            qv_class = "extreme"
        
        # Success probability estimates
        success_prob = self._estimate_success_probability(characteristics, complexity)
        noisy_success = self._estimate_noisy_success(characteristics, complexity)
        
        # Critical depth (where noise dominates)
        critical_depth = self._calculate_critical_depth(characteristics)
        
        # Limiting factors
        coherence_limited = characteristics.depth > critical_depth
        gate_error_limited = (
            complexity.two_qubit_gate_ratio > 0.3 and 
            characteristics.gate_count > 100
        )
        
        estimate = QuantumVolumeEstimate(
            effective_qubits=effective_qubits,
            effective_depth=effective_depth,
            log_quantum_volume=log_qv,
            quantum_volume_class=qv_class,
            estimated_success_probability=success_prob,
            estimated_noisy_success=noisy_success,
            critical_depth=critical_depth,
            coherence_limited=coherence_limited,
            gate_error_limited=gate_error_limited,
        )
        
        self._cache[cache_key] = estimate
        return estimate
    
    def _count_effective_qubits(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> int:
        """Count effectively used qubits."""
        used_qubits: set[int] = set()
        for gate in circuit.gates:
            used_qubits.update(gate.qubits)
        return len(used_qubits)
    
    def _calculate_effective_depth(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> int:
        """Calculate effective circuit depth (non-trivial ops)."""
        # Filter out identity, barrier, measure
        trivial = {"i", "id", "identity", "barrier", "measure", "m"}
        nontrivial_count = sum(
            1 for g in circuit.gates if g.name.lower() not in trivial
        )
        return max(1, nontrivial_count // chars.qubit_count)
    
    def _estimate_success_probability(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> float:
        """Estimate success probability without noise."""
        # Perfect simulation
        return 1.0
    
    def _estimate_noisy_success(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> float:
        """Estimate success probability with typical noise."""
        # Rough error model: each gate has some error probability
        one_q_gates = chars.gate_count - complexity.cnot_count
        two_q_gates = complexity.cnot_count
        
        # Probability of no error
        p_no_error = (
            (1 - self.GATE_ERROR_1Q) ** one_q_gates *
            (1 - self.GATE_ERROR_2Q) ** two_q_gates
        )
        
        return max(0.0, min(1.0, p_no_error))
    
    def _calculate_critical_depth(
        self,
        chars: CircuitCharacteristics,
    ) -> int:
        """Calculate depth where noise starts dominating."""
        # Based on typical T1/T2 times and gate times
        # Very rough: ~1000 gates before decoherence
        return max(100, 1000 // max(1, chars.qubit_count))


class ParallelizabilityAnalyzer:
    """Analyzes parallelization potential of circuits."""
    
    def __init__(self) -> None:
        self._cache: dict[int, ParallelizabilityMetrics] = {}
    
    def analyze(
        self,
        circuit: Circuit,
        characteristics: CircuitCharacteristics,
    ) -> ParallelizabilityMetrics:
        """Analyze parallelization potential."""
        cache_key = hash((
            characteristics.qubit_count,
            characteristics.gate_count,
            characteristics.depth,
        ))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Layer analysis
        layers = self._compute_layers(circuit, characteristics)
        total_layers = len(layers)
        parallelizable = sum(1 for l in layers if len(l) > 1)
        max_gates = max(len(l) for l in layers) if layers else 0
        avg_gates = (
            sum(len(l) for l in layers) / len(layers) if layers else 0
        )
        
        # Parallelism score
        parallelism = parallelizable / max(total_layers, 1)
        
        # Qubit utilization
        utilization = self._calculate_utilization(layers, characteristics)
        
        # Critical path (DAG analysis)
        critical_path = self._find_critical_path(circuit, characteristics)
        
        # DAG metrics
        dag_width = max_gates
        dag_height = critical_path
        dependencies = self._count_dependencies(circuit)
        
        # Speedup estimates
        gpu_speedup = self._estimate_gpu_speedup(characteristics, parallelism)
        dist_speedup = self._estimate_distributed_speedup(characteristics, parallelism)
        amdahl = parallelism
        
        metrics = ParallelizabilityMetrics(
            total_layers=total_layers,
            parallelizable_layers=parallelizable,
            max_gates_per_layer=max_gates,
            average_gates_per_layer=avg_gates,
            circuit_parallelism=parallelism,
            qubit_utilization=utilization,
            critical_path_length=critical_path,
            dag_width=dag_width,
            dag_height=dag_height,
            gate_dependencies=dependencies,
            theoretical_gpu_speedup=gpu_speedup,
            theoretical_distributed_speedup=dist_speedup,
            amdahl_parallel_fraction=amdahl,
        )
        
        self._cache[cache_key] = metrics
        return metrics
    
    def _compute_layers(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> list[list[int]]:
        """Compute gate layers (parallel groups)."""
        if not circuit.gates:
            return []
        
        layers: list[list[int]] = []
        qubit_last_layer: dict[int, int] = {}
        
        for idx, gate in enumerate(circuit.gates):
            # Find earliest layer this gate can go in
            max_prev_layer = -1
            for q in gate.qubits:
                if q in qubit_last_layer:
                    max_prev_layer = max(max_prev_layer, qubit_last_layer[q])
            
            target_layer = max_prev_layer + 1
            
            # Extend layers if needed
            while len(layers) <= target_layer:
                layers.append([])
            
            layers[target_layer].append(idx)
            
            # Update qubit tracking
            for q in gate.qubits:
                qubit_last_layer[q] = target_layer
        
        return layers
    
    def _calculate_utilization(
        self,
        layers: list[list[int]],
        chars: CircuitCharacteristics,
    ) -> float:
        """Calculate average qubit utilization across layers."""
        if not layers:
            return 0.0
        
        # Rough estimate: gates touch 1-2 qubits on average
        total_qubit_ops = chars.gate_count * 1.5  # Avg qubits per gate
        total_slots = len(layers) * chars.qubit_count
        
        return min(1.0, total_qubit_ops / max(total_slots, 1))
    
    def _find_critical_path(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> int:
        """Find critical path length in gate dependency DAG."""
        # Simplified: use depth as approximation
        return chars.depth
    
    def _count_dependencies(self, circuit: Circuit) -> int:
        """Count gate-to-gate dependencies."""
        # Each gate depends on previous gates on same qubits
        dependencies = 0
        qubit_last_gate: dict[int, int] = {}
        
        for idx, gate in enumerate(circuit.gates):
            for q in gate.qubits:
                if q in qubit_last_gate:
                    dependencies += 1
                qubit_last_gate[q] = idx
        
        return dependencies
    
    def _estimate_gpu_speedup(
        self,
        chars: CircuitCharacteristics,
        parallelism: float,
    ) -> float:
        """Estimate theoretical GPU speedup."""
        if chars.qubit_count < 10:
            return 1.0  # Too small for GPU benefit
        
        # GPU excels at parallel ops
        base_speedup = 10.0 * parallelism
        
        # Memory bound for large circuits
        if chars.estimated_memory_mb > 1000:
            base_speedup *= 0.5
        
        return max(1.0, base_speedup)
    
    def _estimate_distributed_speedup(
        self,
        chars: CircuitCharacteristics,
        parallelism: float,
    ) -> float:
        """Estimate theoretical distributed speedup."""
        if chars.qubit_count < 20:
            return 1.0  # Too small for distribution overhead
        
        # Distributed benefits large state vectors
        n = chars.qubit_count
        nodes = min(16, 2 ** (n - 20)) if n > 20 else 1
        
        # Communication overhead
        efficiency = 0.7 * parallelism
        
        return max(1.0, nodes * efficiency)


class SimulationFeasibilityAnalyzer:
    """Analyzes classical simulation feasibility."""
    
    def __init__(self) -> None:
        self._cache: dict[int, ClassicalSimulationFeasibility] = {}
    
    def analyze(
        self,
        circuit: Circuit,
        characteristics: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> ClassicalSimulationFeasibility:
        """Analyze classical simulation feasibility."""
        cache_key = hash((
            characteristics.qubit_count,
            characteristics.gate_count,
            characteristics.depth,
            complexity.estimated_non_clifford_count,
        ))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        n = characteristics.qubit_count
        
        # Determine simulation method
        method, is_efficient = self._determine_method(characteristics, complexity)
        
        # Resource estimates
        time_s = self._estimate_time(characteristics, complexity, method)
        memory_gb = self._estimate_memory(characteristics, method)
        flops = self._estimate_flops(characteristics, complexity)
        
        # Bottleneck analysis
        bottleneck, desc = self._identify_bottleneck(
            characteristics, complexity, time_s, memory_gb
        )
        
        # Tensor network analysis
        can_tn = self._can_use_tensor_network(characteristics, complexity)
        tn_speedup = self._tensor_network_speedup(characteristics, complexity) if can_tn else 1.0
        
        # Stabilizer analysis
        clifford_fraction = (
            complexity.estimated_clifford_count / max(characteristics.gate_count, 1)
        )
        can_stabilizer = clifford_fraction > 0.99
        
        # Recommend simulator
        simulator = self._recommend_simulator(characteristics, complexity, method)
        optimizations = self._get_optimizations(characteristics, complexity, method)
        
        feasibility = ClassicalSimulationFeasibility(
            is_efficiently_simulable=is_efficient,
            simulation_method=method,
            estimated_time_seconds=time_s,
            estimated_memory_gb=memory_gb,
            estimated_flops=flops,
            primary_bottleneck=bottleneck,
            bottleneck_description=desc,
            can_use_tensor_network=can_tn,
            tensor_network_speedup=tn_speedup,
            can_use_stabilizer_simulation=can_stabilizer,
            stabilizer_fraction=clifford_fraction,
            recommended_simulator=simulator,
            simulator_specific_optimizations=optimizations,
        )
        
        self._cache[cache_key] = feasibility
        return feasibility
    
    def _determine_method(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> tuple[str, bool]:
        """Determine best simulation method and if efficient."""
        n = chars.qubit_count
        
        # Pure Clifford -> stabilizer
        if complexity.estimated_non_clifford_count == 0:
            return "stabilizer", True
        
        # Low entanglement -> tensor network
        if chars.entanglement_density < 0.2:
            return "tensor_network", n <= 100
        
        # Small circuits -> state vector
        if n <= 30:
            return "state_vector", True
        
        # Medium circuits -> may still be tractable
        if n <= 40:
            return "state_vector", complexity.t_count < 100
        
        # Large circuits
        return "state_vector", False
    
    def _estimate_time(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        method: str,
    ) -> float:
        """Estimate simulation time in seconds."""
        n = chars.qubit_count
        
        if method == "stabilizer":
            # O(n^2 * gates)
            return n**2 * chars.gate_count * 1e-9
        
        if method == "tensor_network":
            # Depends on bond dimension
            bond_dim = 2 ** int(chars.entanglement_density * 10)
            return chars.gate_count * bond_dim**3 * 1e-9
        
        # State vector
        state_size = 2**n
        ops_per_gate = state_size  # Each gate touches full state
        total_ops = ops_per_gate * chars.gate_count
        
        # Assume ~1 GFLOP/s for complex ops
        return total_ops * 1e-9
    
    def _estimate_memory(
        self,
        chars: CircuitCharacteristics,
        method: str,
    ) -> float:
        """Estimate memory requirement in GB."""
        if method == "stabilizer":
            # O(n^2) for tableau
            return (chars.qubit_count ** 2) * 8 / 1e9
        
        if method == "tensor_network":
            # Depends on bond dimension
            return chars.qubit_count * (2**10) * 16 / 1e9  # Assume chi=1024
        
        # State vector: 2^n complex numbers
        return chars.estimated_memory_mb / 1024
    
    def _estimate_flops(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> float:
        """Estimate total floating point operations."""
        n = chars.qubit_count
        state_size = 2**n
        
        # Each gate is roughly O(2^(k*n/2)) for k-qubit gate
        one_q_flops = 4 * 2**(n-1)  # Multiply 2x2 matrix
        two_q_flops = 16 * 2**(n-2)  # Multiply 4x4 matrix
        
        one_q_count = chars.gate_count - complexity.cnot_count
        two_q_count = complexity.cnot_count
        
        return one_q_count * one_q_flops + two_q_count * two_q_flops
    
    def _identify_bottleneck(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        time_s: float,
        memory_gb: float,
    ) -> tuple[str, str]:
        """Identify primary simulation bottleneck."""
        # Memory check
        available_memory = 16.0  # Assume 16 GB
        if memory_gb > available_memory:
            return "memory", f"Requires {memory_gb:.1f} GB, exceeds typical RAM"
        
        # Time check
        if time_s > 3600:  # More than 1 hour
            return "time", f"Estimated time: {time_s/3600:.1f} hours"
        
        # Entanglement check
        if chars.entanglement_density > 0.7:
            return "entanglement", "High entanglement prevents tensor network methods"
        
        return "none", "Simulation is feasible with available resources"
    
    def _can_use_tensor_network(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> bool:
        """Check if tensor network methods are applicable."""
        return chars.entanglement_density < 0.3
    
    def _tensor_network_speedup(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
    ) -> float:
        """Estimate speedup from tensor network methods."""
        if chars.entanglement_density >= 0.3:
            return 1.0
        
        # Speedup scales with qubit count for low entanglement
        n = chars.qubit_count
        if n <= 20:
            return 1.0
        
        # Exponential speedup for large low-entanglement circuits
        return 2 ** (n - 20)
    
    def _recommend_simulator(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        method: str,
    ) -> str:
        """Recommend the best simulator."""
        if method == "stabilizer":
            return "qiskit_stim"
        
        if method == "tensor_network":
            return "quimb"
        
        # State vector
        if chars.qubit_count <= 20:
            return "numpy"
        if chars.qubit_count <= 30:
            return "qsim"
        
        return "cuquantum"
    
    def _get_optimizations(
        self,
        chars: CircuitCharacteristics,
        complexity: ComplexityMetrics,
        method: str,
    ) -> list[str]:
        """Get simulator-specific optimization hints."""
        opts = []
        
        if method == "state_vector":
            opts.append("Enable gate fusion for adjacent gates")
            if chars.qubit_count > 25:
                opts.append("Use GPU acceleration if available")
            if chars.depth > 100:
                opts.append("Consider checkpointing for memory")
        
        if method == "tensor_network":
            opts.append("Set appropriate bond dimension cutoff")
            opts.append("Use SVD truncation for memory efficiency")
        
        if complexity.t_count > 0:
            opts.append("Consider T-gate optimization passes")
        
        return opts


class EntanglementStructureAnalyzer:
    """Analyzes entanglement structure in circuits."""
    
    def __init__(self) -> None:
        self._cache: dict[int, EntanglementStructure] = {}
    
    def analyze(
        self,
        circuit: Circuit,
        characteristics: CircuitCharacteristics,
    ) -> EntanglementStructure:
        """Analyze entanglement structure."""
        cache_key = hash((
            characteristics.qubit_count,
            characteristics.gate_count,
            characteristics.entanglement_density,
        ))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        n = characteristics.qubit_count
        
        # Build qubit interaction graph
        edges = self._build_qubit_graph(circuit)
        diameter = self._compute_graph_diameter(edges, n)
        is_planar = self._check_planarity(edges, n)
        connectivity = self._determine_connectivity_type(edges, n)
        
        # Entanglement metrics
        schmidt_rank = self._estimate_schmidt_rank(characteristics)
        bond_dim = self._estimate_bond_dimension(characteristics)
        entropy = self._estimate_entanglement_entropy(characteristics)
        
        # Structure patterns
        is_layered = self._is_layered(circuit, characteristics)
        is_localized = self._is_localized(edges, n)
        light_cone = self._compute_light_cone(circuit, n)
        
        structure = EntanglementStructure(
            qubit_graph_edges=len(edges),
            graph_diameter=diameter,
            is_planar=is_planar,
            connectivity_type=connectivity,
            max_schmidt_rank=schmidt_rank,
            estimated_bond_dimension=bond_dim,
            entanglement_entropy_estimate=entropy,
            is_layered=is_layered,
            has_localized_entanglement=is_localized,
            entanglement_light_cone=light_cone,
        )
        
        self._cache[cache_key] = structure
        return structure
    
    def _build_qubit_graph(self, circuit: Circuit) -> set[tuple[int, int]]:
        """Build qubit interaction graph from two-qubit gates."""
        edges: set[tuple[int, int]] = set()
        
        for gate in circuit.gates:
            if len(gate.qubits) >= 2:
                for i in range(len(gate.qubits)):
                    for j in range(i + 1, len(gate.qubits)):
                        q1, q2 = gate.qubits[i], gate.qubits[j]
                        edges.add((min(q1, q2), max(q1, q2)))
        
        return edges
    
    def _compute_graph_diameter(
        self,
        edges: set[tuple[int, int]],
        n: int,
    ) -> int:
        """Compute diameter of qubit graph (max shortest path)."""
        if not edges:
            return 0
        
        # Build adjacency list
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for q1, q2 in edges:
            adj[q1].add(q2)
            adj[q2].add(q1)
        
        # BFS from each node (simplified - just sample a few)
        max_dist = 0
        sample_nodes = list(adj.keys())[:min(5, len(adj))]
        
        for start in sample_nodes:
            visited = {start: 0}
            queue = [start]
            
            while queue:
                node = queue.pop(0)
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited:
                        visited[neighbor] = visited[node] + 1
                        queue.append(neighbor)
                        max_dist = max(max_dist, visited[neighbor])
        
        return max_dist
    
    def _check_planarity(
        self,
        edges: set[tuple[int, int]],
        n: int,
    ) -> bool:
        """Check if qubit graph is planar (simplified heuristic)."""
        # Kuratowski's theorem: planar if |E| <= 3|V| - 6
        if n < 3:
            return True
        return len(edges) <= 3 * n - 6
    
    def _determine_connectivity_type(
        self,
        edges: set[tuple[int, int]],
        n: int,
    ) -> str:
        """Determine the type of qubit connectivity."""
        if not edges:
            return "none"
        
        # Check for all-to-all
        max_edges = n * (n - 1) // 2
        if len(edges) == max_edges:
            return "all-to-all"
        
        # Check for linear
        is_linear = all(abs(q1 - q2) == 1 for q1, q2 in edges)
        if is_linear:
            return "linear"
        
        # Check for ring
        if len(edges) == n and all(
            (i, (i + 1) % n) in edges or ((i + 1) % n, i) in edges
            for i in range(n)
        ):
            return "ring"
        
        return "custom"
    
    def _estimate_schmidt_rank(
        self,
        chars: CircuitCharacteristics,
    ) -> int:
        """Estimate maximum Schmidt rank across bipartitions."""
        # Upper bound: 2^min(n/2, depth)
        n = chars.qubit_count
        return min(2 ** (n // 2), 2 ** chars.depth)
    
    def _estimate_bond_dimension(
        self,
        chars: CircuitCharacteristics,
    ) -> int:
        """Estimate bond dimension for MPS representation."""
        # Based on entanglement
        if chars.entanglement_density < 0.1:
            return 4
        elif chars.entanglement_density < 0.3:
            return 32
        elif chars.entanglement_density < 0.5:
            return 256
        else:
            return 2 ** (chars.qubit_count // 2)
    
    def _estimate_entanglement_entropy(
        self,
        chars: CircuitCharacteristics,
    ) -> float:
        """Estimate average bipartite entanglement entropy."""
        # Rough estimate based on density
        max_entropy = chars.qubit_count / 2  # Max entropy in bits
        return chars.entanglement_density * max_entropy
    
    def _is_layered(
        self,
        circuit: Circuit,
        chars: CircuitCharacteristics,
    ) -> bool:
        """Check if circuit has clear layer structure."""
        # Check if gates can be grouped into clear layers
        return chars.gate_count >= chars.depth  # At least 1 gate per layer
    
    def _is_localized(
        self,
        edges: set[tuple[int, int]],
        n: int,
    ) -> bool:
        """Check if entanglement is spatially localized."""
        if not edges:
            return True
        
        # Check if all edges are between nearby qubits
        max_distance = max(abs(q1 - q2) for q1, q2 in edges)
        return max_distance <= n // 2
    
    def _compute_light_cone(
        self,
        circuit: Circuit,
        n: int,
    ) -> int:
        """Compute size of entanglement light cone."""
        # How many qubits can become entangled from any starting qubit
        if not circuit.gates:
            return 1
        
        # Track reachability
        reachable: set[int] = {0}  # Start from qubit 0
        
        for gate in circuit.gates:
            qubits = set(gate.qubits)
            if qubits & reachable:
                reachable.update(qubits)
        
        return len(reachable)


class AdvancedComplexityAnalyzer:
    """Main analyzer combining all advanced complexity analysis components."""
    
    def __init__(self) -> None:
        self._theoretical = TheoreticalComplexityAnalyzer()
        self._synthesis = GateSynthesisAnalyzer()
        self._quantum_volume = QuantumVolumeAnalyzer()
        self._parallelizability = ParallelizabilityAnalyzer()
        self._simulation = SimulationFeasibilityAnalyzer()
        self._entanglement = EntanglementStructureAnalyzer()
        self._complexity = CircuitComplexityAnalyzer()
        self._pattern_detector = GatePatternDetector()
    
    def analyze(
        self,
        circuit: Circuit,
        precision_bits: int = 16,
    ) -> AdvancedCircuitAnalysis:
        """Perform comprehensive advanced circuit analysis."""
        start_time = time.time()
        
        # Basic analysis
        characteristics = CircuitCharacteristics.from_circuit(circuit)
        complexity = self._complexity.analyze(circuit, characteristics)
        patterns = self._pattern_detector.detect_patterns(circuit, characteristics)
        
        # Advanced analysis
        theoretical = self._theoretical.analyze(
            circuit, characteristics, complexity, patterns
        )
        synthesis = self._synthesis.analyze(circuit, characteristics, precision_bits)
        qv = self._quantum_volume.analyze(circuit, characteristics, complexity)
        parallel = self._parallelizability.analyze(circuit, characteristics)
        simulation = self._simulation.analyze(circuit, characteristics, complexity)
        entanglement = self._entanglement.analyze(circuit, characteristics)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return AdvancedCircuitAnalysis(
            characteristics=characteristics,
            complexity=complexity,
            theoretical_bounds=theoretical,
            synthesis_analysis=synthesis,
            quantum_volume=qv,
            parallelizability=parallel,
            simulation_feasibility=simulation,
            entanglement_structure=entanglement,
            analysis_duration_ms=duration_ms,
        )
    
    def quick_complexity_summary(
        self,
        circuit: Circuit,
    ) -> dict[str, Any]:
        """Get a quick complexity summary without full analysis."""
        chars = CircuitCharacteristics.from_circuit(circuit)
        complexity = self._complexity.analyze(circuit, chars)
        
        # Quick estimates
        n = chars.qubit_count
        is_hard = n > 30 and complexity.estimated_non_clifford_count > 0
        method = "state_vector" if n <= 30 else "tensor_network"
        
        return {
            "qubits": n,
            "depth": chars.depth,
            "gate_count": chars.gate_count,
            "complexity_score": complexity.complexity_score,
            "complexity_class": complexity.complexity_class,
            "is_classically_hard": is_hard,
            "recommended_method": method,
            "estimated_memory_mb": chars.estimated_memory_mb,
            "t_count": complexity.t_count,
            "cnot_count": complexity.cnot_count,
        }
    
    def compare_circuits(
        self,
        circuits: list[Circuit],
        labels: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare complexity of multiple circuits."""
        if labels is None:
            labels = [f"circuit_{i}" for i in range(len(circuits))]
        
        comparisons: dict[str, dict[str, Any]] = {}
        
        for circuit, label in zip(circuits, labels):
            analysis = self.analyze(circuit)
            comparisons[label] = {
                "summary": analysis.get_summary(),
                "parallelism": analysis.parallelizability.circuit_parallelism,
                "is_classically_hard": analysis.theoretical_bounds.is_classically_hard,
                "simulation_time": analysis.simulation_feasibility.estimated_time_seconds,
                "quantum_volume_class": analysis.quantum_volume.quantum_volume_class,
            }
        
        return comparisons


# Convenience function for quick analysis
def analyze_circuit_complexity(
    circuit: Circuit,
    precision_bits: int = 16,
) -> AdvancedCircuitAnalysis:
    """Analyze circuit complexity with all advanced metrics.
    
    Args:
        circuit: The quantum circuit to analyze
        precision_bits: Precision for rotation synthesis estimates
        
    Returns:
        Comprehensive AdvancedCircuitAnalysis
    """
    analyzer = AdvancedComplexityAnalyzer()
    return analyzer.analyze(circuit, precision_bits)


def get_complexity_summary(circuit: Circuit) -> dict[str, Any]:
    """Get a quick complexity summary for a circuit.
    
    Args:
        circuit: The quantum circuit to analyze
        
    Returns:
        Dictionary with key complexity metrics
    """
    analyzer = AdvancedComplexityAnalyzer()
    return analyzer.quick_complexity_summary(circuit)


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
