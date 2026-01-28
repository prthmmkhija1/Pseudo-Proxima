"""Test Circuit Library.

Provides a collection of standard quantum circuits for testing backends.
Includes circuits for validation, performance, and correctness testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import math
import random


class CircuitType(Enum):
    """Types of test circuits available."""
    BELL = "bell"
    GHZ = "ghz"
    QFT = "qft"
    RANDOM = "random"
    SINGLE_QUBIT = "single_qubit"
    TWO_QUBIT = "two_qubit"
    PARAMETRIC = "parametric"
    GROVER = "grover"
    VQE = "vqe"


@dataclass
class GateOperation:
    """Represents a quantum gate operation."""
    name: str
    qubits: List[int]
    parameters: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "qubits": self.qubits,
        }
        if self.parameters:
            result["parameters"] = self.parameters
        return result


@dataclass
class TestCircuit:
    """A quantum circuit for testing.
    
    Provides multiple representations of a circuit for testing
    different backend input formats.
    """
    name: str
    description: str
    num_qubits: int
    num_classical_bits: int
    operations: List[GateOperation]
    expected_results: Optional[Dict[str, float]] = None
    circuit_type: CircuitType = CircuitType.BELL
    
    @property
    def depth(self) -> int:
        """Calculate circuit depth (simplified)."""
        # Simplified depth calculation
        return len(self.operations)
    
    @property
    def gate_count(self) -> int:
        """Count total gates."""
        return len(self.operations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "num_qubits": self.num_qubits,
            "num_classical_bits": self.num_classical_bits,
            "operations": [op.to_dict() for op in self.operations],
            "expected_results": self.expected_results,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def to_qasm(self) -> str:
        """Convert to OpenQASM 2.0 format."""
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self.num_qubits}];",
            f"creg c[{self.num_classical_bits}];",
            "",
        ]
        
        for op in self.operations:
            qasm_gate = self._gate_to_qasm(op)
            if qasm_gate:
                lines.append(qasm_gate)
        
        return "\n".join(lines)
    
    def _gate_to_qasm(self, op: GateOperation) -> Optional[str]:
        """Convert a gate operation to QASM format."""
        name = op.name.lower()
        qubits = op.qubits
        params = op.parameters
        
        # Single qubit gates
        if name in ["h", "x", "y", "z", "s", "t", "sdg", "tdg"]:
            return f"{name} q[{qubits[0]}];"
        
        # Rotation gates
        elif name in ["rx", "ry", "rz"]:
            return f"{name}({params[0]}) q[{qubits[0]}];"
        
        # Two qubit gates
        elif name in ["cx", "cnot"]:
            return f"cx q[{qubits[0]}],q[{qubits[1]}];"
        
        elif name in ["cz"]:
            return f"cz q[{qubits[0]}],q[{qubits[1]}];"
        
        elif name in ["swap"]:
            return f"swap q[{qubits[0]}],q[{qubits[1]}];"
        
        # Controlled rotation
        elif name in ["crx", "cry", "crz"]:
            return f"{name}({params[0]}) q[{qubits[0]}],q[{qubits[1]}];"
        
        # Controlled phase
        elif name in ["cp", "cphase"]:
            return f"cp({params[0]}) q[{qubits[0]}],q[{qubits[1]}];"
        
        # Three qubit gates
        elif name in ["ccx", "toffoli"]:
            return f"ccx q[{qubits[0]}],q[{qubits[1]}],q[{qubits[2]}];"
        
        elif name in ["cswap", "fredkin"]:
            return f"cswap q[{qubits[0]}],q[{qubits[1]}],q[{qubits[2]}];"
        
        # Measure
        elif name == "measure":
            if len(qubits) == 1:
                return f"measure q[{qubits[0]}] -> c[{qubits[0]}];"
            else:
                return None
        
        elif name == "measure_all":
            return "\n".join(
                f"measure q[{i}] -> c[{i}];"
                for i in range(self.num_qubits)
            )
        
        # Barrier
        elif name == "barrier":
            qubit_list = ",".join(f"q[{q}]" for q in qubits)
            return f"barrier {qubit_list};"
        
        return None


class TestCircuitLibrary:
    """Library of standard test circuits.
    
    Provides pre-built circuits for testing backends with known
    expected outcomes for validation.
    """
    
    def __init__(self):
        """Initialize the circuit library."""
        self._circuits: Dict[str, TestCircuit] = {}
        self._load_standard_circuits()
    
    def _load_standard_circuits(self) -> None:
        """Load all standard test circuits."""
        # Bell State
        self._circuits["bell"] = TestCircuit(
            name="Bell State",
            description="Creates a maximally entangled Bell state |Φ+⟩",
            num_qubits=2,
            num_classical_bits=2,
            circuit_type=CircuitType.BELL,
            operations=[
                GateOperation("h", [0]),
                GateOperation("cx", [0, 1]),
                GateOperation("measure_all", list(range(2))),
            ],
            expected_results={"00": 0.5, "11": 0.5},
        )
        
        # GHZ State (3 qubits)
        self._circuits["ghz_3"] = TestCircuit(
            name="GHZ State (3 qubits)",
            description="Creates a 3-qubit GHZ state",
            num_qubits=3,
            num_classical_bits=3,
            circuit_type=CircuitType.GHZ,
            operations=[
                GateOperation("h", [0]),
                GateOperation("cx", [0, 1]),
                GateOperation("cx", [1, 2]),
                GateOperation("measure_all", list(range(3))),
            ],
            expected_results={"000": 0.5, "111": 0.5},
        )
        
        # GHZ State (5 qubits)
        self._circuits["ghz_5"] = TestCircuit(
            name="GHZ State (5 qubits)",
            description="Creates a 5-qubit GHZ state",
            num_qubits=5,
            num_classical_bits=5,
            circuit_type=CircuitType.GHZ,
            operations=[
                GateOperation("h", [0]),
                GateOperation("cx", [0, 1]),
                GateOperation("cx", [1, 2]),
                GateOperation("cx", [2, 3]),
                GateOperation("cx", [3, 4]),
                GateOperation("measure_all", list(range(5))),
            ],
            expected_results={"00000": 0.5, "11111": 0.5},
        )
        
        # Quantum Fourier Transform (3 qubits)
        self._circuits["qft_3"] = TestCircuit(
            name="QFT (3 qubits)",
            description="3-qubit Quantum Fourier Transform",
            num_qubits=3,
            num_classical_bits=3,
            circuit_type=CircuitType.QFT,
            operations=[
                GateOperation("h", [0]),
                GateOperation("cp", [0, 1], [math.pi / 2]),
                GateOperation("cp", [0, 2], [math.pi / 4]),
                GateOperation("h", [1]),
                GateOperation("cp", [1, 2], [math.pi / 2]),
                GateOperation("h", [2]),
                GateOperation("swap", [0, 2]),
                GateOperation("measure_all", list(range(3))),
            ],
            expected_results=None,  # QFT has uniform distribution
        )
        
        # Single qubit test
        self._circuits["single_qubit"] = TestCircuit(
            name="Single Qubit Gates",
            description="Tests all single-qubit gates",
            num_qubits=1,
            num_classical_bits=1,
            circuit_type=CircuitType.SINGLE_QUBIT,
            operations=[
                GateOperation("h", [0]),
                GateOperation("x", [0]),
                GateOperation("y", [0]),
                GateOperation("z", [0]),
                GateOperation("s", [0]),
                GateOperation("t", [0]),
                GateOperation("h", [0]),
                GateOperation("measure", [0]),
            ],
            expected_results=None,
        )
        
        # Two qubit gates test
        self._circuits["two_qubit"] = TestCircuit(
            name="Two Qubit Gates",
            description="Tests common two-qubit gates",
            num_qubits=2,
            num_classical_bits=2,
            circuit_type=CircuitType.TWO_QUBIT,
            operations=[
                GateOperation("h", [0]),
                GateOperation("cx", [0, 1]),
                GateOperation("swap", [0, 1]),
                GateOperation("cz", [0, 1]),
                GateOperation("measure_all", list(range(2))),
            ],
            expected_results=None,
        )
        
        # Parametric circuit
        self._circuits["parametric"] = TestCircuit(
            name="Parametric Circuit",
            description="Tests parameterized rotation gates",
            num_qubits=2,
            num_classical_bits=2,
            circuit_type=CircuitType.PARAMETRIC,
            operations=[
                GateOperation("rx", [0], [math.pi / 4]),
                GateOperation("ry", [1], [math.pi / 3]),
                GateOperation("rz", [0], [math.pi / 6]),
                GateOperation("cx", [0, 1]),
                GateOperation("crx", [0, 1], [math.pi / 2]),
                GateOperation("measure_all", list(range(2))),
            ],
            expected_results=None,
        )
        
        # X gate test (|0⟩ → |1⟩)
        self._circuits["x_gate"] = TestCircuit(
            name="X Gate Test",
            description="Applies X gate to flip |0⟩ to |1⟩",
            num_qubits=1,
            num_classical_bits=1,
            circuit_type=CircuitType.SINGLE_QUBIT,
            operations=[
                GateOperation("x", [0]),
                GateOperation("measure", [0]),
            ],
            expected_results={"1": 1.0},
        )
        
        # Hadamard test
        self._circuits["hadamard"] = TestCircuit(
            name="Hadamard Test",
            description="Applies H gate for equal superposition",
            num_qubits=1,
            num_classical_bits=1,
            circuit_type=CircuitType.SINGLE_QUBIT,
            operations=[
                GateOperation("h", [0]),
                GateOperation("measure", [0]),
            ],
            expected_results={"0": 0.5, "1": 0.5},
        )
    
    def get_circuit(self, name: str) -> Optional[TestCircuit]:
        """Get a circuit by name.
        
        Args:
            name: Circuit identifier
            
        Returns:
            TestCircuit or None if not found
        """
        return self._circuits.get(name.lower())
    
    def get_circuit_by_type(self, circuit_type: CircuitType) -> List[TestCircuit]:
        """Get all circuits of a specific type.
        
        Args:
            circuit_type: Type of circuits to retrieve
            
        Returns:
            List of matching circuits
        """
        return [
            c for c in self._circuits.values()
            if c.circuit_type == circuit_type
        ]
    
    def list_circuits(self) -> List[Tuple[str, str]]:
        """List all available circuits.
        
        Returns:
            List of (name, description) tuples
        """
        return [
            (name, circuit.description)
            for name, circuit in self._circuits.items()
        ]
    
    def create_random_circuit(
        self,
        num_qubits: int = 2,
        depth: int = 10,
        seed: Optional[int] = None
    ) -> TestCircuit:
        """Create a random circuit for testing.
        
        Args:
            num_qubits: Number of qubits
            depth: Approximate circuit depth
            seed: Random seed for reproducibility
            
        Returns:
            Random test circuit
        """
        if seed is not None:
            random.seed(seed)
        
        single_qubit_gates = ["h", "x", "y", "z", "s", "t"]
        two_qubit_gates = ["cx", "cz", "swap"]
        rotation_gates = ["rx", "ry", "rz"]
        
        operations = []
        
        for _ in range(depth):
            gate_type = random.choice(["single", "two", "rotation"])
            
            if gate_type == "single":
                gate = random.choice(single_qubit_gates)
                qubit = random.randint(0, num_qubits - 1)
                operations.append(GateOperation(gate, [qubit]))
            
            elif gate_type == "two" and num_qubits >= 2:
                gate = random.choice(two_qubit_gates)
                q1 = random.randint(0, num_qubits - 1)
                q2 = random.randint(0, num_qubits - 1)
                while q2 == q1:
                    q2 = random.randint(0, num_qubits - 1)
                operations.append(GateOperation(gate, [q1, q2]))
            
            elif gate_type == "rotation":
                gate = random.choice(rotation_gates)
                qubit = random.randint(0, num_qubits - 1)
                angle = random.uniform(0, 2 * math.pi)
                operations.append(GateOperation(gate, [qubit], [angle]))
        
        # Add measurement
        operations.append(GateOperation("measure_all", list(range(num_qubits))))
        
        return TestCircuit(
            name=f"Random Circuit (n={num_qubits}, d={depth})",
            description=f"Randomly generated circuit with {num_qubits} qubits",
            num_qubits=num_qubits,
            num_classical_bits=num_qubits,
            circuit_type=CircuitType.RANDOM,
            operations=operations,
            expected_results=None,
        )
    
    def get_validation_suite(self) -> List[TestCircuit]:
        """Get a suite of circuits for validation testing.
        
        Returns:
            List of circuits for comprehensive validation
        """
        return [
            self._circuits["bell"],
            self._circuits["ghz_3"],
            self._circuits["single_qubit"],
            self._circuits["two_qubit"],
            self._circuits["x_gate"],
            self._circuits["hadamard"],
        ]
    
    def get_performance_suite(
        self,
        max_qubits: int = 10
    ) -> List[TestCircuit]:
        """Get a suite of circuits for performance testing.
        
        Args:
            max_qubits: Maximum number of qubits to test
            
        Returns:
            List of circuits for performance testing
        """
        circuits = []
        
        # Add progressively larger GHZ states
        for n in range(3, min(max_qubits + 1, 12)):
            ops = [GateOperation("h", [0])]
            for i in range(n - 1):
                ops.append(GateOperation("cx", [i, i + 1]))
            ops.append(GateOperation("measure_all", list(range(n))))
            
            circuits.append(TestCircuit(
                name=f"GHZ-{n}",
                description=f"{n}-qubit GHZ state",
                num_qubits=n,
                num_classical_bits=n,
                circuit_type=CircuitType.GHZ,
                operations=ops,
                expected_results=None,
            ))
        
        # Add random circuits of increasing size
        for n in range(2, min(max_qubits + 1, 8)):
            circuits.append(self.create_random_circuit(
                num_qubits=n,
                depth=n * 5,
                seed=42 + n,
            ))
        
        return circuits
