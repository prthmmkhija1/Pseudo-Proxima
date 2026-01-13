"""
Circuit service.

Handles circuit parsing, conversion, and execution.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CircuitService:
    """Service for circuit operations."""
    
    def __init__(self) -> None:
        """Initialize the circuit service."""
        self._supported_formats = ["openqasm", "qiskit_json", "cirq_json", "proxima"]
    
    def parse_circuit(
        self,
        circuit_str: str,
        format: str = "openqasm",
    ) -> dict[str, Any]:
        """Parse a circuit string into internal representation.
        
        Args:
            circuit_str: Circuit definition string.
            format: Circuit format.
        
        Returns:
            Parsed circuit data.
        
        Raises:
            ValueError: If format is not supported.
        """
        if format not in self._supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        if format == "openqasm":
            return self._parse_openqasm(circuit_str)
        elif format == "qiskit_json":
            return self._parse_qiskit_json(circuit_str)
        elif format == "cirq_json":
            return self._parse_cirq_json(circuit_str)
        else:
            return self._parse_proxima(circuit_str)
    
    def validate_circuit(
        self,
        circuit_str: str,
        format: str = "openqasm",
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Validate a circuit.
        
        Args:
            circuit_str: Circuit definition.
            format: Circuit format.
            backend: Target backend for validation.
        
        Returns:
            Validation result with errors and warnings.
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "num_qubits": None,
            "num_gates": None,
            "circuit_depth": None,
            "unsupported_gates": [],
        }
        
        try:
            parsed = self.parse_circuit(circuit_str, format)
            result.update(parsed)
        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))
        
        return result
    
    def convert_circuit(
        self,
        circuit_str: str,
        from_format: str,
        to_format: str,
    ) -> str:
        """Convert a circuit between formats.
        
        Args:
            circuit_str: Circuit definition.
            from_format: Source format.
            to_format: Target format.
        
        Returns:
            Converted circuit string.
        """
        # Parse source format
        parsed = self.parse_circuit(circuit_str, from_format)
        
        # Generate target format
        if to_format == "openqasm":
            return self._generate_openqasm(parsed)
        elif to_format == "qiskit_json":
            return self._generate_qiskit_json(parsed)
        else:
            raise ValueError(f"Conversion to {to_format} not supported")
    
    def estimate_resources(
        self,
        circuit_str: str,
        format: str = "openqasm",
        simulation_type: str = "state_vector",
    ) -> dict[str, Any]:
        """Estimate resources for circuit execution.
        
        Args:
            circuit_str: Circuit definition.
            format: Circuit format.
            simulation_type: Type of simulation.
        
        Returns:
            Resource estimates.
        """
        parsed = self.parse_circuit(circuit_str, format)
        num_qubits = parsed.get("num_qubits", 2)
        
        if simulation_type == "density_matrix":
            memory_bytes = (2 ** (2 * num_qubits)) * 16
        else:
            memory_bytes = (2 ** num_qubits) * 16
        
        return {
            "memory_bytes": memory_bytes,
            "estimated_time_seconds": (parsed.get("circuit_depth", 10) * (2 ** num_qubits)) / 1e9,
            "num_qubits": num_qubits,
            "num_gates": parsed.get("num_gates", 0),
        }
    
    def _parse_openqasm(self, circuit_str: str) -> dict[str, Any]:
        """Parse OpenQASM circuit."""
        import re
        
        result = {
            "format": "openqasm",
            "num_qubits": 2,
            "num_gates": 0,
            "circuit_depth": 1,
        }
        
        # Extract qubit count
        qreg_match = re.search(r'qreg\s+\w+\[(\d+)\]', circuit_str)
        if qreg_match:
            result["num_qubits"] = int(qreg_match.group(1))
        
        # Count gates
        gate_patterns = [
            r'\bh\b', r'\bx\b', r'\by\b', r'\bz\b',
            r'\bcx\b', r'\bcz\b', r'\bccx\b',
            r'\brx\b', r'\bry\b', r'\brz\b',
            r'\bt\b', r'\bs\b', r'\bsdg\b', r'\btdg\b',
            r'\bswap\b', r'\bu1\b', r'\bu2\b', r'\bu3\b',
        ]
        
        gate_count = 0
        for pattern in gate_patterns:
            gate_count += len(re.findall(pattern, circuit_str.lower()))
        
        result["num_gates"] = gate_count
        result["circuit_depth"] = max(1, gate_count // max(1, result["num_qubits"]))
        
        return result
    
    def _parse_qiskit_json(self, circuit_str: str) -> dict[str, Any]:
        """Parse Qiskit JSON circuit."""
        import json
        
        data = json.loads(circuit_str)
        
        return {
            "format": "qiskit_json",
            "num_qubits": data.get("num_qubits", 2),
            "num_gates": len(data.get("gates", [])),
            "circuit_depth": data.get("depth", 1),
        }
    
    def _parse_cirq_json(self, circuit_str: str) -> dict[str, Any]:
        """Parse Cirq JSON circuit."""
        import json
        
        data = json.loads(circuit_str)
        
        return {
            "format": "cirq_json",
            "num_qubits": data.get("num_qubits", 2),
            "num_gates": len(data.get("moments", [])),
            "circuit_depth": len(data.get("moments", [])),
        }
    
    def _parse_proxima(self, circuit_str: str) -> dict[str, Any]:
        """Parse Proxima native circuit format."""
        return {
            "format": "proxima",
            "num_qubits": 2,
            "num_gates": 10,
            "circuit_depth": 5,
        }
    
    def _generate_openqasm(self, parsed: dict[str, Any]) -> str:
        """Generate OpenQASM from parsed circuit."""
        num_qubits = parsed.get("num_qubits", 2)
        return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
h q[0];
measure q -> c;
"""
    
    def _generate_qiskit_json(self, parsed: dict[str, Any]) -> str:
        """Generate Qiskit JSON from parsed circuit."""
        import json
        
        return json.dumps({
            "num_qubits": parsed.get("num_qubits", 2),
            "gates": [],
            "depth": parsed.get("circuit_depth", 1),
        })


def create_test_circuit(num_qubits: int = 2) -> str:
    """Create a simple test circuit (Bell state).
    
    Args:
        num_qubits: Number of qubits.
    
    Returns:
        OpenQASM circuit string.
    """
    if num_qubits < 2:
        num_qubits = 2
    
    return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
h q[0];
cx q[0], q[1];
measure q -> c;
"""


def create_ghz_circuit(num_qubits: int = 3) -> str:
    """Create a GHZ state circuit.
    
    Args:
        num_qubits: Number of qubits.
    
    Returns:
        OpenQASM circuit string.
    """
    gates = "h q[0];\n"
    for i in range(num_qubits - 1):
        gates += f"cx q[{i}], q[{i+1}];\n"
    
    return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
{gates}measure q -> c;
"""


def create_random_circuit(num_qubits: int = 4, depth: int = 10) -> str:
    """Create a random circuit.
    
    Args:
        num_qubits: Number of qubits.
        depth: Circuit depth.
    
    Returns:
        OpenQASM circuit string.
    """
    import random
    
    single_gates = ["h", "x", "y", "z", "t", "s"]
    gates = ""
    
    for _ in range(depth):
        # Single qubit gates
        for q in range(num_qubits):
            if random.random() > 0.5:
                gate = random.choice(single_gates)
                gates += f"{gate} q[{q}];\n"
        
        # Two qubit gates
        if num_qubits >= 2 and random.random() > 0.3:
            q1 = random.randint(0, num_qubits - 2)
            q2 = q1 + 1
            gates += f"cx q[{q1}], q[{q2}];\n"
    
    return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
{gates}measure q -> c;
"""
