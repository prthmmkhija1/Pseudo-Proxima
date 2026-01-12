"""Quantum circuit runner implementation.

Converts plans to quantum circuits and executes them using backend adapters.
"""

from __future__ import annotations

from typing import Any

import cirq

from proxima.backends.base import SimulatorType
from proxima.backends.registry import BackendRegistry
from proxima.utils.logging import get_logger

logger = get_logger("runner")


def create_bell_state_circuit() -> cirq.Circuit:
    """Create a 2-qubit Bell state circuit."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),  # Hadamard on qubit 0
        cirq.CNOT(q0, q1),  # CNOT with control=q0, target=q1
        cirq.measure(q0, q1, key="result"),  # Measure both qubits
    )
    return circuit


def create_ghz_state_circuit(num_qubits: int = 3) -> cirq.Circuit:
    """Create a GHZ state circuit."""
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),  # Hadamard on first qubit
    )
    # CNOT cascade
    for i in range(num_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit


def create_teleportation_circuit() -> cirq.Circuit:
    """Create a quantum teleportation circuit.

    Teleports the state of qubit 0 to qubit 2 using qubit 1 as an entangled resource.
    """
    q0, q1, q2 = cirq.LineQubit.range(3)

    circuit = cirq.Circuit(
        # Prepare state to teleport (arbitrary superposition)
        cirq.H(q0),
        cirq.T(q0),
        # Create Bell pair between q1 and q2
        cirq.H(q1),
        cirq.CNOT(q1, q2),
        # Bell measurement on q0 and q1
        cirq.CNOT(q0, q1),
        cirq.H(q0),
        cirq.measure(q0, q1, key="bell_measurement"),
        # Conditional corrections on q2 (simplified for simulation)
        # In real implementation, these would be conditional on measurements
        # For demonstration, we measure the final state
        cirq.measure(q2, key="result"),
    )
    return circuit


def parse_objective(objective: str) -> dict[str, Any]:
    """Parse natural language objective into circuit specification.

    Args:
        objective: Natural language description of the quantum circuit

    Returns:
        Dictionary with circuit_type and parameters
    """
    objective_lower = objective.lower()

    if "bell" in objective_lower:
        return {"circuit_type": "bell", "qubits": 2}
    elif "ghz" in objective_lower:
        # Try to extract qubit count
        import re

        match = re.search(r"(\d+)[-\s]*qubit", objective_lower)
        num_qubits = int(match.group(1)) if match else 3
        return {"circuit_type": "ghz", "qubits": num_qubits}
    elif "teleport" in objective_lower:
        return {"circuit_type": "teleportation", "qubits": 3}
    else:
        # Default to simple bell state
        return {"circuit_type": "bell", "qubits": 2}


def quantum_runner(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute a quantum circuit based on the plan.

    Args:
        plan: Execution plan with objective and configuration

    Returns:
        Execution results including counts and metadata
    """
    logger.info("runner.start", plan=plan)

    # Extract configuration from plan
    objective = plan.get("objective", "demo")
    backend_name = plan.get("backend", "cirq")
    shots = plan.get("shots", 1024)
    timeout_seconds = plan.get("timeout_seconds")

    # Handle auto backend selection
    if backend_name == "auto":
        registry = BackendRegistry()
        registry.discover()
        available = registry.list_available()
        if available:
            backend_name = available[0]  # Use first available backend
            logger.info(
                "runner.auto_backend_selected",
                backend=backend_name,
                available=available,
            )
        else:
            return {
                "status": "error",
                "error": "No backends available",
            }

    # Parse objective to determine circuit type
    circuit_spec = parse_objective(objective)
    logger.info("runner.parsed", circuit_spec=circuit_spec)

    # Create circuit based on type
    if circuit_spec["circuit_type"] == "bell":
        circuit = create_bell_state_circuit()
    elif circuit_spec["circuit_type"] == "ghz":
        circuit = create_ghz_state_circuit(circuit_spec["qubits"])
    elif circuit_spec["circuit_type"] == "teleportation":
        circuit = create_teleportation_circuit()
    else:
        # Default
        circuit = create_bell_state_circuit()

    logger.info("runner.circuit_created", qubits=len(circuit.all_qubits()))

    # Get backend adapter
    registry = BackendRegistry()
    registry.discover()  # Discover available backends
    adapter = registry.get(backend_name)

    if not adapter.is_available():
        return {
            "status": "error",
            "error": f"Backend {backend_name} not available",
        }

    # Execute circuit
    options = {
        "simulator_type": SimulatorType.STATE_VECTOR,
        "shots": shots,
        "repetitions": shots,
    }

    # Add timeout if specified
    if timeout_seconds is not None:
        options["timeout_seconds"] = timeout_seconds

    try:
        result = adapter.execute(circuit, options)
        logger.info(
            "runner.executed", backend=backend_name, time_ms=result.execution_time_ms
        )

        # Format results
        counts = result.data.get("counts", {})

        # Calculate percentages and format output
        total_shots = sum(counts.values())
        formatted_counts = {}
        for state, count in sorted(counts.items(), key=lambda x: -x[1]):
            percentage = (count / total_shots * 100) if total_shots > 0 else 0
            formatted_counts[state] = {
                "count": count,
                "percentage": round(percentage, 2),
            }

        return {
            "status": "success",
            "backend": backend_name,
            "circuit_type": circuit_spec["circuit_type"],
            "qubits": circuit_spec["qubits"],
            "shots": shots,
            "execution_time_ms": result.execution_time_ms,
            "counts": formatted_counts,
            "raw_counts": counts,
        }

    except Exception as e:
        logger.error("runner.failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }
