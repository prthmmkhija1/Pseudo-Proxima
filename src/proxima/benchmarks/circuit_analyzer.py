"""Circuit analysis utilities for benchmarking metadata.

Extracts simple characteristics (qubits, gates, depth, entanglement density, measurements)
from circuit objects to enrich benchmark results.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


class CircuitAnalyzer:
    """Lightweight circuit analyzer to derive metadata for benchmarks."""

    @staticmethod
    def analyze_circuit(circuit: Any) -> Dict[str, Any]:
        """Analyze a circuit object and return characteristics as a dict.

        The analyzer is defensive: it works with objects that expose either
        `.gates`, `.operations`, or are iterable. Missing attributes result in
        zeroed metrics, not failures.
        """

        gates = CircuitAnalyzer._extract_gates(circuit)
        gate_count = len(gates)

        # Qubit count heuristic: max qubit index + 1 if available, else fallback
        qubit_count = CircuitAnalyzer._count_qubits(gates)

        depth = CircuitAnalyzer._estimate_depth(gates)
        gate_types = {g.get("name", "?") for g in gates if isinstance(g, dict)}

        entangling_gates = sum(
            1
            for g in gates
            if isinstance(g, dict)
            and isinstance(g.get("qubits"), Iterable)
            and len(list(g.get("qubits"))) > 1
        )
        entanglement_density = (
            float(entangling_gates) / gate_count if gate_count > 0 else 0.0
        )

        has_measurements = any(
            CircuitAnalyzer._is_measurement_gate(g) for g in gates if isinstance(g, dict)
        )
        has_mid_circuit_measurements = CircuitAnalyzer._has_mid_circuit_measurements(gates)

        return {
            "qubit_count": qubit_count,
            "gate_count": gate_count,
            "depth": depth,
            "gate_types": sorted(gate_types),
            "entangling_gates": entangling_gates,
            "entanglement_density": entanglement_density,
            "has_measurements": has_measurements,
            "has_mid_circuit_measurements": has_mid_circuit_measurements,
        }

    @staticmethod
    def _extract_gates(circuit: Any) -> list[dict[str, Any]]:
        if hasattr(circuit, "gates"):
            raw = getattr(circuit, "gates")
        elif hasattr(circuit, "operations"):
            raw = getattr(circuit, "operations")
        elif isinstance(circuit, Iterable):
            raw = circuit
        else:
            return []

        gates: list[dict[str, Any]] = []
        for g in raw:
            if isinstance(g, dict):
                gates.append(g)
            else:
                # Attempt to adapt simple gate objects
                name = getattr(g, "name", None)
                if name is None:
                    name = type(g).__name__
                qubits = getattr(g, "qubits", getattr(g, "targets", None))
                gates.append({"name": str(name), "qubits": qubits})
        return gates

    @staticmethod
    def _count_qubits(gates: list[dict[str, Any]]) -> int:
        qubit_indices = []
        for g in gates:
            qubits = g.get("qubits")
            if qubits is None:
                continue
            try:
                for q in qubits:
                    qubit_indices.append(int(q))
            except Exception:
                continue
        return (max(qubit_indices) + 1) if qubit_indices else 0

    @staticmethod
    def _estimate_depth(gates: list[dict[str, Any]]) -> int:
        """Estimate circuit depth using qubit occupation tracking.
        
        Depth is the longest path from input to output. We track when each
        qubit becomes available (after its last gate completes) to approximate
        the critical path length.
        """
        if not gates:
            return 0
        
        # Track the "time step" at which each qubit becomes available
        qubit_availability: dict[int, int] = {}
        
        for gate in gates:
            qubits = gate.get("qubits")
            if qubits is None:
                continue
            
            try:
                qubit_list = [int(q) for q in qubits]
            except (TypeError, ValueError):
                continue
            
            if not qubit_list:
                continue
            
            # Gate starts after all its qubits are available
            start_time = max((qubit_availability.get(q, 0) for q in qubit_list), default=0)
            finish_time = start_time + 1
            
            # Update availability for all qubits used by this gate
            for q in qubit_list:
                qubit_availability[q] = finish_time
        
        return max(qubit_availability.values(), default=0)

    @staticmethod
    def _is_measurement_gate(gate: dict[str, Any]) -> bool:
        name = str(gate.get("name", "")).lower()
        return name in {"measure", "m", "measurement"}

    @staticmethod
    def _has_mid_circuit_measurements(gates: list[dict[str, Any]]) -> bool:
        # Detect any measurement not at the end (simple heuristic)
        measurement_indices = [i for i, g in enumerate(gates) if CircuitAnalyzer._is_measurement_gate(g)]
        if not measurement_indices:
            return False
        last_gate_idx = len(gates) - 1
        return any(idx < last_gate_idx for idx in measurement_indices)
