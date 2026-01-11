"""Execution planner implementation.

Planner delegates the reasoning step to an injected callable (LLM or local
model). It drives the execution state machine through planning states.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from proxima.core.state import ExecutionStateMachine
from proxima.utils.logging import get_logger

PlanFunction = Callable[[str], dict[str, Any]]


class Planner:
    """LLM-assisted planner that produces an execution plan from an objective."""

    def __init__(self, state_machine: ExecutionStateMachine, plan_fn: PlanFunction | None = None):
        self.state_machine = state_machine
        self.plan_fn = plan_fn
        self.logger = get_logger("planner")

    def plan(self, objective: str) -> dict[str, Any]:
        """Generate a plan for the given objective using the configured model."""

        self.state_machine.start()
        try:
            if not self.plan_fn:
                # Generate a meaningful plan when no model is provided
                plan = self._generate_default_plan(objective)
            else:
                plan = self.plan_fn(objective)

            self.state_machine.plan_complete()
            self.logger.info("planning.complete", plan_summary=list(plan.keys()))
            return plan
        except Exception as exc:  # noqa: BLE001
            self.state_machine.plan_failed()
            self.logger.error("planning.failed", error=str(exc))
            raise

    def _generate_default_plan(self, objective: str) -> dict[str, Any]:
        """Generate a default plan based on objective keywords.

        Analyzes the objective to determine:
        - Circuit type (bell, ghz, teleportation, etc.)
        - Execution mode (single, comparison)
        - Required backends
        - Number of shots
        """
        objective_lower = objective.lower()

        # Determine circuit type from objective
        circuit_type = "bell"  # default
        qubits = 2
        if "ghz" in objective_lower:
            circuit_type = "ghz"
            qubits = 3
            # Try to extract qubit count
            import re
            match = re.search(r"(\d+)[-\s]*qubit", objective_lower)
            if match:
                qubits = int(match.group(1))
        elif "teleport" in objective_lower:
            circuit_type = "teleportation"
            qubits = 3
        elif "superposition" in objective_lower or "hadamard" in objective_lower:
            circuit_type = "superposition"
            qubits = 1
        elif "entangle" in objective_lower:
            circuit_type = "bell"
            qubits = 2

        # Determine execution mode
        execution_mode = "single"
        backends = []
        if "compare" in objective_lower or "comparison" in objective_lower:
            execution_mode = "comparison"
            backends = ["cirq", "qiskit"]
        elif "all backend" in objective_lower:
            execution_mode = "comparison"
            backends = ["cirq", "qiskit", "lret"]

        # Extract shots if mentioned
        shots = 1024
        import re
        shots_match = re.search(r"(\d+)\s*shots?", objective_lower)
        if shots_match:
            shots = int(shots_match.group(1))

        # Build plan steps
        steps = [
            {
                "step": 1,
                "action": "create_circuit",
                "description": f"Create {circuit_type} circuit with {qubits} qubits",
                "parameters": {"circuit_type": circuit_type, "qubits": qubits},
            },
            {
                "step": 2,
                "action": "execute",
                "description": f"Execute circuit with {shots} shots",
                "parameters": {"shots": shots, "backends": backends or ["auto"]},
            },
            {
                "step": 3,
                "action": "collect_results",
                "description": "Collect and normalize results",
                "parameters": {},
            },
        ]

        if execution_mode == "comparison":
            steps.append({
                "step": 4,
                "action": "compare",
                "description": "Compare results across backends",
                "parameters": {"backends": backends},
            })

        return {
            "objective": objective,
            "circuit_type": circuit_type,
            "qubits": qubits,
            "shots": shots,
            "execution_mode": execution_mode,
            "backends": backends,
            "steps": steps,
            "generated_by": "default_planner",
        }
