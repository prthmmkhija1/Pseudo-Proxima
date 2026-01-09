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
                # Placeholder plan when no model is provided.
                plan: dict[str, Any] = {"objective": objective, "steps": []}
            else:
                plan = self.plan_fn(objective)

            self.state_machine.plan_complete()
            self.logger.info("planning.complete", plan_summary=list(plan.keys()))
            return plan
        except Exception as exc:  # noqa: BLE001
            self.state_machine.plan_failed()
            self.logger.error("planning.failed", error=str(exc))
            raise
