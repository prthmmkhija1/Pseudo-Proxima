"""Task executor implementation.

Executor drives the state machine through execution states and delegates actual
work to an injected callable (which can internally use local or remote LLMs).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from proxima.core.state import ExecutionStateMachine
from proxima.utils.logging import get_logger


ExecuteFunction = Callable[[Dict[str, Any]], Any]


class Executor:
    """Executes a prepared plan using a provided runner callable."""

    def __init__(
        self, state_machine: ExecutionStateMachine, runner: Optional[ExecuteFunction] = None
    ):
        self.state_machine = state_machine
        self.runner = runner
        self.logger = get_logger("executor")

    def run(self, plan: Dict[str, Any]) -> Any:
        """Execute the plan; transitions the FSM accordingly."""

        self.state_machine.execute()
        try:
            if not self.runner:
                result: Any = {"status": "skipped", "reason": "no runner configured"}
            else:
                result = self.runner(plan)

            self.state_machine.complete()
            self.logger.info("execution.complete")
            return result
        except Exception as exc:  # noqa: BLE001
            self.state_machine.error()
            self.logger.error("execution.failed", error=str(exc))
            raise
