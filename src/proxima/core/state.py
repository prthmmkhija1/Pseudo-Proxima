"""State machine implementation for execution flow.

Implements the Phase 1 execution lifecycle with explicit states and
transitions. Planning and execution can be driven by LLM-backed planners or
local models; this machine is agnostic and only manages lifecycle and
visibility.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from transitions import Machine

from proxima.utils.logging import get_logger


class ExecutionState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"


TRANSITIONS: list[dict[str, Any]] = [
    {"trigger": "start", "source": ExecutionState.IDLE, "dest": ExecutionState.PLANNING},
    {"trigger": "plan_complete", "source": ExecutionState.PLANNING, "dest": ExecutionState.READY},
    {"trigger": "plan_failed", "source": ExecutionState.PLANNING, "dest": ExecutionState.ERROR},
    {"trigger": "execute", "source": ExecutionState.READY, "dest": ExecutionState.RUNNING},
    {"trigger": "pause", "source": ExecutionState.RUNNING, "dest": ExecutionState.PAUSED},
    {"trigger": "resume", "source": ExecutionState.PAUSED, "dest": ExecutionState.RUNNING},
    {"trigger": "complete", "source": ExecutionState.RUNNING, "dest": ExecutionState.COMPLETED},
    {
        "trigger": "abort",
        "source": [ExecutionState.RUNNING, ExecutionState.PAUSED],
        "dest": ExecutionState.ABORTED,
    },
    {"trigger": "error", "source": ExecutionState.RUNNING, "dest": ExecutionState.ERROR},
    {"trigger": "reset", "source": "*", "dest": ExecutionState.IDLE},
]


class ExecutionStateMachine:
    """Finite state machine for a single execution.

    Parameters
    ----------
    execution_id: Optional[str]
            Identifier to bind into logs for traceability.
    """

    def __init__(self, execution_id: str | None = None):
        self.execution_id = execution_id
        self.logger = get_logger("state").bind(execution_id=execution_id)
        self.history: list[str] = []

        self._machine = Machine(
            model=self,
            states=[state.value for state in ExecutionState],
            transitions=TRANSITIONS,
            initial=ExecutionState.IDLE.value,
            auto_transitions=False,
            ignore_invalid_triggers=True,
            after_state_change=self._record_transition,
            send_event=False,
        )

    # region callbacks
    def _record_transition(self) -> None:
        self.history.append(self.state)
        self.logger.info("state.transition", state=self.state)

    # endregion

    # Convenience accessors
    @property
    def state_enum(self) -> ExecutionState:
        return ExecutionState(self.state)

    def snapshot(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "state": self.state,
            "history": list(self.history),
        }
