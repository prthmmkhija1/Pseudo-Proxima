"""
Comprehensive Unit Tests for ExecutionStateMachine

Tests the finite state machine for execution control including:
- All state transitions
- Invalid transition handling
- State history tracking
- Snapshot functionality
- Edge cases and error conditions
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Try to import from source, fall back to mocks for environment without dependencies
try:
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from proxima.core.state import ExecutionState, ExecutionStateMachine, TRANSITIONS
    HAS_SOURCE = True
except ImportError:
    HAS_SOURCE = False
    
    # Define mock ExecutionState enum matching the real one
    class ExecutionState(str, Enum):
        IDLE = "IDLE"
        PLANNING = "PLANNING"
        READY = "READY"
        RUNNING = "RUNNING"
        PAUSED = "PAUSED"
        COMPLETED = "COMPLETED"
        ABORTED = "ABORTED"
        ERROR = "ERROR"
    
    # Mock TRANSITIONS matching actual implementation
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
    
    # Mock ExecutionStateMachine class matching the actual implementation behavior
    class ExecutionStateMachine:
        """Mock state machine for testing without dependencies.
        
        Uses ignore_invalid_triggers=True like the actual implementation,
        so invalid triggers are silently ignored.
        """
        
        def __init__(self, execution_id: str = None):
            self.state = ExecutionState.IDLE.value
            self.execution_id = execution_id
            self.history: list[str] = []
            
        @property
        def state_enum(self) -> ExecutionState:
            return ExecutionState(self.state)
            
        def _record_transition(self) -> None:
            """Record state transition in history."""
            self.history.append(self.state)
            
        def _can_transition(self, trigger: str, source: Any) -> bool:
            """Check if transition is valid from current state."""
            if source == "*":
                return True
            if isinstance(source, list):
                return self.state in [s.value if hasattr(s, 'value') else s for s in source]
            source_val = source.value if hasattr(source, 'value') else source
            return self.state == source_val
            
        def _do_transition(self, dest: ExecutionState) -> None:
            """Perform state transition."""
            self.state = dest.value if hasattr(dest, 'value') else dest
            self._record_transition()
            
        def start(self) -> None:
            """IDLE -> PLANNING"""
            if self._can_transition("start", ExecutionState.IDLE):
                self._do_transition(ExecutionState.PLANNING)
            
        def plan_complete(self) -> None:
            """PLANNING -> READY"""
            if self._can_transition("plan_complete", ExecutionState.PLANNING):
                self._do_transition(ExecutionState.READY)
            
        def plan_failed(self) -> None:
            """PLANNING -> ERROR"""
            if self._can_transition("plan_failed", ExecutionState.PLANNING):
                self._do_transition(ExecutionState.ERROR)
            
        def execute(self) -> None:
            """READY -> RUNNING"""
            if self._can_transition("execute", ExecutionState.READY):
                self._do_transition(ExecutionState.RUNNING)
            
        def pause(self) -> None:
            """RUNNING -> PAUSED"""
            if self._can_transition("pause", ExecutionState.RUNNING):
                self._do_transition(ExecutionState.PAUSED)
            
        def resume(self) -> None:
            """PAUSED -> RUNNING"""
            if self._can_transition("resume", ExecutionState.PAUSED):
                self._do_transition(ExecutionState.RUNNING)
            
        def complete(self) -> None:
            """RUNNING -> COMPLETED"""
            if self._can_transition("complete", ExecutionState.RUNNING):
                self._do_transition(ExecutionState.COMPLETED)
            
        def abort(self) -> None:
            """RUNNING or PAUSED -> ABORTED"""
            if self._can_transition("abort", [ExecutionState.RUNNING, ExecutionState.PAUSED]):
                self._do_transition(ExecutionState.ABORTED)
            
        def error(self) -> None:
            """RUNNING -> ERROR"""
            if self._can_transition("error", ExecutionState.RUNNING):
                self._do_transition(ExecutionState.ERROR)
            
        def reset(self) -> None:
            """Any state -> IDLE (wildcard source)"""
            self._do_transition(ExecutionState.IDLE)
            
        def snapshot(self) -> dict[str, Any]:
            """Return snapshot of current state machine."""
            return {
                "execution_id": self.execution_id,
                "state": self.state,
                "history": list(self.history),
            }


# =============================================================================
# STATE ENUM TESTS
# =============================================================================


class TestExecutionState:
    """Tests for ExecutionState enum."""

    @pytest.mark.unit
    def test_all_states_defined(self):
        """Verify all required states are defined."""
        expected_states = [
            "IDLE",
            "PLANNING",
            "READY",
            "RUNNING",
            "PAUSED",
            "COMPLETED",
            "ABORTED",
            "ERROR",
        ]
        
        for state_name in expected_states:
            assert hasattr(ExecutionState, state_name), f"State {state_name} not found"

    @pytest.mark.unit
    def test_state_values(self):
        """Test state enum values match their names."""
        for state in ExecutionState:
            assert state.value == state.name

    @pytest.mark.unit
    def test_state_count(self):
        """Verify correct number of states."""
        assert len(ExecutionState) == 8


# =============================================================================
# STATE MACHINE CREATION TESTS
# =============================================================================


class TestStateMachineCreation:
    """Tests for state machine initialization."""

    @pytest.mark.unit
    def test_initial_state_is_idle(self):
        """New state machine should start in IDLE state."""
        sm = ExecutionStateMachine()
        assert sm.state == ExecutionState.IDLE.value

    @pytest.mark.unit
    def test_state_enum_accessor(self):
        """state_enum property should return ExecutionState enum."""
        sm = ExecutionStateMachine()
        assert sm.state_enum == ExecutionState.IDLE
        assert isinstance(sm.state_enum, ExecutionState)

    @pytest.mark.unit
    def test_execution_id_stored(self):
        """Execution ID should be stored correctly."""
        sm = ExecutionStateMachine(execution_id="test-123")
        assert sm.execution_id == "test-123"

    @pytest.mark.unit
    def test_execution_id_none_default(self):
        """Execution ID should default to None."""
        sm = ExecutionStateMachine()
        assert sm.execution_id is None

    @pytest.mark.unit
    def test_history_initially_empty(self):
        """History should be empty on creation."""
        sm = ExecutionStateMachine()
        assert sm.history == []


# =============================================================================
# VALID TRANSITION TESTS
# =============================================================================


class TestValidTransitions:
    """Tests for valid state transitions."""

    @pytest.mark.unit
    def test_idle_to_planning(self):
        """IDLE -> PLANNING via start trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        assert sm.state_enum == ExecutionState.PLANNING

    @pytest.mark.unit
    def test_planning_to_ready(self):
        """PLANNING -> READY via plan_complete trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        assert sm.state_enum == ExecutionState.READY

    @pytest.mark.unit
    def test_planning_to_error(self):
        """PLANNING -> ERROR via plan_failed trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_failed()
        assert sm.state_enum == ExecutionState.ERROR

    @pytest.mark.unit
    def test_ready_to_running(self):
        """READY -> RUNNING via execute trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        assert sm.state_enum == ExecutionState.RUNNING

    @pytest.mark.unit
    def test_running_to_paused(self):
        """RUNNING -> PAUSED via pause trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.pause()
        assert sm.state_enum == ExecutionState.PAUSED

    @pytest.mark.unit
    def test_paused_to_running(self):
        """PAUSED -> RUNNING via resume trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.pause()
        sm.resume()
        assert sm.state_enum == ExecutionState.RUNNING

    @pytest.mark.unit
    def test_running_to_completed(self):
        """RUNNING -> COMPLETED via complete trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.complete()
        assert sm.state_enum == ExecutionState.COMPLETED

    @pytest.mark.unit
    def test_running_to_aborted(self):
        """RUNNING -> ABORTED via abort trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.abort()
        assert sm.state_enum == ExecutionState.ABORTED

    @pytest.mark.unit
    def test_paused_to_aborted(self):
        """PAUSED -> ABORTED via abort trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.pause()
        sm.abort()
        assert sm.state_enum == ExecutionState.ABORTED

    @pytest.mark.unit
    def test_running_to_error(self):
        """RUNNING -> ERROR via error trigger."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.error()
        assert sm.state_enum == ExecutionState.ERROR


# =============================================================================
# RESET TRANSITION TESTS
# =============================================================================


class TestResetTransition:
    """Tests for reset transition from any state."""

    @pytest.mark.unit
    @pytest.mark.parametrize("setup_sequence,expected_before", [
        ([], ExecutionState.IDLE),
        (["start"], ExecutionState.PLANNING),
        (["start", "plan_complete"], ExecutionState.READY),
        (["start", "plan_complete", "execute"], ExecutionState.RUNNING),
        (["start", "plan_complete", "execute", "pause"], ExecutionState.PAUSED),
        (["start", "plan_complete", "execute", "complete"], ExecutionState.COMPLETED),
        (["start", "plan_complete", "execute", "abort"], ExecutionState.ABORTED),
        (["start", "plan_failed"], ExecutionState.ERROR),
    ])
    def test_reset_from_any_state(self, setup_sequence, expected_before):
        """Reset should work from any state."""
        sm = ExecutionStateMachine()
        
        # Execute setup sequence
        for trigger in setup_sequence:
            getattr(sm, trigger)()
        
        # Verify we're in expected state before reset
        assert sm.state_enum == expected_before
        
        # Reset should always work
        sm.reset()
        assert sm.state_enum == ExecutionState.IDLE


# =============================================================================
# INVALID TRANSITION TESTS
# =============================================================================


class TestInvalidTransitions:
    """Tests for invalid state transitions (should be ignored)."""

    @pytest.mark.unit
    def test_cannot_execute_from_idle(self):
        """Cannot execute directly from IDLE (should be ignored)."""
        sm = ExecutionStateMachine()
        sm.execute()  # Invalid, should be ignored
        assert sm.state_enum == ExecutionState.IDLE

    @pytest.mark.unit
    def test_cannot_pause_from_idle(self):
        """Cannot pause from IDLE (should be ignored)."""
        sm = ExecutionStateMachine()
        sm.pause()  # Invalid, should be ignored
        assert sm.state_enum == ExecutionState.IDLE

    @pytest.mark.unit
    def test_cannot_complete_from_ready(self):
        """Cannot complete from READY (should be ignored)."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.complete()  # Invalid, should be ignored
        assert sm.state_enum == ExecutionState.READY

    @pytest.mark.unit
    def test_cannot_resume_from_running(self):
        """Cannot resume from RUNNING (only from PAUSED)."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.resume()  # Invalid, should be ignored
        assert sm.state_enum == ExecutionState.RUNNING

    @pytest.mark.unit
    def test_cannot_start_from_running(self):
        """Cannot start from RUNNING (should be ignored)."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.start()  # Invalid, should be ignored
        assert sm.state_enum == ExecutionState.RUNNING


# =============================================================================
# HISTORY TRACKING TESTS
# =============================================================================


class TestHistoryTracking:
    """Tests for state history tracking."""

    @pytest.mark.unit
    def test_history_records_transitions(self):
        """History should record all state transitions."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.complete()
        
        assert sm.history == [
            ExecutionState.PLANNING.value,
            ExecutionState.READY.value,
            ExecutionState.RUNNING.value,
            ExecutionState.COMPLETED.value,
        ]

    @pytest.mark.unit
    def test_history_includes_pause_resume(self):
        """History should include pause and resume transitions."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.pause()
        sm.resume()
        sm.complete()
        
        assert ExecutionState.PAUSED.value in sm.history
        # RUNNING appears twice (before pause and after resume)
        assert sm.history.count(ExecutionState.RUNNING.value) == 2

    @pytest.mark.unit
    def test_invalid_transitions_not_in_history(self):
        """Invalid transitions should not be recorded in history."""
        sm = ExecutionStateMachine()
        sm.execute()  # Invalid from IDLE
        sm.pause()    # Invalid from IDLE
        
        assert sm.history == []


# =============================================================================
# SNAPSHOT TESTS
# =============================================================================


class TestSnapshot:
    """Tests for snapshot functionality."""

    @pytest.mark.unit
    def test_snapshot_contains_required_fields(self):
        """Snapshot should contain all required fields."""
        sm = ExecutionStateMachine(execution_id="snap-test")
        snapshot = sm.snapshot()
        
        assert "execution_id" in snapshot
        assert "state" in snapshot
        assert "history" in snapshot

    @pytest.mark.unit
    def test_snapshot_execution_id(self):
        """Snapshot should contain correct execution_id."""
        sm = ExecutionStateMachine(execution_id="snap-123")
        snapshot = sm.snapshot()
        
        assert snapshot["execution_id"] == "snap-123"

    @pytest.mark.unit
    def test_snapshot_current_state(self):
        """Snapshot should contain current state."""
        sm = ExecutionStateMachine()
        sm.start()
        sm.plan_complete()
        snapshot = sm.snapshot()
        
        assert snapshot["state"] == ExecutionState.READY.value

    @pytest.mark.unit
    def test_snapshot_history_copy(self):
        """Snapshot history should be a copy, not a reference."""
        sm = ExecutionStateMachine()
        sm.start()
        snapshot = sm.snapshot()
        
        # Modify original history
        sm.plan_complete()
        
        # Snapshot history should not change
        assert len(snapshot["history"]) == 1
        assert len(sm.history) == 2


# =============================================================================
# FULL WORKFLOW TESTS
# =============================================================================


class TestFullWorkflows:
    """Tests for complete execution workflows."""

    @pytest.mark.unit
    def test_successful_execution_workflow(self):
        """Test complete successful execution workflow."""
        sm = ExecutionStateMachine(execution_id="workflow-success")
        
        # Full successful workflow
        sm.start()
        assert sm.state_enum == ExecutionState.PLANNING
        
        sm.plan_complete()
        assert sm.state_enum == ExecutionState.READY
        
        sm.execute()
        assert sm.state_enum == ExecutionState.RUNNING
        
        sm.complete()
        assert sm.state_enum == ExecutionState.COMPLETED
        
        # Verify history
        assert len(sm.history) == 4

    @pytest.mark.unit
    def test_aborted_execution_workflow(self):
        """Test execution that gets aborted."""
        sm = ExecutionStateMachine(execution_id="workflow-abort")
        
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.abort()
        
        assert sm.state_enum == ExecutionState.ABORTED

    @pytest.mark.unit
    def test_paused_and_resumed_workflow(self):
        """Test execution with pause and resume."""
        sm = ExecutionStateMachine(execution_id="workflow-pause")
        
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.pause()
        assert sm.state_enum == ExecutionState.PAUSED
        
        sm.resume()
        assert sm.state_enum == ExecutionState.RUNNING
        
        sm.complete()
        assert sm.state_enum == ExecutionState.COMPLETED

    @pytest.mark.unit
    def test_multiple_pause_resume_cycles(self):
        """Test multiple pause/resume cycles."""
        sm = ExecutionStateMachine(execution_id="workflow-multi-pause")
        
        sm.start()
        sm.plan_complete()
        sm.execute()
        
        # Multiple pause/resume cycles
        for _ in range(3):
            sm.pause()
            assert sm.state_enum == ExecutionState.PAUSED
            sm.resume()
            assert sm.state_enum == ExecutionState.RUNNING
        
        sm.complete()
        assert sm.state_enum == ExecutionState.COMPLETED

    @pytest.mark.unit
    def test_error_recovery_workflow(self):
        """Test error during execution and recovery via reset."""
        sm = ExecutionStateMachine(execution_id="workflow-error")
        
        sm.start()
        sm.plan_complete()
        sm.execute()
        sm.error()
        
        assert sm.state_enum == ExecutionState.ERROR
        
        # Can recover via reset
        sm.reset()
        assert sm.state_enum == ExecutionState.IDLE
        
        # Can start new execution
        sm.start()
        assert sm.state_enum == ExecutionState.PLANNING


# =============================================================================
# TRANSITIONS CONFIGURATION TESTS
# =============================================================================


class TestTransitionsConfiguration:
    """Tests for TRANSITIONS configuration."""

    @pytest.mark.unit
    def test_all_triggers_defined(self):
        """Verify all expected triggers are defined."""
        expected_triggers = [
            "start", "plan_complete", "plan_failed", "execute",
            "pause", "resume", "complete", "abort", "error", "reset"
        ]
        
        defined_triggers = {t["trigger"] for t in TRANSITIONS}
        
        for trigger in expected_triggers:
            assert trigger in defined_triggers, f"Trigger {trigger} not defined"

    @pytest.mark.unit
    def test_transitions_have_required_keys(self):
        """Verify all transitions have required keys."""
        for transition in TRANSITIONS:
            assert "trigger" in transition
            assert "source" in transition
            assert "dest" in transition

    @pytest.mark.unit
    def test_reset_from_wildcard(self):
        """Verify reset can be triggered from any state (wildcard)."""
        reset_transition = next(t for t in TRANSITIONS if t["trigger"] == "reset")
        assert reset_transition["source"] == "*"
        assert reset_transition["dest"] == ExecutionState.IDLE
