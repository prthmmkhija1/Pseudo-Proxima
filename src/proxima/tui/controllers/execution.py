"""Execution Controller for Proxima TUI.

Handles execution management, progress tracking, and control signals.
"""

from typing import Optional, Callable, List, Dict, Any
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

try:
    from proxima.resources.control import ExecutionController as CoreExecutionController, ControlSignal as CoreControlSignal, ControlState
    from proxima.resources.control import CheckpointManager
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

from ..state import TUIState
from ..state.tui_state import StageInfo, CheckpointInfo
from ..state.events import (
    ExecutionStarted,
    ExecutionProgress,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionPaused,
    ExecutionResumed,
    ExecutionAborted,
    StageStarted,
    StageCompleted,
    CheckpointCreated,
    RollbackCompleted,
)


class ControlSignal(Enum):
    """Control signals for execution."""
    NONE = auto()
    START = auto()
    PAUSE = auto()
    RESUME = auto()
    ABORT = auto()
    ROLLBACK = auto()


class ExecutionController:
    """Controller for execution management.

    Handles starting, pausing, resuming, and aborting executions,
    as well as tracking progress and managing checkpoints.
    """

    def __init__(self, state: TUIState):
        """Initialize the execution controller.

        Args:
            state: The TUI state instance
        """
        self.state = state
        self._executor = None  # Will be set when Proxima core is available
        self._control = None   # Will be set when Proxima core is available
        self._event_callbacks: List[Callable] = []
        self._core_controller = None
        self._checkpoint_manager = None
        self._checkpoints: List[Dict[str, Any]] = []
        
        if CORE_AVAILABLE:
            try:
                self._core_controller = CoreExecutionController()
                checkpoint_dir = Path.home() / ".proxima" / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self._checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            except Exception:
                pass  # Core not available, use simulated mode
    
    def get_status(self) -> Dict[str, Any]:
        """Get current execution status.
        
        Returns:
            Dictionary with status info
        """
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'progress': self.state.progress_percent,
            'stage': self.state.current_stage,
            'stage_index': self.state.stage_index,
            'elapsed_ms': self.state.elapsed_ms,
            'eta_ms': self.state.eta_ms,
            'checkpoint_count': self.state.checkpoint_count,
            'rollback_available': self.can_rollback,
        }

    @property
    def is_running(self) -> bool:
        """Check if execution is running."""
        return self.state.execution_status == "RUNNING"

    @property
    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self.state.execution_status == "PAUSED"

    @property
    def is_idle(self) -> bool:
        """Check if no execution is active."""
        return self.state.execution_status == "IDLE"

    @property
    def can_pause(self) -> bool:
        """Check if execution can be paused."""
        return self.is_running

    @property
    def can_resume(self) -> bool:
        """Check if execution can be resumed."""
        return self.is_paused

    @property
    def can_abort(self) -> bool:
        """Check if execution can be aborted."""
        return self.is_running or self.is_paused

    @property
    def can_rollback(self) -> bool:
        """Check if rollback is available."""
        return self.state.rollback_available and self.state.checkpoint_count > 0

    def start_execution(
        self,
        task: str,
        backend: str,
        simulator: str = "statevector",
        qubits: int = 2,
        shots: int = 1024,
        **config,
    ) -> bool:
        """Start a new execution.

        Args:
            task: Task/circuit name
            backend: Backend name
            simulator: Simulator type
            qubits: Number of qubits
            shots: Number of shots
            **config: Additional configuration

        Returns:
            True if execution started successfully
        """
        if not self.is_idle:
            return False

        # Update state
        self.state.execution_status = "PLANNING"
        self.state.current_task = task
        self.state.current_task_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state.current_backend = backend
        self.state.current_simulator = simulator
        self.state.qubits = qubits
        self.state.shots = shots
        self.state.progress_percent = 0.0
        self.state.elapsed_ms = 0.0
        self.state.eta_ms = None

        # Initialize stages
        self.state.all_stages = [
            StageInfo(name="Planning", index=0),
            StageInfo(name="Backend Initialization", index=1),
            StageInfo(name="Simulation", index=2),
            StageInfo(name="Analysis", index=3),
            StageInfo(name="Report Generation", index=4),
        ]
        self.state.total_stages = len(self.state.all_stages)
        self.state.stage_index = 0
        self.state.current_stage = "Planning"

        # Emit event
        self._emit_event(ExecutionStarted(
            task=task,
            backend=backend,
            simulator=simulator,
            session_id=self.state.active_session_id or "default",
            qubits=qubits,
            shots=shots,
        ))

        # Start execution via Proxima core if available
        if self._core_controller:
            try:
                self._core_controller.start(
                    stages=["Planning", "Backend Init", "Simulation", "Analysis", "Report"],
                    total_stages=5,
                )
                self.state.execution_status = "RUNNING"
            except Exception as e:
                self._emit_event(ExecutionFailed(error=str(e), stage="start"))
                return False
        else:
            # Simulated mode - just update status
            self.state.execution_status = "RUNNING"

        return True

    def pause(self) -> bool:
        """Pause current execution.

        Returns:
            True if pause was successful
        """
        if not self.can_pause:
            return False

        self.state.execution_status = "PAUSED"

        # Create checkpoint with full state
        checkpoint_id = f"cp_{datetime.now().strftime('%H%M%S_%f')}"
        checkpoint = CheckpointInfo(
            id=checkpoint_id,
            stage_index=self.state.stage_index,
            timestamp=datetime.now(),
        )
        
        # Store checkpoint data for rollback
        checkpoint_data = {
            'id': checkpoint_id,
            'stage_index': self.state.stage_index,
            'stage_name': self.state.current_stage,
            'progress_percent': self.state.progress_percent,
            'elapsed_ms': self.state.elapsed_ms,
            'task': self.state.current_task,
            'backend': self.state.current_backend,
            'timestamp': datetime.now().isoformat(),
        }
        self._checkpoints.append(checkpoint_data)
        
        self.state.latest_checkpoint = checkpoint
        self.state.checkpoint_count += 1
        self.state.last_checkpoint_time = datetime.now()
        self.state.rollback_available = True

        # Save checkpoint to disk if manager available
        if self._checkpoint_manager:
            try:
                self._checkpoint_manager.save_checkpoint(checkpoint_id, checkpoint_data)
            except Exception:
                pass  # Continue even if save fails

        self._emit_event(ExecutionPaused(
            checkpoint_id=checkpoint.id,
            stage_index=checkpoint.stage_index,
        ))

        self._emit_event(CheckpointCreated(
            checkpoint_id=checkpoint_id,
            stage_index=self.state.stage_index,
            description=f"Checkpoint at {self.state.current_stage}",
        ))

        # Send pause signal to Proxima core if available
        if self._core_controller:
            try:
                self._core_controller.pause(reason="User requested pause")
            except Exception as e:
                pass  # Continue with TUI state update regardless

        return True

    def resume(self) -> bool:
        """Resume paused execution.

        Returns:
            True if resume was successful
        """
        if not self.can_resume:
            return False

        self.state.execution_status = "RUNNING"

        checkpoint_id = self.state.latest_checkpoint.id if self.state.latest_checkpoint else "unknown"

        self._emit_event(ExecutionResumed(
            checkpoint_id=checkpoint_id,
            stage_index=self.state.stage_index,
        ))

        # Send resume signal to Proxima core if available
        if self._core_controller:
            try:
                self._core_controller.resume()
            except Exception as e:
                pass  # Continue with TUI state update regardless

        return True

    def abort(self, reason: str = "User requested") -> bool:
        """Abort current execution.

        Args:
            reason: Reason for abort

        Returns:
            True if abort was successful
        """
        if not self.can_abort:
            return False

        self.state.execution_status = "ABORTED"

        self._emit_event(ExecutionAborted(reason=reason))

        # Send abort signal to Proxima core if available
        if self._core_controller:
            try:
                self._core_controller.abort(reason=reason)
            except Exception as e:
                pass  # Continue with TUI state update regardless

        # Clear execution state after short delay
        self._cleanup_execution()

        return True

    def rollback(self, checkpoint_id: Optional[str] = None) -> bool:
        """Rollback to a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint ID (uses latest if None)

        Returns:
            True if rollback was successful
        """
        if not self.can_rollback:
            return False

        # Find the checkpoint to rollback to
        checkpoint_data = None
        if checkpoint_id:
            # Find specific checkpoint
            for cp in self._checkpoints:
                if cp['id'] == checkpoint_id:
                    checkpoint_data = cp
                    break
        else:
            # Use latest checkpoint
            if self._checkpoints:
                checkpoint_data = self._checkpoints[-1]
            elif self.state.latest_checkpoint:
                checkpoint_data = {
                    'id': self.state.latest_checkpoint.id,
                    'stage_index': self.state.latest_checkpoint.stage_index,
                }

        if checkpoint_data is None:
            return False

        # Restore state from checkpoint
        stage_index = checkpoint_data.get('stage_index', 0)
        self.state.stage_index = stage_index
        
        if stage_index < len(self.state.all_stages):
            self.state.current_stage = self.state.all_stages[stage_index].name
        
        # Restore progress if available
        if 'progress_percent' in checkpoint_data:
            self.state.progress_percent = checkpoint_data['progress_percent']
        
        if 'elapsed_ms' in checkpoint_data:
            self.state.elapsed_ms = checkpoint_data['elapsed_ms']

        # Mark stages after checkpoint as pending
        for i, stage in enumerate(self.state.all_stages):
            if i > stage_index:
                stage.status = "pending"
            elif i == stage_index:
                stage.status = "running"

        # Remove checkpoints after this one
        if checkpoint_id:
            idx = next((i for i, cp in enumerate(self._checkpoints) if cp['id'] == checkpoint_id), -1)
            if idx >= 0:
                self._checkpoints = self._checkpoints[:idx + 1]

        self._emit_event(RollbackCompleted(
            checkpoint_id=checkpoint_data['id'],
            stage_index=stage_index,
        ))

        # Send rollback signal to Proxima core if available
        if self._core_controller:
            try:
                self._core_controller.rollback_to_checkpoint(checkpoint_data['id'])
            except Exception as e:
                pass  # Continue with TUI state update regardless

        # Resume execution
        self.state.execution_status = "RUNNING"

        return True
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint data dictionaries
        """
        return self._checkpoints.copy()

    def update_progress(
        self,
        percent: float,
        stage: str,
        stage_index: int,
        elapsed_ms: float,
        eta_ms: Optional[float] = None,
    ) -> None:
        """Update execution progress.

        Args:
            percent: Progress percentage (0-100)
            stage: Current stage name
            stage_index: Current stage index
            elapsed_ms: Elapsed time in milliseconds
            eta_ms: Estimated time remaining in milliseconds
        """
        self.state.update_progress(
            percent=percent,
            stage=stage,
            stage_index=stage_index,
            total_stages=self.state.total_stages,
            elapsed_ms=elapsed_ms,
            eta_ms=eta_ms,
        )

        self._emit_event(ExecutionProgress(
            progress=percent,
            stage=stage,
            stage_index=stage_index,
            total_stages=self.state.total_stages,
            elapsed_ms=elapsed_ms,
            eta_ms=eta_ms,
        ))

    def complete_stage(self, stage_index: int, duration_ms: float) -> None:
        """Mark a stage as completed.

        Args:
            stage_index: Index of the completed stage
            duration_ms: Stage duration in milliseconds
        """
        if stage_index < len(self.state.all_stages):
            stage = self.state.all_stages[stage_index]
            stage.status = "done"
            stage.duration_ms = duration_ms
            stage.end_time = datetime.now()

            self.state.completed_stages.append(stage)

            self._emit_event(StageCompleted(
                stage_name=stage.name,
                stage_index=stage_index,
                duration_ms=duration_ms,
                success=True,
            ))

    def complete_execution(self, result: Dict[str, Any]) -> None:
        """Mark execution as completed.

        Args:
            result: Execution result data
        """
        self.state.execution_status = "COMPLETED"

        self._emit_event(ExecutionCompleted(
            result=result,
            total_time_ms=self.state.elapsed_ms,
        ))

        # Clear execution state after short delay
        self._cleanup_execution()

    def fail_execution(self, error: str, stage: str) -> None:
        """Mark execution as failed.

        Args:
            error: Error message
            stage: Stage where error occurred
        """
        self.state.execution_status = "ERROR"

        self._emit_event(ExecutionFailed(
            error=error,
            stage=stage,
            partial_result=None,
        ))

        # Clear execution state after short delay
        self._cleanup_execution()

    def _cleanup_execution(self) -> None:
        """Clean up execution state."""
        # Don't clear immediately - let the UI show final state
        pass

    def reset(self) -> None:
        """Reset execution state."""
        self.state.clear_execution()

    def on_event(self, callback: Callable) -> None:
        """Register an event callback.

        Args:
            callback: Function to call on events
        """
        self._event_callbacks.append(callback)

    def _emit_event(self, event: Any) -> None:
        """Emit an event to callbacks.

        Args:
            event: Event to emit
        """
        for callback in self._event_callbacks:
            callback(event)
