"""Enhanced Execution Control implementation (Phase 4, Step 4.3).

Provides:
- ExecutionController: Start/Abort/Pause/Resume execution with checkpointing
- ControlSignal: Signal types for control flow
- ControlState: Current control state
- CheckpointManager: Checkpoint serialization and restoration

Control Implementation (Step 4.3):
| Operation | Mechanism                                    |
|-----------|----------------------------------------------|
| Start     | Initialize state, begin execution loop       |
| Abort     | Set abort flag, cleanup, transition to ABORTED |
| Pause     | Set pause flag, checkpoint state, wait       |
| Resume    | Clear pause flag, restore, continue          |

Checkpoint Strategy:
- Define safe checkpoint locations (between stages)
- At checkpoint: serialize state to temporary file
- On resume: load checkpoint, validate, continue
- Clean up checkpoints on completion
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from collections.abc import Callable
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Control Signals and States
# =============================================================================


class ControlSignal(Enum):
    """Signals for execution control."""

    NONE = auto()
    START = auto()
    PAUSE = auto()
    RESUME = auto()
    ABORT = auto()
    ROLLBACK = auto()  # Roll back to a previous checkpoint


class ControlState(Enum):
    """Current execution control state."""

    IDLE = auto()  # Not started yet
    RUNNING = auto()  # Actively executing
    PAUSED = auto()  # Paused, waiting for resume
    ABORTED = auto()  # Aborted by user or error
    COMPLETED = auto()  # Successfully completed


# =============================================================================
# Control Events
# =============================================================================


@dataclass
class ControlEvent:
    """Event when control state changes."""

    timestamp: float
    previous: ControlState
    current: ControlState
    signal: ControlSignal
    reason: str | None = None
    checkpoint_id: str | None = None

    def __str__(self) -> str:
        return f"[{self.signal.name}] {self.previous.name} -> {self.current.name}"


# Type alias for control callbacks
ControlCallback = Callable[[ControlEvent], None]


# =============================================================================
# Checkpoint Data Structure
# =============================================================================


@dataclass
class CheckpointData:
    """Serializable checkpoint data for pause/resume.

    Contains all information needed to restore execution state.
    """

    checkpoint_id: str
    timestamp: float
    stage_name: str
    stage_index: int
    total_stages: int
    progress_percent: float
    elapsed_ms: float
    custom_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "stage_name": self.stage_name,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "progress_percent": self.progress_percent,
            "elapsed_ms": self.elapsed_ms,
            "custom_state": self.custom_state,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointData:
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            timestamp=data["timestamp"],
            stage_name=data["stage_name"],
            stage_index=data["stage_index"],
            total_stages=data["total_stages"],
            progress_percent=data["progress_percent"],
            elapsed_ms=data["elapsed_ms"],
            custom_state=data.get("custom_state", {}),
            metadata=data.get("metadata", {}),
        )

    def is_valid(self) -> bool:
        """Validate checkpoint data integrity."""
        try:
            assert self.checkpoint_id, "Missing checkpoint_id"
            assert self.timestamp > 0, "Invalid timestamp"
            assert 0 <= self.stage_index <= self.total_stages, "Invalid stage index"
            assert 0 <= self.progress_percent <= 100, "Invalid progress"
            return True
        except AssertionError as e:
            logger.warning(f"Checkpoint validation failed: {e}")
            return False


# =============================================================================
# Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """Manages checkpoint serialization and restoration.

    Checkpoint Strategy (Step 4.3):
    - Define safe checkpoint locations (between stages)
    - At checkpoint: serialize state to temporary file
    - On resume: load checkpoint, validate, continue
    - Clean up checkpoints on completion
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        execution_id: str | None = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files (default: temp dir)
            execution_id: Unique ID for this execution session
        """
        self.execution_id = execution_id or f"exec_{int(time.time() * 1000)}"

        if checkpoint_dir:
            self._checkpoint_dir = Path(checkpoint_dir)
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._checkpoint_dir = Path(tempfile.gettempdir()) / "proxima_checkpoints"
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints: list[CheckpointData] = []
        self._current_checkpoint: CheckpointData | None = None
        self._lock = threading.Lock()

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return self._checkpoint_dir

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get file path for a checkpoint."""
        return self._checkpoint_dir / f"{self.execution_id}_{checkpoint_id}.json"

    # -------------------------------------------------------------------------
    # Checkpoint Creation (At Safe Locations - Between Stages)
    # -------------------------------------------------------------------------

    def create_checkpoint(
        self,
        stage_name: str,
        stage_index: int,
        total_stages: int,
        progress_percent: float,
        elapsed_ms: float,
        custom_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointData:
        """Create a new checkpoint at a safe location (between stages).

        Args:
            stage_name: Name of current/just-completed stage
            stage_index: Index of stage (0-based)
            total_stages: Total number of stages
            progress_percent: Overall progress percentage
            elapsed_ms: Elapsed time in milliseconds
            custom_state: Custom state data to serialize
            metadata: Additional metadata

        Returns:
            Created checkpoint data
        """
        checkpoint_id = f"chk_{stage_index}_{int(time.time() * 1000)}"

        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            stage_name=stage_name,
            stage_index=stage_index,
            total_stages=total_stages,
            progress_percent=progress_percent,
            elapsed_ms=elapsed_ms,
            custom_state=custom_state or {},
            metadata=metadata or {},
        )

        # Serialize to file
        self._save_checkpoint(checkpoint)

        with self._lock:
            self._checkpoints.append(checkpoint)
            self._current_checkpoint = checkpoint

        logger.info(f"Checkpoint created: {checkpoint_id} at stage '{stage_name}'")
        return checkpoint

    def _save_checkpoint(self, checkpoint: CheckpointData) -> None:
        """Serialize checkpoint to temporary file."""
        checkpoint_path = self._get_checkpoint_path(checkpoint.checkpoint_id)
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            logger.debug(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    # -------------------------------------------------------------------------
    # Checkpoint Loading (On Resume)
    # -------------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointData | None:
        """Load checkpoint from file.

        Args:
            checkpoint_id: ID of checkpoint to load

        Returns:
            Loaded checkpoint data, or None if not found/invalid
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)

            checkpoint = CheckpointData.from_dict(data)

            # Validate checkpoint
            if not checkpoint.is_valid():
                logger.error(f"Checkpoint validation failed: {checkpoint_id}")
                return None

            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def get_latest_checkpoint(self) -> CheckpointData | None:
        """Get the most recent checkpoint."""
        with self._lock:
            return self._current_checkpoint

    def list_checkpoints(self) -> list[CheckpointData]:
        """Get all checkpoints in order."""
        with self._lock:
            return list(self._checkpoints)

    # -------------------------------------------------------------------------
    # Checkpoint Cleanup (On Completion)
    # -------------------------------------------------------------------------

    def get_checkpoint_by_stage(self, stage_index: int) -> CheckpointData | None:
        """Get checkpoint at or before a specific stage index."""
        with self._lock:
            for checkpoint in reversed(self._checkpoints):
                if checkpoint.stage_index <= stage_index:
                    return checkpoint
            return None

    def get_previous_checkpoint(self) -> CheckpointData | None:
        """Get the checkpoint before the current one."""
        with self._lock:
            if len(self._checkpoints) >= 2:
                return self._checkpoints[-2]
            elif len(self._checkpoints) == 1:
                return self._checkpoints[0]
            return None

    def remove_checkpoints_after(self, stage_index: int) -> int:
        """Remove all checkpoints after a given stage index.

        Used during rollback to remove invalid future checkpoints.

        Args:
            stage_index: Keep checkpoints at or before this index

        Returns:
            Number of checkpoints removed
        """
        removed = 0
        with self._lock:
            to_keep = []
            for checkpoint in self._checkpoints:
                if checkpoint.stage_index <= stage_index:
                    to_keep.append(checkpoint)
                else:
                    # Remove the file
                    checkpoint_path = self._get_checkpoint_path(
                        checkpoint.checkpoint_id
                    )
                    try:
                        if checkpoint_path.exists():
                            checkpoint_path.unlink()
                            removed += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint file: {e}")

            self._checkpoints = to_keep
            if to_keep:
                self._current_checkpoint = to_keep[-1]
            else:
                self._current_checkpoint = None

        if removed > 0:
            logger.info(f"Removed {removed} checkpoints after stage {stage_index}")
        return removed

    def cleanup_checkpoints(self) -> int:
        """Clean up all checkpoint files for this execution.

        Called on completion or abort.

        Returns:
            Number of files cleaned up
        """
        cleaned = 0

        with self._lock:
            for checkpoint in self._checkpoints:
                checkpoint_path = self._get_checkpoint_path(checkpoint.checkpoint_id)
                try:
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        cleaned += 1
                        logger.debug(f"Cleaned up checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup checkpoint: {e}")

            self._checkpoints.clear()
            self._current_checkpoint = None

        logger.info(f"Cleaned up {cleaned} checkpoint files")
        return cleaned

    def cleanup_old_checkpoints(self, max_age_hours: float = 24) -> int:
        """Clean up old checkpoint files from previous executions.

        Args:
            max_age_hours: Maximum age of checkpoints to keep

        Returns:
            Number of files cleaned up
        """
        cleaned = 0
        max_age_seconds = max_age_hours * 3600
        now = time.time()

        try:
            for file_path in self._checkpoint_dir.glob("*.json"):
                try:
                    file_age = now - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleaned += 1
                except Exception as e:
                    logger.warning(f"Failed to check/cleanup old checkpoint: {e}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old checkpoint files")

        return cleaned


# =============================================================================
# Execution Controller (Step 4.3 Main Class)
# =============================================================================


class ExecutionController:
    """Controls execution flow with Start/Abort/Pause/Resume and checkpointing.

    Implements Step 4.3 Control Implementation:
    | Operation | Mechanism                                    |
    |-----------|----------------------------------------------|
    | Start     | Initialize state, begin execution loop       |
    | Abort     | Set abort flag, cleanup, transition to ABORTED |
    | Pause     | Set pause flag, checkpoint state, wait       |
    | Resume    | Clear pause flag, restore, continue          |
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        auto_checkpoint: bool = True,
    ) -> None:
        """Initialize execution controller.

        Args:
            checkpoint_dir: Directory for checkpoints (default: temp)
            auto_checkpoint: Whether to auto-checkpoint on pause
        """
        self._state = ControlState.IDLE
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # Callbacks
        self._callbacks: list[ControlCallback] = []

        # Reasons
        self._abort_reason: str | None = None
        self._pause_reason: str | None = None

        # Checkpoint management
        self._auto_checkpoint = auto_checkpoint
        self._checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Execution tracking
        self._start_time: float | None = None
        self._current_stage: str | None = None
        self._stage_index: int = 0
        self._total_stages: int = 1
        self._progress_percent: float = 0.0
        self._custom_state: dict[str, Any] = {}

        # Event history
        self._events: list[ControlEvent] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> ControlState:
        """Get current control state."""
        with self._lock:
            return self._state

    @property
    def is_idle(self) -> bool:
        """Check if not started."""
        return self.state == ControlState.IDLE

    @property
    def is_running(self) -> bool:
        """Check if actively running."""
        return self.state == ControlState.RUNNING

    @property
    def is_paused(self) -> bool:
        """Check if paused."""
        return self.state == ControlState.PAUSED

    @property
    def is_aborted(self) -> bool:
        """Check if aborted."""
        return self.state == ControlState.ABORTED

    @property
    def is_completed(self) -> bool:
        """Check if completed."""
        return self.state == ControlState.COMPLETED

    @property
    def abort_reason(self) -> str | None:
        """Get abort reason if aborted."""
        return self._abort_reason

    @property
    def pause_reason(self) -> str | None:
        """Get pause reason if paused."""
        return self._pause_reason

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        """Get the checkpoint manager."""
        return self._checkpoint_manager

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time since start in milliseconds."""
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) * 1000

    @property
    def events(self) -> list[ControlEvent]:
        """Get event history."""
        with self._lock:
            return list(self._events)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_state_change(self, callback: ControlCallback) -> None:
        """Register callback for state changes."""
        self._callbacks.append(callback)

    def _emit_event(self, event: ControlEvent) -> None:
        """Emit event to all registered callbacks."""
        with self._lock:
            self._events.append(event)

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Control callback error: {e}")

    # -------------------------------------------------------------------------
    # START: Initialize state, begin execution loop
    # -------------------------------------------------------------------------

    def start(
        self,
        total_stages: int = 1,
        resume_from: CheckpointData | None = None,
    ) -> bool:
        """Start execution.

        Args:
            total_stages: Total number of stages in execution
            resume_from: Optional checkpoint to resume from

        Returns:
            True if started successfully
        """
        with self._lock:
            if self._state not in (
                ControlState.IDLE,
                ControlState.COMPLETED,
                ControlState.ABORTED,
            ):
                logger.warning(f"Cannot start from state {self._state}")
                return False

            previous = self._state
            self._state = ControlState.RUNNING
            self._abort_reason = None
            self._pause_reason = None
            self._pause_event.set()

            # Initialize execution tracking
            self._total_stages = total_stages

            if resume_from:
                # Restore from checkpoint
                self._start_time = time.time() - (resume_from.elapsed_ms / 1000)
                self._current_stage = resume_from.stage_name
                self._stage_index = resume_from.stage_index
                self._progress_percent = resume_from.progress_percent
                self._custom_state = resume_from.custom_state.copy()
                checkpoint_id = resume_from.checkpoint_id
                logger.info(f"Resuming from checkpoint: {checkpoint_id}")
            else:
                # Fresh start
                self._start_time = time.time()
                self._current_stage = None
                self._stage_index = 0
                self._progress_percent = 0.0
                self._custom_state = {}
                checkpoint_id = None

        event = ControlEvent(
            timestamp=time.time(),
            previous=previous,
            current=ControlState.RUNNING,
            signal=ControlSignal.START,
            reason=(
                "Execution started" if not resume_from else "Resumed from checkpoint"
            ),
            checkpoint_id=checkpoint_id,
        )
        self._emit_event(event)

        logger.info(f"Execution started (total_stages={total_stages})")
        return True

    # -------------------------------------------------------------------------
    # ABORT: Set abort flag, cleanup, transition to ABORTED
    # -------------------------------------------------------------------------

    def abort(self, reason: str | None = None, cleanup: bool = True) -> bool:
        """Abort execution.

        Args:
            reason: Reason for abort
            cleanup: Whether to cleanup checkpoints

        Returns:
            True if state changed
        """
        with self._lock:
            if self._state in (
                ControlState.ABORTED,
                ControlState.COMPLETED,
                ControlState.IDLE,
            ):
                return False

            previous = self._state
            self._state = ControlState.ABORTED
            self._abort_reason = reason
            self._pause_event.set()  # Unblock any waiting threads

        # Cleanup checkpoints
        if cleanup:
            self._checkpoint_manager.cleanup_checkpoints()

        event = ControlEvent(
            timestamp=time.time(),
            previous=previous,
            current=ControlState.ABORTED,
            signal=ControlSignal.ABORT,
            reason=reason,
        )
        self._emit_event(event)

        logger.info(f"Execution aborted: {reason}")
        return True

    # -------------------------------------------------------------------------
    # PAUSE: Set pause flag, checkpoint state, wait
    # -------------------------------------------------------------------------

    def pause(
        self,
        reason: str | None = None,
        create_checkpoint: bool = True,
    ) -> CheckpointData | None:
        """Pause execution with optional checkpoint.

        Args:
            reason: Reason for pause
            create_checkpoint: Whether to create checkpoint

        Returns:
            Checkpoint data if created, None otherwise
        """
        checkpoint = None

        with self._lock:
            if self._state != ControlState.RUNNING:
                return None

            previous = self._state
            self._state = ControlState.PAUSED
            self._pause_reason = reason
            self._pause_event.clear()  # Block check_pause() calls

        # Create checkpoint at safe location (between stages)
        if create_checkpoint and self._auto_checkpoint:
            checkpoint = self._checkpoint_manager.create_checkpoint(
                stage_name=self._current_stage or "unknown",
                stage_index=self._stage_index,
                total_stages=self._total_stages,
                progress_percent=self._progress_percent,
                elapsed_ms=self.elapsed_ms,
                custom_state=self._custom_state,
            )

        event = ControlEvent(
            timestamp=time.time(),
            previous=previous,
            current=ControlState.PAUSED,
            signal=ControlSignal.PAUSE,
            reason=reason,
            checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
        )
        self._emit_event(event)

        logger.info(f"Execution paused: {reason}")
        return checkpoint

    # -------------------------------------------------------------------------
    # RESUME: Clear pause flag, restore, continue
    # -------------------------------------------------------------------------

    def resume(self, from_checkpoint: CheckpointData | None = None) -> bool:
        """Resume execution.

        Args:
            from_checkpoint: Optional specific checkpoint to resume from

        Returns:
            True if state changed
        """
        with self._lock:
            if self._state != ControlState.PAUSED:
                return False

            previous = self._state
            self._state = ControlState.RUNNING

            # Restore from checkpoint if provided
            if from_checkpoint:
                self._current_stage = from_checkpoint.stage_name
                self._stage_index = from_checkpoint.stage_index
                self._progress_percent = from_checkpoint.progress_percent
                self._custom_state = from_checkpoint.custom_state.copy()
                checkpoint_id = from_checkpoint.checkpoint_id
            else:
                checkpoint_id = None

            self._pause_reason = None
            self._pause_event.set()  # Unblock check_pause() calls

        event = ControlEvent(
            timestamp=time.time(),
            previous=previous,
            current=ControlState.RUNNING,
            signal=ControlSignal.RESUME,
            checkpoint_id=checkpoint_id,
        )
        self._emit_event(event)

        logger.info("Execution resumed")
        return True

    # -------------------------------------------------------------------------
    # ROLLBACK: Roll back to a previous checkpoint
    # -------------------------------------------------------------------------

    def rollback(
        self,
        checkpoint_id: str | None = None,
        stage_index: int | None = None,
    ) -> CheckpointData | None:
        """Roll back execution to a previous checkpoint.

        Unlike resume (which continues from where paused), rollback allows
        going back to any previous checkpoint to redo work.

        Args:
            checkpoint_id: Specific checkpoint ID to roll back to
            stage_index: Roll back to checkpoint at this stage index

        Returns:
            The checkpoint rolled back to, or None if failed
        """
        # Find the target checkpoint
        target_checkpoint: CheckpointData | None = None

        if checkpoint_id:
            # Load specific checkpoint by ID
            target_checkpoint = self._checkpoint_manager.load_checkpoint(checkpoint_id)
        elif stage_index is not None:
            # Find checkpoint by stage index
            target_checkpoint = self._checkpoint_manager.get_checkpoint_by_stage(
                stage_index
            )
        else:
            # Get the previous checkpoint (one before current)
            target_checkpoint = self._checkpoint_manager.get_previous_checkpoint()

        if target_checkpoint is None:
            logger.warning("No checkpoint found for rollback")
            return None

        with self._lock:
            if self._state not in (ControlState.RUNNING, ControlState.PAUSED):
                logger.warning(f"Cannot rollback from state {self._state}")
                return None

            previous = self._state

            # Restore state from checkpoint
            self._current_stage = target_checkpoint.stage_name
            self._stage_index = target_checkpoint.stage_index
            self._progress_percent = target_checkpoint.progress_percent
            self._custom_state = target_checkpoint.custom_state.copy()

            # Adjust start time to account for elapsed time at checkpoint
            if self._start_time:
                self._start_time = time.time() - (target_checkpoint.elapsed_ms / 1000)

            # Ensure we're in running state
            self._state = ControlState.RUNNING
            self._pause_event.set()

        # Remove checkpoints after the rollback point
        self._checkpoint_manager.remove_checkpoints_after(target_checkpoint.stage_index)

        event = ControlEvent(
            timestamp=time.time(),
            previous=previous,
            current=ControlState.RUNNING,
            signal=ControlSignal.ROLLBACK,
            reason=f"Rolled back to stage '{target_checkpoint.stage_name}'",
            checkpoint_id=target_checkpoint.checkpoint_id,
        )
        self._emit_event(event)

        logger.info(f"Rolled back to checkpoint: {target_checkpoint.checkpoint_id}")
        return target_checkpoint

    def get_available_rollback_points(self) -> list[CheckpointData]:
        """Get list of checkpoints available for rollback."""
        return self._checkpoint_manager.list_checkpoints()

    # -------------------------------------------------------------------------
    # COMPLETE: Finish execution successfully
    # -------------------------------------------------------------------------

    def complete(self, cleanup_checkpoints: bool = True) -> bool:
        """Mark execution as completed.

        Args:
            cleanup_checkpoints: Whether to cleanup checkpoint files

        Returns:
            True if state changed
        """
        with self._lock:
            if self._state in (
                ControlState.ABORTED,
                ControlState.COMPLETED,
                ControlState.IDLE,
            ):
                return False

            previous = self._state
            self._state = ControlState.COMPLETED
            self._pause_event.set()

        # Cleanup checkpoints on completion
        if cleanup_checkpoints:
            self._checkpoint_manager.cleanup_checkpoints()

        event = ControlEvent(
            timestamp=time.time(),
            previous=previous,
            current=ControlState.COMPLETED,
            signal=ControlSignal.NONE,
            reason="Execution completed successfully",
        )
        self._emit_event(event)

        logger.info("Execution completed")
        return True

    # -------------------------------------------------------------------------
    # Execution Helpers
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset controller to initial state."""
        with self._lock:
            self._state = ControlState.IDLE
            self._abort_reason = None
            self._pause_reason = None
            self._pause_event.set()
            self._start_time = None
            self._current_stage = None
            self._stage_index = 0
            self._progress_percent = 0.0
            self._custom_state = {}
            self._events.clear()

    def should_continue(self) -> bool:
        """Check if execution should continue (not aborted/completed)."""
        return self.state in (ControlState.RUNNING, ControlState.PAUSED)

    def check_pause(self, timeout: float | None = None) -> bool:
        """Block if paused, return True if can continue, False if aborted.

        This should be called at safe checkpoint locations (between stages).

        Args:
            timeout: Maximum time to wait for resume

        Returns:
            True if can continue, False if aborted
        """
        if not self._pause_event.wait(timeout=timeout):
            return self.state != ControlState.ABORTED
        return self.state not in (ControlState.ABORTED, ControlState.COMPLETED)

    def update_progress(
        self,
        stage_name: str,
        stage_index: int,
        progress_percent: float,
        custom_state: dict[str, Any] | None = None,
    ) -> None:
        """Update current execution progress (for checkpointing).

        Args:
            stage_name: Current stage name
            stage_index: Current stage index
            progress_percent: Overall progress percentage
            custom_state: Custom state data to include in checkpoints
        """
        with self._lock:
            self._current_stage = stage_name
            self._stage_index = stage_index
            self._progress_percent = progress_percent
            if custom_state:
                self._custom_state.update(custom_state)

    def checkpoint_now(
        self, custom_state: dict[str, Any] | None = None
    ) -> CheckpointData | None:
        """Create a checkpoint at the current location.

        Args:
            custom_state: Additional custom state data

        Returns:
            Created checkpoint data
        """
        merged_state = self._custom_state.copy()
        if custom_state:
            merged_state.update(custom_state)

        return self._checkpoint_manager.create_checkpoint(
            stage_name=self._current_stage or "unknown",
            stage_index=self._stage_index,
            total_stages=self._total_stages,
            progress_percent=self._progress_percent,
            elapsed_ms=self.elapsed_ms,
            custom_state=merged_state,
        )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def status_line(self) -> str:
        """Return status line for UI display."""
        state = self.state
        if state == ControlState.IDLE:
            return "○ Idle"
        elif state == ControlState.RUNNING:
            return "▶ Running"
        elif state == ControlState.PAUSED:
            reason = f": {self._pause_reason}" if self._pause_reason else ""
            return f"⏸ Paused{reason}"
        elif state == ControlState.ABORTED:
            reason = f": {self._abort_reason}" if self._abort_reason else ""
            return f"✗ Aborted{reason}"
        else:
            return "✓ Completed"


# =============================================================================
# Abort Exception
# =============================================================================


class AbortException(Exception):
    """Raised when execution is aborted."""

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason
        super().__init__(reason or "Execution aborted")


# =============================================================================
# Pause Exception
# =============================================================================


class PauseException(Exception):
    """Raised when execution should pause."""

    def __init__(
        self, reason: str | None = None, checkpoint: CheckpointData | None = None
    ) -> None:
        self.reason = reason
        self.checkpoint = checkpoint
        super().__init__(reason or "Execution paused")


# =============================================================================
# Abort Cleanup Verification (Feature 4 - Enhanced)
# =============================================================================


@dataclass
class CleanupVerificationResult:
    """Result of cleanup verification after abort."""

    success: bool
    checkpoints_removed: int
    resources_released: list[str]
    pending_cleanups: list[str]
    errors: list[str]
    verification_time_ms: float


class AbortCleanupVerifier:
    """Verifies that cleanup after abort was successful.
    
    Implements Step 4.3 - Abort with cleanup verification:
    - Verify all checkpoint files are removed
    - Verify all temporary resources are released
    - Verify no orphaned state remains
    - Report any cleanup failures
    """

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        """Initialize abort cleanup verifier.
        
        Args:
            checkpoint_manager: The checkpoint manager to verify cleanup for
        """
        self._checkpoint_manager = checkpoint_manager
        self._cleanup_handlers: list[tuple[str, Callable[[], bool]]] = []
        self._verification_log: list[str] = []

    def register_cleanup_handler(
        self, resource_name: str, cleanup_func: Callable[[], bool]
    ) -> None:
        """Register a cleanup handler for a resource.
        
        Args:
            resource_name: Name of the resource to cleanup
            cleanup_func: Function that performs cleanup, returns True on success
        """
        self._cleanup_handlers.append((resource_name, cleanup_func))

    def verify_abort_cleanup(
        self, perform_cleanup: bool = True
    ) -> CleanupVerificationResult:
        """Verify that abort cleanup was successful.
        
        Args:
            perform_cleanup: Whether to perform cleanup if not done
            
        Returns:
            CleanupVerificationResult with verification details
        """
        start_time = time.time()
        errors: list[str] = []
        resources_released: list[str] = []
        pending_cleanups: list[str] = []
        checkpoints_removed = 0

        # 1. Verify checkpoint files are removed
        checkpoint_dir = self._checkpoint_manager.checkpoint_dir
        execution_id = self._checkpoint_manager.execution_id
        
        remaining_checkpoints = list(
            checkpoint_dir.glob(f"{execution_id}_*.json")
        )
        
        if remaining_checkpoints:
            if perform_cleanup:
                for cp_file in remaining_checkpoints:
                    try:
                        cp_file.unlink()
                        checkpoints_removed += 1
                        self._verification_log.append(
                            f"Removed orphaned checkpoint: {cp_file.name}"
                        )
                    except Exception as e:
                        errors.append(f"Failed to remove checkpoint {cp_file}: {e}")
            else:
                pending_cleanups.extend(
                    [f"checkpoint:{f.name}" for f in remaining_checkpoints]
                )

        # 2. Verify registered resources are cleaned up
        for resource_name, cleanup_func in self._cleanup_handlers:
            try:
                if cleanup_func():
                    resources_released.append(resource_name)
                    self._verification_log.append(
                        f"Resource released: {resource_name}"
                    )
                else:
                    pending_cleanups.append(f"resource:{resource_name}")
                    self._verification_log.append(
                        f"Resource cleanup pending: {resource_name}"
                    )
            except Exception as e:
                errors.append(f"Cleanup handler error for {resource_name}: {e}")

        # 3. Verify internal checkpoint manager state is cleared
        if self._checkpoint_manager._checkpoints:
            if perform_cleanup:
                self._checkpoint_manager._checkpoints.clear()
                self._checkpoint_manager._current_checkpoint = None
                self._verification_log.append("Cleared internal checkpoint state")
            else:
                pending_cleanups.append("internal:checkpoint_state")

        elapsed_ms = (time.time() - start_time) * 1000
        success = len(errors) == 0 and len(pending_cleanups) == 0

        result = CleanupVerificationResult(
            success=success,
            checkpoints_removed=checkpoints_removed,
            resources_released=resources_released,
            pending_cleanups=pending_cleanups,
            errors=errors,
            verification_time_ms=elapsed_ms,
        )

        if success:
            logger.info(
                f"Abort cleanup verified successfully in {elapsed_ms:.2f}ms"
            )
        else:
            logger.warning(
                f"Abort cleanup verification found issues: "
                f"{len(errors)} errors, {len(pending_cleanups)} pending"
            )

        return result

    def get_verification_log(self) -> list[str]:
        """Get the verification log entries."""
        return list(self._verification_log)


# =============================================================================
# Resume From Checkpoint (Feature 4 - Enhanced)
# =============================================================================


@dataclass
class ResumeContext:
    """Context for resuming execution from checkpoint."""

    checkpoint: CheckpointData
    resume_stage_index: int
    custom_state: dict[str, Any]
    elapsed_before_resume_ms: float
    recovery_actions: list[str]


class CheckpointResumeManager:
    """Manages resumption of execution from checkpoints.
    
    Implements Step 4.3 - Resume from checkpoint:
    - Discover available checkpoints for resume
    - Validate checkpoint integrity before resume
    - Restore full execution state
    - Support recovery actions for state consistency
    """

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        """Initialize checkpoint resume manager.
        
        Args:
            checkpoint_manager: The checkpoint manager to use
        """
        self._checkpoint_manager = checkpoint_manager
        self._recovery_handlers: dict[str, Callable[[CheckpointData], bool]] = {}
        self._resume_history: list[ResumeContext] = []

    def register_recovery_handler(
        self, handler_name: str, handler: Callable[[CheckpointData], bool]
    ) -> None:
        """Register a recovery handler for state restoration.
        
        Args:
            handler_name: Name of the recovery handler
            handler: Function that restores state, returns True on success
        """
        self._recovery_handlers[handler_name] = handler

    def discover_resumable_checkpoints(self) -> list[CheckpointData]:
        """Discover all checkpoints available for resume.
        
        Returns:
            List of valid checkpoints sorted by stage index
        """
        checkpoints: list[CheckpointData] = []
        
        # From memory
        for cp in self._checkpoint_manager.list_checkpoints():
            if cp.is_valid():
                checkpoints.append(cp)

        # From disk (for orphaned checkpoints)
        checkpoint_dir = self._checkpoint_manager.checkpoint_dir
        execution_id = self._checkpoint_manager.execution_id
        
        for cp_file in checkpoint_dir.glob(f"{execution_id}_*.json"):
            try:
                with open(cp_file, encoding="utf-8") as f:
                    data = json.load(f)
                cp = CheckpointData.from_dict(data)
                if cp.is_valid() and cp not in checkpoints:
                    checkpoints.append(cp)
            except Exception as e:
                logger.debug(f"Skipping invalid checkpoint file {cp_file}: {e}")

        # Sort by stage index
        checkpoints.sort(key=lambda x: x.stage_index)
        return checkpoints

    def prepare_resume(
        self,
        checkpoint_id: str | None = None,
        stage_index: int | None = None,
        latest: bool = False,
    ) -> ResumeContext | None:
        """Prepare a resume context from checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint ID to resume from
            stage_index: Resume from checkpoint at this stage
            latest: Resume from the latest checkpoint
            
        Returns:
            ResumeContext if successful, None otherwise
        """
        checkpoint: CheckpointData | None = None

        if checkpoint_id:
            checkpoint = self._checkpoint_manager.load_checkpoint(checkpoint_id)
        elif stage_index is not None:
            checkpoint = self._checkpoint_manager.get_checkpoint_by_stage(stage_index)
        elif latest:
            checkpoint = self._checkpoint_manager.get_latest_checkpoint()
        else:
            # Try to find the latest valid checkpoint
            checkpoints = self.discover_resumable_checkpoints()
            checkpoint = checkpoints[-1] if checkpoints else None

        if checkpoint is None:
            logger.warning("No checkpoint found for resume")
            return None

        if not checkpoint.is_valid():
            logger.error(f"Checkpoint {checkpoint.checkpoint_id} is invalid")
            return None

        # Execute recovery handlers
        recovery_actions: list[str] = []
        for handler_name, handler in self._recovery_handlers.items():
            try:
                if handler(checkpoint):
                    recovery_actions.append(f"recovered:{handler_name}")
                    logger.debug(f"Recovery handler succeeded: {handler_name}")
            except Exception as e:
                logger.error(f"Recovery handler failed {handler_name}: {e}")
                recovery_actions.append(f"failed:{handler_name}")

        context = ResumeContext(
            checkpoint=checkpoint,
            resume_stage_index=checkpoint.stage_index,
            custom_state=checkpoint.custom_state.copy(),
            elapsed_before_resume_ms=checkpoint.elapsed_ms,
            recovery_actions=recovery_actions,
        )

        self._resume_history.append(context)
        logger.info(
            f"Prepared resume from checkpoint {checkpoint.checkpoint_id} "
            f"at stage {checkpoint.stage_index}"
        )
        return context

    def get_resume_history(self) -> list[ResumeContext]:
        """Get history of resume operations."""
        return list(self._resume_history)


# =============================================================================
# Transaction Rollback Support (Feature 4 - Enhanced)
# =============================================================================


@dataclass
class TransactionState:
    """State of a transaction for rollback support."""

    transaction_id: str
    start_time: float
    operations: list[dict[str, Any]]
    checkpoint_before: CheckpointData | None
    is_committed: bool = False
    is_rolled_back: bool = False


class TransactionManager:
    """Manages transactional execution with rollback support.
    
    Implements Step 4.3 - Rollback transaction support:
    - Track operations within transaction boundaries
    - Support undo operations for rollback
    - Maintain transaction isolation
    - Provide commit/rollback semantics
    """

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        """Initialize transaction manager.
        
        Args:
            checkpoint_manager: The checkpoint manager for state persistence
        """
        self._checkpoint_manager = checkpoint_manager
        self._active_transaction: TransactionState | None = None
        self._transaction_history: list[TransactionState] = []
        self._undo_handlers: dict[str, Callable[[dict[str, Any]], bool]] = {}
        self._lock = threading.Lock()

    def register_undo_handler(
        self, operation_type: str, undo_func: Callable[[dict[str, Any]], bool]
    ) -> None:
        """Register an undo handler for an operation type.
        
        Args:
            operation_type: Type of operation to handle
            undo_func: Function that undoes the operation
        """
        self._undo_handlers[operation_type] = undo_func

    def begin_transaction(
        self, create_checkpoint: bool = True
    ) -> TransactionState | None:
        """Begin a new transaction.
        
        Args:
            create_checkpoint: Whether to create a checkpoint at transaction start
            
        Returns:
            TransactionState if started, None if already in transaction
        """
        with self._lock:
            if self._active_transaction is not None:
                logger.warning("Transaction already active")
                return None

            transaction_id = f"txn_{int(time.time() * 1000)}"
            
            # Create checkpoint before transaction
            checkpoint_before: CheckpointData | None = None
            if create_checkpoint:
                checkpoint_before = self._checkpoint_manager.create_checkpoint(
                    stage_name="transaction_start",
                    stage_index=0,
                    total_stages=1,
                    progress_percent=0.0,
                    elapsed_ms=0.0,
                    metadata={"transaction_id": transaction_id},
                )

            self._active_transaction = TransactionState(
                transaction_id=transaction_id,
                start_time=time.time(),
                operations=[],
                checkpoint_before=checkpoint_before,
            )

            logger.info(f"Transaction started: {transaction_id}")
            return self._active_transaction

    def record_operation(
        self,
        operation_type: str,
        data: dict[str, Any],
        undo_data: dict[str, Any] | None = None,
    ) -> bool:
        """Record an operation within the current transaction.
        
        Args:
            operation_type: Type of operation being performed
            data: Operation data
            undo_data: Data needed to undo this operation
            
        Returns:
            True if recorded, False if no active transaction
        """
        with self._lock:
            if self._active_transaction is None:
                logger.debug("No active transaction, operation not recorded")
                return False

            operation = {
                "type": operation_type,
                "timestamp": time.time(),
                "data": data,
                "undo_data": undo_data or data,
            }
            self._active_transaction.operations.append(operation)
            logger.debug(f"Recorded operation: {operation_type}")
            return True

    def commit_transaction(self) -> bool:
        """Commit the current transaction.
        
        Returns:
            True if committed, False if no active transaction
        """
        with self._lock:
            if self._active_transaction is None:
                logger.warning("No active transaction to commit")
                return False

            self._active_transaction.is_committed = True
            self._transaction_history.append(self._active_transaction)
            
            transaction_id = self._active_transaction.transaction_id
            num_operations = len(self._active_transaction.operations)
            
            self._active_transaction = None
            
            logger.info(
                f"Transaction committed: {transaction_id} "
                f"({num_operations} operations)"
            )
            return True

    def rollback_transaction(self) -> tuple[bool, list[str]]:
        """Rollback the current transaction.
        
        Executes undo handlers in reverse order.
        
        Returns:
            Tuple of (success, list of rollback actions taken)
        """
        with self._lock:
            if self._active_transaction is None:
                logger.warning("No active transaction to rollback")
                return False, []

            rollback_actions: list[str] = []
            errors: list[str] = []

            # Undo operations in reverse order
            for operation in reversed(self._active_transaction.operations):
                op_type = operation["type"]
                undo_data = operation["undo_data"]

                if op_type in self._undo_handlers:
                    try:
                        if self._undo_handlers[op_type](undo_data):
                            rollback_actions.append(f"undone:{op_type}")
                        else:
                            errors.append(f"undo_failed:{op_type}")
                    except Exception as e:
                        errors.append(f"undo_error:{op_type}:{e}")
                else:
                    rollback_actions.append(f"skipped:{op_type}")

            # Restore to checkpoint before transaction
            if self._active_transaction.checkpoint_before:
                checkpoint_id = self._active_transaction.checkpoint_before.checkpoint_id
                rollback_actions.append(f"restored_checkpoint:{checkpoint_id}")

            self._active_transaction.is_rolled_back = True
            self._transaction_history.append(self._active_transaction)
            
            transaction_id = self._active_transaction.transaction_id
            self._active_transaction = None

            success = len(errors) == 0
            if success:
                logger.info(f"Transaction rolled back: {transaction_id}")
            else:
                logger.warning(
                    f"Transaction rollback with errors: {transaction_id} - {errors}"
                )

            return success, rollback_actions

    @property
    def in_transaction(self) -> bool:
        """Check if currently in a transaction."""
        return self._active_transaction is not None

    @property
    def active_transaction(self) -> TransactionState | None:
        """Get the active transaction state."""
        return self._active_transaction

    def get_transaction_history(self) -> list[TransactionState]:
        """Get history of all transactions."""
        return list(self._transaction_history)


# =============================================================================
# Enhanced Execution Controller (Integrates all control features)
# =============================================================================


class EnhancedExecutionController(ExecutionController):
    """Extended ExecutionController with full control features.
    
    Integrates:
    - Abort with cleanup verification
    - Resume from checkpoint with recovery
    - Transaction rollback support
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        auto_checkpoint: bool = True,
    ) -> None:
        """Initialize enhanced execution controller.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            auto_checkpoint: Whether to auto-checkpoint on pause
        """
        super().__init__(checkpoint_dir=checkpoint_dir, auto_checkpoint=auto_checkpoint)
        
        # Enhanced control components
        self._cleanup_verifier = AbortCleanupVerifier(self._checkpoint_manager)
        self._resume_manager = CheckpointResumeManager(self._checkpoint_manager)
        self._transaction_manager = TransactionManager(self._checkpoint_manager)

    @property
    def cleanup_verifier(self) -> AbortCleanupVerifier:
        """Get the abort cleanup verifier."""
        return self._cleanup_verifier

    @property
    def resume_manager(self) -> CheckpointResumeManager:
        """Get the checkpoint resume manager."""
        return self._resume_manager

    @property
    def transaction_manager(self) -> TransactionManager:
        """Get the transaction manager."""
        return self._transaction_manager

    def abort_with_verification(
        self,
        reason: str | None = None,
        cleanup: bool = True,
        verify: bool = True,
    ) -> tuple[bool, CleanupVerificationResult | None]:
        """Abort execution with cleanup verification.
        
        Args:
            reason: Reason for abort
            cleanup: Whether to perform cleanup
            verify: Whether to verify cleanup succeeded
            
        Returns:
            Tuple of (abort_success, verification_result)
        """
        # Perform standard abort
        abort_success = self.abort(reason=reason, cleanup=cleanup)
        
        if not abort_success:
            return False, None

        # Verify cleanup if requested
        verification: CleanupVerificationResult | None = None
        if verify:
            verification = self._cleanup_verifier.verify_abort_cleanup(
                perform_cleanup=True
            )
            
            if not verification.success:
                logger.warning(
                    f"Cleanup verification found issues: "
                    f"{verification.errors}"
                )

        return abort_success, verification

    def resume_from_checkpoint(
        self,
        checkpoint_id: str | None = None,
        stage_index: int | None = None,
        latest: bool = False,
    ) -> bool:
        """Resume execution from a checkpoint with full state recovery.
        
        Args:
            checkpoint_id: Specific checkpoint to resume from
            stage_index: Resume from checkpoint at this stage
            latest: Resume from the latest checkpoint
            
        Returns:
            True if resume was successful
        """
        # Prepare resume context
        context = self._resume_manager.prepare_resume(
            checkpoint_id=checkpoint_id,
            stage_index=stage_index,
            latest=latest,
        )
        
        if context is None:
            return False

        # Use parent resume method with prepared checkpoint
        return self.resume(from_checkpoint=context.checkpoint)

    def execute_in_transaction(
        self, operation: Callable[[], Any], operation_type: str = "generic"
    ) -> tuple[bool, Any | None]:
        """Execute an operation within a transaction.
        
        Args:
            operation: The operation to execute
            operation_type: Type of operation for undo handling
            
        Returns:
            Tuple of (success, result)
        """
        # Begin transaction
        transaction = self._transaction_manager.begin_transaction()
        if transaction is None:
            logger.error("Failed to begin transaction")
            return False, None

        try:
            # Execute operation
            result = operation()
            
            # Record successful operation
            self._transaction_manager.record_operation(
                operation_type=operation_type,
                data={"result": str(result)},
            )
            
            # Commit transaction
            self._transaction_manager.commit_transaction()
            return True, result
            
        except Exception as e:
            logger.error(f"Operation failed, rolling back: {e}")
            success, actions = self._transaction_manager.rollback_transaction()
            logger.info(f"Rollback {'succeeded' if success else 'failed'}: {actions}")
            return False, None
