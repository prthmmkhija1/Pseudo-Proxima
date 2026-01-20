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


# =============================================================================
# DISTRIBUTED ROLLBACK (2% Gap Coverage)
# =============================================================================


class DistributedNodeState(Enum):
    """State of a node in distributed rollback."""
    
    UNKNOWN = auto()
    READY = auto()
    PREPARING = auto()
    PREPARED = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ROLLING_BACK = auto()
    ROLLED_BACK = auto()
    FAILED = auto()
    UNREACHABLE = auto()


@dataclass
class DistributedNode:
    """Represents a node in distributed computation."""
    
    node_id: str
    address: str  # e.g., "host:port" or "localhost"
    state: DistributedNodeState = DistributedNodeState.UNKNOWN
    last_heartbeat: float = field(default_factory=time.time)
    checkpoint_id: str | None = None
    transaction_id: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    
    def is_healthy(self, timeout_seconds: float = 30.0) -> bool:
        """Check if node is healthy based on last heartbeat."""
        return (time.time() - self.last_heartbeat) < timeout_seconds
    
    def can_retry(self) -> bool:
        """Check if node can be retried."""
        return self.retry_count < self.max_retries


@dataclass
class DistributedRollbackResult:
    """Result of distributed rollback operation."""
    
    success: bool
    nodes_rolled_back: list[str]
    nodes_failed: list[str]
    nodes_unreachable: list[str]
    partial_rollback: bool = False
    error_messages: dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0
    coordinator_id: str = ""
    transaction_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "nodes_rolled_back": self.nodes_rolled_back,
            "nodes_failed": self.nodes_failed,
            "nodes_unreachable": self.nodes_unreachable,
            "partial_rollback": self.partial_rollback,
            "error_messages": self.error_messages,
            "duration_ms": self.duration_ms,
            "coordinator_id": self.coordinator_id,
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
        }


class TwoPhaseCommitProtocol(Enum):
    """Two-phase commit protocol phases."""
    
    IDLE = auto()
    PREPARE = auto()
    VOTE = auto()
    COMMIT = auto()
    ABORT = auto()
    COMPLETE = auto()


@dataclass
class DistributedTransaction:
    """State of a distributed transaction."""
    
    transaction_id: str
    coordinator_id: str
    participating_nodes: list[str]
    phase: TwoPhaseCommitProtocol = TwoPhaseCommitProtocol.IDLE
    votes: dict[str, bool] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    prepare_deadline: float = 0.0
    commit_deadline: float = 0.0
    rollback_checkpoints: dict[str, str] = field(default_factory=dict)
    is_committed: bool = False
    is_aborted: bool = False
    
    def all_votes_received(self) -> bool:
        """Check if all votes have been received."""
        return len(self.votes) == len(self.participating_nodes)
    
    def all_votes_yes(self) -> bool:
        """Check if all votes are yes."""
        return self.all_votes_received() and all(self.votes.values())


class DistributedRollbackCoordinator:
    """Coordinates distributed rollback across multiple nodes.
    
    Implements Two-Phase Commit (2PC) protocol for distributed transactions
    with support for:
    - Network partition handling
    - Node failure recovery
    - Timeout-based decision making
    - Partial rollback for graceful degradation
    """
    
    def __init__(
        self,
        coordinator_id: str | None = None,
        prepare_timeout_ms: float = 5000.0,
        commit_timeout_ms: float = 10000.0,
        heartbeat_interval_ms: float = 1000.0,
    ) -> None:
        """Initialize distributed rollback coordinator.
        
        Args:
            coordinator_id: Unique ID for this coordinator
            prepare_timeout_ms: Timeout for prepare phase
            commit_timeout_ms: Timeout for commit phase
            heartbeat_interval_ms: Interval for node heartbeats
        """
        self._coordinator_id = coordinator_id or f"coord_{int(time.time() * 1000)}"
        self._prepare_timeout_ms = prepare_timeout_ms
        self._commit_timeout_ms = commit_timeout_ms
        self._heartbeat_interval_ms = heartbeat_interval_ms
        
        self._nodes: dict[str, DistributedNode] = {}
        self._active_transaction: DistributedTransaction | None = None
        self._transaction_history: list[DistributedTransaction] = []
        
        self._lock = threading.Lock()
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False
        
        # Callbacks for node communication (to be implemented by user)
        self._node_prepare_callback: Callable[[str, str], bool] | None = None
        self._node_commit_callback: Callable[[str, str], bool] | None = None
        self._node_rollback_callback: Callable[[str, str, str], bool] | None = None
        self._node_heartbeat_callback: Callable[[str], bool] | None = None
    
    def register_callbacks(
        self,
        prepare: Callable[[str, str], bool] | None = None,
        commit: Callable[[str, str], bool] | None = None,
        rollback: Callable[[str, str, str], bool] | None = None,
        heartbeat: Callable[[str], bool] | None = None,
    ) -> None:
        """Register callbacks for node communication.
        
        Args:
            prepare: Callback(node_id, transaction_id) -> success
            commit: Callback(node_id, transaction_id) -> success
            rollback: Callback(node_id, transaction_id, checkpoint_id) -> success
            heartbeat: Callback(node_id) -> is_alive
        """
        if prepare:
            self._node_prepare_callback = prepare
        if commit:
            self._node_commit_callback = commit
        if rollback:
            self._node_rollback_callback = rollback
        if heartbeat:
            self._node_heartbeat_callback = heartbeat
    
    def register_node(
        self,
        node_id: str,
        address: str,
    ) -> DistributedNode:
        """Register a node for distributed transactions."""
        with self._lock:
            node = DistributedNode(
                node_id=node_id,
                address=address,
                state=DistributedNodeState.READY,
            )
            self._nodes[node_id] = node
            logger.info(f"Registered node: {node_id} at {address}")
            return node
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node."""
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info(f"Unregistered node: {node_id}")
                return True
            return False
    
    def get_node(self, node_id: str) -> DistributedNode | None:
        """Get node by ID."""
        return self._nodes.get(node_id)
    
    def get_healthy_nodes(self) -> list[DistributedNode]:
        """Get all healthy nodes."""
        with self._lock:
            return [n for n in self._nodes.values() if n.is_healthy()]
    
    def start_heartbeat_monitor(self) -> None:
        """Start background heartbeat monitoring."""
        if self._running:
            return
        
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()
        logger.info("Started heartbeat monitor")
    
    def stop_heartbeat_monitor(self) -> None:
        """Stop heartbeat monitoring."""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        logger.info("Stopped heartbeat monitor")
    
    def _heartbeat_loop(self) -> None:
        """Background loop for node heartbeats."""
        while self._running:
            try:
                self._check_node_health()
            except Exception as e:
                logger.warning(f"Heartbeat check failed: {e}")
            
            time.sleep(self._heartbeat_interval_ms / 1000)
    
    def _check_node_health(self) -> None:
        """Check health of all nodes."""
        with self._lock:
            for node in self._nodes.values():
                if self._node_heartbeat_callback:
                    try:
                        if self._node_heartbeat_callback(node.node_id):
                            node.last_heartbeat = time.time()
                            if node.state == DistributedNodeState.UNREACHABLE:
                                node.state = DistributedNodeState.READY
                                logger.info(f"Node {node.node_id} recovered")
                        else:
                            if node.is_healthy():
                                node.state = DistributedNodeState.UNREACHABLE
                                logger.warning(f"Node {node.node_id} unreachable")
                    except Exception as e:
                        logger.debug(f"Heartbeat to {node.node_id} failed: {e}")
    
    def begin_distributed_transaction(
        self,
        node_ids: list[str] | None = None,
        checkpoints: dict[str, str] | None = None,
    ) -> DistributedTransaction | None:
        """Begin a distributed transaction.
        
        Args:
            node_ids: Nodes to include (default: all healthy nodes)
            checkpoints: Checkpoint IDs for each node (for rollback)
            
        Returns:
            Transaction if started, None if failed
        """
        with self._lock:
            if self._active_transaction:
                logger.warning("Distributed transaction already active")
                return None
            
            # Select participating nodes
            if node_ids is None:
                participating = [n.node_id for n in self.get_healthy_nodes()]
            else:
                participating = [
                    nid for nid in node_ids 
                    if nid in self._nodes and self._nodes[nid].is_healthy()
                ]
            
            if not participating:
                logger.error("No healthy nodes for distributed transaction")
                return None
            
            transaction_id = f"dtxn_{int(time.time() * 1000)}"
            current_time = time.time()
            
            self._active_transaction = DistributedTransaction(
                transaction_id=transaction_id,
                coordinator_id=self._coordinator_id,
                participating_nodes=participating,
                phase=TwoPhaseCommitProtocol.IDLE,
                prepare_deadline=current_time + self._prepare_timeout_ms / 1000,
                commit_deadline=current_time + self._commit_timeout_ms / 1000,
                rollback_checkpoints=checkpoints or {},
            )
            
            # Update node states
            for node_id in participating:
                self._nodes[node_id].transaction_id = transaction_id
            
            logger.info(
                f"Started distributed transaction {transaction_id} "
                f"with {len(participating)} nodes"
            )
            return self._active_transaction
    
    def prepare_phase(self) -> tuple[bool, dict[str, bool]]:
        """Execute prepare phase of 2PC.
        
        Returns:
            Tuple of (all_prepared, vote_results)
        """
        with self._lock:
            if not self._active_transaction:
                return False, {}
            
            self._active_transaction.phase = TwoPhaseCommitProtocol.PREPARE
            votes: dict[str, bool] = {}
            
            for node_id in self._active_transaction.participating_nodes:
                node = self._nodes.get(node_id)
                if not node or not node.is_healthy():
                    votes[node_id] = False
                    continue
                
                node.state = DistributedNodeState.PREPARING
                
                # Ask node to prepare
                if self._node_prepare_callback:
                    try:
                        prepared = self._node_prepare_callback(
                            node_id,
                            self._active_transaction.transaction_id,
                        )
                        votes[node_id] = prepared
                        node.state = (
                            DistributedNodeState.PREPARED 
                            if prepared 
                            else DistributedNodeState.FAILED
                        )
                    except Exception as e:
                        logger.warning(f"Prepare failed for {node_id}: {e}")
                        votes[node_id] = False
                        node.state = DistributedNodeState.FAILED
                else:
                    # No callback - assume prepared
                    votes[node_id] = True
                    node.state = DistributedNodeState.PREPARED
            
            self._active_transaction.votes = votes
            self._active_transaction.phase = TwoPhaseCommitProtocol.VOTE
            
            all_prepared = all(votes.values())
            return all_prepared, votes
    
    def commit_phase(self) -> DistributedRollbackResult:
        """Execute commit phase of 2PC.
        
        Returns:
            Result of commit operation
        """
        start_time = time.time()
        
        with self._lock:
            if not self._active_transaction:
                return DistributedRollbackResult(
                    success=False,
                    nodes_rolled_back=[],
                    nodes_failed=["coordinator"],
                    nodes_unreachable=[],
                    error_messages={"coordinator": "No active transaction"},
                )
            
            self._active_transaction.phase = TwoPhaseCommitProtocol.COMMIT
            committed_nodes: list[str] = []
            failed_nodes: list[str] = []
            error_messages: dict[str, str] = {}
            
            for node_id in self._active_transaction.participating_nodes:
                node = self._nodes.get(node_id)
                if not node or node.state != DistributedNodeState.PREPARED:
                    failed_nodes.append(node_id)
                    error_messages[node_id] = "Not in prepared state"
                    continue
                
                node.state = DistributedNodeState.COMMITTING
                
                if self._node_commit_callback:
                    try:
                        if self._node_commit_callback(
                            node_id,
                            self._active_transaction.transaction_id,
                        ):
                            node.state = DistributedNodeState.COMMITTED
                            committed_nodes.append(node_id)
                        else:
                            node.state = DistributedNodeState.FAILED
                            failed_nodes.append(node_id)
                            error_messages[node_id] = "Commit returned false"
                    except Exception as e:
                        node.state = DistributedNodeState.FAILED
                        failed_nodes.append(node_id)
                        error_messages[node_id] = str(e)
                else:
                    # No callback - assume committed
                    node.state = DistributedNodeState.COMMITTED
                    committed_nodes.append(node_id)
            
            self._active_transaction.is_committed = len(failed_nodes) == 0
            self._active_transaction.phase = TwoPhaseCommitProtocol.COMPLETE
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = DistributedRollbackResult(
                success=len(failed_nodes) == 0,
                nodes_rolled_back=committed_nodes,
                nodes_failed=failed_nodes,
                nodes_unreachable=[],
                partial_rollback=len(committed_nodes) > 0 and len(failed_nodes) > 0,
                error_messages=error_messages,
                duration_ms=duration_ms,
                coordinator_id=self._coordinator_id,
                transaction_id=self._active_transaction.transaction_id,
            )
            
            # Archive transaction
            self._transaction_history.append(self._active_transaction)
            self._active_transaction = None
            
            return result
    
    def distributed_rollback(
        self,
        force: bool = False,
        partial_allowed: bool = True,
    ) -> DistributedRollbackResult:
        """Perform distributed rollback across all participating nodes.
        
        This is the main entry point for distributed rollback.
        
        Args:
            force: Force rollback even if some nodes are unreachable
            partial_allowed: Allow partial rollback if some nodes fail
            
        Returns:
            Result of rollback operation
        """
        start_time = time.time()
        
        with self._lock:
            if not self._active_transaction:
                return DistributedRollbackResult(
                    success=False,
                    nodes_rolled_back=[],
                    nodes_failed=[],
                    nodes_unreachable=[],
                    error_messages={"coordinator": "No active transaction"},
                )
            
            self._active_transaction.phase = TwoPhaseCommitProtocol.ABORT
            
            rolled_back_nodes: list[str] = []
            failed_nodes: list[str] = []
            unreachable_nodes: list[str] = []
            error_messages: dict[str, str] = {}
            
            for node_id in self._active_transaction.participating_nodes:
                node = self._nodes.get(node_id)
                
                if not node:
                    unreachable_nodes.append(node_id)
                    continue
                
                if not node.is_healthy():
                    if force:
                        unreachable_nodes.append(node_id)
                        node.state = DistributedNodeState.UNREACHABLE
                    else:
                        failed_nodes.append(node_id)
                        error_messages[node_id] = "Node unhealthy, force not set"
                    continue
                
                node.state = DistributedNodeState.ROLLING_BACK
                checkpoint_id = self._active_transaction.rollback_checkpoints.get(
                    node_id, ""
                )
                
                # Attempt rollback with retries
                success = self._rollback_node_with_retry(
                    node,
                    self._active_transaction.transaction_id,
                    checkpoint_id,
                )
                
                if success:
                    node.state = DistributedNodeState.ROLLED_BACK
                    rolled_back_nodes.append(node_id)
                else:
                    node.state = DistributedNodeState.FAILED
                    failed_nodes.append(node_id)
                    error_messages[node_id] = node.error_message or "Rollback failed"
            
            # Determine success
            all_success = len(failed_nodes) == 0 and len(unreachable_nodes) == 0
            partial_success = len(rolled_back_nodes) > 0
            
            if all_success:
                success = True
            elif partial_allowed and partial_success:
                success = True
            else:
                success = False
            
            self._active_transaction.is_aborted = True
            self._active_transaction.phase = TwoPhaseCommitProtocol.COMPLETE
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = DistributedRollbackResult(
                success=success,
                nodes_rolled_back=rolled_back_nodes,
                nodes_failed=failed_nodes,
                nodes_unreachable=unreachable_nodes,
                partial_rollback=partial_success and not all_success,
                error_messages=error_messages,
                duration_ms=duration_ms,
                coordinator_id=self._coordinator_id,
                transaction_id=self._active_transaction.transaction_id,
            )
            
            logger.info(
                f"Distributed rollback complete: "
                f"success={success}, "
                f"rolled_back={len(rolled_back_nodes)}, "
                f"failed={len(failed_nodes)}, "
                f"unreachable={len(unreachable_nodes)}"
            )
            
            # Archive transaction
            self._transaction_history.append(self._active_transaction)
            self._active_transaction = None
            
            return result
    
    def _rollback_node_with_retry(
        self,
        node: DistributedNode,
        transaction_id: str,
        checkpoint_id: str,
    ) -> bool:
        """Attempt to rollback a node with retries."""
        while node.can_retry():
            if self._node_rollback_callback:
                try:
                    if self._node_rollback_callback(
                        node.node_id,
                        transaction_id,
                        checkpoint_id,
                    ):
                        return True
                except Exception as e:
                    node.error_message = str(e)
            else:
                # No callback - assume success
                return True
            
            node.retry_count += 1
            time.sleep(0.1 * (2 ** node.retry_count))  # Exponential backoff
        
        return False
    
    def handle_network_partition(
        self,
        isolated_nodes: list[str],
        strategy: str = "pessimistic",
    ) -> DistributedRollbackResult:
        """Handle network partition during distributed transaction.
        
        Strategies:
        - pessimistic: Abort transaction, rollback reachable nodes
        - optimistic: Continue with reachable nodes, reconcile later
        - quorum: Proceed if majority of nodes are reachable
        
        Args:
            isolated_nodes: Nodes that are isolated
            strategy: Partition handling strategy
            
        Returns:
            Result of partition handling
        """
        with self._lock:
            if not self._active_transaction:
                return DistributedRollbackResult(
                    success=False,
                    nodes_rolled_back=[],
                    nodes_failed=[],
                    nodes_unreachable=isolated_nodes,
                    error_messages={"coordinator": "No active transaction"},
                )
            
            # Mark isolated nodes
            for node_id in isolated_nodes:
                if node_id in self._nodes:
                    self._nodes[node_id].state = DistributedNodeState.UNREACHABLE
            
            reachable = [
                nid for nid in self._active_transaction.participating_nodes
                if nid not in isolated_nodes
            ]
            
            if strategy == "pessimistic":
                # Abort and rollback
                return self.distributed_rollback(force=True, partial_allowed=True)
            
            elif strategy == "quorum":
                # Check if we have quorum
                total = len(self._active_transaction.participating_nodes)
                if len(reachable) > total // 2:
                    logger.info("Quorum maintained, continuing transaction")
                    return DistributedRollbackResult(
                        success=True,
                        nodes_rolled_back=[],
                        nodes_failed=[],
                        nodes_unreachable=isolated_nodes,
                        partial_rollback=False,
                        coordinator_id=self._coordinator_id,
                        transaction_id=self._active_transaction.transaction_id,
                    )
                else:
                    logger.warning("Quorum lost, aborting transaction")
                    return self.distributed_rollback(force=True, partial_allowed=True)
            
            else:  # optimistic
                logger.info(
                    f"Optimistic strategy: continuing with {len(reachable)} nodes"
                )
                return DistributedRollbackResult(
                    success=True,
                    nodes_rolled_back=[],
                    nodes_failed=[],
                    nodes_unreachable=isolated_nodes,
                    partial_rollback=False,
                    coordinator_id=self._coordinator_id,
                    transaction_id=self._active_transaction.transaction_id,
                )
    
    def get_transaction_status(self) -> dict[str, Any]:
        """Get current distributed transaction status."""
        with self._lock:
            if not self._active_transaction:
                return {"active": False}
            
            node_states = {
                nid: self._nodes[nid].state.name 
                for nid in self._active_transaction.participating_nodes
                if nid in self._nodes
            }
            
            return {
                "active": True,
                "transaction_id": self._active_transaction.transaction_id,
                "phase": self._active_transaction.phase.name,
                "participating_nodes": self._active_transaction.participating_nodes,
                "node_states": node_states,
                "votes": self._active_transaction.votes,
                "is_committed": self._active_transaction.is_committed,
                "is_aborted": self._active_transaction.is_aborted,
            }
    
    def get_coordinator_summary(self) -> dict[str, Any]:
        """Get summary of coordinator state."""
        with self._lock:
            return {
                "coordinator_id": self._coordinator_id,
                "registered_nodes": len(self._nodes),
                "healthy_nodes": len([n for n in self._nodes.values() if n.is_healthy()]),
                "active_transaction": self._active_transaction is not None,
                "transaction_history_count": len(self._transaction_history),
                "heartbeat_running": self._running,
            }


# =============================================================================
# TRANSACTION EDGE CASES (2% Gap Coverage)
# Handles nested transactions, timeouts, deadlocks, and partial failures
# =============================================================================


class TransactionIsolationLevel(Enum):
    """Transaction isolation levels."""
    
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


class NestedTransactionMode(Enum):
    """Mode for handling nested transactions."""
    
    FLAT = "flat"  # All operations in single transaction
    SAVEPOINT = "savepoint"  # Use savepoints for nesting
    AUTONOMOUS = "autonomous"  # Separate independent transactions


@dataclass
class TransactionTimeout:
    """Transaction timeout configuration."""
    
    soft_timeout_seconds: float  # Warning threshold
    hard_timeout_seconds: float  # Forced rollback threshold
    auto_rollback: bool = True
    rollback_callback: Callable[[], None] | None = None
    
    def check_soft(self, elapsed: float) -> bool:
        """Check if soft timeout exceeded."""
        return elapsed >= self.soft_timeout_seconds
    
    def check_hard(self, elapsed: float) -> bool:
        """Check if hard timeout exceeded."""
        return elapsed >= self.hard_timeout_seconds


@dataclass
class Savepoint:
    """A transaction savepoint for nested operations."""
    
    savepoint_id: str
    name: str
    parent_savepoint_id: str | None
    created_at: float
    state_snapshot: dict[str, Any]
    is_released: bool = False
    is_rolled_back: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "savepoint_id": self.savepoint_id,
            "name": self.name,
            "parent_savepoint_id": self.parent_savepoint_id,
            "created_at": self.created_at,
            "is_released": self.is_released,
            "is_rolled_back": self.is_rolled_back,
        }


@dataclass
class DeadlockInfo:
    """Information about a detected deadlock."""
    
    detection_time: float
    involved_transactions: list[str]
    wait_graph: dict[str, list[str]]  # Transaction -> transactions it's waiting for
    victim_transaction: str | None
    resolution_strategy: str
    resolved: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detection_time": self.detection_time,
            "involved_transactions": self.involved_transactions,
            "wait_graph": self.wait_graph,
            "victim_transaction": self.victim_transaction,
            "resolution_strategy": self.resolution_strategy,
            "resolved": self.resolved,
        }


class TransactionEdgeCaseResult(Enum):
    """Result of edge case handling."""
    
    RESOLVED = "resolved"
    ROLLBACK_REQUIRED = "rollback_required"
    RETRY_SUGGESTED = "retry_suggested"
    ESCALATION_NEEDED = "escalation_needed"
    PARTIAL_SUCCESS = "partial_success"


@dataclass
class EdgeCaseResolution:
    """Resolution of a transaction edge case."""
    
    edge_case_type: str
    result: TransactionEdgeCaseResult
    message: str
    retry_delay_seconds: float | None = None
    partial_results: dict[str, Any] | None = None
    corrective_action: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_case_type": self.edge_case_type,
            "result": self.result.value,
            "message": self.message,
            "retry_delay_seconds": self.retry_delay_seconds,
            "partial_results": self.partial_results,
            "corrective_action": self.corrective_action,
        }


class NestedTransactionManager:
    """Manages nested transactions with savepoints.
    
    Features:
    - Savepoint-based nesting
    - Selective rollback to savepoints
    - Nested isolation levels
    - Automatic savepoint release
    """
    
    def __init__(
        self,
        mode: NestedTransactionMode = NestedTransactionMode.SAVEPOINT,
        max_depth: int = 10,
    ) -> None:
        """Initialize nested transaction manager.
        
        Args:
            mode: Nested transaction mode
            max_depth: Maximum nesting depth
        """
        self._mode = mode
        self._max_depth = max_depth
        self._lock = threading.Lock()
        
        # Transaction state
        self._active = False
        self._savepoints: list[Savepoint] = []
        self._current_state: dict[str, Any] = {}
        self._savepoint_counter = 0
    
    def begin(self) -> str:
        """Begin or enter a transaction.
        
        Returns:
            Transaction or savepoint ID
        """
        with self._lock:
            if not self._active:
                # Start new root transaction
                self._active = True
                self._current_state = {}
                return "txn_root"
            
            if self._mode == NestedTransactionMode.FLAT:
                # Flat mode - just continue
                return "txn_root"
            
            if self._mode == NestedTransactionMode.SAVEPOINT:
                # Create savepoint
                return self._create_savepoint(f"auto_{len(self._savepoints)}")
            
            # Autonomous mode - not supported in this simplified version
            return "txn_autonomous"
    
    def _create_savepoint(self, name: str) -> str:
        """Create a new savepoint.
        
        Args:
            name: Savepoint name
            
        Returns:
            Savepoint ID
        """
        if len(self._savepoints) >= self._max_depth:
            raise ValueError(f"Maximum nesting depth ({self._max_depth}) exceeded")
        
        self._savepoint_counter += 1
        savepoint_id = f"sp_{self._savepoint_counter}"
        
        parent_id = self._savepoints[-1].savepoint_id if self._savepoints else None
        
        savepoint = Savepoint(
            savepoint_id=savepoint_id,
            name=name,
            parent_savepoint_id=parent_id,
            created_at=time.time(),
            state_snapshot=self._current_state.copy(),
        )
        
        self._savepoints.append(savepoint)
        return savepoint_id
    
    def savepoint(self, name: str) -> str:
        """Create a named savepoint.
        
        Args:
            name: Savepoint name
            
        Returns:
            Savepoint ID
        """
        with self._lock:
            if not self._active:
                raise ValueError("No active transaction")
            
            return self._create_savepoint(name)
    
    def rollback_to_savepoint(self, savepoint_id: str) -> bool:
        """Rollback to a specific savepoint.
        
        Args:
            savepoint_id: ID of savepoint to rollback to
            
        Returns:
            True if successful
        """
        with self._lock:
            # Find savepoint
            target_idx = None
            for i, sp in enumerate(self._savepoints):
                if sp.savepoint_id == savepoint_id:
                    target_idx = i
                    break
            
            if target_idx is None:
                return False
            
            # Restore state from savepoint
            target = self._savepoints[target_idx]
            self._current_state = target.state_snapshot.copy()
            
            # Mark later savepoints as rolled back
            for sp in self._savepoints[target_idx + 1:]:
                sp.is_rolled_back = True
            
            # Remove rolled back savepoints
            self._savepoints = self._savepoints[:target_idx + 1]
            
            return True
    
    def release_savepoint(self, savepoint_id: str) -> bool:
        """Release a savepoint, merging changes up.
        
        Args:
            savepoint_id: ID of savepoint to release
            
        Returns:
            True if successful
        """
        with self._lock:
            for sp in self._savepoints:
                if sp.savepoint_id == savepoint_id:
                    sp.is_released = True
                    return True
            return False
    
    def commit(self) -> bool:
        """Commit the transaction.
        
        Returns:
            True if successful
        """
        with self._lock:
            if not self._active:
                return False
            
            # Release all savepoints
            for sp in self._savepoints:
                sp.is_released = True
            
            self._active = False
            self._savepoints = []
            return True
    
    def rollback(self) -> bool:
        """Rollback entire transaction.
        
        Returns:
            True if successful
        """
        with self._lock:
            if not self._active:
                return False
            
            # Clear state
            self._current_state = {}
            
            # Mark all savepoints as rolled back
            for sp in self._savepoints:
                sp.is_rolled_back = True
            
            self._active = False
            self._savepoints = []
            return True
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a value in transaction state.
        
        Args:
            key: State key
            value: State value
        """
        with self._lock:
            self._current_state[key] = value
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value from transaction state.
        
        Args:
            key: State key
            default: Default value if not found
            
        Returns:
            State value
        """
        with self._lock:
            return self._current_state.get(key, default)
    
    def get_current_depth(self) -> int:
        """Get current nesting depth.
        
        Returns:
            Number of active savepoints
        """
        with self._lock:
            return len(self._savepoints)
    
    def get_savepoints(self) -> list[dict[str, Any]]:
        """Get list of active savepoints.
        
        Returns:
            List of savepoint information
        """
        with self._lock:
            return [sp.to_dict() for sp in self._savepoints if not sp.is_released and not sp.is_rolled_back]


class TransactionTimeoutManager:
    """Manages transaction timeouts with automatic rollback.
    
    Features:
    - Soft timeout warnings
    - Hard timeout with auto-rollback
    - Configurable callbacks
    - Timeout extension
    """
    
    def __init__(self) -> None:
        """Initialize timeout manager."""
        self._lock = threading.Lock()
        self._timeouts: dict[str, tuple[TransactionTimeout, float]] = {}  # txn_id -> (config, start_time)
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._timeout_callbacks: dict[str, Callable[[str, bool], None]] = {}  # txn_id -> callback
    
    def register_timeout(
        self,
        transaction_id: str,
        timeout: TransactionTimeout,
        callback: Callable[[str, bool], None] | None = None,
    ) -> None:
        """Register a timeout for a transaction.
        
        Args:
            transaction_id: Transaction ID
            timeout: Timeout configuration
            callback: Callback for timeout events (txn_id, is_hard_timeout)
        """
        with self._lock:
            self._timeouts[transaction_id] = (timeout, time.time())
            if callback:
                self._timeout_callbacks[transaction_id] = callback
    
    def unregister_timeout(self, transaction_id: str) -> None:
        """Unregister a transaction timeout.
        
        Args:
            transaction_id: Transaction ID
        """
        with self._lock:
            self._timeouts.pop(transaction_id, None)
            self._timeout_callbacks.pop(transaction_id, None)
    
    def extend_timeout(
        self,
        transaction_id: str,
        additional_seconds: float,
    ) -> bool:
        """Extend a transaction's timeout.
        
        Args:
            transaction_id: Transaction ID
            additional_seconds: Additional time to add
            
        Returns:
            True if extended
        """
        with self._lock:
            if transaction_id not in self._timeouts:
                return False
            
            timeout, start_time = self._timeouts[transaction_id]
            # Effectively extend by moving start time forward
            new_start = start_time + additional_seconds
            self._timeouts[transaction_id] = (timeout, new_start)
            return True
    
    def check_timeout(self, transaction_id: str) -> tuple[bool, bool]:
        """Check timeout status for a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Tuple of (soft_timeout_exceeded, hard_timeout_exceeded)
        """
        with self._lock:
            if transaction_id not in self._timeouts:
                return (False, False)
            
            timeout, start_time = self._timeouts[transaction_id]
            elapsed = time.time() - start_time
            
            return (timeout.check_soft(elapsed), timeout.check_hard(elapsed))
    
    def start_monitoring(self, check_interval: float = 1.0) -> None:
        """Start background timeout monitoring.
        
        Args:
            check_interval: Interval between checks in seconds
        """
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        def monitor():
            while not self._stop_event.is_set():
                self._check_all_timeouts()
                self._stop_event.wait(check_interval)
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_event.set()
        self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def _check_all_timeouts(self) -> None:
        """Check all registered timeouts."""
        with self._lock:
            now = time.time()
            expired = []
            
            for txn_id, (timeout, start_time) in self._timeouts.items():
                elapsed = now - start_time
                
                if timeout.check_hard(elapsed):
                    # Hard timeout - trigger callback
                    callback = self._timeout_callbacks.get(txn_id)
                    if callback:
                        try:
                            callback(txn_id, True)  # True = hard timeout
                        except Exception:
                            pass
                    
                    if timeout.rollback_callback:
                        try:
                            timeout.rollback_callback()
                        except Exception:
                            pass
                    
                    expired.append(txn_id)
                
                elif timeout.check_soft(elapsed):
                    # Soft timeout - warning
                    callback = self._timeout_callbacks.get(txn_id)
                    if callback:
                        try:
                            callback(txn_id, False)  # False = soft timeout
                        except Exception:
                            pass
            
            # Remove expired timeouts
            for txn_id in expired:
                self._timeouts.pop(txn_id, None)
                self._timeout_callbacks.pop(txn_id, None)


class DeadlockDetector:
    """Detects and resolves transaction deadlocks.
    
    Uses wait-for graph analysis to detect cycles.
    
    Features:
    - Wait-for graph tracking
    - Cycle detection
    - Victim selection strategies
    - Automatic resolution
    """
    
    def __init__(
        self,
        detection_interval: float = 1.0,
        victim_strategy: str = "youngest",  # youngest, oldest, least_work
    ) -> None:
        """Initialize deadlock detector.
        
        Args:
            detection_interval: Interval between detection runs
            victim_strategy: Strategy for selecting deadlock victim
        """
        self._interval = detection_interval
        self._strategy = victim_strategy
        self._lock = threading.Lock()
        
        # Wait-for graph: transaction -> set of transactions it waits for
        self._wait_graph: dict[str, set[str]] = {}
        
        # Transaction metadata for victim selection
        self._transaction_meta: dict[str, dict[str, Any]] = {}
        
        # History of detected deadlocks
        self._deadlock_history: list[DeadlockInfo] = []
        
        # Monitoring
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        # Callbacks
        self._deadlock_callback: Callable[[DeadlockInfo], None] | None = None
    
    def register_transaction(
        self,
        transaction_id: str,
        start_time: float | None = None,
        work_units: int = 0,
    ) -> None:
        """Register a transaction for deadlock detection.
        
        Args:
            transaction_id: Transaction ID
            start_time: When transaction started
            work_units: Amount of work done (for victim selection)
        """
        with self._lock:
            self._wait_graph[transaction_id] = set()
            self._transaction_meta[transaction_id] = {
                "start_time": start_time or time.time(),
                "work_units": work_units,
            }
    
    def unregister_transaction(self, transaction_id: str) -> None:
        """Unregister a transaction.
        
        Args:
            transaction_id: Transaction ID
        """
        with self._lock:
            self._wait_graph.pop(transaction_id, None)
            self._transaction_meta.pop(transaction_id, None)
            
            # Remove from others' wait sets
            for waits in self._wait_graph.values():
                waits.discard(transaction_id)
    
    def add_wait(self, waiter: str, waiting_for: str) -> None:
        """Record that a transaction is waiting for another.
        
        Args:
            waiter: Transaction that is waiting
            waiting_for: Transaction being waited for
        """
        with self._lock:
            if waiter in self._wait_graph:
                self._wait_graph[waiter].add(waiting_for)
    
    def remove_wait(self, waiter: str, waiting_for: str) -> None:
        """Remove a wait relationship.
        
        Args:
            waiter: Transaction that was waiting
            waiting_for: Transaction that was waited for
        """
        with self._lock:
            if waiter in self._wait_graph:
                self._wait_graph[waiter].discard(waiting_for)
    
    def set_deadlock_callback(
        self,
        callback: Callable[[DeadlockInfo], None],
    ) -> None:
        """Set callback for deadlock detection.
        
        Args:
            callback: Function to call when deadlock detected
        """
        self._deadlock_callback = callback
    
    def detect_deadlock(self) -> DeadlockInfo | None:
        """Detect deadlock in current wait graph.
        
        Returns:
            DeadlockInfo if deadlock found, None otherwise
        """
        with self._lock:
            cycle = self._find_cycle()
            
            if not cycle:
                return None
            
            # Build wait graph for just the cycle
            cycle_graph = {
                txn: list(self._wait_graph.get(txn, set()) & set(cycle))
                for txn in cycle
            }
            
            # Select victim
            victim = self._select_victim(cycle)
            
            deadlock = DeadlockInfo(
                detection_time=time.time(),
                involved_transactions=cycle,
                wait_graph=cycle_graph,
                victim_transaction=victim,
                resolution_strategy=self._strategy,
            )
            
            self._deadlock_history.append(deadlock)
            
            return deadlock
    
    def _find_cycle(self) -> list[str] | None:
        """Find a cycle in the wait-for graph using DFS.
        
        Returns:
            List of transaction IDs in cycle, or None
        """
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> list[str] | None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._wait_graph.get(node, set()):
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle - extract it
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]
            
            path.pop()
            rec_stack.remove(node)
            return None
        
        for node in self._wait_graph:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle
        
        return None
    
    def _select_victim(self, cycle: list[str]) -> str:
        """Select a victim transaction to abort.
        
        Args:
            cycle: List of transactions in deadlock
            
        Returns:
            Transaction ID to abort
        """
        if not cycle:
            return ""
        
        if self._strategy == "youngest":
            # Abort the most recently started transaction
            return max(
                cycle,
                key=lambda t: self._transaction_meta.get(t, {}).get("start_time", 0)
            )
        
        elif self._strategy == "oldest":
            # Abort the oldest transaction
            return min(
                cycle,
                key=lambda t: self._transaction_meta.get(t, {}).get("start_time", float("inf"))
            )
        
        elif self._strategy == "least_work":
            # Abort the transaction with least work done
            return min(
                cycle,
                key=lambda t: self._transaction_meta.get(t, {}).get("work_units", 0)
            )
        
        # Default: first in cycle
        return cycle[0]
    
    def start_detection(self) -> None:
        """Start background deadlock detection."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        def detect_loop():
            while not self._stop_event.is_set():
                deadlock = self.detect_deadlock()
                
                if deadlock and self._deadlock_callback:
                    try:
                        self._deadlock_callback(deadlock)
                    except Exception:
                        pass
                
                self._stop_event.wait(self._interval)
        
        self._monitor_thread = threading.Thread(target=detect_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_detection(self) -> None:
        """Stop background detection."""
        self._stop_event.set()
        self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def get_deadlock_history(self) -> list[dict[str, Any]]:
        """Get history of detected deadlocks.
        
        Returns:
            List of deadlock information
        """
        with self._lock:
            return [d.to_dict() for d in self._deadlock_history]


class PartialFailureHandler:
    """Handles partial transaction failures.
    
    When some operations in a transaction succeed and others fail,
    this handler determines the appropriate response.
    
    Features:
    - Partial commit support
    - Compensation transactions
    - Result aggregation
    - Recovery suggestions
    """
    
    def __init__(
        self,
        allow_partial_commit: bool = False,
        min_success_ratio: float = 0.5,
    ) -> None:
        """Initialize partial failure handler.
        
        Args:
            allow_partial_commit: Whether to allow partial commits
            min_success_ratio: Minimum success ratio for partial commit
        """
        self._allow_partial = allow_partial_commit
        self._min_ratio = min_success_ratio
        self._lock = threading.Lock()
        
        # Operation results
        self._operations: dict[str, list[tuple[str, bool, Any]]] = {}  # txn_id -> [(op_id, success, result)]
        
        # Compensation actions
        self._compensations: dict[str, list[Callable[[], None]]] = {}  # txn_id -> [compensation_fn]
    
    def register_transaction(self, transaction_id: str) -> None:
        """Register a transaction for failure handling.
        
        Args:
            transaction_id: Transaction ID
        """
        with self._lock:
            self._operations[transaction_id] = []
            self._compensations[transaction_id] = []
    
    def record_operation(
        self,
        transaction_id: str,
        operation_id: str,
        success: bool,
        result: Any = None,
        compensation: Callable[[], None] | None = None,
    ) -> None:
        """Record an operation result.
        
        Args:
            transaction_id: Transaction ID
            operation_id: Operation identifier
            success: Whether operation succeeded
            result: Operation result or error
            compensation: Function to undo this operation if needed
        """
        with self._lock:
            if transaction_id in self._operations:
                self._operations[transaction_id].append((operation_id, success, result))
                
                if compensation and success:
                    self._compensations[transaction_id].append(compensation)
    
    def handle_partial_failure(
        self,
        transaction_id: str,
    ) -> EdgeCaseResolution:
        """Handle a partial failure scenario.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            EdgeCaseResolution with recommended action
        """
        with self._lock:
            operations = self._operations.get(transaction_id, [])
            
            if not operations:
                return EdgeCaseResolution(
                    edge_case_type="partial_failure",
                    result=TransactionEdgeCaseResult.RESOLVED,
                    message="No operations recorded",
                )
            
            successes = [op for op in operations if op[1]]
            failures = [op for op in operations if not op[1]]
            
            success_ratio = len(successes) / len(operations)
            
            if not failures:
                # All succeeded
                return EdgeCaseResolution(
                    edge_case_type="partial_failure",
                    result=TransactionEdgeCaseResult.RESOLVED,
                    message="All operations succeeded",
                )
            
            if not successes:
                # All failed
                return EdgeCaseResolution(
                    edge_case_type="partial_failure",
                    result=TransactionEdgeCaseResult.ROLLBACK_REQUIRED,
                    message="All operations failed",
                )
            
            # Partial success
            if self._allow_partial and success_ratio >= self._min_ratio:
                return EdgeCaseResolution(
                    edge_case_type="partial_failure",
                    result=TransactionEdgeCaseResult.PARTIAL_SUCCESS,
                    message=f"Partial success: {len(successes)}/{len(operations)} operations succeeded",
                    partial_results={
                        "succeeded": [op[0] for op in successes],
                        "failed": [op[0] for op in failures],
                        "success_ratio": success_ratio,
                    },
                )
            
            # Need to rollback
            return EdgeCaseResolution(
                edge_case_type="partial_failure",
                result=TransactionEdgeCaseResult.ROLLBACK_REQUIRED,
                message=f"Partial failure below threshold: {success_ratio:.1%} < {self._min_ratio:.1%}",
                corrective_action="Execute compensating transactions for succeeded operations",
                partial_results={
                    "to_compensate": [op[0] for op in successes],
                    "already_failed": [op[0] for op in failures],
                },
            )
    
    def execute_compensations(self, transaction_id: str) -> list[tuple[int, bool]]:
        """Execute compensation actions for a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            List of (index, success) for each compensation
        """
        with self._lock:
            compensations = self._compensations.get(transaction_id, [])
        
        results = []
        
        # Execute in reverse order
        for i, comp in enumerate(reversed(compensations)):
            try:
                comp()
                results.append((len(compensations) - 1 - i, True))
            except Exception:
                results.append((len(compensations) - 1 - i, False))
        
        return results
    
    def cleanup(self, transaction_id: str) -> None:
        """Clean up tracking for a transaction.
        
        Args:
            transaction_id: Transaction ID
        """
        with self._lock:
            self._operations.pop(transaction_id, None)
            self._compensations.pop(transaction_id, None)


class TransactionEdgeCaseManager:
    """Comprehensive manager for transaction edge cases.
    
    Combines:
    - Nested transactions
    - Timeouts
    - Deadlock detection
    - Partial failure handling
    """
    
    def __init__(
        self,
        nested_mode: NestedTransactionMode = NestedTransactionMode.SAVEPOINT,
        default_timeout: TransactionTimeout | None = None,
        deadlock_detection: bool = True,
        allow_partial_commits: bool = False,
    ) -> None:
        """Initialize edge case manager.
        
        Args:
            nested_mode: Mode for nested transactions
            default_timeout: Default timeout configuration
            deadlock_detection: Whether to enable deadlock detection
            allow_partial_commits: Whether to allow partial commits
        """
        self._nested_mgr = NestedTransactionManager(mode=nested_mode)
        self._timeout_mgr = TransactionTimeoutManager()
        self._deadlock_detector = DeadlockDetector() if deadlock_detection else None
        self._failure_handler = PartialFailureHandler(allow_partial_commit=allow_partial_commits)
        
        self._default_timeout = default_timeout
        self._lock = threading.Lock()
        
        # Active transactions
        self._transactions: dict[str, dict[str, Any]] = {}
        self._txn_counter = 0
    
    def start_transaction(
        self,
        timeout: TransactionTimeout | None = None,
        isolation_level: TransactionIsolationLevel = TransactionIsolationLevel.READ_COMMITTED,
    ) -> str:
        """Start a new transaction with edge case handling.
        
        Args:
            timeout: Optional timeout configuration
            isolation_level: Isolation level for this transaction
            
        Returns:
            Transaction ID
        """
        with self._lock:
            self._txn_counter += 1
            txn_id = f"txn_{self._txn_counter}_{int(time.time())}"
            
            # Set up nested transaction
            nested_id = self._nested_mgr.begin()
            
            # Register timeout
            effective_timeout = timeout or self._default_timeout
            if effective_timeout:
                self._timeout_mgr.register_timeout(
                    txn_id,
                    effective_timeout,
                    callback=self._on_timeout,
                )
            
            # Register for deadlock detection
            if self._deadlock_detector:
                self._deadlock_detector.register_transaction(txn_id)
            
            # Register for failure handling
            self._failure_handler.register_transaction(txn_id)
            
            self._transactions[txn_id] = {
                "nested_id": nested_id,
                "isolation_level": isolation_level,
                "start_time": time.time(),
                "status": "active",
            }
            
            return txn_id
    
    def _on_timeout(self, txn_id: str, is_hard: bool) -> None:
        """Handle timeout callback."""
        if is_hard:
            # Force rollback
            self.rollback_transaction(txn_id)
    
    def record_operation(
        self,
        transaction_id: str,
        operation_id: str,
        success: bool,
        result: Any = None,
        compensation: Callable[[], None] | None = None,
    ) -> None:
        """Record an operation within a transaction.
        
        Args:
            transaction_id: Transaction ID
            operation_id: Operation identifier
            success: Whether operation succeeded
            result: Operation result
            compensation: Compensation function
        """
        self._failure_handler.record_operation(
            transaction_id, operation_id, success, result, compensation
        )
        
        # Update work units for deadlock victim selection
        if self._deadlock_detector:
            with self._lock:
                if transaction_id in self._transactions:
                    work = self._transactions[transaction_id].get("work_units", 0)
                    self._transactions[transaction_id]["work_units"] = work + 1
    
    def create_savepoint(self, transaction_id: str, name: str) -> str:
        """Create a savepoint in a transaction.
        
        Args:
            transaction_id: Transaction ID
            name: Savepoint name
            
        Returns:
            Savepoint ID
        """
        with self._lock:
            if transaction_id not in self._transactions:
                raise ValueError(f"Unknown transaction: {transaction_id}")
        
        return self._nested_mgr.savepoint(name)
    
    def rollback_to_savepoint(
        self,
        transaction_id: str,
        savepoint_id: str,
    ) -> bool:
        """Rollback to a savepoint.
        
        Args:
            transaction_id: Transaction ID
            savepoint_id: Savepoint ID
            
        Returns:
            True if successful
        """
        with self._lock:
            if transaction_id not in self._transactions:
                return False
        
        return self._nested_mgr.rollback_to_savepoint(savepoint_id)
    
    def commit_transaction(self, transaction_id: str) -> EdgeCaseResolution:
        """Commit a transaction with edge case handling.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            EdgeCaseResolution describing the outcome
        """
        with self._lock:
            if transaction_id not in self._transactions:
                return EdgeCaseResolution(
                    edge_case_type="commit",
                    result=TransactionEdgeCaseResult.ROLLBACK_REQUIRED,
                    message=f"Unknown transaction: {transaction_id}",
                )
        
        # Check for partial failures
        failure_result = self._failure_handler.handle_partial_failure(transaction_id)
        
        if failure_result.result == TransactionEdgeCaseResult.ROLLBACK_REQUIRED:
            # Execute compensations and rollback
            self._failure_handler.execute_compensations(transaction_id)
            self._nested_mgr.rollback()
            self._cleanup_transaction(transaction_id)
            return failure_result
        
        # Commit
        self._nested_mgr.commit()
        self._cleanup_transaction(transaction_id)
        
        if failure_result.result == TransactionEdgeCaseResult.PARTIAL_SUCCESS:
            return failure_result
        
        return EdgeCaseResolution(
            edge_case_type="commit",
            result=TransactionEdgeCaseResult.RESOLVED,
            message="Transaction committed successfully",
        )
    
    def rollback_transaction(self, transaction_id: str) -> EdgeCaseResolution:
        """Rollback a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            EdgeCaseResolution describing the outcome
        """
        with self._lock:
            if transaction_id not in self._transactions:
                return EdgeCaseResolution(
                    edge_case_type="rollback",
                    result=TransactionEdgeCaseResult.RESOLVED,
                    message=f"Transaction not found: {transaction_id}",
                )
        
        # Execute compensations
        comp_results = self._failure_handler.execute_compensations(transaction_id)
        
        # Rollback nested
        self._nested_mgr.rollback()
        
        self._cleanup_transaction(transaction_id)
        
        failed_comps = [r for r in comp_results if not r[1]]
        if failed_comps:
            return EdgeCaseResolution(
                edge_case_type="rollback",
                result=TransactionEdgeCaseResult.PARTIAL_SUCCESS,
                message=f"Rollback completed with {len(failed_comps)} compensation failures",
            )
        
        return EdgeCaseResolution(
            edge_case_type="rollback",
            result=TransactionEdgeCaseResult.RESOLVED,
            message="Transaction rolled back successfully",
        )
    
    def _cleanup_transaction(self, transaction_id: str) -> None:
        """Clean up transaction resources."""
        with self._lock:
            self._transactions.pop(transaction_id, None)
        
        self._timeout_mgr.unregister_timeout(transaction_id)
        
        if self._deadlock_detector:
            self._deadlock_detector.unregister_transaction(transaction_id)
        
        self._failure_handler.cleanup(transaction_id)
    
    def start_monitoring(self) -> None:
        """Start background monitoring for timeouts and deadlocks."""
        self._timeout_mgr.start_monitoring()
        
        if self._deadlock_detector:
            self._deadlock_detector.start_detection()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._timeout_mgr.stop_monitoring()
        
        if self._deadlock_detector:
            self._deadlock_detector.stop_detection()
    
    def get_transaction_status(self, transaction_id: str) -> dict[str, Any]:
        """Get status of a transaction.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Status information
        """
        with self._lock:
            txn = self._transactions.get(transaction_id, {})
            
            return {
                "transaction_id": transaction_id,
                "status": txn.get("status", "unknown"),
                "isolation_level": txn.get("isolation_level", TransactionIsolationLevel.READ_COMMITTED).value,
                "nesting_depth": self._nested_mgr.get_current_depth(),
                "savepoints": self._nested_mgr.get_savepoints(),
                "timeout_status": self._timeout_mgr.check_timeout(transaction_id),
            }

