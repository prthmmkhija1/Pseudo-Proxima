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
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import logging

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
    IDLE = auto()       # Not started yet
    RUNNING = auto()    # Actively executing
    PAUSED = auto()     # Paused, waiting for resume
    ABORTED = auto()    # Aborted by user or error
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
    reason: Optional[str] = None
    checkpoint_id: Optional[str] = None

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
    custom_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointData:
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
        checkpoint_dir: Optional[Path] = None,
        execution_id: Optional[str] = None,
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
        
        self._checkpoints: List[CheckpointData] = []
        self._current_checkpoint: Optional[CheckpointData] = None
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
        custom_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            logger.debug(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    # -------------------------------------------------------------------------
    # Checkpoint Loading (On Resume)
    # -------------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
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
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
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

    def get_latest_checkpoint(self) -> Optional[CheckpointData]:
        """Get the most recent checkpoint."""
        with self._lock:
            return self._current_checkpoint

    def list_checkpoints(self) -> List[CheckpointData]:
        """Get all checkpoints in order."""
        with self._lock:
            return list(self._checkpoints)

    # -------------------------------------------------------------------------
    # Checkpoint Cleanup (On Completion)
    # -------------------------------------------------------------------------

    def get_checkpoint_by_stage(self, stage_index: int) -> Optional[CheckpointData]:
        """Get checkpoint at or before a specific stage index."""
        with self._lock:
            for checkpoint in reversed(self._checkpoints):
                if checkpoint.stage_index <= stage_index:
                    return checkpoint
            return None

    def get_previous_checkpoint(self) -> Optional[CheckpointData]:
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
                    checkpoint_path = self._get_checkpoint_path(checkpoint.checkpoint_id)
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
        checkpoint_dir: Optional[Path] = None,
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
        self._callbacks: List[ControlCallback] = []
        
        # Reasons
        self._abort_reason: Optional[str] = None
        self._pause_reason: Optional[str] = None
        
        # Checkpoint management
        self._auto_checkpoint = auto_checkpoint
        self._checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        
        # Execution tracking
        self._start_time: Optional[float] = None
        self._current_stage: Optional[str] = None
        self._stage_index: int = 0
        self._total_stages: int = 1
        self._progress_percent: float = 0.0
        self._custom_state: Dict[str, Any] = {}
        
        # Event history
        self._events: List[ControlEvent] = []

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
    def abort_reason(self) -> Optional[str]:
        """Get abort reason if aborted."""
        return self._abort_reason

    @property
    def pause_reason(self) -> Optional[str]:
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
    def events(self) -> List[ControlEvent]:
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
        resume_from: Optional[CheckpointData] = None,
    ) -> bool:
        """Start execution.
        
        Args:
            total_stages: Total number of stages in execution
            resume_from: Optional checkpoint to resume from
            
        Returns:
            True if started successfully
        """
        with self._lock:
            if self._state not in (ControlState.IDLE, ControlState.COMPLETED, ControlState.ABORTED):
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
            reason="Execution started" if not resume_from else "Resumed from checkpoint",
            checkpoint_id=checkpoint_id,
        )
        self._emit_event(event)
        
        logger.info(f"Execution started (total_stages={total_stages})")
        return True

    # -------------------------------------------------------------------------
    # ABORT: Set abort flag, cleanup, transition to ABORTED
    # -------------------------------------------------------------------------

    def abort(self, reason: Optional[str] = None, cleanup: bool = True) -> bool:
        """Abort execution.
        
        Args:
            reason: Reason for abort
            cleanup: Whether to cleanup checkpoints
            
        Returns:
            True if state changed
        """
        with self._lock:
            if self._state in (ControlState.ABORTED, ControlState.COMPLETED, ControlState.IDLE):
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
        reason: Optional[str] = None,
        create_checkpoint: bool = True,
    ) -> Optional[CheckpointData]:
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

    def resume(self, from_checkpoint: Optional[CheckpointData] = None) -> bool:
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
        checkpoint_id: Optional[str] = None,
        stage_index: Optional[int] = None,
    ) -> Optional[CheckpointData]:
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
        target_checkpoint: Optional[CheckpointData] = None
        
        if checkpoint_id:
            # Load specific checkpoint by ID
            target_checkpoint = self._checkpoint_manager.load_checkpoint(checkpoint_id)
        elif stage_index is not None:
            # Find checkpoint by stage index
            target_checkpoint = self._checkpoint_manager.get_checkpoint_by_stage(stage_index)
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

    def get_available_rollback_points(self) -> List[CheckpointData]:
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
            if self._state in (ControlState.ABORTED, ControlState.COMPLETED, ControlState.IDLE):
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

    def check_pause(self, timeout: Optional[float] = None) -> bool:
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
        custom_state: Optional[Dict[str, Any]] = None,
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

    def checkpoint_now(self, custom_state: Optional[Dict[str, Any]] = None) -> Optional[CheckpointData]:
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

    def __init__(self, reason: Optional[str] = None) -> None:
        self.reason = reason
        super().__init__(reason or "Execution aborted")


# =============================================================================
# Pause Exception
# =============================================================================

class PauseException(Exception):
    """Raised when execution should pause."""

    def __init__(self, reason: Optional[str] = None, checkpoint: Optional[CheckpointData] = None) -> None:
        self.reason = reason
        self.checkpoint = checkpoint
        super().__init__(reason or "Execution paused")
