"""Enhanced Execution Timer implementation (Phase 4, Step 4.2).

Provides:
- ExecutionTimer: Global and per-stage timing with display updates
- ProgressTracker: Percentage completion with 10% increment updates
- ETACalculator: Estimated time remaining with smoothing
- DisplayController: Batched display updates to avoid flicker

Timer Components (Step 4.2):
├── GlobalTimer - Total elapsed since start
├── StageTimer - Per-stage elapsed times
├── ETACalculator - Estimated time remaining
└── ProgressTracker - Percentage completion

Display Update Strategy:
- Update every 100ms for active stages
- Update on stage transitions
- Update on significant progress (10% increments)
- Batch updates to avoid flicker
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# Stage Information
# =============================================================================


@dataclass
class StageInfo:
    """Information about a single execution stage."""

    name: str
    start_time: float
    end_time: float | None = None
    weight: float = 1.0  # For weighted progress calculation

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        end = self.end_time if self.end_time else time.perf_counter()
        return (end - self.start_time) * 1000

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ms / 1000

    @property
    def is_complete(self) -> bool:
        """Check if stage is complete."""
        return self.end_time is not None

    def __str__(self) -> str:
        status = "✓" if self.is_complete else "..."
        return f"[{status}] {self.name}: {self.elapsed_ms:.0f}ms"


# =============================================================================
# Display Update Types
# =============================================================================


class UpdateReason(Enum):
    """Reason for display update."""

    TIMER_TICK = auto()  # Regular 100ms tick
    STAGE_TRANSITION = auto()  # Stage started or ended
    PROGRESS_MILESTONE = auto()  # 10% progress increment
    MANUAL = auto()  # Manual update request
    COMPLETION = auto()  # Execution completed


@dataclass
class DisplayUpdate:
    """Batched display update information."""

    timestamp: float
    reason: UpdateReason
    elapsed_ms: float
    current_stage: str | None
    progress_percent: float
    eta_display: str
    message: str | None = None

    def format_line(self) -> str:
        """Format as a single display line."""
        stage = self.current_stage or "idle"
        elapsed_sec = self.elapsed_ms / 1000
        return (
            f"[{stage}] {elapsed_sec:.1f}s | {self.progress_percent:.0f}% | ETA: {self.eta_display}"
        )


# =============================================================================
# Display Controller (Batched Updates)
# =============================================================================

# Type alias for display callbacks
DisplayCallback = Callable[[DisplayUpdate], None]


class DisplayController:
    """Controls display updates with batching to avoid flicker.

    Implements Step 4.2 Display Update Strategy:
    - Update every 100ms for active stages
    - Update on stage transitions
    - Update on significant progress (10% increments)
    - Batch updates to avoid flicker
    """

    UPDATE_INTERVAL_MS: float = 100.0  # 100ms updates
    PROGRESS_INCREMENT: float = 10.0  # 10% increments

    def __init__(self) -> None:
        self._callbacks: list[DisplayCallback] = []
        self._last_update_time: float = 0.0
        self._last_progress_milestone: int = 0  # Last 10% milestone
        self._pending_update: DisplayUpdate | None = None
        self._lock = threading.Lock()

        # Automatic timer thread
        self._running = False
        self._timer_thread: threading.Thread | None = None

    def on_update(self, callback: DisplayCallback) -> None:
        """Register callback for display updates."""
        self._callbacks.append(callback)

    def _should_update(self, reason: UpdateReason, progress: float) -> bool:
        """Determine if update should be sent based on strategy."""
        now = time.perf_counter() * 1000  # ms

        # Always update for transitions and completion
        if reason in (UpdateReason.STAGE_TRANSITION, UpdateReason.COMPLETION, UpdateReason.MANUAL):
            return True

        # Check 100ms interval for timer ticks
        if reason == UpdateReason.TIMER_TICK:
            elapsed_since_update = now - self._last_update_time
            if elapsed_since_update >= self.UPDATE_INTERVAL_MS:
                return True

        # Check 10% progress milestone
        current_milestone = int(progress // self.PROGRESS_INCREMENT)
        if current_milestone > self._last_progress_milestone:
            self._last_progress_milestone = current_milestone
            return True

        return False

    def queue_update(self, update: DisplayUpdate) -> bool:
        """Queue an update, returns True if update was sent."""
        with self._lock:
            if not self._should_update(update.reason, update.progress_percent):
                self._pending_update = update  # Hold for batching
                return False

            # Send update
            self._last_update_time = time.perf_counter() * 1000
            self._pending_update = None

        # Notify callbacks outside lock
        self._notify(update)
        return True

    def flush(self) -> None:
        """Force send any pending update."""
        with self._lock:
            pending = self._pending_update
            self._pending_update = None

        if pending:
            self._notify(pending)

    def _notify(self, update: DisplayUpdate) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Display callback error: {e}")

    def reset(self) -> None:
        """Reset display state for new execution."""
        with self._lock:
            self._last_update_time = 0.0
            self._last_progress_milestone = 0
            self._pending_update = None


# =============================================================================
# Progress Tracker (with 10% milestones)
# =============================================================================


class ProgressTracker:
    """Tracks percentage completion with milestone detection.

    Features:
    - Percentage completion tracking
    - Automatic 10% milestone detection
    - Weighted stage progress support
    """

    MILESTONE_INCREMENT: float = 10.0  # 10% increments

    def __init__(self, total_steps: int = 100) -> None:
        self.total_steps = total_steps
        self.current_step = 0
        self._last_milestone: int = 0
        self._milestone_callbacks: list[Callable[[float], None]] = []

    def on_milestone(self, callback: Callable[[float], None]) -> None:
        """Register callback for 10% milestone crossings."""
        self._milestone_callbacks.append(callback)

    def advance(self, steps: int = 1) -> bool:
        """Advance progress, returns True if milestone crossed."""
        self.current_step = min(self.current_step + steps, self.total_steps)
        return self._check_milestone()

    def set(self, step: int) -> bool:
        """Set progress directly, returns True if milestone crossed."""
        self.current_step = max(0, min(step, self.total_steps))
        return self._check_milestone()

    def set_percentage(self, percent: float) -> bool:
        """Set progress by percentage, returns True if milestone crossed."""
        step = int((percent / 100.0) * self.total_steps)
        return self.set(step)

    def _check_milestone(self) -> bool:
        """Check if we crossed a 10% milestone."""
        current_milestone = int(self.percentage // self.MILESTONE_INCREMENT)
        if current_milestone > self._last_milestone:
            self._last_milestone = current_milestone
            percent = current_milestone * self.MILESTONE_INCREMENT
            # Notify callbacks
            for cb in self._milestone_callbacks:
                try:
                    cb(percent)
                except Exception as e:
                    logger.error(f"Milestone callback error: {e}")
            return True
        return False

    @property
    def percentage(self) -> float:
        """Get current percentage (0-100)."""
        if self.total_steps == 0:
            return 100.0
        return (self.current_step / self.total_steps) * 100

    @property
    def fraction(self) -> float:
        """Get current fraction (0-1)."""
        return self.percentage / 100.0

    @property
    def is_complete(self) -> bool:
        """Check if 100% complete."""
        return self.current_step >= self.total_steps

    @property
    def current_milestone(self) -> int:
        """Get current 10% milestone (0, 10, 20, ..., 100)."""
        return int(self.percentage // self.MILESTONE_INCREMENT) * int(self.MILESTONE_INCREMENT)

    def reset(self) -> None:
        """Reset progress to 0."""
        self.current_step = 0
        self._last_milestone = 0

    def __str__(self) -> str:
        return f"{self.percentage:.1f}% ({self.current_step}/{self.total_steps})"


# =============================================================================
# ETA Calculator (with smoothing)
# =============================================================================


class ETACalculator:
    """Estimates time remaining based on progress with smoothing.

    Features:
    - ETA calculation from progress rate
    - Exponential smoothing for stable estimates
    - Stage-aware ETA for multi-stage execution
    """

    SMOOTHING_FACTOR: float = 0.3  # EMA smoothing (0-1, higher = more responsive)

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._progress: float = 0.0
        self._smoothed_rate: float | None = None  # Progress per second
        self._last_update_time: float = 0.0
        self._last_progress: float = 0.0

    def start(self) -> None:
        """Start ETA tracking."""
        self._start_time = time.perf_counter()
        self._progress = 0.0
        self._smoothed_rate = None
        self._last_update_time = self._start_time
        self._last_progress = 0.0

    def update(self, progress_fraction: float) -> None:
        """Update with current progress (0-1)."""
        now = time.perf_counter()
        self._progress = max(0.0, min(1.0, progress_fraction))

        # Calculate instantaneous rate
        time_delta = now - self._last_update_time
        progress_delta = self._progress - self._last_progress

        if time_delta > 0.01:  # Avoid division issues
            instant_rate = progress_delta / time_delta

            # Apply exponential smoothing
            if self._smoothed_rate is None:
                self._smoothed_rate = instant_rate
            else:
                self._smoothed_rate = (
                    self.SMOOTHING_FACTOR * instant_rate
                    + (1 - self.SMOOTHING_FACTOR) * self._smoothed_rate
                )

            self._last_update_time = now
            self._last_progress = self._progress

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    @property
    def eta_seconds(self) -> float | None:
        """Get estimated seconds remaining."""
        if self._start_time is None or self._progress <= 0:
            return None

        if self._progress >= 1.0:
            return 0.0

        remaining_fraction = 1.0 - self._progress

        # Use smoothed rate if available, otherwise simple calculation
        if self._smoothed_rate and self._smoothed_rate > 0:
            return remaining_fraction / self._smoothed_rate
        else:
            # Fallback to simple elapsed-based calculation
            elapsed = self.elapsed_seconds
            if elapsed > 0 and self._progress > 0:
                rate = self._progress / elapsed
                return remaining_fraction / rate if rate > 0 else None

        return None

    def eta_display(self) -> str:
        """Get human-readable ETA string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."
        if eta < 0:
            return "any moment..."
        if eta < 60:
            return f"{eta:.0f}s remaining"
        elif eta < 3600:
            minutes = int(eta // 60)
            seconds = int(eta % 60)
            return f"{minutes}m {seconds}s remaining"
        else:
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            return f"{hours}h {minutes}m remaining"

    def reset(self) -> None:
        """Reset ETA calculator."""
        self._start_time = None
        self._progress = 0.0
        self._smoothed_rate = None
        self._last_update_time = 0.0
        self._last_progress = 0.0


# =============================================================================
# Execution Timer (Main Class - Step 4.2)
# =============================================================================


class ExecutionTimer:
    """Comprehensive execution timer with display updates.

    Implements Step 4.2 Timer Components:
    - GlobalTimer: Total elapsed since start
    - StageTimer: Per-stage elapsed times
    - ETACalculator: Estimated time remaining
    - ProgressTracker: Percentage completion

    Display Update Strategy:
    - Update every 100ms for active stages
    - Update on stage transitions
    - Update on significant progress (10% increments)
    - Batch updates to avoid flicker
    """

    def __init__(self, total_stages: int = 1) -> None:
        # Global timer
        self._global_start: float | None = None
        self._global_end: float | None = None

        # Stage tracking
        self._stages: dict[str, StageInfo] = {}
        self._stage_order: list[str] = []
        self._current_stage: str | None = None
        self._total_stages = total_stages
        self._completed_stages = 0

        # Progress and ETA
        self._progress = ProgressTracker(total_steps=total_stages * 100)
        self._eta = ETACalculator()

        # Display controller
        self._display = DisplayController()

        # Auto-update thread for 100ms ticks
        self._auto_update = False
        self._update_thread: threading.Thread | None = None
        self._running = False

    # -------------------------------------------------------------------------
    # Display Callbacks
    # -------------------------------------------------------------------------

    def on_display_update(self, callback: DisplayCallback) -> None:
        """Register callback for display updates."""
        self._display.on_update(callback)

    def _emit_update(self, reason: UpdateReason, message: str | None = None) -> None:
        """Emit a display update."""
        update = DisplayUpdate(
            timestamp=time.perf_counter(),
            reason=reason,
            elapsed_ms=self.total_elapsed_ms,
            current_stage=self._current_stage,
            progress_percent=self._progress.percentage,
            eta_display=self._eta.eta_display(),
            message=message,
        )
        self._display.queue_update(update)

    # -------------------------------------------------------------------------
    # Global Timer
    # -------------------------------------------------------------------------

    def start(self, auto_update: bool = False) -> None:
        """Start the global timer.

        Args:
            auto_update: If True, emit 100ms updates automatically
        """
        self._global_start = time.perf_counter()
        self._global_end = None
        self._stages.clear()
        self._stage_order.clear()
        self._current_stage = None
        self._completed_stages = 0
        self._progress.reset()
        self._eta.start()
        self._display.reset()

        self._emit_update(UpdateReason.STAGE_TRANSITION, "Execution started")

        # Start auto-update thread if requested
        if auto_update:
            self._start_auto_update()

    def stop(self) -> None:
        """Stop the global timer."""
        # Stop auto-update first
        self._stop_auto_update()

        # End current stage if any
        if self._current_stage:
            self.end_stage(self._current_stage)

        self._global_end = time.perf_counter()
        self._progress.set_percentage(100)
        self._display.flush()
        self._emit_update(UpdateReason.COMPLETION, "Execution completed")

    def _start_auto_update(self) -> None:
        """Start automatic 100ms update thread."""
        if self._running:
            return

        self._running = True
        self._auto_update = True
        self._update_thread = threading.Thread(
            target=self._auto_update_loop, daemon=True, name="ExecutionTimer-AutoUpdate"
        )
        self._update_thread.start()

    def _stop_auto_update(self) -> None:
        """Stop automatic update thread."""
        self._running = False
        self._auto_update = False
        if self._update_thread:
            self._update_thread.join(timeout=0.5)
            self._update_thread = None

    def _auto_update_loop(self) -> None:
        """Background loop for 100ms updates."""
        while self._running:
            if self.is_running:
                self._emit_update(UpdateReason.TIMER_TICK)
            time.sleep(0.1)  # 100ms

    @property
    def is_running(self) -> bool:
        """Check if timer is currently running."""
        return self._global_start is not None and self._global_end is None

    @property
    def total_elapsed_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        if self._global_start is None:
            return 0.0
        end = self._global_end if self._global_end else time.perf_counter()
        return (end - self._global_start) * 1000

    @property
    def total_elapsed_seconds(self) -> float:
        """Get total elapsed time in seconds."""
        return self.total_elapsed_ms / 1000

    # -------------------------------------------------------------------------
    # Stage Timer
    # -------------------------------------------------------------------------

    def begin_stage(self, name: str, weight: float = 1.0) -> None:
        """Begin a new stage.

        Args:
            name: Stage name
            weight: Weight for progress calculation (default 1.0)
        """
        # End current stage if any
        if self._current_stage:
            self.end_stage(self._current_stage)

        # Create new stage
        self._stages[name] = StageInfo(
            name=name,
            start_time=time.perf_counter(),
            weight=weight,
        )
        self._stage_order.append(name)
        self._current_stage = name

        self._emit_update(UpdateReason.STAGE_TRANSITION, f"Stage '{name}' started")

    def end_stage(self, name: str) -> None:
        """End a stage."""
        if name in self._stages and self._stages[name].end_time is None:
            self._stages[name].end_time = time.perf_counter()
            self._completed_stages += 1

            # Update progress based on completed stages
            stage_progress = (self._completed_stages / self._total_stages) * 100
            milestone = self._progress.set_percentage(stage_progress)
            self._eta.update(self._progress.fraction)

            if self._current_stage == name:
                self._current_stage = None

            reason = UpdateReason.PROGRESS_MILESTONE if milestone else UpdateReason.STAGE_TRANSITION
            self._emit_update(reason, f"Stage '{name}' completed")

    def update_stage_progress(self, percent_within_stage: float) -> None:
        """Update progress within current stage (0-100)."""
        if self._current_stage is None:
            return

        # Calculate overall progress
        completed_progress = (self._completed_stages / self._total_stages) * 100
        current_stage_contribution = (percent_within_stage / 100) * (100 / self._total_stages)
        total_progress = completed_progress + current_stage_contribution

        milestone = self._progress.set_percentage(total_progress)
        self._eta.update(self._progress.fraction)

        if milestone:
            self._emit_update(UpdateReason.PROGRESS_MILESTONE)

    @property
    def current_stage(self) -> str | None:
        """Get current stage name."""
        return self._current_stage

    def stage_elapsed_ms(self, name: str) -> float:
        """Get elapsed time for a specific stage."""
        if name not in self._stages:
            return 0.0
        return self._stages[name].elapsed_ms

    def all_stages(self) -> list[StageInfo]:
        """Get all stages in order."""
        return [self._stages[name] for name in self._stage_order]

    # -------------------------------------------------------------------------
    # Progress & ETA
    # -------------------------------------------------------------------------

    @property
    def progress(self) -> ProgressTracker:
        """Get progress tracker."""
        return self._progress

    @property
    def eta(self) -> ETACalculator:
        """Get ETA calculator."""
        return self._eta

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def display_line(self) -> str:
        """Get formatted display line."""
        stage = self._current_stage or "idle"
        elapsed = self.total_elapsed_seconds
        progress = self._progress.percentage
        eta = self._eta.eta_display()
        return f"[{stage}] {elapsed:.1f}s | {progress:.0f}% | ETA: {eta}"

    def summary(self) -> dict[str, float]:
        """Get timing summary."""
        result: dict[str, float] = {
            "total_ms": self.total_elapsed_ms,
            "progress_percent": self._progress.percentage,
        }
        for name in self._stage_order:
            result[f"stage_{name}_ms"] = self._stages[name].elapsed_ms
        return result

    def detailed_summary(self) -> str:
        """Get detailed human-readable summary."""
        lines = [
            f"Total Time: {self.total_elapsed_seconds:.2f}s",
            f"Progress: {self._progress.percentage:.1f}%",
            "",
            "Stages:",
        ]
        for stage in self.all_stages():
            lines.append(f"  {stage}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize timer state to dictionary for persistence."""
        return {
            "current_stage": self._current_stage,
            "stage_order": list(self._stage_order),
            "stages": {
                name: {
                    "name": stage.name,
                    "elapsed_ms": stage.elapsed_ms,
                    "started": stage.started,
                    "completed": stage.completed,
                }
                for name, stage in self._stages.items()
            },
            "progress": {
                "total": self._progress.total,
                "completed": self._progress.completed,
            },
            "global_start": self._global_start,
            "global_end": self._global_end,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionTimer:
        """Restore timer state from dictionary."""
        timer = cls()
        timer._current_stage = data.get("current_stage")
        timer._stage_order = list(data.get("stage_order", []))
        timer._global_start = data.get("global_start")
        timer._global_end = data.get("global_end")

        for name, stage_data in data.get("stages", {}).items():
            stage = StageInfo(
                name=stage_data["name"],
                elapsed_ms=stage_data.get("elapsed_ms", 0.0),
                started=stage_data.get("started", False),
                completed=stage_data.get("completed", False),
            )
            timer._stages[name] = stage

        progress_data = data.get("progress", {})
        if progress_data.get("total"):
            timer._progress.set_total(progress_data["total"])
            timer._progress.update(progress_data.get("completed", 0))

        return timer

    def save_to_file(self, path: str) -> None:
        """Save timer state to JSON file.

        Args:
            path: File path to save to.
        """
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_from_file(cls, path: str) -> ExecutionTimer:
        """Load timer state from JSON file.

        Args:
            path: File path to load from.

        Returns:
            Restored ExecutionTimer instance.
        """
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
