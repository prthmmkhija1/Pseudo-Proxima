"""Build Progress Tracker for Proxima Agent.

Phase 4: Backend Building & Compilation System

Provides build progress tracking including:
- Step-by-step progress monitoring
- Output parsing for progress indicators
- Time estimation
- Event emission for UI updates
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Pattern

from proxima.utils.logging import get_logger

logger = get_logger("agent.build_progress_tracker")


class BuildStepStatus(Enum):
    """Status of a build step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BuildPhase(Enum):
    """Build phases."""
    INITIALIZATION = "initialization"
    DEPENDENCY_CHECK = "dependency_check"
    DEPENDENCY_INSTALL = "dependency_install"
    CONFIGURATION = "configuration"
    COMPILATION = "compilation"
    LINKING = "linking"
    TESTING = "testing"
    VERIFICATION = "verification"
    FINALIZATION = "finalization"


@dataclass
class BuildStep:
    """A single step in the build process."""
    step_id: str
    name: str
    description: str
    phase: BuildPhase
    weight: float = 1.0  # Relative weight for progress calculation
    timeout_seconds: int = 300
    status: BuildStepStatus = BuildStepStatus.PENDING
    progress_percent: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: str = ""
    error_message: Optional[str] = None
    
    def start(self) -> None:
        """Mark step as started."""
        self.status = BuildStepStatus.RUNNING
        self.started_at = datetime.now()
    
    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark step as completed."""
        self.completed_at = datetime.now()
        self.status = BuildStepStatus.COMPLETED if success else BuildStepStatus.FAILED
        self.progress_percent = 100.0 if success else self.progress_percent
        self.error_message = error
    
    def skip(self) -> None:
        """Mark step as skipped."""
        self.status = BuildStepStatus.SKIPPED
        self.completed_at = datetime.now()
    
    @property
    def duration_seconds(self) -> float:
        """Get step duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "phase": self.phase.value,
            "weight": self.weight,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


@dataclass
class ProgressPattern:
    """Pattern for extracting progress from build output."""
    pattern: Pattern[str]
    group_name: str = "progress"
    is_percentage: bool = True
    multiplier: float = 1.0


@dataclass
class BuildProgress:
    """Overall build progress."""
    backend_name: str
    total_steps: int
    completed_steps: int
    current_step: Optional[BuildStep]
    overall_percent: float
    phase: BuildPhase
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float]
    status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_name": self.backend_name,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "current_step": self.current_step.to_dict() if self.current_step else None,
            "overall_percent": self.overall_percent,
            "phase": self.phase.value,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "status": self.status,
        }


# Type alias for progress callback
ProgressCallback = Callable[[BuildProgress], None]


class BuildProgressTracker:
    """Track and report build progress.
    
    Features:
    - Step-by-step tracking
    - Progress parsing from build output
    - Time estimation
    - Callback-based UI updates
    
    Example:
        >>> tracker = BuildProgressTracker("qsim")
        >>> 
        >>> # Add steps
        >>> tracker.add_step("deps", "Install Dependencies", BuildPhase.DEPENDENCY_INSTALL, weight=2.0)
        >>> tracker.add_step("compile", "Compile", BuildPhase.COMPILATION, weight=5.0)
        >>> tracker.add_step("test", "Run Tests", BuildPhase.TESTING, weight=1.0)
        >>> 
        >>> # Register callback
        >>> tracker.on_progress(lambda p: print(f"{p.overall_percent:.1f}%"))
        >>> 
        >>> # Start tracking
        >>> tracker.start_step("deps")
        >>> tracker.update_step_progress("deps", 50.0)
        >>> tracker.complete_step("deps")
    """
    
    # Common progress patterns
    DEFAULT_PATTERNS = [
        # Percentage patterns
        ProgressPattern(re.compile(r"\[(?P<progress>\d+)%\]"), "progress"),
        ProgressPattern(re.compile(r"(?P<progress>\d+)% complete"), "progress"),
        ProgressPattern(re.compile(r"Progress: (?P<progress>\d+(?:\.\d+)?)%"), "progress"),
        
        # Fraction patterns (e.g., "10/100")
        ProgressPattern(
            re.compile(r"(?P<current>\d+)/(?P<total>\d+)"),
            "current",
            is_percentage=False,
        ),
        
        # CMake patterns
        ProgressPattern(re.compile(r"\[(?P<progress>\d+)%\] Building"), "progress"),
        ProgressPattern(re.compile(r"\[(?P<progress>\d+)%\] Linking"), "progress"),
        
        # pip patterns
        ProgressPattern(re.compile(r"Installing collected packages"), "progress", multiplier=0),
        ProgressPattern(re.compile(r"Successfully installed"), "progress", multiplier=0),
        
        # Make patterns
        ProgressPattern(re.compile(r"make\[(\d+)\]"), "progress", is_percentage=False),
    ]
    
    def __init__(
        self,
        backend_name: str,
        patterns: Optional[List[ProgressPattern]] = None,
    ):
        """Initialize the tracker.
        
        Args:
            backend_name: Name of the backend being built
            patterns: Custom progress patterns (defaults to DEFAULT_PATTERNS)
        """
        self.backend_name = backend_name
        self.patterns = patterns or self.DEFAULT_PATTERNS
        
        self.steps: Dict[str, BuildStep] = {}
        self.step_order: List[str] = []
        self.callbacks: List[ProgressCallback] = []
        
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.current_step_id: Optional[str] = None
        
        # Historical data for time estimation
        self._step_history: Dict[str, List[float]] = {}
    
    def add_step(
        self,
        step_id: str,
        name: str,
        phase: BuildPhase,
        description: str = "",
        weight: float = 1.0,
        timeout_seconds: int = 300,
    ) -> BuildStep:
        """Add a build step.
        
        Args:
            step_id: Unique step identifier
            name: Display name
            phase: Build phase
            description: Step description
            weight: Relative weight for progress
            timeout_seconds: Step timeout
            
        Returns:
            Created BuildStep
        """
        step = BuildStep(
            step_id=step_id,
            name=name,
            description=description or name,
            phase=phase,
            weight=weight,
            timeout_seconds=timeout_seconds,
        )
        
        self.steps[step_id] = step
        self.step_order.append(step_id)
        
        return step
    
    def add_steps_from_config(self, steps_config: List[Dict[str, Any]]) -> None:
        """Add steps from configuration.
        
        Args:
            steps_config: List of step configurations
        """
        phase_map = {p.value: p for p in BuildPhase}
        
        for config in steps_config:
            phase_str = config.get("phase", "compilation")
            phase = phase_map.get(phase_str, BuildPhase.COMPILATION)
            
            self.add_step(
                step_id=config.get("step_id", config.get("name", "").lower().replace(" ", "_")),
                name=config.get("name", config.get("step_id", "Unknown")),
                phase=phase,
                description=config.get("description", ""),
                weight=config.get("weight", 1.0),
                timeout_seconds=config.get("timeout", 300),
            )
    
    def on_progress(self, callback: ProgressCallback) -> None:
        """Register a progress callback.
        
        Args:
            callback: Function called with BuildProgress
        """
        self.callbacks.append(callback)
    
    def start(self) -> None:
        """Start the build tracking."""
        self.started_at = datetime.now()
        self._notify_progress()
        logger.info(f"Build started for {self.backend_name}")
    
    def start_step(self, step_id: str) -> None:
        """Start a specific step.
        
        Args:
            step_id: Step identifier
        """
        if step_id not in self.steps:
            logger.warning(f"Unknown step: {step_id}")
            return
        
        step = self.steps[step_id]
        step.start()
        self.current_step_id = step_id
        
        self._notify_progress()
        logger.debug(f"Started step: {step.name}")
    
    def update_step_progress(
        self,
        step_id: str,
        progress: float,
        output: Optional[str] = None,
    ) -> None:
        """Update progress for a step.
        
        Args:
            step_id: Step identifier
            progress: Progress percentage (0-100)
            output: Optional output text
        """
        if step_id not in self.steps:
            return
        
        step = self.steps[step_id]
        step.progress_percent = min(100.0, max(0.0, progress))
        
        if output:
            step.output += output
        
        self._notify_progress()
    
    def complete_step(
        self,
        step_id: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Complete a step.
        
        Args:
            step_id: Step identifier
            success: Whether step succeeded
            error: Error message if failed
        """
        if step_id not in self.steps:
            return
        
        step = self.steps[step_id]
        step.complete(success, error)
        
        # Record duration for future estimation
        if step.duration_seconds > 0:
            if step_id not in self._step_history:
                self._step_history[step_id] = []
            self._step_history[step_id].append(step.duration_seconds)
            # Keep last 5 durations
            self._step_history[step_id] = self._step_history[step_id][-5:]
        
        if self.current_step_id == step_id:
            self.current_step_id = None
        
        self._notify_progress()
        logger.debug(f"Completed step: {step.name} (success={success})")
    
    def skip_step(self, step_id: str) -> None:
        """Skip a step.
        
        Args:
            step_id: Step identifier
        """
        if step_id not in self.steps:
            return
        
        step = self.steps[step_id]
        step.skip()
        
        self._notify_progress()
        logger.debug(f"Skipped step: {step.name}")
    
    def complete(self, success: bool = True) -> None:
        """Complete the build.
        
        Args:
            success: Whether build succeeded
        """
        self.completed_at = datetime.now()
        self._notify_progress()
        
        status = "succeeded" if success else "failed"
        logger.info(f"Build {status} for {self.backend_name}")
    
    def parse_output(self, output: str, step_id: Optional[str] = None) -> Optional[float]:
        """Parse build output for progress.
        
        Args:
            output: Build output text
            step_id: Optional step to update
            
        Returns:
            Extracted progress percentage, or None
        """
        progress = None
        
        for pattern in self.patterns:
            match = pattern.pattern.search(output)
            if match:
                if pattern.is_percentage:
                    try:
                        progress = float(match.group(pattern.group_name))
                    except (ValueError, IndexError):
                        continue
                else:
                    # Handle fraction patterns
                    try:
                        current = float(match.group("current"))
                        total = float(match.group("total"))
                        if total > 0:
                            progress = (current / total) * 100
                    except (ValueError, IndexError):
                        try:
                            progress = float(match.group(pattern.group_name)) * pattern.multiplier
                        except (ValueError, IndexError):
                            continue
                
                if progress is not None:
                    break
        
        # Update step if provided
        if progress is not None and step_id:
            self.update_step_progress(step_id, progress, output)
        
        return progress
    
    def get_progress(self) -> BuildProgress:
        """Get current build progress.
        
        Returns:
            BuildProgress object
        """
        # Calculate overall progress
        total_weight = sum(s.weight for s in self.steps.values())
        completed_weight = 0.0
        
        for step in self.steps.values():
            if step.status == BuildStepStatus.COMPLETED:
                completed_weight += step.weight
            elif step.status == BuildStepStatus.RUNNING:
                completed_weight += step.weight * (step.progress_percent / 100)
            elif step.status == BuildStepStatus.SKIPPED:
                total_weight -= step.weight
        
        overall_percent = (completed_weight / total_weight * 100) if total_weight > 0 else 0
        
        # Count completed steps
        completed_steps = sum(
            1 for s in self.steps.values()
            if s.status in (BuildStepStatus.COMPLETED, BuildStepStatus.SKIPPED)
        )
        
        # Get current step
        current_step = self.steps.get(self.current_step_id) if self.current_step_id else None
        
        # Get current phase
        phase = current_step.phase if current_step else BuildPhase.INITIALIZATION
        
        # Calculate elapsed time
        elapsed = 0.0
        if self.started_at:
            end = self.completed_at or datetime.now()
            elapsed = (end - self.started_at).total_seconds()
        
        # Estimate remaining time
        remaining = self._estimate_remaining(overall_percent, elapsed)
        
        # Determine status
        if self.completed_at:
            failed_steps = sum(
                1 for s in self.steps.values()
                if s.status == BuildStepStatus.FAILED
            )
            status = "failed" if failed_steps > 0 else "completed"
        elif self.started_at:
            status = "running"
        else:
            status = "pending"
        
        return BuildProgress(
            backend_name=self.backend_name,
            total_steps=len(self.steps),
            completed_steps=completed_steps,
            current_step=current_step,
            overall_percent=overall_percent,
            phase=phase,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=remaining,
            status=status,
        )
    
    def _estimate_remaining(
        self,
        current_percent: float,
        elapsed: float,
    ) -> Optional[float]:
        """Estimate remaining time.
        
        Args:
            current_percent: Current progress percentage
            elapsed: Elapsed time in seconds
            
        Returns:
            Estimated remaining seconds, or None
        """
        if current_percent <= 0 or elapsed <= 0:
            return None
        
        # Simple linear estimation
        total_estimated = elapsed / (current_percent / 100)
        remaining = total_estimated - elapsed
        
        return max(0, remaining)
    
    def _notify_progress(self) -> None:
        """Notify all callbacks of progress update."""
        progress = self.get_progress()
        
        for callback in self.callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def get_step_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all steps.
        
        Returns:
            List of step summaries
        """
        return [
            self.steps[step_id].to_dict()
            for step_id in self.step_order
        ]
    
    def get_failed_steps(self) -> List[BuildStep]:
        """Get list of failed steps.
        
        Returns:
            List of failed BuildStep objects
        """
        return [
            s for s in self.steps.values()
            if s.status == BuildStepStatus.FAILED
        ]
    
    def reset(self) -> None:
        """Reset tracker for a new build."""
        self.started_at = None
        self.completed_at = None
        self.current_step_id = None
        
        for step in self.steps.values():
            step.status = BuildStepStatus.PENDING
            step.progress_percent = 0.0
            step.started_at = None
            step.completed_at = None
            step.output = ""
            step.error_message = None


class MultiBackendProgressTracker:
    """Track progress across multiple backend builds.
    
    Example:
        >>> tracker = MultiBackendProgressTracker()
        >>> tracker.add_backend("qsim")
        >>> tracker.add_backend("cuquantum")
        >>> 
        >>> # Get overall progress
        >>> progress = tracker.get_overall_progress()
    """
    
    def __init__(self):
        """Initialize the multi-backend tracker."""
        self.trackers: Dict[str, BuildProgressTracker] = {}
        self.callbacks: List[Callable[[Dict[str, BuildProgress]], None]] = []
    
    def add_backend(self, backend_name: str) -> BuildProgressTracker:
        """Add a backend tracker.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            Created BuildProgressTracker
        """
        tracker = BuildProgressTracker(backend_name)
        tracker.on_progress(lambda p: self._on_backend_progress(backend_name, p))
        self.trackers[backend_name] = tracker
        return tracker
    
    def get_tracker(self, backend_name: str) -> Optional[BuildProgressTracker]:
        """Get tracker for a specific backend.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            BuildProgressTracker or None
        """
        return self.trackers.get(backend_name)
    
    def on_progress(
        self,
        callback: Callable[[Dict[str, BuildProgress]], None],
    ) -> None:
        """Register a progress callback.
        
        Args:
            callback: Function called with dict of all progress
        """
        self.callbacks.append(callback)
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress across all backends.
        
        Returns:
            Progress summary
        """
        all_progress = {
            name: tracker.get_progress()
            for name, tracker in self.trackers.items()
        }
        
        # Calculate overall percentage
        total = len(all_progress)
        if total == 0:
            overall_percent = 0.0
        else:
            overall_percent = sum(
                p.overall_percent for p in all_progress.values()
            ) / total
        
        # Count by status
        status_counts = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        for p in all_progress.values():
            status_counts[p.status] = status_counts.get(p.status, 0) + 1
        
        return {
            "backends": {
                name: p.to_dict() for name, p in all_progress.items()
            },
            "overall_percent": overall_percent,
            "total_backends": total,
            "status_counts": status_counts,
        }
    
    def _on_backend_progress(
        self,
        backend_name: str,
        progress: BuildProgress,
    ) -> None:
        """Handle progress update from a backend.
        
        Args:
            backend_name: Name of the backend
            progress: Progress update
        """
        all_progress = {
            name: tracker.get_progress()
            for name, tracker in self.trackers.items()
        }
        
        for callback in self.callbacks:
            try:
                callback(all_progress)
            except Exception as e:
                logger.warning(f"Multi-backend progress callback error: {e}")
