"""Task executor with comprehensive execution control.

Executor drives the state machine through execution states and delegates actual
work to an injected callable (which can internally use local or remote LLMs).

This module implements:
- Feature 4: Execution Control (Start/Abort/Rollback/Pause/Resume)
- Feature 1: Execution Timer & Transparency (progress tracking)

Works with:
- ExecutionStateMachine: State lifecycle management
- ExecutionController: Pause/resume/abort control (from resources.control)
- ExecutionTimer: Timing and progress tracking (from resources.timer)
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from proxima.core.state import ExecutionState, ExecutionStateMachine
from proxima.utils.logging import get_logger

T = TypeVar("T")
ExecuteFunction = Callable[[dict[str, Any]], Any]

# Type alias for progress callbacks
ProgressCallback = Callable[["ExecutionProgress"], None]


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass
class ExecutionProgress:
    """Progress information for execution updates.
    
    Provides real-time progress tracking for UI updates and monitoring.
    """
    
    current_stage: str
    stage_index: int
    total_stages: int
    progress_percent: float
    elapsed_ms: float
    eta_ms: float | None = None
    message: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "current_stage": self.current_stage,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "progress_percent": self.progress_percent,
            "elapsed_ms": self.elapsed_ms,
            "eta_ms": self.eta_ms,
            "message": self.message,
        }


@dataclass
class StageInfo:
    """Information about an execution stage."""
    
    name: str
    start_time: float
    end_time: float | None = None
    weight: float = 1.0
    
    @property
    def duration_ms(self) -> float:
        """Get stage duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


@dataclass
class CheckpointData:
    """Checkpoint for pause/resume and rollback support.
    
    Captures execution state at a point in time for recovery.
    """
    
    checkpoint_id: str
    stage_index: int
    stage_name: str
    timestamp: float
    state: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage_index": self.stage_index,
            "stage_name": self.stage_name,
            "timestamp": self.timestamp,
            "state": self.state,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResult:
    """Result of plan execution.
    
    Contains execution outcome, timing information, and metadata.
    """
    
    success: bool
    state: ExecutionState
    output: Any = None
    error: str | None = None
    error_traceback: str | None = None
    elapsed_ms: float = 0.0
    stages_completed: int = 0
    total_stages: int = 0
    checkpoints: list[CheckpointData] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "state": self.state.value if hasattr(self.state, "value") else str(self.state),
            "output": self.output,
            "error": self.error,
            "error_traceback": self.error_traceback,
            "elapsed_ms": self.elapsed_ms,
            "stages_completed": self.stages_completed,
            "total_stages": self.total_stages,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "metadata": self.metadata,
        }


# ==============================================================================
# EXECUTOR CLASS
# ==============================================================================


class Executor:
    """Executes plans with full execution control support.
    
    Implements Feature 4: Execution Control (Start/Abort/Rollback/Pause/Resume)
    Implements Feature 1: Execution Timer & Transparency
    
    Works with:
    - ExecutionStateMachine: State management
    - Progress tracking with callbacks
    - Checkpoint creation for pause/resume/rollback
    
    Example:
        >>> from proxima.core.state import ExecutionStateMachine
        >>> fsm = ExecutionStateMachine()
        >>> executor = Executor(fsm, runner=my_runner_function)
        >>> result = executor.run(plan)
        >>> if not result.success:
        ...     print(f"Failed: {result.error}")
    """
    
    def __init__(
        self,
        state_machine: ExecutionStateMachine,
        runner: ExecuteFunction | None = None,
        auto_checkpoint: bool = True,
        checkpoint_interval: int = 1,  # Create checkpoint every N stages
    ) -> None:
        """Initialize the executor.
        
        Args:
            state_machine: State machine for execution lifecycle
            runner: Callable to execute individual steps (optional)
            auto_checkpoint: Whether to auto-create checkpoints
            checkpoint_interval: Stages between auto-checkpoints
        """
        self.state_machine = state_machine
        self.runner = runner
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.logger = get_logger("executor")
        
        # Execution state
        self._start_time: float = 0.0
        self._current_stage_index: int = 0
        self._total_stages: int = 0
        self._current_stage: StageInfo | None = None
        self._completed_stages: list[StageInfo] = []
        self._checkpoints: list[CheckpointData] = []
        self._last_error: Exception | None = None
        self._error_traceback: str | None = None
        
        # Control flags (thread-safe)
        self._abort_flag = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        self._pause_reason: str | None = None
        
        # Progress callbacks
        self._progress_callbacks: list[ProgressCallback] = []
        self._cleanup_callbacks: list[Callable[[], None]] = []
        
        # Execution context (for checkpoints)
        self._execution_context: dict[str, Any] = {}
        
        # Lock for thread safety
        self._lock = threading.Lock()

    # ==================== PROPERTIES ====================
    
    @property
    def state(self) -> ExecutionState:
        """Get current execution state from the state machine."""
        return self.state_machine.current_state
    
    @property
    def is_running(self) -> bool:
        """Check if execution is currently active."""
        return self.state in (ExecutionState.RUNNING, ExecutionState.PAUSED)
    
    @property
    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self.state == ExecutionState.PAUSED or not self._pause_event.is_set()
    
    @property
    def is_aborted(self) -> bool:
        """Check if execution has been aborted."""
        return self._abort_flag.is_set() or self.state == ExecutionState.ABORTED
    
    @property
    def progress(self) -> ExecutionProgress:
        """Get current execution progress."""
        stage_name = self._current_stage.name if self._current_stage else "initializing"
        
        # Calculate progress percentage
        if self._total_stages > 0:
            progress_percent = (self._current_stage_index / self._total_stages) * 100
        else:
            progress_percent = 0.0
        
        elapsed_ms = self.elapsed_ms
        
        # Estimate ETA based on average stage time
        eta_ms = None
        if self._completed_stages and self._current_stage_index < self._total_stages:
            avg_stage_time = sum(s.duration_ms for s in self._completed_stages) / len(self._completed_stages)
            remaining_stages = self._total_stages - self._current_stage_index
            eta_ms = avg_stage_time * remaining_stages
        
        return ExecutionProgress(
            current_stage=stage_name,
            stage_index=self._current_stage_index,
            total_stages=self._total_stages,
            progress_percent=progress_percent,
            elapsed_ms=elapsed_ms,
            eta_ms=eta_ms,
        )
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed execution time in milliseconds."""
        if self._start_time == 0:
            return 0.0
        return (time.time() - self._start_time) * 1000

    # ==================== MAIN EXECUTION METHODS ====================
    
    def run(
        self,
        plan: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        """Execute a plan with full lifecycle management.
        
        This is the main entry point for synchronous execution.
        
        Args:
            plan: Execution plan from Planner containing steps to execute
            timeout_seconds: Optional timeout for entire execution
            
        Returns:
            ExecutionResult with success status, output, and metadata
            
        Raises:
            TimeoutError: If execution exceeds timeout_seconds
        """
        self._reset_state()
        self._start_time = time.time()
        
        # Extract steps from plan
        steps = plan.get("steps", [])
        if not steps and plan.get("task"):
            # Simple plan with single task
            steps = [{"name": "execute", "task": plan}]
        
        self._total_stages = len(steps) if steps else 1
        
        # Transition to RUNNING state
        try:
            self.state_machine.execute()
            self.logger.info("execution.started", total_stages=self._total_stages)
        except Exception as exc:
            self.logger.error("execution.start_failed", error=str(exc))
            return ExecutionResult(
                success=False,
                state=self.state,
                error=f"Failed to start execution: {exc}",
                elapsed_ms=self.elapsed_ms,
            )
        
        try:
            # Execute with optional timeout
            if timeout_seconds:
                result = self._run_with_timeout(plan, steps, timeout_seconds)
            else:
                result = self._run_steps(plan, steps)
            
            return result
            
        except TimeoutError as exc:
            self._handle_error(exc)
            return ExecutionResult(
                success=False,
                state=self.state,
                error=f"Execution timed out after {timeout_seconds}s",
                elapsed_ms=self.elapsed_ms,
                stages_completed=self._current_stage_index,
                total_stages=self._total_stages,
                checkpoints=self._checkpoints,
            )
        except Exception as exc:
            self._handle_error(exc)
            return ExecutionResult(
                success=False,
                state=self.state,
                error=str(exc),
                error_traceback=self._error_traceback,
                elapsed_ms=self.elapsed_ms,
                stages_completed=self._current_stage_index,
                total_stages=self._total_stages,
                checkpoints=self._checkpoints,
            )
        finally:
            self._cleanup()

    async def run_async(
        self,
        plan: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        """Execute a plan asynchronously.
        
        Args:
            plan: Execution plan from Planner
            timeout_seconds: Optional timeout for entire execution
            
        Returns:
            ExecutionResult with success status and output
        """
        # Run synchronous execution in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.run(plan, timeout_seconds)
        )

    def _run_with_timeout(
        self, 
        plan: dict[str, Any], 
        steps: list[dict[str, Any]], 
        timeout: float,
    ) -> ExecutionResult:
        """Execute steps with a timeout."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_steps, plan, steps)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                self.abort("Execution timeout exceeded")
                raise TimeoutError(f"Execution exceeded {timeout}s timeout")

    def _run_steps(
        self, 
        plan: dict[str, Any], 
        steps: Sequence[dict[str, Any]],
    ) -> ExecutionResult:
        """Execute all steps in the plan."""
        results: list[Any] = []
        
        for idx, step in enumerate(steps):
            # Check control signals before each step
            if not self._check_control_signals():
                # Aborted or paused indefinitely
                if self.is_aborted:
                    return ExecutionResult(
                        success=False,
                        state=self.state,
                        error="Execution aborted",
                        output=results if results else None,
                        elapsed_ms=self.elapsed_ms,
                        stages_completed=self._current_stage_index,
                        total_stages=self._total_stages,
                        checkpoints=self._checkpoints,
                    )
            
            # Execute the step
            step_result = self._execute_step(step, idx)
            results.append(step_result)
            
            # Auto-checkpoint if enabled
            if self.auto_checkpoint and (idx + 1) % self.checkpoint_interval == 0:
                self.create_checkpoint({"step_results": results.copy()})
        
        # Successful completion
        self.state_machine.complete()
        self.logger.info("execution.complete", elapsed_ms=self.elapsed_ms)
        
        return ExecutionResult(
            success=True,
            state=self.state,
            output=results[-1] if len(results) == 1 else results,
            elapsed_ms=self.elapsed_ms,
            stages_completed=self._total_stages,
            total_stages=self._total_stages,
            checkpoints=self._checkpoints,
            metadata={
                "step_count": len(steps),
                "completed_stages": [s.name for s in self._completed_stages],
            },
        )

    def _execute_step(
        self,
        step: dict[str, Any],
        step_index: int,
    ) -> Any:
        """Execute a single plan step with pause/abort checks.
        
        Args:
            step: Step definition from plan
            step_index: Index of the step
            
        Returns:
            Step execution result
        """
        step_name = step.get("name", f"step_{step_index}")
        self.start_stage(step_name)
        self._emit_progress(f"Executing: {step_name}")
        
        try:
            if self.runner:
                result = self.runner(step)
            else:
                # No runner configured - return step as-is with status
                result = {"status": "skipped", "reason": "no runner configured", "step": step}
            
            self.complete_stage(step_name)
            return result
            
        except Exception as exc:
            self.logger.error("step.failed", step=step_name, error=str(exc))
            raise

    # ==================== CONTROL OPERATIONS (Feature 4) ====================
    
    def abort(self, reason: str | None = None) -> bool:
        """Abort execution with cleanup.
        
        Signals the execution loop to stop and performs resource cleanup.
        
        Args:
            reason: Optional reason for abort
            
        Returns:
            True if abort was initiated
        """
        with self._lock:
            if self.is_aborted:
                return False
            
            self._abort_flag.set()
            self._pause_event.set()  # Unblock if paused
            
            try:
                self.state_machine.abort()
                self.logger.info("execution.aborted", reason=reason)
            except Exception as exc:
                self.logger.warning("execution.abort_transition_failed", error=str(exc))
            
            return True
    
    def pause(self, reason: str | None = None) -> CheckpointData | None:
        """Pause execution at next safe checkpoint.
        
        Creates a checkpoint and pauses the execution loop.
        
        Args:
            reason: Optional reason for pause
            
        Returns:
            Checkpoint data if created, None otherwise
        """
        with self._lock:
            if not self.is_running or self.is_paused:
                return None
            
            self._pause_event.clear()
            self._pause_reason = reason
            
            try:
                self.state_machine.pause()
                self.logger.info("execution.paused", reason=reason)
            except Exception as exc:
                self.logger.warning("execution.pause_transition_failed", error=str(exc))
            
            # Create checkpoint at pause point
            checkpoint = self.create_checkpoint({"pause_reason": reason})
            return checkpoint
    
    def resume(self, from_checkpoint: CheckpointData | None = None) -> bool:
        """Resume paused execution.
        
        Args:
            from_checkpoint: Optional specific checkpoint to resume from
            
        Returns:
            True if resumed successfully
        """
        with self._lock:
            if not self.is_paused:
                return False
            
            if from_checkpoint:
                self._restore_checkpoint(from_checkpoint)
            
            self._pause_event.set()
            self._pause_reason = None
            
            try:
                self.state_machine.resume()
                self.logger.info("execution.resumed")
            except Exception as exc:
                self.logger.warning("execution.resume_transition_failed", error=str(exc))
            
            return True
    
    def rollback(
        self,
        checkpoint_id: str | None = None,
        stage_index: int | None = None,
    ) -> CheckpointData | None:
        """Roll back to a previous checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint ID to roll back to
            stage_index: Roll back to checkpoint at this stage index
            
        Returns:
            Checkpoint rolled back to, or None if failed
        """
        with self._lock:
            if not self._checkpoints:
                self.logger.warning("rollback.no_checkpoints")
                return None
            
            target_checkpoint: CheckpointData | None = None
            
            if checkpoint_id:
                for cp in self._checkpoints:
                    if cp.checkpoint_id == checkpoint_id:
                        target_checkpoint = cp
                        break
            elif stage_index is not None:
                for cp in reversed(self._checkpoints):
                    if cp.stage_index <= stage_index:
                        target_checkpoint = cp
                        break
            else:
                # Roll back to most recent checkpoint
                target_checkpoint = self._checkpoints[-1]
            
            if target_checkpoint:
                self._restore_checkpoint(target_checkpoint)
                self.logger.info(
                    "execution.rollback",
                    checkpoint_id=target_checkpoint.checkpoint_id,
                    stage_index=target_checkpoint.stage_index,
                )
                return target_checkpoint
            
            return None

    # ==================== PROGRESS TRACKING (Feature 1) ====================
    
    def on_progress(self, callback: ProgressCallback) -> None:
        """Register callback for progress updates.
        
        Args:
            callback: Function called with ExecutionProgress on updates
        """
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: ProgressCallback) -> bool:
        """Remove a registered progress callback.
        
        Args:
            callback: Callback to remove
            
        Returns:
            True if removed
        """
        try:
            self._progress_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # ==================== STAGE MANAGEMENT ====================
    
    def start_stage(self, name: str, weight: float = 1.0) -> None:
        """Mark the start of an execution stage.
        
        Args:
            name: Stage name
            weight: Relative weight for progress calculation
        """
        self._current_stage = StageInfo(
            name=name,
            start_time=time.time(),
            weight=weight,
        )
        self._emit_progress(f"Starting: {name}")
    
    def complete_stage(self, name: str | None = None) -> None:
        """Mark the completion of an execution stage.
        
        Args:
            name: Stage name (uses current if None)
        """
        if self._current_stage:
            self._current_stage.end_time = time.time()
            self._completed_stages.append(self._current_stage)
            self._current_stage_index += 1
            
            self.logger.debug(
                "stage.completed",
                name=self._current_stage.name,
                duration_ms=self._current_stage.duration_ms,
            )
        
        self._current_stage = None
        self._emit_progress(f"Completed: {name or 'stage'}")

    # ==================== CHECKPOINTS ====================
    
    def create_checkpoint(
        self,
        custom_state: dict[str, Any] | None = None,
    ) -> CheckpointData:
        """Create a checkpoint at the current execution point.
        
        Args:
            custom_state: Additional state to serialize
            
        Returns:
            Created checkpoint data
        """
        checkpoint_id = f"cp_{self._current_stage_index}_{int(time.time() * 1000)}"
        
        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            stage_index=self._current_stage_index,
            stage_name=self._current_stage.name if self._current_stage else "unknown",
            timestamp=time.time(),
            state={
                "execution_context": self._execution_context.copy(),
                "completed_stages": [s.name for s in self._completed_stages],
                "custom": custom_state or {},
            },
            metadata={
                "elapsed_ms": self.elapsed_ms,
                "total_stages": self._total_stages,
            },
        )
        
        self._checkpoints.append(checkpoint)
        self.logger.debug("checkpoint.created", checkpoint_id=checkpoint_id)
        
        return checkpoint
    
    def get_available_checkpoints(self) -> list[CheckpointData]:
        """Get list of available checkpoints for rollback."""
        return self._checkpoints.copy()

    # ==================== RESOURCE MANAGEMENT ====================
    
    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback for abort.
        
        Args:
            callback: Function called during abort cleanup
        """
        self._cleanup_callbacks.append(callback)
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a value in the execution context.
        
        Context values are included in checkpoints.
        
        Args:
            key: Context key
            value: Context value
        """
        self._execution_context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the execution context.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        return self._execution_context.get(key, default)

    # ==================== INTERNAL METHODS ====================
    
    def _reset_state(self) -> None:
        """Reset executor state for a new execution."""
        self._start_time = 0.0
        self._current_stage_index = 0
        self._total_stages = 0
        self._current_stage = None
        self._completed_stages = []
        self._checkpoints = []
        self._last_error = None
        self._error_traceback = None
        self._abort_flag.clear()
        self._pause_event.set()
        self._pause_reason = None
        self._execution_context = {}
    
    def _check_control_signals(self) -> bool:
        """Check for pause/abort signals.
        
        Blocks if paused, returns False if aborted.
        
        Returns:
            True if execution should continue, False if aborted
        """
        # Check abort first
        if self._abort_flag.is_set():
            return False
        
        # Wait on pause event (blocks if paused)
        # Use timeout to periodically check abort flag
        while not self._pause_event.wait(timeout=0.1):
            if self._abort_flag.is_set():
                return False
        
        return not self._abort_flag.is_set()
    
    def _emit_progress(self, message: str | None = None) -> None:
        """Emit progress update to all registered callbacks."""
        progress = self.progress
        if message:
            progress.message = message
        
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as exc:
                self.logger.warning("progress.callback_error", error=str(exc))
    
    def _handle_error(self, error: Exception) -> None:
        """Handle execution error with state transition."""
        self._last_error = error
        self._error_traceback = traceback.format_exc()
        
        try:
            self.state_machine.error()
        except Exception:
            pass  # State machine may already be in error state
        
        self.logger.error(
            "execution.failed",
            error=str(error),
            stage=self._current_stage.name if self._current_stage else "unknown",
        )
    
    def _restore_checkpoint(self, checkpoint: CheckpointData) -> None:
        """Restore execution state from a checkpoint."""
        self._current_stage_index = checkpoint.stage_index
        self._execution_context = checkpoint.state.get("execution_context", {})
        
        # Trim completed stages to checkpoint point
        stage_names = checkpoint.state.get("completed_stages", [])
        self._completed_stages = [
            s for s in self._completed_stages 
            if s.name in stage_names
        ]
        
        self.logger.debug(
            "checkpoint.restored",
            checkpoint_id=checkpoint.checkpoint_id,
        )
    
    def _cleanup(self) -> None:
        """Clean up resources after execution."""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as exc:
                self.logger.warning("cleanup.callback_error", error=str(exc))
        
        self._cleanup_callbacks.clear()


# ==============================================================================
# ASYNC EXECUTOR
# ==============================================================================


class AsyncExecutor:
    """Async-native executor for concurrent execution scenarios.
    
    Provides an asyncio-first interface for executing plans with
    non-blocking pause/resume support.
    """
    
    def __init__(
        self,
        state_machine: ExecutionStateMachine,
        runner: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """Initialize async executor.
        
        Args:
            state_machine: State machine for execution lifecycle
            runner: Async or sync callable to execute steps
        """
        self._executor = Executor(state_machine, runner)
        self._task: asyncio.Task[ExecutionResult] | None = None
    
    @property
    def state(self) -> ExecutionState:
        """Get current execution state."""
        return self._executor.state
    
    @property
    def progress(self) -> ExecutionProgress:
        """Get current execution progress."""
        return self._executor.progress
    
    async def run(
        self,
        plan: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        """Execute a plan asynchronously.
        
        Args:
            plan: Execution plan
            timeout_seconds: Optional timeout
            
        Returns:
            ExecutionResult with outcome
        """
        self._task = asyncio.current_task()
        return await self._executor.run_async(plan, timeout_seconds)
    
    def abort(self, reason: str | None = None) -> bool:
        """Abort execution."""
        return self._executor.abort(reason)
    
    def pause(self, reason: str | None = None) -> CheckpointData | None:
        """Pause execution."""
        return self._executor.pause(reason)
    
    def resume(self) -> bool:
        """Resume execution."""
        return self._executor.resume()
    
    def on_progress(self, callback: ProgressCallback) -> None:
        """Register progress callback."""
        self._executor.on_progress(callback)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================


def create_executor(
    runner: ExecuteFunction | None = None,
    auto_checkpoint: bool = True,
) -> Executor:
    """Create an executor with a new state machine.
    
    Convenience function for simple use cases.
    
    Args:
        runner: Optional execution function
        auto_checkpoint: Whether to auto-create checkpoints
        
    Returns:
        Configured Executor instance
    """
    fsm = ExecutionStateMachine()
    return Executor(fsm, runner=runner, auto_checkpoint=auto_checkpoint)


def run_plan(
    plan: dict[str, Any],
    runner: ExecuteFunction | None = None,
    timeout_seconds: float | None = None,
) -> ExecutionResult:
    """Execute a plan in a one-shot manner.
    
    Creates an executor, runs the plan, and returns the result.
    
    Args:
        plan: Execution plan to run
        runner: Optional execution function
        timeout_seconds: Optional timeout
        
    Returns:
        ExecutionResult with outcome
    """
    executor = create_executor(runner)
    return executor.run(plan, timeout_seconds)
