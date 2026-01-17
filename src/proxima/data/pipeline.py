"""Step 5.6: Data Pipeline - Stage-based execution with timeouts, retries, and cancellation.

Implements a robust data pipeline framework for quantum simulation workflows:
- Stage-based execution with dependencies
- Timeout handling per stage and globally
- Retry mechanisms with configurable strategies
- Cancellation support with cleanup
- Progress tracking and reporting
- Error handling and recovery
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Generic, TypeVar

from proxima.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Enums and Status Types
# =============================================================================


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    SKIPPED = auto()
    RETRYING = auto()
    TIMED_OUT = auto()


class PipelineStatus(Enum):
    """Status of the entire pipeline."""

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    PARTIALLY_COMPLETED = auto()


class RetryStrategy(Enum):
    """Retry strategy for failed stages."""

    NONE = "none"  # No retry
    IMMEDIATE = "immediate"  # Retry immediately
    LINEAR_BACKOFF = "linear_backoff"  # Linear delay increase
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential delay increase
    CONSTANT = "constant"  # Constant delay between retries


class CancellationReason(Enum):
    """Reason for pipeline or stage cancellation."""

    USER_REQUESTED = "user_requested"
    TIMEOUT = "timeout"
    DEPENDENCY_FAILED = "dependency_failed"
    RESOURCE_LIMIT = "resource_limit"
    ERROR = "error"
    EXTERNAL = "external"


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        strategy: Retry strategy to use.
        initial_delay_seconds: Initial delay before first retry.
        max_delay_seconds: Maximum delay between retries.
        backoff_factor: Multiplier for backoff strategies.
        retry_on_exceptions: Exception types to retry on (None = all).
        retry_on_timeout: Whether to retry on timeout.
    """

    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_factor: float = 2.0
    retry_on_exceptions: tuple[type[Exception], ...] | None = None
    retry_on_timeout: bool = True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt.

        Args:
            attempt: The retry attempt number (0-indexed).

        Returns:
            Delay in seconds before the next retry.
        """
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        elif self.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif self.strategy == RetryStrategy.CONSTANT:
            return self.initial_delay_seconds
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.initial_delay_seconds + (attempt * self.initial_delay_seconds)
            return min(delay, self.max_delay_seconds)
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.initial_delay_seconds * (self.backoff_factor**attempt)
            return min(delay, self.max_delay_seconds)
        else:
            return self.initial_delay_seconds


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior.

    Attributes:
        stage_timeout_seconds: Default timeout per stage.
        pipeline_timeout_seconds: Total pipeline timeout.
        cleanup_timeout_seconds: Timeout for cleanup operations.
        grace_period_seconds: Grace period before forceful termination.
    """

    stage_timeout_seconds: float | None = 300.0  # 5 minutes per stage
    pipeline_timeout_seconds: float | None = 3600.0  # 1 hour total
    cleanup_timeout_seconds: float = 30.0
    grace_period_seconds: float = 5.0


@dataclass
class PipelineConfig:
    """Overall pipeline configuration.

    Attributes:
        name: Pipeline name for identification.
        retry: Retry configuration.
        timeout: Timeout configuration.
        continue_on_error: Whether to continue if a non-critical stage fails.
        parallel_stages: Whether to run independent stages in parallel.
        max_parallel: Maximum number of parallel stages.
        collect_metrics: Whether to collect detailed metrics.
        cleanup_on_cancel: Whether to run cleanup on cancellation.
    """

    name: str = "default_pipeline"
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    continue_on_error: bool = False
    parallel_stages: bool = True
    max_parallel: int = 4
    collect_metrics: bool = True
    cleanup_on_cancel: bool = True


# =============================================================================
# Result and Context Classes
# =============================================================================


@dataclass
class StageResult(Generic[T]):
    """Result of executing a pipeline stage.

    Attributes:
        stage_id: Unique identifier for the stage.
        stage_name: Human-readable stage name.
        status: Final status of the stage.
        result: The result value if successful.
        error: Error message if failed.
        exception: Exception object if failed.
        start_time: When the stage started.
        end_time: When the stage finished.
        attempt_count: Number of attempts made.
        metrics: Additional metrics collected.
    """

    stage_id: str
    stage_name: str
    status: StageStatus
    result: T | None = None
    error: str | None = None
    exception: Exception | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    attempt_count: int = 1
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration of the stage in milliseconds."""
        if self.end_time > 0 and self.start_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def is_success(self) -> bool:
        """Whether the stage completed successfully."""
        return self.status == StageStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "status": self.status.name,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attempt_count": self.attempt_count,
            "metrics": self.metrics,
        }


@dataclass
class PipelineResult:
    """Result of executing an entire pipeline.

    Attributes:
        pipeline_id: Unique identifier for this execution.
        pipeline_name: Name of the pipeline.
        status: Final status of the pipeline.
        stage_results: Results from each stage.
        start_time: When the pipeline started.
        end_time: When the pipeline finished.
        cancellation_reason: Reason if cancelled.
        metadata: Additional metadata.
    """

    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    stage_results: dict[str, StageResult[Any]] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    cancellation_reason: CancellationReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Total duration of the pipeline in milliseconds."""
        if self.end_time > 0 and self.start_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def successful_stages(self) -> list[str]:
        """List of successfully completed stage IDs."""
        return [
            stage_id
            for stage_id, result in self.stage_results.items()
            if result.status == StageStatus.COMPLETED
        ]

    @property
    def failed_stages(self) -> list[str]:
        """List of failed stage IDs."""
        return [
            stage_id
            for stage_id, result in self.stage_results.items()
            if result.status in (StageStatus.FAILED, StageStatus.TIMED_OUT)
        ]

    @property
    def is_success(self) -> bool:
        """Whether the pipeline completed successfully."""
        return self.status == PipelineStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.name,
            "stage_results": {
                stage_id: result.to_dict()
                for stage_id, result in self.stage_results.items()
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "successful_stages": self.successful_stages,
            "failed_stages": self.failed_stages,
            "cancellation_reason": (
                self.cancellation_reason.value if self.cancellation_reason else None
            ),
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            f"PIPELINE: {self.pipeline_name}",
            "=" * 60,
            f"Status: {self.status.name}",
            f"Duration: {self.duration_ms:.2f} ms",
            f"Stages: {len(self.stage_results)} total",
            f"  Successful: {len(self.successful_stages)}",
            f"  Failed: {len(self.failed_stages)}",
        ]

        if self.cancellation_reason:
            lines.append(f"Cancellation Reason: {self.cancellation_reason.value}")

        lines.append("")
        lines.append("STAGE DETAILS:")
        for stage_id, result in self.stage_results.items():
            status_icon = "✓" if result.is_success else "✗"
            lines.append(
                f"  {status_icon} {result.stage_name}: {result.status.name} "
                f"({result.duration_ms:.2f} ms)"
            )
            if result.error:
                lines.append(f"      Error: {result.error}")

        lines.append("=" * 60)
        return "\n".join(lines)


class PipelineContext:
    """Shared context for pipeline execution.

    Provides a way to pass data between stages and track pipeline state.
    """

    def __init__(self, pipeline_id: str, config: PipelineConfig) -> None:
        """Initialize the context.

        Args:
            pipeline_id: Unique identifier for this execution.
            config: Pipeline configuration.
        """
        self.pipeline_id = pipeline_id
        self.config = config
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._paused = threading.Event()
        self._stage_results: dict[str, StageResult[Any]] = {}

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context.

        Args:
            key: The key to set.
            value: The value to store.
        """
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context.

        Args:
            key: The key to retrieve.
            default: Default value if key not found.

        Returns:
            The stored value or default.
        """
        with self._lock:
            return self._data.get(key, default)

    def get_stage_result(self, stage_id: str) -> StageResult[Any] | None:
        """Get the result of a previously executed stage.

        Args:
            stage_id: The stage ID to look up.

        Returns:
            The stage result, or None if not found.
        """
        with self._lock:
            return self._stage_results.get(stage_id)

    def store_stage_result(self, result: StageResult[Any]) -> None:
        """Store a stage result in the context.

        Args:
            result: The stage result to store.
        """
        with self._lock:
            self._stage_results[result.stage_id] = result

    def cancel(self, reason: CancellationReason = CancellationReason.USER_REQUESTED) -> None:
        """Request cancellation of the pipeline.

        Args:
            reason: The reason for cancellation.
        """
        self._cancelled.set()
        self.set("_cancellation_reason", reason)
        logger.info(
            "pipeline.cancellation_requested",
            pipeline_id=self.pipeline_id,
            reason=reason.value,
        )

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled.is_set()

    def pause(self) -> None:
        """Request the pipeline to pause."""
        self._paused.set()
        logger.info("pipeline.pause_requested", pipeline_id=self.pipeline_id)

    def resume(self) -> None:
        """Resume a paused pipeline."""
        self._paused.clear()
        logger.info("pipeline.resumed", pipeline_id=self.pipeline_id)

    @property
    def is_paused(self) -> bool:
        """Check if the pipeline is paused."""
        return self._paused.is_set()

    async def wait_if_paused(self, check_interval: float = 0.5) -> None:
        """Wait while the pipeline is paused.

        Args:
            check_interval: How often to check pause status.
        """
        while self.is_paused and not self.is_cancelled:
            await asyncio.sleep(check_interval)

    def check_cancelled(self) -> None:
        """Check if cancelled and raise exception if so.

        Raises:
            PipelineCancelledException: If cancellation was requested.
        """
        if self.is_cancelled:
            reason = self.get("_cancellation_reason", CancellationReason.USER_REQUESTED)
            raise PipelineCancelledException(reason)


# =============================================================================
# Exceptions
# =============================================================================


class PipelineException(Exception):
    """Base exception for pipeline errors."""

    pass


class StageTimeoutException(PipelineException):
    """Raised when a stage times out."""

    def __init__(self, stage_id: str, timeout_seconds: float) -> None:
        self.stage_id = stage_id
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Stage {stage_id} timed out after {timeout_seconds}s")


class PipelineTimeoutException(PipelineException):
    """Raised when the entire pipeline times out."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Pipeline timed out after {timeout_seconds}s")


class PipelineCancelledException(PipelineException):
    """Raised when the pipeline is cancelled."""

    def __init__(self, reason: CancellationReason) -> None:
        self.reason = reason
        super().__init__(f"Pipeline cancelled: {reason.value}")


class StageExecutionException(PipelineException):
    """Raised when a stage fails execution."""

    def __init__(self, stage_id: str, error: str, cause: Exception | None = None) -> None:
        self.stage_id = stage_id
        self.error = error
        self.cause = cause
        super().__init__(f"Stage {stage_id} failed: {error}")


class DependencyFailedException(PipelineException):
    """Raised when a stage's dependency failed."""

    def __init__(self, stage_id: str, failed_dependency: str) -> None:
        self.stage_id = stage_id
        self.failed_dependency = failed_dependency
        super().__init__(
            f"Stage {stage_id} skipped: dependency {failed_dependency} failed"
        )


# =============================================================================
# Stage Definition
# =============================================================================


@dataclass
class Stage(Generic[T, R]):
    """Definition of a pipeline stage.

    Attributes:
        stage_id: Unique identifier for the stage.
        name: Human-readable name.
        handler: The async function to execute.
        dependencies: List of stage IDs this stage depends on.
        timeout_seconds: Timeout for this stage (overrides config).
        retry_config: Retry configuration for this stage.
        critical: Whether failure should stop the pipeline.
        cleanup_handler: Optional cleanup function.
        description: Optional description.
        tags: Optional tags for categorization.
    """

    stage_id: str
    name: str
    handler: Callable[[PipelineContext, T], Awaitable[R]]
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: float | None = None
    retry_config: RetryConfig | None = None
    critical: bool = True
    cleanup_handler: Callable[[PipelineContext], Awaitable[None]] | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.stage_id)


# =============================================================================
# Pipeline Implementation
# =============================================================================


class Pipeline:
    """Orchestrates execution of pipeline stages.

    Provides stage-based execution with:
    - Dependency resolution and topological ordering
    - Timeout handling per stage and globally
    - Retry mechanisms with configurable strategies
    - Cancellation support with cleanup
    - Parallel execution of independent stages
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self._stages: dict[str, Stage[Any, Any]] = {}
        self._stage_order: list[str] = []
        self._context: PipelineContext | None = None
        self._result: PipelineResult | None = None
        self._running = False
        self._lock = asyncio.Lock()

    def add_stage(self, stage: Stage[Any, Any]) -> "Pipeline":
        """Add a stage to the pipeline.

        Args:
            stage: The stage to add.

        Returns:
            Self for method chaining.
        """
        self._stages[stage.stage_id] = stage
        self._recompute_order()
        return self

    def add_stages(self, stages: list[Stage[Any, Any]]) -> "Pipeline":
        """Add multiple stages to the pipeline.

        Args:
            stages: List of stages to add.

        Returns:
            Self for method chaining.
        """
        for stage in stages:
            self._stages[stage.stage_id] = stage
        self._recompute_order()
        return self

    def _recompute_order(self) -> None:
        """Recompute the topological order of stages."""
        self._stage_order = self._topological_sort()

    def _topological_sort(self) -> list[str]:
        """Compute topological order of stages using Kahn's algorithm.

        Returns:
            List of stage IDs in execution order.

        Raises:
            ValueError: If a cycle is detected.
        """
        # Build in-degree map
        in_degree: dict[str, int] = {stage_id: 0 for stage_id in self._stages}
        for stage in self._stages.values():
            for dep_id in stage.dependencies:
                if dep_id in in_degree:
                    in_degree[stage.stage_id] += 1

        # Start with stages that have no dependencies
        queue: list[str] = [
            stage_id for stage_id, degree in in_degree.items() if degree == 0
        ]
        result: list[str] = []

        while queue:
            # Sort by stage name for deterministic ordering
            queue.sort(key=lambda x: self._stages[x].name)
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for dependent stages
            for stage in self._stages.values():
                if current in stage.dependencies:
                    in_degree[stage.stage_id] -= 1
                    if in_degree[stage.stage_id] == 0:
                        queue.append(stage.stage_id)

        if len(result) != len(self._stages):
            raise ValueError("Cycle detected in pipeline stages")

        return result

    def get_ready_stages(
        self, completed: set[str], running: set[str]
    ) -> list[Stage[Any, Any]]:
        """Get stages that are ready to execute.

        Args:
            completed: Set of completed stage IDs.
            running: Set of currently running stage IDs.

        Returns:
            List of stages ready to execute.
        """
        ready: list[Stage[Any, Any]] = []
        for stage_id in self._stage_order:
            if stage_id in completed or stage_id in running:
                continue

            stage = self._stages[stage_id]
            # Check if all dependencies are completed
            deps_met = all(dep_id in completed for dep_id in stage.dependencies)
            if deps_met:
                ready.append(stage)

        return ready

    async def execute(
        self,
        initial_input: Any = None,
        context: PipelineContext | None = None,
    ) -> PipelineResult:
        """Execute the pipeline.

        Args:
            initial_input: Input to pass to stages without dependencies.
            context: Optional pre-configured context.

        Returns:
            PipelineResult with all stage results.
        """
        async with self._lock:
            if self._running:
                raise RuntimeError("Pipeline is already running")
            self._running = True

        pipeline_id = str(uuid.uuid4())[:8]
        self._context = context or PipelineContext(pipeline_id, self.config)
        self._context.set("_initial_input", initial_input)

        result = PipelineResult(
            pipeline_id=pipeline_id,
            pipeline_name=self.config.name,
            status=PipelineStatus.RUNNING,
            start_time=time.time(),
        )

        logger.info(
            "pipeline.started",
            pipeline_id=pipeline_id,
            name=self.config.name,
            stage_count=len(self._stages),
        )

        try:
            # Execute with global timeout if configured
            if self.config.timeout.pipeline_timeout_seconds:
                try:
                    await asyncio.wait_for(
                        self._execute_stages(result),
                        timeout=self.config.timeout.pipeline_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    raise PipelineTimeoutException(
                        self.config.timeout.pipeline_timeout_seconds
                    )
            else:
                await self._execute_stages(result)

            # Determine final status
            if self._context.is_cancelled:
                result.status = PipelineStatus.CANCELLED
                result.cancellation_reason = self._context.get(
                    "_cancellation_reason", CancellationReason.USER_REQUESTED
                )
            elif result.failed_stages:
                if result.successful_stages:
                    result.status = PipelineStatus.PARTIALLY_COMPLETED
                else:
                    result.status = PipelineStatus.FAILED
            else:
                result.status = PipelineStatus.COMPLETED

        except PipelineCancelledException as e:
            result.status = PipelineStatus.CANCELLED
            result.cancellation_reason = e.reason
            logger.warning(
                "pipeline.cancelled",
                pipeline_id=pipeline_id,
                reason=e.reason.value,
            )
            if self.config.cleanup_on_cancel:
                await self._run_cleanup()

        except PipelineTimeoutException as e:
            result.status = PipelineStatus.FAILED
            result.cancellation_reason = CancellationReason.TIMEOUT
            logger.error(
                "pipeline.timeout",
                pipeline_id=pipeline_id,
                timeout_seconds=e.timeout_seconds,
            )

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.metadata["error"] = str(e)
            result.metadata["traceback"] = traceback.format_exc()
            logger.exception("pipeline.error", pipeline_id=pipeline_id, error=str(e))

        finally:
            result.end_time = time.time()
            result.stage_results = dict(self._context._stage_results)
            self._result = result
            self._running = False

            logger.info(
                "pipeline.finished",
                pipeline_id=pipeline_id,
                status=result.status.name,
                duration_ms=result.duration_ms,
                successful_stages=len(result.successful_stages),
                failed_stages=len(result.failed_stages),
            )

        return result

    async def _execute_stages(self, result: PipelineResult) -> None:
        """Execute all stages in order.

        Args:
            result: The result object to update.
        """
        completed: set[str] = set()
        failed: set[str] = set()
        running: set[str] = set()

        while len(completed) + len(failed) < len(self._stages):
            # Check for cancellation
            self._context.check_cancelled()

            # Wait if paused
            await self._context.wait_if_paused()

            # Get stages ready to execute
            ready_stages = self.get_ready_stages(completed | failed, running)

            if not ready_stages and not running:
                # No stages ready and none running - check for dependency failures
                for stage_id in self._stage_order:
                    if stage_id in completed or stage_id in failed:
                        continue
                    stage = self._stages[stage_id]
                    failed_deps = [d for d in stage.dependencies if d in failed]
                    if failed_deps:
                        # Mark as skipped due to dependency failure
                        stage_result = StageResult[Any](
                            stage_id=stage_id,
                            stage_name=stage.name,
                            status=StageStatus.SKIPPED,
                            error=f"Dependencies failed: {', '.join(failed_deps)}",
                        )
                        self._context.store_stage_result(stage_result)
                        failed.add(stage_id)
                break

            # Determine execution mode
            if self.config.parallel_stages and len(ready_stages) > 1:
                # Execute ready stages in parallel (up to max_parallel)
                batch = ready_stages[: self.config.max_parallel]
                running.update(s.stage_id for s in batch)

                tasks = [self._execute_single_stage(stage) for stage in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for stage, stage_result in zip(batch, results):
                    running.discard(stage.stage_id)

                    if isinstance(stage_result, Exception):
                        # Handle exception
                        failed_result = StageResult[Any](
                            stage_id=stage.stage_id,
                            stage_name=stage.name,
                            status=StageStatus.FAILED,
                            error=str(stage_result),
                            exception=stage_result,
                        )
                        self._context.store_stage_result(failed_result)
                        failed.add(stage.stage_id)

                        if stage.critical and not self.config.continue_on_error:
                            raise StageExecutionException(
                                stage.stage_id, str(stage_result), stage_result
                            )
                    else:
                        if stage_result.is_success:
                            completed.add(stage.stage_id)
                        else:
                            failed.add(stage.stage_id)
                            if stage.critical and not self.config.continue_on_error:
                                raise StageExecutionException(
                                    stage.stage_id,
                                    stage_result.error or "Unknown error",
                                )
            else:
                # Execute sequentially
                for stage in ready_stages[:1]:  # One at a time
                    running.add(stage.stage_id)
                    stage_result = await self._execute_single_stage(stage)
                    running.discard(stage.stage_id)

                    if stage_result.is_success:
                        completed.add(stage.stage_id)
                    else:
                        failed.add(stage.stage_id)
                        if stage.critical and not self.config.continue_on_error:
                            raise StageExecutionException(
                                stage.stage_id,
                                stage_result.error or "Unknown error",
                            )

    async def _execute_single_stage(self, stage: Stage[Any, Any]) -> StageResult[Any]:
        """Execute a single stage with retry and timeout handling.

        Args:
            stage: The stage to execute.

        Returns:
            StageResult with execution details.
        """
        retry_config = stage.retry_config or self.config.retry
        timeout = stage.timeout_seconds or self.config.timeout.stage_timeout_seconds

        attempt = 0
        last_error: Exception | None = None
        start_time = time.time()

        logger.info(
            "stage.started",
            pipeline_id=self._context.pipeline_id,
            stage_id=stage.stage_id,
            stage_name=stage.name,
        )

        while attempt <= retry_config.max_retries:
            attempt += 1
            self._context.check_cancelled()

            try:
                # Get input - from dependency results or initial input
                if stage.dependencies:
                    # Use result from last dependency
                    last_dep = stage.dependencies[-1]
                    dep_result = self._context.get_stage_result(last_dep)
                    stage_input = dep_result.result if dep_result else None
                else:
                    stage_input = self._context.get("_initial_input")

                # Execute with timeout
                if timeout:
                    try:
                        result_value = await asyncio.wait_for(
                            stage.handler(self._context, stage_input),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        raise StageTimeoutException(stage.stage_id, timeout)
                else:
                    result_value = await stage.handler(self._context, stage_input)

                # Success
                stage_result = StageResult[Any](
                    stage_id=stage.stage_id,
                    stage_name=stage.name,
                    status=StageStatus.COMPLETED,
                    result=result_value,
                    start_time=start_time,
                    end_time=time.time(),
                    attempt_count=attempt,
                )
                self._context.store_stage_result(stage_result)

                logger.info(
                    "stage.completed",
                    pipeline_id=self._context.pipeline_id,
                    stage_id=stage.stage_id,
                    stage_name=stage.name,
                    duration_ms=stage_result.duration_ms,
                    attempts=attempt,
                )

                return stage_result

            except StageTimeoutException as e:
                last_error = e
                if retry_config.retry_on_timeout and attempt <= retry_config.max_retries:
                    delay = retry_config.get_delay(attempt - 1)
                    logger.warning(
                        "stage.timeout_retry",
                        stage_id=stage.stage_id,
                        attempt=attempt,
                        delay_seconds=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    break

            except PipelineCancelledException:
                raise

            except Exception as e:
                last_error = e
                should_retry = (
                    retry_config.retry_on_exceptions is None
                    or isinstance(e, retry_config.retry_on_exceptions)
                )

                if should_retry and attempt <= retry_config.max_retries:
                    delay = retry_config.get_delay(attempt - 1)
                    logger.warning(
                        "stage.error_retry",
                        stage_id=stage.stage_id,
                        error=str(e),
                        attempt=attempt,
                        delay_seconds=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    break

        # All attempts failed
        stage_result = StageResult[Any](
            stage_id=stage.stage_id,
            stage_name=stage.name,
            status=(
                StageStatus.TIMED_OUT
                if isinstance(last_error, StageTimeoutException)
                else StageStatus.FAILED
            ),
            error=str(last_error) if last_error else "Unknown error",
            exception=last_error,
            start_time=start_time,
            end_time=time.time(),
            attempt_count=attempt,
        )
        self._context.store_stage_result(stage_result)

        logger.error(
            "stage.failed",
            pipeline_id=self._context.pipeline_id,
            stage_id=stage.stage_id,
            stage_name=stage.name,
            error=str(last_error),
            attempts=attempt,
        )

        return stage_result

    async def _run_cleanup(self) -> None:
        """Run cleanup handlers for all stages."""
        logger.info("pipeline.cleanup_started", pipeline_id=self._context.pipeline_id)

        for stage_id in reversed(self._stage_order):
            stage = self._stages[stage_id]
            if stage.cleanup_handler:
                try:
                    await asyncio.wait_for(
                        stage.cleanup_handler(self._context),
                        timeout=self.config.timeout.cleanup_timeout_seconds,
                    )
                except Exception as e:
                    logger.warning(
                        "stage.cleanup_error",
                        stage_id=stage_id,
                        error=str(e),
                    )

        logger.info("pipeline.cleanup_completed", pipeline_id=self._context.pipeline_id)

    def cancel(
        self, reason: CancellationReason = CancellationReason.USER_REQUESTED
    ) -> None:
        """Request cancellation of the pipeline.

        Args:
            reason: The reason for cancellation.
        """
        if self._context:
            self._context.cancel(reason)

    def pause(self) -> None:
        """Pause the pipeline execution."""
        if self._context:
            self._context.pause()

    def resume(self) -> None:
        """Resume the pipeline execution."""
        if self._context:
            self._context.resume()

    @property
    def is_running(self) -> bool:
        """Check if the pipeline is currently running."""
        return self._running

    @property
    def result(self) -> PipelineResult | None:
        """Get the result of the last execution."""
        return self._result


# =============================================================================
# Utility Functions and Decorators
# =============================================================================


def stage(
    stage_id: str,
    name: str,
    dependencies: list[str] | None = None,
    timeout_seconds: float | None = None,
    critical: bool = True,
    description: str = "",
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Awaitable[R]]], Stage[Any, R]]:
    """Decorator to create a stage from an async function.

    Args:
        stage_id: Unique identifier for the stage.
        name: Human-readable name.
        dependencies: List of stage IDs this stage depends on.
        timeout_seconds: Timeout for this stage.
        critical: Whether failure should stop the pipeline.
        description: Optional description.
        tags: Optional tags for categorization.

    Returns:
        Decorator that creates a Stage from the function.
    """

    def decorator(func: Callable[..., Awaitable[R]]) -> Stage[Any, R]:
        return Stage(
            stage_id=stage_id,
            name=name,
            handler=func,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
            critical=critical,
            description=description or func.__doc__ or "",
            tags=tags or [],
        )

    return decorator


def with_retry(
    max_retries: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_delay: float = 1.0,
) -> Callable[[Callable[..., Awaitable[R]]], Callable[..., Awaitable[R]]]:
    """Decorator to add retry logic to an async function.

    Args:
        max_retries: Maximum number of retry attempts.
        strategy: Retry strategy to use.
        initial_delay: Initial delay before first retry.

    Returns:
        Decorator that adds retry logic.
    """
    config = RetryConfig(
        max_retries=max_retries,
        strategy=strategy,
        initial_delay_seconds=initial_delay,
    )

    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            last_error: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            "retry.attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            delay_seconds=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)

            raise last_error or RuntimeError("Retry failed with no error")

        return wrapper

    return decorator


# =============================================================================
# Builder Pattern for Pipeline Construction
# =============================================================================


class PipelineBuilder:
    """Fluent builder for constructing pipelines."""

    def __init__(self, name: str = "default_pipeline") -> None:
        """Initialize the builder.

        Args:
            name: Name for the pipeline.
        """
        self._config = PipelineConfig(name=name)
        self._stages: list[Stage[Any, Any]] = []

    def with_retry(
        self,
        max_retries: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> "PipelineBuilder":
        """Configure retry behavior.

        Args:
            max_retries: Maximum number of retry attempts.
            strategy: Retry strategy to use.
            initial_delay: Initial delay before first retry.
            max_delay: Maximum delay between retries.

        Returns:
            Self for method chaining.
        """
        self._config.retry = RetryConfig(
            max_retries=max_retries,
            strategy=strategy,
            initial_delay_seconds=initial_delay,
            max_delay_seconds=max_delay,
        )
        return self

    def with_timeout(
        self,
        stage_timeout: float | None = 300.0,
        pipeline_timeout: float | None = 3600.0,
    ) -> "PipelineBuilder":
        """Configure timeout behavior.

        Args:
            stage_timeout: Default timeout per stage in seconds.
            pipeline_timeout: Total pipeline timeout in seconds.

        Returns:
            Self for method chaining.
        """
        self._config.timeout = TimeoutConfig(
            stage_timeout_seconds=stage_timeout,
            pipeline_timeout_seconds=pipeline_timeout,
        )
        return self

    def with_parallel(
        self, enabled: bool = True, max_parallel: int = 4
    ) -> "PipelineBuilder":
        """Configure parallel execution.

        Args:
            enabled: Whether to enable parallel execution.
            max_parallel: Maximum number of parallel stages.

        Returns:
            Self for method chaining.
        """
        self._config.parallel_stages = enabled
        self._config.max_parallel = max_parallel
        return self

    def continue_on_error(self, enabled: bool = True) -> "PipelineBuilder":
        """Configure error handling.

        Args:
            enabled: Whether to continue on non-critical errors.

        Returns:
            Self for method chaining.
        """
        self._config.continue_on_error = enabled
        return self

    def add_stage(
        self,
        stage_id: str,
        name: str,
        handler: Callable[[PipelineContext, Any], Awaitable[Any]],
        dependencies: list[str] | None = None,
        timeout: float | None = None,
        critical: bool = True,
    ) -> "PipelineBuilder":
        """Add a stage to the pipeline.

        Args:
            stage_id: Unique identifier for the stage.
            name: Human-readable name.
            handler: The async function to execute.
            dependencies: List of stage IDs this stage depends on.
            timeout: Timeout for this stage.
            critical: Whether failure should stop the pipeline.

        Returns:
            Self for method chaining.
        """
        self._stages.append(
            Stage(
                stage_id=stage_id,
                name=name,
                handler=handler,
                dependencies=dependencies or [],
                timeout_seconds=timeout,
                critical=critical,
            )
        )
        return self

    def build(self) -> Pipeline:
        """Build and return the pipeline.

        Returns:
            Configured Pipeline instance.
        """
        pipeline = Pipeline(self._config)
        pipeline.add_stages(self._stages)
        return pipeline


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_pipeline(
    stages: list[Stage[Any, Any]],
    initial_input: Any = None,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Convenience function to create and run a pipeline.

    Args:
        stages: List of stages to execute.
        initial_input: Input to pass to stages without dependencies.
        config: Optional pipeline configuration.

    Returns:
        PipelineResult with all stage results.
    """
    pipeline = Pipeline(config)
    pipeline.add_stages(stages)
    return await pipeline.execute(initial_input)


def create_stage(
    stage_id: str,
    name: str,
    handler: Callable[[PipelineContext, Any], Awaitable[Any]],
    dependencies: list[str] | None = None,
    timeout_seconds: float | None = None,
    critical: bool = True,
) -> Stage[Any, Any]:
    """Convenience function to create a stage.

    Args:
        stage_id: Unique identifier for the stage.
        name: Human-readable name.
        handler: The async function to execute.
        dependencies: List of stage IDs this stage depends on.
        timeout_seconds: Timeout for this stage.
        critical: Whether failure should stop the pipeline.

    Returns:
        Stage instance.
    """
    return Stage(
        stage_id=stage_id,
        name=name,
        handler=handler,
        dependencies=dependencies or [],
        timeout_seconds=timeout_seconds,
        critical=critical,
    )


# =============================================================================
# Data Flow Coordination - Inter-stage data passing
# =============================================================================


@dataclass
class DataFlowChannel(Generic[T]):
    """Channel for coordinated data flow between pipeline stages.

    Provides a typed channel for passing data between stages with
    buffering, backpressure, and synchronization capabilities.
    """

    name: str
    capacity: int = 10
    _buffer: list[T] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _not_full: threading.Condition = field(default_factory=lambda: threading.Condition())
    _not_empty: threading.Condition = field(default_factory=lambda: threading.Condition())
    _closed: bool = False

    def __post_init__(self) -> None:
        """Initialize conditions with proper locks."""
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def send(self, item: T, timeout: float | None = None) -> bool:
        """Send an item to the channel.

        Args:
            item: Item to send.
            timeout: Optional timeout in seconds.

        Returns:
            True if sent successfully, False if channel closed or timeout.
        """
        with self._not_full:
            start = time.time()
            while len(self._buffer) >= self.capacity:
                if self._closed:
                    return False
                remaining = None
                if timeout is not None:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        return False
                if not self._not_full.wait(timeout=remaining):
                    return False
            
            if self._closed:
                return False
            
            self._buffer.append(item)
            self._not_empty.notify()
            return True

    def receive(self, timeout: float | None = None) -> tuple[T | None, bool]:
        """Receive an item from the channel.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            Tuple of (item, success). Item is None if closed or timeout.
        """
        with self._not_empty:
            start = time.time()
            while not self._buffer:
                if self._closed:
                    return None, False
                remaining = None
                if timeout is not None:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        return None, False
                if not self._not_empty.wait(timeout=remaining):
                    return None, False
            
            item = self._buffer.pop(0)
            self._not_full.notify()
            return item, True

    def close(self) -> None:
        """Close the channel."""
        with self._lock:
            self._closed = True
            self._not_full.notify_all()
            self._not_empty.notify_all()

    @property
    def is_closed(self) -> bool:
        """Check if channel is closed."""
        return self._closed

    def __len__(self) -> int:
        """Get number of items in buffer."""
        with self._lock:
            return len(self._buffer)


class DataFlowCoordinator:
    """Coordinates data flow between multiple pipeline stages.

    Provides:
    - Named channels for inter-stage communication
    - Fan-out/fan-in patterns
    - Data transformation pipelines
    - Backpressure handling
    """

    def __init__(self) -> None:
        """Initialize the coordinator."""
        self._channels: dict[str, DataFlowChannel[Any]] = {}
        self._transforms: dict[str, Callable[[Any], Any]] = {}
        self._lock = threading.Lock()

    def create_channel(
        self,
        name: str,
        capacity: int = 10,
    ) -> DataFlowChannel[Any]:
        """Create a named data flow channel.

        Args:
            name: Channel name.
            capacity: Buffer capacity.

        Returns:
            The created channel.
        """
        with self._lock:
            if name in self._channels:
                raise ValueError(f"Channel '{name}' already exists")
            channel: DataFlowChannel[Any] = DataFlowChannel(name=name, capacity=capacity)
            self._channels[name] = channel
            return channel

    def get_channel(self, name: str) -> DataFlowChannel[Any] | None:
        """Get a channel by name.

        Args:
            name: Channel name.

        Returns:
            The channel or None if not found.
        """
        with self._lock:
            return self._channels.get(name)

    def register_transform(
        self,
        name: str,
        transform: Callable[[Any], Any],
    ) -> None:
        """Register a data transformation function.

        Args:
            name: Transform name.
            transform: Transformation function.
        """
        with self._lock:
            self._transforms[name] = transform

    def apply_transform(self, name: str, data: Any) -> Any:
        """Apply a registered transformation.

        Args:
            name: Transform name.
            data: Data to transform.

        Returns:
            Transformed data.

        Raises:
            KeyError: If transform not found.
        """
        with self._lock:
            transform = self._transforms.get(name)
            if not transform:
                raise KeyError(f"Transform '{name}' not found")
        return transform(data)

    def fan_out(
        self,
        source_channel: str,
        target_channels: list[str],
        transform: Callable[[Any], list[Any]] | None = None,
    ) -> None:
        """Send data from one channel to multiple channels.

        Args:
            source_channel: Source channel name.
            target_channels: List of target channel names.
            transform: Optional function to split data for each target.
        """
        source = self.get_channel(source_channel)
        targets = [self.get_channel(name) for name in target_channels]

        if not source or not all(targets):
            raise ValueError("Invalid channel names")

        item, ok = source.receive()
        if not ok:
            return

        if transform:
            items = transform(item)
            for target, data in zip(targets, items):
                if target:
                    target.send(data)
        else:
            for target in targets:
                if target:
                    target.send(item)

    def fan_in(
        self,
        source_channels: list[str],
        target_channel: str,
        aggregator: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        """Collect data from multiple channels into one.

        Args:
            source_channels: List of source channel names.
            target_channel: Target channel name.
            aggregator: Optional function to aggregate collected data.
        """
        sources = [self.get_channel(name) for name in source_channels]
        target = self.get_channel(target_channel)

        if not all(sources) or not target:
            raise ValueError("Invalid channel names")

        collected = []
        for source in sources:
            if source:
                item, ok = source.receive(timeout=5.0)
                if ok:
                    collected.append(item)

        if aggregator:
            result = aggregator(collected)
            target.send(result)
        else:
            for item in collected:
                target.send(item)

    def close_all(self) -> None:
        """Close all channels."""
        with self._lock:
            for channel in self._channels.values():
                channel.close()


# =============================================================================
# Pipeline Checkpointing - Save/Restore State
# =============================================================================


@dataclass
class PipelineCheckpoint:
    """Checkpoint of pipeline execution state.

    Allows saving and restoring pipeline progress for:
    - Pause/resume functionality
    - Crash recovery
    - Long-running pipelines
    """

    checkpoint_id: str
    pipeline_id: str
    pipeline_name: str
    timestamp: float
    completed_stages: list[str]
    failed_stages: list[str]
    stage_results: dict[str, dict[str, Any]]
    context_data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "timestamp": self.timestamp,
            "completed_stages": self.completed_stages,
            "failed_stages": self.failed_stages,
            "stage_results": self.stage_results,
            "context_data": self.context_data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineCheckpoint":
        """Create checkpoint from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            pipeline_id=data["pipeline_id"],
            pipeline_name=data["pipeline_name"],
            timestamp=data["timestamp"],
            completed_stages=data["completed_stages"],
            failed_stages=data["failed_stages"],
            stage_results=data["stage_results"],
            context_data=data["context_data"],
            metadata=data.get("metadata", {}),
        )


class CheckpointManager:
    """Manages pipeline checkpoints for persistence and recovery."""

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize the checkpoint manager.

        Args:
            storage_path: Path to store checkpoints. Uses temp dir if None.
        """
        import tempfile
        from pathlib import Path
        
        if storage_path:
            self._storage_path = Path(storage_path)
        else:
            self._storage_path = Path(tempfile.gettempdir()) / "proxima_checkpoints"
        
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._checkpoints: dict[str, PipelineCheckpoint] = {}
        self._lock = threading.Lock()

    def save_checkpoint(
        self,
        pipeline: Pipeline,
        context: PipelineContext,
        completed: set[str],
        failed: set[str],
    ) -> PipelineCheckpoint:
        """Save a checkpoint of current pipeline state.

        Args:
            pipeline: The pipeline being executed.
            context: Current execution context.
            completed: Set of completed stage IDs.
            failed: Set of failed stage IDs.

        Returns:
            Created checkpoint.
        """
        import json

        checkpoint_id = str(uuid.uuid4())[:8]
        
        # Serialize stage results
        stage_results = {}
        for stage_id, result in context._stage_results.items():
            stage_results[stage_id] = result.to_dict()

        checkpoint = PipelineCheckpoint(
            checkpoint_id=checkpoint_id,
            pipeline_id=context.pipeline_id,
            pipeline_name=pipeline.config.name,
            timestamp=time.time(),
            completed_stages=list(completed),
            failed_stages=list(failed),
            stage_results=stage_results,
            context_data=dict(context._data),
        )

        with self._lock:
            self._checkpoints[checkpoint_id] = checkpoint
            
            # Persist to disk
            checkpoint_file = self._storage_path / f"{checkpoint_id}.json"
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(
            "checkpoint.saved",
            checkpoint_id=checkpoint_id,
            pipeline_id=context.pipeline_id,
            completed_count=len(completed),
        )

        return checkpoint

    def load_checkpoint(self, checkpoint_id: str) -> PipelineCheckpoint | None:
        """Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID to load.

        Returns:
            Loaded checkpoint or None if not found.
        """
        import json

        with self._lock:
            if checkpoint_id in self._checkpoints:
                return self._checkpoints[checkpoint_id]

        # Try loading from disk
        checkpoint_file = self._storage_path / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                data = json.load(f)
                checkpoint = PipelineCheckpoint.from_dict(data)
                with self._lock:
                    self._checkpoints[checkpoint_id] = checkpoint
                return checkpoint

        return None

    def list_checkpoints(
        self,
        pipeline_id: str | None = None,
    ) -> list[PipelineCheckpoint]:
        """List all checkpoints, optionally filtered by pipeline.

        Args:
            pipeline_id: Optional pipeline ID filter.

        Returns:
            List of matching checkpoints.
        """
        import json

        checkpoints = []
        
        # Load from disk
        for checkpoint_file in self._storage_path.glob("*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                    checkpoint = PipelineCheckpoint.from_dict(data)
                    if pipeline_id is None or checkpoint.pipeline_id == pipeline_id:
                        checkpoints.append(checkpoint)
            except (json.JSONDecodeError, KeyError):
                continue

        return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            self._checkpoints.pop(checkpoint_id, None)

        checkpoint_file = self._storage_path / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True

        return False

    def cleanup_old_checkpoints(
        self,
        max_age_seconds: float = 86400,
    ) -> int:
        """Remove checkpoints older than max age.

        Args:
            max_age_seconds: Maximum age in seconds.

        Returns:
            Number of checkpoints removed.
        """
        cutoff = time.time() - max_age_seconds
        removed = 0

        for checkpoint_file in self._storage_path.glob("*.json"):
            try:
                if checkpoint_file.stat().st_mtime < cutoff:
                    checkpoint_file.unlink()
                    removed += 1
            except OSError:
                continue

        return removed


# =============================================================================
# Progress Tracking - Real-time updates
# =============================================================================


@dataclass
class ProgressUpdate:
    """Progress update event for pipeline execution."""

    pipeline_id: str
    stage_id: str | None
    stage_name: str | None
    progress_percent: float
    message: str
    timestamp: float = field(default_factory=time.time)
    metrics: dict[str, Any] = field(default_factory=dict)


class ProgressCallback(ABC):
    """Abstract base class for progress callbacks."""

    @abstractmethod
    def on_progress(self, update: ProgressUpdate) -> None:
        """Called when progress is updated.

        Args:
            update: Progress update event.
        """
        pass

    @abstractmethod
    def on_stage_start(self, stage_id: str, stage_name: str) -> None:
        """Called when a stage starts.

        Args:
            stage_id: Stage ID.
            stage_name: Stage name.
        """
        pass

    @abstractmethod
    def on_stage_complete(
        self,
        stage_id: str,
        stage_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Called when a stage completes.

        Args:
            stage_id: Stage ID.
            stage_name: Stage name.
            success: Whether stage succeeded.
            duration_ms: Execution duration in milliseconds.
        """
        pass


class ConsoleProgressCallback(ProgressCallback):
    """Progress callback that prints to console."""

    def __init__(self, show_spinner: bool = True) -> None:
        """Initialize the callback.

        Args:
            show_spinner: Whether to show a spinner for running stages.
        """
        self._show_spinner = show_spinner
        self._spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0

    def on_progress(self, update: ProgressUpdate) -> None:
        """Print progress update."""
        if self._show_spinner:
            spinner = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
            self._spinner_idx += 1
        else:
            spinner = "→"
        
        print(f"\r{spinner} [{update.progress_percent:5.1f}%] {update.message}", end="", flush=True)

    def on_stage_start(self, stage_id: str, stage_name: str) -> None:
        """Print stage start."""
        print(f"\n▶ Starting: {stage_name} ({stage_id})")

    def on_stage_complete(
        self,
        stage_id: str,
        stage_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Print stage completion."""
        icon = "✓" if success else "✗"
        status = "completed" if success else "failed"
        print(f"\n{icon} {stage_name} {status} in {duration_ms:.2f}ms")


class ProgressTracker:
    """Tracks and broadcasts pipeline progress."""

    def __init__(self) -> None:
        """Initialize the progress tracker."""
        self._callbacks: list[ProgressCallback] = []
        self._stage_count: int = 0
        self._completed_count: int = 0
        self._current_stage: str | None = None
        self._lock = threading.Lock()

    def add_callback(self, callback: ProgressCallback) -> None:
        """Add a progress callback.

        Args:
            callback: Callback to add.
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: ProgressCallback) -> None:
        """Remove a progress callback.

        Args:
            callback: Callback to remove.
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def set_stage_count(self, count: int) -> None:
        """Set total number of stages.

        Args:
            count: Total stage count.
        """
        with self._lock:
            self._stage_count = count

    def notify_stage_start(self, stage_id: str, stage_name: str) -> None:
        """Notify callbacks of stage start.

        Args:
            stage_id: Stage ID.
            stage_name: Stage name.
        """
        with self._lock:
            self._current_stage = stage_id
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback.on_stage_start(stage_id, stage_name)
            except Exception as e:
                logger.warning("progress_callback.error", error=str(e))

    def notify_stage_complete(
        self,
        stage_id: str,
        stage_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Notify callbacks of stage completion.

        Args:
            stage_id: Stage ID.
            stage_name: Stage name.
            success: Whether stage succeeded.
            duration_ms: Execution duration.
        """
        with self._lock:
            self._completed_count += 1
            self._current_stage = None
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback.on_stage_complete(stage_id, stage_name, success, duration_ms)
            except Exception as e:
                logger.warning("progress_callback.error", error=str(e))

    def notify_progress(
        self,
        pipeline_id: str,
        message: str,
        progress_override: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Broadcast a progress update.

        Args:
            pipeline_id: Pipeline ID.
            message: Progress message.
            progress_override: Override calculated progress percentage.
            metrics: Optional metrics to include.
        """
        with self._lock:
            if progress_override is not None:
                progress = progress_override
            elif self._stage_count > 0:
                progress = (self._completed_count / self._stage_count) * 100
            else:
                progress = 0.0
            
            update = ProgressUpdate(
                pipeline_id=pipeline_id,
                stage_id=self._current_stage,
                stage_name=None,
                progress_percent=progress,
                message=message,
                metrics=metrics or {},
            )
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback.on_progress(update)
            except Exception as e:
                logger.warning("progress_callback.error", error=str(e))

    @property
    def progress_percent(self) -> float:
        """Get current progress percentage."""
        with self._lock:
            if self._stage_count == 0:
                return 0.0
            return (self._completed_count / self._stage_count) * 100


# =============================================================================
# Pipeline Aggregator - Combine multiple pipelines
# =============================================================================


class PipelineAggregator:
    """Aggregates and coordinates multiple pipelines.

    Provides:
    - Sequential pipeline execution
    - Parallel pipeline execution
    - Data passing between pipelines
    - Combined result aggregation
    """

    def __init__(self, name: str = "aggregated_pipeline") -> None:
        """Initialize the aggregator.

        Args:
            name: Name for the aggregated pipeline.
        """
        self.name = name
        self._pipelines: list[tuple[str, Pipeline]] = []
        self._results: dict[str, PipelineResult] = {}

    def add_pipeline(self, name: str, pipeline: Pipeline) -> "PipelineAggregator":
        """Add a pipeline to the aggregator.

        Args:
            name: Name for this pipeline instance.
            pipeline: Pipeline to add.

        Returns:
            Self for method chaining.
        """
        self._pipelines.append((name, pipeline))
        return self

    async def execute_sequential(
        self,
        initial_input: Any = None,
        pass_results: bool = True,
    ) -> dict[str, PipelineResult]:
        """Execute pipelines sequentially.

        Args:
            initial_input: Input for first pipeline.
            pass_results: Whether to pass output of one pipeline to next.

        Returns:
            Dictionary of pipeline names to results.
        """
        current_input = initial_input
        
        for name, pipeline in self._pipelines:
            logger.info(
                "aggregator.executing_pipeline",
                aggregator=self.name,
                pipeline=name,
            )
            
            result = await pipeline.execute(current_input)
            self._results[name] = result
            
            if not result.is_success:
                logger.warning(
                    "aggregator.pipeline_failed",
                    aggregator=self.name,
                    pipeline=name,
                    status=result.status.name,
                )
                break
            
            if pass_results:
                # Pass last stage result as input to next pipeline
                last_stage = result.successful_stages[-1] if result.successful_stages else None
                if last_stage:
                    last_result = result.stage_results.get(last_stage)
                    current_input = last_result.result if last_result else None

        return self._results

    async def execute_parallel(
        self,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, PipelineResult]:
        """Execute all pipelines in parallel.

        Args:
            inputs: Optional dict of pipeline name to input.

        Returns:
            Dictionary of pipeline names to results.
        """
        inputs = inputs or {}
        
        async def run_pipeline(name: str, pipeline: Pipeline) -> tuple[str, PipelineResult]:
            input_data = inputs.get(name)
            result = await pipeline.execute(input_data)
            return name, result

        tasks = [run_pipeline(name, pipeline) for name, pipeline in self._pipelines]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error("aggregator.parallel_error", error=str(result))
            else:
                name, pipeline_result = result
                self._results[name] = pipeline_result

        return self._results

    def get_combined_summary(self) -> str:
        """Generate a combined summary of all pipeline results.

        Returns:
            Summary string.
        """
        lines = [
            "=" * 60,
            f"AGGREGATED PIPELINE: {self.name}",
            "=" * 60,
            f"Total Pipelines: {len(self._pipelines)}",
            f"Executed: {len(self._results)}",
        ]
        
        successful = sum(1 for r in self._results.values() if r.is_success)
        lines.append(f"Successful: {successful}")
        lines.append(f"Failed: {len(self._results) - successful}")
        lines.append("")
        
        total_duration = sum(r.duration_ms for r in self._results.values())
        lines.append(f"Total Duration: {total_duration:.2f}ms")
        lines.append("")
        lines.append("PIPELINE DETAILS:")
        
        for name, result in self._results.items():
            status_icon = "✓" if result.is_success else "✗"
            lines.append(f"  {status_icon} {name}: {result.status.name} ({result.duration_ms:.2f}ms)")
        
        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def results(self) -> dict[str, PipelineResult]:
        """Get all pipeline results."""
        return self._results

    @property
    def all_successful(self) -> bool:
        """Check if all pipelines succeeded."""
        return all(r.is_success for r in self._results.values())
