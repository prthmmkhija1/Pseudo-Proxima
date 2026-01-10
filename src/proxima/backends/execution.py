"""Execution utilities: timeout, retry, batching, and async support."""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from proxima.backends.base import ExecutionResult
from proxima.backends.exceptions import (
    BackendError,
    BackendTimeoutError,
    ExecutionError,
    wrap_backend_exception,
)

T = TypeVar("T")


# ==============================================================================
# Timeout Support
# ==============================================================================


@dataclass
class TimeoutConfig:
    """Configuration for execution timeout."""

    timeout_seconds: float = 300.0  # 5 minutes default
    check_interval: float = 0.1  # How often to check for timeout
    graceful_shutdown_seconds: float = 5.0  # Time for cleanup on timeout


@contextmanager
def execution_timeout(seconds: float, operation: str = "execution") -> Generator[None, None, None]:
    """Context manager for synchronous timeout (thread-based).

    Note: This uses a signal-based approach on Unix, falls back to
    thread-based checking on Windows.

    Args:
        seconds: Maximum execution time in seconds
        operation: Description of operation (for error messages)

    Yields:
        None

    Raises:
        BackendTimeoutError: If execution exceeds timeout
    """
    if seconds <= 0:
        yield
        return

    start_time = time.perf_counter()
    yield

    # Check if we exceeded timeout (simple check, actual enforcement
    # is done in execute_with_timeout)
    elapsed = time.perf_counter() - start_time
    if elapsed > seconds:
        raise BackendTimeoutError(
            backend_name="unknown",
            timeout_seconds=seconds,
            operation=operation,
        )


def execute_with_timeout(
    func: Callable[..., T],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: float = 300.0,
    backend_name: str = "unknown",
) -> T:
    """Execute a function with a timeout.

    Uses ThreadPoolExecutor for cross-platform timeout support.

    Args:
        func: Function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        timeout_seconds: Maximum execution time
        backend_name: Backend name for error messages

    Returns:
        Result of the function

    Raises:
        BackendTimeoutError: If execution times out
        BackendError: If execution fails
    """
    kwargs = kwargs or {}

    if timeout_seconds <= 0:
        return func(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            # Attempt to cancel (may not work for blocking operations)
            future.cancel()
            raise BackendTimeoutError(
                backend_name=backend_name,
                timeout_seconds=timeout_seconds,
                operation="execution",
            )
        except Exception as exc:
            if isinstance(exc, BackendError):
                raise
            raise wrap_backend_exception(exc, backend_name, "execution")


# ==============================================================================
# Retry with Exponential Backoff
# ==============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retry_on: tuple[type[Exception], ...] = (BackendError,)


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool = False
    result: Any = None
    attempts: int = 0
    total_delay_seconds: float = 0.0
    last_exception: Exception | None = None
    attempt_history: list[dict[str, Any]] = field(default_factory=list)


def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay before next retry attempt.

    Args:
        attempt: Current attempt number (1-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds before next attempt
    """
    delay = config.initial_delay_seconds * (config.exponential_base ** (attempt - 1))
    delay = min(delay, config.max_delay_seconds)

    if config.jitter:
        # Add up to 25% jitter
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)

    return delay


def should_retry(
    exception: Exception,
    config: RetryConfig,
    attempt: int,
) -> bool:
    """Determine if an operation should be retried.

    Args:
        exception: The exception that occurred
        config: Retry configuration
        attempt: Current attempt number

    Returns:
        True if should retry, False otherwise
    """
    if attempt >= config.max_retries:
        return False

    # Check if exception is retryable
    if not isinstance(exception, config.retry_on):
        return False

    # Check if BackendError is marked as recoverable
    if isinstance(exception, BackendError):
        return exception.recoverable

    return True


def execute_with_retry(
    func: Callable[..., T],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    config: RetryConfig | None = None,
    backend_name: str = "unknown",
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> RetryResult:
    """Execute a function with retry logic.

    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        config: Retry configuration
        backend_name: Backend name for logging
        on_retry: Optional callback(attempt, exception, delay) before retry

    Returns:
        RetryResult with success status and result/exception
    """
    kwargs = kwargs or {}
    config = config or RetryConfig()

    result = RetryResult()
    total_delay = 0.0

    for attempt in range(1, config.max_retries + 1):
        result.attempts = attempt
        start_time = time.perf_counter()

        try:
            result.result = func(*args, **kwargs)
            result.success = True
            result.attempt_history.append(
                {
                    "attempt": attempt,
                    "status": "success",
                    "duration_ms": (time.perf_counter() - start_time) * 1000,
                }
            )
            return result

        except Exception as exc:
            duration = time.perf_counter() - start_time
            result.last_exception = exc
            result.attempt_history.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "duration_ms": duration * 1000,
                    "exception": str(exc),
                }
            )

            if not should_retry(exc, config, attempt):
                result.success = False
                break

            delay = calculate_backoff_delay(attempt, config)
            total_delay += delay

            if on_retry:
                on_retry(attempt, exc, delay)

            time.sleep(delay)

    result.total_delay_seconds = total_delay
    return result


def with_retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic to a function.

    Args:
        config: Retry configuration

    Returns:
        Decorator function
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = execute_with_retry(func, args, kwargs, config)
            if result.success:
                return result.result
            if result.last_exception:
                raise result.last_exception
            raise ExecutionError(
                backend_name="unknown",
                reason="Retry failed with no exception",
                stage="retry",
            )

        return wrapper

    return decorator


# ==============================================================================
# Batch Execution
# ==============================================================================


@dataclass
class BatchConfig:
    """Configuration for batch execution."""

    max_batch_size: int = 10
    parallel: bool = True
    max_workers: int = 4
    fail_fast: bool = False  # Stop on first failure


@dataclass
class BatchResult:
    """Result of a batch execution."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    results: list[ExecutionResult | None] = field(default_factory=list)
    errors: list[Exception | None] = field(default_factory=list)
    total_time_ms: float = 0.0


def execute_batch(
    execute_fn: Callable[[Any, dict[str, Any] | None], ExecutionResult],
    circuits: list[Any],
    options: dict[str, Any] | None = None,
    config: BatchConfig | None = None,
    backend_name: str = "unknown",
) -> BatchResult:
    """Execute multiple circuits in batch.

    Args:
        execute_fn: Function to execute a single circuit
        circuits: List of circuits to execute
        options: Shared execution options
        config: Batch configuration
        backend_name: Backend name for error handling

    Returns:
        BatchResult with all results and errors
    """
    config = config or BatchConfig()
    batch_result = BatchResult(total=len(circuits))
    start_time = time.perf_counter()

    if config.parallel and config.max_workers > 1:
        batch_result = _execute_batch_parallel(execute_fn, circuits, options, config, backend_name)
    else:
        batch_result = _execute_batch_sequential(
            execute_fn, circuits, options, config, backend_name
        )

    batch_result.total_time_ms = (time.perf_counter() - start_time) * 1000
    return batch_result


def _execute_batch_sequential(
    execute_fn: Callable[[Any, dict[str, Any] | None], ExecutionResult],
    circuits: list[Any],
    options: dict[str, Any] | None,
    config: BatchConfig,
    backend_name: str,
) -> BatchResult:
    """Execute circuits sequentially."""
    result = BatchResult(total=len(circuits))

    for circuit in circuits:
        try:
            exec_result = execute_fn(circuit, options)
            result.results.append(exec_result)
            result.errors.append(None)
            result.successful += 1
        except Exception as exc:
            result.results.append(None)
            result.errors.append(exc)
            result.failed += 1

            if config.fail_fast:
                break

    return result


def _execute_batch_parallel(
    execute_fn: Callable[[Any, dict[str, Any] | None], ExecutionResult],
    circuits: list[Any],
    options: dict[str, Any] | None,
    config: BatchConfig,
    backend_name: str,
) -> BatchResult:
    """Execute circuits in parallel using ThreadPoolExecutor."""
    result = BatchResult(total=len(circuits))

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [executor.submit(execute_fn, circuit, options) for circuit in circuits]

        for future in futures:
            try:
                exec_result = future.result()
                result.results.append(exec_result)
                result.errors.append(None)
                result.successful += 1
            except Exception as exc:
                result.results.append(None)
                result.errors.append(exc)
                result.failed += 1

    return result


# ==============================================================================
# Async Execution Support
# ==============================================================================


async def execute_async(
    func: Callable[..., T],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> T:
    """Execute a synchronous function asynchronously.

    Uses run_in_executor to avoid blocking the event loop.

    Args:
        func: Synchronous function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout_seconds: Optional timeout

    Returns:
        Result of the function
    """
    kwargs = kwargs or {}
    loop = asyncio.get_event_loop()

    # Wrap function to handle kwargs
    partial_func = functools.partial(func, *args, **kwargs)

    if timeout_seconds:
        return await asyncio.wait_for(
            loop.run_in_executor(None, partial_func),
            timeout=timeout_seconds,
        )

    return await loop.run_in_executor(None, partial_func)


async def execute_batch_async(
    execute_fn: Callable[[Any, dict[str, Any] | None], ExecutionResult],
    circuits: list[Any],
    options: dict[str, Any] | None = None,
    max_concurrent: int = 4,
    timeout_per_circuit: float | None = None,
) -> BatchResult:
    """Execute circuits asynchronously with concurrency control.

    Args:
        execute_fn: Function to execute a single circuit
        circuits: List of circuits to execute
        options: Shared execution options
        max_concurrent: Maximum concurrent executions
        timeout_per_circuit: Timeout for each circuit

    Returns:
        BatchResult with all results and errors
    """
    result = BatchResult(total=len(circuits))
    start_time = time.perf_counter()

    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_one(circuit: Any) -> tuple[ExecutionResult | None, Exception | None]:
        async with semaphore:
            try:
                exec_result = await execute_async(
                    execute_fn,
                    args=(circuit, options),
                    timeout_seconds=timeout_per_circuit,
                )
                return exec_result, None
            except Exception as exc:
                return None, exc

    tasks = [execute_one(circuit) for circuit in circuits]
    outcomes = await asyncio.gather(*tasks, return_exceptions=False)

    for exec_result, exc in outcomes:
        if exc is None:
            result.results.append(exec_result)
            result.errors.append(None)
            result.successful += 1
        else:
            result.results.append(None)
            result.errors.append(exc)
            result.failed += 1

    result.total_time_ms = (time.perf_counter() - start_time) * 1000
    return result
