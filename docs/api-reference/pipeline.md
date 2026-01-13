# Data Pipeline API Reference

Complete API documentation for the `proxima.data.pipeline` module.

## Overview

The data pipeline module provides a robust framework for stage-based execution with:

- **Stage Management**: Define and chain execution stages
- **Timeout Handling**: Per-stage and global timeouts
- **Retry Mechanisms**: Configurable retry strategies
- **Cancellation Support**: Graceful cancellation with cleanup
- **Parallel Execution**: Run independent stages concurrently

---

## Enums

### StageStatus

Status of a pipeline stage.

```python
from proxima.data.pipeline import StageStatus

class StageStatus(Enum):
    PENDING = auto()      # Not yet started
    QUEUED = auto()       # Waiting to execute
    RUNNING = auto()      # Currently executing
    COMPLETED = auto()    # Successfully completed
    FAILED = auto()       # Execution failed
    CANCELLED = auto()    # Cancelled by user
    SKIPPED = auto()      # Skipped due to dependency failure
    RETRYING = auto()     # Retrying after failure
    TIMED_OUT = auto()    # Exceeded timeout
```

### PipelineStatus

Status of the entire pipeline.

```python
from proxima.data.pipeline import PipelineStatus

class PipelineStatus(Enum):
    IDLE = auto()                 # Not started
    RUNNING = auto()              # Currently executing
    PAUSED = auto()               # Paused by user
    COMPLETED = auto()            # Successfully completed
    FAILED = auto()               # Execution failed
    CANCELLED = auto()            # Cancelled by user
    PARTIALLY_COMPLETED = auto()  # Some stages completed
```

### RetryStrategy

Retry strategy for failed stages.

```python
from proxima.data.pipeline import RetryStrategy

class RetryStrategy(Enum):
    NONE = "none"                           # No retry
    IMMEDIATE = "immediate"                 # Retry immediately
    LINEAR_BACKOFF = "linear_backoff"       # Linear delay increase
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential delay
    CONSTANT = "constant"                   # Fixed delay between retries
```

### CancellationReason

Reason for pipeline or stage cancellation.

```python
from proxima.data.pipeline import CancellationReason

class CancellationReason(Enum):
    USER_REQUESTED = "user_requested"
    TIMEOUT = "timeout"
    DEPENDENCY_FAILED = "dependency_failed"
    RESOURCE_LIMIT = "resource_limit"
    ERROR = "error"
    EXTERNAL = "external"
```

---

## Configuration Classes

### RetryConfig

Configuration for retry behavior.

```python
from proxima.data.pipeline import RetryConfig, RetryStrategy

config = RetryConfig(
    max_retries=3,                    # Maximum retry attempts
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_delay_seconds=1.0,        # Initial delay before first retry
    max_delay_seconds=60.0,           # Maximum delay cap
    backoff_factor=2.0,               # Multiplier for backoff
    retry_on_exceptions=None,         # Exception types to retry (None = all)
    retry_on_timeout=True,            # Whether to retry on timeout
)

# Calculate delay for attempt
delay = config.get_delay(attempt=2)  # Returns delay in seconds
```

### TimeoutConfig

Configuration for timeout behavior.

```python
from proxima.data.pipeline import TimeoutConfig

config = TimeoutConfig(
    stage_timeout_seconds=300.0,      # 5 minutes per stage
    pipeline_timeout_seconds=3600.0,  # 1 hour total
    cleanup_timeout_seconds=30.0,     # Timeout for cleanup
    grace_period_seconds=5.0,         # Grace period before force termination
)
```

### PipelineConfig

Overall pipeline configuration.

```python
from proxima.data.pipeline import PipelineConfig, RetryConfig, TimeoutConfig

config = PipelineConfig(
    name="my_pipeline",
    retry=RetryConfig(max_retries=3),
    timeout=TimeoutConfig(stage_timeout_seconds=60.0),
    continue_on_error=False,          # Stop on first error
    parallel_stages=True,             # Run independent stages in parallel
    max_parallel=4,                   # Max concurrent stages
    collect_metrics=True,             # Collect detailed metrics
    cleanup_on_cancel=True,           # Run cleanup on cancellation
)
```

---

## Result Classes

### StageResult

Result of executing a pipeline stage.

```python
from proxima.data.pipeline import StageResult, StageStatus

result = StageResult(
    stage_id="process_data",
    stage_name="Process Data",
    status=StageStatus.COMPLETED,
    result={"processed": 100},        # The actual result value
    error=None,                       # Error message if failed
    exception=None,                   # Exception object if failed
    start_time=1704067200.0,
    end_time=1704067210.0,
    attempt_count=1,
    metrics={"rows_processed": 100},
)

# Properties
print(result.duration_ms)    # Duration in milliseconds
print(result.is_success)     # True if completed successfully
print(result.to_dict())      # Dictionary representation
```

### PipelineResult

Result of executing an entire pipeline.

```python
from proxima.data.pipeline import PipelineResult, PipelineStatus

result = PipelineResult(
    pipeline_id="abc123",
    pipeline_name="data_pipeline",
    status=PipelineStatus.COMPLETED,
    stage_results={...},              # Dict of stage_id -> StageResult
    start_time=1704067200.0,
    end_time=1704067300.0,
    cancellation_reason=None,
    metadata={},
)

# Properties
print(result.duration_ms)           # Total duration in ms
print(result.successful_stages)     # List of successful stage IDs
print(result.failed_stages)         # List of failed stage IDs
print(result.is_success)            # True if all stages completed
print(result.summary())             # Human-readable summary
print(result.to_dict())             # Dictionary representation
```

---

## Core Classes

### Stage

Definition of a pipeline stage.

```python
from proxima.data.pipeline import Stage, PipelineContext, RetryConfig

async def my_handler(ctx: PipelineContext, input_data) -> dict:
    # Process input_data
    ctx.set("intermediate_result", {"key": "value"})
    return {"processed": True}

async def my_cleanup(ctx: PipelineContext) -> None:
    # Cleanup resources
    pass

stage = Stage(
    stage_id="process",
    name="Process Data",
    handler=my_handler,
    dependencies=["load_data"],       # Stage IDs this depends on
    timeout_seconds=60.0,             # Override default timeout
    retry_config=RetryConfig(max_retries=2),
    critical=True,                    # If True, failure stops pipeline
    cleanup_handler=my_cleanup,
    description="Process loaded data",
    tags=["data", "processing"],
)
```

### PipelineContext

Shared context for pipeline execution.

```python
from proxima.data.pipeline import PipelineContext, PipelineConfig, CancellationReason

context = PipelineContext(
    pipeline_id="abc123",
    config=PipelineConfig(name="test"),
)

# Store and retrieve data
context.set("key", "value")
value = context.get("key", default=None)

# Access previous stage results
prev_result = context.get_stage_result("previous_stage")

# Cancellation
context.cancel(CancellationReason.USER_REQUESTED)
if context.is_cancelled:
    print("Pipeline was cancelled")

# Check for cancellation (raises PipelineCancelledException)
context.check_cancelled()

# Pause/Resume
context.pause()
context.resume()
await context.wait_if_paused()
```

### Pipeline

Orchestrates execution of pipeline stages.

```python
from proxima.data.pipeline import Pipeline, PipelineConfig, Stage

# Create pipeline
config = PipelineConfig(name="my_pipeline")
pipeline = Pipeline(config)

# Add stages
pipeline.add_stage(stage1)
pipeline.add_stages([stage2, stage3])

# Execute
result = await pipeline.execute(initial_input={"data": "..."})

# Check status
print(result.status)
print(result.is_success)

# Control execution
pipeline.cancel()   # Request cancellation
pipeline.pause()    # Pause execution
pipeline.resume()   # Resume execution

# Properties
print(pipeline.is_running)
print(pipeline.result)  # Last execution result
```

---

## PipelineBuilder

Fluent builder for constructing pipelines.

```python
from proxima.data.pipeline import PipelineBuilder, RetryStrategy

async def stage_a(ctx, _):
    return {"a": 1}

async def stage_b(ctx, a_result):
    return {"b": a_result["a"] * 2}

pipeline = (
    PipelineBuilder("my_pipeline")
    .with_retry(
        max_retries=3,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        initial_delay=1.0,
        max_delay=60.0,
    )
    .with_timeout(
        stage_timeout=300.0,
        pipeline_timeout=3600.0,
    )
    .with_parallel(enabled=True, max_parallel=4)
    .continue_on_error(enabled=False)
    .add_stage("a", "Stage A", stage_a)
    .add_stage("b", "Stage B", stage_b, dependencies=["a"])
    .build()
)

result = await pipeline.execute()
```

---

## Decorators

### @stage

Decorator to create a stage from an async function.

```python
from proxima.data.pipeline import stage

@stage(
    stage_id="process",
    name="Process Data",
    dependencies=["load"],
    timeout_seconds=60.0,
    critical=True,
)
async def process_data(ctx, input_data):
    """Process the loaded data."""
    return {"processed": True}

# Use directly as a Stage object
pipeline.add_stage(process_data)
```

### @with_retry

Decorator to add retry logic to an async function.

```python
from proxima.data.pipeline import with_retry, RetryStrategy

@with_retry(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    initial_delay=1.0,
)
async def unreliable_operation():
    # This will retry up to 3 times on failure
    return await external_service.call()
```

---

## Convenience Functions

### run_pipeline

Convenience function to create and run a pipeline.

```python
from proxima.data.pipeline import run_pipeline, Stage

stages = [
    Stage(stage_id="a", name="A", handler=handler_a),
    Stage(stage_id="b", name="B", handler=handler_b, dependencies=["a"]),
]

result = await run_pipeline(
    stages=stages,
    initial_input={"data": "..."},
    config=PipelineConfig(name="quick_pipeline"),
)
```

### create_stage

Convenience function to create a stage.

```python
from proxima.data.pipeline import create_stage

stage = create_stage(
    stage_id="process",
    name="Process Data",
    handler=my_handler,
    dependencies=["load"],
    timeout_seconds=60.0,
    critical=True,
)
```

---

## Exceptions

### PipelineException

Base exception for pipeline errors.

```python
from proxima.data.pipeline import PipelineException

try:
    result = await pipeline.execute()
except PipelineException as e:
    print(f"Pipeline error: {e}")
```

### StageTimeoutException

Raised when a stage times out.

```python
from proxima.data.pipeline import StageTimeoutException

try:
    result = await pipeline.execute()
except StageTimeoutException as e:
    print(f"Stage {e.stage_id} timed out after {e.timeout_seconds}s")
```

### PipelineTimeoutException

Raised when the entire pipeline times out.

```python
from proxima.data.pipeline import PipelineTimeoutException

try:
    result = await pipeline.execute()
except PipelineTimeoutException as e:
    print(f"Pipeline timed out after {e.timeout_seconds}s")
```

### PipelineCancelledException

Raised when the pipeline is cancelled.

```python
from proxima.data.pipeline import PipelineCancelledException

try:
    result = await pipeline.execute()
except PipelineCancelledException as e:
    print(f"Pipeline cancelled: {e.reason.value}")
```

### StageExecutionException

Raised when a stage fails execution.

```python
from proxima.data.pipeline import StageExecutionException

try:
    result = await pipeline.execute()
except StageExecutionException as e:
    print(f"Stage {e.stage_id} failed: {e.error}")
    if e.cause:
        print(f"Caused by: {e.cause}")
```

### DependencyFailedException

Raised when a stage's dependency failed.

```python
from proxima.data.pipeline import DependencyFailedException

try:
    result = await pipeline.execute()
except DependencyFailedException as e:
    print(f"Stage {e.stage_id} skipped: dependency {e.failed_dependency} failed")
```

---

## Examples

### Basic Pipeline

```python
from proxima.data.pipeline import Pipeline, PipelineConfig, Stage

async def load_data(ctx, _):
    data = await fetch_from_source()
    ctx.set("raw_data", data)
    return data

async def process_data(ctx, data):
    processed = transform(data)
    return processed

async def save_results(ctx, processed):
    await store_results(processed)
    return {"saved": True}

config = PipelineConfig(name="etl_pipeline")
pipeline = Pipeline(config)

pipeline.add_stages([
    Stage(stage_id="load", name="Load Data", handler=load_data),
    Stage(stage_id="process", name="Process", handler=process_data, dependencies=["load"]),
    Stage(stage_id="save", name="Save", handler=save_results, dependencies=["process"]),
])

result = await pipeline.execute()
print(result.summary())
```

### Pipeline with Error Handling

```python
from proxima.data.pipeline import (
    PipelineBuilder,
    RetryStrategy,
    PipelineStatus,
)

async def flaky_stage(ctx, _):
    # May fail randomly
    if random.random() < 0.3:
        raise ConnectionError("Network error")
    return {"success": True}

pipeline = (
    PipelineBuilder("robust_pipeline")
    .with_retry(max_retries=5, strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
    .with_timeout(stage_timeout=30.0, pipeline_timeout=300.0)
    .continue_on_error(enabled=True)
    .add_stage("flaky", "Flaky Stage", flaky_stage, critical=False)
    .add_stage("important", "Important Stage", important_handler, critical=True)
    .build()
)

result = await pipeline.execute()

if result.status == PipelineStatus.COMPLETED:
    print("All stages completed successfully")
elif result.status == PipelineStatus.PARTIALLY_COMPLETED:
    print(f"Some stages failed: {result.failed_stages}")
else:
    print(f"Pipeline failed: {result.status.name}")
```

### Parallel Stage Execution

```python
from proxima.data.pipeline import PipelineBuilder

async def process_backend(ctx, _, backend_name):
    result = await execute_on_backend(backend_name)
    return {backend_name: result}

pipeline = (
    PipelineBuilder("parallel_comparison")
    .with_parallel(enabled=True, max_parallel=3)
    .add_stage("cirq", "Cirq Backend", lambda c, i: process_backend(c, i, "cirq"))
    .add_stage("qiskit", "Qiskit Backend", lambda c, i: process_backend(c, i, "qiskit"))
    .add_stage("lret", "LRET Backend", lambda c, i: process_backend(c, i, "lret"))
    .add_stage("compare", "Compare Results", compare_handler, 
               dependencies=["cirq", "qiskit", "lret"])
    .build()
)

result = await pipeline.execute()
```
