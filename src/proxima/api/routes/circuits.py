"""
Circuit execution endpoints.

Provides REST API for submitting and executing quantum circuits.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


# =============================================================================
# Enums and Models
# =============================================================================

class CircuitFormat(str, Enum):
    """Supported circuit input formats."""
    
    OPENQASM = "openqasm"
    QISKIT_JSON = "qiskit_json"
    CIRQ_JSON = "cirq_json"
    PROXIMA = "proxima"


class ExecutionStatus(str, Enum):
    """Circuit execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CircuitSubmitRequest(BaseModel):
    """Request to submit a circuit for execution."""
    
    circuit: str = Field(..., description="Circuit definition")
    format: CircuitFormat = Field(default=CircuitFormat.OPENQASM, description="Circuit format")
    backend: str = Field(default="auto", description="Backend to use (or 'auto' for automatic selection)")
    shots: int = Field(default=1000, ge=1, le=100000, description="Number of measurement shots")
    options: dict[str, Any] = Field(default_factory=dict, description="Additional execution options")
    async_execution: bool = Field(default=False, description="Execute asynchronously")


class CircuitSubmitResponse(BaseModel):
    """Response from circuit submission."""
    
    job_id: str
    status: ExecutionStatus
    backend: str
    submitted_at: str
    message: str | None = None


class ExecutionResult(BaseModel):
    """Circuit execution result."""
    
    job_id: str
    status: ExecutionStatus
    backend: str
    counts: dict[str, int] | None = None
    statevector: list[complex] | None = None
    probabilities: dict[str, float] | None = None
    execution_time_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JobStatusResponse(BaseModel):
    """Job status response."""
    
    job_id: str
    status: ExecutionStatus
    backend: str
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress: float | None = None
    error: str | None = None


class CircuitValidateRequest(BaseModel):
    """Request to validate a circuit."""
    
    circuit: str = Field(..., description="Circuit definition")
    format: CircuitFormat = Field(default=CircuitFormat.OPENQASM, description="Circuit format")
    backend: str = Field(default="auto", description="Target backend for validation")


class CircuitValidateResponse(BaseModel):
    """Circuit validation response."""
    
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    num_qubits: int | None = None
    num_gates: int | None = None
    circuit_depth: int | None = None
    unsupported_gates: list[str] = Field(default_factory=list)


class CircuitOptimizeRequest(BaseModel):
    """Request to optimize a circuit."""
    
    circuit: str = Field(..., description="Circuit definition")
    format: CircuitFormat = Field(default=CircuitFormat.OPENQASM, description="Circuit format")
    backend: str = Field(default="auto", description="Target backend for optimization")
    optimization_level: int = Field(default=1, ge=0, le=3, description="Optimization level (0-3)")


class CircuitOptimizeResponse(BaseModel):
    """Circuit optimization response."""
    
    optimized_circuit: str
    format: CircuitFormat
    original_gates: int
    optimized_gates: int
    original_depth: int
    optimized_depth: int
    reduction_percentage: float


# =============================================================================
# In-memory job storage (for demo purposes)
# =============================================================================

_jobs: dict[str, dict[str, Any]] = {}


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/submit", response_model=CircuitSubmitResponse)
async def submit_circuit(
    submit_request: CircuitSubmitRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> CircuitSubmitResponse:
    """Submit a circuit for execution.
    
    The circuit can be executed synchronously or asynchronously.
    For async execution, use the job_id to poll for status.
    
    Args:
        submit_request: Circuit submission request.
        background_tasks: FastAPI background tasks.
        request: FastAPI request object.
    
    Returns:
        CircuitSubmitResponse: Submission confirmation with job ID.
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    
    # Determine backend
    backend = submit_request.backend
    if backend == "auto":
        # Auto-select based on circuit characteristics
        backend = "cirq"  # Default for now
    
    # Create job record
    job = {
        "job_id": job_id,
        "status": ExecutionStatus.PENDING,
        "backend": backend,
        "circuit": submit_request.circuit,
        "format": submit_request.format,
        "shots": submit_request.shots,
        "options": submit_request.options,
        "submitted_at": now.isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    _jobs[job_id] = job
    
    if submit_request.async_execution:
        # Queue for background execution
        background_tasks.add_task(_execute_circuit_async, job_id, request)
        message = "Circuit queued for execution"
    else:
        # Execute synchronously
        await _execute_circuit(job_id, request)
        job = _jobs[job_id]
        message = "Circuit executed successfully" if job["status"] == ExecutionStatus.COMPLETED else job.get("error")
    
    job = _jobs[job_id]
    return CircuitSubmitResponse(
        job_id=job_id,
        status=job["status"],
        backend=backend,
        submitted_at=job["submitted_at"],
        message=message,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of a submitted job.
    
    Args:
        job_id: The job identifier.
    
    Returns:
        JobStatusResponse: Current job status.
    
    Raises:
        HTTPException: If job not found.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = _jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        backend=job["backend"],
        submitted_at=job["submitted_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        progress=job.get("progress"),
        error=job.get("error"),
    )


@router.get("/jobs/{job_id}/result", response_model=ExecutionResult)
async def get_job_result(job_id: str) -> ExecutionResult:
    """Get the result of a completed job.
    
    Args:
        job_id: The job identifier.
    
    Returns:
        ExecutionResult: Execution results.
    
    Raises:
        HTTPException: If job not found or not completed.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = _jobs[job_id]
    
    if job["status"] not in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete. Current status: {job['status']}"
        )
    
    result = job.get("result", {})
    
    return ExecutionResult(
        job_id=job_id,
        status=job["status"],
        backend=job["backend"],
        counts=result.get("counts"),
        statevector=result.get("statevector"),
        probabilities=result.get("probabilities"),
        execution_time_ms=result.get("execution_time_ms"),
        metadata=result.get("metadata", {}),
        error=job.get("error"),
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict[str, str]:
    """Cancel a pending or running job.
    
    Args:
        job_id: The job identifier.
    
    Returns:
        dict: Cancellation confirmation.
    
    Raises:
        HTTPException: If job not found or cannot be cancelled.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = _jobs[job_id]
    
    if job["status"] in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be cancelled. Current status: {job['status']}"
        )
    
    job["status"] = ExecutionStatus.CANCELLED
    job["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    return {"message": f"Job '{job_id}' cancelled successfully"}


@router.post("/validate", response_model=CircuitValidateResponse)
async def validate_circuit(
    validate_request: CircuitValidateRequest,
    request: Request,
) -> CircuitValidateResponse:
    """Validate a circuit without executing it.
    
    Checks circuit syntax, gate support, and resource requirements.
    
    Args:
        validate_request: Validation request.
        request: FastAPI request object.
    
    Returns:
        CircuitValidateResponse: Validation results.
    """
    errors: list[str] = []
    warnings: list[str] = []
    unsupported_gates: list[str] = []
    
    # Parse circuit based on format
    num_qubits = None
    num_gates = None
    circuit_depth = None
    
    try:
        if validate_request.format == CircuitFormat.OPENQASM:
            # Basic QASM validation
            circuit = validate_request.circuit
            
            # Count qubits
            import re
            qreg_match = re.search(r'qreg\s+\w+\[(\d+)\]', circuit)
            if qreg_match:
                num_qubits = int(qreg_match.group(1))
            
            # Count gates (rough estimate)
            gate_patterns = ['h ', 'x ', 'y ', 'z ', 'cx ', 'cz ', 'rx', 'ry', 'rz', 't ', 's ']
            num_gates = sum(circuit.lower().count(g) for g in gate_patterns)
            
            # Estimate depth
            circuit_depth = num_gates // max(1, num_qubits or 1) + 1
            
            # Check for unsupported gates
            if 'ccx' in circuit.lower() or 'toffoli' in circuit.lower():
                warnings.append("Toffoli gates may be decomposed on some backends")
            
        else:
            warnings.append(f"Format {validate_request.format} validation is limited")
        
        valid = len(errors) == 0
        
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        valid = False
    
    return CircuitValidateResponse(
        valid=valid,
        errors=errors,
        warnings=warnings,
        num_qubits=num_qubits,
        num_gates=num_gates,
        circuit_depth=circuit_depth,
        unsupported_gates=unsupported_gates,
    )


@router.post("/optimize", response_model=CircuitOptimizeResponse)
async def optimize_circuit(
    optimize_request: CircuitOptimizeRequest,
    request: Request,
) -> CircuitOptimizeResponse:
    """Optimize a circuit for a specific backend.
    
    Applies gate optimization, decomposition, and other transformations.
    
    Args:
        optimize_request: Optimization request.
        request: FastAPI request object.
    
    Returns:
        CircuitOptimizeResponse: Optimized circuit and statistics.
    """
    # For demo purposes, return a mock optimization
    original_gates = 100
    original_depth = 50
    
    # Simulate optimization based on level
    level = optimize_request.optimization_level
    reduction_factor = 1 - (level * 0.15)  # 15% reduction per level
    
    optimized_gates = int(original_gates * reduction_factor)
    optimized_depth = int(original_depth * reduction_factor)
    
    reduction_percentage = (1 - (optimized_gates / original_gates)) * 100
    
    return CircuitOptimizeResponse(
        optimized_circuit=optimize_request.circuit,  # Return original for now
        format=optimize_request.format,
        original_gates=original_gates,
        optimized_gates=optimized_gates,
        original_depth=original_depth,
        optimized_depth=optimized_depth,
        reduction_percentage=reduction_percentage,
    )


# =============================================================================
# Helper Functions
# =============================================================================

async def _execute_circuit(job_id: str, request: Request) -> None:
    """Execute a circuit synchronously."""
    import time
    
    if job_id not in _jobs:
        return
    
    job = _jobs[job_id]
    job["status"] = ExecutionStatus.RUNNING
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    
    try:
        start_time = time.perf_counter()
        
        # Simulate execution (replace with actual backend call)
        import random
        await _simulate_execution_delay()
        
        # Generate mock results
        num_qubits = 2
        num_states = 2 ** num_qubits
        counts = {}
        remaining = job["shots"]
        
        for i in range(num_states - 1):
            count = random.randint(0, remaining)
            if count > 0:
                state = format(i, f'0{num_qubits}b')
                counts[state] = count
                remaining -= count
        
        if remaining > 0:
            state = format(num_states - 1, f'0{num_qubits}b')
            counts[state] = remaining
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        job["status"] = ExecutionStatus.COMPLETED
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["result"] = {
            "counts": counts,
            "execution_time_ms": execution_time,
            "metadata": {
                "backend": job["backend"],
                "shots": job["shots"],
            },
        }
        
    except Exception as e:
        job["status"] = ExecutionStatus.FAILED
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["error"] = str(e)


async def _execute_circuit_async(job_id: str, request: Request) -> None:
    """Execute a circuit asynchronously (background task)."""
    await _execute_circuit(job_id, request)


async def _simulate_execution_delay() -> None:
    """Simulate execution delay for demo purposes."""
    import asyncio
    await asyncio.sleep(0.1)  # 100ms delay


# =============================================================================
# WebSocket Polling and Long-Polling Endpoints
# =============================================================================


class PollRequest(BaseModel):
    """Request for long-polling job status."""
    
    job_ids: list[str] = Field(
        ...,
        description="List of job IDs to poll",
        max_length=100,
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=60.0,
        description="Maximum time to wait for updates",
    )
    last_states: dict[str, str] = Field(
        default_factory=dict,
        description="Last known states for each job (for change detection)",
    )


class PollUpdate(BaseModel):
    """Update for a single job."""
    
    job_id: str
    status: ExecutionStatus
    progress: float | None = None
    changed: bool = False
    completed: bool = False
    error: str | None = None


class PollResponse(BaseModel):
    """Response from long-polling."""
    
    updates: list[PollUpdate]
    has_changes: bool
    poll_time_ms: float
    next_poll_recommended_ms: int


class StreamUpdate(BaseModel):
    """Update for streaming endpoint."""
    
    type: str  # "progress", "status", "result", "error"
    job_id: str
    timestamp: str
    data: dict[str, Any] = Field(default_factory=dict)


@router.post("/jobs/poll", response_model=PollResponse)
async def poll_jobs(poll_request: PollRequest) -> PollResponse:
    """Long-poll for job status updates.
    
    This endpoint allows efficient polling of multiple jobs at once.
    It will return immediately if any job has changed state, or wait
    up to timeout_seconds for changes.
    
    This is more efficient than individual polling for each job and
    provides a fallback when WebSocket connections are not available.
    
    Args:
        poll_request: Polling request with job IDs and timeout.
    
    Returns:
        PollResponse: Status updates for requested jobs.
    
    Example:
        ```python
        # Client-side polling loop
        last_states = {}
        while any_incomplete:
            response = requests.post("/api/v1/circuits/jobs/poll", json={
                "job_ids": ["job1", "job2"],
                "timeout_seconds": 30,
                "last_states": last_states
            })
            for update in response.json()["updates"]:
                last_states[update["job_id"]] = update["status"]
                if update["changed"]:
                    print(f"Job {update['job_id']} is now {update['status']}")
        ```
    """
    import asyncio
    import time
    
    start_time = time.perf_counter()
    timeout = poll_request.timeout_seconds
    check_interval = 0.1  # 100ms
    
    updates: list[PollUpdate] = []
    has_changes = False
    
    # Initial check
    for job_id in poll_request.job_ids:
        job = _jobs.get(job_id)
        if job is None:
            updates.append(PollUpdate(
                job_id=job_id,
                status=ExecutionStatus.FAILED,
                error="Job not found",
                changed=True,
                completed=True,
            ))
            has_changes = True
        else:
            current_status = job["status"]
            last_status = poll_request.last_states.get(job_id)
            changed = last_status != current_status.value
            completed = current_status in [
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            ]
            
            updates.append(PollUpdate(
                job_id=job_id,
                status=current_status,
                progress=job.get("progress"),
                changed=changed,
                completed=completed,
                error=job.get("error"),
            ))
            
            if changed:
                has_changes = True
    
    # If changes detected immediately, return
    if has_changes:
        poll_time = (time.perf_counter() - start_time) * 1000
        return PollResponse(
            updates=updates,
            has_changes=True,
            poll_time_ms=poll_time,
            next_poll_recommended_ms=1000,  # Poll again in 1 second
        )
    
    # Long-poll: wait for changes up to timeout
    elapsed = 0.0
    while elapsed < timeout:
        await asyncio.sleep(check_interval)
        elapsed = time.perf_counter() - start_time
        
        # Check for changes
        for i, job_id in enumerate(poll_request.job_ids):
            job = _jobs.get(job_id)
            if job is None:
                continue
            
            current_status = job["status"]
            last_status = poll_request.last_states.get(job_id)
            
            if last_status != current_status.value:
                # Update the response
                updates[i] = PollUpdate(
                    job_id=job_id,
                    status=current_status,
                    progress=job.get("progress"),
                    changed=True,
                    completed=current_status in [
                        ExecutionStatus.COMPLETED,
                        ExecutionStatus.FAILED,
                        ExecutionStatus.CANCELLED,
                    ],
                    error=job.get("error"),
                )
                has_changes = True
        
        if has_changes:
            break
    
    poll_time = (time.perf_counter() - start_time) * 1000
    
    # Determine recommended poll interval based on job states
    all_complete = all(u.completed for u in updates)
    if all_complete:
        next_poll = 0  # No more polling needed
    elif has_changes:
        next_poll = 500  # Changes detected, poll soon
    else:
        next_poll = 5000  # No changes, poll less frequently
    
    return PollResponse(
        updates=updates,
        has_changes=has_changes,
        poll_time_ms=poll_time,
        next_poll_recommended_ms=next_poll,
    )


@router.get("/jobs/{job_id}/wait", response_model=ExecutionResult)
async def wait_for_job(
    job_id: str,
    timeout_seconds: float = 60.0,
) -> ExecutionResult:
    """Wait for a job to complete.
    
    This endpoint will block until the job completes or times out.
    Useful for simple synchronous workflows where you want to wait
    for results without implementing polling logic.
    
    Args:
        job_id: The job identifier.
        timeout_seconds: Maximum time to wait (default: 60s, max: 300s).
    
    Returns:
        ExecutionResult: Job results when complete.
    
    Raises:
        HTTPException: If job not found or times out.
    """
    import asyncio
    import time
    
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    # Clamp timeout
    timeout = min(max(timeout_seconds, 1.0), 300.0)
    
    start_time = time.perf_counter()
    check_interval = 0.1  # 100ms
    
    while True:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        
        if job["status"] in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
            # Job complete
            result = job.get("result", {})
            return ExecutionResult(
                job_id=job_id,
                status=job["status"],
                backend=job["backend"],
                counts=result.get("counts"),
                statevector=result.get("statevector"),
                probabilities=result.get("probabilities"),
                execution_time_ms=result.get("execution_time_ms"),
                metadata=result.get("metadata", {}),
                error=job.get("error"),
            )
        
        # Check timeout
        elapsed = time.perf_counter() - start_time
        if elapsed >= timeout:
            raise HTTPException(
                status_code=408,
                detail=f"Timeout waiting for job '{job_id}'. Current status: {job['status']}"
            )
        
        await asyncio.sleep(check_interval)


class SubscriptionRequest(BaseModel):
    """Request to create a polling subscription."""
    
    job_ids: list[str] = Field(..., description="Jobs to subscribe to")
    webhook_url: str | None = Field(None, description="Optional webhook for push notifications")
    include_progress: bool = Field(True, description="Include progress updates")


class SubscriptionResponse(BaseModel):
    """Polling subscription info."""
    
    subscription_id: str
    job_ids: list[str]
    expires_at: str
    poll_endpoint: str
    websocket_topic: str


# In-memory subscription storage
_subscriptions: dict[str, dict[str, Any]] = {}


@router.post("/jobs/subscribe", response_model=SubscriptionResponse)
async def create_subscription(
    subscription_request: SubscriptionRequest,
) -> SubscriptionResponse:
    """Create a subscription for job updates.
    
    Creates a subscription that groups multiple jobs together
    for efficient polling. The subscription provides both a
    polling endpoint and a WebSocket topic.
    
    Args:
        subscription_request: Subscription configuration.
    
    Returns:
        SubscriptionResponse: Subscription details with endpoints.
    """
    subscription_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    
    from datetime import timedelta
    expires_at = now + timedelta(hours=1)
    
    subscription = {
        "subscription_id": subscription_id,
        "job_ids": subscription_request.job_ids,
        "webhook_url": subscription_request.webhook_url,
        "include_progress": subscription_request.include_progress,
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "last_poll": None,
    }
    
    _subscriptions[subscription_id] = subscription
    
    return SubscriptionResponse(
        subscription_id=subscription_id,
        job_ids=subscription_request.job_ids,
        expires_at=expires_at.isoformat(),
        poll_endpoint=f"/api/v1/circuits/subscriptions/{subscription_id}/poll",
        websocket_topic=f"subscription:{subscription_id}",
    )


@router.get("/subscriptions/{subscription_id}/poll")
async def poll_subscription(
    subscription_id: str,
    timeout_seconds: float = 30.0,
) -> PollResponse:
    """Poll a subscription for updates.
    
    Convenience endpoint that polls all jobs in a subscription.
    
    Args:
        subscription_id: The subscription identifier.
        timeout_seconds: Maximum time to wait for updates.
    
    Returns:
        PollResponse: Updates for all jobs in subscription.
    """
    if subscription_id not in _subscriptions:
        raise HTTPException(
            status_code=404,
            detail=f"Subscription '{subscription_id}' not found"
        )
    
    subscription = _subscriptions[subscription_id]
    
    # Check expiration
    expires_at = datetime.fromisoformat(subscription["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        del _subscriptions[subscription_id]
        raise HTTPException(
            status_code=410,
            detail="Subscription has expired"
        )
    
    # Get last known states
    last_states = subscription.get("last_states", {})
    
    # Create poll request
    poll_request = PollRequest(
        job_ids=subscription["job_ids"],
        timeout_seconds=min(timeout_seconds, 60.0),
        last_states=last_states,
    )
    
    response = await poll_jobs(poll_request)
    
    # Update subscription state
    subscription["last_poll"] = datetime.now(timezone.utc).isoformat()
    subscription["last_states"] = {
        u.job_id: u.status.value for u in response.updates
    }
    
    return response


@router.delete("/subscriptions/{subscription_id}")
async def delete_subscription(subscription_id: str) -> dict[str, str]:
    """Delete a subscription.
    
    Args:
        subscription_id: The subscription identifier.
    
    Returns:
        Confirmation message.
    """
    if subscription_id not in _subscriptions:
        raise HTTPException(
            status_code=404,
            detail=f"Subscription '{subscription_id}' not found"
        )
    
    del _subscriptions[subscription_id]
    
    return {"message": f"Subscription '{subscription_id}' deleted"}


@router.get("/jobs/{job_id}/events")
async def get_job_events(
    job_id: str,
    since: str | None = None,
) -> list[StreamUpdate]:
    """Get event history for a job.
    
    Returns all events that have occurred for a job, useful
    for reconstructing state or debugging.
    
    Args:
        job_id: The job identifier.
        since: Only return events after this timestamp.
    
    Returns:
        List of events for the job.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = _jobs[job_id]
    
    events = [
        StreamUpdate(
            type="submitted",
            job_id=job_id,
            timestamp=job["submitted_at"],
            data={"backend": job["backend"], "shots": job["shots"]},
        ),
    ]
    
    if job.get("started_at"):
        events.append(StreamUpdate(
            type="started",
            job_id=job_id,
            timestamp=job["started_at"],
            data={"status": "running"},
        ))
    
    if job.get("completed_at"):
        result = job.get("result", {})
        events.append(StreamUpdate(
            type="completed" if job["status"] == ExecutionStatus.COMPLETED else "failed",
            job_id=job_id,
            timestamp=job["completed_at"],
            data={
                "status": job["status"].value,
                "execution_time_ms": result.get("execution_time_ms"),
                "error": job.get("error"),
            },
        ))
    
    # Filter by since timestamp if provided
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            events = [
                e for e in events
                if datetime.fromisoformat(e.timestamp.replace("Z", "+00:00")) > since_dt
            ]
        except ValueError:
            pass  # Ignore invalid timestamps
    
    return events
