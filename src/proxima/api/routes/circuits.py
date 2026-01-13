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
