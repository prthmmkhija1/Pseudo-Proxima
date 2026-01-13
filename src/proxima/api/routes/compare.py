"""
Backend comparison endpoints.

Provides REST API for comparing quantum backends.
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
# Models
# =============================================================================

class ComparisonStatus(str, Enum):
    """Comparison job status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ComparisonRequest(BaseModel):
    """Request to compare backends."""
    
    circuit: str = Field(..., description="Circuit definition (OpenQASM)")
    backends: list[str] = Field(..., min_length=2, max_length=6, description="Backends to compare")
    shots: int = Field(default=1000, ge=1, le=100000, description="Number of shots per backend")
    options: dict[str, Any] = Field(default_factory=dict, description="Additional options")
    async_execution: bool = Field(default=False, description="Run comparison asynchronously")


class BackendResult(BaseModel):
    """Result from a single backend."""
    
    backend: str
    success: bool
    execution_time_ms: float | None = None
    counts: dict[str, int] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonMetrics(BaseModel):
    """Comparison metrics between backends."""
    
    fidelity_matrix: dict[str, dict[str, float]] | None = None
    statistical_distance: dict[str, dict[str, float]] | None = None
    fastest_backend: str | None = None
    most_accurate_backend: str | None = None
    speedup_ratios: dict[str, float] = Field(default_factory=dict)


class ComparisonResponse(BaseModel):
    """Comparison job response."""
    
    comparison_id: str
    status: ComparisonStatus
    created_at: str
    completed_at: str | None = None
    backends_requested: list[str]
    results: list[BackendResult] = Field(default_factory=list)
    metrics: ComparisonMetrics | None = None
    summary: str | None = None


class ComparisonListResponse(BaseModel):
    """List of comparisons."""
    
    comparisons: list[ComparisonResponse]
    total: int


class QuickCompareRequest(BaseModel):
    """Quick comparison request for simple circuits."""
    
    num_qubits: int = Field(default=2, ge=1, le=10, description="Number of qubits")
    circuit_type: str = Field(default="bell", description="Circuit type: bell, ghz, random")
    backends: list[str] = Field(default_factory=lambda: ["cirq", "qiskit_aer"], description="Backends")
    shots: int = Field(default=1000, ge=1, le=10000, description="Number of shots")


# =============================================================================
# In-memory storage
# =============================================================================

_comparisons: dict[str, dict[str, Any]] = {}


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=ComparisonResponse, status_code=201)
async def create_comparison(
    compare_request: ComparisonRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> ComparisonResponse:
    """Create a new backend comparison.
    
    Executes the same circuit on multiple backends and compares results.
    
    Args:
        compare_request: Comparison request.
        background_tasks: FastAPI background tasks.
        request: FastAPI request object.
    
    Returns:
        ComparisonResponse: Comparison job information.
    """
    comparison_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    
    comparison = {
        "comparison_id": comparison_id,
        "status": ComparisonStatus.PENDING,
        "created_at": now.isoformat(),
        "completed_at": None,
        "backends_requested": compare_request.backends,
        "circuit": compare_request.circuit,
        "shots": compare_request.shots,
        "options": compare_request.options,
        "results": [],
        "metrics": None,
        "summary": None,
    }
    
    _comparisons[comparison_id] = comparison
    
    if compare_request.async_execution:
        background_tasks.add_task(_run_comparison, comparison_id, request)
    else:
        await _run_comparison(comparison_id, request)
        comparison = _comparisons[comparison_id]
    
    return _build_comparison_response(comparison)


@router.get("", response_model=ComparisonListResponse)
async def list_comparisons(
    limit: int = 50,
    offset: int = 0,
) -> ComparisonListResponse:
    """List all comparisons.
    
    Args:
        limit: Maximum number of comparisons.
        offset: Number to skip.
    
    Returns:
        ComparisonListResponse: List of comparisons.
    """
    comparisons = list(_comparisons.values())
    comparisons.sort(key=lambda c: c["created_at"], reverse=True)
    
    total = len(comparisons)
    comparisons_page = comparisons[offset:offset + limit]
    
    return ComparisonListResponse(
        comparisons=[_build_comparison_response(c) for c in comparisons_page],
        total=total,
    )


@router.get("/{comparison_id}", response_model=ComparisonResponse)
async def get_comparison(comparison_id: str) -> ComparisonResponse:
    """Get comparison details.
    
    Args:
        comparison_id: The comparison identifier.
    
    Returns:
        ComparisonResponse: Comparison details.
    
    Raises:
        HTTPException: If comparison not found.
    """
    if comparison_id not in _comparisons:
        raise HTTPException(status_code=404, detail=f"Comparison '{comparison_id}' not found")
    
    return _build_comparison_response(_comparisons[comparison_id])


@router.delete("/{comparison_id}")
async def delete_comparison(comparison_id: str) -> dict[str, str]:
    """Delete a comparison.
    
    Args:
        comparison_id: The comparison identifier.
    
    Returns:
        dict: Confirmation message.
    
    Raises:
        HTTPException: If comparison not found.
    """
    if comparison_id not in _comparisons:
        raise HTTPException(status_code=404, detail=f"Comparison '{comparison_id}' not found")
    
    del _comparisons[comparison_id]
    
    return {"message": f"Comparison '{comparison_id}' deleted successfully"}


@router.post("/quick", response_model=ComparisonResponse)
async def quick_compare(
    quick_request: QuickCompareRequest,
    request: Request,
) -> ComparisonResponse:
    """Perform a quick comparison with a predefined circuit.
    
    Useful for quick backend evaluation without providing a custom circuit.
    
    Args:
        quick_request: Quick comparison request.
        request: FastAPI request object.
    
    Returns:
        ComparisonResponse: Comparison results.
    """
    # Generate circuit based on type
    circuit = _generate_circuit(quick_request.circuit_type, quick_request.num_qubits)
    
    # Create full comparison request
    full_request = ComparisonRequest(
        circuit=circuit,
        backends=quick_request.backends,
        shots=quick_request.shots,
        async_execution=False,
    )
    
    from fastapi import BackgroundTasks
    return await create_comparison(full_request, BackgroundTasks(), request)


@router.get("/{comparison_id}/report")
async def get_comparison_report(
    comparison_id: str,
    format: str = "json",
) -> dict[str, Any]:
    """Get a detailed comparison report.
    
    Args:
        comparison_id: The comparison identifier.
        format: Report format (json, markdown).
    
    Returns:
        dict: Comparison report.
    
    Raises:
        HTTPException: If comparison not found.
    """
    if comparison_id not in _comparisons:
        raise HTTPException(status_code=404, detail=f"Comparison '{comparison_id}' not found")
    
    comparison = _comparisons[comparison_id]
    
    if comparison["status"] != ComparisonStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Comparison not complete. Status: {comparison['status']}"
        )
    
    report = {
        "comparison_id": comparison_id,
        "created_at": comparison["created_at"],
        "completed_at": comparison["completed_at"],
        "backends": comparison["backends_requested"],
        "results": comparison["results"],
        "metrics": comparison["metrics"],
        "summary": comparison["summary"],
        "recommendations": _generate_recommendations(comparison),
    }
    
    if format == "markdown":
        report["markdown"] = _generate_markdown_report(comparison)
    
    return report


# =============================================================================
# Helper Functions
# =============================================================================

def _build_comparison_response(comparison: dict[str, Any]) -> ComparisonResponse:
    """Build a ComparisonResponse from storage dict."""
    metrics = None
    if comparison.get("metrics"):
        metrics = ComparisonMetrics(**comparison["metrics"])
    
    return ComparisonResponse(
        comparison_id=comparison["comparison_id"],
        status=comparison["status"],
        created_at=comparison["created_at"],
        completed_at=comparison.get("completed_at"),
        backends_requested=comparison["backends_requested"],
        results=[BackendResult(**r) for r in comparison.get("results", [])],
        metrics=metrics,
        summary=comparison.get("summary"),
    )


async def _run_comparison(comparison_id: str, request: Request) -> None:
    """Run a backend comparison."""
    import time
    import random
    
    if comparison_id not in _comparisons:
        return
    
    comparison = _comparisons[comparison_id]
    comparison["status"] = ComparisonStatus.RUNNING
    
    try:
        results = []
        execution_times = {}
        
        for backend in comparison["backends_requested"]:
            start_time = time.perf_counter()
            
            try:
                # Simulate execution delay
                import asyncio
                await asyncio.sleep(0.05)  # 50ms
                
                # Generate mock results
                num_qubits = 2
                num_states = 2 ** num_qubits
                counts = {}
                remaining = comparison["shots"]
                
                for i in range(num_states - 1):
                    count = random.randint(0, remaining)
                    if count > 0:
                        state = format(i, f'0{num_qubits}b')
                        counts[state] = count
                        remaining -= count
                
                if remaining > 0:
                    state = format(num_states - 1, f'0{num_qubits}b')
                    counts[state] = remaining
                
                exec_time = (time.perf_counter() - start_time) * 1000
                execution_times[backend] = exec_time
                
                results.append({
                    "backend": backend,
                    "success": True,
                    "execution_time_ms": exec_time,
                    "counts": counts,
                    "metadata": {"shots": comparison["shots"]},
                })
                
            except Exception as e:
                exec_time = (time.perf_counter() - start_time) * 1000
                results.append({
                    "backend": backend,
                    "success": False,
                    "execution_time_ms": exec_time,
                    "error": str(e),
                })
        
        comparison["results"] = results
        
        # Calculate metrics
        if execution_times:
            fastest = min(execution_times, key=execution_times.get)
            fastest_time = execution_times[fastest]
            
            speedup_ratios = {
                b: t / fastest_time for b, t in execution_times.items()
            }
            
            comparison["metrics"] = {
                "fastest_backend": fastest,
                "speedup_ratios": speedup_ratios,
            }
        
        # Generate summary
        successful = [r for r in results if r["success"]]
        comparison["summary"] = (
            f"Compared {len(comparison['backends_requested'])} backends. "
            f"{len(successful)} succeeded. "
            f"Fastest: {comparison['metrics']['fastest_backend'] if comparison.get('metrics') else 'N/A'}"
        )
        
        comparison["status"] = ComparisonStatus.COMPLETED
        comparison["completed_at"] = datetime.now(timezone.utc).isoformat()
        
    except Exception as e:
        comparison["status"] = ComparisonStatus.FAILED
        comparison["summary"] = f"Comparison failed: {str(e)}"
        comparison["completed_at"] = datetime.now(timezone.utc).isoformat()


def _generate_circuit(circuit_type: str, num_qubits: int) -> str:
    """Generate a circuit based on type."""
    if circuit_type == "bell":
        return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
    elif circuit_type == "ghz":
        gates = "h q[0];\n"
        for i in range(num_qubits - 1):
            gates += f"cx q[{i}], q[{i+1}];\n"
        return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
{gates}measure q -> c;
"""
    else:  # random
        import random
        gates = ""
        gate_set = ["h", "x", "y", "z", "t", "s"]
        for _ in range(num_qubits * 3):
            gate = random.choice(gate_set)
            qubit = random.randint(0, num_qubits - 1)
            gates += f"{gate} q[{qubit}];\n"
        return f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
{gates}measure q -> c;
"""


def _generate_recommendations(comparison: dict[str, Any]) -> list[str]:
    """Generate recommendations based on comparison results."""
    recommendations = []
    
    if comparison.get("metrics"):
        fastest = comparison["metrics"].get("fastest_backend")
        if fastest:
            recommendations.append(f"For fastest execution, use {fastest}")
    
    successful = [r for r in comparison.get("results", []) if r.get("success")]
    if len(successful) < len(comparison.get("backends_requested", [])):
        recommendations.append("Some backends failed - check installation and configuration")
    
    return recommendations


def _generate_markdown_report(comparison: dict[str, Any]) -> str:
    """Generate a markdown report."""
    lines = [
        f"# Backend Comparison Report",
        f"",
        f"**Comparison ID:** {comparison['comparison_id']}",
        f"**Created:** {comparison['created_at']}",
        f"**Backends:** {', '.join(comparison['backends_requested'])}",
        f"",
        f"## Results",
        f"",
        f"| Backend | Success | Time (ms) |",
        f"|---------|---------|-----------|",
    ]
    
    for result in comparison.get("results", []):
        success = "✓" if result.get("success") else "✗"
        time_ms = f"{result.get('execution_time_ms', 0):.2f}" if result.get("execution_time_ms") else "N/A"
        lines.append(f"| {result['backend']} | {success} | {time_ms} |")
    
    if comparison.get("summary"):
        lines.extend([
            "",
            "## Summary",
            "",
            comparison["summary"],
        ])
    
    return "\n".join(lines)
