"""
Backend management endpoints.

Provides REST API for discovering, configuring, and managing quantum backends.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class BackendInfo(BaseModel):
    """Information about a quantum backend."""
    
    name: str = Field(..., description="Backend identifier")
    version: str = Field(..., description="Backend version")
    available: bool = Field(..., description="Whether backend is currently available")
    description: str = Field("", description="Backend description")
    capabilities: dict[str, Any] = Field(default_factory=dict, description="Backend capabilities")


class BackendListResponse(BaseModel):
    """List of available backends."""
    
    backends: list[BackendInfo]
    total: int
    available_count: int


class BackendDetailResponse(BaseModel):
    """Detailed backend information."""
    
    name: str
    version: str
    available: bool
    description: str
    capabilities: dict[str, Any]
    configuration: dict[str, Any]
    health_status: str
    last_used: str | None = None


class BackendConfigRequest(BaseModel):
    """Request to configure a backend."""
    
    config: dict[str, Any] = Field(..., description="Backend configuration options")


class BackendTestRequest(BaseModel):
    """Request to test a backend."""
    
    num_qubits: int = Field(default=2, ge=1, le=30, description="Number of qubits for test circuit")
    shots: int = Field(default=100, ge=1, le=10000, description="Number of measurement shots")


class BackendTestResponse(BaseModel):
    """Response from backend test."""
    
    backend: str
    success: bool
    execution_time_ms: float
    result: dict[str, Any] | None = None
    error: str | None = None


class ResourceEstimateRequest(BaseModel):
    """Request for resource estimation."""
    
    num_qubits: int = Field(..., ge=1, le=50, description="Number of qubits")
    circuit_depth: int = Field(default=10, ge=1, description="Circuit depth")
    simulation_type: str = Field(default="state_vector", description="Simulation type")


class ResourceEstimateResponse(BaseModel):
    """Resource estimation response."""
    
    backend: str
    memory_bytes: int
    memory_human: str
    estimated_time_seconds: float
    warnings: list[str] = Field(default_factory=list)
    feasible: bool


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=BackendListResponse)
async def list_backends(request: Request) -> BackendListResponse:
    """List all registered quantum backends.
    
    Returns information about all backends, including their availability status.
    
    Returns:
        BackendListResponse: List of backends with summary info.
    """
    backends: list[BackendInfo] = []
    
    # Get backend registry
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            
            # Get all backend names
            all_backends = registry.list_all() if hasattr(registry, 'list_all') else []
            
            for name in all_backends:
                try:
                    backend = registry.get(name)
                    if backend:
                        backends.append(BackendInfo(
                            name=name,
                            version=backend.get_version() if hasattr(backend, 'get_version') else "unknown",
                            available=backend.is_available() if hasattr(backend, 'is_available') else False,
                            description=getattr(backend, 'description', ''),
                            capabilities=backend.get_capabilities().__dict__ if hasattr(backend, 'get_capabilities') else {},
                        ))
                except Exception:
                    backends.append(BackendInfo(
                        name=name,
                        version="unknown",
                        available=False,
                        description="Error loading backend info",
                    ))
    except Exception as e:
        # Return default backends if registry not available
        default_backends = ["cirq", "qiskit_aer", "lret", "quest", "cuquantum", "qsim"]
        for name in default_backends:
            backends.append(BackendInfo(
                name=name,
                version="unknown",
                available=False,
                description=f"Backend {name}",
            ))
    
    available_count = sum(1 for b in backends if b.available)
    
    return BackendListResponse(
        backends=backends,
        total=len(backends),
        available_count=available_count,
    )


@router.get("/{backend_name}", response_model=BackendDetailResponse)
async def get_backend(backend_name: str, request: Request) -> BackendDetailResponse:
    """Get detailed information about a specific backend.
    
    Args:
        backend_name: The backend identifier.
        request: FastAPI request object.
    
    Returns:
        BackendDetailResponse: Detailed backend information.
    
    Raises:
        HTTPException: If backend not found.
    """
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            backend = registry.get(backend_name)
            
            if backend is None:
                raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
            
            # Get capabilities
            capabilities = {}
            if hasattr(backend, 'get_capabilities'):
                caps = backend.get_capabilities()
                capabilities = caps.__dict__ if hasattr(caps, '__dict__') else {}
            
            # Get configuration
            config = {}
            if hasattr(backend, 'get_config'):
                cfg = backend.get_config()
                config = cfg.__dict__ if hasattr(cfg, '__dict__') else {}
            
            # Check health
            health_status = "unknown"
            if hasattr(backend, 'health_check'):
                try:
                    health = backend.health_check()
                    health_status = "healthy" if health else "unhealthy"
                except Exception:
                    health_status = "error"
            
            return BackendDetailResponse(
                name=backend_name,
                version=backend.get_version() if hasattr(backend, 'get_version') else "unknown",
                available=backend.is_available() if hasattr(backend, 'is_available') else False,
                description=getattr(backend, 'description', ''),
                capabilities=capabilities,
                configuration=config,
                health_status=health_status,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")


@router.post("/{backend_name}/configure", response_model=BackendDetailResponse)
async def configure_backend(
    backend_name: str,
    config_request: BackendConfigRequest,
    request: Request,
) -> BackendDetailResponse:
    """Configure a backend with new settings.
    
    Args:
        backend_name: The backend identifier.
        config_request: New configuration settings.
        request: FastAPI request object.
    
    Returns:
        BackendDetailResponse: Updated backend information.
    
    Raises:
        HTTPException: If backend not found or configuration fails.
    """
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            backend = registry.get(backend_name)
            
            if backend is None:
                raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
            
            # Apply configuration
            if hasattr(backend, 'configure'):
                backend.configure(config_request.config)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Backend '{backend_name}' does not support configuration"
                )
            
            # Return updated info
            return await get_backend(backend_name, request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")


@router.post("/{backend_name}/test", response_model=BackendTestResponse)
async def test_backend(
    backend_name: str,
    test_request: BackendTestRequest,
    request: Request,
) -> BackendTestResponse:
    """Test a backend with a simple circuit.
    
    Creates and executes a simple test circuit to verify backend functionality.
    
    Args:
        backend_name: The backend identifier.
        test_request: Test parameters.
        request: FastAPI request object.
    
    Returns:
        BackendTestResponse: Test results including execution time.
    
    Raises:
        HTTPException: If backend not found.
    """
    import time
    
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            backend = registry.get(backend_name)
            
            if backend is None:
                raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
            
            if not backend.is_available():
                return BackendTestResponse(
                    backend=backend_name,
                    success=False,
                    execution_time_ms=0,
                    error="Backend is not available",
                )
            
            # Create test circuit (Bell state)
            start_time = time.perf_counter()
            
            try:
                # Execute simple test
                result = None
                if hasattr(backend, 'execute'):
                    # Create a simple test circuit
                    from proxima.api.services.circuit_service import create_test_circuit
                    circuit = create_test_circuit(test_request.num_qubits)
                    result = backend.execute(circuit, options={'shots': test_request.shots})
                
                execution_time = (time.perf_counter() - start_time) * 1000
                
                return BackendTestResponse(
                    backend=backend_name,
                    success=True,
                    execution_time_ms=execution_time,
                    result=result.__dict__ if hasattr(result, '__dict__') else {"raw": str(result)},
                )
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                return BackendTestResponse(
                    backend=backend_name,
                    success=False,
                    execution_time_ms=execution_time,
                    error=str(e),
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")


@router.post("/{backend_name}/estimate", response_model=ResourceEstimateResponse)
async def estimate_resources(
    backend_name: str,
    estimate_request: ResourceEstimateRequest,
    request: Request,
) -> ResourceEstimateResponse:
    """Estimate resources required for a circuit.
    
    Args:
        backend_name: The backend identifier.
        estimate_request: Circuit parameters for estimation.
        request: FastAPI request object.
    
    Returns:
        ResourceEstimateResponse: Resource estimates.
    
    Raises:
        HTTPException: If backend not found.
    """
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            backend = registry.get(backend_name)
            
            if backend is None:
                raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
            
            # Calculate memory estimate
            num_qubits = estimate_request.num_qubits
            sim_type = estimate_request.simulation_type
            
            if sim_type == "density_matrix":
                # Density matrix: 2^(2n) * 16 bytes
                memory_bytes = (2 ** (2 * num_qubits)) * 16
            else:
                # State vector: 2^n * 16 bytes
                memory_bytes = (2 ** num_qubits) * 16
            
            # Convert to human readable
            if memory_bytes < 1024:
                memory_human = f"{memory_bytes} B"
            elif memory_bytes < 1024 ** 2:
                memory_human = f"{memory_bytes / 1024:.2f} KB"
            elif memory_bytes < 1024 ** 3:
                memory_human = f"{memory_bytes / (1024 ** 2):.2f} MB"
            else:
                memory_human = f"{memory_bytes / (1024 ** 3):.2f} GB"
            
            # Estimate time (rough heuristic)
            estimated_time = (estimate_request.circuit_depth * (2 ** num_qubits)) / 1e9
            
            # Check feasibility
            warnings = []
            feasible = True
            
            if memory_bytes > 16 * (1024 ** 3):  # 16 GB
                warnings.append("Memory requirements exceed typical system RAM")
                feasible = False
            elif memory_bytes > 8 * (1024 ** 3):  # 8 GB
                warnings.append("High memory usage - ensure sufficient RAM available")
            
            if num_qubits > 25:
                warnings.append("Large qubit count may result in very long execution times")
            
            return ResourceEstimateResponse(
                backend=backend_name,
                memory_bytes=memory_bytes,
                memory_human=memory_human,
                estimated_time_seconds=estimated_time,
                warnings=warnings,
                feasible=feasible,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
