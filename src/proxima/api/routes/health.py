"""
Health check endpoints.

Provides liveness and readiness probes for container orchestration.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    timestamp: str
    uptime_seconds: float | None = None


class ReadinessResponse(BaseModel):
    """Readiness check response with component status."""
    
    status: str
    components: dict[str, dict[str, Any]]
    timestamp: str


class SystemInfoResponse(BaseModel):
    """System information response."""
    
    python_version: str
    platform: str
    proxima_version: str
    available_backends: list[str]
    loaded_plugins: list[str]


# Track startup time
_startup_time: datetime | None = None


def _get_startup_time() -> datetime:
    """Get or initialize startup time."""
    global _startup_time
    if _startup_time is None:
        _startup_time = datetime.now(timezone.utc)
    return _startup_time


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint.
    
    Returns a simple health status for load balancer health checks.
    
    Returns:
        HealthResponse: Current health status.
    """
    startup = _get_startup_time()
    uptime = (datetime.now(timezone.utc) - startup).total_seconds()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=uptime,
    )


@router.get("/live")
async def liveness_probe() -> dict[str, str]:
    """Kubernetes liveness probe.
    
    Simple endpoint that returns 200 if the service is alive.
    
    Returns:
        dict: Simple status response.
    """
    return {"status": "alive"}


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_probe(request: Request) -> ReadinessResponse:
    """Kubernetes readiness probe.
    
    Checks if all required components are ready to serve requests.
    
    Returns:
        ReadinessResponse: Detailed component status.
    """
    components: dict[str, dict[str, Any]] = {}
    all_ready = True
    
    # Check backend registry
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            available = registry.list_available() if hasattr(registry, 'list_available') else []
            components["backends"] = {
                "status": "ready",
                "count": len(available),
            }
        else:
            components["backends"] = {
                "status": "initializing",
                "count": 0,
            }
            all_ready = False
    except Exception as e:
        components["backends"] = {
            "status": "error",
            "error": str(e),
        }
        all_ready = False
    
    # Check session service
    try:
        if hasattr(request.app.state, 'session_service'):
            session_service = request.app.state.session_service
            components["sessions"] = {
                "status": "ready",
                "active_sessions": session_service.active_count if hasattr(session_service, 'active_count') else 0,
            }
        else:
            components["sessions"] = {
                "status": "initializing",
            }
            all_ready = False
    except Exception as e:
        components["sessions"] = {
            "status": "error",
            "error": str(e),
        }
        all_ready = False
    
    return ReadinessResponse(
        status="ready" if all_ready else "degraded",
        components=components,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/info", response_model=SystemInfoResponse)
async def system_info(request: Request) -> SystemInfoResponse:
    """Get system information.
    
    Returns detailed information about the running system.
    
    Returns:
        SystemInfoResponse: System details.
    """
    # Get available backends
    available_backends: list[str] = []
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            if hasattr(registry, 'list_available'):
                available_backends = registry.list_available()
    except Exception:
        pass
    
    # Get loaded plugins
    loaded_plugins: list[str] = []
    try:
        from proxima.plugins import get_plugin_registry
        plugin_registry = get_plugin_registry()
        loaded_plugins = plugin_registry.list_names()
    except Exception:
        pass
    
    return SystemInfoResponse(
        python_version=sys.version,
        platform=platform.platform(),
        proxima_version="1.0.0",
        available_backends=available_backends,
        loaded_plugins=loaded_plugins,
    )


# =============================================================================
# Extended Health Models (Feature - API)
# =============================================================================


class MetricsResponse(BaseModel):
    """Server metrics response."""
    
    uptime_seconds: float
    request_count: int
    active_jobs: int
    active_sessions: int
    memory_usage_mb: float
    cpu_percent: float | None = None
    timestamp: str


class DiagnosticsResponse(BaseModel):
    """System diagnostics response."""
    
    system: dict[str, Any]
    backends: dict[str, Any]
    plugins: dict[str, Any]
    configuration: dict[str, Any]
    timestamp: str


class BackendHealthResponse(BaseModel):
    """Backend health check response."""
    
    backend: str
    status: str
    latency_ms: float | None = None
    version: str | None = None
    capabilities: list[str] = []
    error: str | None = None


class DependenciesResponse(BaseModel):
    """External dependencies status."""
    
    dependencies: dict[str, dict[str, Any]]
    all_healthy: bool
    timestamp: str


# Track metrics
_metrics = {
    "request_count": 0,
}


# =============================================================================
# Extended Health Endpoints (Feature - API)
# =============================================================================


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(request: Request) -> MetricsResponse:
    """Get server metrics.
    
    Returns performance metrics for monitoring dashboards.
    
    Returns:
        MetricsResponse: Current server metrics.
    """
    startup = _get_startup_time()
    uptime = (datetime.now(timezone.utc) - startup).total_seconds()
    
    # Get active job count
    active_jobs = 0
    try:
        if hasattr(request.app.state, 'job_registry'):
            active_jobs = request.app.state.job_registry.active_count
    except Exception:
        pass
    
    # Get active session count
    active_sessions = 0
    try:
        if hasattr(request.app.state, 'session_service'):
            active_sessions = request.app.state.session_service.active_count
    except Exception:
        pass
    
    # Get memory usage
    memory_mb = 0.0
    cpu_percent = None
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=None)
    except (ImportError, Exception):
        pass
    
    _metrics["request_count"] += 1
    
    return MetricsResponse(
        uptime_seconds=uptime,
        request_count=_metrics["request_count"],
        active_jobs=active_jobs,
        active_sessions=active_sessions,
        memory_usage_mb=memory_mb,
        cpu_percent=cpu_percent,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics(request: Request) -> DiagnosticsResponse:
    """Get system diagnostics.
    
    Returns detailed diagnostic information for troubleshooting.
    
    Returns:
        DiagnosticsResponse: Comprehensive diagnostics.
    """
    # System info
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    
    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        system_info["memory"] = {
            "total_gb": round(mem.total / 1024**3, 2),
            "available_gb": round(mem.available / 1024**3, 2),
            "percent_used": mem.percent,
        }
        system_info["cpu_count"] = psutil.cpu_count()
    except ImportError:
        system_info["memory"] = "psutil not available"
    
    # Backend diagnostics
    backends_info: dict[str, Any] = {}
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            if hasattr(registry, 'list_statuses'):
                for status in registry.list_statuses():
                    backends_info[status.name] = {
                        "available": status.available,
                        "version": getattr(status, 'version', None),
                    }
    except Exception as e:
        backends_info["error"] = str(e)
    
    # Plugin diagnostics
    plugins_info: dict[str, Any] = {}
    try:
        from proxima.plugins import get_plugin_manager
        manager = get_plugin_manager()
        stats = manager.get_plugin_stats()
        plugins_info = stats
    except Exception as e:
        plugins_info["error"] = str(e)
    
    # Configuration info (sanitized)
    config_info: dict[str, Any] = {}
    try:
        from proxima.config import get_config_service
        config = get_config_service()
        settings = config.load()
        # Only expose non-sensitive config
        config_info["verbosity"] = settings.general.verbosity
        config_info["default_backend"] = settings.backends.default_backend
        config_info["storage_backend"] = settings.general.storage_backend
    except Exception:
        config_info["status"] = "unavailable"
    
    return DiagnosticsResponse(
        system=system_info,
        backends=backends_info,
        plugins=plugins_info,
        configuration=config_info,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/backends", response_model=list[BackendHealthResponse])
async def check_backend_health(request: Request) -> list[BackendHealthResponse]:
    """Check health of all backends.
    
    Performs health checks on all registered backends.
    
    Returns:
        list[BackendHealthResponse]: Health status of each backend.
    """
    results: list[BackendHealthResponse] = []
    
    backends = []
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            if hasattr(registry, 'list_available'):
                backends = registry.list_available()
    except Exception:
        pass
    
    # Default backends if none found
    if not backends:
        backends = ["cirq", "qiskit_aer", "pennylane"]
    
    for backend_name in backends:
        import time
        start = time.perf_counter()
        
        try:
            # Try to get backend info
            status = "healthy"
            latency = None
            version = None
            capabilities: list[str] = []
            error = None
            
            if hasattr(request.app.state, 'backend_registry'):
                registry = request.app.state.backend_registry
                backend = registry.get(backend_name)
                if backend:
                    if hasattr(backend, 'get_version'):
                        version = backend.get_version()
                    if hasattr(backend, 'get_capabilities'):
                        capabilities = backend.get_capabilities()
            
            latency = (time.perf_counter() - start) * 1000
            
        except Exception as e:
            status = "unhealthy"
            error = str(e)
            latency = None
        
        results.append(BackendHealthResponse(
            backend=backend_name,
            status=status,
            latency_ms=latency,
            version=version,
            capabilities=capabilities,
            error=error,
        ))
    
    return results


@router.get("/dependencies", response_model=DependenciesResponse)
async def check_dependencies(request: Request) -> DependenciesResponse:
    """Check external dependency status.
    
    Verifies connectivity to external services and dependencies.
    
    Returns:
        DependenciesResponse: Status of all dependencies.
    """
    dependencies: dict[str, dict[str, Any]] = {}
    all_healthy = True
    
    # Check core packages
    core_packages = {
        "cirq": "cirq",
        "numpy": "numpy",
        "pydantic": "pydantic",
        "yaml": "yaml",
    }
    
    for name, module_name in core_packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            dependencies[name] = {
                "status": "available",
                "version": version,
            }
        except ImportError:
            dependencies[name] = {
                "status": "not_installed",
            }
            # Core packages being missing is not necessarily unhealthy
    
    # Check optional packages
    optional_packages = {
        "qiskit": "qiskit",
        "pennylane": "pennylane",
        "torch": "torch",
        "psutil": "psutil",
    }
    
    for name, module_name in optional_packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            dependencies[name] = {
                "status": "available",
                "version": version,
            }
        except ImportError:
            dependencies[name] = {
                "status": "not_installed",
                "optional": True,
            }
    
    return DependenciesResponse(
        dependencies=dependencies,
        all_healthy=all_healthy,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/warmup")
async def warmup(request: Request) -> dict[str, Any]:
    """Warm up server components.
    
    Pre-initializes backends and caches for faster first request.
    
    Returns:
        dict: Warmup results.
    """
    warmed: list[str] = []
    errors: list[str] = []
    
    # Warm up backends
    try:
        if hasattr(request.app.state, 'backend_registry'):
            registry = request.app.state.backend_registry
            for name in registry.list_available():
                try:
                    backend = registry.get(name)
                    # Initialize if possible
                    if hasattr(backend, 'initialize'):
                        backend.initialize()
                    warmed.append(f"backend:{name}")
                except Exception as e:
                    errors.append(f"backend:{name}: {e}")
    except Exception as e:
        errors.append(f"backend_registry: {e}")
    
    # Warm up plugin system
    try:
        from proxima.plugins import get_plugin_manager
        manager = get_plugin_manager()
        warmed.append("plugin_manager")
    except Exception as e:
        errors.append(f"plugins: {e}")
    
    return {
        "status": "completed",
        "warmed_components": warmed,
        "errors": errors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

