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
