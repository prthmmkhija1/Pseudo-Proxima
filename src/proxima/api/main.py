"""
Proxima Web API - Main Application.

FastAPI-based REST API for quantum simulation orchestration.
Provides endpoints for:
- Circuit execution
- Backend management
- Session management
- Results and comparisons
- Real-time WebSocket updates
- API versioning and authentication
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from proxima.api.routes import backends, circuits, compare, health, sessions
from proxima.api.middleware import (
    RequestLoggingMiddleware,
    TimingMiddleware,
    EnhancedAuthMiddleware,
    AuthConfig,
    RateLimitMiddleware,
    RateLimitConfig,
    APIVersioningMiddleware,
    VersionConfig,
    websocket_manager,
)

logger = logging.getLogger(__name__)

# Global app instance
_app: FastAPI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Proxima API server...")
    
    # Initialize backend registry
    from proxima.backends import get_backend_registry
    registry = get_backend_registry()
    app.state.backend_registry = registry
    
    # Initialize session store
    from proxima.api.services.session_service import SessionService
    app.state.session_service = SessionService()
    
    # Initialize WebSocket manager
    app.state.websocket_manager = websocket_manager
    
    logger.info("Proxima API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Proxima API server...")
    
    # Cleanup sessions
    if hasattr(app.state, 'session_service'):
        await app.state.session_service.cleanup()
    
    logger.info("Proxima API server shutdown complete")


def create_app(
    title: str = "Proxima API",
    description: str = "REST API for Proxima quantum simulation orchestration",
    version: str = "1.0.0",
    debug: bool = False,
    cors_origins: list[str] | None = None,
    auth_config: AuthConfig | None = None,
    rate_limit_config: RateLimitConfig | None = None,
    version_config: VersionConfig | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        title: API title for OpenAPI docs.
        description: API description.
        version: API version string.
        debug: Enable debug mode.
        cors_origins: Allowed CORS origins. Defaults to ["*"] in debug mode.
        auth_config: Authentication configuration.
        rate_limit_config: Rate limiting configuration.
        version_config: API versioning configuration.
    
    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Configure CORS
    if cors_origins is None:
        cors_origins = ["*"] if debug else []
    
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add custom middleware (order matters - last added runs first)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add rate limiting middleware
    if rate_limit_config is None:
        rate_limit_config = RateLimitConfig(enabled=not debug)
    app.add_middleware(RateLimitMiddleware, config=rate_limit_config)
    
    # Add API versioning middleware
    if version_config is None:
        version_config = VersionConfig()
    app.add_middleware(APIVersioningMiddleware, config=version_config)
    
    # Add enhanced authentication middleware
    if auth_config is None:
        auth_config = AuthConfig(enabled=not debug)
    app.add_middleware(EnhancedAuthMiddleware, config=auth_config)
    
    # Register exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(exc) if debug else "An internal error occurred",
                "path": str(request.url.path),
            },
        )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(backends.router, prefix="/api/v1/backends", tags=["Backends"])
    app.include_router(circuits.router, prefix="/api/v1/circuits", tags=["Circuits"])
    app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])
    app.include_router(compare.router, prefix="/api/v1/compare", tags=["Comparison"])
    
    # ==========================================================================
    # WebSocket Endpoints for Real-Time Updates
    # ==========================================================================
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint for real-time updates.
        
        Clients can subscribe to topics:
        - execution:{execution_id} - Progress updates for specific execution
        - backends - Backend status updates
        - system - System-wide notifications
        
        Message format:
        {
            "action": "subscribe" | "unsubscribe" | "ping",
            "topic": "execution:abc123"
        }
        """
        client = await websocket_manager.connect(websocket)
        
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")
                
                if action == "subscribe":
                    topic = data.get("topic", "")
                    await websocket_manager.subscribe(client.client_id, topic)
                    await websocket_manager.send_to_client(client.client_id, {
                        "type": "subscribed",
                        "topic": topic,
                    })
                    
                elif action == "unsubscribe":
                    topic = data.get("topic", "")
                    await websocket_manager.unsubscribe(client.client_id, topic)
                    await websocket_manager.send_to_client(client.client_id, {
                        "type": "unsubscribed",
                        "topic": topic,
                    })
                    
                elif action == "ping":
                    await websocket_manager.send_to_client(client.client_id, {
                        "type": "pong",
                        "timestamp": __import__("time").time(),
                    })
                    
                else:
                    await websocket_manager.send_to_client(client.client_id, {
                        "type": "error",
                        "message": f"Unknown action: {action}",
                    })
                    
        except WebSocketDisconnect:
            await websocket_manager.disconnect(client.client_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket_manager.disconnect(client.client_id)
    
    @app.get("/ws/status", tags=["WebSocket"])
    async def websocket_status() -> dict[str, Any]:
        """Get WebSocket connection status."""
        return {
            "total_connections": websocket_manager.get_connection_count(),
            "connections": websocket_manager.list_connections(),
        }
    
    # ==========================================================================
    # API Versioning Info Endpoints
    # ==========================================================================
    
    @app.get("/api/versions", tags=["Versioning"])
    async def list_api_versions() -> dict[str, Any]:
        """List supported API versions.
        
        Returns comprehensive version information including:
        - Current default version
        - All supported versions with their status
        - Deprecated versions with sunset dates
        - Migration guides
        """
        return {
            "current_version": version_config.default_version,
            "supported_versions": version_config.supported_versions,
            "deprecated_versions": version_config.deprecated_versions,
            "version_header": version_config.version_header,
            "versions": {
                "v1": {
                    "status": "current",
                    "released": "2024-01-01",
                    "sunset_date": None,
                    "documentation": "/docs",
                    "changelog": "/api/versions/v1/changelog",
                },
            },
            "negotiation": {
                "url_prefix": "/api/{version}/",
                "header": version_config.version_header,
                "accept_header": "application/vnd.proxima.{version}+json",
            },
        }
    
    @app.get("/api/versions/{version}", tags=["Versioning"])
    async def get_version_info(version: str) -> dict[str, Any]:
        """Get detailed information about a specific API version.
        
        Args:
            version: API version (e.g., "v1")
            
        Returns:
            Version details including endpoints, changes, and migration info.
        """
        if version not in version_config.supported_versions:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "version_not_found",
                    "message": f"API version '{version}' not found",
                    "supported_versions": version_config.supported_versions,
                }
            )
        
        version_details = {
            "v1": {
                "version": "v1",
                "status": "current",
                "released": "2024-01-01",
                "sunset_date": None,
                "description": "Current stable API version",
                "endpoints": {
                    "backends": {
                        "list": "GET /api/v1/backends",
                        "get": "GET /api/v1/backends/{backend_id}",
                        "status": "GET /api/v1/backends/{backend_id}/status",
                    },
                    "circuits": {
                        "execute": "POST /api/v1/circuits/execute",
                        "validate": "POST /api/v1/circuits/validate",
                        "parse": "POST /api/v1/circuits/parse",
                    },
                    "sessions": {
                        "list": "GET /api/v1/sessions",
                        "create": "POST /api/v1/sessions",
                        "get": "GET /api/v1/sessions/{session_id}",
                        "delete": "DELETE /api/v1/sessions/{session_id}",
                    },
                    "compare": {
                        "run": "POST /api/v1/compare",
                        "get": "GET /api/v1/compare/{comparison_id}",
                    },
                },
                "features": [
                    "Multi-backend execution",
                    "Session management",
                    "Backend comparison",
                    "WebSocket real-time updates",
                    "Rate limiting",
                    "API key authentication",
                ],
                "breaking_changes": [],
                "deprecations": [],
            },
        }
        
        return version_details.get(version, {"version": version, "status": "unknown"})
    
    @app.get("/api/versions/{version}/changelog", tags=["Versioning"])
    async def get_version_changelog(version: str) -> dict[str, Any]:
        """Get changelog for a specific API version.
        
        Args:
            version: API version (e.g., "v1")
            
        Returns:
            Changelog entries for the version.
        """
        changelogs = {
            "v1": {
                "version": "v1",
                "entries": [
                    {
                        "version": "1.0.0",
                        "date": "2024-01-01",
                        "type": "release",
                        "changes": [
                            "Initial API release",
                            "Backend management endpoints",
                            "Circuit execution endpoints",
                            "Session management",
                            "WebSocket support for real-time updates",
                        ],
                    },
                    {
                        "version": "1.1.0",
                        "date": "2024-02-01",
                        "type": "minor",
                        "changes": [
                            "Added comparison endpoints",
                            "Rate limiting support",
                            "Enhanced authentication",
                            "API versioning middleware",
                        ],
                    },
                ],
            },
        }
        
        return changelogs.get(version, {"version": version, "entries": []})
    
    @app.get("/api/versions/{from_version}/migrate/{to_version}", tags=["Versioning"])
    async def get_migration_guide(from_version: str, to_version: str) -> dict[str, Any]:
        """Get migration guide between API versions.
        
        Args:
            from_version: Source API version
            to_version: Target API version
            
        Returns:
            Migration guide with breaking changes and update steps.
        """
        if from_version not in version_config.supported_versions + version_config.deprecated_versions:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "version_not_found",
                    "message": f"Source version '{from_version}' not found",
                }
            )
        
        if to_version not in version_config.supported_versions:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "version_not_found", 
                    "message": f"Target version '{to_version}' not found",
                }
            )
        
        # Migration guides between versions
        migrations = {
            # Future migrations would be defined here
            # ("v1", "v2"): { ... }
        }
        
        key = (from_version, to_version)
        if key in migrations:
            return migrations[key]
        
        return {
            "from_version": from_version,
            "to_version": to_version,
            "status": "no_migration_needed" if from_version == to_version else "migration_guide_unavailable",
            "message": (
                "No migration needed" if from_version == to_version
                else f"Migration guide from {from_version} to {to_version} is not yet available"
            ),
            "breaking_changes": [],
            "deprecations": [],
            "steps": [],
        }
    
    @app.get("/api/deprecations", tags=["Versioning"])
    async def list_deprecations() -> dict[str, Any]:
        """List all API deprecations.
        
        Returns:
            All deprecated endpoints, features, and versions with sunset dates.
        """
        return {
            "deprecated_versions": [
                {
                    "version": v,
                    "deprecated_since": "2024-01-01",
                    "sunset_date": "2025-01-01",
                    "migration_target": version_config.default_version,
                }
                for v in version_config.deprecated_versions
            ],
            "deprecated_endpoints": [],
            "deprecated_features": [],
            "upcoming_deprecations": [],
        }
    
    return app


def get_app() -> FastAPI:
    """Get or create the global FastAPI application instance.
    
    Returns:
        The FastAPI application instance.
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app


# Create default app for uvicorn
app = get_app()
