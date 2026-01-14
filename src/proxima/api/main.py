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
        """List supported API versions."""
        return {
            "current_version": "v1",
            "supported_versions": ["v1"],
            "deprecated_versions": [],
            "version_header": "X-API-Version",
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
