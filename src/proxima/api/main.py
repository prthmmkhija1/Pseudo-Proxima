"""
Proxima Web API - Main Application.

FastAPI-based REST API for quantum simulation orchestration.
Provides endpoints for:
- Circuit execution
- Backend management
- Session management
- Results and comparisons
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from proxima.api.routes import backends, circuits, compare, health, sessions
from proxima.api.middleware import RequestLoggingMiddleware, TimingMiddleware

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
) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        title: API title for OpenAPI docs.
        description: API description.
        version: API version string.
        debug: Enable debug mode.
        cors_origins: Allowed CORS origins. Defaults to ["*"] in debug mode.
    
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
    
    # Add custom middleware
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    
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
