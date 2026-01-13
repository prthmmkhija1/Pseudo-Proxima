"""
API middleware components.

Provides request logging, timing, and other cross-cutting concerns.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"client={request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        logger.info(
            f"[{request_id}] Response status={response.status_code}"
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware for timing request processing."""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Time request processing and add header."""
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication.
    
    This is a placeholder for future authentication implementation.
    Currently allows all requests through.
    """
    
    def __init__(self, app, api_keys: list[str] | None = None):
        super().__init__(app)
        self.api_keys = set(api_keys) if api_keys else None
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Validate API key if authentication is enabled."""
        # Skip auth for health endpoints
        if request.url.path in ["/health", "/ready", "/live"]:
            return await call_next(request)
        
        # Skip if no API keys configured
        if not self.api_keys:
            return await call_next(request)
        
        # Check API key header
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in self.api_keys:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "message": "Invalid or missing API key"},
            )
        
        return await call_next(request)
