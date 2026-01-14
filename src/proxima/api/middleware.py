"""
API middleware components.

Provides request logging, timing, and other cross-cutting concerns.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request, Response, WebSocket
from fastapi.responses import JSONResponse
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


# =============================================================================
# Enhanced Authentication & Authorization (Feature - Web API)
# =============================================================================


@dataclass
class AuthUser:
    """Authenticated user information."""
    
    user_id: str
    username: str
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    api_key: str | None = None
    token_expiry: float | None = None


@dataclass  
class AuthConfig:
    """Authentication configuration."""
    
    enabled: bool = False
    api_keys: dict[str, AuthUser] = field(default_factory=dict)
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    allow_anonymous: bool = True
    public_paths: list[str] = field(default_factory=lambda: [
        "/health", "/ready", "/live", "/docs", "/redoc", "/openapi.json"
    ])


class EnhancedAuthMiddleware(BaseHTTPMiddleware):
    """Enhanced authentication middleware with JWT and API key support.
    
    Features:
    - API key authentication (X-API-Key header)
    - JWT Bearer token authentication (Authorization header)
    - Role-based access control
    - Permission-based access control
    - Anonymous access configuration
    """
    
    def __init__(
        self,
        app,
        config: AuthConfig | None = None,
    ) -> None:
        """Initialize enhanced auth middleware.
        
        Args:
            app: FastAPI application
            config: Authentication configuration
        """
        super().__init__(app)
        self.config = config or AuthConfig()
        self._user_cache: dict[str, AuthUser] = {}
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Authenticate request and attach user to state."""
        # Skip auth for public paths
        if request.url.path in self.config.public_paths:
            return await call_next(request)
        
        # Skip if auth not enabled
        if not self.config.enabled:
            return await call_next(request)
        
        user: AuthUser | None = None
        
        # Try API key authentication
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user = await self._authenticate_api_key(api_key)
        
        # Try JWT authentication
        if user is None:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user = await self._authenticate_jwt(token)
        
        # Check if anonymous access is allowed
        if user is None and not self.config.allow_anonymous:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )
        
        # Attach user to request state
        request.state.user = user
        request.state.authenticated = user is not None
        
        return await call_next(request)
    
    async def _authenticate_api_key(self, api_key: str) -> AuthUser | None:
        """Authenticate using API key."""
        # Check cache first
        cache_key = f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        if cache_key in self._user_cache:
            return self._user_cache[cache_key]
        
        # Look up in configured keys
        if api_key in self.config.api_keys:
            user = self.config.api_keys[api_key]
            self._user_cache[cache_key] = user
            return user
        
        return None
    
    async def _authenticate_jwt(self, token: str) -> AuthUser | None:
        """Authenticate using JWT token."""
        try:
            import jwt
            
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
            )
            
            # Check expiry
            exp = payload.get("exp", 0)
            if exp < time.time():
                return None
            
            # Build user from payload
            user = AuthUser(
                user_id=payload.get("sub", ""),
                username=payload.get("username", ""),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                token_expiry=exp,
            )
            
            return user
            
        except Exception as e:
            logger.debug(f"JWT authentication failed: {e}")
            return None
    
    def create_jwt_token(self, user: AuthUser) -> str:
        """Create a JWT token for a user.
        
        Args:
            user: The user to create token for
            
        Returns:
            JWT token string
        """
        import jwt
        
        expiry = time.time() + (self.config.jwt_expiry_hours * 3600)
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "permissions": user.permissions,
            "exp": expiry,
            "iat": time.time(),
        }
        
        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm,
        )


def require_auth(roles: list[str] | None = None, permissions: list[str] | None = None):
    """Dependency for requiring authentication with optional role/permission checks.
    
    Args:
        roles: Required roles (any match)
        permissions: Required permissions (all required)
        
    Returns:
        FastAPI dependency function
    """
    from fastapi import Depends, HTTPException
    
    async def dependency(request: Request) -> AuthUser:
        user: AuthUser | None = getattr(request.state, "user", None)
        
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Check roles
        if roles:
            if not any(role in user.roles for role in roles):
                raise HTTPException(status_code=403, detail="Insufficient role")
        
        # Check permissions  
        if permissions:
            if not all(perm in user.permissions for perm in permissions):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        return user
    
    return dependency


# =============================================================================
# Rate Limiting Middleware (Feature - Web API)
# =============================================================================


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    
    enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    by_ip: bool = True
    by_api_key: bool = True
    exempt_paths: list[str] = field(default_factory=lambda: [
        "/health", "/ready", "/live"
    ])


@dataclass
class RateLimitState:
    """State for rate limit tracking."""
    
    minute_count: int = 0
    hour_count: int = 0
    minute_window_start: float = 0.0
    hour_window_start: float = 0.0
    burst_tokens: float = 0.0
    last_request_time: float = 0.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with sliding window and token bucket.
    
    Features:
    - Per-minute and per-hour limits
    - Burst allowance with token bucket
    - Per-IP and per-API-key tracking
    - Configurable exempt paths
    - Returns rate limit headers
    """
    
    def __init__(
        self,
        app,
        config: RateLimitConfig | None = None,
    ) -> None:
        """Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            config: Rate limit configuration
        """
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self._states: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Check rate limits and process request."""
        # Skip for exempt paths
        if request.url.path in self.config.exempt_paths:
            return await call_next(request)
        
        # Skip if disabled
        if not self.config.enabled:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limits
        async with self._lock:
            allowed, state, retry_after = self._check_rate_limit(client_id)
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(self.config.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.config.requests_per_minute - state.minute_count)
        reset_time = int(state.minute_window_start + 60)
        
        response.headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        parts: list[str] = []
        
        # By API key
        if self.config.by_api_key:
            api_key = request.headers.get("X-API-Key")
            if api_key:
                parts.append(f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:8]}")
        
        # By IP
        if self.config.by_ip and request.client:
            parts.append(f"ip:{request.client.host}")
        
        return ":".join(parts) if parts else "anonymous"
    
    def _check_rate_limit(
        self, client_id: str
    ) -> tuple[bool, RateLimitState, float]:
        """Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, state, retry_after_seconds)
        """
        now = time.time()
        state = self._states[client_id]
        
        # Reset minute window if needed
        if now - state.minute_window_start >= 60:
            state.minute_count = 0
            state.minute_window_start = now
        
        # Reset hour window if needed
        if now - state.hour_window_start >= 3600:
            state.hour_count = 0
            state.hour_window_start = now
        
        # Replenish burst tokens (token bucket)
        time_since_last = now - state.last_request_time
        state.burst_tokens = min(
            self.config.burst_limit,
            state.burst_tokens + time_since_last * (self.config.burst_limit / 60),
        )
        
        # Check limits
        if state.minute_count >= self.config.requests_per_minute:
            retry_after = 60 - (now - state.minute_window_start)
            return False, state, max(1, retry_after)
        
        if state.hour_count >= self.config.requests_per_hour:
            retry_after = 3600 - (now - state.hour_window_start)
            return False, state, max(1, retry_after)
        
        # Consume tokens for burst protection
        if state.burst_tokens < 1:
            retry_after = (1 - state.burst_tokens) * (60 / self.config.burst_limit)
            return False, state, max(0.1, retry_after)
        
        # Allow request
        state.minute_count += 1
        state.hour_count += 1
        state.burst_tokens -= 1
        state.last_request_time = now
        
        return True, state, 0


# =============================================================================
# API Versioning Middleware (Feature - Web API)
# =============================================================================


@dataclass
class VersionConfig:
    """API versioning configuration."""
    
    default_version: str = "v1"
    supported_versions: list[str] = field(default_factory=lambda: ["v1"])
    deprecated_versions: list[str] = field(default_factory=list)
    version_header: str = "X-API-Version"
    accept_header_version: bool = True
    url_version_prefix: bool = True


class APIVersioningMiddleware(BaseHTTPMiddleware):
    """API versioning middleware.
    
    Features:
    - URL path versioning (/api/v1/...)
    - Header-based versioning (X-API-Version)
    - Accept header versioning (Accept: application/vnd.proxima.v1+json)
    - Version deprecation warnings
    - Default version fallback
    """
    
    def __init__(
        self,
        app,
        config: VersionConfig | None = None,
    ) -> None:
        """Initialize versioning middleware.
        
        Args:
            app: FastAPI application
            config: Versioning configuration
        """
        super().__init__(app)
        self.config = config or VersionConfig()
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Extract and validate API version."""
        version = self._extract_version(request)
        
        # Validate version
        if version not in self.config.supported_versions:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "unsupported_version",
                    "message": f"API version '{version}' is not supported",
                    "supported_versions": self.config.supported_versions,
                },
            )
        
        # Attach version to request state
        request.state.api_version = version
        
        # Process request
        response = await call_next(request)
        
        # Add version headers
        response.headers["X-API-Version"] = version
        
        # Add deprecation warning if applicable
        if version in self.config.deprecated_versions:
            response.headers["X-API-Deprecated"] = "true"
            response.headers["Warning"] = (
                f'299 - "API version {version} is deprecated. '
                f'Please migrate to {self.config.default_version}"'
            )
        
        return response
    
    def _extract_version(self, request: Request) -> str:
        """Extract API version from request.
        
        Priority:
        1. URL path prefix (/api/v1/...)
        2. X-API-Version header
        3. Accept header (application/vnd.proxima.v1+json)
        4. Default version
        """
        # 1. URL path
        if self.config.url_version_prefix:
            path = request.url.path
            for version in self.config.supported_versions:
                if f"/api/{version}/" in path or path.startswith(f"/{version}/"):
                    return version
        
        # 2. Custom header
        header_version = request.headers.get(self.config.version_header)
        if header_version and header_version in self.config.supported_versions:
            return header_version
        
        # 3. Accept header
        if self.config.accept_header_version:
            accept = request.headers.get("Accept", "")
            for version in self.config.supported_versions:
                if f"vnd.proxima.{version}" in accept:
                    return version
        
        # 4. Default
        return self.config.default_version


# =============================================================================
# WebSocket Connection Manager (Feature - Web API)
# =============================================================================


@dataclass
class WebSocketClient:
    """WebSocket client connection info."""
    
    client_id: str
    websocket: WebSocket
    connected_at: float
    subscriptions: set[str] = field(default_factory=set)
    user: AuthUser | None = None


class WebSocketConnectionManager:
    """Manages WebSocket connections for real-time updates.
    
    Features:
    - Connection lifecycle management
    - Topic-based subscriptions
    - Broadcast to all or filtered clients
    - Automatic reconnection handling
    - Connection state tracking
    """
    
    def __init__(self) -> None:
        """Initialize WebSocket connection manager."""
        self._connections: dict[str, WebSocketClient] = {}
        self._topic_subscribers: dict[str, set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str | None = None,
        user: AuthUser | None = None,
    ) -> WebSocketClient:
        """Accept and register a WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            client_id: Optional client ID (auto-generated if not provided)
            user: Optional authenticated user
            
        Returns:
            WebSocketClient instance
        """
        await websocket.accept()
        
        if client_id is None:
            client_id = str(uuid.uuid4())[:8]
        
        client = WebSocketClient(
            client_id=client_id,
            websocket=websocket,
            connected_at=time.time(),
            user=user,
        )
        
        async with self._lock:
            self._connections[client_id] = client
        
        logger.info(f"WebSocket connected: {client_id}")
        
        # Send welcome message
        await self.send_to_client(client_id, {
            "type": "connected",
            "client_id": client_id,
            "timestamp": time.time(),
        })
        
        return client
    
    async def disconnect(self, client_id: str) -> None:
        """Disconnect and cleanup a WebSocket connection.
        
        Args:
            client_id: Client to disconnect
        """
        async with self._lock:
            if client_id in self._connections:
                client = self._connections[client_id]
                
                # Remove from all topics
                for topic in client.subscriptions:
                    self._topic_subscribers[topic].discard(client_id)
                
                del self._connections[client_id]
                logger.info(f"WebSocket disconnected: {client_id}")
    
    async def subscribe(self, client_id: str, topic: str) -> bool:
        """Subscribe a client to a topic.
        
        Args:
            client_id: Client to subscribe
            topic: Topic to subscribe to
            
        Returns:
            True if subscribed
        """
        async with self._lock:
            if client_id not in self._connections:
                return False
            
            self._connections[client_id].subscriptions.add(topic)
            self._topic_subscribers[topic].add(client_id)
            
        logger.debug(f"Client {client_id} subscribed to {topic}")
        return True
    
    async def unsubscribe(self, client_id: str, topic: str) -> bool:
        """Unsubscribe a client from a topic.
        
        Args:
            client_id: Client to unsubscribe
            topic: Topic to unsubscribe from
            
        Returns:
            True if unsubscribed
        """
        async with self._lock:
            if client_id not in self._connections:
                return False
            
            self._connections[client_id].subscriptions.discard(topic)
            self._topic_subscribers[topic].discard(client_id)
            
        logger.debug(f"Client {client_id} unsubscribed from {topic}")
        return True
    
    async def send_to_client(
        self, client_id: str, message: dict[str, Any]
    ) -> bool:
        """Send a message to a specific client.
        
        Args:
            client_id: Target client
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        async with self._lock:
            client = self._connections.get(client_id)
        
        if client is None:
            return False
        
        try:
            await client.websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send to {client_id}: {e}")
            await self.disconnect(client_id)
            return False
    
    async def broadcast(
        self,
        message: dict[str, Any],
        topic: str | None = None,
        exclude: set[str] | None = None,
    ) -> int:
        """Broadcast a message to clients.
        
        Args:
            message: Message to broadcast
            topic: Only send to subscribers of this topic
            exclude: Client IDs to exclude
            
        Returns:
            Number of clients message was sent to
        """
        exclude = exclude or set()
        sent_count = 0
        
        async with self._lock:
            if topic:
                client_ids = self._topic_subscribers.get(topic, set()) - exclude
            else:
                client_ids = set(self._connections.keys()) - exclude
        
        for client_id in client_ids:
            if await self.send_to_client(client_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_execution_progress(
        self,
        execution_id: str,
        progress: float,
        stage: str,
        status: str,
    ) -> int:
        """Broadcast execution progress update.
        
        Args:
            execution_id: Execution ID
            progress: Progress percentage (0-100)
            stage: Current stage name
            status: Execution status
            
        Returns:
            Number of clients notified
        """
        message = {
            "type": "execution_progress",
            "execution_id": execution_id,
            "progress": progress,
            "stage": stage,
            "status": status,
            "timestamp": time.time(),
        }
        
        return await self.broadcast(
            message,
            topic=f"execution:{execution_id}",
        )
    
    def get_connection_count(self) -> int:
        """Get total number of connected clients."""
        return len(self._connections)
    
    def get_topic_subscriber_count(self, topic: str) -> int:
        """Get number of subscribers for a topic."""
        return len(self._topic_subscribers.get(topic, set()))
    
    def list_connections(self) -> list[dict[str, Any]]:
        """List all active connections."""
        return [
            {
                "client_id": c.client_id,
                "connected_at": c.connected_at,
                "subscriptions": list(c.subscriptions),
                "authenticated": c.user is not None,
            }
            for c in self._connections.values()
        ]


# Global WebSocket manager instance
websocket_manager = WebSocketConnectionManager()
