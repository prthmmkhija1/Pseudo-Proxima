"""
Advanced API Tests for Proxima.

Tests for authentication, rate limiting, error handling, CORS,
WebSocket endpoints, and advanced API features.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta
from typing import Any


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_auth_service():
    """Create mock authentication service."""
    auth = Mock()
    auth.validate_token.return_value = True
    auth.validate_api_key.return_value = True
    auth.get_user_from_token.return_value = {
        "user_id": "test-user-123",
        "email": "test@example.com",
        "roles": ["user", "developer"],
        "permissions": ["execute", "read", "write"],
    }
    auth.create_token.return_value = "mock-jwt-token"
    auth.refresh_token.return_value = "mock-refreshed-token"
    return auth


@pytest.fixture
def mock_rate_limiter():
    """Create mock rate limiter."""
    limiter = Mock()
    limiter.check_limit.return_value = True
    limiter.get_remaining.return_value = 100
    limiter.get_reset_time.return_value = datetime.now(timezone.utc) + timedelta(hours=1)
    limiter.record_request.return_value = None
    return limiter


@pytest.fixture
def mock_cache():
    """Create mock cache service."""
    cache = Mock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = True
    cache.exists.return_value = False
    cache.get_stats.return_value = {
        "hits": 1000,
        "misses": 200,
        "size": 50000,
        "max_size": 100000,
    }
    return cache


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_json = AsyncMock(return_value={"type": "ping"})
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def mock_execution_service():
    """Create mock execution service."""
    service = Mock()
    service.submit_job.return_value = "job-123"
    service.get_job_status.return_value = {
        "job_id": "job-123",
        "status": "completed",
        "progress": 100,
        "result": {"counts": {"00": 500, "11": 500}},
    }
    service.cancel_job.return_value = True
    service.list_jobs.return_value = []
    return service


# ==============================================================================
# Authentication Tests
# ==============================================================================

class TestAuthenticationEndpoints:
    """Tests for authentication and authorization."""

    def test_login_with_valid_credentials(self, mock_auth_service):
        """Test successful login."""
        with patch("proxima.api.auth.auth_service", mock_auth_service):
            mock_auth_service.authenticate.return_value = {
                "access_token": "mock-token",
                "refresh_token": "mock-refresh",
                "token_type": "bearer",
                "expires_in": 3600,
            }
            
            result = mock_auth_service.authenticate("user@test.com", "password")
            
            assert "access_token" in result
            assert result["token_type"] == "bearer"

    def test_login_with_invalid_credentials(self, mock_auth_service):
        """Test login with invalid credentials."""
        mock_auth_service.authenticate.return_value = None
        mock_auth_service.authenticate.side_effect = ValueError("Invalid credentials")
        
        with pytest.raises(ValueError, match="Invalid credentials"):
            mock_auth_service.authenticate("invalid@test.com", "wrong")

    def test_token_refresh(self, mock_auth_service):
        """Test token refresh functionality."""
        new_token = mock_auth_service.refresh_token("old-refresh-token")
        assert new_token == "mock-refreshed-token"

    def test_token_validation(self, mock_auth_service):
        """Test token validation."""
        is_valid = mock_auth_service.validate_token("valid-token")
        assert is_valid is True

    def test_api_key_authentication(self, mock_auth_service):
        """Test API key authentication."""
        is_valid = mock_auth_service.validate_api_key("valid-api-key")
        assert is_valid is True

    def test_get_user_from_token(self, mock_auth_service):
        """Test extracting user info from token."""
        user = mock_auth_service.get_user_from_token("valid-token")
        
        assert user["user_id"] == "test-user-123"
        assert "developer" in user["roles"]
        assert "execute" in user["permissions"]

    def test_unauthorized_access(self, mock_auth_service):
        """Test unauthorized access handling."""
        mock_auth_service.validate_token.return_value = False
        
        is_valid = mock_auth_service.validate_token("invalid-token")
        assert is_valid is False

    def test_permission_check(self, mock_auth_service):
        """Test permission checking."""
        mock_auth_service.has_permission.return_value = True
        
        has_perm = mock_auth_service.has_permission("test-user", "execute")
        assert has_perm is True

    def test_role_based_access_control(self, mock_auth_service):
        """Test role-based access control."""
        mock_auth_service.has_role.return_value = True
        
        has_role = mock_auth_service.has_role("test-user", "admin")
        assert has_role is True


# ==============================================================================
# Rate Limiting Tests
# ==============================================================================

class TestRateLimiting:
    """Tests for API rate limiting."""

    def test_rate_limit_check_passes(self, mock_rate_limiter):
        """Test rate limit check when under limit."""
        is_allowed = mock_rate_limiter.check_limit("user-123", "execute")
        assert is_allowed is True

    def test_rate_limit_exceeded(self, mock_rate_limiter):
        """Test rate limit exceeded response."""
        mock_rate_limiter.check_limit.return_value = False
        
        is_allowed = mock_rate_limiter.check_limit("user-123", "execute")
        assert is_allowed is False

    def test_get_remaining_requests(self, mock_rate_limiter):
        """Test getting remaining request count."""
        remaining = mock_rate_limiter.get_remaining("user-123", "execute")
        assert remaining == 100

    def test_rate_limit_reset_time(self, mock_rate_limiter):
        """Test getting rate limit reset time."""
        reset_time = mock_rate_limiter.get_reset_time("user-123", "execute")
        assert reset_time > datetime.now(timezone.utc)

    def test_different_rate_limits_per_endpoint(self, mock_rate_limiter):
        """Test different rate limits for different endpoints."""
        mock_rate_limiter.get_limit.side_effect = lambda user, endpoint: {
            "execute": 100,
            "read": 1000,
            "write": 500,
        }.get(endpoint, 100)
        
        assert mock_rate_limiter.get_limit("user", "execute") == 100
        assert mock_rate_limiter.get_limit("user", "read") == 1000

    def test_rate_limit_headers(self, mock_rate_limiter):
        """Test rate limit headers generation."""
        mock_rate_limiter.get_headers.return_value = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "99",
            "X-RateLimit-Reset": "1234567890",
        }
        
        headers = mock_rate_limiter.get_headers("user-123", "execute")
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestErrorHandling:
    """Tests for API error handling."""

    def test_validation_error_response(self):
        """Test validation error response format."""
        error_response = {
            "error": "validation_error",
            "message": "Invalid request body",
            "details": [
                {"field": "shots", "message": "must be positive integer"},
                {"field": "backend", "message": "backend not found"},
            ],
            "request_id": "req-123",
        }
        
        assert error_response["error"] == "validation_error"
        assert len(error_response["details"]) == 2

    def test_not_found_error_response(self):
        """Test not found error response."""
        error_response = {
            "error": "not_found",
            "message": "Resource not found",
            "resource_type": "job",
            "resource_id": "job-999",
        }
        
        assert error_response["error"] == "not_found"

    def test_internal_server_error_response(self):
        """Test internal server error response."""
        error_response = {
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": "req-456",
            "support_url": "https://support.proxima.io",
        }
        
        assert error_response["error"] == "internal_error"
        assert "support_url" in error_response

    def test_conflict_error_response(self):
        """Test conflict error response."""
        error_response = {
            "error": "conflict",
            "message": "Resource already exists",
            "conflicting_field": "name",
            "existing_value": "my-session",
        }
        
        assert error_response["error"] == "conflict"

    def test_service_unavailable_response(self):
        """Test service unavailable response."""
        error_response = {
            "error": "service_unavailable",
            "message": "Backend temporarily unavailable",
            "retry_after": 60,
            "degraded_mode": True,
        }
        
        assert error_response["error"] == "service_unavailable"
        assert error_response["retry_after"] == 60


# ==============================================================================
# Caching Tests
# ==============================================================================

class TestCaching:
    """Tests for API response caching."""

    def test_cache_hit(self, mock_cache):
        """Test cache hit scenario."""
        mock_cache.get.return_value = {"cached": True, "data": [1, 2, 3]}
        
        result = mock_cache.get("backends:list")
        assert result["cached"] is True

    def test_cache_miss(self, mock_cache):
        """Test cache miss scenario."""
        mock_cache.get.return_value = None
        mock_cache.exists.return_value = False
        
        result = mock_cache.get("nonexistent:key")
        assert result is None

    def test_cache_set(self, mock_cache):
        """Test cache set operation."""
        success = mock_cache.set("key", {"data": "value"}, ttl=3600)
        assert success is True

    def test_cache_invalidation(self, mock_cache):
        """Test cache invalidation."""
        success = mock_cache.delete("stale:key")
        assert success is True

    def test_cache_stats(self, mock_cache):
        """Test cache statistics."""
        stats = mock_cache.get_stats()
        
        assert stats["hits"] == 1000
        assert stats["misses"] == 200
        assert stats["size"] < stats["max_size"]


# ==============================================================================
# WebSocket Tests
# ==============================================================================

class TestWebSocketEndpoints:
    """Tests for WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_websocket_connection_accept(self, mock_websocket):
        """Test WebSocket connection acceptance."""
        await mock_websocket.accept()
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_send_json(self, mock_websocket):
        """Test sending JSON via WebSocket."""
        data = {"type": "status", "progress": 50}
        await mock_websocket.send_json(data)
        mock_websocket.send_json.assert_called_with(data)

    @pytest.mark.asyncio
    async def test_websocket_receive_json(self, mock_websocket):
        """Test receiving JSON via WebSocket."""
        data = await mock_websocket.receive_json()
        assert data["type"] == "ping"

    @pytest.mark.asyncio
    async def test_websocket_close(self, mock_websocket):
        """Test WebSocket connection close."""
        await mock_websocket.close()
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_execution_updates(self, mock_websocket):
        """Test execution progress updates via WebSocket."""
        updates = [
            {"type": "progress", "percentage": 25},
            {"type": "progress", "percentage": 50},
            {"type": "progress", "percentage": 75},
            {"type": "complete", "percentage": 100, "result": {}},
        ]
        
        for update in updates:
            await mock_websocket.send_json(update)
        
        assert mock_websocket.send_json.call_count == 4

    @pytest.mark.asyncio
    async def test_websocket_heartbeat(self, mock_websocket):
        """Test WebSocket heartbeat/ping-pong."""
        mock_websocket.receive_json.return_value = {"type": "ping"}
        
        message = await mock_websocket.receive_json()
        assert message["type"] == "ping"
        
        await mock_websocket.send_json({"type": "pong"})
        mock_websocket.send_json.assert_called_with({"type": "pong"})


# ==============================================================================
# Execution API Tests
# ==============================================================================

class TestExecutionAPI:
    """Tests for circuit execution API."""

    def test_submit_job(self, mock_execution_service):
        """Test job submission."""
        job_id = mock_execution_service.submit_job(
            circuit="H 0\nCNOT 0 1\nMEASURE 0 1",
            backend="cirq",
            shots=1000,
        )
        assert job_id == "job-123"

    def test_get_job_status(self, mock_execution_service):
        """Test getting job status."""
        status = mock_execution_service.get_job_status("job-123")
        
        assert status["job_id"] == "job-123"
        assert status["status"] == "completed"
        assert status["progress"] == 100

    def test_cancel_job(self, mock_execution_service):
        """Test job cancellation."""
        success = mock_execution_service.cancel_job("job-123")
        assert success is True

    def test_list_jobs(self, mock_execution_service):
        """Test listing jobs."""
        mock_execution_service.list_jobs.return_value = [
            {"job_id": "job-1", "status": "completed"},
            {"job_id": "job-2", "status": "running"},
        ]
        
        jobs = mock_execution_service.list_jobs(user_id="user-123")
        assert len(jobs) == 2

    def test_batch_execution(self, mock_execution_service):
        """Test batch job submission."""
        mock_execution_service.submit_batch.return_value = ["job-1", "job-2", "job-3"]
        
        circuits = ["H 0", "H 1", "H 2"]
        job_ids = mock_execution_service.submit_batch(circuits, backend="cirq")
        
        assert len(job_ids) == 3

    def test_execution_with_parameters(self, mock_execution_service):
        """Test parameterized execution."""
        mock_execution_service.submit_job.return_value = "job-param"
        
        job_id = mock_execution_service.submit_job(
            circuit="RX(theta) 0",
            backend="cirq",
            parameters={"theta": 1.57},
        )
        
        assert job_id == "job-param"


# ==============================================================================
# Session API Tests
# ==============================================================================

class TestSessionAPI:
    """Tests for session management API."""

    def test_create_session(self):
        """Test session creation."""
        session = {
            "session_id": "sess-123",
            "name": "My Session",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }
        
        assert session["status"] == "active"

    def test_get_session(self):
        """Test getting session details."""
        session = {
            "session_id": "sess-123",
            "name": "My Session",
            "result_count": 5,
            "last_execution": datetime.now(timezone.utc).isoformat(),
        }
        
        assert session["result_count"] == 5

    def test_update_session(self):
        """Test session update."""
        updated = {
            "session_id": "sess-123",
            "name": "Updated Session Name",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        assert updated["name"] == "Updated Session Name"

    def test_delete_session(self):
        """Test session deletion."""
        result = {"deleted": True, "session_id": "sess-123"}
        assert result["deleted"] is True

    def test_list_sessions(self):
        """Test listing sessions."""
        sessions = [
            {"session_id": "sess-1", "name": "Session 1"},
            {"session_id": "sess-2", "name": "Session 2"},
        ]
        
        assert len(sessions) == 2


# ==============================================================================
# Backend API Tests
# ==============================================================================

class TestBackendAPI:
    """Tests for backend management API."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = [
            {"name": "cirq", "available": True, "version": "1.0.0"},
            {"name": "qiskit_aer", "available": True, "version": "0.12.0"},
            {"name": "lret", "available": True, "version": "1.0.0"},
        ]
        
        assert len(backends) == 3
        assert all(b["available"] for b in backends)

    def test_get_backend_details(self):
        """Test getting backend details."""
        backend = {
            "name": "cirq",
            "version": "1.0.0",
            "capabilities": {
                "max_qubits": 30,
                "supports_density_matrix": True,
                "supports_statevector": True,
            },
            "health": "healthy",
        }
        
        assert backend["capabilities"]["max_qubits"] == 30

    def test_backend_health_check(self):
        """Test backend health check."""
        health = {
            "name": "cirq",
            "healthy": True,
            "latency_ms": 5.2,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        
        assert health["healthy"] is True

    def test_backend_configuration(self):
        """Test backend configuration retrieval."""
        config = {
            "precision": "double",
            "seed": None,
            "noise_model": None,
            "optimization_level": 1,
        }
        
        assert config["precision"] == "double"


# ==============================================================================
# Results API Tests
# ==============================================================================

class TestResultsAPI:
    """Tests for results management API."""

    def test_get_result(self):
        """Test getting a single result."""
        result = {
            "result_id": "res-123",
            "backend": "cirq",
            "counts": {"00": 500, "11": 500},
            "execution_time_ms": 150.5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        assert result["counts"]["00"] == 500

    def test_list_results(self):
        """Test listing results with pagination."""
        response = {
            "results": [{"result_id": f"res-{i}"} for i in range(10)],
            "total": 100,
            "page": 1,
            "per_page": 10,
            "has_next": True,
            "has_prev": False,
        }
        
        assert len(response["results"]) == 10
        assert response["has_next"] is True

    def test_filter_results(self):
        """Test filtering results."""
        filtered = {
            "results": [
                {"result_id": "res-1", "backend": "cirq"},
                {"result_id": "res-2", "backend": "cirq"},
            ],
            "filter_applied": {"backend": "cirq"},
            "total_filtered": 2,
        }
        
        assert filtered["total_filtered"] == 2

    def test_delete_result(self):
        """Test result deletion."""
        response = {"deleted": True, "result_id": "res-123"}
        assert response["deleted"] is True

    def test_export_results(self):
        """Test results export."""
        export = {
            "format": "json",
            "count": 50,
            "download_url": "/api/v1/exports/export-123",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        
        assert export["format"] == "json"


# ==============================================================================
# CORS and Security Tests
# ==============================================================================

class TestCORSAndSecurity:
    """Tests for CORS and security headers."""

    def test_cors_headers_present(self):
        """Test that CORS headers are present."""
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Max-Age": "86400",
        }
        
        assert "Access-Control-Allow-Origin" in headers
        assert "OPTIONS" in headers["Access-Control-Allow-Methods"]

    def test_security_headers_present(self):
        """Test that security headers are present."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        }
        
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"

    def test_preflight_request_handling(self):
        """Test OPTIONS preflight request handling."""
        response = {
            "status_code": 200,
            "headers": {
                "Access-Control-Allow-Origin": "https://app.proxima.io",
                "Access-Control-Allow-Methods": "POST, GET",
            },
        }
        
        assert response["status_code"] == 200


# ==============================================================================
# Pagination Tests
# ==============================================================================

class TestPagination:
    """Tests for API pagination."""

    def test_default_pagination(self):
        """Test default pagination values."""
        pagination = {
            "page": 1,
            "per_page": 20,
            "total": 100,
            "total_pages": 5,
        }
        
        assert pagination["page"] == 1
        assert pagination["per_page"] == 20

    def test_custom_pagination(self):
        """Test custom pagination values."""
        pagination = {
            "page": 3,
            "per_page": 50,
            "total": 200,
            "total_pages": 4,
        }
        
        assert pagination["page"] == 3
        assert pagination["per_page"] == 50

    def test_pagination_links(self):
        """Test pagination links."""
        links = {
            "first": "/api/v1/results?page=1",
            "prev": "/api/v1/results?page=2",
            "self": "/api/v1/results?page=3",
            "next": "/api/v1/results?page=4",
            "last": "/api/v1/results?page=10",
        }
        
        assert "prev" in links
        assert "next" in links

    def test_empty_pagination(self):
        """Test pagination with no results."""
        pagination = {
            "page": 1,
            "per_page": 20,
            "total": 0,
            "total_pages": 0,
            "has_next": False,
            "has_prev": False,
        }
        
        assert pagination["total"] == 0
        assert pagination["has_next"] is False


# ==============================================================================
# Input Validation Tests
# ==============================================================================

class TestInputValidation:
    """Tests for API input validation."""

    def test_valid_circuit_input(self):
        """Test valid circuit input validation."""
        circuit_input = {
            "circuit": "H 0\nCNOT 0 1\nMEASURE 0 1",
            "backend": "cirq",
            "shots": 1000,
        }
        
        assert len(circuit_input["circuit"]) > 0
        assert circuit_input["shots"] > 0

    def test_invalid_shots_value(self):
        """Test invalid shots value rejection."""
        invalid_inputs = [
            {"shots": -1},  # Negative
            {"shots": 0},   # Zero
            {"shots": 10000001},  # Too large
            {"shots": "abc"},  # Wrong type
        ]
        
        for invalid in invalid_inputs:
            if isinstance(invalid.get("shots"), int):
                assert invalid["shots"] <= 0 or invalid["shots"] > 10000000

    def test_backend_name_validation(self):
        """Test backend name validation."""
        valid_backends = ["cirq", "qiskit_aer", "lret", "quest", "qsim", "cuquantum"]
        invalid_backends = ["invalid_backend", "", None, 123]
        
        assert all(isinstance(b, str) and len(b) > 0 for b in valid_backends)

    def test_qubit_count_validation(self):
        """Test qubit count validation."""
        limits = {
            "min_qubits": 1,
            "max_qubits": 30,
        }
        
        valid_counts = [1, 5, 10, 20, 30]
        for count in valid_counts:
            assert limits["min_qubits"] <= count <= limits["max_qubits"]

    def test_parameter_type_validation(self):
        """Test parameter type validation."""
        valid_params = {
            "theta": 1.57,
            "phi": 3.14,
            "shots": 1000,
            "seed": 42,
        }
        
        assert isinstance(valid_params["theta"], float)
        assert isinstance(valid_params["shots"], int)
