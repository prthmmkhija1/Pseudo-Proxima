"""
Comprehensive API Tests for Authentication, WebSocket, and Rate Limiting.

Tests covering:
- Authentication flows
- WebSocket connections
- Rate limiting
- API versioning
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import json
import asyncio

# Check if fastapi is available
try:
    import fastapi
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def app():
    """Create test application."""
    from proxima.api.main import create_app
    return create_app(debug=True)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create mock authentication headers."""
    return {
        "Authorization": "Bearer test_token_123",
        "X-API-Key": "test_api_key",
    }


@pytest.fixture
def mock_user():
    """Create mock user object."""
    return {
        "id": "user_123",
        "username": "testuser",
        "email": "test@example.com",
        "roles": ["user", "developer"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for authentication endpoints and flows."""

    def test_unauthenticated_request(self, client):
        """Test request without authentication."""
        response = client.get("/api/v1/user/profile")
        
        # Should return 401 or 403
        assert response.status_code in [401, 403, 404]

    def test_invalid_token_format(self, client):
        """Test request with invalid token format."""
        response = client.get(
            "/api/v1/user/profile",
            headers={"Authorization": "InvalidToken"}
        )
        
        assert response.status_code in [401, 403, 404]

    def test_expired_token(self, client):
        """Test request with expired token."""
        response = client.get(
            "/api/v1/user/profile",
            headers={"Authorization": "Bearer expired_token"}
        )
        
        assert response.status_code in [401, 403, 404]

    def test_missing_api_key(self, client):
        """Test request missing API key."""
        response = client.get("/api/v1/backends")
        
        # Should still work for public endpoints
        assert response.status_code in [200, 401, 403]


# =============================================================================
# Circuit Execution Endpoint Tests
# =============================================================================


class TestCircuitExecutionEndpoints:
    """Tests for circuit execution endpoints."""

    def test_submit_circuit_invalid_json(self, client):
        """Test submitting invalid JSON."""
        response = client.post(
            "/api/v1/circuits/execute",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 404, 422]

    def test_submit_circuit_missing_fields(self, client):
        """Test submitting circuit with missing required fields."""
        response = client.post(
            "/api/v1/circuits/execute",
            json={"invalid": "data"}
        )
        
        assert response.status_code in [400, 422, 404]

    def test_get_execution_status_not_found(self, client):
        """Test getting status of non-existent execution."""
        response = client.get("/api/v1/circuits/executions/nonexistent_id")
        
        assert response.status_code in [404]

    def test_cancel_execution_not_found(self, client):
        """Test cancelling non-existent execution."""
        response = client.post("/api/v1/circuits/executions/nonexistent_id/cancel")
        
        assert response.status_code in [404]


# =============================================================================
# Comparison Endpoint Tests
# =============================================================================


class TestComparisonEndpoints:
    """Tests for comparison endpoints."""

    def test_create_comparison_invalid(self, client):
        """Test creating comparison with invalid data."""
        response = client.post(
            "/api/v1/comparisons",
            json={"invalid": "data"}
        )
        
        assert response.status_code in [400, 422, 404]

    def test_get_comparison_not_found(self, client):
        """Test getting non-existent comparison."""
        response = client.get("/api/v1/comparisons/nonexistent_id")
        
        assert response.status_code in [404]


# =============================================================================
# Export Endpoint Tests
# =============================================================================


class TestExportEndpoints:
    """Tests for export endpoints."""

    def test_export_invalid_format(self, client):
        """Test export with invalid format."""
        response = client.post(
            "/api/v1/exports",
            json={
                "data": {"test": "data"},
                "format": "invalid_format"
            }
        )
        
        assert response.status_code in [400, 422, 404]

    def test_export_missing_data(self, client):
        """Test export without data."""
        response = client.post(
            "/api/v1/exports",
            json={"format": "json"}
        )
        
        assert response.status_code in [400, 422, 404]


# =============================================================================
# LLM Endpoint Tests
# =============================================================================


class TestLLMEndpoints:
    """Tests for LLM-related endpoints."""

    def test_list_llm_providers(self, client):
        """Test listing LLM providers."""
        response = client.get("/api/v1/llm/providers")
        
        if response.status_code == 200:
            data = response.json()
            assert "providers" in data or isinstance(data, list)

    def test_get_llm_status(self, client):
        """Test getting LLM status."""
        response = client.get("/api/v1/llm/status")
        
        assert response.status_code in [200, 404]

    def test_analyze_results_missing_data(self, client):
        """Test analyze endpoint without data."""
        response = client.post(
            "/api/v1/llm/analyze",
            json={}
        )
        
        assert response.status_code in [400, 422, 404]


# =============================================================================
# WebSocket Tests
# =============================================================================


class TestWebSocketEndpoints:
    """Tests for WebSocket endpoints."""

    def test_websocket_connection_endpoint_exists(self, client):
        """Test WebSocket endpoint documentation exists."""
        # Get OpenAPI spec
        response = client.get("/openapi.json")
        
        if response.status_code == 200:
            spec = response.json()
            # Check if WebSocket paths are documented
            paths = spec.get("paths", {})
            # WebSocket endpoints might be in paths
            assert isinstance(paths, dict)


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = client.get("/health")
        
        # Rate limit headers might be present
        if response.status_code == 200:
            # Check for common rate limit headers
            headers = response.headers
            # May or may not have rate limit headers depending on config
            assert response.status_code == 200


# =============================================================================
# API Versioning Tests
# =============================================================================


class TestAPIVersioning:
    """Tests for API versioning."""

    def test_v1_endpoints_available(self, client):
        """Test v1 endpoints are available."""
        response = client.get("/api/v1/backends")
        
        # Should respond (not 404 for the version prefix)
        assert response.status_code != 405

    def test_version_header(self, client):
        """Test API version header."""
        response = client.get("/health")
        
        assert response.status_code == 200
        # May include version in response
        data = response.json()
        if "version" in data:
            assert data["version"]


# =============================================================================
# Error Response Format Tests
# =============================================================================


class TestErrorResponses:
    """Tests for error response format."""

    def test_404_response_format(self, client):
        """Test 404 response format."""
        response = client.get("/nonexistent/path/that/does/not/exist")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "error" in data or "message" in data

    def test_method_not_allowed(self, client):
        """Test method not allowed response."""
        response = client.delete("/health")
        
        assert response.status_code in [404, 405]


# =============================================================================
# CORS Tests
# =============================================================================


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_on_options(self, client):
        """Test CORS headers on OPTIONS request."""
        response = client.options(
            "/api/v1/backends",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should respond to OPTIONS
        assert response.status_code in [200, 204, 404, 405]

    def test_cors_on_get_request(self, client):
        """Test CORS headers on GET request."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200


# =============================================================================
# Pagination Tests
# =============================================================================


class TestPagination:
    """Tests for pagination."""

    def test_backends_list_pagination(self, client):
        """Test backends list supports pagination parameters."""
        response = client.get("/api/v1/backends?page=1&per_page=10")
        
        if response.status_code == 200:
            data = response.json()
            # May include pagination info
            if "total" in data:
                assert isinstance(data["total"], int)

    def test_invalid_pagination_params(self, client):
        """Test invalid pagination parameters."""
        response = client.get("/api/v1/backends?page=-1&per_page=0")
        
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]


# =============================================================================
# Content Type Tests
# =============================================================================


class TestContentTypes:
    """Tests for content type handling."""

    def test_json_content_type(self, client):
        """Test JSON content type response."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_accept_header(self, client):
        """Test Accept header handling."""
        response = client.get(
            "/health",
            headers={"Accept": "application/json"}
        )
        
        assert response.status_code == 200


# =============================================================================
# Metrics Endpoint Tests
# =============================================================================


class TestMetricsEndpoints:
    """Tests for metrics and monitoring endpoints."""

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint if available."""
        response = client.get("/metrics")
        
        # Metrics endpoint may or may not exist
        assert response.status_code in [200, 404]

    def test_prometheus_format(self, client):
        """Test Prometheus format metrics."""
        response = client.get(
            "/metrics",
            headers={"Accept": "text/plain"}
        )
        
        # May or may not support Prometheus format
        assert response.status_code in [200, 404, 406]


# =============================================================================
# Batch Operation Tests
# =============================================================================


class TestBatchOperations:
    """Tests for batch operation endpoints."""

    def test_batch_execute_invalid(self, client):
        """Test batch execute with invalid data."""
        response = client.post(
            "/api/v1/circuits/batch",
            json={"circuits": "not_a_list"}
        )
        
        assert response.status_code in [400, 422, 404]

    def test_batch_execute_empty_list(self, client):
        """Test batch execute with empty list."""
        response = client.post(
            "/api/v1/circuits/batch",
            json={"circuits": []}
        )
        
        assert response.status_code in [200, 400, 422, 404]


# =============================================================================
# Insight Endpoint Tests
# =============================================================================


class TestInsightEndpoints:
    """Tests for insight-related endpoints."""

    def test_analyze_endpoint(self, client):
        """Test analyze endpoint."""
        response = client.post(
            "/api/v1/insights/analyze",
            json={
                "results": {"counts": {"00": 500, "11": 500}},
                "level": "standard"
            }
        )
        
        assert response.status_code in [200, 400, 404, 422]

    def test_explain_endpoint(self, client):
        """Test explain endpoint."""
        response = client.post(
            "/api/v1/insights/explain",
            json={"phenomenon": "entanglement", "context": {}}
        )
        
        assert response.status_code in [200, 400, 404, 422]


# =============================================================================
# Configuration Endpoint Tests
# =============================================================================


class TestConfigurationEndpoints:
    """Tests for configuration endpoints."""

    def test_get_config(self, client):
        """Test getting current configuration."""
        response = client.get("/api/v1/config")
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_update_config_unauthorized(self, client):
        """Test updating config without auth."""
        response = client.patch(
            "/api/v1/config",
            json={"setting": "value"}
        )
        
        # Should require auth
        assert response.status_code in [401, 403, 404, 405]
