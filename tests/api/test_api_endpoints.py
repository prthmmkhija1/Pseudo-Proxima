"""
Tests for Proxima Web API endpoints.

Comprehensive tests for all REST API routes.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone


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
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def mock_backend_registry():
    """Create mock backend registry."""
    registry = Mock()
    registry.list_all.return_value = ["cirq", "qiskit_aer", "lret"]
    registry.list_available.return_value = ["cirq", "qiskit_aer"]
    
    mock_backend = Mock()
    mock_backend.get_version.return_value = "1.0.0"
    mock_backend.is_available.return_value = True
    mock_backend.get_capabilities.return_value = Mock(
        supports_density_matrix=True,
        supports_statevector=True,
        max_qubits=30,
    )
    mock_backend.get_config.return_value = Mock(precision="double")
    mock_backend.health_check.return_value = True
    
    registry.get.return_value = mock_backend
    return registry


# =============================================================================
# Health Endpoint Tests
# =============================================================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check_returns_healthy(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
    
    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ready", "degraded"]
        assert "components" in data
        assert "timestamp" in data
    
    def test_system_info(self, client):
        """Test system information endpoint."""
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "python_version" in data
        assert "platform" in data
        assert "proxima_version" in data
    
    def test_health_includes_uptime(self, client):
        """Test that health check includes uptime."""
        response = client.get("/health")
        
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# =============================================================================
# Backend Endpoint Tests
# =============================================================================

class TestBackendEndpoints:
    """Tests for backend management endpoints."""
    
    def test_list_backends(self, client):
        """Test listing all backends."""
        response = client.get("/api/v1/backends")
        
        assert response.status_code == 200
        data = response.json()
        assert "backends" in data
        assert "total" in data
        assert "available_count" in data
    
    def test_get_backend_not_found(self, client):
        """Test getting non-existent backend."""
        response = client.get("/api/v1/backends/nonexistent")
        
        assert response.status_code == 404
    
    def test_test_backend_not_found(self, client):
        """Test testing non-existent backend."""
        response = client.post(
            "/api/v1/backends/nonexistent/test",
            json={"num_qubits": 2, "shots": 100}
        )
        
        assert response.status_code == 404
    
    def test_estimate_resources_not_found(self, client):
        """Test resource estimation for non-existent backend."""
        response = client.post(
            "/api/v1/backends/nonexistent/estimate",
            json={"num_qubits": 10, "circuit_depth": 20}
        )
        
        assert response.status_code == 404


# =============================================================================
# Circuit Endpoint Tests
# =============================================================================

class TestCircuitEndpoints:
    """Tests for circuit execution endpoints."""
    
    def test_submit_circuit_sync(self, client):
        """Test synchronous circuit submission."""
        circuit = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
        response = client.post(
            "/api/v1/circuits/submit",
            json={
                "circuit": circuit,
                "format": "openqasm",
                "backend": "cirq",
                "shots": 100,
                "async_execution": False,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["backend"] == "cirq"
    
    def test_submit_circuit_async(self, client):
        """Test asynchronous circuit submission."""
        circuit = "OPENQASM 2.0;\nqreg q[2];\nh q[0];"
        
        response = client.post(
            "/api/v1/circuits/submit",
            json={
                "circuit": circuit,
                "format": "openqasm",
                "backend": "auto",
                "shots": 1000,
                "async_execution": True,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
    
    def test_get_job_status_not_found(self, client):
        """Test getting status of non-existent job."""
        response = client.get("/api/v1/circuits/jobs/nonexistent-job-id")
        
        assert response.status_code == 404
    
    def test_get_job_result_not_found(self, client):
        """Test getting result of non-existent job."""
        response = client.get("/api/v1/circuits/jobs/nonexistent-job-id/result")
        
        assert response.status_code == 404
    
    def test_cancel_job_not_found(self, client):
        """Test cancelling non-existent job."""
        response = client.delete("/api/v1/circuits/jobs/nonexistent-job-id")
        
        assert response.status_code == 404
    
    def test_validate_circuit_openqasm(self, client):
        """Test circuit validation with OpenQASM."""
        circuit = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0], q[1];
"""
        response = client.post(
            "/api/v1/circuits/validate",
            json={
                "circuit": circuit,
                "format": "openqasm",
                "backend": "cirq",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "num_qubits" in data
    
    def test_optimize_circuit(self, client):
        """Test circuit optimization."""
        circuit = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\nh q[0];"
        
        response = client.post(
            "/api/v1/circuits/optimize",
            json={
                "circuit": circuit,
                "format": "openqasm",
                "backend": "cirq",
                "optimization_level": 2,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "optimized_circuit" in data
        assert "reduction_percentage" in data
    
    def test_job_lifecycle(self, client):
        """Test complete job lifecycle."""
        # Submit
        response = client.post(
            "/api/v1/circuits/submit",
            json={
                "circuit": "OPENQASM 2.0;\nqreg q[2];\nh q[0];",
                "format": "openqasm",
                "backend": "cirq",
                "shots": 100,
                "async_execution": False,
            }
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        
        # Get status
        response = client.get(f"/api/v1/circuits/jobs/{job_id}")
        assert response.status_code == 200
        
        # Get result
        response = client.get(f"/api/v1/circuits/jobs/{job_id}/result")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["completed", "failed"]


# =============================================================================
# Session Endpoint Tests
# =============================================================================

class TestSessionEndpoints:
    """Tests for session management endpoints."""
    
    def test_create_session(self, client):
        """Test session creation."""
        response = client.post(
            "/api/v1/sessions",
            json={
                "name": "Test Session",
                "backend": "cirq",
                "ttl_minutes": 30,
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "session_id" in data
        assert data["name"] == "Test Session"
        assert data["status"] == "active"
    
    def test_list_sessions(self, client):
        """Test listing sessions."""
        # Create a session first
        client.post(
            "/api/v1/sessions",
            json={"name": "Test"}
        )
        
        response = client.get("/api/v1/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
    
    def test_get_session_not_found(self, client):
        """Test getting non-existent session."""
        response = client.get("/api/v1/sessions/nonexistent-session-id")
        
        assert response.status_code == 404
    
    def test_update_session(self, client):
        """Test session update."""
        # Create session
        create_response = client.post(
            "/api/v1/sessions",
            json={"name": "Original"}
        )
        session_id = create_response.json()["session_id"]
        
        # Update
        response = client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"name": "Updated"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated"
    
    def test_pause_resume_session(self, client):
        """Test pause and resume session."""
        # Create session
        create_response = client.post(
            "/api/v1/sessions",
            json={"name": "Test"}
        )
        session_id = create_response.json()["session_id"]
        
        # Pause
        response = client.post(f"/api/v1/sessions/{session_id}/pause")
        assert response.status_code == 200
        
        # Verify paused
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.json()["status"] == "paused"
        
        # Resume
        response = client.post(f"/api/v1/sessions/{session_id}/resume")
        assert response.status_code == 200
        
        # Verify active
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.json()["status"] == "active"
    
    def test_delete_session(self, client):
        """Test session deletion."""
        # Create session
        create_response = client.post(
            "/api/v1/sessions",
            json={"name": "To Delete"}
        )
        session_id = create_response.json()["session_id"]
        
        # Delete
        response = client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        
        # Verify deleted
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 404
    
    def test_session_history(self, client):
        """Test session history."""
        # Create session
        create_response = client.post(
            "/api/v1/sessions",
            json={"name": "Test"}
        )
        session_id = create_response.json()["session_id"]
        
        # Get history
        response = client.get(f"/api/v1/sessions/{session_id}/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert len(data["history"]) >= 1  # At least creation event


# =============================================================================
# Comparison Endpoint Tests
# =============================================================================

class TestComparisonEndpoints:
    """Tests for backend comparison endpoints."""
    
    def test_create_comparison(self, client):
        """Test creating a comparison."""
        circuit = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0], q[1];"
        
        response = client.post(
            "/api/v1/compare",
            json={
                "circuit": circuit,
                "backends": ["cirq", "qiskit_aer"],
                "shots": 100,
                "async_execution": False,
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "comparison_id" in data
        assert data["backends_requested"] == ["cirq", "qiskit_aer"]
    
    def test_list_comparisons(self, client):
        """Test listing comparisons."""
        response = client.get("/api/v1/compare")
        
        assert response.status_code == 200
        data = response.json()
        assert "comparisons" in data
        assert "total" in data
    
    def test_get_comparison_not_found(self, client):
        """Test getting non-existent comparison."""
        response = client.get("/api/v1/compare/nonexistent-comparison-id")
        
        assert response.status_code == 404
    
    def test_quick_compare(self, client):
        """Test quick comparison endpoint."""
        response = client.post(
            "/api/v1/compare/quick",
            json={
                "num_qubits": 3,
                "circuit_type": "bell",
                "backends": ["cirq", "qiskit_aer"],
                "shots": 500,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "comparison_id" in data
    
    def test_comparison_report_not_complete(self, client):
        """Test getting report for incomplete comparison."""
        # Create async comparison
        response = client.post(
            "/api/v1/compare",
            json={
                "circuit": "OPENQASM 2.0;\nqreg q[2];\nh q[0];",
                "backends": ["cirq", "qiskit_aer"],
                "shots": 100,
                "async_execution": True,
            }
        )
        comparison_id = response.json()["comparison_id"]
        
        # Try to get report immediately (may fail if still running)
        response = client.get(f"/api/v1/compare/{comparison_id}/report")
        # Status depends on timing - either 200 or 400
        assert response.status_code in [200, 400]
    
    def test_delete_comparison(self, client):
        """Test deleting a comparison."""
        # Create comparison
        response = client.post(
            "/api/v1/compare",
            json={
                "circuit": "OPENQASM 2.0;\nqreg q[2];\nh q[0];",
                "backends": ["cirq", "qiskit_aer"],
                "shots": 100,
                "async_execution": False,
            }
        )
        comparison_id = response.json()["comparison_id"]
        
        # Delete
        response = client.delete(f"/api/v1/compare/{comparison_id}")
        assert response.status_code == 200
        
        # Verify deleted
        response = client.get(f"/api/v1/compare/{comparison_id}")
        assert response.status_code == 404


# =============================================================================
# Middleware Tests
# =============================================================================

class TestMiddleware:
    """Tests for API middleware."""
    
    def test_request_id_header(self, client):
        """Test that X-Request-ID header is added."""
        response = client.get("/health")
        
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) == 8
    
    def test_process_time_header(self, client):
        """Test that X-Process-Time header is added."""
        response = client.get("/health")
        
        assert "x-process-time" in response.headers
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for API error handling."""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/circuits/submit",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_field(self, client):
        """Test handling of missing required field."""
        response = client.post(
            "/api/v1/circuits/submit",
            json={
                "format": "openqasm",
                "backend": "cirq",
                # Missing 'circuit' field
            }
        )
        
        assert response.status_code == 422
    
    def test_invalid_enum_value(self, client):
        """Test handling of invalid enum value."""
        response = client.post(
            "/api/v1/circuits/submit",
            json={
                "circuit": "test",
                "format": "invalid_format",
                "backend": "cirq",
            }
        )
        
        assert response.status_code == 422
    
    def test_value_out_of_range(self, client):
        """Test handling of value out of range."""
        response = client.post(
            "/api/v1/circuits/submit",
            json={
                "circuit": "test",
                "format": "openqasm",
                "backend": "cirq",
                "shots": -1,  # Invalid
            }
        )
        
        assert response.status_code == 422
