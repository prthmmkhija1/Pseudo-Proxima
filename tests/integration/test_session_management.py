"""
Integration tests for session management.

Tests the complete session lifecycle including creation,
execution, pause/resume, and cleanup.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Check if fastapi is available
try:
    import fastapi
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def session_service():
    """Create session service instance."""
    from proxima.api.services.session_service import SessionService
    return SessionService()


@pytest.fixture
def mock_backend():
    """Create mock backend."""
    backend = Mock()
    backend.name = "cirq"
    backend.run = Mock(return_value={
        "counts": {"00": 500, "11": 500},
        "execution_time": 0.1,
    })
    return backend


@pytest.fixture
def session_config():
    """Default session configuration."""
    return {
        "name": "Test Session",
        "backend": "cirq",
        "ttl_minutes": 30,
        "max_jobs": 100,
        "auto_cleanup": True,
    }


# =============================================================================
# Session Lifecycle Tests
# =============================================================================

class TestSessionLifecycle:
    """Tests for complete session lifecycle."""
    
    def test_create_session(self, session_service, session_config):
        """Test session creation."""
        session = session_service.create_session(**session_config)
        
        assert session["session_id"] is not None
        assert session["name"] == "Test Session"
        assert session["status"] == "active"
        assert session["backend"] == "cirq"
    
    def test_session_has_unique_id(self, session_service):
        """Test that each session has unique ID."""
        sessions = [
            session_service.create_session(name=f"Session {i}")
            for i in range(10)
        ]
        
        ids = [s["session_id"] for s in sessions]
        assert len(set(ids)) == 10  # All unique
    
    def test_get_session(self, session_service, session_config):
        """Test retrieving session."""
        created = session_service.create_session(**session_config)
        
        retrieved = session_service.get_session(created["session_id"])
        
        assert retrieved is not None
        assert retrieved["session_id"] == created["session_id"]
        assert retrieved["name"] == created["name"]
    
    def test_get_nonexistent_session(self, session_service):
        """Test retrieving non-existent session."""
        session = session_service.get_session("nonexistent")
        
        assert session is None
    
    def test_update_session(self, session_service, session_config):
        """Test updating session."""
        created = session_service.create_session(**session_config)
        
        updated = session_service.update_session(
            created["session_id"],
            name="Updated Name",
            ttl_minutes=60
        )
        
        assert updated["name"] == "Updated Name"
        assert updated["ttl_minutes"] == 60
    
    def test_delete_session(self, session_service, session_config):
        """Test deleting session."""
        created = session_service.create_session(**session_config)
        
        result = session_service.delete_session(created["session_id"])
        
        assert result is True
        assert session_service.get_session(created["session_id"]) is None
    
    def test_delete_nonexistent_session(self, session_service):
        """Test deleting non-existent session."""
        result = session_service.delete_session("nonexistent")
        
        assert result is False
    
    def test_list_sessions(self, session_service):
        """Test listing all sessions."""
        # Create multiple sessions
        for i in range(5):
            session_service.create_session(name=f"Session {i}")
        
        sessions = session_service.list_sessions()
        
        assert len(sessions) >= 5
    
    def test_list_sessions_with_status_filter(self, session_service):
        """Test listing sessions with status filter."""
        # Create sessions
        s1 = session_service.create_session(name="Active")
        s2 = session_service.create_session(name="Paused")
        
        # Pause one
        session_service.pause_session(s2["session_id"])
        
        active = session_service.list_sessions(status="active")
        paused = session_service.list_sessions(status="paused")
        
        assert all(s["status"] == "active" for s in active)
        assert all(s["status"] == "paused" for s in paused)


# =============================================================================
# Session State Tests
# =============================================================================

class TestSessionState:
    """Tests for session state transitions."""
    
    def test_pause_session(self, session_service, session_config):
        """Test pausing session."""
        created = session_service.create_session(**session_config)
        
        paused = session_service.pause_session(created["session_id"])
        
        assert paused["status"] == "paused"
    
    def test_resume_session(self, session_service, session_config):
        """Test resuming session."""
        created = session_service.create_session(**session_config)
        session_service.pause_session(created["session_id"])
        
        resumed = session_service.resume_session(created["session_id"])
        
        assert resumed["status"] == "active"
    
    def test_cannot_pause_already_paused(self, session_service, session_config):
        """Test pausing already paused session."""
        created = session_service.create_session(**session_config)
        session_service.pause_session(created["session_id"])
        
        # Should either return None or raise
        result = session_service.pause_session(created["session_id"])
        
        # Implementation dependent - verify session is still paused
        session = session_service.get_session(created["session_id"])
        assert session["status"] == "paused"
    
    def test_cannot_resume_active(self, session_service, session_config):
        """Test resuming already active session."""
        created = session_service.create_session(**session_config)
        
        result = session_service.resume_session(created["session_id"])
        
        # Should still be active
        session = session_service.get_session(created["session_id"])
        assert session["status"] == "active"
    
    def test_session_state_history(self, session_service, session_config):
        """Test session state history tracking."""
        created = session_service.create_session(**session_config)
        session_id = created["session_id"]
        
        # Perform state changes
        session_service.pause_session(session_id)
        session_service.resume_session(session_id)
        
        history = session_service.get_session_history(session_id)
        
        assert len(history) >= 3  # Created, paused, resumed
        assert history[0]["event"] == "created"


# =============================================================================
# Session Execution Tests
# =============================================================================

class TestSessionExecution:
    """Tests for circuit execution within sessions."""
    
    def test_submit_job_to_session(self, session_service, session_config):
        """Test submitting job to session."""
        created = session_service.create_session(**session_config)
        
        job = session_service.submit_job(
            created["session_id"],
            circuit="OPENQASM 2.0;\nqreg q[2];\nh q[0];",
            shots=100
        )
        
        assert job["job_id"] is not None
        assert job["session_id"] == created["session_id"]
    
    def test_cannot_submit_to_paused_session(self, session_service, session_config):
        """Test that jobs cannot be submitted to paused session."""
        created = session_service.create_session(**session_config)
        session_service.pause_session(created["session_id"])
        
        with pytest.raises(ValueError, match="paused|inactive"):
            session_service.submit_job(
                created["session_id"],
                circuit="OPENQASM 2.0;\nqreg q[2];\nh q[0];",
                shots=100
            )
    
    def test_list_session_jobs(self, session_service, session_config):
        """Test listing jobs in session."""
        created = session_service.create_session(**session_config)
        session_id = created["session_id"]
        
        # Submit multiple jobs
        for i in range(3):
            session_service.submit_job(
                session_id,
                circuit=f"OPENQASM 2.0;\nqreg q[{i+1}];\nh q[0];",
                shots=100
            )
        
        jobs = session_service.get_session_jobs(session_id)
        
        assert len(jobs) == 3
    
    def test_session_job_count_limit(self, session_service):
        """Test session job count limit."""
        created = session_service.create_session(
            name="Limited",
            max_jobs=2
        )
        
        # Submit up to limit
        session_service.submit_job(
            created["session_id"],
            circuit="test",
            shots=100
        )
        session_service.submit_job(
            created["session_id"],
            circuit="test",
            shots=100
        )
        
        # Third should fail
        with pytest.raises(ValueError, match="limit|maximum"):
            session_service.submit_job(
                created["session_id"],
                circuit="test",
                shots=100
            )


# =============================================================================
# Session TTL Tests
# =============================================================================

class TestSessionTTL:
    """Tests for session TTL (time to live) functionality."""
    
    def test_session_has_expiry(self, session_service):
        """Test that session has expiry time."""
        created = session_service.create_session(
            name="TTL Test",
            ttl_minutes=30
        )
        
        assert "expires_at" in created
        # Expiry should be in the future
        expires = datetime.fromisoformat(created["expires_at"])
        assert expires > datetime.now(timezone.utc)
    
    def test_session_extends_on_activity(self, session_service):
        """Test that session TTL extends on activity."""
        created = session_service.create_session(
            name="TTL Test",
            ttl_minutes=30
        )
        original_expiry = created["expires_at"]
        
        # Submit a job (activity)
        session_service.submit_job(
            created["session_id"],
            circuit="test",
            shots=100
        )
        
        updated = session_service.get_session(created["session_id"])
        
        # Expiry should be extended
        new_expiry = updated["expires_at"]
        assert new_expiry >= original_expiry
    
    def test_identify_expired_sessions(self, session_service):
        """Test identifying expired sessions."""
        # Create session with very short TTL
        with patch.object(session_service, '_get_current_time') as mock_time:
            # Create at t=0
            mock_time.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            created = session_service.create_session(
                name="Expire Soon",
                ttl_minutes=1
            )
            
            # Check at t=2 minutes (expired)
            mock_time.return_value = datetime(2024, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
            expired = session_service.get_expired_sessions()
            
            session_ids = [s["session_id"] for s in expired]
            assert created["session_id"] in session_ids


# =============================================================================
# Session Cleanup Tests
# =============================================================================

class TestSessionCleanup:
    """Tests for session cleanup functionality."""
    
    def test_cleanup_removes_expired(self, session_service):
        """Test that cleanup removes expired sessions."""
        # Create session with very short TTL
        with patch.object(session_service, '_get_current_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            created = session_service.create_session(
                name="Expire Soon",
                ttl_minutes=1
            )
            session_id = created["session_id"]
            
            # Run cleanup at t=2 minutes
            mock_time.return_value = datetime(2024, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
            cleaned = session_service.cleanup_expired_sessions()
            
            assert cleaned >= 1
            assert session_service.get_session(session_id) is None
    
    def test_cleanup_preserves_active(self, session_service):
        """Test that cleanup preserves active sessions."""
        created = session_service.create_session(
            name="Active Session",
            ttl_minutes=60
        )
        
        cleaned = session_service.cleanup_expired_sessions()
        
        # Session should still exist
        assert session_service.get_session(created["session_id"]) is not None
    
    def test_cleanup_cancels_pending_jobs(self, session_service):
        """Test that cleanup cancels pending jobs."""
        with patch.object(session_service, '_get_current_time') as mock_time:
            mock_time.return_value = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            created = session_service.create_session(
                name="With Jobs",
                ttl_minutes=1
            )
            
            # Submit job
            job = session_service.submit_job(
                created["session_id"],
                circuit="test",
                shots=100
            )
            
            # Cleanup at expired time
            mock_time.return_value = datetime(2024, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
            session_service.cleanup_expired_sessions()
            
            # Job should be cancelled
            job_status = session_service.get_job_status(job["job_id"])
            assert job_status is None or job_status["status"] == "cancelled"


# =============================================================================
# Session Event Tests
# =============================================================================

class TestSessionEvents:
    """Tests for session event handling."""
    
    def test_session_emits_created_event(self, session_service):
        """Test that session emits created event."""
        events = []
        session_service.on_event("session_created", events.append)
        
        created = session_service.create_session(name="Test")
        
        assert len(events) >= 1
        assert events[0]["session_id"] == created["session_id"]
    
    def test_session_emits_paused_event(self, session_service):
        """Test that session emits paused event."""
        created = session_service.create_session(name="Test")
        
        events = []
        session_service.on_event("session_paused", events.append)
        
        session_service.pause_session(created["session_id"])
        
        assert len(events) >= 1
    
    def test_session_emits_deleted_event(self, session_service):
        """Test that session emits deleted event."""
        created = session_service.create_session(name="Test")
        
        events = []
        session_service.on_event("session_deleted", events.append)
        
        session_service.delete_session(created["session_id"])
        
        assert len(events) >= 1


# =============================================================================
# Concurrent Session Tests
# =============================================================================

class TestConcurrentSessions:
    """Tests for concurrent session handling."""
    
    def test_multiple_concurrent_sessions(self, session_service):
        """Test handling multiple concurrent sessions."""
        sessions = [
            session_service.create_session(name=f"Concurrent {i}")
            for i in range(10)
        ]
        
        # All should be active
        for session in sessions:
            retrieved = session_service.get_session(session["session_id"])
            assert retrieved["status"] == "active"
    
    def test_concurrent_job_submission(self, session_service):
        """Test concurrent job submission to same session."""
        created = session_service.create_session(name="Concurrent Jobs")
        
        # Submit multiple jobs
        jobs = []
        for i in range(5):
            job = session_service.submit_job(
                created["session_id"],
                circuit=f"circuit_{i}",
                shots=100
            )
            jobs.append(job)
        
        # All jobs should have unique IDs
        job_ids = [j["job_id"] for j in jobs]
        assert len(set(job_ids)) == 5
    
    def test_session_isolation(self, session_service):
        """Test that sessions are isolated."""
        s1 = session_service.create_session(name="Session 1")
        s2 = session_service.create_session(name="Session 2")
        
        # Submit job to s1
        session_service.submit_job(s1["session_id"], circuit="test", shots=100)
        
        # s2 should have no jobs
        s2_jobs = session_service.get_session_jobs(s2["session_id"])
        assert len(s2_jobs) == 0
        
        # s1 should have 1 job
        s1_jobs = session_service.get_session_jobs(s1["session_id"])
        assert len(s1_jobs) == 1


# =============================================================================
# Session Persistence Tests
# =============================================================================

class TestSessionPersistence:
    """Tests for session persistence (if implemented)."""
    
    def test_session_survives_service_restart(self, session_service):
        """Test session persistence across service restarts."""
        created = session_service.create_session(name="Persistent")
        session_id = created["session_id"]
        
        # If service supports persistence, this would test reload
        if hasattr(session_service, 'save_state') and hasattr(session_service, 'load_state'):
            state = session_service.save_state()
            
            new_service = session_service.__class__()
            new_service.load_state(state)
            
            retrieved = new_service.get_session(session_id)
            assert retrieved is not None
            assert retrieved["name"] == "Persistent"
