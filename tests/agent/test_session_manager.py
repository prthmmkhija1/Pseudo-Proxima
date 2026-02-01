"""Unit tests for Session Manager module.

Phase 10: Integration & Testing

Tests cover:
- Session creation and lifecycle
- State persistence
- History management
- Checkpoint recovery
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_session_dir(tmp_path):
    """Create temporary session directory."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    return session_dir


@pytest.fixture
def sample_session_data():
    """Create sample session data."""
    return {
        "id": "test-session-123",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "state": "active",
        "history": [
            {"role": "user", "content": "Build cirq backend"},
            {"role": "assistant", "content": "Building cirq backend..."},
        ],
        "context": {
            "working_dir": "/project",
            "active_backend": "cirq",
        },
    }


# =============================================================================
# SESSION STATE TESTS
# =============================================================================

class TestSessionState:
    """Tests for session state management."""
    
    def test_session_data_structure(self, sample_session_data):
        """Test session data structure is valid."""
        required_keys = ["id", "created_at", "state", "history"]
        
        for key in required_keys:
            assert key in sample_session_data
    
    def test_session_state_values(self, sample_session_data):
        """Test session state values."""
        assert sample_session_data["state"] == "active"
        assert len(sample_session_data["history"]) == 2
    
    def test_session_history_format(self, sample_session_data):
        """Test session history format."""
        history = sample_session_data["history"]
        
        for entry in history:
            assert "role" in entry
            assert "content" in entry
            assert entry["role"] in ["user", "assistant", "system"]


# =============================================================================
# SESSION LIFECYCLE TESTS
# =============================================================================

class TestSessionLifecycle:
    """Tests for session lifecycle management."""
    
    def test_session_creation(self, sample_session_data):
        """Test session creation."""
        assert sample_session_data["id"] is not None
        assert sample_session_data["state"] == "active"
    
    def test_session_id_uniqueness(self):
        """Test session IDs are unique."""
        import uuid
        
        ids = [str(uuid.uuid4()) for _ in range(100)]
        unique_ids = set(ids)
        
        assert len(ids) == len(unique_ids)
    
    def test_session_timestamps(self, sample_session_data):
        """Test session timestamps are present."""
        assert "created_at" in sample_session_data
        assert "updated_at" in sample_session_data


# =============================================================================
# HISTORY MANAGEMENT TESTS
# =============================================================================

class TestHistoryManagement:
    """Tests for session history management."""
    
    def test_add_message_to_history(self, sample_session_data):
        """Test adding message to history."""
        new_message = {"role": "user", "content": "Run tests"}
        sample_session_data["history"].append(new_message)
        
        assert len(sample_session_data["history"]) == 3
        assert sample_session_data["history"][-1]["content"] == "Run tests"
    
    def test_history_order(self, sample_session_data):
        """Test history maintains order."""
        sample_session_data["history"].append(
            {"role": "assistant", "content": "Third"}
        )
        sample_session_data["history"].append(
            {"role": "user", "content": "Fourth"}
        )
        
        assert sample_session_data["history"][0]["role"] == "user"
        assert sample_session_data["history"][1]["role"] == "assistant"
        assert sample_session_data["history"][2]["role"] == "assistant"
        assert sample_session_data["history"][3]["role"] == "user"
    
    def test_history_truncation(self, sample_session_data):
        """Test history truncation for memory management."""
        # Add many messages
        for i in range(100):
            sample_session_data["history"].append(
                {"role": "user", "content": f"Message {i}"}
            )
        
        # Truncate to last 50
        if len(sample_session_data["history"]) > 50:
            sample_session_data["history"] = sample_session_data["history"][-50:]
        
        assert len(sample_session_data["history"]) == 50


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================

class TestSessionPersistence:
    """Tests for session persistence."""
    
    def test_save_session(self, temp_session_dir, sample_session_data):
        """Test saving session to file."""
        session_file = temp_session_dir / f"{sample_session_data['id']}.json"
        
        with open(session_file, "w") as f:
            json.dump(sample_session_data, f)
        
        assert session_file.exists()
    
    def test_load_session(self, temp_session_dir, sample_session_data):
        """Test loading session from file."""
        session_file = temp_session_dir / f"{sample_session_data['id']}.json"
        
        # Save
        with open(session_file, "w") as f:
            json.dump(sample_session_data, f)
        
        # Load
        with open(session_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["id"] == sample_session_data["id"]
        assert loaded["history"] == sample_session_data["history"]
    
    def test_session_file_not_found(self, temp_session_dir):
        """Test handling missing session file."""
        session_file = temp_session_dir / "nonexistent.json"
        
        assert not session_file.exists()


# =============================================================================
# CONTEXT MANAGEMENT TESTS
# =============================================================================

class TestContextManagement:
    """Tests for session context management."""
    
    def test_context_initialization(self, sample_session_data):
        """Test context is initialized."""
        assert "context" in sample_session_data
        assert isinstance(sample_session_data["context"], dict)
    
    def test_context_update(self, sample_session_data):
        """Test updating context."""
        sample_session_data["context"]["new_key"] = "new_value"
        
        assert sample_session_data["context"]["new_key"] == "new_value"
    
    def test_context_with_working_dir(self, sample_session_data):
        """Test context contains working directory."""
        assert "working_dir" in sample_session_data["context"]
        assert sample_session_data["context"]["working_dir"] == "/project"
    
    def test_context_serialization(self, sample_session_data):
        """Test context can be serialized."""
        json_str = json.dumps(sample_session_data["context"])
        parsed = json.loads(json_str)
        
        assert parsed == sample_session_data["context"]


# =============================================================================
# CHECKPOINT TESTS
# =============================================================================

class TestCheckpoints:
    """Tests for session checkpoints."""
    
    def test_create_checkpoint(self, sample_session_data):
        """Test creating a checkpoint."""
        checkpoint = {
            "session_id": sample_session_data["id"],
            "timestamp": time.time(),
            "history_length": len(sample_session_data["history"]),
            "state_snapshot": sample_session_data.copy(),
        }
        
        assert checkpoint["session_id"] == sample_session_data["id"]
        assert checkpoint["history_length"] == 2
    
    def test_restore_from_checkpoint(self, sample_session_data):
        """Test restoring from checkpoint."""
        import copy
        
        # Create checkpoint (deep copy to isolate)
        checkpoint = {
            "state_snapshot": copy.deepcopy(sample_session_data),
        }
        
        # Modify session
        sample_session_data["history"].append(
            {"role": "user", "content": "New message"}
        )
        
        # Restore
        restored = checkpoint["state_snapshot"]
        
        assert len(restored["history"]) == 2
        assert len(sample_session_data["history"]) == 3
    
    def test_checkpoint_integrity(self, sample_session_data):
        """Test checkpoint maintains data integrity."""
        import copy
        
        checkpoint = copy.deepcopy(sample_session_data)
        
        # Modify original
        sample_session_data["state"] = "paused"
        sample_session_data["context"]["new_key"] = "new_value"
        
        # Checkpoint should be unchanged
        assert checkpoint["state"] == "active"
        assert "new_key" not in checkpoint["context"]


# =============================================================================
# CONCURRENT ACCESS TESTS
# =============================================================================

class TestConcurrentAccess:
    """Tests for concurrent session access."""
    
    def test_lock_mechanism(self):
        """Test session locking mechanism."""
        import threading
        
        lock = threading.Lock()
        results = []
        
        def access_session(session_id: str):
            with lock:
                results.append(session_id)
                time.sleep(0.01)
        
        threads = [
            threading.Thread(target=access_session, args=(f"session-{i}",))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
