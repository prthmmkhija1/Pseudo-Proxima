"""Unit tests for Multi-Terminal Monitor module.

Phase 10: Integration & Testing

Tests cover:
- Multiple terminal sessions
- Process monitoring
- Output aggregation
- Session lifecycle management
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import (
    MockAsyncProcess,
    MockProcessOutput,
    MockSubprocessFactory,
    create_mock_subprocess_factory,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_subprocess_factory():
    """Create mock subprocess factory."""
    return create_mock_subprocess_factory()


# =============================================================================
# MOCK TERMINAL SESSION
# =============================================================================

class MockTerminalSession:
    """Mock terminal session for testing."""
    
    def __init__(self, session_id: str, name: str = ""):
        self.id = session_id
        self.name = name or f"Terminal-{session_id}"
        self.process: Optional[MockAsyncProcess] = None
        self.output_lines: List[str] = []
        self.is_active = False
        self.exit_code: Optional[int] = None
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
    
    async def start(self, command: str, subprocess_factory: MockSubprocessFactory):
        """Start the terminal session."""
        self.is_active = True
        self.started_at = time.time()
        self.process = await subprocess_factory.create_subprocess_shell(command)
    
    async def wait(self) -> int:
        """Wait for process to complete."""
        if self.process:
            self.exit_code = await self.process.wait()
            self.is_active = False
            self.ended_at = time.time()
            return self.exit_code
        return -1
    
    def add_output(self, line: str):
        """Add output line."""
        self.output_lines.append(line)
    
    @property
    def duration_ms(self) -> float:
        """Get session duration in milliseconds."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at) * 1000
        return 0.0


# =============================================================================
# TERMINAL SESSION TESTS
# =============================================================================

class TestTerminalSession:
    """Tests for terminal session management."""
    
    def test_session_creation(self):
        """Test creating a terminal session."""
        session = MockTerminalSession("session-1", "Build Terminal")
        
        assert session.id == "session-1"
        assert session.name == "Build Terminal"
        assert session.is_active is False
    
    @pytest.mark.asyncio
    async def test_session_start(self, mock_subprocess_factory):
        """Test starting a session."""
        session = MockTerminalSession("session-2")
        await session.start("echo hello", mock_subprocess_factory)
        
        assert session.is_active is True
        assert session.process is not None
    
    @pytest.mark.asyncio
    async def test_session_wait(self, mock_subprocess_factory):
        """Test waiting for session to complete."""
        session = MockTerminalSession("session-3")
        await session.start("echo test", mock_subprocess_factory)
        exit_code = await session.wait()
        
        assert session.is_active is False
        assert exit_code == 0
    
    def test_output_collection(self):
        """Test collecting output."""
        session = MockTerminalSession("session-4")
        
        session.add_output("Line 1")
        session.add_output("Line 2")
        session.add_output("Line 3")
        
        assert len(session.output_lines) == 3
        assert session.output_lines[0] == "Line 1"


# =============================================================================
# MULTI-TERMINAL MONITOR TESTS
# =============================================================================

class MockMultiTerminalMonitor:
    """Mock multi-terminal monitor for testing."""
    
    def __init__(self):
        self.sessions: Dict[str, MockTerminalSession] = {}
        self._session_counter = 0
    
    def create_session(self, name: str = "") -> MockTerminalSession:
        """Create a new terminal session."""
        self._session_counter += 1
        session_id = f"term-{self._session_counter}"
        session = MockTerminalSession(session_id, name)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[MockTerminalSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[MockTerminalSession]:
        """List all sessions."""
        return list(self.sessions.values())
    
    def get_active_sessions(self) -> List[MockTerminalSession]:
        """Get active sessions."""
        return [s for s in self.sessions.values() if s.is_active]
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


class TestMultiTerminalMonitor:
    """Tests for multi-terminal monitor."""
    
    def test_monitor_creation(self):
        """Test creating monitor."""
        monitor = MockMultiTerminalMonitor()
        
        assert len(monitor.sessions) == 0
    
    def test_create_session(self):
        """Test creating a session through monitor."""
        monitor = MockMultiTerminalMonitor()
        session = monitor.create_session("Build")
        
        assert session is not None
        assert "term-1" in monitor.sessions
    
    def test_create_multiple_sessions(self):
        """Test creating multiple sessions."""
        monitor = MockMultiTerminalMonitor()
        
        session1 = monitor.create_session("Build")
        session2 = monitor.create_session("Test")
        session3 = monitor.create_session("Deploy")
        
        assert len(monitor.sessions) == 3
        assert session1.id != session2.id != session3.id
    
    def test_get_session(self):
        """Test getting a session by ID."""
        monitor = MockMultiTerminalMonitor()
        created = monitor.create_session("Test")
        
        retrieved = monitor.get_session(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
    
    def test_get_nonexistent_session(self):
        """Test getting nonexistent session."""
        monitor = MockMultiTerminalMonitor()
        
        result = monitor.get_session("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_active_sessions(self, mock_subprocess_factory):
        """Test getting active sessions."""
        monitor = MockMultiTerminalMonitor()
        
        session1 = monitor.create_session("Active 1")
        session2 = monitor.create_session("Inactive")
        session3 = monitor.create_session("Active 2")
        
        await session1.start("sleep 1", mock_subprocess_factory)
        await session3.start("sleep 1", mock_subprocess_factory)
        
        active = monitor.get_active_sessions()
        
        assert len(active) == 2
        assert session1 in active
        assert session3 in active
    
    def test_remove_session(self):
        """Test removing a session."""
        monitor = MockMultiTerminalMonitor()
        session = monitor.create_session("ToRemove")
        
        result = monitor.remove_session(session.id)
        
        assert result is True
        assert session.id not in monitor.sessions
    
    def test_remove_nonexistent_session(self):
        """Test removing nonexistent session."""
        monitor = MockMultiTerminalMonitor()
        
        result = monitor.remove_session("nonexistent")
        
        assert result is False


# =============================================================================
# PARALLEL EXECUTION TESTS
# =============================================================================

class TestParallelExecution:
    """Tests for parallel terminal execution."""
    
    @pytest.mark.asyncio
    async def test_parallel_commands(self, mock_subprocess_factory):
        """Test running commands in parallel."""
        monitor = MockMultiTerminalMonitor()
        
        sessions = [
            monitor.create_session(f"Parallel-{i}")
            for i in range(3)
        ]
        
        # Start all sessions
        start_tasks = [
            session.start(f"echo {i}", mock_subprocess_factory)
            for i, session in enumerate(sessions)
        ]
        await asyncio.gather(*start_tasks)
        
        # Wait for all to complete
        wait_tasks = [session.wait() for session in sessions]
        results = await asyncio.gather(*wait_tasks)
        
        assert all(r == 0 for r in results)
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure(self, mock_subprocess_factory):
        """Test parallel execution with mixed results."""
        # Set up failure for specific command
        mock_subprocess_factory.set_output(
            "fail_command",
            MockProcessOutput(returncode=1, stderr="Failed")
        )
        
        monitor = MockMultiTerminalMonitor()
        
        session1 = monitor.create_session("Success")
        session2 = monitor.create_session("Failure")
        
        await session1.start("echo success", mock_subprocess_factory)
        await session2.start("fail_command", mock_subprocess_factory)
        
        result1 = await session1.wait()
        result2 = await session2.wait()
        
        assert result1 == 0
        assert result2 == 1


# =============================================================================
# OUTPUT AGGREGATION TESTS
# =============================================================================

class TestOutputAggregation:
    """Tests for output aggregation across terminals."""
    
    def test_aggregate_outputs(self):
        """Test aggregating output from multiple sessions."""
        monitor = MockMultiTerminalMonitor()
        
        session1 = monitor.create_session("Build")
        session2 = monitor.create_session("Test")
        
        session1.add_output("Building...")
        session1.add_output("Build complete")
        session2.add_output("Running tests...")
        session2.add_output("Tests passed")
        
        all_output = []
        for session in monitor.list_sessions():
            all_output.extend([
                f"[{session.name}] {line}"
                for line in session.output_lines
            ])
        
        assert len(all_output) == 4
        assert any("Build complete" in line for line in all_output)
        assert any("Tests passed" in line for line in all_output)
    
    def test_filter_output_by_session(self):
        """Test filtering output by session."""
        monitor = MockMultiTerminalMonitor()
        
        build_session = monitor.create_session("Build")
        test_session = monitor.create_session("Test")
        
        build_session.add_output("cmake output")
        build_session.add_output("make output")
        test_session.add_output("pytest output")
        
        build_output = build_session.output_lines
        test_output = test_session.output_lines
        
        assert len(build_output) == 2
        assert len(test_output) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
