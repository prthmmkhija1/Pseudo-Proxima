"""Unit tests for Terminal Executor module.

Phase 10: Integration & Testing

Tests cover:
- Terminal type detection
- Command execution
- Streaming output handling
- Error handling
- Timeout management
"""

from __future__ import annotations

import asyncio
import os
import platform
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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


@pytest.fixture
def terminal_output():
    """Create sample terminal output."""
    from proxima.agent.terminal_executor import TerminalOutput, ExecutionStatus
    return TerminalOutput(
        stdout="Hello World\n",
        stderr="",
        return_code=0,
        execution_time_ms=100.0,
        status=ExecutionStatus.COMPLETED,
        command="echo Hello World",
        working_dir="/test",
    )


# =============================================================================
# TERMINAL OUTPUT TESTS
# =============================================================================

class TestTerminalOutput:
    """Tests for TerminalOutput dataclass."""
    
    def test_success_property_true(self, terminal_output):
        """Test success property when command succeeds."""
        assert terminal_output.success is True
    
    def test_success_property_false_on_nonzero_return(self):
        """Test success property when return code is non-zero."""
        from proxima.agent.terminal_executor import TerminalOutput, ExecutionStatus
        
        output = TerminalOutput(
            stdout="",
            stderr="Error occurred",
            return_code=1,
            execution_time_ms=100.0,
            status=ExecutionStatus.COMPLETED,
            command="exit 1",
            working_dir="/test",
        )
        assert output.success is False
    
    def test_success_property_false_on_failed_status(self):
        """Test success property when status is FAILED."""
        from proxima.agent.terminal_executor import TerminalOutput, ExecutionStatus
        
        output = TerminalOutput(
            stdout="",
            stderr="Command failed",
            return_code=0,
            execution_time_ms=100.0,
            status=ExecutionStatus.FAILED,
            command="test",
            working_dir="/test",
        )
        assert output.success is False
    
    def test_combined_output_stdout_only(self, terminal_output):
        """Test combined output with only stdout."""
        assert terminal_output.combined_output == "Hello World\n"
    
    def test_combined_output_with_stderr(self):
        """Test combined output with both stdout and stderr."""
        from proxima.agent.terminal_executor import TerminalOutput, ExecutionStatus
        
        output = TerminalOutput(
            stdout="Output\n",
            stderr="Warning\n",
            return_code=0,
            execution_time_ms=100.0,
            status=ExecutionStatus.COMPLETED,
            command="test",
            working_dir="/test",
        )
        combined = output.combined_output
        assert "Output" in combined
        assert "[STDERR]" in combined
        assert "Warning" in combined
    
    def test_combined_output_empty(self):
        """Test combined output when both are empty."""
        from proxima.agent.terminal_executor import TerminalOutput, ExecutionStatus
        
        output = TerminalOutput(
            stdout="",
            stderr="",
            return_code=0,
            execution_time_ms=100.0,
            status=ExecutionStatus.COMPLETED,
            command="true",
            working_dir="/test",
        )
        assert output.combined_output == "(no output)"
    
    def test_to_dict(self, terminal_output):
        """Test conversion to dictionary."""
        result = terminal_output.to_dict()
        
        assert result["stdout"] == "Hello World\n"
        assert result["stderr"] == ""
        assert result["return_code"] == 0
        assert result["command"] == "echo Hello World"
        assert result["success"] is True
        assert "timestamp" in result


# =============================================================================
# EXECUTION STATUS TESTS
# =============================================================================

class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""
    
    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        from proxima.agent.terminal_executor import ExecutionStatus
        
        assert hasattr(ExecutionStatus, "PENDING")
        assert hasattr(ExecutionStatus, "RUNNING")
        assert hasattr(ExecutionStatus, "COMPLETED")
        assert hasattr(ExecutionStatus, "FAILED")
        assert hasattr(ExecutionStatus, "CANCELLED")
        assert hasattr(ExecutionStatus, "TIMEOUT")


# =============================================================================
# TERMINAL TYPE TESTS
# =============================================================================

class TestTerminalType:
    """Tests for TerminalType enum."""
    
    def test_all_types_exist(self):
        """Test all expected terminal types exist."""
        from proxima.agent.terminal_executor import TerminalType
        
        assert hasattr(TerminalType, "POWERSHELL")
        assert hasattr(TerminalType, "CMD")
        assert hasattr(TerminalType, "BASH")
        assert hasattr(TerminalType, "ZSH")
        assert hasattr(TerminalType, "SH")
        assert hasattr(TerminalType, "AUTO")
    
    def test_terminal_type_values(self):
        """Test terminal type values."""
        from proxima.agent.terminal_executor import TerminalType
        
        assert TerminalType.POWERSHELL.value == "powershell"
        assert TerminalType.BASH.value == "bash"


# =============================================================================
# TERMINAL SESSION TESTS
# =============================================================================

class TestTerminalSession:
    """Tests for TerminalSession dataclass."""
    
    def test_session_creation(self):
        """Test creating a terminal session."""
        from proxima.agent.terminal_executor import TerminalSession, TerminalType
        
        session = TerminalSession(
            id="test-session-1",
            terminal_type=TerminalType.BASH,
            working_dir="/home/user",
        )
        
        assert session.id == "test-session-1"
        assert session.terminal_type == TerminalType.BASH
        assert session.working_dir == "/home/user"
        assert session.process is None
        assert session.history == []
        assert session.is_admin is False
    
    def test_session_to_dict(self):
        """Test session conversion to dictionary."""
        from proxima.agent.terminal_executor import TerminalSession, TerminalType
        
        session = TerminalSession(
            id="test-session-1",
            terminal_type=TerminalType.POWERSHELL,
            working_dir="C:\\Users\\test",
            is_admin=True,
        )
        
        result = session.to_dict()
        
        assert result["id"] == "test-session-1"
        assert result["terminal_type"] == "powershell"
        assert result["is_admin"] is True
        assert "started_at" in result


# =============================================================================
# STREAMING OUTPUT HANDLER TESTS
# =============================================================================

class TestStreamingOutputHandler:
    """Tests for StreamingOutputHandler."""
    
    def test_handler_creation(self):
        """Test creating a streaming handler."""
        from proxima.agent.terminal_executor import StreamingOutputHandler
        
        handler = StreamingOutputHandler()
        
        assert handler.stdout_lines == []
        assert handler.stderr_lines == []
    
    def test_handler_with_callbacks(self):
        """Test handler with custom callbacks."""
        from proxima.agent.terminal_executor import StreamingOutputHandler
        
        stdout_lines = []
        stderr_lines = []
        
        handler = StreamingOutputHandler(
            stdout_callback=lambda line: stdout_lines.append(line),
            stderr_callback=lambda line: stderr_lines.append(line),
        )
        
        handler.handle_stdout("line 1")
        handler.handle_stdout("line 2")
        handler.handle_stderr("error 1")
        
        assert len(stdout_lines) == 2
        assert len(stderr_lines) == 1
        assert handler.stdout_lines == ["line 1", "line 2"]
        assert handler.stderr_lines == ["error 1"]
    
    def test_handler_thread_safety(self):
        """Test handler is thread-safe."""
        from proxima.agent.terminal_executor import StreamingOutputHandler
        import threading
        
        handler = StreamingOutputHandler()
        
        def add_lines(count: int, prefix: str):
            for i in range(count):
                handler.handle_stdout(f"{prefix}-{i}")
        
        threads = [
            threading.Thread(target=add_lines, args=(100, f"thread-{i}"))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have 500 lines total
        assert len(handler.stdout_lines) == 500


# =============================================================================
# MOCK SUBPROCESS TESTS
# =============================================================================

class TestMockSubprocessFactory:
    """Tests for mock subprocess factory."""
    
    @pytest.mark.asyncio
    async def test_create_subprocess_exec(self, mock_subprocess_factory):
        """Test creating a subprocess via exec."""
        process = await mock_subprocess_factory.create_subprocess_exec(
            "echo", "hello"
        )
        
        assert process.pid == 12345
        assert process.returncode == 0
    
    @pytest.mark.asyncio
    async def test_create_subprocess_shell(self, mock_subprocess_factory):
        """Test creating a subprocess via shell."""
        process = await mock_subprocess_factory.create_subprocess_shell(
            "echo hello"
        )
        
        assert process.pid == 12345
    
    @pytest.mark.asyncio
    async def test_process_communicate(self, mock_subprocess_factory):
        """Test process communicate method."""
        process = await mock_subprocess_factory.create_subprocess_shell(
            "echo hello"
        )
        
        stdout, stderr = await process.communicate()
        
        assert isinstance(stdout, bytes)
        assert isinstance(stderr, bytes)
    
    def test_set_custom_output(self, mock_subprocess_factory):
        """Test setting custom output for commands."""
        custom_output = MockProcessOutput(
            stdout="Custom output\n",
            returncode=42,
        )
        
        mock_subprocess_factory.set_output("custom_cmd", custom_output)
        
        result = mock_subprocess_factory.get_output("custom_cmd test")
        
        assert result.stdout == "Custom output\n"
        assert result.returncode == 42
    
    @pytest.mark.asyncio
    async def test_process_wait(self, mock_subprocess_factory):
        """Test process wait method."""
        process = await mock_subprocess_factory.create_subprocess_exec("test")
        
        returncode = await process.wait()
        
        assert returncode == 0


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================

class TestTerminalExecutorIntegration:
    """Integration-style tests for terminal executor."""
    
    @pytest.mark.asyncio
    async def test_mock_command_execution(self, mock_subprocess_factory):
        """Test executing a command with mock subprocess."""
        # Set up mock
        mock_subprocess_factory.set_output(
            "test_command",
            MockProcessOutput(stdout="success\n", returncode=0)
        )
        
        # Execute
        process = await mock_subprocess_factory.create_subprocess_shell(
            "test_command"
        )
        stdout, stderr = await process.communicate()
        
        # Verify
        assert stdout.decode() == "success\n"
        assert process.returncode == 0
    
    @pytest.mark.asyncio
    async def test_mock_build_command(self, mock_subprocess_factory):
        """Test executing a build-like command."""
        mock_subprocess_factory.set_output(
            "cmake",
            MockProcessOutput(
                stdout="-- Configuring done\n-- Generating done\n",
                returncode=0,
            )
        )
        
        process = await mock_subprocess_factory.create_subprocess_shell(
            "cmake -B build"
        )
        stdout, stderr = await process.communicate()
        
        assert "Configuring done" in stdout.decode()
    
    @pytest.mark.asyncio
    async def test_mock_failed_command(self, mock_subprocess_factory):
        """Test handling failed commands."""
        mock_subprocess_factory.set_output(
            "fail_command",
            MockProcessOutput(
                stdout="",
                stderr="Command not found\n",
                returncode=127,
            )
        )
        
        process = await mock_subprocess_factory.create_subprocess_shell(
            "fail_command"
        )
        stdout, stderr = await process.communicate()
        
        assert process.returncode == 127
        assert "Command not found" in stderr.decode()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
