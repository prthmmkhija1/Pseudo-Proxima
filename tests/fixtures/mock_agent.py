"""Mock fixtures for Agent module testing.

Phase 10: Integration & Testing

Provides comprehensive mocks for:
- LLM API responses
- Terminal subprocess execution
- File system operations
- Git operations
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# LLM MOCK RESPONSES
# =============================================================================

@dataclass
class MockLLMResponse:
    """Mock LLM API response."""
    content: str
    model: str = "gpt-4"
    prompt_tokens: int = 100
    completion_tokens: int = 50
    finish_reason: str = "stop"
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self.content,
                    },
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic API format."""
        return {
            "id": f"msg_{int(time.time())}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": self.content}],
            "model": self.model,
            "stop_reason": self.finish_reason,
            "usage": {
                "input_tokens": self.prompt_tokens,
                "output_tokens": self.completion_tokens,
            },
        }


class MockLLMClient:
    """Mock LLM client for testing.
    
    Provides canned responses for different query types.
    """
    
    # Predefined responses for different intents
    RESPONSES = {
        "build": MockLLMResponse(
            content='{"action": "build_backend", "backend": "cirq", "options": {"gpu": false}}',
        ),
        "git_clone": MockLLMResponse(
            content='{"action": "git_clone", "url": "https://github.com/test/repo.git"}',
        ),
        "file_read": MockLLMResponse(
            content='{"action": "read_file", "path": "test.py"}',
        ),
        "file_write": MockLLMResponse(
            content='{"action": "write_file", "path": "test.py", "content": "# Test"}',
        ),
        "run_command": MockLLMResponse(
            content='{"action": "run_command", "command": "echo hello"}',
        ),
        "default": MockLLMResponse(
            content="I understand your request. How can I help you further?",
        ),
    }
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        latency_ms: float = 100.0,
        fail_rate: float = 0.0,
    ):
        """Initialize mock client."""
        self.provider = provider
        self.model = model
        self.latency_ms = latency_ms
        self.fail_rate = fail_rate
        self._request_count = 0
        self._responses: List[MockLLMResponse] = []
    
    def queue_response(self, response: MockLLMResponse) -> None:
        """Queue a specific response."""
        self._responses.append(response)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Simulate chat completion."""
        self._request_count += 1
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Simulate failures
        import random
        if random.random() < self.fail_rate:
            raise Exception("Simulated API failure")
        
        # Get response
        if self._responses:
            response = self._responses.pop(0)
        else:
            # Determine response based on last message content
            last_message = messages[-1]["content"].lower() if messages else ""
            response = self._get_response_for_query(last_message)
        
        if self.provider == "anthropic":
            return response.to_anthropic_format()
        return response.to_openai_format()
    
    def _get_response_for_query(self, query: str) -> MockLLMResponse:
        """Get appropriate response for query."""
        if "build" in query:
            return self.RESPONSES["build"]
        elif "clone" in query or "git" in query:
            return self.RESPONSES["git_clone"]
        elif "read" in query and "file" in query:
            return self.RESPONSES["file_read"]
        elif "write" in query and "file" in query:
            return self.RESPONSES["file_write"]
        elif "run" in query or "execute" in query or "command" in query:
            return self.RESPONSES["run_command"]
        return self.RESPONSES["default"]


# =============================================================================
# SUBPROCESS / TERMINAL MOCKS
# =============================================================================

@dataclass
class MockProcessOutput:
    """Mock process output."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration_ms: float = 100.0


class MockAsyncProcess:
    """Mock async subprocess."""
    
    def __init__(
        self,
        output: MockProcessOutput,
        stream_output: bool = False,
    ):
        """Initialize mock process."""
        self._output = output
        self._stream_output = stream_output
        self._stdout_lines = output.stdout.split('\n') if output.stdout else []
        self._stderr_lines = output.stderr.split('\n') if output.stderr else []
        self._line_index = 0
        self.returncode = output.returncode
        self.pid = 12345
        
        # Create mock streams
        self.stdout = MagicMock()
        self.stderr = MagicMock()
        
        if stream_output:
            self.stdout.readline = AsyncMock(side_effect=self._read_stdout_line)
            self.stderr.readline = AsyncMock(side_effect=self._read_stderr_line)
        else:
            self.stdout.read = AsyncMock(return_value=output.stdout.encode())
            self.stderr.read = AsyncMock(return_value=output.stderr.encode())
    
    async def _read_stdout_line(self) -> bytes:
        """Read next stdout line."""
        if self._line_index < len(self._stdout_lines):
            line = self._stdout_lines[self._line_index]
            self._line_index += 1
            await asyncio.sleep(0.01)  # Simulate delay
            return (line + '\n').encode()
        return b''
    
    async def _read_stderr_line(self) -> bytes:
        """Read stderr line."""
        if self._stderr_lines:
            return (self._stderr_lines.pop(0) + '\n').encode()
        return b''
    
    async def wait(self) -> int:
        """Wait for process completion."""
        await asyncio.sleep(self._output.duration_ms / 1000)
        return self.returncode
    
    async def communicate(self) -> Tuple[bytes, bytes]:
        """Get stdout and stderr."""
        await asyncio.sleep(self._output.duration_ms / 1000)
        return (
            self._output.stdout.encode(),
            self._output.stderr.encode(),
        )
    
    def terminate(self) -> None:
        """Terminate process."""
        pass
    
    def kill(self) -> None:
        """Kill process."""
        pass


class MockSubprocessFactory:
    """Factory for creating mock subprocesses.
    
    Provides canned outputs for different commands.
    """
    
    # Predefined outputs for common commands
    COMMAND_OUTPUTS = {
        "echo": MockProcessOutput(stdout="hello\n", returncode=0),
        "git status": MockProcessOutput(
            stdout="On branch main\nnothing to commit, working tree clean\n",
            returncode=0,
        ),
        "git clone": MockProcessOutput(
            stdout="Cloning into 'repo'...\ndone.\n",
            returncode=0,
        ),
        "pip install": MockProcessOutput(
            stdout="Successfully installed package\n",
            returncode=0,
        ),
        "python": MockProcessOutput(stdout="Python 3.11.0\n", returncode=0),
        "cmake": MockProcessOutput(stdout="-- Build files generated\n", returncode=0),
        "make": MockProcessOutput(stdout="Build complete\n", returncode=0),
    }
    
    def __init__(self):
        """Initialize factory."""
        self._custom_outputs: Dict[str, MockProcessOutput] = {}
    
    def set_output(self, command_pattern: str, output: MockProcessOutput) -> None:
        """Set custom output for a command pattern."""
        self._custom_outputs[command_pattern] = output
    
    def get_output(self, command: str) -> MockProcessOutput:
        """Get output for a command."""
        # Check custom outputs first
        for pattern, output in self._custom_outputs.items():
            if pattern in command:
                return output
        
        # Check predefined outputs
        for pattern, output in self.COMMAND_OUTPUTS.items():
            if pattern in command:
                return output
        
        # Default output
        return MockProcessOutput(stdout="", returncode=0)
    
    async def create_subprocess_exec(
        self,
        *args: str,
        **kwargs: Any,
    ) -> MockAsyncProcess:
        """Create mock subprocess."""
        command = " ".join(args)
        output = self.get_output(command)
        return MockAsyncProcess(output, stream_output=True)
    
    async def create_subprocess_shell(
        self,
        command: str,
        **kwargs: Any,
    ) -> MockAsyncProcess:
        """Create mock subprocess from shell command."""
        output = self.get_output(command)
        return MockAsyncProcess(output, stream_output=True)


# =============================================================================
# FILE SYSTEM MOCKS
# =============================================================================

class MockFileSystem:
    """Mock file system for testing.
    
    Provides in-memory file storage.
    """
    
    def __init__(self):
        """Initialize mock file system."""
        self._files: Dict[str, str] = {}
        self._directories: set = {"/"}
    
    def write_file(self, path: str, content: str) -> None:
        """Write file."""
        self._files[path] = content
        # Create parent directories
        parent = str(Path(path).parent)
        self._directories.add(parent)
    
    def read_file(self, path: str) -> str:
        """Read file."""
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        return self._files[path]
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        return path in self._files or path in self._directories
    
    def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        return path in self._files
    
    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        return path in self._directories
    
    def mkdir(self, path: str, parents: bool = False) -> None:
        """Create directory."""
        self._directories.add(path)
    
    def delete(self, path: str) -> None:
        """Delete file or directory."""
        if path in self._files:
            del self._files[path]
        self._directories.discard(path)
    
    def list_dir(self, path: str) -> List[str]:
        """List directory contents."""
        result = []
        for file_path in self._files:
            if str(Path(file_path).parent) == path:
                result.append(Path(file_path).name)
        return result


# =============================================================================
# GIT OPERATION MOCKS
# =============================================================================

class MockGitRepository:
    """Mock git repository for testing."""
    
    def __init__(self, path: str = "/mock/repo"):
        """Initialize mock repository."""
        self.path = path
        self.current_branch = "main"
        self.branches = ["main"]
        self.commits: List[Dict[str, Any]] = []
        self.staged_files: List[str] = []
        self.modified_files: List[str] = []
        self.remote_url = "https://github.com/test/repo.git"
    
    def add_commit(
        self,
        message: str,
        files: List[str],
        author: str = "Test User",
    ) -> str:
        """Add a commit."""
        commit_hash = f"abc{len(self.commits):04d}"
        self.commits.append({
            "hash": commit_hash,
            "message": message,
            "files": files,
            "author": author,
            "timestamp": time.time(),
        })
        return commit_hash
    
    def create_branch(self, name: str) -> bool:
        """Create a branch."""
        if name not in self.branches:
            self.branches.append(name)
            return True
        return False
    
    def checkout(self, branch: str) -> bool:
        """Checkout a branch."""
        if branch in self.branches:
            self.current_branch = branch
            return True
        return False
    
    def stage(self, files: List[str]) -> None:
        """Stage files."""
        self.staged_files.extend(files)
    
    def get_status(self) -> Dict[str, Any]:
        """Get repository status."""
        return {
            "branch": self.current_branch,
            "staged": self.staged_files,
            "modified": self.modified_files,
            "untracked": [],
            "clean": len(self.staged_files) == 0 and len(self.modified_files) == 0,
        }


# =============================================================================
# TELEMETRY MOCKS
# =============================================================================

class MockTelemetry:
    """Mock telemetry for testing."""
    
    def __init__(self):
        """Initialize mock telemetry."""
        self.events: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}
    
    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an event."""
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        })
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a metric."""
        self.metrics[name] = value
    
    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recorded events."""
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self.events.clear()
        self.metrics.clear()


# =============================================================================
# CONSENT MOCK
# =============================================================================

class MockConsentManager:
    """Mock consent manager for testing."""
    
    def __init__(self, auto_approve: bool = True):
        """Initialize mock consent manager."""
        self.auto_approve = auto_approve
        self.requests: List[Dict[str, Any]] = []
    
    async def request_consent(
        self,
        operation: str,
        description: str,
        risk_level: str = "low",
    ) -> bool:
        """Request consent for operation."""
        self.requests.append({
            "operation": operation,
            "description": description,
            "risk_level": risk_level,
            "timestamp": time.time(),
            "approved": self.auto_approve,
        })
        return self.auto_approve
    
    def get_requests(self) -> List[Dict[str, Any]]:
        """Get consent requests."""
        return self.requests


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_mock_llm_client(**kwargs: Any) -> MockLLMClient:
    """Create a mock LLM client."""
    return MockLLMClient(**kwargs)


def create_mock_subprocess_factory() -> MockSubprocessFactory:
    """Create a mock subprocess factory."""
    return MockSubprocessFactory()


def create_mock_file_system() -> MockFileSystem:
    """Create a mock file system."""
    return MockFileSystem()


def create_mock_git_repo(path: str = "/mock/repo") -> MockGitRepository:
    """Create a mock git repository."""
    return MockGitRepository(path)


def create_mock_telemetry() -> MockTelemetry:
    """Create a mock telemetry instance."""
    return MockTelemetry()


def create_mock_consent_manager(auto_approve: bool = True) -> MockConsentManager:
    """Create a mock consent manager."""
    return MockConsentManager(auto_approve)
