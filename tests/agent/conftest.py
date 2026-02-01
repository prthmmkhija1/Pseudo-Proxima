"""Agent-specific pytest configuration and fixtures.

Phase 10: Integration & Testing

Provides fixtures specifically for agent module testing.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.mock_agent import (
    MockLLMClient,
    MockLLMResponse,
    MockSubprocessFactory,
    MockProcessOutput,
    MockFileSystem,
    MockGitRepository,
    MockTelemetry,
    MockConsentManager,
    create_mock_llm_client,
    create_mock_subprocess_factory,
    create_mock_file_system,
    create_mock_git_repo,
    create_mock_telemetry,
    create_mock_consent_manager,
)


# =============================================================================
# LLM FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create a mock LLM client with default settings."""
    return create_mock_llm_client()


@pytest.fixture
def fast_llm_client() -> MockLLMClient:
    """Create a fast mock LLM client for performance testing."""
    return create_mock_llm_client(latency_ms=5.0)


@pytest.fixture
def failing_llm_client() -> MockLLMClient:
    """Create an LLM client that fails frequently."""
    return create_mock_llm_client(fail_rate=0.5)


# =============================================================================
# SUBPROCESS FIXTURES
# =============================================================================

@pytest.fixture
def mock_subprocess_factory() -> MockSubprocessFactory:
    """Create a mock subprocess factory."""
    return create_mock_subprocess_factory()


@pytest.fixture
def configured_subprocess_factory() -> MockSubprocessFactory:
    """Create a subprocess factory with common commands configured."""
    factory = create_mock_subprocess_factory()
    
    # Configure common commands
    factory.set_output("echo", MockProcessOutput(stdout="hello\n", returncode=0))
    factory.set_output("pwd", MockProcessOutput(stdout="/test/dir\n", returncode=0))
    factory.set_output("ls", MockProcessOutput(stdout="file1.py\nfile2.py\n", returncode=0))
    factory.set_output("cmake", MockProcessOutput(stdout="-- Build files generated\n", returncode=0))
    factory.set_output("make", MockProcessOutput(stdout="[100%] Built\n", returncode=0))
    factory.set_output("pip", MockProcessOutput(stdout="Successfully installed\n", returncode=0))
    
    return factory


# =============================================================================
# FILE SYSTEM FIXTURES
# =============================================================================

@pytest.fixture
def mock_file_system() -> MockFileSystem:
    """Create a mock file system."""
    return create_mock_file_system()


@pytest.fixture
def populated_file_system() -> MockFileSystem:
    """Create a file system with sample project files."""
    fs = create_mock_file_system()
    
    # Create project structure
    fs.write_file("/project/src/main.py", "# Main module\n")
    fs.write_file("/project/src/utils.py", "# Utilities\n")
    fs.write_file("/project/tests/test_main.py", "# Tests\n")
    fs.write_file("/project/README.md", "# Project\n")
    fs.write_file("/project/requirements.txt", "pytest>=7.0\n")
    
    return fs


# =============================================================================
# GIT FIXTURES
# =============================================================================

@pytest.fixture
def mock_git_repo() -> MockGitRepository:
    """Create a mock git repository."""
    return create_mock_git_repo()


@pytest.fixture
def git_repo_with_history() -> MockGitRepository:
    """Create a git repository with commit history."""
    repo = create_mock_git_repo()
    
    repo.add_commit("Initial commit", ["README.md"])
    repo.add_commit("Add main module", ["src/main.py"])
    repo.add_commit("Add tests", ["tests/test_main.py"])
    
    repo.create_branch("develop")
    repo.create_branch("feature/new-feature")
    
    return repo


# =============================================================================
# TELEMETRY FIXTURES
# =============================================================================

@pytest.fixture
def mock_telemetry() -> MockTelemetry:
    """Create a mock telemetry instance."""
    return create_mock_telemetry()


@pytest.fixture
def telemetry_with_data() -> MockTelemetry:
    """Create telemetry with pre-recorded data."""
    telemetry = create_mock_telemetry()
    
    # Record some events
    telemetry.record_event("session_start", {"session_id": "test-1"})
    telemetry.record_event("command_executed", {"command": "echo", "exit_code": 0})
    telemetry.record_event("llm_request", {"model": "gpt-4", "tokens": 100})
    
    # Record some metrics
    telemetry.record_metric("latency_ms", 150.0)
    telemetry.record_metric("memory_mb", 256.0)
    
    return telemetry


# =============================================================================
# CONSENT FIXTURES
# =============================================================================

@pytest.fixture
def mock_consent_manager() -> MockConsentManager:
    """Create a mock consent manager that auto-approves."""
    return create_mock_consent_manager(auto_approve=True)


@pytest.fixture
def strict_consent_manager() -> MockConsentManager:
    """Create a consent manager that denies all requests."""
    return create_mock_consent_manager(auto_approve=False)


# =============================================================================
# COMBINED FIXTURES
# =============================================================================

@pytest.fixture
def agent_components() -> Dict[str, Any]:
    """Create all mock agent components."""
    return {
        "llm": create_mock_llm_client(),
        "subprocess": create_mock_subprocess_factory(),
        "fs": create_mock_file_system(),
        "git": create_mock_git_repo(),
        "telemetry": create_mock_telemetry(),
        "consent": create_mock_consent_manager(),
    }


@pytest.fixture
def integration_setup(
    mock_llm_client,
    configured_subprocess_factory,
    populated_file_system,
    git_repo_with_history,
    mock_telemetry,
    mock_consent_manager,
) -> Dict[str, Any]:
    """Create a fully configured integration test setup."""
    return {
        "llm": mock_llm_client,
        "subprocess": configured_subprocess_factory,
        "fs": populated_file_system,
        "git": git_repo_with_history,
        "telemetry": mock_telemetry,
        "consent": mock_consent_manager,
    }


# =============================================================================
# ASYNC HELPERS
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
