"""Agent test fixtures package.

Phase 10: Integration & Testing

Provides reusable fixtures for testing agent modules.
"""

from .mock_agent import (
    # Response types
    MockLLMResponse,
    MockProcessOutput,
    
    # Mock classes
    MockLLMClient,
    MockAsyncProcess,
    MockSubprocessFactory,
    MockFileSystem,
    MockGitRepository,
    MockTelemetry,
    MockConsentManager,
    
    # Factory functions
    create_mock_llm_client,
    create_mock_subprocess_factory,
    create_mock_file_system,
    create_mock_git_repo,
    create_mock_telemetry,
    create_mock_consent_manager,
)

__all__ = [
    # Response types
    "MockLLMResponse",
    "MockProcessOutput",
    
    # Mock classes
    "MockLLMClient",
    "MockAsyncProcess",
    "MockSubprocessFactory",
    "MockFileSystem",
    "MockGitRepository",
    "MockTelemetry",
    "MockConsentManager",
    
    # Factory functions
    "create_mock_llm_client",
    "create_mock_subprocess_factory",
    "create_mock_file_system",
    "create_mock_git_repo",
    "create_mock_telemetry",
    "create_mock_consent_manager",
]
