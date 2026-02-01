"""Agent test package.

Phase 10: Integration & Testing

This package contains unit tests for all agent modules:
- terminal_executor - Terminal command execution
- session_manager - Session state management
- tools - Agent tool definitions and execution
- safety - Consent and safety boundaries
- git_operations - Git repository operations
- backend_modifier - Backend code modification
- multi_terminal_monitor - Multi-terminal session management
- agent_controller - Main controller integration
- telemetry - Metrics and telemetry collection
"""

from pathlib import Path

# Package root
AGENT_TESTS_ROOT = Path(__file__).parent

# Test data directory
TEST_DATA_DIR = AGENT_TESTS_ROOT / "data"
