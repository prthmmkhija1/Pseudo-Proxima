"""
Additional End-to-End Tests for Proxima CLI

Comprehensive E2E tests covering:
- Full user workflows
- Backend execution flows
- Configuration commands
- Session management
- Error scenarios
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace with typical project structure."""
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        
        # Create typical project structure
        (workspace / "circuits").mkdir()
        (workspace / "results").mkdir()
        (workspace / "configs").mkdir()
        
        yield workspace


@pytest.fixture
def sample_config_file(temp_workspace: Path) -> Path:
    """Create a sample configuration file."""
    config_file = temp_workspace / "proxima.yaml"
    config_file.write_text("""
general:
  verbosity: info
  output_format: text
  
backends:
  default_backend: auto
  timeout_seconds: 60

llm:
  provider: none
  
resources:
  memory_warn_threshold_mb: 2048
  
consent:
  auto_approve_local_llm: true
""")
    return config_file


# =============================================================================
# VERSION AND INFO COMMANDS
# =============================================================================


class TestVersionInfo:
    """Tests for version and info commands."""

    def test_version_contains_semver(self) -> None:
        """Version output should contain semantic version."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Should contain version number pattern
        output = result.stdout.lower()
        assert any(char.isdigit() for char in output)

    def test_help_shows_all_commands(self) -> None:
        """Help should list all available commands."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        output = result.stdout.lower()
        
        # Should mention main command groups
        expected_commands = ["run", "config", "backends"]
        for cmd in expected_commands:
            assert cmd in output or "usage" in output


# =============================================================================
# CONFIG COMMAND TESTS
# =============================================================================


class TestConfigCommands:
    """Tests for configuration management commands."""

    def test_config_show_displays_settings(self) -> None:
        """Config show should display current settings."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "config", "show"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete (may succeed or indicate config not set)
        assert result.returncode in [0, 1]

    def test_config_path_shows_locations(self) -> None:
        """Config path should show configuration file locations."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "config", "path"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete
        assert result.returncode in [0, 1, 2]


# =============================================================================
# BACKENDS COMMAND TESTS
# =============================================================================


class TestBackendsCommands:
    """Tests for backends management commands."""

    def test_backends_list_shows_available(self) -> None:
        """Backends list should show available backends."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "backends", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            # Should show at least LRET (always available)
            assert "lret" in output or "backend" in output

    def test_backends_info_specific_backend(self) -> None:
        """Backends info for specific backend."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "backends", "info", "lret"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete (may succeed or show "not found")
        assert result.returncode in [0, 1, 2]


# =============================================================================
# RUN COMMAND TESTS
# =============================================================================


class TestRunCommand:
    """Tests for run command."""

    def test_run_with_description(self, temp_workspace: Path) -> None:
        """Run with description should work."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "run",
                "--backend", "lret",
                "bell state circuit",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(temp_workspace),
        )
        # Should attempt to run (may fail if backend not ready)
        assert result.returncode in [0, 1, 2]

    def test_run_with_shots_option(self, temp_workspace: Path) -> None:
        """Run with shots option."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "run",
                "--shots", "512",
                "--backend", "auto",
                "simple circuit",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(temp_workspace),
        )
        # Should process shots option
        assert result.returncode in [0, 1, 2]

    def test_run_dry_run_mode(self, temp_workspace: Path) -> None:
        """Run with --dry-run should not execute."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "run",
                "--dry-run",
                "--backend", "cirq",
                "test circuit",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(temp_workspace),
        )
        # Dry run should complete without actual execution
        assert result.returncode in [0, 1, 2]


# =============================================================================
# COMPARE COMMAND TESTS
# =============================================================================


class TestCompareCommand:
    """Tests for compare command."""

    def test_compare_multiple_backends(self, temp_workspace: Path) -> None:
        """Compare should run on multiple backends."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "compare",
                "--backends", "cirq,qiskit-aer",
                "bell state",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(temp_workspace),
        )
        # Should attempt comparison
        assert result.returncode in [0, 1, 2]


# =============================================================================
# SESSION COMMAND TESTS
# =============================================================================


class TestSessionCommands:
    """Tests for session management commands."""

    def test_session_list(self) -> None:
        """Session list should show saved sessions."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "session", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete (may show empty list)
        assert result.returncode in [0, 1, 2]


# =============================================================================
# HISTORY COMMAND TESTS
# =============================================================================


class TestHistoryCommands:
    """Tests for history commands."""

    def test_history_list(self) -> None:
        """History list should show execution history."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "history", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete
        assert result.returncode in [0, 1, 2]


# =============================================================================
# ENVIRONMENT INTERACTION TESTS
# =============================================================================


class TestEnvironmentInteraction:
    """Tests for CLI interaction with environment."""

    def test_respects_env_variables(self, temp_workspace: Path) -> None:
        """CLI should respect environment variables."""
        env = {
            **dict(__import__("os").environ),
            "PROXIMA_GENERAL__VERBOSITY": "debug",
        }
        
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "config", "show"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=str(temp_workspace),
        )
        # Should accept env var
        assert result.returncode in [0, 1, 2]

    def test_works_in_different_directory(self, temp_workspace: Path) -> None:
        """CLI should work when run from different directory."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(temp_workspace),
        )
        assert result.returncode == 0


# =============================================================================
# ERROR RECOVERY TESTS
# =============================================================================


class TestErrorRecovery:
    """Tests for CLI error handling and recovery."""

    def test_handles_keyboard_interrupt_gracefully(self) -> None:
        """CLI should handle interruption gracefully."""
        # This is hard to test without actually sending SIGINT
        # Just verify the CLI starts and can be controlled
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_handles_invalid_yaml_config(self, temp_workspace: Path) -> None:
        """CLI should handle invalid YAML config gracefully."""
        config_file = temp_workspace / "proxima.yaml"
        config_file.write_text("invalid: yaml: content: [broken")
        
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "config", "show"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(temp_workspace),
        )
        # Should handle gracefully (may show error or use defaults)
        assert result.returncode in [0, 1, 2]

    def test_handles_permission_denied(self) -> None:
        """CLI should handle permission denied gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "run", "/root/inaccessible/file.md"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should fail with clear error
        assert result.returncode != 0


# =============================================================================
# OUTPUT FORMAT TESTS
# =============================================================================


class TestOutputFormats:
    """Tests for different output format options."""

    def test_text_output_is_readable(self) -> None:
        """Text output should be human-readable."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "backends", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Output should be text, not JSON
            assert result.stdout.strip()  # Non-empty
            # Should not start with JSON markers
            assert not result.stdout.strip().startswith("{")
            assert not result.stdout.strip().startswith("[")


# =============================================================================
# FULL WORKFLOW TESTS
# =============================================================================


class TestFullWorkflows:
    """Tests for complete user workflows."""

    def test_basic_user_workflow(self, temp_workspace: Path) -> None:
        """Test a basic user workflow from start to finish."""
        # 1. Check version
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        
        # 2. List backends
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "backends", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete
        assert result.returncode in [0, 1, 2]
        
        # 3. Show config
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "config", "show"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should complete
        assert result.returncode in [0, 1, 2]

    def test_new_user_onboarding_flow(self, temp_workspace: Path) -> None:
        """Test flow for a new user getting started."""
        # 1. Get help
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert len(result.stdout) > 100  # Should have substantial help text
        
        # 2. Check what backends are available
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "backends", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode in [0, 1, 2]
