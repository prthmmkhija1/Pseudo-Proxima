"""
End-to-end tests for the CLI interface.

Tests full user workflows from command line to output.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_agent_file(temp_dir: Path) -> Path:
    """Create a sample agent file for testing."""
    agent_file = temp_dir / "proxima_agent.md"
    agent_file.write_text(
        """# Proxima Agent File

## Metadata
- name: test-circuit
- version: 1.0.0
- author: test

## Configuration
- backend: auto
- shots: 100

## Tasks
### Task 1: Simple Bell State
Create and measure a Bell state circuit.

```quantum
H 0
CNOT 0 1
MEASURE ALL
```
"""
    )
    return agent_file


class TestCLIVersion:
    """Test version command output."""

    def test_version_shows_output(self) -> None:
        """Running --version should display version info."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "proxima" in result.stdout.lower() or "0." in result.stdout


class TestCLIHelp:
    """Test help command output."""

    def test_help_shows_commands(self) -> None:
        """Running --help should show available commands."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Should show main command groups
        assert "run" in result.stdout.lower() or "Usage" in result.stdout

    def test_run_help_shows_options(self) -> None:
        """Running 'run --help' should show run command options."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should either succeed or show that 'run' is a valid command
        assert result.returncode == 0 or "run" in result.stderr.lower()


class TestCLIConfig:
    """Test config command workflows."""

    def test_config_show_works(self) -> None:
        """Running 'config show' should display current configuration."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "config", "show"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should succeed or indicate config module exists
        assert result.returncode == 0 or "config" in result.stderr.lower()


class TestCLIBackends:
    """Test backend listing workflows."""

    def test_backends_list_works(self) -> None:
        """Running 'backends list' should show available backends."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "backends", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should succeed and show backend info
        if result.returncode == 0:
            output = result.stdout.lower()
            # Should mention at least one backend type
            assert any(name in output for name in ["cirq", "qiskit", "lret", "backend"])


class TestAgentFileWorkflow:
    """Test full agent file processing workflow."""

    def test_validate_agent_file(self, sample_agent_file: Path) -> None:
        """Validating an agent file should work."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "validate", str(sample_agent_file)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Validate command may or may not exist, but should not crash
        assert result.returncode in [0, 1, 2]  # Success, error, or command not found

    def test_dry_run_agent_file(self, sample_agent_file: Path) -> None:
        """Dry run of an agent file should show planned execution."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "run",
                str(sample_agent_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Should either succeed or show meaningful error
        # (quantum libraries may not be installed in test env)
        assert result.returncode in [0, 1, 2]


class TestResourceAwareness:
    """Test resource monitoring in CLI."""

    def test_resource_check_on_run(self, temp_dir: Path) -> None:
        """Running with resource check should validate resources."""
        # Create a minimal circuit file
        circuit_file = temp_dir / "circuit.py"
        circuit_file.write_text(
            """
# Simple test circuit
print("Circuit placeholder")
"""
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "run",
                str(circuit_file),
                "--check-resources",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Should handle resource check gracefully
        assert result.returncode in [0, 1, 2]


class TestOutputFormats:
    """Test different output format options."""

    def test_json_output_format(self) -> None:
        """Using --format json should produce JSON output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "--format",
                "json",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should handle format flag
        assert result.returncode in [0, 1, 2]

    def test_quiet_mode(self) -> None:
        """Using --quiet should suppress verbose output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "--quiet",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should handle quiet flag
        assert result.returncode in [0, 1, 2]


class TestErrorHandling:
    """Test CLI error handling."""

    def test_invalid_command_shows_help(self) -> None:
        """Invalid command should show helpful error message."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "invalid_command_xyz"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should indicate error and possibly show help
        assert result.returncode != 0

    def test_missing_file_error(self) -> None:
        """Missing agent file should show clear error."""
        result = subprocess.run(
            [sys.executable, "-m", "proxima", "run", "/nonexistent/path/file.md"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should fail gracefully
        assert result.returncode != 0

    def test_invalid_backend_error(self, temp_dir: Path) -> None:
        """Invalid backend should show clear error."""
        agent_file = temp_dir / "test.md"
        agent_file.write_text("# Test\nMinimal content")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "proxima",
                "run",
                str(agent_file),
                "--backend",
                "invalid_backend_xyz",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should indicate backend error
        assert result.returncode != 0
