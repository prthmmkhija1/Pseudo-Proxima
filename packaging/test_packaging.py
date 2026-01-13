#!/usr/bin/env python3
"""Packaging Test Script for Proxima Agent.

This script validates the packaging process by:
1. Building the package
2. Installing it in a temporary environment
3. Running basic functionality tests
4. Validating binary distribution

Usage:
    python test_packaging.py [--skip-binary] [--skip-pypi] [--verbose]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import NamedTuple


class TestResult(NamedTuple):
    """Result of a packaging test."""

    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0


class PackagingTester:
    """Test harness for packaging validation."""

    def __init__(self, project_root: Path, verbose: bool = False) -> None:
        """Initialize the tester.

        Args:
            project_root: Root directory of the project.
            verbose: Whether to show verbose output.
        """
        self.project_root = project_root
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.temp_dir: Path | None = None

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message.

        Args:
            message: The message to log.
            level: Log level (INFO, WARN, ERROR).
        """
        prefix = {"INFO": "ℹ️ ", "WARN": "⚠️ ", "ERROR": "❌", "SUCCESS": "✅"}
        print(f"{prefix.get(level, '')} {message}")

    def run_command(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Run a command and capture output.

        Args:
            cmd: Command to run.
            cwd: Working directory.
            env: Environment variables.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        if self.verbose:
            self.log(f"Running: {' '.join(cmd)}")

        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        result = subprocess.run(
            cmd,
            cwd=cwd or self.project_root,
            capture_output=True,
            text=True,
            env=process_env,
        )

        if self.verbose and result.stdout:
            print(result.stdout[:500])

        return result.returncode, result.stdout, result.stderr

    def add_result(
        self,
        name: str,
        passed: bool,
        message: str,
        duration_ms: float = 0.0,
    ) -> None:
        """Add a test result.

        Args:
            name: Test name.
            passed: Whether the test passed.
            message: Result message.
            duration_ms: Test duration in milliseconds.
        """
        result = TestResult(name, passed, message, duration_ms)
        self.results.append(result)

        level = "SUCCESS" if passed else "ERROR"
        self.log(f"{name}: {message}", level)

    # =========================================================================
    # Individual Tests
    # =========================================================================

    def test_pyproject_toml(self) -> bool:
        """Test that pyproject.toml is valid."""
        import time
        import tomllib

        start = time.time()

        pyproject_path = self.project_root / "pyproject.toml"

        if not pyproject_path.exists():
            self.add_result(
                "pyproject.toml exists",
                False,
                "pyproject.toml not found",
            )
            return False

        try:
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)

            # Check required fields
            required = [
                ("project", "name"),
                ("project", "version"),
                ("project", "description"),
                ("build-system", "requires"),
            ]

            missing = []
            for section, key in required:
                if section not in config or key not in config[section]:
                    missing.append(f"{section}.{key}")

            if missing:
                self.add_result(
                    "pyproject.toml valid",
                    False,
                    f"Missing fields: {missing}",
                    (time.time() - start) * 1000,
                )
                return False

            self.add_result(
                "pyproject.toml valid",
                True,
                f"Project: {config['project']['name']} v{config['project']['version']}",
                (time.time() - start) * 1000,
            )
            return True

        except Exception as e:
            self.add_result(
                "pyproject.toml valid",
                False,
                f"Parse error: {e}",
                (time.time() - start) * 1000,
            )
            return False

    def test_build_sdist(self) -> bool:
        """Test building source distribution."""
        import time

        start = time.time()

        # Clean previous builds
        dist_dir = self.project_root / "dist"
        if dist_dir.exists():
            shutil.rmtree(dist_dir)

        # Build sdist
        returncode, stdout, stderr = self.run_command(
            [sys.executable, "-m", "build", "--sdist"],
        )

        duration = (time.time() - start) * 1000

        if returncode != 0:
            self.add_result(
                "Build sdist",
                False,
                f"Build failed: {stderr[:200]}",
                duration,
            )
            return False

        # Check sdist was created
        sdist_files = list(dist_dir.glob("*.tar.gz"))
        if not sdist_files:
            self.add_result(
                "Build sdist",
                False,
                "No sdist file created",
                duration,
            )
            return False

        self.add_result(
            "Build sdist",
            True,
            f"Created: {sdist_files[0].name}",
            duration,
        )
        return True

    def test_build_wheel(self) -> bool:
        """Test building wheel distribution."""
        import time

        start = time.time()

        # Build wheel
        returncode, stdout, stderr = self.run_command(
            [sys.executable, "-m", "build", "--wheel"],
        )

        duration = (time.time() - start) * 1000

        if returncode != 0:
            self.add_result(
                "Build wheel",
                False,
                f"Build failed: {stderr[:200]}",
                duration,
            )
            return False

        # Check wheel was created
        dist_dir = self.project_root / "dist"
        wheel_files = list(dist_dir.glob("*.whl"))
        if not wheel_files:
            self.add_result(
                "Build wheel",
                False,
                "No wheel file created",
                duration,
            )
            return False

        self.add_result(
            "Build wheel",
            True,
            f"Created: {wheel_files[0].name}",
            duration,
        )
        return True

    def test_install_in_venv(self) -> bool:
        """Test installing package in a virtual environment."""
        import time

        start = time.time()

        # Create temp directory for venv
        self.temp_dir = Path(tempfile.mkdtemp(prefix="proxima_test_"))
        venv_path = self.temp_dir / "venv"

        try:
            # Create virtual environment
            venv.create(venv_path, with_pip=True)

            # Determine pip path
            if sys.platform == "win32":
                pip_path = venv_path / "Scripts" / "pip.exe"
                python_path = venv_path / "Scripts" / "python.exe"
            else:
                pip_path = venv_path / "bin" / "pip"
                python_path = venv_path / "bin" / "python"

            # Find wheel
            dist_dir = self.project_root / "dist"
            wheel_files = list(dist_dir.glob("*.whl"))
            if not wheel_files:
                self.add_result(
                    "Install in venv",
                    False,
                    "No wheel file found",
                    (time.time() - start) * 1000,
                )
                return False

            # Install wheel
            returncode, stdout, stderr = self.run_command(
                [str(pip_path), "install", str(wheel_files[0])],
            )

            if returncode != 0:
                self.add_result(
                    "Install in venv",
                    False,
                    f"Install failed: {stderr[:200]}",
                    (time.time() - start) * 1000,
                )
                return False

            # Verify import works
            returncode, stdout, stderr = self.run_command(
                [str(python_path), "-c", "import proxima; print(proxima.__version__)"],
            )

            duration = (time.time() - start) * 1000

            if returncode != 0:
                self.add_result(
                    "Install in venv",
                    False,
                    f"Import failed: {stderr[:200]}",
                    duration,
                )
                return False

            version = stdout.strip()
            self.add_result(
                "Install in venv",
                True,
                f"Installed version: {version}",
                duration,
            )
            return True

        except Exception as e:
            self.add_result(
                "Install in venv",
                False,
                f"Error: {e}",
                (time.time() - start) * 1000,
            )
            return False

    def test_cli_entrypoint(self) -> bool:
        """Test CLI entrypoint works."""
        import time

        start = time.time()

        if self.temp_dir is None:
            self.add_result(
                "CLI entrypoint",
                False,
                "No venv available",
            )
            return False

        venv_path = self.temp_dir / "venv"
        if sys.platform == "win32":
            cli_path = venv_path / "Scripts" / "proxima.exe"
        else:
            cli_path = venv_path / "bin" / "proxima"

        if not cli_path.exists():
            # Try without .exe
            cli_path = cli_path.with_suffix("")

        # Test version command
        returncode, stdout, stderr = self.run_command(
            [str(cli_path), "--version"],
        )

        duration = (time.time() - start) * 1000

        if returncode != 0:
            self.add_result(
                "CLI entrypoint",
                False,
                f"CLI failed: {stderr[:200]}",
                duration,
            )
            return False

        self.add_result(
            "CLI entrypoint",
            True,
            f"Version: {stdout.strip()}",
            duration,
        )
        return True

    def test_cli_help(self) -> bool:
        """Test CLI help command."""
        import time

        start = time.time()

        if self.temp_dir is None:
            self.add_result(
                "CLI help",
                False,
                "No venv available",
            )
            return False

        venv_path = self.temp_dir / "venv"
        if sys.platform == "win32":
            cli_path = venv_path / "Scripts" / "proxima.exe"
        else:
            cli_path = venv_path / "bin" / "proxima"

        returncode, stdout, stderr = self.run_command(
            [str(cli_path), "--help"],
        )

        duration = (time.time() - start) * 1000

        if returncode != 0:
            self.add_result(
                "CLI help",
                False,
                f"Help failed: {stderr[:200]}",
                duration,
            )
            return False

        # Check for expected commands
        expected_commands = ["run", "compare", "config"]
        found = sum(1 for cmd in expected_commands if cmd in stdout)

        self.add_result(
            "CLI help",
            True,
            f"Found {found}/{len(expected_commands)} expected commands",
            duration,
        )
        return True

    def test_binary_build(self) -> bool:
        """Test PyInstaller binary build."""
        import time

        start = time.time()

        spec_file = self.project_root / "packaging" / "proxima.spec"
        if not spec_file.exists():
            self.add_result(
                "Binary build",
                False,
                "proxima.spec not found",
            )
            return False

        # Check PyInstaller is available
        returncode, stdout, stderr = self.run_command(
            [sys.executable, "-m", "PyInstaller", "--version"],
        )

        if returncode != 0:
            self.add_result(
                "Binary build",
                False,
                "PyInstaller not installed",
                (time.time() - start) * 1000,
            )
            return False

        # Build binary (this can take a while)
        returncode, stdout, stderr = self.run_command(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--clean",
                "--noconfirm",
                str(spec_file),
            ],
            cwd=self.project_root / "packaging",
        )

        duration = (time.time() - start) * 1000

        if returncode != 0:
            self.add_result(
                "Binary build",
                False,
                f"Build failed: {stderr[:200]}",
                duration,
            )
            return False

        # Check binary was created
        dist_path = self.project_root / "packaging" / "dist"
        if sys.platform == "win32":
            binary = dist_path / "proxima" / "proxima.exe"
        else:
            binary = dist_path / "proxima" / "proxima"

        if not binary.exists():
            self.add_result(
                "Binary build",
                False,
                f"Binary not found at {binary}",
                duration,
            )
            return False

        # Get binary size
        size_mb = binary.stat().st_size / (1024 * 1024)

        self.add_result(
            "Binary build",
            True,
            f"Binary created: {size_mb:.1f} MB",
            duration,
        )
        return True

    def test_binary_runs(self) -> bool:
        """Test that the binary runs correctly."""
        import time

        start = time.time()

        dist_path = self.project_root / "packaging" / "dist"
        if sys.platform == "win32":
            binary = dist_path / "proxima" / "proxima.exe"
        else:
            binary = dist_path / "proxima" / "proxima"

        if not binary.exists():
            self.add_result(
                "Binary runs",
                False,
                "Binary not found",
            )
            return False

        # Test version command
        returncode, stdout, stderr = self.run_command(
            [str(binary), "--version"],
        )

        duration = (time.time() - start) * 1000

        if returncode != 0:
            self.add_result(
                "Binary runs",
                False,
                f"Binary failed: {stderr[:200]}",
                duration,
            )
            return False

        self.add_result(
            "Binary runs",
            True,
            f"Version: {stdout.strip()}",
            duration,
        )
        return True

    # =========================================================================
    # Test Runner
    # =========================================================================

    def run_tests(
        self,
        skip_binary: bool = False,
        skip_pypi: bool = False,
    ) -> bool:
        """Run all packaging tests.

        Args:
            skip_binary: Skip binary build tests.
            skip_pypi: Skip PyPI upload tests.

        Returns:
            True if all tests passed.
        """
        self.log("=" * 60)
        self.log("Proxima Packaging Test Suite")
        self.log("=" * 60)
        self.log("")

        try:
            # Basic validation
            self.log("Phase 1: Project Validation")
            self.log("-" * 40)
            self.test_pyproject_toml()

            # Build tests
            self.log("")
            self.log("Phase 2: Build Distribution")
            self.log("-" * 40)
            if not self.test_build_sdist():
                self.log("Skipping wheel build due to sdist failure", "WARN")
            else:
                self.test_build_wheel()

            # Installation tests
            self.log("")
            self.log("Phase 3: Installation Verification")
            self.log("-" * 40)
            if self.test_install_in_venv():
                self.test_cli_entrypoint()
                self.test_cli_help()

            # Binary tests
            if not skip_binary:
                self.log("")
                self.log("Phase 4: Binary Distribution")
                self.log("-" * 40)
                if self.test_binary_build():
                    self.test_binary_runs()

        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception:
                    pass

        # Summary
        self.log("")
        self.log("=" * 60)
        self.log("Test Summary")
        self.log("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.name}: {result.message}")

        self.log("")
        self.log(f"Results: {passed} passed, {failed} failed")

        return failed == 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Proxima packaging")
    parser.add_argument("--skip-binary", action="store_true", help="Skip binary tests")
    parser.add_argument("--skip-pypi", action="store_true", help="Skip PyPI tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    tester = PackagingTester(project_root, verbose=args.verbose)

    success = tester.run_tests(
        skip_binary=args.skip_binary,
        skip_pypi=args.skip_pypi,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
