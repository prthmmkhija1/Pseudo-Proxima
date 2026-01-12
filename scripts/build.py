#!/usr/bin/env python3
"""
Proxima Agent - Build and Release Script
=========================================

This script provides utilities for building, testing, and releasing Proxima Agent.

Usage:
    python scripts/build.py [command] [options]

Commands:
    build       Build the Python package
    test        Run the test suite
    lint        Run linting and formatting checks
    typecheck   Run type checking
    docker      Build Docker image
    release     Prepare a release
    clean       Clean build artifacts
    all         Run all checks and build

Examples:
    python scripts/build.py build
    python scripts/build.py test --coverage
    python scripts/build.py release --version 0.1.0
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
DOCS_DIR = PROJECT_ROOT / "docs"


# =============================================================================
# Utility Functions
# =============================================================================


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def ensure_venv():
    """Ensure we're running in a virtual environment."""
    if sys.prefix == sys.base_prefix:
        print("Warning: Not running in a virtual environment!")


# =============================================================================
# Build Commands
# =============================================================================


def cmd_clean():
    """Clean build artifacts."""
    print("\n=== Cleaning build artifacts ===\n")

    dirs_to_remove = [
        DIST_DIR,
        BUILD_DIR,
        PROJECT_ROOT / "*.egg-info",
        PROJECT_ROOT / ".pytest_cache",
        PROJECT_ROOT / ".mypy_cache",
        PROJECT_ROOT / ".ruff_cache",
        PROJECT_ROOT / "htmlcov",
        PROJECT_ROOT / ".coverage",
    ]

    for pattern in dirs_to_remove:
        for path in PROJECT_ROOT.glob(
            str(pattern.name) if pattern.parent == PROJECT_ROOT else str(pattern)
        ):
            if path.exists():
                print(f"Removing: {path}")
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    # Clean __pycache__ directories
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        print(f"Removing: {pycache}")
        shutil.rmtree(pycache)

    print("Clean complete!")


def cmd_lint():
    """Run linting and formatting checks."""
    print("\n=== Running linting ===\n")

    # Run ruff
    print("\n--- Ruff ---")
    run_command(["ruff", "check", "src/", "tests/"])

    # Run black
    print("\n--- Black ---")
    run_command(["black", "--check", "src/", "tests/"])

    print("\nLinting complete!")


def cmd_format():
    """Format code with black and ruff."""
    print("\n=== Formatting code ===\n")

    # Run black
    print("\n--- Black ---")
    run_command(["black", "src/", "tests/"])

    # Run ruff fix
    print("\n--- Ruff fix ---")
    run_command(["ruff", "check", "--fix", "src/", "tests/"])

    print("\nFormatting complete!")


def cmd_typecheck():
    """Run type checking with mypy."""
    print("\n=== Running type checking ===\n")

    run_command(["mypy", "src/", "--ignore-missing-imports"])

    print("\nType checking complete!")


def cmd_test(coverage: bool = False, verbose: bool = False):
    """Run the test suite."""
    print("\n=== Running tests ===\n")

    cmd = ["pytest", "tests/"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=proxima", "--cov-report=term-missing", "--cov-report=html"])

    run_command(cmd)

    print("\nTests complete!")


def cmd_build():
    """Build the Python package."""
    print("\n=== Building package ===\n")

    # Clean first
    cmd_clean()

    # Install build dependencies
    run_command([sys.executable, "-m", "pip", "install", "build", "twine"])

    # Build
    run_command([sys.executable, "-m", "build"])

    # Check
    print("\n--- Checking package ---")
    run_command(["twine", "check", "dist/*"])

    print(f"\nBuild complete! Packages in: {DIST_DIR}")


def cmd_docker(tag: str = "latest", push: bool = False):
    """Build Docker image."""
    print("\n=== Building Docker image ===\n")

    image_name = f"proxima-agent:{tag}"

    run_command(["docker", "build", "-t", image_name, "--target", "runtime", "."])

    print(f"\nDocker image built: {image_name}")

    if push:
        print("\n--- Pushing image ---")
        run_command(["docker", "push", image_name])


def cmd_docs():
    """Build documentation."""
    print("\n=== Building documentation ===\n")

    run_command(["mkdocs", "build", "--strict"])

    print(f"\nDocumentation built in: {PROJECT_ROOT / 'site'}")


def cmd_release(version: str, dry_run: bool = True):
    """Prepare a release."""
    print(f"\n=== Preparing release v{version} ===\n")

    if dry_run:
        print("DRY RUN - No changes will be made\n")

    # 1. Run all checks
    print("Step 1: Running quality checks...")
    cmd_lint()
    cmd_typecheck()
    cmd_test(coverage=True)

    # 2. Update version in pyproject.toml
    print(f"\nStep 2: Updating version to {version}...")
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    content = pyproject_path.read_text()

    import re

    new_content = re.sub(r'version = "[^"]*"', f'version = "{version}"', content)

    if not dry_run:
        pyproject_path.write_text(new_content)
        print("Updated pyproject.toml")
    else:
        print("Would update pyproject.toml")

    # 3. Build package
    print("\nStep 3: Building package...")
    if not dry_run:
        cmd_build()
    else:
        print("Would build package")

    # 4. Create git tag
    print(f"\nStep 4: Creating git tag v{version}...")
    if not dry_run:
        run_command(["git", "add", "pyproject.toml"])
        run_command(["git", "commit", "-m", f"Release v{version}"])
        run_command(["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"])
        print(f"Created tag v{version}")
        print("\nTo push the release:")
        print("  git push origin main")
        print(f"  git push origin v{version}")
    else:
        print(f"Would create tag v{version}")

    print("\nRelease preparation complete!")


def cmd_all():
    """Run all checks and build."""
    print("\n=== Running all ===\n")

    cmd_clean()
    cmd_lint()
    cmd_typecheck()
    cmd_test(coverage=True)
    cmd_build()
    cmd_docs()

    print("\nAll complete!")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Proxima Agent build and release script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # clean
    subparsers.add_parser("clean", help="Clean build artifacts")

    # lint
    subparsers.add_parser("lint", help="Run linting checks")

    # format
    subparsers.add_parser("format", help="Format code")

    # typecheck
    subparsers.add_parser("typecheck", help="Run type checking")

    # test
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--coverage", action="store_true", help="Enable coverage")
    test_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    # build
    subparsers.add_parser("build", help="Build the package")

    # docker
    docker_parser = subparsers.add_parser("docker", help="Build Docker image")
    docker_parser.add_argument("--tag", default="latest", help="Image tag")
    docker_parser.add_argument("--push", action="store_true", help="Push image")

    # docs
    subparsers.add_parser("docs", help="Build documentation")

    # release
    release_parser = subparsers.add_parser("release", help="Prepare a release")
    release_parser.add_argument("--version", required=True, help="Version number")
    release_parser.add_argument(
        "--no-dry-run", action="store_true", help="Actually make changes"
    )

    # all
    subparsers.add_parser("all", help="Run all checks and build")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    ensure_venv()
    os.chdir(PROJECT_ROOT)

    try:
        if args.command == "clean":
            cmd_clean()
        elif args.command == "lint":
            cmd_lint()
        elif args.command == "format":
            cmd_format()
        elif args.command == "typecheck":
            cmd_typecheck()
        elif args.command == "test":
            cmd_test(coverage=args.coverage, verbose=args.verbose)
        elif args.command == "build":
            cmd_build()
        elif args.command == "docker":
            cmd_docker(tag=args.tag, push=args.push)
        elif args.command == "docs":
            cmd_docs()
        elif args.command == "release":
            cmd_release(version=args.version, dry_run=not args.no_dry_run)
        elif args.command == "all":
            cmd_all()
        else:
            parser.print_help()
            return 1
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command failed with exit code {e.returncode}")
        return e.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
