"""LRET Variant Installer Module.

Provides functionality to install, verify, and manage LRET backend variants:
1. cirq_scalability (cirq-scalability-comparison branch)
2. pennylane_hybrid (pennylane-documentation-benchmarking branch)
3. phase7_unified (phase-7 branch)

Each variant is cloned from the LRET GitHub repository and built locally.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from proxima.backends.lret.config import (
    LRETConfig,
    LRETVariantType,
    get_lret_config,
    save_lret_config,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# LRET Variant Definitions
# ==============================================================================

LRET_REPO_URL = "https://github.com/kunal5556/LRET.git"

LRET_VARIANTS: dict[str, dict[str, Any]] = {
    "cirq_scalability": {
        "repo": LRET_REPO_URL,
        "branch": "cirq-scalability-comparison",
        "display_name": "Cirq Scalability Comparison",
        "description": "LRET vs Cirq FDM performance benchmarking and comparison tools",
        "requires": [
            "cirq-core>=1.0.0",
            "pandas>=1.3",
            "matplotlib>=3.5",
        ],
        "optional": [
            "numpy>=1.21",
        ],
        "python_package_dir": "python",
        "has_cpp_build": True,
        "features": [
            "Cirq FDM comparison",
            "Scalability benchmarks",
            "CSV export",
            "Performance visualization",
        ],
    },
    "pennylane_hybrid": {
        "repo": LRET_REPO_URL,
        "branch": "pennylane-documentation-benchmarking",
        "display_name": "PennyLane Hybrid",
        "description": "PennyLane device plugin for VQE, QAOA, and gradient-based optimization",
        "requires": [
            "pennylane>=0.33.0",
        ],
        "optional": [
            "jax>=0.4.0",
            "torch>=2.0",
            "pennylane-cirq>=0.33.0",
        ],
        "python_package_dir": "python",
        "has_cpp_build": True,
        "features": [
            "QLRETDevice",
            "VQE algorithm",
            "QAOA algorithm",
            "Gradient computation",
            "Noise models",
        ],
    },
    "phase7_unified": {
        "repo": LRET_REPO_URL,
        "branch": "phase-7",
        "display_name": "Phase 7 Unified",
        "description": "Multi-framework integration with Cirq, PennyLane, and Qiskit support",
        "requires": [
            "cirq-core>=1.0.0",
            "pennylane>=0.33.0",
            "qiskit>=0.45.0",
        ],
        "optional": [
            "qiskit-aer>=0.13.0",
            "cuquantum-python>=23.0",
        ],
        "python_package_dir": "python",
        "has_cpp_build": True,
        "features": [
            "Multi-framework execution",
            "Gate fusion",
            "GPU acceleration",
            "Cross-platform support",
            "Unified API",
        ],
    },
}


class InstallationStatus(str, Enum):
    """Status of variant installation."""
    
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    INSTALL_FAILED = "install_failed"
    PARTIALLY_INSTALLED = "partially_installed"
    OUTDATED = "outdated"


@dataclass
class InstallationResult:
    """Result of an installation operation.
    
    Attributes:
        success: Whether installation succeeded
        variant: Variant name
        status: Installation status
        message: Status message
        install_path: Path where variant was installed
        version: Installed version (if available)
        errors: List of error messages
        warnings: List of warning messages
    """
    
    success: bool
    variant: str
    status: InstallationStatus
    message: str
    install_path: Optional[Path] = None
    version: Optional[str] = None
    errors: list[str] = None
    warnings: list[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class VariantStatus:
    """Status information for an LRET variant.
    
    Attributes:
        variant: Variant name
        installed: Whether the variant is installed
        enabled: Whether the variant is enabled in config
        functional: Whether the variant is fully functional
        version: Installed version
        install_path: Installation path
        python_importable: Whether Python package can be imported
        cpp_built: Whether C++ components are built
        missing_deps: List of missing Python dependencies
    """
    
    variant: str
    installed: bool = False
    enabled: bool = False
    functional: bool = False
    version: Optional[str] = None
    install_path: Optional[Path] = None
    python_importable: bool = False
    cpp_built: bool = False
    missing_deps: list[str] = None
    
    def __post_init__(self):
        if self.missing_deps is None:
            self.missing_deps = []


# ==============================================================================
# Installation Functions
# ==============================================================================

def get_variant_install_path(variant_name: str, config: Optional[LRETConfig] = None) -> Path:
    """Get the installation path for a variant.
    
    Args:
        variant_name: Name of the variant
        config: Optional LRETConfig. Uses global config if not provided.
        
    Returns:
        Path where the variant should be installed
    """
    if config is None:
        config = get_lret_config()
    
    variant_config = config.get_variant_config(variant_name)
    if variant_config.install_path:
        return Path(variant_config.install_path)
    
    return Path(config.install_base_dir) / variant_name


def _run_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    capture_output: bool = True,
    timeout: int = 600,
) -> tuple[bool, str, str]:
    """Run a shell command.
    
    Args:
        cmd: Command and arguments
        cwd: Working directory
        capture_output: Whether to capture stdout/stderr
        timeout: Command timeout in seconds
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except FileNotFoundError as e:
        return False, "", f"Command not found: {e}"
    except Exception as e:
        return False, "", f"Command failed: {e}"


def _check_git_available() -> bool:
    """Check if git is available on the system."""
    success, _, _ = _run_command(["git", "--version"])
    return success


def _check_cmake_available() -> bool:
    """Check if CMake is available on the system."""
    success, _, _ = _run_command(["cmake", "--version"])
    return success


def _check_python_package_installed(package: str) -> bool:
    """Check if a Python package is installed.
    
    Args:
        package: Package name (can include version specifier)
        
    Returns:
        True if package is installed
    """
    # Extract package name from version specifier
    package_name = package.split(">=")[0].split("==")[0].split("<")[0].strip()
    
    try:
        import importlib.util
        spec = importlib.util.find_spec(package_name.replace("-", "_"))
        return spec is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _install_python_dependencies(
    requirements: list[str],
    optional: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[bool, list[str]]:
    """Install Python dependencies.
    
    Args:
        requirements: List of package requirements
        optional: Whether these are optional dependencies
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (success, list_of_failed_packages)
    """
    failed = []
    total = len(requirements)
    
    for i, req in enumerate(requirements):
        if progress_callback:
            progress_callback(f"Installing {req}...", (i / total) * 100)
        
        package_name = req.split(">=")[0].split("==")[0].split("<")[0].strip()
        
        if _check_python_package_installed(package_name):
            logger.debug(f"Package {package_name} already installed")
            continue
        
        success, stdout, stderr = _run_command([
            sys.executable, "-m", "pip", "install", req, "--quiet"
        ])
        
        if not success:
            if optional:
                logger.warning(f"Failed to install optional package {req}: {stderr}")
            else:
                logger.error(f"Failed to install required package {req}: {stderr}")
                failed.append(req)
    
    return len(failed) == 0, failed


def _clone_repository(
    repo_url: str,
    branch: str,
    target_dir: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[bool, str]:
    """Clone a git repository.
    
    Args:
        repo_url: Repository URL
        branch: Branch to checkout
        target_dir: Target directory
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (success, error_message)
    """
    if progress_callback:
        progress_callback(f"Cloning {branch} branch...", 10)
    
    # Remove existing directory if present
    if target_dir.exists():
        logger.info(f"Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)
    
    # Create parent directory
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Clone repository
    success, stdout, stderr = _run_command([
        "git", "clone",
        "--branch", branch,
        "--depth", "1",
        "--single-branch",
        repo_url,
        str(target_dir),
    ], timeout=300)
    
    if not success:
        return False, f"Git clone failed: {stderr}"
    
    if progress_callback:
        progress_callback("Repository cloned", 30)
    
    return True, ""


def _build_cpp_components(
    source_dir: Path,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[bool, str]:
    """Build C++ components using CMake.
    
    Args:
        source_dir: Source directory containing CMakeLists.txt
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (success, error_message)
    """
    cmake_file = source_dir / "CMakeLists.txt"
    if not cmake_file.exists():
        logger.info("No CMakeLists.txt found, skipping C++ build")
        return True, ""
    
    if not _check_cmake_available():
        return False, "CMake not found. Please install CMake to build C++ components."
    
    if progress_callback:
        progress_callback("Configuring CMake...", 50)
    
    build_dir = source_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Configure
    success, stdout, stderr = _run_command([
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ], timeout=300)
    
    if not success:
        return False, f"CMake configure failed: {stderr}"
    
    if progress_callback:
        progress_callback("Building C++ components...", 65)
    
    # Build
    success, stdout, stderr = _run_command([
        "cmake", "--build", str(build_dir),
        "--parallel",
        "--config", "Release",
    ], timeout=600)
    
    if not success:
        return False, f"CMake build failed: {stderr}"
    
    if progress_callback:
        progress_callback("C++ build complete", 80)
    
    return True, ""


def _install_python_package(
    source_dir: Path,
    package_dir: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[bool, str]:
    """Install Python package from source.
    
    Args:
        source_dir: Root source directory
        package_dir: Subdirectory containing setup.py/pyproject.toml
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (success, error_message)
    """
    package_path = source_dir / package_dir
    
    if not package_path.exists():
        # Try installing from root if package_dir doesn't exist
        package_path = source_dir
    
    setup_py = package_path / "setup.py"
    pyproject = package_path / "pyproject.toml"
    
    if not setup_py.exists() and not pyproject.exists():
        logger.warning(f"No setup.py or pyproject.toml found in {package_path}")
        return True, ""  # Not an error, just no Python package
    
    if progress_callback:
        progress_callback("Installing Python package...", 85)
    
    success, stdout, stderr = _run_command([
        sys.executable, "-m", "pip", "install", "-e", str(package_path), "--quiet"
    ], timeout=300)
    
    if not success:
        return False, f"Python package installation failed: {stderr}"
    
    if progress_callback:
        progress_callback("Python package installed", 95)
    
    return True, ""


def install_lret_variant(
    variant_name: str,
    config: Optional[LRETConfig] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    skip_optional_deps: bool = False,
) -> InstallationResult:
    """Install a specific LRET variant.
    
    Args:
        variant_name: Name of the variant to install
        config: Optional LRETConfig. Uses global config if not provided.
        progress_callback: Optional callback for progress updates (message, percentage)
        skip_optional_deps: Whether to skip optional dependencies
        
    Returns:
        InstallationResult with installation status
    """
    if variant_name not in LRET_VARIANTS:
        return InstallationResult(
            success=False,
            variant=variant_name,
            status=InstallationStatus.INSTALL_FAILED,
            message=f"Unknown variant: {variant_name}",
            errors=[f"Variant '{variant_name}' not found. Available: {list(LRET_VARIANTS.keys())}"],
        )
    
    if config is None:
        config = get_lret_config()
    
    variant_info = LRET_VARIANTS[variant_name]
    install_path = get_variant_install_path(variant_name, config)
    
    logger.info(f"Installing LRET variant: {variant_name}")
    logger.info(f"Branch: {variant_info['branch']}")
    logger.info(f"Install path: {install_path}")
    
    errors = []
    warnings = []
    
    # Check prerequisites
    if not _check_git_available():
        return InstallationResult(
            success=False,
            variant=variant_name,
            status=InstallationStatus.INSTALL_FAILED,
            message="Git is not available",
            errors=["Git is required but not found. Please install Git."],
        )
    
    if progress_callback:
        progress_callback("Checking prerequisites...", 5)
    
    # Install required Python dependencies
    if progress_callback:
        progress_callback("Installing dependencies...", 10)
    
    success, failed = _install_python_dependencies(
        variant_info["requires"],
        optional=False,
        progress_callback=progress_callback,
    )
    
    if not success:
        errors.extend([f"Failed to install: {pkg}" for pkg in failed])
        return InstallationResult(
            success=False,
            variant=variant_name,
            status=InstallationStatus.INSTALL_FAILED,
            message="Failed to install required dependencies",
            errors=errors,
        )
    
    # Install optional dependencies
    if not skip_optional_deps and variant_info.get("optional"):
        _, optional_failed = _install_python_dependencies(
            variant_info["optional"],
            optional=True,
            progress_callback=progress_callback,
        )
        if optional_failed:
            warnings.extend([f"Optional package not installed: {pkg}" for pkg in optional_failed])
    
    # Clone repository
    success, error = _clone_repository(
        variant_info["repo"],
        variant_info["branch"],
        install_path,
        progress_callback=progress_callback,
    )
    
    if not success:
        errors.append(error)
        return InstallationResult(
            success=False,
            variant=variant_name,
            status=InstallationStatus.INSTALL_FAILED,
            message="Failed to clone repository",
            install_path=install_path,
            errors=errors,
            warnings=warnings,
        )
    
    # Build C++ components if needed
    if variant_info.get("has_cpp_build", False):
        success, error = _build_cpp_components(install_path, progress_callback)
        if not success:
            warnings.append(f"C++ build warning: {error}")
            logger.warning(f"C++ build failed for {variant_name}: {error}")
    
    # Install Python package
    success, error = _install_python_package(
        install_path,
        variant_info.get("python_package_dir", "python"),
        progress_callback=progress_callback,
    )
    
    if not success:
        errors.append(error)
        return InstallationResult(
            success=False,
            variant=variant_name,
            status=InstallationStatus.PARTIALLY_INSTALLED,
            message="Python package installation failed",
            install_path=install_path,
            errors=errors,
            warnings=warnings,
        )
    
    # Update configuration
    variant_config = config.get_variant_config(variant_name)
    variant_config.install_path = str(install_path)
    variant_config.enabled = True
    save_lret_config(config)
    
    if progress_callback:
        progress_callback("Installation complete!", 100)
    
    logger.info(f"✓ {variant_name} installed successfully at {install_path}")
    
    return InstallationResult(
        success=True,
        variant=variant_name,
        status=InstallationStatus.INSTALLED,
        message=f"Successfully installed {variant_info['display_name']}",
        install_path=install_path,
        warnings=warnings,
    )


def uninstall_lret_variant(
    variant_name: str,
    config: Optional[LRETConfig] = None,
    remove_files: bool = True,
) -> InstallationResult:
    """Uninstall an LRET variant.
    
    Args:
        variant_name: Name of the variant to uninstall
        config: Optional LRETConfig
        remove_files: Whether to remove installation files
        
    Returns:
        InstallationResult with uninstallation status
    """
    if config is None:
        config = get_lret_config()
    
    install_path = get_variant_install_path(variant_name, config)
    
    if remove_files and install_path.exists():
        try:
            shutil.rmtree(install_path)
            logger.info(f"Removed installation directory: {install_path}")
        except Exception as e:
            return InstallationResult(
                success=False,
                variant=variant_name,
                status=InstallationStatus.INSTALL_FAILED,
                message=f"Failed to remove files: {e}",
                errors=[str(e)],
            )
    
    # Update configuration
    variant_config = config.get_variant_config(variant_name)
    variant_config.enabled = False
    save_lret_config(config)
    
    return InstallationResult(
        success=True,
        variant=variant_name,
        status=InstallationStatus.NOT_INSTALLED,
        message=f"Successfully uninstalled {variant_name}",
    )


def check_variant_availability(variant_name: str) -> VariantStatus:
    """Check if an LRET variant is available and operational.
    
    Args:
        variant_name: Name of the variant to check
        
    Returns:
        VariantStatus with availability information
    """
    config = get_lret_config()
    variant_config = config.get_variant_config(variant_name)
    install_path = get_variant_install_path(variant_name, config)
    
    status = VariantStatus(
        variant=variant_name,
        enabled=variant_config.enabled,
        install_path=install_path if install_path.exists() else None,
    )
    
    # Check if installation directory exists
    if not install_path.exists():
        return status
    
    status.installed = True
    
    # Check if Python package is importable
    try:
        if variant_name == "cirq_scalability":
            # Try importing LRET cirq comparison module
            try:
                import lret
                status.python_importable = True
                status.version = getattr(lret, "__version__", "unknown")
            except ImportError:
                pass
        elif variant_name == "pennylane_hybrid":
            # Try importing QLRET device
            try:
                import qlret
                status.python_importable = True
                status.version = getattr(qlret, "__version__", "unknown")
            except ImportError:
                try:
                    import lret
                    status.python_importable = True
                    status.version = getattr(lret, "__version__", "unknown")
                except ImportError:
                    pass
        elif variant_name == "phase7_unified":
            # Try importing phase7 module
            try:
                import lret
                status.python_importable = True
                status.version = getattr(lret, "__version__", "unknown")
            except ImportError:
                pass
    except Exception as e:
        logger.debug(f"Import check failed for {variant_name}: {e}")
    
    # Check for C++ build artifacts
    build_dir = install_path / "build"
    if build_dir.exists():
        # Check for common build outputs
        for lib_pattern in ["*.so", "*.pyd", "*.dll", "*.dylib"]:
            if list(build_dir.glob(f"**/{lib_pattern}")):
                status.cpp_built = True
                break
    
    # Check for missing required dependencies
    if variant_name in LRET_VARIANTS:
        for req in LRET_VARIANTS[variant_name]["requires"]:
            package_name = req.split(">=")[0].split("==")[0].split("<")[0].strip()
            if not _check_python_package_installed(package_name):
                status.missing_deps.append(req)
    
    # Determine if fully functional
    status.functional = (
        status.installed and
        status.python_importable and
        len(status.missing_deps) == 0
    )
    
    return status


def verify_variant_installation(variant_name: str) -> tuple[bool, str]:
    """Verify that a variant installation is complete and functional.
    
    Args:
        variant_name: Name of the variant to verify
        
    Returns:
        Tuple of (is_valid, message)
    """
    status = check_variant_availability(variant_name)
    
    if not status.installed:
        return False, f"Variant {variant_name} is not installed"
    
    if not status.python_importable:
        return False, f"Python package for {variant_name} cannot be imported"
    
    if status.missing_deps:
        return False, f"Missing dependencies: {', '.join(status.missing_deps)}"
    
    if not status.functional:
        return False, f"Variant {variant_name} is not fully functional"
    
    return True, f"Variant {variant_name} is properly installed and functional"


def list_installed_variants() -> list[VariantStatus]:
    """List all installed LRET variants with their status.
    
    Returns:
        List of VariantStatus for all variants
    """
    statuses = []
    for variant_name in LRET_VARIANTS:
        status = check_variant_availability(variant_name)
        statuses.append(status)
    return statuses


def get_variant_info(variant_name: str) -> Optional[dict[str, Any]]:
    """Get information about a specific variant.
    
    Args:
        variant_name: Name of the variant
        
    Returns:
        Variant info dict or None if not found
    """
    return LRET_VARIANTS.get(variant_name)


def get_all_variants_info() -> dict[str, dict[str, Any]]:
    """Get information about all available variants.
    
    Returns:
        Dictionary of variant name to info
    """
    return LRET_VARIANTS.copy()
