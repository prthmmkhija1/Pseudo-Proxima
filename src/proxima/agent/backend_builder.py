"""Backend Builder for Proxima Agent.

Phase 4: Backend Building & Compilation System

Provides backend building capabilities:
- Load build profiles from YAML config
- Pre-build validation (dependencies, GPU requirements)
- Dependency installation
- Compilation with progress tracking
- Post-build verification
- Artifact management
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import subprocess
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from proxima.utils.logging import get_logger
from proxima.agent.gpu_detector import (
    GPUDetector,
    GPUEnvironment,
    GPUCapability,
    get_gpu_detector,
)
from proxima.agent.build_artifact_manager import (
    BuildArtifactManager,
    BuildManifest,
    ArtifactType,
    ArtifactInfo,
    generate_build_id,
)
from proxima.agent.build_progress_tracker import (
    BuildProgressTracker,
    BuildProgress,
    BuildStep,
    BuildPhase,
    BuildStepStatus,
    ProgressCallback,
)

logger = get_logger("agent.backend_builder")


class BuildStatus(Enum):
    """Status of a build."""
    PENDING = "pending"
    VALIDATING = "validating"
    INSTALLING_DEPS = "installing_dependencies"
    BUILDING = "building"
    TESTING = "testing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DependencyCheck:
    """Result of a dependency check."""
    name: str
    required_version: Optional[str]
    installed_version: Optional[str]
    is_installed: bool
    is_compatible: bool
    
    @property
    def satisfied(self) -> bool:
        """Check if dependency is satisfied."""
        return self.is_installed and self.is_compatible


@dataclass
class ValidationResult:
    """Result of build validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_dependencies: List[str] = field(default_factory=list)
    gpu_available: bool = False
    gpu_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_dependencies": self.missing_dependencies,
            "gpu_available": self.gpu_available,
            "gpu_info": self.gpu_info,
        }


@dataclass
class BuildStepResult:
    """Result of a build step."""
    step_id: str
    success: bool
    output: str
    error: Optional[str] = None
    duration_seconds: float = 0.0
    return_code: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "output": self.output[:1000] if self.output else "",
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "return_code": self.return_code,
        }


@dataclass
class BuildResult:
    """Result of a complete build."""
    backend_name: str
    success: bool
    status: BuildStatus
    build_id: str
    build_dir: Optional[Path]
    validation: Optional[ValidationResult]
    step_results: List[BuildStepResult] = field(default_factory=list)
    artifacts: List[ArtifactInfo] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get total build duration."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_name": self.backend_name,
            "success": self.success,
            "status": self.status.value,
            "build_id": self.build_id,
            "build_dir": str(self.build_dir) if self.build_dir else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "step_results": [s.to_dict() for s in self.step_results],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


class BuildProfileLoader:
    """Load and manage build profiles from YAML configuration."""
    
    DEFAULT_CONFIG_PATH = Path("configs/backend_build_profiles.yaml")
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the loader.
        
        Args:
            config_path: Path to build profiles YAML (default: configs/backend_build_profiles.yaml)
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Optional[Dict[str, Any]] = None
    
    def load(self) -> Dict[str, Any]:
        """Load the configuration.
        
        Returns:
            Configuration dictionary
        """
        if self._config is not None:
            return self._config
        
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            self._config = self._get_default_config()
            return self._config
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded build profiles from {self.config_path}")
            return self._config
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config: {e}")
            self._config = self._get_default_config()
            return self._config
    
    def get_backend_config(self, backend_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific backend.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            Backend configuration or None
        """
        config = self.load()
        backends = config.get("backends", {})
        return backends.get(backend_name)
    
    def get_build_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific build profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Profile configuration or None
        """
        config = self.load()
        profiles = config.get("build_profiles", {})
        return profiles.get(profile_name)
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Get global settings.
        
        Returns:
            Global settings dictionary
        """
        config = self.load()
        return config.get("global_settings", {})
    
    def get_error_patterns(self) -> List[Dict[str, Any]]:
        """Get error detection patterns.
        
        Returns:
            List of error pattern configurations
        """
        config = self.load()
        return config.get("error_patterns", [])
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backends.
        
        Returns:
            List of backend names
        """
        config = self.load()
        return list(config.get("backends", {}).keys())
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "global_settings": {
                "default_timeout": 600,
                "retry_count": 1,
                "artifacts_dir": "build",
                "max_artifact_versions": 3,
            },
            "backends": {
                "cirq": {
                    "name": "Cirq",
                    "dependencies": ["cirq-core>=1.0.0"],
                    "build_steps": [
                        {"step_id": "install", "command": "pip install cirq-core", "description": "Install Cirq"},
                    ],
                    "verification": {"import_check": "import cirq"},
                },
                "qiskit": {
                    "name": "Qiskit",
                    "dependencies": ["qiskit>=0.45.0"],
                    "build_steps": [
                        {"step_id": "install", "command": "pip install qiskit qiskit-aer", "description": "Install Qiskit"},
                    ],
                    "verification": {"import_check": "import qiskit"},
                },
            },
            "build_profiles": {},
            "error_patterns": [],
        }
    
    def reload(self) -> Dict[str, Any]:
        """Force reload configuration.
        
        Returns:
            Configuration dictionary
        """
        self._config = None
        return self.load()


class BackendBuilder:
    """Build backends with progress tracking and artifact management.
    
    Example:
        >>> builder = BackendBuilder()
        >>> 
        >>> # Register progress callback
        >>> builder.on_progress(lambda p: print(f"{p.overall_percent:.1f}%"))
        >>> 
        >>> # Build a backend
        >>> result = await builder.build("qsim")
        >>> 
        >>> if result.success:
        ...     print(f"Build completed in {result.duration_seconds:.1f}s")
        ... else:
        ...     print(f"Build failed: {result.error_message}")
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
        gpu_detector: Optional[GPUDetector] = None,
    ):
        """Initialize the builder.
        
        Args:
            config_path: Path to build profiles YAML
            artifacts_dir: Directory for build artifacts
            gpu_detector: GPU detector instance
        """
        self.profile_loader = BuildProfileLoader(config_path)
        self.gpu_detector = gpu_detector or get_gpu_detector()
        
        # Get global settings
        settings = self.profile_loader.get_global_settings()
        artifacts_base = artifacts_dir or Path(settings.get("artifacts_dir", "build"))
        max_versions = settings.get("max_artifact_versions", 3)
        
        self.artifact_manager = BuildArtifactManager(artifacts_base, max_versions)
        
        self.progress_callbacks: List[ProgressCallback] = []
        self._current_build: Optional[BuildResult] = None
        self._cancelled = False
    
    def on_progress(self, callback: ProgressCallback) -> None:
        """Register a progress callback.
        
        Args:
            callback: Function called with BuildProgress
        """
        self.progress_callbacks.append(callback)
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backends.
        
        Returns:
            List of backend names
        """
        return self.profile_loader.get_available_backends()
    
    def get_backend_info(self, backend_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a backend.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            Backend configuration or None
        """
        return self.profile_loader.get_backend_config(backend_name)
    
    async def validate_backend(self, backend_name: str) -> ValidationResult:
        """Validate that a backend can be built.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)
        
        # Get backend config
        config = self.profile_loader.get_backend_config(backend_name)
        if config is None:
            result.valid = False
            result.errors.append(f"Unknown backend: {backend_name}")
            return result
        
        # Check platform compatibility
        supported_platforms = config.get("platform", ["all"])
        if supported_platforms != ["all"]:
            current_platform = platform.system().lower()
            if current_platform not in [p.lower() for p in supported_platforms]:
                result.valid = False
                result.errors.append(
                    f"Backend {backend_name} not supported on {current_platform}"
                )
        
        # Check GPU requirements
        if config.get("gpu_required", False):
            gpu_env = self.gpu_detector.detect()
            result.gpu_available = gpu_env.has_gpu
            result.gpu_info = gpu_env.to_dict()
            
            if not gpu_env.has_gpu:
                result.valid = False
                result.errors.append("GPU required but not detected")
            
            # Check CUDA version if specified
            cuda_version = config.get("cuda_version")
            if cuda_version and gpu_env.cuda_available:
                if not self.gpu_detector.check_cuda_compatibility(cuda_version):
                    result.warnings.append(
                        f"CUDA {cuda_version}+ recommended, found {gpu_env.cuda_version}"
                    )
        
        # Check dependencies
        dependencies = config.get("dependencies", [])
        for dep in dependencies:
            if not self._check_dependency(dep):
                result.missing_dependencies.append(dep)
        
        if result.missing_dependencies:
            result.warnings.append(
                f"Missing dependencies will be installed: {', '.join(result.missing_dependencies)}"
            )
        
        # Check build tools
        build_tools = config.get("build_tools", [])
        for tool in build_tools:
            if not shutil.which(tool):
                result.warnings.append(f"Build tool not found: {tool}")
        
        return result
    
    def _check_dependency(self, dependency: str) -> bool:
        """Check if a dependency is installed.
        
        Args:
            dependency: Dependency specification (e.g., "numpy>=1.20.0")
            
        Returns:
            True if installed
        """
        # Parse dependency name
        match = re.match(r"([a-zA-Z0-9_-]+)", dependency)
        if not match:
            return False
        
        pkg_name = match.group(1)
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", pkg_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def build(
        self,
        backend_name: str,
        profile: Optional[str] = None,
        skip_validation: bool = False,
        skip_deps: bool = False,
        environment: Optional[Dict[str, str]] = None,
    ) -> BuildResult:
        """Build a backend.
        
        Args:
            backend_name: Name of the backend
            profile: Build profile name (e.g., "minimal", "full", "gpu")
            skip_validation: Skip pre-build validation
            skip_deps: Skip dependency installation
            environment: Additional environment variables
            
        Returns:
            BuildResult
        """
        self._cancelled = False
        timestamp = datetime.now()
        build_id = generate_build_id(backend_name, timestamp)
        
        result = BuildResult(
            backend_name=backend_name,
            success=False,
            status=BuildStatus.PENDING,
            build_id=build_id,
            build_dir=None,
            validation=None,
            started_at=timestamp,
        )
        self._current_build = result
        
        # Get backend config
        config = self.profile_loader.get_backend_config(backend_name)
        if config is None:
            result.status = BuildStatus.FAILED
            result.error_message = f"Unknown backend: {backend_name}"
            result.completed_at = datetime.now()
            return result
        
        # Apply profile overrides
        if profile:
            profile_config = self.profile_loader.get_build_profile(profile)
            if profile_config:
                config = self._merge_configs(config, profile_config)
        
        # Create progress tracker
        tracker = self._create_progress_tracker(backend_name, config)
        
        try:
            # Phase 1: Validation
            if not skip_validation:
                result.status = BuildStatus.VALIDATING
                tracker.start()
                tracker.start_step("validation")
                
                validation = await self.validate_backend(backend_name)
                result.validation = validation
                
                if not validation.valid:
                    tracker.complete_step("validation", success=False)
                    result.status = BuildStatus.FAILED
                    result.error_message = "; ".join(validation.errors)
                    result.completed_at = datetime.now()
                    return result
                
                tracker.complete_step("validation")
            else:
                tracker.start()
                tracker.skip_step("validation")
            
            # Create build directory
            build_dir = self.artifact_manager.create_build_directory(backend_name, timestamp)
            result.build_dir = build_dir
            
            # Phase 2: Install dependencies
            if not skip_deps:
                result.status = BuildStatus.INSTALLING_DEPS
                tracker.start_step("dependencies")
                
                dep_result = await self._install_dependencies(config, build_dir, environment)
                result.step_results.append(dep_result)
                
                if not dep_result.success:
                    tracker.complete_step("dependencies", success=False)
                    result.status = BuildStatus.FAILED
                    result.error_message = dep_result.error
                    result.completed_at = datetime.now()
                    return result
                
                tracker.complete_step("dependencies")
            else:
                tracker.skip_step("dependencies")
            
            # Phase 3: Execute build steps
            result.status = BuildStatus.BUILDING
            build_steps = config.get("build_steps", [])
            
            for step_config in build_steps:
                if self._cancelled:
                    result.status = BuildStatus.CANCELLED
                    result.error_message = "Build cancelled"
                    break
                
                step_id = step_config.get("step_id", "build")
                tracker.start_step(step_id)
                
                step_result = await self._execute_build_step(
                    step_config, build_dir, environment, tracker
                )
                result.step_results.append(step_result)
                
                if not step_result.success:
                    tracker.complete_step(step_id, success=False, error=step_result.error)
                    result.status = BuildStatus.FAILED
                    result.error_message = step_result.error
                    result.completed_at = datetime.now()
                    return result
                
                tracker.complete_step(step_id)
            
            # Phase 4: Verification
            result.status = BuildStatus.VERIFYING
            tracker.start_step("verification")
            
            verification_config = config.get("verification", {})
            verify_result = await self._verify_build(verification_config, build_dir)
            result.step_results.append(verify_result)
            
            if not verify_result.success:
                tracker.complete_step("verification", success=False)
                result.status = BuildStatus.FAILED
                result.error_message = verify_result.error
                result.completed_at = datetime.now()
                return result
            
            tracker.complete_step("verification")
            
            # Success! Create manifest
            result.status = BuildStatus.COMPLETED
            result.success = True
            result.completed_at = datetime.now()
            
            # Register artifacts and save manifest
            artifacts = await self._collect_artifacts(build_dir, config)
            result.artifacts = artifacts
            
            manifest = BuildManifest(
                build_id=build_id,
                backend_name=backend_name,
                version=config.get("version", "1.0.0"),
                timestamp=timestamp,
                duration_seconds=result.duration_seconds,
                success=True,
                artifacts=artifacts,
                environment=environment or {},
                build_config=config,
            )
            self.artifact_manager.save_manifest(build_dir, manifest)
            
            # Cleanup old builds
            self.artifact_manager.cleanup_old_builds(backend_name)
            
            tracker.complete(success=True)
            
        except Exception as e:
            logger.exception(f"Build failed with exception: {e}")
            result.status = BuildStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            tracker.complete(success=False)
        
        finally:
            self._current_build = None
        
        return result
    
    def _create_progress_tracker(
        self,
        backend_name: str,
        config: Dict[str, Any],
    ) -> BuildProgressTracker:
        """Create a progress tracker for the build.
        
        Args:
            backend_name: Name of the backend
            config: Backend configuration
            
        Returns:
            Configured BuildProgressTracker
        """
        tracker = BuildProgressTracker(backend_name)
        
        # Add standard steps
        tracker.add_step(
            "validation", "Validation", BuildPhase.INITIALIZATION,
            "Pre-build validation", weight=0.5
        )
        tracker.add_step(
            "dependencies", "Dependencies", BuildPhase.DEPENDENCY_INSTALL,
            "Install dependencies", weight=1.0
        )
        
        # Add build steps from config
        build_steps = config.get("build_steps", [])
        for i, step in enumerate(build_steps):
            step_id = step.get("step_id", f"build_{i}")
            name = step.get("description", step.get("command", "Build")[:30])
            weight = step.get("weight", 2.0)
            
            tracker.add_step(
                step_id, name, BuildPhase.COMPILATION,
                step.get("description", ""), weight=weight
            )
        
        tracker.add_step(
            "verification", "Verification", BuildPhase.VERIFICATION,
            "Post-build verification", weight=0.5
        )
        
        # Register callbacks
        for callback in self.progress_callbacks:
            tracker.on_progress(callback)
        
        return tracker
    
    async def _install_dependencies(
        self,
        config: Dict[str, Any],
        build_dir: Path,
        environment: Optional[Dict[str, str]],
    ) -> BuildStepResult:
        """Install build dependencies.
        
        Args:
            config: Backend configuration
            build_dir: Build directory
            environment: Environment variables
            
        Returns:
            BuildStepResult
        """
        dependencies = config.get("dependencies", [])
        
        if not dependencies:
            return BuildStepResult(
                step_id="dependencies",
                success=True,
                output="No dependencies to install",
            )
        
        # Build pip install command
        deps_str = " ".join(f'"{d}"' for d in dependencies)
        command = f"{sys.executable} -m pip install {deps_str}"
        
        start_time = time.time()
        
        try:
            env = os.environ.copy()
            if environment:
                env.update(environment)
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                cwd=str(build_dir),
            )
            
            duration = time.time() - start_time
            
            return BuildStepResult(
                step_id="dependencies",
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                duration_seconds=duration,
                return_code=result.returncode,
            )
            
        except subprocess.TimeoutExpired:
            return BuildStepResult(
                step_id="dependencies",
                success=False,
                output="",
                error="Dependency installation timed out",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return BuildStepResult(
                step_id="dependencies",
                success=False,
                output="",
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
    
    async def _execute_build_step(
        self,
        step_config: Dict[str, Any],
        build_dir: Path,
        environment: Optional[Dict[str, str]],
        tracker: BuildProgressTracker,
    ) -> BuildStepResult:
        """Execute a single build step.
        
        Args:
            step_config: Step configuration
            build_dir: Build directory
            environment: Environment variables
            tracker: Progress tracker
            
        Returns:
            BuildStepResult
        """
        step_id = step_config.get("step_id", "build")
        command = step_config.get("command", "")
        timeout = step_config.get("timeout", 600)
        working_dir = step_config.get("working_dir", str(build_dir))
        retry = step_config.get("retry", 0)
        
        if not command:
            return BuildStepResult(
                step_id=step_id,
                success=True,
                output="No command specified",
            )
        
        # Expand variables in command
        command = self._expand_variables(command, build_dir, environment)
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(retry + 1):
            if self._cancelled:
                return BuildStepResult(
                    step_id=step_id,
                    success=False,
                    output="",
                    error="Build cancelled",
                )
            
            try:
                env = os.environ.copy()
                if environment:
                    env.update(environment)
                
                # Add GPU environment if available
                gpu_env = self.gpu_detector.detect()
                if gpu_env.cuda_home:
                    env["CUDA_HOME"] = gpu_env.cuda_home
                    env["PATH"] = f"{gpu_env.cuda_home}/bin:{env.get('PATH', '')}"
                
                # Execute with real-time output
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=working_dir,
                )
                
                output_lines = []
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        output_lines.append(line)
                        # Try to parse progress
                        tracker.parse_output(line, step_id)
                
                process.wait(timeout=timeout)
                
                duration = time.time() - start_time
                output = "".join(output_lines)
                
                if process.returncode == 0:
                    return BuildStepResult(
                        step_id=step_id,
                        success=True,
                        output=output,
                        duration_seconds=duration,
                        return_code=process.returncode,
                    )
                else:
                    last_error = self._parse_error(output)
                    if attempt < retry:
                        logger.info(f"Retrying step {step_id} (attempt {attempt + 2})")
                        await asyncio.sleep(2)
                    
            except subprocess.TimeoutExpired:
                last_error = f"Step timed out after {timeout} seconds"
            except Exception as e:
                last_error = str(e)
        
        return BuildStepResult(
            step_id=step_id,
            success=False,
            output=output if 'output' in dir() else "",
            error=last_error,
            duration_seconds=time.time() - start_time,
        )
    
    async def _verify_build(
        self,
        verification_config: Dict[str, Any],
        build_dir: Path,
    ) -> BuildStepResult:
        """Verify the build completed successfully.
        
        Args:
            verification_config: Verification configuration
            build_dir: Build directory
            
        Returns:
            BuildStepResult
        """
        start_time = time.time()
        
        # Check import
        import_check = verification_config.get("import_check")
        if import_check:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", import_check],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                
                if result.returncode != 0:
                    return BuildStepResult(
                        step_id="verification",
                        success=False,
                        output=result.stdout,
                        error=f"Import check failed: {result.stderr}",
                        duration_seconds=time.time() - start_time,
                    )
            except Exception as e:
                return BuildStepResult(
                    step_id="verification",
                    success=False,
                    output="",
                    error=f"Import check error: {e}",
                    duration_seconds=time.time() - start_time,
                )
        
        # Check command
        check_command = verification_config.get("check_command")
        if check_command:
            try:
                result = subprocess.run(
                    check_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(build_dir),
                )
                
                if result.returncode != 0:
                    return BuildStepResult(
                        step_id="verification",
                        success=False,
                        output=result.stdout,
                        error=f"Check command failed: {result.stderr}",
                        duration_seconds=time.time() - start_time,
                    )
            except Exception as e:
                return BuildStepResult(
                    step_id="verification",
                    success=False,
                    output="",
                    error=f"Check command error: {e}",
                    duration_seconds=time.time() - start_time,
                )
        
        # Check required files
        required_files = verification_config.get("required_files", [])
        for file_pattern in required_files:
            matches = list(build_dir.glob(file_pattern))
            if not matches:
                return BuildStepResult(
                    step_id="verification",
                    success=False,
                    output="",
                    error=f"Required file not found: {file_pattern}",
                    duration_seconds=time.time() - start_time,
                )
        
        return BuildStepResult(
            step_id="verification",
            success=True,
            output="All verification checks passed",
            duration_seconds=time.time() - start_time,
        )
    
    async def _collect_artifacts(
        self,
        build_dir: Path,
        config: Dict[str, Any],
    ) -> List[ArtifactInfo]:
        """Collect build artifacts.
        
        Args:
            build_dir: Build directory
            config: Backend configuration
            
        Returns:
            List of ArtifactInfo
        """
        artifacts = []
        
        # Collect binaries
        for pattern in ["*.so", "*.dll", "*.pyd", "*.dylib"]:
            for path in build_dir.rglob(pattern):
                try:
                    artifact = self.artifact_manager.register_artifact(
                        build_dir, path, ArtifactType.BINARY
                    )
                    artifacts.append(artifact)
                except Exception as e:
                    logger.warning(f"Failed to register artifact {path}: {e}")
        
        # Collect logs
        logs_dir = build_dir / "logs"
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.log"):
                try:
                    artifact = self.artifact_manager.register_artifact(
                        build_dir, log_file, ArtifactType.LOG
                    )
                    artifacts.append(artifact)
                except Exception as e:
                    logger.warning(f"Failed to register log {log_file}: {e}")
        
        return artifacts
    
    def _expand_variables(
        self,
        command: str,
        build_dir: Path,
        environment: Optional[Dict[str, str]],
    ) -> str:
        """Expand variables in a command string.
        
        Args:
            command: Command string
            build_dir: Build directory
            environment: Environment variables
            
        Returns:
            Expanded command
        """
        # Standard variables
        variables = {
            "${BUILD_DIR}": str(build_dir),
            "${PYTHON}": sys.executable,
            "${NPROC}": str(os.cpu_count() or 4),
        }
        
        # GPU variables
        gpu_env = self.gpu_detector.detect()
        if gpu_env.cuda_home:
            variables["${CUDA_HOME}"] = gpu_env.cuda_home
        
        # Environment variables
        if environment:
            for key, value in environment.items():
                variables[f"${{{key}}}"] = value
        
        # Replace variables
        for var, value in variables.items():
            command = command.replace(var, value)
        
        return command
    
    def _parse_error(self, output: str) -> str:
        """Parse build output for error messages.
        
        Args:
            output: Build output
            
        Returns:
            Extracted error message
        """
        error_patterns = self.profile_loader.get_error_patterns()
        
        for pattern_config in error_patterns:
            pattern = pattern_config.get("pattern", "")
            try:
                match = re.search(pattern, output, re.MULTILINE | re.IGNORECASE)
                if match:
                    suggestion = pattern_config.get("suggestion", "")
                    error_type = pattern_config.get("type", "unknown")
                    return f"[{error_type}] {match.group(0)}. {suggestion}"
            except re.error:
                continue
        
        # Return last few lines if no pattern matched
        lines = output.strip().split("\n")
        return "\n".join(lines[-5:]) if lines else "Unknown error"
    
    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge two configurations.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
        
        return result
    
    def cancel(self) -> None:
        """Cancel the current build."""
        self._cancelled = True
        logger.info("Build cancellation requested")
    
    @property
    def is_building(self) -> bool:
        """Check if a build is in progress."""
        return self._current_build is not None
    
    def get_current_build(self) -> Optional[BuildResult]:
        """Get the current build result (if building)."""
        return self._current_build


# Convenience functions

def get_backend_builder() -> BackendBuilder:
    """Get a BackendBuilder instance."""
    return BackendBuilder()


async def build_backend(
    backend_name: str,
    profile: Optional[str] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> BuildResult:
    """Convenience function to build a backend.
    
    Args:
        backend_name: Name of the backend
        profile: Build profile
        on_progress: Progress callback
        
    Returns:
        BuildResult
    """
    builder = BackendBuilder()
    if on_progress:
        builder.on_progress(on_progress)
    return await builder.build(backend_name, profile)
