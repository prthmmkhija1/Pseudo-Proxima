"""Deployment Manager.

Coordinates the deployment of generated backend code.
Handles file writing, registry updates, and rollback on failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio
import shutil

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Stages of deployment process."""
    INITIALIZING = "initializing"
    VALIDATING = "validating"
    CREATING_BACKUP = "creating_backup"
    WRITING_FILES = "writing_files"
    UPDATING_REGISTRY = "updating_registry"
    RUNNING_VERIFICATION = "running_verification"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentStatus(Enum):
    """Status of deployment."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DeploymentProgress:
    """Progress information for deployment."""
    stage: DeploymentStage
    progress: float  # 0.0 to 1.0
    message: str
    current_file: Optional[str] = None
    files_written: int = 0
    total_files: int = 0


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool
    backend_name: str
    display_name: str
    created_files: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    deployment_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "backend_name": self.backend_name,
            "display_name": self.display_name,
            "created_files": self.created_files,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "deployment_time": self.deployment_time,
            "timestamp": self.timestamp.isoformat(),
        }


class DeploymentManager:
    """Manages the deployment of generated backend code.
    
    Provides:
    - Coordinated file writing with progress
    - Backup creation and rollback
    - Registry integration
    - Verification checks
    - Progress callbacks
    """
    
    def __init__(
        self,
        proxima_root: Optional[Path] = None,
        on_progress: Optional[Callable[[DeploymentProgress], None]] = None
    ):
        """Initialize deployment manager.
        
        Args:
            proxima_root: Root directory of Proxima project
            on_progress: Callback for progress updates
        """
        self.proxima_root = proxima_root or Path.cwd()
        self.backends_dir = self.proxima_root / "src" / "proxima" / "backends"
        self.contrib_dir = self.backends_dir / "contrib"
        self.tests_dir = self.proxima_root / "tests" / "backends"
        
        self._on_progress = on_progress
        self._backup_dir: Optional[Path] = None
        self._created_files: List[Path] = []
        self._cancelled = False
        self._current_stage = DeploymentStage.INITIALIZING
    
    def _report_progress(
        self,
        stage: DeploymentStage,
        progress: float,
        message: str,
        current_file: Optional[str] = None
    ) -> None:
        """Report progress to callback."""
        self._current_stage = stage
        
        if self._on_progress:
            self._on_progress(DeploymentProgress(
                stage=stage,
                progress=progress,
                message=message,
                current_file=current_file,
                files_written=len(self._created_files),
                total_files=self._total_files
            ))
    
    async def deploy_backend(
        self,
        backend_name: str,
        display_name: str,
        generated_code: Dict[str, str],
        run_verification: bool = True
    ) -> DeploymentResult:
        """Deploy a generated backend.
        
        Args:
            backend_name: Internal name of the backend
            display_name: User-friendly display name
            generated_code: Dictionary mapping file paths to content
            run_verification: Whether to run verification after deployment
            
        Returns:
            DeploymentResult with success status and details
        """
        import time
        start_time = time.time()
        
        self._total_files = len(generated_code)
        self._created_files = []
        self._cancelled = False
        warnings: List[str] = []
        
        try:
            # Stage 1: Initialization
            self._report_progress(
                DeploymentStage.INITIALIZING,
                0.0,
                "Initializing deployment..."
            )
            await asyncio.sleep(0.1)  # Allow UI to update
            
            if self._cancelled:
                return self._create_cancelled_result(backend_name, display_name)
            
            # Stage 2: Validation
            self._report_progress(
                DeploymentStage.VALIDATING,
                0.1,
                "Validating generated code..."
            )
            
            validation_errors = await self._validate_code(generated_code)
            if validation_errors:
                for error in validation_errors:
                    warnings.append(f"Validation warning: {error}")
            
            if self._cancelled:
                return self._create_cancelled_result(backend_name, display_name)
            
            # Stage 3: Create backup
            self._report_progress(
                DeploymentStage.CREATING_BACKUP,
                0.2,
                "Creating backup of existing files..."
            )
            
            backend_dir = self.contrib_dir / backend_name
            if backend_dir.exists():
                await self._create_backup(backend_dir)
            
            if self._cancelled:
                await self._rollback()
                return self._create_cancelled_result(backend_name, display_name)
            
            # Stage 4: Write files
            self._report_progress(
                DeploymentStage.WRITING_FILES,
                0.3,
                "Writing backend files..."
            )
            
            output_path = await self._write_files(backend_name, generated_code)
            
            if self._cancelled:
                await self._rollback()
                return self._create_cancelled_result(backend_name, display_name)
            
            # Stage 5: Update registry
            self._report_progress(
                DeploymentStage.UPDATING_REGISTRY,
                0.7,
                "Updating backend registry..."
            )
            
            await self._update_registry(backend_name)
            
            if self._cancelled:
                await self._rollback()
                return self._create_cancelled_result(backend_name, display_name)
            
            # Stage 6: Run verification (optional)
            if run_verification:
                self._report_progress(
                    DeploymentStage.RUNNING_VERIFICATION,
                    0.8,
                    "Running verification checks..."
                )
                
                verification_passed, verification_msg = await self._run_verification(
                    backend_name
                )
                if not verification_passed:
                    warnings.append(f"Verification warning: {verification_msg}")
            
            # Stage 7: Finalize
            self._report_progress(
                DeploymentStage.FINALIZING,
                0.95,
                "Finalizing deployment..."
            )
            
            # Clean up backup on success
            await self._cleanup_backup()
            
            # Complete
            self._report_progress(
                DeploymentStage.COMPLETED,
                1.0,
                "Deployment completed successfully!"
            )
            
            deployment_time = time.time() - start_time
            
            return DeploymentResult(
                success=True,
                backend_name=backend_name,
                display_name=display_name,
                created_files=[str(p) for p in self._created_files],
                output_path=str(output_path),
                warnings=warnings,
                deployment_time=deployment_time
            )
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}", exc_info=True)
            
            # Rollback
            self._report_progress(
                DeploymentStage.FAILED,
                0.0,
                f"Deployment failed: {e}"
            )
            
            await self._rollback()
            
            self._report_progress(
                DeploymentStage.ROLLED_BACK,
                0.0,
                "Changes rolled back"
            )
            
            deployment_time = time.time() - start_time
            
            return DeploymentResult(
                success=False,
                backend_name=backend_name,
                display_name=display_name,
                error_message=str(e),
                warnings=warnings,
                deployment_time=deployment_time
            )
    
    def cancel(self) -> None:
        """Cancel the current deployment."""
        self._cancelled = True
        logger.info("Deployment cancellation requested")
    
    async def _validate_code(self, generated_code: Dict[str, str]) -> List[str]:
        """Validate generated code.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for file_name, code in generated_code.items():
            if file_name.endswith('.py'):
                # Check for syntax errors
                try:
                    compile(code, file_name, 'exec')
                except SyntaxError as e:
                    errors.append(f"{file_name}: {e}")
        
        return errors
    
    async def _create_backup(self, directory: Path) -> None:
        """Create backup of existing directory."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._backup_dir = directory.parent / f".backup_{directory.name}_{timestamp}"
            
            if directory.exists():
                shutil.copytree(directory, self._backup_dir)
                logger.info(f"Created backup: {self._backup_dir}")
                
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")
            self._backup_dir = None
    
    async def _write_files(
        self,
        backend_name: str,
        generated_code: Dict[str, str]
    ) -> Path:
        """Write generated files to disk.
        
        Returns:
            Path to the backend directory
        """
        backend_dir = self.contrib_dir / backend_name
        backend_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(generated_code)
        
        for i, (file_name, content) in enumerate(generated_code.items()):
            progress = 0.3 + (0.4 * (i / total))
            
            self._report_progress(
                DeploymentStage.WRITING_FILES,
                progress,
                f"Writing {file_name}...",
                current_file=file_name
            )
            
            # Determine file path
            if file_name.startswith("tests/"):
                # Test file
                relative_path = file_name.replace("tests/", "")
                file_path = self.tests_dir / relative_path
            else:
                # Backend file
                file_path = backend_dir / file_name
            
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            file_path.write_text(content, encoding='utf-8')
            self._created_files.append(file_path)
            
            logger.info(f"Created: {file_path}")
            
            # Small delay for UI
            await asyncio.sleep(0.05)
        
        return backend_dir
    
    async def _update_registry(self, backend_name: str) -> None:
        """Update backend registry."""
        # Check contrib __init__.py
        contrib_init = self.contrib_dir / "__init__.py"
        
        if not contrib_init.exists():
            # Create new contrib init
            content = self._create_contrib_init(backend_name)
            contrib_init.write_text(content, encoding='utf-8')
            self._created_files.append(contrib_init)
            logger.info(f"Created: {contrib_init}")
            return
        
        # Update existing contrib init
        content = contrib_init.read_text(encoding='utf-8')
        
        if backend_name in content:
            logger.info(f"Backend {backend_name} already in registry")
            return
        
        # Add import
        class_name = self._to_class_name(backend_name)
        import_line = f"from .{backend_name} import {class_name}Adapter"
        
        # Find where to insert
        lines = content.split('\n')
        import_index = -1
        
        for i, line in enumerate(lines):
            if line.startswith('from .') or line.startswith('import '):
                import_index = i
        
        if import_index >= 0:
            lines.insert(import_index + 1, import_line)
        else:
            # Add after docstring
            for i, line in enumerate(lines):
                if not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                    if not line.strip().startswith('#') and line.strip():
                        lines.insert(i, import_line)
                        break
        
        # Write updated registry
        contrib_init.write_text('\n'.join(lines), encoding='utf-8')
        logger.info(f"Updated registry with {backend_name}")
    
    def _create_contrib_init(self, backend_name: str) -> str:
        """Create initial contrib __init__.py content."""
        class_name = self._to_class_name(backend_name)
        
        return f'''"""Proxima Contributed Backends Package.

User-contributed quantum computing backend adapters.
These backends are created through the Backend Addition Wizard.
"""

from .{backend_name} import {class_name}Adapter

__all__ = [
    "{class_name}Adapter",
]


def get_contributed_backends() -> list:
    """Get list of contributed backend names."""
    return [
        "{backend_name}",
    ]


def get_backend(name: str, **kwargs):
    """Get a contributed backend adapter by name.
    
    Args:
        name: Backend name
        **kwargs: Backend configuration
        
    Returns:
        Backend adapter instance
        
    Raises:
        ValueError: If backend not found
    """
    backends = {{
        "{backend_name}": {class_name}Adapter,
    }}
    
    if name not in backends:
        available = list(backends.keys())
        raise ValueError(f"Unknown backend: {{name}}. Available: {{available}}")
    
    return backends[name](**kwargs)
'''
    
    def _to_class_name(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
    
    async def _run_verification(self, backend_name: str) -> Tuple[bool, str]:
        """Run verification checks on deployed backend.
        
        Returns:
            Tuple of (passed, message)
        """
        try:
            # Check that files exist
            backend_dir = self.contrib_dir / backend_name
            
            required_files = ["__init__.py", "adapter.py"]
            missing = []
            
            for f in required_files:
                if not (backend_dir / f).exists():
                    missing.append(f)
            
            if missing:
                return False, f"Missing required files: {missing}"
            
            # Try to import the module
            try:
                import importlib.util
                
                init_path = backend_dir / "__init__.py"
                spec = importlib.util.spec_from_file_location(
                    backend_name,
                    init_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Check for required exports
                    if not hasattr(module, 'get_adapter'):
                        return False, "Module missing get_adapter function"
                    
            except Exception as e:
                return False, f"Import failed: {e}"
            
            return True, "Verification passed"
            
        except Exception as e:
            return False, str(e)
    
    async def _rollback(self) -> None:
        """Rollback changes on failure."""
        try:
            # Remove created files
            for file_path in reversed(self._created_files):
                try:
                    if file_path.exists():
                        file_path.unlink()
                        logger.info(f"Rolled back: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
            
            # Remove empty directories
            for file_path in self._created_files:
                try:
                    parent = file_path.parent
                    if parent.exists() and not list(parent.iterdir()):
                        parent.rmdir()
                except Exception:
                    pass
            
            # Restore backup if exists
            if self._backup_dir and self._backup_dir.exists():
                original_name = self._backup_dir.name.split("_")[1]
                original_dir = self._backup_dir.parent / original_name
                
                if not original_dir.exists():
                    shutil.copytree(self._backup_dir, original_dir)
                    logger.info(f"Restored from backup: {original_dir}")
                
                await self._cleanup_backup()
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    async def _cleanup_backup(self) -> None:
        """Clean up backup directory."""
        if self._backup_dir and self._backup_dir.exists():
            try:
                shutil.rmtree(self._backup_dir, ignore_errors=True)
                logger.info(f"Cleaned up backup: {self._backup_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up backup: {e}")
    
    def _create_cancelled_result(
        self,
        backend_name: str,
        display_name: str
    ) -> DeploymentResult:
        """Create a cancelled deployment result."""
        return DeploymentResult(
            success=False,
            backend_name=backend_name,
            display_name=display_name,
            error_message="Deployment cancelled by user"
        )


class BatchDeploymentManager:
    """Manage batch deployment of multiple backends."""
    
    def __init__(
        self,
        proxima_root: Optional[Path] = None,
        on_progress: Optional[Callable[[str, DeploymentProgress], None]] = None
    ):
        """Initialize batch deployment manager.
        
        Args:
            proxima_root: Root directory of Proxima project
            on_progress: Callback for progress updates (backend_name, progress)
        """
        self.proxima_root = proxima_root or Path.cwd()
        self._on_progress = on_progress
        self._results: Dict[str, DeploymentResult] = {}
    
    async def deploy_all(
        self,
        backends: Dict[str, Dict[str, Any]]
    ) -> Dict[str, DeploymentResult]:
        """Deploy multiple backends.
        
        Args:
            backends: Dictionary mapping backend names to their configuration:
                {
                    "backend_name": {
                        "display_name": "Display Name",
                        "generated_code": {...}
                    }
                }
                
        Returns:
            Dictionary mapping backend names to deployment results
        """
        self._results = {}
        
        for backend_name, config in backends.items():
            def on_backend_progress(progress: DeploymentProgress):
                if self._on_progress:
                    self._on_progress(backend_name, progress)
            
            manager = DeploymentManager(
                proxima_root=self.proxima_root,
                on_progress=on_backend_progress
            )
            
            result = await manager.deploy_backend(
                backend_name=backend_name,
                display_name=config.get("display_name", backend_name),
                generated_code=config.get("generated_code", {})
            )
            
            self._results[backend_name] = result
            
            if not result.success:
                # Stop on first failure
                break
        
        return self._results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get deployment summary.
        
        Returns:
            Summary of all deployment results
        """
        total = len(self._results)
        successful = sum(1 for r in self._results.values() if r.success)
        failed = total - successful
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "results": {
                name: result.to_dict()
                for name, result in self._results.items()
            }
        }
