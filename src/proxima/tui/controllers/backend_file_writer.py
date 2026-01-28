"""Backend File Writer.

Writes generated backend files to disk and updates the backend registry.
Handles file creation, directory structure, and cleanup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class BackendFileWriter:
    """Write generated backend files to disk.
    
    Handles:
    - Creating backend package directory structure
    - Writing generated files
    - Updating backend registry
    - Backup and rollback on failure
    """
    
    def __init__(self, proxima_root: Optional[Path] = None):
        """Initialize file writer.
        
        Args:
            proxima_root: Root directory of Proxima project.
                         Defaults to current working directory.
        """
        self.proxima_root = proxima_root or Path.cwd()
        self.backends_dir = self.proxima_root / "src" / "proxima" / "backends"
        self.tests_dir = self.proxima_root / "tests" / "backends"
        self._backup_dir: Optional[Path] = None
        self._created_files: List[Path] = []
    
    async def write_all_files(
        self,
        backend_name: str,
        generated_code: Dict[str, str]
    ) -> Tuple[bool, List[str], Optional[str]]:
        """Write all generated files.
        
        Args:
            backend_name: Name of the backend
            generated_code: Dictionary mapping file paths to content
            
        Returns:
            Tuple of (success, list of created file paths, error message or None)
        """
        self._created_files = []
        
        try:
            # Create backup if backend directory already exists
            backend_dir = self.backends_dir / backend_name
            if backend_dir.exists():
                await self._create_backup(backend_dir)
            
            # Create backend directory
            backend_dir.mkdir(parents=True, exist_ok=True)
            
            # Write each file
            for file_name, code in generated_code.items():
                file_path = await self._write_file(backend_name, file_name, code)
                if file_path:
                    self._created_files.append(file_path)
                    logger.info(f"Created: {file_path}")
            
            # Update backend registry
            await self._update_registry(backend_name)
            
            # Clean up backup on success
            if self._backup_dir and self._backup_dir.exists():
                shutil.rmtree(self._backup_dir, ignore_errors=True)
            
            return True, [str(p) for p in self._created_files], None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error writing backend files: {e}", exc_info=True)
            
            # Rollback: remove created files
            await self._rollback()
            
            return False, [], error_msg
    
    async def _write_file(
        self,
        backend_name: str,
        file_name: str,
        content: str
    ) -> Optional[Path]:
        """Write a single file.
        
        Args:
            backend_name: Name of the backend
            file_name: Relative file path
            content: File content
            
        Returns:
            Path to created file or None on failure
        """
        try:
            # Determine full path
            if file_name.startswith("tests/"):
                # Test file
                relative_path = file_name.replace("tests/", "")
                file_path = self.tests_dir / relative_path
            elif "/" in file_name:
                # File with path (e.g., "my_backend/adapter.py")
                file_path = self.backends_dir / file_name
            else:
                # Single file
                file_path = self.backends_dir / backend_name / file_name
            
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            file_path.write_text(content, encoding='utf-8')
            
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to write {file_name}: {e}")
            return None
    
    async def _create_backup(self, backend_dir: Path) -> None:
        """Create backup of existing backend directory.
        
        Args:
            backend_dir: Directory to backup
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._backup_dir = backend_dir.parent / f".backup_{backend_dir.name}_{timestamp}"
            
            if backend_dir.exists():
                shutil.copytree(backend_dir, self._backup_dir)
                logger.info(f"Created backup: {self._backup_dir}")
                
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")
            self._backup_dir = None
    
    async def _rollback(self) -> None:
        """Rollback changes on failure."""
        try:
            # Remove created files
            for file_path in self._created_files:
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
                # Determine original directory
                original_name = self._backup_dir.name.split("_")[1]  # .backup_NAME_timestamp
                original_dir = self._backup_dir.parent / original_name
                
                if not original_dir.exists():
                    shutil.copytree(self._backup_dir, original_dir)
                    logger.info(f"Restored from backup: {original_dir}")
                
                shutil.rmtree(self._backup_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    async def _update_registry(self, backend_name: str) -> None:
        """Update backend registry to include new backend.
        
        Args:
            backend_name: Name of the new backend
        """
        registry_file = self.backends_dir / "__init__.py"
        
        if not registry_file.exists():
            # Create new registry file
            content = self._create_registry_content(backend_name)
            registry_file.write_text(content, encoding='utf-8')
            self._created_files.append(registry_file)
            logger.info(f"Created backend registry: {registry_file}")
            return
        
        # Read existing registry
        content = registry_file.read_text(encoding='utf-8')
        
        # Check if already registered
        if backend_name in content:
            logger.info(f"Backend {backend_name} already in registry")
            return
        
        # Add import and registration
        class_name = self._to_class_name(backend_name)
        import_line = f"from .{backend_name} import {class_name}Adapter"
        
        # Find where to insert import
        lines = content.split('\n')
        import_index = -1
        for i, line in enumerate(lines):
            if line.startswith('from .'):
                import_index = i
        
        if import_index >= 0:
            lines.insert(import_index + 1, import_line)
        else:
            # Add at the beginning after docstring
            for i, line in enumerate(lines):
                if not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                    if not line.strip().startswith('#') and line.strip():
                        lines.insert(i, import_line)
                        break
        
        # Write updated registry
        registry_file.write_text('\n'.join(lines), encoding='utf-8')
        logger.info(f"Updated backend registry with {backend_name}")
    
    def _create_registry_content(self, backend_name: str) -> str:
        """Create initial registry file content.
        
        Args:
            backend_name: Name of the first backend to register
            
        Returns:
            Registry file content
        """
        class_name = self._to_class_name(backend_name)
        
        return f'''"""Proxima Backends Package.

Contains all quantum computing backend adapters.
"""

from .{backend_name} import {class_name}Adapter

__all__ = [
    "{class_name}Adapter",
]


def get_available_backends():
    """Get list of available backend names."""
    return [
        "{backend_name}",
    ]


def get_backend(name: str, **kwargs):
    """Get a backend adapter by name.
    
    Args:
        name: Backend name
        **kwargs: Backend configuration
        
    Returns:
        Backend adapter instance
    """
    backends = {{
        "{backend_name}": {class_name}Adapter,
    }}
    
    if name not in backends:
        raise ValueError(f"Unknown backend: {{name}}. Available: {{list(backends.keys())}}")
    
    return backends[name](**kwargs)
'''
    
    def _to_class_name(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
    
    async def remove_backend(self, backend_name: str) -> Tuple[bool, Optional[str]]:
        """Remove a backend from the project.
        
        Args:
            backend_name: Name of the backend to remove
            
        Returns:
            Tuple of (success, error message or None)
        """
        try:
            # Create backup first
            backend_dir = self.backends_dir / backend_name
            if backend_dir.exists():
                await self._create_backup(backend_dir)
                shutil.rmtree(backend_dir)
                logger.info(f"Removed backend directory: {backend_dir}")
            
            # Remove test file
            test_file = self.tests_dir / f"test_{backend_name}.py"
            if test_file.exists():
                test_file.unlink()
                logger.info(f"Removed test file: {test_file}")
            
            # Update registry
            await self._remove_from_registry(backend_name)
            
            return True, None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to remove backend: {e}")
            return False, error_msg
    
    async def _remove_from_registry(self, backend_name: str) -> None:
        """Remove backend from registry.
        
        Args:
            backend_name: Name of the backend to remove
        """
        registry_file = self.backends_dir / "__init__.py"
        
        if not registry_file.exists():
            return
        
        content = registry_file.read_text(encoding='utf-8')
        
        # Remove import line
        class_name = self._to_class_name(backend_name)
        import_line = f"from .{backend_name} import {class_name}Adapter"
        
        content = content.replace(import_line + "\n", "")
        content = content.replace(import_line, "")
        
        # Remove from __all__
        content = content.replace(f'    "{class_name}Adapter",\n', "")
        
        # Write updated content
        registry_file.write_text(content, encoding='utf-8')
        logger.info(f"Removed {backend_name} from registry")
