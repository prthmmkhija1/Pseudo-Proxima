"""Secure Temporary File Manager for Proxima Agent.

Phase 5: File System Operations & Administrative Access

Provides secure temporary file management including:
- Agent-specific temp directory
- Automatic cleanup on shutdown
- Restrictive file permissions
- Secure filename generation
"""

from __future__ import annotations

import atexit
import hashlib
import os
import secrets
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.temp_file_manager")


class TempFileType(Enum):
    """Types of temporary files."""
    BUILD_ARTIFACT = "build_artifact"
    CACHE = "cache"
    DOWNLOAD = "download"
    SCRATCH = "scratch"
    OUTPUT = "output"
    SESSION = "session"


@dataclass
class TempFileInfo:
    """Information about a temporary file."""
    path: Path
    file_type: TempFileType
    created_at: datetime
    expires_at: Optional[datetime]
    size_bytes: int
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "file_type": self.file_type.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "size_bytes": self.size_bytes,
            "description": self.description,
        }
    
    @property
    def is_expired(self) -> bool:
        """Check if file has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def age_hours(self) -> float:
        """Get age in hours."""
        delta = datetime.now() - self.created_at
        return delta.total_seconds() / 3600


@dataclass
class CleanupStats:
    """Statistics from cleanup operation."""
    files_removed: int
    directories_removed: int
    bytes_freed: int
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files_removed": self.files_removed,
            "directories_removed": self.directories_removed,
            "bytes_freed": self.bytes_freed,
            "errors": self.errors,
        }


class TempFileManager:
    """Manage temporary files securely.
    
    Features:
    - Agent-specific temp directory
    - Automatic cleanup on shutdown
    - Configurable expiration
    - Secure filename generation
    - Restrictive permissions
    
    Example:
        >>> manager = TempFileManager()
        >>> 
        >>> # Create a temp file
        >>> path = manager.create_temp_file(
        ...     content="build output",
        ...     file_type=TempFileType.BUILD_ARTIFACT,
        ...     suffix=".log",
        ... )
        >>> 
        >>> # Create a temp directory
        >>> dir_path = manager.create_temp_directory(
        ...     file_type=TempFileType.CACHE,
        ...     prefix="build_cache_",
        ... )
        >>> 
        >>> # Cleanup old files
        >>> stats = manager.cleanup_expired()
    """
    
    DEFAULT_EXPIRY_HOURS = 24
    AGENT_TEMP_PREFIX = "proxima_agent_"
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        default_expiry_hours: float = DEFAULT_EXPIRY_HOURS,
        auto_cleanup: bool = True,
        cleanup_interval_hours: float = 1.0,
        secure_permissions: bool = True,
    ):
        """Initialize the temp file manager.
        
        Args:
            base_dir: Base directory for temp files (default: system temp)
            default_expiry_hours: Default file expiration time
            auto_cleanup: Enable automatic background cleanup
            cleanup_interval_hours: Cleanup interval
            secure_permissions: Set restrictive file permissions
        """
        # Create agent-specific temp directory
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            system_temp = Path(tempfile.gettempdir())
            self.base_dir = system_temp / f"{self.AGENT_TEMP_PREFIX}{os.getpid()}"
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_expiry_hours = default_expiry_hours
        self.secure_permissions = secure_permissions
        
        # Track created files
        self._files: Dict[str, TempFileInfo] = {}
        self._lock = threading.Lock()
        
        # Auto-cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        
        if auto_cleanup:
            self._start_cleanup_thread(cleanup_interval_hours)
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
        
        logger.info(f"TempFileManager initialized at {self.base_dir}")
    
    def _start_cleanup_thread(self, interval_hours: float) -> None:
        """Start background cleanup thread."""
        self._running = True
        interval_seconds = interval_hours * 3600
        
        def cleanup_loop():
            while self._running:
                try:
                    self.cleanup_expired()
                except Exception as e:
                    logger.warning(f"Cleanup error: {e}")
                
                # Sleep in small intervals to allow quick shutdown
                for _ in range(int(interval_seconds)):
                    if not self._running:
                        break
                    time.sleep(1)
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _generate_secure_name(
        self,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Generate a secure random filename.
        
        Args:
            prefix: Filename prefix
            suffix: Filename suffix/extension
            
        Returns:
            Secure random filename
        """
        # Use cryptographically secure random bytes
        random_bytes = secrets.token_bytes(16)
        random_hex = random_bytes.hex()
        
        return f"{prefix}{random_hex}{suffix}"
    
    def _set_secure_permissions(self, path: Path) -> None:
        """Set secure permissions on a file or directory.
        
        Args:
            path: Path to secure
        """
        if not self.secure_permissions:
            return
        
        try:
            if path.is_dir():
                # Directory: owner rwx only (0700)
                os.chmod(path, stat.S_IRWXU)
            else:
                # File: owner rw only (0600)
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as e:
            logger.warning(f"Failed to set permissions on {path}: {e}")
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a filename to prevent injection.
        
        Args:
            name: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove path separators
        name = name.replace("/", "_").replace("\\", "_")
        
        # Remove null bytes and control characters
        name = "".join(c for c in name if ord(c) >= 32)
        
        # Limit length
        if len(name) > 200:
            name = name[:200]
        
        # Ensure not empty
        if not name:
            name = "unnamed"
        
        return name
    
    def get_type_directory(self, file_type: TempFileType) -> Path:
        """Get subdirectory for a file type.
        
        Args:
            file_type: Type of temp file
            
        Returns:
            Path to type-specific directory
        """
        type_dir = self.base_dir / file_type.value
        type_dir.mkdir(parents=True, exist_ok=True)
        self._set_secure_permissions(type_dir)
        return type_dir
    
    def create_temp_file(
        self,
        content: Union[str, bytes, None] = None,
        file_type: TempFileType = TempFileType.SCRATCH,
        prefix: str = "",
        suffix: str = "",
        description: str = "",
        expiry_hours: Optional[float] = None,
    ) -> Path:
        """Create a temporary file.
        
        Args:
            content: Initial content to write
            file_type: Type of temp file
            prefix: Filename prefix
            suffix: Filename suffix/extension
            description: Description for tracking
            expiry_hours: Hours until expiration (None = default)
            
        Returns:
            Path to created file
        """
        type_dir = self.get_type_directory(file_type)
        
        # Generate secure filename
        filename = self._generate_secure_name(
            prefix=self._sanitize_filename(prefix),
            suffix=self._sanitize_filename(suffix),
        )
        
        file_path = type_dir / filename
        
        # Write content
        if content is not None:
            if isinstance(content, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(file_path, "wb") as f:
                    f.write(content)
        else:
            file_path.touch()
        
        # Set secure permissions
        self._set_secure_permissions(file_path)
        
        # Calculate expiry
        if expiry_hours is None:
            expiry_hours = self.default_expiry_hours
        
        expires_at = datetime.now() + timedelta(hours=expiry_hours) if expiry_hours > 0 else None
        
        # Track file
        info = TempFileInfo(
            path=file_path,
            file_type=file_type,
            created_at=datetime.now(),
            expires_at=expires_at,
            size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            description=description,
        )
        
        with self._lock:
            self._files[str(file_path)] = info
        
        logger.debug(f"Created temp file: {file_path}")
        return file_path
    
    def create_temp_directory(
        self,
        file_type: TempFileType = TempFileType.SCRATCH,
        prefix: str = "",
        description: str = "",
        expiry_hours: Optional[float] = None,
    ) -> Path:
        """Create a temporary directory.
        
        Args:
            file_type: Type of temp file
            prefix: Directory name prefix
            description: Description for tracking
            expiry_hours: Hours until expiration (None = default)
            
        Returns:
            Path to created directory
        """
        type_dir = self.get_type_directory(file_type)
        
        # Generate secure dirname
        dirname = self._generate_secure_name(
            prefix=self._sanitize_filename(prefix),
        )
        
        dir_path = type_dir / dirname
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions
        self._set_secure_permissions(dir_path)
        
        # Calculate expiry
        if expiry_hours is None:
            expiry_hours = self.default_expiry_hours
        
        expires_at = datetime.now() + timedelta(hours=expiry_hours) if expiry_hours > 0 else None
        
        # Track directory
        info = TempFileInfo(
            path=dir_path,
            file_type=file_type,
            created_at=datetime.now(),
            expires_at=expires_at,
            size_bytes=0,
            description=description,
        )
        
        with self._lock:
            self._files[str(dir_path)] = info
        
        logger.debug(f"Created temp directory: {dir_path}")
        return dir_path
    
    def create_named_temp_file(
        self,
        name: str,
        content: Union[str, bytes, None] = None,
        file_type: TempFileType = TempFileType.SCRATCH,
        description: str = "",
        expiry_hours: Optional[float] = None,
    ) -> Path:
        """Create a temporary file with a specific name.
        
        Args:
            name: Desired filename (will be sanitized)
            content: Initial content to write
            file_type: Type of temp file
            description: Description for tracking
            expiry_hours: Hours until expiration
            
        Returns:
            Path to created file
        """
        type_dir = self.get_type_directory(file_type)
        
        # Sanitize filename but add random suffix for uniqueness
        safe_name = self._sanitize_filename(name)
        random_suffix = secrets.token_hex(4)
        
        # Insert random suffix before extension
        parts = safe_name.rsplit(".", 1)
        if len(parts) == 2:
            filename = f"{parts[0]}_{random_suffix}.{parts[1]}"
        else:
            filename = f"{safe_name}_{random_suffix}"
        
        file_path = type_dir / filename
        
        # Write content
        if content is not None:
            if isinstance(content, str):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(file_path, "wb") as f:
                    f.write(content)
        else:
            file_path.touch()
        
        # Set secure permissions
        self._set_secure_permissions(file_path)
        
        # Calculate expiry
        if expiry_hours is None:
            expiry_hours = self.default_expiry_hours
        
        expires_at = datetime.now() + timedelta(hours=expiry_hours) if expiry_hours > 0 else None
        
        # Track file
        info = TempFileInfo(
            path=file_path,
            file_type=file_type,
            created_at=datetime.now(),
            expires_at=expires_at,
            size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            description=description,
        )
        
        with self._lock:
            self._files[str(file_path)] = info
        
        logger.debug(f"Created named temp file: {file_path}")
        return file_path
    
    def get_file_info(self, path: Path) -> Optional[TempFileInfo]:
        """Get information about a tracked temp file.
        
        Args:
            path: Path to file
            
        Returns:
            TempFileInfo or None if not tracked
        """
        with self._lock:
            return self._files.get(str(path))
    
    def list_files(
        self,
        file_type: Optional[TempFileType] = None,
        include_expired: bool = False,
    ) -> List[TempFileInfo]:
        """List tracked temporary files.
        
        Args:
            file_type: Filter by type
            include_expired: Include expired files
            
        Returns:
            List of TempFileInfo objects
        """
        with self._lock:
            files = list(self._files.values())
        
        if file_type:
            files = [f for f in files if f.file_type == file_type]
        
        if not include_expired:
            files = [f for f in files if not f.is_expired]
        
        return sorted(files, key=lambda f: f.created_at, reverse=True)
    
    def delete_file(self, path: Path) -> bool:
        """Delete a temporary file.
        
        Args:
            path: Path to file
            
        Returns:
            True if deleted successfully
        """
        path = Path(path).resolve()
        
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            
            with self._lock:
                if str(path) in self._files:
                    del self._files[str(path)]
            
            logger.debug(f"Deleted temp file: {path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")
            return False
    
    def extend_expiry(
        self,
        path: Path,
        additional_hours: float,
    ) -> bool:
        """Extend the expiry time of a file.
        
        Args:
            path: Path to file
            additional_hours: Hours to add
            
        Returns:
            True if extended successfully
        """
        with self._lock:
            info = self._files.get(str(path))
            if info is None:
                return False
            
            if info.expires_at:
                info.expires_at += timedelta(hours=additional_hours)
            else:
                info.expires_at = datetime.now() + timedelta(hours=additional_hours)
        
        return True
    
    def cleanup_expired(self) -> CleanupStats:
        """Clean up expired temporary files.
        
        Returns:
            CleanupStats with results
        """
        stats = CleanupStats(
            files_removed=0,
            directories_removed=0,
            bytes_freed=0,
        )
        
        # Get expired files
        with self._lock:
            expired = [
                (path, info) for path, info in self._files.items()
                if info.is_expired
            ]
        
        for path_str, info in expired:
            path = Path(path_str)
            
            try:
                if path.exists():
                    stats.bytes_freed += info.size_bytes
                    
                    if path.is_dir():
                        # Calculate actual directory size
                        for f in path.rglob("*"):
                            if f.is_file():
                                stats.bytes_freed += f.stat().st_size
                        
                        shutil.rmtree(path)
                        stats.directories_removed += 1
                    else:
                        path.unlink()
                        stats.files_removed += 1
                
                with self._lock:
                    del self._files[path_str]
                    
            except Exception as e:
                stats.errors.append(f"{path}: {e}")
        
        if stats.files_removed > 0 or stats.directories_removed > 0:
            logger.info(
                f"Cleanup: removed {stats.files_removed} files, "
                f"{stats.directories_removed} dirs, freed {stats.bytes_freed} bytes"
            )
        
        return stats
    
    def cleanup_by_type(self, file_type: TempFileType) -> CleanupStats:
        """Clean up all files of a specific type.
        
        Args:
            file_type: Type to clean up
            
        Returns:
            CleanupStats with results
        """
        stats = CleanupStats(
            files_removed=0,
            directories_removed=0,
            bytes_freed=0,
        )
        
        with self._lock:
            to_remove = [
                (path, info) for path, info in self._files.items()
                if info.file_type == file_type
            ]
        
        for path_str, info in to_remove:
            path = Path(path_str)
            
            try:
                if path.exists():
                    stats.bytes_freed += info.size_bytes
                    
                    if path.is_dir():
                        shutil.rmtree(path)
                        stats.directories_removed += 1
                    else:
                        path.unlink()
                        stats.files_removed += 1
                
                with self._lock:
                    del self._files[path_str]
                    
            except Exception as e:
                stats.errors.append(f"{path}: {e}")
        
        return stats
    
    def cleanup_all(self) -> CleanupStats:
        """Clean up all temporary files and the base directory.
        
        Returns:
            CleanupStats with results
        """
        self._running = False
        
        stats = CleanupStats(
            files_removed=0,
            directories_removed=0,
            bytes_freed=0,
        )
        
        try:
            # Calculate total size
            for path in self.base_dir.rglob("*"):
                if path.is_file():
                    stats.bytes_freed += path.stat().st_size
                    stats.files_removed += 1
                elif path.is_dir():
                    stats.directories_removed += 1
            
            # Remove base directory
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
            
            with self._lock:
                self._files.clear()
            
            logger.info(f"Cleaned up temp directory: {self.base_dir}")
            
        except Exception as e:
            stats.errors.append(str(e))
            logger.warning(f"Failed to clean up {self.base_dir}: {e}")
        
        return stats
    
    def get_total_size(self) -> int:
        """Get total size of all tracked temp files.
        
        Returns:
            Total size in bytes
        """
        total = 0
        
        try:
            for path in self.base_dir.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        except Exception:
            pass
        
        return total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about temp file usage.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            files = list(self._files.values())
        
        by_type: Dict[str, int] = {}
        for info in files:
            type_name = info.file_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return {
            "base_dir": str(self.base_dir),
            "total_files": len(files),
            "by_type": by_type,
            "total_size_bytes": self.get_total_size(),
            "expired_count": sum(1 for f in files if f.is_expired),
        }


# Global instance
_manager: Optional[TempFileManager] = None


def get_temp_file_manager() -> TempFileManager:
    """Get the global TempFileManager instance."""
    global _manager
    if _manager is None:
        _manager = TempFileManager()
    return _manager


def create_temp_file(
    content: Union[str, bytes, None] = None,
    suffix: str = "",
    file_type: TempFileType = TempFileType.SCRATCH,
) -> Path:
    """Convenience function to create a temp file."""
    return get_temp_file_manager().create_temp_file(
        content=content,
        suffix=suffix,
        file_type=file_type,
    )


def create_temp_directory(
    prefix: str = "",
    file_type: TempFileType = TempFileType.SCRATCH,
) -> Path:
    """Convenience function to create a temp directory."""
    return get_temp_file_manager().create_temp_directory(
        prefix=prefix,
        file_type=file_type,
    )
