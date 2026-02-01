"""File System Operations for Proxima Agent.

Phase 5: File System Operations & Administrative Access

Provides safe file system operations including:
- Path validation to prevent directory traversal
- Safe file reading with size limits
- Atomic file writing with backups
- Directory listing and search
- Permission checking
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.file_system_operations")


class FileOperationType(Enum):
    """Types of file operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    CREATE = "create"
    LIST = "list"
    SEARCH = "search"


class FileType(Enum):
    """Types of files."""
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    UNKNOWN = "unknown"


@dataclass
class FileInfo:
    """Information about a file or directory."""
    path: Path
    name: str
    file_type: FileType
    size_bytes: int
    modified_time: datetime
    created_time: Optional[datetime]
    permissions: str
    is_hidden: bool
    is_readonly: bool
    owner: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "name": self.name,
            "file_type": self.file_type.value,
            "size_bytes": self.size_bytes,
            "modified_time": self.modified_time.isoformat(),
            "created_time": self.created_time.isoformat() if self.created_time else None,
            "permissions": self.permissions,
            "is_hidden": self.is_hidden,
            "is_readonly": self.is_readonly,
            "owner": self.owner,
        }
    
    @property
    def size_human(self) -> str:
        """Get human-readable size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


@dataclass
class FileOperationResult:
    """Result of a file operation."""
    success: bool
    operation: FileOperationType
    path: Path
    message: str
    data: Any = None
    error: Optional[str] = None
    backup_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "operation": self.operation.value,
            "path": str(self.path),
            "message": self.message,
            "error": self.error,
            "backup_path": str(self.backup_path) if self.backup_path else None,
        }


@dataclass
class SearchMatch:
    """A search match in a file."""
    file_path: Path
    line_number: int
    line_content: str
    match_start: int
    match_end: int
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "line_content": self.line_content,
            "match_start": self.match_start,
            "match_end": self.match_end,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


class PathValidator:
    """Validate file paths for security."""
    
    # System directories that should never be accessed
    BLOCKED_PATHS_WINDOWS = [
        "C:\\Windows\\System32",
        "C:\\Windows\\SysWOW64",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData",
    ]
    
    BLOCKED_PATHS_UNIX = [
        "/etc",
        "/sys",
        "/proc",
        "/boot",
        "/root",
        "/usr/bin",
        "/usr/sbin",
    ]
    
    # Dangerous file patterns
    DANGEROUS_PATTERNS = [
        r"\.\.[\\/]",           # Directory traversal
        r"^~",                   # Home directory expansion
        r"[\x00-\x1f]",          # Control characters
        r"\$\{.*\}",             # Shell variable expansion
        r"`.*`",                 # Backtick execution
    ]
    
    def __init__(
        self,
        allowed_roots: Optional[List[Path]] = None,
        blocked_paths: Optional[List[Path]] = None,
    ):
        """Initialize the validator.
        
        Args:
            allowed_roots: List of allowed root directories
            blocked_paths: Additional blocked paths
        """
        self.allowed_roots = [Path(p).resolve() for p in (allowed_roots or [])]
        
        # Add system blocked paths
        if platform.system() == "Windows":
            default_blocked = [Path(p) for p in self.BLOCKED_PATHS_WINDOWS]
        else:
            default_blocked = [Path(p) for p in self.BLOCKED_PATHS_UNIX]
        
        self.blocked_paths = default_blocked + [
            Path(p).resolve() for p in (blocked_paths or [])
        ]
        
        # Compile dangerous patterns
        self._dangerous_re = [re.compile(p) for p in self.DANGEROUS_PATTERNS]
    
    def validate(self, path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate a path for safety.
        
        Args:
            path: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        path_str = str(path)
        
        # Check for dangerous patterns
        for pattern in self._dangerous_re:
            if pattern.search(path_str):
                return False, f"Path contains dangerous pattern: {pattern.pattern}"
        
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError) as e:
            return False, f"Invalid path: {e}"
        
        # Check against blocked paths
        for blocked in self.blocked_paths:
            try:
                if resolved == blocked or blocked in resolved.parents:
                    return False, f"Access to {blocked} is blocked"
            except (ValueError, OSError):
                continue
        
        # Check if within allowed roots (if specified)
        if self.allowed_roots:
            is_allowed = False
            for root in self.allowed_roots:
                try:
                    resolved.relative_to(root)
                    is_allowed = True
                    break
                except ValueError:
                    continue
            
            if not is_allowed:
                return False, "Path is outside allowed directories"
        
        return True, ""
    
    def normalize(self, path: Union[str, Path]) -> Path:
        """Normalize and resolve a path.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized absolute path
        """
        return Path(path).resolve()


class FileSystemOperations:
    """Safe file system operations for the agent.
    
    Features:
    - Path validation to prevent directory traversal
    - File size limits for reading
    - Atomic writes with backup
    - Permission checking
    - Recursive directory operations
    
    Example:
        >>> fs = FileSystemOperations(project_root=Path("."))
        >>> 
        >>> # Read a file
        >>> result = fs.read_file("src/main.py")
        >>> if result.success:
        ...     print(result.data)
        >>> 
        >>> # Write with backup
        >>> result = fs.write_file("config.yaml", "key: value", create_backup=True)
        >>> 
        >>> # Search for pattern
        >>> matches = fs.search_content("def main", file_pattern="*.py")
    """
    
    DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    DEFAULT_BACKUP_DIR = ".proxima_backups"
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        backup_dir: Optional[Path] = None,
        allowed_extensions: Optional[Set[str]] = None,
        blocked_extensions: Optional[Set[str]] = None,
    ):
        """Initialize file system operations.
        
        Args:
            project_root: Root directory for operations
            max_file_size: Maximum file size for read operations
            backup_dir: Directory for file backups
            allowed_extensions: Whitelist of allowed file extensions
            blocked_extensions: Blacklist of blocked file extensions
        """
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.max_file_size = max_file_size
        self.backup_dir = backup_dir or self.project_root / self.DEFAULT_BACKUP_DIR
        
        self.allowed_extensions = allowed_extensions
        self.blocked_extensions = blocked_extensions or {
            ".exe", ".dll", ".so", ".dylib",  # Executables
            ".sys", ".drv",                    # System files
            ".msi", ".deb", ".rpm",            # Installers
        }
        
        # Initialize path validator
        self.validator = PathValidator(
            allowed_roots=[self.project_root, Path(tempfile.gettempdir())],
        )
        
        # Operation callbacks
        self._operation_callbacks: List[Callable[[FileOperationType, Path], None]] = []
    
    def on_operation(
        self,
        callback: Callable[[FileOperationType, Path], None],
    ) -> None:
        """Register an operation callback.
        
        Args:
            callback: Function called before each operation
        """
        self._operation_callbacks.append(callback)
    
    def _notify_operation(self, operation: FileOperationType, path: Path) -> None:
        """Notify callbacks of an operation."""
        for callback in self._operation_callbacks:
            try:
                callback(operation, path)
            except Exception as e:
                logger.warning(f"Operation callback error: {e}")
    
    def _validate_path(self, path: Union[str, Path]) -> Tuple[bool, Path, str]:
        """Validate and normalize a path.
        
        Args:
            path: Path to validate
            
        Returns:
            Tuple of (is_valid, normalized_path, error_message)
        """
        is_valid, error = self.validator.validate(path)
        if not is_valid:
            return False, Path(path), error
        
        normalized = self.validator.normalize(path)
        return True, normalized, ""
    
    def _check_extension(self, path: Path) -> Tuple[bool, str]:
        """Check if file extension is allowed.
        
        Args:
            path: Path to check
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        ext = path.suffix.lower()
        
        if self.blocked_extensions and ext in self.blocked_extensions:
            return False, f"File extension {ext} is blocked"
        
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return False, f"File extension {ext} is not in allowed list"
        
        return True, ""
    
    def get_file_info(self, path: Union[str, Path]) -> FileOperationResult:
        """Get information about a file or directory.
        
        Args:
            path: Path to file or directory
            
        Returns:
            FileOperationResult with FileInfo data
        """
        is_valid, normalized, error = self._validate_path(path)
        if not is_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=Path(path),
                message="Path validation failed",
                error=error,
            )
        
        if not normalized.exists():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=normalized,
                message="Path does not exist",
                error="File or directory not found",
            )
        
        try:
            stat_info = normalized.stat()
            
            # Determine file type
            if normalized.is_symlink():
                file_type = FileType.SYMLINK
            elif normalized.is_dir():
                file_type = FileType.DIRECTORY
            elif normalized.is_file():
                file_type = FileType.FILE
            else:
                file_type = FileType.UNKNOWN
            
            # Get permissions string
            mode = stat_info.st_mode
            permissions = stat.filemode(mode)
            
            # Check if hidden
            is_hidden = normalized.name.startswith(".")
            if platform.system() == "Windows":
                try:
                    import ctypes
                    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(normalized))
                    is_hidden = bool(attrs & 2)  # FILE_ATTRIBUTE_HIDDEN
                except Exception:
                    pass
            
            # Get owner (Unix only)
            owner = None
            if platform.system() != "Windows":
                try:
                    import pwd
                    owner = pwd.getpwuid(stat_info.st_uid).pw_name
                except Exception:
                    pass
            
            file_info = FileInfo(
                path=normalized,
                name=normalized.name,
                file_type=file_type,
                size_bytes=stat_info.st_size if file_type == FileType.FILE else 0,
                modified_time=datetime.fromtimestamp(stat_info.st_mtime),
                created_time=datetime.fromtimestamp(stat_info.st_ctime),
                permissions=permissions,
                is_hidden=is_hidden,
                is_readonly=not os.access(normalized, os.W_OK),
                owner=owner,
            )
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.READ,
                path=normalized,
                message="File info retrieved",
                data=file_info,
            )
            
        except OSError as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=normalized,
                message="Failed to get file info",
                error=str(e),
            )
    
    def read_file(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        binary: bool = False,
    ) -> FileOperationResult:
        """Read contents of a file.
        
        Args:
            path: Path to file
            encoding: Text encoding (ignored for binary)
            start_line: Start line number (1-indexed)
            end_line: End line number (inclusive)
            binary: Read as binary
            
        Returns:
            FileOperationResult with file contents
        """
        is_valid, normalized, error = self._validate_path(path)
        if not is_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=Path(path),
                message="Path validation failed",
                error=error,
            )
        
        self._notify_operation(FileOperationType.READ, normalized)
        
        if not normalized.exists():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=normalized,
                message="File not found",
                error="The specified file does not exist",
            )
        
        if not normalized.is_file():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=normalized,
                message="Not a file",
                error="Path is not a regular file",
            )
        
        # Check file size
        file_size = normalized.stat().st_size
        if file_size > self.max_file_size:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=normalized,
                message="File too large",
                error=f"File size ({file_size} bytes) exceeds limit ({self.max_file_size} bytes)",
            )
        
        try:
            if binary:
                with open(normalized, "rb") as f:
                    content = f.read()
            else:
                # Try to detect encoding
                try:
                    with open(normalized, "r", encoding=encoding) as f:
                        if start_line is not None or end_line is not None:
                            lines = f.readlines()
                            start = (start_line - 1) if start_line else 0
                            end = end_line if end_line else len(lines)
                            content = "".join(lines[start:end])
                        else:
                            content = f.read()
                except UnicodeDecodeError:
                    # Fallback to latin-1
                    with open(normalized, "r", encoding="latin-1") as f:
                        content = f.read()
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.READ,
                path=normalized,
                message="File read successfully",
                data=content,
            )
            
        except OSError as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=normalized,
                message="Failed to read file",
                error=str(e),
            )
    
    def write_file(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        create_backup: bool = True,
        create_dirs: bool = True,
        atomic: bool = True,
    ) -> FileOperationResult:
        """Write content to a file.
        
        Args:
            path: Path to file
            content: Content to write
            encoding: Text encoding
            create_backup: Create backup of existing file
            create_dirs: Create parent directories if needed
            atomic: Use atomic write (temp file + rename)
            
        Returns:
            FileOperationResult
        """
        is_valid, normalized, error = self._validate_path(path)
        if not is_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.WRITE,
                path=Path(path),
                message="Path validation failed",
                error=error,
            )
        
        ext_ok, ext_error = self._check_extension(normalized)
        if not ext_ok:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.WRITE,
                path=normalized,
                message="Extension check failed",
                error=ext_error,
            )
        
        self._notify_operation(FileOperationType.WRITE, normalized)
        
        backup_path = None
        
        try:
            # Create parent directories if needed
            if create_dirs and not normalized.parent.exists():
                normalized.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            if create_backup and normalized.exists():
                backup_path = self._create_backup(normalized)
            
            # Write content
            if atomic:
                # Write to temp file first, then rename
                temp_fd, temp_path = tempfile.mkstemp(
                    dir=normalized.parent,
                    prefix=f".{normalized.name}.",
                    suffix=".tmp",
                )
                try:
                    if isinstance(content, bytes):
                        os.write(temp_fd, content)
                    else:
                        os.write(temp_fd, content.encode(encoding))
                    os.close(temp_fd)
                    
                    # Rename temp to target
                    shutil.move(temp_path, normalized)
                except Exception:
                    os.close(temp_fd)
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise
            else:
                # Direct write
                mode = "wb" if isinstance(content, bytes) else "w"
                with open(normalized, mode, encoding=None if isinstance(content, bytes) else encoding) as f:
                    f.write(content)
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.WRITE,
                path=normalized,
                message="File written successfully",
                backup_path=backup_path,
            )
            
        except OSError as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.WRITE,
                path=normalized,
                message="Failed to write file",
                error=str(e),
                backup_path=backup_path,
            )
    
    def _create_backup(self, path: Path) -> Optional[Path]:
        """Create a backup of a file.
        
        Args:
            path: Path to file
            
        Returns:
            Path to backup file, or None
        """
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.name}.{timestamp}.bak"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(path, backup_path)
            
            logger.debug(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            return None
    
    def delete_file(
        self,
        path: Union[str, Path],
        create_backup: bool = True,
    ) -> FileOperationResult:
        """Delete a file.
        
        Args:
            path: Path to file
            create_backup: Create backup before deletion
            
        Returns:
            FileOperationResult
        """
        is_valid, normalized, error = self._validate_path(path)
        if not is_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.DELETE,
                path=Path(path),
                message="Path validation failed",
                error=error,
            )
        
        if not normalized.exists():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.DELETE,
                path=normalized,
                message="File not found",
                error="The specified file does not exist",
            )
        
        self._notify_operation(FileOperationType.DELETE, normalized)
        
        backup_path = None
        
        try:
            if create_backup:
                backup_path = self._create_backup(normalized)
            
            if normalized.is_dir():
                shutil.rmtree(normalized)
            else:
                normalized.unlink()
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.DELETE,
                path=normalized,
                message="File deleted successfully",
                backup_path=backup_path,
            )
            
        except OSError as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.DELETE,
                path=normalized,
                message="Failed to delete file",
                error=str(e),
                backup_path=backup_path,
            )
    
    def list_directory(
        self,
        path: Union[str, Path],
        pattern: Optional[str] = None,
        recursive: bool = False,
        max_depth: int = 10,
        include_hidden: bool = False,
    ) -> FileOperationResult:
        """List contents of a directory.
        
        Args:
            path: Path to directory
            pattern: Glob pattern to filter files
            recursive: List recursively
            max_depth: Maximum recursion depth
            include_hidden: Include hidden files
            
        Returns:
            FileOperationResult with list of FileInfo
        """
        is_valid, normalized, error = self._validate_path(path)
        if not is_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.LIST,
                path=Path(path),
                message="Path validation failed",
                error=error,
            )
        
        if not normalized.exists():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.LIST,
                path=normalized,
                message="Directory not found",
                error="The specified directory does not exist",
            )
        
        if not normalized.is_dir():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.LIST,
                path=normalized,
                message="Not a directory",
                error="Path is not a directory",
            )
        
        self._notify_operation(FileOperationType.LIST, normalized)
        
        try:
            entries = []
            
            def process_entry(entry_path: Path, depth: int = 0) -> None:
                if depth > max_depth:
                    return
                
                # Check hidden
                if not include_hidden and entry_path.name.startswith("."):
                    return
                
                # Get file info
                result = self.get_file_info(entry_path)
                if result.success:
                    entries.append(result.data)
                
                # Recurse into directories
                if recursive and entry_path.is_dir() and depth < max_depth:
                    try:
                        for child in entry_path.iterdir():
                            process_entry(child, depth + 1)
                    except PermissionError:
                        pass
            
            # List entries
            if pattern:
                for entry_path in normalized.glob(pattern):
                    process_entry(entry_path)
            else:
                for entry_path in normalized.iterdir():
                    process_entry(entry_path)
            
            # Sort: directories first, then by name
            entries.sort(key=lambda e: (e.file_type != FileType.DIRECTORY, e.name.lower()))
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.LIST,
                path=normalized,
                message=f"Listed {len(entries)} entries",
                data=entries,
            )
            
        except OSError as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.LIST,
                path=normalized,
                message="Failed to list directory",
                error=str(e),
            )
    
    def copy_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path],
        overwrite: bool = False,
    ) -> FileOperationResult:
        """Copy a file or directory.
        
        Args:
            source: Source path
            destination: Destination path
            overwrite: Overwrite if destination exists
            
        Returns:
            FileOperationResult
        """
        src_valid, src_norm, src_error = self._validate_path(source)
        if not src_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.COPY,
                path=Path(source),
                message="Source path validation failed",
                error=src_error,
            )
        
        dst_valid, dst_norm, dst_error = self._validate_path(destination)
        if not dst_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.COPY,
                path=Path(destination),
                message="Destination path validation failed",
                error=dst_error,
            )
        
        if not src_norm.exists():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.COPY,
                path=src_norm,
                message="Source not found",
                error="The source file does not exist",
            )
        
        if dst_norm.exists() and not overwrite:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.COPY,
                path=dst_norm,
                message="Destination exists",
                error="Use overwrite=True to replace existing file",
            )
        
        self._notify_operation(FileOperationType.COPY, src_norm)
        
        try:
            if src_norm.is_dir():
                if dst_norm.exists():
                    shutil.rmtree(dst_norm)
                shutil.copytree(src_norm, dst_norm)
            else:
                dst_norm.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_norm, dst_norm)
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.COPY,
                path=dst_norm,
                message="Copy completed successfully",
            )
            
        except OSError as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.COPY,
                path=dst_norm,
                message="Failed to copy",
                error=str(e),
            )
    
    def move_file(
        self,
        source: Union[str, Path],
        destination: Union[str, Path],
        overwrite: bool = False,
    ) -> FileOperationResult:
        """Move a file or directory.
        
        Args:
            source: Source path
            destination: Destination path
            overwrite: Overwrite if destination exists
            
        Returns:
            FileOperationResult
        """
        src_valid, src_norm, src_error = self._validate_path(source)
        if not src_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.MOVE,
                path=Path(source),
                message="Source path validation failed",
                error=src_error,
            )
        
        dst_valid, dst_norm, dst_error = self._validate_path(destination)
        if not dst_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.MOVE,
                path=Path(destination),
                message="Destination path validation failed",
                error=dst_error,
            )
        
        if not src_norm.exists():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.MOVE,
                path=src_norm,
                message="Source not found",
                error="The source file does not exist",
            )
        
        if dst_norm.exists() and not overwrite:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.MOVE,
                path=dst_norm,
                message="Destination exists",
                error="Use overwrite=True to replace existing file",
            )
        
        self._notify_operation(FileOperationType.MOVE, src_norm)
        
        try:
            if dst_norm.exists():
                if dst_norm.is_dir():
                    shutil.rmtree(dst_norm)
                else:
                    dst_norm.unlink()
            
            dst_norm.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_norm), str(dst_norm))
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.MOVE,
                path=dst_norm,
                message="Move completed successfully",
            )
            
        except OSError as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.MOVE,
                path=dst_norm,
                message="Failed to move",
                error=str(e),
            )
    
    def search_content(
        self,
        pattern: str,
        path: Optional[Union[str, Path]] = None,
        file_pattern: str = "*",
        is_regex: bool = False,
        case_sensitive: bool = True,
        context_lines: int = 3,
        max_results: int = 100,
    ) -> FileOperationResult:
        """Search for content in files.
        
        Args:
            pattern: Search pattern
            path: Directory to search in (default: project root)
            file_pattern: Glob pattern for files to search
            is_regex: Treat pattern as regex
            case_sensitive: Case-sensitive search
            context_lines: Lines of context before/after match
            max_results: Maximum number of results
            
        Returns:
            FileOperationResult with list of SearchMatch
        """
        search_path = Path(path) if path else self.project_root
        
        is_valid, normalized, error = self._validate_path(search_path)
        if not is_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.SEARCH,
                path=search_path,
                message="Path validation failed",
                error=error,
            )
        
        self._notify_operation(FileOperationType.SEARCH, normalized)
        
        try:
            if is_regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                regex = re.compile(pattern, flags)
            else:
                if not case_sensitive:
                    pattern = pattern.lower()
            
            matches = []
            
            for file_path in normalized.rglob(file_pattern):
                if not file_path.is_file():
                    continue
                
                if len(matches) >= max_results:
                    break
                
                # Skip binary files
                if file_path.suffix.lower() in self.blocked_extensions:
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines):
                        search_line = line if case_sensitive else line.lower()
                        
                        if is_regex:
                            match = regex.search(line)
                            if match:
                                matches.append(SearchMatch(
                                    file_path=file_path,
                                    line_number=i + 1,
                                    line_content=line.rstrip("\n\r"),
                                    match_start=match.start(),
                                    match_end=match.end(),
                                    context_before=[
                                        lines[j].rstrip("\n\r")
                                        for j in range(max(0, i - context_lines), i)
                                    ],
                                    context_after=[
                                        lines[j].rstrip("\n\r")
                                        for j in range(i + 1, min(len(lines), i + 1 + context_lines))
                                    ],
                                ))
                        else:
                            if pattern in search_line:
                                start = search_line.find(pattern)
                                matches.append(SearchMatch(
                                    file_path=file_path,
                                    line_number=i + 1,
                                    line_content=line.rstrip("\n\r"),
                                    match_start=start,
                                    match_end=start + len(pattern),
                                    context_before=[
                                        lines[j].rstrip("\n\r")
                                        for j in range(max(0, i - context_lines), i)
                                    ],
                                    context_after=[
                                        lines[j].rstrip("\n\r")
                                        for j in range(i + 1, min(len(lines), i + 1 + context_lines))
                                    ],
                                ))
                        
                        if len(matches) >= max_results:
                            break
                            
                except (OSError, UnicodeDecodeError):
                    continue
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.SEARCH,
                path=normalized,
                message=f"Found {len(matches)} matches",
                data=matches,
            )
            
        except re.error as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.SEARCH,
                path=normalized,
                message="Invalid regex pattern",
                error=str(e),
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.SEARCH,
                path=normalized,
                message="Search failed",
                error=str(e),
            )
    
    def compute_checksum(
        self,
        path: Union[str, Path],
        algorithm: str = "sha256",
    ) -> FileOperationResult:
        """Compute checksum of a file.
        
        Args:
            path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            FileOperationResult with checksum string
        """
        is_valid, normalized, error = self._validate_path(path)
        if not is_valid:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=Path(path),
                message="Path validation failed",
                error=error,
            )
        
        if not normalized.is_file():
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=normalized,
                message="Not a file",
                error="Path is not a regular file",
            )
        
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(normalized, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            
            return FileOperationResult(
                success=True,
                operation=FileOperationType.READ,
                path=normalized,
                message=f"{algorithm.upper()} checksum computed",
                data=hash_obj.hexdigest(),
            )
            
        except Exception as e:
            return FileOperationResult(
                success=False,
                operation=FileOperationType.READ,
                path=normalized,
                message="Failed to compute checksum",
                error=str(e),
            )


def get_file_system_operations(project_root: Optional[Path] = None) -> FileSystemOperations:
    """Get a FileSystemOperations instance."""
    return FileSystemOperations(project_root)
