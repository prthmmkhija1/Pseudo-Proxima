"""File Backup System for Safe Code Modifications.

Phase 8: Backend Code Modification with Safety

Provides comprehensive file backup capabilities including:
- Automatic backup before modifications
- Checksum verification
- Manifest-based backup organization
- Automatic cleanup of old backups
- Incremental backup support
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.file_backup_system")


class BackupType(Enum):
    """Type of backup."""
    FULL = "full"           # Complete file content
    DIFF = "diff"           # Only changes (for large files)
    COMPRESSED = "compressed"  # Gzip compressed


@dataclass
class FileSnapshot:
    """Snapshot of a file at a point in time."""
    path: str  # Original file path
    relative_path: str  # Relative to project root
    checksum: str  # SHA256 hash
    size: int
    mtime: float  # Modification time
    backup_type: BackupType = BackupType.FULL
    backup_path: Optional[str] = None  # Where backup is stored
    content: Optional[str] = None  # For small files (< 1MB)
    diff_content: Optional[str] = None  # For diff backups
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "checksum": self.checksum,
            "size": self.size,
            "mtime": self.mtime,
            "backup_type": self.backup_type.value,
            "backup_path": self.backup_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileSnapshot":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            relative_path=data["relative_path"],
            checksum=data["checksum"],
            size=data["size"],
            mtime=data["mtime"],
            backup_type=BackupType(data.get("backup_type", "full")),
            backup_path=data.get("backup_path"),
        )


@dataclass
class BackupManifest:
    """Manifest for a backup operation."""
    operation_id: str
    timestamp: str
    operation_type: str
    description: str
    files: List[FileSnapshot] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "timestamp": self.timestamp,
            "operation_type": self.operation_type,
            "description": self.description,
            "files": [f.to_dict() for f in self.files],
            "metadata": self.metadata,
            "completed": self.completed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupManifest":
        """Create from dictionary."""
        manifest = cls(
            operation_id=data["operation_id"],
            timestamp=data["timestamp"],
            operation_type=data["operation_type"],
            description=data["description"],
            metadata=data.get("metadata", {}),
            completed=data.get("completed", False),
        )
        manifest.files = [FileSnapshot.from_dict(f) for f in data.get("files", [])]
        return manifest
    
    def save(self, path: Path) -> bool:
        """Save manifest to file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            return False
    
    @classmethod
    def load(cls, path: Path) -> Optional["BackupManifest"]:
        """Load manifest from file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return None


@dataclass
class RestoreResult:
    """Result of a restore operation."""
    success: bool
    message: str
    restored_files: List[str] = field(default_factory=list)
    failed_files: List[Tuple[str, str]] = field(default_factory=list)  # (path, error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "restored_files": self.restored_files,
            "failed_files": self.failed_files,
        }


class FileBackupSystem:
    """Comprehensive file backup system.
    
    Features:
    - Automatic backup before modifications
    - Organized backup directory structure
    - Manifest-based backup tracking
    - Checksum verification on restore
    - Automatic cleanup of old backups
    
    Example:
        >>> backup_system = FileBackupSystem(project_root="/path/to/project")
        >>> 
        >>> # Create backup before modification
        >>> manifest = backup_system.create_backup(
        ...     operation_id="abc123",
        ...     operation_type="modify_backend_code",
        ...     description="Update simulator function",
        ...     files=["src/backend/simulator.py"]
        ... )
        >>> 
        >>> # ... perform modification ...
        >>> 
        >>> # Restore if needed
        >>> result = backup_system.restore_backup(manifest.operation_id)
    """
    
    # Size threshold for diff-only backup (1MB)
    DIFF_THRESHOLD = 1024 * 1024
    
    # Size threshold for compression (10KB)
    COMPRESS_THRESHOLD = 10 * 1024
    
    # Maximum number of backups to retain
    MAX_BACKUPS = 50
    
    # Maximum age of backups in hours
    MAX_BACKUP_AGE_HOURS = 168  # 1 week
    
    def __init__(
        self,
        project_root: Optional[str] = None,
        backup_dir: Optional[str] = None,
        max_backups: int = MAX_BACKUPS,
    ):
        """Initialize backup system.
        
        Args:
            project_root: Root directory of the project
            backup_dir: Directory for storing backups
            max_backups: Maximum number of backups to retain
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = self.project_root / ".proxima" / "backups"
        
        self.max_backups = max_backups
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileBackupSystem initialized: {self.backup_dir}")
    
    def _compute_checksum(self, content: bytes) -> str:
        """Compute SHA256 checksum."""
        return hashlib.sha256(content).hexdigest()
    
    def _compute_file_checksum(self, path: Path) -> Optional[str]:
        """Compute checksum of a file."""
        try:
            with open(path, "rb") as f:
                return self._compute_checksum(f.read())
        except Exception as e:
            logger.error(f"Failed to compute checksum for {path}: {e}")
            return None
    
    def _get_relative_path(self, path: Path) -> str:
        """Get path relative to project root."""
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)
    
    def _create_snapshot(self, file_path: str) -> Optional[FileSnapshot]:
        """Create a snapshot of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileSnapshot or None if failed
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.project_root / path
            
            if not path.exists():
                logger.warning(f"File not found: {path}")
                return None
            
            stat = path.stat()
            
            with open(path, "rb") as f:
                content_bytes = f.read()
            
            checksum = self._compute_checksum(content_bytes)
            
            # Determine backup type based on size
            if stat.st_size > self.DIFF_THRESHOLD:
                backup_type = BackupType.DIFF
            elif stat.st_size > self.COMPRESS_THRESHOLD:
                backup_type = BackupType.COMPRESSED
            else:
                backup_type = BackupType.FULL
            
            content = content_bytes.decode("utf-8", errors="replace")
            
            return FileSnapshot(
                path=str(path),
                relative_path=self._get_relative_path(path),
                checksum=checksum,
                size=stat.st_size,
                mtime=stat.st_mtime,
                backup_type=backup_type,
                content=content if backup_type == BackupType.FULL else None,
            )
            
        except Exception as e:
            logger.error(f"Failed to create snapshot for {file_path}: {e}")
            return None
    
    def _save_snapshot_file(
        self,
        snapshot: FileSnapshot,
        backup_path: Path,
    ) -> bool:
        """Save snapshot file content to backup location.
        
        Args:
            snapshot: File snapshot
            backup_path: Directory to save to
            
        Returns:
            True if successful
        """
        try:
            # Preserve directory structure
            rel_path = Path(snapshot.relative_path)
            target_dir = backup_path / rel_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            
            source = Path(snapshot.path)
            target = backup_path / rel_path
            
            if snapshot.backup_type == BackupType.COMPRESSED:
                # Save compressed
                target = target.with_suffix(target.suffix + ".gz")
                with open(source, "rb") as f_in:
                    with gzip.open(target, "wb") as f_out:
                        f_out.write(f_in.read())
            else:
                # Save as-is
                shutil.copy2(source, target)
            
            snapshot.backup_path = str(target)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save snapshot file: {e}")
            return False
    
    def create_backup(
        self,
        operation_id: str,
        operation_type: str,
        description: str,
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[BackupManifest]:
        """Create a backup before an operation.
        
        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation
            description: Human-readable description
            files: List of file paths to backup
            metadata: Additional metadata
            
        Returns:
            BackupManifest or None if failed
        """
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Create backup directory for this operation
        backup_path = self.backup_dir / f"{timestamp_str}_{operation_id}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        manifest = BackupManifest(
            operation_id=operation_id,
            timestamp=timestamp.isoformat(),
            operation_type=operation_type,
            description=description,
            metadata=metadata or {},
        )
        
        # Create snapshots and save files
        for file_path in files:
            snapshot = self._create_snapshot(file_path)
            if snapshot:
                if self._save_snapshot_file(snapshot, backup_path):
                    manifest.files.append(snapshot)
                else:
                    logger.warning(f"Failed to backup: {file_path}")
        
        if not manifest.files:
            logger.error("No files were backed up")
            # Clean up empty backup directory
            shutil.rmtree(backup_path, ignore_errors=True)
            return None
        
        # Save manifest
        manifest_path = backup_path / "manifest.json"
        manifest.save(manifest_path)
        
        # Cleanup old backups
        self._cleanup_old_backups()
        
        logger.info(f"Created backup {operation_id} with {len(manifest.files)} files")
        return manifest
    
    def restore_backup(
        self,
        operation_id: str,
        verify_checksum: bool = True,
        create_current_backup: bool = True,
    ) -> RestoreResult:
        """Restore files from a backup.
        
        Args:
            operation_id: Operation ID to restore
            verify_checksum: Verify checksums after restore
            create_current_backup: Create backup of current state first
            
        Returns:
            RestoreResult
        """
        # Find backup directory
        manifest = self._find_manifest(operation_id)
        if not manifest:
            return RestoreResult(
                success=False,
                message=f"Backup not found for operation: {operation_id}",
            )
        
        # Optionally create backup of current state
        if create_current_backup:
            current_files = [f.path for f in manifest.files if Path(f.path).exists()]
            if current_files:
                self.create_backup(
                    operation_id=f"pre_restore_{operation_id[:8]}",
                    operation_type="pre_restore_backup",
                    description=f"Backup before restoring {operation_id}",
                    files=current_files,
                )
        
        restored: List[str] = []
        failed: List[Tuple[str, str]] = []
        
        for snapshot in manifest.files:
            result = self._restore_file(snapshot, verify_checksum)
            if result[0]:
                restored.append(snapshot.path)
            else:
                failed.append((snapshot.path, result[1]))
        
        if failed:
            return RestoreResult(
                success=len(failed) < len(manifest.files),
                message=f"Restored {len(restored)}/{len(manifest.files)} files",
                restored_files=restored,
                failed_files=failed,
            )
        
        return RestoreResult(
            success=True,
            message=f"Successfully restored {len(restored)} files",
            restored_files=restored,
        )
    
    def _restore_file(
        self,
        snapshot: FileSnapshot,
        verify_checksum: bool = True,
    ) -> Tuple[bool, str]:
        """Restore a single file from snapshot.
        
        Args:
            snapshot: File snapshot to restore
            verify_checksum: Verify checksum after restore
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not snapshot.backup_path:
                return False, "No backup path in snapshot"
            
            backup_file = Path(snapshot.backup_path)
            if not backup_file.exists():
                return False, f"Backup file not found: {snapshot.backup_path}"
            
            target = Path(snapshot.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle compressed files
            if backup_file.suffix == ".gz":
                with gzip.open(backup_file, "rb") as f_in:
                    content = f_in.read()
                with open(target, "wb") as f_out:
                    f_out.write(content)
            else:
                shutil.copy2(backup_file, target)
            
            # Verify checksum
            if verify_checksum:
                current_checksum = self._compute_file_checksum(target)
                if current_checksum != snapshot.checksum:
                    return False, "Checksum mismatch after restore"
            
            return True, "Restored successfully"
            
        except Exception as e:
            return False, str(e)
    
    def _find_manifest(self, operation_id: str) -> Optional[BackupManifest]:
        """Find a backup manifest by operation ID."""
        for entry in self.backup_dir.iterdir():
            if entry.is_dir() and operation_id in entry.name:
                manifest_path = entry / "manifest.json"
                if manifest_path.exists():
                    return BackupManifest.load(manifest_path)
        return None
    
    def list_backups(self, limit: int = 20) -> List[BackupManifest]:
        """List available backups.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of manifests (newest first)
        """
        backups: List[Tuple[datetime, BackupManifest]] = []
        
        for entry in self.backup_dir.iterdir():
            if entry.is_dir():
                manifest_path = entry / "manifest.json"
                if manifest_path.exists():
                    manifest = BackupManifest.load(manifest_path)
                    if manifest:
                        try:
                            dt = datetime.fromisoformat(manifest.timestamp)
                            backups.append((dt, manifest))
                        except Exception:
                            pass
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x[0], reverse=True)
        
        return [m for _, m in backups[:limit]]
    
    def get_backup(self, operation_id: str) -> Optional[BackupManifest]:
        """Get a specific backup manifest."""
        return self._find_manifest(operation_id)
    
    def delete_backup(self, operation_id: str) -> bool:
        """Delete a backup.
        
        Args:
            operation_id: Operation ID to delete
            
        Returns:
            True if deleted
        """
        for entry in self.backup_dir.iterdir():
            if entry.is_dir() and operation_id in entry.name:
                try:
                    shutil.rmtree(entry)
                    logger.info(f"Deleted backup: {operation_id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete backup: {e}")
                    return False
        return False
    
    def _cleanup_old_backups(self) -> int:
        """Remove old backups beyond max limit.
        
        Returns:
            Number of backups removed
        """
        backups = self.list_backups(limit=1000)  # Get all
        
        if len(backups) <= self.max_backups:
            return 0
        
        # Remove oldest backups
        removed = 0
        for manifest in backups[self.max_backups:]:
            if self.delete_backup(manifest.operation_id):
                removed += 1
        
        if removed:
            logger.info(f"Cleaned up {removed} old backups")
        
        return removed
    
    def verify_backup(self, operation_id: str) -> Dict[str, Any]:
        """Verify integrity of a backup.
        
        Args:
            operation_id: Operation ID to verify
            
        Returns:
            Verification result
        """
        manifest = self._find_manifest(operation_id)
        if not manifest:
            return {"valid": False, "error": "Backup not found"}
        
        results = {
            "valid": True,
            "operation_id": operation_id,
            "files_checked": 0,
            "files_valid": 0,
            "files_missing": [],
            "files_corrupted": [],
        }
        
        for snapshot in manifest.files:
            results["files_checked"] += 1
            
            if not snapshot.backup_path:
                results["files_missing"].append(snapshot.relative_path)
                results["valid"] = False
                continue
            
            backup_file = Path(snapshot.backup_path)
            if not backup_file.exists():
                results["files_missing"].append(snapshot.relative_path)
                results["valid"] = False
                continue
            
            # Check checksum
            try:
                if backup_file.suffix == ".gz":
                    with gzip.open(backup_file, "rb") as f:
                        content = f.read()
                    current_checksum = self._compute_checksum(content)
                else:
                    current_checksum = self._compute_file_checksum(backup_file)
                
                if current_checksum != snapshot.checksum:
                    results["files_corrupted"].append(snapshot.relative_path)
                    results["valid"] = False
                else:
                    results["files_valid"] += 1
            except Exception as e:
                results["files_corrupted"].append(snapshot.relative_path)
                results["valid"] = False
        
        return results
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Storage statistics
        """
        total_size = 0
        total_files = 0
        backup_count = 0
        
        for entry in self.backup_dir.iterdir():
            if entry.is_dir():
                backup_count += 1
                for file in entry.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
                        total_files += 1
        
        return {
            "backup_count": backup_count,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "backup_directory": str(self.backup_dir),
            "max_backups": self.max_backups,
        }


# Global instance
_backup_system: Optional[FileBackupSystem] = None


def get_file_backup_system(
    project_root: Optional[str] = None,
) -> FileBackupSystem:
    """Get the global FileBackupSystem instance."""
    global _backup_system
    if _backup_system is None:
        _backup_system = FileBackupSystem(project_root=project_root)
    return _backup_system


def create_backup(
    operation_id: str,
    files: List[str],
    description: str = "",
) -> Optional[BackupManifest]:
    """Convenience function to create a backup."""
    return get_file_backup_system().create_backup(
        operation_id=operation_id,
        operation_type="manual",
        description=description,
        files=files,
    )


def restore_backup(operation_id: str) -> RestoreResult:
    """Convenience function to restore a backup."""
    return get_file_backup_system().restore_backup(operation_id)
