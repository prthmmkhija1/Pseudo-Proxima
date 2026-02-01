"""Checkpoint Manager for Undo/Redo/Rollback.

Phase 8: Backend Code Modification with Safety

Provides comprehensive checkpoint management including:
- Checkpoint creation and management
- Undo/redo stack operations
- Selective rollback
- Checkpoint validation
- History management
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque

from proxima.utils.logging import get_logger

logger = get_logger("agent.checkpoint_manager")


@dataclass
class FileState:
    """State of a file at a checkpoint."""
    path: str
    content: Optional[str] = None  # For small files
    checksum: str = ""
    size: int = 0
    exists: bool = True
    backup_path: Optional[str] = None  # For large files
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "checksum": self.checksum,
            "size": self.size,
            "exists": self.exists,
            "backup_path": self.backup_path,
            "has_content": self.content is not None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileState":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            checksum=data.get("checksum", ""),
            size=data.get("size", 0),
            exists=data.get("exists", True),
            backup_path=data.get("backup_path"),
        )


@dataclass
class Checkpoint:
    """A checkpoint representing a point in time."""
    id: str
    timestamp: str
    operation: str
    description: str
    files: List[FileState] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Optional[Dict[str, Any]] = None
    completed: bool = False
    rolled_back: bool = False
    invalidated: bool = False
    
    @property
    def can_rollback(self) -> bool:
        """Check if checkpoint can be rolled back."""
        return self.completed and not self.rolled_back and not self.invalidated
    
    @property
    def file_count(self) -> int:
        """Number of files in checkpoint."""
        return len(self.files)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "description": self.description,
            "files": [f.to_dict() for f in self.files],
            "metadata": self.metadata,
            "completed": self.completed,
            "rolled_back": self.rolled_back,
            "invalidated": self.invalidated,
            "can_rollback": self.can_rollback,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        checkpoint = cls(
            id=data["id"],
            timestamp=data["timestamp"],
            operation=data["operation"],
            description=data["description"],
            metadata=data.get("metadata", {}),
            completed=data.get("completed", False),
            rolled_back=data.get("rolled_back", False),
            invalidated=data.get("invalidated", False),
        )
        checkpoint.files = [FileState.from_dict(f) for f in data.get("files", [])]
        return checkpoint


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    message: str
    checkpoint_id: str
    restored_files: List[str] = field(default_factory=list)
    failed_files: List[Tuple[str, str]] = field(default_factory=list)
    changes_lost: bool = False  # If rollback lost recent changes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "checkpoint_id": self.checkpoint_id,
            "restored_files": self.restored_files,
            "failed_files": self.failed_files,
            "changes_lost": self.changes_lost,
        }


@dataclass
class UndoRedoState:
    """State of undo/redo stacks."""
    can_undo: bool = False
    can_redo: bool = False
    undo_description: Optional[str] = None
    redo_description: Optional[str] = None
    undo_count: int = 0
    redo_count: int = 0


class CheckpointManager:
    """Manages checkpoints for undo/redo/rollback.
    
    Features:
    - Create checkpoints before operations
    - Undo/redo with separate stacks
    - Selective rollback to any checkpoint
    - File change detection
    - Automatic checkpoint cleanup
    
    Example:
        >>> manager = CheckpointManager()
        >>> 
        >>> # Create checkpoint before modification
        >>> checkpoint = manager.create_checkpoint(
        ...     operation="modify_code",
        ...     description="Update function signature",
        ...     files=["src/backend.py"]
        ... )
        >>> 
        >>> # ... perform modification ...
        >>> manager.complete_checkpoint(checkpoint.id)
        >>> 
        >>> # Undo if needed
        >>> result = manager.undo()
        >>> 
        >>> # Or redo
        >>> result = manager.redo()
    """
    
    # Size threshold for storing content vs backup file (1MB)
    CONTENT_SIZE_THRESHOLD = 1024 * 1024
    
    # Maximum checkpoints in each stack
    MAX_STACK_SIZE = 20
    
    # Maximum total checkpoints
    MAX_CHECKPOINTS = 100
    
    def __init__(
        self,
        backup_dir: Optional[str] = None,
        max_stack_size: int = MAX_STACK_SIZE,
        max_checkpoints: int = MAX_CHECKPOINTS,
    ):
        """Initialize checkpoint manager.
        
        Args:
            backup_dir: Directory for checkpoint backups
            max_stack_size: Maximum size of undo/redo stacks
            max_checkpoints: Maximum total checkpoints
        """
        self.backup_dir = Path(backup_dir) if backup_dir else Path.home() / ".proxima" / "checkpoints"
        self.max_stack_size = max_stack_size
        self.max_checkpoints = max_checkpoints
        
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._undo_stack: Deque[str] = deque(maxlen=max_stack_size)
        self._redo_stack: Deque[str] = deque(maxlen=max_stack_size)
        self._lock = threading.Lock()
        self._checkpoint_counter = 0
        
        # Callbacks
        self._on_checkpoint_created: Optional[Callable[[Checkpoint], None]] = None
        self._on_rollback: Optional[Callable[[RollbackResult], None]] = None
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager initialized: {self.backup_dir}")
    
    def _generate_id(self) -> str:
        """Generate unique checkpoint ID."""
        self._checkpoint_counter += 1
        timestamp = int(time.time() * 1000)
        return f"ckpt_{timestamp}_{self._checkpoint_counter}"
    
    def _compute_checksum(self, content: str) -> str:
        """Compute SHA256 checksum."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def _capture_file_state(self, file_path: str) -> Optional[FileState]:
        """Capture current state of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileState or None if failed
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return FileState(
                    path=str(path),
                    exists=False,
                    checksum="",
                    size=0,
                )
            
            content = path.read_text(encoding="utf-8")
            size = len(content.encode("utf-8"))
            checksum = self._compute_checksum(content)
            
            state = FileState(
                path=str(path),
                checksum=checksum,
                size=size,
                exists=True,
            )
            
            # Store content for small files, backup path for large files
            if size <= self.CONTENT_SIZE_THRESHOLD:
                state.content = content
            else:
                # Create backup file
                backup_name = f"{checksum[:16]}_{path.name}"
                backup_path = self.backup_dir / backup_name
                shutil.copy2(path, backup_path)
                state.backup_path = str(backup_path)
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to capture file state: {file_path}: {e}")
            return None
    
    def _restore_file_state(self, state: FileState) -> Tuple[bool, str]:
        """Restore a file to its captured state.
        
        Args:
            state: File state to restore
            
        Returns:
            Tuple of (success, message)
        """
        try:
            path = Path(state.path)
            
            if not state.exists:
                # File should not exist - delete if it does
                if path.exists():
                    path.unlink()
                return True, f"Deleted {state.path}"
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if state.content is not None:
                path.write_text(state.content, encoding="utf-8")
            elif state.backup_path:
                shutil.copy2(state.backup_path, path)
            else:
                return False, f"No content or backup for {state.path}"
            
            # Verify checksum
            current_checksum = self._compute_checksum(path.read_text(encoding="utf-8"))
            if current_checksum != state.checksum:
                return False, f"Checksum mismatch after restore: {state.path}"
            
            return True, f"Restored {state.path}"
            
        except Exception as e:
            return False, str(e)
    
    def _check_file_modified(self, state: FileState) -> bool:
        """Check if file has been modified since checkpoint.
        
        Args:
            state: File state from checkpoint
            
        Returns:
            True if modified
        """
        try:
            path = Path(state.path)
            
            if not path.exists():
                return state.exists  # Modified if it existed before
            
            if not state.exists:
                return True  # Modified if it exists now but didn't before
            
            current_content = path.read_text(encoding="utf-8")
            current_checksum = self._compute_checksum(current_content)
            
            return current_checksum != state.checksum
            
        except Exception:
            return True  # Assume modified on error
    
    def create_checkpoint(
        self,
        operation: str,
        description: str,
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """Create a checkpoint before an operation.
        
        Args:
            operation: Operation identifier
            description: Human-readable description
            files: Files to capture
            metadata: Additional metadata
            
        Returns:
            Created checkpoint
        """
        with self._lock:
            # Cleanup if needed
            if len(self._checkpoints) >= self.max_checkpoints:
                self._cleanup_old_checkpoints()
            
            checkpoint = Checkpoint(
                id=self._generate_id(),
                timestamp=datetime.now().isoformat(),
                operation=operation,
                description=description,
                metadata=metadata or {},
            )
            
            # Capture file states
            for file_path in files:
                state = self._capture_file_state(file_path)
                if state:
                    checkpoint.files.append(state)
            
            self._checkpoints[checkpoint.id] = checkpoint
            
            logger.info(f"Created checkpoint {checkpoint.id}: {operation}")
            
            if self._on_checkpoint_created:
                self._on_checkpoint_created(checkpoint)
            
            return checkpoint
    
    def complete_checkpoint(
        self,
        checkpoint_id: str,
        state_after: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark checkpoint as complete and add to undo stack.
        
        Args:
            checkpoint_id: Checkpoint ID
            state_after: State after operation
            
        Returns:
            True if completed
        """
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return False
            
            checkpoint.completed = True
            checkpoint.state_after = state_after
            
            # Add to undo stack
            self._undo_stack.append(checkpoint_id)
            
            # Clear redo stack on new operation
            self._redo_stack.clear()
            
            logger.info(f"Completed checkpoint {checkpoint_id}")
            return True
    
    def undo(self) -> RollbackResult:
        """Undo the last operation.
        
        Returns:
            RollbackResult
        """
        with self._lock:
            if not self._undo_stack:
                return RollbackResult(
                    success=False,
                    message="Nothing to undo",
                    checkpoint_id="",
                )
            
            checkpoint_id = self._undo_stack.pop()
            checkpoint = self._checkpoints.get(checkpoint_id)
            
            if not checkpoint:
                return RollbackResult(
                    success=False,
                    message=f"Checkpoint not found: {checkpoint_id}",
                    checkpoint_id=checkpoint_id,
                )
            
            # Check for modifications since checkpoint
            modified_files = [
                state.path for state in checkpoint.files
                if self._check_file_modified(state)
            ]
            
            # Capture current state for redo
            redo_checkpoint = self._create_redo_checkpoint(checkpoint)
            
            # Restore files
            result = self._restore_checkpoint(checkpoint)
            
            if result.success:
                # Add to redo stack
                self._redo_stack.append(redo_checkpoint.id)
                checkpoint.rolled_back = True
                result.changes_lost = len(modified_files) > 0
            
            if self._on_rollback:
                self._on_rollback(result)
            
            return result
    
    def redo(self) -> RollbackResult:
        """Redo the last undone operation.
        
        Returns:
            RollbackResult
        """
        with self._lock:
            if not self._redo_stack:
                return RollbackResult(
                    success=False,
                    message="Nothing to redo",
                    checkpoint_id="",
                )
            
            checkpoint_id = self._redo_stack.pop()
            checkpoint = self._checkpoints.get(checkpoint_id)
            
            if not checkpoint:
                return RollbackResult(
                    success=False,
                    message=f"Checkpoint not found: {checkpoint_id}",
                    checkpoint_id=checkpoint_id,
                )
            
            # Create undo checkpoint for this redo
            undo_checkpoint = self._create_undo_checkpoint(checkpoint)
            
            # Restore files
            result = self._restore_checkpoint(checkpoint)
            
            if result.success:
                # Add back to undo stack
                self._undo_stack.append(undo_checkpoint.id)
                checkpoint.rolled_back = False
            
            if self._on_rollback:
                self._on_rollback(result)
            
            return result
    
    def _create_redo_checkpoint(self, original: Checkpoint) -> Checkpoint:
        """Create checkpoint for redo operation."""
        checkpoint = Checkpoint(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            operation=f"redo_{original.operation}",
            description=f"Redo: {original.description}",
            metadata={"original_checkpoint": original.id},
        )
        
        # Capture current state of files
        for file_state in original.files:
            current_state = self._capture_file_state(file_state.path)
            if current_state:
                checkpoint.files.append(current_state)
        
        checkpoint.completed = True
        self._checkpoints[checkpoint.id] = checkpoint
        
        return checkpoint
    
    def _create_undo_checkpoint(self, original: Checkpoint) -> Checkpoint:
        """Create checkpoint for undo operation after redo."""
        checkpoint = Checkpoint(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            operation=f"undo_{original.operation}",
            description=f"Undo: {original.description}",
            metadata={"original_checkpoint": original.id},
        )
        
        # Capture current state
        for file_state in original.files:
            current_state = self._capture_file_state(file_state.path)
            if current_state:
                checkpoint.files.append(current_state)
        
        checkpoint.completed = True
        self._checkpoints[checkpoint.id] = checkpoint
        
        return checkpoint
    
    def _restore_checkpoint(self, checkpoint: Checkpoint) -> RollbackResult:
        """Restore files from a checkpoint."""
        restored: List[str] = []
        failed: List[Tuple[str, str]] = []
        
        for state in checkpoint.files:
            success, message = self._restore_file_state(state)
            if success:
                restored.append(state.path)
            else:
                failed.append((state.path, message))
        
        if failed:
            return RollbackResult(
                success=len(failed) < len(checkpoint.files),
                message=f"Restored {len(restored)}/{len(checkpoint.files)} files",
                checkpoint_id=checkpoint.id,
                restored_files=restored,
                failed_files=failed,
            )
        
        return RollbackResult(
            success=True,
            message=f"Successfully restored {len(restored)} files",
            checkpoint_id=checkpoint.id,
            restored_files=restored,
        )
    
    def rollback_to(self, checkpoint_id: str) -> RollbackResult:
        """Rollback to a specific checkpoint.
        
        Args:
            checkpoint_id: Target checkpoint ID
            
        Returns:
            RollbackResult
        """
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            
            if not checkpoint:
                return RollbackResult(
                    success=False,
                    message=f"Checkpoint not found: {checkpoint_id}",
                    checkpoint_id=checkpoint_id,
                )
            
            if not checkpoint.can_rollback:
                return RollbackResult(
                    success=False,
                    message="Checkpoint cannot be rolled back",
                    checkpoint_id=checkpoint_id,
                )
            
            # Create backup of current state
            backup = self.create_checkpoint(
                operation="pre_rollback_backup",
                description=f"Backup before rollback to {checkpoint_id}",
                files=[state.path for state in checkpoint.files],
            )
            backup.completed = True
            
            # Restore checkpoint
            result = self._restore_checkpoint(checkpoint)
            
            if result.success:
                checkpoint.rolled_back = True
                # Invalidate newer checkpoints
                self._invalidate_newer_checkpoints(checkpoint.timestamp)
            
            if self._on_rollback:
                self._on_rollback(result)
            
            return result
    
    def _invalidate_newer_checkpoints(self, timestamp: str) -> None:
        """Invalidate checkpoints newer than timestamp."""
        for checkpoint in self._checkpoints.values():
            if checkpoint.timestamp > timestamp:
                checkpoint.invalidated = True
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints."""
        # Keep only most recent checkpoints
        sorted_checkpoints = sorted(
            self._checkpoints.items(),
            key=lambda x: x[1].timestamp,
            reverse=True,
        )
        
        to_remove = sorted_checkpoints[self.max_checkpoints:]
        
        for checkpoint_id, checkpoint in to_remove:
            # Remove backup files
            for state in checkpoint.files:
                if state.backup_path:
                    try:
                        Path(state.backup_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            
            del self._checkpoints[checkpoint_id]
    
    def get_undo_redo_state(self) -> UndoRedoState:
        """Get current undo/redo state."""
        undo_desc = None
        redo_desc = None
        
        if self._undo_stack:
            undo_checkpoint = self._checkpoints.get(self._undo_stack[-1])
            if undo_checkpoint:
                undo_desc = undo_checkpoint.description
        
        if self._redo_stack:
            redo_checkpoint = self._checkpoints.get(self._redo_stack[-1])
            if redo_checkpoint:
                redo_desc = redo_checkpoint.description
        
        return UndoRedoState(
            can_undo=len(self._undo_stack) > 0,
            can_redo=len(self._redo_stack) > 0,
            undo_description=undo_desc,
            redo_description=redo_desc,
            undo_count=len(self._undo_stack),
            redo_count=len(self._redo_stack),
        )
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self, limit: int = 20) -> List[Checkpoint]:
        """List checkpoints (newest first)."""
        sorted_checkpoints = sorted(
            self._checkpoints.values(),
            key=lambda x: x.timestamp,
            reverse=True,
        )
        return sorted_checkpoints[:limit]
    
    def get_history(self) -> List[Checkpoint]:
        """Get undo history."""
        return [
            self._checkpoints[cid]
            for cid in reversed(self._undo_stack)
            if cid in self._checkpoints
        ]
    
    def clear_history(self) -> None:
        """Clear all checkpoints and history."""
        with self._lock:
            # Remove backup files
            for checkpoint in self._checkpoints.values():
                for state in checkpoint.files:
                    if state.backup_path:
                        try:
                            Path(state.backup_path).unlink(missing_ok=True)
                        except Exception:
                            pass
            
            self._checkpoints.clear()
            self._undo_stack.clear()
            self._redo_stack.clear()
            
            logger.info("Cleared checkpoint history")
    
    def set_callbacks(
        self,
        on_checkpoint_created: Optional[Callable[[Checkpoint], None]] = None,
        on_rollback: Optional[Callable[[RollbackResult], None]] = None,
    ) -> None:
        """Set callback functions."""
        self._on_checkpoint_created = on_checkpoint_created
        self._on_rollback = on_rollback


# Global instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the global CheckpointManager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager


def create_checkpoint(
    operation: str,
    files: List[str],
    description: str = "",
) -> Checkpoint:
    """Convenience function to create a checkpoint."""
    return get_checkpoint_manager().create_checkpoint(
        operation=operation,
        description=description,
        files=files,
    )


def undo() -> RollbackResult:
    """Convenience function to undo."""
    return get_checkpoint_manager().undo()


def redo() -> RollbackResult:
    """Convenience function to redo."""
    return get_checkpoint_manager().redo()
