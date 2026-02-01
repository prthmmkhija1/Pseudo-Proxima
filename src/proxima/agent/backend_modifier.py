"""Backend Code Modifier for Proxima Agent.

Provides safe code modification capabilities for quantum backends:
- Code search and replace
- Line-based insertions
- Safe deletions with backups
- Undo/redo support via rollback manager

Supports backends:
- LRET variants (cirq_scalability, pennylane_hybrid, phase_7_unified)
- Cirq, Qiskit Aer, QuEST, qsim, cuQuantum
- Custom backends

All modifications create checkpoints for rollback capability.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger
from .safety import RollbackManager, OperationCheckpoint

logger = get_logger("agent.backend_modifier")


class ModificationType(Enum):
    """Types of code modifications."""
    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"
    APPEND = "append"
    PREPEND = "prepend"


@dataclass
class CodeChange:
    """Represents a code change."""
    id: str
    file_path: str
    modification_type: ModificationType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    line_number: Optional[int] = None
    line_range: Optional[Tuple[int, int]] = None  # (start, end) inclusive
    context_before: str = ""  # Lines before change for context
    context_after: str = ""   # Lines after change for context
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    applied: bool = False
    checkpoint_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "modification_type": self.modification_type.value,
            "old_content": self.old_content[:200] + "..." if self.old_content and len(self.old_content) > 200 else self.old_content,
            "new_content": self.new_content[:200] + "..." if self.new_content and len(self.new_content) > 200 else self.new_content,
            "line_number": self.line_number,
            "line_range": self.line_range,
            "timestamp": self.timestamp,
            "applied": self.applied,
            "checkpoint_id": self.checkpoint_id,
        }
    
    def get_diff(self) -> str:
        """Get unified diff of the change."""
        if not self.old_content or not self.new_content:
            return ""
        
        old_lines = self.old_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{os.path.basename(self.file_path)}",
            tofile=f"b/{os.path.basename(self.file_path)}",
        )
        return "".join(diff)


@dataclass
class ModificationResult:
    """Result of a modification operation."""
    success: bool
    change: Optional[CodeChange] = None
    message: str = ""
    error: Optional[str] = None
    diff: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "change": self.change.to_dict() if self.change else None,
            "message": self.message,
            "error": self.error,
            "diff_preview": self.diff[:500] if self.diff else None,
        }


class BackendModifier:
    """Modify backend source code with safety features.
    
    Provides:
    - Safe code modifications with backup/rollback
    - Search and replace across files
    - Line-based insertions and deletions
    - Diff preview before applying changes
    - Integration with rollback manager
    
    Example:
        >>> modifier = BackendModifier(rollback_manager)
        >>> 
        >>> # Preview a change
        >>> result = modifier.preview_replace(
        ...     "/path/to/backend.py",
        ...     "old_code",
        ...     "new_code"
        ... )
        >>> print(result.diff)
        >>> 
        >>> # Apply the change
        >>> result = modifier.apply_change(result.change)
        >>> 
        >>> # Rollback if needed
        >>> modifier.rollback(result.change.id)
    """
    
    # Known backend directories (relative to project root)
    BACKEND_PATHS = {
        "lret_cirq_scalability": "src/proxima/backends/lret_cirq_scalability",
        "lret_pennylane_hybrid": "src/proxima/backends/lret_pennylane_hybrid",
        "lret_phase_7_unified": "src/proxima/backends/lret_phase_7_unified",
        "cirq": "src/proxima/backends/cirq",
        "qiskit_aer": "src/proxima/backends/qiskit_aer",
        "quest": "src/proxima/backends/quest",
        "qsim": "src/proxima/backends/qsim",
        "cuquantum": "src/proxima/backends/cuquantum",
    }
    
    def __init__(
        self,
        rollback_manager: Optional[RollbackManager] = None,
        project_root: Optional[str] = None,
        create_backup: bool = True,
    ):
        """Initialize backend modifier.
        
        Args:
            rollback_manager: Rollback manager for checkpoints
            project_root: Root directory of the project
            create_backup: Always create backups before modifications
        """
        self.rollback_manager = rollback_manager or RollbackManager()
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.create_backup = create_backup
        
        self._changes: Dict[str, CodeChange] = {}
        self._change_counter = 0
        
        logger.info(f"BackendModifier initialized with project root: {self.project_root}")
    
    def _generate_change_id(self) -> str:
        """Generate unique change ID."""
        self._change_counter += 1
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"change_{timestamp}_{self._change_counter}"
    
    def _read_file(self, path: str) -> Tuple[str, bool]:
        """Read file content.
        
        Args:
            path: Path to file
            
        Returns:
            Tuple of (content, success)
        """
        try:
            abs_path = Path(path) if Path(path).is_absolute() else self.project_root / path
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read(), True
        except Exception as e:
            return str(e), False
    
    def _write_file(self, path: str, content: str) -> Tuple[str, bool]:
        """Write content to file.
        
        Args:
            path: Path to file
            content: Content to write
            
        Returns:
            Tuple of (message, success)
        """
        try:
            abs_path = Path(path) if Path(path).is_absolute() else self.project_root / path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written to {abs_path}", True
        except Exception as e:
            return str(e), False
    
    def _get_context(
        self,
        lines: List[str],
        start_line: int,
        end_line: int,
        context_lines: int = 3,
    ) -> Tuple[str, str]:
        """Get context lines before and after a range.
        
        Args:
            lines: All file lines
            start_line: Start of range (1-indexed)
            end_line: End of range (1-indexed)
            context_lines: Number of context lines
            
        Returns:
            Tuple of (context_before, context_after)
        """
        start_idx = start_line - 1
        end_idx = end_line
        
        before_start = max(0, start_idx - context_lines)
        after_end = min(len(lines), end_idx + context_lines)
        
        context_before = "\n".join(lines[before_start:start_idx])
        context_after = "\n".join(lines[end_idx:after_end])
        
        return context_before, context_after
    
    def get_backend_path(self, backend_name: str) -> Optional[Path]:
        """Get the path to a backend directory.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            Path to backend directory, or None if not found
        """
        normalized = backend_name.lower().replace("-", "_").replace(" ", "_")
        
        if normalized in self.BACKEND_PATHS:
            path = self.project_root / self.BACKEND_PATHS[normalized]
            if path.exists():
                return path
        
        # Try to find in backends directory
        backends_dir = self.project_root / "src" / "proxima" / "backends"
        if backends_dir.exists():
            for entry in backends_dir.iterdir():
                if entry.is_dir() and normalized in entry.name.lower():
                    return entry
        
        return None
    
    def list_backend_files(
        self,
        backend_name: str,
        pattern: str = "*.py",
    ) -> List[str]:
        """List files in a backend directory.
        
        Args:
            backend_name: Name of the backend
            pattern: Glob pattern for files
            
        Returns:
            List of file paths (relative to project root)
        """
        backend_path = self.get_backend_path(backend_name)
        if not backend_path:
            return []
        
        files = []
        for path in backend_path.rglob(pattern):
            if path.is_file():
                try:
                    rel_path = path.relative_to(self.project_root)
                    files.append(str(rel_path))
                except ValueError:
                    files.append(str(path))
        
        return sorted(files)
    
    def search_in_file(
        self,
        file_path: str,
        pattern: str,
        is_regex: bool = False,
        context_lines: int = 2,
    ) -> List[Dict[str, Any]]:
        """Search for pattern in a file.
        
        Args:
            file_path: Path to file
            pattern: Search pattern
            is_regex: Whether pattern is regex
            context_lines: Context lines to include
            
        Returns:
            List of matches with context
        """
        content, success = self._read_file(file_path)
        if not success:
            return []
        
        lines = content.split("\n")
        matches = []
        
        if is_regex:
            regex = re.compile(pattern)
            for i, line in enumerate(lines):
                if regex.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    matches.append({
                        "line_number": i + 1,
                        "line": line,
                        "context": "\n".join(lines[start:end]),
                        "match": regex.search(line).group() if regex.search(line) else pattern,
                    })
        else:
            for i, line in enumerate(lines):
                if pattern in line:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    matches.append({
                        "line_number": i + 1,
                        "line": line,
                        "context": "\n".join(lines[start:end]),
                        "match": pattern,
                    })
        
        return matches
    
    def search_in_backend(
        self,
        backend_name: str,
        pattern: str,
        is_regex: bool = False,
        file_pattern: str = "*.py",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search for pattern across all backend files.
        
        Args:
            backend_name: Name of the backend
            pattern: Search pattern
            is_regex: Whether pattern is regex
            file_pattern: File glob pattern
            
        Returns:
            Dict of file path -> list of matches
        """
        results = {}
        
        for file_path in self.list_backend_files(backend_name, file_pattern):
            matches = self.search_in_file(file_path, pattern, is_regex)
            if matches:
                results[file_path] = matches
        
        return results
    
    def preview_replace(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        occurrence: int = 0,  # 0 = all, 1 = first, -1 = last
    ) -> ModificationResult:
        """Preview a replace operation without applying.
        
        Args:
            file_path: Path to file
            old_content: Content to find
            new_content: Content to replace with
            occurrence: Which occurrence (0=all, 1=first, -1=last)
            
        Returns:
            Modification result with diff preview
        """
        content, success = self._read_file(file_path)
        if not success:
            return ModificationResult(
                success=False,
                error=f"Failed to read file: {content}",
            )
        
        if old_content not in content:
            return ModificationResult(
                success=False,
                error=f"Content not found in file",
            )
        
        # Count occurrences
        count = content.count(old_content)
        
        # Perform replacement
        if occurrence == 0:
            new_file_content = content.replace(old_content, new_content)
        elif occurrence == 1:
            new_file_content = content.replace(old_content, new_content, 1)
        elif occurrence == -1:
            # Replace last occurrence
            parts = content.rsplit(old_content, 1)
            new_file_content = new_content.join(parts)
        else:
            # Replace nth occurrence
            parts = content.split(old_content)
            if len(parts) <= occurrence:
                return ModificationResult(
                    success=False,
                    error=f"Only {count} occurrences found",
                )
            new_file_content = old_content.join(parts[:occurrence]) + new_content + old_content.join(parts[occurrence:])
        
        # Create change object
        change = CodeChange(
            id=self._generate_change_id(),
            file_path=file_path,
            modification_type=ModificationType.REPLACE,
            old_content=content,
            new_content=new_file_content,
        )
        
        return ModificationResult(
            success=True,
            change=change,
            message=f"Preview: {count} occurrence(s) would be replaced",
            diff=change.get_diff(),
        )
    
    def preview_insert(
        self,
        file_path: str,
        content: str,
        line_number: int,
        after: bool = True,
    ) -> ModificationResult:
        """Preview an insert operation.
        
        Args:
            file_path: Path to file
            content: Content to insert
            line_number: Line number for insertion
            after: Insert after line (True) or before (False)
            
        Returns:
            Modification result with diff preview
        """
        file_content, success = self._read_file(file_path)
        if not success:
            return ModificationResult(
                success=False,
                error=f"Failed to read file: {file_content}",
            )
        
        lines = file_content.split("\n")
        
        if line_number < 1 or line_number > len(lines) + 1:
            return ModificationResult(
                success=False,
                error=f"Line number {line_number} out of range (1-{len(lines)})",
            )
        
        insert_idx = line_number if after else line_number - 1
        
        # Get context
        context_before, context_after = self._get_context(lines, line_number, line_number)
        
        # Create new content
        new_lines = lines[:insert_idx] + content.split("\n") + lines[insert_idx:]
        new_file_content = "\n".join(new_lines)
        
        change = CodeChange(
            id=self._generate_change_id(),
            file_path=file_path,
            modification_type=ModificationType.INSERT,
            old_content=file_content,
            new_content=new_file_content,
            line_number=line_number,
            context_before=context_before,
            context_after=context_after,
        )
        
        return ModificationResult(
            success=True,
            change=change,
            message=f"Preview: Content would be inserted at line {line_number}",
            diff=change.get_diff(),
        )
    
    def preview_delete(
        self,
        file_path: str,
        start_line: int,
        end_line: Optional[int] = None,
    ) -> ModificationResult:
        """Preview a delete operation.
        
        Args:
            file_path: Path to file
            start_line: Start line (1-indexed)
            end_line: End line (inclusive, defaults to start_line)
            
        Returns:
            Modification result with diff preview
        """
        file_content, success = self._read_file(file_path)
        if not success:
            return ModificationResult(
                success=False,
                error=f"Failed to read file: {file_content}",
            )
        
        lines = file_content.split("\n")
        end_line = end_line or start_line
        
        if start_line < 1 or end_line > len(lines):
            return ModificationResult(
                success=False,
                error=f"Line range {start_line}-{end_line} out of bounds (1-{len(lines)})",
            )
        
        # Get context
        context_before, context_after = self._get_context(lines, start_line, end_line)
        
        # Delete lines
        deleted = "\n".join(lines[start_line-1:end_line])
        new_lines = lines[:start_line-1] + lines[end_line:]
        new_file_content = "\n".join(new_lines)
        
        change = CodeChange(
            id=self._generate_change_id(),
            file_path=file_path,
            modification_type=ModificationType.DELETE,
            old_content=file_content,
            new_content=new_file_content,
            line_range=(start_line, end_line),
            context_before=context_before,
            context_after=context_after,
        )
        
        return ModificationResult(
            success=True,
            change=change,
            message=f"Preview: Lines {start_line}-{end_line} would be deleted ({end_line - start_line + 1} lines)",
            diff=change.get_diff(),
        )
    
    def preview_append(
        self,
        file_path: str,
        content: str,
    ) -> ModificationResult:
        """Preview appending content to end of file.
        
        Args:
            file_path: Path to file
            content: Content to append
            
        Returns:
            Modification result with diff preview
        """
        file_content, success = self._read_file(file_path)
        if not success:
            return ModificationResult(
                success=False,
                error=f"Failed to read file: {file_content}",
            )
        
        new_file_content = file_content + "\n" + content if file_content else content
        
        change = CodeChange(
            id=self._generate_change_id(),
            file_path=file_path,
            modification_type=ModificationType.APPEND,
            old_content=file_content,
            new_content=new_file_content,
        )
        
        return ModificationResult(
            success=True,
            change=change,
            message="Preview: Content would be appended to file",
            diff=change.get_diff(),
        )
    
    def apply_change(self, change: CodeChange) -> ModificationResult:
        """Apply a previewed change.
        
        Args:
            change: Change to apply
            
        Returns:
            Modification result
        """
        if change.applied:
            return ModificationResult(
                success=False,
                change=change,
                error="Change has already been applied",
            )
        
        if not change.new_content:
            return ModificationResult(
                success=False,
                change=change,
                error="No new content to apply",
            )
        
        # Create checkpoint
        abs_path = Path(change.file_path) if Path(change.file_path).is_absolute() else self.project_root / change.file_path
        
        checkpoint = self.rollback_manager.create_checkpoint(
            operation=f"modify_{change.modification_type.value}",
            files=[str(abs_path)],
            state={"old_content": change.old_content},
        )
        change.checkpoint_id = checkpoint.id
        
        # Apply change
        message, success = self._write_file(str(abs_path), change.new_content)
        
        if success:
            change.applied = True
            self._changes[change.id] = change
            
            # Complete checkpoint with new state
            self.rollback_manager.complete_checkpoint(
                checkpoint.id,
                state_after={str(abs_path): change.new_content},
            )
            
            logger.info(f"Applied change {change.id} to {change.file_path}")
            
            return ModificationResult(
                success=True,
                change=change,
                message=f"Change applied successfully",
                diff=change.get_diff(),
            )
        else:
            return ModificationResult(
                success=False,
                change=change,
                error=f"Failed to apply change: {message}",
            )
    
    def rollback_change(self, change_id: str) -> ModificationResult:
        """Rollback a specific change.
        
        Args:
            change_id: ID of change to rollback
            
        Returns:
            Modification result
        """
        change = self._changes.get(change_id)
        if not change:
            return ModificationResult(
                success=False,
                error=f"Change not found: {change_id}",
            )
        
        if not change.applied:
            return ModificationResult(
                success=False,
                change=change,
                error="Change was not applied",
            )
        
        if change.checkpoint_id:
            success, message = self.rollback_manager.rollback(change.checkpoint_id)
            if success:
                change.applied = False
                return ModificationResult(
                    success=True,
                    change=change,
                    message=f"Rolled back: {message}",
                )
            else:
                return ModificationResult(
                    success=False,
                    change=change,
                    error=f"Rollback failed: {message}",
                )
        else:
            # Manual rollback using old_content
            if change.old_content:
                message, success = self._write_file(change.file_path, change.old_content)
                if success:
                    change.applied = False
                    return ModificationResult(
                        success=True,
                        change=change,
                        message="Rolled back manually",
                    )
                else:
                    return ModificationResult(
                        success=False,
                        change=change,
                        error=f"Manual rollback failed: {message}",
                    )
        
        return ModificationResult(
            success=False,
            change=change,
            error="Unable to rollback - no checkpoint or old content",
        )
    
    def undo(self) -> ModificationResult:
        """Undo the last modification."""
        success, message = self.rollback_manager.undo()
        return ModificationResult(
            success=success,
            message=message if success else "",
            error=message if not success else None,
        )
    
    def redo(self) -> ModificationResult:
        """Redo the last undone modification."""
        success, message = self.rollback_manager.redo()
        return ModificationResult(
            success=success,
            message=message if success else "",
            error=message if not success else None,
        )
    
    def get_change(self, change_id: str) -> Optional[CodeChange]:
        """Get a change by ID."""
        return self._changes.get(change_id)
    
    def list_changes(self, limit: int = 20) -> List[CodeChange]:
        """List recent changes.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of changes (newest first)
        """
        changes = sorted(
            self._changes.values(),
            key=lambda c: c.timestamp,
            reverse=True,
        )
        return changes[:limit]
    
    def get_file_diff(
        self,
        file_path: str,
        original_content: Optional[str] = None,
    ) -> str:
        """Get diff between original and current file content.
        
        Args:
            file_path: Path to file
            original_content: Original content (reads from last backup if not provided)
            
        Returns:
            Unified diff string
        """
        current, success = self._read_file(file_path)
        if not success:
            return f"Error reading file: {current}"
        
        if original_content is None:
            # Try to get from most recent change
            for change in reversed(list(self._changes.values())):
                if change.file_path == file_path and change.old_content:
                    original_content = change.old_content
                    break
        
        if original_content is None:
            return "No original content available for comparison"
        
        old_lines = original_content.splitlines(keepends=True)
        new_lines = current.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{os.path.basename(file_path)}",
            tofile=f"b/{os.path.basename(file_path)}",
        )
        return "".join(diff)
