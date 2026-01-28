"""Change Tracker for Backend Wizard.

Tracks all modifications made during backend generation.
Provides undo/redo functionality and change export.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import difflib
import json
import logging

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of file change."""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


@dataclass
class FileChange:
    """Represents a change to a file.
    
    Tracks the before/after state of a file for diff generation
    and undo/redo functionality.
    """
    
    file_path: str
    change_type: ChangeType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    approved: bool = False
    ai_generated: bool = True
    
    @property
    def lines_added(self) -> int:
        """Count lines added."""
        if self.change_type == ChangeType.CREATE:
            return len(self.new_content.split('\n')) if self.new_content else 0
        elif self.old_content and self.new_content:
            diff = list(difflib.unified_diff(
                self.old_content.split('\n'),
                self.new_content.split('\n')
            ))
            return sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        return 0
    
    @property
    def lines_removed(self) -> int:
        """Count lines removed."""
        if self.change_type == ChangeType.DELETE:
            return len(self.old_content.split('\n')) if self.old_content else 0
        elif self.old_content and self.new_content:
            diff = list(difflib.unified_diff(
                self.old_content.split('\n'),
                self.new_content.split('\n')
            ))
            return sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        return 0
    
    def get_unified_diff(self) -> str:
        """Get unified diff output.
        
        Returns:
            Unified diff string for display
        """
        if self.change_type == ChangeType.CREATE:
            lines = [
                f"New file: {self.file_path}",
                "=" * 60,
            ]
            if self.new_content:
                for i, line in enumerate(self.new_content.split('\n'), 1):
                    lines.append(f"+{i:4d} | {line}")
            return '\n'.join(lines)
        
        elif self.change_type == ChangeType.DELETE:
            lines = [
                f"Deleted file: {self.file_path}",
                "=" * 60,
            ]
            if self.old_content:
                for i, line in enumerate(self.old_content.split('\n'), 1):
                    lines.append(f"-{i:4d} | {line}")
            return '\n'.join(lines)
        
        else:
            # Modified file - show unified diff
            old_lines = self.old_content.split('\n') if self.old_content else []
            new_lines = self.new_content.split('\n') if self.new_content else []
            diff = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{self.file_path}",
                tofile=f"b/{self.file_path}",
                lineterm=''
            )
            return '\n'.join(diff)
    
    def get_side_by_side_diff(self, width: int = 80) -> str:
        """Get side-by-side diff output.
        
        Args:
            width: Total width for the diff display
            
        Returns:
            Side-by-side diff string
        """
        if not self.old_content and not self.new_content:
            return "No content"
        
        old_lines = self.old_content.split('\n') if self.old_content else []
        new_lines = self.new_content.split('\n') if self.new_content else []
        
        differ = difflib.Differ()
        diff = list(differ.compare(old_lines, new_lines))
        
        half_width = width // 2 - 2
        lines = [
            "OLD".center(half_width) + " | " + "NEW".center(half_width),
            "-" * half_width + " | " + "-" * half_width,
        ]
        
        for line in diff:
            if line.startswith('  '):
                # Unchanged
                text = line[2:][:half_width].ljust(half_width)
                lines.append(f"{text} | {text}")
            elif line.startswith('- '):
                # Removed
                text = line[2:][:half_width].ljust(half_width)
                lines.append(f"{text} | ")
            elif line.startswith('+ '):
                # Added
                text = line[2:][:half_width].ljust(half_width)
                lines.append(f"{' ' * half_width} | {text}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'file_path': self.file_path,
            'change_type': self.change_type.value,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'lines_added': self.lines_added,
            'lines_removed': self.lines_removed,
            'approved': self.approved,
            'ai_generated': self.ai_generated,
        }


class ChangeTracker:
    """Track all changes made during backend generation.
    
    Provides:
    - Change history with full content tracking
    - Undo/redo functionality
    - Change approval workflow
    - Export to various formats
    """
    
    def __init__(self):
        """Initialize change tracker."""
        self.changes: List[FileChange] = []
        self.undo_stack: List[List[FileChange]] = []
        self.redo_stack: List[List[FileChange]] = []
        self._snapshots: Dict[str, List[FileChange]] = {}
    
    def add_change(self, change: FileChange) -> None:
        """Add a new change.
        
        Args:
            change: The file change to track
        """
        self.changes.append(change)
        self.undo_stack.append([change])
        self.redo_stack.clear()
        
        logger.debug(f"Added change: {change.file_path} ({change.change_type.value})")
    
    def add_batch_changes(self, changes: List[FileChange]) -> None:
        """Add multiple changes as a batch.
        
        Batch changes are undone/redone together.
        
        Args:
            changes: List of file changes
        """
        self.changes.extend(changes)
        self.undo_stack.append(changes)
        self.redo_stack.clear()
        
        logger.debug(f"Added batch of {len(changes)} changes")
    
    def undo(self) -> bool:
        """Undo last change.
        
        Returns:
            True if undo was successful, False if nothing to undo
        """
        if not self.undo_stack:
            return False
        
        changes = self.undo_stack.pop()
        self.redo_stack.append(changes)
        
        for change in changes:
            if change in self.changes:
                self.changes.remove(change)
        
        logger.info(f"Undid {len(changes)} change(s)")
        return True
    
    def redo(self) -> bool:
        """Redo last undone change.
        
        Returns:
            True if redo was successful, False if nothing to redo
        """
        if not self.redo_stack:
            return False
        
        changes = self.redo_stack.pop()
        self.undo_stack.append(changes)
        self.changes.extend(changes)
        
        logger.info(f"Redid {len(changes)} change(s)")
        return True
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self.redo_stack) > 0
    
    def approve_change(self, file_path: str) -> bool:
        """Approve a specific change.
        
        Args:
            file_path: Path of the file change to approve
            
        Returns:
            True if change was found and approved
        """
        for change in self.changes:
            if change.file_path == file_path:
                change.approved = True
                return True
        return False
    
    def approve_all(self) -> int:
        """Approve all changes.
        
        Returns:
            Number of changes approved
        """
        count = 0
        for change in self.changes:
            if not change.approved:
                change.approved = True
                count += 1
        return count
    
    def reject_change(self, file_path: str) -> bool:
        """Reject a specific change.
        
        Args:
            file_path: Path of the file change to reject
            
        Returns:
            True if change was found and removed
        """
        for change in self.changes:
            if change.file_path == file_path:
                self.changes.remove(change)
                return True
        return False
    
    def reject_all(self) -> None:
        """Reject all changes."""
        self.changes.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        logger.info("Rejected all changes")
    
    def get_pending_changes(self) -> List[FileChange]:
        """Get all unapproved changes.
        
        Returns:
            List of changes not yet approved
        """
        return [c for c in self.changes if not c.approved]
    
    def get_approved_changes(self) -> List[FileChange]:
        """Get all approved changes.
        
        Returns:
            List of approved changes
        """
        return [c for c in self.changes if c.approved]
    
    def get_stats(self) -> Dict[str, int]:
        """Get change statistics.
        
        Returns:
            Dictionary with change statistics
        """
        return {
            'total_files': len(set(c.file_path for c in self.changes)),
            'total_changes': len(self.changes),
            'lines_added': sum(c.lines_added for c in self.changes),
            'lines_removed': sum(c.lines_removed for c in self.changes),
            'approved': sum(1 for c in self.changes if c.approved),
            'pending': sum(1 for c in self.changes if not c.approved),
            'creates': sum(1 for c in self.changes if c.change_type == ChangeType.CREATE),
            'modifies': sum(1 for c in self.changes if c.change_type == ChangeType.MODIFY),
            'deletes': sum(1 for c in self.changes if c.change_type == ChangeType.DELETE),
        }
    
    def create_snapshot(self, name: str) -> None:
        """Create a named snapshot of current changes.
        
        Args:
            name: Name for the snapshot
        """
        self._snapshots[name] = self.changes.copy()
        logger.info(f"Created snapshot: {name}")
    
    def restore_snapshot(self, name: str) -> bool:
        """Restore a named snapshot.
        
        Args:
            name: Name of the snapshot to restore
            
        Returns:
            True if snapshot was found and restored
        """
        if name not in self._snapshots:
            return False
        
        self.changes = self._snapshots[name].copy()
        self.undo_stack.clear()
        self.redo_stack.clear()
        
        logger.info(f"Restored snapshot: {name}")
        return True
    
    def list_snapshots(self) -> List[str]:
        """Get list of snapshot names.
        
        Returns:
            List of snapshot names
        """
        return list(self._snapshots.keys())
    
    def export_changes(self, format: str = 'json') -> str:
        """Export changes to a format.
        
        Args:
            format: Export format ('json' or 'patch')
            
        Returns:
            Exported changes as string
        """
        if format == 'json':
            return json.dumps([c.to_dict() for c in self.changes], indent=2)
        
        elif format == 'patch':
            patches = []
            for change in self.changes:
                patches.append(f"# {change.description}")
                patches.append(change.get_unified_diff())
                patches.append("")
            return '\n'.join(patches)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_changes(self, data: str, format: str = 'json') -> int:
        """Import changes from exported data.
        
        Args:
            data: Exported change data
            format: Format of the data ('json')
            
        Returns:
            Number of changes imported
        """
        if format != 'json':
            raise ValueError(f"Import only supports 'json' format")
        
        items = json.loads(data)
        count = 0
        
        for item in items:
            change = FileChange(
                file_path=item['file_path'],
                change_type=ChangeType(item['change_type']),
                description=item.get('description', ''),
                approved=item.get('approved', False),
            )
            self.add_change(change)
            count += 1
        
        return count
    
    def get_change_summary(self) -> str:
        """Get a text summary of all changes.
        
        Returns:
            Human-readable summary
        """
        stats = self.get_stats()
        lines = [
            "Change Summary",
            "=" * 40,
            f"Total Files: {stats['total_files']}",
            f"Total Changes: {stats['total_changes']}",
            f"Lines Added: +{stats['lines_added']}",
            f"Lines Removed: -{stats['lines_removed']}",
            "",
            f"Creates: {stats['creates']}",
            f"Modifies: {stats['modifies']}",
            f"Deletes: {stats['deletes']}",
            "",
            f"Approved: {stats['approved']}",
            f"Pending: {stats['pending']}",
            "",
            "Files:",
        ]
        
        for change in self.changes:
            status = "✓" if change.approved else "○"
            type_icon = {
                ChangeType.CREATE: "+",
                ChangeType.MODIFY: "~",
                ChangeType.DELETE: "-",
            }.get(change.change_type, "?")
            lines.append(f"  {status} [{type_icon}] {change.file_path}")
        
        return '\n'.join(lines)
