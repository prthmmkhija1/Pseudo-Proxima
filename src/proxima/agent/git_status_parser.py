"""Git Status Parser for Enhanced Status Information.

Phase 7: Git Operations Integration

Provides comprehensive git status parsing including:
- Porcelain format parsing
- File status categorization
- Merge conflict detection
- Summary generation for UI
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.git_status_parser")


class FileStatusCode(Enum):
    """Git file status codes from porcelain format."""
    UNMODIFIED = " "
    MODIFIED = "M"
    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UNMERGED = "U"
    UNTRACKED = "?"
    IGNORED = "!"
    
    @classmethod
    def from_char(cls, char: str) -> "FileStatusCode":
        """Create from character."""
        for status in cls:
            if status.value == char:
                return status
        return cls.UNMODIFIED


class MergeConflictType(Enum):
    """Types of merge conflicts."""
    BOTH_MODIFIED = "both_modified"
    BOTH_ADDED = "both_added"
    DELETED_BY_US = "deleted_by_us"
    DELETED_BY_THEM = "deleted_by_them"
    ADDED_BY_US = "added_by_us"
    ADDED_BY_THEM = "added_by_them"


@dataclass
class GitFileStatus:
    """Detailed status of a single file."""
    path: str
    index_status: FileStatusCode  # X from XY
    worktree_status: FileStatusCode  # Y from XY
    original_path: Optional[str] = None  # For renames/copies
    
    @property
    def staged(self) -> bool:
        """Check if file is staged."""
        return (
            self.index_status != FileStatusCode.UNMODIFIED
            and self.index_status != FileStatusCode.UNTRACKED
            and self.index_status != FileStatusCode.IGNORED
        )
    
    @property
    def has_unstaged_changes(self) -> bool:
        """Check if file has unstaged changes."""
        return (
            self.worktree_status != FileStatusCode.UNMODIFIED
            and self.worktree_status != FileStatusCode.UNTRACKED
        )
    
    @property
    def is_untracked(self) -> bool:
        """Check if file is untracked."""
        return self.index_status == FileStatusCode.UNTRACKED
    
    @property
    def is_ignored(self) -> bool:
        """Check if file is ignored."""
        return self.index_status == FileStatusCode.IGNORED
    
    @property
    def is_conflict(self) -> bool:
        """Check if file has merge conflict."""
        return (
            self.index_status == FileStatusCode.UNMERGED
            or self.worktree_status == FileStatusCode.UNMERGED
        )
    
    @property
    def display_status(self) -> str:
        """Get human-readable status."""
        if self.is_conflict:
            return "conflict"
        if self.is_untracked:
            return "untracked"
        if self.is_ignored:
            return "ignored"
        if self.index_status == FileStatusCode.ADDED:
            return "added"
        if self.index_status == FileStatusCode.DELETED or self.worktree_status == FileStatusCode.DELETED:
            return "deleted"
        if self.index_status == FileStatusCode.RENAMED:
            return "renamed"
        if self.index_status == FileStatusCode.COPIED:
            return "copied"
        if self.staged or self.has_unstaged_changes:
            return "modified"
        return "unmodified"
    
    @property
    def status_color(self) -> str:
        """Get color for status display."""
        colors = {
            "conflict": "red",
            "untracked": "cyan",
            "ignored": "dim",
            "added": "green",
            "deleted": "red",
            "renamed": "yellow",
            "copied": "yellow",
            "modified": "yellow",
            "unmodified": "dim",
        }
        return colors.get(self.display_status, "white")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "index_status": self.index_status.value,
            "worktree_status": self.worktree_status.value,
            "original_path": self.original_path,
            "staged": self.staged,
            "has_unstaged_changes": self.has_unstaged_changes,
            "is_untracked": self.is_untracked,
            "is_conflict": self.is_conflict,
            "display_status": self.display_status,
            "status_color": self.status_color,
        }


@dataclass
class ConflictMarker:
    """A conflict marker section in a file."""
    start_line: int
    separator_line: int
    end_line: int
    ours_content: str
    theirs_content: str
    ours_label: str = "HEAD"
    theirs_label: str = ""


@dataclass
class ConflictFile:
    """A file with merge conflicts."""
    path: str
    conflict_type: MergeConflictType
    markers: List[ConflictMarker] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "conflict_type": self.conflict_type.value,
            "conflict_count": len(self.markers),
        }


@dataclass
class BranchStatus:
    """Status of the current branch."""
    name: str
    tracking: Optional[str] = None
    ahead: int = 0
    behind: int = 0
    is_detached: bool = False
    
    @property
    def sync_status(self) -> str:
        """Get synchronization status description."""
        if not self.tracking:
            return "no tracking"
        if self.ahead == 0 and self.behind == 0:
            return "up to date"
        parts = []
        if self.ahead > 0:
            parts.append(f"ahead {self.ahead}")
        if self.behind > 0:
            parts.append(f"behind {self.behind}")
        return ", ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "tracking": self.tracking,
            "ahead": self.ahead,
            "behind": self.behind,
            "is_detached": self.is_detached,
            "sync_status": self.sync_status,
        }


@dataclass
class RepositoryStatus:
    """Complete repository status."""
    branch: BranchStatus
    files: List[GitFileStatus]
    conflicts: List[ConflictFile] = field(default_factory=list)
    
    @property
    def is_clean(self) -> bool:
        """Check if working tree is clean."""
        return len(self.files) == 0
    
    @property
    def has_staged(self) -> bool:
        """Check if there are staged changes."""
        return any(f.staged for f in self.files)
    
    @property
    def has_unstaged(self) -> bool:
        """Check if there are unstaged changes."""
        return any(f.has_unstaged_changes for f in self.files)
    
    @property
    def has_untracked(self) -> bool:
        """Check if there are untracked files."""
        return any(f.is_untracked for f in self.files)
    
    @property
    def has_conflicts(self) -> bool:
        """Check if there are merge conflicts."""
        return len(self.conflicts) > 0
    
    @property
    def staged_files(self) -> List[GitFileStatus]:
        """Get list of staged files."""
        return [f for f in self.files if f.staged]
    
    @property
    def unstaged_files(self) -> List[GitFileStatus]:
        """Get list of files with unstaged changes."""
        return [f for f in self.files if f.has_unstaged_changes and not f.is_untracked]
    
    @property
    def untracked_files(self) -> List[GitFileStatus]:
        """Get list of untracked files."""
        return [f for f in self.files if f.is_untracked]
    
    @property
    def modified_count(self) -> int:
        """Count of modified files."""
        return len([f for f in self.files if f.display_status == "modified"])
    
    @property
    def added_count(self) -> int:
        """Count of added files."""
        return len([f for f in self.files if f.display_status == "added"])
    
    @property
    def deleted_count(self) -> int:
        """Count of deleted files."""
        return len([f for f in self.files if f.display_status == "deleted"])
    
    @property
    def untracked_count(self) -> int:
        """Count of untracked files."""
        return len(self.untracked_files)
    
    @property
    def conflict_count(self) -> int:
        """Count of files with conflicts."""
        return len(self.conflicts)
    
    def get_summary(self) -> str:
        """Get human-readable status summary."""
        if self.is_clean:
            return "Working tree clean"
        
        parts = []
        if self.modified_count:
            parts.append(f"{self.modified_count} modified")
        if self.added_count:
            parts.append(f"{self.added_count} added")
        if self.deleted_count:
            parts.append(f"{self.deleted_count} deleted")
        if self.untracked_count:
            parts.append(f"{self.untracked_count} untracked")
        if self.conflict_count:
            parts.append(f"{self.conflict_count} conflicts")
        
        return ", ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "branch": self.branch.to_dict(),
            "files": [f.to_dict() for f in self.files],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "is_clean": self.is_clean,
            "has_staged": self.has_staged,
            "has_unstaged": self.has_unstaged,
            "has_untracked": self.has_untracked,
            "has_conflicts": self.has_conflicts,
            "modified_count": self.modified_count,
            "added_count": self.added_count,
            "deleted_count": self.deleted_count,
            "untracked_count": self.untracked_count,
            "conflict_count": self.conflict_count,
            "summary": self.get_summary(),
        }


class GitStatusParser:
    """Parse git status output.
    
    Parses porcelain format output from `git status --porcelain=v1 -b`
    and provides structured status information.
    
    Example:
        >>> parser = GitStatusParser()
        >>> status = parser.parse(porcelain_output)
        >>> print(status.get_summary())
        >>> 
        >>> for file in status.staged_files:
        ...     print(f"Staged: {file.path}")
    """
    
    # Patterns for parsing
    BRANCH_PATTERN = re.compile(
        r"^## (?P<branch>[\w\-./]+?)(?:\.\.\.(?P<tracking>[\w\-./]+))?"
        r"(?: \[(?:ahead (?P<ahead>\d+))?(?:, )?(?:behind (?P<behind>\d+))?\])?$"
    )
    
    DETACHED_PATTERN = re.compile(r"^## HEAD \(no branch\)$")
    
    # Conflict status combinations (index, worktree)
    CONFLICT_STATUSES = {
        ("D", "D"): MergeConflictType.BOTH_MODIFIED,  # Both deleted
        ("A", "U"): MergeConflictType.ADDED_BY_US,
        ("U", "D"): MergeConflictType.DELETED_BY_THEM,
        ("U", "A"): MergeConflictType.ADDED_BY_THEM,
        ("D", "U"): MergeConflictType.DELETED_BY_US,
        ("A", "A"): MergeConflictType.BOTH_ADDED,
        ("U", "U"): MergeConflictType.BOTH_MODIFIED,
    }
    
    def __init__(self):
        """Initialize the parser."""
        pass
    
    def parse(self, output: str) -> RepositoryStatus:
        """Parse git status porcelain output.
        
        Args:
            output: Output from `git status --porcelain=v1 -b`
            
        Returns:
            RepositoryStatus with parsed information
        """
        lines = output.strip().split("\n") if output.strip() else []
        
        branch = BranchStatus(name="unknown")
        files: List[GitFileStatus] = []
        conflicts: List[ConflictFile] = []
        
        for line in lines:
            if not line:
                continue
            
            if line.startswith("##"):
                branch = self._parse_branch_line(line)
            else:
                file_status = self._parse_file_line(line)
                if file_status:
                    files.append(file_status)
                    
                    # Check for conflict
                    if file_status.is_conflict:
                        conflict_type = self._get_conflict_type(
                            file_status.index_status.value,
                            file_status.worktree_status.value,
                        )
                        conflicts.append(ConflictFile(
                            path=file_status.path,
                            conflict_type=conflict_type,
                        ))
        
        return RepositoryStatus(
            branch=branch,
            files=files,
            conflicts=conflicts,
        )
    
    def _parse_branch_line(self, line: str) -> BranchStatus:
        """Parse the branch status line."""
        # Check for detached HEAD
        if self.DETACHED_PATTERN.match(line):
            return BranchStatus(name="HEAD", is_detached=True)
        
        match = self.BRANCH_PATTERN.match(line)
        if match:
            return BranchStatus(
                name=match.group("branch"),
                tracking=match.group("tracking"),
                ahead=int(match.group("ahead") or 0),
                behind=int(match.group("behind") or 0),
            )
        
        # Fallback: extract branch name
        if line.startswith("## "):
            branch_part = line[3:].split("...")[0].strip()
            return BranchStatus(name=branch_part)
        
        return BranchStatus(name="unknown")
    
    def _parse_file_line(self, line: str) -> Optional[GitFileStatus]:
        """Parse a file status line."""
        if len(line) < 4:
            return None
        
        index_char = line[0]
        worktree_char = line[1]
        path = line[3:]
        
        # Handle renames (path contains " -> ")
        original_path = None
        if " -> " in path:
            original_path, path = path.split(" -> ", 1)
        
        return GitFileStatus(
            path=path,
            index_status=FileStatusCode.from_char(index_char),
            worktree_status=FileStatusCode.from_char(worktree_char),
            original_path=original_path,
        )
    
    def _get_conflict_type(
        self,
        index_status: str,
        worktree_status: str,
    ) -> MergeConflictType:
        """Determine the conflict type from status codes."""
        key = (index_status, worktree_status)
        return self.CONFLICT_STATUSES.get(key, MergeConflictType.BOTH_MODIFIED)
    
    def parse_conflict_markers(self, content: str) -> List[ConflictMarker]:
        """Parse conflict markers from file content.
        
        Args:
            content: File content with conflict markers
            
        Returns:
            List of ConflictMarker objects
        """
        markers: List[ConflictMarker] = []
        lines = content.split("\n")
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for start marker
            if line.startswith("<<<<<<<"):
                start_line = i
                ours_label = line[7:].strip() or "HEAD"
                ours_content_lines = []
                
                i += 1
                # Collect "ours" content until separator
                while i < len(lines) and not lines[i].startswith("======="):
                    ours_content_lines.append(lines[i])
                    i += 1
                
                if i >= len(lines):
                    break
                
                separator_line = i
                theirs_content_lines = []
                
                i += 1
                # Collect "theirs" content until end marker
                while i < len(lines) and not lines[i].startswith(">>>>>>>"):
                    theirs_content_lines.append(lines[i])
                    i += 1
                
                if i >= len(lines):
                    break
                
                end_line = i
                theirs_label = lines[i][7:].strip()
                
                markers.append(ConflictMarker(
                    start_line=start_line,
                    separator_line=separator_line,
                    end_line=end_line,
                    ours_content="\n".join(ours_content_lines),
                    theirs_content="\n".join(theirs_content_lines),
                    ours_label=ours_label,
                    theirs_label=theirs_label,
                ))
            
            i += 1
        
        return markers
    
    def get_files_by_status(
        self,
        status: RepositoryStatus,
    ) -> Dict[str, List[GitFileStatus]]:
        """Group files by their display status.
        
        Args:
            status: Repository status
            
        Returns:
            Dictionary of status -> list of files
        """
        groups: Dict[str, List[GitFileStatus]] = {
            "staged": [],
            "modified": [],
            "added": [],
            "deleted": [],
            "renamed": [],
            "untracked": [],
            "conflict": [],
        }
        
        for file in status.files:
            display = file.display_status
            
            if file.is_conflict:
                groups["conflict"].append(file)
            elif file.staged:
                groups["staged"].append(file)
            elif display in groups:
                groups[display].append(file)
        
        return {k: v for k, v in groups.items() if v}


# Global instance
_parser: Optional[GitStatusParser] = None


def get_git_status_parser() -> GitStatusParser:
    """Get the global GitStatusParser instance."""
    global _parser
    if _parser is None:
        _parser = GitStatusParser()
    return _parser


def parse_git_status(output: str) -> RepositoryStatus:
    """Convenience function to parse git status output."""
    return get_git_status_parser().parse(output)
