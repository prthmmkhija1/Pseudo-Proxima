"""Git Conflict Resolver for Merge Conflicts.

Phase 7: Git Operations Integration

Provides comprehensive conflict resolution including:
- Conflict detection from status
- Conflict marker parsing
- Resolution strategies (ours/theirs/manual)
- Auto-resolution for simple cases
- UI-ready conflict presentation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.git_conflict_resolver")


class ConflictType(Enum):
    """Types of merge conflicts."""
    CONTENT = "content"         # Content conflict in file
    MODIFY_DELETE = "modify_delete"  # Modified vs deleted
    RENAME_RENAME = "rename_rename"  # Renamed differently
    RENAME_DELETE = "rename_delete"  # Renamed vs deleted
    ADD_ADD = "add_add"             # Both added same file
    SUBMODULE = "submodule"         # Submodule conflict


class ResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    OURS = "ours"       # Keep our version
    THEIRS = "theirs"   # Keep their version
    BOTH = "both"       # Keep both versions
    MANUAL = "manual"   # Manual resolution required
    MERGE = "merge"     # Attempt automatic merge


@dataclass
class ConflictSection:
    """A single conflict section in a file."""
    start_line: int
    separator_line: int
    end_line: int
    ours_content: List[str]
    theirs_content: List[str]
    ours_label: str = "HEAD"
    theirs_label: str = ""
    base_content: Optional[List[str]] = None  # For diff3 format
    
    @property
    def ours_text(self) -> str:
        """Get ours content as text."""
        return "\n".join(self.ours_content)
    
    @property
    def theirs_text(self) -> str:
        """Get theirs content as text."""
        return "\n".join(self.theirs_content)
    
    @property
    def line_count(self) -> int:
        """Total lines including markers."""
        return self.end_line - self.start_line + 1
    
    @property
    def is_simple(self) -> bool:
        """Check if conflict is simple (can be auto-resolved)."""
        # Simple if one side is empty or content is identical
        if not self.ours_content or not self.theirs_content:
            return True
        if self.ours_content == self.theirs_content:
            return True
        # Simple if very small change
        if len(self.ours_content) <= 2 and len(self.theirs_content) <= 2:
            return True
        return False
    
    def resolve(self, strategy: ResolutionStrategy) -> List[str]:
        """Resolve this conflict with given strategy.
        
        Args:
            strategy: Resolution strategy
            
        Returns:
            Resolved content lines
        """
        if strategy == ResolutionStrategy.OURS:
            return self.ours_content
        elif strategy == ResolutionStrategy.THEIRS:
            return self.theirs_content
        elif strategy == ResolutionStrategy.BOTH:
            return self.ours_content + self.theirs_content
        else:
            # Return original conflict markers for manual resolution
            lines = [f"<<<<<<< {self.ours_label}"]
            lines.extend(self.ours_content)
            lines.append("=======")
            lines.extend(self.theirs_content)
            lines.append(f">>>>>>> {self.theirs_label}")
            return lines
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_line": self.start_line,
            "separator_line": self.separator_line,
            "end_line": self.end_line,
            "ours_content": self.ours_content,
            "theirs_content": self.theirs_content,
            "ours_label": self.ours_label,
            "theirs_label": self.theirs_label,
            "is_simple": self.is_simple,
        }


@dataclass
class ConflictFile:
    """A file with merge conflicts."""
    path: str
    conflict_type: ConflictType
    sections: List[ConflictSection] = field(default_factory=list)
    original_content: Optional[str] = None
    
    @property
    def conflict_count(self) -> int:
        """Number of conflict sections."""
        return len(self.sections)
    
    @property
    def total_conflict_lines(self) -> int:
        """Total lines involved in conflicts."""
        return sum(s.line_count for s in self.sections)
    
    @property
    def is_simple(self) -> bool:
        """Check if all conflicts are simple."""
        return all(s.is_simple for s in self.sections)
    
    @property
    def can_auto_resolve(self) -> bool:
        """Check if file can be auto-resolved."""
        if self.conflict_type != ConflictType.CONTENT:
            return False
        return self.is_simple
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "conflict_type": self.conflict_type.value,
            "sections": [s.to_dict() for s in self.sections],
            "conflict_count": self.conflict_count,
            "is_simple": self.is_simple,
            "can_auto_resolve": self.can_auto_resolve,
        }


@dataclass
class ConflictResolutionResult:
    """Result of conflict resolution."""
    path: str
    success: bool
    strategy: ResolutionStrategy
    message: str
    resolved_content: Optional[str] = None
    remaining_conflicts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "success": self.success,
            "strategy": self.strategy.value,
            "message": self.message,
            "remaining_conflicts": self.remaining_conflicts,
        }


@dataclass
class MergeStatus:
    """Status of an ongoing merge."""
    in_progress: bool
    merge_head: Optional[str] = None  # Commit being merged
    merge_msg: Optional[str] = None   # Merge message
    conflict_files: List[ConflictFile] = field(default_factory=list)
    
    @property
    def has_conflicts(self) -> bool:
        """Check if there are unresolved conflicts."""
        return len(self.conflict_files) > 0
    
    @property
    def conflict_count(self) -> int:
        """Total number of conflicts."""
        return sum(f.conflict_count for f in self.conflict_files)
    
    @property
    def file_count(self) -> int:
        """Number of files with conflicts."""
        return len(self.conflict_files)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "in_progress": self.in_progress,
            "merge_head": self.merge_head,
            "merge_msg": self.merge_msg,
            "has_conflicts": self.has_conflicts,
            "conflict_count": self.conflict_count,
            "file_count": self.file_count,
            "conflict_files": [f.to_dict() for f in self.conflict_files],
        }


class ConflictParser:
    """Parse conflict markers in files.
    
    Supports standard git conflict format and diff3 format.
    
    Example:
        >>> parser = ConflictParser()
        >>> sections = parser.parse_file_content(content)
        >>> for section in sections:
        ...     print(f"Conflict at line {section.start_line}")
    """
    
    # Conflict marker patterns
    START_MARKER = re.compile(r"^<<<<<<<\s*(.*)$")
    SEPARATOR_MARKER = re.compile(r"^=======$")
    END_MARKER = re.compile(r"^>>>>>>>\s*(.*)$")
    BASE_MARKER = re.compile(r"^\|\|\|\|\|\|\|\s*(.*)$")  # For diff3
    
    def __init__(self):
        """Initialize parser."""
        pass
    
    def parse_file_content(self, content: str) -> List[ConflictSection]:
        """Parse conflict markers from file content.
        
        Args:
            content: File content with conflict markers
            
        Returns:
            List of ConflictSection objects
        """
        sections: List[ConflictSection] = []
        lines = content.split("\n")
        
        i = 0
        while i < len(lines):
            start_match = self.START_MARKER.match(lines[i])
            if start_match:
                section, i = self._parse_conflict_section(lines, i, start_match.group(1))
                if section:
                    sections.append(section)
            else:
                i += 1
        
        return sections
    
    def _parse_conflict_section(
        self,
        lines: List[str],
        start_idx: int,
        ours_label: str,
    ) -> Tuple[Optional[ConflictSection], int]:
        """Parse a single conflict section."""
        ours_content: List[str] = []
        theirs_content: List[str] = []
        base_content: Optional[List[str]] = None
        
        separator_line = -1
        i = start_idx + 1
        
        # Collect ours content (and possibly base for diff3)
        in_base = False
        while i < len(lines):
            if self.SEPARATOR_MARKER.match(lines[i]):
                separator_line = i
                i += 1
                break
            
            base_match = self.BASE_MARKER.match(lines[i])
            if base_match:
                # diff3 format - switch to base
                in_base = True
                base_content = []
                i += 1
                continue
            
            if in_base:
                base_content.append(lines[i])
            else:
                ours_content.append(lines[i])
            i += 1
        
        if separator_line == -1:
            return None, start_idx + 1
        
        # Collect theirs content
        theirs_label = ""
        while i < len(lines):
            end_match = self.END_MARKER.match(lines[i])
            if end_match:
                theirs_label = end_match.group(1)
                end_line = i
                i += 1
                break
            
            theirs_content.append(lines[i])
            i += 1
        else:
            # No end marker found
            return None, start_idx + 1
        
        return ConflictSection(
            start_line=start_idx,
            separator_line=separator_line,
            end_line=end_line,
            ours_content=ours_content,
            theirs_content=theirs_content,
            ours_label=ours_label or "HEAD",
            theirs_label=theirs_label,
            base_content=base_content,
        ), i
    
    def has_conflicts(self, content: str) -> bool:
        """Check if content has conflict markers.
        
        Args:
            content: File content
            
        Returns:
            True if conflicts found
        """
        return bool(self.START_MARKER.search(content))
    
    def count_conflicts(self, content: str) -> int:
        """Count conflict sections in content.
        
        Args:
            content: File content
            
        Returns:
            Number of conflicts
        """
        return len(self.START_MARKER.findall(content))


class GitConflictResolver:
    """Resolve git merge conflicts.
    
    Provides tools for detecting, analyzing, and resolving conflicts.
    
    Example:
        >>> resolver = GitConflictResolver(git_ops)
        >>> 
        >>> # Get merge status
        >>> status = await resolver.get_merge_status()
        >>> if status.has_conflicts:
        ...     for file in status.conflict_files:
        ...         print(f"Conflict in {file.path}")
        >>> 
        >>> # Resolve a file
        >>> result = await resolver.resolve_file(
        ...     "src/main.py",
        ...     ResolutionStrategy.OURS
        ... )
    """
    
    def __init__(
        self,
        git_operations: Any,
        repo_path: Optional[str] = None,
    ):
        """Initialize resolver.
        
        Args:
            git_operations: GitOperations instance
            repo_path: Repository path
        """
        self.git_ops = git_operations
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.parser = ConflictParser()
    
    async def get_merge_status(self) -> MergeStatus:
        """Get current merge status.
        
        Returns:
            MergeStatus with conflict information
        """
        # Check if merge is in progress
        merge_head_path = self.repo_path / ".git" / "MERGE_HEAD"
        in_progress = merge_head_path.exists()
        
        merge_head = None
        merge_msg = None
        
        if in_progress:
            try:
                merge_head = merge_head_path.read_text().strip()[:8]
                merge_msg_path = self.repo_path / ".git" / "MERGE_MSG"
                if merge_msg_path.exists():
                    merge_msg = merge_msg_path.read_text().strip()
            except Exception:
                pass
        
        # Get conflicted files from git status
        conflict_files: List[ConflictFile] = []
        
        result = await self.git_ops.status()
        if result.success and result.data:
            for file_status in result.data.get("files", []):
                if file_status.get("is_conflict"):
                    conflict_file = await self._analyze_conflict_file(file_status["path"])
                    if conflict_file:
                        conflict_files.append(conflict_file)
        
        return MergeStatus(
            in_progress=in_progress,
            merge_head=merge_head,
            merge_msg=merge_msg,
            conflict_files=conflict_files,
        )
    
    async def _analyze_conflict_file(self, path: str) -> Optional[ConflictFile]:
        """Analyze a conflicted file."""
        full_path = self.repo_path / path
        
        if not full_path.exists():
            return ConflictFile(
                path=path,
                conflict_type=ConflictType.MODIFY_DELETE,
            )
        
        try:
            content = full_path.read_text()
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            return None
        
        sections = self.parser.parse_file_content(content)
        
        return ConflictFile(
            path=path,
            conflict_type=ConflictType.CONTENT if sections else ConflictType.MODIFY_DELETE,
            sections=sections,
            original_content=content,
        )
    
    async def get_conflict_details(self, path: str) -> Optional[ConflictFile]:
        """Get detailed conflict information for a file.
        
        Args:
            path: File path
            
        Returns:
            ConflictFile with detailed sections
        """
        return await self._analyze_conflict_file(path)
    
    async def resolve_file(
        self,
        path: str,
        strategy: ResolutionStrategy,
        section_index: Optional[int] = None,
    ) -> ConflictResolutionResult:
        """Resolve conflicts in a file.
        
        Args:
            path: File path
            strategy: Resolution strategy
            section_index: Specific section to resolve (None for all)
            
        Returns:
            ConflictResolutionResult
        """
        conflict_file = await self._analyze_conflict_file(path)
        
        if not conflict_file:
            return ConflictResolutionResult(
                path=path,
                success=False,
                strategy=strategy,
                message="Could not analyze file",
            )
        
        if not conflict_file.sections:
            # No content conflicts - might be other type
            if strategy == ResolutionStrategy.OURS:
                result = await self._checkout_version(path, "HEAD")
            elif strategy == ResolutionStrategy.THEIRS:
                result = await self._checkout_version(path, "MERGE_HEAD")
            else:
                return ConflictResolutionResult(
                    path=path,
                    success=False,
                    strategy=strategy,
                    message="Manual resolution required for this conflict type",
                )
            return result
        
        # Resolve content conflicts
        content = conflict_file.original_content or ""
        lines = content.split("\n")
        
        # Process sections in reverse order to maintain line numbers
        sections_to_resolve = conflict_file.sections
        if section_index is not None:
            if 0 <= section_index < len(sections_to_resolve):
                sections_to_resolve = [sections_to_resolve[section_index]]
            else:
                return ConflictResolutionResult(
                    path=path,
                    success=False,
                    strategy=strategy,
                    message=f"Invalid section index: {section_index}",
                )
        
        # Sort by start line in reverse
        sections_to_resolve = sorted(
            sections_to_resolve,
            key=lambda s: s.start_line,
            reverse=True,
        )
        
        for section in sections_to_resolve:
            resolved_lines = section.resolve(strategy)
            # Replace conflict section with resolved content
            lines[section.start_line:section.end_line + 1] = resolved_lines
        
        resolved_content = "\n".join(lines)
        
        # Write resolved content
        full_path = self.repo_path / path
        try:
            full_path.write_text(resolved_content)
        except Exception as e:
            return ConflictResolutionResult(
                path=path,
                success=False,
                strategy=strategy,
                message=f"Failed to write resolved content: {e}",
            )
        
        # Check for remaining conflicts
        remaining = self.parser.count_conflicts(resolved_content)
        
        if remaining == 0:
            # Stage the resolved file
            await self.git_ops.add(path)
        
        return ConflictResolutionResult(
            path=path,
            success=True,
            strategy=strategy,
            message="Conflicts resolved" if remaining == 0 else f"{remaining} conflicts remaining",
            resolved_content=resolved_content,
            remaining_conflicts=remaining,
        )
    
    async def _checkout_version(
        self,
        path: str,
        version: str,
    ) -> ConflictResolutionResult:
        """Checkout a specific version of a file."""
        result = await self.git_ops._execute_git([
            "checkout", f"--{version.lower()}", "--", path
        ])
        
        if result.success:
            # Stage the file
            await self.git_ops.add(path)
            return ConflictResolutionResult(
                path=path,
                success=True,
                strategy=ResolutionStrategy.OURS if version == "HEAD" else ResolutionStrategy.THEIRS,
                message=f"Resolved using {version}",
            )
        
        return ConflictResolutionResult(
            path=path,
            success=False,
            strategy=ResolutionStrategy.MANUAL,
            message=f"Failed to checkout {version}: {result.message}",
        )
    
    async def resolve_all(
        self,
        strategy: ResolutionStrategy,
    ) -> List[ConflictResolutionResult]:
        """Resolve all conflicts with given strategy.
        
        Args:
            strategy: Resolution strategy
            
        Returns:
            List of results for each file
        """
        status = await self.get_merge_status()
        results: List[ConflictResolutionResult] = []
        
        for conflict_file in status.conflict_files:
            result = await self.resolve_file(conflict_file.path, strategy)
            results.append(result)
        
        return results
    
    async def abort_merge(self) -> Tuple[bool, str]:
        """Abort the current merge.
        
        Returns:
            Tuple of (success, message)
        """
        result = await self.git_ops._execute_git(["merge", "--abort"])
        
        if result.success:
            return True, "Merge aborted successfully"
        return False, f"Failed to abort merge: {result.message}"
    
    async def continue_merge(self) -> Tuple[bool, str]:
        """Continue merge after resolving conflicts.
        
        Returns:
            Tuple of (success, message)
        """
        # Check for remaining conflicts
        status = await self.get_merge_status()
        if status.has_conflicts:
            return False, f"{status.conflict_count} conflicts still unresolved"
        
        result = await self.git_ops._execute_git(["merge", "--continue"])
        
        if result.success:
            return True, "Merge completed successfully"
        return False, f"Failed to continue merge: {result.message}"
    
    async def get_file_versions(
        self,
        path: str,
    ) -> Dict[str, Optional[str]]:
        """Get different versions of a conflicted file.
        
        Args:
            path: File path
            
        Returns:
            Dictionary with 'ours', 'theirs', 'base' content
        """
        versions: Dict[str, Optional[str]] = {
            "ours": None,
            "theirs": None,
            "base": None,
        }
        
        # Get ours (HEAD)
        ours_result = await self.git_ops._execute_git([
            "show", f"HEAD:{path}"
        ])
        if ours_result.success:
            versions["ours"] = ours_result.data
        
        # Get theirs (MERGE_HEAD)
        theirs_result = await self.git_ops._execute_git([
            "show", f"MERGE_HEAD:{path}"
        ])
        if theirs_result.success:
            versions["theirs"] = theirs_result.data
        
        # Get base (merge base)
        base_commit_result = await self.git_ops._execute_git([
            "merge-base", "HEAD", "MERGE_HEAD"
        ])
        if base_commit_result.success and base_commit_result.data:
            base_commit = base_commit_result.data.strip()
            base_result = await self.git_ops._execute_git([
                "show", f"{base_commit}:{path}"
            ])
            if base_result.success:
                versions["base"] = base_result.data
        
        return versions
    
    def suggest_resolution(
        self,
        section: ConflictSection,
    ) -> ResolutionStrategy:
        """Suggest a resolution strategy for a conflict.
        
        Args:
            section: Conflict section
            
        Returns:
            Suggested strategy
        """
        # If one side is empty, use the other
        if not section.ours_content:
            return ResolutionStrategy.THEIRS
        if not section.theirs_content:
            return ResolutionStrategy.OURS
        
        # If content is identical
        if section.ours_content == section.theirs_content:
            return ResolutionStrategy.OURS
        
        # If ours is subset of theirs or vice versa
        ours_text = section.ours_text
        theirs_text = section.theirs_text
        
        if ours_text in theirs_text:
            return ResolutionStrategy.THEIRS
        if theirs_text in ours_text:
            return ResolutionStrategy.OURS
        
        # Default to manual
        return ResolutionStrategy.MANUAL


# Convenience functions
def get_conflict_resolver(
    git_operations: Any,
    repo_path: Optional[str] = None,
) -> GitConflictResolver:
    """Get a GitConflictResolver instance."""
    return GitConflictResolver(git_operations, repo_path)


def parse_conflict_markers(content: str) -> List[ConflictSection]:
    """Convenience function to parse conflict markers."""
    parser = ConflictParser()
    return parser.parse_file_content(content)
