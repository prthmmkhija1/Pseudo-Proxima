"""Git Branch Manager for Branch Operations.

Phase 7: Git Operations Integration

Provides comprehensive branch management including:
- Branch listing (local/remote)
- Branch creation/deletion
- Safe branch switching
- Branch history visualization
- Tracking configuration
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.git_branch_manager")


class BranchType(Enum):
    """Type of branch."""
    LOCAL = "local"
    REMOTE = "remote"
    TRACKING = "tracking"  # Local with remote tracking


@dataclass
class BranchInfo:
    """Detailed branch information."""
    name: str
    branch_type: BranchType
    is_current: bool = False
    remote: Optional[str] = None  # Remote name for remote branches
    tracking: Optional[str] = None  # Tracking branch for local branches
    ahead: int = 0  # Commits ahead of tracking
    behind: int = 0  # Commits behind tracking
    last_commit_hash: Optional[str] = None
    last_commit_message: Optional[str] = None
    last_commit_date: Optional[datetime] = None
    author: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        """Get full branch name."""
        if self.branch_type == BranchType.REMOTE and self.remote:
            return f"{self.remote}/{self.name}"
        return self.name
    
    @property
    def display_name(self) -> str:
        """Get display name."""
        name = self.name
        if self.is_current:
            name = f"* {name}"
        return name
    
    @property
    def sync_status(self) -> str:
        """Get synchronization status."""
        if not self.tracking:
            return "no tracking"
        if self.ahead == 0 and self.behind == 0:
            return "up to date"
        
        parts = []
        if self.ahead > 0:
            parts.append(f"↑{self.ahead}")
        if self.behind > 0:
            parts.append(f"↓{self.behind}")
        return " ".join(parts)
    
    @property
    def color(self) -> str:
        """Get color for display."""
        if self.is_current:
            return "green"
        if self.branch_type == BranchType.REMOTE:
            return "red"
        return "white"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "branch_type": self.branch_type.value,
            "is_current": self.is_current,
            "remote": self.remote,
            "tracking": self.tracking,
            "ahead": self.ahead,
            "behind": self.behind,
            "last_commit_hash": self.last_commit_hash,
            "last_commit_message": self.last_commit_message,
            "last_commit_date": self.last_commit_date.isoformat() if self.last_commit_date else None,
            "author": self.author,
            "sync_status": self.sync_status,
        }


@dataclass
class BranchComparisonResult:
    """Result of comparing two branches."""
    base_branch: str
    compare_branch: str
    ahead: int  # Commits in compare but not in base
    behind: int  # Commits in base but not in compare
    diverged: bool
    common_ancestor: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_branch": self.base_branch,
            "compare_branch": self.compare_branch,
            "ahead": self.ahead,
            "behind": self.behind,
            "diverged": self.diverged,
            "common_ancestor": self.common_ancestor,
        }


@dataclass
class MergePreview:
    """Preview of a merge operation."""
    source_branch: str
    target_branch: str
    can_fast_forward: bool
    commits_to_merge: int
    files_changed: int
    conflicts_expected: List[str] = field(default_factory=list)
    
    @property
    def has_conflicts(self) -> bool:
        """Check if conflicts are expected."""
        return len(self.conflicts_expected) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_branch": self.source_branch,
            "target_branch": self.target_branch,
            "can_fast_forward": self.can_fast_forward,
            "commits_to_merge": self.commits_to_merge,
            "files_changed": self.files_changed,
            "conflicts_expected": self.conflicts_expected,
            "has_conflicts": self.has_conflicts,
        }


@dataclass
class BranchListResult:
    """Result of listing branches."""
    local_branches: List[BranchInfo]
    remote_branches: List[BranchInfo]
    current_branch: Optional[str] = None
    
    @property
    def all_branches(self) -> List[BranchInfo]:
        """Get all branches."""
        return self.local_branches + self.remote_branches
    
    @property
    def local_count(self) -> int:
        """Count of local branches."""
        return len(self.local_branches)
    
    @property
    def remote_count(self) -> int:
        """Count of remote branches."""
        return len(self.remote_branches)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "local_branches": [b.to_dict() for b in self.local_branches],
            "remote_branches": [b.to_dict() for b in self.remote_branches],
            "current_branch": self.current_branch,
            "local_count": self.local_count,
            "remote_count": self.remote_count,
        }


class GitBranchManager:
    """Manage git branches.
    
    Provides comprehensive branch operations with safety checks.
    
    Example:
        >>> manager = GitBranchManager(git_ops)
        >>> 
        >>> # List branches
        >>> branches = await manager.list_branches()
        >>> print(f"Current: {branches.current_branch}")
        >>> 
        >>> # Create branch
        >>> result = await manager.create_branch("feature/new-feature")
        >>> 
        >>> # Switch branch (with safety check)
        >>> result = await manager.switch_branch("develop", auto_stash=True)
    """
    
    # Branch name validation pattern
    BRANCH_NAME_PATTERN = re.compile(
        r"^(?!.*[./]{2})(?!.*@\{)(?!^/)(?!.*/$)"
        r"[^\x00-\x1f\x7f ~^:?*\[\\]+$"
    )
    
    # Common branch prefixes
    BRANCH_PREFIXES = {
        "feature": "feature/",
        "bugfix": "bugfix/",
        "hotfix": "hotfix/",
        "release": "release/",
        "support": "support/",
    }
    
    def __init__(self, git_operations: Any):
        """Initialize manager.
        
        Args:
            git_operations: GitOperations instance
        """
        self.git_ops = git_operations
    
    async def list_branches(
        self,
        include_remote: bool = True,
        fetch_first: bool = False,
    ) -> BranchListResult:
        """List all branches.
        
        Args:
            include_remote: Include remote branches
            fetch_first: Fetch from remotes before listing
            
        Returns:
            BranchListResult
        """
        # Optionally fetch first
        if fetch_first:
            await self.git_ops._execute_git(["fetch", "--all", "--prune"])
        
        # Get branch info with verbose output
        result = await self.git_ops._execute_git([
            "branch", "-vv", "--format",
            "%(HEAD)|%(refname:short)|%(objectname:short)|%(upstream:short)|%(upstream:track)|%(subject)|%(authorname)|%(committerdate:iso)"
        ])
        
        local_branches: List[BranchInfo] = []
        current_branch: Optional[str] = None
        
        if result.success and result.data:
            for line in result.data.strip().split("\n"):
                if not line.strip():
                    continue
                
                branch = self._parse_branch_line(line)
                if branch:
                    if branch.is_current:
                        current_branch = branch.name
                    local_branches.append(branch)
        
        # Get remote branches
        remote_branches: List[BranchInfo] = []
        if include_remote:
            result = await self.git_ops._execute_git([
                "branch", "-r", "--format",
                "%(refname:short)|%(objectname:short)|%(subject)|%(authorname)|%(committerdate:iso)"
            ])
            
            if result.success and result.data:
                for line in result.data.strip().split("\n"):
                    if not line.strip():
                        continue
                    
                    branch = self._parse_remote_branch_line(line)
                    if branch:
                        remote_branches.append(branch)
        
        return BranchListResult(
            local_branches=local_branches,
            remote_branches=remote_branches,
            current_branch=current_branch,
        )
    
    def _parse_branch_line(self, line: str) -> Optional[BranchInfo]:
        """Parse a branch line from git branch -vv."""
        parts = line.split("|")
        if len(parts) < 6:
            return None
        
        is_current = parts[0] == "*"
        name = parts[1]
        commit_hash = parts[2]
        tracking = parts[3] if parts[3] else None
        track_info = parts[4]
        message = parts[5]
        author = parts[6] if len(parts) > 6 else None
        date_str = parts[7] if len(parts) > 7 else None
        
        # Parse tracking info
        ahead = 0
        behind = 0
        if track_info:
            ahead_match = re.search(r"ahead (\d+)", track_info)
            behind_match = re.search(r"behind (\d+)", track_info)
            if ahead_match:
                ahead = int(ahead_match.group(1))
            if behind_match:
                behind = int(behind_match.group(1))
        
        # Parse date
        commit_date = None
        if date_str:
            try:
                commit_date = datetime.fromisoformat(date_str.replace(" ", "T").strip())
            except Exception:
                pass
        
        branch_type = BranchType.TRACKING if tracking else BranchType.LOCAL
        
        return BranchInfo(
            name=name,
            branch_type=branch_type,
            is_current=is_current,
            tracking=tracking,
            ahead=ahead,
            behind=behind,
            last_commit_hash=commit_hash,
            last_commit_message=message,
            last_commit_date=commit_date,
            author=author,
        )
    
    def _parse_remote_branch_line(self, line: str) -> Optional[BranchInfo]:
        """Parse a remote branch line."""
        parts = line.split("|")
        if len(parts) < 3:
            return None
        
        full_name = parts[0]
        commit_hash = parts[1]
        message = parts[2]
        author = parts[3] if len(parts) > 3 else None
        date_str = parts[4] if len(parts) > 4 else None
        
        # Parse remote and branch name
        remote = None
        name = full_name
        if "/" in full_name:
            remote, name = full_name.split("/", 1)
        
        # Parse date
        commit_date = None
        if date_str:
            try:
                commit_date = datetime.fromisoformat(date_str.replace(" ", "T").strip())
            except Exception:
                pass
        
        return BranchInfo(
            name=name,
            branch_type=BranchType.REMOTE,
            remote=remote,
            last_commit_hash=commit_hash,
            last_commit_message=message,
            last_commit_date=commit_date,
            author=author,
        )
    
    async def get_current_branch(self) -> Optional[str]:
        """Get current branch name."""
        result = await self.git_ops._execute_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if result.success and result.data:
            return result.data.strip()
        return None
    
    def validate_branch_name(self, name: str) -> Tuple[bool, str]:
        """Validate a branch name.
        
        Args:
            name: Branch name to validate
            
        Returns:
            Tuple of (valid, message)
        """
        if not name:
            return False, "Branch name cannot be empty"
        
        if name in ("HEAD", "-"):
            return False, f"'{name}' is not a valid branch name"
        
        if not self.BRANCH_NAME_PATTERN.match(name):
            return False, "Branch name contains invalid characters"
        
        if name.startswith("-"):
            return False, "Branch name cannot start with '-'"
        
        if ".." in name:
            return False, "Branch name cannot contain '..'"
        
        if name.endswith(".lock"):
            return False, "Branch name cannot end with '.lock'"
        
        return True, "Valid branch name"
    
    async def create_branch(
        self,
        name: str,
        start_point: Optional[str] = None,
        checkout: bool = False,
        track: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Create a new branch.
        
        Args:
            name: Branch name
            start_point: Starting commit/branch (default: HEAD)
            checkout: Switch to branch after creation
            track: Set up tracking for remote branch
            
        Returns:
            Tuple of (success, message)
        """
        # Validate name
        valid, msg = self.validate_branch_name(name)
        if not valid:
            return False, msg
        
        # Build command
        cmd = ["branch"]
        if track:
            cmd.extend(["--track", track])
        cmd.append(name)
        if start_point:
            cmd.append(start_point)
        
        result = await self.git_ops._execute_git(cmd)
        
        if not result.success:
            return False, f"Failed to create branch: {result.message}"
        
        # Checkout if requested
        if checkout:
            checkout_result = await self.git_ops.checkout(name)
            if not checkout_result.success:
                return True, f"Branch created but checkout failed: {checkout_result.message}"
            return True, f"Created and switched to branch '{name}'"
        
        return True, f"Created branch '{name}'"
    
    async def delete_branch(
        self,
        name: str,
        force: bool = False,
        delete_remote: bool = False,
        remote: str = "origin",
    ) -> Tuple[bool, str]:
        """Delete a branch.
        
        Args:
            name: Branch name
            force: Force delete even if not merged
            delete_remote: Also delete remote branch
            remote: Remote name for remote deletion
            
        Returns:
            Tuple of (success, message)
        """
        # Check if it's current branch
        current = await self.get_current_branch()
        if current == name:
            return False, "Cannot delete the current branch"
        
        # Delete local branch
        flag = "-D" if force else "-d"
        result = await self.git_ops._execute_git(["branch", flag, name])
        
        if not result.success:
            if "not fully merged" in (result.message or "").lower():
                return False, f"Branch '{name}' is not fully merged. Use force=True to delete anyway."
            return False, f"Failed to delete branch: {result.message}"
        
        # Delete remote if requested
        if delete_remote:
            remote_result = await self.git_ops._execute_git([
                "push", remote, "--delete", name
            ])
            if not remote_result.success:
                return True, f"Local branch deleted but remote deletion failed: {remote_result.message}"
            return True, f"Deleted branch '{name}' (local and remote)"
        
        return True, f"Deleted branch '{name}'"
    
    async def switch_branch(
        self,
        name: str,
        create: bool = False,
        auto_stash: bool = False,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """Switch to a branch.
        
        Args:
            name: Branch name
            create: Create branch if it doesn't exist
            auto_stash: Automatically stash and pop changes
            force: Force switch even with uncommitted changes
            
        Returns:
            Tuple of (success, message)
        """
        # Check for uncommitted changes
        status = await self.git_ops.status()
        has_changes = False
        if status.success and status.data:
            files = status.data.get("files", [])
            has_changes = len(files) > 0
        
        stashed = False
        if has_changes and not force:
            if auto_stash:
                # Stash changes
                stash_result = await self.git_ops.stash(
                    message="Auto-stash for branch switch"
                )
                if stash_result.success:
                    stashed = True
                else:
                    return False, f"Failed to stash changes: {stash_result.message}"
            else:
                return False, "Working directory has uncommitted changes. Use auto_stash=True or force=True."
        
        # Build checkout command
        cmd = ["checkout"]
        if create:
            cmd.append("-b")
        if force and not auto_stash:
            cmd.append("-f")
        cmd.append(name)
        
        result = await self.git_ops._execute_git(cmd)
        
        if not result.success:
            # Try to pop stash if we stashed
            if stashed:
                await self.git_ops.stash(operation="pop")
            return False, f"Failed to switch branch: {result.message}"
        
        # Pop stash if we stashed
        if stashed:
            pop_result = await self.git_ops.stash(operation="pop")
            if not pop_result.success:
                return True, f"Switched to '{name}' but failed to restore stash: {pop_result.message}"
            return True, f"Switched to '{name}' (changes restored from stash)"
        
        return True, f"Switched to branch '{name}'"
    
    async def compare_branches(
        self,
        base: str,
        compare: str,
    ) -> BranchComparisonResult:
        """Compare two branches.
        
        Args:
            base: Base branch name
            compare: Branch to compare
            
        Returns:
            BranchComparisonResult
        """
        # Get commits ahead
        ahead_result = await self.git_ops._execute_git([
            "rev-list", "--count", f"{base}..{compare}"
        ])
        ahead = int(ahead_result.data.strip()) if ahead_result.success and ahead_result.data else 0
        
        # Get commits behind
        behind_result = await self.git_ops._execute_git([
            "rev-list", "--count", f"{compare}..{base}"
        ])
        behind = int(behind_result.data.strip()) if behind_result.success and behind_result.data else 0
        
        # Get merge base
        merge_base_result = await self.git_ops._execute_git([
            "merge-base", base, compare
        ])
        common_ancestor = None
        if merge_base_result.success and merge_base_result.data:
            common_ancestor = merge_base_result.data.strip()[:8]
        
        return BranchComparisonResult(
            base_branch=base,
            compare_branch=compare,
            ahead=ahead,
            behind=behind,
            diverged=ahead > 0 and behind > 0,
            common_ancestor=common_ancestor,
        )
    
    async def preview_merge(
        self,
        source: str,
        target: Optional[str] = None,
    ) -> MergePreview:
        """Preview a merge operation.
        
        Args:
            source: Source branch to merge
            target: Target branch (default: current)
            
        Returns:
            MergePreview
        """
        if not target:
            target = await self.get_current_branch() or "HEAD"
        
        # Compare branches
        comparison = await self.compare_branches(target, source)
        
        # Check if fast-forward is possible
        can_ff = comparison.behind == 0 and comparison.ahead > 0
        
        # Get files that would change
        diff_result = await self.git_ops._execute_git([
            "diff", "--name-only", f"{target}...{source}"
        ])
        files_changed = 0
        if diff_result.success and diff_result.data:
            files_changed = len([f for f in diff_result.data.strip().split("\n") if f])
        
        # Check for potential conflicts
        # This is a heuristic - check if same files modified in both branches
        conflicts_expected: List[str] = []
        if comparison.diverged:
            base_files_result = await self.git_ops._execute_git([
                "diff", "--name-only", f"{comparison.common_ancestor}..{target}"
            ])
            source_files_result = await self.git_ops._execute_git([
                "diff", "--name-only", f"{comparison.common_ancestor}..{source}"
            ])
            
            base_files = set()
            source_files = set()
            
            if base_files_result.success and base_files_result.data:
                base_files = set(f for f in base_files_result.data.strip().split("\n") if f)
            if source_files_result.success and source_files_result.data:
                source_files = set(f for f in source_files_result.data.strip().split("\n") if f)
            
            conflicts_expected = list(base_files & source_files)
        
        return MergePreview(
            source_branch=source,
            target_branch=target,
            can_fast_forward=can_ff,
            commits_to_merge=comparison.ahead,
            files_changed=files_changed,
            conflicts_expected=conflicts_expected,
        )
    
    async def merge_branch(
        self,
        source: str,
        no_ff: bool = False,
        squash: bool = False,
        message: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Merge a branch.
        
        Args:
            source: Source branch to merge
            no_ff: Create merge commit even if fast-forward possible
            squash: Squash commits
            message: Custom merge commit message
            
        Returns:
            Tuple of (success, message)
        """
        cmd = ["merge"]
        if no_ff:
            cmd.append("--no-ff")
        if squash:
            cmd.append("--squash")
        if message:
            cmd.extend(["-m", message])
        cmd.append(source)
        
        result = await self.git_ops._execute_git(cmd)
        
        if not result.success:
            if "CONFLICT" in (result.message or ""):
                return False, f"Merge resulted in conflicts. Resolve them and commit."
            return False, f"Merge failed: {result.message}"
        
        if squash:
            return True, f"Squashed '{source}'. Don't forget to commit."
        
        return True, f"Merged '{source}' successfully"
    
    async def rename_branch(
        self,
        old_name: str,
        new_name: str,
    ) -> Tuple[bool, str]:
        """Rename a branch.
        
        Args:
            old_name: Current branch name
            new_name: New branch name
            
        Returns:
            Tuple of (success, message)
        """
        # Validate new name
        valid, msg = self.validate_branch_name(new_name)
        if not valid:
            return False, msg
        
        result = await self.git_ops._execute_git([
            "branch", "-m", old_name, new_name
        ])
        
        if result.success:
            return True, f"Renamed branch '{old_name}' to '{new_name}'"
        return False, f"Rename failed: {result.message}"
    
    async def set_upstream(
        self,
        branch: Optional[str] = None,
        upstream: str = "origin",
    ) -> Tuple[bool, str]:
        """Set upstream tracking branch.
        
        Args:
            branch: Local branch (default: current)
            upstream: Remote name
            
        Returns:
            Tuple of (success, message)
        """
        if not branch:
            branch = await self.get_current_branch()
        if not branch:
            return False, "Could not determine current branch"
        
        result = await self.git_ops._execute_git([
            "branch", f"--set-upstream-to={upstream}/{branch}", branch
        ])
        
        if result.success:
            return True, f"Set upstream for '{branch}' to '{upstream}/{branch}'"
        return False, f"Failed to set upstream: {result.message}"


# Convenience function
def get_branch_manager(git_operations: Any) -> GitBranchManager:
    """Get a GitBranchManager instance."""
    return GitBranchManager(git_operations)
