"""Git Operations for Proxima Agent.

Provides git operations through terminal execution:
- Clone repositories
- Pull/push changes
- Commit management
- Branch operations
- Status and diff

All operations are executed through the terminal executor
for consistency and proper environment handling.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger
from .terminal_executor import TerminalExecutor, TerminalOutput

logger = get_logger("agent.git")


class GitStatus(Enum):
    """File status in git."""
    MODIFIED = "M"
    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UNTRACKED = "?"
    IGNORED = "!"
    UNMERGED = "U"


@dataclass
class GitFileStatus:
    """Status of a file in git."""
    path: str
    status: GitStatus
    staged: bool = False
    original_path: Optional[str] = None  # For renames
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "status": self.status.value,
            "staged": self.staged,
            "original_path": self.original_path,
        }


@dataclass
class GitBranch:
    """Information about a git branch."""
    name: str
    is_current: bool = False
    is_remote: bool = False
    tracking: Optional[str] = None
    ahead: int = 0
    behind: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "is_current": self.is_current,
            "is_remote": self.is_remote,
            "tracking": self.tracking,
            "ahead": self.ahead,
            "behind": self.behind,
        }


@dataclass
class GitCommit:
    """Information about a git commit."""
    hash: str
    short_hash: str
    author: str
    author_email: str
    date: str
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hash": self.hash,
            "short_hash": self.short_hash,
            "author": self.author,
            "author_email": self.author_email,
            "date": self.date,
            "message": self.message,
        }


@dataclass
class GitResult:
    """Result of a git operation."""
    success: bool
    operation: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    terminal_output: Optional[TerminalOutput] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "operation": self.operation,
            "message": self.message,
            "data": self.data,
            "return_code": self.terminal_output.return_code if self.terminal_output else None,
        }


class GitOperations:
    """Git operations through terminal execution.
    
    Provides git commands with proper parsing and error handling.
    All operations use the terminal executor for execution.
    
    Example:
        >>> git = GitOperations(executor)
        >>> 
        >>> # Clone a repository
        >>> result = git.clone("https://github.com/user/repo.git")
        >>> 
        >>> # Get status
        >>> result = git.status("/path/to/repo")
        >>> print(result.data["files"])
        >>> 
        >>> # Pull changes
        >>> result = git.pull("/path/to/repo")
    """
    
    def __init__(
        self,
        executor: TerminalExecutor,
        default_author: Optional[str] = None,
        default_email: Optional[str] = None,
    ):
        """Initialize git operations.
        
        Args:
            executor: Terminal executor for running commands
            default_author: Default commit author name
            default_email: Default commit author email
        """
        self.executor = executor
        self.default_author = default_author
        self.default_email = default_email
        
        logger.info("GitOperations initialized")
    
    def _execute_git(
        self,
        command: str,
        working_dir: str,
        timeout: Optional[float] = None,
    ) -> TerminalOutput:
        """Execute a git command.
        
        Args:
            command: Git command (without 'git' prefix)
            working_dir: Repository directory
            timeout: Command timeout
            
        Returns:
            Terminal output
        """
        full_command = f"git {command}"
        return self.executor.execute(
            full_command,
            working_dir=working_dir,
            timeout=timeout or 300.0,
        )
    
    def is_repository(self, path: str) -> bool:
        """Check if a path is a git repository.
        
        Args:
            path: Path to check
            
        Returns:
            True if it's a git repository
        """
        git_dir = Path(path) / ".git"
        if git_dir.is_dir():
            return True
        
        # Also check using git command
        output = self._execute_git("rev-parse --git-dir", path)
        return output.success
    
    def clone(
        self,
        url: str,
        destination: Optional[str] = None,
        branch: Optional[str] = None,
        depth: Optional[int] = None,
        working_dir: Optional[str] = None,
        stdout_callback: Optional[Callable[[str], None]] = None,
    ) -> GitResult:
        """Clone a git repository.
        
        Args:
            url: Repository URL
            destination: Destination directory
            branch: Branch to clone
            depth: Shallow clone depth
            working_dir: Working directory for clone
            stdout_callback: Callback for progress output
            
        Returns:
            Git result
        """
        cmd_parts = ["clone"]
        
        if branch:
            cmd_parts.extend(["--branch", branch])
        
        if depth:
            cmd_parts.extend(["--depth", str(depth)])
        
        cmd_parts.append(url)
        
        if destination:
            cmd_parts.append(destination)
        
        command = " ".join(cmd_parts)
        work_dir = working_dir or self.executor.default_working_dir
        
        if stdout_callback:
            output = self.executor.execute_streaming(
                f"git {command}",
                stdout_callback=stdout_callback,
                stderr_callback=stdout_callback,
                working_dir=work_dir,
            )
        else:
            output = self._execute_git(command, work_dir)
        
        if output.success:
            # Determine the actual destination
            actual_dest = destination or url.split("/")[-1].replace(".git", "")
            return GitResult(
                success=True,
                operation="clone",
                message=f"Successfully cloned {url}",
                data={
                    "url": url,
                    "destination": actual_dest,
                    "branch": branch,
                },
                terminal_output=output,
            )
        else:
            return GitResult(
                success=False,
                operation="clone",
                message=f"Clone failed: {output.stderr}",
                data={"url": url},
                terminal_output=output,
            )
    
    def pull(
        self,
        repo_path: str,
        remote: str = "origin",
        branch: Optional[str] = None,
        rebase: bool = False,
        stdout_callback: Optional[Callable[[str], None]] = None,
    ) -> GitResult:
        """Pull changes from remote.
        
        Args:
            repo_path: Path to repository
            remote: Remote name
            branch: Branch to pull
            rebase: Use rebase instead of merge
            stdout_callback: Callback for progress
            
        Returns:
            Git result
        """
        cmd_parts = ["pull"]
        
        if rebase:
            cmd_parts.append("--rebase")
        
        cmd_parts.append(remote)
        
        if branch:
            cmd_parts.append(branch)
        
        command = " ".join(cmd_parts)
        
        if stdout_callback:
            output = self.executor.execute_streaming(
                f"git {command}",
                stdout_callback=stdout_callback,
                stderr_callback=stdout_callback,
                working_dir=repo_path,
            )
        else:
            output = self._execute_git(command, repo_path)
        
        if output.success:
            return GitResult(
                success=True,
                operation="pull",
                message=f"Successfully pulled from {remote}",
                data={
                    "remote": remote,
                    "branch": branch,
                    "output": output.stdout,
                },
                terminal_output=output,
            )
        else:
            return GitResult(
                success=False,
                operation="pull",
                message=f"Pull failed: {output.stderr}",
                data={"remote": remote},
                terminal_output=output,
            )
    
    def push(
        self,
        repo_path: str,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        set_upstream: bool = False,
        stdout_callback: Optional[Callable[[str], None]] = None,
    ) -> GitResult:
        """Push changes to remote.
        
        Args:
            repo_path: Path to repository
            remote: Remote name
            branch: Branch to push
            force: Force push
            set_upstream: Set upstream tracking
            stdout_callback: Callback for progress
            
        Returns:
            Git result
        """
        cmd_parts = ["push"]
        
        if force:
            cmd_parts.append("--force")
        
        if set_upstream:
            cmd_parts.append("-u")
        
        cmd_parts.append(remote)
        
        if branch:
            cmd_parts.append(branch)
        
        command = " ".join(cmd_parts)
        
        if stdout_callback:
            output = self.executor.execute_streaming(
                f"git {command}",
                stdout_callback=stdout_callback,
                stderr_callback=stdout_callback,
                working_dir=repo_path,
            )
        else:
            output = self._execute_git(command, repo_path)
        
        if output.success:
            return GitResult(
                success=True,
                operation="push",
                message=f"Successfully pushed to {remote}",
                data={
                    "remote": remote,
                    "branch": branch,
                },
                terminal_output=output,
            )
        else:
            return GitResult(
                success=False,
                operation="push",
                message=f"Push failed: {output.stderr}",
                data={"remote": remote},
                terminal_output=output,
            )
    
    def status(self, repo_path: str) -> GitResult:
        """Get repository status.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Git result with status information
        """
        # Use porcelain format for easy parsing
        output = self._execute_git("status --porcelain=v1 -b", repo_path)
        
        if not output.success:
            return GitResult(
                success=False,
                operation="status",
                message=f"Status failed: {output.stderr}",
                terminal_output=output,
            )
        
        files: List[GitFileStatus] = []
        branch_info = {}
        
        lines = output.stdout.strip().split("\n") if output.stdout.strip() else []
        
        for line in lines:
            if not line:
                continue
            
            if line.startswith("##"):
                # Branch line
                branch_match = re.match(r"## (\S+)(?:\.\.\.(\S+))?", line)
                if branch_match:
                    branch_info["current"] = branch_match.group(1)
                    if branch_match.group(2):
                        branch_info["tracking"] = branch_match.group(2)
                
                # Check for ahead/behind
                ahead_match = re.search(r"ahead (\d+)", line)
                behind_match = re.search(r"behind (\d+)", line)
                if ahead_match:
                    branch_info["ahead"] = int(ahead_match.group(1))
                if behind_match:
                    branch_info["behind"] = int(behind_match.group(1))
            else:
                # File status line
                if len(line) >= 3:
                    index_status = line[0]
                    worktree_status = line[1]
                    path = line[3:]
                    
                    # Handle renames (path contains " -> ")
                    original = None
                    if " -> " in path:
                        original, path = path.split(" -> ")
                    
                    status_char = worktree_status if worktree_status != " " else index_status
                    staged = index_status != " " and index_status != "?"
                    
                    try:
                        status = GitStatus(status_char)
                    except ValueError:
                        status = GitStatus.MODIFIED
                    
                    files.append(GitFileStatus(
                        path=path,
                        status=status,
                        staged=staged,
                        original_path=original,
                    ))
        
        return GitResult(
            success=True,
            operation="status",
            message="Status retrieved",
            data={
                "branch": branch_info,
                "files": [f.to_dict() for f in files],
                "clean": len(files) == 0,
                "modified_count": len([f for f in files if f.status == GitStatus.MODIFIED]),
                "untracked_count": len([f for f in files if f.status == GitStatus.UNTRACKED]),
            },
            terminal_output=output,
        )
    
    def add(
        self,
        repo_path: str,
        paths: Optional[List[str]] = None,
        all: bool = False,
    ) -> GitResult:
        """Stage files for commit.
        
        Args:
            repo_path: Path to repository
            paths: Specific paths to add
            all: Add all changes
            
        Returns:
            Git result
        """
        if all:
            command = "add -A"
        elif paths:
            paths_str = " ".join(f'"{p}"' for p in paths)
            command = f"add {paths_str}"
        else:
            command = "add ."
        
        output = self._execute_git(command, repo_path)
        
        return GitResult(
            success=output.success,
            operation="add",
            message="Files staged" if output.success else f"Add failed: {output.stderr}",
            data={"paths": paths, "all": all},
            terminal_output=output,
        )
    
    def commit(
        self,
        repo_path: str,
        message: str,
        add_all: bool = False,
        amend: bool = False,
        author: Optional[str] = None,
    ) -> GitResult:
        """Create a commit.
        
        Args:
            repo_path: Path to repository
            message: Commit message
            add_all: Stage all changes first
            amend: Amend the previous commit
            author: Override author
            
        Returns:
            Git result
        """
        if add_all:
            self.add(repo_path, all=True)
        
        cmd_parts = ["commit", "-m", f'"{message}"']
        
        if amend:
            cmd_parts.append("--amend")
        
        if author or self.default_author:
            auth = author or f"{self.default_author} <{self.default_email or 'noreply@proxima.local'}>"
            cmd_parts.extend(["--author", f'"{auth}"'])
        
        command = " ".join(cmd_parts)
        output = self._execute_git(command, repo_path)
        
        if output.success:
            # Parse commit hash from output
            commit_match = re.search(r"\[[\w/-]+ ([a-f0-9]+)\]", output.stdout)
            commit_hash = commit_match.group(1) if commit_match else "unknown"
            
            return GitResult(
                success=True,
                operation="commit",
                message=f"Committed: {commit_hash}",
                data={
                    "hash": commit_hash,
                    "message": message,
                },
                terminal_output=output,
            )
        else:
            return GitResult(
                success=False,
                operation="commit",
                message=f"Commit failed: {output.stderr}",
                data={"message": message},
                terminal_output=output,
            )
    
    def log(
        self,
        repo_path: str,
        limit: int = 10,
        oneline: bool = False,
    ) -> GitResult:
        """Get commit log.
        
        Args:
            repo_path: Path to repository
            limit: Maximum number of commits
            oneline: Use oneline format
            
        Returns:
            Git result with commits
        """
        if oneline:
            command = f"log --oneline -n {limit}"
        else:
            command = f'log --format="%H|%h|%an|%ae|%ad|%s" --date=iso -n {limit}'
        
        output = self._execute_git(command, repo_path)
        
        if not output.success:
            return GitResult(
                success=False,
                operation="log",
                message=f"Log failed: {output.stderr}",
                terminal_output=output,
            )
        
        commits: List[GitCommit] = []
        
        for line in output.stdout.strip().split("\n"):
            if not line:
                continue
            
            if oneline:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    commits.append(GitCommit(
                        hash=parts[0],
                        short_hash=parts[0],
                        author="",
                        author_email="",
                        date="",
                        message=parts[1],
                    ))
            else:
                parts = line.split("|")
                if len(parts) >= 6:
                    commits.append(GitCommit(
                        hash=parts[0],
                        short_hash=parts[1],
                        author=parts[2],
                        author_email=parts[3],
                        date=parts[4],
                        message=parts[5],
                    ))
        
        return GitResult(
            success=True,
            operation="log",
            message=f"Retrieved {len(commits)} commits",
            data={
                "commits": [c.to_dict() for c in commits],
                "count": len(commits),
            },
            terminal_output=output,
        )
    
    def branch(
        self,
        repo_path: str,
        name: Optional[str] = None,
        create: bool = False,
        delete: bool = False,
        list_all: bool = True,
    ) -> GitResult:
        """Manage branches.
        
        Args:
            repo_path: Path to repository
            name: Branch name (for create/delete)
            create: Create new branch
            delete: Delete branch
            list_all: List all branches
            
        Returns:
            Git result
        """
        if create and name:
            output = self._execute_git(f"checkout -b {name}", repo_path)
            return GitResult(
                success=output.success,
                operation="branch",
                message=f"Created branch {name}" if output.success else f"Failed: {output.stderr}",
                data={"name": name, "action": "create"},
                terminal_output=output,
            )
        
        if delete and name:
            output = self._execute_git(f"branch -d {name}", repo_path)
            return GitResult(
                success=output.success,
                operation="branch",
                message=f"Deleted branch {name}" if output.success else f"Failed: {output.stderr}",
                data={"name": name, "action": "delete"},
                terminal_output=output,
            )
        
        # List branches
        output = self._execute_git("branch -vv" + (" -a" if list_all else ""), repo_path)
        
        if not output.success:
            return GitResult(
                success=False,
                operation="branch",
                message=f"Failed: {output.stderr}",
                terminal_output=output,
            )
        
        branches: List[GitBranch] = []
        
        for line in output.stdout.strip().split("\n"):
            if not line:
                continue
            
            is_current = line.startswith("*")
            line = line.lstrip("* ")
            
            is_remote = line.startswith("remotes/")
            if is_remote:
                line = line[8:]  # Remove "remotes/" prefix
            
            parts = line.split()
            if parts:
                name = parts[0]
                
                # Parse tracking info
                tracking = None
                ahead = 0
                behind = 0
                
                bracket_match = re.search(r"\[([^\]]+)\]", line)
                if bracket_match:
                    tracking_info = bracket_match.group(1)
                    tracking_parts = tracking_info.split(":")
                    tracking = tracking_parts[0]
                    
                    if len(tracking_parts) > 1:
                        ahead_match = re.search(r"ahead (\d+)", tracking_parts[1])
                        behind_match = re.search(r"behind (\d+)", tracking_parts[1])
                        if ahead_match:
                            ahead = int(ahead_match.group(1))
                        if behind_match:
                            behind = int(behind_match.group(1))
                
                branches.append(GitBranch(
                    name=name,
                    is_current=is_current,
                    is_remote=is_remote,
                    tracking=tracking,
                    ahead=ahead,
                    behind=behind,
                ))
        
        return GitResult(
            success=True,
            operation="branch",
            message=f"Found {len(branches)} branches",
            data={
                "branches": [b.to_dict() for b in branches],
                "current": next((b.name for b in branches if b.is_current), None),
            },
            terminal_output=output,
        )
    
    def checkout(
        self,
        repo_path: str,
        ref: str,
    ) -> GitResult:
        """Checkout a branch or commit.
        
        Args:
            repo_path: Path to repository
            ref: Branch name or commit hash
            
        Returns:
            Git result
        """
        output = self._execute_git(f"checkout {ref}", repo_path)
        
        return GitResult(
            success=output.success,
            operation="checkout",
            message=f"Checked out {ref}" if output.success else f"Failed: {output.stderr}",
            data={"ref": ref},
            terminal_output=output,
        )
    
    def diff(
        self,
        repo_path: str,
        staged: bool = False,
        path: Optional[str] = None,
    ) -> GitResult:
        """Get diff of changes.
        
        Args:
            repo_path: Path to repository
            staged: Show staged changes
            path: Specific file path
            
        Returns:
            Git result with diff
        """
        cmd_parts = ["diff"]
        
        if staged:
            cmd_parts.append("--cached")
        
        if path:
            cmd_parts.append(f'"{path}"')
        
        command = " ".join(cmd_parts)
        output = self._execute_git(command, repo_path)
        
        return GitResult(
            success=output.success,
            operation="diff",
            message="Diff retrieved" if output.success else f"Failed: {output.stderr}",
            data={
                "diff": output.stdout,
                "staged": staged,
                "path": path,
            },
            terminal_output=output,
        )
    
    def stash(
        self,
        repo_path: str,
        action: str = "push",
        message: Optional[str] = None,
    ) -> GitResult:
        """Manage stash.
        
        Args:
            repo_path: Path to repository
            action: Stash action (push, pop, list, drop)
            message: Stash message (for push)
            
        Returns:
            Git result
        """
        if action == "push" and message:
            command = f'stash push -m "{message}"'
        else:
            command = f"stash {action}"
        
        output = self._execute_git(command, repo_path)
        
        return GitResult(
            success=output.success,
            operation="stash",
            message=f"Stash {action} completed" if output.success else f"Failed: {output.stderr}",
            data={"action": action, "output": output.stdout},
            terminal_output=output,
        )
    
    def reset(
        self,
        repo_path: str,
        mode: str = "mixed",
        ref: str = "HEAD",
    ) -> GitResult:
        """Reset repository state.
        
        Args:
            repo_path: Path to repository
            mode: Reset mode (soft, mixed, hard)
            ref: Reference to reset to
            
        Returns:
            Git result
        """
        output = self._execute_git(f"reset --{mode} {ref}", repo_path)
        
        return GitResult(
            success=output.success,
            operation="reset",
            message=f"Reset to {ref}" if output.success else f"Failed: {output.stderr}",
            data={"mode": mode, "ref": ref},
            terminal_output=output,
        )
    
    def remote(
        self,
        repo_path: str,
        action: str = "list",
        name: Optional[str] = None,
        url: Optional[str] = None,
    ) -> GitResult:
        """Manage remotes.
        
        Args:
            repo_path: Path to repository
            action: Remote action (list, add, remove, get-url)
            name: Remote name
            url: Remote URL (for add)
            
        Returns:
            Git result
        """
        if action == "list":
            output = self._execute_git("remote -v", repo_path)
            
            if output.success:
                remotes = {}
                for line in output.stdout.strip().split("\n"):
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            remote_name = parts[0]
                            remote_url = parts[1]
                            if remote_name not in remotes:
                                remotes[remote_name] = {"fetch": None, "push": None}
                            if "(fetch)" in line:
                                remotes[remote_name]["fetch"] = remote_url
                            elif "(push)" in line:
                                remotes[remote_name]["push"] = remote_url
                
                return GitResult(
                    success=True,
                    operation="remote",
                    message=f"Found {len(remotes)} remotes",
                    data={"remotes": remotes},
                    terminal_output=output,
                )
        
        elif action == "add" and name and url:
            output = self._execute_git(f"remote add {name} {url}", repo_path)
        
        elif action == "remove" and name:
            output = self._execute_git(f"remote remove {name}", repo_path)
        
        elif action == "get-url" and name:
            output = self._execute_git(f"remote get-url {name}", repo_path)
            if output.success:
                return GitResult(
                    success=True,
                    operation="remote",
                    message=f"URL: {output.stdout.strip()}",
                    data={"name": name, "url": output.stdout.strip()},
                    terminal_output=output,
                )
        
        else:
            return GitResult(
                success=False,
                operation="remote",
                message="Invalid remote action or missing parameters",
                data={"action": action},
            )
        
        return GitResult(
            success=output.success,
            operation="remote",
            message=f"Remote {action} completed" if output.success else f"Failed: {output.stderr}",
            data={"action": action, "name": name, "url": url},
            terminal_output=output,
        )
