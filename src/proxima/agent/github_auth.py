"""GitHub Authentication Module for Proxima Agent.

Provides authentication handling for GitHub operations:
- Uses GitHub CLI (gh) when available
- Falls back to credential prompting
- Supports token-based authentication
- No hardcoded credentials

This module enables the AI agent to properly authenticate with GitHub
for operations like creating repositories, pushing code, etc.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Callable

from proxima.utils.logging import get_logger

logger = get_logger("agent.github_auth")


class AuthStatus(Enum):
    """Authentication status."""
    AUTHENTICATED = "authenticated"
    NOT_AUTHENTICATED = "not_authenticated"
    TOKEN_EXPIRED = "token_expired"
    NO_GH_CLI = "no_gh_cli"
    ERROR = "error"


class AuthMethod(Enum):
    """Authentication method."""
    GH_CLI = "gh_cli"
    CREDENTIAL_HELPER = "credential_helper"
    MANUAL_TOKEN = "manual_token"
    NONE = "none"


@dataclass
class GitHubAuthResult:
    """Result of GitHub authentication check or operation."""
    status: AuthStatus
    method: AuthMethod
    username: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    
    @property
    def is_authenticated(self) -> bool:
        """Check if authenticated."""
        return self.status == AuthStatus.AUTHENTICATED


class GitHubAuth:
    """GitHub Authentication Manager.
    
    Handles GitHub authentication for git operations without hardcoding credentials.
    Uses GitHub CLI (gh) when available, otherwise prompts for credentials.
    
    Example:
        >>> auth = GitHubAuth()
        >>> result = auth.check_auth_status()
        >>> if not result.is_authenticated:
        ...     auth.authenticate(credential_callback=lambda: ("user", input("Token: ")))
    """
    
    def __init__(
        self,
        credential_callback: Optional[Callable[[], Tuple[str, str]]] = None,
    ):
        """Initialize GitHub Auth manager.
        
        Args:
            credential_callback: Optional callback to get credentials (username, token).
                                 Called when authentication is needed.
        """
        self._credential_callback = credential_callback
        self._cached_status: Optional[GitHubAuthResult] = None
        self._gh_cli_path: Optional[str] = None
        
        # Find gh CLI
        self._detect_gh_cli()
        
        logger.info("GitHubAuth initialized", gh_available=self.has_gh_cli)
    
    def _detect_gh_cli(self) -> None:
        """Detect if GitHub CLI is available."""
        self._gh_cli_path = shutil.which("gh")
    
    @property
    def has_gh_cli(self) -> bool:
        """Check if GitHub CLI is available."""
        return self._gh_cli_path is not None
    
    def check_auth_status(self, force_refresh: bool = False) -> GitHubAuthResult:
        """Check current GitHub authentication status.
        
        Args:
            force_refresh: Force a fresh check instead of using cache.
            
        Returns:
            GitHubAuthResult with current status.
        """
        if self._cached_status and not force_refresh:
            return self._cached_status
        
        # Try gh CLI first
        if self.has_gh_cli:
            result = self._check_gh_cli_auth()
            if result.status != AuthStatus.NO_GH_CLI:
                self._cached_status = result
                return result
        
        # Try git credential helper
        result = self._check_credential_helper()
        self._cached_status = result
        return result
    
    def _check_gh_cli_auth(self) -> GitHubAuthResult:
        """Check authentication via gh CLI."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                # Parse username from output
                username = None
                for line in result.stdout.split("\n") + result.stderr.split("\n"):
                    if "Logged in to github.com" in line:
                        # Extract username from line like "✓ Logged in to github.com as USERNAME"
                        parts = line.split(" as ")
                        if len(parts) > 1:
                            username = parts[1].strip().split()[0]
                    elif "account" in line.lower():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.lower() == "account":
                                if i + 1 < len(parts):
                                    username = parts[i + 1].strip("()")
                
                return GitHubAuthResult(
                    status=AuthStatus.AUTHENTICATED,
                    method=AuthMethod.GH_CLI,
                    username=username,
                    message=f"Authenticated via GitHub CLI{f' as {username}' if username else ''}",
                )
            else:
                return GitHubAuthResult(
                    status=AuthStatus.NOT_AUTHENTICATED,
                    method=AuthMethod.GH_CLI,
                    message="Not authenticated. Use 'gh auth login' to authenticate.",
                    error=result.stderr,
                )
        except subprocess.TimeoutExpired:
            return GitHubAuthResult(
                status=AuthStatus.ERROR,
                method=AuthMethod.GH_CLI,
                message="Authentication check timed out",
                error="Command timed out",
            )
        except FileNotFoundError:
            return GitHubAuthResult(
                status=AuthStatus.NO_GH_CLI,
                method=AuthMethod.NONE,
                message="GitHub CLI not found",
            )
        except Exception as e:
            return GitHubAuthResult(
                status=AuthStatus.ERROR,
                method=AuthMethod.GH_CLI,
                message=f"Error checking authentication: {e}",
                error=str(e),
            )
    
    def _check_credential_helper(self) -> GitHubAuthResult:
        """Check authentication via git credential helper."""
        try:
            result = subprocess.run(
                ["git", "config", "--global", "credential.helper"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0 and result.stdout.strip():
                helper = result.stdout.strip()
                return GitHubAuthResult(
                    status=AuthStatus.AUTHENTICATED,
                    method=AuthMethod.CREDENTIAL_HELPER,
                    message=f"Using git credential helper: {helper}",
                )
            else:
                return GitHubAuthResult(
                    status=AuthStatus.NOT_AUTHENTICATED,
                    method=AuthMethod.NONE,
                    message="No git credential helper configured",
                )
        except Exception as e:
            return GitHubAuthResult(
                status=AuthStatus.ERROR,
                method=AuthMethod.NONE,
                message=f"Error checking credential helper: {e}",
                error=str(e),
            )
    
    def authenticate_with_gh_cli(self) -> GitHubAuthResult:
        """Initiate authentication via gh CLI (interactive).
        
        This launches the interactive gh auth login flow.
        
        Returns:
            GitHubAuthResult with authentication status.
        """
        if not self.has_gh_cli:
            return GitHubAuthResult(
                status=AuthStatus.NO_GH_CLI,
                method=AuthMethod.NONE,
                message="GitHub CLI not installed. Install it from https://cli.github.com/",
            )
        
        try:
            # Run gh auth login interactively
            # Note: This won't work in non-interactive contexts
            result = subprocess.run(
                ["gh", "auth", "login"],
                timeout=300,  # 5 minute timeout for interactive login
            )
            
            if result.returncode == 0:
                # Verify authentication
                return self.check_auth_status(force_refresh=True)
            else:
                return GitHubAuthResult(
                    status=AuthStatus.NOT_AUTHENTICATED,
                    method=AuthMethod.GH_CLI,
                    message="Authentication cancelled or failed",
                )
        except subprocess.TimeoutExpired:
            return GitHubAuthResult(
                status=AuthStatus.ERROR,
                method=AuthMethod.GH_CLI,
                message="Authentication timed out",
                error="Process timed out after 5 minutes",
            )
        except Exception as e:
            return GitHubAuthResult(
                status=AuthStatus.ERROR,
                method=AuthMethod.GH_CLI,
                message=f"Authentication error: {e}",
                error=str(e),
            )
    
    def authenticate_with_token(self, token: str, protocol: str = "https") -> GitHubAuthResult:
        """Authenticate using a personal access token.
        
        Args:
            token: GitHub personal access token.
            protocol: Protocol to use (https or ssh).
            
        Returns:
            GitHubAuthResult with authentication status.
        """
        if not self.has_gh_cli:
            # Configure git credential helper instead
            return self._configure_git_credential(token)
        
        try:
            # Use gh auth login with token
            result = subprocess.run(
                ["gh", "auth", "login", "--with-token"],
                input=token,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                self._cached_status = None  # Clear cache
                return self.check_auth_status(force_refresh=True)
            else:
                return GitHubAuthResult(
                    status=AuthStatus.NOT_AUTHENTICATED,
                    method=AuthMethod.GH_CLI,
                    message="Token authentication failed",
                    error=result.stderr,
                )
        except Exception as e:
            return GitHubAuthResult(
                status=AuthStatus.ERROR,
                method=AuthMethod.GH_CLI,
                message=f"Token authentication error: {e}",
                error=str(e),
            )
    
    def _configure_git_credential(self, token: str) -> GitHubAuthResult:
        """Configure git credential helper with token (fallback method)."""
        try:
            # Set up credential helper to store
            subprocess.run(
                ["git", "config", "--global", "credential.helper", "store"],
                check=True,
                timeout=10,
            )
            
            # Write credentials to ~/.git-credentials
            credentials_file = Path.home() / ".git-credentials"
            credential_line = f"https://token:{token}@github.com\n"
            
            with open(credentials_file, "a") as f:
                f.write(credential_line)
            
            return GitHubAuthResult(
                status=AuthStatus.AUTHENTICATED,
                method=AuthMethod.CREDENTIAL_HELPER,
                message="Configured git credential helper with token",
            )
        except Exception as e:
            return GitHubAuthResult(
                status=AuthStatus.ERROR,
                method=AuthMethod.CREDENTIAL_HELPER,
                message=f"Failed to configure credentials: {e}",
                error=str(e),
            )
    
    def create_repo(
        self,
        name: str,
        description: str = "",
        private: bool = False,
        working_dir: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Create a GitHub repository.
        
        Args:
            name: Repository name.
            description: Repository description.
            private: Whether to create a private repo.
            working_dir: Directory to create repo from.
            
        Returns:
            Tuple of (success, message, repo_url).
        """
        # Check authentication first
        auth_result = self.check_auth_status()
        if not auth_result.is_authenticated:
            return (
                False,
                f"Not authenticated with GitHub. {auth_result.message}",
                None,
            )
        
        if not self.has_gh_cli:
            return (
                False,
                "GitHub CLI required to create repositories. Install from https://cli.github.com/",
                None,
            )
        
        try:
            cmd = ["gh", "repo", "create", name]
            
            if description:
                cmd.extend(["--description", description])
            
            if private:
                cmd.append("--private")
            else:
                cmd.append("--public")
            
            # Initialize from current directory if working_dir is provided
            if working_dir:
                cmd.extend(["--source", working_dir])
                cmd.append("--push")  # Push existing code
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=working_dir,
            )
            
            if result.returncode == 0:
                # Parse repo URL from output
                output = result.stdout + result.stderr
                repo_url = None
                for line in output.split("\n"):
                    if "github.com" in line:
                        repo_url = line.strip()
                        break
                
                return (
                    True,
                    f"Successfully created repository: {name}",
                    repo_url,
                )
            else:
                return (
                    False,
                    f"Failed to create repository: {result.stderr}",
                    None,
                )
        except subprocess.TimeoutExpired:
            return (
                False,
                "Repository creation timed out",
                None,
            )
        except Exception as e:
            return (
                False,
                f"Error creating repository: {e}",
                None,
            )
    
    def push_with_auth(
        self,
        repo_path: str,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        set_upstream: bool = False,
    ) -> Tuple[bool, str]:
        """Push to GitHub with proper authentication.
        
        Args:
            repo_path: Path to the repository.
            remote: Remote name.
            branch: Branch to push.
            force: Force push.
            set_upstream: Set upstream tracking.
            
        Returns:
            Tuple of (success, message).
        """
        # Check authentication first
        auth_result = self.check_auth_status()
        if not auth_result.is_authenticated:
            return (
                False,
                f"Not authenticated with GitHub.\n\n{self.get_auth_instructions()}",
            )
        
        try:
            cmd = ["git", "push"]
            
            if force:
                cmd.append("--force")
            
            if set_upstream:
                cmd.append("-u")
            
            cmd.append(remote)
            
            if branch:
                cmd.append(branch)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=repo_path,
            )
            
            if result.returncode == 0:
                return (True, f"Successfully pushed to {remote}")
            else:
                error = result.stderr or result.stdout
                # Check for auth errors
                if "authentication" in error.lower() or "permission" in error.lower():
                    return (
                        False,
                        f"Authentication failed.\n\n{self.get_auth_instructions()}\n\nError: {error}",
                    )
                return (False, f"Push failed: {error}")
        except subprocess.TimeoutExpired:
            return (False, "Push timed out")
        except Exception as e:
            return (False, f"Push error: {e}")
    
    def get_auth_instructions(self) -> str:
        """Get instructions for authenticating with GitHub.
        
        Returns:
            Human-readable authentication instructions.
        """
        if self.has_gh_cli:
            return """**To authenticate with GitHub:**

1. Run this command in your terminal:
   ```
   gh auth login
   ```
2. Follow the prompts to authenticate via browser or token.

Or authenticate with a token:
   ```
   gh auth login --with-token
   ```
   Then paste your GitHub personal access token."""
        else:
            return """**To authenticate with GitHub:**

**Option 1: Install GitHub CLI (Recommended)**
   - Download from: https://cli.github.com/
   - Then run: `gh auth login`

**Option 2: Use Git Credential Helper**
   1. Generate a personal access token at:
      https://github.com/settings/tokens
   2. Run: `git config --global credential.helper store`
   3. The next git push will prompt for credentials.

**Option 3: SSH Key**
   1. Generate SSH key: `ssh-keygen -t ed25519`
   2. Add to GitHub: Settings > SSH and GPG keys
   3. Use SSH URL: `git remote set-url origin git@github.com:user/repo.git`"""
    
    def logout(self) -> GitHubAuthResult:
        """Logout from GitHub.
        
        Returns:
            GitHubAuthResult with logout status.
        """
        if not self.has_gh_cli:
            return GitHubAuthResult(
                status=AuthStatus.NO_GH_CLI,
                method=AuthMethod.NONE,
                message="Cannot logout without GitHub CLI",
            )
        
        try:
            result = subprocess.run(
                ["gh", "auth", "logout"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            self._cached_status = None
            
            return GitHubAuthResult(
                status=AuthStatus.NOT_AUTHENTICATED,
                method=AuthMethod.NONE,
                message="Successfully logged out",
            )
        except Exception as e:
            return GitHubAuthResult(
                status=AuthStatus.ERROR,
                method=AuthMethod.NONE,
                message=f"Logout error: {e}",
                error=str(e),
            )


# Singleton instance
_github_auth: Optional[GitHubAuth] = None


def get_github_auth() -> GitHubAuth:
    """Get the singleton GitHubAuth instance.
    
    Returns:
        GitHubAuth instance.
    """
    global _github_auth
    if _github_auth is None:
        _github_auth = GitHubAuth()
    return _github_auth
