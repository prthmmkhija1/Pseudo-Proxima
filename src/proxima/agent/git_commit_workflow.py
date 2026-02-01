"""Git Commit Workflow for Staged Commits.

Phase 7: Git Operations Integration

Provides comprehensive commit workflow including:
- Staged commit management
- Commit message validation
- Amend commits
- LLM-based message generation
- Push preview with authentication
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.git_commit_workflow")


class CommitType(Enum):
    """Conventional commit types."""
    FEAT = "feat"       # New feature
    FIX = "fix"         # Bug fix
    DOCS = "docs"       # Documentation
    STYLE = "style"     # Formatting
    REFACTOR = "refactor"  # Code restructuring
    TEST = "test"       # Tests
    CHORE = "chore"     # Maintenance
    PERF = "perf"       # Performance
    CI = "ci"           # CI/CD
    BUILD = "build"     # Build system
    REVERT = "revert"   # Revert commit
    
    @property
    def description(self) -> str:
        """Get type description."""
        descriptions = {
            CommitType.FEAT: "A new feature",
            CommitType.FIX: "A bug fix",
            CommitType.DOCS: "Documentation only changes",
            CommitType.STYLE: "Formatting, missing semi colons, etc",
            CommitType.REFACTOR: "Code change that neither fixes a bug nor adds a feature",
            CommitType.TEST: "Adding or updating tests",
            CommitType.CHORE: "Updating grunt tasks etc; no production code change",
            CommitType.PERF: "A code change that improves performance",
            CommitType.CI: "CI configuration files and scripts",
            CommitType.BUILD: "Changes to the build process",
            CommitType.REVERT: "Reverts a previous commit",
        }
        return descriptions.get(self, "")
    
    @property
    def emoji(self) -> str:
        """Get emoji for type."""
        emojis = {
            CommitType.FEAT: "✨",
            CommitType.FIX: "🐛",
            CommitType.DOCS: "📚",
            CommitType.STYLE: "💎",
            CommitType.REFACTOR: "♻️",
            CommitType.TEST: "🧪",
            CommitType.CHORE: "🔧",
            CommitType.PERF: "⚡",
            CommitType.CI: "👷",
            CommitType.BUILD: "📦",
            CommitType.REVERT: "⏪",
        }
        return emojis.get(self, "📝")


@dataclass
class CommitMessageParts:
    """Parts of a commit message."""
    type: Optional[CommitType] = None
    scope: Optional[str] = None
    description: str = ""
    body: Optional[str] = None
    footer: Optional[str] = None
    breaking: bool = False
    
    @property
    def subject(self) -> str:
        """Get the subject line."""
        parts = []
        if self.type:
            parts.append(self.type.value)
            if self.scope:
                parts.append(f"({self.scope})")
            if self.breaking:
                parts.append("!")
            parts.append(": ")
        parts.append(self.description)
        return "".join(parts)
    
    @property
    def full_message(self) -> str:
        """Get the full commit message."""
        lines = [self.subject]
        if self.body:
            lines.extend(["", self.body])
        if self.footer:
            lines.extend(["", self.footer])
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value if self.type else None,
            "scope": self.scope,
            "description": self.description,
            "body": self.body,
            "footer": self.footer,
            "breaking": self.breaking,
            "subject": self.subject,
        }


@dataclass
class ValidationResult:
    """Result of commit message validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


@dataclass
class StagedFile:
    """A staged file for commit."""
    path: str
    status: str  # 'A', 'M', 'D', 'R', 'C'
    additions: int = 0
    deletions: int = 0
    diff_preview: Optional[str] = None


@dataclass
class CommitPreview:
    """Preview of a commit."""
    staged_files: List[StagedFile]
    message: CommitMessageParts
    author: str
    email: str
    timestamp: datetime
    is_amend: bool = False
    parent_hash: Optional[str] = None
    
    @property
    def total_additions(self) -> int:
        """Total added lines."""
        return sum(f.additions for f in self.staged_files)
    
    @property
    def total_deletions(self) -> int:
        """Total deleted lines."""
        return sum(f.deletions for f in self.staged_files)
    
    @property
    def file_count(self) -> int:
        """Number of files."""
        return len(self.staged_files)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "staged_files": [
                {"path": f.path, "status": f.status}
                for f in self.staged_files
            ],
            "message": self.message.to_dict(),
            "author": self.author,
            "email": self.email,
            "timestamp": self.timestamp.isoformat(),
            "is_amend": self.is_amend,
            "file_count": self.file_count,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
        }


@dataclass
class PushPreview:
    """Preview of a push operation."""
    remote: str
    branch: str
    commits: List[Dict[str, str]]  # List of commit info
    ahead: int
    behind: int
    force_required: bool = False
    requires_auth: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "remote": self.remote,
            "branch": self.branch,
            "commits": self.commits,
            "ahead": self.ahead,
            "behind": self.behind,
            "force_required": self.force_required,
            "requires_auth": self.requires_auth,
        }


class CommitMessageValidator:
    """Validate commit messages.
    
    Supports conventional commit format validation.
    
    Example:
        >>> validator = CommitMessageValidator()
        >>> result = validator.validate("feat(ui): add new button")
        >>> if result.valid:
        ...     print("Message is valid!")
    """
    
    # Conventional commit pattern
    CONVENTIONAL_PATTERN = re.compile(
        r"^(?P<type>\w+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<description>.+)$",
        re.MULTILINE,
    )
    
    # Valid types
    VALID_TYPES = {t.value for t in CommitType}
    
    def __init__(
        self,
        max_subject_length: int = 72,
        max_body_line_length: int = 100,
        require_conventional: bool = True,
    ):
        """Initialize validator.
        
        Args:
            max_subject_length: Maximum subject line length
            max_body_line_length: Maximum body line length
            require_conventional: Whether to require conventional commit format
        """
        self.max_subject_length = max_subject_length
        self.max_body_line_length = max_body_line_length
        self.require_conventional = require_conventional
    
    def validate(self, message: str) -> ValidationResult:
        """Validate a commit message.
        
        Args:
            message: Commit message to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []
        
        if not message or not message.strip():
            errors.append("Commit message cannot be empty")
            return ValidationResult(valid=False, errors=errors)
        
        lines = message.strip().split("\n")
        subject = lines[0] if lines else ""
        
        # Check subject length
        if len(subject) > self.max_subject_length:
            errors.append(
                f"Subject line exceeds {self.max_subject_length} characters "
                f"({len(subject)} chars)"
            )
        
        if len(subject) < 10:
            warnings.append("Subject line is very short")
        
        # Check for conventional format
        if self.require_conventional:
            match = self.CONVENTIONAL_PATTERN.match(subject)
            if not match:
                errors.append(
                    "Subject must follow conventional commit format: "
                    "type(scope): description"
                )
            else:
                commit_type = match.group("type")
                if commit_type not in self.VALID_TYPES:
                    errors.append(
                        f"Unknown commit type '{commit_type}'. "
                        f"Valid types: {', '.join(sorted(self.VALID_TYPES))}"
                    )
                
                description = match.group("description")
                if description and description[0].isupper():
                    warnings.append("Description should start with lowercase")
                
                if description and description.endswith("."):
                    warnings.append("Description should not end with period")
        else:
            # Basic validation
            if subject[0].islower():
                suggestions.append("Consider starting with uppercase")
        
        # Check body
        if len(lines) > 1:
            if lines[1].strip():
                warnings.append(
                    "There should be a blank line between subject and body"
                )
            
            # Check body line lengths
            for i, line in enumerate(lines[2:], start=3):
                if len(line) > self.max_body_line_length:
                    warnings.append(
                        f"Line {i} exceeds {self.max_body_line_length} characters"
                    )
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )
    
    def parse(self, message: str) -> CommitMessageParts:
        """Parse a commit message into parts.
        
        Args:
            message: Commit message
            
        Returns:
            CommitMessageParts
        """
        lines = message.strip().split("\n")
        subject = lines[0] if lines else ""
        
        parts = CommitMessageParts(description=subject)
        
        # Try to parse conventional format
        match = self.CONVENTIONAL_PATTERN.match(subject)
        if match:
            type_str = match.group("type")
            for ct in CommitType:
                if ct.value == type_str:
                    parts.type = ct
                    break
            
            parts.scope = match.group("scope")
            parts.breaking = match.group("breaking") == "!"
            parts.description = match.group("description")
        
        # Parse body and footer
        if len(lines) > 2:
            body_lines: List[str] = []
            footer_lines: List[str] = []
            in_footer = False
            
            for line in lines[2:]:
                # Check for footer (key: value or BREAKING CHANGE:)
                if re.match(r"^[\w-]+: ", line) or line.startswith("BREAKING CHANGE:"):
                    in_footer = True
                
                if in_footer:
                    footer_lines.append(line)
                else:
                    body_lines.append(line)
            
            if body_lines:
                parts.body = "\n".join(body_lines).strip()
            if footer_lines:
                parts.footer = "\n".join(footer_lines).strip()
                if "BREAKING CHANGE:" in parts.footer:
                    parts.breaking = True
        
        return parts


class CommitMessageGenerator:
    """Generate commit messages using LLM or templates.
    
    Example:
        >>> generator = CommitMessageGenerator()
        >>> message = generator.generate_from_diff(diff_text)
        >>> print(message.full_message)
    """
    
    # Templates for common patterns
    TEMPLATES: Dict[str, str] = {
        "add_file": "feat: add {filename}",
        "fix_bug": "fix: resolve {issue}",
        "update_docs": "docs: update {filename}",
        "refactor": "refactor: improve {component}",
        "add_tests": "test: add tests for {component}",
        "update_deps": "chore: update dependencies",
        "fix_style": "style: fix formatting in {filename}",
    }
    
    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
    ):
        """Initialize generator.
        
        Args:
            llm_callback: Optional LLM callback for AI-generated messages
        """
        self.llm_callback = llm_callback
    
    def generate_from_diff(
        self,
        diff: str,
        file_paths: List[str],
        use_llm: bool = True,
    ) -> CommitMessageParts:
        """Generate commit message from diff.
        
        Args:
            diff: Diff content
            file_paths: List of changed file paths
            use_llm: Whether to use LLM if available
            
        Returns:
            CommitMessageParts
        """
        # Try LLM first
        if use_llm and self.llm_callback:
            try:
                prompt = self._build_prompt(diff, file_paths)
                response = self.llm_callback(prompt)
                return self._parse_llm_response(response)
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
        
        # Fallback to heuristics
        return self._generate_heuristic(diff, file_paths)
    
    def _build_prompt(self, diff: str, file_paths: List[str]) -> str:
        """Build prompt for LLM."""
        files_summary = "\n".join(f"- {p}" for p in file_paths[:10])
        
        # Truncate diff if too long
        max_diff_len = 3000
        if len(diff) > max_diff_len:
            diff = diff[:max_diff_len] + "\n... (truncated)"
        
        return f"""Generate a conventional commit message for these changes.

Files changed:
{files_summary}

Diff:
```
{diff}
```

Format: type(scope): description

Where type is one of: feat, fix, docs, style, refactor, test, chore, perf, ci, build

Respond with ONLY the commit message, nothing else."""
    
    def _parse_llm_response(self, response: str) -> CommitMessageParts:
        """Parse LLM response."""
        validator = CommitMessageValidator()
        return validator.parse(response.strip())
    
    def _generate_heuristic(
        self,
        diff: str,
        file_paths: List[str],
    ) -> CommitMessageParts:
        """Generate message using heuristics."""
        if not file_paths:
            return CommitMessageParts(
                type=CommitType.CHORE,
                description="update files",
            )
        
        # Analyze file patterns
        extensions = {p.split(".")[-1] for p in file_paths if "." in p}
        
        # Detect commit type
        commit_type = CommitType.CHORE
        scope = None
        description = "update files"
        
        # Check for specific patterns
        if any("test" in p.lower() for p in file_paths):
            commit_type = CommitType.TEST
            description = "update tests"
        elif any(p.endswith(".md") or "doc" in p.lower() for p in file_paths):
            commit_type = CommitType.DOCS
            description = "update documentation"
        elif any(p.endswith((".yml", ".yaml", ".json", ".toml")) for p in file_paths):
            commit_type = CommitType.CHORE
            description = "update configuration"
        
        # Try to extract scope from paths
        if file_paths:
            parts = file_paths[0].split("/")
            if len(parts) > 1:
                scope = parts[-2] if len(parts) > 2 else parts[0]
        
        # Analyze diff for additions/deletions
        additions = diff.count("\n+") - diff.count("\n+++")
        deletions = diff.count("\n-") - diff.count("\n---")
        
        if additions > deletions * 2:
            if commit_type == CommitType.CHORE:
                commit_type = CommitType.FEAT
                description = "add new features"
        elif deletions > additions * 2:
            description = "remove unused code"
        
        return CommitMessageParts(
            type=commit_type,
            scope=scope,
            description=description,
        )
    
    def from_template(
        self,
        template_name: str,
        **kwargs: str,
    ) -> CommitMessageParts:
        """Generate from template.
        
        Args:
            template_name: Name of template
            **kwargs: Template parameters
            
        Returns:
            CommitMessageParts
        """
        template = self.TEMPLATES.get(template_name, "{description}")
        message = template.format(**kwargs)
        
        validator = CommitMessageValidator()
        return validator.parse(message)


class GitCommitWorkflow:
    """Manage git commit workflow.
    
    Provides a structured workflow for staging and committing changes.
    
    Example:
        >>> workflow = GitCommitWorkflow(git_ops)
        >>> 
        >>> # Stage files
        >>> workflow.stage_files(["src/main.py", "src/utils.py"])
        >>> 
        >>> # Generate message
        >>> message = workflow.generate_message()
        >>> 
        >>> # Preview commit
        >>> preview = workflow.preview_commit(message)
        >>> 
        >>> # Execute commit
        >>> result = workflow.commit(message)
    """
    
    def __init__(
        self,
        git_operations: Any,  # GitOperations instance
        llm_callback: Optional[Callable[[str], str]] = None,
    ):
        """Initialize workflow.
        
        Args:
            git_operations: GitOperations instance
            llm_callback: Optional LLM callback for message generation
        """
        self.git_ops = git_operations
        self.validator = CommitMessageValidator()
        self.generator = CommitMessageGenerator(llm_callback)
        self._staged_files: List[str] = []
    
    async def get_staged_files(self) -> List[StagedFile]:
        """Get list of currently staged files."""
        result = await self.git_ops.status()
        if not result.success or not result.data:
            return []
        
        staged = []
        for file_status in result.data.get("files", []):
            if file_status.get("staged"):
                # Get diff stats for the file
                diff_result = await self.git_ops.diff(
                    staged=True,
                    path=file_status["path"],
                )
                
                additions = 0
                deletions = 0
                diff_preview = None
                
                if diff_result.success and diff_result.data:
                    diff_text = diff_result.data.get("diff", "")
                    additions = diff_text.count("\n+") - diff_text.count("\n+++")
                    deletions = diff_text.count("\n-") - diff_text.count("\n---")
                    diff_preview = diff_text[:500] if diff_text else None
                
                staged.append(StagedFile(
                    path=file_status["path"],
                    status=file_status.get("index_status", "M"),
                    additions=additions,
                    deletions=deletions,
                    diff_preview=diff_preview,
                ))
        
        return staged
    
    async def stage_files(self, paths: List[str]) -> bool:
        """Stage files for commit.
        
        Args:
            paths: List of file paths to stage
            
        Returns:
            True if successful
        """
        for path in paths:
            result = await self.git_ops.add(path)
            if not result.success:
                logger.error(f"Failed to stage {path}: {result.message}")
                return False
        
        self._staged_files = paths
        return True
    
    async def unstage_files(self, paths: List[str]) -> bool:
        """Unstage files.
        
        Args:
            paths: List of file paths to unstage
            
        Returns:
            True if successful
        """
        result = await self.git_ops.reset(
            paths=paths,
            mode="mixed",
        )
        return result.success
    
    async def generate_message(
        self,
        use_llm: bool = True,
    ) -> CommitMessageParts:
        """Generate commit message from staged changes.
        
        Args:
            use_llm: Whether to use LLM if available
            
        Returns:
            CommitMessageParts
        """
        # Get staged diff
        diff_result = await self.git_ops.diff(staged=True)
        diff_text = ""
        if diff_result.success and diff_result.data:
            diff_text = diff_result.data.get("diff", "")
        
        # Get staged file paths
        staged = await self.get_staged_files()
        file_paths = [f.path for f in staged]
        
        return self.generator.generate_from_diff(
            diff_text,
            file_paths,
            use_llm=use_llm,
        )
    
    def validate_message(self, message: str) -> ValidationResult:
        """Validate a commit message.
        
        Args:
            message: Commit message
            
        Returns:
            ValidationResult
        """
        return self.validator.validate(message)
    
    async def preview_commit(
        self,
        message: CommitMessageParts,
        amend: bool = False,
    ) -> CommitPreview:
        """Preview a commit before executing.
        
        Args:
            message: Commit message parts
            amend: Whether this is an amend
            
        Returns:
            CommitPreview
        """
        staged = await self.get_staged_files()
        
        # Get author info from git config
        author = "Unknown"
        email = "unknown@example.com"
        
        # Try to get from git config
        try:
            config_name = await self.git_ops._execute_git(["config", "user.name"])
            config_email = await self.git_ops._execute_git(["config", "user.email"])
            if config_name.success:
                author = config_name.data.strip() if config_name.data else author
            if config_email.success:
                email = config_email.data.strip() if config_email.data else email
        except Exception:
            pass
        
        return CommitPreview(
            staged_files=staged,
            message=message,
            author=author,
            email=email,
            timestamp=datetime.now(),
            is_amend=amend,
        )
    
    async def commit(
        self,
        message: CommitMessageParts,
        amend: bool = False,
        allow_empty: bool = False,
    ) -> Tuple[bool, str]:
        """Execute the commit.
        
        Args:
            message: Commit message
            amend: Whether to amend the last commit
            allow_empty: Whether to allow empty commits
            
        Returns:
            Tuple of (success, result_message)
        """
        # Validate first
        validation = self.validate_message(message.full_message)
        if not validation.valid:
            return False, f"Invalid message: {'; '.join(validation.errors)}"
        
        # Execute commit
        result = await self.git_ops.commit(
            message=message.full_message,
            amend=amend,
            allow_empty=allow_empty,
        )
        
        if result.success:
            return True, f"Commit successful: {result.data.get('hash', 'unknown')}"
        else:
            return False, f"Commit failed: {result.message}"
    
    async def preview_push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> PushPreview:
        """Preview a push operation.
        
        Args:
            remote: Remote name
            branch: Branch name (uses current if not specified)
            
        Returns:
            PushPreview
        """
        # Get current branch if not specified
        if not branch:
            status_result = await self.git_ops.status()
            if status_result.success and status_result.data:
                branch = status_result.data.get("branch", {}).get("name", "main")
            else:
                branch = "main"
        
        # Get commits to push
        log_result = await self.git_ops.log(
            count=10,
            format_string="%H|%s|%an",
        )
        
        commits: List[Dict[str, str]] = []
        ahead = 0
        behind = 0
        
        if log_result.success and log_result.data:
            for commit in log_result.data.get("commits", []):
                commits.append({
                    "hash": commit.get("hash", "")[:8],
                    "message": commit.get("message", ""),
                    "author": commit.get("author", ""),
                })
        
        # Check if behind
        try:
            fetch_result = await self.git_ops._execute_git([
                "rev-list", "--count", f"HEAD..{remote}/{branch}"
            ])
            if fetch_result.success and fetch_result.data:
                behind = int(fetch_result.data.strip() or 0)
            
            ahead_result = await self.git_ops._execute_git([
                "rev-list", "--count", f"{remote}/{branch}..HEAD"
            ])
            if ahead_result.success and ahead_result.data:
                ahead = int(ahead_result.data.strip() or 0)
        except Exception:
            pass
        
        return PushPreview(
            remote=remote,
            branch=branch,
            commits=commits[:ahead] if ahead > 0 else commits,
            ahead=ahead,
            behind=behind,
            force_required=behind > 0,
            requires_auth=True,  # Assume auth needed
        )
    
    async def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        set_upstream: bool = False,
    ) -> Tuple[bool, str]:
        """Execute push.
        
        Args:
            remote: Remote name
            branch: Branch name
            force: Whether to force push
            set_upstream: Whether to set upstream tracking
            
        Returns:
            Tuple of (success, result_message)
        """
        result = await self.git_ops.push(
            remote=remote,
            branch=branch,
            force=force,
            set_upstream=set_upstream,
        )
        
        if result.success:
            return True, "Push successful"
        else:
            return False, f"Push failed: {result.message}"


# Convenience function
def get_commit_workflow(
    git_operations: Any,
    llm_callback: Optional[Callable[[str], str]] = None,
) -> GitCommitWorkflow:
    """Get a GitCommitWorkflow instance."""
    return GitCommitWorkflow(git_operations, llm_callback)
