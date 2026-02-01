"""Git Diff Parser for Visualizing Changes.

Phase 7: Git Operations Integration

Provides comprehensive diff parsing including:
- Unified diff format parsing
- Hunk extraction with line numbers
- Word-level diff analysis
- Statistics and summaries
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.git_diff_parser")


class LineType(Enum):
    """Type of diff line."""
    CONTEXT = "context"  # Unchanged line
    ADDITION = "addition"  # Added line
    DELETION = "deletion"  # Deleted line
    HEADER = "header"  # Diff header


@dataclass
class DiffLine:
    """A single line in a diff."""
    content: str
    line_type: LineType
    old_line_num: Optional[int] = None  # Line number in old file
    new_line_num: Optional[int] = None  # Line number in new file
    
    @property
    def prefix(self) -> str:
        """Get the diff line prefix."""
        prefixes = {
            LineType.CONTEXT: " ",
            LineType.ADDITION: "+",
            LineType.DELETION: "-",
            LineType.HEADER: "",
        }
        return prefixes.get(self.line_type, " ")
    
    @property
    def color(self) -> str:
        """Get color for display."""
        colors = {
            LineType.CONTEXT: "white",
            LineType.ADDITION: "green",
            LineType.DELETION: "red",
            LineType.HEADER: "cyan",
        }
        return colors.get(self.line_type, "white")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "line_type": self.line_type.value,
            "old_line_num": self.old_line_num,
            "new_line_num": self.new_line_num,
        }


@dataclass
class WordDiff:
    """Word-level difference."""
    text: str
    is_added: bool
    is_removed: bool
    
    @property
    def color(self) -> str:
        """Get color for display."""
        if self.is_added:
            return "green"
        if self.is_removed:
            return "red"
        return "white"


@dataclass
class DiffHunk:
    """A hunk in a diff (a contiguous block of changes)."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section_header: str = ""  # Function/class name if available
    lines: List[DiffLine] = field(default_factory=list)
    
    @property
    def header(self) -> str:
        """Get the hunk header line."""
        return f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@ {self.section_header}"
    
    @property
    def additions(self) -> int:
        """Count of added lines."""
        return len([l for l in self.lines if l.line_type == LineType.ADDITION])
    
    @property
    def deletions(self) -> int:
        """Count of deleted lines."""
        return len([l for l in self.lines if l.line_type == LineType.DELETION])
    
    @property
    def context_lines(self) -> int:
        """Count of context lines."""
        return len([l for l in self.lines if l.line_type == LineType.CONTEXT])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "old_start": self.old_start,
            "old_count": self.old_count,
            "new_start": self.new_start,
            "new_count": self.new_count,
            "section_header": self.section_header,
            "additions": self.additions,
            "deletions": self.deletions,
            "lines": [l.to_dict() for l in self.lines],
        }


@dataclass
class FileDiff:
    """Diff for a single file."""
    old_path: str
    new_path: str
    hunks: List[DiffHunk] = field(default_factory=list)
    is_binary: bool = False
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    similarity: int = 0  # Similarity percentage for renames
    mode_change: Optional[str] = None
    
    @property
    def path(self) -> str:
        """Get the most relevant path."""
        if self.is_deleted:
            return self.old_path
        return self.new_path
    
    @property
    def additions(self) -> int:
        """Total added lines."""
        return sum(h.additions for h in self.hunks)
    
    @property
    def deletions(self) -> int:
        """Total deleted lines."""
        return sum(h.deletions for h in self.hunks)
    
    @property
    def changes(self) -> int:
        """Total changed lines."""
        return self.additions + self.deletions
    
    @property
    def status_label(self) -> str:
        """Get status label for display."""
        if self.is_binary:
            return "binary"
        if self.is_new:
            return "new file"
        if self.is_deleted:
            return "deleted"
        if self.is_renamed:
            return f"renamed ({self.similarity}% similar)"
        if self.mode_change:
            return f"mode {self.mode_change}"
        return "modified"
    
    @property
    def stats_display(self) -> str:
        """Get stats display string."""
        if self.is_binary:
            return "Binary file"
        return f"+{self.additions} -{self.deletions}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "old_path": self.old_path,
            "new_path": self.new_path,
            "path": self.path,
            "is_binary": self.is_binary,
            "is_new": self.is_new,
            "is_deleted": self.is_deleted,
            "is_renamed": self.is_renamed,
            "similarity": self.similarity,
            "mode_change": self.mode_change,
            "status_label": self.status_label,
            "additions": self.additions,
            "deletions": self.deletions,
            "hunks": [h.to_dict() for h in self.hunks],
        }


@dataclass
class DiffResult:
    """Complete diff result."""
    files: List[FileDiff] = field(default_factory=list)
    commit_info: Optional[Dict[str, str]] = None
    
    @property
    def total_additions(self) -> int:
        """Total added lines across all files."""
        return sum(f.additions for f in self.files)
    
    @property
    def total_deletions(self) -> int:
        """Total deleted lines across all files."""
        return sum(f.deletions for f in self.files)
    
    @property
    def total_changes(self) -> int:
        """Total changed lines."""
        return self.total_additions + self.total_deletions
    
    @property
    def files_changed(self) -> int:
        """Number of files changed."""
        return len(self.files)
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"{self.files_changed} file(s) changed, "
            f"{self.total_additions} insertion(s), "
            f"{self.total_deletions} deletion(s)"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files": [f.to_dict() for f in self.files],
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "files_changed": self.files_changed,
            "summary": self.get_summary(),
            "commit_info": self.commit_info,
        }


class GitDiffParser:
    """Parse git diff output.
    
    Parses unified diff format output from various git diff commands.
    
    Example:
        >>> parser = GitDiffParser()
        >>> result = parser.parse(diff_output)
        >>> 
        >>> print(result.get_summary())
        >>> for file in result.files:
        ...     print(f"{file.path}: {file.stats_display}")
    """
    
    # Regex patterns
    DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
    INDEX_LINE = re.compile(r"^index ([a-f0-9]+)\.\.([a-f0-9]+)(?: (\d+))?$")
    OLD_FILE = re.compile(r"^--- (?:a/)?(.+)$")
    NEW_FILE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
    HUNK_HEADER = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
    )
    BINARY_FILE = re.compile(r"^Binary files")
    RENAME_FROM = re.compile(r"^rename from (.+)$")
    RENAME_TO = re.compile(r"^rename to (.+)$")
    SIMILARITY = re.compile(r"^similarity index (\d+)%$")
    NEW_FILE_MODE = re.compile(r"^new file mode (\d+)$")
    DELETED_FILE_MODE = re.compile(r"^deleted file mode (\d+)$")
    OLD_MODE = re.compile(r"^old mode (\d+)$")
    NEW_MODE = re.compile(r"^new mode (\d+)$")
    
    def __init__(self):
        """Initialize the parser."""
        pass
    
    def parse(self, output: str) -> DiffResult:
        """Parse git diff output.
        
        Args:
            output: Output from git diff command
            
        Returns:
            DiffResult with parsed diff information
        """
        result = DiffResult()
        
        if not output or not output.strip():
            return result
        
        lines = output.split("\n")
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Look for diff header
            header_match = self.DIFF_HEADER.match(line)
            if header_match:
                file_diff, i = self._parse_file_diff(lines, i)
                if file_diff:
                    result.files.append(file_diff)
            else:
                i += 1
        
        return result
    
    def _parse_file_diff(
        self,
        lines: List[str],
        start: int,
    ) -> Tuple[Optional[FileDiff], int]:
        """Parse a single file diff."""
        header_match = self.DIFF_HEADER.match(lines[start])
        if not header_match:
            return None, start + 1
        
        old_path = header_match.group(1)
        new_path = header_match.group(2)
        
        file_diff = FileDiff(old_path=old_path, new_path=new_path)
        i = start + 1
        
        # Parse extended headers
        while i < len(lines):
            line = lines[i]
            
            # Check for next file diff
            if self.DIFF_HEADER.match(line):
                break
            
            # New file
            if self.NEW_FILE_MODE.match(line):
                file_diff.is_new = True
                i += 1
                continue
            
            # Deleted file
            if self.DELETED_FILE_MODE.match(line):
                file_diff.is_deleted = True
                i += 1
                continue
            
            # Mode change
            old_mode_match = self.OLD_MODE.match(line)
            if old_mode_match:
                new_mode_line = lines[i + 1] if i + 1 < len(lines) else ""
                new_mode_match = self.NEW_MODE.match(new_mode_line)
                if new_mode_match:
                    file_diff.mode_change = (
                        f"{old_mode_match.group(1)} → {new_mode_match.group(1)}"
                    )
                    i += 2
                    continue
            
            # Similarity (for renames)
            sim_match = self.SIMILARITY.match(line)
            if sim_match:
                file_diff.similarity = int(sim_match.group(1))
                file_diff.is_renamed = True
                i += 1
                continue
            
            # Rename from
            if self.RENAME_FROM.match(line):
                i += 1
                continue
            
            # Rename to
            if self.RENAME_TO.match(line):
                file_diff.is_renamed = True
                i += 1
                continue
            
            # Binary file
            if self.BINARY_FILE.match(line):
                file_diff.is_binary = True
                i += 1
                continue
            
            # Index line
            if self.INDEX_LINE.match(line):
                i += 1
                continue
            
            # Old file header
            if self.OLD_FILE.match(line):
                i += 1
                continue
            
            # New file header
            if self.NEW_FILE.match(line):
                i += 1
                continue
            
            # Hunk header - parse hunk
            hunk_match = self.HUNK_HEADER.match(line)
            if hunk_match:
                hunk, i = self._parse_hunk(lines, i)
                if hunk:
                    file_diff.hunks.append(hunk)
                continue
            
            i += 1
        
        return file_diff, i
    
    def _parse_hunk(
        self,
        lines: List[str],
        start: int,
    ) -> Tuple[Optional[DiffHunk], int]:
        """Parse a diff hunk."""
        hunk_match = self.HUNK_HEADER.match(lines[start])
        if not hunk_match:
            return None, start + 1
        
        old_start = int(hunk_match.group(1))
        old_count = int(hunk_match.group(2) or 1)
        new_start = int(hunk_match.group(3))
        new_count = int(hunk_match.group(4) or 1)
        section_header = hunk_match.group(5).strip()
        
        hunk = DiffHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            section_header=section_header,
        )
        
        i = start + 1
        old_line = old_start
        new_line = new_start
        
        while i < len(lines):
            line = lines[i]
            
            # Check for end of hunk
            if (
                not line
                or self.DIFF_HEADER.match(line)
                or self.HUNK_HEADER.match(line)
            ):
                break
            
            # Handle "\ No newline at end of file"
            if line.startswith("\\ "):
                i += 1
                continue
            
            if line.startswith("+"):
                # Addition
                hunk.lines.append(DiffLine(
                    content=line[1:],
                    line_type=LineType.ADDITION,
                    new_line_num=new_line,
                ))
                new_line += 1
            elif line.startswith("-"):
                # Deletion
                hunk.lines.append(DiffLine(
                    content=line[1:],
                    line_type=LineType.DELETION,
                    old_line_num=old_line,
                ))
                old_line += 1
            else:
                # Context
                content = line[1:] if line.startswith(" ") else line
                hunk.lines.append(DiffLine(
                    content=content,
                    line_type=LineType.CONTEXT,
                    old_line_num=old_line,
                    new_line_num=new_line,
                ))
                old_line += 1
                new_line += 1
            
            i += 1
        
        return hunk, i
    
    def compute_word_diff(
        self,
        old_text: str,
        new_text: str,
    ) -> List[WordDiff]:
        """Compute word-level differences.
        
        Args:
            old_text: Original text
            new_text: New text
            
        Returns:
            List of WordDiff objects
        """
        # Simple word-level diff
        old_words = old_text.split()
        new_words = new_text.split()
        
        result: List[WordDiff] = []
        
        # Use longest common subsequence approach
        lcs = self._lcs(old_words, new_words)
        
        old_idx = 0
        new_idx = 0
        
        for word in lcs:
            # Add removed words
            while old_idx < len(old_words) and old_words[old_idx] != word:
                result.append(WordDiff(
                    text=old_words[old_idx],
                    is_added=False,
                    is_removed=True,
                ))
                old_idx += 1
            
            # Add new words
            while new_idx < len(new_words) and new_words[new_idx] != word:
                result.append(WordDiff(
                    text=new_words[new_idx],
                    is_added=True,
                    is_removed=False,
                ))
                new_idx += 1
            
            # Add common word
            result.append(WordDiff(
                text=word,
                is_added=False,
                is_removed=False,
            ))
            old_idx += 1
            new_idx += 1
        
        # Handle remaining words
        while old_idx < len(old_words):
            result.append(WordDiff(
                text=old_words[old_idx],
                is_added=False,
                is_removed=True,
            ))
            old_idx += 1
        
        while new_idx < len(new_words):
            result.append(WordDiff(
                text=new_words[new_idx],
                is_added=True,
                is_removed=False,
            ))
            new_idx += 1
        
        return result
    
    def _lcs(self, a: List[str], b: List[str]) -> List[str]:
        """Find longest common subsequence."""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Backtrack
        result: List[str] = []
        i, j = m, n
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                result.append(a[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        
        return list(reversed(result))
    
    def get_side_by_side(
        self,
        hunk: DiffHunk,
    ) -> List[Tuple[Optional[DiffLine], Optional[DiffLine]]]:
        """Convert hunk to side-by-side format.
        
        Args:
            hunk: Diff hunk
            
        Returns:
            List of (old_line, new_line) tuples
        """
        result: List[Tuple[Optional[DiffLine], Optional[DiffLine]]] = []
        
        old_lines: List[DiffLine] = []
        new_lines: List[DiffLine] = []
        
        i = 0
        while i < len(hunk.lines):
            line = hunk.lines[i]
            
            if line.line_type == LineType.CONTEXT:
                # Flush accumulated changes
                self._merge_changes(result, old_lines, new_lines)
                old_lines = []
                new_lines = []
                # Add context line to both sides
                result.append((line, line))
            elif line.line_type == LineType.DELETION:
                old_lines.append(line)
            elif line.line_type == LineType.ADDITION:
                new_lines.append(line)
            
            i += 1
        
        # Flush remaining changes
        self._merge_changes(result, old_lines, new_lines)
        
        return result
    
    def _merge_changes(
        self,
        result: List[Tuple[Optional[DiffLine], Optional[DiffLine]]],
        old_lines: List[DiffLine],
        new_lines: List[DiffLine],
    ) -> None:
        """Merge change blocks into side-by-side format."""
        max_len = max(len(old_lines), len(new_lines))
        for i in range(max_len):
            old = old_lines[i] if i < len(old_lines) else None
            new = new_lines[i] if i < len(new_lines) else None
            result.append((old, new))


# Convenience functions

_parser: Optional[GitDiffParser] = None


def get_git_diff_parser() -> GitDiffParser:
    """Get the global GitDiffParser instance."""
    global _parser
    if _parser is None:
        _parser = GitDiffParser()
    return _parser


def parse_git_diff(output: str) -> DiffResult:
    """Convenience function to parse git diff output."""
    return get_git_diff_parser().parse(output)
