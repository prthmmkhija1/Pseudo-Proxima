"""Modification Preview System for Safe Code Changes.

Phase 8: Backend Code Modification with Safety

Provides comprehensive modification preview including:
- Side-by-side diff visualization
- Impact analysis
- Syntax highlighting support
- Dry-run mode
- Rich preview formatting
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.modification_preview")


class ModificationScope(Enum):
    """Scope of modification."""
    LINE = "line"           # Single line
    LINES = "lines"         # Multiple lines
    BLOCK = "block"         # Code block (function, class)
    FILE = "file"           # Entire file
    MULTI_FILE = "multi_file"  # Multiple files


class DiffLineType(Enum):
    """Type of diff line."""
    CONTEXT = "context"
    ADDITION = "addition"
    DELETION = "deletion"
    HEADER = "header"


@dataclass
class DiffLine:
    """A single line in a diff."""
    content: str
    line_type: DiffLineType
    old_line_num: Optional[int] = None
    new_line_num: Optional[int] = None
    
    @property
    def prefix(self) -> str:
        """Get line prefix."""
        prefixes = {
            DiffLineType.CONTEXT: " ",
            DiffLineType.ADDITION: "+",
            DiffLineType.DELETION: "-",
            DiffLineType.HEADER: "@",
        }
        return prefixes.get(self.line_type, " ")
    
    @property
    def color(self) -> str:
        """Get color for display."""
        colors = {
            DiffLineType.CONTEXT: "white",
            DiffLineType.ADDITION: "green",
            DiffLineType.DELETION: "red",
            DiffLineType.HEADER: "cyan",
        }
        return colors.get(self.line_type, "white")


@dataclass
class DiffHunk:
    """A hunk in a diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[DiffLine] = field(default_factory=list)
    
    @property
    def header(self) -> str:
        """Get hunk header."""
        return f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@"
    
    @property
    def additions(self) -> int:
        """Count of added lines."""
        return len([l for l in self.lines if l.line_type == DiffLineType.ADDITION])
    
    @property
    def deletions(self) -> int:
        """Count of deleted lines."""
        return len([l for l in self.lines if l.line_type == DiffLineType.DELETION])


@dataclass
class ImpactAnalysis:
    """Analysis of modification impact."""
    lines_added: int = 0
    lines_deleted: int = 0
    lines_modified: int = 0
    old_size_bytes: int = 0
    new_size_bytes: int = 0
    size_change_bytes: int = 0
    functions_affected: List[str] = field(default_factory=list)
    classes_affected: List[str] = field(default_factory=list)
    imports_affected: bool = False
    risk_level: str = "low"  # low, medium, high
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
            "lines_modified": self.lines_modified,
            "old_size_bytes": self.old_size_bytes,
            "new_size_bytes": self.new_size_bytes,
            "size_change_bytes": self.size_change_bytes,
            "size_change_percent": round(
                (self.size_change_bytes / max(self.old_size_bytes, 1)) * 100, 1
            ),
            "functions_affected": self.functions_affected,
            "classes_affected": self.classes_affected,
            "imports_affected": self.imports_affected,
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        parts = []
        
        if self.lines_added:
            parts.append(f"+{self.lines_added} lines")
        if self.lines_deleted:
            parts.append(f"-{self.lines_deleted} lines")
        if self.lines_modified:
            parts.append(f"~{self.lines_modified} modified")
        
        if self.size_change_bytes != 0:
            sign = "+" if self.size_change_bytes > 0 else ""
            parts.append(f"{sign}{self.size_change_bytes} bytes")
        
        return ", ".join(parts) if parts else "No changes"


@dataclass
class ModificationPreview:
    """Complete preview of a modification."""
    id: str
    file_path: str
    modification_type: str
    description: str
    old_content: str
    new_content: str
    diff_text: str
    diff_hunks: List[DiffHunk] = field(default_factory=list)
    impact: Optional[ImpactAnalysis] = None
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    approved: bool = False
    applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "modification_type": self.modification_type,
            "description": self.description,
            "diff_text": self.diff_text[:1000] if self.diff_text else "",
            "impact": self.impact.to_dict() if self.impact else None,
            "timestamp": self.timestamp,
            "approved": self.approved,
            "applied": self.applied,
        }
    
    def get_side_by_side(self) -> List[Tuple[Optional[str], Optional[str]]]:
        """Get side-by-side diff view.
        
        Returns:
            List of (old_line, new_line) tuples
        """
        old_lines = self.old_content.splitlines()
        new_lines = self.new_content.splitlines()
        
        result: List[Tuple[Optional[str], Optional[str]]] = []
        
        # Use difflib to get opcodes
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for old, new in zip(old_lines[i1:i2], new_lines[j1:j2]):
                    result.append((old, new))
            elif tag == "delete":
                for old in old_lines[i1:i2]:
                    result.append((old, None))
            elif tag == "insert":
                for new in new_lines[j1:j2]:
                    result.append((None, new))
            elif tag == "replace":
                # Pair up lines as much as possible
                old_chunk = old_lines[i1:i2]
                new_chunk = new_lines[j1:j2]
                max_len = max(len(old_chunk), len(new_chunk))
                for k in range(max_len):
                    old = old_chunk[k] if k < len(old_chunk) else None
                    new = new_chunk[k] if k < len(new_chunk) else None
                    result.append((old, new))
        
        return result


class ModificationPreviewGenerator:
    """Generate modification previews.
    
    Creates detailed previews of code modifications including:
    - Unified diff
    - Side-by-side comparison
    - Impact analysis
    - Risk assessment
    
    Example:
        >>> generator = ModificationPreviewGenerator()
        >>> preview = generator.generate_replace_preview(
        ...     file_path="src/backend.py",
        ...     old_content=original,
        ...     new_content=modified,
        ...     description="Update function signature"
        ... )
        >>> print(preview.diff_text)
    """
    
    # Patterns for detecting affected code elements
    FUNCTION_PATTERN = re.compile(r"^\s*(async\s+)?def\s+(\w+)")
    CLASS_PATTERN = re.compile(r"^\s*class\s+(\w+)")
    IMPORT_PATTERN = re.compile(r"^\s*(import|from)\s+")
    
    def __init__(self, context_lines: int = 5):
        """Initialize generator.
        
        Args:
            context_lines: Number of context lines in diff
        """
        self.context_lines = context_lines
        self._preview_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique preview ID."""
        self._preview_counter += 1
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"preview_{timestamp}_{self._preview_counter}"
    
    def generate_preview(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        modification_type: str = "modify",
        description: str = "",
    ) -> ModificationPreview:
        """Generate a modification preview.
        
        Args:
            file_path: Path to file
            old_content: Original content
            new_content: Modified content
            modification_type: Type of modification
            description: Description of change
            
        Returns:
            ModificationPreview
        """
        # Generate unified diff
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{Path(file_path).name}",
            tofile=f"b/{Path(file_path).name}",
            n=self.context_lines,
        )
        diff_text = "".join(diff)
        
        # Parse diff into hunks
        hunks = self._parse_diff_hunks(diff_text)
        
        # Analyze impact
        impact = self._analyze_impact(old_content, new_content, hunks)
        
        # Get context
        context_before, context_after = self._get_change_context(
            old_lines, new_lines
        )
        
        return ModificationPreview(
            id=self._generate_id(),
            file_path=file_path,
            modification_type=modification_type,
            description=description,
            old_content=old_content,
            new_content=new_content,
            diff_text=diff_text,
            diff_hunks=hunks,
            impact=impact,
            context_before=context_before,
            context_after=context_after,
        )
    
    def _parse_diff_hunks(self, diff_text: str) -> List[DiffHunk]:
        """Parse unified diff into hunks."""
        hunks: List[DiffHunk] = []
        current_hunk: Optional[DiffHunk] = None
        
        hunk_header = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        
        old_line = 0
        new_line = 0
        
        for line in diff_text.splitlines():
            header_match = hunk_header.match(line)
            
            if header_match:
                if current_hunk:
                    hunks.append(current_hunk)
                
                old_start = int(header_match.group(1))
                old_count = int(header_match.group(2) or 1)
                new_start = int(header_match.group(3))
                new_count = int(header_match.group(4) or 1)
                
                current_hunk = DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                )
                old_line = old_start
                new_line = new_start
                
            elif current_hunk is not None:
                if line.startswith("+++") or line.startswith("---"):
                    continue
                elif line.startswith("+"):
                    current_hunk.lines.append(DiffLine(
                        content=line[1:],
                        line_type=DiffLineType.ADDITION,
                        new_line_num=new_line,
                    ))
                    new_line += 1
                elif line.startswith("-"):
                    current_hunk.lines.append(DiffLine(
                        content=line[1:],
                        line_type=DiffLineType.DELETION,
                        old_line_num=old_line,
                    ))
                    old_line += 1
                elif line.startswith(" "):
                    current_hunk.lines.append(DiffLine(
                        content=line[1:],
                        line_type=DiffLineType.CONTEXT,
                        old_line_num=old_line,
                        new_line_num=new_line,
                    ))
                    old_line += 1
                    new_line += 1
        
        if current_hunk:
            hunks.append(current_hunk)
        
        return hunks
    
    def _analyze_impact(
        self,
        old_content: str,
        new_content: str,
        hunks: List[DiffHunk],
    ) -> ImpactAnalysis:
        """Analyze the impact of changes."""
        impact = ImpactAnalysis(
            old_size_bytes=len(old_content.encode("utf-8")),
            new_size_bytes=len(new_content.encode("utf-8")),
        )
        
        impact.size_change_bytes = impact.new_size_bytes - impact.old_size_bytes
        
        # Count line changes
        for hunk in hunks:
            impact.lines_added += hunk.additions
            impact.lines_deleted += hunk.deletions
        
        # Modified = min of added/deleted (rough approximation)
        impact.lines_modified = min(impact.lines_added, impact.lines_deleted)
        
        # Find affected functions and classes
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        # Track changed line numbers
        changed_old_lines = set()
        changed_new_lines = set()
        
        for hunk in hunks:
            for line in hunk.lines:
                if line.old_line_num and line.line_type == DiffLineType.DELETION:
                    changed_old_lines.add(line.old_line_num - 1)
                if line.new_line_num and line.line_type == DiffLineType.ADDITION:
                    changed_new_lines.add(line.new_line_num - 1)
        
        # Find affected code elements
        impact.functions_affected = self._find_affected_functions(
            old_lines, changed_old_lines
        ) | self._find_affected_functions(new_lines, changed_new_lines)
        impact.functions_affected = list(impact.functions_affected)
        
        impact.classes_affected = self._find_affected_classes(
            old_lines, changed_old_lines
        ) | self._find_affected_classes(new_lines, changed_new_lines)
        impact.classes_affected = list(impact.classes_affected)
        
        # Check if imports affected
        impact.imports_affected = any(
            self.IMPORT_PATTERN.match(old_lines[i])
            for i in changed_old_lines if i < len(old_lines)
        ) or any(
            self.IMPORT_PATTERN.match(new_lines[i])
            for i in changed_new_lines if i < len(new_lines)
        )
        
        # Assess risk
        impact.risk_level, impact.risk_factors = self._assess_risk(impact)
        
        return impact
    
    def _find_affected_functions(
        self,
        lines: List[str],
        changed_lines: set,
    ) -> set:
        """Find functions affected by changes."""
        affected = set()
        current_function = None
        current_indent = 0
        
        for i, line in enumerate(lines):
            match = self.FUNCTION_PATTERN.match(line)
            if match:
                current_function = match.group(2)
                current_indent = len(line) - len(line.lstrip())
            elif current_function:
                # Check if we're still in the function
                stripped = line.lstrip()
                if stripped and not stripped.startswith("#"):
                    indent = len(line) - len(stripped)
                    if indent <= current_indent and not line.strip().startswith("@"):
                        current_function = None
            
            if i in changed_lines and current_function:
                affected.add(current_function)
        
        return affected
    
    def _find_affected_classes(
        self,
        lines: List[str],
        changed_lines: set,
    ) -> set:
        """Find classes affected by changes."""
        affected = set()
        current_class = None
        current_indent = 0
        
        for i, line in enumerate(lines):
            match = self.CLASS_PATTERN.match(line)
            if match:
                current_class = match.group(1)
                current_indent = len(line) - len(line.lstrip())
            elif current_class:
                stripped = line.lstrip()
                if stripped and not stripped.startswith("#"):
                    indent = len(line) - len(stripped)
                    if indent <= current_indent and not line.strip().startswith("@"):
                        current_class = None
            
            if i in changed_lines and current_class:
                affected.add(current_class)
        
        return affected
    
    def _assess_risk(self, impact: ImpactAnalysis) -> Tuple[str, List[str]]:
        """Assess risk level of changes."""
        risk_factors: List[str] = []
        score = 0
        
        # Large number of changes
        if impact.lines_added + impact.lines_deleted > 100:
            risk_factors.append("Large number of line changes (>100)")
            score += 2
        elif impact.lines_added + impact.lines_deleted > 50:
            risk_factors.append("Many line changes (>50)")
            score += 1
        
        # Many functions affected
        if len(impact.functions_affected) > 5:
            risk_factors.append("Multiple functions affected (>5)")
            score += 2
        elif len(impact.functions_affected) > 2:
            risk_factors.append("Multiple functions affected")
            score += 1
        
        # Classes affected
        if len(impact.classes_affected) > 2:
            risk_factors.append("Multiple classes affected")
            score += 2
        elif len(impact.classes_affected) > 0:
            score += 1
        
        # Imports affected
        if impact.imports_affected:
            risk_factors.append("Import statements modified")
            score += 1
        
        # Size change
        if abs(impact.size_change_bytes) > 10000:
            risk_factors.append("Large size change (>10KB)")
            score += 1
        
        # Determine risk level
        if score >= 4:
            return "high", risk_factors
        elif score >= 2:
            return "medium", risk_factors
        else:
            return "low", risk_factors
    
    def _get_change_context(
        self,
        old_lines: List[str],
        new_lines: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Get context before and after changes."""
        # Find first and last changed lines
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        first_change = None
        last_change = None
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                if first_change is None:
                    first_change = i1
                last_change = i2
        
        if first_change is None:
            return [], []
        
        # Get context
        before_start = max(0, first_change - self.context_lines)
        after_end = min(len(old_lines), last_change + self.context_lines)
        
        context_before = old_lines[before_start:first_change]
        context_after = old_lines[last_change:after_end]
        
        return context_before, context_after
    
    def format_preview_text(
        self,
        preview: ModificationPreview,
        include_context: bool = True,
        max_lines: int = 100,
    ) -> str:
        """Format preview as text for display.
        
        Args:
            preview: Preview to format
            include_context: Include context lines
            max_lines: Maximum lines to include
            
        Returns:
            Formatted text
        """
        lines: List[str] = []
        
        # Header
        lines.append(f"File: {preview.file_path}")
        lines.append(f"Type: {preview.modification_type}")
        if preview.description:
            lines.append(f"Description: {preview.description}")
        lines.append("")
        
        # Impact summary
        if preview.impact:
            lines.append("Impact Analysis:")
            lines.append(f"  {preview.impact.get_summary()}")
            lines.append(f"  Risk Level: {preview.impact.risk_level.upper()}")
            if preview.impact.functions_affected:
                lines.append(f"  Functions: {', '.join(preview.impact.functions_affected)}")
            if preview.impact.classes_affected:
                lines.append(f"  Classes: {', '.join(preview.impact.classes_affected)}")
            lines.append("")
        
        # Diff
        lines.append("Changes:")
        lines.append("-" * 60)
        
        diff_lines = preview.diff_text.splitlines()
        if len(diff_lines) > max_lines:
            lines.extend(diff_lines[:max_lines])
            lines.append(f"... ({len(diff_lines) - max_lines} more lines)")
        else:
            lines.extend(diff_lines)
        
        lines.append("-" * 60)
        
        return "\n".join(lines)


# Global instance
_generator: Optional[ModificationPreviewGenerator] = None


def get_preview_generator() -> ModificationPreviewGenerator:
    """Get the global ModificationPreviewGenerator instance."""
    global _generator
    if _generator is None:
        _generator = ModificationPreviewGenerator()
    return _generator


def generate_preview(
    file_path: str,
    old_content: str,
    new_content: str,
    description: str = "",
) -> ModificationPreview:
    """Convenience function to generate a preview."""
    return get_preview_generator().generate_preview(
        file_path=file_path,
        old_content=old_content,
        new_content=new_content,
        description=description,
    )
