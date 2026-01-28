"""Diff Viewer Widget.

A rich text widget for displaying unified or side-by-side diffs
with syntax highlighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, ListView, ListItem
from textual.containers import Horizontal, Vertical, ScrollableContainer
from rich.text import Text
from rich.panel import Panel


class DiffViewMode(Enum):
    """Mode for displaying diff."""
    UNIFIED = "unified"
    SIDE_BY_SIDE = "side_by_side"
    INLINE = "inline"


@dataclass
class DiffLine:
    """Represents a single line in a diff."""
    line_type: str  # 'add', 'remove', 'context', 'header'
    content: str
    old_line_num: Optional[int] = None
    new_line_num: Optional[int] = None
    
    @property
    def prefix(self) -> str:
        """Get the diff prefix for this line."""
        if self.line_type == "add":
            return "+"
        elif self.line_type == "remove":
            return "-"
        elif self.line_type == "header":
            return "@"
        return " "


class DiffViewer(Widget):
    """Widget for viewing file diffs with syntax highlighting."""
    
    DEFAULT_CSS = """
    DiffViewer {
        width: 100%;
        height: 100%;
        background: $surface;
    }
    
    DiffViewer .diff-container {
        width: 100%;
        height: 100%;
        overflow-y: auto;
    }
    
    DiffViewer .diff-header {
        background: $primary-darken-3;
        color: $text;
        padding: 1;
        text-style: bold;
    }
    
    DiffViewer .diff-line {
        width: 100%;
        padding: 0 1;
    }
    
    DiffViewer .line-add {
        background: $success-darken-3;
        color: $success;
    }
    
    DiffViewer .line-remove {
        background: $error-darken-3;
        color: $error;
    }
    
    DiffViewer .line-context {
        background: $surface;
        color: $text-muted;
    }
    
    DiffViewer .line-header {
        background: $primary-darken-2;
        color: $primary;
    }
    
    DiffViewer .line-number {
        width: 6;
        text-align: right;
        padding-right: 1;
        color: $text-disabled;
    }
    
    DiffViewer .stats-bar {
        background: $surface-darken-1;
        padding: 0 1;
        height: 1;
    }
    
    DiffViewer .stats-added {
        color: $success;
    }
    
    DiffViewer .stats-removed {
        color: $error;
    }
    """
    
    def __init__(
        self,
        old_content: str = "",
        new_content: str = "",
        file_name: str = "file",
        mode: DiffViewMode = DiffViewMode.UNIFIED,
        **kwargs
    ):
        """Initialize the diff viewer.
        
        Args:
            old_content: Original file content
            new_content: New file content
            file_name: Name of the file being diffed
            mode: Display mode (unified, side-by-side, inline)
        """
        super().__init__(**kwargs)
        self.old_content = old_content
        self.new_content = new_content
        self.file_name = file_name
        self.mode = mode
        self._diff_lines: List[DiffLine] = []
    
    def compose(self) -> ComposeResult:
        """Create the diff viewer widgets."""
        # Header with file name
        yield Static(f"📄 {self.file_name}", classes="diff-header")
        
        # Stats bar
        stats = self._calculate_stats()
        stats_text = Text()
        stats_text.append(f"+{stats[0]}", style="green")
        stats_text.append(" / ")
        stats_text.append(f"-{stats[1]}", style="red")
        yield Static(stats_text, classes="stats-bar")
        
        # Diff content
        with ScrollableContainer(classes="diff-container"):
            self._generate_diff()
            for diff_line in self._diff_lines:
                yield self._render_diff_line(diff_line)
    
    def _generate_diff(self) -> None:
        """Generate diff lines from old and new content."""
        old_lines = self.old_content.splitlines() if self.old_content else []
        new_lines = self.new_content.splitlines() if self.new_content else []
        
        self._diff_lines = []
        
        # Simple diff algorithm (line-by-line comparison)
        # For production, use difflib.unified_diff or similar
        import difflib
        
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        old_line_num = 1
        new_line_num = 1
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for line in old_lines[i1:i2]:
                    self._diff_lines.append(DiffLine(
                        line_type="context",
                        content=line,
                        old_line_num=old_line_num,
                        new_line_num=new_line_num
                    ))
                    old_line_num += 1
                    new_line_num += 1
            
            elif tag == 'delete':
                for line in old_lines[i1:i2]:
                    self._diff_lines.append(DiffLine(
                        line_type="remove",
                        content=line,
                        old_line_num=old_line_num,
                        new_line_num=None
                    ))
                    old_line_num += 1
            
            elif tag == 'insert':
                for line in new_lines[j1:j2]:
                    self._diff_lines.append(DiffLine(
                        line_type="add",
                        content=line,
                        old_line_num=None,
                        new_line_num=new_line_num
                    ))
                    new_line_num += 1
            
            elif tag == 'replace':
                # Show removed lines first
                for line in old_lines[i1:i2]:
                    self._diff_lines.append(DiffLine(
                        line_type="remove",
                        content=line,
                        old_line_num=old_line_num,
                        new_line_num=None
                    ))
                    old_line_num += 1
                
                # Then added lines
                for line in new_lines[j1:j2]:
                    self._diff_lines.append(DiffLine(
                        line_type="add",
                        content=line,
                        old_line_num=None,
                        new_line_num=new_line_num
                    ))
                    new_line_num += 1
    
    def _render_diff_line(self, diff_line: DiffLine) -> Static:
        """Render a single diff line as a Static widget."""
        text = Text()
        
        # Line numbers
        if self.mode == DiffViewMode.UNIFIED:
            old_num = str(diff_line.old_line_num) if diff_line.old_line_num else ""
            new_num = str(diff_line.new_line_num) if diff_line.new_line_num else ""
            text.append(f"{old_num:>4} {new_num:>4} ", style="dim")
        
        # Prefix and content
        prefix = diff_line.prefix
        content = diff_line.content
        
        if diff_line.line_type == "add":
            text.append(f"{prefix} {content}", style="green")
            css_class = "diff-line line-add"
        elif diff_line.line_type == "remove":
            text.append(f"{prefix} {content}", style="red")
            css_class = "diff-line line-remove"
        elif diff_line.line_type == "header":
            text.append(f"{prefix} {content}", style="cyan")
            css_class = "diff-line line-header"
        else:
            text.append(f"{prefix} {content}", style="dim")
            css_class = "diff-line line-context"
        
        return Static(text, classes=css_class)
    
    def _calculate_stats(self) -> Tuple[int, int]:
        """Calculate lines added and removed."""
        self._generate_diff()
        added = sum(1 for l in self._diff_lines if l.line_type == "add")
        removed = sum(1 for l in self._diff_lines if l.line_type == "remove")
        return added, removed
    
    def update_diff(
        self,
        old_content: str,
        new_content: str,
        file_name: Optional[str] = None
    ) -> None:
        """Update the diff with new content.
        
        Args:
            old_content: New old content
            new_content: New new content
            file_name: Optional new file name
        """
        self.old_content = old_content
        self.new_content = new_content
        if file_name:
            self.file_name = file_name
        self.refresh(recompose=True)


class SideBySideDiffViewer(Widget):
    """Side-by-side diff viewer with synchronized scrolling."""
    
    DEFAULT_CSS = """
    SideBySideDiffViewer {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }
    
    SideBySideDiffViewer .diff-panel {
        width: 1fr;
        height: 100%;
        border: solid $surface-darken-1;
    }
    
    SideBySideDiffViewer .panel-header {
        background: $primary-darken-3;
        padding: 0 1;
        text-style: bold;
    }
    
    SideBySideDiffViewer .old-panel .panel-header {
        color: $error;
    }
    
    SideBySideDiffViewer .new-panel .panel-header {
        color: $success;
    }
    
    SideBySideDiffViewer .panel-content {
        height: 100%;
        overflow-y: auto;
    }
    
    SideBySideDiffViewer .line-removed {
        background: $error-darken-3;
    }
    
    SideBySideDiffViewer .line-added {
        background: $success-darken-3;
    }
    """
    
    def __init__(
        self,
        old_content: str = "",
        new_content: str = "",
        old_label: str = "Original",
        new_label: str = "Modified",
        **kwargs
    ):
        """Initialize side-by-side diff viewer.
        
        Args:
            old_content: Original content
            new_content: Modified content
            old_label: Label for old content panel
            new_label: Label for new content panel
        """
        super().__init__(**kwargs)
        self.old_content = old_content
        self.new_content = new_content
        self.old_label = old_label
        self.new_label = new_label
    
    def compose(self) -> ComposeResult:
        """Create side-by-side panels."""
        # Old content panel
        with Vertical(classes="diff-panel old-panel"):
            yield Static(f"← {self.old_label}", classes="panel-header")
            with ScrollableContainer(classes="panel-content", id="old-scroll"):
                for i, line in enumerate(self.old_content.splitlines(), 1):
                    yield Static(f"{i:>4} │ {line}", classes="diff-line")
        
        # New content panel
        with Vertical(classes="diff-panel new-panel"):
            yield Static(f"{self.new_label} →", classes="panel-header")
            with ScrollableContainer(classes="panel-content", id="new-scroll"):
                for i, line in enumerate(self.new_content.splitlines(), 1):
                    yield Static(f"{i:>4} │ {line}", classes="diff-line")


class InlineDiffViewer(Widget):
    """Inline diff viewer showing changes within lines."""
    
    DEFAULT_CSS = """
    InlineDiffViewer {
        width: 100%;
        height: 100%;
    }
    
    InlineDiffViewer .inline-line {
        width: 100%;
        padding: 0 1;
    }
    """
    
    def __init__(
        self,
        old_content: str = "",
        new_content: str = "",
        **kwargs
    ):
        """Initialize inline diff viewer."""
        super().__init__(**kwargs)
        self.old_content = old_content
        self.new_content = new_content
    
    def compose(self) -> ComposeResult:
        """Create inline diff display."""
        import difflib
        
        old_lines = self.old_content.splitlines()
        new_lines = self.new_content.splitlines()
        
        with ScrollableContainer():
            for i, line in enumerate(difflib.unified_diff(
                old_lines,
                new_lines,
                lineterm=""
            )):
                if line.startswith("+++") or line.startswith("---"):
                    yield Static(line, classes="inline-line line-header")
                elif line.startswith("+"):
                    yield Static(line, classes="inline-line line-add")
                elif line.startswith("-"):
                    yield Static(line, classes="inline-line line-remove")
                elif line.startswith("@@"):
                    yield Static(line, classes="inline-line line-range")
                else:
                    yield Static(line, classes="inline-line")
