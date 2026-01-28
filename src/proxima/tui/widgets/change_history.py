"""Change History Widget.

Rich widget for displaying the history of all AI-generated changes
with visual indicators and interactive controls.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable, TYPE_CHECKING
from datetime import datetime

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Button, Label, ListView, ListItem
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from rich.text import Text
from rich.table import Table
from rich.console import Console

if TYPE_CHECKING:
    from proxima.tui.dialogs.backend_wizard.change_tracker import (
        ChangeTracker, FileChange, ChangeType
    )
else:
    # Dynamic import to avoid circular dependencies
    try:
        from proxima.tui.dialogs.backend_wizard.change_tracker import (
            ChangeTracker, FileChange, ChangeType
        )
    except ImportError:
        # Fallback types for standalone use
        ChangeTracker = Any
        FileChange = Any
        ChangeType = Any


class ChangeHistoryItem(ListItem):
    """A single item in the change history list."""
    
    DEFAULT_CSS = """
    ChangeHistoryItem {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    
    ChangeHistoryItem.approved {
        background: $success 10%;
        border-left: thick $success;
    }
    
    ChangeHistoryItem.rejected {
        background: $error 10%;
        border-left: thick $error;
    }
    
    ChangeHistoryItem.pending {
        border-left: thick $warning;
    }
    
    ChangeHistoryItem:hover {
        background: $primary 20%;
    }
    
    ChangeHistoryItem.-highlight {
        background: $primary 30%;
    }
    """
    
    def __init__(self, change: FileChange, index: int, **kwargs):
        """Initialize history item.
        
        Args:
            change: The file change
            index: Index in the list
        """
        super().__init__(**kwargs)
        self.change = change
        self.index = index
        self._update_class()
    
    def _update_class(self) -> None:
        """Update CSS class based on approval state."""
        self.remove_class("approved", "rejected", "pending")
        if self.change.approved:
            self.add_class("approved")
        else:
            self.add_class("pending")
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        # Build the display text
        type_icons = {
            ChangeType.CREATE: ("📄", "green", "New"),
            ChangeType.MODIFY: ("📝", "yellow", "Modified"),
            ChangeType.DELETE: ("🗑️", "red", "Deleted"),
        }
        
        icon, color, type_label = type_icons.get(
            self.change.change_type,
            ("📄", "white", "Unknown")
        )
        
        # Status indicator
        status = "✓" if self.change.approved else "○"
        status_color = "green" if self.change.approved else "yellow"
        
        # Build rich text
        text = Text()
        text.append(f"{status} ", style=status_color)
        text.append(f"{icon} ", style="bold")
        text.append(f"{self.change.file_path.split('/')[-1]}", style="bold")
        text.append(f" ({type_label})", style=f"dim {color}")
        
        # Lines changed
        text.append(" ")
        text.append(f"+{self.change.lines_added}", style="green")
        text.append("/")
        text.append(f"-{self.change.lines_removed}", style="red")
        
        yield Static(text)
        
        # Description if present
        if self.change.description:
            yield Static(
                f"  └─ {self.change.description[:50]}...",
                classes="description"
            )


class ChangeHistoryWidget(Widget):
    """Widget displaying the complete change history."""
    
    DEFAULT_CSS = """
    ChangeHistoryWidget {
        width: 100%;
        height: 100%;
        background: $surface;
        border: solid $primary-darken-3;
    }
    
    ChangeHistoryWidget .header {
        width: 100%;
        height: auto;
        background: $primary-darken-2;
        padding: 1;
    }
    
    ChangeHistoryWidget .header-title {
        text-style: bold;
        color: $text;
    }
    
    ChangeHistoryWidget .history-list {
        height: 1fr;
        overflow-y: auto;
    }
    
    ChangeHistoryWidget .footer {
        width: 100%;
        height: auto;
        background: $surface-darken-1;
        padding: 1;
        border-top: solid $primary-darken-3;
    }
    
    ChangeHistoryWidget .stats-text {
        color: $text-muted;
    }
    
    ChangeHistoryWidget .empty-state {
        width: 100%;
        height: 100%;
        align: center middle;
        color: $text-muted;
        padding: 2;
    }
    """
    
    selected_index = reactive(-1)
    
    def __init__(
        self,
        tracker: ChangeTracker,
        on_select: Optional[Callable[[FileChange], None]] = None,
        **kwargs
    ):
        """Initialize change history widget.
        
        Args:
            tracker: Change tracker instance
            on_select: Callback when a change is selected
        """
        super().__init__(**kwargs)
        self.tracker = tracker
        self._on_select = on_select
        self._items: List[ChangeHistoryItem] = []
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        # Header
        with Horizontal(classes="header"):
            yield Static("📋 Change History", classes="header-title")
            yield Static(
                f"({len(self.tracker.changes)} changes)",
                classes="header-count"
            )
        
        # History list
        if self.tracker.changes:
            with ListView(classes="history-list", id="history_list"):
                for i, change in enumerate(self.tracker.changes):
                    item = ChangeHistoryItem(change, i, id=f"history_{i}")
                    self._items.append(item)
                    yield item
        else:
            yield Static(
                "No changes recorded yet.\n"
                "Generate code to see changes here.",
                classes="empty-state"
            )
        
        # Footer with stats
        with Horizontal(classes="footer"):
            stats = self.tracker.get_stats()
            stats_text = Text()
            stats_text.append(f"Files: {stats['total_files']} | ")
            stats_text.append(f"+{stats['lines_added']}", style="green")
            stats_text.append(" / ")
            stats_text.append(f"-{stats['lines_removed']}", style="red")
            stats_text.append(f" | Approved: {stats['approved']}/{stats['total_changes']}")
            
            yield Static(stats_text, classes="stats-text")
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list selection."""
        if event.item and hasattr(event.item, 'index'):
            self.selected_index = event.item.index
            
            if self._on_select and 0 <= self.selected_index < len(self.tracker.changes):
                self._on_select(self.tracker.changes[self.selected_index])
    
    def refresh_history(self) -> None:
        """Refresh the history display."""
        self._items.clear()
        self.refresh(recompose=True)
    
    def highlight_change(self, index: int) -> None:
        """Highlight a specific change.
        
        Args:
            index: Index of the change to highlight
        """
        for i, item in enumerate(self._items):
            if i == index:
                item.add_class("-highlight")
            else:
                item.remove_class("-highlight")


class ChangeTimelineWidget(Widget):
    """Widget showing changes as a timeline."""
    
    DEFAULT_CSS = """
    ChangeTimelineWidget {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    ChangeTimelineWidget .timeline-container {
        width: 100%;
        height: auto;
    }
    
    ChangeTimelineWidget .timeline-item {
        width: 100%;
        height: auto;
        padding: 0 0 0 2;
        border-left: solid $primary;
    }
    
    ChangeTimelineWidget .timeline-item:last-child {
        border-left: none;
    }
    
    ChangeTimelineWidget .timeline-dot {
        color: $primary;
        text-style: bold;
    }
    
    ChangeTimelineWidget .timeline-dot.create {
        color: $success;
    }
    
    ChangeTimelineWidget .timeline-dot.modify {
        color: $warning;
    }
    
    ChangeTimelineWidget .timeline-dot.delete {
        color: $error;
    }
    
    ChangeTimelineWidget .timeline-content {
        margin-left: 1;
    }
    
    ChangeTimelineWidget .timeline-time {
        color: $text-muted;
        text-style: italic;
    }
    """
    
    def __init__(self, tracker: ChangeTracker, **kwargs):
        """Initialize timeline widget.
        
        Args:
            tracker: Change tracker instance
        """
        super().__init__(**kwargs)
        self.tracker = tracker
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Vertical(classes="timeline-container"):
            # Sort changes by timestamp
            sorted_changes = sorted(
                self.tracker.changes,
                key=lambda c: c.timestamp,
                reverse=True
            )
            
            for change in sorted_changes:
                with Horizontal(classes="timeline-item"):
                    # Dot
                    dot_class = f"timeline-dot {change.change_type.value}"
                    yield Static("●", classes=dot_class)
                    
                    # Content
                    with Vertical(classes="timeline-content"):
                        yield Static(
                            f"{change.file_path.split('/')[-1]}",
                            classes="timeline-title"
                        )
                        yield Static(
                            change.timestamp.strftime("%H:%M:%S"),
                            classes="timeline-time"
                        )


class ChangeSummaryTable(Widget):
    """Widget showing a summary table of all changes."""
    
    DEFAULT_CSS = """
    ChangeSummaryTable {
        width: 100%;
        height: auto;
    }
    """
    
    def __init__(self, tracker: ChangeTracker, **kwargs):
        """Initialize summary table.
        
        Args:
            tracker: Change tracker instance
        """
        super().__init__(**kwargs)
        self.tracker = tracker
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        # Build table using Rich
        console = Console()
        
        table = Table(
            title="AI-Generated Changes",
            show_header=True,
            header_style="bold magenta",
        )
        
        table.add_column("Status", style="cyan", width=8)
        table.add_column("File", style="white")
        table.add_column("Type", style="yellow", width=10)
        table.add_column("+/-", style="green", width=12)
        table.add_column("Time", style="dim", width=10)
        
        for change in self.tracker.changes:
            status = "✓" if change.approved else "○"
            status_style = "green" if change.approved else "yellow"
            
            lines = f"+{change.lines_added}/-{change.lines_removed}"
            time_str = change.timestamp.strftime("%H:%M:%S")
            
            table.add_row(
                Text(status, style=status_style),
                change.file_path.split('/')[-1],
                change.change_type.value,
                lines,
                time_str,
            )
        
        # Render to string and display
        with console.capture() as capture:
            console.print(table)
        
        yield Static(capture.get())


class ApprovalProgressWidget(Widget):
    """Widget showing approval progress."""
    
    DEFAULT_CSS = """
    ApprovalProgressWidget {
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
    }
    
    ApprovalProgressWidget .progress-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    ApprovalProgressWidget .progress-bar {
        width: 100%;
        height: 1;
        background: $surface-darken-2;
    }
    
    ApprovalProgressWidget .progress-fill {
        height: 1;
        background: $success;
    }
    
    ApprovalProgressWidget .progress-stats {
        margin-top: 1;
        color: $text-muted;
    }
    """
    
    def __init__(self, tracker: ChangeTracker, **kwargs):
        """Initialize approval progress widget.
        
        Args:
            tracker: Change tracker instance
        """
        super().__init__(**kwargs)
        self.tracker = tracker
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        stats = self.tracker.get_stats()
        total = stats['total_changes']
        approved = stats['approved']
        
        # Progress percentage
        percentage = (approved / total * 100) if total > 0 else 0
        
        yield Static("Approval Progress", classes="progress-title")
        
        # ASCII progress bar
        bar_width = 40
        filled = int(percentage / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        progress_text = Text()
        progress_text.append("[")
        progress_text.append(bar[:filled], style="green")
        progress_text.append(bar[filled:], style="dim")
        progress_text.append(f"] {percentage:.0f}%")
        
        yield Static(progress_text)
        
        # Stats
        stats_text = Text()
        stats_text.append(f"Approved: {approved}/{total}")
        if stats['pending'] > 0:
            stats_text.append(f" | Pending: {stats['pending']}", style="yellow")
        
        yield Static(stats_text, classes="progress-stats")
    
    def refresh_progress(self) -> None:
        """Refresh the progress display."""
        self.refresh(recompose=True)
