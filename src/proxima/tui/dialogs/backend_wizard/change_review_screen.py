"""Change Review Screen.

Full-screen view for reviewing all changes before deployment.
Includes diff viewer, approval controls, and summary statistics.
"""

from __future__ import annotations

from typing import Optional, Dict, List

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Static, Button, Header, Footer, 
    TabbedContent, TabPane, DataTable,
    Checkbox, Label
)
from textual.containers import (
    Horizontal, Vertical, ScrollableContainer, Center
)
from textual.binding import Binding
from rich.text import Text

from .change_tracker import ChangeTracker, FileChange, ChangeType


class ChangeReviewScreen(Screen):
    """Full-screen review of all pending changes.
    
    Allows users to review, approve, or reject individual
    changes before deployment.
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+a", "approve_all", "Approve All"),
        Binding("ctrl+r", "reject_all", "Reject All"),
        Binding("ctrl+s", "save", "Deploy Changes"),
        Binding("ctrl+z", "undo", "Undo"),
        Binding("ctrl+y", "redo", "Redo"),
        Binding("tab", "next_change", "Next Change"),
        Binding("shift+tab", "prev_change", "Previous Change"),
    ]
    
    DEFAULT_CSS = """
    ChangeReviewScreen {
        background: $surface;
    }
    
    ChangeReviewScreen .main-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }
    
    ChangeReviewScreen .sidebar {
        width: 35;
        height: 100%;
        background: $surface-darken-1;
        border-right: solid $primary;
    }
    
    ChangeReviewScreen .sidebar-header {
        background: $primary-darken-2;
        padding: 1;
        text-style: bold;
    }
    
    ChangeReviewScreen .change-list {
        height: 1fr;
        overflow-y: auto;
    }
    
    ChangeReviewScreen .change-item {
        padding: 1;
        border-bottom: solid $surface;
    }
    
    ChangeReviewScreen .change-item:hover {
        background: $primary-darken-3;
    }
    
    ChangeReviewScreen .change-item.selected {
        background: $primary;
    }
    
    ChangeReviewScreen .change-item.approved {
        border-left: thick $success;
    }
    
    ChangeReviewScreen .change-item.rejected {
        border-left: thick $error;
    }
    
    ChangeReviewScreen .content-area {
        width: 1fr;
        height: 100%;
    }
    
    ChangeReviewScreen .diff-container {
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    ChangeReviewScreen .stats-bar {
        height: 3;
        background: $surface-darken-1;
        padding: 1;
        border-top: solid $primary;
    }
    
    ChangeReviewScreen .stat-item {
        margin-right: 2;
    }
    
    ChangeReviewScreen .action-bar {
        height: 3;
        background: $primary-darken-3;
        padding: 0 1;
        align: center middle;
    }
    
    ChangeReviewScreen .action-button {
        margin: 0 1;
    }
    
    ChangeReviewScreen .summary-panel {
        padding: 1;
        background: $surface-darken-1;
        margin: 1;
        border: solid $primary;
    }
    
    ChangeReviewScreen .empty-state {
        width: 100%;
        height: 100%;
        align: center middle;
        color: $text-muted;
    }
    
    ChangeReviewScreen .file-icon {
        margin-right: 1;
    }
    
    ChangeReviewScreen .lines-added {
        color: $success;
    }
    
    ChangeReviewScreen .lines-removed {
        color: $error;
    }
    """
    
    def __init__(
        self,
        change_tracker: Optional[ChangeTracker] = None,
        backend_name: str = "backend",
        on_deploy: Optional[callable] = None
    ):
        """Initialize the change review screen.
        
        Args:
            change_tracker: ChangeTracker instance with changes
            backend_name: Name of the backend being deployed
            on_deploy: Callback when deployment is confirmed
        """
        super().__init__()
        self.change_tracker = change_tracker or ChangeTracker()
        self.backend_name = backend_name
        self.on_deploy = on_deploy
        self._selected_index = 0
        self._changes: List[FileChange] = []
    
    def compose(self) -> ComposeResult:
        """Create the review screen layout."""
        yield Header(show_clock=True)
        
        with Horizontal(classes="main-container"):
            # Sidebar with change list
            with Vertical(classes="sidebar"):
                yield Static(
                    f"📋 Changes ({len(self.change_tracker.pending_changes)})",
                    classes="sidebar-header"
                )
                
                with ScrollableContainer(classes="change-list"):
                    self._changes = self.change_tracker.pending_changes
                    if self._changes:
                        for i, change in enumerate(self._changes):
                            yield self._create_change_item(change, i)
                    else:
                        yield Static(
                            "No pending changes",
                            classes="empty-state"
                        )
                
                # Summary panel
                with Vertical(classes="summary-panel"):
                    stats = self.change_tracker.get_stats()
                    yield Static(
                        f"Total Files: {stats.get('total', 0)}\n"
                        f"✓ Approved: {stats.get('approved', 0)}\n"
                        f"✗ Rejected: {stats.get('rejected', 0)}\n"
                        f"⏳ Pending: {stats.get('pending', 0)}"
                    )
            
            # Main content area
            with Vertical(classes="content-area"):
                # Tabbed content for diff views
                with TabbedContent():
                    with TabPane("Unified Diff", id="tab-unified"):
                        with ScrollableContainer(classes="diff-container"):
                            yield Static(
                                self._render_unified_diff(),
                                id="diff-display"
                            )
                    
                    with TabPane("Side by Side", id="tab-side"):
                        yield Static(
                            "Side-by-side view",
                            classes="diff-container"
                        )
                    
                    with TabPane("Summary", id="tab-summary"):
                        yield Static(
                            self._render_summary(),
                            id="summary-display"
                        )
                
                # Statistics bar
                with Horizontal(classes="stats-bar"):
                    yield self._create_stats_display()
                
                # Action buttons
                with Horizontal(classes="action-bar"):
                    yield Button(
                        "✓ Approve Selected",
                        id="btn_approve",
                        variant="success",
                        classes="action-button"
                    )
                    yield Button(
                        "✗ Reject Selected",
                        id="btn_reject",
                        variant="error",
                        classes="action-button"
                    )
                    yield Button(
                        "↩ Undo",
                        id="btn_undo",
                        variant="default",
                        classes="action-button"
                    )
                    yield Button(
                        "↪ Redo",
                        id="btn_redo",
                        variant="default",
                        classes="action-button"
                    )
                    yield Button(
                        "🚀 Deploy All",
                        id="btn_deploy",
                        variant="primary",
                        classes="action-button"
                    )
                    yield Button(
                        "Cancel",
                        id="btn_cancel",
                        variant="default",
                        classes="action-button"
                    )
        
        yield Footer()
    
    def _create_change_item(self, change: FileChange, index: int) -> Static:
        """Create a list item for a file change."""
        # Determine icon based on change type
        type_icons = {
            ChangeType.CREATE: "📄",
            ChangeType.MODIFY: "📝",
            ChangeType.DELETE: "🗑️",
        }
        icon = type_icons.get(change.change_type, "📄")
        
        # File name (truncated if needed)
        name = change.file_path.split("/")[-1]
        if len(name) > 20:
            name = name[:17] + "..."
        
        # Lines changed
        added = change.lines_added
        removed = change.lines_removed
        
        text = Text()
        text.append(f"{icon} ", style="bold")
        text.append(f"{name}\n", style="bold")
        text.append(f"   +{added}", style="green")
        text.append(" / ")
        text.append(f"-{removed}", style="red")
        
        # Determine CSS class
        classes = "change-item"
        if index == self._selected_index:
            classes += " selected"
        if change.approved:
            classes += " approved"
        elif hasattr(change, 'rejected') and change.rejected:
            classes += " rejected"
        
        return Static(text, classes=classes, id=f"change-{index}")
    
    def _create_stats_display(self) -> Static:
        """Create the statistics display bar."""
        total_added = sum(c.lines_added for c in self._changes)
        total_removed = sum(c.lines_removed for c in self._changes)
        
        text = Text()
        text.append("📊 ")
        text.append(f"{len(self._changes)} files ", style="bold")
        text.append("| ")
        text.append(f"+{total_added} ", style="green")
        text.append(f"-{total_removed}", style="red")
        
        return Static(text)
    
    def _render_unified_diff(self) -> str:
        """Render the unified diff for selected change."""
        if not self._changes or self._selected_index >= len(self._changes):
            return "No change selected"
        
        change = self._changes[self._selected_index]
        return change.get_unified_diff()
    
    def _render_summary(self) -> str:
        """Render a summary of all changes."""
        lines = [
            f"# Change Summary for '{self.backend_name}'",
            "",
            f"Total files: {len(self._changes)}",
            "",
            "## Files to be created:",
        ]
        
        for change in self._changes:
            if change.change_type == ChangeType.CREATE:
                lines.append(f"  - {change.file_path}")
        
        lines.extend([
            "",
            "## Files to be modified:",
        ])
        
        for change in self._changes:
            if change.change_type == ChangeType.MODIFY:
                lines.append(f"  - {change.file_path}")
        
        return "\n".join(lines)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn_approve":
            self._approve_selected()
        
        elif button_id == "btn_reject":
            self._reject_selected()
        
        elif button_id == "btn_undo":
            self.action_undo()
        
        elif button_id == "btn_redo":
            self.action_redo()
        
        elif button_id == "btn_deploy":
            self._deploy_changes()
        
        elif button_id == "btn_cancel":
            self.action_cancel()
    
    def _approve_selected(self) -> None:
        """Approve the currently selected change."""
        if self._changes and self._selected_index < len(self._changes):
            change = self._changes[self._selected_index]
            change.approved = True
            self.notify(f"Approved: {change.file_path}", severity="information")
            self.refresh()
    
    def _reject_selected(self) -> None:
        """Reject the currently selected change."""
        if self._changes and self._selected_index < len(self._changes):
            change = self._changes[self._selected_index]
            change.approved = False
            if hasattr(change, 'rejected'):
                change.rejected = True
            self.notify(f"Rejected: {change.file_path}", severity="warning")
            self.refresh()
    
    def _deploy_changes(self) -> None:
        """Deploy all approved changes."""
        approved = [c for c in self._changes if c.approved]
        
        if not approved:
            self.notify(
                "No approved changes to deploy",
                severity="warning"
            )
            return
        
        if self.on_deploy:
            self.on_deploy(approved)
        
        self.dismiss(True)
    
    def action_cancel(self) -> None:
        """Cancel and return to previous screen."""
        self.dismiss(False)
    
    def action_approve_all(self) -> None:
        """Approve all pending changes."""
        self.change_tracker.approve_all()
        self.notify("All changes approved", severity="information")
        self.refresh()
    
    def action_reject_all(self) -> None:
        """Reject all pending changes."""
        self.change_tracker.reject_all()
        self.notify("All changes rejected", severity="warning")
        self.refresh()
    
    def action_undo(self) -> None:
        """Undo last action."""
        if self.change_tracker.undo():
            self.notify("Undone", severity="information")
            self.refresh()
        else:
            self.notify("Nothing to undo", severity="warning")
    
    def action_redo(self) -> None:
        """Redo last undone action."""
        if self.change_tracker.redo():
            self.notify("Redone", severity="information")
            self.refresh()
        else:
            self.notify("Nothing to redo", severity="warning")
    
    def action_next_change(self) -> None:
        """Select next change in the list."""
        if self._changes and self._selected_index < len(self._changes) - 1:
            self._selected_index += 1
            self.refresh()
    
    def action_prev_change(self) -> None:
        """Select previous change in the list."""
        if self._changes and self._selected_index > 0:
            self._selected_index -= 1
            self.refresh()
    
    def action_save(self) -> None:
        """Save/deploy all approved changes."""
        self._deploy_changes()
