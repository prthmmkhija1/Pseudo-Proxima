"""Change Management Step for Backend Wizard.

Provides a dedicated wizard step for reviewing all AI-generated changes
before proceeding to deployment. Implements the approval workflow.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import (
    Static, Button, Label, Checkbox, 
    DataTable, TabbedContent, TabPane, RichLog
)
from textual.screen import ModalScreen
from textual.reactive import reactive
from rich.text import Text

from .wizard_state import BackendWizardState
from .change_tracker import ChangeTracker, FileChange, ChangeType


class ChangeItemWidget(Static):
    """Widget displaying a single file change with approval checkbox."""
    
    DEFAULT_CSS = """
    ChangeItemWidget {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
    }
    
    ChangeItemWidget:hover {
        background: $primary-darken-3;
    }
    
    ChangeItemWidget.approved {
        border-left: thick $success;
    }
    
    ChangeItemWidget.rejected {
        border-left: thick $error;
    }
    
    ChangeItemWidget.selected {
        background: $primary-darken-2;
        border: double $primary;
    }
    
    ChangeItemWidget .file-name {
        text-style: bold;
        color: $text;
    }
    
    ChangeItemWidget .change-type {
        color: $accent;
    }
    
    ChangeItemWidget .change-details {
        color: $text-muted;
        margin-left: 2;
    }
    
    ChangeItemWidget .lines-info {
        margin-left: 2;
    }
    
    ChangeItemWidget .lines-added {
        color: $success;
    }
    
    ChangeItemWidget .lines-removed {
        color: $error;
    }
    """
    
    selected = reactive(False)
    
    def __init__(
        self,
        change: FileChange,
        index: int,
        on_select: Optional[callable] = None,
        on_approve: Optional[callable] = None,
        **kwargs
    ):
        """Initialize change item widget.
        
        Args:
            change: The file change to display
            index: Index in the change list
            on_select: Callback when selected
            on_approve: Callback when approval toggled
        """
        super().__init__(**kwargs)
        self.change = change
        self.index = index
        self._on_select = on_select
        self._on_approve = on_approve
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        # Type icons
        type_icons = {
            ChangeType.CREATE: "📄 New",
            ChangeType.MODIFY: "📝 Modified",
            ChangeType.DELETE: "🗑️ Deleted",
        }
        type_label = type_icons.get(self.change.change_type, "📄")
        
        with Horizontal():
            # Approval checkbox
            yield Checkbox(
                "",
                value=self.change.approved,
                id=f"approve_{self.index}"
            )
            
            # File info
            with Vertical():
                yield Static(
                    f"{type_label}: {self.change.file_path}",
                    classes="file-name"
                )
                
                if self.change.description:
                    yield Static(
                        f"  • {self.change.description}",
                        classes="change-details"
                    )
                
                # Lines info
                lines_text = Text()
                lines_text.append(f"  +{self.change.lines_added}", style="green")
                lines_text.append(" / ")
                lines_text.append(f"-{self.change.lines_removed}", style="red")
                lines_text.append(" lines")
                yield Static(lines_text, classes="lines-info")
            
            # View diff button
            yield Button("View Diff", id=f"diff_{self.index}", variant="default")
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle approval checkbox toggle."""
        self.change.approved = event.value
        
        self.remove_class("approved", "rejected")
        if event.value:
            self.add_class("approved")
        
        if self._on_approve:
            self._on_approve(self.index, event.value)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id.startswith("diff_"):
            if self._on_select:
                self._on_select(self.index)
    
    def watch_selected(self, selected: bool) -> None:
        """Watch selected state changes."""
        if selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")


class DiffPreviewWidget(Static):
    """Widget showing diff preview of selected change."""
    
    DEFAULT_CSS = """
    DiffPreviewWidget {
        width: 100%;
        height: 100%;
        background: $surface;
        border: solid $primary-darken-3;
    }
    
    DiffPreviewWidget .diff-header {
        background: $primary-darken-2;
        padding: 1;
        text-style: bold;
    }
    
    DiffPreviewWidget .diff-content {
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    DiffPreviewWidget .line-add {
        color: $success;
        background: $success 10%;
    }
    
    DiffPreviewWidget .line-remove {
        color: $error;
        background: $error 10%;
    }
    
    DiffPreviewWidget .line-context {
        color: $text-muted;
    }
    
    DiffPreviewWidget .line-header {
        color: $primary;
        text-style: bold;
    }
    
    DiffPreviewWidget .empty-state {
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    """
    
    def __init__(self, change: Optional[FileChange] = None, **kwargs):
        """Initialize diff preview.
        
        Args:
            change: File change to preview
        """
        super().__init__(**kwargs)
        self.change = change
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        if not self.change:
            yield Static(
                "Select a change to view its diff",
                classes="empty-state"
            )
            return
        
        # Header
        yield Static(
            f"Diff: {self.change.file_path}",
            classes="diff-header"
        )
        
        # Diff content
        with ScrollableContainer(classes="diff-content"):
            diff_text = self.change.get_unified_diff()
            
            for line in diff_text.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    yield Static(line, classes="line-add")
                elif line.startswith('-') and not line.startswith('---'):
                    yield Static(line, classes="line-remove")
                elif line.startswith('@@'):
                    yield Static(line, classes="line-header")
                else:
                    yield Static(line, classes="line-context")
    
    def update_change(self, change: FileChange) -> None:
        """Update the displayed change."""
        self.change = change
        self.refresh(recompose=True)


class ChangeStatsWidget(Static):
    """Widget showing change statistics."""
    
    DEFAULT_CSS = """
    ChangeStatsWidget {
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
    }
    
    ChangeStatsWidget .stat-row {
        width: 100%;
    }
    
    ChangeStatsWidget .stat-label {
        color: $text-muted;
    }
    
    ChangeStatsWidget .stat-value {
        color: $text;
        text-style: bold;
    }
    
    ChangeStatsWidget .approved-count {
        color: $success;
    }
    
    ChangeStatsWidget .pending-count {
        color: $warning;
    }
    """
    
    def __init__(self, tracker: ChangeTracker, **kwargs):
        """Initialize stats widget.
        
        Args:
            tracker: Change tracker instance
        """
        super().__init__(**kwargs)
        self.tracker = tracker
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        stats = self.tracker.get_stats()
        
        yield Static("📊 Change Summary", classes="stat-label")
        yield Static("")
        
        with Horizontal(classes="stat-row"):
            yield Static(f"Total Files: ", classes="stat-label")
            yield Static(f"{stats['total_files']}", classes="stat-value")
        
        with Horizontal(classes="stat-row"):
            yield Static(f"Lines Added: ", classes="stat-label")
            text = Text()
            text.append(f"+{stats['lines_added']}", style="green")
            yield Static(text)
        
        with Horizontal(classes="stat-row"):
            yield Static(f"Lines Removed: ", classes="stat-label")
            text = Text()
            text.append(f"-{stats['lines_removed']}", style="red")
            yield Static(text)
        
        yield Static("")
        
        with Horizontal(classes="stat-row"):
            yield Static(f"Approved: ", classes="stat-label")
            yield Static(f"{stats['approved']}", classes="approved-count")
        
        with Horizontal(classes="stat-row"):
            yield Static(f"Pending: ", classes="stat-label")
            yield Static(f"{stats['pending']}", classes="pending-count")
    
    def refresh_stats(self) -> None:
        """Refresh the statistics display."""
        self.refresh(recompose=True)


class ChangeManagementStepScreen(ModalScreen[dict]):
    """Step 6.5: Change Management and Approval Screen.
    
    Allows users to review all AI-generated changes before deployment.
    Supports individual approval/rejection, undo/redo, and export.
    """
    
    DEFAULT_CSS = """
    ChangeManagementStepScreen {
        align: center middle;
    }
    
    ChangeManagementStepScreen .wizard-container {
        width: 95%;
        height: 90%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    ChangeManagementStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    ChangeManagementStepScreen .main-content {
        height: 1fr;
        layout: horizontal;
    }
    
    ChangeManagementStepScreen .changes-panel {
        width: 50%;
        height: 100%;
        padding: 1;
    }
    
    ChangeManagementStepScreen .preview-panel {
        width: 50%;
        height: 100%;
        padding: 1;
    }
    
    ChangeManagementStepScreen .changes-list {
        height: 1fr;
        overflow-y: auto;
    }
    
    ChangeManagementStepScreen .action-bar {
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border-top: solid $primary-darken-3;
    }
    
    ChangeManagementStepScreen .nav-buttons {
        width: 100%;
        height: auto;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    ChangeManagementStepScreen .action-button {
        margin: 0 1;
    }
    
    ChangeManagementStepScreen .nav-button {
        margin: 0 1;
        min-width: 14;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+z", "undo", "Undo"),
        ("ctrl+y", "redo", "Redo"),
        ("ctrl+a", "approve_all", "Approve All"),
    ]
    
    selected_index = reactive(-1)
    
    def __init__(self, state: BackendWizardState, tracker: Optional[ChangeTracker] = None):
        """Initialize change management screen.
        
        Args:
            state: Wizard state
            tracker: Change tracker (creates new if None)
        """
        super().__init__()
        self.state = state
        self.tracker = tracker or ChangeTracker()
        self._change_items: List[ChangeItemWidget] = []
    
    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "📋 Change Management - Review AI-Generated Changes",
                    classes="wizard-title"
                )
                
                yield Static(
                    "Review and approve all changes before deployment:",
                    classes="field-hint"
                )
                
                # Main content - split view
                with Horizontal(classes="main-content"):
                    # Left panel - changes list
                    with Vertical(classes="changes-panel"):
                        yield Static("AI-Generated Changes:", classes="section-title")
                        
                        with ScrollableContainer(classes="changes-list", id="changes_list"):
                            if self.tracker.changes:
                                for i, change in enumerate(self.tracker.changes):
                                    item = ChangeItemWidget(
                                        change=change,
                                        index=i,
                                        on_select=self._on_change_selected,
                                        on_approve=self._on_change_approved,
                                        id=f"change_item_{i}"
                                    )
                                    self._change_items.append(item)
                                    yield item
                            else:
                                yield Static(
                                    "No changes to review.\n"
                                    "Generate code first to see changes here.",
                                    classes="empty-state"
                                )
                        
                        # Stats
                        yield ChangeStatsWidget(self.tracker, id="stats_widget")
                    
                    # Right panel - diff preview
                    with Vertical(classes="preview-panel"):
                        yield Static("Diff Preview:", classes="section-title")
                        yield DiffPreviewWidget(id="diff_preview")
                
                # Action bar
                with Horizontal(classes="action-bar"):
                    yield Button("↩ Undo", id="btn_undo", classes="action-button")
                    yield Button("↪ Redo", id="btn_redo", classes="action-button")
                    yield Button(
                        "✓ Approve All",
                        id="btn_approve_all",
                        variant="success",
                        classes="action-button"
                    )
                    yield Button(
                        "✗ Reject All",
                        id="btn_reject_all",
                        variant="error",
                        classes="action-button"
                    )
                    yield Button("📄 Export", id="btn_export", classes="action-button")
                    yield Button("🔍 View All Diffs", id="btn_view_all", classes="action-button")
                
                # Navigation buttons
                with Horizontal(classes="nav-buttons"):
                    yield Button(
                        "← Back",
                        id="btn_back",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Cancel",
                        id="btn_cancel",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Next →",
                        id="btn_next",
                        variant="primary",
                        classes="nav-button",
                        disabled=not self._can_proceed()
                    )
    
    def _can_proceed(self) -> bool:
        """Check if user can proceed to next step."""
        if not self.tracker.changes:
            return True  # No changes to approve
        
        # All changes must be approved to proceed
        return all(c.approved for c in self.tracker.changes)
    
    def _on_change_selected(self, index: int) -> None:
        """Handle change selection."""
        self.selected_index = index
        
        # Update selection state
        for i, item in enumerate(self._change_items):
            item.selected = (i == index)
        
        # Update diff preview
        if 0 <= index < len(self.tracker.changes):
            preview = self.query_one("#diff_preview", DiffPreviewWidget)
            preview.update_change(self.tracker.changes[index])
    
    def _on_change_approved(self, index: int, approved: bool) -> None:
        """Handle change approval toggle."""
        self._refresh_stats()
        self._update_next_button()
    
    def _refresh_stats(self) -> None:
        """Refresh the stats widget."""
        try:
            stats = self.query_one("#stats_widget", ChangeStatsWidget)
            stats.refresh_stats()
        except Exception:
            pass
    
    def _update_next_button(self) -> None:
        """Update the next button state."""
        try:
            btn = self.query_one("#btn_next", Button)
            btn.disabled = not self._can_proceed()
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn_back":
            self.dismiss({"action": "back", "state": self.state})
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_next":
            if self._can_proceed():
                self.dismiss({
                    "action": "next",
                    "state": self.state,
                    "tracker": self.tracker
                })
        
        elif event.button.id == "btn_undo":
            self.action_undo()
        
        elif event.button.id == "btn_redo":
            self.action_redo()
        
        elif event.button.id == "btn_approve_all":
            self.action_approve_all()
        
        elif event.button.id == "btn_reject_all":
            self._confirm_reject_all()
        
        elif event.button.id == "btn_export":
            self._export_changes()
        
        elif event.button.id == "btn_view_all":
            self._view_all_diffs()
    
    def action_undo(self) -> None:
        """Undo last change."""
        if self.tracker.undo():
            self.notify("Undone last change", severity="information")
            self._refresh_changes_list()
        else:
            self.notify("Nothing to undo", severity="warning")
    
    def action_redo(self) -> None:
        """Redo last undone change."""
        if self.tracker.redo():
            self.notify("Redone last change", severity="information")
            self._refresh_changes_list()
        else:
            self.notify("Nothing to redo", severity="warning")
    
    def action_approve_all(self) -> None:
        """Approve all changes."""
        count = self.tracker.approve_all()
        self.notify(f"Approved {count} changes", severity="information")
        
        # Update all checkboxes
        for item in self._change_items:
            try:
                checkbox = item.query_one(Checkbox)
                checkbox.value = True
            except Exception:
                pass
        
        self._refresh_stats()
        self._update_next_button()
    
    def _confirm_reject_all(self) -> None:
        """Show confirmation dialog for rejecting all changes."""
        
        class RejectConfirmDialog(ModalScreen[bool]):
            """Confirmation dialog for rejecting all changes."""
            
            DEFAULT_CSS = """
            RejectConfirmDialog {
                align: center middle;
            }
            
            RejectConfirmDialog .dialog-container {
                width: 50;
                height: auto;
                border: double $error;
                background: $surface;
                padding: 2;
            }
            
            RejectConfirmDialog .dialog-title {
                text-align: center;
                text-style: bold;
                color: $error;
                margin-bottom: 1;
            }
            
            RejectConfirmDialog .dialog-message {
                text-align: center;
                margin-bottom: 1;
            }
            
            RejectConfirmDialog .button-row {
                width: 100%;
                height: auto;
                align: center middle;
            }
            """
            
            def compose(self) -> ComposeResult:
                with Center():
                    with Vertical(classes="dialog-container"):
                        yield Static("⚠️ Reject All Changes?", classes="dialog-title")
                        yield Static(
                            "This will remove all AI-generated changes.\n"
                            "This action cannot be undone.",
                            classes="dialog-message"
                        )
                        with Horizontal(classes="button-row"):
                            yield Button("Cancel", id="cancel")
                            yield Button("Reject All", id="confirm", variant="error")
            
            def on_button_pressed(self, event: Button.Pressed) -> None:
                self.dismiss(event.button.id == "confirm")
        
        async def handle_result(confirmed: bool) -> None:
            if confirmed:
                self.tracker.reject_all()
                self.notify("All changes rejected", severity="warning")
                self._refresh_changes_list()
        
        self.app.push_screen(RejectConfirmDialog(), handle_result)
    
    def _export_changes(self) -> None:
        """Export changes to file."""
        from pathlib import Path
        
        try:
            # Export as JSON
            json_content = self.tracker.export_changes(format='json')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_file = Path.home() / f"proxima_changes_{timestamp}.json"
            json_file.write_text(json_content, encoding='utf-8')
            
            # Export as patch
            patch_content = self.tracker.export_changes(format='patch')
            patch_file = Path.home() / f"proxima_changes_{timestamp}.patch"
            patch_file.write_text(patch_content, encoding='utf-8')
            
            self.notify(
                f"Exported to: {json_file.parent}",
                severity="information",
                timeout=5
            )
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def _view_all_diffs(self) -> None:
        """View all diffs in a full-screen view."""
        from .change_review_screen import ChangeReviewScreen
        
        self.app.push_screen(ChangeReviewScreen(
            change_tracker=self.tracker,
            backend_name=self.state.backend_name or "backend"
        ))
    
    def _refresh_changes_list(self) -> None:
        """Refresh the entire changes list."""
        self.refresh(recompose=True)
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss({"action": "cancel"})
