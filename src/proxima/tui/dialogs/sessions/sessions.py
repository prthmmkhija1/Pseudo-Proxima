"""Session switcher dialog for Proxima TUI.

Dialog for managing and switching sessions.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, ListView, Button
from textual import on
from rich.text import Text

from ...styles.theme import get_theme
from ...styles.icons import ICON_SESSION
from .session_item import SessionItem, SessionInfo


class SessionsDialog(ModalScreen):
    """Dialog for session management.
    
    Features:
    - List all sessions
    - Create new session
    - Switch between sessions
    - Delete sessions
    """
    
    DEFAULT_CSS = """
    SessionsDialog {
        align: center middle;
    }
    
    SessionsDialog > .dialog-container {
        width: 70;
        height: 30;
        border: thick $primary;
        background: $surface;
    }
    
    SessionsDialog .dialog-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        padding: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    SessionsDialog .search-input {
        margin: 1;
        border: solid $primary-darken-2;
    }
    
    SessionsDialog .sessions-list {
        height: 1fr;
        margin: 0 1;
    }
    
    SessionsDialog .actions-section {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
        layout: horizontal;
    }
    
    SessionsDialog .action-btn {
        margin-right: 1;
    }
    
    SessionsDialog .footer {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
        text-align: center;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "select", "Select"),
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("n", "new_session", "New"),
        ("d", "delete_session", "Delete"),
    ]
    
    def __init__(
        self,
        sessions: Optional[List[SessionInfo]] = None,
        current_session_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the sessions dialog.
        
        Args:
            sessions: Available sessions
            current_session_id: ID of current session
        """
        super().__init__(**kwargs)
        self.sessions = sessions or []
        self.filtered_sessions = self.sessions.copy()
        self.current_session_id = current_session_id
        self.selected_index = 0
    
    def compose(self):
        """Compose the dialog layout."""
        with Vertical(classes="dialog-container"):
            yield Static(f"{ICON_SESSION} Sessions", classes="dialog-title")
            yield Input(
                placeholder="Search sessions...",
                classes="search-input",
            )
            yield SessionsListView(
                self.filtered_sessions,
                self.current_session_id,
                classes="sessions-list",
            )
            
            with Horizontal(classes="actions-section"):
                yield Button("[N] New Session", id="btn-new", classes="action-btn", variant="primary")
                yield Button("[D] Delete", id="btn-delete", classes="action-btn", variant="error")
                yield Button("Export", id="btn-export", classes="action-btn")
            
            yield SessionsFooter(classes="footer")
    
    def on_mount(self) -> None:
        """Focus search on mount."""
        self.query_one(Input).focus()
    
    @on(Input.Changed)
    def filter_sessions(self, event: Input.Changed) -> None:
        """Filter sessions by search query."""
        query = event.value.lower()
        
        if query:
            self.filtered_sessions = [
                s for s in self.sessions
                if query in s.title.lower() or
                   query in s.id.lower()
            ]
        else:
            self.filtered_sessions = self.sessions.copy()
        
        # Update list
        sessions_list = self.query_one(SessionsListView)
        sessions_list.update_sessions(self.filtered_sessions)
        self.selected_index = 0
    
    def action_cancel(self) -> None:
        """Cancel and close."""
        self.dismiss(None)
    
    def action_select(self) -> None:
        """Select current session."""
        if self.filtered_sessions and self.selected_index < len(self.filtered_sessions):
            session = self.filtered_sessions[self.selected_index]
            self.dismiss({"action": "switch", "session": session})
    
    def action_move_up(self) -> None:
        """Move selection up."""
        if self.selected_index > 0:
            self.selected_index -= 1
            self.query_one(SessionsListView).index = self.selected_index
    
    def action_move_down(self) -> None:
        """Move selection down."""
        if self.selected_index < len(self.filtered_sessions) - 1:
            self.selected_index += 1
            self.query_one(SessionsListView).index = self.selected_index
    
    def action_new_session(self) -> None:
        """Create new session."""
        self.dismiss({"action": "new"})
    
    def action_delete_session(self) -> None:
        """Delete selected session."""
        if self.filtered_sessions and self.selected_index < len(self.filtered_sessions):
            session = self.filtered_sessions[self.selected_index]
            if session.id != self.current_session_id:
                self.dismiss({"action": "delete", "session": session})
            else:
                self.notify("Cannot delete active session", severity="warning")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-new":
            self.action_new_session()
        elif event.button.id == "btn-delete":
            self.action_delete_session()
        elif event.button.id == "btn-export":
            if self.filtered_sessions and self.selected_index < len(self.filtered_sessions):
                session = self.filtered_sessions[self.selected_index]
                self.dismiss({"action": "export", "session": session})


class SessionsListView(ListView):
    """List view for sessions."""
    
    def __init__(
        self,
        sessions: List[SessionInfo],
        current_id: Optional[str],
        **kwargs,
    ):
        """Initialize the list."""
        super().__init__(**kwargs)
        self._sessions = sessions
        self._current_id = current_id
    
    def on_mount(self) -> None:
        """Populate the list."""
        self.update_sessions(self._sessions)
    
    def update_sessions(self, sessions: List[SessionInfo]) -> None:
        """Update the displayed sessions."""
        self.clear()
        for session in sessions:
            is_current = session.id == self._current_id
            self.append(SessionItem(session, is_current))
        self._sessions = sessions


class SessionsFooter(Static):
    """Footer with keybindings."""
    
    def render(self) -> Text:
        """Render the footer."""
        theme = get_theme()
        text = Text()
        
        bindings = [
            ("↑↓", "navigate"),
            ("enter", "switch"),
            ("n", "new"),
            ("d", "delete"),
            ("esc", "cancel"),
        ]
        
        for i, (key, desc) in enumerate(bindings):
            if i > 0:
                text.append(" │ ", style=theme.border)
            text.append(key, style=f"bold {theme.accent}")
            text.append(f" {desc}", style=theme.fg_muted)
        
        return text
