"""Session item component for Proxima TUI.

Individual session item in the sessions dialog.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from textual.widgets import ListItem, Label
from rich.text import Text

from ...styles.theme import get_theme
from ...styles.icons import ICON_SESSION, ICON_CHECK, ICON_PAUSED, ICON_ERROR


@dataclass
class SessionInfo:
    """Information about a session."""
    
    id: str
    title: str
    status: str  # active, paused, completed, error
    created_at: datetime
    last_activity: Optional[datetime] = None
    task_count: int = 0
    backend: Optional[str] = None


class SessionItem(ListItem):
    """A single session item in the list."""
    
    DEFAULT_CSS = """
    SessionItem {
        height: 3;
        padding: 0 1;
    }
    
    SessionItem:hover {
        background: $primary-darken-3;
    }
    
    SessionItem.-current {
        background: $primary-darken-2;
    }
    """
    
    def __init__(self, session: SessionInfo, is_current: bool = False, **kwargs):
        """Initialize the session item.
        
        Args:
            session: Session information
            is_current: Whether this is the current session
        """
        super().__init__(**kwargs)
        self.session = session
        self.is_current = is_current
        if is_current:
            self.add_class("-current")
    
    def compose(self):
        """Compose the item content."""
        yield Label(self._render_content())
    
    def _render_content(self) -> Text:
        """Render the session item content."""
        theme = get_theme()
        text = Text()
        
        # Status icon
        status_icons = {
            "active": (ICON_SESSION, theme.success),
            "paused": (ICON_PAUSED, theme.warning),
            "completed": (ICON_CHECK, theme.success),
            "error": (ICON_ERROR, theme.error),
        }
        icon, color = status_icons.get(self.session.status, (ICON_SESSION, theme.fg_muted))
        text.append(icon, style=f"bold {color}")
        text.append(" ")
        
        # Title
        text.append(self.session.title, style=f"bold {theme.fg_base}")
        
        # Current indicator
        if self.is_current:
            text.append(" (active)", style=theme.accent)
        
        # ID (truncated)
        text.append(f"  [{self.session.id[:8]}]", style=theme.fg_subtle)
        
        text.append("\n  ")
        
        # Status and metadata
        text.append(self.session.status.capitalize(), style=color)
        
        if self.session.backend:
            text.append(f"  •  {self.session.backend}", style=theme.fg_muted)
        
        text.append(f"  •  {self.session.task_count} tasks", style=theme.fg_muted)
        
        # Time
        time_str = self._format_time(self.session.last_activity or self.session.created_at)
        text.append(f"  •  {time_str}", style=theme.fg_subtle)
        
        return text
    
    def _format_time(self, dt: datetime) -> str:
        """Format datetime as relative time."""
        now = datetime.now()
        delta = now - dt
        
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds >= 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"
