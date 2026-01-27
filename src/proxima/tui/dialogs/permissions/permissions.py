"""Permissions dialog for Proxima TUI.

Consent dialog for LLM and agent actions.
"""

from dataclasses import dataclass
from typing import Optional

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Button
from rich.text import Text
from rich.panel import Panel

from ...styles.theme import get_theme
from ...styles.icons import ICONS


@dataclass
class PermissionRequest:
    """A permission request from the LLM/agent."""
    
    action_type: str
    description: str
    details: Optional[str] = None
    risk_level: str = "low"  # low, medium, high
    
    def get_icon(self) -> str:
        """Get the appropriate icon for this action."""
        icons = {
            "file_read": ICONS.get("file", "ðŸ“„"),
            "file_write": ICONS.get("edit", "âœï¸"),
            "file_delete": "ðŸ—‘ï¸",
            "command": ICONS.get("terminal", "ðŸ’»"),
            "network": "ðŸŒ",
            "api_call": ICONS.get("cloud", "â˜ï¸"),
        }
        return icons.get(self.action_type, "â“")


class PermissionsDialog(ModalScreen):
    """Dialog for requesting user consent.
    
    Shows:
    - Action description
    - Risk indicator
    - Allow/Deny buttons
    """
    
    DEFAULT_CSS = """
    PermissionsDialog {
        align: center middle;
    }
    
    PermissionsDialog > .dialog-container {
        width: 60;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    
    PermissionsDialog .dialog-title {
        text-style: bold;
        color: $warning;
        text-align: center;
        margin-bottom: 1;
    }
    
    PermissionsDialog .action-icon {
        text-align: center;
        margin-bottom: 1;
    }
    
    PermissionsDialog .action-description {
        margin-bottom: 1;
        padding: 1;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
    }
    
    PermissionsDialog .action-details {
        margin-bottom: 1;
        padding: 1;
        background: $surface-darken-2;
        color: $text-muted;
    }
    
    PermissionsDialog .risk-indicator {
        margin-bottom: 1;
        text-align: center;
    }
    
    PermissionsDialog .buttons {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    
    PermissionsDialog .btn {
        margin-right: 1;
        min-width: 16;
    }
    
    PermissionsDialog .btn-allow {
        background: $success-darken-2;
    }
    
    PermissionsDialog .btn-allow-session {
        background: $primary-darken-2;
    }
    
    PermissionsDialog .btn-allow-task {
        background: $warning-darken-2;
    }

    PermissionsDialog .btn-deny {
        background: $error-darken-2;
    }
    """
    
    BINDINGS = [
        ("escape", "deny", "Deny"),
        ("a", "allow", "Allow Once"),
        ("s", "allow_session", "Allow for Session"),
        ("t", "allow_task", "Allow for Task"),
        ("d", "deny", "Deny"),
    ]
    
    def __init__(
        self,
        request: PermissionRequest,
        **kwargs,
    ):
        """Initialize the permissions dialog.
        
        Args:
            request: The permission request to display
        """
        super().__init__(**kwargs)
        self.request = request
    
    def compose(self):
        """Compose the dialog layout."""
        theme = get_theme()
        
        with Vertical(classes="dialog-container"):
            yield Static("Permission Required", classes="dialog-title")
            
            # Action icon
            yield Static(
                self.request.get_icon() + " " + self.request.action_type.replace("_", " ").title(),
                classes="action-icon",
            )
            
            # Description
            yield Static(
                self.request.description,
                classes="action-description",
            )
            
            # Details (if provided)
            if self.request.details:
                yield Static(
                    self.request.details,
                    classes="action-details",
                )
            
            # Risk indicator
            yield RiskIndicator(self.request.risk_level, classes="risk-indicator")
            
            # Buttons
            with Horizontal(classes="buttons"):
                yield Button(
                    "[A] Allow",
                    id="btn-allow",
                    classes="btn btn-allow",
                    variant="success",
                )
                yield Button(
                    "[S] Allow Session",
                    id="btn-allow-session",
                    classes="btn btn-allow-session",
                    variant="primary",
                )
                yield Button(
                    "[T] Allow Task",
                    id="btn-allow-task",
                    classes="btn btn-allow-task",
                    variant="warning",
                )
                yield Button(
                    "[D] Deny",
                    id="btn-deny",
                    classes="btn btn-deny",
                    variant="error",
                )
    
    def action_allow(self) -> None:
        """Allow this action once."""
        self.dismiss("allow")
    
    def action_allow_session(self) -> None:
        """Allow this action for the session."""
        self.dismiss("allow_session")

    def action_allow_task(self) -> None:
        """Allow this action for the current task only."""
        self.dismiss("allow_task")
        
    def action_deny(self) -> None:
        """Deny this action."""
        self.dismiss("deny")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-allow":
            self.action_allow()
        elif button_id == "btn-allow-session":
            self.action_allow_session()
        elif button_id == "btn-allow-task":
            self.action_allow_task()
        elif button_id == "btn-deny":
            self.action_deny()


class RiskIndicator(Static):
    """Risk level indicator."""
    
    def __init__(self, risk_level: str, **kwargs):
        """Initialize the risk indicator."""
        super().__init__(**kwargs)
        self.risk_level = risk_level
    
    def render(self) -> Text:
        """Render the risk indicator."""
        theme = get_theme()
        text = Text()
        
        text.append("Risk Level: ", style=theme.fg_muted)
        
        if self.risk_level == "low":
            text.append("â— LOW", style=f"bold {theme.success}")
        elif self.risk_level == "medium":
            text.append("â— MEDIUM", style=f"bold {theme.warning}")
        else:
            text.append("â— HIGH", style=f"bold {theme.error}")
        
        return text