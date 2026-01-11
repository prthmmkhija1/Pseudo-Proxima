"""Main Proxima TUI Application.

Step 6.1: Terminal UI using Textual framework.

Screens:
1. Dashboard   - System status, recent executions
2. Execution   - Real-time progress, logs
3. Configuration - Settings management
4. Results     - Browse and analyze results
5. Backends    - Backend status and management

Design Principles:
- Keyboard-first navigation
- Responsive to terminal size
- Consistent color theme
- Contextual help (press ? for help)
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding

from .screens import (
    BackendsScreen,
    ConfigurationScreen,
    DashboardScreen,
    ExecutionScreen,
    ResultsScreen,
)

# Custom CSS theme for Proxima
PROXIMA_CSS = """
/* Proxima TUI Theme */

$primary: #3498db;
$secondary: #2ecc71;
$background: #1a1a2e;
$surface: #16213e;
$surface-lighten-1: #1f3460;
$surface-lighten-2: #2a4a7c;
$error: #e74c3c;
$warning: #f39c12;
$success: #2ecc71;
$text: #ecf0f1;
$text-muted: #95a5a6;

Screen {
    background: $background;
}

Header {
    background: $primary;
    color: $text;
}

Footer {
    background: $surface;
    color: $text;
}

Button {
    margin: 0 1;
}

Button.-primary {
    background: $primary;
}

Button.-success {
    background: $success;
}

Button.-error {
    background: $error;
}

.section-title {
    text-style: bold;
    padding: 1;
    background: $surface;
    color: $text;
}

/* Notifications */
Toast {
    background: $surface;
    border: solid $primary;
}
"""


class ProximaApp(App):
    """Proxima Terminal User Interface Application.

    A keyboard-first TUI for managing Proxima agent executions,
    backend configurations, and result analysis.

    Screens:
    - Dashboard (1): System overview and quick actions
    - Execution (2): Run and monitor executions
    - Configuration (3): Manage settings
    - Results (4): Browse execution results
    - Backends (5): Manage backend connections

    Press ? at any time for keyboard shortcuts help.
    """

    TITLE = "Proxima Agent"
    SUB_TITLE = "Terminal Interface"

    CSS = PROXIMA_CSS

    BINDINGS = [
        Binding("1", "show_dashboard", "Dashboard", show=True, priority=True),
        Binding("2", "show_execution", "Execution", show=True, priority=True),
        Binding("3", "show_config", "Config", show=True, priority=True),
        Binding("4", "show_results", "Results", show=True, priority=True),
        Binding("5", "show_backends", "Backends", show=True, priority=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "show_help", "Help", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "execution": ExecutionScreen,
        "configuration": ConfigurationScreen,
        "results": ResultsScreen,
        "backends": BackendsScreen,
    }

    def __init__(
        self,
        config_path: Path | None = None,
        theme: str = "dark",
        initial_screen: str = "dashboard",
        **kwargs,
    ) -> None:
        """Initialize the Proxima TUI.

        Args:
            config_path: Optional path to configuration file
            theme: Color theme ('dark' or 'light')
            initial_screen: Screen to show on startup
        """
        super().__init__(**kwargs)
        self._config_path = config_path
        self._theme = theme
        self._current_screen = initial_screen
        self._apply_theme(theme)

    def _apply_theme(self, theme: str) -> None:
        """Apply color theme to the app."""
        if theme == "light":
            # Light theme overrides
            self.dark = False
        else:
            # Dark theme (default)
            self.dark = True

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Start with the configured initial screen
        self.push_screen(self._current_screen)

    def action_show_dashboard(self) -> None:
        """Switch to dashboard screen."""
        self._switch_screen("dashboard")

    def action_show_execution(self) -> None:
        """Switch to execution screen."""
        self._switch_screen("execution")

    def action_show_config(self) -> None:
        """Switch to configuration screen."""
        self._switch_screen("configuration")

    def action_show_results(self) -> None:
        """Switch to results screen."""
        self._switch_screen("results")

    def action_show_backends(self) -> None:
        """Switch to backends screen."""
        self._switch_screen("backends")

    def action_show_help(self) -> None:
        """Show help modal on current screen."""
        from .widgets import HelpModal

        current = self.screen
        if not current.query("HelpModal"):
            current.mount(HelpModal())

    def _switch_screen(self, screen_name: str) -> None:
        """Switch to a screen, replacing the current one."""
        if self._current_screen != screen_name:
            # Pop current screen and push new one
            while len(self.screen_stack) > 0:
                self.pop_screen()
            self.push_screen(screen_name)
            self._current_screen = screen_name

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float = 5.0,
    ) -> None:
        """Show a notification toast.

        Args:
            message: The notification message
            title: Optional title
            severity: One of 'information', 'warning', 'error'
            timeout: How long to show the notification
        """
        super().notify(message, title=title, severity=severity, timeout=timeout)


def run_tui(config_path: Path | None = None) -> None:
    """Run the Proxima TUI application.

    Args:
        config_path: Optional path to configuration file
    """
    app = ProximaApp(config_path=config_path)
    app.run()


def main() -> None:
    """Entry point for the TUI."""
    run_tui()


if __name__ == "__main__":
    main()
