"""Main ProximaTUI Application.

The main Textual application class for the Proxima TUI.
"""

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen

from .state import TUIState
from .styles.theme import get_theme

# CSS file path
CSS_PATH = Path(__file__).parent / "styles" / "base.tcss"


class ProximaTUI(App):
    """Proxima Terminal User Interface.
    
    A professional, Crush-inspired TUI for quantum simulation orchestration.
    """
    
    TITLE = "Proxima"
    SUB_TITLE = "Quantum Simulation Orchestration"
    
    CSS_PATH = CSS_PATH
    
    BINDINGS = [
        Binding("1", "goto_dashboard", "Dashboard", show=True),
        Binding("2", "goto_execution", "Execution", show=True),
        Binding("3", "goto_results", "Results", show=True),
        Binding("4", "goto_backends", "Backends", show=True),
        Binding("5", "goto_settings", "Settings", show=True),
        Binding("ctrl+p", "open_commands", "Commands", show=True),
        Binding("question_mark", "show_help", "Help", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]
    
    def __init__(
        self,
        theme: str = "dark",
        initial_screen: str = "dashboard",
    ):
        """Initialize the ProximaTUI application.
        
        Args:
            theme: Theme name ('dark' or 'light')
            initial_screen: Initial screen to show
        """
        super().__init__()
        self.theme_name = theme
        self.initial_screen_name = initial_screen
        self.state = TUIState()
        self._theme = get_theme()
    
    def on_mount(self) -> None:
        """Handle application mount."""
        from .screens import DashboardScreen
        
        self.title = "Proxima"
        self.sub_title = "Quantum Simulation Orchestration"
        
        # Auto-load last session settings
        self._auto_load_session()
        
        # Apply saved theme
        self._apply_saved_theme()
        
        # Install and push the initial screen
        self.push_screen(DashboardScreen(state=self.state))
    
    def _auto_load_session(self) -> None:
        """Auto-load the last active session on startup."""
        import json
        
        try:
            config_dir = Path.home() / ".proxima"
            session_file = config_dir / "last_session.json"
            
            if session_file.exists():
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Restore session state
                if 'session_id' in session_data:
                    self.state.active_session_id = session_data['session_id']
                if 'session_title' in session_data:
                    self.state.session_title = session_data['session_title']
                if 'working_directory' in session_data:
                    self.state.working_directory = session_data['working_directory']
                if 'current_backend' in session_data:
                    self.state.current_backend = session_data['current_backend']
                if 'shots' in session_data:
                    self.state.shots = session_data['shots']
                
                # Notify user
                self.notify(f"Session restored: {self.state.session_title or 'Last session'}", severity="information")
        except Exception:
            # Silently fail - no previous session or corrupted
            pass
    
    def _apply_saved_theme(self) -> None:
        """Apply the saved theme from settings."""
        import json
        
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                
                display = settings.get('display', {})
                saved_theme = display.get('theme', 'dark')
                
                # Apply theme
                self.theme_name = saved_theme
                self.dark = (saved_theme == 'dark')
        except Exception:
            # Use default theme
            pass
    
    def save_session_state(self) -> None:
        """Save current session state for auto-load on next startup."""
        import json
        
        try:
            config_dir = Path.home() / ".proxima"
            config_dir.mkdir(parents=True, exist_ok=True)
            session_file = config_dir / "last_session.json"
            
            session_data = {
                'session_id': self.state.active_session_id,
                'session_title': self.state.session_title,
                'working_directory': self.state.working_directory,
                'current_backend': self.state.current_backend,
                'shots': self.state.shots,
                'saved_at': __import__('time').time(),
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception:
            pass  # Silently fail
    
    def _navigate_to_screen(self, screen_name: str) -> None:
        """Navigate to a specific screen.
        
        Args:
            screen_name: Name of the screen to navigate to
        """
        from .screens import (
            DashboardScreen,
            ExecutionScreen,
            ResultsScreen,
            BackendsScreen,
            SettingsScreen,
            HelpScreen,
        )
        
        screens = {
            "dashboard": DashboardScreen,
            "execution": ExecutionScreen,
            "results": ResultsScreen,
            "backends": BackendsScreen,
            "settings": SettingsScreen,
            "help": HelpScreen,
        }
        
        screen_class = screens.get(screen_name)
        if screen_class:
            self.state.current_screen = screen_name
            self.push_screen(screen_class(state=self.state))
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard screen."""
        self._navigate_to_screen("dashboard")
    
    def action_goto_execution(self) -> None:
        """Navigate to execution screen."""
        self._navigate_to_screen("execution")
    
    def action_goto_results(self) -> None:
        """Navigate to results screen."""
        self._navigate_to_screen("results")
    
    def action_goto_backends(self) -> None:
        """Navigate to backends screen."""
        self._navigate_to_screen("backends")
    
    def action_goto_settings(self) -> None:
        """Navigate to settings screen."""
        self._navigate_to_screen("settings")
    
    def action_show_help(self) -> None:
        """Show help screen."""
        self._navigate_to_screen("help")
    
    def action_open_commands(self) -> None:
        """Open command palette."""
        from .dialogs import CommandPalette
        
        def handle_command(command):
            if command:
                self._execute_command(command)
        
        self.push_screen(CommandPalette(), handle_command)
    
    def _execute_command(self, command) -> None:
        """Execute a command from the command palette.
        
        Args:
            command: Command object with action_name or action
        """
        # First, try direct action
        if command.action:
            command.action()
            return
        
        # Map action names to app actions
        action_map = {
            # Navigation
            "goto_dashboard": self.action_goto_dashboard,
            "goto_execution": self.action_goto_execution,
            "goto_results": self.action_goto_results,
            "goto_backends": self.action_goto_backends,
            "goto_settings": self.action_goto_settings,
            "show_help": self.action_show_help,
            "quit": self.action_quit,
            
            # Execution
            "show_simulation_dialog": self._action_show_simulation_dialog,
            "pause_execution": self._action_pause_execution,
            "resume_execution": self._action_resume_execution,
            "abort_execution": self._action_abort_execution,
            "rollback": self._action_rollback,
            
            # Session
            "new_session": self._action_new_session,
            "switch_session": self._action_switch_session,
            "export_session": self._action_export_session,
            "view_history": self._action_view_history,
            
            # Backend
            "switch_backend": self._action_switch_backend,
            "run_health_check": self._action_run_health_check,
            "compare_backends": self._action_compare_backends,
            
            # LLM
            "configure_llm": self._action_configure_llm,
            "toggle_thinking": self._action_toggle_thinking,
            "switch_provider": self._action_switch_provider,
        }
        
        if command.action_name and command.action_name in action_map:
            action_map[command.action_name]()
        else:
            self.notify(f"Executing: {command.name}")
    
    def _action_show_simulation_dialog(self) -> None:
        """Show simulation dialog."""
        from .dialogs import SimulationDialog
        
        def handle_config(config):
            if config:
                self.notify(f"🚀 Starting: {config.description or config.circuit_type}")
                self.action_goto_execution()
        
        self.push_screen(SimulationDialog(), handle_config)
    
    def _action_pause_execution(self) -> None:
        """Pause current execution."""
        from .screens import ExecutionScreen
        if isinstance(self.screen, ExecutionScreen):
            self.screen.action_pause_execution()
        else:
            self.notify("Switch to Execution screen first", severity="warning")
    
    def _action_resume_execution(self) -> None:
        """Resume paused execution."""
        from .screens import ExecutionScreen
        if isinstance(self.screen, ExecutionScreen):
            self.screen.action_resume_execution()
        else:
            self.notify("Switch to Execution screen first", severity="warning")
    
    def _action_abort_execution(self) -> None:
        """Abort current execution."""
        from .screens import ExecutionScreen
        if isinstance(self.screen, ExecutionScreen):
            self.screen.action_abort_execution()
        else:
            self.action_goto_execution()
            self.notify("Navigate to Execution to abort", severity="information")
    
    def _action_rollback(self) -> None:
        """Rollback to checkpoint."""
        from .screens import ExecutionScreen
        if isinstance(self.screen, ExecutionScreen):
            self.screen.action_rollback()
        else:
            self.notify("Switch to Execution screen first", severity="warning")
    
    def _action_new_session(self) -> None:
        """Create new session."""
        self.notify("Creating new session...", severity="information")
        # Reset state for new session
        self.state = TUIState()
        self.action_goto_dashboard()
        self.notify("✓ New session created", severity="success")
    
    def _action_switch_session(self) -> None:
        """Switch session."""
        from .dialogs import SessionsDialog
        self.push_screen(SessionsDialog(sessions=[]))
    
    def _action_export_session(self) -> None:
        """Export current session."""
        self.notify("Exporting session...", severity="information")
    
    def _action_view_history(self) -> None:
        """View execution history."""
        self.action_goto_results()
    
    def _action_switch_backend(self) -> None:
        """Switch backend."""
        self.action_goto_backends()
    
    def _action_run_health_check(self) -> None:
        """Run health check."""
        self.action_goto_backends()
        self.notify("Run Health Check from Backends screen", severity="information")
    
    def _action_compare_backends(self) -> None:
        """Compare backends."""
        self.action_goto_backends()
    
    def _action_configure_llm(self) -> None:
        """Configure LLM."""
        self.action_goto_settings()
    
    def _action_toggle_thinking(self) -> None:
        """Toggle LLM thinking mode."""
        self.state.llm_thinking_enabled = not getattr(self.state, 'llm_thinking_enabled', False)
        status = "enabled" if self.state.llm_thinking_enabled else "disabled"
        self.notify(f"LLM thinking mode {status}")
    
    def _action_switch_provider(self) -> None:
        """Switch LLM provider."""
        self.action_goto_settings()
    
    def action_quit(self) -> None:
        """Quit the application, saving session state."""
        # Save session state before quitting
        self.save_session_state()
        self.exit()


def launch(theme: str = "dark", initial_screen: str = "dashboard") -> None:
    """Launch the Proxima TUI.
    
    Args:
        theme: Theme name ('dark' or 'light')
        initial_screen: Initial screen to show
    """
    app = ProximaTUI(theme=theme, initial_screen=initial_screen)
    app.run()


# Backward compatibility alias
ProximaApp = ProximaTUI


if __name__ == "__main__":
    launch()
