"""Command palette dialog for Proxima TUI.

Fuzzy search command palette inspired by VS Code.
"""

from dataclasses import dataclass
from typing import Callable, Optional, List

from textual.screen import ModalScreen
from textual.containers import Vertical
from textual.widgets import Static, Input, ListView
from textual import on
from rich.text import Text

from ...styles.theme import get_theme
from ...styles.icons import ICONS
from .command_item import CommandItem


@dataclass
class Command:
    """A command in the palette."""
    
    name: str
    description: str
    keybinding: Optional[str] = None
    category: str = "General"
    action: Optional[Callable] = None
    action_name: Optional[str] = None  # For app-level actions
    
    def matches(self, query: str) -> bool:
        """Check if the command matches a search query."""
        query_lower = query.lower()
        return (
            query_lower in self.name.lower() or
            query_lower in self.description.lower() or
            query_lower in self.category.lower()
        )


def get_default_commands() -> List[Command]:
    """Get the default commands with action names for dynamic binding.
    
    Returns:
        List of default commands
    """
    return [
        # Execution
        Command("Run Simulation", "Start a new simulation run", "Ctrl+R", "Execution", action_name="show_simulation_dialog"),
        Command("Pause Execution", "Pause current execution", "P", "Execution", action_name="pause_execution"),
        Command("Resume Execution", "Resume paused execution", "R", "Execution", action_name="resume_execution"),
        Command("Abort Execution", "Abort current execution", "A", "Execution", action_name="abort_execution"),
        Command("Rollback", "Rollback to last checkpoint", "Z", "Execution", action_name="rollback"),
        
        # Session
        Command("New Session", "Create a new session", "Ctrl+N", "Session", action_name="new_session"),
        Command("Switch Session", "Switch to another session", None, "Session", action_name="switch_session"),
        Command("Export Session", "Export current session", None, "Session", action_name="export_session"),
        Command("View History", "View execution history", None, "Session", action_name="view_history"),
        
        # Backend
        Command("Switch Backend", "Change simulation backend", None, "Backend", action_name="switch_backend"),
        Command("Health Check", "Run backend health checks", None, "Backend", action_name="run_health_check"),
        Command("Compare Backends", "Compare backend performance", None, "Backend", action_name="compare_backends"),
        
        # LRET Variants
        Command("Install LRET Variants", "Install LRET backend variants", None, "LRET", action_name="show_lret_installer"),
        Command("Configure LRET", "Configure LRET variant settings", None, "LRET", action_name="show_lret_config"),
        Command("LRET Benchmarks", "Run LRET vs Cirq benchmark comparison", None, "LRET", action_name="show_lret_benchmark"),
        Command("Variational Algorithms", "Run VQE, QAOA, QNN with PennyLane", None, "LRET", action_name="show_pennylane_algorithms"),
        Command("Phase 7 Unified", "Configure multi-framework unified execution", None, "LRET", action_name="show_phase7_unified"),
        Command("Variant Analysis", "Analyze and compare LRET backend variants", None, "LRET", action_name="show_variant_analysis"),
        Command("Benchmark Comparison", "Compare LRET vs Cirq with visualization", None, "LRET", action_name="show_benchmark_comparison"),
        Command("Algorithm Wizard", "6-step wizard for VQE/QAOA/QNN", None, "LRET", action_name="show_algorithm_wizard"),
        
        # LLM
        Command("Configure LLM", "Configure language model", None, "LLM", action_name="configure_llm"),
        Command("Toggle Thinking", "Toggle LLM thinking mode", None, "LLM", action_name="toggle_thinking"),
        Command("Switch Provider", "Change LLM provider", None, "LLM", action_name="switch_provider"),
        Command("Show AI Thinking", "View AI reasoning panel", "Ctrl+T", "LLM", action_name="show_ai_thinking"),
        
        # Navigation
        Command("Go to Dashboard", "Navigate to dashboard", "1", "Navigation", action_name="goto_dashboard"),
        Command("Go to Execution", "Navigate to execution monitor", "2", "Navigation", action_name="goto_execution"),
        Command("Go to Results", "Navigate to results browser", "3", "Navigation", action_name="goto_results"),
        Command("Go to Backends", "Navigate to backend management", "4", "Navigation", action_name="goto_backends"),
        Command("Go to Settings", "Navigate to settings", "5", "Navigation", action_name="goto_settings"),
        Command("Show Help", "Show help documentation", "?", "Navigation", action_name="show_help"),
        
        # System
        Command("Quit", "Exit Proxima", "Ctrl+Q", "System", action_name="quit"),
    ]


# Default commands - for backward compatibility
DEFAULT_COMMANDS = get_default_commands()


class CommandPalette(ModalScreen):
    """Command palette with fuzzy search.
    
    Provides quick access to all commands via search.
    """
    
    DIALOG_TITLE = "Command Palette"
    
    DEFAULT_CSS = """
    CommandPalette {
        align: center top;
        padding-top: 5;
    }
    
    CommandPalette > .palette-container {
        width: 70;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
    }
    
    CommandPalette .search-input {
        margin: 1;
        border: solid $primary-darken-2;
    }
    
    CommandPalette .search-input:focus {
        border: solid $primary;
    }
    
    CommandPalette .commands-list {
        height: auto;
        max-height: 30;
        padding: 0 1;
    }
    
    CommandPalette .command-item {
        padding: 0 1;
        height: 3;
    }
    
    CommandPalette .command-item:hover {
        background: $primary-darken-3;
    }
    
    CommandPalette .command-item.-highlighted {
        background: $primary-darken-2;
    }
    
    CommandPalette .category-tabs {
        layout: horizontal;
        height: auto;
        margin: 0 1 1 1;
        padding: 0 1;
        border-bottom: solid $primary-darken-3;
    }
    
    CommandPalette .category-tab {
        padding: 0 1;
        margin-right: 2;
        color: $text-muted;
    }
    
    CommandPalette .category-tab.-active {
        color: $primary;
        text-style: bold;
        border-bottom: solid $primary;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("enter", "execute", "Execute"),
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("tab", "next_category", "Next Category"),
    ]
    
    def __init__(
        self,
        commands: Optional[List[Command]] = None,
        **kwargs,
    ):
        """Initialize the command palette.
        
        Args:
            commands: List of commands (uses defaults if None)
        """
        super().__init__(**kwargs)
        self.commands = commands or DEFAULT_COMMANDS
        self.filtered_commands = self.commands.copy()
        self.selected_index = 0
        self.current_category = "All"
        self.categories = ["All"] + sorted(set(c.category for c in self.commands))
    
    def compose(self):
        """Compose the palette layout."""
        with Vertical(classes="palette-container"):
            yield Input(
                placeholder="Search commands...",
                classes="search-input",
            )
            yield CategoryTabs(self.categories, classes="category-tabs")
            yield CommandsListView(
                self.filtered_commands,
                classes="commands-list",
            )
    
    def on_mount(self) -> None:
        """Focus the search input on mount."""
        self.query_one(Input).focus()
    
    @on(Input.Changed)
    def filter_commands(self, event: Input.Changed) -> None:
        """Filter commands based on search query."""
        query = event.value
        
        if query:
            self.filtered_commands = [
                c for c in self.commands
                if c.matches(query) and (
                    self.current_category == "All" or
                    c.category == self.current_category
                )
            ]
        else:
            self.filtered_commands = [
                c for c in self.commands
                if self.current_category == "All" or
                c.category == self.current_category
            ]
        
        # Update the list
        commands_list = self.query_one(CommandsListView)
        commands_list.update_commands(self.filtered_commands)
        
        self.selected_index = 0
    
    def action_close(self) -> None:
        """Close the palette."""
        self.dismiss(None)
    
    def action_execute(self) -> None:
        """Execute the selected command."""
        if self.filtered_commands and self.selected_index < len(self.filtered_commands):
            command = self.filtered_commands[self.selected_index]
            self.dismiss(command)
    
    def action_move_up(self) -> None:
        """Move selection up."""
        if self.selected_index > 0:
            self.selected_index -= 1
            self._update_selection()
    
    def action_move_down(self) -> None:
        """Move selection down."""
        if self.selected_index < len(self.filtered_commands) - 1:
            self.selected_index += 1
            self._update_selection()
    
    def action_next_category(self) -> None:
        """Switch to next category."""
        current_idx = self.categories.index(self.current_category)
        next_idx = (current_idx + 1) % len(self.categories)
        self.current_category = self.categories[next_idx]
        
        # Update tabs
        tabs = self.query_one(CategoryTabs)
        tabs.set_active(self.current_category)
        
        # Re-filter commands
        self.filter_commands(Input.Changed(self.query_one(Input), self.query_one(Input).value))
    
    def _update_selection(self) -> None:
        """Update the visual selection in the list."""
        commands_list = self.query_one(CommandsListView)
        commands_list.index = self.selected_index


class CategoryTabs(Static):
    """Category tabs for filtering commands."""
    
    def __init__(self, categories: List[str], **kwargs):
        """Initialize the tabs."""
        super().__init__(**kwargs)
        self.categories = categories
        self.active = "All"
    
    def render(self) -> Text:
        """Render the category tabs."""
        theme = get_theme()
        text = Text()
        
        for i, category in enumerate(self.categories):
            if i > 0:
                text.append("  │  ", style=theme.border)
            
            if category == self.active:
                text.append(category, style=f"bold {theme.primary}")
            else:
                text.append(category, style=theme.fg_muted)
        
        return text
    
    def set_active(self, category: str) -> None:
        """Set the active category."""
        self.active = category
        self.refresh()


class CommandsListView(ListView):
    """List view for displaying commands."""
    
    def __init__(self, commands: List[Command], **kwargs):
        """Initialize the list view."""
        super().__init__(**kwargs)
        self._commands = commands
    
    def on_mount(self) -> None:
        """Populate the list."""
        self.update_commands(self._commands)
    
    def update_commands(self, commands: List[Command]) -> None:
        """Update the displayed commands."""
        self.clear()
        for command in commands:
            self.append(CommandItem(command))
        self._commands = commands
