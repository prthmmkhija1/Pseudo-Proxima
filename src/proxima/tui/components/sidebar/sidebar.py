"""Main Sidebar component for Proxima TUI.

Right-aligned status panel displaying session info, backends, memory, etc.
Real-time updates every 500ms for dynamic stats.
"""

from typing import Optional

from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical, Container
from textual.reactive import reactive
from textual.timer import Timer
from rich.text import Text

from ...state import TUIState
from ...styles.theme import get_theme
from ..logo import Logo
from .backends_section import BackendsSection
from .memory_section import MemorySection
from .session_section import SessionSection
from .results_section import ResultsSection


class Sidebar(Vertical):
    """Right-aligned sidebar with real-time status.
    
    Sections:
    1. Logo (7 lines full, 2 lines compact)
    2. Current Task (2 lines)
    3. Model Info (3 lines)
    4. Results (dynamic)
    5. Backends (dynamic)
    6. Memory (3 lines)
    7. Checkpoints (during execution)
    
    Auto-refreshes every 500ms for real-time stats.
    """
    
    DEFAULT_CSS = """
    Sidebar {
        width: 32;
        height: 100%;
        background: $surface;
        border-left: solid $primary;
        padding: 1;
        dock: right;
    }
    
    Sidebar.-compact {
        width: 28;
    }
    
    Sidebar .sidebar-logo {
        height: auto;
        margin-bottom: 1;
    }
    
    Sidebar .sidebar-section {
        margin-bottom: 1;
    }
    
    Sidebar .sidebar-divider {
        height: 1;
        margin: 0 0 1 0;
    }
    """
    
    compact_mode = reactive(False)
    _refresh_timer: Optional[Timer] = None
    
    def __init__(
        self,
        state: TUIState,
        compact: bool = False,
        **kwargs,
    ):
        """Initialize the sidebar.
        
        Args:
            state: TUI state instance
            compact: Whether to use compact mode
        """
        super().__init__(**kwargs)
        self.state = state
        self.compact_mode = compact
    
    def on_mount(self) -> None:
        """Start the real-time refresh timer when mounted."""
        # Update every 500ms for real-time stats
        self._refresh_timer = self.set_interval(0.5, self._refresh_stats)
    
    def on_unmount(self) -> None:
        """Stop the timer when unmounted."""
        if self._refresh_timer:
            self._refresh_timer.stop()
            self._refresh_timer = None
    
    def _refresh_stats(self) -> None:
        """Refresh all sidebar sections for real-time updates."""
        try:
            # Update memory stats from system
            self.state.update_memory_stats()
            
            # Refresh all section widgets
            for child in self.children:
                if hasattr(child, 'refresh'):
                    child.refresh()
        except Exception:
            pass  # Silently ignore refresh errors
    
    def compose(self):
        """Compose the sidebar layout."""
        # Logo
        yield Logo(
            compact=self.compact_mode,
            show_version=True,
            version="0.3.0",
            classes="sidebar-logo",
        )
        
        # Session/Task section
        yield SessionSection(
            self.state,
            classes="sidebar-section",
        )
        
        # Model info section
        yield ModelInfoSection(
            self.state,
            classes="sidebar-section",
        )
        
        # Results section
        yield ResultsSection(
            self.state,
            classes="sidebar-section",
        )
        
        # Backends section
        yield BackendsSection(
            self.state,
            classes="sidebar-section",
        )
        
        # Memory section
        yield MemorySection(
            self.state,
            classes="sidebar-section",
        )
        
        # Checkpoints section (only shown during execution)
        yield CheckpointsSection(
            self.state,
            classes="sidebar-section",
        )
    
    def watch_compact_mode(self, compact: bool) -> None:
        """Handle compact mode changes."""
        self.set_class(compact, "-compact")
        
        # Update logo (only if composed)
        try:
            logo = self.query_one(Logo)
            if logo:
                logo.compact = compact
                logo.refresh()
        except Exception:
            # Logo not yet composed, ignore
            pass


class ModelInfoSection(Static):
    """Model/LLM information display."""
    
    DEFAULT_CSS = """
    ModelInfoSection {
        height: auto;
        padding: 0;
    }
    """
    
    def __init__(self, state: TUIState, **kwargs):
        """Initialize the model info section."""
        super().__init__(**kwargs)
        self.state = state
    
    def render(self) -> Text:
        """Render the model info."""
        theme = get_theme()
        text = Text()
        
        # Model name
        model = self.state.llm_model or "Not connected"
        provider = self.state.llm_provider or ""
        
        text.append("◈ ", style=f"bold {theme.secondary}")
        
        if self.state.llm_connected:
            text.append(f"{provider}", style=f"bold {theme.fg_base}")
            if model:
                text.append(f" ({model})", style=theme.fg_muted)
        else:
            text.append(model, style=theme.fg_subtle)
        
        text.append("\n")
        
        # Thinking status
        if self.state.thinking_enabled:
            text.append("  Thinking ", style=theme.fg_muted)
            text.append("On", style=f"bold {theme.success}")
        else:
            text.append("  Thinking ", style=theme.fg_muted)
            text.append("Off", style=theme.fg_subtle)
        
        text.append("\n")
        
        # Token/cost info
        if self.state.prompt_tokens > 0 or self.state.completion_tokens > 0:
            text.append("  ", style="")
            text.append(self.state.get_token_summary(), style=theme.fg_muted)
        
        return text


class CheckpointsSection(Static):
    """Checkpoint information display (shown during execution)."""
    
    DEFAULT_CSS = """
    CheckpointsSection {
        height: auto;
        padding: 0;
    }
    
    CheckpointsSection.-hidden {
        display: none;
    }
    """
    
    def __init__(self, state: TUIState, **kwargs):
        """Initialize the checkpoints section."""
        super().__init__(**kwargs)
        self.state = state
    
    def on_mount(self) -> None:
        """Handle mount - check if should be visible."""
        self._update_visibility()
    
    def _update_visibility(self) -> None:
        """Update section visibility based on execution state."""
        has_execution = self.state.execution_status not in ["IDLE", "COMPLETED"]
        self.set_class(not has_execution, "-hidden")
    
    def render(self) -> Text:
        """Render the checkpoints info."""
        theme = get_theme()
        text = Text()
        
        # Header
        text.append("Checkpoints ", style=f"bold {theme.fg_subtle}")
        text.append("────────", style=theme.border)
        text.append("\n")
        
        checkpoint = self.state.latest_checkpoint
        
        if checkpoint:
            # Checkpoint ID
            text.append(f"{checkpoint.id}", style=theme.fg_base)
            
            # Time ago
            if self.state.last_checkpoint_time:
                from datetime import datetime
                delta = datetime.now() - self.state.last_checkpoint_time
                seconds = int(delta.total_seconds())
                text.append(f"  {seconds}s ago", style=theme.fg_muted)
            
            text.append("\n")
            
            # Stages info
            text.append(f"Stages: {checkpoint.stage_index}/{self.state.total_stages} complete",
                       style=theme.fg_muted)
            
            if self.state.rollback_available:
                text.append("\n")
                text.append("rollback available", style=f"italic {theme.info}")
        else:
            text.append("No checkpoints", style=theme.fg_subtle)
        
        return text
