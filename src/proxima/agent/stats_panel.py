"""Statistics Panel Widget for TUI.

Phase 9: Agent Statistics & Telemetry System

Provides real-time statistics display:
- Card-based layout with categories
- Real-time metric updates
- Toggle visibility (Ctrl+I)
- Compact and full modes
- Color-coded values
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static, Label, Button

from proxima.utils.logging import get_logger

logger = get_logger("agent.stats_panel")


# ========== Formatting Helpers ==========

def format_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    elif isinstance(value, float) and value != int(value):
        return f"{value:.1f}"
    else:
        return str(int(value))


def format_bytes(bytes_value: int) -> str:
    """Format bytes with appropriate unit."""
    if bytes_value >= 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024 * 1024):.1f}GB"
    elif bytes_value >= 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f}MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.1f}KB"
    else:
        return f"{bytes_value}B"


def format_duration(seconds: float) -> str:
    """Format duration in appropriate units."""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    elif seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    elif seconds >= 1:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds * 1000:.0f}ms"


def format_percentage(value: float) -> str:
    """Format percentage."""
    return f"{value:.1f}%"


def format_currency(value: float, symbol: str = "$") -> str:
    """Format currency."""
    return f"{symbol}{value:.2f}"


# ========== Stats Card Components ==========

class StatItem(Static):
    """A single stat item with label and value."""
    
    DEFAULT_CSS = """
    StatItem {
        width: 1fr;
        height: auto;
        padding: 0 1;
    }
    
    StatItem .stat-label {
        color: $text-muted;
    }
    
    StatItem .stat-value {
        color: $text;
    }
    
    StatItem .stat-value.highlight {
        color: $success;
    }
    
    StatItem .stat-value.warning {
        color: $warning;
    }
    
    StatItem .stat-value.error {
        color: $error;
    }
    """
    
    value = reactive("")
    
    def __init__(
        self,
        label: str,
        value: str = "0",
        value_class: str = "",
        id: Optional[str] = None,
    ):
        """Initialize stat item."""
        super().__init__(id=id)
        self._label = label
        self._value_class = value_class
        self.value = value
    
    def compose(self) -> ComposeResult:
        """Compose the stat item."""
        yield Static(f"{self._label}:", classes="stat-label")
        yield Static(self.value, classes=f"stat-value {self._value_class}", id=f"{self.id}-value" if self.id else None)
    
    def watch_value(self, new_value: str) -> None:
        """Update displayed value."""
        try:
            value_widget = self.query_one(".stat-value", Static)
            value_widget.update(new_value)
        except NoMatches:
            pass
    
    def set_value_class(self, class_name: str) -> None:
        """Set value styling class."""
        try:
            value_widget = self.query_one(".stat-value", Static)
            value_widget.remove_class("highlight", "warning", "error")
            if class_name:
                value_widget.add_class(class_name)
        except NoMatches:
            pass


class StatsCard(Container):
    """A card containing a category of stats."""
    
    DEFAULT_CSS = """
    StatsCard {
        width: 100%;
        height: auto;
        border: round $primary;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    
    StatsCard .card-title {
        text-style: bold;
        color: $primary;
        padding: 0 0 1 0;
    }
    
    StatsCard .card-content {
        width: 100%;
    }
    
    StatsCard .stat-row {
        width: 100%;
        height: 1;
    }
    """
    
    def __init__(
        self,
        title: str,
        icon: str = "",
        id: Optional[str] = None,
    ):
        """Initialize stats card."""
        super().__init__(id=id)
        self._title = title
        self._icon = icon
        self._stats: Dict[str, StatItem] = {}
    
    def compose(self) -> ComposeResult:
        """Compose the card."""
        title_text = f"{self._icon} {self._title}" if self._icon else self._title
        yield Static(title_text, classes="card-title")
        yield Container(classes="card-content", id=f"{self.id}-content" if self.id else None)
    
    def add_stat_row(self, items: List[tuple]) -> None:
        """Add a row of stat items.
        
        Args:
            items: List of (label, value, id) tuples
        """
        try:
            content = self.query_one(".card-content", Container)
            row = Horizontal(classes="stat-row")
            
            for item in items:
                label, value, item_id = item if len(item) >= 3 else (*item, None)
                stat_item = StatItem(label=label, value=value, id=item_id)
                if item_id:
                    self._stats[item_id] = stat_item
                row.compose_add_child(stat_item)
            
            content.mount(row)
        except NoMatches:
            pass
    
    def update_stat(self, stat_id: str, value: str, value_class: str = "") -> None:
        """Update a stat value."""
        if stat_id in self._stats:
            self._stats[stat_id].value = value
            if value_class:
                self._stats[stat_id].set_value_class(value_class)


# ========== Main Stats Panel ==========

class StatsPanel(Container):
    """Statistics panel showing agent telemetry.
    
    Features:
    - Real-time metric display
    - Toggle visibility with Ctrl+I
    - Compact and full modes
    - Color-coded values for alerts
    
    Example:
        >>> panel = StatsPanel()
        >>> panel.update_metrics(telemetry.get_snapshot())
    """
    
    DEFAULT_CSS = """
    StatsPanel {
        width: 100%;
        height: auto;
        max-height: 20;
        padding: 1;
        background: $surface;
        border-top: solid $primary;
        display: block;
    }
    
    StatsPanel.hidden {
        display: none;
    }
    
    StatsPanel.compact {
        max-height: 4;
    }
    
    StatsPanel .panel-header {
        width: 100%;
        height: 1;
        padding: 0 1;
    }
    
    StatsPanel .panel-title {
        text-style: bold;
        color: $primary;
    }
    
    StatsPanel .panel-controls {
        dock: right;
    }
    
    StatsPanel .stats-grid {
        width: 100%;
        height: auto;
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
    }
    
    StatsPanel .compact-stats {
        width: 100%;
        height: 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+i", "toggle_stats", "Toggle Stats", show=True),
    ]
    
    # Reactive properties
    visible = reactive(True)
    compact = reactive(False)
    
    class StatsToggled(Message):
        """Message sent when stats panel is toggled."""
        def __init__(self, visible: bool):
            self.visible = visible
            super().__init__()
    
    def __init__(
        self,
        telemetry: Optional[Any] = None,
        compact_mode: bool = False,
        id: str = "stats-panel",
    ):
        """Initialize stats panel.
        
        Args:
            telemetry: Optional telemetry instance
            compact_mode: Start in compact mode
            id: Widget ID
        """
        super().__init__(id=id)
        self._telemetry = telemetry
        self.compact = compact_mode
        
        # Card references
        self._llm_card: Optional[StatsCard] = None
        self._perf_card: Optional[StatsCard] = None
        self._terminal_card: Optional[StatsCard] = None
    
    def compose(self) -> ComposeResult:
        """Compose the stats panel."""
        with Horizontal(classes="panel-header"):
            yield Static("📊 Agent Statistics", classes="panel-title")
            with Horizontal(classes="panel-controls"):
                yield Button("▼", id="toggle-mode", variant="default")
                yield Button("✕", id="close-stats", variant="default")
        
        # Full stats grid
        with Container(classes="stats-grid", id="full-stats"):
            # LLM Card
            self._llm_card = StatsCard(title="LLM", icon="🤖", id="llm-card")
            yield self._llm_card
            
            # Performance Card
            self._perf_card = StatsCard(title="Performance", icon="⚙️", id="perf-card")
            yield self._perf_card
            
            # Terminal Card
            self._terminal_card = StatsCard(title="Terminal", icon="🖥️", id="terminal-card")
            yield self._terminal_card
        
        # Compact stats (single row)
        yield Horizontal(classes="compact-stats hidden", id="compact-stats")
    
    def on_mount(self) -> None:
        """Handle mount event."""
        # Initialize cards with stat rows
        if self._llm_card:
            self._llm_card.add_stat_row([
                ("Provider", "N/A", "llm-provider"),
                ("Model", "N/A", "llm-model"),
            ])
            self._llm_card.add_stat_row([
                ("Tokens", "0", "llm-tokens"),
                ("Requests", "0", "llm-requests"),
            ])
            self._llm_card.add_stat_row([
                ("Avg Response", "0ms", "llm-response"),
                ("Cost", "$0.00", "llm-cost"),
            ])
        
        if self._perf_card:
            self._perf_card.add_stat_row([
                ("Uptime", "0s", "perf-uptime"),
                ("Messages", "0", "perf-messages"),
            ])
            self._perf_card.add_stat_row([
                ("Tools Run", "0", "perf-tools"),
                ("Success", "100%", "perf-success"),
            ])
            self._perf_card.add_stat_row([
                ("Errors", "0", "perf-errors"),
                ("Avg Time", "0ms", "perf-avg-time"),
            ])
        
        if self._terminal_card:
            self._terminal_card.add_stat_row([
                ("Active", "0", "term-active"),
                ("Completed", "0", "term-completed"),
            ])
            self._terminal_card.add_stat_row([
                ("Output Lines", "0", "term-output"),
                ("Processes", "0", "term-processes"),
            ])
    
    def watch_visible(self, visible: bool) -> None:
        """Handle visibility change."""
        if visible:
            self.remove_class("hidden")
        else:
            self.add_class("hidden")
        self.post_message(self.StatsToggled(visible))
    
    def watch_compact(self, compact: bool) -> None:
        """Handle compact mode change."""
        try:
            full_stats = self.query_one("#full-stats", Container)
            compact_stats = self.query_one("#compact-stats", Horizontal)
            toggle_btn = self.query_one("#toggle-mode", Button)
            
            if compact:
                self.add_class("compact")
                full_stats.add_class("hidden")
                compact_stats.remove_class("hidden")
                toggle_btn.label = "▲"
            else:
                self.remove_class("compact")
                full_stats.remove_class("hidden")
                compact_stats.add_class("hidden")
                toggle_btn.label = "▼"
        except NoMatches:
            pass
    
    def action_toggle_stats(self) -> None:
        """Toggle stats visibility."""
        self.visible = not self.visible
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "toggle-mode":
            self.compact = not self.compact
        elif event.button.id == "close-stats":
            self.visible = False
    
    def update_metrics(self, snapshot: Any) -> None:
        """Update displayed metrics from snapshot.
        
        Args:
            snapshot: TelemetrySnapshot instance
        """
        if not snapshot:
            return
        
        try:
            # LLM metrics
            if self._llm_card and hasattr(snapshot, 'llm'):
                llm = snapshot.llm
                self._llm_card.update_stat("llm-provider", llm.provider or "N/A")
                self._llm_card.update_stat("llm-model", llm.model or "N/A")
                self._llm_card.update_stat("llm-tokens", format_number(llm.total_tokens))
                self._llm_card.update_stat("llm-requests", format_number(llm.requests_sent))
                self._llm_card.update_stat("llm-response", f"{llm.avg_response_time_ms:.0f}ms")
                self._llm_card.update_stat("llm-cost", format_currency(llm.cost_estimate))
            
            # Performance metrics
            if self._perf_card and hasattr(snapshot, 'performance'):
                perf = snapshot.performance
                self._perf_card.update_stat("perf-uptime", perf.uptime_formatted)
                self._perf_card.update_stat("perf-messages", format_number(perf.messages_processed))
                self._perf_card.update_stat("perf-tools", format_number(perf.tools_executed))
                
                # Color-code success rate
                success_class = ""
                if perf.success_rate < 90:
                    success_class = "error"
                elif perf.success_rate < 95:
                    success_class = "warning"
                self._perf_card.update_stat("perf-success", format_percentage(perf.success_rate), success_class)
                
                # Color-code errors
                error_class = "error" if perf.errors_encountered > 0 else ""
                self._perf_card.update_stat("perf-errors", format_number(perf.errors_encountered), error_class)
                self._perf_card.update_stat("perf-avg-time", f"{perf.avg_tool_time_ms:.0f}ms")
            
            # Terminal metrics
            if self._terminal_card and hasattr(snapshot, 'terminal'):
                term = snapshot.terminal
                self._terminal_card.update_stat("term-active", format_number(term.active_terminals))
                self._terminal_card.update_stat("term-completed", format_number(term.completed_processes))
                self._terminal_card.update_stat("term-output", format_number(term.total_output_lines))
                self._terminal_card.update_stat("term-processes", format_number(term.total_processes))
        
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def set_telemetry(self, telemetry: Any) -> None:
        """Set telemetry instance."""
        self._telemetry = telemetry
    
    def refresh_from_telemetry(self) -> None:
        """Refresh metrics from telemetry."""
        if self._telemetry:
            snapshot = self._telemetry.get_snapshot()
            self.update_metrics(snapshot)


class CompactStatsBar(Static):
    """Compact single-line stats bar.
    
    Shows key metrics in a single line for minimal space usage.
    """
    
    DEFAULT_CSS = """
    CompactStatsBar {
        width: 100%;
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    
    CompactStatsBar .stat-item {
        padding: 0 2;
    }
    
    CompactStatsBar .stat-label {
        color: $text-muted;
    }
    
    CompactStatsBar .stat-value {
        color: $text;
    }
    """
    
    def __init__(self, id: str = "compact-stats-bar"):
        """Initialize compact stats bar."""
        super().__init__(id=id)
        self._content = ""
    
    def update_metrics(self, snapshot: Any) -> None:
        """Update with snapshot data."""
        if not snapshot:
            return
        
        try:
            parts = []
            
            # LLM
            if hasattr(snapshot, 'llm'):
                parts.append(f"🤖 {format_number(snapshot.llm.total_tokens)} tokens")
            
            # Performance
            if hasattr(snapshot, 'performance'):
                parts.append(f"⚙️ {snapshot.performance.uptime_formatted}")
                parts.append(f"✓ {format_percentage(snapshot.performance.success_rate)}")
            
            # Terminal
            if hasattr(snapshot, 'terminal'):
                parts.append(f"🖥️ {snapshot.terminal.active_terminals} active")
            
            self._content = "  │  ".join(parts)
            self.update(self._content)
        
        except Exception as e:
            logger.error(f"Failed to update compact stats: {e}")


# ========== Export Functions ==========

def create_stats_panel(
    telemetry: Optional[Any] = None,
    compact: bool = False,
) -> StatsPanel:
    """Create a new stats panel.
    
    Args:
        telemetry: Optional telemetry instance
        compact: Start in compact mode
        
    Returns:
        StatsPanel instance
    """
    return StatsPanel(telemetry=telemetry, compact_mode=compact)


def create_compact_stats_bar() -> CompactStatsBar:
    """Create a compact stats bar."""
    return CompactStatsBar()
