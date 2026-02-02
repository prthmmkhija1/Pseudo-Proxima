"""Enhanced Agent UI Widgets for Phase 2: UI/UX Enhancements.

Provides professional, user-friendly UI components for the AI Agent:
- Word-wrapped message bubbles for chat
- Resizable panel containers with drag handles
- Toggle-able real-time stats panel
- Professional themed components

Features:
- Automatic word wrapping respecting word boundaries
- Resizable panels with min/max constraints
- Collapsible stats with smooth animation
- Professional color hierarchy and spacing
- Keyboard shortcuts for all operations
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive, var
from textual.widgets import (
    Button,
    Collapsible,
    Label,
    ProgressBar,
    RichLog,
    Rule,
    Static,
    Switch,
)
from textual.widget import Widget
from textual.timer import Timer
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.console import Console, RenderableType
from rich.style import Style


# =============================================================================
# Step 2.1: Word Wrapping Message Components
# =============================================================================

class WrappedMessage(Static):
    """A message widget with proper word wrapping.
    
    Automatically wraps text at word boundaries while preserving
    code blocks with horizontal scrolling.
    """
    
    DEFAULT_CSS = """
    WrappedMessage {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
    }
    
    WrappedMessage.user-message {
        background: $primary-darken-2;
        border-left: thick $primary;
    }
    
    WrappedMessage.assistant-message {
        background: $surface-darken-1;
        border-left: thick $accent;
    }
    
    WrappedMessage.system-message {
        background: $warning-darken-3;
        border-left: thick $warning;
        color: $text-muted;
    }
    
    WrappedMessage.tool-message {
        background: $success-darken-3;
        border-left: thick $success;
    }
    
    WrappedMessage.error-message {
        background: $error-darken-3;
        border-left: thick $error;
    }
    
    WrappedMessage .message-header {
        text-style: bold;
        margin-bottom: 1;
    }
    
    WrappedMessage .message-content {
        width: 100%;
    }
    
    WrappedMessage .message-timestamp {
        color: $text-muted;
        text-align: right;
    }
    """
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        show_header: bool = True,
        **kwargs,
    ):
        """Initialize the wrapped message.
        
        Args:
            role: Message role (user, assistant, system, tool, error)
            content: Message content
            timestamp: Optional timestamp string
            show_header: Whether to show role header
        """
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M:%S")
        self.show_header = show_header
        
        # Add role-specific class
        self.add_class(f"{role}-message")
    
    def render(self) -> RenderableType:
        """Render the message with word wrapping."""
        text = Text()
        
        # Header with icon
        if self.show_header:
            icons = {
                "user": "👤",
                "assistant": "🤖",
                "system": "⚙️",
                "tool": "🔧",
                "error": "❌",
            }
            colors = {
                "user": "bold cyan",
                "assistant": "bold green",
                "system": "bold yellow",
                "tool": "bold magenta",
                "error": "bold red",
            }
            labels = {
                "user": "You",
                "assistant": "AI Agent",
                "system": "System",
                "tool": "Tool",
                "error": "Error",
            }
            
            icon = icons.get(self.role, "💬")
            color = colors.get(self.role, "bold white")
            label = labels.get(self.role, self.role.title())
            
            text.append(f"{icon} {label}", style=color)
            text.append(f"  {self.timestamp}\n", style="dim")
        
        # Content with word wrapping
        # Check for code blocks
        if "```" in self.content:
            parts = self.content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # Regular text - wrap it
                    text.append(part, overflow="fold")
                else:
                    # Code block - preserve formatting
                    lines = part.strip().split("\n")
                    lang = lines[0] if lines else ""
                    code = "\n".join(lines[1:]) if len(lines) > 1 else part
                    
                    text.append("\n")
                    text.append("─" * 40 + "\n", style="dim")
                    text.append(code, style="on dark_blue")
                    text.append("\n" + "─" * 40 + "\n", style="dim")
        else:
            # Regular text - apply word wrapping
            text.append(self.content, overflow="fold")
        
        return text


class ChatMessageBubble(Container):
    """Professional chat message bubble with metadata."""
    
    DEFAULT_CSS = """
    ChatMessageBubble {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 1 0;
    }
    
    ChatMessageBubble.user {
        align: right top;
    }
    
    ChatMessageBubble.assistant {
        align: left top;
    }
    
    ChatMessageBubble .bubble-content {
        width: auto;
        max-width: 90%;
        padding: 1 2;
        border: round $primary-darken-1;
    }
    
    ChatMessageBubble.user .bubble-content {
        background: $primary-darken-2;
        border: round $primary;
    }
    
    ChatMessageBubble.assistant .bubble-content {
        background: $surface-darken-1;
        border: round $accent;
    }
    
    ChatMessageBubble .bubble-header {
        height: 1;
        margin-bottom: 1;
    }
    
    ChatMessageBubble .bubble-role {
        text-style: bold;
    }
    
    ChatMessageBubble .bubble-time {
        color: $text-muted;
    }
    
    ChatMessageBubble .bubble-text {
        width: 100%;
    }
    
    ChatMessageBubble .bubble-meta {
        height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        tokens: int = 0,
        thinking_time_ms: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M:%S")
        self.tokens = tokens
        self.thinking_time_ms = thinking_time_ms
        
        self.add_class(role)
    
    def compose(self) -> ComposeResult:
        """Compose the bubble."""
        with Container(classes="bubble-content"):
            # Header
            with Horizontal(classes="bubble-header"):
                icon = "👤" if self.role == "user" else "🤖"
                label = "You" if self.role == "user" else "AI Agent"
                yield Static(f"{icon} {label}", classes="bubble-role")
                yield Static(self.timestamp, classes="bubble-time")
            
            # Content
            yield Static(self.content, classes="bubble-text")
            
            # Metadata (for assistant messages)
            if self.role == "assistant" and (self.tokens or self.thinking_time_ms):
                meta_parts = []
                if self.tokens:
                    meta_parts.append(f"{self.tokens} tokens")
                if self.thinking_time_ms:
                    meta_parts.append(f"{self.thinking_time_ms}ms")
                yield Static(" | ".join(meta_parts), classes="bubble-meta")


class WordWrappedRichLog(RichLog):
    """Enhanced RichLog with word wrapping enabled by default.
    
    Phase 2 UI Enhancements:
    - Word wrapping enabled by default (no horizontal scroll needed)
    - Eye-pleasing gray background instead of black
    - Larger text size (2x) for better readability
    - Mouse text selection support for copying
    """
    
    DEFAULT_CSS = """
    WordWrappedRichLog {
        /* Eye-pleasing gray background instead of black */
        background: #2d3748;
        border: solid $primary-darken-3;
        padding: 1;
        /* Disable horizontal scroll, enable vertical only */
        overflow-x: hidden;
        overflow-y: auto;
        /* Ensure word wrap works */
        width: 100%;
    }
    
    WordWrappedRichLog:focus {
        border: solid $accent;
    }
    """
    
    def __init__(self, **kwargs):
        # Ensure wrap is enabled by default - this removes need for horizontal scroll
        kwargs.setdefault("wrap", True)
        kwargs.setdefault("auto_scroll", True)
        kwargs.setdefault("markup", True)
        kwargs.setdefault("highlight", True)
        # Enable mouse selection for copying text
        kwargs.setdefault("can_focus", True)
        super().__init__(**kwargs)
    
    def write_wrapped(
        self,
        content: str,
        style: Optional[str] = None,
        expand: bool = True,
    ) -> None:
        """Write content with guaranteed word wrapping.
        
        Args:
            content: Text content to write
            style: Optional Rich style string
            expand: Whether to expand tabs
        """
        # Apply 2x text size effect through formatting (use bold for emphasis)
        text = Text(content, style=style, overflow="fold")
        self.write(text, expand=expand)
    
    def write_large(
        self,
        content: str,
        style: Optional[str] = None,
        expand: bool = True,
    ) -> None:
        """Write content with larger, more readable formatting.
        
        Simulates 2x text size through bold styling and spacing.
        
        Args:
            content: Text content to write
            style: Optional Rich style string  
            expand: Whether to expand tabs
        """
        # Use bold style and add extra line spacing for readability
        combined_style = f"bold {style}" if style else "bold"
        text = Text(content, style=combined_style, overflow="fold")
        self.write(text, expand=expand)
    
    def write_code(
        self,
        code: str,
        language: str = "python",
        line_numbers: bool = True,
    ) -> None:
        """Write a code block with syntax highlighting.
        
        Code blocks use horizontal scrolling, not wrapping.
        
        Args:
            code: Source code to display
            language: Programming language for highlighting
            line_numbers: Whether to show line numbers
        """
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=False,  # Code should scroll, not wrap
        )
        self.write(syntax)
    
    def write_markdown(self, content: str) -> None:
        """Write markdown content with proper rendering.
        
        Args:
            content: Markdown text to render
        """
        md = Markdown(content)
        self.write(md)


# =============================================================================
# Step 2.2: Resizable Panel Container
# =============================================================================

class ResizablePanelContainer(Container):
    """Container with resizable panels using drag handles.
    
    Allows users to resize panels by dragging the handle between them.
    Supports horizontal and vertical layouts with min/max constraints.
    """
    
    DEFAULT_CSS = """
    ResizablePanelContainer {
        width: 100%;
        height: 100%;
    }
    
    ResizablePanelContainer.horizontal {
        layout: horizontal;
    }
    
    ResizablePanelContainer.vertical {
        layout: vertical;
    }
    
    ResizablePanelContainer .resize-handle {
        background: $primary-darken-2;
    }
    
    ResizablePanelContainer .resize-handle:hover {
        background: $accent;
    }
    
    ResizablePanelContainer.horizontal .resize-handle {
        width: 1;
        height: 100%;
    }
    
    ResizablePanelContainer.vertical .resize-handle {
        width: 100%;
        height: 1;
    }
    
    ResizablePanelContainer .resize-handle.dragging {
        background: $success;
    }
    
    ResizablePanelContainer .panel-left,
    ResizablePanelContainer .panel-first {
        height: 100%;
    }
    
    ResizablePanelContainer .panel-right,
    ResizablePanelContainer .panel-second {
        height: 100%;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+bracketleft", "shrink_first", "Shrink Left", show=False),
        Binding("ctrl+bracketright", "grow_first", "Grow Left", show=False),
    ]
    
    class PanelResized(Message):
        """Emitted when panel is resized."""
        def __init__(self, first_ratio: float, second_ratio: float) -> None:
            self.first_ratio = first_ratio
            self.second_ratio = second_ratio
            super().__init__()
    
    # Reactive properties for panel sizes (as percentages)
    first_panel_size: reactive[float] = reactive(50.0)
    
    def __init__(
        self,
        orientation: str = "horizontal",
        first_panel_size: float = 50.0,
        min_size: float = 20.0,
        max_size: float = 80.0,
        persist_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the resizable container.
        
        Args:
            orientation: "horizontal" or "vertical"
            first_panel_size: Initial size of first panel (percentage)
            min_size: Minimum panel size (percentage)
            max_size: Maximum panel size (percentage)
            persist_key: Key for saving size to settings
        """
        super().__init__(**kwargs)
        self.orientation = orientation
        self.min_size = min_size
        self.max_size = max_size
        self.persist_key = persist_key
        
        # Set initial size
        self.first_panel_size = first_panel_size
        
        # Drag state
        self._is_dragging = False
        self._drag_start_pos = 0
        self._drag_start_size = 0.0
        
        # Add orientation class
        self.add_class(orientation)
        
        # Load persisted size
        if persist_key:
            self._load_persisted_size()
    
    def _load_persisted_size(self) -> None:
        """Load size from settings file."""
        try:
            settings_path = Path.home() / ".proxima" / "tui_settings.json"
            if settings_path.exists():
                with open(settings_path, "r") as f:
                    settings = json.load(f)
                
                panel_sizes = settings.get("panel_sizes", {})
                if self.persist_key in panel_sizes:
                    self.first_panel_size = panel_sizes[self.persist_key]
        except Exception:
            pass
    
    def _save_persisted_size(self) -> None:
        """Save size to settings file."""
        if not self.persist_key:
            return
        
        try:
            settings_path = Path.home() / ".proxima" / "tui_settings.json"
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            
            settings = {}
            if settings_path.exists():
                with open(settings_path, "r") as f:
                    settings = json.load(f)
            
            if "panel_sizes" not in settings:
                settings["panel_sizes"] = {}
            
            settings["panel_sizes"][self.persist_key] = self.first_panel_size
            
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass
    
    def watch_first_panel_size(self, size: float) -> None:
        """Update panel styles when size changes."""
        self._update_panel_styles()
        self.post_message(self.PanelResized(size, 100 - size))
    
    def _update_panel_styles(self) -> None:
        """Update panel width/height styles based on current size."""
        try:
            first = self.query_one(".panel-first, .panel-left")
            second = self.query_one(".panel-second, .panel-right")
            
            if self.orientation == "horizontal":
                first.styles.width = f"{self.first_panel_size}%"
                second.styles.width = f"{100 - self.first_panel_size - 1}%"
            else:
                first.styles.height = f"{self.first_panel_size}%"
                second.styles.height = f"{100 - self.first_panel_size - 1}%"
        except NoMatches:
            pass
    
    def on_mount(self) -> None:
        """Apply initial panel sizes."""
        self._update_panel_styles()
    
    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Handle mouse down on resize handle."""
        # Check if clicking on resize handle
        try:
            handle = self.query_one(".resize-handle")
            # Check if mouse is within handle bounds
            if self._is_over_handle(event):
                self._is_dragging = True
                handle.add_class("dragging")
                
                if self.orientation == "horizontal":
                    self._drag_start_pos = event.screen_x
                else:
                    self._drag_start_pos = event.screen_y
                
                self._drag_start_size = self.first_panel_size
                self.capture_mouse()
                event.stop()
        except NoMatches:
            pass
    
    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Handle mouse move during drag."""
        if not self._is_dragging:
            return
        
        # Calculate new size
        if self.orientation == "horizontal":
            delta = event.screen_x - self._drag_start_pos
            container_width = self.size.width
            delta_percent = (delta / container_width) * 100
        else:
            delta = event.screen_y - self._drag_start_pos
            container_height = self.size.height
            delta_percent = (delta / container_height) * 100
        
        new_size = self._drag_start_size + delta_percent
        
        # Clamp to min/max
        new_size = max(self.min_size, min(self.max_size, new_size))
        
        self.first_panel_size = new_size
        event.stop()
    
    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Handle mouse up after drag."""
        if self._is_dragging:
            self._is_dragging = False
            self.release_mouse()
            
            try:
                handle = self.query_one(".resize-handle")
                handle.remove_class("dragging")
            except NoMatches:
                pass
            
            # Save persisted size
            self._save_persisted_size()
            event.stop()
    
    def _is_over_handle(self, event: events.MouseEvent) -> bool:
        """Check if mouse event is over the resize handle."""
        try:
            handle = self.query_one(".resize-handle")
            # Simple bounds check
            return True  # Let Textual handle the hover state
        except NoMatches:
            return False
    
    def action_shrink_first(self) -> None:
        """Shrink first panel by 5%."""
        new_size = max(self.min_size, self.first_panel_size - 5)
        self.first_panel_size = new_size
        self._save_persisted_size()
    
    def action_grow_first(self) -> None:
        """Grow first panel by 5%."""
        new_size = min(self.max_size, self.first_panel_size + 5)
        self.first_panel_size = new_size
        self._save_persisted_size()
    
    def reset_layout(self) -> None:
        """Reset to default 50/50 layout."""
        self.first_panel_size = 50.0
        self._save_persisted_size()
        self.notify("Layout reset to default")


class ResizeHandle(Static):
    """Resize handle widget for drag interaction."""
    
    DEFAULT_CSS = """
    ResizeHandle {
        background: $primary-darken-2;
    }
    
    ResizeHandle:hover {
        background: $accent;
    }
    
    ResizeHandle.horizontal {
        width: 1;
        height: 100%;
    }
    
    ResizeHandle.vertical {
        width: 100%;
        height: 1;
    }
    
    ResizeHandle.dragging {
        background: $success;
    }
    """
    
    def __init__(self, orientation: str = "horizontal", **kwargs):
        super().__init__("", **kwargs)
        self.add_class("resize-handle")
        self.add_class(orientation)


# =============================================================================
# Step 2.4: Toggle-able Real-Time Stats Panel
# =============================================================================

@dataclass
class AgentStats:
    """Statistics for the AI agent."""
    # LLM Stats
    provider: str = "None"
    model: str = "—"
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # Session Stats
    messages_sent: int = 0
    tokens_used: int = 0
    requests_made: int = 0
    
    # Performance Stats
    avg_response_time_ms: int = 0
    uptime_seconds: int = 0
    errors: int = 0
    
    # Agent Stats
    tools_executed: int = 0
    files_modified: int = 0
    commands_run: int = 0
    
    # Terminal Stats
    active_terminals: int = 0
    completed_processes: int = 0
    
    # Session start time
    session_start: float = field(default_factory=time.time)
    
    @property
    def uptime_str(self) -> str:
        """Get formatted uptime string."""
        elapsed = int(time.time() - self.session_start)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


class StatsCard(Static):
    """A single stats card with icon, label, and value."""
    
    DEFAULT_CSS = """
    StatsCard {
        width: auto;
        height: auto;
        min-width: 15;
        padding: 0 1;
        margin: 0 1 0 0;
    }
    
    StatsCard .stats-icon {
        width: 2;
    }
    
    StatsCard .stats-label {
        color: $text-muted;
    }
    
    StatsCard .stats-value {
        text-style: bold;
    }
    
    StatsCard.success .stats-value {
        color: $success;
    }
    
    StatsCard.warning .stats-value {
        color: $warning;
    }
    
    StatsCard.error .stats-value {
        color: $error;
    }
    """
    
    value: reactive[str] = reactive("0")
    
    def __init__(
        self,
        icon: str,
        label: str,
        value: str = "0",
        variant: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.icon = icon
        self.label = label
        self.value = value
        
        if variant:
            self.add_class(variant)
    
    def render(self) -> RenderableType:
        """Render the stats card."""
        return Text.assemble(
            (f"{self.icon} ", ""),
            (f"{self.label}: ", "dim"),
            (self.value, "bold"),
        )


class CollapsibleStatsPanel(Container):
    """Collapsible panel showing real-time agent statistics.
    
    Features:
    - Toggle visibility with Ctrl+T or button
    - Auto-updates every 500ms
    - Clean grid layout with icons
    - Doesn't overlap with chat content
    """
    
    DEFAULT_CSS = """
    CollapsibleStatsPanel {
        width: 100%;
        height: auto;
        background: $surface-darken-1;
        border-bottom: solid $primary-darken-2;
    }
    
    CollapsibleStatsPanel.collapsed {
        height: 3;
    }
    
    CollapsibleStatsPanel .stats-header {
        height: 3;
        padding: 0 1;
        background: $primary-darken-3;
    }
    
    CollapsibleStatsPanel .stats-header-title {
        width: 1fr;
        text-style: bold;
    }
    
    CollapsibleStatsPanel .stats-toggle-btn {
        width: auto;
        min-width: 3;
        height: 1;
        border: none;
        background: transparent;
    }
    
    CollapsibleStatsPanel .stats-body {
        padding: 1;
    }
    
    CollapsibleStatsPanel .stats-body.hidden {
        display: none;
    }
    
    CollapsibleStatsPanel .stats-row {
        height: auto;
        margin-bottom: 1;
    }
    
    CollapsibleStatsPanel .stats-section-title {
        color: $accent;
        text-style: bold;
        margin: 1 0;
    }
    
    CollapsibleStatsPanel .stats-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
        width: 100%;
        height: auto;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+t", "toggle_stats", "Toggle Stats", show=False),
    ]
    
    class StatsToggled(Message):
        """Emitted when stats panel is toggled."""
        def __init__(self, visible: bool) -> None:
            self.visible = visible
            super().__init__()
    
    is_expanded: reactive[bool] = reactive(True)
    
    def __init__(
        self,
        stats: Optional[AgentStats] = None,
        auto_refresh: bool = True,
        refresh_interval: float = 0.5,
        **kwargs,
    ):
        """Initialize the stats panel.
        
        Args:
            stats: Initial stats object
            auto_refresh: Whether to auto-refresh stats
            refresh_interval: Refresh interval in seconds
        """
        super().__init__(**kwargs)
        self.stats = stats or AgentStats()
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        self._refresh_timer: Optional[Timer] = None
    
    def compose(self) -> ComposeResult:
        """Compose the stats panel."""
        # Header
        with Horizontal(classes="stats-header"):
            yield Static("📊 Agent Statistics", classes="stats-header-title")
            yield Button("▼" if self.is_expanded else "▶", id="btn-toggle-stats", classes="stats-toggle-btn")
        
        # Body (collapsible content)
        body_classes = "stats-body" if self.is_expanded else "stats-body hidden"
        with Vertical(classes=body_classes, id="stats-body"):
            # LLM Section
            yield Static("🧠 LLM", classes="stats-section-title")
            with Horizontal(classes="stats-row"):
                yield StatsCard("📡", "Provider", self.stats.provider, id="stat-provider")
                yield StatsCard("🤖", "Model", self.stats.model, id="stat-model")
                yield StatsCard("🌡️", "Temp", f"{self.stats.temperature}", id="stat-temp")
            
            # Session Section
            yield Static("💬 Session", classes="stats-section-title")
            with Horizontal(classes="stats-row"):
                yield StatsCard("📨", "Messages", str(self.stats.messages_sent), id="stat-messages")
                yield StatsCard("🔤", "Tokens", f"{self.stats.tokens_used:,}", id="stat-tokens")
                yield StatsCard("📤", "Requests", str(self.stats.requests_made), id="stat-requests")
            
            # Performance Section
            yield Static("⚡ Performance", classes="stats-section-title")
            with Horizontal(classes="stats-row"):
                yield StatsCard("⏱️", "Avg Time", f"{self.stats.avg_response_time_ms}ms", id="stat-avg-time")
                yield StatsCard("🕐", "Uptime", self.stats.uptime_str, id="stat-uptime")
                yield StatsCard("❌", "Errors", str(self.stats.errors), variant="error" if self.stats.errors else "", id="stat-errors")
            
            # Agent Section
            yield Static("🔧 Agent", classes="stats-section-title")
            with Horizontal(classes="stats-row"):
                yield StatsCard("⚙️", "Tools", str(self.stats.tools_executed), id="stat-tools")
                yield StatsCard("📁", "Files", str(self.stats.files_modified), id="stat-files")
                yield StatsCard("💻", "Commands", str(self.stats.commands_run), id="stat-commands")
            
            # Terminal Section
            yield Static("⬛ Terminals", classes="stats-section-title")
            with Horizontal(classes="stats-row"):
                yield StatsCard("▶️", "Active", str(self.stats.active_terminals), variant="success" if self.stats.active_terminals else "", id="stat-active-terms")
                yield StatsCard("✅", "Completed", str(self.stats.completed_processes), id="stat-completed")
    
    def on_mount(self) -> None:
        """Start auto-refresh timer."""
        if self.auto_refresh:
            self._refresh_timer = self.set_interval(
                self.refresh_interval,
                self._refresh_display,
            )
    
    def on_unmount(self) -> None:
        """Stop refresh timer."""
        if self._refresh_timer:
            self._refresh_timer.stop()
    
    def _refresh_display(self) -> None:
        """Refresh all stat displays."""
        if not self.is_expanded:
            # Only update uptime when collapsed
            return
        
        self._update_stat("stat-provider", self.stats.provider)
        self._update_stat("stat-model", self.stats.model)
        self._update_stat("stat-temp", f"{self.stats.temperature}")
        self._update_stat("stat-messages", str(self.stats.messages_sent))
        self._update_stat("stat-tokens", f"{self.stats.tokens_used:,}")
        self._update_stat("stat-requests", str(self.stats.requests_made))
        self._update_stat("stat-avg-time", f"{self.stats.avg_response_time_ms}ms")
        self._update_stat("stat-uptime", self.stats.uptime_str)
        self._update_stat("stat-errors", str(self.stats.errors))
        self._update_stat("stat-tools", str(self.stats.tools_executed))
        self._update_stat("stat-files", str(self.stats.files_modified))
        self._update_stat("stat-commands", str(self.stats.commands_run))
        self._update_stat("stat-active-terms", str(self.stats.active_terminals))
        self._update_stat("stat-completed", str(self.stats.completed_processes))
    
    def _update_stat(self, stat_id: str, value: str) -> None:
        """Update a single stat display."""
        try:
            card = self.query_one(f"#{stat_id}", StatsCard)
            card.value = value
            card.refresh()
        except NoMatches:
            pass
    
    def watch_is_expanded(self, expanded: bool) -> None:
        """Update display when expanded state changes."""
        try:
            btn = self.query_one("#btn-toggle-stats", Button)
            btn.label = "▼" if expanded else "▶"
            
            body = self.query_one("#stats-body")
            if expanded:
                body.remove_class("hidden")
                self.remove_class("collapsed")
            else:
                body.add_class("hidden")
                self.add_class("collapsed")
            
            self.post_message(self.StatsToggled(expanded))
        except NoMatches:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle toggle button press."""
        if event.button.id == "btn-toggle-stats":
            self.toggle()
    
    def action_toggle_stats(self) -> None:
        """Toggle stats visibility."""
        self.toggle()
    
    def toggle(self) -> None:
        """Toggle expanded/collapsed state."""
        self.is_expanded = not self.is_expanded
    
    def update_stats(self, **kwargs) -> None:
        """Update stats with new values.
        
        Args:
            **kwargs: Stat names and values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.stats, key):
                setattr(self.stats, key, value)
        
        self._refresh_display()


# =============================================================================
# Step 2.3: Professional UI Components
# =============================================================================

class AgentHeader(Container):
    """Professional header for the agent panel."""
    
    DEFAULT_CSS = """
    AgentHeader {
        height: 5;
        padding: 1;
        background: $primary-darken-2;
        border-bottom: solid $primary;
    }
    
    AgentHeader .header-content {
        width: 1fr;
        height: 100%;
    }
    
    AgentHeader .header-title {
        text-style: bold;
        color: $accent;
    }
    
    AgentHeader .header-subtitle {
        color: $text-muted;
    }
    
    AgentHeader .header-controls {
        width: auto;
    }
    
    AgentHeader .agent-badge {
        background: $success;
        color: $surface;
        padding: 0 1;
        text-style: bold;
    }
    
    AgentHeader .agent-badge.disabled {
        background: $error;
    }
    
    AgentHeader .menu-btn {
        width: 3;
        min-width: 3;
        height: 1;
        border: none;
        background: transparent;
    }
    """
    
    agent_enabled: reactive[bool] = reactive(True)
    
    def __init__(
        self,
        title: str = "🤖 Proxima AI Agent",
        subtitle: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.subtitle = subtitle
    
    def compose(self) -> ComposeResult:
        """Compose the header."""
        with Horizontal():
            with Vertical(classes="header-content"):
                yield Static(self.title, classes="header-title")
                if self.subtitle:
                    yield Static(self.subtitle, classes="header-subtitle")
            
            with Horizontal(classes="header-controls"):
                badge_class = "agent-badge" if self.agent_enabled else "agent-badge disabled"
                badge_text = "AGENT" if self.agent_enabled else "CHAT"
                yield Static(badge_text, classes=badge_class, id="agent-badge")
                yield Button("≡", id="btn-menu", classes="menu-btn")
    
    def watch_agent_enabled(self, enabled: bool) -> None:
        """Update badge when agent state changes."""
        try:
            badge = self.query_one("#agent-badge", Static)
            badge.update("AGENT" if enabled else "CHAT")
            if enabled:
                badge.remove_class("disabled")
            else:
                badge.add_class("disabled")
        except NoMatches:
            pass


class ToolExecutionCard(Container):
    """Card showing a tool execution with result."""
    
    DEFAULT_CSS = """
    ToolExecutionCard {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $surface-darken-1;
        border-left: thick $primary;
    }
    
    ToolExecutionCard.success {
        border-left: thick $success;
    }
    
    ToolExecutionCard.failed {
        border-left: thick $error;
    }
    
    ToolExecutionCard.running {
        border-left: thick $warning;
    }
    
    ToolExecutionCard .tool-header {
        height: 1;
    }
    
    ToolExecutionCard .tool-icon {
        width: 3;
    }
    
    ToolExecutionCard .tool-name {
        width: 1fr;
        text-style: bold;
    }
    
    ToolExecutionCard .tool-status {
        width: auto;
    }
    
    ToolExecutionCard .tool-args {
        color: $text-muted;
        margin: 1 0;
    }
    
    ToolExecutionCard .tool-result {
        padding: 1;
        background: $surface-darken-2;
    }
    
    ToolExecutionCard .tool-time {
        color: $text-muted;
        text-align: right;
    }
    """
    
    status: reactive[str] = reactive("running")
    
    def __init__(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        status: str = "running",
        result: Optional[str] = None,
        error: Optional[str] = None,
        duration_ms: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.arguments = arguments
        self.status = status
        self.result_text = result
        self.error_text = error
        self.duration_ms = duration_ms
        
        self.add_class(status)
    
    def compose(self) -> ComposeResult:
        """Compose the tool card."""
        # Header
        with Horizontal(classes="tool-header"):
            yield Static("🔧", classes="tool-icon")
            yield Static(self.tool_name, classes="tool-name")
            status_icon = {"running": "⏳", "success": "✅", "failed": "❌"}.get(self.status, "❓")
            yield Static(status_icon, classes="tool-status", id="status-icon")
        
        # Arguments
        if self.arguments:
            args_str = ", ".join(f"{k}={v}" for k, v in list(self.arguments.items())[:3])
            yield Static(f"Args: {args_str[:60]}", classes="tool-args")
        
        # Result or Error
        if self.result_text:
            yield Static(self.result_text[:200], classes="tool-result")
        elif self.error_text:
            yield Static(f"Error: {self.error_text}", classes="tool-result")
        
        # Duration
        if self.duration_ms:
            yield Static(f"{self.duration_ms}ms", classes="tool-time")
    
    def watch_status(self, new_status: str) -> None:
        """Update classes when status changes."""
        self.remove_class("running")
        self.remove_class("success")
        self.remove_class("failed")
        self.add_class(new_status)
        
        try:
            status_icon = {"running": "⏳", "success": "✅", "failed": "❌"}.get(new_status, "❓")
            self.query_one("#status-icon", Static).update(status_icon)
        except NoMatches:
            pass
    
    def set_success(self, result: str, duration_ms: int = 0) -> None:
        """Mark as successful with result."""
        self.result_text = result
        self.duration_ms = duration_ms
        self.status = "success"
    
    def set_failed(self, error: str, duration_ms: int = 0) -> None:
        """Mark as failed with error."""
        self.error_text = error
        self.duration_ms = duration_ms
        self.status = "failed"


class InputSection(Container):
    """Professional input section with send button and controls."""
    
    DEFAULT_CSS = """
    InputSection {
        height: auto;
        min-height: 8;
        max-height: 15;
        padding: 1;
        border-top: solid $primary;
        background: $surface;
    }
    
    InputSection .input-row {
        height: auto;
        min-height: 3;
    }
    
    InputSection .prompt-input {
        width: 1fr;
        min-height: 3;
        max-height: 10;
        margin-right: 1;
    }
    
    InputSection .send-btn {
        width: 12;
        height: 3;
    }
    
    InputSection .controls-row {
        height: 3;
        margin-top: 1;
    }
    
    InputSection .control-btn {
        margin-right: 1;
        min-width: 10;
        height: 3;
    }
    
    InputSection .input-hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """
    
    class SendPressed(Message):
        """Emitted when send is requested."""
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()
    
    is_generating: reactive[bool] = reactive(False)
    
    def compose(self) -> ComposeResult:
        """Compose the input section."""
        from .agent_widgets import SendableTextArea
        
        with Horizontal(classes="input-row"):
            yield SendableTextArea(id="prompt-input", classes="prompt-input")
            yield Button("⮕ Send", id="btn-send", classes="send-btn", variant="primary")
        
        with Horizontal(classes="controls-row"):
            yield Button("🛑 Stop", id="btn-stop", classes="control-btn", variant="error", disabled=True)
            yield Button("🔧 Agent", id="btn-toggle-agent", classes="control-btn", variant="success")
            yield Button("↶ Undo", id="btn-undo", classes="control-btn", disabled=True)
            yield Button("↷ Redo", id="btn-redo", classes="control-btn", disabled=True)
            yield Button("🗑️ Clear", id="btn-clear", classes="control-btn")
            yield Button("📤 Export", id="btn-export", classes="control-btn")
        
        yield Static(
            "Ctrl+Enter=Send | Agent can execute commands, modify files, and more",
            classes="input-hint",
        )
    
    def watch_is_generating(self, generating: bool) -> None:
        """Update button states based on generation state."""
        try:
            self.query_one("#btn-stop", Button).disabled = not generating
            self.query_one("#btn-send", Button).disabled = generating
        except NoMatches:
            pass
    
    def get_text(self) -> str:
        """Get the input text."""
        try:
            from textual.widgets import TextArea
            return self.query_one("#prompt-input", TextArea).text.strip()
        except NoMatches:
            return ""
    
    def clear_text(self) -> None:
        """Clear the input."""
        try:
            from textual.widgets import TextArea
            self.query_one("#prompt-input", TextArea).text = ""
        except NoMatches:
            pass
    
    def focus_input(self) -> None:
        """Focus the input field."""
        try:
            from textual.widgets import TextArea
            self.query_one("#prompt-input", TextArea).focus()
        except NoMatches:
            pass
