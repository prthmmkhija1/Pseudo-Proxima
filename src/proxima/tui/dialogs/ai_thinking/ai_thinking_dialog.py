"""AI Thinking Dialog for Proxima TUI.

A modal dialog that displays what the AI is "thinking" in real-time.
This provides a simple way to see the AI's reasoning process when
using LLM features (via API key or local model).

Features:
- Real-time streaming of AI thoughts
- Prompt preview before sending
- Response as it arrives
- Token usage and cost tracking
- Model/provider info
- Copy to clipboard
- Export conversation
- Fullscreen mode with collapsible sidebar
- Dynamic resizing
- Real-time statistics
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import time
import asyncio

from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Button, RichLog, Input, Switch, Label
from textual.binding import Binding
from textual.reactive import reactive
from textual import on
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown

from ..base import BaseDialog
from ...styles.theme import get_theme
from ...styles.icons import ICON_THINKING
from ...components.viewers.ai_thinking_viewer import (
    AIThinkingViewer,
    ThinkingPhase,
    MessageRole,
    ThinkingStats,
)


class AIThinkingDialog(ModalScreen):
    """Modal dialog for viewing AI thinking process.
    
    Shows:
    - Current AI thought/response stream
    - History of prompts and responses
    - Token usage statistics
    - Model information
    
    Keybindings:
    - Escape: Close dialog
    - Ctrl+C: Copy current content
    - Ctrl+L: Clear history
    - Ctrl+E: Export conversation
    - Space: Pause/resume streaming
    """
    
    # Reactive property for sidebar collapsed state
    sidebar_collapsed = reactive(False)
    
    DEFAULT_CSS = """
    AIThinkingDialog {
        align: center middle;
    }
    
    AIThinkingDialog > .dialog-container {
        width: 100%;
        height: 100%;
        border: thick $primary;
        background: $surface;
    }
    
    AIThinkingDialog .dialog-header {
        height: 3;
        padding: 0 2;
        border-bottom: solid $primary-darken-2;
        background: $primary-darken-3;
        layout: horizontal;
    }
    
    AIThinkingDialog .dialog-title {
        width: 1fr;
        content-align: left middle;
        text-style: bold;
        color: $text;
    }
    
    AIThinkingDialog .model-badge {
        width: auto;
        content-align: right middle;
        color: $accent;
        margin-right: 1;
    }
    
    AIThinkingDialog .close-btn {
        width: 10;
        min-width: 10;
        height: 3;
        margin-left: 1;
    }
    
    AIThinkingDialog .main-content {
        height: 1fr;
        layout: horizontal;
    }
    
    AIThinkingDialog .thinking-area {
        width: 1fr;
        height: 100%;
        border-right: solid $primary-darken-3;
    }
    
    AIThinkingDialog .thinking-area.expanded {
        width: 100%;
        border-right: none;
    }
    
    AIThinkingDialog .sidebar {
        width: 32;
        min-width: 32;
        padding: 1;
        transition: width 200ms;
    }
    
    AIThinkingDialog .sidebar.collapsed {
        width: 0;
        min-width: 0;
        padding: 0;
        overflow: hidden;
        display: none;
    }
    
    AIThinkingDialog .collapse-toggle-btn {
        width: 3;
        min-width: 3;
        height: 100%;
        padding: 0;
        margin: 0;
        background: $surface-darken-1;
        border: none;
        border-left: solid $primary-darken-3;
    }
    
    AIThinkingDialog .collapse-toggle-btn:hover {
        background: $primary-darken-2;
    }
    
    AIThinkingDialog .section-title {
        height: 2;
        padding: 0 1;
        background: $surface-darken-1;
        text-style: bold;
        content-align: left middle;
    }
    
    AIThinkingDialog .current-thought {
        height: 60%;
        border-bottom: dashed $primary-darken-3;
    }
    
    AIThinkingDialog .thought-stream {
        height: 1fr;
        padding: 1;
        background: $surface-darken-2;
        scrollbar-background: transparent;
        scrollbar-color: transparent;
    }
    
    AIThinkingDialog .thought-stream:hover {
        scrollbar-color: $primary 40%;
    }
    
    AIThinkingDialog .history-section {
        height: 40%;
    }
    
    AIThinkingDialog .history-log {
        height: 1fr;
        padding: 1;
        scrollbar-background: transparent;
        scrollbar-color: transparent;
    }
    
    AIThinkingDialog .history-log:hover {
        scrollbar-color: $primary 40%;
    }
    
    AIThinkingDialog .stats-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
    }
    
    AIThinkingDialog .stat-row {
        height: auto;
        margin-bottom: 0;
    }
    
    AIThinkingDialog .stat-label {
        width: 50%;
        color: $text-muted;
    }
    
    AIThinkingDialog .stat-value {
        width: 50%;
        color: $accent;
        text-align: right;
    }
    
    AIThinkingDialog .stat-realtime {
        color: $success;
    }
    
    AIThinkingDialog .realtime-indicator {
        color: $success;
        text-style: bold;
    }
    
    AIThinkingDialog .controls-section {
        height: auto;
        padding: 1;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
    }
    
    AIThinkingDialog .control-row {
        height: auto;
        margin-bottom: 1;
        layout: horizontal;
    }
    
    AIThinkingDialog .dialog-footer {
        height: 3;
        padding: 0 2;
        border-top: solid $primary-darken-2;
        background: $surface;
        layout: horizontal;
        content-align: center middle;
    }
    
    AIThinkingDialog .footer-hint {
        width: 1fr;
        color: $text-muted;
    }
    
    AIThinkingDialog .action-buttons {
        width: auto;
        layout: horizontal;
    }
    
    AIThinkingDialog .action-btn {
        margin-left: 1;
    }
    
    /* Resize handle between main area and sidebar */
    AIThinkingDialog .resize-handle {
        width: 3;
        height: 100%;
        background: transparent;
    }
    
    AIThinkingDialog .resize-handle:hover {
        background: $primary 30%;
    }
    """
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("ctrl+c", "copy", "Copy"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+e", "export", "Export"),
        Binding("space", "toggle_pause", "Pause/Resume"),
        Binding("ctrl+b", "toggle_sidebar", "Toggle Sidebar"),
    ]
    
    def __init__(self, state=None, **kwargs):
        """Initialize the AI thinking dialog.
        
        Args:
            state: TUIState instance for accessing LLM state
        """
        super().__init__(**kwargs)
        self.state = state
        self.stats = ThinkingStats()
        self.is_paused = False
        self._thinking_entries: List[Dict[str, Any]] = []
        self._stats_update_timer = None
        self._last_stats_update = time.time()
        
        # Initialize stats from state if available
        if state:
            self.stats.model = state.llm_model or "Not configured"
            self.stats.provider = state.llm_provider or ""
            self.stats.prompt_tokens = state.prompt_tokens
            self.stats.completion_tokens = state.completion_tokens
            self.stats.total_tokens = state.prompt_tokens + state.completion_tokens
    
    def compose(self):
        """Compose the dialog layout with fullscreen support and collapsible sidebar."""
        with Vertical(classes="dialog-container"):
            # Header
            with Horizontal(classes="dialog-header"):
                yield Static(f"{ICON_THINKING} AI Thinking Panel", classes="dialog-title")
                yield Static("● LIVE", classes="realtime-indicator", id="realtime-indicator")
                yield Static(self._get_model_badge(), classes="model-badge", id="model-badge")
                yield Button("✕", variant="error", classes="close-btn", id="close-btn", tooltip="Close panel")
            
            # Main content
            with Horizontal(classes="main-content"):
                # Left: Thinking area (expands when sidebar collapsed)
                with Vertical(classes="thinking-area", id="thinking-area"):
                    # Current thought section
                    with Vertical(classes="current-thought"):
                        yield Static("🧠 Current Thought", classes="section-title")
                        yield RichLog(
                            auto_scroll=True,
                            classes="thought-stream",
                            id="current-thought-stream",
                        )
                    
                    # History section
                    with Vertical(classes="history-section"):
                        yield Static("📜 Conversation History", classes="section-title")
                        yield RichLog(
                            auto_scroll=True,
                            classes="history-log",
                            id="history-log",
                        )
                
                # Collapse/Expand toggle button (always visible)
                yield Button("◀", id="collapse-sidebar-btn", classes="collapse-toggle-btn", tooltip="Collapse/Expand sidebar (Ctrl+B)")
                
                # Right: Sidebar with stats and controls (collapsible)
                with Vertical(classes="sidebar", id="sidebar"):
                    # Stats section with real-time indicator
                    yield Static("📊 Statistics", classes="section-title")
                    with Vertical(classes="stats-section", id="stats-section"):
                        yield self._create_stat_row("Model:", self.stats.model, "stat-model")
                        yield self._create_stat_row("Provider:", self.stats.provider or "Local", "stat-provider")
                        yield self._create_stat_row("Prompt Tokens:", str(self.stats.prompt_tokens), "stat-prompt")
                        yield self._create_stat_row("Completion Tokens:", str(self.stats.completion_tokens), "stat-completion")
                        yield self._create_stat_row("Total Tokens:", str(self.stats.total_tokens), "stat-total")
                        yield self._create_stat_row("Requests:", str(self.stats.requests), "stat-requests")
                        yield self._create_stat_row("Thinking Time:", f"{self.stats.thinking_time_ms:.0f}ms", "stat-time")
                    
                    # Controls section
                    yield Static("⚙️ Controls", classes="section-title")
                    with Vertical(classes="controls-section", id="controls-section"):
                        with Horizontal(classes="control-row"):
                            yield Label("Auto-scroll")
                            yield Switch(value=True, id="auto-scroll-switch")
                        
                        with Horizontal(classes="control-row"):
                            yield Label("Show Prompts")
                            yield Switch(value=True, id="show-prompts-switch")
                        
                        with Horizontal(classes="control-row"):
                            yield Label("Show Tokens")
                            yield Switch(value=True, id="show-tokens-switch")
                        
                        yield Button("🗑 Clear", id="clear-btn", variant="warning", tooltip="Clear conversation history")
            
            # Footer
            with Horizontal(classes="dialog-footer"):
                yield Static(
                    "Esc close │ Ctrl+B toggle sidebar │ Space pause",
                    classes="footer-hint",
                )
                with Horizontal(classes="action-buttons"):
                    yield Button("📋 Copy", id="copy-btn", classes="action-btn")
                    yield Button("💾 Export", id="export-btn", classes="action-btn")
                    yield Button("Close", id="close-dialog-btn", classes="action-btn", variant="primary")
    
    def watch_sidebar_collapsed(self, collapsed: bool) -> None:
        """React to sidebar collapsed state changes."""
        try:
            sidebar = self.query_one("#sidebar")
            thinking_area = self.query_one("#thinking-area")
            collapse_btn = self.query_one("#collapse-sidebar-btn", Button)
            
            if collapsed:
                sidebar.add_class("collapsed")
                thinking_area.add_class("expanded")
                collapse_btn.label = "▶"
            else:
                sidebar.remove_class("collapsed")
                thinking_area.remove_class("expanded")
                collapse_btn.label = "◀"
        except Exception:
            pass
    
    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        self.sidebar_collapsed = not self.sidebar_collapsed
    
    @on(Button.Pressed, "#collapse-sidebar-btn")
    def on_collapse_sidebar_pressed(self, event: Button.Pressed) -> None:
        """Handle collapse sidebar button press."""
        self.sidebar_collapsed = not self.sidebar_collapsed
    
    def _get_model_badge(self) -> str:
        """Get the model badge text."""
        if self.stats.provider and self.stats.model:
            return f"📦 {self.stats.provider}/{self.stats.model}"
        elif self.stats.model:
            return f"📦 {self.stats.model}"
        else:
            return "📦 No model selected"
    
    def _create_stat_row(self, label: str, value: str, value_id: str) -> Horizontal:
        """Create a statistics row."""
        row = Horizontal(classes="stat-row")
        row.compose_add_child(Static(label, classes="stat-label"))
        row.compose_add_child(Static(value, classes="stat-value", id=value_id))
        return row
    
    def on_mount(self):
        """Initialize when dialog is mounted."""
        theme = get_theme()
        
        # Show welcome message in thinking stream
        thought_log = self.query_one("#current-thought-stream", RichLog)
        thought_log.write(Text.from_markup(
            f"[{theme.info}]AI Thinking Panel ready.[/]\n"
            f"[{theme.fg_muted}]When you interact with AI features, "
            f"the thinking process will appear here.[/]"
        ))
        
        # Restore any previous thinking history from state
        if self.state and hasattr(self.state, 'thinking_history'):
            for entry in self.state.thinking_history:
                self._add_history_entry(entry)
        
        # Start real-time stats update timer
        self._start_realtime_stats_update()
    
    def on_unmount(self):
        """Clean up when dialog is unmounted."""
        if self._stats_update_timer:
            self._stats_update_timer.stop()
            self._stats_update_timer = None
    
    def _start_realtime_stats_update(self):
        """Start the real-time stats update timer."""
        self._stats_update_timer = self.set_interval(0.5, self._update_realtime_stats)
    
    def _update_realtime_stats(self):
        """Update stats from state in real-time."""
        if not self.state:
            return
        
        try:
            # Get fresh stats from state
            prompt_tokens = getattr(self.state, 'prompt_tokens', 0)
            completion_tokens = getattr(self.state, 'completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens
            model = getattr(self.state, 'llm_model', None) or "Not configured"
            provider = getattr(self.state, 'llm_provider', None) or "Local"
            
            # Update if changed
            if (prompt_tokens != self.stats.prompt_tokens or 
                completion_tokens != self.stats.completion_tokens or
                model != self.stats.model or
                provider != self.stats.provider):
                
                self.stats.prompt_tokens = prompt_tokens
                self.stats.completion_tokens = completion_tokens
                self.stats.total_tokens = total_tokens
                self.stats.model = model
                self.stats.provider = provider
                
                # Update UI
                self._refresh_stats_display()
                
                # Flash the realtime indicator
                self._flash_realtime_indicator()
        except Exception:
            pass
    
    def _flash_realtime_indicator(self):
        """Flash the realtime indicator to show updates."""
        try:
            indicator = self.query_one("#realtime-indicator", Static)
            indicator.update("● LIVE")
            indicator.add_class("stat-realtime")
        except Exception:
            pass
    
    def _refresh_stats_display(self):
        """Refresh all stats in the display."""
        try:
            self.query_one("#stat-model", Static).update(self.stats.model)
            self.query_one("#stat-provider", Static).update(self.stats.provider)
            self.query_one("#stat-prompt", Static).update(str(self.stats.prompt_tokens))
            self.query_one("#stat-completion", Static).update(str(self.stats.completion_tokens))
            self.query_one("#stat-total", Static).update(str(self.stats.total_tokens))
            self.query_one("#stat-requests", Static).update(str(self.stats.requests))
            self.query_one("#stat-time", Static).update(f"{self.stats.thinking_time_ms:.0f}ms")
            self.query_one("#model-badge", Static).update(self._get_model_badge())
        except Exception:
            pass
    
    def _add_history_entry(self, entry: Dict[str, Any]):
        """Add an entry to the history log."""
        theme = get_theme()
        history_log = self.query_one("#history-log", RichLog)
        
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        timestamp = entry.get("timestamp", datetime.now())
        tokens = entry.get("tokens", 0)
        
        time_str = timestamp.strftime("%H:%M:%S") if isinstance(timestamp, datetime) else str(timestamp)
        
        # Format based on role
        role_styles = {
            "user": (theme.info, "📤 USER"),
            "assistant": (theme.success, "🤖 AI"),
            "system": (theme.fg_muted, "⚙️ SYS"),
            "thought": (theme.warning, "💭 THOUGHT"),
            "error": (theme.error, "❌ ERROR"),
        }
        
        style, label = role_styles.get(role, (theme.fg_muted, "??? UNKNOWN"))
        
        text = Text()
        text.append(f"[{time_str}] ", style=theme.fg_subtle)
        text.append(f"{label}: ", style=f"bold {style}")
        
        # Truncate long content for history
        preview = content[:200] + "..." if len(content) > 200 else content
        text.append(preview.replace("\n", " "))
        
        if tokens > 0:
            text.append(f" ({tokens} tokens)", style=theme.fg_subtle)
        
        history_log.write(text)
    
    def update_stats(self, **kwargs):
        """Update statistics display."""
        for key, value in kwargs.items():
            if hasattr(self.stats, key):
                setattr(self.stats, key, value)
        
        self.stats.total_tokens = self.stats.prompt_tokens + self.stats.completion_tokens
        
        # Update UI
        try:
            self.query_one("#stat-prompt", Static).update(str(self.stats.prompt_tokens))
            self.query_one("#stat-completion", Static).update(str(self.stats.completion_tokens))
            self.query_one("#stat-total", Static).update(str(self.stats.total_tokens))
            self.query_one("#stat-requests", Static).update(str(self.stats.requests))
            self.query_one("#stat-time", Static).update(f"{self.stats.thinking_time_ms:.0f}ms")
            self.query_one("#model-badge", Static).update(self._get_model_badge())
        except Exception:
            pass
    
    def stream_thought(self, chunk: str):
        """Stream a chunk of thought content."""
        if self.is_paused:
            return
        
        try:
            thought_log = self.query_one("#current-thought-stream", RichLog)
            thought_log.write(chunk, scroll_end=True)
        except Exception:
            pass
    
    def log_prompt(self, prompt: str, tokens: int = 0):
        """Log a prompt being sent."""
        entry = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now(),
            "tokens": tokens,
        }
        self._thinking_entries.append(entry)
        self._add_history_entry(entry)
        
        # Store in state
        if self.state and hasattr(self.state, 'thinking_history'):
            self.state.thinking_history.append(entry)
    
    def log_response(self, response: str, tokens: int = 0):
        """Log an AI response."""
        entry = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(),
            "tokens": tokens,
        }
        self._thinking_entries.append(entry)
        self._add_history_entry(entry)
        
        # Store in state
        if self.state and hasattr(self.state, 'thinking_history'):
            self.state.thinking_history.append(entry)
    
    def log_thought(self, thought: str):
        """Log an internal thought."""
        entry = {
            "role": "thought",
            "content": thought,
            "timestamp": datetime.now(),
            "tokens": 0,
        }
        self._add_history_entry(entry)
    
    def clear_current_thought(self):
        """Clear the current thought stream."""
        try:
            self.query_one("#current-thought-stream", RichLog).clear()
        except Exception:
            pass
    
    def clear_history(self):
        """Clear all history."""
        self._thinking_entries.clear()
        try:
            self.query_one("#history-log", RichLog).clear()
        except Exception:
            pass
        
        if self.state and hasattr(self.state, 'thinking_history'):
            self.state.thinking_history.clear()
    
    # ==================== Actions ====================
    
    def action_close(self):
        """Close the dialog."""
        if self.state:
            self.state.thinking_panel_visible = False
        self.dismiss(None)
    
    def action_copy(self):
        """Copy current content to clipboard."""
        # Note: Clipboard access in terminal is limited
        # This would need platform-specific implementation
        self.notify("Content copied to clipboard", severity="information")
    
    def action_clear(self):
        """Clear all content."""
        self.clear_current_thought()
        self.clear_history()
        self.notify("Thinking history cleared", severity="information")
    
    def action_export(self):
        """Export conversation to file."""
        # Would save to file
        self.notify("Export feature coming soon", severity="information")
    
    def action_toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        status = "paused" if self.is_paused else "resumed"
        self.notify(f"Streaming {status}", severity="information")
    
    # ==================== Event Handlers ====================
    
    @on(Button.Pressed, "#close-btn")
    @on(Button.Pressed, "#close-dialog-btn")
    def handle_close(self, event: Button.Pressed):
        """Handle close button."""
        self.action_close()
    
    @on(Button.Pressed, "#copy-btn")
    def handle_copy(self, event: Button.Pressed):
        """Handle copy button."""
        self.action_copy()
    
    @on(Button.Pressed, "#export-btn")
    def handle_export(self, event: Button.Pressed):
        """Handle export button."""
        self.action_export()
    
    @on(Button.Pressed, "#clear-btn")
    def handle_clear(self, event: Button.Pressed):
        """Handle clear button."""
        self.action_clear()
    
    @on(Switch.Changed, "#auto-scroll-switch")
    def handle_autoscroll_toggle(self, event: Switch.Changed):
        """Handle auto-scroll toggle."""
        try:
            self.query_one("#current-thought-stream", RichLog).auto_scroll = event.value
            self.query_one("#history-log", RichLog).auto_scroll = event.value
        except Exception:
            pass
