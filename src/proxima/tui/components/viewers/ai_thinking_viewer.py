"""AI Thinking Viewer Component for Proxima TUI.

A streaming viewer that displays what the AI (LLM) is "thinking" in real-time.
Shows the AI's reasoning process, prompts being sent, and streaming responses.

Inspired by:
- Claude's extended thinking display
- Crush agent's reasoning panel
- OpenCode's streaming thoughts

Features:
- Real-time streaming of AI thoughts
- Token usage tracking
- Model/provider info display
- Collapsible sections for prompts and responses
- Auto-scroll with manual override
- Clear/copy functionality
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from enum import Enum

from textual.widgets import RichLog, Static, Button
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.message import Message
from textual import on
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import RenderableType

from ...styles.theme import get_theme
from ...styles.icons import ICON_THINKING


# ======================== ENUMS ========================


class ThinkingPhase(Enum):
    """Phases of AI thinking."""
    IDLE = "idle"
    PREPARING = "preparing"
    SENDING = "sending"
    THINKING = "thinking"
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"


class MessageRole(Enum):
    """Role of a message in the thinking process."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    THOUGHT = "thought"
    ERROR = "error"


# ======================== DATACLASSES ========================


@dataclass
class ThinkingEntry:
    """A single entry in the thinking log."""
    
    timestamp: datetime
    phase: ThinkingPhase
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    
    def render(self, theme) -> Text:
        """Render the entry as Rich Text."""
        text = Text()
        
        # Timestamp
        time_str = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        text.append(f"[{time_str}] ", style=theme.fg_subtle)
        
        # Phase icon
        phase_icons = {
            ThinkingPhase.IDLE: "○",
            ThinkingPhase.PREPARING: "◎",
            ThinkingPhase.SENDING: "◉",
            ThinkingPhase.THINKING: "◆",
            ThinkingPhase.STREAMING: "▶",
            ThinkingPhase.COMPLETE: "✓",
            ThinkingPhase.ERROR: "✗",
        }
        icon = phase_icons.get(self.phase, "•")
        
        phase_colors = {
            ThinkingPhase.IDLE: theme.fg_muted,
            ThinkingPhase.PREPARING: theme.info,
            ThinkingPhase.SENDING: theme.accent,
            ThinkingPhase.THINKING: theme.warning,
            ThinkingPhase.STREAMING: theme.success,
            ThinkingPhase.COMPLETE: theme.success,
            ThinkingPhase.ERROR: theme.error,
        }
        phase_color = phase_colors.get(self.phase, theme.fg_muted)
        text.append(f"{icon} ", style=f"bold {phase_color}")
        
        # Role label
        role_labels = {
            MessageRole.SYSTEM: "[SYS]",
            MessageRole.USER: "[USR]",
            MessageRole.ASSISTANT: "[AI]",
            MessageRole.THOUGHT: "[THK]",
            MessageRole.ERROR: "[ERR]",
        }
        role_label = role_labels.get(self.role, "[???]")
        
        role_colors = {
            MessageRole.SYSTEM: theme.fg_muted,
            MessageRole.USER: theme.info,
            MessageRole.ASSISTANT: theme.accent,
            MessageRole.THOUGHT: theme.warning,
            MessageRole.ERROR: theme.error,
        }
        role_color = role_colors.get(self.role, theme.fg_muted)
        text.append(f"{role_label} ", style=f"bold {role_color}")
        
        # Content (truncated if too long for inline display)
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        text.append(content_preview.replace("\n", "↵"), style=theme.fg_base)
        
        # Token count if available
        if self.tokens > 0:
            text.append(f" ({self.tokens} tokens)", style=theme.fg_subtle)
        
        return text


@dataclass
class ThinkingStats:
    """Statistics about AI thinking."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thinking_time_ms: float = 0.0
    model: str = ""
    provider: str = ""
    requests: int = 0
    estimated_cost: float = 0.0


# ======================== MESSAGES ========================


class ThinkingToggled(Message):
    """Message when thinking visibility is toggled."""
    
    def __init__(self, visible: bool):
        super().__init__()
        self.visible = visible


class ThinkingCleared(Message):
    """Message when thinking log is cleared."""
    pass


# ======================== WIDGETS ========================


class ThinkingHeader(Static):
    """Header showing current thinking status and controls."""
    
    phase = reactive(ThinkingPhase.IDLE)
    model = reactive("")
    provider = reactive("")
    
    DEFAULT_CSS = """
    ThinkingHeader {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary-darken-3;
        background: $surface;
        layout: horizontal;
    }
    
    ThinkingHeader .status {
        width: 1fr;
    }
    
    ThinkingHeader .model-info {
        width: auto;
        color: $text-muted;
    }
    
    ThinkingHeader .controls {
        width: auto;
        layout: horizontal;
    }
    """
    
    def compose(self):
        """Compose the header."""
        yield Static(classes="status", id="thinking-status")
        yield Static(classes="model-info", id="model-info")
        with Horizontal(classes="controls"):
            yield Button("⏸", variant="default", id="pause-btn")
            yield Button("🗑", variant="default", id="clear-btn")
    
    def on_mount(self):
        """Initialize on mount."""
        self._update_display()
    
    def watch_phase(self, phase: ThinkingPhase):
        """React to phase changes."""
        self._update_display()
    
    def watch_model(self, model: str):
        """React to model changes."""
        self._update_display()
    
    def _update_display(self):
        """Update the display based on current state."""
        theme = get_theme()
        
        # Status
        phase_messages = {
            ThinkingPhase.IDLE: "🔵 Ready",
            ThinkingPhase.PREPARING: "⚙️ Preparing prompt...",
            ThinkingPhase.SENDING: "📤 Sending to AI...",
            ThinkingPhase.THINKING: "🧠 AI is thinking...",
            ThinkingPhase.STREAMING: "✨ Receiving response...",
            ThinkingPhase.COMPLETE: "✅ Complete",
            ThinkingPhase.ERROR: "❌ Error occurred",
        }
        
        status_text = phase_messages.get(self.phase, "Unknown")
        try:
            self.query_one("#thinking-status", Static).update(status_text)
        except Exception:
            pass
        
        # Model info
        if self.model and self.provider:
            model_info = f"📦 {self.provider}/{self.model}"
        elif self.model:
            model_info = f"📦 {self.model}"
        else:
            model_info = "No model selected"
        
        try:
            self.query_one("#model-info", Static).update(model_info)
        except Exception:
            pass


class ThinkingTokenStats(Static):
    """Token usage statistics display."""
    
    DEFAULT_CSS = """
    ThinkingTokenStats {
        height: 2;
        padding: 0 1;
        border-top: solid $primary-darken-3;
        background: $surface;
        layout: horizontal;
    }
    
    ThinkingTokenStats .stat {
        width: 1fr;
        content-align: center middle;
    }
    """
    
    def __init__(self, stats: Optional[ThinkingStats] = None, **kwargs):
        super().__init__(**kwargs)
        self.stats = stats or ThinkingStats()
    
    def update_stats(self, stats: ThinkingStats):
        """Update the statistics display."""
        self.stats = stats
        self.refresh()
    
    def render(self) -> Text:
        """Render the stats bar."""
        theme = get_theme()
        text = Text()
        
        # Prompt tokens
        text.append("📝 Prompt: ", style=theme.fg_muted)
        text.append(f"{self.stats.prompt_tokens:,}", style=theme.info)
        text.append("  │  ", style=theme.border)
        
        # Completion tokens
        text.append("💬 Completion: ", style=theme.fg_muted)
        text.append(f"{self.stats.completion_tokens:,}", style=theme.success)
        text.append("  │  ", style=theme.border)
        
        # Total tokens
        text.append("📊 Total: ", style=theme.fg_muted)
        text.append(f"{self.stats.total_tokens:,}", style=theme.accent)
        text.append("  │  ", style=theme.border)
        
        # Time
        text.append("⏱️ ", style=theme.fg_muted)
        time_sec = self.stats.thinking_time_ms / 1000
        text.append(f"{time_sec:.2f}s", style=theme.warning)
        
        # Cost estimate
        if self.stats.estimated_cost > 0:
            text.append("  │  ", style=theme.border)
            text.append("💰 $", style=theme.fg_muted)
            text.append(f"{self.stats.estimated_cost:.4f}", style=theme.warning)
        
        return text


class StreamingThoughtDisplay(RichLog):
    """A streaming display for AI thoughts/responses.
    
    Shows content as it streams in, with special formatting for:
    - Code blocks
    - Markdown
    - JSON
    - Plain text
    """
    
    DEFAULT_CSS = """
    StreamingThoughtDisplay {
        height: 1fr;
        padding: 1;
        background: $surface-darken-1;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(auto_scroll=True, **kwargs)
        self._current_text = ""
        self._in_code_block = False
    
    def append_chunk(self, chunk: str):
        """Append a chunk of streaming text."""
        self._current_text += chunk
        # For streaming, we just write the chunk
        self.write(chunk, scroll_end=True)
    
    def set_content(self, content: str):
        """Set the full content (replacing current)."""
        self.clear()
        self._current_text = content
        self.write(content)
    
    def clear_content(self):
        """Clear all content."""
        self.clear()
        self._current_text = ""


# ======================== MAIN COMPONENT ========================


class AIThinkingViewer(Vertical):
    """Main AI Thinking Viewer component.
    
    Shows what the AI is thinking in real-time with:
    - Status header
    - Streaming thought display
    - Entry log
    - Token statistics
    
    Keybindings:
    - t: Toggle thinking view
    - c: Clear thinking log
    - p: Pause/resume auto-scroll
    """
    
    DEFAULT_CSS = """
    AIThinkingViewer {
        height: 100%;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
    }
    
    AIThinkingViewer.collapsed {
        height: 3;
    }
    
    AIThinkingViewer .thinking-header {
        height: 3;
        padding: 0 1;
        border-bottom: solid $primary-darken-3;
        background: $surface;
    }
    
    AIThinkingViewer .thinking-content {
        height: 1fr;
    }
    
    AIThinkingViewer .current-thought-section {
        height: 40%;
        border-bottom: dashed $primary-darken-3;
    }
    
    AIThinkingViewer .section-label {
        height: 1;
        padding: 0 1;
        background: $primary-darken-3;
        color: $text;
        text-style: bold;
    }
    
    AIThinkingViewer .thought-log-section {
        height: 60%;
    }
    
    AIThinkingViewer .thinking-footer {
        height: 2;
    }
    """
    
    visible = reactive(True)
    phase = reactive(ThinkingPhase.IDLE)
    auto_scroll = reactive(True)
    
    BINDINGS = [
        ("t", "toggle_view", "Toggle"),
        ("c", "clear_log", "Clear"),
        ("p", "toggle_scroll", "Scroll"),
    ]
    
    def __init__(
        self,
        title: str = "AI Thinking",
        show_token_stats: bool = True,
        max_entries: int = 500,
        **kwargs,
    ):
        """Initialize the AI thinking viewer.
        
        Args:
            title: Viewer title
            show_token_stats: Whether to show token statistics
            max_entries: Maximum log entries to keep
        """
        super().__init__(**kwargs)
        self.title = title
        self.show_token_stats = show_token_stats
        self.max_entries = max_entries
        self.entries: List[ThinkingEntry] = []
        self.stats = ThinkingStats()
        self._callbacks: Dict[str, List[Callable]] = {
            "on_phase_change": [],
            "on_entry_added": [],
            "on_cleared": [],
        }
    
    def compose(self):
        """Compose the viewer layout."""
        # Header with status
        yield ThinkingHeader(classes="thinking-header")
        
        # Main content area
        with Vertical(classes="thinking-content"):
            # Current streaming thought
            with Vertical(classes="current-thought-section"):
                yield Static(f"{ICON_THINKING} Current Thought", classes="section-label")
                yield StreamingThoughtDisplay(id="current-thought")
            
            # Thought history/log
            with Vertical(classes="thought-log-section"):
                yield Static("📜 Thinking History", classes="section-label")
                yield RichLog(auto_scroll=True, id="thought-log", classes="thought-log")
        
        # Footer with stats
        if self.show_token_stats:
            yield ThinkingTokenStats(stats=self.stats, classes="thinking-footer", id="token-stats")
    
    def on_mount(self):
        """Initialize on mount."""
        self._update_header()
    
    def _update_header(self):
        """Update the header with current state."""
        try:
            header = self.query_one(ThinkingHeader)
            header.phase = self.phase
            header.model = self.stats.model
            header.provider = self.stats.provider
        except Exception:
            pass
    
    def watch_phase(self, phase: ThinkingPhase):
        """React to phase changes."""
        self._update_header()
        for callback in self._callbacks.get("on_phase_change", []):
            callback(phase)
    
    def watch_visible(self, visible: bool):
        """React to visibility changes."""
        if visible:
            self.remove_class("collapsed")
        else:
            self.add_class("collapsed")
        self.post_message(ThinkingToggled(visible))
    
    # ==================== Public API ====================
    
    def set_phase(self, phase: ThinkingPhase):
        """Set the current thinking phase."""
        self.phase = phase
    
    def set_model(self, model: str, provider: str = ""):
        """Set the current model and provider."""
        self.stats.model = model
        self.stats.provider = provider
        self._update_header()
    
    def stream_thought(self, chunk: str):
        """Stream a chunk of thought content."""
        try:
            display = self.query_one("#current-thought", StreamingThoughtDisplay)
            display.append_chunk(chunk)
        except Exception:
            pass
    
    def set_current_thought(self, content: str):
        """Set the current thought content (replaces previous)."""
        try:
            display = self.query_one("#current-thought", StreamingThoughtDisplay)
            display.set_content(content)
        except Exception:
            pass
    
    def clear_current_thought(self):
        """Clear the current thought display."""
        try:
            display = self.query_one("#current-thought", StreamingThoughtDisplay)
            display.clear_content()
        except Exception:
            pass
    
    def add_entry(
        self,
        phase: ThinkingPhase,
        role: MessageRole,
        content: str,
        tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add an entry to the thinking log.
        
        Args:
            phase: Current thinking phase
            role: Message role
            content: Entry content
            tokens: Token count (if known)
            metadata: Additional metadata
        """
        entry = ThinkingEntry(
            timestamp=datetime.now(),
            phase=phase,
            role=role,
            content=content,
            tokens=tokens,
            metadata=metadata or {},
        )
        
        self.entries.append(entry)
        
        # Trim if needed
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Render to log
        theme = get_theme()
        try:
            log = self.query_one("#thought-log", RichLog)
            log.write(entry.render(theme))
        except Exception:
            pass
        
        for callback in self._callbacks.get("on_entry_added", []):
            callback(entry)
    
    def log_prompt(self, prompt: str, tokens: int = 0):
        """Log a prompt being sent to the AI."""
        self.add_entry(
            phase=ThinkingPhase.SENDING,
            role=MessageRole.USER,
            content=prompt,
            tokens=tokens,
        )
    
    def log_response(self, response: str, tokens: int = 0):
        """Log a response from the AI."""
        self.add_entry(
            phase=ThinkingPhase.COMPLETE,
            role=MessageRole.ASSISTANT,
            content=response,
            tokens=tokens,
        )
    
    def log_thought(self, thought: str):
        """Log an internal AI thought/reasoning."""
        self.add_entry(
            phase=ThinkingPhase.THINKING,
            role=MessageRole.THOUGHT,
            content=thought,
        )
    
    def log_error(self, error: str):
        """Log an error."""
        self.add_entry(
            phase=ThinkingPhase.ERROR,
            role=MessageRole.ERROR,
            content=error,
        )
    
    def update_stats(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_time_ms: float = 0.0,
        estimated_cost: float = 0.0,
    ):
        """Update token/cost statistics."""
        self.stats.prompt_tokens += prompt_tokens
        self.stats.completion_tokens += completion_tokens
        self.stats.total_tokens = self.stats.prompt_tokens + self.stats.completion_tokens
        self.stats.thinking_time_ms += thinking_time_ms
        self.stats.estimated_cost += estimated_cost
        self.stats.requests += 1
        
        try:
            stats_widget = self.query_one("#token-stats", ThinkingTokenStats)
            stats_widget.update_stats(self.stats)
        except Exception:
            pass
    
    def clear(self):
        """Clear all entries and stats."""
        self.entries.clear()
        self.stats = ThinkingStats(model=self.stats.model, provider=self.stats.provider)
        
        try:
            self.query_one("#current-thought", StreamingThoughtDisplay).clear_content()
            self.query_one("#thought-log", RichLog).clear()
            self.query_one("#token-stats", ThinkingTokenStats).update_stats(self.stats)
        except Exception:
            pass
        
        self.post_message(ThinkingCleared())
        for callback in self._callbacks.get("on_cleared", []):
            callback()
    
    def on_callback(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    # ==================== Actions ====================
    
    def action_toggle_view(self):
        """Toggle visibility."""
        self.visible = not self.visible
    
    def action_clear_log(self):
        """Clear the thinking log."""
        self.clear()
    
    def action_toggle_scroll(self):
        """Toggle auto-scroll."""
        self.auto_scroll = not self.auto_scroll
        try:
            log = self.query_one("#thought-log", RichLog)
            log.auto_scroll = self.auto_scroll
            display = self.query_one("#current-thought", StreamingThoughtDisplay)
            display.auto_scroll = self.auto_scroll
        except Exception:
            pass
    
    @on(Button.Pressed, "#clear-btn")
    def handle_clear_button(self, event: Button.Pressed):
        """Handle clear button press."""
        self.clear()
    
    @on(Button.Pressed, "#pause-btn")
    def handle_pause_button(self, event: Button.Pressed):
        """Handle pause button press."""
        self.action_toggle_scroll()


# ======================== CONVENIENCE FUNCTIONS ========================


def create_thinking_panel(
    title: str = "AI Thinking",
    collapsible: bool = True,
    show_stats: bool = True,
) -> AIThinkingViewer:
    """Create a pre-configured AI thinking panel.
    
    Args:
        title: Panel title
        collapsible: Whether the panel can be collapsed
        show_stats: Whether to show token statistics
    
    Returns:
        Configured AIThinkingViewer instance
    """
    return AIThinkingViewer(
        title=title,
        show_token_stats=show_stats,
    )
