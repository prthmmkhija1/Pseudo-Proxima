"""AI Assistant Screen for Proxima TUI.

Full-featured AI chat interface with:
- Persistent conversation memory
- Import/export chat functionality
- Keyboard shortcuts
- Real-time stats tracking
- Multi-line input support
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Button, Input, RichLog, TextArea, Label, Select
from textual.binding import Binding
from textual.screen import ModalScreen
from textual import events
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown

from .base import BaseScreen
from ..styles.theme import get_theme
from textual.message import Message

# Import LLM components
try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class SendableTextArea(TextArea):
    """Custom TextArea that sends a message on Ctrl+Enter."""
    
    class SendRequested(Message):
        """Posted when user presses Ctrl+Enter."""
        pass
    
    async def _on_key(self, event: events.Key) -> None:
        """Intercept Ctrl+Enter before parent handles it."""
        # Ctrl+Enter to send - ctrl+m is the terminal code for Ctrl+Enter
        if event.key in ("ctrl+m", "ctrl+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.post_message(self.SendRequested())
            return
        # Let parent TextArea handle everything else
        await super()._on_key(event)


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tokens: int = 0
    thinking_time_ms: int = 0


@dataclass
class ChatSession:
    """A complete chat session with metadata."""
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    name: str = "New Chat"
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_tokens: int = 0
    total_requests: int = 0
    provider: str = ""
    model: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'messages': [asdict(m) for m in self.messages],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'total_tokens': self.total_tokens,
            'total_requests': self.total_requests,
            'provider': self.provider,
            'model': self.model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        """Create from dictionary."""
        session = cls(
            id=data.get('id', datetime.now().strftime("%Y%m%d_%H%M%S")),
            name=data.get('name', 'Imported Chat'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            total_tokens=data.get('total_tokens', 0),
            total_requests=data.get('total_requests', 0),
            provider=data.get('provider', ''),
            model=data.get('model', ''),
        )
        session.messages = [
            ChatMessage(**m) for m in data.get('messages', [])
        ]
        return session


class ExportNameDialog(ModalScreen[str]):
    """Dialog for entering custom export filename."""
    
    DEFAULT_CSS = """
    ExportNameDialog {
        align: center middle;
    }
    
    ExportNameDialog > Vertical {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    
    ExportNameDialog Label {
        margin-bottom: 1;
    }
    
    ExportNameDialog Input {
        margin-bottom: 1;
    }
    
    ExportNameDialog Horizontal {
        height: auto;
        align: center middle;
    }
    
    ExportNameDialog Button {
        margin: 0 1;
    }
    """
    
    def __init__(self, default_name: str = "") -> None:
        super().__init__()
        self._default_name = default_name
    
    def compose(self):
        with Vertical():
            yield Label("Enter a name for the exported chat:")
            yield Input(value=self._default_name, id="export-name-input", placeholder="e.g., quantum_algorithms_discussion")
            with Horizontal():
                yield Button("Export", variant="primary", id="export-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-btn":
            name_input = self.query_one("#export-name-input", Input)
            self.dismiss(name_input.value.strip())
        else:
            self.dismiss("")
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input."""
        self.dismiss(event.value.strip())


class ImportSelectDialog(ModalScreen[str]):
    """Dialog for selecting which chat to import."""
    
    DEFAULT_CSS = """
    ImportSelectDialog {
        align: center middle;
    }
    
    ImportSelectDialog > Vertical {
        width: 80;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    
    ImportSelectDialog Label {
        margin-bottom: 1;
    }
    
    ImportSelectDialog Select {
        margin-bottom: 1;
        width: 100%;
    }
    
    ImportSelectDialog Horizontal {
        height: auto;
        align: center middle;
    }
    
    ImportSelectDialog Button {
        margin: 0 1;
    }
    """
    
    def __init__(self, chat_files: List[tuple]) -> None:
        """Initialize with list of (display_name, file_path) tuples."""
        super().__init__()
        self._chat_files = chat_files
    
    def compose(self):
        with Vertical():
            yield Label("Select a chat to import:")
            options = [(name, path) for name, path in self._chat_files]
            yield Select(options, id="chat-select", prompt="Choose a chat...")
            with Horizontal():
                yield Button("Import", variant="primary", id="import-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "import-btn":
            select = self.query_one("#chat-select", Select)
            if select.value and select.value != Select.BLANK:
                self.dismiss(str(select.value))
            else:
                self.dismiss("")
        else:
            self.dismiss("")


class AIAssistantScreen(BaseScreen):
    """Full-featured AI Assistant screen.
    
    Features:
    - Persistent chat history (survives tab switches)
    - Multi-line input (Enter for new line, Ctrl+Enter to send)
    - Import/export conversations with custom names
    - Real-time statistics
    - Keyboard shortcuts
    """
    
    SCREEN_NAME = "ai_assistant"
    SCREEN_TITLE = "AI Assistant"
    SHOW_SIDEBAR = False  # Full screen mode
    
    BINDINGS = [
        Binding("ctrl+m", "send_on_enter", "Send", show=False, priority=True),
        Binding("ctrl+z", "undo", "Undo", show=True),
        Binding("ctrl+y", "redo", "Redo", show=True),
        Binding("ctrl+a", "select_all", "Select All", show=False),
        Binding("ctrl+x", "cut", "Cut", show=False),
        Binding("ctrl+c", "copy_or_cancel", "Copy/Cancel", show=False),
        Binding("ctrl+j", "prev_prompt", "Previous Prompt", show=True),
        Binding("ctrl+l", "next_prompt", "Next Prompt", show=True),
        Binding("ctrl+n", "new_chat", "New Chat", show=True),
        Binding("ctrl+s", "export_chat", "Export Chat", show=True),
        Binding("ctrl+o", "import_chat", "Import Chat", show=True),
        Binding("ctrl+shift+c", "clear_chat", "Clear Chat", show=True),
        Binding("escape", "go_back", "Back"),
        Binding("f1", "show_shortcuts", "Show Shortcuts"),
        # Navigation
        ("1", "goto_dashboard", "Dashboard"),
        ("2", "goto_execution", "Execution"),
        ("3", "goto_results", "Results"),
        ("4", "goto_backends", "Backends"),
        ("5", "goto_settings", "Settings"),
        ("6", "goto_ai_assistant", "AI Assistant"),
    ]
    
    DEFAULT_CSS = """
    AIAssistantScreen {
        layout: horizontal;
    }
    
    AIAssistantScreen .main-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }
    
    AIAssistantScreen .chat-area {
        width: 80%;
        height: 100%;
        border-right: solid $primary;
    }
    
    AIAssistantScreen .sidebar-panel {
        width: 20%;
        height: 100%;
        background: $surface-darken-1;
    }
    
    AIAssistantScreen .header-section {
        height: 5;
        padding: 1;
        background: $primary-darken-2;
        border-bottom: solid $primary;
    }
    
    AIAssistantScreen .header-title {
        text-style: bold;
        color: $accent;
        text-align: center;
    }
    
    AIAssistantScreen .header-subtitle {
        color: $text-muted;
        text-align: center;
    }
    
    AIAssistantScreen .chat-log-container {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    AIAssistantScreen .chat-log {
        height: 100%;
        background: $surface-darken-2;
        padding: 1;
        border: solid $primary-darken-3;
    }
    
    AIAssistantScreen .input-section {
        height: auto;
        min-height: 8;
        max-height: 15;
        padding: 1;
        border-top: solid $primary;
        background: $surface;
    }
    
    AIAssistantScreen .input-container {
        height: auto;
        min-height: 3;
        layout: horizontal;
    }
    
    AIAssistantScreen .prompt-input {
        width: 1fr;
        min-height: 3;
        max-height: 10;
        margin-right: 1;
    }
    
    AIAssistantScreen .send-btn {
        width: 12;
        height: 3;
    }
    
    AIAssistantScreen .controls-row {
        height: 3;
        layout: horizontal;
        margin-top: 1;
    }
    
    AIAssistantScreen .control-btn {
        margin-right: 1;
        min-width: 12;
        height: 3;
    }
    
    AIAssistantScreen .input-hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    
    /* Sidebar styles */
    AIAssistantScreen .sidebar-section {
        height: auto;
        padding: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    AIAssistantScreen .sidebar-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    AIAssistantScreen .stats-grid {
        height: auto;
    }
    
    AIAssistantScreen .stat-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 0;
    }
    
    AIAssistantScreen .stat-label {
        width: 12;
        color: $text-muted;
    }
    
    AIAssistantScreen .stat-value {
        width: 1fr;
        color: $text;
        text-align: right;
    }
    
    AIAssistantScreen .stat-highlight {
        color: $accent;
    }
    
    AIAssistantScreen .history-list {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    AIAssistantScreen .history-item {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        background: $surface-darken-2;
        border: solid $primary-darken-3;
    }
    
    AIAssistantScreen .history-item:hover {
        background: $primary-darken-2;
        border: solid $primary;
    }
    
    AIAssistantScreen .shortcuts-panel {
        height: auto;
        padding: 1;
    }
    
    AIAssistantScreen .shortcut-row {
        height: auto;
        layout: horizontal;
    }
    
    AIAssistantScreen .shortcut-key {
        width: 10;
        color: $accent;
    }
    
    AIAssistantScreen .shortcut-desc {
        width: 1fr;
        color: $text-muted;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the AI Assistant screen."""
        super().__init__(**kwargs)
        
        # Chat state - will be restored from TUIState if available
        self._current_session: ChatSession = ChatSession()
        self._sessions: List[ChatSession] = []
        self._prompt_history: List[str] = []
        self._prompt_history_index: int = -1
        self._undo_stack: List[str] = []
        self._redo_stack: List[str] = []
        
        # LLM state
        self._llm_router: Optional[LLMRouter] = None
        self._llm_provider: str = 'none'
        self._llm_model: str = ''
        self._is_generating: bool = False
        
        # Stats
        self._stats = {
            'total_messages': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_requests': 0,
            'avg_response_time': 0,
            'session_start': time.time(),
        }
        
        # Initialize LLM router
        if LLM_AVAILABLE:
            try:
                def auto_consent(prompt: str) -> bool:
                    return True
                self._llm_router = LLMRouter(consent_prompt=auto_consent)
            except Exception:
                pass
        
        # Load settings and history
        self._load_settings()
        self._load_chat_history()
        
        # Restore chat state from TUIState (for persistence across tab switches)
        self._restore_from_state()
    
    def _load_settings(self) -> None:
        """Load LLM settings from config."""
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                
                llm = settings.get('llm', {})
                self._llm_provider = llm.get('mode', 'none')
                
                # Get model based on provider
                model_keys = {
                    'local': 'local_model', 'ollama': 'local_model',
                    'openai': 'openai_model', 'anthropic': 'anthropic_model',
                    'google': 'google_model', 'xai': 'xai_model',
                    'deepseek': 'deepseek_model', 'mistral': 'mistral_model',
                    'groq': 'groq_model', 'together': 'together_model',
                    'perplexity': 'perplexity_model', 'fireworks': 'fireworks_model',
                    'cohere': 'cohere_model', 'openrouter': 'openrouter_model',
                    'huggingface': 'huggingface_model', 'replicate': 'replicate_model',
                }
                self._llm_model = llm.get(model_keys.get(self._llm_provider, ''), '')
                
                # Get and register API key with the router
                api_key_map = {
                    'openai': 'openai_key', 'anthropic': 'anthropic_key',
                    'google': 'google_key', 'gemini': 'google_key',
                    'xai': 'xai_key', 'grok': 'xai_key',
                    'deepseek': 'deepseek_key', 'mistral': 'mistral_key',
                    'groq': 'groq_key', 'together': 'together_key',
                    'cohere': 'cohere_key', 'perplexity': 'perplexity_key',
                    'fireworks': 'fireworks_key', 'huggingface': 'huggingface_key',
                    'replicate': 'replicate_key', 'openrouter': 'openrouter_key',
                    'azure_openai': 'azure_openai_key', 'azure': 'azure_openai_key',
                }
                api_key_field = api_key_map.get(self._llm_provider)
                api_key = llm.get(api_key_field, '') if api_key_field else ''
                
                # Map provider to router name for API key registration
                provider_map = {
                    'local': 'ollama', 'ollama': 'ollama',
                    'openai': 'openai', 'anthropic': 'anthropic',
                    'google': 'google', 'gemini': 'google',
                    'xai': 'xai', 'grok': 'xai',
                    'deepseek': 'deepseek', 'mistral': 'mistral',
                    'groq': 'groq', 'together': 'together',
                    'openrouter': 'openrouter', 'cohere': 'cohere',
                    'lmstudio': 'lmstudio', 'llamacpp': 'llama_cpp',
                    'perplexity': 'perplexity', 'fireworks': 'fireworks',
                    'huggingface': 'huggingface', 'replicate': 'replicate',
                    'azure_openai': 'azure_openai', 'azure': 'azure_openai',
                }
                router_provider = provider_map.get(self._llm_provider, self._llm_provider)
                
                # Register API key with the router for proper authentication
                if api_key and self._llm_router and router_provider:
                    try:
                        self._llm_router.api_keys.store_key(router_provider, api_key)
                    except Exception:
                        pass  # Silently ignore if API key storage fails
                
                # Update session info
                self._current_session.provider = self._llm_provider
                self._current_session.model = self._llm_model
        except Exception:
            pass
    
    def _restore_from_state(self) -> None:
        """Restore chat state from TUIState for persistence across tab switches."""
        try:
            if hasattr(self, 'state') and self.state:
                # Restore messages
                if self.state.ai_chat_messages:
                    self._current_session.messages = [
                        ChatMessage(**m) for m in self.state.ai_chat_messages
                    ]
                
                # Restore session info
                if self.state.ai_chat_session_id:
                    self._current_session.id = self.state.ai_chat_session_id
                if self.state.ai_chat_session_name:
                    self._current_session.name = self.state.ai_chat_session_name
                
                # Restore stats
                if self.state.ai_chat_stats:
                    self._stats.update(self.state.ai_chat_stats)
                    self._current_session.total_tokens = self._stats.get('total_tokens', 0)
                    self._current_session.total_requests = self._stats.get('total_requests', 0)
        except Exception:
            pass
    
    def _save_to_state(self) -> None:
        """Save chat state to TUIState for persistence across tab switches."""
        try:
            if hasattr(self, 'state') and self.state:
                # Save messages
                self.state.ai_chat_messages = [asdict(m) for m in self._current_session.messages]
                
                # Save session info
                self.state.ai_chat_session_id = self._current_session.id
                self.state.ai_chat_session_name = self._current_session.name
                
                # Save stats
                self.state.ai_chat_stats = {
                    'total_messages': len(self._current_session.messages),
                    'total_tokens': self._current_session.total_tokens,
                    'total_requests': self._current_session.total_requests,
                    'avg_response_time': self._stats.get('avg_response_time', 0),
                    'session_start': self._stats.get('session_start', time.time()),
                }
        except Exception:
            pass
    
    def _load_chat_history(self) -> None:
        """Load chat history from disk."""
        try:
            history_dir = Path.home() / ".proxima" / "chat_history"
            if not history_dir.exists():
                return
            
            for file_path in sorted(history_dir.glob("*.json"), reverse=True)[:10]:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    session = ChatSession.from_dict(data)
                    self._sessions.append(session)
                except Exception:
                    continue
        except Exception:
            pass
    
    def _save_current_session(self) -> None:
        """Save current session to disk and TUIState."""
        try:
            history_dir = Path.home() / ".proxima" / "chat_history"
            history_dir.mkdir(parents=True, exist_ok=True)
            
            self._current_session.updated_at = datetime.now().isoformat()
            file_path = history_dir / f"chat_{self._current_session.id}.json"
            
            with open(file_path, 'w') as f:
                json.dump(self._current_session.to_dict(), f, indent=2)
            
            # Also save to TUIState for persistence
            self._save_to_state()
        except Exception:
            pass
    
    def on_mount(self) -> None:
        """Called when screen is mounted."""
        self._update_stats_display()
        
        # Restore previous chat messages if any
        if self._current_session.messages:
            self._restore_chat_display()
        else:
            self._show_welcome_message()
        
        # Focus the input
        self.set_timer(0.1, self._focus_input)
    
    def on_unmount(self) -> None:
        """Called when screen is unmounted - save state for persistence."""
        self._save_to_state()
        self._save_current_session()
    
    def _restore_chat_display(self) -> None:
        """Restore chat messages to display after returning to screen."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            # Show welcome header
            welcome = Text()
            welcome.append("🤖 AI Assistant\n", style=f"bold {theme.accent}")
            welcome.append("━" * 50 + "\n", style=theme.border)
            welcome.append("(Chat restored from previous session)\n\n", style=theme.fg_muted)
            chat_log.write(welcome)
            
            # Replay all messages
            for msg in self._current_session.messages:
                text = Text()
                if msg.role == 'user':
                    text.append("\n👤 You\n", style=f"bold {theme.primary}")
                else:
                    text.append("\n🤖 AI\n", style=f"bold {theme.accent}")
                text.append(msg.content + "\n", style=theme.fg_base)
                chat_log.write(text)
        except Exception:
            self._show_welcome_message()
    
    def _focus_input(self) -> None:
        """Focus the prompt input."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            input_widget.focus()
        except Exception:
            pass
    
    def _show_welcome_message(self) -> None:
        """Show welcome message in chat log."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            welcome = Text()
            welcome.append("🤖 AI Assistant\n", style=f"bold {theme.accent}")
            welcome.append("━" * 50 + "\n\n", style=theme.border)
            
            welcome.append("Welcome! I'm your AI assistant for Proxima.\n\n", style=theme.fg_base)
            welcome.append("I can help you with:\n", style=theme.fg_muted)
            welcome.append("• Quantum circuit design and optimization\n", style=theme.fg_base)
            welcome.append("• Backend configuration and selection\n", style=theme.fg_base)
            welcome.append("• Performance analysis and troubleshooting\n", style=theme.fg_base)
            welcome.append("• Understanding quantum computing concepts\n\n", style=theme.fg_base)
            
            if self._llm_provider and self._llm_provider != 'none':
                welcome.append(f"Provider: ", style=theme.fg_muted)
                welcome.append(f"{self._llm_provider}\n", style=theme.accent)
                if self._llm_model:
                    welcome.append(f"Model: ", style=theme.fg_muted)
                    welcome.append(f"{self._llm_model}\n", style=theme.fg_base)
            else:
                welcome.append("⚠️ No LLM provider configured. Using simulation mode.\n", style=theme.warning)
                welcome.append("Configure in Settings → AI Settings.\n", style=theme.fg_muted)
            
            welcome.append("\n" + "━" * 50 + "\n", style=theme.border)
            welcome.append("Press F1 for keyboard shortcuts\n", style=theme.fg_subtle)
            
            chat_log.write(welcome)
        except Exception:
            pass
    
    def compose_main(self):
        """Compose the main content."""
        with Horizontal(classes="main-container"):
            # Chat area (left side)
            with Vertical(classes="chat-area"):
                # Header
                with Container(classes="header-section"):
                    yield Static("🤖 Proxima AI Assistant", classes="header-title")
                    yield Static("Your quantum computing copilot", classes="header-subtitle")
                
                # Chat log
                with ScrollableContainer(classes="chat-log-container"):
                    yield RichLog(
                        auto_scroll=True,
                        classes="chat-log",
                        id="chat-log",
                        markup=True,
                        highlight=True,
                    )
                
                # Input section
                with Vertical(classes="input-section"):
                    with Horizontal(classes="input-container"):
                        yield SendableTextArea(
                            id="prompt-input",
                            classes="prompt-input",
                        )
                        yield Button(
                            "Send",
                            id="btn-send",
                            classes="send-btn",
                            variant="primary",
                        )
                    
                    with Horizontal(classes="controls-row"):
                        yield Button("🛑 Stop", id="btn-stop", classes="control-btn", variant="error", disabled=True)
                        yield Button("🗑️ Clear", id="btn-clear", classes="control-btn")
                        yield Button("📥 Import", id="btn-import", classes="control-btn")
                        yield Button("📤 Export", id="btn-export", classes="control-btn")
                        yield Button("🆕 New Chat", id="btn-new", classes="control-btn", variant="success")
                    
                    yield Static(
                        "Enter = New Line | Ctrl+Enter = Send | Ctrl+J/L = History | F1 = Shortcuts",
                        classes="input-hint"
                    )
            
            # Sidebar (right side)
            with Vertical(classes="sidebar-panel"):
                # Stats section
                with Container(classes="sidebar-section"):
                    yield Static("📊 Session Statistics", classes="sidebar-title")
                    with Vertical(classes="stats-grid"):
                        with Horizontal(classes="stat-row"):
                            yield Static("Provider:", classes="stat-label")
                            yield Static("—", classes="stat-value", id="stat-provider")
                        with Horizontal(classes="stat-row"):
                            yield Static("Model:", classes="stat-label")
                            yield Static("—", classes="stat-value", id="stat-model")
                        with Horizontal(classes="stat-row"):
                            yield Static("Messages:", classes="stat-label")
                            yield Static("0", classes="stat-value stat-highlight", id="stat-messages")
                        with Horizontal(classes="stat-row"):
                            yield Static("Tokens:", classes="stat-label")
                            yield Static("0", classes="stat-value", id="stat-tokens")
                        with Horizontal(classes="stat-row"):
                            yield Static("Requests:", classes="stat-label")
                            yield Static("0", classes="stat-value", id="stat-requests")
                        with Horizontal(classes="stat-row"):
                            yield Static("Avg Time:", classes="stat-label")
                            yield Static("0ms", classes="stat-value", id="stat-avg-time")
                        with Horizontal(classes="stat-row"):
                            yield Static("Session:", classes="stat-label")
                            yield Static("0m", classes="stat-value", id="stat-session-time")
                
                # Shortcuts section
                with Container(classes="sidebar-section shortcuts-panel"):
                    yield Static("⌨️ Keyboard Shortcuts", classes="sidebar-title")
                    with Vertical(classes="shortcuts-list"):
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Enter", classes="shortcut-key")
                            yield Static("New line", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+↵", classes="shortcut-key")
                            yield Static("Send message", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+J", classes="shortcut-key")
                            yield Static("Previous prompt", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+L", classes="shortcut-key")
                            yield Static("Next prompt", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+N", classes="shortcut-key")
                            yield Static("New chat", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+S", classes="shortcut-key")
                            yield Static("Export chat", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+O", classes="shortcut-key")
                            yield Static("Import chat", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+Z", classes="shortcut-key")
                            yield Static("Undo", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+Y", classes="shortcut-key")
                            yield Static("Redo", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Esc", classes="shortcut-key")
                            yield Static("Go back", classes="shortcut-desc")
                
                # Recent chats section
                with Container(classes="sidebar-section"):
                    yield Static("📜 Recent Chats", classes="sidebar-title")
                
                with ScrollableContainer(classes="history-list", id="history-list"):
                    pass  # Will be populated dynamically
    
    def _update_stats_display(self) -> None:
        """Update the statistics display."""
        try:
            # Provider
            provider = self._llm_provider or 'none'
            provider_names = {
                'none': 'Not configured',
                'local': 'Ollama (Local)',
                'ollama': 'Ollama (Local)',
                'openai': 'OpenAI',
                'anthropic': 'Anthropic',
                'google': 'Google AI',
                'xai': 'xAI (Grok)',
                'deepseek': 'DeepSeek',
                'mistral': 'Mistral AI',
                'groq': 'Groq',
                'together': 'Together AI',
            }
            self.query_one("#stat-provider", Static).update(
                provider_names.get(provider, provider.title())
            )
            
            # Model
            model = self._llm_model or '—'
            if len(model) > 20:
                model = model[:17] + "..."
            self.query_one("#stat-model", Static).update(model)
            
            # Messages
            self.query_one("#stat-messages", Static).update(
                str(len(self._current_session.messages))
            )
            
            # Tokens
            self.query_one("#stat-tokens", Static).update(
                str(self._current_session.total_tokens)
            )
            
            # Requests
            self.query_one("#stat-requests", Static).update(
                str(self._current_session.total_requests)
            )
            
            # Average time
            if self._stats['total_requests'] > 0:
                avg_time = self._stats['avg_response_time'] / self._stats['total_requests']
                self.query_one("#stat-avg-time", Static).update(f"{avg_time:.0f}ms")
            
            # Session time
            elapsed = int(time.time() - self._stats['session_start'])
            minutes = elapsed // 60
            self.query_one("#stat-session-time", Static).update(f"{minutes}m")
            
        except Exception:
            pass
    
    def _update_history_list(self) -> None:
        """Update the recent chats list."""
        try:
            container = self.query_one("#history-list", ScrollableContainer)
            
            # Clear existing
            for child in list(container.children):
                child.remove()
            
            # Add recent sessions
            for session in self._sessions[:5]:
                name = session.name[:20] + "..." if len(session.name) > 20 else session.name
                msg_count = len(session.messages)
                
                item_text = f"{name}\n{msg_count} messages"
                container.mount(Static(item_text, classes="history-item"))
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id
        
        if btn_id == "btn-send":
            self._send_message()
        elif btn_id == "btn-stop":
            self._stop_generation()
        elif btn_id == "btn-clear":
            self.action_clear_chat()
        elif btn_id == "btn-import":
            self.action_import_chat()
        elif btn_id == "btn-export":
            self.action_export_chat()
        elif btn_id == "btn-new":
            self.action_new_chat()
    
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes for undo tracking."""
        if event.text_area.id == "prompt-input":
            # Track for undo
            current = event.text_area.text
            if self._undo_stack and self._undo_stack[-1] == current:
                return
            self._undo_stack.append(current)
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
    
    def action_send_on_enter(self) -> None:
        """Send message when Ctrl+Enter is pressed (binding action)."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            if input_widget.has_focus:
                self._send_message()
        except Exception:
            pass
    
    def on_sendable_text_area_send_requested(self, event: SendableTextArea.SendRequested) -> None:
        """Handle Ctrl+Enter from the custom TextArea."""
        self._send_message()
    
    def _send_message(self) -> None:
        """Send the current message."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            message = input_widget.text.strip()
            
            if not message:
                return
            
            # Clear input
            input_widget.text = ""
            
            # Add to history
            if message not in self._prompt_history:
                self._prompt_history.append(message)
            self._prompt_history_index = -1
            
            # Show user message
            self._show_user_message(message)
            
            # Add to session
            self._current_session.messages.append(
                ChatMessage(role='user', content=message)
            )
            
            # Generate response
            self._is_generating = True
            self.query_one("#btn-stop", Button).disabled = False
            self.query_one("#btn-send", Button).disabled = True
            
            self._generate_response(message)
            
        except Exception as e:
            self._show_error(str(e))
    
    def _show_user_message(self, message: str) -> None:
        """Display user message in chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            text = Text()
            text.append("\n👤 You\n", style=f"bold {theme.primary}")
            text.append(message + "\n", style=theme.fg_base)
            
            chat_log.write(text)
        except Exception:
            pass
    
    def _show_ai_message(self, message: str) -> None:
        """Display AI message in chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            text = Text()
            text.append("\n🤖 AI\n", style=f"bold {theme.accent}")
            text.append(message + "\n", style=theme.fg_base)
            
            chat_log.write(text)
        except Exception:
            pass
    
    def _show_error(self, error: str) -> None:
        """Display error in chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            text = Text()
            text.append("\n❌ Error: ", style=f"bold {theme.error}")
            text.append(error + "\n", style=theme.fg_muted)
            
            chat_log.write(text)
        except Exception:
            pass
    
    def _generate_response(self, message: str) -> None:
        """Generate AI response."""
        start_time = time.time()
        
        # Provider mapping - map TUI names to LLM router provider names
        # All major providers are now supported
        provider_map = {
            'local': 'ollama', 'ollama': 'ollama',
            'openai': 'openai', 'anthropic': 'anthropic',
            'google': 'google', 'gemini': 'google',  # Google Gemini
            'xai': 'xai', 'grok': 'xai',              # xAI Grok
            'deepseek': 'deepseek', 'mistral': 'mistral',
            'groq': 'groq', 'together': 'together',
            'openrouter': 'openrouter', 'cohere': 'cohere',
            'lmstudio': 'lmstudio', 'llamacpp': 'llama_cpp',
            'perplexity': 'perplexity', 'fireworks': 'fireworks',
            'huggingface': 'huggingface', 'replicate': 'replicate',
            'azure_openai': 'azure_openai', 'azure': 'azure_openai',
        }
        # Only a few providers still need simulation (no direct API support)
        simulation_providers = {'vertex_ai', 'aws_bedrock', 'ai21', 'deepinfra'}
        
        # Use simulation for unsupported providers
        if self._llm_provider in simulation_providers:
            router_provider = None  # Will trigger simulation
        else:
            router_provider = provider_map.get(self._llm_provider, self._llm_provider)
        
        llm_success = False
        
        if router_provider and router_provider != 'none' and self._llm_router and LLM_AVAILABLE:
            try:
                # Build context from recent messages
                context = "\n".join([
                    f"{'User' if m.role == 'user' else 'AI'}: {m.content}"
                    for m in self._current_session.messages[-5:]
                ])
                
                request = LLMRequest(
                    prompt=f"{message}",
                    system_prompt=f"""You are a helpful AI assistant for the Proxima quantum computing platform.
You help users with quantum circuit design, backend configuration, performance optimization, and understanding quantum computing concepts.
Be concise, accurate, and helpful. Format code with proper syntax.

Recent conversation context:
{context}""",
                    temperature=0.7,
                    max_tokens=1024,
                    provider=router_provider,
                    model=self._llm_model if self._llm_model else None,
                )
                
                response = self._llm_router.route(request)
                
                # Check for valid response
                response_text = ""
                if response:
                    if hasattr(response, 'text') and response.text:
                        response_text = response.text.strip()
                    
                    # Check for error
                    if hasattr(response, 'error') and response.error:
                        # Show error but continue with simulation
                        self._show_error(f"LLM Error: {response.error}")
                
                if response_text:
                    self._show_ai_message(response_text)
                    
                    # Update stats
                    elapsed = int((time.time() - start_time) * 1000)
                    tokens = getattr(response, 'tokens_used', 0) or getattr(response, 'total_tokens', 0) or 0
                    
                    self._current_session.messages.append(
                        ChatMessage(
                            role='assistant',
                            content=response_text,
                            tokens=tokens,
                            thinking_time_ms=elapsed
                        )
                    )
                    self._current_session.total_tokens += tokens
                    self._current_session.total_requests += 1
                    self._stats['total_requests'] += 1
                    self._stats['avg_response_time'] += elapsed
                    
                    self._update_stats_display()
                    self._save_current_session()
                    
                    self._finish_generation()
                    llm_success = True
                    
            except Exception as e:
                # Show what went wrong
                error_msg = str(e)
                if '404' not in error_msg and 'connection' not in error_msg.lower():
                    self._show_error(f"LLM Error: {error_msg}")
        
        if not llm_success:
            # Fallback to intelligent simulation
            self._simulate_response(message, start_time)
    
    def _simulate_response(self, message: str, start_time: float) -> None:
        """Generate simulated response."""
        msg_lower = message.lower()
        
        if any(word in msg_lower for word in ['hello', 'hi', 'hey', 'greet']):
            response = "Hello! I'm your AI assistant for Proxima. I can help you with quantum computing tasks, backend configuration, and understanding quantum concepts. What would you like to know?"
        
        elif any(word in msg_lower for word in ['backend', 'configure', 'setup']):
            response = """For backend configuration, here are the key steps:

1. **Choose a backend type:**
   - CPU Simulator: Good for small circuits (<20 qubits)
   - GPU Accelerated: Required for large circuits (>20 qubits)

2. **Configure settings:**
   - Set max qubits based on your hardware
   - Enable GPU if available (CUDA required)
   - Choose precision (float32 for speed, float64 for accuracy)

3. **Test the connection:**
   - Use the "Test Connection" button
   - Check logs for any errors

Would you like more details on any specific aspect?"""
        
        elif any(word in msg_lower for word in ['circuit', 'quantum', 'gate']):
            response = """Quantum circuits in Proxima can be created using various gates:

**Single-qubit gates:**
- H (Hadamard): Creates superposition
- X, Y, Z (Pauli): Bit/phase flips
- T, S: Phase gates

**Two-qubit gates:**
- CNOT: Controlled-NOT for entanglement
- CZ: Controlled-Z
- SWAP: Swaps qubit states

**Example Bell State:**
```
H(0)  # Create superposition
CNOT(0, 1)  # Entangle qubits
```

What specific circuit would you like to create?"""
        
        elif any(word in msg_lower for word in ['help', 'what', 'how', 'can you']):
            response = """I can help you with:

• **Circuit Design**: Creating and optimizing quantum circuits
• **Backend Selection**: Choosing the right simulator for your task
• **Performance**: Optimizing execution speed and memory usage
• **Troubleshooting**: Diagnosing common issues
• **Concepts**: Explaining quantum computing fundamentals

Just ask your question and I'll do my best to help!"""
        
        else:
            response = """I understand you're asking about quantum computing. Here are some things I can help with:

• Quantum circuit design and optimization
• Backend configuration (CPU, GPU, cloud)
• Performance tuning and benchmarking
• Understanding quantum algorithms

Could you provide more details about what you'd like to accomplish?"""
        
        # Show response
        self._show_ai_message(response)
        
        # Update stats
        elapsed = int((time.time() - start_time) * 1000)
        
        self._current_session.messages.append(
            ChatMessage(
                role='assistant',
                content=response,
                thinking_time_ms=elapsed
            )
        )
        self._current_session.total_requests += 1
        self._stats['total_requests'] += 1
        self._stats['avg_response_time'] += elapsed
        
        self._update_stats_display()
        self._save_current_session()
        self._finish_generation()
    
    def _finish_generation(self) -> None:
        """Finish generation and reset UI."""
        self._is_generating = False
        try:
            self.query_one("#btn-stop", Button).disabled = True
            self.query_one("#btn-send", Button).disabled = False
        except Exception:
            pass
    
    def _stop_generation(self) -> None:
        """Stop current generation."""
        self._is_generating = False
        self._finish_generation()
        self.notify("Generation stopped", severity="warning")
    
    # Action handlers
    def action_undo(self) -> None:
        """Undo last input change."""
        try:
            if len(self._undo_stack) > 1:
                current = self._undo_stack.pop()
                self._redo_stack.append(current)
                input_widget = self.query_one("#prompt-input", TextArea)
                input_widget.text = self._undo_stack[-1] if self._undo_stack else ""
        except Exception:
            pass
    
    def action_redo(self) -> None:
        """Redo last undone change."""
        try:
            if self._redo_stack:
                text = self._redo_stack.pop()
                self._undo_stack.append(text)
                input_widget = self.query_one("#prompt-input", TextArea)
                input_widget.text = text
        except Exception:
            pass
    
    def action_prev_prompt(self) -> None:
        """Navigate to previous prompt in history."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            
            if self._prompt_history_index == -1:
                self._prompt_history_index = len(self._prompt_history) - 1
            elif self._prompt_history_index > 0:
                self._prompt_history_index -= 1
            
            if 0 <= self._prompt_history_index < len(self._prompt_history):
                input_widget.text = self._prompt_history[self._prompt_history_index]
        except Exception:
            pass
    
    def action_next_prompt(self) -> None:
        """Navigate to next prompt in history."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            
            if self._prompt_history_index < len(self._prompt_history) - 1:
                self._prompt_history_index += 1
                input_widget.text = self._prompt_history[self._prompt_history_index]
            else:
                self._prompt_history_index = -1
                input_widget.text = ""
        except Exception:
            pass
    
    def action_new_chat(self) -> None:
        """Start a new chat session."""
        # Save current session if it has messages
        if self._current_session.messages:
            self._save_current_session()
            self._sessions.insert(0, self._current_session)
        
        # Create new session
        self._current_session = ChatSession(
            provider=self._llm_provider,
            model=self._llm_model,
        )
        
        # Clear chat log
        try:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.clear()
        except Exception:
            pass
        
        self._show_welcome_message()
        self._update_stats_display()
        self._update_history_list()
        self.notify("New chat started", severity="success")
    
    def action_clear_chat(self) -> None:
        """Clear current chat."""
        try:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.clear()
            self._current_session.messages.clear()
            self._show_welcome_message()
            self._update_stats_display()
            self.notify("Chat cleared", severity="information")
        except Exception:
            pass
    
    def action_export_chat(self) -> None:
        """Export current chat to file with custom name."""
        if not self._current_session.messages:
            self.notify("No messages to export", severity="warning")
            return
        
        # Generate default name from session name or first message
        default_name = self._current_session.name
        if default_name == "New Chat" and self._current_session.messages:
            # Use first few words of first user message
            first_msg = self._current_session.messages[0].content
            words = first_msg.split()[:5]
            default_name = "_".join(words).replace("?", "").replace("!", "")[:30]
        
        # Show dialog for custom name
        def handle_export_name(name: str) -> None:
            if not name:
                return  # User cancelled
            
            try:
                export_dir = Path("exports")
                export_dir.mkdir(exist_ok=True)
                
                # Sanitize filename
                safe_name = "".join(c if c.isalnum() or c in "_- " else "_" for c in name)
                safe_name = safe_name.strip().replace(" ", "_")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = export_dir / f"ai_conversation_{safe_name}_{timestamp}.json"
                
                # Update session name
                self._current_session.name = name
                
                with open(filename, 'w') as f:
                    json.dump(self._current_session.to_dict(), f, indent=2)
                
                self.notify(f"Chat exported to {filename}", severity="success")
            except Exception as e:
                self.notify(f"Export failed: {e}", severity="error")
        
        self.app.push_screen(ExportNameDialog(default_name), handle_export_name)
    
    def action_import_chat(self) -> None:
        """Import chat from file with selection dialog."""
        try:
            # Look for exports in both exports directory and chat history
            export_dir = Path("exports")
            history_dir = Path.home() / ".proxima" / "chat_history"
            
            chat_files = []
            
            # Find exported chats
            if export_dir.exists():
                for f in export_dir.glob("ai_*.json"):
                    chat_files.append(f)
                for f in export_dir.glob("ai_conversation_*.json"):
                    if f not in chat_files:
                        chat_files.append(f)
            
            # Find chat history files
            if history_dir.exists():
                for f in history_dir.glob("chat_*.json"):
                    chat_files.append(f)
            
            if not chat_files:
                self.notify("No chat exports found", severity="warning")
                return
            
            # Sort by modification time (newest first)
            chat_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # If only one file, import directly
            if len(chat_files) == 1:
                self._import_chat_file(chat_files[0])
                return
            
            # Build display names for selection
            file_options = []
            for f in chat_files[:20]:  # Limit to 20 most recent
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                    name = data.get('name', f.stem)
                    created = data.get('created_at', '')[:10]  # Just date
                    msg_count = len(data.get('messages', []))
                    display_name = f"{name} ({msg_count} msgs, {created})"
                except Exception:
                    display_name = f.stem
                file_options.append((display_name, str(f)))
            
            # Show selection dialog
            def handle_import_selection(file_path: str) -> None:
                if not file_path:
                    return  # User cancelled
                self._import_chat_file(Path(file_path))
            
            self.app.push_screen(ImportSelectDialog(file_options), handle_import_selection)
            
        except Exception as e:
            self.notify(f"Import failed: {e}", severity="error")
    
    def _import_chat_file(self, file_path: Path) -> None:
        """Import chat from a specific file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Load session
            self._current_session = ChatSession.from_dict(data)
            
            # Restore chat log
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.clear()
            
            self._show_welcome_message()
            
            # Replay messages
            theme = get_theme()
            for msg in self._current_session.messages:
                text = Text()
                if msg.role == 'user':
                    text.append("\n👤 You\n", style=f"bold {theme.primary}")
                else:
                    text.append("\n🤖 AI\n", style=f"bold {theme.accent}")
                text.append(msg.content + "\n", style=theme.fg_base)
                chat_log.write(text)
            
            self._update_stats_display()
            self._save_to_state()  # Save to state for persistence
            self.notify(f"Chat imported: {self._current_session.name}", severity="success")
            
        except Exception as e:
            self.notify(f"Import failed: {e}", severity="error")
    
    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcuts help."""
        self.notify(
            "Enter=New Line | Ctrl+Enter=Send | Ctrl+J/L=History | Ctrl+N=New | Ctrl+S=Export | Ctrl+O=Import",
            severity="information",
            timeout=5
        )
    
    def action_copy_or_cancel(self) -> None:
        """Handle Ctrl+C - copy or cancel generation."""
        if self._is_generating:
            self._stop_generation()
    
    def action_select_all(self) -> None:
        """Select all text in input."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            input_widget.select_all()
        except Exception:
            pass
    
    def action_goto_ai_assistant(self) -> None:
        """Already on AI Assistant screen."""
        pass
