"""AI Assistant Screen for Proxima TUI.

Full-featured AI chat interface with AGENT CAPABILITIES:
- Persistent conversation memory
- Import/export chat functionality
- Keyboard shortcuts
- Real-time stats tracking
- Multi-line input support

AGENT FEATURES (Phase 10):
- Clone repositories from GitHub
- Build backends (clone and compile)
- Monitor multiple terminals
- Administrative control through consent
- Read/write files and folders
- Run scripts (Python, PowerShell, Bash)
- Git operations (clone, pull, push, commit)

EXPERIMENT EXECUTION FEATURES:
- Run experiments/scripts with real-time monitoring
- Track execution details in Execution tab
- Display analysis and results in Results tab
- Terminal output streaming
"""

import json
import time
import subprocess
import os
import platform
import re
import shutil
import stat
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Button, Input, RichLog, TextArea, Label, Select
from textual.binding import Binding
from textual.screen import ModalScreen
from textual import events
from textual.timer import Timer
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.syntax import Syntax

from .base import BaseScreen
from ..styles.theme import get_theme
from textual.message import Message

# Import LLM components
try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import Agent components
try:
    from proxima.agent import (
        AgentController,
        TerminalEvent,
        ConsentRequest,
        ConsentResponse,
        ToolResult,
    )
    from proxima.agent.safety import ConsentType, ConsentDecision
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


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
    
    # Reactive state for sidebar visibility and panel width
    sidebar_visible: bool = True
    chat_panel_width: float = 75.0
    stats_visible: bool = True
    shortcuts_visible: bool = True
    
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
        width: 75%;
        height: 100%;
        border-right: solid $primary;
    }
    
    /* Resize handle/slider for adjusting panel width */
    AIAssistantScreen .resize-handle {
        width: 3;
        height: 100%;
        background: $primary-darken-2;
        content-align: center middle;
    }
    
    AIAssistantScreen .resize-handle:hover {
        background: $accent;
    }
    
    AIAssistantScreen .resize-handle.dragging {
        background: $success;
    }
    
    AIAssistantScreen .sidebar-panel {
        width: 22%;
        height: 100%;
        background: $surface-darken-1;
    }
    
    AIAssistantScreen .sidebar-panel.hidden {
        display: none;
    }
    
    AIAssistantScreen .header-section {
        height: 4;
        padding: 0 1;
        background: $primary-darken-2;
        border-bottom: solid $primary;
        layout: horizontal;
        align: left middle;
    }
    
    AIAssistantScreen .header-title {
        text-style: bold;
        color: $accent;
        width: 1fr;
    }
    
    /* Agent status badge */
    AIAssistantScreen .agent-badge {
        background: $success;
        color: $surface;
        padding: 0 1;
        text-style: bold;
        margin-right: 1;
    }
    
    AIAssistantScreen .agent-badge.disabled {
        background: $warning;
    }
    
    /* Sidebar toggle button */
    AIAssistantScreen .sidebar-toggle-btn {
        width: 4;
        min-width: 4;
        height: 3;
        background: transparent;
    }
    
    AIAssistantScreen .header-subtitle {
        color: $text-muted;
        text-align: center;
    }
    
    AIAssistantScreen .chat-log-container {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-background: transparent;
        scrollbar-color: transparent;
    }
    
    AIAssistantScreen .chat-log-container:hover {
        scrollbar-color: $primary 40%;
    }
    
    AIAssistantScreen .chat-log {
        height: 100%;
        /* Eye-pleasing gray background instead of black */
        background: #2d3748;
        padding: 1;
        border: solid $primary-darken-3;
        /* Word wrap enabled, no horizontal scroll */
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-background: transparent;
        scrollbar-color: transparent;
    }
    
    AIAssistantScreen .chat-log:hover {
        scrollbar-color: $primary 40%;
    }
    
    /* Sidebar panel styles with collapse support */
    AIAssistantScreen .sidebar-panel {
        width: 22%;
        height: 100%;
        background: $surface-darken-1;
        transition: width 200ms;
        scrollbar-background: transparent;
        scrollbar-color: transparent;
    }
    
    AIAssistantScreen .sidebar-panel:hover {
        scrollbar-color: $primary 40%;
    }
    
    AIAssistantScreen .sidebar-panel.collapsed {
        width: 6;
        min-width: 6;
        padding: 0;
    }
    
    AIAssistantScreen .chat-area.expanded {
        width: 1fr;
        border-right: none;
    }
    
    AIAssistantScreen .sidebar-panel.hidden {
        display: none;
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
    
    /* Sidebar styles - collapsible sections */
    AIAssistantScreen .sidebar-section {
        height: auto;
        padding: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    AIAssistantScreen .sidebar-section.hidden {
        display: none;
    }
    
    AIAssistantScreen .sidebar-header {
        layout: horizontal;
        height: 3;
        align: left middle;
        border-bottom: solid $primary-darken-3;
        padding: 0 1;
        background: $surface-darken-2;
    }
    
    AIAssistantScreen .sidebar-title {
        text-style: bold;
        color: $accent;
        width: 1fr;
    }
    
    AIAssistantScreen .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    AIAssistantScreen .toggle-btn {
        width: auto;
        min-width: 4;
        height: 3;
        background: transparent;
        border: none;
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
    
    /* Large text for AI responses (2x effect) */
    AIAssistantScreen .ai-response {
        text-style: bold;
        padding: 1;
    }
    
    /* Sidebar collapse button */
    AIAssistantScreen .sidebar-collapse-btn {
        width: auto;
        min-width: 3;
        height: 100%;
        background: $primary-darken-2;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the AI Assistant screen with AGENT capabilities."""
        super().__init__(**kwargs)
        
        # Chat state - will be restored from TUIState if available
        self._current_session: ChatSession = ChatSession()
        self._sessions: List[ChatSession] = []
        self._prompt_history: List[str] = []
        self._prompt_history_index: int = -1
        self._undo_stack: List[str] = []
        self._redo_stack: List[str] = []
        
        # Experiment Execution state (uses app's Execution screen)
        self._current_experiment_process: Optional[subprocess.Popen] = None
        self._experiment_output_thread: Optional[threading.Thread] = None
        self._experiment_stop_flag: bool = False
        self._experiment_output_lines: List[str] = []
        self._experiment_start_time: Optional[float] = None
        
        # Multi-terminal monitoring support
        self._active_processes: Dict[str, subprocess.Popen] = {}  # process_id -> Popen
        self._process_threads: Dict[str, threading.Thread] = {}   # process_id -> Thread
        self._process_outputs: Dict[str, List[str]] = {}          # process_id -> output lines
        self._process_info: Dict[str, Dict] = {}                  # process_id -> metadata
        
        # LLM state
        self._llm_router: Optional[LLMRouter] = None
        self._llm_provider: str = 'none'
        self._llm_model: str = ''
        self._is_generating: bool = False
        
        # AGENT state - enables all agentic capabilities
        self._agent: Optional[Any] = None
        self._agent_enabled: bool = True
        self._pending_consents: List[Any] = []
        self._terminal_outputs: Dict[str, List[str]] = {}
        
        # Stats
        self._stats = {
            'total_messages': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_requests': 0,
            'avg_response_time': 0,
            'session_start': time.time(),
            'commands_run': 0,
            'files_modified': 0,
            'repos_cloned': 0,
        }
        
        # Initialize Agent Controller
        self._initialize_agent()
        
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
    
    def _initialize_agent(self) -> None:
        """Initialize the Agent Controller with all capabilities."""
        if AGENT_AVAILABLE:
            try:
                project_root = Path.cwd()
                self._agent = AgentController(
                    project_root=str(project_root),
                    auto_approve_safe=True,
                    consent_callback=self._handle_consent_request,
                )
                self._agent.start()
                self._agent.add_event_listener(self._on_terminal_event)
            except Exception as e:
                self._agent = None
        else:
            self._agent = None
    
    def _handle_consent_request(self, request: Any) -> Any:
        """Handle consent requests from agent for dangerous operations."""
        if not AGENT_AVAILABLE:
            return None
        self._pending_consents.append(request)
        # Auto-approve for now (can add dialog later)
        return ConsentResponse(
            request_id=request.id,
            decision=ConsentDecision.APPROVED,
        )
    
    def _on_terminal_event(self, event: Any) -> None:
        """Handle terminal events from agent."""
        try:
            if hasattr(event, 'terminal_id') and hasattr(event, 'data'):
                term_id = event.terminal_id
                if term_id not in self._terminal_outputs:
                    self._terminal_outputs[term_id] = []
                
                if hasattr(event, 'event_type'):
                    if event.event_type.name == "OUTPUT":
                        output = event.data.get("output", "")
                        self._terminal_outputs[term_id].append(output)
                    elif event.event_type.name == "COMMAND_COMPLETED":
                        self._stats['commands_run'] += 1
        except Exception:
            pass
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
        """Show welcome message in chat log with AGENT capabilities."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            welcome = Text()
            welcome.append("🤖 AI Assistant with AGENT Capabilities\n", style=f"bold {theme.accent}")
            welcome.append("━" * 50 + "\n\n", style=theme.border)
            
            welcome.append("Welcome! I'm your AI assistant for Proxima.\n\n", style=theme.fg_base)
            
            # Show agent status
            if self._agent and AGENT_AVAILABLE:
                welcome.append("✅ Agent Mode ACTIVE\n", style=f"bold {theme.success}")
                welcome.append("I can execute real commands and operations!\n\n", style=theme.fg_base)
            else:
                welcome.append("ℹ️ Chat Mode (Agent unavailable)\n\n", style=theme.warning)
            
            welcome.append("🔧 Agent Capabilities:\n", style=f"bold {theme.accent}")
            welcome.append("• Clone repositories: ", style=theme.fg_base)
            welcome.append("clone https://github.com/...\n", style=theme.fg_muted)
            welcome.append("• Run commands: ", style=theme.fg_base)
            welcome.append("run pip list\n", style=theme.fg_muted)
            welcome.append("• Read/write files: ", style=theme.fg_base)
            welcome.append("read file config.yaml\n", style=theme.fg_muted)
            welcome.append("• Build backends: ", style=theme.fg_base)
            welcome.append("build qiskit backend\n", style=theme.fg_muted)
            welcome.append("• Git operations: ", style=theme.fg_base)
            welcome.append("git status, git pull, git commit\n", style=theme.fg_muted)
            welcome.append("• List files: ", style=theme.fg_base)
            welcome.append("list files or dir\n\n", style=theme.fg_muted)
            
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
            welcome.append("Type 'help' to see all commands | F1 for shortcuts\n", style=theme.fg_subtle)
            
            chat_log.write(welcome)
        except Exception:
            pass
    
    def compose_main(self):
        """Compose the main content with AGENT capabilities (simple chat interface)."""
        with Horizontal(classes="main-container"):
            # Main chat area (left side)
            with Vertical(classes="chat-area", id="chat-panel"):
                # Header with agent status and sidebar toggle
                with Horizontal(classes="header-section"):
                    yield Static("🤖 Proxima AI Assistant", classes="header-title")
                    # Agent status badge
                    badge_text = "AGENT" if (self._agent and AGENT_AVAILABLE) else "CHAT"
                    badge_class = "agent-badge" if (self._agent and AGENT_AVAILABLE) else "agent-badge disabled"
                    yield Static(badge_text, classes=badge_class, id="agent-badge")
                    # Sidebar toggle button
                    yield Button("◀", id="btn-toggle-sidebar", variant="default", classes="sidebar-toggle-btn")
                
                # Chat log with word wrap enabled
                with ScrollableContainer(classes="chat-log-container"):
                    yield RichLog(
                        auto_scroll=True,
                        wrap=True,
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
            
            # Resize handle/slider for adjusting panel width
            yield Static("⋮", classes="resize-handle", id="resize-handle")
            
            # Sidebar (right side) - collapsible stats and shortcuts
            with Vertical(classes="sidebar-panel", id="sidebar-panel"):
                # Single toggle button for entire sidebar content
                with Horizontal(classes="sidebar-header"):
                    yield Static("📊 Stats & Shortcuts", classes="sidebar-title", id="sidebar-title")
                    yield Button("👁", id="btn-collapse-all", classes="toggle-btn", variant="default")
                
                # Stats section - collapsible
                with Container(classes="sidebar-section", id="stats-section"):
                    with Vertical(classes="stats-grid", id="stats-content"):
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
                            yield Static("Cmds Run:", classes="stat-label")
                            yield Static("0", classes="stat-value", id="stat-commands")
                        with Horizontal(classes="stat-row"):
                            yield Static("Session:", classes="stat-label")
                            yield Static("0m", classes="stat-value", id="stat-session-time")
                
                # Shortcuts section - collapsible
                with Container(classes="sidebar-section shortcuts-panel", id="shortcuts-section"):
                    yield Static("⌨️ Shortcuts", classes="section-title")
                    with Vertical(classes="shortcuts-list", id="shortcuts-content"):
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+↵", classes="shortcut-key")
                            yield Static("Send message", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+J/L", classes="shortcut-key")
                            yield Static("History nav", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+N", classes="shortcut-key")
                            yield Static("New chat", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+S", classes="shortcut-key")
                            yield Static("Export chat", classes="shortcut-desc")
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
            
            # Commands run (agent stats)
            try:
                self.query_one("#stat-commands", Static).update(
                    str(self._stats.get('commands_run', 0))
                )
            except Exception:
                pass
            
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
        elif btn_id == "btn-toggle-stats":
            self._toggle_stats()
        elif btn_id == "btn-toggle-shortcuts":
            self._toggle_shortcuts()
        elif btn_id == "btn-toggle-sidebar":
            self._toggle_sidebar()
        elif btn_id == "btn-collapse-all":
            self._toggle_sidebar_content()
    
    def _toggle_sidebar(self) -> None:
        """Toggle sidebar visibility - makes chat fullscreen when hidden."""
        try:
            sidebar = self.query_one("#sidebar-panel")
            resize_handle = self.query_one("#resize-handle")
            chat_panel = self.query_one("#chat-panel")
            btn = self.query_one("#btn-toggle-sidebar", Button)
            
            self.sidebar_visible = not self.sidebar_visible
            
            if self.sidebar_visible:
                # Show sidebar
                sidebar.display = True
                resize_handle.display = True
                chat_panel.styles.width = "75%"
                btn.label = "◀"
            else:
                # Hide sidebar - fullscreen chat
                sidebar.display = False
                resize_handle.display = False
                chat_panel.styles.width = "100%"
                btn.label = "▶"
        except Exception:
            pass
    
    def _toggle_sidebar_content(self) -> None:
        """Toggle all sidebar content visibility - makes chat fullscreen when collapsed."""
        try:
            sidebar = self.query_one("#sidebar-panel")
            chat_area = self.query_one("#chat-panel")
            resize_handle = self.query_one("#resize-handle")
            btn = self.query_one("#btn-collapse-all", Button)
            title = self.query_one("#sidebar-title", Static)
            stats_section = self.query_one("#stats-section")
            shortcuts_section = self.query_one("#shortcuts-section")
            
            # Check current state by checking if sidebar has collapsed class
            is_collapsed = sidebar.has_class("collapsed")
            
            if is_collapsed:
                # Expand sidebar - show stats and shortcuts
                sidebar.remove_class("collapsed")
                chat_area.remove_class("expanded")
                resize_handle.display = True
                stats_section.display = True
                shortcuts_section.display = True
                btn.label = "◀"
                title.update("📊 Stats & Shortcuts")
            else:
                # Collapse sidebar - make chat fullscreen
                sidebar.add_class("collapsed")
                chat_area.add_class("expanded")
                resize_handle.display = False
                stats_section.display = False
                shortcuts_section.display = False
                btn.label = "▶"
                title.update("📊")
        except Exception:
            pass
        except Exception:
            pass
    
    def _toggle_stats(self) -> None:
        """Toggle visibility of stats section."""
        try:
            stats_content = self.query_one("#stats-content")
            btn = self.query_one("#btn-toggle-stats", Button)
            if stats_content.display:
                stats_content.display = False
                btn.label = "👁‍🗨"  # Closed eye
            else:
                stats_content.display = True
                btn.label = "👁"  # Open eye
        except Exception:
            pass
    
    def _toggle_shortcuts(self) -> None:
        """Toggle visibility of shortcuts section."""
        try:
            shortcuts_content = self.query_one("#shortcuts-content")
            btn = self.query_one("#btn-toggle-shortcuts", Button)
            if shortcuts_content.display:
                shortcuts_content.display = False
                btn.label = "👁‍🗨"  # Closed eye
            else:
                shortcuts_content.display = True
                btn.label = "👁"  # Open eye
        except Exception:
            pass
    
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
        """Display AI message in chat with larger/bolder text."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            text = Text()
            text.append("\n🤖 AI\n", style=f"bold {theme.accent}")
            # Make AI response text bold for better visibility (2x effect)
            text.append(message + "\n", style=f"bold {theme.fg_base}")
            
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
        """Generate AI response - Use LLM to understand intent and execute operations."""
        start_time = time.time()
        
        # PHASE 1: Use LLM to analyze intent and extract operation parameters
        # This allows natural language understanding for ANY sentence structure
        operation_result = self._analyze_and_execute_with_llm(message, start_time)
        
        if operation_result:
            # Operation was successfully analyzed and executed
            return
        
        # PHASE 2: Fall back to keyword-based detection for simple cases
        # This is faster for obvious commands like "git status"
        if self._is_agent_request(message):
            self._handle_agent_request(message, start_time)
            return
        
        # PHASE 3: Regular LLM response for general questions
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
            # Fallback to intelligent simulation for non-agent queries
            self._simulate_response(message, start_time)
    
    def _analyze_and_execute_with_llm(self, message: str, start_time: float) -> bool:
        """Use the integrated LLM to analyze user intent and execute operations.
        
        This method sends the user's natural language request to the LLM with a special
        system prompt that instructs it to extract structured operation data.
        Then it executes the operation based on the LLM's analysis.
        
        Returns True if an operation was identified and executed, False otherwise.
        """
        import json
        import re
        
        # Check if LLM is available
        if not self._llm_router or not LLM_AVAILABLE:
            return False
        
        # Get provider mapping
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
        if not router_provider or router_provider == 'none':
            return False
        
        # System prompt for intent extraction
        intent_extraction_prompt = '''You are an intent analyzer for a command execution system. Analyze the user's request and determine if they want to perform a file system, git, or terminal operation.

IMPORTANT: You must respond ONLY with a JSON object, nothing else. No explanations, no markdown, just pure JSON.

Supported operations:
- FILE_CREATE: Create a file (optionally with content)
- FILE_READ: Read/view a file
- FILE_WRITE: Write content to existing file
- FILE_DELETE: Delete a file
- FILE_COPY: Copy a file
- FILE_MOVE: Move/rename a file
- FILE_APPEND: Append content to a file
- DIR_CREATE: Create a directory/folder
- DIR_DELETE: Delete a directory/folder
- DIR_LIST: List directory contents
- DIR_NAVIGATE: Change to a directory (cd)
- GIT_CLONE: Clone a repository
- GIT_STATUS: Show git status
- GIT_PULL: Pull from remote
- GIT_PUSH: Push to remote
- GIT_COMMIT: Commit changes
- GIT_ADD: Stage files
- GIT_BRANCH: Create/switch branch
- GIT_CHECKOUT: Checkout branch/file
- GIT_LOG: Show git log
- GIT_DIFF: Show git diff
- TERMINAL_CMD: Run a terminal command
- PWD: Show current directory
- NONE: Not an operation request (general question/conversation)

Response format (JSON only):
{
  "operation": "OPERATION_TYPE",
  "params": {
    "path": "/path/to/file/or/dir",
    "content": "content if applicable",
    "destination": "dest path for copy/move",
    "command": "command for terminal",
    "url": "url for git clone",
    "message": "commit message for git",
    "branch": "branch name if applicable"
  },
  "confidence": 0.95,
  "explanation": "brief explanation"
}

Examples:
User: "at C:\\Users\\test make file.txt with hello"
{"operation": "FILE_CREATE", "params": {"path": "C:\\Users\\test\\file.txt", "content": "hello"}, "confidence": 0.95, "explanation": "Create file with content"}

User: "put qwerty in new.txt at desktop"
{"operation": "FILE_CREATE", "params": {"path": "C:\\Users\\<user>\\Desktop\\new.txt", "content": "qwerty"}, "confidence": 0.9, "explanation": "Create file with content at desktop"}

User: "show me what's in the folder"
{"operation": "DIR_LIST", "params": {"path": "."}, "confidence": 0.85, "explanation": "List current directory"}

User: "what is quantum computing?"
{"operation": "NONE", "params": {}, "confidence": 0.95, "explanation": "General knowledge question, not a file operation"}

Now analyze this request:'''

        try:
            request = LLMRequest(
                prompt=message,
                system_prompt=intent_extraction_prompt,
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=500,
                provider=router_provider,
                model=self._llm_model if self._llm_model else None,
            )
            
            response = self._llm_router.route(request)
            
            if not response or not hasattr(response, 'text') or not response.text:
                return False
            
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return False
            
            json_str = json_match.group(0)
            
            try:
                intent_data = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                json_str = json_str.replace("'", '"')
                try:
                    intent_data = json.loads(json_str)
                except:
                    return False
            
            operation = intent_data.get('operation', 'NONE')
            params = intent_data.get('params', {})
            confidence = intent_data.get('confidence', 0)
            explanation = intent_data.get('explanation', '')
            
            # Skip if not an operation or low confidence
            if operation == 'NONE' or confidence < 0.6:
                return False
            
            # Show what we're doing
            self._show_ai_message(f"🔍 Understood: {explanation}")
            
            # Execute the operation based on type
            result = self._execute_llm_operation(operation, params)
            
            if result:
                self._show_ai_message(result)
                
                # Update stats
                elapsed = int((time.time() - start_time) * 1000)
                self._current_session.messages.append(
                    ChatMessage(role='assistant', content=result, thinking_time_ms=elapsed)
                )
                self._update_stats_display()
                self._save_current_session()
                self._finish_generation()
                return True
            
            return False
            
        except Exception as e:
            # Don't show error, just fall through to other methods
            return False
    
    def _execute_llm_operation(self, operation: str, params: dict) -> Optional[str]:
        """Execute an operation based on LLM-extracted intent."""
        import os
        import shutil
        import subprocess
        from pathlib import Path
        
        try:
            path = params.get('path', '')
            content = params.get('content', '')
            destination = params.get('destination', '')
            command = params.get('command', '')
            url = params.get('url', '')
            message = params.get('message', '')
            branch = params.get('branch', '')
            
            # Expand user paths and environment variables
            if path:
                path = os.path.expanduser(os.path.expandvars(path))
            if destination:
                destination = os.path.expanduser(os.path.expandvars(destination))
            
            # FILE OPERATIONS
            if operation == 'FILE_CREATE':
                if not path:
                    return "❌ No file path specified"
                
                # Create parent directories if needed
                parent = Path(path).parent
                if not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content or '')
                
                result = f"✅ Created file: `{path}`"
                if content:
                    result += f"\n📝 Content: `{content[:100]}{'...' if len(content) > 100 else ''}`"
                return result
            
            elif operation == 'FILE_READ':
                if not path:
                    return "❌ No file path specified"
                if not os.path.exists(path):
                    return f"❌ File not found: `{path}`"
                
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_content = f.read(5000)
                
                truncated = len(file_content) >= 5000
                return f"📄 **File: {path}**\n```\n{file_content}\n```" + ("\n(truncated)" if truncated else "")
            
            elif operation == 'FILE_WRITE':
                if not path:
                    return "❌ No file path specified"
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content or '')
                return f"✅ Written to file: `{path}`"
            
            elif operation == 'FILE_DELETE':
                if not path:
                    return "❌ No file path specified"
                if not os.path.exists(path):
                    return f"❌ File not found: `{path}`"
                
                os.remove(path)
                return f"✅ Deleted file: `{path}`"
            
            elif operation == 'FILE_COPY':
                if not path or not destination:
                    return "❌ Source and destination paths required"
                if not os.path.exists(path):
                    return f"❌ Source not found: `{path}`"
                
                shutil.copy2(path, destination)
                return f"✅ Copied `{path}` to `{destination}`"
            
            elif operation == 'FILE_MOVE':
                if not path or not destination:
                    return "❌ Source and destination paths required"
                if not os.path.exists(path):
                    return f"❌ Source not found: `{path}`"
                
                shutil.move(path, destination)
                return f"✅ Moved `{path}` to `{destination}`"
            
            elif operation == 'FILE_APPEND':
                if not path:
                    return "❌ No file path specified"
                
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(content or '')
                return f"✅ Appended to file: `{path}`"
            
            # DIRECTORY OPERATIONS
            elif operation == 'DIR_CREATE':
                if not path:
                    return "❌ No directory path specified"
                
                Path(path).mkdir(parents=True, exist_ok=True)
                return f"✅ Created directory: `{path}`"
            
            elif operation == 'DIR_DELETE':
                if not path:
                    return "❌ No directory path specified"
                if not os.path.exists(path):
                    return f"❌ Directory not found: `{path}`"
                
                shutil.rmtree(path)
                return f"✅ Deleted directory: `{path}`"
            
            elif operation == 'DIR_LIST':
                list_path = path or '.'
                if not os.path.exists(list_path):
                    return f"❌ Directory not found: `{list_path}`"
                
                entries = os.listdir(list_path)
                dirs = []
                files = []
                
                for entry in entries[:50]:
                    full_path = os.path.join(list_path, entry)
                    if os.path.isdir(full_path):
                        dirs.append(f"📁 {entry}/")
                    else:
                        files.append(f"📄 {entry}")
                
                result = sorted(dirs) + sorted(files)
                output = "\n".join(result[:50])
                if len(entries) > 50:
                    output += f"\n... and {len(entries) - 50} more"
                
                return f"📂 **Contents of `{list_path}`** ({len(entries)} items):\n```\n{output}\n```"
            
            elif operation == 'DIR_NAVIGATE':
                if not path:
                    return "❌ No directory path specified"
                if not os.path.exists(path):
                    return f"❌ Directory not found: `{path}`"
                if not os.path.isdir(path):
                    return f"❌ Not a directory: `{path}`"
                
                os.chdir(path)
                return f"✅ Changed directory to: `{path}`"
            
            elif operation == 'PWD':
                return f"📍 Current directory: `{os.getcwd()}`"
            
            # GIT OPERATIONS
            elif operation == 'GIT_CLONE':
                if not url:
                    return "❌ No repository URL specified"
                
                clone_path = path or '.'
                result = subprocess.run(
                    ['git', 'clone', url, clone_path] if path else ['git', 'clone', url],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    return f"✅ Cloned repository: `{url}`\n```\n{result.stdout}\n```"
                else:
                    return f"❌ Clone failed:\n```\n{result.stderr}\n```"
            
            elif operation == 'GIT_STATUS':
                result = subprocess.run(['git', 'status'], capture_output=True, text=True, timeout=30)
                return f"📊 **Git Status:**\n```\n{result.stdout or result.stderr}\n```"
            
            elif operation == 'GIT_PULL':
                result = subprocess.run(['git', 'pull'], capture_output=True, text=True, timeout=60)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Pull:**\n```\n{result.stdout or result.stderr}\n```"
            
            elif operation == 'GIT_PUSH':
                result = subprocess.run(['git', 'push'], capture_output=True, text=True, timeout=60)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Push:**\n```\n{result.stdout or result.stderr}\n```"
            
            elif operation == 'GIT_COMMIT':
                commit_msg = message or 'Update'
                # First add all changes
                subprocess.run(['git', 'add', '.'], capture_output=True, timeout=30)
                result = subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True, timeout=30)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Commit:**\n```\n{result.stdout or result.stderr}\n```"
            
            elif operation == 'GIT_ADD':
                add_path = path or '.'
                result = subprocess.run(['git', 'add', add_path], capture_output=True, text=True, timeout=30)
                return f"✅ Staged: `{add_path}`"
            
            elif operation == 'GIT_BRANCH':
                if branch:
                    # Create and switch to branch
                    result = subprocess.run(['git', 'checkout', '-b', branch], capture_output=True, text=True, timeout=30)
                    if result.returncode != 0:
                        result = subprocess.run(['git', 'checkout', branch], capture_output=True, text=True, timeout=30)
                    status = "✅" if result.returncode == 0 else "❌"
                    return f"{status} Branch `{branch}`:\n```\n{result.stdout or result.stderr}\n```"
                else:
                    result = subprocess.run(['git', 'branch', '-a'], capture_output=True, text=True, timeout=30)
                    return f"📊 **Git Branches:**\n```\n{result.stdout}\n```"
            
            elif operation == 'GIT_CHECKOUT':
                if not branch and not path:
                    return "❌ No branch or file specified"
                target = branch or path
                result = subprocess.run(['git', 'checkout', target], capture_output=True, text=True, timeout=30)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} Checkout `{target}`:\n```\n{result.stdout or result.stderr}\n```"
            
            elif operation == 'GIT_LOG':
                result = subprocess.run(['git', 'log', '--oneline', '-15'], capture_output=True, text=True, timeout=30)
                return f"📜 **Git Log:**\n```\n{result.stdout}\n```"
            
            elif operation == 'GIT_DIFF':
                result = subprocess.run(['git', 'diff'], capture_output=True, text=True, timeout=30)
                diff_output = result.stdout[:3000] if result.stdout else "No changes"
                return f"📝 **Git Diff:**\n```diff\n{diff_output}\n```"
            
            # TERMINAL COMMAND
            elif operation == 'TERMINAL_CMD':
                if not command:
                    return "❌ No command specified"
                
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
                output = result.stdout or result.stderr or "Command completed"
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Executed:** `{command}`\n```\n{output[:3000]}\n```"
            
            else:
                return None
                
        except subprocess.TimeoutExpired:
            return "❌ Command timed out"
        except PermissionError:
            return f"❌ Permission denied"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _is_agent_request(self, message: str) -> bool:
        """Check if the message is requesting agent action."""
        msg_lower = message.lower()
        agent_keywords = [
            # Git/GitHub
            'clone', 'git clone', 'github', 'repo', 'repository',
            'pull', 'push', 'commit', 'checkout', 'branch', 'switch',
            'git status', 'git log', 'git diff', 'git fetch', 'git merge',
            'git stash', 'git reset', 'git rebase', 'git init', 'git add',
            'git remote', 'git tag', 'git cherry-pick',
            # Navigation - including typos and variations
            'go inside', 'gi inside', 'go to', 'goto', 'navigate', 
            'cd ', 'change directory', 'chdir',
            'open folder', 'enter folder', 'enter directory',
            'go in ', 'get inside', 'get into', 'move to', 'switch to folder',
            # Build/Execute
            'build', 'compile', 'make', 'cmake',
            'run', 'execute', 'script', 'python', 'pip',
            # File operations - comprehensive
            'create file', 'create a file', 'make file', 'make a file',
            'write file', 'write a file', 'write to', 'write into',
            'read file', 'read a file', 'delete file', 'delete a file',
            'remove file', 'rename file', 'move file', 'copy file',
            'touch ', 'cat ', 'type ', 'view file', 'show file', 'open file',
            'append to', 'save file', 'new file', 'generate file',
            'file with name', 'file named', 'file called',
            'with content', 'containing', 'written in',
            # Directory operations
            'mkdir', 'create folder', 'create directory', 'make folder',
            'delete folder', 'remove folder', 'delete directory', 'rmdir',
            'list files', 'show files', 'ls', 'dir', 'list directory',
            'create a folder', 'make a folder', 'create a directory',
            # Terminal/Shell
            'terminal', 'command', 'shell', 'powershell', 'bash', 'cmd',
            # Package management
            'install', 'uninstall', 'npm', 'yarn', 'apt', 'brew', 'conda',
            # System
            'admin', 'sudo', 'elevated', 'permission',
            'pwd', 'current directory', 'where am i', 'working directory',
            'env', 'environment variable',
            # Search
            'find file', 'search file', 'locate', 'grep', 'search in', 'find text',
            # File info
            'file info', 'file size', 'stat ',
            # Experiment/Benchmark execution with monitoring
            'run experiment', 'execute experiment', 'start experiment', 'launch experiment',
            'run benchmark', 'execute benchmark', 'start benchmark', 'launch benchmark',
            'run test', 'execute test', 'run tests', 'execute tests',
            'monitor', 'track execution', 'watch execution',
            'run algorithm', 'execute algorithm', 'run circuit', 'execute circuit',
            'run simulation', 'execute simulation', 'start simulation',
            'run analysis', 'execute analysis', 'analyze',
        ]
        return any(kw in msg_lower for kw in agent_keywords)
    
    def _handle_agent_request(self, message: str, start_time: float) -> None:
        """Handle agent-related requests with REAL execution - COMPREHENSIVE."""
        msg_lower = message.lower()
        response = ""
        executed = False
        
        try:
            # ===== MULTI-STEP OPERATIONS =====
            # Detect complex multi-step requests and execute them in sequence
            if self._is_multi_step_request(message):
                response = self._execute_multi_step(message)
                executed = True
            
            # ===== COMBINED OPERATIONS =====
            # CD + FILE CREATE (e.g., "go inside X and create a file named Y with content Z")
            elif any(kw in msg_lower for kw in ['go inside', 'gi inside', 'go to', 'navigate', 'cd ', 'change directory']) and \
                 any(kw in msg_lower for kw in ['create file', 'create a file', 'make file', 'make a file', 'new file', 'write file']):
                dir_path = self._extract_directory_path(message)
                if dir_path:
                    response = self._execute_cd_and_create_file(message, dir_path)
                    executed = True
                else:
                    response = "Please provide a valid directory path. Example:\n`go inside C:\\path\\to\\folder and create file test.txt with hello`"
            
            # CD + GIT OPERATION (e.g., "go inside X and then git switch branch Y")
            elif any(kw in msg_lower for kw in ['go inside', 'gi inside', 'go to', 'navigate', 'cd ', 'change directory']) and 'git' in msg_lower:
                dir_path = self._extract_directory_path(message)
                if dir_path:
                    git_cmd = self._extract_git_command(message)
                    if git_cmd:
                        response = self._execute_cd_and_git(dir_path, git_cmd)
                        executed = True
                    else:
                        response = self._execute_cd(dir_path)
                        executed = True
                else:
                    response = "Please provide a valid directory path. Example:\n`go inside C:\\path\\to\\folder and git status`"
            
            # ===== DIRECTORY NAVIGATION =====
            elif any(kw in msg_lower for kw in ['go inside', 'gi inside', 'go to', 'navigate to', 'cd ', 'change directory', 'open folder', 'enter folder', 'enter directory']):
                dir_path = self._extract_directory_path(message)
                if dir_path:
                    response = self._execute_cd(dir_path)
                    executed = True
                else:
                    response = "Please provide a directory path. Example:\n`go inside C:\\Users\\project` or `cd /home/user/project`"
            
            # ===== GIT OPERATIONS =====
            # Git switch/checkout branch
            elif ('switch' in msg_lower or 'checkout' in msg_lower) and ('branch' in msg_lower or re.search(r'switch\s+\S+', msg_lower)):
                branch_name = self._extract_branch_name(message)
                if branch_name:
                    response = self._execute_git_branch_switch(branch_name)
                    executed = True
                else:
                    response = "Please specify the branch name. Example:\n`git switch main` or `checkout feature-x`"
            
            # Git clone
            elif 'clone' in msg_lower and ('git' in msg_lower or 'github' in msg_lower or 'repo' in msg_lower or 'http' in msg_lower):
                url_match = re.search(r'(https?://[^\s]+|git@[^\s]+)', message)
                if url_match:
                    url = url_match.group(1)
                    dest = self._extract_clone_destination(message, url)
                    response = self._execute_git_clone(url, dest)
                    executed = True
                else:
                    response = "Please provide a repository URL. Example:\n`clone https://github.com/user/repo.git`"
            
            # Any git command
            elif 'git' in msg_lower:
                response = self._execute_git_operation(message)
                executed = True
            
            # ===== FILE OPERATIONS =====
            # Delete file
            elif any(kw in msg_lower for kw in ['delete file', 'remove file', 'rm ', 'del ']):
                file_path = self._extract_file_path(message)
                if file_path:
                    response = self._execute_delete_file(file_path)
                    executed = True
                else:
                    response = "Please provide the file path to delete. Example:\n`delete file old_config.txt`"
            
            # Rename/move file
            elif any(kw in msg_lower for kw in ['rename file', 'move file', 'mv ']):
                response = self._execute_rename_or_move(message)
                executed = True
            
            # Copy file
            elif any(kw in msg_lower for kw in ['copy file', 'cp ', 'duplicate file']):
                response = self._execute_copy_file(message)
                executed = True
            
            # Read file
            elif any(kw in msg_lower for kw in ['read file', 'show file', 'cat ', 'type ', 'open file', 'view file', 'display file']):
                file_path = self._extract_file_path(message)
                if file_path:
                    response = self._execute_read_file(file_path)
                    executed = True
                else:
                    response = "Please provide a file path. Example:\n`read file config.yaml`"
            
            # Write/create file with content
            elif any(kw in msg_lower for kw in ['write to file', 'write file', 'create file', 'save file', 'make file']) and ('content' in msg_lower or 'with' in msg_lower):
                response = self._execute_write_file(message)
                executed = True
            
            # Create empty file
            elif any(kw in msg_lower for kw in ['create file', 'touch ', 'new file', 'make file']):
                file_path = self._extract_file_path(message)
                if file_path:
                    response = self._execute_create_file(file_path)
                    executed = True
                else:
                    response = "Please provide a file name. Example:\n`create file script.py`"
            
            # Append to file
            elif any(kw in msg_lower for kw in ['append to', 'add to file']):
                response = self._execute_append_file(message)
                executed = True
            
            # ===== DIRECTORY OPERATIONS =====
            # Delete directory
            elif any(kw in msg_lower for kw in ['delete folder', 'delete directory', 'remove folder', 'remove directory', 'rmdir', 'rd ']):
                dir_path = self._extract_directory_path(message)
                if dir_path:
                    response = self._execute_delete_directory(dir_path, 'force' in msg_lower or 'recursive' in msg_lower)
                    executed = True
                else:
                    response = "Please provide the directory path to delete. Example:\n`delete folder old_project`"
            
            # Create directory
            elif any(kw in msg_lower for kw in ['create folder', 'create directory', 'mkdir', 'make folder', 'make directory', 'new folder']):
                dir_path = self._extract_directory_path(message) or self._extract_path(message)
                if dir_path:
                    response = self._execute_mkdir(dir_path)
                    executed = True
                else:
                    response = "Please provide a directory name. Example:\n`create folder my_project`"
            
            # List directory
            elif any(kw in msg_lower for kw in ['list files', 'show files', 'ls', 'dir', 'list directory', 'show directory', 'what files', 'show contents']):
                dir_path = self._extract_directory_path(message) or self._extract_path(message) or '.'
                response = self._execute_list_dir(dir_path)
                executed = True
            
            # ===== BUILD OPERATIONS =====
            elif 'build' in msg_lower and ('backend' in msg_lower or 'compile' in msg_lower):
                backend_name = self._extract_backend_name(message)
                if backend_name:
                    response = self._execute_build_backend(backend_name)
                    executed = True
                else:
                    response = "Please specify which backend to build. Available: qiskit, cirq, pennylane, braket, lret, quest, etc."
            
            # ===== SCRIPT EXECUTION =====
            # ===== EXPERIMENT EXECUTION WITH MONITORING (CHECK FIRST!) =====
            # This must come BEFORE generic terminal commands to catch "run experiment", "run benchmark", etc.
            elif any(kw in msg_lower for kw in [
                'run experiment', 'execute experiment', 'start experiment', 'launch experiment',
                'run benchmark', 'execute benchmark', 'start benchmark', 'launch benchmark',
                'run test', 'execute test', 'run tests', 'execute tests',
                'run algorithm', 'execute algorithm', 'run circuit', 'execute circuit',
                'run simulation', 'execute simulation', 'start simulation',
                'run analysis', 'execute analysis',
                'monitor', 'track execution', 'watch execution'
            ]):
                # Use the experiment execution with monitoring
                result = self._execute_experiment_from_chat(message)
                if result:
                    response = result
                    executed = True
                else:
                    # Provide guidance on how to run experiments
                    response = """🔬 **Experiment Execution Guide**

To run an experiment with real-time monitoring, use one of these formats:

**Run a Python script:**
• `run experiment my_script.py`
• `execute experiment path/to/script.py`
• `start benchmark analyze_all.py`

**Run a command with monitoring:**
• `run experiment "python -m pytest tests/"`
• `execute experiment "python analyze_backends.py"`
• `monitor "python benchmark.py --backend qiskit"`

**Switch to Execution tab (Tab 2) to:**
• See real-time terminal output
• Monitor CPU/memory usage
• View execution status and progress
• Pause/resume/stop execution

**Switch to Results tab (Tab 3) to:**
• View execution summary
• Get AI-generated analysis
• See recommendations

Current working directory: `{cwd}`""".format(cwd=os.getcwd())
                    executed = True

            # ===== SCRIPT EXECUTION =====
            elif ('run' in msg_lower or 'execute' in msg_lower) and any(ext in msg_lower for ext in ['.py', '.sh', '.ps1', '.bat', '.cmd', 'script']):
                script_path = self._extract_file_path(message) or self._extract_path(message)
                if script_path:
                    response = self._execute_script(script_path)
                    executed = True
                else:
                    response = "Please provide the script path. Example:\n`run script.py` or `execute ./build.sh`"
            
            # ===== TERMINAL/SHELL COMMANDS =====
            elif any(kw in msg_lower for kw in ['run', 'execute', 'command', 'terminal', 'shell', 'powershell', 'cmd', 'bash']):
                command = self._extract_command(message)
                if command:
                    response = self._execute_command(command)
                    executed = True
                else:
                    response = "Please provide a command to run. Example:\n`run pip list` or `execute dir`"
            
            # ===== PACKAGE MANAGEMENT =====
            elif 'install' in msg_lower and any(kw in msg_lower for kw in ['pip', 'npm', 'package', 'yarn', 'conda']):
                package = self._extract_package_name(message)
                if package:
                    response = self._execute_install(package, message)
                    executed = True
                else:
                    response = "Please specify which package to install. Example:\n`pip install numpy`"
            
            # Uninstall package
            elif 'uninstall' in msg_lower and any(kw in msg_lower for kw in ['pip', 'npm', 'package']):
                package = self._extract_package_name(message)
                if package:
                    response = self._execute_uninstall(package, message)
                    executed = True
                else:
                    response = "Please specify which package to uninstall. Example:\n`pip uninstall numpy`"
            
            # ===== FILE INFO/PERMISSIONS =====
            elif any(kw in msg_lower for kw in ['file info', 'file size', 'file details', 'stat ']):
                file_path = self._extract_file_path(message)
                if file_path:
                    response = self._execute_file_info(file_path)
                    executed = True
                else:
                    response = "Please provide a file path. Example:\n`file info config.yaml`"
            
            # ===== SEARCH =====
            elif any(kw in msg_lower for kw in ['find file', 'search file', 'locate', 'where is']):
                response = self._execute_find_file(message)
                executed = True
            
            elif any(kw in msg_lower for kw in ['search in', 'grep', 'find text', 'search for']):
                response = self._execute_search_in_files(message)
                executed = True
            
            # ===== ENVIRONMENT =====
            elif any(kw in msg_lower for kw in ['pwd', 'current directory', 'where am i', 'current folder', 'working directory']):
                response = self._execute_pwd()
                executed = True
            
            elif any(kw in msg_lower for kw in ['env', 'environment', 'set variable', 'get variable']):
                response = self._execute_env_operation(message)
                executed = True
            
            # ===== DEFAULT =====
            else:
                response = self._get_agent_capabilities_message()
            
        except Exception as e:
            response = f"❌ Error executing command: {str(e)}"
        
        # Show response
        self._show_ai_message(response)
        
        # Update stats
        elapsed = int((time.time() - start_time) * 1000)
        self._current_session.messages.append(
            ChatMessage(role='assistant', content=response, thinking_time_ms=elapsed)
        )
        self._update_stats_display()
        self._save_current_session()
        self._finish_generation()
    
    def _extract_clone_destination(self, message: str, url: str) -> Optional[str]:
        """Extract the destination path from a clone command message."""
        message_lower = message.lower()
        
        # Common patterns for destination specification
        patterns = [
            r'(?:in|to|at|into)\s+(?:this\s+)?(?:folder|directory|path|location)?\s*[:\s]+([A-Za-z]:[\\\/][^\s,]+|[\/~][^\s,]+)',
            r'(?:in|to|at|into)\s+([A-Za-z]:[\\\/][^\s,]+)',
            r'(?:in|to|at|into)\s+([\/~][^\s,]+)',
            r'(?:clone\s+\S+\s+)([A-Za-z]:[\\\/][^\s,]+)',
            r'(?:clone\s+\S+\s+)([\/~][^\s,]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                dest = match.group(1).strip()
                # Clean up the destination path
                dest = dest.rstrip('/\\')
                return dest
        
        return None
    
    def _execute_git_clone(self, url: str, destination: Optional[str] = None) -> str:
        """Clone a git repository to a specified destination."""
        try:
            # Use agent if available
            if self._agent and AGENT_AVAILABLE:
                args = {"url": url}
                if destination:
                    args["destination"] = destination
                result = self._agent.execute_tool("git_clone", args)
                if result.success:
                    self._stats['repos_cloned'] += 1
                    dest_str = destination or result.result.get('destination', 'current directory')
                    return f"✅ Successfully cloned repository:\n`{url}`\n\nCloned to: {dest_str}"
                else:
                    return f"❌ Clone failed: {result.error}"
            
            # Fallback to direct execution
            if destination:
                dest = destination
                cmd = ['git', 'clone', url, destination]
            else:
                dest = url.split('/')[-1].replace('.git', '')
                cmd = ['git', 'clone', url]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                self._stats['repos_cloned'] += 1
                return f"✅ Successfully cloned repository:\n`{url}`\n\nCloned to: {dest}\n\n{result.stdout}"
            else:
                return f"❌ Clone failed:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "❌ Clone timed out. The repository might be too large."
        except Exception as e:
            return f"❌ Clone error: {str(e)}"
    
    def _execute_build_backend(self, backend_name: str) -> str:
        """Build a backend."""
        try:
            if self._agent and AGENT_AVAILABLE:
                result = self._agent.execute_tool("build_backend", {"backend_name": backend_name})
                if result.success:
                    return f"✅ Backend '{backend_name}' built successfully!\n\nSteps completed: {result.result.get('successful_steps', 0)}/{result.result.get('steps', 0)}"
                else:
                    return f"❌ Build failed: {result.error}"
            
            # Fallback - provide instructions
            return f"""To build the {backend_name} backend:

1. Clone the repository (if not done):
   `git clone https://github.com/.../{backend_name}.git`

2. Install dependencies:
   `pip install -e ./{backend_name}`

3. Or use the build command:
   `python -m proxima build {backend_name}`

Would you like me to try cloning and building it automatically?"""
        except Exception as e:
            return f"❌ Build error: {str(e)}"
    
    def _execute_command(self, command: str) -> str:
        """Execute a shell command."""
        try:
            if self._agent and AGENT_AVAILABLE:
                result = self._agent.execute_tool("execute_command", {"command": command})
                self._stats['commands_run'] += 1
                if result.success:
                    output = result.result.get('stdout', '')
                    return f"✅ Command executed:\n`{command}`\n\n```\n{output}\n```"
                else:
                    return f"❌ Command failed:\n{result.error}"
            
            # Fallback to direct execution
            shell = True if platform.system() == 'Windows' else False
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=60
            )
            self._stats['commands_run'] += 1
            if result.returncode == 0:
                return f"✅ Command executed:\n`{command}`\n\n```\n{result.stdout}\n```"
            else:
                return f"⚠️ Command completed with errors:\n```\n{result.stderr}\n```"
        except subprocess.TimeoutExpired:
            return "❌ Command timed out after 60 seconds."
        except Exception as e:
            return f"❌ Execution error: {str(e)}"
    
    def _execute_script(self, script_path: str) -> str:
        """Execute a script file."""
        try:
            path = Path(script_path)
            if not path.exists():
                return f"❌ Script not found: {script_path}"
            
            # Determine how to run the script
            if path.suffix == '.py':
                cmd = f'python "{script_path}"'
            elif path.suffix in ('.sh', '.bash'):
                cmd = f'bash "{script_path}"'
            elif path.suffix == '.ps1':
                cmd = f'powershell -ExecutionPolicy Bypass -File "{script_path}"'
            elif path.suffix in ('.bat', '.cmd'):
                cmd = f'"{script_path}"'
            else:
                cmd = f'"{script_path}"'
            
            return self._execute_command(cmd)
        except Exception as e:
            return f"❌ Script execution error: {str(e)}"
    
    def _execute_read_file(self, file_path: str) -> str:
        """Read a file and return its contents."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"❌ File not found: {file_path}"
            
            content = path.read_text(encoding='utf-8', errors='replace')
            lines = content.split('\n')
            if len(lines) > 50:
                content = '\n'.join(lines[:50]) + f"\n\n... ({len(lines) - 50} more lines)"
            
            return f"📄 **{file_path}**\n\n```\n{content}\n```"
        except Exception as e:
            return f"❌ Read error: {str(e)}"
    
    def _execute_list_dir(self, dir_path: str) -> str:
        """List directory contents."""
        try:
            path = Path(dir_path)
            if not path.exists():
                return f"❌ Directory not found: {dir_path}"
            
            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                    items.append(f"📄 {item.name} ({size_str})")
            
            if not items:
                return f"📂 **{dir_path}** (empty directory)"
            
            return f"📂 **{dir_path}**\n\n" + "\n".join(items[:30])
        except Exception as e:
            return f"❌ List error: {str(e)}"
    
    def _execute_mkdir(self, dir_path: str) -> str:
        """Create a directory."""
        try:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            return f"✅ Directory created: `{dir_path}`"
        except Exception as e:
            return f"❌ Could not create directory: {str(e)}"
    
    def _execute_git_operation(self, message: str) -> str:
        """Execute git operations - handles ALL git commands."""
        msg_lower = message.lower()
        try:
            # Git status
            if 'status' in msg_lower:
                return self._execute_command('git status')
            
            # Git pull
            elif 'pull' in msg_lower:
                # Check for remote/branch specification
                match = re.search(r'pull\s+(\S+)\s+(\S+)', msg_lower)
                if match:
                    return self._execute_command(f'git pull {match.group(1)} {match.group(2)}')
                match = re.search(r'pull\s+(\S+)', msg_lower)
                if match and match.group(1) not in ['from', 'the']:
                    return self._execute_command(f'git pull {match.group(1)}')
                return self._execute_command('git pull')
            
            # Git push
            elif 'push' in msg_lower:
                match = re.search(r'push\s+(\S+)\s+(\S+)', msg_lower)
                if match:
                    return self._execute_command(f'git push {match.group(1)} {match.group(2)}')
                match = re.search(r'push\s+(\S+)', msg_lower)
                if match and match.group(1) not in ['to', 'the']:
                    return self._execute_command(f'git push {match.group(1)}')
                return self._execute_command('git push')
            
            # Git commit
            elif 'commit' in msg_lower:
                match = re.search(r'["\']([^"\']+)["\']|message\s+(.+)', message)
                if match:
                    msg = match.group(1) or match.group(2)
                    return self._execute_command(f'git commit -m "{msg}"')
                return "Please provide a commit message. Example:\n`git commit \"Your message here\"`"
            
            # Git switch/checkout branch
            elif 'switch' in msg_lower or ('checkout' in msg_lower and 'branch' in msg_lower):
                branch = self._extract_branch_name(message)
                if branch:
                    return self._execute_git_branch_switch(branch)
                return "Please specify the branch name."
            
            # Git checkout (file or commit)
            elif 'checkout' in msg_lower:
                match = re.search(r'checkout\s+(\S+)', msg_lower)
                if match:
                    return self._execute_command(f'git checkout {match.group(1)}')
                return self._execute_command('git checkout')
            
            # Git branch (list, create, delete)
            elif 'branch' in msg_lower:
                if 'create' in msg_lower or 'new' in msg_lower:
                    branch = self._extract_branch_name(message)
                    if branch:
                        return self._execute_command(f'git branch {branch}')
                elif 'delete' in msg_lower or 'remove' in msg_lower:
                    branch = self._extract_branch_name(message)
                    if branch:
                        return self._execute_command(f'git branch -d {branch}')
                elif 'list' in msg_lower or 'show' in msg_lower:
                    return self._execute_command('git branch -a')
                return self._execute_command('git branch')
            
            # Git fetch
            elif 'fetch' in msg_lower:
                match = re.search(r'fetch\s+(\S+)', msg_lower)
                if match and match.group(1) not in ['from', 'the', 'all']:
                    return self._execute_command(f'git fetch {match.group(1)}')
                if 'all' in msg_lower:
                    return self._execute_command('git fetch --all')
                return self._execute_command('git fetch')
            
            # Git merge
            elif 'merge' in msg_lower:
                branch = self._extract_branch_name(message)
                if branch:
                    return self._execute_command(f'git merge {branch}')
                return "Please specify the branch to merge. Example:\n`git merge feature-branch`"
            
            # Git rebase
            elif 'rebase' in msg_lower:
                branch = self._extract_branch_name(message)
                if branch:
                    return self._execute_command(f'git rebase {branch}')
                return "Please specify the branch to rebase onto. Example:\n`git rebase main`"
            
            # Git stash
            elif 'stash' in msg_lower:
                if 'pop' in msg_lower:
                    return self._execute_command('git stash pop')
                elif 'list' in msg_lower:
                    return self._execute_command('git stash list')
                elif 'apply' in msg_lower:
                    return self._execute_command('git stash apply')
                elif 'drop' in msg_lower:
                    return self._execute_command('git stash drop')
                return self._execute_command('git stash')
            
            # Git log
            elif 'log' in msg_lower:
                if 'oneline' in msg_lower or 'short' in msg_lower:
                    return self._execute_command('git log --oneline -20')
                return self._execute_command('git log -10')
            
            # Git diff
            elif 'diff' in msg_lower:
                if 'staged' in msg_lower or 'cached' in msg_lower:
                    return self._execute_command('git diff --staged')
                return self._execute_command('git diff')
            
            # Git add
            elif 'add' in msg_lower:
                if 'all' in msg_lower or '.' in message:
                    return self._execute_command('git add .')
                # Try to extract file path
                path = self._extract_path(message)
                if path:
                    return self._execute_command(f'git add {path}')
                return self._execute_command('git add .')
            
            # Git reset
            elif 'reset' in msg_lower:
                if 'hard' in msg_lower:
                    return self._execute_command('git reset --hard')
                elif 'soft' in msg_lower:
                    return self._execute_command('git reset --soft HEAD~1')
                return self._execute_command('git reset')
            
            # Git init
            elif 'init' in msg_lower:
                return self._execute_command('git init')
            
            # Git remote
            elif 'remote' in msg_lower:
                if 'add' in msg_lower:
                    match = re.search(r'add\s+(\S+)\s+(https?://\S+|git@\S+)', message)
                    if match:
                        return self._execute_command(f'git remote add {match.group(1)} {match.group(2)}')
                return self._execute_command('git remote -v')
            
            # Default - show status
            else:
                return self._execute_command('git status')
                
        except Exception as e:
            return f"❌ Git error: {str(e)}"
    
    def _execute_install(self, package: str, message: str) -> str:
        """Install a package."""
        msg_lower = message.lower()
        try:
            if 'pip' in msg_lower:
                return self._execute_command(f'pip install {package}')
            elif 'npm' in msg_lower:
                return self._execute_command(f'npm install {package}')
            else:
                # Default to pip for Python
                return self._execute_command(f'pip install {package}')
        except Exception as e:
            return f"❌ Install error: {str(e)}"
    
    def _extract_directory_path(self, message: str) -> Optional[str]:
        """Extract a directory path from message."""
        # Pattern for Windows paths (C:\...) or Unix paths (/...)
        patterns = [
            r'(?:inside|to|into|in|at|cd|navigate\s+to?)\s+([A-Za-z]:[\\\/][^\s]+)',  # Windows: C:\path
            r'(?:inside|to|into|in|at|cd|navigate\s+to?)\s+([\/~][^\s]+)',  # Unix: /path or ~/path
            r'([A-Za-z]:[\\\/][^\s]+)',  # Windows path anywhere
            r'(?:^|\s)([\/~][^\s]+)',  # Unix path
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                path = match.group(1).strip()
                # Clean up - remove trailing punctuation
                path = path.rstrip('.,;:')
                return path
        return None
    
    def _extract_branch_name(self, message: str) -> Optional[str]:
        """Extract a git branch name from message."""
        msg_lower = message.lower()
        
        # Skip words that are NOT branch names
        skip_words = {'the', 'a', 'to', 'into', 'from', 'branch', 'named', 'called', 
                      'git', 'this', 'that', 'and', 'then', 'go', 'directory', 'folder',
                      'run', 'execute', 'script', 'file', 'switch', 'checkout'}
        
        # Try various patterns - ORDER MATTERS (most specific first)
        patterns = [
            # "switch to git branch pennylane-documentation-benchmarking" - MOST SPECIFIC
            r'(?:switch|checkout)\s+(?:to\s+)?git\s+branch\s+([\w\-\.\/]+)',
            # "git switch branch-name" or "git checkout branch-name"
            r'git\s+(?:switch|checkout)\s+([\w\-\.\/]+)',
            # "switch to branch X" or "checkout branch X"
            r'(?:switch|checkout)\s+(?:to\s+)?branch\s+([\w\-\.\/]+)',
            # "branch pennylane-..." (look for hyphenated names specifically)
            r'branch\s+([\w]+-[\w\-\.]+)',
            # Generic "branch X" but X must look like a branch name (has hyphen or slash)
            r'branch\s+([\w\-\.\/]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                branch = match.group(1).strip()
                # Validate: not a skip word and looks like a valid branch
                if branch.lower() not in skip_words and len(branch) > 1:
                    return branch
        
        # Fallback: look for hyphenated words that look like branch names
        # after keywords like "branch", "switch", "checkout"
        fallback_match = re.search(
            r'(?:branch|switch|checkout)[^a-zA-Z]+([\w]+-[\w\-\.]+)', 
            message, re.IGNORECASE
        )
        if fallback_match:
            branch = fallback_match.group(1).strip()
            if branch.lower() not in skip_words:
                return branch
        
        return None

    def _extract_git_command(self, message: str) -> Optional[str]:
        """Extract git command from message (for combined cd + git operations)."""
        msg_lower = message.lower()
        
        # Define boundaries where we should STOP extracting (next action)
        stop_boundaries = [
            'and then go', 'then go', 'and go',
            'and then run', 'then run', 'and run',
            'and then execute', 'then execute', 'and execute',
            'and then navigate', 'then navigate',
            'and then open', 'then open',
            'and then cd', 'then cd',
            'and then create', 'then create',
            'and then delete', 'then delete',
            'and then list', 'then list',
        ]
        
        # Look for git commands with specific patterns
        # First, look for 'switch to git branch X' or 'git switch X' or 'git checkout X'
        patterns = [
            # "switch to git branch X" or "switch to branch X"
            r'switch\s+(?:to\s+)?(?:git\s+)?branch\s+([\w\-\.]+)',
            # "git switch X" 
            r'git\s+switch\s+([\w\-\.]+)',
            # "git checkout X"
            r'git\s+checkout\s+([\w\-\.]+)',
            # "checkout branch X"
            r'checkout\s+(?:branch\s+)?([\w\-\.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                branch_name = match.group(1).strip()
                # Return as a proper git switch command
                return f'git switch {branch_name}'
        
        # For other git commands, extract carefully
        git_match = re.search(r'(?:and\s+then\s+|then\s+|and\s+)?(git\s+\w+)', message, re.IGNORECASE)
        if git_match:
            git_start = git_match.start(1)
            remaining = message[git_start:]
            
            # Find where to stop
            stop_pos = len(remaining)
            for boundary in stop_boundaries:
                pos = remaining.lower().find(boundary)
                if pos > 0 and pos < stop_pos:
                    stop_pos = pos
            
            # Also stop at common sentence boundaries
            for boundary in ['. ', ', and then', ' and then ']:
                pos = remaining.lower().find(boundary)
                if pos > 0 and pos < stop_pos:
                    stop_pos = pos
            
            git_cmd = remaining[:stop_pos].strip().rstrip('.,;:')
            if git_cmd:
                return git_cmd
        
        return None
    
    def _is_multi_step_request(self, message: str) -> bool:
        """Check if message contains multiple sequential actions."""
        msg_lower = message.lower()
        # Count action indicators
        action_indicators = [
            'and then', 'then go', 'then run', 'then execute', 
            'then switch', 'then navigate', 'then cd', 'then open'
        ]
        count = sum(1 for ind in action_indicators if ind in msg_lower)
        # If 2+ "then" actions, it's multi-step
        return count >= 2
    
    def _execute_multi_step(self, message: str) -> str:
        """Execute a multi-step request sequentially."""
        results = []
        msg_lower = message.lower()
        
        # Step 1: Extract and execute first directory navigation
        dir_path = self._extract_directory_path(message)
        if dir_path:
            try:
                path = Path(dir_path)
                if path.exists() and path.is_dir():
                    os.chdir(dir_path)
                    results.append(f"✅ **Step 1:** Changed to `{dir_path}`")
                else:
                    results.append(f"❌ **Step 1:** Directory not found: `{dir_path}`")
                    return "\n\n".join(results)
            except Exception as e:
                results.append(f"❌ **Step 1:** Error: {str(e)}")
                return "\n\n".join(results)
        
        # Step 2: Extract and execute git branch switch
        if 'switch' in msg_lower or 'checkout' in msg_lower or 'branch' in msg_lower:
            branch_name = self._extract_branch_name(message)
            if branch_name:
                try:
                    # First, fetch latest from remote
                    subprocess.run(['git', 'fetch', '--all'], capture_output=True, timeout=30)
                    
                    # Try git switch first (handles both local and remote tracking)
                    result = subprocess.run(
                        ['git', 'switch', branch_name],
                        capture_output=True, text=True, timeout=30
                    )
                    
                    if result.returncode != 0:
                        # Try checkout (also handles remote branches)
                        result = subprocess.run(
                            ['git', 'checkout', branch_name],
                            capture_output=True, text=True, timeout=30
                        )
                    
                    if result.returncode != 0:
                        # Try creating tracking branch from remote
                        result = subprocess.run(
                            ['git', 'checkout', '-b', branch_name, f'origin/{branch_name}'],
                            capture_output=True, text=True, timeout=30
                        )
                    
                    if result.returncode != 0:
                        # Last resort: try switch -c to create from remote
                        result = subprocess.run(
                            ['git', 'switch', '-c', branch_name, f'origin/{branch_name}'],
                            capture_output=True, text=True, timeout=30
                        )
                    
                    if result.returncode == 0:
                        results.append(f"✅ **Step 2:** Switched to branch `{branch_name}`")
                    else:
                        error = result.stderr.strip() if result.stderr else 'Failed to switch'
                        results.append(f"❌ **Step 2:** {error}")
                        return "\n\n".join(results)
                except Exception as e:
                    results.append(f"❌ **Step 2:** Git error: {str(e)}")
                    return "\n\n".join(results)
        
        # Step 3: Extract and execute second navigation (subdirectory)
        # Look for "go into this X directory" or "go into X" AFTER the git command
        subdir_patterns = [
            # "go into this benchmarks/pennylane directory"
            r'go\s+into\s+(?:this\s+)?([a-zA-Z_][a-zA-Z0-9_/\\-]+)(?:\s+directory)',
            # "go into benchmarks/pennylane"
            r'(?:and\s+then\s+)?go\s+into\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_/\\-]+)',
            # "navigate to benchmarks/pennylane"
            r'navigate\s+(?:to|into)\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_/\\-]+)',
            # "cd benchmarks/pennylane"
            r'cd\s+([a-zA-Z_][a-zA-Z0-9_/\\-]+)',
        ]
        
        # Find the LAST "go into" or "navigate" that's AFTER the git branch switch
        # This ensures we get the subdirectory, not the initial directory
        branch_keywords = ['switch', 'checkout', 'branch']
        last_branch_pos = -1
        for kw in branch_keywords:
            pos = msg_lower.rfind(kw)
            if pos > last_branch_pos:
                last_branch_pos = pos
        
        if last_branch_pos > 0:
            # Look for directory navigation AFTER the branch keyword
            remaining_msg = message[last_branch_pos:]
            
            # Skip past the branch name itself
            branch_name_match = re.search(r'[\w\-\.]+', remaining_msg[10:])  # Skip "switch/checkout"
            if branch_name_match:
                search_start = last_branch_pos + 10 + branch_name_match.end()
                remaining_msg = message[search_start:]
            
            for pattern in subdir_patterns:
                match = re.search(pattern, remaining_msg, re.IGNORECASE)
                if match:
                    subdir = match.group(1).strip()
                    # Clean up the path - remove trailing words
                    subdir = re.sub(r'\s+(directory|folder|and|then).*$', '', subdir, flags=re.IGNORECASE)
                    subdir = subdir.strip()
                    
                    # Skip if it looks like the branch name or a skip word
                    skip_words = {'git', 'branch', 'switch', 'checkout', 'and', 'then', 'this', 'run'}
                    if subdir.lower() in skip_words or subdir == branch_name:
                        continue
                    
                    try:
                        subdir_path = Path(os.getcwd()) / subdir
                        if subdir_path.exists() and subdir_path.is_dir():
                            os.chdir(str(subdir_path))
                            results.append(f"✅ **Step 3:** Changed to `{subdir}`")
                            break
                        else:
                            # Try with forward slashes converted
                            subdir_alt = subdir.replace('/', os.sep)
                            subdir_path_alt = Path(os.getcwd()) / subdir_alt
                            if subdir_path_alt.exists() and subdir_path_alt.is_dir():
                                os.chdir(str(subdir_path_alt))
                                results.append(f"✅ **Step 3:** Changed to `{subdir}`")
                                break
                            else:
                                results.append(f"❌ **Step 3:** Subdirectory not found: `{subdir}`")
                            return "\n\n".join(results)
                    except Exception as e:
                        results.append(f"❌ **Step 3:** Error: {str(e)}")
                        return "\n\n".join(results)
                    break
        
        # Step 4: Extract and execute script
        script_patterns = [
            r'run\s+(?:this\s+)?(\S+\.py)',
            r'execute\s+(?:this\s+)?(\S+\.py)',
            r'(\S+\.py)\s+script',
        ]
        
        for pattern in script_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                script_name = match.group(1).strip()
                script_path = Path(os.getcwd()) / script_name
                
                if script_path.exists():
                    try:
                        result = subprocess.run(
                            ['python', str(script_path)],
                            capture_output=True, text=True, timeout=120
                        )
                        output = result.stdout if result.stdout else result.stderr
                        status = "✅" if result.returncode == 0 else "❌"
                        results.append(f"""{status} **Step 4:** Ran `{script_name}`
```
{output.strip()[:1000] if output else 'Completed.'}
```""")
                    except subprocess.TimeoutExpired:
                        results.append(f"❌ **Step 4:** Script timed out: `{script_name}`")
                    except Exception as e:
                        results.append(f"❌ **Step 4:** Error running script: {str(e)}")
                else:
                    results.append(f"❌ **Step 4:** Script not found: `{script_name}`")
                break
        
        if not results:
            return "❌ Could not parse the multi-step request. Please break it into separate commands."
        
        return "\n\n".join(results)
    
    def _execute_cd(self, dir_path: str) -> str:
        """Change to a directory and return status."""
        try:
            path = Path(dir_path)
            if not path.exists():
                return f"❌ Directory not found: `{dir_path}`"
            if not path.is_dir():
                return f"❌ Not a directory: `{dir_path}`"
            
            # Change the working directory
            os.chdir(dir_path)
            
            # Verify and list contents
            contents = list(path.iterdir())[:10]
            files_list = ", ".join([f.name for f in contents])
            if len(contents) == 10:
                files_list += "..."
            
            return f"""✅ Changed directory to:
`{dir_path}`

📁 Contents: {files_list}

Current working directory is now set. You can run git commands here."""
        except PermissionError:
            return f"❌ Permission denied: Cannot access `{dir_path}`"
        except Exception as e:
            return f"❌ Error changing directory: {str(e)}"
    
    def _execute_cd_and_git(self, dir_path: str, git_cmd: str) -> str:
        """Change directory and run a git command."""
        try:
            path = Path(dir_path)
            if not path.exists():
                return f"❌ Directory not found: `{dir_path}`"
            if not path.is_dir():
                return f"❌ Not a directory: `{dir_path}`"
            
            # Change directory
            os.chdir(dir_path)
            
            # Execute the git command
            result = subprocess.run(
                git_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=dir_path
            )
            
            output = result.stdout if result.stdout else result.stderr
            status = "✅" if result.returncode == 0 else "❌"
            
            return f"""{status} Executed in `{dir_path}`:
`{git_cmd}`

```
{output.strip() if output else 'Command completed successfully.'}
```"""
        except subprocess.TimeoutExpired:
            return f"❌ Command timed out: `{git_cmd}`"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _execute_cd_and_create_file(self, message: str, dir_path: str) -> str:
        """Change directory and create a file with optional content."""
        import re
        
        try:
            path = Path(dir_path)
            
            # Create directory if it doesn't exist
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                dir_created = True
            else:
                dir_created = False
            
            if not path.is_dir():
                return f"❌ Not a directory: `{dir_path}`"
            
            # Change to the directory
            os.chdir(dir_path)
            
            # Extract file name from message
            msg_lower = message.lower()
            file_name = None
            
            # Try different patterns to extract filename
            patterns = [
                r'(?:file\s+)?(?:with\s+)?name[d]?\s+([^\s]+\.?\w*)',  # "with name X" or "named X"
                r'(?:file\s+)?called\s+([^\s]+\.?\w*)',  # "called X"
                r'create\s+(?:a\s+)?file\s+([^\s]+\.?\w*)',  # "create file X"
                r'make\s+(?:a\s+)?file\s+([^\s]+\.?\w*)',  # "make file X"
                r'new\s+file\s+([^\s]+\.?\w*)',  # "new file X"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    file_name = match.group(1).strip()
                    # Clean up - remove trailing words like "with", "containing"
                    file_name = re.split(r'\s+(?:with|containing|written)\s*', file_name, flags=re.IGNORECASE)[0]
                    break
            
            if not file_name:
                return f"❌ Could not extract file name from message. Please specify like: `create file test.txt`"
            
            # Extract content if any
            content = ""
            content_patterns = [
                r'(?:with|containing)\s+["\']?([^"\']+)["\']?\s+(?:written|in|inside)',  # with X written in it
                r'(?:with|containing)\s+["\']?(.+?)["\']?\s*$',  # with X at end
                r'written\s+(?:in\s+it\s+)?["\']?(.+?)["\']?\s*$',  # written X
                r'content\s*[=:]\s*["\']?(.+?)["\']?\s*$',  # content = X
            ]
            
            for pattern in content_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    break
            
            # Create the file
            file_path = path / file_name
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            result_msg = f"✅ "
            if dir_created:
                result_msg += f"Created directory: `{dir_path}`\n"
            result_msg += f"✅ Created file: `{file_path}`"
            if content:
                result_msg += f"\n📝 Content: `{content[:100]}{'...' if len(content) > 100 else ''}`"
            
            return result_msg
            
        except PermissionError:
            return f"❌ Permission denied. Cannot write to `{dir_path}`"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _execute_git_branch_switch(self, branch_name: str) -> str:
        """Switch to a git branch (handles local and remote branches)."""
        try:
            # First fetch latest from remote
            subprocess.run(['git', 'fetch', '--all'], capture_output=True, timeout=30)
            
            # Try 'git switch' (modern)
            result = subprocess.run(
                ['git', 'switch', branch_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return f"""✅ Switched to branch: `{branch_name}`

```
{result.stdout.strip() if result.stdout else 'Successfully switched branch.'}
```"""
            
            # If switch fails, try checkout (older method)
            result = subprocess.run(
                ['git', 'checkout', branch_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return f"""✅ Checked out branch: `{branch_name}`

```
{result.stdout.strip() if result.stdout else 'Successfully switched branch.'}
```"""
            
            # Try creating a tracking branch from remote
            result = subprocess.run(
                ['git', 'checkout', '-b', branch_name, f'origin/{branch_name}'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return f"""✅ Created and switched to branch: `{branch_name}` (tracking origin)

```
{result.stdout.strip() if result.stdout else 'Successfully created tracking branch.'}
```"""
            
            # Last resort: git switch -c from remote
            result = subprocess.run(
                ['git', 'switch', '-c', branch_name, f'origin/{branch_name}'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return f"""✅ Created and switched to branch: `{branch_name}` (from remote)

```
{result.stdout.strip() if result.stdout else 'Successfully created branch from remote.'}
```"""
            
            # All methods failed
            error = result.stderr.strip() if result.stderr else 'Unknown error'
            
            # Show available branches
            return f"""❌ Branch `{branch_name}` not found locally or remotely.

Error: {error}

Available branches:
{self._get_git_branches()}"""
            
        except subprocess.TimeoutExpired:
            return "❌ Git command timed out."
        except Exception as e:
            return f"❌ Git error: {str(e)}"
    
    def _get_git_branches(self) -> str:
        """Get list of git branches."""
        try:
            result = subprocess.run(
                ['git', 'branch', '-a'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return "Could not list branches."
        except:
            return "Could not list branches."
    
    # ===== FILE OPERATIONS =====
    
    def _extract_file_path(self, message: str) -> Optional[str]:
        """Extract a file path from message - comprehensive."""
        # Look for quoted paths first
        match = re.search(r'["\']([^"\']+)["\']', message)
        if match:
            return match.group(1)
        
        # Windows paths (C:\...)
        match = re.search(r'([A-Za-z]:[\\\/][^\s,;]+)', message)
        if match:
            return match.group(1).rstrip('.,;:')
        
        # Unix paths (/path or ~/path)
        match = re.search(r'(?:^|\s)([\/~][^\s,;]+)', message)
        if match:
            return match.group(1).strip().rstrip('.,;:')
        
        # File with extension
        match = re.search(r'(\S+\.(?:py|sh|ps1|bat|cmd|yaml|yml|json|txt|md|toml|csv|xml|html|css|js|ts|jsx|tsx|c|cpp|h|java|go|rs|rb|php))', message, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Relative path pattern
        match = re.search(r'(?:file|path)\s+(\S+)', message, re.IGNORECASE)
        if match:
            return match.group(1).rstrip('.,;:')
        
        return None
    
    def _execute_delete_file(self, file_path: str) -> str:
        """Delete a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"❌ File not found: `{file_path}`"
            if path.is_dir():
                return f"❌ `{file_path}` is a directory. Use `delete folder` instead."
            
            path.unlink()
            return f"✅ File deleted: `{file_path}`"
        except PermissionError:
            return f"❌ Permission denied: Cannot delete `{file_path}`"
        except Exception as e:
            return f"❌ Delete error: {str(e)}"
    
    def _execute_rename_or_move(self, message: str) -> str:
        """Rename or move a file."""
        try:
            # Try to extract source and destination
            match = re.search(r'(?:rename|move|mv)\s+(?:file\s+)?["\']?([^\s"\']+)["\']?\s+(?:to|as)\s+["\']?([^\s"\']+)["\']?', message, re.IGNORECASE)
            if not match:
                match = re.search(r'(?:rename|move|mv)\s+["\']?([^\s"\']+)["\']?\s+["\']?([^\s"\']+)["\']?', message, re.IGNORECASE)
            
            if match:
                src = match.group(1)
                dst = match.group(2)
                
                src_path = Path(src)
                if not src_path.exists():
                    return f"❌ Source not found: `{src}`"
                
                dst_path = Path(dst)
                src_path.rename(dst_path)
                return f"✅ Renamed/moved:\n`{src}` → `{dst}`"
            
            return "Please specify source and destination. Example:\n`rename file old.txt to new.txt`"
        except PermissionError:
            return "❌ Permission denied"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _execute_copy_file(self, message: str) -> str:
        """Copy a file."""
        try:
            import shutil
            match = re.search(r'(?:copy|cp|duplicate)\s+(?:file\s+)?["\']?([^\s"\']+)["\']?\s+(?:to|as)\s+["\']?([^\s"\']+)["\']?', message, re.IGNORECASE)
            if not match:
                match = re.search(r'(?:copy|cp)\s+["\']?([^\s"\']+)["\']?\s+["\']?([^\s"\']+)["\']?', message, re.IGNORECASE)
            
            if match:
                src = match.group(1)
                dst = match.group(2)
                
                src_path = Path(src)
                if not src_path.exists():
                    return f"❌ Source not found: `{src}`"
                
                if src_path.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                return f"✅ Copied:\n`{src}` → `{dst}`"
            
            return "Please specify source and destination. Example:\n`copy file config.yaml to config_backup.yaml`"
        except Exception as e:
            return f"❌ Copy error: {str(e)}"
    
    def _execute_create_file(self, file_path: str, content: str = "") -> str:
        """Create a new file."""
        try:
            path = Path(file_path)
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if path.exists():
                return f"⚠️ File already exists: `{file_path}`\nUse `write file` to overwrite."
            
            path.write_text(content, encoding='utf-8')
            return f"✅ File created: `{file_path}`"
        except PermissionError:
            return f"❌ Permission denied: Cannot create `{file_path}`"
        except Exception as e:
            return f"❌ Create error: {str(e)}"
    
    def _execute_write_file(self, message: str) -> str:
        """Write content to a file."""
        try:
            # Extract file path and content
            match = re.search(r'(?:write|create|save)\s+(?:to\s+)?(?:file\s+)?["\']?([^\s"\']+)["\']?\s+(?:with\s+)?content\s+(.+)', message, re.IGNORECASE | re.DOTALL)
            if not match:
                # Try alternate pattern
                match = re.search(r'(?:write|save)\s+["\'](.+?)["\']\s+(?:to|in)\s+["\']?([^\s"\']+)["\']?', message, re.IGNORECASE)
                if match:
                    content = match.group(1)
                    file_path = match.group(2)
                else:
                    return "Please specify file path and content. Example:\n`create file test.py with content print('hello')`"
            else:
                file_path = match.group(1)
                content = match.group(2).strip()
            
            # Remove surrounding quotes from content if present
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            elif content.startswith("'") and content.endswith("'"):
                content = content[1:-1]
            
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            
            return f"✅ File written: `{file_path}`\n\nContent ({len(content)} chars):\n```\n{content[:200]}{'...' if len(content) > 200 else ''}\n```"
        except Exception as e:
            return f"❌ Write error: {str(e)}"
    
    def _execute_append_file(self, message: str) -> str:
        """Append content to a file."""
        try:
            match = re.search(r'append\s+["\']?(.+?)["\']?\s+to\s+(?:file\s+)?["\']?([^\s"\']+)["\']?', message, re.IGNORECASE)
            if not match:
                return "Please specify content and file. Example:\n`append 'new line' to file.txt`"
            
            content = match.group(1)
            file_path = match.group(2)
            
            path = Path(file_path)
            if not path.exists():
                return f"❌ File not found: `{file_path}`"
            
            with open(path, 'a', encoding='utf-8') as f:
                f.write('\n' + content)
            
            return f"✅ Appended to `{file_path}`:\n```\n{content}\n```"
        except Exception as e:
            return f"❌ Append error: {str(e)}"
    
    def _execute_file_info(self, file_path: str) -> str:
        """Get file information."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"❌ File not found: `{file_path}`"
            
            stat = path.stat()
            size = stat.st_size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.2f} MB"
            
            from datetime import datetime
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            created = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            
            file_type = "Directory" if path.is_dir() else path.suffix or "File"
            
            return f"""📄 **File Info: {path.name}**

• **Path:** `{path.absolute()}`
• **Type:** {file_type}
• **Size:** {size_str}
• **Modified:** {modified}
• **Created:** {created}
• **Readable:** {'Yes' if os.access(path, os.R_OK) else 'No'}
• **Writable:** {'Yes' if os.access(path, os.W_OK) else 'No'}"""
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    # ===== DIRECTORY OPERATIONS =====
    
    def _execute_delete_directory(self, dir_path: str, force: bool = False) -> str:
        """Delete a directory."""
        try:
            import shutil
            path = Path(dir_path)
            
            if not path.exists():
                return f"❌ Directory not found: `{dir_path}`"
            if not path.is_dir():
                return f"❌ `{dir_path}` is a file. Use `delete file` instead."
            
            contents = list(path.iterdir())
            if contents and not force:
                return f"""⚠️ Directory `{dir_path}` is not empty ({len(contents)} items).
                
Use `delete folder {dir_path} force` to delete with all contents."""
            
            shutil.rmtree(path)
            return f"✅ Directory deleted: `{dir_path}`"
        except PermissionError:
            return f"❌ Permission denied: Cannot delete `{dir_path}`"
        except Exception as e:
            return f"❌ Delete error: {str(e)}"
    
    # ===== SEARCH OPERATIONS =====
    
    def _execute_find_file(self, message: str) -> str:
        """Find files matching a pattern."""
        try:
            # Extract search pattern
            match = re.search(r'(?:find|search|locate|where is)\s+(?:file\s+)?["\']?([^\s"\']+)["\']?', message, re.IGNORECASE)
            if not match:
                return "Please specify a filename. Example:\n`find file config.yaml`"
            
            pattern = match.group(1)
            
            # Search in current directory and subdirectories
            cwd = Path.cwd()
            matches = []
            
            for p in cwd.rglob(f'*{pattern}*'):
                if len(matches) < 20:
                    matches.append(str(p.relative_to(cwd)))
            
            if matches:
                return f"""🔍 Found {len(matches)} file(s) matching `{pattern}`:

{chr(10).join(['• ' + m for m in matches])}"""
            else:
                return f"❌ No files found matching `{pattern}` in current directory."
        except Exception as e:
            return f"❌ Search error: {str(e)}"
    
    def _execute_search_in_files(self, message: str) -> str:
        """Search for text within files."""
        try:
            # Extract search term and optional file pattern
            match = re.search(r'(?:grep|search for|find text)\s+["\']?([^"\']+)["\']?\s+(?:in\s+)?(?:files?\s+)?["\']?([^\s"\']*)["\']?', message, re.IGNORECASE)
            if not match:
                match = re.search(r'(?:grep|search)\s+["\']?([^"\']+)["\']?', message, re.IGNORECASE)
                if match:
                    search_term = match.group(1)
                    file_pattern = '*.py'
                else:
                    return "Please specify search term. Example:\n`search for 'TODO' in *.py`"
            else:
                search_term = match.group(1)
                file_pattern = match.group(2) or '*.py'
            
            cwd = Path.cwd()
            results = []
            
            for file_path in cwd.rglob(file_pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        for i, line in enumerate(content.split('\n'), 1):
                            if search_term.lower() in line.lower():
                                results.append(f"{file_path.name}:{i}: {line.strip()[:60]}")
                                if len(results) >= 20:
                                    break
                    except:
                        pass
                if len(results) >= 20:
                    break
            
            if results:
                return f"""🔍 Found {len(results)} match(es) for `{search_term}`:

```
{chr(10).join(results)}
```"""
            else:
                return f"❌ No matches found for `{search_term}` in `{file_pattern}`"
        except Exception as e:
            return f"❌ Search error: {str(e)}"
    
    # ===== ENVIRONMENT OPERATIONS =====
    
    def _execute_pwd(self) -> str:
        """Print working directory."""
        try:
            cwd = os.getcwd()
            contents = list(Path(cwd).iterdir())[:15]
            files_list = "\n".join([f"{'📁' if p.is_dir() else '📄'} {p.name}" for p in sorted(contents)])
            
            return f"""📂 **Current Directory:**
`{cwd}`

**Contents ({len(list(Path(cwd).iterdir()))} items):**
{files_list}{'...' if len(contents) >= 15 else ''}"""
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _execute_env_operation(self, message: str) -> str:
        """Handle environment variable operations."""
        msg_lower = message.lower()
        try:
            if 'set' in msg_lower:
                match = re.search(r'set\s+(?:variable\s+)?(\w+)\s*=\s*(.+)', message, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    var_value = match.group(2).strip('"\'')
                    os.environ[var_name] = var_value
                    return f"✅ Environment variable set:\n`{var_name}={var_value}`"
                return "Please specify variable and value. Example:\n`set variable MY_VAR=value`"
            
            elif 'get' in msg_lower:
                match = re.search(r'get\s+(?:variable\s+)?(\w+)', message, re.IGNORECASE)
                if match:
                    var_name = match.group(1)
                    value = os.environ.get(var_name, None)
                    if value:
                        return f"✅ `{var_name}={value}`"
                    return f"❌ Variable `{var_name}` not set."
                return "Please specify variable name. Example:\n`get variable PATH`"
            
            else:
                # List common environment variables
                common_vars = ['PATH', 'HOME', 'USER', 'SHELL', 'PYTHON', 'VIRTUAL_ENV']
                env_list = []
                for var in common_vars:
                    val = os.environ.get(var, '')
                    if val:
                        env_list.append(f"• {var}={val[:50]}{'...' if len(val) > 50 else ''}")
                
                return f"""🔧 **Environment Variables:**

{chr(10).join(env_list)}

Use `get variable NAME` or `set variable NAME=value`"""
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _execute_uninstall(self, package: str, message: str) -> str:
        """Uninstall a package."""
        msg_lower = message.lower()
        try:
            if 'pip' in msg_lower:
                return self._execute_command(f'pip uninstall -y {package}')
            elif 'npm' in msg_lower:
                return self._execute_command(f'npm uninstall {package}')
            else:
                return self._execute_command(f'pip uninstall -y {package}')
        except Exception as e:
            return f"❌ Uninstall error: {str(e)}"
    
    def _extract_path(self, message: str) -> Optional[str]:
        """Extract a file/directory path from message."""
        # Look for quoted paths first
        match = re.search(r'["\']([^"\']+)["\']', message)
        if match:
            return match.group(1)
        # Look for path-like strings
        patterns = [
            r'(\S+\.(?:py|sh|ps1|bat|cmd|yaml|yml|json|txt|md|toml))',  # Files with extensions
            r'([./\\]\S+)',  # Paths starting with ./ or /
            r'(\w+[/\\]\w+)',  # folder/file patterns
        ]
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)
        return None
    
    def _extract_command(self, message: str) -> Optional[str]:
        """Extract a command from message."""
        # Look for backtick-wrapped commands
        match = re.search(r'`([^`]+)`', message)
        if match:
            return match.group(1)
        # Look for "run X" or "execute X" patterns
        match = re.search(r'(?:run|execute|command)\s+(.+)', message, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_backend_name(self, message: str) -> Optional[str]:
        """Extract backend name from message."""
        backends = [
            'qiskit', 'cirq', 'pennylane', 'braket', 'pyquil', 'projectq', 
            'qsharp', 'quest', 'qulacs', 'lret', 'lret-pennylane', 'lret-cirq',
            'lret-phase', 'lret-phase7', 'lret-phase-7', 'stim', 'pyqrack',
            'quantumsim', 'qibo', 'qsim', 'tensorflow-quantum', 'yao', 'quirk'
        ]
        msg_lower = message.lower()
        for backend in backends:
            if backend in msg_lower:
                return backend
        return None
    
    def _extract_package_name(self, message: str) -> Optional[str]:
        """Extract package name from message."""
        match = re.search(r'install\s+(\S+)', message, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _get_agent_capabilities_message(self) -> str:
        """Return message about agent capabilities."""
        return """🤖 **I can execute ALL these operations for you:**

**🔧 Git & GitHub:**
• `clone https://github.com/user/repo.git` - Clone repository
• `git switch branch-name` / `git checkout branch` - Switch branches
• `git pull` / `git push` / `git status` - Basic operations
• `git commit "message"` / `git add .` - Stage & commit
• `git log` / `git diff` / `git stash` - View history
• `git fetch` / `git merge` / `git rebase` - Advanced ops
• `git remote -v` / `git branch -a` - Remote & branches

**📂 Navigation:**
• `go inside C:\\path\\to\\folder` - Navigate to directory
• `cd /path/to/folder` - Change directory  
• `pwd` / `current directory` - Show current location

**📄 File Operations:**
• `read file config.yaml` - Read file contents
• `create file test.py with content print('hello')` - Create file
• `write file data.txt with content Hello World` - Write to file
• `delete file old.txt` - Delete file
• `rename file old.txt to new.txt` - Rename/move file
• `copy file src.txt to dest.txt` - Copy file
• `append 'text' to file.txt` - Append to file
• `file info config.yaml` - Get file details

**📁 Directory Operations:**
• `list files` / `ls` / `dir` - List directory contents
• `create folder my_project` - Create directory
• `delete folder old_project force` - Delete directory

**🔍 Search:**
• `find file config.yaml` - Find files
• `search for 'TODO' in *.py` - Search in files (grep)

**🏗️ Build & Execute:**
• `build qiskit backend` / `build lret backend` - Build backends
• `run script.py` - Execute scripts
• `run pip list` - Run any terminal command

**📦 Package Management:**
• `pip install numpy` / `pip uninstall package` - Python packages
• `npm install package` - Node.js packages

**🔧 Environment:**
• `set variable MY_VAR=value` - Set env variable
• `get variable PATH` - Get env variable

**💡 Combined Operations:**
• `go inside C:\\project and git switch main` - Navigate + git

Just tell me what you need!"""

    def _simulate_response(self, message: str, start_time: float) -> None:
        """Generate simulated response."""
        msg_lower = message.lower()
        
        if any(word in msg_lower for word in ['hello', 'hi', 'hey', 'greet']):
            response = """Hello! I'm your AI assistant for Proxima with **full agent capabilities**. 

I can help you with:
• 🔧 Clone repos & build backends
• 📁 Read/write files and folders
• 💻 Run scripts and terminal commands
• 🔄 Git operations (clone, pull, push, commit)

What would you like to do?"""
        
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

Would you like me to build a specific backend? Just say:
`build qiskit backend` or `clone https://github.com/.../backend.git`"""
        
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
            response = self._get_agent_capabilities_message()
        
        else:
            response = """I understand you're asking about quantum computing. Here are some things I can help with:

• Quantum circuit design and optimization
• Backend configuration (CPU, GPU, cloud)
• Performance tuning and benchmarking
• Understanding quantum algorithms

**Or try agent commands:**
• `clone https://github.com/...` - Clone a repo
• `run pip list` - Run commands
• `list files` - Browse directories

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

    # ============================================
    # EXPERIMENT EXECUTION METHODS
    # Uses app's Execution screen (key 2) and Results screen (key 3)
    # ============================================
    
    def _start_experiment_execution(self, command: str, name: str = "", working_dir: str = "") -> str:
        """Start executing an experiment/script with monitoring in app's Execution screen (key 2)."""
        try:
            # Store execution info
            self._experiment_start_time = time.time()
            self._experiment_output_lines = []
            self._experiment_stop_flag = False
            
            # Generate a unique task ID
            task_id = f"ai-exp-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment_name = name or f"Experiment {datetime.now().strftime('%H:%M:%S')}"
            
            # Store experiment metadata in app state
            self.state.current_experiment = {
                'name': experiment_name,
                'command': command,
                'working_dir': working_dir or os.getcwd(),
                'started_at': datetime.now().isoformat(),
                'status': 'running',
            }
            
            # Update app state for execution tracking (used by Execution screen)
            self.state.execution_status = "RUNNING"
            self.state.is_running = True
            self.state.is_paused = False
            self.state.current_task = experiment_name  # Shows in Execution screen info panel
            self.state.current_task_id = task_id
            self.state.current_stage = f"Running: {experiment_name}"
            self.state.current_backend = "AI Assistant"
            self.state.current_simulator = "subprocess"
            self.state.progress_percent = 0
            self.state.elapsed_ms = 0
            self.state.eta_ms = 0
            
            # Start the process
            self._current_experiment_process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=working_dir or os.getcwd(),
            )
            
            # Start output reader thread
            self._experiment_output_thread = threading.Thread(
                target=self._read_experiment_output_and_update_screen,
                daemon=True
            )
            self._experiment_output_thread.start()
            
            # Write start message to state FIRST (before screen switch)
            self._write_to_execution_screen(
                f"▶️ Starting Experiment: {experiment_name}\n"
                f"Command: {command}\n"
                f"Directory: {working_dir or os.getcwd()}\n"
                f"{'━' * 50}",
                level="info"
            )
            
            # Switch to Execution screen (key 2) with slight delay to ensure state is set
            self.set_timer(0.1, lambda: self.app.action_goto_execution())
            
            self.notify(f"Started: {experiment_name}", severity="information")
            return f"✅ Started experiment: {experiment_name}\n\n🔄 Switching to Execution tab (key 2) for real-time monitoring.\n\nPress 3 to see Results when complete."
            
        except Exception as e:
            self.state.execution_status = "FAILED"
            self.state.is_running = False
            return f"❌ Failed to start experiment: {str(e)}"
    
    def _read_experiment_output_and_update_screen(self) -> None:
        """Read output from experiment process and update the Execution screen."""
        if not self._current_experiment_process:
            return
        
        process = self._current_experiment_process
        errors_count = 0
        warnings_count = 0
        lines_count = 0
        
        try:
            for line in iter(process.stdout.readline, ''):
                if self._experiment_stop_flag:
                    break
                if line:
                    line = line.rstrip('\n')
                    self._experiment_output_lines.append(line)
                    lines_count += 1
                    
                    # Count errors and warnings
                    line_lower = line.lower()
                    if 'error' in line_lower or 'exception' in line_lower or 'traceback' in line_lower:
                        errors_count += 1
                        level = "error"
                    elif 'warning' in line_lower or 'warn' in line_lower:
                        warnings_count += 1
                        level = "warning"
                    elif 'success' in line_lower or 'passed' in line_lower or 'complete' in line_lower:
                        level = "success"
                    else:
                        level = "output"
                    
                    # Write to Execution screen's log (thread-safe via call_from_thread)
                    self.app.call_from_thread(self._write_to_execution_screen, line, level)
                    
                    # Update progress in state (approximate)
                    if lines_count % 10 == 0:
                        self.state.progress_percent = min(90, self.state.progress_percent + 5)
            
            # Process finished
            process.wait()
            exit_code = process.returncode
            
            if not self._experiment_stop_flag:
                # Store metrics
                duration = time.time() - (self._experiment_start_time or time.time())
                
                # Update state with results
                self.state.current_experiment['status'] = 'completed' if exit_code == 0 else 'failed'
                self.state.current_experiment['exit_code'] = exit_code
                self.state.current_experiment['duration'] = duration
                self.state.current_experiment['output'] = '\n'.join(self._experiment_output_lines)
                self.state.current_experiment['metrics'] = {
                    'lines_processed': lines_count,
                    'errors_count': errors_count,
                    'warnings_count': warnings_count,
                }
                
                # Update app state
                self.state.execution_status = "COMPLETED" if exit_code == 0 else "FAILED"
                self.state.is_running = False
                self.state.progress_percent = 100
                
                # Finalize - call from main thread
                self.app.call_from_thread(self._finalize_experiment_execution, exit_code)
                
        except Exception as e:
            if not self._experiment_stop_flag:
                self.state.execution_status = "FAILED"
                self.state.is_running = False
                self.app.call_from_thread(self._write_to_execution_screen, f"Error: {str(e)}", "error")
    
    def _write_to_execution_screen(self, message: str, level: str = "output") -> None:
        """Write a message to the Execution screen's log via shared state."""
        try:
            from .execution import ExecutionScreen, ExecutionLog
            from rich.text import Text
            
            theme = get_theme()
            
            # Build the formatted text
            text = Text()
            timestamp = datetime.now().strftime("%H:%M:%S")
            text.append(f"[{timestamp}] ", style=theme.fg_subtle)
            
            if level == "error":
                text.append(message, style=theme.error)
            elif level == "warning":
                text.append(message, style=theme.warning)
            elif level == "success":
                text.append(message, style=theme.success)
            elif level == "info":
                text.append(message, style=f"bold {theme.info}")
            else:
                text.append(message, style=theme.fg_base)
            
            # ALWAYS store in state - this is the primary mechanism
            if not hasattr(self.state, 'pending_execution_logs'):
                self.state.pending_execution_logs = []
            self.state.pending_execution_logs.append({
                'text': text, 
                'message': message, 
                'level': level,
                'timestamp': timestamp
            })
            
            # Also try direct write if on Execution screen
            try:
                current_screen = self.app.screen
                if isinstance(current_screen, ExecutionScreen):
                    log = current_screen.query_one("#execution-log", ExecutionLog)
                    log.write(text)
                    # Mark as written so it's not duplicated
                    if self.state.pending_execution_logs:
                        self.state.pending_execution_logs[-1]['written'] = True
            except Exception:
                pass
        except Exception:
            pass
    
    def _finalize_experiment_execution(self, exit_code: int) -> None:
        """Finalize experiment and update Results screen (key 3)."""
        experiment = getattr(self.state, 'current_experiment', None)
        if not experiment:
            return
        
        theme = get_theme()
        duration = experiment.get('duration', 0)
        
        # Update state to mark completion
        self.state.execution_status = "COMPLETED" if exit_code == 0 else "FAILED"
        self.state.is_running = False
        self.state.progress_percent = 100
        self.state.current_stage = "Completed" if exit_code == 0 else "Failed"
        
        # Write completion message to Execution screen
        if exit_code == 0:
            self._write_to_execution_screen(
                f"\n{'━' * 50}\n✅ Experiment completed successfully!\nExit code: {exit_code}\nDuration: {duration:.2f}s",
                level="success"
            )
        else:
            self._write_to_execution_screen(
                f"\n{'━' * 50}\n❌ Experiment failed with exit code {exit_code}\nDuration: {duration:.2f}s",
                level="error"
            )
        
        # Create result data for Results screen
        result_data = {
            'id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'name': experiment.get('name', 'Experiment'),
            'status': 'Success' if exit_code == 0 else 'Failed',
            'backend': 'AI Assistant',
            'duration': duration,
            'success_rate': 1.0 if exit_code == 0 else 0.0,
            'avg_fidelity': 1.0 if exit_code == 0 else 0.0,
            'total_shots': experiment.get('metrics', {}).get('lines_processed', 0),
            'exit_code': exit_code,
            'command': experiment.get('command', ''),
            'output': experiment.get('output', ''),
            'metrics': experiment.get('metrics', {}),
            'metadata': {
                'started_at': experiment.get('started_at', ''),
                'working_dir': experiment.get('working_dir', ''),
            },
            'insights': self._generate_experiment_insights(experiment, exit_code),
        }
        
        # Store result in state for Results screen
        if not hasattr(self.state, 'experiment_results'):
            self.state.experiment_results = []
        self.state.experiment_results.insert(0, result_data)
        
        # Notify user and auto-switch to Results screen after delay
        if exit_code == 0:
            self.notify("✅ Experiment complete! Switching to Results...", severity="success")
        else:
            self.notify("❌ Experiment failed. Switching to Results...", severity="error")
        
        # Auto-switch to Results screen (key 3) after 2 seconds
        self.set_timer(2.0, self._auto_switch_to_results)
    
    def _auto_switch_to_results(self) -> None:
        """Automatically switch to the Results screen (key 3)."""
        try:
            self.app.action_goto_results()
        except Exception:
            pass
    
    def _generate_experiment_insights(self, experiment: Dict, exit_code: int) -> List[str]:
        """Generate AI insights for the experiment results."""
        insights = []
        metrics = experiment.get('metrics', {})
        
        if exit_code == 0:
            insights.append("✅ Experiment completed successfully")
            if metrics.get('warnings_count', 0) > 0:
                insights.append(f"⚠️ {metrics['warnings_count']} warnings detected - review recommended")
        else:
            insights.append(f"❌ Experiment failed with exit code {exit_code}")
            if metrics.get('errors_count', 0) > 0:
                insights.append(f"🔴 {metrics['errors_count']} errors detected - check execution log")
            insights.append("💡 Review the error messages and check dependencies")
        
        duration = experiment.get('duration', 0)
        if duration > 60:
            insights.append(f"⏱️ Long execution time ({duration:.1f}s) - consider optimization")
        
        lines = metrics.get('lines_processed', 0)
        if lines > 1000:
            insights.append(f"📊 Processed {lines} lines of output")
        
        return insights
    
    def _stop_experiment(self) -> None:
        """Stop the currently running experiment."""
        if self._current_experiment_process:
            try:
                self._experiment_stop_flag = True
                self._current_experiment_process.terminate()
                self._current_experiment_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._current_experiment_process.kill()
            except Exception:
                pass
            
            self.state.execution_status = "CANCELLED"
            self.state.is_running = False
            
            self._write_to_execution_screen("\n⛔ Experiment stopped by user", "warning")
            self.notify("Experiment stopped", severity="warning")
    
    # ============================================
    # MULTI-TERMINAL MONITORING METHODS
    # Supports running and monitoring multiple processes simultaneously
    # ============================================
    
    def _start_monitored_process(self, command: str, name: str = "", working_dir: str = "") -> str:
        """Start a new monitored process (supports multiple concurrent processes)."""
        process_id = f"proc-{datetime.now().strftime('%H%M%S')}-{len(self._active_processes)}"
        process_name = name or f"Process {len(self._active_processes) + 1}"
        
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=working_dir or os.getcwd(),
            )
            
            # Store process info
            self._active_processes[process_id] = process
            self._process_outputs[process_id] = []
            self._process_info[process_id] = {
                'name': process_name,
                'command': command,
                'working_dir': working_dir or os.getcwd(),
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'pid': process.pid,
            }
            
            # Start monitoring thread
            thread = threading.Thread(
                target=self._monitor_process,
                args=(process_id,),
                daemon=True
            )
            thread.start()
            self._process_threads[process_id] = thread
            
            # Write to execution log
            self._write_to_execution_screen(
                f"🚀 Started: {process_name} (PID: {process.pid})\n"
                f"   Command: {command[:60]}{'...' if len(command) > 60 else ''}\n"
                f"   ID: {process_id}",
                level="info"
            )
            
            return process_id
        except Exception as e:
            return f"Error starting process: {str(e)}"
    
    def _monitor_process(self, process_id: str) -> None:
        """Monitor a process and update execution screen."""
        if process_id not in self._active_processes:
            return
        
        process = self._active_processes[process_id]
        process_info = self._process_info[process_id]
        process_name = process_info['name']
        
        errors_count = 0
        warnings_count = 0
        lines_count = 0
        
        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                
                line = line.rstrip('\n')
                self._process_outputs[process_id].append(line)
                lines_count += 1
                
                # Classify line
                line_lower = line.lower()
                if 'error' in line_lower or 'exception' in line_lower or 'traceback' in line_lower:
                    errors_count += 1
                    level = "error"
                elif 'warning' in line_lower or 'warn' in line_lower:
                    warnings_count += 1
                    level = "warning"
                elif 'success' in line_lower or 'passed' in line_lower:
                    level = "success"
                else:
                    level = "output"
                
                # Write to execution screen (with process identifier)
                self.app.call_from_thread(
                    self._write_to_execution_screen,
                    f"[{process_name}] {line}",
                    level
                )
            
            # Process finished
            process.wait()
            exit_code = process.returncode
            
            # Update process info
            self._process_info[process_id]['status'] = 'completed' if exit_code == 0 else 'failed'
            self._process_info[process_id]['exit_code'] = exit_code
            self._process_info[process_id]['lines_count'] = lines_count
            self._process_info[process_id]['errors_count'] = errors_count
            self._process_info[process_id]['warnings_count'] = warnings_count
            
            # Write completion to execution screen
            if exit_code == 0:
                self.app.call_from_thread(
                    self._write_to_execution_screen,
                    f"✅ [{process_name}] Completed (exit code: {exit_code})",
                    "success"
                )
            else:
                self.app.call_from_thread(
                    self._write_to_execution_screen,
                    f"❌ [{process_name}] Failed (exit code: {exit_code})",
                    "error"
                )
            
            # Clean up
            del self._active_processes[process_id]
            
        except Exception as e:
            self._process_info[process_id]['status'] = 'error'
            self.app.call_from_thread(
                self._write_to_execution_screen,
                f"❌ [{process_name}] Error: {str(e)}",
                "error"
            )
    
    def _get_active_processes(self) -> List[Dict]:
        """Get list of active processes."""
        return [
            {
                'id': pid,
                **self._process_info.get(pid, {}),
                'output_lines': len(self._process_outputs.get(pid, [])),
            }
            for pid in self._active_processes.keys()
        ]
    
    def _stop_process(self, process_id: str) -> bool:
        """Stop a specific monitored process."""
        if process_id not in self._active_processes:
            return False
        
        try:
            process = self._active_processes[process_id]
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception:
            pass
        
        self._process_info[process_id]['status'] = 'stopped'
        self._write_to_execution_screen(
            f"⛔ Process {process_id} stopped by user",
            "warning"
        )
        return True
    
    def _stop_all_processes(self) -> int:
        """Stop all active monitored processes."""
        stopped = 0
        for process_id in list(self._active_processes.keys()):
            if self._stop_process(process_id):
                stopped += 1
        return stopped
    
    def _execute_experiment_from_chat(self, message: str) -> str:
        """Parse and execute experiment request from chat."""
        msg_lower = message.lower()
        
        # Extract script/command to run
        script_patterns = [
            r'run\s+(?:this\s+)?(?:experiment\s+)?(\S+\.py)',
            r'execute\s+(?:this\s+)?(?:experiment\s+)?(\S+\.py)',
            r'run\s+(?:the\s+)?(?:script\s+)?(\S+\.py)',
            r'start\s+(?:experiment\s+)?(\S+\.py)',
            r'(\S+\.py)\s+(?:experiment|script)',
            r'run\s+benchmark\s+(\S+\.py)',
            r'execute\s+benchmark\s+(\S+\.py)',
            r'run\s+test\s+(\S+\.py)',
            r'execute\s+test\s+(\S+\.py)',
        ]
        
        for pattern in script_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                script_name = match.group(1).strip()
                
                # Check if script exists
                script_path = Path(script_name)
                if not script_path.is_absolute():
                    # Try multiple locations
                    possible_paths = [
                        Path(os.getcwd()) / script_name,
                        Path(os.getcwd()) / 'scripts' / script_name,
                        Path(os.getcwd()) / 'tests' / script_name,
                        Path(os.getcwd()) / 'src' / script_name,
                    ]
                    for p in possible_paths:
                        if p.exists():
                            script_path = p
                            break
                    else:
                        script_path = Path(os.getcwd()) / script_name
                
                if script_path.exists():
                    # Build the command
                    command = f"python {script_path}"
                    
                    # Extract experiment name
                    name_match = re.search(r'(?:experiment|named?|called)\s+["\']?([^"\']+)["\']?', message, re.IGNORECASE)
                    exp_name = name_match.group(1) if name_match else script_path.stem
                    
                    return self._start_experiment_execution(
                        command=command,
                        name=exp_name,
                        working_dir=str(script_path.parent)
                    )
                else:
                    return f"❌ Script not found: `{script_name}`\n\nSearched in:\n• {os.getcwd()}\n• scripts/\n• tests/\n• src/\n\n**Tip:** Use full path or ensure the script exists."
        
        # Generic command execution with monitoring
        cmd_patterns = [
            r'run\s+(?:experiment\s+)?["`]([^"`]+)["`]',
            r"run\s+(?:experiment\s+)?'([^']+)'",
            r'execute\s+(?:experiment\s+)?["`]([^"`]+)["`]',
            r"execute\s+(?:experiment\s+)?'([^']+)'",
            r'monitor\s+(?:this\s+)?(?:command\s+)?["`]([^"`]+)["`]',
            r"monitor\s+(?:this\s+)?(?:command\s+)?'([^']+)'",
            r'track\s+(?:execution\s+)?(?:of\s+)?["`]([^"`]+)["`]',
            r"track\s+(?:execution\s+)?(?:of\s+)?'([^']+)'",
            r'run\s+benchmark\s+["`]([^"`]+)["`]',
            r"run\s+benchmark\s+'([^']+)'",
            r'run\s+test\s+["`]([^"`]+)["`]',
            r"run\s+test\s+'([^']+)'",
        ]
        
        for pattern in cmd_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                command = match.group(1).strip()
                # Determine experiment name from command
                if 'pytest' in command or 'test' in command.lower():
                    exp_name = "Test Suite Execution"
                elif 'benchmark' in command.lower():
                    exp_name = "Benchmark Execution"
                elif 'analyze' in command.lower():
                    exp_name = "Analysis Execution"
                else:
                    exp_name = f"Command: {command[:30]}..." if len(command) > 30 else f"Command: {command}"
                    
                return self._start_experiment_execution(
                    command=command,
                    name=exp_name
                )
        
        # Handle benchmark suite names (no .py extension)
        benchmark_patterns = [
            r'run\s+(?:the\s+)?(\w+)\s+benchmark',
            r'execute\s+(?:the\s+)?(\w+)\s+benchmark',
            r'benchmark\s+(\w+)',
        ]
        
        for pattern in benchmark_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                benchmark_name = match.group(1).strip().lower()
                
                # Map common benchmark names to actual commands
                benchmark_commands = {
                    'qiskit': 'python -m pytest tests/backends/test_qiskit.py -v',
                    'cirq': 'python -m pytest tests/backends/test_cirq.py -v',
                    'pennylane': 'python -m pytest tests/backends/test_pennylane.py -v',
                    'braket': 'python -m pytest tests/backends/test_braket.py -v',
                    'all': 'python -m pytest tests/ -v',
                    'core': 'python analyze_core.py',
                    'backends': 'python analyze_backends.py',
                    'full': 'python analyze_all.py',
                }
                
                if benchmark_name in benchmark_commands:
                    return self._start_experiment_execution(
                        command=benchmark_commands[benchmark_name],
                        name=f"{benchmark_name.title()} Benchmark"
                    )
                else:
                    return f"❌ Unknown benchmark: `{benchmark_name}`\n\nAvailable benchmarks: {', '.join(benchmark_commands.keys())}"
        
        # Handle pytest specifically
        if 'pytest' in msg_lower or ('run' in msg_lower and 'test' in msg_lower):
            test_path = 'tests/'
            test_match = re.search(r'(?:tests?/|test_)\S+', message)
            if test_match:
                test_path = test_match.group(0)
            
            return self._start_experiment_execution(
                command=f"python -m pytest {test_path} -v",
                name="Test Suite Execution"
            )
        
        return None  # Not an experiment request
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        # Chat button handlers
        if button_id == "btn-send":
            self._send_message()
        elif button_id == "btn-stop":
            self._stop_generation()
            # Also stop any running experiment
            if self._current_experiment_process:
                self._stop_experiment()
        elif button_id == "btn-clear":
            self.action_clear_chat()
        elif button_id == "btn-new":
            self._new_chat()
        elif button_id == "btn-import":
            self.action_import_chat()
        elif button_id == "btn-export":
            self.action_export_chat()
        elif button_id == "btn-toggle-sidebar":
            self._toggle_sidebar()
        elif button_id == "btn-collapse-all":
            self._toggle_sidebar_content()
