"""Agent-Enhanced AI Assistant Screen for Proxima TUI.

Extended AI chat interface with full agent capabilities and Phase 2 UI enhancements:
- Word wrapping for all text content
- Resizable panels with drag handles
- Collapsible real-time statistics panel
- Professional theming and layout
- Terminal command execution
- File system operations
- Git operations
- Backend code modification
- Multi-terminal monitoring
- Consent management
- Tool execution visualization
"""

import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Button, Input, RichLog, TextArea, Label, TabbedContent, TabPane
from textual.binding import Binding
from textual.screen import ModalScreen
from textual import events
from textual.app import ComposeResult
from textual.timer import Timer
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from .base import BaseScreen
from ..styles.theme import get_theme
from ..widgets.agent_widgets import (
    ConsentDialog,
    ConsentDisplayInfo,
    TerminalOutputPanel,
    ToolExecutionView,
    MultiTerminalView,
    AgentPlanView,
    UndoRedoPanel,
)
# Phase 2: Enhanced UI Widgets
from ..widgets.agent_ui_enhanced import (
    WrappedMessage,
    ChatMessageBubble,
    WordWrappedRichLog,
    ResizablePanelContainer,
    ResizeHandle,
    CollapsibleStatsPanel,
    AgentStats,
    StatsCard,
    AgentHeader,
    ToolExecutionCard,
    InputSection,
    # New sliding stats panel (no blinking)
    SlidingStatsPanel,
    SlidingStatsTrigger,
)
from textual.message import Message

# Import agent components
try:
    from proxima.agent import (
        AgentController,
        TerminalEvent,
        ConsentRequest,
        ConsentResponse,
        ToolResult,
    )
    # Phase 3: Enhanced terminal management
    from proxima.agent.multi_terminal import (
        get_multi_terminal_monitor,
        get_session_manager,
        get_command_normalizer,
        TerminalState,
        TerminalEventType,
    )
    from proxima.agent.terminal_state_machine import (
        get_terminal_state_machine,
        TerminalProcessState,
    )
    from proxima.agent.safety import ConsentType, ConsentDecision
    # GitHub Authentication
    from proxima.agent.github_auth import get_github_auth, GitHubAuth, AuthStatus
    AGENT_AVAILABLE = True
    GITHUB_AUTH_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    GITHUB_AUTH_AVAILABLE = False

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
        if event.key in ("ctrl+m", "ctrl+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.post_message(self.SendRequested())
            return
        await super()._on_key(event)


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tokens: int = 0
    thinking_time_ms: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentChatSession:
    """An agent-enhanced chat session."""
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    name: str = "Agent Chat"
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_tokens: int = 0
    total_requests: int = 0
    tool_executions: int = 0
    provider: str = ""
    model: str = ""
    agent_enabled: bool = True


class AgentAIAssistantScreen(BaseScreen):
    """AI Assistant with full agent capabilities and Phase 2 UI enhancements.
    
    Phase 2 UI/UX Enhancements:
    - Word wrapping for all text content (auto-wrap in messages)
    - Resizable panels with drag handles and keyboard shortcuts
    - Collapsible real-time statistics panel (Ctrl+T toggle)
    - Professional theming and layout
    
    Agent Features:
    - Natural language command execution
    - File system access
    - Git operations
    - Backend building and modification
    - Multi-terminal monitoring
    - Consent management for dangerous operations
    - Undo/redo for modifications
    """
    
    SCREEN_NAME = "agent_ai_assistant"
    SCREEN_TITLE = "AI Agent Assistant"
    SHOW_SIDEBAR = False
    
    BINDINGS = [
        Binding("ctrl+m", "send_on_enter", "Send", show=False, priority=True),
        Binding("ctrl+z", "undo_modification", "Undo", show=True),
        Binding("ctrl+y", "redo_modification", "Redo", show=True),
        Binding("ctrl+t", "toggle_stats", "Toggle Stats", show=True),
        Binding("ctrl+p", "toggle_panel", "Toggle Panel", show=True),
        Binding("ctrl+bracketleft", "shrink_chat", "Shrink Chat", show=False),
        Binding("ctrl+bracketright", "grow_chat", "Grow Chat", show=False),
        Binding("ctrl+n", "new_chat", "New Chat", show=True),
        Binding("ctrl+s", "export_chat", "Export Chat", show=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=False),
        Binding("escape", "go_back", "Back"),
        Binding("f1", "show_help", "Help"),
        # Navigation
        ("1", "goto_dashboard", "Dashboard"),
        ("2", "goto_execution", "Execution"),
        ("3", "goto_results", "Results"),
        ("4", "goto_backends", "Backends"),
        ("5", "goto_settings", "Settings"),
        ("6", "goto_ai_assistant", "AI Assistant"),
    ]
    
    DEFAULT_CSS = """
    /* ========================================================================
       CRUSH-STYLE DARK THEME FOR AI ASSISTANT
       Inspired by CRUSH UI - Dark purple/magenta aesthetic
       ======================================================================== */
    
    AgentAIAssistantScreen {
        layout: vertical;
        background: #0d0d14;
    }
    
    /* ========================================================================
       MAIN LAYOUT: 3-column design (code viewer | chat | sidebar)
       ======================================================================== */
    
    AgentAIAssistantScreen .main-container {
        width: 100%;
        height: 1fr;
        layout: horizontal;
        background: #0d0d14;
    }
    
    /* ========================================================================
       CODE VIEWER PANEL (Left side) - Shows file content with line numbers
       ======================================================================== */
    
    AgentAIAssistantScreen .code-viewer-panel {
        width: 40%;
        height: 100%;
        background: #0d0d14;
        border-right: solid #2a2a3e;
    }
    
    AgentAIAssistantScreen .code-viewer-header {
        height: 3;
        background: #13131d;
        padding: 0 1;
        layout: horizontal;
        border-bottom: solid #2a2a3e;
    }
    
    AgentAIAssistantScreen .file-path-label {
        width: 1fr;
        color: #8888aa;
    }
    
    AgentAIAssistantScreen .view-indicator {
        color: #00cc66;
        width: auto;
    }
    
    AgentAIAssistantScreen .code-content {
        height: 1fr;
        padding: 0;
        background: #0d0d14;
        overflow-y: auto;
        overflow-x: auto;
    }
    
    AgentAIAssistantScreen .code-text {
        background: #0d0d14;
        padding: 1;
    }
    
    /* ========================================================================
       CHAT AREA (Center) - Main conversation area
       ======================================================================== */
    
    AgentAIAssistantScreen .chat-area {
        width: 35%;
        height: 100%;
        background: #0d0d14;
        border-right: solid #2a2a3e;
    }
    
    AgentAIAssistantScreen .chat-area.fullscreen {
        width: 75%;
        border-right: none;
    }
    
    /* Chat log container - where messages appear */
    AgentAIAssistantScreen .chat-log-container {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
        overflow-x: hidden;
        background: #0d0d14;
    }
    
    AgentAIAssistantScreen .chat-log {
        height: 100%;
        background: #0d0d14;
        padding: 1;
        overflow-x: hidden;
        overflow-y: auto;
    }
    
    /* Thinking indicator at bottom of chat */
    AgentAIAssistantScreen .thinking-indicator {
        height: 3;
        background: #13131d;
        padding: 0 2;
        layout: horizontal;
        border-top: solid #2a2a3e;
    }
    
    AgentAIAssistantScreen .thinking-label {
        color: #cc66ff;
        width: 1fr;
    }
    
    AgentAIAssistantScreen .thinking-hash {
        color: #666688;
        width: auto;
    }
    
    /* ========================================================================
       INPUT SECTION - Bottom input area with prompt
       ======================================================================== */
    
    AgentAIAssistantScreen .input-section {
        height: auto;
        min-height: 5;
        max-height: 12;
        padding: 1;
        background: #13131d;
        border-top: solid #2a2a3e;
    }
    
    AgentAIAssistantScreen .input-container {
        height: auto;
        min-height: 3;
        layout: horizontal;
    }
    
    AgentAIAssistantScreen .prompt-input {
        width: 1fr;
        min-height: 3;
        max-height: 8;
        margin-right: 1;
        background: #1a1a28;
        border: solid #3a3a5e;
        color: #ccccee;
    }
    
    AgentAIAssistantScreen .prompt-input:focus {
        border: solid #cc66ff;
    }
    
    AgentAIAssistantScreen .send-btn {
        width: 10;
        height: 3;
        background: #cc66ff;
        color: #0d0d14;
        text-style: bold;
    }
    
    AgentAIAssistantScreen .send-btn:hover {
        background: #dd88ff;
    }
    
    /* Controls row below input */
    AgentAIAssistantScreen .controls-row {
        height: 3;
        layout: horizontal;
        margin-top: 1;
    }
    
    AgentAIAssistantScreen .control-btn {
        margin-right: 1;
        min-width: 6;
        height: 3;
        background: #2a2a3e;
        border: none;
        color: #aaaacc;
    }
    
    AgentAIAssistantScreen .control-btn:hover {
        background: #3a3a5e;
        color: #cc66ff;
    }
    
    AgentAIAssistantScreen .input-hint {
        display: none;
    }
    
    /* ========================================================================
       SIDEBAR (Right side) - CRUSH-style info panel
       ======================================================================== */
    
    AgentAIAssistantScreen .crush-sidebar {
        width: 25%;
        min-width: 30;
        height: 100%;
        background: #0d0d14;
        padding: 0;
    }
    
    AgentAIAssistantScreen .crush-sidebar.collapsed {
        width: 0;
        display: none;
    }
    
    /* PROXIMA Title - Retro pixel style */
    AgentAIAssistantScreen .sidebar-title-container {
        height: 6;
        background: #13131d;
        padding: 1;
        border-bottom: solid #2a2a3e;
    }
    
    AgentAIAssistantScreen .proxima-title {
        text-align: center;
        text-style: bold;
        color: #cc66ff;
    }
    
    AgentAIAssistantScreen .sidebar-subtitle {
        color: #8888aa;
        text-align: center;
        margin-top: 1;
    }
    
    AgentAIAssistantScreen .sidebar-path {
        color: #666688;
        text-align: center;
    }
    
    /* Model Info Section */
    AgentAIAssistantScreen .model-info-section {
        height: auto;
        padding: 1;
        background: #0d0d14;
        border-bottom: solid #2a2a3e;
    }
    
    AgentAIAssistantScreen .model-name {
        color: #ffffff;
        text-style: bold;
        margin-bottom: 1;
    }
    
    AgentAIAssistantScreen .model-status-row {
        layout: horizontal;
        height: auto;
    }
    
    AgentAIAssistantScreen .status-badge {
        color: #00cc66;
        background: #1a2a1a;
        padding: 0 1;
        margin-right: 1;
    }
    
    AgentAIAssistantScreen .status-badge.inactive {
        color: #cc6666;
        background: #2a1a1a;
    }
    
    AgentAIAssistantScreen .cost-label {
        color: #8888aa;
        width: 1fr;
        text-align: right;
    }
    
    /* Section Headers */
    AgentAIAssistantScreen .sidebar-section {
        height: auto;
        padding: 1;
        border-bottom: solid #1a1a28;
    }
    
    AgentAIAssistantScreen .section-header {
        color: #666688;
        text-style: bold;
        margin-bottom: 1;
    }
    
    AgentAIAssistantScreen .section-empty {
        color: #444466;
        padding-left: 1;
    }
    
    /* LSPs Section with colored dots */
    AgentAIAssistantScreen .lsp-item {
        layout: horizontal;
        height: auto;
        padding-left: 1;
        margin-bottom: 0;
    }
    
    AgentAIAssistantScreen .lsp-dot {
        color: #00cc66;
        width: 2;
    }
    
    AgentAIAssistantScreen .lsp-dot.inactive {
        color: #cc6666;
    }
    
    AgentAIAssistantScreen .lsp-name {
        color: #aaaacc;
        width: 1fr;
    }
    
    /* MCPs Section */
    AgentAIAssistantScreen .mcp-item {
        layout: horizontal;
        height: auto;
        padding-left: 1;
    }
    
    AgentAIAssistantScreen .mcp-dot {
        color: #cc66ff;
        width: 2;
    }
    
    AgentAIAssistantScreen .mcp-name {
        color: #aaaacc;
        width: 1fr;
    }
    
    /* Stats Section */
    AgentAIAssistantScreen .stats-section {
        height: auto;
        padding: 1;
        border-bottom: solid #1a1a28;
    }
    
    AgentAIAssistantScreen .stats-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 0;
    }
    
    AgentAIAssistantScreen .stats-label {
        width: 14;
        color: #666688;
    }
    
    AgentAIAssistantScreen .stats-value {
        width: 1fr;
        text-align: right;
        color: #cc66ff;
    }
    
    /* Terminal Section in sidebar */
    AgentAIAssistantScreen .terminal-section {
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    /* ========================================================================
       BOTTOM KEYBOARD SHORTCUTS BAR
       ======================================================================== */
    
    AgentAIAssistantScreen .keyboard-bar {
        height: 2;
        background: #0a0a10;
        layout: horizontal;
        padding: 0 1;
        border-top: solid #2a2a3e;
    }
    
    AgentAIAssistantScreen .kbd-item {
        color: #666688;
        margin-right: 2;
        width: auto;
    }
    
    AgentAIAssistantScreen .kbd-key {
        color: #888899;
        background: #1a1a28;
        padding: 0 1;
    }
    
    AgentAIAssistantScreen .kbd-action {
        color: #555566;
        margin-left: 1;
    }
    
    /* ========================================================================
       MESSAGE STYLES - Chat bubbles
       ======================================================================== */
    
    AgentAIAssistantScreen .user-message {
        margin: 1 0;
        padding: 1;
        background: #1a1a2e;
        border-left: thick #cc66ff;
    }
    
    AgentAIAssistantScreen .ai-message {
        margin: 1 0;
        padding: 1;
        background: #13131d;
        border-left: thick #00cc66;
    }
    
    AgentAIAssistantScreen .tool-message {
        margin: 1 0;
        padding: 1;
        background: #1a2a1a;
        border-left: thick #00cc66;
    }
    
    AgentAIAssistantScreen .error-message {
        margin: 1 0;
        padding: 1;
        background: #2a1a1a;
        border-left: thick #cc6666;
    }
    
    AgentAIAssistantScreen .tool-name {
        text-style: bold;
        color: #cc66ff;
    }
    
    AgentAIAssistantScreen .tool-result {
        color: #8888aa;
        margin-top: 1;
    }
    
    /* AI response text styling */
    AgentAIAssistantScreen .ai-response-text {
        text-style: bold;
        padding: 1;
    }
    
    /* ========================================================================
       LEGACY STYLES (kept for compatibility)
       ======================================================================== */
    
    AgentAIAssistantScreen .resize-handle {
        width: 1;
        height: 100%;
        background: transparent;
        content-align: center middle;
    }
    
    AgentAIAssistantScreen .resize-handle:hover {
        background: rgba(204, 102, 255, 0.3);
    }
    
    AgentAIAssistantScreen .resize-handle.dragging {
        background: rgba(0, 204, 102, 0.5);
    }
    
    AgentAIAssistantScreen .agent-panel {
        display: none;
    }
    
    AgentAIAssistantScreen .header-section {
        display: none;
    }
    
    AgentAIAssistantScreen .panel-tabs {
        height: 100%;
    }
    
    AgentAIAssistantScreen .tab-content {
        padding: 1;
        height: 100%;
    }
    
    AgentAIAssistantScreen .tools-section {
        height: auto;
        max-height: 50%;
        overflow-y: auto;
    }
    
    AgentAIAssistantScreen .section-title {
        text-style: bold;
        color: #cc66ff;
        margin-bottom: 1;
    }
    
    AgentAIAssistantScreen .panel-header {
        height: 3;
        padding: 0 1;
        background: #13131d;
        border-bottom: solid #2a2a3e;
        layout: horizontal;
        align: left middle;
    }
    
    AgentAIAssistantScreen .panel-header-title {
        width: 1fr;
        text-style: bold;
        color: #cc66ff;
    }
    
    AgentAIAssistantScreen .realtime-stats {
        display: none;
    }
    
    AgentAIAssistantScreen .shortcuts-panel {
        display: none;
    }
    
    AgentAIAssistantScreen .shortcut-item {
        color: #666688;
        height: auto;
    }
    
    AgentAIAssistantScreen .status-section {
        height: auto;
        padding: 1;
        background: #13131d;
        margin-bottom: 1;
    }
    
    AgentAIAssistantScreen .status-row {
        layout: horizontal;
        height: auto;
    }
    
    AgentAIAssistantScreen .status-label {
        width: 12;
        color: #666688;
    }
    
    AgentAIAssistantScreen .status-value {
        width: 1fr;
        text-align: right;
        color: #cc66ff;
    }
    
    AgentAIAssistantScreen .agent-badge {
        background: #00cc66;
        color: #0d0d14;
        padding: 0 1;
        text-style: bold;
    }
    
    AgentAIAssistantScreen .agent-badge.disabled {
        background: #cc6666;
    }
    """
    
    # Phase 2: Reactive properties for panel state
    chat_panel_width: reactive[float] = reactive(65.0)
    stats_visible: reactive[bool] = reactive(True)
    side_panel_visible: reactive[bool] = reactive(True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Chat state
        self._current_session = AgentChatSession()
        self._prompt_history: List[str] = []
        self._prompt_history_index: int = -1
        self._is_generating: bool = False
        
        # Agent state
        self._agent: Optional[AgentController] = None
        self._agent_enabled: bool = True
        self._pending_consents: List[ConsentRequest] = []
        self._active_tools: List[ToolExecutionView] = []
        
        # LLM state
        self._llm_router: Optional[LLMRouter] = None
        self._llm_provider: str = 'none'
        self._llm_model: str = ''
        
        # Phase 2: Enhanced stats tracking
        self._agent_stats = AgentStats()
        self._response_times: List[int] = []
        
        # Phase 2: Panel resize state
        self._is_resizing = False
        self._resize_start_x = 0
        self._resize_start_width = 0.0
        
        # Legacy stats dict for backward compatibility
        self._stats = {
            'total_messages': 0,
            'total_tokens': 0,
            'total_requests': 0,
            'tool_executions': 0,
            'session_start': time.time(),
        }
        
        # Initialize components
        self._initialize_components()
        self._load_panel_settings()
    
    def _initialize_components(self) -> None:
        """Initialize agent and LLM components."""
        # Initialize agent
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
            except Exception:
                self._agent = None
        
        # Initialize LLM
        if LLM_AVAILABLE:
            try:
                def auto_consent(prompt: str) -> bool:
                    return True
                self._llm_router = LLMRouter(consent_prompt=auto_consent)
            except Exception:
                pass
        
        self._load_settings()
    
    def _load_settings(self) -> None:
        """Load LLM settings from config."""
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                
                llm = settings.get('llm', {})
                self._llm_provider = llm.get('mode', 'none')
                
                model_keys = {
                    'local': 'local_model', 'ollama': 'local_model',
                    'openai': 'openai_model', 'anthropic': 'anthropic_model',
                    'google': 'google_model', 'xai': 'xai_model',
                }
                self._llm_model = llm.get(model_keys.get(self._llm_provider, ''), '')
                
                # Register API key
                api_key_map = {
                    'openai': 'openai_key', 'anthropic': 'anthropic_key',
                    'google': 'google_key', 'xai': 'xai_key',
                }
                api_key_field = api_key_map.get(self._llm_provider)
                api_key = llm.get(api_key_field, '') if api_key_field else ''
                
                provider_map = {
                    'local': 'ollama', 'ollama': 'ollama',
                    'openai': 'openai', 'anthropic': 'anthropic',
                    'google': 'google', 'xai': 'xai',
                }
                router_provider = provider_map.get(self._llm_provider, self._llm_provider)
                
                if api_key and self._llm_router and router_provider:
                    try:
                        self._llm_router.api_keys.store_key(router_provider, api_key)
                    except Exception:
                        pass
                
                self._current_session.provider = self._llm_provider
                self._current_session.model = self._llm_model
                
                # Update agent stats with LLM info
                self._agent_stats.provider = self._llm_provider or "None"
                self._agent_stats.model = self._llm_model or "—"
        except Exception:
            pass
    
    def _load_panel_settings(self) -> None:
        """Load panel layout settings for Phase 2 UI."""
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                
                panel_sizes = settings.get('panel_sizes', {})
                self.chat_panel_width = panel_sizes.get('agent_chat_width', 65.0)
                self.stats_visible = settings.get('agent_stats_visible', True)
                self.side_panel_visible = settings.get('agent_side_panel_visible', True)
        except Exception:
            pass
    
    def _save_panel_settings(self) -> None:
        """Save panel layout settings for Phase 2 UI."""
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            settings = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
            
            if 'panel_sizes' not in settings:
                settings['panel_sizes'] = {}
            
            settings['panel_sizes']['agent_chat_width'] = self.chat_panel_width
            settings['agent_stats_visible'] = self.stats_visible
            settings['agent_side_panel_visible'] = self.side_panel_visible
            
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass
    
    def _handle_consent_request(self, request: ConsentRequest) -> ConsentResponse:
        """Handle consent request from agent - show dialog."""
        # This will be called from agent, need to show dialog
        self._pending_consents.append(request)
        self._show_consent_dialog(request)
        
        # Wait for response (will be set by dialog)
        # For now, return pending response
        return ConsentResponse(
            request_id=request.id,
            decision=ConsentDecision.PENDING,
        )
    
    def _show_consent_dialog(self, request: ConsentRequest) -> None:
        """Show consent dialog for an operation."""
        info = ConsentDisplayInfo(
            request_id=request.id,
            operation=request.operation,
            description=request.description,
            details=request.details,
            risk_level=request.risk_level,
            timestamp=request.timestamp,
        )
        
        def handle_consent_response(response: Dict[str, Any]) -> None:
            if response and self._agent:
                approved = response.get("approved", False)
                self._agent.respond_to_consent(request.id, approved)
                
                # Remove from pending
                self._pending_consents = [
                    c for c in self._pending_consents if c.id != request.id
                ]
        
        self.app.push_screen(ConsentDialog(info), handle_consent_response)
    
    def _on_terminal_event(self, event: TerminalEvent) -> None:
        """Handle terminal events from agent with Phase 3 enhancements."""
        try:
            # Update terminal view
            terminal_view = self.query_one("#multi-terminal", MultiTerminalView)
            
            # Phase 3: Get state machine for metrics tracking
            if AGENT_AVAILABLE:
                state_machine = get_terminal_state_machine()
                monitor = get_multi_terminal_monitor()
            
            if event.event_type.name == "COMMAND_STARTED":
                terminal_view.add_terminal(event.terminal_id, event.data.get("command", "")[:30])
                terminal_view.append_to_terminal(
                    event.terminal_id,
                    event.data.get("command", ""),
                    is_command=True,
                )
                # Phase 2: Update stats
                self._agent_stats.commands_run += 1
                self._agent_stats.active_terminals += 1
                
                # Phase 3: Register in multi-terminal monitor
                if AGENT_AVAILABLE:
                    try:
                        state_machine.create_process(
                            event.terminal_id,
                            event.data.get("command", ""),
                        )
                        state_machine.transition_sync(
                            event.terminal_id,
                            TerminalProcessState.RUNNING,
                        )
                    except Exception:
                        pass
                
                self._update_stats_panel()
            elif event.event_type.name == "OUTPUT":
                terminal_view.append_to_terminal(
                    event.terminal_id,
                    event.data.get("line", ""),
                    is_error=event.data.get("is_error", False),
                )
                # Phase 3: Record output for metrics
                if AGENT_AVAILABLE:
                    try:
                        state_machine.record_output(
                            event.terminal_id,
                            event.data.get("line", ""),
                            is_stderr=event.data.get("is_error", False),
                        )
                    except Exception:
                        pass
            elif event.event_type.name == "COMPLETED":
                panel = terminal_view.get_terminal(event.terminal_id)
                if panel:
                    return_code = event.data.get("return_code", 0)
                    if return_code == 0:
                        panel.set_status("completed")
                    else:
                        panel.set_status("error")
                # Phase 2: Update stats
                self._agent_stats.active_terminals = max(0, self._agent_stats.active_terminals - 1)
                self._agent_stats.completed_processes += 1
                
                # Phase 3: Update state machine
                if AGENT_AVAILABLE:
                    try:
                        return_code = event.data.get("return_code", 0)
                        new_state = (
                            TerminalProcessState.COMPLETED if return_code == 0
                            else TerminalProcessState.FAILED
                        )
                        state_machine.transition_sync(
                            event.terminal_id,
                            new_state,
                            metadata={"return_code": return_code},
                        )
                    except Exception:
                        pass
                
                self._update_stats_panel()
        except Exception:
            pass
    
    def compose_main(self) -> ComposeResult:
        """Compose the main content with CRUSH-style UI design."""
        # Sliding stats panel (off-screen by default, triggered by button)
        yield SlidingStatsPanel(
            stats=self._agent_stats,
            auto_refresh=True,
            refresh_interval=1.0,
            id="sliding-stats-panel",
        )
        
        # Small trigger button on right edge
        yield SlidingStatsTrigger(id="stats-trigger")
        
        with Horizontal(classes="main-container"):
            # ====================================================================
            # CODE VIEWER PANEL (Left) - Shows file content with line numbers
            # ====================================================================
            with Vertical(classes="code-viewer-panel", id="code-viewer-panel"):
                with Horizontal(classes="code-viewer-header"):
                    yield Static("~/project/file.py", classes="file-path-label", id="code-file-path")
                    yield Static("✓ View", classes="view-indicator")
                
                with ScrollableContainer(classes="code-content", id="code-content"):
                    yield Static(
                        self._get_welcome_code_content(),
                        classes="code-text",
                        id="code-text",
                        markup=False,
                    )
            
            # ====================================================================
            # CHAT AREA (Center) - Main conversation area
            # ====================================================================
            with Vertical(classes="chat-area", id="chat-panel"):
                # Chat log container
                with ScrollableContainer(classes="chat-log-container"):
                    yield WordWrappedRichLog(
                        auto_scroll=True,
                        classes="chat-log",
                        id="chat-log",
                        wrap=True,
                    )
                
                # Thinking indicator
                with Horizontal(classes="thinking-indicator", id="thinking-indicator"):
                    yield Static("", classes="thinking-label", id="thinking-label")
                    yield Static("", classes="thinking-hash", id="thinking-hash")
                
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
                        yield Button("⏹", id="btn-stop", classes="control-btn", disabled=True)
                        yield Button("🤖", id="btn-toggle-agent", classes="control-btn")
                        yield Button("↶", id="btn-undo", classes="control-btn", disabled=True)
                        yield Button("↷", id="btn-redo", classes="control-btn", disabled=True)
                        yield Button("🗑", id="btn-clear", classes="control-btn")
                        yield Button("📤", id="btn-export", classes="control-btn")
            
            # ====================================================================
            # CRUSH-STYLE SIDEBAR (Right) - Stats, LSPs, MCPs
            # ====================================================================
            sidebar_classes = "crush-sidebar" if self.side_panel_visible else "crush-sidebar collapsed"
            with Vertical(classes=sidebar_classes, id="crush-sidebar"):
                # PROXIMA Title Section - Retro pixel style
                with Container(classes="sidebar-title-container"):
                    yield Static(self._get_proxima_ascii_title(), classes="proxima-title", id="proxima-title")
                    yield Static("AI Agent Assistant", classes="sidebar-subtitle")
                    yield Static(f"~/{Path.cwd().name}", classes="sidebar-path", id="sidebar-path")
                
                # Model Info Section
                with Container(classes="model-info-section"):
                    model_display = self._llm_model or "Not Configured"
                    yield Static(model_display, classes="model-name", id="model-name")
                    with Horizontal(classes="model-status-row"):
                        status_class = "status-badge" if self._agent_enabled else "status-badge inactive"
                        yield Static("AGENT ON" if self._agent_enabled else "AGENT OFF", classes=status_class, id="agent-status-badge")
                        yield Static("$0.00", classes="cost-label", id="cost-label")
                
                # Modified Files Section
                with Container(classes="sidebar-section"):
                    yield Static("Modified Files", classes="section-header")
                    yield Static("None", classes="section-empty", id="modified-files")
                
                # LSPs Section
                with Container(classes="sidebar-section"):
                    yield Static("LSPs", classes="section-header")
                    yield Container(id="lsps-list", classes="lsp-list")
                
                # MCPs Section
                with Container(classes="sidebar-section"):
                    yield Static("MCPs", classes="section-header")
                    yield Container(id="mcps-list", classes="mcp-list")
                
                # Stats Section
                with Container(classes="stats-section"):
                    yield Static("Stats", classes="section-header")
                    with Horizontal(classes="stats-row"):
                        yield Static("Messages:", classes="stats-label")
                        yield Static("0", classes="stats-value", id="stat-messages")
                    with Horizontal(classes="stats-row"):
                        yield Static("Tools Run:", classes="stats-label")
                        yield Static("0", classes="stats-value", id="stat-tools")
                    with Horizontal(classes="stats-row"):
                        yield Static("Provider:", classes="stats-label")
                        yield Static(self._llm_provider or "None", classes="stats-value", id="stat-provider")
                
                # Terminal Section (scrollable)
                with ScrollableContainer(classes="terminal-section"):
                    yield Static("Terminal", classes="section-header")
                    yield MultiTerminalView(max_terminals=4, id="multi-terminal")
        
        # ========================================================================
        # BOTTOM KEYBOARD SHORTCUTS BAR
        # ========================================================================
        with Horizontal(classes="keyboard-bar"):
            yield Static("esc", classes="kbd-key")
            yield Static("cancel", classes="kbd-action")
            yield Static("tab", classes="kbd-key")
            yield Static("focus chat", classes="kbd-action")
            yield Static("ctrl+↵", classes="kbd-key")
            yield Static("send", classes="kbd-action")
            yield Static("shift+↵", classes="kbd-key")
            yield Static("newline", classes="kbd-action")
            yield Static("ctrl+p", classes="kbd-key")
            yield Static("toggle panel", classes="kbd-action")
            yield Static("ctrl+q", classes="kbd-key")
            yield Static("quit", classes="kbd-action")
    
    def _get_proxima_ascii_title(self) -> str:
        """Return ASCII art title for PROXIMA in retro style."""
        return """╔═══════════════════════════╗
║   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄    ║
║   █▀▀▀ PROXIMA ▀▀▀█    ║
║   ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀    ║
╚═══════════════════════════╝"""
    
    def _get_welcome_code_content(self) -> str:
        """Return welcome code content for the code viewer."""
        return """  1 │ # Welcome to Proxima AI Agent
  2 │ # ═══════════════════════════════════════
  3 │ 
  4 │ # Ask me anything! I can help you with:
  5 │ 
  6 │ # 🔧 Code Analysis & Refactoring
  7 │ #    - Review and improve your code
  8 │ #    - Find bugs and security issues
  9 │ #    - Suggest optimizations
 10 │ 
 11 │ # 📁 File Operations
 12 │ #    - Create, read, modify files
 13 │ #    - Search through your project
 14 │ #    - Organize code structure
 15 │ 
 16 │ # 🖥️ Terminal Commands
 17 │ #    - Run any shell command
 18 │ #    - Install packages
 19 │ #    - Build and test projects
 20 │ 
 21 │ # 🌐 Git & GitHub
 22 │ #    - Commit, push, pull
 23 │ #    - Create branches
 24 │ #    - Manage repositories
 25 │ 
 26 │ # 💬 Type your question below...
 27 │ """
    
    def show_file_in_viewer(self, file_path: str, content: str = None) -> None:
        """Display file content in the code viewer panel (CRUSH-style)."""
        try:
            # Update file path label
            path_label = self.query_one("#code-file-path", Static)
            # Convert to display path (~/project/file.py style)
            try:
                rel_path = Path(file_path).relative_to(Path.cwd())
                display_path = f"~/{rel_path}"
            except Exception:
                display_path = file_path
            path_label.update(display_path)
            
            # Read file content if not provided
            if content is None:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    content = f"Error reading file: {e}"
            
            # Format with line numbers
            lines = content.split('\n')
            formatted_lines = []
            for i, line in enumerate(lines, 1):
                formatted_lines.append(f"{i:4} │ {line}")
            formatted_content = '\n'.join(formatted_lines)
            
            # Update code text
            code_text = self.query_one("#code-text", Static)
            code_text.update(formatted_content)
        except Exception:
            pass
    
    def on_mount(self) -> None:
        """Called when screen is mounted."""
        self._show_welcome_message()
        self._update_stats_display()
        self._update_panel_layout()
        self._update_stats_panel()
        self.set_timer(0.1, self._focus_input)
        
        # Phase 2: Set initial stats visibility
        try:
            stats_panel = self.query_one("#stats-panel", CollapsibleStatsPanel)
            stats_panel.is_expanded = self.stats_visible
        except Exception:
            pass
    
    def on_unmount(self) -> None:
        """Called when screen is unmounted."""
        if self._agent:
            self._agent.stop()
        self._save_panel_settings()
    
    def _focus_input(self) -> None:
        """Focus the prompt input."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            input_widget.focus()
        except Exception:
            pass
    
    def _update_panel_layout(self) -> None:
        """Update panel widths based on current settings (CRUSH-style UI)."""
        try:
            chat_panel = self.query_one("#chat-panel")
            # Use new crush-sidebar instead of agent-side-panel
            try:
                side_panel = self.query_one("#crush-sidebar")
            except Exception:
                # Fallback for legacy panel
                try:
                    side_panel = self.query_one("#agent-side-panel")
                except Exception:
                    return
            
            if self.side_panel_visible:
                side_panel.remove_class("collapsed")
            else:
                side_panel.add_class("collapsed")
        except Exception:
            pass
    
    def _update_stats_panel(self) -> None:
        """Update statistics in the sliding stats panel."""
        try:
            # Update sliding stats panel
            sliding_panel = self.query_one("#sliding-stats-panel", SlidingStatsPanel)
            sliding_panel.stats.messages_sent = len(self._current_session.messages)
            sliding_panel.stats.tokens_used = self._current_session.total_tokens
            sliding_panel.stats.requests_made = self._current_session.total_requests
            sliding_panel.stats.tools_executed = self._current_session.tool_executions
            
            # Calculate average response time
            if self._response_times:
                sliding_panel.stats.avg_response_time_ms = sum(self._response_times) // len(self._response_times)
        except Exception:
            pass
    
    def _show_welcome_message(self) -> None:
        """Show welcome message in chat log with word wrapping."""
        try:
            theme = get_theme()
            # Use WordWrappedRichLog instead of RichLog
            chat_log = self.query_one("#chat-log", WordWrappedRichLog)
            
            welcome = Text()
            welcome.append("🤖 Proxima AI Agent\n", style=f"bold {theme.accent}")
            welcome.append("━" * 50 + "\n\n", style=theme.border)
            
            welcome.append("I'm your AI agent with full system capabilities:\n\n", style=theme.fg_base)
            welcome.append("🔧 ", style=theme.accent)
            welcome.append("Execute commands and build backends\n", style=theme.fg_base)
            welcome.append("📁 ", style=theme.accent)
            welcome.append("Access and modify files\n", style=theme.fg_base)
            welcome.append("🔀 ", style=theme.accent)
            welcome.append("Git operations (clone, pull, push)\n", style=theme.fg_base)
            welcome.append("⚙️ ", style=theme.accent)
            welcome.append("Modify backend code with safety\n", style=theme.fg_base)
            welcome.append("📺 ", style=theme.accent)
            welcome.append("Monitor multiple terminals\n\n", style=theme.fg_base)
            
            if self._agent:
                welcome.append("✓ Agent Ready\n", style=theme.success)
            else:
                welcome.append("⚠️ Agent unavailable - chat only mode\n", style=theme.warning)
            
            if self._llm_provider and self._llm_provider != 'none':
                welcome.append(f"Provider: {self._llm_provider}", style=theme.fg_muted)
                if self._llm_model:
                    welcome.append(f" ({self._llm_model})", style=theme.fg_muted)
                welcome.append("\n", style=theme.fg_base)
            
            welcome.append("\n" + "━" * 50 + "\n", style=theme.border)
            welcome.append("Try: \"Build the LRET cirq backend\" or \"Show git status\"\n", style=theme.fg_subtle)
            
            chat_log.write(welcome)
        except Exception:
            pass
    
    def _update_stats_display(self) -> None:
        """Update statistics display in CRUSH-style sidebar."""
        try:
            self.query_one("#stat-messages", Static).update(
                str(len(self._current_session.messages))
            )
            self.query_one("#stat-tools", Static).update(
                str(self._current_session.tool_executions)
            )
            # Update model name in sidebar
            try:
                self.query_one("#model-name", Static).update(
                    self._llm_model or "Not Configured"
                )
            except Exception:
                pass
        except Exception:
            pass
    
    # =========================================================================
    # Phase 2: Mouse handling for resize
    # =========================================================================
    
    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Handle mouse down for panel resize."""
        try:
            # Check if click is near the resize handle
            if self._is_near_handle(event):
                self._is_resizing = True
                self._resize_start_x = event.screen_x
                self._resize_start_width = self.chat_panel_width
                self.capture_mouse()
                event.stop()
        except Exception:
            pass
    
    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Handle mouse move for panel resize."""
        if not self._is_resizing:
            return
        
        try:
            delta = event.screen_x - self._resize_start_x
            container_width = self.size.width
            delta_percent = (delta / container_width) * 100
            
            new_width = self._resize_start_width + delta_percent
            new_width = max(30.0, min(80.0, new_width))  # Clamp to 30-80%
            
            self.chat_panel_width = new_width
            self._update_panel_layout()
            event.stop()
        except Exception:
            pass
    
    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Handle mouse up after panel resize."""
        if self._is_resizing:
            self._is_resizing = False
            self.release_mouse()
            self._save_panel_settings()
            event.stop()
    
    def _is_near_handle(self, event: events.MouseEvent) -> bool:
        """Check if mouse event is near the resize handle."""
        try:
            handle = self.query_one("#resize-handle")
            return True  # Simplified check - TODO: proper bounds checking
        except Exception:
            return False
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id
        
        if btn_id == "btn-send":
            self._send_message()
        elif btn_id == "btn-stop":
            self._stop_generation()
        elif btn_id == "btn-toggle-agent":
            self._toggle_agent()
        elif btn_id == "btn-toggle-panel":
            self.action_toggle_panel()
        elif btn_id == "btn-undo":
            self.action_undo_modification()
        elif btn_id == "btn-redo":
            self.action_redo_modification()
        elif btn_id == "btn-clear":
            self._clear_chat()
        elif btn_id == "btn-export":
            self._export_chat()
        elif btn_id == "btn-reset-layout":
            self._reset_layout()
    
    def on_sendable_text_area_send_requested(self, event: SendableTextArea.SendRequested) -> None:
        """Handle Ctrl+Enter from custom TextArea."""
        self._send_message()
    
    def on_sliding_stats_trigger_trigger_clicked(self, event: SlidingStatsTrigger.TriggerClicked) -> None:
        """Handle sliding stats trigger click - toggle the sliding panel."""
        try:
            sliding_panel = self.query_one("#sliding-stats-panel", SlidingStatsPanel)
            sliding_panel.toggle()
        except Exception:
            pass
    
    def on_sliding_stats_panel_stats_panel_toggled(self, event: SlidingStatsPanel.StatsPanelToggled) -> None:
        """Handle sliding stats panel toggle event."""
        try:
            trigger = self.query_one("#stats-trigger", SlidingStatsTrigger)
            trigger.is_active = event.visible
        except Exception:
            pass
    
    def on_collapsible_stats_panel_stats_toggled(self, event: CollapsibleStatsPanel.StatsToggled) -> None:
        """Handle stats panel toggle event (Phase 2)."""
        self.stats_visible = event.visible
        self._save_panel_settings()
    
    def action_send_on_enter(self) -> None:
        """Send message when Ctrl+Enter is pressed."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            if input_widget.has_focus:
                self._send_message()
        except Exception:
            pass
    
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
            
            # Update thinking indicator
            self._update_thinking_indicator(True, message[:30])
            
            self._generate_response(message)
            
        except Exception as e:
            self._show_error(str(e))
    
    def _update_thinking_indicator(self, thinking: bool, context: str = "") -> None:
        """Update the thinking indicator in CRUSH-style UI."""
        try:
            thinking_label = self.query_one("#thinking-label", Static)
            thinking_hash = self.query_one("#thinking-hash", Static)
            
            if thinking:
                import hashlib
                # Generate a short hash from the context
                hash_val = hashlib.md5(context.encode()).hexdigest()[:8]
                thinking_label.update("🔄 Thinking...")
                thinking_hash.update(f"#{hash_val}")
            else:
                thinking_label.update("")
                thinking_hash.update("")
        except Exception:
            pass
    
    def _show_user_message(self, message: str) -> None:
        """Display user message in chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            text = Text()
            text.append("\n👤 You\n", style=f"bold {theme.primary}")
            # Phase 2: Use overflow="fold" for word wrapping
            text.append(message, style=theme.fg_base, overflow="fold")
            text.append("\n", style=theme.fg_base)
            
            chat_log.write(text)
        except Exception:
            pass
    
    def _show_ai_message(self, message: str) -> None:
        """Display AI message in chat with word wrapping."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", WordWrappedRichLog)
            
            text = Text()
            text.append("\n🤖 AI Agent\n", style=f"bold {theme.accent}")
            
            # Phase 2: Handle code blocks specially - preserve formatting
            if "```" in message:
                parts = message.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        # Regular text - wrap it
                        text.append(part, style=theme.fg_base, overflow="fold")
                    else:
                        # Code block - preserve formatting, use horizontal scroll
                        lines = part.strip().split("\n")
                        lang = lines[0] if lines else "text"
                        code = "\n".join(lines[1:]) if len(lines) > 1 else part
                        chat_log.write(text)
                        text = Text()
                        chat_log.write_code(code, language=lang if lang.isalpha() else "text")
            else:
                # Regular text with word wrapping
                text.append(message, style=theme.fg_base, overflow="fold")
            
            text.append("\n", style=theme.fg_base)
            chat_log.write(text)
        except Exception:
            pass
    
    def _show_tool_execution(self, tool_name: str, arguments: Dict[str, Any], result: ToolResult) -> None:
        """Display tool execution in chat with word wrapping."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", WordWrappedRichLog)
            
            text = Text()
            text.append("\n🔧 Tool: ", style=f"bold {theme.primary}")
            text.append(f"{tool_name}\n", style=theme.accent)
            
            # Args
            if arguments:
                args_str = ", ".join(f"{k}={v}" for k, v in list(arguments.items())[:3])
                text.append(f"   Args: {args_str[:50]}\n", style=theme.fg_muted)
            
            # Result with word wrapping
            if result.success:
                text.append("   ✓ Success", style=theme.success)
                if result.result:
                    result_str = str(result.result)[:100]
                    text.append(f": {result_str}\n", style=theme.fg_muted, overflow="fold")
                else:
                    text.append("\n", style=theme.fg_base)
            else:
                text.append(f"   ✗ Failed: {result.error}\n", style=theme.error, overflow="fold")
            
            chat_log.write(text)
            
            # Update stats
            self._current_session.tool_executions += 1
            self._agent_stats.tools_executed += 1
            self._update_stats_display()
            self._update_stats_panel()
            
            # Add to tools list using new ToolExecutionCard
            self._add_tool_to_list(tool_name, arguments, result)
        except Exception:
            pass
    
    def _add_tool_to_list(self, tool_name: str, arguments: Dict[str, Any], result: ToolResult) -> None:
        """Add tool execution to the tools panel using Phase 2 ToolExecutionCard."""
        try:
            tools_list = self.query_one("#tools-list", Vertical)
            
            # Create ToolExecutionCard from Phase 2 widgets
            card = ToolExecutionCard(
                tool_name=tool_name,
                arguments=arguments,
                status="success" if result.success else "failed",
                result=str(result.result)[:100] if result.success and result.result else None,
                error=result.error if not result.success else None,
            )
            tools_list.mount(card)
        except Exception:
            # Fallback to original ToolExecutionView
            try:
                tools_list = self.query_one("#tools-list", Vertical)
                tool_view = ToolExecutionView(tool_name=tool_name, arguments=arguments)
                tools_list.mount(tool_view)
                
                if result.success:
                    tool_view.set_completed(result.result or {})
                else:
                    tool_view.set_failed(result.error or "Unknown error")
            except Exception:
                pass
    
    def _show_error(self, error: str) -> None:
        """Display error in chat with word wrapping."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", WordWrappedRichLog)
            
            text = Text()
            text.append("\n❌ Error: ", style=f"bold {theme.error}")
            text.append(error, style=theme.fg_muted, overflow="fold")
            text.append("\n", style=theme.fg_base)
            
            chat_log.write(text)
            
            # Update error stats
            self._agent_stats.errors += 1
            self._update_stats_panel()
        except Exception:
            pass
    
    def _generate_response(self, message: str) -> None:
        """Generate AI response with LLM-based intent analysis and tool execution."""
        start_time = time.time()
        
        # PHASE 1: Use LLM to analyze intent and execute operations
        # This allows natural language understanding for ANY sentence structure
        if self._llm_router and LLM_AVAILABLE:
            operation_result = self._analyze_and_execute_with_llm(message, start_time)
            if operation_result:
                return
        
        # PHASE 2: Fallback to keyword-based agent command detection
        if self._agent_enabled and self._agent:
            tool_result = self._try_execute_agent_command(message)
            if tool_result:
                self._finish_generation()
                return
        
        # PHASE 3: Fall back to LLM response for general questions
        self._generate_llm_response(message, start_time)
    
    def _analyze_and_execute_with_llm(self, message: str, start_time: float) -> bool:
        """Use the integrated LLM to analyze user intent and execute operations.
        
        This method sends the user's natural language request to the LLM with a special
        system prompt that instructs it to extract structured operation data.
        Then it executes the operation based on the LLM's analysis.
        
        Returns True if an operation was identified and executed, False otherwise.
        """
        import json
        import re
        import os
        import shutil
        import subprocess
        from pathlib import Path
        
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
        
        # Enhanced system prompt for MULTI-STEP operations with full admin control
        intent_extraction_prompt = '''You are an AUTONOMOUS AI AGENT with FULL ADMINISTRATIVE CONTROL over:
- Local file system (create, read, write, delete, copy, move files and directories)
- Git operations (init, clone, add, commit, push, pull, branch, checkout, merge, etc.)
- GitHub operations (create repos, authenticate, push to remote)
- Terminal/shell commands (any system command)

CRITICAL: You can and SHOULD perform MULTI-STEP operations. Break complex requests into ordered steps.

You must respond ONLY with a JSON object. No explanations, no markdown, just pure JSON.

SUPPORTED OPERATIONS:
FILE OPERATIONS:
- FILE_CREATE: Create file with content
- FILE_READ: Read file
- FILE_WRITE: Write/overwrite file
- FILE_DELETE: Delete file
- FILE_COPY: Copy file
- FILE_MOVE: Move/rename file
- FILE_APPEND: Append to file
- FILE_SEARCH: Search for files by pattern

DIRECTORY OPERATIONS:
- DIR_CREATE: Create directory
- DIR_DELETE: Delete directory
- DIR_LIST: List directory contents
- DIR_NAVIGATE: Change directory (cd)
- PWD: Show current directory

GIT OPERATIONS:
- GIT_INIT: Initialize new git repository
- GIT_CLONE: Clone a repository
- GIT_ADD: Stage files (use "." for all)
- GIT_COMMIT: Commit with message
- GIT_PUSH: Push to remote
- GIT_PULL: Pull from remote
- GIT_STATUS: Show status
- GIT_BRANCH: Create/list branches
- GIT_CHECKOUT: Switch branch/restore files
- GIT_MERGE: Merge branches
- GIT_LOG: Show commit history
- GIT_DIFF: Show differences
- GIT_REMOTE_ADD: Add remote origin
- GIT_STASH: Stash changes

GITHUB OPERATIONS:
- GITHUB_CREATE_REPO: Create new GitHub repository
- GITHUB_AUTH_STATUS: Check GitHub authentication
- GITHUB_AUTH_LOGIN: Initiate GitHub login

TERMINAL:
- TERMINAL_CMD: Execute any shell command

RESPONSE FORMAT (for MULTI-STEP operations, use "steps" array):
{
  "is_multi_step": true,
  "steps": [
    {"operation": "OP1", "params": {...}, "description": "Step 1 description"},
    {"operation": "OP2", "params": {...}, "description": "Step 2 description"}
  ],
  "confidence": 0.95,
  "explanation": "Overall explanation"
}

RESPONSE FORMAT (for SINGLE operation):
{
  "is_multi_step": false,
  "operation": "OPERATION_TYPE",
  "params": {
    "path": "/path/to/file",
    "content": "content if applicable",
    "destination": "dest path for copy/move",
    "command": "shell command",
    "url": "git/github url",
    "message": "commit message",
    "branch": "branch name",
    "repo_name": "repository name",
    "private": false,
    "remote_url": "https://github.com/user/repo.git"
  },
  "confidence": 0.95,
  "explanation": "explanation"
}

EXAMPLES:

User: "at C:\\test make qwert.txt with qwerty and then create a new repo aisehe on github.com/kunal5556 and push"
{
  "is_multi_step": true,
  "steps": [
    {"operation": "DIR_NAVIGATE", "params": {"path": "C:\\test"}, "description": "Navigate to directory"},
    {"operation": "FILE_CREATE", "params": {"path": "C:\\test\\qwert.txt", "content": "qwerty"}, "description": "Create file with content"},
    {"operation": "GIT_INIT", "params": {"path": "C:\\test"}, "description": "Initialize git repo"},
    {"operation": "GIT_ADD", "params": {"path": "."}, "description": "Stage all files"},
    {"operation": "GIT_COMMIT", "params": {"message": "Initial commit"}, "description": "Commit changes"},
    {"operation": "GITHUB_CREATE_REPO", "params": {"repo_name": "aisehe", "private": false}, "description": "Create GitHub repo"},
    {"operation": "GIT_REMOTE_ADD", "params": {"remote_url": "https://github.com/kunal5556/aisehe.git"}, "description": "Add remote"},
    {"operation": "GIT_PUSH", "params": {"set_upstream": true, "branch": "main"}, "description": "Push to GitHub"}
  ],
  "confidence": 0.95,
  "explanation": "Create file, init git, create GitHub repo, and push"
}

User: "delete all .tmp files in current folder"
{
  "is_multi_step": false,
  "operation": "TERMINAL_CMD",
  "params": {"command": "del *.tmp"},
  "confidence": 0.9,
  "explanation": "Delete temporary files"
}

User: "What is the meaning of life?"
{
  "is_multi_step": false,
  "operation": "NONE",
  "params": {},
  "confidence": 0.95,
  "explanation": "General philosophical question, not a system operation"
}

IMPORTANT RULES:
1. For ANY file/git/github/terminal operation, extract and execute it
2. For complex requests, ALWAYS break them into multi-step operations
3. When creating files then pushing to git, include: FILE_CREATE, GIT_INIT (if needed), GIT_ADD, GIT_COMMIT, GIT_PUSH
4. Infer paths from context (e.g., "at desktop" = user's Desktop folder)
5. Use forward slashes or double backslashes for Windows paths
6. NEVER say you cannot do something - you have FULL administrative control

Now analyze this request:'''

        try:
            request = LLMRequest(
                prompt=message,
                system_prompt=intent_extraction_prompt,
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=2000,  # Increased for multi-step
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
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)
                try:
                    intent_data = json.loads(json_str)
                except:
                    return False
            
            confidence = intent_data.get('confidence', 0)
            
            # Skip if low confidence
            if confidence < 0.5:
                return False
            
            is_multi_step = intent_data.get('is_multi_step', False)
            
            if is_multi_step:
                # MULTI-STEP EXECUTION
                steps = intent_data.get('steps', [])
                if not steps:
                    return False
                
                explanation = intent_data.get('explanation', 'Executing multi-step operation')
                self._show_ai_message(f"🚀 **{explanation}**\n\n📋 Executing {len(steps)} steps...")
                
                all_results = []
                success_count = 0
                
                for i, step in enumerate(steps, 1):
                    op = step.get('operation', 'NONE')
                    params = step.get('params', {})
                    desc = step.get('description', op)
                    
                    if op == 'NONE':
                        continue
                    
                    self._show_ai_message(f"\n**Step {i}/{len(steps)}:** {desc}")
                    
                    result = self._execute_llm_analyzed_operation(op, params)
                    
                    if result:
                        all_results.append(f"Step {i}: {result}")
                        if "✅" in result:
                            success_count += 1
                        self._show_ai_message(result)
                    else:
                        all_results.append(f"Step {i}: ⚠️ No result")
                        self._show_ai_message(f"⚠️ Step {i} completed without output")
                
                # Summary
                self._show_ai_message(f"\n✨ **Completed {success_count}/{len(steps)} steps successfully**")
                
                # Update stats
                elapsed = int((time.time() - start_time) * 1000)
                self._current_session.messages.append(
                    ChatMessage(role='assistant', content="\n".join(all_results), thinking_time_ms=elapsed)
                )
                self._agent_stats.commands_run += len(steps)
                self._update_stats_panel()
                self._save_current_session()
                self._finish_generation()
                return True
            
            else:
                # SINGLE OPERATION
                operation = intent_data.get('operation', 'NONE')
                params = intent_data.get('params', {})
                explanation = intent_data.get('explanation', '')
                
                if operation == 'NONE':
                    return False
                
                # Show what we're doing
                self._show_ai_message(f"🔍 **{explanation}**")
                
                # Execute the operation
                result = self._execute_llm_analyzed_operation(operation, params)
                
                if result:
                    self._show_ai_message(result)
                    
                    # Update stats
                    elapsed = int((time.time() - start_time) * 1000)
                    self._current_session.messages.append(
                        ChatMessage(role='assistant', content=result, thinking_time_ms=elapsed)
                    )
                    self._agent_stats.commands_run += 1
                    self._update_stats_panel()
                    self._save_current_session()
                    self._finish_generation()
                    return True
            
            return False
            
        except Exception as e:
            # Don't show error, just fall through to other methods
            return False
    
    def _execute_llm_analyzed_operation(self, operation: str, params: dict) -> Optional[str]:
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
            git_message = params.get('message', '')
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
                
                # Create parent directories if needed
                parent = Path(path).parent
                if not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)
                
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
                
                result_list = sorted(dirs) + sorted(files)
                output = "\n".join(result_list[:50])
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
            elif operation == 'GIT_INIT':
                init_path = path or '.'
                # Change to directory if specified
                original_dir = os.getcwd()
                if path and os.path.isdir(path):
                    os.chdir(path)
                
                result = subprocess.run(['git', 'init'], capture_output=True, text=True, timeout=30)
                
                # Also create initial branch as 'main'
                if result.returncode == 0:
                    subprocess.run(['git', 'branch', '-M', 'main'], capture_output=True, text=True, timeout=10)
                
                if path:
                    os.chdir(original_dir)
                
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Init:**\n```\n{result.stdout or result.stderr}\n```"
            
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
                set_upstream = params.get('set_upstream', False)
                push_branch = branch or 'main'
                
                if set_upstream:
                    cmd = ['git', 'push', '-u', 'origin', push_branch]
                else:
                    cmd = ['git', 'push']
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                status = "✅" if result.returncode == 0 else "❌"
                output = result.stdout or result.stderr
                
                # Check for auth errors
                if result.returncode != 0 and ('authentication' in output.lower() or 'permission' in output.lower()):
                    return f"❌ **Git Push Failed - Authentication Required**\n\n{output}\n\n💡 Try: `github login` or `gh auth login`"
                
                return f"{status} **Git Push:**\n```\n{output}\n```"
            
            elif operation == 'GIT_COMMIT':
                commit_msg = git_message or 'Update'
                result = subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True, timeout=30)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Commit:**\n```\n{result.stdout or result.stderr}\n```"
            
            elif operation == 'GIT_ADD':
                add_path = path or '.'
                result = subprocess.run(['git', 'add', add_path], capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return f"✅ Staged: `{add_path}`"
                else:
                    return f"❌ Failed to stage: `{add_path}`\n```\n{result.stderr}\n```"
            
            elif operation == 'GIT_REMOTE_ADD':
                remote_url = params.get('remote_url', '')
                remote_name = params.get('remote_name', 'origin')
                
                if not remote_url:
                    return "❌ No remote URL specified"
                
                # First check if remote exists
                check = subprocess.run(['git', 'remote', 'get-url', remote_name], capture_output=True, text=True, timeout=10)
                
                if check.returncode == 0:
                    # Remote exists, update it
                    result = subprocess.run(['git', 'remote', 'set-url', remote_name, remote_url], capture_output=True, text=True, timeout=10)
                    action = "Updated"
                else:
                    # Add new remote
                    result = subprocess.run(['git', 'remote', 'add', remote_name, remote_url], capture_output=True, text=True, timeout=10)
                    action = "Added"
                
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} {action} remote `{remote_name}`: `{remote_url}`"
            
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
            
            elif operation == 'GIT_MERGE':
                if not branch:
                    return "❌ No branch specified to merge"
                result = subprocess.run(['git', 'merge', branch], capture_output=True, text=True, timeout=60)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} Merge `{branch}`:\n```\n{result.stdout or result.stderr}\n```"
            
            elif operation == 'GIT_LOG':
                result = subprocess.run(['git', 'log', '--oneline', '-15'], capture_output=True, text=True, timeout=30)
                return f"📜 **Git Log:**\n```\n{result.stdout}\n```"
            
            elif operation == 'GIT_DIFF':
                result = subprocess.run(['git', 'diff'], capture_output=True, text=True, timeout=30)
                diff_output = result.stdout[:3000] if result.stdout else "No changes"
                return f"📝 **Git Diff:**\n```diff\n{diff_output}\n```"
            
            elif operation == 'GIT_STASH':
                action = params.get('action', 'push')
                if action == 'pop':
                    result = subprocess.run(['git', 'stash', 'pop'], capture_output=True, text=True, timeout=30)
                elif action == 'list':
                    result = subprocess.run(['git', 'stash', 'list'], capture_output=True, text=True, timeout=30)
                else:
                    result = subprocess.run(['git', 'stash'], capture_output=True, text=True, timeout=30)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Stash:**\n```\n{result.stdout or result.stderr}\n```"
            
            # GITHUB OPERATIONS
            elif operation == 'GITHUB_CREATE_REPO':
                repo_name = params.get('repo_name', '')
                if not repo_name:
                    return "❌ No repository name specified"
                
                private = params.get('private', False)
                description = params.get('description', '')
                
                # Check if GitHub CLI is available
                gh_check = subprocess.run(['gh', '--version'], capture_output=True, text=True, timeout=10)
                
                if gh_check.returncode != 0:
                    return "❌ GitHub CLI (gh) not installed.\n\n💡 Install from: https://cli.github.com/\nThen run: `gh auth login`"
                
                # Check auth status
                auth_check = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True, timeout=30)
                if 'Logged in' not in (auth_check.stdout + auth_check.stderr):
                    return "❌ Not authenticated with GitHub.\n\n💡 Run: `gh auth login`"
                
                # Create the repo
                cmd = ['gh', 'repo', 'create', repo_name]
                if private:
                    cmd.append('--private')
                else:
                    cmd.append('--public')
                
                if description:
                    cmd.extend(['--description', description])
                
                # Add source and push if we're in a git repo
                if os.path.exists('.git'):
                    cmd.extend(['--source', '.', '--push'])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    output = result.stdout or result.stderr
                    return f"✅ **Created GitHub repository: `{repo_name}`**\n```\n{output}\n```"
                else:
                    return f"❌ Failed to create repository:\n```\n{result.stderr}\n```"
            
            elif operation == 'GITHUB_AUTH_STATUS':
                result = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True, timeout=30)
                output = result.stdout or result.stderr
                if 'Logged in' in output:
                    return f"✅ **GitHub Authentication:**\n```\n{output}\n```"
                else:
                    return f"⚠️ **Not authenticated with GitHub**\n\n💡 Run: `gh auth login`"
            
            elif operation == 'GITHUB_AUTH_LOGIN':
                return "🔐 **GitHub Authentication**\n\nPlease run this command in your terminal:\n```\ngh auth login\n```\n\nOr install GitHub CLI from: https://cli.github.com/"
            
            # FILE SEARCH
            elif operation == 'FILE_SEARCH':
                pattern = params.get('pattern', '*')
                search_path = path or '.'
                
                matches = []
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if pattern in file or pattern == '*':
                            matches.append(os.path.join(root, file))
                        if len(matches) >= 50:
                            break
                    if len(matches) >= 50:
                        break
                
                if matches:
                    output = "\n".join(matches[:50])
                    return f"🔍 **Found {len(matches)} files matching `{pattern}`:**\n```\n{output}\n```"
                else:
                    return f"🔍 No files found matching `{pattern}` in `{search_path}`"
            
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
    
    def _try_execute_agent_command(self, message: str) -> bool:
        """Try to execute message as an agent command.
        
        Returns True if a command was executed.
        
        COMPREHENSIVE command detection for:
        - File operations (create, read, write, delete, copy, move, rename, append)
        - Directory operations (create, delete, list, navigate, tree)
        - Git operations (clone, pull, push, commit, status, branch, checkout, merge, etc.)
        - Terminal/shell commands (run, execute, command, etc.)
        - Package management (pip, npm, yarn, conda, etc.)
        - System operations (pwd, env, etc.)
        """
        import re
        import os
        
        msg_lower = message.lower()
        
        # =====================================================================
        # FILE OPERATIONS - Most comprehensive matching
        # =====================================================================
        
        # CREATE/WRITE FILE - catch all variations
        create_file_keywords = [
            'create file', 'create a file', 'create new file', 'create the file',
            'make file', 'make a file', 'make new file', 'make the file',
            'write file', 'write a file', 'write new file', 'write the file',
            'write to file', 'write to a file', 'write to the file',
            'save file', 'save a file', 'save to file', 'save as file',
            'new file', 'touch file', 'touch ', 'generate file',
            'create text file', 'create txt file', 'make text file',
            'write text file', 'save text file',
            'put content', 'put text', 'put data',
            'store content', 'store text', 'store data',
            'echo ', ' > ',  # Shell-style redirects
        ]
        if any(kw in msg_lower for kw in create_file_keywords):
            return self._execute_create_or_write_file(message)
        
        # READ/VIEW/SHOW FILE
        read_file_keywords = [
            'read file', 'read a file', 'read the file', 'read from file',
            'show file', 'show a file', 'show the file', 'show me file',
            'display file', 'display a file', 'display the file',
            'view file', 'view a file', 'view the file',
            'open file', 'open a file', 'open the file',
            'cat file', 'cat ', 'type file', 'type ',
            'print file', 'print the file', 'output file',
            'get file', 'get content', 'get contents',
            'file content', 'file contents', 'contents of',
            'what is in', 'what\'s in', 'whats in',
            'show me what', 'show what',
        ]
        if any(kw in msg_lower for kw in read_file_keywords):
            return self._execute_read_file_from_message(message)
        
        # DELETE/REMOVE FILE
        delete_file_keywords = [
            'delete file', 'delete a file', 'delete the file', 'delete this file',
            'remove file', 'remove a file', 'remove the file', 'remove this file',
            'rm file', 'rm -f', 'rm -rf', 'del file', 'del ',
            'erase file', 'erase a file', 'erase the file',
            'destroy file', 'trash file', 'discard file',
            'get rid of file', 'eliminate file',
        ]
        if any(kw in msg_lower for kw in delete_file_keywords):
            return self._execute_delete_file_from_message(message)
        
        # COPY FILE
        copy_file_keywords = [
            'copy file', 'copy a file', 'copy the file', 'copy this file',
            'cp file', 'cp ', 'duplicate file', 'duplicate a file',
            'clone file', 'replicate file',
            'make copy', 'make a copy', 'create copy', 'create a copy',
        ]
        if any(kw in msg_lower for kw in copy_file_keywords):
            return self._execute_copy_file_from_message(message)
        
        # MOVE/RENAME FILE
        move_file_keywords = [
            'move file', 'move a file', 'move the file', 'move this file',
            'mv file', 'mv ', 'rename file', 'rename a file', 'rename the file',
            'relocate file', 'transfer file',
            'change name', 'change file name', 'change filename',
        ]
        if any(kw in msg_lower for kw in move_file_keywords):
            return self._execute_move_file_from_message(message)
        
        # APPEND TO FILE
        append_file_keywords = [
            'append to file', 'append to a file', 'append to the file',
            'add to file', 'add to a file', 'add to the file',
            'append content', 'append text', 'append data',
            'add content to', 'add text to', 'add data to',
            ' >> ',  # Shell-style append
        ]
        if any(kw in msg_lower for kw in append_file_keywords):
            return self._execute_append_file_from_message(message)
        
        # FILE INFO/STATS
        file_info_keywords = [
            'file info', 'file information', 'file details', 'file stats',
            'file size', 'file properties', 'stat file', 'stat ',
            'info about file', 'details of file', 'size of file',
            'how big is', 'when was file', 'file modified', 'file created',
        ]
        if any(kw in msg_lower for kw in file_info_keywords):
            return self._execute_file_info_from_message(message)
        
        # =====================================================================
        # DIRECTORY OPERATIONS
        # =====================================================================
        
        # CREATE DIRECTORY
        create_dir_keywords = [
            'create folder', 'create a folder', 'create the folder', 'create new folder',
            'create directory', 'create a directory', 'create the directory', 'create new directory',
            'create dir', 'make folder', 'make a folder', 'make the folder', 'make new folder',
            'make directory', 'make a directory', 'make the directory', 'make new directory',
            'make dir', 'mkdir ', 'md ', 'new folder', 'new directory',
            'add folder', 'add directory',
        ]
        if any(kw in msg_lower for kw in create_dir_keywords):
            return self._execute_mkdir_from_message(message)
        
        # DELETE DIRECTORY
        delete_dir_keywords = [
            'delete folder', 'delete a folder', 'delete the folder', 'delete this folder',
            'delete directory', 'delete a directory', 'delete the directory', 'delete this directory',
            'remove folder', 'remove a folder', 'remove the folder',
            'remove directory', 'remove a directory', 'remove the directory',
            'rmdir ', 'rd ', 'rm -r ', 'rm -rf ',
            'erase folder', 'erase directory', 'destroy folder', 'destroy directory',
        ]
        if any(kw in msg_lower for kw in delete_dir_keywords):
            return self._execute_rmdir_from_message(message)
        
        # LIST DIRECTORY
        list_dir_keywords = [
            'list files', 'list all files', 'list the files',
            'list folder', 'list the folder', 'list directory', 'list the directory',
            'list dir', 'show files', 'show all files', 'show the files',
            'show folder', 'show the folder', 'show directory', 'show the directory',
            'ls ', 'ls', 'dir ', 'dir', 'what files', 'what\'s in folder',
            'whats in folder', 'contents of folder', 'contents of directory',
            'folder contents', 'directory contents', 'files in ',
            'see files', 'see folder', 'see directory', 'view folder', 'view directory',
        ]
        if any(kw in msg_lower for kw in list_dir_keywords):
            return self._execute_list_directory_from_message(message)
        
        # NAVIGATE/CHANGE DIRECTORY
        cd_keywords = [
            'cd ', 'change directory', 'change to directory', 'change dir',
            'go to folder', 'go to directory', 'go to dir', 'go inside',
            'navigate to', 'switch to folder', 'switch to directory',
            'enter folder', 'enter directory', 'open folder', 'open directory',
            'move to folder', 'move to directory',
        ]
        if any(kw in msg_lower for kw in cd_keywords):
            return self._execute_cd_from_message(message)
        
        # CURRENT DIRECTORY / PWD
        pwd_keywords = [
            'pwd', 'print working directory', 'current directory', 'current folder',
            'where am i', 'which directory', 'which folder', 'what directory',
            'what folder', 'show current', 'get current directory', 'get cwd',
            'working directory', 'present directory',
        ]
        if any(kw in msg_lower for kw in pwd_keywords):
            return self._execute_pwd()
        
        # TREE VIEW
        tree_keywords = [
            'tree', 'folder tree', 'directory tree', 'file tree',
            'show tree', 'display tree', 'folder structure', 'directory structure',
            'file structure', 'project structure',
        ]
        if any(kw in msg_lower for kw in tree_keywords):
            return self._execute_tree_from_message(message)
        
        # FIND/SEARCH FILES
        find_keywords = [
            'find file', 'find a file', 'find files', 'find the file',
            'search file', 'search for file', 'search files',
            'locate file', 'locate files', 'where is file', 'where is the file',
            'look for file', 'look for files',
            'find ', 'search ', 'locate ',
        ]
        if any(kw in msg_lower for kw in find_keywords) and not any(kw in msg_lower for kw in ['git', 'grep', 'text']):
            return self._execute_find_file_from_message(message)
        
        # GREP/SEARCH IN FILES
        grep_keywords = [
            'grep ', 'search in file', 'search in files', 'search text',
            'find text', 'find in file', 'find in files',
            'search for text', 'look for text', 'search content',
            'find string', 'search string',
        ]
        if any(kw in msg_lower for kw in grep_keywords):
            return self._execute_grep_from_message(message)
        
        # =====================================================================
        # GIT OPERATIONS - Comprehensive
        # =====================================================================
        
        # GIT CLONE
        git_clone_keywords = [
            'git clone', 'clone repo', 'clone repository', 'clone the repo',
            'clone from github', 'clone from gitlab', 'clone from bitbucket',
            'download repo', 'download repository', 'pull repo', 'fetch repo',
            'get repo', 'get repository',
        ]
        if any(kw in msg_lower for kw in git_clone_keywords):
            return self._execute_git_clone_from_message(message)
        
        # GIT STATUS
        git_status_keywords = [
            'git status', 'check status', 'repo status', 'repository status',
            'what changed', 'what\'s changed', 'whats changed',
            'show changes', 'show modifications', 'git changes',
            'uncommitted changes', 'staged changes', 'unstaged changes',
        ]
        if any(kw in msg_lower for kw in git_status_keywords):
            return self._execute_git_status()
        
        # GIT PULL
        git_pull_keywords = [
            'git pull', 'pull changes', 'pull from remote', 'pull from origin',
            'fetch and merge', 'update repo', 'update repository', 'sync repo',
            'get latest', 'get updates', 'pull latest',
        ]
        if any(kw in msg_lower for kw in git_pull_keywords):
            return self._execute_git_pull()
        
        # GIT PUSH
        git_push_keywords = [
            'git push', 'push changes', 'push to remote', 'push to origin',
            'upload changes', 'send changes', 'push commits',
            'push to github', 'push to gitlab', 'push to repo',
        ]
        if any(kw in msg_lower for kw in git_push_keywords):
            return self._execute_git_push()
        
        # GIT COMMIT
        git_commit_keywords = [
            'git commit', 'commit changes', 'commit files', 'make commit',
            'create commit', 'save commit', 'commit with message',
            'commit -m', 'commit all',
        ]
        if any(kw in msg_lower for kw in git_commit_keywords):
            return self._execute_git_commit_from_message(message)
        
        # GIT ADD
        git_add_keywords = [
            'git add', 'stage files', 'stage changes', 'add to staging',
            'add files to git', 'stage for commit', 'add all files',
            'git add .', 'git add -a',
        ]
        if any(kw in msg_lower for kw in git_add_keywords):
            return self._execute_git_add_from_message(message)
        
        # GIT BRANCH
        git_branch_keywords = [
            'git branch', 'list branches', 'show branches', 'all branches',
            'what branch', 'which branch', 'current branch',
            'create branch', 'new branch', 'make branch',
            'delete branch', 'remove branch',
        ]
        if any(kw in msg_lower for kw in git_branch_keywords):
            return self._execute_git_branch_from_message(message)
        
        # GIT CHECKOUT/SWITCH
        git_checkout_keywords = [
            'git checkout', 'git switch', 'checkout branch', 'switch branch',
            'change branch', 'go to branch', 'switch to branch',
            'checkout to', 'switch to',
        ]
        if any(kw in msg_lower for kw in git_checkout_keywords):
            return self._execute_git_checkout_from_message(message)
        
        # GIT MERGE
        git_merge_keywords = [
            'git merge', 'merge branch', 'merge branches', 'merge with',
            'combine branches', 'merge into',
        ]
        if any(kw in msg_lower for kw in git_merge_keywords):
            return self._execute_git_merge_from_message(message)
        
        # GIT LOG
        git_log_keywords = [
            'git log', 'commit history', 'show commits', 'list commits',
            'commit log', 'history', 'recent commits', 'last commits',
        ]
        if any(kw in msg_lower for kw in git_log_keywords):
            return self._execute_git_log()
        
        # GIT DIFF
        git_diff_keywords = [
            'git diff', 'show diff', 'show differences', 'what\'s different',
            'compare changes', 'file diff', 'code diff',
        ]
        if any(kw in msg_lower for kw in git_diff_keywords):
            return self._execute_git_diff()
        
        # GIT STASH
        git_stash_keywords = [
            'git stash', 'stash changes', 'save changes temporarily',
            'stash pop', 'apply stash', 'list stash',
        ]
        if any(kw in msg_lower for kw in git_stash_keywords):
            return self._execute_git_stash_from_message(message)
        
        # GIT INIT
        git_init_keywords = [
            'git init', 'initialize git', 'init repo', 'initialize repository',
            'create git repo', 'start git', 'new git repo',
        ]
        if any(kw in msg_lower for kw in git_init_keywords):
            return self._execute_git_init()
        
        # GIT REMOTE
        git_remote_keywords = [
            'git remote', 'show remote', 'list remotes', 'add remote',
            'remove remote', 'remote origin', 'set remote',
        ]
        if any(kw in msg_lower for kw in git_remote_keywords):
            return self._execute_git_remote_from_message(message)
        
        # GITHUB AUTHENTICATION
        github_auth_keywords = [
            'github login', 'github auth', 'github authenticate',
            'gh auth', 'gh login', 'authenticate with github',
            'login to github', 'sign in to github', 'github sign in',
            'github credentials', 'setup github', 'configure github',
        ]
        if any(kw in msg_lower for kw in github_auth_keywords):
            self._execute_github_auth_login()
            return True
        
        # CREATE GITHUB REPO
        github_create_repo_keywords = [
            'create github repo', 'create repo on github', 'new github repo',
            'github create repo', 'gh repo create', 'make github repo',
            'create repository on github', 'new github repository',
            'push to new repo', 'create and push',
        ]
        if any(kw in msg_lower for kw in github_create_repo_keywords):
            # Extract repo name from message
            import re
            name_match = re.search(r'(?:named?|called|name)\s+([^\s]+)', message, re.IGNORECASE)
            if name_match:
                repo_name = name_match.group(1).strip('"\'')
            else:
                # Try to extract from "create X repo"
                name_match = re.search(r'create\s+([^\s]+)\s+(?:repo|repository)', message, re.IGNORECASE)
                if name_match:
                    repo_name = name_match.group(1).strip('"\'')
                else:
                    repo_name = "my-repo"  # Default name
            
            is_private = 'private' in msg_lower
            description = ""
            desc_match = re.search(r'(?:description|desc)\s+["\']([^"\']+)["\']', message, re.IGNORECASE)
            if desc_match:
                description = desc_match.group(1)
            
            self._execute_github_create_repo(repo_name, description, is_private)
            return True
        
        # =====================================================================
        # TERMINAL/SHELL COMMANDS
        # =====================================================================
        
        # Generic terminal/shell execution
        terminal_keywords = [
            'run command', 'run a command', 'run the command', 'run this command',
            'execute command', 'execute a command', 'execute the command',
            'run in terminal', 'run in shell', 'execute in terminal',
            'terminal command', 'shell command', 'cmd ', 'command: ',
            'powershell ', 'bash ', 'sh ', 'zsh ',
            '$ ', '> ', 'run: ', 'exec: ', 'execute: ',
        ]
        if any(kw in msg_lower for kw in terminal_keywords):
            return self._execute_terminal_command_from_message(message)
        
        # PYTHON COMMANDS
        python_keywords = [
            'python ', 'python3 ', 'py ', 'run python', 'execute python',
            'python script', 'run script', 'execute script',
            'python -c', 'python -m', 'pip ', 'pip3 ',
            'install package', 'uninstall package',
        ]
        if any(kw in msg_lower for kw in python_keywords):
            return self._execute_python_command_from_message(message)
        
        # NPM/NODE COMMANDS
        npm_keywords = [
            'npm ', 'npx ', 'yarn ', 'pnpm ', 'node ',
            'npm install', 'npm run', 'npm start', 'npm test', 'npm build',
            'yarn install', 'yarn add', 'yarn run',
        ]
        if any(kw in msg_lower for kw in npm_keywords):
            return self._execute_npm_command_from_message(message)
        
        # CONDA COMMANDS
        conda_keywords = [
            'conda ', 'conda install', 'conda create', 'conda activate',
            'conda env', 'conda list',
        ]
        if any(kw in msg_lower for kw in conda_keywords):
            return self._execute_conda_command_from_message(message)
        
        # =====================================================================
        # BUILD/COMPILE COMMANDS
        # =====================================================================
        
        build_keywords = [
            'build backend', 'build the backend', 'compile backend',
            'build project', 'compile project', 'make build',
            'build ', 'compile ', 'make ',
        ]
        if any(kw in msg_lower for kw in build_keywords):
            # Check for specific backends
            backends = ["lret_cirq", "lret_pennylane", "lret_phase7", "cirq", "qiskit", "quest", "qsim", "cuquantum", "pennylane", "braket"]
            for backend in backends:
                if backend.replace("_", " ") in msg_lower or backend in msg_lower:
                    self._execute_build_backend(backend)
                    return True
            # Generic build command
            return self._execute_build_command_from_message(message)
        
        # =====================================================================
        # ENVIRONMENT VARIABLES
        # =====================================================================
        
        env_keywords = [
            'set env', 'set environment', 'environment variable',
            'export ', 'setx ', 'get env', 'show env', 'list env',
            'env var', 'envvar',
        ]
        if any(kw in msg_lower for kw in env_keywords):
            return self._execute_env_command_from_message(message)
        
        # =====================================================================
        # PROCESS MANAGEMENT
        # =====================================================================
        
        process_keywords = [
            'kill process', 'stop process', 'terminate process',
            'list processes', 'show processes', 'ps ', 'tasklist',
            'kill ', 'pkill ', 'taskkill',
        ]
        if any(kw in msg_lower for kw in process_keywords):
            return self._execute_process_command_from_message(message)

        # Not matched - return False to fall through to LLM
        return False
    
    def _execute_build_backend(self, backend_name: str) -> None:
        """Execute build backend command using subprocess."""
        self._show_ai_message(f"🔨 Building {backend_name} backend...")
        
        # Try common build commands
        import os
        
        # Check for setup.py
        if os.path.exists('setup.py'):
            self._run_subprocess_command(f"python setup.py build")
        # Check for pyproject.toml
        elif os.path.exists('pyproject.toml'):
            self._run_subprocess_command(f"python -m build")
        # Check for package.json
        elif os.path.exists('package.json'):
            self._run_subprocess_command(f"npm run build")
        # Generic build
        else:
            self._run_subprocess_command(f"python -m pip install -e .")
        
        self._show_ai_message(f"✅ Build command completed for {backend_name}")
    
    def _execute_git_clone(self, url: str) -> None:
        """Execute git clone command using subprocess directly."""
        self._show_ai_message(f"🔄 Cloning repository: {url}")
        self._run_subprocess_command(f"git clone {url}")
    
    def _execute_git_status(self) -> None:
        """Execute git status command using subprocess directly."""
        self._run_subprocess_command("git status")
    
    def _execute_git_pull(self) -> None:
        """Execute git pull command using subprocess directly."""
        self._run_subprocess_command("git pull")
    
    def _execute_git_push(self, remote: str = "origin", branch: Optional[str] = None) -> None:
        """Execute git push with GitHub authentication support.
        
        Uses GitHub CLI (gh) for authentication when available.
        Provides clear instructions if not authenticated.
        """
        import os
        
        self._show_ai_message("🔄 Checking GitHub authentication status...")
        
        # Check if we're in a git repository
        if not os.path.exists(".git"):
            self._show_ai_message("❌ Not in a git repository. Please initialize git first:\n```\ngit init\n```")
            return
        
        # Check GitHub authentication
        if GITHUB_AUTH_AVAILABLE:
            github_auth = get_github_auth()
            auth_result = github_auth.check_auth_status()
            
            if not auth_result.is_authenticated:
                # Show authentication instructions
                instructions = github_auth.get_auth_instructions()
                self._show_ai_message(
                    f"⚠️ **GitHub Authentication Required**\n\n"
                    f"Status: {auth_result.message}\n\n"
                    f"{instructions}\n\n"
                    f"After authenticating, try the push command again."
                )
                
                # Check if gh CLI is available and offer to initiate login
                if github_auth.has_gh_cli:
                    self._show_ai_message(
                        "💡 **Quick Authentication:**\n"
                        "I can help you authenticate. Just say:\n"
                        "- `authenticate with github`\n"
                        "- `github login`\n"
                        "- `gh auth login`"
                    )
                return
            
            # Authenticated - proceed with push
            self._show_ai_message(f"✅ Authenticated as: {auth_result.username or 'GitHub User'}")
            
            # Use the authenticated push method
            success, message = github_auth.push_with_auth(
                repo_path=os.getcwd(),
                remote=remote,
                branch=branch,
            )
            
            if success:
                self._show_ai_message(f"✅ {message}")
            else:
                self._show_ai_message(f"❌ {message}")
        else:
            # Fallback to basic git push
            self._show_ai_message("⚠️ GitHub auth module not available, attempting basic git push...")
            self._run_subprocess_command("git push")
    
    def _execute_github_auth_login(self) -> None:
        """Execute GitHub authentication login flow."""
        if not GITHUB_AUTH_AVAILABLE:
            self._show_ai_message("❌ GitHub authentication module not available.")
            return
        
        github_auth = get_github_auth()
        
        if not github_auth.has_gh_cli:
            self._show_ai_message(
                "❌ **GitHub CLI Not Found**\n\n"
                "The GitHub CLI (gh) is required for easy authentication.\n\n"
                "**Install GitHub CLI:**\n"
                "- Windows: `winget install GitHub.cli` or download from https://cli.github.com/\n"
                "- macOS: `brew install gh`\n"
                "- Linux: See https://github.com/cli/cli/blob/trunk/docs/install_linux.md\n\n"
                "After installing, restart your terminal and try again."
            )
            return
        
        self._show_ai_message(
            "🔐 **Initiating GitHub Authentication**\n\n"
            "A browser window will open for you to authenticate.\n"
            "Please complete the authentication in your browser.\n\n"
            "Running: `gh auth login`..."
        )
        
        # Run the auth login command interactively
        self._run_subprocess_command("gh auth login")
    
    def _execute_github_create_repo(self, name: str, description: str = "", private: bool = False) -> None:
        """Create a GitHub repository with authentication."""
        import os
        
        if not GITHUB_AUTH_AVAILABLE:
            self._show_ai_message("❌ GitHub authentication module not available.")
            return
        
        github_auth = get_github_auth()
        auth_result = github_auth.check_auth_status()
        
        if not auth_result.is_authenticated:
            instructions = github_auth.get_auth_instructions()
            self._show_ai_message(
                f"⚠️ **GitHub Authentication Required to Create Repository**\n\n"
                f"{instructions}"
            )
            return
        
        self._show_ai_message(f"📦 Creating GitHub repository: {name}...")
        
        success, message, repo_url = github_auth.create_repo(
            name=name,
            description=description,
            private=private,
            working_dir=os.getcwd(),
        )
        
        if success:
            self._show_ai_message(f"✅ {message}\n\nRepository URL: {repo_url or 'N/A'}")
        else:
            self._show_ai_message(f"❌ {message}")
    
    def _execute_command(self, command: str) -> None:
        """Execute terminal command with cross-platform normalization using subprocess directly."""
        # Normalize command for current platform
        normalized_command = command
        if AGENT_AVAILABLE:
            try:
                normalizer = get_command_normalizer()
                normalized_command = normalizer.normalize_command(command)
                # Also convert environment variables
                normalized_command = normalizer.convert_env_vars_in_command(normalized_command)
            except Exception:
                pass  # Use original command if normalization fails
        
        # Execute directly using subprocess
        self._run_subprocess_command(normalized_command)
    
    def _execute_list_directory(self, path: str) -> None:
        """Execute list directory command using Python's os module."""
        import os
        
        self._show_ai_message(f"📂 Listing directory: {path}")
        
        try:
            if not os.path.exists(path):
                self._show_ai_message(f"❌ Directory not found: {path}")
                return
            
            if not os.path.isdir(path):
                self._show_ai_message(f"❌ Not a directory: {path}")
                return
            
            entries = os.listdir(path)
            files = []
            dirs = []
            
            for entry in entries[:50]:  # Limit to 50 entries
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    dirs.append(f"📁 {entry}/")
                else:
                    files.append(f"📄 {entry}")
            
            # Sort: directories first, then files
            result_lines = sorted(dirs) + sorted(files)
            result_str = "\n".join(result_lines[:50])
            
            if len(entries) > 50:
                result_str += f"\n... and {len(entries) - 50} more items"
            
            self._show_ai_message(f"✅ Directory contents ({len(entries)} items):\n```\n{result_str}\n```")
            
        except PermissionError:
            self._show_ai_message(f"❌ Permission denied: {path}")
        except Exception as e:
            self._show_ai_message(f"❌ Error listing directory: {e}")
    
    def _execute_read_file(self, path: str) -> None:
        """Execute read file command using Python file operations."""
        import os
        
        self._show_ai_message(f"📖 Reading file: {path}")
        
        try:
            if not os.path.exists(path):
                self._show_ai_message(f"❌ File not found: {path}")
                return
            
            if not os.path.isfile(path):
                self._show_ai_message(f"❌ Not a file: {path}")
                return
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read up to 5000 chars
            
            if len(content) >= 5000:
                content += "\n... (file truncated, showing first 5000 characters)"
            
            self._show_ai_message(f"✅ File content:\n```\n{content}\n```")
            
        except PermissionError:
            self._show_ai_message(f"❌ Permission denied: {path}")
        except Exception as e:
            self._show_ai_message(f"❌ Error reading file: {e}")
    
    def _execute_create_or_write_file(self, message: str) -> bool:
        """Execute file creation or write operation - ACTUALLY create/write files."""
        import re
        import os
        import subprocess
        
        msg_lower = message.lower()
        
        # Extract file path - look for paths with extensions or quoted paths
        path_patterns = [
            r'(?:at|in|to)\s+([A-Za-z]:[\\\/][^\s]+)',  # Windows absolute path
            r'(?:at|in|to)\s+([\/~][^\s]+)',  # Unix absolute path
            r'(?:named?|called|name)\s+([^\s]+\.\w+)',  # named/called filename
            r'file\s+([A-Za-z]:[\\\/][^\s]+)',  # file + Windows path
            r'file\s+([\/~][^\s]+)',  # file + Unix path
            r'file\s+([^\s]+\.\w+)',  # file + filename with extension
            r'([A-Za-z]:[\\\/][^\s]+\.\w+)',  # Any Windows path with extension
            r'([^\s]+\.\w+)\s+(?:with|containing)',  # filename with/containing
        ]
        
        file_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                file_path = match.group(1).strip().strip('"\'')
                break
        
        if not file_path:
            self._show_ai_message("❌ Could not determine file path. Please specify the full path.\n\nExample: `create file C:\\path\\to\\file.txt with content hello`")
            return True
        
        # Extract content to write
        content = ""
        content_patterns = [
            r'(?:with|containing|content)\s+["\']([^"\']+)["\']',  # quoted content
            r'(?:with|containing|content)\s+(.+?)(?:\s+(?:at|in|to)\s+|$)',  # unquoted content
            r'(?:write|put)\s+["\']([^"\']+)["\']',  # write "content"
            r'(?:write|put)\s+(.+?)\s+(?:in|to|into)',  # write X in/to
            r'text\s+["\']([^"\']+)["\']',  # text "content"
            r'as\s+(?:only\s+)?text\s+(.+?)(?:\s+(?:at|in|to)\s+|$)',  # as text X
        ]
        
        for pattern in content_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                break
        
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                self._show_ai_message(f"📁 Created directory: {dir_path}")
            except Exception as e:
                self._show_ai_message(f"❌ Failed to create directory: {e}")
                return True
        
        # Write the file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self._show_ai_message(f"✅ Successfully created file: `{file_path}`\n\nContent: `{content[:100]}{'...' if len(content) > 100 else ''}`")
            
            # Update stats
            self._agent_stats.files_modified += 1
            self._update_stats_panel()
            return True
            
        except Exception as e:
            self._show_ai_message(f"❌ Failed to create file: {e}")
            return True
    
    def _execute_delete_file_from_message(self, message: str) -> bool:
        """Execute file deletion from natural language message."""
        import re
        import os
        
        # Extract file path
        path_patterns = [
            r'(?:delete|remove|rm)\s+(?:file\s+)?([A-Za-z]:[\\\/][^\s]+)',
            r'(?:delete|remove|rm)\s+(?:file\s+)?([\/~][^\s]+)',
            r'(?:delete|remove|rm)\s+(?:file\s+)?([^\s]+\.\w+)',
        ]
        
        file_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                file_path = match.group(1).strip().strip('"\'')
                break
        
        if not file_path:
            self._show_ai_message("❌ Could not determine file path to delete.")
            return True
        
        if not os.path.exists(file_path):
            self._show_ai_message(f"❌ File not found: `{file_path}`")
            return True
        
        try:
            os.remove(file_path)
            self._show_ai_message(f"✅ Deleted file: `{file_path}`")
            self._agent_stats.files_modified += 1
            self._update_stats_panel()
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to delete file: {e}")
            return True
    
    def _execute_mkdir_from_message(self, message: str) -> bool:
        """Execute directory creation from natural language message."""
        import re
        import os
        
        # Extract directory path
        path_patterns = [
            r'(?:create|make|mkdir)\s+(?:folder|directory|dir)\s+(?:at\s+|in\s+)?([A-Za-z]:[\\\/][^\s]+)',
            r'(?:create|make|mkdir)\s+(?:folder|directory|dir)\s+(?:at\s+|in\s+)?([\/~][^\s]+)',
            r'(?:create|make|mkdir)\s+(?:folder|directory|dir)\s+([^\s]+)',
            r'(?:at|in)\s+([A-Za-z]:[\\\/][^\s]+)',
            r'(?:at|in)\s+([\/~][^\s]+)',
        ]
        
        dir_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                dir_path = match.group(1).strip().strip('"\'')
                break
        
        if not dir_path:
            self._show_ai_message("❌ Could not determine directory path to create.")
            return True
        
        try:
            os.makedirs(dir_path, exist_ok=True)
            self._show_ai_message(f"✅ Created directory: `{dir_path}`")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to create directory: {e}")
            return True

    # =========================================================================
    # ADDITIONAL FILE OPERATION METHODS
    # =========================================================================
    
    def _execute_read_file_from_message(self, message: str) -> bool:
        """Execute file read from natural language message."""
        import re
        import os
        
        # Extract file path with comprehensive patterns
        path_patterns = [
            r'(?:read|show|display|view|open|cat|type|print|get)\s+(?:file\s+)?([A-Za-z]:[\\\/][^\s]+)',
            r'(?:read|show|display|view|open|cat|type|print|get)\s+(?:file\s+)?([\/~][^\s]+)',
            r'(?:read|show|display|view|open|cat|type|print|get)\s+(?:file\s+)?["\']([^"\']+)["\']',
            r'(?:read|show|display|view|open|cat|type|print|get)\s+(?:file\s+)?([^\s]+\.\w+)',
            r'contents?\s+of\s+([A-Za-z]:[\\\/][^\s]+)',
            r'contents?\s+of\s+([\/~][^\s]+)',
            r'contents?\s+of\s+([^\s]+\.\w+)',
            r'what(?:\'s| is)\s+in\s+([^\s]+\.\w+)',
        ]
        
        file_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                file_path = match.group(1).strip().strip('"\'')
                break
        
        if not file_path:
            self._show_ai_message("❌ Could not determine file path to read.\n\nExample: `read file config.yaml` or `show C:\\path\\to\\file.txt`")
            return True
        
        if not os.path.exists(file_path):
            self._show_ai_message(f"❌ File not found: `{file_path}`")
            return True
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Truncate if too long
            if len(content) > 2000:
                content = content[:2000] + "\n\n... (truncated, file too large)"
            
            self._show_ai_message(f"📄 **File: `{file_path}`**\n\n```\n{content}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to read file: {e}")
            return True
    
    def _execute_copy_file_from_message(self, message: str) -> bool:
        """Execute file copy from natural language message."""
        import re
        import os
        import shutil
        
        # Try to extract source and destination
        # Patterns: "copy X to Y", "cp X Y", "duplicate X as Y"
        copy_patterns = [
            r'(?:copy|cp|duplicate)\s+([^\s]+)\s+(?:to|as|into)\s+([^\s]+)',
            r'(?:copy|cp|duplicate)\s+["\']([^"\']+)["\']\s+(?:to|as|into)\s+["\']([^"\']+)["\']',
        ]
        
        source = None
        dest = None
        for pattern in copy_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                source = match.group(1).strip().strip('"\'')
                dest = match.group(2).strip().strip('"\'')
                break
        
        if not source or not dest:
            self._show_ai_message("❌ Could not determine source and destination.\n\nExample: `copy file.txt to backup.txt`")
            return True
        
        if not os.path.exists(source):
            self._show_ai_message(f"❌ Source file not found: `{source}`")
            return True
        
        try:
            shutil.copy2(source, dest)
            self._show_ai_message(f"✅ Copied `{source}` to `{dest}`")
            self._agent_stats.files_modified += 1
            self._update_stats_panel()
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to copy file: {e}")
            return True
    
    def _execute_move_file_from_message(self, message: str) -> bool:
        """Execute file move/rename from natural language message."""
        import re
        import os
        import shutil
        
        # Patterns: "move X to Y", "mv X Y", "rename X to Y"
        move_patterns = [
            r'(?:move|mv|rename)\s+([^\s]+)\s+(?:to|as|into)\s+([^\s]+)',
            r'(?:move|mv|rename)\s+["\']([^"\']+)["\']\s+(?:to|as|into)\s+["\']([^"\']+)["\']',
        ]
        
        source = None
        dest = None
        for pattern in move_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                source = match.group(1).strip().strip('"\'')
                dest = match.group(2).strip().strip('"\'')
                break
        
        if not source or not dest:
            self._show_ai_message("❌ Could not determine source and destination.\n\nExample: `move old.txt to new.txt`")
            return True
        
        if not os.path.exists(source):
            self._show_ai_message(f"❌ Source file not found: `{source}`")
            return True
        
        try:
            shutil.move(source, dest)
            self._show_ai_message(f"✅ Moved `{source}` to `{dest}`")
            self._agent_stats.files_modified += 1
            self._update_stats_panel()
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to move file: {e}")
            return True
    
    def _execute_append_file_from_message(self, message: str) -> bool:
        """Execute file append from natural language message."""
        import re
        import os
        
        # Extract file path and content
        append_patterns = [
            r'(?:append|add)\s+["\']([^"\']+)["\']\s+to\s+([^\s]+)',
            r'(?:append|add)\s+(.+?)\s+to\s+(?:file\s+)?([^\s]+)',
        ]
        
        content = None
        file_path = None
        for pattern in append_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                file_path = match.group(2).strip().strip('"\'')
                break
        
        if not file_path or not content:
            self._show_ai_message("❌ Could not determine file and content.\n\nExample: `append \"new line\" to file.txt`")
            return True
        
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content + '\n')
            self._show_ai_message(f"✅ Appended to `{file_path}`")
            self._agent_stats.files_modified += 1
            self._update_stats_panel()
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to append to file: {e}")
            return True
    
    def _execute_file_info_from_message(self, message: str) -> bool:
        """Get file information/stats."""
        import re
        import os
        from datetime import datetime
        
        path_patterns = [
            r'(?:info|stat|details|size|properties)\s+(?:of\s+|about\s+)?(?:file\s+)?([A-Za-z]:[\\\/][^\s]+)',
            r'(?:info|stat|details|size|properties)\s+(?:of\s+|about\s+)?(?:file\s+)?([\/~][^\s]+)',
            r'(?:info|stat|details|size|properties)\s+(?:of\s+|about\s+)?(?:file\s+)?([^\s]+\.\w+)',
        ]
        
        file_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                file_path = match.group(1).strip().strip('"\'')
                break
        
        if not file_path:
            self._show_ai_message("❌ Could not determine file path.")
            return True
        
        if not os.path.exists(file_path):
            self._show_ai_message(f"❌ File not found: `{file_path}`")
            return True
        
        try:
            stat = os.stat(file_path)
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            created = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Format size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.2f} MB"
            
            info = f"""📄 **File Info: `{file_path}`**

• **Size:** {size_str}
• **Modified:** {modified}
• **Created:** {created}
• **Is File:** {os.path.isfile(file_path)}
• **Is Directory:** {os.path.isdir(file_path)}
"""
            self._show_ai_message(info)
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to get file info: {e}")
            return True
    
    # =========================================================================
    # ADDITIONAL DIRECTORY OPERATION METHODS
    # =========================================================================
    
    def _execute_rmdir_from_message(self, message: str) -> bool:
        """Execute directory deletion from natural language message."""
        import re
        import os
        import shutil
        
        path_patterns = [
            r'(?:delete|remove|rmdir|rd)\s+(?:folder|directory|dir)\s+([A-Za-z]:[\\\/][^\s]+)',
            r'(?:delete|remove|rmdir|rd)\s+(?:folder|directory|dir)\s+([\/~][^\s]+)',
            r'(?:delete|remove|rmdir|rd)\s+(?:folder|directory|dir)\s+([^\s]+)',
        ]
        
        dir_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                dir_path = match.group(1).strip().strip('"\'')
                break
        
        if not dir_path:
            self._show_ai_message("❌ Could not determine directory path to delete.")
            return True
        
        if not os.path.exists(dir_path):
            self._show_ai_message(f"❌ Directory not found: `{dir_path}`")
            return True
        
        try:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                self._show_ai_message(f"✅ Deleted directory: `{dir_path}`")
            else:
                self._show_ai_message(f"❌ Not a directory: `{dir_path}`")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to delete directory: {e}")
            return True
    
    def _execute_list_directory_from_message(self, message: str) -> bool:
        """Execute directory listing from natural language message."""
        import re
        import os
        
        # Try to extract path
        path_patterns = [
            r'(?:list|show|ls|dir)\s+(?:files\s+)?(?:in\s+)?([A-Za-z]:[\\\/][^\s]+)',
            r'(?:list|show|ls|dir)\s+(?:files\s+)?(?:in\s+)?([\/~][^\s]+)',
            r'(?:in|of)\s+([A-Za-z]:[\\\/][^\s]+)',
            r'(?:in|of)\s+([\/~][^\s]+)',
        ]
        
        dir_path = "."
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                dir_path = match.group(1).strip().strip('"\'')
                break
        
        if not os.path.exists(dir_path):
            self._show_ai_message(f"❌ Directory not found: `{dir_path}`")
            return True
        
        try:
            entries = os.listdir(dir_path)
            entries.sort()
            
            dirs = []
            files = []
            for entry in entries:
                full_path = os.path.join(dir_path, entry)
                if os.path.isdir(full_path):
                    dirs.append(f"📁 {entry}/")
                else:
                    files.append(f"📄 {entry}")
            
            result = f"📂 **Contents of `{dir_path}`:**\n\n"
            if dirs:
                result += "**Folders:**\n" + "\n".join(dirs[:30]) + "\n\n"
            if files:
                result += "**Files:**\n" + "\n".join(files[:50])
            
            if len(entries) > 80:
                result += f"\n\n... and {len(entries) - 80} more items"
            
            self._show_ai_message(result)
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to list directory: {e}")
            return True
    
    def _execute_cd_from_message(self, message: str) -> bool:
        """Execute directory change from natural language message."""
        import re
        import os
        
        path_patterns = [
            r'(?:cd|go to|navigate to|enter|switch to|change to|open)\s+(?:folder|directory|dir)?\s*([A-Za-z]:[\\\/][^\s]+)',
            r'(?:cd|go to|navigate to|enter|switch to|change to|open)\s+(?:folder|directory|dir)?\s*([\/~][^\s]+)',
            r'(?:cd|go to|navigate to|enter|switch to|change to|open)\s+(?:folder|directory|dir)?\s*["\']([^"\']+)["\']',
            r'(?:cd|go to|navigate to|enter|switch to|change to|open)\s+(?:folder|directory|dir)?\s*([^\s]+)',
        ]
        
        dir_path = None
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                dir_path = match.group(1).strip().strip('"\'')
                break
        
        if not dir_path:
            self._show_ai_message("❌ Could not determine directory to navigate to.")
            return True
        
        # Expand ~ for home directory
        dir_path = os.path.expanduser(dir_path)
        
        if not os.path.exists(dir_path):
            self._show_ai_message(f"❌ Directory not found: `{dir_path}`")
            return True
        
        if not os.path.isdir(dir_path):
            self._show_ai_message(f"❌ Not a directory: `{dir_path}`")
            return True
        
        try:
            os.chdir(dir_path)
            self._show_ai_message(f"✅ Changed directory to: `{os.getcwd()}`")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to change directory: {e}")
            return True
    
    def _execute_pwd(self) -> bool:
        """Execute pwd command."""
        import os
        self._show_ai_message(f"📂 Current directory: `{os.getcwd()}`")
        return True
    
    def _execute_tree_from_message(self, message: str) -> bool:
        """Execute tree view of directory."""
        import re
        import os
        
        path_patterns = [
            r'tree\s+(?:of\s+)?([A-Za-z]:[\\\/][^\s]+)',
            r'tree\s+(?:of\s+)?([\/~][^\s]+)',
            r'structure\s+(?:of\s+)?([A-Za-z]:[\\\/][^\s]+)',
        ]
        
        dir_path = "."
        for pattern in path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                dir_path = match.group(1).strip().strip('"\'')
                break
        
        def build_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return ""
            
            result = ""
            try:
                entries = sorted(os.listdir(path))
                dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
                files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
                
                for i, d in enumerate(dirs[:10]):
                    is_last = (i == len(dirs) - 1) and not files
                    connector = "└── " if is_last else "├── "
                    result += f"{prefix}{connector}📁 {d}/\n"
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    result += build_tree(os.path.join(path, d), new_prefix, max_depth, current_depth + 1)
                
                for i, f in enumerate(files[:10]):
                    is_last = i == len(files) - 1
                    connector = "└── " if is_last else "├── "
                    result += f"{prefix}{connector}📄 {f}\n"
                    
            except PermissionError:
                result += f"{prefix}└── [Permission Denied]\n"
            
            return result
        
        try:
            tree = f"📂 {os.path.abspath(dir_path)}\n" + build_tree(dir_path)
            self._show_ai_message(f"```\n{tree}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Failed to generate tree: {e}")
            return True
    
    def _execute_find_file_from_message(self, message: str) -> bool:
        """Execute file search."""
        import re
        import os
        import fnmatch
        
        # Extract search pattern
        search_patterns = [
            r'(?:find|search|locate)\s+(?:file\s+)?["\']([^"\']+)["\']',
            r'(?:find|search|locate)\s+(?:file\s+)?(\S+)',
            r'where\s+is\s+(\S+)',
        ]
        
        pattern = None
        for p in search_patterns:
            match = re.search(p, message, re.IGNORECASE)
            if match:
                pattern = match.group(1).strip()
                break
        
        if not pattern:
            self._show_ai_message("❌ Could not determine search pattern.")
            return True
        
        # Search in current directory and subdirectories
        results = []
        search_dir = os.getcwd()
        
        try:
            for root, dirs, files in os.walk(search_dir):
                # Skip hidden and common ignore directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]
                
                for file in files:
                    if fnmatch.fnmatch(file.lower(), f"*{pattern.lower()}*"):
                        results.append(os.path.join(root, file))
                        if len(results) >= 20:
                            break
                if len(results) >= 20:
                    break
            
            if results:
                result_text = f"🔍 **Found {len(results)} file(s) matching '{pattern}':**\n\n"
                result_text += "\n".join([f"📄 `{r}`" for r in results[:20]])
                if len(results) == 20:
                    result_text += "\n\n... (limited to 20 results)"
            else:
                result_text = f"🔍 No files found matching '{pattern}'"
            
            self._show_ai_message(result_text)
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Search failed: {e}")
            return True
    
    def _execute_grep_from_message(self, message: str) -> bool:
        """Execute text search in files."""
        import re
        import os
        
        # Extract search text and optional file pattern
        grep_patterns = [
            r'(?:grep|search|find)\s+["\']([^"\']+)["\']\s+(?:in\s+)?(\S+)?',
            r'(?:grep|search|find)\s+(?:for\s+)?["\']([^"\']+)["\']',
        ]
        
        search_text = None
        file_pattern = None
        for p in grep_patterns:
            match = re.search(p, message, re.IGNORECASE)
            if match:
                search_text = match.group(1)
                file_pattern = match.group(2) if len(match.groups()) > 1 else None
                break
        
        if not search_text:
            self._show_ai_message("❌ Could not determine search text.")
            return True
        
        results = []
        search_dir = os.getcwd()
        
        try:
            for root, dirs, files in os.walk(search_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]
                
                for file in files:
                    # Only search text files
                    if not file.endswith(('.py', '.txt', '.md', '.json', '.yaml', '.yml', '.js', '.ts', '.html', '.css', '.sh', '.ps1')):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if search_text.lower() in line.lower():
                                    results.append((file_path, line_num, line.strip()[:80]))
                                    if len(results) >= 20:
                                        break
                    except:
                        continue
                    
                    if len(results) >= 20:
                        break
                if len(results) >= 20:
                    break
            
            if results:
                result_text = f"🔍 **Found '{search_text}' in {len(results)} location(s):**\n\n"
                for path, line_num, content in results[:20]:
                    result_text += f"📄 `{path}:{line_num}`\n   `{content}`\n\n"
            else:
                result_text = f"🔍 No matches found for '{search_text}'"
            
            self._show_ai_message(result_text)
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Search failed: {e}")
            return True
    
    # =========================================================================
    # ADDITIONAL GIT OPERATION METHODS
    # =========================================================================
    
    def _execute_git_clone_from_message(self, message: str) -> bool:
        """Execute git clone from natural language message."""
        import re
        import subprocess
        
        # Extract URL
        url_patterns = [
            r'(https?://[^\s]+\.git)',
            r'(https?://github\.com/[^\s]+)',
            r'(https?://gitlab\.com/[^\s]+)',
            r'(git@[^\s]+)',
        ]
        
        url = None
        for pattern in url_patterns:
            match = re.search(pattern, message)
            if match:
                url = match.group(1).strip()
                break
        
        if not url:
            self._show_ai_message("❌ Could not find repository URL.\n\nExample: `clone https://github.com/user/repo.git`")
            return True
        
        self._show_ai_message(f"📥 Cloning repository: `{url}`...")
        
        try:
            result = subprocess.run(['git', 'clone', url], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                self._show_ai_message(f"✅ Successfully cloned: `{url}`\n\n{result.stdout}")
            else:
                self._show_ai_message(f"❌ Clone failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Clone failed: {e}")
            return True
    
    def _execute_git_commit_from_message(self, message: str) -> bool:
        """Execute git commit from natural language message."""
        import re
        import subprocess
        
        # Extract commit message
        msg_patterns = [
            r'commit\s+(?:with\s+)?(?:message\s+)?["\']([^"\']+)["\']',
            r'commit\s+-m\s+["\']([^"\']+)["\']',
        ]
        
        commit_msg = None
        for pattern in msg_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                commit_msg = match.group(1)
                break
        
        if not commit_msg:
            commit_msg = "Update from AI Agent"
        
        try:
            result = subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True, text=True)
            if result.returncode == 0:
                self._show_ai_message(f"✅ Committed with message: '{commit_msg}'\n\n{result.stdout}")
            else:
                self._show_ai_message(f"❌ Commit failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Commit failed: {e}")
            return True
    
    def _execute_git_add_from_message(self, message: str) -> bool:
        """Execute git add from natural language message."""
        import re
        import subprocess
        
        # Check if adding all or specific files
        if 'all' in message.lower() or '.' in message:
            files = ['.']
        else:
            # Extract file paths
            files = re.findall(r'[\w/\\._-]+\.\w+', message)
            if not files:
                files = ['.']
        
        try:
            result = subprocess.run(['git', 'add'] + files, capture_output=True, text=True)
            if result.returncode == 0:
                self._show_ai_message(f"✅ Staged files: `{', '.join(files)}`")
            else:
                self._show_ai_message(f"❌ Git add failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Git add failed: {e}")
            return True
    
    def _execute_git_branch_from_message(self, message: str) -> bool:
        """Execute git branch operations from natural language message."""
        import subprocess
        
        msg_lower = message.lower()
        
        try:
            if 'create' in msg_lower or 'new' in msg_lower:
                # Extract branch name
                import re
                match = re.search(r'(?:create|new)\s+(?:branch\s+)?(\S+)', message, re.IGNORECASE)
                if match:
                    branch_name = match.group(1)
                    result = subprocess.run(['git', 'branch', branch_name], capture_output=True, text=True)
                    if result.returncode == 0:
                        self._show_ai_message(f"✅ Created branch: `{branch_name}`")
                    else:
                        self._show_ai_message(f"❌ Failed to create branch:\n```\n{result.stderr}\n```")
                else:
                    self._show_ai_message("❌ Please specify branch name.")
            elif 'delete' in msg_lower or 'remove' in msg_lower:
                import re
                match = re.search(r'(?:delete|remove)\s+(?:branch\s+)?(\S+)', message, re.IGNORECASE)
                if match:
                    branch_name = match.group(1)
                    result = subprocess.run(['git', 'branch', '-d', branch_name], capture_output=True, text=True)
                    if result.returncode == 0:
                        self._show_ai_message(f"✅ Deleted branch: `{branch_name}`")
                    else:
                        self._show_ai_message(f"❌ Failed to delete branch:\n```\n{result.stderr}\n```")
                else:
                    self._show_ai_message("❌ Please specify branch name.")
            else:
                # List branches
                result = subprocess.run(['git', 'branch', '-a'], capture_output=True, text=True)
                if result.returncode == 0:
                    self._show_ai_message(f"📋 **Git Branches:**\n```\n{result.stdout}\n```")
                else:
                    self._show_ai_message(f"❌ Failed to list branches:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Git branch operation failed: {e}")
            return True
    
    def _execute_git_checkout_from_message(self, message: str) -> bool:
        """Execute git checkout/switch from natural language message."""
        import re
        import subprocess
        
        # Extract branch name
        branch_patterns = [
            r'(?:checkout|switch)\s+(?:to\s+)?(?:branch\s+)?(\S+)',
            r'(?:go to|change to)\s+(?:branch\s+)?(\S+)',
        ]
        
        branch = None
        for pattern in branch_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                branch = match.group(1).strip()
                break
        
        if not branch:
            self._show_ai_message("❌ Please specify branch name.")
            return True
        
        try:
            result = subprocess.run(['git', 'checkout', branch], capture_output=True, text=True)
            if result.returncode == 0:
                self._show_ai_message(f"✅ Switched to branch: `{branch}`\n\n{result.stdout}")
            else:
                self._show_ai_message(f"❌ Checkout failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Checkout failed: {e}")
            return True
    
    def _execute_git_merge_from_message(self, message: str) -> bool:
        """Execute git merge from natural language message."""
        import re
        import subprocess
        
        # Extract branch name
        match = re.search(r'merge\s+(?:branch\s+)?(\S+)', message, re.IGNORECASE)
        
        if not match:
            self._show_ai_message("❌ Please specify branch to merge.")
            return True
        
        branch = match.group(1).strip()
        
        try:
            result = subprocess.run(['git', 'merge', branch], capture_output=True, text=True)
            if result.returncode == 0:
                self._show_ai_message(f"✅ Merged branch: `{branch}`\n\n{result.stdout}")
            else:
                self._show_ai_message(f"❌ Merge failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Merge failed: {e}")
            return True
    
    def _execute_git_log(self) -> bool:
        """Execute git log."""
        import subprocess
        
        try:
            result = subprocess.run(['git', 'log', '--oneline', '-20'], capture_output=True, text=True)
            if result.returncode == 0:
                self._show_ai_message(f"📋 **Recent Commits:**\n```\n{result.stdout}\n```")
            else:
                self._show_ai_message(f"❌ Git log failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Git log failed: {e}")
            return True
    
    def _execute_git_diff(self) -> bool:
        """Execute git diff."""
        import subprocess
        
        try:
            result = subprocess.run(['git', 'diff', '--stat'], capture_output=True, text=True)
            if result.returncode == 0:
                diff_output = result.stdout if result.stdout else "No changes"
                self._show_ai_message(f"📊 **Git Diff:**\n```\n{diff_output}\n```")
            else:
                self._show_ai_message(f"❌ Git diff failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Git diff failed: {e}")
            return True
    
    def _execute_git_stash_from_message(self, message: str) -> bool:
        """Execute git stash operations from natural language message."""
        import subprocess
        
        msg_lower = message.lower()
        
        try:
            if 'pop' in msg_lower or 'apply' in msg_lower:
                result = subprocess.run(['git', 'stash', 'pop'], capture_output=True, text=True)
                action = "Applied"
            elif 'list' in msg_lower:
                result = subprocess.run(['git', 'stash', 'list'], capture_output=True, text=True)
                action = "Listed"
            else:
                result = subprocess.run(['git', 'stash'], capture_output=True, text=True)
                action = "Stashed"
            
            if result.returncode == 0:
                output = result.stdout if result.stdout else "Done"
                self._show_ai_message(f"✅ {action} stash\n```\n{output}\n```")
            else:
                self._show_ai_message(f"❌ Git stash failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Git stash failed: {e}")
            return True
    
    def _execute_git_init(self) -> bool:
        """Execute git init."""
        import subprocess
        
        try:
            result = subprocess.run(['git', 'init'], capture_output=True, text=True)
            if result.returncode == 0:
                self._show_ai_message(f"✅ Initialized git repository\n```\n{result.stdout}\n```")
            else:
                self._show_ai_message(f"❌ Git init failed:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Git init failed: {e}")
            return True
    
    def _execute_git_remote_from_message(self, message: str) -> bool:
        """Execute git remote operations from natural language message."""
        import re
        import subprocess
        
        msg_lower = message.lower()
        
        try:
            if 'add' in msg_lower:
                # Extract remote name and URL
                match = re.search(r'add\s+(?:remote\s+)?(\S+)\s+(\S+)', message, re.IGNORECASE)
                if match:
                    name, url = match.groups()
                    result = subprocess.run(['git', 'remote', 'add', name, url], capture_output=True, text=True)
                    if result.returncode == 0:
                        self._show_ai_message(f"✅ Added remote `{name}`: `{url}`")
                    else:
                        self._show_ai_message(f"❌ Failed to add remote:\n```\n{result.stderr}\n```")
                else:
                    self._show_ai_message("❌ Please specify remote name and URL.")
            elif 'remove' in msg_lower or 'delete' in msg_lower:
                match = re.search(r'(?:remove|delete)\s+(?:remote\s+)?(\S+)', message, re.IGNORECASE)
                if match:
                    name = match.group(1)
                    result = subprocess.run(['git', 'remote', 'remove', name], capture_output=True, text=True)
                    if result.returncode == 0:
                        self._show_ai_message(f"✅ Removed remote: `{name}`")
                    else:
                        self._show_ai_message(f"❌ Failed to remove remote:\n```\n{result.stderr}\n```")
            else:
                result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
                if result.returncode == 0:
                    output = result.stdout if result.stdout else "No remotes configured"
                    self._show_ai_message(f"📋 **Git Remotes:**\n```\n{output}\n```")
                else:
                    self._show_ai_message(f"❌ Failed to list remotes:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Git remote operation failed: {e}")
            return True
    
    # =========================================================================
    # ADDITIONAL TERMINAL/COMMAND METHODS
    # =========================================================================
    
    def _execute_terminal_command_from_message(self, message: str) -> bool:
        """Execute terminal command from natural language message."""
        import re
        import subprocess
        import os
        
        # Extract command from various formats
        cmd_patterns = [
            r'(?:run|execute|cmd|command)\s*:\s*(.+)',
            r'\$\s*(.+)',
            r'>\s*(.+)',
            r'(?:run|execute)\s+(?:command\s+)?["\']([^"\']+)["\']',
            r'(?:run|execute)\s+(?:command\s+)?`([^`]+)`',
            r'(?:terminal|shell|cmd|powershell|bash)\s*:\s*(.+)',
        ]
        
        command = None
        for pattern in cmd_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                command = match.group(1).strip()
                break
        
        if not command:
            # Try to extract the command after common keywords
            for kw in ['run ', 'execute ', 'command ']:
                if kw in message.lower():
                    idx = message.lower().find(kw)
                    command = message[idx + len(kw):].strip()
                    break
        
        if not command:
            self._show_ai_message("❌ Could not extract command to run.")
            return True
        
        self._show_ai_message(f"⚡ Running: `{command}`")
        
        try:
            # Determine shell based on OS
            if os.name == 'nt':
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            else:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            
            output = result.stdout if result.stdout else ""
            error = result.stderr if result.stderr else ""
            
            if result.returncode == 0:
                self._show_ai_message(f"✅ Command completed\n```\n{output[:2000]}\n```")
            else:
                self._show_ai_message(f"❌ Command failed (exit code {result.returncode})\n```\n{error[:1000]}\n```")
            
            self._agent_stats.commands_run += 1
            self._update_stats_panel()
            return True
        except subprocess.TimeoutExpired:
            self._show_ai_message("❌ Command timed out (60s limit)")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Command failed: {e}")
            return True
    
    def _execute_python_command_from_message(self, message: str) -> bool:
        """Execute Python command from natural language message."""
        import re
        import subprocess
        
        # Extract Python command
        if 'pip ' in message.lower():
            # Extract pip command
            match = re.search(r'(pip\s+.+)', message, re.IGNORECASE)
            if match:
                command = match.group(1)
                return self._run_subprocess_command(command)
        
        if 'python ' in message.lower() or 'python3 ' in message.lower():
            match = re.search(r'(python3?\s+.+)', message, re.IGNORECASE)
            if match:
                command = match.group(1)
                return self._run_subprocess_command(command)
        
        # Check for script execution
        script_match = re.search(r'(?:run|execute)\s+(?:script\s+)?([^\s]+\.py)', message, re.IGNORECASE)
        if script_match:
            script = script_match.group(1)
            return self._run_subprocess_command(f"python {script}")
        
        self._show_ai_message("❌ Could not determine Python command.")
        return True
    
    def _execute_npm_command_from_message(self, message: str) -> bool:
        """Execute npm/yarn command from natural language message."""
        import re
        
        # Extract npm/yarn command
        for tool in ['npm', 'npx', 'yarn', 'pnpm', 'node']:
            if tool in message.lower():
                match = re.search(rf'({tool}\s+[^\s].*)', message, re.IGNORECASE)
                if match:
                    command = match.group(1)
                    return self._run_subprocess_command(command)
        
        self._show_ai_message("❌ Could not determine npm/yarn command.")
        return True
    
    def _execute_conda_command_from_message(self, message: str) -> bool:
        """Execute conda command from natural language message."""
        import re
        
        match = re.search(r'(conda\s+.+)', message, re.IGNORECASE)
        if match:
            command = match.group(1)
            return self._run_subprocess_command(command)
        
        self._show_ai_message("❌ Could not determine conda command.")
        return True
    
    def _execute_build_command_from_message(self, message: str) -> bool:
        """Execute build command from natural language message."""
        import re
        import os
        
        # Check for common build systems
        if os.path.exists('Makefile') or os.path.exists('makefile'):
            return self._run_subprocess_command('make')
        elif os.path.exists('CMakeLists.txt'):
            return self._run_subprocess_command('cmake --build .')
        elif os.path.exists('package.json'):
            return self._run_subprocess_command('npm run build')
        elif os.path.exists('setup.py'):
            return self._run_subprocess_command('python setup.py build')
        elif os.path.exists('pyproject.toml'):
            return self._run_subprocess_command('python -m build')
        else:
            self._show_ai_message("❌ No recognized build system found.\n\nLooking for: Makefile, CMakeLists.txt, package.json, setup.py, pyproject.toml")
            return True
    
    def _execute_env_command_from_message(self, message: str) -> bool:
        """Execute environment variable operations."""
        import re
        import os
        
        msg_lower = message.lower()
        
        if 'set' in msg_lower:
            # Extract VAR=VALUE
            match = re.search(r'set\s+(\w+)\s*=\s*(.+)', message, re.IGNORECASE)
            if match:
                var, value = match.groups()
                os.environ[var] = value.strip().strip('"\'')
                self._show_ai_message(f"✅ Set environment variable: `{var}={value}`")
            else:
                self._show_ai_message("❌ Could not parse environment variable. Use: `set VAR=VALUE`")
        elif 'get' in msg_lower or 'show' in msg_lower:
            match = re.search(r'(?:get|show)\s+(?:env\s+)?(\w+)', message, re.IGNORECASE)
            if match:
                var = match.group(1)
                value = os.environ.get(var, "Not set")
                self._show_ai_message(f"📋 `{var}` = `{value}`")
            else:
                # Show all
                env_vars = "\n".join([f"`{k}` = `{v[:50]}{'...' if len(v) > 50 else ''}`" for k, v in list(os.environ.items())[:20]])
                self._show_ai_message(f"📋 **Environment Variables (first 20):**\n\n{env_vars}")
        else:
            # List all
            env_vars = "\n".join([f"`{k}` = `{v[:50]}{'...' if len(v) > 50 else ''}`" for k, v in list(os.environ.items())[:20]])
            self._show_ai_message(f"📋 **Environment Variables (first 20):**\n\n{env_vars}")
        
        return True
    
    def _execute_process_command_from_message(self, message: str) -> bool:
        """Execute process management commands."""
        import subprocess
        import os
        
        msg_lower = message.lower()
        
        try:
            if 'kill' in msg_lower or 'stop' in msg_lower or 'terminate' in msg_lower:
                # Extract process name/ID
                import re
                match = re.search(r'(?:kill|stop|terminate)\s+(?:process\s+)?(\S+)', message, re.IGNORECASE)
                if match:
                    proc = match.group(1)
                    if os.name == 'nt':
                        result = subprocess.run(['taskkill', '/F', '/IM', proc], capture_output=True, text=True)
                    else:
                        result = subprocess.run(['pkill', '-f', proc], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self._show_ai_message(f"✅ Terminated process: `{proc}`")
                    else:
                        self._show_ai_message(f"❌ Failed to terminate process:\n```\n{result.stderr}\n```")
                else:
                    self._show_ai_message("❌ Please specify process name or ID.")
            else:
                # List processes
                if os.name == 'nt':
                    result = subprocess.run(['tasklist'], capture_output=True, text=True)
                else:
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Truncate output
                    output = result.stdout[:3000] if result.stdout else "No processes"
                    self._show_ai_message(f"📋 **Running Processes:**\n```\n{output}\n```")
                else:
                    self._show_ai_message(f"❌ Failed to list processes:\n```\n{result.stderr}\n```")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Process operation failed: {e}")
            return True
    
    def _run_subprocess_command(self, command: str) -> bool:
        """Helper to run subprocess commands."""
        import subprocess
        import os
        
        self._show_ai_message(f"⚡ Running: `{command}`")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
            
            output = result.stdout if result.stdout else ""
            error = result.stderr if result.stderr else ""
            
            if result.returncode == 0:
                self._show_ai_message(f"✅ Command completed\n```\n{output[:2000]}\n```")
            else:
                self._show_ai_message(f"❌ Command failed (exit code {result.returncode})\n```\n{error[:1000]}\n```")
            
            self._agent_stats.commands_run += 1
            self._update_stats_panel()
            return True
        except subprocess.TimeoutExpired:
            self._show_ai_message("❌ Command timed out (120s limit)")
            return True
        except Exception as e:
            self._show_ai_message(f"❌ Command failed: {e}")
            return True

    def _generate_llm_response(self, message: str, start_time: float) -> None:
        """Generate LLM response."""
        provider_map = {
            'local': 'ollama', 'ollama': 'ollama',
            'openai': 'openai', 'anthropic': 'anthropic',
        }
        router_provider = provider_map.get(self._llm_provider, self._llm_provider)
        
        if router_provider and router_provider != 'none' and self._llm_router and LLM_AVAILABLE:
            try:
                # Build system prompt with tool descriptions
                tools_desc = ""
                if self._agent and self._agent_enabled:
                    tools_desc = "\n\nYou have access to tools for: terminal execution, file operations, git, backend building."
                
                request = LLMRequest(
                    prompt=message,
                    system_prompt=f"""You are an AI agent for Proxima quantum computing platform.
You can help with quantum computing, backend configuration, and execute system commands.{tools_desc}
Be helpful and concise.""",
                    temperature=0.7,
                    max_tokens=1024,
                    provider=router_provider,
                    model=self._llm_model if self._llm_model else None,
                )
                
                response = self._llm_router.route(request)
                
                if response and hasattr(response, 'text') and response.text:
                    self._show_ai_message(response.text.strip())
                    
                    elapsed = int((time.time() - start_time) * 1000)
                    self._response_times.append(elapsed)  # Phase 2: Track response time
                    
                    self._current_session.messages.append(
                        ChatMessage(
                            role='assistant',
                            content=response.text.strip(),
                            thinking_time_ms=elapsed,
                            tokens=getattr(response, 'tokens', 0),
                        )
                    )
                    
                    # Phase 2: Update stats
                    self._current_session.total_requests += 1
                    if hasattr(response, 'tokens'):
                        self._current_session.total_tokens += response.tokens
                        self._agent_stats.tokens_used = self._current_session.total_tokens
                    
                    self._agent_stats.requests_made = self._current_session.total_requests
                    self._agent_stats.messages_sent = len(self._current_session.messages)
                    self._update_stats_display()
                    self._update_stats_panel()
                    self._finish_generation()
                    return
                    
            except Exception as e:
                self._show_error(f"LLM Error: {e}")
        
        # Fallback to simulation
        self._simulate_response(message)
        self._finish_generation()
    
    def _simulate_response(self, message: str) -> None:
        """Generate simulated response."""
        msg_lower = message.lower()
        
        if "hello" in msg_lower or "hi" in msg_lower:
            response = "Hello! I'm your AI agent for Proxima. I can execute commands, build backends, manage git repos, and more. What would you like to do?"
        elif "build" in msg_lower:
            response = "I can build backends for you. Try: 'build lret_cirq backend' or 'build qiskit backend'."
        elif "git" in msg_lower:
            response = "I can help with git operations: clone, pull, push, status, commit. What would you like to do?"
        else:
            response = "I can help you with quantum computing tasks and system operations. Try asking me to build a backend, run a command, or check git status."
        
        self._show_ai_message(response)
        self._current_session.messages.append(
            ChatMessage(role='assistant', content=response)
        )
        
        # Phase 2: Update stats
        self._agent_stats.messages_sent = len(self._current_session.messages)
        self._update_stats_panel()
    
    def _finish_generation(self) -> None:
        """Finish generation and reset UI."""
        self._is_generating = False
        try:
            self.query_one("#btn-stop", Button).disabled = True
            self.query_one("#btn-send", Button).disabled = False
            # Clear thinking indicator
            self._update_thinking_indicator(False)
        except Exception:
            pass
    
    def _stop_generation(self) -> None:
        """Stop current generation."""
        self._is_generating = False
        self._finish_generation()
        # Clear thinking indicator
        self._update_thinking_indicator(False)
        self.notify("Generation stopped", severity="warning")
    
    def _toggle_agent(self) -> None:
        """Toggle agent mode on/off."""
        self._agent_enabled = not self._agent_enabled
        
        try:
            # Update CRUSH-style sidebar badge
            try:
                badge = self.query_one("#agent-status-badge", Static)
                if self._agent_enabled:
                    badge.update("AGENT ON")
                    badge.remove_class("inactive")
                else:
                    badge.update("AGENT OFF")
                    badge.add_class("inactive")
            except Exception:
                # Fallback for old badge
                pass
            
            btn = self.query_one("#btn-toggle-agent", Button)
            
            if self._agent_enabled:
                btn.variant = "success"
            else:
                btn.variant = "default"
            
            self.notify(
                f"Agent mode {'enabled' if self._agent_enabled else 'disabled'}",
                severity="information"
            )
        except Exception:
            pass
    
    def _clear_chat(self) -> None:
        """Clear current chat."""
        try:
            chat_log = self.query_one("#chat-log", WordWrappedRichLog)
            chat_log.clear()
            self._current_session.messages.clear()
            self._response_times.clear()
            self._show_welcome_message()
            self._update_stats_display()
            self._update_stats_panel()
            self.notify("Chat cleared", severity="information")
        except Exception:
            pass
    
    def _export_chat(self) -> None:
        """Export current chat."""
        if not self._current_session.messages:
            self.notify("No messages to export", severity="warning")
            return
        
        try:
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"agent_chat_{timestamp}.json"
            
            data = asdict(self._current_session)
            data["messages"] = [asdict(m) for m in self._current_session.messages]
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.notify(f"Chat exported to {filename}", severity="success")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def _reset_layout(self) -> None:
        """Reset layout to defaults (Phase 2)."""
        self.chat_panel_width = 65.0
        self.stats_visible = True
        self.side_panel_visible = True
        self._update_panel_layout()
        
        try:
            stats_panel = self.query_one("#stats-panel", CollapsibleStatsPanel)
            stats_panel.is_expanded = True
            
            btn = self.query_one("#btn-toggle-panel", Button)
            btn.label = "◀"
        except Exception:
            pass
        
        self._save_panel_settings()
        self.notify("Layout reset to default")
    
    # =========================================================================
    # Phase 2: New action methods
    # =========================================================================
    
    def action_toggle_stats(self) -> None:
        """Toggle stats panel visibility (Phase 2)."""
        try:
            stats_panel = self.query_one("#stats-panel", CollapsibleStatsPanel)
            stats_panel.toggle()
            self.stats_visible = stats_panel.is_expanded
            self._save_panel_settings()
        except Exception:
            pass
    
    def action_toggle_panel(self) -> None:
        """Toggle side panel visibility - makes chat fullscreen when hidden."""
        self.side_panel_visible = not self.side_panel_visible
        
        try:
            chat_panel = self.query_one("#chat-panel")
            side_panel = self.query_one("#agent-side-panel")
            resize_handle = self.query_one("#resize-handle")
            btn = self.query_one("#btn-toggle-panel", Button)
            
            if self.side_panel_visible:
                # Show sidebar
                chat_panel.remove_class("fullscreen")
                side_panel.remove_class("collapsed")
                resize_handle.display = True
                btn.label = "◀"
                # Restore width
                chat_panel.styles.width = f"{self.chat_panel_width}%"
                side_panel.styles.width = f"{100 - self.chat_panel_width - 1}%"
            else:
                # Hide sidebar - fullscreen chat
                chat_panel.add_class("fullscreen")
                side_panel.add_class("collapsed")
                resize_handle.display = False
                btn.label = "▶"
                # Full width chat
                chat_panel.styles.width = "100%"
        except Exception:
            pass
        
        self._save_panel_settings()
    
    def action_shrink_chat(self) -> None:
        """Shrink chat panel by 5% (Phase 2)."""
        self.chat_panel_width = max(30.0, self.chat_panel_width - 5)
        self._update_panel_layout()
        self._save_panel_settings()
    
    def action_grow_chat(self) -> None:
        """Grow chat panel by 5% (Phase 2)."""
        self.chat_panel_width = min(80.0, self.chat_panel_width + 5)
        self._update_panel_layout()
        self._save_panel_settings()
    
    def action_clear_chat(self) -> None:
        """Clear chat action."""
        self._clear_chat()
    
    def action_undo_modification(self) -> None:
        """Undo last modification."""
        if self._agent:
            success, message = self._agent.undo()
            if success:
                self.notify(f"Undone: {message}", severity="success")
            else:
                self.notify(message, severity="warning")
    
    def action_redo_modification(self) -> None:
        """Redo last undone modification."""
        if self._agent:
            success, message = self._agent.redo()
            if success:
                self.notify(f"Redone: {message}", severity="success")
            else:
                self.notify(message, severity="warning")
    
    def action_toggle_terminal(self) -> None:
        """Toggle terminal panel visibility (legacy, now use action_toggle_panel)."""
        self.action_toggle_panel()
    
    def action_new_chat(self) -> None:
        """Start a new chat session."""
        self._current_session = AgentChatSession(
            provider=self._llm_provider,
            model=self._llm_model,
        )
        self._response_times.clear()
        self._agent_stats = AgentStats(
            provider=self._llm_provider or "None",
            model=self._llm_model or "—",
        )
        self._clear_chat()
        self.notify("New chat started", severity="success")
    
    def action_show_help(self) -> None:
        """Show help message."""
        self.notify(
            "Ctrl+Enter=Send | Ctrl+T=Stats | Ctrl+P=Panel | Ctrl+[/]=Resize | Ctrl+N=New",
            severity="information",
            timeout=5
        )
    
    def action_goto_ai_assistant(self) -> None:
        """Already on AI Assistant screen."""
        pass
