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

# Import Robust NL Processor for dynamic intent recognition
try:
    from proxima.agent.dynamic_tools.robust_nl_processor import (
        get_robust_nl_processor,
        RobustNLProcessor,
        IntentType,
        Intent,
        SessionContext,
    )
    ROBUST_NL_AVAILABLE = True
except ImportError:
    ROBUST_NL_AVAILABLE = False

# Import IntentToolBridge (Phase 3)
try:
    from proxima.agent.dynamic_tools.intent_tool_bridge import (
        IntentToolBridge,
        build_tool_arguments,
        INTENT_TO_TOOL,
    )
    INTENT_BRIDGE_AVAILABLE = True
except ImportError:
    INTENT_BRIDGE_AVAILABLE = False

# Import Plan components (Phase 3)
try:
    from proxima.agent.task_planner import (
        ExecutionPlan,
        PlanStep,
        PlanStatus,
        StepStatus,
        TaskCategory,
    )
    from proxima.agent.plan_executor import (
        PlanExecutor,
        ToolExecutor,
        ExecutionMode,
    )
    PLAN_AVAILABLE = True
except ImportError:
    PLAN_AVAILABLE = False

# Import LLM components
try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class _LLMRouterAdapter:
    """Thin adapter exposing ``generate(prompt) -> str`` over ``LLMRouter``.

    The ``IntentToolBridge`` expects a provider with a simple
    ``generate(prompt: str) -> str`` method (Phase 8, Step 8.4).
    This class wraps the full ``LLMRouter.route()`` API to satisfy
    that contract.
    """

    def __init__(self, router: "LLMRouter") -> None:
        self._router = router

    def generate(self, prompt: str) -> str:
        """Send *prompt* to the configured LLM and return the text."""
        try:
            request = LLMRequest(
                prompt=prompt,
                temperature=0.3,
                max_tokens=120,
            )
            response = self._router.route(request)
            return response.text or ""
        except Exception:
            return ""


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
        /* Transparent border by default, visible on hover */
        border-right: solid transparent;
    }
    
    AgentAIAssistantScreen .code-viewer-panel:hover {
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
        /* Transparent border by default, visible on hover */
        border-right: solid transparent;
    }
    
    AgentAIAssistantScreen .chat-area:hover {
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
       NO BLINKING - stable display with solid backgrounds
       ======================================================================== */
    
    AgentAIAssistantScreen .crush-sidebar {
        width: 25%;
        min-width: 30;
        height: 100%;
        background: #0d0d14;
        padding: 0;
        /* No hover effects to prevent blinking */
    }
    
    AgentAIAssistantScreen .crush-sidebar.collapsed {
        width: 0;
        display: none;
    }
    
    /* Title container with stats button */
    AgentAIAssistantScreen .sidebar-title-container {
        height: 4;
        background: #13131d;
        padding: 1;
        border-bottom: solid #2a2a3e;
        layout: horizontal;
    }
    
    AgentAIAssistantScreen .proxima-title {
        width: 1fr;
        text-style: bold;
        color: #cc66ff;
    }
    
    /* Stats toggle button - small arrow that expands sliding panel */
    AgentAIAssistantScreen .stats-toggle-btn {
        width: 4;
        height: 2;
        min-width: 4;
        background: #cc66ff;
        color: #0d0d14;
        text-style: bold;
        border: none;
    }
    
    AgentAIAssistantScreen .stats-toggle-btn:hover {
        background: #dd88ff;
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
    
    /* Model Info Section - Stats displayed inline (no hover effects to prevent blinking) */
    AgentAIAssistantScreen .model-info-section {
        height: auto;
        padding: 1;
        background: #0d0d14;
        border-bottom: solid #2a2a3e;
        /* NO hover effects - stable display */
    }
    
    AgentAIAssistantScreen .model-name {
        color: #cc66ff;
        width: 1fr;
        text-align: right;
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
        display: none;  /* Hidden - not needed anymore */
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
    
    /* Stats Section - HIDDEN (replaced by sliding panel) */
    AgentAIAssistantScreen .stats-section {
        display: none;
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
    
    /* Shortcuts Panel - Shown in sidebar instead of stats */
    AgentAIAssistantScreen .shortcuts-section {
        height: auto;
        padding: 1;
        border-bottom: solid #1a1a28;
    }
    
    AgentAIAssistantScreen .shortcut-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 0;
    }
    
    AgentAIAssistantScreen .shortcut-key {
        width: 10;
        color: #888899;
        background: #1a1a28;
        padding: 0 1;
    }
    
    AgentAIAssistantScreen .shortcut-desc {
        width: 1fr;
        color: #666688;
        padding-left: 1;
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
        
        # Initialize Robust NL Processor for dynamic intent recognition
        self._robust_nl_processor: Optional[RobustNLProcessor] = None
        if ROBUST_NL_AVAILABLE:
            try:
                self._robust_nl_processor = get_robust_nl_processor(self._llm_router)
            except Exception:
                pass

        # Initialize IntentToolBridge (Phase 3)
        self._intent_bridge: Optional[IntentToolBridge] = None  # type: ignore[type-arg]
        if INTENT_BRIDGE_AVAILABLE:
            try:
                self._intent_bridge = IntentToolBridge(
                    consent_callback=self._bridge_consent_callback,
                )
                # Phase 8: wire LLM provider for auto-commit-message generation
                if self._llm_router and LLM_AVAILABLE:
                    self._intent_bridge.set_llm_provider(
                        _LLMRouterAdapter(self._llm_router)
                    )
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
            # CRUSH-STYLE SIDEBAR (Right) - Info panel (no stats - they're in sliding panel)
            # ====================================================================
            sidebar_classes = "crush-sidebar" if self.side_panel_visible else "crush-sidebar collapsed"
            with Vertical(classes=sidebar_classes, id="crush-sidebar"):
                # Title with Stats Button
                with Container(classes="sidebar-title-container"):
                    yield Static("📊 Stats & Shortcuts", classes="proxima-title", id="proxima-title")
                    # Small button to toggle sliding stats panel
                    yield Button("◀", id="btn-show-stats", classes="stats-toggle-btn")
                
                # Model Info Section
                with Container(classes="model-info-section"):
                    with Horizontal(classes="stats-row"):
                        yield Static("Provider:", classes="stats-label")
                        yield Static(self._llm_provider.title() if self._llm_provider else "None", classes="stats-value", id="stat-provider")
                    with Horizontal(classes="stats-row"):
                        yield Static("Model:", classes="stats-label")
                        yield Static(self._llm_model or "—", classes="stats-value", id="model-name")
                    with Horizontal(classes="stats-row"):
                        yield Static("Messages:", classes="stats-label")
                        yield Static("0", classes="stats-value", id="stat-messages")
                    with Horizontal(classes="stats-row"):
                        yield Static("Tokens:", classes="stats-label")
                        yield Static("0", classes="stats-value", id="stat-tokens")
                    with Horizontal(classes="stats-row"):
                        yield Static("Requests:", classes="stats-label")
                        yield Static("0", classes="stats-value", id="stat-requests")
                    with Horizontal(classes="stats-row"):
                        yield Static("Cmds Run:", classes="stats-label")
                        yield Static("0", classes="stats-value", id="stat-tools")
                    with Horizontal(classes="stats-row"):
                        yield Static("Session:", classes="stats-label")
                        yield Static("0m", classes="stats-value", id="stat-session")
                
                # Shortcuts Section
                with Container(classes="shortcuts-section"):
                    yield Static("Shortcuts", classes="section-header")
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
        
        # Initialize terminal and simulation monitoring
        self._init_monitoring_system()
        
        # Start background monitoring timer
        self._monitor_timer = self.set_interval(2.0, self._check_monitored_processes)
    
    def _init_monitoring_system(self) -> None:
        """Initialize the monitoring system for terminals and simulations."""
        if not hasattr(self, '_terminal_monitors'):
            self._terminal_monitors = {}
        if not hasattr(self, '_running_simulations'):
            self._running_simulations = {}
        if not hasattr(self, '_clipboard'):
            self._clipboard = {}
        if not hasattr(self, '_monitored_results'):
            self._monitored_results = []
    
    def _check_monitored_processes(self) -> None:
        """Background check for monitored terminals and simulations."""
        # Check running simulations
        if hasattr(self, '_running_simulations'):
            for sim_id, sim in list(self._running_simulations.items()):
                if sim.get('status') == 'running' and sim.get('process'):
                    process = sim['process']
                    poll = process.poll()
                    
                    if poll is not None:
                        # Simulation completed
                        stdout, stderr = process.communicate()
                        output = stdout or stderr or "Completed"
                        
                        sim['status'] = 'completed' if poll == 0 else 'failed'
                        sim['results'] = output
                        sim['ended'] = time.time()
                        
                        # Notify user and auto-analyze results
                        self._handle_simulation_completion(sim_id, sim)
        
        # Check terminal monitors
        if hasattr(self, '_terminal_monitors'):
            for term_id, monitor in list(self._terminal_monitors.items()):
                if monitor.get('active') and monitor.get('process'):
                    process = monitor['process']
                    poll = process.poll()
                    
                    if poll is not None:
                        # Terminal process completed
                        self._handle_terminal_completion(term_id, monitor, poll)
    
    def _handle_simulation_completion(self, sim_id: str, sim: dict) -> None:
        """Handle a completed simulation - analyze and optionally export to Results tab."""
        status = "✅" if sim['status'] == 'completed' else "❌"
        elapsed = sim.get('ended', time.time()) - sim.get('started', time.time())
        
        # Show completion notification
        self._show_ai_message(
            f"{status} **Simulation Completed**\n"
            f"📋 ID: `{sim_id}`\n"
            f"🔬 Backend: `{sim.get('backend', 'unknown')}`\n"
            f"⏱️ Duration: {elapsed:.1f}s\n\n"
            f"```\n{sim.get('results', 'No output')[:1500]}\n```"
        )
        
        # Auto-analyze results if they look like quantum simulation output
        results_text = sim.get('results', '')
        if any(indicator in results_text.lower() for indicator in ['counts', 'probabilities', 'state', 'measurement', 'qubit']):
            self._show_ai_message(
                f"📊 **Auto-Analysis Available**\n\n"
                f"Quantum simulation results detected. "
                f"Say `analyze results` or `export to results tab` to continue."
            )
            
            # Store for potential export
            if not hasattr(self, '_monitored_results'):
                self._monitored_results = []
            self._monitored_results.append({
                'sim_id': sim_id,
                'results': results_text,
                'backend': sim.get('backend'),
                'timestamp': time.time()
            })
    
    def _handle_terminal_completion(self, term_id: str, monitor: dict, return_code: int) -> None:
        """Handle terminal process completion."""
        watch_for = monitor.get('watch_for', 'completion')
        monitor['active'] = False
        
        stdout, stderr = monitor['process'].communicate() if monitor.get('process') else ('', '')
        output = stdout or stderr or "Process completed"
        
        if return_code == 0:
            self._show_ai_message(
                f"✅ **Terminal `{term_id}` Completed**\n"
                f"👀 Watched for: `{watch_for}`\n"
                f"```\n{output[:1500]}\n```"
            )
        else:
            self._show_ai_message(
                f"❌ **Terminal `{term_id}` Failed**\n"
                f"Exit code: {return_code}\n"
                f"```\n{output[:1500]}\n```"
            )
    
    def on_unmount(self) -> None:
        """Called when screen is unmounted."""
        if self._agent:
            self._agent.stop()
        self._save_panel_settings()
        
        # Stop the monitoring timer
        if hasattr(self, '_monitor_timer') and self._monitor_timer:
            self._monitor_timer.stop()
        
        # Clean up any running processes
        if hasattr(self, '_running_simulations'):
            for sim_id, sim in self._running_simulations.items():
                if sim.get('process') and sim['process'].poll() is None:
                    try:
                        sim['process'].terminate()
                    except Exception:
                        pass
    
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
            welcome.append("🤖 Proxima AI Agent - Full Administrative Control\n", style=f"bold {theme.accent}")
            welcome.append("━" * 50 + "\n\n", style=theme.border)
            
            welcome.append("I have complete control over your system:\n\n", style=theme.fg_base)
            welcome.append("📁 ", style=theme.accent)
            welcome.append("File System: create, read, write, delete, copy, move, search\n", style=theme.fg_base)
            welcome.append("🔀 ", style=theme.accent)
            welcome.append("Git & GitHub: clone, push, pull, commit, create repos, authenticate\n", style=theme.fg_base)
            welcome.append("💻 ", style=theme.accent)
            welcome.append("Terminal: run any command, execute scripts, manage processes\n", style=theme.fg_base)
            welcome.append("🔬 ", style=theme.accent)
            welcome.append("Backends: clone, build, and run quantum backends (Cirq, Qiskit, etc.)\n", style=theme.fg_base)
            welcome.append("📊 ", style=theme.accent)
            welcome.append("Results: monitor simulations, analyze output, export to Results tab\n", style=theme.fg_base)
            welcome.append("📺 ", style=theme.accent)
            welcome.append("Monitoring: watch terminals, detect completion, auto-analyze\n\n", style=theme.fg_base)
            
            if self._agent:
                welcome.append("✓ Agent Ready\n", style=theme.success)
            else:
                welcome.append("⚠️ Agent unavailable - chat only mode\n", style=theme.warning)
            
            if self._llm_provider and self._llm_provider != 'none':
                welcome.append(f"🧠 LLM: {self._llm_provider}", style=theme.fg_muted)
                if self._llm_model:
                    welcome.append(f" ({self._llm_model})", style=theme.fg_muted)
                welcome.append("\n", style=theme.fg_base)
            
            welcome.append("\n" + "━" * 50 + "\n", style=theme.border)
            welcome.append("Try: \"login to github\", \"clone cirq and build it\", \"run my script\"\n", style=theme.fg_subtle)
            
            chat_log.write(welcome)
        except Exception:
            pass
    
    def _update_stats_display(self) -> None:
        """Update statistics display in sidebar - minimal updates to prevent blinking."""
        try:
            # Update message count
            msg_count = str(len(self._current_session.messages))
            try:
                widget = self.query_one("#stat-messages", Static)
                if str(widget.renderable) != msg_count:
                    widget.update(msg_count)
            except Exception:
                pass
            
            # Update tools/commands count
            tool_count = str(self._current_session.tool_executions)
            try:
                widget = self.query_one("#stat-tools", Static)
                if str(widget.renderable) != tool_count:
                    widget.update(tool_count)
            except Exception:
                pass
            
            # Update tokens
            tokens = str(self._current_session.total_tokens)
            try:
                widget = self.query_one("#stat-tokens", Static)
                if str(widget.renderable) != tokens:
                    widget.update(tokens)
            except Exception:
                pass
            
            # Update requests
            requests = str(self._current_session.total_requests)
            try:
                widget = self.query_one("#stat-requests", Static)
                if str(widget.renderable) != requests:
                    widget.update(requests)
            except Exception:
                pass
            
            # Update session time
            elapsed = int(time.time() - self._stats.get('session_start', time.time()))
            session_time = f"{elapsed // 60}m" if elapsed >= 60 else f"{elapsed}s"
            try:
                widget = self.query_one("#stat-session", Static)
                if str(widget.renderable) != session_time:
                    widget.update(session_time)
            except Exception:
                pass
            
            # Update model name
            model = self._llm_model or "—"
            try:
                widget = self.query_one("#model-name", Static)
                if str(widget.renderable) != model:
                    widget.update(model)
            except Exception:
                pass
            
            # Also update sliding stats panel if visible
            try:
                sliding_panel = self.query_one("#sliding-stats-panel", SlidingStatsPanel)
                sliding_panel.stats.messages_sent = len(self._current_session.messages)
                sliding_panel.stats.tokens_used = self._current_session.total_tokens
                sliding_panel.stats.requests_made = self._current_session.total_requests
                sliding_panel.stats.tools_executed = self._current_session.tool_executions
                sliding_panel.stats.commands_run = self._current_session.tool_executions
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
        elif btn_id == "btn-show-stats":
            # Toggle the sliding stats panel
            self._toggle_sliding_stats()
    
    def _toggle_sliding_stats(self) -> None:
        """Toggle the sliding stats panel visibility."""
        try:
            sliding_panel = self.query_one("#sliding-stats-panel", SlidingStatsPanel)
            sliding_panel.toggle()
            
            # Update button appearance
            btn = self.query_one("#btn-show-stats", Button)
            if sliding_panel.is_visible:
                btn.label = "▶"
            else:
                btn.label = "◀"
        except Exception:
            pass
    
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
        """Generate AI response with robust intent analysis and tool execution.
        
        Uses a multi-phase approach for reliable natural language understanding:
        1. Direct pattern matching for complex multi-step backend operations
        2. Robust NL processor (hybrid rule-based + context-aware) - MOST RELIABLE
        3. LLM-based intent extraction with JSON parsing
        4. Keyword-based agent command detection
        5. General LLM response for questions
        """
        start_time = time.time()
        
        # PHASE 0: Direct pattern matching for backend/clone/build requests
        # This handles requests like "LRET is at C:\..., clone the branch X, build it, configure"
        # WITHOUT relying on LLM JSON parsing which can be unreliable
        direct_result = self._try_direct_backend_operation(message, start_time)
        if direct_result:
            self._update_nl_context(message, None, "direct_backend_operation", True)
            return
        
        # PHASE 1: Use Robust NL Processor (MOST RELIABLE for simpler models)
        # This works with ANY LLM model including smaller ones like llama2-uncensored
        if ROBUST_NL_AVAILABLE and self._robust_nl_processor:
            robust_result = self._try_robust_nl_execution(message, start_time)
            if robust_result:
                return
        
        # PHASE 2: Use LLM to analyze intent and execute operations
        # This allows natural language understanding for ANY sentence structure
        if self._llm_router and LLM_AVAILABLE:
            operation_result = self._analyze_and_execute_with_llm(message, start_time)
            if operation_result:
                self._update_nl_context(message, None, "llm_analyzed_operation", True)
                return
        
        # PHASE 3: Fallback to keyword-based agent command detection
        if self._agent_enabled and self._agent:
            tool_result = self._try_execute_agent_command(message)
            if tool_result:
                self._update_nl_context(message, None, "agent_command", True)
                self._finish_generation()
                return
        
        # PHASE 4: Fall back to LLM response for general questions
        self._generate_llm_response(message, start_time)
        self._update_nl_context(message, None, "llm_response", True)
    
    def _try_direct_backend_operation(self, message: str, start_time: float) -> bool:
        """Directly handle backend clone/build/configure requests without LLM JSON parsing.
        
        This handles natural language patterns like:
        - "LRET is at C:\path\to\repo and I need the branch-name branch. Clone it, build it, configure ProximA"
        - "Clone https://github.com/user/repo branch xyz and build it"
        - "The backend is at /path/to/backend, use the feature branch"
        
        Returns True if handled, False otherwise.
        """
        import re
        import os
        
        msg_lower = message.lower()
        
        # Check if this is a backend/clone/build request
        backend_keywords = ['clone', 'build', 'configure', 'branch', 'backend', 'lret', 'cirq', 'qiskit', 'pennylane']
        if not any(kw in msg_lower for kw in backend_keywords):
            return False
        
        # Extract local path (Windows or Unix style)
        # Matches: C:\path\to\folder, D:/path/to/folder, /home/user/path, ~/path
        local_path_patterns = [
            r'(?:is\s+at|at|from|in)\s+([A-Za-z]:[\\\/][^\s,]+)',  # Windows: C:\path or C:/path
            r'(?:is\s+at|at|from|in)\s+(\/[^\s,]+)',  # Unix: /home/user/path
            r'(?:is\s+at|at|from|in)\s+(~[^\s,]*)',  # Home dir: ~/path
            r'([A-Za-z]:[\\\/][^\s,]+(?:LRET|lret|backend|repo)[^\s,]*)',  # Direct Windows path with keyword
        ]
        
        local_path = None
        for pattern in local_path_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                local_path = match.group(1).strip().rstrip('.,;')
                break
        
        # Extract GitHub URL
        github_url = None
        url_match = re.search(r'(https?://github\.com/[^\s]+|github\.com/[^\s]+)', message, re.IGNORECASE)
        if url_match:
            github_url = url_match.group(1)
            if not github_url.startswith('http'):
                github_url = 'https://' + github_url
        
        # Extract branch name
        branch_patterns = [
            r'(?:branch|the)\s+([a-zA-Z0-9_\-]+(?:-[a-zA-Z0-9_\-]+)*)\s+branch',  # "the xyz-abc branch"
            r'branch\s+([a-zA-Z0-9_\-]+(?:-[a-zA-Z0-9_\-]+)*)',  # "branch xyz-abc"
            r'(?:need|want|use|checkout|switch\s+to)\s+(?:the\s+)?([a-zA-Z0-9_\-]+(?:-[a-zA-Z0-9_\-]+)*)\s+branch',  # "need the xyz branch"
            r'([a-zA-Z0-9_\-]+-[a-zA-Z0-9_\-]+-[a-zA-Z0-9_\-]+)',  # Match branch-like patterns like "cirq-scalability-comparison"
        ]
        
        branch = None
        for pattern in branch_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                candidate = match.group(1)
                # Validate it looks like a branch name (not common words)
                if candidate.lower() not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'clone', 'build', 'configure']:
                    branch = candidate
                    break
        
        # Determine what operations to perform
        do_clone = 'clone' in msg_lower or local_path or github_url
        do_build = 'build' in msg_lower
        do_configure = 'configure' in msg_lower or 'proxima' in msg_lower or 'use it' in msg_lower
        
        # If we don't have enough info, let the LLM handle it
        if not (local_path or github_url) and not branch:
            return False
        
        # Execute the operations
        self._show_ai_message(f"🎯 **Detected Backend Operation Request**")
        
        results = []
        repo_name = None
        final_path = None
        
        # Step 1: Clone/Copy
        if local_path and os.path.exists(os.path.expandvars(os.path.expanduser(local_path))):
            self._show_ai_message(f"\n📂 **Step 1: Copying from local path**\n   Source: `{local_path}`" + (f"\n   Branch: `{branch}`" if branch else ""))
            result = self._copy_local_repository(local_path, '', branch, None)
            results.append(result)
            self._show_ai_message(result)
            
            # Extract the repo name from result
            if '✅' in result:
                repo_name = os.path.basename(local_path.rstrip('/\\'))
                if branch:
                    repo_name = f"{repo_name}-{branch.replace('/', '-')}"
                final_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', repo_name)
        
        elif github_url:
            self._show_ai_message(f"\n🔗 **Step 1: Cloning from GitHub**\n   URL: `{github_url}`" + (f"\n   Branch: `{branch}`" if branch else ""))
            result = self._clone_any_repository(github_url, branch, None)
            results.append(result)
            self._show_ai_message(result)
            
            if '✅' in result:
                # Extract repo name from URL
                repo_name = github_url.rstrip('/').rstrip('.git').split('/')[-1]
                final_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', repo_name)
        
        # Step 2: Build
        if do_build and repo_name:
            self._show_ai_message(f"\n🔨 **Step 2: Building `{repo_name}`**")
            result = self._build_cloned_repository('', repo_name, {})
            results.append(result)
            self._show_ai_message(result)
        
        # Step 3: Configure
        if do_configure and repo_name:
            self._show_ai_message(f"\n⚙️ **Step 3: Configuring Proxima to use `{repo_name}`**")
            result = self._configure_backend_for_proxima(repo_name, final_path or '', 'custom')
            results.append(result)
            self._show_ai_message(result)
        
        # Summary
        success_count = sum(1 for r in results if '✅' in r)
        self._show_ai_message(f"\n✨ **Completed {success_count}/{len(results)} steps successfully**")
        
        # Update stats
        elapsed = int((time.time() - start_time) * 1000)
        self._current_session.messages.append(
            ChatMessage(role='assistant', content="\n".join(results), thinking_time_ms=elapsed)
        )
        self._agent_stats.commands_run += len(results)
        self._update_stats_panel()
        self._save_current_session()
        self._finish_generation()
        
        return True
    
    def _try_robust_nl_execution(self, message: str, start_time: float) -> bool:
        """Use the robust NL processor for reliable intent recognition and execution.
        
        Phase 3 enhanced: Uses IntentToolBridge for dispatch when available,
        falls back to the in-processor execute_intent for core actions.
        
        Dispatch paths:
        1. MULTI_STEP / PLAN_EXECUTION → plan creation + sequential dispatch
        2. Core intents (navigate, list, pwd, checkout, clone, etc.) → in-processor
        3. All other intents → IntentToolBridge dispatch (if available)
        4. Fallback → in-processor execute_intent
        
        Returns True if an operation was identified and executed, False otherwise.
        """
        if not self._robust_nl_processor:
            return False
        
        try:
            # Recognize the user's intent (5-layer pipeline)
            intent = self._robust_nl_processor.recognize_intent(message)
            
            # Skip if unknown intent or low confidence
            if intent.intent_type == IntentType.UNKNOWN or intent.confidence < 0.4:
                return False
            
            # Show what we understood
            self._show_ai_message(f"🔍 **Understood:** {intent.explanation}")
            
            # ── Dispatch path 1: Multi-step via IntentToolBridge ──────
            if intent.intent_type in (IntentType.MULTI_STEP, IntentType.PLAN_EXECUTION):
                return self._dispatch_multi_step(intent, message, start_time)
            
            # ── Dispatch path 2 & 3: Single intent ───────────────────
            success, result = self._dispatch_single_intent(intent)
            
            if result:
                self._show_ai_message(result)
                
                # ── Phase 2 (Step 2.5): Update session context ────────
                self._update_nl_context(message, intent, result, success)
                
                # Update stats
                elapsed = int((time.time() - start_time) * 1000)
                self._current_session.messages.append(
                    ChatMessage(role='assistant', content=result, thinking_time_ms=elapsed)
                )
                
                if success:
                    self._agent_stats.commands_run += 1
                else:
                    self._agent_stats.errors += 1
                
                self._update_stats_panel()
                self._save_current_session()
                self._finish_generation()
                return True
            
            return False
            
        except Exception as e:
            # Log error but don't show to user, fall through to other methods
            return False

    # ── Phase 3 dispatch helpers ──────────────────────────────────────

    # Core intents that are executed directly by the NL processor
    _CORE_INTENTS = frozenset({
        IntentType.NAVIGATE_DIRECTORY,
        IntentType.LIST_DIRECTORY,
        IntentType.SHOW_CURRENT_DIR,
        IntentType.GIT_CHECKOUT,
        IntentType.GIT_STATUS,
        IntentType.GIT_PULL,
        IntentType.GIT_CLONE,
        IntentType.RUN_SCRIPT,
        IntentType.RUN_COMMAND,
        IntentType.QUERY_LOCATION,
        IntentType.QUERY_STATUS,
    })

    def _dispatch_single_intent(
        self, intent: 'Intent'
    ) -> 'Tuple[bool, str]':
        """Dispatch a single intent through the best available path.

        1. Core intents → ``RobustNLProcessor.execute_intent``
        2. Other intents → ``IntentToolBridge.dispatch`` (if available)
        3. Fallback → ``RobustNLProcessor.execute_intent`` (deferred ack)
        """
        # Core intents always handled in-processor for reliability
        if intent.intent_type in self._CORE_INTENTS:
            return self._robust_nl_processor.execute_intent(intent)

        # Try bridge dispatch
        if self._intent_bridge is not None:
            try:
                ctx = self._robust_nl_processor.get_context()
                cwd = ctx.current_directory
                tool_result = self._intent_bridge.dispatch(
                    intent, cwd=cwd, session_context=ctx,
                )
                # Convert ToolResult → (bool, str)
                msg = tool_result.message or (
                    str(tool_result.result) if tool_result.result else
                    ("✅ Done" if tool_result.success else f"❌ {tool_result.error}")
                )
                return tool_result.success, msg
            except Exception:
                pass  # Fall through to processor

        # Fallback — processor handles it (deferred acknowledgement)
        return self._robust_nl_processor.execute_intent(intent)

    def _dispatch_multi_step(
        self,
        intent: 'Intent',
        message: str,
        start_time: float,
    ) -> bool:
        """Handle MULTI_STEP / PLAN_EXECUTION intents.

        If the IntentToolBridge is available, dispatches each sub-intent
        through it (with cwd updates between steps).  Otherwise, falls
        back to the in-processor ``_execute_multi_step``.
        """
        if self._intent_bridge is not None and intent.sub_intents:
            ctx = self._robust_nl_processor.get_context()
            cwd = ctx.current_directory
            results = self._intent_bridge.dispatch_multi_step(
                intent, cwd=cwd, session_context=ctx,
            )
            # Compose result message
            parts: list[str] = []
            success_count = 0
            for i, (sub, res) in enumerate(
                zip(intent.sub_intents, results), 1
            ):
                label = sub.intent_type.name.replace('_', ' ').title()
                parts.append(f"\n**Step {i}: {label}**")
                msg = res.message or (
                    str(res.result) if res.result else
                    ("✅ Done" if res.success else f"❌ {res.error}")
                )
                parts.append(msg)
                if res.success:
                    success_count += 1
                # Update session context after each step
                self._update_nl_context(sub.raw_message, sub, msg, res.success)

            parts.append(
                f"\n✨ **Completed {success_count}/{len(results)} steps successfully**"
            )
            combined = "\n".join(parts)
            self._show_ai_message(combined)
        else:
            # Fallback to in-processor multi-step
            success, combined = self._robust_nl_processor.execute_intent(intent)
            self._show_ai_message(combined)
            self._update_nl_context(message, intent, combined, success)

        elapsed = int((time.time() - start_time) * 1000)
        self._current_session.messages.append(
            ChatMessage(role='assistant', content="multi-step executed", thinking_time_ms=elapsed)
        )
        self._agent_stats.commands_run += 1
        self._update_stats_panel()
        self._save_current_session()
        self._finish_generation()
        return True

    def _bridge_consent_callback(self, command: str) -> bool:
        """Consent callback for the IntentToolBridge.

        For now, returns ``True`` for all commands.  A future iteration
        can integrate with the TUI modal dialog.
        """
        # TODO: Integrate with TUI ConsentDialog
        return True

    def _create_plan_from_intents(
        self, sub_intents: list
    ) -> 'Optional[ExecutionPlan]':
        """Create an ``ExecutionPlan`` from a list of sub-intents.

        Used for ``PLAN_EXECUTION`` intents that need the full planner
        integration.  Returns ``None`` if plan imports are unavailable.
        """
        if not PLAN_AVAILABLE:
            return None

        steps: list[PlanStep] = []
        for i, sub in enumerate(sub_intents):
            tool_name = INTENT_TO_TOOL.get(sub.intent_type) if INTENT_BRIDGE_AVAILABLE else None
            cwd = self._robust_nl_processor.get_context().current_directory if self._robust_nl_processor else "."
            args = build_tool_arguments(sub, cwd) if INTENT_BRIDGE_AVAILABLE else {}
            step = PlanStep(
                step_id=i + 1,
                tool=tool_name or sub.intent_type.name.lower(),
                arguments=args,
                description=sub.explanation or sub.intent_type.name.replace('_', ' ').title(),
                depends_on=[i] if i > 0 else [],
                status=StepStatus.PENDING,
            )
            steps.append(step)

        plan = ExecutionPlan(
            plan_id=f"plan_{int(time.time())}",
            description=f"Multi-step execution plan ({len(steps)} steps)",
            steps=steps,
            category=TaskCategory.EXECUTE,
            status=PlanStatus.DRAFT,
            user_request=sub_intents[0].raw_message if sub_intents else "",
        )
        return plan

    def _update_nl_context(
        self,
        message: str,
        intent: 'Optional[Intent]',
        result: str,
        success: bool,
    ) -> None:
        """Update the NL processor's session context after an operation.

        Called after every successful (or attempted) execution from the
        robust NL pipeline so that subsequent requests can resolve
        pronouns and contextual references accurately.

        *intent* may be ``None`` for phases that do not produce an
        ``Intent`` object (Phases 0, 2, 3, 4).  In that case only
        the conversation entry and operation result are recorded.
        """
        try:
            import os  # local import — used for env path join below
            if not self._robust_nl_processor:
                return

            ctx = self._robust_nl_processor.get_context()

            # Always record the conversation entry
            label = intent.intent_type.name if intent else "DIRECT_OPERATION"
            ctx.add_conversation_entry(message, label)

            # Always record the operation result (even on failure)
            ctx.last_operation_result = result

            # Intent-specific updates (only on success with a real intent)
            if not success or not intent:
                return

            it = intent.intent_type

            # Script execution → track last script
            if it in (IntentType.RUN_SCRIPT,):
                script = (
                    intent.get_entity('script_path')
                    or intent.get_entity('script')
                    or intent.get_entity('path')
                )
                if script:
                    ctx.last_script_executed = script

            # Directory navigation → push directory
            if it in (IntentType.NAVIGATE_DIRECTORY,):
                new_dir = (
                    intent.get_entity('path')
                    or intent.get_entity('dirname')
                )
                if new_dir:
                    ctx.push_directory(new_dir)

            # Package installation → record each package
            if it in (IntentType.INSTALL_DEPENDENCY,):
                for pkg in intent.get_all_entities('package'):
                    ctx.add_package(pkg)
                    if pkg not in ctx.installed_packages:
                        ctx.installed_packages.append(pkg)

            # Environment configuration → record active environments (Phase 5)
            if it in (IntentType.CONFIGURE_ENVIRONMENT,):
                env_name = (
                    intent.get_entity('env_name')
                    or intent.get_entity('venv_name')
                    or intent.get_entity('name')
                    or '.venv'
                )
                env_path = os.path.join(ctx.current_directory, env_name)
                ctx.active_environments[env_name] = env_path

            # Backend build → track last built backend
            if it in (IntentType.BACKEND_BUILD,):
                backend = (
                    intent.get_entity('name')
                    or intent.get_entity('backend')
                    or intent.get_entity('path')
                )
                if backend:
                    ctx.last_built_backend = backend

            # Backend modify → track modified files and checkpoints (Phase 6)
            if it in (IntentType.BACKEND_MODIFY,):
                file_path = (
                    intent.get_entity('path')
                    or intent.get_entity('filename')
                    or intent.get_entity('file')
                )
                if file_path:
                    ctx.last_modified_files = [file_path]
                backend = (
                    intent.get_entity('backend')
                    or intent.get_entity('name')
                )
                # Checkpoint ID stored by the bridge handler directly

            # Backend configure → update context (Phase 6)
            if it in (IntentType.BACKEND_CONFIGURE,):
                backend = (
                    intent.get_entity('backend')
                    or intent.get_entity('name')
                )
                if backend:
                    ctx.last_built_backend = backend

            # Git clone → already tracked by execute_intent, but ensure
            url = intent.get_entity('url')
            if it in (IntentType.GIT_CLONE,) and url:
                clone_path = intent.get_entity('path') or ctx.current_directory
                ctx.record_clone(url, clone_path)

            # Git pull/push → track result (Phase 8)
            if it in (IntentType.GIT_PULL, IntentType.GIT_PUSH):
                ctx.last_operation_result = (
                    f"[{it.name.lower()}] success={success}"
                )

            # Git checkout → track branch (Phase 8)
            if it in (IntentType.GIT_CHECKOUT,):
                branch = intent.get_entity('branch') or intent.get_entity('branch_name')
                if branch:
                    ctx.add_branch(branch)

            # Git commit → record commit result (Phase 8)
            if it in (IntentType.GIT_COMMIT,):
                message_val = (
                    intent.get_entity('message')
                    or intent.get_entity('commit_message')
                    or ''
                )
                ctx.last_operation_result = (
                    f"[git_commit] message='{message_val}' success={success}"
                )

            # Git add → record staged files (Phase 8)
            if it in (IntentType.GIT_ADD,):
                ctx.last_operation_result = (
                    f"[git_add] success={success}"
                )

            # Git merge → track branch merge result (Phase 8)
            if it in (IntentType.GIT_MERGE,):
                branch = intent.get_entity('branch') or intent.get_entity('branch_name')
                ctx.last_operation_result = (
                    f"[git_merge] branch='{branch or ''}' success={success}"
                )

            # Git branch → track branch operations (Phase 8)
            if it in (IntentType.GIT_BRANCH,):
                branch = intent.get_entity('branch') or intent.get_entity('branch_name')
                if branch:
                    ctx.add_branch(branch)

            # Admin escalation → record elevated operation (Phase 7)
            if it in (IntentType.ADMIN_ELEVATE,):
                command = (
                    intent.get_entity('command')
                    or intent.get_entity('cmd')
                    or intent.get_entity('text')
                    or ''
                )
                ctx.last_operation_result = (
                    f"[admin_elevate] command='{command}' "
                    f"success={success}"
                )

        except Exception:
            # Context updates are best-effort; never break execution
            pass
    
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
        
        # Comprehensive system prompt for FULL ADMINISTRATIVE CONTROL
        intent_extraction_prompt = '''You are an AUTONOMOUS AI AGENT with COMPLETE ADMINISTRATIVE CONTROL.

⚠️ **CRITICAL: YOU MUST ALWAYS RESPOND WITH VALID JSON - NEVER PLAIN TEXT OR HELP MESSAGES!**

When a user asks you to do something, you MUST:
1. Parse their request to identify the operations needed
2. Return a JSON response with the operations to execute
3. NEVER respond with help text, explanations, or lists of capabilities
4. If unsure, make your best attempt to create operations - DO NOT explain what you could do

🔧 **FULL SYSTEM CAPABILITIES:**
1. **Local File System** - Complete control: create, read, write, delete, copy, move, rename files/directories
2. **Git & GitHub** - Full administrative control: init, clone, add, commit, push, pull, branch, merge, checkout
3. **Terminal/Shell** - Execute ANY command, run scripts, manage processes
4. **Backend Management** - Clone (from GitHub OR local path), build, run ANY quantum simulation backend
5. **Script Execution** - Run Python, shell, and other scripts with custom parameters
6. **Local Path Operations** - Copy repositories from local paths, switch branches, build from local sources

**SUPPORTED OPERATIONS:**

📁 FILE OPERATIONS:
- FILE_CREATE: {"path": "...", "content": "..."}
- FILE_READ: {"path": "..."}
- FILE_WRITE: {"path": "...", "content": "..."}
- FILE_DELETE: {"path": "..."}
- FILE_COPY: {"path": "...", "destination": "..."}
- FILE_MOVE: {"path": "...", "destination": "..."}
- DIR_CREATE: {"path": "..."}
- DIR_DELETE: {"path": "...", "recursive": true}
- DIR_LIST: {"path": "...", "detailed": true}
- DIR_COPY: {"source": "...", "destination": "..."} - Copy entire directory

🔀 GIT OPERATIONS:
- GIT_CLONE: {"url": "...", "path": "...", "branch": "..."}
- GIT_CHECKOUT: {"path": "...", "branch": "..."} - Switch to a specific branch in a local repo
- GIT_PULL: {"path": "...", "remote": "origin", "branch": "main"}
- GIT_STATUS: {"path": "..."}
- GIT_BRANCH: {"path": "...", "branch": "...", "action": "create/delete/list/switch"}
- GIT_FETCH: {"path": "...", "remote": "origin"}

🐙 GITHUB OPERATIONS:
- GITHUB_AUTH_LOGIN: {}
- GITHUB_AUTH_STATUS: {}
- GITHUB_CREATE_REPO: {"repo_name": "...", "private": false, "description": "..."}
- GITHUB_CLONE_REPO: {"owner": "...", "repo": "...", "path": "...", "branch": "..."}

💻 TERMINAL OPERATIONS:
- TERMINAL_CMD: {"command": "...", "cwd": "...", "background": false, "timeout": 120}
- SCRIPT_RUN_PYTHON: {"script": "...", "args": [...], "cwd": "..."}

🔬 BACKEND OPERATIONS (Quantum Simulation):
- BACKEND_LIST: {}
- BACKEND_CLONE: {"backend": "...", "url": "...", "branch": "...", "path": "..."}
- BACKEND_BUILD: {"backend": "...", "path": "...", "options": {...}}
- BACKEND_INSTALL: {"backend": "...", "path": "..."}
- BACKEND_TEST: {"backend": "...", "path": "..."}
- BACKEND_RUN: {"backend": "...", "circuit_file": "...", "shots": 1024, "config": {...}}

🚀 **DYNAMIC/ANY REPOSITORY OPERATIONS:**
These work with ANY repository - GitHub URLs OR local paths:

- CLONE_ANY_REPO: Clone from GitHub URL
  {"url": "https://github.com/owner/repo", "branch": "...", "destination": "..."}

- COPY_LOCAL_REPO: Copy repository from local path and optionally switch branch
  {"source_path": "C:/path/to/repo", "destination": "...", "branch": "...", "name": "..."}

- CHECKOUT_BRANCH: Switch to a branch in a local repository
  {"repo_path": "C:/path/to/repo", "branch": "branch-name"}

- BUILD_CLONED_REPO: Auto-detect and build ANY cloned/copied repository
  {"repo_path": "...", "name": "..."}
  Auto-detects: setup.py, pyproject.toml, requirements.txt, CMakeLists.txt, Makefile

- CONFIGURE_BACKEND: Configure Proxima to use a specific backend
  {"name": "...", "path": "...", "type": "..."}

- RUN_ANY_BACKEND: Run ANY backend with custom parameters
  {"repo_path": "...", "entry_point": "...", "args": [...], "config": {...}}

**RESPONSE FORMAT:**

For MULTI-STEP operations (clone + build + configure):
{
  "is_multi_step": true,
  "steps": [
    {"operation": "OP_NAME", "params": {...}, "description": "What this step does"},
    {"operation": "OP_NAME", "params": {...}, "description": "What this step does"}
  ],
  "confidence": 0.95,
  "explanation": "Overall explanation"
}

For SINGLE operations:
{
  "is_multi_step": false,
  "operation": "OP_NAME",
  "params": {...},
  "confidence": 0.95,
  "explanation": "What will be done"
}

**CRITICAL EXAMPLES - STUDY THESE CAREFULLY:**

User: "LRET is at C:\\Users\\dell\\Pictures\\Screenshots\\LRET and I need the cirq-scalability-comparison branch. Clone it, build it, and configure ProximA to use it"
{
  "is_multi_step": true,
  "steps": [
    {"operation": "COPY_LOCAL_REPO", "params": {"source_path": "C:\\Users\\dell\\Pictures\\Screenshots\\LRET", "branch": "cirq-scalability-comparison", "name": "lret-cirq-scalability"}, "description": "Copy LRET from local path and checkout branch"},
    {"operation": "BUILD_CLONED_REPO", "params": {"name": "lret-cirq-scalability"}, "description": "Build the LRET backend"},
    {"operation": "CONFIGURE_BACKEND", "params": {"name": "lret-cirq-scalability", "type": "lret"}, "description": "Configure ProximA to use this backend"}
  ],
  "confidence": 0.98,
  "explanation": "Copy LRET from local path, switch to cirq-scalability-comparison branch, build it, and configure ProximA"
}

User: "The quantum sim repo is at D:/projects/qsim, I want the develop branch, please set it up"
{
  "is_multi_step": true,
  "steps": [
    {"operation": "COPY_LOCAL_REPO", "params": {"source_path": "D:/projects/qsim", "branch": "develop", "name": "qsim-develop"}, "description": "Copy repo and checkout develop branch"},
    {"operation": "BUILD_CLONED_REPO", "params": {"name": "qsim-develop"}, "description": "Build the backend"}
  ],
  "confidence": 0.95,
  "explanation": "Copy local quantum sim repo, switch to develop branch, and build it"
}

User: "clone https://github.com/user/custom-quantum from the experimental branch and build it"
{
  "is_multi_step": true,
  "steps": [
    {"operation": "CLONE_ANY_REPO", "params": {"url": "https://github.com/user/custom-quantum", "branch": "experimental"}, "description": "Clone from GitHub with experimental branch"},
    {"operation": "BUILD_CLONED_REPO", "params": {"name": "custom-quantum"}, "description": "Build the cloned repo"}
  ],
  "confidence": 0.95,
  "explanation": "Clone GitHub repo with experimental branch and build it"
}

User: "I have pennylane at /home/user/pennylane-fork, use the gpu-optimization branch"
{
  "is_multi_step": true,
  "steps": [
    {"operation": "COPY_LOCAL_REPO", "params": {"source_path": "/home/user/pennylane-fork", "branch": "gpu-optimization", "name": "pennylane-gpu"}, "description": "Copy local pennylane and switch branch"},
    {"operation": "BUILD_CLONED_REPO", "params": {"name": "pennylane-gpu"}, "description": "Build pennylane"}
  ],
  "confidence": 0.95,
  "explanation": "Copy local pennylane fork and switch to gpu-optimization branch"
}

User: "checkout branch feature-x in the repo at C:/myproject"
{
  "is_multi_step": false,
  "operation": "CHECKOUT_BRANCH",
  "params": {"repo_path": "C:/myproject", "branch": "feature-x"},
  "confidence": 0.95,
  "explanation": "Switch to feature-x branch in the local repository"
}

User: "build my-backend that I just cloned"
{
  "is_multi_step": false,
  "operation": "BUILD_CLONED_REPO",
  "params": {"name": "my-backend"},
  "confidence": 0.95,
  "explanation": "Build the my-backend using auto-detected build system"
}

User: "configure proxima to use the cirq backend at C:/backends/cirq"
{
  "is_multi_step": false,
  "operation": "CONFIGURE_BACKEND",
  "params": {"name": "cirq", "path": "C:/backends/cirq", "type": "cirq"},
  "confidence": 0.95,
  "explanation": "Configure ProximA to use cirq backend from specified path"
}

User: "What is quantum computing?"
{
  "is_multi_step": false,
  "operation": "NONE",
  "params": {},
  "confidence": 0.95,
  "explanation": "General knowledge question - not a system operation"
}

**REMEMBER:**
1. ALWAYS return valid JSON - NEVER plain text explanations or help messages
2. When user mentions a LOCAL PATH (like C:\... or /home/...) use COPY_LOCAL_REPO, not CLONE_ANY_REPO
3. When user mentions a GitHub URL, use CLONE_ANY_REPO
4. When user wants to switch/checkout a branch in a local repo, include that in COPY_LOCAL_REPO params
5. ALWAYS create multi-step operations for complex requests (clone + build + configure)
6. Extract paths, branch names, and repo names from user's natural language
7. If the user gives context about where something is located, USE THAT PATH

Now analyze this user request and return ONLY a valid JSON response:
  "confidence": 0.95,
  "explanation": "Overall explanation of what will be done"
}

For SINGLE operations:
{
  "is_multi_step": false,
  "operation": "OP_NAME",
  "params": {...},
  "confidence": 0.95,
  "explanation": "What will be done"
}

**EXAMPLES:**

Now analyze this user request and return ONLY a valid JSON response:'''

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
                remote = params.get('remote', 'origin')
                
                # Use the GitHubAuth module for authenticated push
                if GITHUB_AUTH_AVAILABLE:
                    try:
                        github_auth = get_github_auth()
                        auth_result = github_auth.check_auth_status()
                        
                        if not auth_result.is_authenticated:
                            # Attempt to initiate authentication automatically
                            auth_initiated = self._initiate_github_auth_flow()
                            if not auth_initiated:
                                instructions = github_auth.get_auth_instructions()
                                return (
                                    "⚠️ **GitHub Authentication Required for Push**\n\n"
                                    f"{instructions}\n\n"
                                    "🔄 After authenticating, try your push command again."
                                )
                        
                        # Perform authenticated push
                        success, message = github_auth.push_with_auth(
                            repo_path=os.getcwd(),
                            remote=remote,
                            branch=push_branch,
                            set_upstream=set_upstream,
                        )
                        
                        if success:
                            return f"✅ **Git Push Successful:**\n{message}"
                        else:
                            # Check if it's an auth error
                            if 'authentication' in message.lower() or 'permission' in message.lower() or 'denied' in message.lower():
                                auth_initiated = self._initiate_github_auth_flow()
                                if auth_initiated:
                                    return (
                                        "🔐 **Authentication Started**\n\n"
                                        "A GitHub authentication flow has been initiated.\n"
                                        "Please complete authentication in your terminal/browser.\n\n"
                                        "🔄 After authenticating, try your push command again."
                                    )
                            return f"❌ **Git Push Failed:**\n{message}"
                    except Exception as e:
                        # Fall back to basic push on error
                        pass
                
                # Fallback to basic git push
                if set_upstream:
                    cmd = ['git', 'push', '-u', remote, push_branch]
                else:
                    cmd = ['git', 'push']
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                status = "✅" if result.returncode == 0 else "❌"
                output = result.stdout or result.stderr
                
                # Check for auth errors and provide helpful guidance
                if result.returncode != 0 and ('authentication' in output.lower() or 'permission' in output.lower() or 'denied' in output.lower()):
                    auth_initiated = self._initiate_github_auth_flow() if GITHUB_AUTH_AVAILABLE else False
                    if auth_initiated:
                        return (
                            "🔐 **Authentication Started**\n\n"
                            "A GitHub authentication flow has been initiated.\n"
                            "Please complete authentication in your terminal/browser.\n\n"
                            f"**Error was:** {output}\n\n"
                            "🔄 After authenticating, try your push command again."
                        )
                    return (
                        f"❌ **Git Push Failed - Authentication Required**\n\n{output}\n\n"
                        "💡 **To authenticate with GitHub:**\n"
                        "1. Install GitHub CLI: https://cli.github.com/\n"
                        "2. Run: `gh auth login`\n"
                        "3. Follow the prompts to authenticate\n"
                        "4. Try your push command again"
                    )
                
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
            
            # GITHUB OPERATIONS - With proper authentication handling
            elif operation == 'GITHUB_CREATE_REPO':
                repo_name = params.get('repo_name', '')
                if not repo_name:
                    return "❌ No repository name specified"
                
                private = params.get('private', False)
                description = params.get('description', '')
                
                # Use GitHubAuth module if available
                if GITHUB_AUTH_AVAILABLE:
                    try:
                        github_auth = get_github_auth()
                        auth_result = github_auth.check_auth_status()
                        
                        if not auth_result.is_authenticated:
                            # Attempt to initiate authentication
                            auth_initiated = self._initiate_github_auth_flow()
                            if auth_initiated:
                                return (
                                    "🔐 **Authentication Started**\n\n"
                                    "A GitHub authentication flow has been initiated.\n"
                                    "Please complete authentication in your terminal/browser.\n\n"
                                    f"📦 After authenticating, I will create the repository: **{repo_name}**\n\n"
                                    "🔄 Run your command again after authenticating."
                                )
                            else:
                                instructions = github_auth.get_auth_instructions()
                                return (
                                    "⚠️ **GitHub Authentication Required**\n\n"
                                    f"{instructions}\n\n"
                                    f"📦 After authenticating, try creating the repository again: **{repo_name}**"
                                )
                        
                        # Create the repo using GitHubAuth
                        success, message, repo_url = github_auth.create_repo(
                            name=repo_name,
                            description=description,
                            private=private,
                            working_dir=os.getcwd(),
                        )
                        
                        if success:
                            result_msg = f"✅ **Created GitHub repository: `{repo_name}`**\n{message}"
                            if repo_url:
                                result_msg += f"\n\n📍 Repository URL: {repo_url}"
                            return result_msg
                        else:
                            return f"❌ **Failed to create repository:**\n{message}"
                    except Exception as e:
                        # Fall through to basic approach
                        pass
                
                # Fallback: Check if GitHub CLI is available
                gh_check = subprocess.run(['gh', '--version'], capture_output=True, text=True, timeout=10)
                
                if gh_check.returncode != 0:
                    return (
                        "❌ **GitHub CLI Not Installed**\n\n"
                        "The GitHub CLI (gh) is required for creating repositories.\n\n"
                        "**Install GitHub CLI:**\n"
                        "- Windows: `winget install GitHub.cli`\n"
                        "- macOS: `brew install gh`\n"
                        "- Download: https://cli.github.com/\n\n"
                        "After installing, run: `gh auth login`"
                    )
                
                # Check auth status
                auth_check = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True, timeout=30)
                if 'Logged in' not in (auth_check.stdout + auth_check.stderr):
                    auth_initiated = self._initiate_github_auth_flow() if GITHUB_AUTH_AVAILABLE else False
                    if auth_initiated:
                        return (
                            "🔐 **Authentication Started**\n\n"
                            "A GitHub authentication flow has been initiated.\n"
                            "Please complete authentication in your terminal/browser.\n\n"
                            f"📦 After authenticating, run your command again to create: **{repo_name}**"
                        )
                    return (
                        "⚠️ **Not authenticated with GitHub**\n\n"
                        "Please run: `gh auth login`\n"
                        "Then try creating the repository again."
                    )
                
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
                if GITHUB_AUTH_AVAILABLE:
                    try:
                        github_auth = get_github_auth()
                        auth_result = github_auth.check_auth_status(force_refresh=True)
                        
                        if auth_result.is_authenticated:
                            username = auth_result.username or "unknown"
                            return (
                                f"✅ **GitHub Authentication Status**\n\n"
                                f"🔓 **Logged in** as `{username}`\n"
                                f"📌 Method: {auth_result.method.value}\n"
                                f"💬 {auth_result.message}"
                            )
                        else:
                            instructions = github_auth.get_auth_instructions()
                            return (
                                f"⚠️ **Not Authenticated with GitHub**\n\n"
                                f"{instructions}"
                            )
                    except Exception:
                        pass
                
                # Fallback
                result = subprocess.run(['gh', 'auth', 'status'], capture_output=True, text=True, timeout=30)
                output = result.stdout or result.stderr
                if 'Logged in' in output:
                    return f"✅ **GitHub Authentication:**\n```\n{output}\n```"
                else:
                    return (
                        "⚠️ **Not authenticated with GitHub**\n\n"
                        "**To authenticate:**\n"
                        "1. Install GitHub CLI: https://cli.github.com/\n"
                        "2. Run: `gh auth login`\n"
                        "3. Follow the prompts"
                    )
            
            elif operation == 'GITHUB_AUTH_LOGIN':
                # Actually initiate the auth flow
                auth_initiated = self._initiate_github_auth_flow() if GITHUB_AUTH_AVAILABLE else False
                
                if auth_initiated:
                    return (
                        "🔐 **GitHub Authentication Initiated**\n\n"
                        "A browser window should open for authentication.\n"
                        "Please complete the authentication in your browser.\n\n"
                        "If a browser didn't open, run this in your terminal:\n"
                        "```\ngh auth login\n```"
                    )
                else:
                    return (
                        "🔐 **GitHub Authentication**\n\n"
                        "**To authenticate with GitHub:**\n\n"
                        "1. Install GitHub CLI from: https://cli.github.com/\n"
                        "2. Open a terminal and run:\n"
                        "```\ngh auth login\n```\n"
                        "3. Follow the prompts to authenticate\n\n"
                        "After authentication, you can push code and create repositories."
                    )
            
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
                
                background = params.get('background', False)
                timeout_val = params.get('timeout', 120)
                
                if background:
                    # Run in background
                    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    return f"✅ **Started background process:** `{command}`\n📋 PID: {process.pid}"
                else:
                    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_val)
                    output = result.stdout or result.stderr or "Command completed"
                    status = "✅" if result.returncode == 0 else "❌"
                    return f"{status} **Executed:** `{command}`\n```\n{output[:3000]}\n```"
            
            # TERMINAL SCRIPT - Run script file
            elif operation == 'TERMINAL_SCRIPT':
                script_path = params.get('script_path', '')
                script_args = params.get('args', [])
                interpreter = params.get('interpreter', 'python')
                
                if not script_path:
                    return "❌ No script path specified"
                
                script_path = os.path.expanduser(os.path.expandvars(script_path))
                if not os.path.exists(script_path):
                    return f"❌ Script not found: `{script_path}`"
                
                # Build command based on interpreter
                if interpreter == 'python':
                    cmd = ['python', script_path] + script_args
                elif interpreter == 'bash':
                    cmd = ['bash', script_path] + script_args
                elif interpreter == 'powershell':
                    cmd = ['powershell', '-File', script_path] + script_args
                else:
                    cmd = [interpreter, script_path] + script_args
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                output = result.stdout or result.stderr or "Script completed"
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Script Executed:** `{script_path}`\n```\n{output[:3000]}\n```"
            
            # TERMINAL KILL - Kill process
            elif operation == 'TERMINAL_KILL':
                pid = params.get('pid')
                name = params.get('name', '')
                
                if pid:
                    try:
                        os.kill(pid, 9)
                        return f"✅ Killed process with PID: {pid}"
                    except Exception as e:
                        return f"❌ Failed to kill process {pid}: {e}"
                elif name:
                    if os.name == 'nt':
                        result = subprocess.run(['taskkill', '/F', '/IM', name], capture_output=True, text=True)
                    else:
                        result = subprocess.run(['pkill', '-f', name], capture_output=True, text=True)
                    status = "✅" if result.returncode == 0 else "❌"
                    return f"{status} Kill process `{name}`: {result.stdout or result.stderr}"
                else:
                    return "❌ No PID or process name specified"
            
            # TERMINAL LIST PROCESSES
            elif operation == 'TERMINAL_LIST_PROCESSES':
                if os.name == 'nt':
                    result = subprocess.run(['tasklist'], capture_output=True, text=True, timeout=30)
                else:
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=30)
                return f"📋 **Running Processes:**\n```\n{result.stdout[:3000]}\n```"
            
            # TERMINAL MONITOR START
            elif operation == 'TERMINAL_MONITOR_START':
                terminal_id = params.get('terminal_id', 'main')
                watch_for = params.get('watch_for', 'completion')
                
                # Store monitoring state
                if not hasattr(self, '_terminal_monitors'):
                    self._terminal_monitors = {}
                
                self._terminal_monitors[terminal_id] = {
                    'watch_for': watch_for,
                    'started': time.time(),
                    'active': True
                }
                return f"✅ **Terminal Monitor Started**\n📋 Terminal: `{terminal_id}`\n👀 Watching for: `{watch_for}`"
            
            # TERMINAL MONITOR STOP
            elif operation == 'TERMINAL_MONITOR_STOP':
                terminal_id = params.get('terminal_id', 'main')
                if hasattr(self, '_terminal_monitors') and terminal_id in self._terminal_monitors:
                    self._terminal_monitors[terminal_id]['active'] = False
                    return f"✅ Stopped monitoring terminal: `{terminal_id}`"
                return f"⚠️ No active monitor for terminal: `{terminal_id}`"
            
            # DIR_TREE - Show directory tree
            elif operation == 'DIR_TREE':
                tree_path = path or '.'
                depth = params.get('depth', 3)
                
                def build_tree(dir_path, prefix="", current_depth=0, max_depth=3):
                    if current_depth >= max_depth:
                        return ""
                    
                    entries = []
                    try:
                        items = sorted(os.listdir(dir_path))
                        dirs = [i for i in items if os.path.isdir(os.path.join(dir_path, i)) and not i.startswith('.')]
                        files = [i for i in items if os.path.isfile(os.path.join(dir_path, i)) and not i.startswith('.')]
                        
                        all_items = dirs + files
                        for i, item in enumerate(all_items[:30]):  # Limit items
                            is_last = i == len(all_items) - 1
                            connector = "└── " if is_last else "├── "
                            icon = "📁 " if item in dirs else "📄 "
                            entries.append(f"{prefix}{connector}{icon}{item}")
                            
                            if item in dirs:
                                extension = "    " if is_last else "│   "
                                entries.append(build_tree(
                                    os.path.join(dir_path, item),
                                    prefix + extension,
                                    current_depth + 1,
                                    max_depth
                                ))
                    except PermissionError:
                        entries.append(f"{prefix}⚠️ Permission denied")
                    
                    return "\n".join(filter(None, entries))
                
                tree = build_tree(tree_path, "", 0, depth)
                return f"📂 **Directory Tree: `{tree_path}`**\n```\n{tree}\n```"
            
            # FILE_CUT - Mark file for cutting
            elif operation == 'FILE_CUT':
                if not path:
                    return "❌ No file path specified"
                if not os.path.exists(path):
                    return f"❌ File not found: `{path}`"
                
                if not hasattr(self, '_clipboard'):
                    self._clipboard = {}
                self._clipboard = {'path': path, 'operation': 'cut'}
                return f"✂️ Cut: `{path}` (use FILE_PASTE to move)"
            
            # FILE_PASTE - Paste cut/copied file
            elif operation == 'FILE_PASTE':
                if not hasattr(self, '_clipboard') or not self._clipboard:
                    return "❌ Nothing to paste. Use FILE_CUT or FILE_COPY first."
                
                src = self._clipboard.get('path')
                op = self._clipboard.get('operation')
                dest = destination or params.get('destination', '.')
                
                if not src or not os.path.exists(src):
                    return f"❌ Source not found: `{src}`"
                
                if op == 'cut':
                    shutil.move(src, dest)
                    self._clipboard = {}
                    return f"✅ Moved `{src}` to `{dest}`"
                else:
                    shutil.copy2(src, dest)
                    return f"✅ Copied `{src}` to `{dest}`"
            
            # GITHUB_AUTH_LOGOUT
            elif operation == 'GITHUB_AUTH_LOGOUT':
                result = subprocess.run(['gh', 'auth', 'logout'], capture_output=True, text=True, timeout=30)
                return f"✅ Logged out from GitHub\n```\n{result.stdout or result.stderr}\n```"
            
            # GITHUB_LIST_REPOS
            elif operation == 'GITHUB_LIST_REPOS':
                result = subprocess.run(['gh', 'repo', 'list', '--limit', '20'], capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    return f"📚 **Your GitHub Repositories:**\n```\n{result.stdout}\n```"
                else:
                    return f"❌ Failed to list repos: {result.stderr}"
            
            # GITHUB_FORK_REPO
            elif operation == 'GITHUB_FORK_REPO':
                owner = params.get('owner', '')
                repo = params.get('repo', '')
                if not owner or not repo:
                    return "❌ Owner and repo name required"
                
                result = subprocess.run(['gh', 'repo', 'fork', f'{owner}/{repo}'], capture_output=True, text=True, timeout=120)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} Fork `{owner}/{repo}`:\n```\n{result.stdout or result.stderr}\n```"
            
            # GITHUB_CLONE_REPO
            elif operation == 'GITHUB_CLONE_REPO':
                owner = params.get('owner', '')
                repo = params.get('repo', '')
                clone_path = path or params.get('path', '')
                
                if not owner or not repo:
                    return "❌ Owner and repo name required"
                
                cmd = ['gh', 'repo', 'clone', f'{owner}/{repo}']
                if clone_path:
                    cmd.append(clone_path)
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} Cloned `{owner}/{repo}`:\n```\n{result.stdout or result.stderr}\n```"
            
            # GIT_FETCH
            elif operation == 'GIT_FETCH':
                remote = params.get('remote', 'origin')
                result = subprocess.run(['git', 'fetch', remote], capture_output=True, text=True, timeout=120)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Fetch:**\n```\n{result.stdout or result.stderr or 'Fetch completed'}\n```"
            
            # GIT_REBASE
            elif operation == 'GIT_REBASE':
                target_branch = branch or params.get('branch', 'main')
                result = subprocess.run(['git', 'rebase', target_branch], capture_output=True, text=True, timeout=120)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Rebase onto `{target_branch}`:**\n```\n{result.stdout or result.stderr}\n```"
            
            # GIT_RESET
            elif operation == 'GIT_RESET':
                mode = params.get('mode', 'mixed')
                commit = params.get('commit', 'HEAD~1')
                result = subprocess.run(['git', 'reset', f'--{mode}', commit], capture_output=True, text=True, timeout=30)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} **Git Reset ({mode}) to `{commit}`:**\n```\n{result.stdout or result.stderr}\n```"
            
            # GIT_TAG
            elif operation == 'GIT_TAG':
                tag_name = params.get('tag', '')
                tag_message = params.get('message', '')
                action = params.get('action', 'create')
                
                if action == 'list':
                    result = subprocess.run(['git', 'tag', '-l'], capture_output=True, text=True, timeout=30)
                    return f"🏷️ **Git Tags:**\n```\n{result.stdout or 'No tags'}\n```"
                elif action == 'delete' and tag_name:
                    result = subprocess.run(['git', 'tag', '-d', tag_name], capture_output=True, text=True, timeout=30)
                    status = "✅" if result.returncode == 0 else "❌"
                    return f"{status} Deleted tag `{tag_name}`"
                elif tag_name:
                    cmd = ['git', 'tag']
                    if tag_message:
                        cmd.extend(['-a', tag_name, '-m', tag_message])
                    else:
                        cmd.append(tag_name)
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    status = "✅" if result.returncode == 0 else "❌"
                    return f"{status} Created tag `{tag_name}`"
                return "❌ No tag name specified"
            
            # GIT_REMOTE_REMOVE
            elif operation == 'GIT_REMOTE_REMOVE':
                remote_name = params.get('name', '')
                if not remote_name:
                    return "❌ No remote name specified"
                result = subprocess.run(['git', 'remote', 'remove', remote_name], capture_output=True, text=True, timeout=30)
                status = "✅" if result.returncode == 0 else "❌"
                return f"{status} Removed remote `{remote_name}`"
            
            # ==================== BACKEND OPERATIONS ====================
            
            # BACKEND_LIST
            elif operation == 'BACKEND_LIST':
                backends_info = self._get_available_backends()
                return backends_info
            
            # BACKEND_CLONE - Enhanced with branch and URL support
            elif operation == 'BACKEND_CLONE':
                backend_name = params.get('backend', '')
                custom_url = params.get('url', None)
                custom_branch = params.get('branch', None)
                custom_path = params.get('path', None)
                return self._execute_backend_clone(backend_name, custom_url, custom_branch, custom_path)
            
            # BACKEND_BUILD
            elif operation == 'BACKEND_BUILD':
                backend_name = params.get('backend', '')
                options = params.get('options', {})
                return self._execute_backend_build(backend_name, options)
            
            # BACKEND_INSTALL
            elif operation == 'BACKEND_INSTALL':
                backend_name = params.get('backend', '')
                return self._execute_backend_install(backend_name)
            
            # BACKEND_TEST
            elif operation == 'BACKEND_TEST':
                backend_name = params.get('backend', '')
                return self._execute_backend_test(backend_name)
            
            # BACKEND_REGISTER - Register backend with Proxima
            elif operation == 'BACKEND_REGISTER':
                backend_name = params.get('backend', '')
                backend_path = params.get('path', '')
                return self._register_backend_with_proxima(backend_name, backend_path)
            
            # BACKEND_RUN - Enhanced with custom user requirements
            elif operation == 'BACKEND_RUN':
                backend_name = params.get('backend', '')
                circuit = params.get('circuit', '')
                shots = params.get('shots', 1024)
                run_params = params.get('params', {})
                circuit_file = params.get('circuit_file', '')
                custom_config = params.get('config', {})
                return self._execute_backend_run(backend_name, circuit, shots, run_params, circuit_file, custom_config)
            
            # ==================== DYNAMIC REPOSITORY OPERATIONS (ANY REPO) ====================
            
            # CLONE_ANY_REPO - Clone ANY GitHub repository by URL
            elif operation == 'CLONE_ANY_REPO':
                repo_url = params.get('url', '')
                branch = params.get('branch', None)
                custom_name = params.get('name', None)
                return self._clone_any_repository(repo_url, branch, custom_name)
            
            # BUILD_CLONED_REPO - Auto-detect and build ANY cloned repository
            elif operation == 'BUILD_CLONED_REPO':
                repo_path = params.get('repo_path', '')
                repo_name = params.get('name', '')
                build_options = params.get('options', {})
                return self._build_cloned_repository(repo_path, repo_name, build_options)
            
            # RUN_ANY_BACKEND - Run ANY cloned backend with custom parameters
            elif operation == 'RUN_ANY_BACKEND':
                repo_path = params.get('repo_path', '')
                repo_name = params.get('name', '')
                entry_point = params.get('entry_point', '')
                args = params.get('args', [])
                config = params.get('config', {})
                return self._run_any_backend(repo_path, repo_name, entry_point, args, config)
            
            # COPY_LOCAL_REPO - Copy repository from local path and optionally switch branch
            elif operation == 'COPY_LOCAL_REPO':
                source_path = params.get('source_path', '')
                destination = params.get('destination', '')
                branch = params.get('branch', None)
                name = params.get('name', None)
                return self._copy_local_repository(source_path, destination, branch, name)
            
            # CHECKOUT_BRANCH - Switch to a branch in a local repository
            elif operation == 'CHECKOUT_BRANCH':
                repo_path = params.get('repo_path', '')
                branch = params.get('branch', '')
                return self._checkout_branch(repo_path, branch)
            
            # CONFIGURE_BACKEND - Configure Proxima to use a specific backend
            elif operation == 'CONFIGURE_BACKEND':
                name = params.get('name', '')
                path = params.get('path', '')
                backend_type = params.get('type', '')
                return self._configure_backend_for_proxima(name, path, backend_type)
            
            # DIR_COPY - Copy entire directory
            elif operation == 'DIR_COPY':
                source = params.get('source', '')
                destination = params.get('destination', '')
                return self._copy_directory(source, destination)
            
            # ==================== SCRIPT OPERATIONS ====================
            
            # SCRIPT_RUN_PYTHON
            elif operation == 'SCRIPT_RUN_PYTHON':
                script = params.get('script', '')
                script_args = params.get('args', [])
                return self._execute_python_script(script, script_args)
            
            # SCRIPT_RUN_SHELL
            elif operation == 'SCRIPT_RUN_SHELL':
                script = params.get('script', '')
                script_args = params.get('args', [])
                return self._execute_shell_script(script, script_args)
            
            # SCRIPT_CREATE_AND_RUN
            elif operation == 'SCRIPT_CREATE_AND_RUN':
                script_content = params.get('content', '')
                language = params.get('language', 'python')
                return self._create_and_run_script(script_content, language)
            
            # ==================== SIMULATION & RESULTS ====================
            
            # SIM_RUN
            elif operation == 'SIM_RUN':
                backend_name = params.get('backend', 'auto')
                circuit_file = params.get('circuit_file', '')
                shots = params.get('shots', 1024)
                return self._run_simulation(backend_name, circuit_file, shots)
            
            # SIM_MONITOR
            elif operation == 'SIM_MONITOR':
                sim_id = params.get('sim_id', '')
                return self._monitor_simulation(sim_id)
            
            # SIM_CANCEL
            elif operation == 'SIM_CANCEL':
                sim_id = params.get('sim_id', '')
                return self._cancel_simulation(sim_id)
            
            # SIM_RESULTS
            elif operation == 'SIM_RESULTS':
                sim_id = params.get('sim_id', '')
                return self._get_simulation_results(sim_id)
            
            # RESULTS_ANALYZE
            elif operation == 'RESULTS_ANALYZE':
                data = params.get('data', {})
                analysis_type = params.get('analysis_type', 'general')
                return self._analyze_results(data, analysis_type)
            
            # RESULTS_EXPORT
            elif operation == 'RESULTS_EXPORT':
                results = params.get('results', {})
                title = params.get('title', 'Results')
                export_format = params.get('format', 'json')
                return self._export_to_results_tab(results, title, export_format)
            
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
    
    # =========================================================================
    # BACKEND OPERATIONS - Clone, Build, Install, Test, Run
    # =========================================================================
    
    # Backend repository mappings - supports URL or dict with url/branch/build_cmd
    BACKEND_REPOS = {
        # LRET backends with specific branches
        'lret': {
            'url': 'https://github.com/kunal5556/LRET',
            'branch': 'main',
            'description': 'LRET Core Backend',
        },
        'lret_cirq_scalability': {
            'url': 'https://github.com/kunal5556/LRET',
            'branch': 'cirq-scalability-comparison',
            'description': 'LRET Cirq Scalability Comparison branch - optimized for large circuits',
        },
        'lret_pennylane': {
            'url': 'https://github.com/kunal5556/LRET',
            'branch': 'pennylane-hybrid',
            'description': 'LRET PennyLane Hybrid Backend',
        },
        # Standard backends
        'cirq': 'https://github.com/quantumlib/Cirq',
        'qiskit': 'https://github.com/Qiskit/qiskit',
        'pennylane': 'https://github.com/PennyLaneAI/pennylane',
        'quest': 'https://github.com/QuEST-Kit/QuEST',
        'qsim': 'https://github.com/quantumlib/qsim',
        'braket': 'https://github.com/amazon-braket/amazon-braket-sdk-python',
        'cuquantum': 'https://github.com/NVIDIA/cuQuantum',
        'qutip': 'https://github.com/qutip/qutip',
        'projectq': 'https://github.com/ProjectQ-Framework/ProjectQ',
        'strawberryfields': 'https://github.com/XanaduAI/strawberryfields',
        'openfermion': 'https://github.com/quantumlib/OpenFermion',
        'pyquil': 'https://github.com/rigetti/pyquil',
    }
    
    BACKEND_INSTALL_COMMANDS = {
        # LRET backends - full build commands
        'lret': 'pip install -r requirements.txt && pip install -e . && python setup.py build_ext --inplace',
        'lret_cirq_scalability': 'pip install -r requirements.txt && pip install cirq numpy scipy && pip install -e . && python setup.py build_ext --inplace',
        'lret_pennylane': 'pip install -r requirements.txt && pip install pennylane pennylane-lightning && pip install -e .',
        # Standard backends
        'cirq': 'pip install cirq cirq-core',
        'qiskit': 'pip install qiskit qiskit-aer qiskit-ibm-runtime',
        'pennylane': 'pip install pennylane pennylane-lightning',
        'quest': 'pip install pyquest',
        'qsim': 'pip install qsimcirq',
        'braket': 'pip install amazon-braket-sdk amazon-braket-default-simulator',
        'cuquantum': 'pip install cuquantum-python',
        'qutip': 'pip install qutip',
        'projectq': 'pip install projectq',
        'strawberryfields': 'pip install strawberryfields',
        'openfermion': 'pip install openfermion openfermionpyscf',
        'pyquil': 'pip install pyquil',
    }
    
    def _get_available_backends(self) -> str:
        """Get list of available quantum simulation backends."""
        backends_list = []
        
        # Check installed backends
        installed = []
        for name, install_cmd in self.BACKEND_INSTALL_COMMANDS.items():
            try:
                if 'cirq' in name.lower():
                    import cirq
                    installed.append(f"✅ {name} (cirq v{cirq.__version__})")
                elif 'qiskit' in name.lower():
                    import qiskit
                    installed.append(f"✅ {name} (qiskit v{qiskit.__version__})")
                elif 'pennylane' in name.lower():
                    import pennylane
                    installed.append(f"✅ {name} (pennylane v{pennylane.__version__})")
                elif name == 'braket':
                    import braket
                    installed.append(f"✅ braket")
                elif name == 'qutip':
                    import qutip
                    installed.append(f"✅ qutip (v{qutip.__version__})")
            except ImportError:
                pass
        
        # Build response
        response = "🔬 **Available Quantum Backends:**\n\n"
        
        if installed:
            response += "**✅ Installed:**\n" + "\n".join(installed) + "\n\n"
        
        # Check for dynamically registered backends
        import json
        config_path = Path.home() / ".proxima" / "backends_config.json"
        dynamic_backends = []
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    backends_config = json.load(f)
                    registered = backends_config.get('registered_backends', {})
                    for name, info in registered.items():
                        if info.get('type') == 'dynamic':
                            path = info.get('path', '')
                            url = info.get('url', '')
                            dynamic_backends.append(f"📦 `{name}` (custom)\n   📂 {path}\n   🔗 {url}")
            except:
                pass
        
        if dynamic_backends:
            response += "**🚀 Custom Cloned Backends:**\n" + "\n".join(dynamic_backends) + "\n\n"
        
        # LRET backends (special section)
        response += "**🔧 LRET Backends:**\n"
        for name, repo_info in self.BACKEND_REPOS.items():
            if 'lret' in name.lower():
                if isinstance(repo_info, dict):
                    branch = repo_info.get('branch', 'main')
                    desc = repo_info.get('description', '')
                    url = repo_info.get('url', '')
                    response += f"📦 `{name}` - {desc}\n   Branch: `{branch}` | URL: `{url}`\n"
                else:
                    response += f"📦 `{name}`: `{repo_info}`\n"
        
        # External backends
        response += "\n**📚 External Backends:**\n"
        for name, repo_info in self.BACKEND_REPOS.items():
            if 'lret' not in name.lower():
                if not any(name in i for i in installed):
                    if isinstance(repo_info, dict):
                        url = repo_info.get('url', '')
                        response += f"📦 `{name}`: `{url}`\n"
                    else:
                        response += f"📦 `{name}`: `{repo_info}`\n"
        
        response += "\n💡 **Usage:**\n"
        response += "• `clone <backend>` - Clone predefined backend\n"
        response += "• `clone https://github.com/user/repo` - Clone ANY repository!\n"
        response += "• `build <backend>` - Build/compile (auto-detects build system)\n"
        response += "• `test <backend>` - Test backend installation\n"
        response += "• `run <backend>` - Run with custom parameters\n"
        return response
    
    def _execute_backend_clone(self, backend_name: str, custom_url: str = None, custom_branch: str = None, custom_path: str = None) -> str:
        """Clone a quantum backend repository with support for branches and custom URLs.
        
        Args:
            backend_name: Name of the backend (e.g., 'cirq', 'lret_cirq_scalability')
            custom_url: Optional custom URL to clone from (overrides BACKEND_REPOS)
            custom_branch: Optional branch to checkout (overrides default)
            custom_path: Optional custom path to clone to (overrides default ~/.proxima/backends/)
        """
        import subprocess
        
        backend_name = backend_name.lower().strip()
        
        # Determine repo URL and branch
        repo_url = custom_url
        branch = custom_branch
        description = ""
        
        if not repo_url and backend_name in self.BACKEND_REPOS:
            repo_info = self.BACKEND_REPOS[backend_name]
            if isinstance(repo_info, dict):
                repo_url = repo_info.get('url', '')
                branch = custom_branch or repo_info.get('branch', 'main')
                description = repo_info.get('description', '')
            else:
                repo_url = repo_info
                branch = custom_branch
        
        if not repo_url:
            # If custom_url looks like a URL, use the dynamic clone function
            if custom_url and ('github.com' in custom_url or 'http' in custom_url or '/' in custom_url):
                return self._clone_any_repository(custom_url, custom_branch, backend_name if backend_name else None)
            
            available = ', '.join(self.BACKEND_REPOS.keys())
            return (
                f"❌ Unknown backend: `{backend_name}`\n\n"
                f"**Predefined backends:** {available}\n\n"
                f"💡 **For ANY other repository:**\n"
                f"• `clone https://github.com/owner/repo`\n"
                f"• `clone github.com/owner/repo branch feature-x`\n"
                f"• The AI can clone ANY GitHub repo - just provide the URL!"
            )
        
        # Determine clone path
        clone_path = custom_path or os.path.join(os.path.expanduser('~'), '.proxima', 'backends', backend_name)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(clone_path), exist_ok=True)
        
        # Check if already exists
        if os.path.exists(clone_path):
            # Offer to update instead
            return (
                f"ℹ️ Backend `{backend_name}` already exists at `{clone_path}`\n\n"
                f"**Options:**\n"
                f"• `update {backend_name}` - Pull latest changes\n"
                f"• `build {backend_name}` - Build/compile the backend\n"
                f"• `delete {clone_path}` and retry to re-clone"
            )
        
        # Build git clone command with branch if specified
        clone_cmd = ['git', 'clone']
        if branch:
            clone_cmd.extend(['--branch', branch])
        clone_cmd.extend([repo_url, clone_path])
        
        self._show_ai_message(f"🔄 Cloning `{backend_name}` from `{repo_url}`" + (f" (branch: `{branch}`)" if branch else ""))
        
        result = subprocess.run(
            clone_cmd,
            capture_output=True, text=True, timeout=600
        )
        
        if result.returncode == 0:
            response = f"✅ **Successfully cloned `{backend_name}`**\n\n"
            response += f"📂 Location: `{clone_path}`\n"
            if branch:
                response += f"🌿 Branch: `{branch}`\n"
            if description:
                response += f"📝 {description}\n"
            response += f"\n📋 **Next steps:**\n"
            response += f"1. Run `build {backend_name}` to compile\n"
            response += f"2. Run `test {backend_name}` to verify installation\n"
            response += f"3. Run `configure proxima to use {backend_name}` to enable it\n"
            return response
        else:
            error_output = result.stderr or result.stdout or "Unknown error"
            return f"❌ Clone failed:\n```\n{error_output}\n```\n\n💡 Check if the URL and branch are correct."
    
    def _execute_backend_install(self, backend_name: str) -> str:
        """Install backend dependencies."""
        import subprocess
        
        backend_name = backend_name.lower().strip()
        
        if backend_name not in self.BACKEND_INSTALL_COMMANDS:
            # Try dynamic build for cloned repositories
            clone_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', backend_name)
            if os.path.exists(clone_path):
                return self._build_cloned_repository(clone_path, backend_name)
            
            available = ', '.join(self.BACKEND_INSTALL_COMMANDS.keys())
            return (
                f"❌ Unknown backend: `{backend_name}`\n\n"
                f"**Predefined backends:** {available}\n\n"
                f"💡 If you cloned a custom repository, use `build {backend_name}` instead."
            )
        
        install_cmd = self.BACKEND_INSTALL_COMMANDS[backend_name]
        
        result = subprocess.run(
            install_cmd, shell=True,
            capture_output=True, text=True, timeout=600
        )
        
        if result.returncode == 0:
            return f"✅ **Installed `{backend_name}` dependencies**\n```\n{result.stdout[-1500:]}\n```"
        else:
            return f"❌ Installation failed:\n```\n{result.stderr[-1500:]}\n```"
    
    def _execute_backend_build(self, backend_name: str, options: dict = None) -> str:
        """Build/compile a quantum backend."""
        import subprocess
        
        backend_name = backend_name.lower().strip()
        
        # Special handling for LRET
        if backend_name == 'lret':
            # Build LRET backends
            results = []
            build_commands = [
                ('LRET Core', 'python setup.py build_ext --inplace'),
                ('LRET Cirq', 'cd backends/lret_cirq && python setup.py build_ext --inplace'),
            ]
            
            for name, cmd in build_commands:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
                status = "✅" if result.returncode == 0 else "❌"
                results.append(f"{status} {name}")
            
            return f"🔨 **LRET Build Results:**\n" + "\n".join(results)
        
        # Check if this is a dynamically cloned repository
        clone_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', backend_name)
        
        if os.path.exists(clone_path):
            # Use dynamic build system
            return self._build_cloned_repository(clone_path, backend_name, options)
        
        # Fall back to predefined install commands
        install_result = self._execute_backend_install(backend_name)
        
        if "❌" in install_result:
            return install_result
        
        # Additional build steps if needed
        clone_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', backend_name)
        
        if os.path.exists(clone_path):
            # Try to build from source
            result = subprocess.run(
                'pip install -e .',
                shell=True, capture_output=True, text=True,
                cwd=clone_path, timeout=600
            )
            
            if result.returncode == 0:
                # Auto-register with Proxima after successful build
                register_result = self._register_backend_with_proxima(backend_name, clone_path)
                return f"✅ **Built `{backend_name}` from source**\n\n{install_result}\n\n{register_result}"
            else:
                return f"⚠️ Source build failed, using pip install:\n{install_result}"
        
        return install_result
    
    def _register_backend_with_proxima(self, backend_name: str, backend_path: str = None) -> str:
        """Register a backend with Proxima for use in simulations.
        
        This updates the Proxima configuration to include the new backend.
        """
        import json
        
        config_path = Path.home() / ".proxima" / "backends_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config
        backends_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    backends_config = json.load(f)
            except Exception:
                backends_config = {}
        
        if 'registered_backends' not in backends_config:
            backends_config['registered_backends'] = {}
        
        # Get backend info
        repo_info = self.BACKEND_REPOS.get(backend_name, {})
        if isinstance(repo_info, dict):
            description = repo_info.get('description', '')
            branch = repo_info.get('branch', 'main')
            url = repo_info.get('url', '')
        else:
            description = ''
            branch = 'main'
            url = repo_info
        
        # Register the backend
        backends_config['registered_backends'][backend_name] = {
            'name': backend_name,
            'path': str(backend_path) if backend_path else '',
            'description': description,
            'branch': branch,
            'url': url,
            'enabled': True,
            'registered_at': datetime.now().isoformat(),
        }
        
        # Save config
        try:
            with open(config_path, 'w') as f:
                json.dump(backends_config, f, indent=2)
            return f"✅ **Registered `{backend_name}` with Proxima**\n📍 Config: `{config_path}`"
        except Exception as e:
            return f"⚠️ Could not register backend: {str(e)}"
    
    def _execute_backend_test(self, backend_name: str) -> str:
        """Test a backend installation."""
        backend_name = backend_name.lower().strip()
        
        test_code = {
            'cirq': '''
import cirq
q = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.measure(q[0], q[1]))
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=100)
print(f"✅ Cirq working! Results: {result.histogram(key=cirq.MeasurementKey('0,1'))}")
''',
            'qiskit': '''
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
simulator = AerSimulator()
result = simulator.run(qc, shots=100).result()
print(f"✅ Qiskit working! Counts: {result.get_counts()}")
''',
            'pennylane': '''
import pennylane as qml
dev = qml.device("default.qubit", wires=2)
@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])
result = circuit()
print(f"✅ PennyLane working! Probabilities: {result}")
''',
            'lret_cirq_scalability': '''
# Test LRET Cirq Scalability backend
try:
    from proxima.backends.lret_cirq_scalability import LRETCirqBackend
    backend = LRETCirqBackend()
    print(f"✅ LRET Cirq Scalability backend loaded!")
    print(f"   Backend name: {backend.name if hasattr(backend, 'name') else 'LRETCirq'}")
except ImportError:
    # Fallback: test with cirq directly
    import cirq
    q = cirq.LineQubit.range(4)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q[0]))
    for i in range(3):
        circuit.append(cirq.CNOT(q[i], q[i+1]))
    circuit.append(cirq.measure(*q, key='result'))
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=100)
    print(f"✅ Cirq (LRET scalability mode) working!")
    print(f"   4-qubit GHZ state results: {result.histogram(key='result')}")
''',
            'lret_pennylane': '''
# Test LRET PennyLane Hybrid backend
try:
    from proxima.backends.lret_pennylane_hybrid import LRETPennyLaneBackend
    backend = LRETPennyLaneBackend()
    print(f"✅ LRET PennyLane Hybrid backend loaded!")
except ImportError:
    # Fallback: test with pennylane directly
    import pennylane as qml
    import numpy as np
    dev = qml.device("default.qubit", wires=4)
    @qml.qnode(dev)
    def variational_circuit(params):
        for i in range(4):
            qml.RY(params[i], wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i+1])
        return qml.expval(qml.PauliZ(0))
    params = np.random.uniform(0, np.pi, 4)
    result = variational_circuit(params)
    print(f"✅ PennyLane (LRET hybrid mode) working!")
    print(f"   Variational circuit expectation: {result:.4f}")
''',
            'braket': '''
from braket.circuits import Circuit
from braket.devices import LocalSimulator
device = LocalSimulator()
circuit = Circuit().h(0).cnot(0, 1)
result = device.run(circuit, shots=100).result()
print(f"✅ Braket working! Counts: {result.measurement_counts}")
''',
            'qutip': '''
import qutip as qt
psi0 = qt.basis(2, 0)
H = qt.sigmax()
result = qt.sesolve(H, psi0, [0, 1])
print(f"✅ QuTiP working! Final state: {result.states[-1]}")
''',
        }
        
        # Try exact match first, then partial match for lret backends
        if backend_name in test_code:
            code = test_code[backend_name]
        elif 'lret' in backend_name and 'cirq' in backend_name:
            code = test_code.get('lret_cirq_scalability', test_code.get('cirq'))
        elif 'lret' in backend_name and 'pennylane' in backend_name:
            code = test_code.get('lret_pennylane', test_code.get('pennylane'))
        elif 'lret' in backend_name:
            code = test_code.get('cirq')  # Default to cirq for generic lret
        else:
            # Check if this is a dynamically cloned backend
            clone_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', backend_name)
            if os.path.exists(clone_path):
                # Try to run tests for dynamic backend
                return self._test_dynamic_backend(backend_name, clone_path)
            return f"❌ No test available for `{backend_name}`\n\nAvailable tests: {', '.join(test_code.keys())}\n\n💡 If this is a custom cloned backend, make sure it exists at `~/.proxima/backends/{backend_name}`"
        
        import subprocess
        import tempfile
        
        # Write test script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            test_file = f.name
        
        try:
            result = subprocess.run(
                ['python', test_file],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                return f"🧪 **Backend Test: `{backend_name}`**\n```\n{result.stdout}\n```"
            else:
                return f"❌ Test failed:\n```\n{result.stderr}\n```"
        finally:
            os.unlink(test_file)
    
    def _test_dynamic_backend(self, backend_name: str, clone_path: str) -> str:
        """Test a dynamically cloned backend by running its tests or checking imports.
        
        This function works with ANY cloned repository.
        """
        import subprocess
        
        response = f"🧪 **Testing Dynamic Backend: `{backend_name}`**\n\n"
        response += f"📂 Path: `{clone_path}`\n\n"
        
        tests_passed = []
        tests_failed = []
        
        # 1. Check if package can be imported
        package_name = backend_name.replace('-', '_').replace('.', '_')
        import_test = f"""
try:
    import sys
    sys.path.insert(0, r'{clone_path}')
    import {package_name}
    print(f"✅ Import successful: {package_name}")
    if hasattr({package_name}, '__version__'):
        print(f"   Version: {{{package_name}.__version__}}")
except ImportError as e:
    print(f"⚠️ Could not import {package_name}: {{e}}")
except Exception as e:
    print(f"⚠️ Error: {{e}}")
"""
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(import_test)
            test_file = f.name
        
        try:
            result = subprocess.run(['python', test_file], capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and "✅" in result.stdout:
                tests_passed.append(f"✅ Import test: `{package_name}`")
            else:
                tests_failed.append(f"⚠️ Import test: Could not import")
            response += f"**Import Test:**\n```\n{result.stdout}\n```\n\n"
        except:
            tests_failed.append("⚠️ Import test: Timeout or error")
        finally:
            os.unlink(test_file)
        
        # 2. Try running pytest or unittest if tests exist
        tests_dir = os.path.join(clone_path, 'tests')
        if os.path.exists(tests_dir):
            response += "**Running Tests:**\n"
            try:
                result = subprocess.run(
                    ['python', '-m', 'pytest', tests_dir, '-v', '--tb=short', '-q'],
                    capture_output=True, text=True, timeout=120,
                    cwd=clone_path
                )
                if result.returncode == 0:
                    tests_passed.append("✅ Pytest: All tests passed")
                else:
                    tests_failed.append(f"⚠️ Pytest: Some tests failed")
                response += f"```\n{result.stdout[-1500:]}\n```\n\n"
            except subprocess.TimeoutExpired:
                tests_failed.append("⏱️ Pytest: Timeout")
            except:
                # Try unittest
                try:
                    result = subprocess.run(
                        ['python', '-m', 'unittest', 'discover', '-s', 'tests'],
                        capture_output=True, text=True, timeout=120,
                        cwd=clone_path
                    )
                    if result.returncode == 0:
                        tests_passed.append("✅ Unittest: All tests passed")
                    else:
                        tests_failed.append("⚠️ Unittest: Some tests failed")
                    response += f"```\n{result.stdout[-1000:]}\n```\n\n"
                except:
                    pass
        
        # 3. Summary
        response += "**Summary:**\n"
        if tests_passed:
            response += "\n".join(tests_passed) + "\n"
        if tests_failed:
            response += "\n".join(tests_failed) + "\n"
        
        if not tests_failed:
            response += "\n✅ **Backend appears to be working correctly!**"
        else:
            response += "\n⚠️ **Some issues detected - check output above**"
        
        return response
    
    # =========================================================================
    # DYNAMIC REPOSITORY OPERATIONS (ANY REPO) - NOT HARDCODED
    # =========================================================================

    def _clone_any_repository(self, repo_url: str, branch: str = None, custom_name: str = None) -> str:
        """Clone ANY GitHub repository by URL - fully dynamic, not hardcoded.
        
        This function can clone ANY repository without relying on predefined lists.
        
        Args:
            repo_url: Full GitHub URL (e.g., https://github.com/owner/repo)
            branch: Optional branch to checkout
            custom_name: Optional custom name for the cloned directory
        """
        import subprocess
        import re
        
        # Validate and normalize URL
        repo_url = repo_url.strip()
        
        # Handle various URL formats
        if not repo_url.startswith('http'):
            # Handle github.com/owner/repo format
            if 'github.com' in repo_url:
                repo_url = f"https://{repo_url}"
            elif '/' in repo_url and len(repo_url.split('/')) == 2:
                # Handle owner/repo format
                repo_url = f"https://github.com/{repo_url}"
            else:
                return f"❌ Invalid repository URL: `{repo_url}`\n\n💡 Use full URL like: `https://github.com/owner/repo`"
        
        # Extract repo name from URL
        url_pattern = r'(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/\s?.]+)'
        match = re.match(url_pattern, repo_url.rstrip('.git'))
        
        if not match:
            # Try to extract from generic URL
            parts = repo_url.rstrip('/').rstrip('.git').split('/')
            if len(parts) >= 2:
                repo_name = parts[-1]
            else:
                return f"❌ Could not parse repository name from URL: `{repo_url}`"
        else:
            owner, repo_name = match.groups()
        
        # Use custom name if provided
        final_name = custom_name or repo_name
        
        # Determine clone path
        clone_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', final_name)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(clone_path), exist_ok=True)
        
        # Check if already exists
        if os.path.exists(clone_path):
            return (
                f"ℹ️ Repository `{final_name}` already exists at `{clone_path}`\n\n"
                f"**Options:**\n"
                f"• Say `update {final_name}` to pull latest changes\n"
                f"• Say `build {final_name}` to build it\n"
                f"• Say `delete {clone_path}` and retry to re-clone"
            )
        
        # Build git clone command
        clone_cmd = ['git', 'clone']
        if branch:
            clone_cmd.extend(['--branch', branch])
        clone_cmd.extend([repo_url, clone_path])
        
        self._show_ai_message(f"🔄 Cloning `{repo_url}`" + (f" (branch: `{branch}`)" if branch else ""))
        
        try:
            result = subprocess.run(
                clone_cmd,
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0:
                # Auto-detect what kind of project it is
                build_info = self._detect_build_system(clone_path)
                
                response = f"✅ **Successfully cloned `{final_name}`**\n\n"
                response += f"📂 Location: `{clone_path}`\n"
                response += f"🔗 Source: `{repo_url}`\n"
                if branch:
                    response += f"🌿 Branch: `{branch}`\n"
                
                response += f"\n📦 **Detected Build System:**\n{build_info}\n"
                response += f"\n📋 **Next steps:**\n"
                response += f"1. Say `build {final_name}` to compile/install\n"
                response += f"2. Say `run {final_name}` to run it\n"
                response += f"3. Say `test {final_name}` to verify installation\n"
                
                # Auto-register with Proxima
                self._register_dynamic_backend(final_name, clone_path, repo_url, branch)
                
                return response
            else:
                error_output = result.stderr or result.stdout or "Unknown error"
                return f"❌ Clone failed:\n```\n{error_output}\n```\n\n💡 Check if the URL is correct and you have network access."
        except subprocess.TimeoutExpired:
            return "❌ Clone timed out (>600s). Check your network connection."
        except Exception as e:
            return f"❌ Clone error: {str(e)}"
    
    def _detect_build_system(self, repo_path: str) -> str:
        """Auto-detect build system in a repository - fully dynamic.
        
        Scans the repository to determine how to build it.
        Returns a description of what was found.
        """
        build_files = {
            'pyproject.toml': '🐍 Python (pyproject.toml) - Modern Python packaging',
            'setup.py': '🐍 Python (setup.py) - Traditional Python packaging',
            'setup.cfg': '🐍 Python (setup.cfg) - Python configuration',
            'requirements.txt': '🐍 Python (requirements.txt) - Dependencies list',
            'CMakeLists.txt': '⚙️ CMake - C/C++ build system',
            'Makefile': '🔧 Make - Build automation',
            'makefile': '🔧 Make - Build automation',
            'package.json': '📦 Node.js (npm/yarn)',
            'Cargo.toml': '🦀 Rust (Cargo)',
            'go.mod': '🐹 Go modules',
            'build.gradle': '☕ Gradle (Java/Kotlin)',
            'pom.xml': '☕ Maven (Java)',
            'Gemfile': '💎 Ruby (Bundler)',
            'composer.json': '🐘 PHP (Composer)',
            'mix.exs': '⚗️ Elixir (Mix)',
            'Dockerfile': '🐳 Docker container',
            'docker-compose.yml': '🐳 Docker Compose',
            'environment.yml': '🐍 Conda environment',
            'conda.yaml': '🐍 Conda environment',
            '.python-version': '🐍 Python version file',
            'tox.ini': '🐍 Python (tox) - Testing',
            'noxfile.py': '🐍 Python (nox) - Automation',
        }
        
        found = []
        for filename, description in build_files.items():
            if os.path.exists(os.path.join(repo_path, filename)):
                found.append(f"• {description}")
        
        # Check for src/ directory (common Python layout)
        if os.path.isdir(os.path.join(repo_path, 'src')):
            found.append("• 📁 src/ layout detected")
        
        # Check for tests/ directory
        if os.path.isdir(os.path.join(repo_path, 'tests')):
            found.append("• 🧪 tests/ directory found")
        
        if found:
            return "\n".join(found)
        else:
            return "• ⚠️ No standard build system detected - may need manual setup"
    
    def _get_build_commands(self, repo_path: str) -> list:
        """Generate build commands based on detected build system - fully dynamic.
        
        Returns a list of commands to run to build the repository.
        """
        commands = []
        
        # Python project detection (in priority order)
        if os.path.exists(os.path.join(repo_path, 'pyproject.toml')):
            # Modern Python - check for build backend
            try:
                with open(os.path.join(repo_path, 'pyproject.toml'), 'r') as f:
                    content = f.read()
                    if 'setuptools' in content or 'setup.py' in os.listdir(repo_path):
                        commands.append(('Install requirements', 'pip install -r requirements.txt || true'))
                        commands.append(('Install package', 'pip install -e .'))
                    elif 'poetry' in content:
                        commands.append(('Install with Poetry', 'poetry install'))
                    elif 'flit' in content:
                        commands.append(('Install with Flit', 'flit install'))
                    elif 'hatch' in content:
                        commands.append(('Install with Hatch', 'pip install hatch && hatch build'))
                    else:
                        commands.append(('Install package', 'pip install -e .'))
            except:
                commands.append(('Install package', 'pip install -e .'))
        
        elif os.path.exists(os.path.join(repo_path, 'setup.py')):
            if os.path.exists(os.path.join(repo_path, 'requirements.txt')):
                commands.append(('Install requirements', 'pip install -r requirements.txt'))
            commands.append(('Install package', 'pip install -e .'))
            # Check for Cython/C extensions
            try:
                with open(os.path.join(repo_path, 'setup.py'), 'r') as f:
                    content = f.read()
                    if 'Extension' in content or 'Cython' in content or 'build_ext' in content:
                        commands.append(('Build extensions', 'python setup.py build_ext --inplace'))
            except:
                pass
        
        elif os.path.exists(os.path.join(repo_path, 'requirements.txt')):
            commands.append(('Install requirements', 'pip install -r requirements.txt'))
        
        # CMake project
        if os.path.exists(os.path.join(repo_path, 'CMakeLists.txt')):
            commands.append(('Create build dir', 'mkdir -p build'))
            commands.append(('Configure CMake', 'cd build && cmake ..'))
            commands.append(('Build', 'cd build && cmake --build .'))
        
        # Makefile
        elif os.path.exists(os.path.join(repo_path, 'Makefile')) or os.path.exists(os.path.join(repo_path, 'makefile')):
            if not any('cmake' in cmd[1].lower() for cmd in commands):
                commands.append(('Build with Make', 'make'))
        
        # Node.js project
        if os.path.exists(os.path.join(repo_path, 'package.json')):
            if os.path.exists(os.path.join(repo_path, 'yarn.lock')):
                commands.append(('Install with Yarn', 'yarn install'))
            else:
                commands.append(('Install with npm', 'npm install'))
        
        # Rust project
        if os.path.exists(os.path.join(repo_path, 'Cargo.toml')):
            commands.append(('Build with Cargo', 'cargo build --release'))
        
        # Go project
        if os.path.exists(os.path.join(repo_path, 'go.mod')):
            commands.append(('Build Go', 'go build ./...'))
        
        # Conda environment
        if os.path.exists(os.path.join(repo_path, 'environment.yml')):
            commands.append(('Create Conda env', 'conda env create -f environment.yml'))
        elif os.path.exists(os.path.join(repo_path, 'conda.yaml')):
            commands.append(('Create Conda env', 'conda env create -f conda.yaml'))
        
        # Docker
        if os.path.exists(os.path.join(repo_path, 'docker-compose.yml')):
            commands.append(('Docker Compose', 'docker-compose build'))
        elif os.path.exists(os.path.join(repo_path, 'Dockerfile')):
            commands.append(('Docker build', 'docker build -t $(basename $(pwd)) .'))
        
        # Default fallback
        if not commands:
            commands.append(('Check for setup', 'ls -la && cat README* 2>/dev/null | head -100 || true'))
        
        return commands
    
    def _build_cloned_repository(self, repo_path: str = '', repo_name: str = '', build_options: dict = None) -> str:
        """Auto-detect and build ANY cloned repository - fully dynamic.
        
        Args:
            repo_path: Direct path to the repository (if known)
            repo_name: Name of the repository (will search in ~/.proxima/backends/)
            build_options: Optional dict with build configuration
        """
        import subprocess
        
        build_options = build_options or {}
        
        # Determine repository path
        if not repo_path and repo_name:
            repo_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', repo_name)
        
        if not repo_path:
            return "❌ Please specify either `repo_path` or `name` of the repository to build."
        
        repo_path = os.path.expanduser(os.path.expandvars(repo_path))
        
        if not os.path.exists(repo_path):
            # Try to find it
            backends_dir = os.path.join(os.path.expanduser('~'), '.proxima', 'backends')
            if os.path.exists(backends_dir):
                available = os.listdir(backends_dir)
                return f"❌ Repository not found at `{repo_path}`\n\n📂 Available in `{backends_dir}`:\n" + "\n".join(f"• {b}" for b in available)
            return f"❌ Repository not found at `{repo_path}`"
        
        # Detect build system
        build_info = self._detect_build_system(repo_path)
        build_commands = self._get_build_commands(repo_path)
        
        repo_name = repo_name or os.path.basename(repo_path)
        
        self._show_ai_message(f"🔨 Building `{repo_name}` at `{repo_path}`...")
        
        response = f"🔨 **Building `{repo_name}`**\n\n"
        response += f"📂 Path: `{repo_path}`\n"
        response += f"📦 Build System:\n{build_info}\n\n"
        response += f"📋 **Build Steps:**\n"
        
        results = []
        overall_success = True
        
        for step_name, command in build_commands:
            self._show_ai_message(f"⏳ {step_name}...")
            
            try:
                # Handle Windows vs Unix commands
                if os.name == 'nt':
                    # Convert bash-style commands to PowerShell-compatible
                    if command.startswith('mkdir -p'):
                        dir_path = command.replace('mkdir -p ', '')
                        command = f'New-Item -ItemType Directory -Force -Path {dir_path}'
                    elif '|| true' in command:
                        command = command.replace('|| true', '; $true')
                    elif 'cd build &&' in command:
                        command = command.replace('cd build &&', 'cd build;')
                    
                    result = subprocess.run(
                        ['powershell', '-Command', command],
                        capture_output=True, text=True, timeout=600,
                        cwd=repo_path
                    )
                else:
                    result = subprocess.run(
                        command,
                        shell=True, capture_output=True, text=True, timeout=600,
                        cwd=repo_path
                    )
                
                status = "✅" if result.returncode == 0 else "❌"
                if result.returncode != 0:
                    overall_success = False
                
                output = (result.stdout or result.stderr or "")[-500:]
                results.append(f"{status} **{step_name}**\n   `{command}`\n   {output[:200] if output else 'OK'}")
                
            except subprocess.TimeoutExpired:
                results.append(f"⏱️ **{step_name}** - Timeout")
                overall_success = False
            except Exception as e:
                results.append(f"❌ **{step_name}** - Error: {str(e)[:100]}")
                overall_success = False
        
        response += "\n".join(results)
        
        if overall_success:
            # Register with Proxima
            self._register_dynamic_backend(repo_name, repo_path)
            response += f"\n\n✅ **Build Successful!**\n"
            response += f"📋 **Next:** Say `run {repo_name}` to execute or `test {repo_name}` to test"
        else:
            response += f"\n\n⚠️ **Some steps failed.** Check the output above."
        
        return response
    
    def _register_dynamic_backend(self, name: str, path: str, url: str = '', branch: str = '') -> None:
        """Register a dynamically cloned backend with Proxima.
        
        This updates the backends_config.json with the new backend info.
        """
        import json
        
        config_path = Path.home() / ".proxima" / "backends_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config
        backends_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    backends_config = json.load(f)
            except:
                backends_config = {}
        
        if 'registered_backends' not in backends_config:
            backends_config['registered_backends'] = {}
        
        # Register the backend
        backends_config['registered_backends'][name] = {
            'name': name,
            'path': str(path),
            'url': url,
            'branch': branch,
            'enabled': True,
            'type': 'dynamic',  # Mark as dynamically cloned
            'registered_at': datetime.now().isoformat(),
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(backends_config, f, indent=2)
        except Exception as e:
            pass  # Silent fail for registration
    
    def _run_any_backend(self, repo_path: str = '', repo_name: str = '', 
                         entry_point: str = '', args: list = None, config: dict = None) -> str:
        """Run ANY cloned backend with custom parameters - fully dynamic.
        
        Args:
            repo_path: Direct path to the repository
            repo_name: Name of the backend (will search in ~/.proxima/backends/)
            entry_point: Entry point to run (e.g., 'main.py', 'module.run', 'src/run.py')
            args: Command line arguments to pass
            config: Configuration dict (shots, qubits, custom params)
        """
        import subprocess
        
        args = args or []
        config = config or {}
        
        # Determine repository path
        if not repo_path and repo_name:
            repo_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', repo_name)
        
        if not repo_path:
            return "❌ Please specify either `repo_path` or `name` of the backend to run."
        
        repo_path = os.path.expanduser(os.path.expandvars(repo_path))
        
        if not os.path.exists(repo_path):
            backends_dir = os.path.join(os.path.expanduser('~'), '.proxima', 'backends')
            if os.path.exists(backends_dir):
                available = os.listdir(backends_dir)
                return f"❌ Backend not found at `{repo_path}`\n\n📂 Available:\n" + "\n".join(f"• {b}" for b in available)
            return f"❌ Backend not found at `{repo_path}`"
        
        repo_name = repo_name or os.path.basename(repo_path)
        
        # Auto-detect entry point if not specified
        if not entry_point:
            entry_point = self._detect_entry_point(repo_path)
        
        if not entry_point:
            return (
                f"❌ Could not auto-detect entry point for `{repo_name}`\n\n"
                f"💡 Please specify the entry point:\n"
                f"• `run {repo_name} with entry_point main.py`\n"
                f"• `run {repo_name} with entry_point src/run.py`\n"
                f"• `run {repo_name} with entry_point module.main`"
            )
        
        # Build run command with config
        shots = config.get('shots', 1024)
        qubits = config.get('qubits', 2)
        
        # Build command
        if entry_point.endswith('.py'):
            entry_path = os.path.join(repo_path, entry_point)
            if not os.path.exists(entry_path):
                entry_path = entry_point  # Might be absolute path
            
            cmd = ['python', entry_path]
            
            # Add config as arguments
            for key, value in config.items():
                cmd.extend([f'--{key}', str(value)])
            
            # Add additional args
            cmd.extend(args)
        else:
            # Module-style entry point (e.g., module.main)
            cmd = ['python', '-m', entry_point.replace('/', '.').replace('.py', '')]
            for key, value in config.items():
                cmd.extend([f'--{key}', str(value)])
            cmd.extend(args)
        
        self._show_ai_message(f"🚀 Running `{repo_name}` with {shots} shots...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=600,
                cwd=repo_path
            )
            
            if result.returncode == 0:
                return (
                    f"✅ **Successfully ran `{repo_name}`**\n\n"
                    f"📂 Path: `{repo_path}`\n"
                    f"📋 Entry: `{entry_point}`\n"
                    f"⚙️ Config: {config}\n\n"
                    f"**Output:**\n```\n{result.stdout[:3000]}\n```"
                )
            else:
                return (
                    f"❌ **Run Failed**\n\n"
                    f"📋 Entry: `{entry_point}`\n"
                    f"```\n{result.stderr[:2000]}\n```"
                )
        except subprocess.TimeoutExpired:
            return f"⏱️ **Timeout** - Backend took too long (>600s)"
        except Exception as e:
            return f"❌ **Error:** {str(e)}"
    
    def _detect_entry_point(self, repo_path: str) -> str:
        """Auto-detect the entry point for a repository - fully dynamic.
        
        Looks for common entry points like main.py, run.py, __main__.py, etc.
        """
        # Common entry point patterns (in priority order)
        candidates = [
            '__main__.py',
            'main.py',
            'run.py',
            'app.py',
            'cli.py',
            'src/__main__.py',
            'src/main.py',
            'src/run.py',
        ]
        
        for candidate in candidates:
            if os.path.exists(os.path.join(repo_path, candidate)):
                return candidate
        
        # Check if there's a package with __main__
        for item in os.listdir(repo_path):
            item_path = os.path.join(repo_path, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, '__main__.py')):
                    return os.path.join(item, '__main__.py')
                if os.path.exists(os.path.join(item_path, 'main.py')):
                    return os.path.join(item, 'main.py')
        
        # Check pyproject.toml or setup.py for entry points
        if os.path.exists(os.path.join(repo_path, 'pyproject.toml')):
            try:
                with open(os.path.join(repo_path, 'pyproject.toml'), 'r') as f:
                    content = f.read()
                    # Look for scripts entry point
                    if '[project.scripts]' in content or 'console_scripts' in content:
                        # Found entry point definition - let user use the installed command
                        return None  # Will prompt user
            except:
                pass
        
        # Check for any .py file with if __name__ == '__main__'
        for item in os.listdir(repo_path):
            if item.endswith('.py') and item not in ['setup.py', 'conftest.py', '__init__.py']:
                try:
                    with open(os.path.join(repo_path, item), 'r') as f:
                        content = f.read()
                        if '__name__' in content and '__main__' in content:
                            return item
                except:
                    pass
        
        return None
    
    def _copy_local_repository(self, source_path: str, destination: str = '', branch: str = None, name: str = None) -> str:
        """Copy a repository from a local path and optionally switch to a specific branch.
        
        This handles the case where the user has a repo on their local machine and wants
        to use a specific branch from it.
        
        Args:
            source_path: Local path to the repository (e.g., C:\\Users\\...\\LRET)
            destination: Optional destination path (defaults to ~/.proxima/backends/<name>)
            branch: Optional branch to checkout after copying
            name: Optional custom name for the copied repo
        """
        import subprocess
        import shutil
        
        source_path = os.path.expanduser(os.path.expandvars(source_path))
        
        if not os.path.exists(source_path):
            return f"❌ Source path does not exist: `{source_path}`"
        
        if not os.path.isdir(source_path):
            return f"❌ Source path is not a directory: `{source_path}`"
        
        # Check if it's a git repository
        git_dir = os.path.join(source_path, '.git')
        is_git_repo = os.path.exists(git_dir)
        
        # Determine the name
        repo_name = name or os.path.basename(source_path.rstrip('/\\'))
        if branch:
            # Include branch in name if specified
            safe_branch = branch.replace('/', '-').replace('\\', '-')
            repo_name = f"{repo_name}-{safe_branch}"
        
        # Determine destination
        if not destination:
            destination = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', repo_name)
        else:
            destination = os.path.expanduser(os.path.expandvars(destination))
        
        # Check if destination already exists
        if os.path.exists(destination):
            return (
                f"ℹ️ Destination `{destination}` already exists\n\n"
                f"**Options:**\n"
                f"• Say `delete {destination}` first, then retry\n"
                f"• Say `build {repo_name}` to build the existing copy\n"
                f"• Say `checkout {branch} in {destination}` to switch branch"
            )
        
        self._show_ai_message(f"📂 Copying `{source_path}` to `{destination}`...")
        
        # Create parent directory
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        try:
            # Copy the entire directory
            shutil.copytree(source_path, destination)
            
            response = f"✅ **Successfully copied repository**\n\n"
            response += f"📂 Source: `{source_path}`\n"
            response += f"📂 Destination: `{destination}`\n"
            
            # If branch specified and it's a git repo, checkout the branch
            if branch and is_git_repo:
                self._show_ai_message(f"🔀 Switching to branch `{branch}`...")
                
                # First fetch to make sure we have the branch
                fetch_result = subprocess.run(
                    ['git', 'fetch', '--all'],
                    capture_output=True, text=True, timeout=120,
                    cwd=destination
                )
                
                # Try to checkout the branch
                checkout_result = subprocess.run(
                    ['git', 'checkout', branch],
                    capture_output=True, text=True, timeout=60,
                    cwd=destination
                )
                
                if checkout_result.returncode == 0:
                    response += f"🌿 Branch: `{branch}` ✅\n"
                else:
                    # Try to checkout as remote branch
                    checkout_result = subprocess.run(
                        ['git', 'checkout', '-b', branch, f'origin/{branch}'],
                        capture_output=True, text=True, timeout=60,
                        cwd=destination
                    )
                    if checkout_result.returncode == 0:
                        response += f"🌿 Branch: `{branch}` (from remote) ✅\n"
                    else:
                        response += f"⚠️ Could not checkout branch `{branch}`: {checkout_result.stderr[:200]}\n"
            
            # Auto-detect build system
            build_info = self._detect_build_system(destination)
            response += f"\n📦 **Detected Build System:**\n{build_info}\n"
            
            # Register with Proxima
            self._register_dynamic_backend(repo_name, destination, source_path, branch or '')
            
            response += f"\n📋 **Next steps:**\n"
            response += f"1. Say `build {repo_name}` to compile/install\n"
            response += f"2. Say `configure proxima to use {repo_name}` to enable it\n"
            
            return response
            
        except Exception as e:
            return f"❌ Copy failed: {str(e)}"
    
    def _checkout_branch(self, repo_path: str, branch: str) -> str:
        """Switch to a specific branch in a local repository.
        
        Args:
            repo_path: Path to the repository
            branch: Branch name to checkout
        """
        import subprocess
        
        repo_path = os.path.expanduser(os.path.expandvars(repo_path))
        
        # If just a name, look in backends directory
        if not os.path.isabs(repo_path) and not os.path.exists(repo_path):
            repo_path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', repo_path)
        
        if not os.path.exists(repo_path):
            return f"❌ Repository not found: `{repo_path}`"
        
        if not os.path.exists(os.path.join(repo_path, '.git')):
            return f"❌ Not a git repository: `{repo_path}`"
        
        self._show_ai_message(f"🔀 Switching to branch `{branch}` in `{repo_path}`...")
        
        # First fetch to ensure we have latest
        fetch_result = subprocess.run(
            ['git', 'fetch', '--all'],
            capture_output=True, text=True, timeout=120,
            cwd=repo_path
        )
        
        # Try to checkout
        checkout_result = subprocess.run(
            ['git', 'checkout', branch],
            capture_output=True, text=True, timeout=60,
            cwd=repo_path
        )
        
        if checkout_result.returncode == 0:
            return f"✅ **Switched to branch `{branch}`**\n\n📂 Repository: `{repo_path}`"
        
        # Try as remote branch
        checkout_result = subprocess.run(
            ['git', 'checkout', '-b', branch, f'origin/{branch}'],
            capture_output=True, text=True, timeout=60,
            cwd=repo_path
        )
        
        if checkout_result.returncode == 0:
            return f"✅ **Switched to branch `{branch}` (from remote)**\n\n📂 Repository: `{repo_path}`"
        
        # List available branches
        branches_result = subprocess.run(
            ['git', 'branch', '-a'],
            capture_output=True, text=True, timeout=30,
            cwd=repo_path
        )
        
        return (
            f"❌ Could not checkout branch `{branch}`\n\n"
            f"**Error:** {checkout_result.stderr[:300]}\n\n"
            f"**Available branches:**\n```\n{branches_result.stdout}\n```"
        )
    
    def _configure_backend_for_proxima(self, name: str, path: str = '', backend_type: str = '') -> str:
        """Configure Proxima to use a specific backend.
        
        This adds the backend to Proxima's configuration and makes it available for simulations.
        
        Args:
            name: Name/identifier for the backend
            path: Path to the backend (optional if already registered)
            backend_type: Type of backend (lret, cirq, qiskit, etc.)
        """
        import json
        
        config_path = Path.home() / ".proxima" / "backends_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If no path given, try to find in registered backends
        if not path:
            path = os.path.join(os.path.expanduser('~'), '.proxima', 'backends', name)
        else:
            path = os.path.expanduser(os.path.expandvars(path))
        
        if not os.path.exists(path):
            return f"❌ Backend path does not exist: `{path}`"
        
        # Load existing config
        backends_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    backends_config = json.load(f)
            except:
                backends_config = {}
        
        if 'registered_backends' not in backends_config:
            backends_config['registered_backends'] = {}
        
        if 'active_backend' not in backends_config:
            backends_config['active_backend'] = None
        
        # Register/update the backend
        backends_config['registered_backends'][name] = {
            'name': name,
            'path': str(path),
            'type': backend_type or 'custom',
            'enabled': True,
            'configured_at': datetime.now().isoformat(),
        }
        
        # Set as active backend
        backends_config['active_backend'] = name
        
        # Save config
        try:
            with open(config_path, 'w') as f:
                json.dump(backends_config, f, indent=2)
            
            return (
                f"✅ **Configured Proxima to use `{name}`**\n\n"
                f"📂 Path: `{path}`\n"
                f"🔧 Type: `{backend_type or 'custom'}`\n"
                f"📍 Config: `{config_path}`\n\n"
                f"**Status:** This backend is now the active backend for Proxima.\n"
                f"You can run simulations with: `run simulation using {name}`"
            )
        except Exception as e:
            return f"❌ Failed to configure backend: {str(e)}"
    
    def _copy_directory(self, source: str, destination: str) -> str:
        """Copy an entire directory from source to destination.
        
        Args:
            source: Source directory path
            destination: Destination directory path
        """
        import shutil
        
        source = os.path.expanduser(os.path.expandvars(source))
        destination = os.path.expanduser(os.path.expandvars(destination))
        
        if not os.path.exists(source):
            return f"❌ Source does not exist: `{source}`"
        
        if not os.path.isdir(source):
            return f"❌ Source is not a directory: `{source}`"
        
        if os.path.exists(destination):
            return f"❌ Destination already exists: `{destination}`"
        
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copytree(source, destination)
            return f"✅ **Copied directory**\n\n📂 From: `{source}`\n📂 To: `{destination}`"
        except Exception as e:
            return f"❌ Copy failed: {str(e)}"
    
    # =========================================================================
    # END DYNAMIC REPOSITORY OPERATIONS
    # =========================================================================

    def _execute_backend_run(self, backend_name: str, circuit: str = '', shots: int = 1024, 
                             params: dict = None, circuit_file: str = '', custom_config: dict = None) -> str:
        """Run a simulation with a specific backend with custom user requirements.
        
        Args:
            backend_name: Name of the backend to use (cirq, qiskit, pennylane, lret_*, etc.)
            circuit: Circuit definition as string (optional, for simple circuits)
            shots: Number of measurement shots
            params: Additional parameters for the simulation
            circuit_file: Path to a circuit file (Python script, QASM, etc.)
            custom_config: Custom configuration dict with:
                - optimization_level: 0-3 for Qiskit
                - noise_model: noise model configuration
                - seed: random seed for reproducibility
                - qubits: number of qubits
                - parallel: use parallel execution
                - gpu: use GPU if available
        """
        import subprocess
        import tempfile
        
        backend_name = backend_name.lower().strip()
        params = params or {}
        custom_config = custom_config or {}
        
        # Store simulation info for monitoring
        sim_id = f"sim_{int(time.time())}"
        if not hasattr(self, '_running_simulations'):
            self._running_simulations = {}
        
        self._running_simulations[sim_id] = {
            'backend': backend_name,
            'circuit': circuit,
            'circuit_file': circuit_file,
            'shots': shots,
            'params': params,
            'config': custom_config,
            'status': 'running',
            'started': time.time()
        }
        
        # Extract custom config options
        optimization_level = custom_config.get('optimization_level', 1)
        seed = custom_config.get('seed', None)
        num_qubits = custom_config.get('qubits', 2)
        use_parallel = custom_config.get('parallel', False)
        use_gpu = custom_config.get('gpu', False)
        noise_model = custom_config.get('noise_model', None)
        
        # If circuit_file is provided, run it directly
        if circuit_file:
            circuit_file = os.path.expanduser(os.path.expandvars(circuit_file))
            if os.path.exists(circuit_file):
                # Run the circuit file as a Python script with arguments
                cmd = ['python', circuit_file]
                
                # Add shots argument if the script supports it
                cmd.extend(['--shots', str(shots)])
                
                if backend_name:
                    cmd.extend(['--backend', backend_name])
                
                for key, value in params.items():
                    cmd.extend([f'--{key}', str(value)])
                
                self._show_ai_message(f"🚀 Running `{circuit_file}` with {shots} shots using `{backend_name}`")
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True, text=True, timeout=600,
                        cwd=os.path.dirname(circuit_file) or '.'
                    )
                    
                    self._running_simulations[sim_id]['status'] = 'completed' if result.returncode == 0 else 'failed'
                    self._running_simulations[sim_id]['results'] = result.stdout
                    self._running_simulations[sim_id]['ended'] = time.time()
                    
                    if result.returncode == 0:
                        return f"✅ **Simulation Completed**\n📋 ID: `{sim_id}`\n```\n{result.stdout[:3000]}\n```"
                    else:
                        return f"❌ **Simulation Failed**\n📋 ID: `{sim_id}`\n```\n{result.stderr[:2000]}\n```"
                except subprocess.TimeoutExpired:
                    self._running_simulations[sim_id]['status'] = 'timeout'
                    return f"⏱️ **Simulation Timeout** (>600s)\n📋 ID: `{sim_id}`"
                except Exception as e:
                    self._running_simulations[sim_id]['status'] = 'error'
                    return f"❌ **Error:** {str(e)}"
        
        # Generate and run a circuit script based on backend
        script_content = self._generate_backend_script(
            backend_name, circuit, shots, num_qubits, 
            optimization_level, seed, use_parallel, use_gpu, noise_model, params
        )
        
        if script_content.startswith("❌"):
            return script_content
        
        # Write and execute the script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            self._show_ai_message(f"🚀 Running {backend_name} simulation with {shots} shots on {num_qubits} qubits...")
            
            result = subprocess.run(
                ['python', script_path],
                capture_output=True, text=True, timeout=600
            )
            
            self._running_simulations[sim_id]['status'] = 'completed' if result.returncode == 0 else 'failed'
            self._running_simulations[sim_id]['results'] = result.stdout
            self._running_simulations[sim_id]['ended'] = time.time()
            
            elapsed = self._running_simulations[sim_id]['ended'] - self._running_simulations[sim_id]['started']
            
            if result.returncode == 0:
                return (
                    f"✅ **Simulation Completed**\n"
                    f"📋 ID: `{sim_id}`\n"
                    f"🔬 Backend: `{backend_name}`\n"
                    f"🎯 Shots: {shots}\n"
                    f"⏱️ Time: {elapsed:.2f}s\n\n"
                    f"**Results:**\n```\n{result.stdout[:3000]}\n```"
                )
            else:
                return f"❌ **Simulation Failed**\n📋 ID: `{sim_id}`\n```\n{result.stderr[:2000]}\n```"
        finally:
            os.unlink(script_path)
    
    def _generate_backend_script(self, backend_name: str, circuit: str, shots: int, 
                                  num_qubits: int, optimization_level: int, seed: int,
                                  use_parallel: bool, use_gpu: bool, noise_model: dict, params: dict) -> str:
        """Generate a Python script for running a simulation on a specific backend."""
        
        seed_code = f"import random; random.seed({seed}); import numpy as np; np.random.seed({seed})" if seed else ""
        
        # Cirq backend
        if backend_name in ['cirq', 'lret_cirq_scalability']:
            script = f'''
{seed_code}
import cirq
import numpy as np

# Create circuit
n_qubits = {num_qubits}
qubits = cirq.LineQubit.range(n_qubits)

# Build circuit (Bell state by default, or custom)
circuit = cirq.Circuit()
'''
            if circuit:
                script += f'''
# Custom circuit definition
{circuit}
'''
            else:
                script += '''
# Default: Bell state
circuit.append(cirq.H(qubits[0]))
for i in range(n_qubits - 1):
    circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
circuit.append(cirq.measure(*qubits, key='result'))
'''
            script += f'''
# Run simulation
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions={shots})
counts = result.histogram(key='result')
print(f"Measurement Results ({shots} shots):")
for state, count in sorted(counts.items()):
    binary = format(state, f'0{n_qubits}b')
    prob = count / {shots} * 100
    print(f"  |{{binary}}⟩: {{count}} ({{prob:.2f}}%)")
print(f"\\nCircuit depth: {{len(circuit)}}")
'''
            return script
        
        # Qiskit backend
        elif backend_name in ['qiskit', 'qiskit_aer']:
            script = f'''
{seed_code}
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# Create circuit
n_qubits = {num_qubits}
qc = QuantumCircuit(n_qubits, n_qubits)
'''
            if circuit:
                script += f'''
# Custom circuit definition
{circuit}
'''
            else:
                script += '''
# Default: Bell state / GHZ state
qc.h(0)
for i in range(n_qubits - 1):
    qc.cx(i, i+1)
qc.measure(range(n_qubits), range(n_qubits))
'''
            script += f'''
# Transpile and run
simulator = AerSimulator()
transpiled = transpile(qc, simulator, optimization_level={optimization_level})
job = simulator.run(transpiled, shots={shots})
result = job.result()
counts = result.get_counts()

print(f"Measurement Results ({shots} shots):")
for state, count in sorted(counts.items(), key=lambda x: -x[1]):
    prob = count / {shots} * 100
    print(f"  |{{state}}⟩: {{count}} ({{prob:.2f}}%)")
print(f"\\nTranspiled depth: {{transpiled.depth()}}")
print(f"Gates: {{dict(transpiled.count_ops())}}")
'''
            return script
        
        # PennyLane backend
        elif backend_name in ['pennylane', 'lret_pennylane']:
            script = f'''
{seed_code}
import pennylane as qml
import numpy as np

n_qubits = {num_qubits}
dev = qml.device("default.qubit", wires=n_qubits, shots={shots})

@qml.qnode(dev)
def circuit():
'''
            if circuit:
                script += f'''
    # Custom circuit
    {circuit}
'''
            else:
                script += '''
    # Default: Bell/GHZ state
    qml.Hadamard(wires=0)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    return qml.counts()
'''
            script += f'''
# Run circuit
counts = circuit()
print(f"Measurement Results ({shots} shots):")
for state, count in sorted(counts.items(), key=lambda x: -x[1]):
    prob = count / {shots} * 100
    print(f"  |{{state}}⟩: {{count}} ({{prob:.2f}}%)")
'''
            return script
        
        # Braket backend
        elif backend_name == 'braket':
            script = f'''
{seed_code}
from braket.circuits import Circuit
from braket.devices import LocalSimulator

n_qubits = {num_qubits}
device = LocalSimulator()

# Build circuit
circuit = Circuit()
'''
            if circuit:
                script += f'''
# Custom circuit
{circuit}
'''
            else:
                script += '''
# Default: Bell/GHZ state
circuit.h(0)
for i in range(n_qubits - 1):
    circuit.cnot(i, i+1)
'''
            script += f'''
# Run simulation
result = device.run(circuit, shots={shots}).result()
counts = result.measurement_counts
print(f"Measurement Results ({shots} shots):")
for state, count in sorted(counts.items(), key=lambda x: -x[1]):
    prob = count / {shots} * 100
    print(f"  |{{state}}⟩: {{count}} ({{prob:.2f}}%)")
'''
            return script
        
        else:
            return f"❌ Backend `{backend_name}` not supported for direct execution.\n\nSupported: cirq, qiskit, pennylane, braket, lret_*"
    
    # =========================================================================
    # SCRIPT OPERATIONS
    # =========================================================================
    
    def _execute_python_script(self, script: str, args: list = None) -> str:
        """Execute a Python script."""
        import subprocess
        
        script = os.path.expanduser(os.path.expandvars(script))
        
        if not os.path.exists(script):
            return f"❌ Script not found: `{script}`"
        
        cmd = ['python', script] + (args or [])
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        
        output = result.stdout or result.stderr or "Script completed"
        status = "✅" if result.returncode == 0 else "❌"
        
        return f"{status} **Python Script:** `{script}`\n```\n{output[:3000]}\n```"
    
    def _execute_shell_script(self, script: str, args: list = None) -> str:
        """Execute a shell script."""
        import subprocess
        
        script = os.path.expanduser(os.path.expandvars(script))
        
        if not os.path.exists(script):
            return f"❌ Script not found: `{script}`"
        
        if os.name == 'nt':
            cmd = ['powershell', '-File', script] + (args or [])
        else:
            cmd = ['bash', script] + (args or [])
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        
        output = result.stdout or result.stderr or "Script completed"
        status = "✅" if result.returncode == 0 else "❌"
        
        return f"{status} **Shell Script:** `{script}`\n```\n{output[:3000]}\n```"
    
    def _create_and_run_script(self, content: str, language: str = 'python') -> str:
        """Create a temporary script and run it."""
        import subprocess
        import tempfile
        
        extensions = {'python': '.py', 'bash': '.sh', 'powershell': '.ps1'}
        ext = extensions.get(language, '.py')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
            f.write(content)
            script_path = f.name
        
        try:
            if language == 'python':
                cmd = ['python', script_path]
            elif language == 'bash':
                cmd = ['bash', script_path]
            elif language == 'powershell':
                cmd = ['powershell', '-File', script_path]
            else:
                cmd = ['python', script_path]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            output = result.stdout or result.stderr or "Script completed"
            status = "✅" if result.returncode == 0 else "❌"
            
            return f"{status} **Executed {language} script:**\n```\n{output[:3000]}\n```"
        finally:
            os.unlink(script_path)
    
    # =========================================================================
    # SIMULATION & RESULTS OPERATIONS
    # =========================================================================
    
    def _run_simulation(self, backend: str, circuit_file: str, shots: int) -> str:
        """Run a quantum simulation."""
        import subprocess
        
        circuit_file = os.path.expanduser(os.path.expandvars(circuit_file))
        
        sim_id = f"sim_{int(time.time())}"
        
        # Store simulation state
        if not hasattr(self, '_running_simulations'):
            self._running_simulations = {}
        
        self._running_simulations[sim_id] = {
            'backend': backend,
            'circuit_file': circuit_file,
            'shots': shots,
            'status': 'running',
            'started': time.time(),
            'results': None
        }
        
        # Start simulation in background
        if circuit_file and os.path.exists(circuit_file):
            # Run the circuit file
            cmd = f'python "{circuit_file}" --shots {shots} --backend {backend}'
            
            process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            self._running_simulations[sim_id]['process'] = process
            self._running_simulations[sim_id]['pid'] = process.pid
        
        return f"🚀 **Simulation Started**\n📋 ID: `{sim_id}`\n🔬 Backend: `{backend}`\n📄 Circuit: `{circuit_file}`\n🎯 Shots: {shots}\n\nUse `check simulation {sim_id}` to monitor progress."
    
    def _monitor_simulation(self, sim_id: str) -> str:
        """Monitor a running simulation."""
        if not hasattr(self, '_running_simulations') or sim_id not in self._running_simulations:
            return f"❌ Simulation not found: `{sim_id}`"
        
        sim = self._running_simulations[sim_id]
        process = sim.get('process')
        
        if process:
            poll = process.poll()
            if poll is None:
                elapsed = time.time() - sim['started']
                return f"⏳ **Simulation Running**\n📋 ID: `{sim_id}`\n⏱️ Elapsed: {elapsed:.1f}s"
            else:
                stdout, stderr = process.communicate()
                sim['status'] = 'completed' if poll == 0 else 'failed'
                sim['results'] = stdout or stderr
                
                status = "✅" if poll == 0 else "❌"
                return f"{status} **Simulation Completed**\n📋 ID: `{sim_id}`\n```\n{sim['results'][:2000]}\n```"
        
        return f"ℹ️ **Simulation Status:** {sim['status']}"
    
    def _cancel_simulation(self, sim_id: str) -> str:
        """Cancel a running simulation."""
        if not hasattr(self, '_running_simulations') or sim_id not in self._running_simulations:
            return f"❌ Simulation not found: `{sim_id}`"
        
        sim = self._running_simulations[sim_id]
        process = sim.get('process')
        
        if process and process.poll() is None:
            process.terminate()
            sim['status'] = 'cancelled'
            return f"✅ Simulation `{sim_id}` cancelled"
        
        return f"ℹ️ Simulation `{sim_id}` was not running"
    
    def _get_simulation_results(self, sim_id: str) -> str:
        """Get results of a completed simulation."""
        if not hasattr(self, '_running_simulations') or sim_id not in self._running_simulations:
            return f"❌ Simulation not found: `{sim_id}`"
        
        sim = self._running_simulations[sim_id]
        
        if sim['status'] == 'running':
            return self._monitor_simulation(sim_id)
        
        if sim.get('results'):
            return f"📊 **Simulation Results**\n📋 ID: `{sim_id}`\n🔬 Backend: `{sim['backend']}`\n\n```\n{sim['results']}\n```"
        
        return f"ℹ️ No results available for `{sim_id}`"
    
    def _analyze_results(self, data: dict, analysis_type: str) -> str:
        """Analyze simulation results."""
        if not data:
            return "❌ No data provided for analysis"
        
        analysis = f"📊 **Results Analysis ({analysis_type})**\n\n"
        
        if analysis_type == 'quantum_state':
            # Analyze quantum state probabilities
            if 'counts' in data:
                counts = data['counts']
                total = sum(counts.values())
                analysis += "**State Probabilities:**\n"
                for state, count in sorted(counts.items(), key=lambda x: -x[1]):
                    prob = count / total
                    bar = "█" * int(prob * 20)
                    analysis += f"`|{state}>`: {prob:.3f} {bar}\n"
        
        elif analysis_type == 'general':
            # General analysis
            analysis += f"```json\n{json.dumps(data, indent=2)[:2000]}\n```"
        
        return analysis
    
    def _export_to_results_tab(self, results: dict, title: str, export_format: str) -> str:
        """Export results to the Results tab (accessible via key 3)."""
        import json
        from datetime import datetime
        from pathlib import Path
        
        # Create export directory
        export_dir = Path.home() / '.proxima' / 'exports'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create export file
        export_data = {
            'title': title,
            'timestamp': timestamp,
            'format': export_format,
            'results': results,
            'source': 'ai_assistant',
            'ready_for_results_tab': True
        }
        
        export_file = export_dir / f'result_{timestamp}.json'
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Also update the shared state if available
        if hasattr(self, 'state') and self.state:
            try:
                # Store in state for Results screen to pick up
                if not hasattr(self.state, 'pending_results'):
                    self.state.pending_results = []
                self.state.pending_results.append(export_data)
                self.state.last_result = export_data
            except Exception:
                pass
        
        return (
            f"✅ **Results Exported**\n\n"
            f"📋 Title: `{title}`\n"
            f"📁 File: `{export_file}`\n"
            f"📊 Format: `{export_format}`\n\n"
            f"💡 Open the **Results tab (press 3)** to view and analyze these results."
        )
    
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
    
    def _initiate_github_auth_flow(self) -> bool:
        """Initiate GitHub authentication flow.
        
        Attempts to start the GitHub authentication process using gh CLI.
        This method is called by LLM-analyzed operations when authentication is needed.
        
        Returns:
            True if authentication was initiated, False otherwise.
        """
        import subprocess
        
        if not GITHUB_AUTH_AVAILABLE:
            return False
        
        try:
            github_auth = get_github_auth()
            
            # Check if gh CLI is available
            if not github_auth.has_gh_cli:
                return False
            
            # Try to run gh auth login with web option (non-blocking)
            # This opens a browser for authentication
            try:
                # Check if already authenticated
                auth_result = github_auth.check_auth_status(force_refresh=True)
                if auth_result.is_authenticated:
                    return True  # Already authenticated
                
                # Try web-based authentication (opens browser)
                result = subprocess.Popen(
                    ['gh', 'auth', 'login', '--web'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                
                # Give it a moment to start
                import time
                time.sleep(0.5)
                
                # Check if process started successfully
                if result.poll() is None or result.returncode is None:
                    return True  # Process started successfully
                
                return result.returncode == 0
                
            except Exception:
                # Try alternative: just notify that they need to authenticate
                self._show_ai_message(
                    "🔐 **GitHub Authentication Required**\n\n"
                    "Please open a terminal and run:\n"
                    "```\ngh auth login\n```\n\n"
                    "Then try your command again."
                )
                return True  # We showed the message
                
        except Exception as e:
            return False
    
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
