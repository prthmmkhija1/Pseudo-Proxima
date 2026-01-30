"""Multi-backend execution dialog for running multiple backends simultaneously for comparison."""

from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal, ScrollableContainer, Container
from textual.widgets import Static, Button, Input, Checkbox, Select, ProgressBar, RichLog
from textual.timer import Timer
from rich.text import Text
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import time

try:
    from proxima.tui.styles.theme import get_theme
except ImportError:
    def get_theme():
        class DefaultTheme:
            primary = "blue"
            accent = "cyan"
            success = "green"
            warning = "yellow"
            error = "red"
            fg_base = "white"
            fg_muted = "gray"
            fg_subtle = "dim"
        return DefaultTheme()

try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMRouter = None
    LLMRequest = None


class AnalysisExportNameDialog(ModalScreen[str]):
    """Dialog for entering custom export filename for AI analysis."""
    
    DEFAULT_CSS = """
    AnalysisExportNameDialog {
        align: center middle;
    }
    
    AnalysisExportNameDialog > Vertical {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    
    AnalysisExportNameDialog .dialog-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    AnalysisExportNameDialog Input {
        margin-bottom: 1;
    }
    
    AnalysisExportNameDialog Horizontal {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    
    AnalysisExportNameDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    """
    
    def __init__(self, default_name: str = "") -> None:
        super().__init__()
        self._default_name = default_name
    
    def compose(self):
        with Vertical():
            yield Static("📤 Export Analysis", classes="dialog-title")
            yield Static("Enter a name for the exported analysis:")
            yield Input(value=self._default_name, id="export-name-input", placeholder="e.g., benchmark_comparison_analysis")
            with Horizontal():
                yield Button("✓ Export", variant="primary", id="export-btn")
                yield Button("✕ Cancel", variant="default", id="cancel-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-btn":
            name_input = self.query_one("#export-name-input", Input)
            self.dismiss(name_input.value.strip())
        else:
            self.dismiss("")
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input."""
        self.dismiss(event.value.strip())


class AnalysisImportSelectDialog(ModalScreen[str]):
    """Dialog for selecting which analysis to import."""
    
    DEFAULT_CSS = """
    AnalysisImportSelectDialog {
        align: center middle;
    }
    
    AnalysisImportSelectDialog > Vertical {
        width: 80;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    
    AnalysisImportSelectDialog .dialog-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    AnalysisImportSelectDialog Select {
        margin-bottom: 1;
        width: 100%;
    }
    
    AnalysisImportSelectDialog Horizontal {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    
    AnalysisImportSelectDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    
    AnalysisImportSelectDialog .hint-text {
        color: $text-muted;
        margin-bottom: 1;
    }
    """
    
    def __init__(self, chat_files: List[tuple]) -> None:
        """Initialize with list of (display_name, file_path) tuples."""
        super().__init__()
        self._chat_files = chat_files
    
    def compose(self):
        with Vertical():
            yield Static("📥 Import Analysis", classes="dialog-title")
            yield Static("Select a previously exported analysis:", classes="hint-text")
            options = [(name, path) for name, path in self._chat_files]
            yield Select(options, id="chat-select", prompt="Choose an analysis...")
            with Horizontal():
                yield Button("✓ Import", variant="primary", id="import-btn")
                yield Button("✕ Cancel", variant="default", id="cancel-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "import-btn":
            select = self.query_one("#chat-select", Select)
            if select.value and select.value != Select.BLANK:
                self.dismiss(str(select.value))
            else:
                self.dismiss("")
        else:
            self.dismiss("")


@dataclass
class BackendResult:
    """Results from a single backend execution."""
    backend_name: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    execution_time_ms: float = 0
    memory_mb: float = 0
    fidelity: float = 0
    shots_completed: int = 0
    error: Optional[str] = None


class MultiBackendRunDialog(ModalScreen):
    """Dialog for running multiple backends with the same parameters for comparison."""

    DEFAULT_CSS = """
    MultiBackendRunDialog { align: center middle; }
    
    MultiBackendRunDialog > .dialog-container {
        padding: 1 2;
        border: thick $accent;
        background: $surface;
        width: 100%;
        height: 100%;
        max-width: 100%;
        max-height: 100%;
    }
    
    MultiBackendRunDialog .dialog-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
        height: 2;
    }
    
    MultiBackendRunDialog .main-layout {
        layout: horizontal;
        height: 1fr;
    }
    
    MultiBackendRunDialog .config-panel {
        width: 40%;
        height: 1fr;
        border-right: solid $primary-darken-2;
        padding-right: 1;
    }
    
    MultiBackendRunDialog .results-panel {
        width: 35%;
        height: 1fr;
        padding: 0 1;
    }
    
    MultiBackendRunDialog .ai-panel {
        width: 25%;
        height: 1fr;
        padding-left: 1;
        border-left: solid $primary-darken-2;
    }
    
    MultiBackendRunDialog .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        height: 2;
    }
    
    MultiBackendRunDialog .section-content {
        height: 1fr;
        padding: 1;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
        overflow-y: auto;
    }
    
    MultiBackendRunDialog .param-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    MultiBackendRunDialog .param-label {
        width: 12;
        height: 3;
        content-align: left middle;
    }
    
    MultiBackendRunDialog .param-input {
        width: 1fr;
    }
    
    MultiBackendRunDialog .backend-check {
        height: 3;
        margin-bottom: 0;
    }
    
    MultiBackendRunDialog .result-item {
        height: auto;
        min-height: 4;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary-darken-3;
        background: $surface;
    }
    
    MultiBackendRunDialog .result-header {
        layout: horizontal;
        height: 2;
    }
    
    MultiBackendRunDialog .result-name {
        width: 1fr;
        text-style: bold;
    }
    
    MultiBackendRunDialog .result-status {
        width: auto;
    }
    
    MultiBackendRunDialog .result-metrics {
        color: $text-muted;
        height: auto;
    }
    
    MultiBackendRunDialog .ai-header {
        height: 3;
        layout: horizontal;
        align: left middle;
        padding: 0 1;
        background: $primary-darken-2;
    }
    
    MultiBackendRunDialog .ai-title {
        width: 1fr;
        text-style: bold;
        color: $accent;
    }
    
    MultiBackendRunDialog .ai-controls-row {
        height: 3;
        layout: horizontal;
        padding: 0 1;
        margin-top: 1;
        align: right middle;
    }
    
    MultiBackendRunDialog .chat-ctrl-btn {
        min-width: 10;
        width: auto;
        height: 3;
        margin-left: 1;
    }
    
    MultiBackendRunDialog .ai-chat-log {
        height: 1fr;
        background: $surface-darken-2;
        padding: 1;
        margin-bottom: 1;
    }
    
    MultiBackendRunDialog .ai-input-row {
        height: 3;
        layout: horizontal;
    }
    
    MultiBackendRunDialog .ai-input {
        width: 1fr;
        margin-right: 1;
    }
    
    MultiBackendRunDialog .ai-stats {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        margin-bottom: 1;
        border: solid $primary-darken-3;
    }
    
    MultiBackendRunDialog .footer {
        height: 4;
        layout: horizontal;
        margin-top: 1;
        padding: 1 2;
        border-top: solid $primary-darken-3;
        background: $surface-darken-1;
        align: left middle;
    }
    
    MultiBackendRunDialog .footer Button {
        margin-right: 2;
        min-width: 12;
        height: 3;
    }
    
    MultiBackendRunDialog ProgressBar {
        margin: 1 0;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+r", "run_all", "Run All"),
        ("ctrl+s", "stop_all", "Stop"),
    ]

    def __init__(self, on_complete: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self._on_complete = on_complete
        self._backends: List[Dict[str, Any]] = []
        self._selected_backends: set = set()
        self._results: Dict[str, BackendResult] = {}
        self._is_running = False
        self._run_timer: Optional[Timer] = None
        self._llm_router = None
        self._ai_conversation = []
        self._ai_stats = {
            'requests': 0,
            'tokens': 0,
            'thinking_time_ms': 0,
        }
        
        # Initialize LLM
        if LLM_AVAILABLE:
            try:
                # Auto-consent function for TUI mode - user has already configured provider
                def auto_consent(prompt: str) -> bool:
                    return True  # Auto-approve in TUI since user explicitly configured
                
                self._llm_router = LLMRouter(consent_prompt=auto_consent)
                self._init_llm_with_settings()
            except Exception:
                pass
    
    def _init_llm_with_settings(self) -> None:
        """Load LLM settings from saved config."""
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if not config_path.exists():
                return
            
            with open(config_path, 'r') as f:
                settings = json.load(f)
            
            llm = settings.get('llm', {})
            self._llm_provider = llm.get('mode', 'none')
            
            if self._llm_provider == 'none':
                return
            
            model_key_map = {
                'local': 'local_model',
                'openai': 'openai_model',
                'anthropic': 'anthropic_model',
                'google': 'google_model',
                'xai': 'xai_model',
                'deepseek': 'deepseek_model',
                'mistral': 'mistral_model',
                'groq': 'groq_model',
                'together': 'together_model',
                'openrouter': 'openrouter_model',
                'cohere': 'cohere_model',
                'perplexity': 'perplexity_model',
            }
            self._llm_model = llm.get(model_key_map.get(self._llm_provider, ''), '')
            
            # Get and register API key
            api_key_map = {
                'openai': 'openai_key',
                'anthropic': 'anthropic_key',
                'google': 'google_key',
                'xai': 'xai_key',
                'deepseek': 'deepseek_key',
                'mistral': 'mistral_key',
                'groq': 'groq_key',
                'together': 'together_key',
                'openrouter': 'openrouter_key',
                'cohere': 'cohere_key',
                'perplexity': 'perplexity_key',
            }
            
            api_key_field = api_key_map.get(self._llm_provider)
            if api_key_field:
                api_key = llm.get(api_key_field, '')
                if api_key and self._llm_router:
                    self._llm_router.api_keys.store_key(self._llm_provider, api_key)
                    
        except Exception:
            self._llm_provider = None
            self._llm_model = None

    def compose(self):
        theme = get_theme()
        
        with Vertical(classes="dialog-container"):
            yield Static("🔄 Multi-Backend Comparison Run", classes="dialog-title")
            
            with Horizontal(classes="main-layout"):
                # Left: Configuration panel
                with Vertical(classes="config-panel"):
                    yield Static("⚙️ Run Parameters", classes="section-title")
                    
                    with ScrollableContainer(classes="section-content"):
                        # Common parameters
                        with Vertical():
                            yield Static("📊 Common Parameters", classes="section-title")
                            
                            with Horizontal(classes="param-row"):
                                yield Static("Qubits:", classes="param-label")
                                yield Input(value="10", id="input-qubits", classes="param-input")
                            
                            with Horizontal(classes="param-row"):
                                yield Static("Shots:", classes="param-label")
                                yield Input(value="1000", id="input-shots", classes="param-input")
                            
                            with Horizontal(classes="param-row"):
                                yield Static("Circuit:", classes="param-label")
                                yield Select(
                                    [(name, name) for name in [
                                        "Bell State", "GHZ State", "Random Circuit",
                                        "QFT", "Grover's Search", "VQE Ansatz"
                                    ]],
                                    id="select-circuit",
                                    value="Bell State",
                                    classes="param-input"
                                )
                            
                            yield Static("─" * 30, classes="section-title")
                            yield Static("📦 Select Backends", classes="section-title")
                            yield Static("Check backends to include in comparison:", 
                                       id="backend-hint", classes="result-metrics")
                            
                            # Backend checkboxes
                            for backend in self._get_available_backends():
                                yield Checkbox(
                                    f"{backend['name']} ({backend['type']})",
                                    id=f"check-{backend['id']}",
                                    classes="backend-check",
                                    value=backend.get('default', False)
                                )
                
                # Middle: Results panel
                with Vertical(classes="results-panel"):
                    yield Static("📈 Execution Results", classes="section-title")
                    
                    with ScrollableContainer(classes="section-content", id="results-container"):
                        yield Static("Select backends and click 'Run All' to start comparison.",
                                   id="results-placeholder", classes="result-metrics")
                    
                    # Overall progress
                    yield ProgressBar(total=100, id="overall-progress")
                    yield Static("Ready", id="progress-label", classes="result-metrics")
                
                # Right: AI Assistant panel
                with Vertical(classes="ai-panel"):
                    with Horizontal(classes="ai-header"):
                        yield Static("🤖 AI Analysis", classes="ai-title")
                    
                    with Container(classes="ai-stats", id="ai-stats"):
                        yield Static("Model: —", id="ai-stat-model")
                        yield Static("Requests: 0", id="ai-stat-requests")
                    
                    yield RichLog(
                        auto_scroll=True,
                        classes="ai-chat-log",
                        id="ai-chat-log",
                    )
                    
                    with Horizontal(classes="ai-input-row"):
                        yield Input(
                            placeholder="Ask AI about results...",
                            id="ai-input",
                            classes="ai-input"
                        )
                        yield Button("Ask", id="btn-ai-ask", variant="primary")
                    
                    # Chat control buttons at bottom
                    with Horizontal(classes="ai-controls-row"):
                        yield Button("📝 New", id="btn-new-chat", variant="default", classes="chat-ctrl-btn")
                        yield Button("📤 Export", id="btn-export-chat", variant="default", classes="chat-ctrl-btn")
                        yield Button("📥 Import", id="btn-import-chat", variant="default", classes="chat-ctrl-btn")
                        yield Button("🗑 Clear", id="btn-clear-chat", variant="warning", classes="chat-ctrl-btn")
            
            # Footer
            with Horizontal(classes="footer"):
                yield Button("▶ Run All", id="btn-run", variant="success")
                yield Button("⏹ Stop", id="btn-stop", variant="error", disabled=True)
                yield Button("📊 Export Results", id="btn-export", variant="default")
                yield Button("🔄 Reset", id="btn-reset", variant="warning")
                yield Button("✕ Close", id="btn-close", variant="default")

    def _get_available_backends(self) -> List[Dict[str, Any]]:
        """Get list of available backends."""
        # Default backends
        backends = [
            {"id": "simulator", "name": "Simulator", "type": "CPU", "default": True},
            {"id": "lret_v1", "name": "LRET Phase 1", "type": "CPU", "default": False},
            {"id": "lret_v7", "name": "LRET Phase 7", "type": "GPU", "default": True},
            {"id": "cirq", "name": "Cirq", "type": "CPU", "default": False},
            {"id": "qiskit", "name": "Qiskit Aer", "type": "CPU", "default": False},
            {"id": "cuquantum", "name": "cuQuantum", "type": "GPU", "default": False},
        ]
        
        # Load custom backends
        try:
            custom_path = Path.home() / ".proxima" / "custom_backends.json"
            if custom_path.exists():
                with open(custom_path, 'r') as f:
                    custom_backends = json.load(f)
                for cb in custom_backends:
                    backends.append({
                        "id": cb.get('id', cb.get('name', '').lower()),
                        "name": cb.get('display_name', cb.get('name', 'Custom')),
                        "type": cb.get('type', 'Custom'),
                        "default": False
                    })
        except Exception:
            pass
        
        self._backends = backends
        return backends

    def on_mount(self):
        """Initialize on mount and restore chat state."""
        # Restore chat from TUIState
        self._restore_chat_state()
        
        # Only show welcome if no previous chat
        if not self._ai_conversation:
            self._update_ai_welcome()
        
        self._update_ai_stats()

    def on_unmount(self):
        """Save chat state before closing."""
        self._save_chat_state()

    def _restore_chat_state(self) -> None:
        """Restore chat messages from TUIState."""
        try:
            from ...state.tui_state import TUIState
            
            # Get TUIState from app
            if hasattr(self.app, 'state') and isinstance(self.app.state, TUIState):
                state = self.app.state
                
                if state.comparison_chat_messages:
                    self._ai_conversation = state.comparison_chat_messages.copy()
                    self._ai_stats = state.comparison_chat_stats.copy()
                    
                    # Restore chat display
                    self._restore_chat_display()
        except Exception:
            pass

    def _save_chat_state(self) -> None:
        """Save chat messages to TUIState."""
        try:
            from ...state.tui_state import TUIState
            
            # Get TUIState from app
            if hasattr(self.app, 'state') and isinstance(self.app.state, TUIState):
                state = self.app.state
                state.comparison_chat_messages = self._ai_conversation.copy()
                state.comparison_chat_stats = self._ai_stats.copy()
        except Exception:
            pass

    def _restore_chat_display(self) -> None:
        """Restore chat messages to the display."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#ai-chat-log", RichLog)
            chat_log.clear()
            
            for msg in self._ai_conversation:
                text = Text()
                if msg.get('role') == 'user':
                    text.append("👤 You: ", style=f"bold {theme.primary}")
                    text.append(msg.get('content', '') + "\n\n", style=theme.fg_base)
                else:
                    text.append("🤖 AI: ", style=f"bold {theme.accent}")
                    text.append(msg.get('content', '') + "\n\n", style=theme.fg_base)
                chat_log.write(text)
            
            # Update stats display
            self._update_ai_stats()
        except Exception:
            pass

    def _update_ai_welcome(self):
        """Show welcome message in AI chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            welcome = Text()
            welcome.append("🤖 AI Comparison Assistant\n", style=f"bold {theme.accent}")
            welcome.append("─" * 25 + "\n", style=theme.fg_subtle)
            welcome.append("I can help you:\n", style=theme.fg_base)
            welcome.append("• Analyze comparison results\n", style=theme.fg_muted)
            welcome.append("• Recommend optimal backends\n", style=theme.fg_muted)
            welcome.append("• Explain performance metrics\n\n", style=theme.fg_muted)
            welcome.append("Run a comparison first, then ask me questions!", style=theme.fg_subtle)
            
            chat_log.write(welcome)
        except Exception:
            pass

    def _update_ai_stats(self):
        """Update AI stats display."""
        try:
            model = getattr(self, '_llm_model', '—') or '—'
            self.query_one("#ai-stat-model", Static).update(f"Model: {model}")
            self.query_one("#ai-stat-requests", Static).update(
                f"Requests: {self._ai_stats['requests']}"
            )
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id
        
        if btn_id == "btn-run":
            self._start_comparison()
        elif btn_id == "btn-stop":
            self._stop_comparison()
        elif btn_id == "btn-export":
            self._export_results()
        elif btn_id == "btn-reset":
            self._reset_comparison()
        elif btn_id == "btn-close":
            self.dismiss(self._results)
        elif btn_id == "btn-ai-ask":
            self._send_ai_message()
        elif btn_id == "btn-new-chat":
            self._new_chat()
        elif btn_id == "btn-clear-chat":
            self._clear_chat()
        elif btn_id == "btn-export-chat":
            self._export_chat()
        elif btn_id == "btn-import-chat":
            self._import_chat()

    def _new_chat(self) -> None:
        """Start a new chat, clearing the current conversation."""
        try:
            # Clear conversation history
            self._ai_conversation = []
            self._ai_stats = {
                'requests': 0,
                'tokens': 0,
                'thinking_time_ms': 0,
            }
            
            # Clear display
            chat_log = self.query_one("#ai-chat-log", RichLog)
            chat_log.clear()
            
            # Show welcome again
            self._update_ai_welcome()
            
            # Update stats display
            self._update_ai_stats()
            
            # Save cleared state
            self._save_chat_state()
            
            self.notify("Started new chat", severity="information")
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def _clear_chat(self) -> None:
        """Clear chat history without resetting stats."""
        try:
            # Clear only the conversation but keep stats
            self._ai_conversation = []
            
            # Clear display
            chat_log = self.query_one("#ai-chat-log", RichLog)
            chat_log.clear()
            
            # Show welcome again
            self._update_ai_welcome()
            
            # Save state
            self._save_chat_state()
            
            self.notify("Chat cleared", severity="information")
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def _export_chat(self) -> None:
        """Export chat history to a file with custom name."""
        try:
            from datetime import datetime
            
            if not self._ai_conversation:
                self.notify("No messages to export", severity="warning")
                return
            
            # Generate default name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"comparison_analysis_{timestamp}"
            
            # Show export name dialog
            def handle_export_name(name: str) -> None:
                if not name:
                    return  # User cancelled
                
                try:
                    # Create export directory
                    export_dir = Path.home() / ".proxima" / "chat_exports"
                    export_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Sanitize filename
                    safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-', ' ')).strip()
                    if not safe_name:
                        safe_name = f"analysis_{timestamp}"
                    
                    filename = f"{safe_name}.json"
                    export_path = export_dir / filename
                    
                    # Prepare export data
                    export_data = {
                        "type": "comparison_chat",
                        "name": name,
                        "timestamp": timestamp,
                        "messages": self._ai_conversation,
                        "stats": self._ai_stats,
                        "results": {
                            bid: {
                                "name": r.backend_name,
                                "status": r.status,
                                "execution_time_ms": r.execution_time_ms,
                                "fidelity": r.fidelity,
                                "memory_mb": r.memory_mb,
                            }
                            for bid, r in self._results.items()
                        } if self._results else {},
                    }
                    
                    # Save to file
                    with open(export_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                    
                    self.notify(f"✅ Analysis exported: {filename}", severity="success")
                except Exception as e:
                    self.notify(f"Export failed: {e}", severity="error")
            
            self.app.push_screen(AnalysisExportNameDialog(default_name), handle_export_name)
            
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")

    def _import_chat(self) -> None:
        """Import chat history from a file with selection dialog."""
        try:
            # Look for chat exports
            export_dir = Path.home() / ".proxima" / "chat_exports"
            
            if not export_dir.exists():
                self.notify("No exported analyses found", severity="warning")
                return
            
            # Find all chat export files
            chat_files = list(export_dir.glob("*.json"))
            
            if not chat_files:
                self.notify("No exported analyses found", severity="warning")
                return
            
            # Build file list with display names
            file_options = []
            for f in sorted(chat_files, key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    with open(f, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                    # Use saved name or filename
                    display_name = data.get('name', f.stem)
                    timestamp = data.get('timestamp', '')
                    msg_count = len(data.get('messages', []))
                    label = f"{display_name} ({msg_count} msgs)"
                    if timestamp:
                        label = f"{display_name} - {timestamp[:8]} ({msg_count} msgs)"
                    file_options.append((label, str(f)))
                except Exception:
                    file_options.append((f.stem, str(f)))
            
            # Show import selection dialog
            def handle_import_select(file_path: str) -> None:
                if not file_path:
                    return  # User cancelled
                
                try:
                    # Load the chat
                    with open(file_path, 'r', encoding='utf-8') as f:
                        import_data = json.load(f)
                    
                    # Restore conversation
                    if 'messages' in import_data:
                        self._ai_conversation = import_data['messages']
                    
                    if 'stats' in import_data:
                        self._ai_stats = import_data['stats']
                    
                    # Restore display
                    self._restore_chat_display()
                    
                    # Update stats
                    self._update_ai_stats()
                    
                    # Save state
                    self._save_chat_state()
                    
                    name = import_data.get('name', Path(file_path).stem)
                    self.notify(f"✅ Analysis imported: {name}", severity="success")
                except Exception as e:
                    self.notify(f"Import failed: {e}", severity="error")
            
            self.app.push_screen(AnalysisImportSelectDialog(file_options), handle_import_select)
            
        except Exception as e:
            self.notify(f"Import failed: {e}", severity="error")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input."""
        if event.input.id == "ai-input":
            self._send_ai_message()

    def _get_selected_backends(self) -> List[str]:
        """Get list of selected backend IDs."""
        selected = []
        for backend in self._backends:
            try:
                checkbox = self.query_one(f"#check-{backend['id']}", Checkbox)
                if checkbox.value:
                    selected.append(backend['id'])
            except Exception:
                pass
        return selected

    def _start_comparison(self):
        """Start running comparison on selected backends."""
        selected = self._get_selected_backends()
        
        if not selected:
            self.notify("Please select at least one backend!", severity="warning")
            return
        
        # Get parameters
        try:
            qubits = int(self.query_one("#input-qubits", Input).value)
            shots = int(self.query_one("#input-shots", Input).value)
            circuit = self.query_one("#select-circuit", Select).value
        except ValueError:
            self.notify("Invalid parameters!", severity="error")
            return
        
        self._is_running = True
        self._selected_backends = set(selected)
        
        # Initialize results
        for backend_id in selected:
            backend_info = next((b for b in self._backends if b['id'] == backend_id), None)
            if backend_info:
                self._results[backend_id] = BackendResult(
                    backend_name=backend_info['name'],
                    status='pending'
                )
        
        # Update UI
        self.query_one("#btn-run", Button).disabled = True
        self.query_one("#btn-stop", Button).disabled = False
        self._update_results_display()
        
        # Start simulation timer
        self._run_timer = self.set_interval(0.5, self._simulate_progress)
        
        self.notify(f"Starting comparison on {len(selected)} backends...", severity="information")

    def _simulate_progress(self):
        """Simulate backend execution progress."""
        import random
        
        all_complete = True
        completed = 0
        total = len(self._selected_backends)
        
        for backend_id, result in self._results.items():
            if result.status == 'pending':
                result.status = 'running'
                result.execution_time_ms = 0
                all_complete = False
            elif result.status == 'running':
                # Simulate progress
                result.execution_time_ms += random.uniform(100, 500)
                
                # Random completion
                if result.execution_time_ms > random.uniform(2000, 5000):
                    result.status = 'completed'
                    result.fidelity = random.uniform(0.95, 0.999)
                    result.memory_mb = random.uniform(50, 500)
                    result.shots_completed = int(self.query_one("#input-shots", Input).value)
                    completed += 1
                else:
                    all_complete = False
            elif result.status == 'completed':
                completed += 1
        
        # Update progress
        progress = (completed / total * 100) if total > 0 else 0
        self.query_one("#overall-progress", ProgressBar).update(progress=progress)
        self.query_one("#progress-label", Static).update(
            f"Progress: {completed}/{total} backends completed"
        )
        
        self._update_results_display()
        
        if all_complete:
            self._complete_comparison()

    def _complete_comparison(self):
        """Handle comparison completion."""
        if self._run_timer:
            self._run_timer.stop()
            self._run_timer = None
        
        self._is_running = False
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True
        
        self.notify("Comparison complete! Ask AI for analysis.", severity="success")
        
        # Auto-generate AI analysis
        self._generate_ai_analysis()

    def _stop_comparison(self):
        """Stop the running comparison."""
        if self._run_timer:
            self._run_timer.stop()
            self._run_timer = None
        
        self._is_running = False
        
        for result in self._results.values():
            if result.status == 'running':
                result.status = 'stopped'
        
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True
        
        self._update_results_display()
        self.notify("Comparison stopped", severity="warning")

    def _update_results_display(self):
        """Update the results display."""
        try:
            theme = get_theme()
            container = self.query_one("#results-container", ScrollableContainer)
            
            # Clear existing content
            for child in list(container.children):
                child.remove()
            
            if not self._results:
                container.mount(Static(
                    "Select backends and click 'Run All' to start comparison.",
                    classes="result-metrics"
                ))
                return
            
            # Sort by execution time (completed first)
            sorted_results = sorted(
                self._results.values(),
                key=lambda r: (r.status != 'completed', r.execution_time_ms)
            )
            
            for result in sorted_results:
                status_icons = {
                    'pending': '⏳',
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌',
                    'stopped': '⏹️'
                }
                
                result_text = Text()
                result_text.append(f"{result.backend_name}\n", style=f"bold {theme.fg_base}")
                result_text.append(
                    f"{status_icons.get(result.status, '?')} {result.status.upper()}\n",
                    style=theme.fg_muted
                )
                
                if result.status == 'completed':
                    result_text.append(
                        f"Time: {result.execution_time_ms:.0f}ms | "
                        f"Fidelity: {result.fidelity:.4f} | "
                        f"Memory: {result.memory_mb:.0f}MB\n",
                        style=theme.fg_subtle
                    )
                elif result.status == 'running':
                    result_text.append(
                        f"Running... {result.execution_time_ms:.0f}ms\n",
                        style=theme.warning
                    )
                
                container.mount(Static(result_text, classes="result-item"))
                
        except Exception:
            pass

    def _reset_comparison(self):
        """Reset the comparison state."""
        self._results = {}
        self.query_one("#overall-progress", ProgressBar).update(progress=0)
        self.query_one("#progress-label", Static).update("Ready")
        self._update_results_display()
        self.notify("Comparison reset", severity="information")

    def _export_results(self):
        """Export comparison results to file."""
        from datetime import datetime
        
        try:
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"multi_backend_comparison_{timestamp}.json"
            
            export_data = {
                "timestamp": timestamp,
                "parameters": {
                    "qubits": self.query_one("#input-qubits", Input).value,
                    "shots": self.query_one("#input-shots", Input).value,
                    "circuit": self.query_one("#select-circuit", Select).value,
                },
                "results": {
                    bid: {
                        "name": r.backend_name,
                        "status": r.status,
                        "execution_time_ms": r.execution_time_ms,
                        "fidelity": r.fidelity,
                        "memory_mb": r.memory_mb,
                        "shots_completed": r.shots_completed,
                    }
                    for bid, r in self._results.items()
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.notify(f"Results exported to {filename}", severity="success")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")

    def _send_ai_message(self):
        """Send message to AI assistant."""
        try:
            theme = get_theme()
            input_widget = self.query_one("#ai-input", Input)
            message = input_widget.value.strip()
            
            if not message:
                return
            
            input_widget.value = ""
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            # Show user message
            user_text = Text()
            user_text.append("👤 You: ", style=f"bold {theme.primary}")
            user_text.append(message + "\n", style=theme.fg_base)
            chat_log.write(user_text)
            
            # Save user message
            self._ai_conversation.append({
                'role': 'user',
                'content': message,
            })
            
            self._ai_stats['requests'] += 1
            self._update_ai_stats()
            
            # Generate response
            self._generate_ai_response(message)
            
        except Exception as e:
            self._show_ai_error(str(e))

    def _generate_ai_analysis(self):
        """Auto-generate AI analysis of results."""
        if not self._results:
            return
        
        # Build analysis
        completed = [r for r in self._results.values() if r.status == 'completed']
        if not completed:
            return
        
        # Find best performers
        fastest = min(completed, key=lambda r: r.execution_time_ms)
        most_accurate = max(completed, key=lambda r: r.fidelity)
        most_efficient = min(completed, key=lambda r: r.memory_mb)
        
        theme = get_theme()
        chat_log = self.query_one("#ai-chat-log", RichLog)
        
        analysis = Text()
        analysis.append("\n📊 Comparison Analysis\n", style=f"bold {theme.accent}")
        analysis.append("─" * 25 + "\n", style=theme.fg_subtle)
        analysis.append(f"🏆 Fastest: {fastest.backend_name} ({fastest.execution_time_ms:.0f}ms)\n",
                       style=f"bold {theme.success}")
        analysis.append(f"🎯 Most Accurate: {most_accurate.backend_name} ({most_accurate.fidelity:.4f})\n",
                       style=f"bold {theme.primary}")
        analysis.append(f"💾 Most Efficient: {most_efficient.backend_name} ({most_efficient.memory_mb:.0f}MB)\n\n",
                       style=f"bold {theme.warning}")
        analysis.append("Ask me for detailed recommendations!\n", style=theme.fg_muted)
        
        chat_log.write(analysis)

    def _generate_ai_response(self, message: str):
        """Generate AI response to user query."""
        theme = get_theme()
        chat_log = self.query_one("#ai-chat-log", RichLog)
        
        # Build context
        results_summary = []
        for bid, r in self._results.items():
            if r.status == 'completed':
                results_summary.append(
                    f"- {r.backend_name}: {r.execution_time_ms:.0f}ms, "
                    f"fidelity={r.fidelity:.4f}, memory={r.memory_mb:.0f}MB"
                )
        
        # Build conversation history for context continuity
        # This allows imported chats to maintain context
        conversation_context = ""
        if self._ai_conversation:
            # Include recent conversation history (last 10 exchanges)
            recent_messages = self._ai_conversation[-20:]  # Last 20 messages (10 exchanges)
            for msg in recent_messages:
                role = "User" if msg.get('role') == 'user' else "Assistant"
                content = msg.get('content', '')
                conversation_context += f"{role}: {content}\n\n"
        
        # Try LLM or simulate
        provider = getattr(self, '_llm_provider', None)
        
        # Comprehensive mapping of TUI provider names to LLM router provider names
        # All major providers are now supported
        provider_name_map = {
            # Local providers
            'local': 'ollama',
            'ollama': 'ollama',
            'lmstudio': 'lmstudio',
            'llamacpp': 'llama_cpp',
            'llama_cpp': 'llama_cpp',
            'localai': 'ollama',
            # Cloud providers - all supported
            'openai': 'openai',
            'anthropic': 'anthropic',
            'google': 'google',  # Google Gemini
            'gemini': 'google',
            'xai': 'xai',        # xAI Grok
            'grok': 'xai',
            'deepseek': 'deepseek',
            'mistral': 'mistral',
            'groq': 'groq',
            'together': 'together',
            'openrouter': 'openrouter',
            'cohere': 'cohere',
            'perplexity': 'perplexity',
            'azure_openai': 'azure_openai',
            'azure': 'azure_openai',
            'huggingface': 'huggingface',
            'fireworks': 'fireworks',
            'replicate': 'replicate',
        }
        # Only a few providers still need simulation (no direct API support)
        unsupported_providers = {'vertex_ai', 'aws_bedrock', 'ai21', 'deepinfra'}
        if provider in unsupported_providers:
            router_provider = None  # Will trigger simulation fallback
        else:
            router_provider = provider_name_map.get(provider, provider) if provider else None
        
        if router_provider and router_provider != 'none' and self._llm_router and LLM_AVAILABLE:
            try:
                # Build full context with conversation history
                context = f"""Comparison Results:
{chr(10).join(results_summary) if results_summary else 'No results yet'}
"""
                if conversation_context:
                    context += f"\nPrevious conversation:\n{conversation_context}\n"
                context += f"Current user query: {message}"

                request = LLMRequest(
                    prompt=context,
                    system_prompt="""You are an AI assistant analyzing quantum backend comparison results.
Provide clear, actionable recommendations based on the metrics.
Consider execution time, fidelity, and memory usage trade-offs.
Be concise and helpful.

You have access to the previous conversation history, so maintain context and refer back to earlier discussions when relevant.""",
                    temperature=0.7,
                    max_tokens=512,
                    provider=router_provider,
                    model=getattr(self, '_llm_model', None),
                )
                
                response = self._llm_router.route(request)
                
                # Check if we got a valid response with actual content
                if response and hasattr(response, 'text') and response.text and response.text.strip():
                    ai_text = Text()
                    ai_text.append("🤖 AI: ", style=f"bold {theme.accent}")
                    ai_text.append(response.text.strip() + "\n\n", style=theme.fg_base)
                    chat_log.write(ai_text)
                    
                    # Save AI response
                    self._ai_conversation.append({
                        'role': 'assistant',
                        'content': response.text.strip(),
                    })
                    self._save_chat_state()
                    return
                # If empty response, fall through to simulation
                
            except PermissionError as e:
                # Consent required - silently fall through to simulation
                pass
            except ConnectionError as e:
                # Local provider not running - fall through to simulation
                pass
            except ValueError as e:
                # Provider not found - silently fall through to simulation
                pass
            except Exception as e:
                # Any other error - fall through to simulated response
                pass
        
        # Simulated response
        msg_lower = message.lower()
        
        if "fastest" in msg_lower or "speed" in msg_lower:
            if self._results:
                completed = [r for r in self._results.values() if r.status == 'completed']
                if completed:
                    fastest = min(completed, key=lambda r: r.execution_time_ms)
                    response = f"The fastest backend is **{fastest.backend_name}** with {fastest.execution_time_ms:.0f}ms execution time. This is ideal for rapid prototyping and iterative development."
                else:
                    response = "No completed results yet. Run the comparison first!"
            else:
                response = "Please run a comparison first to get speed metrics."
        elif "accurate" in msg_lower or "fidelity" in msg_lower:
            if self._results:
                completed = [r for r in self._results.values() if r.status == 'completed']
                if completed:
                    best = max(completed, key=lambda r: r.fidelity)
                    response = f"The most accurate backend is **{best.backend_name}** with {best.fidelity:.4f} fidelity. Higher fidelity means more reliable quantum simulation results."
                else:
                    response = "No completed results yet. Run the comparison first!"
            else:
                response = "Please run a comparison first to get fidelity metrics."
        elif "recommend" in msg_lower or "best" in msg_lower:
            if self._results:
                completed = [r for r in self._results.values() if r.status == 'completed']
                if completed:
                    # Weighted score
                    for r in completed:
                        r._score = (1000 / max(r.execution_time_ms, 1)) * r.fidelity * (1000 / max(r.memory_mb, 1))
                    best = max(completed, key=lambda r: r._score)
                    response = f"Based on overall performance, I recommend **{best.backend_name}**. It offers the best balance of speed ({best.execution_time_ms:.0f}ms), accuracy ({best.fidelity:.4f}), and efficiency ({best.memory_mb:.0f}MB memory)."
                else:
                    response = "No completed results yet. Run the comparison first!"
            else:
                response = "Please run a comparison first to get recommendations."
        else:
            response = "I can help you analyze:\n• Which backend is fastest\n• Which has highest fidelity\n• Overall recommendations\n• Trade-off analysis\n\nWhat would you like to know?"
        
        ai_text = Text()
        ai_text.append("🤖 AI: ", style=f"bold {theme.accent}")
        ai_text.append(response + "\n\n", style=theme.fg_base)
        chat_log.write(ai_text)
        
        # Save simulated AI response
        self._ai_conversation.append({
            'role': 'assistant',
            'content': response,
        })
        self._save_chat_state()

    def _show_ai_error(self, error: str):
        """Show error in AI chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            error_text = Text()
            error_text.append("❌ Error: ", style=f"bold {theme.error}")
            error_text.append(error + "\n", style=theme.fg_muted)
            chat_log.write(error_text)
        except Exception:
            pass

    def action_close(self):
        """Close the dialog."""
        self.dismiss(self._results)

    def action_run_all(self):
        """Keyboard shortcut to run all."""
        self._start_comparison()

    def action_stop_all(self):
        """Keyboard shortcut to stop."""
        self._stop_comparison()
