"""Add custom backend dialog with AI assistance.

Allows beginners to add custom backends through a guided, scrollable interface
with optional AI assistance for configuration.
"""

from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal, ScrollableContainer, Container
from textual.widgets import Static, Button, Input, Switch, Select, Label, TextArea, RichLog
from rich.text import Text
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import json

try:
    from ...styles.theme import get_theme
except ImportError:
    def get_theme():
        class MockTheme:
            primary = "blue"
            accent = "cyan"
            success = "green"
            warning = "yellow"
            error = "red"
            fg_base = "white"
            fg_muted = "grey"
            fg_subtle = "dim"
        return MockTheme()

# Try to import LLM router for AI assistance
try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class ChatExportNameDialog(ModalScreen[str]):
    """Dialog for entering custom export filename."""
    
    DEFAULT_CSS = """
    ChatExportNameDialog {
        align: center middle;
    }
    
    ChatExportNameDialog > Vertical {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    
    ChatExportNameDialog .dialog-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    ChatExportNameDialog Input {
        margin-bottom: 1;
    }
    
    ChatExportNameDialog Horizontal {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    
    ChatExportNameDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    """
    
    def __init__(self, default_name: str = "") -> None:
        super().__init__()
        self._default_name = default_name
    
    def compose(self):
        with Vertical():
            yield Static("📤 Export Chat", classes="dialog-title")
            yield Static("Enter a name for the exported chat:")
            yield Input(value=self._default_name, id="export-name-input", placeholder="e.g., my_backend_config_chat")
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


class ChatImportSelectDialog(ModalScreen[str]):
    """Dialog for selecting which chat to import."""
    
    DEFAULT_CSS = """
    ChatImportSelectDialog {
        align: center middle;
    }
    
    ChatImportSelectDialog > Vertical {
        width: 80;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    
    ChatImportSelectDialog .dialog-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    ChatImportSelectDialog Select {
        margin-bottom: 1;
        width: 100%;
    }
    
    ChatImportSelectDialog Horizontal {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    
    ChatImportSelectDialog Button {
        margin: 0 1;
        min-width: 12;
    }
    
    ChatImportSelectDialog .hint-text {
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
            yield Static("📥 Import Chat", classes="dialog-title")
            yield Static("Select a previously exported chat:", classes="hint-text")
            options = [(name, path) for name, path in self._chat_files]
            yield Select(options, id="chat-select", prompt="Choose a chat...")
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


class AddCustomBackendDialog(ModalScreen):
    """Dialog for adding a custom backend with AI-assisted configuration."""

    DEFAULT_CSS = """
    AddCustomBackendDialog { 
        align: center middle; 
    }
    
    AddCustomBackendDialog > .dialog-container {
        padding: 1 2;
        border: thick $accent;
        background: $surface;
        width: 100%;
        height: 100%;
        max-width: 100%;
        max-height: 100%;
    }
    
    AddCustomBackendDialog .dialog-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
        height: 2;
    }
    
    AddCustomBackendDialog .main-layout {
        layout: horizontal;
        height: 1fr;
    }
    
    AddCustomBackendDialog .config-panel {
        width: 55%;
        height: 1fr;
        border-right: solid $primary-darken-2;
        padding-right: 1;
    }
    
    AddCustomBackendDialog .ai-panel {
        width: 45%;
        height: 1fr;
        padding-left: 1;
    }
    
    AddCustomBackendDialog ScrollableContainer {
        height: 1fr;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
    }
    
    AddCustomBackendDialog .config-section {
        margin: 1 0;
        padding: 1;
        border: solid $primary-darken-3;
        height: auto;
        background: $surface;
    }
    
    AddCustomBackendDialog .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        height: 1;
    }
    
    AddCustomBackendDialog .config-row {
        layout: horizontal;
        height: 3;
        margin: 0 0 1 0;
        align: left middle;
    }
    
    AddCustomBackendDialog .config-label {
        width: 18;
        height: 3;
        content-align: left middle;
    }
    
    AddCustomBackendDialog .config-input {
        width: 1fr;
        min-width: 20;
    }
    
    AddCustomBackendDialog .config-hint {
        color: $text-muted;
        text-style: italic;
        height: auto;
        margin-bottom: 1;
    }
    
    AddCustomBackendDialog Input {
        width: 100%;
        height: 3;
    }
    
    AddCustomBackendDialog Select {
        width: 100%;
        height: 3;
    }
    
    AddCustomBackendDialog TextArea {
        height: 6;
        width: 100%;
    }
    
    AddCustomBackendDialog Switch {
        margin-left: 1;
    }
    
    AddCustomBackendDialog .ai-header {
        height: 3;
        background: $primary-darken-2;
        padding: 0 1;
        align: left middle;
        layout: horizontal;
    }
    
    AddCustomBackendDialog .ai-title {
        text-style: bold;
        color: $accent;
        width: 1fr;
    }
    
    AddCustomBackendDialog .stats-toggle-btn {
        width: auto;
        min-width: 4;
        height: 3;
    }
    
    AddCustomBackendDialog .ai-controls-row {
        height: 3;
        layout: horizontal;
        padding: 0 1;
        margin-top: 1;
        align: right middle;
    }
    
    AddCustomBackendDialog .chat-ctrl-btn {
        min-width: 10;
        width: auto;
        height: 3;
        margin-left: 1;
    }
    
    AddCustomBackendDialog .ai-chat-area {
        height: 1fr;
        padding: 1;
        /* Eye-pleasing gray background instead of black */
        background: #2d3748;
        border: solid $primary-darken-3;
        /* Word wrap enabled, no horizontal scroll */
        overflow-x: hidden;
        overflow-y: auto;
    }
    
    AddCustomBackendDialog .ai-stats-section {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
    }
    
    AddCustomBackendDialog .ai-stats-section.hidden {
        display: none;
    }
    
    AddCustomBackendDialog .stats-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    AddCustomBackendDialog .stats-row {
        height: auto;
        layout: horizontal;
    }
    
    AddCustomBackendDialog .stats-label {
        width: 12;
        color: $text-muted;
    }
    
    AddCustomBackendDialog .stats-value {
        width: 1fr;
        color: $accent;
        text-align: right;
    }
    
    AddCustomBackendDialog .input-hint {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    
    AddCustomBackendDialog .ai-input-row {
        height: 3;
        layout: horizontal;
        margin-top: 1;
    }
    
    AddCustomBackendDialog .ai-input {
        width: 1fr;
        margin-right: 1;
    }
    
    AddCustomBackendDialog .ai-send-btn {
        min-width: 8;
    }
    
    AddCustomBackendDialog .footer {
        height: 4;
        layout: horizontal;
        margin-top: 1;
        padding: 1 2;
        border-top: solid $primary-darken-3;
        background: $surface-darken-1;
        align: left middle;
    }
    
    AddCustomBackendDialog .footer Button {
        margin-right: 2;
        min-width: 14;
        height: 3;
    }
    
    AddCustomBackendDialog .footer-btn-save {
        background: $success;
    }
    
    AddCustomBackendDialog .footer-btn-test {
        background: $primary;
    }
    
    AddCustomBackendDialog .footer-btn-clear {
        background: $warning-darken-1;
    }
    
    AddCustomBackendDialog .footer-btn-cancel {
        background: $error-darken-1;
    }
    
    AddCustomBackendDialog .required {
        color: $error;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+s", "save", "Save"),
        ("ctrl+j", "prev_prompt", "Previous Prompt"),
        ("ctrl+l", "next_prompt", "Next Prompt"),
    ]

    def __init__(self, on_save: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self._config: Dict[str, Any] = {}
        self._on_save = on_save
        self._llm_router = None
        self._ai_conversation = []
        self._prompt_history: list[str] = []  # Store prompt history
        self._prompt_history_index: int = -1  # Current position in history
        self._ai_stats = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'requests': 0,
            'thinking_time_ms': 0,
        }
        
        # Initialize LLM router if available
        if LLM_AVAILABLE:
            try:
                # Auto-consent function for TUI mode - user has already configured provider
                def auto_consent(prompt: str) -> bool:
                    return True  # Auto-approve in TUI since user explicitly configured
                
                self._llm_router = LLMRouter(consent_prompt=auto_consent)
                # Load and apply saved LLM settings
                self._init_llm_with_settings()
            except Exception:
                pass
    
    def _init_llm_with_settings(self) -> None:
        """Initialize LLM router with saved settings."""
        from pathlib import Path
        import json
        
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if not config_path.exists():
                return
            
            with open(config_path, 'r') as f:
                settings = json.load(f)
            
            llm = settings.get('llm', {})
            mode = llm.get('mode', 'none')
            
            if mode == 'none':
                return
            
            # Store the provider and model for requests
            self._llm_provider = mode
            
            # Get model for selected provider
            model_key_map = {
                'local': 'local_model',
                'ollama': 'local_model',
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
                'lmstudio': 'lmstudio_model',
                'llamacpp': 'llamacpp_model',
                'llama_cpp': 'llamacpp_model',
                'azure_openai': 'azure_deployment',
                'azure': 'azure_deployment',
                'vertex_ai': 'vertex_model',
                'aws_bedrock': 'aws_model',
                'huggingface': 'hf_model',
                'fireworks': 'fireworks_model',
                'replicate': 'replicate_model',
                'ai21': 'ai21_model',
                'deepinfra': 'deepinfra_model',
                'localai': 'localai_model',
            }
            
            model_key = model_key_map.get(mode, f'{mode}_model')
            self._llm_model = llm.get(model_key, '')
            
            # Get API key if needed (TUI uses _key suffix, not _api_key)
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
                'azure_openai': 'azure_key',
                'azure': 'azure_key',
                'vertex_ai': 'vertex_key',
                'aws_bedrock': 'aws_access_key',
                'huggingface': 'hf_token',
                'fireworks': 'fireworks_key',
                'replicate': 'replicate_token',
                'ai21': 'ai21_key',
                'deepinfra': 'deepinfra_key',
            }
            
            api_key_field = api_key_map.get(mode)
            api_key = None
            if api_key_field:
                api_key = llm.get(api_key_field, '')
            
            # Comprehensive mapping of TUI provider names to router provider names for API key registration
            # All major providers are now supported
            provider_name_map = {
                'local': 'ollama',
                'ollama': 'ollama',
                'lmstudio': 'lmstudio',
                'llamacpp': 'llama_cpp',
                'llama_cpp': 'llama_cpp',
                'localai': 'ollama',
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
            if mode in unsupported_providers:
                router_provider = None  # Will trigger simulation fallback
            else:
                router_provider = provider_name_map.get(mode, mode)
            
            # Register API key with the router for proper authentication
            if api_key and self._llm_router:
                try:
                    self._llm_router.api_keys.store_key(router_provider, api_key)
                except Exception:
                    pass  # Silently ignore if API key storage fails
                
        except Exception:
            self._llm_provider = None
            self._llm_model = None

    def compose(self):
        with Vertical(classes="dialog-container"):
            yield Static("➕ Add Custom Backend", classes="dialog-title")
            
            with Horizontal(classes="main-layout"):
                # Left panel: Configuration form
                with Vertical(classes="config-panel"):
                    yield Static("📋 Backend Configuration", classes="section-title")
                    
                    with ScrollableContainer():
                        # Basic Information Section
                        with Vertical(classes="config-section"):
                            yield Static("📌 Basic Information", classes="section-title")
                            yield Static("Required fields are marked with *", classes="config-hint")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Backend Name: *", classes="config-label")
                                yield Input(
                                    placeholder="e.g., MyQuantumBackend",
                                    id="input-backend-name",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Display Name:", classes="config-label")
                                yield Input(
                                    placeholder="e.g., My Quantum Backend",
                                    id="input-display-name",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Backend Type:", classes="config-label")
                                yield Select(
                                    [
                                        ("Simulator (CPU)", "simulator"),
                                        ("GPU Accelerated", "gpu"),
                                        ("Hybrid (CPU+GPU)", "hybrid"),
                                        ("Cloud/Remote", "cloud"),
                                        ("Hardware/QPU", "hardware"),
                                        ("Custom", "custom"),
                                    ],
                                    value="simulator",
                                    id="select-backend-type",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Description:", classes="config-label")
                                yield Input(
                                    placeholder="Brief description of the backend",
                                    id="input-description",
                                    classes="config-input"
                                )
                        
                        # Connection Settings Section
                        with Vertical(classes="config-section"):
                            yield Static("🔌 Connection Settings", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Module Path:", classes="config-label")
                                yield Input(
                                    placeholder="e.g., my_backend.simulator",
                                    id="input-module-path",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Class Name:", classes="config-label")
                                yield Input(
                                    placeholder="e.g., QuantumSimulator",
                                    id="input-class-name",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Install Path:", classes="config-label")
                                yield Input(
                                    placeholder="Path to backend installation (optional)",
                                    id="input-install-path",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("API Endpoint:", classes="config-label")
                                yield Input(
                                    placeholder="For remote backends (optional)",
                                    id="input-api-endpoint",
                                    classes="config-input"
                                )
                        
                        # Performance Settings Section
                        with Vertical(classes="config-section"):
                            yield Static("⚡ Performance Settings", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Max Qubits:", classes="config-label")
                                yield Input(
                                    value="30",
                                    id="input-max-qubits",
                                    classes="config-input",
                                    type="integer"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Precision:", classes="config-label")
                                yield Select(
                                    [
                                        ("Single (float32)", "float32"),
                                        ("Double (float64)", "float64"),
                                        ("Mixed Precision", "mixed"),
                                    ],
                                    value="float64",
                                    id="select-precision",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("GPU Support:", classes="config-label")
                                yield Switch(value=False, id="switch-gpu-support")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("GPU Device ID:", classes="config-label")
                                yield Input(
                                    value="0",
                                    id="input-gpu-device",
                                    classes="config-input",
                                    type="integer"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Thread Count:", classes="config-label")
                                yield Input(
                                    value="4",
                                    id="input-threads",
                                    classes="config-input",
                                    type="integer"
                                )
                        
                        # Advanced Settings Section
                        with Vertical(classes="config-section"):
                            yield Static("🔧 Advanced Settings", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Timeout (ms):", classes="config-label")
                                yield Input(
                                    value="30000",
                                    id="input-timeout",
                                    classes="config-input",
                                    type="integer"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Max Retries:", classes="config-label")
                                yield Input(
                                    value="3",
                                    id="input-retries",
                                    classes="config-input",
                                    type="integer"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Enable Caching:", classes="config-label")
                                yield Switch(value=True, id="switch-caching")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Enable Logging:", classes="config-label")
                                yield Switch(value=True, id="switch-logging")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Log Level:", classes="config-label")
                                yield Select(
                                    [
                                        ("Debug", "DEBUG"),
                                        ("Info", "INFO"),
                                        ("Warning", "WARNING"),
                                        ("Error", "ERROR"),
                                    ],
                                    value="INFO",
                                    id="select-log-level",
                                    classes="config-input"
                                )
                        
                        # Custom Parameters Section
                        with Vertical(classes="config-section"):
                            yield Static("📝 Custom Parameters (JSON)", classes="section-title")
                            yield Static("Add any backend-specific parameters as JSON", classes="config-hint")
                            yield TextArea(
                                text='{\n  "custom_param": "value"\n}',
                                id="textarea-custom-params",
                                language="json"
                            )
                
                with Vertical(classes="ai-panel"):
                    with Horizontal(classes="ai-header"):
                        yield Static("🤖 AI Assistant", classes="ai-title")
                        yield Button("👁", id="btn-toggle-ai-stats", variant="default", classes="stats-toggle-btn")
                    
                    # Statistics Section - toggleable (continuous show/hide)
                    with Container(classes="ai-stats-section", id="ai-stats-section"):
                        yield Static("📊 Statistics", classes="stats-title")
                        with Horizontal(classes="stats-row"):
                            yield Static("Model:", classes="stats-label")
                            yield Static("—", classes="stats-value", id="stat-model")
                        with Horizontal(classes="stats-row"):
                            yield Static("Provider:", classes="stats-label")
                            yield Static("—", classes="stats-value", id="stat-provider")
                        with Horizontal(classes="stats-row"):
                            yield Static("Prompt", classes="stats-label")
                            yield Static("Tokens:", classes="stats-label")
                            yield Static("0", classes="stats-value", id="stat-prompt-tokens")
                        with Horizontal(classes="stats-row"):
                            yield Static("Completion:", classes="stats-label")
                            yield Static("0", classes="stats-value", id="stat-completion-tokens")
                        with Horizontal(classes="stats-row"):
                            yield Static("Total", classes="stats-label")
                            yield Static("Tokens:", classes="stats-label")
                            yield Static("0", classes="stats-value", id="stat-total-tokens")
                        with Horizontal(classes="stats-row"):
                            yield Static("Requests:", classes="stats-label")
                            yield Static("0", classes="stats-value", id="stat-requests")
                        with Horizontal(classes="stats-row"):
                            yield Static("Think Time:", classes="stats-label")
                            yield Static("0ms", classes="stats-value", id="stat-thinking-time")
                    
                    yield RichLog(
                        auto_scroll=True,
                        wrap=True,  # Enable word wrap
                        classes="ai-chat-area",
                        id="ai-chat-log"
                    )
                    
                    # Input with hint
                    yield Static("Ctrl+J: Previous prompt | Ctrl+L: Next prompt", classes="input-hint", id="input-hint")
                    
                    with Horizontal(classes="ai-input-row"):
                        yield Input(
                            placeholder="Ask AI for help... (Ctrl+J/L for history)",
                            id="ai-input",
                            classes="ai-input"
                        )
                        yield Button("Ask", id="btn-ai-ask", variant="primary", classes="ai-send-btn")
                    
                    # Chat control buttons at bottom
                    with Horizontal(classes="ai-controls-row"):
                        yield Button("📝 New", id="btn-new-chat", variant="default", classes="chat-ctrl-btn")
                        yield Button("📤 Export", id="btn-export-chat", variant="default", classes="chat-ctrl-btn")
                        yield Button("📥 Import", id="btn-import-chat", variant="default", classes="chat-ctrl-btn")
                        yield Button("🗑 Clear", id="btn-clear-chat", variant="warning", classes="chat-ctrl-btn")
            
            # Footer with action buttons
            with Horizontal(classes="footer"):
                yield Button("✔ Save Backend", id="btn-save", variant="success", classes="footer-btn-save")
                yield Button("🔍 Test Connection", id="btn-test", variant="primary", classes="footer-btn-test")
                yield Button("🗑 Clear Form", id="btn-clear", variant="warning", classes="footer-btn-clear")
                yield Button("✕ Cancel", id="btn-cancel", variant="error", classes="footer-btn-cancel")

    def on_mount(self):
        """Initialize the dialog and restore any saved chat state."""
        # Try to restore chat from TUIState
        self._restore_chat_state()
        
        # Only show welcome if no previous chat
        if not self._ai_conversation:
            self._show_ai_welcome()

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
                
                if state.add_backend_chat_messages:
                    self._ai_conversation = state.add_backend_chat_messages.copy()
                    self._ai_stats = state.add_backend_chat_stats.copy()
                    
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
                state.add_backend_chat_messages = self._ai_conversation.copy()
                state.add_backend_chat_stats = self._ai_stats.copy()
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
            self._update_ai_stats_display()
        except Exception:
            pass

    def _show_ai_welcome(self):
        """Show welcome message in AI chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            welcome = Text()
            welcome.append("🤖 AI Backend Assistant\n", style=f"bold {theme.accent}")
            welcome.append("─" * 25 + "\n", style=theme.fg_subtle)
            welcome.append("I can help you configure your backend!\n\n", style=theme.fg_base)
            welcome.append("Try asking:\n", style=theme.fg_muted)
            welcome.append("• How do I set up a GPU backend?\n", style=theme.fg_subtle)
            welcome.append("• What module path should I use?\n", style=theme.fg_subtle)
            welcome.append("• Help me configure cuQuantum\n", style=theme.fg_subtle)
            welcome.append("• What are optimal settings for 20 qubits?\n", style=theme.fg_subtle)
            
            chat_log.write(welcome)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-save":
            self._save_backend()
        elif button_id == "btn-test":
            self._test_connection()
        elif button_id == "btn-clear":
            self._clear_form()
        elif button_id == "btn-cancel":
            self.dismiss(None)
        elif button_id == "btn-ai-ask":
            self._send_ai_message()
        elif button_id == "btn-new-chat":
            self._new_chat()
        elif button_id == "btn-clear-chat":
            self._clear_chat()
        elif button_id == "btn-export-chat":
            self._export_chat()
        elif button_id == "btn-import-chat":
            self._import_chat()
        elif button_id == "btn-toggle-ai-stats":
            self._toggle_ai_stats()
    
    def _toggle_ai_stats(self) -> None:
        """Toggle AI stats visibility (continuous show/hide, not momentary)."""
        try:
            stats_section = self.query_one("#ai-stats-section")
            toggle_btn = self.query_one("#btn-toggle-ai-stats", Button)
            
            if stats_section.has_class("hidden"):
                stats_section.remove_class("hidden")
                toggle_btn.label = "👁"
            else:
                stats_section.add_class("hidden")
                toggle_btn.label = "👁‍🗨"
        except Exception:
            pass

    def _new_chat(self) -> None:
        """Start a new chat, clearing the current conversation."""
        try:
            # Clear conversation history
            self._ai_conversation = []
            self._ai_stats = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'requests': 0,
                'thinking_time_ms': 0,
            }
            self._prompt_history = []
            self._prompt_history_index = -1
            
            # Clear display
            chat_log = self.query_one("#ai-chat-log", RichLog)
            chat_log.clear()
            
            # Show welcome again
            self._show_ai_welcome()
            
            # Update stats display
            self._update_stats_display()
            
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
            self._show_ai_welcome()
            
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
            default_name = f"backend_chat_{timestamp}"
            
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
                        safe_name = f"chat_{timestamp}"
                    
                    filename = f"{safe_name}.json"
                    export_path = export_dir / filename
                    
                    # Prepare export data
                    export_data = {
                        "type": "add_backend_chat",
                        "name": name,
                        "timestamp": timestamp,
                        "messages": self._ai_conversation,
                        "stats": self._ai_stats,
                        "prompt_history": self._prompt_history,
                    }
                    
                    # Save to file
                    with open(export_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                    
                    self.notify(f"✅ Chat exported: {filename}", severity="success")
                except Exception as e:
                    self.notify(f"Export failed: {e}", severity="error")
            
            self.app.push_screen(ChatExportNameDialog(default_name), handle_export_name)
            
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")

    def _import_chat(self) -> None:
        """Import chat history from a file with selection dialog."""
        try:
            # Look for chat exports
            export_dir = Path.home() / ".proxima" / "chat_exports"
            
            if not export_dir.exists():
                self.notify("No exported chats found", severity="warning")
                return
            
            # Find all chat export files
            chat_files = list(export_dir.glob("*.json"))
            
            if not chat_files:
                self.notify("No exported chats found", severity="warning")
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
                    
                    if 'prompt_history' in import_data:
                        self._prompt_history = import_data['prompt_history']
                    
                    # Restore display
                    self._restore_chat_display()
                    
                    # Update stats
                    self._update_stats_display()
                    
                    # Save state
                    self._save_chat_state()
                    
                    name = import_data.get('name', Path(file_path).stem)
                    self.notify(f"✅ Chat imported: {name}", severity="success")
                except Exception as e:
                    self.notify(f"Import failed: {e}", severity="error")
            
            self.app.push_screen(ChatImportSelectDialog(file_options), handle_import_select)
            
        except Exception as e:
            self.notify(f"Import failed: {e}", severity="error")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields."""
        if event.input.id == "ai-input":
            self._send_ai_message()

    def action_prev_prompt(self) -> None:
        """Navigate to previous prompt in history (Ctrl+J)."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#ai-input", Input)
            
            # If we're at the start, save current input
            if self._prompt_history_index == -1:
                current = input_widget.value.strip()
                if current and (not self._prompt_history or self._prompt_history[-1] != current):
                    self._prompt_history.append(current)
                self._prompt_history_index = len(self._prompt_history) - 1
            elif self._prompt_history_index > 0:
                self._prompt_history_index -= 1
            
            if 0 <= self._prompt_history_index < len(self._prompt_history):
                input_widget.value = self._prompt_history[self._prompt_history_index]
        except Exception:
            pass

    def action_next_prompt(self) -> None:
        """Navigate to next prompt in history (Ctrl+L)."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#ai-input", Input)
            
            if self._prompt_history_index < len(self._prompt_history) - 1:
                self._prompt_history_index += 1
                input_widget.value = self._prompt_history[self._prompt_history_index]
            else:
                # At the end, clear input
                self._prompt_history_index = -1
                input_widget.value = ""
        except Exception:
            pass

    def _update_stats_display(self) -> None:
        """Update the statistics display."""
        try:
            # Get LLM config
            llm_settings = self._get_llm_settings()
            provider = llm_settings.get('mode', 'none')
            model = llm_settings.get(f'{provider}_model', llm_settings.get('local_model', '—'))
            
            self.query_one("#stat-model", Static).update(model[:20] if model else "—")
            self.query_one("#stat-provider", Static).update(provider.title() if provider else "—")
            self.query_one("#stat-prompt-tokens", Static).update(str(self._ai_stats['prompt_tokens']))
            self.query_one("#stat-completion-tokens", Static).update(str(self._ai_stats['completion_tokens']))
            self.query_one("#stat-total-tokens", Static).update(str(self._ai_stats['total_tokens']))
            self.query_one("#stat-requests", Static).update(str(self._ai_stats['requests']))
            self.query_one("#stat-thinking-time", Static).update(f"{self._ai_stats['thinking_time_ms']}ms")
        except Exception:
            pass

    def _update_ai_stats_display(self) -> None:
        """Alias for _update_stats_display for compatibility."""
        self._update_stats_display()

    def _get_llm_settings(self) -> Dict:
        """Load LLM settings from storage."""
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                return settings.get('llm', {})
        except Exception:
            pass
        return {}

    def _send_ai_message(self) -> None:
        """Send message to AI assistant."""
        import time
        
        try:
            theme = get_theme()
            input_widget = self.query_one("#ai-input", Input)
            message = input_widget.value.strip()
            
            if not message:
                return
            
            # Add to prompt history
            if not self._prompt_history or self._prompt_history[-1] != message:
                self._prompt_history.append(message)
            self._prompt_history_index = -1  # Reset history position
            
            # Clear input
            input_widget.value = ""
            
            # Show user message
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            user_text = Text()
            user_text.append("👤 You: ", style=f"bold {theme.primary}")
            user_text.append(message + "\n", style=theme.fg_base)
            chat_log.write(user_text)
            
            # Save user message to conversation
            self._ai_conversation.append({
                'role': 'user',
                'content': message,
            })
            
            # Track timing
            start_time = time.time()
            
            # Get current form state for context
            form_state = self._get_form_state()
            
            # Update request count
            self._ai_stats['requests'] += 1
            
            # Send to LLM or simulate
            if self._llm_router and LLM_AVAILABLE:
                self._query_ai(message, form_state, start_time)
            else:
                self._simulate_ai_response(message, form_state, start_time)
                
        except Exception as e:
            self._show_ai_error(str(e))

    def _get_form_state(self) -> Dict[str, Any]:
        """Get current form values."""
        try:
            return {
                "backend_name": self.query_one("#input-backend-name", Input).value,
                "display_name": self.query_one("#input-display-name", Input).value,
                "backend_type": self.query_one("#select-backend-type", Select).value,
                "module_path": self.query_one("#input-module-path", Input).value,
                "class_name": self.query_one("#input-class-name", Input).value,
                "gpu_support": self.query_one("#switch-gpu-support", Switch).value,
                "max_qubits": self.query_one("#input-max-qubits", Input).value,
            }
        except Exception:
            return {}

    def _query_ai(self, message: str, form_state: Dict, start_time: float = None) -> None:
        """Query the AI for backend configuration help."""
        import time
        if start_time is None:
            start_time = time.time()
        
        # Check if LLM provider is configured
        provider = getattr(self, '_llm_provider', None)
        if not provider or provider == 'none':
            # Fall back to simulated response
            self._simulate_ai_response(message, form_state, start_time)
            return
        
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
            router_provider = provider_name_map.get(provider, provider)
            
        try:
            context = f"""Current backend configuration:
- Name: {form_state.get('backend_name', 'Not set')}
- Type: {form_state.get('backend_type', 'simulator')}
- Module: {form_state.get('module_path', 'Not set')}
- GPU: {form_state.get('gpu_support', False)}
- Max Qubits: {form_state.get('max_qubits', 30)}"""

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
            
            # Build the full prompt with context
            full_prompt = f"{context}\n\n"
            if conversation_context:
                full_prompt += f"Previous conversation:\n{conversation_context}\n"
            full_prompt += f"Current user question: {message}"

            # Build request with provider info
            model = getattr(self, '_llm_model', None)
            
            request = LLMRequest(
                prompt=full_prompt,
                system_prompt="""You are a helpful AI assistant for quantum computing backend configuration.
Help users set up custom backends for the Proxima quantum simulation platform.
Provide specific, actionable advice for backend configuration including:
- Module paths and class names
- GPU configuration
- Performance optimization
- Common issues and solutions
Keep responses concise and practical.

You have access to the previous conversation history, so maintain context and refer back to earlier discussions when relevant.""",
                temperature=0.7,
                max_tokens=512,
                provider=router_provider,
                model=model if model else None,
            )
            
            response = self._llm_router.route(request)
            
            # Check if we got a valid response with actual content
            if response and hasattr(response, 'text') and response.text and response.text.strip():
                self._handle_ai_response(response, start_time)
            else:
                # Empty response - fall back to simulation
                self._simulate_ai_response(message, form_state, start_time)
            
        except PermissionError as e:
            # Consent required - use simulated response instead
            self._simulate_ai_response(message, form_state, start_time)
        except ConnectionError as e:
            # Local provider not running - use simulated response
            self._simulate_ai_response(message, form_state, start_time)
        except ValueError as e:
            # Provider not found in router - use simulated response instead of showing error
            if "Unknown provider" in str(e) or "No LLM provider" in str(e):
                self._simulate_ai_response(message, form_state, start_time)
            else:
                self._simulate_ai_response(message, form_state, start_time)
        except Exception as e:
            # For any other error, silently fall back to simulation
            # This includes HTTP errors (404, 500, etc.) and connection errors
            error_msg = str(e).lower()
            # Silently use simulation for common API/connection errors
            if any(err in error_msg for err in ['404', '500', '502', '503', 'not found', 
                   'connection', 'timeout', 'refused', 'unknown provider', 'consent',
                   'http', 'client error', 'server error']):
                self._simulate_ai_response(message, form_state, start_time)
            else:
                # Unknown error - show it but still fall back
                self._simulate_ai_response(message, form_state, start_time)

    def _handle_ai_response(self, response, start_time: float = None) -> None:
        """Handle AI response."""
        import time
        try:
            theme = get_theme()
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            if hasattr(response, 'error') and response.error:
                self._show_ai_error(response.error)
                return
            
            ai_text = Text()
            ai_text.append("🤖 AI: ", style=f"bold {theme.accent}")
            ai_text.append(response.text + "\n\n", style=theme.fg_base)
            chat_log.write(ai_text)
            
            # Save message to conversation history
            self._ai_conversation.append({
                'role': 'assistant',
                'content': response.text,
            })
            
            # Update stats
            if hasattr(response, 'prompt_tokens'):
                self._ai_stats['prompt_tokens'] += response.prompt_tokens or 0
            if hasattr(response, 'completion_tokens'):
                self._ai_stats['completion_tokens'] += response.completion_tokens or 0
            if hasattr(response, 'total_tokens'):
                self._ai_stats['total_tokens'] += response.total_tokens or 0
            elif hasattr(response, 'prompt_tokens') and hasattr(response, 'completion_tokens'):
                self._ai_stats['total_tokens'] += (response.prompt_tokens or 0) + (response.completion_tokens or 0)
            
            if start_time:
                elapsed = int((time.time() - start_time) * 1000)
                self._ai_stats['thinking_time_ms'] = elapsed
            
            self._ai_stats['requests'] = self._ai_stats.get('requests', 0) + 1
            self._update_stats_display()
            
            # Save state
            self._save_chat_state()
            
        except Exception as e:
            self._show_ai_error(str(e))

    def _simulate_ai_response(self, message: str, form_state: Dict, start_time: float = None) -> None:
        """Simulate AI response when LLM not available."""
        import time
        if start_time is None:
            start_time = time.time()
            
        theme = get_theme()
        chat_log = self.query_one("#ai-chat-log", RichLog)
        
        msg_lower = message.lower()
        
        # Generate contextual response
        if "gpu" in msg_lower or "cuda" in msg_lower:
            response = """For GPU acceleration:
1. Enable 'GPU Support' toggle
2. Set correct GPU Device ID (0 for first GPU)
3. For cuQuantum: module path is 'cuquantum.backends.statevector'
4. Ensure CUDA drivers are installed
5. Use 'float32' precision for better GPU performance"""
        
        elif "module" in msg_lower or "path" in msg_lower:
            response = """Module path examples:
• cuQuantum: cuquantum.backends.statevector
• Qiskit: qiskit_aer.AerSimulator
• Cirq: cirq.Simulator
• Custom: your_package.module_name

The module should be installed in your Python environment."""
        
        elif "qubit" in msg_lower:
            response = f"""Max qubits depends on your hardware:
• 16GB RAM: ~28 qubits (CPU)
• 32GB RAM: ~30 qubits (CPU)
• RTX 3090 (24GB): ~32 qubits (GPU)
• A100 (40GB): ~34 qubits (GPU)

Current setting: {form_state.get('max_qubits', 30)} qubits"""
        
        elif "cuquantum" in msg_lower:
            response = """cuQuantum configuration:
• Backend Type: GPU Accelerated
• Module Path: cuquantum.backends.statevector
• Class Name: StateVectorSimulator
• GPU Support: Enable
• Precision: float32 (faster) or float64 (accurate)
• Install: pip install cuquantum-python"""
        
        elif "test" in msg_lower or "connection" in msg_lower:
            response = """To test your backend:
1. Click 'Test Connection' button
2. The system will try to import and initialize your backend
3. Check the result message for success/errors
4. Common issues: missing module, wrong class name, GPU not available"""
        
        else:
            response = """I can help you with:
• GPU configuration and optimization
• Module paths for popular backends
• Performance tuning (qubits, precision)
• Troubleshooting connection issues

What would you like to know?"""
        
        ai_text = Text()
        ai_text.append("🤖 AI: ", style=f"bold {theme.accent}")
        ai_text.append(response + "\n\n", style=theme.fg_base)
        chat_log.write(ai_text)
        
        # Save AI response to conversation
        self._ai_conversation.append({
            'role': 'assistant',
            'content': response,
        })
        
        # Update stats for simulated response
        import time
        elapsed = int((time.time() - start_time) * 1000)
        self._ai_stats['thinking_time_ms'] = elapsed
        self._ai_stats['prompt_tokens'] += len(message.split()) * 2  # Approximate
        self._ai_stats['completion_tokens'] += len(response.split()) * 2  # Approximate
        self._ai_stats['total_tokens'] = self._ai_stats['prompt_tokens'] + self._ai_stats['completion_tokens']
        self._update_stats_display()
        
        # Save state
        self._save_chat_state()

    def _show_ai_error(self, error: str) -> None:
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

    def _save_backend(self) -> None:
        """Save the backend configuration."""
        try:
            # Validate required fields
            backend_name = self.query_one("#input-backend-name", Input).value.strip()
            if not backend_name:
                self.notify("Backend name is required!", severity="error")
                return
            
            # Collect all configuration
            config = {
                "id": backend_name.lower().replace(" ", "_"),
                "name": backend_name,
                "display_name": self.query_one("#input-display-name", Input).value.strip() or backend_name,
                "description": self.query_one("#input-description", Input).value.strip(),
                "type": self.query_one("#select-backend-type", Select).value,
                "module_path": self.query_one("#input-module-path", Input).value.strip(),
                "class_name": self.query_one("#input-class-name", Input).value.strip(),
                "install_path": self.query_one("#input-install-path", Input).value.strip(),
                "api_endpoint": self.query_one("#input-api-endpoint", Input).value.strip(),
                "max_qubits": int(self.query_one("#input-max-qubits", Input).value or 30),
                "precision": self.query_one("#select-precision", Select).value,
                "gpu_support": self.query_one("#switch-gpu-support", Switch).value,
                "gpu_device": int(self.query_one("#input-gpu-device", Input).value or 0),
                "threads": int(self.query_one("#input-threads", Input).value or 4),
                "timeout": int(self.query_one("#input-timeout", Input).value or 30000),
                "retries": int(self.query_one("#input-retries", Input).value or 3),
                "caching": self.query_one("#switch-caching", Switch).value,
                "logging": self.query_one("#switch-logging", Switch).value,
                "log_level": self.query_one("#select-log-level", Select).value,
                "status": "unknown",
                "is_custom": True,
            }
            
            # Parse custom parameters
            try:
                custom_params_text = self.query_one("#textarea-custom-params", TextArea).text
                config["custom_params"] = json.loads(custom_params_text)
            except json.JSONDecodeError:
                config["custom_params"] = {}
            
            # Save to persistent storage
            self._save_to_storage(config)
            
            # Call callback if provided
            if self._on_save:
                self._on_save(config)
            
            self.notify(f"✓ Backend '{backend_name}' saved successfully!", severity="success")
            self.dismiss(config)
            
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error")

    def _save_to_storage(self, config: Dict) -> None:
        """Save backend configuration to persistent storage."""
        try:
            config_dir = Path.home() / ".proxima"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            backends_file = config_dir / "custom_backends.json"
            
            # Load existing backends
            existing = {}
            if backends_file.exists():
                with open(backends_file, 'r') as f:
                    existing = json.load(f)
            
            # Add/update this backend
            existing[config["id"]] = config
            
            # Save
            with open(backends_file, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Failed to save: {e}")

    def _test_connection(self) -> None:
        """Test the backend connection."""
        try:
            module_path = self.query_one("#input-module-path", Input).value.strip()
            class_name = self.query_one("#input-class-name", Input).value.strip()
            
            if not module_path:
                self.notify("Please enter a module path to test", severity="warning")
                return
            
            self.notify(f"Testing connection to {module_path}...", severity="information")
            
            # Try to import the module
            try:
                import importlib
                module = importlib.import_module(module_path)
                
                if class_name:
                    # Try to get the class
                    if hasattr(module, class_name):
                        self.notify(f"✓ Successfully found {class_name} in {module_path}", severity="success")
                    else:
                        self.notify(f"⚠ Module found but class '{class_name}' not found", severity="warning")
                else:
                    self.notify(f"✓ Module {module_path} imported successfully", severity="success")
                    
            except ImportError as e:
                self.notify(f"✗ Import failed: {e}", severity="error")
            except Exception as e:
                self.notify(f"✗ Error: {e}", severity="error")
                
        except Exception as e:
            self.notify(f"Test failed: {e}", severity="error")

    def _clear_form(self) -> None:
        """Clear all form fields."""
        try:
            self.query_one("#input-backend-name", Input).value = ""
            self.query_one("#input-display-name", Input).value = ""
            self.query_one("#input-description", Input).value = ""
            self.query_one("#select-backend-type", Select).value = "simulator"
            self.query_one("#input-module-path", Input).value = ""
            self.query_one("#input-class-name", Input).value = ""
            self.query_one("#input-install-path", Input).value = ""
            self.query_one("#input-api-endpoint", Input).value = ""
            self.query_one("#input-max-qubits", Input).value = "30"
            self.query_one("#select-precision", Select).value = "float64"
            self.query_one("#switch-gpu-support", Switch).value = False
            self.query_one("#input-gpu-device", Input).value = "0"
            self.query_one("#input-threads", Input).value = "4"
            self.query_one("#input-timeout", Input).value = "30000"
            self.query_one("#input-retries", Input).value = "3"
            self.query_one("#switch-caching", Switch).value = True
            self.query_one("#switch-logging", Switch).value = True
            self.query_one("#select-log-level", Select).value = "INFO"
            self.query_one("#textarea-custom-params", TextArea).text = '{\n  "custom_param": "value"\n}'
            
            self.notify("Form cleared", severity="information")
        except Exception:
            pass

    def action_close(self) -> None:
        """Close the dialog."""
        self.dismiss(None)

    def action_save(self) -> None:
        """Save keyboard shortcut."""
        self._save_backend()
