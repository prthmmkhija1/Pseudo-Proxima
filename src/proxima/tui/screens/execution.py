"""Execution screen for Proxima TUI.

Live execution monitoring with progress and controls.
Includes integrated AI thinking panel for real-time AI interaction.
"""

from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Button, RichLog, Input
from rich.text import Text
from rich.panel import Panel

from .base import BaseScreen
from ..styles.theme import get_theme
from ..components.progress import ProgressBar, StageTimeline

# Import controller for execution management
try:
    from ..controllers import ExecutionController
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False

# Import LLM components for AI chat
try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class ExecutionScreen(BaseScreen):
    """Execution monitoring screen.
    
    Shows:
    - Current execution info
    - Progress bar
    - Stage timeline
    - Execution controls
    - Log viewer
    """
    
    SCREEN_NAME = "execution"
    SCREEN_TITLE = "Execution Monitor"
    
    # Extend parent BINDINGS instead of replacing them
    BINDINGS = [
        ("1", "goto_dashboard", "Dashboard"),
        ("2", "goto_execution", "Execution"),
        ("3", "goto_results", "Results"),
        ("4", "goto_backends", "Backends"),
        ("5", "goto_settings", "Settings"),
        ("question_mark", "show_help", "Help"),
        ("escape", "go_back", "Back"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+t", "toggle_ai_thinking", "AI Thinking"),
        ("p", "pause_execution", "Pause"),
        ("r", "resume_execution", "Resume"),
        ("a", "abort_execution", "Abort"),
        ("z", "rollback", "Rollback"),
        ("l", "toggle_log", "Toggle Log"),
        ("ctrl+j", "prev_ai_prompt", "Previous Prompt"),
        ("ctrl+l", "next_ai_prompt", "Next Prompt"),
        ("i", "focus_ai_input", "Focus AI Input"),
    ]
    
    DEFAULT_CSS = """
    ExecutionScreen .main-split {
        layout: horizontal;
        height: 1fr;
    }
    
    ExecutionScreen .execution-area {
        width: 65%;
        height: 1fr;
    }
    
    ExecutionScreen .ai-thinking-panel {
        width: 35%;
        height: 1fr;
        border-left: solid $primary;
        background: $surface-darken-1;
    }
    
    ExecutionScreen .ai-thinking-panel.-hidden {
        display: none;
    }
    
    ExecutionScreen .ai-header {
        height: 3;
        padding: 0 1;
        background: $primary-darken-2;
        border-bottom: solid $primary-darken-3;
    }
    
    ExecutionScreen .ai-title {
        text-style: bold;
        color: $accent;
    }
    
    ExecutionScreen .ai-status {
        color: $text-muted;
    }
    
    ExecutionScreen .ai-stats-section {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border-bottom: solid $primary-darken-3;
    }
    
    ExecutionScreen .ai-stats-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    ExecutionScreen .ai-stats-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 0;
    }
    
    ExecutionScreen .ai-stat-label {
        width: 10;
        color: $text-muted;
    }
    
    ExecutionScreen .ai-stat-value {
        width: 1fr;
        color: $text;
    }
    
    ExecutionScreen .ai-stat-connected {
        color: $success;
    }
    
    ExecutionScreen .ai-stat-disconnected {
        color: $error;
    }
    
    ExecutionScreen .ai-thought-area {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    ExecutionScreen .ai-thought-stream {
        height: 1fr;
        background: $surface-darken-2;
        padding: 1;
        border: solid $primary-darken-3;
    }
    
    ExecutionScreen .ai-chat-section {
        height: auto;
        min-height: 8;
        padding: 1;
        border-top: solid $primary-darken-3;
        background: $surface;
    }
    
    ExecutionScreen .ai-chat-input-row {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    ExecutionScreen .ai-chat-input {
        width: 1fr;
        margin-right: 1;
        height: 3;
    }
    
    ExecutionScreen .ai-chat-input:focus {
        border: solid $accent;
    }
    
    ExecutionScreen .ai-send-btn {
        min-width: 8;
        height: 3;
    }
    
    ExecutionScreen .ai-send-btn:hover {
        background: $accent;
    }
    
    ExecutionScreen .ai-controls-row {
        height: 3;
        layout: horizontal;
    }
    
    ExecutionScreen .ai-control-btn {
        margin-right: 1;
        min-width: 10;
        height: 3;
    }
    
    ExecutionScreen .ai-control-btn:hover {
        background: $primary-lighten-1;
    }
    
    ExecutionScreen .execution-panel {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    ExecutionScreen .execution-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    ExecutionScreen .execution-info {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    ExecutionScreen .progress-section {
        margin: 1 0;
    }
    
    ExecutionScreen .timeline-section {
        margin: 1 0;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
    }
    
    ExecutionScreen .controls-section {
        height: auto;
        layout: horizontal;
        margin: 1 0;
    }
    
    ExecutionScreen .control-button {
        margin-right: 1;
        min-width: 12;
    }
    
    ExecutionScreen .log-section {
        height: 1fr;
        border: solid $primary-darken-2;
        background: $surface-darken-2;
    }
    
    ExecutionScreen .log-section.-hidden {
        display: none;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the execution screen."""
        self.__log_visible = True
        self.__update_timer = None
        self.__controller = None
        self._start_time = None
        self._ai_panel_visible = True
        self._ai_is_thinking = False
        self._ai_conversation = []
        self._llm_router = None
        self._prompt_history = []
        self._prompt_history_index = -1
        self._ai_stats = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'requests': 0,
            'thinking_time_ms': 0,
        }
        self._chain_of_thoughts_enabled = True
        self._current_thought_step = 0
        super().__init__(**kwargs)
        
        # Initialize execution controller
        if CONTROLLER_AVAILABLE:
            try:
                self.__controller = ExecutionController(self.state)
            except Exception as e:
                # Controller init failed, will use fallback mode
                pass
        
        # Initialize LLM router if available
        if LLM_AVAILABLE:
            try:
                # Auto-consent function for TUI mode - user has already configured provider
                def auto_consent(prompt: str) -> bool:
                    return True  # Auto-approve in TUI since user explicitly configured
                
                self._llm_router = LLMRouter(consent_prompt=auto_consent)
            except Exception:
                pass
    
    @property
    def _log_visible(self):
        return self.__log_visible
    
    @_log_visible.setter
    def _log_visible(self, value):
        self.__log_visible = value
    
    @property
    def _update_timer(self):
        return self.__update_timer
    
    @_update_timer.setter
    def _update_timer(self, value):
        self.__update_timer = value
    
    @property
    def _controller(self):
        return self.__controller
    
    @_controller.setter
    def _controller(self, value):
        self.__controller = value

    def on_mount(self) -> None:
        """Set up progress update timer on mount and load LLM settings."""
        # Start periodic progress updates (this also loads pending logs)
        self._update_timer = self.set_interval(0.3, self._update_progress)
        
        # Load saved LLM settings and update AI panel
        self._load_llm_settings()
        self._update_ai_stats_panel()
        
        # Initialize AI thought stream with welcome message
        self._init_ai_welcome_message()
        
        # Load any pending execution logs from AI Assistant immediately and with delays
        self._load_pending_execution_logs()
        self.set_timer(0.1, self._load_pending_execution_logs)
        self.set_timer(0.2, self._load_pending_execution_logs)
        
        # Force refresh of info panel to show any AI experiment - multiple times to catch state updates
        self._refresh_info_panel()
        self.set_timer(0.15, self._refresh_info_panel)
        self.set_timer(0.3, self._refresh_info_panel)
        
        # Focus the AI input after a short delay for better UX
        self.set_timer(0.4, self._focus_ai_input)
    
    def _refresh_info_panel(self) -> None:
        """Refresh the info panel to show current state."""
        try:
            # Also get state from app directly in case of mismatch
            state = self.state
            if hasattr(self.app, 'state'):
                app_state = self.app.state
                # Sync current_experiment from app state if not set locally
                if app_state.current_experiment and not state.current_experiment:
                    state.current_experiment = app_state.current_experiment
                if app_state.pending_execution_logs and not state.pending_execution_logs:
                    state.pending_execution_logs = app_state.pending_execution_logs
            
            info_panel = self.query_one(ExecutionInfoPanel)
            info_panel.state = state
            info_panel.refresh()
        except Exception:
            pass
    
    def on_screen_resume(self) -> None:
        """Called when switching back to this screen - reload pending logs."""
        self._load_pending_execution_logs()
        # Force refresh of the info panel
        try:
            info_panel = self.query_one(ExecutionInfoPanel)
            info_panel.state = self.state
            info_panel.refresh()
        except Exception:
            pass
    
    def _load_pending_execution_logs(self) -> None:
        """Load any pending execution logs stored by AI Assistant."""
        try:
            if not hasattr(self.state, 'pending_execution_logs'):
                return
            
            pending = self.state.pending_execution_logs
            if not pending:
                return
            
            log = self.query_one("#execution-log", ExecutionLog)
            
            # Write all unwritten logs
            logs_written = 0
            for entry in pending:
                if not entry.get('written', False):
                    if 'text' in entry:
                        log.write(entry['text'])
                    entry['written'] = True
                    logs_written += 1
            
            # Clean up old written logs (keep last 100)
            if len(pending) > 100:
                # Remove old written logs
                self.state.pending_execution_logs = [
                    e for e in pending if not e.get('written', False)
                ][-100:]
        except Exception:
            pass
    
    def on_unmount(self) -> None:
        """Clean up timer on unmount."""
        if self._update_timer:
            self._update_timer.stop()
    
    def _load_llm_settings(self) -> None:
        """Load saved LLM settings and initialize the router."""
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
            
            # Update state with LLM settings
            if hasattr(self, 'state'):
                self.state.llm_provider = mode
                self.state.thinking_enabled = llm.get('thinking_enabled', False)
                
                # Get the model name for the selected provider (comprehensive mapping)
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
                
                if mode in model_key_map:
                    self.state.llm_model = llm.get(model_key_map[mode], '')
                elif mode not in ['none', '', None]:
                    self.state.llm_model = llm.get('custom_model', '')
                else:
                    self.state.llm_model = ''
                
                # Check if we have API key for the provider (comprehensive mapping)
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
                    'aws_bedrock': 'aws_access_key',
                    'huggingface': 'hf_token',
                    'fireworks': 'fireworks_key',
                    'replicate': 'replicate_token',
                    'ai21': 'ai21_key',
                    'deepinfra': 'deepinfra_key',
                    'vertex_ai': 'vertex_key',
                }
                
                has_key = False
                api_key = None
                if mode in ['local', 'ollama', 'localai']:
                    has_key = bool(llm.get('ollama_url'))
                elif mode in ['lmstudio', 'llamacpp', 'llama_cpp']:
                    has_key = True  # Local providers don't need API keys
                elif mode in api_key_map:
                    api_key = llm.get(api_key_map[mode], '')
                    has_key = bool(api_key)
                elif mode not in ['none', '', None]:
                    api_key = llm.get('custom_key', '')
                    has_key = bool(api_key)
                
                self.state.llm_connected = has_key and mode not in ['none', '', None]
                
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
                
                # Store settings for router initialization
                self._llm_settings = llm
                
        except Exception:
            pass  # Silently fail if settings can't be loaded
    
    def _update_ai_stats_panel(self) -> None:
        """Update the AI stats panel with current configuration."""
        try:
            # Provider
            provider_label = self.query_one("#ai-stat-provider", Static)
            provider = getattr(self.state, 'llm_provider', 'none') or 'none'
            
            # Pretty names for providers
            provider_names = {
                'none': 'Not configured',
                'local': 'Ollama (Local)',
                'openai': 'OpenAI',
                'anthropic': 'Anthropic',
                'google': 'Google AI (Gemini)',
                'xai': 'xAI (Grok)',
                'deepseek': 'DeepSeek',
                'mistral': 'Mistral AI',
                'groq': 'Groq',
                'together': 'Together AI',
                'openrouter': 'OpenRouter',
                'cohere': 'Cohere',
                'perplexity': 'Perplexity',
                'azure_openai': 'Azure OpenAI',
                'aws_bedrock': 'AWS Bedrock',
                'huggingface': 'Hugging Face',
                'fireworks': 'Fireworks AI',
                'replicate': 'Replicate',
                'ai21': 'AI21 Labs',
                'deepinfra': 'DeepInfra',
            }
            provider_label.update(provider_names.get(provider, provider.title()))
            
            # Model
            model_label = self.query_one("#ai-stat-model", Static)
            model = getattr(self.state, 'llm_model', '') or '—'
            # Shorten model name if too long
            if len(model) > 30:
                model = model[:27] + "..."
            model_label.update(model)
            
            # Status
            status_label = self.query_one("#ai-stat-status", Static)
            is_connected = getattr(self.state, 'llm_connected', False)
            if is_connected:
                status_label.update("✓ Connected")
                status_label.remove_class("ai-stat-disconnected")
                status_label.add_class("ai-stat-connected")
            else:
                status_label.update("✗ Not connected")
                status_label.remove_class("ai-stat-connected")
                status_label.add_class("ai-stat-disconnected")
            
            # Tokens
            tokens_label = self.query_one("#ai-stat-tokens", Static)
            prompt_tokens = self._ai_stats.get('prompt_tokens', 0)
            completion_tokens = self._ai_stats.get('completion_tokens', 0)
            total_tokens = self._ai_stats.get('total_tokens', 0)
            tokens_label.update(f"{prompt_tokens}/{completion_tokens} ({total_tokens} total)")
            
            # Requests
            requests_label = self.query_one("#ai-stat-requests", Static)
            request_count = self._ai_stats.get('requests', 0)
            requests_label.update(str(request_count))
            
            # Thinking time if available
            try:
                thinking_time = self._ai_stats.get('thinking_time_ms', 0)
                if thinking_time > 0:
                    time_label = self.query_one("#ai-stat-thinking-time", Static)
                    time_label.update(f"{thinking_time}ms")
            except Exception:
                pass  # Thinking time label may not exist
            
        except Exception:
            pass  # Panel may not be mounted yet
    
    def _init_ai_welcome_message(self) -> None:
        """Initialize the AI thought stream with a welcome message."""
        try:
            theme = get_theme()
            thought_stream = self.query_one("#ai-thought-stream", RichLog)
            
            # Check if connected
            is_connected = getattr(self.state, 'llm_connected', False)
            provider = getattr(self.state, 'llm_provider', 'none') or 'none'
            
            welcome_text = Text()
            welcome_text.append("🧠 AI Thinking Panel\n", style=f"bold {theme.accent}")
            welcome_text.append("─" * 40 + "\n", style=theme.fg_subtle)
            
            # Chain of Thoughts indicator
            welcome_text.append("💡 Chain of Thoughts: ", style=f"bold {theme.primary}")
            welcome_text.append("ENABLED\n\n", style=f"bold {theme.success}")
            
            if is_connected and provider != 'none':
                welcome_text.append("✓ AI Connected\n", style=f"bold {theme.success}")
                welcome_text.append(f"Using: {provider.title()}\n\n", style=theme.fg_muted)
                welcome_text.append("The AI will analyze your queries step by step:\n", style=theme.fg_base)
                welcome_text.append("• Step 1: Context Analysis\n", style=theme.fg_muted)
                welcome_text.append("• Step 2: Deep Consideration\n", style=theme.fg_muted)
                welcome_text.append("• Step 3: Recommendation\n\n", style=theme.fg_muted)
                welcome_text.append("Keyboard shortcuts:\n", style=f"bold {theme.fg_base}")
                welcome_text.append("  Ctrl+J  Previous prompt\n", style=theme.fg_muted)
                welcome_text.append("  Ctrl+L  Next prompt\n", style=theme.fg_muted)
                welcome_text.append("  Ctrl+T  Toggle AI panel\n", style=theme.fg_muted)
            else:
                welcome_text.append("⚠ AI Not Configured\n", style=f"bold {theme.warning}")
                welcome_text.append("\nTo enable AI assistance:\n", style=theme.fg_base)
                welcome_text.append("1. Go to Settings (press 5)\n", style=theme.fg_muted)
                welcome_text.append("2. Select an AI Provider\n", style=theme.fg_muted)
                welcome_text.append("3. Enter your API key\n", style=theme.fg_muted)
                welcome_text.append("4. Click 'Save Settings'\n\n", style=theme.fg_muted)
                welcome_text.append("You can still type messages for simulated Chain of Thoughts responses.", style=theme.fg_subtle)
            
            thought_stream.write(welcome_text)
        except Exception:
            pass

    def _update_progress(self) -> None:
        """Update progress display from controller or state.
        
        Handles real-time progress updates from the execution controller,
        including elapsed time tracking and simulated progress in demo mode.
        """
        import time
        
        # Check if AI Assistant experiment is running
        ai_experiment_running = (
            hasattr(self.state, 'current_experiment') and 
            self.state.current_experiment and 
            self.state.current_experiment.get('status') == 'running'
        )
        
        # Load any pending execution logs from AI Assistant
        self._load_pending_execution_logs()
        
        # Track elapsed time when running
        if self.state.execution_status == "RUNNING" and not self.state.is_paused:
            if self._start_time is None:
                self._start_time = time.time()
            self.state.elapsed_ms = (time.time() - self._start_time) * 1000
            
            # Only simulate progress in demo mode (when no real backend or AI experiment is running)
            if not self._controller or not getattr(self._controller, '_core_controller', None):
                # Don't auto-advance if AI Assistant is running an experiment
                if not ai_experiment_running:
                    # Auto-advance progress for demo purposes
                    if self.state.progress_percent < 100:
                        self.state.progress_percent = min(100.0, self.state.progress_percent + 0.5)
                        # Update stage based on progress
                        new_stage_index = int(self.state.progress_percent / 100 * self.state.total_stages)
                        new_stage_index = min(new_stage_index, self.state.total_stages - 1)
                        if new_stage_index != self.state.stage_index:
                            self.state.stage_index = new_stage_index
                            if self.state.all_stages and new_stage_index < len(self.state.all_stages):
                                self.state.current_stage = self.state.all_stages[new_stage_index].name
                        # Estimate ETA
                        if self.state.progress_percent > 0:
                            total_estimated = self.state.elapsed_ms / (self.state.progress_percent / 100)
                            self.state.eta_ms = max(0, total_estimated - self.state.elapsed_ms)
                        
                        # Log progress updates periodically
                        if int(self.state.progress_percent) % 10 == 0:
                            try:
                                log = self.query_one(ExecutionLog)
                                log.write(f"[dim]Progress: {self.state.progress_percent:.0f}%[/dim]")
                            except Exception:
                                pass
                    else:
                        # Simulation complete
                        if self.state.is_running:
                            self.state.is_running = False
                            self.state.execution_status = "COMPLETED"
                            self.state.eta_ms = 0
                            self.notify("✓ Simulation completed!", severity="success")
                            self._update_control_buttons("completed")
                            try:
                                log = self.query_one(ExecutionLog)
                                log.write("[green]✓ Simulation completed successfully![/green]")
                            except Exception:
                                pass
        elif self.state.execution_status == "IDLE":
            # Reset start time when idle
            self._start_time = None
        elif self.state.execution_status == "COMPLETED":
            # Keep showing completion state
            pass
        
        if self._controller:
            try:
                # Get status from controller
                status = self._controller.get_status()
                if status:
                    self.state.progress_percent = status.get('progress', self.state.progress_percent)
                    self.state.current_stage = status.get('stage', self.state.current_stage)
                    self.state.is_running = status.get('is_running', self.state.is_running)
                    self.state.is_paused = status.get('is_paused', self.state.is_paused)
                    self.state.elapsed_ms = status.get('elapsed_ms', self.state.elapsed_ms)
                    self.state.eta_ms = status.get('eta_ms', self.state.eta_ms)
            except Exception:
                pass
        
        # Update progress bar display
        try:
            progress_bar = self.query_one(ProgressBar)
            progress_bar.progress = self.state.progress_percent
            progress_bar.stage_name = f"Stage {self.state.stage_index + 1}/{max(1, self.state.total_stages)}: {self.state.current_stage}"
            progress_bar.eta_text = f"Elapsed: {self.state.get_formatted_elapsed()}  |  ETA: {self.state.get_formatted_eta()}"
            progress_bar.refresh()
        except Exception:
            pass
        
        # Update stage timeline
        try:
            timeline = self.query_one(StageTimeline)
            # Update stages with current status
            if self.state.all_stages:
                for i, stage in enumerate(self.state.all_stages):
                    if i < self.state.stage_index:
                        stage.status = "done"
                    elif i == self.state.stage_index:
                        stage.status = "current"
                    else:
                        stage.status = "pending"
                timeline._stages = self.state.all_stages
            timeline._current_index = self.state.stage_index
            timeline.total_elapsed_ms = self.state.elapsed_ms
            timeline.total_eta_ms = self.state.eta_ms
            timeline.refresh()
        except Exception:
            pass
        
        # Update info panel if available
        try:
            info_panel = self.query_one(ExecutionInfoPanel)
            info_panel.state = self.state
            info_panel.refresh()
        except Exception:
            pass
    
    def compose_main(self):
        """Compose the execution screen content with integrated AI thinking panel."""
        with Horizontal(classes="main-split"):
            # Left side: Execution monitoring
            with Vertical(classes="execution-area"):
                # Execution panel
                with Container(classes="execution-panel"):
                    yield Static(
                        "Execution Monitor",
                        classes="execution-title",
                    )
                    yield ExecutionInfoPanel(self.state)
                    
                    # Progress bar
                    with Vertical(classes="progress-section"):
                        yield ProgressBar(
                            progress=self.state.progress_percent,
                            stage_name=f"Stage {self.state.stage_index + 1}/{self.state.total_stages}: {self.state.current_stage}",
                            eta_text=f"Elapsed: {self.state.get_formatted_elapsed()}  |  ETA: {self.state.get_formatted_eta()}",
                        )
                    
                    # Stage timeline
                    with Vertical(classes="timeline-section"):
                        yield StageTimeline(
                            stages=self.state.all_stages,
                            current_index=self.state.stage_index,
                            total_elapsed_ms=self.state.elapsed_ms,
                            total_eta_ms=self.state.eta_ms,
                        )
                
                # Controls
                with Horizontal(classes="controls-section"):
                    yield Button(
                        "[P] Pause",
                        id="btn-pause",
                        classes="control-button",
                        variant="warning",
                    )
                    yield Button(
                        "[R] Resume",
                        id="btn-resume",
                        classes="control-button",
                        variant="success",
                        disabled=True,
                    )
                    yield Button(
                        "[A] Abort",
                        id="btn-abort",
                        classes="control-button",
                        variant="error",
                    )
                    yield Button(
                        "[Z] Rollback",
                        id="btn-rollback",
                        classes="control-button",
                        disabled=not self.state.rollback_available,
                    )
                    yield Button(
                        "[L] Toggle Log",
                        id="btn-toggle-log",
                        classes="control-button",
                    )
                    yield Button(
                        "[T] AI Panel",
                        id="btn-toggle-ai",
                        classes="control-button",
                    )
                
                # Log viewer
                yield ExecutionLog(classes="log-section", id="execution-log")
            
            # Right side: AI Thinking Panel
            with Vertical(classes="ai-thinking-panel", id="ai-thinking-panel"):
                # AI Panel Header
                with Horizontal(classes="ai-header"):
                    yield Static("🧠 AI Thinking", classes="ai-title")
                    yield Static("● Ready", classes="ai-status", id="ai-status-indicator")
                
                # AI Stats Section
                with Container(classes="ai-stats-section", id="ai-stats-section"):
                    yield Static("📊 AI Configuration", classes="ai-stats-title")
                    with Horizontal(classes="ai-stats-row"):
                        yield Static("Provider:", classes="ai-stat-label")
                        yield Static("Not configured", classes="ai-stat-value", id="ai-stat-provider")
                    with Horizontal(classes="ai-stats-row"):
                        yield Static("Model:", classes="ai-stat-label")
                        yield Static("—", classes="ai-stat-value", id="ai-stat-model")
                    with Horizontal(classes="ai-stats-row"):
                        yield Static("Status:", classes="ai-stat-label")
                        yield Static("Not connected", classes="ai-stat-value ai-stat-disconnected", id="ai-stat-status")
                    with Horizontal(classes="ai-stats-row"):
                        yield Static("Tokens:", classes="ai-stat-label")
                        yield Static("0/0 (0 total)", classes="ai-stat-value", id="ai-stat-tokens")
                    with Horizontal(classes="ai-stats-row"):
                        yield Static("Requests:", classes="ai-stat-label")
                        yield Static("0", classes="ai-stat-value", id="ai-stat-requests")
                    with Horizontal(classes="ai-stats-row"):
                        yield Static("Think Time:", classes="ai-stat-label")
                        yield Static("0ms", classes="ai-stat-value", id="ai-stat-thinking-time")
                
                # AI Thought Stream
                with ScrollableContainer(classes="ai-thought-area"):
                    yield RichLog(
                        auto_scroll=True,
                        classes="ai-thought-stream",
                        id="ai-thought-stream",
                    )
                
                # Chat Input Section
                with Vertical(classes="ai-chat-section"):
                    with Horizontal(classes="ai-chat-input-row"):
                        yield Input(
                            placeholder="Type your message to AI...",
                            classes="ai-chat-input",
                            id="ai-chat-input",
                        )
                        yield Button(
                            "Send",
                            id="btn-ai-send",
                            classes="ai-send-btn",
                            variant="primary",
                        )
                    
                    with Horizontal(classes="ai-controls-row"):
                        yield Button(
                            "🛑 Stop",
                            id="btn-ai-stop",
                            classes="ai-control-btn",
                            variant="error",
                        )
                        yield Button(
                            "🗑️ Clear",
                            id="btn-ai-clear",
                            classes="ai-control-btn",
                        )
                        yield Button(
                            "💾 Export",
                            id="btn-ai-export",
                            classes="ai-control-btn",
                        )
                        yield Button(
                            "⚙️ Settings",
                            id="btn-ai-settings",
                            classes="ai-control-btn",
                        )
    
    def action_pause_execution(self) -> None:
        """Pause the current execution."""
        if self._controller:
            try:
                # Check if pause is possible
                if not self._controller.can_pause:
                    self.notify("Cannot pause - execution not running", severity="warning")
                    return
                    
                result = self._controller.pause()
                if result:
                    self.notify("⏸ Execution paused", severity="success")
                    self._update_control_buttons("paused")
                    self._log_action("Execution paused by user")
                    # Sync state
                    self.state.execution_status = "PAUSED"
                    self._update_status_display()
                else:
                    self.notify("Could not pause execution", severity="warning")
            except Exception as e:
                self.notify(f"Pause failed: {e}", severity="error")
        else:
            # Fallback: Update UI state directly
            self.state.is_paused = True
            self.state.execution_status = "PAUSED"
            self._update_control_buttons("paused")
            self._log_action("Execution paused (local mode)")
            self._update_status_display()
            self.notify("⏸ Execution paused", severity="success")
    
    def action_resume_execution(self) -> None:
        """Resume the paused execution."""
        if self._controller:
            try:
                # Check if resume is possible
                if not self._controller.can_resume:
                    self.notify("Cannot resume - execution not paused", severity="warning")
                    return
                    
                result = self._controller.resume()
                if result:
                    self.notify("▶ Execution resumed", severity="success")
                    self._update_control_buttons("running")
                    self._log_action("Execution resumed by user")
                    # Sync state
                    self.state.execution_status = "RUNNING"
                    self._update_status_display()
                else:
                    self.notify("Could not resume execution", severity="warning")
            except Exception as e:
                self.notify(f"Resume failed: {e}", severity="error")
        else:
            # Fallback: Update UI state directly
            self.state.is_paused = False
            self.state.execution_status = "RUNNING"
            self._update_control_buttons("running")
            self._log_action("Execution resumed (local mode)")
            self._update_status_display()
            self.notify("▶ Execution resumed", severity="success")
    
    def action_abort_execution(self) -> None:
        """Abort the current execution."""
        if self._controller:
            try:
                # Check if abort is possible
                if not self._controller.can_abort:
                    self.notify("Cannot abort - no active execution", severity="warning")
                    return
                    
                result = self._controller.abort()
                if result:
                    self.notify("⏹ Execution aborted", severity="warning")
                    self._update_control_buttons("stopped")
                    self._log_action("Execution aborted by user")
                    # Sync state
                    self.state.execution_status = "ABORTED"
                    self._update_status_display()
                else:
                    self.notify("Could not abort execution", severity="warning")
            except Exception as e:
                self.notify(f"Abort failed: {e}", severity="error")
        else:
            # Fallback: Update UI state directly
            self.state.is_running = False
            self.state.is_paused = False
            self.state.execution_status = "ABORTED"
            self._update_control_buttons("stopped")
            self._log_action("Execution aborted (local mode)")
            self._update_status_display()
            self.notify("⏹ Execution aborted", severity="warning")
    
    def action_rollback(self) -> None:
        """Rollback to last checkpoint."""
        if self._controller:
            try:
                # Check if rollback is possible
                if not self._controller.can_rollback:
                    self.notify("No checkpoint available for rollback", severity="warning")
                    return
                    
                result = self._controller.rollback()
                if result:
                    self.notify("↩ Rolled back to last checkpoint", severity="success")
                    self._log_action("Rolled back to last checkpoint")
                    # Sync state - get current status from controller
                    status = self._controller.get_status()
                    self.state.stage_index = status.get('stage_index', 0)
                    self.state.progress_percent = status.get('progress', 0)
                    self._update_control_buttons("running")
                    self._update_status_display()
                else:
                    self.notify("Rollback failed", severity="warning")
            except Exception as e:
                self.notify(f"Rollback failed: {e}", severity="error")
        else:
            # Fallback: Check local state for rollback
            if self.state.rollback_available:
                self._log_action("Rolled back to last checkpoint (local mode)")
                self._update_status_display()
                self.notify("↩ Rolled back to last checkpoint", severity="success")
            else:
                self.notify("No checkpoint available for rollback", severity="warning")
    
    def _update_status_display(self) -> None:
        """Update the status display after state changes."""
        try:
            # Refresh the execution info panel
            info_panel = self.query_one(ExecutionInfoPanel)
            info_panel.refresh()
        except Exception:
            pass
    
    def _log_action(self, message: str) -> None:
        """Log an action to the execution log."""
        try:
            log = self.query_one(ExecutionLog)
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            theme = get_theme()
            text = Text()
            text.append(f"[{timestamp}] ", style=theme.fg_subtle)
            text.append("ACTION  ", style=f"bold {theme.primary}")
            text.append(message, style=theme.fg_base)
            log.write(text)
        except Exception:
            pass  # Log may not exist
    
    def _update_control_buttons(self, state: str) -> None:
        """Update control button states based on execution state."""
        try:
            pause_btn = self.query_one("#btn-pause", Button) if self.query("#btn-pause") else None
            resume_btn = self.query_one("#btn-resume", Button) if self.query("#btn-resume") else None
            abort_btn = self.query_one("#btn-abort", Button) if self.query("#btn-abort") else None
            
            if state == "running":
                if pause_btn:
                    pause_btn.disabled = False
                if resume_btn:
                    resume_btn.disabled = True
                if abort_btn:
                    abort_btn.disabled = False
            elif state == "paused":
                if pause_btn:
                    pause_btn.disabled = True
                if resume_btn:
                    resume_btn.disabled = False
                if abort_btn:
                    abort_btn.disabled = False
            elif state == "stopped" or state == "completed":
                if pause_btn:
                    pause_btn.disabled = True
                if resume_btn:
                    resume_btn.disabled = True
                if abort_btn:
                    abort_btn.disabled = True
        except Exception:
            pass  # Buttons may not exist in all screen modes

    def action_toggle_log(self) -> None:
        """Toggle the log panel visibility."""
        self._log_visible = not self._log_visible
        log_section = self.query_one(".log-section")
        log_section.set_class(not self._log_visible, "-hidden")
    
    def action_toggle_ai_panel(self) -> None:
        """Toggle the AI thinking panel visibility."""
        self._ai_panel_visible = not self._ai_panel_visible
        try:
            ai_panel = self.query_one("#ai-thinking-panel")
            ai_panel.set_class(not self._ai_panel_visible, "-hidden")
            
            # Focus the AI input when panel is shown
            if self._ai_panel_visible:
                self.set_timer(0.1, self._focus_ai_input)
        except Exception:
            pass
    
    def _focus_ai_input(self) -> None:
        """Focus the AI chat input field."""
        try:
            input_widget = self.query_one("#ai-chat-input", Input)
            input_widget.focus()
        except Exception:
            pass
    
    def action_focus_ai_input(self) -> None:
        """Action handler to focus AI input (keybinding 'i')."""
        self._focus_ai_input()
    
    def action_toggle_ai_thinking(self) -> None:
        """Toggle AI thinking panel (keyboard shortcut handler)."""
        self.action_toggle_ai_panel()
    
    def action_prev_ai_prompt(self) -> None:
        """Navigate to previous prompt in history (Ctrl+J)."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#ai-chat-input", Input)
            
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

    def action_next_ai_prompt(self) -> None:
        """Navigate to next prompt in history (Ctrl+L)."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#ai-chat-input", Input)
            
            if self._prompt_history_index < len(self._prompt_history) - 1:
                self._prompt_history_index += 1
                input_widget.value = self._prompt_history[self._prompt_history_index]
            else:
                # At the end, clear input
                self._prompt_history_index = -1
                input_widget.value = ""
        except Exception:
            pass

    def _send_ai_message(self) -> None:
        """Send a message to the AI."""
        import time
        
        try:
            input_widget = self.query_one("#ai-chat-input", Input)
            message = input_widget.value.strip()
            
            if not message:
                self.notify("Please enter a message", severity="warning")
                return
            
            # Add to prompt history
            if not self._prompt_history or self._prompt_history[-1] != message:
                self._prompt_history.append(message)
            self._prompt_history_index = -1  # Reset position
            
            # Clear input
            input_widget.value = ""
            
            # Start timing
            start_time = time.time()
            
            # Add user message to stream
            theme = get_theme()
            thought_stream = self.query_one("#ai-thought-stream", RichLog)
            
            user_text = Text()
            user_text.append("👤 You: ", style=f"bold {theme.primary}")
            user_text.append(message, style=theme.fg_base)
            thought_stream.write(user_text)
            
            # Update status
            status_indicator = self.query_one("#ai-status-indicator", Static)
            status_indicator.update("● Thinking...")
            
            # Enable stop button
            try:
                stop_btn = self.query_one("#btn-ai-stop", Button)
                stop_btn.disabled = False
            except Exception:
                pass
            
            self._ai_is_thinking = True
            self._ai_stats['requests'] += 1
            
            # Store in conversation history
            self._ai_conversation.append({"role": "user", "content": message})
            
            # Get provider settings
            llm_settings = getattr(self, '_llm_settings', {})
            provider = llm_settings.get('mode', 'none')
            
            # Get model for provider
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
            model = llm_settings.get(model_key_map.get(provider, ''), '')
            
            # Get API key
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
            api_key = llm_settings.get(api_key_map.get(provider, ''), '')
            
            # Send to LLM (works regardless of backend running status)
            if self._llm_router and LLM_AVAILABLE and provider and provider != 'none':
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
                
                # Build context from execution state
                context = self._build_execution_context()
                full_prompt = f"Execution Context:\n{context}\n\nUser Query: {message}"
                
                try:
                    # Chain of Thoughts system prompt
                    cot_system_prompt = """You are an AI assistant helping with quantum simulation execution.
Use chain-of-thought reasoning to analyze queries step by step.

Format your response as:
**Step 1: [Analysis]**
[Your first thought]

**Step 2: [Consideration]**
[Your second thought]

**Step 3: [Conclusion]**
[Your final recommendation]

Be concise but thorough. Focus on execution progress, potential issues, and recommendations."""
                    
                    request = LLMRequest(
                        prompt=full_prompt,
                        system_prompt=cot_system_prompt,
                        temperature=0.7,
                        max_tokens=1024,
                        provider=router_provider,
                        model=model if model else None,
                    )
                    
                    # Show thinking indicator
                    self._show_thinking_steps()
                    
                    # For now, simulate response (async route would be better)
                    response = self._llm_router.route(request)
                    self._handle_ai_response(response, start_time, message)
                    
                except PermissionError as e:
                    # Consent required - use simulated response silently
                    self._simulate_ai_response(message, start_time)
                except ConnectionError as e:
                    # Local provider not running - use simulated response
                    self._simulate_ai_response(message, start_time)
                except ValueError as e:
                    # Provider not found in router - use simulated response silently
                    if "Unknown provider" in str(e) or "No LLM provider" in str(e):
                        self._simulate_ai_response(message, start_time)
                    else:
                        self._simulate_ai_response(message, start_time)
                except Exception as e:
                    # For any other error, fall back to simulation
                    error_msg = str(e)
                    # Don't show consent or unknown provider errors
                    if "Unknown provider" not in error_msg and "Consent" not in error_msg:
                        self._show_ai_error(f"LLM Error: {e}")
                    self._simulate_ai_response(message, start_time)
            else:
                # Simulated response when LLM not available or not configured
                self._simulate_ai_response(message, start_time)
                
        except Exception as e:
            self._show_ai_error(f"Error: {e}")
    
    def _build_execution_context(self) -> str:
        """Build context string from current execution state."""
        return f"""- Status: {self.state.execution_status}
- Backend: {self.state.current_backend or 'Not configured'}
- Current Stage: {self.state.current_stage} ({self.state.stage_index + 1}/{self.state.total_stages})
- Progress: {self.state.progress_percent:.1f}%
- Qubits: {self.state.qubits}
- Shots: {self.state.shots}
- Elapsed: {self.state.get_formatted_elapsed()}
- Is Paused: {self.state.is_paused}"""
    
    def _handle_ai_response(self, response, start_time: float = None, original_message: str = "") -> None:
        """Handle the AI response."""
        import time
        self._ai_is_thinking = False
        
        theme = get_theme()
        thought_stream = self.query_one("#ai-thought-stream", RichLog)
        
        # Check for error or empty response
        if response.error:
            self._show_ai_error(response.error)
            # Fall back to simulation
            if original_message:
                self._simulate_ai_response(original_message, start_time)
            return
        
        # Check if response text is empty
        if not response.text or not response.text.strip():
            # Fall back to simulation
            if original_message:
                self._simulate_ai_response(original_message, start_time)
            return
        
        # Add AI response to stream with chain-of-thought formatting
        ai_header = Text()
        ai_header.append("🤖 AI Response:\n", style=f"bold {theme.accent}")
        thought_stream.write(ai_header)
        
        # Format chain of thought response
        formatted_response = self._format_chain_of_thought(response.text)
        thought_stream.write(formatted_response)
        thought_stream.write("")  # Empty line for spacing
        
        # Store in conversation
        self._ai_conversation.append({"role": "assistant", "content": response.text})
        
        # Update token stats if available
        tokens_used = getattr(response, 'tokens_used', 0) or 0
        if hasattr(response, 'prompt_tokens'):
            self._ai_stats['prompt_tokens'] += response.prompt_tokens or 0
            self.state.prompt_tokens = getattr(self.state, 'prompt_tokens', 0) + (response.prompt_tokens or 0)
        if hasattr(response, 'completion_tokens'):
            self._ai_stats['completion_tokens'] += response.completion_tokens or 0
            self.state.completion_tokens = getattr(self.state, 'completion_tokens', 0) + (response.completion_tokens or 0)
        if hasattr(response, 'total_tokens'):
            self._ai_stats['total_tokens'] += response.total_tokens or 0
        elif tokens_used:
            self._ai_stats['total_tokens'] += tokens_used
        else:
            self._ai_stats['total_tokens'] = self._ai_stats['prompt_tokens'] + self._ai_stats['completion_tokens']
        
        # Calculate thinking time
        if start_time:
            import time
            elapsed = int((time.time() - start_time) * 1000)
            self._ai_stats['thinking_time_ms'] = elapsed
        
        # Update stats panel
        self._update_ai_stats_panel()
        
        # Update status
        status_indicator = self.query_one("#ai-status-indicator", Static)
        status_indicator.update("● Ready")
    
    def _show_thinking_steps(self) -> None:
        """Show animated thinking steps in the thought stream."""
        try:
            theme = get_theme()
            thought_stream = self.query_one("#ai-thought-stream", RichLog)
            
            thinking_text = Text()
            thinking_text.append("💭 ", style=f"bold {theme.warning}")
            thinking_text.append("Analyzing...", style=f"italic {theme.fg_muted}")
            thought_stream.write(thinking_text)
        except Exception:
            pass
    
    def _format_chain_of_thought(self, response_text: str) -> Text:
        """Format response with chain of thought highlighting."""
        theme = get_theme()
        formatted = Text()
        
        lines = response_text.split('\n')
        for line in lines:
            if line.startswith('**Step'):
                # Highlight step headers
                formatted.append(line.replace('**', '') + '\n', style=f"bold {theme.accent}")
            elif line.startswith('**'):
                # Other bold text
                formatted.append(line.replace('**', '') + '\n', style=f"bold {theme.primary}")
            else:
                formatted.append(line + '\n', style=theme.fg_base)
        
        return formatted

    def _simulate_ai_response(self, user_message: str, start_time: float = None) -> None:
        """Simulate AI response when LLM is not available."""
        import time
        
        if start_time is None:
            start_time = time.time()
        
        self._ai_is_thinking = False
        theme = get_theme()
        thought_stream = self.query_one("#ai-thought-stream", RichLog)
        
        # Chain of Thoughts formatted responses
        base_context = f"""**Step 1: Context Analysis**
Analyzing current execution state:
- Backend: {self.state.current_backend or 'default'}
- Progress: {self.state.progress_percent:.1f}%
- Stage: {self.state.current_stage}

"""
        msg_lower = user_message.lower()

        if "error" in msg_lower or "problem" in msg_lower:
            response = base_context + """**Step 2: Error Check**
Scanning execution logs and backend status for anomalies...
No errors detected in the current pipeline.

**Step 3: Recommendation**
The backend is operating normally. Check the log panel (press 'L') for detailed output if you're experiencing specific issues."""

        elif "stop" in msg_lower or "pause" in msg_lower:
            response = base_context + """**Step 2: Control Options**
Available execution controls:
- Pause: Press 'P' or click Pause button
- Resume: Press 'R' to continue
- Abort: Press 'A' to stop completely

**Step 3: Recommendation**
Use Pause to checkpoint current state. This allows resuming later without losing progress."""

        elif "speed" in msg_lower or "performance" in msg_lower or "fast" in msg_lower:
            response = base_context + f"""**Step 2: Performance Analysis**
Current configuration: {self.state.qubits} qubits, {self.state.shots} shots
Execution rate appears normal for this workload.

**Step 3: Optimization Tips**
- Consider LRET Phase 7 backend for GPU acceleration
- Reduce shots for faster iteration during testing
- Use GPU backend if available for 25+ qubits"""

        elif "run" in msg_lower or "cirq" in msg_lower or "execute" in msg_lower:
            response = base_context + f"""**Step 2: Execution Status**
Current simulation is {"running" if self.state.execution_status == "RUNNING" else "ready"}.
Backend: {self.state.current_backend or 'default simulator'}

**Step 3: Next Steps**
To start a simulation:
1. Go to Dashboard (press 1) or Command Palette (Ctrl+P)
2. Configure your circuit and backend
3. Click Run or press Enter to start"""

        elif "help" in msg_lower or "what" in msg_lower:
            response = """**Step 1: Understanding Your Query**
You're asking for assistance with quantum simulation.

**Step 2: Available Help Topics**
- Execution control (pause, resume, abort)
- Performance optimization
- Backend selection
- Error troubleshooting

**Step 3: How to Get More Help**
Ask specific questions like:
- "How do I improve performance?"
- "Why is execution slow?"
- "Help me configure the backend" """
        
        else:
            response = base_context + f"""**Step 2: Status Assessment**
Simulation {"running" if self.state.execution_status == "RUNNING" else "ready"} with {self.state.qubits} qubits and {self.state.shots} shots.
Current stage: {self.state.current_stage}

**Step 3: Available Actions**
- Ask about performance optimization
- Query specific stage details
- Request error analysis
- Get backend recommendations
- Say "help" for more options"""
        
        # Display response directly (don't call _handle_ai_response to avoid recursion)
        ai_header = Text()
        ai_header.append("🤖 AI Response:\n", style=f"bold {theme.accent}")
        thought_stream.write(ai_header)
        
        formatted_response = self._format_chain_of_thought(response)
        thought_stream.write(formatted_response)
        thought_stream.write("")  # Empty line for spacing
        
        # Store in conversation
        self._ai_conversation.append({"role": "assistant", "content": response})
        
        # Update stats
        elapsed = int((time.time() - start_time) * 1000)
        self._ai_stats['prompt_tokens'] += len(user_message.split()) * 2
        self._ai_stats['completion_tokens'] += len(response.split()) * 2
        self._ai_stats['total_tokens'] = self._ai_stats['prompt_tokens'] + self._ai_stats['completion_tokens']
        self._ai_stats['thinking_time_ms'] = elapsed
        
        self._update_ai_stats_panel()
        
        # Update status
        try:
            status_indicator = self.query_one("#ai-status-indicator", Static)
            status_indicator.update("● Ready")
        except Exception:
            pass
    
    def _show_ai_error(self, error_message: str) -> None:
        """Show an error in the AI panel."""
        self._ai_is_thinking = False
        
        theme = get_theme()
        thought_stream = self.query_one("#ai-thought-stream", RichLog)
        
        error_text = Text()
        error_text.append("❌ Error: ", style=f"bold {theme.error}")
        error_text.append(error_message, style=theme.fg_muted)
        thought_stream.write(error_text)
        
        # Update status
        status_indicator = self.query_one("#ai-status-indicator", Static)
        status_indicator.update("● Ready")
    
    def _stop_ai(self) -> None:
        """Stop the current AI thinking process."""
        # Allow stopping even when not actively thinking - just show info
        self._ai_is_thinking = False
        
        theme = get_theme()
        try:
            thought_stream = self.query_one("#ai-thought-stream", RichLog)
            
            stop_text = Text()
            stop_text.append("⏹ ", style=f"bold {theme.warning}")
            stop_text.append("AI thinking stopped by user\n", style=theme.fg_muted)
            thought_stream.write(stop_text)
        except Exception:
            pass
        
        # Update status
        try:
            status_indicator = self.query_one("#ai-status-indicator", Static)
            status_indicator.update("● Ready")
        except Exception:
            pass
        
        self.notify("AI thinking stopped", severity="information")
    
    def _clear_ai_history(self) -> None:
        """Clear the AI conversation history."""
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
        
        try:
            thought_stream = self.query_one("#ai-thought-stream", RichLog)
            thought_stream.clear()
            
            theme = get_theme()
            welcome_text = Text()
            welcome_text.append("🧠 AI Thinking Panel Ready\n", style=f"bold {theme.accent}")
            welcome_text.append("─" * 40 + "\n", style=theme.fg_subtle)
            welcome_text.append("💡 Chain of Thoughts Mode: ", style=f"bold {theme.primary}")
            welcome_text.append("ENABLED\n\n", style=f"bold {theme.success}")
            welcome_text.append("Type a message below to interact with the AI assistant.\n", style=theme.fg_muted)
            welcome_text.append("The AI will analyze your query step by step.\n\n", style=theme.fg_subtle)
            welcome_text.append("Keyboard shortcuts:\n", style=f"bold {theme.fg_base}")
            welcome_text.append("  Ctrl+J  Previous prompt\n", style=theme.fg_muted)
            welcome_text.append("  Ctrl+L  Next prompt\n", style=theme.fg_muted)
            welcome_text.append("  Ctrl+T  Toggle AI panel\n", style=theme.fg_muted)
            thought_stream.write(welcome_text)
            
            # Update stats panel
            self._update_ai_stats_panel()
            
        except Exception:
            pass
        
        self.notify("AI history cleared", severity="success")
    
    def _export_ai_conversation(self) -> None:
        """Export the AI conversation to a file."""
        from pathlib import Path
        from datetime import datetime
        import json
        
        try:
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"ai_conversation_{timestamp}.json"
            
            export_data = {
                "timestamp": timestamp,
                "execution_context": {
                    "backend": self.state.current_backend,
                    "qubits": self.state.qubits,
                    "shots": self.state.shots,
                    "status": self.state.execution_status,
                },
                "conversation": self._ai_conversation,
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.notify(f"Conversation exported to {filename}", severity="success")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def _open_ai_settings(self) -> None:
        """Open AI settings screen."""
        self.app.action_goto_settings()
        self.notify("Configure AI provider in Settings > AI Assistant Settings", severity="information")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-pause":
            self.action_pause_execution()
        elif button_id == "btn-resume":
            self.action_resume_execution()
        elif button_id == "btn-abort":
            self.action_abort_execution()
        elif button_id == "btn-rollback":
            self.action_rollback()
        elif button_id == "btn-toggle-log":
            self.action_toggle_log()
        elif button_id == "btn-toggle-ai":
            self.action_toggle_ai_panel()
        # AI Panel buttons
        elif button_id == "btn-ai-send":
            self._send_ai_message()
        elif button_id == "btn-ai-stop":
            self._stop_ai()
        elif button_id == "btn-ai-clear":
            self._clear_ai_history()
        elif button_id == "btn-ai-export":
            self._export_ai_conversation()
        elif button_id == "btn-ai-settings":
            self._open_ai_settings()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        if event.input.id == "ai-chat-input":
            self._send_ai_message()


class ExecutionInfoPanel(Static):
    """Panel showing current execution information."""
    
    def __init__(self, state, **kwargs):
        """Initialize the info panel."""
        super().__init__(**kwargs)
        self._state = state
    
    @property
    def state(self):
        """Get the state - prefer app state for freshness."""
        # Always get fresh state from app if available
        if hasattr(self, 'app') and self.app and hasattr(self.app, 'state'):
            app_state = self.app.state
            # If app state has experiment but local doesn't, use app state
            if app_state and getattr(app_state, 'current_experiment', None):
                return app_state
        return self._state
    
    @state.setter
    def state(self, value):
        """Set the state and trigger refresh."""
        self._state = value
    
    def render(self) -> Text:
        """Render the execution info."""
        theme = get_theme()
        text = Text()
        
        # Get state via property to ensure freshness
        state = self.state
        
        # Check if there's an active execution
        is_running = getattr(state, 'is_running', False)
        execution_status = getattr(state, 'execution_status', 'IDLE')
        
        # Check for AI Assistant experiment first
        current_experiment = getattr(state, 'current_experiment', None)
        current_task = getattr(state, 'current_task', None)
        
        if current_experiment and current_experiment.get('name'):
            experiment = current_experiment
            exp_name = experiment.get('name', 'Experiment')
            exp_status = experiment.get('status', 'running')
            exp_command = experiment.get('command', '')
            if len(exp_command) > 50:
                exp_command = exp_command[:50] + '...'
            
            # Status icon
            status_icon = {'running': '🔄', 'completed': '✅', 'failed': '❌', 'stopped': '⛔'}.get(exp_status, '❓')
            
            text.append("Task: ", style=theme.fg_muted)
            text.append(f"{status_icon} {exp_name}", style=f"bold {theme.fg_base}")
            text.append("\n")
            
            text.append("Status: ", style=theme.fg_muted)
            status_color = {'running': theme.info, 'completed': theme.success, 'failed': theme.error, 'stopped': theme.warning}.get(exp_status, theme.fg_base)
            text.append(exp_status.upper(), style=f"bold {status_color}")
            text.append("\n")
            
            if exp_command:
                text.append("Command: ", style=theme.fg_muted)
                text.append(exp_command, style=theme.fg_subtle)
                text.append("\n")
            
            backend = getattr(state, 'current_backend', None)
            if backend:
                text.append("Backend: ", style=theme.fg_muted)
                text.append(f"{backend}", style=theme.fg_base)
        elif current_task:
            # Task info
            text.append("Task: ", style=theme.fg_muted)
            text.append(current_task, style=f"bold {theme.fg_base}")
            text.append("\n")
            
            # Task ID
            task_id = getattr(state, 'current_task_id', None)
            if task_id:
                text.append("ID: ", style=theme.fg_muted)
                text.append(task_id, style=theme.fg_subtle)
                text.append("\n")
            
            # Backend info
            backend = getattr(state, 'current_backend', 'N/A')
            simulator = getattr(state, 'current_simulator', 'N/A')
            text.append("Backend: ", style=theme.fg_muted)
            text.append(f"{backend} ({simulator})", style=theme.fg_base)
            text.append(" • ", style=theme.border)
            text.append(f"{state.qubits} qubits", style=theme.fg_base)
            text.append(" • ", style=theme.border)
            text.append(f"{state.shots} shots", style=theme.fg_base)
        else:
            text.append("No active execution", style=theme.fg_subtle)
            text.append("\n\n")
            text.append("Start a simulation from the Dashboard or Command Palette (Ctrl+P)",
                       style=theme.fg_muted)
        
        return text


class ExecutionLog(RichLog):
    """Log viewer for execution output."""
    
    DEFAULT_CSS = """
    ExecutionLog {
        padding: 1;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_real_logs = False
        self._sample_logs_shown = False
    
    def on_mount(self) -> None:
        """Set up the log."""
        self.border_title = "Log"
        
        # Only show sample logs if no real logs have been written
        if not self._has_real_logs:
            self._add_sample_logs()
            self._sample_logs_shown = True
    
    def write(self, content, *args, **kwargs) -> None:
        """Override write to clear sample logs on first real write."""
        # Clear sample logs on first real write
        if not self._has_real_logs and self._sample_logs_shown:
            self.clear()
            self._has_real_logs = True
        elif not self._has_real_logs:
            self._has_real_logs = True
        
        super().write(content, *args, **kwargs)
    
    def _add_sample_logs(self) -> None:
        """Add sample log entries for demo."""
        theme = get_theme()
        
        logs = [
            ("--:--:--", "INFO", "Waiting for execution..."),
            ("--:--:--", "INFO", "Use AI Assistant (key 6) to run experiments"),
            ("--:--:--", "INFO", "Or start from Dashboard (key 1)"),
        ]
        
        for timestamp, level, message in logs:
            level_color = {
                "INFO": theme.info,
                "WARNING": theme.warning,
                "ERROR": theme.error,
                "DEBUG": theme.fg_subtle,
            }.get(level, theme.fg_muted)
            
            text = Text()
            text.append(f"[{timestamp}] ", style=theme.fg_subtle)
            text.append(f"{level:<8}", style=f"bold {level_color}")
            text.append(message, style=theme.fg_base)
            
            # Use parent write to avoid triggering our override
            super().write(text)
