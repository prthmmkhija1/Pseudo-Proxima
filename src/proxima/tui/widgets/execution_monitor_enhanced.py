"""Phase 3 Enhanced Execution Monitor with Terminal Integration.

Extends the base ExecutionMonitor with:
- Integration with MultiTerminalMonitor
- Session management via SessionManager
- Cross-platform command normalization
- Terminal state machine tracking
- Event debouncing for performance
- Enhanced multi-terminal grid
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive, var
from textual.widgets import (
    Button,
    DataTable,
    Label,
    ProgressBar,
    RichLog,
    Rule,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widget import Widget
from rich.text import Text
from rich.style import Style
from rich.console import RenderableType
from rich.table import Table

from proxima.core.event_bus import (
    Event,
    EventBus,
    EventType,
    get_event_bus,
)
from proxima.core.process_streaming import ProcessInfo, ProcessState
from proxima.agent.multi_terminal import (
    MultiTerminalMonitor,
    TerminalInfo,
    TerminalState,
    TerminalEvent,
    TerminalEventType,
    SessionManager,
    AgentSession,
    CommandNormalizer,
    CommandQueue,
    CommandPriority,
    get_multi_terminal_monitor,
    get_session_manager,
    get_command_normalizer,
)
from proxima.agent.terminal_state_machine import (
    TerminalStateMachine,
    TerminalProcessState,
    TerminalStateEvent,
    ProcessMetrics,
    StateContext,
    get_terminal_state_machine,
)
from proxima.utils.logging import get_logger

logger = get_logger("tui.execution_monitor_enhanced")


# =============================================================================
# Phase 3: Enhanced Status Bar
# =============================================================================

class EnhancedStatusBar(Static):
    """Status bar with Phase 3 enhancements.
    
    Shows:
    - Process name and PID
    - Current state with color coding
    - Elapsed time
    - Output line count
    - Error indicator
    - Memory/CPU (if available)
    """
    
    DEFAULT_CSS = """
    EnhancedStatusBar {
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
        border-bottom: solid $primary;
    }
    
    EnhancedStatusBar .status-running {
        color: $success;
    }
    
    EnhancedStatusBar .status-failed {
        color: $error;
    }
    
    EnhancedStatusBar .status-completed {
        color: $primary;
    }
    
    EnhancedStatusBar .status-timeout {
        color: $warning;
    }
    """
    
    process_name: reactive[str] = reactive("No Process")
    process_state: reactive[str] = reactive("Idle")
    pid: reactive[Optional[int]] = reactive(None)
    elapsed_time: reactive[str] = reactive("--:--")
    line_count: reactive[int] = reactive(0)
    error_count: reactive[int] = reactive(0)
    
    def render(self) -> RenderableType:
        """Render the enhanced status bar."""
        # State styling
        state_styles = {
            "Running": ("bold green", "🟢"),
            "Starting": ("bold yellow", "🟡"),
            "Pending": ("dim", "⏳"),
            "Completed": ("bold blue", "✅"),
            "Failed": ("bold red", "❌"),
            "Cancelled": ("bold yellow", "🚫"),
            "Timeout": ("bold magenta", "⏰"),
            "Idle": ("dim", "⚪"),
        }
        
        style, icon = state_styles.get(self.process_state, ("", ""))
        state_text = Text(f"{icon} {self.process_state}", style=style)
        
        # Build status line
        parts = [
            ("📋 ", "bold"),
            (f"{self.process_name[:25]} ", "bold"),
        ]
        
        if self.pid:
            parts.append((f"[PID: {self.pid}] ", "dim"))
        
        parts.extend([
            ("| ", "dim"),
            state_text,
            (" | ", "dim"),
            ("⏱️ ", ""),
            (self.elapsed_time, ""),
            (" | ", "dim"),
            ("📝 ", ""),
            (f"{self.line_count} lines", ""),
        ])
        
        if self.error_count > 0:
            parts.extend([
                (" | ", "dim"),
                ("⚠️ ", ""),
                (f"{self.error_count} errors", "red"),
            ])
        
        return Text.assemble(*parts)


# =============================================================================
# Phase 3: Session Selector Widget
# =============================================================================

class SessionSelector(Static):
    """Widget for selecting and managing agent sessions."""
    
    DEFAULT_CSS = """
    SessionSelector {
        height: 4;
        padding: 0 1;
        background: $surface-darken-1;
        border-bottom: solid $primary-darken-2;
    }
    
    SessionSelector .session-list {
        layout: horizontal;
        height: auto;
        overflow-x: auto;
    }
    
    SessionSelector .session-btn {
        margin-right: 1;
        min-width: 15;
    }
    
    SessionSelector .session-btn.active {
        background: $accent;
    }
    """
    
    class SessionSelected(Message):
        """Emitted when a session is selected."""
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            super().__init__()
    
    class SessionCreated(Message):
        """Emitted when a new session is created."""
        def __init__(self, session: AgentSession) -> None:
            self.session = session
            super().__init__()
    
    def __init__(
        self,
        session_manager: SessionManager,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session_manager = session_manager
        self._active_session_id: Optional[str] = None
    
    def compose(self) -> ComposeResult:
        """Compose the session selector."""
        yield Label("Sessions: ", id="session-label")
        with Horizontal(classes="session-list", id="session-list"):
            pass  # Buttons added dynamically
        yield Button("+ New", id="btn-new-session", variant="success")
    
    def on_mount(self) -> None:
        """Initialize with existing sessions."""
        self._refresh_sessions()
    
    def _refresh_sessions(self) -> None:
        """Refresh the session list."""
        try:
            container = self.query_one("#session-list", Horizontal)
            container.remove_children()
            
            sessions = self._session_manager.get_all_sessions()
            for session_id, session in sessions.items():
                btn = Button(
                    session.name[:12],
                    id=f"session-{session_id}",
                    classes="session-btn",
                    variant="primary" if session_id == self._active_session_id else "default",
                )
                container.mount(btn)
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle session button clicks."""
        btn_id = event.button.id or ""
        
        if btn_id == "btn-new-session":
            session = self._session_manager.create_session()
            self._active_session_id = session.id
            self._refresh_sessions()
            self.post_message(self.SessionCreated(session))
        elif btn_id.startswith("session-"):
            session_id = btn_id[8:]
            self._active_session_id = session_id
            self._refresh_sessions()
            self.post_message(self.SessionSelected(session_id))
    
    def set_active_session(self, session_id: str) -> None:
        """Set the active session."""
        self._active_session_id = session_id
        self._refresh_sessions()


# =============================================================================
# Phase 3: Multi-Terminal Dashboard
# =============================================================================

class TerminalDashboard(Widget):
    """Dashboard showing all active terminals with state."""
    
    DEFAULT_CSS = """
    TerminalDashboard {
        height: auto;
        min-height: 5;
        max-height: 15;
        padding: 1;
        background: $surface-darken-2;
        border: solid $primary-darken-2;
    }
    
    TerminalDashboard DataTable {
        height: auto;
        max-height: 12;
    }
    """
    
    class TerminalSelected(Message):
        """Emitted when a terminal is selected in dashboard."""
        def __init__(self, terminal_id: str) -> None:
            self.terminal_id = terminal_id
            super().__init__()
    
    def __init__(
        self,
        monitor: Optional[MultiTerminalMonitor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._monitor = monitor or get_multi_terminal_monitor()
        self._table: Optional[DataTable] = None
    
    def compose(self) -> ComposeResult:
        """Compose the dashboard."""
        yield Label("📊 Active Terminals", id="dashboard-title")
        yield DataTable(id="terminal-table")
    
    def on_mount(self) -> None:
        """Initialize the table."""
        self._table = self.query_one("#terminal-table", DataTable)
        self._table.add_columns("ID", "Command", "State", "Time", "Lines")
        self._table.cursor_type = "row"
        
        # Listen for monitor events
        self._monitor.add_listener(self._on_terminal_event)
        
        # Initial refresh
        self._refresh_table()
    
    def on_unmount(self) -> None:
        """Clean up listeners."""
        self._monitor.remove_listener(self._on_terminal_event)
    
    def _on_terminal_event(self, event: TerminalEvent) -> None:
        """Handle terminal events."""
        self._refresh_table()
    
    def _refresh_table(self) -> None:
        """Refresh the terminal table."""
        if not self._table:
            return
        
        self._table.clear()
        
        terminals = self._monitor.get_all_terminals()
        for terminal_id, terminal in terminals.items():
            state_display = self._format_state(terminal.state)
            duration = self._format_duration(terminal.duration_ms)
            
            self._table.add_row(
                terminal_id[:10],
                terminal.command[:30],
                state_display,
                duration,
                str(terminal.output_buffer.line_count),
                key=terminal_id,
            )
    
    def _format_state(self, state: TerminalState) -> Text:
        """Format state with color."""
        styles = {
            TerminalState.PENDING: ("Pending", "dim"),
            TerminalState.STARTING: ("Starting", "yellow"),
            TerminalState.RUNNING: ("Running", "green"),
            TerminalState.COMPLETED: ("Done", "blue"),
            TerminalState.FAILED: ("Failed", "red"),
            TerminalState.TIMEOUT: ("Timeout", "magenta"),
            TerminalState.CANCELLED: ("Cancelled", "yellow"),
        }
        text, style = styles.get(state, (state.name, ""))
        return Text(text, style=style)
    
    def _format_duration(self, duration_ms: Optional[float]) -> str:
        """Format duration in human readable form."""
        if duration_ms is None:
            return "--:--"
        
        seconds = int(duration_ms / 1000)
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key:
            self.post_message(self.TerminalSelected(str(event.row_key.value)))


# =============================================================================
# Phase 3: Enhanced Output Panel
# =============================================================================

class EnhancedOutputPanel(RichLog):
    """Output panel with Phase 3 enhancements.
    
    Features:
    - 10,000 line circular buffer
    - Search capability
    - Line filtering
    - Timestamp toggling
    - Error highlighting
    - State machine integration
    """
    
    DEFAULT_CSS = """
    EnhancedOutputPanel {
        height: 1fr;
        border: solid $primary;
        background: $surface;
        scrollbar-gutter: stable;
    }
    
    EnhancedOutputPanel:focus {
        border: double $accent;
    }
    
    EnhancedOutputPanel.has-errors {
        border: solid $error;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+s", "toggle_scroll", "Toggle Scroll"),
        Binding("ctrl+f", "find", "Find"),
        Binding("ctrl+t", "toggle_timestamps", "Timestamps"),
        Binding("ctrl+e", "filter_errors", "Errors Only"),
    ]
    
    auto_scroll: reactive[bool] = reactive(True)
    show_timestamps: reactive[bool] = reactive(True)
    has_errors: reactive[bool] = reactive(False)
    errors_only: reactive[bool] = reactive(False)
    
    def __init__(
        self,
        terminal_id: str = "",
        max_lines: int = 10000,
        state_machine: Optional[TerminalStateMachine] = None,
        **kwargs,
    ):
        super().__init__(
            highlight=True,
            markup=True,
            wrap=False,
            max_lines=max_lines,
            **kwargs,
        )
        self.terminal_id = terminal_id
        self._state_machine = state_machine or get_terminal_state_machine()
        self._line_count = 0
        self._error_count = 0
        self._all_lines: List[Dict[str, Any]] = []  # Store for filtering
    
    def watch_has_errors(self, has_errors: bool) -> None:
        """Update styling when errors are detected."""
        self.set_class(has_errors, "has-errors")
    
    def add_output_line(
        self,
        line: str,
        is_error: bool = False,
        timestamp: bool = True,
    ) -> None:
        """Add a line to the output panel.
        
        Args:
            line: The line to add
            is_error: Whether this is an error line
            timestamp: Whether to include timestamp
        """
        self._line_count += 1
        
        if is_error:
            self._error_count += 1
            self.has_errors = True
            style = Style(color="red")
        else:
            style = None
        
        # Store line for filtering
        line_data = {
            "content": line,
            "is_error": is_error,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "line_number": self._line_count,
        }
        self._all_lines.append(line_data)
        
        # Keep buffer size manageable
        if len(self._all_lines) > 10000:
            self._all_lines = self._all_lines[-10000:]
        
        # Apply filter
        if self.errors_only and not is_error:
            return
        
        # Format line
        if self.show_timestamps:
            ts = line_data["timestamp"]
            prefix = Text(f"[{ts}] ", style="dim")
            text = Text.assemble(prefix, (line, style) if style else line)
        else:
            text = Text(line, style=style) if style else line
        
        self.write(text)
        
        if self.auto_scroll:
            self.scroll_end(animate=False)
        
        # Update state machine metrics
        if self.terminal_id:
            self._state_machine.record_output(
                self.terminal_id,
                line,
                is_error=is_error,
            )
    
    def action_clear(self) -> None:
        """Clear the output panel."""
        self.clear()
        self._line_count = 0
        self._error_count = 0
        self._all_lines = []
        self.has_errors = False
    
    def action_toggle_scroll(self) -> None:
        """Toggle auto-scroll."""
        self.auto_scroll = not self.auto_scroll
        self.notify(f"Auto-scroll: {'On' if self.auto_scroll else 'Off'}")
    
    def action_toggle_timestamps(self) -> None:
        """Toggle timestamp display."""
        self.show_timestamps = not self.show_timestamps
        self.notify(f"Timestamps: {'On' if self.show_timestamps else 'Off'}")
        self._rerender()
    
    def action_filter_errors(self) -> None:
        """Toggle error-only filter."""
        self.errors_only = not self.errors_only
        self.notify(f"Errors only: {'On' if self.errors_only else 'Off'}")
        self._rerender()
    
    def _rerender(self) -> None:
        """Re-render all lines with current settings."""
        self.clear()
        
        for line_data in self._all_lines:
            if self.errors_only and not line_data["is_error"]:
                continue
            
            style = Style(color="red") if line_data["is_error"] else None
            
            if self.show_timestamps:
                prefix = Text(f"[{line_data['timestamp']}] ", style="dim")
                text = Text.assemble(
                    prefix,
                    (line_data["content"], style) if style else line_data["content"],
                )
            else:
                content = line_data["content"]
                text = Text(content, style=style) if style else content
            
            self.write(text)
    
    def search(self, pattern: str, case_sensitive: bool = False) -> List[int]:
        """Search output for pattern.
        
        Args:
            pattern: Search pattern
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matching line numbers
        """
        if not case_sensitive:
            pattern = pattern.lower()
        
        matches = []
        for line_data in self._all_lines:
            content = line_data["content"]
            if not case_sensitive:
                content = content.lower()
            
            if pattern in content:
                matches.append(line_data["line_number"])
        
        return matches
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get output statistics."""
        return {
            "line_count": self._line_count,
            "error_count": self._error_count,
        }


# =============================================================================
# Phase 3: Enhanced Terminal View
# =============================================================================

class EnhancedTerminalView(Container):
    """Enhanced single terminal view with Phase 3 features."""
    
    DEFAULT_CSS = """
    EnhancedTerminalView {
        height: 1fr;
        width: 1fr;
        border: solid $primary;
    }
    
    EnhancedTerminalView:focus-within {
        border: double $accent;
    }
    
    EnhancedTerminalView.selected {
        border: double $success;
    }
    
    EnhancedTerminalView.has-errors {
        border: solid $error;
    }
    """
    
    class Selected(Message):
        """Emitted when this terminal is selected."""
        def __init__(self, terminal_id: str) -> None:
            self.terminal_id = terminal_id
            super().__init__()
    
    def __init__(
        self,
        terminal_info: TerminalInfo,
        session: Optional[AgentSession] = None,
        show_status: bool = True,
        show_progress: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.terminal_info = terminal_info
        self.session = session
        self.show_status = show_status
        self.show_progress = show_progress
        self._monitor = get_multi_terminal_monitor()
        self._state_machine = get_terminal_state_machine()
    
    def compose(self) -> ComposeResult:
        """Compose the terminal view."""
        if self.show_status:
            yield EnhancedStatusBar(id="status")
        if self.show_progress:
            yield ProgressIndicator(id="progress")
        yield EnhancedOutputPanel(
            terminal_id=self.terminal_info.terminal_id,
            id="output",
        )
    
    def on_mount(self) -> None:
        """Subscribe to events when mounted."""
        self._monitor.add_listener(self._on_monitor_event)
        self._update_status()
        
        # Load existing output
        output = self.query_one("#output", EnhancedOutputPanel)
        for line in self.terminal_info.output_buffer.get_lines():
            output.add_output_line(
                line.content,
                is_error=line.is_stderr,
                timestamp=False,  # Already has timestamp
            )
    
    def on_unmount(self) -> None:
        """Clean up when unmounted."""
        self._monitor.remove_listener(self._on_monitor_event)
    
    def _on_monitor_event(self, event: TerminalEvent) -> None:
        """Handle monitor events."""
        if event.terminal_id != self.terminal_info.terminal_id:
            return
        
        if event.event_type == TerminalEventType.OUTPUT_RECEIVED:
            self._add_output(
                event.data.get("content", ""),
                is_error=event.data.get("is_stderr", False),
            )
        elif event.event_type == TerminalEventType.ERROR_RECEIVED:
            self._add_output(event.data.get("content", ""), is_error=True)
        elif event.event_type == TerminalEventType.STATE_CHANGED:
            self._update_status()
    
    def _add_output(self, content: str, is_error: bool = False) -> None:
        """Add output to the panel."""
        try:
            output = self.query_one("#output", EnhancedOutputPanel)
            output.add_output_line(content, is_error=is_error)
            
            # Update status
            if self.show_status:
                status = self.query_one("#status", EnhancedStatusBar)
                status.line_count = output._line_count
                status.error_count = output._error_count
        except NoMatches:
            pass
    
    def _update_status(self) -> None:
        """Update the status bar."""
        if not self.show_status:
            return
        
        try:
            status = self.query_one("#status", EnhancedStatusBar)
            status.process_name = self.terminal_info.command[:25]
            status.process_state = self.terminal_info.state.name.title()
            status.pid = self.terminal_info.pid
            
            duration = self.terminal_info.duration_ms
            if duration:
                seconds = int(duration / 1000)
                status.elapsed_time = f"{seconds // 60:02d}:{seconds % 60:02d}"
            else:
                status.elapsed_time = "--:--"
        except NoMatches:
            pass
    
    def on_click(self) -> None:
        """Handle click to select this terminal."""
        self.post_message(self.Selected(self.terminal_info.terminal_id))


# =============================================================================
# Progress Indicator (reused from base)
# =============================================================================

class ProgressIndicator(Widget):
    """Progress indicator with percentage and stage info."""
    
    DEFAULT_CSS = """
    ProgressIndicator {
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    
    ProgressIndicator ProgressBar {
        width: 1fr;
    }
    
    ProgressIndicator .progress-label {
        text-align: right;
        width: 10;
    }
    
    ProgressIndicator .stage-label {
        color: $text-muted;
    }
    """
    
    progress: reactive[float] = reactive(0.0)
    stage: reactive[str] = reactive("")
    
    def compose(self) -> ComposeResult:
        """Compose the progress indicator."""
        with Horizontal():
            yield ProgressBar(total=100, id="progress-bar")
            yield Label("0%", classes="progress-label", id="progress-pct")
        yield Label("", classes="stage-label", id="stage-label")
    
    def watch_progress(self, value: float) -> None:
        """Update progress bar."""
        try:
            bar = self.query_one("#progress-bar", ProgressBar)
            bar.progress = value
            label = self.query_one("#progress-pct", Label)
            label.update(f"{value:.0f}%")
        except NoMatches:
            pass
    
    def watch_stage(self, value: str) -> None:
        """Update stage label."""
        try:
            label = self.query_one("#stage-label", Label)
            label.update(f"Stage: {value}" if value else "")
        except NoMatches:
            pass


# =============================================================================
# Phase 3: Enhanced Execution Monitor
# =============================================================================

class EnhancedExecutionMonitor(Container):
    """Phase 3 Enhanced Execution Monitor.
    
    Integrates:
    - MultiTerminalMonitor for process tracking
    - SessionManager for session persistence
    - CommandNormalizer for cross-platform support
    - TerminalStateMachine for state tracking
    - Event debouncing for performance
    
    Features:
    - Session selector
    - Terminal dashboard
    - Multi-terminal grid (up to 4)
    - Cross-platform command execution
    - State machine metrics
    """
    
    DEFAULT_CSS = """
    EnhancedExecutionMonitor {
        height: 1fr;
        width: 100%;
    }
    
    EnhancedExecutionMonitor #toolbar {
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
        layout: horizontal;
    }
    
    EnhancedExecutionMonitor #toolbar Button {
        margin-right: 1;
    }
    
    EnhancedExecutionMonitor #session-selector {
        dock: top;
        height: 4;
    }
    
    EnhancedExecutionMonitor #dashboard {
        dock: top;
        height: auto;
        max-height: 15;
    }
    
    EnhancedExecutionMonitor #terminal-grid {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "cancel_process", "Cancel"),
        Binding("ctrl+l", "clear_output", "Clear"),
        Binding("ctrl+enter", "run_command", "Run"),
        Binding("alt+1", "select_terminal(1)", "Terminal 1", show=False),
        Binding("alt+2", "select_terminal(2)", "Terminal 2", show=False),
        Binding("alt+3", "select_terminal(3)", "Terminal 3", show=False),
        Binding("alt+4", "select_terminal(4)", "Terminal 4", show=False),
        Binding("alt+n", "new_terminal", "New Terminal"),
        Binding("alt+d", "toggle_dashboard", "Dashboard"),
    ]
    
    class ProcessStarted(Message):
        """Emitted when a process starts."""
        def __init__(
            self,
            terminal_id: str,
            command: str,
            session_id: Optional[str] = None,
        ) -> None:
            self.terminal_id = terminal_id
            self.command = command
            self.session_id = session_id
            super().__init__()
    
    class ProcessCompleted(Message):
        """Emitted when a process completes."""
        def __init__(
            self,
            terminal_id: str,
            return_code: int,
            duration_ms: float,
            session_id: Optional[str] = None,
        ) -> None:
            self.terminal_id = terminal_id
            self.return_code = return_code
            self.duration_ms = duration_ms
            self.session_id = session_id
            super().__init__()
    
    show_dashboard: reactive[bool] = reactive(True)
    selected_terminal_index: reactive[int] = reactive(0)
    
    def __init__(
        self,
        show_toolbar: bool = True,
        show_sessions: bool = True,
        max_terminals: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.show_toolbar = show_toolbar
        self.show_sessions = show_sessions
        self.max_terminals = max_terminals
        
        # Phase 3 components
        self._monitor = get_multi_terminal_monitor()
        self._session_manager = get_session_manager()
        self._normalizer = get_command_normalizer()
        self._state_machine = get_terminal_state_machine()
        self._command_queue = CommandQueue(max_concurrent=max_terminals)
        
        # Active session
        self._current_session: Optional[AgentSession] = None
        
        # Terminal views
        self._terminal_views: List[EnhancedTerminalView] = []
        self._terminal_ids: List[str] = []
    
    def compose(self) -> ComposeResult:
        """Compose the execution monitor."""
        if self.show_toolbar:
            with Horizontal(id="toolbar"):
                yield Button("▶ Run", id="btn-run", variant="success")
                yield Button("⏹ Stop", id="btn-stop", variant="error")
                yield Button("🗑 Clear", id="btn-clear", variant="default")
                yield Button("+ Terminal", id="btn-new-terminal", variant="primary")
                yield Button("📊 Dashboard", id="btn-dashboard", variant="default")
        
        if self.show_sessions:
            yield SessionSelector(
                session_manager=self._session_manager,
                id="session-selector",
            )
        
        yield TerminalDashboard(
            monitor=self._monitor,
            id="dashboard",
            classes="" if self.show_dashboard else "hidden",
        )
        
        yield Container(id="terminal-grid")
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        # Create default session
        self._current_session = self._session_manager.create_session("Main Session")
        
        # Add initial terminal
        self._add_terminal()
        
        # Listen for monitor events
        self._monitor.add_listener(self._on_monitor_event)
    
    def on_unmount(self) -> None:
        """Clean up on unmount."""
        self._monitor.remove_listener(self._on_monitor_event)
        
        # Save sessions
        self._session_manager.save_all()
    
    def _on_monitor_event(self, event: TerminalEvent) -> None:
        """Handle monitor events for UI updates."""
        if event.event_type == TerminalEventType.COMPLETED:
            terminal = self._monitor.get_terminal(event.terminal_id)
            if terminal:
                self.post_message(self.ProcessCompleted(
                    terminal_id=event.terminal_id,
                    return_code=terminal.return_code or 0,
                    duration_ms=terminal.duration_ms or 0,
                    session_id=self._current_session.id if self._current_session else None,
                ))
    
    def _add_terminal(self, name: Optional[str] = None) -> Optional[str]:
        """Add a new terminal view.
        
        Args:
            name: Terminal name
            
        Returns:
            Terminal ID or None if max reached
        """
        if len(self._terminal_ids) >= self.max_terminals:
            self.notify(f"Maximum {self.max_terminals} terminals reached")
            return None
        
        # Create terminal in monitor
        terminal = self._monitor.register(
            command="",
            working_dir=self._current_session.working_dir if self._current_session else ".",
        )
        
        # Create view
        view = EnhancedTerminalView(
            terminal_info=terminal,
            session=self._current_session,
            id=f"terminal-{terminal.terminal_id}",
        )
        
        self._terminal_views.append(view)
        self._terminal_ids.append(terminal.terminal_id)
        
        # Mount
        try:
            grid = self.query_one("#terminal-grid", Container)
            grid.mount(view)
        except NoMatches:
            pass
        
        return terminal.terminal_id
    
    def _remove_terminal(self, index: int) -> bool:
        """Remove a terminal by index.
        
        Args:
            index: Index of terminal to remove
            
        Returns:
            True if removed
        """
        if not (0 <= index < len(self._terminal_ids)):
            return False
        
        if len(self._terminal_ids) <= 1:
            self.notify("Cannot remove last terminal")
            return False
        
        terminal_id = self._terminal_ids.pop(index)
        view = self._terminal_views.pop(index)
        view.remove()
        
        self._monitor.remove_terminal(terminal_id)
        
        return True
    
    async def execute_command(
        self,
        command: str,
        terminal_index: Optional[int] = None,
        normalize: bool = True,
    ) -> str:
        """Execute a command in a terminal.
        
        Args:
            command: Command to execute
            terminal_index: Terminal index (None = selected)
            normalize: Whether to normalize command for platform
            
        Returns:
            Terminal ID
        """
        index = terminal_index if terminal_index is not None else self.selected_terminal_index
        
        if not (0 <= index < len(self._terminal_ids)):
            terminal_id = self._add_terminal()
            if not terminal_id:
                raise RuntimeError("Cannot create terminal")
            index = len(self._terminal_ids) - 1
        
        terminal_id = self._terminal_ids[index]
        
        # Normalize command if requested
        if normalize:
            command = self._normalizer.normalize_command(command)
        
        # Update terminal info
        terminal = self._monitor.get_terminal(terminal_id)
        if terminal:
            # Create new process in state machine
            self._state_machine.create_process(terminal_id, command)
        
        # Add to session history
        if self._current_session:
            self._current_session.add_to_history(command)
        
        # Emit started event
        self.post_message(self.ProcessStarted(
            terminal_id=terminal_id,
            command=command,
            session_id=self._current_session.id if self._current_session else None,
        ))
        
        return terminal_id
    
    def watch_show_dashboard(self, value: bool) -> None:
        """Toggle dashboard visibility."""
        try:
            dashboard = self.query_one("#dashboard", TerminalDashboard)
            dashboard.set_class(not value, "hidden")
        except NoMatches:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle toolbar button presses."""
        button_id = event.button.id
        
        if button_id == "btn-run":
            self.action_run_command()
        elif button_id == "btn-stop":
            self.action_cancel_process()
        elif button_id == "btn-clear":
            self.action_clear_output()
        elif button_id == "btn-new-terminal":
            self._add_terminal()
        elif button_id == "btn-dashboard":
            self.action_toggle_dashboard()
    
    def on_session_selector_session_selected(
        self,
        event: SessionSelector.SessionSelected,
    ) -> None:
        """Handle session selection."""
        session = self._session_manager.get_session(event.session_id)
        if session:
            self._current_session = session
            self.notify(f"Switched to session: {session.name}")
    
    def on_session_selector_session_created(
        self,
        event: SessionSelector.SessionCreated,
    ) -> None:
        """Handle new session creation."""
        self._current_session = event.session
        self.notify(f"Created session: {event.session.name}")
    
    def on_terminal_dashboard_terminal_selected(
        self,
        event: TerminalDashboard.TerminalSelected,
    ) -> None:
        """Handle terminal selection from dashboard."""
        for i, tid in enumerate(self._terminal_ids):
            if tid == event.terminal_id:
                self.selected_terminal_index = i
                break
    
    def action_run_command(self) -> None:
        """Run command (placeholder - override in subclass)."""
        self.notify("Override action_run_command to implement")
    
    def action_cancel_process(self) -> None:
        """Cancel current process."""
        if not self._terminal_ids:
            return
        
        terminal_id = self._terminal_ids[self.selected_terminal_index]
        terminal = self._monitor.get_terminal(terminal_id)
        
        if terminal and terminal.is_running:
            self._monitor.update_state(terminal_id, TerminalState.CANCELLED)
            self._state_machine.transition_sync(
                terminal_id,
                TerminalProcessState.CANCELLED,
            )
            self.notify("Process cancelled")
    
    def action_clear_output(self) -> None:
        """Clear output in selected terminal."""
        if not self._terminal_views:
            return
        
        try:
            view = self._terminal_views[self.selected_terminal_index]
            output = view.query_one("#output", EnhancedOutputPanel)
            output.action_clear()
        except (IndexError, NoMatches):
            pass
    
    def action_toggle_dashboard(self) -> None:
        """Toggle dashboard visibility."""
        self.show_dashboard = not self.show_dashboard
    
    def action_select_terminal(self, number: int) -> None:
        """Select terminal by number (1-based)."""
        index = number - 1
        if 0 <= index < len(self._terminal_views):
            self.selected_terminal_index = index
            self._terminal_views[index].focus()
    
    def action_new_terminal(self) -> None:
        """Add a new terminal."""
        self._add_terminal()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary from state machine.
        
        Returns:
            Metrics summary dictionary
        """
        return self._state_machine.get_summary()
