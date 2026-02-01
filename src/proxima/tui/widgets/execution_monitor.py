"""Real-Time Execution Monitor Widget for Proxima TUI.

Provides a rich, reactive widget for monitoring process execution
with live output streaming, progress indicators, and multi-process
support in a split-pane grid layout.

Features:
- Real-time output streaming with auto-scroll
- ANSI color code support
- Progress bar integration
- Split-pane 2x2 grid for multiple processes
- Terminal switching via Alt+1-9
- Virtual scrolling for large output (10K+ lines)
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
    Footer,
    Header,
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

from proxima.core.event_bus import (
    Event,
    EventBus,
    EventType,
    get_event_bus,
)
from proxima.core.process_streaming import ProcessInfo, ProcessState


@dataclass
class TerminalSession:
    """Represents a terminal session in the monitor."""
    session_id: str
    name: str
    process_info: Optional[ProcessInfo] = None
    lines: List[str] = field(default_factory=list)
    max_lines: int = 10000
    is_active: bool = True
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def add_line(self, line: str) -> None:
        """Add a line to the session, maintaining max_lines limit."""
        self.lines.append(line)
        if len(self.lines) > self.max_lines:
            # Remove oldest lines (virtual scrolling buffer)
            self.lines = self.lines[-self.max_lines:]
    
    @property
    def state_display(self) -> str:
        """Get display string for current state."""
        if not self.process_info:
            return "Idle"
        return self.process_info.state.name.title()
    
    @property
    def output(self) -> str:
        """Get all output as a single string."""
        return "\n".join(self.lines)


class ProcessStatusBar(Static):
    """Status bar showing process state and metrics."""
    
    DEFAULT_CSS = """
    ProcessStatusBar {
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
        border-bottom: solid $primary;
    }
    
    ProcessStatusBar .status-running {
        color: $success;
    }
    
    ProcessStatusBar .status-failed {
        color: $error;
    }
    
    ProcessStatusBar .status-completed {
        color: $primary;
    }
    """
    
    process_name: reactive[str] = reactive("No Process")
    process_state: reactive[str] = reactive("Idle")
    elapsed_time: reactive[str] = reactive("--:--")
    line_count: reactive[int] = reactive(0)
    
    def render(self) -> RenderableType:
        """Render the status bar."""
        state_style = {
            "Running": "bold green",
            "Completed": "bold blue",
            "Failed": "bold red",
            "Cancelled": "bold yellow",
            "Timeout": "bold magenta",
            "Idle": "dim",
        }.get(self.process_state, "")
        
        state_text = Text(self.process_state, style=state_style)
        
        return Text.assemble(
            ("📋 ", "bold"),
            (f"{self.process_name} ", "bold"),
            ("| ", "dim"),
            ("State: ", "dim"),
            state_text,
            (" | ", "dim"),
            ("⏱️ ", ""),
            (self.elapsed_time, ""),
            (" | ", "dim"),
            ("📝 ", ""),
            (f"{self.line_count} lines", ""),
        )


class OutputStreamPanel(RichLog):
    """Enhanced RichLog for streaming process output.
    
    Features auto-scroll, ANSI support, and virtual scrolling
    for handling large output volumes efficiently.
    """
    
    DEFAULT_CSS = """
    OutputStreamPanel {
        height: 1fr;
        border: solid $primary;
        background: $surface;
        scrollbar-gutter: stable;
    }
    
    OutputStreamPanel:focus {
        border: double $accent;
    }
    
    OutputStreamPanel.has-errors {
        border: solid $error;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+s", "toggle_scroll", "Toggle Scroll"),
        Binding("ctrl+f", "find", "Find"),
    ]
    
    auto_scroll: reactive[bool] = reactive(True)
    has_errors: reactive[bool] = reactive(False)
    
    def __init__(
        self,
        session_id: str = "",
        max_lines: int = 10000,
        **kwargs,
    ):
        super().__init__(
            highlight=True,
            markup=True,
            wrap=False,
            max_lines=max_lines,
            **kwargs,
        )
        self.session_id = session_id
        self._line_count = 0
        self._error_count = 0
    
    def watch_has_errors(self, has_errors: bool) -> None:
        """Update styling when errors are detected."""
        self.set_class(has_errors, "has-errors")
    
    def add_output_line(
        self,
        line: str,
        is_error: bool = False,
        timestamp: bool = False,
    ) -> None:
        """Add a line to the output panel.
        
        Args:
            line: The line to add
            is_error: Whether this is an error line
            timestamp: Whether to prefix with timestamp
        """
        self._line_count += 1
        
        if is_error:
            self._error_count += 1
            self.has_errors = True
            style = Style(color="red")
        else:
            style = None
        
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            prefix = Text(f"[{ts}] ", style="dim")
            text = Text.assemble(prefix, (line, style) if style else line)
        else:
            text = Text(line, style=style) if style else line
        
        self.write(text)
        
        if self.auto_scroll:
            self.scroll_end(animate=False)
    
    def action_clear(self) -> None:
        """Clear the output panel."""
        self.clear()
        self._line_count = 0
        self._error_count = 0
        self.has_errors = False
    
    def action_toggle_scroll(self) -> None:
        """Toggle auto-scroll."""
        self.auto_scroll = not self.auto_scroll
        self.notify(f"Auto-scroll: {'On' if self.auto_scroll else 'Off'}")
    
    def action_find(self) -> None:
        """Open find dialog (placeholder)."""
        self.notify("Find: Not implemented yet")
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get output statistics."""
        return {
            "line_count": self._line_count,
            "error_count": self._error_count,
        }


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


class SingleTerminalView(Container):
    """Single terminal view with output and status."""
    
    DEFAULT_CSS = """
    SingleTerminalView {
        height: 1fr;
        width: 1fr;
        border: solid $primary;
    }
    
    SingleTerminalView:focus-within {
        border: double $accent;
    }
    
    SingleTerminalView.selected {
        border: double $success;
    }
    """
    
    class Selected(Message):
        """Emitted when this terminal is selected."""
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            super().__init__()
    
    def __init__(
        self,
        session: TerminalSession,
        show_status: bool = True,
        show_progress: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session = session
        self.show_status = show_status
        self.show_progress = show_progress
        self._subscription_id: Optional[str] = None
    
    def compose(self) -> ComposeResult:
        """Compose the terminal view."""
        if self.show_status:
            yield ProcessStatusBar(id="status")
        if self.show_progress:
            yield ProgressIndicator(id="progress")
        yield OutputStreamPanel(
            session_id=self.session.session_id,
            id="output",
        )
    
    def on_mount(self) -> None:
        """Subscribe to events when mounted."""
        self._subscribe_to_events()
        # Load existing lines
        output = self.query_one("#output", OutputStreamPanel)
        for line in self.session.lines:
            output.add_output_line(line)
    
    def on_unmount(self) -> None:
        """Unsubscribe from events when unmounted."""
        if self._subscription_id:
            get_event_bus().unsubscribe(self._subscription_id)
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        self._subscription_id = get_event_bus().subscribe(
            self._handle_event,
            event_types={
                EventType.OUTPUT_LINE,
                EventType.ERROR_LINE,
                EventType.PROGRESS_UPDATE,
                EventType.PROCESS_STARTED,
                EventType.PROCESS_COMPLETED,
                EventType.PROCESS_FAILED,
            },
            source_filter=self.session.session_id,
        )
    
    async def _handle_event(self, event: Event) -> None:
        """Handle incoming events."""
        if event.event_type in (EventType.OUTPUT_LINE, EventType.ERROR_LINE):
            line = event.payload.get("line", "")
            is_error = event.event_type == EventType.ERROR_LINE
            
            # Update session
            self.session.add_line(line)
            
            # Update UI
            try:
                output = self.query_one("#output", OutputStreamPanel)
                output.add_output_line(line, is_error=is_error, timestamp=True)
                
                if self.show_status:
                    status = self.query_one("#status", ProcessStatusBar)
                    status.line_count = len(self.session.lines)
            except NoMatches:
                pass
        
        elif event.event_type == EventType.PROGRESS_UPDATE:
            progress = event.payload.get("progress", 0)
            message = event.payload.get("message", "")
            
            try:
                if self.show_progress:
                    indicator = self.query_one("#progress", ProgressIndicator)
                    indicator.progress = progress
                    indicator.stage = message
            except NoMatches:
                pass
        
        elif event.event_type == EventType.PROCESS_STARTED:
            try:
                if self.show_status:
                    status = self.query_one("#status", ProcessStatusBar)
                    status.process_state = "Running"
                    status.process_name = event.payload.get("command", "Process")[:30]
            except NoMatches:
                pass
        
        elif event.event_type in (EventType.PROCESS_COMPLETED, EventType.PROCESS_FAILED):
            state = "Completed" if event.event_type == EventType.PROCESS_COMPLETED else "Failed"
            try:
                if self.show_status:
                    status = self.query_one("#status", ProcessStatusBar)
                    status.process_state = state
            except NoMatches:
                pass
    
    def on_click(self) -> None:
        """Handle click to select this terminal."""
        self.post_message(self.Selected(self.session.session_id))


class MultiTerminalGrid(Container):
    """2x2 grid layout for multiple terminal views.
    
    Supports up to 4 concurrent terminal views in a grid layout
    with terminal switching via Alt+1-4.
    """
    
    DEFAULT_CSS = """
    MultiTerminalGrid {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        height: 1fr;
        padding: 1;
    }
    
    MultiTerminalGrid.single-view {
        grid-size: 1 1;
    }
    
    MultiTerminalGrid.dual-view {
        grid-size: 2 1;
    }
    """
    
    BINDINGS = [
        Binding("alt+1", "select_terminal(1)", "Terminal 1", show=False),
        Binding("alt+2", "select_terminal(2)", "Terminal 2", show=False),
        Binding("alt+3", "select_terminal(3)", "Terminal 3", show=False),
        Binding("alt+4", "select_terminal(4)", "Terminal 4", show=False),
        Binding("alt+n", "new_terminal", "New Terminal"),
        Binding("alt+w", "close_terminal", "Close Terminal"),
    ]
    
    selected_index: reactive[int] = reactive(0)
    
    def __init__(self, max_terminals: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.max_terminals = max_terminals
        self._sessions: List[TerminalSession] = []
        self._views: List[SingleTerminalView] = []
    
    def compose(self) -> ComposeResult:
        """Initial empty composition."""
        pass
    
    def on_mount(self) -> None:
        """Create initial terminal on mount."""
        self.add_terminal("Main")
    
    def add_terminal(self, name: str = "Terminal") -> Optional[TerminalSession]:
        """Add a new terminal session.
        
        Args:
            name: Display name for the terminal
            
        Returns:
            The created session, or None if max reached
        """
        if len(self._sessions) >= self.max_terminals:
            self.notify(f"Maximum {self.max_terminals} terminals reached")
            return None
        
        session_id = f"term_{len(self._sessions) + 1}_{id(self)}"
        session = TerminalSession(
            session_id=session_id,
            name=f"{name} {len(self._sessions) + 1}",
        )
        self._sessions.append(session)
        
        view = SingleTerminalView(session, id=f"view_{session_id}")
        self._views.append(view)
        self.mount(view)
        
        self._update_layout()
        self.selected_index = len(self._sessions) - 1
        
        return session
    
    def remove_terminal(self, index: int) -> bool:
        """Remove a terminal by index.
        
        Args:
            index: Index of terminal to remove
            
        Returns:
            True if removed successfully
        """
        if not (0 <= index < len(self._sessions)):
            return False
        
        if len(self._sessions) <= 1:
            self.notify("Cannot remove last terminal")
            return False
        
        session = self._sessions.pop(index)
        view = self._views.pop(index)
        view.remove()
        
        self._update_layout()
        
        if self.selected_index >= len(self._sessions):
            self.selected_index = len(self._sessions) - 1
        
        return True
    
    def _update_layout(self) -> None:
        """Update grid layout based on terminal count."""
        count = len(self._sessions)
        
        self.remove_class("single-view")
        self.remove_class("dual-view")
        
        if count == 1:
            self.add_class("single-view")
        elif count == 2:
            self.add_class("dual-view")
    
    def watch_selected_index(self, index: int) -> None:
        """Update selected terminal highlighting."""
        for i, view in enumerate(self._views):
            view.set_class(i == index, "selected")
    
    def action_select_terminal(self, number: int) -> None:
        """Select terminal by number (1-based)."""
        index = number - 1
        if 0 <= index < len(self._sessions):
            self.selected_index = index
            self._views[index].focus()
    
    def action_new_terminal(self) -> None:
        """Add a new terminal."""
        self.add_terminal()
    
    def action_close_terminal(self) -> None:
        """Close the currently selected terminal."""
        self.remove_terminal(self.selected_index)
    
    def get_session(self, index: int) -> Optional[TerminalSession]:
        """Get session by index."""
        if 0 <= index < len(self._sessions):
            return self._sessions[index]
        return None
    
    @property
    def current_session(self) -> Optional[TerminalSession]:
        """Get the currently selected session."""
        return self.get_session(self.selected_index)
    
    @property
    def sessions(self) -> List[TerminalSession]:
        """Get all sessions."""
        return self._sessions.copy()
    
    def on_single_terminal_view_selected(
        self,
        event: SingleTerminalView.Selected,
    ) -> None:
        """Handle terminal selection from click."""
        for i, session in enumerate(self._sessions):
            if session.session_id == event.session_id:
                self.selected_index = i
                break


class ExecutionMonitor(Container):
    """Main execution monitor widget combining all components.
    
    Provides a complete interface for monitoring process execution
    with real-time output, progress tracking, and multi-terminal support.
    """
    
    DEFAULT_CSS = """
    ExecutionMonitor {
        height: 1fr;
        width: 100%;
    }
    
    ExecutionMonitor #toolbar {
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
    }
    
    ExecutionMonitor #toolbar Button {
        margin-right: 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "cancel_process", "Cancel"),
        Binding("ctrl+l", "clear_output", "Clear"),
        Binding("ctrl+enter", "run_command", "Run"),
    ]
    
    class ProcessStarted(Message):
        """Emitted when a process starts."""
        def __init__(self, session_id: str, command: str) -> None:
            self.session_id = session_id
            self.command = command
            super().__init__()
    
    class ProcessCompleted(Message):
        """Emitted when a process completes."""
        def __init__(
            self,
            session_id: str,
            return_code: int,
            duration_ms: float,
        ) -> None:
            self.session_id = session_id
            self.return_code = return_code
            self.duration_ms = duration_ms
            super().__init__()
    
    def __init__(
        self,
        show_toolbar: bool = True,
        max_terminals: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.show_toolbar = show_toolbar
        self.max_terminals = max_terminals
        self._event_bus = get_event_bus()
    
    def compose(self) -> ComposeResult:
        """Compose the execution monitor."""
        if self.show_toolbar:
            with Horizontal(id="toolbar"):
                yield Button("▶ Run", id="btn-run", variant="success")
                yield Button("⏹ Stop", id="btn-stop", variant="error")
                yield Button("🗑 Clear", id="btn-clear", variant="default")
                yield Button("+ Terminal", id="btn-new-terminal", variant="primary")
        
        yield MultiTerminalGrid(
            max_terminals=self.max_terminals,
            id="terminal-grid",
        )
    
    async def on_mount(self) -> None:
        """Start event bus on mount."""
        if not self._event_bus.is_running:
            await self._event_bus.start()
    
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
            self._get_grid().action_new_terminal()
    
    def _get_grid(self) -> MultiTerminalGrid:
        """Get the terminal grid widget."""
        return self.query_one("#terminal-grid", MultiTerminalGrid)
    
    def action_run_command(self) -> None:
        """Run command (placeholder - override in subclass)."""
        self.notify("Override action_run_command to implement")
    
    def action_cancel_process(self) -> None:
        """Cancel current process (placeholder)."""
        self.notify("Cancel requested")
    
    def action_clear_output(self) -> None:
        """Clear output in selected terminal."""
        grid = self._get_grid()
        if grid._views and 0 <= grid.selected_index < len(grid._views):
            try:
                view = grid._views[grid.selected_index]
                output = view.query_one("#output", OutputStreamPanel)
                output.action_clear()
            except NoMatches:
                pass
    
    def add_output(
        self,
        line: str,
        is_error: bool = False,
        terminal_index: Optional[int] = None,
    ) -> None:
        """Add output line to a terminal.
        
        Args:
            line: The line to add
            is_error: Whether this is an error
            terminal_index: Target terminal (None = selected)
        """
        grid = self._get_grid()
        index = terminal_index if terminal_index is not None else grid.selected_index
        
        if 0 <= index < len(grid._views):
            try:
                view = grid._views[index]
                output = view.query_one("#output", OutputStreamPanel)
                output.add_output_line(line, is_error=is_error, timestamp=True)
            except NoMatches:
                pass
    
    def set_progress(
        self,
        progress: float,
        stage: str = "",
        terminal_index: Optional[int] = None,
    ) -> None:
        """Set progress for a terminal.
        
        Args:
            progress: Progress percentage (0-100)
            stage: Current stage name
            terminal_index: Target terminal (None = selected)
        """
        grid = self._get_grid()
        index = terminal_index if terminal_index is not None else grid.selected_index
        
        if 0 <= index < len(grid._views):
            try:
                view = grid._views[index]
                indicator = view.query_one("#progress", ProgressIndicator)
                indicator.progress = progress
                indicator.stage = stage
            except NoMatches:
                pass
