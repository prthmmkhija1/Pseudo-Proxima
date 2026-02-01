"""Agent UI Widgets for Proxima TUI.

Provides UI components for agent functionality:
- Consent dialogs
- Terminal output displays
- Tool execution views
- Progress indicators
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Button, RichLog, Label, DataTable, ProgressBar
from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.app import ComposeResult
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
from rich.console import Group

from ..styles.theme import get_theme


@dataclass
class ConsentDisplayInfo:
    """Information for displaying a consent request."""
    request_id: str
    operation: str
    description: str
    details: Dict[str, Any]
    risk_level: str
    timestamp: str


class ConsentDialog(ModalScreen[Dict[str, Any]]):
    """Dialog for requesting user consent for agent operations.
    
    Displays operation details and allows user to approve/deny.
    """
    
    BINDINGS = [
        Binding("enter", "approve", "Approve"),
        Binding("escape", "deny", "Deny"),
        Binding("a", "approve_always", "Always"),
        Binding("s", "approve_session", "This Session"),
    ]
    
    DEFAULT_CSS = """
    ConsentDialog {
        align: center middle;
    }
    
    ConsentDialog > Vertical {
        width: 70;
        height: auto;
        max-height: 80%;
        border: thick $error;
        background: $surface;
        padding: 1 2;
    }
    
    ConsentDialog .consent-header {
        height: auto;
        margin-bottom: 1;
    }
    
    ConsentDialog .consent-title {
        text-style: bold;
        color: $warning;
        text-align: center;
    }
    
    ConsentDialog .consent-subtitle {
        color: $text-muted;
        text-align: center;
    }
    
    ConsentDialog .consent-operation {
        height: auto;
        margin: 1 0;
        padding: 1;
        background: $surface-darken-2;
        border: solid $warning;
    }
    
    ConsentDialog .operation-name {
        text-style: bold;
        color: $accent;
    }
    
    ConsentDialog .operation-desc {
        color: $text;
        margin-top: 1;
    }
    
    ConsentDialog .details-section {
        height: auto;
        max-height: 15;
        overflow-y: auto;
        margin: 1 0;
        padding: 1;
        background: $surface-darken-3;
    }
    
    ConsentDialog .risk-high {
        color: $error;
        text-style: bold;
    }
    
    ConsentDialog .risk-medium {
        color: $warning;
    }
    
    ConsentDialog .risk-low {
        color: $success;
    }
    
    ConsentDialog .button-row {
        height: auto;
        margin-top: 1;
        layout: horizontal;
        align: center middle;
    }
    
    ConsentDialog .consent-btn {
        margin: 0 1;
    }
    
    ConsentDialog .shortcuts-hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """
    
    def __init__(self, consent_info: ConsentDisplayInfo) -> None:
        super().__init__()
        self._info = consent_info
    
    def compose(self) -> ComposeResult:
        with Vertical():
            # Header
            with Container(classes="consent-header"):
                yield Static("⚠️ Agent Action Requires Consent", classes="consent-title")
                yield Static("The AI agent wants to perform the following action:", classes="consent-subtitle")
            
            # Operation info
            with Container(classes="consent-operation"):
                yield Static(f"Operation: {self._info.operation}", classes="operation-name")
                yield Static(self._info.description, classes="operation-desc")
            
            # Details
            if self._info.details:
                with ScrollableContainer(classes="details-section"):
                    yield Static("Details:", classes="operation-name")
                    for key, value in self._info.details.items():
                        yield Static(f"  {key}: {value}")
            
            # Risk level
            risk_class = f"risk-{self._info.risk_level.lower()}"
            yield Static(f"Risk Level: {self._info.risk_level.upper()}", classes=risk_class)
            
            # Buttons
            with Horizontal(classes="button-row"):
                yield Button("✓ Approve", id="btn-approve", variant="success", classes="consent-btn")
                yield Button("↺ This Session", id="btn-session", variant="primary", classes="consent-btn")
                yield Button("∀ Always", id="btn-always", variant="default", classes="consent-btn")
                yield Button("✗ Deny", id="btn-deny", variant="error", classes="consent-btn")
            
            yield Static("Enter=Approve | S=Session | A=Always | Esc=Deny", classes="shortcuts-hint")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-approve":
            self.dismiss({"approved": True, "scope": "once"})
        elif btn_id == "btn-session":
            self.dismiss({"approved": True, "scope": "session"})
        elif btn_id == "btn-always":
            self.dismiss({"approved": True, "scope": "always"})
        else:
            self.dismiss({"approved": False, "scope": None})
    
    def action_approve(self) -> None:
        self.dismiss({"approved": True, "scope": "once"})
    
    def action_approve_always(self) -> None:
        self.dismiss({"approved": True, "scope": "always"})
    
    def action_approve_session(self) -> None:
        self.dismiss({"approved": True, "scope": "session"})
    
    def action_deny(self) -> None:
        self.dismiss({"approved": False, "scope": None})


class TerminalOutputPanel(Static):
    """Widget displaying live terminal output."""
    
    DEFAULT_CSS = """
    TerminalOutputPanel {
        height: 100%;
        background: $surface-darken-3;
        border: solid $primary-darken-2;
        padding: 0 1;
    }
    
    TerminalOutputPanel .terminal-header {
        height: 3;
        background: $surface-darken-2;
        border-bottom: solid $primary-darken-3;
        padding: 1;
    }
    
    TerminalOutputPanel .terminal-title {
        text-style: bold;
        color: $accent;
    }
    
    TerminalOutputPanel .terminal-status {
        color: $success;
    }
    
    TerminalOutputPanel .terminal-status-error {
        color: $error;
    }
    
    TerminalOutputPanel .terminal-output {
        height: 1fr;
        overflow-y: auto;
    }
    
    TerminalOutputPanel .output-line {
        margin: 0;
    }
    
    TerminalOutputPanel .output-error {
        color: $error;
    }
    
    TerminalOutputPanel .output-command {
        color: $accent;
        text-style: bold;
    }
    """
    
    terminal_id: reactive[str] = reactive("")
    terminal_name: reactive[str] = reactive("Terminal")
    status: reactive[str] = reactive("idle")  # idle, running, completed, error
    
    def __init__(
        self,
        terminal_id: str = "",
        name: str = "Terminal",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.terminal_id = terminal_id
        self.terminal_name = name
        self._output_lines: List[Text] = []
        self._max_lines = 500
    
    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="terminal-header"):
                yield Static(f"⬛ {self.terminal_name}", classes="terminal-title", id="term-title")
                yield Static("● Idle", classes="terminal-status", id="term-status")
            
            yield RichLog(
                auto_scroll=True,
                classes="terminal-output",
                id="term-output",
                highlight=True,
            )
    
    def append_output(self, line: str, is_error: bool = False, is_command: bool = False) -> None:
        """Append output line to terminal."""
        try:
            theme = get_theme()
            output = self.query_one("#term-output", RichLog)
            
            text = Text()
            if is_command:
                text.append(f"$ {line}", style=f"bold {theme.accent}")
            elif is_error:
                text.append(line, style=theme.error)
            else:
                text.append(line, style=theme.fg_base)
            
            output.write(text)
            
            # Track lines
            self._output_lines.append(text)
            if len(self._output_lines) > self._max_lines:
                self._output_lines.pop(0)
        except Exception:
            pass
    
    def set_status(self, status: str) -> None:
        """Update terminal status."""
        self.status = status
        try:
            status_widget = self.query_one("#term-status", Static)
            theme = get_theme()
            
            status_icons = {
                "idle": ("● Idle", theme.fg_muted),
                "running": ("◉ Running", theme.warning),
                "completed": ("✓ Completed", theme.success),
                "error": ("✗ Error", theme.error),
            }
            
            icon, color = status_icons.get(status, ("● Unknown", theme.fg_muted))
            status_widget.update(Text(icon, style=color))
        except Exception:
            pass
    
    def clear(self) -> None:
        """Clear terminal output."""
        try:
            output = self.query_one("#term-output", RichLog)
            output.clear()
            self._output_lines.clear()
        except Exception:
            pass


class ToolExecutionView(Static):
    """Widget showing current tool execution status."""
    
    DEFAULT_CSS = """
    ToolExecutionView {
        height: auto;
        margin: 1 0;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-2;
    }
    
    ToolExecutionView .tool-header {
        height: auto;
        layout: horizontal;
    }
    
    ToolExecutionView .tool-name {
        text-style: bold;
        color: $accent;
        width: 1fr;
    }
    
    ToolExecutionView .tool-status {
        width: auto;
    }
    
    ToolExecutionView .tool-args {
        color: $text-muted;
        margin-top: 1;
    }
    
    ToolExecutionView .tool-progress {
        margin-top: 1;
    }
    
    ToolExecutionView .tool-result {
        margin-top: 1;
        padding: 1;
        background: $surface-darken-2;
    }
    
    ToolExecutionView .result-success {
        color: $success;
    }
    
    ToolExecutionView .result-error {
        color: $error;
    }
    """
    
    tool_name: reactive[str] = reactive("")
    status: reactive[str] = reactive("pending")  # pending, running, completed, failed
    
    def __init__(
        self,
        tool_name: str = "",
        arguments: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self._arguments = arguments or {}
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
    
    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="tool-header"):
                yield Static(f"🔧 {self.tool_name}", classes="tool-name", id="tool-name")
                yield Static("⏳ Pending", classes="tool-status", id="tool-status")
            
            # Arguments
            if self._arguments:
                args_text = ", ".join(f"{k}={v}" for k, v in self._arguments.items())
                if len(args_text) > 60:
                    args_text = args_text[:57] + "..."
                yield Static(f"Args: {args_text}", classes="tool-args")
            
            yield ProgressBar(total=100, show_eta=False, classes="tool-progress", id="tool-progress")
            
            yield Static("", classes="tool-result", id="tool-result")
    
    def set_running(self) -> None:
        """Mark tool as running."""
        self.status = "running"
        try:
            self.query_one("#tool-status", Static).update("⚙️ Running")
            progress = self.query_one("#tool-progress", ProgressBar)
            progress.update(progress=50)
        except Exception:
            pass
    
    def set_completed(self, result: Dict[str, Any]) -> None:
        """Mark tool as completed with result."""
        self.status = "completed"
        self._result = result
        try:
            theme = get_theme()
            self.query_one("#tool-status", Static).update(
                Text("✓ Completed", style=theme.success)
            )
            progress = self.query_one("#tool-progress", ProgressBar)
            progress.update(progress=100)
            
            result_text = str(result)[:100]
            self.query_one("#tool-result", Static).update(
                Text(f"Result: {result_text}", style=theme.success)
            )
        except Exception:
            pass
    
    def set_failed(self, error: str) -> None:
        """Mark tool as failed with error."""
        self.status = "failed"
        self._error = error
        try:
            theme = get_theme()
            self.query_one("#tool-status", Static).update(
                Text("✗ Failed", style=theme.error)
            )
            progress = self.query_one("#tool-progress", ProgressBar)
            progress.update(progress=100)
            
            self.query_one("#tool-result", Static).update(
                Text(f"Error: {error}", style=theme.error)
            )
        except Exception:
            pass


class MultiTerminalView(Container):
    """Widget showing multiple terminal outputs in split view."""
    
    DEFAULT_CSS = """
    MultiTerminalView {
        height: 100%;
        layout: horizontal;
    }
    
    MultiTerminalView .terminal-col {
        width: 1fr;
        height: 100%;
        margin: 0 1;
    }
    
    MultiTerminalView .no-terminals {
        align: center middle;
        color: $text-muted;
    }
    """
    
    def __init__(self, max_terminals: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_terminals = max_terminals
        self._terminals: Dict[str, TerminalOutputPanel] = {}
    
    def compose(self) -> ComposeResult:
        yield Static("No active terminals", classes="no-terminals", id="no-terminals")
    
    def add_terminal(self, terminal_id: str, name: str = "Terminal") -> TerminalOutputPanel:
        """Add a terminal panel."""
        if len(self._terminals) >= self._max_terminals:
            # Remove oldest terminal
            oldest_id = list(self._terminals.keys())[0]
            self.remove_terminal(oldest_id)
        
        # Hide no-terminals message
        try:
            self.query_one("#no-terminals").display = False
        except Exception:
            pass
        
        # Create new terminal panel
        panel = TerminalOutputPanel(terminal_id=terminal_id, name=name)
        self._terminals[terminal_id] = panel
        
        with Vertical(classes="terminal-col", id=f"col-{terminal_id}"):
            self.mount(panel)
        
        return panel
    
    def remove_terminal(self, terminal_id: str) -> bool:
        """Remove a terminal panel."""
        if terminal_id not in self._terminals:
            return False
        
        try:
            col = self.query_one(f"#col-{terminal_id}")
            col.remove()
            del self._terminals[terminal_id]
            
            # Show no-terminals if empty
            if not self._terminals:
                self.query_one("#no-terminals").display = True
            
            return True
        except Exception:
            return False
    
    def get_terminal(self, terminal_id: str) -> Optional[TerminalOutputPanel]:
        """Get a terminal panel by ID."""
        return self._terminals.get(terminal_id)
    
    def append_to_terminal(
        self,
        terminal_id: str,
        line: str,
        is_error: bool = False,
        is_command: bool = False,
    ) -> None:
        """Append output to a specific terminal."""
        panel = self._terminals.get(terminal_id)
        if panel:
            panel.append_output(line, is_error, is_command)


class AgentPlanView(Static):
    """Widget showing agent execution plan."""
    
    DEFAULT_CSS = """
    AgentPlanView {
        height: auto;
        margin: 1 0;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary;
    }
    
    AgentPlanView .plan-header {
        height: auto;
        margin-bottom: 1;
    }
    
    AgentPlanView .plan-title {
        text-style: bold;
        color: $accent;
    }
    
    AgentPlanView .plan-desc {
        color: $text-muted;
    }
    
    AgentPlanView .step-list {
        height: auto;
    }
    
    AgentPlanView .step-item {
        height: auto;
        layout: horizontal;
        margin: 0 0 0 2;
    }
    
    AgentPlanView .step-status {
        width: 3;
    }
    
    AgentPlanView .step-pending {
        color: $text-muted;
    }
    
    AgentPlanView .step-current {
        color: $warning;
        text-style: bold;
    }
    
    AgentPlanView .step-completed {
        color: $success;
    }
    
    AgentPlanView .step-failed {
        color: $error;
    }
    """
    
    def __init__(
        self,
        plan_id: str,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._plan_id = plan_id
        self._name = name
        self._description = description
        self._steps = steps
        self._current_step = -1
        self._step_results: List[str] = ["pending"] * len(steps)
    
    def compose(self) -> ComposeResult:
        with Vertical():
            with Container(classes="plan-header"):
                yield Static(f"📋 Plan: {self._name}", classes="plan-title")
                yield Static(self._description, classes="plan-desc")
            
            with Vertical(classes="step-list", id="step-list"):
                for i, step in enumerate(self._steps):
                    tool_name = step.get("tool", "unknown")
                    with Horizontal(classes="step-item"):
                        yield Static("○", classes="step-status step-pending", id=f"step-status-{i}")
                        yield Static(f"{i+1}. {tool_name}")
    
    def set_step_status(self, step_index: int, status: str) -> None:
        """Update step status (pending, current, completed, failed)."""
        if step_index < 0 or step_index >= len(self._steps):
            return
        
        self._step_results[step_index] = status
        
        try:
            status_widget = self.query_one(f"#step-status-{step_index}", Static)
            
            icons = {
                "pending": ("○", "step-pending"),
                "current": ("◉", "step-current"),
                "completed": ("✓", "step-completed"),
                "failed": ("✗", "step-failed"),
            }
            
            icon, css_class = icons.get(status, ("○", "step-pending"))
            status_widget.update(icon)
            status_widget.set_classes(f"step-status {css_class}")
        except Exception:
            pass
    
    def set_current_step(self, step_index: int) -> None:
        """Mark a step as currently executing."""
        self._current_step = step_index
        self.set_step_status(step_index, "current")


class UndoRedoPanel(Static):
    """Widget showing undo/redo history."""
    
    DEFAULT_CSS = """
    UndoRedoPanel {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-2;
    }
    
    UndoRedoPanel .undo-header {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    UndoRedoPanel .undo-buttons {
        layout: horizontal;
        height: auto;
    }
    
    UndoRedoPanel .undo-btn {
        margin-right: 1;
    }
    
    UndoRedoPanel .history-count {
        color: $text-muted;
        margin-top: 1;
    }
    """
    
    def __init__(
        self,
        on_undo: Optional[Callable[[], None]] = None,
        on_redo: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._on_undo = on_undo
        self._on_redo = on_redo
        self._undo_count = 0
        self._redo_count = 0
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("↶ Undo/Redo", classes="undo-header")
            with Horizontal(classes="undo-buttons"):
                yield Button("↶ Undo", id="btn-undo", classes="undo-btn", disabled=True)
                yield Button("↷ Redo", id="btn-redo", classes="undo-btn", disabled=True)
            yield Static("No modifications to undo", classes="history-count", id="history-count")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-undo" and self._on_undo:
            self._on_undo()
        elif event.button.id == "btn-redo" and self._on_redo:
            self._on_redo()
    
    def update_counts(self, undo_count: int, redo_count: int) -> None:
        """Update undo/redo button states and counts."""
        self._undo_count = undo_count
        self._redo_count = redo_count
        
        try:
            undo_btn = self.query_one("#btn-undo", Button)
            redo_btn = self.query_one("#btn-redo", Button)
            count_label = self.query_one("#history-count", Static)
            
            undo_btn.disabled = undo_count == 0
            redo_btn.disabled = redo_count == 0
            
            count_label.update(f"{undo_count} undoable, {redo_count} redoable")
        except Exception:
            pass
