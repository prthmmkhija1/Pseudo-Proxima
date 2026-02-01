"""Proxima TUI Widgets Package.

Custom Textual widgets for the Proxima TUI interface.
"""

from .diff_viewer import (
    DiffViewer,
    SideBySideDiffViewer,
    InlineDiffViewer,
    DiffViewMode,
    DiffLine
)

# Phase 4: Testing & Validation Interface
from .test_results_display import (
    TestResultsDisplay,
    TestProgressBar,
    TestCategoryWidget,
    TestResultItem,
    TestSummaryWidget,
    MeasurementResultsWidget,
)

# Phase 6: Change Management & Approval System
from .change_history import (
    ChangeHistoryWidget,
    ChangeHistoryItem,
    ChangeTimelineWidget,
    ChangeSummaryTable,
    ApprovalProgressWidget,
)

# Agent UI Widgets
from .agent_widgets import (
    ConsentDialog,
    ConsentDisplayInfo,
    TerminalOutputPanel,
    ToolExecutionView,
    MultiTerminalView,
    AgentPlanView,
    UndoRedoPanel,
)

# Phase 2: Agent UI/UX Enhanced Widgets
from .agent_ui_enhanced import (
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
)

# Phase 1: Real-Time Execution Monitor Widgets
from .execution_monitor import (
    TerminalSession,
    ProcessStatusBar,
    OutputStreamPanel,
    ProgressIndicator,
    SingleTerminalView,
    MultiTerminalGrid,
    ExecutionMonitor,
)

# Phase 1: Real-Time Results Viewer Widgets
from .results_viewer import (
    ExecutionResult,
    ResultsDatabase,
    ResultSummaryCard,
    ResultDetailsPanel,
    ResultsTable,
    RealTimeResultsViewer,
)

# Phase 3: Enhanced Execution Monitor
from .execution_monitor_enhanced import (
    EnhancedStatusBar,
    SessionSelector,
    TerminalDashboard,
    EnhancedOutputPanel,
    EnhancedTerminalView,
    EnhancedExecutionMonitor,
)

# Backward compatibility imports from widgets_compat module
# These were originally in the old widgets.py file
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time

from textual.widget import Widget
from textual.widgets import Static
from textual.screen import ModalScreen


class StatusLevel(Enum):
    """Status level enum for widgets."""
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class BackendStatus(Enum):
    """Backend status enum."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class StatusItem:
    """Status item for display."""
    label: str
    value: str
    level: StatusLevel = StatusLevel.INFO
    icon: str = ""


@dataclass
class LogEntry:
    """Log entry for log viewers."""
    timestamp: float
    level: str
    message: str
    component: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def format_level(self) -> str:
        return self.level.upper()
    
    def format_timestamp(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


@dataclass
class BackendInfo:
    """Backend information."""
    name: str
    backend_type: str
    status: BackendStatus = BackendStatus.UNKNOWN
    total_executions: int = 0
    avg_latency_ms: Optional[float] = None
    last_error: str = ""
    last_used: Optional[str] = None


class StatusPanel(Static):
    """Status panel widget."""
    def __init__(self, title: str = "Status", items: Optional[List[StatusItem]] = None, **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._items = items or []


class LogPanel(Static):
    """Log panel widget."""
    def __init__(self, title: str = "Log", max_lines: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._max_lines = max_lines
        self._entries: List[LogEntry] = []


class LogViewer(Static):
    """Log viewer widget."""
    def __init__(self, max_entries: int = 100, auto_scroll: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._max_entries = max_entries
        self._auto_scroll = auto_scroll
        self._entries: List[LogEntry] = []


class BackendCard(Static):
    """Backend card widget."""
    def __init__(self, backend: Optional[BackendInfo] = None, info: Optional[BackendInfo] = None, **kwargs):
        super().__init__(**kwargs)
        # Support both 'backend' and 'info' parameter names for compatibility
        self._backend = backend or info
        self._info = self._backend


class ResultsTable(Static):
    """Results table widget."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: List[Any] = []


class ExecutionTimer(Static):
    """Execution timer widget."""
    def __init__(self, label: str = "", **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0
    
    @property
    def elapsed(self) -> float:
        return self._elapsed


class MetricDisplay(Static):
    """Metric display widget."""
    def __init__(self, label: str = "", value: str = "", unit: str = "", **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._unit = unit
    
    @property
    def value(self) -> str:
        return self._value


class MetricsDisplay(Static):
    """Metrics display widget (alias)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ExecutionProgress(Static):
    """Execution progress widget."""
    def __init__(self, title: str = "", **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._progress: float = 0.0
    
    @property
    def progress(self) -> float:
        return self._progress


class StatusIndicator(Static):
    """Status indicator widget."""
    
    ICONS = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "ℹ",
        "pending": "⏳",
        "unknown": "?",
    }
    
    def __init__(self, status: str = "unknown", label: str = "", **kwargs):
        super().__init__(**kwargs)
        self._status = status
        self._label = label
    
    @property
    def status(self) -> str:
        return self._status


class HelpModal(ModalScreen):
    """Help modal widget."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ConfigInput(Static):
    """Config input widget."""
    def __init__(self, key: str = "", label: str = "", value: str = "", **kwargs):
        super().__init__(**kwargs)
        self._key = key
        self._label = label
        self._value = value


class ConfigToggle(Static):
    """Config toggle widget."""
    def __init__(self, key: str = "", label: str = "", value: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._key = key
        self._label = label
        self._value = value


class ExecutionCard(Static):
    """Execution card widget."""
    def __init__(
        self, 
        execution_id: str = "", 
        backend: str = "", 
        status: str = "",
        duration_ms: float = 0.0,
        timestamp: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._id = execution_id
        self._backend = backend
        self._status = status
        self._duration_ms = duration_ms
        self._timestamp = timestamp


class ProgressBar(Static):
    """Progress bar widget."""
    def __init__(self, label: str = "Progress", total: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._total = total
        self._current = 0.0

__all__ = [
    # Diff Viewer
    "DiffViewer",
    "SideBySideDiffViewer",
    "InlineDiffViewer",
    "DiffViewMode",
    "DiffLine",
    # Test Results Display
    "TestResultsDisplay",
    "TestProgressBar",
    "TestCategoryWidget",
    "TestResultItem",
    "TestSummaryWidget",
    "MeasurementResultsWidget",
    # Change History (Phase 6)
    "ChangeHistoryWidget",
    "ChangeHistoryItem",
    "ChangeTimelineWidget",
    "ChangeSummaryTable",
    "ApprovalProgressWidget",
    # Phase 2: Agent UI/UX Enhanced Widgets
    "WrappedMessage",
    "ChatMessageBubble",
    "WordWrappedRichLog",
    "ResizablePanelContainer",
    "ResizeHandle",
    "CollapsibleStatsPanel",
    "AgentStats",
    "StatsCard",
    "AgentHeader",
    "ToolExecutionCard",
    "InputSection",
    # Phase 3: Enhanced Execution Monitor
    "EnhancedStatusBar",
    "SessionSelector",
    "TerminalDashboard",
    "EnhancedOutputPanel",
    "EnhancedTerminalView",
    "EnhancedExecutionMonitor",
    # Backward compatibility widgets
    "StatusLevel",
    "BackendStatus",
    "StatusItem",
    "LogEntry",
    "BackendInfo",
    "StatusPanel",
    "LogPanel",
    "LogViewer",
    "BackendCard",
    "ResultsTable",
    "ExecutionTimer",
    "MetricDisplay",
    "MetricsDisplay",
    "ExecutionProgress",
    "StatusIndicator",
    "HelpModal",
    "ConfigInput",
    "ConfigToggle",
    "ExecutionCard",
    "ProgressBar",
]
