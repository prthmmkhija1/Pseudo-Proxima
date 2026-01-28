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
    def __init__(self, max_lines: int = 100, **kwargs):
        super().__init__(**kwargs)
        self._max_lines = max_lines
        self._entries: List[LogEntry] = []


class BackendCard(Static):
    """Backend card widget."""
    def __init__(self, info: Optional[BackendInfo] = None, **kwargs):
        super().__init__(**kwargs)
        self._info = info


class ResultsTable(Static):
    """Results table widget."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ExecutionTimer(Static):
    """Execution timer widget."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start_time: Optional[float] = None


class MetricDisplay(Static):
    """Metric display widget."""
    def __init__(self, label: str = "", value: str = "", **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._value = value


class MetricsDisplay(Static):
    """Metrics display widget (alias)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ExecutionProgress(Static):
    """Execution progress widget."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class StatusIndicator(Static):
    """Status indicator widget."""
    def __init__(self, status: str = "unknown", **kwargs):
        super().__init__(**kwargs)
        self._status = status


class HelpModal(ModalScreen):
    """Help modal widget."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ConfigInput(Static):
    """Config input widget."""
    def __init__(self, label: str = "", value: str = "", **kwargs):
        super().__init__(**kwargs)


class ConfigToggle(Static):
    """Config toggle widget."""
    def __init__(self, label: str = "", value: bool = False, **kwargs):
        super().__init__(**kwargs)


class ExecutionCard(Static):
    """Execution card widget."""
    def __init__(self, execution_id: str = "", backend: str = "", status: str = "", **kwargs):
        super().__init__(**kwargs)


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
