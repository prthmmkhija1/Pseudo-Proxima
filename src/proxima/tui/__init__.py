"""Step 6.1: Terminal UI - Textual-based TUI for Proxima.

Screens:
1. Dashboard   - System status, recent executions
2. Execution   - Real-time progress, logs
3. Configuration - Settings management
4. Results     - Browse and analyze results
5. Backends    - Backend status and management

Design Principles:
- Keyboard-first navigation
- Responsive to terminal size
- Consistent color theme
- Contextual help (press ? for help)
"""

from .app import ProximaApp
from .screens import (
    BackendsScreen,
    ConfigurationScreen,
    DashboardScreen,
    ExecutionScreen,
    ResultsScreen,
)
from .widgets import (
    BackendCard,
    HelpModal,
    LogViewer,
    ProgressBar,
    ResultsTable,
    StatusPanel,
)

__all__ = [
    # Main app
    "ProximaApp",
    # Screens
    "DashboardScreen",
    "ExecutionScreen",
    "ConfigurationScreen",
    "ResultsScreen",
    "BackendsScreen",
    # Widgets
    "StatusPanel",
    "LogViewer",
    "ProgressBar",
    "BackendCard",
    "ResultsTable",
    "HelpModal",
]
