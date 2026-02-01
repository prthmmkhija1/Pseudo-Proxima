"""Proxima TUI Screens Package.

Contains all screen definitions for the TUI.
"""

from .base import BaseScreen
from .dashboard import DashboardScreen
from .execution import ExecutionScreen
from .results import ResultsScreen
from .backends import BackendsScreen
from .settings import SettingsScreen
from .help import HelpScreen
from .benchmark_comparison import BenchmarkComparisonScreen
from .ai_assistant import AIAssistantScreen
from .agent_ai_assistant import AgentAIAssistantScreen

__all__ = [
    "BaseScreen",
    "DashboardScreen",
    "ExecutionScreen",
    "ResultsScreen",
    "BackendsScreen",
    "SettingsScreen",
    "HelpScreen",
    "BenchmarkComparisonScreen",
    "AIAssistantScreen",
    "AgentAIAssistantScreen",
]
