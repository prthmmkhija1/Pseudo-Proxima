"""Proxima TUI Viewers Components Package.

Contains viewer widgets for displaying content.
"""

from .log_viewer import LogViewer
from .result_viewer import ResultViewer
from .diff_viewer import DiffViewer, DiffLine, DiffHunk
from .code_viewer import CodeViewer
from .ai_thinking_viewer import (
    AIThinkingViewer,
    ThinkingPhase,
    MessageRole,
    ThinkingEntry,
    ThinkingStats,
    create_thinking_panel,
)

__all__ = [
    "LogViewer",
    "ResultViewer",
    "DiffViewer",
    "DiffLine",
    "DiffHunk",
    "CodeViewer",
    # AI Thinking
    "AIThinkingViewer",
    "ThinkingPhase",
    "MessageRole",
    "ThinkingEntry",
    "ThinkingStats",
    "create_thinking_panel",
]
