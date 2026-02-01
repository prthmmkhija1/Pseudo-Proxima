"""Agent Telemetry Data Model.

Phase 9: Agent Statistics & Telemetry System

Provides comprehensive metrics tracking for all agent operations:
- LLM metrics (tokens, requests, costs)
- Agent performance (uptime, success rates)
- Terminal statistics
- Git operations
- File operations
- Backend building
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TypeVar, Generic
import json

from proxima.utils.logging import get_logger

logger = get_logger("agent.telemetry")


class MetricCategory(Enum):
    """Categories of metrics."""
    LLM = "llm"
    PERFORMANCE = "performance"
    TERMINAL = "terminal"
    GIT = "git"
    FILE = "file"
    BUILD = "build"
    SYSTEM = "system"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"      # Cumulative value (always increases)
    GAUGE = "gauge"          # Current value (can go up/down)
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"           # Values per time unit


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: float
    timestamp: float  # Unix timestamp
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
        }


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    category: MetricCategory
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)


@dataclass
class LLMMetrics:
    """LLM-specific metrics."""
    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests_sent: int = 0
    requests_failed: int = 0
    total_response_time_ms: float = 0.0
    cost_estimate: float = 0.0
    
    @property
    def avg_response_time_ms(self) -> float:
        """Average response time in milliseconds."""
        if self.requests_sent == 0:
            return 0.0
        return self.total_response_time_ms / self.requests_sent
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        total = self.requests_sent
        if total == 0:
            return 100.0
        return ((total - self.requests_failed) / total) * 100
    
    @property
    def tokens_per_second(self) -> float:
        """Token rate."""
        if self.total_response_time_ms == 0:
            return 0.0
        return (self.total_tokens / self.total_response_time_ms) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "requests_sent": self.requests_sent,
            "requests_failed": self.requests_failed,
            "avg_response_time_ms": self.avg_response_time_ms,
            "success_rate": self.success_rate,
            "tokens_per_second": self.tokens_per_second,
            "cost_estimate": self.cost_estimate,
        }


@dataclass
class PerformanceMetrics:
    """Agent performance metrics."""
    start_time: float = field(default_factory=time.time)
    messages_processed: int = 0
    tools_executed: int = 0
    tools_by_type: Dict[str, int] = field(default_factory=dict)
    commands_executed: int = 0
    files_modified: int = 0
    errors_encountered: int = 0
    total_tool_time_ms: float = 0.0
    
    @property
    def uptime_seconds(self) -> float:
        """Uptime in seconds."""
        return time.time() - self.start_time
    
    @property
    def uptime_formatted(self) -> str:
        """Formatted uptime string."""
        seconds = int(self.uptime_seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        total = self.tools_executed
        if total == 0:
            return 100.0
        return ((total - self.errors_encountered) / total) * 100
    
    @property
    def avg_tool_time_ms(self) -> float:
        """Average tool execution time."""
        if self.tools_executed == 0:
            return 0.0
        return self.total_tool_time_ms / self.tools_executed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "uptime_formatted": self.uptime_formatted,
            "messages_processed": self.messages_processed,
            "tools_executed": self.tools_executed,
            "tools_by_type": self.tools_by_type,
            "commands_executed": self.commands_executed,
            "files_modified": self.files_modified,
            "errors_encountered": self.errors_encountered,
            "success_rate": self.success_rate,
            "avg_tool_time_ms": self.avg_tool_time_ms,
        }


@dataclass
class TerminalMetrics:
    """Terminal operation metrics."""
    active_terminals: int = 0
    completed_processes: int = 0
    failed_processes: int = 0
    total_output_lines: int = 0
    total_processes: int = 0
    peak_terminal_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "active_terminals": self.active_terminals,
            "completed_processes": self.completed_processes,
            "failed_processes": self.failed_processes,
            "total_output_lines": self.total_output_lines,
            "total_processes": self.total_processes,
            "peak_terminal_count": self.peak_terminal_count,
        }


@dataclass
class GitMetrics:
    """Git operation metrics."""
    commits_made: int = 0
    branches_created: int = 0
    files_staged: int = 0
    pushes_executed: int = 0
    pulls_executed: int = 0
    merge_conflicts_resolved: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "commits_made": self.commits_made,
            "branches_created": self.branches_created,
            "files_staged": self.files_staged,
            "pushes_executed": self.pushes_executed,
            "pulls_executed": self.pulls_executed,
            "merge_conflicts_resolved": self.merge_conflicts_resolved,
        }


@dataclass
class FileMetrics:
    """File operation metrics."""
    files_read: int = 0
    files_written: int = 0
    files_deleted: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    search_operations: int = 0
    
    @property
    def bytes_read_formatted(self) -> str:
        """Formatted bytes read."""
        return format_bytes(self.bytes_read)
    
    @property
    def bytes_written_formatted(self) -> str:
        """Formatted bytes written."""
        return format_bytes(self.bytes_written)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files_read": self.files_read,
            "files_written": self.files_written,
            "files_deleted": self.files_deleted,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "bytes_read_formatted": self.bytes_read_formatted,
            "bytes_written_formatted": self.bytes_written_formatted,
            "search_operations": self.search_operations,
        }


@dataclass
class BuildMetrics:
    """Backend build metrics."""
    builds_initiated: int = 0
    builds_succeeded: int = 0
    builds_failed: int = 0
    total_build_time_seconds: float = 0.0
    
    @property
    def avg_build_time_seconds(self) -> float:
        """Average build time."""
        total = self.builds_succeeded + self.builds_failed
        if total == 0:
            return 0.0
        return self.total_build_time_seconds / total
    
    @property
    def success_rate(self) -> float:
        """Build success rate."""
        total = self.builds_initiated
        if total == 0:
            return 100.0
        return (self.builds_succeeded / total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "builds_initiated": self.builds_initiated,
            "builds_succeeded": self.builds_succeeded,
            "builds_failed": self.builds_failed,
            "total_build_time_seconds": self.total_build_time_seconds,
            "avg_build_time_seconds": self.avg_build_time_seconds,
            "success_rate": self.success_rate,
        }


@dataclass
class TelemetrySnapshot:
    """Complete telemetry snapshot."""
    timestamp: str
    llm: LLMMetrics
    performance: PerformanceMetrics
    terminal: TerminalMetrics
    git: GitMetrics
    file: FileMetrics
    build: BuildMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "llm": self.llm.to_dict(),
            "performance": self.performance.to_dict(),
            "terminal": self.terminal.to_dict(),
            "git": self.git.to_dict(),
            "file": self.file.to_dict(),
            "build": self.build.to_dict(),
        }


class CircularBuffer(Generic[TypeVar('T')]):
    """Thread-safe circular buffer for storing recent values."""
    
    def __init__(self, maxlen: int = 100):
        """Initialize buffer."""
        self._buffer: Deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
    
    def append(self, value: Any) -> None:
        """Append value."""
        with self._lock:
            self._buffer.append(value)
    
    def get_all(self) -> List[Any]:
        """Get all values."""
        with self._lock:
            return list(self._buffer)
    
    def get_last(self, n: int = 1) -> List[Any]:
        """Get last n values."""
        with self._lock:
            return list(self._buffer)[-n:]
    
    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        """Get length."""
        with self._lock:
            return len(self._buffer)


class AgentTelemetry:
    """Main telemetry collector for agent operations.
    
    Features:
    - Real-time metric collection
    - Thread-safe operations
    - Historical data storage
    - Snapshot generation
    - Event callbacks
    
    Example:
        >>> telemetry = AgentTelemetry()
        >>> 
        >>> # Record LLM request
        >>> telemetry.record_llm_request(
        ...     provider="openai",
        ...     model="gpt-4",
        ...     prompt_tokens=100,
        ...     completion_tokens=50,
        ...     response_time_ms=1200,
        ... )
        >>> 
        >>> # Get current snapshot
        >>> snapshot = telemetry.get_snapshot()
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        persist_path: Optional[str] = None,
    ):
        """Initialize telemetry.
        
        Args:
            history_size: Number of recent operations to keep
            persist_path: Path to persist telemetry data
        """
        self._lock = threading.RLock()
        
        # Metric stores
        self.llm = LLMMetrics()
        self.performance = PerformanceMetrics()
        self.terminal = TerminalMetrics()
        self.git = GitMetrics()
        self.file = FileMetrics()
        self.build = BuildMetrics()
        
        # Historical data
        self._history_size = history_size
        self._operation_history: CircularBuffer = CircularBuffer(history_size)
        self._response_times: CircularBuffer = CircularBuffer(100)
        self._error_history: CircularBuffer = CircularBuffer(100)
        
        # Persistence
        self._persist_path = Path(persist_path) if persist_path else None
        
        # Callbacks
        self._on_update: List[Callable[[str, Any], None]] = []
        self._on_alert: List[Callable[[str, str, Any], None]] = []
        
        logger.info("AgentTelemetry initialized")
    
    def _notify_update(self, metric_name: str, value: Any) -> None:
        """Notify listeners of metric update."""
        for callback in self._on_update:
            try:
                callback(metric_name, value)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    # ========== LLM Metrics ==========
    
    def set_llm_config(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 0,
    ) -> None:
        """Set LLM configuration."""
        with self._lock:
            self.llm.provider = provider
            self.llm.model = model
            self.llm.temperature = temperature
            self.llm.max_tokens = max_tokens
        self._notify_update("llm.config", {"provider": provider, "model": model})
    
    def record_llm_request(
        self,
        provider: str = "",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        response_time_ms: float = 0.0,
        success: bool = True,
        cost: float = 0.0,
    ) -> None:
        """Record an LLM request."""
        with self._lock:
            if provider:
                self.llm.provider = provider
            if model:
                self.llm.model = model
            
            self.llm.prompt_tokens += prompt_tokens
            self.llm.completion_tokens += completion_tokens
            self.llm.total_tokens += prompt_tokens + completion_tokens
            self.llm.requests_sent += 1
            self.llm.total_response_time_ms += response_time_ms
            self.llm.cost_estimate += cost
            
            if not success:
                self.llm.requests_failed += 1
            
            self._response_times.append(response_time_ms)
        
        self._notify_update("llm.request", {
            "tokens": prompt_tokens + completion_tokens,
            "response_time_ms": response_time_ms,
        })
    
    # ========== Performance Metrics ==========
    
    def record_message(self) -> None:
        """Record a processed message."""
        with self._lock:
            self.performance.messages_processed += 1
        self._notify_update("performance.message", self.performance.messages_processed)
    
    def record_tool_execution(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool = True,
    ) -> None:
        """Record tool execution."""
        with self._lock:
            self.performance.tools_executed += 1
            self.performance.total_tool_time_ms += duration_ms
            
            # Track by type
            count = self.performance.tools_by_type.get(tool_name, 0)
            self.performance.tools_by_type[tool_name] = count + 1
            
            if not success:
                self.performance.errors_encountered += 1
                self._error_history.append({
                    "tool": tool_name,
                    "timestamp": time.time(),
                })
        
        self._operation_history.append({
            "type": "tool",
            "name": tool_name,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": time.time(),
        })
        
        self._notify_update("performance.tool", {"tool": tool_name, "duration_ms": duration_ms})
    
    def record_command_execution(self, success: bool = True) -> None:
        """Record command execution."""
        with self._lock:
            self.performance.commands_executed += 1
            if not success:
                self.performance.errors_encountered += 1
        self._notify_update("performance.command", self.performance.commands_executed)
    
    def record_file_modification(self) -> None:
        """Record file modification."""
        with self._lock:
            self.performance.files_modified += 1
        self._notify_update("performance.file_modified", self.performance.files_modified)
    
    def record_error(self, error_type: str = "unknown") -> None:
        """Record an error."""
        with self._lock:
            self.performance.errors_encountered += 1
            self._error_history.append({
                "type": error_type,
                "timestamp": time.time(),
            })
        self._notify_update("performance.error", error_type)
    
    # ========== Terminal Metrics ==========
    
    def update_terminal_count(self, active: int) -> None:
        """Update active terminal count."""
        with self._lock:
            self.terminal.active_terminals = active
            if active > self.terminal.peak_terminal_count:
                self.terminal.peak_terminal_count = active
        self._notify_update("terminal.active", active)
    
    def record_terminal_process(self, success: bool = True) -> None:
        """Record terminal process completion."""
        with self._lock:
            self.terminal.total_processes += 1
            if success:
                self.terminal.completed_processes += 1
            else:
                self.terminal.failed_processes += 1
        self._notify_update("terminal.process", {"success": success})
    
    def record_terminal_output(self, lines: int) -> None:
        """Record terminal output lines."""
        with self._lock:
            self.terminal.total_output_lines += lines
        self._notify_update("terminal.output", lines)
    
    # ========== Git Metrics ==========
    
    def record_git_commit(self) -> None:
        """Record git commit."""
        with self._lock:
            self.git.commits_made += 1
        self._notify_update("git.commit", self.git.commits_made)
    
    def record_git_branch_create(self) -> None:
        """Record branch creation."""
        with self._lock:
            self.git.branches_created += 1
        self._notify_update("git.branch_create", self.git.branches_created)
    
    def record_git_stage(self, file_count: int = 1) -> None:
        """Record files staged."""
        with self._lock:
            self.git.files_staged += file_count
        self._notify_update("git.stage", self.git.files_staged)
    
    def record_git_push(self) -> None:
        """Record git push."""
        with self._lock:
            self.git.pushes_executed += 1
        self._notify_update("git.push", self.git.pushes_executed)
    
    def record_git_pull(self) -> None:
        """Record git pull."""
        with self._lock:
            self.git.pulls_executed += 1
        self._notify_update("git.pull", self.git.pulls_executed)
    
    def record_merge_conflict_resolved(self) -> None:
        """Record merge conflict resolution."""
        with self._lock:
            self.git.merge_conflicts_resolved += 1
        self._notify_update("git.conflict_resolved", self.git.merge_conflicts_resolved)
    
    # ========== File Metrics ==========
    
    def record_file_read(self, bytes_count: int = 0) -> None:
        """Record file read."""
        with self._lock:
            self.file.files_read += 1
            self.file.bytes_read += bytes_count
        self._notify_update("file.read", {"bytes": bytes_count})
    
    def record_file_write(self, bytes_count: int = 0) -> None:
        """Record file write."""
        with self._lock:
            self.file.files_written += 1
            self.file.bytes_written += bytes_count
        self._notify_update("file.write", {"bytes": bytes_count})
    
    def record_file_delete(self) -> None:
        """Record file deletion."""
        with self._lock:
            self.file.files_deleted += 1
        self._notify_update("file.delete", self.file.files_deleted)
    
    def record_search_operation(self) -> None:
        """Record search operation."""
        with self._lock:
            self.file.search_operations += 1
        self._notify_update("file.search", self.file.search_operations)
    
    # ========== Build Metrics ==========
    
    def record_build_start(self) -> None:
        """Record build start."""
        with self._lock:
            self.build.builds_initiated += 1
        self._notify_update("build.start", self.build.builds_initiated)
    
    def record_build_complete(self, success: bool, duration_seconds: float) -> None:
        """Record build completion."""
        with self._lock:
            self.build.total_build_time_seconds += duration_seconds
            if success:
                self.build.builds_succeeded += 1
            else:
                self.build.builds_failed += 1
        self._notify_update("build.complete", {"success": success, "duration": duration_seconds})
    
    # ========== Snapshots & History ==========
    
    def get_snapshot(self) -> TelemetrySnapshot:
        """Get current telemetry snapshot."""
        with self._lock:
            return TelemetrySnapshot(
                timestamp=datetime.now().isoformat(),
                llm=LLMMetrics(
                    provider=self.llm.provider,
                    model=self.llm.model,
                    temperature=self.llm.temperature,
                    max_tokens=self.llm.max_tokens,
                    prompt_tokens=self.llm.prompt_tokens,
                    completion_tokens=self.llm.completion_tokens,
                    total_tokens=self.llm.total_tokens,
                    requests_sent=self.llm.requests_sent,
                    requests_failed=self.llm.requests_failed,
                    total_response_time_ms=self.llm.total_response_time_ms,
                    cost_estimate=self.llm.cost_estimate,
                ),
                performance=PerformanceMetrics(
                    start_time=self.performance.start_time,
                    messages_processed=self.performance.messages_processed,
                    tools_executed=self.performance.tools_executed,
                    tools_by_type=dict(self.performance.tools_by_type),
                    commands_executed=self.performance.commands_executed,
                    files_modified=self.performance.files_modified,
                    errors_encountered=self.performance.errors_encountered,
                    total_tool_time_ms=self.performance.total_tool_time_ms,
                ),
                terminal=TerminalMetrics(
                    active_terminals=self.terminal.active_terminals,
                    completed_processes=self.terminal.completed_processes,
                    failed_processes=self.terminal.failed_processes,
                    total_output_lines=self.terminal.total_output_lines,
                    total_processes=self.terminal.total_processes,
                    peak_terminal_count=self.terminal.peak_terminal_count,
                ),
                git=GitMetrics(
                    commits_made=self.git.commits_made,
                    branches_created=self.git.branches_created,
                    files_staged=self.git.files_staged,
                    pushes_executed=self.git.pushes_executed,
                    pulls_executed=self.git.pulls_executed,
                    merge_conflicts_resolved=self.git.merge_conflicts_resolved,
                ),
                file=FileMetrics(
                    files_read=self.file.files_read,
                    files_written=self.file.files_written,
                    files_deleted=self.file.files_deleted,
                    bytes_read=self.file.bytes_read,
                    bytes_written=self.file.bytes_written,
                    search_operations=self.file.search_operations,
                ),
                build=BuildMetrics(
                    builds_initiated=self.build.builds_initiated,
                    builds_succeeded=self.build.builds_succeeded,
                    builds_failed=self.build.builds_failed,
                    total_build_time_seconds=self.build.total_build_time_seconds,
                ),
            )
    
    def get_recent_operations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent operations."""
        return self._operation_history.get_last(limit)
    
    def get_recent_errors(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self._error_history.get_last(limit)
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """Get response time statistics."""
        times = self._response_times.get_all()
        if not times:
            return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p90": 0, "p99": 0}
        
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / n,
            "p50": sorted_times[int(n * 0.5)] if n > 0 else 0,
            "p90": sorted_times[int(n * 0.9)] if n > 0 else 0,
            "p99": sorted_times[int(n * 0.99)] if n > 0 else 0,
        }
    
    # ========== Callbacks ==========
    
    def on_update(self, callback: Callable[[str, Any], None]) -> None:
        """Register update callback."""
        self._on_update.append(callback)
    
    def on_alert(self, callback: Callable[[str, str, Any], None]) -> None:
        """Register alert callback."""
        self._on_alert.append(callback)
    
    # ========== Persistence ==========
    
    def save(self, path: Optional[str] = None) -> bool:
        """Save telemetry to file."""
        save_path = Path(path) if path else self._persist_path
        if not save_path:
            return False
        
        try:
            snapshot = self.get_snapshot()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                json.dumps(snapshot.to_dict(), indent=2),
                encoding="utf-8"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save telemetry: {e}")
            return False
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.llm = LLMMetrics()
            self.performance = PerformanceMetrics()
            self.terminal = TerminalMetrics()
            self.git = GitMetrics()
            self.file = FileMetrics()
            self.build = BuildMetrics()
            self._operation_history.clear()
            self._response_times.clear()
            self._error_history.clear()
        
        logger.info("Telemetry reset")


# ========== Utility Functions ==========

def format_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return str(int(value))


def format_bytes(bytes_value: int) -> str:
    """Format bytes with appropriate unit."""
    if bytes_value >= 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024 * 1024):.1f} GB"
    elif bytes_value >= 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f} MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.1f} KB"
    else:
        return f"{bytes_value} B"


def format_duration(seconds: float) -> str:
    """Format duration in appropriate units."""
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    elif seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    elif seconds >= 1:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds * 1000:.0f}ms"


def format_percentage(value: float) -> str:
    """Format percentage."""
    return f"{value:.1f}%"


def format_currency(value: float, symbol: str = "$") -> str:
    """Format currency."""
    return f"{symbol}{value:.2f}"


# ========== Global Instance ==========

_telemetry: Optional[AgentTelemetry] = None


def get_telemetry() -> AgentTelemetry:
    """Get the global AgentTelemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = AgentTelemetry()
    return _telemetry


def reset_telemetry() -> None:
    """Reset the global telemetry instance."""
    global _telemetry
    if _telemetry:
        _telemetry.reset()
