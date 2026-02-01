"""Metric Collector Module.

Phase 9: Agent Statistics & Telemetry System

Provides instrumentation and metric collection:
- Decorator-based instrumentation
- Async metric collection
- Background aggregation
- Metric emission events
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
import inspect

from proxima.utils.logging import get_logger

logger = get_logger("agent.metric_collector")

F = TypeVar('F', bound=Callable[..., Any])


class OperationType(Enum):
    """Types of operations to track."""
    LLM_REQUEST = "llm_request"
    TOOL_EXECUTION = "tool_execution"
    COMMAND_EXECUTION = "command_execution"
    FILE_OPERATION = "file_operation"
    GIT_OPERATION = "git_operation"
    BUILD_OPERATION = "build_operation"
    TERMINAL_OPERATION = "terminal_operation"
    GENERIC = "generic"


@dataclass
class OperationMetric:
    """Metric for a single operation."""
    operation_type: OperationType
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark operation as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.operation_type.value,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedMetric:
    """Aggregated metric over time window."""
    name: str
    count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = field(default_factory=time.time)
    
    @property
    def avg_duration_ms(self) -> float:
        """Average duration."""
        if self.count == 0:
            return 0.0
        return self.total_duration_ms / self.count
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.count == 0:
            return 100.0
        return (self.success_count / self.count) * 100
    
    def record(self, duration_ms: float, success: bool) -> None:
        """Record a new value."""
        self.count += 1
        self.total_duration_ms += duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.last_updated = time.time()
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms if self.count > 0 else 0,
            "max_duration_ms": self.max_duration_ms,
            "success_rate": self.success_rate,
            "last_updated": self.last_updated,
        }


class MetricEmitter:
    """Emits metric events to listeners."""
    
    def __init__(self):
        """Initialize emitter."""
        self._listeners: List[Callable[[OperationMetric], None]] = []
        self._async_listeners: List[Callable[[OperationMetric], Awaitable[None]]] = []
        self._lock = threading.Lock()
    
    def add_listener(
        self,
        listener: Union[Callable[[OperationMetric], None], Callable[[OperationMetric], Awaitable[None]]],
    ) -> None:
        """Add a metric listener."""
        with self._lock:
            if asyncio.iscoroutinefunction(listener):
                self._async_listeners.append(listener)
            else:
                self._listeners.append(listener)
    
    def remove_listener(
        self,
        listener: Union[Callable[[OperationMetric], None], Callable[[OperationMetric], Awaitable[None]]],
    ) -> None:
        """Remove a metric listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
            if listener in self._async_listeners:
                self._async_listeners.remove(listener)
    
    def emit(self, metric: OperationMetric) -> None:
        """Emit metric to all listeners."""
        with self._lock:
            listeners = list(self._listeners)
            async_listeners = list(self._async_listeners)
        
        # Sync listeners
        for listener in listeners:
            try:
                listener(metric)
            except Exception as e:
                logger.error(f"Metric listener error: {e}")
        
        # Async listeners (schedule in event loop)
        if async_listeners:
            try:
                loop = asyncio.get_event_loop()
                for listener in async_listeners:
                    loop.create_task(self._safe_async_emit(listener, metric))
            except RuntimeError:
                # No event loop running
                pass
    
    async def _safe_async_emit(
        self,
        listener: Callable[[OperationMetric], Awaitable[None]],
        metric: OperationMetric,
    ) -> None:
        """Safely emit to async listener."""
        try:
            await listener(metric)
        except Exception as e:
            logger.error(f"Async metric listener error: {e}")


class MetricCollector:
    """Collects and aggregates metrics.
    
    Features:
    - Decorator-based instrumentation
    - Context manager for manual tracking
    - Automatic aggregation
    - Background processing
    
    Example:
        >>> collector = MetricCollector()
        >>> 
        >>> # Using decorator
        >>> @collector.track(OperationType.TOOL_EXECUTION, "my_tool")
        >>> async def my_tool():
        ...     # Tool implementation
        ...     pass
        >>> 
        >>> # Using context manager
        >>> with collector.track_operation(OperationType.FILE_OPERATION, "read_file"):
        ...     # Read file
        ...     pass
    """
    
    def __init__(
        self,
        telemetry: Optional[Any] = None,  # AgentTelemetry
        emit_interval: float = 1.0,
    ):
        """Initialize collector.
        
        Args:
            telemetry: Optional telemetry instance to record to
            emit_interval: Interval for background emission
        """
        self._telemetry = telemetry
        self._emitter = MetricEmitter()
        self._lock = threading.Lock()
        
        # Aggregated metrics by operation type and name
        self._aggregated: Dict[str, AggregatedMetric] = {}
        
        # Pending metrics for batch processing
        self._pending: List[OperationMetric] = []
        
        # Background processing
        self._emit_interval = emit_interval
        self._running = False
        self._background_task: Optional[asyncio.Task] = None
        
        logger.info("MetricCollector initialized")
    
    def set_telemetry(self, telemetry: Any) -> None:
        """Set telemetry instance."""
        self._telemetry = telemetry
    
    @property
    def emitter(self) -> MetricEmitter:
        """Get the metric emitter."""
        return self._emitter
    
    def _record_to_telemetry(self, metric: OperationMetric) -> None:
        """Record metric to telemetry."""
        if not self._telemetry:
            return
        
        try:
            if metric.operation_type == OperationType.LLM_REQUEST:
                self._telemetry.record_llm_request(
                    response_time_ms=metric.duration_ms or 0,
                    success=metric.success,
                    **metric.metadata,
                )
            elif metric.operation_type == OperationType.TOOL_EXECUTION:
                self._telemetry.record_tool_execution(
                    tool_name=metric.name,
                    duration_ms=metric.duration_ms or 0,
                    success=metric.success,
                )
            elif metric.operation_type == OperationType.COMMAND_EXECUTION:
                self._telemetry.record_command_execution(success=metric.success)
            elif metric.operation_type == OperationType.FILE_OPERATION:
                if "read" in metric.name.lower():
                    self._telemetry.record_file_read(metric.metadata.get("bytes", 0))
                elif "write" in metric.name.lower():
                    self._telemetry.record_file_write(metric.metadata.get("bytes", 0))
                elif "delete" in metric.name.lower():
                    self._telemetry.record_file_delete()
            elif metric.operation_type == OperationType.GIT_OPERATION:
                if "commit" in metric.name.lower():
                    self._telemetry.record_git_commit()
                elif "push" in metric.name.lower():
                    self._telemetry.record_git_push()
                elif "pull" in metric.name.lower():
                    self._telemetry.record_git_pull()
            elif metric.operation_type == OperationType.BUILD_OPERATION:
                if metric.end_time and metric.start_time:
                    duration = metric.end_time - metric.start_time
                    self._telemetry.record_build_complete(
                        success=metric.success,
                        duration_seconds=duration,
                    )
            elif metric.operation_type == OperationType.TERMINAL_OPERATION:
                self._telemetry.record_terminal_process(success=metric.success)
        except Exception as e:
            logger.error(f"Failed to record to telemetry: {e}")
    
    def _aggregate_metric(self, metric: OperationMetric) -> None:
        """Aggregate metric."""
        key = f"{metric.operation_type.value}:{metric.name}"
        
        with self._lock:
            if key not in self._aggregated:
                self._aggregated[key] = AggregatedMetric(name=metric.name)
            
            self._aggregated[key].record(
                duration_ms=metric.duration_ms or 0,
                success=metric.success,
            )
    
    def record(self, metric: OperationMetric) -> None:
        """Record a completed metric."""
        # Aggregate
        self._aggregate_metric(metric)
        
        # Record to telemetry
        self._record_to_telemetry(metric)
        
        # Emit to listeners
        self._emitter.emit(metric)
    
    @contextmanager
    def track_operation(
        self,
        operation_type: OperationType,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracking operations.
        
        Example:
            >>> with collector.track_operation(OperationType.FILE_OPERATION, "read"):
            ...     content = file.read()
        """
        metric = OperationMetric(
            operation_type=operation_type,
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
        )
        
        try:
            yield metric
            metric.complete(success=True)
        except Exception as e:
            metric.complete(success=False, error=str(e))
            raise
        finally:
            self.record(metric)
    
    def track(
        self,
        operation_type: OperationType,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[[F], F]:
        """Decorator for tracking function execution.
        
        Example:
            >>> @collector.track(OperationType.TOOL_EXECUTION)
            >>> async def my_tool():
            ...     pass
        """
        def decorator(func: F) -> F:
            operation_name = name or func.__name__
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    metric = OperationMetric(
                        operation_type=operation_type,
                        name=operation_name,
                        start_time=time.time(),
                        metadata=metadata or {},
                    )
                    
                    try:
                        result = await func(*args, **kwargs)
                        metric.complete(success=True)
                        return result
                    except Exception as e:
                        metric.complete(success=False, error=str(e))
                        raise
                    finally:
                        self.record(metric)
                
                return async_wrapper  # type: ignore
            else:
                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    metric = OperationMetric(
                        operation_type=operation_type,
                        name=operation_name,
                        start_time=time.time(),
                        metadata=metadata or {},
                    )
                    
                    try:
                        result = func(*args, **kwargs)
                        metric.complete(success=True)
                        return result
                    except Exception as e:
                        metric.complete(success=False, error=str(e))
                        raise
                    finally:
                        self.record(metric)
                
                return sync_wrapper  # type: ignore
        
        return decorator
    
    def get_aggregated(self, operation_type: Optional[OperationType] = None) -> Dict[str, AggregatedMetric]:
        """Get aggregated metrics."""
        with self._lock:
            if operation_type:
                prefix = f"{operation_type.value}:"
                return {
                    k: v for k, v in self._aggregated.items()
                    if k.startswith(prefix)
                }
            return dict(self._aggregated)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary: Dict[str, Any] = {}
            
            for key, metric in self._aggregated.items():
                op_type, name = key.split(":", 1)
                
                if op_type not in summary:
                    summary[op_type] = {
                        "total_operations": 0,
                        "total_duration_ms": 0,
                        "success_count": 0,
                        "failure_count": 0,
                        "operations": {},
                    }
                
                summary[op_type]["total_operations"] += metric.count
                summary[op_type]["total_duration_ms"] += metric.total_duration_ms
                summary[op_type]["success_count"] += metric.success_count
                summary[op_type]["failure_count"] += metric.failure_count
                summary[op_type]["operations"][name] = metric.to_dict()
            
            return summary
    
    def reset(self) -> None:
        """Reset all aggregated metrics."""
        with self._lock:
            self._aggregated.clear()
            self._pending.clear()
        logger.info("MetricCollector reset")
    
    async def start_background_processing(self) -> None:
        """Start background metric processing."""
        if self._running:
            return
        
        self._running = True
        self._background_task = asyncio.create_task(self._process_loop())
        logger.info("Started background metric processing")
    
    async def stop_background_processing(self) -> None:
        """Stop background processing."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped background metric processing")
    
    async def _process_loop(self) -> None:
        """Background processing loop."""
        while self._running:
            try:
                await asyncio.sleep(self._emit_interval)
                # Process pending metrics
                with self._lock:
                    pending = list(self._pending)
                    self._pending.clear()
                
                for metric in pending:
                    self._emitter.emit(metric)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background processing error: {e}")


# ========== Instrumentation Helpers ==========

def track_llm_request(collector: MetricCollector):
    """Decorator for LLM request tracking."""
    return collector.track(OperationType.LLM_REQUEST)


def track_tool(collector: MetricCollector, tool_name: Optional[str] = None):
    """Decorator for tool execution tracking."""
    return collector.track(OperationType.TOOL_EXECUTION, name=tool_name)


def track_command(collector: MetricCollector):
    """Decorator for command execution tracking."""
    return collector.track(OperationType.COMMAND_EXECUTION)


def track_file_operation(collector: MetricCollector, operation: str = ""):
    """Decorator for file operation tracking."""
    return collector.track(OperationType.FILE_OPERATION, name=operation)


def track_git_operation(collector: MetricCollector, operation: str = ""):
    """Decorator for git operation tracking."""
    return collector.track(OperationType.GIT_OPERATION, name=operation)


def track_build(collector: MetricCollector):
    """Decorator for build operation tracking."""
    return collector.track(OperationType.BUILD_OPERATION)


# ========== Global Instance ==========

_collector: Optional[MetricCollector] = None


def get_metric_collector() -> MetricCollector:
    """Get the global MetricCollector instance."""
    global _collector
    if _collector is None:
        _collector = MetricCollector()
    return _collector


def set_metric_collector(collector: MetricCollector) -> None:
    """Set the global MetricCollector instance."""
    global _collector
    _collector = collector
