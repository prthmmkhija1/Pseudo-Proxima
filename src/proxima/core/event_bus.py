"""Real-Time Event Bus System for Proxima.

Provides a high-performance, asynchronous event bus for real-time
communication between components. Supports:
- Non-blocking message passing via asyncio.Queue
- Event type filtering and subscription
- Weak references to prevent memory leaks
- Circular buffer for event history/replay
- Debouncing for high-frequency events

This is the core infrastructure for real-time execution monitoring.
"""

from __future__ import annotations

import asyncio
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from typing import TypeAlias

# Type aliases for callbacks
SyncCallback = Callable[["Event"], None]
AsyncCallback = Callable[["Event"], Coroutine[Any, Any, None]]
EventCallback = Union[SyncCallback, AsyncCallback]


class EventType(Enum):
    """Event types for the real-time system."""
    
    # Process lifecycle events
    PROCESS_STARTED = auto()
    PROCESS_COMPLETED = auto()
    PROCESS_FAILED = auto()
    PROCESS_KILLED = auto()
    PROCESS_TIMEOUT = auto()
    
    # Output events
    OUTPUT_LINE = auto()
    ERROR_LINE = auto()
    OUTPUT_CHUNK = auto()
    
    # Progress events
    PROGRESS_UPDATE = auto()
    STAGE_CHANGED = auto()
    STAGE_STARTED = auto()
    STAGE_COMPLETED = auto()
    
    # Execution events
    EXECUTION_STARTED = auto()
    EXECUTION_COMPLETED = auto()
    EXECUTION_FAILED = auto()
    EXECUTION_PAUSED = auto()
    EXECUTION_RESUMED = auto()
    EXECUTION_CANCELLED = auto()
    
    # Result events
    RESULT_AVAILABLE = auto()
    RESULT_PARTIAL = auto()
    RESULT_UPDATED = auto()
    
    # Terminal events
    TERMINAL_CREATED = auto()
    TERMINAL_CLOSED = auto()
    TERMINAL_OUTPUT = auto()
    TERMINAL_ERROR = auto()
    TERMINAL_COMMAND = auto()
    
    # Agent events
    AGENT_TOOL_STARTED = auto()
    AGENT_TOOL_COMPLETED = auto()
    AGENT_TOOL_FAILED = auto()
    AGENT_MESSAGE = auto()
    AGENT_THINKING = auto()
    
    # Backend events
    BUILD_STARTED = auto()
    BUILD_PROGRESS = auto()
    BUILD_COMPLETED = auto()
    BUILD_FAILED = auto()
    
    # System events
    SYSTEM_INFO = auto()
    SYSTEM_WARNING = auto()
    SYSTEM_ERROR = auto()
    
    # Custom/generic events
    CUSTOM = auto()


@dataclass(frozen=True)
class Event:
    """Immutable event data structure.
    
    All events carry a type, timestamp, source identifier, and payload.
    Events are immutable to ensure thread-safety and consistency.
    
    Attributes:
        event_type: The type of event
        source_id: Identifier of the event source (process ID, terminal ID, etc.)
        payload: Event-specific data
        timestamp: When the event was created (auto-generated)
        event_id: Unique identifier for this event (auto-generated)
        correlation_id: ID to link related events (e.g., all events from one execution)
        metadata: Additional optional metadata
    """
    event_type: EventType
    source_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def timestamp_str(self) -> str:
        """Get formatted timestamp string."""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3]
    
    @property
    def age_ms(self) -> float:
        """Get event age in milliseconds."""
        return (time.time() - self.timestamp) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.name,
            "source_id": self.source_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_type=EventType[data["event_type"]],
            source_id=data["source_id"],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            event_id=data.get("event_id", str(uuid.uuid4())[:8]),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Subscription:
    """Event subscription with optional filtering."""
    callback: EventCallback
    event_types: Optional[Set[EventType]] = None  # None means all events
    source_filter: Optional[str] = None  # Filter by source_id pattern
    correlation_filter: Optional[str] = None  # Filter by correlation_id
    weak_ref: bool = False  # Use weak reference to callback owner
    priority: int = 0  # Higher priority = called first
    
    def matches(self, event: Event) -> bool:
        """Check if event matches subscription filters."""
        # Type filter
        if self.event_types is not None:
            if event.event_type not in self.event_types:
                return False
        
        # Source filter (supports wildcard prefix matching)
        if self.source_filter is not None:
            if self.source_filter.endswith("*"):
                if not event.source_id.startswith(self.source_filter[:-1]):
                    return False
            elif event.source_id != self.source_filter:
                return False
        
        # Correlation filter
        if self.correlation_filter is not None:
            if event.correlation_id != self.correlation_filter:
                return False
        
        return True


class EventBus:
    """High-performance asynchronous event bus.
    
    Features:
    - Non-blocking event dispatch via asyncio
    - Type-safe event subscriptions
    - Weak reference support to prevent memory leaks
    - Circular buffer for event history (replay capability)
    - Debouncing for high-frequency events
    - Priority-based callback ordering
    
    Example:
        >>> bus = EventBus()
        >>> 
        >>> def on_output(event):
        ...     print(f"Output: {event.payload.get('line')}")
        >>> 
        >>> bus.subscribe(on_output, event_types={EventType.OUTPUT_LINE})
        >>> await bus.start()
        >>> 
        >>> bus.emit(Event(
        ...     event_type=EventType.OUTPUT_LINE,
        ...     source_id="terminal_1",
        ...     payload={"line": "Hello, world!"}
        ... ))
    """
    
    # Default configuration
    DEFAULT_HISTORY_SIZE = 10000
    DEFAULT_BATCH_INTERVAL_MS = 16  # ~60 FPS
    DEFAULT_MAX_QUEUE_SIZE = 50000
    
    def __init__(
        self,
        history_size: int = DEFAULT_HISTORY_SIZE,
        batch_interval_ms: float = DEFAULT_BATCH_INTERVAL_MS,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
    ):
        """Initialize the event bus.
        
        Args:
            history_size: Maximum events to keep in history buffer
            batch_interval_ms: Minimum interval between batched dispatches
            max_queue_size: Maximum pending events before dropping
        """
        self._history_size = history_size
        self._batch_interval_ms = batch_interval_ms
        self._max_queue_size = max_queue_size
        
        # Event queue for async dispatch
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        
        # Circular buffer for event history
        self._history: deque[Event] = deque(maxlen=history_size)
        
        # Subscription management
        self._subscriptions: List[Tuple[str, Subscription]] = []  # (sub_id, subscription)
        self._subscription_counter = 0
        
        # Weak reference tracking for auto-cleanup
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        # State
        self._running = False
        self._dispatch_task: Optional[asyncio.Task] = None
        self._last_dispatch_time = 0.0
        
        # Statistics
        self._stats = {
            "events_emitted": 0,
            "events_dispatched": 0,
            "events_dropped": 0,
            "callbacks_invoked": 0,
            "errors": 0,
        }
        
        # Debouncing state
        self._debounce_state: Dict[str, Tuple[Event, float]] = {}  # key -> (last_event, last_time)
    
    @property
    def is_running(self) -> bool:
        """Check if event bus is running."""
        return self._running
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get event bus statistics."""
        return self._stats.copy()
    
    @property
    def history(self) -> List[Event]:
        """Get event history (oldest to newest)."""
        return list(self._history)
    
    @property
    def pending_count(self) -> int:
        """Get number of pending events in queue."""
        return self._queue.qsize()
    
    async def start(self) -> None:
        """Start the event bus dispatcher."""
        if self._running:
            return
        
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
    
    async def stop(self) -> None:
        """Stop the event bus dispatcher."""
        self._running = False
        
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
            self._dispatch_task = None
    
    def subscribe(
        self,
        callback: EventCallback,
        event_types: Optional[Set[EventType]] = None,
        source_filter: Optional[str] = None,
        correlation_filter: Optional[str] = None,
        weak_ref: bool = False,
        priority: int = 0,
    ) -> str:
        """Subscribe to events.
        
        Args:
            callback: Function to call when event matches
            event_types: Set of event types to receive (None = all)
            source_filter: Filter by source_id (supports * wildcard at end)
            correlation_filter: Filter by correlation_id
            weak_ref: Use weak reference (auto-cleanup when owner dies)
            priority: Higher priority callbacks are called first
            
        Returns:
            Subscription ID for later unsubscription
        """
        self._subscription_counter += 1
        sub_id = f"sub_{self._subscription_counter}"
        
        subscription = Subscription(
            callback=callback,
            event_types=event_types,
            source_filter=source_filter,
            correlation_filter=correlation_filter,
            weak_ref=weak_ref,
            priority=priority,
        )
        
        # Store weak reference if requested
        if weak_ref and hasattr(callback, "__self__"):
            self._weak_refs[sub_id] = weakref.ref(callback.__self__)
        
        # Insert sorted by priority (highest first)
        inserted = False
        for i, (_, existing_sub) in enumerate(self._subscriptions):
            if subscription.priority > existing_sub.priority:
                self._subscriptions.insert(i, (sub_id, subscription))
                inserted = True
                break
        
        if not inserted:
            self._subscriptions.append((sub_id, subscription))
        
        return sub_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.
        
        Args:
            subscription_id: ID returned from subscribe()
            
        Returns:
            True if subscription was found and removed
        """
        for i, (sub_id, _) in enumerate(self._subscriptions):
            if sub_id == subscription_id:
                self._subscriptions.pop(i)
                self._weak_refs.pop(sub_id, None)
                return True
        return False
    
    def emit(self, event: Event) -> bool:
        """Emit an event (non-blocking).
        
        Args:
            event: Event to emit
            
        Returns:
            True if event was queued, False if dropped
        """
        try:
            self._queue.put_nowait(event)
            self._stats["events_emitted"] += 1
            return True
        except asyncio.QueueFull:
            self._stats["events_dropped"] += 1
            return False
    
    def emit_sync(
        self,
        event_type: EventType,
        source_id: str,
        payload: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Convenience method to emit event synchronously.
        
        Args:
            event_type: Type of event
            source_id: Source identifier
            payload: Event payload
            correlation_id: Correlation ID for related events
            **kwargs: Additional metadata
            
        Returns:
            True if event was queued
        """
        event = Event(
            event_type=event_type,
            source_id=source_id,
            payload=payload or {},
            correlation_id=correlation_id,
            metadata=kwargs,
        )
        return self.emit(event)
    
    def emit_debounced(
        self,
        event: Event,
        debounce_key: str,
        debounce_ms: float = 100,
    ) -> bool:
        """Emit event with debouncing.
        
        Only the last event within the debounce window is actually emitted.
        Useful for high-frequency events like progress updates.
        
        Args:
            event: Event to emit
            debounce_key: Key for debouncing (same key = same debounce group)
            debounce_ms: Debounce window in milliseconds
            
        Returns:
            True if event was emitted, False if debounced
        """
        now = time.time()
        last_event, last_time = self._debounce_state.get(debounce_key, (None, 0))
        
        if (now - last_time) * 1000 >= debounce_ms:
            # Emit immediately
            self._debounce_state[debounce_key] = (event, now)
            return self.emit(event)
        else:
            # Update stored event but don't emit yet
            self._debounce_state[debounce_key] = (event, last_time)
            return False
    
    async def flush_debounced(self) -> int:
        """Flush all pending debounced events.
        
        Returns:
            Number of events flushed
        """
        flushed = 0
        now = time.time()
        
        for key, (event, _) in list(self._debounce_state.items()):
            if self.emit(event):
                flushed += 1
            del self._debounce_state[key]
        
        return flushed
    
    def get_history(
        self,
        event_types: Optional[Set[EventType]] = None,
        source_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        since_timestamp: Optional[float] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get filtered events from history.
        
        Args:
            event_types: Filter by event types
            source_id: Filter by source
            correlation_id: Filter by correlation
            since_timestamp: Only events after this time
            limit: Maximum events to return
            
        Returns:
            List of matching events (newest first)
        """
        results = []
        
        for event in reversed(self._history):
            # Apply filters
            if event_types and event.event_type not in event_types:
                continue
            if source_id and event.source_id != source_id:
                continue
            if correlation_id and event.correlation_id != correlation_id:
                continue
            if since_timestamp and event.timestamp < since_timestamp:
                continue
            
            results.append(event)
            if len(results) >= limit:
                break
        
        return results
    
    async def _dispatch_loop(self) -> None:
        """Main dispatch loop."""
        batch: List[Event] = []
        
        while self._running:
            try:
                # Wait for event with timeout
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=self._batch_interval_ms / 1000,
                    )
                    batch.append(event)
                    
                    # Collect more events without waiting (batching)
                    while not self._queue.empty() and len(batch) < 100:
                        try:
                            event = self._queue.get_nowait()
                            batch.append(event)
                        except asyncio.QueueEmpty:
                            break
                            
                except asyncio.TimeoutError:
                    pass
                
                # Dispatch batch
                if batch:
                    await self._dispatch_batch(batch)
                    batch = []
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats["errors"] += 1
                # Log error but continue running
                batch = []
    
    async def _dispatch_batch(self, events: List[Event]) -> None:
        """Dispatch a batch of events to subscribers."""
        # Clean up dead weak references
        self._cleanup_dead_refs()
        
        for event in events:
            # Add to history
            self._history.append(event)
            
            # Dispatch to matching subscribers
            for sub_id, subscription in self._subscriptions:
                if subscription.matches(event):
                    try:
                        # Check weak ref is still alive
                        if subscription.weak_ref and sub_id in self._weak_refs:
                            ref = self._weak_refs[sub_id]
                            if ref() is None:
                                continue
                        
                        # Call callback
                        if asyncio.iscoroutinefunction(subscription.callback):
                            await subscription.callback(event)
                        else:
                            subscription.callback(event)
                        
                        self._stats["callbacks_invoked"] += 1
                        
                    except Exception as e:
                        self._stats["errors"] += 1
            
            self._stats["events_dispatched"] += 1
    
    def _cleanup_dead_refs(self) -> None:
        """Remove subscriptions with dead weak references."""
        dead_subs = []
        
        for sub_id, subscription in self._subscriptions:
            if subscription.weak_ref and sub_id in self._weak_refs:
                ref = self._weak_refs[sub_id]
                if ref() is None:
                    dead_subs.append(sub_id)
        
        for sub_id in dead_subs:
            self.unsubscribe(sub_id)


# Global event bus instance (singleton)
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance.
    
    Creates the bus on first call.
    
    Returns:
        The global EventBus instance
    """
    global _global_event_bus
    
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    
    return _global_event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _global_event_bus
    _global_event_bus = None


# Convenience functions for common operations
def emit_process_started(
    source_id: str,
    command: str,
    working_dir: str,
    correlation_id: Optional[str] = None,
) -> bool:
    """Emit a PROCESS_STARTED event."""
    return get_event_bus().emit_sync(
        EventType.PROCESS_STARTED,
        source_id,
        payload={"command": command, "working_dir": working_dir},
        correlation_id=correlation_id,
    )


def emit_output_line(
    source_id: str,
    line: str,
    is_stderr: bool = False,
    correlation_id: Optional[str] = None,
) -> bool:
    """Emit an OUTPUT_LINE or ERROR_LINE event."""
    event_type = EventType.ERROR_LINE if is_stderr else EventType.OUTPUT_LINE
    return get_event_bus().emit_sync(
        event_type,
        source_id,
        payload={"line": line, "is_stderr": is_stderr},
        correlation_id=correlation_id,
    )


def emit_process_completed(
    source_id: str,
    return_code: int,
    duration_ms: float,
    correlation_id: Optional[str] = None,
) -> bool:
    """Emit a PROCESS_COMPLETED event."""
    event_type = EventType.PROCESS_COMPLETED if return_code == 0 else EventType.PROCESS_FAILED
    return get_event_bus().emit_sync(
        event_type,
        source_id,
        payload={"return_code": return_code, "duration_ms": duration_ms},
        correlation_id=correlation_id,
    )


def emit_progress_update(
    source_id: str,
    progress: float,
    message: str = "",
    correlation_id: Optional[str] = None,
) -> bool:
    """Emit a PROGRESS_UPDATE event (debounced)."""
    event = Event(
        event_type=EventType.PROGRESS_UPDATE,
        source_id=source_id,
        payload={"progress": progress, "message": message},
        correlation_id=correlation_id,
    )
    return get_event_bus().emit_debounced(event, f"progress_{source_id}", debounce_ms=50)


def emit_result_available(
    source_id: str,
    result_data: Dict[str, Any],
    correlation_id: Optional[str] = None,
) -> bool:
    """Emit a RESULT_AVAILABLE event."""
    return get_event_bus().emit_sync(
        EventType.RESULT_AVAILABLE,
        source_id,
        payload=result_data,
        correlation_id=correlation_id,
    )
