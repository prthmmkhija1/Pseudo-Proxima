"""TUI State Controllers for Proxima.

Step 6.1: State management and event handling:
- Screen state management
- Event bus for cross-component communication
- Data binding helpers
- Navigation controller
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

from textual.message import Message

# ========== Event Types ==========


class EventType(Enum):
    """Types of TUI events."""

    # Navigation events
    SCREEN_CHANGED = "screen_changed"
    SCREEN_PUSHED = "screen_pushed"
    SCREEN_POPPED = "screen_popped"

    # Data events
    DATA_LOADED = "data_loaded"
    DATA_UPDATED = "data_updated"
    DATA_ERROR = "data_error"

    # Execution events
    EXECUTION_STARTED = "execution_started"
    EXECUTION_PROGRESS = "execution_progress"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_CANCELLED = "execution_cancelled"

    # Backend events
    BACKEND_CONNECTED = "backend_connected"
    BACKEND_DISCONNECTED = "backend_disconnected"
    BACKEND_ERROR = "backend_error"

    # Settings events
    SETTINGS_CHANGED = "settings_changed"
    SETTINGS_SAVED = "settings_saved"
    SETTINGS_LOADED = "settings_loaded"

    # User interaction events
    CONSENT_REQUESTED = "consent_requested"
    CONSENT_GRANTED = "consent_granted"
    CONSENT_DENIED = "consent_denied"

    # Log events
    LOG_ENTRY = "log_entry"
    LOG_CLEARED = "log_cleared"


@dataclass
class TUIEvent:
    """Event for TUI communication."""

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def __post_init__(self):
        if not self.source:
            self.source = "unknown"


# ========== Event Bus ==========


class EventBus:
    """Central event bus for TUI components."""

    _instance: EventBus | None = None
    _handlers: dict[EventType, list[Callable[[TUIEvent], None]]]
    _async_handlers: dict[EventType, list[Callable[[TUIEvent], Any]]]

    def __new__(cls) -> EventBus:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handlers = {}
            cls._instance._async_handlers = {}
        return cls._instance

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[TUIEvent], None],
    ) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function for the event
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_async(
        self,
        event_type: EventType,
        handler: Callable[[TUIEvent], Any],
    ) -> None:
        """Subscribe to an event type with async handler.

        Args:
            event_type: Type of event to subscribe to
            handler: Async callback function for the event
        """
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []
        self._async_handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[TUIEvent], None],
    ) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event
            handler: Handler to remove
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    def publish(self, event: TUIEvent) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception:
                    pass  # Don't let one handler break others

    async def publish_async(self, event: TUIEvent) -> None:
        """Publish an event to all async subscribers.

        Args:
            event: Event to publish
        """
        # Handle sync handlers first
        self.publish(event)

        # Handle async handlers
        if event.event_type in self._async_handlers:
            tasks = []
            for handler in self._async_handlers[event.event_type]:
                tasks.append(asyncio.create_task(handler(event)))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def clear(self) -> None:
        """Clear all handlers."""
        self._handlers.clear()
        self._async_handlers.clear()


# Global event bus instance
event_bus = EventBus()


# ========== State Management ==========


T = TypeVar("T")


class StateChange(Message, Generic[T]):
    """Message for state changes."""

    def __init__(self, key: str, old_value: T | None, new_value: T) -> None:
        super().__init__()
        self.key = key
        self.old_value = old_value
        self.new_value = new_value


@dataclass
class State(Generic[T]):
    """Observable state container."""

    _value: T
    _listeners: list[Callable[[T, T], None]] = field(default_factory=list)
    _name: str = "state"

    @property
    def value(self) -> T:
        """Get current value."""
        return self._value

    @value.setter
    def value(self, new_value: T) -> None:
        """Set value and notify listeners."""
        if new_value != self._value:
            old_value = self._value
            self._value = new_value
            self._notify(old_value, new_value)

    def subscribe(self, listener: Callable[[T, T], None]) -> None:
        """Subscribe to state changes.

        Args:
            listener: Callback (old_value, new_value) -> None
        """
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[T, T], None]) -> None:
        """Unsubscribe from state changes."""
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def _notify(self, old_value: T, new_value: T) -> None:
        """Notify all listeners of change."""
        for listener in self._listeners:
            try:
                listener(old_value, new_value)
            except Exception:
                pass


class StateStore:
    """Centralized state store for the TUI."""

    _instance: StateStore | None = None
    _states: dict[str, State]
    _history: list[tuple[str, Any, Any]]
    _max_history: int = 100

    def __new__(cls) -> StateStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._states = {}
            cls._instance._history = []
        return cls._instance

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value.

        Args:
            key: State key
            default: Default value if not found

        Returns:
            State value or default
        """
        if key in self._states:
            return self._states[key].value
        return default

    def set(self, key: str, value: Any) -> None:
        """Set state value.

        Args:
            key: State key
            value: New value
        """
        if key not in self._states:
            self._states[key] = State(_value=value, _name=key)
        else:
            old_value = self._states[key].value
            self._states[key].value = value
            self._record_history(key, old_value, value)

    def subscribe(self, key: str, listener: Callable[[Any, Any], None]) -> None:
        """Subscribe to state changes.

        Args:
            key: State key
            listener: Callback function
        """
        if key not in self._states:
            self._states[key] = State(_value=None, _name=key)
        self._states[key].subscribe(listener)

    def _record_history(self, key: str, old_value: Any, new_value: Any) -> None:
        """Record state change in history."""
        self._history.append((key, old_value, new_value))
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def get_history(self, key: str | None = None) -> list[tuple[str, Any, Any]]:
        """Get state change history.

        Args:
            key: Optional filter by key

        Returns:
            List of (key, old_value, new_value) tuples
        """
        if key:
            return [(k, o, n) for k, o, n in self._history if k == key]
        return self._history.copy()

    def clear(self) -> None:
        """Clear all state."""
        self._states.clear()
        self._history.clear()


# Global state store
state_store = StateStore()


# ========== Navigation Controller ==========


class NavigationController:
    """Controller for screen navigation."""

    _instance: NavigationController | None = None
    _app: Any = None  # Will be set to Textual App
    _history: list[str]
    _current_screen: str

    def __new__(cls) -> NavigationController:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._history = []
            cls._instance._current_screen = "dashboard"
        return cls._instance

    def bind_app(self, app: Any) -> None:
        """Bind to a Textual app instance.

        Args:
            app: The Textual App instance
        """
        self._app = app

    @property
    def current_screen(self) -> str:
        """Get current screen name."""
        return self._current_screen

    @property
    def history(self) -> list[str]:
        """Get navigation history."""
        return self._history.copy()

    def navigate_to(self, screen_name: str) -> None:
        """Navigate to a screen.

        Args:
            screen_name: Name of the screen to navigate to
        """
        if self._app is None:
            return

        if self._current_screen != screen_name:
            self._history.append(self._current_screen)
            self._current_screen = screen_name

            # Publish navigation event
            event_bus.publish(
                TUIEvent(
                    event_type=EventType.SCREEN_CHANGED,
                    data={
                        "from": self._history[-1] if self._history else None,
                        "to": screen_name,
                    },
                    source="navigation",
                )
            )

    def go_back(self) -> bool:
        """Go back to previous screen.

        Returns:
            True if navigation occurred, False if no history
        """
        if not self._history:
            return False

        previous = self._history.pop()
        self._current_screen = previous
        return True

    def clear_history(self) -> None:
        """Clear navigation history."""
        self._history.clear()


# Global navigation controller
nav_controller = NavigationController()


# ========== Data Controller ==========


@dataclass
class DataRequest:
    """Request for data loading."""

    source: str
    params: dict[str, Any] = field(default_factory=dict)
    cache: bool = True


@dataclass
class DataResponse:
    """Response from data loading."""

    success: bool
    data: Any = None
    error: str | None = None
    cached: bool = False


class DataController:
    """Controller for data loading and caching."""

    _instance: DataController | None = None
    _cache: dict[str, Any]
    _loaders: dict[str, Callable[[DataRequest], DataResponse]]

    def __new__(cls) -> DataController:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance._loaders = {}
        return cls._instance

    def register_loader(
        self,
        source: str,
        loader: Callable[[DataRequest], DataResponse],
    ) -> None:
        """Register a data loader.

        Args:
            source: Data source name
            loader: Loader function
        """
        self._loaders[source] = loader

    def load(self, request: DataRequest) -> DataResponse:
        """Load data from a source.

        Args:
            request: Data request

        Returns:
            DataResponse with loaded data
        """
        cache_key = f"{request.source}:{hash(frozenset(request.params.items()))}"

        # Check cache first
        if request.cache and cache_key in self._cache:
            event_bus.publish(
                TUIEvent(
                    event_type=EventType.DATA_LOADED,
                    data={"source": request.source, "cached": True},
                    source="data",
                )
            )
            return DataResponse(
                success=True,
                data=self._cache[cache_key],
                cached=True,
            )

        # Load from source
        if request.source not in self._loaders:
            return DataResponse(
                success=False,
                error=f"Unknown data source: {request.source}",
            )

        try:
            loader = self._loaders[request.source]
            response = loader(request)

            # Cache successful responses
            if response.success and request.cache:
                self._cache[cache_key] = response.data

            event_bus.publish(
                TUIEvent(
                    event_type=(
                        EventType.DATA_LOADED
                        if response.success
                        else EventType.DATA_ERROR
                    ),
                    data={"source": request.source, "error": response.error},
                    source="data",
                )
            )

            return response

        except Exception as e:
            event_bus.publish(
                TUIEvent(
                    event_type=EventType.DATA_ERROR,
                    data={"source": request.source, "error": str(e)},
                    source="data",
                )
            )
            return DataResponse(success=False, error=str(e))

    def invalidate(self, source: str | None = None) -> None:
        """Invalidate cached data.

        Args:
            source: Optional source to invalidate (None = all)
        """
        if source:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{source}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()

    def clear(self) -> None:
        """Clear all cache and loaders."""
        self._cache.clear()
        self._loaders.clear()


# Global data controller
data_controller = DataController()


# ========== Execution Controller ==========


class ExecutionStatus(Enum):
    """Status of an execution."""

    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionState:
    """State of an execution."""

    id: str
    status: ExecutionStatus = ExecutionStatus.IDLE
    progress: float = 0.0
    current_stage: str = ""
    message: str = ""
    start_time: float | None = None
    end_time: float | None = None
    result: Any = None
    error: str | None = None


class ExecutionController:
    """Controller for execution state and control."""

    _instance: ExecutionController | None = None
    _executions: dict[str, ExecutionState]
    _active_execution: str | None

    def __new__(cls) -> ExecutionController:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._executions = {}
            cls._instance._active_execution = None
        return cls._instance

    @property
    def active_execution(self) -> ExecutionState | None:
        """Get the active execution state."""
        if self._active_execution:
            return self._executions.get(self._active_execution)
        return None

    def start_execution(self, execution_id: str) -> ExecutionState:
        """Start a new execution.

        Args:
            execution_id: Unique execution ID

        Returns:
            ExecutionState for the new execution
        """
        import time

        state = ExecutionState(
            id=execution_id,
            status=ExecutionStatus.RUNNING,
            start_time=time.time(),
        )
        self._executions[execution_id] = state
        self._active_execution = execution_id

        event_bus.publish(
            TUIEvent(
                event_type=EventType.EXECUTION_STARTED,
                data={"id": execution_id},
                source="execution",
            )
        )

        return state

    def update_progress(
        self,
        execution_id: str,
        progress: float,
        stage: str = "",
        message: str = "",
    ) -> None:
        """Update execution progress.

        Args:
            execution_id: Execution ID
            progress: Progress percentage (0-100)
            stage: Current stage name
            message: Status message
        """
        if execution_id not in self._executions:
            return

        state = self._executions[execution_id]
        state.progress = progress
        state.current_stage = stage
        state.message = message

        event_bus.publish(
            TUIEvent(
                event_type=EventType.EXECUTION_PROGRESS,
                data={
                    "id": execution_id,
                    "progress": progress,
                    "stage": stage,
                    "message": message,
                },
                source="execution",
            )
        )

    def complete_execution(
        self,
        execution_id: str,
        result: Any = None,
    ) -> None:
        """Mark execution as completed.

        Args:
            execution_id: Execution ID
            result: Execution result
        """
        import time

        if execution_id not in self._executions:
            return

        state = self._executions[execution_id]
        state.status = ExecutionStatus.COMPLETED
        state.progress = 100.0
        state.end_time = time.time()
        state.result = result

        if self._active_execution == execution_id:
            self._active_execution = None

        event_bus.publish(
            TUIEvent(
                event_type=EventType.EXECUTION_COMPLETED,
                data={"id": execution_id, "result": result},
                source="execution",
            )
        )

    def fail_execution(
        self,
        execution_id: str,
        error: str,
    ) -> None:
        """Mark execution as failed.

        Args:
            execution_id: Execution ID
            error: Error message
        """
        import time

        if execution_id not in self._executions:
            return

        state = self._executions[execution_id]
        state.status = ExecutionStatus.FAILED
        state.end_time = time.time()
        state.error = error

        if self._active_execution == execution_id:
            self._active_execution = None

        event_bus.publish(
            TUIEvent(
                event_type=EventType.EXECUTION_FAILED,
                data={"id": execution_id, "error": error},
                source="execution",
            )
        )

    def cancel_execution(self, execution_id: str) -> None:
        """Cancel an execution.

        Args:
            execution_id: Execution ID
        """
        import time

        if execution_id not in self._executions:
            return

        state = self._executions[execution_id]
        state.status = ExecutionStatus.CANCELLED
        state.end_time = time.time()

        if self._active_execution == execution_id:
            self._active_execution = None

        event_bus.publish(
            TUIEvent(
                event_type=EventType.EXECUTION_CANCELLED,
                data={"id": execution_id},
                source="execution",
            )
        )

    def get_execution(self, execution_id: str) -> ExecutionState | None:
        """Get execution state by ID."""
        return self._executions.get(execution_id)

    def list_executions(self) -> list[ExecutionState]:
        """Get all execution states."""
        return list(self._executions.values())

    def clear_history(self) -> None:
        """Clear execution history (keeping active)."""
        if self._active_execution:
            active = self._executions[self._active_execution]
            self._executions = {self._active_execution: active}
        else:
            self._executions.clear()


# Global execution controller
exec_controller = ExecutionController()


# ========== Binding Helpers ==========


def bind_state_to_label(
    state_key: str,
    label,
    formatter: Callable[[Any], str] | None = None,
) -> None:
    """Bind a state value to a label's content.

    Args:
        state_key: State store key
        label: Textual Label widget
        formatter: Optional value formatter
    """

    def update_label(old: Any, new: Any) -> None:
        text = formatter(new) if formatter else str(new)
        label.update(text)

    state_store.subscribe(state_key, update_label)


def bind_event_to_handler(
    event_type: EventType,
    handler: Callable[[TUIEvent], None],
) -> Callable[[], None]:
    """Bind an event type to a handler.

    Args:
        event_type: Event type to subscribe to
        handler: Event handler

    Returns:
        Unsubscribe function
    """
    event_bus.subscribe(event_type, handler)
    return lambda: event_bus.unsubscribe(event_type, handler)
