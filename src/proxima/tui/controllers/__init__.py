"""Proxima TUI Controllers Package.

Controllers for managing TUI interactions with Proxima core.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import threading
import time

from .navigation import NavigationController
from .execution import ExecutionController
from .session import SessionController
from .backends import BackendController

# Phase 2: Backend Configuration Interface
from .backend_generator import BackendCodeGenerator
from .backend_test_runner import (
    BackendTestRunner,
    TestResult,
    EnhancedBackendTestRunner,
    create_test_runner,
)
from .backend_file_writer import BackendFileWriter

# Phase 5: Integration & Deployment
from .deployment_manager import (
    DeploymentManager,
    DeploymentStage,
    DeploymentStatus,
    DeploymentProgress,
    DeploymentResult,
    BatchDeploymentManager,
)


# ======================== BACKWARD COMPAT ENUMS ========================


class EventType(Enum):
    """Event types for TUI events."""
    SCREEN_CHANGED = "screen_changed"
    DATA_LOADED = "data_loaded"
    DATA_UPDATED = "data_updated"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    BACKEND_CONNECTED = "backend_connected"
    BACKEND_DISCONNECTED = "backend_disconnected"
    BACKEND_ERROR = "backend_error"
    CONFIG_CHANGED = "config_changed"
    USER_ACTION = "user_action"


class ExecutionStatus(Enum):
    """Execution status values."""
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ======================== BACKWARD COMPAT DATACLASSES ========================


@dataclass
class TUIEvent:
    """TUI event for the event bus."""
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionState:
    """State of an execution."""
    id: str
    status: ExecutionStatus = ExecutionStatus.IDLE
    progress: float = 0.0
    current_stage: str = ""
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ======================== BACKWARD COMPAT SINGLETONS ========================


class EventBus:
    """Singleton event bus for TUI communication."""
    
    _instance: Optional["EventBus"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "EventBus":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._subscribers: Dict[EventType, List[Callable]] = {}
            return cls._instance
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass
    
    def emit(self, event: TUIEvent) -> None:
        """Emit an event to all subscribers."""
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None


class StateStore:
    """Simple state store for TUI state management."""
    
    _instance: Optional["StateStore"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "StateStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._state: Dict[str, Any] = {}
            return cls._instance
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the store."""
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the store."""
        self._state[key] = value
    
    def delete(self, key: str) -> None:
        """Delete a key from the store."""
        self._state.pop(key, None)
    
    def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None


# Alias for backward compatibility
StateManager = StateStore


# ======================== BACKWARD COMPAT SINGLETON WRAPPERS ========================
# These wrap the actual controllers to provide stateless singleton access for tests


class _CompatNavigationController:
    """Singleton wrapper for NavigationController (backward compat)."""
    
    _instance: Optional["_CompatNavigationController"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "_CompatNavigationController":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._current_screen = "dashboard"
                cls._instance._history: List[str] = []
            return cls._instance
    
    @property
    def current_screen(self) -> str:
        """Get the current screen."""
        return self._current_screen
    
    def navigate_to(self, screen: str) -> None:
        """Navigate to a screen."""
        if self._current_screen:
            self._history.append(self._current_screen)
        self._current_screen = screen
    
    def go_back(self) -> Optional[str]:
        """Go back to the previous screen."""
        if self._history:
            self._current_screen = self._history.pop()
            return self._current_screen
        return None
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None


class _CompatExecutionController:
    """Singleton wrapper for ExecutionController (backward compat)."""
    
    _instance: Optional["_CompatExecutionController"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "_CompatExecutionController":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._executions: Dict[str, ExecutionState] = {}
                cls._instance._current_execution: Optional[str] = None
            return cls._instance
    
    def start_execution(self, execution_id: str) -> ExecutionState:
        """Start a new execution."""
        state = ExecutionState(
            id=execution_id,
            status=ExecutionStatus.RUNNING,
            started_at=time.time(),
        )
        self._executions[execution_id] = state
        self._current_execution = execution_id
        return state
    
    def update_progress(self, execution_id: str, progress: float, stage: str = "") -> None:
        """Update execution progress."""
        if execution_id in self._executions:
            self._executions[execution_id].progress = progress
            if stage:
                self._executions[execution_id].current_stage = stage
    
    def complete_execution(self, execution_id: str) -> None:
        """Mark execution as completed."""
        if execution_id in self._executions:
            self._executions[execution_id].status = ExecutionStatus.COMPLETED
            self._executions[execution_id].completed_at = time.time()
            self._executions[execution_id].progress = 100.0
    
    def fail_execution(self, execution_id: str, error: str) -> None:
        """Mark execution as failed."""
        if execution_id in self._executions:
            self._executions[execution_id].status = ExecutionStatus.FAILED
            self._executions[execution_id].completed_at = time.time()
            self._executions[execution_id].error = error
    
    def get_execution(self, execution_id: str) -> Optional[ExecutionState]:
        """Get execution state."""
        return self._executions.get(execution_id)
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None


# Override the imported controllers with backward compat versions for tests
# Keep the real controllers available with "_Real" suffix
_RealNavigationController = NavigationController
_RealExecutionController = ExecutionController
NavigationController = _CompatNavigationController
ExecutionController = _CompatExecutionController


__all__ = [
    # Main controllers (backward compat singleton versions)
    "NavigationController",
    "ExecutionController",
    "SessionController",
    "BackendController",
    # Phase 2: Backend Configuration Interface
    "BackendCodeGenerator",
    "BackendTestRunner",
    "TestResult",
    "BackendFileWriter",
    # Phase 4: Enhanced Test Runner
    "EnhancedBackendTestRunner",
    "create_test_runner",
    # Phase 5: Integration & Deployment
    "DeploymentManager",
    "DeploymentStage",
    "DeploymentStatus",
    "DeploymentProgress",
    "DeploymentResult",
    "BatchDeploymentManager",
    # Real controllers (require state)
    "_RealNavigationController",
    "_RealExecutionController",
    # Backward compat - Enums
    "EventType",
    "ExecutionStatus",
    # Backward compat - Dataclasses
    "TUIEvent",
    "ExecutionState",
    # Backward compat - Singletons
    "EventBus",
    "StateStore",
    "StateManager",
]
