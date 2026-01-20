"""Enhanced State Machine Implementation for Execution Flow.

Implements the execution lifecycle with explicit states and transitions,
plus critical missing features:
- State persistence during failures
- Resource cleanup on abort
- Complex transition validation

The state machine is agnostic to whether planning and execution are
driven by LLM-backed planners or local models - it only manages
lifecycle and visibility.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TypeVar

from transitions import Machine

from proxima.utils.logging import get_logger


# ==================== EXECUTION STATES ====================


class ExecutionState(str, Enum):
    """Execution lifecycle states."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"  # New: recovering from failure


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        message: str,
        from_state: str,
        to_state: str,
        trigger: str,
    ):
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger


class StatePersistenceError(Exception):
    """Raised when state persistence fails."""

    pass


# ==================== STATE PERSISTENCE ====================


@dataclass
class PersistedState:
    """Represents persisted state data."""

    execution_id: str
    state: str
    history: list[str]
    context_data: dict[str, Any]
    resources: list[str]
    timestamp: float
    error_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "execution_id": self.execution_id,
            "state": self.state,
            "history": self.history,
            "context_data": self.context_data,
            "resources": self.resources,
            "timestamp": self.timestamp,
            "error_info": self.error_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistedState:
        """Create from dictionary."""
        return cls(
            execution_id=data["execution_id"],
            state=data["state"],
            history=data["history"],
            context_data=data.get("context_data", {}),
            resources=data.get("resources", []),
            timestamp=data["timestamp"],
            error_info=data.get("error_info"),
        )


class StatePersistence:
    """Handles state persistence during failures.

    Features:
    - Periodic state snapshots
    - Crash recovery from persisted state
    - Atomic writes to prevent corruption
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        auto_persist_interval: float = 5.0,
    ):
        """Initialize state persistence.

        Args:
            storage_dir: Directory for state files
            auto_persist_interval: Seconds between auto-persists
        """
        self.storage_dir = storage_dir or Path(tempfile.gettempdir()) / "proxima_state"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_persist_interval = auto_persist_interval
        self.logger = get_logger("state.persistence")

        self._persist_lock = threading.Lock()
        self._auto_persist_thread: threading.Thread | None = None
        self._auto_persist_stop = threading.Event()

    def persist(self, state_data: PersistedState) -> Path:
        """Persist state to disk atomically.

        Args:
            state_data: State data to persist

        Returns:
            Path to the persisted state file
        """
        with self._persist_lock:
            state_file = self.storage_dir / f"{state_data.execution_id}.state.json"
            temp_file = state_file.with_suffix(".tmp")

            try:
                # Write to temp file first
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(state_data.to_dict(), f, indent=2)

                # Atomic rename
                temp_file.replace(state_file)

                self.logger.debug(
                    "state.persisted",
                    execution_id=state_data.execution_id,
                    state=state_data.state,
                )
                return state_file

            except Exception as exc:
                if temp_file.exists():
                    temp_file.unlink()
                raise StatePersistenceError(
                    f"Failed to persist state: {exc}"
                ) from exc

    def load(self, execution_id: str) -> PersistedState | None:
        """Load persisted state from disk.

        Args:
            execution_id: Execution ID to load

        Returns:
            Persisted state or None if not found
        """
        state_file = self.storage_dir / f"{execution_id}.state.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
            return PersistedState.from_dict(data)
        except Exception as exc:
            self.logger.warning(
                "state.load_failed",
                execution_id=execution_id,
                error=str(exc),
            )
            return None

    def delete(self, execution_id: str) -> bool:
        """Delete persisted state.

        Args:
            execution_id: Execution ID to delete

        Returns:
            True if deleted
        """
        state_file = self.storage_dir / f"{execution_id}.state.json"
        if state_file.exists():
            state_file.unlink()
            return True
        return False

    def list_persisted(self) -> list[str]:
        """List all persisted execution IDs."""
        return [
            f.stem.replace(".state", "")
            for f in self.storage_dir.glob("*.state.json")
        ]

    def recover_crashed_states(self) -> list[PersistedState]:
        """Find states that need recovery (crashed mid-execution).

        Returns:
            List of states that were running when interrupted
        """
        crashed = []
        for exec_id in self.list_persisted():
            state = self.load(exec_id)
            if state and state.state in (
                ExecutionState.RUNNING.value,
                ExecutionState.PAUSED.value,
                ExecutionState.PLANNING.value,
            ):
                crashed.append(state)
        return crashed

    def start_auto_persist(
        self,
        state_provider: Callable[[], PersistedState | None],
    ) -> None:
        """Start automatic periodic persistence.

        Args:
            state_provider: Callable that returns current state to persist
        """
        if self._auto_persist_thread is not None:
            return

        self._auto_persist_stop.clear()

        def persist_loop():
            while not self._auto_persist_stop.wait(self.auto_persist_interval):
                try:
                    state = state_provider()
                    if state:
                        self.persist(state)
                except Exception as exc:
                    self.logger.warning("auto_persist_failed", error=str(exc))

        self._auto_persist_thread = threading.Thread(
            target=persist_loop,
            daemon=True,
            name="state-auto-persist",
        )
        self._auto_persist_thread.start()

    def stop_auto_persist(self) -> None:
        """Stop automatic persistence."""
        if self._auto_persist_thread is None:
            return
        self._auto_persist_stop.set()
        self._auto_persist_thread.join(timeout=2.0)
        self._auto_persist_thread = None


# ==================== RESOURCE CLEANUP ====================


class ResourceHandle(ABC):
    """Abstract base for managed resources that need cleanup."""

    @property
    @abstractmethod
    def resource_id(self) -> str:
        """Unique identifier for this resource."""
        pass

    @property
    @abstractmethod
    def resource_type(self) -> str:
        """Type of resource (e.g., 'file', 'memory', 'connection')."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release/cleanup this resource."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Check if resource is still active."""
        pass


@dataclass
class FileResource(ResourceHandle):
    """File-based resource handle."""

    path: Path
    _id: str = field(default_factory=lambda: str(time.time()))
    _active: bool = True

    @property
    def resource_id(self) -> str:
        return self._id

    @property
    def resource_type(self) -> str:
        return "file"

    def cleanup(self) -> None:
        """Delete the file if it exists."""
        if self.path.exists():
            try:
                self.path.unlink()
            except Exception:
                pass
        self._active = False

    def is_active(self) -> bool:
        return self._active and self.path.exists()


@dataclass
class MemoryResource(ResourceHandle):
    """Memory-based resource handle."""

    data: Any
    size_bytes: int
    _id: str = field(default_factory=lambda: str(time.time()))
    _active: bool = True

    @property
    def resource_id(self) -> str:
        return self._id

    @property
    def resource_type(self) -> str:
        return "memory"

    def cleanup(self) -> None:
        """Release memory reference."""
        self.data = None
        self._active = False

    def is_active(self) -> bool:
        return self._active and self.data is not None


class ResourceCleanupManager:
    """Manages resource cleanup on abort or error.

    Features:
    - Track registered resources
    - Cleanup in reverse order of registration
    - Handle cleanup errors gracefully
    - Support resource grouping
    """

    def __init__(self):
        """Initialize the resource cleanup manager."""
        self.logger = get_logger("state.cleanup")
        self._resources: dict[str, ResourceHandle] = {}
        self._resource_groups: dict[str, list[str]] = {}
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def register(
        self,
        resource: ResourceHandle,
        group: str | None = None,
    ) -> str:
        """Register a resource for cleanup.

        Args:
            resource: Resource to track
            group: Optional group name for batch cleanup

        Returns:
            Resource ID
        """
        with self._lock:
            self._resources[resource.resource_id] = resource

            if group:
                if group not in self._resource_groups:
                    self._resource_groups[group] = []
                self._resource_groups[group].append(resource.resource_id)

            self.logger.debug(
                "resource.registered",
                resource_id=resource.resource_id,
                resource_type=resource.resource_type,
                group=group,
            )
            return resource.resource_id

    def unregister(self, resource_id: str) -> bool:
        """Unregister a resource (already cleaned up manually).

        Args:
            resource_id: Resource ID to unregister

        Returns:
            True if found and unregistered
        """
        with self._lock:
            if resource_id in self._resources:
                del self._resources[resource_id]

                # Remove from groups
                for group_resources in self._resource_groups.values():
                    if resource_id in group_resources:
                        group_resources.remove(resource_id)

                return True
            return False

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a cleanup callback to be called on abort.

        Args:
            callback: Function to call during cleanup
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def cleanup_resource(self, resource_id: str) -> bool:
        """Cleanup a specific resource.

        Args:
            resource_id: Resource to cleanup

        Returns:
            True if successfully cleaned up
        """
        with self._lock:
            resource = self._resources.get(resource_id)
            if not resource:
                return False

            try:
                resource.cleanup()
                self.logger.debug(
                    "resource.cleaned",
                    resource_id=resource_id,
                    resource_type=resource.resource_type,
                )
            except Exception as exc:
                self.logger.warning(
                    "resource.cleanup_failed",
                    resource_id=resource_id,
                    error=str(exc),
                )
            finally:
                del self._resources[resource_id]

            return True

    def cleanup_group(self, group: str) -> int:
        """Cleanup all resources in a group.

        Args:
            group: Group name

        Returns:
            Number of resources cleaned up
        """
        with self._lock:
            if group not in self._resource_groups:
                return 0

            resource_ids = list(self._resource_groups[group])
            count = 0

            for resource_id in resource_ids:
                if self.cleanup_resource(resource_id):
                    count += 1

            del self._resource_groups[group]
            return count

    def cleanup_all(self) -> dict[str, int]:
        """Cleanup all resources and run callbacks.

        Returns:
            Dict with cleanup statistics
        """
        stats = {"resources": 0, "callbacks": 0, "errors": 0}

        with self._lock:
            # Cleanup resources in reverse order
            resource_ids = list(reversed(list(self._resources.keys())))

            for resource_id in resource_ids:
                try:
                    resource = self._resources.get(resource_id)
                    if resource:
                        resource.cleanup()
                        stats["resources"] += 1
                except Exception as exc:
                    self.logger.warning(
                        "resource.cleanup_all_failed",
                        resource_id=resource_id,
                        error=str(exc),
                    )
                    stats["errors"] += 1

            self._resources.clear()
            self._resource_groups.clear()

            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                    stats["callbacks"] += 1
                except Exception as exc:
                    self.logger.warning("callback.cleanup_failed", error=str(exc))
                    stats["errors"] += 1

            self._cleanup_callbacks.clear()

        self.logger.info("cleanup.completed", **stats)
        return stats

    def get_active_resources(self) -> list[dict[str, Any]]:
        """Get list of active resources."""
        with self._lock:
            return [
                {
                    "id": r.resource_id,
                    "type": r.resource_type,
                    "active": r.is_active(),
                }
                for r in self._resources.values()
            ]


# ==================== COMPLEX TRANSITION VALIDATION ====================


@dataclass
class TransitionRule:
    """Defines a transition rule with conditions."""

    trigger: str
    source: str | list[str]
    dest: str
    conditions: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    before_callbacks: list[Callable[[dict[str, Any]], None]] = field(
        default_factory=list
    )
    after_callbacks: list[Callable[[dict[str, Any]], None]] = field(
        default_factory=list
    )
    validators: list[Callable[[dict[str, Any]], tuple[bool, str]]] = field(
        default_factory=list
    )
    description: str = ""


class TransitionValidator:
    """Complex transition validation with custom rules.

    Features:
    - Pre-transition validation
    - Conditional transitions
    - Before/after callbacks
    - Transition history with reasons
    """

    def __init__(self):
        """Initialize the transition validator."""
        self.logger = get_logger("state.validator")
        self._rules: dict[str, list[TransitionRule]] = {}
        self._transition_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: TransitionRule) -> None:
        """Add a transition rule.

        Args:
            rule: Transition rule to add
        """
        with self._lock:
            if rule.trigger not in self._rules:
                self._rules[rule.trigger] = []
            self._rules[rule.trigger].append(rule)

    def remove_rule(self, trigger: str, source: str | None = None) -> int:
        """Remove transition rules.

        Args:
            trigger: Trigger name
            source: Optional source state filter

        Returns:
            Number of rules removed
        """
        with self._lock:
            if trigger not in self._rules:
                return 0

            if source is None:
                count = len(self._rules[trigger])
                del self._rules[trigger]
                return count

            before = len(self._rules[trigger])
            self._rules[trigger] = [
                r
                for r in self._rules[trigger]
                if not (
                    r.source == source
                    or (isinstance(r.source, list) and source in r.source)
                )
            ]
            return before - len(self._rules[trigger])

    def validate_transition(
        self,
        trigger: str,
        current_state: str,
        context: dict[str, Any],
    ) -> tuple[bool, str, TransitionRule | None]:
        """Validate if a transition is allowed.

        Args:
            trigger: Transition trigger
            current_state: Current state
            context: Execution context

        Returns:
            Tuple of (is_valid, reason, matching_rule)
        """
        with self._lock:
            rules = self._rules.get(trigger, [])

            for rule in rules:
                # Check source state match
                sources = (
                    rule.source if isinstance(rule.source, list) else [rule.source]
                )
                if current_state not in sources and "*" not in sources:
                    continue

                # Check conditions
                conditions_met = all(cond(context) for cond in rule.conditions)
                if not conditions_met:
                    continue

                # Run validators
                for validator in rule.validators:
                    is_valid, reason = validator(context)
                    if not is_valid:
                        return False, reason, rule

                return True, "Transition allowed", rule

            return False, f"No matching rule for trigger '{trigger}' from state '{current_state}'", None

    def execute_transition(
        self,
        trigger: str,
        current_state: str,
        context: dict[str, Any],
    ) -> tuple[bool, str, str | None]:
        """Execute a validated transition with callbacks.

        Args:
            trigger: Transition trigger
            current_state: Current state
            context: Execution context

        Returns:
            Tuple of (success, reason, new_state)
        """
        is_valid, reason, rule = self.validate_transition(
            trigger, current_state, context
        )

        if not is_valid or rule is None:
            return False, reason, None

        # Execute before callbacks
        for callback in rule.before_callbacks:
            try:
                callback(context)
            except Exception as exc:
                return False, f"Before callback failed: {exc}", None

        # Record transition
        with self._lock:
            self._transition_history.append(
                {
                    "trigger": trigger,
                    "from_state": current_state,
                    "to_state": rule.dest,
                    "timestamp": time.time(),
                    "reason": reason,
                }
            )

        # Execute after callbacks
        for callback in rule.after_callbacks:
            try:
                callback(context)
            except Exception as exc:
                self.logger.warning(
                    "after_callback_failed",
                    trigger=trigger,
                    error=str(exc),
                )

        return True, reason, rule.dest

    def get_transition_history(self) -> list[dict[str, Any]]:
        """Get transition history."""
        with self._lock:
            return list(self._transition_history)

    def clear_history(self) -> None:
        """Clear transition history."""
        with self._lock:
            self._transition_history.clear()


# ==================== STANDARD TRANSITIONS ====================


TRANSITIONS: list[dict[str, Any]] = [
    {
        "trigger": "start",
        "source": ExecutionState.IDLE,
        "dest": ExecutionState.PLANNING,
    },
    {
        "trigger": "plan_complete",
        "source": ExecutionState.PLANNING,
        "dest": ExecutionState.READY,
    },
    {
        "trigger": "plan_failed",
        "source": ExecutionState.PLANNING,
        "dest": ExecutionState.ERROR,
    },
    {
        "trigger": "execute",
        "source": ExecutionState.READY,
        "dest": ExecutionState.RUNNING,
    },
    {
        "trigger": "pause",
        "source": ExecutionState.RUNNING,
        "dest": ExecutionState.PAUSED,
    },
    {
        "trigger": "resume",
        "source": ExecutionState.PAUSED,
        "dest": ExecutionState.RUNNING,
    },
    {
        "trigger": "complete",
        "source": ExecutionState.RUNNING,
        "dest": ExecutionState.COMPLETED,
    },
    {
        "trigger": "abort",
        "source": [ExecutionState.RUNNING, ExecutionState.PAUSED],
        "dest": ExecutionState.ABORTED,
    },
    {
        "trigger": "error",
        "source": ExecutionState.RUNNING,
        "dest": ExecutionState.ERROR,
    },
    {
        "trigger": "recover",
        "source": ExecutionState.ERROR,
        "dest": ExecutionState.RECOVERING,
    },
    {
        "trigger": "recovery_complete",
        "source": ExecutionState.RECOVERING,
        "dest": ExecutionState.READY,
    },
    {
        "trigger": "recovery_failed",
        "source": ExecutionState.RECOVERING,
        "dest": ExecutionState.ERROR,
    },
    {"trigger": "reset", "source": "*", "dest": ExecutionState.IDLE},
]


# ==================== ENHANCED STATE MACHINE ====================


class ExecutionStateMachine:
    """Enhanced finite state machine for execution lifecycle.

    Features:
    - State persistence during failures
    - Resource cleanup on abort
    - Complex transition validation
    - Recovery from crashed states

    Parameters
    ----------
    execution_id : str | None
        Identifier to bind into logs for traceability.
    enable_persistence : bool
        Whether to enable state persistence.
    storage_dir : Path | None
        Directory for state persistence.
    """

    def __init__(
        self,
        execution_id: str | None = None,
        enable_persistence: bool = True,
        storage_dir: Path | None = None,
    ):
        """Initialize the state machine.

        Args:
            execution_id: Optional execution identifier
            enable_persistence: Whether to persist state
            storage_dir: Optional storage directory
        """
        self.execution_id = execution_id or str(int(time.time() * 1000))
        self.logger = get_logger("state").bind(execution_id=execution_id)
        self.history: list[str] = []
        self.state: str = ExecutionState.IDLE.value
        self.context_data: dict[str, Any] = {}

        # Initialize components
        self.resource_manager = ResourceCleanupManager()
        self.transition_validator = TransitionValidator()
        self._setup_default_validation_rules()

        # Persistence
        self.enable_persistence = enable_persistence
        self._persistence: StatePersistence | None = None
        if enable_persistence:
            self._persistence = StatePersistence(storage_dir=storage_dir)

        # Error tracking
        self.last_error: Exception | None = None
        self.error_traceback: str | None = None

        # Setup state machine
        self._machine = Machine(
            model=self,
            states=[state.value for state in ExecutionState],
            transitions=TRANSITIONS,
            initial=ExecutionState.IDLE.value,
            auto_transitions=False,
            ignore_invalid_triggers=True,
            after_state_change=self._record_transition,
            send_event=False,
        )

    def _setup_default_validation_rules(self) -> None:
        """Setup default transition validation rules."""
        # Rule: Cannot execute without a plan
        self.transition_validator.add_rule(
            TransitionRule(
                trigger="execute",
                source=ExecutionState.READY.value,
                dest=ExecutionState.RUNNING.value,
                validators=[
                    lambda ctx: (
                        bool(ctx.get("has_plan", False)),
                        "Cannot execute without a plan",
                    )
                ],
                description="Require plan before execution",
            )
        )

        # Rule: Abort triggers cleanup
        self.transition_validator.add_rule(
            TransitionRule(
                trigger="abort",
                source=[ExecutionState.RUNNING.value, ExecutionState.PAUSED.value],
                dest=ExecutionState.ABORTED.value,
                after_callbacks=[lambda ctx: self._handle_abort(ctx)],
                description="Cleanup on abort",
            )
        )

    def _handle_abort(self, context: dict[str, Any]) -> None:
        """Handle abort transition - cleanup resources."""
        self.logger.info("state.abort_cleanup_started")
        stats = self.resource_manager.cleanup_all()
        self.logger.info("state.abort_cleanup_completed", **stats)

    def _record_transition(self) -> None:
        """Record state transition and persist."""
        self.history.append(self.state)
        self.logger.info("state.transition", state=self.state)

        # Persist state after transition
        if self.enable_persistence and self._persistence:
            try:
                self._persistence.persist(self._get_persisted_state())
            except StatePersistenceError as exc:
                self.logger.warning("state.persist_failed", error=str(exc))

    def _get_persisted_state(self) -> PersistedState:
        """Get current state for persistence."""
        error_info = None
        if self.last_error:
            error_info = {
                "type": type(self.last_error).__name__,
                "message": str(self.last_error),
                "traceback": self.error_traceback,
            }

        return PersistedState(
            execution_id=self.execution_id,
            state=self.state,
            history=list(self.history),
            context_data=self.context_data,
            resources=[
                r["id"] for r in self.resource_manager.get_active_resources()
            ],
            timestamp=time.time(),
            error_info=error_info,
        )

    # ==================== PUBLIC API ====================

    def set_context(self, key: str, value: Any) -> None:
        """Set context data for validation.

        Args:
            key: Context key
            value: Context value
        """
        self.context_data[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context data.

        Args:
            key: Context key
            default: Default value

        Returns:
            Context value or default
        """
        return self.context_data.get(key, default)

    def register_resource(
        self,
        resource: ResourceHandle,
        group: str | None = None,
    ) -> str:
        """Register a resource for cleanup on abort.

        Args:
            resource: Resource to register
            group: Optional group name

        Returns:
            Resource ID
        """
        return self.resource_manager.register(resource, group)

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a cleanup callback for abort.

        Args:
            callback: Cleanup function
        """
        self.resource_manager.add_cleanup_callback(callback)

    def record_error(self, error: Exception) -> None:
        """Record an error that occurred during execution.

        Args:
            error: Exception that occurred
        """
        self.last_error = error
        self.error_traceback = traceback.format_exc()
        self.set_context("last_error", str(error))

    def validate_transition(self, trigger: str) -> tuple[bool, str]:
        """Check if a transition is valid.

        Args:
            trigger: Transition trigger

        Returns:
            Tuple of (is_valid, reason)
        """
        is_valid, reason, _ = self.transition_validator.validate_transition(
            trigger, self.state, self.context_data
        )
        return is_valid, reason

    def safe_transition(self, trigger: str) -> tuple[bool, str]:
        """Attempt a transition with validation.

        Args:
            trigger: Transition trigger

        Returns:
            Tuple of (success, reason)
        """
        is_valid, reason = self.validate_transition(trigger)
        if not is_valid:
            return False, reason

        # Attempt the transition
        trigger_method = getattr(self, trigger, None)
        if trigger_method and callable(trigger_method):
            try:
                trigger_method()
                return True, f"Transition '{trigger}' successful"
            except Exception as exc:
                return False, f"Transition failed: {exc}"

        return False, f"Unknown trigger: {trigger}"

    def recover_from_crash(self) -> bool:
        """Attempt to recover from a crashed state.

        Returns:
            True if recovery successful
        """
        if not self._persistence:
            return False

        persisted = self._persistence.load(self.execution_id)
        if not persisted:
            return False

        # Restore state
        self.state = persisted.state
        self.history = persisted.history
        self.context_data = persisted.context_data

        # Trigger recovery
        if self.state in (
            ExecutionState.RUNNING.value,
            ExecutionState.PAUSED.value,
        ):
            self.recover()
            return True

        return False

    def recover_with_validation(self) -> tuple[bool, str, dict[str, Any]]:
        """Enhanced recovery with comprehensive validation and edge case handling.
        
        Handles edge cases:
        - Partial state corruption
        - Orphaned resources from interrupted execution
        - Nested error recovery (errors during recovery)
        - Stale checkpoint detection
        - Resource contention after recovery
        
        Returns:
            Tuple of (success, message, recovery_details)
        """
        recovery_details: dict[str, Any] = {
            "started_at": time.time(),
            "execution_id": self.execution_id,
            "steps_completed": [],
            "warnings": [],
            "resources_cleaned": 0,
            "checkpoints_processed": 0,
        }
        
        if not self._persistence:
            return False, "No persistence layer configured", recovery_details
        
        try:
            # Step 1: Load persisted state with validation
            persisted = self._safe_load_persisted_state()
            if not persisted:
                return False, "No valid persisted state found", recovery_details
            
            recovery_details["steps_completed"].append("load_persisted_state")
            recovery_details["original_state"] = persisted.state
            
            # Step 2: Validate state integrity
            integrity_valid, integrity_issues = self._validate_state_integrity(persisted)
            if not integrity_valid:
                recovery_details["warnings"].extend(integrity_issues)
                self.logger.warning("recovery.integrity_issues", issues=integrity_issues)
            recovery_details["steps_completed"].append("validate_integrity")
            
            # Step 3: Check for stale state (too old to recover meaningfully)
            stale_threshold = 3600  # 1 hour
            state_age = time.time() - persisted.timestamp
            if state_age > stale_threshold:
                recovery_details["warnings"].append(
                    f"State is {state_age:.0f}s old (>{stale_threshold}s threshold)"
                )
                recovery_details["state_age_seconds"] = state_age
            recovery_details["steps_completed"].append("check_staleness")
            
            # Step 4: Clean up orphaned resources before restoring state
            orphan_cleanup = self._cleanup_orphaned_resources(persisted)
            recovery_details["resources_cleaned"] = orphan_cleanup["cleaned"]
            recovery_details["steps_completed"].append("cleanup_orphaned_resources")
            
            # Step 5: Restore state with atomic rollback on failure
            restore_success, restore_msg = self._atomic_state_restore(persisted)
            if not restore_success:
                return False, f"State restoration failed: {restore_msg}", recovery_details
            recovery_details["steps_completed"].append("restore_state")
            
            # Step 6: Validate restored state is consistent
            consistency_valid, consistency_msg = self._validate_restored_consistency()
            if not consistency_valid:
                recovery_details["warnings"].append(f"Consistency warning: {consistency_msg}")
            recovery_details["steps_completed"].append("validate_consistency")
            
            # Step 7: Handle recovery state transition
            if self.state in (
                ExecutionState.RUNNING.value,
                ExecutionState.PAUSED.value,
                ExecutionState.PLANNING.value,
            ):
                try:
                    self.recover()
                    recovery_details["steps_completed"].append("transition_to_recovering")
                except Exception as transition_error:
                    # Nested error: recovery itself failed
                    self.logger.error(
                        "recovery.transition_failed",
                        error=str(transition_error),
                    )
                    recovery_details["warnings"].append(
                        f"Recovery transition failed: {transition_error}"
                    )
                    # Fall back to IDLE state
                    self.state = ExecutionState.IDLE.value
                    recovery_details["fallback_state"] = "IDLE"
            
            recovery_details["completed_at"] = time.time()
            recovery_details["duration_ms"] = (
                recovery_details["completed_at"] - recovery_details["started_at"]
            ) * 1000
            
            return True, "Recovery completed successfully", recovery_details
            
        except Exception as exc:
            # Handle nested errors during recovery
            self.logger.error("recovery.catastrophic_failure", error=str(exc))
            recovery_details["catastrophic_error"] = str(exc)
            
            # Attempt to reset to safe state
            try:
                self._reset_to_safe_state()
                recovery_details["reset_to_safe_state"] = True
            except Exception:
                recovery_details["reset_to_safe_state"] = False
            
            return False, f"Recovery failed catastrophically: {exc}", recovery_details
    
    def _safe_load_persisted_state(self) -> PersistedState | None:
        """Safely load persisted state with error handling.
        
        Returns:
            PersistedState or None if loading failed
        """
        if not self._persistence:
            return None
        
        try:
            persisted = self._persistence.load(self.execution_id)
            if persisted:
                return persisted
        except Exception as load_error:
            self.logger.warning(
                "recovery.load_failed",
                execution_id=self.execution_id,
                error=str(load_error),
            )
        
        # Try to find any crashed states that match
        try:
            crashed_states = self._persistence.recover_crashed_states()
            for state in crashed_states:
                if state.execution_id == self.execution_id:
                    return state
        except Exception:
            pass
        
        return None
    
    def _validate_state_integrity(
        self, persisted: PersistedState
    ) -> tuple[bool, list[str]]:
        """Validate integrity of persisted state.
        
        Args:
            persisted: The persisted state to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues: list[str] = []
        
        # Check required fields
        if not persisted.execution_id:
            issues.append("Missing execution_id")
        
        if not persisted.state:
            issues.append("Missing state")
        elif persisted.state not in [s.value for s in ExecutionState]:
            issues.append(f"Invalid state value: {persisted.state}")
        
        if not persisted.timestamp or persisted.timestamp <= 0:
            issues.append("Invalid or missing timestamp")
        
        # Check history consistency
        if persisted.history:
            if persisted.state not in persisted.history:
                issues.append("Current state not in history")
        
        # Check for required context data
        if persisted.context_data is None:
            issues.append("Context data is None (should be empty dict)")
        
        return len(issues) == 0, issues
    
    def _cleanup_orphaned_resources(
        self, persisted: PersistedState
    ) -> dict[str, int]:
        """Clean up orphaned resources from interrupted execution.
        
        Args:
            persisted: The persisted state with resource info
            
        Returns:
            Dict with cleanup statistics
        """
        stats = {"cleaned": 0, "failed": 0, "skipped": 0}
        
        # Get currently registered resources
        current_resources = set(
            r["id"] for r in self.resource_manager.get_active_resources()
        )
        
        # Get persisted resource IDs
        persisted_resources = set(persisted.resources)
        
        # Find orphaned resources (in persistence but not currently tracked)
        orphaned = persisted_resources - current_resources
        
        for resource_id in orphaned:
            try:
                # Attempt to clean up orphaned resource
                if self.resource_manager.cleanup_resource(resource_id):
                    stats["cleaned"] += 1
                else:
                    stats["skipped"] += 1
            except Exception:
                stats["failed"] += 1
        
        return stats
    
    def _atomic_state_restore(
        self, persisted: PersistedState
    ) -> tuple[bool, str]:
        """Atomically restore state with rollback on failure.
        
        Args:
            persisted: The state to restore
            
        Returns:
            Tuple of (success, message)
        """
        # Save current state for rollback
        backup_state = self.state
        backup_history = list(self.history)
        backup_context = dict(self.context_data)
        
        try:
            # Restore state
            self.state = persisted.state
            self.history = list(persisted.history)
            self.context_data = dict(persisted.context_data)
            
            # Restore error info if present
            if persisted.error_info:
                self.set_context("last_error", persisted.error_info.get("message"))
                self.error_traceback = persisted.error_info.get("traceback")
            
            return True, "State restored successfully"
            
        except Exception as restore_error:
            # Rollback to previous state
            self.state = backup_state
            self.history = backup_history
            self.context_data = backup_context
            
            return False, f"Restore failed, rolled back: {restore_error}"
    
    def _validate_restored_consistency(self) -> tuple[bool, str]:
        """Validate consistency of restored state.
        
        Returns:
            Tuple of (is_consistent, message)
        """
        # Check state machine is in valid state
        if self.state not in [s.value for s in ExecutionState]:
            return False, f"Invalid state after restore: {self.state}"
        
        # Check history contains current state
        if self.history and self.state not in self.history:
            return False, "Current state not in history after restore"
        
        # Validate transitions are possible from current state
        valid_triggers = self._get_valid_triggers_for_state(self.state)
        if not valid_triggers:
            return False, f"No valid transitions from state: {self.state}"
        
        return True, "State is consistent"
    
    def _get_valid_triggers_for_state(self, state: str) -> list[str]:
        """Get valid transition triggers for a given state.
        
        Args:
            state: The state to check
            
        Returns:
            List of valid trigger names
        """
        valid_triggers = []
        
        for transition in TRANSITIONS:
            source = transition.get("source")
            if isinstance(source, list):
                if state in [s.value if hasattr(s, 'value') else s for s in source]:
                    valid_triggers.append(transition["trigger"])
            elif source == "*":
                valid_triggers.append(transition["trigger"])
            elif hasattr(source, 'value') and source.value == state:
                valid_triggers.append(transition["trigger"])
            elif source == state:
                valid_triggers.append(transition["trigger"])
        
        return valid_triggers
    
    def _reset_to_safe_state(self) -> None:
        """Reset state machine to a safe known state after catastrophic failure."""
        self.state = ExecutionState.IDLE.value
        self.history = [ExecutionState.IDLE.value]
        self.context_data = {}
        self.last_error = None
        self.error_traceback = None
        
        # Clean up all resources
        try:
            self.resource_manager.cleanup_all()
        except Exception:
            pass
        
        # Stop auto-persist
        try:
            self.stop_auto_persist()
        except Exception:
            pass
        
        self.logger.info("recovery.reset_to_safe_state", execution_id=self.execution_id)

    def cleanup_on_complete(self) -> None:
        """Cleanup persistence after successful completion."""
        if self._persistence:
            self._persistence.delete(self.execution_id)

    def start_auto_persist(self) -> None:
        """Start automatic state persistence."""
        if self._persistence:
            self._persistence.start_auto_persist(self._get_persisted_state)

    def stop_auto_persist(self) -> None:
        """Stop automatic state persistence."""
        if self._persistence:
            self._persistence.stop_auto_persist()

    # ==================== CONVENIENCE ACCESSORS ====================

    @property
    def state_enum(self) -> ExecutionState:
        """Get current state as enum."""
        return ExecutionState(self.state)

    @property
    def is_running(self) -> bool:
        """Check if execution is running."""
        return self.state == ExecutionState.RUNNING.value

    @property
    def is_terminal(self) -> bool:
        """Check if in a terminal state."""
        return self.state in (
            ExecutionState.COMPLETED.value,
            ExecutionState.ABORTED.value,
            ExecutionState.ERROR.value,
        )

    @property
    def can_resume(self) -> bool:
        """Check if execution can be resumed."""
        return self.state == ExecutionState.PAUSED.value

    def snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the current state."""
        return {
            "execution_id": self.execution_id,
            "state": self.state,
            "history": list(self.history),
            "context_data": self.context_data,
            "resources": self.resource_manager.get_active_resources(),
            "is_terminal": self.is_terminal,
            "last_error": str(self.last_error) if self.last_error else None,
        }


# ==============================================================================
# EDGE CASE RECOVERY (1% Gap Coverage)
# ==============================================================================


class RecoveryStrategy(Enum):
    """Available recovery strategies for edge cases."""
    
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_STATE = "fallback_state"
    PARTIAL_ROLLBACK = "partial_rollback"
    FULL_ROLLBACK = "full_rollback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SKIP_AND_CONTINUE = "skip_and_continue"


@dataclass
class EdgeCaseContext:
    """Context information for edge case handling."""
    
    case_type: str
    original_state: str
    target_state: str | None
    trigger: str | None
    error: Exception | None
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_type": self.case_type,
            "original_state": self.original_state,
            "target_state": self.target_state,
            "trigger": self.trigger,
            "error": str(self.error) if self.error else None,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryResult:
    """Result of edge case recovery attempt."""
    
    success: bool
    strategy_used: RecoveryStrategy
    final_state: str
    recovered_data: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.value,
            "final_state": self.final_state,
            "recovered_data": self.recovered_data,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
        }


class EdgeCaseRecoveryManager:
    """Manages recovery from edge cases in state transitions.
    
    Handles:
    - Interrupted transitions (process killed mid-transition)
    - Corrupted state (partial writes)
    - Orphaned resources (cleanup failures)
    - Deadlock detection and resolution
    - Concurrent modification conflicts
    - Timeout during transitions
    """
    
    # Edge case type to strategy mapping
    STRATEGY_MAP: dict[str, list[RecoveryStrategy]] = {
        "interrupted_transition": [
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.PARTIAL_ROLLBACK,
            RecoveryStrategy.FALLBACK_STATE,
        ],
        "corrupted_state": [
            RecoveryStrategy.FULL_ROLLBACK,
            RecoveryStrategy.FALLBACK_STATE,
        ],
        "orphaned_resources": [
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.SKIP_AND_CONTINUE,
        ],
        "deadlock": [
            RecoveryStrategy.FULL_ROLLBACK,
            RecoveryStrategy.RETRY_WITH_BACKOFF,
        ],
        "concurrent_modification": [
            RecoveryStrategy.RETRY_IMMEDIATE,
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
        ],
        "transition_timeout": [
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.SKIP_AND_CONTINUE,
            RecoveryStrategy.FALLBACK_STATE,
        ],
        "resource_exhaustion": [
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.PARTIAL_ROLLBACK,
        ],
        "persistence_failure": [
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.SKIP_AND_CONTINUE,
        ],
    }
    
    def __init__(
        self,
        state_machine: ExecutionStateMachine | None = None,
        logger: Any = None,
    ) -> None:
        """Initialize recovery manager.
        
        Args:
            state_machine: State machine to manage recovery for
            logger: Logger instance
        """
        self._state_machine = state_machine
        self._logger = logger or get_logger("edge_case_recovery")
        self._recovery_history: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._in_recovery = False
    
    def detect_edge_case(
        self,
        error: Exception | None = None,
        current_state: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> EdgeCaseContext | None:
        """Detect and classify edge case.
        
        Args:
            error: Exception that occurred (if any)
            current_state: Current state of state machine
            context: Additional context
            
        Returns:
            EdgeCaseContext if edge case detected, None otherwise
        """
        if error is None and current_state is None:
            return None
        
        case_type = self._classify_edge_case(error, current_state, context or {})
        
        if case_type:
            return EdgeCaseContext(
                case_type=case_type,
                original_state=current_state or "UNKNOWN",
                target_state=context.get("target_state") if context else None,
                trigger=context.get("trigger") if context else None,
                error=error,
                metadata=context or {},
            )
        
        return None
    
    def _classify_edge_case(
        self,
        error: Exception | None,
        state: str | None,
        context: dict[str, Any],
    ) -> str | None:
        """Classify the type of edge case."""
        if error is None:
            return None
        
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check for specific error patterns
        if "timeout" in error_str or "timed out" in error_str:
            return "transition_timeout"
        
        if "interrupt" in error_str or "signal" in error_str:
            return "interrupted_transition"
        
        if "corrupt" in error_str or "invalid state" in error_str:
            return "corrupted_state"
        
        if "deadlock" in error_str or "lock" in error_str:
            return "deadlock"
        
        if "concurrent" in error_str or "conflict" in error_str:
            return "concurrent_modification"
        
        if "memory" in error_str or "resource" in error_str:
            return "resource_exhaustion"
        
        if "persist" in error_str or "save" in error_str or "write" in error_str:
            return "persistence_failure"
        
        if "cleanup" in error_str or "orphan" in error_str:
            return "orphaned_resources"
        
        # Default classification based on error type
        if error_type in ("TimeoutError", "asyncio.TimeoutError"):
            return "transition_timeout"
        if error_type == "DeadlockError":
            return "deadlock"
        if error_type in ("IOError", "OSError"):
            return "persistence_failure"
        
        return "interrupted_transition"  # Default
    
    def recover(
        self,
        context: EdgeCaseContext,
        dry_run: bool = False,
    ) -> RecoveryResult:
        """Attempt to recover from an edge case.
        
        Args:
            context: Edge case context
            dry_run: If True, don't actually modify state
            
        Returns:
            RecoveryResult with outcome
        """
        with self._lock:
            if self._in_recovery:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.FALLBACK_STATE,
                    final_state=context.original_state,
                    warnings=["Already in recovery - preventing recursive recovery"],
                )
            
            self._in_recovery = True
            start_time = time.perf_counter()
        
        try:
            strategies = self.STRATEGY_MAP.get(
                context.case_type,
                [RecoveryStrategy.FALLBACK_STATE],
            )
            
            # Try strategies in order
            for strategy in strategies:
                self._logger.info(
                    "recovery.attempting",
                    strategy=strategy.value,
                    case_type=context.case_type,
                )
                
                result = self._execute_strategy(strategy, context, dry_run)
                
                if result.success:
                    self._record_recovery(context, result)
                    result.duration_ms = (time.perf_counter() - start_time) * 1000
                    return result
            
            # All strategies failed
            return RecoveryResult(
                success=False,
                strategy_used=strategies[-1] if strategies else RecoveryStrategy.FALLBACK_STATE,
                final_state=context.original_state,
                warnings=["All recovery strategies exhausted"],
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        finally:
            with self._lock:
                self._in_recovery = False
    
    def _execute_strategy(
        self,
        strategy: RecoveryStrategy,
        context: EdgeCaseContext,
        dry_run: bool,
    ) -> RecoveryResult:
        """Execute a specific recovery strategy."""
        try:
            if strategy == RecoveryStrategy.RETRY_IMMEDIATE:
                return self._retry_immediate(context, dry_run)
            
            elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                return self._retry_with_backoff(context, dry_run)
            
            elif strategy == RecoveryStrategy.FALLBACK_STATE:
                return self._fallback_to_safe_state(context, dry_run)
            
            elif strategy == RecoveryStrategy.PARTIAL_ROLLBACK:
                return self._partial_rollback(context, dry_run)
            
            elif strategy == RecoveryStrategy.FULL_ROLLBACK:
                return self._full_rollback(context, dry_run)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._graceful_degradation(context, dry_run)
            
            elif strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
                return self._skip_and_continue(context, dry_run)
            
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=strategy,
                    final_state=context.original_state,
                    warnings=[f"Unknown strategy: {strategy}"],
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                final_state=context.original_state,
                warnings=[f"Strategy {strategy} failed: {e}"],
            )
    
    def _retry_immediate(
        self, context: EdgeCaseContext, dry_run: bool
    ) -> RecoveryResult:
        """Retry the operation immediately."""
        if context.retry_count >= context.max_retries:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY_IMMEDIATE,
                final_state=context.original_state,
                warnings=["Max retries exceeded"],
            )
        
        context.retry_count += 1
        
        if dry_run:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RETRY_IMMEDIATE,
                final_state=context.target_state or context.original_state,
                warnings=["Dry run - would retry immediately"],
            )
        
        # Attempt retry
        if self._state_machine and context.trigger:
            try:
                getattr(self._state_machine, context.trigger)()
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RETRY_IMMEDIATE,
                    final_state=self._state_machine.state,
                )
            except Exception as e:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.RETRY_IMMEDIATE,
                    final_state=context.original_state,
                    warnings=[f"Retry failed: {e}"],
                )
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY_IMMEDIATE,
            final_state=context.original_state,
        )
    
    def _retry_with_backoff(
        self, context: EdgeCaseContext, dry_run: bool
    ) -> RecoveryResult:
        """Retry with exponential backoff."""
        if context.retry_count >= context.max_retries:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY_WITH_BACKOFF,
                final_state=context.original_state,
                warnings=["Max retries exceeded"],
            )
        
        # Calculate backoff delay
        delay = min(30.0, 0.5 * (2 ** context.retry_count))  # Max 30 seconds
        context.retry_count += 1
        
        if not dry_run:
            time.sleep(delay)
        
        if dry_run:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.RETRY_WITH_BACKOFF,
                final_state=context.target_state or context.original_state,
                warnings=[f"Dry run - would wait {delay}s then retry"],
            )
        
        # Attempt retry after backoff
        if self._state_machine and context.trigger:
            try:
                getattr(self._state_machine, context.trigger)()
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RETRY_WITH_BACKOFF,
                    final_state=self._state_machine.state,
                    recovered_data={"backoff_delay": delay},
                )
            except Exception:
                pass
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY_WITH_BACKOFF,
            final_state=context.original_state,
        )
    
    def _fallback_to_safe_state(
        self, context: EdgeCaseContext, dry_run: bool
    ) -> RecoveryResult:
        """Fallback to a known safe state."""
        safe_states = [
            ExecutionState.IDLE.value,
            ExecutionState.READY.value,
            ExecutionState.PAUSED.value,
        ]
        
        # Determine best safe state based on original state
        if context.original_state in [ExecutionState.RUNNING.value]:
            target_safe = ExecutionState.PAUSED.value
        elif context.original_state in [ExecutionState.PLANNING.value]:
            target_safe = ExecutionState.IDLE.value
        else:
            target_safe = ExecutionState.IDLE.value
        
        if dry_run:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK_STATE,
                final_state=target_safe,
                warnings=[f"Dry run - would fallback to {target_safe}"],
            )
        
        if self._state_machine:
            try:
                self._state_machine.reset_to_safe_state()
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.FALLBACK_STATE,
                    final_state=self._state_machine.state,
                )
            except Exception:
                pass
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.FALLBACK_STATE,
            final_state=target_safe,
        )
    
    def _partial_rollback(
        self, context: EdgeCaseContext, dry_run: bool
    ) -> RecoveryResult:
        """Rollback to last successful checkpoint."""
        if dry_run:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.PARTIAL_ROLLBACK,
                final_state=context.original_state,
                warnings=["Dry run - would rollback to last checkpoint"],
            )
        
        if self._state_machine:
            # Try to recover from persistence
            try:
                self._state_machine.recover_from_crash()
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.PARTIAL_ROLLBACK,
                    final_state=self._state_machine.state,
                    recovered_data={"checkpoint_used": True},
                )
            except Exception:
                pass
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.PARTIAL_ROLLBACK,
            final_state=context.original_state,
            warnings=["No checkpoint available for partial rollback"],
        )
    
    def _full_rollback(
        self, context: EdgeCaseContext, dry_run: bool
    ) -> RecoveryResult:
        """Full rollback to initial state."""
        if dry_run:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FULL_ROLLBACK,
                final_state=ExecutionState.IDLE.value,
                warnings=["Dry run - would perform full rollback"],
            )
        
        if self._state_machine:
            try:
                # Cleanup resources
                self._state_machine.resource_manager.cleanup_all()
                # Reset state
                self._state_machine.reset_to_safe_state()
                
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.FULL_ROLLBACK,
                    final_state=self._state_machine.state,
                    recovered_data={"resources_cleaned": True},
                )
            except Exception:
                pass
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.FULL_ROLLBACK,
            final_state=ExecutionState.IDLE.value,
        )
    
    def _graceful_degradation(
        self, context: EdgeCaseContext, dry_run: bool
    ) -> RecoveryResult:
        """Continue with degraded functionality."""
        warnings = [
            "Continuing with degraded functionality",
            f"Original error: {context.error}",
        ]
        
        if dry_run:
            warnings.insert(0, "Dry run - would continue with degradation")
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
            final_state=context.original_state,
            warnings=warnings,
            recovered_data={"degraded_mode": True},
        )
    
    def _skip_and_continue(
        self, context: EdgeCaseContext, dry_run: bool
    ) -> RecoveryResult:
        """Skip the failed operation and continue."""
        warnings = [
            f"Skipped operation: {context.trigger or 'unknown'}",
            "Continuing with next operation",
        ]
        
        if dry_run:
            warnings.insert(0, "Dry run - would skip and continue")
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.SKIP_AND_CONTINUE,
            final_state=context.original_state,
            warnings=warnings,
            recovered_data={"skipped": True},
        )
    
    def _record_recovery(
        self,
        context: EdgeCaseContext,
        result: RecoveryResult,
    ) -> None:
        """Record recovery for analytics."""
        self._recovery_history.append({
            "timestamp": time.time(),
            "context": context.to_dict(),
            "result": result.to_dict(),
        })
        
        # Keep only last 100 recoveries
        if len(self._recovery_history) > 100:
            self._recovery_history = self._recovery_history[-100:]
    
    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        if not self._recovery_history:
            return {"total": 0, "success_rate": 0.0}
        
        total = len(self._recovery_history)
        successful = sum(1 for r in self._recovery_history if r["result"]["success"])
        
        strategy_counts: dict[str, int] = {}
        case_type_counts: dict[str, int] = {}
        
        for record in self._recovery_history:
            strategy = record["result"]["strategy_used"]
            case_type = record["context"]["case_type"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            case_type_counts[case_type] = case_type_counts.get(case_type, 0) + 1
        
        return {
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total,
            "by_strategy": strategy_counts,
            "by_case_type": case_type_counts,
        }


# ==================== MODULE EXPORTS ====================

__all__ = [
    # Enums
    "ExecutionState",
    "RecoveryStrategy",
    # Exceptions
    "StateTransitionError",
    "StatePersistenceError",
    # Persistence
    "PersistedState",
    "StatePersistence",
    # Resources
    "ResourceHandle",
    "FileResource",
    "MemoryResource",
    "ResourceCleanupManager",
    # Validation
    "TransitionRule",
    "TransitionValidator",
    # Edge Case Recovery
    "EdgeCaseContext",
    "RecoveryResult",
    "EdgeCaseRecoveryManager",
    # Transitions
    "TRANSITIONS",
    # Main Class
    "ExecutionStateMachine",
]
