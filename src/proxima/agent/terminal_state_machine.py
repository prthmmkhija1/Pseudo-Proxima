"""Terminal State Machine Manager for Proxima Agent.

Phase 3: Terminal State Machine with Event Debouncing

Provides:
- Complete state machine for terminal lifecycle
- Transition validation
- Event debouncing for rapid updates
- State persistence and recovery
- Metrics tracking per state

The state machine follows this flow::

    PENDING -> STARTING -> RUNNING -> COMPLETED
                                   |-> FAILED
                                   |-> TIMEOUT
                                   |-> CANCELLED
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from proxima.utils.logging import get_logger

logger = get_logger("agent.terminal_state_machine")


# =============================================================================
# State Definitions
# =============================================================================

class TerminalProcessState(Enum):
    """Terminal process states for state machine."""
    
    PENDING = "pending"       # Command queued, not started
    STARTING = "starting"     # Process being created
    RUNNING = "running"       # Process actively executing
    PAUSED = "paused"         # Process suspended (if supported)
    COMPLETED = "completed"   # Process finished successfully (code 0)
    FAILED = "failed"         # Process exited with error (code != 0)
    TIMEOUT = "timeout"       # Process killed due to timeout
    CANCELLED = "cancelled"   # Process cancelled by user
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal (final) state."""
        return self in {
            TerminalProcessState.COMPLETED,
            TerminalProcessState.FAILED,
            TerminalProcessState.TIMEOUT,
            TerminalProcessState.CANCELLED,
        }
    
    def is_active(self) -> bool:
        """Check if process is in an active state."""
        return self in {
            TerminalProcessState.STARTING,
            TerminalProcessState.RUNNING,
            TerminalProcessState.PAUSED,
        }
    
    def is_idle(self) -> bool:
        """Check if process hasn't started yet."""
        return self == TerminalProcessState.PENDING


# =============================================================================
# State Transitions
# =============================================================================

@dataclass
class StateTransition:
    """Definition of a valid state transition."""
    from_state: TerminalProcessState
    to_state: TerminalProcessState
    condition: Optional[Callable[[], bool]] = None
    on_transition: Optional[Callable[["StateContext"], None]] = None
    
    def can_transition(self) -> bool:
        """Check if transition condition is met."""
        if self.condition is None:
            return True
        return self.condition()


@dataclass
class StateContext:
    """Context passed during state transitions."""
    process_id: str
    from_state: TerminalProcessState
    to_state: TerminalProcessState
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Valid state transitions
VALID_TRANSITIONS: Dict[TerminalProcessState, Set[TerminalProcessState]] = {
    TerminalProcessState.PENDING: {
        TerminalProcessState.STARTING,
        TerminalProcessState.CANCELLED,
    },
    TerminalProcessState.STARTING: {
        TerminalProcessState.RUNNING,
        TerminalProcessState.FAILED,
        TerminalProcessState.CANCELLED,
        TerminalProcessState.TIMEOUT,
    },
    TerminalProcessState.RUNNING: {
        TerminalProcessState.PAUSED,
        TerminalProcessState.COMPLETED,
        TerminalProcessState.FAILED,
        TerminalProcessState.TIMEOUT,
        TerminalProcessState.CANCELLED,
    },
    TerminalProcessState.PAUSED: {
        TerminalProcessState.RUNNING,
        TerminalProcessState.CANCELLED,
        TerminalProcessState.TIMEOUT,
    },
    # Terminal states have no outgoing transitions
    TerminalProcessState.COMPLETED: set(),
    TerminalProcessState.FAILED: set(),
    TerminalProcessState.TIMEOUT: set(),
    TerminalProcessState.CANCELLED: set(),
}


# =============================================================================
# Event Types and Debouncing
# =============================================================================

class TerminalEventType(Enum):
    """Types of terminal state events."""
    STATE_CHANGED = "state_changed"
    OUTPUT_RECEIVED = "output_received"
    ERROR_RECEIVED = "error_received"
    PROGRESS_UPDATED = "progress_updated"
    METRICS_UPDATED = "metrics_updated"


@dataclass
class TerminalStateEvent:
    """Event emitted by state machine."""
    event_type: TerminalEventType
    process_id: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "process_id": self.process_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class EventDebouncer:
    """Debounce rapid events to prevent UI thrashing.
    
    Groups rapid events and emits aggregated updates at a controlled rate.
    
    Example:
        >>> debouncer = EventDebouncer(interval=0.1)  # 100ms debounce
        >>> debouncer.add_listener(handle_event)
        >>> 
        >>> # These rapid events will be batched
        >>> debouncer.emit(event1)
        >>> debouncer.emit(event2)
        >>> debouncer.emit(event3)  # All emitted as one batch
    """
    
    def __init__(
        self,
        interval: float = 0.1,
        max_batch_size: int = 100,
    ):
        """Initialize the debouncer.
        
        Args:
            interval: Debounce interval in seconds
            max_batch_size: Maximum events per batch
        """
        self.interval = interval
        self.max_batch_size = max_batch_size
        
        self._pending_events: Dict[str, List[TerminalStateEvent]] = defaultdict(list)
        self._listeners: List[Callable[[List[TerminalStateEvent]], None]] = []
        self._timers: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    def add_listener(
        self,
        callback: Callable[[List[TerminalStateEvent]], None],
    ) -> None:
        """Add an event listener.
        
        Args:
            callback: Function to call with batched events
        """
        if callback not in self._listeners:
            self._listeners.append(callback)
    
    def remove_listener(
        self,
        callback: Callable[[List[TerminalStateEvent]], None],
    ) -> None:
        """Remove an event listener.
        
        Args:
            callback: Function to remove
        """
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    async def emit(self, event: TerminalStateEvent) -> None:
        """Emit an event (will be debounced).
        
        Args:
            event: Event to emit
        """
        process_id = event.process_id
        
        async with self._lock:
            self._pending_events[process_id].append(event)
            
            # Check if we should flush immediately (max batch size)
            if len(self._pending_events[process_id]) >= self.max_batch_size:
                await self._flush_events(process_id)
                return
            
            # Cancel existing timer
            if process_id in self._timers:
                self._timers[process_id].cancel()
            
            # Start new timer
            self._timers[process_id] = asyncio.create_task(
                self._delayed_flush(process_id)
            )
    
    async def _delayed_flush(self, process_id: str) -> None:
        """Flush events after delay.
        
        Args:
            process_id: Process ID to flush
        """
        await asyncio.sleep(self.interval)
        await self._flush_events(process_id)
    
    async def _flush_events(self, process_id: str) -> None:
        """Flush pending events for a process.
        
        Args:
            process_id: Process ID
        """
        async with self._lock:
            events = self._pending_events.pop(process_id, [])
            if process_id in self._timers:
                del self._timers[process_id]
        
        if events:
            self._notify_listeners(events)
    
    def _notify_listeners(self, events: List[TerminalStateEvent]) -> None:
        """Notify all listeners.
        
        Args:
            events: List of events to send
        """
        for listener in self._listeners:
            try:
                listener(events)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
    
    async def flush_all(self) -> None:
        """Flush all pending events immediately."""
        async with self._lock:
            process_ids = list(self._pending_events.keys())
        
        for process_id in process_ids:
            await self._flush_events(process_id)
    
    def emit_sync(self, event: TerminalStateEvent) -> None:
        """Emit an event synchronously (for non-async contexts).
        
        Note: Creates a new task, so event may be batched.
        
        Args:
            event: Event to emit
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event))
        except RuntimeError:
            # No running loop - emit directly
            self._pending_events[event.process_id].append(event)


# =============================================================================
# Process Metrics
# =============================================================================

@dataclass
class ProcessMetrics:
    """Metrics for a terminal process."""
    process_id: str
    state: TerminalProcessState = TerminalProcessState.PENDING
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    
    # State durations
    time_in_pending: float = 0.0
    time_in_starting: float = 0.0
    time_in_running: float = 0.0
    time_in_paused: float = 0.0
    
    # Output statistics
    stdout_lines: int = 0
    stderr_lines: int = 0
    total_bytes: int = 0
    
    # State transition count
    transition_count: int = 0
    
    @property
    def total_time(self) -> float:
        """Get total time from creation to end."""
        end = self.ended_at or time.time()
        return end - self.created_at
    
    @property
    def active_time(self) -> float:
        """Get time spent in active states."""
        return (
            self.time_in_starting +
            self.time_in_running +
            self.time_in_paused
        )
    
    def update_state_time(
        self,
        old_state: TerminalProcessState,
        duration: float,
    ) -> None:
        """Update time spent in a state.
        
        Args:
            old_state: State that was exited
            duration: Time spent in state
        """
        if old_state == TerminalProcessState.PENDING:
            self.time_in_pending += duration
        elif old_state == TerminalProcessState.STARTING:
            self.time_in_starting += duration
        elif old_state == TerminalProcessState.RUNNING:
            self.time_in_running += duration
        elif old_state == TerminalProcessState.PAUSED:
            self.time_in_paused += duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "process_id": self.process_id,
            "state": self.state.value,
            "total_time": self.total_time,
            "active_time": self.active_time,
            "time_in_pending": self.time_in_pending,
            "time_in_starting": self.time_in_starting,
            "time_in_running": self.time_in_running,
            "time_in_paused": self.time_in_paused,
            "stdout_lines": self.stdout_lines,
            "stderr_lines": self.stderr_lines,
            "total_bytes": self.total_bytes,
            "transition_count": self.transition_count,
        }


# =============================================================================
# Terminal State Machine
# =============================================================================

class TerminalStateMachine:
    """State machine manager for terminal processes.
    
    Manages state transitions, validates transitions, tracks metrics,
    and emits debounced events for UI updates.
    
    Example:
        >>> machine = TerminalStateMachine()
        >>> machine.add_listener(handle_state_change)
        >>> 
        >>> # Create a process
        >>> machine.create_process("proc_1", "pip install qiskit")
        >>> 
        >>> # Transition through states
        >>> machine.transition("proc_1", TerminalProcessState.STARTING)
        >>> machine.transition("proc_1", TerminalProcessState.RUNNING)
        >>> machine.transition("proc_1", TerminalProcessState.COMPLETED)
        >>> 
        >>> # Get metrics
        >>> metrics = machine.get_metrics("proc_1")
    """
    
    def __init__(
        self,
        debounce_interval: float = 0.1,
    ):
        """Initialize the state machine.
        
        Args:
            debounce_interval: Event debounce interval
        """
        self._processes: Dict[str, TerminalProcessState] = {}
        self._metrics: Dict[str, ProcessMetrics] = {}
        self._state_entered_at: Dict[str, float] = {}
        self._process_commands: Dict[str, str] = {}
        
        # Event handling
        self._debouncer = EventDebouncer(interval=debounce_interval)
        self._state_listeners: List[Callable[[StateContext], None]] = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def add_listener(
        self,
        callback: Callable[[StateContext], None],
    ) -> None:
        """Add a state change listener.
        
        Args:
            callback: Function to call on state change
        """
        if callback not in self._state_listeners:
            self._state_listeners.append(callback)
    
    def remove_listener(
        self,
        callback: Callable[[StateContext], None],
    ) -> None:
        """Remove a state change listener.
        
        Args:
            callback: Function to remove
        """
        if callback in self._listeners:
            self._state_listeners.remove(callback)
    
    def add_event_listener(
        self,
        callback: Callable[[List[TerminalStateEvent]], None],
    ) -> None:
        """Add a debounced event listener.
        
        Args:
            callback: Function to call with batched events
        """
        self._debouncer.add_listener(callback)
    
    def remove_event_listener(
        self,
        callback: Callable[[List[TerminalStateEvent]], None],
    ) -> None:
        """Remove a debounced event listener.
        
        Args:
            callback: Function to remove
        """
        self._debouncer.remove_listener(callback)
    
    def create_process(
        self,
        process_id: str,
        command: str,
    ) -> TerminalProcessState:
        """Create a new process in PENDING state.
        
        Args:
            process_id: Unique process ID
            command: Command being executed
            
        Returns:
            Initial state (PENDING)
            
        Raises:
            ValueError: If process_id already exists
        """
        if process_id in self._processes:
            raise ValueError(f"Process {process_id} already exists")
        
        self._processes[process_id] = TerminalProcessState.PENDING
        self._metrics[process_id] = ProcessMetrics(process_id=process_id)
        self._state_entered_at[process_id] = time.time()
        self._process_commands[process_id] = command
        
        logger.debug(f"Created process {process_id}: '{command[:50]}'")
        
        return TerminalProcessState.PENDING
    
    def get_state(self, process_id: str) -> Optional[TerminalProcessState]:
        """Get current state of a process.
        
        Args:
            process_id: Process ID
            
        Returns:
            Current state or None if not found
        """
        return self._processes.get(process_id)
    
    def can_transition(
        self,
        process_id: str,
        to_state: TerminalProcessState,
    ) -> bool:
        """Check if transition is valid.
        
        Args:
            process_id: Process ID
            to_state: Target state
            
        Returns:
            True if transition is valid
        """
        current = self._processes.get(process_id)
        if current is None:
            return False
        
        valid = VALID_TRANSITIONS.get(current, set())
        return to_state in valid
    
    async def transition(
        self,
        process_id: str,
        to_state: TerminalProcessState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Transition a process to a new state.
        
        Args:
            process_id: Process ID
            to_state: Target state
            metadata: Optional transition metadata
            
        Returns:
            True if transition succeeded
        """
        async with self._lock:
            return await self._do_transition(process_id, to_state, metadata)
    
    async def _do_transition(
        self,
        process_id: str,
        to_state: TerminalProcessState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Internal transition implementation.
        
        Args:
            process_id: Process ID
            to_state: Target state
            metadata: Optional transition metadata
            
        Returns:
            True if transition succeeded
        """
        if not self.can_transition(process_id, to_state):
            logger.warning(
                f"Invalid transition for {process_id}: "
                f"{self._processes.get(process_id)} -> {to_state}"
            )
            return False
        
        from_state = self._processes[process_id]
        now = time.time()
        
        # Update time spent in previous state
        entered_at = self._state_entered_at.get(process_id, now)
        duration = now - entered_at
        metrics = self._metrics[process_id]
        metrics.update_state_time(from_state, duration)
        metrics.transition_count += 1
        metrics.state = to_state
        
        # Update timing for terminal states
        if to_state == TerminalProcessState.STARTING:
            metrics.started_at = now
        elif to_state.is_terminal():
            metrics.ended_at = now
        
        # Update state
        self._processes[process_id] = to_state
        self._state_entered_at[process_id] = now
        
        # Create context
        context = StateContext(
            process_id=process_id,
            from_state=from_state,
            to_state=to_state,
            timestamp=now,
            metadata=metadata or {},
        )
        
        # Notify listeners
        self._notify_state_listeners(context)
        
        # Emit debounced event
        event = TerminalStateEvent(
            event_type=TerminalEventType.STATE_CHANGED,
            process_id=process_id,
            data={
                "from_state": from_state.value,
                "to_state": to_state.value,
                "metadata": metadata,
            },
        )
        await self._debouncer.emit(event)
        
        logger.debug(f"Process {process_id}: {from_state.value} -> {to_state.value}")
        
        return True
    
    def transition_sync(
        self,
        process_id: str,
        to_state: TerminalProcessState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Synchronous transition (for non-async contexts).
        
        Args:
            process_id: Process ID
            to_state: Target state
            metadata: Optional transition metadata
            
        Returns:
            True if transition succeeded
        """
        if not self.can_transition(process_id, to_state):
            return False
        
        from_state = self._processes[process_id]
        now = time.time()
        
        # Update time spent in previous state
        entered_at = self._state_entered_at.get(process_id, now)
        duration = now - entered_at
        metrics = self._metrics[process_id]
        metrics.update_state_time(from_state, duration)
        metrics.transition_count += 1
        metrics.state = to_state
        
        # Update timing for terminal states
        if to_state == TerminalProcessState.STARTING:
            metrics.started_at = now
        elif to_state.is_terminal():
            metrics.ended_at = now
        
        # Update state
        self._processes[process_id] = to_state
        self._state_entered_at[process_id] = now
        
        # Create context
        context = StateContext(
            process_id=process_id,
            from_state=from_state,
            to_state=to_state,
            timestamp=now,
            metadata=metadata or {},
        )
        
        # Notify listeners
        self._notify_state_listeners(context)
        
        # Emit event synchronously
        event = TerminalStateEvent(
            event_type=TerminalEventType.STATE_CHANGED,
            process_id=process_id,
            data={
                "from_state": from_state.value,
                "to_state": to_state.value,
                "metadata": metadata,
            },
        )
        self._debouncer.emit_sync(event)
        
        return True
    
    def _notify_state_listeners(self, context: StateContext) -> None:
        """Notify state change listeners.
        
        Args:
            context: State change context
        """
        for listener in self._state_listeners:
            try:
                listener(context)
            except Exception as e:
                logger.error(f"State listener error: {e}")
    
    def record_output(
        self,
        process_id: str,
        content: str,
        is_stderr: bool = False,
    ) -> None:
        """Record output for metrics.
        
        Args:
            process_id: Process ID
            content: Output content
            is_stderr: Whether content is from stderr
        """
        if process_id not in self._metrics:
            return
        
        metrics = self._metrics[process_id]
        if is_stderr:
            metrics.stderr_lines += 1
        else:
            metrics.stdout_lines += 1
        metrics.total_bytes += len(content.encode('utf-8'))
        
        # Emit debounced output event
        event = TerminalStateEvent(
            event_type=(
                TerminalEventType.ERROR_RECEIVED if is_stderr
                else TerminalEventType.OUTPUT_RECEIVED
            ),
            process_id=process_id,
            data={"content": content, "is_stderr": is_stderr},
        )
        self._debouncer.emit_sync(event)
    
    def get_metrics(self, process_id: str) -> Optional[ProcessMetrics]:
        """Get metrics for a process.
        
        Args:
            process_id: Process ID
            
        Returns:
            ProcessMetrics or None
        """
        return self._metrics.get(process_id)
    
    def get_all_metrics(self) -> Dict[str, ProcessMetrics]:
        """Get metrics for all processes.
        
        Returns:
            Dictionary of process_id -> ProcessMetrics
        """
        return dict(self._metrics)
    
    def get_processes_in_state(
        self,
        state: TerminalProcessState,
    ) -> List[str]:
        """Get all processes in a specific state.
        
        Args:
            state: State to filter by
            
        Returns:
            List of process IDs
        """
        return [
            pid for pid, s in self._processes.items()
            if s == state
        ]
    
    def get_active_processes(self) -> List[str]:
        """Get all active (non-terminal) processes.
        
        Returns:
            List of process IDs
        """
        return [
            pid for pid, state in self._processes.items()
            if state.is_active()
        ]
    
    def get_completed_processes(self) -> List[str]:
        """Get all completed processes.
        
        Returns:
            List of process IDs
        """
        return [
            pid for pid, state in self._processes.items()
            if state.is_terminal()
        ]
    
    def cleanup_process(self, process_id: str) -> bool:
        """Remove a process from tracking.
        
        Args:
            process_id: Process ID
            
        Returns:
            True if process was removed
        """
        if process_id not in self._processes:
            return False
        
        del self._processes[process_id]
        self._metrics.pop(process_id, None)
        self._state_entered_at.pop(process_id, None)
        self._process_commands.pop(process_id, None)
        
        return True
    
    def cleanup_completed(self) -> int:
        """Remove all completed processes.
        
        Returns:
            Number of processes removed
        """
        completed = self.get_completed_processes()
        for pid in completed:
            self.cleanup_process(pid)
        return len(completed)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Summary dictionary
        """
        states = defaultdict(int)
        for state in self._processes.values():
            states[state.value] += 1
        
        total_active_time = sum(
            m.active_time for m in self._metrics.values()
        )
        total_output_lines = sum(
            m.stdout_lines + m.stderr_lines for m in self._metrics.values()
        )
        
        return {
            "total_processes": len(self._processes),
            "states": dict(states),
            "total_active_time": total_active_time,
            "total_output_lines": total_output_lines,
            "total_transitions": sum(
                m.transition_count for m in self._metrics.values()
            ),
        }


# =============================================================================
# Global Instance
# =============================================================================

_state_machine: Optional[TerminalStateMachine] = None


def get_terminal_state_machine() -> TerminalStateMachine:
    """Get the global terminal state machine instance."""
    global _state_machine
    if _state_machine is None:
        _state_machine = TerminalStateMachine()
    return _state_machine
