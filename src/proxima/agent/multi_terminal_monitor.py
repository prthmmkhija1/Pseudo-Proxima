"""Multi-Terminal Monitor for Proxima Agent.

Provides real-time monitoring of multiple terminal sessions:
- Live output streaming from multiple terminals
- Terminal event aggregation
- Output buffering and history
- Terminal state tracking

This enables the AI agent to observe and coordinate
multiple concurrent terminal executions.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from proxima.utils.logging import get_logger

logger = get_logger("agent.multi_terminal")


class TerminalEventType(Enum):
    """Types of terminal events."""
    OUTPUT = auto()       # New output line
    ERROR = auto()        # Error output
    STARTED = auto()      # Command started
    COMPLETED = auto()    # Command completed
    FAILED = auto()       # Command failed
    TIMEOUT = auto()      # Command timed out
    CANCELLED = auto()    # Command cancelled
    STATE_CHANGE = auto() # Terminal state changed


@dataclass
class TerminalEvent:
    """An event from a terminal session."""
    terminal_id: str
    event_type: TerminalEventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "terminal_id": self.terminal_id,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "data": self.data,
            "message": self.message,
        }


class TerminalState(Enum):
    """State of a monitored terminal."""
    IDLE = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()


@dataclass
class MonitoredTerminal:
    """A terminal being monitored."""
    id: str
    name: str
    state: TerminalState = TerminalState.IDLE
    current_command: Optional[str] = None
    started_at: Optional[str] = None
    working_dir: str = ""
    output_buffer: Deque[str] = field(default_factory=lambda: deque(maxlen=1000))
    error_buffer: Deque[str] = field(default_factory=lambda: deque(maxlen=100))
    return_code: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_recent_output(self, lines: int = 50) -> List[str]:
        """Get recent output lines."""
        output_list = list(self.output_buffer)
        return output_list[-lines:] if lines < len(output_list) else output_list
    
    def get_all_output(self) -> str:
        """Get all buffered output."""
        return "\n".join(self.output_buffer)
    
    def clear_buffer(self) -> None:
        """Clear output buffers."""
        self.output_buffer.clear()
        self.error_buffer.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.name,
            "current_command": self.current_command,
            "started_at": self.started_at,
            "working_dir": self.working_dir,
            "output_lines": len(self.output_buffer),
            "error_lines": len(self.error_buffer),
            "return_code": self.return_code,
            "metadata": self.metadata,
        }


class MultiTerminalMonitor:
    """Monitor multiple terminal sessions with real-time output.
    
    Provides:
    - Registration and tracking of multiple terminals
    - Real-time event streaming
    - Output aggregation and buffering
    - State management for terminals
    
    Example:
        >>> monitor = MultiTerminalMonitor()
        >>> 
        >>> # Register callback for events
        >>> def on_event(event):
        ...     print(f"[{event.terminal_id}] {event.message}")
        >>> monitor.add_event_listener(on_event)
        >>> 
        >>> # Register a terminal
        >>> monitor.register_terminal("build", "Backend Build")
        >>> 
        >>> # Push events (usually from executor)
        >>> monitor.push_output("build", "Building...")
        >>> monitor.push_completed("build", 0)
    """
    
    def __init__(
        self,
        max_output_lines: int = 1000,
        event_queue_size: int = 10000,
    ):
        """Initialize the monitor.
        
        Args:
            max_output_lines: Maximum output lines per terminal
            event_queue_size: Maximum queued events
        """
        self.max_output_lines = max_output_lines
        self._terminals: Dict[str, MonitoredTerminal] = {}
        self._event_queue: queue.Queue[TerminalEvent] = queue.Queue(maxsize=event_queue_size)
        self._event_listeners: List[Callable[[TerminalEvent], None]] = []
        self._lock = threading.RLock()
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._running = False
        
        logger.info("MultiTerminalMonitor initialized")
    
    def start(self) -> None:
        """Start the event dispatcher."""
        if self._running:
            return
        
        self._running = True
        self._dispatcher_thread = threading.Thread(
            target=self._event_dispatcher,
            daemon=True,
            name="terminal-event-dispatcher",
        )
        self._dispatcher_thread.start()
        logger.info("Event dispatcher started")
    
    def stop(self) -> None:
        """Stop the event dispatcher."""
        self._running = False
        if self._dispatcher_thread:
            # Push a dummy event to unblock the queue
            try:
                self._event_queue.put_nowait(TerminalEvent(
                    terminal_id="__shutdown__",
                    event_type=TerminalEventType.STATE_CHANGE,
                ))
            except queue.Full:
                pass
            self._dispatcher_thread.join(timeout=2.0)
            self._dispatcher_thread = None
        logger.info("Event dispatcher stopped")
    
    def _event_dispatcher(self) -> None:
        """Dispatch events to listeners."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.5)
                if event.terminal_id == "__shutdown__":
                    continue
                
                # Dispatch to listeners
                for listener in self._event_listeners:
                    try:
                        listener(event)
                    except Exception as e:
                        logger.error(f"Event listener error: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Event dispatcher error: {e}")
    
    def add_event_listener(
        self,
        listener: Callable[[TerminalEvent], None],
    ) -> None:
        """Add an event listener.
        
        Args:
            listener: Callback function for events
        """
        if listener not in self._event_listeners:
            self._event_listeners.append(listener)
    
    def remove_event_listener(
        self,
        listener: Callable[[TerminalEvent], None],
    ) -> bool:
        """Remove an event listener.
        
        Args:
            listener: Listener to remove
            
        Returns:
            True if removed
        """
        try:
            self._event_listeners.remove(listener)
            return True
        except ValueError:
            return False
    
    def _emit_event(self, event: TerminalEvent) -> None:
        """Emit an event to the queue."""
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("Event queue full, dropping event")
    
    def register_terminal(
        self,
        terminal_id: str,
        name: str,
        working_dir: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonitoredTerminal:
        """Register a terminal for monitoring.
        
        Args:
            terminal_id: Unique terminal ID
            name: Display name
            working_dir: Working directory
            metadata: Additional metadata
            
        Returns:
            Monitored terminal object
        """
        with self._lock:
            terminal = MonitoredTerminal(
                id=terminal_id,
                name=name,
                working_dir=working_dir,
                output_buffer=deque(maxlen=self.max_output_lines),
                metadata=metadata or {},
            )
            self._terminals[terminal_id] = terminal
            
            self._emit_event(TerminalEvent(
                terminal_id=terminal_id,
                event_type=TerminalEventType.STATE_CHANGE,
                message=f"Terminal '{name}' registered",
                data={"state": "registered"},
            ))
            
            logger.info(f"Registered terminal: {terminal_id} ({name})")
            return terminal
    
    def unregister_terminal(self, terminal_id: str) -> bool:
        """Unregister a terminal.
        
        Args:
            terminal_id: Terminal ID to unregister
            
        Returns:
            True if unregistered
        """
        with self._lock:
            if terminal_id in self._terminals:
                del self._terminals[terminal_id]
                
                self._emit_event(TerminalEvent(
                    terminal_id=terminal_id,
                    event_type=TerminalEventType.STATE_CHANGE,
                    message="Terminal unregistered",
                    data={"state": "unregistered"},
                ))
                
                return True
            return False
    
    def get_terminal(self, terminal_id: str) -> Optional[MonitoredTerminal]:
        """Get a monitored terminal by ID."""
        return self._terminals.get(terminal_id)
    
    def get_all_terminals(self) -> List[MonitoredTerminal]:
        """Get all monitored terminals."""
        return list(self._terminals.values())
    
    def push_command_started(
        self,
        terminal_id: str,
        command: str,
        working_dir: Optional[str] = None,
    ) -> None:
        """Signal that a command has started.
        
        Args:
            terminal_id: Terminal ID
            command: Command being executed
            working_dir: Working directory
        """
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                terminal.state = TerminalState.RUNNING
                terminal.current_command = command
                terminal.started_at = datetime.now().isoformat()
                terminal.return_code = None
                if working_dir:
                    terminal.working_dir = working_dir
        
        self._emit_event(TerminalEvent(
            terminal_id=terminal_id,
            event_type=TerminalEventType.STARTED,
            message=f"Started: {command}",
            data={"command": command, "working_dir": working_dir},
        ))
    
    def push_output(
        self,
        terminal_id: str,
        line: str,
        is_error: bool = False,
    ) -> None:
        """Push an output line from a terminal.
        
        Args:
            terminal_id: Terminal ID
            line: Output line
            is_error: Whether this is stderr
        """
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                if is_error:
                    terminal.error_buffer.append(line)
                else:
                    terminal.output_buffer.append(line)
        
        self._emit_event(TerminalEvent(
            terminal_id=terminal_id,
            event_type=TerminalEventType.ERROR if is_error else TerminalEventType.OUTPUT,
            message=line,
            data={"is_error": is_error},
        ))
    
    def push_completed(
        self,
        terminal_id: str,
        return_code: int,
        execution_time_ms: Optional[float] = None,
    ) -> None:
        """Signal that a command has completed.
        
        Args:
            terminal_id: Terminal ID
            return_code: Command return code
            execution_time_ms: Execution time in milliseconds
        """
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                terminal.state = TerminalState.COMPLETED if return_code == 0 else TerminalState.FAILED
                terminal.return_code = return_code
        
        event_type = TerminalEventType.COMPLETED if return_code == 0 else TerminalEventType.FAILED
        
        self._emit_event(TerminalEvent(
            terminal_id=terminal_id,
            event_type=event_type,
            message=f"Completed with code {return_code}",
            data={
                "return_code": return_code,
                "execution_time_ms": execution_time_ms,
            },
        ))
    
    def push_timeout(
        self,
        terminal_id: str,
        timeout_seconds: float,
    ) -> None:
        """Signal that a command timed out.
        
        Args:
            terminal_id: Terminal ID
            timeout_seconds: Timeout value
        """
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                terminal.state = TerminalState.FAILED
                terminal.return_code = -1
        
        self._emit_event(TerminalEvent(
            terminal_id=terminal_id,
            event_type=TerminalEventType.TIMEOUT,
            message=f"Command timed out after {timeout_seconds}s",
            data={"timeout_seconds": timeout_seconds},
        ))
    
    def push_cancelled(self, terminal_id: str) -> None:
        """Signal that a command was cancelled.
        
        Args:
            terminal_id: Terminal ID
        """
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                terminal.state = TerminalState.FAILED
                terminal.return_code = -1
        
        self._emit_event(TerminalEvent(
            terminal_id=terminal_id,
            event_type=TerminalEventType.CANCELLED,
            message="Command cancelled",
        ))
    
    def get_aggregated_output(
        self,
        terminal_ids: Optional[List[str]] = None,
        lines: int = 100,
    ) -> Dict[str, List[str]]:
        """Get aggregated output from multiple terminals.
        
        Args:
            terminal_ids: Terminals to include (all if None)
            lines: Lines per terminal
            
        Returns:
            Dict of terminal_id -> output lines
        """
        result = {}
        ids_to_check = terminal_ids or list(self._terminals.keys())
        
        for terminal_id in ids_to_check:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                result[terminal_id] = terminal.get_recent_output(lines)
        
        return result
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all terminal statuses.
        
        Returns:
            Summary dictionary
        """
        with self._lock:
            terminals = []
            running_count = 0
            completed_count = 0
            failed_count = 0
            
            for terminal in self._terminals.values():
                terminals.append(terminal.to_dict())
                if terminal.state == TerminalState.RUNNING:
                    running_count += 1
                elif terminal.state == TerminalState.COMPLETED:
                    completed_count += 1
                elif terminal.state == TerminalState.FAILED:
                    failed_count += 1
            
            return {
                "total": len(terminals),
                "running": running_count,
                "completed": completed_count,
                "failed": failed_count,
                "idle": len(terminals) - running_count - completed_count - failed_count,
                "terminals": terminals,
            }
    
    def clear_terminal_output(self, terminal_id: str) -> bool:
        """Clear output buffer for a terminal.
        
        Args:
            terminal_id: Terminal ID
            
        Returns:
            True if cleared
        """
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                terminal.clear_buffer()
                return True
            return False
    
    def reset_terminal(self, terminal_id: str) -> bool:
        """Reset a terminal to idle state.
        
        Args:
            terminal_id: Terminal ID
            
        Returns:
            True if reset
        """
        with self._lock:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                terminal.state = TerminalState.IDLE
                terminal.current_command = None
                terminal.started_at = None
                terminal.return_code = None
                terminal.clear_buffer()
                
                self._emit_event(TerminalEvent(
                    terminal_id=terminal_id,
                    event_type=TerminalEventType.STATE_CHANGE,
                    message="Terminal reset",
                    data={"state": "idle"},
                ))
                
                return True
            return False
    
    def create_combined_view(
        self,
        terminal_ids: Optional[List[str]] = None,
        interleave: bool = True,
        max_lines: int = 500,
    ) -> List[Dict[str, Any]]:
        """Create a combined view of multiple terminals.
        
        Args:
            terminal_ids: Terminals to include (all if None)
            interleave: Interleave output chronologically
            max_lines: Maximum total lines
            
        Returns:
            List of output entries with terminal info
        """
        # This is a simplified implementation
        # A full implementation would track timestamps for interleaving
        result = []
        ids_to_check = terminal_ids or list(self._terminals.keys())
        
        for terminal_id in ids_to_check:
            terminal = self._terminals.get(terminal_id)
            if terminal:
                for line in terminal.output_buffer:
                    result.append({
                        "terminal_id": terminal_id,
                        "terminal_name": terminal.name,
                        "line": line,
                        "is_error": False,
                    })
                for line in terminal.error_buffer:
                    result.append({
                        "terminal_id": terminal_id,
                        "terminal_name": terminal.name,
                        "line": line,
                        "is_error": True,
                    })
        
        # Limit output
        if len(result) > max_lines:
            result = result[-max_lines:]
        
        return result
