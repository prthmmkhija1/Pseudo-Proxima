"""Multi-Terminal Monitor and Session Management for Proxima Agent.

Phase 3: Terminal Integration & Multi-Process Management

Provides comprehensive terminal management with:
- MultiTerminalMonitor: Track all active processes
- AgentSession: Store session state with command history
- SessionManager: Manage multiple concurrent sessions
- Cross-platform command normalization
- Terminal state machine with event emitter

Features:
- Support for 10 concurrent sessions
- Circular output buffer (10,000 lines per terminal)
- Event-based state notifications
- Session persistence to disk
- Priority queue for command execution
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from proxima.utils.logging import get_logger

logger = get_logger("agent.multi_terminal")


# =============================================================================
# Terminal State Machine
# =============================================================================

class TerminalState(Enum):
    """Terminal process state machine states."""
    PENDING = auto()      # Command queued but not started
    STARTING = auto()     # Process spawning in progress
    RUNNING = auto()      # Process actively executing
    COMPLETED = auto()    # Process finished successfully (return code 0)
    FAILED = auto()       # Process exited with non-zero return code
    TIMEOUT = auto()      # Process killed due to timeout
    CANCELLED = auto()    # User manually stopped process
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal (final) state."""
        return self in (
            TerminalState.COMPLETED,
            TerminalState.FAILED,
            TerminalState.TIMEOUT,
            TerminalState.CANCELLED,
        )
    
    def is_running(self) -> bool:
        """Check if process is actively running."""
        return self in (TerminalState.STARTING, TerminalState.RUNNING)


class TerminalEventType(Enum):
    """Types of terminal events."""
    STARTED = "started"
    OUTPUT_RECEIVED = "output_received"
    ERROR_RECEIVED = "error_received"
    STATE_CHANGED = "state_changed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class TerminalEvent:
    """Event emitted by terminal monitor."""
    event_type: TerminalEventType
    terminal_id: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "terminal_id": self.terminal_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclass
class OutputLine:
    """Single line of terminal output with metadata."""
    content: str
    line_number: int
    timestamp: float = field(default_factory=time.time)
    is_stderr: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "line_number": self.line_number,
            "timestamp": self.timestamp,
            "is_stderr": self.is_stderr,
        }


# =============================================================================
# Terminal Info and Circular Buffer
# =============================================================================

class CircularOutputBuffer:
    """Circular buffer for terminal output lines.
    
    Maintains a fixed-size buffer of output lines with efficient
    append and retrieval operations.
    """
    
    def __init__(self, max_lines: int = 10000):
        """Initialize the circular buffer.
        
        Args:
            max_lines: Maximum number of lines to store
        """
        self.max_lines = max_lines
        self._buffer: Deque[OutputLine] = deque(maxlen=max_lines)
        self._total_lines = 0
    
    def append(self, content: str, is_stderr: bool = False) -> OutputLine:
        """Append a line to the buffer.
        
        Args:
            content: Line content
            is_stderr: Whether line is from stderr
            
        Returns:
            The created OutputLine
        """
        self._total_lines += 1
        line = OutputLine(
            content=content,
            line_number=self._total_lines,
            is_stderr=is_stderr,
        )
        self._buffer.append(line)
        return line
    
    def get_lines(
        self,
        start: int = 0,
        count: Optional[int] = None,
    ) -> List[OutputLine]:
        """Get lines from buffer.
        
        Args:
            start: Starting index
            count: Number of lines (all if None)
            
        Returns:
            List of OutputLine objects
        """
        lines = list(self._buffer)
        if count is None:
            return lines[start:]
        return lines[start:start + count]
    
    def get_text(
        self,
        include_stderr: bool = True,
        separator: str = "\n",
    ) -> str:
        """Get all output as text.
        
        Args:
            include_stderr: Whether to include stderr lines
            separator: Line separator
            
        Returns:
            Combined output text
        """
        lines = self._buffer
        if not include_stderr:
            lines = [l for l in lines if not l.is_stderr]
        return separator.join(l.content for l in lines)
    
    def search(self, pattern: str, case_sensitive: bool = False) -> List[OutputLine]:
        """Search for lines matching pattern.
        
        Args:
            pattern: Search pattern
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matching lines
        """
        if not case_sensitive:
            pattern = pattern.lower()
        
        results = []
        for line in self._buffer:
            content = line.content if case_sensitive else line.content.lower()
            if pattern in content:
                results.append(line)
        return results
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_lines = 0
    
    @property
    def line_count(self) -> int:
        """Get current number of lines in buffer."""
        return len(self._buffer)
    
    @property
    def total_lines(self) -> int:
        """Get total number of lines ever added."""
        return self._total_lines
    
    def __len__(self) -> int:
        return len(self._buffer)


@dataclass
class TerminalInfo:
    """Information about a terminal process."""
    terminal_id: str
    command: str
    working_dir: str
    state: TerminalState = TerminalState.PENDING
    pid: Optional[int] = None
    return_code: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output_buffer: CircularOutputBuffer = field(default_factory=CircularOutputBuffer)
    environment: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get execution duration in milliseconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    @property
    def is_running(self) -> bool:
        """Check if terminal is currently running."""
        return self.state.is_running()
    
    @property
    def is_complete(self) -> bool:
        """Check if terminal has finished."""
        return self.state.is_terminal()
    
    @property
    def success(self) -> bool:
        """Check if terminal completed successfully."""
        return self.state == TerminalState.COMPLETED and self.return_code == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without buffer content)."""
        return {
            "terminal_id": self.terminal_id,
            "command": self.command,
            "working_dir": self.working_dir,
            "state": self.state.name,
            "pid": self.pid,
            "return_code": self.return_code,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "line_count": self.output_buffer.line_count,
            "total_lines": self.output_buffer.total_lines,
        }


# =============================================================================
# Multi-Terminal Monitor
# =============================================================================

class MultiTerminalMonitor:
    """Monitor and manage multiple terminal processes.
    
    Provides a central registry for all active terminal processes
    with event-based notifications for state changes and output.
    
    Example:
        >>> monitor = MultiTerminalMonitor()
        >>> monitor.add_listener(lambda e: print(f"Event: {e.event_type}"))
        >>> 
        >>> # Register a new terminal
        >>> terminal = monitor.register("pip install qiskit", "/project")
        >>> 
        >>> # Update terminal state
        >>> monitor.update_state(terminal.terminal_id, TerminalState.RUNNING)
        >>> 
        >>> # Add output
        >>> monitor.append_output(terminal.terminal_id, "Installing...", False)
    """
    
    def __init__(self, max_terminals: int = 10):
        """Initialize the monitor.
        
        Args:
            max_terminals: Maximum concurrent terminals to track
        """
        self.max_terminals = max_terminals
        self._terminals: Dict[str, TerminalInfo] = {}
        self._listeners: List[Callable[[TerminalEvent], None]] = []
        self._history: Deque[TerminalInfo] = deque(maxlen=100)
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_processes": 0,
            "successful": 0,
            "failed": 0,
            "cancelled": 0,
            "timeouts": 0,
            "active": 0,
        }
    
    def add_listener(self, callback: Callable[[TerminalEvent], None]) -> None:
        """Add an event listener.
        
        Args:
            callback: Function to call for each event
        """
        if callback not in self._listeners:
            self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[TerminalEvent], None]) -> None:
        """Remove an event listener.
        
        Args:
            callback: Function to remove
        """
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _emit_event(self, event: TerminalEvent) -> None:
        """Emit an event to all listeners.
        
        Args:
            event: Event to emit
        """
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")
    
    def register(
        self,
        command: str,
        working_dir: str,
        terminal_id: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TerminalInfo:
        """Register a new terminal process.
        
        Args:
            command: Command being executed
            working_dir: Working directory
            terminal_id: Custom ID (auto-generated if not provided)
            environment: Environment variables
            metadata: Additional metadata
            
        Returns:
            TerminalInfo for the new terminal
        """
        # Generate ID
        terminal_id = terminal_id or f"term_{uuid.uuid4().hex[:8]}"
        
        # Create terminal info
        terminal = TerminalInfo(
            terminal_id=terminal_id,
            command=command,
            working_dir=working_dir,
            environment=environment or {},
            metadata=metadata or {},
        )
        
        # Store
        self._terminals[terminal_id] = terminal
        self._stats["total_processes"] += 1
        self._stats["active"] += 1
        
        logger.debug(f"Registered terminal: {terminal_id} for '{command[:50]}'")
        
        return terminal
    
    def get_terminal(self, terminal_id: str) -> Optional[TerminalInfo]:
        """Get terminal info by ID.
        
        Args:
            terminal_id: Terminal ID
            
        Returns:
            TerminalInfo or None if not found
        """
        return self._terminals.get(terminal_id)
    
    def get_all_terminals(self) -> Dict[str, TerminalInfo]:
        """Get all registered terminals.
        
        Returns:
            Dictionary of terminal_id -> TerminalInfo
        """
        return dict(self._terminals)
    
    def get_active_terminals(self) -> Dict[str, TerminalInfo]:
        """Get all currently running terminals.
        
        Returns:
            Dictionary of running terminals
        """
        return {
            tid: term for tid, term in self._terminals.items()
            if term.is_running
        }
    
    def update_state(
        self,
        terminal_id: str,
        new_state: TerminalState,
        return_code: Optional[int] = None,
        pid: Optional[int] = None,
    ) -> bool:
        """Update terminal state.
        
        Args:
            terminal_id: Terminal ID
            new_state: New state
            return_code: Process return code (for completed states)
            pid: Process ID (for starting state)
            
        Returns:
            True if state was updated
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return False
        
        old_state = terminal.state
        terminal.state = new_state
        
        # Update timestamps
        if new_state == TerminalState.STARTING:
            terminal.start_time = time.time()
            if pid:
                terminal.pid = pid
        elif new_state.is_terminal():
            terminal.end_time = time.time()
            if return_code is not None:
                terminal.return_code = return_code
            self._stats["active"] = max(0, self._stats["active"] - 1)
            
            # Update success/failure stats
            if new_state == TerminalState.COMPLETED:
                self._stats["successful"] += 1
            elif new_state == TerminalState.FAILED:
                self._stats["failed"] += 1
            elif new_state == TerminalState.CANCELLED:
                self._stats["cancelled"] += 1
            elif new_state == TerminalState.TIMEOUT:
                self._stats["timeouts"] += 1
            
            # Move to history
            self._history.append(terminal)
        
        # Emit state change event
        event = TerminalEvent(
            event_type=TerminalEventType.STATE_CHANGED,
            terminal_id=terminal_id,
            data={
                "old_state": old_state.name,
                "new_state": new_state.name,
                "return_code": return_code,
            },
        )
        self._emit_event(event)
        
        # Emit specific events
        if new_state == TerminalState.RUNNING:
            self._emit_event(TerminalEvent(
                event_type=TerminalEventType.STARTED,
                terminal_id=terminal_id,
                data={"command": terminal.command, "pid": pid},
            ))
        elif new_state == TerminalState.COMPLETED:
            self._emit_event(TerminalEvent(
                event_type=TerminalEventType.COMPLETED,
                terminal_id=terminal_id,
                data={
                    "return_code": return_code,
                    "duration_ms": terminal.duration_ms,
                },
            ))
        elif new_state == TerminalState.TIMEOUT:
            self._emit_event(TerminalEvent(
                event_type=TerminalEventType.TIMEOUT,
                terminal_id=terminal_id,
                data={"duration_ms": terminal.duration_ms},
            ))
        elif new_state == TerminalState.CANCELLED:
            self._emit_event(TerminalEvent(
                event_type=TerminalEventType.CANCELLED,
                terminal_id=terminal_id,
            ))
        
        return True
    
    def append_output(
        self,
        terminal_id: str,
        content: str,
        is_stderr: bool = False,
    ) -> Optional[OutputLine]:
        """Append output to terminal buffer.
        
        Args:
            terminal_id: Terminal ID
            content: Output content
            is_stderr: Whether content is from stderr
            
        Returns:
            OutputLine or None if terminal not found
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return None
        
        line = terminal.output_buffer.append(content, is_stderr)
        
        # Emit event (debounced in practice)
        event_type = (
            TerminalEventType.ERROR_RECEIVED if is_stderr
            else TerminalEventType.OUTPUT_RECEIVED
        )
        self._emit_event(TerminalEvent(
            event_type=event_type,
            terminal_id=terminal_id,
            data={
                "content": content,
                "line_number": line.line_number,
                "is_stderr": is_stderr,
            },
        ))
        
        return line
    
    def remove_terminal(self, terminal_id: str) -> bool:
        """Remove a terminal from tracking.
        
        Args:
            terminal_id: Terminal ID
            
        Returns:
            True if terminal was removed
        """
        if terminal_id in self._terminals:
            terminal = self._terminals.pop(terminal_id)
            if terminal.is_running:
                self._stats["active"] = max(0, self._stats["active"] - 1)
            return True
        return False
    
    def filter_terminals(
        self,
        states: Optional[Set[TerminalState]] = None,
        command_pattern: Optional[str] = None,
    ) -> List[TerminalInfo]:
        """Filter terminals by criteria.
        
        Args:
            states: Set of states to include
            command_pattern: Pattern to match in command
            
        Returns:
            List of matching terminals
        """
        results = []
        for terminal in self._terminals.values():
            # Check state filter
            if states and terminal.state not in states:
                continue
            
            # Check command pattern
            if command_pattern:
                if command_pattern.lower() not in terminal.command.lower():
                    continue
            
            results.append(terminal)
        
        return results
    
    def search_output(
        self,
        terminal_id: str,
        pattern: str,
        case_sensitive: bool = False,
    ) -> List[OutputLine]:
        """Search output in a specific terminal.
        
        Args:
            terminal_id: Terminal ID
            pattern: Search pattern
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            List of matching lines
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return []
        return terminal.output_buffer.search(pattern, case_sensitive)
    
    def search_all_output(
        self,
        pattern: str,
        case_sensitive: bool = False,
    ) -> Dict[str, List[OutputLine]]:
        """Search output across all terminals.
        
        Args:
            pattern: Search pattern
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            Dictionary of terminal_id -> matching lines
        """
        results = {}
        for terminal_id, terminal in self._terminals.items():
            matches = terminal.output_buffer.search(pattern, case_sensitive)
            if matches:
                results[terminal_id] = matches
        return results
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get monitor statistics."""
        return self._stats.copy()
    
    @property
    def history(self) -> List[TerminalInfo]:
        """Get completed terminal history."""
        return list(self._history)


# =============================================================================
# Agent Session Manager
# =============================================================================

@dataclass
class CommandHistoryEntry:
    """Entry in command history."""
    command: str
    working_dir: str
    timestamp: float = field(default_factory=time.time)
    return_code: Optional[int] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "working_dir": self.working_dir,
            "timestamp": self.timestamp,
            "return_code": self.return_code,
            "duration_ms": self.duration_ms,
        }


class SessionState(Enum):
    """Agent session lifecycle states."""
    CREATED = "created"
    ACTIVE = "active"
    IDLE = "idle"
    DESTROYED = "destroyed"


@dataclass
class AgentSession:
    """Agent terminal session with state persistence.
    
    Stores:
    - Session ID and name
    - Working directory stack
    - Environment variables
    - Command history (last 100 commands)
    - Session statistics
    """
    id: str
    name: str = "Session"
    working_dir: str = field(default_factory=os.getcwd)
    working_dir_stack: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    command_history: List[CommandHistoryEntry] = field(default_factory=list)
    state: SessionState = SessionState.CREATED
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    max_history: int = 100
    
    def pushd(self, path: str) -> bool:
        """Push current directory and change to new path.
        
        Args:
            path: Path to change to
            
        Returns:
            True if successful
        """
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            self.working_dir_stack.append(self.working_dir)
            self.working_dir = abs_path
            self.last_activity = time.time()
            return True
        return False
    
    def popd(self) -> Optional[str]:
        """Pop directory from stack.
        
        Returns:
            Previous working directory or None if stack empty
        """
        if self.working_dir_stack:
            self.working_dir = self.working_dir_stack.pop()
            self.last_activity = time.time()
            return self.working_dir
        return None
    
    def cd(self, path: str) -> bool:
        """Change working directory.
        
        Args:
            path: Path to change to
            
        Returns:
            True if successful
        """
        if path == "-" and self.working_dir_stack:
            return self.popd() is not None
        
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            self.working_dir = abs_path
            self.last_activity = time.time()
            return True
        return False
    
    def add_to_history(
        self,
        command: str,
        return_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Add command to history.
        
        Args:
            command: Executed command
            return_code: Command return code
            duration_ms: Execution duration
        """
        entry = CommandHistoryEntry(
            command=command,
            working_dir=self.working_dir,
            return_code=return_code,
            duration_ms=duration_ms,
        )
        self.command_history.append(entry)
        
        # Trim to max history
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]
        
        self.last_activity = time.time()
    
    def set_env(self, key: str, value: str) -> None:
        """Set environment variable.
        
        Args:
            key: Variable name
            value: Variable value
        """
        self.environment[key] = value
        self.last_activity = time.time()
    
    def unset_env(self, key: str) -> bool:
        """Unset environment variable.
        
        Args:
            key: Variable name
            
        Returns:
            True if variable was set
        """
        if key in self.environment:
            del self.environment[key]
            self.last_activity = time.time()
            return True
        return False
    
    def get_full_env(self) -> Dict[str, str]:
        """Get full environment (parent + session).
        
        Returns:
            Combined environment dictionary
        """
        env = os.environ.copy()
        env.update(self.environment)
        return env
    
    @property
    def idle_time(self) -> float:
        """Get time since last activity in seconds."""
        return time.time() - self.last_activity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "id": self.id,
            "name": self.name,
            "working_dir": self.working_dir,
            "working_dir_stack": self.working_dir_stack,
            "environment": self.environment,
            "command_history": [e.to_dict() for e in self.command_history],
            "state": self.state.value,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSession":
        """Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            AgentSession instance
        """
        session = cls(
            id=data["id"],
            name=data.get("name", "Session"),
            working_dir=data.get("working_dir", os.getcwd()),
        )
        session.working_dir_stack = data.get("working_dir_stack", [])
        session.environment = data.get("environment", {})
        session.state = SessionState(data.get("state", "created"))
        session.created_at = data.get("created_at", time.time())
        session.last_activity = data.get("last_activity", time.time())
        
        # Restore command history
        for entry_data in data.get("command_history", []):
            session.command_history.append(CommandHistoryEntry(
                command=entry_data["command"],
                working_dir=entry_data["working_dir"],
                timestamp=entry_data.get("timestamp", 0),
                return_code=entry_data.get("return_code"),
                duration_ms=entry_data.get("duration_ms"),
            ))
        
        return session


class SessionManager:
    """Manage multiple agent sessions with persistence.
    
    Features:
    - Create/destroy sessions
    - Session pool with max limit (10 concurrent)
    - Idle session reclamation (5 minutes)
    - Persistence to ~/.proxima/agent_sessions/
    - Session import/export
    
    Example:
        >>> manager = SessionManager()
        >>> session = manager.create_session("Build Session")
        >>> session.cd("/project")
        >>> session.add_to_history("pip install qiskit", return_code=0)
        >>> manager.save_session(session.id)
    """
    
    def __init__(
        self,
        max_sessions: int = 10,
        idle_timeout: float = 300.0,  # 5 minutes
        storage_dir: Optional[Path] = None,
    ):
        """Initialize the session manager.
        
        Args:
            max_sessions: Maximum concurrent sessions
            idle_timeout: Idle timeout in seconds
            storage_dir: Directory for session storage
        """
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self.storage_dir = storage_dir or (Path.home() / ".proxima" / "agent_sessions")
        
        self._sessions: Dict[str, AgentSession] = {}
        self._session_counter = 0
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(
        self,
        name: Optional[str] = None,
        working_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AgentSession:
        """Create a new session.
        
        Args:
            name: Session name
            working_dir: Initial working directory
            session_id: Custom session ID
            
        Returns:
            New AgentSession
            
        Raises:
            RuntimeError: If max sessions reached
        """
        # Check if we need to reclaim idle sessions
        if len(self._sessions) >= self.max_sessions:
            self._reclaim_idle_sessions()
        
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError(f"Maximum sessions ({self.max_sessions}) reached")
        
        # Generate ID
        self._session_counter += 1
        session_id = session_id or f"session_{self._session_counter}_{int(time.time())}"
        name = name or f"Session {self._session_counter}"
        
        session = AgentSession(
            id=session_id,
            name=name,
            working_dir=working_dir or os.getcwd(),
        )
        session.state = SessionState.ACTIVE
        
        self._sessions[session_id] = session
        logger.info(f"Created session: {session_id} ({name})")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get a session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            AgentSession or None
        """
        session = self._sessions.get(session_id)
        if session:
            session.last_activity = time.time()
            if session.state == SessionState.IDLE:
                session.state = SessionState.ACTIVE
        return session
    
    def get_all_sessions(self) -> Dict[str, AgentSession]:
        """Get all active sessions.
        
        Returns:
            Dictionary of session_id -> AgentSession
        """
        return dict(self._sessions)
    
    def destroy_session(self, session_id: str, save: bool = True) -> bool:
        """Destroy a session.
        
        Args:
            session_id: Session ID
            save: Whether to save before destroying
            
        Returns:
            True if session was destroyed
        """
        if session_id not in self._sessions:
            return False
        
        session = self._sessions[session_id]
        
        # Save before destroying
        if save:
            self.save_session(session_id)
        
        session.state = SessionState.DESTROYED
        del self._sessions[session_id]
        
        logger.info(f"Destroyed session: {session_id}")
        return True
    
    def _reclaim_idle_sessions(self) -> int:
        """Reclaim idle sessions.
        
        Returns:
            Number of sessions reclaimed
        """
        reclaimed = 0
        now = time.time()
        
        for session_id, session in list(self._sessions.items()):
            if session.idle_time > self.idle_timeout:
                # Mark as idle first
                session.state = SessionState.IDLE
                # Save and destroy
                self.destroy_session(session_id, save=True)
                reclaimed += 1
        
        if reclaimed:
            logger.info(f"Reclaimed {reclaimed} idle sessions")
        
        return reclaimed
    
    def save_session(self, session_id: str) -> bool:
        """Save session to disk.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if saved successfully
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        filepath = self.storage_dir / f"{session_id}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            logger.debug(f"Saved session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[AgentSession]:
        """Load session from disk.
        
        Args:
            session_id: Session ID
            
        Returns:
            AgentSession or None
        """
        filepath = self.storage_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            session = AgentSession.from_dict(data)
            session.state = SessionState.ACTIVE
            self._sessions[session_id] = session
            
            logger.info(f"Loaded session: {session_id}")
            return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """List all saved sessions.
        
        Returns:
            List of session info dictionaries
        """
        sessions = []
        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                sessions.append({
                    "id": data.get("id"),
                    "name": data.get("name"),
                    "created_at": data.get("created_at"),
                    "last_activity": data.get("last_activity"),
                    "filepath": str(filepath),
                })
            except Exception:
                continue
        return sessions
    
    def save_all(self) -> int:
        """Save all active sessions.
        
        Returns:
            Number of sessions saved
        """
        saved = 0
        for session_id in self._sessions:
            if self.save_session(session_id):
                saved += 1
        return saved


# =============================================================================
# Cross-Platform Command Normalizer
# =============================================================================

class CommandNormalizer:
    """Cross-platform command translation and normalization.
    
    Handles:
    - Command translation (ls -> dir, etc.)
    - Path separator conversion
    - Environment variable syntax conversion
    - Home directory expansion
    
    Example:
        >>> normalizer = CommandNormalizer()
        >>> cmd = normalizer.normalize("ls -la ~/projects", for_platform="windows")
        >>> print(cmd)  # Get-ChildItem -Force C:\\Users\\name\\projects
    """
    
    # Command mappings: generic -> platform-specific
    COMMAND_MAPPINGS = {
        # List directory
        "ls": {
            "windows_powershell": "Get-ChildItem",
            "windows_cmd": "dir",
            "unix": "ls",
        },
        # Copy file
        "cp": {
            "windows_powershell": "Copy-Item",
            "windows_cmd": "copy",
            "unix": "cp",
        },
        # Move/rename
        "mv": {
            "windows_powershell": "Move-Item",
            "windows_cmd": "move",
            "unix": "mv",
        },
        # Remove file
        "rm": {
            "windows_powershell": "Remove-Item",
            "windows_cmd": "del",
            "unix": "rm",
        },
        # Remove directory
        "rmdir": {
            "windows_powershell": "Remove-Item -Recurse",
            "windows_cmd": "rmdir /s /q",
            "unix": "rm -rf",
        },
        # Create directory
        "mkdir": {
            "windows_powershell": "New-Item -ItemType Directory -Force",
            "windows_cmd": "mkdir",
            "unix": "mkdir -p",
        },
        # Display file
        "cat": {
            "windows_powershell": "Get-Content",
            "windows_cmd": "type",
            "unix": "cat",
        },
        # Find text
        "grep": {
            "windows_powershell": "Select-String",
            "windows_cmd": "findstr",
            "unix": "grep",
        },
        # Current directory
        "pwd": {
            "windows_powershell": "Get-Location",
            "windows_cmd": "cd",
            "unix": "pwd",
        },
        # Clear screen
        "clear": {
            "windows_powershell": "Clear-Host",
            "windows_cmd": "cls",
            "unix": "clear",
        },
        # Environment variables
        "env": {
            "windows_powershell": "Get-ChildItem Env:",
            "windows_cmd": "set",
            "unix": "env",
        },
        # Echo
        "echo": {
            "windows_powershell": "Write-Output",
            "windows_cmd": "echo",
            "unix": "echo",
        },
    }
    
    # Flag mappings for common commands
    FLAG_MAPPINGS = {
        "ls": {
            "-la": {
                "windows_powershell": "-Force",
                "windows_cmd": "/a",
                "unix": "-la",
            },
            "-l": {
                "windows_powershell": "",
                "windows_cmd": "",
                "unix": "-l",
            },
            "-a": {
                "windows_powershell": "-Force -Hidden",
                "windows_cmd": "/a",
                "unix": "-a",
            },
        },
        "rm": {
            "-rf": {
                "windows_powershell": "-Recurse -Force",
                "windows_cmd": "/s /q",
                "unix": "-rf",
            },
            "-r": {
                "windows_powershell": "-Recurse",
                "windows_cmd": "/s",
                "unix": "-r",
            },
            "-f": {
                "windows_powershell": "-Force",
                "windows_cmd": "/f",
                "unix": "-f",
            },
        },
    }
    
    def __init__(self):
        """Initialize the normalizer."""
        self._platform = self._detect_platform()
    
    @staticmethod
    def _detect_platform() -> str:
        """Detect current platform type.
        
        Returns:
            Platform identifier string
        """
        system = platform.system().lower()
        if system == "windows":
            # Check for PowerShell
            if shutil.which("pwsh") or shutil.which("powershell"):
                return "windows_powershell"
            return "windows_cmd"
        return "unix"
    
    def normalize_path(self, path: str, target_platform: Optional[str] = None) -> str:
        """Normalize path for target platform.
        
        Args:
            path: Path to normalize
            target_platform: Target platform (uses current if None)
            
        Returns:
            Normalized path
        """
        target = target_platform or self._platform
        
        # Expand home directory
        if path.startswith("~"):
            if "windows" in target:
                home = os.environ.get("USERPROFILE", "C:\\Users\\Default")
                path = path.replace("~", home, 1)
            else:
                home = os.environ.get("HOME", "/home")
                path = path.replace("~", home, 1)
        
        # Convert separators
        if "windows" in target:
            path = path.replace("/", "\\")
        else:
            path = path.replace("\\", "/")
        
        return path
    
    def normalize_env_var(
        self,
        var_reference: str,
        target_platform: Optional[str] = None,
    ) -> str:
        """Convert environment variable reference.
        
        Args:
            var_reference: Variable reference (e.g., $HOME, %HOME%, $env:HOME)
            target_platform: Target platform
            
        Returns:
            Converted reference
        """
        target = target_platform or self._platform
        
        # Extract variable name
        var_name = var_reference
        for prefix in ["$env:", "$", "%"]:
            if var_name.startswith(prefix):
                var_name = var_name[len(prefix):]
                break
        var_name = var_name.rstrip("%")
        
        # Convert to target format
        if target == "windows_powershell":
            return f"$env:{var_name}"
        elif target == "windows_cmd":
            return f"%{var_name}%"
        else:
            return f"${var_name}"
    
    def normalize_command(
        self,
        command: str,
        target_platform: Optional[str] = None,
    ) -> str:
        """Normalize command for target platform.
        
        Args:
            command: Command to normalize
            target_platform: Target platform
            
        Returns:
            Normalized command
        """
        target = target_platform or self._platform
        
        # Split command into parts
        parts = command.split()
        if not parts:
            return command
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        # Check for command mapping
        if cmd in self.COMMAND_MAPPINGS:
            mapping = self.COMMAND_MAPPINGS[cmd]
            if target in mapping:
                cmd = mapping[target]
        
        # Process flags
        if cmd in self.FLAG_MAPPINGS:
            flag_map = self.FLAG_MAPPINGS[cmd]
            new_args = []
            for arg in args:
                if arg in flag_map and target in flag_map[arg]:
                    mapped_flag = flag_map[arg][target]
                    if mapped_flag:
                        new_args.append(mapped_flag)
                else:
                    # Normalize paths in arguments
                    new_args.append(self.normalize_path(arg, target))
            args = new_args
        else:
            # Just normalize paths
            args = [self.normalize_path(arg, target) for arg in args]
        
        # Reconstruct command
        if args:
            return f"{cmd} {' '.join(args)}"
        return cmd
    
    def detect_env_vars(self, command: str) -> List[str]:
        """Detect environment variable references in command.
        
        Args:
            command: Command string
            
        Returns:
            List of variable references found
        """
        import re
        
        # Match various formats
        patterns = [
            r'\$env:(\w+)',      # PowerShell
            r'\$\{(\w+)\}',      # Unix ${VAR}
            r'\$(\w+)',          # Unix $VAR
            r'%(\w+)%',          # CMD
        ]
        
        found = []
        for pattern in patterns:
            matches = re.findall(pattern, command)
            found.extend(matches)
        
        return list(set(found))
    
    def convert_env_vars_in_command(
        self,
        command: str,
        target_platform: Optional[str] = None,
    ) -> str:
        """Convert all env var references in a command.
        
        Args:
            command: Command with env var references
            target_platform: Target platform
            
        Returns:
            Command with converted references
        """
        import re
        
        target = target_platform or self._platform
        
        # Convert PowerShell format
        command = re.sub(
            r'\$env:(\w+)',
            lambda m: self.normalize_env_var(m.group(0), target),
            command
        )
        
        # Convert Unix ${VAR} format
        command = re.sub(
            r'\$\{(\w+)\}',
            lambda m: self.normalize_env_var(m.group(0), target),
            command
        )
        
        # Convert Unix $VAR format (careful not to match $env:)
        command = re.sub(
            r'(?<!\w)\$(?!env:)(\w+)',
            lambda m: self.normalize_env_var(m.group(0), target),
            command
        )
        
        # Convert CMD %VAR% format
        command = re.sub(
            r'%(\w+)%',
            lambda m: self.normalize_env_var(m.group(0), target),
            command
        )
        
        return command


# =============================================================================
# Command Queue with Priority
# =============================================================================

class CommandPriority(Enum):
    """Command execution priority levels."""
    CRITICAL = 0   # Highest priority
    USER = 1       # User-initiated commands
    BACKGROUND = 2 # Background tasks
    CLEANUP = 3    # Cleanup/maintenance tasks
    LOW = 4        # Lowest priority


@dataclass(order=True)
class QueuedCommand:
    """Command queued for execution."""
    priority: int
    timestamp: float = field(compare=False)
    command: str = field(compare=False)
    working_dir: str = field(compare=False)
    session_id: Optional[str] = field(default=None, compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


class CommandQueue:
    """Priority queue for command execution.
    
    Features:
    - Priority-based ordering
    - Concurrency limiting
    - Queue monitoring
    """
    
    def __init__(self, max_concurrent: int = 10):
        """Initialize the queue.
        
        Args:
            max_concurrent: Maximum concurrent executions
        """
        self.max_concurrent = max_concurrent
        self._queue: List[QueuedCommand] = []
        self._active_count = 0
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
    
    async def enqueue(
        self,
        command: str,
        working_dir: str,
        priority: CommandPriority = CommandPriority.USER,
        session_id: Optional[str] = None,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueuedCommand:
        """Add command to queue.
        
        Args:
            command: Command to execute
            working_dir: Working directory
            priority: Execution priority
            session_id: Associated session ID
            callback: Completion callback
            metadata: Additional metadata
            
        Returns:
            QueuedCommand instance
        """
        queued = QueuedCommand(
            priority=priority.value,
            timestamp=time.time(),
            command=command,
            working_dir=working_dir,
            session_id=session_id,
            callback=callback,
            metadata=metadata or {},
        )
        
        async with self._lock:
            # Insert in priority order
            import bisect
            bisect.insort(self._queue, queued)
        
        return queued
    
    async def dequeue(self) -> Optional[QueuedCommand]:
        """Get next command from queue.
        
        Returns:
            Next QueuedCommand or None if empty
        """
        async with self._lock:
            if self._queue:
                return self._queue.pop(0)
        return None
    
    async def acquire_slot(self) -> bool:
        """Acquire execution slot.
        
        Returns:
            True if slot acquired
        """
        await self._semaphore.acquire()
        self._active_count += 1
        return True
    
    def release_slot(self) -> None:
        """Release execution slot."""
        self._semaphore.release()
        self._active_count = max(0, self._active_count - 1)
    
    @property
    def queue_length(self) -> int:
        """Get number of queued commands."""
        return len(self._queue)
    
    @property
    def active_count(self) -> int:
        """Get number of active executions."""
        return self._active_count
    
    @property
    def available_slots(self) -> int:
        """Get number of available slots."""
        return self.max_concurrent - self._active_count


# =============================================================================
# Convenience functions
# =============================================================================

_monitor: Optional[MultiTerminalMonitor] = None
_session_manager: Optional[SessionManager] = None
_normalizer: Optional[CommandNormalizer] = None


def get_multi_terminal_monitor() -> MultiTerminalMonitor:
    """Get the global multi-terminal monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = MultiTerminalMonitor()
    return _monitor


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_command_normalizer() -> CommandNormalizer:
    """Get the global command normalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = CommandNormalizer()
    return _normalizer
