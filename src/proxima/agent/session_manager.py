"""Agent Session Manager for Proxima.

Manages terminal sessions with pooling and lifecycle management.
Provides a centralized way to manage multiple terminal sessions
for concurrent execution and monitoring.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger
from .terminal_executor import TerminalExecutor, TerminalSession, TerminalOutput, TerminalType

logger = get_logger("agent.session_manager")


class SessionState(Enum):
    """State of an agent session."""
    IDLE = auto()
    ACTIVE = auto()
    EXECUTING = auto()
    PAUSED = auto()
    TERMINATED = auto()


@dataclass
class AgentSession:
    """An agent working session with context and history."""
    
    id: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    state: SessionState = SessionState.IDLE
    working_directory: str = ""
    environment_vars: Dict[str, str] = field(default_factory=dict)
    terminal_sessions: List[str] = field(default_factory=list)  # Terminal session IDs
    command_history: List[TerminalOutput] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)  # Arbitrary context storage
    max_history: int = 1000
    
    def add_to_history(self, output: TerminalOutput) -> None:
        """Add command output to history with size limit."""
        self.command_history.append(output)
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]
    
    def get_recent_history(self, count: int = 10) -> List[TerminalOutput]:
        """Get recent command history."""
        return self.command_history[-count:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "state": self.state.name,
            "working_directory": self.working_directory,
            "terminal_sessions": self.terminal_sessions,
            "history_count": len(self.command_history),
            "context_keys": list(self.context.keys()),
        }


class TerminalPool:
    """Pool of terminal sessions for efficient resource management.
    
    Manages a pool of pre-created terminal sessions for quick
    command execution without session creation overhead.
    """
    
    def __init__(
        self,
        executor: TerminalExecutor,
        pool_size: int = 5,
        terminal_type: Optional[TerminalType] = None,
    ):
        """Initialize terminal pool.
        
        Args:
            executor: Terminal executor instance
            pool_size: Maximum number of pooled sessions
            terminal_type: Terminal type for pooled sessions
        """
        self.executor = executor
        self.pool_size = pool_size
        self.terminal_type = terminal_type
        self._available: List[TerminalSession] = []
        self._in_use: Dict[str, TerminalSession] = {}
        self._lock = threading.Lock()
        self._initialized = False
    
    def initialize(self) -> None:
        """Pre-create pooled sessions."""
        with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.pool_size):
                session = self.executor.create_session(
                    terminal_type=self.terminal_type
                )
                self._available.append(session)
            
            self._initialized = True
            logger.info(f"Terminal pool initialized with {self.pool_size} sessions")
    
    def acquire(self, working_dir: Optional[str] = None) -> TerminalSession:
        """Acquire a terminal session from the pool.
        
        Args:
            working_dir: Working directory for the session
            
        Returns:
            Terminal session
        """
        with self._lock:
            if self._available:
                session = self._available.pop()
            else:
                # Create new session if pool exhausted
                session = self.executor.create_session(
                    terminal_type=self.terminal_type
                )
            
            if working_dir:
                session.working_dir = working_dir
            
            self._in_use[session.id] = session
            return session
    
    def release(self, session_id: str) -> bool:
        """Release a session back to the pool.
        
        Args:
            session_id: ID of session to release
            
        Returns:
            True if released, False if not found
        """
        with self._lock:
            if session_id in self._in_use:
                session = self._in_use.pop(session_id)
                
                # Only return to pool if under limit
                if len(self._available) < self.pool_size:
                    self._available.append(session)
                else:
                    self.executor.close_session(session_id)
                
                return True
            return False
    
    def shutdown(self) -> None:
        """Shutdown all pooled sessions."""
        with self._lock:
            for session in self._available:
                self.executor.close_session(session.id)
            for session in self._in_use.values():
                self.executor.close_session(session.id)
            
            self._available.clear()
            self._in_use.clear()
            self._initialized = False
            
            logger.info("Terminal pool shutdown complete")
    
    @property
    def available_count(self) -> int:
        """Number of available sessions."""
        return len(self._available)
    
    @property
    def in_use_count(self) -> int:
        """Number of in-use sessions."""
        return len(self._in_use)


class AgentSessionManager:
    """Manages agent sessions with terminal execution capabilities.
    
    Provides:
    - Session lifecycle management
    - Terminal pooling for efficient execution
    - Command history and context tracking
    - Concurrent execution support
    
    Example:
        >>> manager = AgentSessionManager()
        >>> session = manager.create_session("backend_build")
        >>> output = manager.execute(session.id, "pip install qiskit")
        >>> print(output.stdout)
    """
    
    def __init__(
        self,
        terminal_executor: Optional[TerminalExecutor] = None,
        pool_size: int = 5,
        max_sessions: int = 20,
    ):
        """Initialize the session manager.
        
        Args:
            terminal_executor: Terminal executor (creates new if not provided)
            pool_size: Size of terminal session pool
            max_sessions: Maximum concurrent agent sessions
        """
        self.executor = terminal_executor or TerminalExecutor()
        self.pool_size = pool_size
        self.max_sessions = max_sessions
        
        self._sessions: OrderedDict[str, AgentSession] = OrderedDict()
        self._terminal_pool: Optional[TerminalPool] = None
        self._lock = threading.Lock()
        self._callbacks: Dict[str, List[Callable[[TerminalOutput], None]]] = {}
        
        logger.info("AgentSessionManager initialized")
    
    def _get_pool(self) -> TerminalPool:
        """Get or create terminal pool."""
        if self._terminal_pool is None:
            self._terminal_pool = TerminalPool(
                self.executor,
                pool_size=self.pool_size,
            )
            self._terminal_pool.initialize()
        return self._terminal_pool
    
    def create_session(
        self,
        name: str = "default",
        working_dir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentSession:
        """Create a new agent session.
        
        Args:
            name: Session name
            working_dir: Initial working directory
            environment: Environment variables
            context: Initial context data
            
        Returns:
            New agent session
        """
        with self._lock:
            # Enforce max sessions limit
            if len(self._sessions) >= self.max_sessions:
                # Remove oldest idle session
                for session_id, session in list(self._sessions.items()):
                    if session.state == SessionState.IDLE:
                        self._cleanup_session(session_id)
                        break
                else:
                    # No idle sessions, remove oldest
                    oldest_id = next(iter(self._sessions))
                    self._cleanup_session(oldest_id)
            
            session_id = f"agent_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            session = AgentSession(
                id=session_id,
                name=name,
                working_directory=working_dir or self.executor.default_working_dir,
                environment_vars=environment or {},
                context=context or {},
            )
            
            self._sessions[session_id] = session
            logger.info(f"Created agent session: {session_id} ({name})")
            return session
    
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def list_sessions(self) -> List[AgentSession]:
        """List all active sessions."""
        return list(self._sessions.values())
    
    def _cleanup_session(self, session_id: str) -> None:
        """Clean up a session's resources."""
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            
            # Close any terminal sessions
            for term_id in session.terminal_sessions:
                self.executor.close_session(term_id)
            
            # Remove callbacks
            if session_id in self._callbacks:
                del self._callbacks[session_id]
            
            logger.info(f"Cleaned up session: {session_id}")
    
    def close_session(self, session_id: str) -> bool:
        """Close and cleanup an agent session.
        
        Args:
            session_id: ID of session to close
            
        Returns:
            True if closed, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                self._cleanup_session(session_id)
                return True
            return False
    
    def execute(
        self,
        session_id: str,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
        use_pool: bool = True,
    ) -> TerminalOutput:
        """Execute a command in an agent session.
        
        Args:
            session_id: Agent session ID
            command: Command to execute
            working_dir: Working directory override
            timeout: Command timeout
            use_pool: Whether to use terminal pool
            
        Returns:
            Terminal output
        """
        session = self._sessions.get(session_id)
        if not session:
            return TerminalOutput(
                stdout="",
                stderr=f"Session not found: {session_id}",
                return_code=-1,
                execution_time_ms=0,
                status=TerminalOutput.ExecutionStatus.FAILED,
                command=command,
                working_dir=working_dir or "",
            )
        
        # Update session state
        session.state = SessionState.EXECUTING
        
        # Determine working directory
        work_dir = working_dir or session.working_directory
        
        # Merge environment variables
        env = {**session.environment_vars}
        
        try:
            if use_pool:
                pool = self._get_pool()
                term_session = pool.acquire(work_dir)
                try:
                    output = self.executor.execute(
                        command,
                        working_dir=work_dir,
                        timeout=timeout,
                        environment=env,
                        session_id=term_session.id,
                    )
                finally:
                    pool.release(term_session.id)
            else:
                output = self.executor.execute(
                    command,
                    working_dir=work_dir,
                    timeout=timeout,
                    environment=env,
                )
            
            # Record in history
            session.add_to_history(output)
            
            # Trigger callbacks
            if session_id in self._callbacks:
                for callback in self._callbacks[session_id]:
                    try:
                        callback(output)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
            return output
            
        finally:
            session.state = SessionState.ACTIVE
    
    def execute_streaming(
        self,
        session_id: str,
        command: str,
        stdout_callback: Optional[Callable[[str], None]] = None,
        stderr_callback: Optional[Callable[[str], None]] = None,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> TerminalOutput:
        """Execute a command with streaming output.
        
        Args:
            session_id: Agent session ID
            command: Command to execute
            stdout_callback: Stdout line callback
            stderr_callback: Stderr line callback
            working_dir: Working directory override
            timeout: Command timeout
            
        Returns:
            Terminal output
        """
        session = self._sessions.get(session_id)
        if not session:
            return TerminalOutput(
                stdout="",
                stderr=f"Session not found: {session_id}",
                return_code=-1,
                execution_time_ms=0,
                status=TerminalOutput.ExecutionStatus.FAILED,
                command=command,
                working_dir=working_dir or "",
            )
        
        session.state = SessionState.EXECUTING
        work_dir = working_dir or session.working_directory
        env = {**session.environment_vars}
        
        try:
            output = self.executor.execute_streaming(
                command,
                stdout_callback=stdout_callback,
                stderr_callback=stderr_callback,
                working_dir=work_dir,
                timeout=timeout,
                environment=env,
            )
            
            session.add_to_history(output)
            return output
            
        finally:
            session.state = SessionState.ACTIVE
    
    async def execute_async(
        self,
        session_id: str,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> TerminalOutput:
        """Execute a command asynchronously.
        
        Args:
            session_id: Agent session ID
            command: Command to execute
            working_dir: Working directory override
            timeout: Command timeout
            
        Returns:
            Terminal output
        """
        session = self._sessions.get(session_id)
        if not session:
            return TerminalOutput(
                stdout="",
                stderr=f"Session not found: {session_id}",
                return_code=-1,
                execution_time_ms=0,
                status=TerminalOutput.ExecutionStatus.FAILED,
                command=command,
                working_dir=working_dir or "",
            )
        
        session.state = SessionState.EXECUTING
        work_dir = working_dir or session.working_directory
        env = {**session.environment_vars}
        
        try:
            output = await self.executor.execute_async(
                command,
                working_dir=work_dir,
                timeout=timeout,
                environment=env,
            )
            
            session.add_to_history(output)
            return output
            
        finally:
            session.state = SessionState.ACTIVE
    
    def execute_multi(
        self,
        session_id: str,
        commands: List[str],
        stop_on_error: bool = True,
        working_dir: Optional[str] = None,
    ) -> List[TerminalOutput]:
        """Execute multiple commands sequentially.
        
        Args:
            session_id: Agent session ID
            commands: List of commands to execute
            stop_on_error: Stop if a command fails
            working_dir: Working directory
            
        Returns:
            List of terminal outputs
        """
        results = []
        for cmd in commands:
            output = self.execute(session_id, cmd, working_dir=working_dir)
            results.append(output)
            
            if stop_on_error and not output.success:
                break
        
        return results
    
    def register_callback(
        self,
        session_id: str,
        callback: Callable[[TerminalOutput], None],
    ) -> None:
        """Register a callback for command completion.
        
        Args:
            session_id: Session ID
            callback: Callback function
        """
        if session_id not in self._callbacks:
            self._callbacks[session_id] = []
        self._callbacks[session_id].append(callback)
    
    def unregister_callback(
        self,
        session_id: str,
        callback: Callable[[TerminalOutput], None],
    ) -> bool:
        """Unregister a callback.
        
        Args:
            session_id: Session ID
            callback: Callback to remove
            
        Returns:
            True if removed
        """
        if session_id in self._callbacks:
            try:
                self._callbacks[session_id].remove(callback)
                return True
            except ValueError:
                pass
        return False
    
    def set_working_directory(self, session_id: str, path: str) -> bool:
        """Set working directory for a session.
        
        Args:
            session_id: Session ID
            path: New working directory
            
        Returns:
            True if set successfully
        """
        session = self._sessions.get(session_id)
        if session:
            if self.executor.navigate_to(path):
                session.working_directory = path
                return True
        return False
    
    def get_history(
        self,
        session_id: str,
        count: Optional[int] = None,
    ) -> List[TerminalOutput]:
        """Get command history for a session.
        
        Args:
            session_id: Session ID
            count: Number of entries to return (all if None)
            
        Returns:
            List of terminal outputs
        """
        session = self._sessions.get(session_id)
        if session:
            if count:
                return session.get_recent_history(count)
            return session.command_history.copy()
        return []
    
    def set_context(
        self,
        session_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """Set context data for a session.
        
        Args:
            session_id: Session ID
            key: Context key
            value: Context value
            
        Returns:
            True if set
        """
        session = self._sessions.get(session_id)
        if session:
            session.context[key] = value
            return True
        return False
    
    def get_context(
        self,
        session_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get context data from a session.
        
        Args:
            session_id: Session ID
            key: Context key
            default: Default value
            
        Returns:
            Context value or default
        """
        session = self._sessions.get(session_id)
        if session:
            return session.context.get(key, default)
        return default
    
    def shutdown(self) -> None:
        """Shutdown the session manager and all sessions."""
        with self._lock:
            # Close all sessions
            for session_id in list(self._sessions.keys()):
                self._cleanup_session(session_id)
            
            # Shutdown terminal pool
            if self._terminal_pool:
                self._terminal_pool.shutdown()
            
            logger.info("AgentSessionManager shutdown complete")
