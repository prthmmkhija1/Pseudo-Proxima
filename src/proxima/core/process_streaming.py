"""Process Output Streaming Module for Real-Time Execution.

Provides asyncio-based subprocess management with real-time output
streaming, integrated with the event bus system.

Features:
- Non-blocking subprocess execution
- Separate stdout/stderr streams
- Line-based buffering with ANSI code preservation
- Event bus integration for real-time updates
- Process lifecycle management
- Timeout and cancellation support
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from proxima.core.event_bus import (
    Event,
    EventBus,
    EventType,
    get_event_bus,
    emit_process_started,
    emit_output_line,
    emit_process_completed,
    emit_progress_update,
)


class ProcessState(Enum):
    """State of a managed process."""
    PENDING = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class ProcessInfo:
    """Information about a managed process."""
    process_id: str
    command: Union[str, List[str]]
    working_dir: str
    state: ProcessState = ProcessState.PENDING
    pid: Optional[int] = None
    return_code: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    stdout_lines: List[str] = field(default_factory=list)
    stderr_lines: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get process duration in milliseconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    @property
    def is_running(self) -> bool:
        """Check if process is currently running."""
        return self.state in (ProcessState.STARTING, ProcessState.RUNNING)
    
    @property
    def is_complete(self) -> bool:
        """Check if process has completed (success or failure)."""
        return self.state in (
            ProcessState.COMPLETED,
            ProcessState.FAILED,
            ProcessState.CANCELLED,
            ProcessState.TIMEOUT,
        )
    
    @property
    def output(self) -> str:
        """Get combined stdout output."""
        return "\n".join(self.stdout_lines)
    
    @property
    def errors(self) -> str:
        """Get combined stderr output."""
        return "\n".join(self.stderr_lines)


class ProcessOutputStream:
    """Real-time output stream handler.
    
    Buffers partial lines and emits complete lines to the event bus.
    Handles ANSI escape codes properly.
    """
    
    def __init__(
        self,
        process_id: str,
        is_stderr: bool = False,
        event_bus: Optional[EventBus] = None,
        correlation_id: Optional[str] = None,
        max_line_length: int = 10000,
    ):
        """Initialize the output stream.
        
        Args:
            process_id: ID of the source process
            is_stderr: Whether this is stderr stream
            event_bus: Event bus for emitting events
            correlation_id: Correlation ID for events
            max_line_length: Maximum line length before forced split
        """
        self.process_id = process_id
        self.is_stderr = is_stderr
        self.event_bus = event_bus or get_event_bus()
        self.correlation_id = correlation_id
        self.max_line_length = max_line_length
        
        self._buffer = ""
        self._lines: List[str] = []
        self._total_bytes = 0
        self._total_lines = 0
    
    def feed(self, data: bytes) -> List[str]:
        """Feed data into the stream buffer.
        
        Args:
            data: Raw bytes from process output
            
        Returns:
            List of complete lines extracted
        """
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("latin-1", errors="replace")
        
        self._total_bytes += len(data)
        
        # Add to buffer
        self._buffer += text
        
        # Extract complete lines
        complete_lines = []
        
        while "\n" in self._buffer or len(self._buffer) > self.max_line_length:
            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
            else:
                # Force split on max length
                line = self._buffer[:self.max_line_length]
                self._buffer = self._buffer[self.max_line_length:]
            
            # Handle carriage return (for progress updates)
            if "\r" in line:
                # Keep only the last part after carriage return
                parts = line.split("\r")
                line = parts[-1] if parts[-1] else parts[-2] if len(parts) > 1 else ""
            
            complete_lines.append(line)
            self._lines.append(line)
            self._total_lines += 1
            
            # Emit event
            event = Event(
                event_type=EventType.ERROR_LINE if self.is_stderr else EventType.OUTPUT_LINE,
                source_id=self.process_id,
                payload={
                    "line": line,
                    "line_number": self._total_lines,
                    "is_stderr": self.is_stderr,
                },
                correlation_id=self.correlation_id,
            )
            self.event_bus.emit(event)
        
        return complete_lines
    
    def flush(self) -> Optional[str]:
        """Flush any remaining buffered content.
        
        Returns:
            Final line if any content was buffered
        """
        if self._buffer:
            line = self._buffer
            self._buffer = ""
            self._lines.append(line)
            self._total_lines += 1
            
            # Emit event
            event = Event(
                event_type=EventType.ERROR_LINE if self.is_stderr else EventType.OUTPUT_LINE,
                source_id=self.process_id,
                payload={
                    "line": line,
                    "line_number": self._total_lines,
                    "is_stderr": self.is_stderr,
                    "is_final": True,
                },
                correlation_id=self.correlation_id,
            )
            self.event_bus.emit(event)
            
            return line
        return None
    
    @property
    def lines(self) -> List[str]:
        """Get all captured lines."""
        return self._lines.copy()
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get stream statistics."""
        return {
            "total_bytes": self._total_bytes,
            "total_lines": self._total_lines,
            "buffer_size": len(self._buffer),
        }


class ProcessExecutor:
    """Manages asynchronous process execution with real-time streaming.
    
    Example:
        >>> executor = ProcessExecutor()
        >>> 
        >>> async def run():
        ...     process = await executor.execute(
        ...         ["python", "-c", "print('Hello')"],
        ...         working_dir=".",
        ...     )
        ...     print(f"Return code: {process.return_code}")
        ...     print(f"Output: {process.output}")
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        default_timeout: Optional[float] = None,
        shell: bool = False,
    ):
        """Initialize the executor.
        
        Args:
            event_bus: Event bus for emitting events
            default_timeout: Default timeout in seconds
            shell: Whether to run commands through shell
        """
        self.event_bus = event_bus or get_event_bus()
        self.default_timeout = default_timeout
        self.shell = shell
        
        # Active processes
        self._processes: Dict[str, tuple] = {}  # process_id -> (asyncio.Process, ProcessInfo)
        
        # Statistics
        self._stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "cancelled": 0,
            "timeouts": 0,
        }
    
    @property
    def active_processes(self) -> Dict[str, ProcessInfo]:
        """Get info about currently active processes."""
        return {
            pid: info for pid, (_, info) in self._processes.items()
            if info.is_running
        }
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get executor statistics."""
        return self._stats.copy()
    
    async def execute(
        self,
        command: Union[str, List[str]],
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        correlation_id: Optional[str] = None,
        process_id: Optional[str] = None,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[ProcessInfo], None]] = None,
    ) -> ProcessInfo:
        """Execute a command with real-time output streaming.
        
        Args:
            command: Command to execute (string or list)
            working_dir: Working directory for the process
            env: Environment variables (merged with current env)
            timeout: Timeout in seconds (overrides default)
            correlation_id: ID to correlate related events
            process_id: Custom process ID (auto-generated if not provided)
            on_stdout: Callback for each stdout line
            on_stderr: Callback for each stderr line
            on_complete: Callback when process completes
            
        Returns:
            ProcessInfo with execution results
        """
        # Generate IDs
        process_id = process_id or f"proc_{uuid.uuid4().hex[:8]}"
        correlation_id = correlation_id or process_id
        
        # Resolve working directory
        working_dir = working_dir or os.getcwd()
        working_dir = str(Path(working_dir).resolve())
        
        # Create process info
        info = ProcessInfo(
            process_id=process_id,
            command=command,
            working_dir=working_dir,
            correlation_id=correlation_id,
        )
        
        # Prepare environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        # Prepare command
        if isinstance(command, str):
            if self.shell:
                cmd_args = command
            else:
                # Simple split for non-shell mode
                import shlex
                cmd_args = shlex.split(command)
        else:
            cmd_args = command
        
        try:
            info.state = ProcessState.STARTING
            info.start_time = time.time()
            
            # Emit process started event
            emit_process_started(
                process_id,
                str(command),
                working_dir,
                correlation_id,
            )
            
            # Start the subprocess
            if self.shell and isinstance(cmd_args, str):
                process = await asyncio.create_subprocess_shell(
                    cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                    env=process_env,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                    env=process_env,
                )
            
            info.pid = process.pid
            info.state = ProcessState.RUNNING
            self._processes[process_id] = (process, info)
            self._stats["total_executions"] += 1
            
            # Create output stream handlers
            stdout_stream = ProcessOutputStream(
                process_id,
                is_stderr=False,
                event_bus=self.event_bus,
                correlation_id=correlation_id,
            )
            stderr_stream = ProcessOutputStream(
                process_id,
                is_stderr=True,
                event_bus=self.event_bus,
                correlation_id=correlation_id,
            )
            
            # Create tasks for reading stdout and stderr
            async def read_stdout():
                while True:
                    try:
                        chunk = await process.stdout.read(4096)
                        if not chunk:
                            break
                        lines = stdout_stream.feed(chunk)
                        for line in lines:
                            if on_stdout:
                                on_stdout(line)
                    except Exception:
                        break
                stdout_stream.flush()
            
            async def read_stderr():
                while True:
                    try:
                        chunk = await process.stderr.read(4096)
                        if not chunk:
                            break
                        lines = stderr_stream.feed(chunk)
                        for line in lines:
                            if on_stderr:
                                on_stderr(line)
                    except Exception:
                        break
                stderr_stream.flush()
            
            # Run reading tasks concurrently
            effective_timeout = timeout or self.default_timeout
            
            try:
                if effective_timeout:
                    _, _, return_code = await asyncio.wait_for(
                        asyncio.gather(
                            read_stdout(),
                            read_stderr(),
                            process.wait(),
                        ),
                        timeout=effective_timeout,
                    )
                else:
                    _, _, return_code = await asyncio.gather(
                        read_stdout(),
                        read_stderr(),
                        process.wait(),
                    )
                
                info.return_code = return_code
                info.state = ProcessState.COMPLETED if return_code == 0 else ProcessState.FAILED
                
                if return_code == 0:
                    self._stats["successful"] += 1
                else:
                    self._stats["failed"] += 1
                    
            except asyncio.TimeoutError:
                # Kill the process on timeout
                process.kill()
                await process.wait()
                info.state = ProcessState.TIMEOUT
                info.return_code = -1
                self._stats["timeouts"] += 1
                
            except asyncio.CancelledError:
                # Kill the process on cancellation
                process.kill()
                await process.wait()
                info.state = ProcessState.CANCELLED
                info.return_code = -1
                self._stats["cancelled"] += 1
                raise
            
            # Store captured output
            info.stdout_lines = stdout_stream.lines
            info.stderr_lines = stderr_stream.lines
            
        except Exception as e:
            info.state = ProcessState.FAILED
            info.stderr_lines.append(f"Execution error: {str(e)}")
            self._stats["failed"] += 1
            
        finally:
            info.end_time = time.time()
            self._processes.pop(process_id, None)
            
            # Emit completion event
            emit_process_completed(
                process_id,
                info.return_code or -1,
                info.duration_ms or 0,
                correlation_id,
            )
            
            # Call completion callback
            if on_complete:
                on_complete(info)
        
        return info
    
    async def execute_multiple(
        self,
        commands: List[Dict[str, Any]],
        max_concurrent: int = 4,
        correlation_id: Optional[str] = None,
    ) -> List[ProcessInfo]:
        """Execute multiple commands with concurrency limit.
        
        Args:
            commands: List of command dictionaries with execute() parameters
            max_concurrent: Maximum concurrent processes
            correlation_id: Shared correlation ID for all processes
            
        Returns:
            List of ProcessInfo results (in order of completion)
        """
        correlation_id = correlation_id or f"batch_{uuid.uuid4().hex[:8]}"
        results: List[ProcessInfo] = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(cmd_config: Dict[str, Any]) -> ProcessInfo:
            async with semaphore:
                cmd_config.setdefault("correlation_id", correlation_id)
                result = await self.execute(**cmd_config)
                results.append(result)
                return result
        
        tasks = [run_with_semaphore(cmd) for cmd in commands]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def kill(self, process_id: str) -> bool:
        """Kill a running process.
        
        Args:
            process_id: ID of the process to kill
            
        Returns:
            True if process was killed
        """
        if process_id not in self._processes:
            return False
        
        process, info = self._processes[process_id]
        
        try:
            process.kill()
            await process.wait()
            info.state = ProcessState.CANCELLED
            return True
        except Exception:
            return False
    
    async def kill_all(self) -> int:
        """Kill all running processes.
        
        Returns:
            Number of processes killed
        """
        killed = 0
        for process_id in list(self._processes.keys()):
            if await self.kill(process_id):
                killed += 1
        return killed


class StreamingProcessRunner:
    """High-level runner for streaming process execution.
    
    Provides a simple interface for running commands with real-time
    output and automatic event bus integration.
    """
    
    def __init__(self):
        self._executor = ProcessExecutor()
        self._event_bus = get_event_bus()
    
    async def run(
        self,
        command: Union[str, List[str]],
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
        capture_output: bool = True,
    ) -> ProcessInfo:
        """Run a command with streaming output.
        
        Args:
            command: Command to run
            working_dir: Working directory
            timeout: Timeout in seconds
            capture_output: Whether to capture output lines
            
        Returns:
            ProcessInfo with results
        """
        return await self._executor.execute(
            command,
            working_dir=working_dir,
            timeout=timeout,
        )
    
    async def run_script(
        self,
        script: str,
        interpreter: str = "python",
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ProcessInfo:
        """Run a script with an interpreter.
        
        Args:
            script: Script content
            interpreter: Interpreter command (python, bash, etc.)
            working_dir: Working directory
            timeout: Timeout in seconds
            
        Returns:
            ProcessInfo with results
        """
        # Use -c flag for inline script execution
        if interpreter in ("python", "python3"):
            command = [interpreter, "-c", script]
        elif interpreter in ("bash", "sh"):
            command = [interpreter, "-c", script]
        elif interpreter == "powershell":
            command = ["powershell", "-Command", script]
        else:
            command = [interpreter, "-c", script]
        
        return await self._executor.execute(
            command,
            working_dir=working_dir,
            timeout=timeout,
        )
    
    def subscribe_to_output(
        self,
        callback: Callable[[Event], None],
        source_filter: Optional[str] = None,
    ) -> str:
        """Subscribe to output events.
        
        Args:
            callback: Function to call for each output event
            source_filter: Filter by process ID
            
        Returns:
            Subscription ID
        """
        return self._event_bus.subscribe(
            callback,
            event_types={EventType.OUTPUT_LINE, EventType.ERROR_LINE},
            source_filter=source_filter,
        )
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        return self._event_bus.unsubscribe(subscription_id)


# Convenience function for quick execution
async def run_command(
    command: Union[str, List[str]],
    working_dir: Optional[str] = None,
    timeout: Optional[float] = None,
) -> ProcessInfo:
    """Run a command and return results.
    
    This is a convenience function for simple command execution
    with real-time event bus integration.
    
    Args:
        command: Command to execute
        working_dir: Working directory
        timeout: Timeout in seconds
        
    Returns:
        ProcessInfo with execution results
    """
    executor = ProcessExecutor()
    return await executor.execute(command, working_dir=working_dir, timeout=timeout)
