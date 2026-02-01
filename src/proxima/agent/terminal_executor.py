"""Terminal Executor for Proxima Agent.

Provides capabilities for:
- Executing shell commands (PowerShell on Windows, bash on Unix)
- Building and compiling backends (LRET, Cirq, Qiskit, cuQuantum, etc.)
- Running scripts with real-time output streaming
- Administrative access management
- Directory navigation and file operations

This module enables the AI agent to interact with the system terminal
for backend compilation, script execution, and system operations.
"""

from __future__ import annotations

import asyncio
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.terminal")


class TerminalType(Enum):
    """Terminal shell type."""
    POWERSHELL = "powershell"
    CMD = "cmd"
    BASH = "bash"
    ZSH = "zsh"
    SH = "sh"
    AUTO = "auto"


class ExecutionStatus(Enum):
    """Status of terminal execution."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class TerminalOutput:
    """Output from a terminal command."""
    stdout: str
    stderr: str
    return_code: int
    execution_time_ms: float
    status: ExecutionStatus
    command: str
    working_dir: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success(self) -> bool:
        """Check if command executed successfully."""
        return self.return_code == 0 and self.status == ExecutionStatus.COMPLETED
    
    @property
    def combined_output(self) -> str:
        """Get combined stdout and stderr."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[STDERR]\n{self.stderr}")
        return "\n".join(parts) if parts else "(no output)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time_ms": self.execution_time_ms,
            "status": self.status.name,
            "command": self.command,
            "working_dir": self.working_dir,
            "timestamp": self.timestamp,
            "success": self.success,
        }


@dataclass
class TerminalSession:
    """An active terminal session."""
    id: str
    terminal_type: TerminalType
    working_dir: str
    process: Optional[subprocess.Popen] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    history: List[TerminalOutput] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    is_admin: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "terminal_type": self.terminal_type.value,
            "working_dir": self.working_dir,
            "started_at": self.started_at,
            "history_count": len(self.history),
            "is_admin": self.is_admin,
        }


class StreamingOutputHandler:
    """Handler for streaming command output in real-time."""
    
    def __init__(
        self,
        stdout_callback: Optional[Callable[[str], None]] = None,
        stderr_callback: Optional[Callable[[str], None]] = None,
    ):
        self.stdout_callback = stdout_callback
        self.stderr_callback = stderr_callback
        self.stdout_lines: List[str] = []
        self.stderr_lines: List[str] = []
        self._lock = threading.Lock()
    
    def handle_stdout(self, line: str) -> None:
        """Handle a line of stdout."""
        with self._lock:
            self.stdout_lines.append(line)
        if self.stdout_callback:
            self.stdout_callback(line)
    
    def handle_stderr(self, line: str) -> None:
        """Handle a line of stderr."""
        with self._lock:
            self.stderr_lines.append(line)
        if self.stderr_callback:
            self.stderr_callback(line)
    
    def get_stdout(self) -> str:
        """Get all stdout."""
        with self._lock:
            return "\n".join(self.stdout_lines)
    
    def get_stderr(self) -> str:
        """Get all stderr."""
        with self._lock:
            return "\n".join(self.stderr_lines)


class TerminalExecutor:
    """Execute commands in system terminal.
    
    Provides cross-platform terminal execution with:
    - Real-time output streaming
    - Timeout management
    - Working directory control
    - Environment variable management
    - Administrative privilege handling
    
    Example:
        >>> executor = TerminalExecutor()
        >>> output = executor.execute("pip install qiskit")
        >>> if output.success:
        ...     print("Installation complete!")
        
        # With streaming
        >>> def on_output(line):
        ...     print(f">> {line}")
        >>> output = executor.execute_streaming("python build.py", stdout_callback=on_output)
    """
    
    # Backend build commands mapping
    BACKEND_BUILD_COMMANDS = {
        "lret_cirq_scalability": [
            "pip install -e .",
            "python setup.py build",
        ],
        "lret_pennylane_hybrid": [
            "pip install -e .",
            "pip install pennylane",
        ],
        "lret_phase_7_unified": [
            "pip install -e .",
            "python scripts/build_unified.py",
        ],
        "cirq": [
            "pip install cirq",
        ],
        "qiskit_aer": [
            "pip install qiskit qiskit-aer",
        ],
        "quest": [
            "git clone https://github.com/QuEST-Kit/QuEST.git",
            "cd QuEST && mkdir build && cd build && cmake .. && make",
        ],
        "qsim": [
            "pip install qsimcirq",
        ],
        "cuquantum": [
            "pip install cuquantum cuquantum-python",
        ],
    }
    
    def __init__(
        self,
        default_terminal: TerminalType = TerminalType.AUTO,
        default_timeout: float = 300.0,
        default_working_dir: Optional[str] = None,
    ):
        """Initialize the terminal executor.
        
        Args:
            default_terminal: Default terminal type (auto-detected if AUTO)
            default_timeout: Default command timeout in seconds
            default_working_dir: Default working directory for commands
        """
        self.default_terminal = self._detect_terminal() if default_terminal == TerminalType.AUTO else default_terminal
        self.default_timeout = default_timeout
        self.default_working_dir = default_working_dir or os.getcwd()
        self.sessions: Dict[str, TerminalSession] = {}
        self._session_counter = 0
        self._lock = threading.Lock()
        
        logger.info(f"TerminalExecutor initialized with {self.default_terminal.value} shell")
    
    @staticmethod
    def _detect_terminal() -> TerminalType:
        """Detect the appropriate terminal type for the current platform."""
        system = platform.system().lower()
        if system == "windows":
            # Prefer PowerShell on Windows
            if shutil.which("pwsh") or shutil.which("powershell"):
                return TerminalType.POWERSHELL
            return TerminalType.CMD
        elif system == "darwin":
            # macOS - prefer zsh (default since Catalina)
            if shutil.which("zsh"):
                return TerminalType.ZSH
            return TerminalType.BASH
        else:
            # Linux/Unix
            if shutil.which("bash"):
                return TerminalType.BASH
            return TerminalType.SH
    
    def _get_shell_command(self, terminal_type: TerminalType) -> Tuple[str, List[str]]:
        """Get shell executable and args for terminal type."""
        if terminal_type == TerminalType.POWERSHELL:
            # Try pwsh (PowerShell Core) first, then powershell (Windows PowerShell)
            if shutil.which("pwsh"):
                return "pwsh", ["-NoProfile", "-NonInteractive", "-Command"]
            return "powershell", ["-NoProfile", "-NonInteractive", "-Command"]
        elif terminal_type == TerminalType.CMD:
            return "cmd", ["/c"]
        elif terminal_type == TerminalType.BASH:
            return "bash", ["-c"]
        elif terminal_type == TerminalType.ZSH:
            return "zsh", ["-c"]
        else:
            return "sh", ["-c"]
    
    def create_session(
        self,
        working_dir: Optional[str] = None,
        terminal_type: Optional[TerminalType] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> TerminalSession:
        """Create a new terminal session.
        
        Args:
            working_dir: Working directory for the session
            terminal_type: Terminal type (uses default if not specified)
            environment: Additional environment variables
            
        Returns:
            New terminal session
        """
        with self._lock:
            self._session_counter += 1
            session_id = f"session_{self._session_counter}_{int(time.time())}"
        
        session = TerminalSession(
            id=session_id,
            terminal_type=terminal_type or self.default_terminal,
            working_dir=working_dir or self.default_working_dir,
            environment=environment or {},
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created terminal session: {session_id}")
        return session
    
    def close_session(self, session_id: str) -> bool:
        """Close a terminal session.
        
        Args:
            session_id: ID of session to close
            
        Returns:
            True if session was closed, False if not found
        """
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            if session.process and session.process.poll() is None:
                session.process.terminate()
            logger.info(f"Closed terminal session: {session_id}")
            return True
        return False
    
    def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
        terminal_type: Optional[TerminalType] = None,
        environment: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        capture_output: bool = True,
    ) -> TerminalOutput:
        """Execute a command in the terminal.
        
        Args:
            command: Command to execute
            working_dir: Working directory (uses default if not specified)
            timeout: Command timeout in seconds (uses default if not specified)
            terminal_type: Terminal type (uses default if not specified)
            environment: Additional environment variables
            session_id: Optional session ID to use
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            TerminalOutput with command results
        """
        start_time = time.perf_counter()
        
        # Resolve working directory
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            work_dir = working_dir or session.working_dir
            term_type = terminal_type or session.terminal_type
            env = {**session.environment, **(environment or {})}
        else:
            work_dir = working_dir or self.default_working_dir
            term_type = terminal_type or self.default_terminal
            env = environment or {}
        
        # Ensure working directory exists
        work_dir = os.path.abspath(work_dir)
        if not os.path.exists(work_dir):
            return TerminalOutput(
                stdout="",
                stderr=f"Working directory does not exist: {work_dir}",
                return_code=-1,
                execution_time_ms=0,
                status=ExecutionStatus.FAILED,
                command=command,
                working_dir=work_dir,
            )
        
        # Build shell command
        shell_exe, shell_args = self._get_shell_command(term_type)
        full_command = [shell_exe] + shell_args + [command]
        
        # Build environment
        full_env = os.environ.copy()
        full_env.update(env)
        
        timeout_val = timeout or self.default_timeout
        
        try:
            logger.debug(f"Executing: {command} in {work_dir}")
            
            process = subprocess.Popen(
                full_command,
                cwd=work_dir,
                env=full_env,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout_val)
                return_code = process.returncode
                status = ExecutionStatus.COMPLETED
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                status = ExecutionStatus.TIMEOUT
                stderr = f"Command timed out after {timeout_val}s\n" + (stderr or "")
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            output = TerminalOutput(
                stdout=stdout or "",
                stderr=stderr or "",
                return_code=return_code,
                execution_time_ms=elapsed,
                status=status,
                command=command,
                working_dir=work_dir,
            )
            
            # Add to session history if using a session
            if session_id and session_id in self.sessions:
                self.sessions[session_id].history.append(output)
            
            logger.debug(f"Command completed with code {return_code} in {elapsed:.1f}ms")
            return output
            
        except FileNotFoundError:
            elapsed = (time.perf_counter() - start_time) * 1000
            return TerminalOutput(
                stdout="",
                stderr=f"Shell not found: {shell_exe}",
                return_code=-1,
                execution_time_ms=elapsed,
                status=ExecutionStatus.FAILED,
                command=command,
                working_dir=work_dir,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"Command execution error: {e}")
            return TerminalOutput(
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time_ms=elapsed,
                status=ExecutionStatus.FAILED,
                command=command,
                working_dir=work_dir,
            )
    
    def execute_streaming(
        self,
        command: str,
        stdout_callback: Optional[Callable[[str], None]] = None,
        stderr_callback: Optional[Callable[[str], None]] = None,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
        terminal_type: Optional[TerminalType] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> TerminalOutput:
        """Execute a command with real-time output streaming.
        
        Args:
            command: Command to execute
            stdout_callback: Callback for stdout lines
            stderr_callback: Callback for stderr lines
            working_dir: Working directory
            timeout: Command timeout
            terminal_type: Terminal type
            environment: Environment variables
            
        Returns:
            TerminalOutput with results
        """
        start_time = time.perf_counter()
        handler = StreamingOutputHandler(stdout_callback, stderr_callback)
        
        work_dir = working_dir or self.default_working_dir
        term_type = terminal_type or self.default_terminal
        timeout_val = timeout or self.default_timeout
        
        shell_exe, shell_args = self._get_shell_command(term_type)
        full_command = [shell_exe] + shell_args + [command]
        
        full_env = os.environ.copy()
        if environment:
            full_env.update(environment)
        
        try:
            process = subprocess.Popen(
                full_command,
                cwd=work_dir,
                env=full_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )
            
            # Create threads for reading stdout and stderr
            def read_stream(stream, handler_func):
                for line in iter(stream.readline, ""):
                    if line:
                        handler_func(line.rstrip())
                stream.close()
            
            stdout_thread = threading.Thread(
                target=read_stream, args=(process.stdout, handler.handle_stdout)
            )
            stderr_thread = threading.Thread(
                target=read_stream, args=(process.stderr, handler.handle_stderr)
            )
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process with timeout
            try:
                process.wait(timeout=timeout_val)
                status = ExecutionStatus.COMPLETED
            except subprocess.TimeoutExpired:
                process.kill()
                status = ExecutionStatus.TIMEOUT
            
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            return TerminalOutput(
                stdout=handler.get_stdout(),
                stderr=handler.get_stderr(),
                return_code=process.returncode or 0,
                execution_time_ms=elapsed,
                status=status,
                command=command,
                working_dir=work_dir,
            )
            
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return TerminalOutput(
                stdout=handler.get_stdout(),
                stderr=str(e),
                return_code=-1,
                execution_time_ms=elapsed,
                status=ExecutionStatus.FAILED,
                command=command,
                working_dir=work_dir,
            )
    
    async def execute_async(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
        terminal_type: Optional[TerminalType] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> TerminalOutput:
        """Execute a command asynchronously.
        
        Args:
            command: Command to execute
            working_dir: Working directory
            timeout: Command timeout
            terminal_type: Terminal type
            environment: Environment variables
            
        Returns:
            TerminalOutput with results
        """
        start_time = time.perf_counter()
        
        work_dir = working_dir or self.default_working_dir
        term_type = terminal_type or self.default_terminal
        timeout_val = timeout or self.default_timeout
        
        shell_exe, shell_args = self._get_shell_command(term_type)
        full_command = " ".join([shell_exe] + shell_args + [command])
        
        full_env = os.environ.copy()
        if environment:
            full_env.update(environment)
        
        try:
            process = await asyncio.create_subprocess_shell(
                full_command,
                cwd=work_dir,
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_val
                )
                status = ExecutionStatus.COMPLETED
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                stdout, stderr = b"", b"Command timed out"
                status = ExecutionStatus.TIMEOUT
            
            elapsed = (time.perf_counter() - start_time) * 1000
            
            return TerminalOutput(
                stdout=stdout.decode() if stdout else "",
                stderr=stderr.decode() if stderr else "",
                return_code=process.returncode or 0,
                execution_time_ms=elapsed,
                status=status,
                command=command,
                working_dir=work_dir,
            )
            
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return TerminalOutput(
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time_ms=elapsed,
                status=ExecutionStatus.FAILED,
                command=command,
                working_dir=work_dir,
            )
    
    def build_backend(
        self,
        backend_name: str,
        working_dir: Optional[str] = None,
        stdout_callback: Optional[Callable[[str], None]] = None,
        custom_commands: Optional[List[str]] = None,
    ) -> List[TerminalOutput]:
        """Build a quantum computing backend.
        
        Args:
            backend_name: Name of the backend to build
            working_dir: Working directory for build
            stdout_callback: Callback for build output
            custom_commands: Custom build commands (overrides default)
            
        Returns:
            List of TerminalOutput for each build step
        """
        # Normalize backend name
        normalized_name = backend_name.lower().replace("-", "_").replace(" ", "_")
        
        # Get build commands
        if custom_commands:
            commands = custom_commands
        elif normalized_name in self.BACKEND_BUILD_COMMANDS:
            commands = self.BACKEND_BUILD_COMMANDS[normalized_name]
        else:
            # Generic pip install
            commands = [f"pip install {backend_name}"]
        
        results = []
        for cmd in commands:
            if stdout_callback:
                stdout_callback(f"\n=== Executing: {cmd} ===\n")
            
            output = self.execute_streaming(
                cmd,
                stdout_callback=stdout_callback,
                stderr_callback=stdout_callback,  # Also stream stderr
                working_dir=working_dir,
            )
            results.append(output)
            
            if not output.success:
                logger.warning(f"Build step failed: {cmd}")
                break
        
        return results
    
    def check_admin_privileges(self) -> bool:
        """Check if running with administrator/root privileges."""
        if platform.system() == "Windows":
            try:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            except Exception:
                return False
        else:
            return os.geteuid() == 0
    
    def request_admin_execution(
        self,
        command: str,
        working_dir: Optional[str] = None,
    ) -> TerminalOutput:
        """Request execution with elevated privileges.
        
        On Windows, this uses 'runas'. On Unix, it suggests using sudo.
        
        Args:
            command: Command to execute with admin rights
            working_dir: Working directory
            
        Returns:
            TerminalOutput with results
        """
        if platform.system() == "Windows":
            # Use PowerShell Start-Process with -Verb RunAs
            admin_cmd = f'Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -Command {command}"'
            return self.execute(admin_cmd, working_dir=working_dir)
        else:
            # On Unix, prepend sudo
            return self.execute(f"sudo {command}", working_dir=working_dir)
    
    def navigate_to(self, path: str, session_id: Optional[str] = None) -> bool:
        """Change working directory for a session or default.
        
        Args:
            path: Path to navigate to
            session_id: Optional session ID
            
        Returns:
            True if navigation successful
        """
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return False
        
        if session_id and session_id in self.sessions:
            self.sessions[session_id].working_dir = abs_path
        else:
            self.default_working_dir = abs_path
        
        return True
    
    def get_current_dir(self, session_id: Optional[str] = None) -> str:
        """Get current working directory.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Current working directory path
        """
        if session_id and session_id in self.sessions:
            return self.sessions[session_id].working_dir
        return self.default_working_dir
    
    def list_directory(
        self,
        path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List contents of a directory.
        
        Args:
            path: Path to list (uses working dir if not specified)
            session_id: Optional session ID
            
        Returns:
            List of file/directory info dictionaries
        """
        target_path = path or self.get_current_dir(session_id)
        abs_path = os.path.abspath(target_path)
        
        if not os.path.isdir(abs_path):
            return []
        
        entries = []
        try:
            for entry in os.scandir(abs_path):
                try:
                    stat = entry.stat()
                    entries.append({
                        "name": entry.name,
                        "path": entry.path,
                        "is_dir": entry.is_dir(),
                        "is_file": entry.is_file(),
                        "size": stat.st_size if entry.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except (OSError, PermissionError):
                    entries.append({
                        "name": entry.name,
                        "path": entry.path,
                        "is_dir": entry.is_dir(),
                        "is_file": entry.is_file(),
                        "size": 0,
                        "modified": "",
                        "error": "Permission denied",
                    })
        except (OSError, PermissionError) as e:
            logger.error(f"Error listing directory {abs_path}: {e}")
        
        return sorted(entries, key=lambda x: (not x.get("is_dir", False), x.get("name", "").lower()))
    
    def read_file(self, path: str, max_size: int = 1024 * 1024) -> Tuple[str, bool]:
        """Read contents of a file.
        
        Args:
            path: Path to file
            max_size: Maximum file size to read (default 1MB)
            
        Returns:
            Tuple of (content, success)
        """
        try:
            abs_path = os.path.abspath(path)
            if not os.path.isfile(abs_path):
                return f"File not found: {path}", False
            
            size = os.path.getsize(abs_path)
            if size > max_size:
                return f"File too large: {size} bytes (max {max_size})", False
            
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(), True
        except Exception as e:
            return str(e), False
    
    def write_file(self, path: str, content: str) -> Tuple[str, bool]:
        """Write content to a file.
        
        Args:
            path: Path to file
            content: Content to write
            
        Returns:
            Tuple of (message, success)
        """
        try:
            abs_path = os.path.abspath(path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Written {len(content)} bytes to {path}", True
        except Exception as e:
            return str(e), False
