"""Script Execution Framework.

Phase 6: Natural Language Planning & Execution

Provides script execution capabilities including:
- Multi-language script support (Python, Bash, PowerShell, JS)
- Auto-detection of script language
- Environment setup and interpreter selection
- Argument passing and output capture
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.script_executor")


class ScriptLanguage(Enum):
    """Supported scripting languages."""
    PYTHON = "python"
    BASH = "bash"
    POWERSHELL = "powershell"
    JAVASCRIPT = "javascript"
    LUA = "lua"
    SHELL = "shell"  # Generic shell (cmd on Windows, sh on Unix)
    UNKNOWN = "unknown"


class ScriptSource(Enum):
    """Source of the script."""
    FILE = "file"      # Script from a file
    INLINE = "inline"  # Script provided as string
    STDIN = "stdin"    # Script from standard input


@dataclass
class ScriptInfo:
    """Information about a script."""
    path: Optional[Path]
    language: ScriptLanguage
    source: ScriptSource
    content: Optional[str] = None
    size_bytes: int = 0
    shebang: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path) if self.path else None,
            "language": self.language.value,
            "source": self.source.value,
            "size_bytes": self.size_bytes,
            "shebang": self.shebang,
        }


@dataclass
class InterpreterInfo:
    """Information about a script interpreter."""
    name: str
    path: str
    version: Optional[str] = None
    available: bool = True
    args: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "version": self.version,
            "available": self.available,
            "args": self.args,
        }


@dataclass
class ScriptResult:
    """Result of script execution."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    script_info: ScriptInfo
    interpreter: InterpreterInfo
    started_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "script_info": self.script_info.to_dict(),
            "interpreter": self.interpreter.to_dict(),
            "started_at": self.started_at.isoformat(),
        }
    
    @property
    def output(self) -> str:
        """Get combined output."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        return "\n".join(parts)
    
    @property
    def summary(self) -> str:
        """Get execution summary."""
        status = "✅" if self.success else "❌"
        return f"{status} Exit code: {self.exit_code} | Duration: {self.duration_seconds:.2f}s"


# Progress callback type
ProgressCallback = Callable[[str], None]  # Called with output lines


class InterpreterRegistry:
    """Registry of available interpreters."""
    
    def __init__(self):
        """Initialize the registry."""
        self._interpreters: Dict[ScriptLanguage, InterpreterInfo] = {}
        self._detect_interpreters()
    
    def _detect_interpreters(self) -> None:
        """Detect available interpreters."""
        is_windows = platform.system() == "Windows"
        
        # Python - use the same interpreter as Proxima
        python_path = sys.executable
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self._interpreters[ScriptLanguage.PYTHON] = InterpreterInfo(
            name="python",
            path=python_path,
            version=python_version,
            available=True,
        )
        
        # Bash
        if is_windows:
            # Try Git Bash, WSL bash
            git_bash = r"C:\Program Files\Git\bin\bash.exe"
            wsl_bash = r"C:\Windows\System32\bash.exe"
            
            if os.path.exists(git_bash):
                self._interpreters[ScriptLanguage.BASH] = InterpreterInfo(
                    name="bash",
                    path=git_bash,
                    available=True,
                )
            elif os.path.exists(wsl_bash):
                self._interpreters[ScriptLanguage.BASH] = InterpreterInfo(
                    name="bash",
                    path=wsl_bash,
                    available=True,
                )
            else:
                self._interpreters[ScriptLanguage.BASH] = InterpreterInfo(
                    name="bash",
                    path="bash",
                    available=False,
                )
        else:
            bash_path = shutil.which("bash") or "/bin/bash"
            self._interpreters[ScriptLanguage.BASH] = InterpreterInfo(
                name="bash",
                path=bash_path,
                available=os.path.exists(bash_path),
            )
        
        # PowerShell
        if is_windows:
            # Prefer pwsh (PowerShell Core) over Windows PowerShell
            pwsh_path = shutil.which("pwsh")
            if pwsh_path:
                self._interpreters[ScriptLanguage.POWERSHELL] = InterpreterInfo(
                    name="pwsh",
                    path=pwsh_path,
                    available=True,
                    args=["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"],
                )
            else:
                self._interpreters[ScriptLanguage.POWERSHELL] = InterpreterInfo(
                    name="powershell",
                    path="powershell.exe",
                    available=True,
                    args=["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"],
                )
        else:
            pwsh_path = shutil.which("pwsh")
            self._interpreters[ScriptLanguage.POWERSHELL] = InterpreterInfo(
                name="pwsh",
                path=pwsh_path or "pwsh",
                available=pwsh_path is not None,
                args=["-NoProfile", "-File"],
            )
        
        # JavaScript (Node.js)
        node_path = shutil.which("node")
        self._interpreters[ScriptLanguage.JAVASCRIPT] = InterpreterInfo(
            name="node",
            path=node_path or "node",
            available=node_path is not None,
        )
        
        # Lua
        lua_path = shutil.which("lua") or shutil.which("lua5.4") or shutil.which("lua5.3")
        self._interpreters[ScriptLanguage.LUA] = InterpreterInfo(
            name="lua",
            path=lua_path or "lua",
            available=lua_path is not None,
        )
        
        # Generic shell
        if is_windows:
            self._interpreters[ScriptLanguage.SHELL] = InterpreterInfo(
                name="cmd",
                path="cmd.exe",
                available=True,
                args=["/c"],
            )
        else:
            sh_path = shutil.which("sh") or "/bin/sh"
            self._interpreters[ScriptLanguage.SHELL] = InterpreterInfo(
                name="sh",
                path=sh_path,
                available=True,
            )
    
    def get_interpreter(self, language: ScriptLanguage) -> Optional[InterpreterInfo]:
        """Get interpreter for a language."""
        return self._interpreters.get(language)
    
    def is_available(self, language: ScriptLanguage) -> bool:
        """Check if interpreter is available."""
        info = self._interpreters.get(language)
        return info is not None and info.available
    
    def list_available(self) -> List[InterpreterInfo]:
        """List all available interpreters."""
        return [i for i in self._interpreters.values() if i.available]


class ScriptExecutor:
    """Execute scripts in multiple languages.
    
    Features:
    - Auto-detect script language
    - Support for file and inline scripts
    - Argument passing
    - Output capture
    - Environment variable support
    
    Example:
        >>> executor = ScriptExecutor()
        >>> 
        >>> # Execute a Python script file
        >>> result = await executor.execute_file("script.py")
        >>> 
        >>> # Execute inline Python code
        >>> result = await executor.execute_inline(
        ...     "print('Hello, World!')",
        ...     language=ScriptLanguage.PYTHON,
        ... )
        >>> 
        >>> # Execute with arguments
        >>> result = await executor.execute_file(
        ...     "process.py",
        ...     arguments=["--input", "data.txt"],
        ... )
    """
    
    # Extension to language mapping
    EXTENSION_MAP = {
        ".py": ScriptLanguage.PYTHON,
        ".pyw": ScriptLanguage.PYTHON,
        ".sh": ScriptLanguage.BASH,
        ".bash": ScriptLanguage.BASH,
        ".ps1": ScriptLanguage.POWERSHELL,
        ".psm1": ScriptLanguage.POWERSHELL,
        ".js": ScriptLanguage.JAVASCRIPT,
        ".mjs": ScriptLanguage.JAVASCRIPT,
        ".lua": ScriptLanguage.LUA,
        ".bat": ScriptLanguage.SHELL,
        ".cmd": ScriptLanguage.SHELL,
    }
    
    # Shebang to language mapping
    SHEBANG_MAP = {
        "python": ScriptLanguage.PYTHON,
        "python3": ScriptLanguage.PYTHON,
        "bash": ScriptLanguage.BASH,
        "sh": ScriptLanguage.BASH,
        "pwsh": ScriptLanguage.POWERSHELL,
        "powershell": ScriptLanguage.POWERSHELL,
        "node": ScriptLanguage.JAVASCRIPT,
        "lua": ScriptLanguage.LUA,
    }
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        timeout: float = 300.0,  # 5 minutes default
        capture_output: bool = True,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        """Initialize the script executor.
        
        Args:
            working_dir: Default working directory
            timeout: Default execution timeout in seconds
            capture_output: Whether to capture stdout/stderr
            env_vars: Additional environment variables
        """
        self.working_dir = working_dir or Path.cwd()
        self.timeout = timeout
        self.capture_output = capture_output
        self.env_vars = env_vars or {}
        
        self.registry = InterpreterRegistry()
        
        logger.info(f"ScriptExecutor initialized with working_dir={self.working_dir}")
    
    def detect_language(self, path: Path) -> ScriptLanguage:
        """Detect script language from file.
        
        Args:
            path: Path to script file
            
        Returns:
            Detected ScriptLanguage
        """
        # Try extension first
        ext = path.suffix.lower()
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]
        
        # Try shebang
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                if first_line.startswith("#!"):
                    shebang = first_line[2:].strip()
                    for key, lang in self.SHEBANG_MAP.items():
                        if key in shebang.lower():
                            return lang
        except Exception:
            pass
        
        return ScriptLanguage.UNKNOWN
    
    def get_script_info(self, path: Path) -> ScriptInfo:
        """Get information about a script file.
        
        Args:
            path: Path to script file
            
        Returns:
            ScriptInfo with details
        """
        path = Path(path)
        language = self.detect_language(path)
        
        shebang = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                if first_line.startswith("#!"):
                    shebang = first_line
        except Exception:
            pass
        
        size = path.stat().st_size if path.exists() else 0
        
        return ScriptInfo(
            path=path,
            language=language,
            source=ScriptSource.FILE,
            size_bytes=size,
            shebang=shebang,
        )
    
    async def execute_file(
        self,
        path: Union[str, Path],
        arguments: Optional[List[str]] = None,
        working_dir: Optional[Path] = None,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ScriptResult:
        """Execute a script file.
        
        Args:
            path: Path to script file
            arguments: Command-line arguments
            working_dir: Working directory (overrides default)
            env_vars: Additional environment variables
            timeout: Execution timeout (overrides default)
            progress_callback: Callback for output lines
            
        Returns:
            ScriptResult with execution details
        """
        path = Path(path)
        arguments = arguments or []
        working_dir = working_dir or self.working_dir
        timeout = timeout or self.timeout
        
        # Validate file exists
        if not path.exists():
            return ScriptResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Script not found: {path}",
                duration_seconds=0.0,
                script_info=ScriptInfo(
                    path=path,
                    language=ScriptLanguage.UNKNOWN,
                    source=ScriptSource.FILE,
                ),
                interpreter=InterpreterInfo(
                    name="unknown",
                    path="",
                    available=False,
                ),
            )
        
        # Get script info
        script_info = self.get_script_info(path)
        
        # Get interpreter
        interpreter = self.registry.get_interpreter(script_info.language)
        if not interpreter or not interpreter.available:
            return ScriptResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"No interpreter available for {script_info.language.value}",
                duration_seconds=0.0,
                script_info=script_info,
                interpreter=InterpreterInfo(
                    name="unknown",
                    path="",
                    available=False,
                ),
            )
        
        # Build command
        cmd = [interpreter.path] + interpreter.args + [str(path.resolve())] + arguments
        
        # Merge environment variables
        env = os.environ.copy()
        env.update(self.env_vars)
        if env_vars:
            env.update(env_vars)
        
        return await self._execute_command(
            cmd=cmd,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
            script_info=script_info,
            interpreter=interpreter,
            progress_callback=progress_callback,
        )
    
    async def execute_inline(
        self,
        code: str,
        language: ScriptLanguage = ScriptLanguage.PYTHON,
        arguments: Optional[List[str]] = None,
        working_dir: Optional[Path] = None,
        env_vars: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ScriptResult:
        """Execute inline script code.
        
        Args:
            code: Script code to execute
            language: Script language
            arguments: Command-line arguments
            working_dir: Working directory
            env_vars: Additional environment variables
            timeout: Execution timeout
            progress_callback: Callback for output lines
            
        Returns:
            ScriptResult with execution details
        """
        arguments = arguments or []
        working_dir = working_dir or self.working_dir
        timeout = timeout or self.timeout
        
        # Get interpreter
        interpreter = self.registry.get_interpreter(language)
        if not interpreter or not interpreter.available:
            return ScriptResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"No interpreter available for {language.value}",
                duration_seconds=0.0,
                script_info=ScriptInfo(
                    path=None,
                    language=language,
                    source=ScriptSource.INLINE,
                    content=code,
                ),
                interpreter=InterpreterInfo(
                    name="unknown",
                    path="",
                    available=False,
                ),
            )
        
        script_info = ScriptInfo(
            path=None,
            language=language,
            source=ScriptSource.INLINE,
            content=code,
            size_bytes=len(code.encode("utf-8")),
        )
        
        # Build command based on language
        if language == ScriptLanguage.PYTHON:
            cmd = [interpreter.path, "-c", code] + arguments
        elif language == ScriptLanguage.BASH:
            cmd = [interpreter.path, "-c", code] + arguments
        elif language == ScriptLanguage.POWERSHELL:
            cmd = [interpreter.path, "-NoProfile", "-Command", code] + arguments
        elif language == ScriptLanguage.JAVASCRIPT:
            cmd = [interpreter.path, "-e", code] + arguments
        elif language == ScriptLanguage.LUA:
            cmd = [interpreter.path, "-e", code] + arguments
        else:
            # Write to temp file for unknown languages
            return await self._execute_via_temp_file(
                code, language, arguments, working_dir, env_vars, timeout, progress_callback
            )
        
        # Merge environment variables
        env = os.environ.copy()
        env.update(self.env_vars)
        if env_vars:
            env.update(env_vars)
        
        return await self._execute_command(
            cmd=cmd,
            working_dir=working_dir,
            env=env,
            timeout=timeout,
            script_info=script_info,
            interpreter=interpreter,
            progress_callback=progress_callback,
        )
    
    async def _execute_via_temp_file(
        self,
        code: str,
        language: ScriptLanguage,
        arguments: List[str],
        working_dir: Path,
        env_vars: Optional[Dict[str, str]],
        timeout: float,
        progress_callback: Optional[ProgressCallback],
    ) -> ScriptResult:
        """Execute code by writing to a temporary file."""
        # Determine extension
        ext_map = {v: k for k, v in self.EXTENSION_MAP.items()}
        ext = ext_map.get(language, ".txt")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=ext,
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            temp_path = Path(f.name)
        
        try:
            return await self.execute_file(
                path=temp_path,
                arguments=arguments,
                working_dir=working_dir,
                env_vars=env_vars,
                timeout=timeout,
                progress_callback=progress_callback,
            )
        finally:
            # Clean up temp file
            try:
                temp_path.unlink()
            except Exception:
                pass
    
    async def _execute_command(
        self,
        cmd: List[str],
        working_dir: Path,
        env: Dict[str, str],
        timeout: float,
        script_info: ScriptInfo,
        interpreter: InterpreterInfo,
        progress_callback: Optional[ProgressCallback],
    ) -> ScriptResult:
        """Execute a command and capture output."""
        import time
        
        start_time = time.time()
        
        try:
            # Create process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE if self.capture_output else None,
                stderr=asyncio.subprocess.PIPE if self.capture_output else None,
                cwd=str(working_dir),
                env=env,
            )
            
            stdout_lines: List[str] = []
            stderr_lines: List[str] = []
            
            async def read_stream(
                stream: asyncio.StreamReader,
                lines: List[str],
                is_stderr: bool = False,
            ):
                """Read from stream and collect lines."""
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n\r")
                    lines.append(decoded)
                    
                    if progress_callback and not is_stderr:
                        progress_callback(decoded)
            
            # Read stdout and stderr concurrently
            if self.capture_output:
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stream(process.stdout, stdout_lines, False),
                        read_stream(process.stderr, stderr_lines, True),
                    ),
                    timeout=timeout,
                )
            
            # Wait for process to complete
            await asyncio.wait_for(process.wait(), timeout=timeout)
            
            duration = time.time() - start_time
            
            return ScriptResult(
                success=process.returncode == 0,
                exit_code=process.returncode,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                duration_seconds=duration,
                script_info=script_info,
                interpreter=interpreter,
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            
            # Try to terminate the process
            try:
                process.terminate()
                await asyncio.sleep(0.5)
                process.kill()
            except Exception:
                pass
            
            return ScriptResult(
                success=False,
                exit_code=-1,
                stdout="\n".join(stdout_lines) if "stdout_lines" in locals() else "",
                stderr=f"Execution timed out after {timeout} seconds",
                duration_seconds=duration,
                script_info=script_info,
                interpreter=interpreter,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            return ScriptResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                script_info=script_info,
                interpreter=interpreter,
            )
    
    def list_available_languages(self) -> List[ScriptLanguage]:
        """List languages with available interpreters."""
        return [
            lang for lang in ScriptLanguage
            if self.registry.is_available(lang)
        ]


# Global instance
_executor: Optional[ScriptExecutor] = None


def get_script_executor() -> ScriptExecutor:
    """Get the global ScriptExecutor instance."""
    global _executor
    if _executor is None:
        _executor = ScriptExecutor()
    return _executor


async def execute_script(
    path: Union[str, Path],
    arguments: Optional[List[str]] = None,
) -> ScriptResult:
    """Convenience function to execute a script file."""
    return await get_script_executor().execute_file(path, arguments)


async def execute_code(
    code: str,
    language: ScriptLanguage = ScriptLanguage.PYTHON,
) -> ScriptResult:
    """Convenience function to execute inline code."""
    return await get_script_executor().execute_inline(code, language)
