"""Agent Controller for Proxima.

The main controller that integrates all agent components:
- Terminal execution
- File system operations
- Git operations
- Backend modifications
- Safety and consent management
- Multi-terminal monitoring

Provides a unified interface for AI assistant integration
with LLM function calling support.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from proxima.utils.logging import get_logger

from .terminal_executor import TerminalExecutor, TerminalOutput, TerminalType
from .session_manager import AgentSessionManager, AgentSession
from .tools import AgentTools, ToolDefinition, ToolResult, ToolCategory
from .safety import SafetyManager, ConsentRequest, ConsentResponse, ConsentType, ConsentDecision
from .git_operations import GitOperations, GitResult
from .backend_modifier import BackendModifier, CodeChange, ModificationResult
from .multi_terminal_monitor import MultiTerminalMonitor, TerminalEvent, TerminalEventType

logger = get_logger("agent.controller")


@dataclass
class AgentPlan:
    """A plan for multi-step execution."""
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending, executing, completed, failed, cancelled
    current_step: int = 0
    results: List[ToolResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "created_at": self.created_at,
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }


class AgentController:
    """Main controller for the Proxima AI Agent.
    
    Integrates all agent capabilities and provides:
    - Tool execution for LLM function calling
    - Session management for persistent state
    - Safety/consent for dangerous operations
    - Multi-terminal monitoring
    - Plan creation and execution
    
    Example:
        >>> controller = AgentController()
        >>> controller.start()
        >>> 
        >>> # Execute a tool
        >>> result = controller.execute_tool("execute_command", {
        ...     "command": "pip list",
        ...     "working_dir": "/path/to/project"
        ... })
        >>> 
        >>> # Get tools for LLM
        >>> tools = controller.get_openai_tools()
    """
    
    def __init__(
        self,
        project_root: Optional[str] = None,
        auto_approve_safe: bool = True,
        consent_callback: Optional[Callable[[ConsentRequest], ConsentResponse]] = None,
    ):
        """Initialize the agent controller.
        
        Args:
            project_root: Root directory of the project
            auto_approve_safe: Auto-approve safe operations
            consent_callback: Callback for consent requests (for TUI integration)
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        
        # Initialize components
        self.terminal_executor = TerminalExecutor(
            default_working_dir=str(self.project_root)
        )
        self.session_manager = AgentSessionManager(
            terminal_executor=self.terminal_executor
        )
        self.safety_manager = SafetyManager(
            consent_callback=consent_callback,
            auto_approve_safe=auto_approve_safe,
            audit_log_path=str(self.project_root / ".proxima" / "agent_audit.log"),
        )
        self.git_ops = GitOperations(self.terminal_executor)
        self.backend_modifier = BackendModifier(
            rollback_manager=self.safety_manager.rollback_manager,
            project_root=str(self.project_root),
        )
        self.terminal_monitor = MultiTerminalMonitor()
        self.tools = AgentTools()
        
        # State
        self._current_session: Optional[AgentSession] = None
        self._plans: Dict[str, AgentPlan] = {}
        self._plan_counter = 0
        self._running = False
        self._event_listeners: List[Callable[[TerminalEvent], None]] = []
        
        # Register tool handlers
        self._register_tool_handlers()
        
        logger.info(f"AgentController initialized for {self.project_root}")
    
    def _register_tool_handlers(self) -> None:
        """Register handlers for all tools."""
        # Terminal tools
        self.tools.register_handler("execute_command", self._handle_execute_command)
        self.tools.register_handler("build_backend", self._handle_build_backend)
        self.tools.register_handler("run_script", self._handle_run_script)
        
        # File system tools
        self.tools.register_handler("read_file", self._handle_read_file)
        self.tools.register_handler("write_file", self._handle_write_file)
        self.tools.register_handler("list_directory", self._handle_list_directory)
        self.tools.register_handler("navigate_to", self._handle_navigate_to)
        self.tools.register_handler("find_files", self._handle_find_files)
        
        # Git tools
        self.tools.register_handler("git_clone", self._handle_git_clone)
        self.tools.register_handler("git_pull", self._handle_git_pull)
        self.tools.register_handler("git_push", self._handle_git_push)
        self.tools.register_handler("git_status", self._handle_git_status)
        self.tools.register_handler("git_commit", self._handle_git_commit)
        
        # Backend tools
        self.tools.register_handler("modify_backend_code", self._handle_modify_backend_code)
        self.tools.register_handler("rollback_modification", self._handle_rollback_modification)
        
        # System tools
        self.tools.register_handler("get_system_info", self._handle_get_system_info)
        self.tools.register_handler("check_admin", self._handle_check_admin)
        self.tools.register_handler("request_admin", self._handle_request_admin)
        
        # Planning tools
        self.tools.register_handler("create_plan", self._handle_create_plan)
        self.tools.register_handler("execute_plan", self._handle_execute_plan)
    
    def start(self) -> None:
        """Start the agent controller."""
        if self._running:
            return
        
        self._running = True
        self.terminal_monitor.start()
        
        # Create default session
        self._current_session = self.session_manager.create_session(
            name="default",
            working_dir=str(self.project_root),
        )
        
        logger.info("AgentController started")
    
    def stop(self) -> None:
        """Stop the agent controller."""
        self._running = False
        self.terminal_monitor.stop()
        self.session_manager.shutdown()
        logger.info("AgentController stopped")
    
    def add_event_listener(self, listener: Callable[[TerminalEvent], None]) -> None:
        """Add a listener for terminal events.
        
        Args:
            listener: Callback for events
        """
        self._event_listeners.append(listener)
        self.terminal_monitor.add_event_listener(listener)
    
    def remove_event_listener(self, listener: Callable[[TerminalEvent], None]) -> bool:
        """Remove an event listener."""
        try:
            self._event_listeners.remove(listener)
            self.terminal_monitor.remove_event_listener(listener)
            return True
        except ValueError:
            return False
    
    def get_current_session(self) -> Optional[AgentSession]:
        """Get the current agent session."""
        return self._current_session
    
    def get_tools(
        self,
        category: Optional[ToolCategory] = None,
        include_dangerous: bool = False,
    ) -> List[ToolDefinition]:
        """Get available tool definitions.
        
        Args:
            category: Filter by category
            include_dangerous: Include dangerous tools
            
        Returns:
            List of tool definitions
        """
        return self.tools.get_tool_definitions(category, include_dangerous)
    
    def get_openai_tools(
        self,
        category: Optional[ToolCategory] = None,
        include_dangerous: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get tools in OpenAI function calling format.
        
        Args:
            category: Filter by category
            include_dangerous: Include dangerous tools
            
        Returns:
            List of OpenAI tool definitions
        """
        return self.tools.get_openai_tools(category, include_dangerous)
    
    def get_anthropic_tools(
        self,
        category: Optional[ToolCategory] = None,
        include_dangerous: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get tools in Anthropic format.
        
        Args:
            category: Filter by category
            include_dangerous: Include dangerous tools
            
        Returns:
            List of Anthropic tool definitions
        """
        return self.tools.get_anthropic_tools(category, include_dangerous)
    
    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        require_consent: bool = True,
    ) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            require_consent: Whether to require consent for dangerous ops
            
        Returns:
            Tool result
        """
        # Validate arguments
        valid, error = self.tools.validate_arguments(tool_name, arguments)
        if not valid:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=error,
            )
        
        # Get tool definition
        tool = self.tools.get_tool(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool not found: {tool_name}",
            )
        
        # Check consent if required
        if require_consent and tool.requires_consent:
            if self.safety_manager.requires_consent(tool_name):
                # Create consent request
                request = self.safety_manager.request_consent(
                    consent_type=self._get_consent_type(tool),
                    operation=tool_name,
                    description=tool.consent_message or tool.description,
                    tool_name=tool_name,
                    details=arguments,
                    risk_level="high" if tool.dangerous else "medium",
                )
                
                # Wait for consent
                response = self.safety_manager.wait_for_consent(request.id, timeout=300)
                
                if not response or not response.approved:
                    return ToolResult(
                        tool_name=tool_name,
                        success=False,
                        result=None,
                        error="Operation not approved by user",
                    )
        
        # Execute the tool
        return self.tools.execute(tool_name, arguments)
    
    async def execute_tool_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        require_consent: bool = True,
    ) -> ToolResult:
        """Execute a tool asynchronously.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            require_consent: Whether to require consent
            
        Returns:
            Tool result
        """
        # For now, run synchronously in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute_tool(tool_name, arguments, require_consent)
        )
    
    def _get_consent_type(self, tool: ToolDefinition) -> ConsentType:
        """Get consent type for a tool."""
        category_mapping = {
            ToolCategory.TERMINAL: ConsentType.COMMAND_EXECUTION,
            ToolCategory.FILE_SYSTEM: ConsentType.FILE_MODIFICATION,
            ToolCategory.GIT: ConsentType.GIT_OPERATION,
            ToolCategory.BACKEND: ConsentType.BACKEND_MODIFICATION,
            ToolCategory.SYSTEM: ConsentType.SYSTEM_CHANGE,
        }
        return category_mapping.get(tool.category, ConsentType.COMMAND_EXECUTION)
    
    # ========================================================================
    # TOOL HANDLERS
    # ========================================================================
    
    def _handle_execute_command(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """Handle execute_command tool."""
        # Check for blocked commands
        if self.safety_manager.is_blocked(command):
            return ToolResult(
                tool_name="execute_command",
                success=False,
                result=None,
                error="Command blocked for safety reasons",
            )
        
        session_id = self._current_session.id if self._current_session else None
        work_dir = working_dir or str(self.project_root)
        
        # Register terminal for monitoring
        term_id = f"cmd_{int(time.time())}"
        self.terminal_monitor.register_terminal(term_id, f"Command: {command[:30]}...", work_dir)
        self.terminal_monitor.push_command_started(term_id, command, work_dir)
        
        try:
            # Execute with streaming to monitor
            def on_stdout(line: str):
                self.terminal_monitor.push_output(term_id, line, is_error=False)
            
            def on_stderr(line: str):
                self.terminal_monitor.push_output(term_id, line, is_error=True)
            
            output = self.terminal_executor.execute_streaming(
                command,
                stdout_callback=on_stdout,
                stderr_callback=on_stderr,
                working_dir=work_dir,
                timeout=timeout,
            )
            
            self.terminal_monitor.push_completed(term_id, output.return_code, output.execution_time_ms)
            
            return ToolResult(
                tool_name="execute_command",
                success=output.success,
                result={
                    "stdout": output.stdout,
                    "stderr": output.stderr,
                    "return_code": output.return_code,
                    "execution_time_ms": output.execution_time_ms,
                },
                error=output.stderr if not output.success else None,
                metadata={"terminal_id": term_id},
            )
            
        except Exception as e:
            self.terminal_monitor.push_completed(term_id, -1)
            return ToolResult(
                tool_name="execute_command",
                success=False,
                result=None,
                error=str(e),
            )
    
    def _handle_build_backend(
        self,
        backend_name: str,
        working_dir: Optional[str] = None,
        custom_commands: Optional[List[str]] = None,
    ) -> ToolResult:
        """Handle build_backend tool."""
        work_dir = working_dir or str(self.project_root)
        
        # Register terminal for monitoring
        term_id = f"build_{backend_name}_{int(time.time())}"
        self.terminal_monitor.register_terminal(term_id, f"Build: {backend_name}", work_dir)
        
        def on_output(line: str):
            self.terminal_monitor.push_output(term_id, line)
        
        self.terminal_monitor.push_command_started(term_id, f"Build {backend_name}", work_dir)
        
        try:
            results = self.terminal_executor.build_backend(
                backend_name,
                working_dir=work_dir,
                stdout_callback=on_output,
                custom_commands=custom_commands,
            )
            
            success = all(r.success for r in results)
            self.terminal_monitor.push_completed(term_id, 0 if success else 1)
            
            return ToolResult(
                tool_name="build_backend",
                success=success,
                result={
                    "backend": backend_name,
                    "steps": len(results),
                    "successful_steps": sum(1 for r in results if r.success),
                    "outputs": [r.to_dict() for r in results],
                },
                error=None if success else "Build failed",
                metadata={"terminal_id": term_id},
            )
            
        except Exception as e:
            self.terminal_monitor.push_completed(term_id, -1)
            return ToolResult(
                tool_name="build_backend",
                success=False,
                result=None,
                error=str(e),
            )
    
    def _handle_run_script(
        self,
        script_path: str,
        arguments: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
    ) -> ToolResult:
        """Handle run_script tool."""
        args_str = " ".join(arguments) if arguments else ""
        
        # Determine how to run the script
        path = Path(script_path)
        if path.suffix == ".py":
            command = f"python {script_path} {args_str}"
        elif path.suffix in (".sh", ".bash"):
            command = f"bash {script_path} {args_str}"
        elif path.suffix in (".ps1",):
            command = f"powershell -File {script_path} {args_str}"
        elif path.suffix in (".bat", ".cmd"):
            command = f"{script_path} {args_str}"
        else:
            command = f"{script_path} {args_str}"
        
        return self._handle_execute_command(command, working_dir)
    
    def _handle_read_file(
        self,
        path: str,
        max_lines: Optional[int] = None,
    ) -> ToolResult:
        """Handle read_file tool."""
        content, success = self.terminal_executor.read_file(path)
        
        if not success:
            return ToolResult(
                tool_name="read_file",
                success=False,
                result=None,
                error=content,
            )
        
        if max_lines:
            lines = content.split("\n")
            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                content += f"\n... ({len(lines) - max_lines} more lines)"
        
        return ToolResult(
            tool_name="read_file",
            success=True,
            result={
                "path": path,
                "content": content,
                "lines": len(content.split("\n")),
            },
        )
    
    def _handle_write_file(
        self,
        path: str,
        content: str,
        append: bool = False,
    ) -> ToolResult:
        """Handle write_file tool."""
        if append:
            existing, _ = self.terminal_executor.read_file(path)
            content = existing + content
        
        # Create checkpoint
        checkpoint = self.safety_manager.create_checkpoint(
            operation=f"write_file:{path}",
            files=[path] if os.path.exists(path) else None,
        )
        
        message, success = self.terminal_executor.write_file(path, content)
        
        return ToolResult(
            tool_name="write_file",
            success=success,
            result={
                "path": path,
                "bytes_written": len(content),
                "checkpoint_id": checkpoint.id,
            } if success else None,
            error=message if not success else None,
        )
    
    def _handle_list_directory(
        self,
        path: str,
        recursive: bool = False,
        max_depth: int = 2,
    ) -> ToolResult:
        """Handle list_directory tool."""
        entries = self.terminal_executor.list_directory(path)
        
        if recursive and max_depth > 0:
            # Add recursive listing
            all_entries = []
            def add_entries(dir_path: str, depth: int):
                if depth > max_depth:
                    return
                items = self.terminal_executor.list_directory(dir_path)
                for item in items:
                    item["depth"] = depth
                    all_entries.append(item)
                    if item.get("is_dir") and depth < max_depth:
                        add_entries(item["path"], depth + 1)
            
            add_entries(path, 0)
            entries = all_entries
        
        return ToolResult(
            tool_name="list_directory",
            success=True,
            result={
                "path": path,
                "count": len(entries),
                "entries": entries,
            },
        )
    
    def _handle_navigate_to(self, path: str) -> ToolResult:
        """Handle navigate_to tool."""
        session_id = self._current_session.id if self._current_session else None
        success = self.terminal_executor.navigate_to(path, session_id)
        
        return ToolResult(
            tool_name="navigate_to",
            success=success,
            result={"path": path, "current_dir": self.terminal_executor.get_current_dir(session_id)},
            error=None if success else f"Directory not found: {path}",
        )
    
    def _handle_find_files(
        self,
        pattern: str,
        path: Optional[str] = None,
        recursive: bool = True,
    ) -> ToolResult:
        """Handle find_files tool."""
        import fnmatch
        
        search_path = Path(path) if path else self.project_root
        matches = []
        
        try:
            if recursive:
                for file_path in search_path.rglob("*"):
                    if fnmatch.fnmatch(file_path.name, pattern):
                        matches.append(str(file_path))
            else:
                for file_path in search_path.glob(pattern):
                    matches.append(str(file_path))
            
            return ToolResult(
                tool_name="find_files",
                success=True,
                result={
                    "pattern": pattern,
                    "search_path": str(search_path),
                    "count": len(matches),
                    "files": matches[:100],  # Limit results
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name="find_files",
                success=False,
                result=None,
                error=str(e),
            )
    
    def _handle_git_clone(
        self,
        url: str,
        destination: Optional[str] = None,
        branch: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> ToolResult:
        """Handle git_clone tool."""
        result = self.git_ops.clone(url, destination, branch, depth)
        
        return ToolResult(
            tool_name="git_clone",
            success=result.success,
            result=result.data,
            error=result.message if not result.success else None,
        )
    
    def _handle_git_pull(
        self,
        path: str,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> ToolResult:
        """Handle git_pull tool."""
        result = self.git_ops.pull(path, remote, branch)
        
        return ToolResult(
            tool_name="git_pull",
            success=result.success,
            result=result.data,
            error=result.message if not result.success else None,
        )
    
    def _handle_git_push(
        self,
        path: str,
        remote: str = "origin",
        branch: Optional[str] = None,
    ) -> ToolResult:
        """Handle git_push tool."""
        result = self.git_ops.push(path, remote, branch)
        
        return ToolResult(
            tool_name="git_push",
            success=result.success,
            result=result.data,
            error=result.message if not result.success else None,
        )
    
    def _handle_git_status(self, path: str) -> ToolResult:
        """Handle git_status tool."""
        result = self.git_ops.status(path)
        
        return ToolResult(
            tool_name="git_status",
            success=result.success,
            result=result.data,
            error=result.message if not result.success else None,
        )
    
    def _handle_git_commit(
        self,
        path: str,
        message: str,
        add_all: bool = False,
    ) -> ToolResult:
        """Handle git_commit tool."""
        result = self.git_ops.commit(path, message, add_all)
        
        return ToolResult(
            tool_name="git_commit",
            success=result.success,
            result=result.data,
            error=result.message if not result.success else None,
        )
    
    def _handle_modify_backend_code(
        self,
        backend_name: str,
        file_path: str,
        modification_type: str,
        old_content: Optional[str] = None,
        new_content: Optional[str] = None,
        line_number: Optional[int] = None,
        create_backup: bool = True,
    ) -> ToolResult:
        """Handle modify_backend_code tool."""
        # Resolve full path
        backend_path = self.backend_modifier.get_backend_path(backend_name)
        if not backend_path:
            return ToolResult(
                tool_name="modify_backend_code",
                success=False,
                result=None,
                error=f"Backend not found: {backend_name}",
            )
        
        full_path = str(backend_path / file_path)
        
        # Preview the change
        mod_type = modification_type.lower()
        
        if mod_type == "replace":
            if not old_content or new_content is None:
                return ToolResult(
                    tool_name="modify_backend_code",
                    success=False,
                    result=None,
                    error="Replace requires old_content and new_content",
                )
            preview = self.backend_modifier.preview_replace(full_path, old_content, new_content)
        
        elif mod_type == "insert":
            if not new_content or not line_number:
                return ToolResult(
                    tool_name="modify_backend_code",
                    success=False,
                    result=None,
                    error="Insert requires new_content and line_number",
                )
            preview = self.backend_modifier.preview_insert(full_path, new_content, line_number)
        
        elif mod_type == "delete":
            if not line_number:
                return ToolResult(
                    tool_name="modify_backend_code",
                    success=False,
                    result=None,
                    error="Delete requires line_number",
                )
            preview = self.backend_modifier.preview_delete(full_path, line_number)
        
        elif mod_type == "append":
            if not new_content:
                return ToolResult(
                    tool_name="modify_backend_code",
                    success=False,
                    result=None,
                    error="Append requires new_content",
                )
            preview = self.backend_modifier.preview_append(full_path, new_content)
        
        else:
            return ToolResult(
                tool_name="modify_backend_code",
                success=False,
                result=None,
                error=f"Unknown modification type: {modification_type}",
            )
        
        if not preview.success:
            return ToolResult(
                tool_name="modify_backend_code",
                success=False,
                result=None,
                error=preview.error,
            )
        
        # Apply the change
        result = self.backend_modifier.apply_change(preview.change)
        
        return ToolResult(
            tool_name="modify_backend_code",
            success=result.success,
            result={
                "change_id": result.change.id if result.change else None,
                "diff": result.diff[:500] if result.diff else None,
                "message": result.message,
            } if result.success else None,
            error=result.error,
        )
    
    def _handle_rollback_modification(
        self,
        modification_id: str,
    ) -> ToolResult:
        """Handle rollback_modification tool."""
        result = self.backend_modifier.rollback_change(modification_id)
        
        return ToolResult(
            tool_name="rollback_modification",
            success=result.success,
            result={"message": result.message},
            error=result.error,
        )
    
    def _handle_get_system_info(self) -> ToolResult:
        """Handle get_system_info tool."""
        import sys
        
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "cwd": os.getcwd(),
            "user": os.getenv("USER") or os.getenv("USERNAME"),
            "home": str(Path.home()),
        }
        
        # Add memory info if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024**3), 2)
            info["memory_available_gb"] = round(mem.available / (1024**3), 2)
            info["memory_percent"] = mem.percent
        except ImportError:
            pass
        
        return ToolResult(
            tool_name="get_system_info",
            success=True,
            result=info,
        )
    
    def _handle_check_admin(self) -> ToolResult:
        """Handle check_admin tool."""
        is_admin = self.terminal_executor.check_admin_privileges()
        
        return ToolResult(
            tool_name="check_admin",
            success=True,
            result={
                "is_admin": is_admin,
                "platform": platform.system(),
            },
        )
    
    def _handle_request_admin(self, command: str) -> ToolResult:
        """Handle request_admin tool."""
        output = self.terminal_executor.request_admin_execution(command)
        
        return ToolResult(
            tool_name="request_admin",
            success=output.success,
            result=output.to_dict() if output.success else None,
            error=output.stderr if not output.success else None,
        )
    
    def _handle_create_plan(
        self,
        task_description: str,
        steps: List[Dict[str, Any]],
    ) -> ToolResult:
        """Handle create_plan tool."""
        self._plan_counter += 1
        plan_id = f"plan_{int(time.time())}_{self._plan_counter}"
        
        plan = AgentPlan(
            id=plan_id,
            name=f"Plan {self._plan_counter}",
            description=task_description,
            steps=steps,
        )
        
        self._plans[plan_id] = plan
        
        return ToolResult(
            tool_name="create_plan",
            success=True,
            result={
                "plan_id": plan_id,
                "steps": len(steps),
                "description": task_description,
            },
        )
    
    def _handle_execute_plan(
        self,
        plan_id: str,
        stop_on_error: bool = True,
    ) -> ToolResult:
        """Handle execute_plan tool."""
        plan = self._plans.get(plan_id)
        if not plan:
            return ToolResult(
                tool_name="execute_plan",
                success=False,
                result=None,
                error=f"Plan not found: {plan_id}",
            )
        
        plan.status = "executing"
        
        for i, step in enumerate(plan.steps):
            plan.current_step = i
            
            tool_name = step.get("tool")
            arguments = step.get("arguments", {})
            
            if not tool_name:
                continue
            
            result = self.execute_tool(tool_name, arguments)
            plan.results.append(result)
            
            if not result.success and stop_on_error:
                plan.status = "failed"
                return ToolResult(
                    tool_name="execute_plan",
                    success=False,
                    result=plan.to_dict(),
                    error=f"Step {i+1} failed: {result.error}",
                )
        
        plan.status = "completed"
        
        return ToolResult(
            tool_name="execute_plan",
            success=True,
            result=plan.to_dict(),
        )
    
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    
    def execute_command(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """Convenience method to execute a command.
        
        Args:
            command: Command to execute
            working_dir: Working directory
            timeout: Command timeout
            
        Returns:
            Tool result
        """
        return self.execute_tool("execute_command", {
            "command": command,
            "working_dir": working_dir,
            "timeout": timeout,
        })
    
    def build_backend(
        self,
        backend_name: str,
        working_dir: Optional[str] = None,
    ) -> ToolResult:
        """Convenience method to build a backend.
        
        Args:
            backend_name: Backend to build
            working_dir: Working directory
            
        Returns:
            Tool result
        """
        return self.execute_tool("build_backend", {
            "backend_name": backend_name,
            "working_dir": working_dir,
        })
    
    def get_terminal_status(self) -> Dict[str, Any]:
        """Get status of all monitored terminals.
        
        Returns:
            Terminal status summary
        """
        return self.terminal_monitor.get_status_summary()
    
    def get_terminal_output(
        self,
        terminal_id: str,
        lines: int = 50,
    ) -> List[str]:
        """Get output from a specific terminal.
        
        Args:
            terminal_id: Terminal ID
            lines: Number of lines
            
        Returns:
            Output lines
        """
        terminal = self.terminal_monitor.get_terminal(terminal_id)
        if terminal:
            return terminal.get_recent_output(lines)
        return []
    
    def undo(self) -> Tuple[bool, str]:
        """Undo the last modification."""
        return self.safety_manager.undo()
    
    def redo(self) -> Tuple[bool, str]:
        """Redo the last undone modification."""
        return self.safety_manager.redo()
    
    def get_pending_consents(self) -> List[ConsentRequest]:
        """Get pending consent requests."""
        return self.safety_manager.get_pending_consents()
    
    def respond_to_consent(
        self,
        request_id: str,
        approved: bool,
        message: Optional[str] = None,
    ) -> bool:
        """Respond to a consent request.
        
        Args:
            request_id: Request ID
            approved: Whether approved
            message: Optional message
            
        Returns:
            True if response recorded
        """
        decision = ConsentDecision.APPROVED if approved else ConsentDecision.DENIED
        return self.safety_manager.respond_to_consent(request_id, decision, message)
