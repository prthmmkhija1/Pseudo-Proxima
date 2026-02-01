"""Agent Tools for Proxima.

Defines tools that the AI agent can use for terminal operations,
file system access, and backend management. These tools are designed
to be used with LLM function calling capabilities.

Tool categories:
- Terminal: Execute commands, build backends, run scripts
- File System: Read, write, list files and directories
- Git: Clone, pull, push, commit operations
- Backend: Build, configure, modify quantum backends
- System: Get system info, check admin privileges
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.tools")


class ToolCategory(Enum):
    """Categories of agent tools."""
    TERMINAL = "terminal"
    FILE_SYSTEM = "file_system"
    GIT = "git"
    BACKEND = "backend"
    SYSTEM = "system"
    PLANNING = "planning"


class ToolPermission(Enum):
    """Permission levels for tools."""
    READ = auto()      # Read-only operations
    WRITE = auto()     # File/config modifications
    EXECUTE = auto()   # Command execution
    ADMIN = auto()     # Administrative operations
    NETWORK = auto()   # Network/remote operations


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for LLM function calling."""
        schema: Dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Definition of an agent tool."""
    name: str
    description: str
    category: ToolCategory
    permissions: List[ToolPermission]
    parameters: List[ToolParameter] = field(default_factory=list)
    requires_consent: bool = False
    consent_message: Optional[str] = None
    dangerous: bool = False  # Mark tools that could cause damage
    
    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def to_llm_response(self) -> str:
        """Convert to a string suitable for LLM context."""
        if self.success:
            if isinstance(self.result, str):
                return self.result
            return json.dumps(self.result, indent=2, default=str)
        else:
            return f"Error: {self.error}"


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

# Terminal Tools
TOOL_EXECUTE_COMMAND = ToolDefinition(
    name="execute_command",
    description="Execute a shell command in the terminal. Use this for running scripts, installing packages, building projects, etc.",
    category=ToolCategory.TERMINAL,
    permissions=[ToolPermission.EXECUTE],
    parameters=[
        ToolParameter(
            name="command",
            type="string",
            description="The command to execute (e.g., 'pip install qiskit', 'python build.py')",
        ),
        ToolParameter(
            name="working_dir",
            type="string",
            description="Working directory for the command (optional)",
            required=False,
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Timeout in seconds (default: 300)",
            required=False,
            default=300,
        ),
    ],
    requires_consent=True,
    consent_message="This will execute a command in your terminal.",
)

TOOL_BUILD_BACKEND = ToolDefinition(
    name="build_backend",
    description="Build and compile a quantum computing backend (LRET, Cirq, Qiskit Aer, cuQuantum, etc.)",
    category=ToolCategory.BACKEND,
    permissions=[ToolPermission.EXECUTE],
    parameters=[
        ToolParameter(
            name="backend_name",
            type="string",
            description="Name of the backend to build",
            enum=[
                "lret_cirq_scalability",
                "lret_pennylane_hybrid", 
                "lret_phase_7_unified",
                "cirq",
                "qiskit_aer",
                "quest",
                "qsim",
                "cuquantum",
            ],
        ),
        ToolParameter(
            name="working_dir",
            type="string",
            description="Working directory containing the backend source",
            required=False,
        ),
        ToolParameter(
            name="custom_commands",
            type="array",
            description="Custom build commands (overrides default)",
            required=False,
        ),
    ],
    requires_consent=True,
    consent_message="This will build/compile a quantum backend, which may install packages and modify your system.",
)

TOOL_RUN_SCRIPT = ToolDefinition(
    name="run_script",
    description="Execute a script file (Python, shell, batch)",
    category=ToolCategory.TERMINAL,
    permissions=[ToolPermission.EXECUTE],
    parameters=[
        ToolParameter(
            name="script_path",
            type="string",
            description="Path to the script file",
        ),
        ToolParameter(
            name="arguments",
            type="array",
            description="Arguments to pass to the script",
            required=False,
        ),
        ToolParameter(
            name="working_dir",
            type="string",
            description="Working directory for script execution",
            required=False,
        ),
    ],
    requires_consent=True,
    consent_message="This will execute a script on your system.",
)

# File System Tools
TOOL_READ_FILE = ToolDefinition(
    name="read_file",
    description="Read the contents of a file",
    category=ToolCategory.FILE_SYSTEM,
    permissions=[ToolPermission.READ],
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read",
        ),
        ToolParameter(
            name="max_lines",
            type="number",
            description="Maximum number of lines to read (optional)",
            required=False,
        ),
    ],
)

TOOL_WRITE_FILE = ToolDefinition(
    name="write_file",
    description="Write content to a file (creates if doesn't exist)",
    category=ToolCategory.FILE_SYSTEM,
    permissions=[ToolPermission.WRITE],
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to write",
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file",
        ),
        ToolParameter(
            name="append",
            type="boolean",
            description="Append to file instead of overwriting",
            required=False,
            default=False,
        ),
    ],
    requires_consent=True,
    consent_message="This will modify a file on your system.",
)

TOOL_LIST_DIRECTORY = ToolDefinition(
    name="list_directory",
    description="List contents of a directory",
    category=ToolCategory.FILE_SYSTEM,
    permissions=[ToolPermission.READ],
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the directory",
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="List recursively",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="max_depth",
            type="number",
            description="Maximum depth for recursive listing",
            required=False,
            default=2,
        ),
    ],
)

TOOL_NAVIGATE_TO = ToolDefinition(
    name="navigate_to",
    description="Change the current working directory",
    category=ToolCategory.FILE_SYSTEM,
    permissions=[ToolPermission.READ],
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to navigate to",
        ),
    ],
)

TOOL_FIND_FILES = ToolDefinition(
    name="find_files",
    description="Search for files matching a pattern",
    category=ToolCategory.FILE_SYSTEM,
    permissions=[ToolPermission.READ],
    parameters=[
        ToolParameter(
            name="pattern",
            type="string",
            description="Glob pattern to match (e.g., '*.py', 'test_*.py')",
        ),
        ToolParameter(
            name="path",
            type="string",
            description="Directory to search in",
            required=False,
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="Search recursively",
            required=False,
            default=True,
        ),
    ],
)

# Git Tools
TOOL_GIT_CLONE = ToolDefinition(
    name="git_clone",
    description="Clone a git repository",
    category=ToolCategory.GIT,
    permissions=[ToolPermission.NETWORK, ToolPermission.WRITE],
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="Repository URL to clone",
        ),
        ToolParameter(
            name="destination",
            type="string",
            description="Destination directory",
            required=False,
        ),
        ToolParameter(
            name="branch",
            type="string",
            description="Branch to checkout",
            required=False,
        ),
        ToolParameter(
            name="depth",
            type="number",
            description="Shallow clone depth (optional)",
            required=False,
        ),
    ],
    requires_consent=True,
    consent_message="This will clone a git repository to your system.",
)

TOOL_GIT_PULL = ToolDefinition(
    name="git_pull",
    description="Pull latest changes from remote repository",
    category=ToolCategory.GIT,
    permissions=[ToolPermission.NETWORK, ToolPermission.WRITE],
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the git repository",
        ),
        ToolParameter(
            name="remote",
            type="string",
            description="Remote name (default: origin)",
            required=False,
            default="origin",
        ),
        ToolParameter(
            name="branch",
            type="string",
            description="Branch to pull (default: current)",
            required=False,
        ),
    ],
    requires_consent=True,
    consent_message="This will pull changes from a remote repository.",
)

TOOL_GIT_PUSH = ToolDefinition(
    name="git_push",
    description="Push local commits to remote repository",
    category=ToolCategory.GIT,
    permissions=[ToolPermission.NETWORK],
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the git repository",
        ),
        ToolParameter(
            name="remote",
            type="string",
            description="Remote name (default: origin)",
            required=False,
            default="origin",
        ),
        ToolParameter(
            name="branch",
            type="string",
            description="Branch to push (default: current)",
            required=False,
        ),
    ],
    requires_consent=True,
    consent_message="This will push commits to a remote repository.",
    dangerous=True,
)

TOOL_GIT_STATUS = ToolDefinition(
    name="git_status",
    description="Get the status of a git repository",
    category=ToolCategory.GIT,
    permissions=[ToolPermission.READ],
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the git repository",
        ),
    ],
)

TOOL_GIT_COMMIT = ToolDefinition(
    name="git_commit",
    description="Commit staged changes",
    category=ToolCategory.GIT,
    permissions=[ToolPermission.WRITE],
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the git repository",
        ),
        ToolParameter(
            name="message",
            type="string",
            description="Commit message",
        ),
        ToolParameter(
            name="add_all",
            type="boolean",
            description="Stage all changes before committing",
            required=False,
            default=False,
        ),
    ],
    requires_consent=True,
    consent_message="This will create a git commit.",
)

# Backend Modification Tools
TOOL_MODIFY_BACKEND_CODE = ToolDefinition(
    name="modify_backend_code",
    description="Modify source code of a quantum backend with safety features (undo/redo/rollback)",
    category=ToolCategory.BACKEND,
    permissions=[ToolPermission.WRITE],
    parameters=[
        ToolParameter(
            name="backend_name",
            type="string",
            description="Name of the backend to modify",
        ),
        ToolParameter(
            name="file_path",
            type="string",
            description="Relative path to the file within the backend",
        ),
        ToolParameter(
            name="modification_type",
            type="string",
            description="Type of modification",
            enum=["replace", "insert", "delete", "append"],
        ),
        ToolParameter(
            name="old_content",
            type="string",
            description="Content to find/replace (for replace/delete)",
            required=False,
        ),
        ToolParameter(
            name="new_content",
            type="string",
            description="New content to insert/replace with",
            required=False,
        ),
        ToolParameter(
            name="line_number",
            type="number",
            description="Line number for insert operations",
            required=False,
        ),
        ToolParameter(
            name="create_backup",
            type="boolean",
            description="Create backup before modification",
            required=False,
            default=True,
        ),
    ],
    requires_consent=True,
    consent_message="This will modify backend source code. A backup will be created.",
    dangerous=True,
)

TOOL_ROLLBACK_MODIFICATION = ToolDefinition(
    name="rollback_modification",
    description="Rollback a previous code modification",
    category=ToolCategory.BACKEND,
    permissions=[ToolPermission.WRITE],
    parameters=[
        ToolParameter(
            name="modification_id",
            type="string",
            description="ID of the modification to rollback",
        ),
    ],
    requires_consent=True,
    consent_message="This will restore the previous version of the file.",
)

# System Tools
TOOL_GET_SYSTEM_INFO = ToolDefinition(
    name="get_system_info",
    description="Get system information (OS, Python version, available resources)",
    category=ToolCategory.SYSTEM,
    permissions=[ToolPermission.READ],
    parameters=[],
)

TOOL_CHECK_ADMIN = ToolDefinition(
    name="check_admin",
    description="Check if running with administrator/root privileges",
    category=ToolCategory.SYSTEM,
    permissions=[ToolPermission.READ],
    parameters=[],
)

TOOL_REQUEST_ADMIN = ToolDefinition(
    name="request_admin",
    description="Request elevated privileges for a command",
    category=ToolCategory.SYSTEM,
    permissions=[ToolPermission.ADMIN],
    parameters=[
        ToolParameter(
            name="command",
            type="string",
            description="Command to execute with admin privileges",
        ),
    ],
    requires_consent=True,
    consent_message="This will request administrator/root privileges.",
    dangerous=True,
)

# Planning Tools
TOOL_CREATE_PLAN = ToolDefinition(
    name="create_plan",
    description="Create an execution plan for a multi-step task",
    category=ToolCategory.PLANNING,
    permissions=[ToolPermission.READ],
    parameters=[
        ToolParameter(
            name="task_description",
            type="string",
            description="Natural language description of the task",
        ),
        ToolParameter(
            name="steps",
            type="array",
            description="List of planned steps",
        ),
    ],
)

TOOL_EXECUTE_PLAN = ToolDefinition(
    name="execute_plan",
    description="Execute a previously created plan",
    category=ToolCategory.PLANNING,
    permissions=[ToolPermission.EXECUTE],
    parameters=[
        ToolParameter(
            name="plan_id",
            type="string",
            description="ID of the plan to execute",
        ),
        ToolParameter(
            name="stop_on_error",
            type="boolean",
            description="Stop execution if a step fails",
            required=False,
            default=True,
        ),
    ],
    requires_consent=True,
    consent_message="This will execute a multi-step plan.",
)


# ============================================================================
# AGENT TOOLS CLASS
# ============================================================================

class AgentTools:
    """Collection of tools available to the AI agent.
    
    Provides tool definitions and execution capabilities for the agent.
    Tools can be enabled/disabled and filtered by category or permission level.
    
    Example:
        >>> tools = AgentTools()
        >>> definitions = tools.get_tool_definitions()
        >>> result = tools.execute("read_file", {"path": "/path/to/file.txt"})
    """
    
    # All available tools
    ALL_TOOLS: List[ToolDefinition] = [
        # Terminal
        TOOL_EXECUTE_COMMAND,
        TOOL_BUILD_BACKEND,
        TOOL_RUN_SCRIPT,
        # File System
        TOOL_READ_FILE,
        TOOL_WRITE_FILE,
        TOOL_LIST_DIRECTORY,
        TOOL_NAVIGATE_TO,
        TOOL_FIND_FILES,
        # Git
        TOOL_GIT_CLONE,
        TOOL_GIT_PULL,
        TOOL_GIT_PUSH,
        TOOL_GIT_STATUS,
        TOOL_GIT_COMMIT,
        # Backend
        TOOL_MODIFY_BACKEND_CODE,
        TOOL_ROLLBACK_MODIFICATION,
        # System
        TOOL_GET_SYSTEM_INFO,
        TOOL_CHECK_ADMIN,
        TOOL_REQUEST_ADMIN,
        # Planning
        TOOL_CREATE_PLAN,
        TOOL_EXECUTE_PLAN,
    ]
    
    def __init__(
        self,
        enabled_categories: Optional[List[ToolCategory]] = None,
        disabled_tools: Optional[List[str]] = None,
        max_permission_level: ToolPermission = ToolPermission.ADMIN,
    ):
        """Initialize agent tools.
        
        Args:
            enabled_categories: Categories to enable (all if None)
            disabled_tools: Tool names to disable
            max_permission_level: Maximum permission level allowed
        """
        self.enabled_categories = enabled_categories
        self.disabled_tools = set(disabled_tools or [])
        self.max_permission_level = max_permission_level
        self._tool_handlers: Dict[str, Callable[..., ToolResult]] = {}
        
        # Build tool lookup
        self._tools: Dict[str, ToolDefinition] = {
            tool.name: tool for tool in self.ALL_TOOLS
        }
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        if name in self.disabled_tools:
            return None
        tool = self._tools.get(name)
        if tool and self.enabled_categories:
            if tool.category not in self.enabled_categories:
                return None
        return tool
    
    def get_tool_definitions(
        self,
        category: Optional[ToolCategory] = None,
        include_dangerous: bool = False,
    ) -> List[ToolDefinition]:
        """Get tool definitions filtered by category.
        
        Args:
            category: Filter by category (all if None)
            include_dangerous: Include dangerous tools
            
        Returns:
            List of tool definitions
        """
        tools = []
        for tool in self.ALL_TOOLS:
            if tool.name in self.disabled_tools:
                continue
            if self.enabled_categories and tool.category not in self.enabled_categories:
                continue
            if category and tool.category != category:
                continue
            if not include_dangerous and tool.dangerous:
                continue
            tools.append(tool)
        return tools
    
    def get_openai_tools(
        self,
        category: Optional[ToolCategory] = None,
        include_dangerous: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get tool definitions in OpenAI function calling format.
        
        Args:
            category: Filter by category
            include_dangerous: Include dangerous tools
            
        Returns:
            List of OpenAI-formatted tool definitions
        """
        tools = self.get_tool_definitions(category, include_dangerous)
        return [tool.to_openai_function() for tool in tools]
    
    def get_anthropic_tools(
        self,
        category: Optional[ToolCategory] = None,
        include_dangerous: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get tool definitions in Anthropic tool format.
        
        Args:
            category: Filter by category
            include_dangerous: Include dangerous tools
            
        Returns:
            List of Anthropic-formatted tool definitions
        """
        tools = self.get_tool_definitions(category, include_dangerous)
        return [tool.to_anthropic_tool() for tool in tools]
    
    def register_handler(
        self,
        tool_name: str,
        handler: Callable[..., ToolResult],
    ) -> None:
        """Register a handler function for a tool.
        
        Args:
            tool_name: Name of the tool
            handler: Handler function
        """
        if tool_name in self._tools:
            self._tool_handlers[tool_name] = handler
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        import time
        start_time = time.perf_counter()
        
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool not found or disabled: {tool_name}",
            )
        
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"No handler registered for tool: {tool_name}",
            )
        
        try:
            result = handler(**arguments)
            result.execution_time_ms = (time.perf_counter() - start_time) * 1000
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=elapsed,
            )
    
    def validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Validate arguments for a tool.
        
        Args:
            tool_name: Name of the tool
            arguments: Arguments to validate
            
        Returns:
            Tuple of (valid, error_message)
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return False, f"Unknown tool: {tool_name}"
        
        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                return False, f"Missing required parameter: {param.name}"
            
            # Type checking (basic)
            if param.name in arguments:
                value = arguments[param.name]
                if param.type == "string" and not isinstance(value, str):
                    return False, f"Parameter {param.name} must be a string"
                elif param.type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter {param.name} must be a number"
                elif param.type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be a boolean"
                elif param.type == "array" and not isinstance(value, list):
                    return False, f"Parameter {param.name} must be an array"
                elif param.type == "object" and not isinstance(value, dict):
                    return False, f"Parameter {param.name} must be an object"
                
                # Enum validation
                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of: {param.enum}"
        
        return True, None
    
    def get_tool_help(self, tool_name: str) -> str:
        """Get help text for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Help text
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return f"Unknown tool: {tool_name}"
        
        lines = [
            f"**{tool.name}**",
            f"Category: {tool.category.value}",
            "",
            tool.description,
            "",
            "Parameters:",
        ]
        
        for param in tool.parameters:
            req = " (required)" if param.required else " (optional)"
            lines.append(f"  - {param.name} ({param.type}){req}: {param.description}")
            if param.enum:
                lines.append(f"    Allowed values: {', '.join(param.enum)}")
            if param.default is not None:
                lines.append(f"    Default: {param.default}")
        
        if tool.requires_consent:
            lines.append("")
            lines.append(f"⚠️ Requires consent: {tool.consent_message}")
        
        if tool.dangerous:
            lines.append("")
            lines.append("⚠️ This tool is marked as potentially dangerous.")
        
        return "\n".join(lines)
    
    def get_all_help(self) -> str:
        """Get help text for all enabled tools."""
        lines = ["# Available Agent Tools", ""]
        
        for category in ToolCategory:
            tools = self.get_tool_definitions(category=category, include_dangerous=True)
            if not tools:
                continue
            
            lines.append(f"## {category.value.replace('_', ' ').title()}")
            lines.append("")
            
            for tool in tools:
                lines.append(f"### {tool.name}")
                lines.append(tool.description)
                lines.append("")
        
        return "\n".join(lines)
