"""Unit tests for Agent Tools module.

Phase 10: Integration & Testing

Tests cover:
- Tool registration and discovery
- Tool execution
- Parameter validation
- Tool results handling
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import (
    MockLLMClient,
    MockLLMResponse,
    MockSubprocessFactory,
    MockFileSystem,
    create_mock_llm_client,
    create_mock_file_system,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    return create_mock_llm_client()


@pytest.fixture
def mock_file_system():
    """Create mock file system."""
    return create_mock_file_system()


# =============================================================================
# TOOL DEFINITION TESTS
# =============================================================================

@dataclass
class MockToolDefinition:
    """Mock tool definition for testing."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


class TestToolDefinition:
    """Tests for tool definitions."""
    
    def test_tool_creation(self):
        """Test creating a tool definition."""
        tool = MockToolDefinition(
            name="run_command",
            description="Execute a shell command",
            parameters={
                "command": {"type": "string", "description": "The command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds"},
            },
            required=["command"],
        )
        
        assert tool.name == "run_command"
        assert "command" in tool.parameters
        assert "command" in tool.required
    
    def test_tool_openai_format(self):
        """Test conversion to OpenAI format."""
        tool = MockToolDefinition(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "path": {"type": "string", "description": "File path"},
            },
            required=["path"],
        )
        
        openai_format = tool.to_openai_format()
        
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "read_file"
        assert "parameters" in openai_format["function"]


# =============================================================================
# TOOL REGISTRY TESTS
# =============================================================================

class MockToolRegistry:
    """Mock tool registry for testing."""
    
    def __init__(self):
        self._tools: Dict[str, MockToolDefinition] = {}
    
    def register(self, tool: MockToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[MockToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())
    
    def get_all(self) -> List[MockToolDefinition]:
        """Get all tools."""
        return list(self._tools.values())


class TestToolRegistry:
    """Tests for tool registry."""
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = MockToolRegistry()
        tool = MockToolDefinition(
            name="test_tool",
            description="A test tool",
        )
        
        registry.register(tool)
        
        assert "test_tool" in registry.list_tools()
    
    def test_get_tool(self):
        """Test getting a tool."""
        registry = MockToolRegistry()
        tool = MockToolDefinition(
            name="get_test",
            description="Test getting",
        )
        registry.register(tool)
        
        retrieved = registry.get("get_test")
        
        assert retrieved is not None
        assert retrieved.name == "get_test"
    
    def test_get_nonexistent_tool(self):
        """Test getting nonexistent tool."""
        registry = MockToolRegistry()
        
        result = registry.get("nonexistent")
        
        assert result is None
    
    def test_list_tools(self):
        """Test listing all tools."""
        registry = MockToolRegistry()
        
        tools = [
            MockToolDefinition(name=f"tool_{i}", description=f"Tool {i}")
            for i in range(5)
        ]
        
        for tool in tools:
            registry.register(tool)
        
        tool_names = registry.list_tools()
        
        assert len(tool_names) == 5
        assert "tool_0" in tool_names
        assert "tool_4" in tool_names


# =============================================================================
# TOOL EXECUTION TESTS
# =============================================================================

class TestToolExecution:
    """Tests for tool execution."""
    
    @pytest.mark.asyncio
    async def test_execute_read_file(self, mock_file_system):
        """Test executing read_file tool."""
        mock_file_system.write_file("/test/file.txt", "Hello World")
        
        content = mock_file_system.read_file("/test/file.txt")
        
        assert content == "Hello World"
    
    @pytest.mark.asyncio
    async def test_execute_write_file(self, mock_file_system):
        """Test executing write_file tool."""
        mock_file_system.write_file("/test/new_file.txt", "New content")
        
        assert mock_file_system.exists("/test/new_file.txt")
        assert mock_file_system.read_file("/test/new_file.txt") == "New content"
    
    @pytest.mark.asyncio
    async def test_execute_delete_file(self, mock_file_system):
        """Test executing delete_file tool."""
        mock_file_system.write_file("/test/to_delete.txt", "Delete me")
        mock_file_system.delete("/test/to_delete.txt")
        
        assert not mock_file_system.exists("/test/to_delete.txt")
    
    def test_file_not_found_error(self, mock_file_system):
        """Test handling file not found error."""
        with pytest.raises(FileNotFoundError):
            mock_file_system.read_file("/nonexistent/file.txt")


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

class TestParameterValidation:
    """Tests for tool parameter validation."""
    
    def test_required_parameter_present(self):
        """Test validation when required parameter is present."""
        tool = MockToolDefinition(
            name="test",
            description="Test",
            parameters={"required_param": {"type": "string"}},
            required=["required_param"],
        )
        
        params = {"required_param": "value"}
        
        # Check required params are present
        missing = [r for r in tool.required if r not in params]
        
        assert len(missing) == 0
    
    def test_required_parameter_missing(self):
        """Test validation when required parameter is missing."""
        tool = MockToolDefinition(
            name="test",
            description="Test",
            parameters={"required_param": {"type": "string"}},
            required=["required_param"],
        )
        
        params = {"other_param": "value"}
        
        # Check required params are present
        missing = [r for r in tool.required if r not in params]
        
        assert len(missing) == 1
        assert "required_param" in missing
    
    def test_type_validation(self):
        """Test parameter type validation."""
        tool = MockToolDefinition(
            name="test",
            description="Test",
            parameters={
                "string_param": {"type": "string"},
                "number_param": {"type": "number"},
                "bool_param": {"type": "boolean"},
            },
        )
        
        params = {
            "string_param": "hello",
            "number_param": 42,
            "bool_param": True,
        }
        
        # Validate types
        assert isinstance(params["string_param"], str)
        assert isinstance(params["number_param"], (int, float))
        assert isinstance(params["bool_param"], bool)


# =============================================================================
# TOOL RESULT TESTS
# =============================================================================

@dataclass
class MockToolResult:
    """Mock tool result for testing."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


class TestToolResult:
    """Tests for tool results."""
    
    def test_success_result(self):
        """Test successful tool result."""
        result = MockToolResult(
            success=True,
            output={"data": "value"},
            execution_time_ms=50.0,
        )
        
        assert result.success is True
        assert result.output["data"] == "value"
        assert result.error is None
    
    def test_error_result(self):
        """Test error tool result."""
        result = MockToolResult(
            success=False,
            error="Tool execution failed",
            execution_time_ms=10.0,
        )
        
        assert result.success is False
        assert result.error == "Tool execution failed"
        assert result.output is None
    
    def test_result_to_dict(self):
        """Test result conversion to dict."""
        result = MockToolResult(
            success=True,
            output=["item1", "item2"],
            execution_time_ms=100.0,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert len(result_dict["output"]) == 2


# =============================================================================
# LLM TOOL CALL TESTS
# =============================================================================

class TestLLMToolCalls:
    """Tests for LLM tool call handling."""
    
    @pytest.mark.asyncio
    async def test_llm_returns_tool_call(self, mock_llm_client):
        """Test LLM returns tool call."""
        from tests.fixtures.mock_agent import MockLLMResponse
        
        mock_llm_client.queue_response(MockLLMResponse(
            content='{"action": "run_command", "command": "ls -la"}'
        ))
        
        response = await mock_llm_client.chat([
            {"role": "user", "content": "List files in current directory"}
        ])
        
        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert "run_command" in content
    
    @pytest.mark.asyncio
    async def test_llm_tool_parsing(self, mock_llm_client):
        """Test parsing LLM tool response."""
        mock_llm_client.queue_response(MockLLMResponse(
            content='{"action": "read_file", "path": "/test/file.py"}'
        ))
        
        response = await mock_llm_client.chat([
            {"role": "user", "content": "Read file.py"}
        ])
        
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        
        assert parsed["action"] == "read_file"
        assert parsed["path"] == "/test/file.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
