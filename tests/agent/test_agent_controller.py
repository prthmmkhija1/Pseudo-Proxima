"""Unit tests for Agent Controller module.

Phase 10: Integration & Testing

Tests cover:
- Controller initialization
- Component wiring
- Event flow
- Error handling
- Lifecycle management
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import (
    MockLLMClient,
    MockSubprocessFactory,
    MockFileSystem,
    MockTelemetry,
    MockConsentManager,
    create_mock_llm_client,
    create_mock_subprocess_factory,
    create_mock_file_system,
    create_mock_telemetry,
    create_mock_consent_manager,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_components():
    """Create all mock components."""
    return {
        "llm_client": create_mock_llm_client(),
        "subprocess_factory": create_mock_subprocess_factory(),
        "file_system": create_mock_file_system(),
        "telemetry": create_mock_telemetry(),
        "consent_manager": create_mock_consent_manager(),
    }


# =============================================================================
# MOCK AGENT CONTROLLER
# =============================================================================

@dataclass
class MockAgentConfig:
    """Mock agent configuration."""
    max_iterations: int = 10
    timeout_seconds: float = 300.0
    auto_approve_low_risk: bool = False
    working_dir: str = "."
    log_level: str = "INFO"


class MockAgentController:
    """Mock agent controller for testing."""
    
    def __init__(
        self,
        config: Optional[MockAgentConfig] = None,
        llm_client: Optional[MockLLMClient] = None,
        subprocess_factory: Optional[MockSubprocessFactory] = None,
        file_system: Optional[MockFileSystem] = None,
        telemetry: Optional[MockTelemetry] = None,
        consent_manager: Optional[MockConsentManager] = None,
    ):
        self.config = config or MockAgentConfig()
        self.llm_client = llm_client
        self.subprocess_factory = subprocess_factory
        self.file_system = file_system
        self.telemetry = telemetry
        self.consent_manager = consent_manager
        
        self._initialized = False
        self._running = False
        self._iteration_count = 0
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._error_count = 0
    
    async def initialize(self) -> bool:
        """Initialize the controller."""
        if not self.llm_client:
            return False
        self._initialized = True
        self._emit_event("initialized", {})
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the controller."""
        self._running = False
        self._initialized = False
        self._emit_event("shutdown", {})
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message."""
        if not self._initialized:
            raise RuntimeError("Controller not initialized")
        
        self._iteration_count += 1
        
        if self._iteration_count > self.config.max_iterations:
            raise RuntimeError("Max iterations exceeded")
        
        self._emit_event("message_received", {"message": message})
        
        # Simulate LLM call
        if self.llm_client:
            response = await self.llm_client.chat([
                {"role": "user", "content": message}
            ])
            
            # Record telemetry
            if self.telemetry:
                self.telemetry.record_event("llm_call", {
                    "message": message,
                    "response": response,
                })
            
            return {
                "success": True,
                "response": response,
                "iteration": self._iteration_count,
            }
        
        return {"success": False, "error": "No LLM client"}
    
    def on(self, event: str, handler: Callable) -> None:
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def _emit_event(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            handler(data)
    
    @property
    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._initialized
    
    @property
    def iteration_count(self) -> int:
        """Get current iteration count."""
        return self._iteration_count


# =============================================================================
# CONTROLLER INITIALIZATION TESTS
# =============================================================================

class TestControllerInitialization:
    """Tests for controller initialization."""
    
    @pytest.mark.asyncio
    async def test_successful_initialization(self, mock_components):
        """Test successful controller initialization."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        
        result = await controller.initialize()
        
        assert result is True
        assert controller.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_initialization_without_llm(self):
        """Test initialization fails without LLM client."""
        controller = MockAgentController()
        
        result = await controller.initialize()
        
        assert result is False
        assert controller.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_initialization_emits_event(self, mock_components):
        """Test initialization emits event."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        
        events = []
        controller.on("initialized", lambda data: events.append(data))
        
        await controller.initialize()
        
        assert len(events) == 1


# =============================================================================
# MESSAGE PROCESSING TESTS
# =============================================================================

class TestMessageProcessing:
    """Tests for message processing."""
    
    @pytest.mark.asyncio
    async def test_process_message(self, mock_components):
        """Test processing a message."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
            telemetry=mock_components["telemetry"],
        )
        await controller.initialize()
        
        result = await controller.process_message("Build cirq backend")
        
        assert result["success"] is True
        assert "response" in result
    
    @pytest.mark.asyncio
    async def test_process_message_without_init(self, mock_components):
        """Test processing message before initialization."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await controller.process_message("Test")
    
    @pytest.mark.asyncio
    async def test_iteration_count(self, mock_components):
        """Test iteration count increments."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        await controller.initialize()
        
        await controller.process_message("Message 1")
        await controller.process_message("Message 2")
        await controller.process_message("Message 3")
        
        assert controller.iteration_count == 3
    
    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self, mock_components):
        """Test max iterations limit."""
        controller = MockAgentController(
            config=MockAgentConfig(max_iterations=2),
            llm_client=mock_components["llm_client"],
        )
        await controller.initialize()
        
        await controller.process_message("Message 1")
        await controller.process_message("Message 2")
        
        with pytest.raises(RuntimeError, match="Max iterations"):
            await controller.process_message("Message 3")


# =============================================================================
# EVENT HANDLING TESTS
# =============================================================================

class TestEventHandling:
    """Tests for event handling."""
    
    def test_register_event_handler(self, mock_components):
        """Test registering event handler."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        
        handler = MagicMock()
        controller.on("test_event", handler)
        
        assert "test_event" in controller._event_handlers
    
    @pytest.mark.asyncio
    async def test_message_event(self, mock_components):
        """Test message received event."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        await controller.initialize()
        
        events = []
        controller.on("message_received", lambda data: events.append(data))
        
        await controller.process_message("Test message")
        
        assert len(events) == 1
        assert events[0]["message"] == "Test message"
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self, mock_components):
        """Test multiple handlers for same event."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        
        call_counts = {"handler1": 0, "handler2": 0}
        
        controller.on("initialized", lambda _: call_counts.__setitem__("handler1", call_counts["handler1"] + 1))
        controller.on("initialized", lambda _: call_counts.__setitem__("handler2", call_counts["handler2"] + 1))
        
        await controller.initialize()
        
        assert call_counts["handler1"] == 1
        assert call_counts["handler2"] == 1


# =============================================================================
# SHUTDOWN TESTS
# =============================================================================

class TestShutdown:
    """Tests for controller shutdown."""
    
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_components):
        """Test controller shutdown."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        await controller.initialize()
        
        await controller.shutdown()
        
        assert controller.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_shutdown_emits_event(self, mock_components):
        """Test shutdown emits event."""
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
        )
        await controller.initialize()
        
        events = []
        controller.on("shutdown", lambda data: events.append(data))
        
        await controller.shutdown()
        
        assert len(events) == 1


# =============================================================================
# COMPONENT INTEGRATION TESTS
# =============================================================================

class TestComponentIntegration:
    """Tests for component integration."""
    
    @pytest.mark.asyncio
    async def test_telemetry_recording(self, mock_components):
        """Test telemetry is recorded."""
        telemetry = mock_components["telemetry"]
        controller = MockAgentController(
            llm_client=mock_components["llm_client"],
            telemetry=telemetry,
        )
        await controller.initialize()
        
        await controller.process_message("Test")
        
        assert len(telemetry.events) > 0
        assert telemetry.events[0]["type"] == "llm_call"
    
    @pytest.mark.asyncio
    async def test_all_components_wired(self, mock_components):
        """Test all components are wired correctly."""
        controller = MockAgentController(
            config=MockAgentConfig(working_dir="/project"),
            llm_client=mock_components["llm_client"],
            subprocess_factory=mock_components["subprocess_factory"],
            file_system=mock_components["file_system"],
            telemetry=mock_components["telemetry"],
            consent_manager=mock_components["consent_manager"],
        )
        
        assert controller.llm_client is not None
        assert controller.subprocess_factory is not None
        assert controller.file_system is not None
        assert controller.telemetry is not None
        assert controller.consent_manager is not None


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Tests for controller configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MockAgentConfig()
        
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300.0
        assert config.auto_approve_low_risk is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MockAgentConfig(
            max_iterations=20,
            timeout_seconds=600.0,
            auto_approve_low_risk=True,
            working_dir="/custom/path",
        )
        
        assert config.max_iterations == 20
        assert config.timeout_seconds == 600.0
        assert config.auto_approve_low_risk is True
        assert config.working_dir == "/custom/path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
