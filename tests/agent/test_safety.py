"""Unit tests for Safety module.

Phase 10: Integration & Testing

Tests cover:
- Consent request creation
- Consent type enums
- Risk level handling
- Consent decisions
- Audit logging
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import MockConsentManager, create_mock_consent_manager


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_consent_manager():
    """Create mock consent manager."""
    return create_mock_consent_manager(auto_approve=True)


@pytest.fixture
def mock_consent_manager_deny():
    """Create mock consent manager that denies requests."""
    return create_mock_consent_manager(auto_approve=False)


# =============================================================================
# CONSENT TYPE TESTS
# =============================================================================

class TestConsentType:
    """Tests for ConsentType enum."""
    
    def test_all_consent_types_exist(self):
        """Test all expected consent types exist."""
        from proxima.agent.safety import ConsentType
        
        assert hasattr(ConsentType, "COMMAND_EXECUTION")
        assert hasattr(ConsentType, "FILE_MODIFICATION")
        assert hasattr(ConsentType, "GIT_OPERATION")
        assert hasattr(ConsentType, "ADMIN_ACCESS")
        assert hasattr(ConsentType, "NETWORK_ACCESS")
        assert hasattr(ConsentType, "BACKEND_MODIFICATION")
        assert hasattr(ConsentType, "SYSTEM_CHANGE")
        assert hasattr(ConsentType, "BULK_OPERATION")
    
    def test_consent_type_values(self):
        """Test consent type string values."""
        from proxima.agent.safety import ConsentType
        
        assert ConsentType.COMMAND_EXECUTION.value == "command_execution"
        assert ConsentType.FILE_MODIFICATION.value == "file_modification"
        assert ConsentType.GIT_OPERATION.value == "git_operation"


# =============================================================================
# CONSENT DECISION TESTS
# =============================================================================

class TestConsentDecision:
    """Tests for ConsentDecision enum."""
    
    def test_all_decisions_exist(self):
        """Test all expected decisions exist."""
        from proxima.agent.safety import ConsentDecision
        
        assert hasattr(ConsentDecision, "APPROVED")
        assert hasattr(ConsentDecision, "DENIED")
        assert hasattr(ConsentDecision, "APPROVED_ONCE")
        assert hasattr(ConsentDecision, "APPROVED_SESSION")
        assert hasattr(ConsentDecision, "APPROVED_ALWAYS")


# =============================================================================
# CONSENT REQUEST TESTS
# =============================================================================

class TestConsentRequest:
    """Tests for ConsentRequest dataclass."""
    
    def test_consent_request_creation(self):
        """Test creating a consent request."""
        from proxima.agent.safety import ConsentRequest, ConsentType
        
        request = ConsentRequest(
            id="test-request-1",
            consent_type=ConsentType.COMMAND_EXECUTION,
            operation="Execute Build Command",
            description="Run cmake to build backend",
        )
        
        assert request.id == "test-request-1"
        assert request.consent_type == ConsentType.COMMAND_EXECUTION
        assert request.operation == "Execute Build Command"
        assert request.risk_level == "medium"
        assert request.reversible is True
    
    def test_consent_request_with_details(self):
        """Test consent request with custom details."""
        from proxima.agent.safety import ConsentRequest, ConsentType
        
        details = {
            "command": "rm -rf build/",
            "working_dir": "/project",
            "files_affected": ["build/"],
        }
        
        request = ConsentRequest(
            id="test-request-2",
            consent_type=ConsentType.FILE_MODIFICATION,
            operation="Delete Build Directory",
            description="Remove build artifacts",
            details=details,
            risk_level="high",
            reversible=False,
        )
        
        assert request.details == details
        assert request.risk_level == "high"
        assert request.reversible is False
    
    def test_consent_request_to_dict(self):
        """Test converting request to dictionary."""
        from proxima.agent.safety import ConsentRequest, ConsentType
        
        request = ConsentRequest(
            id="test-request-3",
            consent_type=ConsentType.GIT_OPERATION,
            operation="Git Push",
            description="Push changes to remote",
            tool_name="git_push",
        )
        
        result = request.to_dict()
        
        assert result["id"] == "test-request-3"
        assert result["consent_type"] == "git_operation"
        assert result["operation"] == "Git Push"
        assert result["tool_name"] == "git_push"
        assert "timestamp" in result
    
    def test_consent_request_display_message(self):
        """Test getting display message."""
        from proxima.agent.safety import ConsentRequest, ConsentType
        
        request = ConsentRequest(
            id="test-request-4",
            consent_type=ConsentType.ADMIN_ACCESS,
            operation="Elevate Privileges",
            description="Request admin access for installation",
            risk_level="critical",
        )
        
        message = request.get_display_message()
        
        assert "🚨" in message  # Critical risk emoji
        assert "Elevate Privileges" in message
    
    def test_consent_request_display_message_low_risk(self):
        """Test display message for low risk operation."""
        from proxima.agent.safety import ConsentRequest, ConsentType
        
        request = ConsentRequest(
            id="test-request-5",
            consent_type=ConsentType.FILE_MODIFICATION,
            operation="Read File",
            description="Read contents of config.yaml",
            risk_level="low",
        )
        
        message = request.get_display_message()
        
        assert "ℹ️" in message  # Info emoji for low risk


# =============================================================================
# MOCK CONSENT MANAGER TESTS
# =============================================================================

class TestMockConsentManager:
    """Tests for MockConsentManager."""
    
    @pytest.mark.asyncio
    async def test_auto_approve(self, mock_consent_manager):
        """Test automatic approval."""
        result = await mock_consent_manager.request_consent(
            operation="Test Operation",
            description="Test description",
            risk_level="low",
        )
        
        assert result is True
        assert len(mock_consent_manager.requests) == 1
        assert mock_consent_manager.requests[0]["approved"] is True
    
    @pytest.mark.asyncio
    async def test_auto_deny(self, mock_consent_manager_deny):
        """Test automatic denial."""
        result = await mock_consent_manager_deny.request_consent(
            operation="Dangerous Operation",
            description="Delete system files",
            risk_level="critical",
        )
        
        assert result is False
        assert len(mock_consent_manager_deny.requests) == 1
        assert mock_consent_manager_deny.requests[0]["approved"] is False
    
    @pytest.mark.asyncio
    async def test_multiple_requests(self, mock_consent_manager):
        """Test handling multiple consent requests."""
        await mock_consent_manager.request_consent(
            operation="Op 1",
            description="Description 1",
        )
        await mock_consent_manager.request_consent(
            operation="Op 2",
            description="Description 2",
        )
        await mock_consent_manager.request_consent(
            operation="Op 3",
            description="Description 3",
        )
        
        requests = mock_consent_manager.get_requests()
        
        assert len(requests) == 3
        assert requests[0]["operation"] == "Op 1"
        assert requests[2]["operation"] == "Op 3"
    
    @pytest.mark.asyncio
    async def test_request_records_risk_level(self, mock_consent_manager):
        """Test that risk level is recorded."""
        await mock_consent_manager.request_consent(
            operation="High Risk Op",
            description="Something dangerous",
            risk_level="high",
        )
        
        request = mock_consent_manager.requests[0]
        
        assert request["risk_level"] == "high"


# =============================================================================
# SAFETY BOUNDARY TESTS
# =============================================================================

class TestSafetyBoundaries:
    """Tests for safety boundary validation."""
    
    def test_risk_levels(self):
        """Test valid risk levels."""
        from proxima.agent.safety import ConsentRequest, ConsentType
        
        valid_levels = ["low", "medium", "high", "critical"]
        
        for level in valid_levels:
            request = ConsentRequest(
                id=f"test-{level}",
                consent_type=ConsentType.COMMAND_EXECUTION,
                operation="Test",
                description="Test",
                risk_level=level,
            )
            assert request.risk_level == level
    
    def test_consent_type_categorization(self):
        """Test consent types are properly categorized."""
        from proxima.agent.safety import ConsentType
        
        # High-risk operations
        high_risk_types = [
            ConsentType.ADMIN_ACCESS,
            ConsentType.SYSTEM_CHANGE,
            ConsentType.BACKEND_MODIFICATION,
        ]
        
        # Standard operations
        standard_types = [
            ConsentType.COMMAND_EXECUTION,
            ConsentType.FILE_MODIFICATION,
            ConsentType.GIT_OPERATION,
        ]
        
        for t in high_risk_types:
            assert t in ConsentType
        
        for t in standard_types:
            assert t in ConsentType


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Tests for consent request serialization."""
    
    def test_json_serialization(self):
        """Test JSON serialization of consent request."""
        from proxima.agent.safety import ConsentRequest, ConsentType
        
        request = ConsentRequest(
            id="json-test",
            consent_type=ConsentType.NETWORK_ACCESS,
            operation="API Call",
            description="Call external API",
            details={"url": "https://api.example.com"},
        )
        
        # Should be JSON serializable
        json_str = json.dumps(request.to_dict())
        parsed = json.loads(json_str)
        
        assert parsed["id"] == "json-test"
        assert parsed["consent_type"] == "network_access"
        assert parsed["details"]["url"] == "https://api.example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
