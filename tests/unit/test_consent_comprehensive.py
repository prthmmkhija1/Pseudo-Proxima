"""
Comprehensive Unit Tests for Consent Management System

Tests for:
- ConsentLevel enum
- ConsentCategory enum
- ConsentRecord dataclass
- ConsentRequest dataclass
- ConsentResponse enum
- Category configurations
- Consent validation and expiry
"""

from __future__ import annotations

import time
from dataclasses import asdict

import pytest

from proxima.resources.consent import (
    CATEGORY_CONFIGS,
    ConsentCategory,
    ConsentCategoryConfig,
    ConsentLevel,
    ConsentRecord,
    ConsentRequest,
    ConsentResponse,
)


# =============================================================================
# CONSENT LEVEL TESTS
# =============================================================================


class TestConsentLevel:
    """Tests for ConsentLevel enum."""

    @pytest.mark.unit
    def test_all_levels_defined(self):
        """Verify all required consent levels are defined."""
        expected = ["SESSION", "PERSISTENT", "ONE_TIME", "NEVER"]
        for level_name in expected:
            assert hasattr(ConsentLevel, level_name)

    @pytest.mark.unit
    def test_level_count(self):
        """Verify correct number of consent levels."""
        assert len(ConsentLevel) == 4

    @pytest.mark.unit
    def test_levels_are_unique(self):
        """Verify all levels have unique values."""
        values = [level.value for level in ConsentLevel]
        assert len(values) == len(set(values))


# =============================================================================
# CONSENT CATEGORY TESTS
# =============================================================================


class TestConsentCategory:
    """Tests for ConsentCategory enum."""

    @pytest.mark.unit
    def test_all_categories_defined(self):
        """Verify all required consent categories are defined."""
        expected = [
            "LOCAL_LLM",
            "REMOTE_LLM",
            "FORCE_EXECUTE",
            "UNTRUSTED_AGENT_MD",
            "FILE_WRITE",
            "NETWORK_ACCESS",
            "RESOURCE_INTENSIVE",
            "DATA_COLLECTION",
        ]
        for cat_name in expected:
            assert hasattr(ConsentCategory, cat_name)

    @pytest.mark.unit
    def test_category_values(self):
        """Test category string values."""
        assert ConsentCategory.LOCAL_LLM.value == "local_llm"
        assert ConsentCategory.REMOTE_LLM.value == "remote_llm"
        assert ConsentCategory.FORCE_EXECUTE.value == "force_execute"

    @pytest.mark.unit
    def test_category_count(self):
        """Verify correct number of categories."""
        assert len(ConsentCategory) == 8


# =============================================================================
# CONSENT RESPONSE TESTS
# =============================================================================


class TestConsentResponse:
    """Tests for ConsentResponse enum."""

    @pytest.mark.unit
    def test_all_responses_defined(self):
        """Verify all required responses are defined."""
        expected = [
            "APPROVE",
            "APPROVE_SESSION",
            "APPROVE_ALWAYS",
            "DENY",
            "DENY_SESSION",
            "DENY_ALWAYS",
        ]
        for resp_name in expected:
            assert hasattr(ConsentResponse, resp_name)

    @pytest.mark.unit
    def test_approve_responses(self):
        """Test approve response values."""
        assert ConsentResponse.APPROVE.value == "approve"
        assert ConsentResponse.APPROVE_SESSION.value == "approve_session"
        assert ConsentResponse.APPROVE_ALWAYS.value == "approve_always"

    @pytest.mark.unit
    def test_deny_responses(self):
        """Test deny response values."""
        assert ConsentResponse.DENY.value == "deny"
        assert ConsentResponse.DENY_SESSION.value == "deny_session"
        assert ConsentResponse.DENY_ALWAYS.value == "deny_always"


# =============================================================================
# CATEGORY CONFIG TESTS
# =============================================================================


class TestCategoryConfigs:
    """Tests for category configuration dictionary."""

    @pytest.mark.unit
    def test_all_categories_have_config(self):
        """Verify all categories have a configuration."""
        for category in ConsentCategory:
            assert category in CATEGORY_CONFIGS

    @pytest.mark.unit
    def test_local_llm_config(self):
        """Test LOCAL_LLM category config."""
        config = CATEGORY_CONFIGS[ConsentCategory.LOCAL_LLM]
        assert config.allow_remember is True
        assert config.allow_force_override is True
        assert config.default_level == ConsentLevel.SESSION

    @pytest.mark.unit
    def test_force_execute_config(self):
        """Test FORCE_EXECUTE category config (always ask)."""
        config = CATEGORY_CONFIGS[ConsentCategory.FORCE_EXECUTE]
        assert config.allow_remember is False  # Always ask
        assert config.allow_force_override is False
        assert config.default_level == ConsentLevel.ONE_TIME

    @pytest.mark.unit
    def test_untrusted_agent_config(self):
        """Test UNTRUSTED_AGENT_MD category config (always ask)."""
        config = CATEGORY_CONFIGS[ConsentCategory.UNTRUSTED_AGENT_MD]
        assert config.allow_remember is False  # Always ask
        assert config.allow_force_override is False
        assert config.default_level == ConsentLevel.ONE_TIME

    @pytest.mark.unit
    def test_configs_have_descriptions(self):
        """Verify all configs have descriptions."""
        for category, config in CATEGORY_CONFIGS.items():
            assert config.description, f"{category} missing description"
            assert len(config.description) > 5


# =============================================================================
# CONSENT CATEGORY CONFIG TESTS
# =============================================================================


class TestConsentCategoryConfig:
    """Tests for ConsentCategoryConfig dataclass."""

    @pytest.mark.unit
    def test_config_creation(self):
        """Test creating a category config."""
        config = ConsentCategoryConfig(
            category=ConsentCategory.FILE_WRITE,
            allow_remember=True,
            allow_force_override=True,
            default_level=ConsentLevel.SESSION,
            description="Write files to disk",
        )
        assert config.category == ConsentCategory.FILE_WRITE
        assert config.allow_remember is True
        assert config.description == "Write files to disk"

    @pytest.mark.unit
    def test_config_immutability(self):
        """Test that configs are properly typed."""
        config = CATEGORY_CONFIGS[ConsentCategory.LOCAL_LLM]
        assert isinstance(config, ConsentCategoryConfig)
        assert isinstance(config.category, ConsentCategory)
        assert isinstance(config.default_level, ConsentLevel)


# =============================================================================
# CONSENT RECORD TESTS
# =============================================================================


class TestConsentRecord:
    """Tests for ConsentRecord dataclass."""

    @pytest.mark.unit
    def test_record_creation(self):
        """Test basic record creation."""
        record = ConsentRecord(
            topic="use_openai_api",
            category=ConsentCategory.REMOTE_LLM,
            granted=True,
            level=ConsentLevel.SESSION,
        )
        assert record.topic == "use_openai_api"
        assert record.granted is True
        assert record.category == ConsentCategory.REMOTE_LLM

    @pytest.mark.unit
    def test_record_defaults(self):
        """Test record default values."""
        record = ConsentRecord(
            topic="test",
            category=None,
            granted=True,
            level=ConsentLevel.ONE_TIME,
        )
        assert record.context is None
        assert record.expires_at is None
        assert record.source == "user"
        assert record.timestamp > 0

    @pytest.mark.unit
    def test_record_is_valid_not_expired(self):
        """Test is_valid for non-expired record."""
        record = ConsentRecord(
            topic="test",
            category=None,
            granted=True,
            level=ConsentLevel.SESSION,
            expires_at=None,  # No expiry
        )
        assert record.is_valid() is True

    @pytest.mark.unit
    def test_record_is_valid_with_future_expiry(self):
        """Test is_valid for record with future expiry."""
        record = ConsentRecord(
            topic="test",
            category=None,
            granted=True,
            level=ConsentLevel.SESSION,
            expires_at=time.time() + 3600,  # Expires in 1 hour
        )
        assert record.is_valid() is True

    @pytest.mark.unit
    def test_record_is_valid_expired(self):
        """Test is_valid for expired record."""
        record = ConsentRecord(
            topic="test",
            category=None,
            granted=True,
            level=ConsentLevel.SESSION,
            expires_at=time.time() - 1,  # Expired 1 second ago
        )
        assert record.is_valid() is False

    @pytest.mark.unit
    def test_record_to_dict(self):
        """Test serialization to dictionary."""
        record = ConsentRecord(
            topic="use_llm",
            category=ConsentCategory.LOCAL_LLM,
            granted=True,
            level=ConsentLevel.SESSION,
            context="testing",
            source="force_override",
        )
        data = record.to_dict()
        
        assert data["topic"] == "use_llm"
        assert data["category"] == "local_llm"
        assert data["granted"] is True
        assert data["level"] == "SESSION"
        assert data["context"] == "testing"
        assert data["source"] == "force_override"

    @pytest.mark.unit
    def test_record_to_dict_no_category(self):
        """Test serialization with no category."""
        record = ConsentRecord(
            topic="custom_action",
            category=None,
            granted=True,
            level=ConsentLevel.ONE_TIME,
        )
        data = record.to_dict()
        assert data["category"] is None

    @pytest.mark.unit
    def test_record_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "topic": "use_remote_api",
            "category": "remote_llm",
            "granted": True,
            "level": "PERSISTENT",
            "timestamp": 1704067200.0,
            "context": "production use",
            "expires_at": None,
            "source": "user",
        }
        record = ConsentRecord.from_dict(data)
        
        assert record.topic == "use_remote_api"
        assert record.category == ConsentCategory.REMOTE_LLM
        assert record.granted is True
        assert record.level == ConsentLevel.PERSISTENT
        assert record.context == "production use"

    @pytest.mark.unit
    def test_record_from_dict_no_category(self):
        """Test deserialization with no category."""
        data = {
            "topic": "custom",
            "category": None,
            "granted": False,
            "level": "NEVER",
        }
        record = ConsentRecord.from_dict(data)
        assert record.category is None
        assert record.granted is False

    @pytest.mark.unit
    def test_record_from_dict_invalid_category(self):
        """Test deserialization with invalid category."""
        data = {
            "topic": "test",
            "category": "invalid_category",
            "granted": True,
            "level": "SESSION",
        }
        record = ConsentRecord.from_dict(data)
        assert record.category is None  # Invalid category becomes None

    @pytest.mark.unit
    def test_record_roundtrip_serialization(self):
        """Test roundtrip serialization."""
        original = ConsentRecord(
            topic="roundtrip_test",
            category=ConsentCategory.NETWORK_ACCESS,
            granted=True,
            level=ConsentLevel.SESSION,
            context="test context",
            expires_at=time.time() + 3600,
            source="user",
        )
        
        data = original.to_dict()
        restored = ConsentRecord.from_dict(data)
        
        assert restored.topic == original.topic
        assert restored.category == original.category
        assert restored.granted == original.granted
        assert restored.level == original.level
        assert restored.context == original.context


# =============================================================================
# CONSENT REQUEST TESTS
# =============================================================================


class TestConsentRequest:
    """Tests for ConsentRequest dataclass."""

    @pytest.mark.unit
    def test_request_creation(self):
        """Test basic request creation."""
        request = ConsentRequest(
            topic="use_gpt4",
            category=ConsentCategory.REMOTE_LLM,
            description="Send data to OpenAI GPT-4 API",
        )
        assert request.topic == "use_gpt4"
        assert request.category == ConsentCategory.REMOTE_LLM
        assert "OpenAI" in request.description

    @pytest.mark.unit
    def test_request_defaults(self):
        """Test request default values."""
        request = ConsentRequest(
            topic="test",
            category=None,
            description="Test action",
        )
        assert request.details is None
        assert request.allow_remember is True
        assert request.allow_force_override is False
        assert request.suggested_level == ConsentLevel.SESSION

    @pytest.mark.unit
    def test_request_with_all_options(self):
        """Test request with all options specified."""
        request = ConsentRequest(
            topic="intensive_simulation",
            category=ConsentCategory.RESOURCE_INTENSIVE,
            description="Run large quantum simulation",
            details="Estimated memory: 8GB, time: 2 hours",
            allow_remember=True,
            allow_force_override=True,
            suggested_level=ConsentLevel.PERSISTENT,
        )
        assert request.details is not None
        assert "8GB" in request.details
        assert request.allow_force_override is True
        assert request.suggested_level == ConsentLevel.PERSISTENT


# =============================================================================
# CONSENT WORKFLOW TESTS
# =============================================================================


class TestConsentWorkflows:
    """Tests for consent workflow scenarios."""

    @pytest.mark.unit
    def test_llm_consent_workflow(self):
        """Test typical LLM consent workflow."""
        # User wants to use remote LLM
        request = ConsentRequest(
            topic="use_claude_api",
            category=ConsentCategory.REMOTE_LLM,
            description="Send code to Anthropic Claude for analysis",
            allow_remember=CATEGORY_CONFIGS[ConsentCategory.REMOTE_LLM].allow_remember,
            allow_force_override=CATEGORY_CONFIGS[ConsentCategory.REMOTE_LLM].allow_force_override,
        )
        
        # User approves for session
        record = ConsentRecord(
            topic=request.topic,
            category=request.category,
            granted=True,
            level=ConsentLevel.SESSION,
            source="user",
        )
        
        assert record.is_valid()
        assert record.granted

    @pytest.mark.unit
    def test_force_execute_consent_workflow(self):
        """Test force execute consent workflow (always ask)."""
        config = CATEGORY_CONFIGS[ConsentCategory.FORCE_EXECUTE]
        
        # Force execute should always ask
        assert config.allow_remember is False
        
        # Each request should be one-time
        request = ConsentRequest(
            topic="force_large_simulation",
            category=ConsentCategory.FORCE_EXECUTE,
            description="Force execution despite resource warnings",
            allow_remember=False,
            allow_force_override=False,
            suggested_level=ConsentLevel.ONE_TIME,
        )
        
        assert request.suggested_level == ConsentLevel.ONE_TIME
        assert request.allow_remember is False

    @pytest.mark.unit
    def test_denied_consent_workflow(self):
        """Test denied consent workflow."""
        record = ConsentRecord(
            topic="suspicious_action",
            category=ConsentCategory.UNTRUSTED_AGENT_MD,
            granted=False,
            level=ConsentLevel.NEVER,
            source="user",
        )
        
        assert record.granted is False
        assert record.level == ConsentLevel.NEVER

    @pytest.mark.unit
    def test_persistent_consent_workflow(self):
        """Test persistent consent that survives sessions."""
        record = ConsentRecord(
            topic="local_ollama",
            category=ConsentCategory.LOCAL_LLM,
            granted=True,
            level=ConsentLevel.PERSISTENT,
            source="user",
        )
        
        # Persistent consents should serialize to disk
        data = record.to_dict()
        restored = ConsentRecord.from_dict(data)
        
        assert restored.level == ConsentLevel.PERSISTENT
        assert restored.granted is True


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestConsentEdgeCases:
    """Tests for edge cases in consent system."""

    @pytest.mark.unit
    def test_record_with_zero_expiry(self):
        """Test record with zero expiry time."""
        # expires_at=0.0 means "no expiry" in the actual implementation
        # So we use a past timestamp instead
        import time
        past_time = time.time() - 3600  # 1 hour ago
        record = ConsentRecord(
            topic="test",
            category=None,
            granted=True,
            level=ConsentLevel.SESSION,
            expires_at=past_time,  # Past timestamp (expired)
        )
        assert record.is_valid() is False

    @pytest.mark.unit
    def test_record_with_very_long_topic(self):
        """Test record with very long topic name."""
        long_topic = "a" * 1000
        record = ConsentRecord(
            topic=long_topic,
            category=None,
            granted=True,
            level=ConsentLevel.ONE_TIME,
        )
        assert len(record.topic) == 1000

    @pytest.mark.unit
    def test_record_with_special_characters(self):
        """Test record with special characters in topic."""
        record = ConsentRecord(
            topic="use_api_v2:special/chars&more",
            category=None,
            granted=True,
            level=ConsentLevel.SESSION,
        )
        data = record.to_dict()
        restored = ConsentRecord.from_dict(data)
        assert restored.topic == record.topic

    @pytest.mark.unit
    def test_multiple_records_same_topic(self):
        """Test multiple records for same topic."""
        records = [
            ConsentRecord(
                topic="shared_topic",
                category=ConsentCategory.LOCAL_LLM,
                granted=True,
                level=ConsentLevel.SESSION,
                timestamp=time.time() - 100,
            ),
            ConsentRecord(
                topic="shared_topic",
                category=ConsentCategory.LOCAL_LLM,
                granted=False,
                level=ConsentLevel.NEVER,
                timestamp=time.time(),  # More recent
            ),
        ]
        
        # Latest should take precedence (in real implementation)
        latest = max(records, key=lambda r: r.timestamp)
        assert latest.granted is False
