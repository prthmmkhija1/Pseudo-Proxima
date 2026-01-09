"""
Unit tests for the intelligence module.

Tests LLM routing, backend selection, and insights generation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proxima.intelligence.llm_router import (
    AnthropicProvider,
    APIKeyManager,
    ConsentGate,
    LLMRequest,
    LLMResponse,
    LLMRouter,
    LMStudioProvider,
    LocalLLMDetector,
    OllamaProvider,
    OpenAIProvider,
    ProviderRegistry,
)

# ===================== Test Fixtures =====================


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings."""
    settings = MagicMock()
    settings.llm.provider = "openai"
    settings.llm.api_key_env_var = "OPENAI_API_KEY"
    settings.llm.require_consent = False
    settings.llm.local_endpoint = ""
    settings.consent.auto_approve_local_llm = True
    settings.consent.auto_approve_remote_llm = False
    return settings


@pytest.fixture
def registry() -> ProviderRegistry:
    """Create a provider registry."""
    return ProviderRegistry()


# ===================== Provider Tests =====================


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_provider_properties(self) -> None:
        """Test provider properties."""
        provider = OpenAIProvider(default_model="gpt-4")
        assert provider.name == "openai"
        assert provider.is_local is False
        assert provider.requires_api_key is True

    def test_send_returns_stub_response(self) -> None:
        """Test that send returns a stub response."""
        provider = OpenAIProvider(default_model="gpt-4")
        request = LLMRequest(prompt="Hello, world!")

        response = provider.send(request, api_key="test-key")

        assert isinstance(response, LLMResponse)
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert "stub" in response.text.lower()


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_provider_properties(self) -> None:
        """Test provider properties."""
        provider = AnthropicProvider(default_model="claude-3")
        assert provider.name == "anthropic"
        assert provider.is_local is False
        assert provider.requires_api_key is True


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_provider_properties(self) -> None:
        """Test provider properties."""
        provider = OllamaProvider(default_model="llama3")
        assert provider.name == "ollama"
        assert provider.is_local is True
        assert provider.requires_api_key is False


class TestLMStudioProvider:
    """Tests for LMStudioProvider."""

    def test_provider_properties(self) -> None:
        """Test provider properties."""
        provider = LMStudioProvider(default_model="gpt4all")
        assert provider.name == "lmstudio"
        assert provider.is_local is True
        assert provider.requires_api_key is False


# ===================== Provider Registry Tests =====================


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def test_get_known_provider(self, registry: ProviderRegistry) -> None:
        """Test getting a known provider."""
        provider = registry.get("openai")
        assert provider.name == "openai"

    def test_get_unknown_provider(self, registry: ProviderRegistry) -> None:
        """Test getting an unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            registry.get("unknown")  # type: ignore

    def test_all_providers_registered(self, registry: ProviderRegistry) -> None:
        """Test that all standard providers are registered."""
        assert registry.get("openai") is not None
        assert registry.get("anthropic") is not None
        assert registry.get("ollama") is not None
        assert registry.get("lmstudio") is not None


# ===================== Local LLM Detector Tests =====================


class TestLocalLLMDetector:
    """Tests for LocalLLMDetector."""

    def test_default_endpoints(self) -> None:
        """Test default endpoint configuration."""
        detector = LocalLLMDetector()
        assert "ollama" in detector.DEFAULT_ENDPOINTS
        assert "lmstudio" in detector.DEFAULT_ENDPOINTS

    def test_detect_with_no_endpoint(self) -> None:
        """Test detection when no endpoint is available."""
        detector = LocalLLMDetector(timeout_s=0.1)

        with patch.object(detector, "_check", return_value=False):
            result = detector.detect("ollama")

        assert result is None

    def test_detect_with_configured_endpoint(self) -> None:
        """Test detection with a configured endpoint."""
        detector = LocalLLMDetector(timeout_s=0.1)

        with patch.object(detector, "_check", return_value=True):
            result = detector.detect("ollama", configured_endpoint="http://localhost:11434")

        assert result == "http://localhost:11434"

    def test_detect_caches_result(self) -> None:
        """Test that detection results are cached."""
        detector = LocalLLMDetector()
        detector._cache["ollama"] = "http://cached:11434"

        result = detector.detect("ollama")
        assert result == "http://cached:11434"

    def test_detect_force_refresh(self) -> None:
        """Test force refresh bypasses cache."""
        detector = LocalLLMDetector()
        detector._cache["ollama"] = "http://cached:11434"

        with patch.object(detector, "_check", return_value=False):
            result = detector.detect("ollama", force=True)

        assert result is None


# ===================== API Key Manager Tests =====================


class TestAPIKeyManager:
    """Tests for APIKeyManager."""

    def test_get_api_key_from_env(self, mock_settings: MagicMock) -> None:
        """Test getting API key from environment variable."""
        manager = APIKeyManager(mock_settings)
        provider = OpenAIProvider(default_model="gpt-4")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            key = manager.get_api_key(provider)

        assert key == "test-key"

    def test_get_api_key_missing(self, mock_settings: MagicMock) -> None:
        """Test getting API key when not set."""
        mock_settings.llm.api_key_env_var = "NONEXISTENT_KEY"
        manager = APIKeyManager(mock_settings)
        provider = OpenAIProvider(default_model="gpt-4")

        key = manager.get_api_key(provider)
        assert key is None

    def test_validate_local_provider(self, mock_settings: MagicMock) -> None:
        """Test validation skips local providers."""
        manager = APIKeyManager(mock_settings)
        provider = OllamaProvider(default_model="llama3")

        # Should not raise
        manager.validate(provider)

    def test_validate_missing_key(self, mock_settings: MagicMock) -> None:
        """Test validation fails for missing API key."""
        mock_settings.llm.api_key_env_var = "NONEXISTENT_KEY"
        manager = APIKeyManager(mock_settings)
        provider = OpenAIProvider(default_model="gpt-4")

        with pytest.raises(ValueError, match="API key missing"):
            manager.validate(provider)


# ===================== Consent Gate Tests =====================


class TestConsentGate:
    """Tests for ConsentGate."""

    def test_consent_not_required(self, mock_settings: MagicMock) -> None:
        """Test when consent is not required."""
        mock_settings.llm.require_consent = False
        gate = ConsentGate(mock_settings)
        provider = OpenAIProvider(default_model="gpt-4")

        # Should not raise
        gate.require_consent(provider)

    def test_auto_approve_local(self, mock_settings: MagicMock) -> None:
        """Test auto-approval for local providers."""
        mock_settings.llm.require_consent = True
        mock_settings.consent.auto_approve_local_llm = True
        gate = ConsentGate(mock_settings)
        provider = OllamaProvider(default_model="llama3")

        # Should not raise
        gate.require_consent(provider)

    def test_consent_prompt_approved(self, mock_settings: MagicMock) -> None:
        """Test consent prompt when approved."""
        mock_settings.llm.require_consent = True
        mock_settings.consent.auto_approve_remote_llm = False
        prompt_func = MagicMock(return_value=True)
        gate = ConsentGate(mock_settings, prompt_func=prompt_func)
        provider = OpenAIProvider(default_model="gpt-4")

        # Should not raise
        gate.require_consent(provider)
        prompt_func.assert_called_once()

    def test_consent_prompt_denied(self, mock_settings: MagicMock) -> None:
        """Test consent prompt when denied."""
        mock_settings.llm.require_consent = True
        mock_settings.consent.auto_approve_remote_llm = False
        prompt_func = MagicMock(return_value=False)
        gate = ConsentGate(mock_settings, prompt_func=prompt_func)
        provider = OpenAIProvider(default_model="gpt-4")

        with pytest.raises(PermissionError, match="User denied consent"):
            gate.require_consent(provider)

    def test_consent_no_prompt_function(self, mock_settings: MagicMock) -> None:
        """Test consent required but no prompt function."""
        mock_settings.llm.require_consent = True
        mock_settings.consent.auto_approve_remote_llm = False
        gate = ConsentGate(mock_settings, prompt_func=None)
        provider = OpenAIProvider(default_model="gpt-4")

        with pytest.raises(PermissionError, match="Consent required"):
            gate.require_consent(provider)


# ===================== LLM Router Tests =====================


class TestLLMRouter:
    """Tests for LLMRouter."""

    def test_route_request(self, mock_settings: MagicMock) -> None:
        """Test routing a request."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            router = LLMRouter(settings=mock_settings)
            request = LLMRequest(prompt="Hello!")

            response = router.route(request)

        assert isinstance(response, LLMResponse)
        assert response.provider == "openai"

    def test_route_with_explicit_provider(self, mock_settings: MagicMock) -> None:
        """Test routing with explicitly specified provider."""
        mock_settings.llm.provider = None
        router = LLMRouter(settings=mock_settings)
        request = LLMRequest(prompt="Hello!", provider="ollama")

        with patch.object(router.detector, "detect", return_value="http://localhost:11434"):
            response = router.route(request)

        assert response.provider == "ollama"

    def test_route_no_provider_configured(self, mock_settings: MagicMock) -> None:
        """Test error when no provider is configured."""
        mock_settings.llm.provider = "none"
        router = LLMRouter(settings=mock_settings)
        request = LLMRequest(prompt="Hello!")

        with pytest.raises(ValueError, match="No LLM provider configured"):
            router.route(request)

    def test_route_local_unavailable(self, mock_settings: MagicMock) -> None:
        """Test error when local provider is unavailable."""
        mock_settings.llm.provider = "ollama"
        router = LLMRouter(settings=mock_settings)
        request = LLMRequest(prompt="Hello!")

        with patch.object(router.detector, "detect", return_value=None):
            with pytest.raises(ConnectionError, match="Local endpoint not reachable"):
                router.route(request)


# ===================== LLM Request/Response Tests =====================


class TestLLMRequest:
    """Tests for LLMRequest dataclass."""

    def test_request_defaults(self) -> None:
        """Test request default values."""
        request = LLMRequest(prompt="Test prompt")

        assert request.prompt == "Test prompt"
        assert request.model is None
        assert request.temperature == 0.0
        assert request.max_tokens is None
        assert request.provider is None
        assert request.metadata == {}

    def test_request_with_all_fields(self) -> None:
        """Test request with all fields specified."""
        request = LLMRequest(
            prompt="Test",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            provider="openai",
            metadata={"key": "value"},
        )

        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self) -> None:
        """Test creating a response."""
        response = LLMResponse(
            text="Hello!",
            provider="openai",
            model="gpt-4",
            latency_ms=100.5,
        )

        assert response.text == "Hello!"
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert response.latency_ms == 100.5
        assert response.raw is None
