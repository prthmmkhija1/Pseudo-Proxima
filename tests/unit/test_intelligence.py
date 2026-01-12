"""
Comprehensive tests for Intelligence module.

Tests cover:
- LLM providers (OpenAI, Anthropic, Ollama, LM Studio, llama.cpp)
- Local LLM detection
- LLM router
- Backend selector
- Insight engine
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proxima.intelligence import (
    AmplitudeAnalysis,
    AnthropicProvider,
    BackendCapabilities,
    BackendRegistry,
    # Backend Selector
    BackendSelector,
    BackendType,
    CircuitCharacteristics,
    # Insight Engine
    InsightEngine,
    InsightLevel,
    InsightReport,
    LlamaCppProvider,
    # LLM Router
    LLMRequest,
    LLMResponse,
    LMStudioProvider,
    LocalLLMDetector,
    LocalLLMStatus,
    OllamaProvider,
    OpenAIProvider,
    PatternInfo,
    PatternType,
    ProviderRegistry,
    SelectionResult,
    SelectionScore,
    SelectionStrategy,
    StatisticalMetrics,
    analyze_results,
    summarize_results,
)

# =============================================================================
# LLM Request/Response Tests
# =============================================================================


class TestLLMRequest:
    """Tests for LLMRequest dataclass."""

    def test_request_creation_minimal(self) -> None:
        """Test creating a minimal request."""
        request = LLMRequest(prompt="Hello")

        assert request.prompt == "Hello"
        assert request.provider is None
        assert request.model is None
        assert request.system_prompt is None

    def test_request_creation_full(self) -> None:
        """Test creating a full request with all fields."""
        request = LLMRequest(
            prompt="Tell me about quantum computing",
            provider="openai",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            system_prompt="You are a quantum physics expert.",
        )

        assert request.prompt == "Tell me about quantum computing"
        assert request.provider == "openai"
        assert request.model == "gpt-4"
        assert request.temperature == 0.5
        assert request.max_tokens == 2000
        assert request.system_prompt == "You are a quantum physics expert."

    def test_request_temperature_bounds(self) -> None:
        """Test temperature can be set to boundary values."""
        request_zero = LLMRequest(prompt="Test", temperature=0.0)
        request_one = LLMRequest(prompt="Test", temperature=1.0)

        assert request_zero.temperature == 0.0
        assert request_one.temperature == 1.0


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation_success(self) -> None:
        """Test creating a successful response."""
        response = LLMResponse(
            text="Hello! How can I help?",
            provider="openai",
            model="gpt-4",
            latency_ms=150.5,
        )

        assert response.text == "Hello! How can I help?"
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert response.latency_ms == 150.5
        assert response.error is None
        assert response.raw is None

    def test_response_creation_with_error(self) -> None:
        """Test creating an error response."""
        response = LLMResponse(
            text="",
            provider="openai",
            model="gpt-4",
            latency_ms=50.0,
            error="API rate limit exceeded",
        )

        assert response.text == ""
        assert response.error == "API rate limit exceeded"

    def test_response_with_raw_data(self) -> None:
        """Test response with raw API data."""
        raw = {"id": "chatcmpl-123", "usage": {"total_tokens": 50}}
        response = LLMResponse(
            text="Response",
            provider="openai",
            model="gpt-4",
            latency_ms=100.0,
            raw=raw,
        )

        assert response.raw == raw
        assert response.raw["id"] == "chatcmpl-123"


# =============================================================================
# Provider Tests
# =============================================================================


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_provider_properties(self) -> None:
        """Test provider name and type."""
        provider = OpenAIProvider()

        assert provider.name == "openai"
        assert provider.requires_api_key is True
        assert provider.is_local is False

    def test_default_model(self) -> None:
        """Test default model is set."""
        provider = OpenAIProvider()

        # Default model should be gpt-4 or gpt-3.5
        assert provider.default_model is not None
        assert "gpt" in provider.default_model

    @patch("httpx.Client.post")
    def test_send_success(self, mock_post: MagicMock) -> None:
        """Test successful API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}],
            "model": "gpt-4",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        provider = OpenAIProvider()
        request = LLMRequest(prompt="Hi")
        response = provider.send(request, api_key="test-key")

        assert response.text == "Hello!"
        assert response.provider == "openai"

    @patch("httpx.Client.post")
    def test_send_error(self, mock_post: MagicMock) -> None:
        """Test API error handling."""
        mock_post.side_effect = Exception("Connection failed")

        provider = OpenAIProvider()
        request = LLMRequest(prompt="Hi")
        response = provider.send(request, api_key="test-key")

        assert response.error is not None
        assert "Connection failed" in response.error


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_provider_properties(self) -> None:
        """Test provider name and type."""
        provider = AnthropicProvider()

        assert provider.name == "anthropic"
        assert provider.requires_api_key is True
        assert provider.is_local is False

    def test_default_model(self) -> None:
        """Test default model is Claude."""
        provider = AnthropicProvider()

        assert provider.default_model is not None
        assert "claude" in provider.default_model

    @patch("httpx.Client.post")
    def test_send_success(self, mock_post: MagicMock) -> None:
        """Test successful API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": "Hello from Claude!"}],
            "model": "claude-3-sonnet",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        provider = AnthropicProvider()
        request = LLMRequest(prompt="Hi")
        response = provider.send(request, api_key="test-key")

        assert response.text == "Hello from Claude!"
        assert response.provider == "anthropic"


class TestOllamaProvider:
    """Tests for Ollama local provider."""

    def test_provider_properties(self) -> None:
        """Test provider is local."""
        provider = OllamaProvider()

        assert provider.name == "ollama"
        assert provider.requires_api_key is False
        assert provider.is_local is True

    def test_default_model(self) -> None:
        """Test default model."""
        provider = OllamaProvider()

        assert provider.default_model == "llama2"

    @patch("httpx.Client.get")
    def test_health_check_success(self, mock_get: MagicMock) -> None:
        """Test health check when server is running."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        provider = OllamaProvider()
        assert provider.health_check() is True

    @patch("httpx.Client.get")
    def test_health_check_failure(self, mock_get: MagicMock) -> None:
        """Test health check when server is not running."""
        mock_get.side_effect = Exception("Connection refused")

        provider = OllamaProvider()
        assert provider.health_check() is False


class TestLMStudioProvider:
    """Tests for LM Studio provider."""

    def test_provider_properties(self) -> None:
        """Test provider is local."""
        provider = LMStudioProvider()

        assert provider.name == "lmstudio"
        assert provider.requires_api_key is False
        assert provider.is_local is True

    @patch("httpx.Client.get")
    def test_health_check(self, mock_get: MagicMock) -> None:
        """Test health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        provider = LMStudioProvider()
        assert provider.health_check() is True


class TestLlamaCppProvider:
    """Tests for llama.cpp server provider."""

    def test_provider_properties(self) -> None:
        """Test provider is local."""
        provider = LlamaCppProvider()

        assert provider.name == "llama_cpp"
        assert provider.requires_api_key is False
        assert provider.is_local is True


# =============================================================================
# Provider Registry Tests
# =============================================================================


class TestProviderRegistry:
    """Tests for provider registry."""

    def test_default_providers_registered(self) -> None:
        """Test default providers are available."""
        registry = ProviderRegistry()

        assert registry.get("openai") is not None
        assert registry.get("anthropic") is not None
        assert registry.get("ollama") is not None
        assert registry.get("lmstudio") is not None
        assert registry.get("llama_cpp") is not None

    def test_list_providers(self) -> None:
        """Test listing all providers."""
        registry = ProviderRegistry()
        providers = registry.list_providers()

        assert len(providers) >= 5
        assert "openai" in providers
        assert "anthropic" in providers

    def test_get_local_providers(self) -> None:
        """Test getting only local providers."""
        registry = ProviderRegistry()
        local = registry.get_local_providers()

        assert all(p.is_local for p in local)
        assert len(local) >= 3  # ollama, lmstudio, llama_cpp


# =============================================================================
# Local LLM Detection Tests
# =============================================================================


class TestLocalLLMDetector:
    """Tests for local LLM detection."""

    def test_detector_creation(self) -> None:
        """Test creating a detector."""
        detector = LocalLLMDetector()

        assert detector is not None

    def test_detect_all_returns_list(self) -> None:
        """Test detect_all returns list of statuses (mocked to avoid network)."""
        detector = LocalLLMDetector()

        # Mock the detect_all method to avoid actual network calls
        mock_statuses = [
            LocalLLMStatus(
                provider="ollama", available=False, endpoint="http://localhost:11434"
            ),
            LocalLLMStatus(
                provider="lm_studio", available=False, endpoint="http://localhost:1234"
            ),
            LocalLLMStatus(
                provider="llama_cpp", available=False, endpoint="http://localhost:8080"
            ),
        ]
        with patch.object(detector, "detect_all", return_value=mock_statuses):
            statuses = detector.detect_all()

            assert isinstance(statuses, list)
            assert len(statuses) >= 3
            assert all(isinstance(s, LocalLLMStatus) for s in statuses)

    def test_get_first_available_with_mock(self) -> None:
        """Test get_first_available with mocked providers."""
        detector = LocalLLMDetector()

        # Mock to return no available providers
        with patch.object(
            detector,
            "detect_all",
            return_value=[
                LocalLLMStatus(
                    provider="ollama",
                    available=False,
                    endpoint="http://localhost:11434",
                ),
                LocalLLMStatus(
                    provider="lm_studio",
                    available=False,
                    endpoint="http://localhost:1234",
                ),
                LocalLLMStatus(
                    provider="llama_cpp",
                    available=False,
                    endpoint="http://localhost:8080",
                ),
            ],
        ):
            result = detector.get_first_available()
            assert result is None

    def test_get_first_available_with_available_provider(self) -> None:
        """Test get_first_available when a provider is available."""
        detector = LocalLLMDetector()

        # Mock to return one available provider
        with patch.object(
            detector,
            "detect_all",
            return_value=[
                LocalLLMStatus(
                    provider="ollama",
                    available=True,
                    endpoint="http://localhost:11434",
                    models=["llama2"],
                ),
                LocalLLMStatus(
                    provider="lm_studio",
                    available=False,
                    endpoint="http://localhost:1234",
                ),
            ],
        ):
            result = detector.get_first_available()
            assert result is not None
            assert result.provider == "ollama"
            assert result.available is True


# =============================================================================
# Backend Selector Tests
# =============================================================================


class TestCircuitCharacteristics:
    """Tests for circuit characteristics analysis."""

    def test_manual_creation(self) -> None:
        """Test creating characteristics manually."""
        chars = CircuitCharacteristics(
            qubit_count=5,
            gate_count=20,
            depth=10,
            gate_types={"h", "cx", "rz"},
            entanglement_density=0.3,
        )

        assert chars.qubit_count == 5
        assert chars.gate_count == 20
        assert chars.depth == 10
        assert "h" in chars.gate_types

    def test_estimated_memory(self) -> None:
        """Test memory estimation."""
        chars = CircuitCharacteristics(
            qubit_count=10,
            gate_count=50,
            depth=20,
            estimated_memory_mb=16.0,
        )

        assert chars.estimated_memory_mb == 16.0


class TestBackendRegistry:
    """Tests for backend registry."""

    def test_default_backends(self) -> None:
        """Test default backends are registered."""
        registry = BackendRegistry()
        backends = registry.list_all()

        assert len(backends) >= 5
        names = [b.name for b in backends]
        assert "numpy" in names
        assert "cupy" in names
        assert "qiskit" in names

    def test_register_custom_backend(self) -> None:
        """Test registering a custom backend."""
        registry = BackendRegistry()
        custom = BackendCapabilities(
            name="custom",
            backend_type=BackendType.CUSTOM,
            max_qubits=50,
            supports_gpu=True,
        )
        registry.register(custom)

        assert registry.get("custom") is not None
        assert registry.get("custom").max_qubits == 50

    def test_list_compatible_small_circuit(self) -> None:
        """Test listing compatible backends for small circuit."""
        registry = BackendRegistry()
        chars = CircuitCharacteristics(
            qubit_count=5,
            gate_count=20,
            depth=10,
        )

        # Without runtime check, all registered backends are compatible
        compatible = registry.list_compatible(chars, check_runtime=False)

        # All backends should be compatible with 5 qubits
        assert len(compatible) >= 5

    def test_list_compatible_large_circuit(self) -> None:
        """Test listing compatible backends for large circuit."""
        registry = BackendRegistry()
        chars = CircuitCharacteristics(
            qubit_count=35,  # Exceeds most max_qubits
            gate_count=100,
            depth=50,
        )

        compatible = registry.list_compatible(chars)

        # Fewer backends support 35 qubits
        assert len(compatible) < 5


class TestBackendSelector:
    """Tests for backend selector."""

    def test_selector_creation(self) -> None:
        """Test creating a selector."""
        selector = BackendSelector()

        assert selector is not None

    def test_list_backends(self) -> None:
        """Test listing available backends."""
        selector = BackendSelector()
        backends = selector.list_backends()

        assert "numpy" in backends
        assert "cupy" in backends

    def test_select_from_characteristics(self) -> None:
        """Test selection from characteristics."""
        selector = BackendSelector()
        chars = CircuitCharacteristics(
            qubit_count=10,
            gate_count=50,
            depth=20,
            estimated_memory_mb=16.0,
        )

        result = selector.select_from_characteristics(chars)

        assert isinstance(result, SelectionResult)
        assert result.selected_backend is not None
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        assert len(result.scores) > 0
        assert result.explanation != ""

    def test_select_with_performance_strategy(self) -> None:
        """Test selection prioritizing performance."""
        selector = BackendSelector()
        chars = CircuitCharacteristics(
            qubit_count=15,
            gate_count=100,
            depth=30,
        )

        result = selector.select_from_characteristics(
            chars,
            strategy=SelectionStrategy.PERFORMANCE,
        )

        assert result is not None

    def test_select_with_memory_strategy(self) -> None:
        """Test selection prioritizing memory."""
        selector = BackendSelector()
        chars = CircuitCharacteristics(
            qubit_count=20,
            gate_count=200,
            depth=50,
            estimated_memory_mb=1024.0,
        )

        result = selector.select_from_characteristics(
            chars,
            strategy=SelectionStrategy.MEMORY,
        )

        assert result is not None

    def test_selection_result_has_alternatives(self) -> None:
        """Test that selection result includes alternatives."""
        selector = BackendSelector()
        chars = CircuitCharacteristics(
            qubit_count=8,
            gate_count=40,
            depth=15,
        )

        result = selector.select_from_characteristics(chars)

        assert isinstance(result.alternatives, list)

    def test_selection_result_has_reasoning(self) -> None:
        """Test that selection result includes reasoning steps."""
        selector = BackendSelector()
        chars = CircuitCharacteristics(
            qubit_count=10,
            gate_count=50,
            depth=20,
        )

        result = selector.select_from_characteristics(chars)

        assert isinstance(result.reasoning_steps, list)
        assert len(result.reasoning_steps) > 0

    def test_selection_scores_sorted(self) -> None:
        """Test that scores are sorted by total score."""
        selector = BackendSelector()
        chars = CircuitCharacteristics(
            qubit_count=10,
            gate_count=50,
            depth=20,
        )

        result = selector.select_from_characteristics(chars)

        for i in range(len(result.scores) - 1):
            assert result.scores[i].total_score >= result.scores[i + 1].total_score


class TestSelectionScore:
    """Tests for selection score dataclass."""

    def test_score_creation(self) -> None:
        """Test creating a selection score."""
        score = SelectionScore(
            backend_name="numpy",
            total_score=0.75,
            feature_score=0.8,
            performance_score=0.6,
            memory_score=0.7,
            history_score=0.5,
            compatibility_score=0.9,
        )

        assert score.backend_name == "numpy"
        assert score.total_score == 0.75

    def test_score_with_details(self) -> None:
        """Test score with additional details."""
        score = SelectionScore(
            backend_name="cupy",
            total_score=0.85,
            feature_score=0.9,
            performance_score=0.95,
            memory_score=0.5,
            history_score=0.6,
            compatibility_score=0.8,
            details={"gpu_available": True, "qubit_headroom": 15},
        )

        assert score.details["gpu_available"] is True
        assert score.details["qubit_headroom"] == 15


# =============================================================================
# Insight Engine Tests
# =============================================================================


class TestStatisticalMetrics:
    """Tests for statistical metrics."""

    def test_metrics_creation(self) -> None:
        """Test creating metrics manually."""
        metrics = StatisticalMetrics(
            entropy=1.5,
            max_probability=0.5,
            min_probability=0.1,
            mean_probability=0.25,
            std_probability=0.15,
            total_states=4,
            non_zero_states=4,
            dominant_state="00",
            dominant_probability=0.5,
            effective_dimension=2.5,
            gini_coefficient=0.3,
            top_k_coverage=0.9,
        )

        assert metrics.entropy == 1.5
        assert metrics.dominant_state == "00"


class TestPatternInfo:
    """Tests for pattern detection results."""

    def test_pattern_creation(self) -> None:
        """Test creating pattern info."""
        pattern = PatternInfo(
            pattern_type=PatternType.PEAKED,
            confidence=0.85,
            description="Strong peak at state '00'",
            affected_states=["00"],
        )

        assert pattern.pattern_type == PatternType.PEAKED
        assert pattern.confidence == 0.85

    def test_pattern_with_metrics(self) -> None:
        """Test pattern with additional metrics."""
        pattern = PatternInfo(
            pattern_type=PatternType.ENTANGLED,
            confidence=0.95,
            description="Bell state detected",
            affected_states=["00", "11"],
            metrics={"combined_probability": 0.98},
        )

        assert pattern.metrics["combined_probability"] == 0.98


class TestInsightEngine:
    """Tests for insight engine."""

    def test_engine_creation(self) -> None:
        """Test creating an insight engine."""
        engine = InsightEngine()

        assert engine is not None

    def test_analyze_uniform_distribution(self) -> None:
        """Test analyzing uniform distribution."""
        engine = InsightEngine()
        probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}

        report = engine.analyze(probs)

        assert isinstance(report, InsightReport)
        assert report.statistics.entropy == pytest.approx(2.0, abs=0.01)
        assert report.statistics.total_states == 4
        assert report.statistics.non_zero_states == 4

    def test_analyze_peaked_distribution(self) -> None:
        """Test analyzing peaked distribution."""
        engine = InsightEngine()
        probs = {"00": 0.9, "01": 0.05, "10": 0.03, "11": 0.02}

        report = engine.analyze(probs)

        assert report.statistics.dominant_state == "00"
        assert report.statistics.dominant_probability == 0.9

        # Should detect peaked pattern
        peaked_patterns = [
            p for p in report.patterns if p.pattern_type == PatternType.PEAKED
        ]
        assert len(peaked_patterns) > 0

    def test_analyze_bell_state(self) -> None:
        """Test analyzing Bell state (entanglement)."""
        engine = InsightEngine()
        probs = {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}

        report = engine.analyze(probs)

        # Should detect entanglement pattern
        entangled = [
            p for p in report.patterns if p.pattern_type == PatternType.ENTANGLED
        ]
        assert len(entangled) > 0

    def test_analyze_with_amplitudes(self) -> None:
        """Test analysis with amplitude data."""
        engine = InsightEngine()
        probs = {"00": 0.5, "11": 0.5}
        amps = {
            "00": complex(0.707, 0.0),
            "11": complex(0.707, 0.0),
        }

        report = engine.analyze(probs, amplitudes=amps)

        assert report.amplitude_analysis is not None
        assert isinstance(report.amplitude_analysis, AmplitudeAnalysis)

    def test_analyze_levels(self) -> None:
        """Test different analysis levels."""
        engine = InsightEngine()
        probs = {"0": 0.7, "1": 0.3}

        basic = engine.analyze(probs, level=InsightLevel.BASIC)
        detailed = engine.analyze(probs, level=InsightLevel.DETAILED)

        # Detailed should have recommendations
        assert len(detailed.recommendations) >= len(basic.recommendations)

    def test_quick_analyze(self) -> None:
        """Test quick analysis method."""
        engine = InsightEngine()
        probs = {"00": 0.5, "11": 0.5}

        summary = engine.quick_analyze(probs)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_report_has_summary(self) -> None:
        """Test that report has summary."""
        engine = InsightEngine()
        probs = {"0": 1.0}

        report = engine.analyze(probs)

        assert report.summary is not None
        assert len(report.summary) > 0

    def test_report_has_key_findings(self) -> None:
        """Test that report has key findings."""
        engine = InsightEngine()
        probs = {"00": 0.5, "11": 0.5}

        report = engine.analyze(probs)

        assert isinstance(report.key_findings, list)
        assert len(report.key_findings) > 0

    def test_report_has_visualizations(self) -> None:
        """Test that report includes visualization suggestions."""
        engine = InsightEngine()
        probs = {"00": 0.5, "01": 0.3, "10": 0.15, "11": 0.05}

        report = engine.analyze(probs)

        assert isinstance(report.visualizations, list)
        assert len(report.visualizations) > 0

        # Should suggest bar chart
        bar_viz = [v for v in report.visualizations if v.viz_type == "bar"]
        assert len(bar_viz) > 0

    def test_engine_with_llm_callback(self) -> None:
        """Test engine with LLM callback."""

        def mock_llm(prompt: str) -> str:
            return "This is a mock LLM response. What else would you like to know?"

        engine = InsightEngine(llm_callback=mock_llm)
        probs = {"00": 0.5, "11": 0.5}

        report = engine.analyze(probs, level=InsightLevel.EXPERT)

        assert report.llm_synthesis is not None
        assert "mock LLM response" in report.llm_synthesis


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_results(self) -> None:
        """Test analyze_results function."""
        probs = {"0": 0.7, "1": 0.3}

        report = analyze_results(probs)

        assert isinstance(report, InsightReport)

    def test_summarize_results(self) -> None:
        """Test summarize_results function."""
        probs = {"00": 0.9, "01": 0.05, "10": 0.03, "11": 0.02}

        summary = summarize_results(probs)

        assert isinstance(summary, str)
        assert "00" in summary  # Should mention dominant state


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_probabilities(self) -> None:
        """Test analyzing empty probability distribution."""
        engine = InsightEngine()
        probs: dict[str, float] = {}

        report = engine.analyze(probs)

        assert report.statistics.total_states == 0
        assert report.statistics.entropy == 0.0

    def test_single_state_probability(self) -> None:
        """Test analyzing single state with 100% probability."""
        engine = InsightEngine()
        probs = {"0": 1.0}

        report = engine.analyze(probs)

        assert report.statistics.entropy == 0.0
        assert report.statistics.dominant_probability == 1.0
        assert len(report.warnings) > 0  # Should warn about deterministic

    def test_very_small_probabilities(self) -> None:
        """Test handling very small probabilities."""
        engine = InsightEngine()
        # 1e-10 is the cutoff, so 1e-10 itself is filtered out
        probs = {"0": 1e-10, "1": 1.0 - 1e-10}

        report = engine.analyze(probs)

        # Should handle near-zero probabilities
        # Only the large probability counts as non-zero (1e-10 is filtered)
        assert report.statistics.non_zero_states == 1

    def test_large_state_space(self) -> None:
        """Test analyzing large state space."""
        engine = InsightEngine()
        # Create 256 states (8 qubits)
        probs = {format(i, "08b"): 1 / 256 for i in range(256)}

        report = engine.analyze(probs)

        assert report.statistics.total_states == 256
        assert report.statistics.entropy == pytest.approx(8.0, abs=0.01)

    def test_selector_no_compatible_backends(self) -> None:
        """Test selector when no backends are compatible."""
        selector = BackendSelector()
        chars = CircuitCharacteristics(
            qubit_count=100,  # No backend supports this
            gate_count=1000,
            depth=500,
        )

        result = selector.select_from_characteristics(chars)

        # Should fallback gracefully
        assert result.selected_backend == "numpy"
        assert result.confidence == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_selector_with_history(self) -> None:
        """Test selector with history provider."""

        def mock_history(backend: str) -> float:
            history = {"numpy": 0.9, "cupy": 0.7, "qiskit": 0.85}
            return history.get(backend, 0.5)

        selector = BackendSelector(history_provider=mock_history)
        chars = CircuitCharacteristics(
            qubit_count=10,
            gate_count=50,
            depth=20,
        )

        result = selector.select_from_characteristics(chars)

        # History should influence selection
        numpy_score = next(s for s in result.scores if s.backend_name == "numpy")
        assert numpy_score.history_score == 0.9

    def test_insight_with_circuit_info(self) -> None:
        """Test insights with circuit context."""
        engine = InsightEngine()
        probs = {"00": 0.5, "11": 0.5}
        circuit_info = {
            "qubits": 2,
            "gates": 3,
            "depth": 2,
            "description": "Bell state preparation",
        }

        report = engine.analyze(probs, circuit_info=circuit_info)

        assert report is not None
