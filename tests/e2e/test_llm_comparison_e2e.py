"""
E2E Tests for LLM Integration and Multi-Backend Comparison.

Comprehensive end-to-end tests covering:
- LLM router and provider workflows
- Multi-backend comparison scenarios
- Result interpretation pipelines
- Insight generation flows
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return """Based on my analysis of the quantum simulation results:

1. The Bell state shows strong entanglement with 50-50 distribution
2. Entropy of 1.0 indicates maximum superposition
3. The |00⟩ and |11⟩ states dominate as expected

Recommendation: Results look correct for a Bell state preparation."""


@pytest.fixture
def sample_simulation_results():
    """Sample simulation results for testing."""
    return {
        "counts": {
            "00": 498,
            "01": 12,
            "10": 8,
            "11": 482,
        },
        "shots": 1000,
        "backend": "cirq",
        "execution_time_ms": 145.3,
        "circuit_info": {
            "qubits": 2,
            "depth": 3,
            "gates": 4,
        },
    }


@pytest.fixture
def multi_backend_results():
    """Results from multiple backends for comparison."""
    return {
        "cirq": {
            "counts": {"00": 512, "11": 488},
            "execution_time_ms": 145.3,
            "backend": "cirq",
        },
        "qiskit_aer": {
            "counts": {"00": 498, "11": 502},
            "execution_time_ms": 132.7,
            "backend": "qiskit_aer",
        },
        "lret": {
            "counts": {"00": 505, "11": 495},
            "execution_time_ms": 98.2,
            "backend": "lret",
        },
    }


# =============================================================================
# LLM Provider Tests
# =============================================================================


class TestLLMProviders:
    """Tests for LLM provider implementations."""

    def test_provider_registry_creation(self):
        """Test ProviderRegistry can be created."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        
        assert registry is not None
        providers = registry.list_providers()
        assert len(providers) >= 5  # Core + extended providers

    def test_openai_provider_exists(self):
        """Test OpenAI provider is registered."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        provider = registry.get("openai")
        
        assert provider is not None
        assert provider.name == "openai"
        assert provider.is_local is False
        assert provider.requires_api_key is True

    def test_anthropic_provider_exists(self):
        """Test Anthropic provider is registered."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        provider = registry.get("anthropic")
        
        assert provider is not None
        assert provider.name == "anthropic"

    def test_ollama_provider_exists(self):
        """Test Ollama provider is registered."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        provider = registry.get("ollama")
        
        assert provider is not None
        assert provider.is_local is True
        assert provider.requires_api_key is False

    def test_together_provider_exists(self):
        """Test Together AI provider is registered."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        provider = registry.get("together")
        
        assert provider is not None
        assert provider.name == "together"
        assert provider.is_local is False

    def test_groq_provider_exists(self):
        """Test Groq provider is registered."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        provider = registry.get("groq")
        
        assert provider is not None
        assert provider.name == "groq"

    def test_mistral_provider_exists(self):
        """Test Mistral provider is registered."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        provider = registry.get("mistral")
        
        assert provider is not None
        assert provider.name == "mistral"

    def test_azure_openai_provider_exists(self):
        """Test Azure OpenAI provider is registered."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        provider = registry.get("azure_openai")
        
        assert provider is not None
        assert provider.name == "azure_openai"

    def test_cohere_provider_exists(self):
        """Test Cohere provider is registered."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        provider = registry.get("cohere")
        
        assert provider is not None
        assert provider.name == "cohere"

    def test_get_local_providers(self):
        """Test getting local providers."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        local_providers = registry.get_local_providers()
        
        assert len(local_providers) >= 3  # ollama, lmstudio, llama_cpp
        for provider in local_providers:
            assert provider.is_local is True

    def test_get_remote_providers(self):
        """Test getting remote providers."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        remote_providers = registry.get_remote_providers()
        
        assert len(remote_providers) >= 5  # openai, anthropic, together, groq, mistral, azure, cohere
        for provider in remote_providers:
            assert provider.is_local is False


# =============================================================================
# LLM Request/Response Tests
# =============================================================================


class TestLLMRequestResponse:
    """Tests for LLM request and response handling."""

    def test_llm_request_creation(self):
        """Test LLMRequest can be created."""
        from proxima.intelligence.llm_router import LLMRequest
        
        request = LLMRequest(
            prompt="Explain quantum entanglement",
            model="gpt-4",
            temperature=0.7,
            max_tokens=500,
            system_prompt="You are a quantum physics expert.",
        )
        
        assert request.prompt == "Explain quantum entanglement"
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens == 500

    def test_llm_request_with_tools(self):
        """Test LLMRequest with tool definitions."""
        from proxima.intelligence.llm_router import LLMRequest
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_quantum_state",
                    "description": "Get the current quantum state",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "qubit_id": {"type": "integer"},
                        },
                    },
                },
            }
        ]
        
        request = LLMRequest(
            prompt="What is the state of qubit 0?",
            tools=tools,
            tool_choice="auto",
        )
        
        assert request.tools == tools
        assert request.tool_choice == "auto"

    def test_llm_response_creation(self):
        """Test LLMResponse can be created."""
        from proxima.intelligence.llm_router import LLMResponse
        
        response = LLMResponse(
            text="Quantum entanglement is...",
            provider="openai",
            model="gpt-4",
            latency_ms=500.0,
            tokens_used=150,
        )
        
        assert response.text == "Quantum entanglement is..."
        assert response.provider == "openai"
        assert response.latency_ms == 500.0

    def test_function_call_parsing(self):
        """Test FunctionCall parsing."""
        from proxima.intelligence.llm_router import FunctionCall
        
        func_call = FunctionCall(
            name="get_quantum_state",
            arguments={"qubit_id": 0},
            id="call_123",
        )
        
        assert func_call.name == "get_quantum_state"
        assert func_call.arguments["qubit_id"] == 0
        assert func_call.id == "call_123"


# =============================================================================
# Provider Error Handling Tests
# =============================================================================


class TestProviderErrorHandling:
    """Tests for provider error handling."""

    def test_provider_missing_api_key(self):
        """Test provider handles missing API key gracefully."""
        from proxima.intelligence.llm_router import OpenAIProvider, LLMRequest
        
        provider = OpenAIProvider()
        request = LLMRequest(prompt="Test prompt")
        
        response = provider.send(request, api_key=None)
        
        assert response.error is not None
        assert "API key" in response.error

    def test_unknown_provider_raises_error(self):
        """Test getting unknown provider raises error."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        
        with pytest.raises(ValueError, match="Unknown provider"):
            registry.get("nonexistent_provider")

    def test_none_provider_raises_error(self):
        """Test getting 'none' provider raises error."""
        from proxima.intelligence.llm_router import ProviderRegistry
        
        registry = ProviderRegistry()
        
        with pytest.raises(ValueError, match="No LLM provider configured"):
            registry.get("none")


# =============================================================================
# Insight Engine Tests
# =============================================================================


class TestInsightEngine:
    """Tests for insight engine functionality."""

    def test_insight_engine_creation(self):
        """Test InsightEngine can be created."""
        from proxima.intelligence.insights import InsightEngine
        
        engine = InsightEngine()
        
        assert engine is not None

    def test_statistical_analyzer_creation(self):
        """Test StatisticalAnalyzer can be created."""
        from proxima.intelligence.insights import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer()
        
        assert analyzer is not None

    def test_amplitude_analyzer_creation(self):
        """Test AmplitudeAnalyzer can be created."""
        from proxima.intelligence.insights import AmplitudeAnalyzer
        
        analyzer = AmplitudeAnalyzer()
        
        assert analyzer is not None

    def test_enhanced_llm_analyzer_creation(self):
        """Test EnhancedLLMAnalyzer can be created."""
        from proxima.intelligence.insights import EnhancedLLMAnalyzer
        
        analyzer = EnhancedLLMAnalyzer()
        
        assert analyzer is not None
        assert analyzer.available is False  # No callback provided

    def test_enhanced_llm_analyzer_with_callback(self, mock_llm_response):
        """Test EnhancedLLMAnalyzer with LLM callback."""
        from proxima.intelligence.insights import EnhancedLLMAnalyzer
        
        def mock_callback(prompt: str) -> str:
            return mock_llm_response
        
        analyzer = EnhancedLLMAnalyzer(llm_callback=mock_callback)
        
        assert analyzer.available is True

    def test_pattern_type_enum(self):
        """Test PatternType enum values."""
        from proxima.intelligence.insights import PatternType
        
        expected_patterns = [
            "UNIFORM",
            "PEAKED",
            "BIMODAL",
            "EXPONENTIAL",
            "OSCILLATING",
            "RANDOM",
            "ENTANGLED",
            "GHZ",
            "SPARSE",
        ]
        
        for pattern in expected_patterns:
            assert hasattr(PatternType, pattern)

    def test_insight_level_enum(self):
        """Test InsightLevel enum values."""
        from proxima.intelligence.insights import InsightLevel
        
        assert hasattr(InsightLevel, "BASIC")
        assert hasattr(InsightLevel, "STANDARD")
        assert hasattr(InsightLevel, "DETAILED")
        assert hasattr(InsightLevel, "EXPERT")


# =============================================================================
# Multi-Backend Comparison Tests
# =============================================================================


class TestMultiBackendComparison:
    """Tests for multi-backend comparison workflows."""

    def test_comparison_with_mock_backends(self, multi_backend_results):
        """Test comparison across multiple backends."""
        # Calculate fidelity between backends
        def calculate_tvd(counts1, counts2, shots=1000):
            """Calculate total variation distance."""
            all_states = set(counts1.keys()) | set(counts2.keys())
            total = 0
            for state in all_states:
                p1 = counts1.get(state, 0) / shots
                p2 = counts2.get(state, 0) / shots
                total += abs(p1 - p2)
            return total / 2
        
        cirq_counts = multi_backend_results["cirq"]["counts"]
        qiskit_counts = multi_backend_results["qiskit_aer"]["counts"]
        
        tvd = calculate_tvd(cirq_counts, qiskit_counts)
        
        # TVD should be small for similar results
        assert tvd < 0.1

    def test_execution_time_comparison(self, multi_backend_results):
        """Test comparing execution times across backends."""
        times = {
            name: result["execution_time_ms"]
            for name, result in multi_backend_results.items()
        }
        
        fastest = min(times, key=times.get)
        slowest = max(times, key=times.get)
        
        assert times[fastest] < times[slowest]


# =============================================================================
# LLM Router Integration Tests
# =============================================================================


class TestLLMRouterIntegration:
    """Integration tests for LLM router."""

    def test_local_llm_detector_creation(self):
        """Test LocalLLMDetector can be created."""
        from proxima.intelligence.llm_router import LocalLLMDetector
        
        detector = LocalLLMDetector(timeout_s=2.0)
        
        assert detector is not None
        assert detector.timeout == 2.0

    def test_api_key_manager_creation(self):
        """Test APIKeyManager can be created."""
        from proxima.intelligence.llm_router import APIKeyManager
        
        manager = APIKeyManager()
        
        assert manager is not None

    def test_consent_gate_creation(self):
        """Test ConsentGate can be created."""
        from proxima.intelligence.llm_router import ConsentGate
        
        gate = ConsentGate()
        
        assert gate is not None

    def test_llm_router_creation(self):
        """Test LLMRouter can be created."""
        from proxima.intelligence.llm_router import LLMRouter
        
        router = LLMRouter()
        
        assert router is not None


# =============================================================================
# Default Models Tests
# =============================================================================


class TestDefaultModels:
    """Tests for default model configurations."""

    def test_default_models_defined(self):
        """Test default models are defined for all providers."""
        from proxima.intelligence.llm_router import DEFAULT_MODELS
        
        expected_providers = [
            "openai",
            "anthropic",
            "ollama",
            "lmstudio",
            "llama_cpp",
            "together",
            "groq",
            "mistral",
            "azure_openai",
            "cohere",
        ]
        
        for provider in expected_providers:
            assert provider in DEFAULT_MODELS, f"Missing default model for {provider}"
            assert DEFAULT_MODELS[provider], f"Empty default model for {provider}"

    def test_api_bases_defined(self):
        """Test API bases are defined for cloud providers."""
        from proxima.intelligence.llm_router import API_BASES
        
        cloud_providers = [
            "openai",
            "anthropic",
            "together",
            "groq",
            "mistral",
            "cohere",
        ]
        
        for provider in cloud_providers:
            assert provider in API_BASES, f"Missing API base for {provider}"
            assert API_BASES[provider].startswith("https://"), f"Invalid API base for {provider}"
