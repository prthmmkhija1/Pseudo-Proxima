# LLM Router API Reference

Complete API documentation for the `proxima.intelligence.llm_router` module.

## Overview

The LLM Router provides intelligent routing and management of Large Language Model providers, with features for:

- **Multi-Provider Support**: 10+ LLM providers (OpenAI, Anthropic, Google, Cohere, etc.)
- **Automatic Failover**: Seamless switching between providers on failure
- **Consent Management**: User consent for LLM usage with persistent preferences
- **API Key Management**: Secure handling of provider credentials
- **Streaming Support**: Real-time streaming responses
- **Tool Calling**: Function calling support for compatible providers

---

## Enums and Types

### ProviderName

Available LLM provider identifiers.

```python
from proxima.intelligence.llm_router import ProviderName

# Type alias for provider names
ProviderName = Literal[
    "openai",           # OpenAI (GPT-4, GPT-3.5)
    "anthropic",        # Anthropic (Claude)
    "google",           # Google AI (Gemini)
    "local",            # Local models (Ollama, etc.)
    "mock",             # Mock provider for testing
    "together",         # Together AI
    "groq",             # Groq (LPU inference)
    "mistral",          # Mistral AI
    "azure_openai",     # Azure OpenAI Service
    "cohere",           # Cohere
]
```

### ConsentState

User consent states for LLM usage.

```python
from proxima.intelligence.llm_router import ConsentState

class ConsentState(str, Enum):
    GRANTED = "granted"       # User has granted consent
    DENIED = "denied"         # User has denied consent
    PENDING = "pending"       # Consent not yet requested
    EXPIRED = "expired"       # Consent has expired
    REVOKED = "revoked"       # User revoked previous consent
```

### InsightLevel

Levels of detail for LLM-generated insights.

```python
from proxima.intelligence.llm_router import InsightLevel

class InsightLevel(str, Enum):
    BASIC = "basic"           # Simple explanations
    INTERMEDIATE = "intermediate"  # Moderate detail
    ADVANCED = "advanced"     # Technical deep-dives
    EXPERT = "expert"         # Research-level analysis
```

### PatternType

Types of patterns detected in quantum results.

```python
from proxima.intelligence.llm_router import PatternType

class PatternType(str, Enum):
    ENTANGLEMENT = "entanglement"
    SUPERPOSITION = "superposition"
    INTERFERENCE = "interference"
    DECOHERENCE = "decoherence"
    NOISE = "noise"
    PERIODIC = "periodic"
    ANOMALY = "anomaly"
```

---

## Core Classes

### LLMRequest

Request object for LLM interactions.

```python
from proxima.intelligence.llm_router import LLMRequest

@dataclass
class LLMRequest:
    """Request to an LLM provider."""
    
    prompt: str
    """The prompt/query to send to the LLM."""
    
    system_prompt: str | None = None
    """Optional system prompt for context."""
    
    model: str | None = None
    """Specific model to use (provider default if None)."""
    
    temperature: float = 0.7
    """Sampling temperature (0.0-2.0)."""
    
    max_tokens: int = 1000
    """Maximum tokens in response."""
    
    stream: bool = False
    """Enable streaming response."""
    
    tools: list[dict] | None = None
    """Tool/function definitions for function calling."""
    
    tool_choice: str | None = None
    """Tool selection mode: 'auto', 'required', or specific tool name."""
    
    metadata: dict[str, Any] | None = None
    """Additional request metadata."""
```

### LLMResponse

Response object from LLM interactions.

```python
from proxima.intelligence.llm_router import LLMResponse

@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    
    content: str
    """The generated text content."""
    
    model: str
    """Model that generated the response."""
    
    provider: str
    """Provider that handled the request."""
    
    usage: TokenUsage | None = None
    """Token usage statistics."""
    
    finish_reason: str | None = None
    """Reason for completion (stop, length, tool_calls, etc.)."""
    
    tool_calls: list[ToolCall] | None = None
    """Function/tool calls in the response."""
    
    latency_ms: float | None = None
    """Response latency in milliseconds."""
    
    cached: bool = False
    """Whether response was from cache."""


@dataclass
class TokenUsage:
    """Token usage statistics."""
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass  
class ToolCall:
    """A tool/function call from the LLM."""
    
    id: str
    name: str
    arguments: dict[str, Any]
```

---

## Provider Classes

### BaseProvider

Abstract base class for LLM providers.

```python
from proxima.intelligence.llm_router import BaseProvider

class BaseProvider(ABC):
    """Base class for LLM providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this provider."""
        pass
    
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion.
        
        Args:
            request: LLMRequest with prompt and parameters
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            ProviderError: If completion fails
        """
        pass
    
    @abstractmethod
    async def stream(
        self, 
        request: LLMRequest
    ) -> AsyncIterator[str]:
        """
        Stream a completion token by token.
        
        Args:
            request: LLMRequest with prompt and parameters
            
        Yields:
            String tokens as they are generated
        """
        pass
    
    async def health_check(self) -> bool:
        """Check if provider is available."""
        return True
    
    def supports_tools(self) -> bool:
        """Check if provider supports function calling."""
        return False
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return True
```

### OpenAIProvider

OpenAI API provider implementation.

```python
from proxima.intelligence.llm_router import OpenAIProvider

class OpenAIProvider(BaseProvider):
    """OpenAI API provider (GPT-4, GPT-3.5)."""
    
    name = "openai"
    default_model = "gpt-4-turbo-preview"
    
    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (or OPENAI_API_KEY env var)
            organization: Optional organization ID
            base_url: Optional custom API endpoint
        """
    
    def supports_tools(self) -> bool:
        return True  # GPT-4 supports function calling
```

### AnthropicProvider

Anthropic Claude provider implementation.

```python
from proxima.intelligence.llm_router import AnthropicProvider

class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""
    
    name = "anthropic"
    default_model = "claude-3-opus-20240229"
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
            base_url: Optional custom API endpoint
        """
    
    def supports_tools(self) -> bool:
        return True  # Claude 3 supports tool use
```

### GoogleProvider

Google AI (Gemini) provider implementation.

```python
from proxima.intelligence.llm_router import GoogleProvider

class GoogleProvider(BaseProvider):
    """Google AI Gemini provider."""
    
    name = "google"
    default_model = "gemini-pro"
    
    def __init__(
        self,
        api_key: str | None = None,
    ):
        """
        Initialize Google AI provider.
        
        Args:
            api_key: Google AI API key (or GOOGLE_API_KEY env var)
        """
```

### GroqProvider

Groq LPU inference provider.

```python
from proxima.intelligence.llm_router import GroqProvider

class GroqProvider(BaseProvider):
    """Groq LPU inference provider for ultra-fast inference."""
    
    name = "groq"
    default_model = "llama-3.1-70b-versatile"
    
    def __init__(
        self,
        api_key: str | None = None,
    ):
        """
        Initialize Groq provider.
        
        Args:
            api_key: Groq API key (or GROQ_API_KEY env var)
        """
    
    def supports_tools(self) -> bool:
        return True
    
    def supports_streaming(self) -> bool:
        return True
```

### CohereProvider

Cohere enterprise NLP provider.

```python
from proxima.intelligence.llm_router import CohereProvider

class CohereProvider(BaseProvider):
    """Cohere enterprise NLP provider with RAG support."""
    
    name = "cohere"
    default_model = "command-r-plus"
    
    def __init__(
        self,
        api_key: str | None = None,
    ):
        """
        Initialize Cohere provider.
        
        Args:
            api_key: Cohere API key (or COHERE_API_KEY env var)
        """
    
    async def complete_with_rag(
        self,
        request: LLMRequest,
        documents: list[str],
    ) -> LLMResponse:
        """
        Generate completion with RAG (Retrieval Augmented Generation).
        
        Args:
            request: LLM request
            documents: List of document texts for context
            
        Returns:
            LLMResponse with grounded generation
        """
```

---

## Management Classes

### ProviderRegistry

Registry for LLM providers.

```python
from proxima.intelligence.llm_router import ProviderRegistry

class ProviderRegistry:
    """Registry for managing LLM providers."""
    
    def register(
        self, 
        name: str, 
        provider_class: type[BaseProvider],
    ) -> None:
        """
        Register a provider class.
        
        Args:
            name: Provider identifier
            provider_class: Provider class (not instance)
        """
    
    def get(self, name: str) -> BaseProvider | None:
        """
        Get a provider instance by name.
        
        Args:
            name: Provider identifier
            
        Returns:
            Provider instance or None if not found
        """
    
    def list_providers(self) -> list[str]:
        """List all registered provider names."""
    
    def list_available(self) -> list[str]:
        """List providers with valid API keys configured."""
    
    def get_default(self) -> BaseProvider:
        """Get the default provider (first available)."""


# Global registry instance
provider_registry = ProviderRegistry()
```

### APIKeyManager

Secure API key management.

```python
from proxima.intelligence.llm_router import APIKeyManager

class APIKeyManager:
    """Manages API keys for LLM providers."""
    
    def __init__(self, config_path: Path | None = None):
        """
        Initialize key manager.
        
        Args:
            config_path: Path to encrypted key storage
        """
    
    def set_key(self, provider: str, key: str) -> None:
        """
        Store an API key securely.
        
        Args:
            provider: Provider name
            key: API key value
        """
    
    def get_key(self, provider: str) -> str | None:
        """
        Retrieve an API key.
        
        Args:
            provider: Provider name
            
        Returns:
            API key or None if not set
        """
    
    def has_key(self, provider: str) -> bool:
        """Check if a key exists for provider."""
    
    def delete_key(self, provider: str) -> bool:
        """Delete a stored key."""
    
    def list_configured(self) -> list[str]:
        """List providers with configured keys."""
```

### ConsentGate

User consent management for LLM usage.

```python
from proxima.intelligence.llm_router import ConsentGate, ConsentState

class ConsentGate:
    """Manages user consent for LLM usage."""
    
    def __init__(
        self,
        storage_path: Path | None = None,
        expiry_days: int = 30,
    ):
        """
        Initialize consent gate.
        
        Args:
            storage_path: Path for persistent consent storage
            expiry_days: Days until consent expires
        """
    
    def check_consent(self, feature: str = "llm") -> ConsentState:
        """
        Check current consent state.
        
        Args:
            feature: Feature requiring consent
            
        Returns:
            Current ConsentState
        """
    
    def request_consent(
        self, 
        feature: str = "llm",
        description: str | None = None,
    ) -> bool:
        """
        Request user consent (interactive).
        
        Args:
            feature: Feature requiring consent
            description: Optional description of what consent enables
            
        Returns:
            True if consent granted
        """
    
    def grant_consent(
        self, 
        feature: str = "llm",
        duration_days: int | None = None,
    ) -> None:
        """Grant consent programmatically."""
    
    def revoke_consent(self, feature: str = "llm") -> None:
        """Revoke previously granted consent."""
    
    def is_granted(self, feature: str = "llm") -> bool:
        """Quick check if consent is currently granted."""
```

---

## Router Class

### LLMRouter

Main router for LLM requests with failover support.

```python
from proxima.intelligence.llm_router import LLMRouter

class LLMRouter:
    """
    Intelligent LLM request router with failover.
    
    Features:
    - Automatic provider selection based on availability
    - Failover to backup providers on error
    - Request caching for repeated queries
    - Rate limiting and quota management
    """
    
    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        key_manager: APIKeyManager | None = None,
        consent_gate: ConsentGate | None = None,
        cache_enabled: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM router.
        
        Args:
            registry: Provider registry (uses global if None)
            key_manager: API key manager
            consent_gate: Consent manager
            cache_enabled: Enable response caching
            max_retries: Maximum retry attempts per request
        """
    
    async def route(
        self,
        request: LLMRequest,
        preferred_provider: str | None = None,
        fallback_providers: list[str] | None = None,
    ) -> LLMResponse:
        """
        Route a request to the best available provider.
        
        Args:
            request: LLM request to route
            preferred_provider: First-choice provider
            fallback_providers: Backup providers if preferred fails
            
        Returns:
            LLMResponse from successful provider
            
        Raises:
            RouterError: If all providers fail
            ConsentError: If LLM consent not granted
        """
    
    async def stream_route(
        self,
        request: LLMRequest,
        preferred_provider: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Route a streaming request.
        
        Args:
            request: LLM request (stream=True is set automatically)
            preferred_provider: Preferred provider
            
        Yields:
            Token strings as they are generated
        """
    
    def get_available_providers(self) -> list[str]:
        """List currently available providers."""
    
    def get_provider_status(self, provider: str) -> dict[str, Any]:
        """Get detailed status for a provider."""
```

---

## Insight Engine

### InsightEngine

LLM-powered insight generation for quantum results.

```python
from proxima.intelligence.llm_router import InsightEngine, InsightLevel

class InsightEngine:
    """Generate insights from quantum simulation results using LLMs."""
    
    def __init__(
        self,
        router: LLMRouter | None = None,
        default_level: InsightLevel = InsightLevel.INTERMEDIATE,
    ):
        """
        Initialize insight engine.
        
        Args:
            router: LLM router for requests
            default_level: Default detail level for insights
        """
    
    async def analyze_results(
        self,
        results: dict[str, Any],
        level: InsightLevel | None = None,
    ) -> dict[str, Any]:
        """
        Generate insights from simulation results.
        
        Args:
            results: Simulation results dictionary
            level: Detail level for insights
            
        Returns:
            Dictionary with:
            - summary: Brief summary of results
            - patterns: Detected patterns
            - recommendations: Suggested next steps
            - explanation: Detailed explanation
        """
    
    async def explain_circuit(
        self,
        circuit_description: str,
        level: InsightLevel | None = None,
    ) -> str:
        """
        Generate circuit explanation.
        
        Args:
            circuit_description: Circuit in text/QASM format
            level: Detail level
            
        Returns:
            Human-readable explanation
        """
    
    async def compare_backends(
        self,
        comparison_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Analyze multi-backend comparison results.
        
        Args:
            comparison_results: Results from BackendComparator
            
        Returns:
            Analysis with recommendations
        """
    
    def detect_patterns(
        self,
        counts: dict[str, int],
    ) -> list[PatternType]:
        """
        Detect quantum patterns in measurement results.
        
        Args:
            counts: Measurement count dictionary
            
        Returns:
            List of detected pattern types
        """
```

---

## Usage Examples

### Basic Completion

```python
from proxima.intelligence.llm_router import (
    LLMRouter, 
    LLMRequest,
    provider_registry,
)

async def get_explanation():
    router = LLMRouter()
    
    request = LLMRequest(
        prompt="Explain quantum entanglement in simple terms",
        temperature=0.7,
        max_tokens=500,
    )
    
    response = await router.route(request)
    print(f"Provider: {response.provider}")
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage.total_tokens}")
```

### Streaming Response

```python
async def stream_explanation():
    router = LLMRouter()
    
    request = LLMRequest(
        prompt="Describe the Bell state preparation circuit step by step",
        stream=True,
    )
    
    async for token in router.stream_route(request):
        print(token, end="", flush=True)
    print()  # Newline at end
```

### Function Calling

```python
async def use_tools():
    router = LLMRouter()
    
    tools = [{
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "Run a quantum circuit simulation",
            "parameters": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"},
                    "backend": {"type": "string"},
                    "shots": {"type": "integer"}
                },
                "required": ["circuit"]
            }
        }
    }]
    
    request = LLMRequest(
        prompt="Run a Bell state simulation with 1000 shots",
        tools=tools,
        tool_choice="auto",
    )
    
    response = await router.route(request, preferred_provider="openai")
    
    if response.tool_calls:
        for call in response.tool_calls:
            print(f"Tool: {call.name}")
            print(f"Args: {call.arguments}")
```

### Insight Generation

```python
from proxima.intelligence.llm_router import InsightEngine, InsightLevel

async def analyze_simulation():
    engine = InsightEngine()
    
    results = {
        "counts": {"00": 480, "11": 520},
        "execution_time": 0.15,
        "backend": "cirq",
        "num_qubits": 2,
    }
    
    insights = await engine.analyze_results(
        results,
        level=InsightLevel.ADVANCED,
    )
    
    print(f"Summary: {insights['summary']}")
    print(f"Patterns: {insights['patterns']}")
    print(f"Recommendations: {insights['recommendations']}")
```

---

## Error Handling

### Exception Classes

```python
from proxima.intelligence.llm_router import (
    LLMError,
    ProviderError,
    RouterError,
    ConsentError,
    RateLimitError,
    APIKeyError,
)

# Base exception
class LLMError(Exception):
    """Base exception for LLM operations."""
    pass

# Provider-specific error
class ProviderError(LLMError):
    """Error from a specific LLM provider."""
    provider: str
    original_error: Exception | None

# Router error (all providers failed)
class RouterError(LLMError):
    """Error when routing fails for all providers."""
    attempted_providers: list[str]

# Consent not granted
class ConsentError(LLMError):
    """Error when LLM consent is not granted."""
    feature: str

# Rate limit exceeded
class RateLimitError(ProviderError):
    """Rate limit exceeded for provider."""
    retry_after: float | None

# API key issues
class APIKeyError(ProviderError):
    """API key missing or invalid."""
    pass
```

### Error Handling Example

```python
from proxima.intelligence.llm_router import (
    LLMRouter,
    LLMRequest,
    RouterError,
    ConsentError,
    RateLimitError,
)

async def safe_completion():
    router = LLMRouter()
    request = LLMRequest(prompt="Hello, world!")
    
    try:
        response = await router.route(request)
        return response.content
    except ConsentError:
        print("Please grant LLM consent first")
        return None
    except RateLimitError as e:
        print(f"Rate limited, retry after {e.retry_after}s")
        return None
    except RouterError as e:
        print(f"All providers failed: {e.attempted_providers}")
        return None
```

---

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `GOOGLE_API_KEY` | Google AI API key | `AI...` |
| `GROQ_API_KEY` | Groq API key | `gsk_...` |
| `COHERE_API_KEY` | Cohere API key | `...` |
| `TOGETHER_API_KEY` | Together AI key | `...` |
| `MISTRAL_API_KEY` | Mistral AI key | `...` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | `...` |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint | `https://....openai.azure.com` |
| `PROXIMA_LLM_CACHE_DIR` | Cache directory | `~/.proxima/llm_cache` |

### YAML Configuration

```yaml
# proxima.yaml
llm:
  default_provider: openai
  fallback_providers:
    - anthropic
    - groq
  
  cache:
    enabled: true
    ttl_seconds: 3600
    max_size_mb: 100
  
  consent:
    required: true
    expiry_days: 30
  
  providers:
    openai:
      model: gpt-4-turbo-preview
      temperature: 0.7
      max_tokens: 2000
    
    anthropic:
      model: claude-3-opus-20240229
      temperature: 0.7
    
    groq:
      model: llama-3.1-70b-versatile
```
