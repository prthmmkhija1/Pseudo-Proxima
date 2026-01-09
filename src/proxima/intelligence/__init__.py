"""
AI/ML components module.

Phase 3: Intelligence & Decision Systems
- LLM routing (local/remote with consent)
- Backend auto-selection with explanation
- Insight engine for result interpretation
"""

from proxima.intelligence.insights import (
    InsightEngine,
    InsightReport,
    StatisticalMetrics,
)
from proxima.intelligence.llm_router import (
    AnthropicProvider,
    APIKeyManager,
    ConsentGate,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRouter,
    LMStudioProvider,
    LocalLLMDetector,
    OllamaProvider,
    OpenAIProvider,
    ProviderName,
    ProviderRegistry,
    build_router,
)
from proxima.intelligence.selector import (
    BackendScore,
    BackendSelector,
    SelectionInput,
    SelectionResult,
)

__all__ = [
    # LLM Router
    "LLMRouter",
    "LLMRequest",
    "LLMResponse",
    "LLMProvider",
    "ProviderRegistry",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LMStudioProvider",
    "LocalLLMDetector",
    "APIKeyManager",
    "ConsentGate",
    "build_router",
    "ProviderName",
    # Backend Selector
    "BackendSelector",
    "SelectionResult",
    "SelectionInput",
    "BackendScore",
    # Insight Engine
    "InsightEngine",
    "InsightReport",
    "StatisticalMetrics",
]
