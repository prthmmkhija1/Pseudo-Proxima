"""
AI/ML Intelligence Module.

Phase 3: Intelligence & Decision Systems
- LLM routing (local/remote with consent)
- Backend auto-selection with explanation
- Insight engine for result interpretation
"""

from proxima.intelligence.insights import (
    AmplitudeAnalysis,
    InsightEngine,
    InsightLevel,
    InsightReport,
    PatternInfo,
    PatternType,
    Recommendation,
    StatisticalMetrics,
    Visualization,
    analyze_results,
    summarize_results,
)
from proxima.intelligence.llm_router import (
    AnthropicProvider,
    APIKeyManager,
    ConsentGate,
    LlamaCppProvider,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRouter,
    LMStudioProvider,
    LocalLLMDetector,
    LocalLLMStatus,
    OllamaProvider,
    OpenAIProvider,
    ProviderRegistry,
    quick_prompt,
)
from proxima.intelligence.selector import (
    BackendCapabilities,
    BackendRegistry,
    BackendSelector,
    BackendType,
    CircuitCharacteristics,
    SelectionResult,
    SelectionScore,
    SelectionStrategy,
    analyze_circuit,
    select_backend,
)

__all__ = [
    # LLM Router - Core
    "LLMRouter",
    "LLMRequest",
    "LLMResponse",
    "LLMProvider",
    "ProviderRegistry",
    # LLM Router - Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LMStudioProvider",
    "LlamaCppProvider",
    # LLM Router - Utilities
    "LocalLLMDetector",
    "LocalLLMStatus",
    "APIKeyManager",
    "ConsentGate",
    "quick_prompt",
    # Backend Selector - Core
    "BackendSelector",
    "BackendRegistry",
    "BackendCapabilities",
    "BackendType",
    # Backend Selector - Results
    "SelectionResult",
    "SelectionScore",
    "SelectionStrategy",
    "CircuitCharacteristics",
    # Backend Selector - Functions
    "select_backend",
    "analyze_circuit",
    # Insight Engine - Core
    "InsightEngine",
    "InsightReport",
    "InsightLevel",
    # Insight Engine - Data Classes
    "StatisticalMetrics",
    "AmplitudeAnalysis",
    "PatternInfo",
    "PatternType",
    "Recommendation",
    "Visualization",
    # Insight Engine - Functions
    "analyze_results",
    "summarize_results",
]
