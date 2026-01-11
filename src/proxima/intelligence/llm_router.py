"""LLM routing and provider abstraction.

Implements the Step 3.1 architecture from proper_implementation_steps.md:
- ProviderRegistry with pluggable providers (OpenAI, Anthropic, Ollama, LM Studio)
- LocalLLMDetector for lightweight endpoint checks
- APIKeyManager for pulling keys from configured env vars
- ConsentGate to enforce local/remote consent rules
- LLMRouter that chooses the provider, enforces consent, and dispatches requests

This module provides REAL LLM provider implementations that make actual API calls.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from proxima.config.settings import Settings, config_service

ProviderName = Literal["openai", "anthropic", "ollama", "lmstudio", "llama_cpp", "none"]

# Default ports for local LLM servers
DEFAULT_PORTS: dict[str, int] = {
    "ollama": 11434,
    "lmstudio": 1234,
    "llama_cpp": 8080,
}

# Default models for each provider
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4",
    "anthropic": "claude-3-sonnet-20240229",
    "ollama": "llama2",
    "lmstudio": "local-model",
    "llama_cpp": "local-model",
}


@dataclass
class LLMRequest:
    """Request to send to an LLM provider."""

    prompt: str
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    provider: ProviderName | None = None
    system_prompt: str | None = None
    stream: bool = False  # Enable streaming response
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str
    provider: ProviderName
    model: str
    latency_ms: float
    tokens_used: int = 0
    raw: dict | None = None
    error: str | None = None
    is_streaming: bool = False  # Whether this is a streaming response
    stream_chunks: list[str] = field(default_factory=list)  # Collected stream chunks


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    name: ProviderName
    is_local: bool
    requires_api_key: bool

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:  # pragma: no cover
        """Send a request to the LLM provider."""
        ...

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:  # pragma: no cover
        """Send a streaming request to the LLM provider."""
        ...

    def health_check(self, endpoint: str | None = None) -> bool:  # pragma: no cover
        """Check if the provider is available."""
        ...


class _BaseProvider:
    """Base class for LLM providers."""

    name: ProviderName
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, default_model: str, timeout: float = 60.0):
        self.default_model = default_model
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to the LLM provider. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement send()")

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def health_check(self, endpoint: str | None = None) -> bool:
        """Check if the provider is reachable."""
        return True  # Override in subclasses

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Default streaming implementation - falls back to non-streaming."""
        # Default implementation: just call send() and simulate streaming
        response = self.send(request, api_key)
        if callback and response.text:
            # Simulate streaming by sending the full text
            callback(response.text)
        return LLMResponse(
            text=response.text,
            provider=response.provider,
            model=response.model,
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used,
            raw=response.raw,
            error=response.error,
            is_streaming=True,
            stream_chunks=[response.text] if response.text else [],
        )


class OpenAIProvider(_BaseProvider):
    """OpenAI API provider (GPT-4, GPT-3.5, etc.)."""

    name: ProviderName = "openai"
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__(DEFAULT_MODELS["openai"], timeout)
        self._api_base = "https://api.openai.com/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to OpenAI API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for OpenAI",
            )

        model = request.model or self.default_model
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
            )
        except httpx.HTTPStatusError as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def health_check(self, endpoint: str | None = None) -> bool:
        """Check if OpenAI API is reachable."""
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={"Authorization": "Bearer test"},
                timeout=5.0,
            )
            # 401 means API is reachable but key is invalid - that's OK for health check
            return response.status_code in (200, 401)
        except Exception:
            return False

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to OpenAI API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for OpenAI",
            )

        model = request.model or self.default_model
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            chunks: list[str] = []
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self._api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    chunks.append(content)
                                    if callback:
                                        callback(content)
                            except json.JSONDecodeError:
                                continue

            elapsed = (time.perf_counter() - start) * 1000
            full_text = "".join(chunks)

            return LLMResponse(
                text=full_text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                is_streaming=True,
                stream_chunks=chunks,
            )
        except httpx.HTTPStatusError as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


class AnthropicProvider(_BaseProvider):
    """Anthropic API provider (Claude models)."""

    name: ProviderName = "anthropic"
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__(DEFAULT_MODELS["anthropic"], timeout)
        self._api_base = "https://api.anthropic.com/v1"
        self._api_version = "2023-06-01"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Anthropic API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Anthropic",
            )

        model = request.model or self.default_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or 4096,
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        if request.temperature > 0:
            payload["temperature"] = request.temperature

        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_base}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": self._api_version,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            text = data["content"][0]["text"] if data.get("content") else ""
            tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get(
                "output_tokens", 0
            )

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
            )
        except httpx.HTTPStatusError as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def health_check(self, endpoint: str | None = None) -> bool:
        """Check if Anthropic API is reachable."""
        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_base}/messages",
                headers={
                    "x-api-key": "test",
                    "anthropic-version": self._api_version,
                    "Content-Type": "application/json",
                },
                json={"model": "test", "max_tokens": 1, "messages": []},
                timeout=5.0,
            )
            # 401 means API is reachable but key is invalid
            return response.status_code in (200, 400, 401)
        except Exception:
            return False


class OllamaProvider(_BaseProvider):
    """Ollama local LLM provider."""

    name: ProviderName = "ollama"
    is_local: bool = True
    requires_api_key: bool = False

    def __init__(self, timeout: float = 120.0):
        super().__init__(DEFAULT_MODELS["ollama"], timeout)
        self._endpoint: str | None = None

    def send(self, request: LLMRequest, api_key: str | None = None) -> LLMResponse:
        """Send a request to Ollama server."""
        start = time.perf_counter()

        endpoint = self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        model = request.model or self.default_model

        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "stream": False,
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        options: dict[str, Any] = {}
        if request.temperature > 0:
            options["temperature"] = request.temperature
        if request.max_tokens:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options

        try:
            client = self._get_client()
            response = client.post(
                f"{endpoint}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            text = data.get("response", "")

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=data.get("eval_count", 0),
                raw=data,
            )
        except httpx.ConnectError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"Cannot connect to Ollama at {endpoint}. Is it running?",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def health_check(self, endpoint: str | None = None) -> bool:
        """Check if Ollama is running and reachable."""
        check_endpoint = endpoint or self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def set_endpoint(self, endpoint: str) -> None:
        """Set the Ollama endpoint."""
        self._endpoint = endpoint

    def list_models(self, endpoint: str | None = None) -> list[str]:
        """List available models on the Ollama server."""
        check_endpoint = endpoint or self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []


class LMStudioProvider(_BaseProvider):
    """LM Studio local LLM provider (OpenAI-compatible API)."""

    name: ProviderName = "lmstudio"
    is_local: bool = True
    requires_api_key: bool = False

    def __init__(self, timeout: float = 120.0):
        super().__init__(DEFAULT_MODELS["lmstudio"], timeout)
        self._endpoint: str | None = None

    def send(self, request: LLMRequest, api_key: str | None = None) -> LLMResponse:
        """Send a request to LM Studio server (OpenAI-compatible)."""
        start = time.perf_counter()

        endpoint = self._endpoint or f"http://localhost:{DEFAULT_PORTS['lmstudio']}"
        model = request.model or self.default_model

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": False,
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            client = self._get_client()
            response = client.post(
                f"{endpoint}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            text = data["choices"][0]["message"]["content"] if data.get("choices") else ""
            tokens = data.get("usage", {}).get("total_tokens", 0)

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
            )
        except httpx.ConnectError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"Cannot connect to LM Studio at {endpoint}. Is it running?",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def health_check(self, endpoint: str | None = None) -> bool:
        """Check if LM Studio is running and reachable."""
        check_endpoint = (
            endpoint or self._endpoint or f"http://localhost:{DEFAULT_PORTS['lmstudio']}"
        )
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/v1/models", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def set_endpoint(self, endpoint: str) -> None:
        """Set the LM Studio endpoint."""
        self._endpoint = endpoint


class LlamaCppProvider(_BaseProvider):
    """llama.cpp server provider."""

    name: ProviderName = "llama_cpp"
    is_local: bool = True
    requires_api_key: bool = False

    def __init__(self, timeout: float = 120.0):
        super().__init__(DEFAULT_MODELS["llama_cpp"], timeout)
        self._endpoint: str | None = None

    def send(self, request: LLMRequest, api_key: str | None = None) -> LLMResponse:
        """Send a request to llama.cpp server."""
        start = time.perf_counter()

        endpoint = self._endpoint or f"http://localhost:{DEFAULT_PORTS['llama_cpp']}"
        model = request.model or self.default_model

        # Build the prompt with system prompt if provided
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"

        payload: dict[str, Any] = {
            "prompt": full_prompt,
            "temperature": request.temperature,
            "stream": False,
        }

        if request.max_tokens:
            payload["n_predict"] = request.max_tokens

        try:
            client = self._get_client()
            response = client.post(
                f"{endpoint}/completion",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            text = data.get("content", "")

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=data.get("tokens_evaluated", 0) + data.get("tokens_predicted", 0),
                raw=data,
            )
        except httpx.ConnectError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"Cannot connect to llama.cpp at {endpoint}. Is it running?",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def health_check(self, endpoint: str | None = None) -> bool:
        """Check if llama.cpp server is running and reachable."""
        check_endpoint = (
            endpoint or self._endpoint or f"http://localhost:{DEFAULT_PORTS['llama_cpp']}"
        )
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def set_endpoint(self, endpoint: str) -> None:
        """Set the llama.cpp endpoint."""
        self._endpoint = endpoint


class ProviderRegistry:
    """Registry of available LLM providers."""

    def __init__(self) -> None:
        self._providers: dict[ProviderName, LLMProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "ollama": OllamaProvider(),
            "lmstudio": LMStudioProvider(),
            "llama_cpp": LlamaCppProvider(),
        }

    def get(self, name: ProviderName) -> LLMProvider:
        """Get a provider by name."""
        if name == "none":
            raise ValueError("No LLM provider configured")
        provider = self._providers.get(name)
        if not provider:
            raise ValueError(f"Unknown provider: {name}")
        return provider

    def list_providers(self) -> list[ProviderName]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def register(self, name: ProviderName, provider: LLMProvider) -> None:
        """Register a custom provider."""
        self._providers[name] = provider

    def get_local_providers(self) -> list[LLMProvider]:
        """Get all local LLM providers."""
        return [p for p in self._providers.values() if p.is_local]

    def get_remote_providers(self) -> list[LLMProvider]:
        """Get all remote LLM providers."""
        return [p for p in self._providers.values() if not p.is_local]


@dataclass
class LocalLLMStatus:
    """Status of a local LLM endpoint."""

    provider: ProviderName
    available: bool
    endpoint: str
    models: list[str] = field(default_factory=list)
    error: str | None = None


class LocalLLMDetector:
    """Detects locally running LLM servers."""

    def __init__(self, timeout_s: float = 2.0):
        self.timeout = timeout_s
        self._cache: dict[ProviderName, LocalLLMStatus] = {}
        self._registry = ProviderRegistry()

    def detect(
        self,
        provider_name: ProviderName,
        configured_endpoint: str | None = None,
    ) -> str | None:
        """Detect if a local LLM server is running.

        Returns the endpoint if available, None otherwise.
        """
        if provider_name not in ("ollama", "lmstudio", "llama_cpp"):
            return None

        # Check configured endpoint first
        if configured_endpoint:
            provider = self._registry.get(provider_name)
            if hasattr(provider, "set_endpoint"):
                provider.set_endpoint(configured_endpoint)  # type: ignore
            if provider.health_check(configured_endpoint):
                return configured_endpoint

        # Try default port
        default_port = DEFAULT_PORTS.get(provider_name)
        if default_port:
            endpoint = f"http://localhost:{default_port}"
            provider = self._registry.get(provider_name)
            if provider.health_check(endpoint):
                return endpoint

        return None

    def detect_all(self) -> list[LocalLLMStatus]:
        """Detect all available local LLM servers."""
        results: list[LocalLLMStatus] = []

        for name in ("ollama", "lmstudio", "llama_cpp"):
            provider_name: ProviderName = name  # type: ignore
            default_port = DEFAULT_PORTS.get(name, 8080)
            endpoint = f"http://localhost:{default_port}"

            try:
                provider = self._registry.get(provider_name)
                available = provider.health_check(endpoint)

                models: list[str] = []
                if available and hasattr(provider, "list_models"):
                    models = provider.list_models(endpoint)  # type: ignore

                results.append(
                    LocalLLMStatus(
                        provider=provider_name,
                        available=available,
                        endpoint=endpoint,
                        models=models,
                    )
                )
            except Exception as e:
                results.append(
                    LocalLLMStatus(
                        provider=provider_name,
                        available=False,
                        endpoint=endpoint,
                        error=str(e),
                    )
                )

        return results

    def get_first_available(self) -> LocalLLMStatus | None:
        """Get the first available local LLM server."""
        for status in self.detect_all():
            if status.available:
                return status
        return None


class APIKeyManager:
    """Manages API keys for LLM providers."""

    # Environment variable names for each provider
    ENV_VAR_NAMES: dict[str, str] = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    def __init__(self, settings: Settings, storage_path: str | None = None):
        self.settings = settings
        self._storage_path = storage_path
        self._cache: dict[str, str] = {}

    def get_api_key(self, provider: LLMProvider) -> str | None:
        """Get the API key for a provider."""
        return self.get_key(provider.name)

    def get_key(self, provider_name: str) -> str | None:
        """Get the API key for a provider by name."""
        # Check cache first
        if provider_name in self._cache:
            return self._cache[provider_name]

        # Check environment variable
        env_var = self.ENV_VAR_NAMES.get(provider_name)
        if env_var:
            key = os.environ.get(env_var)
            if key:
                self._cache[provider_name] = key
                return key

        # Check settings
        if hasattr(self.settings.llm, "api_key") and self.settings.llm.api_key:
            self._cache[provider_name] = self.settings.llm.api_key
            return self.settings.llm.api_key

        # Check custom env var from settings
        if hasattr(self.settings.llm, "api_key_env_var") and self.settings.llm.api_key_env_var:
            key = os.environ.get(self.settings.llm.api_key_env_var)
            if key:
                self._cache[provider_name] = key
                return key

        # Check file storage
        return self._load_key_from_storage(provider_name)

    def store_key(self, provider_name: str, api_key: str) -> None:
        """Store an API key for a provider."""
        self._cache[provider_name] = api_key

        if self._storage_path:
            self._save_key_to_storage(provider_name, api_key)

    def _save_key_to_storage(self, provider_name: str, api_key: str) -> None:
        """Save API key to file storage."""
        if not self._storage_path:
            return

        storage_path = Path(self._storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            existing = {}
            if storage_path.exists():
                existing = json.loads(storage_path.read_text())
            existing[provider_name] = api_key
            storage_path.write_text(json.dumps(existing, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    def _load_key_from_storage(self, provider_name: str) -> str | None:
        """Load API key from file storage."""
        if not self._storage_path:
            return None

        storage_path = Path(self._storage_path)
        if not storage_path.exists():
            return None

        try:
            keys = json.loads(storage_path.read_text())
            return keys.get(provider_name)
        except (json.JSONDecodeError, OSError):
            return None

    def validate(self, provider: LLMProvider) -> None:
        """Validate that required API key is available."""
        return self.validate_key(provider.name, provider.requires_api_key)

    def validate_key(self, provider_name: str, requires_key: bool = True) -> None:
        """Validate API key for provider."""
        if not requires_key:
            return
        key = self.get_key(provider_name)
        if not key:
            raise ValueError(
                f"API key missing for provider {provider_name}. "
                f"Set the appropriate environment variable or use store_key()."
            )


class ConsentGate:
    """Manages user consent for LLM operations.

    Implements consent flow: Check remembered → Prompt → Approve/Remember/Deny
    """

    def __init__(
        self,
        settings: Settings,
        prompt_func: Callable[[str], bool] | None = None,
        storage_path: str | None = None,
    ):
        self.settings = settings
        self.prompt_func = prompt_func
        self._storage_path = storage_path
        self._session_consents: dict[str, bool] = {}
        self._persistent_consents: dict[str, bool] | None = None

    def check_consent(self, provider: LLMProvider) -> bool | None:
        """Check if consent exists without prompting.

        Returns True/False if consent decision exists, None if not yet decided.
        """
        # Check session consents first
        if provider.name in self._session_consents:
            return self._session_consents[provider.name]

        # Check auto-approve settings
        is_local = provider.is_local
        auto_allow = (
            self.settings.consent.auto_approve_local_llm
            if is_local
            else self.settings.consent.auto_approve_remote_llm
        )
        if auto_allow:
            return True

        # Check persistent consents
        persistent = self._load_persistent_consents()
        if provider.name in persistent:
            return persistent[provider.name]

        return None

    def remember_consent(
        self,
        provider: LLMProvider,
        allowed: bool,
        persistent: bool = False,
    ) -> None:
        """Remember consent decision for a provider.

        Args:
            provider: The LLM provider
            allowed: Whether consent was granted
            persistent: If True, persist across sessions
        """
        self._session_consents[provider.name] = allowed

        if persistent and self.settings.consent.remember_decisions:
            self._save_persistent_consent(provider.name, allowed)

    def revoke_consent(self, provider: LLMProvider) -> None:
        """Revoke previously given consent."""
        self._session_consents.pop(provider.name, None)
        self._remove_persistent_consent(provider.name)

    def _load_persistent_consents(self) -> dict[str, bool]:
        """Load persistent consent decisions from storage."""
        if self._persistent_consents is not None:
            return self._persistent_consents

        storage_path = self._get_storage_path()
        if not storage_path.exists():
            self._persistent_consents = {}
            return self._persistent_consents

        try:
            self._persistent_consents = json.loads(storage_path.read_text())
        except (json.JSONDecodeError, OSError):
            self._persistent_consents = {}

        return self._persistent_consents

    def _save_persistent_consent(self, provider_name: str, allowed: bool) -> None:
        """Save a persistent consent decision."""
        consents = self._load_persistent_consents()
        consents[provider_name] = allowed

        storage_path = self._get_storage_path()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_text(json.dumps(consents, indent=2))

    def _remove_persistent_consent(self, provider_name: str) -> None:
        """Remove a persistent consent decision."""
        consents = self._load_persistent_consents()
        if provider_name in consents:
            del consents[provider_name]
            storage_path = self._get_storage_path()
            storage_path.write_text(json.dumps(consents, indent=2))

    def _get_storage_path(self) -> Path:
        """Get the path for persistent consent storage."""
        if self._storage_path:
            return Path(self._storage_path)
        return Path.home() / ".proxima" / ".consents.json"

    def require_consent(self, provider: LLMProvider) -> None:
        """Require user consent before using a provider.

        Flow: Check remembered → Prompt if needed → Approve/Deny
        """
        if not self.settings.llm.require_consent:
            return

        # Check if already consented
        existing = self.check_consent(provider)
        if existing is True:
            return
        if existing is False:
            raise PermissionError(f"Previously denied consent for provider {provider.name}")

        # Need to prompt
        if not self.prompt_func:
            raise PermissionError(
                f"Consent required for {'local' if provider.is_local else 'remote'} provider {provider.name}"
            )

        allowed = self.prompt_func(
            f"Allow {'local' if provider.is_local else 'remote'} LLM provider '{provider.name}'?"
        )

        # Remember the decision for this session
        self.remember_consent(provider, allowed, persistent=False)

        if not allowed:
            raise PermissionError(f"User denied consent for provider {provider.name}")


class LLMRouter:
    """Routes LLM requests to appropriate providers with consent enforcement."""

    def __init__(
        self,
        settings: Settings | None = None,
        consent_prompt: Callable[[str], bool] | None = None,
    ):
        self.settings = settings or config_service.load()
        self.registry = ProviderRegistry()
        self.detector = LocalLLMDetector(timeout_s=2.0)
        self.api_keys = APIKeyManager(self.settings)
        self.consent_gate = ConsentGate(self.settings, prompt_func=consent_prompt)

    def _pick_provider(self, request: LLMRequest) -> LLMProvider:
        """Pick the appropriate provider for the request."""
        provider_name: ProviderName = request.provider or self.settings.llm.provider or "none"  # type: ignore
        if provider_name == "none":
            raise ValueError("No LLM provider configured. Set llm.provider in config or request.")
        return self.registry.get(provider_name)

    def _ensure_local_available(self, provider: LLMProvider) -> None:
        """Ensure local provider is available."""
        if not provider.is_local:
            return
        endpoint = self.detector.detect(
            provider.name,
            configured_endpoint=self.settings.llm.local_endpoint,
        )
        if not endpoint:
            raise ConnectionError(
                f"Local endpoint not reachable for provider {provider.name}. "
                "Tried configured and default ports."
            )
        # Set the detected endpoint on the provider
        if hasattr(provider, "set_endpoint"):
            provider.set_endpoint(endpoint)  # type: ignore

    def route(self, request: LLMRequest) -> LLMResponse:
        """Route a request to the appropriate provider.

        This method:
        1. Picks the provider based on request/settings
        2. Ensures local providers are available
        3. Enforces consent
        4. Validates API keys
        5. Sends the request
        """
        provider = self._pick_provider(request)
        self._ensure_local_available(provider)
        self.consent_gate.require_consent(provider)
        self.api_keys.validate(provider)
        api_key = self.api_keys.get_api_key(provider)
        return provider.send(request, api_key)

    def route_with_fallback(
        self,
        request: LLMRequest,
        fallback_providers: list[ProviderName] | None = None,
    ) -> LLMResponse:
        """Route with fallback to other providers on failure.

        Args:
            request: The LLM request
            fallback_providers: List of provider names to try on failure

        Returns:
            LLMResponse from the first successful provider
        """
        try:
            response = self.route(request)
            if not response.error:
                return response
        except Exception:
            pass

        # Try fallback providers
        fallback = fallback_providers or []
        for provider_name in fallback:
            try:
                request_copy = LLMRequest(
                    prompt=request.prompt,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    provider=provider_name,
                    system_prompt=request.system_prompt,
                    metadata=request.metadata,
                )
                response = self.route(request_copy)
                if not response.error:
                    return response
            except Exception:
                continue

        # All failed
        return LLMResponse(
            text="",
            provider="none",
            model="",
            latency_ms=0,
            error="All providers failed",
        )

    def get_available_providers(self) -> list[ProviderName]:
        """Get list of available providers (with valid config/keys)."""
        available: list[ProviderName] = []

        for name in self.registry.list_providers():
            try:
                provider = self.registry.get(name)

                # Check local availability
                if provider.is_local:
                    if self.detector.detect(name):
                        available.append(name)
                else:
                    # Check if API key is available
                    try:
                        self.api_keys.validate(provider)
                        available.append(name)
                    except ValueError:
                        pass
            except Exception:
                pass

        return available


def build_router(consent_prompt: Callable[[str], bool] | None = None) -> LLMRouter:
    """Helper to build a router with loaded settings."""
    return LLMRouter(settings=config_service.load(), consent_prompt=consent_prompt)


# Convenience functions for direct LLM calls
def quick_prompt(
    prompt: str,
    provider: ProviderName | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Quick LLM prompt without consent (for testing/scripts).

    Note: This bypasses consent checks. Use LLMRouter for production.
    """
    router = build_router()
    router.consent_gate.settings.llm.require_consent = False

    request = LLMRequest(
        prompt=prompt,
        provider=provider,
        model=model,
        temperature=temperature,
    )

    response = router.route(request)
    if response.error:
        raise RuntimeError(f"LLM error: {response.error}")
    return response.text
