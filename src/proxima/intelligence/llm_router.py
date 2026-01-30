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
import random
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import httpx

from proxima.config.settings import Settings, config_service

# Type variable for generic retry handler
T = TypeVar("T")

# Extended provider support
ProviderName = Literal[
    "openai", "anthropic", "ollama", "lmstudio", "llama_cpp",
    "together", "groq", "mistral", "azure_openai", "cohere", "none"
]

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
    # Extended providers
    "together": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "groq": "mixtral-8x7b-32768",
    "mistral": "mistral-large-latest",
    "azure_openai": "gpt-4",
    "cohere": "command-r-plus",
}

# API base URLs for cloud providers
API_BASES: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "cohere": "https://api.cohere.ai/v1",
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
    # Function calling support
    functions: list[dict[str, Any]] | None = None  # OpenAI-style function definitions
    tools: list[dict[str, Any]] | None = None  # OpenAI tools format
    tool_choice: str | dict[str, Any] | None = None  # "auto", "none", or specific tool


@dataclass
class FunctionCall:
    """Represents a function call returned by the LLM."""
    
    name: str
    arguments: dict[str, Any]
    id: str | None = None  # Tool call ID for OpenAI


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
    # Function calling response
    function_call: FunctionCall | None = None  # Deprecated: use tool_calls
    tool_calls: list[FunctionCall] = field(default_factory=list)  # Multiple tool calls
    finish_reason: str | None = None  # "stop", "function_call", "tool_calls", etc.


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    name: ProviderName
    is_local: bool
    requires_api_key: bool

    def send(
        self, request: LLMRequest, api_key: str | None
    ) -> LLMResponse:  # pragma: no cover
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

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from an OpenAI-compatible /v1/models endpoint.
        
        Override in subclasses for provider-specific implementations.
        
        Args:
            api_base: Base URL for the API (e.g., http://localhost:11434)
            api_key: API key if required
            
        Returns:
            List of model IDs/names available
        """
        # Default implementation for OpenAI-compatible APIs
        base_url = api_base or getattr(self, "_api_base", None) or getattr(self, "_endpoint", None)
        if not base_url:
            return []
        
        try:
            client = self._get_client()
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = client.get(
                f"{base_url.rstrip('/')}/v1/models" if not base_url.endswith("/v1") else f"{base_url}/models",
                headers=headers,
                timeout=5.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                return [m.get("id", m.get("name", "")) for m in models if m.get("id") or m.get("name")]
        except Exception:
            pass
        return []

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

        # Add function calling support
        if request.tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
        elif request.functions:
            # Legacy function calling format
            payload["functions"] = request.functions
            if request.tool_choice:
                payload["function_call"] = request.tool_choice

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
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            tokens = data.get("usage", {}).get("total_tokens", 0)
            finish_reason = choice.get("finish_reason")

            # Parse function call response (legacy format)
            function_call: FunctionCall | None = None
            if "function_call" in message:
                fc = message["function_call"]
                try:
                    args = json.loads(fc.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {"raw": fc.get("arguments", "")}
                function_call = FunctionCall(
                    name=fc.get("name", ""),
                    arguments=args,
                )

            # Parse tool calls (new format)
            tool_calls: list[FunctionCall] | None = None
            if "tool_calls" in message:
                tool_calls = []
                for tc in message["tool_calls"]:
                    if tc.get("type") == "function":
                        fn = tc["function"]
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {"raw": fn.get("arguments", "")}
                        tool_calls.append(FunctionCall(
                            name=fn.get("name", ""),
                            arguments=args,
                            id=tc.get("id"),
                        ))

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
                function_call=function_call,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
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

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from OpenAI API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                # Filter to chat models only
                chat_models = [
                    m.get("id") for m in models 
                    if m.get("id") and ("gpt" in m.get("id", "").lower() or "o1" in m.get("id", "").lower())
                ]
                # Sort with newest/best models first
                priority = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "o1"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model:
                            return (i, model)
                    return (100, model)
                return sorted(chat_models, key=sort_key)
        except Exception:
            pass
        return []

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
            tokens = data.get("usage", {}).get("input_tokens", 0) + data.get(
                "usage", {}
            ).get("output_tokens", 0)

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

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available Anthropic Claude models.
        
        Note: Anthropic doesn't have a public models endpoint, so we return
        the known available models based on API key validation.
        """
        if not api_key:
            return []
        
        # Validate the API key first
        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_base}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": self._api_version,
                    "Content-Type": "application/json",
                },
                json={"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
                timeout=10.0,
            )
            
            # If API key is valid (200) or model works, return all known models
            if response.status_code in (200, 400, 429):
                # Return known Claude models in order of capability
                return [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-opus-20240229",
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                    "claude-2.1",
                    "claude-2.0",
                    "claude-instant-1.2",
                ]
        except Exception:
            pass
        return []

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to Anthropic API."""
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
            "stream": True,
        }

        if request.system_prompt:
            payload["system"] = request.system_prompt

        if request.temperature > 0:
            payload["temperature"] = request.temperature

        # Add tool use support for Anthropic
        if request.tools:
            # Convert OpenAI tool format to Anthropic format
            anthropic_tools = []
            for tool in request.tools:
                if tool.get("type") == "function":
                    fn = tool["function"]
                    anthropic_tools.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    })
            payload["tools"] = anthropic_tools

        try:
            client = self._get_client()
            collected_text = ""
            chunks: list[str] = []
            tool_calls: list[FunctionCall] = []
            input_tokens = 0
            output_tokens = 0

            with client.stream(
                "POST",
                f"{self._api_base}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": self._api_version,
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                current_tool_name = ""
                current_tool_input = ""
                current_tool_id = ""

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type", "")

                    if event_type == "content_block_start":
                        block = data.get("content_block", {})
                        if block.get("type") == "tool_use":
                            current_tool_id = block.get("id", "")
                            current_tool_name = block.get("name", "")
                            current_tool_input = ""

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        delta_type = delta.get("type", "")

                        if delta_type == "text_delta":
                            text = delta.get("text", "")
                            collected_text += text
                            chunks.append(text)
                            if callback:
                                callback(text)
                        elif delta_type == "input_json_delta":
                            current_tool_input += delta.get("partial_json", "")

                    elif event_type == "content_block_stop":
                        if current_tool_name:
                            try:
                                args = json.loads(current_tool_input) if current_tool_input else {}
                            except json.JSONDecodeError:
                                args = {"raw": current_tool_input}
                            tool_calls.append(FunctionCall(
                                name=current_tool_name,
                                arguments=args,
                                id=current_tool_id,
                            ))
                            current_tool_name = ""
                            current_tool_input = ""
                            current_tool_id = ""

                    elif event_type == "message_delta":
                        usage = data.get("usage", {})
                        output_tokens = usage.get("output_tokens", output_tokens)

                    elif event_type == "message_start":
                        message = data.get("message", {})
                        usage = message.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

            elapsed = (time.perf_counter() - start) * 1000

            return LLMResponse(
                text=collected_text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=input_tokens + output_tokens,
                is_streaming=True,
                stream_chunks=chunks,
                tool_calls=tool_calls if tool_calls else None,
                finish_reason="tool_use" if tool_calls else "stop",
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


class OllamaProvider(_BaseProvider):
    """Ollama local LLM provider."""

    name: ProviderName = "ollama"
    is_local: bool = True
    requires_api_key: bool = False

    def __init__(self, timeout: float = 120.0):
        super().__init__(DEFAULT_MODELS["ollama"], timeout)
        self._endpoint: str | None = None
        self._available_models: list[str] | None = None
        self._model_checked: set[str] = set()

    def _get_available_models(self) -> list[str]:
        """Get list of available models from Ollama."""
        if self._available_models is not None:
            return self._available_models
        
        endpoint = self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        try:
            client = self._get_client()
            response = client.get(f"{endpoint}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                self._available_models = [m.get("name", "").split(":")[0] for m in models]
                return self._available_models
        except Exception:
            pass
        return []

    def _find_best_model(self, requested_model: str) -> str:
        """Find the best available model, falling back if requested isn't available."""
        available = self._get_available_models()
        if not available:
            return requested_model
        
        # Check if requested model (or a variant) is available
        for model in available:
            if requested_model in model or model in requested_model:
                return model
        
        # Return first available model as fallback
        return available[0] if available else requested_model

    def send(self, request: LLMRequest, api_key: str | None = None) -> LLMResponse:
        """Send a request to Ollama server using the chat API."""
        start = time.perf_counter()

        endpoint = self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        requested_model = request.model or self.default_model
        
        # Auto-detect best available model if not already checked
        if requested_model not in self._model_checked:
            model = self._find_best_model(requested_model)
            self._model_checked.add(requested_model)
        else:
            model = requested_model

        # Use the chat API format (more reliable)
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

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
                f"{endpoint}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            # Chat API returns message.content
            text = data.get("message", {}).get("content", "") or data.get("response", "")

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
                error=f"Cannot connect to Ollama at {endpoint}. Is it running? Try: ollama serve",
            )
        except httpx.HTTPStatusError as e:
            elapsed = (time.perf_counter() - start) * 1000
            error_msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 404:
                # Try to find an available model
                available = self._get_available_models()
                if available:
                    # Retry with first available model
                    fallback_model = available[0]
                    request_copy = LLMRequest(
                        prompt=request.prompt,
                        system_prompt=request.system_prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        provider=request.provider,
                        model=fallback_model,
                    )
                    self._model_checked.add(fallback_model)
                    return self.send(request_copy, api_key)
                error_msg = f"Model '{model}' not found. Available models: {', '.join(available) if available else 'none'}. Try: ollama pull {model}"
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=error_msg,
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
        check_endpoint = (
            endpoint or self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        )
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from Ollama."""
        endpoint = api_base or self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        try:
            client = self._get_client()
            response = client.get(f"{endpoint}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return [m.get("name", "") for m in models if m.get("name")]
        except Exception:
            pass
        return []

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to Ollama server using the chat API."""
        start = time.perf_counter()

        endpoint = self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        requested_model = request.model or self.default_model
        
        # Auto-detect best available model if not already checked
        if requested_model not in self._model_checked:
            model = self._find_best_model(requested_model)
            self._model_checked.add(requested_model)
        else:
            model = requested_model

        # Use the chat API format
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        options: dict[str, Any] = {}
        if request.temperature > 0:
            options["temperature"] = request.temperature
        if request.max_tokens:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options

        try:
            client = self._get_client()
            collected_text = ""
            chunks: list[str] = []
            total_tokens = 0

            with client.stream(
                "POST",
                f"{endpoint}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Chat API returns message.content
                    text = data.get("message", {}).get("content", "") or data.get("response", "")
                    if text:
                        collected_text += text
                        chunks.append(text)
                        if callback:
                            callback(text)

                    if data.get("done"):
                        total_tokens = data.get("eval_count", 0)

            elapsed = (time.perf_counter() - start) * 1000

            return LLMResponse(
                text=collected_text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=total_tokens,
                is_streaming=True,
                stream_chunks=chunks,
            )
        except httpx.ConnectError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"Cannot connect to Ollama at {endpoint}. Is it running? Try: ollama serve",
            )
        except httpx.HTTPStatusError as e:
            elapsed = (time.perf_counter() - start) * 1000
            error_msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 404:
                error_msg = f"Model '{model}' not found. Try: ollama pull {model}"
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=error_msg,
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

    def set_endpoint(self, endpoint: str) -> None:
        """Set the Ollama endpoint."""
        self._endpoint = endpoint

    def list_models(self, endpoint: str | None = None) -> list[str]:
        """List available models on the Ollama server."""
        check_endpoint = (
            endpoint or self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        )
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
            text = (
                data["choices"][0]["message"]["content"] if data.get("choices") else ""
            )
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
            endpoint
            or self._endpoint
            or f"http://localhost:{DEFAULT_PORTS['lmstudio']}"
        )
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/v1/models", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to LM Studio server."""
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
            "stream": True,
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            client = self._get_client()
            collected_text = ""
            chunks: list[str] = []

            with client.stream(
                "POST",
                f"{endpoint}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            collected_text += text
                            chunks.append(text)
                            if callback:
                                callback(text)

            elapsed = (time.perf_counter() - start) * 1000

            return LLMResponse(
                text=collected_text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                is_streaming=True,
                stream_chunks=chunks,
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
                tokens_used=data.get("tokens_evaluated", 0)
                + data.get("tokens_predicted", 0),
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
            endpoint
            or self._endpoint
            or f"http://localhost:{DEFAULT_PORTS['llama_cpp']}"
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


# =============================================================================
# EXTENDED PROVIDER INTEGRATIONS
# =============================================================================


class TogetherProvider(_BaseProvider):
    """Together AI provider for open-source models at scale.
    
    Supports models like:
    - Mixtral-8x7B-Instruct
    - Llama-2-70B-chat
    - CodeLlama-34B
    - Falcon-180B
    """

    name: ProviderName = "together"
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__(DEFAULT_MODELS["together"], timeout)
        self._api_base = API_BASES["together"]

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Together AI API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Together AI",
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
            choice = data["choices"][0]
            text = choice["message"].get("content", "")
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
        """Check if Together AI API is reachable."""
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={"Authorization": "Bearer test"},
                timeout=5.0,
            )
            return response.status_code in (200, 401)
        except Exception:
            return False

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from Together AI API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                # Together AI returns a list of models directly
                models = data if isinstance(data, list) else data.get("data", data.get("models", []))
                # Filter to chat/instruct models
                chat_models = [
                    m.get("id") or m.get("name") for m in models 
                    if (m.get("id") or m.get("name")) and 
                       ("chat" in str(m.get("id", "")).lower() or 
                        "instruct" in str(m.get("id", "")).lower() or
                        "llama" in str(m.get("id", "")).lower() or
                        "mixtral" in str(m.get("id", "")).lower())
                ]
                # Sort by preference
                priority = ["llama-3", "mixtral", "llama-2-70b", "codellama", "mistral"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(chat_models[:50], key=sort_key)  # Limit to 50 models
        except Exception:
            pass
        return []


class GroqProvider(_BaseProvider):
    """Groq provider for ultra-fast LLM inference.
    
    Known for extremely low latency inference using custom LPU hardware.
    Supports:
    - Mixtral-8x7B
    - Llama-2-70B
    - Gemma-7B
    """

    name: ProviderName = "groq"
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 30.0):  # Lower timeout for fast inference
        super().__init__(DEFAULT_MODELS["groq"], timeout)
        self._api_base = API_BASES["groq"]

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Groq API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Groq",
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

        # Add tool support (Groq uses OpenAI-compatible format)
        if request.tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice

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
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content", "")
            tokens = data.get("usage", {}).get("total_tokens", 0)
            finish_reason = choice.get("finish_reason")

            # Parse tool calls
            tool_calls: list[FunctionCall] = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    if tc.get("type") == "function":
                        fn = tc["function"]
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {"raw": fn.get("arguments", "")}
                        tool_calls.append(FunctionCall(
                            name=fn.get("name", ""),
                            arguments=args,
                            id=tc.get("id"),
                        ))

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
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
        """Check if Groq API is reachable."""
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={"Authorization": "Bearer test"},
                timeout=5.0,
            )
            return response.status_code in (200, 401)
        except Exception:
            return False

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from Groq API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models if m.get("id")]
                # Sort by preference (latest/largest first)
                priority = ["llama-3.3", "llama-3.1-70b", "llama-3.1-8b", "mixtral", "llama3-70b", "llama3-8b", "gemma"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(model_ids, key=sort_key)
        except Exception:
            pass
        return []

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to Groq API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Groq",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


class MistralProvider(_BaseProvider):
    """Mistral AI provider for state-of-the-art open-weight models.
    
    Supports:
    - mistral-large-latest
    - mistral-medium-latest
    - mistral-small-latest
    - codestral-latest
    """

    name: ProviderName = "mistral"
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__(DEFAULT_MODELS["mistral"], timeout)
        self._api_base = API_BASES["mistral"]

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Mistral AI API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Mistral AI",
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

        # Mistral supports tool calling
        if request.tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice

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
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content", "")
            tokens = data.get("usage", {}).get("total_tokens", 0)
            finish_reason = choice.get("finish_reason")

            # Parse tool calls
            tool_calls: list[FunctionCall] = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    if tc.get("type") == "function":
                        fn = tc["function"]
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {"raw": fn.get("arguments", "")}
                        tool_calls.append(FunctionCall(
                            name=fn.get("name", ""),
                            arguments=args,
                            id=tc.get("id"),
                        ))

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
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
        """Check if Mistral AI API is reachable."""
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={"Authorization": "Bearer test"},
                timeout=5.0,
            )
            return response.status_code in (200, 401)
        except Exception:
            return False

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from Mistral AI API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models if m.get("id")]
                # Sort by preference (large > medium > small)
                priority = ["large", "medium", "small", "codestral", "embed"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(model_ids, key=sort_key)
        except Exception:
            pass
        return []


class AzureOpenAIProvider(_BaseProvider):
    """Azure OpenAI provider for enterprise deployments.
    
    Requires:
    - Azure OpenAI endpoint
    - Deployment name
    - API key or Azure AD authentication
    """

    name: ProviderName = "azure_openai"
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(
        self,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str = "2024-02-15-preview",
        timeout: float = 60.0,
    ):
        super().__init__(DEFAULT_MODELS["azure_openai"], timeout)
        self._endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self._deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        self._api_version = api_version

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Azure OpenAI API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self._deployment,
                latency_ms=0,
                error="API key required for Azure OpenAI",
            )

        if not self._endpoint:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self._deployment,
                latency_ms=0,
                error="Azure OpenAI endpoint not configured",
            )

        deployment = request.model or self._deployment
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        # Azure OpenAI supports tool calling
        if request.tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice

        url = (
            f"{self._endpoint.rstrip('/')}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={self._api_version}"
        )

        try:
            client = self._get_client()
            response = client.post(
                url,
                headers={
                    "api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content", "")
            tokens = data.get("usage", {}).get("total_tokens", 0)
            finish_reason = choice.get("finish_reason")

            # Parse tool calls
            tool_calls: list[FunctionCall] = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    if tc.get("type") == "function":
                        fn = tc["function"]
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {"raw": fn.get("arguments", "")}
                        tool_calls.append(FunctionCall(
                            name=fn.get("name", ""),
                            arguments=args,
                            id=tc.get("id"),
                        ))

            return LLMResponse(
                text=text,
                provider=self.name,
                model=deployment,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except httpx.HTTPStatusError as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=deployment,
                latency_ms=elapsed,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=deployment,
                latency_ms=elapsed,
                error=str(e),
            )

    def health_check(self, endpoint: str | None = None) -> bool:
        """Check if Azure OpenAI is reachable."""
        check_endpoint = endpoint or self._endpoint
        if not check_endpoint:
            return False
        try:
            client = self._get_client()
            # Azure uses a different health check endpoint
            response = client.get(
                f"{check_endpoint.rstrip('/')}/openai/deployments?api-version={self._api_version}",
                headers={"api-key": "test"},
                timeout=5.0,
            )
            return response.status_code in (200, 401, 403)
        except Exception:
            return False

    def set_endpoint(self, endpoint: str) -> None:
        """Set the Azure OpenAI endpoint."""
        self._endpoint = endpoint

    def set_deployment(self, deployment: str) -> None:
        """Set the Azure OpenAI deployment name."""
        self._deployment = deployment


class CohereProvider(_BaseProvider):
    """Cohere provider for enterprise NLP and RAG applications.
    
    Supports:
    - command-r-plus (most capable)
    - command-r (balanced)
    - command-light (fast)
    
    Features:
    - Native RAG support
    - Multi-language support
    - Tool/Function calling
    """

    name: ProviderName = "cohere"
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__(DEFAULT_MODELS["cohere"], timeout)
        self._api_base = API_BASES["cohere"]

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Cohere API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Cohere",
            )

        model = request.model or self.default_model

        # Cohere uses a different message format
        chat_history = []
        if request.system_prompt:
            # Cohere uses preamble for system prompt
            preamble = request.system_prompt
        else:
            preamble = None

        payload: dict[str, Any] = {
            "model": model,
            "message": request.prompt,
            "temperature": request.temperature,
        }

        if preamble:
            payload["preamble"] = preamble

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        # Cohere tool support
        if request.tools:
            # Convert OpenAI tool format to Cohere format
            cohere_tools = []
            for tool in request.tools:
                if tool.get("type") == "function":
                    fn = tool["function"]
                    cohere_tools.append({
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameter_definitions": fn.get("parameters", {}).get("properties", {}),
                    })
            if cohere_tools:
                payload["tools"] = cohere_tools

        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_base}/chat",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            text = data.get("text", "")
            
            # Cohere uses different token counting
            tokens = (
                data.get("meta", {}).get("billed_units", {}).get("input_tokens", 0) +
                data.get("meta", {}).get("billed_units", {}).get("output_tokens", 0)
            )
            finish_reason = data.get("finish_reason", "COMPLETE")

            # Parse tool calls from Cohere format
            tool_calls: list[FunctionCall] = []
            if "tool_calls" in data:
                for tc in data["tool_calls"]:
                    tool_calls.append(FunctionCall(
                        name=tc.get("name", ""),
                        arguments=tc.get("parameters", {}),
                        id=tc.get("id"),
                    ))

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
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
        """Check if Cohere API is reachable."""
        try:
            client = self._get_client()
            # Cohere uses a different endpoint structure
            response = client.get(
                f"{self._api_base}/models",
                headers={"Authorization": "Bearer test"},
                timeout=5.0,
            )
            return response.status_code in (200, 401)
        except Exception:
            return False

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from Cohere API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                # Filter to chat-capable models
                chat_models = [
                    m.get("name") for m in models 
                    if m.get("name") and "command" in m.get("name", "").lower()
                ]
                # Sort by preference
                priority = ["command-r-plus", "command-r", "command-light", "command"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(chat_models, key=sort_key) if chat_models else ["command-r-plus", "command-r", "command-light"]
        except Exception:
            pass
        return ["command-r-plus", "command-r", "command-light"]

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to Cohere API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Cohere",
            )

        model = request.model or self.default_model

        payload: dict[str, Any] = {
            "model": model,
            "message": request.prompt,
            "temperature": request.temperature,
            "stream": True,
        }

        if request.system_prompt:
            payload["preamble"] = request.system_prompt

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            chunks: list[str] = []
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self._api_base}/chat",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("event_type") == "text-generation":
                                    content = data.get("text", "")
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


# ==============================================================================
# ADDITIONAL PROVIDERS (2% Gap Coverage)
# ==============================================================================


class PerplexityProvider(_BaseProvider):
    """Perplexity AI API provider with online search capabilities."""

    name: ProviderName = "perplexity"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("sonar-medium-online", timeout)
        self._api_base = "https://api.perplexity.ai"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Perplexity API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Perplexity",
            )

        model = request.model or self.default_model

        # Perplexity uses OpenAI-compatible format
        messages: list[dict[str, str]] = []
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
        """Check if Perplexity API is reachable."""
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={"Authorization": "Bearer test"},
                timeout=5.0,
            )
            return response.status_code in (200, 401)
        except Exception:
            return False

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from Perplexity API."""
        # Perplexity doesn't have a public models endpoint, return known models
        return [
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-sonar-small-128k-chat",
            "sonar-pro",
            "sonar",
        ]


class GoogleGeminiProvider(_BaseProvider):
    """Google Gemini API provider (Gemini Pro, Gemini Flash, etc.)."""

    name: ProviderName = "google"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        # Use stable model as default, user can override
        super().__init__("gemini-1.5-flash-latest", timeout)
        self._api_base = "https://generativelanguage.googleapis.com/v1beta"

    def _get_model_name(self, model: str) -> str:
        """Normalize model name for API compatibility."""
        # Handle various model name formats
        model = model.strip()
        
        # Map common aliases to correct model names
        # Google requires specific model names for the API
        model_aliases = {
            # Stable models - use -latest suffix for most recent
            'gemini-flash': 'gemini-1.5-flash-latest',
            'gemini-pro': 'gemini-1.5-pro-latest',
            'flash': 'gemini-1.5-flash-latest',
            'pro': 'gemini-1.5-pro-latest',
            'gemini-1.5-flash': 'gemini-1.5-flash-latest',
            'gemini-1.5-pro': 'gemini-1.5-pro-latest',
            # Keep -latest versions as-is
            'gemini-1.5-flash-latest': 'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest': 'gemini-1.5-pro-latest',
            # Older models
            'gemini-pro-vision': 'gemini-pro-vision',
            # Experimental models - use as-is
            'gemini-2.0-flash-exp': 'gemini-1.5-flash-latest',  # Fallback to stable if exp unavailable
            'gemini-exp-1206': 'gemini-exp-1206',
        }
        
        return model_aliases.get(model, model)

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Google Gemini API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Google Gemini",
            )

        model = self._get_model_name(request.model or self.default_model)
        
        # Build contents array for Gemini API
        contents = []
        
        # Add system instruction if provided (Gemini uses systemInstruction separately)
        system_instruction = None
        if request.system_prompt:
            system_instruction = {"parts": [{"text": request.system_prompt}]}
        
        # Add user message
        contents.append({
            "role": "user",
            "parts": [{"text": request.prompt}]
        })

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        if request.max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = request.max_tokens

        try:
            client = self._get_client()
            # Gemini API uses model name in URL path
            response = client.post(
                f"{self._api_base}/models/{model}:generateContent?key={api_key}",
                headers={
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            
            # Extract text from Gemini response format
            text = ""
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    text = "".join(part.get("text", "") for part in parts)
            
            # Get token count from usage metadata
            tokens = 0
            if "usageMetadata" in data:
                usage = data["usageMetadata"]
                tokens = usage.get("totalTokenCount", 0)

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
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_msg = f"{error_msg}: {error_data['error'].get('message', str(error_data['error']))}"
            except Exception:
                error_msg = f"{error_msg}: {e.response.text[:200]}"
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=error_msg,
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

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to Google Gemini API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Google Gemini",
            )

        model = self._get_model_name(request.model or self.default_model)

        contents = []
        system_instruction = None
        if request.system_prompt:
            system_instruction = {"parts": [{"text": request.system_prompt}]}
        
        contents.append({
            "role": "user",
            "parts": [{"text": request.prompt}]
        })

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        if request.max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = request.max_tokens

        try:
            chunks: list[str] = []
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self._api_base}/models/{model}:streamGenerateContent?key={api_key}&alt=sse",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                if "candidates" in data and data["candidates"]:
                                    candidate = data["candidates"][0]
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            content = part.get("text", "")
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def is_available(self) -> bool:
        """Check if Google Gemini API is available."""
        return True  # Always return True since we need API key to check

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from Google Gemini API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models?key={api_key}",
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                # Filter to generative models and extract clean names
                model_names = []
                for m in models:
                    name = m.get("name", "")
                    # Format: models/gemini-1.5-flash-latest -> gemini-1.5-flash-latest
                    if name.startswith("models/"):
                        name = name[7:]
                    # Only include text generation models (exclude vision-only, embedding)
                    if "generateContent" in m.get("supportedGenerationMethods", []):
                        model_names.append(name)
                
                # Sort by preference
                priority = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-1.0"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(model_names, key=sort_key)
        except Exception:
            pass
        return []


class XAIProvider(_BaseProvider):
    """xAI (Grok) API provider - uses OpenAI-compatible API."""

    name: ProviderName = "xai"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("grok-beta", timeout)
        self._api_base = "https://api.x.ai/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to xAI API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for xAI",
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

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to xAI API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for xAI",
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
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def is_available(self) -> bool:
        """Check if xAI API is available."""
        return True

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from xAI API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models if m.get("id")]
                # Sort with grok-2 first
                priority = ["grok-2", "grok-2-vision", "grok-2-mini", "grok-beta"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(model_ids, key=sort_key) if model_ids else ["grok-2", "grok-2-vision", "grok-2-mini", "grok-beta"]
        except Exception:
            pass
        return ["grok-2", "grok-2-vision", "grok-2-mini", "grok-beta"]


class DeepSeekProvider(_BaseProvider):
    """DeepSeek AI API provider for advanced reasoning."""

    name: ProviderName = "deepseek"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("deepseek-chat", timeout)
        self._api_base = "https://api.deepseek.com/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to DeepSeek API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for DeepSeek",
            )

        model = request.model or self.default_model

        messages: list[dict[str, str]] = []
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to DeepSeek API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for DeepSeek",
            )

        model = request.model or self.default_model

        messages: list[dict[str, str]] = []
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
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from DeepSeek API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models if m.get("id")]
                # Sort with chat/reasoner models first
                priority = ["deepseek-reasoner", "deepseek-chat", "deepseek-coder"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(model_ids, key=sort_key)
        except Exception:
            pass
        # Return known models if API call fails
        return ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]


class FireworksProvider(_BaseProvider):
    """Fireworks AI API provider for fast inference."""

    name: ProviderName = "fireworks"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("accounts/fireworks/models/llama-v3p1-70b-instruct", timeout)
        self._api_base = "https://api.fireworks.ai/inference/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Fireworks AI API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Fireworks AI",
            )

        model = request.model or self.default_model

        messages: list[dict[str, str]] = []
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from Fireworks AI API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(models, list):
                    model_ids = [m.get("id") or m.get("name") for m in models if m.get("id") or m.get("name")]
                    # Sort by preference
                    priority = ["llama-v3", "llama-3", "mixtral", "qwen", "deepseek"]
                    def sort_key(model):
                        for i, p in enumerate(priority):
                            if p in model.lower():
                                return (i, model)
                        return (100, model)
                    return sorted(model_ids[:30], key=sort_key)
        except Exception:
            pass
        # Return known models as fallback
        return [
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/qwen2-72b-instruct",
        ]


class HuggingFaceInferenceProvider(_BaseProvider):
    """HuggingFace Inference API provider."""

    name: ProviderName = "huggingface"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("meta-llama/Llama-3.1-70B-Instruct", timeout)
        self._api_base = "https://api-inference.huggingface.co/models"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to HuggingFace Inference API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for HuggingFace",
            )

        model = request.model or self.default_model

        # Build prompt with system message if provided
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant:"

        payload: dict[str, Any] = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": request.temperature,
                "return_full_text": False,
            },
        }

        if request.max_tokens:
            payload["parameters"]["max_new_tokens"] = request.max_tokens

        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_base}/{model}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000

            # HuggingFace returns list of generated texts
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "")
            else:
                text = data.get("generated_text", "")

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                raw=data,
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

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List popular models available on HuggingFace."""
        # HuggingFace has too many models to list, return popular LLM models
        return [
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "microsoft/Phi-3-mini-4k-instruct",
            "google/gemma-2-9b-it",
            "Qwen/Qwen2-72B-Instruct",
            "bigcode/starcoder2-15b",
        ]


class ReplicateProvider(_BaseProvider):
    """Replicate API provider for running models."""

    name: ProviderName = "replicate"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 120.0):  # Longer timeout for Replicate
        super().__init__("meta/llama-3.1-405b-instruct", timeout)
        self._api_base = "https://api.replicate.com/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Replicate API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Replicate",
            )

        model = request.model or self.default_model

        # Build prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"<|system|>{request.system_prompt}<|end|><|user|>{request.prompt}<|end|><|assistant|>"

        payload: dict[str, Any] = {
            "input": {
                "prompt": full_prompt,
                "temperature": request.temperature,
            },
        }

        if request.max_tokens:
            payload["input"]["max_tokens"] = request.max_tokens

        try:
            client = self._get_client()
            
            # Create prediction
            response = client.post(
                f"{self._api_base}/models/{model}/predictions",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            prediction = response.json()
            
            # Poll for completion
            prediction_id = prediction.get("id")
            max_polls = 60
            poll_count = 0
            
            while poll_count < max_polls:
                poll_response = client.get(
                    f"{self._api_base}/predictions/{prediction_id}",
                    headers={"Authorization": f"Token {api_key}"},
                )
                poll_data = poll_response.json()
                status = poll_data.get("status")
                
                if status == "succeeded":
                    output = poll_data.get("output", [])
                    text = "".join(output) if isinstance(output, list) else str(output)
                    elapsed = (time.perf_counter() - start) * 1000
                    return LLMResponse(
                        text=text,
                        provider=self.name,
                        model=model,
                        latency_ms=elapsed,
                        raw=poll_data,
                    )
                elif status in ("failed", "canceled"):
                    elapsed = (time.perf_counter() - start) * 1000
                    return LLMResponse(
                        text="",
                        provider=self.name,
                        model=model,
                        latency_ms=elapsed,
                        error=f"Prediction {status}: {poll_data.get('error', 'Unknown error')}",
                    )
                
                poll_count += 1
                time.sleep(1)
            
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error="Prediction timed out",
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

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List popular models available on Replicate."""
        # Return popular LLM models on Replicate
        return [
            "meta/llama-3.1-405b-instruct",
            "meta/llama-3-70b-instruct",
            "meta/llama-3-8b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "mistralai/mistral-7b-instruct-v0.2",
            "anthropic/claude-instant-1.2",
            "stability-ai/stablelm-tuned-alpha-7b",
        ]


class OpenRouterProvider(_BaseProvider):
    """OpenRouter API provider for unified access to multiple models."""

    name: ProviderName = "openrouter"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("anthropic/claude-3.5-sonnet", timeout)
        self._api_base = "https://openrouter.ai/api/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to OpenRouter API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for OpenRouter",
            )

        model = request.model or self.default_model

        messages: list[dict[str, str]] = []
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
                    "HTTP-Referer": "https://github.com/proxima",
                    "X-Title": "Proxima",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to OpenRouter API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for OpenRouter",
            )

        model = request.model or self.default_model

        messages: list[dict[str, str]] = []
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
                        "HTTP-Referer": "https://github.com/proxima",
                    },
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from OpenRouter API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models if m.get("id")]
                # Sort by popularity/preference
                priority = ["claude-3.5", "gpt-4o", "claude-3", "llama-3.3", "gpt-4", "gemini"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(model_ids[:100], key=sort_key)  # Limit to 100 models
        except Exception:
            pass
        return []


# =============================================================================
# Additional Provider Implementations (Full Settings Coverage)
# =============================================================================

class AI21Provider(_BaseProvider):
    """AI21 Labs (Jamba) provider - SSM-Transformer hybrid with 256K context."""

    name: ProviderName = "ai21"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("jamba-1.5-large", timeout)
        self._api_base = "https://api.ai21.com/studio/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to AI21 API."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for AI21",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from AI21 API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data if isinstance(data, list) else data.get("data", data.get("models", []))
                if isinstance(models, list):
                    return [m.get("id") or m.get("name") or str(m) for m in models if m][:20]
        except Exception:
            pass
        # Return known models as fallback
        return ["jamba-1.5-large", "jamba-1.5-mini", "j2-ultra", "j2-mid", "j2-light"]


class DeepInfraProvider(_BaseProvider):
    """DeepInfra provider - fast inference for open-source models."""

    name: ProviderName = "deepinfra"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("meta-llama/Meta-Llama-3.1-70B-Instruct", timeout)
        self._api_base = "https://api.deepinfra.com/v1/openai"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to DeepInfra API (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for DeepInfra",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def list_models(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> list[str]:
        """List available models from DeepInfra API."""
        if not api_key:
            return []
        
        try:
            client = self._get_client()
            response = client.get(
                f"{self._api_base}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id") for m in models if m.get("id")]
                # Sort by preference
                priority = ["llama-3", "mixtral", "qwen", "mistral", "gemma"]
                def sort_key(model):
                    for i, p in enumerate(priority):
                        if p in model.lower():
                            return (i, model)
                    return (100, model)
                return sorted(model_ids[:30], key=sort_key)
        except Exception:
            pass
        return [
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Qwen/Qwen2-72B-Instruct",
        ]


class AnyscaleProvider(_BaseProvider):
    """Anyscale Endpoints provider - scalable model serving."""

    name: ProviderName = "anyscale"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("meta-llama/Llama-3-70b-chat-hf", timeout)
        self._api_base = "https://api.endpoints.anyscale.com/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Anyscale API (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Anyscale",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Anyscale."""
        # Return popular models available on Anyscale
        known_models = [
            "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3-70b-chat-hf",
            "meta-llama/Llama-3-8b-chat-hf",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "codellama/CodeLlama-70b-Instruct-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
        ]
        return known_models


class VLLMProvider(_BaseProvider):
    """vLLM self-hosted provider - high-throughput inference server."""

    name: ProviderName = "vllm"  # type: ignore
    is_local: bool = True
    requires_api_key: bool = False

    def __init__(self, timeout: float = 120.0):
        super().__init__("local-model", timeout)
        self._endpoint = "http://localhost:8000"

    def set_endpoint(self, endpoint: str) -> None:
        """Set the vLLM server endpoint."""
        self._endpoint = endpoint.rstrip("/")

    def send(self, request: LLMRequest, api_key: str | None = None) -> LLMResponse:
        """Send a request to vLLM server (OpenAI-compatible)."""
        start = time.perf_counter()

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
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = client.post(
                f"{self._endpoint}/v1/chat/completions",
                headers=headers,
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
        except httpx.ConnectError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"Cannot connect to vLLM at {self._endpoint}. Is it running?",
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
        """Check if vLLM server is running."""
        check_endpoint = endpoint or self._endpoint
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/v1/models", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False


class LocalAIProvider(_BaseProvider):
    """LocalAI provider - self-hosted OpenAI-compatible server."""

    name: ProviderName = "localai"  # type: ignore
    is_local: bool = True
    requires_api_key: bool = False

    def __init__(self, timeout: float = 120.0):
        super().__init__("gpt-3.5-turbo", timeout)
        self._endpoint = "http://localhost:8080"

    def set_endpoint(self, endpoint: str) -> None:
        """Set the LocalAI server endpoint."""
        self._endpoint = endpoint.rstrip("/")

    def send(self, request: LLMRequest, api_key: str | None = None) -> LLMResponse:
        """Send a request to LocalAI server."""
        start = time.perf_counter()

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
                f"{self._endpoint}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
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
        except httpx.ConnectError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"Cannot connect to LocalAI at {self._endpoint}. Is it running?",
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
        """Check if LocalAI server is running."""
        check_endpoint = endpoint or self._endpoint
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/v1/models", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False


class TextGenWebUIProvider(_BaseProvider):
    """Text Generation WebUI (Oobabooga) provider."""

    name: ProviderName = "textgen_webui"  # type: ignore
    is_local: bool = True
    requires_api_key: bool = False

    def __init__(self, timeout: float = 120.0):
        super().__init__("local-model", timeout)
        self._endpoint = "http://localhost:5000"

    def set_endpoint(self, endpoint: str) -> None:
        """Set the server endpoint."""
        self._endpoint = endpoint.rstrip("/")

    def send(self, request: LLMRequest, api_key: str | None = None) -> LLMResponse:
        """Send a request to Text Generation WebUI (OpenAI-compatible API)."""
        start = time.perf_counter()

        model = request.model or self.default_model
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "mode": "instruct",
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        try:
            client = self._get_client()
            response = client.post(
                f"{self._endpoint}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
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
        except httpx.ConnectError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=f"Cannot connect to TextGen WebUI at {self._endpoint}. Is it running?",
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
        """Check if server is running."""
        check_endpoint = endpoint or self._endpoint
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/v1/models", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False


# Alias for Oobabooga (same as TextGenWebUI)
class OobaboogaProvider(TextGenWebUIProvider):
    """Oobabooga (Text Generation WebUI) - alias for TextGenWebUIProvider."""
    name: ProviderName = "oobabooga"  # type: ignore


class AWSBedrockProvider(_BaseProvider):
    """AWS Bedrock provider for Claude, Llama, and Titan models."""

    name: ProviderName = "aws_bedrock"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("anthropic.claude-3-5-sonnet-20241022-v2:0", timeout)
        self._region = "us-east-1"

    def set_region(self, region: str) -> None:
        """Set the AWS region."""
        self._region = region

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to AWS Bedrock (requires boto3 + AWS creds)."""
        start = time.perf_counter()
        model = request.model or self.default_model

        try:
            import boto3

            # Parse credentials if provided as "access_key:secret_key"
            if api_key and ":" in api_key:
                access_key, secret_key = api_key.split(":", 1)
                client = boto3.client(
                    "bedrock-runtime",
                    region_name=self._region,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
            else:
                client = boto3.client("bedrock-runtime", region_name=self._region)

            # Build request based on model type
            if "anthropic" in model.lower():
                messages = [{"role": "user", "content": request.prompt}]
                body: dict[str, Any] = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": request.max_tokens or 4096,
                    "messages": messages,
                }
                if request.system_prompt:
                    body["system"] = request.system_prompt
                if request.temperature > 0:
                    body["temperature"] = request.temperature
            else:
                prompt = request.prompt
                if request.system_prompt:
                    prompt = f"{request.system_prompt}\n\n{prompt}"
                body = {
                    "prompt": prompt,
                    "max_gen_len": request.max_tokens or 2048,
                    "temperature": request.temperature,
                }

            response = client.invoke_model(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            elapsed = (time.perf_counter() - start) * 1000

            if "anthropic" in model.lower():
                text = response_body.get("content", [{}])[0].get("text", "")
                tokens = response_body.get("usage", {}).get("input_tokens", 0) + \
                         response_body.get("usage", {}).get("output_tokens", 0)
            else:
                text = response_body.get("generation", response_body.get("output", ""))
                tokens = 0

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=response_body,
            )
        except ImportError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error="AWS Bedrock requires boto3. Install with: pip install boto3",
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


class VertexAIProvider(_BaseProvider):
    """Google Vertex AI provider for Gemini models."""

    name: ProviderName = "vertex_ai"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("gemini-1.5-pro", timeout)
        self._project_id: str | None = None
        self._location = "us-central1"

    def set_project(self, project_id: str, location: str = "us-central1") -> None:
        """Set the GCP project and location."""
        self._project_id = project_id
        self._location = location

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Vertex AI (requires google-cloud-aiplatform)."""
        start = time.perf_counter()
        model = request.model or self.default_model

        try:
            from google.cloud import aiplatform
            from vertexai.generative_models import GenerativeModel

            if self._project_id:
                aiplatform.init(project=self._project_id, location=self._location)

            gen_model = GenerativeModel(model)
            prompt = request.prompt
            if request.system_prompt:
                prompt = f"{request.system_prompt}\n\n{prompt}"

            generation_config: dict[str, Any] = {"temperature": request.temperature}
            if request.max_tokens:
                generation_config["max_output_tokens"] = request.max_tokens

            response = gen_model.generate_content(
                prompt, generation_config=generation_config
            )
            elapsed = (time.perf_counter() - start) * 1000
            text = response.text if hasattr(response, "text") else str(response)

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
            )
        except ImportError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error="Vertex AI requires google-cloud-aiplatform. Install with: pip install google-cloud-aiplatform",
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


class WatsonxProvider(_BaseProvider):
    """IBM watsonx.ai provider."""

    name: ProviderName = "watsonx"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("ibm/granite-13b-chat-v2", timeout)
        self._api_base = "https://us-south.ml.cloud.ibm.com"
        self._project_id: str | None = None

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to watsonx.ai."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for watsonx.ai",
            )

        model = request.model or self.default_model
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{prompt}"

        payload: dict[str, Any] = {
            "model_id": model,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": request.max_tokens or 1024,
                "temperature": request.temperature,
            },
        }
        if self._project_id:
            payload["project_id"] = self._project_id

        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_base}/ml/v1/text/generation?version=2024-03-14",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            text = data.get("results", [{}])[0].get("generated_text", "")
            tokens = data.get("results", [{}])[0].get("generated_token_count", 0)

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                tokens_used=tokens,
                raw=data,
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


class SambaNovaProvider(_BaseProvider):
    """SambaNova provider - ultra-fast RDU-based inference."""

    name: ProviderName = "sambanova"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("Meta-Llama-3.1-70B-Instruct", timeout)
        self._api_base = "https://api.sambanova.ai/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to SambaNova API (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for SambaNova",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from SambaNova."""
        base = api_base or self._api_base
        key = api_key
        
        if not key:
            return [
                "Meta-Llama-3.1-405B-Instruct",
                "Meta-Llama-3.1-70B-Instruct",
                "Meta-Llama-3.1-8B-Instruct",
                "Meta-Llama-3.2-3B-Instruct",
                "Meta-Llama-3.2-1B-Instruct",
                "Qwen2.5-72B-Instruct",
                "Qwen2.5-Coder-32B-Instruct",
            ]
        
        try:
            client = self._get_client()
            response = client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            data = response.json()
            
            models = [m["id"] for m in data.get("data", [])]
            # Prioritize larger models
            priority = ["405B", "70B", "72B", "32B", "8B", "3B", "1B"]
            def sort_key(m: str) -> tuple[int, str]:
                for i, p in enumerate(priority):
                    if p in m:
                        return (i, m)
                return (100, m)
            
            return sorted(models, key=sort_key) if models else self.list_models(api_base, None)
        except Exception:
            return self.list_models(api_base, None)


class CerebrasProvider(_BaseProvider):
    """Cerebras provider - fastest inference with custom AI accelerators."""

    name: ProviderName = "cerebras"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 30.0):
        super().__init__("llama3.1-70b", timeout)
        self._api_base = "https://api.cerebras.ai/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Cerebras API (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Cerebras",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Cerebras (ultra-fast inference)."""
        base = api_base or self._api_base
        key = api_key
        
        if not key:
            return [
                "llama3.1-70b",
                "llama3.1-8b",
                "llama-3.3-70b",
            ]
        
        try:
            client = self._get_client()
            response = client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            data = response.json()
            
            models = [m["id"] for m in data.get("data", [])]
            # Prioritize larger models
            priority = ["70b", "8b", "3b"]
            def sort_key(m: str) -> tuple[int, str]:
                m_lower = m.lower()
                for i, p in enumerate(priority):
                    if p in m_lower:
                        return (i, m)
                return (100, m)
            
            return sorted(models, key=sort_key) if models else self.list_models(api_base, None)
        except Exception:
            return self.list_models(api_base, None)


class LeptonProvider(_BaseProvider):
    """Lepton AI provider - serverless GPU inference."""

    name: ProviderName = "lepton"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("llama3-70b", timeout)
        self._api_base = "https://llama3-70b.lepton.run/api/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Lepton AI (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Lepton AI",
            )

        model = request.model or self.default_model
        api_base = f"https://{model}.lepton.run/api/v1"

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
                f"{api_base}/chat/completions",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Lepton AI."""
        # Lepton has dynamic model endpoints; return popular ones
        return [
            "llama3.1-405b",
            "llama3.1-70b",
            "llama3.1-8b",
            "llama3-70b",
            "llama3-8b",
            "mixtral-8x22b",
            "mixtral-8x7b",
            "qwen2-72b",
        ]


class NovitaProvider(_BaseProvider):
    """Novita AI provider - GPU cloud for AI models."""

    name: ProviderName = "novita"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("meta-llama/llama-3.1-70b-instruct", timeout)
        self._api_base = "https://api.novita.ai/v3/openai"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Novita AI (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Novita AI",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Novita AI."""
        base = api_base or self._api_base
        key = api_key
        
        if not key:
            return [
                "meta-llama/llama-3.1-405b-instruct",
                "meta-llama/llama-3.1-70b-instruct",
                "meta-llama/llama-3.1-8b-instruct",
                "qwen/qwen-2.5-72b-instruct",
                "mistralai/mixtral-8x22b-instruct",
                "deepseek/deepseek-v2.5",
            ]
        
        try:
            client = self._get_client()
            response = client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            data = response.json()
            
            models = [m["id"] for m in data.get("data", [])]
            return models if models else self.list_models(api_base, None)
        except Exception:
            return self.list_models(api_base, None)


class FriendliProvider(_BaseProvider):
    """Friendli AI provider - optimized inference serving."""

    name: ProviderName = "friendli"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("meta-llama-3.1-70b-instruct", timeout)
        self._api_base = "https://inference.friendli.ai/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Friendli AI (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Friendli AI",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Friendli AI."""
        base = api_base or self._api_base
        key = api_key
        
        if not key:
            return [
                "meta-llama-3.1-70b-instruct",
                "meta-llama-3.1-8b-instruct",
                "mistral-7b-instruct-v0.3",
                "mixtral-8x7b-instruct-v0.1",
            ]
        
        try:
            client = self._get_client()
            response = client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            data = response.json()
            
            models = [m["id"] for m in data.get("data", [])]
            return models if models else self.list_models(api_base, None)
        except Exception:
            return self.list_models(api_base, None)


class RekaProvider(_BaseProvider):
    """Reka AI provider - multimodal foundation models."""

    name: ProviderName = "reka"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("reka-flash", timeout)
        self._api_base = "https://api.reka.ai/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Reka AI."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Reka AI",
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
                    "X-Api-Key": api_key,
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Reka AI."""
        # Reka has limited models
        return [
            "reka-core",
            "reka-flash",
            "reka-edge",
        ]


class WriterProvider(_BaseProvider):
    """Writer AI provider - enterprise AI platform."""

    name: ProviderName = "writer"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("palmyra-x-004", timeout)
        self._api_base = "https://api.writer.com/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Writer AI (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Writer AI",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Writer AI."""
        # Writer's Palmyra models
        return [
            "palmyra-x-004",
            "palmyra-x-003-instruct",
            "palmyra-x-002-32k",
            "palmyra-med",
            "palmyra-fin",
        ]


class BasetenProvider(_BaseProvider):
    """Baseten provider - model deployment platform."""

    name: ProviderName = "baseten"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("llama-3.1-70b", timeout)
        self._api_base = "https://bridge.baseten.co/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Baseten (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Baseten",
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
                    "Authorization": f"Api-Key {api_key}",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


class ModalProvider(_BaseProvider):
    """Modal provider - serverless cloud for ML."""

    name: ProviderName = "modal"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("local-model", timeout)
        self._endpoint = "https://your-app.modal.run/v1"

    def set_endpoint(self, endpoint: str) -> None:
        """Set the Modal endpoint (user must deploy their own)."""
        self._endpoint = endpoint.rstrip("/")

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Modal (OpenAI-compatible)."""
        start = time.perf_counter()

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
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = client.post(
                f"{self._endpoint}/chat/completions",
                headers=headers,
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


class RunPodProvider(_BaseProvider):
    """RunPod provider - GPU cloud for AI inference."""

    name: ProviderName = "runpod"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("llama-3.1-70b", timeout)
        self._endpoint = "https://api.runpod.ai/v2"

    def set_endpoint(self, endpoint: str) -> None:
        """Set the RunPod serverless endpoint."""
        self._endpoint = endpoint.rstrip("/")

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to RunPod (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for RunPod",
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
                f"{self._endpoint}/openai/v1/chat/completions",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


class LambdaProvider(_BaseProvider):
    """Lambda Labs provider - GPU cloud inference."""

    name: ProviderName = "lambda"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("llama3.1-70b-instruct-fp8", timeout)
        self._api_base = "https://api.lambdalabs.com/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Lambda Labs (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Lambda Labs",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Lambda Labs."""
        base = api_base or self._api_base
        key = api_key
        
        if not key:
            return [
                "llama3.1-70b-instruct-fp8",
                "llama3.1-8b-instruct-fp8",
                "llama3.1-405b-instruct-fp8",
                "hermes-3-llama-3.1-405b-fp8",
            ]
        
        try:
            client = self._get_client()
            response = client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            data = response.json()
            
            models = [m["id"] for m in data.get("data", [])]
            return models if models else self.list_models(api_base, None)
        except Exception:
            return self.list_models(api_base, None)


class MonsterProvider(_BaseProvider):
    """Monster API provider - GPU cloud for AI."""

    name: ProviderName = "monster"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("meta-llama/Meta-Llama-3.1-70B-Instruct", timeout)
        self._api_base = "https://llm.monsterapi.ai/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Monster API (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Monster API",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Monster API."""
        base = api_base or self._api_base
        key = api_key
        
        if not key:
            return [
                "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "microsoft/Phi-3.5-mini-instruct",
                "google/gemma-2-27b-it",
            ]
        
        try:
            client = self._get_client()
            response = client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            data = response.json()
            
            models = [m["id"] for m in data.get("data", [])]
            return models if models else self.list_models(api_base, None)
        except Exception:
            return self.list_models(api_base, None)


class HyperbolicProvider(_BaseProvider):
    """Hyperbolic provider - decentralized AI compute."""

    name: ProviderName = "hyperbolic"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("meta-llama/Meta-Llama-3.1-70B-Instruct", timeout)
        self._api_base = "https://api.hyperbolic.xyz/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Hyperbolic (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Hyperbolic",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Hyperbolic."""
        base = api_base or self._api_base
        key = api_key
        
        if not key:
            return [
                "meta-llama/Meta-Llama-3.1-405B-Instruct",
                "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                "deepseek-ai/DeepSeek-V2.5",
                "mistralai/Mixtral-8x22B-Instruct-v0.1",
            ]
        
        try:
            client = self._get_client()
            response = client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            data = response.json()
            
            models = [m["id"] for m in data.get("data", [])]
            return models if models else self.list_models(api_base, None)
        except Exception:
            return self.list_models(api_base, None)


class KlusterProvider(_BaseProvider):
    """Kluster AI provider - distributed GPU cloud."""

    name: ProviderName = "kluster"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("llama-3.1-70b", timeout)
        self._api_base = "https://api.kluster.ai/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Kluster AI (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Kluster AI",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )

    def list_models(self, api_base: str | None = None, api_key: str | None = None) -> list[str]:
        """List available models from Kluster AI."""
        base = api_base or self._api_base
        key = api_key
        
        if not key:
            return [
                "llama-3.1-405b-instruct",
                "llama-3.1-70b-instruct",
                "llama-3.1-8b-instruct",
                "mixtral-8x22b-instruct",
                "qwen-2.5-72b-instruct",
            ]
        
        try:
            client = self._get_client()
            response = client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            data = response.json()
            
            models = [m["id"] for m in data.get("data", [])]
            return models if models else self.list_models(api_base, None)
        except Exception:
            return self.list_models(api_base, None)


class OracleAIProvider(_BaseProvider):
    """Oracle OCI Generative AI provider."""

    name: ProviderName = "oracle_ai"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("cohere.command-r-plus", timeout)
        self._api_base = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130"
        self._compartment_id: str | None = None

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Oracle OCI Generative AI."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Oracle AI",
            )

        model = request.model or self.default_model
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{prompt}"

        payload: dict[str, Any] = {
            "servingMode": {"servingType": "ON_DEMAND", "modelId": model},
            "inferenceRequest": {
                "runtimeType": "COHERE",
                "prompt": prompt,
                "maxTokens": request.max_tokens or 1024,
                "temperature": request.temperature,
            },
        }
        if self._compartment_id:
            payload["compartmentId"] = self._compartment_id

        try:
            client = self._get_client()
            response = client.post(
                f"{self._api_base}/actions/generateText",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            elapsed = (time.perf_counter() - start) * 1000
            text = data.get("inferenceResponse", {}).get("generatedTexts", [{}])[0].get("text", "")

            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                raw=data,
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


class AlibabaQwenProvider(_BaseProvider):
    """Alibaba Cloud DashScope provider for Qwen models."""

    name: ProviderName = "alibaba_qwen"  # type: ignore
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, timeout: float = 60.0):
        super().__init__("qwen-max", timeout)
        self._api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
        """Send a request to Alibaba DashScope (OpenAI-compatible)."""
        start = time.perf_counter()

        if not api_key:
            return LLMResponse(
                text="",
                provider=self.name,
                model=request.model or self.default_model,
                latency_ms=0,
                error="API key required for Alibaba Qwen",
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
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResponse(
                text="",
                provider=self.name,
                model=model,
                latency_ms=elapsed,
                error=str(e),
            )


class ProviderRegistry:
    """Registry of available LLM providers.

    Includes both core providers and extended integrations:
    - Core: OpenAI, Anthropic, Ollama, LM Studio, llama.cpp
    - Extended: Together AI, Groq, Mistral, Azure OpenAI, Cohere
    - Additional: Perplexity, DeepSeek, Fireworks, HuggingFace, Replicate, OpenRouter
    - Custom: Google Gemini, xAI (Grok)
    - New: AI21, DeepInfra, Anyscale, vLLM, LocalAI, TextGen WebUI, Oobabooga
    - Enterprise: AWS Bedrock, Vertex AI, watsonx, Oracle AI, Alibaba Qwen
    - Fast Inference: SambaNova, Cerebras, Lepton, Novita, Friendli
    - Specialized: Reka, Writer, Baseten, Modal, RunPod, Lambda, Monster, Hyperbolic, Kluster
    """

    def __init__(self) -> None:
        self._providers: dict[ProviderName, LLMProvider] = {
            # Core providers
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "ollama": OllamaProvider(),
            "lmstudio": LMStudioProvider(),
            "llama_cpp": LlamaCppProvider(),
            # Extended providers
            "together": TogetherProvider(),
            "groq": GroqProvider(),
            "mistral": MistralProvider(),
            "azure_openai": AzureOpenAIProvider(),
            "cohere": CohereProvider(),
            # Additional providers (2% gap coverage)
            "perplexity": PerplexityProvider(),  # type: ignore
            "deepseek": DeepSeekProvider(),  # type: ignore
            "fireworks": FireworksProvider(),  # type: ignore
            "huggingface": HuggingFaceInferenceProvider(),  # type: ignore
            "replicate": ReplicateProvider(),  # type: ignore
            "openrouter": OpenRouterProvider(),  # type: ignore
            # Google Gemini and xAI
            "google": GoogleGeminiProvider(),  # type: ignore
            "xai": XAIProvider(),  # type: ignore
            # NEW: Additional cloud providers
            "ai21": AI21Provider(),  # type: ignore
            "deepinfra": DeepInfraProvider(),  # type: ignore
            "anyscale": AnyscaleProvider(),  # type: ignore
            # NEW: Local inference servers
            "vllm": VLLMProvider(),  # type: ignore
            "localai": LocalAIProvider(),  # type: ignore
            "textgen_webui": TextGenWebUIProvider(),  # type: ignore
            "oobabooga": OobaboogaProvider(),  # type: ignore
            # NEW: Enterprise cloud providers
            "aws_bedrock": AWSBedrockProvider(),  # type: ignore
            "vertex_ai": VertexAIProvider(),  # type: ignore
            "watsonx": WatsonxProvider(),  # type: ignore
            "oracle_ai": OracleAIProvider(),  # type: ignore
            "alibaba_qwen": AlibabaQwenProvider(),  # type: ignore
            # NEW: Fast inference providers
            "sambanova": SambaNovaProvider(),  # type: ignore
            "cerebras": CerebrasProvider(),  # type: ignore
            "lepton": LeptonProvider(),  # type: ignore
            "novita": NovitaProvider(),  # type: ignore
            "friendli": FriendliProvider(),  # type: ignore
            # NEW: Specialized providers
            "reka": RekaProvider(),  # type: ignore
            "writer": WriterProvider(),  # type: ignore
            "baseten": BasetenProvider(),  # type: ignore
            "modal": ModalProvider(),  # type: ignore
            "runpod": RunPodProvider(),  # type: ignore
            "lambda": LambdaProvider(),  # type: ignore
            "monster": MonsterProvider(),  # type: ignore
            "hyperbolic": HyperbolicProvider(),  # type: ignore
            "kluster": KlusterProvider(),  # type: ignore
        }

    def get(self, name: ProviderName) -> LLMProvider:
        """Get a provider by name."""
        if name == "none":
            raise ValueError("No LLM provider configured")
        provider = self._providers.get(name)  # type: ignore
        if not provider:
            raise ValueError(f"Unknown provider: {name}")
        return provider

    def list_providers(self) -> list[ProviderName]:
        """List all registered provider names."""
        return list(self._providers.keys())  # type: ignore

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
        # Core providers
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "replicate": "REPLICATE_API_TOKEN",
        "openrouter": "OPENROUTER_API_KEY",
        # Enterprise providers
        "azure_openai": "AZURE_OPENAI_API_KEY",
        "aws_bedrock": "AWS_ACCESS_KEY_ID",  # Uses AWS creds format
        "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",  # Uses GCP creds
        "watsonx": "WATSONX_API_KEY",
        "oracle_ai": "OCI_API_KEY",
        "alibaba_qwen": "DASHSCOPE_API_KEY",
        "huggingface": "HF_TOKEN",
        # New cloud providers
        "ai21": "AI21_API_KEY",
        "deepinfra": "DEEPINFRA_API_KEY",
        "anyscale": "ANYSCALE_API_KEY",
        # Fast inference providers
        "sambanova": "SAMBANOVA_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "lepton": "LEPTON_API_KEY",
        "novita": "NOVITA_API_KEY",
        "friendli": "FRIENDLI_API_KEY",
        # Specialized providers
        "reka": "REKA_API_KEY",
        "writer": "WRITER_API_KEY",
        "baseten": "BASETEN_API_KEY",
        "modal": "MODAL_API_KEY",
        "runpod": "RUNPOD_API_KEY",
        "lambda": "LAMBDA_API_KEY",
        "monster": "MONSTER_API_KEY",
        "hyperbolic": "HYPERBOLIC_API_KEY",
        "kluster": "KLUSTER_API_KEY",
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
        if (
            hasattr(self.settings.llm, "api_key_env_var")
            and self.settings.llm.api_key_env_var
        ):
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
            raise PermissionError(
                f"Previously denied consent for provider {provider.name}"
            )

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
            raise ValueError(
                "No LLM provider configured. Set llm.provider in config or request."
            )
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


# =============================================================================
# Token Usage Tracking & Cost Estimation (Feature - LLM Router)
# =============================================================================


@dataclass
class TokenPricing:
    """Pricing information for a model."""
    
    provider: ProviderName
    model: str
    input_cost_per_1k: float  # USD per 1000 input tokens
    output_cost_per_1k: float  # USD per 1000 output tokens
    context_window: int = 128000  # Max context length
    last_updated: str = "2025-01-15"


@dataclass
class UsageRecord:
    """Record of a single LLM usage."""
    
    timestamp: float
    provider: ProviderName
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageSummary:
    """Summary of token usage over a period."""
    
    period_start: float
    period_end: float
    total_requests: int
    successful_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    average_latency_ms: float
    by_provider: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)


class TokenUsageTracker:
    """Tracks token usage and estimates costs across LLM providers.
    
    Features:
    - Real-time token counting
    - Cost estimation per request
    - Usage history and analytics
    - Budget alerts and limits
    - Export usage reports
    """
    
    # Default pricing (as of January 2025)
    DEFAULT_PRICING: dict[str, TokenPricing] = {
        # OpenAI models
        "gpt-4": TokenPricing("openai", "gpt-4", 0.03, 0.06, 8192),
        "gpt-4-turbo": TokenPricing("openai", "gpt-4-turbo", 0.01, 0.03, 128000),
        "gpt-4o": TokenPricing("openai", "gpt-4o", 0.005, 0.015, 128000),
        "gpt-4o-mini": TokenPricing("openai", "gpt-4o-mini", 0.00015, 0.0006, 128000),
        "gpt-3.5-turbo": TokenPricing("openai", "gpt-3.5-turbo", 0.0005, 0.0015, 16385),
        "o1": TokenPricing("openai", "o1", 0.015, 0.06, 200000),
        "o1-mini": TokenPricing("openai", "o1-mini", 0.003, 0.012, 128000),
        # Anthropic models
        "claude-3-opus-20240229": TokenPricing("anthropic", "claude-3-opus-20240229", 0.015, 0.075, 200000),
        "claude-3-sonnet-20240229": TokenPricing("anthropic", "claude-3-sonnet-20240229", 0.003, 0.015, 200000),
        "claude-3-5-sonnet-20241022": TokenPricing("anthropic", "claude-3-5-sonnet-20241022", 0.003, 0.015, 200000),
        "claude-3-haiku-20240307": TokenPricing("anthropic", "claude-3-haiku-20240307", 0.00025, 0.00125, 200000),
        # Local models (free)
        "ollama": TokenPricing("ollama", "local", 0.0, 0.0, 32768),
        "lmstudio": TokenPricing("lmstudio", "local", 0.0, 0.0, 32768),
        "llama_cpp": TokenPricing("llama_cpp", "local", 0.0, 0.0, 32768),
    }
    
    def __init__(
        self,
        storage_path: str | None = None,
        budget_limit_usd: float | None = None,
        alert_threshold_pct: float = 80.0,
    ) -> None:
        """Initialize tracker.
        
        Args:
            storage_path: Path to store usage history
            budget_limit_usd: Optional monthly budget limit
            alert_threshold_pct: Percentage of budget to trigger alert
        """
        self._storage_path = storage_path
        self._budget_limit = budget_limit_usd
        self._alert_threshold = alert_threshold_pct
        self._usage_history: list[UsageRecord] = []
        self._custom_pricing: dict[str, TokenPricing] = {}
        self._session_total_cost = 0.0
        self._session_total_tokens = 0
        self._load_history()
    
    def _load_history(self) -> None:
        """Load usage history from storage."""
        if not self._storage_path:
            return
        
        storage_path = Path(self._storage_path)
        if not storage_path.exists():
            return
        
        try:
            data = json.loads(storage_path.read_text())
            for record in data.get("history", []):
                self._usage_history.append(UsageRecord(
                    timestamp=record["timestamp"],
                    provider=record["provider"],
                    model=record["model"],
                    input_tokens=record["input_tokens"],
                    output_tokens=record["output_tokens"],
                    total_tokens=record["total_tokens"],
                    cost_usd=record["cost_usd"],
                    latency_ms=record["latency_ms"],
                    success=record["success"],
                    request_id=record.get("request_id", ""),
                    metadata=record.get("metadata", {}),
                ))
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    
    def _save_history(self) -> None:
        """Save usage history to storage."""
        if not self._storage_path:
            return
        
        storage_path = Path(self._storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Keep last 10000 records
        recent_history = self._usage_history[-10000:]
        
        data = {
            "history": [
                {
                    "timestamp": r.timestamp,
                    "provider": r.provider,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_tokens": r.total_tokens,
                    "cost_usd": r.cost_usd,
                    "latency_ms": r.latency_ms,
                    "success": r.success,
                    "request_id": r.request_id,
                    "metadata": r.metadata,
                }
                for r in recent_history
            ],
            "last_updated": time.time(),
        }
        
        try:
            storage_path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass
    
    def get_pricing(self, provider: ProviderName, model: str) -> TokenPricing:
        """Get pricing for a model.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            TokenPricing for the model
        """
        # Check custom pricing first
        key = f"{provider}:{model}"
        if key in self._custom_pricing:
            return self._custom_pricing[key]
        
        # Check default pricing
        if model in self.DEFAULT_PRICING:
            return self.DEFAULT_PRICING[model]
        
        # Check provider default
        if provider in self.DEFAULT_PRICING:
            return self.DEFAULT_PRICING[provider]
        
        # Local providers are free
        if provider in ("ollama", "lmstudio", "llama_cpp"):
            return TokenPricing(provider, model, 0.0, 0.0, 32768)
        
        # Unknown model - use conservative estimate
        return TokenPricing(provider, model, 0.01, 0.03, 8192)
    
    def set_custom_pricing(self, pricing: TokenPricing) -> None:
        """Set custom pricing for a model.
        
        Args:
            pricing: Custom pricing to set
        """
        key = f"{pricing.provider}:{pricing.model}"
        self._custom_pricing[key] = pricing
    
    def estimate_cost(
        self,
        provider: ProviderName,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a request.
        
        Args:
            provider: Provider name
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        pricing = self.get_pricing(provider, model)
        
        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
        
        return input_cost + output_cost
    
    def estimate_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Estimate token count for text.
        
        Uses a simple approximation: ~4 characters per token for English.
        For accurate counts, use tiktoken library.
        
        Args:
            text: Text to estimate tokens for
            model: Model name (affects tokenization)
            
        Returns:
            Estimated token count
        """
        # Simple approximation
        # Average: ~4 chars per token for GPT models, ~3.5 for Claude
        if "claude" in model.lower():
            chars_per_token = 3.5
        else:
            chars_per_token = 4.0
        
        return max(1, int(len(text) / chars_per_token))
    
    def record_usage(
        self,
        response: LLMResponse,
        input_text: str | None = None,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Record usage from an LLM response.
        
        Args:
            response: LLM response with token counts
            input_text: Optional input text for estimation
            request_id: Optional unique request ID
            metadata: Optional metadata to store
            
        Returns:
            UsageRecord for the request
        """
        # Extract or estimate token counts
        if response.raw and "usage" in response.raw:
            usage = response.raw["usage"]
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        else:
            input_tokens = self.estimate_tokens(input_text or "", response.model) if input_text else 0
            output_tokens = self.estimate_tokens(response.text, response.model)
        
        total_tokens = response.tokens_used or (input_tokens + output_tokens)
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        cost = self.estimate_cost(response.provider, response.model, input_tokens, output_tokens)
        
        # Create record
        record = UsageRecord(
            timestamp=time.time(),
            provider=response.provider,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=response.latency_ms,
            success=not bool(response.error),
            request_id=request_id or f"req_{int(time.time() * 1000)}",
            metadata=metadata or {},
        )
        
        # Update tracking
        self._usage_history.append(record)
        self._session_total_cost += cost
        self._session_total_tokens += total_tokens
        
        # Save periodically (every 10 requests)
        if len(self._usage_history) % 10 == 0:
            self._save_history()
        
        # Check budget
        self._check_budget_alert()
        
        return record
    
    def _check_budget_alert(self) -> None:
        """Check if budget alert should be triggered."""
        if not self._budget_limit:
            return
        
        monthly_cost = self.get_monthly_cost()
        threshold = self._budget_limit * (self._alert_threshold / 100)
        
        if monthly_cost >= threshold:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"LLM budget alert: ${monthly_cost:.2f} of ${self._budget_limit:.2f} used "
                f"({monthly_cost / self._budget_limit * 100:.1f}%)"
            )
    
    def get_monthly_cost(self) -> float:
        """Get total cost for current month."""
        now = time.time()
        month_start = time.mktime(time.strptime(
            time.strftime("%Y-%m-01", time.localtime(now)),
            "%Y-%m-%d"
        ))
        
        return sum(
            r.cost_usd for r in self._usage_history
            if r.timestamp >= month_start
        )
    
    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics for current session."""
        return {
            "total_requests": len([r for r in self._usage_history if r.timestamp >= self._session_start()]),
            "total_tokens": self._session_total_tokens,
            "total_cost_usd": self._session_total_cost,
            "providers_used": list(set(r.provider for r in self._usage_history)),
        }
    
    def _session_start(self) -> float:
        """Get session start timestamp."""
        # Approximate: first request in history or now - 1 hour
        if self._usage_history:
            recent = [r for r in self._usage_history if r.timestamp > time.time() - 3600]
            return min(r.timestamp for r in recent) if recent else time.time()
        return time.time()
    
    def get_usage_summary(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> UsageSummary:
        """Get usage summary for a time period.
        
        Args:
            start_time: Start of period (default: 30 days ago)
            end_time: End of period (default: now)
            
        Returns:
            UsageSummary with aggregated statistics
        """
        end_time = end_time or time.time()
        start_time = start_time or (end_time - 30 * 24 * 3600)  # 30 days
        
        # Filter records
        records = [
            r for r in self._usage_history
            if start_time <= r.timestamp <= end_time
        ]
        
        if not records:
            return UsageSummary(
                period_start=start_time,
                period_end=end_time,
                total_requests=0,
                successful_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_cost_usd=0.0,
                average_latency_ms=0.0,
            )
        
        # Aggregate by provider
        by_provider: dict[str, dict[str, Any]] = {}
        for r in records:
            if r.provider not in by_provider:
                by_provider[r.provider] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                }
            by_provider[r.provider]["requests"] += 1
            by_provider[r.provider]["tokens"] += r.total_tokens
            by_provider[r.provider]["cost_usd"] += r.cost_usd
        
        # Aggregate by model
        by_model: dict[str, dict[str, Any]] = {}
        for r in records:
            if r.model not in by_model:
                by_model[r.model] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                }
            by_model[r.model]["requests"] += 1
            by_model[r.model]["tokens"] += r.total_tokens
            by_model[r.model]["cost_usd"] += r.cost_usd
        
        return UsageSummary(
            period_start=start_time,
            period_end=end_time,
            total_requests=len(records),
            successful_requests=len([r for r in records if r.success]),
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
            total_tokens=sum(r.total_tokens for r in records),
            total_cost_usd=sum(r.cost_usd for r in records),
            average_latency_ms=sum(r.latency_ms for r in records) / len(records),
            by_provider=by_provider,
            by_model=by_model,
        )
    
    def export_report(
        self,
        output_path: str,
        format: str = "json",
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> None:
        """Export usage report to file.
        
        Args:
            output_path: Path to write report
            format: 'json' or 'csv'
            start_time: Start of period
            end_time: End of period
        """
        summary = self.get_usage_summary(start_time, end_time)
        
        if format == "json":
            report = {
                "summary": {
                    "period_start": summary.period_start,
                    "period_end": summary.period_end,
                    "total_requests": summary.total_requests,
                    "successful_requests": summary.successful_requests,
                    "total_tokens": summary.total_tokens,
                    "total_cost_usd": summary.total_cost_usd,
                },
                "by_provider": summary.by_provider,
                "by_model": summary.by_model,
            }
            Path(output_path).write_text(json.dumps(report, indent=2))
        
        elif format == "csv":
            end_time = end_time or time.time()
            start_time = start_time or (end_time - 30 * 24 * 3600)
            
            records = [
                r for r in self._usage_history
                if start_time <= r.timestamp <= end_time
            ]
            
            lines = ["timestamp,provider,model,input_tokens,output_tokens,total_tokens,cost_usd,latency_ms,success"]
            for r in records:
                lines.append(
                    f"{r.timestamp},{r.provider},{r.model},{r.input_tokens},"
                    f"{r.output_tokens},{r.total_tokens},{r.cost_usd:.6f},"
                    f"{r.latency_ms:.1f},{r.success}"
                )
            Path(output_path).write_text("\n".join(lines))
    
    def check_budget(self) -> tuple[bool, str]:
        """Check if within budget.
        
        Returns:
            Tuple of (within_budget, message)
        """
        if not self._budget_limit:
            return True, "No budget limit set"
        
        monthly_cost = self.get_monthly_cost()
        remaining = self._budget_limit - monthly_cost
        pct_used = (monthly_cost / self._budget_limit) * 100
        
        if monthly_cost >= self._budget_limit:
            return False, f"Budget exceeded: ${monthly_cost:.2f} / ${self._budget_limit:.2f}"
        
        return True, f"Budget OK: ${monthly_cost:.2f} / ${self._budget_limit:.2f} ({pct_used:.1f}% used, ${remaining:.2f} remaining)"


# =============================================================================
# Enhanced Local LLM Auto-Detection (Feature - LLM Router)
# =============================================================================


@dataclass
class LocalLLMInfo:
    """Detailed information about a detected local LLM."""
    
    provider: ProviderName
    endpoint: str
    available: bool
    models: list[str]
    default_model: str | None
    version: str | None
    gpu_enabled: bool
    context_length: int
    features: list[str]
    error: str | None = None


class EnhancedLocalLLMDetector:
    """Enhanced auto-detection for local LLM servers.
    
    Features:
    - Parallel endpoint scanning
    - Model capability detection
    - GPU status detection
    - Health monitoring
    - Auto-configuration
    """
    
    # Common ports to scan
    SCAN_PORTS = {
        "ollama": [11434],
        "lmstudio": [1234, 1235],
        "llama_cpp": [8080, 8000],
        "text-gen-webui": [5000, 7860],
        "vllm": [8000],
        "localai": [8080],
    }
    
    def __init__(self, timeout: float = 2.0) -> None:
        """Initialize detector with timeout."""
        self.timeout = timeout
        self._cache: dict[str, LocalLLMInfo] = {}
        self._last_scan: float = 0
        self._cache_ttl = 60.0  # 1 minute cache
    
    def detect_all(self, force_rescan: bool = False) -> list[LocalLLMInfo]:
        """Detect all available local LLM servers.
        
        Args:
            force_rescan: Force fresh scan ignoring cache
            
        Returns:
            List of detected local LLM info
        """
        # Check cache
        if not force_rescan and time.time() - self._last_scan < self._cache_ttl:
            return list(self._cache.values())
        
        results: list[LocalLLMInfo] = []
        
        # Scan each provider
        for provider, ports in self.SCAN_PORTS.items():
            for port in ports:
                info = self._probe_endpoint(provider, f"http://localhost:{port}")
                if info.available:
                    results.append(info)
                    self._cache[f"{provider}:{port}"] = info
                    break  # Found one for this provider
        
        self._last_scan = time.time()
        return results
    
    def _probe_endpoint(self, provider: str, endpoint: str) -> LocalLLMInfo:
        """Probe a specific endpoint for LLM server."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                if provider == "ollama":
                    return self._probe_ollama(client, endpoint)
                elif provider == "lmstudio":
                    return self._probe_lmstudio(client, endpoint)
                elif provider == "llama_cpp":
                    return self._probe_llama_cpp(client, endpoint)
                elif provider == "vllm":
                    return self._probe_vllm(client, endpoint)
                else:
                    return self._probe_openai_compatible(client, endpoint, provider)
        except Exception as e:
            return LocalLLMInfo(
                provider=provider,  # type: ignore
                endpoint=endpoint,
                available=False,
                models=[],
                default_model=None,
                version=None,
                gpu_enabled=False,
                context_length=0,
                features=[],
                error=str(e),
            )
    
    def _probe_ollama(self, client: httpx.Client, endpoint: str) -> LocalLLMInfo:
        """Probe Ollama server."""
        try:
            # Get server info
            response = client.get(f"{endpoint}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama not responding")
            
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            # Check if GPU is enabled (via Ollama API)
            gpu_enabled = False
            try:
                response = client.get(f"{endpoint}/api/version")
                if response.status_code == 200:
                    version_data = response.json()
                    # Ollama doesn't directly expose GPU info, but we can infer
                    gpu_enabled = "cuda" in str(version_data).lower() or "gpu" in str(version_data).lower()
            except Exception:
                pass
            
            return LocalLLMInfo(
                provider="ollama",
                endpoint=endpoint,
                available=True,
                models=models,
                default_model=models[0] if models else None,
                version=None,
                gpu_enabled=gpu_enabled,
                context_length=32768,
                features=["streaming", "chat", "embeddings", "vision"],
            )
        except Exception as e:
            return LocalLLMInfo(
                provider="ollama",
                endpoint=endpoint,
                available=False,
                models=[],
                default_model=None,
                version=None,
                gpu_enabled=False,
                context_length=0,
                features=[],
                error=str(e),
            )
    
    def _probe_lmstudio(self, client: httpx.Client, endpoint: str) -> LocalLLMInfo:
        """Probe LM Studio server."""
        try:
            response = client.get(f"{endpoint}/v1/models")
            if response.status_code != 200:
                raise ConnectionError("LM Studio not responding")
            
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            
            return LocalLLMInfo(
                provider="lmstudio",
                endpoint=endpoint,
                available=True,
                models=models,
                default_model=models[0] if models else None,
                version=None,
                gpu_enabled=True,  # LM Studio typically uses GPU
                context_length=32768,
                features=["streaming", "chat", "openai_compatible"],
            )
        except Exception as e:
            return LocalLLMInfo(
                provider="lmstudio",
                endpoint=endpoint,
                available=False,
                models=[],
                default_model=None,
                version=None,
                gpu_enabled=False,
                context_length=0,
                features=[],
                error=str(e),
            )
    
    def _probe_llama_cpp(self, client: httpx.Client, endpoint: str) -> LocalLLMInfo:
        """Probe llama.cpp server."""
        try:
            response = client.get(f"{endpoint}/health")
            if response.status_code != 200:
                raise ConnectionError("llama.cpp not responding")
            
            # Get model info
            model_name = "unknown"
            gpu_enabled = False
            context_length = 4096
            
            try:
                props_response = client.get(f"{endpoint}/props")
                if props_response.status_code == 200:
                    props = props_response.json()
                    model_name = props.get("model_name", "unknown")
                    context_length = props.get("n_ctx", 4096)
            except Exception:
                pass
            
            return LocalLLMInfo(
                provider="llama_cpp",
                endpoint=endpoint,
                available=True,
                models=[model_name],
                default_model=model_name,
                version=None,
                gpu_enabled=gpu_enabled,
                context_length=context_length,
                features=["streaming", "completion", "embeddings"],
            )
        except Exception as e:
            return LocalLLMInfo(
                provider="llama_cpp",
                endpoint=endpoint,
                available=False,
                models=[],
                default_model=None,
                version=None,
                gpu_enabled=False,
                context_length=0,
                features=[],
                error=str(e),
            )
    
    def _probe_vllm(self, client: httpx.Client, endpoint: str) -> LocalLLMInfo:
        """Probe vLLM server."""
        try:
            response = client.get(f"{endpoint}/v1/models")
            if response.status_code != 200:
                raise ConnectionError("vLLM not responding")
            
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            
            return LocalLLMInfo(
                provider="ollama",  # vLLM uses OpenAI-compatible API
                endpoint=endpoint,
                available=True,
                models=models,
                default_model=models[0] if models else None,
                version=None,
                gpu_enabled=True,  # vLLM requires GPU
                context_length=32768,
                features=["streaming", "chat", "openai_compatible", "high_throughput"],
            )
        except Exception as e:
            return LocalLLMInfo(
                provider="ollama",
                endpoint=endpoint,
                available=False,
                models=[],
                default_model=None,
                version=None,
                gpu_enabled=False,
                context_length=0,
                features=[],
                error=str(e),
            )
    
    def _probe_openai_compatible(
        self,
        client: httpx.Client,
        endpoint: str,
        provider: str,
    ) -> LocalLLMInfo:
        """Probe generic OpenAI-compatible server."""
        try:
            response = client.get(f"{endpoint}/v1/models")
            if response.status_code != 200:
                raise ConnectionError(f"{provider} not responding")
            
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            
            return LocalLLMInfo(
                provider="ollama",  # Treat as ollama-like
                endpoint=endpoint,
                available=True,
                models=models,
                default_model=models[0] if models else None,
                version=None,
                gpu_enabled=False,
                context_length=4096,
                features=["streaming", "chat", "openai_compatible"],
            )
        except Exception as e:
            return LocalLLMInfo(
                provider="ollama",
                endpoint=endpoint,
                available=False,
                models=[],
                default_model=None,
                version=None,
                gpu_enabled=False,
                context_length=0,
                features=[],
                error=str(e),
            )
    
    def get_best_local_llm(self) -> LocalLLMInfo | None:
        """Get the best available local LLM.
        
        Prioritizes:
        1. GPU-enabled servers
        2. Servers with more models
        3. Higher context length
        
        Returns:
            Best LocalLLMInfo or None if none available
        """
        available = [info for info in self.detect_all() if info.available]
        
        if not available:
            return None
        
        # Score each option
        def score(info: LocalLLMInfo) -> float:
            s = 0.0
            if info.gpu_enabled:
                s += 100
            s += len(info.models) * 10
            s += info.context_length / 1000
            return s
        
        return max(available, key=score)
    
    def auto_configure_router(self, router: LLMRouter) -> str | None:
        """Auto-configure router with best available local LLM.
        
        Args:
            router: LLMRouter to configure
            
        Returns:
            Configured provider name or None
        """
        best = self.get_best_local_llm()
        
        if not best:
            return None
        
        # Configure the provider
        provider = router.registry.get(best.provider)
        if hasattr(provider, "set_endpoint"):
            provider.set_endpoint(best.endpoint)  # type: ignore
        
        return best.provider


# =============================================================================
# Enhanced Streaming Response Handler (Feature - LLM Router)
# =============================================================================


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""
    
    content: str
    index: int
    timestamp: float
    token_count: int = 1  # Estimated
    finish_reason: str | None = None
    tool_call: FunctionCall | None = None


@dataclass
class StreamMetrics:
    """Metrics for a streaming response."""
    
    total_chunks: int
    total_tokens: int
    first_token_latency_ms: float
    total_latency_ms: float
    tokens_per_second: float
    average_chunk_size: float


class StreamingResponseHandler:
    """Advanced streaming response handler.
    
    Features:
    - Token-by-token callbacks
    - Buffered line callbacks
    - Progress tracking
    - Cancellation support
    - Metrics collection
    """
    
    def __init__(
        self,
        on_token: Callable[[str], None] | None = None,
        on_line: Callable[[str], None] | None = None,
        on_complete: Callable[[str, StreamMetrics], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        buffer_lines: bool = False,
    ) -> None:
        """Initialize handler.
        
        Args:
            on_token: Callback for each token
            on_line: Callback for complete lines (if buffer_lines=True)
            on_complete: Callback when streaming completes
            on_error: Callback on error
            buffer_lines: Whether to buffer and emit complete lines
        """
        self.on_token = on_token
        self.on_line = on_line
        self.on_complete = on_complete
        self.on_error = on_error
        self.buffer_lines = buffer_lines
        
        self._chunks: list[StreamChunk] = []
        self._line_buffer = ""
        self._cancelled = False
        self._start_time: float | None = None
        self._first_token_time: float | None = None
    
    def cancel(self) -> None:
        """Cancel the streaming response."""
        self._cancelled = True
    
    def is_cancelled(self) -> bool:
        """Check if streaming was cancelled."""
        return self._cancelled
    
    def handle_chunk(self, content: str, finish_reason: str | None = None) -> bool:
        """Handle a streaming chunk.
        
        Args:
            content: Chunk content
            finish_reason: If set, indicates stream end
            
        Returns:
            False if cancelled, True otherwise
        """
        if self._cancelled:
            return False
        
        now = time.time()
        
        if self._start_time is None:
            self._start_time = now
        
        if self._first_token_time is None and content:
            self._first_token_time = now
        
        # Create chunk
        chunk = StreamChunk(
            content=content,
            index=len(self._chunks),
            timestamp=now,
            finish_reason=finish_reason,
        )
        self._chunks.append(chunk)
        
        # Token callback
        if self.on_token and content:
            self.on_token(content)
        
        # Line buffering
        if self.buffer_lines and self.on_line:
            self._line_buffer += content
            while "\n" in self._line_buffer:
                line, self._line_buffer = self._line_buffer.split("\n", 1)
                self.on_line(line)
        
        return True
    
    def finish(self) -> tuple[str, StreamMetrics]:
        """Finish streaming and return results.
        
        Returns:
            Tuple of (full_text, metrics)
        """
        now = time.time()
        full_text = "".join(c.content for c in self._chunks)
        
        # Flush remaining line buffer
        if self.buffer_lines and self.on_line and self._line_buffer:
            self.on_line(self._line_buffer)
        
        # Calculate metrics
        total_latency = (now - self._start_time) * 1000 if self._start_time else 0
        first_token_latency = (
            (self._first_token_time - self._start_time) * 1000
            if self._first_token_time and self._start_time
            else 0
        )
        
        total_tokens = len(self._chunks)
        tokens_per_second = (
            total_tokens / (total_latency / 1000) if total_latency > 0 else 0
        )
        
        avg_chunk_size = (
            len(full_text) / len(self._chunks) if self._chunks else 0
        )
        
        metrics = StreamMetrics(
            total_chunks=len(self._chunks),
            total_tokens=total_tokens,
            first_token_latency_ms=first_token_latency,
            total_latency_ms=total_latency,
            tokens_per_second=tokens_per_second,
            average_chunk_size=avg_chunk_size,
        )
        
        # Completion callback
        if self.on_complete:
            self.on_complete(full_text, metrics)
        
        return full_text, metrics
    
    def handle_error(self, error: str) -> None:
        """Handle streaming error.
        
        Args:
            error: Error message
        """
        if self.on_error:
            self.on_error(error)


def stream_with_handler(
    router: LLMRouter,
    request: LLMRequest,
    handler: StreamingResponseHandler,
) -> LLMResponse:
    """Stream a request with advanced handler.
    
    Args:
        router: LLM router
        request: Request to stream
        handler: Stream handler
        
    Returns:
        LLMResponse with complete text
    """
    request.stream = True
    
    provider = router._pick_provider(request)
    router._ensure_local_available(provider)
    router.consent_gate.require_consent(provider)
    router.api_keys.validate(provider)
    api_key = router.api_keys.get_api_key(provider)
    
    def chunk_callback(content: str) -> None:
        if not handler.handle_chunk(content):
            raise InterruptedError("Streaming cancelled")
    
    try:
        response = provider.stream_send(request, api_key, callback=chunk_callback)
        full_text, metrics = handler.finish()
        
        # Update response with full text
        response.text = full_text
        
        return response
    except InterruptedError:
        full_text, metrics = handler.finish()
        return LLMResponse(
            text=full_text,
            provider=provider.name,
            model=request.model or "",
            latency_ms=metrics.total_latency_ms,
            error="Streaming cancelled",
            is_streaming=True,
        )
    except Exception as e:
        handler.handle_error(str(e))
        return LLMResponse(
            text="",
            provider=provider.name,
            model=request.model or "",
            latency_ms=0,
            error=str(e),
        )


# =============================================================================
# PROVIDER EDGE CASES (2% Gap Coverage)
# Rate Limiting, Retries, Circuit Breaker, Health Monitoring, Connection Pooling
# =============================================================================


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
        limit_type: str = "requests",
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.limit_type = limit_type  # "requests", "tokens", "concurrent"


class ProviderUnavailableError(Exception):
    """Raised when provider is temporarily unavailable (circuit open)."""
    
    def __init__(
        self,
        provider: str,
        message: str,
        recovery_time: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.recovery_time = recovery_time


class AuthenticationError(Exception):
    """Raised when API key is invalid or expired."""
    
    def __init__(
        self,
        provider: str,
        message: str,
        is_expired: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.is_expired = is_expired


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    max_concurrent: int = 10
    burst_allowance: float = 1.2  # Allow 20% burst
    cooldown_seconds: float = 1.0  # Base cooldown after hitting limit


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)
    retry_on_exceptions: tuple[type, ...] = (
        ConnectionError, TimeoutError, OSError
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes to close circuit
    half_open_timeout_seconds: float = 30.0  # Time before trying again
    failure_window_seconds: float = 60.0  # Window for counting failures


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ProviderHealthStatus:
    """Health status of a provider."""
    
    provider: ProviderName
    is_healthy: bool
    circuit_state: CircuitState
    last_success: float | None
    last_failure: float | None
    failure_count: int
    success_count: int
    average_latency_ms: float
    error_rate: float  # 0-1
    current_concurrency: int
    rate_limit_remaining: int | None
    last_check: float


class TokenBucket:
    """Token bucket rate limiter with burst support."""
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,  # Tokens per second
        burst_multiplier: float = 1.2,
    ) -> None:
        """Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            burst_multiplier: Allow burst up to capacity * multiplier
        """
        self._capacity = capacity
        self._burst_capacity = int(capacity * burst_multiplier)
        self._refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            self._burst_capacity,
            self._tokens + elapsed * self._refill_rate
        )
        self._last_update = now
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without blocking.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False if insufficient tokens
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False
    
    def acquire(
        self,
        tokens: int = 1,
        timeout: float | None = None,
    ) -> bool:
        """Acquire tokens, optionally waiting.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if self.try_acquire(tokens):
                return True
            
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            # Calculate wait time until enough tokens
            with self._lock:
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._refill_rate
            
            # Wait with small intervals for responsiveness
            time.sleep(min(wait_time, 0.1))
    
    def wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for tokens.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Estimated wait time in seconds (0 if available now)
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            tokens_needed = tokens - self._tokens
            return tokens_needed / self._refill_rate
    
    @property
    def available(self) -> float:
        """Get currently available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class ConcurrencyLimiter:
    """Limits concurrent requests to a provider."""
    
    def __init__(self, max_concurrent: int) -> None:
        """Initialize limiter.
        
        Args:
            max_concurrent: Maximum concurrent requests
        """
        self._max_concurrent = max_concurrent
        self._current = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
    
    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a slot for a request.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if acquired, False if timeout
        """
        with self._condition:
            start = time.time()
            while self._current >= self._max_concurrent:
                remaining = None
                if timeout is not None:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        return False
                self._condition.wait(timeout=remaining)
            self._current += 1
            return True
    
    def release(self) -> None:
        """Release a slot after request completion."""
        with self._condition:
            self._current = max(0, self._current - 1)
            self._condition.notify()
    
    @property
    def current(self) -> int:
        """Current number of concurrent requests."""
        with self._lock:
            return self._current
    
    @contextmanager
    def acquire_context(self, timeout: float | None = None) -> Generator[bool, None, None]:
        """Context manager for acquiring/releasing slots.
        
        Args:
            timeout: Maximum time to wait
            
        Yields:
            True if acquired, False if timeout
        """
        acquired = self.acquire(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()


class CircuitBreaker:
    """Circuit breaker for provider failure protection."""
    
    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_state_change = time.time()
        self._failure_times: list[float] = []
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_half_open()
            return self._state
    
    def _check_half_open(self) -> None:
        """Check if should transition to half-open."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_state_change
            if elapsed >= self._config.half_open_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._last_state_change = time.time()
                self._success_count = 0
    
    def _clean_old_failures(self) -> None:
        """Remove failures outside the window."""
        cutoff = time.time() - self._config.failure_window_seconds
        self._failure_times = [t for t in self._failure_times if t > cutoff]
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed.
        
        Returns:
            True if request allowed, False if circuit is open
        """
        with self._lock:
            self._check_half_open()
            return self._state != CircuitState.OPEN
    
    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._success_count += 1
            
            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self._config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._last_state_change = time.time()
                    self._failure_count = 0
                    self._failure_times.clear()
            elif self._state == CircuitState.CLOSED:
                # Decay failure count on success
                self._failure_count = max(0, self._failure_count - 1)
    
    def record_failure(self) -> None:
        """Record a failed request."""
        now = time.time()
        with self._lock:
            self._failure_count += 1
            self._failure_times.append(now)
            self._last_failure_time = now
            self._clean_old_failures()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                self._state = CircuitState.OPEN
                self._last_state_change = now
            elif self._state == CircuitState.CLOSED:
                # Check if should open
                if len(self._failure_times) >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = now
    
    def get_recovery_time(self) -> float | None:
        """Get estimated time until circuit might close.
        
        Returns:
            Seconds until half-open, None if already closed/half-open
        """
        with self._lock:
            if self._state != CircuitState.OPEN:
                return None
            elapsed = time.time() - self._last_state_change
            remaining = self._config.half_open_timeout_seconds - elapsed
            return max(0, remaining)


@dataclass
class ProviderRateLimitState:
    """State for provider rate limiting."""
    
    request_bucket: TokenBucket
    token_bucket: TokenBucket
    concurrency_limiter: ConcurrencyLimiter
    circuit_breaker: CircuitBreaker
    latency_samples: list[float]
    error_count: int = 0
    success_count: int = 0
    last_rate_limit_error: float | None = None


class ProviderRateLimiter:
    """Rate limiter for all providers with health tracking."""
    
    # Default limits per provider
    DEFAULT_LIMITS: dict[ProviderName, RateLimitConfig] = {
        "openai": RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=150000,
            max_concurrent=20,
        ),
        "anthropic": RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100000,
            max_concurrent=10,
        ),
        "groq": RateLimitConfig(
            requests_per_minute=30,
            tokens_per_minute=30000,
            max_concurrent=5,
        ),
        "together": RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100000,
            max_concurrent=15,
        ),
        "mistral": RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100000,
            max_concurrent=10,
        ),
        "cohere": RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=200000,
            max_concurrent=20,
        ),
        "azure_openai": RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=200000,
            max_concurrent=30,
        ),
        "ollama": RateLimitConfig(
            requests_per_minute=1000,  # Local, essentially unlimited
            tokens_per_minute=1000000,
            max_concurrent=1,  # But single model at a time
        ),
        "lmstudio": RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=1000000,
            max_concurrent=1,
        ),
        "llama_cpp": RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=1000000,
            max_concurrent=1,
        ),
    }
    
    def __init__(self) -> None:
        """Initialize rate limiter."""
        self._provider_states: dict[ProviderName, ProviderRateLimitState] = {}
        self._lock = threading.Lock()
    
    def _get_state(self, provider: ProviderName) -> ProviderRateLimitState:
        """Get or create state for a provider."""
        with self._lock:
            if provider not in self._provider_states:
                config = self.DEFAULT_LIMITS.get(
                    provider,
                    RateLimitConfig()  # Default values
                )
                
                # Create rate limiters
                request_bucket = TokenBucket(
                    capacity=config.requests_per_minute,
                    refill_rate=config.requests_per_minute / 60.0,
                    burst_multiplier=config.burst_allowance,
                )
                token_bucket = TokenBucket(
                    capacity=config.tokens_per_minute,
                    refill_rate=config.tokens_per_minute / 60.0,
                    burst_multiplier=config.burst_allowance,
                )
                concurrency_limiter = ConcurrencyLimiter(config.max_concurrent)
                circuit_breaker = CircuitBreaker()
                
                self._provider_states[provider] = ProviderRateLimitState(
                    request_bucket=request_bucket,
                    token_bucket=token_bucket,
                    concurrency_limiter=concurrency_limiter,
                    circuit_breaker=circuit_breaker,
                    latency_samples=[],
                )
            
            return self._provider_states[provider]
    
    def check_rate_limit(
        self,
        provider: ProviderName,
        estimated_tokens: int = 1000,
        timeout: float = 30.0,
    ) -> None:
        """Check and enforce rate limits before a request.
        
        Args:
            provider: Provider name
            estimated_tokens: Estimated tokens for request
            timeout: Maximum time to wait for rate limit
            
        Raises:
            RateLimitError: If rate limit exceeded and timeout
            ProviderUnavailableError: If circuit is open
        """
        state = self._get_state(provider)
        
        # Check circuit breaker
        if not state.circuit_breaker.allow_request():
            recovery = state.circuit_breaker.get_recovery_time()
            raise ProviderUnavailableError(
                provider=provider,
                message=f"Provider {provider} is temporarily unavailable (circuit open)",
                recovery_time=recovery,
            )
        
        # Check request rate limit
        if not state.request_bucket.acquire(1, timeout=timeout):
            wait = state.request_bucket.wait_time(1)
            raise RateLimitError(
                f"Request rate limit exceeded for {provider}",
                retry_after_seconds=wait,
                limit_type="requests",
            )
        
        # Check token rate limit
        if not state.token_bucket.acquire(estimated_tokens, timeout=timeout):
            wait = state.token_bucket.wait_time(estimated_tokens)
            raise RateLimitError(
                f"Token rate limit exceeded for {provider}",
                retry_after_seconds=wait,
                limit_type="tokens",
            )
    
    def acquire_concurrency(
        self,
        provider: ProviderName,
        timeout: float = 30.0,
    ) -> bool:
        """Acquire a concurrency slot.
        
        Args:
            provider: Provider name
            timeout: Maximum time to wait
            
        Returns:
            True if acquired, False if timeout
        """
        state = self._get_state(provider)
        return state.concurrency_limiter.acquire(timeout)
    
    def release_concurrency(self, provider: ProviderName) -> None:
        """Release a concurrency slot."""
        state = self._get_state(provider)
        state.concurrency_limiter.release()
    
    def record_success(
        self,
        provider: ProviderName,
        latency_ms: float,
    ) -> None:
        """Record a successful request."""
        state = self._get_state(provider)
        state.circuit_breaker.record_success()
        state.success_count += 1
        state.latency_samples.append(latency_ms)
        
        # Keep last 100 samples
        if len(state.latency_samples) > 100:
            state.latency_samples = state.latency_samples[-100:]
    
    def record_failure(
        self,
        provider: ProviderName,
        is_rate_limit: bool = False,
    ) -> None:
        """Record a failed request."""
        state = self._get_state(provider)
        state.error_count += 1
        
        if is_rate_limit:
            state.last_rate_limit_error = time.time()
        else:
            state.circuit_breaker.record_failure()
    
    def get_health_status(self, provider: ProviderName) -> ProviderHealthStatus:
        """Get health status for a provider."""
        state = self._get_state(provider)
        
        total_requests = state.success_count + state.error_count
        error_rate = state.error_count / max(total_requests, 1)
        avg_latency = (
            sum(state.latency_samples) / len(state.latency_samples)
            if state.latency_samples else 0.0
        )
        
        return ProviderHealthStatus(
            provider=provider,
            is_healthy=state.circuit_breaker.state == CircuitState.CLOSED,
            circuit_state=state.circuit_breaker.state,
            last_success=None,  # Could track if needed
            last_failure=state.last_rate_limit_error,
            failure_count=state.error_count,
            success_count=state.success_count,
            average_latency_ms=avg_latency,
            error_rate=error_rate,
            current_concurrency=state.concurrency_limiter.current,
            rate_limit_remaining=int(state.request_bucket.available),
            last_check=time.time(),
        )
    
    def set_custom_limits(
        self,
        provider: ProviderName,
        config: RateLimitConfig,
    ) -> None:
        """Set custom rate limits for a provider.
        
        Args:
            provider: Provider name
            config: Rate limit configuration
        """
        with self._lock:
            # Remove existing state to force recreation with new config
            if provider in self._provider_states:
                del self._provider_states[provider]
            
            # Pre-create with new config
            self.DEFAULT_LIMITS[provider] = config


class RetryHandler:
    """Handles retries with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig | None = None) -> None:
        """Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self._config = config or RetryConfig()
        self._random = random.Random()
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt.
        
        Args:
            attempt: Attempt number (0 = first retry)
            
        Returns:
            Delay in seconds
        """
        delay = self._config.initial_delay_seconds * (
            self._config.exponential_base ** attempt
        )
        delay = min(delay, self._config.max_delay_seconds)
        
        if self._config.jitter:
            # Add 0-50% jitter
            jitter = delay * self._random.uniform(0, 0.5)
            delay += jitter
        
        return delay
    
    def should_retry(
        self,
        attempt: int,
        error: Exception | None = None,
        status_code: int | None = None,
    ) -> bool:
        """Determine if should retry.
        
        Args:
            attempt: Current attempt number
            error: Exception that occurred
            status_code: HTTP status code if applicable
            
        Returns:
            True if should retry
        """
        if attempt >= self._config.max_retries:
            return False
        
        if status_code and status_code in self._config.retry_on_status:
            return True
        
        if error and isinstance(error, self._config.retry_on_exceptions):
            return True
        
        return False
    
    def execute_with_retry(
        self,
        func: Callable[[], T],
        on_retry: Callable[[int, float, Exception | None], None] | None = None,
    ) -> T:
        """Execute a function with retry logic.
        
        Args:
            func: Function to execute
            on_retry: Callback before each retry (attempt, delay, error)
            
        Returns:
            Result from successful function call
            
        Raises:
            Last exception if all retries exhausted
        """
        last_error: Exception | None = None
        
        for attempt in range(self._config.max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_error = e
                
                # Check if we should retry
                status_code = getattr(e, "status_code", None)
                if not self.should_retry(attempt, e, status_code):
                    raise
                
                # Calculate delay
                delay = self.calculate_delay(attempt)
                
                # Callback
                if on_retry:
                    on_retry(attempt, delay, e)
                
                # Wait before retry
                time.sleep(delay)
        
        # Should not reach here, but raise last error if we do
        if last_error:
            raise last_error
        raise RuntimeError("Retry logic error")


@dataclass
class FallbackChainConfig:
    """Configuration for fallback chain."""
    
    primary_provider: ProviderName
    fallback_providers: list[ProviderName]
    fallback_on_rate_limit: bool = True
    fallback_on_error: bool = True
    fallback_on_timeout: bool = True
    preserve_model: bool = False  # Try same model on fallback


class FallbackChainManager:
    """Manages provider fallback chains."""
    
    def __init__(
        self,
        rate_limiter: ProviderRateLimiter | None = None,
    ) -> None:
        """Initialize fallback manager.
        
        Args:
            rate_limiter: Rate limiter for health checks
        """
        self._rate_limiter = rate_limiter or ProviderRateLimiter()
        self._chains: dict[str, FallbackChainConfig] = {}
    
    def register_chain(
        self,
        name: str,
        config: FallbackChainConfig,
    ) -> None:
        """Register a fallback chain.
        
        Args:
            name: Chain name
            config: Chain configuration
        """
        self._chains[name] = config
    
    def get_providers_in_order(
        self,
        chain_name: str | None = None,
        primary: ProviderName | None = None,
        fallbacks: list[ProviderName] | None = None,
    ) -> list[ProviderName]:
        """Get providers to try in order.
        
        Args:
            chain_name: Named chain to use
            primary: Override primary provider
            fallbacks: Override fallback providers
            
        Returns:
            List of providers to try in order (healthy ones first)
        """
        if chain_name and chain_name in self._chains:
            config = self._chains[chain_name]
            providers = [config.primary_provider] + config.fallback_providers
        elif primary:
            providers = [primary] + (fallbacks or [])
        else:
            # Default chain
            providers = [
                "openai",
                "anthropic",
                "groq",
                "together",
                "ollama",
            ]
        
        # Sort by health status
        def health_score(p: ProviderName) -> float:
            status = self._rate_limiter.get_health_status(p)
            score = 0.0
            
            if status.is_healthy:
                score += 100
            if status.circuit_state == CircuitState.CLOSED:
                score += 50
            elif status.circuit_state == CircuitState.HALF_OPEN:
                score += 25
            
            # Factor in error rate
            score -= status.error_rate * 30
            
            # Factor in latency (prefer faster)
            if status.average_latency_ms > 0:
                score -= min(status.average_latency_ms / 100, 20)
            
            return score
        
        return sorted(providers, key=health_score, reverse=True)
    
    def should_fallback(
        self,
        error: Exception,
        config: FallbackChainConfig | None = None,
    ) -> bool:
        """Determine if should fallback based on error.
        
        Args:
            error: The error that occurred
            config: Chain configuration (uses defaults if None)
            
        Returns:
            True if should try fallback
        """
        if config is None:
            return True
        
        if isinstance(error, RateLimitError) and config.fallback_on_rate_limit:
            return True
        
        if isinstance(error, TimeoutError) and config.fallback_on_timeout:
            return True
        
        if config.fallback_on_error:
            return True
        
        return False


class ConnectionPoolManager:
    """Manages connection pools for providers."""
    
    def __init__(self) -> None:
        """Initialize connection pool manager."""
        self._pools: dict[str, httpx.Client] = {}
        self._pool_configs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def _get_pool_key(
        self,
        provider: ProviderName,
        endpoint: str | None = None,
    ) -> str:
        """Get unique key for a pool."""
        return f"{provider}:{endpoint or 'default'}"
    
    def get_client(
        self,
        provider: ProviderName,
        endpoint: str | None = None,
        timeout: float = 60.0,
        max_connections: int = 10,
        max_keepalive: int = 5,
    ) -> httpx.Client:
        """Get or create a pooled HTTP client.
        
        Args:
            provider: Provider name
            endpoint: Custom endpoint (optional)
            timeout: Request timeout
            max_connections: Maximum connections in pool
            max_keepalive: Maximum keepalive connections
            
        Returns:
            HTTP client with connection pooling
        """
        key = self._get_pool_key(provider, endpoint)
        
        with self._lock:
            if key not in self._pools:
                limits = httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive,
                )
                
                self._pools[key] = httpx.Client(
                    timeout=timeout,
                    limits=limits,
                    http2=True,  # Enable HTTP/2 for better multiplexing
                )
                
                self._pool_configs[key] = {
                    "provider": provider,
                    "endpoint": endpoint,
                    "timeout": timeout,
                    "max_connections": max_connections,
                }
            
            return self._pools[key]
    
    def close_pool(
        self,
        provider: ProviderName,
        endpoint: str | None = None,
    ) -> None:
        """Close a connection pool.
        
        Args:
            provider: Provider name
            endpoint: Custom endpoint (optional)
        """
        key = self._get_pool_key(provider, endpoint)
        
        with self._lock:
            if key in self._pools:
                try:
                    self._pools[key].close()
                except Exception:
                    pass
                del self._pools[key]
                del self._pool_configs[key]
    
    def close_all(self) -> None:
        """Close all connection pools."""
        with self._lock:
            for client in self._pools.values():
                try:
                    client.close()
                except Exception:
                    pass
            self._pools.clear()
            self._pool_configs.clear()
    
    def get_pool_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all pools.
        
        Returns:
            Dict mapping pool key to stats
        """
        with self._lock:
            return dict(self._pool_configs)


class ResilientLLMRouter:
    """Enhanced LLM router with resilience features.
    
    Features:
    - Rate limiting with token bucket
    - Automatic retries with exponential backoff
    - Circuit breaker for failing providers
    - Fallback chains
    - Connection pooling
    - Health monitoring
    """
    
    def __init__(
        self,
        base_router: LLMRouter | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize resilient router.
        
        Args:
            base_router: Underlying LLM router
            retry_config: Retry configuration
        """
        self._router = base_router or build_router()
        self._rate_limiter = ProviderRateLimiter()
        self._retry_handler = RetryHandler(retry_config)
        self._fallback_manager = FallbackChainManager(self._rate_limiter)
        self._pool_manager = ConnectionPoolManager()
        self._request_counter = 0
        self._lock = threading.Lock()
    
    def route(
        self,
        request: LLMRequest,
        fallback_providers: list[ProviderName] | None = None,
        retry: bool = True,
        use_rate_limiting: bool = True,
    ) -> LLMResponse:
        """Route request with resilience features.
        
        Args:
            request: LLM request
            fallback_providers: Providers to try on failure
            retry: Whether to retry on transient failures
            use_rate_limiting: Whether to enforce rate limits
            
        Returns:
            LLM response
        """
        provider_name: ProviderName = (
            request.provider or self._router.settings.llm.provider or "openai"
        )
        
        # Get providers to try
        providers = self._fallback_manager.get_providers_in_order(
            primary=provider_name,
            fallbacks=fallback_providers,
        )
        
        last_error: Exception | None = None
        
        for provider in providers:
            try:
                # Check rate limits
                if use_rate_limiting:
                    estimated_tokens = len(request.prompt.split()) * 2  # Rough estimate
                    self._rate_limiter.check_rate_limit(
                        provider,
                        estimated_tokens=estimated_tokens,
                    )
                
                # Acquire concurrency slot
                if not self._rate_limiter.acquire_concurrency(provider, timeout=30.0):
                    continue  # Try next provider
                
                try:
                    # Execute with retry
                    def do_request() -> LLMResponse:
                        req = LLMRequest(
                            prompt=request.prompt,
                            model=request.model,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            provider=provider,
                            system_prompt=request.system_prompt,
                            metadata=request.metadata,
                            functions=request.functions,
                            tools=request.tools,
                            tool_choice=request.tool_choice,
                            stream=request.stream,
                        )
                        return self._router.route(req)
                    
                    if retry:
                        response = self._retry_handler.execute_with_retry(do_request)
                    else:
                        response = do_request()
                    
                    # Record success
                    if not response.error:
                        self._rate_limiter.record_success(provider, response.latency_ms)
                        return response
                    else:
                        raise RuntimeError(response.error)
                        
                finally:
                    self._rate_limiter.release_concurrency(provider)
                    
            except RateLimitError as e:
                last_error = e
                self._rate_limiter.record_failure(provider, is_rate_limit=True)
                continue  # Try next provider
                
            except ProviderUnavailableError as e:
                last_error = e
                continue  # Try next provider
                
            except Exception as e:
                last_error = e
                self._rate_limiter.record_failure(provider)
                
                if self._fallback_manager.should_fallback(e):
                    continue
                raise
        
        # All providers failed
        return LLMResponse(
            text="",
            provider="none",
            model="",
            latency_ms=0,
            error=f"All providers failed: {last_error}",
        )
    
    def get_all_health_status(self) -> dict[ProviderName, ProviderHealthStatus]:
        """Get health status for all providers.
        
        Returns:
            Dict mapping provider name to health status
        """
        result: dict[ProviderName, ProviderHealthStatus] = {}
        
        for provider in self._router.registry.list_providers():
            result[provider] = self._rate_limiter.get_health_status(provider)
        
        return result
    
    def reset_circuit_breaker(self, provider: ProviderName) -> None:
        """Manually reset circuit breaker for a provider.
        
        Args:
            provider: Provider name
        """
        state = self._rate_limiter._get_state(provider)
        with state.circuit_breaker._lock:
            state.circuit_breaker._state = CircuitState.CLOSED
            state.circuit_breaker._failure_count = 0
            state.circuit_breaker._failure_times.clear()
    
    def close(self) -> None:
        """Clean up resources."""
        self._pool_manager.close_all()


# Singleton instances for convenience
_global_rate_limiter: ProviderRateLimiter | None = None
_global_connection_pool: ConnectionPoolManager | None = None


def get_global_rate_limiter() -> ProviderRateLimiter:
    """Get or create global rate limiter."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = ProviderRateLimiter()
    return _global_rate_limiter


def get_global_connection_pool() -> ConnectionPoolManager:
    """Get or create global connection pool manager."""
    global _global_connection_pool
    if _global_connection_pool is None:
        _global_connection_pool = ConnectionPoolManager()
    return _global_connection_pool


def build_resilient_router(
    settings: Settings | None = None,
    retry_config: RetryConfig | None = None,
) -> ResilientLLMRouter:
    """Build a resilient LLM router with all edge case handling.
    
    Args:
        settings: Configuration settings
        retry_config: Retry configuration
        
    Returns:
        Configured ResilientLLMRouter
    """
    base_router = LLMRouter(settings=settings or config_service.load())
    return ResilientLLMRouter(
        base_router=base_router,
        retry_config=retry_config,
    )
