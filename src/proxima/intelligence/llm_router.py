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
        check_endpoint = (
            endpoint or self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        )
        try:
            client = self._get_client()
            response = client.get(f"{check_endpoint}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def stream_send(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        callback: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a streaming request to Ollama server."""
        start = time.perf_counter()

        endpoint = self._endpoint or f"http://localhost:{DEFAULT_PORTS['ollama']}"
        model = request.model or self.default_model

        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "stream": True,
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
            collected_text = ""
            chunks: list[str] = []
            total_tokens = 0

            with client.stream(
                "POST",
                f"{endpoint}/api/generate",
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

                    text = data.get("response", "")
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
