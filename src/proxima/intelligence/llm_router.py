"""LLM routing and provider abstraction.

Implements the Step 3.1 architecture from proper_implementation_steps.md:
- ProviderRegistry with pluggable providers (OpenAI, Anthropic, Ollama, LM Studio)
- LocalLLMDetector for lightweight endpoint checks
- APIKeyManager for pulling keys from configured env vars
- ConsentGate to enforce local/remote consent rules
- LLMRouter that chooses the provider, enforces consent, and dispatches requests

This is an architectural scaffold. Provider senders are lightweight stubs that
return structured responses without performing real network calls. The intent
is to make routing testable now and to be swapped with real provider clients
later.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Protocol, Tuple

import httpx

from proxima.config.settings import Settings, config_service

ProviderName = Literal["openai", "anthropic", "ollama", "lmstudio", "none"]


@dataclass
class LLMRequest:
	prompt: str
	model: Optional[str] = None
	temperature: float = 0.0
	max_tokens: Optional[int] = None
	provider: Optional[ProviderName] = None
	metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMResponse:
	text: str
	provider: ProviderName
	model: str
	latency_ms: float
	raw: Optional[dict] = None


class LLMProvider(Protocol):
	name: ProviderName
	is_local: bool
	requires_api_key: bool

	def send(self, request: LLMRequest, api_key: Optional[str]) -> LLMResponse:  # pragma: no cover - interface
		...


class _BaseProvider:
	name: ProviderName
	is_local: bool = False
	requires_api_key: bool = True

	def __init__(self, default_model: str):
		self.default_model = default_model

	def send(self, request: LLMRequest, api_key: Optional[str]) -> LLMResponse:
		start = time.perf_counter()
		text = self._stub_text(request)
		elapsed = (time.perf_counter() - start) * 1000
		return LLMResponse(
			text=text,
			provider=self.name,
			model=request.model or self.default_model,
			latency_ms=elapsed,
			raw={"note": "stub provider response"},
		)

	def _stub_text(self, request: LLMRequest) -> str:
		return (
			f"[stub:{self.name}] model={request.model or self.default_model} "
			f"temp={request.temperature} prompt_preview={request.prompt[:60]}"
		)


class OpenAIProvider(_BaseProvider):
	name: ProviderName = "openai"
	is_local = False
	requires_api_key = True


class AnthropicProvider(_BaseProvider):
	name: ProviderName = "anthropic"
	is_local = False
	requires_api_key = True


class OllamaProvider(_BaseProvider):
	name: ProviderName = "ollama"
	is_local = True
	requires_api_key = False


class LMStudioProvider(_BaseProvider):
	name: ProviderName = "lmstudio"
	is_local = True
	requires_api_key = False


class ProviderRegistry:
	def __init__(self) -> None:
		self.providers: Dict[ProviderName, LLMProvider] = {
			"openai": OpenAIProvider(default_model="gpt-4"),
			"anthropic": AnthropicProvider(default_model="claude-3"),
			"ollama": OllamaProvider(default_model="llama3"),
			"lmstudio": LMStudioProvider(default_model="gpt4all"),
		}

	def get(self, name: ProviderName) -> LLMProvider:
		if name not in self.providers:
			raise ValueError(f"Unknown provider: {name}")
		return self.providers[name]


class LocalLLMDetector:
	"""Detects availability of local LLM runtimes (Ollama, LM Studio, llama.cpp).

	Detection Flow (Step 3.2):
	1) Check configured endpoint first (settings.llm.local_endpoint)
	2) Try default ports for known runtimes
	3) Verify with a quick HTTP call (GET) using a short timeout
	4) Cache the last known good endpoint per provider; allow force re-detect
	"""

	DEFAULT_ENDPOINTS: Dict[str, Tuple[str, ...]] = {
		"ollama": ("http://127.0.0.1:11434/api/health",),
		"lmstudio": (
			"http://127.0.0.1:1234/v1/models",
			"http://127.0.0.1:1234",
		),
		"llama_cpp": ("http://127.0.0.1:8080/health",),
	}

	def __init__(self, timeout_s: float = 0.35):
		self.timeout_s = timeout_s
		self._cache: Dict[str, Optional[str]] = {}

	def _check(self, endpoint: str) -> bool:
		try:
			response = httpx.get(endpoint, timeout=self.timeout_s)
			return response.status_code < 500
		except Exception:
			return False

	def detect(self, provider: ProviderName, configured_endpoint: str = "", force: bool = False) -> Optional[str]:
		if not force and provider in self._cache:
			return self._cache[provider]

		candidates = []
		if configured_endpoint:
			candidates.append(configured_endpoint)

		defaults = self.DEFAULT_ENDPOINTS.get(provider, ())
		candidates.extend(defaults)

		for endpoint in candidates:
			if self._check(endpoint):
				self._cache[provider] = endpoint
				return endpoint

		self._cache[provider] = None
		return None


class APIKeyManager:
	def __init__(self, settings: Settings):
		self.settings = settings

	def get_api_key(self, provider: LLMProvider) -> Optional[str]:
		env_var = self.settings.llm.api_key_env_var
		if not env_var:
			return None
		return os.environ.get(env_var)

	def validate(self, provider: LLMProvider) -> None:
		if not provider.requires_api_key:
			return
		key = self.get_api_key(provider)
		if not key:
			raise ValueError(
				f"API key missing for provider {provider.name}. Set env var {self.settings.llm.api_key_env_var}."
			)


class ConsentGate:
	def __init__(self, settings: Settings, prompt_func: Optional[Callable[[str], bool]] = None):
		self.settings = settings
		self.prompt_func = prompt_func

	def require_consent(self, provider: LLMProvider) -> None:
		if not self.settings.llm.require_consent:
			return

		is_local = provider.is_local
		auto_allow = (
			self.settings.consent.auto_approve_local_llm if is_local else self.settings.consent.auto_approve_remote_llm
		)
		if auto_allow:
			return

		if not self.prompt_func:
			raise PermissionError(
				f"Consent required for {'local' if is_local else 'remote'} provider {provider.name}"
			)

		allowed = self.prompt_func(
			f"Allow {'local' if is_local else 'remote'} LLM provider '{provider.name}'?"
		)
		if not allowed:
			raise PermissionError(f"User denied consent for provider {provider.name}")


class LLMRouter:
	def __init__(self, settings: Optional[Settings] = None, consent_prompt: Optional[Callable[[str], bool]] = None):
		self.settings = settings or config_service.load()
		self.registry = ProviderRegistry()
		self.detector = LocalLLMDetector(timeout_s=0.35)
		self.api_keys = APIKeyManager(self.settings)
		self.consent_gate = ConsentGate(self.settings, prompt_func=consent_prompt)

	def _pick_provider(self, request: LLMRequest) -> LLMProvider:
		provider_name: ProviderName = request.provider or self.settings.llm.provider or "none"  # type: ignore
		if provider_name == "none":
			raise ValueError("No LLM provider configured. Set llm.provider in config or request.")
		return self.registry.get(provider_name)

	def _ensure_local_available(self, provider: LLMProvider) -> None:
		if not provider.is_local:
			return
		endpoint = self.detector.detect(provider.name, configured_endpoint=self.settings.llm.local_endpoint)  # type: ignore[arg-type]
		if not endpoint:
			raise ConnectionError(
				f"Local endpoint not reachable for provider {provider.name}. Tried configured and default ports."
			)

	def route(self, request: LLMRequest) -> LLMResponse:
		provider = self._pick_provider(request)
		self._ensure_local_available(provider)
		self.consent_gate.require_consent(provider)
		self.api_keys.validate(provider)
		api_key = self.api_keys.get_api_key(provider)
		return provider.send(request, api_key)


def build_router(consent_prompt: Optional[Callable[[str], bool]] = None) -> LLMRouter:
	"""Helper to build a router with loaded settings."""

	return LLMRouter(settings=config_service.load(), consent_prompt=consent_prompt)
