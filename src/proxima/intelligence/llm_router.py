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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx

from proxima.config.settings import Settings, config_service

ProviderName = Literal["openai", "anthropic", "ollama", "lmstudio", "llama_cpp", "none"]


@dataclass
class LLMRequest:
    prompt: str
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    provider: ProviderName | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    provider: ProviderName
    model: str
    latency_ms: float
    raw: dict | None = None


class LLMProvider(Protocol):
    name: ProviderName
    is_local: bool
    requires_api_key: bool

    def send(
        self, request: LLMRequest, api_key: str | None
    ) -> LLMResponse:  # pragma: no cover - interface
        ...


class _BaseProvider:
    name: ProviderName
    is_local: bool = False
    requires_api_key: bool = True

    def __init__(self, default_model: str):
        self.default_model = default_model

    def send(self, request: LLMRequest, api_key: str | None) -> LLMResponse:
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
        self.providers: dict[ProviderName, LLMProvider] = {
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

    DEFAULT_ENDPOINTS: dict[str, tuple[str, ...]] = {
        "ollama": ("http://127.0.0.1:11434/api/health",),
        "lmstudio": (
            "http://127.0.0.1:1234/v1/models",
            "http://127.0.0.1:1234",
        ),
        "llama_cpp": ("http://127.0.0.1:8080/health",),
    }

    # Common locations for local model files
    MODEL_FILE_PATTERNS: tuple[str, ...] = (
        "~/.ollama/models",
        "~/.cache/lm-studio/models",
        "~/.local/share/llama.cpp/models",
        "~/Library/Application Support/LM Studio/models",
    )

    def __init__(self, timeout_s: float = 0.35):
        self.timeout_s = timeout_s
        self._cache: dict[str, str | None] = {}
        self._model_files_cache: dict[str, list] | None = None

    def _check(self, endpoint: str) -> bool:
        try:
            response = httpx.get(endpoint, timeout=self.timeout_s)
            return response.status_code < 500
        except Exception:
            return False

    def check_ollama(self, configured_endpoint: str = "", force: bool = False) -> str | None:
        """Dedicated method to check Ollama availability.

        Returns the working endpoint URL or None if unavailable.
        """
        return self.detect("ollama", configured_endpoint, force)

    def check_lm_studio(self, configured_endpoint: str = "", force: bool = False) -> str | None:
        """Dedicated method to check LM Studio availability.

        Returns the working endpoint URL or None if unavailable.
        """
        return self.detect("lmstudio", configured_endpoint, force)

    def check_llama_cpp(self, configured_endpoint: str = "", force: bool = False) -> str | None:
        """Dedicated method to check llama.cpp server availability.

        Returns the working endpoint URL or None if unavailable.
        """
        return self.detect("llama_cpp", configured_endpoint, force)

    def scan_model_files(self, force: bool = False) -> dict[str, list]:
        """Scan common directories for local model files.

        Returns a dict mapping provider names to lists of discovered model paths.
        Results are cached unless force=True.
        """
        from pathlib import Path

        if not force and self._model_files_cache is not None:
            return self._model_files_cache

        results: dict[str, list] = {
            "ollama": [],
            "lmstudio": [],
            "llama_cpp": [],
            "unknown": [],
        }

        model_extensions = ("*.gguf", "*.ggml", "*.bin", "*.safetensors")

        for pattern_base in self.MODEL_FILE_PATTERNS:
            base_path = Path(pattern_base).expanduser()
            if not base_path.exists():
                continue

            # Determine provider from path
            provider = "unknown"
            path_str = str(base_path).lower()
            if "ollama" in path_str:
                provider = "ollama"
            elif "lm-studio" in path_str or "lm studio" in path_str:
                provider = "lmstudio"
            elif "llama" in path_str:
                provider = "llama_cpp"

            # Scan for model files
            for ext in model_extensions:
                for model_file in base_path.rglob(ext):
                    results[provider].append(str(model_file))

        self._model_files_cache = results
        return results

    def detect(
        self, provider: ProviderName, configured_endpoint: str = "", force: bool = False
    ) -> str | None:
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

    def detect_all_parallel(self, force: bool = False) -> dict[str, str | None]:
        """Detect all local LLM providers in parallel for faster startup.

        Uses concurrent.futures for parallel HTTP checks, reducing total
        detection time from ~1s (sequential) to ~0.35s (parallel).

        Returns:
                Dict mapping provider names to their detected endpoints (or None).
        """
        import concurrent.futures

        providers_to_check = ["ollama", "lmstudio", "llama_cpp"]
        results: dict[str, str | None] = {}

        # Skip if all are cached and not forcing refresh
        if not force and all(p in self._cache for p in providers_to_check):
            return {p: self._cache[p] for p in providers_to_check}

        def check_provider(provider: str) -> tuple[str, str | None]:
            """Check a single provider and return (provider, endpoint)."""
            if not force and provider in self._cache:
                return (provider, self._cache[provider])

            for endpoint in self.DEFAULT_ENDPOINTS.get(provider, ()):
                if self._check(endpoint):
                    return (provider, endpoint)
            return (provider, None)

        # Run checks in parallel with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(check_provider, p): p for p in providers_to_check}

            for future in concurrent.futures.as_completed(futures, timeout=1.0):
                try:
                    provider, endpoint = future.result()
                    self._cache[provider] = endpoint
                    results[provider] = endpoint
                except Exception:
                    provider = futures[future]
                    self._cache[provider] = None
                    results[provider] = None

        return results


class APIKeyManager:
    """Manages API keys for LLM providers.

    Supports reading from environment variables and optional secure storage.
    """

    def __init__(self, settings: Settings, storage_path: str | None = None):
        self.settings = settings
        self._storage_path = storage_path
        self._key_cache: dict[str, str] = {}

    def get_api_key(self, provider: LLMProvider) -> str | None:
        """Get API key for a provider (alias for get_key)."""
        return self.get_key(provider.name)

    def get_key(self, provider_name: str) -> str | None:
        """Retrieve API key for provider from cache, env, or storage."""
        # Check cache first
        if provider_name in self._key_cache:
            return self._key_cache[provider_name]

        # Check provider-specific env var
        provider_env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": None,  # No key needed
            "lmstudio": None,  # No key needed
        }

        env_var = provider_env_vars.get(provider_name) or self.settings.llm.api_key_env_var
        if env_var:
            key = os.environ.get(env_var)
            if key:
                self._key_cache[provider_name] = key
                return key

        # Try to load from secure storage if available
        if self._storage_path:
            key = self._load_from_storage(provider_name)
            if key:
                self._key_cache[provider_name] = key
                return key

        return None

    def store_key(self, provider_name: str, api_key: str) -> bool:
        """Store API key in secure storage.

        Returns True if stored successfully, False if storage not configured.
        """
        import json
        from pathlib import Path

        if not self._storage_path:
            # Default to user config directory
            self._storage_path = str(Path.home() / ".proxima" / ".keys.json")

        storage_path = Path(self._storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing keys
        keys: dict[str, str] = {}
        if storage_path.exists():
            try:
                keys = json.loads(storage_path.read_text())
            except (json.JSONDecodeError, OSError):
                keys = {}

        # Store the new key
        keys[provider_name] = api_key

        # Write back (with restricted permissions on Unix)
        storage_path.write_text(json.dumps(keys, indent=2))
        try:
            storage_path.chmod(0o600)  # Owner read/write only
        except OSError:
            pass  # Windows may not support chmod

        # Update cache
        self._key_cache[provider_name] = api_key
        return True

    def _load_from_storage(self, provider_name: str) -> str | None:
        """Load API key from secure storage file."""
        import json
        from pathlib import Path

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
        import json

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
        import json

        consents = self._load_persistent_consents()
        consents[provider_name] = allowed

        storage_path = self._get_storage_path()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_text(json.dumps(consents, indent=2))

    def _remove_persistent_consent(self, provider_name: str) -> None:
        """Remove a persistent consent decision."""
        import json

        consents = self._load_persistent_consents()
        if provider_name in consents:
            del consents[provider_name]
            storage_path = self._get_storage_path()
            storage_path.write_text(json.dumps(consents, indent=2))

    def _get_storage_path(self):
        """Get the path for persistent consent storage."""
        from pathlib import Path

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
    def __init__(
        self, settings: Settings | None = None, consent_prompt: Callable[[str], bool] | None = None
    ):
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


def build_router(consent_prompt: Callable[[str], bool] | None = None) -> LLMRouter:
    """Helper to build a router with loaded settings."""

    return LLMRouter(settings=config_service.load(), consent_prompt=consent_prompt)
