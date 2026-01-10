"""Secure secrets handling for configuration.

Provides secure storage and retrieval of sensitive configuration values
like API keys, tokens, and credentials using multiple backends:
- System keyring (most secure)
- Encrypted file storage
- Environment variables (fallback)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class SecretBackend(Enum):
    """Available secret storage backends."""

    KEYRING = "keyring"  # System keyring (recommended)
    ENCRYPTED = "encrypted"  # Encrypted file storage
    ENVIRONMENT = "environment"  # Environment variables
    MEMORY = "memory"  # In-memory only (testing)


@dataclass
class SecretMetadata:
    """Metadata about a stored secret."""

    key: str
    backend: SecretBackend
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if secret has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class SecretResult:
    """Result of a secret operation."""

    success: bool
    value: str | None = None
    error: str | None = None
    metadata: SecretMetadata | None = None

    @classmethod
    def ok(cls, value: str, metadata: SecretMetadata | None = None) -> SecretResult:
        return cls(success=True, value=value, metadata=metadata)

    @classmethod
    def fail(cls, error: str) -> SecretResult:
        return cls(success=False, error=error)


# =============================================================================
# ABSTRACT SECRET BACKEND
# =============================================================================


class SecretStorage(ABC):
    """Abstract base class for secret storage backends."""

    SERVICE_NAME = "proxima"

    @abstractmethod
    def store(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> SecretResult:
        """Store a secret value."""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> SecretResult:
        """Retrieve a secret value."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a secret."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a secret exists."""
        pass

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all stored secret keys."""
        pass

    def rotate(self, key: str, new_value: str) -> SecretResult:
        """Rotate a secret to a new value."""
        if not self.exists(key):
            return SecretResult.fail(f"Secret not found: {key}")

        # Store new value
        result = self.store(key, new_value)
        if not result.success:
            return result

        return result


# =============================================================================
# KEYRING BACKEND (Most Secure)
# =============================================================================


class KeyringStorage(SecretStorage):
    """Store secrets in system keyring (Windows Credential Manager, macOS Keychain, etc.)."""

    def __init__(self) -> None:
        self._keyring: Any = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if keyring is available."""
        try:
            import keyring

            self._keyring = keyring
            # Test keyring access
            keyring.get_password(self.SERVICE_NAME, "__test__")
            return True
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def store(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> SecretResult:
        if not self._available:
            return SecretResult.fail("Keyring not available")

        try:
            self._keyring.set_password(self.SERVICE_NAME, key, value)

            # Store metadata separately
            if metadata:
                meta_key = f"{key}__meta"
                meta_value = json.dumps(metadata)
                self._keyring.set_password(self.SERVICE_NAME, meta_key, meta_value)

            return SecretResult.ok(value)
        except Exception as e:
            return SecretResult.fail(f"Failed to store secret: {e}")

    def retrieve(self, key: str) -> SecretResult:
        if not self._available:
            return SecretResult.fail("Keyring not available")

        try:
            value = self._keyring.get_password(self.SERVICE_NAME, key)
            if value is None:
                return SecretResult.fail(f"Secret not found: {key}")

            # Try to get metadata
            metadata = None
            try:
                meta_value = self._keyring.get_password(self.SERVICE_NAME, f"{key}__meta")
                if meta_value:
                    meta_dict = json.loads(meta_value)
                    metadata = SecretMetadata(
                        key=key,
                        backend=SecretBackend.KEYRING,
                        created_at=datetime.fromisoformat(
                            meta_dict.get("created_at", datetime.now().isoformat())
                        ),
                        updated_at=datetime.fromisoformat(
                            meta_dict.get("updated_at", datetime.now().isoformat())
                        ),
                    )
            except Exception:
                pass

            return SecretResult.ok(value, metadata)
        except Exception as e:
            return SecretResult.fail(f"Failed to retrieve secret: {e}")

    def delete(self, key: str) -> bool:
        if not self._available:
            return False

        try:
            self._keyring.delete_password(self.SERVICE_NAME, key)
            # Also delete metadata
            try:
                self._keyring.delete_password(self.SERVICE_NAME, f"{key}__meta")
            except Exception:
                pass
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        if not self._available:
            return False

        try:
            return self._keyring.get_password(self.SERVICE_NAME, key) is not None
        except Exception:
            return False

    def list_keys(self) -> list[str]:
        # Keyring doesn't support listing keys natively
        # Return empty list as we can't enumerate
        return []


# =============================================================================
# ENCRYPTED FILE BACKEND
# =============================================================================


class EncryptedFileStorage(SecretStorage):
    """Store secrets in an encrypted file."""

    def __init__(
        self, storage_path: Path | None = None, encryption_key: bytes | None = None
    ) -> None:
        self.storage_path = storage_path or (Path.home() / ".proxima" / "secrets.enc")
        self._encryption_key = encryption_key
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key."""
        if self._encryption_key:
            return self._encryption_key

        # Try to get from environment
        env_key = os.environ.get("PROXIMA_SECRET_KEY")
        if env_key:
            return hashlib.sha256(env_key.encode()).digest()

        # Generate machine-specific key
        machine_id = self._get_machine_id()
        return hashlib.sha256(machine_id.encode()).digest()

    def _get_machine_id(self) -> str:
        """Get a machine-specific identifier for key derivation."""
        import platform
        import uuid

        components = [
            platform.node(),
            platform.machine(),
            str(uuid.getnode()),  # MAC address based
        ]
        return ":".join(components)

    def _encrypt(self, data: str) -> bytes:
        """Simple XOR-based encryption (use cryptography lib in production)."""
        key = self._get_encryption_key()
        data_bytes = data.encode("utf-8")
        encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data_bytes))
        return base64.b64encode(encrypted)

    def _decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt data."""
        key = self._get_encryption_key()
        data_bytes = base64.b64decode(encrypted_data)
        decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data_bytes))
        return decrypted.decode("utf-8")

    def _load_cache(self) -> None:
        """Load encrypted secrets from file."""
        if not self.storage_path.exists():
            self._cache = {}
            return

        try:
            encrypted_content = self.storage_path.read_bytes()
            decrypted = self._decrypt(encrypted_content)
            self._cache = json.loads(decrypted)
        except Exception:
            self._cache = {}

    def _save_cache(self) -> None:
        """Save secrets to encrypted file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self._cache)
        encrypted = self._encrypt(content)
        self.storage_path.write_bytes(encrypted)

    def store(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> SecretResult:
        try:
            now = datetime.now().isoformat()
            self._cache[key] = {
                "value": value,
                "created_at": self._cache.get(key, {}).get("created_at", now),
                "updated_at": now,
                "metadata": metadata or {},
            }
            self._save_cache()
            return SecretResult.ok(value)
        except Exception as e:
            return SecretResult.fail(f"Failed to store secret: {e}")

    def retrieve(self, key: str) -> SecretResult:
        try:
            if key not in self._cache:
                return SecretResult.fail(f"Secret not found: {key}")

            entry = self._cache[key]
            metadata = SecretMetadata(
                key=key,
                backend=SecretBackend.ENCRYPTED,
                created_at=datetime.fromisoformat(
                    entry.get("created_at", datetime.now().isoformat())
                ),
                updated_at=datetime.fromisoformat(
                    entry.get("updated_at", datetime.now().isoformat())
                ),
            )

            return SecretResult.ok(entry["value"], metadata)
        except Exception as e:
            return SecretResult.fail(f"Failed to retrieve secret: {e}")

    def delete(self, key: str) -> bool:
        if key not in self._cache:
            return False

        try:
            del self._cache[key]
            self._save_cache()
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        return key in self._cache

    def list_keys(self) -> list[str]:
        return list(self._cache.keys())


# =============================================================================
# ENVIRONMENT VARIABLE BACKEND
# =============================================================================


class EnvironmentStorage(SecretStorage):
    """Store/retrieve secrets from environment variables."""

    PREFIX = "PROXIMA_SECRET_"

    def store(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> SecretResult:
        # Can't actually store - just set in current process
        env_key = f"{self.PREFIX}{key.upper()}"
        os.environ[env_key] = value
        return SecretResult.ok(value)

    def retrieve(self, key: str) -> SecretResult:
        # Try with prefix
        env_key = f"{self.PREFIX}{key.upper()}"
        value = os.environ.get(env_key)

        # Try without prefix (for backward compatibility)
        if value is None:
            value = os.environ.get(key.upper())

        if value is None:
            return SecretResult.fail(f"Environment variable not found: {env_key}")

        metadata = SecretMetadata(
            key=key,
            backend=SecretBackend.ENVIRONMENT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        return SecretResult.ok(value, metadata)

    def delete(self, key: str) -> bool:
        env_key = f"{self.PREFIX}{key.upper()}"
        if env_key in os.environ:
            del os.environ[env_key]
            return True
        return False

    def exists(self, key: str) -> bool:
        env_key = f"{self.PREFIX}{key.upper()}"
        return env_key in os.environ or key.upper() in os.environ

    def list_keys(self) -> list[str]:
        keys = []
        for key in os.environ:
            if key.startswith(self.PREFIX):
                keys.append(key[len(self.PREFIX) :].lower())
        return keys


# =============================================================================
# MEMORY BACKEND (Testing)
# =============================================================================


class MemoryStorage(SecretStorage):
    """In-memory secret storage for testing."""

    def __init__(self) -> None:
        self._secrets: dict[str, dict[str, Any]] = {}

    def store(self, key: str, value: str, metadata: dict[str, Any] | None = None) -> SecretResult:
        now = datetime.now()
        self._secrets[key] = {
            "value": value,
            "created_at": self._secrets.get(key, {}).get("created_at", now),
            "updated_at": now,
            "metadata": metadata or {},
        }
        return SecretResult.ok(value)

    def retrieve(self, key: str) -> SecretResult:
        if key not in self._secrets:
            return SecretResult.fail(f"Secret not found: {key}")

        entry = self._secrets[key]
        metadata = SecretMetadata(
            key=key,
            backend=SecretBackend.MEMORY,
            created_at=entry["created_at"],
            updated_at=entry["updated_at"],
        )
        return SecretResult.ok(entry["value"], metadata)

    def delete(self, key: str) -> bool:
        if key in self._secrets:
            del self._secrets[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        return key in self._secrets

    def list_keys(self) -> list[str]:
        return list(self._secrets.keys())

    def clear(self) -> None:
        """Clear all secrets (for testing)."""
        self._secrets.clear()


# =============================================================================
# SECRET MANAGER
# =============================================================================


class SecretManager:
    """High-level manager for secrets with fallback backends."""

    # Standard secret keys used by Proxima
    STANDARD_KEYS = {
        "openai_api_key": "OpenAI API key for GPT models",
        "anthropic_api_key": "Anthropic API key for Claude models",
        "ibm_quantum_token": "IBM Quantum API token",
        "aws_access_key": "AWS access key for Braket",
        "azure_quantum_key": "Azure Quantum API key",
    }

    def __init__(self, preferred_backend: SecretBackend = SecretBackend.KEYRING) -> None:
        self._backends: dict[SecretBackend, SecretStorage] = {}
        self._preferred_backend = preferred_backend
        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize available backends."""
        # Always have memory backend
        self._backends[SecretBackend.MEMORY] = MemoryStorage()

        # Try keyring
        keyring_storage = KeyringStorage()
        if keyring_storage.is_available:
            self._backends[SecretBackend.KEYRING] = keyring_storage

        # Always have encrypted file storage
        self._backends[SecretBackend.ENCRYPTED] = EncryptedFileStorage()

        # Always have environment
        self._backends[SecretBackend.ENVIRONMENT] = EnvironmentStorage()

    def get_backend(self, backend: SecretBackend | None = None) -> SecretStorage:
        """Get a specific backend or the preferred one."""
        backend = backend or self._preferred_backend

        if backend in self._backends:
            return self._backends[backend]

        # Fallback chain
        fallback_order = [
            SecretBackend.KEYRING,
            SecretBackend.ENCRYPTED,
            SecretBackend.ENVIRONMENT,
            SecretBackend.MEMORY,
        ]

        for fb in fallback_order:
            if fb in self._backends:
                return self._backends[fb]

        return self._backends[SecretBackend.MEMORY]

    def store(
        self,
        key: str,
        value: str,
        backend: SecretBackend | None = None,
        expires_in: timedelta | None = None,
    ) -> SecretResult:
        """Store a secret."""
        storage = self.get_backend(backend)

        metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        if expires_in:
            metadata["expires_at"] = (datetime.now() + expires_in).isoformat()

        return storage.store(key, value, metadata)

    def retrieve(self, key: str, backend: SecretBackend | None = None) -> SecretResult:
        """Retrieve a secret, trying fallback backends if needed."""
        # Try specific backend first
        if backend:
            storage = self.get_backend(backend)
            result = storage.retrieve(key)
            if result.success:
                return result

        # Try all backends in order
        for backend_type in [
            SecretBackend.KEYRING,
            SecretBackend.ENCRYPTED,
            SecretBackend.ENVIRONMENT,
        ]:
            if backend_type in self._backends:
                result = self._backends[backend_type].retrieve(key)
                if result.success:
                    return result

        return SecretResult.fail(f"Secret not found in any backend: {key}")

    def delete(self, key: str, backend: SecretBackend | None = None) -> bool:
        """Delete a secret from one or all backends."""
        if backend:
            return self.get_backend(backend).delete(key)

        # Delete from all backends
        deleted = False
        for storage in self._backends.values():
            if storage.delete(key):
                deleted = True
        return deleted

    def exists(self, key: str) -> bool:
        """Check if a secret exists in any backend."""
        for storage in self._backends.values():
            if storage.exists(key):
                return True
        return False

    def list_all(self) -> dict[SecretBackend, list[str]]:
        """List all secrets from all backends."""
        result = {}
        for backend, storage in self._backends.items():
            keys = storage.list_keys()
            if keys:
                result[backend] = keys
        return result

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider."""
        key_mapping = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "ibm": "ibm_quantum_token",
            "aws": "aws_access_key",
            "azure": "azure_quantum_key",
        }

        secret_key = key_mapping.get(provider.lower(), f"{provider.lower()}_api_key")
        result = self.retrieve(secret_key)

        return result.value if result.success else None

    def set_api_key(self, provider: str, value: str) -> SecretResult:
        """Set API key for a specific provider."""
        key_mapping = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "ibm": "ibm_quantum_token",
            "aws": "aws_access_key",
            "azure": "azure_quantum_key",
        }

        secret_key = key_mapping.get(provider.lower(), f"{provider.lower()}_api_key")
        return self.store(secret_key, value)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def generate_secret_key(length: int = 32) -> str:
    """Generate a cryptographically secure random key."""
    return secrets.token_hex(length)


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """Mask a secret value for display."""
    if len(value) <= visible_chars * 2:
        return "*" * len(value)

    return value[:visible_chars] + "*" * (len(value) - visible_chars * 2) + value[-visible_chars:]


def validate_api_key_format(key: str, provider: str) -> tuple[bool, str]:
    """Validate API key format for common providers."""
    validations = {
        "openai": (lambda k: k.startswith("sk-") and len(k) > 20, "OpenAI keys start with 'sk-'"),
        "anthropic": (
            lambda k: k.startswith("sk-ant-") and len(k) > 30,
            "Anthropic keys start with 'sk-ant-'",
        ),
    }

    if provider.lower() not in validations:
        return True, "No format validation available for this provider"

    validator, message = validations[provider.lower()]
    if validator(key):
        return True, "Key format is valid"
    return False, message


# Singleton instance
_secret_manager: SecretManager | None = None


def get_secret_manager() -> SecretManager:
    """Get the global secret manager instance."""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager
