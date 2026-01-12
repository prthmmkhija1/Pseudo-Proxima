"""
Integration Points Module - External System Integration
========================================================

This module provides unified interfaces for integrating with external systems:

┌─────────────────────────────────────────────────────────────┐
│                        PROXIMA                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │   Quantum   │      │    LLM      │      │   System    │ │
│  │  Libraries  │      │ Providers   │      │  Resources  │ │
│  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘ │
│         │                    │                    │         │
└─────────┼────────────────────┼────────────────────┼─────────┘
          │                    │                    │
          ▼                    ▼                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │    Cirq     │     │   OpenAI    │     │   psutil    │
    │   Qiskit    │     │  Anthropic  │     │  (Memory)   │
    │    LRET     │     │   Ollama    │     │             │
    └─────────────┘     └─────────────┘     └─────────────┘

Integration Responsibilities:
1. Quantum Libraries - Backend adapters for circuit execution
2. LLM Providers - AI-powered insights and auto-selection
3. System Resources - Memory, CPU monitoring and thresholds
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# INTEGRATION STATUS AND TYPES
# =============================================================================


class IntegrationStatus(Enum):
    """Status of an external integration."""

    AVAILABLE = auto()
    UNAVAILABLE = auto()
    ERROR = auto()
    NOT_CONFIGURED = auto()
    REQUIRES_AUTH = auto()


class IntegrationType(Enum):
    """Types of external integrations."""

    QUANTUM_BACKEND = "quantum"
    LLM_PROVIDER = "llm"
    SYSTEM_RESOURCE = "system"
    STORAGE = "storage"
    NOTIFICATION = "notification"


@dataclass
class IntegrationHealth:
    """Health status of an integration."""

    name: str
    integration_type: IntegrationType
    status: IntegrationStatus
    message: str = ""
    latency_ms: float | None = None
    version: str | None = None
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if the integration is healthy."""
        return self.status == IntegrationStatus.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.integration_type.value,
            "status": self.status.name,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "version": self.version,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
        }


# =============================================================================
# BASE INTEGRATION INTERFACE
# =============================================================================


class ExternalIntegration(ABC):
    """Base class for all external integrations."""

    def __init__(self, name: str, integration_type: IntegrationType):
        self.name = name
        self.integration_type = integration_type
        self._connected = False
        self._health: IntegrationHealth | None = None

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the external system."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the external system."""
        pass

    @abstractmethod
    async def health_check(self) -> IntegrationHealth:
        """Check the health of the integration."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    async def ensure_connected(self) -> bool:
        """Ensure connection is established."""
        if not self._connected:
            return await self.connect()
        return True


# =============================================================================
# QUANTUM LIBRARY INTEGRATION
# =============================================================================


@dataclass
class QuantumLibraryInfo:
    """Information about a quantum library."""

    name: str
    version: str
    simulator_types: list[str]
    max_qubits: int
    supports_noise: bool
    supports_gpu: bool
    is_installed: bool = False


class QuantumIntegration(ExternalIntegration):
    """Integration with quantum computing libraries."""

    SUPPORTED_LIBRARIES = ["cirq", "qiskit", "qiskit-aer", "lret"]

    def __init__(self):
        super().__init__("quantum_libraries", IntegrationType.QUANTUM_BACKEND)
        self._available_libraries: dict[str, QuantumLibraryInfo] = {}

    async def connect(self) -> bool:
        """Discover and connect to available quantum libraries."""
        self._available_libraries = await self._discover_libraries()
        self._connected = len(self._available_libraries) > 0
        logger.info(
            "quantum_integration_connected",
            libraries=list(self._available_libraries.keys()),
            count=len(self._available_libraries),
        )
        return self._connected

    async def disconnect(self) -> bool:
        """Disconnect from quantum libraries."""
        self._available_libraries = {}
        self._connected = False
        return True

    async def health_check(self) -> IntegrationHealth:
        """Check quantum library availability."""
        available = list(self._available_libraries.keys())

        if not available:
            return IntegrationHealth(
                name=self.name,
                integration_type=self.integration_type,
                status=IntegrationStatus.UNAVAILABLE,
                message="No quantum libraries installed",
            )

        return IntegrationHealth(
            name=self.name,
            integration_type=self.integration_type,
            status=IntegrationStatus.AVAILABLE,
            message=f"{len(available)} libraries available",
            capabilities=available,
            metadata={
                "libraries": {
                    k: v.__dict__ for k, v in self._available_libraries.items()
                }
            },
        )

    async def _discover_libraries(self) -> dict[str, QuantumLibraryInfo]:
        """Discover installed quantum libraries."""
        libraries = {}

        # Check Cirq
        try:
            import cirq

            libraries["cirq"] = QuantumLibraryInfo(
                name="cirq",
                version=getattr(cirq, "__version__", "unknown"),
                simulator_types=["state_vector", "density_matrix"],
                max_qubits=32,
                supports_noise=True,
                supports_gpu=False,
                is_installed=True,
            )
        except ImportError:
            pass

        # Check Qiskit Aer
        try:
            import qiskit_aer

            libraries["qiskit-aer"] = QuantumLibraryInfo(
                name="qiskit-aer",
                version=getattr(qiskit_aer, "__version__", "unknown"),
                simulator_types=["state_vector", "density_matrix", "qasm"],
                max_qubits=30,
                supports_noise=True,
                supports_gpu=True,
                is_installed=True,
            )
        except ImportError:
            pass

        # Check LRET (custom quantum framework)
        try:
            import lret  # noqa: F401

            libraries["lret"] = QuantumLibraryInfo(
                name="lret",
                version=getattr(lret, "__version__", "0.1.0"),
                simulator_types=["framework", "hybrid"],
                max_qubits=getattr(lret, "MAX_QUBITS", 20),
                supports_noise=getattr(lret, "SUPPORTS_NOISE", False),
                supports_gpu=getattr(lret, "SUPPORTS_GPU", False),
                is_installed=True,
            )
            logger.debug("lret_detected", version=libraries["lret"].version)
        except ImportError:
            # LRET not installed - this is expected in most environments
            logger.debug("lret_not_available", reason="package not installed")
        except Exception as e:
            # LRET installed but failed to load
            logger.warning("lret_load_failed", error=str(e))

        return libraries

    def get_library_info(self, name: str) -> QuantumLibraryInfo | None:
        """Get information about a specific library."""
        return self._available_libraries.get(name)

    @property
    def available_backends(self) -> list[str]:
        """Get list of available backend names."""
        return list(self._available_libraries.keys())


# =============================================================================
# LLM PROVIDER INTEGRATION
# =============================================================================


@dataclass
class LLMProviderInfo:
    """Information about an LLM provider."""

    name: str
    provider_type: str  # "remote" or "local"
    endpoint: str
    models: list[str]
    requires_api_key: bool
    is_configured: bool = False
    is_available: bool = False


class LLMIntegration(ExternalIntegration):
    """Integration with LLM providers."""

    DEFAULT_ENDPOINTS = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "ollama": "http://localhost:11434",
        "lmstudio": "http://localhost:1234/v1",
    }

    def __init__(self):
        super().__init__("llm_providers", IntegrationType.LLM_PROVIDER)
        self._providers: dict[str, LLMProviderInfo] = {}

    async def connect(self) -> bool:
        """Discover and check LLM providers."""
        self._providers = await self._discover_providers()
        self._connected = any(p.is_available for p in self._providers.values())
        logger.info(
            "llm_integration_connected",
            providers=list(self._providers.keys()),
            available=[k for k, v in self._providers.items() if v.is_available],
        )
        return self._connected

    async def disconnect(self) -> bool:
        """Disconnect from LLM providers."""
        self._providers = {}
        self._connected = False
        return True

    async def health_check(self) -> IntegrationHealth:
        """Check LLM provider availability."""
        available = [k for k, v in self._providers.items() if v.is_available]

        if not available:
            return IntegrationHealth(
                name=self.name,
                integration_type=self.integration_type,
                status=IntegrationStatus.NOT_CONFIGURED,
                message="No LLM providers available",
            )

        return IntegrationHealth(
            name=self.name,
            integration_type=self.integration_type,
            status=IntegrationStatus.AVAILABLE,
            message=f"{len(available)} providers available",
            capabilities=available,
        )

    async def _discover_providers(self) -> dict[str, LLMProviderInfo]:
        """Discover available LLM providers."""
        providers = {}

        # Check OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        providers["openai"] = LLMProviderInfo(
            name="openai",
            provider_type="remote",
            endpoint=self.DEFAULT_ENDPOINTS["openai"],
            models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            requires_api_key=True,
            is_configured=bool(openai_key),
            is_available=bool(openai_key),
        )

        # Check Anthropic
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        providers["anthropic"] = LLMProviderInfo(
            name="anthropic",
            provider_type="remote",
            endpoint=self.DEFAULT_ENDPOINTS["anthropic"],
            models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            requires_api_key=True,
            is_configured=bool(anthropic_key),
            is_available=bool(anthropic_key),
        )

        # Check Ollama (local)
        ollama_available = await self._check_local_llm(self.DEFAULT_ENDPOINTS["ollama"])
        providers["ollama"] = LLMProviderInfo(
            name="ollama",
            provider_type="local",
            endpoint=self.DEFAULT_ENDPOINTS["ollama"],
            models=["llama3", "mistral", "codellama"],
            requires_api_key=False,
            is_configured=True,
            is_available=ollama_available,
        )

        # Check LM Studio (local)
        lmstudio_available = await self._check_local_llm(
            self.DEFAULT_ENDPOINTS["lmstudio"]
        )
        providers["lmstudio"] = LLMProviderInfo(
            name="lmstudio",
            provider_type="local",
            endpoint=self.DEFAULT_ENDPOINTS["lmstudio"],
            models=["local-model"],
            requires_api_key=False,
            is_configured=True,
            is_available=lmstudio_available,
        )

        return providers

    async def _check_local_llm(self, endpoint: str) -> bool:
        """Check if a local LLM endpoint is available."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                # Try Ollama-style health check
                if "11434" in endpoint:
                    response = await client.get(f"{endpoint}/api/tags")
                else:
                    response = await client.get(f"{endpoint}/models")
                return response.status_code == 200
        except Exception:
            return False

    def get_provider_info(self, name: str) -> LLMProviderInfo | None:
        """Get information about a specific provider."""
        return self._providers.get(name)

    @property
    def available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return [k for k, v in self._providers.items() if v.is_available]

    @property
    def local_providers(self) -> list[str]:
        """Get list of available local providers."""
        return [
            k
            for k, v in self._providers.items()
            if v.is_available and v.provider_type == "local"
        ]


# =============================================================================
# SYSTEM RESOURCES INTEGRATION
# =============================================================================


@dataclass
class SystemResourceInfo:
    """Information about system resources."""

    total_memory_mb: int
    available_memory_mb: int
    memory_percent: float
    cpu_count: int
    cpu_percent: float
    disk_free_gb: float

    @property
    def is_memory_low(self) -> bool:
        """Check if memory is low (>80% used)."""
        return self.memory_percent > 80.0

    @property
    def is_memory_critical(self) -> bool:
        """Check if memory is critical (>95% used)."""
        return self.memory_percent > 95.0


class SystemResourceIntegration(ExternalIntegration):
    """Integration with system resource monitoring."""

    def __init__(
        self,
        memory_warn_threshold_mb: int = 4096,
        memory_critical_threshold_mb: int = 8192,
    ):
        super().__init__("system_resources", IntegrationType.SYSTEM_RESOURCE)
        self.memory_warn_threshold_mb = memory_warn_threshold_mb
        self.memory_critical_threshold_mb = memory_critical_threshold_mb
        self._psutil_available = False

    async def connect(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil

            self._psutil_available = True
            self._connected = True
        except ImportError:
            self._psutil_available = False
            self._connected = False
        return self._connected

    async def disconnect(self) -> bool:
        """Disconnect (no-op for system resources)."""
        self._connected = False
        return True

    async def health_check(self) -> IntegrationHealth:
        """Check system resource status."""
        if not self._psutil_available:
            return IntegrationHealth(
                name=self.name,
                integration_type=self.integration_type,
                status=IntegrationStatus.UNAVAILABLE,
                message="psutil not installed",
            )

        info = await self.get_resource_info()
        status = IntegrationStatus.AVAILABLE
        message = "Resources OK"

        if info.is_memory_critical:
            status = IntegrationStatus.ERROR
            message = f"Critical memory usage: {info.memory_percent:.1f}%"
        elif info.is_memory_low:
            message = f"High memory usage: {info.memory_percent:.1f}%"

        return IntegrationHealth(
            name=self.name,
            integration_type=self.integration_type,
            status=status,
            message=message,
            metadata={
                "memory_percent": info.memory_percent,
                "available_mb": info.available_memory_mb,
                "cpu_percent": info.cpu_percent,
            },
        )

    async def get_resource_info(self) -> SystemResourceInfo:
        """Get current system resource information."""
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return SystemResourceInfo(
            total_memory_mb=memory.total // (1024 * 1024),
            available_memory_mb=memory.available // (1024 * 1024),
            memory_percent=memory.percent,
            cpu_count=psutil.cpu_count() or 1,
            cpu_percent=psutil.cpu_percent(interval=0.1),
            disk_free_gb=disk.free / (1024**3),
        )

    async def estimate_can_execute(self, required_memory_mb: int) -> tuple[bool, str]:
        """Estimate if execution is possible given memory requirements."""
        info = await self.get_resource_info()

        if info.available_memory_mb < required_memory_mb:
            return False, (
                f"Insufficient memory: {info.available_memory_mb}MB available, "
                f"{required_memory_mb}MB required"
            )

        if info.available_memory_mb < self.memory_warn_threshold_mb:
            return True, (
                f"Warning: Low memory ({info.available_memory_mb}MB available). "
                "Execution may be slow."
            )

        return True, "Resources sufficient for execution"


# =============================================================================
# STORAGE INTEGRATION
# =============================================================================


@dataclass
class StorageInfo:
    """Information about a storage backend."""

    name: str
    storage_type: str  # "local", "s3", "gcs", "azure"
    path_or_endpoint: str
    is_available: bool = False
    is_writable: bool = False
    free_space_mb: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "storage_type": self.storage_type,
            "path_or_endpoint": self.path_or_endpoint,
            "is_available": self.is_available,
            "is_writable": self.is_writable,
            "free_space_mb": self.free_space_mb,
        }


class StorageIntegration(ExternalIntegration):
    """Integration with storage backends for result persistence."""

    SUPPORTED_TYPES = ["local", "s3", "gcs", "azure"]

    def __init__(self):
        super().__init__("storage", IntegrationType.STORAGE)
        self._backends: dict[str, StorageInfo] = {}
        self._default_backend: str = "local"

    async def connect(self) -> bool:
        """Discover and connect to storage backends."""
        self._backends = await self._discover_storage()
        self._connected = len(self._backends) > 0
        logger.info(
            "storage_integration_connected",
            backends=list(self._backends.keys()),
        )
        return self._connected

    async def disconnect(self) -> bool:
        """Disconnect from storage backends."""
        self._backends = {}
        self._connected = False
        return True

    async def health_check(self) -> IntegrationHealth:
        """Check storage backend availability."""
        available = [k for k, v in self._backends.items() if v.is_available]

        if not available:
            return IntegrationHealth(
                name=self.name,
                integration_type=self.integration_type,
                status=IntegrationStatus.UNAVAILABLE,
                message="No storage backends available",
            )

        return IntegrationHealth(
            name=self.name,
            integration_type=self.integration_type,
            status=IntegrationStatus.AVAILABLE,
            message=f"{len(available)} storage backends available",
            capabilities=available,
            metadata={"backends": {k: v.to_dict() for k, v in self._backends.items()}},
        )

    async def _discover_storage(self) -> dict[str, StorageInfo]:
        """Discover available storage backends."""
        backends = {}

        # Local filesystem storage (always available)
        import os

        local_path = os.path.expanduser("~/.proxima/results")
        os.makedirs(local_path, exist_ok=True)

        try:
            # Check if writable
            test_file = os.path.join(local_path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            is_writable = True
        except Exception:
            is_writable = False

        # Get free space
        try:
            if hasattr(os, "statvfs"):
                stat = os.statvfs(local_path)
                free_space_mb = (stat.f_frsize * stat.f_bavail) // (1024 * 1024)
            else:
                import shutil

                total, used, free = shutil.disk_usage(local_path)
                free_space_mb = free // (1024 * 1024)
        except Exception:
            free_space_mb = 0

        backends["local"] = StorageInfo(
            name="local",
            storage_type="local",
            path_or_endpoint=local_path,
            is_available=True,
            is_writable=is_writable,
            free_space_mb=free_space_mb,
        )
        logger.debug("local_storage_detected", path=local_path, writable=is_writable)

        # Check for S3 configuration
        s3_bucket = os.environ.get("PROXIMA_S3_BUCKET")
        if s3_bucket:
            try:
                import boto3  # noqa: F401

                backends["s3"] = StorageInfo(
                    name="s3",
                    storage_type="s3",
                    path_or_endpoint=f"s3://{s3_bucket}",
                    is_available=True,
                    is_writable=True,
                    free_space_mb=-1,  # Unlimited
                )
                logger.debug("s3_storage_detected", bucket=s3_bucket)
            except ImportError:
                logger.debug("s3_not_available", reason="boto3 not installed")

        # Check for GCS configuration
        gcs_bucket = os.environ.get("PROXIMA_GCS_BUCKET")
        if gcs_bucket:
            try:
                from google.cloud import storage as gcs  # noqa: F401

                backends["gcs"] = StorageInfo(
                    name="gcs",
                    storage_type="gcs",
                    path_or_endpoint=f"gs://{gcs_bucket}",
                    is_available=True,
                    is_writable=True,
                    free_space_mb=-1,
                )
                logger.debug("gcs_storage_detected", bucket=gcs_bucket)
            except ImportError:
                logger.debug(
                    "gcs_not_available", reason="google-cloud-storage not installed"
                )

        # Check for Azure Blob configuration
        azure_container = os.environ.get("PROXIMA_AZURE_CONTAINER")
        if azure_container:
            try:
                from azure.storage.blob import BlobServiceClient  # noqa: F401

                backends["azure"] = StorageInfo(
                    name="azure",
                    storage_type="azure",
                    path_or_endpoint=f"azure://{azure_container}",
                    is_available=True,
                    is_writable=True,
                    free_space_mb=-1,
                )
                logger.debug("azure_storage_detected", container=azure_container)
            except ImportError:
                logger.debug(
                    "azure_not_available", reason="azure-storage-blob not installed"
                )

        return backends

    def get_backend(self, name: str | None = None) -> StorageInfo | None:
        """Get a specific storage backend or the default."""
        if name is None:
            name = self._default_backend
        return self._backends.get(name)

    @property
    def available_backends(self) -> list[str]:
        """Get list of available storage backend names."""
        return [k for k, v in self._backends.items() if v.is_available]

    async def save_result(
        self, execution_id: str, data: dict[str, Any], backend: str | None = None
    ) -> str:
        """Save execution result to storage.

        Args:
            execution_id: Unique execution identifier.
            data: Result data to save.
            backend: Storage backend name (uses default if None).

        Returns:
            Path or URI where data was saved.
        """
        import json
        from datetime import datetime

        storage = self.get_backend(backend)
        if not storage or not storage.is_available:
            raise RuntimeError(
                f"Storage backend '{backend or self._default_backend}' not available"
            )

        if storage.storage_type == "local":
            import os

            filename = f"{execution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(storage.path_or_endpoint, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("result_saved", backend="local", path=filepath)
            return filepath

        # For cloud backends, would implement upload logic
        raise NotImplementedError(
            f"Cloud storage backend '{storage.storage_type}' save not yet implemented"
        )

    async def load_result(self, path_or_uri: str) -> dict[str, Any]:
        """Load execution result from storage.

        Args:
            path_or_uri: Path or URI to load from.

        Returns:
            Loaded result data.
        """
        import json
        import os

        if os.path.exists(path_or_uri):
            with open(path_or_uri, encoding="utf-8") as f:
                return json.load(f)

        raise FileNotFoundError(f"Result not found: {path_or_uri}")


# =============================================================================
# NOTIFICATION INTEGRATION
# =============================================================================


@dataclass
class NotificationChannel:
    """Information about a notification channel."""

    name: str
    channel_type: str  # "console", "email", "slack", "webhook"
    endpoint: str
    is_configured: bool = False
    is_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "channel_type": self.channel_type,
            "endpoint": self.endpoint,
            "is_configured": self.is_configured,
            "is_available": self.is_available,
        }


class NotificationIntegration(ExternalIntegration):
    """Integration for sending notifications about execution status."""

    SUPPORTED_TYPES = ["console", "email", "slack", "webhook"]

    def __init__(self):
        super().__init__("notifications", IntegrationType.NOTIFICATION)
        self._channels: dict[str, NotificationChannel] = {}

    async def connect(self) -> bool:
        """Discover and configure notification channels."""
        self._channels = await self._discover_channels()
        self._connected = len(self._channels) > 0
        logger.info(
            "notification_integration_connected",
            channels=list(self._channels.keys()),
        )
        return self._connected

    async def disconnect(self) -> bool:
        """Disconnect notification channels."""
        self._channels = {}
        self._connected = False
        return True

    async def health_check(self) -> IntegrationHealth:
        """Check notification channel availability."""
        available = [k for k, v in self._channels.items() if v.is_available]

        if not available:
            return IntegrationHealth(
                name=self.name,
                integration_type=self.integration_type,
                status=IntegrationStatus.NOT_CONFIGURED,
                message="No notification channels configured",
            )

        return IntegrationHealth(
            name=self.name,
            integration_type=self.integration_type,
            status=IntegrationStatus.AVAILABLE,
            message=f"{len(available)} notification channels available",
            capabilities=available,
            metadata={"channels": {k: v.to_dict() for k, v in self._channels.items()}},
        )

    async def _discover_channels(self) -> dict[str, NotificationChannel]:
        """Discover configured notification channels."""
        import os

        channels = {}

        # Console is always available
        channels["console"] = NotificationChannel(
            name="console",
            channel_type="console",
            endpoint="stdout",
            is_configured=True,
            is_available=True,
        )

        # Check for Slack webhook
        slack_webhook = os.environ.get("PROXIMA_SLACK_WEBHOOK")
        if slack_webhook:
            channels["slack"] = NotificationChannel(
                name="slack",
                channel_type="slack",
                endpoint=slack_webhook[:50] + "...",  # Truncate for logging
                is_configured=True,
                is_available=True,
            )
            logger.debug("slack_channel_detected")

        # Check for generic webhook
        webhook_url = os.environ.get("PROXIMA_WEBHOOK_URL")
        if webhook_url:
            channels["webhook"] = NotificationChannel(
                name="webhook",
                channel_type="webhook",
                endpoint=webhook_url[:50] + "...",
                is_configured=True,
                is_available=True,
            )
            logger.debug("webhook_channel_detected")

        # Check for email configuration
        smtp_host = os.environ.get("PROXIMA_SMTP_HOST")
        email_to = os.environ.get("PROXIMA_EMAIL_TO")
        if smtp_host and email_to:
            channels["email"] = NotificationChannel(
                name="email",
                channel_type="email",
                endpoint=email_to,
                is_configured=True,
                is_available=True,
            )
            logger.debug("email_channel_detected", to=email_to)

        return channels

    def get_channel(self, name: str) -> NotificationChannel | None:
        """Get a specific notification channel."""
        return self._channels.get(name)

    @property
    def available_channels(self) -> list[str]:
        """Get list of available notification channel names."""
        return [k for k, v in self._channels.items() if v.is_available]

    async def send(
        self,
        message: str,
        title: str = "Proxima Notification",
        level: str = "info",
        channels: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """Send notification to specified channels.

        Args:
            message: Notification message.
            title: Notification title.
            level: Notification level (info, warning, error, success).
            channels: List of channel names (uses all available if None).
            metadata: Additional data to include.

        Returns:
            Dict mapping channel names to success status.
        """
        import os

        if channels is None:
            channels = self.available_channels

        results = {}

        for channel_name in channels:
            channel = self._channels.get(channel_name)
            if not channel or not channel.is_available:
                results[channel_name] = False
                continue

            try:
                if channel.channel_type == "console":
                    # Console notification
                    level_icons = {
                        "info": "ℹ️",
                        "warning": "⚠️",
                        "error": "❌",
                        "success": "✅",
                    }
                    icon = level_icons.get(level, "📢")
                    print(f"\n{icon} [{title}] {message}")
                    if metadata:
                        print(f"   Details: {metadata}")
                    results[channel_name] = True

                elif channel.channel_type == "slack":
                    # Slack webhook notification
                    webhook_url = os.environ.get("PROXIMA_SLACK_WEBHOOK")
                    if webhook_url:
                        results[channel_name] = await self._send_slack(
                            webhook_url, title, message, level, metadata
                        )
                    else:
                        results[channel_name] = False

                elif channel.channel_type == "webhook":
                    # Generic webhook notification
                    webhook_url = os.environ.get("PROXIMA_WEBHOOK_URL")
                    if webhook_url:
                        results[channel_name] = await self._send_webhook(
                            webhook_url, title, message, level, metadata
                        )
                    else:
                        results[channel_name] = False

                elif channel.channel_type == "email":
                    # Email notification
                    results[channel_name] = await self._send_email(
                        title, message, level, metadata
                    )

                else:
                    results[channel_name] = False

            except Exception as e:
                logger.error(
                    "notification_send_failed", channel=channel_name, error=str(e)
                )
                results[channel_name] = False

        return results

    async def _send_slack(
        self,
        webhook_url: str,
        title: str,
        message: str,
        level: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Send Slack notification."""
        try:
            import httpx

            color_map = {
                "info": "#2196F3",
                "warning": "#FF9800",
                "error": "#F44336",
                "success": "#4CAF50",
            }

            payload = {
                "attachments": [
                    {
                        "color": color_map.get(level, "#2196F3"),
                        "title": title,
                        "text": message,
                        "fields": [
                            {"title": k, "value": str(v), "short": True}
                            for k, v in (metadata or {}).items()
                        ][
                            :10
                        ],  # Limit fields
                    }
                ]
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload, timeout=10.0)
                return response.status_code == 200

        except ImportError:
            logger.debug("slack_send_failed", reason="httpx not installed")
            return False
        except Exception as e:
            logger.error("slack_send_error", error=str(e))
            return False

    async def _send_webhook(
        self,
        webhook_url: str,
        title: str,
        message: str,
        level: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Send generic webhook notification."""
        try:
            from datetime import datetime

            import httpx

            payload = {
                "title": title,
                "message": message,
                "level": level,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload, timeout=10.0)
                return 200 <= response.status_code < 300

        except ImportError:
            logger.debug("webhook_send_failed", reason="httpx not installed")
            return False
        except Exception as e:
            logger.error("webhook_send_error", error=str(e))
            return False

    async def _send_email(
        self,
        title: str,
        message: str,
        level: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Send email notification."""
        import os

        smtp_host = os.environ.get("PROXIMA_SMTP_HOST")
        smtp_port = int(os.environ.get("PROXIMA_SMTP_PORT", "587"))
        smtp_user = os.environ.get("PROXIMA_SMTP_USER")
        smtp_pass = os.environ.get("PROXIMA_SMTP_PASS")
        email_from = os.environ.get("PROXIMA_EMAIL_FROM", smtp_user)
        email_to = os.environ.get("PROXIMA_EMAIL_TO")

        if not smtp_host or not email_to:
            return False

        # Set default for email_from after validation
        sender = email_from or smtp_user or "proxima@localhost"

        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = email_to
            msg["Subject"] = f"[Proxima - {level.upper()}] {title}"

            body = f"{message}\n\n"
            if metadata:
                body += "Details:\n"
                for k, v in metadata.items():
                    body += f"  - {k}: {v}\n"

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)

            logger.info("email_notification_sent", to=email_to)
            return True

        except Exception as e:
            logger.error("email_send_error", error=str(e))
            return False

    async def notify_execution_complete(
        self,
        execution_id: str,
        status: str,
        duration_seconds: float,
        backend: str | None = None,
    ) -> dict[str, bool]:
        """Send notification about execution completion.

        Args:
            execution_id: Execution identifier.
            status: Execution status (success, failed, cancelled).
            duration_seconds: Execution duration.
            backend: Backend used for execution.

        Returns:
            Dict mapping channel names to success status.
        """
        level = (
            "success"
            if status == "success"
            else "error" if status == "failed" else "warning"
        )

        return await self.send(
            message=f"Execution '{execution_id}' completed with status: {status}",
            title="Execution Complete",
            level=level,
            metadata={
                "execution_id": execution_id,
                "status": status,
                "duration": f"{duration_seconds:.2f}s",
                "backend": backend or "auto",
            },
        )


# =============================================================================
# INTEGRATION MANAGER
# =============================================================================


class IntegrationManager:
    """
    Manages all external integrations for Proxima.

    Provides a unified interface for:
    - Connecting to external systems
    - Health checks
    - Status reporting
    """

    def __init__(self):
        self._integrations: dict[str, ExternalIntegration] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all integrations."""
        # Register standard integrations
        self._integrations["quantum"] = QuantumIntegration()
        self._integrations["llm"] = LLMIntegration()
        self._integrations["system"] = SystemResourceIntegration()
        self._integrations["storage"] = StorageIntegration()
        self._integrations["notifications"] = NotificationIntegration()

        # Connect all
        for name, integration in self._integrations.items():
            try:
                await integration.connect()
                logger.info("integration_initialized", name=name)
            except Exception as e:
                logger.error("integration_init_failed", name=name, error=str(e))

        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown all integrations."""
        for name, integration in self._integrations.items():
            try:
                await integration.disconnect()
            except Exception as e:
                logger.error("integration_shutdown_failed", name=name, error=str(e))

        self._initialized = False

    async def health_check_all(self) -> dict[str, IntegrationHealth]:
        """Check health of all integrations."""
        results = {}
        for name, integration in self._integrations.items():
            try:
                results[name] = await integration.health_check()
            except Exception as e:
                results[name] = IntegrationHealth(
                    name=name,
                    integration_type=integration.integration_type,
                    status=IntegrationStatus.ERROR,
                    message=str(e),
                )
        return results

    def get_integration(self, name: str) -> ExternalIntegration | None:
        """Get a specific integration by name."""
        return self._integrations.get(name)

    @property
    def quantum(self) -> QuantumIntegration:
        """Get the quantum integration."""
        return self._integrations["quantum"]  # type: ignore

    @property
    def llm(self) -> LLMIntegration:
        """Get the LLM integration."""
        return self._integrations["llm"]  # type: ignore

    @property
    def system(self) -> SystemResourceIntegration:
        """Get the system resource integration."""
        return self._integrations["system"]  # type: ignore

    @property
    def storage(self) -> StorageIntegration:
        """Get the storage integration."""
        return self._integrations["storage"]  # type: ignore

    @property
    def notifications(self) -> NotificationIntegration:
        """Get the notification integration."""
        return self._integrations["notifications"]  # type: ignore

    async def get_status_report(self) -> dict[str, Any]:
        """Get a comprehensive status report."""
        health = await self.health_check_all()

        return {
            "initialized": self._initialized,
            "integrations": {name: h.to_dict() for name, h in health.items()},
            "summary": {
                "total": len(health),
                "healthy": sum(1 for h in health.values() if h.is_healthy),
                "unhealthy": sum(1 for h in health.values() if not h.is_healthy),
            },
        }


# Global integration manager instance
_integration_manager: IntegrationManager | None = None


async def get_integration_manager() -> IntegrationManager:
    """Get or create the global integration manager."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = IntegrationManager()
        await _integration_manager.initialize()
    return _integration_manager


__all__ = [
    "IntegrationStatus",
    "IntegrationType",
    "IntegrationHealth",
    "ExternalIntegration",
    "QuantumIntegration",
    "QuantumLibraryInfo",
    "LLMIntegration",
    "LLMProviderInfo",
    "SystemResourceIntegration",
    "SystemResourceInfo",
    "StorageIntegration",
    "StorageInfo",
    "NotificationIntegration",
    "NotificationChannel",
    "IntegrationManager",
    "get_integration_manager",
]
