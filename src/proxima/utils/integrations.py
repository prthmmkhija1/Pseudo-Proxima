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
            metadata={"libraries": {k: v.__dict__ for k, v in self._available_libraries.items()}},
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

        # Check LRET (custom framework)
        try:
            # LRET may not be a standard package
            libraries["lret"] = QuantumLibraryInfo(
                name="lret",
                version="0.1.0",
                simulator_types=["framework"],
                max_qubits=20,
                supports_noise=False,
                supports_gpu=False,
                is_installed=True,  # Placeholder - actual check needed
            )
        except Exception:
            pass

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
        lmstudio_available = await self._check_local_llm(self.DEFAULT_ENDPOINTS["lmstudio"])
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
            k for k, v in self._providers.items() if v.is_available and v.provider_type == "local"
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
    "IntegrationManager",
    "get_integration_manager",
]
