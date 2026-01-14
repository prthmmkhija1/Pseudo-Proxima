"""
Plugin base classes and interfaces.

Defines the plugin protocol and metadata structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar


class PluginType(str, Enum):
    """Types of plugins supported by Proxima."""

    BACKEND = "backend"
    LLM_PROVIDER = "llm_provider"
    EXPORTER = "exporter"
    ANALYZER = "analyzer"
    HOOK = "hook"


class PluginError(Exception):
    """Base exception for plugin errors."""

    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""

    pass


class PluginValidationError(PluginError):
    """Raised when a plugin fails validation."""

    pass


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    homepage: str = ""
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type.value,
            "description": self.description,
            "author": self.author,
            "homepage": self.homepage,
            "requires": self.requires,
            "provides": self.provides,
            "config_schema": self.config_schema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginMetadata:
        """Create metadata from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            plugin_type=PluginType(data["plugin_type"]),
            description=data.get("description", ""),
            author=data.get("author", ""),
            homepage=data.get("homepage", ""),
            requires=data.get("requires", []),
            provides=data.get("provides", []),
            config_schema=data.get("config_schema"),
        )


@dataclass
class PluginContext:
    """Runtime context for plugin execution."""
    
    backend_name: str
    """Name of the current backend (e.g., 'cirq', 'qiskit_aer')."""
    
    num_qubits: int
    """Number of qubits in the current circuit."""
    
    shots: int = 1000
    """Number of measurement shots."""
    
    config: dict[str, Any] | None = None
    """Additional configuration options."""
    
    session_id: str | None = None
    """Current session identifier, if any."""
    
    metadata: dict[str, Any] | None = None
    """Additional metadata."""


class Plugin(ABC):
    """Base class for all Proxima plugins.

    To create a plugin:
    1. Subclass Plugin or a specific plugin type (BackendPlugin, etc.)
    2. Define the METADATA class variable
    3. Implement required abstract methods
    4. Register via entry_points in pyproject.toml or place in plugins directory
    """

    METADATA: ClassVar[PluginMetadata]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the plugin with optional configuration."""
        self._config = config or {}
        self._enabled = True

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self.METADATA.name

    @property
    def version(self) -> str:
        """Get plugin version."""
        return self.METADATA.version

    @property
    def plugin_type(self) -> PluginType:
        """Get plugin type."""
        return self.METADATA.plugin_type

    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the plugin."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the plugin."""
        self._enabled = False

    def configure(self, config: dict[str, Any]) -> None:
        """Update plugin configuration."""
        self._config.update(config)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    @abstractmethod
    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize the plugin. Called after loading.
        
        Args:
            context: Optional plugin context with runtime information
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up the plugin. Called before unloading."""
        ...

    def validate(self) -> list[str]:
        """Validate plugin configuration. Returns list of error messages."""
        errors: list[str] = []
        if not hasattr(self, "METADATA"):
            errors.append("Plugin missing METADATA class variable")
        return errors


class BackendPlugin(Plugin):
    """Base class for quantum backend plugins."""

    METADATA: ClassVar[PluginMetadata]

    @abstractmethod
    def get_backend_class(self) -> type[Any]:
        """Return the backend adapter class."""
        ...

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the backend identifier name."""
        ...

    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize backend plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown backend plugin."""
        pass


class LLMProviderPlugin(Plugin):
    """Base class for LLM provider plugins."""

    METADATA: ClassVar[PluginMetadata]

    @abstractmethod
    def get_provider_class(self) -> type[Any]:
        """Return the LLM provider class."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider identifier name."""
        ...

    @abstractmethod
    def get_api_key_env_var(self) -> str | None:
        """Return the environment variable name for API key, or None."""
        ...

    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize LLM provider plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown LLM provider plugin."""
        pass


class ExporterPlugin(Plugin):
    """Base class for result exporter plugins."""

    METADATA: ClassVar[PluginMetadata]

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the export format name (e.g., 'json', 'csv')."""
        ...

    @abstractmethod
    def export(self, data: Any, destination: str) -> None:
        """Export data to the specified destination."""
        ...

    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize exporter plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown exporter plugin."""
        pass


class AnalyzerPlugin(Plugin):
    """Base class for result analyzer plugins."""

    METADATA: ClassVar[PluginMetadata]

    @abstractmethod
    def get_analyzer_name(self) -> str:
        """Return the analyzer identifier name."""
        ...

    @abstractmethod
    def analyze(self, results: Any) -> dict[str, Any]:
        """Analyze results and return insights."""
        ...

    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize analyzer plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown analyzer plugin."""
        pass
