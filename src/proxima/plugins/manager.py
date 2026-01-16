"""
Plugin Manager with persistence support.

Handles plugin state management, configuration persistence,
and plugin lifecycle events with session integration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from proxima.plugins.base import Plugin, PluginMetadata, PluginType
from proxima.plugins.loader import PluginRegistry, get_plugin_registry


class PluginState(str, Enum):
    """Plugin lifecycle states."""

    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UNINSTALLED = "uninstalled"


@dataclass
class PluginConfig:
    """Persistent plugin configuration."""

    name: str
    version: str
    plugin_type: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    state: PluginState = PluginState.ENABLED
    installed_at: datetime | None = None
    last_used: datetime | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type,
            "enabled": self.enabled,
            "config": self.config,
            "state": self.state.value,
            "installed_at": self.installed_at.isoformat() if self.installed_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginConfig:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            plugin_type=data["plugin_type"],
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
            state=PluginState(data.get("state", "enabled")),
            installed_at=datetime.fromisoformat(data["installed_at"]) if data.get("installed_at") else None,
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            error_message=data.get("error_message"),
        )


@dataclass
class PluginStateSnapshot:
    """Snapshot of all plugin states for persistence."""

    configs: dict[str, PluginConfig] = field(default_factory=dict)
    last_saved: datetime | None = None
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "last_saved": self.last_saved.isoformat() if self.last_saved else None,
            "configs": {name: cfg.to_dict() for name, cfg in self.configs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginStateSnapshot:
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            last_saved=datetime.fromisoformat(data["last_saved"]) if data.get("last_saved") else None,
            configs={name: PluginConfig.from_dict(cfg) for name, cfg in data.get("configs", {}).items()},
        )


class PluginManager:
    """
    Manages plugin lifecycle with persistent state.
    
    Features:
    - Enable/disable plugins
    - Persist plugin configurations
    - Track plugin usage
    - Manage plugin dependencies
    - Session integration
    """

    DEFAULT_CONFIG_PATH = Path.home() / ".proxima" / "plugins.json"

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Initialize the plugin manager.

        Args:
            registry: Plugin registry to manage. Uses global if not provided.
            config_path: Path to store plugin configuration. Uses default if not provided.
        """
        self._registry = registry or get_plugin_registry()
        self._config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._snapshot = PluginStateSnapshot()
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted plugin state from disk."""
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                self._snapshot = PluginStateSnapshot.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                self._snapshot = PluginStateSnapshot()

    def _save_state(self) -> None:
        """Save plugin state to disk."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot.last_saved = datetime.utcnow()
        self._config_path.write_text(
            json.dumps(self._snapshot.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin and update persistent state."""
        name = plugin.name
        
        # Create or update config
        if name not in self._snapshot.configs:
            self._snapshot.configs[name] = PluginConfig(
                name=name,
                version=plugin.version,
                plugin_type=plugin.plugin_type.value,
                enabled=True,
                installed_at=datetime.utcnow(),
            )
        
        # Apply persisted enabled state
        config = self._snapshot.configs[name]
        if not config.enabled:
            plugin.disable()
        
        # Apply persisted configuration
        if config.config:
            plugin.configure(config.config)
        
        self._save_state()

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin by name."""
        plugin = self._registry.get(name)
        if not plugin:
            return False
        
        plugin.enable()
        
        if name in self._snapshot.configs:
            self._snapshot.configs[name].enabled = True
            self._snapshot.configs[name].state = PluginState.ENABLED
        
        self._save_state()
        return True

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin by name."""
        plugin = self._registry.get(name)
        if not plugin:
            return False
        
        plugin.disable()
        
        if name in self._snapshot.configs:
            self._snapshot.configs[name].enabled = False
            self._snapshot.configs[name].state = PluginState.DISABLED
        
        self._save_state()
        return True

    def configure_plugin(self, name: str, config: dict[str, Any]) -> bool:
        """Update a plugin's configuration."""
        plugin = self._registry.get(name)
        if not plugin:
            return False
        
        plugin.configure(config)
        
        if name in self._snapshot.configs:
            self._snapshot.configs[name].config.update(config)
        
        self._save_state()
        return True

    def mark_plugin_used(self, name: str) -> None:
        """Mark a plugin as recently used."""
        if name in self._snapshot.configs:
            self._snapshot.configs[name].last_used = datetime.utcnow()
            self._save_state()

    def set_plugin_error(self, name: str, error: str) -> None:
        """Set error state for a plugin."""
        if name in self._snapshot.configs:
            self._snapshot.configs[name].state = PluginState.ERROR
            self._snapshot.configs[name].error_message = error
            self._save_state()

    def clear_plugin_error(self, name: str) -> None:
        """Clear error state for a plugin."""
        if name in self._snapshot.configs:
            config = self._snapshot.configs[name]
            config.error_message = None
            config.state = PluginState.ENABLED if config.enabled else PluginState.DISABLED
            self._save_state()

    def get_plugin_config(self, name: str) -> PluginConfig | None:
        """Get the persistent configuration for a plugin."""
        return self._snapshot.configs.get(name)

    def get_all_configs(self) -> dict[str, PluginConfig]:
        """Get all plugin configurations."""
        return dict(self._snapshot.configs)

    def get_enabled_plugins(self) -> list[str]:
        """Get list of enabled plugin names."""
        return [
            name for name, cfg in self._snapshot.configs.items()
            if cfg.enabled and cfg.state != PluginState.ERROR
        ]

    def get_disabled_plugins(self) -> list[str]:
        """Get list of disabled plugin names."""
        return [
            name for name, cfg in self._snapshot.configs.items()
            if not cfg.enabled
        ]

    def get_plugins_by_type(self, plugin_type: PluginType) -> list[str]:
        """Get plugin names by type."""
        return [
            name for name, cfg in self._snapshot.configs.items()
            if cfg.plugin_type == plugin_type.value
        ]

    def get_plugin_stats(self) -> dict[str, Any]:
        """Get statistics about plugins."""
        configs = self._snapshot.configs
        return {
            "total": len(configs),
            "enabled": sum(1 for c in configs.values() if c.enabled),
            "disabled": sum(1 for c in configs.values() if not c.enabled),
            "error": sum(1 for c in configs.values() if c.state == PluginState.ERROR),
            "by_type": {
                pt.value: sum(1 for c in configs.values() if c.plugin_type == pt.value)
                for pt in PluginType
            },
        }

    def export_state(self) -> dict[str, Any]:
        """Export the complete plugin state for backup."""
        return self._snapshot.to_dict()

    def import_state(self, data: dict[str, Any], merge: bool = False) -> int:
        """Import plugin state from backup.
        
        Args:
            data: State data to import
            merge: If True, merge with existing state; if False, replace
            
        Returns:
            Number of plugins imported
        """
        imported = PluginStateSnapshot.from_dict(data)
        
        if merge:
            for name, cfg in imported.configs.items():
                self._snapshot.configs[name] = cfg
        else:
            self._snapshot = imported
        
        self._save_state()
        
        # Apply state to registered plugins
        for name, cfg in self._snapshot.configs.items():
            plugin = self._registry.get(name)
            if plugin:
                if cfg.enabled:
                    plugin.enable()
                else:
                    plugin.disable()
                if cfg.config:
                    plugin.configure(cfg.config)
        
        return len(imported.configs)

    def reset_all(self) -> None:
        """Reset all plugin configurations to defaults."""
        self._snapshot = PluginStateSnapshot()
        self._save_state()
        
        # Re-enable all registered plugins
        for plugin in self._registry.list_all():
            plugin.enable()


# Global plugin manager singleton
_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def reset_plugin_manager() -> None:
    """Reset the global plugin manager."""
    global _plugin_manager
    _plugin_manager = None
