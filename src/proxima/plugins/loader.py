"""
Plugin loader and registry.

Handles discovering, loading, and managing plugins from:
- Entry points (installed packages)
- Plugin directories
- Explicit registration
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from proxima.plugins.base import (
    Plugin,
    PluginError,
    PluginLoadError,
    PluginType,
    PluginValidationError,
)

# Entry point group names
ENTRY_POINT_GROUPS = {
    PluginType.BACKEND: "proxima.backends",
    PluginType.LLM_PROVIDER: "proxima.llm_providers",
    PluginType.EXPORTER: "proxima.exporters",
    PluginType.ANALYZER: "proxima.analyzers",
    PluginType.HOOK: "proxima.hooks",
}


class PluginRegistry:
    """Central registry for all loaded plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._by_type: dict[PluginType, dict[str, Plugin]] = {pt: {} for pt in PluginType}
        self._load_callbacks: list[Callable[[Plugin], None]] = []
        self._unload_callbacks: list[Callable[[Plugin], None]] = []

    def register(self, plugin: Plugin) -> None:
        """Register a plugin instance."""
        name = plugin.name
        ptype = plugin.plugin_type

        # Validate before registering
        errors = plugin.validate()
        if errors:
            raise PluginValidationError(f"Plugin '{name}' validation failed: {'; '.join(errors)}")

        if name in self._plugins:
            raise PluginError(f"Plugin '{name}' is already registered")

        self._plugins[name] = plugin
        self._by_type[ptype][name] = plugin

        # Initialize the plugin
        try:
            plugin.initialize()
        except Exception as e:
            # Rollback registration on init failure
            del self._plugins[name]
            del self._by_type[ptype][name]
            raise PluginLoadError(f"Plugin '{name}' initialization failed: {e}") from e

        # Notify callbacks
        for callback in self._load_callbacks:
            try:
                callback(plugin)
            except Exception:
                pass  # Don't fail on callback errors

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name."""
        if name not in self._plugins:
            return False

        plugin = self._plugins[name]

        # Notify callbacks before shutdown
        for callback in self._unload_callbacks:
            try:
                callback(plugin)
            except Exception:
                pass

        # Shutdown the plugin
        try:
            plugin.shutdown()
        except Exception:
            pass  # Best effort shutdown

        del self._plugins[name]
        del self._by_type[plugin.plugin_type][name]
        return True

    def get(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_by_type(self, plugin_type: PluginType) -> dict[str, Plugin]:
        """Get all plugins of a specific type."""
        return dict(self._by_type[plugin_type])

    def list_all(self) -> list[Plugin]:
        """List all registered plugins."""
        return list(self._plugins.values())

    def list_names(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins

    def on_load(self, callback: Callable[[Plugin], None]) -> None:
        """Register a callback for plugin load events."""
        self._load_callbacks.append(callback)

    def on_unload(self, callback: Callable[[Plugin], None]) -> None:
        """Register a callback for plugin unload events."""
        self._unload_callbacks.append(callback)

    def clear(self) -> None:
        """Unregister all plugins."""
        for name in list(self._plugins.keys()):
            self.unregister(name)


class PluginLoader:
    """Loads plugins from various sources."""

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry
        self._plugin_dirs: list[Path] = []

    def add_plugin_dir(self, path: Path) -> None:
        """Add a directory to search for plugins."""
        if path.is_dir():
            self._plugin_dirs.append(path)

    def load_from_entry_points(self, plugin_type: PluginType | None = None) -> int:
        """Load plugins from installed package entry points.

        Returns the number of plugins loaded.
        """
        loaded = 0
        groups = (
            [ENTRY_POINT_GROUPS[plugin_type]] if plugin_type else list(ENTRY_POINT_GROUPS.values())
        )

        for group in groups:
            try:
                eps = importlib.metadata.entry_points(group=group)
            except TypeError:
                # Python 3.9 compatibility
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get(group, [])

            for ep in eps:
                try:
                    plugin_class = ep.load()
                    if isinstance(plugin_class, type) and issubclass(plugin_class, Plugin):
                        plugin = plugin_class()
                        self.registry.register(plugin)
                        loaded += 1
                except Exception as e:
                    # Log but don't fail on individual plugin load errors
                    print(f"Warning: Failed to load plugin '{ep.name}': {e}")

        return loaded

    def load_from_directory(self, directory: Path | None = None) -> int:
        """Load plugins from a directory.

        Looks for Python files or packages with a `Plugin` class.
        Returns the number of plugins loaded.
        """
        loaded = 0
        dirs = [directory] if directory else self._plugin_dirs

        for plugin_dir in dirs:
            if not plugin_dir or not plugin_dir.is_dir():
                continue

            for path in plugin_dir.iterdir():
                if path.suffix == ".py" and not path.name.startswith("_"):
                    try:
                        plugin = self._load_from_file(path)
                        if plugin:
                            self.registry.register(plugin)
                            loaded += 1
                    except Exception as e:
                        print(f"Warning: Failed to load plugin from '{path}': {e}")

                elif path.is_dir() and (path / "__init__.py").exists():
                    try:
                        plugin = self._load_from_package(path)
                        if plugin:
                            self.registry.register(plugin)
                            loaded += 1
                    except Exception as e:
                        print(f"Warning: Failed to load plugin from '{path}': {e}")

        return loaded

    def load_from_class(self, plugin_class: type[Plugin], config: dict | None = None) -> Plugin:
        """Load a plugin from a class directly."""
        plugin = plugin_class(config)
        self.registry.register(plugin)
        return plugin

    def _load_from_file(self, path: Path) -> Plugin | None:
        """Load a plugin from a Python file."""
        module_name = f"proxima_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return self._find_plugin_in_module(module)

    def _load_from_package(self, path: Path) -> Plugin | None:
        """Load a plugin from a package directory."""
        module_name = f"proxima_plugin_{path.name}"
        spec = importlib.util.spec_from_file_location(
            module_name, path / "__init__.py", submodule_search_locations=[str(path)]
        )
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return self._find_plugin_in_module(module)

    def _find_plugin_in_module(self, module: Any) -> Plugin | None:
        """Find and instantiate a Plugin class in a module."""
        # Look for a class named 'Plugin' or any subclass of Plugin
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Plugin)
                and obj is not Plugin
                and hasattr(obj, "METADATA")
            ):
                return obj()
        return None


# Global registry singleton
_registry: PluginRegistry | None = None
_loader: PluginLoader | None = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def get_plugin_loader() -> PluginLoader:
    """Get the global plugin loader."""
    global _loader
    if _loader is None:
        _loader = PluginLoader(get_plugin_registry())
    return _loader


def discover_plugins(
    plugin_type: PluginType | None = None,
    include_entry_points: bool = True,
    plugin_dirs: list[Path] | None = None,
) -> int:
    """Discover and load plugins from all sources.

    Returns the total number of plugins loaded.
    """
    loader = get_plugin_loader()
    loaded = 0

    if include_entry_points:
        loaded += loader.load_from_entry_points(plugin_type)

    if plugin_dirs:
        for pdir in plugin_dirs:
            loader.add_plugin_dir(pdir)
        loaded += loader.load_from_directory()

    return loaded


def load_plugin(plugin_class: type[Plugin], config: dict | None = None) -> Plugin:
    """Load a single plugin from a class."""
    return get_plugin_loader().load_from_class(plugin_class, config)
