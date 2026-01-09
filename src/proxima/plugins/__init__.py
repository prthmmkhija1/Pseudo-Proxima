"""
Plugin system for Proxima.

Provides extensibility through custom backends, LLM providers, and exporters.
"""

from proxima.plugins.base import (
    Plugin,
    PluginError,
    PluginLoadError,
    PluginMetadata,
    PluginType,
    PluginValidationError,
)
from proxima.plugins.hooks import (
    HookManager,
    HookType,
    get_hook_manager,
)
from proxima.plugins.loader import (
    PluginLoader,
    PluginRegistry,
    discover_plugins,
    get_plugin_registry,
    load_plugin,
)

__all__ = [
    # Base
    "Plugin",
    "PluginType",
    "PluginMetadata",
    "PluginError",
    "PluginLoadError",
    "PluginValidationError",
    # Loader
    "PluginLoader",
    "PluginRegistry",
    "get_plugin_registry",
    "discover_plugins",
    "load_plugin",
    # Hooks
    "HookType",
    "HookManager",
    "get_hook_manager",
]
