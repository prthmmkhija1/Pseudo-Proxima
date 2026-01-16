"""
Plugin system for Proxima.

Provides extensibility through custom backends, LLM providers, and exporters.
"""

from proxima.plugins.base import (
    AnalyzerPlugin,
    BackendPlugin,
    ExporterPlugin,
    LLMProviderPlugin,
    Plugin,
    PluginContext,
    PluginError,
    PluginLoadError,
    PluginMetadata,
    PluginType,
    PluginValidationError,
)
from proxima.plugins.hooks import (
    HookContext,
    HookManager,
    HookType,
    RegisteredHook,
    get_hook_manager,
    hook,
)
from proxima.plugins.loader import (
    PluginLoader,
    PluginRegistry,
    discover_plugins,
    get_plugin_loader,
    get_plugin_registry,
    load_plugin,
)
from proxima.plugins.manager import (
    PluginConfig,
    PluginManager,
    PluginState,
    PluginStateSnapshot,
    get_plugin_manager,
    reset_plugin_manager,
)

__all__ = [
    # Base classes
    "Plugin",
    "BackendPlugin",
    "LLMProviderPlugin",
    "ExporterPlugin",
    "AnalyzerPlugin",
    "PluginContext",
    "PluginType",
    "PluginMetadata",
    "PluginError",
    "PluginLoadError",
    "PluginValidationError",
    # Loader
    "PluginLoader",
    "PluginRegistry",
    "get_plugin_registry",
    "get_plugin_loader",
    "discover_plugins",
    "load_plugin",
    # Hooks
    "HookType",
    "HookContext",
    "HookManager",
    "RegisteredHook",
    "get_hook_manager",
    "hook",
    # Manager
    "PluginManager",
    "PluginConfig",
    "PluginState",
    "PluginStateSnapshot",
    "get_plugin_manager",
    "reset_plugin_manager",
]
