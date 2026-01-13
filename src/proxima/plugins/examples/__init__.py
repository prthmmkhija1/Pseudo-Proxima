"""
Example Plugins for Proxima.

This module contains example plugin implementations that demonstrate
how to extend Proxima's functionality through the plugin system.

Available Example Plugins:
- JSONExporterPlugin: Export results to JSON format
- CSVExporterPlugin: Export results to CSV format
- StatisticalAnalyzerPlugin: Analyze circuit execution statistics
- FidelityAnalyzerPlugin: Analyze fidelity between backends
- LoggingHookPlugin: Log execution events

Usage:
    # Register example plugins
    from proxima.plugins.examples import register_example_plugins
    register_example_plugins()

    # Or import individual plugins
    from proxima.plugins.examples import JSONExporterPlugin
"""

from proxima.plugins.examples.exporters import (
    JSONExporterPlugin,
    CSVExporterPlugin,
    MarkdownExporterPlugin,
)
from proxima.plugins.examples.analyzers import (
    StatisticalAnalyzerPlugin,
    FidelityAnalyzerPlugin,
    PerformanceAnalyzerPlugin,
)
from proxima.plugins.examples.hooks import (
    LoggingHookPlugin,
    MetricsHookPlugin,
)

__all__ = [
    # Exporters
    "JSONExporterPlugin",
    "CSVExporterPlugin",
    "MarkdownExporterPlugin",
    # Analyzers
    "StatisticalAnalyzerPlugin",
    "FidelityAnalyzerPlugin",
    "PerformanceAnalyzerPlugin",
    # Hooks
    "LoggingHookPlugin",
    "MetricsHookPlugin",
    # Registration
    "register_example_plugins",
]


def register_example_plugins() -> list[str]:
    """Register all example plugins with the plugin registry.
    
    Returns:
        List of registered plugin names.
    """
    from proxima.plugins import get_plugin_registry
    
    registry = get_plugin_registry()
    registered = []
    
    example_plugins = [
        JSONExporterPlugin,
        CSVExporterPlugin,
        MarkdownExporterPlugin,
        StatisticalAnalyzerPlugin,
        FidelityAnalyzerPlugin,
        PerformanceAnalyzerPlugin,
        LoggingHookPlugin,
        MetricsHookPlugin,
    ]
    
    for plugin_class in example_plugins:
        try:
            plugin = plugin_class()
            registry.register(plugin)
            registered.append(plugin.name)
        except Exception as e:
            # Log but don't fail
            import logging
            logging.warning(f"Failed to register {plugin_class.__name__}: {e}")
    
    return registered
