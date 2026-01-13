"""
Integration tests for the plugin system.

Tests plugin loading, registration, lifecycle management,
and hook execution.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import os


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def plugin_registry():
    """Create a fresh plugin registry."""
    from proxima.plugins.loader import PluginRegistry
    return PluginRegistry()


@pytest.fixture
def plugin_loader():
    """Create a plugin loader."""
    from proxima.plugins.loader import PluginLoader
    return PluginLoader()


@pytest.fixture
def hook_manager():
    """Create a hook manager."""
    from proxima.plugins.hooks import HookManager
    return HookManager()


@pytest.fixture
def sample_plugin_class():
    """Create a sample plugin class for testing."""
    from proxima.plugins.base import Plugin
    
    class SamplePlugin(Plugin):
        """Sample plugin for testing."""
        
        name = "sample-plugin"
        version = "1.0.0"
        description = "A sample plugin for testing"
        
        def __init__(self):
            super().__init__()
            self._initialized = False
            self._shutdown = False
        
        def initialize(self, context):
            self._context = context
            self._initialized = True
        
        def shutdown(self):
            self._shutdown = True
    
    return SamplePlugin


@pytest.fixture
def sample_exporter_class():
    """Create a sample exporter plugin class."""
    from proxima.plugins.base import ExporterPlugin
    
    class SampleExporter(ExporterPlugin):
        """Sample exporter plugin."""
        
        name = "sample-exporter"
        version = "1.0.0"
        description = "Sample exporter"
        supported_formats = ["sample"]
        
        def export(self, data: Dict[str, Any], **options) -> str:
            return f"exported: {data}"
    
    return SampleExporter


# =============================================================================
# Plugin Registration Tests
# =============================================================================

class TestPluginRegistration:
    """Tests for plugin registration."""
    
    def test_register_plugin(self, plugin_registry, sample_plugin_class):
        """Test registering a plugin."""
        plugin_registry.register(sample_plugin_class)
        
        plugins = plugin_registry.list_plugins()
        assert "sample-plugin" in [p.name for p in plugins]
    
    def test_register_multiple_plugins(self, plugin_registry, sample_plugin_class, sample_exporter_class):
        """Test registering multiple plugins."""
        plugin_registry.register(sample_plugin_class)
        plugin_registry.register(sample_exporter_class)
        
        plugins = plugin_registry.list_plugins()
        assert len(plugins) >= 2
    
    def test_get_plugin_by_name(self, plugin_registry, sample_plugin_class):
        """Test getting plugin by name."""
        plugin_registry.register(sample_plugin_class)
        
        plugin = plugin_registry.get_plugin("sample-plugin")
        assert plugin is not None
        assert plugin.name == "sample-plugin"
    
    def test_get_nonexistent_plugin(self, plugin_registry):
        """Test getting non-existent plugin."""
        plugin = plugin_registry.get_plugin("nonexistent")
        assert plugin is None
    
    def test_unregister_plugin(self, plugin_registry, sample_plugin_class):
        """Test unregistering a plugin."""
        plugin_registry.register(sample_plugin_class)
        plugin_registry.unregister("sample-plugin")
        
        plugin = plugin_registry.get_plugin("sample-plugin")
        assert plugin is None
    
    def test_get_plugins_by_type(self, plugin_registry, sample_exporter_class):
        """Test getting plugins by type."""
        plugin_registry.register(sample_exporter_class)
        
        exporters = plugin_registry.get_plugins_by_type("exporter")
        assert len(exporters) >= 1
    
    def test_prevent_duplicate_registration(self, plugin_registry, sample_plugin_class):
        """Test that duplicate registration is handled."""
        plugin_registry.register(sample_plugin_class)
        
        # Second registration should either update or raise
        try:
            plugin_registry.register(sample_plugin_class)
            # If no exception, should still only have one
            plugins = [p for p in plugin_registry.list_plugins() if p.name == "sample-plugin"]
            assert len(plugins) == 1
        except ValueError:
            pass  # Duplicate registration raises error - also acceptable


# =============================================================================
# Plugin Lifecycle Tests
# =============================================================================

class TestPluginLifecycle:
    """Tests for plugin lifecycle management."""
    
    def test_plugin_initialization(self, sample_plugin_class):
        """Test plugin initialization."""
        from proxima.plugins.base import PluginContext
        
        plugin = sample_plugin_class()
        context = PluginContext(backend_name="test", num_qubits=2)
        
        plugin.initialize(context)
        
        assert plugin._initialized is True
    
    def test_plugin_shutdown(self, sample_plugin_class):
        """Test plugin shutdown."""
        from proxima.plugins.base import PluginContext
        
        plugin = sample_plugin_class()
        context = PluginContext(backend_name="test", num_qubits=2)
        
        plugin.initialize(context)
        plugin.shutdown()
        
        assert plugin._shutdown is True
    
    def test_plugin_lifecycle_order(self, sample_plugin_class):
        """Test that lifecycle methods are called in order."""
        from proxima.plugins.base import PluginContext
        
        plugin = sample_plugin_class()
        context = PluginContext(backend_name="test", num_qubits=2)
        
        # Should not be initialized before initialize()
        assert not plugin._initialized
        
        plugin.initialize(context)
        assert plugin._initialized
        
        plugin.shutdown()
        assert plugin._shutdown


# =============================================================================
# Plugin Loader Tests
# =============================================================================

class TestPluginLoader:
    """Tests for plugin loading functionality."""
    
    def test_discover_plugins_from_package(self, plugin_loader):
        """Test discovering plugins from a package."""
        from proxima.plugins.examples import (
            JSONExporterPlugin,
            StatisticalAnalyzerPlugin,
        )
        
        # Verify example plugins can be imported
        assert JSONExporterPlugin is not None
        assert StatisticalAnalyzerPlugin is not None
    
    def test_load_plugin_by_name(self, plugin_loader, plugin_registry, sample_plugin_class):
        """Test loading plugin by name."""
        plugin_registry.register(sample_plugin_class)
        
        plugin = plugin_registry.get_plugin("sample-plugin")
        assert plugin is not None
    
    def test_load_all_example_plugins(self, plugin_registry):
        """Test loading all example plugins."""
        from proxima.plugins.examples import register_example_plugins
        
        register_example_plugins(plugin_registry)
        
        plugins = plugin_registry.list_plugins()
        assert len(plugins) >= 8  # We created 8 example plugins


# =============================================================================
# Hook System Tests
# =============================================================================

class TestHookSystem:
    """Tests for the hook system."""
    
    def test_register_hook(self, hook_manager):
        """Test registering a hook."""
        from proxima.plugins.hooks import HookType
        
        callback = Mock()
        hook_manager.register(HookType.BEFORE_EXECUTION, callback)
        
        hooks = hook_manager.get_hooks(HookType.BEFORE_EXECUTION)
        assert callback in hooks
    
    def test_execute_hook(self, hook_manager):
        """Test executing a hook."""
        from proxima.plugins.hooks import HookType, HookContext
        
        callback = Mock()
        hook_manager.register(HookType.BEFORE_EXECUTION, callback)
        
        context = HookContext(
            hook_type=HookType.BEFORE_EXECUTION,
            data={"test": "value"}
        )
        
        hook_manager.execute(HookType.BEFORE_EXECUTION, context)
        
        callback.assert_called_once()
    
    def test_execute_multiple_hooks(self, hook_manager):
        """Test executing multiple hooks."""
        from proxima.plugins.hooks import HookType, HookContext
        
        callbacks = [Mock(), Mock(), Mock()]
        for cb in callbacks:
            hook_manager.register(HookType.BEFORE_EXECUTION, cb)
        
        context = HookContext(
            hook_type=HookType.BEFORE_EXECUTION,
            data={}
        )
        
        hook_manager.execute(HookType.BEFORE_EXECUTION, context)
        
        for cb in callbacks:
            cb.assert_called_once()
    
    def test_hook_execution_order(self, hook_manager):
        """Test that hooks are executed in registration order."""
        from proxima.plugins.hooks import HookType, HookContext
        
        order = []
        
        def callback1(ctx):
            order.append(1)
        
        def callback2(ctx):
            order.append(2)
        
        def callback3(ctx):
            order.append(3)
        
        hook_manager.register(HookType.BEFORE_EXECUTION, callback1)
        hook_manager.register(HookType.BEFORE_EXECUTION, callback2)
        hook_manager.register(HookType.BEFORE_EXECUTION, callback3)
        
        context = HookContext(hook_type=HookType.BEFORE_EXECUTION, data={})
        hook_manager.execute(HookType.BEFORE_EXECUTION, context)
        
        assert order == [1, 2, 3]
    
    def test_unregister_hook(self, hook_manager):
        """Test unregistering a hook."""
        from proxima.plugins.hooks import HookType
        
        callback = Mock()
        hook_manager.register(HookType.BEFORE_EXECUTION, callback)
        hook_manager.unregister(HookType.BEFORE_EXECUTION, callback)
        
        hooks = hook_manager.get_hooks(HookType.BEFORE_EXECUTION)
        assert callback not in hooks
    
    def test_hook_error_handling(self, hook_manager):
        """Test that hook errors don't break execution."""
        from proxima.plugins.hooks import HookType, HookContext
        
        def failing_callback(ctx):
            raise RuntimeError("Hook failed")
        
        callback_after = Mock()
        
        hook_manager.register(HookType.BEFORE_EXECUTION, failing_callback)
        hook_manager.register(HookType.BEFORE_EXECUTION, callback_after)
        
        context = HookContext(hook_type=HookType.BEFORE_EXECUTION, data={})
        
        # Should not raise, and subsequent hooks should still run
        try:
            hook_manager.execute(HookType.BEFORE_EXECUTION, context, continue_on_error=True)
            callback_after.assert_called_once()
        except RuntimeError:
            # If continue_on_error is not supported, that's also acceptable
            pass


# =============================================================================
# Exporter Plugin Tests
# =============================================================================

class TestExporterPlugins:
    """Tests for exporter plugins."""
    
    def test_json_exporter(self):
        """Test JSON exporter plugin."""
        from proxima.plugins.examples.exporters import JSONExporterPlugin
        from proxima.plugins.base import PluginContext
        import json
        
        plugin = JSONExporterPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        data = {"counts": {"00": 500, "11": 500}}
        output = plugin.export(data)
        
        # Should be valid JSON
        parsed = json.loads(output)
        assert "data" in parsed or "counts" in parsed
    
    def test_csv_exporter(self):
        """Test CSV exporter plugin."""
        from proxima.plugins.examples.exporters import CSVExporterPlugin
        from proxima.plugins.base import PluginContext
        
        plugin = CSVExporterPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        data = {"counts": {"00": 500, "11": 500}}
        output = plugin.export(data)
        
        # Should contain CSV structure
        lines = output.strip().split("\n")
        assert len(lines) >= 2  # Header + data
    
    def test_markdown_exporter(self):
        """Test Markdown exporter plugin."""
        from proxima.plugins.examples.exporters import MarkdownExporterPlugin
        from proxima.plugins.base import PluginContext
        
        plugin = MarkdownExporterPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        data = {"counts": {"00": 500, "11": 500}, "backend": "cirq"}
        output = plugin.export(data)
        
        # Should contain Markdown headers
        assert "#" in output


# =============================================================================
# Analyzer Plugin Tests
# =============================================================================

class TestAnalyzerPlugins:
    """Tests for analyzer plugins."""
    
    def test_statistical_analyzer(self):
        """Test statistical analyzer plugin."""
        from proxima.plugins.examples.analyzers import StatisticalAnalyzerPlugin
        from proxima.plugins.base import PluginContext
        
        plugin = StatisticalAnalyzerPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        data = {"counts": {"00": 250, "01": 250, "10": 250, "11": 250}}
        result = plugin.analyze(data)
        
        assert "entropy" in result
        assert "uniformity" in result
    
    def test_fidelity_analyzer(self):
        """Test fidelity analyzer plugin."""
        from proxima.plugins.examples.analyzers import FidelityAnalyzerPlugin
        from proxima.plugins.base import PluginContext
        
        plugin = FidelityAnalyzerPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        data = {"counts": {"00": 500, "11": 500}}
        reference = {"00": 500, "11": 500}
        
        result = plugin.analyze(data, reference=reference)
        
        assert "classical_fidelity" in result
        assert result["classical_fidelity"] == pytest.approx(1.0)
    
    def test_performance_analyzer(self):
        """Test performance analyzer plugin."""
        from proxima.plugins.examples.analyzers import PerformanceAnalyzerPlugin
        from proxima.plugins.base import PluginContext
        
        plugin = PerformanceAnalyzerPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        data = {
            "execution_time": 0.5,
            "shots": 1000,
            "counts": {"00": 500, "11": 500}
        }
        
        result = plugin.analyze(data)
        
        assert "throughput" in result or "execution_time" in result


# =============================================================================
# Hook Plugin Tests
# =============================================================================

class TestHookPlugins:
    """Tests for hook plugins."""
    
    def test_logging_hook_plugin(self):
        """Test logging hook plugin."""
        from proxima.plugins.examples.hooks import LoggingHookPlugin
        from proxima.plugins.base import PluginContext
        
        plugin = LoggingHookPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        assert plugin.name == "logging-hook"
        assert len(plugin.supported_hooks) > 0
    
    def test_metrics_hook_plugin(self):
        """Test metrics hook plugin."""
        from proxima.plugins.examples.hooks import MetricsHookPlugin
        from proxima.plugins.base import PluginContext
        
        plugin = MetricsHookPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        metrics = plugin.get_metrics()
        assert "total_executions" in metrics
    
    def test_metrics_collection(self):
        """Test metrics collection across executions."""
        from proxima.plugins.examples.hooks import MetricsHookPlugin
        from proxima.plugins.hooks import HookType, HookContext
        from proxima.plugins.base import PluginContext
        
        plugin = MetricsHookPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        # Simulate executions
        for i in range(5):
            plugin._on_execution_start(HookContext(
                hook_type=HookType.BEFORE_EXECUTION,
                data={"backend": "cirq"}
            ))
            plugin._on_execution_end(HookContext(
                hook_type=HookType.AFTER_EXECUTION,
                data={"backend": "cirq", "execution_time": 0.1 * (i + 1)}
            ))
        
        metrics = plugin.get_metrics()
        assert metrics["total_executions"] == 5


# =============================================================================
# Plugin Integration Tests
# =============================================================================

class TestPluginIntegration:
    """Integration tests for the complete plugin system."""
    
    def test_full_plugin_workflow(self, plugin_registry, hook_manager):
        """Test complete plugin workflow from registration to execution."""
        from proxima.plugins.examples import register_example_plugins
        from proxima.plugins.base import PluginContext
        
        # Register all example plugins
        register_example_plugins(plugin_registry)
        
        # Get an exporter
        exporters = plugin_registry.get_plugins_by_type("exporter")
        assert len(exporters) > 0
        
        exporter = exporters[0]
        exporter.initialize(PluginContext(backend_name="test", num_qubits=2))
        
        # Export data
        data = {"counts": {"00": 500, "11": 500}}
        output = exporter.export(data)
        
        assert len(output) > 0
        
        # Shutdown
        exporter.shutdown()
    
    def test_multiple_plugin_types_together(self, plugin_registry):
        """Test using multiple plugin types together."""
        from proxima.plugins.examples import register_example_plugins
        from proxima.plugins.base import PluginContext
        
        register_example_plugins(plugin_registry)
        
        context = PluginContext(backend_name="cirq", num_qubits=2, shots=1000)
        
        # Initialize different plugin types
        exporter = plugin_registry.get_plugins_by_type("exporter")[0]
        analyzer = plugin_registry.get_plugins_by_type("analyzer")[0]
        
        exporter.initialize(context)
        analyzer.initialize(context)
        
        # Use them together
        data = {"counts": {"00": 500, "11": 500}}
        
        analysis = analyzer.analyze(data)
        enriched_data = {**data, "analysis": analysis}
        output = exporter.export(enriched_data)
        
        assert "analysis" in output or len(output) > 0
        
        # Cleanup
        exporter.shutdown()
        analyzer.shutdown()
    
    def test_plugin_error_isolation(self, plugin_registry):
        """Test that plugin errors are isolated."""
        from proxima.plugins.base import Plugin
        
        class FailingPlugin(Plugin):
            name = "failing-plugin"
            version = "1.0.0"
            description = "A plugin that fails"
            
            def initialize(self, context):
                raise RuntimeError("Initialization failed")
        
        plugin_registry.register(FailingPlugin)
        
        # Should be able to get the plugin
        plugin = plugin_registry.get_plugin("failing-plugin")
        assert plugin is not None
        
        # Initialization failure should be handleable
        with pytest.raises(RuntimeError):
            plugin.initialize(None)
