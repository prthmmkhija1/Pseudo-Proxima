"""
Comprehensive Plugin System Tests.

Tests for plugin base classes, registry, loader, and manager.
This complements test_example_plugins.py by testing the core
plugin infrastructure.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List


# =============================================================================
# Plugin Base Classes Tests
# =============================================================================


class TestPluginType:
    """Tests for PluginType enum."""

    def test_all_plugin_types_exist(self):
        """Test all expected plugin types exist."""
        from proxima.plugins.base import PluginType

        expected_types = ["BACKEND", "LLM_PROVIDER", "EXPORTER", "ANALYZER", "HOOK"]

        for ptype in expected_types:
            assert hasattr(PluginType, ptype)

    def test_plugin_type_values(self):
        """Test plugin type string values."""
        from proxima.plugins.base import PluginType

        assert PluginType.BACKEND.value == "backend"
        assert PluginType.LLM_PROVIDER.value == "llm_provider"
        assert PluginType.EXPORTER.value == "exporter"
        assert PluginType.ANALYZER.value == "analyzer"
        assert PluginType.HOOK.value == "hook"

    def test_plugin_type_is_string_enum(self):
        """Test PluginType is a string enum."""
        from proxima.plugins.base import PluginType

        assert isinstance(PluginType.BACKEND, str)
        assert PluginType.BACKEND == "backend"


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    def test_create_minimal_metadata(self):
        """Test creating metadata with required fields only."""
        from proxima.plugins.base import PluginMetadata, PluginType

        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            plugin_type=PluginType.EXPORTER,
        )

        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.EXPORTER
        assert metadata.description == ""
        assert metadata.author == ""
        assert metadata.requires == []
        assert metadata.provides == []

    def test_create_full_metadata(self):
        """Test creating metadata with all fields."""
        from proxima.plugins.base import PluginMetadata, PluginType

        metadata = PluginMetadata(
            name="full_plugin",
            version="2.0.0",
            plugin_type=PluginType.ANALYZER,
            description="A full plugin",
            author="Test Author",
            homepage="https://example.com",
            requires=["numpy", "scipy"],
            provides=["analysis", "metrics"],
            config_schema={"type": "object"},
        )

        assert metadata.description == "A full plugin"
        assert metadata.author == "Test Author"
        assert metadata.homepage == "https://example.com"
        assert "numpy" in metadata.requires
        assert "analysis" in metadata.provides
        assert metadata.config_schema == {"type": "object"}

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        from proxima.plugins.base import PluginMetadata, PluginType

        metadata = PluginMetadata(
            name="dict_plugin",
            version="1.0.0",
            plugin_type=PluginType.HOOK,
            description="Test description",
        )

        d = metadata.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "dict_plugin"
        assert d["version"] == "1.0.0"
        assert d["plugin_type"] == "hook"
        assert d["description"] == "Test description"

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        from proxima.plugins.base import PluginMetadata, PluginType

        data = {
            "name": "from_dict_plugin",
            "version": "3.0.0",
            "plugin_type": "exporter",
            "description": "From dict",
            "author": "Author",
            "homepage": "https://test.com",
            "requires": ["dep1"],
            "provides": ["feature1"],
        }

        metadata = PluginMetadata.from_dict(data)

        assert metadata.name == "from_dict_plugin"
        assert metadata.version == "3.0.0"
        assert metadata.plugin_type == PluginType.EXPORTER
        assert metadata.description == "From dict"

    def test_metadata_roundtrip(self):
        """Test metadata serialization roundtrip."""
        from proxima.plugins.base import PluginMetadata, PluginType

        original = PluginMetadata(
            name="roundtrip_plugin",
            version="1.2.3",
            plugin_type=PluginType.BACKEND,
            description="Roundtrip test",
            requires=["req1", "req2"],
        )

        d = original.to_dict()
        restored = PluginMetadata.from_dict(d)

        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.plugin_type == original.plugin_type
        assert restored.description == original.description
        assert restored.requires == original.requires


class TestPluginContext:
    """Tests for PluginContext dataclass."""

    def test_create_minimal_context(self):
        """Test creating context with required fields."""
        from proxima.plugins.base import PluginContext

        context = PluginContext(
            backend_name="cirq",
            num_qubits=4,
        )

        assert context.backend_name == "cirq"
        assert context.num_qubits == 4
        assert context.shots == 1000  # Default
        assert context.config is None

    def test_create_full_context(self):
        """Test creating context with all fields."""
        from proxima.plugins.base import PluginContext

        context = PluginContext(
            backend_name="qiskit_aer",
            num_qubits=8,
            shots=2000,
            config={"precision": "double"},
            session_id="session_123",
            metadata={"circuit_name": "bell_state"},
        )

        assert context.shots == 2000
        assert context.config["precision"] == "double"
        assert context.session_id == "session_123"
        assert context.metadata["circuit_name"] == "bell_state"


class TestPluginErrors:
    """Tests for plugin error classes."""

    def test_plugin_error(self):
        """Test PluginError base exception."""
        from proxima.plugins.base import PluginError

        with pytest.raises(PluginError):
            raise PluginError("Test error")

    def test_plugin_load_error(self):
        """Test PluginLoadError exception."""
        from proxima.plugins.base import PluginLoadError, PluginError

        with pytest.raises(PluginLoadError):
            raise PluginLoadError("Load failed")

        # Should be a subclass of PluginError
        assert issubclass(PluginLoadError, PluginError)

    def test_plugin_validation_error(self):
        """Test PluginValidationError exception."""
        from proxima.plugins.base import PluginValidationError, PluginError

        with pytest.raises(PluginValidationError):
            raise PluginValidationError("Validation failed")

        assert issubclass(PluginValidationError, PluginError)


# =============================================================================
# Plugin Registry Tests
# =============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        from proxima.plugins.loader import PluginRegistry

        return PluginRegistry()

    @pytest.fixture
    def mock_plugin(self):
        """Create a mock plugin."""
        from proxima.plugins.base import PluginMetadata, PluginType, Plugin

        plugin = Mock(spec=Plugin)
        plugin.name = "test_plugin"
        plugin.plugin_type = PluginType.EXPORTER
        plugin.METADATA = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            plugin_type=PluginType.EXPORTER,
        )
        plugin.validate.return_value = []  # No errors
        plugin.initialize.return_value = None
        plugin.shutdown.return_value = None
        return plugin

    def test_register_plugin(self, registry, mock_plugin):
        """Test registering a plugin."""
        registry.register(mock_plugin)

        assert registry.is_registered("test_plugin")
        assert registry.get("test_plugin") is mock_plugin

    def test_register_duplicate_raises_error(self, registry, mock_plugin):
        """Test registering duplicate plugin raises error."""
        from proxima.plugins.base import PluginError

        registry.register(mock_plugin)

        with pytest.raises(PluginError):
            registry.register(mock_plugin)

    def test_unregister_plugin(self, registry, mock_plugin):
        """Test unregistering a plugin."""
        registry.register(mock_plugin)
        result = registry.unregister("test_plugin")

        assert result is True
        assert not registry.is_registered("test_plugin")
        mock_plugin.shutdown.assert_called_once()

    def test_unregister_nonexistent_returns_false(self, registry):
        """Test unregistering nonexistent plugin returns False."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_by_type(self, registry, mock_plugin):
        """Test getting plugins by type."""
        from proxima.plugins.base import PluginType

        registry.register(mock_plugin)

        exporters = registry.get_by_type(PluginType.EXPORTER)

        assert "test_plugin" in exporters
        assert len(registry.get_by_type(PluginType.ANALYZER)) == 0

    def test_list_all(self, registry, mock_plugin):
        """Test listing all plugins."""
        registry.register(mock_plugin)

        all_plugins = registry.list_all()

        assert len(all_plugins) == 1
        assert mock_plugin in all_plugins

    def test_list_names(self, registry, mock_plugin):
        """Test listing plugin names."""
        registry.register(mock_plugin)

        names = registry.list_names()

        assert "test_plugin" in names

    def test_clear(self, registry, mock_plugin):
        """Test clearing all plugins."""
        registry.register(mock_plugin)
        registry.clear()

        assert len(registry.list_all()) == 0

    def test_on_load_callback(self, registry, mock_plugin):
        """Test load callback is called."""
        callback = Mock()
        registry.on_load(callback)

        registry.register(mock_plugin)

        callback.assert_called_once_with(mock_plugin)

    def test_on_unload_callback(self, registry, mock_plugin):
        """Test unload callback is called."""
        callback = Mock()
        registry.on_unload(callback)

        registry.register(mock_plugin)
        registry.unregister("test_plugin")

        callback.assert_called_once_with(mock_plugin)

    def test_validation_failure_prevents_registration(self, registry):
        """Test plugin validation failure prevents registration."""
        from proxima.plugins.base import (
            PluginMetadata,
            PluginType,
            Plugin,
            PluginValidationError,
        )

        plugin = Mock(spec=Plugin)
        plugin.name = "invalid_plugin"
        plugin.plugin_type = PluginType.EXPORTER
        plugin.validate.return_value = ["Error 1", "Error 2"]

        with pytest.raises(PluginValidationError):
            registry.register(plugin)

        assert not registry.is_registered("invalid_plugin")


# =============================================================================
# Plugin Loader Tests
# =============================================================================


class TestPluginLoader:
    """Tests for PluginLoader."""

    @pytest.fixture
    def loader(self):
        """Create a plugin loader."""
        from proxima.plugins.loader import PluginLoader, PluginRegistry

        registry = PluginRegistry()
        return PluginLoader(registry)

    def test_add_plugin_dir(self, loader, tmp_path):
        """Test adding a plugin directory."""
        loader.add_plugin_dir(tmp_path)

        assert tmp_path in loader._plugin_dirs

    def test_add_nonexistent_dir_ignored(self, loader):
        """Test adding nonexistent directory is ignored."""
        fake_path = Path("/nonexistent/path")
        loader.add_plugin_dir(fake_path)

        assert fake_path not in loader._plugin_dirs


class TestGlobalPluginRegistry:
    """Tests for global plugin registry functions."""

    def test_get_plugin_registry(self):
        """Test getting global registry."""
        from proxima.plugins.loader import get_plugin_registry

        registry = get_plugin_registry()

        assert registry is not None
        # Should return same instance
        assert get_plugin_registry() is registry


# =============================================================================
# Plugin Manager Tests
# =============================================================================


class TestPluginState:
    """Tests for PluginState enum."""

    def test_all_states_exist(self):
        """Test all plugin states exist."""
        from proxima.plugins.manager import PluginState

        expected = ["INSTALLED", "ENABLED", "DISABLED", "ERROR", "UNINSTALLED"]

        for state in expected:
            assert hasattr(PluginState, state)

    def test_state_values(self):
        """Test state string values."""
        from proxima.plugins.manager import PluginState

        assert PluginState.INSTALLED.value == "installed"
        assert PluginState.ENABLED.value == "enabled"
        assert PluginState.DISABLED.value == "disabled"
        assert PluginState.ERROR.value == "error"


class TestPluginConfig:
    """Tests for PluginConfig dataclass."""

    def test_create_config(self):
        """Test creating plugin config."""
        from proxima.plugins.manager import PluginConfig, PluginState

        config = PluginConfig(
            name="test_plugin",
            version="1.0.0",
            plugin_type="exporter",
        )

        assert config.name == "test_plugin"
        assert config.enabled is True
        assert config.state == PluginState.ENABLED

    def test_config_to_dict(self):
        """Test config serialization."""
        from proxima.plugins.manager import PluginConfig, PluginState

        config = PluginConfig(
            name="test_plugin",
            version="1.0.0",
            plugin_type="exporter",
            enabled=False,
            state=PluginState.DISABLED,
        )

        d = config.to_dict()

        assert d["name"] == "test_plugin"
        assert d["enabled"] is False
        assert d["state"] == "disabled"

    def test_config_from_dict(self):
        """Test config deserialization."""
        from proxima.plugins.manager import PluginConfig, PluginState

        data = {
            "name": "from_dict",
            "version": "2.0.0",
            "plugin_type": "analyzer",
            "enabled": True,
            "state": "enabled",
            "config": {"key": "value"},
        }

        config = PluginConfig.from_dict(data)

        assert config.name == "from_dict"
        assert config.version == "2.0.0"
        assert config.config == {"key": "value"}


class TestPluginStateSnapshot:
    """Tests for PluginStateSnapshot."""

    def test_create_snapshot(self):
        """Test creating a state snapshot."""
        from proxima.plugins.manager import PluginStateSnapshot

        snapshot = PluginStateSnapshot()

        assert snapshot.configs == {}
        assert snapshot.version == "1.0"

    def test_snapshot_roundtrip(self):
        """Test snapshot serialization roundtrip."""
        from proxima.plugins.manager import PluginStateSnapshot, PluginConfig

        snapshot = PluginStateSnapshot()
        snapshot.configs["test"] = PluginConfig(
            name="test",
            version="1.0.0",
            plugin_type="hook",
        )

        d = snapshot.to_dict()
        restored = PluginStateSnapshot.from_dict(d)

        assert "test" in restored.configs
        assert restored.configs["test"].name == "test"


class TestPluginManager:
    """Tests for PluginManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a plugin manager with temp config path."""
        from proxima.plugins.manager import PluginManager
        from proxima.plugins.loader import PluginRegistry

        registry = PluginRegistry()
        config_path = tmp_path / "plugins.json"
        return PluginManager(registry=registry, config_path=config_path)

    @pytest.fixture
    def mock_plugin(self):
        """Create a mock plugin for testing."""
        from proxima.plugins.base import PluginMetadata, PluginType, Plugin

        plugin = Mock(spec=Plugin)
        plugin.name = "managed_plugin"
        plugin.version = "1.0.0"
        plugin.plugin_type = PluginType.EXPORTER
        plugin.METADATA = PluginMetadata(
            name="managed_plugin",
            version="1.0.0",
            plugin_type=PluginType.EXPORTER,
        )
        plugin.validate.return_value = []
        plugin.initialize.return_value = None
        plugin.shutdown.return_value = None
        plugin.enable.return_value = None
        plugin.disable.return_value = None
        plugin.enabled = True
        return plugin

    def test_register_plugin(self, manager, mock_plugin):
        """Test registering plugin with manager."""
        manager.register_plugin(mock_plugin)

        # Should be in registry
        assert manager._registry.is_registered("managed_plugin")

    def test_state_persistence(self, manager, mock_plugin, tmp_path):
        """Test plugin state is persisted."""
        manager.register_plugin(mock_plugin)

        # State should be saved
        config_path = tmp_path / "plugins.json"
        assert config_path.exists()

        # Load state and verify
        data = json.loads(config_path.read_text())
        assert "configs" in data


# =============================================================================
# Plugin Lifecycle Tests
# =============================================================================


class TestPluginLifecycle:
    """Tests for plugin lifecycle management."""

    def test_plugin_initialization(self):
        """Test plugin initialize method is called."""
        from proxima.plugins.base import Plugin, PluginMetadata, PluginType

        # Create a concrete test plugin
        class TestPlugin(Plugin):
            METADATA = PluginMetadata(
                name="lifecycle_test",
                version="1.0.0",
                plugin_type=PluginType.HOOK,
            )

            def __init__(self):
                super().__init__()
                self.initialized = False
                self.shutdown_called = False

            def initialize(self, context=None):
                self.initialized = True

            def shutdown(self):
                self.shutdown_called = True

            def validate(self):
                return []

        plugin = TestPlugin()
        plugin.initialize()

        assert plugin.initialized is True

    def test_plugin_enable_disable(self):
        """Test plugin enable/disable."""
        from proxima.plugins.base import Plugin, PluginMetadata, PluginType

        class TogglePlugin(Plugin):
            METADATA = PluginMetadata(
                name="toggle_test",
                version="1.0.0",
                plugin_type=PluginType.EXPORTER,
            )

            def initialize(self, context=None):
                pass

            def shutdown(self):
                pass

            def validate(self):
                return []

        plugin = TogglePlugin()

        assert plugin.enabled is True
        plugin.disable()
        assert plugin.enabled is False
        plugin.enable()
        assert plugin.enabled is True


# =============================================================================
# Entry Point Discovery Tests
# =============================================================================


class TestEntryPointGroups:
    """Tests for entry point configuration."""

    def test_entry_point_groups_defined(self):
        """Test entry point groups are defined for all types."""
        from proxima.plugins.loader import ENTRY_POINT_GROUPS
        from proxima.plugins.base import PluginType

        for ptype in PluginType:
            assert ptype in ENTRY_POINT_GROUPS

    def test_entry_point_group_names(self):
        """Test entry point group names follow convention."""
        from proxima.plugins.loader import ENTRY_POINT_GROUPS
        from proxima.plugins.base import PluginType

        assert ENTRY_POINT_GROUPS[PluginType.BACKEND] == "proxima.backends"
        assert ENTRY_POINT_GROUPS[PluginType.LLM_PROVIDER] == "proxima.llm_providers"
        assert ENTRY_POINT_GROUPS[PluginType.EXPORTER] == "proxima.exporters"
        assert ENTRY_POINT_GROUPS[PluginType.ANALYZER] == "proxima.analyzers"
        assert ENTRY_POINT_GROUPS[PluginType.HOOK] == "proxima.hooks"


# =============================================================================
# Plugin Discovery Tests
# =============================================================================


class TestPluginDiscovery:
    """Tests for plugin discovery mechanisms."""

    def test_discover_from_directory(self, tmp_path):
        """Test discovering plugins from directory."""
        from proxima.plugins.loader import PluginLoader, PluginRegistry

        # Create a mock plugin file
        plugin_file = tmp_path / "my_plugin.py"
        plugin_file.write_text("""
from proxima.plugins.base import Plugin, PluginMetadata, PluginType

class MyPlugin(Plugin):
    METADATA = PluginMetadata(
        name="my_plugin",
        version="1.0.0",
        plugin_type=PluginType.EXPORTER,
    )
    
    def initialize(self, context=None):
        pass
    
    def shutdown(self):
        pass
    
    def validate(self):
        return []
""")

        registry = PluginRegistry()
        loader = PluginLoader(registry)
        loader.add_plugin_dir(tmp_path)

        # Directory should be added
        assert tmp_path in loader._plugin_dirs


# =============================================================================
# Hook System Tests
# =============================================================================


class TestHookType:
    """Tests for HookType enum."""

    def test_all_hook_types_exist(self):
        """Test all expected hook types exist."""
        from proxima.plugins.hooks import HookType

        expected = [
            "PRE_EXECUTE",
            "POST_EXECUTE",
            "PRE_COMPARE",
            "POST_COMPARE",
            "ON_ERROR",
            "ON_BACKEND_CHANGE",
        ]

        for hook in expected:
            assert hasattr(HookType, hook)


class TestHookContext:
    """Tests for HookContext."""

    def test_create_hook_context(self):
        """Test creating a hook context."""
        from proxima.plugins.hooks import HookContext, HookType

        context = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            data={"circuit": "test"},
        )

        assert context.hook_type == HookType.PRE_EXECUTE
        assert context.data == {"circuit": "test"}


class TestHookManager:
    """Tests for HookManager."""

    def test_get_hook_manager(self):
        """Test getting global hook manager."""
        from proxima.plugins.hooks import get_hook_manager

        manager = get_hook_manager()
        assert manager is not None

    def test_register_hook(self):
        """Test registering a hook."""
        from proxima.plugins.hooks import HookManager, HookType

        manager = HookManager()
        callback = Mock()

        manager.register(HookType.PRE_EXECUTE, callback)

        # Hook should be registered
        assert HookType.PRE_EXECUTE in manager._hooks

    def test_trigger_hook(self):
        """Test triggering a hook."""
        from proxima.plugins.hooks import HookManager, HookType, HookContext

        manager = HookManager()
        callback = Mock()
        manager.register(HookType.PRE_EXECUTE, callback)

        context = HookContext(hook_type=HookType.PRE_EXECUTE, data={})
        manager.trigger(HookType.PRE_EXECUTE, context)

        callback.assert_called_once_with(context)

    def test_unregister_hook(self):
        """Test unregistering a hook."""
        from proxima.plugins.hooks import HookManager, HookType

        manager = HookManager()
        callback = Mock()

        manager.register(HookType.PRE_EXECUTE, callback)
        manager.unregister(HookType.PRE_EXECUTE, callback)

        # Triggering should not call the callback
        context = Mock()
        manager.trigger(HookType.PRE_EXECUTE, context)
        callback.assert_not_called()


# =============================================================================
# Integration Tests
# =============================================================================


class TestPluginSystemIntegration:
    """Integration tests for the plugin system."""

    def test_full_plugin_workflow(self, tmp_path):
        """Test complete plugin registration and usage workflow."""
        from proxima.plugins.base import Plugin, PluginMetadata, PluginType, PluginContext
        from proxima.plugins.loader import PluginRegistry

        # Create a concrete plugin
        class FullTestPlugin(Plugin):
            METADATA = PluginMetadata(
                name="full_test",
                version="1.0.0",
                plugin_type=PluginType.ANALYZER,
                description="Full test plugin",
            )

            def __init__(self):
                super().__init__()
                self.analyze_count = 0

            def initialize(self, context=None):
                pass

            def shutdown(self):
                pass

            def validate(self):
                return []

            def analyze(self, data):
                self.analyze_count += 1
                return {"analyzed": True, "count": self.analyze_count}

        # Create registry and register plugin
        registry = PluginRegistry()
        plugin = FullTestPlugin()
        registry.register(plugin)

        # Use the plugin
        retrieved = registry.get("full_test")
        result = retrieved.analyze({"test": "data"})

        assert result["analyzed"] is True
        assert result["count"] == 1

        # Unregister
        registry.unregister("full_test")
        assert not registry.is_registered("full_test")

    def test_multiple_plugin_types(self):
        """Test registering multiple plugin types."""
        from proxima.plugins.base import Plugin, PluginMetadata, PluginType
        from proxima.plugins.loader import PluginRegistry

        class ExporterPlugin(Plugin):
            METADATA = PluginMetadata(
                name="exporter1", version="1.0.0", plugin_type=PluginType.EXPORTER
            )

            def initialize(self, context=None):
                pass

            def shutdown(self):
                pass

            def validate(self):
                return []

        class AnalyzerPlugin(Plugin):
            METADATA = PluginMetadata(
                name="analyzer1", version="1.0.0", plugin_type=PluginType.ANALYZER
            )

            def initialize(self, context=None):
                pass

            def shutdown(self):
                pass

            def validate(self):
                return []

        registry = PluginRegistry()
        registry.register(ExporterPlugin())
        registry.register(AnalyzerPlugin())

        exporters = registry.get_by_type(PluginType.EXPORTER)
        analyzers = registry.get_by_type(PluginType.ANALYZER)

        assert len(exporters) == 1
        assert len(analyzers) == 1
        assert "exporter1" in exporters
        assert "analyzer1" in analyzers

        # Cleanup
        registry.clear()
