"""
Unit tests for the plugin system.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from proxima.plugins.base import (
    AnalyzerPlugin,
    BackendPlugin,
    ExporterPlugin,
    LLMProviderPlugin,
    Plugin,
    PluginError,
    PluginLoadError,
    PluginMetadata,
    PluginType,
)
from proxima.plugins.hooks import (
    HookContext,
    HookManager,
    HookType,
    get_hook_manager,
    hook,
)
from proxima.plugins.loader import (
    PluginLoader,
    PluginRegistry,
    get_plugin_registry,
)

# ===================== Test Fixtures =====================


class MockPlugin(Plugin):
    """Mock plugin for testing."""

    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="mock-plugin",
        version="1.0.0",
        plugin_type=PluginType.HOOK,
        description="A mock plugin for testing",
        author="Test Author",
    )

    initialized: bool = False
    shutdown_called: bool = False

    def initialize(self) -> None:
        self.initialized = True

    def shutdown(self) -> None:
        self.shutdown_called = True


class MockBackendPlugin(BackendPlugin):
    """Mock backend plugin for testing."""

    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="mock-backend",
        version="1.0.0",
        plugin_type=PluginType.BACKEND,
        description="A mock backend plugin",
    )

    def get_backend_class(self) -> type[Any]:
        return object

    def get_backend_name(self) -> str:
        return "mock"


class MockLLMPlugin(LLMProviderPlugin):
    """Mock LLM provider plugin for testing."""

    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="mock-llm",
        version="1.0.0",
        plugin_type=PluginType.LLM_PROVIDER,
        description="A mock LLM plugin",
    )

    def get_provider_class(self) -> type[Any]:
        return object

    def get_provider_name(self) -> str:
        return "mock-llm"

    def get_api_key_env_var(self) -> str | None:
        return "MOCK_LLM_API_KEY"


class MockExporterPlugin(ExporterPlugin):
    """Mock exporter plugin for testing."""

    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="mock-exporter",
        version="1.0.0",
        plugin_type=PluginType.EXPORTER,
        description="A mock exporter plugin",
    )

    def get_format_name(self) -> str:
        return "mock"

    def export(self, data: Any, destination: str) -> None:
        pass


class MockAnalyzerPlugin(AnalyzerPlugin):
    """Mock analyzer plugin for testing."""

    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="mock-analyzer",
        version="1.0.0",
        plugin_type=PluginType.ANALYZER,
        description="A mock analyzer plugin",
    )

    def get_analyzer_name(self) -> str:
        return "mock-analyzer"

    def analyze(self, results: Any) -> dict[str, Any]:
        return {"analyzed": True}


class FailingPlugin(Plugin):
    """Plugin that fails on initialization."""

    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="failing-plugin",
        version="1.0.0",
        plugin_type=PluginType.HOOK,
    )

    def initialize(self) -> None:
        raise RuntimeError("Initialization failed")

    def shutdown(self) -> None:
        pass


@pytest.fixture
def registry() -> PluginRegistry:
    """Create a fresh plugin registry."""
    return PluginRegistry()


@pytest.fixture
def hook_manager() -> HookManager:
    """Create a fresh hook manager."""
    return HookManager()


# ===================== Plugin Base Tests =====================


class TestPluginMetadata:
    """Tests for PluginMetadata."""

    def test_metadata_creation(self) -> None:
        """Test creating plugin metadata."""
        meta = PluginMetadata(
            name="test",
            version="1.0.0",
            plugin_type=PluginType.BACKEND,
            description="Test plugin",
        )
        assert meta.name == "test"
        assert meta.version == "1.0.0"
        assert meta.plugin_type == PluginType.BACKEND

    def test_metadata_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        meta = PluginMetadata(
            name="test",
            version="1.0.0",
            plugin_type=PluginType.BACKEND,
        )
        data = meta.to_dict()
        assert data["name"] == "test"
        assert data["plugin_type"] == "backend"

    def test_metadata_from_dict(self) -> None:
        """Test creating metadata from dictionary."""
        data = {
            "name": "test",
            "version": "2.0.0",
            "plugin_type": "exporter",
        }
        meta = PluginMetadata.from_dict(data)
        assert meta.name == "test"
        assert meta.version == "2.0.0"
        assert meta.plugin_type == PluginType.EXPORTER


class TestPlugin:
    """Tests for Plugin base class."""

    def test_plugin_properties(self) -> None:
        """Test plugin properties."""
        plugin = MockPlugin()
        assert plugin.name == "mock-plugin"
        assert plugin.version == "1.0.0"
        assert plugin.plugin_type == PluginType.HOOK

    def test_plugin_enable_disable(self) -> None:
        """Test enabling and disabling plugins."""
        plugin = MockPlugin()
        assert plugin.enabled is True

        plugin.disable()
        assert plugin.enabled is False

        plugin.enable()
        assert plugin.enabled is True

    def test_plugin_configuration(self) -> None:
        """Test plugin configuration."""
        plugin = MockPlugin(config={"key": "value"})
        assert plugin.get_config("key") == "value"
        assert plugin.get_config("missing", "default") == "default"

        plugin.configure({"key2": "value2"})
        assert plugin.get_config("key2") == "value2"

    def test_plugin_validation(self) -> None:
        """Test plugin validation."""
        plugin = MockPlugin()
        errors = plugin.validate()
        assert len(errors) == 0


class TestBackendPlugin:
    """Tests for BackendPlugin."""

    def test_backend_plugin(self) -> None:
        """Test backend plugin interface."""
        plugin = MockBackendPlugin()
        assert plugin.get_backend_name() == "mock"
        assert plugin.get_backend_class() is object


class TestLLMProviderPlugin:
    """Tests for LLMProviderPlugin."""

    def test_llm_plugin(self) -> None:
        """Test LLM provider plugin interface."""
        plugin = MockLLMPlugin()
        assert plugin.get_provider_name() == "mock-llm"
        assert plugin.get_api_key_env_var() == "MOCK_LLM_API_KEY"


class TestExporterPlugin:
    """Tests for ExporterPlugin."""

    def test_exporter_plugin(self) -> None:
        """Test exporter plugin interface."""
        plugin = MockExporterPlugin()
        assert plugin.get_format_name() == "mock"
        plugin.export({}, "/tmp/test")  # Should not raise


class TestAnalyzerPlugin:
    """Tests for AnalyzerPlugin."""

    def test_analyzer_plugin(self) -> None:
        """Test analyzer plugin interface."""
        plugin = MockAnalyzerPlugin()
        assert plugin.get_analyzer_name() == "mock-analyzer"
        result = plugin.analyze({})
        assert result == {"analyzed": True}


# ===================== Plugin Registry Tests =====================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_register_plugin(self, registry: PluginRegistry) -> None:
        """Test registering a plugin."""
        plugin = MockPlugin()
        registry.register(plugin)

        assert registry.is_registered("mock-plugin")
        assert plugin.initialized is True

    def test_unregister_plugin(self, registry: PluginRegistry) -> None:
        """Test unregistering a plugin."""
        plugin = MockPlugin()
        registry.register(plugin)

        result = registry.unregister("mock-plugin")
        assert result is True
        assert registry.is_registered("mock-plugin") is False
        assert plugin.shutdown_called is True

    def test_unregister_nonexistent(self, registry: PluginRegistry) -> None:
        """Test unregistering a non-existent plugin."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_plugin(self, registry: PluginRegistry) -> None:
        """Test getting a plugin by name."""
        plugin = MockPlugin()
        registry.register(plugin)

        retrieved = registry.get("mock-plugin")
        assert retrieved is plugin

    def test_get_nonexistent(self, registry: PluginRegistry) -> None:
        """Test getting a non-existent plugin."""
        retrieved = registry.get("nonexistent")
        assert retrieved is None

    def test_get_by_type(self, registry: PluginRegistry) -> None:
        """Test getting plugins by type."""
        plugin = MockPlugin()
        registry.register(plugin)

        hooks = registry.get_by_type(PluginType.HOOK)
        assert "mock-plugin" in hooks

        backends = registry.get_by_type(PluginType.BACKEND)
        assert len(backends) == 0

    def test_list_all(self, registry: PluginRegistry) -> None:
        """Test listing all plugins."""
        plugin1 = MockPlugin()
        registry.register(plugin1)

        plugins = registry.list_all()
        assert len(plugins) == 1
        assert plugin1 in plugins

    def test_duplicate_registration(self, registry: PluginRegistry) -> None:
        """Test that duplicate registration raises an error."""
        plugin = MockPlugin()
        registry.register(plugin)

        with pytest.raises(PluginError, match="already registered"):
            registry.register(MockPlugin())

    def test_failing_plugin_rollback(self, registry: PluginRegistry) -> None:
        """Test that failed initialization rolls back registration."""
        plugin = FailingPlugin()

        with pytest.raises(PluginLoadError, match="initialization failed"):
            registry.register(plugin)

        assert registry.is_registered("failing-plugin") is False

    def test_load_callbacks(self, registry: PluginRegistry) -> None:
        """Test load callbacks are called."""
        loaded_plugins: list[Plugin] = []
        registry.on_load(lambda p: loaded_plugins.append(p))

        plugin = MockPlugin()
        registry.register(plugin)

        assert plugin in loaded_plugins

    def test_unload_callbacks(self, registry: PluginRegistry) -> None:
        """Test unload callbacks are called."""
        unloaded_plugins: list[Plugin] = []
        registry.on_unload(lambda p: unloaded_plugins.append(p))

        plugin = MockPlugin()
        registry.register(plugin)
        registry.unregister("mock-plugin")

        assert plugin in unloaded_plugins

    def test_clear(self, registry: PluginRegistry) -> None:
        """Test clearing all plugins."""
        plugin = MockPlugin()
        registry.register(plugin)

        registry.clear()
        assert len(registry.list_all()) == 0
        assert plugin.shutdown_called is True


# ===================== Plugin Loader Tests =====================


class TestPluginLoader:
    """Tests for PluginLoader."""

    def test_load_from_class(self, registry: PluginRegistry) -> None:
        """Test loading a plugin from a class."""
        loader = PluginLoader(registry)
        plugin = loader.load_from_class(MockPlugin)

        assert registry.is_registered("mock-plugin")
        assert plugin.initialized is True

    def test_load_with_config(self, registry: PluginRegistry) -> None:
        """Test loading a plugin with configuration."""
        loader = PluginLoader(registry)
        plugin = loader.load_from_class(MockPlugin, config={"key": "value"})

        assert plugin.get_config("key") == "value"


# ===================== Hook Manager Tests =====================


class TestHookContext:
    """Tests for HookContext."""

    def test_context_creation(self) -> None:
        """Test creating a hook context."""
        ctx = HookContext(hook_type=HookType.PRE_EXECUTE)
        assert ctx.hook_type == HookType.PRE_EXECUTE
        assert ctx.cancelled is False
        assert ctx.result is None

    def test_context_cancel(self) -> None:
        """Test cancelling a context."""
        ctx = HookContext(hook_type=HookType.PRE_EXECUTE)
        ctx.cancel()
        assert ctx.cancelled is True

    def test_context_data(self) -> None:
        """Test context data access."""
        ctx = HookContext(
            hook_type=HookType.PRE_EXECUTE,
            data={"key": "value"},
        )
        assert ctx.get("key") == "value"
        assert ctx.get("missing", "default") == "default"

        ctx.set("new_key", "new_value")
        assert ctx.get("new_key") == "new_value"


class TestHookManager:
    """Tests for HookManager."""

    def test_register_handler(self, hook_manager: HookManager) -> None:
        """Test registering a hook handler."""

        def handler(ctx: HookContext) -> None:
            pass

        hook_manager.register(HookType.PRE_EXECUTE, handler)
        assert hook_manager.has_handlers(HookType.PRE_EXECUTE)

    def test_unregister_by_handler(self, hook_manager: HookManager) -> None:
        """Test unregistering by handler function."""

        def handler(ctx: HookContext) -> None:
            pass

        hook_manager.register(HookType.PRE_EXECUTE, handler)
        removed = hook_manager.unregister(HookType.PRE_EXECUTE, handler=handler)

        assert removed == 1
        assert not hook_manager.has_handlers(HookType.PRE_EXECUTE)

    def test_unregister_by_name(self, hook_manager: HookManager) -> None:
        """Test unregistering by name."""
        hook_manager.register(
            HookType.PRE_EXECUTE,
            lambda ctx: None,
            name="my-handler",
        )
        removed = hook_manager.unregister(HookType.PRE_EXECUTE, name="my-handler")

        assert removed == 1

    def test_unregister_by_plugin(self, hook_manager: HookManager) -> None:
        """Test unregistering by plugin name."""
        hook_manager.register(
            HookType.PRE_EXECUTE,
            lambda ctx: None,
            plugin_name="my-plugin",
        )
        hook_manager.register(
            HookType.POST_EXECUTE,
            lambda ctx: None,
            plugin_name="my-plugin",
        )

        removed = hook_manager.unregister_plugin("my-plugin")
        assert removed == 2

    def test_trigger_hook(self, hook_manager: HookManager) -> None:
        """Test triggering a hook."""
        results: list[str] = []

        def handler(ctx: HookContext) -> None:
            results.append("executed")

        hook_manager.register(HookType.PRE_EXECUTE, handler)
        ctx = hook_manager.trigger(HookType.PRE_EXECUTE)

        assert len(results) == 1
        assert not ctx.cancelled

    def test_trigger_with_data(self, hook_manager: HookManager) -> None:
        """Test triggering with initial data."""

        def handler(ctx: HookContext) -> None:
            assert ctx.get("input") == "test"
            ctx.set("output", "result")

        hook_manager.register(HookType.PRE_EXECUTE, handler)
        ctx = hook_manager.trigger(HookType.PRE_EXECUTE, data={"input": "test"})

        assert ctx.get("output") == "result"

    def test_trigger_priority_order(self, hook_manager: HookManager) -> None:
        """Test that handlers run in priority order."""
        order: list[int] = []

        hook_manager.register(
            HookType.PRE_EXECUTE,
            lambda ctx: order.append(1),
            priority=1,
        )
        hook_manager.register(
            HookType.PRE_EXECUTE,
            lambda ctx: order.append(10),
            priority=10,
        )
        hook_manager.register(
            HookType.PRE_EXECUTE,
            lambda ctx: order.append(5),
            priority=5,
        )

        hook_manager.trigger(HookType.PRE_EXECUTE)
        assert order == [10, 5, 1]

    def test_trigger_cancellation(self, hook_manager: HookManager) -> None:
        """Test that cancellation stops subsequent handlers."""
        executed: list[str] = []

        def handler1(ctx: HookContext) -> None:
            executed.append("first")
            ctx.cancel()

        def handler2(ctx: HookContext) -> None:
            executed.append("second")

        hook_manager.register(HookType.PRE_EXECUTE, handler1, priority=10)
        hook_manager.register(HookType.PRE_EXECUTE, handler2, priority=1)

        ctx = hook_manager.trigger(HookType.PRE_EXECUTE)

        assert executed == ["first"]
        assert ctx.cancelled is True

    def test_trigger_error_handling(self, hook_manager: HookManager) -> None:
        """Test that errors are captured but don't stop execution."""
        executed: list[str] = []

        def failing_handler(ctx: HookContext) -> None:
            raise ValueError("Handler error")

        def working_handler(ctx: HookContext) -> None:
            executed.append("worked")

        hook_manager.register(HookType.PRE_EXECUTE, failing_handler, priority=10)
        hook_manager.register(HookType.PRE_EXECUTE, working_handler, priority=1)

        ctx = hook_manager.trigger(HookType.PRE_EXECUTE)

        assert "worked" in executed
        assert len(ctx.data.get("errors", [])) == 1

    def test_clear_hooks(self, hook_manager: HookManager) -> None:
        """Test clearing hooks."""
        hook_manager.register(HookType.PRE_EXECUTE, lambda ctx: None)
        hook_manager.register(HookType.POST_EXECUTE, lambda ctx: None)

        hook_manager.clear(HookType.PRE_EXECUTE)
        assert not hook_manager.has_handlers(HookType.PRE_EXECUTE)
        assert hook_manager.has_handlers(HookType.POST_EXECUTE)

        hook_manager.clear()
        assert not hook_manager.has_handlers(HookType.POST_EXECUTE)


class TestHookDecorator:
    """Tests for @hook decorator."""

    def test_hook_decorator(self) -> None:
        """Test the hook decorator."""
        # Get a fresh manager for this test
        manager = get_hook_manager()
        initial_count = len(manager.get_handlers(HookType.PRE_INIT))

        @hook(HookType.PRE_INIT, priority=5, name="test-hook")
        def my_hook(ctx: HookContext) -> None:
            ctx.set("decorated", True)

        handlers = manager.get_handlers(HookType.PRE_INIT)
        assert len(handlers) > initial_count

        # Clean up
        manager.unregister(HookType.PRE_INIT, name="test-hook")


# ===================== Global Functions Tests =====================


class TestGlobalFunctions:
    """Tests for global plugin functions."""

    def test_get_plugin_registry(self) -> None:
        """Test getting the global registry."""
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()
        assert registry1 is registry2

    def test_get_hook_manager(self) -> None:
        """Test getting the global hook manager."""
        manager1 = get_hook_manager()
        manager2 = get_hook_manager()
        assert manager1 is manager2
