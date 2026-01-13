# Plugin API Reference

Complete API documentation for Proxima's plugin system.

## Base Classes

### Plugin

Base class for all plugins.

```python
class Plugin:
    """Base plugin interface."""
    
    # Required metadata
    name: str                    # Unique plugin identifier
    version: str                 # Semantic version (e.g., "1.0.0")
    description: str             # Human-readable description
    
    def initialize(self, context: PluginContext) -> None:
        """
        Initialize the plugin with execution context.
        
        Args:
            context: PluginContext with runtime information
        
        Called once when the plugin is first used. Use this to set up
        any resources, connections, or state needed by the plugin.
        """
        pass
    
    def shutdown(self) -> None:
        """
        Clean up plugin resources.
        
        Called when the plugin is being unloaded or the application
        is shutting down. Use this to close connections, save state,
        or release resources.
        """
        pass
```

---

### PluginContext

Context provided to plugins during initialization.

```python
@dataclass
class PluginContext:
    """Runtime context for plugin execution."""
    
    backend_name: str
    """Name of the current backend (e.g., 'cirq', 'qiskit_aer')."""
    
    num_qubits: int
    """Number of qubits in the current circuit."""
    
    shots: int = 1000
    """Number of measurement shots."""
    
    config: Dict[str, Any] = None
    """Additional configuration options."""
    
    session_id: Optional[str] = None
    """Current session identifier, if any."""
    
    metadata: Dict[str, Any] = None
    """Additional metadata."""
```

---

### ExporterPlugin

Base class for result exporters.

```python
class ExporterPlugin(Plugin):
    """Base class for result export plugins."""
    
    supported_formats: List[str]
    """List of supported format extensions (e.g., ['json', 'yaml'])."""
    
    @abstractmethod
    def export(
        self,
        data: Dict[str, Any],
        **options
    ) -> str:
        """
        Export data to the target format.
        
        Args:
            data: Dictionary containing simulation results.
                  Typically includes:
                  - counts: Dict[str, int] - measurement counts
                  - execution_time: float - execution duration
                  - backend: str - backend name
                  - metadata: Dict - additional metadata
            **options: Format-specific options (e.g., indent, encoding)
        
        Returns:
            Formatted string output ready for saving or display.
        
        Raises:
            ExportError: If export fails.
        """
        pass
    
    def supports_format(self, format_name: str) -> bool:
        """
        Check if this exporter supports the given format.
        
        Args:
            format_name: Format extension (e.g., 'json', 'csv')
        
        Returns:
            True if format is supported.
        """
        return format_name.lower() in [f.lower() for f in self.supported_formats]
```

---

### AnalyzerPlugin

Base class for result analyzers.

```python
class AnalyzerPlugin(Plugin):
    """Base class for result analysis plugins."""
    
    analysis_types: List[str]
    """List of analysis types this plugin provides."""
    
    @abstractmethod
    def analyze(
        self,
        data: Dict[str, Any],
        **options
    ) -> Dict[str, Any]:
        """
        Analyze simulation results.
        
        Args:
            data: Dictionary containing simulation results.
                  Required fields depend on analysis type.
            **options: Analysis-specific options
        
        Returns:
            Dictionary containing analysis results.
            Structure depends on the analysis type.
        
        Raises:
            AnalysisError: If analysis fails.
        """
        pass
    
    def supports_analysis(self, analysis_type: str) -> bool:
        """
        Check if this analyzer supports the given analysis type.
        
        Args:
            analysis_type: Type of analysis (e.g., 'entropy', 'fidelity')
        
        Returns:
            True if analysis type is supported.
        """
        return analysis_type.lower() in [a.lower() for a in self.analysis_types]
```

---

### BackendPlugin

Base class for backend integrations.

```python
class BackendPlugin(Plugin):
    """Base class for quantum backend plugins."""
    
    backend_name: str
    """Name used to reference this backend."""
    
    @abstractmethod
    def create_backend(self, **config) -> Any:
        """
        Create and return a backend instance.
        
        Args:
            **config: Backend configuration options
        
        Returns:
            Configured backend instance.
        
        Raises:
            BackendError: If backend creation fails.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> "BackendCapabilities":
        """
        Return backend capabilities.
        
        Returns:
            BackendCapabilities dataclass with feature flags.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the backend is available.
        
        Returns:
            True if backend dependencies are installed and accessible.
        """
        pass
    
    def get_version(self) -> str:
        """
        Get backend library version.
        
        Returns:
            Version string of the underlying library.
        """
        return "unknown"


@dataclass
class BackendCapabilities:
    """Backend capability flags."""
    
    max_qubits: int
    """Maximum supported qubit count."""
    
    supports_density_matrix: bool = False
    """Supports density matrix simulation."""
    
    supports_statevector: bool = True
    """Supports statevector simulation."""
    
    supports_gpu: bool = False
    """Supports GPU acceleration."""
    
    supports_distributed: bool = False
    """Supports distributed computation."""
    
    noise_models: List[str] = None
    """List of supported noise model names."""
    
    native_gates: List[str] = None
    """List of native gate names."""
```

---

### LLMProviderPlugin

Base class for LLM provider integrations.

```python
class LLMProviderPlugin(Plugin):
    """Base class for LLM provider plugins."""
    
    provider_name: str
    """Name of the LLM provider."""
    
    @abstractmethod
    def get_provider(self, **config) -> Any:
        """
        Get configured LLM provider instance.
        
        Args:
            **config: Provider configuration (API keys, endpoints, etc.)
        
        Returns:
            Configured provider instance.
        """
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model identifiers.
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if provider is available.
        
        Returns:
            True if provider is configured and accessible.
        """
        return True
```

---

## Hook System

### HookType

Enumeration of available hook points.

```python
class HookType(Enum):
    """Execution lifecycle hook points."""
    
    # Circuit execution hooks
    BEFORE_EXECUTION = "before_execution"
    """Called before circuit execution starts."""
    
    AFTER_EXECUTION = "after_execution"
    """Called after circuit execution completes."""
    
    # Circuit transformation hooks
    BEFORE_OPTIMIZATION = "before_optimization"
    """Called before circuit optimization."""
    
    AFTER_OPTIMIZATION = "after_optimization"
    """Called after circuit optimization."""
    
    BEFORE_COMPILATION = "before_compilation"
    """Called before circuit compilation to native gates."""
    
    AFTER_COMPILATION = "after_compilation"
    """Called after circuit compilation."""
    
    BEFORE_TRANSPILATION = "before_transpilation"
    """Called before circuit transpilation."""
    
    AFTER_TRANSPILATION = "after_transpilation"
    """Called after circuit transpilation."""
    
    # Backend hooks
    ON_BACKEND_SWITCH = "on_backend_switch"
    """Called when switching between backends."""
    
    BEFORE_BACKEND_INIT = "before_backend_init"
    """Called before backend initialization."""
    
    AFTER_BACKEND_INIT = "after_backend_init"
    """Called after backend initialization."""
    
    # Comparison hooks
    BEFORE_COMPARISON = "before_comparison"
    """Called before multi-backend comparison."""
    
    AFTER_COMPARISON = "after_comparison"
    """Called after multi-backend comparison."""
    
    # Error handling
    ON_ERROR = "on_error"
    """Called when an error occurs."""
    
    ON_WARNING = "on_warning"
    """Called for non-fatal warnings."""
    
    # Session hooks
    ON_SESSION_START = "on_session_start"
    """Called when a session starts."""
    
    ON_SESSION_END = "on_session_end"
    """Called when a session ends."""
    
    # Result hooks
    ON_RESULT_READY = "on_result_ready"
    """Called when execution results are ready."""
    
    BEFORE_RESULT_EXPORT = "before_result_export"
    """Called before exporting results."""
    
    AFTER_RESULT_EXPORT = "after_result_export"
    """Called after exporting results."""
```

---

### HookContext

Context passed to hook callbacks.

```python
@dataclass
class HookContext:
    """Context information for hook execution."""
    
    hook_type: HookType
    """Type of hook being executed."""
    
    data: Dict[str, Any]
    """Hook-specific data. Contents vary by hook type:
    
    BEFORE_EXECUTION / AFTER_EXECUTION:
        - circuit: str or Circuit object
        - backend: str - backend name
        - shots: int - number of shots
        - execution_id: str - unique execution ID
        - result: Dict (AFTER only) - execution result
    
    BEFORE_OPTIMIZATION / AFTER_OPTIMIZATION:
        - circuit: str or Circuit object
        - optimization_level: int
        - optimized_circuit: Circuit (AFTER only)
    
    ON_ERROR:
        - error: Exception object
        - error_type: str - exception class name
        - traceback: str - formatted traceback
        - context: Dict - additional context
    
    ON_BACKEND_SWITCH:
        - from_backend: str - previous backend
        - to_backend: str - new backend
        - reason: str - switch reason
    """
    
    timestamp: datetime = None
    """When the hook was triggered."""
    
    session_id: Optional[str] = None
    """Current session ID, if any."""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
```

---

### HookManager

Manages hook registration and execution.

```python
class HookManager:
    """Manages execution hooks."""
    
    def register(
        self,
        hook_type: HookType,
        callback: Callable[[HookContext], None],
        priority: int = 0
    ) -> None:
        """
        Register a callback for a hook type.
        
        Args:
            hook_type: Type of hook to register for.
            callback: Function to call when hook triggers.
                      Signature: (context: HookContext) -> None
            priority: Execution priority (higher = earlier).
                      Default is 0.
        
        Callbacks with equal priority execute in registration order.
        """
        pass
    
    def unregister(
        self,
        hook_type: HookType,
        callback: Callable[[HookContext], None]
    ) -> bool:
        """
        Unregister a callback.
        
        Args:
            hook_type: Hook type the callback is registered for.
            callback: The callback function to remove.
        
        Returns:
            True if callback was found and removed.
        """
        pass
    
    def execute(
        self,
        hook_type: HookType,
        context: HookContext,
        continue_on_error: bool = True
    ) -> List[Any]:
        """
        Execute all callbacks for a hook type.
        
        Args:
            hook_type: Type of hook to execute.
            context: Context to pass to callbacks.
            continue_on_error: If True, continue executing callbacks
                              after one raises an exception.
        
        Returns:
            List of return values from callbacks.
        
        Raises:
            HookExecutionError: If a callback fails and
                               continue_on_error is False.
        """
        pass
    
    def get_hooks(self, hook_type: HookType) -> List[Callable]:
        """
        Get all callbacks registered for a hook type.
        
        Args:
            hook_type: Hook type to query.
        
        Returns:
            List of registered callback functions.
        """
        pass
    
    def clear(self, hook_type: Optional[HookType] = None) -> None:
        """
        Clear registered hooks.
        
        Args:
            hook_type: Specific hook type to clear.
                      If None, clears all hooks.
        """
        pass
```

---

## Plugin Loader

### PluginRegistry

Central registry for plugins.

```python
class PluginRegistry:
    """Central registry for plugin management."""
    
    def register(
        self,
        plugin_class: Type[Plugin],
        override: bool = False
    ) -> None:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Plugin class (not instance).
            override: If True, replace existing plugin with same name.
        
        Raises:
            ValueError: If plugin with same name exists and
                       override is False.
        """
        pass
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin by name.
        
        Args:
            name: Plugin name to unregister.
        
        Returns:
            True if plugin was found and removed.
        """
        pass
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a plugin instance by name.
        
        Args:
            name: Plugin name.
        
        Returns:
            Plugin instance or None if not found.
        """
        pass
    
    def list_plugins(self) -> List[Plugin]:
        """
        List all registered plugins.
        
        Returns:
            List of plugin instances.
        """
        pass
    
    def get_plugins_by_type(
        self,
        plugin_type: str
    ) -> List[Plugin]:
        """
        Get plugins by type.
        
        Args:
            plugin_type: One of 'exporter', 'analyzer', 'backend',
                        'llm', 'hook', or 'generic'.
        
        Returns:
            List of plugins matching the type.
        """
        pass
    
    def clear(self) -> None:
        """Remove all registered plugins."""
        pass
```

---

### PluginLoader

Discovers and loads plugins.

```python
class PluginLoader:
    """Discovers and loads plugins from various sources."""
    
    def load_from_module(
        self,
        module_path: str,
        registry: PluginRegistry
    ) -> int:
        """
        Load plugins from a Python module.
        
        Args:
            module_path: Dotted module path (e.g., 'my_package.plugins')
            registry: Registry to register plugins in.
        
        Returns:
            Number of plugins loaded.
        """
        pass
    
    def load_from_directory(
        self,
        directory: str,
        registry: PluginRegistry
    ) -> int:
        """
        Load plugins from a directory.
        
        Args:
            directory: Path to directory containing plugin files.
            registry: Registry to register plugins in.
        
        Returns:
            Number of plugins loaded.
        """
        pass
    
    def load_from_entry_points(
        self,
        group: str,
        registry: PluginRegistry
    ) -> int:
        """
        Load plugins from package entry points.
        
        Args:
            group: Entry point group name (e.g., 'proxima.plugins')
            registry: Registry to register plugins in.
        
        Returns:
            Number of plugins loaded.
        """
        pass


def discover_plugins(group: str = "proxima.plugins") -> List[Type[Plugin]]:
    """
    Discover plugins from entry points.
    
    Args:
        group: Entry point group to search.
    
    Returns:
        List of discovered plugin classes.
    """
    pass
```

---

## Exceptions

### PluginError

Base exception for plugin errors.

```python
class PluginError(Exception):
    """Base exception for plugin-related errors."""
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        self.plugin_name = plugin_name
        self.cause = cause
        super().__init__(message)


class PluginInitializationError(PluginError):
    """Plugin failed to initialize."""
    pass


class PluginNotFoundError(PluginError):
    """Requested plugin was not found."""
    pass


class ExportError(PluginError):
    """Export operation failed."""
    pass


class AnalysisError(PluginError):
    """Analysis operation failed."""
    pass


class HookExecutionError(PluginError):
    """Hook callback execution failed."""
    
    def __init__(
        self,
        message: str,
        hook_type: HookType,
        callback: Callable,
        cause: Exception
    ):
        self.hook_type = hook_type
        self.callback = callback
        super().__init__(message, cause=cause)
```

---

## Type Definitions

```python
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

# Generic plugin type
P = TypeVar("P", bound=Plugin)

# Hook callback signature
HookCallback = Callable[[HookContext], Optional[Any]]

# Result data structure
ResultData = Dict[str, Any]
"""Standard structure:
{
    "counts": Dict[str, int],
    "execution_time": float,
    "backend": str,
    "shots": int,
    "metadata": Dict[str, Any],
}
"""

# Analysis result structure
AnalysisResult = Dict[str, Any]
"""Structure varies by analysis type."""

# Plugin configuration
PluginConfig = Dict[str, Any]
"""Plugin-specific configuration dictionary."""
```

---

## See Also

- [Plugin Development Guide](plugin-development.md) - How to create custom plugins
- [Example Plugins](example-plugins.md) - Working plugin examples
