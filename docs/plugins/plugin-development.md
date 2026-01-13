# Plugin Development Guide

This guide covers how to develop plugins for Proxima, extending its functionality with custom exporters, analyzers, backend integrations, and execution hooks.

## Overview

Proxima's plugin system provides a flexible architecture for extending the framework's capabilities. Plugins can:

- **Export** simulation results to various formats (JSON, CSV, custom formats)
- **Analyze** results with custom metrics and statistical methods
- **Integrate** new quantum backends and simulators
- **Hook** into the execution lifecycle for logging, metrics, and more

## Plugin Architecture

### Core Components

```
src/proxima/plugins/
├── __init__.py          # Package exports
├── base.py              # Base plugin classes
├── hooks.py             # Hook system
├── loader.py            # Plugin discovery and loading
└── examples/            # Example plugin implementations
    ├── __init__.py
    ├── exporters.py
    ├── analyzers.py
    └── hooks.py
```

### Plugin Base Classes

All plugins inherit from base classes defined in `proxima.plugins.base`:

| Base Class | Purpose | Key Methods |
|------------|---------|-------------|
| `Plugin` | Base for all plugins | `initialize()`, `shutdown()` |
| `ExporterPlugin` | Export results | `export()`, `supported_formats` |
| `AnalyzerPlugin` | Analyze results | `analyze()`, `analysis_types` |
| `BackendPlugin` | Integrate backends | `create_backend()`, `get_capabilities()` |
| `LLMProviderPlugin` | LLM integration | `get_provider()`, `get_models()` |

## Creating Your First Plugin

### Step 1: Choose Plugin Type

Determine what type of plugin you need:

```python
from proxima.plugins.base import (
    Plugin,              # Generic plugin
    ExporterPlugin,      # Result export
    AnalyzerPlugin,      # Result analysis
    BackendPlugin,       # Backend integration
    LLMProviderPlugin,   # LLM integration
)
```

### Step 2: Implement Plugin Class

Here's a minimal exporter plugin:

```python
from proxima.plugins.base import ExporterPlugin, PluginContext
from typing import Dict, Any

class YAMLExporterPlugin(ExporterPlugin):
    """Export results to YAML format."""
    
    # Required metadata
    name = "yaml-exporter"
    version = "1.0.0"
    description = "Export simulation results to YAML format"
    
    # Exporter-specific
    supported_formats = ["yaml", "yml"]
    
    def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with context."""
        self._context = context
        # Setup any resources
    
    def export(self, data: Dict[str, Any], **options) -> str:
        """Export data to YAML format."""
        import yaml
        return yaml.dump(data, default_flow_style=False)
    
    def shutdown(self) -> None:
        """Cleanup resources."""
        pass
```

### Step 3: Register Plugin

Register your plugin with the plugin registry:

```python
from proxima.plugins.loader import PluginRegistry

registry = PluginRegistry()
registry.register(YAMLExporterPlugin)
```

## Exporter Plugins

Exporter plugins convert simulation results to various output formats.

### Required Properties

- `name`: Unique plugin identifier
- `version`: Semantic version string
- `description`: Human-readable description
- `supported_formats`: List of format extensions

### Required Methods

```python
def export(self, data: Dict[str, Any], **options) -> str:
    """
    Export data to the target format.
    
    Args:
        data: Dictionary containing simulation results
        **options: Format-specific options
        
    Returns:
        Formatted string output
    """
```

### Example: CSV Exporter

```python
from proxima.plugins.base import ExporterPlugin
import csv
import io

class CSVExporterPlugin(ExporterPlugin):
    name = "csv-exporter"
    version = "1.0.0"
    description = "Export to CSV format"
    supported_formats = ["csv"]
    
    def export(self, data: Dict[str, Any], **options) -> str:
        output = io.StringIO()
        
        if "counts" in data:
            writer = csv.writer(output)
            writer.writerow(["state", "count"])
            for state, count in data["counts"].items():
                writer.writerow([state, count])
        
        return output.getvalue()
```

## Analyzer Plugins

Analyzer plugins perform statistical analysis on simulation results.

### Required Properties

- `name`, `version`, `description`
- `analysis_types`: List of analysis types supported

### Required Methods

```python
def analyze(
    self, 
    data: Dict[str, Any], 
    **options
) -> Dict[str, Any]:
    """
    Analyze simulation results.
    
    Args:
        data: Dictionary containing simulation results
        **options: Analysis-specific options
        
    Returns:
        Dictionary containing analysis results
    """
```

### Example: Entropy Analyzer

```python
from proxima.plugins.base import AnalyzerPlugin
import math

class EntropyAnalyzerPlugin(AnalyzerPlugin):
    name = "entropy-analyzer"
    version = "1.0.0"
    description = "Calculate Shannon entropy of results"
    analysis_types = ["entropy", "information"]
    
    def analyze(self, data: Dict[str, Any], **options) -> Dict[str, Any]:
        counts = data.get("counts", {})
        total = sum(counts.values())
        
        if total == 0:
            return {"entropy": 0.0}
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return {
            "entropy": entropy,
            "max_entropy": math.log2(len(counts)),
            "normalized_entropy": entropy / math.log2(len(counts)) if len(counts) > 1 else 0
        }
```

## Backend Plugins

Backend plugins integrate new quantum simulators or hardware.

### Required Properties

- `name`, `version`, `description`
- `backend_name`: Name used to reference the backend

### Required Methods

```python
def create_backend(self, **config) -> Any:
    """Create and return a backend instance."""

def get_capabilities(self) -> BackendCapabilities:
    """Return backend capabilities."""

def is_available(self) -> bool:
    """Check if backend is available."""
```

### Example: Custom Simulator Backend

```python
from proxima.plugins.base import BackendPlugin, BackendCapabilities

class CustomSimulatorPlugin(BackendPlugin):
    name = "custom-simulator-plugin"
    version = "1.0.0"
    description = "Integration for CustomSim quantum simulator"
    backend_name = "custom_sim"
    
    def create_backend(self, **config):
        from custom_sim import Simulator
        return Simulator(**config)
    
    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=30,
            supports_density_matrix=True,
            supports_statevector=True,
            supports_gpu=False,
            noise_models=["depolarizing", "amplitude_damping"]
        )
    
    def is_available(self) -> bool:
        try:
            import custom_sim
            return True
        except ImportError:
            return False
```

## Hook Plugins

Hook plugins allow you to intercept and react to events in the execution lifecycle.

### Hook Types

The following hook types are available (from `proxima.plugins.hooks.HookType`):

| Hook Type | Triggered When |
|-----------|----------------|
| `BEFORE_EXECUTION` | Before circuit execution |
| `AFTER_EXECUTION` | After circuit execution |
| `BEFORE_OPTIMIZATION` | Before circuit optimization |
| `AFTER_OPTIMIZATION` | After circuit optimization |
| `BEFORE_COMPILATION` | Before circuit compilation |
| `AFTER_COMPILATION` | After circuit compilation |
| `ON_ERROR` | When an error occurs |
| `ON_BACKEND_SWITCH` | When switching backends |
| `BEFORE_COMPARISON` | Before backend comparison |
| `AFTER_COMPARISON` | After backend comparison |

### Required Properties

- `name`, `version`, `description`
- `supported_hooks`: List of `HookType` values

### Example: Metrics Hook

```python
from proxima.plugins.base import Plugin
from proxima.plugins.hooks import HookType, HookContext, HookManager
from collections import defaultdict
import time

class MetricsHookPlugin(Plugin):
    name = "metrics-hook"
    version = "1.0.0"
    description = "Collect execution metrics"
    supported_hooks = [HookType.BEFORE_EXECUTION, HookType.AFTER_EXECUTION]
    
    def __init__(self):
        self._metrics = defaultdict(int)
        self._timings = []
        self._start_time = None
    
    def initialize(self, context, hook_manager: HookManager = None):
        self._context = context
        if hook_manager:
            hook_manager.register(HookType.BEFORE_EXECUTION, self._on_start)
            hook_manager.register(HookType.AFTER_EXECUTION, self._on_end)
    
    def _on_start(self, ctx: HookContext):
        self._start_time = time.perf_counter()
        self._metrics["total_executions"] += 1
    
    def _on_end(self, ctx: HookContext):
        if self._start_time:
            elapsed = time.perf_counter() - self._start_time
            self._timings.append(elapsed)
            self._start_time = None
    
    def get_metrics(self):
        return {
            "total_executions": self._metrics["total_executions"],
            "mean_time": sum(self._timings) / len(self._timings) if self._timings else 0,
        }
```

## Plugin Context

The `PluginContext` provides runtime information to plugins:

```python
@dataclass
class PluginContext:
    backend_name: str              # Current backend
    num_qubits: int               # Number of qubits
    shots: int = 1000             # Number of shots
    config: Dict[str, Any] = None # Additional configuration
    session_id: str = None        # Current session ID
```

## Plugin Discovery

### Automatic Discovery

Plugins can be automatically discovered from installed packages:

```python
from proxima.plugins.loader import discover_plugins

# Discover plugins from entry points
plugins = discover_plugins("proxima.plugins")
```

### Package Entry Points

Register your plugin in `pyproject.toml`:

```toml
[project.entry-points."proxima.plugins"]
my-plugin = "my_package.plugins:MyPlugin"
```

Or in `setup.py`:

```python
setup(
    # ...
    entry_points={
        "proxima.plugins": [
            "my-plugin = my_package.plugins:MyPlugin",
        ],
    },
)
```

## Best Practices

### 1. Proper Initialization

Always implement `initialize()` and `shutdown()`:

```python
def initialize(self, context: PluginContext) -> None:
    self._context = context
    self._resources = self._setup_resources()

def shutdown(self) -> None:
    if self._resources:
        self._resources.close()
        self._resources = None
```

### 2. Error Handling

Handle errors gracefully:

```python
def export(self, data: Dict[str, Any], **options) -> str:
    try:
        return self._do_export(data)
    except Exception as e:
        self._logger.error(f"Export failed: {e}")
        raise ExportError(f"Failed to export: {e}") from e
```

### 3. Logging

Use the standard logging module:

```python
import logging

class MyPlugin(Plugin):
    def __init__(self):
        self._logger = logging.getLogger(f"proxima.plugins.{self.name}")
    
    def initialize(self, context):
        self._logger.info(f"Initializing {self.name} v{self.version}")
```

### 4. Configuration

Support configuration through options:

```python
def export(self, data: Dict[str, Any], **options) -> str:
    indent = options.get("indent", 2)
    include_metadata = options.get("include_metadata", True)
    # Use configuration...
```

### 5. Type Hints

Use type hints for better IDE support and documentation:

```python
from typing import Dict, Any, List, Optional

def analyze(
    self,
    data: Dict[str, Any],
    reference: Optional[Dict[str, int]] = None,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    ...
```

## Testing Plugins

### Unit Testing

```python
import pytest
from my_plugin import MyExporterPlugin
from proxima.plugins.base import PluginContext

class TestMyExporter:
    @pytest.fixture
    def plugin(self):
        plugin = MyExporterPlugin()
        plugin.initialize(PluginContext(backend_name="test", num_qubits=2))
        return plugin
    
    def test_export_valid_data(self, plugin):
        data = {"counts": {"00": 500, "11": 500}}
        output = plugin.export(data)
        assert len(output) > 0
    
    def test_shutdown(self, plugin):
        plugin.shutdown()
        # Verify cleanup
```

### Integration Testing

```python
def test_plugin_with_registry():
    from proxima.plugins.loader import PluginRegistry
    
    registry = PluginRegistry()
    registry.register(MyPlugin)
    
    plugin = registry.get_plugin("my-plugin")
    assert plugin is not None
```

## Distributing Plugins

### Package Structure

```
my-proxima-plugin/
├── pyproject.toml
├── README.md
├── src/
│   └── my_plugin/
│       ├── __init__.py
│       └── plugin.py
└── tests/
    └── test_plugin.py
```

### pyproject.toml

```toml
[project]
name = "proxima-my-plugin"
version = "1.0.0"
description = "My custom Proxima plugin"
dependencies = ["proxima>=0.1.0"]

[project.entry-points."proxima.plugins"]
my-plugin = "my_plugin.plugin:MyPlugin"
```

## Troubleshooting

### Common Issues

**Plugin not found**
- Verify the plugin is registered with the correct name
- Check that entry points are configured correctly

**Import errors**
- Ensure all dependencies are installed
- Check import paths in plugin code

**Initialization failures**
- Verify PluginContext contains required fields
- Check for missing configuration

### Debug Logging

Enable debug logging:

```python
import logging
logging.getLogger("proxima.plugins").setLevel(logging.DEBUG)
```

## API Reference

See [Plugin API Reference](plugin-api-reference.md) for complete API documentation.

## Examples

See [Example Plugins](example-plugins.md) for complete working examples of each plugin type.
