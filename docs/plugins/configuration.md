# Plugin Configuration Guide

Complete guide for configuring and customizing Proxima plugins.

## Overview

Plugins in Proxima can be configured through:

- **YAML Configuration Files** - Global and per-plugin settings
- **Environment Variables** - Runtime overrides
- **Programmatic Configuration** - Code-based setup
- **Plugin-Specific Config Files** - Individual plugin configuration

---

## Configuration File Structure

### Main Configuration (proxima.yaml)

```yaml
# Plugin configuration section
plugins:
  # Global plugin settings
  enabled: true
  auto_discover: true
  plugin_paths:
    - ~/.proxima/plugins
    - ./plugins
  
  # Per-plugin configuration
  exporters:
    json_exporter:
      enabled: true
      config:
        indent: 2
        include_metadata: true
    
    csv_exporter:
      enabled: true
      config:
        delimiter: ","
        quote_char: '"'
    
    markdown_exporter:
      enabled: true
      config:
        template: default
        include_charts: true
  
  analyzers:
    statistical_analyzer:
      enabled: true
      config:
        confidence_level: 0.95
        bootstrap_samples: 1000
    
    fidelity_analyzer:
      enabled: true
      config:
        epsilon: 1e-10
    
    performance_analyzer:
      enabled: true
      config:
        timing_threshold_ms: 100
  
  hooks:
    logging_hook:
      enabled: true
      config:
        log_level: INFO
        log_file: ~/.proxima/logs/hooks.log
    
    metrics_hook:
      enabled: true
      config:
        collect_timing: true
        collect_memory: true
        export_prometheus: false
```

---

## Plugin Discovery

### Automatic Discovery

Proxima automatically discovers plugins from:

1. **Entry Points** - Installed Python packages
2. **Plugin Directories** - Configured paths
3. **Built-in Plugins** - Included with Proxima

```yaml
plugins:
  auto_discover: true
  
  # Directories to scan for plugins
  plugin_paths:
    - ~/.proxima/plugins
    - ./plugins
    - /opt/proxima/plugins
  
  # Entry point groups to load
  entry_points:
    - proxima.exporters
    - proxima.analyzers
    - proxima.hooks
    - proxima.backends
```

### Manual Registration

Disable auto-discovery and specify plugins explicitly:

```yaml
plugins:
  auto_discover: false
  
  # Explicit plugin list
  load:
    - proxima.plugins.examples.exporters:JSONExporterPlugin
    - proxima.plugins.examples.analyzers:StatisticalAnalyzerPlugin
    - my_package.plugins:CustomPlugin
```

---

## Per-Plugin Configuration

### Exporter Plugins

#### JSONExporterPlugin

```yaml
plugins:
  exporters:
    json_exporter:
      enabled: true
      config:
        # Indentation for pretty printing
        indent: 2
        
        # Include export metadata
        include_metadata: true
        
        # Sort keys alphabetically
        sort_keys: false
        
        # Ensure ASCII output
        ensure_ascii: false
        
        # Custom date format
        date_format: "%Y-%m-%dT%H:%M:%SZ"
```

#### CSVExporterPlugin

```yaml
plugins:
  exporters:
    csv_exporter:
      enabled: true
      config:
        # Field delimiter
        delimiter: ","
        
        # Quote character
        quote_char: '"'
        
        # Quoting mode: minimal, all, nonnumeric, none
        quoting: minimal
        
        # Include header row
        include_header: true
        
        # Flatten nested dictionaries
        flatten_nested: true
        
        # Separator for flattened keys
        flatten_separator: "."
```

#### MarkdownExporterPlugin

```yaml
plugins:
  exporters:
    markdown_exporter:
      enabled: true
      config:
        # Template style: default, minimal, detailed
        template: default
        
        # Include ASCII charts
        include_charts: true
        
        # Include metadata section
        include_metadata: true
        
        # Table style: github, simple, grid
        table_style: github
        
        # Maximum table width
        max_table_width: 120
```

### Analyzer Plugins

#### StatisticalAnalyzerPlugin

```yaml
plugins:
  analyzers:
    statistical_analyzer:
      enabled: true
      config:
        # Confidence level for intervals
        confidence_level: 0.95
        
        # Number of bootstrap samples
        bootstrap_samples: 1000
        
        # Random seed for reproducibility
        seed: null
        
        # Metrics to calculate
        metrics:
          - entropy
          - uniformity
          - confidence_intervals
          - mean
          - std
          - median
```

#### FidelityAnalyzerPlugin

```yaml
plugins:
  analyzers:
    fidelity_analyzer:
      enabled: true
      config:
        # Epsilon for numerical stability
        epsilon: 1e-10
        
        # Fidelity type: classical, quantum
        fidelity_type: classical
        
        # Distance metrics
        metrics:
          - fidelity
          - tvd
          - hellinger
          - kl_divergence
          - js_divergence
```

#### PerformanceAnalyzerPlugin

```yaml
plugins:
  analyzers:
    performance_analyzer:
      enabled: true
      config:
        # Timing threshold for warnings (ms)
        timing_threshold_ms: 100
        
        # Memory threshold for warnings (MB)
        memory_threshold_mb: 512
        
        # Generate recommendations
        generate_recommendations: true
        
        # Compare to baseline
        baseline_file: null
```

### Hook Plugins

#### LoggingHookPlugin

```yaml
plugins:
  hooks:
    logging_hook:
      enabled: true
      config:
        # Minimum log level
        log_level: INFO
        
        # Log file path
        log_file: ~/.proxima/logs/hooks.log
        
        # Log format
        format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Events to log
        events:
          - pre_execute
          - post_execute
          - on_error
        
        # Include data in logs
        include_data: false
        
        # Max data length to log
        max_data_length: 1000
```

#### MetricsHookPlugin

```yaml
plugins:
  hooks:
    metrics_hook:
      enabled: true
      config:
        # Collect execution timing
        collect_timing: true
        
        # Collect memory usage
        collect_memory: true
        
        # Collect backend statistics
        collect_backend_stats: true
        
        # Aggregation window (seconds)
        aggregation_window: 60
        
        # Export to Prometheus
        export_prometheus: false
        prometheus_port: 9090
        
        # Export to file
        export_file: ~/.proxima/metrics.json
        export_interval: 300
```

---

## Environment Variables

Override configuration with environment variables:

```bash
# Global plugin settings
export PROXIMA_PLUGINS_ENABLED=true
export PROXIMA_PLUGINS_AUTO_DISCOVER=true
export PROXIMA_PLUGINS_PATH="~/.proxima/plugins:./plugins"

# Per-plugin configuration
export PROXIMA_PLUGIN_JSON_EXPORTER_INDENT=4
export PROXIMA_PLUGIN_LOGGING_HOOK_LOG_LEVEL=DEBUG
export PROXIMA_PLUGIN_METRICS_HOOK_PROMETHEUS_PORT=9091
```

### Environment Variable Format

```
PROXIMA_PLUGIN_<PLUGIN_NAME>_<CONFIG_KEY>=<VALUE>
```

- Plugin name: UPPER_SNAKE_CASE
- Config key: UPPER_SNAKE_CASE
- Nested keys: Separated by double underscore

```bash
# Nested configuration
export PROXIMA_PLUGIN_STATISTICAL_ANALYZER__METRICS__ENTROPY=true
```

---

## Programmatic Configuration

### Using Python Code

```python
from proxima.plugins.loader import PluginRegistry
from proxima.plugins.base import PluginContext

# Get global registry
registry = PluginRegistry()

# Configure a specific plugin
json_exporter = registry.get("json_exporter")
if json_exporter:
    json_exporter.configure({
        "indent": 4,
        "include_metadata": True,
        "sort_keys": True,
    })

# Initialize with context
context = PluginContext(
    backend_name="cirq",
    num_qubits=4,
    shots=1000,
    config={"precision": "double"},
)

json_exporter.initialize(context)
```

### Plugin Factory Pattern

```python
from proxima.plugins.examples.exporters import JSONExporterPlugin

# Create with configuration
plugin = JSONExporterPlugin(config={
    "indent": 2,
    "include_metadata": True,
})

# Or use factory function
from proxima.plugins.loader import create_plugin

plugin = create_plugin(
    "json_exporter",
    config={"indent": 4},
)
```

---

## Configuration Validation

### Schema Validation

Plugins can define configuration schemas for validation:

```python
from proxima.plugins.base import Plugin, PluginMetadata, PluginType

class MyPlugin(Plugin):
    METADATA = PluginMetadata(
        name="my_plugin",
        version="1.0.0",
        plugin_type=PluginType.EXPORTER,
        config_schema={
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3600,
                    "default": 30,
                },
                "retries": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 3,
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "required": [],
        },
    )
```

### Validation Commands

```bash
# Validate plugin configuration
proxima plugin validate my_plugin

# Validate all plugins
proxima plugin validate --all

# Show configuration schema
proxima plugin schema my_plugin
```

---

## Configuration Inheritance

### Default Configuration

Plugins have built-in defaults:

```python
class MyExporter(ExporterPlugin):
    DEFAULT_CONFIG = {
        "indent": 2,
        "encoding": "utf-8",
        "include_timestamp": True,
    }
    
    def __init__(self, config: dict | None = None):
        # Merge with defaults
        final_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(config=final_config)
```

### Configuration Priority

1. **Environment Variables** (highest)
2. **User Configuration File** (proxima.yaml)
3. **Plugin-Specific Config File**
4. **Programmatic Configuration**
5. **Plugin Defaults** (lowest)

---

## Plugin State Persistence

### Saving Plugin State

```yaml
plugins:
  persistence:
    enabled: true
    path: ~/.proxima/plugin_state.json
    
    # What to persist
    persist:
      - enabled_status
      - configuration
      - metrics
      - last_used
```

### State File Format

```json
{
  "version": "1.0",
  "last_saved": "2024-01-15T10:30:00Z",
  "plugins": {
    "json_exporter": {
      "enabled": true,
      "config": {"indent": 4},
      "last_used": "2024-01-15T10:25:00Z"
    },
    "metrics_hook": {
      "enabled": true,
      "config": {"collect_timing": true},
      "metrics": {
        "execution_count": 150,
        "total_time_ms": 45230
      }
    }
  }
}
```

---

## Troubleshooting

### Common Issues

#### Plugin Not Loading

```yaml
# Enable debug logging
logging:
  level: DEBUG

plugins:
  # Show discovery details
  debug_discovery: true
```

```bash
# Check plugin status
proxima plugin list --verbose

# Test plugin loading
proxima plugin info my_plugin --debug
```

#### Configuration Not Applied

1. Check configuration file syntax
2. Verify environment variable names
3. Check configuration priority
4. Validate against schema

```bash
# Show effective configuration
proxima plugin config my_plugin --effective

# Compare with defaults
proxima plugin config my_plugin --diff
```

#### Plugin Conflicts

```yaml
plugins:
  # Disable conflicting plugins
  disabled:
    - conflicting_plugin
  
  # Set explicit priorities
  priority:
    primary_plugin: 10
    secondary_plugin: 5
```
