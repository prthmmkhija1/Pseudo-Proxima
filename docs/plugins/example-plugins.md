# Example Plugins

This document provides complete, working examples of each plugin type in Proxima. These examples are included in the `src/proxima/plugins/examples/` directory.

## Overview

Proxima includes 8 example plugins demonstrating all plugin types:

| Plugin | Type | Description |
|--------|------|-------------|
| JSONExporterPlugin | Exporter | Export to formatted JSON |
| CSVExporterPlugin | Exporter | Export to CSV with nested flattening |
| MarkdownExporterPlugin | Exporter | Generate Markdown reports |
| StatisticalAnalyzerPlugin | Analyzer | Entropy, uniformity, confidence intervals |
| FidelityAnalyzerPlugin | Analyzer | Classical fidelity, KL divergence |
| PerformanceAnalyzerPlugin | Analyzer | Timing analysis, recommendations |
| LoggingHookPlugin | Hook | Event logging with timing |
| MetricsHookPlugin | Hook | Metrics collection and aggregation |

## Using Example Plugins

### Quick Start

```python
from proxima.plugins.examples import register_example_plugins
from proxima.plugins.loader import PluginRegistry

# Create registry and register all examples
registry = PluginRegistry()
register_example_plugins(registry)

# List all registered plugins
for plugin in registry.list_plugins():
    print(f"{plugin.name} v{plugin.version}: {plugin.description}")
```

### Individual Plugin Usage

```python
from proxima.plugins.examples.exporters import JSONExporterPlugin
from proxima.plugins.base import PluginContext

# Create and initialize
plugin = JSONExporterPlugin()
plugin.initialize(PluginContext(backend_name="cirq", num_qubits=2))

# Use the plugin
data = {"counts": {"00": 500, "11": 500}, "execution_time": 0.5}
json_output = plugin.export(data, indent=2, include_metadata=True)

# Cleanup
plugin.shutdown()
```

---

## Exporter Plugins

### JSONExporterPlugin

Exports simulation results to pretty-printed JSON with optional metadata.

**Source:** `src/proxima/plugins/examples/exporters.py`

```python
from proxima.plugins.base import ExporterPlugin, PluginContext
from typing import Dict, Any
import json
from datetime import datetime, timezone

class JSONExporterPlugin(ExporterPlugin):
    """Export results to formatted JSON with metadata."""
    
    name = "json-exporter"
    version = "1.0.0"
    description = "Export simulation results to formatted JSON"
    supported_formats = ["json"]
    
    def initialize(self, context: PluginContext) -> None:
        self._context = context
    
    def export(
        self, 
        data: Dict[str, Any],
        indent: int = 2,
        include_metadata: bool = True,
        **options
    ) -> str:
        output = {"data": data}
        
        if include_metadata:
            output["export_timestamp"] = datetime.now(timezone.utc).isoformat()
            output["plugin_version"] = self.version
            if self._context:
                output["context"] = {
                    "backend": self._context.backend_name,
                    "num_qubits": self._context.num_qubits,
                }
        
        return json.dumps(output, indent=indent, default=str)
    
    def shutdown(self) -> None:
        pass
```

**Usage Example:**

```python
from proxima.plugins.examples.exporters import JSONExporterPlugin
from proxima.plugins.base import PluginContext

plugin = JSONExporterPlugin()
plugin.initialize(PluginContext(backend_name="cirq", num_qubits=4))

results = {
    "counts": {"0000": 480, "1111": 520},
    "execution_time": 0.15,
    "backend": "cirq"
}

json_output = plugin.export(results, indent=4, include_metadata=True)
print(json_output)
```

**Output:**

```json
{
    "data": {
        "counts": {
            "0000": 480,
            "1111": 520
        },
        "execution_time": 0.15,
        "backend": "cirq"
    },
    "export_timestamp": "2024-01-15T10:30:00+00:00",
    "plugin_version": "1.0.0",
    "context": {
        "backend": "cirq",
        "num_qubits": 4
    }
}
```

---

### CSVExporterPlugin

Exports results to CSV format with automatic nested dictionary flattening.

**Source:** `src/proxima/plugins/examples/exporters.py`

```python
class CSVExporterPlugin(ExporterPlugin):
    """Export results to CSV format with nested flattening."""
    
    name = "csv-exporter"
    version = "1.0.0"
    description = "Export simulation results to CSV format"
    supported_formats = ["csv"]
    
    def _flatten_dict(
        self, 
        d: Dict[str, Any], 
        prefix: str = ""
    ) -> Dict[str, Any]:
        """Flatten nested dictionaries with dot notation."""
        items = {}
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(self._flatten_dict(value, new_key))
            else:
                items[new_key] = value
        return items
    
    def export(self, data: Dict[str, Any], **options) -> str:
        import csv
        import io
        
        output = io.StringIO()
        
        # Handle counts specially
        if "counts" in data and isinstance(data["counts"], dict):
            writer = csv.writer(output)
            writer.writerow(["state", "count", "probability"])
            total = sum(data["counts"].values())
            for state, count in sorted(data["counts"].items()):
                prob = count / total if total > 0 else 0
                writer.writerow([state, count, f"{prob:.6f}"])
        else:
            # Flatten and export
            flat = self._flatten_dict(data)
            writer = csv.DictWriter(output, fieldnames=flat.keys())
            writer.writeheader()
            writer.writerow(flat)
        
        return output.getvalue()
```

**Usage Example:**

```python
from proxima.plugins.examples.exporters import CSVExporterPlugin
from proxima.plugins.base import PluginContext

plugin = CSVExporterPlugin()
plugin.initialize(PluginContext(backend_name="qiskit_aer", num_qubits=2))

results = {"counts": {"00": 250, "01": 125, "10": 125, "11": 500}}
csv_output = plugin.export(results)
print(csv_output)
```

**Output:**

```csv
state,count,probability
00,250,0.250000
01,125,0.125000
10,125,0.125000
11,500,0.500000
```

---

### MarkdownExporterPlugin

Generates formatted Markdown reports from simulation results.

```python
class MarkdownExporterPlugin(ExporterPlugin):
    """Generate Markdown reports from simulation results."""
    
    name = "markdown-exporter"
    version = "1.0.0"
    description = "Generate Markdown formatted reports"
    supported_formats = ["md", "markdown"]
    
    def export(self, data: Dict[str, Any], **options) -> str:
        lines = []
        
        # Header
        backend = data.get("backend", "Unknown")
        lines.append(f"# Simulation Results: {backend}")
        lines.append("")
        
        # Execution info
        if "execution_time" in data:
            lines.append("## Execution Summary")
            lines.append(f"- **Execution Time:** {data['execution_time']:.4f}s")
            if "shots" in data:
                lines.append(f"- **Shots:** {data['shots']}")
            lines.append("")
        
        # Measurement results
        if "counts" in data:
            lines.append("## Measurement Results")
            lines.append("")
            lines.append("| State | Count | Probability |")
            lines.append("|-------|-------|-------------|")
            
            total = sum(data["counts"].values())
            for state, count in sorted(
                data["counts"].items(),
                key=lambda x: -x[1]
            ):
                prob = count / total if total > 0 else 0
                lines.append(f"| `{state}` | {count} | {prob:.4f} |")
            lines.append("")
        
        return "\n".join(lines)
```

---

## Analyzer Plugins

### StatisticalAnalyzerPlugin

Performs statistical analysis including entropy, uniformity, and confidence intervals.

**Source:** `src/proxima/plugins/examples/analyzers.py`

```python
from proxima.plugins.base import AnalyzerPlugin, PluginContext
from typing import Dict, Any
import math

class StatisticalAnalyzerPlugin(AnalyzerPlugin):
    """Perform statistical analysis on simulation results."""
    
    name = "statistical-analyzer"
    version = "1.0.0"
    description = "Calculate entropy, uniformity, and confidence intervals"
    analysis_types = ["statistical", "entropy", "uniformity"]
    
    def analyze(
        self,
        data: Dict[str, Any],
        confidence_level: float = 0.95,
        **options
    ) -> Dict[str, Any]:
        counts = data.get("counts", {})
        total = sum(counts.values())
        
        if total == 0:
            return {"error": "No measurement data"}
        
        # Calculate probabilities
        probs = {k: v / total for k, v in counts.items()}
        
        # Shannon entropy
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Maximum entropy (uniform distribution)
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0
        
        # Uniformity score (1.0 = perfectly uniform)
        uniformity = entropy / max_entropy if max_entropy > 0 else 1.0
        
        # Confidence intervals using normal approximation
        z = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        confidence_intervals = {}
        for state, count in counts.items():
            p = count / total
            se = math.sqrt(p * (1 - p) / total)
            confidence_intervals[state] = {
                "probability": p,
                "lower": max(0, p - z * se),
                "upper": min(1, p + z * se),
            }
        
        return {
            "entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0,
            "uniformity": uniformity,
            "num_states": len(counts),
            "total_counts": total,
            "confidence_level": confidence_level,
            "confidence_intervals": confidence_intervals,
        }
```

**Usage Example:**

```python
from proxima.plugins.examples.analyzers import StatisticalAnalyzerPlugin
from proxima.plugins.base import PluginContext

plugin = StatisticalAnalyzerPlugin()
plugin.initialize(PluginContext(backend_name="cirq", num_qubits=2))

# Analyze Bell state results
results = {"counts": {"00": 500, "11": 500}}
analysis = plugin.analyze(results)

print(f"Entropy: {analysis['entropy']:.4f} bits")
print(f"Uniformity: {analysis['uniformity']:.4f}")
print(f"95% CI for |00⟩: [{analysis['confidence_intervals']['00']['lower']:.4f}, "
      f"{analysis['confidence_intervals']['00']['upper']:.4f}]")
```

**Output:**

```
Entropy: 1.0000 bits
Uniformity: 1.0000
95% CI for |00⟩: [0.4690, 0.5310]
```

---

### FidelityAnalyzerPlugin

Compares measurement distributions using fidelity metrics.

```python
class FidelityAnalyzerPlugin(AnalyzerPlugin):
    """Calculate fidelity metrics between distributions."""
    
    name = "fidelity-analyzer"
    version = "1.0.0"
    description = "Compute classical fidelity, KL divergence, Hellinger distance"
    analysis_types = ["fidelity", "divergence"]
    
    def analyze(
        self,
        data: Dict[str, Any],
        reference: Dict[str, int] = None,
        **options
    ) -> Dict[str, Any]:
        counts = data.get("counts", {})
        
        if reference is None:
            return {"error": "Reference distribution required"}
        
        # Normalize distributions
        total_p = sum(counts.values())
        total_q = sum(reference.values())
        
        if total_p == 0 or total_q == 0:
            return {"error": "Empty distribution"}
        
        all_states = set(counts.keys()) | set(reference.keys())
        
        # Classical fidelity: F(P,Q) = (Σ√(p_i * q_i))²
        fidelity = 0.0
        for state in all_states:
            p = counts.get(state, 0) / total_p
            q = reference.get(state, 0) / total_q
            fidelity += math.sqrt(p * q)
        fidelity = fidelity ** 2
        
        # KL divergence: D_KL(P||Q) = Σ p_i * log(p_i / q_i)
        kl_div = 0.0
        for state in all_states:
            p = counts.get(state, 0) / total_p
            q = (reference.get(state, 0) + 1e-10) / total_q
            if p > 0:
                kl_div += p * math.log(p / q)
        
        # Hellinger distance: H(P,Q) = √(1 - Σ√(p_i * q_i))
        bc = 0.0  # Bhattacharyya coefficient
        for state in all_states:
            p = counts.get(state, 0) / total_p
            q = reference.get(state, 0) / total_q
            bc += math.sqrt(p * q)
        hellinger = math.sqrt(max(0, 1 - bc))
        
        return {
            "classical_fidelity": fidelity,
            "kl_divergence": kl_div,
            "hellinger_distance": hellinger,
            "bhattacharyya_coefficient": bc,
        }
```

**Usage Example:**

```python
from proxima.plugins.examples.analyzers import FidelityAnalyzerPlugin
from proxima.plugins.base import PluginContext

plugin = FidelityAnalyzerPlugin()
plugin.initialize(PluginContext(backend_name="cirq", num_qubits=2))

# Compare measured to ideal
measured = {"counts": {"00": 480, "01": 10, "10": 10, "11": 500}}
ideal = {"00": 500, "11": 500}

analysis = plugin.analyze(measured, reference=ideal)

print(f"Classical Fidelity: {analysis['classical_fidelity']:.4f}")
print(f"KL Divergence: {analysis['kl_divergence']:.4f}")
print(f"Hellinger Distance: {analysis['hellinger_distance']:.4f}")
```

---

### PerformanceAnalyzerPlugin

Analyzes execution performance and generates optimization recommendations.

```python
class PerformanceAnalyzerPlugin(AnalyzerPlugin):
    """Analyze execution performance and provide recommendations."""
    
    name = "performance-analyzer"
    version = "1.0.0"
    description = "Analyze timing, throughput, and generate recommendations"
    analysis_types = ["performance", "timing"]
    
    def analyze(self, data: Dict[str, Any], **options) -> Dict[str, Any]:
        result = {}
        recommendations = []
        
        # Timing analysis
        if "execution_time" in data:
            exec_time = data["execution_time"]
            result["execution_time"] = exec_time
            
            shots = data.get("shots", 1000)
            result["throughput"] = shots / exec_time if exec_time > 0 else 0
            
            if exec_time > 5.0:
                recommendations.append(
                    "Consider using a faster backend or reducing circuit depth"
                )
        
        # Circuit analysis
        metadata = data.get("metadata", {})
        if "circuit_depth" in metadata:
            depth = metadata["circuit_depth"]
            result["circuit_depth"] = depth
            if depth > 50:
                recommendations.append(
                    "High circuit depth may affect accuracy; consider optimization"
                )
        
        if "gate_count" in metadata:
            gate_count = metadata["gate_count"]
            result["gate_count"] = gate_count
            if gate_count > 100:
                recommendations.append(
                    "Consider circuit optimization to reduce gate count"
                )
        
        result["recommendations"] = recommendations
        return result
```

---

## Hook Plugins

### LoggingHookPlugin

Logs execution events with timing information.

**Source:** `src/proxima/plugins/examples/hooks.py`

```python
from proxima.plugins.base import Plugin
from proxima.plugins.hooks import HookType, HookContext, HookManager
import logging
import time

class LoggingHookPlugin(Plugin):
    """Log execution events with timing."""
    
    name = "logging-hook"
    version = "1.0.0"
    description = "Comprehensive event logging"
    supported_hooks = [
        HookType.BEFORE_EXECUTION,
        HookType.AFTER_EXECUTION,
        HookType.ON_ERROR,
    ]
    
    def __init__(self):
        self._logger = logging.getLogger("proxima.plugins.logging")
        self._timings = {}
    
    def initialize(
        self, 
        context, 
        hook_manager: HookManager = None
    ) -> None:
        self._context = context
        
        if hook_manager:
            hook_manager.register(
                HookType.BEFORE_EXECUTION, 
                self._on_before_execution
            )
            hook_manager.register(
                HookType.AFTER_EXECUTION, 
                self._on_after_execution
            )
            hook_manager.register(
                HookType.ON_ERROR, 
                self._on_error
            )
    
    def _on_before_execution(self, ctx: HookContext) -> None:
        exec_id = ctx.data.get("execution_id", id(ctx))
        self._timings[exec_id] = time.perf_counter()
        
        self._logger.info(
            f"Starting execution on {ctx.data.get('backend', 'unknown')} "
            f"with {ctx.data.get('shots', 'N/A')} shots"
        )
    
    def _on_after_execution(self, ctx: HookContext) -> None:
        exec_id = ctx.data.get("execution_id", id(ctx))
        start_time = self._timings.pop(exec_id, None)
        
        elapsed = time.perf_counter() - start_time if start_time else 0
        
        self._logger.info(
            f"Completed execution in {elapsed:.4f}s"
        )
    
    def _on_error(self, ctx: HookContext) -> None:
        self._logger.error(
            f"Execution error: {ctx.data.get('error', 'Unknown error')}"
        )
```

---

### MetricsHookPlugin

Collects and aggregates execution metrics.

```python
from collections import defaultdict
import statistics

class MetricsHookPlugin(Plugin):
    """Collect and aggregate execution metrics."""
    
    name = "metrics-hook"
    version = "1.0.0"
    description = "Metrics collection and aggregation"
    supported_hooks = [
        HookType.BEFORE_EXECUTION,
        HookType.AFTER_EXECUTION,
    ]
    
    def __init__(self):
        self._metrics = defaultdict(int)
        self._timings = []
        self._backend_usage = defaultdict(int)
        self._current_start = None
    
    def initialize(self, context, hook_manager: HookManager = None) -> None:
        self._context = context
        if hook_manager:
            hook_manager.register(
                HookType.BEFORE_EXECUTION,
                self._on_execution_start
            )
            hook_manager.register(
                HookType.AFTER_EXECUTION,
                self._on_execution_end
            )
    
    def _on_execution_start(self, ctx: HookContext) -> None:
        self._current_start = time.perf_counter()
        self._metrics["total_executions"] += 1
        
        backend = ctx.data.get("backend", "unknown")
        self._backend_usage[backend] += 1
    
    def _on_execution_end(self, ctx: HookContext) -> None:
        if self._current_start:
            elapsed = time.perf_counter() - self._current_start
            self._timings.append(elapsed)
            self._current_start = None
    
    def get_metrics(self) -> Dict[str, Any]:
        timing_stats = {}
        if self._timings:
            timing_stats = {
                "mean": statistics.mean(self._timings),
                "median": statistics.median(self._timings),
                "min": min(self._timings),
                "max": max(self._timings),
                "stdev": statistics.stdev(self._timings) if len(self._timings) > 1 else 0,
            }
        
        return {
            "total_executions": self._metrics["total_executions"],
            "backend_usage": dict(self._backend_usage),
            "timing": timing_stats,
        }
    
    def reset_metrics(self) -> None:
        self._metrics.clear()
        self._timings.clear()
        self._backend_usage.clear()
```

**Usage Example:**

```python
from proxima.plugins.examples.hooks import MetricsHookPlugin
from proxima.plugins.hooks import HookType, HookContext, HookManager
from proxima.plugins.base import PluginContext

# Setup
hook_manager = HookManager()
plugin = MetricsHookPlugin()
plugin.initialize(
    PluginContext(backend_name="cirq", num_qubits=2),
    hook_manager=hook_manager
)

# Simulate some executions
for i in range(5):
    hook_manager.execute(
        HookType.BEFORE_EXECUTION,
        HookContext(
            hook_type=HookType.BEFORE_EXECUTION,
            data={"backend": "cirq", "execution_id": i}
        )
    )
    # ... simulate work ...
    hook_manager.execute(
        HookType.AFTER_EXECUTION,
        HookContext(
            hook_type=HookType.AFTER_EXECUTION,
            data={"backend": "cirq", "execution_id": i}
        )
    )

# Get aggregated metrics
metrics = plugin.get_metrics()
print(f"Total Executions: {metrics['total_executions']}")
print(f"Backend Usage: {metrics['backend_usage']}")
print(f"Mean Timing: {metrics['timing']['mean']:.4f}s")
```

---

## Complete Integration Example

Here's a complete example using multiple plugins together:

```python
from proxima.plugins.loader import PluginRegistry
from proxima.plugins.hooks import HookManager, HookType, HookContext
from proxima.plugins.base import PluginContext
from proxima.plugins.examples import register_example_plugins

# Initialize the plugin system
registry = PluginRegistry()
hook_manager = HookManager()

# Register all example plugins
register_example_plugins(registry)

# Create context
context = PluginContext(
    backend_name="cirq",
    num_qubits=4,
    shots=1000
)

# Initialize plugins
json_exporter = registry.get_plugin("json-exporter")
stats_analyzer = registry.get_plugin("statistical-analyzer")
metrics_hook = registry.get_plugin("metrics-hook")

json_exporter.initialize(context)
stats_analyzer.initialize(context)
metrics_hook.initialize(context, hook_manager=hook_manager)

# Simulate execution with hooks
hook_manager.execute(
    HookType.BEFORE_EXECUTION,
    HookContext(hook_type=HookType.BEFORE_EXECUTION, data={"backend": "cirq"})
)

# Simulated results
results = {
    "counts": {"0000": 120, "0001": 30, "1110": 30, "1111": 820},
    "execution_time": 0.25,
    "shots": 1000,
}

hook_manager.execute(
    HookType.AFTER_EXECUTION,
    HookContext(hook_type=HookType.AFTER_EXECUTION, data={"backend": "cirq"})
)

# Analyze results
analysis = stats_analyzer.analyze(results)

# Combine data
full_results = {
    **results,
    "analysis": analysis,
}

# Export to JSON
output = json_exporter.export(full_results)
print(output)

# Print metrics
print("\n--- Execution Metrics ---")
print(metrics_hook.get_metrics())

# Cleanup
json_exporter.shutdown()
stats_analyzer.shutdown()
```

---

## See Also

- [Plugin Development Guide](plugin-development.md) - How to create custom plugins
- [Plugin API Reference](plugin-api-reference.md) - Complete API documentation
