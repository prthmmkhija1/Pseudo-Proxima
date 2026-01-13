"""
Tests for example plugin implementations.

Comprehensive tests for exporter, analyzer, and hook plugins.
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any
from io import StringIO


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_counts() -> Dict[str, int]:
    """Sample measurement counts."""
    return {
        "00": 250,
        "01": 125,
        "10": 125,
        "11": 500,
    }


@pytest.fixture
def sample_results(sample_counts) -> Dict[str, Any]:
    """Sample execution results."""
    return {
        "backend": "cirq",
        "num_qubits": 2,
        "shots": 1000,
        "counts": sample_counts,
        "execution_time": 0.5,
        "metadata": {
            "circuit_depth": 5,
            "gate_count": 10,
        }
    }


@pytest.fixture
def sample_statevector() -> Dict[str, complex]:
    """Sample statevector for fidelity tests."""
    import math
    val = 1 / math.sqrt(2)
    return {
        "00": complex(val, 0),
        "01": complex(0, 0),
        "10": complex(0, 0),
        "11": complex(val, 0),
    }


@pytest.fixture
def mock_context():
    """Mock context for plugins."""
    from proxima.plugins.base import PluginContext
    return PluginContext(
        backend_name="cirq",
        num_qubits=2,
        shots=1000,
        config={"precision": "double"},
    )


# =============================================================================
# Exporter Plugin Tests
# =============================================================================

class TestJSONExporterPlugin:
    """Tests for JSON exporter plugin."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        from proxima.plugins.examples.exporters import JSONExporterPlugin
        
        plugin = JSONExporterPlugin()
        
        assert plugin.name == "json-exporter"
        assert plugin.version == "1.0.0"
        assert "json" in plugin.description.lower()
        assert plugin.supported_formats == ["json"]
    
    def test_export_results(self, sample_results, mock_context):
        """Test exporting results to JSON."""
        from proxima.plugins.examples.exporters import JSONExporterPlugin
        
        plugin = JSONExporterPlugin()
        plugin.initialize(mock_context)
        
        output = plugin.export(sample_results)
        
        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["data"] == sample_results
        assert "export_timestamp" in parsed
        assert "plugin_version" in parsed
    
    def test_export_with_custom_options(self, sample_results, mock_context):
        """Test export with custom options."""
        from proxima.plugins.examples.exporters import JSONExporterPlugin
        
        plugin = JSONExporterPlugin()
        plugin.initialize(mock_context)
        
        output = plugin.export(
            sample_results,
            indent=None,
            include_metadata=False
        )
        
        parsed = json.loads(output)
        # With indent=None, output should be compact
        assert "\n" not in output or len(output.split("\n")) <= 2
    
    def test_shutdown_cleans_up(self, mock_context):
        """Test plugin shutdown."""
        from proxima.plugins.examples.exporters import JSONExporterPlugin
        
        plugin = JSONExporterPlugin()
        plugin.initialize(mock_context)
        plugin.shutdown()
        
        # Should not raise


class TestCSVExporterPlugin:
    """Tests for CSV exporter plugin."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        from proxima.plugins.examples.exporters import CSVExporterPlugin
        
        plugin = CSVExporterPlugin()
        
        assert plugin.name == "csv-exporter"
        assert plugin.supported_formats == ["csv"]
    
    def test_export_counts(self, sample_counts, mock_context):
        """Test exporting counts to CSV."""
        from proxima.plugins.examples.exporters import CSVExporterPlugin
        
        plugin = CSVExporterPlugin()
        plugin.initialize(mock_context)
        
        output = plugin.export({"counts": sample_counts})
        
        assert "state,count" in output
        assert "00,250" in output
        assert "11,500" in output
    
    def test_export_full_results(self, sample_results, mock_context):
        """Test exporting full results."""
        from proxima.plugins.examples.exporters import CSVExporterPlugin
        
        plugin = CSVExporterPlugin()
        plugin.initialize(mock_context)
        
        output = plugin.export(sample_results)
        
        # Should contain header and data
        lines = output.strip().split("\n")
        assert len(lines) >= 2  # Header + at least one data row
    
    def test_flatten_nested_dict(self, mock_context):
        """Test flattening nested dictionaries."""
        from proxima.plugins.examples.exporters import CSVExporterPlugin
        
        plugin = CSVExporterPlugin()
        
        nested = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": {"f": {"g": 4}},
        }
        
        flat = plugin._flatten_dict(nested)
        
        assert flat["a"] == 1
        assert flat["b.c"] == 2
        assert flat["b.d"] == 3
        assert flat["e.f.g"] == 4


class TestMarkdownExporterPlugin:
    """Tests for Markdown exporter plugin."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        from proxima.plugins.examples.exporters import MarkdownExporterPlugin
        
        plugin = MarkdownExporterPlugin()
        
        assert plugin.name == "markdown-exporter"
        assert plugin.supported_formats == ["md", "markdown"]
    
    def test_export_produces_markdown(self, sample_results, mock_context):
        """Test that export produces valid Markdown."""
        from proxima.plugins.examples.exporters import MarkdownExporterPlugin
        
        plugin = MarkdownExporterPlugin()
        plugin.initialize(mock_context)
        
        output = plugin.export(sample_results)
        
        # Should contain Markdown headers
        assert "# " in output or "## " in output
        assert "cirq" in output
        assert "1000" in output  # shots
    
    def test_export_includes_table(self, sample_results, mock_context):
        """Test that export includes measurement table."""
        from proxima.plugins.examples.exporters import MarkdownExporterPlugin
        
        plugin = MarkdownExporterPlugin()
        plugin.initialize(mock_context)
        
        output = plugin.export(sample_results)
        
        # Markdown table markers
        assert "|" in output
        assert "---" in output


# =============================================================================
# Analyzer Plugin Tests
# =============================================================================

class TestStatisticalAnalyzerPlugin:
    """Tests for statistical analyzer plugin."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        from proxima.plugins.examples.analyzers import StatisticalAnalyzerPlugin
        
        plugin = StatisticalAnalyzerPlugin()
        
        assert plugin.name == "statistical-analyzer"
        assert "statistical" in plugin.analysis_types
    
    def test_analyze_entropy(self, sample_counts, mock_context):
        """Test entropy calculation."""
        from proxima.plugins.examples.analyzers import StatisticalAnalyzerPlugin
        
        plugin = StatisticalAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        result = plugin.analyze({"counts": sample_counts})
        
        assert "entropy" in result
        assert 0 <= result["entropy"] <= 2  # Max entropy for 4 states
    
    def test_analyze_uniformity(self, mock_context):
        """Test uniformity calculation for uniform distribution."""
        from proxima.plugins.examples.analyzers import StatisticalAnalyzerPlugin
        
        plugin = StatisticalAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        uniform_counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        result = plugin.analyze({"counts": uniform_counts})
        
        assert "uniformity" in result
        assert result["uniformity"] > 0.9  # Should be close to 1
    
    def test_analyze_confidence_interval(self, sample_counts, mock_context):
        """Test confidence interval calculation."""
        from proxima.plugins.examples.analyzers import StatisticalAnalyzerPlugin
        
        plugin = StatisticalAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        result = plugin.analyze({"counts": sample_counts})
        
        assert "confidence_intervals" in result
        for state in sample_counts:
            assert state in result["confidence_intervals"]
            ci = result["confidence_intervals"][state]
            assert ci["lower"] <= ci["upper"]


class TestFidelityAnalyzerPlugin:
    """Tests for fidelity analyzer plugin."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        from proxima.plugins.examples.analyzers import FidelityAnalyzerPlugin
        
        plugin = FidelityAnalyzerPlugin()
        
        assert plugin.name == "fidelity-analyzer"
        assert "fidelity" in plugin.analysis_types
    
    def test_classical_fidelity_identical(self, sample_counts, mock_context):
        """Test classical fidelity for identical distributions."""
        from proxima.plugins.examples.analyzers import FidelityAnalyzerPlugin
        
        plugin = FidelityAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        result = plugin.analyze(
            {"counts": sample_counts},
            reference=sample_counts
        )
        
        assert result["classical_fidelity"] == pytest.approx(1.0)
    
    def test_kl_divergence_zero_for_identical(self, sample_counts, mock_context):
        """Test KL divergence is zero for identical distributions."""
        from proxima.plugins.examples.analyzers import FidelityAnalyzerPlugin
        
        plugin = FidelityAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        result = plugin.analyze(
            {"counts": sample_counts},
            reference=sample_counts
        )
        
        assert result["kl_divergence"] == pytest.approx(0.0, abs=1e-10)
    
    def test_hellinger_distance(self, mock_context):
        """Test Hellinger distance calculation."""
        from proxima.plugins.examples.analyzers import FidelityAnalyzerPlugin
        
        plugin = FidelityAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        counts1 = {"0": 500, "1": 500}
        counts2 = {"0": 800, "1": 200}
        
        result = plugin.analyze(
            {"counts": counts1},
            reference=counts2
        )
        
        assert "hellinger_distance" in result
        assert 0 <= result["hellinger_distance"] <= 1


class TestPerformanceAnalyzerPlugin:
    """Tests for performance analyzer plugin."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        from proxima.plugins.examples.analyzers import PerformanceAnalyzerPlugin
        
        plugin = PerformanceAnalyzerPlugin()
        
        assert plugin.name == "performance-analyzer"
        assert "performance" in plugin.analysis_types
    
    def test_analyze_timing(self, sample_results, mock_context):
        """Test timing analysis."""
        from proxima.plugins.examples.analyzers import PerformanceAnalyzerPlugin
        
        plugin = PerformanceAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        result = plugin.analyze(sample_results)
        
        assert "execution_time" in result
        assert "throughput" in result
    
    def test_analyze_circuit_stats(self, sample_results, mock_context):
        """Test circuit statistics analysis."""
        from proxima.plugins.examples.analyzers import PerformanceAnalyzerPlugin
        
        plugin = PerformanceAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        result = plugin.analyze(sample_results)
        
        if "metadata" in sample_results:
            assert "circuit_depth" in result or "gate_count" in result
    
    def test_generate_recommendations(self, mock_context):
        """Test recommendation generation."""
        from proxima.plugins.examples.analyzers import PerformanceAnalyzerPlugin
        
        plugin = PerformanceAnalyzerPlugin()
        plugin.initialize(mock_context)
        
        # Slow execution should generate recommendations
        slow_results = {
            "execution_time": 10.0,
            "shots": 1000,
            "metadata": {"circuit_depth": 100, "gate_count": 1000}
        }
        
        result = plugin.analyze(slow_results)
        
        assert "recommendations" in result
        # Should suggest optimization due to slow execution
        if result["recommendations"]:
            assert any("optim" in r.lower() for r in result["recommendations"])


# =============================================================================
# Hook Plugin Tests
# =============================================================================

class TestLoggingHookPlugin:
    """Tests for logging hook plugin."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        from proxima.plugins.examples.hooks import LoggingHookPlugin
        
        plugin = LoggingHookPlugin()
        
        assert plugin.name == "logging-hook"
        assert len(plugin.supported_hooks) > 0
    
    def test_registers_hooks(self, mock_context):
        """Test that hooks are registered."""
        from proxima.plugins.examples.hooks import LoggingHookPlugin
        from proxima.plugins.hooks import HookType
        
        plugin = LoggingHookPlugin()
        hook_manager = Mock()
        hook_manager.register = Mock()
        
        plugin.initialize(mock_context, hook_manager=hook_manager)
        
        # Should register multiple hooks
        assert hook_manager.register.call_count > 0
    
    def test_log_event(self, mock_context, caplog):
        """Test event logging."""
        from proxima.plugins.examples.hooks import LoggingHookPlugin
        from proxima.plugins.hooks import HookType, HookContext
        import logging
        
        plugin = LoggingHookPlugin()
        plugin.initialize(mock_context)
        
        hook_context = HookContext(
            hook_type=HookType.BEFORE_EXECUTION,
            data={"test": "value"}
        )
        
        with caplog.at_level(logging.INFO):
            plugin._log_event(hook_context)
        
        # Should have logged something
        assert len(caplog.records) > 0


class TestMetricsHookPlugin:
    """Tests for metrics hook plugin."""
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        from proxima.plugins.examples.hooks import MetricsHookPlugin
        
        plugin = MetricsHookPlugin()
        
        assert plugin.name == "metrics-hook"
    
    def test_tracks_executions(self, mock_context):
        """Test execution tracking."""
        from proxima.plugins.examples.hooks import MetricsHookPlugin
        from proxima.plugins.hooks import HookType, HookContext
        
        plugin = MetricsHookPlugin()
        plugin.initialize(mock_context)
        
        # Simulate execution
        plugin._on_execution_start(HookContext(
            hook_type=HookType.BEFORE_EXECUTION,
            data={"backend": "cirq"}
        ))
        
        plugin._on_execution_end(HookContext(
            hook_type=HookType.AFTER_EXECUTION,
            data={"backend": "cirq", "execution_time": 0.5}
        ))
        
        metrics = plugin.get_metrics()
        
        assert metrics["total_executions"] == 1
        assert metrics["backend_usage"]["cirq"] == 1
    
    def test_timing_statistics(self, mock_context):
        """Test timing statistics calculation."""
        from proxima.plugins.examples.hooks import MetricsHookPlugin
        from proxima.plugins.hooks import HookType, HookContext
        
        plugin = MetricsHookPlugin()
        plugin.initialize(mock_context)
        
        # Simulate multiple executions
        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        for t in times:
            plugin._on_execution_start(HookContext(
                hook_type=HookType.BEFORE_EXECUTION,
                data={"backend": "cirq"}
            ))
            plugin._on_execution_end(HookContext(
                hook_type=HookType.AFTER_EXECUTION,
                data={"backend": "cirq", "execution_time": t}
            ))
        
        metrics = plugin.get_metrics()
        
        assert "timing" in metrics
        assert metrics["timing"]["mean"] == pytest.approx(0.3, abs=0.01)
        assert metrics["timing"]["min"] == pytest.approx(0.1, abs=0.01)
        assert metrics["timing"]["max"] == pytest.approx(0.5, abs=0.01)
    
    def test_reset_metrics(self, mock_context):
        """Test metrics reset."""
        from proxima.plugins.examples.hooks import MetricsHookPlugin
        from proxima.plugins.hooks import HookType, HookContext
        
        plugin = MetricsHookPlugin()
        plugin.initialize(mock_context)
        
        # Add some metrics
        plugin._on_execution_start(HookContext(
            hook_type=HookType.BEFORE_EXECUTION,
            data={"backend": "cirq"}
        ))
        plugin._on_execution_end(HookContext(
            hook_type=HookType.AFTER_EXECUTION,
            data={"backend": "cirq", "execution_time": 0.5}
        ))
        
        plugin.reset_metrics()
        metrics = plugin.get_metrics()
        
        assert metrics["total_executions"] == 0


# =============================================================================
# Plugin Registration Tests
# =============================================================================

class TestPluginRegistration:
    """Tests for plugin registration."""
    
    def test_register_example_plugins(self):
        """Test registering all example plugins."""
        from proxima.plugins.examples import register_example_plugins
        from proxima.plugins.loader import PluginRegistry
        
        registry = PluginRegistry()
        register_example_plugins(registry)
        
        # Should have registered 8 plugins
        assert len(registry.list_plugins()) == 8
    
    def test_plugin_types_registered(self):
        """Test that all plugin types are registered."""
        from proxima.plugins.examples import register_example_plugins
        from proxima.plugins.loader import PluginRegistry
        
        registry = PluginRegistry()
        register_example_plugins(registry)
        
        # Check for each type
        exporters = registry.get_plugins_by_type("exporter")
        analyzers = registry.get_plugins_by_type("analyzer")
        hooks = registry.get_plugins_by_type("hook")
        
        assert len(exporters) >= 3
        assert len(analyzers) >= 3
        assert len(hooks) >= 2
