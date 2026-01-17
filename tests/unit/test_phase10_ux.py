"""Tests for Phase 10.3: User Experience Enhancements."""

from __future__ import annotations

import io
import sys
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from proxima.benchmarks.ux_enhancements import (
    BenchmarkProgress,
    BenchmarkProgressConfig,
    InteractiveMode,
    ProgressStyle,
    RealTimeMetricsDisplay,
    ResultFormatter,
    SmartDefaults,
    Sparkline,
    WatchMode,
)


class TestBenchmarkProgress:
    """Tests for progress indicators."""

    def test_progress_creation(self) -> None:
        """BenchmarkProgress initializes with correct state."""
        progress = BenchmarkProgress(total=100)

        assert progress.total == 100
        assert progress.config.style == ProgressStyle.DETAILED

    def test_progress_update(self) -> None:
        """Progress updates track completed items."""
        progress = BenchmarkProgress(total=10)
        progress.start()

        progress.update(
            completed=5,
            current_backend="lret",
            current_circuit="bell",
            last_time_ms=15.5,
        )

        assert len(progress._times) == 1
        assert progress._times[0] == 15.5
        assert progress._last_metrics["backend"] == "lret"

        progress.stop()

    def test_progress_summary(self) -> None:
        """Progress provides accurate summary statistics."""
        progress = BenchmarkProgress(total=5)
        progress.start()

        times = [10.0, 20.0, 30.0, 40.0, 50.0]
        for i, t in enumerate(times, 1):
            progress.update(completed=i, last_time_ms=t)

        summary = progress.get_summary()

        assert summary["total"] == 5
        assert summary["completed"] == 5
        assert summary["avg_time_ms"] == 30.0
        assert summary["min_time_ms"] == 10.0
        assert summary["max_time_ms"] == 50.0

        progress.stop()

    def test_progress_eta_estimation(self) -> None:
        """Progress estimates remaining time correctly."""
        progress = BenchmarkProgress(total=10)
        progress.start()
        progress._start_time = time.perf_counter() - 5  # Simulate 5s elapsed

        # 5 completed in 5 seconds = 1 per second
        # 5 remaining = 5 seconds ETA
        eta = progress._estimate_remaining(5)

        assert "5" in eta or "s" in eta  # ~5 seconds remaining

        progress.stop()


class TestResultFormatter:
    """Tests for result formatting."""

    def test_format_simple_result_fallback(self) -> None:
        """Formatter uses fallback when rich not available."""
        formatter = ResultFormatter()

        mock_result = MagicMock()
        mock_result.metrics.backend_name = "lret"
        mock_result.metrics.execution_time_ms = 15.5
        mock_result.metrics.memory_peak_mb = 128.0
        mock_result.metrics.memory_baseline_mb = 64.0
        mock_result.metrics.throughput_shots_per_sec = 66000.0
        mock_result.metrics.success_rate_percent = 100.0
        mock_result.metrics.cpu_usage_percent = 45.0
        mock_result.metrics.gpu_usage_percent = None

        # Should not raise
        with patch.object(formatter, "_console", None):
            formatter._format_simple_result(mock_result)

    def test_time_style_thresholds(self) -> None:
        """Formatter applies correct styles based on time thresholds."""
        formatter = ResultFormatter()

        # Fast (< 50ms)
        style_fast = formatter._get_time_style(30.0)
        assert "green" in style_fast

        # Slow (> 500ms)
        style_slow = formatter._get_time_style(600.0)
        assert "red" in style_slow

        # Medium
        style_medium = formatter._get_time_style(200.0)
        assert "yellow" in style_medium

    def test_format_comparison_fallback(self) -> None:
        """Comparison formatter uses fallback when rich not available."""
        formatter = ResultFormatter()

        mock_results = []
        for backend in ["lret", "qsim"]:
            mock_result = MagicMock()
            mock_result.metrics.backend_name = backend
            mock_result.metrics.execution_time_ms = 50.0
            mock_result.metrics.memory_peak_mb = 100.0
            mock_results.append(mock_result)

        speedups = {"lret": 1.0, "qsim": 2.0}

        # Should not raise
        with patch.object(formatter, "_console", None):
            formatter._format_simple_comparison(mock_results, "qsim", speedups)


class TestSparkline:
    """Tests for text-based sparklines."""

    def test_empty_values(self) -> None:
        """Sparkline handles empty values."""
        result = Sparkline.render([])
        assert result == ""

    def test_single_value(self) -> None:
        """Sparkline handles single value."""
        result = Sparkline.render([50.0])
        assert len(result) == 1

    def test_ascending_values(self) -> None:
        """Sparkline shows ascending pattern."""
        result = Sparkline.render([1, 2, 3, 4, 5])

        # Should have increasing block heights
        assert len(result) == 5
        # Later characters should be taller blocks
        assert ord(result[-1]) >= ord(result[0])

    def test_width_limit(self) -> None:
        """Sparkline respects width limit."""
        values = list(range(100))
        result = Sparkline.render(values, width=10)

        assert len(result) == 10


class TestInteractiveMode:
    """Tests for interactive mode."""

    def test_confirm_backend_yes(self) -> None:
        """Interactive mode confirms backend on 'y' input."""
        mode = InteractiveMode()

        with patch("proxima.benchmarks.ux_enhancements.RICH_AVAILABLE", False):
            with patch("builtins.input", return_value="y"):
                result = mode.confirm_backend("lret", "bell_state")

        assert result is True

    def test_confirm_backend_no(self) -> None:
        """Interactive mode skips backend on 'n' input."""
        mode = InteractiveMode()

        with patch("proxima.benchmarks.ux_enhancements.RICH_AVAILABLE", False):
            with patch("builtins.input", return_value="n"):
                result = mode.confirm_backend("lret", "bell_state")

        assert result is False

    def test_confirm_backend_quit(self) -> None:
        """Interactive mode raises on 'q' input."""
        mode = InteractiveMode()

        with patch("proxima.benchmarks.ux_enhancements.RICH_AVAILABLE", False):
            with patch("builtins.input", return_value="q"):
                with pytest.raises(KeyboardInterrupt):
                    mode.confirm_backend("lret", "bell_state")

    def test_select_backends_all(self) -> None:
        """Interactive mode selects all backends on 'all' input."""
        mode = InteractiveMode()
        backends = ["lret", "qsim", "cirq"]

        with patch("proxima.benchmarks.ux_enhancements.RICH_AVAILABLE", False):
            with patch("builtins.input", return_value="all"):
                result = mode.select_backends(backends)

        assert result == backends

    def test_select_backends_by_number(self) -> None:
        """Interactive mode selects backends by number."""
        mode = InteractiveMode()
        backends = ["lret", "qsim", "cirq"]

        with patch("proxima.benchmarks.ux_enhancements.RICH_AVAILABLE", False):
            with patch("builtins.input", return_value="1,3"):
                result = mode.select_backends(backends)

        assert result == ["lret", "cirq"]


class TestWatchMode:
    """Tests for continuous watch mode."""

    def test_watch_mode_creation(self) -> None:
        """WatchMode initializes correctly."""
        watch = WatchMode(interval_seconds=30.0)

        assert watch._interval == 30.0
        assert watch._running is False

    def test_watch_mode_stop(self) -> None:
        """WatchMode can be stopped."""
        watch = WatchMode()
        watch._running = True

        watch.stop()

        assert watch._running is False

    def test_watch_mode_results_history(self) -> None:
        """WatchMode tracks results history."""
        watch = WatchMode(interval_seconds=0.01)

        call_count = 0

        def mock_benchmark() -> dict:
            nonlocal call_count
            call_count += 1
            return {"iteration": call_count}

        # Run with max 3 iterations
        watch.start(mock_benchmark, max_iterations=3)

        assert len(watch._results_history) == 3
        assert all(r["success"] for r in watch._results_history)


class TestSmartDefaults:
    """Tests for intelligent defaults."""

    def test_suggest_runs_small_circuit(self) -> None:
        """Small circuits get more runs for accuracy."""
        runs = SmartDefaults.suggest_runs(num_qubits=5)
        assert runs == 10

    def test_suggest_runs_large_circuit(self) -> None:
        """Large circuits get fewer runs."""
        runs = SmartDefaults.suggest_runs(num_qubits=35)
        assert runs == 2

    def test_suggest_backends_with_gpu(self) -> None:
        """GPU backends preferred for large circuits when GPU available."""
        available = ["lret", "qsim", "cuquantum", "cirq"]

        suggestions = SmartDefaults.suggest_backends(
            num_qubits=20,
            available_backends=available,
            has_gpu=True,
        )

        # GPU backends should be first
        assert suggestions[0] in ["cuquantum", "qsim"]

    def test_suggest_backends_cpu_only(self) -> None:
        """CPU backends preferred when no GPU."""
        available = ["lret", "qsim", "cirq"]

        suggestions = SmartDefaults.suggest_backends(
            num_qubits=10,
            available_backends=available,
            has_gpu=False,
        )

        # All backends should be included
        assert set(suggestions) == set(available)

    def test_estimate_time_scales_with_qubits(self) -> None:
        """Estimated time increases with qubit count."""
        time_10q = SmartDefaults.estimate_time(
            num_qubits=10,
            shots=1024,
            backend_name="lret",
        )
        time_20q = SmartDefaults.estimate_time(
            num_qubits=20,
            shots=1024,
            backend_name="lret",
        )

        assert time_20q > time_10q

    def test_estimate_time_backend_multiplier(self) -> None:
        """Estimated time varies by backend."""
        time_cuquantum = SmartDefaults.estimate_time(
            num_qubits=15,
            shots=1024,
            backend_name="cuquantum",
        )
        time_qiskit = SmartDefaults.estimate_time(
            num_qubits=15,
            shots=1024,
            backend_name="qiskit",
        )

        # cuquantum should be faster
        assert time_cuquantum < time_qiskit

    def test_warn_if_long_under_threshold(self) -> None:
        """No warning for short benchmarks."""
        result = SmartDefaults.warn_if_long(
            estimated_time=60.0,  # 1 minute
            threshold=300.0,  # 5 minutes
        )

        assert result is True

    def test_suggest_output_path(self) -> None:
        """Output path includes circuit and backend names."""
        path = SmartDefaults.suggest_output_path(
            circuit_name="bell_state",
            backend_name="lret",
        )

        assert "bell_state" in path
        assert "lret" in path
        assert path.startswith("results/")
        assert path.endswith(".json")


class TestProgressConfig:
    """Tests for progress configuration."""

    def test_default_config(self) -> None:
        """Default config has expected values."""
        config = BenchmarkProgressConfig()

        assert config.style == ProgressStyle.DETAILED
        assert config.show_metrics is True
        assert config.refresh_rate == 4.0

    def test_custom_config(self) -> None:
        """Custom config values are applied."""
        config = BenchmarkProgressConfig(
            style=ProgressStyle.MINIMAL,
            show_metrics=False,
            refresh_rate=1.0,
        )

        assert config.style == ProgressStyle.MINIMAL
        assert config.show_metrics is False
        assert config.refresh_rate == 1.0


class TestRealTimeMetricsDisplay:
    """Tests for real-time metrics display."""

    def test_update_metrics(self) -> None:
        """Display updates metrics correctly."""
        display = RealTimeMetricsDisplay()

        display.update({
            "cpu_usage": 45.0,
            "memory_mb": 512.0,
        })

        assert display._metrics["cpu_usage"] == 45.0
        assert display._metrics["memory_mb"] == 512.0

    def test_metrics_accumulate(self) -> None:
        """Multiple updates accumulate metrics."""
        display = RealTimeMetricsDisplay()

        display.update({"metric1": 1.0})
        display.update({"metric2": 2.0})

        assert "metric1" in display._metrics
        assert "metric2" in display._metrics

    def test_context_manager(self) -> None:
        """Display works as context manager."""
        with RealTimeMetricsDisplay() as display:
            display.update({"test": 123})

        # Should not raise
        assert True
