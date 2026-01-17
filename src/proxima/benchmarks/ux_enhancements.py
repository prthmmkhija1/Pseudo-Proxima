"""User experience enhancements for benchmarking CLI (Phase 10.3).

Implements:
- Progress indicators with rich.Progress
- Output formatting with rich.Table and color-coding
- Interactive features (--interactive, --watch modes)
- Helpful defaults based on circuit analysis
"""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich.style import Style

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# =============================================================================
# Phase 10.3.1: Progress Indicators
# =============================================================================


class ProgressStyle(Enum):
    """Visual styles for progress display."""

    MINIMAL = auto()  # Just percentage
    STANDARD = auto()  # Bar with percentage
    DETAILED = auto()  # Bar with ETA, elapsed, stats


@dataclass
class BenchmarkProgressConfig:
    """Configuration for benchmark progress display.

    Attributes:
        style: Progress display style.
        show_metrics: Show real-time metrics during execution.
        refresh_rate: UI refresh rate in Hz.
        show_sparklines: Show mini charts for trends.
    """

    style: ProgressStyle = ProgressStyle.DETAILED
    show_metrics: bool = True
    refresh_rate: float = 4.0
    show_sparklines: bool = True


class BenchmarkProgress:
    """Rich progress display for long-running benchmarks.

    Shows current benchmark, progress bar, ETA, and real-time metrics.

    Example:
        >>> with BenchmarkProgress(total=100) as progress:
        ...     for i, result in enumerate(benchmarks):
        ...         progress.update(
        ...             completed=i + 1,
        ...             current_backend="lret",
        ...             current_circuit="bell_state",
        ...             last_time_ms=result.execution_time_ms,
        ...         )
    """

    def __init__(
        self,
        total: int,
        config: BenchmarkProgressConfig | None = None,
        console: Optional[Any] = None,
    ) -> None:
        self.total = total
        self.config = config or BenchmarkProgressConfig()
        self._console = console if RICH_AVAILABLE else None
        self._progress: Any = None
        self._task_id: Any = None
        self._start_time: float = 0.0
        self._times: List[float] = []
        self._last_metrics: Dict[str, Any] = {}

        if RICH_AVAILABLE and self._console is None:
            self._console = Console()

    def __enter__(self) -> "BenchmarkProgress":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        """Start the progress display."""
        self._start_time = time.perf_counter()

        if not RICH_AVAILABLE:
            print(f"Starting benchmark suite: 0/{self.total}", flush=True)
            return

        # Create progress bar with rich
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            refresh_per_second=self.config.refresh_rate,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            "Running benchmarks...",
            total=self.total,
        )

    def stop(self) -> None:
        """Stop the progress display."""
        if self._progress:
            self._progress.stop()

    def update(
        self,
        completed: int,
        current_backend: str = "",
        current_circuit: str = "",
        last_time_ms: float = 0.0,
        status: str = "running",
    ) -> None:
        """Update progress with current state.

        Args:
            completed: Number of completed benchmarks.
            current_backend: Currently running backend.
            current_circuit: Currently running circuit.
            last_time_ms: Execution time of last benchmark.
            status: Current status (running, success, failed).
        """
        self._times.append(last_time_ms)
        self._last_metrics = {
            "backend": current_backend,
            "circuit": current_circuit,
            "time_ms": last_time_ms,
            "status": status,
        }

        if not RICH_AVAILABLE:
            # Fallback to simple output
            pct = (completed / self.total * 100) if self.total > 0 else 0
            eta = self._estimate_remaining(completed)
            print(
                f"\r[{completed}/{self.total}] {pct:.0f}% | "
                f"{current_backend}:{current_circuit} | "
                f"{last_time_ms:.1f}ms | ETA: {eta}",
                end="",
                flush=True,
            )
            if completed == self.total:
                print()  # New line at end
            return

        # Update rich progress
        description = f"[{current_backend}] {current_circuit}"
        if status == "success":
            description = f"[green]✓[/green] {description}"
        elif status == "failed":
            description = f"[red]✗[/red] {description}"

        self._progress.update(
            self._task_id,
            completed=completed,
            description=description,
        )

    def _estimate_remaining(self, completed: int) -> str:
        """Estimate time remaining."""
        if completed == 0:
            return "calculating..."

        elapsed = time.perf_counter() - self._start_time
        avg_per_item = elapsed / completed
        remaining = (self.total - completed) * avg_per_item

        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining / 60:.1f}m"
        else:
            return f"{remaining / 3600:.1f}h"

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary statistics."""
        elapsed = time.perf_counter() - self._start_time
        return {
            "total": self.total,
            "completed": len(self._times),
            "elapsed_seconds": elapsed,
            "avg_time_ms": sum(self._times) / len(self._times) if self._times else 0,
            "min_time_ms": min(self._times) if self._times else 0,
            "max_time_ms": max(self._times) if self._times else 0,
        }


class RealTimeMetricsDisplay:
    """Real-time updating display for benchmark metrics.

    Shows a continuously updating table with current metrics.
    """

    def __init__(self, console: Optional[Any] = None) -> None:
        self._console = console if RICH_AVAILABLE else None
        self._live: Any = None
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

        if RICH_AVAILABLE and self._console is None:
            self._console = Console()

    def __enter__(self) -> "RealTimeMetricsDisplay":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        """Start the live display."""
        if not RICH_AVAILABLE:
            return

        self._live = Live(
            self._build_table(),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()

    def update(self, metrics: Dict[str, Any]) -> None:
        """Update displayed metrics."""
        with self._lock:
            self._metrics.update(metrics)

        if self._live:
            self._live.update(self._build_table())

    def _build_table(self) -> Any:
        """Build metrics table."""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="Real-time Metrics", expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in self._metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.2f}")
            else:
                table.add_row(key, str(value))

        return table


# =============================================================================
# Phase 10.3.2: Output Formatting
# =============================================================================


class ResultFormatter:
    """Formats benchmark results for terminal display.

    Uses rich.Table for beautiful output with color-coding based
    on performance thresholds.
    """

    # Thresholds for color-coding (ms)
    FAST_THRESHOLD = 50.0  # Below this is green
    SLOW_THRESHOLD = 500.0  # Above this is red

    def __init__(self, console: Optional[Any] = None) -> None:
        self._console = console if RICH_AVAILABLE else None

        if RICH_AVAILABLE and self._console is None:
            self._console = Console()

    def format_single_result(self, result: Any) -> None:
        """Format and display a single benchmark result."""
        if not RICH_AVAILABLE:
            self._format_simple_result(result)
            return

        metrics = result.metrics if hasattr(result, "metrics") else result

        table = Table(title=f"Benchmark: {metrics.backend_name}", expand=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", justify="right", width=20)

        # Add rows with color-coding
        time_style = self._get_time_style(metrics.execution_time_ms)
        table.add_row("Execution time (ms)", Text(f"{metrics.execution_time_ms:.2f}", style=time_style))
        table.add_row("Memory peak (MB)", f"{metrics.memory_peak_mb:.2f}")
        table.add_row("Memory baseline (MB)", f"{metrics.memory_baseline_mb:.2f}")
        table.add_row("Throughput (shots/s)", f"{metrics.throughput_shots_per_sec:.0f}")
        table.add_row("Success rate (%)", self._format_percentage(metrics.success_rate_percent))
        table.add_row("CPU usage (%)", f"{metrics.cpu_usage_percent:.1f}")

        if metrics.gpu_usage_percent is not None:
            table.add_row("GPU usage (%)", f"{metrics.gpu_usage_percent:.1f}")

        self._console.print(table)

    def format_comparison(
        self,
        results: List[Any],
        winner: str | None = None,
        speedup_factors: Dict[str, float] | None = None,
    ) -> None:
        """Format and display backend comparison results."""
        if not results:
            print("No results to display")
            return

        if not RICH_AVAILABLE:
            self._format_simple_comparison(results, winner, speedup_factors)
            return

        table = Table(title="Backend Comparison", expand=True)
        table.add_column("Backend", style="cyan")
        table.add_column("Avg (ms)", justify="right")
        table.add_column("Min (ms)", justify="right")
        table.add_column("Max (ms)", justify="right")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Speedup", justify="right")

        # Sort by execution time
        sorted_results = sorted(
            results,
            key=lambda r: r.metrics.execution_time_ms if r.metrics else float("inf"),
        )

        for result in sorted_results:
            metrics = result.metrics
            backend = metrics.backend_name

            # Highlight winner
            if backend == winner:
                backend_text = Text(f"★ {backend}", style="bold green")
            else:
                backend_text = Text(backend)

            # Color-code time
            time_style = self._get_time_style(metrics.execution_time_ms)

            # Get speedup
            speedup = speedup_factors.get(backend, 1.0) if speedup_factors else 1.0
            speedup_text = f"{speedup:.2f}x"

            table.add_row(
                backend_text,
                Text(f"{metrics.execution_time_ms:.2f}", style=time_style),
                f"{metrics.execution_time_ms:.2f}",  # Placeholder for min
                f"{metrics.execution_time_ms:.2f}",  # Placeholder for max
                f"{metrics.memory_peak_mb:.1f}",
                speedup_text,
            )

        self._console.print(table)

        if winner:
            self._console.print(
                Panel(
                    f"[bold green]✓ Winner: {winner}[/bold green]",
                    expand=False,
                )
            )

    def format_statistics(
        self,
        stats: Dict[str, Any],
        backend_name: str,
        trend_direction: str | None = None,
    ) -> None:
        """Format and display statistics summary."""
        if not RICH_AVAILABLE:
            self._format_simple_stats(stats, backend_name)
            return

        table = Table(title=f"Statistics: {backend_name}", expand=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", justify="right", width=20)

        for key, value in stats.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.2f}")
            else:
                table.add_row(key, str(value))

        # Add trend indicator
        if trend_direction:
            trend_style = (
                "green" if trend_direction == "decreasing"
                else "red" if trend_direction == "increasing"
                else "yellow"
            )
            trend_icon = "↓" if trend_direction == "decreasing" else "↑" if trend_direction == "increasing" else "→"
            table.add_row(
                "Trend",
                Text(f"{trend_icon} {trend_direction}", style=trend_style),
            )

        self._console.print(table)

    def _get_time_style(self, time_ms: float) -> str:
        """Get style based on execution time."""
        if time_ms < self.FAST_THRESHOLD:
            return "bold green"
        elif time_ms > self.SLOW_THRESHOLD:
            return "bold red"
        return "yellow"

    def _format_percentage(self, value: float) -> Any:
        """Format percentage with color coding."""
        if not RICH_AVAILABLE:
            return f"{value:.1f}"

        if value >= 100.0:
            return Text(f"{value:.1f}", style="bold green")
        elif value >= 90.0:
            return Text(f"{value:.1f}", style="yellow")
        else:
            return Text(f"{value:.1f}", style="bold red")

    def _format_simple_result(self, result: Any) -> None:
        """Fallback formatting without rich."""
        metrics = result.metrics if hasattr(result, "metrics") else result
        print(f"\n=== Benchmark: {metrics.backend_name} ===")
        print(f"Execution time: {metrics.execution_time_ms:.2f} ms")
        print(f"Memory peak: {metrics.memory_peak_mb:.2f} MB")
        print(f"Throughput: {metrics.throughput_shots_per_sec:.0f} shots/s")
        print(f"Success rate: {metrics.success_rate_percent:.1f}%")

    def _format_simple_comparison(
        self,
        results: List[Any],
        winner: str | None,
        speedup_factors: Dict[str, float] | None,
    ) -> None:
        """Fallback comparison formatting without rich."""
        print("\n=== Backend Comparison ===")
        print(f"{'Backend':<15} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
        print("-" * 50)

        for result in sorted(results, key=lambda r: r.metrics.execution_time_ms):
            m = result.metrics
            speedup = speedup_factors.get(m.backend_name, 1.0) if speedup_factors else 1.0
            marker = "* " if m.backend_name == winner else "  "
            print(f"{marker}{m.backend_name:<13} {m.execution_time_ms:<12.2f} {m.memory_peak_mb:<12.1f} {speedup:.2f}x")

        if winner:
            print(f"\n✓ Winner: {winner}")

    def _format_simple_stats(self, stats: Dict[str, Any], backend_name: str) -> None:
        """Fallback stats formatting without rich."""
        print(f"\n=== Statistics: {backend_name} ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


class Sparkline:
    """Text-based sparkline for time series visualization.

    Creates a mini-chart using Unicode block characters.
    """

    BLOCKS = " ▁▂▃▄▅▆▇█"

    @classmethod
    def render(cls, values: Sequence[float], width: int = 20) -> str:
        """Render a sparkline for the given values.

        Args:
            values: Sequence of numeric values.
            width: Maximum width of sparkline.

        Returns:
            String containing sparkline characters.
        """
        if not values:
            return ""

        # Resample to fit width
        if len(values) > width:
            step = len(values) / width
            resampled = [values[int(i * step)] for i in range(width)]
        else:
            resampled = list(values)

        # Normalize to 0-8 range
        min_v = min(resampled)
        max_v = max(resampled)
        range_v = max_v - min_v if max_v > min_v else 1

        normalized = [(v - min_v) / range_v * 8 for v in resampled]
        chars = [cls.BLOCKS[min(8, max(0, int(n)))] for n in normalized]

        return "".join(chars)


# =============================================================================
# Phase 10.3.3: Interactive Features
# =============================================================================


class InteractiveMode:
    """Interactive benchmark mode that asks before each backend.

    Allows users to selectively run benchmarks and skip backends.
    """

    def __init__(self, console: Optional[Any] = None) -> None:
        self._console = console if RICH_AVAILABLE else None

        if RICH_AVAILABLE and self._console is None:
            self._console = Console()

    def confirm_backend(
        self,
        backend_name: str,
        circuit_name: str,
        estimated_time: float | None = None,
    ) -> bool:
        """Ask user to confirm running benchmark on a backend.

        Args:
            backend_name: Name of the backend.
            circuit_name: Name of the circuit.
            estimated_time: Estimated execution time in seconds.

        Returns:
            True if user confirms, False to skip.
        """
        msg = f"Run benchmark on '{backend_name}' for '{circuit_name}'?"
        if estimated_time:
            msg += f" (estimated: {estimated_time:.1f}s)"

        if RICH_AVAILABLE:
            self._console.print(f"\n[cyan]{msg}[/cyan]")
            response = self._console.input("[Y/n/q] ").strip().lower()
        else:
            response = input(f"\n{msg} [Y/n/q] ").strip().lower()

        if response == "q":
            raise KeyboardInterrupt("User requested quit")

        return response != "n"

    def select_backends(self, available_backends: List[str]) -> List[str]:
        """Let user select which backends to benchmark.

        Args:
            available_backends: List of available backend names.

        Returns:
            List of selected backend names.
        """
        if RICH_AVAILABLE:
            self._console.print("\n[bold]Available backends:[/bold]")
            for i, backend in enumerate(available_backends, 1):
                self._console.print(f"  {i}. {backend}")

            self._console.print("\nEnter numbers separated by commas, or 'all':")
            response = self._console.input("> ").strip().lower()
        else:
            print("\nAvailable backends:")
            for i, backend in enumerate(available_backends, 1):
                print(f"  {i}. {backend}")

            response = input("\nEnter numbers or 'all': ").strip().lower()

        if response == "all" or not response:
            return available_backends

        try:
            indices = [int(x.strip()) - 1 for x in response.split(",")]
            return [available_backends[i] for i in indices if 0 <= i < len(available_backends)]
        except (ValueError, IndexError):
            return available_backends


class WatchMode:
    """Watch mode for continuously updating benchmark display.

    Runs benchmarks in a loop and updates display in real-time.
    """

    def __init__(
        self,
        interval_seconds: float = 60.0,
        console: Optional[Any] = None,
    ) -> None:
        self._interval = interval_seconds
        self._console = console if RICH_AVAILABLE else None
        self._running = False
        self._results_history: List[Dict[str, Any]] = []

        if RICH_AVAILABLE and self._console is None:
            self._console = Console()

    def start(
        self,
        benchmark_fn: Callable[[], Any],
        max_iterations: int | None = None,
    ) -> None:
        """Start watch mode.

        Args:
            benchmark_fn: Function to run for each benchmark.
            max_iterations: Maximum number of iterations (None for infinite).
        """
        self._running = True
        iteration = 0

        try:
            while self._running:
                if max_iterations and iteration >= max_iterations:
                    break

                iteration += 1
                timestamp = datetime.now()

                # Run benchmark
                try:
                    result = benchmark_fn()
                    self._results_history.append({
                        "iteration": iteration,
                        "timestamp": timestamp,
                        "result": result,
                        "success": True,
                    })
                except Exception as e:
                    self._results_history.append({
                        "iteration": iteration,
                        "timestamp": timestamp,
                        "error": str(e),
                        "success": False,
                    })

                # Display update
                self._display_update(iteration)

                # Wait for next iteration
                if self._running:
                    time.sleep(self._interval)

        except KeyboardInterrupt:
            self._running = False

    def stop(self) -> None:
        """Stop watch mode."""
        self._running = False

    def _display_update(self, iteration: int) -> None:
        """Display current watch mode status."""
        if RICH_AVAILABLE:
            self._console.clear()
            self._console.print(
                Panel(
                    f"Watch Mode - Iteration {iteration}\n"
                    f"Interval: {self._interval}s | Press Ctrl+C to stop",
                    title="Benchmark Watch",
                )
            )

            # Show recent results
            if self._results_history:
                table = Table(title="Recent Results")
                table.add_column("#", width=5)
                table.add_column("Time")
                table.add_column("Status")

                for entry in self._results_history[-10:]:
                    status = "✓" if entry["success"] else "✗"
                    style = "green" if entry["success"] else "red"
                    table.add_row(
                        str(entry["iteration"]),
                        entry["timestamp"].strftime("%H:%M:%S"),
                        Text(status, style=style),
                    )

                self._console.print(table)
        else:
            print(f"\n--- Watch Mode: Iteration {iteration} ---")


# =============================================================================
# Phase 10.3.4: Helpful Defaults
# =============================================================================


class SmartDefaults:
    """Provides intelligent defaults based on circuit analysis.

    Suggests optimal parameters for benchmarking based on
    circuit characteristics.
    """

    # Base run counts by circuit size
    RUNS_BY_QUBITS = {
        (0, 10): 10,   # Small circuits: more runs for accuracy
        (10, 20): 5,   # Medium circuits
        (20, 30): 3,   # Large circuits
        (30, 50): 2,   # Very large circuits
    }

    # Backend suggestions by characteristics
    GPU_PREFERRED_QUBITS = 15  # Prefer GPU backends above this
    STATEVECTOR_MAX_QUBITS = 25  # Use state vector up to this

    @classmethod
    def suggest_runs(cls, num_qubits: int, circuit_depth: int | None = None) -> int:
        """Suggest number of benchmark runs based on circuit size.

        Args:
            num_qubits: Number of qubits in the circuit.
            circuit_depth: Optional circuit depth.

        Returns:
            Suggested number of runs.
        """
        for (min_q, max_q), runs in cls.RUNS_BY_QUBITS.items():
            if min_q <= num_qubits < max_q:
                return runs
        return 1  # Very large circuits

    @classmethod
    def suggest_backends(
        cls,
        num_qubits: int,
        available_backends: List[str],
        has_gpu: bool = False,
    ) -> List[str]:
        """Suggest backends based on circuit characteristics.

        Args:
            num_qubits: Number of qubits in the circuit.
            available_backends: List of available backends.
            has_gpu: Whether GPU is available.

        Returns:
            Ordered list of suggested backends (best first).
        """
        suggestions: List[str] = []

        # GPU backends for large circuits
        if num_qubits >= cls.GPU_PREFERRED_QUBITS and has_gpu:
            gpu_backends = ["cuquantum", "qsim"]  # GPU-capable
            for b in gpu_backends:
                if b in available_backends:
                    suggestions.append(b)

        # Fast CPU backends
        cpu_fast = ["qsim", "quest", "lret"]
        for b in cpu_fast:
            if b in available_backends and b not in suggestions:
                suggestions.append(b)

        # Other backends
        for b in available_backends:
            if b not in suggestions:
                suggestions.append(b)

        return suggestions

    @classmethod
    def estimate_time(
        cls,
        num_qubits: int,
        shots: int,
        backend_name: str,
    ) -> float:
        """Estimate benchmark execution time in seconds.

        Args:
            num_qubits: Number of qubits.
            shots: Number of measurement shots.
            backend_name: Backend to use.

        Returns:
            Estimated time in seconds.
        """
        # Base time scales exponentially with qubits
        base_time = 0.001 * (2 ** (num_qubits / 5))

        # Scale by shots
        shot_factor = shots / 1024

        # Backend-specific multipliers (rough estimates)
        backend_multipliers = {
            "cuquantum": 0.5,
            "qsim": 0.7,
            "quest": 1.0,
            "lret": 1.2,
            "cirq": 1.5,
            "qiskit": 2.0,
        }
        multiplier = backend_multipliers.get(backend_name, 1.0)

        return base_time * shot_factor * multiplier

    @classmethod
    def warn_if_long(
        cls,
        estimated_time: float,
        threshold: float = 300.0,  # 5 minutes
        console: Optional[Any] = None,
    ) -> bool:
        """Warn user if benchmark will take a long time.

        Args:
            estimated_time: Estimated time in seconds.
            threshold: Warning threshold in seconds.
            console: Optional rich Console.

        Returns:
            True if user confirms to continue, False otherwise.
        """
        if estimated_time <= threshold:
            return True

        time_str = (
            f"{estimated_time / 60:.1f} minutes"
            if estimated_time < 3600
            else f"{estimated_time / 3600:.1f} hours"
        )

        warning = f"Benchmark estimated to take {time_str}. Continue?"

        if RICH_AVAILABLE and console:
            console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {warning}")
            response = console.input("[Y/n] ").strip().lower()
        else:
            print(f"Warning: {warning}")
            response = input("[Y/n] ").strip().lower()

        return response != "n"

    @classmethod
    def suggest_output_path(cls, circuit_name: str, backend_name: str) -> str:
        """Suggest a default output path for results.

        Args:
            circuit_name: Name of the circuit.
            backend_name: Name of the backend.

        Returns:
            Suggested output path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"results/benchmark_{circuit_name}_{backend_name}_{timestamp}.json"
