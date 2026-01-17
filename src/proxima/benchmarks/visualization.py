"""Data preparation helpers for plotting benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import pandas as pd

from proxima.data.metrics import BenchmarkComparison, BenchmarkResult
from proxima.data.benchmark_registry import BenchmarkRegistry


class ChartType(str, Enum):
    """Supported chart types for visualization."""
    
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BOX = "box"
    HISTOGRAM = "histogram"
    AREA = "area"
    PIE = "pie"


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    
    chart_type: ChartType = ChartType.BAR
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    width: int = 800
    height: int = 600
    color_palette: str = "viridis"
    show_legend: bool = True
    show_grid: bool = True
    theme: str = "default"


@dataclass
class ChartData:
    """Prepared data for chart rendering."""
    
    data: pd.DataFrame
    config: ChartConfig
    x_column: str = ""
    y_column: str = ""
    group_column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualizationDataBuilder:
    """Prepare pandas DataFrames for charts and visualizations."""

    def __init__(self, registry: BenchmarkRegistry | None = None) -> None:
        self.registry = registry

    # ------------------------------------------------------------------
    # Time series data
    # ------------------------------------------------------------------
    def build_time_series(self, results: List[BenchmarkResult]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for r in results:
            if not r.metrics:
                continue
            rows.append(
                {
                    "timestamp": r.metrics.timestamp,
                    "execution_time_ms": r.metrics.execution_time_ms,
                    "backend_name": r.metrics.backend_name,
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="timestamp")
        return df

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    def build_comparison_table(self, comparison: BenchmarkComparison) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for res in comparison.results:
            if not res.metrics:
                continue
            stats = res.metadata.get("statistics", {}) if res.metadata else {}
            rows.append(
                {
                    "backend": res.metrics.backend_name,
                    "avg_time_ms": res.metrics.execution_time_ms,
                    "min_time_ms": stats.get("min_time_ms", res.metrics.execution_time_ms),
                    "max_time_ms": stats.get("max_time_ms", res.metrics.execution_time_ms),
                    "speedup": comparison.speedup_factors.get(res.metrics.backend_name, 0.0),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="avg_time_ms")
        return df

    # ------------------------------------------------------------------
    # Distribution data
    # ------------------------------------------------------------------
    def build_distribution_data(self, results: List[BenchmarkResult]) -> Dict[str, List[float]]:
        grouped: dict[str, list[float]] = {}
        for r in results:
            if not r.metrics:
                continue
            backend = r.metrics.backend_name
            grouped.setdefault(backend, []).append(r.metrics.execution_time_ms)
        return grouped

    # ------------------------------------------------------------------
    # Heatmap data
    # ------------------------------------------------------------------
    def build_performance_heatmap(self, backend_name: str) -> pd.DataFrame:
        """Build 2D grid: qubits (rows) vs depth (columns) with avg execution times."""
        if self.registry is None:
            raise ValueError("BenchmarkRegistry is required for heatmap generation")
        results = self.registry.get_results_for_backend(backend_name, limit=None)
        rows: list[dict[str, Any]] = []
        for r in results:
            if not r.metrics or not r.metrics.circuit_info:
                continue
            q = r.metrics.circuit_info.get("qubit_count")
            d = r.metrics.circuit_info.get("depth")
            if q is None or d is None:
                continue
            try:
                rows.append(
                    {
                        "qubits": int(q),
                        "depth": int(d),
                        "execution_time_ms": float(r.metrics.execution_time_ms),
                    }
                )
            except (TypeError, ValueError):
                continue  # Skip malformed entries
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        pivot = df.pivot_table(
            index="qubits",
            columns="depth",
            values="execution_time_ms",
            aggfunc="mean",
        )
        return pivot
    
    # =========================================================================
    # Extended Visualization Methods (Feature - Benchmarks)
    # =========================================================================
    
    def build_speedup_chart_data(
        self,
        comparison: BenchmarkComparison,
    ) -> ChartData:
        """Build data for speedup factor bar chart.
        
        Args:
            comparison: Benchmark comparison results.
            
        Returns:
            ChartData ready for rendering.
        """
        speedups = comparison.speedup_factors
        df = pd.DataFrame([
            {"backend": name, "speedup": factor}
            for name, factor in sorted(speedups.items(), key=lambda x: x[1])
        ])
        
        config = ChartConfig(
            chart_type=ChartType.BAR,
            title="Backend Speedup Comparison",
            x_label="Backend",
            y_label="Relative Time (1.0 = fastest)",
        )
        
        return ChartData(
            data=df,
            config=config,
            x_column="backend",
            y_column="speedup",
            metadata={"winner": comparison.winner},
        )
    
    def build_scaling_chart_data(
        self,
        backend_name: str,
        metric: str = "execution_time_ms",
    ) -> ChartData:
        """Build data for qubit scaling line chart.
        
        Args:
            backend_name: Backend to analyze.
            metric: Metric to plot.
            
        Returns:
            ChartData for scaling visualization.
        """
        if self.registry is None:
            raise ValueError("BenchmarkRegistry required")
        
        results = self.registry.get_results_for_backend(backend_name, limit=None)
        
        # Group by qubit count
        data: dict[int, list[float]] = {}
        for r in results:
            if not r.metrics or not r.metrics.circuit_info:
                continue
            qubits = r.metrics.circuit_info.get("qubit_count")
            if qubits is None:
                continue
            data.setdefault(int(qubits), []).append(r.metrics.execution_time_ms)
        
        # Calculate averages
        rows = [
            {"qubits": q, "avg_time_ms": sum(times) / len(times)}
            for q, times in sorted(data.items())
        ]
        df = pd.DataFrame(rows)
        
        config = ChartConfig(
            chart_type=ChartType.LINE,
            title=f"{backend_name} Qubit Scaling",
            x_label="Qubits",
            y_label="Avg Execution Time (ms)",
        )
        
        return ChartData(
            data=df,
            config=config,
            x_column="qubits",
            y_column="avg_time_ms",
        )
    
    def build_box_plot_data(
        self,
        results: List[BenchmarkResult],
    ) -> ChartData:
        """Build data for box plot of execution times by backend.
        
        Args:
            results: Benchmark results to visualize.
            
        Returns:
            ChartData for box plot.
        """
        rows = []
        for r in results:
            if not r.metrics:
                continue
            rows.append({
                "backend": r.metrics.backend_name,
                "execution_time_ms": r.metrics.execution_time_ms,
            })
        
        df = pd.DataFrame(rows)
        
        config = ChartConfig(
            chart_type=ChartType.BOX,
            title="Execution Time Distribution by Backend",
            x_label="Backend",
            y_label="Execution Time (ms)",
        )
        
        return ChartData(
            data=df,
            config=config,
            x_column="backend",
            y_column="execution_time_ms",
        )
    
    def build_memory_chart_data(
        self,
        backend_name: str,
    ) -> ChartData:
        """Build data for memory usage chart.
        
        Args:
            backend_name: Backend to analyze.
            
        Returns:
            ChartData for memory visualization.
        """
        if self.registry is None:
            raise ValueError("BenchmarkRegistry required")
        
        results = self.registry.get_results_for_backend(backend_name, limit=None)
        
        rows = []
        for r in results:
            if not r.metrics or not r.metrics.circuit_info:
                continue
            qubits = r.metrics.circuit_info.get("qubit_count")
            if qubits is None:
                continue
            rows.append({
                "qubits": int(qubits),
                "memory_mb": r.metrics.memory_peak_mb,
                "timestamp": r.metrics.timestamp,
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("qubits")
        
        config = ChartConfig(
            chart_type=ChartType.SCATTER,
            title=f"{backend_name} Memory Usage",
            x_label="Qubits",
            y_label="Peak Memory (MB)",
        )
        
        return ChartData(
            data=df,
            config=config,
            x_column="qubits",
            y_column="memory_mb",
        )
    
    def build_trend_chart_data(
        self,
        backend_name: str,
        days: int = 30,
    ) -> ChartData:
        """Build data for performance trend over time.
        
        Args:
            backend_name: Backend to analyze.
            days: Number of days to include.
            
        Returns:
            ChartData for trend visualization.
        """
        if self.registry is None:
            raise ValueError("BenchmarkRegistry required")
        
        results = self.registry.get_results_for_backend(backend_name, limit=None)
        
        rows = []
        for r in results:
            if not r.metrics:
                continue
            rows.append({
                "timestamp": r.metrics.timestamp,
                "execution_time_ms": r.metrics.execution_time_ms,
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            # Resample to daily averages
            df = df.set_index("timestamp").resample("D").mean().reset_index()
            df = df.dropna()
        
        config = ChartConfig(
            chart_type=ChartType.AREA,
            title=f"{backend_name} Performance Trend",
            x_label="Date",
            y_label="Avg Execution Time (ms)",
        )
        
        return ChartData(
            data=df,
            config=config,
            x_column="timestamp",
            y_column="execution_time_ms",
        )
    
    def build_histogram_data(
        self,
        results: List[BenchmarkResult],
        bins: int = 20,
    ) -> ChartData:
        """Build data for execution time histogram.
        
        Args:
            results: Benchmark results.
            bins: Number of histogram bins.
            
        Returns:
            ChartData for histogram.
        """
        times = [
            r.metrics.execution_time_ms
            for r in results
            if r.metrics
        ]
        
        df = pd.DataFrame({"execution_time_ms": times})
        
        config = ChartConfig(
            chart_type=ChartType.HISTOGRAM,
            title="Execution Time Distribution",
            x_label="Execution Time (ms)",
            y_label="Frequency",
        )
        
        return ChartData(
            data=df,
            config=config,
            x_column="execution_time_ms",
            y_column="",
            metadata={"bins": bins},
        )
    
    def export_to_csv(
        self,
        results: List[BenchmarkResult],
        path: str,
    ) -> None:
        """Export benchmark results to CSV.
        
        Args:
            results: Results to export.
            path: Output file path.
        """
        df = self.build_time_series(results)
        df.to_csv(path, index=False)
    
    def export_to_json(
        self,
        results: List[BenchmarkResult],
        path: str,
    ) -> None:
        """Export benchmark results to JSON.
        
        Args:
            results: Results to export.
            path: Output file path.
        """
        import json
        
        data = [r.to_dict() for r in results if hasattr(r, 'to_dict')]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# =============================================================================
# Rich Console Visualization (Feature - Benchmarks)
# =============================================================================


class ConsoleVisualizer:
    """Generate ASCII charts and tables for console output."""
    
    def __init__(self) -> None:
        try:
            from rich.console import Console
            self._console = Console()
            self._rich_available = True
        except ImportError:
            self._console = None
            self._rich_available = False
    
    def print_comparison_table(
        self,
        comparison: BenchmarkComparison,
    ) -> None:
        """Print comparison results as a rich table."""
        if not self._rich_available:
            self._print_simple_table(comparison)
            return
        
        from rich.table import Table
        
        table = Table(title="Backend Comparison")
        table.add_column("Backend", style="cyan")
        table.add_column("Avg Time (ms)", justify="right")
        table.add_column("Speedup", justify="right")
        table.add_column("Winner", justify="center")
        
        for result in comparison.results:
            if not result.metrics:
                continue
            backend = result.metrics.backend_name
            time_ms = result.metrics.execution_time_ms
            speedup = comparison.speedup_factors.get(backend, 0)
            is_winner = "✓" if backend == comparison.winner else ""
            
            table.add_row(
                backend,
                f"{time_ms:.2f}",
                f"{speedup:.2f}x",
                is_winner,
            )
        
        self._console.print(table)
    
    def print_bar_chart(
        self,
        data: dict[str, float],
        title: str = "",
        max_width: int = 40,
    ) -> None:
        """Print a simple ASCII bar chart."""
        if not data:
            return
        
        max_val = max(data.values())
        
        print(f"\n{title}")
        print("-" * (max_width + 20))
        
        for label, value in sorted(data.items(), key=lambda x: x[1]):
            bar_len = int((value / max_val) * max_width) if max_val > 0 else 0
            bar = "█" * bar_len
            print(f"{label:15} | {bar} {value:.2f}")
        
        print()
    
    def _print_simple_table(self, comparison: BenchmarkComparison) -> None:
        """Print table without rich library."""
        print("\nBackend Comparison")
        print("-" * 60)
        print(f"{'Backend':15} | {'Avg Time (ms)':>15} | {'Speedup':>10} | Winner")
        print("-" * 60)
        
        for result in comparison.results:
            if not result.metrics:
                continue
            backend = result.metrics.backend_name
            time_ms = result.metrics.execution_time_ms
            speedup = comparison.speedup_factors.get(backend, 0)
            is_winner = "✓" if backend == comparison.winner else ""
            
            print(f"{backend:15} | {time_ms:>15.2f} | {speedup:>9.2f}x | {is_winner}")
        
        print()


__all__ = [
    "VisualizationDataBuilder",
    "ConsoleVisualizer",
    "ChartType",
    "ChartConfig",
    "ChartData",
]

