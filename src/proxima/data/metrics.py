"""Benchmark metrics and result data structures.

Implements core benchmarking data models used to record and aggregate
performance measurements across Proxima backends.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class BenchmarkStatus(str, Enum):
    """Lifecycle states for a benchmark execution."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RUNNING = "running"

    def is_terminal(self) -> bool:
        """Return True when the benchmark has reached a final state."""
        return self in {
            BenchmarkStatus.SUCCESS,
            BenchmarkStatus.FAILED,
            BenchmarkStatus.TIMEOUT,
            BenchmarkStatus.CANCELLED,
        }


@dataclass
class BenchmarkMetrics:
    """Measured metrics for a single benchmark run.

    Captures all performance measurements collected during benchmark execution.

    Attributes:
        execution_time_ms: Total wall-clock execution time in milliseconds.
        memory_peak_mb: Maximum memory usage during execution.
        memory_baseline_mb: Memory usage before execution started.
        throughput_shots_per_sec: Number of measurement shots per second.
        success_rate_percent: Percentage of successful runs (0-100).
        cpu_usage_percent: Average CPU utilization during execution.
        gpu_usage_percent: GPU utilization (None if no GPU used).
        timestamp: When the benchmark was executed.
        backend_name: Name of the backend used.
        circuit_info: Additional circuit metadata (qubits, gates, depth).

    Example:
        >>> metrics = BenchmarkMetrics(
        ...     execution_time_ms=15.5, memory_peak_mb=128.0,
        ...     memory_baseline_mb=64.0, throughput_shots_per_sec=66000.0,
        ...     success_rate_percent=100.0, cpu_usage_percent=45.0,
        ...     gpu_usage_percent=None, timestamp=datetime.now(),
        ...     backend_name="lret"
        ... )
    """

    execution_time_ms: float
    memory_peak_mb: float
    memory_baseline_mb: float
    throughput_shots_per_sec: float
    success_rate_percent: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float]
    timestamp: datetime
    backend_name: str
    circuit_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to a JSON-serializable dict."""
        return {
            "execution_time_ms": self.execution_time_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_baseline_mb": self.memory_baseline_mb,
            "throughput_shots_per_sec": self.throughput_shots_per_sec,
            "success_rate_percent": self.success_rate_percent,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gpu_usage_percent": self.gpu_usage_percent,
            "timestamp": self.timestamp.isoformat(),
            "backend_name": self.backend_name,
            "circuit_info": self.circuit_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkMetrics:
        """Create BenchmarkMetrics from a dictionary representation."""
        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            timestamp = datetime.fromisoformat(raw_ts)
        elif isinstance(raw_ts, datetime):
            timestamp = raw_ts
        else:
            timestamp = datetime.utcnow()

        get = data.get  # Local reference for faster lookups
        return cls(
            execution_time_ms=float(get("execution_time_ms") or 0.0),
            memory_peak_mb=float(get("memory_peak_mb") or 0.0),
            memory_baseline_mb=float(get("memory_baseline_mb") or 0.0),
            throughput_shots_per_sec=float(get("throughput_shots_per_sec") or 0.0),
            success_rate_percent=float(get("success_rate_percent") or 0.0),
            cpu_usage_percent=float(get("cpu_usage_percent") or 0.0),
            gpu_usage_percent=cls._parse_optional_float(get("gpu_usage_percent")),
            timestamp=timestamp,
            backend_name=str(get("backend_name") or ""),
            circuit_info=dict(get("circuit_info") or {}),
        )

    @staticmethod
    def _parse_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


@dataclass
class BenchmarkResult:
    """Encapsulates a benchmark execution outcome."""

    benchmark_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    circuit_hash: str = ""
    metrics: BenchmarkMetrics | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: BenchmarkStatus = BenchmarkStatus.RUNNING
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the benchmark result to a dict."""
        return {
            "benchmark_id": self.benchmark_id,
            "circuit_hash": self.circuit_hash,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "metadata": self.metadata,
            "status": self.status.value,
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        """Serialize result to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkResult:
        """Create BenchmarkResult from a dictionary."""
        metrics_raw = data.get("metrics")
        metrics = (
            BenchmarkMetrics.from_dict(metrics_raw)
            if isinstance(metrics_raw, dict)
            else None
        )
        status_raw = data.get("status", BenchmarkStatus.RUNNING)
        status = (
            BenchmarkStatus(status_raw)
            if isinstance(status_raw, str)
            else BenchmarkStatus.RUNNING
        )
        return cls(
            benchmark_id=str(data.get("benchmark_id", str(uuid.uuid4()))),
            circuit_hash=str(data.get("circuit_hash", "")),
            metrics=metrics,
            metadata=dict(data.get("metadata", {})),
            status=status,
            error_message=data.get("error_message"),
        )

    @classmethod
    def from_json(cls, payload: str) -> BenchmarkResult:
        """Deserialize a BenchmarkResult from JSON."""
        return cls.from_dict(json.loads(payload))


@dataclass
class BenchmarkComparison:
    """Comparison of benchmark results across multiple backends."""

    circuit_description: str
    results: List[BenchmarkResult] = field(default_factory=list)
    winner: Optional[str] = None
    speedup_factors: Dict[str, float] = field(default_factory=dict)

    def generate_report(self) -> str:
        """Generate a concise text report of comparison outcomes."""
        lines: List[str] = [f"Benchmark comparison for: {self.circuit_description}"]
        if self.winner:
            lines.append(f"Fastest backend: {self.winner}")
        if self.speedup_factors:
            lines.append("Speedup factors vs baseline:")
            for backend, factor in self.speedup_factors.items():
                lines.append(f"  - {backend}: {factor:.2f}x")
        for result in self.results:
            status = result.status.value
            backend = result.metrics.backend_name if result.metrics else "unknown"
            time_ms = (
                f"{result.metrics.execution_time_ms:.2f} ms"
                if result.metrics
                else "n/a"
            )
            lines.append(f"  * {backend}: {time_ms} ({status})")
        return "\n".join(lines)


__all__ = [
    "BenchmarkStatus",
    "BenchmarkMetrics",
    "BenchmarkResult",
    "BenchmarkComparison",
    "MetricsAggregator",
    "PerformanceProfiler",
    "MetricsExporter",
]


# =============================================================================
# Metrics Aggregator - Statistical Analysis
# =============================================================================


class MetricsAggregator:
    """Aggregates and analyzes multiple benchmark metrics.

    Provides statistical analysis across multiple runs including:
    - Mean, median, standard deviation
    - Percentiles (p50, p90, p95, p99)
    - Trend analysis
    - Outlier detection
    """

    def __init__(self, results: List[BenchmarkResult] | None = None) -> None:
        """Initialize the aggregator.

        Args:
            results: Optional initial list of results.
        """
        self._results: List[BenchmarkResult] = results or []

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a result to the aggregator."""
        self._results.append(result)

    def add_results(self, results: List[BenchmarkResult]) -> None:
        """Add multiple results."""
        self._results.extend(results)

    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()

    @property
    def count(self) -> int:
        """Get number of results."""
        return len(self._results)

    def get_execution_times(self) -> List[float]:
        """Get list of execution times from successful results."""
        return [
            r.metrics.execution_time_ms
            for r in self._results
            if r.metrics and r.status == BenchmarkStatus.SUCCESS
        ]

    def get_memory_usage(self) -> List[float]:
        """Get list of peak memory usage."""
        return [
            r.metrics.memory_peak_mb
            for r in self._results
            if r.metrics and r.status == BenchmarkStatus.SUCCESS
        ]

    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute statistical measures for a list of values.

        Args:
            values: List of numeric values.

        Returns:
            Dictionary of statistical measures.
        """
        import statistics

        if not values:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "stdev": 0.0,
                "variance": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def percentile(p: float) -> float:
            k = (n - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if n > 1 else 0.0,
            "variance": statistics.variance(values) if n > 1 else 0.0,
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
        }

    def execution_time_stats(self) -> Dict[str, float]:
        """Get execution time statistics."""
        return self.compute_statistics(self.get_execution_times())

    def memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return self.compute_statistics(self.get_memory_usage())

    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if not self._results:
            return 0.0
        successful = sum(1 for r in self._results if r.status == BenchmarkStatus.SUCCESS)
        return (successful / len(self._results)) * 100

    def detect_outliers(
        self,
        values: List[float],
        threshold: float = 2.0,
    ) -> List[tuple[int, float]]:
        """Detect outliers using Z-score method.

        Args:
            values: List of values to check.
            threshold: Z-score threshold for outlier detection.

        Returns:
            List of (index, value) tuples for outliers.
        """
        import statistics

        if len(values) < 3:
            return []

        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        
        if stdev == 0:
            return []

        outliers = []
        for i, val in enumerate(values):
            z_score = abs(val - mean) / stdev
            if z_score > threshold:
                outliers.append((i, val))

        return outliers

    def analyze_trend(
        self,
        values: List[float],
    ) -> Dict[str, Any]:
        """Analyze trend in values over time.

        Args:
            values: Time-ordered list of values.

        Returns:
            Trend analysis results.
        """
        if len(values) < 2:
            return {
                "trend": "insufficient_data",
                "slope": 0.0,
                "r_squared": 0.0,
            }

        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return {"trend": "flat", "slope": 0.0, "r_squared": 0.0}

        slope = numerator / denominator

        # Calculate R-squared
        y_pred = [slope * i + (y_mean - slope * x_mean) for i in range(n)]
        ss_res = sum((v - p) ** 2 for v, p in zip(values, y_pred))
        ss_tot = sum((v - y_mean) ** 2 for v in values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine trend direction
        if abs(slope) < 0.01 * y_mean:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            "trend": trend,
            "slope": slope,
            "r_squared": r_squared,
            "prediction_next": slope * n + (y_mean - slope * x_mean),
        }

    def group_by_backend(self) -> Dict[str, List[BenchmarkResult]]:
        """Group results by backend name."""
        groups: Dict[str, List[BenchmarkResult]] = {}
        for result in self._results:
            if result.metrics:
                backend = result.metrics.backend_name
                if backend not in groups:
                    groups[backend] = []
                groups[backend].append(result)
        return groups

    def compare_backends(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across backends.

        Returns:
            Dictionary mapping backend names to their statistics.
        """
        groups = self.group_by_backend()
        comparison = {}

        for backend, results in groups.items():
            times = [r.metrics.execution_time_ms for r in results if r.metrics]
            memory = [r.metrics.memory_peak_mb for r in results if r.metrics]
            
            comparison[backend] = {
                "count": len(results),
                "avg_time_ms": sum(times) / len(times) if times else 0,
                "avg_memory_mb": sum(memory) / len(memory) if memory else 0,
                "success_rate": sum(1 for r in results if r.status == BenchmarkStatus.SUCCESS) / len(results) * 100,
            }

        return comparison

    def summary_report(self) -> str:
        """Generate a text summary report."""
        lines = [
            "=" * 60,
            "BENCHMARK METRICS SUMMARY",
            "=" * 60,
            f"Total Results: {self.count}",
            f"Success Rate: {self.success_rate():.1f}%",
            "",
            "EXECUTION TIME:",
        ]

        time_stats = self.execution_time_stats()
        lines.extend([
            f"  Mean: {time_stats['mean']:.2f} ms",
            f"  Median: {time_stats['median']:.2f} ms",
            f"  Std Dev: {time_stats['stdev']:.2f} ms",
            f"  P95: {time_stats['p95']:.2f} ms",
            f"  P99: {time_stats['p99']:.2f} ms",
            "",
            "MEMORY USAGE:",
        ])

        mem_stats = self.memory_stats()
        lines.extend([
            f"  Mean: {mem_stats['mean']:.1f} MB",
            f"  Peak: {mem_stats['max']:.1f} MB",
            "",
            "BY BACKEND:",
        ])

        for backend, stats in self.compare_backends().items():
            lines.append(f"  {backend}:")
            lines.append(f"    Runs: {stats['count']}")
            lines.append(f"    Avg Time: {stats['avg_time_ms']:.2f} ms")
            lines.append(f"    Success: {stats['success_rate']:.1f}%")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Performance Profiler - Detailed Timing
# =============================================================================


@dataclass
class ProfileSection:
    """A single profiled code section."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    children: List["ProfileSection"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000 if self.end_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }


class PerformanceProfiler:
    """Detailed performance profiler for benchmark execution.

    Provides hierarchical timing of code sections with:
    - Nested section support
    - Memory tracking
    - Automatic overhead compensation
    """

    def __init__(self, name: str = "root") -> None:
        """Initialize the profiler.

        Args:
            name: Name of the root profile section.
        """
        import time
        self._root = ProfileSection(name=name, start_time=time.perf_counter())
        self._stack: List[ProfileSection] = [self._root]
        self._overhead_ns = 0.0
        self._calibrate()

    def _calibrate(self) -> None:
        """Calibrate overhead measurement."""
        import time
        iterations = 100
        start = time.perf_counter_ns()
        for _ in range(iterations):
            _ = time.perf_counter_ns()
        end = time.perf_counter_ns()
        self._overhead_ns = (end - start) / iterations

    def start_section(self, name: str, **metadata: Any) -> None:
        """Start a new profiled section.

        Args:
            name: Section name.
            **metadata: Additional metadata to attach.
        """
        import time
        section = ProfileSection(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata,
        )
        self._stack[-1].children.append(section)
        self._stack.append(section)

    def end_section(self) -> float:
        """End the current section.

        Returns:
            Duration of the section in milliseconds.
        """
        import time
        if len(self._stack) <= 1:
            return 0.0
        
        section = self._stack.pop()
        section.end_time = time.perf_counter()
        return section.duration_ms

    def section(self, name: str, **metadata: Any):
        """Context manager for section timing.

        Args:
            name: Section name.
            **metadata: Additional metadata.

        Returns:
            Context manager.
        """
        from contextlib import contextmanager

        @contextmanager
        def _section():
            self.start_section(name, **metadata)
            try:
                yield
            finally:
                self.end_section()

        return _section()

    def finish(self) -> ProfileSection:
        """Finish profiling and return the root section.

        Returns:
            Root ProfileSection with all timing data.
        """
        import time
        self._root.end_time = time.perf_counter()
        return self._root

    def get_flat_timings(self) -> Dict[str, float]:
        """Get flat dictionary of section timings.

        Returns:
            Dictionary mapping section names to cumulative time in ms.
        """
        timings: Dict[str, float] = {}

        def collect(section: ProfileSection, prefix: str = "") -> None:
            key = f"{prefix}{section.name}" if prefix else section.name
            timings[key] = timings.get(key, 0) + section.duration_ms
            for child in section.children:
                collect(child, f"{key}.")

        collect(self._root)
        return timings

    def get_hotspots(self, threshold_ms: float = 10.0) -> List[tuple[str, float]]:
        """Get sections that took longer than threshold.

        Args:
            threshold_ms: Minimum duration to include.

        Returns:
            List of (name, duration_ms) tuples sorted by duration.
        """
        timings = self.get_flat_timings()
        hotspots = [(k, v) for k, v in timings.items() if v >= threshold_ms]
        return sorted(hotspots, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return self._root.to_dict()

    def summary(self) -> str:
        """Generate a text summary of profiling results."""
        lines = [
            "PERFORMANCE PROFILE:",
            f"Total Time: {self._root.duration_ms:.2f} ms",
            "",
            "Hotspots (>10ms):",
        ]

        for name, duration in self.get_hotspots(10.0):
            pct = (duration / self._root.duration_ms * 100) if self._root.duration_ms > 0 else 0
            lines.append(f"  {name}: {duration:.2f} ms ({pct:.1f}%)")

        return "\n".join(lines)


# =============================================================================
# Metrics Exporter - Multiple Formats
# =============================================================================


class MetricsExporter:
    """Export benchmark metrics to various formats."""

    def __init__(self, results: List[BenchmarkResult]) -> None:
        """Initialize the exporter.

        Args:
            results: Results to export.
        """
        self._results = results

    def to_csv(self, path: str) -> None:
        """Export to CSV format.

        Args:
            path: Output file path.
        """
        import csv

        fieldnames = [
            "benchmark_id",
            "backend_name",
            "status",
            "execution_time_ms",
            "memory_peak_mb",
            "throughput_shots_per_sec",
            "success_rate_percent",
            "timestamp",
            "circuit_hash",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self._results:
                row = {
                    "benchmark_id": result.benchmark_id,
                    "circuit_hash": result.circuit_hash,
                    "status": result.status.value,
                }
                if result.metrics:
                    row.update({
                        "backend_name": result.metrics.backend_name,
                        "execution_time_ms": result.metrics.execution_time_ms,
                        "memory_peak_mb": result.metrics.memory_peak_mb,
                        "throughput_shots_per_sec": result.metrics.throughput_shots_per_sec,
                        "success_rate_percent": result.metrics.success_rate_percent,
                        "timestamp": result.metrics.timestamp.isoformat(),
                    })
                writer.writerow(row)

    def to_json(self, path: str) -> None:
        """Export to JSON format.

        Args:
            path: Output file path.
        """
        data = {
            "export_time": datetime.utcnow().isoformat(),
            "count": len(self._results),
            "results": [r.to_dict() for r in self._results],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def to_dataframe(self) -> Any:
        """Export to pandas DataFrame.

        Returns:
            pandas DataFrame (requires pandas to be installed).
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame export")

        records = []
        for result in self._results:
            record = {
                "benchmark_id": result.benchmark_id,
                "circuit_hash": result.circuit_hash,
                "status": result.status.value,
            }
            if result.metrics:
                record.update({
                    "backend_name": result.metrics.backend_name,
                    "execution_time_ms": result.metrics.execution_time_ms,
                    "memory_peak_mb": result.metrics.memory_peak_mb,
                    "memory_baseline_mb": result.metrics.memory_baseline_mb,
                    "throughput_shots_per_sec": result.metrics.throughput_shots_per_sec,
                    "success_rate_percent": result.metrics.success_rate_percent,
                    "cpu_usage_percent": result.metrics.cpu_usage_percent,
                    "gpu_usage_percent": result.metrics.gpu_usage_percent,
                    "timestamp": result.metrics.timestamp,
                })
            records.append(record)

        return pd.DataFrame(records)
