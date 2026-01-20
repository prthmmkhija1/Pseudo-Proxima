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


# =============================================================================
# Custom Metrics - User-Defined Metrics System
# =============================================================================


class MetricType(str, Enum):
    """Types of custom metric values."""

    COUNTER = "counter"  # Monotonically increasing value
    GAUGE = "gauge"  # Value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    TIMER = "timer"  # Duration measurements
    RATE = "rate"  # Value per time unit
    PERCENTAGE = "percentage"  # 0-100 value


class MetricUnit(str, Enum):
    """Units for custom metrics."""

    NONE = ""
    MILLISECONDS = "ms"
    SECONDS = "s"
    BYTES = "bytes"
    KILOBYTES = "KB"
    MEGABYTES = "MB"
    GIGABYTES = "GB"
    PERCENT = "%"
    COUNT = "count"
    OPERATIONS_PER_SEC = "ops/s"
    BITS = "bits"
    QUBITS = "qubits"
    SHOTS = "shots"


@dataclass
class MetricDefinition:
    """Definition of a custom metric.

    Allows users to define their own metrics with custom
    names, types, units, and computation logic.

    Attributes:
        name: Unique metric name.
        display_name: Human-readable name for display.
        description: Detailed description of the metric.
        metric_type: Type of metric (counter, gauge, etc.).
        unit: Unit of measurement.
        tags: Categorization tags.
        aggregations: Supported aggregation methods.
        thresholds: Warning/critical thresholds.
        formula: Optional formula string for derived metrics.
        dependencies: Other metrics this depends on.

    Example:
        >>> metric_def = MetricDefinition(
        ...     name="quantum_fidelity",
        ...     display_name="Quantum Fidelity",
        ...     description="Measure of circuit execution accuracy",
        ...     metric_type=MetricType.GAUGE,
        ...     unit=MetricUnit.PERCENT,
        ...     thresholds={"warning": 95.0, "critical": 90.0}
        ... )
    """

    name: str
    display_name: str
    description: str
    metric_type: MetricType
    unit: MetricUnit = MetricUnit.NONE
    tags: List[str] = field(default_factory=list)
    aggregations: List[str] = field(
        default_factory=lambda: ["min", "max", "mean", "sum"]
    )
    thresholds: Dict[str, float] = field(default_factory=dict)
    formula: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate metric definition."""
        if not self.name:
            raise ValueError("Metric name cannot be empty")
        if not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid metric name '{self.name}': use alphanumeric and underscores"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "metric_type": self.metric_type.value,
            "unit": self.unit.value,
            "tags": self.tags,
            "aggregations": self.aggregations,
            "thresholds": self.thresholds,
            "formula": self.formula,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricDefinition":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not isinstance(created_at, datetime):
            created_at = datetime.utcnow()

        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            description=data.get("description", ""),
            metric_type=MetricType(data.get("metric_type", "gauge")),
            unit=MetricUnit(data.get("unit", "")),
            tags=data.get("tags", []),
            aggregations=data.get("aggregations", ["min", "max", "mean", "sum"]),
            thresholds=data.get("thresholds", {}),
            formula=data.get("formula"),
            dependencies=data.get("dependencies", []),
            created_at=created_at,
            version=data.get("version", "1.0"),
        )


@dataclass
class CustomMetricValue:
    """A single custom metric measurement.

    Attributes:
        metric_name: Name of the metric.
        value: The measured value.
        timestamp: When the measurement was taken.
        labels: Key-value labels for filtering.
        metadata: Additional context.
    """

    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomMetricValue":
        """Create from dictionary."""
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif not isinstance(ts, datetime):
            ts = datetime.utcnow()

        return cls(
            metric_name=data["metric_name"],
            value=float(data["value"]),
            timestamp=ts,
            labels=data.get("labels", {}),
            metadata=data.get("metadata", {}),
        )


class MetricCalculator:
    """Base class for custom metric calculators.

    Override the `compute` method to implement custom
    metric computation logic.

    Example:
        >>> class FidelityCalculator(MetricCalculator):
        ...     def compute(self, result: BenchmarkResult) -> float:
        ...         # Custom fidelity calculation
        ...         return result.metrics.success_rate_percent / 100
    """

    def __init__(
        self,
        definition: MetricDefinition,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize calculator.

        Args:
            definition: Metric definition.
            config: Optional configuration.
        """
        self.definition = definition
        self.config = config or {}

    def compute(self, result: BenchmarkResult) -> Optional[float]:
        """Compute metric value from benchmark result.

        Override this method in subclasses.

        Args:
            result: Benchmark result to compute metric from.

        Returns:
            Computed metric value or None if not computable.
        """
        raise NotImplementedError("Subclasses must implement compute()")

    def compute_batch(
        self,
        results: List[BenchmarkResult],
    ) -> List[CustomMetricValue]:
        """Compute metric for multiple results.

        Args:
            results: List of benchmark results.

        Returns:
            List of metric values.
        """
        values = []
        for result in results:
            try:
                value = self.compute(result)
                if value is not None:
                    labels = {}
                    if result.metrics:
                        labels["backend"] = result.metrics.backend_name
                        labels["timestamp"] = result.metrics.timestamp.isoformat()
                    
                    values.append(
                        CustomMetricValue(
                            metric_name=self.definition.name,
                            value=value,
                            labels=labels,
                            metadata={"benchmark_id": result.benchmark_id},
                        )
                    )
            except Exception:
                continue
        return values

    def check_thresholds(
        self,
        value: float,
    ) -> tuple[str, Optional[str]]:
        """Check value against thresholds.

        Args:
            value: Metric value.

        Returns:
            Tuple of (status, message).
            Status is "ok", "warning", or "critical".
        """
        thresholds = self.definition.thresholds
        
        critical = thresholds.get("critical")
        warning = thresholds.get("warning")

        if critical is not None and value <= critical:
            return "critical", f"{self.definition.display_name} is critically low: {value}"
        if warning is not None and value <= warning:
            return "warning", f"{self.definition.display_name} is below threshold: {value}"
        
        return "ok", None


class FormulaMetricCalculator(MetricCalculator):
    """Calculator that uses a formula string.

    Supports basic arithmetic on result fields.

    Example:
        >>> calc = FormulaMetricCalculator(
        ...     definition=MetricDefinition(
        ...         name="efficiency",
        ...         display_name="Efficiency",
        ...         description="Shots per MB",
        ...         metric_type=MetricType.GAUGE,
        ...         formula="throughput_shots_per_sec / memory_peak_mb"
        ...     )
        ... )
    """

    def compute(self, result: BenchmarkResult) -> Optional[float]:
        """Compute metric using formula."""
        if not self.definition.formula:
            return None
        if not result.metrics:
            return None

        # Build context from metrics
        context: Dict[str, float] = {
            "execution_time_ms": result.metrics.execution_time_ms,
            "memory_peak_mb": result.metrics.memory_peak_mb,
            "memory_baseline_mb": result.metrics.memory_baseline_mb,
            "throughput_shots_per_sec": result.metrics.throughput_shots_per_sec,
            "success_rate_percent": result.metrics.success_rate_percent,
            "cpu_usage_percent": result.metrics.cpu_usage_percent,
            "gpu_usage_percent": result.metrics.gpu_usage_percent or 0.0,
        }

        # Add circuit info
        for key, val in result.metrics.circuit_info.items():
            if isinstance(val, (int, float)):
                context[key] = float(val)

        try:
            # Safe eval with only math operations
            # Note: In production, use a proper expression parser
            import math
            safe_context = {
                **context,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
                "pow": pow,
            }
            return float(eval(self.definition.formula, {"__builtins__": {}}, safe_context))
        except Exception:
            return None


class CustomMetricsRegistry:
    """Registry for custom metric definitions.

    Manages registration, lookup, and persistence of
    custom metric definitions.

    Example:
        >>> registry = CustomMetricsRegistry()
        >>> registry.register(MetricDefinition(
        ...     name="custom_metric",
        ...     display_name="Custom Metric",
        ...     description="My custom metric",
        ...     metric_type=MetricType.GAUGE
        ... ))
        >>> metric = registry.get("custom_metric")
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._definitions: Dict[str, MetricDefinition] = {}
        self._calculators: Dict[str, MetricCalculator] = {}
        self._load_builtins()

    def _load_builtins(self) -> None:
        """Load built-in metric definitions."""
        builtins = [
            MetricDefinition(
                name="memory_efficiency",
                display_name="Memory Efficiency",
                description="Shots processed per MB of memory",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.OPERATIONS_PER_SEC,
                formula="throughput_shots_per_sec / memory_peak_mb",
                tags=["performance", "memory"],
            ),
            MetricDefinition(
                name="cpu_memory_ratio",
                display_name="CPU/Memory Ratio",
                description="CPU usage relative to memory usage",
                metric_type=MetricType.GAUGE,
                formula="cpu_usage_percent / memory_peak_mb",
                tags=["resource", "balance"],
            ),
            MetricDefinition(
                name="execution_score",
                display_name="Execution Score",
                description="Combined performance score",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.NONE,
                formula="success_rate_percent * (1000 / (execution_time_ms + 1))",
                tags=["overall", "score"],
                thresholds={"warning": 50.0, "critical": 25.0},
            ),
            MetricDefinition(
                name="memory_delta",
                display_name="Memory Delta",
                description="Memory increase during execution",
                metric_type=MetricType.GAUGE,
                unit=MetricUnit.MEGABYTES,
                formula="memory_peak_mb - memory_baseline_mb",
                tags=["memory", "delta"],
            ),
            MetricDefinition(
                name="throughput_per_cpu",
                display_name="Throughput per CPU%",
                description="Shots per second normalized by CPU usage",
                metric_type=MetricType.GAUGE,
                formula="throughput_shots_per_sec / max(cpu_usage_percent, 1)",
                tags=["efficiency", "cpu"],
            ),
        ]

        for defn in builtins:
            self._definitions[defn.name] = defn
            self._calculators[defn.name] = FormulaMetricCalculator(defn)

    def register(
        self,
        definition: MetricDefinition,
        calculator: Optional[MetricCalculator] = None,
        overwrite: bool = False,
    ) -> None:
        """Register a custom metric.

        Args:
            definition: Metric definition.
            calculator: Optional custom calculator.
            overwrite: Whether to overwrite existing.

        Raises:
            ValueError: If metric exists and overwrite is False.
        """
        if definition.name in self._definitions and not overwrite:
            raise ValueError(f"Metric '{definition.name}' already registered")

        self._definitions[definition.name] = definition
        
        if calculator:
            self._calculators[definition.name] = calculator
        elif definition.formula:
            self._calculators[definition.name] = FormulaMetricCalculator(definition)

    def unregister(self, name: str) -> bool:
        """Unregister a metric.

        Args:
            name: Metric name.

        Returns:
            True if metric was removed.
        """
        if name in self._definitions:
            del self._definitions[name]
            self._calculators.pop(name, None)
            return True
        return False

    def get(self, name: str) -> Optional[MetricDefinition]:
        """Get metric definition by name."""
        return self._definitions.get(name)

    def get_calculator(self, name: str) -> Optional[MetricCalculator]:
        """Get calculator for metric."""
        return self._calculators.get(name)

    def list_metrics(
        self,
        tag: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
    ) -> List[MetricDefinition]:
        """List all registered metrics.

        Args:
            tag: Filter by tag.
            metric_type: Filter by type.

        Returns:
            List of matching metric definitions.
        """
        metrics = list(self._definitions.values())

        if tag:
            metrics = [m for m in metrics if tag in m.tags]
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]

        return sorted(metrics, key=lambda m: m.name)

    def export_definitions(self, path: str) -> None:
        """Export all definitions to JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "metrics": [m.to_dict() for m in self._definitions.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_definitions(
        self,
        path: str,
        overwrite: bool = False,
    ) -> int:
        """Import definitions from JSON file.

        Args:
            path: Input file path.
            overwrite: Whether to overwrite existing.

        Returns:
            Number of metrics imported.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for metric_data in data.get("metrics", []):
            try:
                definition = MetricDefinition.from_dict(metric_data)
                self.register(definition, overwrite=overwrite)
                count += 1
            except (ValueError, KeyError):
                continue

        return count


class MetricsCollector:
    """Collector for aggregating custom metric values.

    Collects metric values and provides aggregation,
    time-series, and export capabilities.

    Example:
        >>> collector = MetricsCollector()
        >>> collector.record("my_metric", 42.0, labels={"backend": "lret"})
        >>> collector.record("my_metric", 45.0, labels={"backend": "lret"})
        >>> stats = collector.aggregate("my_metric")
    """

    def __init__(
        self,
        registry: Optional[CustomMetricsRegistry] = None,
        max_values_per_metric: int = 10000,
    ) -> None:
        """Initialize the collector.

        Args:
            registry: Optional metric registry.
            max_values_per_metric: Maximum values to keep per metric.
        """
        self._registry = registry or CustomMetricsRegistry()
        self._values: Dict[str, List[CustomMetricValue]] = {}
        self._max_values = max_values_per_metric

    @property
    def registry(self) -> CustomMetricsRegistry:
        """Get the metric registry."""
        return self._registry

    def record(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CustomMetricValue:
        """Record a metric value.

        Args:
            metric_name: Name of the metric.
            value: Value to record.
            labels: Optional labels.
            metadata: Optional metadata.

        Returns:
            The recorded metric value.
        """
        metric_value = CustomMetricValue(
            metric_name=metric_name,
            value=value,
            labels=labels or {},
            metadata=metadata or {},
        )

        if metric_name not in self._values:
            self._values[metric_name] = []

        self._values[metric_name].append(metric_value)

        # Trim if over limit
        if len(self._values[metric_name]) > self._max_values:
            self._values[metric_name] = self._values[metric_name][-self._max_values:]

        return metric_value

    def record_from_result(
        self,
        result: BenchmarkResult,
        metric_names: Optional[List[str]] = None,
    ) -> List[CustomMetricValue]:
        """Record metrics from a benchmark result.

        Uses calculators from the registry to compute values.

        Args:
            result: Benchmark result.
            metric_names: Specific metrics to compute (all if None).

        Returns:
            List of recorded values.
        """
        recorded = []
        names = metric_names or list(self._registry._calculators.keys())

        for name in names:
            calculator = self._registry.get_calculator(name)
            if calculator:
                try:
                    value = calculator.compute(result)
                    if value is not None:
                        labels = {}
                        if result.metrics:
                            labels["backend"] = result.metrics.backend_name
                        
                        metric_value = self.record(
                            name,
                            value,
                            labels=labels,
                            metadata={"benchmark_id": result.benchmark_id},
                        )
                        recorded.append(metric_value)
                except Exception:
                    continue

        return recorded

    def get_values(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[CustomMetricValue]:
        """Get recorded values for a metric.

        Args:
            metric_name: Metric name.
            labels: Filter by labels.
            since: Filter by start time.
            until: Filter by end time.

        Returns:
            List of matching values.
        """
        values = self._values.get(metric_name, [])

        if labels:
            values = [
                v for v in values
                if all(v.labels.get(k) == lv for k, lv in labels.items())
            ]

        if since:
            values = [v for v in values if v.timestamp >= since]

        if until:
            values = [v for v in values if v.timestamp <= until]

        return values

    def aggregate(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Aggregate metric values.

        Args:
            metric_name: Metric name.
            labels: Optional label filter.

        Returns:
            Dictionary of aggregated statistics.
        """
        import statistics

        values = self.get_values(metric_name, labels=labels)
        nums = [v.value for v in values]

        if not nums:
            return {
                "count": 0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "stdev": 0.0,
            }

        return {
            "count": len(nums),
            "sum": sum(nums),
            "min": min(nums),
            "max": max(nums),
            "mean": statistics.mean(nums),
            "median": statistics.median(nums),
            "stdev": statistics.stdev(nums) if len(nums) > 1 else 0.0,
        }

    def time_series(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
        bucket_seconds: int = 60,
    ) -> List[Dict[str, Any]]:
        """Get time series of metric values.

        Args:
            metric_name: Metric name.
            labels: Optional label filter.
            bucket_seconds: Time bucket size in seconds.

        Returns:
            List of time buckets with aggregated values.
        """
        values = self.get_values(metric_name, labels=labels)
        
        if not values:
            return []

        # Sort by timestamp
        values = sorted(values, key=lambda v: v.timestamp)

        # Bucket values
        buckets: Dict[int, List[float]] = {}
        for v in values:
            bucket_ts = int(v.timestamp.timestamp()) // bucket_seconds * bucket_seconds
            if bucket_ts not in buckets:
                buckets[bucket_ts] = []
            buckets[bucket_ts].append(v.value)

        # Aggregate buckets
        series = []
        for ts, nums in sorted(buckets.items()):
            series.append({
                "timestamp": datetime.utcfromtimestamp(ts).isoformat(),
                "count": len(nums),
                "mean": sum(nums) / len(nums),
                "min": min(nums),
                "max": max(nums),
            })

        return series

    def check_all_thresholds(self) -> List[Dict[str, Any]]:
        """Check all metrics against their thresholds.

        Returns:
            List of threshold violations.
        """
        violations = []

        for metric_name in self._values:
            definition = self._registry.get(metric_name)
            if not definition or not definition.thresholds:
                continue

            calculator = self._registry.get_calculator(metric_name)
            if not calculator:
                continue

            values = self.get_values(metric_name)
            if not values:
                continue

            # Check latest value
            latest = values[-1]
            status, message = calculator.check_thresholds(latest.value)

            if status != "ok":
                violations.append({
                    "metric": metric_name,
                    "status": status,
                    "value": latest.value,
                    "message": message,
                    "timestamp": latest.timestamp.isoformat(),
                    "labels": latest.labels,
                })

        return violations

    def clear(self, metric_name: Optional[str] = None) -> None:
        """Clear recorded values.

        Args:
            metric_name: Specific metric to clear (all if None).
        """
        if metric_name:
            self._values.pop(metric_name, None)
        else:
            self._values.clear()

    def export_values(
        self,
        path: str,
        metric_names: Optional[List[str]] = None,
    ) -> None:
        """Export metric values to JSON.

        Args:
            path: Output file path.
            metric_names: Specific metrics to export (all if None).
        """
        names = metric_names or list(self._values.keys())
        
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "metrics": {},
        }

        for name in names:
            if name in self._values:
                data["metrics"][name] = {
                    "count": len(self._values[name]),
                    "values": [v.to_dict() for v in self._values[name]],
                    "aggregated": self.aggregate(name),
                }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def summary_report(self) -> str:
        """Generate summary report of all metrics."""
        lines = [
            "=" * 60,
            "CUSTOM METRICS SUMMARY",
            "=" * 60,
        ]

        for metric_name in sorted(self._values.keys()):
            definition = self._registry.get(metric_name)
            display = definition.display_name if definition else metric_name
            stats = self.aggregate(metric_name)

            lines.extend([
                "",
                f"{display}:",
                f"  Count: {stats['count']}",
                f"  Mean: {stats['mean']:.4f}",
                f"  Min: {stats['min']:.4f}",
                f"  Max: {stats['max']:.4f}",
                f"  Std Dev: {stats['stdev']:.4f}",
            ])

        # Check thresholds
        violations = self.check_all_thresholds()
        if violations:
            lines.extend(["", "THRESHOLD VIOLATIONS:"])
            for v in violations:
                lines.append(f"  [{v['status'].upper()}] {v['message']}")

        lines.append("=" * 60)
        return "\n".join(lines)


class CompositeMetric(MetricCalculator):
    """Metric that combines multiple other metrics.

    Useful for creating complex derived metrics.

    Example:
        >>> composite = CompositeMetric(
        ...     definition=MetricDefinition(
        ...         name="performance_index",
        ...         display_name="Performance Index",
        ...         description="Combined performance metric",
        ...         metric_type=MetricType.GAUGE,
        ...     ),
        ...     components=["execution_score", "memory_efficiency"],
        ...     weights=[0.7, 0.3],
        ...     combiner=lambda vals: sum(v * w for v, w in zip(vals, [0.7, 0.3]))
        ... )
    """

    def __init__(
        self,
        definition: MetricDefinition,
        components: List[str],
        weights: Optional[List[float]] = None,
        combiner: Optional[callable] = None,
        registry: Optional[CustomMetricsRegistry] = None,
    ) -> None:
        """Initialize composite metric.

        Args:
            definition: Metric definition.
            components: List of component metric names.
            weights: Optional weights for weighted average.
            combiner: Optional custom combining function.
            registry: Optional registry for component lookup.
        """
        super().__init__(definition)
        self.components = components
        self.weights = weights or [1.0] * len(components)
        self.combiner = combiner
        self._registry = registry or CustomMetricsRegistry()

    def compute(self, result: BenchmarkResult) -> Optional[float]:
        """Compute composite metric."""
        values = []

        for component_name in self.components:
            calculator = self._registry.get_calculator(component_name)
            if calculator:
                try:
                    value = calculator.compute(result)
                    if value is not None:
                        values.append(value)
                except Exception:
                    return None
            else:
                return None

        if len(values) != len(self.components):
            return None

        if self.combiner:
            return self.combiner(values)

        # Default: weighted average
        return sum(v * w for v, w in zip(values, self.weights)) / sum(self.weights)


# Built-in custom metric calculators
class EfficiencyScoreCalculator(MetricCalculator):
    """Calculates an efficiency score based on multiple factors."""

    def compute(self, result: BenchmarkResult) -> Optional[float]:
        """Compute efficiency score."""
        if not result.metrics or result.status != BenchmarkStatus.SUCCESS:
            return None

        m = result.metrics

        # Normalize factors (higher is better)
        time_factor = 1000 / (m.execution_time_ms + 1)  # Fast = high
        memory_factor = 100 / (m.memory_peak_mb + 1)  # Low memory = high
        success_factor = m.success_rate_percent / 100  # High success = high
        throughput_factor = min(m.throughput_shots_per_sec / 100000, 1.0)  # Normalized

        # Weighted combination
        score = (
            time_factor * 0.35 +
            memory_factor * 0.20 +
            success_factor * 0.25 +
            throughput_factor * 0.20
        ) * 100

        return min(score, 100.0)


class ResourceUtilizationCalculator(MetricCalculator):
    """Calculates resource utilization efficiency."""

    def compute(self, result: BenchmarkResult) -> Optional[float]:
        """Compute resource utilization."""
        if not result.metrics:
            return None

        m = result.metrics

        # Calculate how efficiently resources are used
        cpu_efficiency = min(m.cpu_usage_percent / 100, 1.0)
        gpu_efficiency = (m.gpu_usage_percent or 0) / 100

        # Memory efficiency: closer to baseline is better
        if m.memory_peak_mb > 0:
            memory_overhead = (m.memory_peak_mb - m.memory_baseline_mb) / m.memory_peak_mb
            memory_efficiency = 1 - min(memory_overhead, 1.0)
        else:
            memory_efficiency = 0.5

        # Combined utilization score
        if m.gpu_usage_percent:
            return (cpu_efficiency * 0.3 + gpu_efficiency * 0.4 + memory_efficiency * 0.3) * 100
        else:
            return (cpu_efficiency * 0.5 + memory_efficiency * 0.5) * 100


# Update __all__ to include new exports
__all__ = [
    "BenchmarkStatus",
    "BenchmarkMetrics",
    "BenchmarkResult",
    "BenchmarkComparison",
    "MetricsAggregator",
    "PerformanceProfiler",
    "MetricsExporter",
    "MetricType",
    "MetricUnit",
    "MetricDefinition",
    "CustomMetricValue",
    "MetricCalculator",
    "FormulaMetricCalculator",
    "CustomMetricsRegistry",
    "MetricsCollector",
    "CompositeMetric",
    "EfficiencyScoreCalculator",
    "ResourceUtilizationCalculator",
]
