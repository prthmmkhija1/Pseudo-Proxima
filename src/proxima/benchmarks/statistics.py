"""Statistical analysis utilities for benchmark data."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

try:  # Optional dependency
    from scipy.stats import linregress  # type: ignore
except Exception:  # pragma: no cover - fallback when scipy missing
    linregress = None  # type: ignore

from proxima.data.metrics import BenchmarkResult


@dataclass(frozen=True, slots=True)
class TrendResult:
    direction: str
    slope: float
    confidence: float
    moving_average: list[float]
    timestamps: list[float]


class StatisticsCalculator:
    """Provides statistical summaries, trend analysis, and outlier detection.

    This class computes descriptive statistics, detects outliers using IQR method,
    and analyzes performance trends over time.

    Example:
        >>> calc = StatisticsCalculator()
        >>> stats = calc.calculate_basic_stats([10.0, 12.0, 11.0, 13.0, 10.5])
        >>> print(stats["mean"], stats["stdev"])
    """

    def calculate_basic_stats(self, values: Sequence[float]) -> Dict[str, float]:
        """Calculate descriptive statistics for a sequence of values.

        Args:
            values: Sequence of numeric values to analyze.

        Returns:
            Dictionary with keys: mean, median, mode, min, max, range,
            stdev, variance, q1, q3.
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "mode": 0.0,
                "min": 0.0,
                "max": 0.0,
                "range": 0.0,
                "stdev": 0.0,
                "variance": 0.0,
                "q1": 0.0,
                "q3": 0.0,
            }

        vals = list(values)
        mean = statistics.mean(vals)
        median = statistics.median(vals)
        try:
            mode = statistics.mode(vals)
        except statistics.StatisticsError:
            mode = median
        min_v = min(vals)
        max_v = max(vals)
        range_v = max_v - min_v
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        variance = statistics.variance(vals) if len(vals) > 1 else 0.0
        q1, q3 = np.percentile(vals, [25, 75]).tolist()

        return {
            "mean": float(mean),
            "median": float(median),
            "mode": float(mode),
            "min": float(min_v),
            "max": float(max_v),
            "range": float(range_v),
            "stdev": float(stdev),
            "variance": float(variance),
            "q1": float(q1),
            "q3": float(q3),
        }

    def calculate_percentiles(self, values: Sequence[float], percentiles: Sequence[int]) -> Dict[int, float]:
        """Calculate specified percentiles for a sequence of values.

        Args:
            values: Sequence of numeric values to analyze.
            percentiles: Sequence of percentile values (0-100) to compute.

        Returns:
            Dictionary mapping percentile -> value.

        Example:
            >>> calc = StatisticsCalculator()
            >>> calc.calculate_percentiles([1, 2, 3, 4, 5], [25, 50, 75])
            {25: 1.5, 50: 3.0, 75: 4.5}
        """
        if not values or not percentiles:
            return {int(p): 0.0 for p in percentiles}
        vals = list(values)
        pct_list = list(percentiles)
        pct_values = np.percentile(vals, pct_list)
        # numpy may return scalar for single percentile
        if not hasattr(pct_values, "__iter__") or isinstance(pct_values, (int, float)):
            pct_values = [pct_values]
        return {int(p): float(v) for p, v in zip(pct_list, pct_values)}

    def detect_outliers(self, results: List[BenchmarkResult]) -> List[str]:
        """Detect outliers using the IQR (Interquartile Range) method.

        Flags results whose execution time falls outside 1.5×IQR of Q1/Q3.

        Args:
            results: List of BenchmarkResult objects to analyze.

        Returns:
            List of benchmark_ids identified as outliers.
        """
        times: list[float] = []
        ids: list[str] = []
        for r in results:
            if r.metrics and r.metrics.execution_time_ms is not None:
                times.append(r.metrics.execution_time_ms)
                ids.append(r.benchmark_id)
        if len(times) < 4:
            return []
        q1, q3 = np.percentile(times, [25, 75]).tolist()
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [bid for bid, t in zip(ids, times) if t < lower or t > upper]

    def analyze_trends(self, results: List[BenchmarkResult]) -> TrendResult:
        """Analyze performance trends over time.

        Computes a linear regression slope to detect increasing, decreasing,
        or stable performance trends. Also computes moving averages.

        Args:
            results: List of BenchmarkResult objects ordered by time.

        Returns:
            TrendResult with direction, slope, confidence, and moving averages.
        """
        if not results:
            return TrendResult("stable", 0.0, 0.0, [], [])

        sorted_results = sorted(
            [r for r in results if r.metrics],
            key=lambda r: r.metrics.timestamp if r.metrics else 0,
        )
        times = [r.metrics.execution_time_ms for r in sorted_results if r.metrics]
        ts = [r.metrics.timestamp.timestamp() for r in sorted_results if r.metrics]

        # Moving average with window size 5
        window = 5
        moving_avg: list[float] = []
        for i in range(len(times)):
            start = max(0, i - window + 1)
            window_vals = times[start : i + 1]
            moving_avg.append(float(statistics.mean(window_vals)))

        slope = 0.0
        confidence = 0.0
        direction = "stable"

        if len(times) >= 2:
            if linregress is not None:
                res = linregress(ts, times)
                slope = float(res.slope)
                confidence = float(res.rvalue**2)
            else:
                # Fallback manual slope
                n = len(times)
                mean_x = sum(ts) / n
                mean_y = sum(times) / n
                num = sum((x - mean_x) * (y - mean_y) for x, y in zip(ts, times))
                den = sum((x - mean_x) ** 2 for x in ts) or 1.0
                slope = num / den
                confidence = 0.0

            eps = 1e-6
            if slope > eps:
                direction = "increasing"
            elif slope < -eps:
                direction = "decreasing"
            else:
                direction = "stable"

        return TrendResult(direction, slope, confidence, moving_avg, ts)


__all__ = ["StatisticsCalculator", "TrendResult"]
