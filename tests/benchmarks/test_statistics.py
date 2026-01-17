from datetime import datetime, timedelta

from proxima.benchmarks.statistics import StatisticsCalculator
from proxima.data.metrics import BenchmarkMetrics, BenchmarkResult, BenchmarkStatus


def _result_with_time(time_ms: float, ts: datetime) -> BenchmarkResult:
    metrics = BenchmarkMetrics(
        execution_time_ms=time_ms,
        memory_peak_mb=0.0,
        memory_baseline_mb=0.0,
        throughput_shots_per_sec=0.0,
        success_rate_percent=100.0,
        cpu_usage_percent=0.0,
        gpu_usage_percent=None,
        timestamp=ts,
        backend_name="mock",
        circuit_info={},
    )
    return BenchmarkResult(metrics=metrics, status=BenchmarkStatus.SUCCESS)


def test_basic_stats_and_percentiles():
    calc = StatisticsCalculator()
    values = [1, 2, 3, 4, 5]

    stats = calc.calculate_basic_stats(values)
    assert stats["mean"] == 3.0
    assert stats["median"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0

    percentiles = calc.calculate_percentiles(values, [25, 50, 75])
    assert percentiles[25] == 2.0
    assert percentiles[50] == 3.0
    assert percentiles[75] == 4.0


def test_outlier_detection():
    calc = StatisticsCalculator()
    now = datetime.utcnow()
    results = [
        _result_with_time(10, now),
        _result_with_time(11, now + timedelta(seconds=1)),
        _result_with_time(12, now + timedelta(seconds=2)),
        _result_with_time(100, now + timedelta(seconds=3)),
    ]

    outliers = calc.detect_outliers(results)
    assert len(outliers) == 1


def test_trend_analysis_increasing():
    calc = StatisticsCalculator()
    now = datetime.utcnow()
    results = [
        _result_with_time(10, now),
        _result_with_time(11, now + timedelta(seconds=1)),
        _result_with_time(13, now + timedelta(seconds=2)),
        _result_with_time(15, now + timedelta(seconds=3)),
    ]

    trend = calc.analyze_trends(results)
    assert trend.direction == "increasing"
    assert trend.slope > 0
    assert len(trend.moving_average) == len(results)


def test_validate_against_numpy():
    """Validate statistical calculations against numpy/scipy results."""
    import numpy as np

    calc = StatisticsCalculator()
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    stats = calc.calculate_basic_stats(values)

    # Validate against numpy
    assert abs(stats["mean"] - np.mean(values)) < 1e-9
    assert abs(stats["median"] - np.median(values)) < 1e-9
    assert abs(stats["min"] - np.min(values)) < 1e-9
    assert abs(stats["max"] - np.max(values)) < 1e-9
    assert abs(stats["stdev"] - np.std(values, ddof=1)) < 1e-9

    # Validate percentiles
    percentiles = calc.calculate_percentiles(values, [25, 50, 75])
    np_percentiles = np.percentile(values, [25, 50, 75])
    assert abs(percentiles[25] - np_percentiles[0]) < 1e-9
    assert abs(percentiles[50] - np_percentiles[1]) < 1e-9
    assert abs(percentiles[75] - np_percentiles[2]) < 1e-9
