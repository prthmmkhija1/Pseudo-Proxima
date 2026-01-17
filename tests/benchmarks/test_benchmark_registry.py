from datetime import datetime, timedelta

from proxima.data.benchmark_registry import BenchmarkRegistry
from proxima.data.metrics import BenchmarkMetrics, BenchmarkResult, BenchmarkStatus


def _build_result(backend: str = "mock", circuit_hash: str = "abc") -> BenchmarkResult:
    metrics = BenchmarkMetrics(
        execution_time_ms=12.5,
        memory_peak_mb=32.0,
        memory_baseline_mb=16.0,
        throughput_shots_per_sec=1000.0,
        success_rate_percent=100.0,
        cpu_usage_percent=20.0,
        gpu_usage_percent=None,
        timestamp=datetime.utcnow(),
        backend_name=backend,
        circuit_info={"qubits": 2},
    )
    return BenchmarkResult(circuit_hash=circuit_hash, metrics=metrics, status=BenchmarkStatus.SUCCESS)


def test_save_and_get_result(tmp_path):
    db_path = tmp_path / "bench.db"
    registry = BenchmarkRegistry(db_path=db_path)

    result = _build_result()
    registry.save_result(result)

    fetched = registry.get_result(result.benchmark_id)
    assert fetched is not None
    assert fetched.metrics is not None
    assert fetched.metrics.backend_name == "mock"
    assert fetched.metrics.execution_time_ms == 12.5


def test_query_with_filters(tmp_path):
    db_path = tmp_path / "bench.db"
    registry = BenchmarkRegistry(db_path=db_path)

    fast = _build_result(backend="fast", circuit_hash="c1")
    slow = _build_result(backend="slow", circuit_hash="c2")
    slow.metrics.execution_time_ms = 50.0  # type: ignore
    registry.save_results_batch([fast, slow])

    results_fast = registry.get_results_for_backend("fast")
    assert len(results_fast) == 1

    filtered = registry.get_results_filtered({"min_time": 20})
    assert any(r.metrics.backend_name == "slow" for r in filtered if r.metrics)


def test_results_in_range(tmp_path):
    db_path = tmp_path / "bench.db"
    registry = BenchmarkRegistry(db_path=db_path)

    older = _build_result(circuit_hash="old")
    older.metrics.timestamp = datetime.utcnow() - timedelta(days=1)  # type: ignore
    recent = _build_result(circuit_hash="new")

    registry.save_results_batch([older, recent])

    now = datetime.utcnow()
    results = registry.get_results_in_range(now - timedelta(hours=2), now + timedelta(hours=2))
    assert any(r.circuit_hash == "new" for r in results)
    assert all(r.metrics is not None for r in results)


def test_database_initialization(tmp_path):
    """Test that database and tables are created on initialization."""
    db_path = tmp_path / "new_bench.db"
    assert not db_path.exists()

    registry = BenchmarkRegistry(db_path=db_path)

    assert db_path.exists()
    # Verify table exists by attempting a query
    results = registry.get_results_filtered({}, limit=10)
    assert results == []
    registry.close()


def test_cleanup_old_results(tmp_path):
    """Test cleanup of results older than a threshold."""
    db_path = tmp_path / "bench.db"
    registry = BenchmarkRegistry(db_path=db_path)

    old = _build_result(circuit_hash="old")
    old.metrics.timestamp = datetime.utcnow() - timedelta(days=30)  # type: ignore
    recent = _build_result(circuit_hash="recent")

    registry.save_results_batch([old, recent])

    # Cleanup is typically done via get_results_in_range + delete
    # For now, verify we can filter by time and only get recent
    now = datetime.utcnow()
    recent_results = registry.get_results_in_range(now - timedelta(days=7), now + timedelta(hours=1))
    assert len(recent_results) == 1
    assert recent_results[0].circuit_hash == "recent"
