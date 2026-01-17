"""Tests for Phase 10.1: Performance Optimizations."""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proxima.benchmarks.optimizations import (
    AdaptiveSampler,
    AdaptiveSamplerConfig,
    ConnectionPool,
    IncrementalStats,
    LazyResultLoader,
    ParallelStatsCalculator,
    PreparedStatements,
    ResultSummary,
    StatisticsCache,
    StreamingResultIterator,
)


class TestAdaptiveSampler:
    """Tests for adaptive sampling with exponential backoff."""

    def test_initial_interval(self) -> None:
        """Sampler returns initial interval at start."""
        config = AdaptiveSamplerConfig(initial_interval=0.05)
        sampler = AdaptiveSampler(config)
        sampler.start()

        interval = sampler.get_next_interval()
        assert interval == 0.05

    def test_exponential_backoff_after_frequent_period(self) -> None:
        """Sampler increases interval after frequent sampling period."""
        config = AdaptiveSamplerConfig(
            initial_interval=0.1,
            max_interval=2.0,
            backoff_factor=2.0,
            frequent_duration=0.0,  # Skip frequent period
        )
        sampler = AdaptiveSampler(config)
        sampler.start()

        # First call - with frequent_duration=0, backoff applies immediately
        interval1 = sampler.get_next_interval()
        assert interval1 == 0.2  # 0.1 * 2.0 = 0.2

        # Second call should have backoff applied again
        interval2 = sampler.get_next_interval()
        assert interval2 == 0.4  # 0.2 * 2.0 = 0.4

        # Third call
        interval3 = sampler.get_next_interval()
        assert interval3 == 0.8  # 0.4 * 2.0 = 0.8

    def test_max_interval_cap(self) -> None:
        """Sampler respects max interval limit."""
        config = AdaptiveSamplerConfig(
            initial_interval=1.0,
            max_interval=2.0,
            backoff_factor=10.0,
            frequent_duration=0.0,
        )
        sampler = AdaptiveSampler(config)
        sampler.start()

        _ = sampler.get_next_interval()
        interval = sampler.get_next_interval()
        assert interval <= 2.0

    def test_sample_count(self) -> None:
        """Sampler tracks sample count."""
        sampler = AdaptiveSampler()
        sampler.start()

        assert sampler.sample_count == 0
        sampler.get_next_interval()
        assert sampler.sample_count == 1
        sampler.get_next_interval()
        assert sampler.sample_count == 2

    def test_overhead_estimate(self) -> None:
        """Sampler provides overhead estimate."""
        sampler = AdaptiveSampler()
        sampler.start()

        for _ in range(10):
            sampler.get_next_interval()

        overhead = sampler.get_overhead_estimate_ms()
        assert overhead == 10.0  # ~1ms per sample


class TestConnectionPool:
    """Tests for SQLite connection pooling."""

    def test_pool_creation(self) -> None:
        """Pool creates specified number of connections."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            pool = ConnectionPool(f.name, pool_size=3)

            # Pool should have 3 connections
            assert pool._pool.qsize() == 3

            pool.close()

    def test_get_and_return_connection(self) -> None:
        """Pool correctly gets and returns connections."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            pool = ConnectionPool(f.name, pool_size=2)

            with pool.get_connection() as conn:
                assert conn is not None
                cursor = conn.execute("SELECT 1")
                assert cursor.fetchone()[0] == 1

            pool.close()

    def test_pool_closes_connections(self) -> None:
        """Pool closes all connections on close()."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            pool = ConnectionPool(f.name, pool_size=2)
            pool.close()

            assert pool._closed is True


class TestStatisticsCache:
    """Tests for statistics caching."""

    def test_cache_hit_and_miss(self) -> None:
        """Cache correctly identifies hits and misses."""
        cache = StatisticsCache(maxsize=10)

        # Miss
        hit, value = cache.get("key1")
        assert hit is False
        assert value is None

        # Set
        cache.set("key1", {"data": 123})

        # Hit
        hit, value = cache.get("key1")
        assert hit is True
        assert value == {"data": 123}

    def test_cache_ttl_expiration(self) -> None:
        """Cache entries expire after TTL."""
        cache = StatisticsCache(maxsize=10, ttl_seconds=0.1)

        cache.set("key1", "value1")
        hit, _ = cache.get("key1")
        assert hit is True

        # Wait for expiration
        time.sleep(0.15)

        hit, _ = cache.get("key1")
        assert hit is False

    def test_cache_lru_eviction(self) -> None:
        """Cache evicts oldest entries when full."""
        cache = StatisticsCache(maxsize=3, ttl_seconds=300)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recent
        cache.get("key1")

        # Add key4 - should evict key2 (oldest)
        cache.set("key4", "value4")

        # key1 should still exist (recently accessed)
        hit, _ = cache.get("key1")
        assert hit is True

        # key2 should be evicted
        hit, _ = cache.get("key2")
        assert hit is False

    def test_cache_decorator(self) -> None:
        """Cached decorator caches function results."""
        cache = StatisticsCache(maxsize=10)
        call_count = 0

        @cache.cached
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - computes
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - cached
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

    def test_cache_hit_rate(self) -> None:
        """Cache tracks hit rate correctly."""
        cache = StatisticsCache(maxsize=10)

        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        cache.get("key1")  # hit

        assert cache.hit_rate == 2 / 3

    def test_invalidate_all(self) -> None:
        """Invalidate clears entire cache when no pattern given."""
        cache = StatisticsCache(maxsize=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        count = cache.invalidate()
        assert count == 2
        assert len(cache._cache) == 0


class TestIncrementalStats:
    """Tests for incremental statistics calculation."""

    def test_running_mean(self) -> None:
        """Incremental stats computes correct running mean."""
        stats = IncrementalStats()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            stats.update(v)

        assert stats.mean == pytest.approx(3.0)
        assert stats.count == 5

    def test_running_stdev(self) -> None:
        """Incremental stats computes correct standard deviation."""
        stats = IncrementalStats()
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

        for v in values:
            stats.update(v)

        # Expected stdev (sample): ~2.14
        assert stats.stdev == pytest.approx(2.138, rel=0.01)

    def test_min_max(self) -> None:
        """Incremental stats tracks min and max."""
        stats = IncrementalStats()

        stats.update(5.0)
        stats.update(2.0)
        stats.update(8.0)
        stats.update(1.0)
        stats.update(9.0)

        assert stats.min_value == 1.0
        assert stats.max_value == 9.0

    def test_batch_update(self) -> None:
        """Batch update produces same results as individual updates."""
        stats1 = IncrementalStats()
        stats2 = IncrementalStats()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            stats1.update(v)

        stats2.update_batch(values)

        assert stats1.mean == stats2.mean
        assert stats1.stdev == stats2.stdev
        assert stats1.count == stats2.count

    def test_to_dict(self) -> None:
        """Stats can be exported to dictionary."""
        stats = IncrementalStats()
        stats.update(10.0)
        stats.update(20.0)

        result = stats.to_dict()

        assert "count" in result
        assert "mean" in result
        assert "stdev" in result
        assert "min" in result
        assert "max" in result
        assert result["count"] == 2.0


class TestLazyResultLoader:
    """Tests for lazy loading of benchmark results."""

    def test_lazy_summary_loading(self) -> None:
        """Loader fetches summaries without loading full results."""
        # Mock that returns 1 item on first call, empty on subsequent
        mock_fetch_summary = MagicMock(side_effect=[
            [ResultSummary(
                benchmark_id="id1",
                backend_name="lret",
                execution_time_ms=10.0,
                timestamp=None,
                status="success",
                circuit_hash="abc",
            )],
            [],  # Empty list for pagination termination
        ])
        mock_fetch_full = MagicMock()

        loader = LazyResultLoader(
            fetch_summary=mock_fetch_summary,
            fetch_full=mock_fetch_full,
            page_size=10,
        )

        summaries = list(loader.get_summaries(limit=5))

        assert len(summaries) == 1
        assert mock_fetch_summary.call_count == 2  # Initial + empty check
        mock_fetch_full.assert_not_called()

    def test_lazy_full_result_loading(self) -> None:
        """Loader fetches full result only on demand."""
        mock_fetch_summary = MagicMock()
        mock_result = MagicMock()
        mock_fetch_full = MagicMock(return_value=mock_result)

        loader = LazyResultLoader(
            fetch_summary=mock_fetch_summary,
            fetch_full=mock_fetch_full,
        )

        result = loader.get_full_result("id1")

        assert result is mock_result
        mock_fetch_full.assert_called_once_with("id1")

    def test_result_caching(self) -> None:
        """Loader caches fetched full results."""
        call_count = 0

        def fetch_full(id: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"id": id}

        loader = LazyResultLoader(
            fetch_summary=MagicMock(),
            fetch_full=fetch_full,
        )

        loader.get_full_result("id1")
        loader.get_full_result("id1")  # Should be cached

        assert call_count == 1


class TestStreamingResultIterator:
    """Tests for streaming result iteration."""

    def test_streaming_pagination(self) -> None:
        """Iterator fetches pages lazily."""
        pages = [
            [1, 2, 3],
            [4, 5, 6],
            [7],  # Partial last page
        ]
        page_index = 0

        def fetch_page(chunk_size: int, offset: int) -> list:
            nonlocal page_index
            if page_index >= len(pages):
                return []
            result = pages[page_index]
            page_index += 1
            return result

        iterator = StreamingResultIterator(fetch_page, chunk_size=3)
        results = list(iterator)

        assert results == [1, 2, 3, 4, 5, 6, 7]

    def test_reset_iterator(self) -> None:
        """Iterator can be reset and reused."""
        fetch_call_count = 0

        def fetch_page(chunk_size: int, offset: int) -> list:
            nonlocal fetch_call_count
            fetch_call_count += 1
            if offset == 0:
                return [1, 2, 3]
            return []

        iterator = StreamingResultIterator(fetch_page, chunk_size=10)

        # First iteration
        list(iterator)
        # With chunk_size=10 and 3 items returned, iterator knows it's exhausted
        # after first fetch (3 < 10), so only 1 call is made
        assert fetch_call_count == 1

        # Reset
        iterator.reset()
        fetch_call_count = 0

        # Second iteration
        list(iterator)
        assert fetch_call_count == 1  # Same optimization applies


class TestParallelStatsCalculator:
    """Tests for parallel statistics calculation."""

    def test_parallel_execution(self) -> None:
        """Calculator executes functions in parallel."""
        def slow_function(x: int) -> int:
            time.sleep(0.1)
            return x * 2

        with ParallelStatsCalculator(max_workers=4) as calc:
            calculations = [
                (slow_function, (1,)),
                (slow_function, (2,)),
                (slow_function, (3,)),
                (slow_function, (4,)),
            ]

            start = time.perf_counter()
            results = calc.calculate_parallel(calculations)
            elapsed = time.perf_counter() - start

        assert results == [2, 4, 6, 8]
        # Parallel should be faster than sequential (~0.4s)
        assert elapsed < 0.3


class TestPreparedStatements:
    """Tests for prepared SQL statements."""

    def test_get_statement(self) -> None:
        """PreparedStatements returns correct SQL."""
        stmts = PreparedStatements()

        assert "SELECT * FROM benchmarks WHERE id = ?" in stmts.get("get_by_id")
        assert "backend_name = ?" in stmts.get("get_by_backend")

    def test_missing_statement_raises(self) -> None:
        """PreparedStatements raises KeyError for unknown statement."""
        stmts = PreparedStatements()

        with pytest.raises(KeyError):
            stmts.get("nonexistent")
