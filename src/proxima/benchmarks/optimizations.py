"""Performance optimizations for benchmarking (Phase 10.1).

Implements:
- AdaptiveSampler: Exponential backoff for resource sampling
- StatisticsCache: Cache calculated statistics
- IncrementalStats: Running average calculations
- LazyResultLoader: Lazy loading for benchmark results
- ConnectionPool: Connection pooling for concurrent database access
"""

from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


# =============================================================================
# Phase 10.1.1: Optimize Resource Monitor Sampling
# =============================================================================


@dataclass
class AdaptiveSamplerConfig:
    """Configuration for adaptive sampling with exponential backoff.

    Attributes:
        initial_interval: Initial sampling interval in seconds.
        max_interval: Maximum sampling interval in seconds.
        backoff_factor: Multiplier for exponential backoff.
        frequent_duration: Duration to use frequent sampling at start.
    """

    initial_interval: float = 0.1  # 100ms at start
    max_interval: float = 2.0  # 2s max
    backoff_factor: float = 1.5
    frequent_duration: float = 5.0  # Frequent sampling for first 5 seconds


class AdaptiveSampler:
    """Adaptive resource sampler with exponential backoff.

    Samples frequently at the start of execution (to capture startup behavior)
    and less frequently later (to reduce overhead on long-running benchmarks).

    Example:
        >>> sampler = AdaptiveSampler()
        >>> sampler.start()
        >>> while not done:
        ...     interval = sampler.get_next_interval()
        ...     sample_resources()
        ...     time.sleep(interval)
        >>> sampler.stop()
    """

    def __init__(self, config: AdaptiveSamplerConfig | None = None) -> None:
        self.config = config or AdaptiveSamplerConfig()
        self._start_time: float | None = None
        self._current_interval: float = self.config.initial_interval
        self._sample_count: int = 0

    def start(self) -> None:
        """Start the adaptive sampler."""
        self._start_time = time.perf_counter()
        self._current_interval = self.config.initial_interval
        self._sample_count = 0

    def stop(self) -> None:
        """Stop the adaptive sampler."""
        self._start_time = None

    def get_next_interval(self) -> float:
        """Get the next sampling interval with exponential backoff.

        Returns:
            Sampling interval in seconds.
        """
        if self._start_time is None:
            return self.config.initial_interval

        elapsed = time.perf_counter() - self._start_time
        self._sample_count += 1

        # Use frequent sampling for the initial period
        if elapsed < self.config.frequent_duration:
            return self.config.initial_interval

        # Apply exponential backoff after initial period
        self._current_interval = min(
            self._current_interval * self.config.backoff_factor,
            self.config.max_interval,
        )
        return self._current_interval

    @property
    def sample_count(self) -> int:
        """Number of samples taken."""
        return self._sample_count

    def get_overhead_estimate_ms(self) -> float:
        """Estimate total sampling overhead in milliseconds."""
        # Rough estimate: ~1ms per sample for CPU/memory check
        return self._sample_count * 1.0


# =============================================================================
# Phase 10.1.2: Optimize Database Queries
# =============================================================================


class ConnectionPool:
    """SQLite connection pool for concurrent database access.

    Provides a pool of reusable database connections with automatic
    connection management and thread safety.

    Example:
        >>> pool = ConnectionPool("benchmarks.db", pool_size=5)
        >>> with pool.get_connection() as conn:
        ...     cursor = conn.execute("SELECT * FROM benchmarks")
        >>> pool.close()
    """

    def __init__(self, db_path: str, pool_size: int = 5) -> None:
        self._db_path = db_path
        self._pool_size = pool_size
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._closed = False

        # Pre-populate the pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def get_connection(self) -> "PooledConnection":
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        try:
            conn = self._pool.get(timeout=5.0)
        except Exception:
            # Pool exhausted, create temporary connection
            conn = self._create_connection()

        return PooledConnection(conn, self)

    def return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        if self._closed:
            conn.close()
            return

        try:
            self._pool.put_nowait(conn)
        except Exception:
            # Pool full, close the connection
            conn.close()

    def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                break


class PooledConnection:
    """Context manager for pooled connections."""

    def __init__(self, conn: sqlite3.Connection, pool: ConnectionPool) -> None:
        self._conn = conn
        self._pool = pool

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._pool.return_connection(self._conn)


@dataclass
class PreparedStatements:
    """Container for prepared SQL statements to reduce parsing overhead.

    Pre-compiles common queries for faster repeated execution.
    """

    _statements: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._statements = {
            "get_by_id": "SELECT * FROM benchmarks WHERE id = ?;",
            "get_by_backend": (
                "SELECT * FROM benchmarks WHERE backend_name = ? "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
            ),
            "get_by_circuit": (
                "SELECT * FROM benchmarks WHERE circuit_hash = ? "
                "ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
            ),
            "get_summary": (
                "SELECT id, backend_name, execution_time_ms, timestamp, status "
                "FROM benchmarks ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
            ),
            "count_by_backend": (
                "SELECT COUNT(*) as cnt FROM benchmarks WHERE backend_name = ?;"
            ),
            "insert_result": (
                "INSERT OR REPLACE INTO benchmarks "
                "(id, circuit_hash, backend_name, timestamp, status, "
                "execution_time_ms, memory_peak_mb, memory_baseline_mb, "
                "throughput_shots_per_sec, success_rate_percent, "
                "cpu_usage_percent, gpu_usage_percent, "
                "metadata_json, circuit_info_json, qubit_count, error_message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
            ),
        }

    def get(self, name: str) -> str:
        """Get a prepared statement by name."""
        return self._statements[name]


# =============================================================================
# Phase 10.1.3: Optimize Statistics Calculations
# =============================================================================


class StatisticsCache:
    """LRU cache for calculated statistics.

    Caches expensive statistical calculations to avoid recomputation
    when the same queries are repeated.

    Example:
        >>> cache = StatisticsCache(maxsize=100)
        >>> @cache.cached
        ... def calculate_percentiles(values):
        ...     return expensive_calculation(values)
    """

    def __init__(self, maxsize: int = 128, ttl_seconds: float = 300.0) -> None:
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function arguments."""
        key_data = json.dumps((func_name, args, sorted(kwargs.items())), default=str)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> tuple[bool, Any]:
        """Get a cached value by key.

        Returns:
            Tuple of (hit, value). If hit is False, value is None.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return False, None

            timestamp, value = self._cache[key]
            if time.time() - timestamp > self._ttl:
                # Entry expired
                del self._cache[key]
                self._misses += 1
                return False, None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return True, value

    def set(self, key: str, value: Any) -> None:
        """Set a cached value."""
        with self._lock:
            self._cache[key] = (time.time(), value)
            self._cache.move_to_end(key)

            # Evict oldest entries if cache is full
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def cached(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to cache function results."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = self._make_key(func.__name__, args, kwargs)
            hit, value = self.get(key)
            if hit:
                return value

            result = func(*args, **kwargs)
            self.set(key, result)
            return result

        return wrapper

    def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: If None, clear entire cache. Otherwise, clear matching keys.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            keys_to_delete = [k for k in self._cache if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }


class IncrementalStats:
    """Incremental statistics calculator using Welford's algorithm.

    Computes running mean, variance, min, and max without storing
    all values in memory.

    Example:
        >>> stats = IncrementalStats()
        >>> for value in stream:
        ...     stats.update(value)
        >>> print(stats.mean, stats.stdev)
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0  # Sum of squared differences from mean
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._sum: float = 0.0

    def update(self, value: float) -> None:
        """Add a new value to the running statistics."""
        self._count += 1
        self._sum += value
        self._min = min(self._min, value)
        self._max = max(self._max, value)

        # Welford's online algorithm for variance
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def update_batch(self, values: List[float]) -> None:
        """Add multiple values efficiently."""
        for v in values:
            self.update(v)

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean if self._count > 0 else 0.0

    @property
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def stdev(self) -> float:
        return self.variance**0.5

    @property
    def min_value(self) -> float:
        return self._min if self._count > 0 else 0.0

    @property
    def max_value(self) -> float:
        return self._max if self._count > 0 else 0.0

    @property
    def sum(self) -> float:
        return self._sum

    def to_dict(self) -> Dict[str, float]:
        """Export statistics as dictionary."""
        return {
            "count": float(self._count),
            "mean": self.mean,
            "variance": self.variance,
            "stdev": self.stdev,
            "min": self.min_value,
            "max": self.max_value,
            "sum": self._sum,
        }


class ParallelStatsCalculator:
    """Parallel statistics calculator using multiprocessing.

    Distributes independent calculations across multiple workers.

    Example:
        >>> calc = ParallelStatsCalculator(max_workers=4)
        >>> results = calc.calculate_multiple([
        ...     ("percentiles", values1),
        ...     ("outliers", values2),
        ... ])
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> "ParallelStatsCalculator":
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def calculate_parallel(
        self,
        calculations: List[tuple[Callable[..., Any], tuple]],
    ) -> List[Any]:
        """Execute multiple calculations in parallel.

        Args:
            calculations: List of (function, args) tuples.

        Returns:
            List of results in same order as input.
        """
        if not self._executor:
            # Fallback to sequential if not in context manager
            return [func(*args) for func, args in calculations]

        futures = [
            self._executor.submit(func, *args) for func, args in calculations
        ]
        return [f.result() for f in futures]


# =============================================================================
# Phase 10.1.4: Implement Lazy Loading
# =============================================================================


@dataclass
class ResultSummary:
    """Lightweight summary of a benchmark result for list views.

    Contains only essential fields to minimize memory usage when
    displaying result lists.
    """

    benchmark_id: str
    backend_name: str
    execution_time_ms: float
    timestamp: datetime
    status: str
    circuit_hash: str


class LazyResultLoader(Generic[T]):
    """Lazy loader for benchmark results.

    Loads only summary data initially, fetching full results on demand.

    Example:
        >>> loader = LazyResultLoader(registry, limit=100)
        >>> for summary in loader.get_summaries():
        ...     if needs_details(summary):
        ...         full = loader.get_full_result(summary.benchmark_id)
    """

    def __init__(
        self,
        fetch_summary: Callable[[int, int], List[ResultSummary]],
        fetch_full: Callable[[str], T],
        page_size: int = 50,
    ) -> None:
        self._fetch_summary = fetch_summary
        self._fetch_full = fetch_full
        self._page_size = page_size
        self._cache: Dict[str, T] = {}

    def get_summaries(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> Iterator[ResultSummary]:
        """Iterate over result summaries with pagination.

        Yields:
            ResultSummary objects without loading full data.
        """
        current_offset = offset
        total_fetched = 0
        effective_limit = limit if limit is not None else float("inf")

        while total_fetched < effective_limit:
            page_limit = min(self._page_size, int(effective_limit - total_fetched))
            summaries = self._fetch_summary(page_limit, current_offset)

            if not summaries:
                break

            for summary in summaries:
                yield summary
                total_fetched += 1
                if total_fetched >= effective_limit:
                    break

            current_offset += len(summaries)

    def get_full_result(self, benchmark_id: str) -> T | None:
        """Load full result data on demand.

        Results are cached to avoid repeated fetches.
        """
        if benchmark_id in self._cache:
            return self._cache[benchmark_id]

        result = self._fetch_full(benchmark_id)
        if result is not None:
            self._cache[benchmark_id] = result
        return result

    def prefetch(self, benchmark_ids: List[str]) -> None:
        """Prefetch multiple results for batch operations."""
        for bid in benchmark_ids:
            if bid not in self._cache:
                result = self._fetch_full(bid)
                if result is not None:
                    self._cache[bid] = result

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()


class StreamingResultIterator:
    """Iterator that streams large result sets without loading all at once.

    Memory-efficient iteration over large datasets using cursor-based
    pagination.

    Example:
        >>> for result in StreamingResultIterator(registry, chunk_size=100):
        ...     process(result)
    """

    def __init__(
        self,
        fetch_page: Callable[[int, int], List[Any]],
        chunk_size: int = 100,
    ) -> None:
        self._fetch_page = fetch_page
        self._chunk_size = chunk_size
        self._offset = 0
        self._buffer: List[Any] = []
        self._exhausted = False

    def __iter__(self) -> "StreamingResultIterator":
        return self

    def __next__(self) -> Any:
        if not self._buffer and not self._exhausted:
            self._buffer = self._fetch_page(self._chunk_size, self._offset)
            self._offset += len(self._buffer)
            if len(self._buffer) < self._chunk_size:
                self._exhausted = True

        if not self._buffer:
            raise StopIteration

        return self._buffer.pop(0)

    def reset(self) -> None:
        """Reset the iterator to the beginning."""
        self._offset = 0
        self._buffer.clear()
        self._exhausted = False
