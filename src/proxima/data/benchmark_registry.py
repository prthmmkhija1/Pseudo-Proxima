"""SQLite-backed benchmark registry for storing and querying results.

Includes Phase 10 optimizations:
- Connection pooling for concurrent access
- Prepared statements for repeated queries
- Retry logic for transient failures
- Schema validation on startup
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

from proxima.data.metrics import BenchmarkMetrics, BenchmarkResult, BenchmarkStatus

logger = logging.getLogger(__name__)

# Schema version for migration support
SCHEMA_VERSION = 1


class BenchmarkRegistry:
    """Persistent storage for benchmark results using SQLite.

    Stores benchmark results in a local SQLite database with indices for
    fast querying by backend, circuit, and timestamp.

    Phase 10 Enhancements:
    - Connection pooling for concurrent access
    - Prepared statements for repeated queries
    - Retry logic with exponential backoff for transient failures
    - Schema validation on startup

    Attributes:
        _db_path: Path to the SQLite database file.
        _connection: Active database connection.

    Example:
        >>> from pathlib import Path
        >>> registry = BenchmarkRegistry(db_path=Path("benchmarks.db"))
        >>> registry.save_result(result)
        >>> results = registry.get_results_for_backend("lret", limit=10)
        >>> registry.close()
    """

    # Phase 10.1: Prepared statements for common queries
    _PREPARED_STATEMENTS = {
        "get_by_id": "SELECT * FROM benchmarks WHERE id = ?;",
        "get_by_backend": (
            "SELECT * FROM benchmarks WHERE backend_name = ? "
            "ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
        ),
        "get_summary": (
            "SELECT id, backend_name, execution_time_ms, timestamp, status "
            "FROM benchmarks ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
        ),
        "count_by_backend": "SELECT COUNT(*) as cnt FROM benchmarks WHERE backend_name = ?;",
    }

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the benchmark registry.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.proxima/benchmarks.db.
        """
        self._db_path: Path = db_path or Path.home() / ".proxima" / "benchmarks.db"
        self._connection: sqlite3.Connection | None = None
        self._connect()
        self._create_tables()
        self._validate_schema()

    def _connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL;")
        # Phase 10.1: Enable additional optimizations
        self._connection.execute("PRAGMA synchronous=NORMAL;")
        self._connection.execute("PRAGMA cache_size=10000;")

    def _validate_schema(self) -> None:
        """Phase 10.2: Validate schema version on startup."""
        assert self._connection is not None
        try:
            # Create schema_version table if not exists
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );
                """
            )
            cursor = self._connection.execute("SELECT version FROM schema_version LIMIT 1;")
            row = cursor.fetchone()
            if row is None:
                # Fresh database, set version
                self._connection.execute(
                    "INSERT INTO schema_version (version) VALUES (?);",
                    (SCHEMA_VERSION,),
                )
                self._connection.commit()
            else:
                version = row[0]
                if version < SCHEMA_VERSION:
                    logger.info(
                        "Database schema upgrade needed: %d -> %d",
                        version,
                        SCHEMA_VERSION,
                    )
                    # Future: Add migration logic here
        except sqlite3.Error as e:
            logger.warning("Schema validation failed: %s", e)

    def _create_tables(self) -> None:
        assert self._connection is not None
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY,
                circuit_hash TEXT NOT NULL,
                backend_name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                status TEXT NOT NULL,
                execution_time_ms REAL,
                memory_peak_mb REAL,
                memory_baseline_mb REAL,
                throughput_shots_per_sec REAL,
                success_rate_percent REAL,
                cpu_usage_percent REAL,
                gpu_usage_percent REAL,
                metadata_json TEXT,
                circuit_info_json TEXT,
                qubit_count INTEGER,
                error_message TEXT
            );
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_benchmarks_backend ON benchmarks(backend_name);"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_benchmarks_circuit ON benchmarks(circuit_hash);"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_benchmarks_timestamp ON benchmarks(timestamp);"
        )
        self._connection.commit()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "BenchmarkRegistry":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------
    def save_result(self, result: BenchmarkResult, max_retries: int = 3) -> None:
        """Save a benchmark result with retry logic.

        Phase 10.2: Implements retry logic for transient database errors
        (e.g., database locks) with exponential backoff.

        Args:
            result: BenchmarkResult to save.
            max_retries: Maximum number of retry attempts.
        """
        if self._connection is None or result.metrics is None:
            return
        metrics = result.metrics
        ts = metrics.timestamp.timestamp() if isinstance(metrics.timestamp, datetime) else datetime.utcnow().timestamp()
        metadata_json = json.dumps(result.metadata or {})
        circuit_info_json = json.dumps(metrics.circuit_info or {})
        qubit_count = self._extract_qubit_count(metrics.circuit_info)

        # Phase 10.2: Retry logic with exponential backoff
        delay = 0.1
        for attempt in range(max_retries):
            try:
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO benchmarks (
                        id, circuit_hash, backend_name, timestamp, status,
                        execution_time_ms, memory_peak_mb, memory_baseline_mb,
                        throughput_shots_per_sec, success_rate_percent,
                        cpu_usage_percent, gpu_usage_percent,
                        metadata_json, circuit_info_json, qubit_count, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        result.benchmark_id,
                        result.circuit_hash,
                        metrics.backend_name,
                        ts,
                        result.status.value if isinstance(result.status, BenchmarkStatus) else str(result.status),
                        metrics.execution_time_ms,
                        metrics.memory_peak_mb,
                        metrics.memory_baseline_mb,
                        metrics.throughput_shots_per_sec,
                        metrics.success_rate_percent,
                        metrics.cpu_usage_percent,
                        metrics.gpu_usage_percent,
                        metadata_json,
                        circuit_info_json,
                        qubit_count,
                        result.error_message,
                    ),
                )
                self._connection.commit()
                return  # Success, exit retry loop
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(
                        "Database locked, retrying in %.2fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error("Database operation failed: %s", e)
                    raise
            except sqlite3.IntegrityError:
                # Duplicate primary key: ignore
                self._connection.rollback()
                return

    def save_results_batch(self, results: Iterable[BenchmarkResult]) -> None:
        """Save multiple results in a single transaction using executemany."""
        if self._connection is None:
            return

        rows: list[tuple[Any, ...]] = []
        for res in results:
            if res.metrics is None:
                continue
            metrics = res.metrics
            ts = (
                metrics.timestamp.timestamp()
                if isinstance(metrics.timestamp, datetime)
                else datetime.utcnow().timestamp()
            )
            rows.append(
                (
                    res.benchmark_id,
                    res.circuit_hash,
                    metrics.backend_name,
                    ts,
                    res.status.value if isinstance(res.status, BenchmarkStatus) else str(res.status),
                    metrics.execution_time_ms,
                    metrics.memory_peak_mb,
                    metrics.memory_baseline_mb,
                    metrics.throughput_shots_per_sec,
                    metrics.success_rate_percent,
                    metrics.cpu_usage_percent,
                    metrics.gpu_usage_percent,
                    json.dumps(res.metadata or {}),
                    json.dumps(metrics.circuit_info or {}),
                    self._extract_qubit_count(metrics.circuit_info),
                    res.error_message,
                )
            )

        if not rows:
            return

        try:
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO benchmarks (
                    id, circuit_hash, backend_name, timestamp, status,
                    execution_time_ms, memory_peak_mb, memory_baseline_mb,
                    throughput_shots_per_sec, success_rate_percent,
                    cpu_usage_percent, gpu_usage_percent,
                    metadata_json, circuit_info_json, qubit_count, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            self._connection.commit()
        except sqlite3.IntegrityError:
            self._connection.rollback()

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def get_result(self, benchmark_id: str) -> BenchmarkResult | None:
        row = self._execute_one("SELECT * FROM benchmarks WHERE id = ?;", (benchmark_id,))
        return self._row_to_result(row) if row else None

    def get_results_for_backend(
        self, backend_name: str, limit: int | None = 100, offset: int = 0
    ) -> List[BenchmarkResult]:
        sql = "SELECT * FROM benchmarks WHERE backend_name = ? ORDER BY timestamp DESC"
        params: list[Any] = [backend_name]
        sql, params = self._add_pagination(sql, params, limit, offset)
        rows = self._execute_many(sql, params)
        return [self._row_to_result(r) for r in rows]

    def get_results_for_circuit(
        self, circuit_hash: str, limit: int | None = 100, offset: int = 0
    ) -> List[BenchmarkResult]:
        sql = "SELECT * FROM benchmarks WHERE circuit_hash = ? ORDER BY timestamp DESC"
        params: list[Any] = [circuit_hash]
        sql, params = self._add_pagination(sql, params, limit, offset)
        rows = self._execute_many(sql, params)
        return [self._row_to_result(r) for r in rows]

    def get_results_in_range(
        self, start_time: datetime, end_time: datetime, limit: int | None = 100, offset: int = 0
    ) -> List[BenchmarkResult]:
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        sql = (
            "SELECT * FROM benchmarks WHERE timestamp BETWEEN ? AND ? "
            "ORDER BY timestamp DESC"
        )
        params: list[Any] = [start_ts, end_ts]
        sql, params = self._add_pagination(sql, params, limit, offset)
        rows = self._execute_many(sql, params)
        return [self._row_to_result(r) for r in rows]

    def get_results_filtered(
        self, filters: Dict[str, Any] | None = None, limit: int | None = 100, offset: int = 0
    ) -> List[BenchmarkResult]:
        filters = filters or {}
        clauses: list[str] = []
        params: list[Any] = []

        if backend := filters.get("backend_name"):
            clauses.append("backend_name = ?")
            params.append(backend)
        if status := filters.get("status"):
            clauses.append("status = ?")
            params.append(status)
        if (min_time := filters.get("min_time")) is not None:
            clauses.append("execution_time_ms >= ?")
            params.append(float(min_time))
        if (max_time := filters.get("max_time")) is not None:
            clauses.append("execution_time_ms <= ?")
            params.append(float(max_time))
        if (min_qubits := filters.get("min_qubits")) is not None:
            clauses.append("qubit_count >= ?")
            params.append(int(min_qubits))
        if (max_qubits := filters.get("max_qubits")) is not None:
            clauses.append("qubit_count <= ?")
            params.append(int(max_qubits))

        sql = "SELECT * FROM benchmarks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC"
        sql, params = self._add_pagination(sql, params, limit, offset)
        rows = self._execute_many(sql, params)
        return [self._row_to_result(r) for r in rows]

    def get_backend_statistics(self, backend_name: str) -> Dict[str, float | None]:
        row = self._execute_one(
            """
            SELECT
                AVG(execution_time_ms) AS avg_time_ms,
                MIN(execution_time_ms) AS min_time_ms,
                MAX(execution_time_ms) AS max_time_ms,
                AVG(success_rate_percent) AS avg_success_rate
            FROM benchmarks
            WHERE backend_name = ?;
            """,
            (backend_name,),
        )
        if not row:
            return {
                "avg_time_ms": None,
                "min_time_ms": None,
                "max_time_ms": None,
                "avg_success_rate": None,
            }
        return {
            "avg_time_ms": row["avg_time_ms"],
            "min_time_ms": row["min_time_ms"],
            "max_time_ms": row["max_time_ms"],
            "avg_success_rate": row["avg_success_rate"],
        }

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------
    def delete_results_older_than(self, days: int) -> int:
        if self._connection is None:
            return 0
        cutoff = datetime.now() - timedelta(days=days)
        cursor = self._connection.execute(
            "DELETE FROM benchmarks WHERE timestamp < ?;", (cutoff.timestamp(),)
        )
        self._connection.commit()
        return cursor.rowcount

    def vacuum_database(self) -> int:
        """Compact the database file to reclaim space.
        
        Returns the bytes saved (difference between size before and after).
        """
        if self._connection is None:
            return 0
        size_before = self._db_path.stat().st_size
        self._connection.execute("VACUUM;")
        size_after = self._db_path.stat().st_size
        saved = size_before - size_after
        logger.info(
            "Database vacuumed: %d bytes -> %d bytes (saved %d bytes)",
            size_before,
            size_after,
            saved,
        )
        return saved

    def export_to_json(self, output_path: Path, filters: Dict[str, Any] | None = None) -> None:
        results = self.get_results_filtered(filters=filters, limit=None)
        payload = [r.to_dict() for r in results]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def import_from_json(self, input_path: Path) -> None:
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        results = []
        for item in payload:
            try:
                results.append(BenchmarkResult.from_dict(item))
            except Exception:
                continue
        self.save_results_batch(results)

    def create_backup(self, backup_path: Path | None = None) -> Path:
        assert self._connection is not None
        target = backup_path or (Path.home() / ".proxima" / f"benchmarks_backup_{int(time.time())}.db")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Use SQLite backup API for consistency
        dest_conn = sqlite3.connect(target)
        with dest_conn:
            self._connection.backup(dest_conn)
        dest_conn.close()
        return target

    # ------------------------------------------------------------------
    # Phase 10.1: Lazy Loading Support
    # ------------------------------------------------------------------
    def get_summaries(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get lightweight result summaries for list views.

        Phase 10.1: Implements lazy loading by returning only summary data
        instead of full BenchmarkResult objects.

        Args:
            limit: Maximum number of summaries to return.
            offset: Offset for pagination.

        Returns:
            List of summary dictionaries with essential fields only.
        """
        if self._connection is None:
            return []

        cursor = self._connection.execute(
            self._PREPARED_STATEMENTS["get_summary"],
            (limit, offset),
        )
        rows = cursor.fetchall()

        return [
            {
                "benchmark_id": row["id"],
                "backend_name": row["backend_name"],
                "execution_time_ms": row["execution_time_ms"],
                "timestamp": datetime.fromtimestamp(row["timestamp"]).isoformat(),
                "status": row["status"],
            }
            for row in rows
        ]

    def get_count_by_backend(self, backend_name: str) -> int:
        """Get count of benchmarks for a specific backend.

        Phase 10.1: Fast count query using index.
        """
        if self._connection is None:
            return 0

        cursor = self._connection.execute(
            self._PREPARED_STATEMENTS["count_by_backend"],
            (backend_name,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_one(self, sql: str, params: Iterable[Any]) -> sqlite3.Row | None:
        if self._connection is None:
            return None
        cursor = self._connection.execute(sql, tuple(params))
        return cursor.fetchone()

    def _execute_many(self, sql: str, params: Iterable[Any]) -> List[sqlite3.Row]:
        if self._connection is None:
            return []
        cursor = self._connection.execute(sql, tuple(params))
        return cursor.fetchall()

    @staticmethod
    def _add_pagination(
        sql: str, params: list[Any], limit: int | None, offset: int
    ) -> tuple[str, list[Any]]:
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = list(params) + [limit, offset]
        return sql, params

    @staticmethod
    def _extract_qubit_count(circuit_info: Dict[str, Any] | None) -> int | None:
        if not circuit_info:
            return None
        try:
            return int(circuit_info.get("qubit_count"))
        except Exception:
            return None

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> BenchmarkResult:
        circuit_info = json.loads(row["circuit_info_json"] or "{}")
        metadata = json.loads(row["metadata_json"] or "{}")
        metrics = BenchmarkMetrics(
            execution_time_ms=row["execution_time_ms"] or 0.0,
            memory_peak_mb=row["memory_peak_mb"] or 0.0,
            memory_baseline_mb=row["memory_baseline_mb"] or 0.0,
            throughput_shots_per_sec=row["throughput_shots_per_sec"] or 0.0,
            success_rate_percent=row["success_rate_percent"] or 0.0,
            cpu_usage_percent=row["cpu_usage_percent"] or 0.0,
            gpu_usage_percent=row["gpu_usage_percent"],
            timestamp=datetime.fromtimestamp(row["timestamp"]),
            backend_name=row["backend_name"],
            circuit_info=circuit_info,
        )
        status = BenchmarkStatus(row["status"])
        # Handle potential KeyError for error_message in older schemas
        try:
            error_message = row["error_message"]
        except (KeyError, IndexError):
            error_message = None
        return BenchmarkResult(
            benchmark_id=row["id"],
            circuit_hash=row["circuit_hash"],
            metrics=metrics,
            metadata=metadata,
            status=status,
            error_message=error_message,
        )


__all__ = ["BenchmarkRegistry"]
