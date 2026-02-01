"""Historical Metrics Storage and Analysis.

Phase 9: Agent Statistics & Telemetry System

Provides time-series storage and analysis:
- SQLite-based metric storage
- Rolling window retention (7 days)
- Statistical analysis
- Export to CSV/JSON
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.historical_metrics")


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""
    timestamp: float
    value: float
    metric_name: str
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "metric_name": self.metric_name,
            "labels": self.labels,
        }
    
    @property
    def datetime(self) -> datetime:
        """Get datetime from timestamp."""
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class TimeSeriesStats:
    """Statistics for a time series."""
    metric_name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    sum_value: float
    stddev: float
    first_timestamp: float
    last_timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "avg": self.avg_value,
            "sum": self.sum_value,
            "stddev": self.stddev,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
        }


@dataclass
class AggregatedSeries:
    """Time series aggregated by time bucket."""
    metric_name: str
    bucket_size_seconds: int
    buckets: List[Tuple[float, float]]  # (timestamp, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "bucket_size_seconds": self.bucket_size_seconds,
            "buckets": [{"timestamp": t, "value": v} for t, v in self.buckets],
        }


class HistoricalMetricsStore:
    """SQLite-based storage for historical metrics.
    
    Features:
    - Time-series storage in SQLite
    - Automatic cleanup of old data
    - Statistical queries
    - Export capabilities
    
    Example:
        >>> store = HistoricalMetricsStore()
        >>> 
        >>> # Record metric
        >>> store.record("llm.tokens", 1500)
        >>> 
        >>> # Query history
        >>> points = store.query("llm.tokens", hours=24)
        >>> 
        >>> # Get statistics
        >>> stats = store.get_stats("llm.tokens")
    """
    
    # Default retention period
    DEFAULT_RETENTION_DAYS = 7
    
    # Cleanup interval
    CLEANUP_INTERVAL_HOURS = 1
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ):
        """Initialize store.
        
        Args:
            db_path: Path to SQLite database
            retention_days: Days to retain data
        """
        self._db_path = Path(db_path) if db_path else Path.home() / ".proxima" / "metrics.db"
        self._retention_days = retention_days
        self._lock = threading.Lock()
        self._last_cleanup = 0.0
        
        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"HistoricalMetricsStore initialized: {self._db_path}")
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Main metrics table
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT  -- JSON
                );
                
                -- Index for efficient queries
                CREATE INDEX IF NOT EXISTS idx_metric_time 
                ON metrics(metric_name, timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON metrics(timestamp);
                
                -- Aggregated metrics table (hourly rollups)
                CREATE TABLE IF NOT EXISTS metrics_hourly (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hour_timestamp INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    min_value REAL,
                    max_value REAL,
                    avg_value REAL,
                    sum_value REAL,
                    count INTEGER,
                    UNIQUE(hour_timestamp, metric_name)
                );
                
                CREATE INDEX IF NOT EXISTS idx_hourly_metric 
                ON metrics_hourly(metric_name, hour_timestamp);
                
                -- Snapshots table for periodic telemetry snapshots
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    data TEXT NOT NULL  -- JSON
                );
                
                CREATE INDEX IF NOT EXISTS idx_snapshot_time
                ON snapshots(timestamp);
            """)
            conn.commit()
    
    def record(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp (defaults to now)
        """
        ts = timestamp or time.time()
        labels_json = json.dumps(labels) if labels else None
        
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO metrics (timestamp, metric_name, value, labels) VALUES (?, ?, ?, ?)",
                    (ts, metric_name, value, labels_json)
                )
                conn.commit()
        
        # Periodic cleanup
        self._maybe_cleanup()
    
    def record_batch(self, points: List[TimeSeriesPoint]) -> None:
        """Record multiple metric points.
        
        Args:
            points: List of TimeSeriesPoint
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.executemany(
                    "INSERT INTO metrics (timestamp, metric_name, value, labels) VALUES (?, ?, ?, ?)",
                    [
                        (p.timestamp, p.metric_name, p.value, json.dumps(p.labels) if p.labels else None)
                        for p in points
                    ]
                )
                conn.commit()
    
    def query(
        self,
        metric_name: str,
        hours: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 10000,
    ) -> List[TimeSeriesPoint]:
        """Query metric history.
        
        Args:
            metric_name: Name of the metric
            hours: Hours to look back
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum points to return
            
        Returns:
            List of TimeSeriesPoint
        """
        if hours:
            start_time = time.time() - (hours * 3600)
        
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, metric_name, value, labels
                FROM metrics
                WHERE metric_name = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (metric_name, start_time, end_time, limit)
            )
            
            points = []
            for row in cursor:
                labels = json.loads(row["labels"]) if row["labels"] else {}
                points.append(TimeSeriesPoint(
                    timestamp=row["timestamp"],
                    metric_name=row["metric_name"],
                    value=row["value"],
                    labels=labels,
                ))
            
            return points
    
    def query_all_metrics(
        self,
        hours: int = 24,
        limit_per_metric: int = 1000,
    ) -> Dict[str, List[TimeSeriesPoint]]:
        """Query all metrics.
        
        Args:
            hours: Hours to look back
            limit_per_metric: Maximum points per metric
            
        Returns:
            Dict of metric name to points
        """
        start_time = time.time() - (hours * 3600)
        
        with self._get_connection() as conn:
            # Get unique metric names
            cursor = conn.execute(
                "SELECT DISTINCT metric_name FROM metrics WHERE timestamp >= ?",
                (start_time,)
            )
            metric_names = [row["metric_name"] for row in cursor]
        
        result = {}
        for name in metric_names:
            result[name] = self.query(name, start_time=start_time, limit=limit_per_metric)
        
        return result
    
    def get_stats(
        self,
        metric_name: str,
        hours: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Optional[TimeSeriesStats]:
        """Get statistics for a metric.
        
        Args:
            metric_name: Metric name
            hours: Hours to analyze
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            TimeSeriesStats or None
        """
        if hours:
            start_time = time.time() - (hours * 3600)
        
        start_time = start_time or 0
        end_time = end_time or time.time()
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as count,
                    MIN(value) as min_value,
                    MAX(value) as max_value,
                    AVG(value) as avg_value,
                    SUM(value) as sum_value,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp
                FROM metrics
                WHERE metric_name = ? AND timestamp >= ? AND timestamp <= ?
                """,
                (metric_name, start_time, end_time)
            )
            
            row = cursor.fetchone()
            if not row or row["count"] == 0:
                return None
            
            # Calculate stddev
            cursor = conn.execute(
                """
                SELECT AVG((value - ?) * (value - ?)) as variance
                FROM metrics
                WHERE metric_name = ? AND timestamp >= ? AND timestamp <= ?
                """,
                (row["avg_value"], row["avg_value"], metric_name, start_time, end_time)
            )
            var_row = cursor.fetchone()
            stddev = (var_row["variance"] ** 0.5) if var_row and var_row["variance"] else 0
            
            return TimeSeriesStats(
                metric_name=metric_name,
                count=row["count"],
                min_value=row["min_value"],
                max_value=row["max_value"],
                avg_value=row["avg_value"],
                sum_value=row["sum_value"],
                stddev=stddev,
                first_timestamp=row["first_timestamp"],
                last_timestamp=row["last_timestamp"],
            )
    
    def aggregate_by_hour(
        self,
        metric_name: str,
        hours: int = 24,
    ) -> AggregatedSeries:
        """Aggregate metric by hour.
        
        Args:
            metric_name: Metric name
            hours: Hours to aggregate
            
        Returns:
            AggregatedSeries
        """
        start_time = time.time() - (hours * 3600)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    (CAST(timestamp / 3600 AS INTEGER) * 3600) as hour_bucket,
                    AVG(value) as avg_value
                FROM metrics
                WHERE metric_name = ? AND timestamp >= ?
                GROUP BY hour_bucket
                ORDER BY hour_bucket ASC
                """,
                (metric_name, start_time)
            )
            
            buckets = [(row["hour_bucket"], row["avg_value"]) for row in cursor]
            
            return AggregatedSeries(
                metric_name=metric_name,
                bucket_size_seconds=3600,
                buckets=buckets,
            )
    
    def aggregate_by_minute(
        self,
        metric_name: str,
        minutes: int = 60,
    ) -> AggregatedSeries:
        """Aggregate metric by minute.
        
        Args:
            metric_name: Metric name
            minutes: Minutes to aggregate
            
        Returns:
            AggregatedSeries
        """
        start_time = time.time() - (minutes * 60)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    (CAST(timestamp / 60 AS INTEGER) * 60) as minute_bucket,
                    AVG(value) as avg_value
                FROM metrics
                WHERE metric_name = ? AND timestamp >= ?
                GROUP BY minute_bucket
                ORDER BY minute_bucket ASC
                """,
                (metric_name, start_time)
            )
            
            buckets = [(row["minute_bucket"], row["avg_value"]) for row in cursor]
            
            return AggregatedSeries(
                metric_name=metric_name,
                bucket_size_seconds=60,
                buckets=buckets,
            )
    
    def get_metric_names(self) -> List[str]:
        """Get all unique metric names."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT metric_name FROM metrics")
            return [row["metric_name"] for row in cursor]
    
    def save_snapshot(self, snapshot_data: Dict[str, Any]) -> None:
        """Save a telemetry snapshot.
        
        Args:
            snapshot_data: Snapshot data as dictionary
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO snapshots (timestamp, data) VALUES (?, ?)",
                    (time.time(), json.dumps(snapshot_data))
                )
                conn.commit()
    
    def get_snapshots(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent snapshots.
        
        Args:
            hours: Hours to look back
            limit: Maximum snapshots
            
        Returns:
            List of snapshot data
        """
        start_time = time.time() - (hours * 3600)
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, data FROM snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (start_time, limit)
            )
            
            return [
                {"timestamp": row["timestamp"], **json.loads(row["data"])}
                for row in cursor
            ]
    
    def _maybe_cleanup(self) -> None:
        """Run cleanup if needed."""
        now = time.time()
        if now - self._last_cleanup < self.CLEANUP_INTERVAL_HOURS * 3600:
            return
        
        self._last_cleanup = now
        self.cleanup()
    
    def cleanup(self, retention_days: Optional[int] = None) -> int:
        """Clean up old data.
        
        Args:
            retention_days: Days to retain (uses default if not specified)
            
        Returns:
            Number of rows deleted
        """
        days = retention_days or self._retention_days
        cutoff = time.time() - (days * 24 * 3600)
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM metrics WHERE timestamp < ?",
                    (cutoff,)
                )
                deleted_metrics = cursor.rowcount
                
                cursor = conn.execute(
                    "DELETE FROM snapshots WHERE timestamp < ?",
                    (cutoff,)
                )
                deleted_snapshots = cursor.rowcount
                
                conn.commit()
        
        total_deleted = deleted_metrics + deleted_snapshots
        if total_deleted > 0:
            logger.info(f"Cleaned up {total_deleted} old records")
        
        return total_deleted
    
    def vacuum(self) -> None:
        """Vacuum database to reclaim space."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
        logger.info("Database vacuumed")
    
    def export_csv(
        self,
        output_path: str,
        metric_name: Optional[str] = None,
        hours: int = 24,
    ) -> int:
        """Export metrics to CSV.
        
        Args:
            output_path: Output file path
            metric_name: Optional specific metric
            hours: Hours to export
            
        Returns:
            Number of rows exported
        """
        import csv
        
        if metric_name:
            points = self.query(metric_name, hours=hours)
        else:
            all_metrics = self.query_all_metrics(hours=hours)
            points = []
            for name, metric_points in all_metrics.items():
                points.extend(metric_points)
        
        # Sort by timestamp
        points.sort(key=lambda p: p.timestamp)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'datetime', 'metric_name', 'value', 'labels'])
            
            for point in points:
                writer.writerow([
                    point.timestamp,
                    point.datetime.isoformat(),
                    point.metric_name,
                    point.value,
                    json.dumps(point.labels) if point.labels else '',
                ])
        
        return len(points)
    
    def export_json(
        self,
        output_path: str,
        metric_name: Optional[str] = None,
        hours: int = 24,
    ) -> int:
        """Export metrics to JSON.
        
        Args:
            output_path: Output file path
            metric_name: Optional specific metric
            hours: Hours to export
            
        Returns:
            Number of records exported
        """
        if metric_name:
            data = {metric_name: [p.to_dict() for p in self.query(metric_name, hours=hours)]}
        else:
            all_metrics = self.query_all_metrics(hours=hours)
            data = {name: [p.to_dict() for p in points] for name, points in all_metrics.items()}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        total = sum(len(points) for points in data.values())
        return total
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Storage statistics
        """
        with self._get_connection() as conn:
            # Count records
            cursor = conn.execute("SELECT COUNT(*) as count FROM metrics")
            metrics_count = cursor.fetchone()["count"]
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM snapshots")
            snapshots_count = cursor.fetchone()["count"]
            
            # Get time range
            cursor = conn.execute("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM metrics")
            row = cursor.fetchone()
            
            # File size
            file_size = self._db_path.stat().st_size if self._db_path.exists() else 0
        
        return {
            "metrics_count": metrics_count,
            "snapshots_count": snapshots_count,
            "first_record": datetime.fromtimestamp(row["min_ts"]).isoformat() if row["min_ts"] else None,
            "last_record": datetime.fromtimestamp(row["max_ts"]).isoformat() if row["max_ts"] else None,
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "db_path": str(self._db_path),
        }


# ========== Global Instance ==========

_historical_store: Optional[HistoricalMetricsStore] = None


def get_historical_store() -> HistoricalMetricsStore:
    """Get the global HistoricalMetricsStore instance."""
    global _historical_store
    if _historical_store is None:
        _historical_store = HistoricalMetricsStore()
    return _historical_store


def record_metric(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Convenience function to record a metric."""
    get_historical_store().record(metric_name, value, labels)
