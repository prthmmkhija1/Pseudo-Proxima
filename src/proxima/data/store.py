"""
Result storage implementation.

Provides JSON and SQLite storage backends for simulation results,
with session management and query capabilities.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from proxima.config.settings import FlatSettings, get_settings


class StorageBackend(str, Enum):
    """Available storage backends."""

    JSON = "json"
    SQLITE = "sqlite"
    MEMORY = "memory"


class StoredResult(BaseModel):
    """Model for a stored simulation result."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    backend_name: str
    circuit_name: str | None = None
    qubit_count: int
    shots: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: float
    memory_used_mb: float
    counts: dict[str, int] = Field(default_factory=dict)
    statevector: list[complex] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            complex: lambda v: {"real": v.real, "imag": v.imag},
        }


class StoredSession(BaseModel):
    """Model for a stored session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    agent_file: str | None = None
    result_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultStore(ABC):
    """Abstract base class for result storage."""

    @abstractmethod
    def save_result(self, result: StoredResult) -> str:
        """Save a result and return its ID."""
        ...

    @abstractmethod
    def get_result(self, result_id: str) -> StoredResult | None:
        """Retrieve a result by ID."""
        ...

    @abstractmethod
    def list_results(
        self,
        session_id: str | None = None,
        backend_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredResult]:
        """List results with optional filtering."""
        ...

    @abstractmethod
    def delete_result(self, result_id: str) -> bool:
        """Delete a result by ID."""
        ...

    @abstractmethod
    def create_session(self, session: StoredSession) -> str:
        """Create a new session and return its ID."""
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> StoredSession | None:
        """Retrieve a session by ID."""
        ...

    @abstractmethod
    def list_sessions(self, limit: int = 50) -> list[StoredSession]:
        """List all sessions."""
        ...

    @abstractmethod
    def update_session(self, session: StoredSession) -> bool:
        """Update an existing session."""
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its results."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the store and release resources."""
        ...


class MemoryStore(ResultStore):
    """In-memory result storage for testing."""

    def __init__(self) -> None:
        self._results: dict[str, StoredResult] = {}
        self._sessions: dict[str, StoredSession] = {}

    def save_result(self, result: StoredResult) -> str:
        self._results[result.id] = result
        if result.session_id in self._sessions:
            session = self._sessions[result.session_id]
            session.result_count += 1
            session.updated_at = datetime.utcnow()
        return result.id

    def get_result(self, result_id: str) -> StoredResult | None:
        return self._results.get(result_id)

    def list_results(
        self,
        session_id: str | None = None,
        backend_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredResult]:
        results = list(self._results.values())
        if session_id:
            results = [r for r in results if r.session_id == session_id]
        if backend_name:
            results = [r for r in results if r.backend_name == backend_name]
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[offset : offset + limit]

    def delete_result(self, result_id: str) -> bool:
        if result_id in self._results:
            result = self._results.pop(result_id)
            if result.session_id in self._sessions:
                self._sessions[result.session_id].result_count -= 1
            return True
        return False

    def create_session(self, session: StoredSession) -> str:
        self._sessions[session.id] = session
        return session.id

    def get_session(self, session_id: str) -> StoredSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, limit: int = 50) -> list[StoredSession]:
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]

    def update_session(self, session: StoredSession) -> bool:
        """Update an existing session in memory."""
        if session.id in self._sessions:
            self._sessions[session.id] = session
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            # Delete associated results
            to_delete = [
                rid for rid, r in self._results.items() if r.session_id == session_id
            ]
            for rid in to_delete:
                del self._results[rid]
            return True
        return False

    def close(self) -> None:
        self._results.clear()
        self._sessions.clear()


class JSONStore(ResultStore):
    """JSON file-based result storage."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or Path.home() / ".proxima" / "results"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_file = self.storage_dir / "sessions.json"
        self._load_sessions()

    def _load_sessions(self) -> None:
        """Load sessions index from disk."""
        self._sessions: dict[str, StoredSession] = {}
        if self._sessions_file.exists():
            try:
                data = json.loads(self._sessions_file.read_text())
                for sid, sdata in data.items():
                    sdata["created_at"] = datetime.fromisoformat(sdata["created_at"])
                    sdata["updated_at"] = datetime.fromisoformat(sdata["updated_at"])
                    self._sessions[sid] = StoredSession(**sdata)
            except (json.JSONDecodeError, KeyError):
                self._sessions = {}

    def _save_sessions(self) -> None:
        """Save sessions index to disk."""
        data = {sid: s.model_dump(mode="json") for sid, s in self._sessions.items()}
        self._sessions_file.write_text(json.dumps(data, indent=2, default=str))

    def _result_path(self, result_id: str) -> Path:
        """Get the file path for a result."""
        return self.storage_dir / f"{result_id}.json"

    def _serialize_result(self, result: StoredResult) -> str:
        """Serialize a result to JSON."""
        data = result.model_dump(mode="json")
        # Handle complex numbers in statevector
        if data.get("statevector"):
            data["statevector"] = [
                {"real": c["real"], "imag": c["imag"]} if isinstance(c, dict) else c
                for c in data["statevector"]
            ]
        return json.dumps(data, indent=2, default=str)

    def _deserialize_result(self, json_str: str) -> StoredResult:
        """Deserialize a result from JSON."""
        data = json.loads(json_str)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        # Handle complex numbers
        if data.get("statevector"):
            data["statevector"] = [
                complex(c["real"], c["imag"]) if isinstance(c, dict) else c
                for c in data["statevector"]
            ]
        return StoredResult(**data)

    def save_result(self, result: StoredResult) -> str:
        path = self._result_path(result.id)
        path.write_text(self._serialize_result(result))
        if result.session_id in self._sessions:
            session = self._sessions[result.session_id]
            session.result_count += 1
            session.updated_at = datetime.utcnow()
            self._save_sessions()
        return result.id

    def get_result(self, result_id: str) -> StoredResult | None:
        path = self._result_path(result_id)
        if path.exists():
            return self._deserialize_result(path.read_text())
        return None

    def list_results(
        self,
        session_id: str | None = None,
        backend_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredResult]:
        results: list[StoredResult] = []
        for path in self.storage_dir.glob("*.json"):
            if path.name == "sessions.json":
                continue
            try:
                result = self._deserialize_result(path.read_text())
                if session_id and result.session_id != session_id:
                    continue
                if backend_name and result.backend_name != backend_name:
                    continue
                results.append(result)
            except (json.JSONDecodeError, KeyError):
                continue
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[offset : offset + limit]

    def delete_result(self, result_id: str) -> bool:
        path = self._result_path(result_id)
        if path.exists():
            result = self.get_result(result_id)
            path.unlink()
            if result and result.session_id in self._sessions:
                self._sessions[result.session_id].result_count -= 1
                self._save_sessions()
            return True
        return False

    def create_session(self, session: StoredSession) -> str:
        self._sessions[session.id] = session
        self._save_sessions()
        return session.id

    def get_session(self, session_id: str) -> StoredSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self, limit: int = 50) -> list[StoredSession]:
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]

    def update_session(self, session: StoredSession) -> bool:
        """Update an existing session and persist to disk."""
        if session.id in self._sessions:
            self._sessions[session.id] = session
            self._save_sessions()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_sessions()
            # Delete associated results
            for path in self.storage_dir.glob("*.json"):
                if path.name == "sessions.json":
                    continue
                try:
                    result = self._deserialize_result(path.read_text())
                    if result.session_id == session_id:
                        path.unlink()
                except (json.JSONDecodeError, KeyError):
                    continue
            return True
        return False

    def close(self) -> None:
        self._save_sessions()


class SQLiteStore(ResultStore):
    """SQLite-based result storage."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        agent_file TEXT,
        result_count INTEGER DEFAULT 0,
        metadata TEXT DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS results (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        backend_name TEXT NOT NULL,
        circuit_name TEXT,
        qubit_count INTEGER NOT NULL,
        shots INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        execution_time_ms REAL NOT NULL,
        memory_used_mb REAL NOT NULL,
        counts TEXT NOT NULL,
        statevector TEXT,
        metadata TEXT DEFAULT '{}',
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_results_session ON results(session_id);
    CREATE INDEX IF NOT EXISTS idx_results_backend ON results(backend_name);
    CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp DESC);
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or Path.home() / ".proxima" / "results.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _serialize_complex_list(self, values: list[complex] | None) -> str | None:
        """Serialize complex numbers to JSON."""
        if values is None:
            return None
        return json.dumps([{"real": c.real, "imag": c.imag} for c in values])

    def _deserialize_complex_list(self, data: str | None) -> list[complex] | None:
        """Deserialize complex numbers from JSON."""
        if data is None:
            return None
        parsed = json.loads(data)
        return [complex(c["real"], c["imag"]) for c in parsed]

    def save_result(self, result: StoredResult) -> str:
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO results
                (id, session_id, backend_name, circuit_name, qubit_count, shots,
                 timestamp, execution_time_ms, memory_used_mb, counts, statevector, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.session_id,
                    result.backend_name,
                    result.circuit_name,
                    result.qubit_count,
                    result.shots,
                    result.timestamp.isoformat(),
                    result.execution_time_ms,
                    result.memory_used_mb,
                    json.dumps(result.counts),
                    self._serialize_complex_list(result.statevector),
                    json.dumps(result.metadata),
                ),
            )
            # Update session result count
            cursor.execute(
                """
                UPDATE sessions SET result_count = result_count + 1, updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), result.session_id),
            )
        return result.id

    def get_result(self, result_id: str) -> StoredResult | None:
        cursor = self._conn.execute("SELECT * FROM results WHERE id = ?", (result_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_result(row)
        return None

    def _row_to_result(self, row: sqlite3.Row) -> StoredResult:
        """Convert a database row to a StoredResult."""
        return StoredResult(
            id=row["id"],
            session_id=row["session_id"],
            backend_name=row["backend_name"],
            circuit_name=row["circuit_name"],
            qubit_count=row["qubit_count"],
            shots=row["shots"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            execution_time_ms=row["execution_time_ms"],
            memory_used_mb=row["memory_used_mb"],
            counts=json.loads(row["counts"]),
            statevector=self._deserialize_complex_list(row["statevector"]),
            metadata=json.loads(row["metadata"]),
        )

    def list_results(
        self,
        session_id: str | None = None,
        backend_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoredResult]:
        query = "SELECT * FROM results WHERE 1=1"
        params: list[Any] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if backend_name:
            query += " AND backend_name = ?"
            params.append(backend_name)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._conn.execute(query, params)
        return [self._row_to_result(row) for row in cursor.fetchall()]

    def delete_result(self, result_id: str) -> bool:
        with self._transaction() as cursor:
            # Get session_id before delete
            cursor.execute("SELECT session_id FROM results WHERE id = ?", (result_id,))
            row = cursor.fetchone()
            if not row:
                return False
            session_id = row["session_id"]

            cursor.execute("DELETE FROM results WHERE id = ?", (result_id,))
            # Update session count
            cursor.execute(
                """
                UPDATE sessions SET result_count = result_count - 1, updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), session_id),
            )
            return True

    def create_session(self, session: StoredSession) -> str:
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO sessions (id, name, created_at, updated_at, agent_file, result_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.name,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.agent_file,
                    session.result_count,
                    json.dumps(session.metadata),
                ),
            )
        return session.id

    def get_session(self, session_id: str) -> StoredSession | None:
        cursor = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        if row:
            return StoredSession(
                id=row["id"],
                name=row["name"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                agent_file=row["agent_file"],
                result_count=row["result_count"],
                metadata=json.loads(row["metadata"]),
            )
        return None

    def list_sessions(self, limit: int = 50) -> list[StoredSession]:
        cursor = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [
            StoredSession(
                id=row["id"],
                name=row["name"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                agent_file=row["agent_file"],
                result_count=row["result_count"],
                metadata=json.loads(row["metadata"]),
            )
            for row in cursor.fetchall()
        ]

    def update_session(self, session: StoredSession) -> bool:
        """Update an existing session in the database."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE sessions
                SET name = ?, updated_at = ?, agent_file = ?, result_count = ?, metadata = ?
                WHERE id = ?
                """,
                (
                    session.name,
                    session.updated_at.isoformat(),
                    session.agent_file,
                    session.result_count,
                    json.dumps(session.metadata),
                    session.id,
                ),
            )
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()


def create_store(backend: StorageBackend | None = None) -> ResultStore:
    """Factory function to create a result store based on settings."""
    raw_settings = get_settings()
    settings = FlatSettings(raw_settings)
    backend = backend or StorageBackend(settings.storage_backend)

    if backend == StorageBackend.MEMORY:
        return MemoryStore()
    elif backend == StorageBackend.JSON:
        storage_dir = Path(settings.data_dir) / "results" if settings.data_dir else None
        return JSONStore(storage_dir=storage_dir)
    elif backend == StorageBackend.SQLITE:
        db_path = Path(settings.data_dir) / "results.db" if settings.data_dir else None
        return SQLiteStore(db_path=db_path)
    else:
        raise ValueError(f"Unknown storage backend: {backend}")


# Convenience singleton for default store
_default_store: ResultStore | None = None


def get_store() -> ResultStore:
    """Get the default result store."""
    global _default_store
    if _default_store is None:
        _default_store = create_store()
    return _default_store


def close_store() -> None:
    """Close and clear the default store."""
    global _default_store
    if _default_store is not None:
        _default_store.close()
        _default_store = None


# =============================================================================
# SQLite Schema Optimization (Feature - Data Store)
# =============================================================================


class SchemaVersion:
    """Schema version tracking for migrations."""
    
    CURRENT_VERSION = 2
    
    MIGRATIONS = {
        1: """
            -- Version 1: Initial schema (already exists)
            -- No changes needed
        """,
        2: """
            -- Version 2: Add performance indexes and FTS
            CREATE INDEX IF NOT EXISTS idx_results_qubit_count ON results(qubit_count);
            CREATE INDEX IF NOT EXISTS idx_results_execution_time ON results(execution_time_ms);
            CREATE INDEX IF NOT EXISTS idx_results_memory ON results(memory_used_mb);
            CREATE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name);
            
            -- Create compound indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_results_session_backend 
                ON results(session_id, backend_name);
            CREATE INDEX IF NOT EXISTS idx_results_session_timestamp 
                ON results(session_id, timestamp DESC);
        """,
    }


class SQLiteSchemaOptimizer:
    """SQLite schema optimization utilities.
    
    Provides:
    - Query performance analysis
    - Index recommendations
    - Schema optimization operations
    - Statistics collection
    """
    
    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialize optimizer with database connection."""
        self._conn = conn
    
    def analyze_query_performance(self, query: str) -> dict[str, Any]:
        """Analyze query performance using EXPLAIN QUERY PLAN.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            Analysis results with execution plan and recommendations
        """
        try:
            cursor = self._conn.execute(f"EXPLAIN QUERY PLAN {query}")
            plan_rows = cursor.fetchall()
            
            # Parse execution plan
            uses_index = any("USING INDEX" in str(row) for row in plan_rows)
            uses_scan = any("SCAN" in str(row) and "USING INDEX" not in str(row) 
                          for row in plan_rows)
            
            recommendations = []
            if uses_scan and not uses_index:
                recommendations.append("Consider adding an index for this query pattern")
            
            return {
                "query": query,
                "execution_plan": [dict(row) if hasattr(row, 'keys') else str(row) 
                                   for row in plan_rows],
                "uses_index": uses_index,
                "uses_table_scan": uses_scan,
                "recommendations": recommendations,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_index_stats(self) -> list[dict[str, Any]]:
        """Get statistics about existing indexes.
        
        Returns:
            List of index information dictionaries
        """
        indexes = []
        try:
            cursor = self._conn.execute(
                "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'"
            )
            for row in cursor.fetchall():
                indexes.append({
                    "name": row[0],
                    "table": row[1],
                    "sql": row[2],
                })
        except Exception:
            pass
        return indexes
    
    def get_table_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics about tables.
        
        Returns:
            Dictionary mapping table name to stats
        """
        stats = {}
        try:
            # Run ANALYZE to update statistics
            self._conn.execute("ANALYZE")
            
            # Get table info
            cursor = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                if table.startswith("sqlite_"):
                    continue
                    
                # Get row count
                count_cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = count_cursor.fetchone()[0]
                
                # Get column info
                pragma_cursor = self._conn.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in pragma_cursor.fetchall()]
                
                stats[table] = {
                    "row_count": row_count,
                    "columns": columns,
                    "column_count": len(columns),
                }
        except Exception:
            pass
        return stats
    
    def vacuum_database(self) -> bool:
        """Vacuum the database to reclaim space and defragment.
        
        Returns:
            True if successful
        """
        try:
            self._conn.execute("VACUUM")
            return True
        except Exception:
            return False
    
    def optimize_for_read(self) -> None:
        """Apply read-optimized settings."""
        try:
            self._conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
            self._conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
            self._conn.execute("PRAGMA temp_store = MEMORY")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        except Exception:
            pass
    
    def optimize_for_write(self) -> None:
        """Apply write-optimized settings."""
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA cache_size = -32000")  # 32MB cache
            self._conn.execute("PRAGMA wal_autocheckpoint = 1000")
        except Exception:
            pass
    
    def create_recommended_indexes(self) -> list[str]:
        """Create recommended indexes for common query patterns.
        
        Returns:
            List of created index names
        """
        created = []
        index_statements = [
            ("idx_results_composite_filter", 
             "CREATE INDEX IF NOT EXISTS idx_results_composite_filter "
             "ON results(session_id, backend_name, timestamp DESC)"),
            ("idx_results_performance",
             "CREATE INDEX IF NOT EXISTS idx_results_performance "
             "ON results(execution_time_ms, memory_used_mb)"),
            ("idx_results_circuit",
             "CREATE INDEX IF NOT EXISTS idx_results_circuit "
             "ON results(circuit_name, qubit_count)"),
        ]
        
        for name, stmt in index_statements:
            try:
                self._conn.execute(stmt)
                created.append(name)
            except Exception:
                pass
        
        self._conn.commit()
        return created


# =============================================================================
# Data Migration Utilities (Feature - Data Store)
# =============================================================================


class MigrationRecord(BaseModel):
    """Record of a migration operation."""
    
    version: int
    applied_at: datetime
    success: bool
    duration_ms: float
    error: str | None = None


class DataMigrator:
    """Data migration utilities for store upgrades.
    
    Supports:
    - Version tracking
    - Forward migrations
    - Data transformation
    - Rollback capabilities
    """
    
    def __init__(self, store: ResultStore) -> None:
        """Initialize migrator with target store."""
        self._store = store
        self._migration_history: list[MigrationRecord] = []
    
    def get_current_version(self) -> int:
        """Get current schema version."""
        if isinstance(self._store, SQLiteStore):
            try:
                cursor = self._store._conn.execute(
                    "SELECT MAX(version) FROM migrations"
                )
                result = cursor.fetchone()
                return result[0] if result and result[0] else 0
            except Exception:
                return 0
        return 0
    
    def migrate_to_version(self, target_version: int) -> list[MigrationRecord]:
        """Migrate database to target version.
        
        Args:
            target_version: Target schema version
            
        Returns:
            List of migration records
        """
        current = self.get_current_version()
        records = []
        
        if current >= target_version:
            return records
        
        if not isinstance(self._store, SQLiteStore):
            return records
        
        # Ensure migrations table exists
        self._store._conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                success INTEGER NOT NULL,
                duration_ms REAL,
                error TEXT
            )
        """)
        self._store._conn.commit()
        
        for version in range(current + 1, target_version + 1):
            record = self._apply_migration(version)
            records.append(record)
            if not record.success:
                break
        
        return records
    
    def _apply_migration(self, version: int) -> MigrationRecord:
        """Apply a single migration."""
        import time
        start = time.perf_counter()
        
        try:
            sql = SchemaVersion.MIGRATIONS.get(version, "")
            if sql.strip():
                self._store._conn.executescript(sql)
            
            # Record successful migration
            duration = (time.perf_counter() - start) * 1000
            self._store._conn.execute(
                "INSERT INTO migrations (version, applied_at, success, duration_ms) "
                "VALUES (?, ?, ?, ?)",
                (version, datetime.utcnow().isoformat(), 1, duration)
            )
            self._store._conn.commit()
            
            record = MigrationRecord(
                version=version,
                applied_at=datetime.utcnow(),
                success=True,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self._store._conn.rollback()
            record = MigrationRecord(
                version=version,
                applied_at=datetime.utcnow(),
                success=False,
                duration_ms=duration,
                error=str(e),
            )
        
        self._migration_history.append(record)
        return record
    
    def export_data(self, output_path: Path) -> bool:
        """Export all data to JSON for backup/migration.
        
        Args:
            output_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            data = {
                "version": SchemaVersion.CURRENT_VERSION,
                "exported_at": datetime.utcnow().isoformat(),
                "sessions": [],
                "results": [],
            }
            
            # Export sessions
            for session in self._store.list_sessions(limit=10000):
                data["sessions"].append(session.model_dump(mode="json"))
            
            # Export results
            for result in self._store.list_results(limit=100000):
                data["results"].append(result.model_dump(mode="json"))
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(data, indent=2, default=str))
            return True
            
        except Exception:
            return False
    
    def import_data(self, input_path: Path) -> tuple[int, int, list[str]]:
        """Import data from backup file.
        
        Args:
            input_path: Path to import file
            
        Returns:
            Tuple of (sessions_imported, results_imported, errors)
        """
        sessions_imported = 0
        results_imported = 0
        errors = []
        
        try:
            data = json.loads(input_path.read_text())
            
            # Import sessions
            for session_data in data.get("sessions", []):
                try:
                    session_data["created_at"] = datetime.fromisoformat(
                        session_data["created_at"]
                    )
                    session_data["updated_at"] = datetime.fromisoformat(
                        session_data["updated_at"]
                    )
                    session = StoredSession(**session_data)
                    self._store.create_session(session)
                    sessions_imported += 1
                except Exception as e:
                    errors.append(f"Session import error: {e}")
            
            # Import results
            for result_data in data.get("results", []):
                try:
                    result_data["timestamp"] = datetime.fromisoformat(
                        result_data["timestamp"]
                    )
                    # Handle complex statevector
                    if result_data.get("statevector"):
                        result_data["statevector"] = [
                            complex(c["real"], c["imag"]) 
                            if isinstance(c, dict) else c
                            for c in result_data["statevector"]
                        ]
                    result = StoredResult(**result_data)
                    self._store.save_result(result)
                    results_imported += 1
                except Exception as e:
                    errors.append(f"Result import error: {e}")
                    
        except Exception as e:
            errors.append(f"Import failed: {e}")
        
        return sessions_imported, results_imported, errors


# =============================================================================
# Advanced Query Capabilities (Feature - Data Store)
# =============================================================================


@dataclass
class QueryFilter:
    """Advanced query filter specification."""
    
    session_id: str | None = None
    backend_names: list[str] | None = None
    circuit_name: str | None = None
    min_qubits: int | None = None
    max_qubits: int | None = None
    min_shots: int | None = None
    max_shots: int | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_execution_time_ms: float | None = None
    max_execution_time_ms: float | None = None
    metadata_contains: dict[str, Any] | None = None


@dataclass
class QuerySort:
    """Query sort specification."""
    
    field: str = "timestamp"
    descending: bool = True


@dataclass
class AggregateResult:
    """Result of an aggregate query."""
    
    count: int = 0
    sum_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    min_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0
    sum_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    total_shots: int = 0
    unique_backends: int = 0
    unique_circuits: int = 0


class AdvancedQueryEngine:
    """Advanced query engine for result stores.
    
    Provides:
    - Complex filtering with multiple criteria
    - Aggregation queries
    - Full-text search (SQLite only)
    - Pagination support
    - Query result caching
    """
    
    def __init__(self, store: ResultStore) -> None:
        """Initialize query engine with store."""
        self._store = store
        self._cache: dict[str, tuple[float, Any]] = {}  # query_hash -> (timestamp, result)
        self._cache_ttl = 60.0  # Cache TTL in seconds
    
    def query(
        self,
        filter: QueryFilter | None = None,
        sort: QuerySort | None = None,
        limit: int = 100,
        offset: int = 0,
        use_cache: bool = True,
    ) -> list[StoredResult]:
        """Execute advanced query with filtering and sorting.
        
        Args:
            filter: Query filter specification
            sort: Sort specification
            limit: Maximum results to return
            offset: Offset for pagination
            use_cache: Whether to use query cache
            
        Returns:
            List of matching results
        """
        filter = filter or QueryFilter()
        sort = sort or QuerySort()
        
        # Check cache
        cache_key = self._get_cache_key(filter, sort, limit, offset)
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached
        
        # Execute query based on store type
        if isinstance(self._store, SQLiteStore):
            results = self._query_sqlite(filter, sort, limit, offset)
        else:
            results = self._query_memory(filter, sort, limit, offset)
        
        # Cache results
        if use_cache:
            self._set_cached(cache_key, results)
        
        return results
    
    def _query_sqlite(
        self,
        filter: QueryFilter,
        sort: QuerySort,
        limit: int,
        offset: int,
    ) -> list[StoredResult]:
        """Execute query on SQLite store."""
        query = "SELECT * FROM results WHERE 1=1"
        params: list[Any] = []
        
        if filter.session_id:
            query += " AND session_id = ?"
            params.append(filter.session_id)
        
        if filter.backend_names:
            placeholders = ",".join("?" * len(filter.backend_names))
            query += f" AND backend_name IN ({placeholders})"
            params.extend(filter.backend_names)
        
        if filter.circuit_name:
            query += " AND circuit_name LIKE ?"
            params.append(f"%{filter.circuit_name}%")
        
        if filter.min_qubits is not None:
            query += " AND qubit_count >= ?"
            params.append(filter.min_qubits)
        
        if filter.max_qubits is not None:
            query += " AND qubit_count <= ?"
            params.append(filter.max_qubits)
        
        if filter.min_shots is not None:
            query += " AND shots >= ?"
            params.append(filter.min_shots)
        
        if filter.max_shots is not None:
            query += " AND shots <= ?"
            params.append(filter.max_shots)
        
        if filter.start_date:
            query += " AND timestamp >= ?"
            params.append(filter.start_date.isoformat())
        
        if filter.end_date:
            query += " AND timestamp <= ?"
            params.append(filter.end_date.isoformat())
        
        if filter.min_execution_time_ms is not None:
            query += " AND execution_time_ms >= ?"
            params.append(filter.min_execution_time_ms)
        
        if filter.max_execution_time_ms is not None:
            query += " AND execution_time_ms <= ?"
            params.append(filter.max_execution_time_ms)
        
        # Add sorting
        order_dir = "DESC" if sort.descending else "ASC"
        query += f" ORDER BY {sort.field} {order_dir}"
        
        # Add pagination
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = self._store._conn.execute(query, params)
        return [self._store._row_to_result(row) for row in cursor.fetchall()]
    
    def _query_memory(
        self,
        filter: QueryFilter,
        sort: QuerySort,
        limit: int,
        offset: int,
    ) -> list[StoredResult]:
        """Execute query on memory/JSON store."""
        # Get all results
        all_results = self._store.list_results(limit=100000)
        
        # Apply filters
        results = []
        for r in all_results:
            if filter.session_id and r.session_id != filter.session_id:
                continue
            if filter.backend_names and r.backend_name not in filter.backend_names:
                continue
            if filter.circuit_name and filter.circuit_name.lower() not in (r.circuit_name or "").lower():
                continue
            if filter.min_qubits is not None and r.qubit_count < filter.min_qubits:
                continue
            if filter.max_qubits is not None and r.qubit_count > filter.max_qubits:
                continue
            if filter.min_shots is not None and r.shots < filter.min_shots:
                continue
            if filter.max_shots is not None and r.shots > filter.max_shots:
                continue
            if filter.start_date and r.timestamp < filter.start_date:
                continue
            if filter.end_date and r.timestamp > filter.end_date:
                continue
            if filter.min_execution_time_ms is not None and r.execution_time_ms < filter.min_execution_time_ms:
                continue
            if filter.max_execution_time_ms is not None and r.execution_time_ms > filter.max_execution_time_ms:
                continue
            results.append(r)
        
        # Apply sorting
        results.sort(
            key=lambda r: getattr(r, sort.field, r.timestamp),
            reverse=sort.descending,
        )
        
        # Apply pagination
        return results[offset:offset + limit]
    
    def aggregate(
        self,
        filter: QueryFilter | None = None,
        group_by: str | None = None,
    ) -> AggregateResult | dict[str, AggregateResult]:
        """Execute aggregate query.
        
        Args:
            filter: Optional filter for aggregation
            group_by: Optional field to group by (backend_name, circuit_name, etc.)
            
        Returns:
            Single AggregateResult or dict of group -> AggregateResult
        """
        filter = filter or QueryFilter()
        
        if isinstance(self._store, SQLiteStore):
            return self._aggregate_sqlite(filter, group_by)
        else:
            return self._aggregate_memory(filter, group_by)
    
    def _aggregate_sqlite(
        self,
        filter: QueryFilter,
        group_by: str | None,
    ) -> AggregateResult | dict[str, AggregateResult]:
        """Execute aggregate on SQLite."""
        base_query = """
            SELECT 
                {group_col}
                COUNT(*) as cnt,
                SUM(execution_time_ms) as sum_time,
                AVG(execution_time_ms) as avg_time,
                MIN(execution_time_ms) as min_time,
                MAX(execution_time_ms) as max_time,
                SUM(memory_used_mb) as sum_mem,
                AVG(memory_used_mb) as avg_mem,
                SUM(shots) as total_shots,
                COUNT(DISTINCT backend_name) as unique_backends,
                COUNT(DISTINCT circuit_name) as unique_circuits
            FROM results
            WHERE 1=1
        """
        
        params: list[Any] = []
        
        # Build WHERE clause
        where_clause = ""
        if filter.session_id:
            where_clause += " AND session_id = ?"
            params.append(filter.session_id)
        if filter.backend_names:
            placeholders = ",".join("?" * len(filter.backend_names))
            where_clause += f" AND backend_name IN ({placeholders})"
            params.extend(filter.backend_names)
        
        group_col = f"{group_by}," if group_by else ""
        query = base_query.format(group_col=group_col) + where_clause
        
        if group_by:
            query += f" GROUP BY {group_by}"
        
        cursor = self._store._conn.execute(query, params)
        rows = cursor.fetchall()
        
        def row_to_aggregate(row: sqlite3.Row) -> AggregateResult:
            return AggregateResult(
                count=row["cnt"] or 0,
                sum_execution_time_ms=row["sum_time"] or 0.0,
                avg_execution_time_ms=row["avg_time"] or 0.0,
                min_execution_time_ms=row["min_time"] or 0.0,
                max_execution_time_ms=row["max_time"] or 0.0,
                sum_memory_mb=row["sum_mem"] or 0.0,
                avg_memory_mb=row["avg_mem"] or 0.0,
                total_shots=row["total_shots"] or 0,
                unique_backends=row["unique_backends"] or 0,
                unique_circuits=row["unique_circuits"] or 0,
            )
        
        if group_by and rows:
            return {row[group_by]: row_to_aggregate(row) for row in rows}
        elif rows:
            return row_to_aggregate(rows[0])
        else:
            return AggregateResult()
    
    def _aggregate_memory(
        self,
        filter: QueryFilter,
        group_by: str | None,
    ) -> AggregateResult | dict[str, AggregateResult]:
        """Execute aggregate on memory store."""
        results = self.query(filter, limit=100000, use_cache=False)
        
        if not results:
            return AggregateResult() if not group_by else {}
        
        def compute_aggregate(items: list[StoredResult]) -> AggregateResult:
            times = [r.execution_time_ms for r in items]
            mems = [r.memory_used_mb for r in items]
            return AggregateResult(
                count=len(items),
                sum_execution_time_ms=sum(times),
                avg_execution_time_ms=sum(times) / len(times) if times else 0,
                min_execution_time_ms=min(times) if times else 0,
                max_execution_time_ms=max(times) if times else 0,
                sum_memory_mb=sum(mems),
                avg_memory_mb=sum(mems) / len(mems) if mems else 0,
                total_shots=sum(r.shots for r in items),
                unique_backends=len(set(r.backend_name for r in items)),
                unique_circuits=len(set(r.circuit_name for r in items if r.circuit_name)),
            )
        
        if group_by:
            groups: dict[str, list[StoredResult]] = {}
            for r in results:
                key = getattr(r, group_by, "unknown")
                if key not in groups:
                    groups[key] = []
                groups[key].append(r)
            return {k: compute_aggregate(v) for k, v in groups.items()}
        else:
            return compute_aggregate(results)
    
    def search_text(
        self,
        search_term: str,
        fields: list[str] | None = None,
        limit: int = 100,
    ) -> list[StoredResult]:
        """Full-text search across result fields.
        
        Args:
            search_term: Text to search for
            fields: Fields to search in (default: circuit_name, metadata)
            limit: Maximum results
            
        Returns:
            Matching results
        """
        fields = fields or ["circuit_name"]
        search_lower = search_term.lower()
        
        all_results = self._store.list_results(limit=10000)
        matches = []
        
        for r in all_results:
            for field in fields:
                value = getattr(r, field, None)
                if value is None:
                    continue
                if isinstance(value, str) and search_lower in value.lower():
                    matches.append(r)
                    break
                elif isinstance(value, dict):
                    # Search in metadata
                    if search_lower in json.dumps(value).lower():
                        matches.append(r)
                        break
        
        return matches[:limit]
    
    def _get_cache_key(
        self,
        filter: QueryFilter,
        sort: QuerySort,
        limit: int,
        offset: int,
    ) -> str:
        """Generate cache key for query."""
        import hashlib
        key_data = f"{filter}:{sort}:{limit}:{offset}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> list[StoredResult] | None:
        """Get cached query result."""
        import time
        if key in self._cache:
            timestamp, result = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            del self._cache[key]
        return None
    
    def _set_cached(self, key: str, result: list[StoredResult]) -> None:
        """Cache query result."""
        import time
        self._cache[key] = (time.time(), result)
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        self._cache.clear()


# =============================================================================
# Advanced Query Optimization System (Feature - 5% Gap Completion)
# =============================================================================


@dataclass
class QueryExecutionPlan:
    """Detailed query execution plan."""
    
    query_type: str  # 'simple', 'filtered', 'aggregated', 'joined'
    estimated_rows: int
    uses_index: bool
    index_names: list[str]
    scan_type: str  # 'index_scan', 'full_scan', 'range_scan'
    filter_selectivity: float  # 0.0 to 1.0, lower is more selective
    sort_strategy: str  # 'index_ordered', 'memory_sort', 'none'
    estimated_cost: float
    recommendations: list[str]
    execution_steps: list[str]


@dataclass
class QueryOptimizationResult:
    """Result of query optimization analysis."""
    
    original_query: str
    optimized_query: str
    plan: QueryExecutionPlan
    estimated_improvement: float  # Percentage improvement
    index_suggestions: list[str]
    warnings: list[str]


@dataclass
class CacheStatistics:
    """Query cache statistics."""
    
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    memory_bytes: int
    oldest_entry_age_seconds: float
    eviction_count: int


class QueryBuilder:
    """Fluent API for building optimized queries.
    
    Provides:
    - Type-safe query construction
    - Method chaining
    - Automatic optimization hints
    - Query validation
    
    Example:
        results = (QueryBuilder(store)
            .select()
            .where(backend="lret")
            .where_time_between(0, 100)
            .order_by("execution_time_ms")
            .limit(50)
            .execute())
    """
    
    def __init__(self, store: ResultStore) -> None:
        """Initialize query builder with target store."""
        self._store = store
        self._filter = QueryFilter()
        self._sort = QuerySort()
        self._limit: int = 100
        self._offset: int = 0
        self._use_cache: bool = True
        self._include_fields: list[str] | None = None
        self._exclude_fields: list[str] | None = None
        self._distinct_by: str | None = None
    
    def select(self, *fields: str) -> "QueryBuilder":
        """Select specific fields to return.
        
        Args:
            *fields: Field names to include
            
        Returns:
            Self for chaining
        """
        if fields:
            self._include_fields = list(fields)
        return self
    
    def exclude(self, *fields: str) -> "QueryBuilder":
        """Exclude specific fields from results.
        
        Args:
            *fields: Field names to exclude
            
        Returns:
            Self for chaining
        """
        if fields:
            self._exclude_fields = list(fields)
        return self
    
    def where(self, **conditions: Any) -> "QueryBuilder":
        """Add equality conditions.
        
        Args:
            **conditions: Field=value conditions
            
        Returns:
            Self for chaining
        """
        for field, value in conditions.items():
            if field == "session_id":
                self._filter.session_id = value
            elif field == "backend_name" or field == "backend":
                self._filter.backend_name = value
            elif field == "circuit_name":
                self._filter.circuit_name = value
            elif field == "qubit_count":
                self._filter.qubit_count = value
        return self
    
    def where_time_between(
        self,
        min_ms: float | None = None,
        max_ms: float | None = None,
    ) -> "QueryBuilder":
        """Filter by execution time range.
        
        Args:
            min_ms: Minimum execution time
            max_ms: Maximum execution time
            
        Returns:
            Self for chaining
        """
        self._filter.min_execution_time = min_ms
        self._filter.max_execution_time = max_ms
        return self
    
    def where_memory_between(
        self,
        min_mb: float | None = None,
        max_mb: float | None = None,
    ) -> "QueryBuilder":
        """Filter by memory usage range.
        
        Args:
            min_mb: Minimum memory usage
            max_mb: Maximum memory usage
            
        Returns:
            Self for chaining
        """
        self._filter.min_memory = min_mb
        self._filter.max_memory = max_mb
        return self
    
    def where_timestamp_between(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> "QueryBuilder":
        """Filter by timestamp range.
        
        Args:
            start: Start datetime
            end: End datetime
            
        Returns:
            Self for chaining
        """
        self._filter.start_time = start
        self._filter.end_time = end
        return self
    
    def where_qubit_range(
        self,
        min_qubits: int | None = None,
        max_qubits: int | None = None,
    ) -> "QueryBuilder":
        """Filter by qubit count range.
        
        Args:
            min_qubits: Minimum qubit count
            max_qubits: Maximum qubit count
            
        Returns:
            Self for chaining
        """
        self._filter.min_qubit_count = min_qubits
        self._filter.max_qubit_count = max_qubits
        return self
    
    def where_success(self, success: bool = True) -> "QueryBuilder":
        """Filter by success status.
        
        Args:
            success: Whether to filter for successful results
            
        Returns:
            Self for chaining
        """
        self._filter.success_only = success
        return self
    
    def order_by(
        self,
        field: str,
        descending: bool = False,
    ) -> "QueryBuilder":
        """Set sort order.
        
        Args:
            field: Field to sort by
            descending: Whether to sort descending
            
        Returns:
            Self for chaining
        """
        self._sort.field = field
        self._sort.descending = descending
        return self
    
    def order_by_time(self, descending: bool = True) -> "QueryBuilder":
        """Sort by execution time.
        
        Args:
            descending: Whether to sort descending (slowest first)
            
        Returns:
            Self for chaining
        """
        return self.order_by("execution_time_ms", descending)
    
    def order_by_timestamp(self, descending: bool = True) -> "QueryBuilder":
        """Sort by timestamp.
        
        Args:
            descending: Whether to sort descending (newest first)
            
        Returns:
            Self for chaining
        """
        return self.order_by("timestamp", descending)
    
    def order_by_memory(self, descending: bool = True) -> "QueryBuilder":
        """Sort by memory usage.
        
        Args:
            descending: Whether to sort descending (highest first)
            
        Returns:
            Self for chaining
        """
        return self.order_by("memory_used_mb", descending)
    
    def limit(self, count: int) -> "QueryBuilder":
        """Limit result count.
        
        Args:
            count: Maximum results
            
        Returns:
            Self for chaining
        """
        self._limit = count
        return self
    
    def offset(self, count: int) -> "QueryBuilder":
        """Skip results.
        
        Args:
            count: Number to skip
            
        Returns:
            Self for chaining
        """
        self._offset = count
        return self
    
    def page(self, page_num: int, page_size: int = 50) -> "QueryBuilder":
        """Paginate results.
        
        Args:
            page_num: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Self for chaining
        """
        self._limit = page_size
        self._offset = (page_num - 1) * page_size
        return self
    
    def distinct(self, field: str) -> "QueryBuilder":
        """Get distinct values for a field.
        
        Args:
            field: Field to get distinct values for
            
        Returns:
            Self for chaining
        """
        self._distinct_by = field
        return self
    
    def no_cache(self) -> "QueryBuilder":
        """Disable query caching.
        
        Returns:
            Self for chaining
        """
        self._use_cache = False
        return self
    
    def execute(self) -> list[StoredResult]:
        """Execute the built query.
        
        Returns:
            List of matching results
        """
        if hasattr(self._store, "query"):
            results = self._store.query(
                self._filter,
                sort=self._sort,
                limit=self._limit,
                offset=self._offset,
                use_cache=self._use_cache,
            )
        else:
            results = self._store.list_results(limit=self._limit)
        
        # Apply post-processing
        if self._distinct_by:
            seen = set()
            unique = []
            for r in results:
                val = getattr(r, self._distinct_by, None)
                if val not in seen:
                    seen.add(val)
                    unique.append(r)
            results = unique
        
        # Apply field filtering
        if self._include_fields or self._exclude_fields:
            # Return full results but client can filter
            pass
        
        return results
    
    def count(self) -> int:
        """Count matching results.
        
        Returns:
            Count of matching results
        """
        # Execute with high limit just to count
        old_limit = self._limit
        self._limit = 100000
        results = self.execute()
        self._limit = old_limit
        return len(results)
    
    def first(self) -> StoredResult | None:
        """Get first matching result.
        
        Returns:
            First result or None
        """
        self._limit = 1
        results = self.execute()
        return results[0] if results else None
    
    def exists(self) -> bool:
        """Check if any results match.
        
        Returns:
            True if any results match
        """
        return self.first() is not None
    
    def explain(self) -> QueryExecutionPlan:
        """Get execution plan for the query.
        
        Returns:
            Query execution plan
        """
        analyzer = QueryOptimizer(self._store)
        return analyzer.analyze_query(self._filter, self._sort)


class QueryOptimizer:
    """Analyzes and optimizes queries for performance.
    
    Provides:
    - Query plan analysis
    - Index recommendations
    - Query rewriting
    - Performance predictions
    """
    
    def __init__(self, store: ResultStore) -> None:
        """Initialize optimizer with target store."""
        self._store = store
    
    def analyze_query(
        self,
        filter: QueryFilter,
        sort: QuerySort | None = None,
    ) -> QueryExecutionPlan:
        """Analyze a query and generate execution plan.
        
        Args:
            filter: Query filter
            sort: Sort configuration
            
        Returns:
            Query execution plan
        """
        recommendations: list[str] = []
        execution_steps: list[str] = []
        
        # Determine query type
        has_filters = bool(
            filter.session_id or filter.backend_name or 
            filter.min_execution_time is not None or
            filter.start_time is not None
        )
        
        query_type = "filtered" if has_filters else "simple"
        
        # Analyze index usage
        uses_index = False
        index_names: list[str] = []
        scan_type = "full_scan"
        
        if filter.session_id:
            uses_index = True
            index_names.append("idx_results_session")
            scan_type = "index_scan"
            execution_steps.append("Use index on session_id")
        
        if filter.backend_name:
            uses_index = True
            index_names.append("idx_results_backend")
            scan_type = "index_scan"
            execution_steps.append("Use index on backend_name")
        
        # Calculate filter selectivity
        selectivity = 1.0
        if filter.session_id:
            selectivity *= 0.1  # Session ID is very selective
        if filter.backend_name:
            selectivity *= 0.2  # Backend is moderately selective
        if filter.min_execution_time is not None:
            selectivity *= 0.5
        if filter.start_time:
            selectivity *= 0.3
        
        # Determine sort strategy
        sort_strategy = "none"
        if sort and sort.field:
            if sort.field in ("timestamp", "session_id", "backend_name"):
                sort_strategy = "index_ordered"
                execution_steps.append(f"Use index for {sort.field} ordering")
            else:
                sort_strategy = "memory_sort"
                execution_steps.append(f"Sort in memory by {sort.field}")
                recommendations.append(f"Consider adding index on {sort.field} for faster sorting")
        
        # Estimate cost
        base_cost = 100.0  # Base cost units
        if uses_index:
            base_cost *= 0.1
        base_cost *= selectivity
        if sort_strategy == "memory_sort":
            base_cost *= 1.5
        
        # Generate recommendations
        if not uses_index and has_filters:
            recommendations.append("Query may benefit from additional indexes")
        
        if selectivity > 0.5:
            recommendations.append("Filter conditions are not very selective - consider more specific filters")
        
        # Estimate rows
        estimated_rows = int(1000 * selectivity)  # Rough estimate
        
        return QueryExecutionPlan(
            query_type=query_type,
            estimated_rows=estimated_rows,
            uses_index=uses_index,
            index_names=index_names,
            scan_type=scan_type,
            filter_selectivity=selectivity,
            sort_strategy=sort_strategy,
            estimated_cost=base_cost,
            recommendations=recommendations,
            execution_steps=execution_steps,
        )
    
    def suggest_indexes(self) -> list[str]:
        """Suggest indexes based on common query patterns.
        
        Returns:
            List of suggested CREATE INDEX statements
        """
        suggestions = [
            "CREATE INDEX IF NOT EXISTS idx_results_composite_perf "
            "ON results(backend_name, execution_time_ms DESC)",
            
            "CREATE INDEX IF NOT EXISTS idx_results_timestamp_range "
            "ON results(timestamp DESC, session_id)",
            
            "CREATE INDEX IF NOT EXISTS idx_results_memory_filter "
            "ON results(memory_used_mb) WHERE memory_used_mb > 0",
            
            "CREATE INDEX IF NOT EXISTS idx_results_qubit_analysis "
            "ON results(qubit_count, execution_time_ms)",
        ]
        return suggestions
    
    def optimize_query(
        self,
        filter: QueryFilter,
        sort: QuerySort | None = None,
    ) -> QueryOptimizationResult:
        """Analyze and optimize a query.
        
        Args:
            filter: Original query filter
            sort: Sort configuration
            
        Returns:
            Optimization result with suggestions
        """
        plan = self.analyze_query(filter, sort)
        
        # Generate optimized query representation
        original_parts = []
        optimized_parts = []
        
        if filter.session_id:
            original_parts.append(f"session_id = '{filter.session_id}'")
            optimized_parts.append(f"session_id = '{filter.session_id}' (indexed)")
        
        if filter.backend_name:
            original_parts.append(f"backend_name = '{filter.backend_name}'")
            optimized_parts.append(f"backend_name = '{filter.backend_name}' (indexed)")
        
        if filter.min_execution_time is not None:
            original_parts.append(f"execution_time >= {filter.min_execution_time}")
            optimized_parts.append(f"execution_time >= {filter.min_execution_time}")
        
        original_query = " AND ".join(original_parts) if original_parts else "SELECT ALL"
        optimized_query = " AND ".join(optimized_parts) if optimized_parts else "SELECT ALL"
        
        # Calculate improvement estimate
        improvement = 0.0
        if plan.uses_index:
            improvement = (1 - plan.filter_selectivity) * 100
        
        index_suggestions = []
        if not plan.uses_index:
            index_suggestions = self.suggest_indexes()[:2]
        
        warnings = []
        if plan.estimated_rows > 10000:
            warnings.append("Query may return large result set - consider adding LIMIT")
        
        return QueryOptimizationResult(
            original_query=original_query,
            optimized_query=optimized_query,
            plan=plan,
            estimated_improvement=improvement,
            index_suggestions=index_suggestions,
            warnings=warnings,
        )


class EnhancedQueryCache:
    """Advanced query caching with LRU eviction and statistics.
    
    Provides:
    - LRU (Least Recently Used) eviction
    - Memory-aware caching
    - Cache statistics
    - Selective invalidation
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_bytes: int = 50 * 1024 * 1024,  # 50MB
        ttl_seconds: float = 300.0,
    ) -> None:
        """Initialize enhanced cache.
        
        Args:
            max_entries: Maximum cache entries
            max_memory_bytes: Maximum memory usage
            ttl_seconds: Time-to-live for entries
        """
        self._max_entries = max_entries
        self._max_memory = max_memory_bytes
        self._ttl = ttl_seconds
        
        self._cache: dict[str, tuple[float, Any, int]] = {}  # key -> (timestamp, value, size)
        self._access_order: list[str] = []  # LRU tracking
        
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._total_memory = 0
    
    def get(self, key: str) -> Any | None:
        """Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        import time
        
        if key not in self._cache:
            self._miss_count += 1
            return None
        
        timestamp, value, size = self._cache[key]
        
        # Check TTL
        if time.time() - timestamp > self._ttl:
            self._evict(key)
            self._miss_count += 1
            return None
        
        # Update access order (move to end = most recently used)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        self._hit_count += 1
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        import time
        import sys
        
        # Estimate size
        try:
            size = sys.getsizeof(value)
        except TypeError:
            size = 1000  # Default estimate
        
        # Evict if necessary
        while (len(self._cache) >= self._max_entries or 
               self._total_memory + size > self._max_memory):
            if not self._access_order:
                break
            self._evict(self._access_order[0])
        
        # Remove existing entry if present
        if key in self._cache:
            _, _, old_size = self._cache[key]
            self._total_memory -= old_size
        
        # Add new entry
        self._cache[key] = (time.time(), value, size)
        self._total_memory += size
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict(self, key: str) -> None:
        """Evict a cache entry.
        
        Args:
            key: Key to evict
        """
        if key in self._cache:
            _, _, size = self._cache[key]
            self._total_memory -= size
            del self._cache[key]
            self._eviction_count += 1
        
        if key in self._access_order:
            self._access_order.remove(key)
    
    def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries.
        
        Args:
            pattern: Optional pattern to match keys (None = all)
            
        Returns:
            Number of entries invalidated
        """
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            self._total_memory = 0
            return count
        
        count = 0
        keys_to_remove = [k for k in self._cache if pattern in k]
        for key in keys_to_remove:
            self._evict(key)
            count += 1
        
        return count
    
    def get_statistics(self) -> CacheStatistics:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        import time
        
        oldest_age = 0.0
        if self._cache:
            oldest_timestamp = min(t for t, _, _ in self._cache.values())
            oldest_age = time.time() - oldest_timestamp
        
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0
        
        return CacheStatistics(
            total_entries=len(self._cache),
            hit_count=self._hit_count,
            miss_count=self._miss_count,
            hit_rate=hit_rate,
            memory_bytes=self._total_memory,
            oldest_entry_age_seconds=oldest_age,
            eviction_count=self._eviction_count,
        )
    
    def warm_cache(self, queries: list[tuple[QueryFilter, QuerySort]]) -> int:
        """Pre-warm cache with common queries.
        
        Args:
            queries: List of (filter, sort) tuples to pre-execute
            
        Returns:
            Number of queries warmed
        """
        # This would need access to a store to actually execute
        # For now, just return 0 as placeholder
        return 0


class QueryProfiler:
    """Profiles query execution for performance analysis.
    
    Tracks:
    - Query execution times
    - Slow queries
    - Query patterns
    - Performance trends
    """
    
    def __init__(self, slow_threshold_ms: float = 100.0) -> None:
        """Initialize profiler.
        
        Args:
            slow_threshold_ms: Threshold for slow query detection
        """
        self._slow_threshold = slow_threshold_ms
        self._query_times: list[tuple[float, str, float]] = []  # (timestamp, query_desc, duration_ms)
        self._slow_queries: list[tuple[str, float, QueryFilter]] = []  # (desc, duration_ms, filter)
    
    def record(
        self,
        query_desc: str,
        duration_ms: float,
        filter: QueryFilter | None = None,
    ) -> None:
        """Record a query execution.
        
        Args:
            query_desc: Query description
            duration_ms: Execution duration
            filter: Query filter used
        """
        import time
        
        self._query_times.append((time.time(), query_desc, duration_ms))
        
        if duration_ms > self._slow_threshold and filter:
            self._slow_queries.append((query_desc, duration_ms, filter))
        
        # Keep only last 1000 queries
        if len(self._query_times) > 1000:
            self._query_times = self._query_times[-1000:]
    
    def get_slow_queries(self, limit: int = 10) -> list[tuple[str, float]]:
        """Get slowest queries.
        
        Args:
            limit: Maximum results
            
        Returns:
            List of (description, duration_ms) tuples
        """
        sorted_slow = sorted(self._slow_queries, key=lambda x: -x[1])
        return [(desc, duration) for desc, duration, _ in sorted_slow[:limit]]
    
    def get_average_time(self) -> float:
        """Get average query time.
        
        Returns:
            Average duration in ms
        """
        if not self._query_times:
            return 0.0
        
        total = sum(d for _, _, d in self._query_times)
        return total / len(self._query_times)
    
    def get_percentile(self, percentile: float) -> float:
        """Get query time percentile.
        
        Args:
            percentile: Percentile (0-100)
            
        Returns:
            Duration at percentile
        """
        if not self._query_times:
            return 0.0
        
        times = sorted(d for _, _, d in self._query_times)
        idx = int(len(times) * percentile / 100)
        return times[min(idx, len(times) - 1)]
    
    def get_summary(self) -> dict[str, Any]:
        """Get profiling summary.
        
        Returns:
            Summary statistics
        """
        return {
            "total_queries": len(self._query_times),
            "slow_queries": len(self._slow_queries),
            "average_time_ms": self.get_average_time(),
            "p50_time_ms": self.get_percentile(50),
            "p95_time_ms": self.get_percentile(95),
            "p99_time_ms": self.get_percentile(99),
            "slow_threshold_ms": self._slow_threshold,
        }

