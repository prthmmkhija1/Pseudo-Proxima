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
