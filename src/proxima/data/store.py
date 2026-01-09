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

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            # Delete associated results
            to_delete = [rid for rid, r in self._results.items() if r.session_id == session_id]
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
        cursor = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
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
