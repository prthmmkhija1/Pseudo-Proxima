"""Enhanced Session Management Implementation.

Provides session lifecycle management with critical missing features:
- Long-term session storage (SQLite/JSON)
- Session export/import
- Concurrent session management
- Session recovery after crashes

Features:
- Session creation and lifecycle management
- Checkpoint persistence for resume capability
- Execution history within session
- State snapshot and restoration
- Session metadata and tagging
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Generator, Protocol, TypeVar

from proxima.core.state import (
    ExecutionState,
    ExecutionStateMachine,
    PersistedState,
    StatePersistence,
)
from proxima.utils.logging import get_logger, set_execution_context


# ==================== SESSION STATUS ====================


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"
    CRASHED = "crashed"
    RECOVERING = "recovering"


class StorageBackend(str, Enum):
    """Storage backend types."""

    JSON = "json"
    SQLITE = "sqlite"


# ==================== CHECKPOINTS ====================


@dataclass
class Checkpoint:
    """Represents a point-in-time snapshot of session state.

    Used for pause/resume and rollback functionality.
    """

    id: str
    session_id: str
    timestamp: datetime
    state: str
    execution_index: int
    plan_snapshot: dict[str, Any]
    results_snapshot: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state,
            "execution_index": self.execution_index,
            "plan_snapshot": self.plan_snapshot,
            "results_snapshot": self.results_snapshot,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create checkpoint from dictionary."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state=data["state"],
            execution_index=data["execution_index"],
            plan_snapshot=data["plan_snapshot"],
            results_snapshot=data["results_snapshot"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionRecord:
    """Records a single execution within a session.

    Tracks task executions including timing, status, and results.
    """

    id: str
    session_id: str
    task_id: str
    timestamp: datetime
    duration_ms: float
    status: str  # 'success', 'failed', 'cancelled'
    backend: str
    shots: int = 0
    result_summary: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "backend": self.backend,
            "shots": self.shots,
            "result_summary": self.result_summary,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionRecord:
        """Create record from dictionary."""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            task_id=data["task_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_ms=data["duration_ms"],
            status=data["status"],
            backend=data["backend"],
            shots=data.get("shots", 0),
            result_summary=data.get("result_summary", {}),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


# ==================== STORAGE BACKENDS ====================


class SessionStorageBackend(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    def save_session(self, session_data: dict[str, Any]) -> None:
        """Save session data."""
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load session data by ID."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        pass

    @abstractmethod
    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        pass

    @abstractmethod
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        pass

    @abstractmethod
    def load_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """Load checkpoints for a session."""
        pass

    @abstractmethod
    def get_crashed_sessions(self) -> list[str]:
        """Get sessions that crashed mid-execution."""
        pass


class JSONStorageBackend(SessionStorageBackend):
    """JSON file-based storage backend."""

    def __init__(self, storage_dir: Path):
        """Initialize JSON storage backend.

        Args:
            storage_dir: Directory for session files
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = storage_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _session_path(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.storage_dir / f"{session_id}.json"

    def save_session(self, session_data: dict[str, Any]) -> None:
        """Save session to JSON file."""
        with self._lock:
            session_path = self._session_path(session_data["id"])
            temp_path = session_path.with_suffix(".tmp")

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, default=str)

            temp_path.replace(session_path)

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load session from JSON file."""
        session_path = self._session_path(session_id)
        if not session_path.exists():
            return None

        with open(session_path, encoding="utf-8") as f:
            return json.load(f)

    def delete_session(self, session_id: str) -> bool:
        """Delete session file."""
        session_path = self._session_path(session_id)
        if session_path.exists():
            session_path.unlink()
            # Also delete checkpoints
            for cp in self.checkpoints_dir.glob(f"{session_id}_*.json"):
                cp.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        sessions = []
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "status": data.get("status"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def session_exists(self, session_id: str) -> bool:
        """Check if session file exists."""
        return self._session_path(session_id).exists()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to JSON file."""
        cp_path = self.checkpoints_dir / f"{checkpoint.session_id}_{checkpoint.id}.json"
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    def load_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """Load all checkpoints for a session."""
        checkpoints = []
        for path in self.checkpoints_dir.glob(f"{session_id}_*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    checkpoints.append(Checkpoint.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(checkpoints, key=lambda c: c.timestamp)

    def get_crashed_sessions(self) -> list[str]:
        """Get sessions that were running when crashed."""
        crashed = []
        for session_info in self.list_sessions():
            if session_info.get("status") in (
                SessionStatus.ACTIVE.value,
                SessionStatus.PAUSED.value,
            ):
                crashed.append(session_info["id"])
        return crashed


class SQLiteStorageBackend(SessionStorageBackend):
    """SQLite database storage backend."""

    def __init__(self, db_path: Path):
        """Initialize SQLite storage backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                data TEXT,
                created_at TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp TEXT,
                state TEXT,
                execution_index INTEGER,
                data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
            CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id);
        """)
        conn.commit()

    def save_session(self, session_data: dict[str, Any]) -> None:
        """Save session to database."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (id, name, status, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_data["id"],
                session_data.get("name", ""),
                session_data.get("status", ""),
                json.dumps(session_data, default=str),
                session_data.get("created_at", datetime.utcnow().isoformat()),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()

    def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load session from database."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row:
            return json.loads(row["data"])
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete session from database."""
        conn = self._get_conn()
        conn.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
        result = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return result.rowcount > 0

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, name, status, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to database."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO checkpoints (id, session_id, timestamp, state, execution_index, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.id,
                checkpoint.session_id,
                checkpoint.timestamp.isoformat(),
                checkpoint.state,
                checkpoint.execution_index,
                json.dumps(checkpoint.to_dict(), default=str),
            ),
        )
        conn.commit()

    def load_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """Load all checkpoints for a session."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT data FROM checkpoints WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [Checkpoint.from_dict(json.loads(row["data"])) for row in rows]

    def get_crashed_sessions(self) -> list[str]:
        """Get sessions that were running when crashed."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM sessions WHERE status IN (?, ?)",
            (SessionStatus.ACTIVE.value, SessionStatus.PAUSED.value),
        ).fetchall()
        return [row["id"] for row in rows]



# ==================== SESSION EXPORT/IMPORT ====================


class SessionExporter:
    """Handles session export to portable formats."""

    def __init__(self, storage: SessionStorageBackend):
        """Initialize exporter.

        Args:
            storage: Storage backend to export from
        """
        self.storage = storage
        self.logger = get_logger("session.exporter")

    def export_to_json(self, session_id: str, output_path: Path) -> Path:
        """Export session to JSON file.

        Args:
            session_id: Session to export
            output_path: Output file path

        Returns:
            Path to exported file
        """
        session_data = self.storage.load_session(session_id)
        if not session_data:
            raise ValueError(f"Session '{session_id}' not found")

        # Include checkpoints
        checkpoints = self.storage.load_checkpoints(session_id)
        export_data = {
            "session": session_data,
            "checkpoints": [cp.to_dict() for cp in checkpoints],
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0",
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info("session.exported", session_id=session_id, path=str(output_path))
        return output_path

    def export_to_archive(self, session_id: str, output_path: Path) -> Path:
        """Export session to compressed archive.

        Args:
            session_id: Session to export
            output_path: Output archive path

        Returns:
            Path to exported archive
        """
        session_data = self.storage.load_session(session_id)
        if not session_data:
            raise ValueError(f"Session '{session_id}' not found")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add session data
            zf.writestr("session.json", json.dumps(session_data, indent=2, default=str))

            # Add checkpoints
            checkpoints = self.storage.load_checkpoints(session_id)
            for i, cp in enumerate(checkpoints):
                zf.writestr(f"checkpoints/cp_{i:04d}.json", json.dumps(cp.to_dict(), indent=2))

            # Add metadata
            metadata = {
                "session_id": session_id,
                "exported_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "checkpoint_count": len(checkpoints),
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        self.logger.info("session.archived", session_id=session_id, path=str(output_path))
        return output_path


class SessionImporter:
    """Handles session import from portable formats."""

    def __init__(self, storage: SessionStorageBackend):
        """Initialize importer.

        Args:
            storage: Storage backend to import to
        """
        self.storage = storage
        self.logger = get_logger("session.importer")

    def import_from_json(
        self,
        input_path: Path,
        new_session_id: str | None = None,
    ) -> str:
        """Import session from JSON file.

        Args:
            input_path: Path to JSON file
            new_session_id: Optional new ID for imported session

        Returns:
            Session ID of imported session
        """
        with open(input_path, encoding="utf-8") as f:
            export_data = json.load(f)

        session_data = export_data["session"]

        # Optionally assign new ID
        if new_session_id:
            old_id = session_data["id"]
            session_data["id"] = new_session_id
        else:
            old_id = session_data["id"]
            new_session_id = session_data["id"]

        # Check for conflicts
        if self.storage.session_exists(new_session_id):
            raise ValueError(f"Session '{new_session_id}' already exists")

        # Import session
        session_data["imported_at"] = datetime.utcnow().isoformat()
        session_data["status"] = SessionStatus.PAUSED.value  # Start paused
        self.storage.save_session(session_data)

        # Import checkpoints
        for cp_data in export_data.get("checkpoints", []):
            cp_data["session_id"] = new_session_id
            cp = Checkpoint.from_dict(cp_data)
            self.storage.save_checkpoint(cp)

        self.logger.info(
            "session.imported",
            session_id=new_session_id,
            path=str(input_path),
        )
        return new_session_id

    def import_from_archive(
        self,
        input_path: Path,
        new_session_id: str | None = None,
    ) -> str:
        """Import session from compressed archive.

        Args:
            input_path: Path to archive
            new_session_id: Optional new ID for imported session

        Returns:
            Session ID of imported session
        """
        with zipfile.ZipFile(input_path, "r") as zf:
            # Read session data
            session_data = json.loads(zf.read("session.json"))

            # Optionally assign new ID
            if new_session_id:
                old_id = session_data["id"]
                session_data["id"] = new_session_id
            else:
                old_id = session_data["id"]
                new_session_id = session_data["id"]

            # Check for conflicts
            if self.storage.session_exists(new_session_id):
                raise ValueError(f"Session '{new_session_id}' already exists")

            # Import session
            session_data["imported_at"] = datetime.utcnow().isoformat()
            session_data["status"] = SessionStatus.PAUSED.value
            self.storage.save_session(session_data)

            # Import checkpoints
            for name in zf.namelist():
                if name.startswith("checkpoints/") and name.endswith(".json"):
                    cp_data = json.loads(zf.read(name))
                    cp_data["session_id"] = new_session_id
                    cp = Checkpoint.from_dict(cp_data)
                    self.storage.save_checkpoint(cp)

        self.logger.info(
            "session.imported_archive",
            session_id=new_session_id,
            path=str(input_path),
        )
        return new_session_id


# ==================== CONCURRENT SESSION MANAGEMENT ====================


class SessionLock:
    """Distributed lock for session access."""

    def __init__(self, lock_dir: Path):
        """Initialize session lock.

        Args:
            lock_dir: Directory for lock files
        """
        self.lock_dir = lock_dir
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._held_locks: dict[str, Path] = {}
        self._lock = threading.Lock()

    def _lock_path(self, session_id: str) -> Path:
        """Get path to lock file."""
        return self.lock_dir / f"{session_id}.lock"

    def acquire(self, session_id: str, timeout: float = 30.0) -> bool:
        """Acquire lock for a session.

        Args:
            session_id: Session to lock
            timeout: Maximum wait time

        Returns:
            True if lock acquired
        """
        lock_path = self._lock_path(session_id)
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._lock:
                if session_id in self._held_locks:
                    return True  # Already held by us

                if not lock_path.exists():
                    try:
                        # Create lock file with our PID
                        lock_path.write_text(str(os.getpid()))
                        self._held_locks[session_id] = lock_path
                        return True
                    except Exception:
                        pass
                else:
                    # Check if holding process is still alive
                    try:
                        pid = int(lock_path.read_text())
                        if not self._process_alive(pid):
                            lock_path.unlink()
                            continue
                    except (ValueError, FileNotFoundError):
                        continue

            time.sleep(0.1)

        return False

    def release(self, session_id: str) -> None:
        """Release lock for a session.

        Args:
            session_id: Session to unlock
        """
        with self._lock:
            if session_id in self._held_locks:
                lock_path = self._held_locks.pop(session_id)
                if lock_path.exists():
                    lock_path.unlink()

    def is_locked(self, session_id: str) -> bool:
        """Check if session is locked.

        Args:
            session_id: Session to check

        Returns:
            True if locked
        """
        return self._lock_path(session_id).exists()

    def _process_alive(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @contextmanager
    def locked(self, session_id: str, timeout: float = 30.0) -> Generator[bool, None, None]:
        """Context manager for session locking.

        Args:
            session_id: Session to lock
            timeout: Maximum wait time

        Yields:
            True if lock acquired
        """
        acquired = self.acquire(session_id, timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release(session_id)


class ConcurrentSessionManager:
    """Manages concurrent access to sessions.

    Features:
    - Session locking
    - Concurrent session limits
    - Session access queue
    """

    def __init__(
        self,
        storage: SessionStorageBackend,
        lock_dir: Path,
        max_concurrent: int = 10,
    ):
        """Initialize concurrent session manager.

        Args:
            storage: Storage backend
            lock_dir: Directory for lock files
            max_concurrent: Maximum concurrent sessions
        """
        self.storage = storage
        self.lock = SessionLock(lock_dir)
        self.max_concurrent = max_concurrent
        self.logger = get_logger("session.concurrent")
        self._active_sessions: set[str] = set()
        self._lock = threading.Lock()

    def can_activate(self) -> bool:
        """Check if a new session can be activated."""
        with self._lock:
            return len(self._active_sessions) < self.max_concurrent

    def activate_session(self, session_id: str, timeout: float = 30.0) -> bool:
        """Activate a session for use.

        Args:
            session_id: Session to activate
            timeout: Lock timeout

        Returns:
            True if activated
        """
        if not self.can_activate():
            return False

        if not self.lock.acquire(session_id, timeout):
            return False

        with self._lock:
            self._active_sessions.add(session_id)

        self.logger.info("session.activated", session_id=session_id)
        return True

    def deactivate_session(self, session_id: str) -> None:
        """Deactivate a session.

        Args:
            session_id: Session to deactivate
        """
        self.lock.release(session_id)
        with self._lock:
            self._active_sessions.discard(session_id)
        self.logger.info("session.deactivated", session_id=session_id)

    def get_active_sessions(self) -> list[str]:
        """Get list of active sessions."""
        with self._lock:
            return list(self._active_sessions)

    @contextmanager
    def session_context(
        self,
        session_id: str,
        timeout: float = 30.0,
    ) -> Generator[bool, None, None]:
        """Context manager for session access.

        Args:
            session_id: Session to access
            timeout: Lock timeout

        Yields:
            True if session activated
        """
        activated = self.activate_session(session_id, timeout)
        try:
            yield activated
        finally:
            if activated:
                self.deactivate_session(session_id)



# ==================== CRASH RECOVERY ====================


class SessionRecoveryManager:
    """Handles session recovery after crashes.

    Features:
    - Detect crashed sessions
    - Recover session state
    - Resume from last checkpoint
    - Cleanup orphaned resources
    """

    def __init__(
        self,
        storage: SessionStorageBackend,
        state_persistence: StatePersistence | None = None,
    ):
        """Initialize recovery manager.

        Args:
            storage: Session storage backend
            state_persistence: Optional state persistence for deeper recovery
        """
        self.storage = storage
        self.state_persistence = state_persistence
        self.logger = get_logger("session.recovery")

    def get_crashed_sessions(self) -> list[dict[str, Any]]:
        """Get list of sessions that need recovery.

        Returns:
            List of crashed session info
        """
        crashed_ids = self.storage.get_crashed_sessions()
        crashed = []

        for session_id in crashed_ids:
            session_data = self.storage.load_session(session_id)
            if session_data:
                checkpoints = self.storage.load_checkpoints(session_id)
                crashed.append({
                    "session_id": session_id,
                    "name": session_data.get("name"),
                    "status": session_data.get("status"),
                    "last_checkpoint": checkpoints[-1].to_dict() if checkpoints else None,
                    "checkpoint_count": len(checkpoints),
                })

        return crashed

    def recover_session(
        self,
        session_id: str,
        from_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Recover a crashed session.

        Args:
            session_id: Session to recover
            from_checkpoint: Optional specific checkpoint to recover from

        Returns:
            Recovery result info
        """
        self.logger.info("session.recovery_started", session_id=session_id)

        session_data = self.storage.load_session(session_id)
        if not session_data:
            raise ValueError(f"Session '{session_id}' not found")

        checkpoints = self.storage.load_checkpoints(session_id)
        if not checkpoints and not from_checkpoint:
            # No checkpoints - mark as error
            session_data["status"] = SessionStatus.ERROR.value
            session_data["recovery_failed"] = True
            session_data["recovery_reason"] = "No checkpoints available"
            self.storage.save_session(session_data)
            return {
                "success": False,
                "reason": "No checkpoints available for recovery",
            }

        # Find checkpoint to recover from
        checkpoint: Checkpoint | None = None
        if from_checkpoint:
            checkpoint = next((c for c in checkpoints if c.id == from_checkpoint), None)
        else:
            checkpoint = checkpoints[-1] if checkpoints else None

        if not checkpoint:
            session_data["status"] = SessionStatus.ERROR.value
            self.storage.save_session(session_data)
            return {
                "success": False,
                "reason": f"Checkpoint '{from_checkpoint}' not found",
            }

        # Restore session from checkpoint
        session_data["state"] = checkpoint.state
        session_data["execution_index"] = checkpoint.execution_index
        session_data["plan"] = checkpoint.plan_snapshot
        session_data["results"] = checkpoint.results_snapshot
        session_data["status"] = SessionStatus.RECOVERING.value
        session_data["recovered_at"] = datetime.utcnow().isoformat()
        session_data["recovered_from"] = checkpoint.id

        self.storage.save_session(session_data)

        self.logger.info(
            "session.recovered",
            session_id=session_id,
            checkpoint_id=checkpoint.id,
        )

        return {
            "success": True,
            "checkpoint_id": checkpoint.id,
            "execution_index": checkpoint.execution_index,
            "state": checkpoint.state,
        }

    def mark_session_recovered(self, session_id: str) -> None:
        """Mark session as fully recovered and ready to resume.

        Args:
            session_id: Session to mark
        """
        session_data = self.storage.load_session(session_id)
        if session_data:
            session_data["status"] = SessionStatus.PAUSED.value
            session_data["recovery_complete"] = True
            self.storage.save_session(session_data)

    def abandon_session(self, session_id: str) -> None:
        """Abandon a crashed session (mark as aborted).

        Args:
            session_id: Session to abandon
        """
        session_data = self.storage.load_session(session_id)
        if session_data:
            session_data["status"] = SessionStatus.ABORTED.value
            session_data["abandoned_at"] = datetime.utcnow().isoformat()
            self.storage.save_session(session_data)

            self.logger.info("session.abandoned", session_id=session_id)

    def cleanup_orphaned_resources(self, session_id: str) -> int:
        """Cleanup resources from crashed session.

        Args:
            session_id: Session to cleanup

        Returns:
            Number of resources cleaned
        """
        # Check state persistence for orphaned resources
        if not self.state_persistence:
            return 0

        persisted = self.state_persistence.load(session_id)
        if not persisted:
            return 0

        cleaned = 0
        for resource_id in persisted.resources:
            # Attempt to cleanup (implementation depends on resource type)
            self.logger.debug("resource.cleanup_orphan", resource_id=resource_id)
            cleaned += 1

        # Delete persisted state
        self.state_persistence.delete(session_id)
        return cleaned


# ==================== MAIN SESSION CLASS ====================


@dataclass
class Session:
    """Represents a single execution session.

    Enhanced with:
    - Checkpoint management
    - State machine integration
    - Resource tracking
    """

    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus
    state_machine: ExecutionStateMachine
    plan: dict[str, Any] | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    execution_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "state": self.state_machine.state,
            "state_history": self.state_machine.history,
            "plan": self.plan,
            "results": self.results,
            "execution_index": self.execution_index,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Create session from dictionary."""
        state_machine = ExecutionStateMachine(
            execution_id=data["id"],
            enable_persistence=True,
        )
        # Restore state machine state
        if "state" in data:
            state_machine.state = data["state"]
        if "state_history" in data:
            state_machine.history = data["state_history"]

        return cls(
            id=data["id"],
            name=data.get("name", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=SessionStatus(data.get("status", "active")),
            state_machine=state_machine,
            plan=data.get("plan"),
            results=data.get("results", []),
            execution_index=data.get("execution_index", 0),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

    def create_checkpoint(self) -> Checkpoint:
        """Create a checkpoint of current session state."""
        return Checkpoint(
            id=str(uuid.uuid4())[:8],
            session_id=self.id,
            timestamp=datetime.utcnow(),
            state=self.state_machine.state,
            execution_index=self.execution_index,
            plan_snapshot=self.plan.copy() if self.plan else {},
            results_snapshot=[r.copy() for r in self.results],
            metadata={"session_status": self.status.value},
        )

    def restore_from_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Restore session state from checkpoint."""
        self.state_machine.state = checkpoint.state
        self.execution_index = checkpoint.execution_index
        self.plan = checkpoint.plan_snapshot.copy()
        self.results = [r.copy() for r in checkpoint.results_snapshot]
        self.updated_at = datetime.utcnow()

    def add_result(self, result: dict[str, Any]) -> None:
        """Add an execution result."""
        self.results.append(result)
        self.execution_index += 1
        self.updated_at = datetime.utcnow()

    def update_status(self, status: SessionStatus) -> None:
        """Update session status."""
        self.status = status
        self.updated_at = datetime.utcnow()

    @staticmethod
    def list_sessions(storage_dir: Path) -> list[dict[str, Any]]:
        """List all sessions in storage directory."""
        sessions = []
        for path in storage_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "status": data.get("status"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions



# ==================== SESSION MANAGER ====================


class SessionManager:
    """Enhanced session manager with all features.

    Features:
    - Multiple storage backends (JSON/SQLite)
    - Session export/import
    - Concurrent session management
    - Crash recovery
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        backend: StorageBackend = StorageBackend.JSON,
        max_concurrent: int = 10,
        enable_recovery: bool = True,
    ):
        """Initialize session manager.

        Args:
            storage_dir: Base storage directory
            backend: Storage backend type
            max_concurrent: Maximum concurrent sessions
            enable_recovery: Whether to enable crash recovery
        """
        self.storage_dir = storage_dir or Path.home() / ".proxima" / "sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage backend
        if backend == StorageBackend.SQLITE:
            db_path = self.storage_dir / "sessions.db"
            self.storage: SessionStorageBackend = SQLiteStorageBackend(db_path)
        else:
            self.storage = JSONStorageBackend(self.storage_dir)

        # Initialize components
        self.exporter = SessionExporter(self.storage)
        self.importer = SessionImporter(self.storage)

        lock_dir = self.storage_dir / "locks"
        self.concurrent_manager = ConcurrentSessionManager(
            self.storage, lock_dir, max_concurrent
        )

        state_persistence = StatePersistence(self.storage_dir / "state")
        self.recovery_manager = SessionRecoveryManager(
            self.storage, state_persistence
        )

        self.logger = get_logger("session.manager")
        self._current_session: Session | None = None
        self._session_cache: dict[str, Session] = {}
        self._lock = threading.Lock()

        # Check for crashed sessions on startup
        if enable_recovery:
            self._check_for_crashed_sessions()

    def _check_for_crashed_sessions(self) -> None:
        """Check for and log crashed sessions."""
        crashed = self.recovery_manager.get_crashed_sessions()
        if crashed:
            self.logger.warning(
                "session.crashed_sessions_found",
                count=len(crashed),
                sessions=[c["session_id"] for c in crashed],
            )

    @property
    def current(self) -> Session | None:
        """Get current active session."""
        return self._current_session

    def new_session(
        self,
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session.

        Args:
            name: Optional session name
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            New session
        """
        session_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow()

        session = Session(
            id=session_id,
            name=name or f"Session-{session_id}",
            created_at=now,
            updated_at=now,
            status=SessionStatus.ACTIVE,
            state_machine=ExecutionStateMachine(
                execution_id=session_id,
                enable_persistence=True,
                storage_dir=self.storage_dir / "state",
            ),
            metadata=metadata or {},
            tags=tags or [],
        )

        # Activate and save
        if not self.concurrent_manager.activate_session(session_id):
            raise RuntimeError("Cannot activate new session - too many concurrent sessions")

        self.storage.save_session(session.to_dict())
        self._session_cache[session_id] = session
        self._current_session = session

        # Set execution context for logging
        set_execution_context(session_id)

        self.logger.info(
            "session.created",
            session_id=session_id,
            name=session.name,
        )
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session or None
        """
        # Check cache first
        if session_id in self._session_cache:
            return self._session_cache[session_id]

        # Load from storage
        session_data = self.storage.load_session(session_id)
        if not session_data:
            return None

        session = Session.from_dict(session_data)
        self._session_cache[session_id] = session
        return session

    def activate_session(self, session_id: str) -> Session | None:
        """Activate an existing session.

        Args:
            session_id: Session to activate

        Returns:
            Activated session or None
        """
        session = self.get_session(session_id)
        if not session:
            return None

        if not self.concurrent_manager.activate_session(session_id):
            self.logger.warning(
                "session.activation_failed",
                session_id=session_id,
                reason="Too many concurrent sessions or lock timeout",
            )
            return None

        self._current_session = session
        set_execution_context(session_id)
        return session

    def deactivate_current(self) -> None:
        """Deactivate the current session."""
        if self._current_session:
            self.save_current()
            self.concurrent_manager.deactivate_session(self._current_session.id)
            self._current_session = None

    def save_current(self) -> None:
        """Save the current session."""
        if self._current_session:
            self._current_session.updated_at = datetime.utcnow()
            self.storage.save_session(self._current_session.to_dict())

    def create_checkpoint(self, session_id: str | None = None) -> Checkpoint:
        """Create a checkpoint for a session.

        Args:
            session_id: Optional session ID (defaults to current)

        Returns:
            Created checkpoint
        """
        session = self.get_session(session_id) if session_id else self._current_session
        if not session:
            raise ValueError("No session to checkpoint")

        checkpoint = session.create_checkpoint()
        self.storage.save_checkpoint(checkpoint)

        self.logger.info(
            "session.checkpoint_created",
            session_id=session.id,
            checkpoint_id=checkpoint.id,
        )
        return checkpoint

    def get_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a session.

        Args:
            session_id: Session ID

        Returns:
            List of checkpoints
        """
        return self.storage.load_checkpoints(session_id)

    def restore_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
    ) -> bool:
        """Restore a session from a checkpoint.

        Args:
            session_id: Session ID
            checkpoint_id: Checkpoint ID

        Returns:
            True if restored
        """
        session = self.get_session(session_id)
        if not session:
            return False

        checkpoints = self.get_checkpoints(session_id)
        checkpoint = next((c for c in checkpoints if c.id == checkpoint_id), None)
        if not checkpoint:
            return False

        session.restore_from_checkpoint(checkpoint)
        self.storage.save_session(session.to_dict())

        self.logger.info(
            "session.restored",
            session_id=session_id,
            checkpoint_id=checkpoint_id,
        )
        return True

    # ==================== EXPORT/IMPORT ====================

    def export_session(
        self,
        session_id: str,
        output_path: Path,
        format: str = "json",
    ) -> Path:
        """Export a session.

        Args:
            session_id: Session to export
            output_path: Output path
            format: Export format (json or archive)

        Returns:
            Path to exported file
        """
        if format == "archive":
            return self.exporter.export_to_archive(session_id, output_path)
        return self.exporter.export_to_json(session_id, output_path)

    def import_session(
        self,
        input_path: Path,
        new_id: str | None = None,
    ) -> str:
        """Import a session.

        Args:
            input_path: Path to import from
            new_id: Optional new ID

        Returns:
            Imported session ID
        """
        if input_path.suffix == ".zip":
            return self.importer.import_from_archive(input_path, new_id)
        return self.importer.import_from_json(input_path, new_id)

    # ==================== RECOVERY ====================

    def get_crashed_sessions(self) -> list[dict[str, Any]]:
        """Get list of crashed sessions."""
        return self.recovery_manager.get_crashed_sessions()

    def recover_session(
        self,
        session_id: str,
        from_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        """Recover a crashed session.

        Args:
            session_id: Session to recover
            from_checkpoint: Optional checkpoint ID

        Returns:
            Recovery result
        """
        result = self.recovery_manager.recover_session(session_id, from_checkpoint)

        # Invalidate cache
        if session_id in self._session_cache:
            del self._session_cache[session_id]

        return result

    def abandon_session(self, session_id: str) -> None:
        """Abandon a crashed session."""
        self.recovery_manager.abandon_session(session_id)

        if session_id in self._session_cache:
            del self._session_cache[session_id]

    # ==================== LISTING/CLEANUP ====================

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        return self.storage.list_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted
        """
        # Deactivate if current
        if self._current_session and self._current_session.id == session_id:
            self.deactivate_current()

        # Remove from cache
        if session_id in self._session_cache:
            del self._session_cache[session_id]

        result = self.storage.delete_session(session_id)
        if result:
            self.logger.info("session.deleted", session_id=session_id)
        return result

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Delete sessions older than specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        deleted = 0

        for session_info in self.list_sessions():
            try:
                updated = datetime.fromisoformat(session_info["updated_at"])
                if updated < cutoff:
                    if self.delete_session(session_info["id"]):
                        deleted += 1
            except (KeyError, ValueError):
                continue

        self.logger.info(
            "session.cleanup", deleted=deleted, max_age_days=max_age_days
        )
        return deleted


# ==================== GLOBAL INSTANCE ====================

_session_manager: SessionManager | None = None


def get_session_manager(
    storage_dir: Path | None = None,
    backend: StorageBackend = StorageBackend.JSON,
) -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(storage_dir=storage_dir, backend=backend)
    return _session_manager


def get_current_session() -> Session | None:
    """Get the current active session."""
    return get_session_manager().current


def new_session(name: str | None = None) -> Session:
    """Create a new session and set it as current."""
    return get_session_manager().new_session(name)


# ==================== MODULE EXPORTS ====================

__all__ = [
    # Enums
    "SessionStatus",
    "StorageBackend",
    # Checkpoint
    "Checkpoint",
    # Storage Backends
    "SessionStorageBackend",
    "JSONStorageBackend",
    "SQLiteStorageBackend",
    # Export/Import
    "SessionExporter",
    "SessionImporter",
    # Concurrent Management
    "SessionLock",
    "ConcurrentSessionManager",
    # Recovery
    "SessionRecoveryManager",
    # Main Classes
    "Session",
    "SessionManager",
    # Global Functions
    "get_session_manager",
    "get_current_session",
    "new_session",
]
