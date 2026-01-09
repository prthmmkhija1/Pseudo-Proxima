"""Session management for persistence and recovery.

Provides:
- Session: Encapsulates execution session state
- SessionManager: Create, save, load, resume sessions
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class SessionStatus(Enum):
    """Session lifecycle status."""

    CREATED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


@dataclass
class SessionMetadata:
    """Session metadata."""

    id: str
    created_at: float
    updated_at: float
    status: SessionStatus
    name: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionCheckpoint:
    """Checkpoint within a session for recovery."""

    id: str
    timestamp: float
    stage: str
    state: dict[str, Any]
    message: str | None = None


@dataclass
class Session:
    """Execution session with state and checkpoints."""

    metadata: SessionMetadata
    config: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[SessionCheckpoint] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:8]
        now = time.time()

        return cls(
            metadata=SessionMetadata(
                id=session_id,
                created_at=now,
                updated_at=now,
                status=SessionStatus.CREATED,
                name=name or f"session-{session_id}",
            ),
            config=config or {},
        )

    def update_status(self, status: SessionStatus) -> None:
        """Update session status."""
        self.metadata.status = status
        self.metadata.updated_at = time.time()

    def checkpoint(
        self,
        stage: str,
        state: dict[str, Any],
        message: str | None = None,
    ) -> SessionCheckpoint:
        """Create a checkpoint."""
        cp = SessionCheckpoint(
            id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            stage=stage,
            state=state.copy(),
            message=message,
        )
        self.checkpoints.append(cp)
        self.metadata.updated_at = time.time()
        return cp

    def latest_checkpoint(self) -> SessionCheckpoint | None:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None

    def add_log(self, level: str, message: str, **extra: Any) -> None:
        """Add a log entry."""
        self.logs.append(
            {
                "timestamp": time.time(),
                "level": level,
                "message": message,
                **extra,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dict."""
        return {
            "metadata": {
                "id": self.metadata.id,
                "created_at": self.metadata.created_at,
                "updated_at": self.metadata.updated_at,
                "status": self.metadata.status.name,
                "name": self.metadata.name,
                "description": self.metadata.description,
                "tags": self.metadata.tags,
            },
            "config": self.config,
            "state": self.state,
            "checkpoints": [
                {
                    "id": cp.id,
                    "timestamp": cp.timestamp,
                    "stage": cp.stage,
                    "state": cp.state,
                    "message": cp.message,
                }
                for cp in self.checkpoints
            ],
            "results": self.results,
            "logs": self.logs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Deserialize session from dict."""
        meta = data["metadata"]
        return cls(
            metadata=SessionMetadata(
                id=meta["id"],
                created_at=meta["created_at"],
                updated_at=meta["updated_at"],
                status=SessionStatus[meta["status"]],
                name=meta.get("name"),
                description=meta.get("description"),
                tags=meta.get("tags", []),
            ),
            config=data.get("config", {}),
            state=data.get("state", {}),
            checkpoints=[
                SessionCheckpoint(
                    id=cp["id"],
                    timestamp=cp["timestamp"],
                    stage=cp["stage"],
                    state=cp["state"],
                    message=cp.get("message"),
                )
                for cp in data.get("checkpoints", [])
            ],
            results=data.get("results", {}),
            logs=data.get("logs", []),
        )


class SessionManager:
    """Manages session lifecycle and persistence."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir = storage_dir
        self._current: Session | None = None
        self._sessions: dict[str, Session] = {}

    @property
    def current(self) -> Session | None:
        return self._current

    def create(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Session:
        """Create and set current session."""
        session = Session.create(name=name, config=config)
        self._sessions[session.metadata.id] = session
        self._current = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Get session by ID."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self.load(session_id)

    def set_current(self, session_id: str) -> bool:
        """Set current session by ID."""
        session = self.get(session_id)
        if session:
            self._current = session
            return True
        return False

    def save(self, session: Session | None = None) -> bool:
        """Save session to storage."""
        session = session or self._current
        if not session or not self._storage_dir:
            return False

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / f"{session.metadata.id}.json"
        path.write_text(json.dumps(session.to_dict(), indent=2))
        return True

    def load(self, session_id: str) -> Session | None:
        """Load session from storage."""
        if not self._storage_dir:
            return None

        path = self._storage_dir / f"{session_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            session = Session.from_dict(data)
            self._sessions[session_id] = session
            return session
        except Exception:
            return None

    def list_sessions(self) -> list[SessionMetadata]:
        """List all available sessions."""
        sessions = list(self._sessions.values())

        if self._storage_dir and self._storage_dir.exists():
            for path in self._storage_dir.glob("*.json"):
                session_id = path.stem
                if session_id not in self._sessions:
                    session = self.load(session_id)
                    if session:
                        sessions.append(session)

        return [s.metadata for s in sessions]

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]

        if self._current and self._current.metadata.id == session_id:
            self._current = None

        if self._storage_dir:
            path = self._storage_dir / f"{session_id}.json"
            if path.exists():
                path.unlink()
                return True

        return session_id in self._sessions

    def resume(self, session_id: str) -> Session | None:
        """Resume a session from its last checkpoint."""
        session = self.get(session_id)
        if not session:
            return None

        checkpoint = session.latest_checkpoint()
        if checkpoint:
            session.state = checkpoint.state.copy()

        session.update_status(SessionStatus.RUNNING)
        self._current = session
        return session
