"""Session management implementation.

Provides session lifecycle management for tracking execution contexts,
persisting state across executions, and supporting resume/undo functionality.

Features:
- Session creation and lifecycle management
- Checkpoint persistence for resume capability
- Execution history within session
- State snapshot and restoration
- Session metadata and tagging
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from proxima.core.state import ExecutionState, ExecutionStateMachine
from proxima.utils.logging import get_logger, set_execution_context


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"


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
    """Record of a single execution within a session."""

    id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"
    backend: str | None = None
    objective: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "status": self.status,
            "backend": self.backend,
            "objective": self.objective,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionRecord:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            status=data.get("status", "unknown"),
            backend=data.get("backend"),
            objective=data.get("objective"),
            result=data.get("result", {}),
            duration_ms=data.get("duration_ms", 0.0),
            error=data.get("error"),
        )


class Session:
    """Manages a single execution session.

    A session groups related executions together and provides:
    - Unified logging context
    - State persistence and checkpointing
    - Resume capability after pause/abort
    - Execution history tracking
    """

    def __init__(
        self,
        session_id: str | None = None,
        name: str | None = None,
        storage_dir: Path | None = None,
    ) -> None:
        """Initialize a new session.

        Args:
            session_id: Optional session ID. Generated if not provided.
            name: Optional human-readable session name.
            storage_dir: Directory for session persistence.
        """
        self.id = session_id or str(uuid.uuid4())
        self.name = name or f"session-{self.id[:8]}"
        self.storage_dir = storage_dir or Path.home() / ".proxima" / "sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.status = SessionStatus.ACTIVE
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        self._state_machine: ExecutionStateMachine | None = None
        self._current_plan: dict[str, Any] = {}
        self._executions: list[ExecutionRecord] = []
        self._checkpoints: list[Checkpoint] = []
        self._current_execution: ExecutionRecord | None = None
        self._metadata: dict[str, Any] = {}
        self._tags: list[str] = []

        self.logger = get_logger("session").bind(session_id=self.id)
        self.logger.info("session.created", name=self.name)

    @property
    def session_file(self) -> Path:
        """Path to session persistence file."""
        return self.storage_dir / f"{self.id}.json"

    @property
    def execution_count(self) -> int:
        """Number of executions in this session."""
        return len(self._executions)

    @property
    def checkpoints(self) -> list[Checkpoint]:
        """List of checkpoints."""
        return list(self._checkpoints)

    @property
    def executions(self) -> list[ExecutionRecord]:
        """List of execution records."""
        return list(self._executions)

    @property
    def current_state(self) -> str | None:
        """Current state machine state."""
        return self._state_machine.state if self._state_machine else None

    @property
    def metadata(self) -> dict[str, Any]:
        """Session metadata."""
        return dict(self._metadata)

    @property
    def tags(self) -> list[str]:
        """Session tags."""
        return list(self._tags)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the session."""
        if tag not in self._tags:
            self._tags.append(tag)
            self._touch()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the session."""
        if tag in self._tags:
            self._tags.remove(tag)
            self._touch()

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a metadata value."""
        self._metadata[key] = value
        self._touch()

    def _touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

    def initialize_state_machine(
        self, execution_id: str | None = None
    ) -> ExecutionStateMachine:
        """Create or reset the state machine for a new execution.

        Args:
            execution_id: Optional execution ID for tracing.

        Returns:
            Initialized ExecutionStateMachine.
        """
        exec_id = execution_id or str(uuid.uuid4())[:8]
        self._state_machine = ExecutionStateMachine(execution_id=exec_id)

        # Set logging context
        set_execution_context(execution_id=exec_id, session_id=self.id)

        self.logger.info("session.state_machine_initialized", execution_id=exec_id)
        return self._state_machine

    def start_execution(
        self,
        objective: str | None = None,
        backend: str | None = None,
        plan: dict[str, Any] | None = None,
    ) -> ExecutionRecord:
        """Start a new execution within this session.

        Args:
            objective: Execution objective/description.
            backend: Backend to use.
            plan: Execution plan.

        Returns:
            New ExecutionRecord.
        """
        if self.status != SessionStatus.ACTIVE:
            raise RuntimeError(
                f"Cannot start execution: session status is {self.status}"
            )

        exec_id = str(uuid.uuid4())[:8]
        self._current_plan = plan or {}

        record = ExecutionRecord(
            id=exec_id,
            started_at=datetime.utcnow(),
            objective=objective,
            backend=backend,
        )
        self._current_execution = record
        self._executions.append(record)

        # Initialize state machine if not exists
        if not self._state_machine:
            self.initialize_state_machine(exec_id)

        self.logger.info(
            "session.execution_started",
            execution_id=exec_id,
            objective=objective,
            backend=backend,
        )
        self._touch()
        return record

    def complete_execution(self, result: dict[str, Any] | None = None) -> None:
        """Mark current execution as completed.

        Args:
            result: Execution result data.
        """
        if self._current_execution:
            self._current_execution.completed_at = datetime.utcnow()
            self._current_execution.status = "completed"
            self._current_execution.result = result or {}

            if self._current_execution.started_at:
                delta = (
                    self._current_execution.completed_at
                    - self._current_execution.started_at
                )
                self._current_execution.duration_ms = delta.total_seconds() * 1000

            self.logger.info(
                "session.execution_completed",
                execution_id=self._current_execution.id,
                duration_ms=self._current_execution.duration_ms,
            )

        self._current_execution = None
        self._touch()

    def fail_execution(self, error: str) -> None:
        """Mark current execution as failed.

        Args:
            error: Error message.
        """
        if self._current_execution:
            self._current_execution.completed_at = datetime.utcnow()
            self._current_execution.status = "error"
            self._current_execution.error = error

            if self._current_execution.started_at:
                delta = (
                    self._current_execution.completed_at
                    - self._current_execution.started_at
                )
                self._current_execution.duration_ms = delta.total_seconds() * 1000

            self.logger.error(
                "session.execution_failed",
                execution_id=self._current_execution.id,
                error=error,
            )

        self._current_execution = None
        self._touch()

    def create_checkpoint(self, name: str | None = None) -> Checkpoint:
        """Create a checkpoint of current session state.

        Args:
            name: Optional checkpoint name/label.

        Returns:
            Created Checkpoint.
        """
        checkpoint = Checkpoint(
            id=str(uuid.uuid4())[:8],
            session_id=self.id,
            timestamp=datetime.utcnow(),
            state=self._state_machine.state if self._state_machine else "IDLE",
            execution_index=len(self._executions),
            plan_snapshot=dict(self._current_plan),
            results_snapshot=[e.to_dict() for e in self._executions],
            metadata={"name": name} if name else {},
        )

        self._checkpoints.append(checkpoint)
        self.logger.info(
            "session.checkpoint_created",
            checkpoint_id=checkpoint.id,
            name=name,
        )
        self._touch()
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore session to a previous checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore.

        Returns:
            True if restored successfully.
        """
        checkpoint = next(
            (cp for cp in self._checkpoints if cp.id == checkpoint_id), None
        )
        if not checkpoint:
            self.logger.warning(
                "session.checkpoint_not_found", checkpoint_id=checkpoint_id
            )
            return False

        # Restore executions
        self._executions = [
            ExecutionRecord.from_dict(data) for data in checkpoint.results_snapshot
        ]
        self._current_plan = dict(checkpoint.plan_snapshot)

        # Reset state machine to checkpoint state
        # Note: reset is a dynamically added trigger method from transitions library
        if self._state_machine:
            self._state_machine.reset()  # type: ignore[attr-defined]
            self.logger.info(
                "session.checkpoint_restored",
                checkpoint_id=checkpoint_id,
                restored_state=checkpoint.state,
            )

        self._touch()
        return True

    def pause(self) -> Checkpoint | None:
        """Pause the session and create a checkpoint.

        Returns:
            Created checkpoint, or None if already paused.
        """
        if self.status == SessionStatus.PAUSED:
            return None

        # Pause state machine if running
        # Note: pause is a dynamically added trigger method from transitions library
        if (
            self._state_machine
            and self._state_machine.state == ExecutionState.RUNNING.value
        ):
            self._state_machine.pause()  # type: ignore[attr-defined]

        self.status = SessionStatus.PAUSED
        checkpoint = self.create_checkpoint("pause_checkpoint")
        self.save()

        self.logger.info("session.paused")
        return checkpoint

    def resume(self) -> bool:
        """Resume a paused session.

        Returns:
            True if resumed successfully.
        """
        if self.status != SessionStatus.PAUSED:
            return False

        # Resume state machine if paused
        # Note: resume is a dynamically added trigger method from transitions library
        if (
            self._state_machine
            and self._state_machine.state == ExecutionState.PAUSED.value
        ):
            self._state_machine.resume()  # type: ignore[attr-defined]

        self.status = SessionStatus.ACTIVE
        self.logger.info("session.resumed")
        self._touch()
        return True

    def abort(self, reason: str | None = None) -> None:
        """Abort the session.

        Args:
            reason: Optional abort reason.
        """
        # Note: abort is a dynamically added trigger method from transitions library
        if self._state_machine and self._state_machine.state in (
            ExecutionState.RUNNING.value,
            ExecutionState.PAUSED.value,
        ):
            self._state_machine.abort()  # type: ignore[attr-defined]

        if self._current_execution:
            self.fail_execution(reason or "Session aborted")

        self.status = SessionStatus.ABORTED
        self.set_metadata("abort_reason", reason)
        self.save()

        self.logger.info("session.aborted", reason=reason)

    def complete(self) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.save()
        self.logger.info("session.completed", execution_count=self.execution_count)

    def save(self) -> Path:
        """Persist session to disk.

        Returns:
            Path to saved session file.
        """
        data = {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "executions": [e.to_dict() for e in self._executions],
            "checkpoints": [c.to_dict() for c in self._checkpoints],
            "current_plan": self._current_plan,
            "metadata": self._metadata,
            "tags": self._tags,
            "state_machine": (
                self._state_machine.snapshot() if self._state_machine else None
            ),
        }

        self.session_file.write_text(json.dumps(data, indent=2, default=str))
        self.logger.debug("session.saved", path=str(self.session_file))
        return self.session_file

    @classmethod
    def load(cls, session_id: str, storage_dir: Path | None = None) -> Session:
        """Load a session from disk.

        Args:
            session_id: Session ID to load.
            storage_dir: Storage directory.

        Returns:
            Loaded Session instance.

        Raises:
            FileNotFoundError: If session file doesn't exist.
        """
        storage = storage_dir or Path.home() / ".proxima" / "sessions"
        path = storage / f"{session_id}.json"

        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        data = json.loads(path.read_text())

        session = cls(
            session_id=data["id"],
            name=data.get("name"),
            storage_dir=storage,
        )

        session.status = SessionStatus(data.get("status", "active"))
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session._executions = [
            ExecutionRecord.from_dict(e) for e in data.get("executions", [])
        ]
        session._checkpoints = [
            Checkpoint.from_dict(c) for c in data.get("checkpoints", [])
        ]
        session._current_plan = data.get("current_plan", {})
        session._metadata = data.get("metadata", {})
        session._tags = data.get("tags", [])

        session.logger.info("session.loaded")
        return session

    @classmethod
    def list_sessions(cls, storage_dir: Path | None = None) -> list[dict[str, Any]]:
        """List all saved sessions.

        Args:
            storage_dir: Storage directory.

        Returns:
            List of session summaries.
        """
        storage = storage_dir or Path.home() / ".proxima" / "sessions"
        if not storage.exists():
            return []

        sessions = []
        for path in storage.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sessions.append(
                    {
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "status": data.get("status"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "execution_count": len(data.get("executions", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "execution_count": self.execution_count,
            "checkpoint_count": len(self._checkpoints),
            "current_state": self.current_state,
            "metadata": self._metadata,
            "tags": self._tags,
        }


class SessionManager:
    """Manages multiple sessions and provides global session access."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize session manager.

        Args:
            storage_dir: Base storage directory for sessions.
        """
        self.storage_dir = storage_dir or Path.home() / ".proxima" / "sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._current_session: Session | None = None
        self._session_cache: dict[str, Session] = {}
        self.logger = get_logger("session_manager")

    @property
    def current(self) -> Session | None:
        """Get current active session."""
        return self._current_session

    def new_session(self, name: str | None = None) -> Session:
        """Create a new session and set it as current.

        Args:
            name: Optional session name.

        Returns:
            New Session instance.
        """
        session = Session(name=name, storage_dir=self.storage_dir)
        self._current_session = session
        self._session_cache[session.id] = session

        self.logger.info(
            "session_manager.new_session", session_id=session.id, name=name
        )
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID, loading from disk if needed.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session instance or None if not found.
        """
        # Check cache first
        if session_id in self._session_cache:
            return self._session_cache[session_id]

        # Try loading from disk
        try:
            session = Session.load(session_id, self.storage_dir)
            self._session_cache[session_id] = session
            return session
        except FileNotFoundError:
            return None

    def set_current(self, session_id: str) -> Session | None:
        """Set a session as current by ID.

        Args:
            session_id: Session ID to make current.

        Returns:
            Session if found, None otherwise.
        """
        session = self.get_session(session_id)
        if session:
            self._current_session = session
            self.logger.info("session_manager.set_current", session_id=session_id)
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all available sessions.

        Returns:
            List of session summaries.
        """
        return Session.list_sessions(self.storage_dir)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID to delete.

        Returns:
            True if deleted.
        """
        session_file = self.storage_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

            # Remove from cache
            if session_id in self._session_cache:
                del self._session_cache[session_id]

            # Clear current if it's the deleted session
            if self._current_session and self._current_session.id == session_id:
                self._current_session = None

            self.logger.info("session_manager.deleted", session_id=session_id)
            return True
        return False

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Delete sessions older than specified age.

        Args:
            max_age_days: Maximum age in days.

        Returns:
            Number of sessions deleted.
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
            "session_manager.cleanup", deleted=deleted, max_age_days=max_age_days
        )
        return deleted


# Global session manager instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_current_session() -> Session | None:
    """Get the current active session."""
    return get_session_manager().current


def new_session(name: str | None = None) -> Session:
    """Create a new session and set it as current."""
    return get_session_manager().new_session(name)
