"""
Session Persistence Module.

Provides complete session state save/restore functionality for Proxima,
including execution context, user preferences, and workflow state.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import uuid


class SessionStatus(str, Enum):
    """Session lifecycle states."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    action: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "parameters": self.parameters,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowStep:
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", ""),
            action=data.get("action", ""),
            parameters=data.get("parameters", {}),
            status=data.get("status", "pending"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class ExecutionContext:
    """Execution context state."""

    current_backend: str | None = None
    current_circuit: str | None = None
    circuit_definition: str | None = None
    qubit_count: int = 0
    shots: int = 1000
    parameters: dict[str, Any] = field(default_factory=dict)
    last_result_id: str | None = None
    execution_mode: str = "simulation"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_backend": self.current_backend,
            "current_circuit": self.current_circuit,
            "circuit_definition": self.circuit_definition,
            "qubit_count": self.qubit_count,
            "shots": self.shots,
            "parameters": self.parameters,
            "last_result_id": self.last_result_id,
            "execution_mode": self.execution_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContext:
        """Create from dictionary."""
        return cls(
            current_backend=data.get("current_backend"),
            current_circuit=data.get("current_circuit"),
            circuit_definition=data.get("circuit_definition"),
            qubit_count=data.get("qubit_count", 0),
            shots=data.get("shots", 1000),
            parameters=data.get("parameters", {}),
            last_result_id=data.get("last_result_id"),
            execution_mode=data.get("execution_mode", "simulation"),
        )


@dataclass
class UserPreferences:
    """User preferences for a session."""

    theme: str = "dark"
    language: str = "en"
    default_backend: str | None = None
    default_shots: int = 1000
    auto_save: bool = True
    auto_save_interval: int = 300  # seconds
    show_notifications: bool = True
    log_level: str = "INFO"
    custom_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theme": self.theme,
            "language": self.language,
            "default_backend": self.default_backend,
            "default_shots": self.default_shots,
            "auto_save": self.auto_save,
            "auto_save_interval": self.auto_save_interval,
            "show_notifications": self.show_notifications,
            "log_level": self.log_level,
            "custom_settings": self.custom_settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserPreferences:
        """Create from dictionary."""
        return cls(
            theme=data.get("theme", "dark"),
            language=data.get("language", "en"),
            default_backend=data.get("default_backend"),
            default_shots=data.get("default_shots", 1000),
            auto_save=data.get("auto_save", True),
            auto_save_interval=data.get("auto_save_interval", 300),
            show_notifications=data.get("show_notifications", True),
            log_level=data.get("log_level", "INFO"),
            custom_settings=data.get("custom_settings", {}),
        )


@dataclass
class SessionState:
    """Complete session state for persistence."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    # Context and state
    execution_context: ExecutionContext = field(default_factory=ExecutionContext)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # Workflow tracking
    workflow_steps: list[WorkflowStep] = field(default_factory=list)
    current_step_index: int = 0
    
    # History and undo support
    command_history: list[str] = field(default_factory=list)
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)
    
    # Tags and metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Result references
    result_ids: list[str] = field(default_factory=list)
    favorite_results: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "execution_context": self.execution_context.to_dict(),
            "preferences": self.preferences.to_dict(),
            "workflow_steps": [s.to_dict() for s in self.workflow_steps],
            "current_step_index": self.current_step_index,
            "command_history": self.command_history[-1000:],  # Keep last 1000
            "undo_stack": self.undo_stack[-100:],  # Keep last 100
            "redo_stack": self.redo_stack[-100:],
            "tags": self.tags,
            "metadata": self.metadata,
            "result_ids": self.result_ids,
            "favorite_results": self.favorite_results,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Create from dictionary."""
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=SessionStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else datetime.utcnow(),
            execution_context=ExecutionContext.from_dict(data.get("execution_context", {})),
            preferences=UserPreferences.from_dict(data.get("preferences", {})),
            workflow_steps=[WorkflowStep.from_dict(s) for s in data.get("workflow_steps", [])],
            current_step_index=data.get("current_step_index", 0),
            command_history=data.get("command_history", []),
            undo_stack=data.get("undo_stack", []),
            redo_stack=data.get("redo_stack", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            result_ids=data.get("result_ids", []),
            favorite_results=data.get("favorite_results", []),
        )

    def get_checksum(self) -> str:
        """Generate a checksum for state validation."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class SessionPersistence:
    """
    Manages session state persistence.
    
    Features:
    - Save/restore complete session state
    - Auto-save with configurable interval
    - Session history and archiving
    - Checkpoint management
    - State validation and recovery
    """

    DEFAULT_SESSIONS_DIR = Path.home() / ".proxima" / "sessions"

    def __init__(
        self,
        sessions_dir: Path | None = None,
        auto_save: bool = True,
        auto_save_interval: int = 300,
    ) -> None:
        """Initialize session persistence.

        Args:
            sessions_dir: Directory to store sessions
            auto_save: Enable automatic saving
            auto_save_interval: Seconds between auto-saves
        """
        self._sessions_dir = sessions_dir or self.DEFAULT_SESSIONS_DIR
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._auto_save = auto_save
        self._auto_save_interval = auto_save_interval
        self._current_session: SessionState | None = None
        self._last_save_time: datetime | None = None
        self._checkpoints_dir = self._sessions_dir / "checkpoints"
        self._checkpoints_dir.mkdir(exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self._sessions_dir / f"{session_id}.json"

    def _checkpoint_path(self, session_id: str, checkpoint_name: str) -> Path:
        """Get the file path for a checkpoint."""
        safe_name = checkpoint_name.replace("/", "_").replace("\\", "_")
        return self._checkpoints_dir / f"{session_id}_{safe_name}.json"

    def create_session(
        self,
        name: str = "",
        description: str = "",
        tags: list[str] | None = None,
    ) -> SessionState:
        """Create a new session."""
        session = SessionState(
            name=name or f"Session-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            description=description,
            tags=tags or [],
        )
        self._current_session = session
        self.save_session(session)
        return session

    def save_session(self, session: SessionState | None = None) -> bool:
        """Save session state to disk."""
        session = session or self._current_session
        if not session:
            return False

        session.updated_at = datetime.utcnow()
        path = self._session_path(session.session_id)
        
        try:
            data = session.to_dict()
            data["_checksum"] = session.get_checksum()
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            self._last_save_time = datetime.utcnow()
            return True
        except Exception:
            return False

    def load_session(self, session_id: str) -> SessionState | None:
        """Load a session from disk."""
        path = self._session_path(session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            session = SessionState.from_dict(data)
            session.last_accessed = datetime.utcnow()
            self._current_session = session
            self.save_session(session)  # Update last_accessed
            return session
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            # Delete associated checkpoints
            for checkpoint in self._checkpoints_dir.glob(f"{session_id}_*.json"):
                checkpoint.unlink()
            if self._current_session and self._current_session.session_id == session_id:
                self._current_session = None
            return True
        return False

    def archive_session(self, session_id: str) -> bool:
        """Archive a session."""
        session = self.load_session(session_id)
        if not session:
            return False

        session.status = SessionStatus.ARCHIVED
        return self.save_session(session)

    def list_sessions(
        self,
        status: SessionStatus | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[SessionState]:
        """List all sessions with optional filtering."""
        sessions: list[SessionState] = []
        
        for path in self._sessions_dir.glob("*.json"):
            if path.name.startswith("_"):  # Skip metadata files
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = SessionState.from_dict(data)
                
                # Apply filters
                if status and session.status != status:
                    continue
                if tags and not any(t in session.tags for t in tags):
                    continue
                
                sessions.append(session)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        # Sort by last accessed, most recent first
        sessions.sort(key=lambda s: s.last_accessed, reverse=True)
        return sessions[:limit]

    def get_current_session(self) -> SessionState | None:
        """Get the current active session."""
        return self._current_session

    def set_current_session(self, session: SessionState) -> None:
        """Set the current active session."""
        self._current_session = session

    # Checkpoint management
    def create_checkpoint(
        self,
        checkpoint_name: str,
        session: SessionState | None = None,
    ) -> bool:
        """Create a named checkpoint of the session state."""
        session = session or self._current_session
        if not session:
            return False

        path = self._checkpoint_path(session.session_id, checkpoint_name)
        data = session.to_dict()
        data["_checkpoint_name"] = checkpoint_name
        data["_checkpoint_time"] = datetime.utcnow().isoformat()
        
        try:
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            return True
        except Exception:
            return False

    def restore_checkpoint(
        self,
        checkpoint_name: str,
        session_id: str | None = None,
    ) -> SessionState | None:
        """Restore session state from a checkpoint."""
        session_id = session_id or (self._current_session.session_id if self._current_session else None)
        if not session_id:
            return None

        path = self._checkpoint_path(session_id, checkpoint_name)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            session = SessionState.from_dict(data)
            session.last_accessed = datetime.utcnow()
            self._current_session = session
            return session
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def list_checkpoints(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """List all checkpoints for a session."""
        session_id = session_id or (self._current_session.session_id if self._current_session else None)
        if not session_id:
            return []

        checkpoints = []
        for path in self._checkpoints_dir.glob(f"{session_id}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                checkpoints.append({
                    "name": data.get("_checkpoint_name", "unknown"),
                    "time": data.get("_checkpoint_time"),
                    "session_id": session_id,
                    "file": path.name,
                })
            except (json.JSONDecodeError, KeyError):
                continue

        checkpoints.sort(key=lambda c: c.get("time", ""), reverse=True)
        return checkpoints

    def delete_checkpoint(self, checkpoint_name: str, session_id: str | None = None) -> bool:
        """Delete a checkpoint."""
        session_id = session_id or (self._current_session.session_id if self._current_session else None)
        if not session_id:
            return False

        path = self._checkpoint_path(session_id, checkpoint_name)
        if path.exists():
            path.unlink()
            return True
        return False

    # Workflow support
    def add_workflow_step(
        self,
        name: str,
        action: str,
        parameters: dict[str, Any] | None = None,
    ) -> WorkflowStep | None:
        """Add a workflow step to the current session."""
        if not self._current_session:
            return None

        step = WorkflowStep(
            name=name,
            action=action,
            parameters=parameters or {},
            status="pending",
        )
        self._current_session.workflow_steps.append(step)
        self.save_session()
        return step

    def update_workflow_step(
        self,
        step_id: str,
        status: str | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> bool:
        """Update a workflow step."""
        if not self._current_session:
            return False

        for step in self._current_session.workflow_steps:
            if step.id == step_id:
                if status:
                    step.status = status
                    if status == "running":
                        step.started_at = datetime.utcnow()
                    elif status in ("completed", "failed"):
                        step.completed_at = datetime.utcnow()
                if result is not None:
                    step.result = result
                if error:
                    step.error = error
                self.save_session()
                return True

        return False

    # Command history
    def add_command(self, command: str) -> None:
        """Add a command to history."""
        if self._current_session:
            self._current_session.command_history.append(command)
            if len(self._current_session.command_history) > 1000:
                self._current_session.command_history = self._current_session.command_history[-1000:]
            self._maybe_auto_save()

    def get_command_history(self, limit: int = 100) -> list[str]:
        """Get recent command history."""
        if not self._current_session:
            return []
        return self._current_session.command_history[-limit:]

    # Undo/Redo support
    def push_undo_state(self, state: dict[str, Any]) -> None:
        """Push state onto undo stack."""
        if self._current_session:
            self._current_session.undo_stack.append(state)
            self._current_session.redo_stack.clear()  # Clear redo on new action
            if len(self._current_session.undo_stack) > 100:
                self._current_session.undo_stack = self._current_session.undo_stack[-100:]

    def pop_undo_state(self) -> dict[str, Any] | None:
        """Pop state from undo stack and push to redo."""
        if not self._current_session or not self._current_session.undo_stack:
            return None
        state = self._current_session.undo_stack.pop()
        self._current_session.redo_stack.append(state)
        return state

    def pop_redo_state(self) -> dict[str, Any] | None:
        """Pop state from redo stack and push to undo."""
        if not self._current_session or not self._current_session.redo_stack:
            return None
        state = self._current_session.redo_stack.pop()
        self._current_session.undo_stack.append(state)
        return state

    # Auto-save support
    def _maybe_auto_save(self) -> None:
        """Check if auto-save should trigger."""
        if not self._auto_save or not self._current_session:
            return

        now = datetime.utcnow()
        if self._last_save_time is None:
            self.save_session()
        elif (now - self._last_save_time).seconds >= self._auto_save_interval:
            self.save_session()

    # Export/Import
    def export_session(self, session_id: str | None = None) -> dict[str, Any] | None:
        """Export session for backup or sharing."""
        session_id = session_id or (self._current_session.session_id if self._current_session else None)
        if not session_id:
            return None

        session = self.load_session(session_id)
        if not session:
            return None

        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "session": session.to_dict(),
            "checkpoints": self.list_checkpoints(session_id),
        }
        return export_data

    def import_session(self, data: dict[str, Any]) -> SessionState | None:
        """Import a session from exported data."""
        try:
            session_data = data.get("session", {})
            session = SessionState.from_dict(session_data)
            
            # Generate new ID to avoid conflicts
            session.session_id = str(uuid.uuid4())
            session.name = f"Imported: {session.name}"
            session.created_at = datetime.utcnow()
            session.last_accessed = datetime.utcnow()
            
            self.save_session(session)
            return session
        except (KeyError, ValueError):
            return None

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics about sessions."""
        sessions = self.list_sessions(limit=10000)
        
        by_status: dict[str, int] = {}
        for status in SessionStatus:
            by_status[status.value] = sum(1 for s in sessions if s.status == status)

        total_results = sum(len(s.result_ids) for s in sessions)
        total_commands = sum(len(s.command_history) for s in sessions)

        return {
            "total_sessions": len(sessions),
            "by_status": by_status,
            "total_results": total_results,
            "total_commands": total_commands,
            "total_checkpoints": len(list(self._checkpoints_dir.glob("*.json"))),
            "storage_size_mb": sum(p.stat().st_size for p in self._sessions_dir.glob("**/*.json")) / 1024 / 1024,
        }


# Global session persistence singleton
_session_persistence: SessionPersistence | None = None


def get_session_persistence() -> SessionPersistence:
    """Get the global session persistence manager."""
    global _session_persistence
    if _session_persistence is None:
        _session_persistence = SessionPersistence()
    return _session_persistence


def reset_session_persistence() -> None:
    """Reset the global session persistence manager."""
    global _session_persistence
    _session_persistence = None
