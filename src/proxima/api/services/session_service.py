"""
Session management service.

Handles session lifecycle, persistence, and cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing execution sessions."""
    
    def __init__(
        self,
        default_ttl_minutes: int = 60,
        cleanup_interval_seconds: int = 300,
        max_sessions: int = 1000,
    ) -> None:
        """Initialize the session service.
        
        Args:
            default_ttl_minutes: Default session TTL.
            cleanup_interval_seconds: Interval for cleanup task.
            max_sessions: Maximum concurrent sessions.
        """
        self._sessions: dict[str, dict[str, Any]] = {}
        self._default_ttl = default_ttl_minutes
        self._cleanup_interval = cleanup_interval_seconds
        self._max_sessions = max_sessions
        self._cleanup_task: asyncio.Task | None = None
        self._event_handlers: dict[str, list[Callable]] = {
            "session_created": [],
            "session_expired": [],
            "session_deleted": [],
        }
    
    @property
    def active_count(self) -> int:
        """Get number of active sessions."""
        return sum(
            1 for s in self._sessions.values()
            if s.get("status") == "active"
        )
    
    @property
    def total_count(self) -> int:
        """Get total number of sessions."""
        return len(self._sessions)
    
    def create_session(
        self,
        session_id: str | None = None,
        name: str = "",
        backend: str = "auto",
        config: dict[str, Any] | None = None,
        ttl_minutes: int | None = None,
        **kwargs,  # Accept additional kwargs for flexibility
    ) -> dict[str, Any]:
        """Create a new session.
        
        Args:
            session_id: Unique session identifier (auto-generated if not provided).
            name: Optional session name.
            backend: Default backend for the session.
            config: Session configuration.
            ttl_minutes: Session TTL in minutes.
            **kwargs: Additional configuration options (ignored).
        
        Returns:
            Created session data.
        
        Raises:
            ValueError: If max sessions reached.
        """
        if len(self._sessions) >= self._max_sessions:
            # Try cleanup first
            self._cleanup_expired()
            if len(self._sessions) >= self._max_sessions:
                raise ValueError("Maximum session limit reached")
        
        # Auto-generate session_id if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        ttl = ttl_minutes or self._default_ttl
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=ttl)
        
        session = {
            "session_id": session_id,
            "name": name or f"Session {session_id[:8]}",
            "status": "active",
            "backend": backend,
            "config": config or {},
            "created_at": now.isoformat(),
            "last_activity": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "ttl_minutes": ttl,
            "jobs": [],
            "history": [],
        }
        
        self._sessions[session_id] = session
        self._notify_handlers("session_created", session)
        
        logger.info(f"Session created: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a session by ID.
        
        Args:
            session_id: The session identifier.
        
        Returns:
            Session data or None if not found.
        """
        session = self._sessions.get(session_id)
        
        if session and self._is_expired(session):
            session["status"] = "expired"
        
        return session
    
    def update_session(
        self,
        session_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Update session data.
        
        Args:
            session_id: The session identifier.
            updates: Fields to update.
        
        Returns:
            Updated session or None if not found.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        now = datetime.now(timezone.utc)
        
        for key, value in updates.items():
            if key in ["session_id", "created_at"]:
                continue  # Don't allow updating these
            if key == "config" and isinstance(value, dict):
                session["config"].update(value)
            else:
                session[key] = value
        
        session["last_activity"] = now.isoformat()
        
        logger.debug(f"Session updated: {session_id}")
        return session
    
    def touch_session(self, session_id: str) -> bool:
        """Update session last activity time.
        
        Args:
            session_id: The session identifier.
        
        Returns:
            True if session exists and was updated.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        now = datetime.now(timezone.utc)
        session["last_activity"] = now.isoformat()
        
        # Extend expiration
        session["expires_at"] = (
            now + timedelta(minutes=session["ttl_minutes"])
        ).isoformat()
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: The session identifier.
        
        Returns:
            True if session was deleted.
        """
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            self._notify_handlers("session_deleted", session)
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
    
    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions with optional filtering.
        
        Args:
            status: Filter by status.
            limit: Maximum number of sessions.
            offset: Number to skip.
        
        Returns:
            List of session data.
        """
        sessions = list(self._sessions.values())
        
        # Update expired status
        for session in sessions:
            if self._is_expired(session):
                session["status"] = "expired"
        
        if status:
            sessions = [s for s in sessions if s.get("status") == status]
        
        # Sort by created_at descending
        sessions.sort(key=lambda s: s["created_at"], reverse=True)
        
        return sessions[offset:offset + limit]
    
    def add_job_to_session(self, session_id: str, job_id: str) -> bool:
        """Add a job to a session.
        
        Args:
            session_id: The session identifier.
            job_id: The job identifier.
        
        Returns:
            True if job was added.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        if job_id not in session["jobs"]:
            session["jobs"].append(job_id)
            self.touch_session(session_id)
        
        return True
    
    def add_history_entry(
        self,
        session_id: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Add a history entry to a session.
        
        Args:
            session_id: The session identifier.
            action: The action performed.
            details: Additional details.
        
        Returns:
            True if entry was added.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details or {},
        }
        
        session["history"].append(entry)
        return True
    
    def on_event(self, event: str, handler: Callable) -> None:
        """Register an event handler.
        
        Args:
            event: Event name.
            handler: Handler function.
        """
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)
    
    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")
    
    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Session cleanup task stopped")
    
    async def cleanup(self) -> None:
        """Cleanup sessions and stop background tasks."""
        await self.stop_cleanup_task()
        
        # Expire all active sessions
        for session in self._sessions.values():
            if session["status"] == "active":
                session["status"] = "expired"
        
        logger.info("Session service cleanup complete")
    
    def _is_expired(self, session: dict[str, Any]) -> bool:
        """Check if a session is expired."""
        if session.get("status") in ["completed", "expired"]:
            return True
        
        try:
            expires_at = datetime.fromisoformat(
                session["expires_at"].replace('Z', '+00:00')
            )
            return datetime.now(timezone.utc) > expires_at
        except (KeyError, ValueError):
            return False
    
    def _cleanup_expired(self) -> int:
        """Remove expired sessions.
        
        Returns:
            Number of sessions cleaned up.
        """
        expired = [
            sid for sid, session in self._sessions.items()
            if self._is_expired(session)
        ]
        
        for session_id in expired:
            session = self._sessions.pop(session_id)
            self._notify_handlers("session_expired", session)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)
    
    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _notify_handlers(self, event: str, data: Any) -> None:
        """Notify registered event handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    def _get_current_time(self) -> datetime:
        """Get current time (can be mocked in tests).
        
        Returns:
            Current datetime in UTC.
        """
        return datetime.now(timezone.utc)

    def cleanup_expired_sessions(self) -> int:
        """Public method to clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up.
        """
        return self._cleanup_expired()

    def pause_session(self, session_id: str) -> dict[str, Any] | None:
        """Pause a session.
        
        Args:
            session_id: The session identifier.
        
        Returns:
            Updated session or None if not found.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if session["status"] == "paused":
            return session  # Already paused
        
        session["status"] = "paused"
        session["last_activity"] = self._get_current_time().isoformat()
        self.add_history_entry(session_id, "paused")
        
        return session

    def resume_session(self, session_id: str) -> dict[str, Any] | None:
        """Resume a paused session.
        
        Args:
            session_id: The session identifier.
        
        Returns:
            Updated session or None if not found.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if session["status"] == "active":
            return session  # Already active
        
        session["status"] = "active"
        session["last_activity"] = self._get_current_time().isoformat()
        self.add_history_entry(session_id, "resumed")
        
        return session

    def submit_job(
        self,
        session_id: str,
        circuit: Any,
        backend: str | None = None,
        shots: int = 1024,
    ) -> dict[str, Any] | None:
        """Submit a job to a session.
        
        Args:
            session_id: The session identifier.
            circuit: The circuit to execute.
            backend: Optional backend override.
            shots: Number of shots.
        
        Returns:
            Job data or None if session not found/inactive.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if session["status"] != "active":
            return None
        
        job_id = str(uuid.uuid4())
        now = self._get_current_time()
        
        job = {
            "job_id": job_id,
            "session_id": session_id,
            "status": "pending",
            "backend": backend or session["backend"],
            "shots": shots,
            "submitted_at": now.isoformat(),
            "circuit": str(circuit),
        }
        
        self.add_job_to_session(session_id, job_id)
        self.touch_session(session_id)
        
        return job

    def get_session_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get session history.
        
        Args:
            session_id: The session identifier.
        
        Returns:
            List of history entries.
        """
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.get("history", [])
