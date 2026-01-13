"""
Session management endpoints.

Provides REST API for managing execution sessions and history.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


# =============================================================================
# Models
# =============================================================================

class SessionStatus(str, Enum):
    """Session status."""
    
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    
    name: str = Field(default="", description="Optional session name")
    backend: str = Field(default="auto", description="Default backend for session")
    config: dict[str, Any] = Field(default_factory=dict, description="Session configuration")
    ttl_minutes: int = Field(default=60, ge=1, le=1440, description="Session time-to-live in minutes")


class SessionInfo(BaseModel):
    """Session information."""
    
    session_id: str
    name: str
    status: SessionStatus
    backend: str
    created_at: str
    last_activity: str
    expires_at: str
    job_count: int
    config: dict[str, Any] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    """List of sessions."""
    
    sessions: list[SessionInfo]
    total: int
    active_count: int


class SessionUpdateRequest(BaseModel):
    """Request to update session settings."""
    
    name: str | None = None
    backend: str | None = None
    config: dict[str, Any] | None = None
    ttl_minutes: int | None = None


class SessionHistoryItem(BaseModel):
    """Session history item."""
    
    timestamp: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)


class SessionHistoryResponse(BaseModel):
    """Session history response."""
    
    session_id: str
    history: list[SessionHistoryItem]
    total_items: int


# =============================================================================
# In-memory session storage (for demo purposes)
# =============================================================================

_sessions: dict[str, dict[str, Any]] = {}


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=SessionInfo, status_code=201)
async def create_session(
    create_request: SessionCreateRequest,
    request: Request,
) -> SessionInfo:
    """Create a new execution session.
    
    Sessions group related circuit executions and maintain state.
    
    Args:
        create_request: Session creation request.
        request: FastAPI request object.
    
    Returns:
        SessionInfo: Created session information.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    
    from datetime import timedelta
    expires_at = now + timedelta(minutes=create_request.ttl_minutes)
    
    session = {
        "session_id": session_id,
        "name": create_request.name or f"Session {session_id[:8]}",
        "status": SessionStatus.ACTIVE,
        "backend": create_request.backend,
        "config": create_request.config,
        "created_at": now.isoformat(),
        "last_activity": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "ttl_minutes": create_request.ttl_minutes,
        "jobs": [],
        "history": [
            {
                "timestamp": now.isoformat(),
                "action": "session_created",
                "details": {"backend": create_request.backend},
            }
        ],
    }
    
    _sessions[session_id] = session
    
    return SessionInfo(
        session_id=session_id,
        name=session["name"],
        status=session["status"],
        backend=session["backend"],
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        expires_at=session["expires_at"],
        job_count=len(session["jobs"]),
        config=session["config"],
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    status: SessionStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> SessionListResponse:
    """List all sessions.
    
    Args:
        status: Filter by session status.
        limit: Maximum number of sessions to return.
        offset: Number of sessions to skip.
    
    Returns:
        SessionListResponse: List of sessions.
    """
    sessions = list(_sessions.values())
    
    # Filter by status
    if status:
        sessions = [s for s in sessions if s["status"] == status]
    
    # Sort by created_at descending
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    
    # Paginate
    total = len(sessions)
    sessions = sessions[offset:offset + limit]
    
    session_infos = [
        SessionInfo(
            session_id=s["session_id"],
            name=s["name"],
            status=s["status"],
            backend=s["backend"],
            created_at=s["created_at"],
            last_activity=s["last_activity"],
            expires_at=s["expires_at"],
            job_count=len(s["jobs"]),
            config=s.get("config", {}),
        )
        for s in sessions
    ]
    
    active_count = sum(1 for s in _sessions.values() if s["status"] == SessionStatus.ACTIVE)
    
    return SessionListResponse(
        sessions=session_infos,
        total=total,
        active_count=active_count,
    )


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    """Get session information.
    
    Args:
        session_id: The session identifier.
    
    Returns:
        SessionInfo: Session information.
    
    Raises:
        HTTPException: If session not found.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    session = _sessions[session_id]
    
    # Check expiration
    expires_at = datetime.fromisoformat(session["expires_at"].replace('Z', '+00:00'))
    if datetime.now(timezone.utc) > expires_at and session["status"] == SessionStatus.ACTIVE:
        session["status"] = SessionStatus.EXPIRED
    
    return SessionInfo(
        session_id=session_id,
        name=session["name"],
        status=session["status"],
        backend=session["backend"],
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        expires_at=session["expires_at"],
        job_count=len(session["jobs"]),
        config=session.get("config", {}),
    )


@router.patch("/{session_id}", response_model=SessionInfo)
async def update_session(
    session_id: str,
    update_request: SessionUpdateRequest,
) -> SessionInfo:
    """Update session settings.
    
    Args:
        session_id: The session identifier.
        update_request: Update request with new values.
    
    Returns:
        SessionInfo: Updated session information.
    
    Raises:
        HTTPException: If session not found.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    session = _sessions[session_id]
    now = datetime.now(timezone.utc)
    
    # Apply updates
    if update_request.name is not None:
        session["name"] = update_request.name
    
    if update_request.backend is not None:
        session["backend"] = update_request.backend
    
    if update_request.config is not None:
        session["config"].update(update_request.config)
    
    if update_request.ttl_minutes is not None:
        from datetime import timedelta
        session["ttl_minutes"] = update_request.ttl_minutes
        session["expires_at"] = (now + timedelta(minutes=update_request.ttl_minutes)).isoformat()
    
    session["last_activity"] = now.isoformat()
    
    # Add to history
    session["history"].append({
        "timestamp": now.isoformat(),
        "action": "session_updated",
        "details": update_request.model_dump(exclude_none=True),
    })
    
    return await get_session(session_id)


@router.post("/{session_id}/pause")
async def pause_session(session_id: str) -> dict[str, str]:
    """Pause an active session.
    
    Args:
        session_id: The session identifier.
    
    Returns:
        dict: Confirmation message.
    
    Raises:
        HTTPException: If session not found or not active.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    session = _sessions[session_id]
    
    if session["status"] != SessionStatus.ACTIVE:
        raise HTTPException(
            status_code=400,
            detail=f"Session cannot be paused. Current status: {session['status']}"
        )
    
    now = datetime.now(timezone.utc)
    session["status"] = SessionStatus.PAUSED
    session["last_activity"] = now.isoformat()
    
    session["history"].append({
        "timestamp": now.isoformat(),
        "action": "session_paused",
        "details": {},
    })
    
    return {"message": f"Session '{session_id}' paused successfully"}


@router.post("/{session_id}/resume")
async def resume_session(session_id: str) -> dict[str, str]:
    """Resume a paused session.
    
    Args:
        session_id: The session identifier.
    
    Returns:
        dict: Confirmation message.
    
    Raises:
        HTTPException: If session not found or not paused.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    session = _sessions[session_id]
    
    if session["status"] != SessionStatus.PAUSED:
        raise HTTPException(
            status_code=400,
            detail=f"Session cannot be resumed. Current status: {session['status']}"
        )
    
    now = datetime.now(timezone.utc)
    session["status"] = SessionStatus.ACTIVE
    session["last_activity"] = now.isoformat()
    
    # Extend expiration
    from datetime import timedelta
    session["expires_at"] = (now + timedelta(minutes=session["ttl_minutes"])).isoformat()
    
    session["history"].append({
        "timestamp": now.isoformat(),
        "action": "session_resumed",
        "details": {},
    })
    
    return {"message": f"Session '{session_id}' resumed successfully"}


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session.
    
    Args:
        session_id: The session identifier.
    
    Returns:
        dict: Confirmation message.
    
    Raises:
        HTTPException: If session not found.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    del _sessions[session_id]
    
    return {"message": f"Session '{session_id}' deleted successfully"}


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
) -> SessionHistoryResponse:
    """Get session history.
    
    Args:
        session_id: The session identifier.
        limit: Maximum number of history items.
        offset: Number of items to skip.
    
    Returns:
        SessionHistoryResponse: Session history.
    
    Raises:
        HTTPException: If session not found.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    session = _sessions[session_id]
    history = session.get("history", [])
    
    # Sort by timestamp descending
    history_sorted = sorted(history, key=lambda h: h["timestamp"], reverse=True)
    
    # Paginate
    total = len(history_sorted)
    history_page = history_sorted[offset:offset + limit]
    
    return SessionHistoryResponse(
        session_id=session_id,
        history=[
            SessionHistoryItem(
                timestamp=h["timestamp"],
                action=h["action"],
                details=h.get("details", {}),
            )
            for h in history_page
        ],
        total_items=total,
    )


@router.get("/{session_id}/jobs")
async def get_session_jobs(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Get jobs associated with a session.
    
    Args:
        session_id: The session identifier.
        limit: Maximum number of jobs.
        offset: Number of jobs to skip.
    
    Returns:
        dict: List of job IDs and count.
    
    Raises:
        HTTPException: If session not found.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    
    session = _sessions[session_id]
    jobs = session.get("jobs", [])
    
    total = len(jobs)
    jobs_page = jobs[offset:offset + limit]
    
    return {
        "session_id": session_id,
        "jobs": jobs_page,
        "total": total,
    }
