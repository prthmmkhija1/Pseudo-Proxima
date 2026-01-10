"""Consent Audit Logging - Comprehensive tracking of all consent operations.

This module provides:
- AuditEvent: Individual audit log entries
- AuditLog: In-memory audit log with filtering
- AuditLogger: Persistent audit logging with rotation
- AuditQueryBuilder: Fluent interface for querying audit logs
- AuditReport: Generate compliance reports
- AuditedConsentManager: Wrapper that adds audit logging to ConsentManager

Audit events are captured for:
- Consent requests (granted/denied)
- Consent changes (grants, revocations)
- Force overrides
- Consent queries
- Configuration changes
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .consent import (
    ConsentCategory,
    ConsentLevel,
    ConsentManager,
    ConsentRecord,
)

# ========== Audit Event Types ==========


class AuditEventType(Enum):
    """Types of auditable consent events."""

    # Consent decision events
    CONSENT_REQUESTED = auto()
    CONSENT_GRANTED = auto()
    CONSENT_DENIED = auto()
    CONSENT_REMEMBERED = auto()

    # Change events
    CONSENT_REVOKED = auto()
    CONSENT_EXPIRED = auto()
    CONSENT_UPDATED = auto()
    ALL_CONSENTS_REVOKED = auto()
    CATEGORY_CONSENTS_REVOKED = auto()

    # Force override events
    FORCE_OVERRIDE_ENABLED = auto()
    FORCE_OVERRIDE_DISABLED = auto()
    FORCE_OVERRIDE_USED = auto()

    # Query events (optional, for full audit trail)
    CONSENT_CHECKED = auto()
    CONSENT_QUERIED = auto()

    # Configuration events
    CONFIG_CHANGED = auto()
    STORAGE_LOADED = auto()
    STORAGE_SAVED = auto()

    # Lifecycle events
    SESSION_STARTED = auto()
    SESSION_ENDED = auto()
    AUDIT_EXPORTED = auto()


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """A single audit log entry."""

    event_type: AuditEventType
    timestamp: float
    event_id: str
    topic: str | None = None
    category: ConsentCategory | None = None
    granted: bool | None = None
    level: ConsentLevel | None = None
    source: str = "unknown"
    user_id: str | None = None
    session_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    severity: AuditSeverity = AuditSeverity.INFO
    checksum: str | None = None

    def __post_init__(self) -> None:
        """Compute checksum after initialization."""
        if self.checksum is None:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute integrity checksum for the event."""
        data = f"{self.event_type.name}:{self.timestamp}:{self.topic}:{self.granted}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify event hasn't been tampered with."""
        return self.checksum == self._compute_checksum()

    @property
    def datetime(self) -> datetime:
        """Get event timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "topic": self.topic,
            "category": self.category.value if self.category else None,
            "granted": self.granted,
            "level": self.level.name if self.level else None,
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "details": self.details,
            "severity": self.severity.value,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Deserialize event from dictionary."""
        category = None
        if data.get("category"):
            try:
                category = ConsentCategory(data["category"])
            except ValueError:
                pass

        level = None
        if data.get("level"):
            try:
                level = ConsentLevel[data["level"]]
            except KeyError:
                pass

        return cls(
            event_type=AuditEventType[data["event_type"]],
            timestamp=data["timestamp"],
            event_id=data["event_id"],
            topic=data.get("topic"),
            category=category,
            granted=data.get("granted"),
            level=level,
            source=data.get("source", "unknown"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            details=data.get("details", {}),
            severity=AuditSeverity(data.get("severity", "info")),
            checksum=data.get("checksum"),
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        dt = self.datetime.strftime("%Y-%m-%d %H:%M:%S")
        granted_str = ""
        if self.granted is not None:
            granted_str = f" [{'GRANTED' if self.granted else 'DENIED'}]"
        topic_str = f" topic='{self.topic}'" if self.topic else ""
        return f"[{dt}] {self.event_type.name}{granted_str}{topic_str}"


# ========== Audit Log Storage ==========


class AuditStorage(ABC):
    """Abstract base for audit log storage backends."""

    @abstractmethod
    def append(self, event: AuditEvent) -> None:
        """Append an event to storage."""
        pass

    @abstractmethod
    def read_all(self) -> list[AuditEvent]:
        """Read all events from storage."""
        pass

    @abstractmethod
    def read_range(
        self, start_time: float | None = None, end_time: float | None = None
    ) -> list[AuditEvent]:
        """Read events within a time range."""
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all events. Returns count cleared."""
        pass


class MemoryAuditStorage(AuditStorage):
    """In-memory audit storage (non-persistent)."""

    def __init__(self, max_events: int = 10000) -> None:
        self._events: list[AuditEvent] = []
        self._max_events = max_events
        self._lock = threading.Lock()

    def append(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)
            # Trim if over limit
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]

    def read_all(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def read_range(
        self, start_time: float | None = None, end_time: float | None = None
    ) -> list[AuditEvent]:
        with self._lock:
            events = self._events
            if start_time is not None:
                events = [e for e in events if e.timestamp >= start_time]
            if end_time is not None:
                events = [e for e in events if e.timestamp <= end_time]
            return list(events)

    def clear(self) -> int:
        with self._lock:
            count = len(self._events)
            self._events = []
            return count

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._events)


class FileAuditStorage(AuditStorage):
    """File-based audit storage with rotation support."""

    def __init__(
        self,
        path: Path,
        max_size_mb: float = 10.0,
        max_files: int = 5,
        compress_rotated: bool = True,
    ) -> None:
        self._path = Path(path)
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._max_files = max_files
        self._compress_rotated = compress_rotated
        self._lock = threading.Lock()

        # Ensure directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AuditEvent) -> None:
        with self._lock:
            self._rotate_if_needed()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

    def _rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        if not self._path.exists():
            return

        if self._path.stat().st_size < self._max_size_bytes:
            return

        # Rotate existing files
        for i in range(self._max_files - 1, 0, -1):
            old_path = self._get_rotated_path(i)
            new_path = self._get_rotated_path(i + 1)
            if old_path.exists():
                if i + 1 >= self._max_files:
                    old_path.unlink()
                else:
                    old_path.rename(new_path)

        # Rotate current file
        rotated_path = self._get_rotated_path(1)
        self._path.rename(rotated_path)

        if self._compress_rotated:
            self._compress_file(rotated_path)

    def _get_rotated_path(self, index: int) -> Path:
        """Get path for rotated file."""
        suffix = f".{index}"
        if self._compress_rotated and index > 0:
            suffix += ".gz"
        return self._path.with_suffix(self._path.suffix + suffix)

    def _compress_file(self, path: Path) -> None:
        """Compress a log file."""
        if not path.exists() or path.suffix == ".gz":
            return

        compressed_path = path.with_suffix(path.suffix + ".gz")
        with open(path, "rb") as f_in:
            with gzip.open(compressed_path, "wb") as f_out:
                f_out.writelines(f_in)
        path.unlink()

    def read_all(self) -> list[AuditEvent]:
        with self._lock:
            events: list[AuditEvent] = []

            # Read current file
            if self._path.exists():
                events.extend(self._read_file(self._path))

            # Read rotated files (oldest to newest)
            for i in range(self._max_files, 0, -1):
                rotated_path = self._get_rotated_path(i)
                if rotated_path.exists():
                    events.extend(self._read_file(rotated_path))

            return sorted(events, key=lambda e: e.timestamp)

    def _read_file(self, path: Path) -> list[AuditEvent]:
        """Read events from a single file."""
        events: list[AuditEvent] = []

        if path.suffix == ".gz":
            opener = gzip.open
            mode = "rt"
        else:
            opener = open  # type: ignore
            mode = "r"

        try:
            with opener(path, mode, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            events.append(AuditEvent.from_dict(data))
                        except (json.JSONDecodeError, KeyError):
                            continue
        except Exception:
            pass

        return events

    def read_range(
        self, start_time: float | None = None, end_time: float | None = None
    ) -> list[AuditEvent]:
        events = self.read_all()
        if start_time is not None:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time is not None:
            events = [e for e in events if e.timestamp <= end_time]
        return events

    def clear(self) -> int:
        with self._lock:
            count = 0

            if self._path.exists():
                count += len(self._read_file(self._path))
                self._path.unlink()

            for i in range(1, self._max_files + 1):
                rotated_path = self._get_rotated_path(i)
                if rotated_path.exists():
                    rotated_path.unlink()

            return count


# ========== Audit Log ==========


class AuditLog:
    """Main audit log interface with filtering and querying."""

    def __init__(
        self,
        storage: AuditStorage | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._storage = storage or MemoryAuditStorage()
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._user_id = user_id
        self._enabled = enabled
        self._listeners: list[Callable[[AuditEvent], None]] = []
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def session_id(self) -> str:
        return self._session_id

    def log(
        self,
        event_type: AuditEventType,
        topic: str | None = None,
        category: ConsentCategory | None = None,
        granted: bool | None = None,
        level: ConsentLevel | None = None,
        source: str = "system",
        details: dict[str, Any] | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
    ) -> AuditEvent | None:
        """Log an audit event."""
        if not self._enabled:
            return None

        event = AuditEvent(
            event_type=event_type,
            timestamp=time.time(),
            event_id=str(uuid.uuid4()),
            topic=topic,
            category=category,
            granted=granted,
            level=level,
            source=source,
            user_id=self._user_id,
            session_id=self._session_id,
            details=details or {},
            severity=severity,
        )

        self._storage.append(event)
        self._notify_listeners(event)
        return event

    def add_listener(self, callback: Callable[[AuditEvent], None]) -> None:
        """Add a listener for real-time audit events."""
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[AuditEvent], None]) -> None:
        """Remove an audit event listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _notify_listeners(self, event: AuditEvent) -> None:
        """Notify all listeners of a new event."""
        with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors affect logging

    # ========== Query Methods ==========

    def get_all(self) -> list[AuditEvent]:
        """Get all audit events."""
        return self._storage.read_all()

    def get_by_type(self, event_type: AuditEventType) -> list[AuditEvent]:
        """Get events by type."""
        return [e for e in self.get_all() if e.event_type == event_type]

    def get_by_topic(self, topic: str) -> list[AuditEvent]:
        """Get events for a specific topic."""
        return [e for e in self.get_all() if e.topic == topic]

    def get_by_category(self, category: ConsentCategory) -> list[AuditEvent]:
        """Get events for a specific category."""
        return [e for e in self.get_all() if e.category == category]

    def get_by_severity(self, severity: AuditSeverity) -> list[AuditEvent]:
        """Get events by severity level."""
        return [e for e in self.get_all() if e.severity == severity]

    def get_range(
        self, start_time: float | None = None, end_time: float | None = None
    ) -> list[AuditEvent]:
        """Get events within a time range."""
        return self._storage.read_range(start_time, end_time)

    def get_recent(self, count: int = 100) -> list[AuditEvent]:
        """Get the most recent events."""
        all_events = self.get_all()
        return all_events[-count:] if len(all_events) > count else all_events

    def get_grants(self) -> list[AuditEvent]:
        """Get all consent grant events."""
        return [e for e in self.get_all() if e.event_type == AuditEventType.CONSENT_GRANTED]

    def get_denials(self) -> list[AuditEvent]:
        """Get all consent denial events."""
        return [e for e in self.get_all() if e.event_type == AuditEventType.CONSENT_DENIED]

    def get_revocations(self) -> list[AuditEvent]:
        """Get all consent revocation events."""
        return [
            e
            for e in self.get_all()
            if e.event_type
            in (
                AuditEventType.CONSENT_REVOKED,
                AuditEventType.ALL_CONSENTS_REVOKED,
                AuditEventType.CATEGORY_CONSENTS_REVOKED,
            )
        ]

    def query(self) -> AuditQueryBuilder:
        """Start a fluent query builder."""
        return AuditQueryBuilder(self)

    # ========== Statistics ==========

    def get_statistics(
        self, start_time: float | None = None, end_time: float | None = None
    ) -> AuditStatistics:
        """Get audit statistics."""
        events = self.get_range(start_time, end_time)
        return AuditStatistics.from_events(events)

    # ========== Export ==========

    def export_json(self, path: Path | str) -> int:
        """Export audit log to JSON file. Returns event count."""
        path = Path(path)
        events = self.get_all()
        data = [e.to_dict() for e in events]

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.log(
            AuditEventType.AUDIT_EXPORTED,
            details={"path": str(path), "event_count": len(events)},
        )

        return len(events)

    def clear(self) -> int:
        """Clear all audit events. Returns count cleared."""
        return self._storage.clear()


# ========== Query Builder ==========


class AuditQueryBuilder:
    """Fluent interface for building audit log queries."""

    def __init__(self, audit_log: AuditLog) -> None:
        self._audit_log = audit_log
        self._filters: list[Callable[[AuditEvent], bool]] = []
        self._sort_key: Callable[[AuditEvent], Any] | None = None
        self._sort_reverse: bool = False
        self._limit: int | None = None
        self._offset: int = 0

    def by_type(self, *event_types: AuditEventType) -> AuditQueryBuilder:
        """Filter by event type(s)."""
        self._filters.append(lambda e: e.event_type in event_types)
        return self

    def by_topic(self, topic: str) -> AuditQueryBuilder:
        """Filter by topic."""
        self._filters.append(lambda e: e.topic == topic)
        return self

    def by_topic_pattern(self, pattern: str) -> AuditQueryBuilder:
        """Filter by topic pattern (simple glob with *)."""
        import fnmatch

        self._filters.append(lambda e: e.topic is not None and fnmatch.fnmatch(e.topic, pattern))
        return self

    def by_category(self, *categories: ConsentCategory) -> AuditQueryBuilder:
        """Filter by category(s)."""
        self._filters.append(lambda e: e.category in categories)
        return self

    def by_severity(self, *severities: AuditSeverity) -> AuditQueryBuilder:
        """Filter by severity level(s)."""
        self._filters.append(lambda e: e.severity in severities)
        return self

    def by_source(self, source: str) -> AuditQueryBuilder:
        """Filter by source."""
        self._filters.append(lambda e: e.source == source)
        return self

    def by_session(self, session_id: str) -> AuditQueryBuilder:
        """Filter by session ID."""
        self._filters.append(lambda e: e.session_id == session_id)
        return self

    def by_user(self, user_id: str) -> AuditQueryBuilder:
        """Filter by user ID."""
        self._filters.append(lambda e: e.user_id == user_id)
        return self

    def granted_only(self) -> AuditQueryBuilder:
        """Only events where consent was granted."""
        self._filters.append(lambda e: e.granted is True)
        return self

    def denied_only(self) -> AuditQueryBuilder:
        """Only events where consent was denied."""
        self._filters.append(lambda e: e.granted is False)
        return self

    def since(self, timestamp: float | datetime) -> AuditQueryBuilder:
        """Events since timestamp."""
        if isinstance(timestamp, datetime):
            timestamp = timestamp.timestamp()
        self._filters.append(lambda e: e.timestamp >= timestamp)
        return self

    def until(self, timestamp: float | datetime) -> AuditQueryBuilder:
        """Events until timestamp."""
        if isinstance(timestamp, datetime):
            timestamp = timestamp.timestamp()
        self._filters.append(lambda e: e.timestamp <= timestamp)
        return self

    def last_hours(self, hours: int) -> AuditQueryBuilder:
        """Events in the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return self.since(cutoff)

    def last_days(self, days: int) -> AuditQueryBuilder:
        """Events in the last N days."""
        cutoff = time.time() - (days * 86400)
        return self.since(cutoff)

    def sort_by(self, key: Callable[[AuditEvent], Any], reverse: bool = False) -> AuditQueryBuilder:
        """Sort results by a key function."""
        self._sort_key = key
        self._sort_reverse = reverse
        return self

    def sort_by_time(self, newest_first: bool = True) -> AuditQueryBuilder:
        """Sort by timestamp."""
        return self.sort_by(lambda e: e.timestamp, reverse=newest_first)

    def limit(self, count: int) -> AuditQueryBuilder:
        """Limit number of results."""
        self._limit = count
        return self

    def offset(self, count: int) -> AuditQueryBuilder:
        """Skip first N results."""
        self._offset = count
        return self

    def execute(self) -> list[AuditEvent]:
        """Execute the query and return results."""
        events = self._audit_log.get_all()

        # Apply filters
        for filter_fn in self._filters:
            events = [e for e in events if filter_fn(e)]

        # Sort
        if self._sort_key:
            events.sort(key=self._sort_key, reverse=self._sort_reverse)

        # Offset and limit
        if self._offset:
            events = events[self._offset :]
        if self._limit:
            events = events[: self._limit]

        return events

    def count(self) -> int:
        """Count matching events."""
        return len(self.execute())

    def first(self) -> AuditEvent | None:
        """Get first matching event."""
        events = self.limit(1).execute()
        return events[0] if events else None

    def last(self) -> AuditEvent | None:
        """Get last matching event."""
        events = self.sort_by_time(newest_first=True).limit(1).execute()
        return events[0] if events else None


# ========== Statistics ==========


@dataclass
class AuditStatistics:
    """Statistics computed from audit events."""

    total_events: int = 0
    grants: int = 0
    denials: int = 0
    revocations: int = 0
    force_overrides: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_event_type: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    time_range_start: float | None = None
    time_range_end: float | None = None
    unique_topics: int = 0
    unique_sessions: int = 0

    @classmethod
    def from_events(cls, events: list[AuditEvent]) -> AuditStatistics:
        """Compute statistics from a list of events."""
        stats = cls()
        stats.total_events = len(events)

        if not events:
            return stats

        topics: set[str] = set()
        sessions: set[str] = set()

        for event in events:
            # Update time range
            if stats.time_range_start is None or event.timestamp < stats.time_range_start:
                stats.time_range_start = event.timestamp
            if stats.time_range_end is None or event.timestamp > stats.time_range_end:
                stats.time_range_end = event.timestamp

            # Count by event type
            type_name = event.event_type.name
            stats.by_event_type[type_name] = stats.by_event_type.get(type_name, 0) + 1

            # Count by severity
            sev_name = event.severity.value
            stats.by_severity[sev_name] = stats.by_severity.get(sev_name, 0) + 1

            # Count by category
            if event.category:
                cat_name = event.category.value
                stats.by_category[cat_name] = stats.by_category.get(cat_name, 0) + 1

            # Specific counts
            if event.event_type == AuditEventType.CONSENT_GRANTED:
                stats.grants += 1
            elif event.event_type == AuditEventType.CONSENT_DENIED:
                stats.denials += 1
            elif event.event_type in (
                AuditEventType.CONSENT_REVOKED,
                AuditEventType.ALL_CONSENTS_REVOKED,
                AuditEventType.CATEGORY_CONSENTS_REVOKED,
            ):
                stats.revocations += 1
            elif event.event_type == AuditEventType.FORCE_OVERRIDE_USED:
                stats.force_overrides += 1

            # Track unique values
            if event.topic:
                topics.add(event.topic)
            if event.session_id:
                sessions.add(event.session_id)

        stats.unique_topics = len(topics)
        stats.unique_sessions = len(sessions)

        return stats

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_events": self.total_events,
            "grants": self.grants,
            "denials": self.denials,
            "revocations": self.revocations,
            "force_overrides": self.force_overrides,
            "by_category": self.by_category,
            "by_event_type": self.by_event_type,
            "by_severity": self.by_severity,
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "unique_topics": self.unique_topics,
            "unique_sessions": self.unique_sessions,
        }

    @property
    def grant_rate(self) -> float:
        """Percentage of grants vs total decisions."""
        total = self.grants + self.denials
        return (self.grants / total * 100) if total > 0 else 0.0

    @property
    def denial_rate(self) -> float:
        """Percentage of denials vs total decisions."""
        total = self.grants + self.denials
        return (self.denials / total * 100) if total > 0 else 0.0


# ========== Compliance Report ==========


@dataclass
class ComplianceReportSection:
    """A section in a compliance report."""

    title: str
    content: str
    events: list[AuditEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AuditReport:
    """Generate compliance reports from audit logs."""

    def __init__(self, audit_log: AuditLog) -> None:
        self._audit_log = audit_log

    def generate_summary(
        self, start_time: float | None = None, end_time: float | None = None
    ) -> str:
        """Generate a text summary report."""
        stats = self._audit_log.get_statistics(start_time, end_time)

        lines = [
            "=" * 60,
            "CONSENT AUDIT SUMMARY REPORT",
            "=" * 60,
            "",
            f"Total Events: {stats.total_events}",
            f"Unique Topics: {stats.unique_topics}",
            f"Unique Sessions: {stats.unique_sessions}",
            "",
            "DECISION BREAKDOWN:",
            f"  Grants: {stats.grants} ({stats.grant_rate:.1f}%)",
            f"  Denials: {stats.denials} ({stats.denial_rate:.1f}%)",
            f"  Revocations: {stats.revocations}",
            f"  Force Overrides: {stats.force_overrides}",
            "",
        ]

        if stats.by_category:
            lines.append("BY CATEGORY:")
            for cat, count in sorted(stats.by_category.items()):
                lines.append(f"  {cat}: {count}")
            lines.append("")

        if stats.by_severity:
            lines.append("BY SEVERITY:")
            for sev, count in sorted(stats.by_severity.items()):
                lines.append(f"  {sev}: {count}")
            lines.append("")

        if stats.time_range_start and stats.time_range_end:
            start_dt = datetime.fromtimestamp(stats.time_range_start)
            end_dt = datetime.fromtimestamp(stats.time_range_end)
            lines.append(f"Time Range: {start_dt} to {end_dt}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def generate_compliance_check(self) -> list[ComplianceReportSection]:
        """Generate a compliance check report."""
        sections: list[ComplianceReportSection] = []

        # Check for force overrides
        force_events = self._audit_log.query().by_type(AuditEventType.FORCE_OVERRIDE_USED).execute()
        if force_events:
            section = ComplianceReportSection(
                title="Force Override Usage",
                content=f"Found {len(force_events)} force override events.",
                events=force_events,
                warnings=["Force overrides bypass consent checks - review for compliance."],
            )
            sections.append(section)

        # Check for denied consents that were later granted
        denials = (
            self._audit_log.query()
            .by_type(AuditEventType.CONSENT_DENIED)
            .sort_by_time(newest_first=False)
            .execute()
        )

        for denial in denials:
            if not denial.topic:
                continue
            # Check if later granted
            later_grants = (
                self._audit_log.query()
                .by_topic(denial.topic)
                .by_type(AuditEventType.CONSENT_GRANTED)
                .since(denial.timestamp)
                .execute()
            )

            if later_grants:
                section = ComplianceReportSection(
                    title=f"Denial-then-Grant: {denial.topic}",
                    content="Consent was denied then later granted.",
                    events=[denial] + later_grants,
                    warnings=["User may have been pressured or misclicked."],
                )
                sections.append(section)

        # Check for sensitive category usage
        sensitive_categories = [
            ConsentCategory.REMOTE_LLM,
            ConsentCategory.DATA_COLLECTION,
        ]
        for category in sensitive_categories:
            events = self._audit_log.query().by_category(category).execute()
            if events:
                section = ComplianceReportSection(
                    title=f"Sensitive Category: {category.value}",
                    content=f"Found {len(events)} events for sensitive category.",
                    events=events,
                )
                sections.append(section)

        return sections

    def export_html(self, path: Path | str) -> None:
        """Export report as HTML file."""
        path = Path(path)
        stats = self._audit_log.get_statistics()
        events = self._audit_log.get_recent(1000)

        html = [
            "<!DOCTYPE html>",
            "<html><head><title>Consent Audit Report</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            ".granted { background-color: #dff0d8; }",
            ".denied { background-color: #f2dede; }",
            "</style></head><body>",
            "<h1>Consent Audit Report</h1>",
            f"<p>Generated: {datetime.now()}</p>",
            f"<p>Total Events: {stats.total_events}</p>",
            f"<p>Grants: {stats.grants} | Denials: {stats.denials}</p>",
            "<h2>Recent Events</h2>",
            "<table><tr><th>Time</th><th>Type</th><th>Topic</th><th>Result</th></tr>",
        ]

        for event in reversed(events[-100:]):
            dt = event.datetime.strftime("%Y-%m-%d %H:%M:%S")
            result = ""
            row_class = ""
            if event.granted is True:
                result = "GRANTED"
                row_class = "granted"
            elif event.granted is False:
                result = "DENIED"
                row_class = "denied"

            html.append(
                f"<tr class='{row_class}'><td>{dt}</td><td>{event.event_type.name}</td>"
                f"<td>{event.topic or '-'}</td><td>{result}</td></tr>"
            )

        html.append("</table></body></html>")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(html), encoding="utf-8")


# ========== Audited Consent Manager ==========


class AuditedConsentManager:
    """Wrapper that adds audit logging to a ConsentManager.

    This provides a drop-in replacement that automatically logs all
    consent operations to an audit log.
    """

    def __init__(
        self,
        consent_manager: ConsentManager | None = None,
        audit_log: AuditLog | None = None,
        storage_path: Path | None = None,
    ) -> None:
        self._manager = consent_manager or ConsentManager()
        self._audit_log = audit_log or AuditLog(
            storage=(
                FileAuditStorage(storage_path or Path.home() / ".proxima" / "audit.log")
                if storage_path
                else MemoryAuditStorage()
            )
        )

        # Log session start
        self._audit_log.log(AuditEventType.SESSION_STARTED)

    @property
    def manager(self) -> ConsentManager:
        """Access the underlying ConsentManager."""
        return self._manager

    @property
    def audit_log(self) -> AuditLog:
        """Access the audit log."""
        return self._audit_log

    def request_consent(
        self,
        topic: str,
        category: ConsentCategory | None = None,
        description: str | None = None,
    ) -> bool:
        """Request consent with audit logging."""
        # Log the request
        self._audit_log.log(
            AuditEventType.CONSENT_REQUESTED,
            topic=topic,
            category=category,
            details={"description": description},
        )

        # Check if already remembered
        remembered = self._manager.check_remembered(topic)
        if remembered.found:
            self._audit_log.log(
                AuditEventType.CONSENT_REMEMBERED,
                topic=topic,
                category=category,
                granted=remembered.granted,
                level=remembered.record.level if remembered.record else None,
            )
            return remembered.granted

        # Request actual consent
        granted = self._manager.request_consent(topic, category, description)

        # Log the decision
        if granted:
            self._audit_log.log(
                AuditEventType.CONSENT_GRANTED,
                topic=topic,
                category=category,
                granted=True,
            )
        else:
            self._audit_log.log(
                AuditEventType.CONSENT_DENIED,
                topic=topic,
                category=category,
                granted=False,
            )

        return granted

    def grant(
        self,
        topic: str,
        level: ConsentLevel = ConsentLevel.SESSION,
        category: ConsentCategory | None = None,
        context: str | None = None,
        duration_seconds: float | None = None,
    ) -> ConsentRecord:
        """Grant consent with audit logging."""
        record = self._manager.grant(topic, level, category, context, duration_seconds)

        self._audit_log.log(
            AuditEventType.CONSENT_GRANTED,
            topic=topic,
            category=category,
            granted=True,
            level=level,
            source="programmatic",
            details={"context": context, "duration": duration_seconds},
        )

        return record

    def deny(
        self,
        topic: str,
        level: ConsentLevel = ConsentLevel.SESSION,
        category: ConsentCategory | None = None,
        context: str | None = None,
    ) -> ConsentRecord:
        """Deny consent with audit logging."""
        record = self._manager.deny(topic, level, category, context)

        self._audit_log.log(
            AuditEventType.CONSENT_DENIED,
            topic=topic,
            category=category,
            granted=False,
            level=level,
            source="programmatic",
            details={"context": context},
        )

        return record

    def revoke(self, topic: str) -> bool:
        """Revoke consent with audit logging."""
        was_granted = self._manager.revoke(topic)

        self._audit_log.log(
            AuditEventType.CONSENT_REVOKED,
            topic=topic,
            details={"was_granted": was_granted},
        )

        return was_granted

    def revoke_all(self) -> None:
        """Revoke all consents with audit logging."""
        self._manager.revoke_all()

        self._audit_log.log(
            AuditEventType.ALL_CONSENTS_REVOKED,
            severity=AuditSeverity.WARNING,
        )

    def revoke_category(self, category: ConsentCategory) -> int:
        """Revoke category consents with audit logging."""
        count = self._manager.revoke_category(category)

        self._audit_log.log(
            AuditEventType.CATEGORY_CONSENTS_REVOKED,
            category=category,
            details={"count": count},
        )

        return count

    def enable_force_override(self) -> None:
        """Enable force override with audit logging."""
        self._manager.enable_force_override()

        self._audit_log.log(
            AuditEventType.FORCE_OVERRIDE_ENABLED,
            severity=AuditSeverity.WARNING,
        )

    def disable_force_override(self) -> None:
        """Disable force override with audit logging."""
        self._manager.disable_force_override()

        self._audit_log.log(AuditEventType.FORCE_OVERRIDE_DISABLED)

    def check(self, topic: str) -> bool | None:
        """Check consent status with audit logging."""
        result = self._manager.check(topic)

        self._audit_log.log(
            AuditEventType.CONSENT_CHECKED,
            topic=topic,
            granted=result,
            severity=AuditSeverity.DEBUG,
        )

        return result

    # Delegate other methods without logging
    def check_remembered(self, topic: str):
        return self._manager.check_remembered(topic)

    def list_granted(self) -> list[str]:
        return self._manager.list_granted()

    def list_denied(self) -> list[str]:
        return self._manager.list_denied()

    def get_record(self, topic: str) -> ConsentRecord | None:
        return self._manager.get_record(topic)

    def summary(self) -> dict:
        return self._manager.summary()

    def save(self) -> None:
        self._manager.save()
        self._audit_log.log(AuditEventType.STORAGE_SAVED)

    def load(self) -> None:
        self._manager.load()
        self._audit_log.log(AuditEventType.STORAGE_LOADED)

    def close(self) -> None:
        """Close the audited consent manager, logging session end."""
        self._audit_log.log(AuditEventType.SESSION_ENDED)


# ========== Convenience Functions ==========


def create_audit_log(
    path: Path | str | None = None,
    in_memory: bool = False,
    max_size_mb: float = 10.0,
    session_id: str | None = None,
) -> AuditLog:
    """Create an audit log with sensible defaults."""
    if in_memory:
        storage: AuditStorage = MemoryAuditStorage()
    else:
        audit_path = Path(path) if path else Path.home() / ".proxima" / "consent_audit.log"
        storage = FileAuditStorage(audit_path, max_size_mb=max_size_mb)

    return AuditLog(storage=storage, session_id=session_id)


def create_audited_manager(
    audit_path: Path | str | None = None,
    consent_storage_path: Path | str | None = None,
) -> AuditedConsentManager:
    """Create an audited consent manager with defaults."""
    consent_path = Path(consent_storage_path) if consent_storage_path else None
    manager = ConsentManager(storage_path=consent_path)

    audit_log = create_audit_log(audit_path)

    return AuditedConsentManager(consent_manager=manager, audit_log=audit_log)
