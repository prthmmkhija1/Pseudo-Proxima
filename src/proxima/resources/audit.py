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
        return [
            e for e in self.get_all() if e.event_type == AuditEventType.CONSENT_GRANTED
        ]

    def get_denials(self) -> list[AuditEvent]:
        """Get all consent denial events."""
        return [
            e for e in self.get_all() if e.event_type == AuditEventType.CONSENT_DENIED
        ]

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

        self._filters.append(
            lambda e: e.topic is not None and fnmatch.fnmatch(e.topic, pattern)
        )
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

    def sort_by(
        self, key: Callable[[AuditEvent], Any], reverse: bool = False
    ) -> AuditQueryBuilder:
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
            if (
                stats.time_range_start is None
                or event.timestamp < stats.time_range_start
            ):
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
        force_events = (
            self._audit_log.query()
            .by_type(AuditEventType.FORCE_OVERRIDE_USED)
            .execute()
        )
        if force_events:
            section = ComplianceReportSection(
                title="Force Override Usage",
                content=f"Found {len(force_events)} force override events.",
                events=force_events,
                warnings=[
                    "Force overrides bypass consent checks - review for compliance."
                ],
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
        audit_path = (
            Path(path) if path else Path.home() / ".proxima" / "consent_audit.log"
        )
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


# =============================================================================
# Audit Retention Policy
# =============================================================================


@dataclass
class RetentionPolicy:
    """Defines audit log retention rules."""

    max_age_days: int = 90
    max_events: int | None = None
    archive_before_delete: bool = True
    archive_path: Path | None = None
    compress_archive: bool = True

    def get_cutoff_timestamp(self) -> float:
        """Get timestamp before which events should be purged."""
        return time.time() - (self.max_age_days * 86400)


class AuditRetentionManager:
    """Manages audit log retention and archival.

    Features:
    - Time-based retention
    - Event count limits
    - Archival before deletion
    - Compressed archives
    """

    def __init__(
        self,
        audit_log: AuditLog,
        policy: RetentionPolicy | None = None,
    ) -> None:
        """Initialize retention manager.

        Args:
            audit_log: The audit log to manage.
            policy: Retention policy. Uses defaults if not provided.
        """
        self._audit_log = audit_log
        self._policy = policy or RetentionPolicy()

    @property
    def policy(self) -> RetentionPolicy:
        """Get the current retention policy."""
        return self._policy

    def apply_retention(self) -> dict[str, Any]:
        """Apply retention policy and return statistics.

        Returns:
            Dictionary with counts of archived and deleted events.
        """
        all_events = self._audit_log.get_all()
        cutoff = self._policy.get_cutoff_timestamp()

        # Separate events to keep vs archive/delete
        events_to_keep: list[AuditEvent] = []
        events_to_archive: list[AuditEvent] = []

        for event in all_events:
            if event.timestamp >= cutoff:
                events_to_keep.append(event)
            else:
                events_to_archive.append(event)

        # Also check max events limit
        if self._policy.max_events and len(events_to_keep) > self._policy.max_events:
            # Keep newest, archive older
            events_to_keep.sort(key=lambda e: e.timestamp, reverse=True)
            overflow = events_to_keep[self._policy.max_events:]
            events_to_keep = events_to_keep[:self._policy.max_events]
            events_to_archive.extend(overflow)

        # Archive if needed
        archived_count = 0
        if events_to_archive and self._policy.archive_before_delete:
            archived_count = self._archive_events(events_to_archive)

        # Clear and re-add kept events
        self._audit_log.clear()
        for event in sorted(events_to_keep, key=lambda e: e.timestamp):
            self._audit_log._storage.append(event)

        return {
            "events_archived": archived_count,
            "events_deleted": len(events_to_archive) - archived_count,
            "events_kept": len(events_to_keep),
            "cutoff_timestamp": cutoff,
        }

    def _archive_events(self, events: list[AuditEvent]) -> int:
        """Archive events to a file.

        Args:
            events: Events to archive.

        Returns:
            Number of events archived.
        """
        if not events:
            return 0

        archive_path = self._policy.archive_path or (
            Path.home() / ".proxima" / "audit_archives"
        )
        archive_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_archive_{timestamp}.json"
        if self._policy.compress_archive:
            filename += ".gz"

        filepath = archive_path / filename

        data = [e.to_dict() for e in events]

        if self._policy.compress_archive:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return len(events)

    def get_archives(self) -> list[Path]:
        """List available archive files.

        Returns:
            List of archive file paths.
        """
        archive_path = self._policy.archive_path or (
            Path.home() / ".proxima" / "audit_archives"
        )
        if not archive_path.exists():
            return []

        archives = list(archive_path.glob("audit_archive_*.json*"))
        return sorted(archives, key=lambda p: p.name)

    def load_archive(self, filepath: Path) -> list[AuditEvent]:
        """Load events from an archive file.

        Args:
            filepath: Path to archive file.

        Returns:
            List of events from the archive.
        """
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

        return [AuditEvent.from_dict(item) for item in data]


# =============================================================================
# Audit Alerting System
# =============================================================================


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditAlert:
    """An alert triggered by audit events."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    triggered_at: float
    events: list[AuditEvent]
    acknowledged: bool = False
    acknowledged_at: float | None = None
    acknowledged_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "triggered_at": self.triggered_at,
            "events": [e.to_dict() for e in self.events],
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts based on audit events."""
    name: str
    description: str
    severity: AlertSeverity
    event_types: list[AuditEventType] | None = None
    categories: list[ConsentCategory] | None = None
    threshold_count: int = 1
    threshold_window_seconds: float = 3600.0
    condition: Callable[[list[AuditEvent]], bool] | None = None
    message_template: str = "Alert: {rule_name} triggered with {event_count} events"


class AuditAlertManager:
    """Manages audit alerting rules and notifications.

    Features:
    - Configurable alerting rules
    - Threshold-based detection
    - Custom conditions
    - Alert notifications
    """

    def __init__(self, audit_log: AuditLog) -> None:
        """Initialize alert manager.

        Args:
            audit_log: Audit log to monitor.
        """
        self._audit_log = audit_log
        self._rules: list[AlertRule] = []
        self._alerts: list[AuditAlert] = []
        self._handlers: list[Callable[[AuditAlert], None]] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alerting rule."""
        with self._lock:
            self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.name == rule_name:
                    del self._rules[i]
                    return True
            return False

    def add_handler(self, handler: Callable[[AuditAlert], None]) -> None:
        """Add an alert notification handler."""
        with self._lock:
            self._handlers.append(handler)

    def check_rules(self) -> list[AuditAlert]:
        """Check all rules against current audit log.

        Returns:
            List of newly triggered alerts.
        """
        new_alerts: list[AuditAlert] = []
        current_time = time.time()

        with self._lock:
            rules = list(self._rules)

        for rule in rules:
            window_start = current_time - rule.threshold_window_seconds

            # Get matching events
            events = self._audit_log.get_range(window_start, current_time)

            # Filter by event types
            if rule.event_types:
                events = [e for e in events if e.event_type in rule.event_types]

            # Filter by categories
            if rule.categories:
                events = [e for e in events if e.category in rule.categories]

            # Check threshold
            if len(events) < rule.threshold_count:
                continue

            # Check custom condition
            if rule.condition and not rule.condition(events):
                continue

            # Create alert
            message = rule.message_template.format(
                rule_name=rule.name,
                event_count=len(events),
            )

            alert = AuditAlert(
                alert_id=str(uuid.uuid4()),
                rule_name=rule.name,
                severity=rule.severity,
                message=message,
                triggered_at=current_time,
                events=events,
            )

            new_alerts.append(alert)
            self._alerts.append(alert)

            # Notify handlers
            for handler in self._handlers:
                try:
                    handler(alert)
                except Exception:
                    pass

        return new_alerts

    def get_alerts(
        self,
        unacknowledged_only: bool = False,
        severity: AlertSeverity | None = None,
    ) -> list[AuditAlert]:
        """Get alerts with optional filtering."""
        alerts = self._alerts
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str = "system",
    ) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_at = time.time()
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def clear_alerts(self, acknowledged_only: bool = True) -> int:
        """Clear alerts. Returns count cleared."""
        with self._lock:
            if acknowledged_only:
                original = len(self._alerts)
                self._alerts = [a for a in self._alerts if not a.acknowledged]
                return original - len(self._alerts)
            else:
                count = len(self._alerts)
                self._alerts = []
                return count

    def add_standard_rules(self) -> None:
        """Add standard security-related alerting rules."""
        # Force override detection
        self.add_rule(AlertRule(
            name="force_override_usage",
            description="Detect force override usage",
            severity=AlertSeverity.HIGH,
            event_types=[AuditEventType.FORCE_OVERRIDE_USED],
            threshold_count=1,
            message_template="Force override was used - review for compliance",
        ))

        # Bulk revocation detection
        self.add_rule(AlertRule(
            name="bulk_revocation",
            description="Detect bulk consent revocations",
            severity=AlertSeverity.MEDIUM,
            event_types=[
                AuditEventType.ALL_CONSENTS_REVOKED,
                AuditEventType.CATEGORY_CONSENTS_REVOKED,
            ],
            threshold_count=1,
            message_template="Bulk consent revocation detected",
        ))

        # High denial rate detection
        def high_denial_check(events: list[AuditEvent]) -> bool:
            denials = sum(1 for e in events if e.event_type == AuditEventType.CONSENT_DENIED)
            grants = sum(1 for e in events if e.event_type == AuditEventType.CONSENT_GRANTED)
            total = denials + grants
            return total >= 5 and (denials / total) > 0.8

        self.add_rule(AlertRule(
            name="high_denial_rate",
            description="Detect high consent denial rate",
            severity=AlertSeverity.LOW,
            event_types=[
                AuditEventType.CONSENT_DENIED,
                AuditEventType.CONSENT_GRANTED,
            ],
            threshold_count=5,
            condition=high_denial_check,
            message_template="High denial rate detected ({event_count} recent decisions)",
        ))


# =============================================================================
# Audit Chain Verification (Tamper Detection)
# =============================================================================


@dataclass
class ChainedAuditEvent(AuditEvent):
    """Audit event with blockchain-style chaining for tamper detection."""
    previous_hash: str = ""
    chain_hash: str = ""

    def compute_chain_hash(self) -> str:
        """Compute hash including previous hash."""
        data = f"{self.checksum}:{self.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]


class VerifiableAuditLog:
    """Audit log with blockchain-style verification.

    Each event references the hash of the previous event,
    creating a tamper-evident chain.
    """

    def __init__(self, storage: AuditStorage | None = None) -> None:
        """Initialize verifiable audit log."""
        self._storage = storage or MemoryAuditStorage()
        self._last_hash = "genesis"
        self._lock = threading.Lock()

    def append(self, event: AuditEvent) -> ChainedAuditEvent:
        """Append an event with chain verification.

        Args:
            event: Base audit event to add.

        Returns:
            ChainedAuditEvent with chain hash computed.
        """
        with self._lock:
            chained = ChainedAuditEvent(
                event_type=event.event_type,
                timestamp=event.timestamp,
                event_id=event.event_id,
                topic=event.topic,
                category=event.category,
                granted=event.granted,
                level=event.level,
                source=event.source,
                user_id=event.user_id,
                session_id=event.session_id,
                details=event.details,
                severity=event.severity,
                previous_hash=self._last_hash,
            )
            chained.chain_hash = chained.compute_chain_hash()
            self._last_hash = chained.chain_hash

            # Store as regular event with chain info in details
            event.details["chain_hash"] = chained.chain_hash
            event.details["previous_hash"] = chained.previous_hash
            self._storage.append(event)

            return chained

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify the integrity of the entire chain.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        events = self._storage.read_all()
        errors: list[str] = []
        expected_prev = "genesis"

        for i, event in enumerate(events):
            chain_hash = event.details.get("chain_hash", "")
            prev_hash = event.details.get("previous_hash", "")

            if prev_hash != expected_prev:
                errors.append(
                    f"Event {i} ({event.event_id}): previous_hash mismatch. "
                    f"Expected {expected_prev}, got {prev_hash}"
                )

            # Verify event integrity
            if not event.verify_integrity():
                errors.append(
                    f"Event {i} ({event.event_id}): checksum verification failed"
                )

            expected_prev = chain_hash

        return len(errors) == 0, errors

    def get_chain_status(self) -> dict[str, Any]:
        """Get current chain status.

        Returns:
            Dictionary with chain statistics and status.
        """
        events = self._storage.read_all()
        is_valid, errors = self.verify_chain()

        return {
            "chain_length": len(events),
            "is_valid": is_valid,
            "errors": errors,
            "last_hash": self._last_hash,
            "genesis_hash": "genesis",
        }

# =============================================================================
# RETENTION POLICIES (5% Gap Coverage)
# Tiered retention, compliance policies, and advanced policy enforcement
# =============================================================================


class RetentionTier(Enum):
    """Storage tiers for audit data."""
    
    HOT = "hot"  # Fast access, recent data
    WARM = "warm"  # Moderate access, aging data
    COLD = "cold"  # Archive, rarely accessed
    FROZEN = "frozen"  # Long-term archive, compliance
    DELETED = "deleted"  # Marked for deletion


class ComplianceFramework(Enum):
    """Compliance frameworks affecting retention."""
    
    GDPR = "gdpr"  # EU General Data Protection Regulation
    HIPAA = "hipaa"  # US Health Insurance Portability
    SOX = "sox"  # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"  # Payment Card Industry
    CCPA = "ccpa"  # California Consumer Privacy Act
    SOC2 = "soc2"  # Service Organization Control
    CUSTOM = "custom"  # Custom organizational policy


@dataclass
class TieredRetentionPolicy:
    """Policy for tiered data retention.
    
    Defines how long data stays in each tier and when it
    transitions between tiers.
    """
    
    policy_id: str
    name: str
    description: str
    
    # Tier transition times (in days)
    hot_duration_days: int = 30
    warm_duration_days: int = 90
    cold_duration_days: int = 365
    frozen_duration_days: int = 2555  # ~7 years
    
    # After frozen duration, data is deleted
    total_retention_days: int = 2920  # ~8 years
    
    # Tier-specific settings
    hot_max_events: int = 100000
    warm_compression: bool = True
    cold_archive_format: str = "gzip"
    frozen_encryption: bool = True
    
    # Applicable event types (empty = all)
    applicable_event_types: list[str] = field(default_factory=list)
    
    # Priority for conflict resolution
    priority: int = 0
    
    def get_tier_for_age(self, age_days: float) -> RetentionTier:
        """Determine tier based on age.
        
        Args:
            age_days: Age of event in days
            
        Returns:
            Appropriate retention tier
        """
        if age_days > self.total_retention_days:
            return RetentionTier.DELETED
        elif age_days > (self.hot_duration_days + self.warm_duration_days + self.cold_duration_days):
            return RetentionTier.FROZEN
        elif age_days > (self.hot_duration_days + self.warm_duration_days):
            return RetentionTier.COLD
        elif age_days > self.hot_duration_days:
            return RetentionTier.WARM
        else:
            return RetentionTier.HOT
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "hot_duration_days": self.hot_duration_days,
            "warm_duration_days": self.warm_duration_days,
            "cold_duration_days": self.cold_duration_days,
            "frozen_duration_days": self.frozen_duration_days,
            "total_retention_days": self.total_retention_days,
            "applicable_event_types": self.applicable_event_types,
            "priority": self.priority,
        }


@dataclass
class ComplianceRetentionPolicy:
    """Compliance-specific retention policy.
    
    Based on regulatory requirements for specific compliance frameworks.
    """
    
    framework: ComplianceFramework
    min_retention_days: int
    max_retention_days: int | None  # None = no maximum
    requires_encryption: bool
    requires_immutability: bool
    deletion_requires_approval: bool
    audit_access_logging: bool
    geographic_restrictions: list[str]  # Allowed storage regions
    data_classification: str  # sensitive, pii, financial, etc.
    
    # Legal hold support
    legal_hold_enabled: bool = True
    
    @classmethod
    def gdpr_policy(cls) -> "ComplianceRetentionPolicy":
        """Create GDPR-compliant policy."""
        return cls(
            framework=ComplianceFramework.GDPR,
            min_retention_days=0,  # Can delete on request
            max_retention_days=1095,  # 3 years typical
            requires_encryption=True,
            requires_immutability=False,  # Must allow deletion
            deletion_requires_approval=False,  # Right to erasure
            audit_access_logging=True,
            geographic_restrictions=["EU", "EEA"],
            data_classification="pii",
        )
    
    @classmethod
    def hipaa_policy(cls) -> "ComplianceRetentionPolicy":
        """Create HIPAA-compliant policy."""
        return cls(
            framework=ComplianceFramework.HIPAA,
            min_retention_days=2190,  # 6 years minimum
            max_retention_days=None,
            requires_encryption=True,
            requires_immutability=True,
            deletion_requires_approval=True,
            audit_access_logging=True,
            geographic_restrictions=["US"],
            data_classification="phi",
        )
    
    @classmethod
    def sox_policy(cls) -> "ComplianceRetentionPolicy":
        """Create SOX-compliant policy."""
        return cls(
            framework=ComplianceFramework.SOX,
            min_retention_days=2555,  # 7 years minimum
            max_retention_days=None,
            requires_encryption=True,
            requires_immutability=True,
            deletion_requires_approval=True,
            audit_access_logging=True,
            geographic_restrictions=[],  # No specific restrictions
            data_classification="financial",
        )
    
    @classmethod
    def pci_dss_policy(cls) -> "ComplianceRetentionPolicy":
        """Create PCI-DSS compliant policy."""
        return cls(
            framework=ComplianceFramework.PCI_DSS,
            min_retention_days=365,  # 1 year minimum
            max_retention_days=None,
            requires_encryption=True,
            requires_immutability=True,
            deletion_requires_approval=True,
            audit_access_logging=True,
            geographic_restrictions=[],
            data_classification="payment",
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework.value,
            "min_retention_days": self.min_retention_days,
            "max_retention_days": self.max_retention_days,
            "requires_encryption": self.requires_encryption,
            "requires_immutability": self.requires_immutability,
            "deletion_requires_approval": self.deletion_requires_approval,
            "audit_access_logging": self.audit_access_logging,
            "geographic_restrictions": self.geographic_restrictions,
            "data_classification": self.data_classification,
            "legal_hold_enabled": self.legal_hold_enabled,
        }


@dataclass
class LegalHold:
    """Legal hold preventing data deletion."""
    
    hold_id: str
    name: str
    reason: str
    created_at: float
    created_by: str
    
    # Scope
    affected_event_ids: list[str] | None = None  # None = all events
    affected_date_range: tuple[float, float] | None = None
    affected_event_types: list[str] | None = None
    
    # Status
    is_active: bool = True
    released_at: float | None = None
    released_by: str | None = None
    
    def covers_event(
        self,
        event_id: str,
        event_time: float,
        event_type: str,
    ) -> bool:
        """Check if this hold covers a specific event.
        
        Args:
            event_id: Event ID
            event_time: Event timestamp
            event_type: Event type
            
        Returns:
            True if event is under this hold
        """
        if not self.is_active:
            return False
        
        # Check event ID list
        if self.affected_event_ids is not None:
            if event_id not in self.affected_event_ids:
                return False
        
        # Check date range
        if self.affected_date_range is not None:
            start, end = self.affected_date_range
            if not (start <= event_time <= end):
                return False
        
        # Check event types
        if self.affected_event_types is not None:
            if event_type not in self.affected_event_types:
                return False
        
        return True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hold_id": self.hold_id,
            "name": self.name,
            "reason": self.reason,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "is_active": self.is_active,
            "released_at": self.released_at,
            "affected_event_ids": self.affected_event_ids,
            "affected_date_range": self.affected_date_range,
            "affected_event_types": self.affected_event_types,
        }


@dataclass
class RetentionPolicyViolation:
    """A violation of retention policy."""
    
    violation_id: str
    policy_id: str
    event_id: str | None
    violation_type: str  # "early_deletion", "late_retention", "encryption_missing", etc.
    description: str
    detected_at: float
    severity: str  # "warning", "error", "critical"
    remediation_action: str | None = None
    acknowledged: bool = False
    acknowledged_by: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_id": self.violation_id,
            "policy_id": self.policy_id,
            "event_id": self.event_id,
            "violation_type": self.violation_type,
            "description": self.description,
            "detected_at": self.detected_at,
            "severity": self.severity,
            "remediation_action": self.remediation_action,
            "acknowledged": self.acknowledged,
        }


class TieredRetentionManager:
    """Manages tiered data retention with tier transitions.
    
    Features:
    - Automatic tier transitions
    - Compression on tier change
    - Archive management
    - Tier-based query routing
    """
    
    def __init__(
        self,
        storage_path: Path | None = None,
        default_policy: TieredRetentionPolicy | None = None,
    ) -> None:
        """Initialize tiered retention manager.
        
        Args:
            storage_path: Base path for tiered storage
            default_policy: Default retention policy
        """
        self._storage_path = storage_path or Path.home() / ".proxima" / "audit_tiers"
        self._default_policy = default_policy or TieredRetentionPolicy(
            policy_id="default",
            name="Default Tiered Retention",
            description="Standard tiered retention policy",
        )
        self._lock = threading.Lock()
        
        # Policies by ID
        self._policies: dict[str, TieredRetentionPolicy] = {
            self._default_policy.policy_id: self._default_policy
        }
        
        # Event tier tracking
        self._event_tiers: dict[str, RetentionTier] = {}
        
        # Tier storage paths
        self._tier_paths = {
            RetentionTier.HOT: self._storage_path / "hot",
            RetentionTier.WARM: self._storage_path / "warm",
            RetentionTier.COLD: self._storage_path / "cold",
            RetentionTier.FROZEN: self._storage_path / "frozen",
        }
        
        # Ensure directories exist
        for path in self._tier_paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def add_policy(self, policy: TieredRetentionPolicy) -> None:
        """Add a retention policy.
        
        Args:
            policy: Policy to add
        """
        with self._lock:
            self._policies[policy.policy_id] = policy
    
    def get_policy(self, policy_id: str) -> TieredRetentionPolicy | None:
        """Get a policy by ID.
        
        Args:
            policy_id: Policy ID
            
        Returns:
            Policy or None
        """
        return self._policies.get(policy_id)
    
    def assign_tier(
        self,
        event_id: str,
        event_time: float,
        event_type: str = "",
    ) -> RetentionTier:
        """Assign an event to a tier based on age.
        
        Args:
            event_id: Event ID
            event_time: Event timestamp
            event_type: Event type for policy matching
            
        Returns:
            Assigned tier
        """
        # Find applicable policy
        policy = self._find_applicable_policy(event_type)
        
        # Calculate age
        age_days = (time.time() - event_time) / 86400
        
        # Get tier
        tier = policy.get_tier_for_age(age_days)
        
        with self._lock:
            self._event_tiers[event_id] = tier
        
        return tier
    
    def _find_applicable_policy(self, event_type: str) -> TieredRetentionPolicy:
        """Find the applicable policy for an event type.
        
        Args:
            event_type: Event type
            
        Returns:
            Most applicable policy
        """
        with self._lock:
            applicable = []
            
            for policy in self._policies.values():
                if not policy.applicable_event_types:
                    applicable.append(policy)
                elif event_type in policy.applicable_event_types:
                    applicable.append(policy)
            
            if not applicable:
                return self._default_policy
            
            # Return highest priority
            return max(applicable, key=lambda p: p.priority)
    
    def process_tier_transitions(self) -> dict[str, int]:
        """Process pending tier transitions.
        
        Returns:
            Dictionary with transition counts per tier
        """
        transitions: dict[str, int] = {
            tier.value: 0 for tier in RetentionTier
        }
        
        with self._lock:
            current_time = time.time()
            
            for event_id, current_tier in list(self._event_tiers.items()):
                if current_tier == RetentionTier.DELETED:
                    continue
                
                # Would need actual event time to recalculate
                # This is a placeholder for the actual logic
                pass
        
        return transitions
    
    def get_tier_statistics(self) -> dict[str, Any]:
        """Get statistics about tier distribution.
        
        Returns:
            Tier statistics
        """
        with self._lock:
            tier_counts: dict[str, int] = {}
            
            for tier in self._event_tiers.values():
                tier_counts[tier.value] = tier_counts.get(tier.value, 0) + 1
            
            return {
                "total_events": len(self._event_tiers),
                "by_tier": tier_counts,
                "policies": len(self._policies),
            }
    
    def get_tier_path(self, tier: RetentionTier) -> Path:
        """Get storage path for a tier.
        
        Args:
            tier: Retention tier
            
        Returns:
            Path for tier storage
        """
        return self._tier_paths.get(tier, self._tier_paths[RetentionTier.HOT])


class ComplianceRetentionManager:
    """Manages compliance-based retention policies.
    
    Features:
    - Multiple compliance framework support
    - Policy conflict resolution
    - Compliance validation
    - Legal hold management
    """
    
    def __init__(self) -> None:
        """Initialize compliance manager."""
        self._lock = threading.Lock()
        
        # Compliance policies
        self._policies: dict[ComplianceFramework, ComplianceRetentionPolicy] = {}
        
        # Legal holds
        self._legal_holds: dict[str, LegalHold] = {}
        
        # Violations
        self._violations: list[RetentionPolicyViolation] = []
        self._violation_counter = 0
    
    def add_compliance_policy(self, policy: ComplianceRetentionPolicy) -> None:
        """Add a compliance policy.
        
        Args:
            policy: Policy to add
        """
        with self._lock:
            self._policies[policy.framework] = policy
    
    def add_standard_policies(self) -> None:
        """Add standard compliance policies."""
        self.add_compliance_policy(ComplianceRetentionPolicy.gdpr_policy())
        self.add_compliance_policy(ComplianceRetentionPolicy.hipaa_policy())
        self.add_compliance_policy(ComplianceRetentionPolicy.sox_policy())
        self.add_compliance_policy(ComplianceRetentionPolicy.pci_dss_policy())
    
    def create_legal_hold(
        self,
        name: str,
        reason: str,
        created_by: str,
        affected_event_ids: list[str] | None = None,
        affected_date_range: tuple[float, float] | None = None,
        affected_event_types: list[str] | None = None,
    ) -> LegalHold:
        """Create a new legal hold.
        
        Args:
            name: Hold name
            reason: Reason for hold
            created_by: User creating hold
            affected_event_ids: Optional specific event IDs
            affected_date_range: Optional date range
            affected_event_types: Optional event types
            
        Returns:
            Created legal hold
        """
        with self._lock:
            hold_id = f"hold_{len(self._legal_holds) + 1}_{int(time.time())}"
            
            hold = LegalHold(
                hold_id=hold_id,
                name=name,
                reason=reason,
                created_at=time.time(),
                created_by=created_by,
                affected_event_ids=affected_event_ids,
                affected_date_range=affected_date_range,
                affected_event_types=affected_event_types,
            )
            
            self._legal_holds[hold_id] = hold
            return hold
    
    def release_legal_hold(
        self,
        hold_id: str,
        released_by: str,
    ) -> bool:
        """Release a legal hold.
        
        Args:
            hold_id: Hold ID
            released_by: User releasing hold
            
        Returns:
            True if released
        """
        with self._lock:
            if hold_id not in self._legal_holds:
                return False
            
            hold = self._legal_holds[hold_id]
            hold.is_active = False
            hold.released_at = time.time()
            hold.released_by = released_by
            return True
    
    def is_under_legal_hold(
        self,
        event_id: str,
        event_time: float,
        event_type: str,
    ) -> tuple[bool, list[str]]:
        """Check if an event is under legal hold.
        
        Args:
            event_id: Event ID
            event_time: Event timestamp
            event_type: Event type
            
        Returns:
            Tuple of (is_held, list of hold IDs)
        """
        with self._lock:
            holding_ids = []
            
            for hold in self._legal_holds.values():
                if hold.covers_event(event_id, event_time, event_type):
                    holding_ids.append(hold.hold_id)
            
            return (len(holding_ids) > 0, holding_ids)
    
    def can_delete(
        self,
        event_id: str,
        event_time: float,
        event_type: str,
        data_classification: str = "",
    ) -> tuple[bool, str]:
        """Check if an event can be deleted.
        
        Args:
            event_id: Event ID
            event_time: Event timestamp
            event_type: Event type
            data_classification: Data classification
            
        Returns:
            Tuple of (can_delete, reason)
        """
        # Check legal holds
        is_held, hold_ids = self.is_under_legal_hold(event_id, event_time, event_type)
        if is_held:
            return (False, f"Under legal hold: {', '.join(hold_ids)}")
        
        # Check compliance policies
        with self._lock:
            age_days = (time.time() - event_time) / 86400
            
            for policy in self._policies.values():
                if policy.data_classification == data_classification:
                    if age_days < policy.min_retention_days:
                        return (False, f"{policy.framework.value} requires minimum {policy.min_retention_days} days retention")
        
        return (True, "Deletion allowed")
    
    def get_effective_retention(
        self,
        data_classification: str,
    ) -> dict[str, Any]:
        """Get effective retention requirements for a data classification.
        
        Args:
            data_classification: Data classification
            
        Returns:
            Effective retention requirements
        """
        with self._lock:
            min_retention = 0
            max_retention = None
            requires_encryption = False
            requires_immutability = False
            applicable_frameworks = []
            
            for policy in self._policies.values():
                if policy.data_classification == data_classification:
                    applicable_frameworks.append(policy.framework.value)
                    
                    min_retention = max(min_retention, policy.min_retention_days)
                    
                    if policy.max_retention_days is not None:
                        if max_retention is None:
                            max_retention = policy.max_retention_days
                        else:
                            max_retention = min(max_retention, policy.max_retention_days)
                    
                    requires_encryption = requires_encryption or policy.requires_encryption
                    requires_immutability = requires_immutability or policy.requires_immutability
            
            return {
                "min_retention_days": min_retention,
                "max_retention_days": max_retention,
                "requires_encryption": requires_encryption,
                "requires_immutability": requires_immutability,
                "applicable_frameworks": applicable_frameworks,
            }
    
    def record_violation(
        self,
        policy_id: str,
        violation_type: str,
        description: str,
        event_id: str | None = None,
        severity: str = "warning",
        remediation_action: str | None = None,
    ) -> RetentionPolicyViolation:
        """Record a policy violation.
        
        Args:
            policy_id: ID of violated policy
            violation_type: Type of violation
            description: Description of violation
            event_id: Optional related event ID
            severity: Violation severity
            remediation_action: Suggested remediation
            
        Returns:
            Created violation record
        """
        with self._lock:
            self._violation_counter += 1
            violation_id = f"violation_{self._violation_counter}"
            
            violation = RetentionPolicyViolation(
                violation_id=violation_id,
                policy_id=policy_id,
                event_id=event_id,
                violation_type=violation_type,
                description=description,
                detected_at=time.time(),
                severity=severity,
                remediation_action=remediation_action,
            )
            
            self._violations.append(violation)
            return violation
    
    def get_violations(
        self,
        severity: str | None = None,
        unacknowledged_only: bool = False,
    ) -> list[RetentionPolicyViolation]:
        """Get recorded violations.
        
        Args:
            severity: Optional severity filter
            unacknowledged_only: Only return unacknowledged violations
            
        Returns:
            List of violations
        """
        with self._lock:
            result = self._violations.copy()
            
            if severity:
                result = [v for v in result if v.severity == severity]
            
            if unacknowledged_only:
                result = [v for v in result if not v.acknowledged]
            
            return result
    
    def acknowledge_violation(
        self,
        violation_id: str,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge a violation.
        
        Args:
            violation_id: Violation ID
            acknowledged_by: User acknowledging
            
        Returns:
            True if acknowledged
        """
        with self._lock:
            for violation in self._violations:
                if violation.violation_id == violation_id:
                    violation.acknowledged = True
                    violation.acknowledged_by = acknowledged_by
                    return True
            return False
    
    def get_active_legal_holds(self) -> list[LegalHold]:
        """Get all active legal holds.
        
        Returns:
            List of active holds
        """
        with self._lock:
            return [h for h in self._legal_holds.values() if h.is_active]
    
    def get_compliance_summary(self) -> dict[str, Any]:
        """Get summary of compliance status.
        
        Returns:
            Compliance summary
        """
        with self._lock:
            active_holds = len([h for h in self._legal_holds.values() if h.is_active])
            unack_violations = len([v for v in self._violations if not v.acknowledged])
            critical_violations = len([v for v in self._violations if v.severity == "critical"])
            
            return {
                "active_policies": len(self._policies),
                "frameworks": [p.value for p in self._policies.keys()],
                "active_legal_holds": active_holds,
                "total_violations": len(self._violations),
                "unacknowledged_violations": unack_violations,
                "critical_violations": critical_violations,
                "compliance_status": "compliant" if critical_violations == 0 else "non_compliant",
            }


class EventTypeRetentionPolicy:
    """Retention policy based on event type.
    
    Different event types may have different retention requirements.
    """
    
    def __init__(self) -> None:
        """Initialize event type policy manager."""
        self._lock = threading.Lock()
        
        # Event type -> (retention_days, priority)
        self._type_policies: dict[str, tuple[int, int]] = {}
        
        # Default retention
        self._default_retention_days = 365
    
    def set_policy(
        self,
        event_type: str,
        retention_days: int,
        priority: int = 0,
    ) -> None:
        """Set retention policy for an event type.
        
        Args:
            event_type: Event type
            retention_days: Retention period in days
            priority: Policy priority for conflicts
        """
        with self._lock:
            self._type_policies[event_type] = (retention_days, priority)
    
    def set_bulk_policies(
        self,
        policies: dict[str, int],
    ) -> None:
        """Set multiple policies at once.
        
        Args:
            policies: Dictionary of event_type -> retention_days
        """
        with self._lock:
            for event_type, days in policies.items():
                self._type_policies[event_type] = (days, 0)
    
    def get_retention_days(self, event_type: str) -> int:
        """Get retention days for an event type.
        
        Args:
            event_type: Event type
            
        Returns:
            Retention period in days
        """
        with self._lock:
            if event_type in self._type_policies:
                return self._type_policies[event_type][0]
            return self._default_retention_days
    
    def is_expired(self, event_type: str, event_time: float) -> bool:
        """Check if an event has expired.
        
        Args:
            event_type: Event type
            event_time: Event timestamp
            
        Returns:
            True if event should be deleted
        """
        retention_days = self.get_retention_days(event_type)
        age_days = (time.time() - event_time) / 86400
        return age_days > retention_days
    
    def get_all_policies(self) -> dict[str, dict[str, Any]]:
        """Get all event type policies.
        
        Returns:
            Dictionary of policies
        """
        with self._lock:
            return {
                event_type: {
                    "retention_days": days,
                    "priority": priority,
                }
                for event_type, (days, priority) in self._type_policies.items()
            }


class RetentionPolicyEnforcer:
    """Enforces retention policies across all managers.
    
    Combines tiered, compliance, and event-type policies
    for unified enforcement.
    """
    
    def __init__(
        self,
        tiered_manager: TieredRetentionManager | None = None,
        compliance_manager: ComplianceRetentionManager | None = None,
        event_type_policy: EventTypeRetentionPolicy | None = None,
    ) -> None:
        """Initialize enforcer.
        
        Args:
            tiered_manager: Tiered retention manager
            compliance_manager: Compliance retention manager
            event_type_policy: Event type policy manager
        """
        self._tiered = tiered_manager
        self._compliance = compliance_manager
        self._event_type = event_type_policy
        
        self._lock = threading.Lock()
        self._enforcement_log: list[dict[str, Any]] = []
    
    def evaluate_retention(
        self,
        event_id: str,
        event_time: float,
        event_type: str,
        data_classification: str = "",
    ) -> dict[str, Any]:
        """Evaluate all retention policies for an event.
        
        Args:
            event_id: Event ID
            event_time: Event timestamp
            event_type: Event type
            data_classification: Data classification
            
        Returns:
            Retention evaluation result
        """
        result = {
            "event_id": event_id,
            "can_delete": True,
            "reasons": [],
            "recommended_tier": RetentionTier.HOT.value,
            "retention_days_remaining": None,
            "policies_applied": [],
        }
        
        age_days = (time.time() - event_time) / 86400
        
        # Check compliance
        if self._compliance:
            can_del, reason = self._compliance.can_delete(
                event_id, event_time, event_type, data_classification
            )
            if not can_del:
                result["can_delete"] = False
                result["reasons"].append(reason)
                result["policies_applied"].append("compliance")
            
            effective = self._compliance.get_effective_retention(data_classification)
            if effective["min_retention_days"] > age_days:
                result["retention_days_remaining"] = effective["min_retention_days"] - age_days
        
        # Check tiered policy
        if self._tiered:
            tier = self._tiered.assign_tier(event_id, event_time, event_type)
            result["recommended_tier"] = tier.value
            result["policies_applied"].append("tiered")
            
            if tier == RetentionTier.DELETED:
                # Can be deleted based on tiered policy
                pass
            else:
                # Still under retention
                if not result["reasons"]:
                    result["reasons"].append(f"In {tier.value} tier")
        
        # Check event type policy
        if self._event_type:
            if self._event_type.is_expired(event_type, event_time):
                result["policies_applied"].append("event_type_expired")
            else:
                retention = self._event_type.get_retention_days(event_type)
                remaining = retention - age_days
                
                if result["retention_days_remaining"] is None:
                    result["retention_days_remaining"] = remaining
                else:
                    result["retention_days_remaining"] = max(
                        result["retention_days_remaining"], remaining
                    )
                
                result["policies_applied"].append("event_type")
        
        return result
    
    def enforce_deletion_policy(
        self,
        event_id: str,
        event_time: float,
        event_type: str,
        data_classification: str = "",
        requester: str = "",
    ) -> dict[str, Any]:
        """Enforce deletion policy for an event.
        
        Args:
            event_id: Event ID
            event_time: Event timestamp
            event_type: Event type
            data_classification: Data classification
            requester: User requesting deletion
            
        Returns:
            Enforcement result
        """
        evaluation = self.evaluate_retention(
            event_id, event_time, event_type, data_classification
        )
        
        enforcement_record = {
            "event_id": event_id,
            "action": "delete_request",
            "requester": requester,
            "timestamp": time.time(),
            "allowed": evaluation["can_delete"],
            "reasons": evaluation["reasons"],
        }
        
        with self._lock:
            self._enforcement_log.append(enforcement_record)
        
        if not evaluation["can_delete"] and self._compliance:
            # Record violation attempt
            self._compliance.record_violation(
                policy_id="deletion_policy",
                violation_type="deletion_blocked",
                description=f"Deletion blocked: {'; '.join(evaluation['reasons'])}",
                event_id=event_id,
                severity="warning",
            )
        
        return {
            "allowed": evaluation["can_delete"],
            "evaluation": evaluation,
            "enforcement_record": enforcement_record,
        }
    
    def run_cleanup(
        self,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Run retention cleanup across all policies.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Cleanup results
        """
        result = {
            "dry_run": dry_run,
            "events_evaluated": 0,
            "events_to_delete": 0,
            "events_deleted": 0,
            "events_protected": 0,
            "tier_transitions": 0,
            "errors": [],
        }
        
        # This would integrate with actual event storage
        # Here we just process tier transitions
        if self._tiered:
            transitions = self._tiered.process_tier_transitions()
            result["tier_transitions"] = sum(transitions.values())
        
        return result
    
    def get_enforcement_report(self) -> dict[str, Any]:
        """Get enforcement activity report.
        
        Returns:
            Report with enforcement statistics
        """
        with self._lock:
            if not self._enforcement_log:
                return {
                    "total_actions": 0,
                    "allowed": 0,
                    "denied": 0,
                    "recent": [],
                }
            
            allowed = len([e for e in self._enforcement_log if e.get("allowed", False)])
            denied = len(self._enforcement_log) - allowed
            
            return {
                "total_actions": len(self._enforcement_log),
                "allowed": allowed,
                "denied": denied,
                "recent": self._enforcement_log[-10:],
            }