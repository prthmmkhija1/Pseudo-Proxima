"""Step 4.4: Consent Management - User consent for various operations.

Implements the consent flow:
    Action Requested  Check Remembered  Found?
     Yes: Proceed
     No: Display Consent Prompt  Approve/Remember/Deny

Consent Categories:
| Category           | Remember Option         | Force Override |
| Local LLM          | Yes (session/permanent) | Yes            |
| Remote LLM         | Yes (session/permanent) | Yes            |
| Force Execute      | No (always ask)         | N/A            |
| Untrusted agent.md | No (always ask)         | No             |
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol


class ConsentLevel(Enum):
    """Levels of consent persistence."""

    SESSION = auto()  # Valid for current session only
    PERSISTENT = auto()  # Saved to disk, persists across sessions
    ONE_TIME = auto()  # Single use, not stored
    NEVER = auto()  # Never allow, persisted denial


class ConsentCategory(Enum):
    """Consent categories per Step 4.4 specifications."""

    LOCAL_LLM = "local_llm"
    REMOTE_LLM = "remote_llm"
    FORCE_EXECUTE = "force_execute"
    UNTRUSTED_AGENT_MD = "untrusted_agent_md"
    FILE_WRITE = "file_write"
    NETWORK_ACCESS = "network_access"
    RESOURCE_INTENSIVE = "resource_intensive"
    DATA_COLLECTION = "data_collection"


@dataclass
class ConsentCategoryConfig:
    """Configuration for each consent category."""

    category: ConsentCategory
    allow_remember: bool
    allow_force_override: bool
    default_level: ConsentLevel
    description: str


# Category configurations per Step 4.4 specifications
CATEGORY_CONFIGS: dict[ConsentCategory, ConsentCategoryConfig] = {
    ConsentCategory.LOCAL_LLM: ConsentCategoryConfig(
        category=ConsentCategory.LOCAL_LLM,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Use local LLM for code analysis",
    ),
    ConsentCategory.REMOTE_LLM: ConsentCategoryConfig(
        category=ConsentCategory.REMOTE_LLM,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Send data to remote LLM API",
    ),
    ConsentCategory.FORCE_EXECUTE: ConsentCategoryConfig(
        category=ConsentCategory.FORCE_EXECUTE,
        allow_remember=False,  # Always ask
        allow_force_override=False,
        default_level=ConsentLevel.ONE_TIME,
        description="Force execution despite warnings",
    ),
    ConsentCategory.UNTRUSTED_AGENT_MD: ConsentCategoryConfig(
        category=ConsentCategory.UNTRUSTED_AGENT_MD,
        allow_remember=False,  # Always ask
        allow_force_override=False,
        default_level=ConsentLevel.ONE_TIME,
        description="Execute untrusted agent.md file",
    ),
    ConsentCategory.FILE_WRITE: ConsentCategoryConfig(
        category=ConsentCategory.FILE_WRITE,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Write files to disk",
    ),
    ConsentCategory.NETWORK_ACCESS: ConsentCategoryConfig(
        category=ConsentCategory.NETWORK_ACCESS,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Make network requests",
    ),
    ConsentCategory.RESOURCE_INTENSIVE: ConsentCategoryConfig(
        category=ConsentCategory.RESOURCE_INTENSIVE,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Run resource-intensive operations",
    ),
    ConsentCategory.DATA_COLLECTION: ConsentCategoryConfig(
        category=ConsentCategory.DATA_COLLECTION,
        allow_remember=True,
        allow_force_override=False,
        default_level=ConsentLevel.SESSION,
        description="Collect usage data",
    ),
}


class ConsentResponse(Enum):
    """Possible responses to a consent prompt."""

    APPROVE = "approve"  # Allow this time
    APPROVE_SESSION = "approve_session"  # Allow for session
    APPROVE_ALWAYS = "approve_always"  # Allow permanently
    DENY = "deny"  # Deny this time
    DENY_SESSION = "deny_session"  # Deny for session
    DENY_ALWAYS = "deny_always"  # Deny permanently


@dataclass
class ConsentRecord:
    """Record of a consent decision."""

    topic: str
    category: ConsentCategory | None
    granted: bool
    level: ConsentLevel
    timestamp: float = field(default_factory=time.time)
    context: str | None = None
    expires_at: float | None = None
    source: str = "user"  # user, force_override, default

    def is_valid(self) -> bool:
        """Check if consent is still valid (not expired)."""
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "topic": self.topic,
            "category": self.category.value if self.category else None,
            "granted": self.granted,
            "level": self.level.name,
            "timestamp": self.timestamp,
            "context": self.context,
            "expires_at": self.expires_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConsentRecord:
        """Deserialize from dictionary."""
        category = None
        if data.get("category"):
            try:
                category = ConsentCategory(data["category"])
            except ValueError:
                pass

        return cls(
            topic=data["topic"],
            category=category,
            granted=data["granted"],
            level=ConsentLevel[data["level"]],
            timestamp=data.get("timestamp", time.time()),
            context=data.get("context"),
            expires_at=data.get("expires_at"),
            source=data.get("source", "user"),
        )


@dataclass
class ConsentRequest:
    """Request for user consent."""

    topic: str
    category: ConsentCategory | None
    description: str
    details: str | None = None
    allow_remember: bool = True
    allow_force_override: bool = False
    suggested_level: ConsentLevel = ConsentLevel.SESSION


class ConsentPrompt(Protocol):
    """Protocol for consent prompt implementations."""

    def prompt(self, request: ConsentRequest) -> ConsentResponse:
        """Display consent prompt and get user response."""
        ...


class DefaultConsentPrompt:
    """Default console-based consent prompt."""

    def prompt(self, request: ConsentRequest) -> ConsentResponse:
        """Simple console prompt for consent."""
        print(f"\n{'='*60}")
        print(f"CONSENT REQUIRED: {request.description}")
        if request.details:
            print(f"Details: {request.details}")
        print(f"Topic: {request.topic}")
        if request.category:
            print(f"Category: {request.category.value}")
        print(f"{'='*60}")

        options = ["[y] Approve", "[n] Deny"]
        if request.allow_remember:
            options.extend(["[s] Approve for session", "[a] Approve always"])

        print(" | ".join(options))

        while True:
            try:
                choice = input("Your choice: ").strip().lower()
                if choice == "y":
                    return ConsentResponse.APPROVE
                elif choice == "n":
                    return ConsentResponse.DENY
                elif choice == "s" and request.allow_remember:
                    return ConsentResponse.APPROVE_SESSION
                elif choice == "a" and request.allow_remember:
                    return ConsentResponse.APPROVE_ALWAYS
                else:
                    print("Invalid choice. Please try again.")
            except (EOFError, KeyboardInterrupt):
                return ConsentResponse.DENY


@dataclass
class ConsentCheckResult:
    """Result of checking consent."""

    found: bool
    granted: bool | None
    record: ConsentRecord | None
    source: str  # "session", "persistent", "not_found"


class ConsentManager:
    """Manages user consent for various operations per Step 4.4.

    Implements the consent flow:
    1. Check if consent is remembered (session or persistent)
    2. If found and valid: use stored decision
    3. If not found: display consent prompt
    4. Handle response: Approve/Remember/Deny
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        auto_load: bool = True,
        prompt: ConsentPrompt | None = None,
    ) -> None:
        self._persistent_records: dict[str, ConsentRecord] = {}
        self._session_records: dict[str, ConsentRecord] = {}
        self._storage_path = storage_path
        self._prompt = prompt or DefaultConsentPrompt()
        self._callbacks: list[Callable[[ConsentRecord], None]] = []
        self._force_override_enabled: bool = False

        if auto_load and storage_path and storage_path.exists():
            self.load()

    # ========== Callback Management ==========

    def on_consent(self, callback: Callable[[ConsentRecord], None]) -> None:
        """Register callback for consent decisions."""
        self._callbacks.append(callback)

    def _notify(self, record: ConsentRecord) -> None:
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(record)
            except Exception:
                pass

    # ========== Force Override ==========

    def enable_force_override(self) -> None:
        """Enable force override mode (for privileged operations)."""
        self._force_override_enabled = True

    def disable_force_override(self) -> None:
        """Disable force override mode."""
        self._force_override_enabled = False

    def is_force_override_enabled(self) -> bool:
        """Check if force override is enabled."""
        return self._force_override_enabled

    # ========== Core Consent Flow ==========

    def check_remembered(self, topic: str) -> ConsentCheckResult:
        """Check if consent is already remembered (Step 1 of flow)."""
        # Check session records first
        if topic in self._session_records:
            record = self._session_records[topic]
            if record.is_valid():
                return ConsentCheckResult(
                    found=True, granted=record.granted, record=record, source="session"
                )
            del self._session_records[topic]

        # Check persistent records
        if topic in self._persistent_records:
            record = self._persistent_records[topic]
            if record.is_valid():
                return ConsentCheckResult(
                    found=True,
                    granted=record.granted,
                    record=record,
                    source="persistent",
                )
            del self._persistent_records[topic]
            self.save()

        return ConsentCheckResult(
            found=False, granted=None, record=None, source="not_found"
        )

    def request_consent(
        self,
        topic: str,
        category: ConsentCategory | None = None,
        description: str | None = None,
        details: str | None = None,
        force_prompt: bool = False,
    ) -> bool:
        """Request consent following the Step 4.4 flow.

        Flow:
        1. Check remembered consent
        2. If found and valid: proceed or return error
        3. If not found: display consent prompt
        4. Handle response and store if remember selected

        Returns:
            True if consent granted, False if denied
        """
        # Get category configuration
        config = CATEGORY_CONFIGS.get(category) if category else None

        # Check if this category allows remembering
        allow_remember = config.allow_remember if config else True
        allow_force = config.allow_force_override if config else False

        # Step 1: Check remembered consent (unless force_prompt)
        if not force_prompt and allow_remember:
            check_result = self.check_remembered(topic)
            if check_result.found:
                return check_result.granted or False

        # Check force override
        if self._force_override_enabled and allow_force:
            self._store_consent(
                topic=topic,
                category=category,
                granted=True,
                level=ConsentLevel.ONE_TIME,
                source="force_override",
            )
            return True

        # Step 2: Display consent prompt
        request = ConsentRequest(
            topic=topic,
            category=category,
            description=description
            or (config.description if config else f"Allow {topic}?"),
            details=details,
            allow_remember=allow_remember,
            allow_force_override=allow_force,
            suggested_level=config.default_level if config else ConsentLevel.SESSION,
        )

        response = self._prompt.prompt(request)

        # Step 3: Handle response
        return self._handle_response(topic, category, response)

    def _handle_response(
        self,
        topic: str,
        category: ConsentCategory | None,
        response: ConsentResponse,
    ) -> bool:
        """Handle user's consent response."""
        granted = response in (
            ConsentResponse.APPROVE,
            ConsentResponse.APPROVE_SESSION,
            ConsentResponse.APPROVE_ALWAYS,
        )

        # Determine storage level based on response
        if response == ConsentResponse.APPROVE:
            level = ConsentLevel.ONE_TIME
        elif response == ConsentResponse.APPROVE_SESSION:
            level = ConsentLevel.SESSION
        elif response == ConsentResponse.APPROVE_ALWAYS:
            level = ConsentLevel.PERSISTENT
        elif response == ConsentResponse.DENY:
            level = ConsentLevel.ONE_TIME
        elif response == ConsentResponse.DENY_SESSION:
            level = ConsentLevel.SESSION
        elif response == ConsentResponse.DENY_ALWAYS:
            level = ConsentLevel.NEVER
        else:
            level = ConsentLevel.ONE_TIME

        # Store the decision if not one-time
        if level != ConsentLevel.ONE_TIME:
            self._store_consent(topic, category, granted, level)

        return granted

    def _store_consent(
        self,
        topic: str,
        category: ConsentCategory | None,
        granted: bool,
        level: ConsentLevel,
        context: str | None = None,
        duration_seconds: float | None = None,
        source: str = "user",
    ) -> ConsentRecord:
        """Store a consent decision."""
        expires_at = None
        if duration_seconds:
            expires_at = time.time() + duration_seconds

        record = ConsentRecord(
            topic=topic,
            category=category,
            granted=granted,
            level=level,
            context=context,
            expires_at=expires_at,
            source=source,
        )

        if level in (ConsentLevel.PERSISTENT, ConsentLevel.NEVER):
            self._persistent_records[topic] = record
            self.save()
        elif level == ConsentLevel.SESSION:
            self._session_records[topic] = record

        self._notify(record)
        return record

    # ========== Convenience Methods ==========

    def grant(
        self,
        topic: str,
        level: ConsentLevel = ConsentLevel.SESSION,
        category: ConsentCategory | None = None,
        context: str | None = None,
        duration_seconds: float | None = None,
    ) -> ConsentRecord:
        """Programmatically grant consent for a topic."""
        return self._store_consent(
            topic=topic,
            category=category,
            granted=True,
            level=level,
            context=context,
            duration_seconds=duration_seconds,
        )

    def deny(
        self,
        topic: str,
        level: ConsentLevel = ConsentLevel.SESSION,
        category: ConsentCategory | None = None,
        context: str | None = None,
    ) -> ConsentRecord:
        """Programmatically deny consent for a topic."""
        return self._store_consent(
            topic=topic,
            category=category,
            granted=False,
            level=level,
            context=context,
        )

    def check(self, topic: str) -> bool | None:
        """Quick check for consent status. Returns None if not decided."""
        result = self.check_remembered(topic)
        return result.granted if result.found else None

    def require(
        self,
        topic: str,
        category: ConsentCategory | None = None,
        description: str | None = None,
    ) -> bool:
        """Require consent, prompting if needed. Alias for request_consent."""
        return self.request_consent(topic, category, description)

    # ========== Revocation ==========

    def revoke(self, topic: str) -> bool:
        """Revoke consent for a topic. Returns True if was granted."""
        was_granted = False

        if topic in self._session_records:
            was_granted = self._session_records[topic].granted
            del self._session_records[topic]

        if topic in self._persistent_records:
            was_granted = was_granted or self._persistent_records[topic].granted
            del self._persistent_records[topic]
            self.save()

        return was_granted

    def revoke_all(self) -> None:
        """Revoke all consents."""
        self._session_records.clear()
        self._persistent_records.clear()
        self.save()

    def revoke_category(self, category: ConsentCategory) -> int:
        """Revoke all consents for a specific category. Returns count revoked."""
        revoked = 0

        session_to_remove = [
            topic
            for topic, record in self._session_records.items()
            if record.category == category
        ]
        for topic in session_to_remove:
            del self._session_records[topic]
            revoked += 1

        persistent_to_remove = [
            topic
            for topic, record in self._persistent_records.items()
            if record.category == category
        ]
        for topic in persistent_to_remove:
            del self._persistent_records[topic]
            revoked += 1

        if persistent_to_remove:
            self.save()

        return revoked

    # ========== Query Methods ==========

    def list_granted(self) -> list[str]:
        """List all topics with granted consent."""
        granted = []
        for topic, record in {
            **self._persistent_records,
            **self._session_records,
        }.items():
            if record.is_valid() and record.granted:
                granted.append(topic)
        return granted

    def list_denied(self) -> list[str]:
        """List all topics with denied consent."""
        denied = []
        for topic, record in {
            **self._persistent_records,
            **self._session_records,
        }.items():
            if record.is_valid() and not record.granted:
                denied.append(topic)
        return denied

    def get_record(self, topic: str) -> ConsentRecord | None:
        """Get the consent record for a topic."""
        result = self.check_remembered(topic)
        return result.record

    def get_records_by_category(self, category: ConsentCategory) -> list[ConsentRecord]:
        """Get all consent records for a category."""
        records = []
        for record in list(self._session_records.values()) + list(
            self._persistent_records.values()
        ):
            if record.category == category and record.is_valid():
                records.append(record)
        return records

    # ========== Persistence ==========

    def save(self) -> None:
        """Save persistent records to storage."""
        if not self._storage_path:
            return

        data = {
            topic: record.to_dict()
            for topic, record in self._persistent_records.items()
        }

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Load persistent records from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text())
            self._persistent_records = {
                topic: ConsentRecord.from_dict(record_data)
                for topic, record_data in data.items()
            }
        except Exception:
            self._persistent_records = {}

    # ========== Summary & Stats ==========

    def summary(self) -> dict:
        """Return consent summary statistics."""
        all_records = {**self._persistent_records, **self._session_records}
        valid_records = {t: r for t, r in all_records.items() if r.is_valid()}

        by_category: dict[str, dict[str, int]] = {}
        for record in valid_records.values():
            cat_name = record.category.value if record.category else "uncategorized"
            if cat_name not in by_category:
                by_category[cat_name] = {"granted": 0, "denied": 0}
            if record.granted:
                by_category[cat_name]["granted"] += 1
            else:
                by_category[cat_name]["denied"] += 1

        return {
            "total_topics": len(valid_records),
            "granted": len([r for r in valid_records.values() if r.granted]),
            "denied": len([r for r in valid_records.values() if not r.granted]),
            "session_records": len(self._session_records),
            "persistent_records": len(self._persistent_records),
            "by_category": by_category,
            "topics": list(valid_records.keys()),
            "force_override_enabled": self._force_override_enabled,
        }


# ========== Consent Decorator ==========


def requires_consent(
    topic: str,
    category: ConsentCategory | None = None,
    description: str | None = None,
    manager: ConsentManager | None = None,
):
    """Decorator to require consent before executing a function."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            consent_manager = manager
            if consent_manager is None:
                # Try to get from first argument if it has consent_manager attr
                if args and hasattr(args[0], "consent_manager"):
                    consent_manager = args[0].consent_manager
                else:
                    # Create a temporary manager
                    consent_manager = ConsentManager()

            if not consent_manager.request_consent(topic, category, description):
                raise PermissionError(f"Consent denied for: {topic}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


class ConsentDeniedException(Exception):
    """Raised when consent is denied for an operation."""

    def __init__(self, topic: str, category: ConsentCategory | None = None):
        self.topic = topic
        self.category = category
        super().__init__(f"Consent denied for: {topic}")
