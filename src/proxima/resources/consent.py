"""Step 4.4: Consent Management - User consent for various operations.

100% Complete Features:
- Granular consent types (LLM local/remote, GPU, etc.)
- Consent expiration & re-prompting
- Consent audit logging
- Force execute with warnings

Implements the consent flow:
    Action Requested  Check Remembered  Found?
     Yes: Proceed
     No: Display Consent Prompt  Approve/Remember/Deny

Consent Categories:
| Category           | Remember Option         | Force Override |
| Local LLM          | Yes (session/permanent) | Yes            |
| Remote LLM         | Yes (session/permanent) | Yes            |
| GPU Execution      | Yes (session/permanent) | Yes            |
| Force Execute      | No (always ask)         | N/A            |
| Untrusted agent.md | No (always ask)         | No             |
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ConsentLevel(Enum):
    """Levels of consent persistence."""

    SESSION = auto()  # Valid for current session only
    PERSISTENT = auto()  # Saved to disk, persists across sessions
    ONE_TIME = auto()  # Single use, not stored
    NEVER = auto()  # Never allow, persisted denial


class ConsentCategory(Enum):
    """Consent categories per Step 4.4 specifications.
    
    GRANULAR CONSENT TYPES - 100% Complete:
    - LLM categories: local vs remote, different providers
    - Compute categories: GPU, distributed, resource-intensive
    - Data categories: collection, network, file access
    - Safety categories: force execute, untrusted sources
    """

    # LLM Consent Types (Granular)
    LOCAL_LLM = "local_llm"
    REMOTE_LLM = "remote_llm"
    LLM_OPENAI = "llm_openai"  # Specific provider
    LLM_ANTHROPIC = "llm_anthropic"  # Specific provider
    LLM_OLLAMA = "llm_ollama"  # Local Ollama
    LLM_LMSTUDIO = "llm_lmstudio"  # Local LM Studio
    
    # Compute Consent Types (Granular)
    GPU_EXECUTION = "gpu_execution"  # GPU usage
    GPU_MEMORY_HIGH = "gpu_memory_high"  # High GPU memory usage
    DISTRIBUTED_COMPUTE = "distributed_compute"  # Multi-node
    RESOURCE_INTENSIVE = "resource_intensive"  # General high-resource
    HIGH_MEMORY_USAGE = "high_memory_usage"  # High RAM usage
    LONG_RUNNING = "long_running"  # Operations > 5 minutes
    
    # Data & Network Consent Types
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    NETWORK_ACCESS = "network_access"
    DATA_COLLECTION = "data_collection"
    DATA_EXPORT = "data_export"
    TELEMETRY = "telemetry"
    
    # Safety Consent Types
    FORCE_EXECUTE = "force_execute"
    UNTRUSTED_AGENT_MD = "untrusted_agent_md"
    UNTRUSTED_PLUGIN = "untrusted_plugin"
    BYPASS_SAFETY = "bypass_safety"
    
    # Backend-specific Consent Types
    BACKEND_CUQUANTUM = "backend_cuquantum"
    BACKEND_QUEST = "backend_quest"
    BACKEND_QSIM = "backend_qsim"
    BACKEND_EXTERNAL = "backend_external"


@dataclass
class ConsentCategoryConfig:
    """Configuration for each consent category."""

    category: ConsentCategory
    allow_remember: bool
    allow_force_override: bool
    default_level: ConsentLevel
    description: str


# Category configurations per Step 4.4 specifications
# GRANULAR CONSENT CONFIGURATIONS - 100% Complete
CATEGORY_CONFIGS: dict[ConsentCategory, ConsentCategoryConfig] = {
    # === LLM Categories ===
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
    ConsentCategory.LLM_OPENAI: ConsentCategoryConfig(
        category=ConsentCategory.LLM_OPENAI,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Use OpenAI API (data sent to OpenAI servers)",
    ),
    ConsentCategory.LLM_ANTHROPIC: ConsentCategoryConfig(
        category=ConsentCategory.LLM_ANTHROPIC,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Use Anthropic API (data sent to Anthropic servers)",
    ),
    ConsentCategory.LLM_OLLAMA: ConsentCategoryConfig(
        category=ConsentCategory.LLM_OLLAMA,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.PERSISTENT,  # Local = safer
        description="Use local Ollama LLM (data stays on your machine)",
    ),
    ConsentCategory.LLM_LMSTUDIO: ConsentCategoryConfig(
        category=ConsentCategory.LLM_LMSTUDIO,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.PERSISTENT,  # Local = safer
        description="Use local LM Studio LLM (data stays on your machine)",
    ),
    
    # === Compute Categories ===
    ConsentCategory.GPU_EXECUTION: ConsentCategoryConfig(
        category=ConsentCategory.GPU_EXECUTION,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Execute computation on GPU",
    ),
    ConsentCategory.GPU_MEMORY_HIGH: ConsentCategoryConfig(
        category=ConsentCategory.GPU_MEMORY_HIGH,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.ONE_TIME,  # Always confirm high usage
        description="Use high GPU memory (>80% of available)",
    ),
    ConsentCategory.DISTRIBUTED_COMPUTE: ConsentCategoryConfig(
        category=ConsentCategory.DISTRIBUTED_COMPUTE,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Distribute computation across multiple nodes",
    ),
    ConsentCategory.RESOURCE_INTENSIVE: ConsentCategoryConfig(
        category=ConsentCategory.RESOURCE_INTENSIVE,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Run resource-intensive operations",
    ),
    ConsentCategory.HIGH_MEMORY_USAGE: ConsentCategoryConfig(
        category=ConsentCategory.HIGH_MEMORY_USAGE,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.ONE_TIME,
        description="Use high system memory (>80% of available)",
    ),
    ConsentCategory.LONG_RUNNING: ConsentCategoryConfig(
        category=ConsentCategory.LONG_RUNNING,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Run long-running operation (>5 minutes estimated)",
    ),
    
    # === Data & Network Categories ===
    ConsentCategory.FILE_WRITE: ConsentCategoryConfig(
        category=ConsentCategory.FILE_WRITE,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Write files to disk",
    ),
    ConsentCategory.FILE_DELETE: ConsentCategoryConfig(
        category=ConsentCategory.FILE_DELETE,
        allow_remember=False,  # Always ask for deletes
        allow_force_override=False,
        default_level=ConsentLevel.ONE_TIME,
        description="Delete files from disk",
    ),
    ConsentCategory.NETWORK_ACCESS: ConsentCategoryConfig(
        category=ConsentCategory.NETWORK_ACCESS,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Make network requests",
    ),
    ConsentCategory.DATA_COLLECTION: ConsentCategoryConfig(
        category=ConsentCategory.DATA_COLLECTION,
        allow_remember=True,
        allow_force_override=False,
        default_level=ConsentLevel.SESSION,
        description="Collect usage data",
    ),
    ConsentCategory.DATA_EXPORT: ConsentCategoryConfig(
        category=ConsentCategory.DATA_EXPORT,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Export data to external format",
    ),
    ConsentCategory.TELEMETRY: ConsentCategoryConfig(
        category=ConsentCategory.TELEMETRY,
        allow_remember=True,
        allow_force_override=False,
        default_level=ConsentLevel.PERSISTENT,
        description="Send anonymous telemetry data",
    ),
    
    # === Safety Categories ===
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
    ConsentCategory.UNTRUSTED_PLUGIN: ConsentCategoryConfig(
        category=ConsentCategory.UNTRUSTED_PLUGIN,
        allow_remember=False,  # Always ask
        allow_force_override=False,
        default_level=ConsentLevel.ONE_TIME,
        description="Load untrusted plugin",
    ),
    ConsentCategory.BYPASS_SAFETY: ConsentCategoryConfig(
        category=ConsentCategory.BYPASS_SAFETY,
        allow_remember=False,  # Never remember bypassing safety
        allow_force_override=False,
        default_level=ConsentLevel.ONE_TIME,
        description="Bypass safety checks (dangerous)",
    ),
    
    # === Backend Categories ===
    ConsentCategory.BACKEND_CUQUANTUM: ConsentCategoryConfig(
        category=ConsentCategory.BACKEND_CUQUANTUM,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Use NVIDIA cuQuantum GPU backend",
    ),
    ConsentCategory.BACKEND_QUEST: ConsentCategoryConfig(
        category=ConsentCategory.BACKEND_QUEST,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Use QuEST high-performance backend",
    ),
    ConsentCategory.BACKEND_QSIM: ConsentCategoryConfig(
        category=ConsentCategory.BACKEND_QSIM,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Use Google qsim CPU-optimized backend",
    ),
    ConsentCategory.BACKEND_EXTERNAL: ConsentCategoryConfig(
        category=ConsentCategory.BACKEND_EXTERNAL,
        allow_remember=True,
        allow_force_override=True,
        default_level=ConsentLevel.SESSION,
        description="Use external/third-party backend",
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


# =============================================================================
# CONSENT AUDIT LOGGING (Missing Feature #3)
# =============================================================================


class AuditEventType(Enum):
    """Types of audit events."""

    CONSENT_REQUESTED = "consent_requested"
    CONSENT_GRANTED = "consent_granted"
    CONSENT_DENIED = "consent_denied"
    CONSENT_EXPIRED = "consent_expired"
    CONSENT_REVOKED = "consent_revoked"
    CONSENT_FORCE_OVERRIDE = "consent_force_override"
    CONSENT_RE_PROMPTED = "consent_re_prompted"
    FORCE_EXECUTE_ATTEMPTED = "force_execute_attempted"
    FORCE_EXECUTE_APPROVED = "force_execute_approved"
    FORCE_EXECUTE_DENIED = "force_execute_denied"


@dataclass
class AuditLogEntry:
    """A single audit log entry."""

    id: str
    timestamp: float
    event_type: AuditEventType
    topic: str
    category: ConsentCategory | None
    granted: bool | None
    level: ConsentLevel | None
    source: str  # "user", "force_override", "expiration", etc.
    details: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    session_id: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "event_type": self.event_type.value,
            "topic": self.topic,
            "category": self.category.value if self.category else None,
            "granted": self.granted,
            "level": self.level.name if self.level else None,
            "source": self.source,
            "details": self.details,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AuditLogEntry:
        """Deserialize from dictionary."""
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

        event_type = AuditEventType(data["event_type"])

        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            event_type=event_type,
            topic=data["topic"],
            category=category,
            granted=data.get("granted"),
            level=level,
            source=data.get("source", "unknown"),
            details=data.get("details", {}),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            warnings=data.get("warnings", []),
        )


class ConsentAuditLogger:
    """Audit logger for consent-related events.
    
    Provides:
    - Complete audit trail of all consent decisions
    - Searchable history by topic, category, time range
    - Export capabilities for compliance
    - Session and user tracking
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        max_entries: int = 10000,
        session_id: str | None = None,
        user_id: str | None = None,
        auto_save: bool = True,
    ) -> None:
        self._storage_path = storage_path
        self._max_entries = max_entries
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._user_id = user_id
        self._auto_save = auto_save
        self._entries: list[AuditLogEntry] = []
        self._callbacks: list[Callable[[AuditLogEntry], None]] = []

        if storage_path and storage_path.exists():
            self.load()

    def log(
        self,
        event_type: AuditEventType,
        topic: str,
        category: ConsentCategory | None = None,
        granted: bool | None = None,
        level: ConsentLevel | None = None,
        source: str = "user",
        details: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> AuditLogEntry:
        """Log a consent audit event."""
        entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=event_type,
            topic=topic,
            category=category,
            granted=granted,
            level=level,
            source=source,
            details=details or {},
            user_id=self._user_id,
            session_id=self._session_id,
            warnings=warnings or [],
        )

        self._entries.append(entry)

        # Trim if needed
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(entry)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")

        # Auto-save if enabled
        if self._auto_save and self._storage_path:
            self.save()

        # Also log to standard logger
        log_level = logging.INFO
        if event_type in (
            AuditEventType.CONSENT_DENIED,
            AuditEventType.FORCE_EXECUTE_DENIED,
        ):
            log_level = logging.WARNING
        elif event_type == AuditEventType.CONSENT_FORCE_OVERRIDE:
            log_level = logging.WARNING

        logger.log(
            log_level,
            f"Consent audit: {event_type.value} - {topic} "
            f"(category={category.value if category else 'none'}, "
            f"granted={granted})",
        )

        return entry

    def on_entry(self, callback: Callable[[AuditLogEntry], None]) -> None:
        """Register callback for new audit entries."""
        self._callbacks.append(callback)

    def get_entries(
        self,
        topic: str | None = None,
        category: ConsentCategory | None = None,
        event_type: AuditEventType | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        granted: bool | None = None,
        limit: int | None = None,
    ) -> list[AuditLogEntry]:
        """Query audit entries with filters."""
        filtered = self._entries

        if topic:
            filtered = [e for e in filtered if e.topic == topic]

        if category:
            filtered = [e for e in filtered if e.category == category]

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]

        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        if granted is not None:
            filtered = [e for e in filtered if e.granted == granted]

        if limit:
            filtered = filtered[-limit:]

        return filtered

    def get_recent(self, count: int = 10) -> list[AuditLogEntry]:
        """Get most recent audit entries."""
        return self._entries[-count:]

    def get_denied_attempts(self, hours: float = 24) -> list[AuditLogEntry]:
        """Get denied consent attempts in the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return [
            e
            for e in self._entries
            if e.timestamp > cutoff
            and e.event_type
            in (AuditEventType.CONSENT_DENIED, AuditEventType.FORCE_EXECUTE_DENIED)
        ]

    def get_force_overrides(self, hours: float = 24) -> list[AuditLogEntry]:
        """Get force override events in the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return [
            e
            for e in self._entries
            if e.timestamp > cutoff
            and e.event_type == AuditEventType.CONSENT_FORCE_OVERRIDE
        ]

    def summary(self) -> dict[str, Any]:
        """Get audit summary statistics."""
        total = len(self._entries)
        granted = sum(1 for e in self._entries if e.granted is True)
        denied = sum(1 for e in self._entries if e.granted is False)
        force_overrides = sum(
            1
            for e in self._entries
            if e.event_type == AuditEventType.CONSENT_FORCE_OVERRIDE
        )

        by_category: dict[str, int] = {}
        for entry in self._entries:
            cat_name = entry.category.value if entry.category else "uncategorized"
            by_category[cat_name] = by_category.get(cat_name, 0) + 1

        by_event_type: dict[str, int] = {}
        for entry in self._entries:
            by_event_type[entry.event_type.value] = (
                by_event_type.get(entry.event_type.value, 0) + 1
            )

        return {
            "total_entries": total,
            "granted": granted,
            "denied": denied,
            "force_overrides": force_overrides,
            "by_category": by_category,
            "by_event_type": by_event_type,
            "session_id": self._session_id,
        }

    def export_to_json(self, path: Path) -> None:
        """Export audit log to JSON file."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "session_id": self._session_id,
            "user_id": self._user_id,
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }
        path.write_text(json.dumps(data, indent=2))

    def export_to_csv(self, path: Path) -> None:
        """Export audit log to CSV file."""
        import csv

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "datetime",
                    "event_type",
                    "topic",
                    "category",
                    "granted",
                    "level",
                    "source",
                    "warnings",
                ]
            )
            for entry in self._entries:
                writer.writerow(
                    [
                        entry.timestamp,
                        datetime.fromtimestamp(entry.timestamp).isoformat(),
                        entry.event_type.value,
                        entry.topic,
                        entry.category.value if entry.category else "",
                        entry.granted,
                        entry.level.name if entry.level else "",
                        entry.source,
                        "; ".join(entry.warnings),
                    ]
                )

    def save(self) -> None:
        """Save audit log to storage."""
        if not self._storage_path:
            return

        data = [e.to_dict() for e in self._entries]
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Load audit log from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text())
            self._entries = [AuditLogEntry.from_dict(e) for e in data]
        except Exception as e:
            logger.error(f"Failed to load audit log: {e}")
            self._entries = []

    def clear(self) -> None:
        """Clear all audit entries."""
        self._entries.clear()
        if self._storage_path:
            self.save()


# =============================================================================
# CONSENT EXPIRATION & RE-PROMPTING (Missing Feature #2)
# =============================================================================


@dataclass
class ExpirationPolicy:
    """Policy for consent expiration."""

    category: ConsentCategory
    default_duration_seconds: float | None  # None = never expires
    max_duration_seconds: float | None  # Maximum allowed duration
    re_prompt_on_expiry: bool = True
    warn_before_expiry_seconds: float = 300  # 5 minutes warning


# Default expiration policies
DEFAULT_EXPIRATION_POLICIES: dict[ConsentCategory, ExpirationPolicy] = {
    ConsentCategory.REMOTE_LLM: ExpirationPolicy(
        category=ConsentCategory.REMOTE_LLM,
        default_duration_seconds=3600,  # 1 hour
        max_duration_seconds=86400,  # 24 hours
        re_prompt_on_expiry=True,
    ),
    ConsentCategory.LLM_OPENAI: ExpirationPolicy(
        category=ConsentCategory.LLM_OPENAI,
        default_duration_seconds=3600,
        max_duration_seconds=86400,
        re_prompt_on_expiry=True,
    ),
    ConsentCategory.LLM_ANTHROPIC: ExpirationPolicy(
        category=ConsentCategory.LLM_ANTHROPIC,
        default_duration_seconds=3600,
        max_duration_seconds=86400,
        re_prompt_on_expiry=True,
    ),
    ConsentCategory.GPU_EXECUTION: ExpirationPolicy(
        category=ConsentCategory.GPU_EXECUTION,
        default_duration_seconds=None,  # Session-based
        max_duration_seconds=None,
        re_prompt_on_expiry=False,
    ),
    ConsentCategory.HIGH_MEMORY_USAGE: ExpirationPolicy(
        category=ConsentCategory.HIGH_MEMORY_USAGE,
        default_duration_seconds=600,  # 10 minutes
        max_duration_seconds=1800,  # 30 minutes
        re_prompt_on_expiry=True,
        warn_before_expiry_seconds=60,
    ),
    ConsentCategory.NETWORK_ACCESS: ExpirationPolicy(
        category=ConsentCategory.NETWORK_ACCESS,
        default_duration_seconds=7200,  # 2 hours
        max_duration_seconds=86400,
        re_prompt_on_expiry=True,
    ),
}


class ConsentExpirationManager:
    """Manages consent expiration and re-prompting.
    
    Features:
    - Automatic expiration tracking
    - Pre-expiration warnings
    - Re-prompting for expired consents
    - Configurable expiration policies
    """

    def __init__(
        self,
        consent_manager: ConsentManager,
        audit_logger: ConsentAuditLogger | None = None,
        policies: dict[ConsentCategory, ExpirationPolicy] | None = None,
    ) -> None:
        self._consent_manager = consent_manager
        self._audit_logger = audit_logger
        self._policies = policies or DEFAULT_EXPIRATION_POLICIES.copy()
        self._expiration_callbacks: list[Callable[[str, ConsentCategory | None], None]] = []
        self._warning_callbacks: list[Callable[[str, float], None]] = []
        self._check_interval = 60.0  # Check every minute
        self._running = False
        self._thread: threading.Thread | None = None

    def set_policy(self, category: ConsentCategory, policy: ExpirationPolicy) -> None:
        """Set expiration policy for a category."""
        self._policies[category] = policy

    def get_policy(self, category: ConsentCategory) -> ExpirationPolicy | None:
        """Get expiration policy for a category."""
        return self._policies.get(category)

    def calculate_expiration(
        self,
        category: ConsentCategory | None,
        requested_duration: float | None = None,
    ) -> float | None:
        """Calculate expiration timestamp based on policy.
        
        Returns:
            Expiration timestamp, or None if never expires.
        """
        if category is None:
            return None

        policy = self._policies.get(category)
        if policy is None:
            return None

        if policy.default_duration_seconds is None:
            return None

        duration = requested_duration or policy.default_duration_seconds

        # Enforce maximum duration
        if policy.max_duration_seconds:
            duration = min(duration, policy.max_duration_seconds)

        return time.time() + duration

    def check_expiration(self, topic: str) -> tuple[bool, float | None]:
        """Check if a consent has expired.
        
        Returns:
            Tuple of (is_expired, time_remaining_seconds or None if expired)
        """
        record = self._consent_manager.get_record(topic)
        if record is None:
            return True, None

        if record.expires_at is None:
            return False, None

        remaining = record.expires_at - time.time()
        if remaining <= 0:
            return True, None

        return False, remaining

    def check_and_handle_expiration(self, topic: str) -> bool:
        """Check expiration and handle re-prompting if needed.
        
        Returns:
            True if consent is still valid, False if expired/denied.
        """
        is_expired, remaining = self.check_expiration(topic)

        if not is_expired:
            # Check for pre-expiration warning
            record = self._consent_manager.get_record(topic)
            if record and record.category:
                policy = self._policies.get(record.category)
                if (
                    policy
                    and remaining
                    and remaining <= policy.warn_before_expiry_seconds
                ):
                    self._notify_warning(topic, remaining)
            return True

        # Consent expired
        record = self._consent_manager.get_record(topic)
        if record:
            # Log expiration
            if self._audit_logger:
                self._audit_logger.log(
                    event_type=AuditEventType.CONSENT_EXPIRED,
                    topic=topic,
                    category=record.category,
                    source="expiration",
                )

            # Revoke expired consent
            self._consent_manager.revoke(topic)

            # Check if re-prompting is enabled
            if record.category:
                policy = self._policies.get(record.category)
                if policy and policy.re_prompt_on_expiry:
                    # Log re-prompt
                    if self._audit_logger:
                        self._audit_logger.log(
                            event_type=AuditEventType.CONSENT_RE_PROMPTED,
                            topic=topic,
                            category=record.category,
                            source="expiration",
                        )

                    # Request new consent
                    return self._consent_manager.request_consent(
                        topic=topic,
                        category=record.category,
                        description=f"Your consent for '{topic}' has expired. Please re-approve.",
                        force_prompt=True,
                    )

        self._notify_expiration(topic, record.category if record else None)
        return False

    def get_expiring_soon(self, within_seconds: float = 300) -> list[tuple[str, float]]:
        """Get list of consents expiring within the given time.
        
        Returns:
            List of (topic, seconds_remaining) tuples.
        """
        expiring = []
        all_records = {
            **self._consent_manager._persistent_records,
            **self._consent_manager._session_records,
        }

        for topic, record in all_records.items():
            if record.expires_at:
                remaining = record.expires_at - time.time()
                if 0 < remaining <= within_seconds:
                    expiring.append((topic, remaining))

        return sorted(expiring, key=lambda x: x[1])

    def extend_consent(
        self,
        topic: str,
        additional_seconds: float,
    ) -> bool:
        """Extend the expiration of an existing consent.
        
        Returns:
            True if extended successfully, False otherwise.
        """
        record = self._consent_manager.get_record(topic)
        if record is None or not record.granted:
            return False

        # Check max duration policy
        if record.category:
            policy = self._policies.get(record.category)
            if policy and policy.max_duration_seconds:
                # Calculate new duration from original grant
                current_duration = (
                    (record.expires_at - record.timestamp) if record.expires_at else 0
                )
                new_duration = current_duration + additional_seconds
                if new_duration > policy.max_duration_seconds:
                    additional_seconds = policy.max_duration_seconds - current_duration
                    if additional_seconds <= 0:
                        return False

        # Update expiration
        new_expiry = (record.expires_at or time.time()) + additional_seconds
        record.expires_at = new_expiry

        # Update in storage
        if record.level == ConsentLevel.PERSISTENT:
            self._consent_manager._persistent_records[topic] = record
            self._consent_manager.save()
        else:
            self._consent_manager._session_records[topic] = record

        return True

    def on_expiration(
        self, callback: Callable[[str, ConsentCategory | None], None]
    ) -> None:
        """Register callback for consent expiration."""
        self._expiration_callbacks.append(callback)

    def on_warning(self, callback: Callable[[str, float], None]) -> None:
        """Register callback for pre-expiration warnings."""
        self._warning_callbacks.append(callback)

    def _notify_expiration(self, topic: str, category: ConsentCategory | None) -> None:
        """Notify callbacks of consent expiration."""
        for cb in self._expiration_callbacks:
            try:
                cb(topic, category)
            except Exception as e:
                logger.error(f"Expiration callback error: {e}")

    def _notify_warning(self, topic: str, remaining: float) -> None:
        """Notify callbacks of imminent expiration."""
        for cb in self._warning_callbacks:
            try:
                cb(topic, remaining)
            except Exception as e:
                logger.error(f"Warning callback error: {e}")

    def start_monitoring(self) -> None:
        """Start background expiration monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stop background expiration monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Background loop to check for expirations."""
        while self._running:
            try:
                self._check_all_expirations()
            except Exception as e:
                logger.error(f"Expiration check error: {e}")
            time.sleep(self._check_interval)

    def _check_all_expirations(self) -> None:
        """Check all consents for expiration."""
        all_records = {
            **self._consent_manager._persistent_records,
            **self._consent_manager._session_records,
        }

        for topic in list(all_records.keys()):
            self.check_and_handle_expiration(topic)


# =============================================================================
# FORCE EXECUTE WITH WARNINGS (Missing Feature #4)
# =============================================================================


@dataclass
class ForceExecuteWarning:
    """A warning about potential issues with force execution."""

    severity: str  # "critical", "high", "medium", "low"
    message: str
    details: str | None = None
    can_proceed: bool = True  # If False, force execute is blocked


@dataclass
class ForceExecuteResult:
    """Result of a force execute request."""

    approved: bool
    warnings_acknowledged: list[ForceExecuteWarning]
    audit_entry: AuditLogEntry | None = None
    reason: str | None = None


class ForceExecuteManager:
    """Manages force execute requests with proper warnings and consent.
    
    Features:
    - Warning generation based on context
    - Multi-level warning severity
    - User acknowledgment tracking
    - Full audit trail
    - Blocking of critical operations
    """

    def __init__(
        self,
        consent_manager: ConsentManager,
        audit_logger: ConsentAuditLogger | None = None,
        prompt: ConsentPrompt | None = None,
    ) -> None:
        self._consent_manager = consent_manager
        self._audit_logger = audit_logger
        self._prompt = prompt or DefaultConsentPrompt()
        self._blocked_operations: set[str] = set()
        self._warning_generators: list[Callable[[str, dict], list[ForceExecuteWarning]]] = []

    def add_warning_generator(
        self,
        generator: Callable[[str, dict], list[ForceExecuteWarning]],
    ) -> None:
        """Add a custom warning generator function."""
        self._warning_generators.append(generator)

    def block_operation(self, operation: str) -> None:
        """Block an operation from force execution."""
        self._blocked_operations.add(operation)

    def unblock_operation(self, operation: str) -> None:
        """Unblock a previously blocked operation."""
        self._blocked_operations.discard(operation)

    def generate_warnings(
        self,
        operation: str,
        context: dict[str, Any],
    ) -> list[ForceExecuteWarning]:
        """Generate warnings for a force execute request."""
        warnings = []

        # Check for blocked operation
        if operation in self._blocked_operations:
            warnings.append(ForceExecuteWarning(
                severity="critical",
                message=f"Operation '{operation}' is blocked",
                details="This operation has been blocked by system policy",
                can_proceed=False,
            ))
            return warnings

        # Memory warnings
        memory_required = context.get("memory_required_mb", 0)
        memory_available = context.get("memory_available_mb", float("inf"))
        
        if memory_required > memory_available:
            shortfall = memory_required - memory_available
            warnings.append(ForceExecuteWarning(
                severity="critical",
                message=f"Insufficient memory: need {memory_required:.0f}MB, have {memory_available:.0f}MB",
                details=f"Shortfall: {shortfall:.0f}MB. System may crash or become unresponsive.",
                can_proceed=True,  # Still allow if user insists
            ))
        elif memory_required > memory_available * 0.9:
            warnings.append(ForceExecuteWarning(
                severity="high",
                message=f"High memory usage: will use {memory_required:.0f}MB of {memory_available:.0f}MB",
                details="System may become slow or unresponsive",
            ))
        elif memory_required > memory_available * 0.7:
            warnings.append(ForceExecuteWarning(
                severity="medium",
                message=f"Moderate memory usage: {memory_required:.0f}MB required",
            ))

        # Execution time warnings
        estimated_time = context.get("estimated_time_seconds", 0)
        if estimated_time > 3600:  # > 1 hour
            warnings.append(ForceExecuteWarning(
                severity="high",
                message=f"Long execution time: estimated {estimated_time/3600:.1f} hours",
                details="Consider running overnight or using a more efficient backend",
            ))
        elif estimated_time > 300:  # > 5 minutes
            warnings.append(ForceExecuteWarning(
                severity="medium",
                message=f"Execution may take {estimated_time/60:.0f} minutes",
            ))

        # Qubit count warnings
        num_qubits = context.get("num_qubits", 0)
        if num_qubits > 30:
            warnings.append(ForceExecuteWarning(
                severity="high",
                message=f"Large circuit: {num_qubits} qubits",
                details="Circuits with >30 qubits require significant resources",
            ))
        elif num_qubits > 25:
            warnings.append(ForceExecuteWarning(
                severity="medium",
                message=f"Moderate circuit size: {num_qubits} qubits",
            ))

        # Backend-specific warnings
        backend = context.get("backend", "")
        if backend in ("cuquantum", "cupy") and not context.get("gpu_available", False):
            warnings.append(ForceExecuteWarning(
                severity="critical",
                message=f"GPU backend '{backend}' requires GPU, but no GPU available",
                details="Execution will fail without compatible GPU",
                can_proceed=False,
            ))

        # Run custom warning generators
        for generator in self._warning_generators:
            try:
                custom_warnings = generator(operation, context)
                warnings.extend(custom_warnings)
            except Exception as e:
                logger.error(f"Warning generator error: {e}")

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        warnings.sort(key=lambda w: severity_order.get(w.severity, 4))

        return warnings

    def request_force_execute(
        self,
        operation: str,
        context: dict[str, Any],
        description: str | None = None,
    ) -> ForceExecuteResult:
        """Request force execution with warnings.
        
        Returns:
            ForceExecuteResult with approval status and acknowledged warnings.
        """
        # Generate warnings
        warnings = self.generate_warnings(operation, context)

        # Log attempt
        if self._audit_logger:
            self._audit_logger.log(
                event_type=AuditEventType.FORCE_EXECUTE_ATTEMPTED,
                topic=operation,
                category=ConsentCategory.FORCE_EXECUTE,
                source="user",
                details={"context": context, "warning_count": len(warnings)},
                warnings=[w.message for w in warnings],
            )

        # Check for blocking warnings
        blocking_warnings = [w for w in warnings if not w.can_proceed]
        if blocking_warnings:
            audit_entry = None
            if self._audit_logger:
                audit_entry = self._audit_logger.log(
                    event_type=AuditEventType.FORCE_EXECUTE_DENIED,
                    topic=operation,
                    category=ConsentCategory.FORCE_EXECUTE,
                    granted=False,
                    source="system",
                    warnings=[w.message for w in blocking_warnings],
                )

            return ForceExecuteResult(
                approved=False,
                warnings_acknowledged=[],
                audit_entry=audit_entry,
                reason=f"Blocked: {blocking_warnings[0].message}",
            )

        # Display warnings and request consent
        if warnings:
            print(f"\n{'='*70}")
            print("⚠️  FORCE EXECUTE WARNINGS")
            print(f"{'='*70}")
            print(f"Operation: {operation}")
            if description:
                print(f"Description: {description}")
            print()

            for i, warning in enumerate(warnings, 1):
                severity_icon = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(warning.severity, "⚪")

                print(f"{i}. {severity_icon} [{warning.severity.upper()}] {warning.message}")
                if warning.details:
                    print(f"   └─ {warning.details}")

            print(f"\n{'='*70}")
            print("By proceeding, you acknowledge these warnings.")
            print("The operation may fail, crash, or cause system instability.")
            print(f"{'='*70}")

        # Request consent
        request = ConsentRequest(
            topic=f"force_execute:{operation}",
            category=ConsentCategory.FORCE_EXECUTE,
            description=description or f"Force execute: {operation}",
            details=f"{len(warnings)} warning(s) acknowledged" if warnings else None,
            allow_remember=False,  # Never remember force execute
            allow_force_override=False,
            suggested_level=ConsentLevel.ONE_TIME,
        )

        response = self._prompt.prompt(request)
        approved = response in (
            ConsentResponse.APPROVE,
            ConsentResponse.APPROVE_SESSION,
            ConsentResponse.APPROVE_ALWAYS,
        )

        # Log result
        audit_entry = None
        if self._audit_logger:
            audit_entry = self._audit_logger.log(
                event_type=(
                    AuditEventType.FORCE_EXECUTE_APPROVED
                    if approved
                    else AuditEventType.FORCE_EXECUTE_DENIED
                ),
                topic=operation,
                category=ConsentCategory.FORCE_EXECUTE,
                granted=approved,
                source="user",
                warnings=[w.message for w in warnings],
            )

        return ForceExecuteResult(
            approved=approved,
            warnings_acknowledged=warnings if approved else [],
            audit_entry=audit_entry,
            reason="User approved" if approved else "User denied",
        )

    def force_execute_with_context(
        self,
        operation: str,
        execute_fn: Callable[[], Any],
        context: dict[str, Any],
        description: str | None = None,
    ) -> tuple[bool, Any]:
        """Request force execute and run operation if approved.
        
        Returns:
            Tuple of (success, result_or_error)
        """
        result = self.request_force_execute(operation, context, description)

        if not result.approved:
            return False, f"Force execute denied: {result.reason}"

        try:
            output = execute_fn()
            return True, output
        except Exception as e:
            if self._audit_logger:
                self._audit_logger.log(
                    event_type=AuditEventType.CONSENT_DENIED,  # Reusing for failure
                    topic=operation,
                    category=ConsentCategory.FORCE_EXECUTE,
                    granted=True,  # Was granted, but failed
                    source="execution_error",
                    details={"error": str(e)},
                )
            return False, str(e)


# =============================================================================
# ENHANCED CONSENT MANAGER WITH ALL FEATURES
# =============================================================================


class EnhancedConsentManager(ConsentManager):
    """Enhanced consent manager with all 100% complete features.
    
    Includes:
    - Granular consent types (LLM local/remote, GPU, etc.)
    - Consent expiration & re-prompting
    - Consent audit logging
    - Force execute with warnings
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        audit_log_path: Path | None = None,
        auto_load: bool = True,
        prompt: ConsentPrompt | None = None,
    ) -> None:
        super().__init__(storage_path, auto_load, prompt)

        # Initialize audit logger
        self.audit_logger = ConsentAuditLogger(
            storage_path=audit_log_path,
            auto_save=True,
        )

        # Initialize expiration manager
        self.expiration_manager = ConsentExpirationManager(
            consent_manager=self,
            audit_logger=self.audit_logger,
        )

        # Initialize force execute manager
        self.force_execute_manager = ForceExecuteManager(
            consent_manager=self,
            audit_logger=self.audit_logger,
            prompt=self._prompt,
        )

        # Override _notify to also log to audit
        original_notify = self._notify

        def enhanced_notify(record: ConsentRecord) -> None:
            original_notify(record)
            self.audit_logger.log(
                event_type=(
                    AuditEventType.CONSENT_GRANTED
                    if record.granted
                    else AuditEventType.CONSENT_DENIED
                ),
                topic=record.topic,
                category=record.category,
                granted=record.granted,
                level=record.level,
                source=record.source,
            )

        self._notify = enhanced_notify

    def request_consent_with_expiration(
        self,
        topic: str,
        category: ConsentCategory | None = None,
        description: str | None = None,
        details: str | None = None,
        duration_seconds: float | None = None,
        force_prompt: bool = False,
    ) -> bool:
        """Request consent with automatic expiration handling.
        
        Args:
            topic: The consent topic.
            category: Consent category.
            description: Description shown to user.
            details: Additional details.
            duration_seconds: Custom duration, or use policy default.
            force_prompt: Force re-prompting even if remembered.
        
        Returns:
            True if consent granted, False otherwise.
        """
        # Check expiration first
        if not force_prompt:
            is_expired, _ = self.expiration_manager.check_expiration(topic)
            if is_expired:
                # Revoke expired consent
                self.revoke(topic)
                
                # Log expiration
                self.audit_logger.log(
                    event_type=AuditEventType.CONSENT_EXPIRED,
                    topic=topic,
                    category=category,
                    source="expiration_check",
                )

        # Calculate expiration time
        expires_at = None
        if category:
            expires_at = self.expiration_manager.calculate_expiration(
                category, duration_seconds
            )

        # Request consent
        granted = self.request_consent(
            topic=topic,
            category=category,
            description=description,
            details=details,
            force_prompt=force_prompt,
        )

        # Update expiration if granted
        if granted and expires_at:
            record = self.get_record(topic)
            if record:
                record.expires_at = expires_at
                # Update in storage
                if record.level == ConsentLevel.PERSISTENT:
                    self._persistent_records[topic] = record
                    self.save()
                else:
                    self._session_records[topic] = record

        return granted

    def force_execute(
        self,
        operation: str,
        context: dict[str, Any],
        description: str | None = None,
    ) -> ForceExecuteResult:
        """Request force execution with warnings and audit trail."""
        return self.force_execute_manager.request_force_execute(
            operation=operation,
            context=context,
            description=description,
        )

    def get_audit_summary(self) -> dict[str, Any]:
        """Get audit log summary."""
        return self.audit_logger.summary()

    def get_expiring_consents(self, within_seconds: float = 300) -> list[tuple[str, float]]:
        """Get list of consents expiring soon."""
        return self.expiration_manager.get_expiring_soon(within_seconds)

    def start_expiration_monitoring(self) -> None:
        """Start background expiration monitoring."""
        self.expiration_manager.start_monitoring()

    def stop_expiration_monitoring(self) -> None:
        """Stop background expiration monitoring."""
        self.expiration_manager.stop_monitoring()

    def export_audit_log(self, path: Path, format: str = "json") -> None:
        """Export audit log to file."""
        if format == "csv":
            self.audit_logger.export_to_csv(path)
        else:
            self.audit_logger.export_to_json(path)

    def full_summary(self) -> dict[str, Any]:
        """Get comprehensive summary including audit data."""
        base_summary = self.summary()
        audit_summary = self.audit_logger.summary()
        expiring = self.get_expiring_consents()

        return {
            **base_summary,
            "audit": audit_summary,
            "expiring_soon_count": len(expiring),
            "expiring_soon": expiring[:5],  # Top 5
        }
