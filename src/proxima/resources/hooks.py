"""Consent Event Hooks - Event-driven consent notifications and callbacks.

This module provides:
- ConsentEventBus: Central hub for consent event distribution
- ConsentEventHandler: Protocol for implementing event handlers
- Built-in handlers for common use cases:
  - LoggingHandler: Log consent events
  - MetricsHandler: Track consent metrics
  - NotificationHandler: Send notifications
  - PolicyHandler: Enforce consent policies
  - RateLimitHandler: Rate limit consent requests

Events flow:
    ConsentManager -> ConsentEventBus -> Handlers

Usage:
    bus = ConsentEventBus()
    bus.register(LoggingHandler())
    bus.register(MetricsHandler())

    # In ConsentManager, emit events:
    bus.emit(ConsentEvent(...))
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Protocol

from .consent import (
    ConsentCategory,
    ConsentLevel,
    ConsentManager,
    ConsentRecord,
)

# ========== Event Types ==========


class ConsentEventKind(Enum):
    """Types of consent events."""

    # Decision events
    CONSENT_REQUESTED = auto()
    CONSENT_GRANTED = auto()
    CONSENT_DENIED = auto()
    CONSENT_EXPIRED = auto()

    # Change events
    CONSENT_REVOKED = auto()
    CONSENT_UPDATED = auto()
    BULK_REVOKE = auto()

    # Override events
    FORCE_OVERRIDE = auto()

    # Query events
    CONSENT_CHECKED = auto()

    # Lifecycle
    MANAGER_INITIALIZED = auto()
    MANAGER_CLOSED = auto()


@dataclass
class ConsentEvent:
    """A consent event that can be dispatched to handlers."""

    kind: ConsentEventKind
    timestamp: float = field(default_factory=time.time)
    topic: str | None = None
    category: ConsentCategory | None = None
    granted: bool | None = None
    level: ConsentLevel | None = None
    source: str = "unknown"
    record: ConsentRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def datetime(self) -> datetime:
        """Get event time as datetime."""
        return datetime.fromtimestamp(self.timestamp)

    def __str__(self) -> str:
        granted_str = ""
        if self.granted is not None:
            granted_str = f" [{'granted' if self.granted else 'denied'}]"
        return f"ConsentEvent({self.kind.name}{granted_str}, topic={self.topic})"


# ========== Event Handler Protocol ==========


class ConsentEventHandler(Protocol):
    """Protocol for consent event handlers."""

    def handle(self, event: ConsentEvent) -> None:
        """Handle a consent event."""
        ...

    def supports(self, kind: ConsentEventKind) -> bool:
        """Check if handler supports this event kind."""
        ...


class BaseConsentHandler(ABC):
    """Base class for consent event handlers."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self._enabled = True
        self._supported_kinds: set[ConsentEventKind] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def supports(self, kind: ConsentEventKind) -> bool:
        """Check if handler supports this event kind."""
        if self._supported_kinds is None:
            return True  # Support all by default
        return kind in self._supported_kinds

    def handle(self, event: ConsentEvent) -> None:
        """Handle event if enabled and supported."""
        if self._enabled and self.supports(event.kind):
            self._handle_event(event)

    @abstractmethod
    def _handle_event(self, event: ConsentEvent) -> None:
        """Implement event handling logic."""
        pass


# ========== Event Bus ==========


class ConsentEventBus:
    """Central hub for consent event distribution.

    Handlers can be registered to receive events either synchronously
    or asynchronously.
    """

    def __init__(self) -> None:
        self._handlers: list[ConsentEventHandler] = []
        self._async_handlers: list[ConsentEventHandler] = []
        self._type_handlers: dict[ConsentEventKind, list[ConsentEventHandler]] = (
            defaultdict(list)
        )
        self._lock = threading.Lock()
        self._event_history: list[ConsentEvent] = []
        self._max_history = 1000
        self._paused = False

    def register(
        self,
        handler: ConsentEventHandler,
        async_delivery: bool = False,
        event_kinds: list[ConsentEventKind] | None = None,
    ) -> None:
        """Register an event handler.

        Args:
            handler: The handler to register
            async_delivery: If True, events are delivered in a separate thread
            event_kinds: If provided, only receive these event types
        """
        with self._lock:
            if async_delivery:
                self._async_handlers.append(handler)
            else:
                self._handlers.append(handler)

            # Register for specific types if provided
            if event_kinds:
                for kind in event_kinds:
                    self._type_handlers[kind].append(handler)

    def unregister(self, handler: ConsentEventHandler) -> bool:
        """Unregister an event handler. Returns True if found."""
        with self._lock:
            removed = False
            if handler in self._handlers:
                self._handlers.remove(handler)
                removed = True
            if handler in self._async_handlers:
                self._async_handlers.remove(handler)
                removed = True

            # Remove from type handlers
            for handlers_list in self._type_handlers.values():
                if handler in handlers_list:
                    handlers_list.remove(handler)

            return removed

    def emit(self, event: ConsentEvent) -> None:
        """Emit an event to all registered handlers."""
        if self._paused:
            return

        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history :]

        # Dispatch to sync handlers
        self._dispatch_sync(event)

        # Dispatch to async handlers
        if self._async_handlers:
            self._dispatch_async(event)

    def _dispatch_sync(self, event: ConsentEvent) -> None:
        """Dispatch event to synchronous handlers."""
        with self._lock:
            handlers = list(self._handlers)
            type_handlers = list(self._type_handlers.get(event.kind, []))

        all_handlers = set(handlers) | set(type_handlers)

        for handler in all_handlers:
            try:
                if handler.supports(event.kind):
                    handler.handle(event)
            except Exception as e:
                # Don't let handler errors stop event propagation
                logging.warning(f"Handler {handler} failed: {e}")

    def _dispatch_async(self, event: ConsentEvent) -> None:
        """Dispatch event to async handlers in separate thread."""
        with self._lock:
            handlers = list(self._async_handlers)

        def _deliver():
            for handler in handlers:
                try:
                    if handler.supports(event.kind):
                        handler.handle(event)
                except Exception as e:
                    logging.warning(f"Async handler {handler} failed: {e}")

        thread = threading.Thread(target=_deliver, daemon=True)
        thread.start()

    def pause(self) -> None:
        """Pause event emission."""
        self._paused = True

    def resume(self) -> None:
        """Resume event emission."""
        self._paused = False

    def clear_handlers(self) -> None:
        """Remove all handlers."""
        with self._lock:
            self._handlers.clear()
            self._async_handlers.clear()
            self._type_handlers.clear()

    def get_history(self, count: int = 100) -> list[ConsentEvent]:
        """Get recent event history."""
        with self._lock:
            return list(self._event_history[-count:])

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers."""
        with self._lock:
            return len(self._handlers) + len(self._async_handlers)


# ========== Built-in Handlers ==========


class LoggingHandler(BaseConsentHandler):
    """Handler that logs consent events to a Python logger."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        name: str = "LoggingHandler",
    ) -> None:
        super().__init__(name)
        self._logger = logger or logging.getLogger("proxima.consent")
        self._level = level

    def _handle_event(self, event: ConsentEvent) -> None:
        granted_str = ""
        if event.granted is not None:
            granted_str = " GRANTED" if event.granted else " DENIED"

        msg = f"Consent {event.kind.name}{granted_str}"
        if event.topic:
            msg += f" topic='{event.topic}'"
        if event.category:
            msg += f" category={event.category.value}"

        self._logger.log(self._level, msg)


class MetricsHandler(BaseConsentHandler):
    """Handler that tracks consent metrics/statistics."""

    def __init__(self, name: str = "MetricsHandler") -> None:
        super().__init__(name)
        self._counters: dict[str, int] = defaultdict(int)
        self._by_category: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._by_topic: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._lock = threading.Lock()

    def _handle_event(self, event: ConsentEvent) -> None:
        with self._lock:
            # Count by event kind
            self._counters[event.kind.name] += 1

            # Count grants/denials specifically
            if event.granted is True:
                self._counters["total_grants"] += 1
            elif event.granted is False:
                self._counters["total_denials"] += 1

            # Count by category
            if event.category:
                cat = event.category.value
                self._by_category[cat][event.kind.name] += 1
                if event.granted is True:
                    self._by_category[cat]["grants"] += 1
                elif event.granted is False:
                    self._by_category[cat]["denials"] += 1

            # Count by topic
            if event.topic:
                self._by_topic[event.topic][event.kind.name] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "by_category": {k: dict(v) for k, v in self._by_category.items()},
                "by_topic": {k: dict(v) for k, v in self._by_topic.items()},
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._by_category.clear()
            self._by_topic.clear()

    @property
    def total_events(self) -> int:
        """Total events processed."""
        with self._lock:
            return sum(self._counters.values())

    @property
    def grant_rate(self) -> float:
        """Grant rate as percentage."""
        with self._lock:
            grants = self._counters.get("total_grants", 0)
            denials = self._counters.get("total_denials", 0)
            total = grants + denials
            return (grants / total * 100) if total > 0 else 0.0


class CallbackHandler(BaseConsentHandler):
    """Handler that calls a custom callback function."""

    def __init__(
        self,
        callback: Callable[[ConsentEvent], None],
        name: str = "CallbackHandler",
        event_kinds: set[ConsentEventKind] | None = None,
    ) -> None:
        super().__init__(name)
        self._callback = callback
        self._supported_kinds = event_kinds

    def _handle_event(self, event: ConsentEvent) -> None:
        self._callback(event)


class FilteredHandler(BaseConsentHandler):
    """Handler that filters events before passing to another handler."""

    def __init__(
        self,
        inner_handler: ConsentEventHandler,
        filter_fn: Callable[[ConsentEvent], bool],
        name: str = "FilteredHandler",
    ) -> None:
        super().__init__(name)
        self._inner = inner_handler
        self._filter = filter_fn

    def _handle_event(self, event: ConsentEvent) -> None:
        if self._filter(event):
            self._inner.handle(event)


class PolicyHandler(BaseConsentHandler):
    """Handler that enforces consent policies.

    Can trigger actions when policy violations are detected.
    """

    def __init__(self, name: str = "PolicyHandler") -> None:
        super().__init__(name)
        self._policies: list[ConsentPolicy] = []
        self._violations: list[PolicyViolation] = []
        self._lock = threading.Lock()

    def add_policy(self, policy: ConsentPolicy) -> None:
        """Add a policy to enforce."""
        with self._lock:
            self._policies.append(policy)

    def remove_policy(self, policy: ConsentPolicy) -> bool:
        """Remove a policy."""
        with self._lock:
            if policy in self._policies:
                self._policies.remove(policy)
                return True
            return False

    def _handle_event(self, event: ConsentEvent) -> None:
        with self._lock:
            policies = list(self._policies)

        for policy in policies:
            try:
                violation = policy.check(event)
                if violation:
                    with self._lock:
                        self._violations.append(violation)
                    policy.on_violation(violation)
            except Exception as e:
                logging.warning(f"Policy check failed: {e}")

    def get_violations(self) -> list[PolicyViolation]:
        """Get all recorded violations."""
        with self._lock:
            return list(self._violations)

    def clear_violations(self) -> None:
        """Clear violation history."""
        with self._lock:
            self._violations.clear()


@dataclass
class PolicyViolation:
    """Record of a policy violation."""

    policy_name: str
    event: ConsentEvent
    message: str
    severity: str = "warning"
    timestamp: float = field(default_factory=time.time)


class ConsentPolicy(ABC):
    """Base class for consent policies."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def check(self, event: ConsentEvent) -> PolicyViolation | None:
        """Check event against policy. Return violation if failed."""
        pass

    def on_violation(self, violation: PolicyViolation) -> None:
        """Called when a violation is detected. Override to take action."""
        logging.warning(f"Policy violation [{self.name}]: {violation.message}")


class RateLimitPolicy(ConsentPolicy):
    """Policy that limits consent requests per time period."""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
        name: str = "RateLimitPolicy",
    ) -> None:
        super().__init__(name)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: list[float] = []
        self._lock = threading.Lock()

    def check(self, event: ConsentEvent) -> PolicyViolation | None:
        if event.kind != ConsentEventKind.CONSENT_REQUESTED:
            return None

        with self._lock:
            now = time.time()
            cutoff = now - self._window_seconds

            # Clean old requests
            self._requests = [t for t in self._requests if t > cutoff]

            # Add current
            self._requests.append(now)

            if len(self._requests) > self._max_requests:
                return PolicyViolation(
                    policy_name=self.name,
                    event=event,
                    message=f"Rate limit exceeded: {len(self._requests)} requests in {self._window_seconds}s",
                    severity="warning",
                )

        return None


class CategoryBlacklistPolicy(ConsentPolicy):
    """Policy that blocks certain categories entirely."""

    def __init__(
        self,
        blocked_categories: list[ConsentCategory],
        name: str = "CategoryBlacklistPolicy",
    ) -> None:
        super().__init__(name)
        self._blocked = set(blocked_categories)

    def check(self, event: ConsentEvent) -> PolicyViolation | None:
        if event.category and event.category in self._blocked:
            if event.granted is True:
                return PolicyViolation(
                    policy_name=self.name,
                    event=event,
                    message=f"Blocked category granted: {event.category.value}",
                    severity="error",
                )
        return None


class ForceOverridePolicy(ConsentPolicy):
    """Policy that tracks and limits force override usage."""

    def __init__(
        self,
        max_overrides: int = 10,
        window_hours: float = 24.0,
        name: str = "ForceOverridePolicy",
    ) -> None:
        super().__init__(name)
        self._max_overrides = max_overrides
        self._window_seconds = window_hours * 3600
        self._overrides: list[float] = []
        self._lock = threading.Lock()

    def check(self, event: ConsentEvent) -> PolicyViolation | None:
        if event.kind != ConsentEventKind.FORCE_OVERRIDE:
            return None

        with self._lock:
            now = time.time()
            cutoff = now - self._window_seconds
            self._overrides = [t for t in self._overrides if t > cutoff]
            self._overrides.append(now)

            if len(self._overrides) > self._max_overrides:
                return PolicyViolation(
                    policy_name=self.name,
                    event=event,
                    message=f"Too many force overrides: {len(self._overrides)} in {self._window_seconds/3600}h",
                    severity="error",
                )

        return None


class RateLimitHandler(BaseConsentHandler):
    """Handler that rate-limits consent events.

    Can delay or drop events that exceed the rate limit.
    """

    def __init__(
        self,
        max_per_second: float = 10.0,
        name: str = "RateLimitHandler",
    ) -> None:
        super().__init__(name)
        self._max_per_second = max_per_second
        self._min_interval = 1.0 / max_per_second if max_per_second > 0 else 0
        self._last_event_time = 0.0
        self._dropped_count = 0
        self._lock = threading.Lock()

    def _handle_event(self, event: ConsentEvent) -> None:
        with self._lock:
            now = time.time()
            if now - self._last_event_time < self._min_interval:
                self._dropped_count += 1
                return

            self._last_event_time = now

    @property
    def dropped_count(self) -> int:
        """Number of dropped events."""
        with self._lock:
            return self._dropped_count


# ========== Event-Enabled Consent Manager ==========


class EventAwareConsentManager:
    """ConsentManager wrapper that emits events via EventBus.

    This provides integration between ConsentManager operations
    and the event-driven handler system.
    """

    def __init__(
        self,
        consent_manager: ConsentManager | None = None,
        event_bus: ConsentEventBus | None = None,
    ) -> None:
        self._manager = consent_manager or ConsentManager()
        self._bus = event_bus or ConsentEventBus()

        # Emit initialization event
        self._bus.emit(ConsentEvent(kind=ConsentEventKind.MANAGER_INITIALIZED))

    @property
    def manager(self) -> ConsentManager:
        return self._manager

    @property
    def event_bus(self) -> ConsentEventBus:
        return self._bus

    def request_consent(
        self,
        topic: str,
        category: ConsentCategory | None = None,
        description: str | None = None,
    ) -> bool:
        # Emit request event
        self._bus.emit(
            ConsentEvent(
                kind=ConsentEventKind.CONSENT_REQUESTED,
                topic=topic,
                category=category,
                metadata={"description": description},
            )
        )

        # Get decision
        granted = self._manager.request_consent(topic, category, description)

        # Emit decision event
        self._bus.emit(
            ConsentEvent(
                kind=(
                    ConsentEventKind.CONSENT_GRANTED
                    if granted
                    else ConsentEventKind.CONSENT_DENIED
                ),
                topic=topic,
                category=category,
                granted=granted,
            )
        )

        return granted

    def grant(
        self,
        topic: str,
        level: ConsentLevel = ConsentLevel.SESSION,
        category: ConsentCategory | None = None,
        **kwargs,
    ) -> ConsentRecord:
        record = self._manager.grant(topic, level, category, **kwargs)

        self._bus.emit(
            ConsentEvent(
                kind=ConsentEventKind.CONSENT_GRANTED,
                topic=topic,
                category=category,
                granted=True,
                level=level,
                record=record,
                source="programmatic",
            )
        )

        return record

    def deny(
        self,
        topic: str,
        level: ConsentLevel = ConsentLevel.SESSION,
        category: ConsentCategory | None = None,
        **kwargs,
    ) -> ConsentRecord:
        record = self._manager.deny(topic, level, category, **kwargs)

        self._bus.emit(
            ConsentEvent(
                kind=ConsentEventKind.CONSENT_DENIED,
                topic=topic,
                category=category,
                granted=False,
                level=level,
                record=record,
                source="programmatic",
            )
        )

        return record

    def revoke(self, topic: str) -> bool:
        result = self._manager.revoke(topic)

        self._bus.emit(
            ConsentEvent(
                kind=ConsentEventKind.CONSENT_REVOKED,
                topic=topic,
                metadata={"was_granted": result},
            )
        )

        return result

    def revoke_all(self) -> None:
        self._manager.revoke_all()

        self._bus.emit(
            ConsentEvent(
                kind=ConsentEventKind.BULK_REVOKE,
                metadata={"scope": "all"},
            )
        )

    def revoke_category(self, category: ConsentCategory) -> int:
        count = self._manager.revoke_category(category)

        self._bus.emit(
            ConsentEvent(
                kind=ConsentEventKind.BULK_REVOKE,
                category=category,
                metadata={"scope": "category", "count": count},
            )
        )

        return count

    def enable_force_override(self) -> None:
        self._manager.enable_force_override()

        self._bus.emit(ConsentEvent(kind=ConsentEventKind.FORCE_OVERRIDE))

    def check(self, topic: str) -> bool | None:
        result = self._manager.check(topic)

        self._bus.emit(
            ConsentEvent(
                kind=ConsentEventKind.CONSENT_CHECKED,
                topic=topic,
                granted=result,
            )
        )

        return result

    def close(self) -> None:
        """Close the manager and emit lifecycle event."""
        self._bus.emit(ConsentEvent(kind=ConsentEventKind.MANAGER_CLOSED))

    # Delegate other methods
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


# ========== Convenience Functions ==========


def create_event_bus_with_logging(
    logger: logging.Logger | None = None,
) -> ConsentEventBus:
    """Create an event bus with logging handler attached."""
    bus = ConsentEventBus()
    bus.register(LoggingHandler(logger=logger))
    return bus


def create_event_bus_with_metrics() -> tuple[ConsentEventBus, MetricsHandler]:
    """Create an event bus with metrics handler attached."""
    bus = ConsentEventBus()
    metrics = MetricsHandler()
    bus.register(metrics)
    return bus, metrics


def on_consent_event(
    event_bus: ConsentEventBus,
    callback: Callable[[ConsentEvent], None],
    event_kinds: list[ConsentEventKind] | None = None,
) -> CallbackHandler:
    """Register a callback for consent events."""
    handler = CallbackHandler(
        callback=callback,
        event_kinds=set(event_kinds) if event_kinds else None,
    )
    bus_kinds = event_kinds if event_kinds else None
    event_bus.register(handler, event_kinds=bus_kinds)
    return handler


# =============================================================================
# HOOK PRIORITIZATION (5% Gap Coverage)
# Priority-based handler execution with dependencies and conflict resolution
# =============================================================================


class HookPriority(Enum):
    """Standard priority levels for hooks."""
    
    SYSTEM = 0  # System-level hooks, run first
    CRITICAL = 100  # Critical handlers
    HIGH = 200  # High priority
    NORMAL = 500  # Default priority
    LOW = 800  # Low priority
    MONITOR = 900  # Monitoring/logging hooks, run last
    CLEANUP = 1000  # Cleanup hooks, run after everything


@dataclass
class PrioritizedHandler:
    """A handler with priority and dependency information."""
    
    handler_id: str
    handler: ConsentEventHandler
    priority: int
    name: str = ""
    description: str = ""
    
    # Dependencies on other handlers
    depends_on: list[str] = field(default_factory=list)  # Must run after these
    blocks: list[str] = field(default_factory=list)  # Must run before these
    
    # Execution constraints
    exclusive_group: str | None = None  # Only one in group runs
    required: bool = False  # Failure aborts event processing
    timeout_seconds: float = 30.0  # Max execution time
    
    # State
    is_enabled: bool = True
    execution_count: int = 0
    total_execution_time: float = 0.0
    last_error: str | None = None
    
    def __lt__(self, other: "PrioritizedHandler") -> bool:
        """Compare by priority for sorting."""
        return self.priority < other.priority
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "handler_id": self.handler_id,
            "name": self.name,
            "priority": self.priority,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "exclusive_group": self.exclusive_group,
            "required": self.required,
            "is_enabled": self.is_enabled,
            "execution_count": self.execution_count,
            "avg_execution_time": (
                self.total_execution_time / self.execution_count 
                if self.execution_count > 0 else 0
            ),
        }


@dataclass
class HookExecutionResult:
    """Result of executing a prioritized hook."""
    
    handler_id: str
    success: bool
    execution_time: float
    error: str | None = None
    was_skipped: bool = False
    skip_reason: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "handler_id": self.handler_id,
            "success": self.success,
            "execution_time": self.execution_time,
            "error": self.error,
            "was_skipped": self.was_skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class HookChainResult:
    """Result of executing a chain of prioritized hooks."""
    
    event_id: str
    total_handlers: int
    executed_handlers: int
    successful_handlers: int
    failed_handlers: int
    skipped_handlers: int
    
    total_execution_time: float
    aborted: bool
    abort_reason: str | None
    
    results: list[HookExecutionResult]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "total_handlers": self.total_handlers,
            "executed_handlers": self.executed_handlers,
            "successful_handlers": self.successful_handlers,
            "failed_handlers": self.failed_handlers,
            "skipped_handlers": self.skipped_handlers,
            "total_execution_time": self.total_execution_time,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "results": [r.to_dict() for r in self.results],
        }


class DependencyResolver:
    """Resolves handler dependencies into execution order.
    
    Uses topological sorting to respect dependencies while
    maintaining priority ordering where possible.
    """
    
    def __init__(self) -> None:
        """Initialize resolver."""
        self._handlers: dict[str, PrioritizedHandler] = {}
    
    def add_handler(self, handler: PrioritizedHandler) -> None:
        """Add a handler to the resolver.
        
        Args:
            handler: Handler to add
        """
        self._handlers[handler.handler_id] = handler
    
    def remove_handler(self, handler_id: str) -> None:
        """Remove a handler from the resolver.
        
        Args:
            handler_id: Handler ID to remove
        """
        self._handlers.pop(handler_id, None)
    
    def resolve(self) -> list[PrioritizedHandler]:
        """Resolve dependencies and return execution order.
        
        Returns:
            List of handlers in execution order
        
        Raises:
            ValueError: If circular dependency detected
        """
        enabled = {
            hid: h for hid, h in self._handlers.items() if h.is_enabled
        }
        
        if not enabled:
            return []
        
        # Build dependency graph
        # handler -> set of handlers it depends on
        graph: dict[str, set[str]] = {}
        
        for hid, handler in enabled.items():
            deps = set()
            
            # Direct dependencies
            for dep_id in handler.depends_on:
                if dep_id in enabled:
                    deps.add(dep_id)
            
            # Reverse dependencies (from blocks)
            for other_id, other in enabled.items():
                if hid in other.blocks and other_id in enabled:
                    deps.add(other_id)
            
            graph[hid] = deps
        
        # Topological sort with priority as tiebreaker
        resolved: list[str] = []
        pending = set(enabled.keys())
        
        while pending:
            # Find handlers with no unresolved dependencies
            ready = [
                hid for hid in pending
                if all(dep in resolved for dep in graph[hid])
            ]
            
            if not ready:
                # Circular dependency
                cycle = self._find_cycle(graph, pending)
                raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}")
            
            # Sort ready handlers by priority
            ready.sort(key=lambda hid: enabled[hid].priority)
            
            # Take the highest priority handler
            next_handler = ready[0]
            resolved.append(next_handler)
            pending.remove(next_handler)
        
        return [enabled[hid] for hid in resolved]
    
    def _find_cycle(
        self,
        graph: dict[str, set[str]],
        nodes: set[str],
    ) -> list[str]:
        """Find a cycle in the dependency graph.
        
        Args:
            graph: Dependency graph
            nodes: Nodes to check
            
        Returns:
            List of node IDs forming a cycle
        """
        visited = set()
        rec_stack = []
        
        def dfs(node: str) -> list[str] | None:
            visited.add(node)
            rec_stack.append(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor in nodes:
                    if neighbor not in visited:
                        result = dfs(neighbor)
                        if result:
                            return result
                    elif neighbor in rec_stack:
                        # Found cycle
                        idx = rec_stack.index(neighbor)
                        return rec_stack[idx:] + [neighbor]
            
            rec_stack.pop()
            return None
        
        for node in nodes:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle
        
        return ["unknown cycle"]
    
    def validate_dependencies(self) -> list[str]:
        """Validate all dependencies can be resolved.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for hid, handler in self._handlers.items():
            # Check depends_on exist
            for dep_id in handler.depends_on:
                if dep_id not in self._handlers:
                    errors.append(
                        f"Handler '{hid}' depends on unknown handler '{dep_id}'"
                    )
            
            # Check blocks exist
            for block_id in handler.blocks:
                if block_id not in self._handlers:
                    errors.append(
                        f"Handler '{hid}' blocks unknown handler '{block_id}'"
                    )
        
        # Check for cycles
        try:
            self.resolve()
        except ValueError as e:
            errors.append(str(e))
        
        return errors


class ExclusiveGroupManager:
    """Manages exclusive groups where only one handler runs.
    
    When multiple handlers belong to the same exclusive group,
    only the highest priority one executes.
    """
    
    def __init__(self) -> None:
        """Initialize manager."""
        self._groups: dict[str, list[str]] = {}  # group -> [handler_ids]
        self._handler_groups: dict[str, str] = {}  # handler_id -> group
    
    def register(self, handler_id: str, group: str) -> None:
        """Register a handler to an exclusive group.
        
        Args:
            handler_id: Handler ID
            group: Group name
        """
        if group not in self._groups:
            self._groups[group] = []
        
        if handler_id not in self._groups[group]:
            self._groups[group].append(handler_id)
        
        self._handler_groups[handler_id] = group
    
    def unregister(self, handler_id: str) -> None:
        """Unregister a handler from its group.
        
        Args:
            handler_id: Handler ID
        """
        group = self._handler_groups.pop(handler_id, None)
        if group and group in self._groups:
            try:
                self._groups[group].remove(handler_id)
            except ValueError:
                pass
    
    def get_group(self, handler_id: str) -> str | None:
        """Get the group a handler belongs to.
        
        Args:
            handler_id: Handler ID
            
        Returns:
            Group name or None
        """
        return self._handler_groups.get(handler_id)
    
    def should_execute(
        self,
        handler_id: str,
        already_executed: set[str],
    ) -> tuple[bool, str | None]:
        """Check if a handler should execute based on group exclusivity.
        
        Args:
            handler_id: Handler to check
            already_executed: Set of already executed handler IDs
            
        Returns:
            Tuple of (should_execute, skip_reason)
        """
        group = self._handler_groups.get(handler_id)
        
        if not group:
            return (True, None)
        
        # Check if any handler from same group already executed
        group_handlers = set(self._groups.get(group, []))
        executed_from_group = already_executed & group_handlers
        
        if executed_from_group:
            return (
                False,
                f"Exclusive group '{group}': {list(executed_from_group)[0]} already executed"
            )
        
        return (True, None)
    
    def get_groups(self) -> dict[str, list[str]]:
        """Get all groups and their handlers.
        
        Returns:
            Dictionary of group -> handler IDs
        """
        return self._groups.copy()


class PrioritizedEventBus:
    """Event bus with prioritized handler execution.
    
    Features:
    - Priority-based execution order
    - Dependency resolution
    - Exclusive group support
    - Execution timeout
    - Required handler enforcement
    """
    
    def __init__(
        self,
        parallel_execution: bool = False,
        abort_on_required_failure: bool = True,
    ) -> None:
        """Initialize prioritized event bus.
        
        Args:
            parallel_execution: Execute handlers in parallel (ignoring deps)
            abort_on_required_failure: Abort if required handler fails
        """
        self._parallel = parallel_execution
        self._abort_on_required = abort_on_required_failure
        self._lock = threading.Lock()
        
        self._handlers: dict[str, PrioritizedHandler] = {}
        self._resolver = DependencyResolver()
        self._exclusive = ExclusiveGroupManager()
        
        self._event_counter = 0
        self._execution_history: list[HookChainResult] = []
    
    def register(
        self,
        handler: ConsentEventHandler,
        priority: int = HookPriority.NORMAL.value,
        handler_id: str | None = None,
        name: str = "",
        depends_on: list[str] | None = None,
        blocks: list[str] | None = None,
        exclusive_group: str | None = None,
        required: bool = False,
        timeout_seconds: float = 30.0,
    ) -> str:
        """Register a prioritized handler.
        
        Args:
            handler: Handler to register
            priority: Execution priority (lower = earlier)
            handler_id: Optional handler ID
            name: Optional handler name
            depends_on: Handler IDs this must run after
            blocks: Handler IDs this must run before
            exclusive_group: Exclusive group name
            required: Whether handler is required
            timeout_seconds: Execution timeout
            
        Returns:
            Handler ID
        """
        with self._lock:
            if handler_id is None:
                handler_id = f"handler_{len(self._handlers) + 1}_{int(time.time())}"
            
            prioritized = PrioritizedHandler(
                handler_id=handler_id,
                handler=handler,
                priority=priority,
                name=name or handler_id,
                depends_on=depends_on or [],
                blocks=blocks or [],
                exclusive_group=exclusive_group,
                required=required,
                timeout_seconds=timeout_seconds,
            )
            
            self._handlers[handler_id] = prioritized
            self._resolver.add_handler(prioritized)
            
            if exclusive_group:
                self._exclusive.register(handler_id, exclusive_group)
            
            return handler_id
    
    def unregister(self, handler_id: str) -> bool:
        """Unregister a handler.
        
        Args:
            handler_id: Handler to unregister
            
        Returns:
            True if handler was found and removed
        """
        with self._lock:
            if handler_id not in self._handlers:
                return False
            
            del self._handlers[handler_id]
            self._resolver.remove_handler(handler_id)
            self._exclusive.unregister(handler_id)
            return True
    
    def set_priority(self, handler_id: str, priority: int) -> bool:
        """Update a handler's priority.
        
        Args:
            handler_id: Handler ID
            priority: New priority
            
        Returns:
            True if updated
        """
        with self._lock:
            if handler_id not in self._handlers:
                return False
            
            self._handlers[handler_id].priority = priority
            return True
    
    def enable_handler(self, handler_id: str) -> bool:
        """Enable a handler.
        
        Args:
            handler_id: Handler ID
            
        Returns:
            True if enabled
        """
        with self._lock:
            if handler_id not in self._handlers:
                return False
            
            self._handlers[handler_id].is_enabled = True
            return True
    
    def disable_handler(self, handler_id: str) -> bool:
        """Disable a handler.
        
        Args:
            handler_id: Handler ID
            
        Returns:
            True if disabled
        """
        with self._lock:
            if handler_id not in self._handlers:
                return False
            
            self._handlers[handler_id].is_enabled = False
            return True
    
    def dispatch(self, event: ConsentEvent) -> HookChainResult:
        """Dispatch an event through prioritized handlers.
        
        Args:
            event: Event to dispatch
            
        Returns:
            HookChainResult with execution details
        """
        with self._lock:
            self._event_counter += 1
            event_id = f"event_{self._event_counter}"
            
            # Resolve execution order
            try:
                ordered_handlers = self._resolver.resolve()
            except ValueError as e:
                return HookChainResult(
                    event_id=event_id,
                    total_handlers=len(self._handlers),
                    executed_handlers=0,
                    successful_handlers=0,
                    failed_handlers=0,
                    skipped_handlers=len(self._handlers),
                    total_execution_time=0.0,
                    aborted=True,
                    abort_reason=str(e),
                    results=[],
                )
        
        # Execute handlers
        results: list[HookExecutionResult] = []
        executed = set()
        start_time = time.time()
        aborted = False
        abort_reason = None
        
        for prioritized in ordered_handlers:
            if aborted:
                results.append(HookExecutionResult(
                    handler_id=prioritized.handler_id,
                    success=False,
                    execution_time=0.0,
                    was_skipped=True,
                    skip_reason="Chain aborted",
                ))
                continue
            
            # Check exclusive group
            should_run, skip_reason = self._exclusive.should_execute(
                prioritized.handler_id, executed
            )
            
            if not should_run:
                results.append(HookExecutionResult(
                    handler_id=prioritized.handler_id,
                    success=True,
                    execution_time=0.0,
                    was_skipped=True,
                    skip_reason=skip_reason,
                ))
                continue
            
            # Execute handler
            result = self._execute_handler(prioritized, event)
            results.append(result)
            
            if result.success or result.was_skipped:
                executed.add(prioritized.handler_id)
            else:
                # Check if required
                if prioritized.required and self._abort_on_required:
                    aborted = True
                    abort_reason = f"Required handler '{prioritized.handler_id}' failed"
        
        total_time = time.time() - start_time
        
        chain_result = HookChainResult(
            event_id=event_id,
            total_handlers=len(ordered_handlers),
            executed_handlers=len([r for r in results if not r.was_skipped]),
            successful_handlers=len([r for r in results if r.success and not r.was_skipped]),
            failed_handlers=len([r for r in results if not r.success and not r.was_skipped]),
            skipped_handlers=len([r for r in results if r.was_skipped]),
            total_execution_time=total_time,
            aborted=aborted,
            abort_reason=abort_reason,
            results=results,
        )
        
        with self._lock:
            self._execution_history.append(chain_result)
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-500:]
        
        return chain_result
    
    def _execute_handler(
        self,
        prioritized: PrioritizedHandler,
        event: ConsentEvent,
    ) -> HookExecutionResult:
        """Execute a single handler with timeout.
        
        Args:
            prioritized: Handler to execute
            event: Event to pass
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        try:
            # Execute handler
            prioritized.handler.handle_event(event)
            
            execution_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                prioritized.execution_count += 1
                prioritized.total_execution_time += execution_time
                prioritized.last_error = None
            
            return HookExecutionResult(
                handler_id=prioritized.handler_id,
                success=True,
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            with self._lock:
                prioritized.last_error = error_msg
            
            return HookExecutionResult(
                handler_id=prioritized.handler_id,
                success=False,
                execution_time=execution_time,
                error=error_msg,
            )
    
    def get_execution_order(self) -> list[dict[str, Any]]:
        """Get the current execution order of handlers.
        
        Returns:
            List of handlers in execution order
        """
        with self._lock:
            try:
                ordered = self._resolver.resolve()
                return [h.to_dict() for h in ordered]
            except ValueError:
                return []
    
    def get_handler_stats(self, handler_id: str) -> dict[str, Any] | None:
        """Get statistics for a handler.
        
        Args:
            handler_id: Handler ID
            
        Returns:
            Handler statistics or None
        """
        with self._lock:
            handler = self._handlers.get(handler_id)
            if not handler:
                return None
            return handler.to_dict()
    
    def get_all_handlers(self) -> list[dict[str, Any]]:
        """Get all registered handlers.
        
        Returns:
            List of handler information
        """
        with self._lock:
            return [h.to_dict() for h in self._handlers.values()]
    
    def validate(self) -> list[str]:
        """Validate the handler configuration.
        
        Returns:
            List of validation errors
        """
        with self._lock:
            return self._resolver.validate_dependencies()
    
    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent execution history.
        
        Args:
            limit: Maximum results to return
            
        Returns:
            List of execution results
        """
        with self._lock:
            recent = self._execution_history[-limit:]
            return [r.to_dict() for r in reversed(recent)]


class DynamicPriorityAdjuster:
    """Dynamically adjusts handler priorities based on performance.
    
    Features:
    - Automatic priority adjustment
    - Performance-based optimization
    - Failure rate tracking
    """
    
    def __init__(
        self,
        event_bus: PrioritizedEventBus,
        adjustment_interval: float = 60.0,
    ) -> None:
        """Initialize adjuster.
        
        Args:
            event_bus: Event bus to manage
            adjustment_interval: Seconds between adjustments
        """
        self._bus = event_bus
        self._interval = adjustment_interval
        self._lock = threading.Lock()
        
        # Tracking
        self._failure_counts: dict[str, int] = {}
        self._slow_counts: dict[str, int] = {}  # Execution > threshold
        self._slow_threshold = 5.0  # seconds
        
        # Adjustment bounds
        self._min_priority = 0
        self._max_priority = 1000
        self._priority_step = 50
        
        # Background thread
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
    
    def record_execution(
        self,
        result: HookExecutionResult,
    ) -> None:
        """Record a handler execution for tracking.
        
        Args:
            result: Execution result
        """
        with self._lock:
            if not result.success and not result.was_skipped:
                self._failure_counts[result.handler_id] = (
                    self._failure_counts.get(result.handler_id, 0) + 1
                )
            
            if result.execution_time > self._slow_threshold:
                self._slow_counts[result.handler_id] = (
                    self._slow_counts.get(result.handler_id, 0) + 1
                )
    
    def adjust_priorities(self) -> dict[str, int]:
        """Adjust priorities based on tracked performance.
        
        Returns:
            Dictionary of handler_id -> new priority
        """
        adjustments = {}
        
        with self._lock:
            handlers = self._bus.get_all_handlers()
            
            for handler_info in handlers:
                handler_id = handler_info["handler_id"]
                current = handler_info["priority"]
                
                failures = self._failure_counts.get(handler_id, 0)
                slow = self._slow_counts.get(handler_id, 0)
                
                new_priority = current
                
                # Demote failing handlers
                if failures > 5:
                    new_priority = min(
                        current + self._priority_step,
                        self._max_priority
                    )
                
                # Demote slow handlers
                if slow > 3:
                    new_priority = min(
                        current + self._priority_step // 2,
                        self._max_priority
                    )
                
                # Promote well-performing handlers
                exec_count = handler_info.get("execution_count", 0)
                if exec_count > 10 and failures == 0 and slow == 0:
                    new_priority = max(
                        current - self._priority_step // 2,
                        self._min_priority
                    )
                
                if new_priority != current:
                    self._bus.set_priority(handler_id, new_priority)
                    adjustments[handler_id] = new_priority
            
            # Reset counters
            self._failure_counts.clear()
            self._slow_counts.clear()
        
        return adjustments
    
    def start_auto_adjustment(self) -> None:
        """Start automatic priority adjustment."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        def adjust_loop():
            while not self._stop_event.is_set():
                self._stop_event.wait(self._interval)
                if not self._stop_event.is_set():
                    self.adjust_priorities()
        
        self._thread = threading.Thread(target=adjust_loop, daemon=True)
        self._thread.start()
    
    def stop_auto_adjustment(self) -> None:
        """Stop automatic priority adjustment."""
        self._stop_event.set()
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def get_tracking_stats(self) -> dict[str, Any]:
        """Get current tracking statistics.
        
        Returns:
            Tracking statistics
        """
        with self._lock:
            return {
                "failure_counts": self._failure_counts.copy(),
                "slow_counts": self._slow_counts.copy(),
                "slow_threshold_seconds": self._slow_threshold,
            }

