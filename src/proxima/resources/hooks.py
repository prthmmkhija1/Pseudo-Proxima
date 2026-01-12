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
