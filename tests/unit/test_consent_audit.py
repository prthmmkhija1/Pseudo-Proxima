"""Tests for consent audit logging and event hooks.

Tests cover:
- AuditEvent creation and serialization
- AuditLog operations and querying
- AuditStorage backends (memory and file)
- AuditQueryBuilder fluent interface
- AuditStatistics computation
- AuditedConsentManager wrapper
- ConsentEventBus and event distribution
- Built-in handlers (Logging, Metrics, Policy)
- Policy enforcement
"""

import json
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_event_creation(self):
        """Test creating an audit event."""
        from proxima.resources.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.CONSENT_GRANTED,
            timestamp=time.time(),
            event_id="test-123",
            topic="test_topic",
        )

        assert event.event_type == AuditEventType.CONSENT_GRANTED
        assert event.topic == "test_topic"
        assert event.event_id == "test-123"
        assert event.checksum is not None

    def test_event_integrity(self):
        """Test event integrity verification."""
        from proxima.resources.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.CONSENT_DENIED,
            timestamp=time.time(),
            event_id="test-456",
            topic="secure_topic",
            granted=False,
        )

        assert event.verify_integrity()

    def test_event_serialization(self):
        """Test event to_dict and from_dict."""
        from proxima.resources.audit import AuditEvent, AuditEventType, AuditSeverity
        from proxima.resources.consent import ConsentCategory, ConsentLevel

        event = AuditEvent(
            event_type=AuditEventType.CONSENT_GRANTED,
            timestamp=1234567890.0,
            event_id="ser-test",
            topic="serialize_test",
            category=ConsentCategory.LOCAL_LLM,
            granted=True,
            level=ConsentLevel.SESSION,
            severity=AuditSeverity.INFO,
        )

        data = event.to_dict()
        restored = AuditEvent.from_dict(data)

        assert restored.event_type == event.event_type
        assert restored.topic == event.topic
        assert restored.category == event.category
        assert restored.granted == event.granted
        assert restored.level == event.level

    def test_event_datetime_property(self):
        """Test datetime property conversion."""
        from proxima.resources.audit import AuditEvent, AuditEventType

        ts = 1704067200.0  # 2024-01-01 00:00:00 UTC
        event = AuditEvent(
            event_type=AuditEventType.CONSENT_REQUESTED,
            timestamp=ts,
            event_id="dt-test",
        )

        dt = event.datetime
        assert isinstance(dt, datetime)

    def test_event_str(self):
        """Test event string representation."""
        from proxima.resources.audit import AuditEvent, AuditEventType

        event = AuditEvent(
            event_type=AuditEventType.CONSENT_GRANTED,
            timestamp=time.time(),
            event_id="str-test",
            topic="my_topic",
            granted=True,
        )

        s = str(event)
        assert "CONSENT_GRANTED" in s
        assert "GRANTED" in s
        assert "my_topic" in s


class TestMemoryAuditStorage:
    """Tests for in-memory audit storage."""

    def test_append_and_read(self):
        """Test appending and reading events."""
        from proxima.resources.audit import AuditEvent, AuditEventType, MemoryAuditStorage

        storage = MemoryAuditStorage(max_events=100)

        event = AuditEvent(
            event_type=AuditEventType.CONSENT_GRANTED,
            timestamp=time.time(),
            event_id="mem-1",
            topic="test",
        )

        storage.append(event)
        events = storage.read_all()

        assert len(events) == 1
        assert events[0].event_id == "mem-1"

    def test_max_events_limit(self):
        """Test that storage respects max_events limit."""
        from proxima.resources.audit import AuditEvent, AuditEventType, MemoryAuditStorage

        storage = MemoryAuditStorage(max_events=5)

        for i in range(10):
            event = AuditEvent(
                event_type=AuditEventType.CONSENT_CHECKED,
                timestamp=time.time(),
                event_id=f"limit-{i}",
            )
            storage.append(event)

        events = storage.read_all()
        assert len(events) <= 5

    def test_read_range(self):
        """Test reading events by time range."""
        from proxima.resources.audit import AuditEvent, AuditEventType, MemoryAuditStorage

        storage = MemoryAuditStorage()
        now = time.time()

        # Add events at different times
        for i in range(5):
            event = AuditEvent(
                event_type=AuditEventType.CONSENT_GRANTED,
                timestamp=now + i * 100,
                event_id=f"range-{i}",
            )
            storage.append(event)

        # Query middle range
        events = storage.read_range(start_time=now + 100, end_time=now + 300)
        assert len(events) >= 2

    def test_clear(self):
        """Test clearing storage."""
        from proxima.resources.audit import AuditEvent, AuditEventType, MemoryAuditStorage

        storage = MemoryAuditStorage()

        for i in range(3):
            storage.append(
                AuditEvent(
                    event_type=AuditEventType.CONSENT_GRANTED,
                    timestamp=time.time(),
                    event_id=f"clear-{i}",
                )
            )

        count = storage.clear()
        assert count == 3
        assert storage.count == 0


class TestFileAuditStorage:
    """Tests for file-based audit storage."""

    def test_append_and_read(self):
        """Test file-based append and read."""
        from proxima.resources.audit import AuditEvent, AuditEventType, FileAuditStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.log"
            storage = FileAuditStorage(path)

            event = AuditEvent(
                event_type=AuditEventType.CONSENT_GRANTED,
                timestamp=time.time(),
                event_id="file-1",
                topic="file_test",
            )

            storage.append(event)
            events = storage.read_all()

            assert len(events) == 1
            assert events[0].topic == "file_test"

    def test_file_persists(self):
        """Test that events persist to disk."""
        from proxima.resources.audit import AuditEvent, AuditEventType, FileAuditStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "persist.log"

            # Write with one instance
            storage1 = FileAuditStorage(path)
            storage1.append(
                AuditEvent(
                    event_type=AuditEventType.CONSENT_DENIED,
                    timestamp=time.time(),
                    event_id="persist-1",
                )
            )

            # Read with new instance
            storage2 = FileAuditStorage(path)
            events = storage2.read_all()

            assert len(events) == 1
            assert events[0].event_id == "persist-1"


class TestAuditLog:
    """Tests for AuditLog main interface."""

    def test_log_event(self):
        """Test logging an event."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        event = log.log(
            AuditEventType.CONSENT_GRANTED,
            topic="log_test",
            granted=True,
        )

        assert event is not None
        assert event.topic == "log_test"

    def test_disabled_logging(self):
        """Test that disabled log doesn't record events."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog(enabled=False)
        event = log.log(AuditEventType.CONSENT_GRANTED, topic="disabled")

        assert event is None
        assert len(log.get_all()) == 0

    def test_get_by_type(self):
        """Test filtering by event type."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="a")
        log.log(AuditEventType.CONSENT_DENIED, topic="b")
        log.log(AuditEventType.CONSENT_GRANTED, topic="c")

        grants = log.get_by_type(AuditEventType.CONSENT_GRANTED)
        assert len(grants) == 2

    def test_get_by_topic(self):
        """Test filtering by topic."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="topic_a")
        log.log(AuditEventType.CONSENT_GRANTED, topic="topic_b")
        log.log(AuditEventType.CONSENT_DENIED, topic="topic_a")

        events = log.get_by_topic("topic_a")
        assert len(events) == 2

    def test_get_grants_denials(self):
        """Test getting grants and denials."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="g1", granted=True)
        log.log(AuditEventType.CONSENT_GRANTED, topic="g2", granted=True)
        log.log(AuditEventType.CONSENT_DENIED, topic="d1", granted=False)

        assert len(log.get_grants()) == 2
        assert len(log.get_denials()) == 1

    def test_event_listener(self):
        """Test real-time event listener."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        received = []

        def listener(event):
            received.append(event)

        log.add_listener(listener)
        log.log(AuditEventType.CONSENT_GRANTED, topic="listen")

        assert len(received) == 1
        assert received[0].topic == "listen"

    def test_export_json(self):
        """Test exporting audit log to JSON."""
        from proxima.resources.audit import AuditEventType, AuditLog

        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog()
            log.log(AuditEventType.CONSENT_GRANTED, topic="export1")
            log.log(AuditEventType.CONSENT_DENIED, topic="export2")

            path = Path(tmpdir) / "export.json"
            count = log.export_json(path)

            # Count includes the AUDIT_EXPORTED event
            assert count >= 2
            assert path.exists()

            data = json.loads(path.read_text())
            assert isinstance(data, list)


class TestAuditQueryBuilder:
    """Tests for fluent query interface."""

    def test_query_by_type(self):
        """Test query filtering by type."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="q1")
        log.log(AuditEventType.CONSENT_DENIED, topic="q2")
        log.log(AuditEventType.CONSENT_REVOKED, topic="q3")

        results = log.query().by_type(AuditEventType.CONSENT_GRANTED).execute()
        assert len(results) == 1
        assert results[0].topic == "q1"

    def test_query_granted_only(self):
        """Test query for granted only."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="g", granted=True)
        log.log(AuditEventType.CONSENT_DENIED, topic="d", granted=False)

        results = log.query().granted_only().execute()
        assert len(results) == 1
        assert results[0].granted is True

    def test_query_chaining(self):
        """Test chaining multiple query conditions."""
        from proxima.resources.audit import AuditEventType, AuditLog
        from proxima.resources.consent import ConsentCategory

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="a", category=ConsentCategory.LOCAL_LLM)
        log.log(AuditEventType.CONSENT_GRANTED, topic="b", category=ConsentCategory.REMOTE_LLM)
        log.log(AuditEventType.CONSENT_DENIED, topic="c", category=ConsentCategory.LOCAL_LLM)

        results = (
            log.query()
            .by_type(AuditEventType.CONSENT_GRANTED)
            .by_category(ConsentCategory.LOCAL_LLM)
            .execute()
        )
        assert len(results) == 1
        assert results[0].topic == "a"

    def test_query_limit_offset(self):
        """Test limit and offset."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        for i in range(10):
            log.log(AuditEventType.CONSENT_GRANTED, topic=f"page-{i}")

        results = log.query().offset(3).limit(3).execute()
        assert len(results) == 3

    def test_query_count(self):
        """Test counting query results."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        for i in range(5):
            log.log(AuditEventType.CONSENT_GRANTED, topic=f"count-{i}")

        count = log.query().by_type(AuditEventType.CONSENT_GRANTED).count()
        assert count == 5

    def test_query_first_last(self):
        """Test getting first/last results."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="first")
        time.sleep(0.01)
        log.log(AuditEventType.CONSENT_GRANTED, topic="last")

        first = log.query().by_type(AuditEventType.CONSENT_GRANTED).first()
        assert first is not None
        assert first.topic == "first"


class TestAuditStatistics:
    """Tests for audit statistics computation."""

    def test_statistics_from_events(self):
        """Test computing statistics from events."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="s1", granted=True)
        log.log(AuditEventType.CONSENT_GRANTED, topic="s2", granted=True)
        log.log(AuditEventType.CONSENT_DENIED, topic="s3", granted=False)
        log.log(AuditEventType.CONSENT_REVOKED, topic="s4")

        stats = log.get_statistics()

        assert stats.total_events == 4
        assert stats.grants == 2
        assert stats.denials == 1
        assert stats.revocations == 1

    def test_grant_rate(self):
        """Test grant rate calculation."""
        from proxima.resources.audit import AuditStatistics

        stats = AuditStatistics(grants=7, denials=3)
        assert stats.grant_rate == 70.0
        assert stats.denial_rate == 30.0

    def test_empty_statistics(self):
        """Test statistics with no events."""
        from proxima.resources.audit import AuditStatistics

        stats = AuditStatistics.from_events([])
        assert stats.total_events == 0
        assert stats.grant_rate == 0.0


class TestAuditReport:
    """Tests for audit report generation."""

    def test_generate_summary(self):
        """Test summary report generation."""
        from proxima.resources.audit import AuditEventType, AuditLog, AuditReport

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="r1", granted=True)
        log.log(AuditEventType.CONSENT_DENIED, topic="r2", granted=False)

        report = AuditReport(log)
        summary = report.generate_summary()

        assert "CONSENT AUDIT SUMMARY" in summary
        assert "Grants" in summary or "grants" in summary.lower()

    def test_export_html(self):
        """Test HTML report export."""
        from proxima.resources.audit import AuditEventType, AuditLog, AuditReport

        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog()
            log.log(AuditEventType.CONSENT_GRANTED, topic="html1")

            report = AuditReport(log)
            path = Path(tmpdir) / "report.html"
            report.export_html(path)

            assert path.exists()
            content = path.read_text()
            assert "<html>" in content
            assert "Consent Audit Report" in content


class TestAuditedConsentManager:
    """Tests for audited consent manager wrapper."""

    def test_grant_logs_event(self):
        """Test that grant operations are logged."""
        from proxima.resources.audit import AuditedConsentManager, AuditEventType
        from proxima.resources.consent import ConsentManager

        manager = AuditedConsentManager(
            consent_manager=ConsentManager(),
        )

        manager.grant("test_topic")

        events = manager.audit_log.get_by_type(AuditEventType.CONSENT_GRANTED)
        assert len(events) >= 1

    def test_revoke_logs_event(self):
        """Test that revoke operations are logged."""
        from proxima.resources.audit import AuditedConsentManager, AuditEventType
        from proxima.resources.consent import ConsentManager

        manager = AuditedConsentManager(consent_manager=ConsentManager())
        manager.grant("to_revoke")
        manager.revoke("to_revoke")

        events = manager.audit_log.get_by_type(AuditEventType.CONSENT_REVOKED)
        assert len(events) >= 1

    def test_session_lifecycle(self):
        """Test session start/end events."""
        from proxima.resources.audit import AuditedConsentManager, AuditEventType

        manager = AuditedConsentManager()
        manager.close()

        events = manager.audit_log.get_all()
        event_types = [e.event_type for e in events]

        assert AuditEventType.SESSION_STARTED in event_types
        assert AuditEventType.SESSION_ENDED in event_types


# ========== Tests for Event Hooks ==========


class TestConsentEventBus:
    """Tests for event bus."""

    def test_register_and_emit(self):
        """Test registering handler and emitting events."""
        from proxima.resources.hooks import (
            CallbackHandler,
            ConsentEvent,
            ConsentEventBus,
            ConsentEventKind,
        )

        bus = ConsentEventBus()
        received = []

        handler = CallbackHandler(callback=lambda e: received.append(e))
        bus.register(handler)

        event = ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED, topic="bus_test")
        bus.emit(event)

        assert len(received) == 1
        assert received[0].topic == "bus_test"

    def test_unregister_handler(self):
        """Test unregistering a handler."""
        from proxima.resources.hooks import (
            CallbackHandler,
            ConsentEvent,
            ConsentEventBus,
            ConsentEventKind,
        )

        bus = ConsentEventBus()
        received = []

        handler = CallbackHandler(callback=lambda e: received.append(e))
        bus.register(handler)
        bus.unregister(handler)

        bus.emit(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED))

        assert len(received) == 0

    def test_pause_resume(self):
        """Test pausing and resuming event emission."""
        from proxima.resources.hooks import (
            CallbackHandler,
            ConsentEvent,
            ConsentEventBus,
            ConsentEventKind,
        )

        bus = ConsentEventBus()
        received = []

        bus.register(CallbackHandler(callback=lambda e: received.append(e)))

        bus.pause()
        bus.emit(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED))
        assert len(received) == 0

        bus.resume()
        bus.emit(ConsentEvent(kind=ConsentEventKind.CONSENT_DENIED))
        assert len(received) == 1

    def test_event_history(self):
        """Test event history tracking."""
        from proxima.resources.hooks import ConsentEvent, ConsentEventBus, ConsentEventKind

        bus = ConsentEventBus()

        for i in range(5):
            bus.emit(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED, topic=f"hist-{i}"))

        history = bus.get_history()
        assert len(history) == 5


class TestLoggingHandler:
    """Tests for logging handler."""

    def test_logs_events(self):
        """Test that events are logged."""
        from proxima.resources.hooks import ConsentEvent, ConsentEventKind, LoggingHandler

        handler = LoggingHandler()
        event = ConsentEvent(
            kind=ConsentEventKind.CONSENT_GRANTED,
            topic="log_test",
            granted=True,
        )

        # Should not raise
        handler.handle(event)


class TestMetricsHandler:
    """Tests for metrics handler."""

    def test_tracks_metrics(self):
        """Test that metrics are tracked."""
        from proxima.resources.hooks import ConsentEvent, ConsentEventKind, MetricsHandler

        handler = MetricsHandler()

        handler.handle(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED, granted=True))
        handler.handle(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED, granted=True))
        handler.handle(ConsentEvent(kind=ConsentEventKind.CONSENT_DENIED, granted=False))

        metrics = handler.get_metrics()
        assert metrics["counters"]["total_grants"] == 2
        assert metrics["counters"]["total_denials"] == 1

    def test_grant_rate(self):
        """Test grant rate calculation."""
        from proxima.resources.hooks import ConsentEvent, ConsentEventKind, MetricsHandler

        handler = MetricsHandler()

        for _ in range(7):
            handler.handle(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED, granted=True))
        for _ in range(3):
            handler.handle(ConsentEvent(kind=ConsentEventKind.CONSENT_DENIED, granted=False))

        assert handler.grant_rate == 70.0

    def test_reset_metrics(self):
        """Test resetting metrics."""
        from proxima.resources.hooks import ConsentEvent, ConsentEventKind, MetricsHandler

        handler = MetricsHandler()
        handler.handle(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED, granted=True))

        handler.reset()
        assert handler.total_events == 0


class TestPolicyHandler:
    """Tests for policy enforcement handler."""

    def test_rate_limit_policy(self):
        """Test rate limit policy enforcement."""
        from proxima.resources.hooks import (
            ConsentEvent,
            ConsentEventKind,
            PolicyHandler,
            RateLimitPolicy,
        )

        policy = RateLimitPolicy(max_requests=3, window_seconds=1.0)
        handler = PolicyHandler()
        handler.add_policy(policy)

        for _i in range(5):
            handler.handle(ConsentEvent(kind=ConsentEventKind.CONSENT_REQUESTED))

        violations = handler.get_violations()
        assert len(violations) >= 1  # Should have at least one rate limit violation

    def test_category_blacklist_policy(self):
        """Test category blacklist policy."""
        from proxima.resources.consent import ConsentCategory
        from proxima.resources.hooks import (
            CategoryBlacklistPolicy,
            ConsentEvent,
            ConsentEventKind,
            PolicyHandler,
        )

        policy = CategoryBlacklistPolicy(blocked_categories=[ConsentCategory.DATA_COLLECTION])
        handler = PolicyHandler()
        handler.add_policy(policy)

        # Grant a blocked category
        handler.handle(
            ConsentEvent(
                kind=ConsentEventKind.CONSENT_GRANTED,
                category=ConsentCategory.DATA_COLLECTION,
                granted=True,
            )
        )

        violations = handler.get_violations()
        assert len(violations) == 1
        assert "Blocked category" in violations[0].message


class TestEventAwareConsentManager:
    """Tests for event-aware consent manager."""

    def test_emits_grant_event(self):
        """Test that grant emits event."""
        from proxima.resources.hooks import (
            CallbackHandler,
            ConsentEventKind,
            EventAwareConsentManager,
        )

        received = []
        manager = EventAwareConsentManager()
        manager.event_bus.register(CallbackHandler(callback=lambda e: received.append(e)))

        manager.grant("event_test")

        # Filter for grant events
        grant_events = [e for e in received if e.kind == ConsentEventKind.CONSENT_GRANTED]
        assert len(grant_events) >= 1

    def test_emits_revoke_event(self):
        """Test that revoke emits event."""
        from proxima.resources.hooks import (
            CallbackHandler,
            ConsentEventKind,
            EventAwareConsentManager,
        )

        received = []
        manager = EventAwareConsentManager()
        manager.event_bus.register(CallbackHandler(callback=lambda e: received.append(e)))

        manager.grant("to_revoke")
        manager.revoke("to_revoke")

        revoke_events = [e for e in received if e.kind == ConsentEventKind.CONSENT_REVOKED]
        assert len(revoke_events) >= 1


class TestConsentIntegration:
    """Integration tests combining audit and hooks."""

    def test_full_audit_workflow(self):
        """Test complete audit workflow."""
        from proxima.resources.audit import (
            create_audited_manager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_audited_manager(
                audit_path=Path(tmpdir) / "audit.log",
                consent_storage_path=Path(tmpdir) / "consent.json",
            )

            # Perform operations
            manager.grant("topic1")
            manager.grant("topic2")
            manager.deny("topic3")
            manager.revoke("topic1")

            # Check audit log
            log = manager.audit_log
            stats = log.get_statistics()

            assert stats.grants >= 2
            assert stats.denials >= 1
            assert stats.revocations >= 1

    def test_hooks_with_audit(self):
        """Test combining hooks with audit logging."""
        from proxima.resources.audit import AuditedConsentManager
        from proxima.resources.hooks import (
            ConsentEvent,
            ConsentEventBus,
            ConsentEventKind,
            MetricsHandler,
        )

        # Create both audit and event tracking
        manager = AuditedConsentManager()
        bus = ConsentEventBus()
        metrics = MetricsHandler()
        bus.register(metrics)

        # Hook audit events to bus
        def forward_to_bus(event):
            bus.emit(
                ConsentEvent(
                    kind=(
                        ConsentEventKind.CONSENT_GRANTED
                        if event.granted
                        else ConsentEventKind.CONSENT_DENIED
                    ),
                    topic=event.topic,
                    granted=event.granted,
                )
            )

        manager.audit_log.add_listener(forward_to_bus)

        # Perform operations
        manager.grant("combined1")
        manager.grant("combined2")

        # Check both systems recorded
        audit_events = manager.audit_log.get_all()
        assert len(audit_events) >= 2

        assert metrics.total_events >= 2


class TestConsentEdgeCases:
    """Edge case tests for consent audit/hooks."""

    def test_concurrent_audit_writes(self):
        """Test thread-safe audit logging."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        errors = []

        def write_events():
            try:
                for i in range(50):
                    log.log(AuditEventType.CONSENT_GRANTED, topic=f"concurrent-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_events) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(log.get_all()) == 250

    def test_handler_error_isolation(self):
        """Test that handler errors don't affect other handlers."""
        from proxima.resources.hooks import (
            CallbackHandler,
            ConsentEvent,
            ConsentEventBus,
            ConsentEventKind,
        )

        bus = ConsentEventBus()
        good_received = []

        def bad_handler(e):
            raise RuntimeError("Intentional error")

        def good_handler(e):
            good_received.append(e)

        bus.register(CallbackHandler(callback=bad_handler))
        bus.register(CallbackHandler(callback=good_handler))

        bus.emit(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED))

        # Good handler should still receive event
        assert len(good_received) == 1

    def test_empty_query_results(self):
        """Test querying with no matching results."""
        from proxima.resources.audit import AuditEventType, AuditLog

        log = AuditLog()
        log.log(AuditEventType.CONSENT_GRANTED, topic="exists")

        results = log.query().by_topic("nonexistent").execute()
        assert len(results) == 0

        first = log.query().by_topic("nonexistent").first()
        assert first is None

    def test_disabled_handler(self):
        """Test that disabled handlers don't receive events."""
        from proxima.resources.hooks import (
            ConsentEvent,
            ConsentEventBus,
            ConsentEventKind,
            MetricsHandler,
        )

        bus = ConsentEventBus()
        handler = MetricsHandler()
        handler.enabled = False
        bus.register(handler)

        bus.emit(ConsentEvent(kind=ConsentEventKind.CONSENT_GRANTED, granted=True))

        # Handler should not have processed the event
        assert handler.total_events == 0
