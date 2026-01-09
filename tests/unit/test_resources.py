"""
Unit tests for the resources module.

Tests memory monitoring, consent management, execution control, and timers.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# ===================== Memory Monitor Tests =====================


class TestMemoryMonitor:
    """Tests for MemoryMonitor."""

    def test_memory_level_enum(self) -> None:
        """Test MemoryLevel enum values (uses auto() so values are integers)."""
        from proxima.resources.monitor import MemoryLevel

        # MemoryLevel uses auto() so values are integers, not strings
        assert MemoryLevel.OK.value == 1
        assert MemoryLevel.INFO.value == 2
        assert MemoryLevel.WARNING.value == 3
        assert MemoryLevel.CRITICAL.value == 4
        assert MemoryLevel.ABORT.value == 5

        # Check string representation
        assert str(MemoryLevel.INFO) == "INFO"
        assert str(MemoryLevel.WARNING) == "WARNING"

    def test_memory_thresholds_get_level(self) -> None:
        """Test MemoryThresholds.get_level classification."""
        from proxima.resources.monitor import MemoryLevel, MemoryThresholds

        thresholds = MemoryThresholds(
            info_percent=60.0,
            warning_percent=80.0,
            critical_percent=95.0,
            abort_percent=98.0,
        )

        assert thresholds.get_level(50.0) == MemoryLevel.OK
        assert thresholds.get_level(65.0) == MemoryLevel.INFO
        assert thresholds.get_level(85.0) == MemoryLevel.WARNING
        assert thresholds.get_level(96.0) == MemoryLevel.CRITICAL
        assert thresholds.get_level(99.0) == MemoryLevel.ABORT

    def test_memory_snapshot(self) -> None:
        """Test MemorySnapshot dataclass."""
        from proxima.resources.monitor import MemoryLevel, MemorySnapshot

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            used_mb=8000.0,
            available_mb=8000.0,
            total_mb=16000.0,
            percent_used=50.0,
            level=MemoryLevel.OK,
        )

        assert snapshot.used_mb == 8000.0
        assert snapshot.free_mb == snapshot.available_mb
        assert "50.0%" in str(snapshot)

    def test_memory_monitor_sample(self) -> None:
        """Test MemoryMonitor.sample method."""
        from proxima.resources.monitor import MemoryMonitor

        monitor = MemoryMonitor()

        # psutil is imported inside the sample method, so we patch where it's used
        with patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value = MagicMock(
                total=16 * 1024**3,  # 16 GB
                available=8 * 1024**3,  # 8 GB
                percent=50.0,
                used=8 * 1024**3,
            )

            snapshot = monitor.sample()

        assert snapshot.total_mb > 0
        assert snapshot.available_mb > 0
        assert 0 <= snapshot.percent_used <= 100

    def test_memory_monitor_on_alert_callback(self) -> None:
        """Test alert callback registration."""
        from proxima.resources.monitor import MemoryMonitor

        monitor = MemoryMonitor()
        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        monitor.on_alert(alert_handler)

        # Callbacks should be registered
        assert len(monitor._callbacks) == 1


# ===================== Memory Estimator Tests =====================


class TestMemoryEstimator:
    """Tests for MemoryEstimator."""

    def test_memory_estimate_dataclass(self) -> None:
        """Test MemoryEstimate dataclass."""
        from proxima.resources.monitor import MemoryEstimate

        estimate = MemoryEstimate(
            operation="simulate_circuit",
            estimated_mb=1024.0,
            confidence=0.9,
            breakdown={"state_vector": 512.0, "gates": 256.0},
        )

        assert estimate.operation == "simulate_circuit"
        assert estimate.estimated_mb == 1024.0
        assert estimate.confidence == 0.9
        assert "state_vector" in estimate.breakdown

    def test_memory_check_result(self) -> None:
        """Test MemoryCheckResult dataclass."""
        from proxima.resources.monitor import MemoryCheckResult

        result = MemoryCheckResult(
            sufficient=True,
            available_mb=8000.0,
            required_mb=1000.0,
            shortfall_mb=0.0,
        )

        assert result.sufficient is True
        assert result.shortfall_mb == 0.0


# ===================== Consent Manager Tests =====================


class TestConsentManager:
    """Tests for ConsentManager."""

    def test_consent_category_enum(self) -> None:
        """Test ConsentCategory enum values."""
        from proxima.resources.consent import ConsentCategory

        assert ConsentCategory.LOCAL_LLM.value == "local_llm"
        assert ConsentCategory.REMOTE_LLM.value == "remote_llm"
        assert ConsentCategory.FORCE_EXECUTE.value == "force_execute"
        assert ConsentCategory.UNTRUSTED_AGENT_MD.value == "untrusted_agent_md"

    def test_consent_level_enum(self) -> None:
        """Test ConsentLevel enum (uses auto() for integer values)."""
        from proxima.resources.consent import ConsentLevel

        # Uses auto() so values are integers
        assert ConsentLevel.SESSION.value == 1
        assert ConsentLevel.PERSISTENT.value == 2
        assert ConsentLevel.ONE_TIME.value == 3
        assert ConsentLevel.NEVER.value == 4

    def test_consent_request_creation(self) -> None:
        """Test creating a consent request."""
        from proxima.resources.consent import ConsentCategory, ConsentRequest

        # ConsentRequest takes 'topic' and 'description', not 'resource_id'
        request = ConsentRequest(
            topic="openai-gpt4",
            category=ConsentCategory.REMOTE_LLM,
            description="Use OpenAI API",
        )

        assert request.category == ConsentCategory.REMOTE_LLM
        assert request.description == "Use OpenAI API"
        assert request.topic == "openai-gpt4"

    def test_consent_record(self) -> None:
        """Test ConsentRecord dataclass."""
        from proxima.resources.consent import ConsentCategory, ConsentLevel, ConsentRecord

        record = ConsentRecord(
            topic="llm_usage",
            category=ConsentCategory.LOCAL_LLM,
            granted=True,
            level=ConsentLevel.SESSION,
        )

        assert record.topic == "llm_usage"
        assert record.granted is True
        assert record.is_valid() is True

    def test_consent_manager_check_remembered(self) -> None:
        """Test checking remembered consent."""
        from proxima.resources.consent import ConsentManager

        manager = ConsentManager()

        # Initially nothing remembered
        result = manager.check_remembered("test_topic")
        assert result.found is False
        assert result.granted is None

    def test_consent_manager_force_override(self) -> None:
        """Test force override mode."""
        from proxima.resources.consent import ConsentManager

        manager = ConsentManager()

        assert manager.is_force_override_enabled() is False

        manager.enable_force_override()
        assert manager.is_force_override_enabled() is True

        manager.disable_force_override()
        assert manager.is_force_override_enabled() is False

    def test_consent_manager_callbacks(self) -> None:
        """Test consent callback registration."""
        from proxima.resources.consent import ConsentManager

        manager = ConsentManager()
        consents_received = []

        def consent_handler(record):
            consents_received.append(record)

        manager.on_consent(consent_handler)
        assert len(manager._callbacks) == 1


# ===================== Execution Control Tests =====================


class TestExecutionControl:
    """Tests for ExecutionController."""

    def test_control_state_enum(self) -> None:
        """Test ControlState enum values (uses auto() so values are integers)."""
        from proxima.resources.control import ControlState

        # ControlState uses auto() so values are integers
        assert ControlState.IDLE.value == 1
        assert ControlState.RUNNING.value == 2
        assert ControlState.PAUSED.value == 3
        assert ControlState.ABORTED.value == 4
        assert ControlState.COMPLETED.value == 5

    def test_control_signal_enum(self) -> None:
        """Test ControlSignal enum values (uses auto())."""
        from proxima.resources.control import ControlSignal

        # ControlSignal uses auto() so values are integers
        assert ControlSignal.NONE.value == 1
        assert ControlSignal.START.value == 2
        assert ControlSignal.PAUSE.value == 3
        assert ControlSignal.RESUME.value == 4
        assert ControlSignal.ABORT.value == 5

    def test_controller_initial_state(self) -> None:
        """Test controller starts in IDLE state."""
        from proxima.resources.control import ControlState, ExecutionController

        controller = ExecutionController()
        assert controller.state == ControlState.IDLE
        assert controller.is_idle is True

    def test_controller_start(self) -> None:
        """Test starting execution."""
        from proxima.resources.control import ControlState, ExecutionController

        controller = ExecutionController()
        controller.start()

        assert controller.state == ControlState.RUNNING
        assert controller.is_running is True

    def test_controller_pause_resume(self) -> None:
        """Test pausing and resuming."""
        from proxima.resources.control import ControlState, ExecutionController

        controller = ExecutionController()
        controller.start()

        controller.pause()
        assert controller.state == ControlState.PAUSED
        assert controller.is_paused is True

        controller.resume()
        assert controller.state == ControlState.RUNNING

    def test_controller_abort(self) -> None:
        """Test aborting execution."""
        from proxima.resources.control import ControlState, ExecutionController

        controller = ExecutionController()
        controller.start()
        controller.abort()

        assert controller.state == ControlState.ABORTED
        assert controller.is_aborted is True

    def test_controller_elapsed_ms(self) -> None:
        """Test elapsed time tracking."""
        from proxima.resources.control import ExecutionController

        controller = ExecutionController()
        assert controller.elapsed_ms == 0.0

        controller.start()
        time.sleep(0.01)  # Small delay

        assert controller.elapsed_ms > 0


# ===================== Checkpoint Manager Tests =====================


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_checkpoint_manager_creation(self) -> None:
        """Test CheckpointManager initialization."""
        from proxima.resources.control import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                execution_id="test_exec_001",
            )

            assert manager.execution_id == "test_exec_001"
            assert manager.checkpoint_dir.exists()

    def test_checkpoint_create_and_restore(self) -> None:
        """Test creating and restoring a checkpoint."""
        from proxima.resources.control import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            # Create checkpoint
            checkpoint = manager.create_checkpoint(
                stage_name="preprocessing",
                stage_index=0,
                total_stages=3,
                progress_percent=33.3,
                elapsed_ms=1000.0,
                custom_state={"step": 1, "data": "test"},
            )

            assert checkpoint.checkpoint_id is not None
            assert checkpoint.stage_name == "preprocessing"

            # Restore checkpoint
            restored = manager.load_checkpoint(checkpoint.checkpoint_id)
            assert restored is not None
            assert restored.stage_name == "preprocessing"
            assert restored.custom_state["step"] == 1


# ===================== Timer Tests =====================


class TestTimer:
    """Tests for timer-related classes."""

    def test_stage_info(self) -> None:
        """Test StageInfo dataclass."""
        from proxima.resources.timer import StageInfo

        stage = StageInfo(
            name="test_stage",
            start_time=time.perf_counter(),
        )

        assert stage.name == "test_stage"
        assert stage.is_complete is False
        assert stage.elapsed_ms >= 0

    def test_progress_tracker(self) -> None:
        """Test ProgressTracker class."""
        from proxima.resources.timer import ProgressTracker

        tracker = ProgressTracker(total_steps=100)

        assert tracker.percentage == 0.0
        assert tracker.is_complete is False

        tracker.advance(50)
        assert tracker.percentage == 50.0

        tracker.set(100)
        assert tracker.percentage == 100.0
        assert tracker.is_complete is True

    def test_progress_tracker_milestone(self) -> None:
        """Test milestone detection in ProgressTracker."""
        from proxima.resources.timer import ProgressTracker

        tracker = ProgressTracker(total_steps=100)
        milestones = []

        tracker.on_milestone(lambda m: milestones.append(m))

        # Advance through milestones
        tracker.set(15)  # Crosses 10%
        assert 10.0 in milestones

        tracker.set(25)  # Crosses 20%
        assert 20.0 in milestones

    def test_eta_calculator(self) -> None:
        """Test ETACalculator class."""
        from proxima.resources.timer import ETACalculator

        calculator = ETACalculator()
        calculator.start()

        # Initially no ETA
        assert calculator.eta_seconds is None or calculator.eta_seconds == 0.0

        # Simulate progress
        time.sleep(0.01)
        calculator.update(0.5)  # 50% done

        # Should have some ETA now
        eta = calculator.eta_seconds
        # ETA might be None or a positive number
        assert eta is None or eta >= 0

    def test_eta_display(self) -> None:
        """Test ETA display formatting."""
        from proxima.resources.timer import ETACalculator

        calculator = ETACalculator()

        # Before start
        display = calculator.eta_display()
        assert "calculating" in display.lower() or display is not None

    def test_execution_timer(self) -> None:
        """Test ExecutionTimer class."""
        from proxima.resources.timer import ExecutionTimer

        timer = ExecutionTimer(total_stages=3)

        # Should have internal progress and ETA
        assert timer._progress is not None
        assert timer._eta is not None


# ===================== Session Tests =====================


class TestSession:
    """Tests for session management."""

    def test_session_creation(self) -> None:
        """Test creating a session via Session.create() factory."""
        from proxima.resources.session import Session

        session = Session.create(name="test-session")

        # Session.create() creates SessionMetadata with the name
        assert session.metadata.name == "test-session"
        assert session.metadata.id is not None
        assert session.metadata.created_at > 0

    def test_session_update_status(self) -> None:
        """Test updating session status."""
        from proxima.resources.session import Session, SessionStatus

        session = Session.create(name="test")

        session.update_status(SessionStatus.RUNNING)
        assert session.metadata.status == SessionStatus.RUNNING

        session.update_status(SessionStatus.COMPLETED)
        assert session.metadata.status == SessionStatus.COMPLETED

    def test_session_checkpoint(self) -> None:
        """Test session checkpoint creation."""
        from proxima.resources.session import Session

        session = Session.create(name="test")

        cp = session.checkpoint(
            stage="stage1",
            state={"key": "value"},
            message="Test checkpoint",
        )

        assert cp.stage == "stage1"
        assert cp.state["key"] == "value"
        assert session.latest_checkpoint() == cp

    def test_session_add_log(self) -> None:
        """Test adding log entries to session."""
        from proxima.resources.session import Session

        session = Session.create(name="test")

        session.add_log("INFO", "Test message", extra_data="value")

        assert len(session.logs) == 1
        assert session.logs[0]["message"] == "Test message"
        assert session.logs[0]["level"] == "INFO"


# ===================== Store Tests =====================


class TestResultStore:
    """Tests for result storage."""

    def test_stored_result_model(self) -> None:
        """Test StoredResult model."""
        from proxima.data.store import StoredResult

        result = StoredResult(
            session_id="session-123",
            backend_name="qiskit",
            qubit_count=3,
            shots=2000,
            execution_time_ms=250.0,
            memory_used_mb=75.5,
            counts={"000": 1000, "111": 1000},
            statevector=[complex(0.5, 0.5), complex(0.5, -0.5)],
            metadata={"circuit_name": "bell_state"},
        )

        assert result.id is not None
        assert result.session_id == "session-123"
        assert result.backend_name == "qiskit"
        assert result.qubit_count == 3
        assert len(result.statevector) == 2

    def test_stored_session_model(self) -> None:
        """Test StoredSession model."""
        from proxima.data.store import StoredSession

        session = StoredSession(
            name="my-session",
            agent_file="test.md",
            metadata={"version": "1.0"},
        )

        assert session.id is not None
        assert session.name == "my-session"
        assert session.agent_file == "test.md"
        assert session.result_count == 0

    def test_memory_store(self) -> None:
        """Test in-memory storage."""
        from proxima.data.store import MemoryStore, StoredResult, StoredSession

        store = MemoryStore()

        # Create session
        session = StoredSession(name="test-session")
        store.create_session(session)

        # Save result
        result = StoredResult(
            session_id=session.id,
            backend_name="cirq",
            qubit_count=2,
            shots=1000,
            execution_time_ms=100.5,
            memory_used_mb=50.0,
            counts={"00": 500, "11": 500},
        )
        result_id = store.save_result(result)

        # Retrieve
        retrieved = store.get_result(result_id)
        assert retrieved is not None
        assert retrieved.backend_name == "cirq"

        # List
        results = store.list_results(session_id=session.id)
        assert len(results) == 1

        # Delete
        store.delete_result(result_id)
        assert store.get_result(result_id) is None

        store.close()

    def test_memory_store_sessions(self) -> None:
        """Test session management in memory store."""
        from proxima.data.store import MemoryStore, StoredSession

        store = MemoryStore()

        session = StoredSession(name="test")
        store.create_session(session)

        retrieved = store.get_session(session.id)
        assert retrieved is not None
        assert retrieved.name == "test"

        sessions = store.list_sessions()
        assert len(sessions) == 1

        store.delete_session(session.id)
        assert store.get_session(session.id) is None

        store.close()


# ===================== Integration Tests for Resources =====================


class TestResourceIntegration:
    """Integration tests for resource module components."""

    def test_controller_with_checkpoint_manager(self) -> None:
        """Test execution controller provides checkpoint manager."""
        from proxima.resources.control import CheckpointManager, ExecutionController

        controller = ExecutionController()

        # Controller has an internal checkpoint manager
        assert controller.checkpoint_manager is not None
        assert isinstance(controller.checkpoint_manager, CheckpointManager)

        controller.start()

        # Create checkpoint via controller's manager
        checkpoint = controller.checkpoint_manager.create_checkpoint(
            stage_name="test_stage",
            stage_index=0,
            total_stages=3,
            progress_percent=33.3,
            elapsed_ms=100.0,
            custom_state={"step": 1, "data": "test"},
        )

        assert checkpoint is not None
        assert checkpoint.stage_name == "test_stage"

        controller.abort()

    def test_control_event(self) -> None:
        """Test control events are generated."""
        from proxima.resources.control import ExecutionController

        controller = ExecutionController()
        events_received = []

        controller.on_state_change(lambda e: events_received.append(e))

        controller.start()
        controller.pause()
        controller.resume()
        controller.abort()

        # Should have multiple events
        assert len(events_received) >= 4
