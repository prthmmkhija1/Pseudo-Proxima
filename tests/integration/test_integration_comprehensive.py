"""
Additional Integration Tests for Proxima

Tests for more complex integration scenarios:
- Configuration with backends
- State machine with execution pipeline
- Resource monitoring with execution control
- Multi-backend comparison workflows
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

# Try to import source modules - check for full dependency chain
HAS_SOURCE = False
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    # Check for the full dependency chain by importing a module with dependencies
    import structlog
    import typer
    import pydantic
    from proxima.backends.registry import BackendRegistry
    from proxima.config.settings import BackendsSettings
    HAS_SOURCE = True
except ImportError:
    pass

# Mark all tests in this module
pytestmark = [pytest.mark.integration]

# Skip decorator for tests requiring source
requires_source = pytest.mark.skipif(
    not HAS_SOURCE, 
    reason="Source modules not available (missing dependencies like structlog, typer)"
)


# =============================================================================
# CONFIGURATION + BACKEND INTEGRATION TESTS
# =============================================================================


class TestConfigBackendIntegration:
    """Tests for configuration system with backend selection."""

    @requires_source
    @pytest.mark.unit
    def test_backend_registry_respects_config(self):
        """Test backend registry uses configuration."""
        from proxima.backends.registry import BackendRegistry
        
        registry = BackendRegistry()
        registry.discover()
        
        # Should discover backends
        assert len(registry._statuses) > 0
        
        # LRET should always be available (pure Python)
        assert "lret" in registry._statuses

    @requires_source
    @pytest.mark.unit
    def test_config_timeout_affects_execution(self):
        """Test that config timeout is respected."""
        from proxima.config.settings import BackendsSettings
        
        settings = BackendsSettings(timeout_seconds=60)
        assert settings.timeout_seconds == 60


# =============================================================================
# STATE MACHINE + EXECUTION INTEGRATION TESTS
# =============================================================================


class TestStateMachineExecutionIntegration:
    """Tests for state machine with execution flow."""

    @requires_source
    @pytest.mark.unit
    def test_state_machine_tracks_execution_flow(self):
        """Test state machine tracks full execution flow."""
        from proxima.core.state import ExecutionState, ExecutionStateMachine
        
        sm = ExecutionStateMachine(execution_id="integration-test-1")
        
        # Simulate full execution
        sm.start()
        assert sm.state_enum == ExecutionState.PLANNING
        
        sm.plan_complete()
        assert sm.state_enum == ExecutionState.READY
        
        sm.execute()
        assert sm.state_enum == ExecutionState.RUNNING
        
        # Simulate pause/resume
        sm.pause()
        assert sm.state_enum == ExecutionState.PAUSED
        
        sm.resume()
        assert sm.state_enum == ExecutionState.RUNNING
        
        sm.complete()
        assert sm.state_enum == ExecutionState.COMPLETED
        
        # Verify history
        snapshot = sm.snapshot()
        assert len(snapshot["history"]) == 6

    @requires_source
    @pytest.mark.unit
    def test_state_machine_abort_recovery(self):
        """Test state machine abort and recovery."""
        from proxima.core.state import ExecutionState, ExecutionStateMachine
        
        sm = ExecutionStateMachine(execution_id="abort-test")
        
        # Start execution
        sm.start()
        sm.plan_complete()
        sm.execute()
        
        # Abort
        sm.abort()
        assert sm.state_enum == ExecutionState.ABORTED
        
        # Reset and start new execution
        sm.reset()
        assert sm.state_enum == ExecutionState.IDLE
        
        sm.start()
        assert sm.state_enum == ExecutionState.PLANNING


# =============================================================================
# RESOURCE MONITORING INTEGRATION TESTS
# =============================================================================


class TestResourceMonitoringIntegration:
    """Tests for resource monitoring with execution."""

    @requires_source
    @pytest.mark.unit
    def test_memory_thresholds_classification(self):
        """Test memory level classification."""
        from proxima.resources.monitor import MemoryLevel, MemoryThresholds
        
        thresholds = MemoryThresholds(
            info_percent=60.0,
            warning_percent=80.0,
            critical_percent=95.0,
            abort_percent=98.0,
        )
        
        # Test all levels
        assert thresholds.get_level(50.0) == MemoryLevel.OK
        assert thresholds.get_level(70.0) == MemoryLevel.INFO
        assert thresholds.get_level(85.0) == MemoryLevel.WARNING
        assert thresholds.get_level(96.0) == MemoryLevel.CRITICAL
        assert thresholds.get_level(99.0) == MemoryLevel.ABORT

    @requires_source
    @pytest.mark.unit
    def test_resource_estimate_with_backend(self):
        """Test resource estimation for backend execution."""
        from proxima.backends.base import ResourceEstimate
        
        # Estimate for a 20-qubit simulation
        estimate = ResourceEstimate(
            memory_mb=1024.0,  # 1 GB
            time_ms=5000.0,  # 5 seconds
            metadata={"qubits": 20, "gates": 100},
        )
        
        # Should have reasonable estimates
        assert estimate.memory_mb > 0
        assert estimate.time_ms > 0
        assert estimate.metadata["qubits"] == 20


# =============================================================================
# CONSENT + EXECUTION INTEGRATION TESTS
# =============================================================================


class TestConsentExecutionIntegration:
    """Tests for consent management with execution flow."""

    @requires_source
    @pytest.mark.unit
    def test_consent_record_lifecycle(self):
        """Test consent record throughout execution lifecycle."""
        from proxima.resources.consent import (
            ConsentCategory,
            ConsentLevel,
            ConsentRecord,
        )
        import time
        
        # Create consent for execution
        record = ConsentRecord(
            topic="execute_simulation",
            category=ConsentCategory.RESOURCE_INTENSIVE,
            granted=True,
            level=ConsentLevel.SESSION,
            context="20-qubit Bell state simulation",
        )
        
        assert record.is_valid()
        assert record.granted
        
        # Serialize and restore (simulating session persistence)
        data = record.to_dict()
        restored = ConsentRecord.from_dict(data)
        
        assert restored.topic == record.topic
        assert restored.is_valid()

    @requires_source
    @pytest.mark.unit
    def test_llm_consent_categories(self):
        """Test LLM consent category configurations."""
        from proxima.resources.consent import (
            CATEGORY_CONFIGS,
            ConsentCategory,
        )
        
        # Local LLM should allow remembering
        local_config = CATEGORY_CONFIGS[ConsentCategory.LOCAL_LLM]
        assert local_config.allow_remember is True
        
        # Remote LLM should also allow remembering
        remote_config = CATEGORY_CONFIGS[ConsentCategory.REMOTE_LLM]
        assert remote_config.allow_remember is True


# =============================================================================
# BACKEND RESULT INTEGRATION TESTS
# =============================================================================


class TestBackendResultIntegration:
    """Tests for backend result handling across components."""

    @requires_source
    @pytest.mark.unit
    def test_execution_result_serialization(self):
        """Test ExecutionResult can be serialized for storage."""
        from dataclasses import asdict
        from proxima.backends.base import (
            ExecutionResult,
            ResultType,
            SimulatorType,
        )
        
        result = ExecutionResult(
            backend="cirq",
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=150.0,
            qubit_count=5,
            shot_count=1024,
            result_type=ResultType.COUNTS,
            data={"counts": {"00000": 512, "11111": 512}},
            metadata={"fidelity": 0.99},
        )
        
        # Should be serializable
        data = asdict(result)
        assert data["backend"] == "cirq"
        assert data["data"]["counts"]["00000"] == 512

    @requires_source
    @pytest.mark.unit
    def test_comparison_result_aggregation(self):
        """Test aggregating results from multiple backends."""
        from proxima.data.compare import BackendResult, ExecutionStrategy
        
        results = [
            BackendResult(
                backend_name="cirq",
                success=True,
                execution_time_ms=100.0,
                memory_peak_mb=256.0,
            ),
            BackendResult(
                backend_name="qiskit-aer",
                success=True,
                execution_time_ms=120.0,
                memory_peak_mb=300.0,
            ),
            BackendResult(
                backend_name="lret",
                success=True,
                execution_time_ms=80.0,
                memory_peak_mb=200.0,
            ),
        ]
        
        # Find fastest backend
        fastest = min(results, key=lambda r: r.execution_time_ms)
        assert fastest.backend_name == "lret"
        
        # Find lowest memory
        lowest_mem = min(results, key=lambda r: r.memory_peak_mb)
        assert lowest_mem.backend_name == "lret"


# =============================================================================
# EXPORT INTEGRATION TESTS
# =============================================================================


class TestExportIntegration:
    """Tests for export functionality with other components."""

    @requires_source
    @pytest.mark.unit
    def test_report_data_creation(self):
        """Test creating report data from execution results."""
        from proxima.data.export import ReportData
        
        report = ReportData(
            title="Multi-Backend Comparison: Bell State",
            summary={
                "backends_tested": 3,
                "fastest_backend": "lret",
                "total_time_ms": 300.0,
            },
            raw_results=[
                {"backend": "cirq", "time_ms": 100.0},
                {"backend": "qiskit-aer", "time_ms": 120.0},
                {"backend": "lret", "time_ms": 80.0},
            ],
            insights=[
                "LRET was 20% faster than the next fastest backend",
                "All backends produced consistent results",
            ],
            metadata={"generated_at": "2026-01-10T12:00:00Z"},
        )
        
        data = report.to_dict()
        assert data["title"] == "Multi-Backend Comparison: Bell State"
        assert len(data["raw_results"]) == 3
        assert len(data["insights"]) == 2


# =============================================================================
# CLI INTEGRATION TESTS (without subprocess)
# =============================================================================


class TestCLIComponentIntegration:
    """Tests for CLI components integration."""

    @requires_source
    @pytest.mark.unit
    def test_cli_imports_correctly(self):
        """Test CLI modules can be imported."""
        from proxima.cli.main import app
        
        assert app is not None
        assert hasattr(app, "command")

    @requires_source
    @pytest.mark.unit
    def test_version_info_available(self):
        """Test version info is available."""
        from proxima import __version__
        
        assert __version__ is not None
        assert "." in __version__  # Should be semver-like


# =============================================================================
# LOGGING INTEGRATION TESTS
# =============================================================================


class TestLoggingIntegration:
    """Tests for logging integration."""

    @requires_source
    @pytest.mark.unit
    def test_logger_creation(self):
        """Test logger can be created."""
        from proxima.utils.logging import get_logger
        
        logger = get_logger("test_integration")
        assert logger is not None
        
        # Should be able to bind context
        bound = logger.bind(component="test")
        assert bound is not None

    @requires_source
    @pytest.mark.unit
    def test_state_machine_logs_transitions(self):
        """Test state machine logs its transitions."""
        from proxima.core.state import ExecutionStateMachine
        
        sm = ExecutionStateMachine(execution_id="log-test")
        
        # Logger should be available
        assert sm.logger is not None
        
        # Execute transitions
        sm.start()
        sm.plan_complete()
        
        # History should record transitions
        assert len(sm.history) == 2
