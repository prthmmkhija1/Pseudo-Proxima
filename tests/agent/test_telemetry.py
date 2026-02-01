"""Unit tests for Telemetry and Metrics modules.

Phase 10: Integration & Testing

Tests cover:
- Telemetry data collection
- Metric recording
- Historical metrics storage
- Alert management
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.mock_agent import MockTelemetry, create_mock_telemetry


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_telemetry():
    """Create mock telemetry instance."""
    return create_mock_telemetry()


@pytest.fixture
def sample_llm_metrics():
    """Sample LLM metrics data."""
    return {
        "request_count": 10,
        "total_tokens": 5000,
        "avg_latency_ms": 250.0,
        "error_count": 1,
    }


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics data."""
    return {
        "cpu_percent": 45.5,
        "memory_mb": 512.0,
        "operations_per_second": 100.0,
    }


# =============================================================================
# MOCK TELEMETRY TESTS
# =============================================================================

class TestMockTelemetry:
    """Tests for MockTelemetry."""
    
    def test_telemetry_creation(self, mock_telemetry):
        """Test creating telemetry instance."""
        assert len(mock_telemetry.events) == 0
        assert len(mock_telemetry.metrics) == 0
    
    def test_record_event(self, mock_telemetry):
        """Test recording an event."""
        mock_telemetry.record_event(
            event_type="command_executed",
            data={"command": "echo hello", "exit_code": 0}
        )
        
        assert len(mock_telemetry.events) == 1
        assert mock_telemetry.events[0]["type"] == "command_executed"
    
    def test_record_multiple_events(self, mock_telemetry):
        """Test recording multiple events."""
        for i in range(5):
            mock_telemetry.record_event(
                event_type=f"event_{i}",
                data={"index": i}
            )
        
        assert len(mock_telemetry.events) == 5
    
    def test_record_metric(self, mock_telemetry):
        """Test recording a metric."""
        mock_telemetry.record_metric("latency_ms", 150.5)
        
        assert mock_telemetry.metrics["latency_ms"] == 150.5
    
    def test_get_events_filtered(self, mock_telemetry):
        """Test getting filtered events."""
        mock_telemetry.record_event("type_a", {"data": 1})
        mock_telemetry.record_event("type_b", {"data": 2})
        mock_telemetry.record_event("type_a", {"data": 3})
        
        type_a_events = mock_telemetry.get_events("type_a")
        
        assert len(type_a_events) == 2
    
    def test_get_all_events(self, mock_telemetry):
        """Test getting all events."""
        mock_telemetry.record_event("type_a", {})
        mock_telemetry.record_event("type_b", {})
        
        all_events = mock_telemetry.get_events()
        
        assert len(all_events) == 2
    
    def test_clear_telemetry(self, mock_telemetry):
        """Test clearing telemetry data."""
        mock_telemetry.record_event("test", {})
        mock_telemetry.record_metric("test", 1.0)
        
        mock_telemetry.clear()
        
        assert len(mock_telemetry.events) == 0
        assert len(mock_telemetry.metrics) == 0


# =============================================================================
# LLM METRICS TESTS
# =============================================================================

class TestLLMMetrics:
    """Tests for LLM metrics tracking."""
    
    def test_track_request_count(self, mock_telemetry, sample_llm_metrics):
        """Test tracking LLM request count."""
        mock_telemetry.record_metric(
            "llm.request_count",
            sample_llm_metrics["request_count"]
        )
        
        assert mock_telemetry.metrics["llm.request_count"] == 10
    
    def test_track_token_usage(self, mock_telemetry, sample_llm_metrics):
        """Test tracking token usage."""
        mock_telemetry.record_metric(
            "llm.total_tokens",
            sample_llm_metrics["total_tokens"]
        )
        
        assert mock_telemetry.metrics["llm.total_tokens"] == 5000
    
    def test_track_latency(self, mock_telemetry, sample_llm_metrics):
        """Test tracking latency."""
        mock_telemetry.record_metric(
            "llm.avg_latency_ms",
            sample_llm_metrics["avg_latency_ms"]
        )
        
        assert mock_telemetry.metrics["llm.avg_latency_ms"] == 250.0
    
    def test_llm_request_event(self, mock_telemetry):
        """Test recording LLM request event."""
        mock_telemetry.record_event(
            event_type="llm_request",
            data={
                "model": "gpt-4",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "latency_ms": 200,
            }
        )
        
        event = mock_telemetry.events[0]
        assert event["type"] == "llm_request"
        assert event["data"]["model"] == "gpt-4"


# =============================================================================
# PERFORMANCE METRICS TESTS
# =============================================================================

class TestPerformanceMetrics:
    """Tests for performance metrics tracking."""
    
    def test_track_cpu_usage(self, mock_telemetry, sample_performance_metrics):
        """Test tracking CPU usage."""
        mock_telemetry.record_metric(
            "perf.cpu_percent",
            sample_performance_metrics["cpu_percent"]
        )
        
        assert mock_telemetry.metrics["perf.cpu_percent"] == 45.5
    
    def test_track_memory_usage(self, mock_telemetry, sample_performance_metrics):
        """Test tracking memory usage."""
        mock_telemetry.record_metric(
            "perf.memory_mb",
            sample_performance_metrics["memory_mb"]
        )
        
        assert mock_telemetry.metrics["perf.memory_mb"] == 512.0
    
    def test_track_operations_rate(self, mock_telemetry, sample_performance_metrics):
        """Test tracking operations rate."""
        mock_telemetry.record_metric(
            "perf.ops_per_second",
            sample_performance_metrics["operations_per_second"]
        )
        
        assert mock_telemetry.metrics["perf.ops_per_second"] == 100.0


# =============================================================================
# TERMINAL METRICS TESTS
# =============================================================================

class TestTerminalMetrics:
    """Tests for terminal metrics tracking."""
    
    def test_command_execution_event(self, mock_telemetry):
        """Test recording command execution event."""
        mock_telemetry.record_event(
            event_type="terminal.command",
            data={
                "command": "cmake -B build",
                "exit_code": 0,
                "duration_ms": 1500,
                "session_id": "term-1",
            }
        )
        
        event = mock_telemetry.events[0]
        assert event["data"]["exit_code"] == 0
        assert event["data"]["duration_ms"] == 1500
    
    def test_terminal_session_metrics(self, mock_telemetry):
        """Test terminal session metrics."""
        mock_telemetry.record_metric("terminal.active_sessions", 3)
        mock_telemetry.record_metric("terminal.total_commands", 50)
        
        assert mock_telemetry.metrics["terminal.active_sessions"] == 3
        assert mock_telemetry.metrics["terminal.total_commands"] == 50


# =============================================================================
# HISTORICAL METRICS TESTS
# =============================================================================

class MockHistoricalMetrics:
    """Mock historical metrics storage."""
    
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
    
    def add_record(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None
    ):
        """Add a historical record."""
        self.records.append({
            "name": metric_name,
            "value": value,
            "timestamp": timestamp or time.time(),
        })
    
    def get_records(
        self,
        metric_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get historical records."""
        results = [r for r in self.records if r["name"] == metric_name]
        
        if start_time:
            results = [r for r in results if r["timestamp"] >= start_time]
        if end_time:
            results = [r for r in results if r["timestamp"] <= end_time]
        
        return results
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for a metric."""
        records = self.get_records(metric_name)
        if not records:
            return 0.0
        return sum(r["value"] for r in records) / len(records)


class TestHistoricalMetrics:
    """Tests for historical metrics storage."""
    
    def test_add_record(self):
        """Test adding historical record."""
        history = MockHistoricalMetrics()
        history.add_record("latency", 100.0)
        
        assert len(history.records) == 1
    
    def test_get_records(self):
        """Test getting historical records."""
        history = MockHistoricalMetrics()
        history.add_record("latency", 100.0)
        history.add_record("latency", 150.0)
        history.add_record("throughput", 50.0)
        
        latency_records = history.get_records("latency")
        
        assert len(latency_records) == 2
    
    def test_get_average(self):
        """Test calculating average."""
        history = MockHistoricalMetrics()
        history.add_record("latency", 100.0)
        history.add_record("latency", 200.0)
        history.add_record("latency", 300.0)
        
        avg = history.get_average("latency")
        
        assert avg == 200.0
    
    def test_time_range_filter(self):
        """Test filtering by time range."""
        history = MockHistoricalMetrics()
        
        now = time.time()
        history.add_record("metric", 1.0, now - 100)
        history.add_record("metric", 2.0, now - 50)
        history.add_record("metric", 3.0, now)
        
        recent = history.get_records("metric", start_time=now - 60)
        
        assert len(recent) == 2


# =============================================================================
# ALERT TESTS
# =============================================================================

class MockAlertManager:
    """Mock alert manager for testing."""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, float] = {}
    
    def set_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a metric."""
        self.thresholds[metric_name] = threshold
    
    def check_value(self, metric_name: str, value: float) -> Optional[Dict]:
        """Check if value exceeds threshold."""
        if metric_name in self.thresholds:
            if value > self.thresholds[metric_name]:
                alert = {
                    "metric": metric_name,
                    "value": value,
                    "threshold": self.thresholds[metric_name],
                    "timestamp": time.time(),
                }
                self.alerts.append(alert)
                return alert
        return None
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts."""
        return self.alerts


class TestAlertManager:
    """Tests for alert management."""
    
    def test_set_threshold(self):
        """Test setting alert threshold."""
        alerts = MockAlertManager()
        alerts.set_threshold("latency_ms", 500.0)
        
        assert alerts.thresholds["latency_ms"] == 500.0
    
    def test_no_alert_under_threshold(self):
        """Test no alert when under threshold."""
        alerts = MockAlertManager()
        alerts.set_threshold("latency_ms", 500.0)
        
        result = alerts.check_value("latency_ms", 250.0)
        
        assert result is None
        assert len(alerts.alerts) == 0
    
    def test_alert_over_threshold(self):
        """Test alert when over threshold."""
        alerts = MockAlertManager()
        alerts.set_threshold("latency_ms", 500.0)
        
        result = alerts.check_value("latency_ms", 750.0)
        
        assert result is not None
        assert result["value"] == 750.0
        assert len(alerts.alerts) == 1
    
    def test_multiple_alerts(self):
        """Test multiple alerts."""
        alerts = MockAlertManager()
        alerts.set_threshold("latency_ms", 500.0)
        alerts.set_threshold("error_rate", 0.05)
        
        alerts.check_value("latency_ms", 600.0)
        alerts.check_value("error_rate", 0.10)
        
        active = alerts.get_active_alerts()
        
        assert len(active) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
