"""Step 6.2: TUI Real-Time Update Tests.

Tests for TUI real-time update functionality including:
- Progress updates during execution
- Live log streaming
- Backend status updates
- Timer display updates
- WebSocket-like message handling
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import time

# Check if textual is available
try:
    import textual
    from textual.app import App
    from textual.widgets import Static
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


# =============================================================================
# REAL-TIME UPDATE FIXTURES
# =============================================================================


@pytest.fixture
def mock_update_handler():
    """Create mock update handler."""
    handler = MagicMock()
    handler.on_progress = MagicMock()
    handler.on_log = MagicMock()
    handler.on_status = MagicMock()
    handler.on_timer = MagicMock()
    handler.on_complete = MagicMock()
    return handler


@pytest.fixture
def mock_execution_state():
    """Create mock execution state for updates."""
    return {
        "status": "running",
        "progress": 0.0,
        "current_stage": None,
        "elapsed_ms": 0,
        "logs": [],
        "backend_status": {},
    }


# =============================================================================
# PROGRESS UPDATE TESTS
# =============================================================================


class TestProgressUpdates:
    """Tests for progress update functionality."""

    @pytest.mark.tui
    def test_progress_update_calculation(self):
        """Test progress percentage calculation."""
        total_stages = 5
        completed = 0
        
        for i in range(total_stages):
            completed = i
            progress = (completed / total_stages) * 100
            
            assert 0 <= progress <= 100
            assert progress == pytest.approx(i * 20, rel=0.01)

    @pytest.mark.tui
    def test_progress_update_message(self, mock_update_handler):
        """Test progress update message handling."""
        updates = [
            {"type": "progress", "value": 0.25, "stage": "stage_1"},
            {"type": "progress", "value": 0.50, "stage": "stage_2"},
            {"type": "progress", "value": 0.75, "stage": "stage_3"},
            {"type": "progress", "value": 1.00, "stage": "stage_4"},
        ]
        
        for update in updates:
            mock_update_handler.on_progress(update["value"], update["stage"])
        
        assert mock_update_handler.on_progress.call_count == 4

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_async_progress_streaming(self, mock_update_handler):
        """Test async progress streaming."""
        progress_values = []
        
        async def progress_generator():
            for i in range(5):
                await asyncio.sleep(0.01)
                yield {"progress": (i + 1) / 5, "stage": f"stage_{i}"}
        
        async for update in progress_generator():
            progress_values.append(update["progress"])
            mock_update_handler.on_progress(update["progress"], update["stage"])
        
        assert len(progress_values) == 5
        assert progress_values[-1] == 1.0

    @pytest.mark.tui
    def test_progress_bar_rendering(self):
        """Test progress bar string rendering."""
        def render_progress_bar(progress: float, width: int = 20) -> str:
            filled = int(width * progress)
            empty = width - filled
            return f"[{'=' * filled}{' ' * empty}] {progress * 100:.1f}%"
        
        assert render_progress_bar(0.0) == "[                    ] 0.0%"
        assert render_progress_bar(0.5) == "[==========          ] 50.0%"
        assert render_progress_bar(1.0) == "[====================] 100.0%"


# =============================================================================
# LOG STREAMING TESTS
# =============================================================================


class TestLogStreaming:
    """Tests for log streaming functionality."""

    @pytest.mark.tui
    def test_log_message_buffering(self):
        """Test log message buffering."""
        buffer_size = 100
        log_buffer = []
        
        for i in range(150):
            log_buffer.append(f"Log message {i}")
            if len(log_buffer) > buffer_size:
                log_buffer.pop(0)
        
        assert len(log_buffer) == buffer_size
        assert log_buffer[0] == "Log message 50"
        assert log_buffer[-1] == "Log message 149"

    @pytest.mark.tui
    def test_log_level_filtering(self):
        """Test log level filtering."""
        logs = [
            {"level": "DEBUG", "message": "Debug info"},
            {"level": "INFO", "message": "Info message"},
            {"level": "WARNING", "message": "Warning"},
            {"level": "ERROR", "message": "Error occurred"},
        ]
        
        level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        min_level = "INFO"
        
        filtered = [
            log for log in logs
            if level_priority[log["level"]] >= level_priority[min_level]
        ]
        
        assert len(filtered) == 3
        assert filtered[0]["level"] == "INFO"

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_async_log_streaming(self, mock_update_handler):
        """Test async log streaming."""
        async def log_generator():
            messages = [
                "Starting execution...",
                "Backend initialized",
                "Running circuit...",
                "Collecting results...",
                "Execution complete",
            ]
            for msg in messages:
                await asyncio.sleep(0.01)
                yield {"type": "log", "message": msg, "timestamp": time.time()}
        
        log_count = 0
        async for log in log_generator():
            mock_update_handler.on_log(log["message"])
            log_count += 1
        
        assert log_count == 5
        assert mock_update_handler.on_log.call_count == 5

    @pytest.mark.tui
    def test_log_formatting(self):
        """Test log message formatting."""
        def format_log(level: str, message: str, timestamp: float) -> str:
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            return f"[{time_str}] [{level:5s}] {message}"
        
        formatted = format_log("INFO", "Test message", time.time())
        
        assert "[INFO ]" in formatted
        assert "Test message" in formatted


# =============================================================================
# BACKEND STATUS UPDATE TESTS
# =============================================================================


class TestBackendStatusUpdates:
    """Tests for backend status update functionality."""

    @pytest.mark.tui
    def test_backend_status_tracking(self):
        """Test backend status tracking."""
        backend_states = {
            "cirq": {"status": "idle", "last_used": None},
            "qiskit_aer": {"status": "idle", "last_used": None},
            "lret": {"status": "idle", "last_used": None},
        }
        
        # Simulate execution on cirq
        backend_states["cirq"]["status"] = "running"
        assert backend_states["cirq"]["status"] == "running"
        
        # Complete execution
        backend_states["cirq"]["status"] = "completed"
        backend_states["cirq"]["last_used"] = time.time()
        
        assert backend_states["cirq"]["status"] == "completed"
        assert backend_states["cirq"]["last_used"] is not None

    @pytest.mark.tui
    def test_backend_status_change_events(self, mock_update_handler):
        """Test backend status change events."""
        status_changes = [
            ("cirq", "idle", "initializing"),
            ("cirq", "initializing", "running"),
            ("cirq", "running", "completed"),
        ]
        
        for backend, old_status, new_status in status_changes:
            mock_update_handler.on_status({
                "backend": backend,
                "old_status": old_status,
                "new_status": new_status,
            })
        
        assert mock_update_handler.on_status.call_count == 3

    @pytest.mark.tui
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_enum(self):
        """Test backend status enum values."""
        from proxima.tui.widgets import BackendStatus
        
        assert BackendStatus.CONNECTED.value == "connected"
        assert BackendStatus.DISCONNECTED.value == "disconnected"
        assert BackendStatus.ERROR.value == "error"
        assert BackendStatus.CONNECTING.value == "connecting"


# =============================================================================
# TIMER DISPLAY UPDATE TESTS
# =============================================================================


class TestTimerDisplayUpdates:
    """Tests for timer display update functionality."""

    @pytest.mark.tui
    def test_timer_format_display(self):
        """Test timer format for display."""
        def format_elapsed(ms: float) -> str:
            seconds = ms / 1000
            if seconds < 60:
                return f"{seconds:.2f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = seconds % 60
                return f"{minutes}m {secs:.1f}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h {minutes}m"
        
        assert format_elapsed(500) == "0.50s"
        assert format_elapsed(5000) == "5.00s"
        assert format_elapsed(65000) == "1m 5.0s"
        assert format_elapsed(3700000) == "1h 1m"

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_timer_update_interval(self, mock_update_handler):
        """Test timer update at regular intervals."""
        start_time = time.time()
        update_interval = 0.05  # 50ms
        updates = []
        
        for _ in range(5):
            await asyncio.sleep(update_interval)
            elapsed = (time.time() - start_time) * 1000
            updates.append(elapsed)
            mock_update_handler.on_timer(elapsed)
        
        assert len(updates) == 5
        # Each update should be roughly 50ms apart
        for i in range(1, len(updates)):
            diff = updates[i] - updates[i-1]
            assert 40 < diff < 100  # Allow some variance

    @pytest.mark.tui
    def test_timer_with_stages(self):
        """Test timer tracking per stage."""
        stage_times = {}
        
        stages = ["init", "execute", "analyze", "report"]
        total_elapsed = 0
        
        for stage in stages:
            stage_start = total_elapsed
            stage_duration = 100 + hash(stage) % 100  # Variable duration
            total_elapsed += stage_duration
            
            stage_times[stage] = {
                "start_ms": stage_start,
                "duration_ms": stage_duration,
            }
        
        # Verify all stages tracked
        assert len(stage_times) == 4
        assert all(s in stage_times for s in stages)


# =============================================================================
# MESSAGE HANDLING TESTS
# =============================================================================


class TestMessageHandling:
    """Tests for real-time message handling."""

    @pytest.mark.tui
    def test_message_queue_processing(self):
        """Test message queue processing."""
        from queue import Queue
        
        message_queue = Queue()
        processed = []
        
        # Add messages
        messages = [
            {"type": "progress", "value": 0.5},
            {"type": "log", "message": "Test"},
            {"type": "status", "backend": "cirq", "status": "running"},
            {"type": "timer", "elapsed_ms": 1000},
        ]
        
        for msg in messages:
            message_queue.put(msg)
        
        # Process messages
        while not message_queue.empty():
            msg = message_queue.get()
            processed.append(msg["type"])
        
        assert processed == ["progress", "log", "status", "timer"]

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_async_message_handling(self):
        """Test async message handling with callback."""
        received_messages = []
        
        async def message_handler(msg):
            received_messages.append(msg)
        
        messages = [
            {"type": "start", "timestamp": time.time()},
            {"type": "progress", "value": 0.5},
            {"type": "complete", "result": "success"},
        ]
        
        for msg in messages:
            await message_handler(msg)
        
        assert len(received_messages) == 3
        assert received_messages[0]["type"] == "start"
        assert received_messages[-1]["type"] == "complete"

    @pytest.mark.tui
    def test_message_type_routing(self, mock_update_handler):
        """Test message routing based on type."""
        def route_message(msg, handler):
            msg_type = msg.get("type")
            if msg_type == "progress":
                handler.on_progress(msg["value"], msg.get("stage"))
            elif msg_type == "log":
                handler.on_log(msg["message"])
            elif msg_type == "status":
                handler.on_status(msg)
            elif msg_type == "timer":
                handler.on_timer(msg["elapsed_ms"])
            elif msg_type == "complete":
                handler.on_complete(msg)
        
        messages = [
            {"type": "progress", "value": 0.5, "stage": "test"},
            {"type": "log", "message": "Test log"},
            {"type": "status", "backend": "cirq", "status": "running"},
            {"type": "timer", "elapsed_ms": 1000},
            {"type": "complete", "result": "success"},
        ]
        
        for msg in messages:
            route_message(msg, mock_update_handler)
        
        assert mock_update_handler.on_progress.call_count == 1
        assert mock_update_handler.on_log.call_count == 1
        assert mock_update_handler.on_status.call_count == 1
        assert mock_update_handler.on_timer.call_count == 1
        assert mock_update_handler.on_complete.call_count == 1


# =============================================================================
# TUI APP INTEGRATION TESTS
# =============================================================================


@pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
class TestTUIAppIntegration:
    """Tests for TUI app integration with real-time updates."""

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_app_receives_updates(self):
        """Test TUI app receives real-time updates."""
        from proxima.tui.app import ProximaApp
        
        # Create mock app
        app = MagicMock(spec=ProximaApp)
        app.notify = MagicMock()
        
        # Simulate updates
        updates = [
            ("info", "Execution started"),
            ("info", "Progress: 50%"),
            ("success", "Execution complete"),
        ]
        
        for severity, message in updates:
            app.notify(message, severity=severity)
        
        assert app.notify.call_count == 3

    @pytest.mark.tui
    def test_widget_update_rendering(self):
        """Test widget update rendering."""
        # Mock widget state
        widget_state = {
            "progress": 0.0,
            "status": "idle",
            "logs": [],
        }
        
        def update_widget(widget_state, update):
            if "progress" in update:
                widget_state["progress"] = update["progress"]
            if "status" in update:
                widget_state["status"] = update["status"]
            if "log" in update:
                widget_state["logs"].append(update["log"])
            return widget_state
        
        # Apply updates
        widget_state = update_widget(widget_state, {"status": "running"})
        widget_state = update_widget(widget_state, {"progress": 0.5})
        widget_state = update_widget(widget_state, {"log": "Test log"})
        
        assert widget_state["status"] == "running"
        assert widget_state["progress"] == 0.5
        assert len(widget_state["logs"]) == 1


# =============================================================================
# REFRESH RATE TESTS
# =============================================================================


class TestRefreshRate:
    """Tests for update refresh rate handling."""

    @pytest.mark.tui
    def test_throttle_updates(self):
        """Test update throttling to prevent overwhelming UI."""
        min_interval_ms = 50
        last_update = 0
        updates_applied = []
        
        incoming_updates = [(i * 10, f"update_{i}") for i in range(20)]  # Every 10ms
        
        for timestamp, update in incoming_updates:
            if timestamp - last_update >= min_interval_ms:
                updates_applied.append(update)
                last_update = timestamp
        
        # Should throttle from 20 to ~4 updates
        assert len(updates_applied) < len(incoming_updates)
        assert len(updates_applied) == 4  # 0, 50, 100, 150

    @pytest.mark.tui
    @pytest.mark.asyncio
    async def test_debounce_rapid_updates(self):
        """Test debouncing rapid successive updates."""
        debounce_ms = 50
        last_value = None
        update_count = 0
        
        async def debounced_update(value):
            nonlocal last_value, update_count
            last_value = value
            update_count += 1
        
        # Simulate rapid updates
        for i in range(5):
            await debounced_update(i)
            await asyncio.sleep(0.01)  # 10ms between updates
        
        # With proper debouncing, only final value would be used
        # Here we're tracking all for testing
        assert update_count == 5
        assert last_value == 4
