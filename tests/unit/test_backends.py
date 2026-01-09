"""Step 6.2: Backend Tests - Testing backend adapters with mocks.

Backend tests focus on:
- Backend adapter functionality
- Mock backends for testing
- Backend status handling
"""

import pytest

# Check if textual is available for TUI-related tests
try:
    import textual

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


# =============================================================================
# BACKEND ADAPTER TESTS
# =============================================================================


class TestBackendAdapters:
    """Tests for backend adapters."""

    @pytest.mark.backend
    def test_mock_backend_name(self, mock_backend):
        """Test mock backend name."""
        assert mock_backend.name == "mock_backend"

    @pytest.mark.backend
    def test_mock_backend_type(self, mock_backend):
        """Test mock backend type."""
        assert mock_backend.backend_type == "simulator"

    @pytest.mark.backend
    def test_mock_backend_connected(self, mock_backend):
        """Test mock backend connection status."""
        assert mock_backend.is_connected() is True


# =============================================================================
# ASYNC BACKEND TESTS
# =============================================================================


class TestAsyncBackends:
    """Tests for async backend operations."""

    @pytest.mark.backend
    @pytest.mark.asyncio
    async def test_async_mock_backend_execute(self, async_mock_backend):
        """Test async mock backend execute."""
        result = await async_mock_backend.execute({"circuit": "async_test"})

        assert result is not None
        assert "status" in result


# =============================================================================
# BACKEND RESULT TESTS
# =============================================================================


class TestBackendResults:
    """Tests for backend result handling."""

    @pytest.mark.backend
    def test_backend_result_creation(self):
        """Test BackendResult creation."""
        from proxima.data.compare import BackendResult

        result = BackendResult(
            backend_name="test_backend",
            success=True,
            execution_time_ms=123.45,
            memory_peak_mb=256.0,
        )

        assert result.backend_name == "test_backend"
        assert result.success is True
        assert result.execution_time_ms == 123.45

    @pytest.mark.backend
    def test_backend_result_with_error(self):
        """Test BackendResult with error."""
        from proxima.data.compare import BackendResult

        result = BackendResult(
            backend_name="failed_backend",
            success=False,
            execution_time_ms=0.0,
            memory_peak_mb=0.0,
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"


# =============================================================================
# BACKEND STATUS TESTS
# =============================================================================


class TestBackendStatus:
    """Tests for backend status handling."""

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_connected(self):
        """Test connected status."""
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.CONNECTED.value == "connected"

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_disconnected(self):
        """Test disconnected status."""
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.DISCONNECTED.value == "disconnected"

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_error(self):
        """Test error status."""
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.ERROR.value == "error"

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_checking(self):
        """Test checking status."""
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.CHECKING.value == "checking"


# =============================================================================
# BACKEND INFO TESTS
# =============================================================================


class TestBackendInfo:
    """Tests for BackendInfo data class."""

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_info_creation(self):
        """Test BackendInfo creation."""
        from proxima.tui.widgets import BackendInfo, BackendStatus

        info = BackendInfo(
            name="Test Backend",
            backend_type="simulator",
            status=BackendStatus.CONNECTED,
            total_executions=100,
            avg_latency_ms=25.5,
            last_used="2024-01-01",
        )

        assert info.name == "Test Backend"
        assert info.backend_type == "simulator"
        assert info.status == BackendStatus.CONNECTED
        assert info.total_executions == 100

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_info_default_values(self):
        """Test BackendInfo default values."""
        from proxima.tui.widgets import BackendInfo, BackendStatus

        info = BackendInfo(
            name="Minimal",
            backend_type="test",
            status=BackendStatus.DISCONNECTED,
        )

        assert info.total_executions == 0
        assert info.avg_latency_ms == 0.0


# =============================================================================
# BACKEND CARD WIDGET TESTS
# =============================================================================


class TestBackendCardWidget:
    """Tests for BackendCard widget."""

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_card_creation(self):
        """Test BackendCard creation."""
        from proxima.tui.widgets import BackendCard, BackendInfo, BackendStatus

        info = BackendInfo(
            name="Card Test",
            backend_type="real",
            status=BackendStatus.CONNECTED,
        )

        card = BackendCard(info)

        assert card.backend == info
        assert card.backend.name == "Card Test"


# =============================================================================
# EXECUTION STRATEGY TESTS
# =============================================================================


class TestExecutionStrategy:
    """Tests for backend execution strategies."""

    @pytest.mark.backend
    def test_sequential_strategy(self):
        """Test sequential execution strategy."""
        from proxima.data.compare import ExecutionStrategy

        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"

    @pytest.mark.backend
    def test_parallel_strategy(self):
        """Test parallel execution strategy."""
        from proxima.data.compare import ExecutionStrategy

        assert ExecutionStrategy.PARALLEL.value == "parallel"

    @pytest.mark.backend
    def test_adaptive_strategy(self):
        """Test adaptive execution strategy."""
        from proxima.data.compare import ExecutionStrategy

        assert ExecutionStrategy.ADAPTIVE.value == "adaptive"
