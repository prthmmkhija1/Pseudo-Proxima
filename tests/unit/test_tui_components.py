"""Test TUI components: styles, modals, controllers, widgets.

Comprehensive tests for the TUI infrastructure modules.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Check if textual is available
try:
    import importlib.util

    HAS_TEXTUAL = importlib.util.find_spec("textual") is not None
except ImportError:
    HAS_TEXTUAL = False

pytestmark = pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")


# ============================ STYLES MODULE ============================


class TestTheme:
    """Test Theme enum."""

    def test_theme_values(self):
        from proxima.tui.styles import Theme

        assert Theme.DARK.value == "dark"
        assert Theme.LIGHT.value == "light"
        assert Theme.OCEAN.value == "ocean"
        assert Theme.FOREST.value == "forest"
        assert Theme.SUNSET.value == "sunset"

    def test_theme_count(self):
        from proxima.tui.styles import Theme

        assert len(Theme) == 5


class TestColorPalette:
    """Test ColorPalette dataclass."""

    def test_palette_default(self):
        from proxima.tui.styles import ColorPalette

        palette = ColorPalette()
        assert palette.primary is not None
        assert palette.background is not None
        assert palette.error is not None

    def test_palette_to_css_vars(self):
        from proxima.tui.styles import ColorPalette

        palette = ColorPalette()
        css = palette.to_css_vars()
        assert "$primary" in css
        assert "$background" in css
        assert "$error" in css


class TestStyleFunctions:
    """Test style utility functions."""

    def test_get_palette(self):
        from proxima.tui.styles import Theme, get_palette

        palette = get_palette(Theme.DARK)
        assert palette is not None
        assert palette.primary is not None

    def test_get_palette_all_themes(self):
        from proxima.tui.styles import Theme, get_palette

        for theme in Theme:
            palette = get_palette(theme)
            assert palette is not None

    def test_build_theme_css(self):
        from proxima.tui.styles import ColorPalette, build_theme_css

        palette = ColorPalette()
        css = build_theme_css(palette)
        assert isinstance(css, str)
        assert len(css) > 100

    def test_get_css(self):
        from proxima.tui.styles import get_css

        css = get_css()
        assert isinstance(css, str)
        assert len(css) > 100


class TestPredefinedPalettes:
    """Test predefined color palettes."""

    def test_dark_palette(self):
        from proxima.tui.styles import DARK_PALETTE

        assert DARK_PALETTE.primary is not None
        assert DARK_PALETTE.background is not None

    def test_light_palette(self):
        from proxima.tui.styles import LIGHT_PALETTE

        assert LIGHT_PALETTE.primary is not None
        assert LIGHT_PALETTE.background is not None


# ============================ MODALS MODULE ============================


class TestDialogResult:
    """Test DialogResult enum."""

    def test_dialog_result_values(self):
        from proxima.tui.modals import DialogResult

        assert DialogResult.CONFIRMED.value == "confirmed"
        assert DialogResult.CANCELLED.value == "cancelled"
        assert DialogResult.DISMISSED.value == "dismissed"


class TestModalResponse:
    """Test ModalResponse dataclass."""

    def test_modal_response_creation(self):
        from proxima.tui.modals import DialogResult, ModalResponse

        response = ModalResponse(result=DialogResult.CONFIRMED, data={"value": "test"})
        assert response.result == DialogResult.CONFIRMED
        assert response.data["value"] == "test"

    def test_modal_response_confirmed_property(self):
        from proxima.tui.modals import DialogResult, ModalResponse

        response = ModalResponse(result=DialogResult.CONFIRMED)
        assert response.confirmed is True
        assert response.cancelled is False

    def test_modal_response_cancelled_property(self):
        from proxima.tui.modals import DialogResult, ModalResponse

        response = ModalResponse(result=DialogResult.CANCELLED)
        assert response.confirmed is False
        assert response.cancelled is True


class TestConfirmModal:
    """Test ConfirmModal class."""

    def test_confirm_modal_creation(self):
        from proxima.tui.modals import ConfirmModal

        modal = ConfirmModal(title="Confirm?", message="Are you sure?")
        assert modal._title == "Confirm?"
        assert modal._message == "Are you sure?"


class TestInputModal:
    """Test InputModal class."""

    def test_input_modal_creation(self):
        from proxima.tui.modals import InputModal

        modal = InputModal(
            title="Enter Name",
            label="Your name:",
            default_value="John",
            placeholder="Enter name...",
        )
        assert modal._title == "Enter Name"
        assert modal._label == "Your name:"
        assert modal._default_value == "John"


class TestChoiceModal:
    """Test ChoiceModal class."""

    def test_choice_modal_creation(self):
        from proxima.tui.modals import ChoiceModal

        choices = [("opt1", "Option 1"), ("opt2", "Option 2")]
        modal = ChoiceModal(
            title="Select Option",
            message="Choose one:",
            choices=choices,
        )
        assert modal._title == "Select Option"
        assert len(modal._choices) == 2


class TestProgressModal:
    """Test ProgressModal class."""

    def test_progress_modal_creation(self):
        from proxima.tui.modals import ProgressModal

        modal = ProgressModal(title="Loading...", message="Please wait")
        assert modal._title == "Loading..."
        assert modal._message == "Please wait"


class TestErrorModal:
    """Test ErrorModal class."""

    def test_error_modal_creation(self):
        from proxima.tui.modals import ErrorModal

        modal = ErrorModal(
            title="Error",
            message="Something went wrong",
            details="Stack trace here",
        )
        assert modal._title == "Error"
        assert modal._details == "Stack trace here"


class TestConsentModal:
    """Test ConsentModal class."""

    def test_consent_modal_creation(self):
        from proxima.tui.modals import ConsentModal

        modal = ConsentModal(
            title="Consent Required",
            operation="delete data",
            implications=["Data will be lost", "This cannot be undone"],
        )
        assert modal._title == "Consent Required"
        assert modal._operation == "delete data"
        assert len(modal._implications) == 2


class TestFormModal:
    """Test FormModal class."""

    def test_form_field_creation(self):
        from proxima.tui.modals import FormField

        field = FormField(
            key="username",
            label="Username",
            default="",
            placeholder="Enter username",
            required=True,
        )
        assert field.key == "username"
        assert field.required is True

    def test_form_modal_creation(self):
        from proxima.tui.modals import FormField, FormModal

        fields = [
            FormField(key="name", label="Name"),
            FormField(key="email", label="Email"),
        ]
        modal = FormModal(title="User Form", fields=fields)
        assert modal._title == "User Form"
        assert len(modal._fields) == 2


# ============================ CONTROLLERS MODULE ============================


class TestEventType:
    """Test EventType enum."""

    def test_event_types_exist(self):
        from proxima.tui.controllers import EventType

        assert EventType.SCREEN_CHANGED is not None
        assert EventType.DATA_LOADED is not None
        assert EventType.EXECUTION_STARTED is not None
        assert EventType.BACKEND_ERROR is not None


class TestTUIEvent:
    """Test TUIEvent dataclass."""

    def test_event_creation(self):
        from proxima.tui.controllers import EventType, TUIEvent

        event = TUIEvent(
            event_type=EventType.DATA_LOADED,
            data={"items": [1, 2, 3]},
            source="test",
        )
        assert event.event_type == EventType.DATA_LOADED
        assert event.data["items"] == [1, 2, 3]
        assert event.source == "test"


class TestEventBus:
    """Test EventBus class."""

    def test_event_bus_singleton(self):
        from proxima.tui.controllers import EventBus

        bus1 = EventBus()
        bus2 = EventBus()
        assert bus1 is bus2


class TestStateStore:
    """Test StateStore class."""

    def test_state_store_get_missing(self):
        from proxima.tui.controllers import StateStore

        store = StateStore()
        assert store.get("nonexistent_key_xyz") is None
        assert store.get("nonexistent_key_xyz", "default") == "default"


class TestNavigationController:
    """Test NavigationController class."""

    def test_navigation_controller_creation(self):
        from proxima.tui.controllers import NavigationController

        nav = NavigationController()
        assert nav.current_screen == "dashboard"


class TestExecutionStatus:
    """Test ExecutionStatus enum."""

    def test_execution_status_values(self):
        from proxima.tui.controllers import ExecutionStatus

        assert ExecutionStatus.IDLE.value == "idle"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"


class TestExecutionState:
    """Test ExecutionState dataclass."""

    def test_execution_state_creation(self):
        from proxima.tui.controllers import ExecutionState, ExecutionStatus

        state = ExecutionState(
            id="test-123",
            status=ExecutionStatus.RUNNING,
            progress=50.0,
            current_stage="Processing",
        )
        assert state.id == "test-123"
        assert state.status == ExecutionStatus.RUNNING
        assert state.progress == 50.0


class TestExecutionController:
    """Test ExecutionController class."""

    def test_execution_controller_creation(self):
        from proxima.tui.controllers import ExecutionController

        ec = ExecutionController()
        assert ec is not None

    def test_execution_controller_singleton(self):
        from proxima.tui.controllers import ExecutionController

        ec1 = ExecutionController()
        ec2 = ExecutionController()
        assert ec1 is ec2


# ============================ WIDGETS MODULE ============================


class TestStatusLevel:
    """Test StatusLevel enum."""

    def test_status_level_values(self):
        from proxima.tui.widgets import StatusLevel

        assert StatusLevel.OK.value == "ok"
        assert StatusLevel.INFO.value == "info"
        assert StatusLevel.WARNING.value == "warning"
        assert StatusLevel.ERROR.value == "error"


class TestStatusItem:
    """Test StatusItem dataclass."""

    def test_status_item_creation(self):
        from proxima.tui.widgets import StatusItem, StatusLevel

        item = StatusItem(
            label="Version",
            value="1.0.0",
            level=StatusLevel.INFO,
        )
        assert item.label == "Version"
        assert item.value == "1.0.0"
        assert item.level == StatusLevel.INFO


class TestStatusIndicator:
    """Test StatusIndicator widget."""

    def test_status_indicator_creation(self):
        from proxima.tui.widgets import StatusIndicator

        indicator = StatusIndicator(status="success", label="OK")
        assert indicator.status == "success"

    def test_status_indicator_icons(self):
        from proxima.tui.widgets import StatusIndicator

        assert StatusIndicator.ICONS["success"] == "✓"
        assert StatusIndicator.ICONS["error"] == "✗"
        assert StatusIndicator.ICONS["warning"] == "⚠"


class TestLogEntry:
    """Test LogEntry dataclass."""

    def test_log_entry_creation(self):
        import time

        from proxima.tui.widgets import LogEntry

        entry = LogEntry(
            timestamp=time.time(),
            level="info",
            message="Test message",
            component="test",
        )
        assert entry.level == "info"
        assert entry.message == "Test message"

    def test_log_entry_format(self):
        from proxima.tui.widgets import LogEntry

        entry = LogEntry(
            timestamp=1704067200.0,
            level="error",
            message="Error!",
        )
        assert entry.format_level() == "ERROR"


class TestBackendStatus:
    """Test BackendStatus enum."""

    def test_backend_status_values(self):
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.CONNECTED.value == "connected"
        assert BackendStatus.DISCONNECTED.value == "disconnected"
        assert BackendStatus.ERROR.value == "error"


class TestBackendInfo:
    """Test BackendInfo dataclass."""

    def test_backend_info_creation(self):
        from proxima.tui.widgets import BackendInfo, BackendStatus

        info = BackendInfo(
            name="Qiskit Aer",
            backend_type="simulator",
            status=BackendStatus.CONNECTED,
            total_executions=100,
            avg_latency_ms=45.2,
        )
        assert info.name == "Qiskit Aer"
        assert info.status == BackendStatus.CONNECTED
        assert info.total_executions == 100


class TestWidgetCreation:
    """Test widget instantiation."""

    def test_status_panel_creation(self):
        from proxima.tui.widgets import StatusItem, StatusLevel, StatusPanel

        items = [
            StatusItem("Test", "Value", StatusLevel.OK),
        ]
        panel = StatusPanel(title="Status", items=items)
        assert panel._title == "Status"

    def test_log_viewer_creation(self):
        from proxima.tui.widgets import LogViewer

        viewer = LogViewer(max_entries=500, auto_scroll=True)
        assert viewer._max_entries == 500
        assert viewer._auto_scroll is True

    def test_backend_card_creation(self):
        from proxima.tui.widgets import BackendCard, BackendInfo, BackendStatus

        info = BackendInfo("Test", "local", BackendStatus.CONNECTED)
        card = BackendCard(backend=info)
        assert card._backend == info

    def test_results_table_creation(self):
        from proxima.tui.widgets import ResultsTable

        table = ResultsTable()
        assert table._results == []

    def test_execution_timer_creation(self):
        from proxima.tui.widgets import ExecutionTimer

        timer = ExecutionTimer(label="Timer")
        assert timer._label == "Timer"
        assert timer.elapsed == 0.0

    def test_metric_display_creation(self):
        from proxima.tui.widgets import MetricDisplay

        metric = MetricDisplay(label="Count", value="42", unit="x")
        assert metric._label == "Count"
        assert metric.value == "42"

    def test_execution_progress_creation(self):
        from proxima.tui.widgets import ExecutionProgress

        progress = ExecutionProgress(title="Progress")
        assert progress._title == "Progress"
        assert progress.progress == 0.0

    def test_help_modal_creation(self):
        from proxima.tui.widgets import HelpModal

        modal = HelpModal()
        assert modal is not None

    def test_config_input_creation(self):
        from proxima.tui.widgets import ConfigInput

        input_widget = ConfigInput(
            key="test_key",
            label="Test Label",
            value="default",
        )
        assert input_widget._key == "test_key"
        assert input_widget._label == "Test Label"

    def test_config_toggle_creation(self):
        from proxima.tui.widgets import ConfigToggle

        toggle = ConfigToggle(
            key="feature",
            label="Enable Feature",
            value=True,
        )
        assert toggle._key == "feature"
        assert toggle._value is True

    def test_execution_card_creation(self):
        import time

        from proxima.tui.widgets import ExecutionCard

        card = ExecutionCard(
            execution_id="exec-001",
            backend="qiskit",
            status="success",
            duration_ms=123.45,
            timestamp=time.time(),
        )
        assert card._id == "exec-001"
        assert card._status == "success"

    def test_progress_bar_creation(self):
        from proxima.tui.widgets import ProgressBar

        bar = ProgressBar(label="Loading", total=100.0)
        assert bar._label == "Loading"
        assert bar._total == 100.0


# ============================ INTEGRATION TESTS ============================


class TestTUIIntegration:
    """Integration tests for TUI components."""

    def test_all_exports_available(self):
        """Verify all exports in __init__.py are importable."""
        from proxima.tui import (
            # Widgets
            ConfirmModal,
            DashboardScreen,
            # Controllers
            EventBus,
            ProximaApp,
            StatusPanel,
            Theme,
        )

        # Verify key items exist
        assert ProximaApp is not None
        assert DashboardScreen is not None
        assert StatusPanel is not None
        assert EventBus is not None
        assert ConfirmModal is not None
        assert Theme is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
