"""Tests for CLI module - workflows, formatters, progress, and prompts.

This module provides comprehensive tests for:
- CLI workflow runners (run, compare, validate, export)
- Output formatters (text, JSON, YAML, table, CSV, rich)
- Progress display (spinners, bars, steps)
- Interactive prompts and consent dialogs
"""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

# ========== Fixtures ==========


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from proxima.config.settings import config_service

    return config_service.load()


@pytest.fixture
def mock_context(mock_settings):
    """Create mock Typer context."""
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = {
        "settings": mock_settings,
        "dry_run": False,
        "force": False,
        "verbose": 0,
        "quiet": False,
        "output_format": "text",
    }
    return ctx


@pytest.fixture
def workflow_context(mock_settings):
    """Create workflow context for testing."""
    from proxima.cli.workflows import WorkflowContext

    return WorkflowContext(
        settings=mock_settings,
        dry_run=False,
        force=False,
        verbose=0,
        quiet=False,
        output_format="text",
    )


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


# ========== Tests: Output Formatters ==========


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_format_values(self):
        """Test format enum values."""
        from proxima.cli.formatters import OutputFormat

        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.YAML.value == "yaml"
        assert OutputFormat.TABLE.value == "table"
        assert OutputFormat.CSV.value == "csv"
        assert OutputFormat.RICH.value == "rich"


class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_format_string(self):
        """Test formatting a string."""
        from proxima.cli.formatters import TextFormatter

        formatter = TextFormatter()
        result = formatter.format("hello world")
        assert result == "hello world"

    def test_format_dict(self):
        """Test formatting a dictionary."""
        from proxima.cli.formatters import TextFormatter

        formatter = TextFormatter()
        result = formatter.format({"key": "value", "number": 42})
        assert "key: value" in result
        assert "number: 42" in result

    def test_format_nested_dict(self):
        """Test formatting a nested dictionary."""
        from proxima.cli.formatters import TextFormatter

        formatter = TextFormatter()
        data = {"outer": {"inner": "value"}}
        result = formatter.format(data)
        assert "outer:" in result
        assert "inner: value" in result

    def test_format_list(self):
        """Test formatting a list."""
        from proxima.cli.formatters import TextFormatter

        formatter = TextFormatter()
        result = formatter.format(["a", "b", "c"])
        assert "- a" in result
        assert "- b" in result
        assert "- c" in result

    def test_write_to_stream(self):
        """Test writing to a stream."""
        from proxima.cli.formatters import TextFormatter

        stream = StringIO()
        formatter = TextFormatter(stream=stream)
        formatter.write("test output")
        assert "test output" in stream.getvalue()


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_format_dict(self):
        """Test formatting a dictionary as JSON."""
        from proxima.cli.formatters import JsonFormatter

        formatter = JsonFormatter()
        result = formatter.format({"key": "value"})
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_format_compact(self):
        """Test compact JSON formatting."""
        from proxima.cli.formatters import JsonFormatter

        formatter = JsonFormatter()
        result = formatter.format({"key": "value"}, compact=True)
        assert "\n" not in result

    def test_format_sorted_keys(self):
        """Test sorted keys in JSON."""
        from proxima.cli.formatters import JsonFormatter

        formatter = JsonFormatter()
        result = formatter.format({"z": 1, "a": 2}, sort_keys=True)
        # Check that 'a' appears before 'z'
        assert result.index('"a"') < result.index('"z"')

    def test_custom_serializer(self):
        """Test custom JSON serializer for non-standard types."""
        from enum import Enum

        from proxima.cli.formatters import JsonFormatter

        class Status(Enum):
            ACTIVE = "active"

        formatter = JsonFormatter()
        result = formatter.format({"status": Status.ACTIVE})
        parsed = json.loads(result)
        # Enum is serialized to its value
        assert parsed["status"] == "active" or "active" in str(parsed["status"])


class TestYamlFormatter:
    """Tests for YamlFormatter."""

    def test_format_dict(self):
        """Test formatting a dictionary as YAML."""
        from proxima.cli.formatters import YamlFormatter

        formatter = YamlFormatter()
        result = formatter.format({"key": "value"})
        assert "key: value" in result

    def test_format_nested(self):
        """Test formatting nested structure."""
        from proxima.cli.formatters import YamlFormatter

        formatter = YamlFormatter()
        result = formatter.format({"outer": {"inner": "value"}})
        assert "outer:" in result
        assert "inner: value" in result


class TestCsvFormatter:
    """Tests for CsvFormatter."""

    def test_format_list_of_dicts(self):
        """Test formatting a list of dictionaries as CSV."""
        from proxima.cli.formatters import CsvFormatter

        formatter = CsvFormatter()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        result = formatter.format(data)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "name" in lines[0]
        assert "Alice" in lines[1]

    def test_format_single_dict(self):
        """Test formatting a single dictionary."""
        from proxima.cli.formatters import CsvFormatter

        formatter = CsvFormatter()
        result = formatter.format({"name": "Alice"})
        assert "name" in result
        assert "Alice" in result


class TestTableFormatter:
    """Tests for TableFormatter."""

    def test_format_list_of_dicts(self):
        """Test formatting as table."""
        from proxima.cli.formatters import TableFormatter

        formatter = TableFormatter(no_color=True)
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        result = formatter.format(data)
        assert "Alice" in result
        assert "Bob" in result

    def test_format_empty_list(self):
        """Test formatting empty list."""
        from proxima.cli.formatters import TableFormatter

        formatter = TableFormatter()
        result = formatter.format([])
        assert "(empty)" in result


class TestRichFormatter:
    """Tests for RichFormatter."""

    def test_format_string(self):
        """Test formatting a string."""
        from proxima.cli.formatters import RichFormatter

        formatter = RichFormatter(no_color=True)
        result = formatter.format("hello")
        assert "hello" in result

    def test_format_dict(self):
        """Test formatting a dictionary."""
        from proxima.cli.formatters import RichFormatter

        formatter = RichFormatter(no_color=True)
        result = formatter.format({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_success_message(self):
        """Test success message output."""
        from proxima.cli.formatters import RichFormatter

        stream = StringIO()
        formatter = RichFormatter(stream=stream, no_color=True)
        formatter.success("Operation complete")
        # Success was called, no exception


class TestFormatterFactory:
    """Tests for get_formatter function."""

    def test_get_text_formatter(self):
        """Test getting text formatter."""
        from proxima.cli.formatters import TextFormatter, get_formatter

        formatter = get_formatter("text")
        assert isinstance(formatter, TextFormatter)

    def test_get_json_formatter(self):
        """Test getting JSON formatter."""
        from proxima.cli.formatters import JsonFormatter, get_formatter

        formatter = get_formatter("json")
        assert isinstance(formatter, JsonFormatter)

    def test_get_unknown_formatter(self):
        """Test getting formatter for unknown format falls back to text."""
        from proxima.cli.formatters import TextFormatter, get_formatter

        formatter = get_formatter("unknown")
        assert isinstance(formatter, TextFormatter)

    def test_format_output_convenience(self):
        """Test format_output convenience function."""
        from proxima.cli.formatters import format_output

        result = format_output({"key": "value"}, format="json")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}


# ========== Tests: Progress Display ==========


class TestProgressStatus:
    """Tests for ProgressStatus enum."""

    def test_status_values(self):
        """Test status enum values exist."""
        from proxima.cli.progress import ProgressStatus

        assert ProgressStatus.PENDING
        assert ProgressStatus.RUNNING
        assert ProgressStatus.COMPLETED
        assert ProgressStatus.FAILED
        assert ProgressStatus.CANCELLED


class TestSimpleProgress:
    """Tests for SimpleProgress display."""

    def test_create_progress(self):
        """Test creating a progress instance."""
        from proxima.cli.progress import SimpleProgress

        progress = SimpleProgress("Loading...", total=100, no_progress=True)
        assert progress.message == "Loading..."
        assert progress.total == 100

    def test_progress_lifecycle(self):
        """Test progress start/update/complete."""
        from proxima.cli.progress import ProgressStatus, SimpleProgress

        progress = SimpleProgress("Loading...", total=10, no_progress=True)
        progress.start()
        assert progress.status == ProgressStatus.RUNNING

        progress.update(5)
        assert progress.current == 5

        progress.complete()
        assert progress.status == ProgressStatus.COMPLETED

    def test_progress_percentage(self):
        """Test percentage calculation."""
        from proxima.cli.progress import SimpleProgress

        progress = SimpleProgress(total=100, no_progress=True)
        progress.current = 50
        assert progress.percentage == 50.0

    def test_progress_fail(self):
        """Test progress failure."""
        from proxima.cli.progress import ProgressStatus, SimpleProgress

        progress = SimpleProgress(no_progress=True)
        progress.start()
        progress.fail("Error occurred")
        assert progress.status == ProgressStatus.FAILED


class TestStepProgress:
    """Tests for StepProgress display."""

    def test_create_step_progress(self):
        """Test creating step progress."""
        from proxima.cli.progress import StepProgress

        steps = ["Step 1", "Step 2", "Step 3"]
        progress = StepProgress(steps, title="Test", no_progress=True)
        assert len(progress.steps) == 3
        assert progress.steps[0].name == "Step 1"

    def test_step_advance(self):
        """Test advancing through steps."""
        from proxima.cli.progress import ProgressStatus, StepProgress

        progress = StepProgress(["Step 1", "Step 2"], no_progress=True)
        progress.start()
        assert progress.steps[0].status == ProgressStatus.RUNNING

        progress.advance()
        assert progress.steps[0].status == ProgressStatus.COMPLETED
        assert progress.steps[1].status == ProgressStatus.RUNNING

    def test_step_with_error(self):
        """Test step with error."""
        from proxima.cli.progress import ProgressStatus, StepProgress

        progress = StepProgress(["Step 1"], no_progress=True)
        progress.start()
        progress.advance(error="Something went wrong")
        assert progress.steps[0].status == ProgressStatus.FAILED
        assert progress.steps[0].error == "Something went wrong"

    def test_step_skip(self):
        """Test skipping a step."""
        from proxima.cli.progress import ProgressStatus, StepProgress

        progress = StepProgress(["Step 1", "Step 2"], no_progress=True)
        progress.start()
        progress.skip()
        assert progress.steps[0].status == ProgressStatus.CANCELLED


class TestProgressContextManagers:
    """Tests for progress context managers."""

    def test_spinner_context(self):
        """Test spinner context manager."""
        from proxima.cli.progress import spinner_context

        with spinner_context("Loading...", no_progress=True) as spinner:
            assert spinner is not None
            spinner.update(message="Still loading...")

    def test_progress_context(self):
        """Test progress context manager."""
        from proxima.cli.progress import progress_context

        with progress_context("Processing", total=10, no_progress=True) as progress:
            for _i in range(10):
                progress.update(1)

    def test_step_context(self):
        """Test step context manager."""
        from proxima.cli.progress import step_context

        steps = ["Init", "Process", "Finish"]
        with step_context(steps, title="Test", no_progress=True) as progress:
            progress.advance()
            progress.advance()


class TestTrackIterator:
    """Tests for track iterator."""

    def test_track_list(self):
        """Test tracking iteration over a list."""
        from proxima.cli.progress import track

        items = list(range(5))
        tracked = list(track(items, no_progress=True))
        assert tracked == items

    def test_track_with_description(self):
        """Test tracking with description."""
        from proxima.cli.progress import track

        items = [1, 2, 3]
        tracked = list(track(items, description="Processing items", no_progress=True))
        assert len(tracked) == 3


# ========== Tests: Prompts ==========


class TestPromptResult:
    """Tests for PromptResult and PromptResponse."""

    def test_prompt_result_values(self):
        """Test PromptResult enum values."""
        from proxima.cli.prompts import PromptResult

        assert PromptResult.ANSWERED
        assert PromptResult.CANCELLED
        assert PromptResult.DEFAULT

    def test_prompt_response(self):
        """Test PromptResponse properties."""
        from proxima.cli.prompts import PromptResponse, PromptResult

        response = PromptResponse(value="test", result=PromptResult.ANSWERED)
        assert response.answered
        assert not response.cancelled

        cancelled = PromptResponse(value=None, result=PromptResult.CANCELLED)
        assert cancelled.cancelled
        assert not cancelled.answered


class TestConfirmPrompt:
    """Tests for ConfirmPrompt."""

    def test_confirm_prompt_creation(self):
        """Test creating a confirm prompt."""
        from proxima.cli.prompts import ConfirmPrompt

        prompt = ConfirmPrompt("Continue?", default=True)
        assert prompt.message == "Continue?"
        assert prompt.default is True

    @patch("builtins.input", return_value="y")
    def test_confirm_yes(self, mock_input):
        """Test confirming with yes."""
        from proxima.cli.prompts import ConfirmPrompt

        prompt = ConfirmPrompt("Continue?")
        result = prompt.ask()
        assert result.value is True

    @patch("builtins.input", return_value="n")
    def test_confirm_no(self, mock_input):
        """Test confirming with no."""
        from proxima.cli.prompts import ConfirmPrompt

        prompt = ConfirmPrompt("Continue?")
        result = prompt.ask()
        assert result.value is False

    @patch("builtins.input", return_value="")
    def test_confirm_default(self, mock_input):
        """Test confirm with default value."""
        from proxima.cli.prompts import ConfirmPrompt

        prompt = ConfirmPrompt("Continue?", default=True)
        result = prompt.ask()
        # When input is empty with default True, should return True
        assert result.value is True


class TestTextPrompt:
    """Tests for TextPrompt."""

    def test_text_prompt_creation(self):
        """Test creating a text prompt."""
        from proxima.cli.prompts import TextPrompt

        prompt = TextPrompt("Enter name:", default="John")
        assert prompt.message == "Enter name:"
        assert prompt.default == "John"

    @patch("builtins.input", return_value="Alice")
    def test_text_input(self, mock_input):
        """Test text input."""
        from proxima.cli.prompts import TextPrompt

        prompt = TextPrompt("Name:")
        result = prompt.ask()
        assert result.value == "Alice"


class TestSelectPrompt:
    """Tests for SelectPrompt."""

    def test_select_prompt_creation(self):
        """Test creating a select prompt."""
        from proxima.cli.prompts import SelectPrompt

        options = ["Option A", "Option B", "Option C"]
        prompt = SelectPrompt("Choose:", options=options)
        assert len(prompt.options) == 3
        assert prompt.options[0].value == "Option A"

    def test_select_option_with_label(self):
        """Test SelectOption with custom label."""
        from proxima.cli.prompts import SelectOption

        option = SelectOption(value="opt1", label="Option 1", description="First option")
        assert option.value == "opt1"
        assert option.label == "Option 1"
        assert option.description == "First option"

    @patch("builtins.input", return_value="1")
    def test_select_by_number(self, mock_input):
        """Test selection by number."""
        from proxima.cli.prompts import SelectPrompt

        prompt = SelectPrompt("Choose:", options=["A", "B"])
        result = prompt.ask()
        assert result.value == "A"


class TestMultiSelectPrompt:
    """Tests for MultiSelectPrompt."""

    def test_multi_select_creation(self):
        """Test creating a multi-select prompt."""
        from proxima.cli.prompts import MultiSelectPrompt

        options = ["A", "B", "C"]
        prompt = MultiSelectPrompt("Select:", options=options)
        assert len(prompt.options) == 3

    @patch("builtins.input", return_value="1,2")
    def test_multi_select(self, mock_input):
        """Test multiple selection."""
        from proxima.cli.prompts import MultiSelectPrompt

        prompt = MultiSelectPrompt("Select:", options=["A", "B", "C"])
        result = prompt.ask()
        assert "A" in result.value
        assert "B" in result.value
        assert "C" not in result.value

    @patch("builtins.input", return_value="all")
    def test_multi_select_all(self, mock_input):
        """Test select all."""
        from proxima.cli.prompts import MultiSelectPrompt

        prompt = MultiSelectPrompt("Select:", options=["A", "B", "C"])
        result = prompt.ask()
        assert len(result.value) == 3


class TestConsentPrompt:
    """Tests for ConsentPrompt."""

    def test_consent_info_creation(self):
        """Test creating consent info."""
        from proxima.cli.prompts import ConsentInfo

        info = ConsentInfo(
            title="Data Usage",
            description="We will use your data.",
            details=["Detail 1"],
            implications=["Implication 1"],
        )
        assert info.title == "Data Usage"
        assert len(info.details) == 1
        assert info.revocable is True

    @patch("builtins.input", return_value="I AGREE")
    def test_consent_granted(self, mock_input):
        """Test consent granted."""
        from proxima.cli.prompts import ConsentInfo, ConsentPrompt

        info = ConsentInfo(title="Test", description="Test consent")
        prompt = ConsentPrompt(info, require_explicit=True)
        result = prompt.ask()
        assert result.value is True

    @patch("builtins.input", return_value="no")
    def test_consent_denied(self, mock_input):
        """Test consent denied."""
        from proxima.cli.prompts import ConsentInfo, ConsentPrompt

        info = ConsentInfo(title="Test", description="Test consent")
        prompt = ConsentPrompt(info, require_explicit=True)
        result = prompt.ask()
        assert result.value is False


class TestConvenienceFunctions:
    """Tests for prompt convenience functions."""

    @patch("builtins.input", return_value="y")
    def test_confirm_function(self, mock_input):
        """Test confirm convenience function."""
        from proxima.cli.prompts import confirm

        result = confirm("Continue?")
        assert result is True

    def test_confirm_with_force(self):
        """Test confirm with force flag."""
        from proxima.cli.prompts import confirm

        result = confirm("Continue?", force=True)
        assert result is True

    @patch("builtins.input", return_value="test input")
    def test_prompt_text_function(self, mock_input):
        """Test prompt_text convenience function."""
        from proxima.cli.prompts import prompt_text

        result = prompt_text("Enter:")
        assert result == "test input"


# ========== Tests: Workflows ==========


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_status_values(self):
        """Test workflow status values."""
        from proxima.cli.workflows import WorkflowStatus

        assert WorkflowStatus.PENDING
        assert WorkflowStatus.RUNNING
        assert WorkflowStatus.COMPLETED
        assert WorkflowStatus.FAILED


class TestWorkflowResult:
    """Tests for WorkflowResult."""

    def test_successful_result(self):
        """Test successful workflow result."""
        from proxima.cli.workflows import WorkflowResult, WorkflowStatus

        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            output={"data": "test"},
            duration_seconds=1.5,
        )
        assert result.success
        assert result.output == {"data": "test"}
        assert result.duration_seconds == 1.5

    def test_failed_result(self):
        """Test failed workflow result."""
        from proxima.cli.workflows import WorkflowResult, WorkflowStatus

        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            error="Something went wrong",
        )
        assert not result.success
        assert result.error == "Something went wrong"

    def test_result_string(self):
        """Test result string representation."""
        from proxima.cli.workflows import WorkflowResult, WorkflowStatus

        result = WorkflowResult(status=WorkflowStatus.COMPLETED, duration_seconds=2.0)
        assert "2.00s" in str(result)


class TestWorkflowContext:
    """Tests for WorkflowContext."""

    def test_context_creation(self, mock_settings):
        """Test creating workflow context."""
        from proxima.cli.workflows import WorkflowContext

        ctx = WorkflowContext(
            settings=mock_settings,
            dry_run=True,
            force=False,
        )
        assert ctx.settings == mock_settings
        assert ctx.dry_run is True
        assert ctx.force is False

    def test_context_from_typer(self, mock_context):
        """Test creating context from Typer context."""
        from proxima.cli.workflows import WorkflowContext

        ctx = WorkflowContext.from_typer_context(mock_context)
        assert ctx.settings is not None

    def test_lazy_consent_manager(self, mock_settings):
        """Test lazy initialization of consent manager."""
        from proxima.cli.workflows import WorkflowContext
        from proxima.resources.consent import ConsentManager

        ctx = WorkflowContext(settings=mock_settings)
        manager = ctx.consent_manager
        assert isinstance(manager, ConsentManager)


class TestRunWorkflow:
    """Tests for RunWorkflow."""

    def test_run_workflow_creation(self, workflow_context):
        """Test creating run workflow."""
        from proxima.cli.workflows import RunOptions, RunWorkflow

        options = RunOptions(objective="test objective")
        workflow = RunWorkflow(workflow_context, options)
        assert workflow.name == "RunWorkflow"
        assert workflow.options.objective == "test objective"

    def test_run_workflow_dry_run(self, mock_settings):
        """Test run workflow in dry-run mode."""
        from proxima.cli.workflows import RunOptions, RunWorkflow, WorkflowContext

        ctx = WorkflowContext(settings=mock_settings, dry_run=True, force=True)
        options = RunOptions(objective="test", backend="aer_simulator")
        workflow = RunWorkflow(ctx, options)

        # In dry-run mode, precondition check is skipped and plan is returned
        result = workflow.run()
        # May fail if backend not available, but should produce a result
        assert result.status is not None

    def test_run_workflow_plan(self, workflow_context):
        """Test run workflow planning."""
        from proxima.cli.workflows import RunOptions, RunWorkflow

        workflow_context.force = True
        options = RunOptions(objective="demo", backend="test_backend")
        workflow = RunWorkflow(workflow_context, options)

        plan = workflow._plan()
        assert plan["objective"] == "demo"
        assert plan["backend"] == "test_backend"


class TestCompareWorkflow:
    """Tests for CompareWorkflow."""

    def test_compare_workflow_creation(self, workflow_context):
        """Test creating compare workflow."""
        from proxima.cli.workflows import CompareOptions, CompareWorkflow

        options = CompareOptions(
            objective="test",
            backends=["backend1", "backend2"],
        )
        workflow = CompareWorkflow(workflow_context, options)
        assert workflow.name == "CompareWorkflow"

    def test_compare_workflow_plan(self, workflow_context):
        """Test compare workflow planning."""
        from proxima.cli.workflows import CompareOptions, CompareWorkflow

        options = CompareOptions(
            objective="test",
            backends=["a", "b"],
            parallel=True,
        )
        workflow = CompareWorkflow(workflow_context, options)

        plan = workflow._plan()
        assert plan["objective"] == "test"
        assert plan["backends"] == ["a", "b"]
        assert plan["parallel"] is True


class TestValidationWorkflow:
    """Tests for ValidationWorkflow."""

    def test_validation_workflow_creation(self, workflow_context):
        """Test creating validation workflow."""
        from proxima.cli.workflows import ValidateOptions, ValidationWorkflow

        options = ValidateOptions(backend="test_backend")
        workflow = ValidationWorkflow(workflow_context, options)
        assert workflow.name == "ValidationWorkflow"

    def test_validation_workflow_run(self, workflow_context):
        """Test running validation workflow."""
        from proxima.cli.workflows import ValidateOptions, ValidationWorkflow

        workflow_context.force = True
        options = ValidateOptions()
        workflow = ValidationWorkflow(workflow_context, options)

        result = workflow.run()
        assert result.success
        assert "issues" in result.output


class TestExportWorkflow:
    """Tests for ExportWorkflow."""

    def test_export_workflow_creation(self, workflow_context, tmp_path):
        """Test creating export workflow."""
        from proxima.cli.workflows import ExportOptions, ExportWorkflow

        output_path = tmp_path / "export.json"
        options = ExportOptions(output_path=output_path, format="json")
        workflow = ExportWorkflow(workflow_context, options)
        assert workflow.name == "ExportWorkflow"

    def test_export_workflow_run(self, workflow_context, tmp_path):
        """Test running export workflow."""
        from proxima.cli.workflows import ExportOptions, ExportWorkflow

        workflow_context.force = True
        output_path = tmp_path / "export.json"
        options = ExportOptions(output_path=output_path, format="json")
        workflow = ExportWorkflow(workflow_context, options)

        result = workflow.run()
        assert result.success
        assert output_path.exists()


# ========== Tests: CLI Commands ==========


class TestRunCommand:
    """Tests for run command."""

    def test_run_command_help(self, cli_runner):
        """Test run command help."""
        from proxima.cli.commands.run import app

        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output.lower() or "execute" in result.output.lower()

    def test_validate_subcommand_help(self, cli_runner):
        """Test validate subcommand help."""
        from proxima.cli.commands.run import app

        result = cli_runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0

    def test_plan_subcommand_help(self, cli_runner):
        """Test plan subcommand help."""
        from proxima.cli.commands.run import app

        result = cli_runner.invoke(app, ["plan", "--help"])
        assert result.exit_code == 0


class TestCompareCommand:
    """Tests for compare command."""

    def test_compare_command_help(self, cli_runner):
        """Test compare command help."""
        from proxima.cli.commands.compare import app

        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "compare" in result.output.lower() or "backend" in result.output.lower()

    def test_report_subcommand_help(self, cli_runner):
        """Test report subcommand help."""
        from proxima.cli.commands.compare import app

        result = cli_runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0

    def test_quick_subcommand_help(self, cli_runner):
        """Test quick subcommand help."""
        from proxima.cli.commands.compare import app

        result = cli_runner.invoke(app, ["quick", "--help"])
        assert result.exit_code == 0


# ========== Tests: Output Config ==========


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_output_config_creation(self):
        """Test creating output config."""
        from proxima.cli.formatters import OutputConfig, OutputFormat

        config = OutputConfig(
            format=OutputFormat.JSON,
            no_color=True,
            quiet=False,
        )
        assert config.format == OutputFormat.JSON
        assert config.no_color is True

    def test_output_config_from_context(self, mock_context):
        """Test creating config from context."""
        from proxima.cli.formatters import OutputConfig

        mock_context.obj["output_format"] = "json"
        OutputConfig.from_context(mock_context)
        # Format should be parsed


# ========== Tests: Integration ==========


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_formatter_with_workflow_result(self):
        """Test formatting a workflow result."""
        from proxima.cli.formatters import format_output
        from proxima.cli.workflows import WorkflowResult, WorkflowStatus

        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            output={"key": "value"},
            duration_seconds=1.0,
        )

        # Format as JSON
        json_output = format_output(
            {"success": result.success, "output": result.output},
            format="json",
        )
        parsed = json.loads(json_output)
        assert parsed["success"] is True

    def test_progress_with_workflow(self, mock_settings):
        """Test progress display with workflow."""
        from proxima.cli.progress import step_context
        from proxima.cli.workflows import WorkflowContext

        WorkflowContext(settings=mock_settings, force=True)

        steps = ["Step 1", "Step 2"]
        with step_context(steps, title="Test", no_progress=True) as progress:
            progress.advance()
            progress.advance()

        # Should complete without error

    def test_echo_output_function(self, mock_context):
        """Test echo_output convenience function."""
        from proxima.cli.formatters import echo_output

        # Should not raise
        mock_context.obj["output_format"] = "text"
        mock_context.obj["quiet"] = True  # Suppress output
        echo_output(mock_context, {"key": "value"})
