"""CLI module for Proxima - Full workflow support.

This module provides:
- Main CLI application entry point
- Workflow runners for complex operations
- Output formatters (text, JSON, table, rich)
- Progress display (spinners, bars, steps)
- Interactive prompts and consent dialogs
"""

# Formatters
from proxima.cli.formatters import (
    CsvFormatter,
    JsonFormatter,
    OutputConfig,
    OutputFormat,
    OutputFormatter,
    RichFormatter,
    TableFormatter,
    TextFormatter,
    YamlFormatter,
    echo_output,
    echo_result,
    format_output,
    get_formatter,
)
from proxima.cli.main import app

# Progress display
from proxima.cli.progress import (
    ProgressCallback,
    ProgressDisplay,
    ProgressStatus,
    SimpleProgress,
    Step,
    StepProgress,
    progress_context,
    spinner_context,
    step_context,
    track,
)

# Try to import Rich progress classes if available
try:
    from proxima.cli.progress import RichProgress, RichSpinner
except ImportError:
    RichSpinner = None  # type: ignore
    RichProgress = None  # type: ignore

# Prompts
from proxima.cli.prompts import (
    ConfirmPrompt,
    ConsentInfo,
    ConsentPrompt,
    MultiSelectPrompt,
    PasswordPrompt,
    Prompt,
    PromptResponse,
    PromptResult,
    SelectOption,
    SelectPrompt,
    TextPrompt,
    confirm,
    context_confirm,
    context_consent,
    prompt_multi_select,
    prompt_password,
    prompt_select,
    prompt_text,
    request_consent,
)

# Workflows
from proxima.cli.workflows import (
    CompareOptions,
    CompareWorkflow,
    ExportOptions,
    ExportWorkflow,
    RunOptions,
    RunWorkflow,
    ValidateOptions,
    ValidationWorkflow,
    WorkflowContext,
    WorkflowResult,
    WorkflowRunner,
    WorkflowStatus,
    compare_backends,
    run_workflow,
    validate_config,
)

__all__ = [
    # Main app
    "app",
    # Formatters
    "OutputFormat",
    "OutputFormatter",
    "TextFormatter",
    "JsonFormatter",
    "YamlFormatter",
    "CsvFormatter",
    "TableFormatter",
    "RichFormatter",
    "get_formatter",
    "format_output",
    "echo_output",
    "echo_result",
    "OutputConfig",
    # Progress
    "ProgressStatus",
    "ProgressDisplay",
    "SimpleProgress",
    "RichSpinner",
    "RichProgress",
    "StepProgress",
    "Step",
    "spinner_context",
    "progress_context",
    "step_context",
    "track",
    "ProgressCallback",
    # Prompts
    "PromptResult",
    "PromptResponse",
    "Prompt",
    "ConfirmPrompt",
    "TextPrompt",
    "PasswordPrompt",
    "SelectOption",
    "SelectPrompt",
    "MultiSelectPrompt",
    "ConsentInfo",
    "ConsentPrompt",
    "confirm",
    "prompt_text",
    "prompt_password",
    "prompt_select",
    "prompt_multi_select",
    "request_consent",
    "context_confirm",
    "context_consent",
    # Workflows
    "WorkflowStatus",
    "WorkflowResult",
    "WorkflowContext",
    "WorkflowRunner",
    "RunOptions",
    "RunWorkflow",
    "CompareOptions",
    "CompareWorkflow",
    "ValidateOptions",
    "ValidationWorkflow",
    "ExportOptions",
    "ExportWorkflow",
    "run_workflow",
    "compare_backends",
    "validate_config",
]
