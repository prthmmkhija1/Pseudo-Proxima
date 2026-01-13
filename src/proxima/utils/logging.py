"""Logging setup and configuration using structlog."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from proxima.config.settings import Settings

_LEVEL_MAP = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def _json_serializer(obj: Any, **kwargs: Any) -> str:
    """Custom JSON serializer that handles special types for log output.
    
    Handles:
    - datetime objects -> ISO format strings
    - Path objects -> string paths
    - Enum values -> their value
    - numpy arrays -> lists
    - bytes -> base64 or hex encoded
    - sets -> sorted lists
    - Complex numbers -> dict with real/imag
    - Any object with __dict__ -> dict representation
    """
    return json.dumps(obj, default=_json_default, **kwargs)


def _json_default(obj: Any) -> Any:
    """Default handler for JSON serialization of special types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.hex()
    if isinstance(obj, set):
        return sorted(list(obj), key=str)
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    # Try numpy array conversion
    try:
        if hasattr(obj, "tolist"):
            return obj.tolist()
    except Exception:
        pass
    # Fallback to string representation
    return str(obj)


def _sanitize_event_dict(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Sanitize event dict for JSON serialization.
    
    - Converts non-serializable types to strings
    - Truncates very long strings
    - Handles circular references
    """
    MAX_STRING_LENGTH = 10000
    
    def sanitize_value(value: Any, depth: int = 0) -> Any:
        if depth > 10:  # Prevent infinite recursion
            return "<max depth exceeded>"
            
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if len(value) > MAX_STRING_LENGTH:
                return value[:MAX_STRING_LENGTH] + "...<truncated>"
            return value
        if isinstance(value, (list, tuple)):
            return [sanitize_value(v, depth + 1) for v in value[:100]]  # Limit list size
        if isinstance(value, dict):
            return {
                str(k): sanitize_value(v, depth + 1)
                for k, v in list(value.items())[:50]  # Limit dict size
            }
        # Use the default serializer for other types
        try:
            return _json_default(value)
        except Exception:
            return str(value)
    
    return {k: sanitize_value(v) for k, v in event_dict.items()}

_LEVEL_MAP = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

# Context variable for execution tracking
_execution_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "execution_context", default=None
)


def _get_context() -> dict[str, Any]:
    """Get the current execution context, initializing if needed."""
    ctx = _execution_context.get()
    if ctx is None:
        ctx = {}
        _execution_context.set(ctx)
    return ctx


def set_execution_context(
    execution_id: str | None = None,
    session_id: str | None = None,
    **extra: Any,
) -> None:
    """Set execution context for log enrichment.

    Parameters
    ----------
    execution_id : Optional[str]
        Unique identifier for the current execution run.
    session_id : Optional[str]
        Session identifier for grouping related executions.
    **extra : Any
        Additional context key-value pairs.
    """
    ctx = _get_context().copy()
    if execution_id:
        ctx["execution_id"] = execution_id
    if session_id:
        ctx["session_id"] = session_id
    ctx.update(extra)
    _execution_context.set(ctx)


def clear_execution_context() -> None:
    """Clear the current execution context."""
    _execution_context.set({})


def generate_execution_id() -> str:
    """Generate a unique execution ID."""
    return str(uuid.uuid4())[:8]


def _add_execution_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor to inject execution context."""
    ctx = _get_context()
    for key, value in ctx.items():
        if key not in event_dict:
            event_dict[key] = value
    return event_dict


def _resolve_level(level: str) -> int:
    return _LEVEL_MAP.get(level.lower(), logging.INFO)


def configure_logging(
    *,
    level: str = "info",
    output_format: str = "text",
    color: bool = True,
    log_file: Path | None = None,
) -> None:
    """Configure structlog + stdlib logging.

    Parameters
    ----------
    level: str
            Minimum level (debug, info, warning, error, critical).
    output_format: str
            "text" for console-friendly rendering, "json" for machine parsing.
    color: bool
            Enable colored console output when using text mode.
    log_file: Optional[Path]
            If provided, also write logs to this file (JSON if output_format is json).
    """

    log_level = _resolve_level(level)
    is_json = output_format.lower() == "json"

    # Configure JSON renderer with proper formatting
    if is_json:
        # Use JSONRenderer with proper formatting for machine parsing
        renderer = structlog.processors.JSONRenderer(
            serializer=_json_serializer,
            sort_keys=True,
        )
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=color)

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            _add_execution_context,  # type: ignore[list-item]
            _sanitize_event_dict,  # Sanitize for JSON output
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    root = logging.getLogger()
    root.handlers = []
    root.setLevel(log_level)
    for handler in handlers:
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)

    # Silence verbose third-party loggers
    logging.getLogger("transitions").setLevel(logging.WARNING)


def configure_from_settings(
    settings: Settings, *, log_file: Path | None = None
) -> None:
    """Configure logging using Settings values."""

    configure_logging(
        level=settings.general.verbosity,
        output_format=settings.general.output_format,
        color=settings.general.color_enabled,
        log_file=log_file,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger."""

    return structlog.get_logger(name) if name else structlog.get_logger()


class timed_operation:
    """Context manager for timing operations and logging duration_ms.

    Usage:
        with timed_operation("simulation_run", logger=log):
            # ... your code ...
        # Logs: {"event": "simulation_run", "duration_ms": 1234.56, ...}

    Can also be used as a decorator:
        @timed_operation("process_data")
        def process_data():
            ...
    """

    def __init__(
        self,
        operation_name: str,
        logger: structlog.stdlib.BoundLogger | None = None,
        log_level: str = "info",
        **extra_context: Any,
    ) -> None:
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.log_level = log_level
        self.extra_context = extra_context
        self._start_time: float = 0.0

    def __enter__(self) -> timed_operation:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = (time.perf_counter() - self._start_time) * 1000
        log_method = getattr(self.logger, self.log_level)

        status = "failed" if exc_type else "completed"
        log_method(
            self.operation_name,
            duration_ms=round(duration_ms, 2),
            status=status,
            **self.extra_context,
        )

    def __call__(self, func: Any) -> Any:
        """Allow usage as a decorator."""
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapper

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds (useful during operation)."""
        return (time.perf_counter() - self._start_time) * 1000
# =============================================================================
# JSON FILE OUTPUT HANDLER
# =============================================================================


class JSONFileHandler(logging.FileHandler):
    """A logging file handler that writes structured JSON logs.
    
    This handler creates a dedicated JSON log file where each log entry
    is written as a complete JSON object on a single line (JSON Lines format).
    This format is ideal for log aggregation tools, analysis, and machine parsing.
    
    Features:
    - Structured JSON output (one JSON object per line)
    - Automatic file rotation support (when used with RotatingFileHandler)
    - Thread-safe writing
    - Custom serialization for complex types (datetime, Path, numpy, etc.)
    
    Example:
        handler = JSONFileHandler('app.log.json')
        logger.addHandler(handler)
    """
    
    def __init__(
        self,
        filename: str | Path,
        mode: str = 'a',
        encoding: str = 'utf-8',
        delay: bool = False,
    ) -> None:
        """Initialize the JSON file handler.
        
        Args:
            filename: Path to the JSON log file
            mode: File mode ('a' for append, 'w' for overwrite)
            encoding: File encoding (default: utf-8)
            delay: If True, defer opening until first write
        """
        super().__init__(str(filename), mode=mode, encoding=encoding, delay=delay)
        self._json_formatter = _JSONLogFormatter()
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as a JSON line.
        
        Args:
            record: The log record to emit
        """
        try:
            msg = self._json_formatter.format(record)
            stream = self.stream
            stream.write(msg + '\n')
            self.flush()
        except Exception:
            self.handleError(record)


class _JSONLogFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON objects."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON string representation of the log record
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add execution context if available
        ctx = _execution_context.get()
        if ctx:
            log_data['context'] = ctx
        
        # Add exception info if present
        if record.exc_info:
            import traceback
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info) if record.exc_info[0] else None,
            }
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in (
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'message', 'thread',
                'threadName', 'taskName',
            ):
                log_data[key] = value
        
        return _json_serializer(log_data)


def configure_logging_with_json_file(
    *,
    level: str = "info",
    output_format: str = "text",
    color: bool = True,
    log_file: Path | None = None,
    json_log_file: Path | None = None,
) -> None:
    """Configure logging with an optional dedicated JSON file output.
    
    This function extends the standard configure_logging() by adding support
    for a separate JSON log file that can be used for structured log analysis.
    
    Parameters
    ----------
    level : str
        Minimum log level (debug, info, warning, error, critical).
    output_format : str
        Console output format: "text" for human-readable, "json" for machine parsing.
    color : bool
        Enable colored console output when using text mode.
    log_file : Optional[Path]
        If provided, write logs to this file (format matches output_format).
    json_log_file : Optional[Path]
        If provided, write structured JSON logs to this dedicated file.
        This file will always use JSON Lines format regardless of output_format.
    
    Example
    -------
    >>> configure_logging_with_json_file(
    ...     level="debug",
    ...     output_format="text",
    ...     color=True,
    ...     log_file=Path("app.log"),
    ...     json_log_file=Path("app.log.json")
    ... )
    
    This will output human-readable logs to console and app.log,
    while also writing structured JSON to app.log.json for analysis.
    """
    # First, configure standard logging
    configure_logging(
        level=level,
        output_format=output_format,
        color=color,
        log_file=log_file,
    )
    
    # Add JSON file handler if specified
    if json_log_file:
        log_level = _resolve_level(level)
        
        # Create parent directories if needed
        json_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add JSON file handler to root logger
        json_handler = JSONFileHandler(json_log_file)
        json_handler.setLevel(log_level)
        
        root = logging.getLogger()
        root.addHandler(json_handler)



