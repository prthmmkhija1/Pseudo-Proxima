"""Logging setup and configuration using structlog."""

from __future__ import annotations

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog

from proxima.config.settings import Settings

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

    renderer = (
        structlog.processors.JSONRenderer()
        if is_json
        else structlog.dev.ConsoleRenderer(colors=color)
    )

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            _add_execution_context,  # type: ignore[list-item]
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
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
