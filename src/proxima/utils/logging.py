"""Logging setup and configuration using structlog."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog

from proxima.config.settings import Settings


_LEVEL_MAP = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def _resolve_level(level: str) -> int:
    return _LEVEL_MAP.get(level.lower(), logging.INFO)


def configure_logging(
    *,
    level: str = "info",
    output_format: str = "text",
    color: bool = True,
    log_file: Optional[Path] = None,
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

    handlers = [logging.StreamHandler(sys.stdout)]
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


def configure_from_settings(settings: Settings, *, log_file: Optional[Path] = None) -> None:
    """Configure logging using Settings values."""

    configure_logging(
        level=settings.general.verbosity,
        output_format=settings.general.output_format,
        color=settings.general.color_enabled,
        log_file=log_file,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger."""

    return structlog.get_logger(name) if name else structlog.get_logger()
