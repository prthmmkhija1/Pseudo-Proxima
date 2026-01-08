"""Default configuration values and constants for configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


# Default configuration tree used when no files are present.
DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "verbosity": "info",
        "output_format": "text",
        "color_enabled": True,
    },
    "backends": {
        "default_backend": "auto",
        "parallel_execution": False,
        "timeout_seconds": 300,
    },
    "llm": {
        "provider": "none",
        "model": "",
        "local_endpoint": "",
        "api_key_env_var": "",
        "require_consent": True,
    },
    "resources": {
        "memory_warn_threshold_mb": 4096,
        "memory_critical_threshold_mb": 8192,
        "max_execution_time_seconds": 3600,
    },
    "consent": {
        "auto_approve_local_llm": False,
        "auto_approve_remote_llm": False,
        "remember_decisions": False,
    },
}


ENV_PREFIX = "PROXIMA"
USER_CONFIG_PATH = Path.home() / ".proxima" / "config.yaml"
PROJECT_CONFIG_FILENAME = "proxima.yaml"
DEFAULT_CONFIG_RELATIVE_PATH = Path("configs") / "default.yaml"
