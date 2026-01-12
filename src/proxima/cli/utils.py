"""CLI utility functions for configuration overrides and verbosity handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from proxima.config.settings import Settings, config_service

_LEVELS = ["critical", "error", "warning", "info", "debug"]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {**base}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def compute_verbosity(base_level: str, verbose: int, quiet: int) -> str:
    idx = (
        _LEVELS.index(base_level.lower())
        if base_level.lower() in _LEVELS
        else _LEVELS.index("info")
    )
    idx = max(0, min(len(_LEVELS) - 1, idx + verbose - quiet))
    return _LEVELS[idx]


def load_settings_with_cli_overrides(
    *,
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load settings and apply optional extra config file and CLI overrides."""

    base = config_service.load().model_dump()

    if config_path:
        extra = (
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        if extra:
            base = _deep_merge(base, extra)

    if cli_overrides:
        base = _deep_merge(base, cli_overrides)

    return Settings.from_dict(base)
