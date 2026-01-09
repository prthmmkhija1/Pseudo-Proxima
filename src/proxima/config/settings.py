"""Configuration system for Proxima.

Implements layered configuration with the following priority (high → low):
1) CLI overrides (explicit flags)
2) Environment variables (prefix: PROXIMA_)
3) User config file (~/.proxima/config.yaml)
4) Project config file (./proxima.yaml)
5) Default config file (configs/default.yaml)
6) Built-in defaults (fallback)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from proxima.config.defaults import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_RELATIVE_PATH,
    ENV_PREFIX,
    PROJECT_CONFIG_FILENAME,
    USER_CONFIG_PATH,
)


class GeneralSettings(BaseModel):
    verbosity: str = Field(default="info")
    output_format: str = Field(default="text")
    color_enabled: bool = Field(default=True)
    data_dir: str = Field(default="")
    storage_backend: str = Field(default="sqlite")


class BackendsSettings(BaseModel):
    default_backend: str = Field(default="auto")
    parallel_execution: bool = Field(default=False)
    timeout_seconds: int = Field(default=300)


class LLMSettings(BaseModel):
    provider: str = Field(default="none")
    model: str = Field(default="")
    local_endpoint: str = Field(default="")
    api_key_env_var: str = Field(default="")
    require_consent: bool = Field(default=True)


class ResourcesSettings(BaseModel):
    memory_warn_threshold_mb: int = Field(default=4096)
    memory_critical_threshold_mb: int = Field(default=8192)
    max_execution_time_seconds: int = Field(default=3600)


class ConsentSettings(BaseModel):
    auto_approve_local_llm: bool = Field(default=False)
    auto_approve_remote_llm: bool = Field(default=False)
    remember_decisions: bool = Field(default=False)


class Settings(BaseModel):
    general: GeneralSettings
    backends: BackendsSettings
    llm: LLMSettings
    resources: ResourcesSettings
    consent: ConsentSettings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        return cls.model_validate(data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""

    result = {**base}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - unlikely with safe_load
        raise ValueError(f"Failed to parse YAML config at {path}: {exc}") from exc


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_scalar(value: str) -> Any:
    """Best-effort parsing for CLI/env string values."""

    trimmed = value.strip()
    # Try JSON (covers numbers, booleans, null, quoted strings)
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        pass

    lowered = trimmed.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    return trimmed


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    current = target
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _get_nested(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


class ConfigService:
    """Loads, merges, and persists Proxima configuration."""

    def __init__(self, env_prefix: str = ENV_PREFIX):
        self.env_prefix = env_prefix
        self.root_dir = self._compute_project_root()
        self.default_config_path = self.root_dir / DEFAULT_CONFIG_RELATIVE_PATH
        self.project_config_path = self.root_dir / PROJECT_CONFIG_FILENAME
        self.user_config_path = USER_CONFIG_PATH

    def load(self, cli_overrides: dict[str, Any] | None = None) -> Settings:
        data = DEFAULT_CONFIG

        file_chain = [
            self.default_config_path,
            self.project_config_path,
            self.user_config_path,
        ]

        for path in file_chain:
            data = _deep_merge(data, _load_yaml(path))

        data = _deep_merge(data, self._env_overrides())
        if cli_overrides:
            data = _deep_merge(data, cli_overrides)

        try:
            return Settings.from_dict(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc

    def save(self, settings: Settings, scope: Literal["user", "project"] = "user") -> Path:
        target = self.user_config_path if scope == "user" else self.project_config_path
        _ensure_dir(target)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(settings.model_dump(), handle, sort_keys=False)
        return target

    def set_value(
        self, key_path: str, value: Any, scope: Literal["user", "project"] = "user"
    ) -> Path:
        target = self.user_config_path if scope == "user" else self.project_config_path
        current_data = _load_yaml(target)
        parts = self._normalize_key_path(key_path)
        _set_nested(current_data, parts, value)
        _ensure_dir(target)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(current_data, handle, sort_keys=False)
        return target

    def get_value(self, key_path: str, cli_overrides: dict[str, Any] | None = None) -> Any:
        data = self.load(cli_overrides=cli_overrides).model_dump()
        parts = self._normalize_key_path(key_path)
        return _get_nested(data, parts)

    def reset(self, scope: Literal["user", "project"] = "user") -> None:
        target = self.user_config_path if scope == "user" else self.project_config_path
        if target.exists():
            target.unlink()

    def _env_overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        prefix = f"{self.env_prefix}_"
        for key, raw_value in os.environ.items():
            if not key.startswith(prefix):
                continue
            path_part = key[len(prefix) :]
            path_segments = self._normalize_env_key(path_part)
            _set_nested(overrides, path_segments, _parse_scalar(raw_value))
        return overrides

    def _normalize_env_key(self, key: str) -> list[str]:
        if "__" in key:
            segments = key.split("__")
        else:
            segments = key.split("_")
        return [segment.lower() for segment in segments if segment]

    def _normalize_key_path(self, key_path: str) -> list[str]:
        if not key_path:
            raise ValueError("Key path cannot be empty")
        return [segment.strip() for segment in key_path.split(".") if segment.strip()]

    def _compute_project_root(self) -> Path:
        # settings.py -> config -> proxima -> src -> project
        return Path(__file__).resolve().parents[3]


config_service = ConfigService()


# Convenience singleton for global settings access
_cached_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = config_service.load()
    return _cached_settings


def reload_settings() -> Settings:
    """Reload settings from configuration sources."""
    global _cached_settings
    _cached_settings = config_service.load()
    return _cached_settings


class FlatSettings:
    """Flat accessor for settings values."""

    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def storage_backend(self) -> str:
        return self._settings.general.storage_backend

    @property
    def data_dir(self) -> str | None:
        return self._settings.general.data_dir or None
