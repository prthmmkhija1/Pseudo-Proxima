"""Configuration export/import utilities.

Provides functionality for:
- Exporting configuration to various formats (YAML, JSON, TOML)
- Importing configuration from external sources
- Configuration backup and restore
- Template generation
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ExportFormat(Enum):
    """Supported configuration export formats."""

    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    ENV = "env"  # Environment variable format

    @classmethod
    def from_extension(cls, path: Path) -> ExportFormat:
        """Determine format from file extension."""
        ext = path.suffix.lower()
        mapping = {
            ".yaml": cls.YAML,
            ".yml": cls.YAML,
            ".json": cls.JSON,
            ".toml": cls.TOML,
            ".env": cls.ENV,
        }
        return mapping.get(ext, cls.YAML)


@dataclass
class ExportOptions:
    """Options for configuration export."""

    include_defaults: bool = True  # Include default values
    include_comments: bool = True  # Include descriptive comments
    redact_secrets: bool = True  # Replace secrets with placeholders
    pretty_print: bool = True  # Format output nicely

    # Keys that should be redacted
    secret_patterns: list[str] = field(
        default_factory=lambda: [
            "api_key",
            "secret",
            "password",
            "token",
            "credential",
        ]
    )


@dataclass
class ImportResult:
    """Result of a configuration import operation."""

    success: bool
    config: dict[str, Any] | None = None
    source: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.success:
            return f"✓ Successfully imported configuration from {self.source}"
        return f"✗ Failed to import from {self.source}: {', '.join(self.errors)}"


@dataclass
class BackupInfo:
    """Information about a configuration backup."""

    path: Path
    created_at: datetime
    size_bytes: int
    description: str = ""

    @classmethod
    def from_path(cls, path: Path) -> BackupInfo:
        """Create BackupInfo from a backup file path."""
        stat = path.stat()
        return cls(
            path=path,
            created_at=datetime.fromtimestamp(stat.st_mtime),
            size_bytes=stat.st_size,
        )


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def _redact_secrets(config: dict[str, Any], patterns: list[str]) -> dict[str, Any]:
    """Replace secret values with placeholders."""
    result: dict[str, Any] = {}
    for key, value in config.items():
        key_lower = key.lower()
        if isinstance(value, dict):
            result[key] = _redact_secrets(value, patterns)
        elif any(pattern in key_lower for pattern in patterns):
            result[key] = "<REDACTED>"
        else:
            result[key] = value
    return result


def _add_yaml_comments(config: dict[str, Any]) -> str:
    """Generate YAML with helpful comments."""
    comments = {
        "general": "# General application settings",
        "backends": "# Quantum backend configuration",
        "llm": "# LLM (Large Language Model) integration settings",
        "resources": "# System resource monitoring thresholds",
        "consent": "# User consent preferences",
    }

    lines = [
        "# Proxima Configuration File",
        f"# Generated: {datetime.now().isoformat()}",
        "# Docs: https://github.com/your-org/proxima#configuration",
        "",
    ]

    yaml_str = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)

    for line in yaml_str.split("\n"):
        # Add section comments
        for section, comment in comments.items():
            if line.startswith(f"{section}:"):
                lines.append("")
                lines.append(comment)
                break
        lines.append(line)

    return "\n".join(lines)


def export_config(
    config: dict[str, Any],
    output_path: Path,
    format: ExportFormat | None = None,
    options: ExportOptions | None = None,
) -> Path:
    """Export configuration to a file.

    Args:
        config: Configuration dictionary to export
        output_path: Path to write the configuration
        format: Output format (auto-detected from extension if None)
        options: Export options

    Returns:
        Path to the created file
    """
    options = options or ExportOptions()
    format = format or ExportFormat.from_extension(output_path)

    # Optionally redact secrets
    export_data = config
    if options.redact_secrets:
        export_data = _redact_secrets(config, options.secret_patterns)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == ExportFormat.YAML:
        if options.include_comments:
            content = _add_yaml_comments(export_data)
        else:
            content = yaml.safe_dump(
                export_data, sort_keys=False, default_flow_style=False
            )
        output_path.write_text(content, encoding="utf-8")

    elif format == ExportFormat.JSON:
        indent = 2 if options.pretty_print else None
        content = json.dumps(export_data, indent=indent, ensure_ascii=False)
        output_path.write_text(content, encoding="utf-8")

    elif format == ExportFormat.TOML:
        try:
            import tomli_w

            output_path.write_bytes(tomli_w.dumps(export_data).encode("utf-8"))
        except ImportError:
            # Fallback: write as simple TOML manually
            content = _dict_to_simple_toml(export_data)
            output_path.write_text(content, encoding="utf-8")

    elif format == ExportFormat.ENV:
        content = _dict_to_env(export_data, prefix="PROXIMA")
        output_path.write_text(content, encoding="utf-8")

    return output_path


def _dict_to_simple_toml(data: dict[str, Any], prefix: str = "") -> str:
    """Convert dict to simple TOML format."""
    lines = []

    # First, output non-dict values
    for key, value in data.items():
        if not isinstance(value, dict):
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")

    # Then output sections (dicts)
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append("")
            section_name = f"{prefix}.{key}" if prefix else key
            lines.append(f"[{section_name}]")
            lines.append(_dict_to_simple_toml(value, section_name))

    return "\n".join(lines)


def _dict_to_env(data: dict[str, Any], prefix: str = "") -> str:
    """Convert dict to environment variable format."""
    lines = [
        "# Proxima Environment Variables",
        f"# Generated: {datetime.now().isoformat()}",
        "",
    ]

    def flatten(d: dict[str, Any], parent_key: str = "") -> list[tuple[str, Any]]:
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}__{key.upper()}" if parent_key else key.upper()
            if isinstance(value, dict):
                items.extend(flatten(value, new_key))
            else:
                items.append((new_key, value))
        return items

    for key, value in flatten(data):
        env_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, bool):
            env_value = "true" if value else "false"
        elif value is None:
            env_value = ""
        else:
            env_value = str(value)
        lines.append(f"{env_key}={env_value}")

    return "\n".join(lines)


# =============================================================================
# IMPORT FUNCTIONS
# =============================================================================


def import_config(source_path: Path) -> ImportResult:
    """Import configuration from a file.

    Args:
        source_path: Path to the configuration file

    Returns:
        ImportResult with the loaded configuration or errors
    """
    result = ImportResult(success=False, source=str(source_path))

    if not source_path.exists():
        result.errors.append(f"File not found: {source_path}")
        return result

    format = ExportFormat.from_extension(source_path)

    try:
        if format == ExportFormat.YAML:
            content = source_path.read_text(encoding="utf-8")
            result.config = yaml.safe_load(content)

        elif format == ExportFormat.JSON:
            content = source_path.read_text(encoding="utf-8")
            result.config = json.loads(content)

        elif format == ExportFormat.TOML:
            try:
                import tomllib

                toml_bytes = source_path.read_bytes()
                result.config = tomllib.loads(toml_bytes.decode("utf-8"))
            except ImportError:
                result.errors.append(
                    "TOML support requires Python 3.11+ or tomli package"
                )
                return result

        elif format == ExportFormat.ENV:
            result.config = _env_to_dict(source_path)

        else:
            result.errors.append(f"Unsupported format: {format}")
            return result

        if result.config is None:
            result.warnings.append("Configuration file is empty")
            result.config = {}

        result.success = True

    except yaml.YAMLError as e:
        result.errors.append(f"YAML parsing error: {e}")
    except json.JSONDecodeError as e:
        result.errors.append(f"JSON parsing error: {e}")
    except Exception as e:
        result.errors.append(f"Failed to read file: {e}")

    return result


def _env_to_dict(path: Path) -> dict[str, Any]:
    """Parse .env file to nested dictionary."""
    result: dict[str, Any] = {}
    content = path.read_text(encoding="utf-8")

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        # Remove PROXIMA_ prefix if present
        if key.startswith("PROXIMA_"):
            key = key[8:]

        # Split by __ into nested structure
        parts = [p.lower() for p in key.split("__")]

        # Build nested dict
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})

        # Convert value types
        converted_value: str | bool | int
        if value.lower() == "true":
            converted_value = True
        elif value.lower() == "false":
            converted_value = False
        elif value.isdigit():
            converted_value = int(value)
        else:
            converted_value = value

        current[parts[-1]] = converted_value

    return result


def import_from_url(url: str) -> ImportResult:
    """Import configuration from a URL.

    Args:
        url: URL to fetch configuration from

    Returns:
        ImportResult with the loaded configuration or errors
    """
    result = ImportResult(success=False, source=url)

    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode("utf-8")

        # Try to determine format from URL or content
        if url.endswith(".json") or content.strip().startswith("{"):
            result.config = json.loads(content)
        else:
            result.config = yaml.safe_load(content)

        result.success = True

    except Exception as e:
        result.errors.append(f"Failed to fetch URL: {e}")

    return result


# =============================================================================
# BACKUP FUNCTIONS
# =============================================================================


def create_backup(
    config_path: Path,
    backup_dir: Path | None = None,
    description: str = "",
) -> BackupInfo:
    """Create a backup of a configuration file.

    Args:
        config_path: Path to the configuration file to backup
        backup_dir: Directory to store backups (default: same dir with .bak suffix)
        description: Optional description for the backup

    Returns:
        BackupInfo about the created backup
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if backup_dir is None:
        backup_dir = config_path.parent / ".proxima_backups"

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{config_path.stem}_{timestamp}{config_path.suffix}"
    backup_path = backup_dir / backup_name

    shutil.copy2(config_path, backup_path)

    info = BackupInfo.from_path(backup_path)
    info.description = description

    return info


def list_backups(backup_dir: Path) -> list[BackupInfo]:
    """List all configuration backups in a directory.

    Args:
        backup_dir: Directory containing backups

    Returns:
        List of BackupInfo sorted by creation time (newest first)
    """
    if not backup_dir.exists():
        return []

    backups = []
    for path in backup_dir.glob("*"):
        if path.is_file() and path.suffix in (".yaml", ".yml", ".json"):
            try:
                backups.append(BackupInfo.from_path(path))
            except Exception:
                continue

    return sorted(backups, key=lambda b: b.created_at, reverse=True)


def restore_backup(backup_info: BackupInfo, target_path: Path) -> None:
    """Restore a configuration backup.

    Args:
        backup_info: BackupInfo of the backup to restore
        target_path: Path to restore the backup to
    """
    if not backup_info.path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_info.path}")

    # Create backup of current config before restoring
    if target_path.exists():
        create_backup(target_path, description="Auto-backup before restore")

    shutil.copy2(backup_info.path, target_path)


def cleanup_old_backups(backup_dir: Path, keep_count: int = 10) -> int:
    """Remove old backups, keeping only the most recent ones.

    Args:
        backup_dir: Directory containing backups
        keep_count: Number of recent backups to keep

    Returns:
        Number of backups removed
    """
    backups = list_backups(backup_dir)

    if len(backups) <= keep_count:
        return 0

    removed = 0
    for backup in backups[keep_count:]:
        try:
            backup.path.unlink()
            removed += 1
        except Exception:
            continue

    return removed


# =============================================================================
# TEMPLATE FUNCTIONS
# =============================================================================


def generate_template(
    output_path: Path,
    template_type: str = "full",
    format: ExportFormat = ExportFormat.YAML,
) -> Path:
    """Generate a configuration template file.

    Args:
        output_path: Path to write the template
        template_type: Type of template ("full", "minimal", "development", "production")
        format: Output format

    Returns:
        Path to the created template
    """
    templates = {
        "full": _get_full_template(),
        "minimal": _get_minimal_template(),
        "development": _get_development_template(),
        "production": _get_production_template(),
    }

    template = templates.get(template_type, templates["full"])
    return export_config(
        template,
        output_path,
        format,
        ExportOptions(redact_secrets=False, include_comments=True),
    )


def _get_full_template() -> dict[str, Any]:
    """Full configuration template with all options."""
    return {
        "general": {
            "verbosity": "info",
            "output_format": "text",
            "color_enabled": True,
            "data_dir": "",
            "storage_backend": "sqlite",
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


def _get_minimal_template() -> dict[str, Any]:
    """Minimal configuration with only essential settings."""
    return {
        "general": {
            "verbosity": "info",
        },
        "backends": {
            "default_backend": "auto",
        },
    }


def _get_development_template() -> dict[str, Any]:
    """Development-focused configuration."""
    return {
        "general": {
            "verbosity": "debug",
            "output_format": "rich",
            "color_enabled": True,
            "storage_backend": "sqlite",
        },
        "backends": {
            "default_backend": "auto",
            "parallel_execution": False,
            "timeout_seconds": 60,  # Shorter for dev
        },
        "llm": {
            "provider": "ollama",
            "local_endpoint": "http://localhost:11434",
            "require_consent": False,  # Skip consent in dev
        },
        "resources": {
            "memory_warn_threshold_mb": 2048,
            "memory_critical_threshold_mb": 4096,
            "max_execution_time_seconds": 300,  # Shorter for dev
        },
        "consent": {
            "auto_approve_local_llm": True,
            "remember_decisions": True,
        },
    }


def _get_production_template() -> dict[str, Any]:
    """Production-ready configuration."""
    return {
        "general": {
            "verbosity": "warning",
            "output_format": "json",
            "color_enabled": False,
            "storage_backend": "sqlite",
        },
        "backends": {
            "default_backend": "auto",
            "parallel_execution": True,
            "timeout_seconds": 600,
        },
        "llm": {
            "provider": "none",
            "require_consent": True,
        },
        "resources": {
            "memory_warn_threshold_mb": 8192,
            "memory_critical_threshold_mb": 16384,
            "max_execution_time_seconds": 7200,
        },
        "consent": {
            "auto_approve_local_llm": False,
            "auto_approve_remote_llm": False,
            "remember_decisions": False,
        },
    }
