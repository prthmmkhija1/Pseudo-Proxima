"""Configuration validation with detailed error messages.

Provides comprehensive validation for all configuration values with
human-readable error messages and suggestions for fixing issues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Must be fixed before use
    WARNING = "warning"  # Should be fixed but won't block
    INFO = "info"  # Informational suggestion


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    path: str  # Dot-separated path to the config key
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    suggestion: str | None = None
    current_value: Any = None
    expected_type: str | None = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.path}: {self.message}"]
        if self.current_value is not None:
            parts.append(f"  Current value: {self.current_value!r}")
        if self.expected_type:
            parts.append(f"  Expected type: {self.expected_type}")
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        return "\n".join(parts)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False

    def add_error(
        self,
        path: str,
        message: str,
        suggestion: str | None = None,
        current_value: Any = None,
        expected_type: str | None = None,
    ) -> None:
        """Add an error issue."""
        self.add_issue(
            ValidationIssue(
                path=path,
                message=message,
                severity=ValidationSeverity.ERROR,
                suggestion=suggestion,
                current_value=current_value,
                expected_type=expected_type,
            )
        )

    def add_warning(
        self,
        path: str,
        message: str,
        suggestion: str | None = None,
        current_value: Any = None,
    ) -> None:
        """Add a warning issue."""
        self.add_issue(
            ValidationIssue(
                path=path,
                message=message,
                severity=ValidationSeverity.WARNING,
                suggestion=suggestion,
                current_value=current_value,
            )
        )

    def add_info(self, path: str, message: str, suggestion: str | None = None) -> None:
        """Add an informational issue."""
        self.add_issue(
            ValidationIssue(
                path=path,
                message=message,
                severity=ValidationSeverity.INFO,
                suggestion=suggestion,
            )
        )

    def errors(self) -> list[ValidationIssue]:
        """Get only error issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationIssue]:
        """Get only warning issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def format_report(self) -> str:
        """Generate a human-readable validation report."""
        if not self.issues:
            return "✓ Configuration is valid"

        lines = []
        errors = self.errors()
        warnings = self.warnings()
        infos = [i for i in self.issues if i.severity == ValidationSeverity.INFO]

        if errors:
            lines.append(f"❌ {len(errors)} error(s) found:")
            for issue in errors:
                lines.append(str(issue))

        if warnings:
            lines.append(f"\n⚠️ {len(warnings)} warning(s):")
            for issue in warnings:
                lines.append(str(issue))

        if infos:
            lines.append(f"\nℹ️ {len(infos)} suggestion(s):")
            for issue in infos:
                lines.append(str(issue))

        return "\n".join(lines)

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result into this one."""
        for issue in other.issues:
            self.add_issue(issue)


# =============================================================================
# VALIDATORS
# =============================================================================


def validate_verbosity(value: Any, path: str, result: ValidationResult) -> None:
    """Validate verbosity setting."""
    valid_values = {"debug", "info", "warning", "error"}
    if not isinstance(value, str):
        result.add_error(
            path, "Verbosity must be a string", current_value=value, expected_type="str"
        )
    elif value.lower() not in valid_values:
        result.add_error(
            path,
            f"Invalid verbosity level: {value}",
            suggestion=f"Use one of: {', '.join(sorted(valid_values))}",
            current_value=value,
        )


def validate_output_format(value: Any, path: str, result: ValidationResult) -> None:
    """Validate output format setting."""
    valid_formats = {"text", "json", "rich"}
    if not isinstance(value, str):
        result.add_error(
            path,
            "Output format must be a string",
            current_value=value,
            expected_type="str",
        )
    elif value.lower() not in valid_formats:
        result.add_error(
            path,
            f"Invalid output format: {value}",
            suggestion=f"Use one of: {', '.join(sorted(valid_formats))}",
            current_value=value,
        )


def validate_backend(value: Any, path: str, result: ValidationResult) -> None:
    """Validate default backend setting."""
    valid_backends = {"auto", "lret", "cirq", "qiskit"}
    if not isinstance(value, str):
        result.add_error(
            path, "Backend must be a string", current_value=value, expected_type="str"
        )
    elif value.lower() not in valid_backends:
        result.add_warning(
            path,
            f"Unknown backend: {value}",
            suggestion=f"Standard backends are: {', '.join(sorted(valid_backends))}. "
            "This may be a custom plugin backend.",
            current_value=value,
        )


def validate_timeout(value: Any, path: str, result: ValidationResult) -> None:
    """Validate timeout value."""
    if not isinstance(value, (int, float)):
        result.add_error(
            path,
            "Timeout must be a number",
            current_value=value,
            expected_type="int or float",
        )
    elif value <= 0:
        result.add_error(
            path,
            "Timeout must be positive",
            suggestion="Set a value greater than 0 seconds",
            current_value=value,
        )
    elif value > 86400:  # 24 hours
        result.add_warning(
            path,
            "Timeout is very large (>24 hours)",
            suggestion="Consider reducing for better responsiveness",
            current_value=value,
        )


def validate_llm_provider(value: Any, path: str, result: ValidationResult) -> None:
    """Validate LLM provider setting."""
    valid_providers = {"none", "openai", "anthropic", "ollama", "lmstudio", "local"}
    if not isinstance(value, str):
        result.add_error(
            path,
            "LLM provider must be a string",
            current_value=value,
            expected_type="str",
        )
    elif value.lower() not in valid_providers:
        result.add_warning(
            path,
            f"Unknown LLM provider: {value}",
            suggestion=f"Standard providers are: {', '.join(sorted(valid_providers))}",
            current_value=value,
        )


def validate_url(value: Any, path: str, result: ValidationResult) -> None:
    """Validate URL format."""
    if not value:  # Empty is allowed
        return
    if not isinstance(value, str):
        result.add_error(
            path, "URL must be a string", current_value=value, expected_type="str"
        )
        return

    try:
        parsed = urlparse(value)
        if not parsed.scheme:
            result.add_error(
                path,
                "URL missing scheme (http:// or https://)",
                suggestion=f"Add scheme: https://{value}",
                current_value=value,
            )
        elif parsed.scheme not in ("http", "https"):
            result.add_warning(
                path,
                f"Unusual URL scheme: {parsed.scheme}",
                suggestion="Standard schemes are http:// or https://",
                current_value=value,
            )
        if not parsed.netloc:
            result.add_error(path, "URL missing host", current_value=value)
    except Exception:
        result.add_error(path, "Invalid URL format", current_value=value)


def validate_memory_threshold(value: Any, path: str, result: ValidationResult) -> None:
    """Validate memory threshold in MB."""
    if not isinstance(value, int):
        result.add_error(
            path,
            "Memory threshold must be an integer",
            current_value=value,
            expected_type="int",
        )
    elif value < 128:
        result.add_warning(
            path,
            "Memory threshold is very low (<128 MB)",
            suggestion="This may cause frequent warnings on modern systems",
            current_value=value,
        )
    elif value > 65536:  # 64 GB
        result.add_warning(
            path,
            "Memory threshold is very high (>64 GB)",
            suggestion="Ensure your system has this much RAM",
            current_value=value,
        )


def validate_path(value: Any, path: str, result: ValidationResult) -> None:
    """Validate file/directory path."""
    if not value:  # Empty is allowed
        return
    if not isinstance(value, str):
        result.add_error(
            path, "Path must be a string", current_value=value, expected_type="str"
        )
        return

    try:
        p = Path(value)
        # Check for invalid characters on Windows
        if any(c in str(p) for c in '<>"|?*'):
            result.add_error(
                path,
                "Path contains invalid characters",
                suggestion='Remove special characters: < > " | ? *',
                current_value=value,
            )
    except Exception:
        result.add_error(path, "Invalid path format", current_value=value)


def validate_storage_backend(value: Any, path: str, result: ValidationResult) -> None:
    """Validate storage backend setting."""
    valid_backends = {"sqlite", "json", "memory"}
    if not isinstance(value, str):
        result.add_error(
            path,
            "Storage backend must be a string",
            current_value=value,
            expected_type="str",
        )
    elif value.lower() not in valid_backends:
        result.add_error(
            path,
            f"Invalid storage backend: {value}",
            suggestion=f"Use one of: {', '.join(sorted(valid_backends))}",
            current_value=value,
        )


def validate_model_name(value: Any, path: str, result: ValidationResult) -> None:
    """Validate LLM model name."""
    if not value:  # Empty is allowed (will use provider default)
        return
    if not isinstance(value, str):
        result.add_error(
            path,
            "Model name must be a string",
            current_value=value,
            expected_type="str",
        )
        return

    # Check for common model name patterns
    common_patterns = [
        r"^gpt-[34]",  # GPT models
        r"^claude-",  # Claude models
        r"^llama",  # Llama models
        r"^mistral",  # Mistral models
        r"^gemma",  # Gemma models
    ]

    if not any(re.match(p, value, re.IGNORECASE) for p in common_patterns):
        result.add_info(
            path,
            f"Model name '{value}' doesn't match common patterns",
            suggestion="This may be a custom or local model",
        )


def validate_env_var_name(value: Any, path: str, result: ValidationResult) -> None:
    """Validate environment variable name."""
    if not value:  # Empty is allowed
        return
    if not isinstance(value, str):
        result.add_error(
            path,
            "Environment variable name must be a string",
            current_value=value,
            expected_type="str",
        )
        return

    if not re.match(r"^[A-Z_][A-Z0-9_]*$", value):
        result.add_warning(
            path,
            "Environment variable name doesn't follow conventions",
            suggestion="Use UPPER_CASE_WITH_UNDERSCORES format",
            current_value=value,
        )


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================


def validate_settings(config: dict[str, Any]) -> ValidationResult:
    """Validate entire configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        ValidationResult with all issues found
    """
    result = ValidationResult()

    # General settings
    general = config.get("general", {})
    if isinstance(general, dict):
        validate_verbosity(
            general.get("verbosity", "info"), "general.verbosity", result
        )
        validate_output_format(
            general.get("output_format", "text"), "general.output_format", result
        )
        validate_path(general.get("data_dir", ""), "general.data_dir", result)
        validate_storage_backend(
            general.get("storage_backend", "sqlite"), "general.storage_backend", result
        )

        if not isinstance(general.get("color_enabled", True), bool):
            result.add_error(
                "general.color_enabled",
                "color_enabled must be a boolean (true/false)",
                current_value=general.get("color_enabled"),
                expected_type="bool",
            )
    else:
        result.add_error("general", "General settings must be a dictionary")

    # Backend settings
    backends = config.get("backends", {})
    if isinstance(backends, dict):
        validate_backend(
            backends.get("default_backend", "auto"), "backends.default_backend", result
        )
        validate_timeout(
            backends.get("timeout_seconds", 300), "backends.timeout_seconds", result
        )

        if not isinstance(backends.get("parallel_execution", False), bool):
            result.add_error(
                "backends.parallel_execution",
                "parallel_execution must be a boolean (true/false)",
                current_value=backends.get("parallel_execution"),
                expected_type="bool",
            )
    else:
        result.add_error("backends", "Backend settings must be a dictionary")

    # LLM settings
    llm = config.get("llm", {})
    if isinstance(llm, dict):
        validate_llm_provider(llm.get("provider", "none"), "llm.provider", result)
        validate_model_name(llm.get("model", ""), "llm.model", result)
        validate_url(llm.get("local_endpoint", ""), "llm.local_endpoint", result)
        validate_env_var_name(
            llm.get("api_key_env_var", ""), "llm.api_key_env_var", result
        )

        if not isinstance(llm.get("require_consent", True), bool):
            result.add_error(
                "llm.require_consent",
                "require_consent must be a boolean (true/false)",
                current_value=llm.get("require_consent"),
                expected_type="bool",
            )
    else:
        result.add_error("llm", "LLM settings must be a dictionary")

    # Resources settings
    resources = config.get("resources", {})
    if isinstance(resources, dict):
        validate_memory_threshold(
            resources.get("memory_warn_threshold_mb", 4096),
            "resources.memory_warn_threshold_mb",
            result,
        )
        validate_memory_threshold(
            resources.get("memory_critical_threshold_mb", 8192),
            "resources.memory_critical_threshold_mb",
            result,
        )
        validate_timeout(
            resources.get("max_execution_time_seconds", 3600),
            "resources.max_execution_time_seconds",
            result,
        )

        # Cross-field validation: warn should be less than critical
        warn_mb = resources.get("memory_warn_threshold_mb", 4096)
        critical_mb = resources.get("memory_critical_threshold_mb", 8192)
        if isinstance(warn_mb, int) and isinstance(critical_mb, int):
            if warn_mb >= critical_mb:
                result.add_error(
                    "resources.memory_warn_threshold_mb",
                    "Warning threshold should be less than critical threshold",
                    suggestion=f"Set warning below {critical_mb} MB",
                    current_value=warn_mb,
                )
    else:
        result.add_error("resources", "Resources settings must be a dictionary")

    # Consent settings
    consent = config.get("consent", {})
    if isinstance(consent, dict):
        for key in [
            "auto_approve_local_llm",
            "auto_approve_remote_llm",
            "remember_decisions",
        ]:
            if key in consent and not isinstance(consent[key], bool):
                result.add_error(
                    f"consent.{key}",
                    f"{key} must be a boolean (true/false)",
                    current_value=consent.get(key),
                    expected_type="bool",
                )
    else:
        result.add_error("consent", "Consent settings must be a dictionary")

    return result


def validate_config_file(path: Path) -> ValidationResult:
    """Validate a configuration file.

    Args:
        path: Path to YAML configuration file

    Returns:
        ValidationResult with all issues found
    """
    import yaml

    result = ValidationResult()

    if not path.exists():
        result.add_info(
            str(path),
            "Configuration file not found",
            suggestion="Using default configuration",
        )
        return result

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.add_error(
            str(path),
            f"Invalid YAML syntax: {e}",
            suggestion="Check YAML formatting (indentation, colons, etc.)",
        )
        return result
    except Exception as e:
        result.add_error(str(path), f"Could not read file: {e}")
        return result

    if config is None:
        result.add_info(str(path), "Configuration file is empty")
        return result

    if not isinstance(config, dict):
        result.add_error(
            str(path),
            "Configuration must be a YAML dictionary/mapping",
            current_value=type(config).__name__,
        )
        return result

    # Validate contents
    content_result = validate_settings(config)
    result.merge(content_result)

    return result
