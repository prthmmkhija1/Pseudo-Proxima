"""Configuration schema documentation and introspection.

Provides utilities for:
- Generating documentation from settings classes
- Schema introspection for CLI help and validation
- JSON Schema generation for external tools
- Auto-completion data generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel


class FieldType(Enum):
    """Types of configuration fields."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"
    PATH = "path"
    URL = "url"
    DURATION = "duration"
    UNKNOWN = "unknown"


@dataclass
class FieldInfo:
    """Information about a configuration field."""

    name: str
    path: str  # Full dot-separated path
    field_type: FieldType
    python_type: str
    default: Any = None
    description: str = ""
    required: bool = False
    deprecated: bool = False
    deprecated_message: str = ""
    enum_values: list[str] = field(default_factory=list)
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None  # Regex pattern for validation
    examples: list[Any] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "path": self.path,
            "type": self.field_type.value,
            "python_type": self.python_type,
            "default": self.default,
            "description": self.description,
            "required": self.required,
            "deprecated": self.deprecated,
            "enum_values": self.enum_values if self.enum_values else None,
            "examples": self.examples if self.examples else None,
        }


@dataclass
class SectionInfo:
    """Information about a configuration section."""

    name: str
    path: str
    description: str = ""
    fields: list[FieldInfo] = field(default_factory=list)
    subsections: list[SectionInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
            "subsections": [s.to_dict() for s in self.subsections],
        }


# =============================================================================
# SCHEMA INTROSPECTION
# =============================================================================

# Field descriptions for documentation
FIELD_DESCRIPTIONS: dict[str, str] = {
    # General
    "general.verbosity": "Logging verbosity level. Controls how much information is output.",
    "general.output_format": "Output format for results. 'text' for human-readable, 'json' for machine-readable, 'rich' for colorized terminal output.",
    "general.color_enabled": "Enable colored output in terminal. Set to false for piping output.",
    "general.data_dir": "Directory for storing data files. Empty uses default location.",
    "general.storage_backend": "Backend for storing results and history. Options: sqlite, json, memory.",
    # Backends
    "backends.default_backend": "Default quantum backend to use. 'auto' selects automatically based on circuit.",
    "backends.parallel_execution": "Enable parallel execution on multiple backends.",
    "backends.timeout_seconds": "Maximum time to wait for backend execution before timeout.",
    # LLM
    "llm.provider": "LLM provider for AI-assisted features. Options: none, openai, anthropic, ollama, lmstudio.",
    "llm.model": "Model name to use. Leave empty for provider default.",
    "llm.local_endpoint": "URL for local LLM server (ollama, lmstudio).",
    "llm.api_key_env_var": "Environment variable name containing the API key.",
    "llm.require_consent": "Require user consent before making LLM API calls.",
    # Resources
    "resources.memory_warn_threshold_mb": "Memory usage warning threshold in megabytes.",
    "resources.memory_critical_threshold_mb": "Critical memory threshold that triggers action.",
    "resources.max_execution_time_seconds": "Maximum allowed execution time before forced stop.",
    # Consent
    "consent.auto_approve_local_llm": "Automatically approve local LLM usage without prompting.",
    "consent.auto_approve_remote_llm": "Automatically approve remote LLM usage without prompting.",
    "consent.remember_decisions": "Remember consent decisions for future sessions.",
}

FIELD_EXAMPLES: dict[str, list[Any]] = {
    "general.verbosity": ["debug", "info", "warning", "error"],
    "general.output_format": ["text", "json", "rich"],
    "backends.default_backend": ["auto", "cirq", "qiskit", "lret"],
    "backends.timeout_seconds": [60, 300, 600, 3600],
    "llm.provider": ["none", "openai", "anthropic", "ollama"],
    "llm.model": ["gpt-4", "claude-3-opus", "llama2", "mistral"],
    "llm.local_endpoint": ["http://localhost:11434", "http://localhost:1234/v1"],
}


def _python_type_to_field_type(python_type: type) -> FieldType:
    """Convert Python type to FieldType."""
    origin = get_origin(python_type)

    if origin is Union:
        # Handle Optional types
        args = get_args(python_type)
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _python_type_to_field_type(non_none[0])

    if python_type is str:
        return FieldType.STRING
    elif python_type is int:
        return FieldType.INTEGER
    elif python_type is float:
        return FieldType.FLOAT
    elif python_type is bool:
        return FieldType.BOOLEAN
    elif origin is list:
        return FieldType.ARRAY
    elif origin is dict or isinstance(python_type, type) and issubclass(python_type, dict):
        return FieldType.OBJECT
    elif isinstance(python_type, type) and issubclass(python_type, Enum):
        return FieldType.ENUM

    return FieldType.UNKNOWN


def _type_to_string(python_type: type) -> str:
    """Convert Python type to string representation."""
    origin = get_origin(python_type)

    if origin is Union:
        args = get_args(python_type)
        arg_strs = [_type_to_string(a) for a in args if a is not type(None)]
        if len(arg_strs) == 1:
            return f"{arg_strs[0]} | None"
        return " | ".join(arg_strs)

    if origin is list:
        args = get_args(python_type)
        if args:
            return f"list[{_type_to_string(args[0])}]"
        return "list"

    if origin is dict:
        args = get_args(python_type)
        if len(args) == 2:
            return f"dict[{_type_to_string(args[0])}, {_type_to_string(args[1])}]"
        return "dict"

    if hasattr(python_type, "__name__"):
        return python_type.__name__

    return str(python_type)


def introspect_model(model_class: type[BaseModel], prefix: str = "") -> SectionInfo:
    """Introspect a Pydantic model to extract schema information.

    Args:
        model_class: Pydantic model class to introspect
        prefix: Path prefix for nested models

    Returns:
        SectionInfo with field and subsection information
    """
    section = SectionInfo(
        name=model_class.__name__.replace("Settings", "").lower() or "root",
        path=prefix or "root",
        description=model_class.__doc__ or "",
    )

    # Get field information from Pydantic model
    hints = get_type_hints(model_class)
    model_fields = model_class.model_fields

    for field_name, field_info in model_fields.items():
        field_path = f"{prefix}.{field_name}" if prefix else field_name
        python_type = hints.get(field_name, type(None))

        # Check if this is a nested model
        if isinstance(python_type, type) and issubclass(python_type, BaseModel):
            subsection = introspect_model(python_type, field_path)
            section.subsections.append(subsection)
        else:
            field = FieldInfo(
                name=field_name,
                path=field_path,
                field_type=_python_type_to_field_type(python_type),
                python_type=_type_to_string(python_type),
                default=field_info.default,
                description=FIELD_DESCRIPTIONS.get(field_path, field_info.description or ""),
                required=field_info.is_required(),
                examples=FIELD_EXAMPLES.get(field_path, []),
            )
            section.fields.append(field)

    return section


# =============================================================================
# SCHEMA DOCUMENTATION
# =============================================================================


def generate_markdown_docs(section: SectionInfo, level: int = 1) -> str:
    """Generate Markdown documentation from schema.

    Args:
        section: SectionInfo to document
        level: Heading level (1-6)

    Returns:
        Markdown string
    """
    lines = []

    # Section header
    heading = "#" * min(level, 6)
    section_title = section.name.replace("_", " ").title()
    lines.append(f"{heading} {section_title}")
    lines.append("")

    if section.description:
        lines.append(section.description)
        lines.append("")

    # Fields table
    if section.fields:
        lines.append("| Setting | Type | Default | Description |")
        lines.append("|---------|------|---------|-------------|")

        for field in section.fields:
            default_str = _format_default(field.default)
            desc = field.description.replace("|", "\\|")
            if field.deprecated:
                desc = f"⚠️ **Deprecated**: {desc}"

            lines.append(f"| `{field.name}` | {field.python_type} | {default_str} | {desc} |")

        lines.append("")

    # Examples
    if section.fields:
        has_examples = any(f.examples for f in section.fields)
        if has_examples:
            lines.append("**Example values:**")
            lines.append("")
            lines.append("```yaml")
            lines.append(f"{section.name}:")
            for field in section.fields:
                if field.examples:
                    example = field.examples[0]
                    lines.append(f"  {field.name}: {_format_yaml_value(example)}")
            lines.append("```")
            lines.append("")

    # Subsections
    for subsection in section.subsections:
        lines.append(generate_markdown_docs(subsection, level + 1))

    return "\n".join(lines)


def _format_default(value: Any) -> str:
    """Format default value for documentation."""
    if value is None:
        return "`None`"
    if isinstance(value, str):
        if not value:
            return '`""`'
        return f'`"{value}"`'
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    return f"`{value}`"


def _format_yaml_value(value: Any) -> str:
    """Format value for YAML example."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


# =============================================================================
# JSON SCHEMA GENERATION
# =============================================================================


def generate_json_schema(section: SectionInfo) -> dict[str, Any]:
    """Generate JSON Schema from section info.

    Args:
        section: SectionInfo to convert

    Returns:
        JSON Schema dictionary
    """
    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": f"Proxima Configuration - {section.name}",
        "description": section.description,
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    required = []

    for fld in section.fields:
        prop = _field_to_json_schema(fld)
        schema["properties"][fld.name] = prop
        if fld.required:
            required.append(fld.name)

    for subsection in section.subsections:
        sub_schema = generate_json_schema(subsection)
        # Remove top-level meta
        sub_schema.pop("$schema", None)
        schema["properties"][subsection.name] = sub_schema

    if required:
        schema["required"] = required

    return schema


def _field_to_json_schema(field: FieldInfo) -> dict[str, Any]:
    """Convert FieldInfo to JSON Schema property."""
    type_mapping = {
        FieldType.STRING: "string",
        FieldType.INTEGER: "integer",
        FieldType.FLOAT: "number",
        FieldType.BOOLEAN: "boolean",
        FieldType.ARRAY: "array",
        FieldType.OBJECT: "object",
        FieldType.PATH: "string",
        FieldType.URL: "string",
        FieldType.DURATION: "string",
    }

    prop: dict[str, Any] = {
        "description": field.description,
    }

    if field.field_type == FieldType.ENUM and field.enum_values:
        prop["enum"] = field.enum_values
    else:
        prop["type"] = type_mapping.get(field.field_type, "string")

    if field.default is not None:
        prop["default"] = field.default

    if field.min_value is not None:
        prop["minimum"] = field.min_value

    if field.max_value is not None:
        prop["maximum"] = field.max_value

    if field.pattern:
        prop["pattern"] = field.pattern

    if field.examples:
        prop["examples"] = field.examples

    if field.deprecated:
        prop["deprecated"] = True

    return prop


# =============================================================================
# AUTO-COMPLETION DATA
# =============================================================================


def generate_completion_data(section: SectionInfo) -> dict[str, Any]:
    """Generate auto-completion data for IDE/CLI.

    Args:
        section: SectionInfo to process

    Returns:
        Completion data dictionary
    """
    completions: dict[str, Any] = {
        "keys": [],
        "values": {},
    }

    def collect_fields(sec: SectionInfo, prefix: str = "") -> None:
        for fld in sec.fields:
            full_path = f"{prefix}.{fld.name}" if prefix else fld.name
            completions["keys"].append(
                {
                    "key": full_path,
                    "description": fld.description,
                    "type": fld.field_type.value,
                }
            )

            if fld.examples:
                completions["values"][full_path] = fld.examples
            elif fld.enum_values:
                completions["values"][full_path] = fld.enum_values

        for subsection in sec.subsections:
            sub_prefix = f"{prefix}.{subsection.name}" if prefix else subsection.name
            collect_fields(subsection, sub_prefix)

    collect_fields(section)
    return completions


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_settings_schema() -> SectionInfo:
    """Get schema for the main Settings class."""
    from proxima.config.settings import Settings

    return introspect_model(Settings)


def get_field_help(path: str) -> str | None:
    """Get help text for a specific field path.

    Args:
        path: Dot-separated path like "general.verbosity"

    Returns:
        Help text or None if not found
    """
    return FIELD_DESCRIPTIONS.get(path)


def get_field_examples(path: str) -> list[Any]:
    """Get example values for a specific field path.

    Args:
        path: Dot-separated path like "general.verbosity"

    Returns:
        List of example values
    """
    return FIELD_EXAMPLES.get(path, [])


def list_all_settings() -> list[str]:
    """Get a flat list of all setting paths.

    Returns:
        List of dot-separated setting paths
    """
    schema = get_settings_schema()
    paths = []

    def collect_paths(section: SectionInfo, prefix: str = "") -> None:
        for fld in section.fields:
            path = f"{prefix}.{fld.name}" if prefix else fld.name
            paths.append(path)
        for subsection in section.subsections:
            sub_prefix = f"{prefix}.{subsection.name}" if prefix else subsection.name
            collect_paths(subsection, sub_prefix)

    collect_paths(schema)
    return paths


def print_settings_tree() -> str:
    """Generate a tree view of all settings.

    Returns:
        Tree representation string
    """
    lines = ["Proxima Configuration Settings", "=" * 40, ""]

    def print_section(section: SectionInfo, indent: int = 0) -> None:
        prefix = "  " * indent

        for fld in section.fields:
            type_str = f"({fld.python_type})"
            default_str = f"= {fld.default!r}" if fld.default is not None else ""
            lines.append(f"{prefix}├── {fld.name} {type_str} {default_str}")

        for i, subsection in enumerate(section.subsections):
            is_last = i == len(section.subsections) - 1
            connector = "└──" if is_last else "├──"
            lines.append(f"{prefix}{connector} {subsection.name}/")
            print_section(subsection, indent + 1)

    schema = get_settings_schema()
    print_section(schema)

    return "\n".join(lines)
