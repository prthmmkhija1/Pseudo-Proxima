"""General helper functions.

Provides commonly used utility functions across the Proxima codebase:
- String formatting and manipulation
- Time and duration utilities
- Data validation helpers
- Safe type conversions
- Path and file utilities
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


# =============================================================================
# STRING UTILITIES
# =============================================================================


def slugify(value: str, allow_unicode: bool = False) -> str:
    """Convert a string to a URL-safe slug.

    Args:
        value: String to slugify.
        allow_unicode: Allow unicode characters if True.

    Returns:
        Slugified string.

    Example:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("Quantum Circuit #1")
        'quantum-circuit-1'
    """
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def truncate(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to add when truncated.

    Returns:
        Truncated text.

    Example:
        >>> truncate("A very long string", 10)
        'A very...'
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase.

    Args:
        snake_str: String in snake_case.

    Returns:
        String in camelCase.

    Example:
        >>> snake_to_camel("hello_world")
        'helloWorld'
    """
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_to_snake(camel_str: str) -> str:
    """Convert camelCase to snake_case.

    Args:
        camel_str: String in camelCase.

    Returns:
        String in snake_case.

    Example:
        >>> camel_to_snake("helloWorld")
        'hello_world'
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_str).lower()


def pluralize(word: str, count: int) -> str:
    """Simple pluralization for English words.

    Args:
        word: Singular word.
        count: Count to determine plural.

    Returns:
        Pluralized word if count != 1.

    Example:
        >>> pluralize("circuit", 5)
        'circuits'
        >>> pluralize("circuit", 1)
        'circuit'
    """
    if count == 1:
        return word
    # Handle common irregular plurals
    irregulars = {
        "qubit": "qubits",
        "matrix": "matrices",
        "basis": "bases",
    }
    if word.lower() in irregulars:
        return irregulars[word.lower()]
    # Basic English rules
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


# =============================================================================
# TIME AND DURATION UTILITIES
# =============================================================================


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable duration.

    Example:
        >>> format_duration(125.5)
        '2m 5s'
        >>> format_duration(3661)
        '1h 1m 1s'
        >>> format_duration(0.05)
        '50ms'
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def format_timestamp(dt: datetime | None = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime to string.

    Args:
        dt: Datetime to format. Uses current time if None.
        fmt: Format string.

    Returns:
        Formatted datetime string.

    Example:
        >>> format_timestamp(datetime(2024, 1, 15, 10, 30))
        '2024-01-15 10:30:00'
    """
    dt = dt or datetime.utcnow()
    return dt.strftime(fmt)


def parse_duration(duration_str: str) -> timedelta:
    """Parse a duration string to timedelta.

    Supports formats: "30s", "5m", "2h", "1d", or combinations like "1h30m".

    Args:
        duration_str: Duration string to parse.

    Returns:
        Parsed timedelta.

    Raises:
        ValueError: If format is invalid.

    Example:
        >>> parse_duration("1h30m")
        datetime.timedelta(seconds=5400)
    """
    pattern = r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
    match = re.fullmatch(pattern, duration_str.strip())

    if not match or not any(match.groups()):
        raise ValueError(f"Invalid duration format: {duration_str}")

    days = int(match.group(1) or 0)
    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = int(match.group(4) or 0)

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def time_ago(dt: datetime) -> str:
    """Get a human-readable "time ago" string.

    Args:
        dt: Past datetime.

    Returns:
        Human-readable relative time.

    Example:
        >>> time_ago(datetime.utcnow() - timedelta(minutes=5))
        '5 minutes ago'
    """
    now = datetime.utcnow()
    diff = now - dt

    seconds = diff.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        mins = int(seconds // 60)
        return f"{mins} {pluralize('minute', mins)} ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} {pluralize('hour', hours)} ago"
    if seconds < 604800:
        days = int(seconds // 86400)
        return f"{days} {pluralize('day', days)} ago"
    if seconds < 2592000:
        weeks = int(seconds // 604800)
        return f"{weeks} {pluralize('week', weeks)} ago"

    return format_timestamp(dt, "%b %d, %Y")


# =============================================================================
# NUMERIC UTILITIES
# =============================================================================


def format_bytes(num_bytes: int | float) -> str:
    """Format bytes to human-readable size.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Human-readable size string.

    Example:
        >>> format_bytes(1536)
        '1.5 KB'
        >>> format_bytes(1_500_000_000)
        '1.40 GB'
    """
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}" if unit != "B" else f"{int(num_bytes)} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} EB"


def format_number(num: int | float, decimals: int = 2) -> str:
    """Format large numbers with K, M, B suffixes.

    Args:
        num: Number to format.
        decimals: Decimal places.

    Returns:
        Formatted number string.

    Example:
        >>> format_number(1500)
        '1.50K'
        >>> format_number(2_500_000)
        '2.50M'
    """
    for suffix, threshold in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(num) >= threshold:
            return f"{num / threshold:.{decimals}f}{suffix}"
    return str(num)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max.

    Args:
        value: Value to clamp.
        min_val: Minimum value.
        max_val: Maximum value.

    Returns:
        Clamped value.

    Example:
        >>> clamp(15, 0, 10)
        10
    """
    return max(min_val, min(max_val, value))


def percentage(part: float, total: float, decimals: int = 1) -> str:
    """Calculate percentage with formatting.

    Args:
        part: Part value.
        total: Total value.
        decimals: Decimal places.

    Returns:
        Formatted percentage string.

    Example:
        >>> percentage(25, 100)
        '25.0%'
    """
    if total == 0:
        return "0%"
    pct = (part / total) * 100
    return f"{pct:.{decimals}f}%"


# =============================================================================
# TYPE CONVERSION UTILITIES
# =============================================================================


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int.

    Args:
        value: Value to convert.
        default: Default if conversion fails.

    Returns:
        Converted int or default.

    Example:
        >>> safe_int("42")
        42
        >>> safe_int("invalid", -1)
        -1
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float.

    Args:
        value: Value to convert.
        default: Default if conversion fails.

    Returns:
        Converted float or default.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert a value to bool.

    Recognizes: "true", "false", "yes", "no", "1", "0", etc.

    Args:
        value: Value to convert.
        default: Default if conversion fails.

    Returns:
        Converted bool or default.

    Example:
        >>> safe_bool("yes")
        True
        >>> safe_bool("0")
        False
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1", "on", "enabled"):
            return True
        if value.lower() in ("false", "no", "0", "off", "disabled"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def coalesce(*values: T | None) -> T | None:
    """Return the first non-None value.

    Args:
        *values: Values to check.

    Returns:
        First non-None value, or None if all are None.

    Example:
        >>> coalesce(None, None, "default")
        'default'
    """
    for value in values:
        if value is not None:
            return value
    return None


# =============================================================================
# PATH AND FILE UTILITIES
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object.

    Example:
        >>> ensure_dir("/tmp/proxima/data")
        PosixPath('/tmp/proxima/data')
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(filename: str, replacement: str = "_") -> str:
    """Create a safe filename by removing invalid characters.

    Args:
        filename: Original filename.
        replacement: Character to replace invalid chars with.

    Returns:
        Safe filename.

    Example:
        >>> safe_filename('file<name>.txt')
        'file_name_.txt'
    """
    # Characters not allowed in filenames on various OSes
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, replacement)
    # Also handle control characters
    filename = "".join(c if ord(c) >= 32 else replacement for c in filename)
    return filename.strip(". ")


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """Calculate hash of a file.

    Args:
        path: File path.
        algorithm: Hash algorithm (md5, sha1, sha256).

    Returns:
        Hex digest of the file.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_size(path: str | Path) -> int:
    """Get file size in bytes.

    Args:
        path: File path.

    Returns:
        File size in bytes, or 0 if file doesn't exist.
    """
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# =============================================================================
# DICTIONARY UTILITIES
# =============================================================================


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary.

    Example:
        >>> deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
        {'a': {'b': 1, 'c': 2}}
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys.
        sep: Separator for nested keys.

    Returns:
        Flattened dictionary.

    Example:
        >>> flatten_dict({"a": {"b": {"c": 1}}})
        {'a.b.c': 1}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_nested(d: dict[str, Any], path: str, default: Any = None, sep: str = ".") -> Any:
    """Get a value from a nested dictionary using dot notation.

    Args:
        d: Dictionary to search.
        path: Dot-separated path to value.
        default: Default value if path not found.
        sep: Path separator.

    Returns:
        Value at path or default.

    Example:
        >>> get_nested({"a": {"b": {"c": 1}}}, "a.b.c")
        1
    """
    keys = path.split(sep)
    result: Any = d
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


def set_nested(d: dict[str, Any], path: str, value: Any, sep: str = ".") -> None:
    """Set a value in a nested dictionary using dot notation.

    Args:
        d: Dictionary to modify.
        path: Dot-separated path.
        value: Value to set.
        sep: Path separator.

    Example:
        >>> d = {}
        >>> set_nested(d, "a.b.c", 1)
        >>> d
        {'a': {'b': {'c': 1}}}
    """
    keys = path.split(sep)
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def is_valid_identifier(name: str) -> bool:
    """Check if a string is a valid Python identifier.

    Args:
        name: String to check.

    Returns:
        True if valid identifier.

    Example:
        >>> is_valid_identifier("my_variable")
        True
        >>> is_valid_identifier("123abc")
        False
    """
    return name.isidentifier()


def is_valid_email(email: str) -> bool:
    """Basic email validation.

    Args:
        email: Email address to validate.

    Returns:
        True if valid format.
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """Basic URL validation.

    Args:
        url: URL to validate.

    Returns:
        True if valid format.
    """
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(pattern, url, re.IGNORECASE))


def validate_range(
    value: float, min_val: float | None = None, max_val: float | None = None
) -> bool:
    """Validate a value is within a range.

    Args:
        value: Value to validate.
        min_val: Minimum (inclusive), or None for no minimum.
        max_val: Maximum (inclusive), or None for no maximum.

    Returns:
        True if value is in range.
    """
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True
