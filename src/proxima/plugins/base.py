"""
Plugin base classes and interfaces.

Defines the plugin protocol and metadata structures.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Any, ClassVar


# =============================================================================
# PLUGIN VERSIONING SYSTEM (5% Gap Coverage)
# =============================================================================


class VersionParseError(Exception):
    """Error parsing version string."""
    pass


@total_ordering
@dataclass
class SemanticVersion:
    """Semantic versioning (SemVer 2.0) implementation.
    
    Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
    
    Examples:
        - 1.0.0
        - 2.1.3-alpha.1
        - 3.0.0-beta.2+build.123
    """
    
    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None
    
    # Regex pattern for semantic version parsing
    VERSION_PATTERN = re.compile(
        r'^(?P<major>0|[1-9]\d*)'
        r'\.(?P<minor>0|[1-9]\d*)'
        r'\.(?P<patch>0|[1-9]\d*)'
        r'(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)'
        r'(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*?))?'
        r'(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    )
    
    # Prerelease precedence order (lower = earlier)
    PRERELEASE_ORDER = ["alpha", "beta", "rc", "preview"]
    
    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse a version string into SemanticVersion.
        
        Args:
            version_str: Version string like "1.2.3" or "2.0.0-alpha.1"
            
        Returns:
            Parsed SemanticVersion
            
        Raises:
            VersionParseError: If version string is invalid
        """
        if not version_str:
            raise VersionParseError("Empty version string")
        
        # Strip 'v' prefix if present
        if version_str.startswith(('v', 'V')):
            version_str = version_str[1:]
        
        match = cls.VERSION_PATTERN.match(version_str.strip())
        if not match:
            # Try simple parsing for basic versions like "1.0"
            simple_match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?$', version_str.strip())
            if simple_match:
                return cls(
                    major=int(simple_match.group(1)),
                    minor=int(simple_match.group(2)),
                    patch=int(simple_match.group(3) or 0),
                )
            raise VersionParseError(f"Invalid version string: {version_str}")
        
        return cls(
            major=int(match.group('major')),
            minor=int(match.group('minor')),
            patch=int(match.group('patch')),
            prerelease=match.group('prerelease'),
            build=match.group('build'),
        )
    
    def __str__(self) -> str:
        """Convert to string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __repr__(self) -> str:
        return f"SemanticVersion({self})"
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = SemanticVersion.parse(other)
            except VersionParseError:
                return False
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        # Build metadata is ignored in equality
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )
    
    def __lt__(self, other: object) -> bool:
        if isinstance(other, str):
            other = SemanticVersion.parse(other)
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        
        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # Prerelease versions have lower precedence
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if not self.prerelease and not other.prerelease:
            return False
        
        # Compare prerelease identifiers
        return self._compare_prerelease(self.prerelease, other.prerelease) < 0
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))
    
    def _compare_prerelease(self, a: str | None, b: str | None) -> int:
        """Compare prerelease identifiers."""
        if a == b:
            return 0
        if a is None:
            return 1
        if b is None:
            return -1
        
        parts_a = a.split('.')
        parts_b = b.split('.')
        
        for i in range(max(len(parts_a), len(parts_b))):
            if i >= len(parts_a):
                return -1
            if i >= len(parts_b):
                return 1
            
            pa, pb = parts_a[i], parts_b[i]
            
            # Numeric comparison
            if pa.isdigit() and pb.isdigit():
                if int(pa) != int(pb):
                    return int(pa) - int(pb)
            # Named prerelease (alpha, beta, rc)
            elif pa in self.PRERELEASE_ORDER and pb in self.PRERELEASE_ORDER:
                idx_a = self.PRERELEASE_ORDER.index(pa)
                idx_b = self.PRERELEASE_ORDER.index(pb)
                if idx_a != idx_b:
                    return idx_a - idx_b
            else:
                # Lexicographic comparison
                if pa != pb:
                    return -1 if pa < pb else 1
        
        return 0
    
    def is_prerelease(self) -> bool:
        """Check if this is a prerelease version."""
        return self.prerelease is not None
    
    def is_stable(self) -> bool:
        """Check if this is a stable release (major >= 1, no prerelease)."""
        return self.major >= 1 and not self.is_prerelease()
    
    def bump_major(self) -> "SemanticVersion":
        """Return new version with bumped major."""
        return SemanticVersion(self.major + 1, 0, 0)
    
    def bump_minor(self) -> "SemanticVersion":
        """Return new version with bumped minor."""
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def bump_patch(self) -> "SemanticVersion":
        """Return new version with bumped patch."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)
    
    def with_prerelease(self, prerelease: str) -> "SemanticVersion":
        """Return new version with prerelease tag."""
        return SemanticVersion(
            self.major, self.minor, self.patch, prerelease, self.build
        )
    
    def with_build(self, build: str) -> "SemanticVersion":
        """Return new version with build metadata."""
        return SemanticVersion(
            self.major, self.minor, self.patch, self.prerelease, build
        )
    
    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Check if this version is compatible with another (same major)."""
        return self.major == other.major
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "build": self.build,
            "string": str(self),
        }


class VersionConstraint(str, Enum):
    """Types of version constraints."""
    
    EXACT = "=="        # Exact match
    GREATER = ">"       # Greater than
    GREATER_EQ = ">="   # Greater or equal
    LESS = "<"          # Less than
    LESS_EQ = "<="      # Less or equal
    COMPATIBLE = "^"    # Compatible (same major)
    TILDE = "~"         # Approximate (same major.minor)
    ANY = "*"           # Any version


@dataclass
class VersionRange:
    """Represents a version range constraint.
    
    Examples:
        - ">=1.0.0"
        - "^2.0.0" (compatible with 2.x.x)
        - "~1.2.0" (approximately 1.2.x)
        - ">=1.0.0 <2.0.0"
        - "*" (any version)
    """
    
    constraint: VersionConstraint
    version: SemanticVersion | None
    max_version: SemanticVersion | None = None  # For range constraints
    
    # Pattern for parsing version constraints
    CONSTRAINT_PATTERN = re.compile(
        r'^(?P<op>>=|<=|>|<|==|\^|~|\*)?(?P<version>[\d\w.\-+]+)?$'
    )
    
    @classmethod
    def parse(cls, constraint_str: str) -> "VersionRange":
        """Parse a version constraint string.
        
        Args:
            constraint_str: Constraint like ">=1.0.0" or "^2.0.0"
            
        Returns:
            Parsed VersionRange
        """
        constraint_str = constraint_str.strip()
        
        if constraint_str == "*" or not constraint_str:
            return cls(VersionConstraint.ANY, None)
        
        # Check for range (e.g., ">=1.0.0 <2.0.0")
        if " " in constraint_str:
            parts = constraint_str.split()
            if len(parts) == 2:
                min_range = cls.parse(parts[0])
                max_range = cls.parse(parts[1])
                return cls(
                    min_range.constraint,
                    min_range.version,
                    max_range.version,
                )
        
        match = cls.CONSTRAINT_PATTERN.match(constraint_str)
        if not match:
            raise VersionParseError(f"Invalid constraint: {constraint_str}")
        
        op = match.group('op') or "=="
        version_str = match.group('version')
        
        constraint_map = {
            "==": VersionConstraint.EXACT,
            ">": VersionConstraint.GREATER,
            ">=": VersionConstraint.GREATER_EQ,
            "<": VersionConstraint.LESS,
            "<=": VersionConstraint.LESS_EQ,
            "^": VersionConstraint.COMPATIBLE,
            "~": VersionConstraint.TILDE,
            "*": VersionConstraint.ANY,
        }
        
        constraint = constraint_map.get(op, VersionConstraint.EXACT)
        version = SemanticVersion.parse(version_str) if version_str else None
        
        return cls(constraint, version)
    
    def satisfies(self, version: SemanticVersion | str) -> bool:
        """Check if a version satisfies this constraint.
        
        Args:
            version: Version to check
            
        Returns:
            True if version satisfies the constraint
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        
        if self.constraint == VersionConstraint.ANY:
            return True
        
        if self.version is None:
            return True
        
        if self.constraint == VersionConstraint.EXACT:
            return version == self.version
        elif self.constraint == VersionConstraint.GREATER:
            return version > self.version
        elif self.constraint == VersionConstraint.GREATER_EQ:
            result = version >= self.version
            if result and self.max_version:
                result = result and version < self.max_version
            return result
        elif self.constraint == VersionConstraint.LESS:
            return version < self.version
        elif self.constraint == VersionConstraint.LESS_EQ:
            return version <= self.version
        elif self.constraint == VersionConstraint.COMPATIBLE:
            # ^1.2.3 allows >=1.2.3 and <2.0.0
            return (
                version >= self.version
                and version.major == self.version.major
            )
        elif self.constraint == VersionConstraint.TILDE:
            # ~1.2.3 allows >=1.2.3 and <1.3.0
            return (
                version >= self.version
                and version.major == self.version.major
                and version.minor == self.version.minor
            )
        
        return False
    
    def __str__(self) -> str:
        if self.constraint == VersionConstraint.ANY:
            return "*"
        result = f"{self.constraint.value}{self.version}"
        if self.max_version:
            result += f" <{self.max_version}"
        return result


@dataclass
class VersionInfo:
    """Complete version information for a plugin."""
    
    version: SemanticVersion
    min_proxima_version: SemanticVersion | None = None
    max_proxima_version: SemanticVersion | None = None
    deprecated: bool = False
    deprecation_message: str | None = None
    sunset_date: str | None = None  # ISO date string
    upgrade_path: str | None = None  # Next recommended version
    breaking_changes: list[str] = field(default_factory=list)
    changelog: str | None = None
    
    def is_compatible_with_proxima(self, proxima_version: str) -> bool:
        """Check if plugin is compatible with Proxima version."""
        pv = SemanticVersion.parse(proxima_version)
        
        if self.min_proxima_version and pv < self.min_proxima_version:
            return False
        if self.max_proxima_version and pv > self.max_proxima_version:
            return False
        return True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": str(self.version),
            "min_proxima_version": str(self.min_proxima_version) if self.min_proxima_version else None,
            "max_proxima_version": str(self.max_proxima_version) if self.max_proxima_version else None,
            "deprecated": self.deprecated,
            "deprecation_message": self.deprecation_message,
            "sunset_date": self.sunset_date,
            "upgrade_path": self.upgrade_path,
            "breaking_changes": self.breaking_changes,
        }


class VersionRegistry:
    """Registry for plugin versions and compatibility tracking."""
    
    def __init__(self) -> None:
        self._versions: dict[str, list[VersionInfo]] = {}  # plugin_name -> versions
        self._current: dict[str, SemanticVersion] = {}  # plugin_name -> current version
        self._compatibility: dict[str, dict[str, VersionRange]] = {}  # plugin -> {dep -> range}
    
    def register_version(self, plugin_name: str, info: VersionInfo) -> None:
        """Register a plugin version."""
        if plugin_name not in self._versions:
            self._versions[plugin_name] = []
        self._versions[plugin_name].append(info)
        self._versions[plugin_name].sort(key=lambda v: v.version, reverse=True)
    
    def set_current(self, plugin_name: str, version: SemanticVersion) -> None:
        """Set the current version for a plugin."""
        self._current[plugin_name] = version
    
    def get_current(self, plugin_name: str) -> SemanticVersion | None:
        """Get the current version of a plugin."""
        return self._current.get(plugin_name)
    
    def get_latest(self, plugin_name: str, stable_only: bool = True) -> VersionInfo | None:
        """Get the latest version of a plugin."""
        versions = self._versions.get(plugin_name, [])
        for info in versions:
            if stable_only and info.version.is_prerelease():
                continue
            return info
        return versions[0] if versions else None
    
    def get_all_versions(self, plugin_name: str) -> list[VersionInfo]:
        """Get all registered versions for a plugin."""
        return list(self._versions.get(plugin_name, []))
    
    def set_compatibility(
        self,
        plugin_name: str,
        dependency: str,
        version_range: VersionRange,
    ) -> None:
        """Set version compatibility between plugins."""
        if plugin_name not in self._compatibility:
            self._compatibility[plugin_name] = {}
        self._compatibility[plugin_name][dependency] = version_range
    
    def check_compatibility(
        self,
        plugin_name: str,
        dependency: str,
        dep_version: SemanticVersion,
    ) -> bool:
        """Check if dependency version is compatible."""
        if plugin_name not in self._compatibility:
            return True
        if dependency not in self._compatibility[plugin_name]:
            return True
        return self._compatibility[plugin_name][dependency].satisfies(dep_version)
    
    def get_upgrade_path(
        self,
        plugin_name: str,
        from_version: SemanticVersion,
        to_version: SemanticVersion | None = None,
    ) -> list[VersionInfo]:
        """Get recommended upgrade path between versions."""
        versions = self._versions.get(plugin_name, [])
        if not versions:
            return []
        
        target = to_version or versions[0].version
        
        # Find versions between from and to
        path: list[VersionInfo] = []
        for info in reversed(versions):
            if info.version > from_version and info.version <= target:
                # Include versions with breaking changes
                if info.breaking_changes or info.version.major > from_version.major:
                    path.append(info)
        
        return path
    
    def find_compatible_version(
        self,
        plugin_name: str,
        constraint: VersionRange,
    ) -> VersionInfo | None:
        """Find a version that satisfies the constraint."""
        versions = self._versions.get(plugin_name, [])
        for info in versions:
            if constraint.satisfies(info.version):
                return info
        return None


# Global version registry
_version_registry: VersionRegistry | None = None


def get_version_registry() -> VersionRegistry:
    """Get the global version registry."""
    global _version_registry
    if _version_registry is None:
        _version_registry = VersionRegistry()
    return _version_registry


class PluginType(str, Enum):
    """Types of plugins supported by Proxima."""

    BACKEND = "backend"
    LLM_PROVIDER = "llm_provider"
    EXPORTER = "exporter"
    ANALYZER = "analyzer"
    HOOK = "hook"


class PluginError(Exception):
    """Base exception for plugin errors."""

    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""

    pass


class PluginValidationError(PluginError):
    """Raised when a plugin fails validation."""

    pass


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""

    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    homepage: str = ""
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type.value,
            "description": self.description,
            "author": self.author,
            "homepage": self.homepage,
            "requires": self.requires,
            "provides": self.provides,
            "config_schema": self.config_schema,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginMetadata:
        """Create metadata from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            plugin_type=PluginType(data["plugin_type"]),
            description=data.get("description", ""),
            author=data.get("author", ""),
            homepage=data.get("homepage", ""),
            requires=data.get("requires", []),
            provides=data.get("provides", []),
            config_schema=data.get("config_schema"),
        )


@dataclass
class PluginContext:
    """Runtime context for plugin execution."""
    
    backend_name: str
    """Name of the current backend (e.g., 'cirq', 'qiskit_aer')."""
    
    num_qubits: int
    """Number of qubits in the current circuit."""
    
    shots: int = 1000
    """Number of measurement shots."""
    
    config: dict[str, Any] | None = None
    """Additional configuration options."""
    
    session_id: str | None = None
    """Current session identifier, if any."""
    
    metadata: dict[str, Any] | None = None
    """Additional metadata."""


class Plugin(ABC):
    """Base class for all Proxima plugins.

    To create a plugin:
    1. Subclass Plugin or a specific plugin type (BackendPlugin, etc.)
    2. Define the METADATA class variable
    3. Implement required abstract methods
    4. Register via entry_points in pyproject.toml or place in plugins directory
    """

    METADATA: ClassVar[PluginMetadata]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the plugin with optional configuration."""
        self._config = config or {}
        self._enabled = True

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self.METADATA.name

    @property
    def version(self) -> str:
        """Get plugin version."""
        return self.METADATA.version

    @property
    def plugin_type(self) -> PluginType:
        """Get plugin type."""
        return self.METADATA.plugin_type

    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the plugin."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the plugin."""
        self._enabled = False

    def configure(self, config: dict[str, Any]) -> None:
        """Update plugin configuration."""
        self._config.update(config)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    @abstractmethod
    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize the plugin. Called after loading.
        
        Args:
            context: Optional plugin context with runtime information
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up the plugin. Called before unloading."""
        ...

    def validate(self) -> list[str]:
        """Validate plugin configuration. Returns list of error messages."""
        errors: list[str] = []
        if not hasattr(self, "METADATA"):
            errors.append("Plugin missing METADATA class variable")
        return errors
    
    # =========================================================================
    # VERSIONING METHODS (5% Gap Coverage)
    # =========================================================================
    
    def get_semantic_version(self) -> SemanticVersion:
        """Get the plugin version as SemanticVersion."""
        return SemanticVersion.parse(self.version)
    
    def check_version_constraint(self, constraint: str) -> bool:
        """Check if plugin version satisfies a constraint.
        
        Args:
            constraint: Version constraint like ">=1.0.0" or "^2.0.0"
            
        Returns:
            True if version satisfies constraint
        """
        version_range = VersionRange.parse(constraint)
        return version_range.satisfies(self.get_semantic_version())
    
    def is_compatible_with(self, other: "Plugin") -> bool:
        """Check if this plugin is compatible with another plugin.
        
        Uses the version registry to check compatibility rules.
        """
        registry = get_version_registry()
        return registry.check_compatibility(
            self.name,
            other.name,
            other.get_semantic_version(),
        )
    
    def get_version_info(self) -> VersionInfo | None:
        """Get detailed version information from registry."""
        registry = get_version_registry()
        versions = registry.get_all_versions(self.name)
        current = self.get_semantic_version()
        for info in versions:
            if info.version == current:
                return info
        return None
    
    def check_upgrade_available(self) -> tuple[bool, SemanticVersion | None]:
        """Check if an upgrade is available.
        
        Returns:
            Tuple of (upgrade_available, latest_version)
        """
        registry = get_version_registry()
        latest = registry.get_latest(self.name, stable_only=True)
        if latest and latest.version > self.get_semantic_version():
            return True, latest.version
        return False, None
    
    def get_upgrade_path(self, target_version: str | None = None) -> list[VersionInfo]:
        """Get recommended upgrade path to target version.
        
        Args:
            target_version: Target version (latest if None)
            
        Returns:
            List of versions to upgrade through
        """
        registry = get_version_registry()
        target = SemanticVersion.parse(target_version) if target_version else None
        return registry.get_upgrade_path(self.name, self.get_semantic_version(), target)


class BackendPlugin(Plugin):
    """Base class for quantum backend plugins."""

    METADATA: ClassVar[PluginMetadata]

    @abstractmethod
    def get_backend_class(self) -> type[Any]:
        """Return the backend adapter class."""
        ...

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the backend identifier name."""
        ...

    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize backend plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown backend plugin."""
        pass


class LLMProviderPlugin(Plugin):
    """Base class for LLM provider plugins."""

    METADATA: ClassVar[PluginMetadata]

    @abstractmethod
    def get_provider_class(self) -> type[Any]:
        """Return the LLM provider class."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider identifier name."""
        ...

    @abstractmethod
    def get_api_key_env_var(self) -> str | None:
        """Return the environment variable name for API key, or None."""
        ...

    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize LLM provider plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown LLM provider plugin."""
        pass


class ExporterPlugin(Plugin):
    """Base class for result exporter plugins."""

    METADATA: ClassVar[PluginMetadata]

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the export format name (e.g., 'json', 'csv')."""
        ...

    @abstractmethod
    def export(self, data: Any, destination: str) -> None:
        """Export data to the specified destination."""
        ...

    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize exporter plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown exporter plugin."""
        pass


class AnalyzerPlugin(Plugin):
    """Base class for result analyzer plugins."""

    METADATA: ClassVar[PluginMetadata]

    @abstractmethod
    def get_analyzer_name(self) -> str:
        """Return the analyzer identifier name."""
        ...

    @abstractmethod
    def analyze(self, results: Any) -> dict[str, Any]:
        """Analyze results and return insights."""
        ...

    def initialize(self, context: PluginContext | None = None) -> None:
        """Initialize analyzer plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown analyzer plugin."""
        pass

# Alias for backward compatibility
BasePlugin = Plugin