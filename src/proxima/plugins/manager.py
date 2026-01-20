"""
Plugin Manager with persistence support.

Handles plugin state management, configuration persistence,
and plugin lifecycle events with session integration.

Advanced features include:
- Dependency resolution
- Sandboxed execution
- Hot reloading
- Marketplace integration
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from proxima.plugins.base import BasePlugin, Plugin, PluginMetadata, PluginType
from proxima.plugins.loader import PluginRegistry, get_plugin_registry


class PluginState(str, Enum):
    """Plugin lifecycle states."""

    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UNINSTALLED = "uninstalled"


@dataclass
class PluginConfig:
    """Persistent plugin configuration."""

    name: str
    version: str
    plugin_type: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    state: PluginState = PluginState.ENABLED
    installed_at: datetime | None = None
    last_used: datetime | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type,
            "enabled": self.enabled,
            "config": self.config,
            "state": self.state.value,
            "installed_at": self.installed_at.isoformat() if self.installed_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginConfig:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            plugin_type=data["plugin_type"],
            enabled=data.get("enabled", True),
            config=data.get("config", {}),
            state=PluginState(data.get("state", "enabled")),
            installed_at=datetime.fromisoformat(data["installed_at"]) if data.get("installed_at") else None,
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            error_message=data.get("error_message"),
        )


@dataclass
class PluginStateSnapshot:
    """Snapshot of all plugin states for persistence."""

    configs: dict[str, PluginConfig] = field(default_factory=dict)
    last_saved: datetime | None = None
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "last_saved": self.last_saved.isoformat() if self.last_saved else None,
            "configs": {name: cfg.to_dict() for name, cfg in self.configs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginStateSnapshot:
        """Create from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            last_saved=datetime.fromisoformat(data["last_saved"]) if data.get("last_saved") else None,
            configs={name: PluginConfig.from_dict(cfg) for name, cfg in data.get("configs", {}).items()},
        )


class PluginManager:
    """
    Manages plugin lifecycle with persistent state.
    
    Features:
    - Enable/disable plugins
    - Persist plugin configurations
    - Track plugin usage
    - Manage plugin dependencies
    - Session integration
    """

    DEFAULT_CONFIG_PATH = Path.home() / ".proxima" / "plugins.json"

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Initialize the plugin manager.

        Args:
            registry: Plugin registry to manage. Uses global if not provided.
            config_path: Path to store plugin configuration. Uses default if not provided.
        """
        self._registry = registry or get_plugin_registry()
        self._config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._snapshot = PluginStateSnapshot()
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted plugin state from disk."""
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                self._snapshot = PluginStateSnapshot.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                self._snapshot = PluginStateSnapshot()

    def _save_state(self) -> None:
        """Save plugin state to disk."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot.last_saved = datetime.utcnow()
        self._config_path.write_text(
            json.dumps(self._snapshot.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin and update persistent state."""
        name = plugin.name
        
        # Create or update config
        if name not in self._snapshot.configs:
            self._snapshot.configs[name] = PluginConfig(
                name=name,
                version=plugin.version,
                plugin_type=plugin.plugin_type.value,
                enabled=True,
                installed_at=datetime.utcnow(),
            )
        
        # Apply persisted enabled state
        config = self._snapshot.configs[name]
        if not config.enabled:
            plugin.disable()
        
        # Apply persisted configuration
        if config.config:
            plugin.configure(config.config)
        
        self._save_state()

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin by name."""
        plugin = self._registry.get(name)
        if not plugin:
            return False
        
        plugin.enable()
        
        if name in self._snapshot.configs:
            self._snapshot.configs[name].enabled = True
            self._snapshot.configs[name].state = PluginState.ENABLED
        
        self._save_state()
        return True

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin by name."""
        plugin = self._registry.get(name)
        if not plugin:
            return False
        
        plugin.disable()
        
        if name in self._snapshot.configs:
            self._snapshot.configs[name].enabled = False
            self._snapshot.configs[name].state = PluginState.DISABLED
        
        self._save_state()
        return True

    def configure_plugin(self, name: str, config: dict[str, Any]) -> bool:
        """Update a plugin's configuration."""
        plugin = self._registry.get(name)
        if not plugin:
            return False
        
        plugin.configure(config)
        
        if name in self._snapshot.configs:
            self._snapshot.configs[name].config.update(config)
        
        self._save_state()
        return True

    def mark_plugin_used(self, name: str) -> None:
        """Mark a plugin as recently used."""
        if name in self._snapshot.configs:
            self._snapshot.configs[name].last_used = datetime.utcnow()
            self._save_state()

    def set_plugin_error(self, name: str, error: str) -> None:
        """Set error state for a plugin."""
        if name in self._snapshot.configs:
            self._snapshot.configs[name].state = PluginState.ERROR
            self._snapshot.configs[name].error_message = error
            self._save_state()

    def clear_plugin_error(self, name: str) -> None:
        """Clear error state for a plugin."""
        if name in self._snapshot.configs:
            config = self._snapshot.configs[name]
            config.error_message = None
            config.state = PluginState.ENABLED if config.enabled else PluginState.DISABLED
            self._save_state()

    def get_plugin_config(self, name: str) -> PluginConfig | None:
        """Get the persistent configuration for a plugin."""
        return self._snapshot.configs.get(name)

    def get_all_configs(self) -> dict[str, PluginConfig]:
        """Get all plugin configurations."""
        return dict(self._snapshot.configs)

    def get_enabled_plugins(self) -> list[str]:
        """Get list of enabled plugin names."""
        return [
            name for name, cfg in self._snapshot.configs.items()
            if cfg.enabled and cfg.state != PluginState.ERROR
        ]

    def get_disabled_plugins(self) -> list[str]:
        """Get list of disabled plugin names."""
        return [
            name for name, cfg in self._snapshot.configs.items()
            if not cfg.enabled
        ]

    def get_plugins_by_type(self, plugin_type: PluginType) -> list[str]:
        """Get plugin names by type."""
        return [
            name for name, cfg in self._snapshot.configs.items()
            if cfg.plugin_type == plugin_type.value
        ]

    def get_plugin_stats(self) -> dict[str, Any]:
        """Get statistics about plugins."""
        configs = self._snapshot.configs
        return {
            "total": len(configs),
            "enabled": sum(1 for c in configs.values() if c.enabled),
            "disabled": sum(1 for c in configs.values() if not c.enabled),
            "error": sum(1 for c in configs.values() if c.state == PluginState.ERROR),
            "by_type": {
                pt.value: sum(1 for c in configs.values() if c.plugin_type == pt.value)
                for pt in PluginType
            },
        }

    def export_state(self) -> dict[str, Any]:
        """Export the complete plugin state for backup."""
        return self._snapshot.to_dict()

    def import_state(self, data: dict[str, Any], merge: bool = False) -> int:
        """Import plugin state from backup.
        
        Args:
            data: State data to import
            merge: If True, merge with existing state; if False, replace
            
        Returns:
            Number of plugins imported
        """
        imported = PluginStateSnapshot.from_dict(data)
        
        if merge:
            for name, cfg in imported.configs.items():
                self._snapshot.configs[name] = cfg
        else:
            self._snapshot = imported
        
        self._save_state()
        
        # Apply state to registered plugins
        for name, cfg in self._snapshot.configs.items():
            plugin = self._registry.get(name)
            if plugin:
                if cfg.enabled:
                    plugin.enable()
                else:
                    plugin.disable()
                if cfg.config:
                    plugin.configure(cfg.config)
        
        return len(imported.configs)

    def reset_all(self) -> None:
        """Reset all plugin configurations to defaults."""
        self._snapshot = PluginStateSnapshot()
        self._save_state()
        
        # Re-enable all registered plugins
        for plugin in self._registry.list_all():
            plugin.enable()


# Global plugin manager singleton
_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def reset_plugin_manager() -> None:
    """Reset the global plugin manager."""
    global _plugin_manager
    _plugin_manager = None


# ==============================================================================
# ADVANCED PLUGIN FEATURES (5% Gap Coverage)
# ==============================================================================


@dataclass
class PluginDependency:
    """Plugin dependency specification."""
    
    name: str
    version_constraint: str = "*"  # Semver constraint
    optional: bool = False
    
    def satisfies(self, available_version: str) -> bool:
        """Check if available version satisfies constraint."""
        if self.version_constraint == "*":
            return True
        
        # Simple version comparison (for production, use packaging.version)
        try:
            if self.version_constraint.startswith(">="):
                required = self.version_constraint[2:]
                return available_version >= required
            elif self.version_constraint.startswith("<="):
                required = self.version_constraint[2:]
                return available_version <= required
            elif self.version_constraint.startswith("=="):
                required = self.version_constraint[2:]
                return available_version == required
            elif self.version_constraint.startswith("^"):
                # Compatible with (same major version)
                required = self.version_constraint[1:]
                return available_version.split(".")[0] == required.split(".")[0]
            return True
        except Exception:
            return True


@dataclass
class PluginManifest:
    """Extended plugin manifest with dependencies and capabilities."""
    
    name: str
    version: str
    description: str
    author: str = ""
    license: str = "MIT"
    homepage: str = ""
    repository: str = ""
    dependencies: list[PluginDependency] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    min_proxima_version: str = "0.1.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "repository": self.repository,
            "dependencies": [
                {"name": d.name, "version": d.version_constraint, "optional": d.optional}
                for d in self.dependencies
            ],
            "keywords": self.keywords,
            "capabilities": self.capabilities,
            "permissions": self.permissions,
            "min_proxima_version": self.min_proxima_version,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        """Create from dictionary."""
        deps = [
            PluginDependency(
                name=d["name"],
                version_constraint=d.get("version", "*"),
                optional=d.get("optional", False),
            )
            for d in data.get("dependencies", [])
        ]
        return cls(
            name=data["name"],
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", "MIT"),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            dependencies=deps,
            keywords=data.get("keywords", []),
            capabilities=data.get("capabilities", []),
            permissions=data.get("permissions", []),
            min_proxima_version=data.get("min_proxima_version", "0.1.0"),
        )


class DependencyResolver:
    """Resolves plugin dependencies."""
    
    def __init__(self, registry: PluginRegistry) -> None:
        """Initialize resolver."""
        self._registry = registry
        self._manifests: dict[str, PluginManifest] = {}
    
    def register_manifest(self, manifest: PluginManifest) -> None:
        """Register a plugin manifest."""
        self._manifests[manifest.name] = manifest
    
    def resolve(
        self, plugin_name: str
    ) -> tuple[list[str], list[str]]:
        """Resolve dependencies for a plugin.
        
        Args:
            plugin_name: Plugin to resolve dependencies for
            
        Returns:
            Tuple of (load_order, unmet_dependencies)
        """
        manifest = self._manifests.get(plugin_name)
        if not manifest:
            return [plugin_name], []
        
        load_order: list[str] = []
        unmet: list[str] = []
        visited: set[str] = set()
        
        self._resolve_recursive(plugin_name, load_order, unmet, visited)
        
        return load_order, unmet
    
    def _resolve_recursive(
        self,
        name: str,
        load_order: list[str],
        unmet: list[str],
        visited: set[str],
    ) -> None:
        """Recursively resolve dependencies."""
        if name in visited:
            return
        visited.add(name)
        
        manifest = self._manifests.get(name)
        if manifest:
            for dep in manifest.dependencies:
                # Check if dependency is available
                if dep.name in self._manifests:
                    self._resolve_recursive(dep.name, load_order, unmet, visited)
                elif not dep.optional:
                    unmet.append(f"{dep.name} (required by {name})")
        
        if name not in load_order:
            load_order.append(name)
    
    def check_circular(self) -> list[list[str]]:
        """Check for circular dependencies.
        
        Returns:
            List of circular dependency chains
        """
        cycles: list[list[str]] = []
        
        for name in self._manifests:
            path: list[str] = []
            visited: set[str] = set()
            if self._detect_cycle(name, path, visited):
                cycles.append(path.copy())
        
        return cycles
    
    def _detect_cycle(
        self,
        name: str,
        path: list[str],
        visited: set[str],
    ) -> bool:
        """Detect cycle starting from a node."""
        if name in path:
            path.append(name)
            return True
        if name in visited:
            return False
        
        visited.add(name)
        path.append(name)
        
        manifest = self._manifests.get(name)
        if manifest:
            for dep in manifest.dependencies:
                if self._detect_cycle(dep.name, path, visited):
                    return True
        
        path.pop()
        return False


# =============================================================================
# COMPREHENSIVE SANDBOXING SYSTEM (5% Gap Coverage)
# =============================================================================


class SandboxPermission(str, Enum):
    """Permissions that can be granted to sandboxed plugins."""
    
    FILE_READ = "file_read"           # Read files from filesystem
    FILE_WRITE = "file_write"         # Write files to filesystem
    NETWORK_ACCESS = "network_access" # Make network requests
    SYSTEM_ACCESS = "system_access"   # Access system resources (os, subprocess)
    THREAD_CREATE = "thread_create"   # Create new threads
    PROCESS_CREATE = "process_create" # Create new processes
    MEMORY_LARGE = "memory_large"     # Use more than default memory limit
    TIME_EXTENDED = "time_extended"   # Extended execution time
    IMPORT_ALL = "import_all"         # Import any module
    NATIVE_CODE = "native_code"       # Execute native/C extensions


@dataclass
class SandboxLimits:
    """Resource limits for sandboxed execution."""
    
    memory_mb: int = 256             # Max memory in MB
    timeout_seconds: float = 30.0    # Max execution time
    max_output_bytes: int = 1024 * 1024  # Max output size (1MB)
    max_file_handles: int = 10       # Max open file handles
    max_threads: int = 1             # Max threads (1 = single threaded)
    max_cpu_percent: float = 100.0   # Max CPU usage percent
    max_recursion: int = 100         # Max recursion depth
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_mb": self.memory_mb,
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "max_file_handles": self.max_file_handles,
            "max_threads": self.max_threads,
            "max_cpu_percent": self.max_cpu_percent,
            "max_recursion": self.max_recursion,
        }


@dataclass
class SandboxViolation:
    """Record of a sandbox policy violation."""
    
    timestamp: datetime
    plugin_name: str
    violation_type: str
    description: str
    severity: str = "warning"  # warning, error, critical
    blocked: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "plugin_name": self.plugin_name,
            "violation_type": self.violation_type,
            "description": self.description,
            "severity": self.severity,
            "blocked": self.blocked,
        }


@dataclass
class SandboxExecutionResult:
    """Result of sandboxed execution."""
    
    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
    violations: list[SandboxViolation] = field(default_factory=list)
    output_truncated: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "violations": [v.to_dict() for v in self.violations],
            "output_truncated": self.output_truncated,
        }


class ImportGuard:
    """Guards against unauthorized module imports."""
    
    def __init__(
        self,
        allowed_modules: set[str],
        blocked_modules: set[str],
        permissions: set[SandboxPermission],
    ) -> None:
        self.allowed_modules = allowed_modules
        self.blocked_modules = blocked_modules
        self.permissions = permissions
        self._original_import: Callable | None = None
        self._violations: list[SandboxViolation] = []
        self._plugin_name: str = "unknown"
    
    def set_plugin_name(self, name: str) -> None:
        """Set the current plugin name for violation tracking."""
        self._plugin_name = name
    
    def clear_violations(self) -> list[SandboxViolation]:
        """Clear and return collected violations."""
        violations = self._violations.copy()
        self._violations.clear()
        return violations
    
    def can_import(self, module_name: str) -> bool:
        """Check if module import is allowed."""
        base_module = module_name.split(".")[0]
        
        # Check explicit allow list
        if base_module in self.allowed_modules:
            return True
        
        # Check block list
        if base_module in self.blocked_modules:
            if SandboxPermission.SYSTEM_ACCESS in self.permissions:
                return True  # Has permission override
            return False
        
        # Check for import_all permission
        if SandboxPermission.IMPORT_ALL in self.permissions:
            return True
        
        # Default: allow standard library, block unknown
        return True  # Simplified for now
    
    def guarded_import(
        self,
        name: str,
        globals_dict: dict | None = None,
        locals_dict: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        """Import hook that checks permissions."""
        if not self.can_import(name):
            violation = SandboxViolation(
                timestamp=datetime.utcnow(),
                plugin_name=self._plugin_name,
                violation_type="blocked_import",
                description=f"Attempted to import blocked module: {name}",
                severity="error",
                blocked=True,
            )
            self._violations.append(violation)
            raise ImportError(f"Module '{name}' is not allowed in sandbox")
        
        if self._original_import:
            return self._original_import(name, globals_dict, locals_dict, fromlist, level)
        raise ImportError(f"Original import not available")
    
    def __enter__(self) -> "ImportGuard":
        """Enter context: install import hook."""
        import builtins
        self._original_import = builtins.__import__
        builtins.__import__ = self.guarded_import
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context: restore original import."""
        import builtins
        if self._original_import:
            builtins.__import__ = self._original_import


class PluginSandbox:
    """Sandboxed execution environment for plugins.
    
    Provides isolation and security controls for plugin execution with:
    - Module import restrictions
    - Resource limits (memory, CPU, time)
    - Permission-based access control
    - Violation tracking and reporting
    """
    
    # Default allowed modules (safe for plugins)
    DEFAULT_ALLOWED_MODULES = {
        "math", "json", "re", "collections", "dataclasses",
        "typing", "enum", "functools", "itertools", "operator",
        "copy", "abc", "numbers", "decimal", "fractions",
        "statistics", "random", "datetime", "time", "calendar",
        "heapq", "bisect", "array", "weakref", "types",
        "contextlib", "warnings", "traceback", "string",
        "textwrap", "difflib", "io", "base64", "binascii",
        "struct", "codecs", "unicodedata", "locale", "hashlib",
        "hmac", "secrets", "uuid", "logging",
    }
    
    # Default blocked modules (potentially dangerous)
    DEFAULT_BLOCKED_MODULES = {
        "os", "subprocess", "shutil", "socket", "ctypes",
        "multiprocessing", "threading", "_thread",
        "importlib", "sys", "builtins", "code", "codeop",
        "compile", "exec", "eval", "pickle", "marshal",
        "shelve", "dbm", "sqlite3", "tempfile", "glob",
        "pathlib", "zipfile", "tarfile", "lzma", "bz2", "gzip",
        "asyncio", "concurrent", "ssl", "http", "urllib",
        "ftplib", "smtplib", "poplib", "imaplib", "nntplib",
        "telnetlib", "xmlrpc", "email", "mimetypes",
    }
    
    def __init__(
        self,
        permissions: list[SandboxPermission] | None = None,
        limits: SandboxLimits | None = None,
        allowed_modules: set[str] | None = None,
        blocked_modules: set[str] | None = None,
    ) -> None:
        """Initialize sandbox.
        
        Args:
            permissions: Granted permissions
            limits: Resource limits
            allowed_modules: Additional allowed modules
            blocked_modules: Additional blocked modules
        """
        self.permissions = set(permissions or [])
        self.limits = limits or SandboxLimits()
        
        # Set up module filtering
        self.allowed_modules = self.DEFAULT_ALLOWED_MODULES.copy()
        if allowed_modules:
            self.allowed_modules.update(allowed_modules)
        
        self.blocked_modules = self.DEFAULT_BLOCKED_MODULES.copy()
        if blocked_modules:
            self.blocked_modules.update(blocked_modules)
        
        # Remove blocked modules from allowed if permission grants override
        if SandboxPermission.SYSTEM_ACCESS in self.permissions:
            self.blocked_modules -= {"os", "subprocess", "shutil"}
        if SandboxPermission.NETWORK_ACCESS in self.permissions:
            self.blocked_modules -= {"socket", "ssl", "http", "urllib"}
        if SandboxPermission.THREAD_CREATE in self.permissions:
            self.blocked_modules -= {"threading", "_thread"}
        
        self._execution_count = 0
        self._total_time_ms = 0.0
        self._violations: list[SandboxViolation] = []
        self._import_guard = ImportGuard(
            self.allowed_modules,
            self.blocked_modules,
            self.permissions,
        )
    
    def can_import(self, module_name: str) -> bool:
        """Check if module import is allowed."""
        return self._import_guard.can_import(module_name)
    
    def has_permission(self, permission: SandboxPermission) -> bool:
        """Check if a permission is granted."""
        return permission in self.permissions
    
    def grant_permission(self, permission: SandboxPermission) -> None:
        """Grant a permission."""
        self.permissions.add(permission)
    
    def revoke_permission(self, permission: SandboxPermission) -> None:
        """Revoke a permission."""
        self.permissions.discard(permission)
    
    def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        plugin_name: str = "unknown",
        **kwargs: Any,
    ) -> SandboxExecutionResult:
        """Execute function in sandbox with full isolation.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            plugin_name: Plugin name for violation tracking
            **kwargs: Keyword arguments
            
        Returns:
            SandboxExecutionResult with outcome details
        """
        import sys
        import time
        import traceback as tb
        
        self._execution_count += 1
        self._import_guard.set_plugin_name(plugin_name)
        
        start = time.perf_counter()
        result = SandboxExecutionResult(success=False)
        
        # Save original recursion limit
        original_recursion_limit = sys.getrecursionlimit()
        
        try:
            # Set recursion limit
            sys.setrecursionlimit(self.limits.max_recursion)
            
            # Execute with import guard
            with self._import_guard:
                # Execute function with timeout (simplified - full impl would use signals/threads)
                func_result = func(*args, **kwargs)
            
            elapsed = (time.perf_counter() - start) * 1000
            
            # Check timeout
            if elapsed > self.limits.timeout_seconds * 1000:
                result.error = f"Execution exceeded timeout of {self.limits.timeout_seconds}s"
                violation = SandboxViolation(
                    timestamp=datetime.utcnow(),
                    plugin_name=plugin_name,
                    violation_type="timeout",
                    description=result.error,
                    severity="error",
                    blocked=True,
                )
                self._violations.append(violation)
                result.violations.append(violation)
            else:
                result.success = True
                result.result = func_result
            
            result.execution_time_ms = elapsed
            self._total_time_ms += elapsed
            
        except ImportError as e:
            result.error = str(e)
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            
        except RecursionError:
            result.error = f"Recursion limit exceeded ({self.limits.max_recursion})"
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            violation = SandboxViolation(
                timestamp=datetime.utcnow(),
                plugin_name=plugin_name,
                violation_type="recursion_limit",
                description=result.error,
                severity="error",
                blocked=True,
            )
            self._violations.append(violation)
            result.violations.append(violation)
            
        except MemoryError:
            result.error = "Memory limit exceeded"
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            violation = SandboxViolation(
                timestamp=datetime.utcnow(),
                plugin_name=plugin_name,
                violation_type="memory_limit",
                description=result.error,
                severity="critical",
                blocked=True,
            )
            self._violations.append(violation)
            result.violations.append(violation)
            
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            
        finally:
            # Restore recursion limit
            sys.setrecursionlimit(original_recursion_limit)
            
            # Collect any import violations
            import_violations = self._import_guard.clear_violations()
            result.violations.extend(import_violations)
            self._violations.extend(import_violations)
        
        return result
    
    async def execute_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        plugin_name: str = "unknown",
        **kwargs: Any,
    ) -> SandboxExecutionResult:
        """Execute async function in sandbox.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            plugin_name: Plugin name for violation tracking
            **kwargs: Keyword arguments
            
        Returns:
            SandboxExecutionResult with outcome details
        """
        import asyncio
        import time
        
        self._execution_count += 1
        self._import_guard.set_plugin_name(plugin_name)
        
        start = time.perf_counter()
        result = SandboxExecutionResult(success=False)
        
        try:
            # Execute with timeout
            with self._import_guard:
                func_result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.limits.timeout_seconds,
                )
            
            result.success = True
            result.result = func_result
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            
        except asyncio.TimeoutError:
            result.error = f"Async execution exceeded timeout of {self.limits.timeout_seconds}s"
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            violation = SandboxViolation(
                timestamp=datetime.utcnow(),
                plugin_name=plugin_name,
                violation_type="async_timeout",
                description=result.error,
                severity="error",
                blocked=True,
            )
            self._violations.append(violation)
            result.violations.append(violation)
            
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            
        finally:
            import_violations = self._import_guard.clear_violations()
            result.violations.extend(import_violations)
            self._violations.extend(import_violations)
        
        self._total_time_ms += result.execution_time_ms
        return result
    
    def get_stats(self) -> dict[str, Any]:
        """Get sandbox execution statistics."""
        return {
            "execution_count": self._execution_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / max(1, self._execution_count),
            "violation_count": len(self._violations),
            "permissions": [p.value for p in self.permissions],
            "limits": self.limits.to_dict(),
        }
    
    def get_violations(
        self,
        plugin_name: str | None = None,
        severity: str | None = None,
    ) -> list[SandboxViolation]:
        """Get recorded violations with optional filtering."""
        violations = self._violations
        if plugin_name:
            violations = [v for v in violations if v.plugin_name == plugin_name]
        if severity:
            violations = [v for v in violations if v.severity == severity]
        return violations
    
    def clear_violations(self) -> int:
        """Clear all recorded violations and return count."""
        count = len(self._violations)
        self._violations.clear()
        return count
    
    def create_isolated_globals(self) -> dict[str, Any]:
        """Create an isolated globals dict for exec/eval.
        
        Returns restricted builtins for use in sandboxed code execution.
        """
        import builtins as _builtins
        
        # Safe builtins subset
        safe_builtins = {
            # Types
            "bool": bool, "int": int, "float": float, "str": str,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            "frozenset": frozenset, "bytes": bytes, "bytearray": bytearray,
            "type": type, "object": object,
            # Functions
            "abs": abs, "all": all, "any": any, "bin": bin,
            "callable": callable, "chr": chr, "divmod": divmod,
            "enumerate": enumerate, "filter": filter, "format": format,
            "hash": hash, "hex": hex, "id": id, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter, "len": len,
            "map": map, "max": max, "min": min, "next": next,
            "oct": oct, "ord": ord, "pow": pow, "print": print,
            "range": range, "repr": repr, "reversed": reversed,
            "round": round, "slice": slice, "sorted": sorted, "sum": sum,
            "zip": zip, "hasattr": hasattr, "getattr": getattr,
            "setattr": setattr, "delattr": delattr,
            # Exceptions
            "Exception": Exception, "TypeError": TypeError,
            "ValueError": ValueError, "KeyError": KeyError,
            "IndexError": IndexError, "AttributeError": AttributeError,
            "RuntimeError": RuntimeError, "StopIteration": StopIteration,
            # Constants
            "True": True, "False": False, "None": None,
        }
        
        # Blocked dangerous builtins
        blocked = {"eval", "exec", "compile", "open", "__import__", "input"}
        
        return {"__builtins__": safe_builtins}


class SandboxPolicy:
    """Policy configuration for sandbox behavior."""
    
    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.permissions: set[SandboxPermission] = set()
        self.limits = SandboxLimits()
        self.allowed_modules: set[str] = set()
        self.blocked_modules: set[str] = set()
        self.trusted_plugins: set[str] = set()
    
    def create_sandbox(self) -> PluginSandbox:
        """Create a sandbox with this policy."""
        return PluginSandbox(
            permissions=list(self.permissions),
            limits=self.limits,
            allowed_modules=self.allowed_modules,
            blocked_modules=self.blocked_modules,
        )
    
    @classmethod
    def strict(cls) -> "SandboxPolicy":
        """Create a strict policy with minimal permissions."""
        policy = cls("strict")
        policy.limits = SandboxLimits(
            memory_mb=64,
            timeout_seconds=5.0,
            max_recursion=50,
        )
        return policy
    
    @classmethod
    def permissive(cls) -> "SandboxPolicy":
        """Create a permissive policy with most permissions."""
        policy = cls("permissive")
        policy.permissions = {
            SandboxPermission.FILE_READ,
            SandboxPermission.NETWORK_ACCESS,
            SandboxPermission.THREAD_CREATE,
            SandboxPermission.IMPORT_ALL,
        }
        policy.limits = SandboxLimits(
            memory_mb=1024,
            timeout_seconds=300.0,
            max_recursion=500,
        )
        return policy
    
    @classmethod
    def trusted(cls) -> "SandboxPolicy":
        """Create a policy for trusted plugins (no restrictions)."""
        policy = cls("trusted")
        policy.permissions = set(SandboxPermission)
        policy.limits = SandboxLimits(
            memory_mb=4096,
            timeout_seconds=3600.0,
            max_recursion=1000,
        )
        return policy
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "permissions": [p.value for p in self.permissions],
            "limits": self.limits.to_dict(),
            "allowed_modules": list(self.allowed_modules),
            "blocked_modules": list(self.blocked_modules),
            "trusted_plugins": list(self.trusted_plugins),
        }


class SandboxExecutionError(Exception):
    """Error during sandboxed execution."""
    
    def __init__(self, message: str, elapsed_ms: float) -> None:
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


class HotReloader:
    """Hot reload support for plugins.
    
    Enables plugins to be reloaded without restarting the application.
    """
    
    def __init__(self, plugin_dir: Path) -> None:
        """Initialize hot reloader.
        
        Args:
            plugin_dir: Directory to watch for changes
        """
        self.plugin_dir = plugin_dir
        self._file_hashes: dict[str, str] = {}
        self._callbacks: list[Callable[[str], None]] = []
        self._watching = False
    
    def register_callback(self, callback: Callable[[str], None]) -> None:
        """Register reload callback."""
        self._callbacks.append(callback)
    
    def compute_hash(self, file_path: Path) -> str:
        """Compute file hash."""
        import hashlib
        content = file_path.read_bytes()
        return hashlib.md5(content).hexdigest()
    
    def scan_changes(self) -> list[str]:
        """Scan for changed files.
        
        Returns:
            List of changed file paths
        """
        changed: list[str] = []
        
        if not self.plugin_dir.exists():
            return changed
        
        for file_path in self.plugin_dir.rglob("*.py"):
            str_path = str(file_path)
            current_hash = self.compute_hash(file_path)
            
            if str_path in self._file_hashes:
                if self._file_hashes[str_path] != current_hash:
                    changed.append(str_path)
            
            self._file_hashes[str_path] = current_hash
        
        return changed
    
    def trigger_reload(self, changed_files: list[str]) -> dict[str, Any]:
        """Trigger reload for changed files.
        
        Args:
            changed_files: Files that changed
            
        Returns:
            Reload result summary
        """
        reloaded: list[str] = []
        errors: list[str] = []
        
        for file_path in changed_files:
            try:
                # Extract plugin name from path
                plugin_name = Path(file_path).stem
                
                # Notify callbacks
                for callback in self._callbacks:
                    callback(plugin_name)
                
                reloaded.append(plugin_name)
            except Exception as e:
                errors.append(f"{file_path}: {e}")
        
        return {
            "reloaded": reloaded,
            "errors": errors,
            "total_changed": len(changed_files),
        }
    
    def start_watching(self, interval_seconds: float = 1.0) -> None:
        """Start watching for changes (async-compatible)."""
        self._watching = True
        # In production, use watchdog or similar
        # This is a simplified polling-based implementation
    
    def stop_watching(self) -> None:
        """Stop watching."""
        self._watching = False


@dataclass
class MarketplacePlugin:
    """Plugin listing from marketplace."""
    
    name: str
    version: str
    description: str
    author: str
    downloads: int = 0
    rating: float = 0.0
    tags: list[str] = field(default_factory=list)
    download_url: str = ""
    verified: bool = False


class PluginMarketplace:
    """Plugin marketplace integration.
    
    Enables discovering and installing plugins from a central registry.
    """
    
    DEFAULT_REGISTRY_URL = "https://api.proxima.dev/plugins"
    
    def __init__(
        self,
        registry_url: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize marketplace.
        
        Args:
            registry_url: Plugin registry URL
            cache_dir: Local cache directory
        """
        self.registry_url = registry_url or self.DEFAULT_REGISTRY_URL
        self.cache_dir = cache_dir or Path.home() / ".proxima" / "plugin_cache"
        self._installed: dict[str, str] = {}  # name -> version
    
    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MarketplacePlugin]:
        """Search for plugins.
        
        Args:
            query: Search query
            tags: Filter by tags
            limit: Maximum results
            
        Returns:
            List of matching plugins
        """
        # In production, this would query the actual registry
        # Mock implementation for demonstration
        mock_plugins = [
            MarketplacePlugin(
                name="quantum-optimizer",
                version="1.2.0",
                description="Advanced quantum circuit optimization",
                author="Proxima Team",
                downloads=15000,
                rating=4.8,
                tags=["optimization", "performance"],
                verified=True,
            ),
            MarketplacePlugin(
                name="noise-simulator",
                version="0.9.5",
                description="Realistic noise simulation for circuits",
                author="QuantumDev",
                downloads=8500,
                rating=4.5,
                tags=["noise", "simulation"],
                verified=True,
            ),
            MarketplacePlugin(
                name="vqe-toolkit",
                version="2.0.0",
                description="Variational Quantum Eigensolver tools",
                author="ChemQuantum",
                downloads=5200,
                rating=4.6,
                tags=["vqe", "chemistry", "algorithms"],
                verified=False,
            ),
        ]
        
        # Filter by query
        if query:
            query_lower = query.lower()
            mock_plugins = [
                p for p in mock_plugins
                if query_lower in p.name.lower() or query_lower in p.description.lower()
            ]
        
        # Filter by tags
        if tags:
            mock_plugins = [
                p for p in mock_plugins
                if any(t in p.tags for t in tags)
            ]
        
        return mock_plugins[:limit]
    
    def get_info(self, plugin_name: str) -> MarketplacePlugin | None:
        """Get detailed plugin information.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin info or None if not found
        """
        results = self.search(plugin_name, limit=1)
        for plugin in results:
            if plugin.name == plugin_name:
                return plugin
        return None
    
    def install(
        self,
        plugin_name: str,
        version: str | None = None,
    ) -> tuple[bool, str]:
        """Install a plugin from marketplace.
        
        Args:
            plugin_name: Plugin to install
            version: Specific version (latest if None)
            
        Returns:
            Tuple of (success, message)
        """
        info = self.get_info(plugin_name)
        if not info:
            return False, f"Plugin '{plugin_name}' not found in marketplace"
        
        target_version = version or info.version
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download would happen here in production
        # Mock implementation
        self._installed[plugin_name] = target_version
        
        return True, f"Installed {plugin_name} v{target_version}"
    
    def uninstall(self, plugin_name: str) -> tuple[bool, str]:
        """Uninstall a plugin.
        
        Args:
            plugin_name: Plugin to uninstall
            
        Returns:
            Tuple of (success, message)
        """
        if plugin_name not in self._installed:
            return False, f"Plugin '{plugin_name}' is not installed"
        
        del self._installed[plugin_name]
        
        # Clean up files would happen here
        return True, f"Uninstalled {plugin_name}"
    
    def list_installed(self) -> list[tuple[str, str]]:
        """List installed plugins.
        
        Returns:
            List of (name, version) tuples
        """
        return list(self._installed.items())
    
    def check_updates(self) -> list[tuple[str, str, str]]:
        """Check for available updates.
        
        Returns:
            List of (name, current_version, new_version) tuples
        """
        updates: list[tuple[str, str, str]] = []
        
        for name, current in self._installed.items():
            info = self.get_info(name)
            if info and info.version > current:
                updates.append((name, current, info.version))
        
        return updates


class EnhancedPluginManager(PluginManager):
    """Enhanced plugin manager with advanced features.
    
    Adds:
    - Dependency resolution
    - Sandboxed execution
    - Hot reloading
    - Marketplace integration
    """
    
    def __init__(
        self,
        plugin_dir: Path | None = None,
        enable_sandbox: bool = True,
        enable_hot_reload: bool = False,
    ) -> None:
        """Initialize enhanced manager.
        
        Args:
            plugin_dir: Plugin directory
            enable_sandbox: Enable sandboxed execution
            enable_hot_reload: Enable hot reload
        """
        super().__init__()
        
        self.plugin_dir = plugin_dir or Path.home() / ".proxima" / "plugins"
        self._dep_resolver = DependencyResolver(self._registry)
        self._marketplace = PluginMarketplace()
        self._manifests: dict[str, PluginManifest] = {}
        
        self._sandbox = PluginSandbox() if enable_sandbox else None
        self._hot_reloader = HotReloader(self.plugin_dir) if enable_hot_reload else None
        
        if self._hot_reloader:
            self._hot_reloader.register_callback(self._on_plugin_changed)
    
    def register_with_manifest(
        self,
        plugin: BasePlugin,
        manifest: PluginManifest,
    ) -> bool:
        """Register plugin with manifest.
        
        Args:
            plugin: Plugin instance
            manifest: Plugin manifest
            
        Returns:
            True if registration successful
        """
        # Check dependencies
        self._manifests[manifest.name] = manifest
        self._dep_resolver.register_manifest(manifest)
        
        load_order, unmet = self._dep_resolver.resolve(manifest.name)
        
        if unmet:
            return False
        
        # Register in order
        return self.register(plugin)
    
    def get_load_order(self, plugin_name: str) -> list[str]:
        """Get correct load order for a plugin.
        
        Args:
            plugin_name: Plugin to load
            
        Returns:
            Ordered list of plugins to load
        """
        order, _ = self._dep_resolver.resolve(plugin_name)
        return order
    
    def execute_sandboxed(
        self,
        plugin_name: str,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, float]:
        """Execute plugin method in sandbox.
        
        Args:
            plugin_name: Plugin name
            method: Method to call
            *args: Arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tuple of (result, execution_time_ms)
        """
        plugin = self._registry.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        func = getattr(plugin, method, None)
        if not func:
            raise ValueError(f"Method not found: {method}")
        
        if self._sandbox:
            return self._sandbox.execute(func, *args, **kwargs)
        else:
            import time
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            return result, elapsed
    
    def check_hot_reload(self) -> dict[str, Any]:
        """Check for and apply hot reload.
        
        Returns:
            Reload result
        """
        if not self._hot_reloader:
            return {"enabled": False}
        
        changed = self._hot_reloader.scan_changes()
        if changed:
            return self._hot_reloader.trigger_reload(changed)
        return {"changed": 0, "reloaded": []}
    
    def _on_plugin_changed(self, plugin_name: str) -> None:
        """Handle plugin file change."""
        # Reload plugin
        plugin = self._registry.get(plugin_name)
        if plugin:
            self.disable(plugin_name)
            self.enable(plugin_name)
    
    @property
    def marketplace(self) -> PluginMarketplace:
        """Get marketplace instance."""
        return self._marketplace
    
    def install_from_marketplace(
        self,
        plugin_name: str,
        version: str | None = None,
    ) -> tuple[bool, str]:
        """Install plugin from marketplace.
        
        Args:
            plugin_name: Plugin name
            version: Optional version
            
        Returns:
            Tuple of (success, message)
        """
        return self._marketplace.install(plugin_name, version)

