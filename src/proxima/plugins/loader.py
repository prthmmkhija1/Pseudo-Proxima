"""
Plugin loader and registry.

Handles discovering, loading, and managing plugins from:
- Entry points (installed packages)
- Plugin directories
- Explicit registration
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proxima.plugins.base import (
    Plugin,
    PluginError,
    PluginLoadError,
    PluginType,
    PluginValidationError,
)

# Entry point group names
ENTRY_POINT_GROUPS = {
    PluginType.BACKEND: "proxima.backends",
    PluginType.LLM_PROVIDER: "proxima.llm_providers",
    PluginType.EXPORTER: "proxima.exporters",
    PluginType.ANALYZER: "proxima.analyzers",
    PluginType.HOOK: "proxima.hooks",
}


class PluginRegistry:
    """Central registry for all loaded plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._by_type: dict[PluginType, dict[str, Plugin]] = {
            pt: {} for pt in PluginType
        }
        self._load_callbacks: list[Callable[[Plugin], None]] = []
        self._unload_callbacks: list[Callable[[Plugin], None]] = []

    def register(self, plugin: Plugin) -> None:
        """Register a plugin instance."""
        name = plugin.name
        ptype = plugin.plugin_type

        # Validate before registering
        errors = plugin.validate()
        if errors:
            raise PluginValidationError(
                f"Plugin '{name}' validation failed: {'; '.join(errors)}"
            )

        if name in self._plugins:
            raise PluginError(f"Plugin '{name}' is already registered")

        self._plugins[name] = plugin
        self._by_type[ptype][name] = plugin

        # Initialize the plugin
        try:
            plugin.initialize()
        except Exception as e:
            # Rollback registration on init failure
            del self._plugins[name]
            del self._by_type[ptype][name]
            raise PluginLoadError(f"Plugin '{name}' initialization failed: {e}") from e

        # Notify callbacks
        for callback in self._load_callbacks:
            try:
                callback(plugin)
            except Exception:
                pass  # Don't fail on callback errors

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name."""
        if name not in self._plugins:
            return False

        plugin = self._plugins[name]

        # Notify callbacks before shutdown
        for callback in self._unload_callbacks:
            try:
                callback(plugin)
            except Exception:
                pass

        # Shutdown the plugin
        try:
            plugin.shutdown()
        except Exception:
            pass  # Best effort shutdown

        del self._plugins[name]
        del self._by_type[plugin.plugin_type][name]
        return True

    def get(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_by_type(self, plugin_type: PluginType) -> dict[str, Plugin]:
        """Get all plugins of a specific type."""
        return dict(self._by_type[plugin_type])

    def list_all(self) -> list[Plugin]:
        """List all registered plugins."""
        return list(self._plugins.values())

    def list_names(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins

    def on_load(self, callback: Callable[[Plugin], None]) -> None:
        """Register a callback for plugin load events."""
        self._load_callbacks.append(callback)

    def on_unload(self, callback: Callable[[Plugin], None]) -> None:
        """Register a callback for plugin unload events."""
        self._unload_callbacks.append(callback)

    def clear(self) -> None:
        """Unregister all plugins."""
        for name in list(self._plugins.keys()):
            self.unregister(name)


class PluginLoader:
    """Loads plugins from various sources."""

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry
        self._plugin_dirs: list[Path] = []

    def add_plugin_dir(self, path: Path) -> None:
        """Add a directory to search for plugins."""
        if path.is_dir():
            self._plugin_dirs.append(path)

    def load_from_entry_points(self, plugin_type: PluginType | None = None) -> int:
        """Load plugins from installed package entry points.

        Returns the number of plugins loaded.
        """
        loaded = 0
        groups = (
            [ENTRY_POINT_GROUPS[plugin_type]]
            if plugin_type
            else list(ENTRY_POINT_GROUPS.values())
        )

        for group in groups:
            try:
                eps = importlib.metadata.entry_points(group=group)
            except TypeError:
                # Python 3.9 compatibility
                all_eps = importlib.metadata.entry_points()
                eps = all_eps.get(group, [])

            for ep in eps:
                try:
                    plugin_class = ep.load()
                    if isinstance(plugin_class, type) and issubclass(
                        plugin_class, Plugin
                    ):
                        plugin = plugin_class()
                        self.registry.register(plugin)
                        loaded += 1
                except Exception as e:
                    # Log but don't fail on individual plugin load errors
                    print(f"Warning: Failed to load plugin '{ep.name}': {e}")

        return loaded

    def load_from_directory(self, directory: Path | None = None) -> int:
        """Load plugins from a directory.

        Looks for Python files or packages with a `Plugin` class.
        Returns the number of plugins loaded.
        """
        loaded = 0
        dirs = [directory] if directory else self._plugin_dirs

        for plugin_dir in dirs:
            if not plugin_dir or not plugin_dir.is_dir():
                continue

            for path in plugin_dir.iterdir():
                if path.suffix == ".py" and not path.name.startswith("_"):
                    try:
                        plugin = self._load_from_file(path)
                        if plugin:
                            self.registry.register(plugin)
                            loaded += 1
                    except Exception as e:
                        print(f"Warning: Failed to load plugin from '{path}': {e}")

                elif path.is_dir() and (path / "__init__.py").exists():
                    try:
                        plugin = self._load_from_package(path)
                        if plugin:
                            self.registry.register(plugin)
                            loaded += 1
                    except Exception as e:
                        print(f"Warning: Failed to load plugin from '{path}': {e}")

        return loaded

    def load_from_class(
        self, plugin_class: type[Plugin], config: dict | None = None
    ) -> Plugin:
        """Load a plugin from a class directly."""
        plugin = plugin_class(config)
        self.registry.register(plugin)
        return plugin

    def _load_from_file(self, path: Path) -> Plugin | None:
        """Load a plugin from a Python file."""
        module_name = f"proxima_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return self._find_plugin_in_module(module)

    def _load_from_package(self, path: Path) -> Plugin | None:
        """Load a plugin from a package directory."""
        module_name = f"proxima_plugin_{path.name}"
        spec = importlib.util.spec_from_file_location(
            module_name, path / "__init__.py", submodule_search_locations=[str(path)]
        )
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return self._find_plugin_in_module(module)

    def _find_plugin_in_module(self, module: Any) -> Plugin | None:
        """Find and instantiate a Plugin class in a module."""
        # Look for a class named 'Plugin' or any subclass of Plugin
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, Plugin)
                and obj is not Plugin
                and hasattr(obj, "METADATA")
            ):
                return obj()
        return None


# Global registry singleton
_registry: PluginRegistry | None = None
_loader: PluginLoader | None = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def get_plugin_loader() -> PluginLoader:
    """Get the global plugin loader."""
    global _loader
    if _loader is None:
        _loader = PluginLoader(get_plugin_registry())
    return _loader


def discover_plugins(
    plugin_type: PluginType | None = None,
    include_entry_points: bool = True,
    plugin_dirs: list[Path] | None = None,
) -> int:
    """Discover and load plugins from all sources.

    Returns the total number of plugins loaded.
    """
    loader = get_plugin_loader()
    loaded = 0

    if include_entry_points:
        loaded += loader.load_from_entry_points(plugin_type)

    if plugin_dirs:
        for pdir in plugin_dirs:
            loader.add_plugin_dir(pdir)
        loaded += loader.load_from_directory()

    return loaded


def load_plugin(plugin_class: type[Plugin], config: dict | None = None) -> Plugin:
    """Load a single plugin from a class."""
    return get_plugin_loader().load_from_class(plugin_class, config)

# =============================================================================
# DEPENDENCY RESOLUTION SYSTEM (5% Gap Coverage)
# =============================================================================


class DependencyError(PluginError):
    """Error related to plugin dependencies."""
    pass


class CircularDependencyError(DependencyError):
    """Circular dependency detected."""
    pass


class UnmetDependencyError(DependencyError):
    """Required dependency not available."""
    pass


class VersionConflictError(DependencyError):
    """Version constraint conflict detected."""
    pass


@dataclass
class PluginDependency:
    """Specification for a plugin dependency.
    
    Attributes:
        name: Name of the required plugin
        version_constraint: SemVer constraint string (e.g., ">=1.0.0", "^2.0.0")
        optional: If True, plugin can load without this dependency
        features: Optional list of required features from the dependency
    """
    
    name: str
    version_constraint: str = "*"  # Any version
    optional: bool = False
    features: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.version_constraint == "*":
            return self.name
        return f"{self.name}{self.version_constraint}"
    
    def satisfies_version(self, version: str) -> bool:
        """Check if given version satisfies the constraint."""
        if self.version_constraint == "*":
            return True
        
        try:
            # Parse version and constraint
            constraint = self.version_constraint.strip()
            
            # Handle different constraint types
            if constraint.startswith(">="):
                required = constraint[2:].strip()
                return self._version_compare(version, required) >= 0
            elif constraint.startswith("<="):
                required = constraint[2:].strip()
                return self._version_compare(version, required) <= 0
            elif constraint.startswith(">"):
                required = constraint[1:].strip()
                return self._version_compare(version, required) > 0
            elif constraint.startswith("<"):
                required = constraint[1:].strip()
                return self._version_compare(version, required) < 0
            elif constraint.startswith("=="):
                required = constraint[2:].strip()
                return version == required
            elif constraint.startswith("^"):
                # Caret: compatible with (same major version)
                required = constraint[1:].strip()
                v_parts = version.split(".")
                r_parts = required.split(".")
                if len(v_parts) >= 1 and len(r_parts) >= 1:
                    return (
                        v_parts[0] == r_parts[0]
                        and self._version_compare(version, required) >= 0
                    )
            elif constraint.startswith("~"):
                # Tilde: approximately (same major.minor)
                required = constraint[1:].strip()
                v_parts = version.split(".")
                r_parts = required.split(".")
                if len(v_parts) >= 2 and len(r_parts) >= 2:
                    return (
                        v_parts[0] == r_parts[0]
                        and v_parts[1] == r_parts[1]
                        and self._version_compare(version, required) >= 0
                    )
            else:
                # Exact match
                return version == constraint
        except Exception:
            return True  # On parse error, assume compatible
        
        return True
    
    def _version_compare(self, v1: str, v2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        def parse_version(v: str) -> tuple:
            # Remove prerelease/build metadata for comparison
            v = v.split("-")[0].split("+")[0]
            parts = []
            for p in v.split("."):
                try:
                    parts.append(int(p))
                except ValueError:
                    parts.append(0)
            return tuple(parts)
        
        p1, p2 = parse_version(v1), parse_version(v2)
        
        # Pad shorter version with zeros
        max_len = max(len(p1), len(p2))
        p1 = p1 + (0,) * (max_len - len(p1))
        p2 = p2 + (0,) * (max_len - len(p2))
        
        if p1 < p2:
            return -1
        elif p1 > p2:
            return 1
        return 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version_constraint": self.version_constraint,
            "optional": self.optional,
            "features": self.features,
        }
    
    @classmethod
    def from_string(cls, dep_str: str) -> "PluginDependency":
        """Parse a dependency string like 'plugin-name>=1.0.0'."""
        import re
        
        # Match patterns like "name>=1.0.0" or "name^2.0.0" or just "name"
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_-]*)((?:>=|<=|>|<|==|\^|~)[\d.]+)?$', dep_str.strip())
        if match:
            name = match.group(1)
            constraint = match.group(2) or "*"
            return cls(name=name, version_constraint=constraint)
        
        return cls(name=dep_str.strip())


@dataclass
class DependencyNode:
    """Node in the dependency graph."""
    
    name: str
    version: str
    dependencies: list[PluginDependency] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)  # Plugins that depend on this
    resolved: bool = False
    error: str | None = None


@dataclass
class ResolutionResult:
    """Result of dependency resolution."""
    
    success: bool
    load_order: list[str] = field(default_factory=list)
    unmet_dependencies: list[tuple[str, PluginDependency]] = field(default_factory=list)
    circular_dependencies: list[list[str]] = field(default_factory=list)
    version_conflicts: list[tuple[str, str, str]] = field(default_factory=list)  # (plugin, required, available)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "load_order": self.load_order,
            "unmet_dependencies": [
                {"plugin": p, "requires": d.to_dict()}
                for p, d in self.unmet_dependencies
            ],
            "circular_dependencies": self.circular_dependencies,
            "version_conflicts": [
                {"plugin": p, "required": r, "available": a}
                for p, r, a in self.version_conflicts
            ],
            "warnings": self.warnings,
        }


class DependencyGraph:
    """Graph representation of plugin dependencies."""
    
    def __init__(self) -> None:
        self._nodes: dict[str, DependencyNode] = {}
        self._edges: dict[str, set[str]] = {}  # from -> set of to
    
    def add_node(
        self,
        name: str,
        version: str,
        dependencies: list[PluginDependency] | None = None,
    ) -> DependencyNode:
        """Add a node to the graph."""
        node = DependencyNode(
            name=name,
            version=version,
            dependencies=dependencies or [],
        )
        self._nodes[name] = node
        self._edges[name] = set()
        
        # Add edges for dependencies
        for dep in node.dependencies:
            self._edges[name].add(dep.name)
            # Update dependent list if target exists
            if dep.name in self._nodes:
                self._nodes[dep.name].dependents.append(name)
        
        return node
    
    def has_node(self, name: str) -> bool:
        """Check if a node exists."""
        return name in self._nodes
    
    def get_node(self, name: str) -> DependencyNode | None:
        """Get a node by name."""
        return self._nodes.get(name)
    
    def get_dependencies(self, name: str) -> list[str]:
        """Get direct dependencies of a node."""
        return list(self._edges.get(name, set()))
    
    def get_dependents(self, name: str) -> list[str]:
        """Get nodes that depend on the given node."""
        node = self._nodes.get(name)
        return node.dependents if node else []
    
    def get_all_dependencies(self, name: str, visited: set[str] | None = None) -> set[str]:
        """Get all transitive dependencies of a node."""
        if visited is None:
            visited = set()
        
        if name in visited:
            return visited
        visited.add(name)
        
        for dep in self._edges.get(name, set()):
            if dep in self._nodes:
                self.get_all_dependencies(dep, visited)
        
        visited.discard(name)  # Don't include self
        return visited
    
    def detect_cycles(self) -> list[list[str]]:
        """Detect all cycles in the graph."""
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: list[str] = []
        
        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.append(node)
            
            for neighbor in self._edges.get(node, set()):
                if neighbor not in visited:
                    if neighbor in self._nodes:
                        dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = rec_stack.index(neighbor)
                    cycle = rec_stack[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            rec_stack.pop()
        
        for node in self._nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def topological_sort(self) -> tuple[list[str], bool]:
        """Perform topological sort on the graph.
        
        Returns:
            Tuple of (sorted_list, success). If cycles exist, success is False.
        """
        in_degree: dict[str, int] = {node: 0 for node in self._nodes}
        
        # Calculate in-degrees
        for node in self._nodes:
            for dep in self._edges.get(node, set()):
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Find nodes with no dependencies
        queue: list[str] = [node for node, degree in in_degree.items() if degree == 0]
        result: list[str] = []
        
        while queue:
            # Sort by name for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            
            # Reduce in-degree of dependents
            for dependent in self._nodes.get(node, DependencyNode("", "")).dependents:
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        success = len(result) == len(self._nodes)
        return result, success


class DependencyResolver:
    """Resolves plugin dependencies with version constraint checking.
    
    Features:
    - Topological sorting for load order
    - Circular dependency detection
    - Version constraint validation
    - Optional dependency handling
    - Conflict detection
    """
    
    def __init__(self, registry: PluginRegistry | None = None) -> None:
        """Initialize resolver.
        
        Args:
            registry: Plugin registry for checking available plugins
        """
        self._registry = registry
        self._graph = DependencyGraph()
        self._plugin_versions: dict[str, str] = {}  # name -> version
        self._plugin_deps: dict[str, list[PluginDependency]] = {}  # name -> dependencies
    
    def register_plugin(
        self,
        name: str,
        version: str,
        dependencies: list[PluginDependency] | None = None,
    ) -> None:
        """Register a plugin and its dependencies for resolution."""
        self._plugin_versions[name] = version
        self._plugin_deps[name] = dependencies or []
        self._graph.add_node(name, version, dependencies)
    
    def register_from_metadata(self, plugin: Plugin) -> None:
        """Register a plugin from its metadata."""
        metadata = plugin.METADATA
        dependencies = [
            PluginDependency.from_string(req) for req in metadata.requires
        ]
        self.register_plugin(metadata.name, metadata.version, dependencies)
    
    def resolve(self, plugin_names: list[str] | None = None) -> ResolutionResult:
        """Resolve dependencies for the given plugins.
        
        Args:
            plugin_names: Plugins to resolve. If None, resolves all registered.
            
        Returns:
            ResolutionResult with load order and any issues.
        """
        result = ResolutionResult(success=True)
        
        # Use all registered plugins if none specified
        targets = plugin_names or list(self._plugin_versions.keys())
        
        # Check for circular dependencies
        cycles = self._graph.detect_cycles()
        if cycles:
            result.circular_dependencies = cycles
            result.success = False
            result.warnings.append(f"Found {len(cycles)} circular dependency chain(s)")
        
        # Check version constraints
        for name in targets:
            deps = self._plugin_deps.get(name, [])
            for dep in deps:
                if dep.name in self._plugin_versions:
                    available_version = self._plugin_versions[dep.name]
                    if not dep.satisfies_version(available_version):
                        result.version_conflicts.append(
                            (name, f"{dep.name}{dep.version_constraint}", available_version)
                        )
                        if not dep.optional:
                            result.success = False
                elif not dep.optional:
                    result.unmet_dependencies.append((name, dep))
                    result.success = False
        
        # Compute load order via topological sort
        load_order, sort_success = self._graph.topological_sort()
        
        if not sort_success and not cycles:
            result.warnings.append("Topological sort incomplete - some dependencies may be missing")
        
        # Reverse order so dependencies load first
        result.load_order = list(reversed(load_order))
        
        # Filter to only requested plugins and their dependencies
        if plugin_names:
            all_deps: set[str] = set()
            for name in plugin_names:
                all_deps.add(name)
                all_deps.update(self._graph.get_all_dependencies(name))
            result.load_order = [p for p in result.load_order if p in all_deps]
        
        return result
    
    def resolve_single(self, plugin_name: str) -> ResolutionResult:
        """Resolve dependencies for a single plugin."""
        return self.resolve([plugin_name])
    
    def get_load_order(self, plugin_names: list[str]) -> list[str]:
        """Get the load order for given plugins (dependencies first)."""
        result = self.resolve(plugin_names)
        return result.load_order
    
    def can_load(self, plugin_name: str) -> tuple[bool, list[str]]:
        """Check if a plugin can be loaded.
        
        Returns:
            Tuple of (can_load, reasons_if_not)
        """
        result = self.resolve_single(plugin_name)
        
        if result.success:
            return True, []
        
        reasons: list[str] = []
        
        for plugin, dep in result.unmet_dependencies:
            if plugin == plugin_name:
                reasons.append(f"Missing required dependency: {dep}")
        
        for cycle in result.circular_dependencies:
            if plugin_name in cycle:
                reasons.append(f"Circular dependency: {' -> '.join(cycle)}")
        
        for plugin, required, available in result.version_conflicts:
            if plugin == plugin_name:
                reasons.append(f"Version conflict: requires {required}, but {available} is available")
        
        return False, reasons
    
    def get_dependents(self, plugin_name: str) -> list[str]:
        """Get all plugins that depend on the given plugin."""
        return self._graph.get_dependents(plugin_name)
    
    def get_all_dependents(self, plugin_name: str) -> set[str]:
        """Get all plugins that transitively depend on the given plugin."""
        all_dependents: set[str] = set()
        to_check = [plugin_name]
        
        while to_check:
            current = to_check.pop()
            for dep in self._graph.get_dependents(current):
                if dep not in all_dependents:
                    all_dependents.add(dep)
                    to_check.append(dep)
        
        return all_dependents
    
    def would_break(self, plugin_name: str) -> list[str]:
        """Check what plugins would break if the given plugin is removed."""
        dependents = self.get_all_dependents(plugin_name)
        
        # Filter to only those with hard dependencies
        broken: list[str] = []
        for dep in dependents:
            deps = self._plugin_deps.get(dep, [])
            for d in deps:
                if d.name == plugin_name and not d.optional:
                    broken.append(dep)
                    break
        
        return broken
    
    def suggest_install_order(self, plugins: list[str]) -> list[str]:
        """Suggest installation order for a list of plugins."""
        return self.get_load_order(plugins)
    
    def find_conflicts(self) -> list[tuple[str, str, str]]:
        """Find all version conflicts in registered plugins."""
        conflicts: list[tuple[str, str, str]] = []
        
        for name, deps in self._plugin_deps.items():
            for dep in deps:
                if dep.name in self._plugin_versions:
                    available = self._plugin_versions[dep.name]
                    if not dep.satisfies_version(available):
                        conflicts.append((name, str(dep), available))
        
        return conflicts


class PluginLoaderWithDependencies(PluginLoader):
    """Extended plugin loader with dependency resolution."""
    
    def __init__(self, registry: PluginRegistry) -> None:
        super().__init__(registry)
        self._resolver = DependencyResolver(registry)
        self._loaded_order: list[str] = []
    
    @property
    def resolver(self) -> DependencyResolver:
        """Get the dependency resolver."""
        return self._resolver
    
    def load_with_dependencies(
        self,
        plugin_class: type[Plugin],
        config: dict | None = None,
        resolve: bool = True,
    ) -> tuple[Plugin | None, ResolutionResult]:
        """Load a plugin with dependency resolution.
        
        Args:
            plugin_class: Plugin class to load
            config: Plugin configuration
            resolve: Whether to check dependencies
            
        Returns:
            Tuple of (plugin instance or None, resolution result)
        """
        # Create temporary instance to get metadata
        temp = plugin_class.__new__(plugin_class)
        metadata = plugin_class.METADATA
        
        # Register for resolution
        dependencies = [
            PluginDependency.from_string(req) for req in metadata.requires
        ]
        self._resolver.register_plugin(metadata.name, metadata.version, dependencies)
        
        if resolve:
            result = self._resolver.resolve_single(metadata.name)
            if not result.success:
                return None, result
        else:
            result = ResolutionResult(success=True, load_order=[metadata.name])
        
        # Load the plugin
        try:
            plugin = self.load_from_class(plugin_class, config)
            self._loaded_order.append(metadata.name)
            return plugin, result
        except PluginError as e:
            result.success = False
            result.warnings.append(f"Load failed: {e}")
            return None, result
    
    def load_all_with_dependencies(
        self,
        plugin_classes: list[type[Plugin]],
    ) -> tuple[list[Plugin], ResolutionResult]:
        """Load multiple plugins in dependency order.
        
        Args:
            plugin_classes: Plugin classes to load
            
        Returns:
            Tuple of (loaded plugins, combined resolution result)
        """
        # Register all plugins
        for cls in plugin_classes:
            metadata = cls.METADATA
            dependencies = [
                PluginDependency.from_string(req) for req in metadata.requires
            ]
            self._resolver.register_plugin(metadata.name, metadata.version, dependencies)
        
        # Resolve all
        all_names = [cls.METADATA.name for cls in plugin_classes]
        result = self._resolver.resolve(all_names)
        
        if not result.success:
            return [], result
        
        # Load in order
        loaded: list[Plugin] = []
        name_to_class = {cls.METADATA.name: cls for cls in plugin_classes}
        
        for name in result.load_order:
            if name in name_to_class:
                try:
                    plugin = self.load_from_class(name_to_class[name])
                    loaded.append(plugin)
                    self._loaded_order.append(name)
                except PluginError as e:
                    result.warnings.append(f"Failed to load {name}: {e}")
        
        return loaded, result
    
    def get_load_order(self) -> list[str]:
        """Get the order in which plugins were loaded."""
        return list(self._loaded_order)
    
    def unload_with_dependents(self, plugin_name: str) -> list[str]:
        """Unload a plugin and all plugins that depend on it.
        
        Returns:
            List of plugins that were unloaded
        """
        # Find all dependents
        dependents = self._resolver.get_all_dependents(plugin_name)
        to_unload = [plugin_name] + list(dependents)
        
        # Reverse order (unload dependents first)
        unload_order = [p for p in reversed(self._loaded_order) if p in to_unload]
        
        unloaded: list[str] = []
        for name in unload_order:
            if self.registry.unregister(name):
                unloaded.append(name)
                self._loaded_order.remove(name)
        
        return unloaded


# Global resolver singleton
_dependency_resolver: DependencyResolver | None = None


def get_dependency_resolver() -> DependencyResolver:
    """Get the global dependency resolver."""
    global _dependency_resolver
    if _dependency_resolver is None:
        _dependency_resolver = DependencyResolver(get_plugin_registry())
    return _dependency_resolver


def resolve_dependencies(plugin_names: list[str]) -> ResolutionResult:
    """Resolve dependencies for plugins."""
    return get_dependency_resolver().resolve(plugin_names)