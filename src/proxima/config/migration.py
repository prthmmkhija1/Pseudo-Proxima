"""Configuration migration system.

Handles versioning and migration of configuration files between versions.
Supports automatic upgrades, manual migration, and rollback capabilities.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# Current configuration schema version
CURRENT_VERSION = "1.0.0"


class MigrationDirection(Enum):
    """Direction of migration."""

    UP = "up"  # Upgrade to newer version
    DOWN = "down"  # Downgrade to older version


@dataclass
class MigrationStep:
    """A single migration step between versions."""

    from_version: str
    to_version: str
    description: str
    migrate_up: Callable[[dict[str, Any]], dict[str, Any]]
    migrate_down: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    breaking: bool = False  # Whether this is a breaking change

    def __str__(self) -> str:
        arrow = "→" if not self.breaking else "⚠️→"
        return f"{self.from_version} {arrow} {self.to_version}: {self.description}"


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    from_version: str
    to_version: str
    config: dict[str, Any] | None = None
    steps_applied: list[MigrationStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    backup_path: Path | None = None

    def __str__(self) -> str:
        if self.success:
            return f"✓ Migrated from {self.from_version} to {self.to_version} ({len(self.steps_applied)} steps)"
        return f"✗ Migration failed: {', '.join(self.errors)}"


# =============================================================================
# MIGRATION REGISTRY
# =============================================================================


class MigrationRegistry:
    """Registry of all available migrations."""

    def __init__(self) -> None:
        self._migrations: list[MigrationStep] = []
        self._register_builtin_migrations()

    def _register_builtin_migrations(self) -> None:
        """Register built-in migrations."""
        # Example migration: 0.0.0 (unversioned) to 1.0.0
        self.register(
            MigrationStep(
                from_version="0.0.0",
                to_version="1.0.0",
                description="Add schema version, restructure settings",
                migrate_up=self._migrate_000_to_100,
                migrate_down=self._migrate_100_to_000,
            )
        )

    def register(self, step: MigrationStep) -> None:
        """Register a migration step."""
        self._migrations.append(step)
        # Keep sorted by version
        self._migrations.sort(key=lambda s: _version_tuple(s.from_version))

    def get_path(self, from_version: str, to_version: str) -> list[MigrationStep]:
        """Get the migration path between two versions."""
        if from_version == to_version:
            return []

        from_tuple = _version_tuple(from_version)
        to_tuple = _version_tuple(to_version)

        if from_tuple < to_tuple:
            # Upgrading
            return self._get_upgrade_path(from_version, to_version)
        else:
            # Downgrading
            return self._get_downgrade_path(from_version, to_version)

    def _get_upgrade_path(self, from_version: str, to_version: str) -> list[MigrationStep]:
        """Get upgrade migration path."""
        path = []
        current = from_version

        while _version_tuple(current) < _version_tuple(to_version):
            # Find next migration step
            next_step = None
            for step in self._migrations:
                if step.from_version == current:
                    if _version_tuple(step.to_version) <= _version_tuple(to_version):
                        next_step = step
                        break

            if next_step is None:
                break

            path.append(next_step)
            current = next_step.to_version

        return path

    def _get_downgrade_path(self, from_version: str, to_version: str) -> list[MigrationStep]:
        """Get downgrade migration path."""
        path = []
        current = from_version

        while _version_tuple(current) > _version_tuple(to_version):
            # Find migration that leads to current version
            prev_step = None
            for step in reversed(self._migrations):
                if step.to_version == current and step.migrate_down is not None:
                    if _version_tuple(step.from_version) >= _version_tuple(to_version):
                        prev_step = step
                        break

            if prev_step is None:
                break

            path.append(prev_step)
            current = prev_step.from_version

        return path

    # ==========================================================================
    # BUILT-IN MIGRATION FUNCTIONS
    # ==========================================================================

    @staticmethod
    def _migrate_000_to_100(config: dict[str, Any]) -> dict[str, Any]:
        """Migrate from unversioned to 1.0.0."""
        result = copy.deepcopy(config)

        # Add version
        result["_version"] = "1.0.0"
        result["_migrated_at"] = datetime.now().isoformat()

        # Ensure all sections exist
        result.setdefault("general", {})
        result.setdefault("backends", {})
        result.setdefault("llm", {})
        result.setdefault("resources", {})
        result.setdefault("consent", {})

        # Add storage_backend if not present
        if "storage_backend" not in result["general"]:
            result["general"]["storage_backend"] = "sqlite"

        # Add data_dir if not present
        if "data_dir" not in result["general"]:
            result["general"]["data_dir"] = ""

        return result

    @staticmethod
    def _migrate_100_to_000(config: dict[str, Any]) -> dict[str, Any]:
        """Downgrade from 1.0.0 to unversioned."""
        result = copy.deepcopy(config)

        # Remove version info
        result.pop("_version", None)
        result.pop("_migrated_at", None)

        # Remove new fields
        if "general" in result:
            result["general"].pop("storage_backend", None)
            result["general"].pop("data_dir", None)

        return result


# =============================================================================
# VERSION UTILITIES
# =============================================================================


def _version_tuple(version: str) -> tuple[int, ...]:
    """Convert version string to comparable tuple."""
    try:
        # Handle versions like "1.0.0", "1.0", "1"
        parts = version.split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings. Returns -1, 0, or 1."""
    t1, t2 = _version_tuple(v1), _version_tuple(v2)
    if t1 < t2:
        return -1
    elif t1 > t2:
        return 1
    return 0


def get_config_version(config: dict[str, Any]) -> str:
    """Get the version from a configuration dictionary."""
    return config.get("_version", "0.0.0")


def set_config_version(config: dict[str, Any], version: str) -> dict[str, Any]:
    """Set the version in a configuration dictionary."""
    result = copy.deepcopy(config)
    result["_version"] = version
    result["_updated_at"] = datetime.now().isoformat()
    return result


def needs_migration(config: dict[str, Any], target_version: str | None = None) -> bool:
    """Check if configuration needs migration."""
    target_version = target_version or CURRENT_VERSION
    current = get_config_version(config)
    return _compare_versions(current, target_version) != 0


# =============================================================================
# MIGRATOR
# =============================================================================


class ConfigMigrator:
    """Handles configuration migration between versions."""

    def __init__(self, registry: MigrationRegistry | None = None) -> None:
        self.registry = registry or MigrationRegistry()

    def migrate(
        self,
        config: dict[str, Any],
        target_version: str | None = None,
        create_backup: bool = True,
        config_path: Path | None = None,
    ) -> MigrationResult:
        """Migrate configuration to target version.

        Args:
            config: Configuration dictionary to migrate
            target_version: Target version (default: CURRENT_VERSION)
            create_backup: Whether to create a backup before migration
            config_path: Path to config file (for backup)

        Returns:
            MigrationResult with migrated configuration or errors
        """
        target_version = target_version or CURRENT_VERSION
        from_version = get_config_version(config)

        result = MigrationResult(
            success=False,
            from_version=from_version,
            to_version=target_version,
        )

        # Check if migration is needed
        if not needs_migration(config, target_version):
            result.success = True
            result.config = config
            return result

        # Get migration path
        path = self.registry.get_path(from_version, target_version)

        if not path:
            # No direct path - check if versions are valid
            if from_version != target_version:
                result.errors.append(f"No migration path from {from_version} to {target_version}")
                return result

        # Create backup if requested
        if create_backup and config_path and config_path.exists():
            backup_path = self._create_backup(config_path)
            result.backup_path = backup_path

        # Apply migrations
        current_config = copy.deepcopy(config)
        direction = (
            MigrationDirection.UP
            if _version_tuple(from_version) < _version_tuple(target_version)
            else MigrationDirection.DOWN
        )

        try:
            for step in path:
                if step.breaking:
                    result.warnings.append(
                        f"Breaking change in {step.from_version} → {step.to_version}: {step.description}"
                    )

                if direction == MigrationDirection.UP:
                    current_config = step.migrate_up(current_config)
                else:
                    if step.migrate_down is None:
                        result.errors.append(
                            f"Downgrade not supported: {step.to_version} → {step.from_version}"
                        )
                        return result
                    current_config = step.migrate_down(current_config)

                result.steps_applied.append(step)

            # Update version
            current_config = set_config_version(current_config, target_version)

            result.success = True
            result.config = current_config

        except Exception as e:
            result.errors.append(f"Migration failed: {e}")

        return result

    def migrate_file(
        self,
        config_path: Path,
        target_version: str | None = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Migrate a configuration file.

        Args:
            config_path: Path to the configuration file
            target_version: Target version (default: CURRENT_VERSION)
            dry_run: If True, don't actually write changes

        Returns:
            MigrationResult with migrated configuration or errors
        """
        target_version = target_version or CURRENT_VERSION

        # Load current config
        if not config_path.exists():
            return MigrationResult(
                success=False,
                from_version="unknown",
                to_version=target_version,
                errors=[f"Configuration file not found: {config_path}"],
            )

        try:
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            return MigrationResult(
                success=False,
                from_version="unknown",
                to_version=target_version,
                errors=[f"Failed to read configuration: {e}"],
            )

        # Perform migration
        result = self.migrate(
            config,
            target_version,
            create_backup=not dry_run,
            config_path=config_path,
        )

        # Write result if not dry run and successful
        if result.success and result.config and not dry_run:
            try:
                with config_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(result.config, f, sort_keys=False)
            except Exception as e:
                result.success = False
                result.errors.append(f"Failed to write migrated config: {e}")

        return result

    def _create_backup(self, config_path: Path) -> Path:
        """Create a backup of the configuration file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = config_path.parent / ".proxima_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_name = f"{config_path.stem}_premigration_{timestamp}{config_path.suffix}"
        backup_path = backup_dir / backup_name

        import shutil

        shutil.copy2(config_path, backup_path)

        return backup_path

    def list_migrations(self) -> list[str]:
        """List all available migrations."""
        return [str(m) for m in self.registry._migrations]

    def get_upgrade_path(self, config: dict[str, Any]) -> list[MigrationStep]:
        """Get the upgrade path for a configuration."""
        from_version = get_config_version(config)
        return self.registry.get_path(from_version, CURRENT_VERSION)


# =============================================================================
# MIGRATION DECORATORS
# =============================================================================


def migration(
    from_version: str,
    to_version: str,
    description: str = "",
    breaking: bool = False,
):
    """Decorator to register a migration function.

    Usage:
        @migration("1.0.0", "1.1.0", "Add new feature settings")
        def migrate_100_to_110(config):
            config["new_feature"] = {"enabled": True}
            return config
    """

    def decorator(func: Callable[[dict[str, Any]], dict[str, Any]]):
        step = MigrationStep(
            from_version=from_version,
            to_version=to_version,
            description=description or func.__doc__ or "No description",
            migrate_up=func,
            breaking=breaking,
        )
        _pending_migrations.append(step)
        return func

    return decorator


# Store migrations registered via decorator
_pending_migrations: list[MigrationStep] = []


def register_pending_migrations(registry: MigrationRegistry) -> None:
    """Register all migrations created via decorator."""
    for step in _pending_migrations:
        registry.register(step)
    _pending_migrations.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton migrator instance
_migrator: ConfigMigrator | None = None


def get_migrator() -> ConfigMigrator:
    """Get the global migrator instance."""
    global _migrator
    if _migrator is None:
        _migrator = ConfigMigrator()
    return _migrator


def auto_migrate(config: dict[str, Any]) -> dict[str, Any]:
    """Automatically migrate config to current version."""
    migrator = get_migrator()
    result = migrator.migrate(config)

    if result.success and result.config:
        return result.config

    # Return original if migration failed
    return config


def check_migration_status(config: dict[str, Any]) -> dict[str, Any]:
    """Check migration status of a configuration.

    Returns:
        Dictionary with migration status information
    """
    current = get_config_version(config)
    needs_upgrade = needs_migration(config, CURRENT_VERSION)

    migrator = get_migrator()
    path = migrator.get_upgrade_path(config) if needs_upgrade else []

    return {
        "current_version": current,
        "target_version": CURRENT_VERSION,
        "needs_migration": needs_upgrade,
        "migration_steps": len(path),
        "breaking_changes": any(s.breaking for s in path),
        "steps": [str(s) for s in path],
    }
