"""
Configuration module for Proxima.

This module provides comprehensive configuration management including:
- Layered configuration loading (CLI > env > user > project > defaults)
- Pydantic-based settings validation
- Secure secrets handling
- Configuration migration between versions
- Export/import utilities
- File watching for auto-reload
- Schema documentation and introspection
"""

from proxima.config.defaults import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_RELATIVE_PATH,
    ENV_PREFIX,
    PROJECT_CONFIG_FILENAME,
    USER_CONFIG_PATH,
)
from proxima.config.export_import import (
    BackupInfo,
    # Enums
    ExportFormat,
    # Data classes
    ExportOptions,
    ImportResult,
    cleanup_old_backups,
    # Backup functions
    create_backup,
    # Export functions
    export_config,
    # Template functions
    generate_template,
    # Import functions
    import_config,
    import_from_url,
    list_backups,
    restore_backup,
)
from proxima.config.migration import (
    # Constants
    CURRENT_VERSION,
    ConfigMigrator,
    # Enums
    MigrationDirection,
    # Classes
    MigrationRegistry,
    MigrationResult,
    # Data classes
    MigrationStep,
    auto_migrate,
    check_migration_status,
    # Functions
    get_config_version,
    get_migrator,
    # Decorators
    migration,
    needs_migration,
    register_pending_migrations,
    set_config_version,
)
from proxima.config.schema import (
    # Data classes
    FieldInfo,
    # Enums
    FieldType,
    SectionInfo,
    generate_completion_data,
    generate_json_schema,
    # Documentation generation
    generate_markdown_docs,
    get_field_examples,
    get_field_help,
    # Convenience
    get_settings_schema,
    # Introspection
    introspect_model,
    list_all_settings,
    print_settings_tree,
)
from proxima.config.secrets import (
    EncryptedFileStorage,
    EnvironmentStorage,
    KeyringStorage,
    MemoryStorage,
    # Enums
    SecretBackend,
    # Main manager
    SecretManager,
    # Data classes
    SecretMetadata,
    SecretResult,
    # Storage backends
    SecretStorage,
    # Utilities
    generate_secret_key,
    get_secret_manager,
    mask_secret,
    validate_api_key_format,
)
from proxima.config.settings import (
    BackendsSettings,
    # Config service
    ConfigService,
    ConsentSettings,
    FlatSettings,
    GeneralSettings,
    LLMSettings,
    ResourcesSettings,
    # Settings classes
    Settings,
    config_service,
    # Convenience functions
    get_settings,
    reload_settings,
)
from proxima.config.validation import (
    # Data classes
    ValidationIssue,
    ValidationResult,
    # Enums
    ValidationSeverity,
    validate_backend,
    validate_config_file,
    validate_env_var_name,
    validate_llm_provider,
    validate_memory_threshold,
    validate_model_name,
    validate_output_format,
    validate_path,
    # Main validation
    validate_settings,
    validate_storage_backend,
    validate_timeout,
    validate_url,
    # Individual validators
    validate_verbosity,
)
from proxima.config.watcher import (
    ConfigWatcher,
    # Data classes
    FileChange,
    # Watchers
    FileWatcher,
    PollingWatcher,
    WatchdogWatcher,
    WatchedConfigService,
    # Enums
    WatchEvent,
    # Convenience
    create_config_watcher,
    watch_config_file,
)

__all__ = [
    # ==========================================================================
    # DEFAULTS
    # ==========================================================================
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_RELATIVE_PATH",
    "ENV_PREFIX",
    "PROJECT_CONFIG_FILENAME",
    "USER_CONFIG_PATH",
    # ==========================================================================
    # SETTINGS
    # ==========================================================================
    # Settings classes
    "Settings",
    "GeneralSettings",
    "BackendsSettings",
    "LLMSettings",
    "ResourcesSettings",
    "ConsentSettings",
    # Config service
    "ConfigService",
    "config_service",
    # Convenience functions
    "get_settings",
    "reload_settings",
    "FlatSettings",
    # ==========================================================================
    # VALIDATION
    # ==========================================================================
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "validate_settings",
    "validate_config_file",
    "validate_verbosity",
    "validate_output_format",
    "validate_backend",
    "validate_timeout",
    "validate_llm_provider",
    "validate_url",
    "validate_memory_threshold",
    "validate_path",
    "validate_storage_backend",
    "validate_model_name",
    "validate_env_var_name",
    # ==========================================================================
    # EXPORT/IMPORT
    # ==========================================================================
    "ExportFormat",
    "ExportOptions",
    "ImportResult",
    "BackupInfo",
    "export_config",
    "import_config",
    "import_from_url",
    "create_backup",
    "list_backups",
    "restore_backup",
    "cleanup_old_backups",
    "generate_template",
    # ==========================================================================
    # SECRETS
    # ==========================================================================
    "SecretBackend",
    "SecretMetadata",
    "SecretResult",
    "SecretStorage",
    "KeyringStorage",
    "EncryptedFileStorage",
    "EnvironmentStorage",
    "MemoryStorage",
    "SecretManager",
    "get_secret_manager",
    "generate_secret_key",
    "mask_secret",
    "validate_api_key_format",
    # ==========================================================================
    # MIGRATION
    # ==========================================================================
    "CURRENT_VERSION",
    "MigrationDirection",
    "MigrationStep",
    "MigrationResult",
    "MigrationRegistry",
    "ConfigMigrator",
    "get_config_version",
    "set_config_version",
    "needs_migration",
    "get_migrator",
    "auto_migrate",
    "check_migration_status",
    "migration",
    "register_pending_migrations",
    # ==========================================================================
    # SCHEMA
    # ==========================================================================
    "FieldType",
    "FieldInfo",
    "SectionInfo",
    "introspect_model",
    "generate_markdown_docs",
    "generate_json_schema",
    "generate_completion_data",
    "get_settings_schema",
    "get_field_help",
    "get_field_examples",
    "list_all_settings",
    "print_settings_tree",
    # ==========================================================================
    # WATCHER
    # ==========================================================================
    "WatchEvent",
    "FileChange",
    "FileWatcher",
    "PollingWatcher",
    "WatchdogWatcher",
    "ConfigWatcher",
    "WatchedConfigService",
    "create_config_watcher",
    "watch_config_file",
]
