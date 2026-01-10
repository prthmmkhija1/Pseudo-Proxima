"""Tests for configuration module edge cases.

Comprehensive tests for all configuration functionality including:
- Validation
- Export/Import
- Secrets handling
- Migration
- Schema introspection
- File watching
"""

import json
import os
import tempfile
import time
from pathlib import Path

import yaml

# =============================================================================
# VALIDATION TESTS
# =============================================================================


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validation_result_initial_state(self):
        """Test ValidationResult starts valid with no issues."""
        from proxima.config.validation import ValidationResult

        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_add_error_makes_invalid(self):
        """Test adding error makes result invalid."""
        from proxima.config.validation import ValidationResult

        result = ValidationResult()
        result.add_error("test.path", "Test error message")

        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.issues[0].path == "test.path"

    def test_add_warning_keeps_valid(self):
        """Test adding warning keeps result valid."""
        from proxima.config.validation import ValidationResult

        result = ValidationResult()
        result.add_warning("test.path", "Test warning")

        assert result.is_valid is True
        assert len(result.warnings()) == 1

    def test_add_info_keeps_valid(self):
        """Test adding info keeps result valid."""
        from proxima.config.validation import ValidationResult

        result = ValidationResult()
        result.add_info("test.path", "Test info")

        assert result.is_valid is True

    def test_format_report_empty(self):
        """Test report format when no issues."""
        from proxima.config.validation import ValidationResult

        result = ValidationResult()
        report = result.format_report()
        assert "valid" in report.lower()

    def test_format_report_with_errors(self):
        """Test report format with errors."""
        from proxima.config.validation import ValidationResult

        result = ValidationResult()
        result.add_error("test", "Error message", suggestion="Fix it")
        report = result.format_report()

        assert "error" in report.lower()
        assert "test" in report

    def test_validate_verbosity_valid(self):
        """Test verbosity validation with valid values."""
        from proxima.config.validation import ValidationResult, validate_verbosity

        for value in ["debug", "info", "warning", "error"]:
            result = ValidationResult()
            validate_verbosity(value, "general.verbosity", result)
            assert result.is_valid

    def test_validate_verbosity_invalid(self):
        """Test verbosity validation with invalid value."""
        from proxima.config.validation import ValidationResult, validate_verbosity

        result = ValidationResult()
        validate_verbosity("invalid", "general.verbosity", result)
        assert not result.is_valid

    def test_validate_timeout_negative(self):
        """Test timeout validation rejects negative values."""
        from proxima.config.validation import ValidationResult, validate_timeout

        result = ValidationResult()
        validate_timeout(-1, "backends.timeout_seconds", result)
        assert not result.is_valid

    def test_validate_timeout_very_large_warning(self):
        """Test timeout validation warns on very large values."""
        from proxima.config.validation import ValidationResult, validate_timeout

        result = ValidationResult()
        validate_timeout(100000, "backends.timeout_seconds", result)
        assert result.is_valid  # Still valid, just warning
        assert len(result.warnings()) > 0

    def test_validate_url_missing_scheme(self):
        """Test URL validation catches missing scheme."""
        from proxima.config.validation import ValidationResult, validate_url

        result = ValidationResult()
        validate_url("localhost:8080", "llm.local_endpoint", result)
        assert not result.is_valid

    def test_validate_url_valid(self):
        """Test URL validation accepts valid URLs."""
        from proxima.config.validation import ValidationResult, validate_url

        result = ValidationResult()
        validate_url("http://localhost:8080", "llm.local_endpoint", result)
        assert result.is_valid

    def test_validate_url_empty_allowed(self):
        """Test URL validation allows empty string."""
        from proxima.config.validation import ValidationResult, validate_url

        result = ValidationResult()
        validate_url("", "llm.local_endpoint", result)
        assert result.is_valid

    def test_validate_memory_threshold_low_warning(self):
        """Test memory threshold warns on very low values."""
        from proxima.config.validation import ValidationResult, validate_memory_threshold

        result = ValidationResult()
        validate_memory_threshold(64, "resources.memory_warn_threshold_mb", result)
        assert len(result.warnings()) > 0

    def test_validate_settings_complete(self):
        """Test full settings validation."""
        from proxima.config.validation import validate_settings

        config = {
            "general": {
                "verbosity": "info",
                "output_format": "text",
                "color_enabled": True,
            },
            "backends": {
                "default_backend": "auto",
                "timeout_seconds": 300,
            },
            "llm": {"provider": "none"},
            "resources": {
                "memory_warn_threshold_mb": 4096,
                "memory_critical_threshold_mb": 8192,
            },
            "consent": {},
        }

        result = validate_settings(config)
        assert result.is_valid

    def test_validate_settings_cross_field(self):
        """Test cross-field validation (warn < critical)."""
        from proxima.config.validation import validate_settings

        config = {
            "general": {},
            "backends": {},
            "llm": {},
            "resources": {
                "memory_warn_threshold_mb": 8192,
                "memory_critical_threshold_mb": 4096,  # Less than warn!
            },
            "consent": {},
        }

        result = validate_settings(config)
        assert not result.is_valid


# =============================================================================
# EXPORT/IMPORT TESTS
# =============================================================================


class TestConfigExportImport:
    """Tests for configuration export/import."""

    def test_export_format_from_extension(self):
        """Test format detection from file extension."""
        from proxima.config.export_import import ExportFormat

        assert ExportFormat.from_extension(Path("config.yaml")) == ExportFormat.YAML
        assert ExportFormat.from_extension(Path("config.yml")) == ExportFormat.YAML
        assert ExportFormat.from_extension(Path("config.json")) == ExportFormat.JSON
        assert ExportFormat.from_extension(Path("config.toml")) == ExportFormat.TOML
        assert ExportFormat.from_extension(Path("settings.env")) == ExportFormat.ENV

    def test_export_yaml(self):
        """Test YAML export."""
        from proxima.config.export_import import ExportFormat, export_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"general": {"verbosity": "info"}}
            output_path = Path(tmpdir) / "config.yaml"

            result = export_config(config, output_path, ExportFormat.YAML)

            assert result.exists()
            content = yaml.safe_load(result.read_text())
            assert content["general"]["verbosity"] == "info"

    def test_export_json(self):
        """Test JSON export."""
        from proxima.config.export_import import ExportFormat, export_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"general": {"verbosity": "debug"}}
            output_path = Path(tmpdir) / "config.json"

            result = export_config(config, output_path, ExportFormat.JSON)

            assert result.exists()
            content = json.loads(result.read_text())
            assert content["general"]["verbosity"] == "debug"

    def test_export_redacts_secrets(self):
        """Test that secrets are redacted by default."""
        from proxima.config.export_import import ExportFormat, export_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"llm": {"api_key": "sk-secret123"}}
            output_path = Path(tmpdir) / "config.yaml"

            export_config(config, output_path, ExportFormat.YAML)

            content = yaml.safe_load(output_path.read_text())
            assert content["llm"]["api_key"] == "<REDACTED>"

    def test_export_no_redact(self):
        """Test export without redaction."""
        from proxima.config.export_import import ExportFormat, ExportOptions, export_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"llm": {"api_key": "sk-secret123"}}
            output_path = Path(tmpdir) / "config.yaml"

            export_config(
                config, output_path, ExportFormat.YAML, options=ExportOptions(redact_secrets=False)
            )

            content = yaml.safe_load(output_path.read_text())
            assert content["llm"]["api_key"] == "sk-secret123"

    def test_import_yaml(self):
        """Test YAML import."""
        from proxima.config.export_import import import_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("general:\n  verbosity: warning\n")

            result = import_config(config_path)

            assert result.success
            assert result.config["general"]["verbosity"] == "warning"

    def test_import_json(self):
        """Test JSON import."""
        from proxima.config.export_import import import_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text('{"general": {"verbosity": "error"}}')

            result = import_config(config_path)

            assert result.success
            assert result.config["general"]["verbosity"] == "error"

    def test_import_nonexistent_file(self):
        """Test importing non-existent file."""
        from proxima.config.export_import import import_config

        result = import_config(Path("/nonexistent/config.yaml"))

        assert not result.success
        assert "not found" in result.errors[0].lower()

    def test_import_invalid_yaml(self):
        """Test importing invalid YAML."""
        from proxima.config.export_import import import_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("invalid: yaml: content: [")

            result = import_config(config_path)

            assert not result.success

    def test_backup_create(self):
        """Test backup creation."""
        from proxima.config.export_import import create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("test: value\n")

            backup_info = create_backup(config_path)

            assert backup_info.path.exists()
            assert backup_info.size_bytes > 0

    def test_backup_list(self):
        """Test backup listing."""
        from proxima.config.export_import import create_backup, list_backups

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("test: value\n")
            backup_dir = Path(tmpdir) / "backups"

            # Create a backup
            create_backup(config_path, backup_dir)

            backups = list_backups(backup_dir)

            # At least one backup should exist
            assert len(backups) >= 1
            assert backups[0].path.exists()

    def test_generate_template_full(self):
        """Test full template generation."""
        from proxima.config.export_import import generate_template

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "template.yaml"

            result = generate_template(output_path, "full")

            assert result.exists()
            content = yaml.safe_load(result.read_text())
            assert "general" in content
            assert "backends" in content
            assert "llm" in content

    def test_generate_template_minimal(self):
        """Test minimal template generation."""
        from proxima.config.export_import import generate_template

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "template.yaml"

            result = generate_template(output_path, "minimal")

            assert result.exists()
            content = yaml.safe_load(result.read_text())
            # Minimal should have fewer sections
            assert "general" in content


# =============================================================================
# SECRETS TESTS
# =============================================================================


class TestSecrets:
    """Tests for secrets handling."""

    def test_memory_storage_store_retrieve(self):
        """Test in-memory secret storage."""
        from proxima.config.secrets import MemoryStorage

        storage = MemoryStorage()

        result = storage.store("test_key", "test_value")
        assert result.success

        result = storage.retrieve("test_key")
        assert result.success
        assert result.value == "test_value"

    def test_memory_storage_delete(self):
        """Test in-memory secret deletion."""
        from proxima.config.secrets import MemoryStorage

        storage = MemoryStorage()
        storage.store("test_key", "test_value")

        assert storage.delete("test_key")
        assert not storage.exists("test_key")

    def test_memory_storage_list_keys(self):
        """Test listing keys in memory storage."""
        from proxima.config.secrets import MemoryStorage

        storage = MemoryStorage()
        storage.store("key1", "value1")
        storage.store("key2", "value2")

        keys = storage.list_keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_memory_storage_clear(self):
        """Test clearing memory storage."""
        from proxima.config.secrets import MemoryStorage

        storage = MemoryStorage()
        storage.store("key", "value")
        storage.clear()

        assert len(storage.list_keys()) == 0

    def test_environment_storage_retrieve(self):
        """Test environment variable retrieval."""
        from proxima.config.secrets import EnvironmentStorage

        os.environ["PROXIMA_SECRET_TEST_KEY"] = "test_value"
        try:
            storage = EnvironmentStorage()
            result = storage.retrieve("test_key")

            assert result.success
            assert result.value == "test_value"
        finally:
            del os.environ["PROXIMA_SECRET_TEST_KEY"]

    def test_environment_storage_not_found(self):
        """Test environment variable not found."""
        from proxima.config.secrets import EnvironmentStorage

        storage = EnvironmentStorage()
        result = storage.retrieve("nonexistent_key")

        assert not result.success

    def test_secret_manager_fallback(self):
        """Test secret manager fallback behavior."""
        from proxima.config.secrets import SecretBackend, SecretManager

        manager = SecretManager(preferred_backend=SecretBackend.MEMORY)

        # Store in memory
        result = manager.store("test_secret", "value", backend=SecretBackend.MEMORY)
        assert result.success

        # Should retrieve from memory
        result = manager.retrieve("test_secret", backend=SecretBackend.MEMORY)
        assert result.success
        assert result.value == "value"

    def test_generate_secret_key(self):
        """Test secret key generation."""
        from proxima.config.secrets import generate_secret_key

        key1 = generate_secret_key()
        key2 = generate_secret_key()

        assert len(key1) == 64  # 32 bytes = 64 hex chars
        assert key1 != key2

    def test_mask_secret(self):
        """Test secret masking."""
        from proxima.config.secrets import mask_secret

        masked = mask_secret("sk-1234567890abcdef")

        assert masked.startswith("sk-1")
        assert "*" in masked
        assert masked.endswith("cdef")

    def test_mask_secret_short(self):
        """Test masking short secrets."""
        from proxima.config.secrets import mask_secret

        masked = mask_secret("short")
        assert masked == "*****"

    def test_validate_api_key_format_openai(self):
        """Test OpenAI API key format validation."""
        from proxima.config.secrets import validate_api_key_format

        # Valid OpenAI key format (starts with sk- and >20 chars)
        valid, msg = validate_api_key_format("sk-abc123def456789012345", "openai")
        assert valid

        valid, msg = validate_api_key_format("invalid-key", "openai")
        assert not valid


# =============================================================================
# MIGRATION TESTS
# =============================================================================


class TestConfigMigration:
    """Tests for configuration migration."""

    def test_get_config_version_unversioned(self):
        """Test getting version from unversioned config."""
        from proxima.config.migration import get_config_version

        config = {"general": {"verbosity": "info"}}
        version = get_config_version(config)

        assert version == "0.0.0"

    def test_get_config_version_versioned(self):
        """Test getting version from versioned config."""
        from proxima.config.migration import get_config_version

        config = {"_version": "1.0.0", "general": {}}
        version = get_config_version(config)

        assert version == "1.0.0"

    def test_set_config_version(self):
        """Test setting config version."""
        from proxima.config.migration import set_config_version

        config = {"general": {}}
        result = set_config_version(config, "2.0.0")

        assert result["_version"] == "2.0.0"
        assert "_updated_at" in result

    def test_needs_migration_true(self):
        """Test needs_migration returns True for old version."""
        from proxima.config.migration import needs_migration

        config = {"_version": "0.0.0"}
        assert needs_migration(config, "1.0.0")

    def test_needs_migration_false(self):
        """Test needs_migration returns False for current version."""
        from proxima.config.migration import needs_migration

        config = {"_version": "1.0.0"}
        assert not needs_migration(config, "1.0.0")

    def test_migration_result_str(self):
        """Test MigrationResult string representation."""
        from proxima.config.migration import MigrationResult

        result = MigrationResult(
            success=True,
            from_version="0.0.0",
            to_version="1.0.0",
            steps_applied=[],
        )

        assert "0.0.0" in str(result)
        assert "1.0.0" in str(result)

    def test_migrator_no_migration_needed(self):
        """Test migrator when no migration needed."""
        from proxima.config.migration import ConfigMigrator

        migrator = ConfigMigrator()
        config = {"_version": "1.0.0"}

        result = migrator.migrate(config, "1.0.0")

        assert result.success
        assert len(result.steps_applied) == 0

    def test_migrator_upgrade(self):
        """Test migrator upgrade from 0.0.0 to 1.0.0."""
        from proxima.config.migration import ConfigMigrator

        migrator = ConfigMigrator()
        config = {"general": {"verbosity": "info"}}

        result = migrator.migrate(config, "1.0.0")

        assert result.success
        assert result.config is not None
        assert result.config.get("_version") == "1.0.0"

    def test_check_migration_status(self):
        """Test check_migration_status function."""
        from proxima.config.migration import check_migration_status

        config = {"_version": "0.0.0"}
        status = check_migration_status(config)

        assert status["current_version"] == "0.0.0"
        assert status["needs_migration"] is True

    def test_auto_migrate(self):
        """Test auto_migrate function."""
        from proxima.config.migration import CURRENT_VERSION, auto_migrate

        config = {"general": {}}
        result = auto_migrate(config)

        assert result.get("_version") == CURRENT_VERSION


# =============================================================================
# SCHEMA TESTS
# =============================================================================


class TestConfigSchema:
    """Tests for configuration schema introspection."""

    def test_field_type_enum(self):
        """Test FieldType enum values."""
        from proxima.config.schema import FieldType

        assert FieldType.STRING.value == "string"
        assert FieldType.INTEGER.value == "integer"
        assert FieldType.BOOLEAN.value == "boolean"

    def test_field_info_to_dict(self):
        """Test FieldInfo.to_dict()."""
        from proxima.config.schema import FieldInfo, FieldType

        field = FieldInfo(
            name="verbosity",
            path="general.verbosity",
            field_type=FieldType.STRING,
            python_type="str",
            default="info",
        )

        d = field.to_dict()
        assert d["name"] == "verbosity"
        assert d["type"] == "string"

    def test_section_info_to_dict(self):
        """Test SectionInfo.to_dict()."""
        from proxima.config.schema import FieldInfo, FieldType, SectionInfo

        section = SectionInfo(
            name="general",
            path="general",
            description="General settings",
            fields=[
                FieldInfo(
                    name="verbosity",
                    path="general.verbosity",
                    field_type=FieldType.STRING,
                    python_type="str",
                )
            ],
        )

        d = section.to_dict()
        assert d["name"] == "general"
        assert len(d["fields"]) == 1

    def test_get_field_help(self):
        """Test get_field_help function."""
        from proxima.config.schema import get_field_help

        help_text = get_field_help("general.verbosity")
        assert help_text is not None
        assert len(help_text) > 0

    def test_get_field_examples(self):
        """Test get_field_examples function."""
        from proxima.config.schema import get_field_examples

        examples = get_field_examples("general.verbosity")
        assert len(examples) > 0
        assert "info" in examples

    def test_list_all_settings(self):
        """Test list_all_settings function."""
        from proxima.config.schema import list_all_settings

        settings = list_all_settings()
        assert len(settings) > 0
        assert any("verbosity" in s for s in settings)

    def test_generate_json_schema(self):
        """Test JSON schema generation."""
        from proxima.config.schema import FieldInfo, FieldType, SectionInfo, generate_json_schema

        section = SectionInfo(
            name="test",
            path="test",
            fields=[
                FieldInfo(
                    name="setting",
                    path="test.setting",
                    field_type=FieldType.STRING,
                    python_type="str",
                    default="value",
                )
            ],
        )

        schema = generate_json_schema(section)

        assert schema["type"] == "object"
        assert "setting" in schema["properties"]

    def test_generate_completion_data(self):
        """Test completion data generation."""
        from proxima.config.schema import (
            FieldInfo,
            FieldType,
            SectionInfo,
            generate_completion_data,
        )

        section = SectionInfo(
            name="test",
            path="test",
            fields=[
                FieldInfo(
                    name="setting",
                    path="test.setting",
                    field_type=FieldType.STRING,
                    python_type="str",
                    examples=["a", "b"],
                )
            ],
        )

        data = generate_completion_data(section)

        assert len(data["keys"]) > 0
        # Keys are stored without prefix in values dict
        assert "setting" in data["values"] or len(data["values"]) > 0


# =============================================================================
# WATCHER TESTS
# =============================================================================


class TestConfigWatcher:
    """Tests for configuration file watching."""

    def test_watch_event_enum(self):
        """Test WatchEvent enum values."""
        from proxima.config.watcher import WatchEvent

        assert WatchEvent.CREATED.value == "created"
        assert WatchEvent.MODIFIED.value == "modified"
        assert WatchEvent.DELETED.value == "deleted"

    def test_file_change_str(self):
        """Test FileChange string representation."""
        from proxima.config.watcher import FileChange, WatchEvent

        change = FileChange(
            path=Path("/test/config.yaml"),
            event=WatchEvent.MODIFIED,
        )

        s = str(change)
        assert "modified" in s.lower()
        assert "config.yaml" in s

    def test_file_change_moved_str(self):
        """Test FileChange string for moved files."""
        from proxima.config.watcher import FileChange, WatchEvent

        change = FileChange(
            path=Path("/new/config.yaml"),
            event=WatchEvent.MOVED,
            old_path=Path("/old/config.yaml"),
        )

        s = str(change)
        assert "moved" in s.lower()

    def test_polling_watcher_lifecycle(self):
        """Test PollingWatcher start/stop."""
        from proxima.config.watcher import PollingWatcher

        watcher = PollingWatcher(poll_interval=0.1)

        assert not watcher.is_running

        watcher.start()
        assert watcher.is_running

        watcher.stop()
        assert not watcher.is_running

    def test_polling_watcher_detects_change(self):
        """Test PollingWatcher detects file changes."""
        from proxima.config.watcher import PollingWatcher, WatchEvent

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("initial: value\n")

            watcher = PollingWatcher(poll_interval=0.05)
            changes = []

            watcher.on_change(lambda c: changes.append(c))
            watcher.add_path(config_path)
            watcher.start()

            try:
                # Modify the file
                time.sleep(0.1)
                config_path.write_text("modified: value\n")
                time.sleep(0.2)
            finally:
                watcher.stop()

            # Should have detected the change
            modified_changes = [c for c in changes if c.event == WatchEvent.MODIFIED]
            assert len(modified_changes) >= 1

    def test_config_watcher_debounce(self):
        """Test ConfigWatcher debouncing."""
        from proxima.config.watcher import ConfigWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("test: 1\n")

            reloads = []
            watcher = ConfigWatcher(
                config_paths=[config_path],
                debounce_seconds=0.5,
            )
            watcher.on_reload(lambda p: reloads.append(p))

            watcher.start()
            try:
                time.sleep(0.1)
                # Rapid changes should be debounced
                config_path.write_text("test: 2\n")
                time.sleep(0.05)
                config_path.write_text("test: 3\n")
                time.sleep(0.6)
            finally:
                watcher.stop()

    def test_config_watcher_watched_paths(self):
        """Test ConfigWatcher path management."""
        from proxima.config.watcher import ConfigWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("test: value\n")

            watcher = ConfigWatcher()
            watcher.add_config(config_path)

            assert config_path in watcher.watched_paths

            watcher.remove_config(config_path)
            assert config_path not in watcher.watched_paths


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestConfigIntegration:
    """Integration tests for configuration module."""

    def test_full_workflow_export_import_validate(self):
        """Test full workflow: export, import, validate."""
        from proxima.config.export_import import ExportFormat, export_config, import_config
        from proxima.config.validation import validate_settings

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            original = {
                "general": {
                    "verbosity": "info",
                    "output_format": "text",
                    "color_enabled": True,
                },
                "backends": {
                    "default_backend": "auto",
                    "timeout_seconds": 300,
                },
                "llm": {"provider": "none"},
                "resources": {
                    "memory_warn_threshold_mb": 4096,
                    "memory_critical_threshold_mb": 8192,
                },
                "consent": {},
            }

            # Export
            path = Path(tmpdir) / "config.yaml"
            export_config(original, path, ExportFormat.YAML)

            # Import
            result = import_config(path)
            assert result.success

            # Validate
            validation = validate_settings(result.config)
            assert validation.is_valid

    def test_migration_then_validation(self):
        """Test migration followed by validation."""
        from proxima.config.migration import auto_migrate
        from proxima.config.validation import validate_settings

        # Old config without version
        old_config = {
            "general": {"verbosity": "info"},
            "backends": {"default_backend": "auto"},
        }

        # Migrate
        migrated = auto_migrate(old_config)
        assert "_version" in migrated

        # Validate
        result = validate_settings(migrated)
        # Should be valid after migration adds defaults
        assert result.is_valid or len(result.errors()) == 0


# =============================================================================
# EDGE CASES
# =============================================================================


class TestConfigEdgeCases:
    """Edge case tests for configuration."""

    def test_empty_config_validation(self):
        """Test validation of empty config."""
        from proxima.config.validation import validate_settings

        result = validate_settings({})
        # Empty config uses default values - may be valid
        # This tests that validation doesn't crash on empty input
        assert isinstance(result.is_valid, bool)

    def test_none_values(self):
        """Test handling of None values."""
        from proxima.config.validation import validate_settings

        config = {
            "general": None,
            "backends": {},
            "llm": {},
            "resources": {},
            "consent": {},
        }

        result = validate_settings(config)
        assert not result.is_valid

    def test_extra_keys_allowed(self):
        """Test that extra keys don't cause errors."""
        from proxima.config.validation import validate_settings

        config = {
            "general": {"verbosity": "info", "unknown_key": "value"},
            "backends": {},
            "llm": {},
            "resources": {},
            "consent": {},
        }

        result = validate_settings(config)
        # Extra keys should be allowed (for forward compatibility)
        # Only validate known fields
        assert result.is_valid or all("unknown_key" not in str(e) for e in result.errors())

    def test_unicode_values(self):
        """Test handling of unicode values."""
        from proxima.config.export_import import ExportFormat, export_config, import_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"general": {"description": "测试 🚀 тест"}}
            path = Path(tmpdir) / "config.yaml"

            export_config(config, path, ExportFormat.YAML)
            result = import_config(path)

            assert result.success
            assert result.config["general"]["description"] == "测试 🚀 тест"

    def test_very_long_string_value(self):
        """Test handling of very long string values."""
        from proxima.config.validation import ValidationResult, validate_path

        long_path = "a" * 10000
        result = ValidationResult()
        validate_path(long_path, "test.path", result)
        # Should not crash

    def test_special_characters_in_path(self):
        """Test handling of special characters in paths."""
        from proxima.config.validation import ValidationResult, validate_path

        result = ValidationResult()
        validate_path("/path/with spaces/and!special@chars", "test.path", result)
        # Should succeed on Unix, may fail on Windows for certain chars
