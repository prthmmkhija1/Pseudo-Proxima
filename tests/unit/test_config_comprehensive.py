"""
Comprehensive Unit Tests for Configuration System

Tests for:
- Settings models (GeneralSettings, BackendsSettings, etc.)
- ConfigService functionality
- YAML loading and merging
- Environment variable parsing
- Config priority (CLI > ENV > User > Project > Default)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from proxima.config.settings import (
    BackendsSettings,
    ConfigService,
    ConsentSettings,
    GeneralSettings,
    LLMSettings,
    ResourcesSettings,
    Settings,
    _deep_merge,
    _get_nested,
    _load_yaml,
    _parse_scalar,
    _set_nested,
)


# =============================================================================
# GENERAL SETTINGS TESTS
# =============================================================================


class TestGeneralSettings:
    """Tests for GeneralSettings model."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values for GeneralSettings."""
        settings = GeneralSettings()
        assert settings.verbosity == "info"
        assert settings.output_format == "text"
        assert settings.color_enabled is True
        assert settings.storage_backend == "sqlite"

    @pytest.mark.unit
    def test_custom_values(self):
        """Test custom values for GeneralSettings."""
        settings = GeneralSettings(
            verbosity="debug",
            output_format="json",
            color_enabled=False,
            data_dir="/custom/path",
        )
        assert settings.verbosity == "debug"
        assert settings.output_format == "json"
        assert settings.color_enabled is False
        assert settings.data_dir == "/custom/path"


# =============================================================================
# BACKENDS SETTINGS TESTS
# =============================================================================


class TestBackendsSettings:
    """Tests for BackendsSettings model."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values for BackendsSettings."""
        settings = BackendsSettings()
        assert settings.default_backend == "auto"
        assert settings.parallel_execution is False
        assert settings.timeout_seconds == 300

    @pytest.mark.unit
    def test_custom_values(self):
        """Test custom values for BackendsSettings."""
        settings = BackendsSettings(
            default_backend="cirq",
            parallel_execution=True,
            timeout_seconds=600,
        )
        assert settings.default_backend == "cirq"
        assert settings.parallel_execution is True
        assert settings.timeout_seconds == 600


# =============================================================================
# LLM SETTINGS TESTS
# =============================================================================


class TestLLMSettings:
    """Tests for LLMSettings model."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values for LLMSettings."""
        settings = LLMSettings()
        assert settings.provider == "none"
        assert settings.model == ""
        assert settings.require_consent is True

    @pytest.mark.unit
    def test_openai_config(self):
        """Test OpenAI configuration."""
        settings = LLMSettings(
            provider="openai",
            model="gpt-4",
            api_key_env_var="OPENAI_API_KEY",
            require_consent=False,
        )
        assert settings.provider == "openai"
        assert settings.model == "gpt-4"
        assert settings.require_consent is False

    @pytest.mark.unit
    def test_local_llm_config(self):
        """Test local LLM configuration."""
        settings = LLMSettings(
            provider="ollama",
            model="llama3",
            local_endpoint="http://localhost:11434",
        )
        assert settings.provider == "ollama"
        assert "localhost" in settings.local_endpoint


# =============================================================================
# RESOURCES SETTINGS TESTS
# =============================================================================


class TestResourcesSettings:
    """Tests for ResourcesSettings model."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values for ResourcesSettings."""
        settings = ResourcesSettings()
        assert settings.memory_warn_threshold_mb == 4096
        assert settings.memory_critical_threshold_mb == 8192
        assert settings.max_execution_time_seconds == 3600

    @pytest.mark.unit
    def test_custom_thresholds(self):
        """Test custom memory thresholds."""
        settings = ResourcesSettings(
            memory_warn_threshold_mb=2048,
            memory_critical_threshold_mb=4096,
            max_execution_time_seconds=7200,
        )
        assert settings.memory_warn_threshold_mb == 2048
        assert settings.memory_critical_threshold_mb == 4096


# =============================================================================
# CONSENT SETTINGS TESTS
# =============================================================================


class TestConsentSettings:
    """Tests for ConsentSettings model."""

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values (secure by default)."""
        settings = ConsentSettings()
        assert settings.auto_approve_local_llm is False
        assert settings.auto_approve_remote_llm is False
        assert settings.remember_decisions is False

    @pytest.mark.unit
    def test_auto_approve_local_only(self):
        """Test auto-approve for local LLM only."""
        settings = ConsentSettings(
            auto_approve_local_llm=True,
            auto_approve_remote_llm=False,
        )
        assert settings.auto_approve_local_llm is True
        assert settings.auto_approve_remote_llm is False


# =============================================================================
# SETTINGS MODEL TESTS
# =============================================================================


class TestSettings:
    """Tests for main Settings model."""

    @pytest.mark.unit
    def test_from_dict(self):
        """Test creating Settings from dictionary."""
        data = {
            "general": {"verbosity": "debug"},
            "backends": {"default_backend": "qiskit-aer"},
            "llm": {"provider": "anthropic"},
            "resources": {"memory_warn_threshold_mb": 8192},
            "consent": {"remember_decisions": True},
        }
        settings = Settings.from_dict(data)
        
        assert settings.general.verbosity == "debug"
        assert settings.backends.default_backend == "qiskit-aer"
        assert settings.llm.provider == "anthropic"

    @pytest.mark.unit
    def test_from_dict_with_defaults(self):
        """Test from_dict fills in defaults."""
        data = {
            "general": {},
            "backends": {},
            "llm": {},
            "resources": {},
            "consent": {},
        }
        settings = Settings.from_dict(data)
        
        # Should have default values
        assert settings.general.verbosity == "info"
        assert settings.backends.timeout_seconds == 300


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestDeepMerge:
    """Tests for _deep_merge function."""

    @pytest.mark.unit
    def test_merge_flat_dicts(self):
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        
        assert result == {"a": 1, "b": 3, "c": 4}
        # Original should be unchanged
        assert base == {"a": 1, "b": 2}

    @pytest.mark.unit
    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"outer": {"inner1": 1, "inner2": 2}}
        override = {"outer": {"inner2": 3, "inner3": 4}}
        result = _deep_merge(base, override)
        
        assert result == {"outer": {"inner1": 1, "inner2": 3, "inner3": 4}}

    @pytest.mark.unit
    def test_merge_override_with_non_dict(self):
        """Test overriding dict with non-dict value."""
        base = {"key": {"nested": "value"}}
        override = {"key": "simple_value"}
        result = _deep_merge(base, override)
        
        assert result == {"key": "simple_value"}

    @pytest.mark.unit
    def test_merge_empty_override(self):
        """Test merging with empty override."""
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {})
        
        assert result == {"a": 1, "b": 2}


class TestParseScalar:
    """Tests for _parse_scalar function."""

    @pytest.mark.unit
    def test_parse_boolean_true(self):
        """Test parsing boolean true values."""
        assert _parse_scalar("true") is True
        assert _parse_scalar("True") is True
        assert _parse_scalar("TRUE") is True

    @pytest.mark.unit
    def test_parse_boolean_false(self):
        """Test parsing boolean false values."""
        assert _parse_scalar("false") is False
        assert _parse_scalar("False") is False
        assert _parse_scalar("FALSE") is False

    @pytest.mark.unit
    def test_parse_none(self):
        """Test parsing null/none values."""
        assert _parse_scalar("null") is None
        assert _parse_scalar("none") is None
        assert _parse_scalar("None") is None

    @pytest.mark.unit
    def test_parse_number(self):
        """Test parsing numeric values."""
        assert _parse_scalar("42") == 42
        assert _parse_scalar("3.14") == 3.14
        assert _parse_scalar("-10") == -10

    @pytest.mark.unit
    def test_parse_string(self):
        """Test parsing string values."""
        assert _parse_scalar("hello") == "hello"
        assert _parse_scalar("  spaced  ") == "spaced"

    @pytest.mark.unit
    def test_parse_json_array(self):
        """Test parsing JSON array."""
        assert _parse_scalar("[1, 2, 3]") == [1, 2, 3]

    @pytest.mark.unit
    def test_parse_json_object(self):
        """Test parsing JSON object."""
        assert _parse_scalar('{"key": "value"}') == {"key": "value"}


class TestSetNested:
    """Tests for _set_nested function."""

    @pytest.mark.unit
    def test_set_single_level(self):
        """Test setting single-level value."""
        target = {}
        _set_nested(target, ["key"], "value")
        assert target == {"key": "value"}

    @pytest.mark.unit
    def test_set_nested_value(self):
        """Test setting nested value."""
        target = {}
        _set_nested(target, ["outer", "inner", "deep"], 42)
        assert target == {"outer": {"inner": {"deep": 42}}}

    @pytest.mark.unit
    def test_set_overwrites_existing(self):
        """Test overwriting existing value."""
        target = {"key": "old"}
        _set_nested(target, ["key"], "new")
        assert target == {"key": "new"}


class TestGetNested:
    """Tests for _get_nested function."""

    @pytest.mark.unit
    def test_get_single_level(self):
        """Test getting single-level value."""
        data = {"key": "value"}
        assert _get_nested(data, ["key"]) == "value"

    @pytest.mark.unit
    def test_get_nested_value(self):
        """Test getting nested value."""
        data = {"outer": {"inner": {"deep": 42}}}
        assert _get_nested(data, ["outer", "inner", "deep"]) == 42

    @pytest.mark.unit
    def test_get_missing_key(self):
        """Test getting missing key raises KeyError."""
        data = {"key": "value"}
        with pytest.raises(KeyError):
            _get_nested(data, ["missing"])

    @pytest.mark.unit
    def test_get_missing_nested_key(self):
        """Test getting missing nested key raises KeyError."""
        data = {"outer": {"inner": "value"}}
        with pytest.raises(KeyError):
            _get_nested(data, ["outer", "missing"])


# =============================================================================
# YAML LOADING TESTS
# =============================================================================


class TestLoadYaml:
    """Tests for _load_yaml function."""

    @pytest.mark.unit
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file returns empty dict."""
        result = _load_yaml(Path("/nonexistent/path/config.yaml"))
        assert result == {}

    @pytest.mark.unit
    def test_load_valid_yaml(self):
        """Test loading valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"key": "value", "number": 42}, f)
            f.flush()
            
            result = _load_yaml(Path(f.name))
            assert result == {"key": "value", "number": 42}
            
            os.unlink(f.name)

    @pytest.mark.unit
    def test_load_empty_yaml(self):
        """Test loading empty YAML file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            
            result = _load_yaml(Path(f.name))
            assert result == {}
            
            os.unlink(f.name)


# =============================================================================
# CONFIG SERVICE TESTS
# =============================================================================


class TestConfigService:
    """Tests for ConfigService class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.mark.unit
    def test_service_creation(self):
        """Test ConfigService can be created."""
        service = ConfigService()
        assert service.env_prefix == "PROXIMA"

    @pytest.mark.unit
    def test_service_custom_prefix(self):
        """Test ConfigService with custom prefix."""
        service = ConfigService(env_prefix="CUSTOM")
        assert service.env_prefix == "CUSTOM"

    @pytest.mark.unit
    def test_load_returns_settings(self):
        """Test load returns Settings instance."""
        service = ConfigService()
        settings = service.load()
        assert isinstance(settings, Settings)

    @pytest.mark.unit
    def test_load_with_cli_overrides(self):
        """Test load respects CLI overrides."""
        service = ConfigService()
        settings = service.load(cli_overrides={
            "general": {"verbosity": "debug"}
        })
        assert settings.general.verbosity == "debug"

    @pytest.mark.unit
    def test_normalize_key_path(self):
        """Test key path normalization."""
        service = ConfigService()
        
        assert service._normalize_key_path("general.verbosity") == ["general", "verbosity"]
        assert service._normalize_key_path("backends.default_backend") == ["backends", "default_backend"]

    @pytest.mark.unit
    def test_normalize_key_path_empty(self):
        """Test empty key path raises error."""
        service = ConfigService()
        with pytest.raises(ValueError):
            service._normalize_key_path("")

    @pytest.mark.unit
    def test_normalize_env_key(self):
        """Test environment key normalization."""
        service = ConfigService()
        
        # Double underscore for nesting
        assert service._normalize_env_key("GENERAL__VERBOSITY") == ["general", "verbosity"]
        
        # Single underscore
        assert service._normalize_env_key("VERBOSITY") == ["verbosity"]

    @pytest.mark.unit
    def test_env_overrides(self):
        """Test environment variable overrides."""
        service = ConfigService()
        
        with patch.dict(os.environ, {"PROXIMA_GENERAL__VERBOSITY": "debug"}):
            overrides = service._env_overrides()
            assert overrides["general"]["verbosity"] == "debug"

    @pytest.mark.unit
    def test_get_value(self):
        """Test getting a config value."""
        service = ConfigService()
        value = service.get_value("general.verbosity")
        assert value in ["info", "debug", "warning", "error"]


# =============================================================================
# CONFIG PRIORITY TESTS
# =============================================================================


class TestConfigPriority:
    """Tests for configuration priority order."""

    @pytest.mark.unit
    def test_cli_overrides_env(self):
        """Test CLI overrides take priority over environment."""
        service = ConfigService()
        
        with patch.dict(os.environ, {"PROXIMA_GENERAL__VERBOSITY": "warning"}):
            settings = service.load(cli_overrides={
                "general": {"verbosity": "error"}
            })
            assert settings.general.verbosity == "error"

    @pytest.mark.unit
    def test_env_applies_when_no_cli(self):
        """Test environment applies when no CLI override."""
        service = ConfigService()
        
        with patch.dict(os.environ, {"PROXIMA_GENERAL__OUTPUT_FORMAT": "json"}):
            settings = service.load()
            assert settings.general.output_format == "json"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestConfigEdgeCases:
    """Tests for edge cases in configuration."""

    @pytest.mark.unit
    def test_deeply_nested_override(self):
        """Test deeply nested configuration override."""
        base = {"a": {"b": {"c": {"d": 1}}}}
        override = {"a": {"b": {"c": {"d": 2, "e": 3}}}}
        result = _deep_merge(base, override)
        
        assert result["a"]["b"]["c"]["d"] == 2
        assert result["a"]["b"]["c"]["e"] == 3

    @pytest.mark.unit
    def test_special_characters_in_values(self):
        """Test config values with special characters."""
        settings = GeneralSettings(
            data_dir="/path/with spaces/and:colons",
        )
        assert "spaces" in settings.data_dir
        assert "colons" in settings.data_dir

    @pytest.mark.unit
    def test_unicode_in_config(self):
        """Test unicode in configuration values."""
        settings = GeneralSettings(
            data_dir="/путь/路径/パス",
        )
        assert "путь" in settings.data_dir
