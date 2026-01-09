"""Step 6.2: Unit Tests - Testing individual functions.

Unit tests form 60% of the test pyramid:
- Test isolated functions/methods
- Use mocks for dependencies
- Fast execution (< 1 second per test)
- No external I/O (network, disk)
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Unit tests for configuration handling."""
    
    @pytest.mark.unit
    def test_sample_config_structure(self, sample_config):
        """Test sample_config fixture has expected structure."""
        assert "version" in sample_config
        assert "backends" in sample_config
        assert "execution" in sample_config
        assert "export" in sample_config
    
    @pytest.mark.unit
    def test_config_nested_values(self, sample_config):
        """Test nested config values are accessible."""
        assert sample_config["execution"]["parallel"] is False
        assert sample_config["backends"]["timeout_s"] == 300
    
    @pytest.mark.unit
    def test_config_to_json(self, sample_config):
        """Test config serialization to JSON."""
        json_str = json.dumps(sample_config)
        parsed = json.loads(json_str)
        assert parsed == sample_config


# =============================================================================
# EXPORT ENGINE TESTS
# =============================================================================

class TestExportEngine:
    """Unit tests for export engine functionality."""
    
    @pytest.mark.unit
    def test_export_format_enum(self):
        """Test ExportFormat enum values."""
        from proxima.data.export import ExportFormat
        
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.CSV.value == "csv"
    
    @pytest.mark.unit
    def test_export_options_defaults(self):
        """Test ExportOptions default values."""
        from proxima.data.export import ExportOptions, ExportFormat
        
        options = ExportOptions(format=ExportFormat.JSON)
        
        assert options.format == ExportFormat.JSON
        assert options.pretty_print is True
    
    @pytest.mark.unit
    def test_report_data_creation(self):
        """Test ReportData creation with all fields."""
        from proxima.data.export import ReportData
        
        data = ReportData(
            title="Test Report",
            summary={"key": "value"},
            raw_results=[{"id": 1}],
            comparison={"metric": 0.5},
            insights=["insight1"],
            metadata={"version": "1.0"},
        )
        
        assert data.title == "Test Report"
        assert data.summary == {"key": "value"}
        assert len(data.raw_results) == 1
    
    @pytest.mark.unit
    def test_report_data_to_dict(self):
        """Test ReportData serialization."""
        from proxima.data.export import ReportData
        
        data = ReportData(title="Test")
        result = data.to_dict()
        
        assert "title" in result
        assert result["title"] == "Test"
    
    @pytest.mark.unit
    def test_export_result_success(self):
        """Test ExportResult for successful export."""
        from proxima.data.export import ExportResult, ExportFormat
        
        result = ExportResult(
            success=True,
            format=ExportFormat.JSON,
            output_path=Path("/tmp/test.json"),
            file_size_bytes=1024,
        )
        
        assert result.success is True
        assert result.error is None
    
    @pytest.mark.unit
    def test_export_result_failure(self):
        """Test ExportResult for failed export."""
        from proxima.data.export import ExportResult, ExportFormat
        
        result = ExportResult(
            success=False,
            format=ExportFormat.JSON,
            output_path=None,
            error="File not found",
        )
        
        assert result.success is False
        assert result.error == "File not found"
    
    @pytest.mark.unit
    def test_json_export(self, temp_dir):
        """Test JSON export function."""
        from proxima.data.export import ReportData, export_to_json
        
        data = ReportData(title="JSON Test")
        output_path = temp_dir / "test.json"
        
        result = export_to_json(data, output_path)
        
        assert result.success is True
        assert output_path.exists()
    
    @pytest.mark.unit
    def test_csv_export(self, temp_dir):
        """Test CSV export function."""
        from proxima.data.export import ReportData, export_to_csv
        
        data = ReportData(
            title="CSV Test",
            raw_results=[
                {"id": "1", "status": "success"},
                {"id": "2", "status": "failed"},
            ]
        )
        output_path = temp_dir / "test.csv"
        
        result = export_to_csv(data, output_path)
        
        assert result.success is True
        assert output_path.exists()


# =============================================================================
# COMPARISON ENGINE TESTS
# =============================================================================

class TestComparisonEngine:
    """Unit tests for comparison engine functionality."""
    
    @pytest.mark.unit
    def test_comparison_status_enum(self):
        """Test ComparisonStatus enum values."""
        from proxima.data.compare import ComparisonStatus
        
        # Uses auto() so values are integers
        assert ComparisonStatus.PENDING.value >= 1
        assert ComparisonStatus.COMPLETED.value >= 1
        assert ComparisonStatus.FAILED.value >= 1
    
    @pytest.mark.unit
    def test_execution_strategy_enum(self):
        """Test ExecutionStrategy enum values."""
        from proxima.data.compare import ExecutionStrategy
        
        assert ExecutionStrategy.PARALLEL.value == "parallel"
        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"
    
    @pytest.mark.unit
    def test_backend_result_creation(self):
        """Test BackendResult dataclass."""
        from proxima.data.compare import BackendResult
        
        result = BackendResult(
            backend_name="test",
            success=True,
            execution_time_ms=100.0,
            memory_peak_mb=256.0,
        )
        
        assert result.backend_name == "test"
        assert result.success is True
        assert result.execution_time_ms == 100.0


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Unit tests for helper functions."""
    
    @pytest.mark.unit
    def test_create_temp_config(self, temp_dir, sample_config, test_helper):
        """Test temporary config file creation."""
        config_path = test_helper.create_temp_config(temp_dir, sample_config)
        
        assert config_path.exists()
        content = json.loads(config_path.read_text())
        assert content == sample_config
    
    @pytest.mark.unit
    def test_create_temp_agent_file(self, temp_dir, test_helper):
        """Test temporary agent file creation."""
        content = "# Test Agent\n\nSome content"
        agent_path = test_helper.create_temp_agent_file(temp_dir, content)
        
        assert agent_path.exists()
        assert agent_path.read_text() == content
    
    @pytest.mark.unit
    def test_wait_for_condition_success(self, test_helper):
        """Test wait_for_condition with immediate success."""
        result = test_helper.wait_for_condition(lambda: True, timeout=1.0)
        assert result is True
    
    @pytest.mark.unit
    def test_wait_for_condition_timeout(self, test_helper):
        """Test wait_for_condition timeout."""
        result = test_helper.wait_for_condition(lambda: False, timeout=0.1)
        assert result is False
    
    @pytest.mark.unit
    def test_assert_dict_subset_pass(self, test_helper):
        """Test assert_dict_subset with matching subset."""
        full = {"a": 1, "b": 2, "c": 3}
        subset = {"a": 1, "b": 2}
        
        # Should not raise
        test_helper.assert_dict_subset(subset, full)
    
    @pytest.mark.unit
    def test_assert_dict_subset_fail(self, test_helper):
        """Test assert_dict_subset with non-matching subset."""
        full = {"a": 1, "b": 2}
        subset = {"a": 1, "c": 3}
        
        with pytest.raises(AssertionError):
            test_helper.assert_dict_subset(subset, full)


# =============================================================================
# MOCK TESTS
# =============================================================================

class TestMocks:
    """Unit tests for mock fixtures."""
    
    @pytest.mark.unit
    def test_mock_backend_properties(self, mock_backend):
        """Test mock backend has expected properties."""
        assert hasattr(mock_backend, "execute")
        assert hasattr(mock_backend, "name")
        assert mock_backend.name == "mock_backend"
    
    @pytest.mark.unit
    def test_mock_config_manager(self, mock_config_manager):
        """Test mock config manager fixture."""
        mock_config_manager.load.return_value = {"test": True}
        
        result = mock_config_manager.load()
        assert result == {"test": True}
    
    @pytest.mark.unit
    def test_mock_consent_manager(self, mock_consent_manager):
        """Test mock consent manager fixture."""
        mock_consent_manager.has_consent.return_value = True
        
        result = mock_consent_manager.has_consent()
        assert result is True


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Unit tests for data validation."""
    
    @pytest.mark.unit
    def test_circuit_data_structure(self, sample_circuit_data):
        """Test sample circuit data has expected fields."""
        assert "name" in sample_circuit_data
        assert "num_qubits" in sample_circuit_data
        assert "gates" in sample_circuit_data
        assert sample_circuit_data["num_qubits"] == 2
    
    @pytest.mark.unit
    def test_execution_result_structure(self, sample_execution_result):
        """Test sample execution result has expected fields."""
        assert "status" in sample_execution_result
        assert "counts" in sample_execution_result
        assert "duration_ms" in sample_execution_result
    
    @pytest.mark.unit
    def test_mock_report_data_structure(self, mock_report_data):
        """Test mock report data has expected fields."""
        assert mock_report_data.title is not None
        assert mock_report_data.summary is not None
        assert mock_report_data.raw_results is not None


# =============================================================================
# TIMING TESTS
# =============================================================================

class TestTiming:
    """Unit tests for timing utilities."""
    
    @pytest.mark.unit
    def test_timing_context_manager(self, timing):
        """Test timing context manager."""
        with timing() as timer:
            time.sleep(0.01)  # 10ms
        
        assert timer.elapsed_ms >= 10.0
    
    @pytest.mark.unit
    def test_timing_multiple_uses(self, timing):
        """Test timing can be reused."""
        with timing() as t1:
            pass
        
        with timing() as t2:
            time.sleep(0.005)
        
        assert t2.elapsed_ms >= t1.elapsed_ms


# =============================================================================
# PATH FIXTURE TESTS
# =============================================================================

class TestPathFixtures:
    """Unit tests for path fixtures."""
    
    @pytest.mark.unit
    def test_project_root_exists(self, project_root):
        """Test project root path exists."""
        assert project_root.exists()
        assert project_root.is_dir()
    
    @pytest.mark.unit
    def test_src_path_exists(self, src_path):
        """Test src path exists."""
        assert src_path.exists()
        assert src_path.is_dir()
    
    @pytest.mark.unit
    def test_temp_dir_is_clean(self, temp_dir):
        """Test temp_dir is empty."""
        files = list(temp_dir.iterdir())
        assert len(files) == 0
    
    @pytest.mark.unit
    def test_temp_file_path_valid(self, temp_file):
        """Test temp_file is a valid file path."""
        assert temp_file.parent.exists()
