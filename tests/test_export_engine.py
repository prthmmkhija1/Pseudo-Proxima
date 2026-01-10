"""Test Step 5.3: Export Engine - Comprehensive tests for all export formats.

Tests cover:
- JSON, CSV, XLSX, HTML, Markdown, YAML export formats
- File and stream (string/bytes) export modes
- Export options and customization
- Error handling and edge cases
- ExportEngine orchestration
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxima.data.export import (
    BaseExporter,
    CSVExporter,
    ExportEngine,
    ExportFormat,
    ExportOptions,
    ExportResult,
    HTMLExporter,
    JSONExporter,
    MarkdownExporter,
    ReportData,
    XLSXExporter,
    YAMLExporter,
    export_to_csv,
    export_to_html,
    export_to_json,
    export_to_markdown,
    export_to_string,
    export_to_xlsx,
    export_to_yaml,
)

# ===== Test Fixtures =====


@pytest.fixture
def sample_data() -> ReportData:
    """Create sample ReportData for testing."""
    return ReportData(
        title="Test Execution Report",
        summary={
            "total_backends": 3,
            "successful": 2,
            "failed": 1,
            "avg_latency_ms": 125.5,
            "best_backend": "openai",
        },
        raw_results=[
            {"backend": "openai", "latency": 120.5, "status": "success", "tokens": 150},
            {"backend": "anthropic", "latency": 130.2, "status": "success", "tokens": 145},
            {"backend": "local", "latency": None, "status": "failed", "tokens": 0},
        ],
        comparison={
            "winner": "openai",
            "metrics": {
                "latency": {"openai": 120.5, "anthropic": 130.2},
                "quality": {"openai": 0.95, "anthropic": 0.92},
            },
            "speedup_factor": 1.08,
        },
        insights=[
            "OpenAI backend shows lowest latency (120.5ms)",
            "Anthropic backend has comparable quality (92% vs 95%)",
            "Local backend failed - check configuration",
        ],
        metadata={
            "version": "1.0.0",
            "environment": "test",
            "python_version": "3.14.0",
            "execution_time_total_s": 2.5,
        },
    )


@pytest.fixture
def minimal_data() -> ReportData:
    """Create minimal ReportData for edge case testing."""
    return ReportData(title="Minimal Report")


@pytest.fixture
def data_with_custom_sections() -> ReportData:
    """Create ReportData with custom sections."""
    return ReportData(
        title="Custom Sections Test",
        summary={"test": "value"},
        custom_sections={
            "Performance Notes": {"peak_memory": "256MB", "cpu_usage": "45%"},
            "Recommendations": ["Increase timeout", "Add retry logic"],
        },
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file output tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ===== ReportData Tests =====


class TestReportData:
    """Tests for ReportData dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        data = ReportData()
        assert data.title == "Proxima Execution Report"
        assert data.generated_at > 0
        assert data.summary == {}
        assert data.raw_results == []
        assert data.insights == []

    def test_to_dict(self, sample_data: ReportData):
        """Test to_dict serialization."""
        d = sample_data.to_dict()
        assert d["title"] == sample_data.title
        assert d["summary"] == sample_data.summary
        assert d["raw_results"] == sample_data.raw_results
        assert d["comparison"] == sample_data.comparison
        assert d["insights"] == sample_data.insights
        assert d["metadata"] == sample_data.metadata

    def test_custom_title(self):
        """Test custom title is set."""
        data = ReportData(title="Custom Report Title")
        assert data.title == "Custom Report Title"


# ===== ExportResult Tests =====


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_success_result(self):
        """Test successful export result."""
        result = ExportResult(
            success=True,
            format=ExportFormat.JSON,
            output_path=Path("/test/output.json"),
            file_size_bytes=1024,
            export_time_ms=5.5,
        )
        assert result.success is True
        assert result.error is None
        assert result.warnings == []

    def test_failure_result(self):
        """Test failed export result."""
        result = ExportResult(
            success=False,
            format=ExportFormat.JSON,
            output_path=None,
            error="Test error message",
        )
        assert result.success is False
        assert result.error == "Test error message"

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = ExportResult(
            success=True,
            format=ExportFormat.JSON,
            output_path=Path("/test/output.json"),
            file_size_bytes=1024,
            export_time_ms=5.5,
            content="test content",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["format"] == "json"
        assert d["has_content"] is True


# ===== ExportOptions Tests =====


class TestExportOptions:
    """Tests for ExportOptions dataclass."""

    def test_default_options(self):
        """Test default export options."""
        options = ExportOptions()
        assert options.format == ExportFormat.JSON
        assert options.output_path is None
        assert options.include_metadata is True
        assert options.include_raw_results is True
        assert options.pretty_print is True
        assert options.stream_output is False

    def test_custom_options(self):
        """Test custom export options."""
        options = ExportOptions(
            format=ExportFormat.CSV,
            include_metadata=False,
            csv_delimiter=";",
            stream_output=True,
        )
        assert options.format == ExportFormat.CSV
        assert options.include_metadata is False
        assert options.csv_delimiter == ";"
        assert options.stream_output is True


# ===== JSON Exporter Tests =====


class TestJSONExporter:
    """Tests for JSONExporter."""

    def test_export_to_file(self, sample_data: ReportData, temp_dir: Path):
        """Test JSON export to file."""
        exporter = JSONExporter()
        output_path = temp_dir / "report.json"
        options = ExportOptions(format=ExportFormat.JSON, output_path=output_path)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.format == ExportFormat.JSON
        assert output_path.exists()
        assert result.file_size_bytes > 0

        # Verify content
        content = json.loads(output_path.read_text())
        assert content["title"] == sample_data.title
        assert "summary" in content

    def test_export_to_stream(self, sample_data: ReportData):
        """Test JSON export to string."""
        exporter = JSONExporter()
        options = ExportOptions(format=ExportFormat.JSON, stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.content is not None
        assert isinstance(result.content, str)

        content = json.loads(result.content)
        assert content["title"] == sample_data.title

    def test_export_pretty_print(self, sample_data: ReportData):
        """Test pretty print option."""
        exporter = JSONExporter()

        # With pretty print
        options_pretty = ExportOptions(stream_output=True, pretty_print=True)
        result_pretty = exporter.export(sample_data, options_pretty)

        # Without pretty print
        options_compact = ExportOptions(stream_output=True, pretty_print=False)
        result_compact = exporter.export(sample_data, options_compact)

        assert len(result_pretty.content) > len(result_compact.content)

    def test_export_exclude_sections(self, sample_data: ReportData):
        """Test excluding sections."""
        exporter = JSONExporter()
        options = ExportOptions(
            stream_output=True,
            include_raw_results=False,
            include_insights=False,
        )

        result = exporter.export(sample_data, options)
        content = json.loads(result.content)

        assert "raw_results" not in content
        assert "insights" not in content

    def test_export_without_path_fails(self, sample_data: ReportData):
        """Test export without path and stream_output=False fails."""
        exporter = JSONExporter()
        options = ExportOptions(format=ExportFormat.JSON, stream_output=False)

        result = exporter.export(sample_data, options)

        assert result.success is False
        assert "requires an output path" in result.error


# ===== CSV Exporter Tests =====


class TestCSVExporter:
    """Tests for CSVExporter."""

    def test_export_to_file(self, sample_data: ReportData, temp_dir: Path):
        """Test CSV export to file."""
        exporter = CSVExporter()
        output_path = temp_dir / "results.csv"
        options = ExportOptions(format=ExportFormat.CSV, output_path=output_path)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert output_path.exists()

        content = output_path.read_text()
        assert "backend" in content
        assert "openai" in content

    def test_export_to_stream(self, sample_data: ReportData):
        """Test CSV export to string."""
        exporter = CSVExporter()
        options = ExportOptions(format=ExportFormat.CSV, stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.content is not None
        assert "backend" in result.content
        assert "openai" in result.content

    def test_export_custom_delimiter(self, sample_data: ReportData):
        """Test custom CSV delimiter."""
        exporter = CSVExporter()
        options = ExportOptions(stream_output=True, csv_delimiter=";")

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert ";" in result.content

    def test_export_minimal_data(self, minimal_data: ReportData):
        """Test CSV export with minimal data (no raw results)."""
        exporter = CSVExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(minimal_data, options)

        assert result.success is True
        # Should export summary as key-value pairs
        assert "Key" in result.content
        assert "Value" in result.content


# ===== XLSX Exporter Tests =====


class TestXLSXExporter:
    """Tests for XLSXExporter."""

    def test_export_to_file(self, sample_data: ReportData, temp_dir: Path):
        """Test XLSX export to file."""
        exporter = XLSXExporter()
        output_path = temp_dir / "report.xlsx"
        options = ExportOptions(format=ExportFormat.XLSX, output_path=output_path)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert output_path.exists()

        # Verify sheets
        import openpyxl

        wb = openpyxl.load_workbook(output_path)
        sheet_names = wb.sheetnames

        assert "Summary" in sheet_names
        assert "Raw Results" in sheet_names
        assert "Backend Comparison" in sheet_names
        assert "Insights" in sheet_names
        assert "Metadata" in sheet_names

    def test_export_to_stream(self, sample_data: ReportData):
        """Test XLSX export to bytes."""
        exporter = XLSXExporter()
        options = ExportOptions(format=ExportFormat.XLSX, stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.content is not None
        assert isinstance(result.content, bytes)
        assert len(result.content) > 0

    def test_export_freeze_panes(self, sample_data: ReportData, temp_dir: Path):
        """Test freeze panes option."""
        exporter = XLSXExporter()
        output_path = temp_dir / "report.xlsx"
        options = ExportOptions(
            format=ExportFormat.XLSX,
            output_path=output_path,
            xlsx_freeze_panes=True,
        )

        result = exporter.export(sample_data, options)
        assert result.success is True


# ===== HTML Exporter Tests =====


class TestHTMLExporter:
    """Tests for HTMLExporter."""

    def test_export_to_file(self, sample_data: ReportData, temp_dir: Path):
        """Test HTML export to file."""
        exporter = HTMLExporter()
        output_path = temp_dir / "report.html"
        options = ExportOptions(format=ExportFormat.HTML, output_path=output_path)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert output_path.exists()

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert sample_data.title in content

    def test_export_to_stream(self, sample_data: ReportData):
        """Test HTML export to string."""
        exporter = HTMLExporter()
        options = ExportOptions(format=ExportFormat.HTML, stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.content is not None
        assert "<!DOCTYPE html>" in result.content

    def test_export_inline_styles(self, sample_data: ReportData):
        """Test inline styles option."""
        exporter = HTMLExporter()
        options = ExportOptions(stream_output=True, html_inline_styles=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert "<style>" in result.content


# ===== Markdown Exporter Tests =====


class TestMarkdownExporter:
    """Tests for MarkdownExporter."""

    def test_export_to_file(self, sample_data: ReportData, temp_dir: Path):
        """Test Markdown export to file."""
        exporter = MarkdownExporter()
        output_path = temp_dir / "report.md"
        options = ExportOptions(format=ExportFormat.MARKDOWN, output_path=output_path)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.format == ExportFormat.MARKDOWN
        assert output_path.exists()

        content = output_path.read_text()
        assert f"# {sample_data.title}" in content
        assert "## Summary" in content
        assert "## Insights" in content

    def test_export_to_stream(self, sample_data: ReportData):
        """Test Markdown export to string."""
        exporter = MarkdownExporter()
        options = ExportOptions(format=ExportFormat.MARKDOWN, stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.content is not None
        assert "# Test Execution Report" in result.content
        assert "| Metric | Value |" in result.content

    def test_export_with_toc(self, sample_data: ReportData):
        """Test table of contents option."""
        exporter = MarkdownExporter()
        options = ExportOptions(stream_output=True, markdown_toc=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert "## Table of Contents" in result.content
        assert "[Summary](#summary)" in result.content

    def test_export_without_toc(self, sample_data: ReportData):
        """Test without table of contents."""
        exporter = MarkdownExporter()
        options = ExportOptions(stream_output=True, markdown_toc=False)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert "## Table of Contents" not in result.content

    def test_export_raw_results_table(self, sample_data: ReportData):
        """Test raw results as markdown table."""
        exporter = MarkdownExporter()
        options = ExportOptions(stream_output=True, include_raw_results=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert "## Raw Results" in result.content
        assert "| backend |" in result.content
        assert "| openai |" in result.content

    def test_export_metadata_code_block(self, sample_data: ReportData):
        """Test metadata as code block."""
        exporter = MarkdownExporter()
        options = ExportOptions(stream_output=True, markdown_code_blocks=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert "```" in result.content

    def test_export_custom_sections(self, data_with_custom_sections: ReportData):
        """Test custom sections export."""
        exporter = MarkdownExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(data_with_custom_sections, options)

        assert result.success is True
        assert "## Performance Notes" in result.content
        assert "## Recommendations" in result.content


# ===== YAML Exporter Tests =====


class TestYAMLExporter:
    """Tests for YAMLExporter."""

    def test_export_to_file(self, sample_data: ReportData, temp_dir: Path):
        """Test YAML export to file."""
        exporter = YAMLExporter()
        output_path = temp_dir / "report.yaml"
        options = ExportOptions(format=ExportFormat.YAML, output_path=output_path)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.format == ExportFormat.YAML
        assert output_path.exists()

        content = output_path.read_text()
        assert "title:" in content
        assert sample_data.title in content

    def test_export_to_stream(self, sample_data: ReportData):
        """Test YAML export to string."""
        exporter = YAMLExporter()
        options = ExportOptions(format=ExportFormat.YAML, stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.content is not None
        assert "title:" in result.content
        assert "summary:" in result.content

    def test_export_includes_all_sections(self, sample_data: ReportData):
        """Test all sections are included."""
        exporter = YAMLExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert "raw_results:" in result.content
        assert "comparison:" in result.content
        assert "insights:" in result.content
        assert "metadata:" in result.content

    def test_export_exclude_sections(self, sample_data: ReportData):
        """Test excluding sections."""
        exporter = YAMLExporter()
        options = ExportOptions(
            stream_output=True,
            include_raw_results=False,
            include_insights=False,
        )

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert "raw_results:" not in result.content
        assert "insights:" not in result.content


# ===== ExportEngine Tests =====


class TestExportEngine:
    """Tests for ExportEngine orchestration."""

    def test_available_formats(self):
        """Test getting available formats."""
        engine = ExportEngine()
        formats = engine.get_available_formats()

        assert ExportFormat.JSON in formats
        assert ExportFormat.CSV in formats
        assert ExportFormat.MARKDOWN in formats
        assert ExportFormat.YAML in formats
        assert ExportFormat.HTML in formats

    def test_export_single_format(self, sample_data: ReportData, temp_dir: Path):
        """Test exporting to a single format."""
        engine = ExportEngine()
        output_path = temp_dir / "report.json"

        result = engine.export(sample_data, ExportFormat.JSON, output_path)

        assert result.success is True
        assert output_path.exists()

    def test_export_to_string(self, sample_data: ReportData):
        """Test export_to_string method."""
        engine = ExportEngine()

        result = engine.export_to_string(sample_data, ExportFormat.MARKDOWN)

        assert result.success is True
        assert result.content is not None
        assert "# Test Execution Report" in result.content

    def test_export_to_bytes(self, sample_data: ReportData):
        """Test export_to_bytes method."""
        engine = ExportEngine()

        result = engine.export_to_bytes(sample_data, ExportFormat.XLSX)

        assert result.success is True
        assert result.content is not None
        assert isinstance(result.content, bytes)

    def test_export_all(self, sample_data: ReportData, temp_dir: Path):
        """Test exporting to all formats."""
        engine = ExportEngine()

        results = engine.export_all(sample_data, temp_dir, "multi")

        for fmt, result in results.items():
            if result.success:
                output_file = temp_dir / f"multi.{fmt.value}"
                assert output_file.exists(), f"Missing {fmt.value} file"

    def test_export_with_kwargs(self, sample_data: ReportData, temp_dir: Path):
        """Test export with additional kwargs."""
        engine = ExportEngine()
        output_path = temp_dir / "report.json"

        result = engine.export(
            sample_data,
            ExportFormat.JSON,
            output_path,
            pretty_print=True,
            include_raw_results=False,
        )

        assert result.success is True
        content = json.loads(output_path.read_text())
        assert "raw_results" not in content

    def test_register_custom_exporter(self, sample_data: ReportData):
        """Test registering a custom exporter."""
        engine = ExportEngine()

        class CustomExporter(BaseExporter):
            def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
                return ExportResult(
                    success=True,
                    format=ExportFormat.JSON,
                    output_path=None,
                    content="custom output",
                )

        engine.register_exporter(ExportFormat.JSON, CustomExporter())
        result = engine.export_to_string(sample_data, ExportFormat.JSON)

        assert result.success is True
        assert result.content == "custom output"


# ===== Convenience Functions Tests =====


class TestConvenienceFunctions:
    """Tests for convenience export functions."""

    def test_export_to_json(self, sample_data: ReportData, temp_dir: Path):
        """Test export_to_json convenience function."""
        output_path = temp_dir / "test.json"
        result = export_to_json(sample_data, output_path)

        assert result.success is True
        assert output_path.exists()

    def test_export_to_csv(self, sample_data: ReportData, temp_dir: Path):
        """Test export_to_csv convenience function."""
        output_path = temp_dir / "test.csv"
        result = export_to_csv(sample_data, output_path)

        assert result.success is True
        assert output_path.exists()

    def test_export_to_xlsx(self, sample_data: ReportData, temp_dir: Path):
        """Test export_to_xlsx convenience function."""
        output_path = temp_dir / "test.xlsx"
        result = export_to_xlsx(sample_data, output_path)

        assert result.success is True
        assert output_path.exists()

    def test_export_to_html(self, sample_data: ReportData, temp_dir: Path):
        """Test export_to_html convenience function."""
        output_path = temp_dir / "test.html"
        result = export_to_html(sample_data, output_path)

        assert result.success is True
        assert output_path.exists()

    def test_export_to_markdown(self, sample_data: ReportData, temp_dir: Path):
        """Test export_to_markdown convenience function."""
        output_path = temp_dir / "test.md"
        result = export_to_markdown(sample_data, output_path)

        assert result.success is True
        assert output_path.exists()

    def test_export_to_yaml(self, sample_data: ReportData, temp_dir: Path):
        """Test export_to_yaml convenience function."""
        output_path = temp_dir / "test.yaml"
        result = export_to_yaml(sample_data, output_path)

        assert result.success is True
        assert output_path.exists()

    def test_export_to_string_func(self, sample_data: ReportData):
        """Test export_to_string convenience function."""
        content = export_to_string(sample_data, ExportFormat.JSON)

        assert content is not None
        assert sample_data.title in content

    def test_export_to_string_markdown(self, sample_data: ReportData):
        """Test export_to_string with Markdown format."""
        content = export_to_string(sample_data, ExportFormat.MARKDOWN)

        assert content is not None
        assert "# Test Execution Report" in content


# ===== Error Handling Tests =====


class TestErrorHandling:
    """Tests for error handling."""

    def test_csv_without_path_fails(self, sample_data: ReportData):
        """Test CSV export without path and stream_output=False fails."""
        exporter = CSVExporter()
        options = ExportOptions(format=ExportFormat.CSV, stream_output=False)

        result = exporter.export(sample_data, options)

        assert result.success is False
        assert "requires an output path" in result.error

    def test_xlsx_without_path_fails(self, sample_data: ReportData):
        """Test XLSX export without path and stream_output=False fails."""
        exporter = XLSXExporter()
        options = ExportOptions(format=ExportFormat.XLSX, stream_output=False)

        result = exporter.export(sample_data, options)

        assert result.success is False
        assert "requires an output path" in result.error

    def test_html_without_path_fails(self, sample_data: ReportData):
        """Test HTML export without path and stream_output=False fails."""
        exporter = HTMLExporter()
        options = ExportOptions(format=ExportFormat.HTML, stream_output=False)

        result = exporter.export(sample_data, options)

        assert result.success is False
        assert "requires an output path" in result.error

    def test_markdown_without_path_fails(self, sample_data: ReportData):
        """Test Markdown export without path and stream_output=False fails."""
        exporter = MarkdownExporter()
        options = ExportOptions(format=ExportFormat.MARKDOWN, stream_output=False)

        result = exporter.export(sample_data, options)

        assert result.success is False
        assert "requires an output path" in result.error

    def test_yaml_without_path_fails(self, sample_data: ReportData):
        """Test YAML export without path and stream_output=False fails."""
        exporter = YAMLExporter()
        options = ExportOptions(format=ExportFormat.YAML, stream_output=False)

        result = exporter.export(sample_data, options)

        assert result.success is False
        assert "requires an output path" in result.error

    def test_export_time_recorded(self, sample_data: ReportData):
        """Test export time is recorded."""
        exporter = JSONExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        assert result.export_time_ms > 0


# ===== Edge Case Tests =====


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_summary(self, minimal_data: ReportData):
        """Test export with empty summary."""
        exporter = JSONExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(minimal_data, options)

        assert result.success is True
        content = json.loads(result.content)
        assert content["summary"] == {}

    def test_special_characters_in_title(self):
        """Test export with special characters in title."""
        data = ReportData(title='Test "Report" with <special> & characters')
        exporter = HTMLExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(data, options)

        assert result.success is True

    def test_unicode_content(self):
        """Test export with unicode content."""
        data = ReportData(
            title="Unicode Test 日本語 العربية 中文",
            insights=["Emoji test: 🚀 📊 ✅"],
        )
        exporter = JSONExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(data, options)

        assert result.success is True
        assert "日本語" in result.content
        assert "🚀" in result.content

    def test_large_raw_results(self):
        """Test export with large raw results."""
        data = ReportData(
            title="Large Data Test",
            raw_results=[{"id": i, "value": i * 1.5} for i in range(1000)],
        )
        exporter = CSVExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(data, options)

        assert result.success is True
        # Should have 1001 lines (header + 1000 rows)
        assert result.content.count("\n") >= 1000

    def test_nested_comparison_data(self, sample_data: ReportData):
        """Test export with deeply nested comparison data."""
        exporter = MarkdownExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        # Nested data should be flattened
        assert "metrics.latency.openai" in result.content or "winner" in result.content

    def test_null_values_in_results(self, sample_data: ReportData):
        """Test export handles null/None values."""
        exporter = MarkdownExporter()
        options = ExportOptions(stream_output=True)

        result = exporter.export(sample_data, options)

        assert result.success is True
        # None value in local backend should be handled
        assert "N/A" in result.content or "None" in result.content


# ===== Main Entry Point =====


def main():
    """Run tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
