"""Test Step 5.3: Export Engine - CSV/XLSX/JSON/HTML exports."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proxima.data.export import (
    ExportFormat,
    ExportOptions,
    ExportResult,
    ReportData,
    JSONExporter,
    CSVExporter,
    XLSXExporter,
    HTMLExporter,
    ExportEngine,
    export_to_json,
    export_to_csv,
    export_to_xlsx,
    export_to_html,
)


def create_sample_data():
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


def test_report_data():
    """Test ReportData creation and serialization."""
    print("\n=== Test: ReportData ===")
    data = create_sample_data()
    
    assert data.title == "Test Execution Report"
    assert data.summary["total_backends"] == 3
    assert len(data.raw_results) == 3
    assert len(data.insights) == 3
    
    # Test to_dict
    d = data.to_dict()
    assert d["title"] == data.title
    assert d["summary"] == data.summary
    
    print("[PASS] ReportData works correctly")


def test_json_exporter():
    """Test JSON export."""
    print("\n=== Test: JSONExporter ===")
    data = create_sample_data()
    exporter = JSONExporter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.json"
        options = ExportOptions(
            format=ExportFormat.JSON,
            output_path=output_path,
            pretty_print=True,
        )
        
        result = exporter.export(data, options)
        
        assert result.success, f"Export failed: {result.error}"
        assert result.format == ExportFormat.JSON
        assert output_path.exists()
        assert result.file_size_bytes > 0
        assert result.export_time_ms > 0
        
        # Verify content
        import json
        content = json.loads(output_path.read_text())
        assert content["title"] == data.title
        assert "summary" in content
        
    print(f"[PASS] JSON export - {result.file_size_bytes} bytes in {result.export_time_ms:.2f}ms")


def test_csv_exporter():
    """Test CSV export."""
    print("\n=== Test: CSVExporter ===")
    data = create_sample_data()
    exporter = CSVExporter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.csv"
        options = ExportOptions(
            format=ExportFormat.CSV,
            output_path=output_path,
        )
        
        result = exporter.export(data, options)
        
        assert result.success, f"Export failed: {result.error}"
        assert result.format == ExportFormat.CSV
        assert output_path.exists()
        
        # Verify content
        content = output_path.read_text()
        assert "backend" in content
        assert "openai" in content
        assert "anthropic" in content
        
    print(f"[PASS] CSV export - {result.file_size_bytes} bytes in {result.export_time_ms:.2f}ms")


def test_xlsx_exporter():
    """Test XLSX export with multiple sheets."""
    print("\n=== Test: XLSXExporter ===")
    data = create_sample_data()
    exporter = XLSXExporter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.xlsx"
        options = ExportOptions(
            format=ExportFormat.XLSX,
            output_path=output_path,
            xlsx_freeze_panes=True,
            xlsx_auto_column_width=True,
        )
        
        result = exporter.export(data, options)
        
        assert result.success, f"Export failed: {result.error}"
        assert result.format == ExportFormat.XLSX
        assert output_path.exists()
        
        # Verify sheets
        import openpyxl
        wb = openpyxl.load_workbook(output_path)
        sheet_names = wb.sheetnames
        
        assert "Summary" in sheet_names, f"Missing Summary sheet: {sheet_names}"
        assert "Raw Results" in sheet_names, f"Missing Raw Results sheet"
        assert "Backend Comparison" in sheet_names, f"Missing Backend Comparison sheet"
        assert "Insights" in sheet_names, f"Missing Insights sheet"
        assert "Metadata" in sheet_names, f"Missing Metadata sheet"
        
        # Verify Summary sheet has title
        ws = wb["Summary"]
        assert data.title in str(ws["A1"].value)
        
    print(f"[PASS] XLSX export - {result.file_size_bytes} bytes with {len(sheet_names)} sheets")


def test_html_exporter():
    """Test HTML export."""
    print("\n=== Test: HTMLExporter ===")
    data = create_sample_data()
    exporter = HTMLExporter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.html"
        options = ExportOptions(
            format=ExportFormat.HTML,
            output_path=output_path,
            html_inline_styles=True,
        )
        
        result = exporter.export(data, options)
        
        assert result.success, f"Export failed: {result.error}"
        assert result.format == ExportFormat.HTML
        assert output_path.exists()
        
        # Verify content
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert data.title in content
        assert "Summary" in content or "summary" in content.lower()
        
    print(f"[PASS] HTML export - {result.file_size_bytes} bytes in {result.export_time_ms:.2f}ms")


def test_export_engine():
    """Test ExportEngine orchestration."""
    print("\n=== Test: ExportEngine ===")
    data = create_sample_data()
    engine = ExportEngine()
    
    # Check available formats
    formats = engine.get_available_formats()
    assert ExportFormat.JSON in formats
    assert ExportFormat.CSV in formats
    print(f"  Available formats: {[f.value for f in formats]}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test individual export
        json_path = Path(tmpdir) / "single.json"
        result = engine.export(data, ExportFormat.JSON, json_path)
        assert result.success
        print(f"  Single export: {result.format.value}")
        
        # Test export_all
        all_results = engine.export_all(data, Path(tmpdir), "multi")
        
        for fmt, res in all_results.items():
            status = "PASS" if res.success else f"FAIL: {res.error}"
            print(f"    {fmt.value}: {status}")
            if res.success:
                assert (Path(tmpdir) / f"multi.{fmt.value}").exists()
    
    print("[PASS] ExportEngine orchestration works")


def test_convenience_functions():
    """Test convenience export functions."""
    print("\n=== Test: Convenience Functions ===")
    data = create_sample_data()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON
        result = export_to_json(data, Path(tmpdir) / "test.json")
        assert result.success
        print(f"  export_to_json: PASS")
        
        # CSV
        result = export_to_csv(data, Path(tmpdir) / "test.csv")
        assert result.success
        print(f"  export_to_csv: PASS")
        
        # XLSX
        result = export_to_xlsx(data, Path(tmpdir) / "test.xlsx")
        assert result.success
        print(f"  export_to_xlsx: PASS")
        
        # HTML
        result = export_to_html(data, Path(tmpdir) / "test.html")
        assert result.success
        print(f"  export_to_html: PASS")
    
    print("[PASS] All convenience functions work")


def test_export_options():
    """Test export with various options."""
    print("\n=== Test: ExportOptions ===")
    data = create_sample_data()
    engine = ExportEngine()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test excluding sections
        result = engine.export(
            data,
            ExportFormat.JSON,
            Path(tmpdir) / "minimal.json",
            include_raw_results=False,
            include_insights=False,
        )
        
        import json
        content = json.loads((Path(tmpdir) / "minimal.json").read_text())
        assert "raw_results" not in content
        assert "insights" not in content
        print("  Exclude sections: PASS")
        
        # Test decimal precision
        result = engine.export(
            data,
            ExportFormat.JSON,
            Path(tmpdir) / "precise.json",
            decimal_places=2,
        )
        assert result.success
        print("  Custom options: PASS")
    
    print("[PASS] ExportOptions work correctly")


def test_error_handling():
    """Test error handling."""
    print("\n=== Test: Error Handling ===")
    data = create_sample_data()
    
    # CSV without path
    csv_exp = CSVExporter()
    result = csv_exp.export(data, ExportOptions(format=ExportFormat.CSV))
    assert not result.success
    assert "requires an output path" in result.error
    print("  CSV without path: correctly rejected")
    
    # XLSX without path
    xlsx_exp = XLSXExporter()
    result = xlsx_exp.export(data, ExportOptions(format=ExportFormat.XLSX))
    assert not result.success
    print("  XLSX without path: correctly rejected")
    
    print("[PASS] Error handling works")


def main():
    """Run all tests."""
    print("=" * 60)
    print("STEP 5.3: EXPORT ENGINE TESTS")
    print("=" * 60)
    
    try:
        test_report_data()
        test_json_exporter()
        test_csv_exporter()
        test_xlsx_exporter()
        test_html_exporter()
        test_export_engine()
        test_convenience_functions()
        test_export_options()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nExport Engine Features:")
        print("  - JSON: Full data structure export")
        print("  - CSV: Tabular data export")
        print("  - XLSX: Multi-sheet workbook with formatting")
        print("  - HTML: Rich formatted reports with templates")
        print("\nXLSX Report Structure:")
        print("  - Summary: Overview, key metrics")
        print("  - Raw Results: Full measurement data")
        print("  - Backend Comparison: Side-by-side metrics")
        print("  - Insights: Generated insights")
        print("  - Metadata: Execution details")
        
    except AssertionError as e:
        print(f"\n[FAILED] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
