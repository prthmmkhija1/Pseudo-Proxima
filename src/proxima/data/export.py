"""Step 5.3: Export Engine - Export results in multiple formats.

Export Formats:
| Format | Library       | Features                    |
| CSV    | csv (stdlib)  | Simple tabular data         |
| XLSX   | openpyxl      | Multiple sheets, formatting |
| JSON   | json (stdlib) | Full data structure         |
| HTML   | jinja2        | Rich formatted reports      |

Report Structure (XLSX):
Workbook:
- Sheet: Summary      - Overview, key metrics
- Sheet: Raw Results  - Full measurement data
- Sheet: Backend Comparison - Side-by-side metrics
- Sheet: Insights     - Generated insights
- Sheet: Metadata     - Execution details
"""

from __future__ import annotations

import csv
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

# Try to import optional dependencies
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    from jinja2 import Template, Environment, BaseLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    HTML = "html"


@dataclass
class ExportOptions:
    """Options for export operations."""
    format: ExportFormat = ExportFormat.JSON
    output_path: Optional[Path] = None
    include_metadata: bool = True
    include_raw_results: bool = True
    include_comparison: bool = True
    include_insights: bool = True
    pretty_print: bool = True
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    decimal_places: int = 6
    # CSV specific
    csv_delimiter: str = ","
    csv_quoting: int = csv.QUOTE_MINIMAL
    # XLSX specific
    xlsx_freeze_panes: bool = True
    xlsx_auto_column_width: bool = True
    # HTML specific
    html_template: Optional[str] = None
    html_inline_styles: bool = True


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    format: ExportFormat
    output_path: Optional[Path]
    file_size_bytes: int = 0
    export_time_ms: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "format": self.format.value,
            "output_path": str(self.output_path) if self.output_path else None,
            "file_size_bytes": self.file_size_bytes,
            "export_time_ms": self.export_time_ms,
            "error": self.error,
            "warnings": self.warnings,
        }


@dataclass
class ReportData:
    """Data structure for export reports."""
    title: str = "Proxima Execution Report"
    generated_at: float = field(default_factory=time.time)
    
    # Summary section
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Raw measurement results
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Backend comparison data
    comparison: Dict[str, Any] = field(default_factory=dict)
    
    # Generated insights
    insights: List[str] = field(default_factory=list)
    
    # Execution metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Custom sections
    custom_sections: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "raw_results": self.raw_results,
            "comparison": self.comparison,
            "insights": self.insights,
            "metadata": self.metadata,
            "custom_sections": self.custom_sections,
        }


class Exporter(Protocol):
    """Protocol for exporters."""
    
    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        """Export data to the specified format."""
        ...


class BaseExporter(ABC):
    """Base class for exporters."""
    
    @abstractmethod
    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        """Export data to the specified format."""
        pass
    
    def _format_timestamp(self, timestamp: float, format_str: str) -> str:
        """Format a Unix timestamp to string."""
        return datetime.fromtimestamp(timestamp).strftime(format_str)
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class JSONExporter(BaseExporter):
    """Export data to JSON format."""
    
    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        start_time = time.perf_counter()
        
        try:
            # Build export data
            export_data = {
                "title": data.title,
                "generated_at": self._format_timestamp(
                    data.generated_at, options.timestamp_format
                ),
                "generated_at_unix": data.generated_at,
            }
            
            if options.include_metadata:
                export_data["summary"] = data.summary
            
            if options.include_raw_results:
                export_data["raw_results"] = data.raw_results
            
            if options.include_comparison:
                export_data["comparison"] = data.comparison
            
            if options.include_insights:
                export_data["insights"] = data.insights
            
            if options.include_metadata:
                export_data["metadata"] = data.metadata
            
            if data.custom_sections:
                export_data["custom_sections"] = data.custom_sections
            
            # Serialize to JSON
            indent = 2 if options.pretty_print else None
            json_str = json.dumps(export_data, indent=indent, default=str)
            
            # Write to file if path specified
            file_size = 0
            if options.output_path:
                options.output_path.parent.mkdir(parents=True, exist_ok=True)
                options.output_path.write_text(json_str, encoding="utf-8")
                file_size = options.output_path.stat().st_size
            
            export_time = (time.perf_counter() - start_time) * 1000
            
            return ExportResult(
                success=True,
                format=ExportFormat.JSON,
                output_path=options.output_path,
                file_size_bytes=file_size,
                export_time_ms=export_time,
            )
            
        except Exception as e:
            return ExportResult(
                success=False,
                format=ExportFormat.JSON,
                output_path=options.output_path,
                export_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )


class CSVExporter(BaseExporter):
    """Export data to CSV format."""
    
    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        start_time = time.perf_counter()
        warnings: List[str] = []
        
        try:
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.CSV,
                    output_path=None,
                    error="CSV export requires an output path",
                )
            
            options.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # CSV works best with tabular data - use raw_results
            rows = data.raw_results if data.raw_results else []
            
            if not rows:
                # Create summary as rows if no raw results
                flat_summary = self._flatten_dict(data.summary)
                rows = [{"key": k, "value": v} for k, v in flat_summary.items()]
                warnings.append("No raw results, exported summary instead")
            
            # Get all unique keys for headers
            headers: List[str] = []
            for row in rows:
                if isinstance(row, dict):
                    for key in row.keys():
                        if key not in headers:
                            headers.append(key)
            
            # Write CSV
            with open(options.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=headers,
                    delimiter=options.csv_delimiter,
                    quoting=options.csv_quoting,
                    extrasaction="ignore",
                )
                writer.writeheader()
                for row in rows:
                    if isinstance(row, dict):
                        # Convert complex values to strings
                        flat_row = {}
                        for k, v in row.items():
                            if isinstance(v, (dict, list)):
                                flat_row[k] = json.dumps(v)
                            else:
                                flat_row[k] = v
                        writer.writerow(flat_row)
            
            file_size = options.output_path.stat().st_size
            export_time = (time.perf_counter() - start_time) * 1000
            
            return ExportResult(
                success=True,
                format=ExportFormat.CSV,
                output_path=options.output_path,
                file_size_bytes=file_size,
                export_time_ms=export_time,
                warnings=warnings,
            )
            
        except Exception as e:
            return ExportResult(
                success=False,
                format=ExportFormat.CSV,
                output_path=options.output_path,
                export_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )


class XLSXExporter(BaseExporter):
    """Export data to XLSX format with multiple sheets.
    
    Report Structure (XLSX):
    - Sheet: Summary      - Overview, key metrics
    - Sheet: Raw Results  - Full measurement data
    - Sheet: Backend Comparison - Side-by-side metrics
    - Sheet: Insights     - Generated insights
    - Sheet: Metadata     - Execution details
    """
    
    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        start_time = time.perf_counter()
        
        if not HAS_OPENPYXL:
            return ExportResult(
                success=False,
                format=ExportFormat.XLSX,
                output_path=options.output_path,
                error="openpyxl is not installed. Install with: pip install openpyxl",
            )
        
        try:
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.XLSX,
                    output_path=None,
                    error="XLSX export requires an output path",
                )
            
            options.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create workbook
            wb = openpyxl.Workbook()
            
            # Define styles
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_alignment = Alignment(horizontal="center", vertical="center")
            
            # Sheet 1: Summary
            ws_summary = wb.active
            ws_summary.title = "Summary"
            self._write_summary_sheet(ws_summary, data, options, header_font, header_fill)
            
            # Sheet 2: Raw Results
            if options.include_raw_results and data.raw_results:
                ws_results = wb.create_sheet("Raw Results")
                self._write_results_sheet(ws_results, data.raw_results, options, header_font, header_fill)
            
            # Sheet 3: Backend Comparison
            if options.include_comparison and data.comparison:
                ws_comparison = wb.create_sheet("Backend Comparison")
                self._write_comparison_sheet(ws_comparison, data.comparison, options, header_font, header_fill)
            
            # Sheet 4: Insights
            if options.include_insights and data.insights:
                ws_insights = wb.create_sheet("Insights")
                self._write_insights_sheet(ws_insights, data.insights, options)
            
            # Sheet 5: Metadata
            if options.include_metadata and data.metadata:
                ws_metadata = wb.create_sheet("Metadata")
                self._write_metadata_sheet(ws_metadata, data.metadata, options, header_font, header_fill)
            
            # Save workbook
            wb.save(options.output_path)
            
            file_size = options.output_path.stat().st_size
            export_time = (time.perf_counter() - start_time) * 1000
            
            return ExportResult(
                success=True,
                format=ExportFormat.XLSX,
                output_path=options.output_path,
                file_size_bytes=file_size,
                export_time_ms=export_time,
            )
            
        except Exception as e:
            return ExportResult(
                success=False,
                format=ExportFormat.XLSX,
                output_path=options.output_path,
                export_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )
    
    def _apply_header_style(self, cell, font, fill):
        """Apply header styling to a cell."""
        cell.font = font
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    
    def _auto_adjust_columns(self, ws):
        """Auto-adjust column widths based on content."""
        for col_num in range(1, ws.max_column + 1):
            max_length = 0
            column_letter = get_column_letter(col_num)
            for row_num in range(1, ws.max_row + 1):
                try:
                    cell = ws.cell(row=row_num, column=col_num)
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            if adjusted_width > 5:  # Only adjust if meaningful content
                ws.column_dimensions[column_letter].width = adjusted_width
    
    def _write_summary_sheet(self, ws, data: ReportData, options: ExportOptions, header_font, header_fill):
        """Write the Summary sheet."""
        # Title
        ws["A1"] = data.title
        ws["A1"].font = Font(bold=True, size=16)
        ws.merge_cells("A1:C1")
        
        # Generated timestamp
        ws["A2"] = "Generated:"
        ws["B2"] = self._format_timestamp(data.generated_at, options.timestamp_format)
        
        # Summary data
        row = 4
        ws[f"A{row}"] = "Key"
        ws[f"B{row}"] = "Value"
        self._apply_header_style(ws[f"A{row}"], header_font, header_fill)
        self._apply_header_style(ws[f"B{row}"], header_font, header_fill)
        
        row += 1
        flat_summary = self._flatten_dict(data.summary)
        for key, value in flat_summary.items():
            ws[f"A{row}"] = key
            ws[f"B{row}"] = str(value) if not isinstance(value, (int, float)) else value
            row += 1
        
        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A5"
        
        if options.xlsx_auto_column_width:
            self._auto_adjust_columns(ws)
    
    def _write_results_sheet(self, ws, raw_results: List[Dict], options: ExportOptions, header_font, header_fill):
        """Write the Raw Results sheet."""
        if not raw_results:
            ws["A1"] = "No results data"
            return
        
        # Get all unique headers
        headers = []
        for result in raw_results:
            if isinstance(result, dict):
                for key in result.keys():
                    if key not in headers:
                        headers.append(key)
        
        # Write headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            self._apply_header_style(cell, header_font, header_fill)
        
        # Write data
        for row_idx, result in enumerate(raw_results, 2):
            if isinstance(result, dict):
                for col_idx, header in enumerate(headers, 1):
                    value = result.get(header, "")
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    ws.cell(row=row_idx, column=col_idx, value=value)
        
        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A2"
        
        if options.xlsx_auto_column_width:
            self._auto_adjust_columns(ws)
    
    def _write_comparison_sheet(self, ws, comparison: Dict, options: ExportOptions, header_font, header_fill):
        """Write the Backend Comparison sheet."""
        row = 1
        
        # Flatten comparison data for display
        flat_comparison = self._flatten_dict(comparison)
        
        ws[f"A{row}"] = "Metric"
        ws[f"B{row}"] = "Value"
        self._apply_header_style(ws[f"A{row}"], header_font, header_fill)
        self._apply_header_style(ws[f"B{row}"], header_font, header_fill)
        
        row += 1
        for key, value in flat_comparison.items():
            ws[f"A{row}"] = key
            if isinstance(value, float):
                ws[f"B{row}"] = round(value, options.decimal_places)
            elif isinstance(value, (dict, list)):
                ws[f"B{row}"] = json.dumps(value)
            else:
                ws[f"B{row}"] = value
            row += 1
        
        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A2"
        
        if options.xlsx_auto_column_width:
            self._auto_adjust_columns(ws)
    
    def _write_insights_sheet(self, ws, insights: List[str], options: ExportOptions):
        """Write the Insights sheet."""
        ws["A1"] = "Insights"
        ws["A1"].font = Font(bold=True, size=14)
        
        for idx, insight in enumerate(insights, 2):
            ws[f"A{idx}"] = f"{idx-1}. {insight}"
        
        if options.xlsx_auto_column_width:
            self._auto_adjust_columns(ws)
    
    def _write_metadata_sheet(self, ws, metadata: Dict, options: ExportOptions, header_font, header_fill):
        """Write the Metadata sheet."""
        ws["A1"] = "Property"
        ws["B1"] = "Value"
        self._apply_header_style(ws["A1"], header_font, header_fill)
        self._apply_header_style(ws["B1"], header_font, header_fill)
        
        flat_metadata = self._flatten_dict(metadata)
        for row, (key, value) in enumerate(flat_metadata.items(), 2):
            ws[f"A{row}"] = key
            if isinstance(value, (dict, list)):
                ws[f"B{row}"] = json.dumps(value)
            else:
                ws[f"B{row}"] = value
        
        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A2"
        
        if options.xlsx_auto_column_width:
            self._auto_adjust_columns(ws)


class HTMLExporter(BaseExporter):
    """Export data to HTML format with rich formatting."""
    
    DEFAULT_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    {% if inline_styles %}
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #3498db; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ecf0f1; }
        tr:hover { background: #f8f9fa; }
        .insight { background: #e8f6ff; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; border-radius: 4px; }
        .metric { display: inline-block; background: #f0f0f0; padding: 5px 10px; margin: 5px; border-radius: 4px; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
    </style>
    {% endif %}
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p class="timestamp">Generated: {{ generated_at }}</p>
        
        {% if summary %}
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {% for key, value in summary.items() %}
            <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if comparison %}
        <h2>Backend Comparison</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {% for key, value in comparison.items() %}
            <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if insights %}
        <h2>Insights</h2>
        {% for insight in insights %}
        <div class="insight">{{ insight }}</div>
        {% endfor %}
        {% endif %}
        
        {% if raw_results %}
        <h2>Results</h2>
        <table>
            {% if raw_results %}
            <tr>
                {% for key in raw_results[0].keys() %}
                <th>{{ key }}</th>
                {% endfor %}
            </tr>
            {% for row in raw_results %}
            <tr>
                {% for value in row.values() %}
                <td>{{ value }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
            {% endif %}
        </table>
        {% endif %}
        
        {% if metadata %}
        <h2>Metadata</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            {% for key, value in metadata.items() %}
            <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
</body>
</html>'''
    
    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        start_time = time.perf_counter()
        
        if not HAS_JINJA2:
            # Fallback to basic HTML without Jinja2
            return self._export_basic_html(data, options, start_time)
        
        try:
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.HTML,
                    output_path=None,
                    error="HTML export requires an output path",
                )
            
            options.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use custom template or default
            template_str = options.html_template or self.DEFAULT_TEMPLATE
            template = Template(template_str)
            
            # Prepare template data
            flat_summary = self._flatten_dict(data.summary)
            flat_comparison = self._flatten_dict(data.comparison)
            flat_metadata = self._flatten_dict(data.metadata)
            
            # Render HTML
            html = template.render(
                title=data.title,
                generated_at=self._format_timestamp(data.generated_at, options.timestamp_format),
                summary=flat_summary if options.include_metadata else {},
                raw_results=data.raw_results if options.include_raw_results else [],
                comparison=flat_comparison if options.include_comparison else {},
                insights=data.insights if options.include_insights else [],
                metadata=flat_metadata if options.include_metadata else {},
                inline_styles=options.html_inline_styles,
            )
            
            # Write to file
            options.output_path.write_text(html, encoding="utf-8")
            
            file_size = options.output_path.stat().st_size
            export_time = (time.perf_counter() - start_time) * 1000
            
            return ExportResult(
                success=True,
                format=ExportFormat.HTML,
                output_path=options.output_path,
                file_size_bytes=file_size,
                export_time_ms=export_time,
            )
            
        except Exception as e:
            return ExportResult(
                success=False,
                format=ExportFormat.HTML,
                output_path=options.output_path,
                export_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )
    
    def _export_basic_html(self, data: ReportData, options: ExportOptions, start_time: float) -> ExportResult:
        """Fallback HTML export without Jinja2."""
        try:
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.HTML,
                    output_path=None,
                    error="HTML export requires an output path",
                )
            
            options.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build simple HTML
            html_parts = [
                "<!DOCTYPE html>",
                "<html><head><meta charset='UTF-8'>",
                f"<title>{data.title}</title>",
                "<style>body{font-family:sans-serif;margin:40px}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:8px}th{background:#4472C4;color:white}</style>",
                "</head><body>",
                f"<h1>{data.title}</h1>",
                f"<p>Generated: {self._format_timestamp(data.generated_at, options.timestamp_format)}</p>",
            ]
            
            # Summary
            if data.summary:
                html_parts.append("<h2>Summary</h2><table><tr><th>Metric</th><th>Value</th></tr>")
                for k, v in self._flatten_dict(data.summary).items():
                    html_parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
                html_parts.append("</table>")
            
            # Insights
            if data.insights:
                html_parts.append("<h2>Insights</h2><ul>")
                for insight in data.insights:
                    html_parts.append(f"<li>{insight}</li>")
                html_parts.append("</ul>")
            
            html_parts.append("</body></html>")
            
            html = "\n".join(html_parts)
            options.output_path.write_text(html, encoding="utf-8")
            
            file_size = options.output_path.stat().st_size
            export_time = (time.perf_counter() - start_time) * 1000
            
            return ExportResult(
                success=True,
                format=ExportFormat.HTML,
                output_path=options.output_path,
                file_size_bytes=file_size,
                export_time_ms=export_time,
                warnings=["Jinja2 not installed, using basic HTML template"],
            )
            
        except Exception as e:
            return ExportResult(
                success=False,
                format=ExportFormat.HTML,
                output_path=options.output_path,
                export_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )


class ExportEngine:
    """Main export engine that handles all export formats.
    
    Export Formats:
    - CSV:  Simple tabular data (csv stdlib)
    - XLSX: Multiple sheets, formatting (openpyxl)
    - JSON: Full data structure (json stdlib)
    - HTML: Rich formatted reports (jinja2)
    """
    
    def __init__(self) -> None:
        self._exporters: Dict[ExportFormat, BaseExporter] = {
            ExportFormat.JSON: JSONExporter(),
            ExportFormat.CSV: CSVExporter(),
            ExportFormat.XLSX: XLSXExporter(),
            ExportFormat.HTML: HTMLExporter(),
        }
    
    def register_exporter(self, format: ExportFormat, exporter: BaseExporter) -> None:
        """Register a custom exporter for a format."""
        self._exporters[format] = exporter
    
    def get_available_formats(self) -> List[ExportFormat]:
        """Get list of available export formats."""
        available = [ExportFormat.JSON, ExportFormat.CSV]  # Always available
        if HAS_OPENPYXL:
            available.append(ExportFormat.XLSX)
        available.append(ExportFormat.HTML)  # Always available (with fallback)
        return available
    
    def export(
        self,
        data: ReportData,
        format: ExportFormat = ExportFormat.JSON,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> ExportResult:
        """Export data to the specified format.
        
        Args:
            data: ReportData to export
            format: Export format (JSON, CSV, XLSX, HTML)
            output_path: Output file path
            **kwargs: Additional options passed to ExportOptions
            
        Returns:
            ExportResult with export status
        """
        options = ExportOptions(
            format=format,
            output_path=output_path,
            **kwargs,
        )
        
        exporter = self._exporters.get(format)
        if not exporter:
            return ExportResult(
                success=False,
                format=format,
                output_path=output_path,
                error=f"No exporter registered for format: {format.value}",
            )
        
        return exporter.export(data, options)
    
    def export_all(
        self,
        data: ReportData,
        output_dir: Path,
        base_name: str = "report",
        formats: Optional[List[ExportFormat]] = None,
    ) -> Dict[ExportFormat, ExportResult]:
        """Export data to multiple formats.
        
        Args:
            data: ReportData to export
            output_dir: Directory for output files
            base_name: Base filename (without extension)
            formats: List of formats to export (default: all available)
            
        Returns:
            Dict mapping format to ExportResult
        """
        if formats is None:
            formats = self.get_available_formats()
        
        results: Dict[ExportFormat, ExportResult] = {}
        
        for format in formats:
            extension = format.value
            output_path = output_dir / f"{base_name}.{extension}"
            results[format] = self.export(data, format, output_path)
        
        return results


# Convenience functions
def export_to_json(
    data: ReportData,
    output_path: Path,
    pretty: bool = True,
) -> ExportResult:
    """Export data to JSON format."""
    engine = ExportEngine()
    return engine.export(data, ExportFormat.JSON, output_path, pretty_print=pretty)


def export_to_csv(
    data: ReportData,
    output_path: Path,
) -> ExportResult:
    """Export data to CSV format."""
    engine = ExportEngine()
    return engine.export(data, ExportFormat.CSV, output_path)


def export_to_xlsx(
    data: ReportData,
    output_path: Path,
) -> ExportResult:
    """Export data to XLSX format."""
    engine = ExportEngine()
    return engine.export(data, ExportFormat.XLSX, output_path)


def export_to_html(
    data: ReportData,
    output_path: Path,
) -> ExportResult:
    """Export data to HTML format."""
    engine = ExportEngine()
    return engine.export(data, ExportFormat.HTML, output_path)
