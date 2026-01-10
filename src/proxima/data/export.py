"""Step 5.3: Export Engine - Export results in multiple formats.

Export Formats:
| Format   | Library       | Features                      |
| CSV      | csv (stdlib)  | Simple tabular data           |
| XLSX     | openpyxl      | Multiple sheets, formatting   |
| JSON     | json (stdlib) | Full data structure           |
| HTML     | jinja2        | Rich formatted reports        |
| MARKDOWN | stdlib        | Documentation-friendly format |
| YAML     | pyyaml/stdlib | Config-friendly format        |

Report Structure (XLSX):
Workbook:
- Sheet: Summary        - Overview, key metrics
- Sheet: Raw Results    - Full measurement data
- Sheet: Backend Comparison - Side-by-side metrics
- Sheet: Insights       - Generated insights
- Sheet: Metadata       - Execution details
"""

from __future__ import annotations

import csv
import io
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

# Try to import optional dependencies
try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    from jinja2 import BaseLoader, Environment, Template

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ExportFormat(Enum):
    """Supported export formats."""

    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "md"
    YAML = "yaml"


@dataclass
class ExportOptions:
    """Options for export operations."""

    format: ExportFormat = ExportFormat.JSON
    output_path: Path | None = None
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
    html_template: str | None = None
    html_inline_styles: bool = True
    # Markdown specific
    markdown_toc: bool = True
    markdown_code_blocks: bool = True
    # YAML specific
    yaml_default_flow_style: bool = False
    yaml_allow_unicode: bool = True
    # Stream export (returns string instead of writing to file)
    stream_output: bool = False


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    format: ExportFormat
    output_path: Path | None
    file_size_bytes: int = 0
    export_time_ms: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    # For stream exports
    content: str | bytes | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "format": self.format.value,
            "output_path": str(self.output_path) if self.output_path else None,
            "file_size_bytes": self.file_size_bytes,
            "export_time_ms": self.export_time_ms,
            "error": self.error,
            "warnings": self.warnings,
            "has_content": self.content is not None,
        }


@dataclass
class ReportData:
    """Data structure for export reports."""

    title: str = "Proxima Execution Report"
    generated_at: float = field(default_factory=time.time)

    # Summary section
    summary: dict[str, Any] = field(default_factory=dict)

    # Raw measurement results
    raw_results: list[dict[str, Any]] = field(default_factory=list)

    # Backend comparison data
    comparison: dict[str, Any] = field(default_factory=dict)

    # Generated insights
    insights: list[str] = field(default_factory=list)

    # Execution metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Custom sections
    custom_sections: dict[str, Any] = field(default_factory=dict)

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
    """Base class for all exporters."""

    @abstractmethod
    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        """Export data to the specified format."""
        ...

    def _format_timestamp(self, timestamp: float, format_str: str) -> str:
        """Format a Unix timestamp to a string."""
        return datetime.fromtimestamp(timestamp).strftime(format_str)

    def _round_floats(self, obj: Any, decimal_places: int) -> Any:
        """Recursively round floats in a data structure."""
        if isinstance(obj, float):
            return round(obj, decimal_places)
        elif isinstance(obj, dict):
            return {k: self._round_floats(v, decimal_places) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_floats(item, decimal_places) for item in obj]
        return obj

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten a nested dictionary."""
        items: list[tuple[str, Any]] = []
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
        """Export data to JSON format."""
        start_time = time.perf_counter()

        try:
            # Build export data
            export_data: dict[str, Any] = {
                "title": data.title,
                "generated_at": self._format_timestamp(data.generated_at, options.timestamp_format),
                "summary": data.summary if data.summary else {},
            }

            if options.include_raw_results and data.raw_results:
                export_data["raw_results"] = data.raw_results

            if options.include_comparison and data.comparison:
                export_data["comparison"] = data.comparison

            if options.include_insights and data.insights:
                export_data["insights"] = data.insights

            if options.include_metadata and data.metadata:
                export_data["metadata"] = data.metadata

            if data.custom_sections:
                export_data["custom_sections"] = data.custom_sections

            # Round floats
            export_data = self._round_floats(export_data, options.decimal_places)

            # Convert to JSON
            indent = 2 if options.pretty_print else None
            json_content = json.dumps(export_data, indent=indent, ensure_ascii=False)

            # Stream output
            if options.stream_output:
                export_time = (time.perf_counter() - start_time) * 1000
                return ExportResult(
                    success=True,
                    format=ExportFormat.JSON,
                    output_path=None,
                    file_size_bytes=len(json_content.encode("utf-8")),
                    export_time_ms=export_time,
                    content=json_content,
                )

            # Write to file
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.JSON,
                    output_path=None,
                    error="JSON export requires an output path or stream_output=True",
                )

            options.output_path.parent.mkdir(parents=True, exist_ok=True)
            options.output_path.write_text(json_content, encoding="utf-8")

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
        """Export data to CSV format (raw results as tabular data)."""
        start_time = time.perf_counter()

        try:
            # Stream output
            if options.stream_output:
                output = io.StringIO()
                self._write_csv(data, options, output)
                csv_content = output.getvalue()
                export_time = (time.perf_counter() - start_time) * 1000
                return ExportResult(
                    success=True,
                    format=ExportFormat.CSV,
                    output_path=None,
                    file_size_bytes=len(csv_content.encode("utf-8")),
                    export_time_ms=export_time,
                    content=csv_content,
                )

            # File output
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.CSV,
                    output_path=None,
                    error="CSV export requires an output path or stream_output=True",
                )

            options.output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(options.output_path, "w", newline="", encoding="utf-8") as f:
                self._write_csv(data, options, f)

            file_size = options.output_path.stat().st_size
            export_time = (time.perf_counter() - start_time) * 1000

            return ExportResult(
                success=True,
                format=ExportFormat.CSV,
                output_path=options.output_path,
                file_size_bytes=file_size,
                export_time_ms=export_time,
            )

        except Exception as e:
            return ExportResult(
                success=False,
                format=ExportFormat.CSV,
                output_path=options.output_path,
                export_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _write_csv(self, data: ReportData, options: ExportOptions, output: Any) -> None:
        """Write CSV data to a file or string buffer."""
        if not data.raw_results:
            # Write summary as key-value pairs
            writer = csv.writer(
                output, delimiter=options.csv_delimiter, quoting=options.csv_quoting
            )
            writer.writerow(["Key", "Value"])
            flat_summary = self._flatten_dict(data.summary)
            for key, value in flat_summary.items():
                if isinstance(value, float):
                    value = round(value, options.decimal_places)
                writer.writerow([key, value])
        else:
            # Write raw results as table
            if data.raw_results:
                headers = list(data.raw_results[0].keys())
                writer = csv.DictWriter(
                    output,
                    fieldnames=headers,
                    delimiter=options.csv_delimiter,
                    quoting=options.csv_quoting,
                )
                writer.writeheader()
                for row in data.raw_results:
                    # Round floats in row
                    processed_row = {}
                    for k, v in row.items():
                        if isinstance(v, float):
                            processed_row[k] = round(v, options.decimal_places)
                        else:
                            processed_row[k] = v
                    writer.writerow(processed_row)


class XLSXExporter(BaseExporter):
    """Export data to XLSX format with multiple sheets."""

    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        """Export data to XLSX format."""
        start_time = time.perf_counter()

        if not HAS_OPENPYXL:
            return ExportResult(
                success=False,
                format=ExportFormat.XLSX,
                output_path=options.output_path,
                error="openpyxl is required for XLSX export. Install with: pip install openpyxl",
            )

        try:
            wb = openpyxl.Workbook()

            # Define styles
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_alignment = Alignment(horizontal="center", vertical="center")
            thin_border = Border(
                left=Side(style="thin"),
                right=Side(style="thin"),
                top=Side(style="thin"),
                bottom=Side(style="thin"),
            )

            # Create Summary sheet
            ws = wb.active
            if ws is not None:
                ws.title = "Summary"
                self._write_summary_sheet(
                    ws,
                    data,
                    options,
                    header_font,
                    header_fill,
                    header_alignment,
                    thin_border,
                )

            # Create Raw Results sheet
            if options.include_raw_results and data.raw_results:
                ws_raw = wb.create_sheet("Raw Results")
                self._write_raw_results_sheet(
                    ws_raw,
                    data,
                    options,
                    header_font,
                    header_fill,
                    header_alignment,
                    thin_border,
                )

            # Create Backend Comparison sheet
            if options.include_comparison and data.comparison:
                ws_comp = wb.create_sheet("Backend Comparison")
                self._write_comparison_sheet(
                    ws_comp,
                    data,
                    options,
                    header_font,
                    header_fill,
                    header_alignment,
                    thin_border,
                )

            # Create Insights sheet
            if options.include_insights and data.insights:
                ws_insights = wb.create_sheet("Insights")
                self._write_insights_sheet(
                    ws_insights,
                    data,
                    options,
                    header_font,
                    header_fill,
                    header_alignment,
                    thin_border,
                )

            # Create Metadata sheet
            if options.include_metadata and data.metadata:
                ws_meta = wb.create_sheet("Metadata")
                self._write_metadata_sheet(
                    ws_meta,
                    data,
                    options,
                    header_font,
                    header_fill,
                    header_alignment,
                    thin_border,
                )

            # Stream output
            if options.stream_output:
                buffer = io.BytesIO()
                wb.save(buffer)
                xlsx_content = buffer.getvalue()
                export_time = (time.perf_counter() - start_time) * 1000
                return ExportResult(
                    success=True,
                    format=ExportFormat.XLSX,
                    output_path=None,
                    file_size_bytes=len(xlsx_content),
                    export_time_ms=export_time,
                    content=xlsx_content,
                )

            # File output
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.XLSX,
                    output_path=None,
                    error="XLSX export requires an output path or stream_output=True",
                )

            options.output_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _write_summary_sheet(
        self,
        ws: Any,
        data: ReportData,
        options: ExportOptions,
        header_font: Any,
        header_fill: Any,
        header_alignment: Any,
        thin_border: Any,
    ) -> None:
        """Write the summary sheet."""
        # Title row (no merge to avoid MergedCell issues)
        ws["A1"] = f"Report: {data.title}"
        ws["A1"].font = Font(bold=True, size=16)

        # Generated timestamp
        ws["A2"] = "Generated:"
        ws["B2"] = self._format_timestamp(data.generated_at, options.timestamp_format)

        # Summary table
        ws["A4"] = "Metric"
        ws["B4"] = "Value"
        ws["A4"].font = header_font
        ws["B4"].font = header_font
        ws["A4"].fill = header_fill
        ws["B4"].fill = header_fill
        ws["A4"].alignment = header_alignment
        ws["B4"].alignment = header_alignment

        row = 5
        flat_summary = self._flatten_dict(data.summary)
        for key, value in flat_summary.items():
            ws[f"A{row}"] = key
            if isinstance(value, float):
                ws[f"B{row}"] = round(value, options.decimal_places)
            else:
                ws[f"B{row}"] = value
            ws[f"A{row}"].border = thin_border
            ws[f"B{row}"].border = thin_border
            row += 1

        # Auto-size columns
        if options.xlsx_auto_column_width:
            self._auto_size_columns(ws)

        # Freeze panes
        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A5"

    def _write_raw_results_sheet(
        self,
        ws: Any,
        data: ReportData,
        options: ExportOptions,
        header_font: Any,
        header_fill: Any,
        header_alignment: Any,
        thin_border: Any,
    ) -> None:
        """Write the raw results sheet."""
        if not data.raw_results:
            return

        headers = list(data.raw_results[0].keys())

        # Write headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border

        # Write data
        for row_idx, result in enumerate(data.raw_results, 2):
            for col_idx, header in enumerate(headers, 1):
                value = result.get(header)
                if isinstance(value, float):
                    value = round(value, options.decimal_places)
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.border = thin_border

        # Auto-size columns
        if options.xlsx_auto_column_width:
            self._auto_size_columns(ws)

        # Freeze panes
        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A2"

    def _write_comparison_sheet(
        self,
        ws: Any,
        data: ReportData,
        options: ExportOptions,
        header_font: Any,
        header_fill: Any,
        header_alignment: Any,
        thin_border: Any,
    ) -> None:
        """Write the backend comparison sheet."""
        ws["A1"] = "Key"
        ws["B1"] = "Value"
        ws["A1"].font = header_font
        ws["B1"].font = header_font
        ws["A1"].fill = header_fill
        ws["B1"].fill = header_fill
        ws["A1"].alignment = header_alignment
        ws["B1"].alignment = header_alignment

        row = 2
        flat_comparison = self._flatten_dict(data.comparison)
        for key, value in flat_comparison.items():
            ws[f"A{row}"] = key
            if isinstance(value, float):
                ws[f"B{row}"] = round(value, options.decimal_places)
            else:
                ws[f"B{row}"] = value
            ws[f"A{row}"].border = thin_border
            ws[f"B{row}"].border = thin_border
            row += 1

        if options.xlsx_auto_column_width:
            self._auto_size_columns(ws)

        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A2"

    def _write_insights_sheet(
        self,
        ws: Any,
        data: ReportData,
        options: ExportOptions,
        header_font: Any,
        header_fill: Any,
        header_alignment: Any,
        thin_border: Any,
    ) -> None:
        """Write the insights sheet."""
        ws["A1"] = "#"
        ws["B1"] = "Insight"
        ws["A1"].font = header_font
        ws["B1"].font = header_font
        ws["A1"].fill = header_fill
        ws["B1"].fill = header_fill
        ws["A1"].alignment = header_alignment
        ws["B1"].alignment = header_alignment

        for row, insight in enumerate(data.insights, 2):
            ws[f"A{row}"] = row - 1
            ws[f"B{row}"] = insight
            ws[f"A{row}"].border = thin_border
            ws[f"B{row}"].border = thin_border

        if options.xlsx_auto_column_width:
            self._auto_size_columns(ws)

        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A2"

    def _write_metadata_sheet(
        self,
        ws: Any,
        data: ReportData,
        options: ExportOptions,
        header_font: Any,
        header_fill: Any,
        header_alignment: Any,
        thin_border: Any,
    ) -> None:
        """Write the metadata sheet."""
        ws["A1"] = "Key"
        ws["B1"] = "Value"
        ws["A1"].font = header_font
        ws["B1"].font = header_font
        ws["A1"].fill = header_fill
        ws["B1"].fill = header_fill
        ws["A1"].alignment = header_alignment
        ws["B1"].alignment = header_alignment

        row = 2
        flat_metadata = self._flatten_dict(data.metadata)
        for key, value in flat_metadata.items():
            ws[f"A{row}"] = key
            if isinstance(value, float):
                ws[f"B{row}"] = round(value, options.decimal_places)
            else:
                ws[f"B{row}"] = value
            ws[f"A{row}"].border = thin_border
            ws[f"B{row}"].border = thin_border
            row += 1

        if options.xlsx_auto_column_width:
            self._auto_size_columns(ws)

        if options.xlsx_freeze_panes:
            ws.freeze_panes = "A2"

    def _auto_size_columns(self, ws: Any) -> None:
        """Auto-size columns based on content."""
        for col_idx, column_cells in enumerate(ws.columns, 1):
            max_length = 0
            for cell in column_cells:
                # Skip merged cells by checking for value attribute
                if hasattr(cell, "value") and cell.value is not None:
                    try:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                    except Exception:
                        pass
            if max_length > 0:
                column_letter = get_column_letter(col_idx)
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width


class HTMLExporter(BaseExporter):
    """Export data to HTML format."""

    DEFAULT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    {% if inline_styles %}
    <style>
        :root {
            --primary-color: #4472C4;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --text-color: #333;
            --bg-color: #f8f9fa;
            --border-color: #dee2e6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background: var(--bg-color);
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 2rem; }
        h1 { color: var(--primary-color); margin-bottom: 0.5rem; }
        h2 { color: var(--primary-color); margin: 2rem 0 1rem; border-bottom: 2px solid var(--primary-color); padding-bottom: 0.5rem; }
        .timestamp { color: #666; margin-bottom: 2rem; }
        table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        th { background: var(--primary-color); color: white; padding: 12px; text-align: left; }
        td { padding: 10px 12px; border-bottom: 1px solid var(--border-color); }
        tr:hover { background: #f5f5f5; }
        .insight { background: #e7f3ff; border-left: 4px solid var(--primary-color); padding: 1rem; margin: 0.5rem 0; border-radius: 4px; }
        .metric-card { display: inline-block; background: var(--bg-color); padding: 1rem; border-radius: 8px; margin: 0.5rem; min-width: 150px; text-align: center; }
        .metric-value { font-size: 1.5rem; font-weight: bold; color: var(--primary-color); }
        .metric-label { color: #666; font-size: 0.9rem; }
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

        {% if raw_results %}
        <h2>Raw Results</h2>
        <table>
            <tr>
            {% for header in raw_results[0].keys() %}
                <th>{{ header }}</th>
            {% endfor %}
            </tr>
            {% for row in raw_results %}
            <tr>
            {% for value in row.values() %}
                <td>{{ value }}</td>
            {% endfor %}
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        {% if comparison %}
        <h2>Backend Comparison</h2>
        <table>
            <tr><th>Key</th><th>Value</th></tr>
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

        {% if metadata %}
        <h2>Metadata</h2>
        <table>
            <tr><th>Key</th><th>Value</th></tr>
            {% for key, value in metadata.items() %}
            <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
</body>
</html>
"""

    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        """Export data to HTML format."""
        start_time = time.perf_counter()

        if HAS_JINJA2:
            return self._export_with_jinja2(data, options, start_time)
        else:
            return self._export_basic_html(data, options, start_time)

    def _export_with_jinja2(
        self, data: ReportData, options: ExportOptions, start_time: float
    ) -> ExportResult:
        """Export using Jinja2 templates."""
        try:
            template_str = options.html_template or self.DEFAULT_TEMPLATE
            template = Template(template_str)

            # Flatten nested dicts for display
            flat_summary = self._flatten_dict(data.summary) if data.summary else {}
            flat_comparison = self._flatten_dict(data.comparison) if data.comparison else {}
            flat_metadata = self._flatten_dict(data.metadata) if data.metadata else {}

            # Round floats
            flat_summary = self._round_floats(flat_summary, options.decimal_places)
            flat_comparison = self._round_floats(flat_comparison, options.decimal_places)
            flat_metadata = self._round_floats(flat_metadata, options.decimal_places)

            html = template.render(
                title=data.title,
                generated_at=self._format_timestamp(data.generated_at, options.timestamp_format),
                summary=flat_summary if data.summary else None,
                raw_results=data.raw_results if options.include_raw_results else [],
                comparison=flat_comparison if options.include_comparison else None,
                insights=data.insights if options.include_insights else [],
                metadata=flat_metadata if options.include_metadata else None,
                inline_styles=options.html_inline_styles,
            )

            # Stream output
            if options.stream_output:
                export_time = (time.perf_counter() - start_time) * 1000
                return ExportResult(
                    success=True,
                    format=ExportFormat.HTML,
                    output_path=None,
                    file_size_bytes=len(html.encode("utf-8")),
                    export_time_ms=export_time,
                    content=html,
                )

            # Write to file
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.HTML,
                    output_path=None,
                    error="HTML export requires an output path or stream_output=True",
                )

            options.output_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _export_basic_html(
        self, data: ReportData, options: ExportOptions, start_time: float
    ) -> ExportResult:
        """Fallback HTML export without Jinja2."""
        try:
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

            # Stream output
            if options.stream_output:
                export_time = (time.perf_counter() - start_time) * 1000
                return ExportResult(
                    success=True,
                    format=ExportFormat.HTML,
                    output_path=None,
                    file_size_bytes=len(html.encode("utf-8")),
                    export_time_ms=export_time,
                    content=html,
                    warnings=["Jinja2 not installed, using basic HTML template"],
                )

            # File output
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.HTML,
                    output_path=None,
                    error="HTML export requires an output path or stream_output=True",
                )

            options.output_path.parent.mkdir(parents=True, exist_ok=True)
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


class MarkdownExporter(BaseExporter):
    """Export data to Markdown format."""

    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        """Export data to Markdown format."""
        start_time = time.perf_counter()

        try:
            md_parts: list[str] = []

            # Title
            md_parts.append(f"# {data.title}")
            md_parts.append("")
            md_parts.append(
                f"*Generated: {self._format_timestamp(data.generated_at, options.timestamp_format)}*"
            )
            md_parts.append("")

            # Table of contents
            if options.markdown_toc:
                md_parts.append("## Table of Contents")
                md_parts.append("")
                if data.summary:
                    md_parts.append("- [Summary](#summary)")
                if options.include_raw_results and data.raw_results:
                    md_parts.append("- [Raw Results](#raw-results)")
                if options.include_comparison and data.comparison:
                    md_parts.append("- [Backend Comparison](#backend-comparison)")
                if options.include_insights and data.insights:
                    md_parts.append("- [Insights](#insights)")
                if options.include_metadata and data.metadata:
                    md_parts.append("- [Metadata](#metadata)")
                md_parts.append("")

            # Summary section
            if data.summary:
                md_parts.append("## Summary")
                md_parts.append("")
                md_parts.append("| Metric | Value |")
                md_parts.append("|--------|-------|")
                flat_summary = self._flatten_dict(data.summary)
                for key, value in flat_summary.items():
                    if isinstance(value, float):
                        value = round(value, options.decimal_places)
                    md_parts.append(f"| {key} | {value} |")
                md_parts.append("")

            # Raw Results section
            if options.include_raw_results and data.raw_results:
                md_parts.append("## Raw Results")
                md_parts.append("")
                headers = list(data.raw_results[0].keys())
                md_parts.append("| " + " | ".join(headers) + " |")
                md_parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in data.raw_results:
                    values = []
                    for h in headers:
                        v = row.get(h, "")
                        if isinstance(v, float):
                            v = round(v, options.decimal_places)
                        values.append(str(v) if v is not None else "N/A")
                    md_parts.append("| " + " | ".join(values) + " |")
                md_parts.append("")

            # Comparison section
            if options.include_comparison and data.comparison:
                md_parts.append("## Backend Comparison")
                md_parts.append("")
                md_parts.append("| Key | Value |")
                md_parts.append("|-----|-------|")
                flat_comparison = self._flatten_dict(data.comparison)
                for key, value in flat_comparison.items():
                    if isinstance(value, float):
                        value = round(value, options.decimal_places)
                    md_parts.append(f"| {key} | {value} |")
                md_parts.append("")

            # Insights section
            if options.include_insights and data.insights:
                md_parts.append("## Insights")
                md_parts.append("")
                for insight in data.insights:
                    md_parts.append(f"- {insight}")
                md_parts.append("")

            # Metadata section
            if options.include_metadata and data.metadata:
                md_parts.append("## Metadata")
                md_parts.append("")
                if options.markdown_code_blocks:
                    md_parts.append("```")
                    flat_metadata = self._flatten_dict(data.metadata)
                    for key, value in flat_metadata.items():
                        md_parts.append(f"{key}: {value}")
                    md_parts.append("```")
                else:
                    md_parts.append("| Key | Value |")
                    md_parts.append("|-----|-------|")
                    flat_metadata = self._flatten_dict(data.metadata)
                    for key, value in flat_metadata.items():
                        md_parts.append(f"| {key} | {value} |")
                md_parts.append("")

            # Custom sections
            if data.custom_sections:
                for section_name, section_data in data.custom_sections.items():
                    md_parts.append(f"## {section_name}")
                    md_parts.append("")
                    if isinstance(section_data, dict):
                        md_parts.append("| Key | Value |")
                        md_parts.append("|-----|-------|")
                        flat_section = self._flatten_dict(section_data)
                        for key, value in flat_section.items():
                            md_parts.append(f"| {key} | {value} |")
                    elif isinstance(section_data, list):
                        for item in section_data:
                            md_parts.append(f"- {item}")
                    else:
                        md_parts.append(str(section_data))
                    md_parts.append("")

            md_content = "\n".join(md_parts)

            # Stream output
            if options.stream_output:
                export_time = (time.perf_counter() - start_time) * 1000
                return ExportResult(
                    success=True,
                    format=ExportFormat.MARKDOWN,
                    output_path=None,
                    file_size_bytes=len(md_content.encode("utf-8")),
                    export_time_ms=export_time,
                    content=md_content,
                )

            # File output
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.MARKDOWN,
                    output_path=None,
                    error="Markdown export requires an output path or stream_output=True",
                )

            options.output_path.parent.mkdir(parents=True, exist_ok=True)
            options.output_path.write_text(md_content, encoding="utf-8")

            file_size = options.output_path.stat().st_size
            export_time = (time.perf_counter() - start_time) * 1000

            return ExportResult(
                success=True,
                format=ExportFormat.MARKDOWN,
                output_path=options.output_path,
                file_size_bytes=file_size,
                export_time_ms=export_time,
            )

        except Exception as e:
            return ExportResult(
                success=False,
                format=ExportFormat.MARKDOWN,
                output_path=options.output_path,
                export_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )


class YAMLExporter(BaseExporter):
    """Export data to YAML format."""

    def export(self, data: ReportData, options: ExportOptions) -> ExportResult:
        """Export data to YAML format."""
        start_time = time.perf_counter()

        try:
            # Build export data
            export_data: dict[str, Any] = {
                "title": data.title,
                "generated_at": self._format_timestamp(data.generated_at, options.timestamp_format),
                "summary": data.summary if data.summary else {},
            }

            if options.include_raw_results and data.raw_results:
                export_data["raw_results"] = data.raw_results

            if options.include_comparison and data.comparison:
                export_data["comparison"] = data.comparison

            if options.include_insights and data.insights:
                export_data["insights"] = data.insights

            if options.include_metadata and data.metadata:
                export_data["metadata"] = data.metadata

            if data.custom_sections:
                export_data["custom_sections"] = data.custom_sections

            # Round floats
            export_data = self._round_floats(export_data, options.decimal_places)

            # Convert to YAML
            if HAS_YAML:
                yaml_content = yaml.dump(
                    export_data,
                    default_flow_style=options.yaml_default_flow_style,
                    allow_unicode=options.yaml_allow_unicode,
                    sort_keys=False,
                )
            else:
                # Fallback: simple YAML-like format without pyyaml
                yaml_content = self._simple_yaml_dump(export_data)

            # Stream output
            if options.stream_output:
                export_time = (time.perf_counter() - start_time) * 1000
                return ExportResult(
                    success=True,
                    format=ExportFormat.YAML,
                    output_path=None,
                    file_size_bytes=len(yaml_content.encode("utf-8")),
                    export_time_ms=export_time,
                    content=yaml_content,
                    warnings=[] if HAS_YAML else ["PyYAML not installed, using simple YAML format"],
                )

            # File output
            if not options.output_path:
                return ExportResult(
                    success=False,
                    format=ExportFormat.YAML,
                    output_path=None,
                    error="YAML export requires an output path or stream_output=True",
                )

            options.output_path.parent.mkdir(parents=True, exist_ok=True)
            options.output_path.write_text(yaml_content, encoding="utf-8")

            file_size = options.output_path.stat().st_size
            export_time = (time.perf_counter() - start_time) * 1000

            return ExportResult(
                success=True,
                format=ExportFormat.YAML,
                output_path=options.output_path,
                file_size_bytes=file_size,
                export_time_ms=export_time,
                warnings=[] if HAS_YAML else ["PyYAML not installed, using simple YAML format"],
            )

        except Exception as e:
            return ExportResult(
                success=False,
                format=ExportFormat.YAML,
                output_path=options.output_path,
                export_time_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _simple_yaml_dump(self, data: Any, indent: int = 0) -> str:
        """Simple YAML serialization without pyyaml."""
        lines: list[str] = []
        prefix = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)) and value:
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._simple_yaml_dump(value, indent + 1))
                elif isinstance(value, str):
                    # Quote strings that might need it
                    if ":" in value or "\n" in value or value.startswith("-"):
                        lines.append(f'{prefix}{key}: "{value}"')
                    else:
                        lines.append(f"{prefix}{key}: {value}")
                elif value is None:
                    lines.append(f"{prefix}{key}: null")
                elif isinstance(value, bool):
                    lines.append(f"{prefix}{key}: {str(value).lower()}")
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(self._simple_yaml_dump(item, indent + 1))
                elif isinstance(item, str):
                    if ":" in item or "\n" in item:
                        lines.append(f'{prefix}- "{item}"')
                    else:
                        lines.append(f"{prefix}- {item}")
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)


class ExportEngine:
    """Main export engine that handles all export formats.

    Export Formats:
    - CSV:      Simple tabular data (csv stdlib)
    - XLSX:     Multiple sheets, formatting (openpyxl)
    - JSON:     Full data structure (json stdlib)
    - HTML:     Rich formatted reports (jinja2)
    - MARKDOWN: Documentation-friendly format (stdlib)
    - YAML:     Config-friendly format (pyyaml/stdlib)
    """

    def __init__(self) -> None:
        self._exporters: dict[ExportFormat, BaseExporter] = {
            ExportFormat.JSON: JSONExporter(),
            ExportFormat.CSV: CSVExporter(),
            ExportFormat.XLSX: XLSXExporter(),
            ExportFormat.HTML: HTMLExporter(),
            ExportFormat.MARKDOWN: MarkdownExporter(),
            ExportFormat.YAML: YAMLExporter(),
        }

    def register_exporter(self, format: ExportFormat, exporter: BaseExporter) -> None:
        """Register a custom exporter for a format."""
        self._exporters[format] = exporter

    def get_available_formats(self) -> list[ExportFormat]:
        """Get list of available export formats."""
        available = [
            ExportFormat.JSON,
            ExportFormat.CSV,
            ExportFormat.MARKDOWN,
            ExportFormat.YAML,
        ]  # Always available
        if HAS_OPENPYXL:
            available.append(ExportFormat.XLSX)
        available.append(ExportFormat.HTML)  # Always available (with fallback)
        return available

    def export(
        self,
        data: ReportData,
        format: ExportFormat = ExportFormat.JSON,
        output_path: Path | None = None,
        **kwargs: Any,
    ) -> ExportResult:
        """Export data to the specified format.

        Args:
            data: ReportData to export
            format: Export format (JSON, CSV, XLSX, HTML, MARKDOWN, YAML)
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

    def export_to_string(
        self,
        data: ReportData,
        format: ExportFormat = ExportFormat.JSON,
        **kwargs: Any,
    ) -> ExportResult:
        """Export data to a string (stream export).

        Args:
            data: ReportData to export
            format: Export format (JSON, CSV, HTML, MARKDOWN, YAML)
            **kwargs: Additional options passed to ExportOptions

        Returns:
            ExportResult with content field containing the exported string
        """
        return self.export(data, format, stream_output=True, **kwargs)

    def export_to_bytes(
        self,
        data: ReportData,
        format: ExportFormat = ExportFormat.XLSX,
        **kwargs: Any,
    ) -> ExportResult:
        """Export data to bytes (stream export for binary formats).

        Args:
            data: ReportData to export
            format: Export format (XLSX)
            **kwargs: Additional options passed to ExportOptions

        Returns:
            ExportResult with content field containing the exported bytes
        """
        return self.export(data, format, stream_output=True, **kwargs)

    def export_all(
        self,
        data: ReportData,
        output_dir: Path,
        base_name: str = "report",
        formats: list[ExportFormat] | None = None,
    ) -> dict[ExportFormat, ExportResult]:
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

        results: dict[ExportFormat, ExportResult] = {}

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


def export_to_markdown(
    data: ReportData,
    output_path: Path,
    toc: bool = True,
) -> ExportResult:
    """Export data to Markdown format."""
    engine = ExportEngine()
    return engine.export(data, ExportFormat.MARKDOWN, output_path, markdown_toc=toc)


def export_to_yaml(
    data: ReportData,
    output_path: Path,
) -> ExportResult:
    """Export data to YAML format."""
    engine = ExportEngine()
    return engine.export(data, ExportFormat.YAML, output_path)


def export_to_string(
    data: ReportData,
    format: ExportFormat = ExportFormat.JSON,
    **kwargs: Any,
) -> str | None:
    """Export data to a string.

    Returns:
        The exported string, or None if export failed.
    """
    engine = ExportEngine()
    result = engine.export_to_string(data, format, **kwargs)
    if result.success and result.content:
        return result.content if isinstance(result.content, str) else result.content.decode("utf-8")
    return None
