"""Test Report Viewer.

Displays detailed test reports with statistics, charts,
and exportable results.

Part of Phase 7: Advanced Testing & Validation.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.widgets import Static, Button, Label, DataTable, TabbedContent, TabPane
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from rich.text import Text
from rich.table import Table
from rich.console import Console
from rich.panel import Panel


class TestReportSummaryWidget(Static):
    """Widget showing test report summary."""
    
    DEFAULT_CSS = """
    TestReportSummaryWidget {
        width: 100%;
        height: auto;
        padding: 1 2;
        background: $boost;
        border: solid $primary;
    }
    
    TestReportSummaryWidget .summary-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    TestReportSummaryWidget .stat-row {
        width: 100%;
        height: auto;
    }
    
    TestReportSummaryWidget .stat-label {
        width: 20;
        color: $text-muted;
    }
    
    TestReportSummaryWidget .stat-value {
        text-style: bold;
    }
    
    TestReportSummaryWidget .stat-passed {
        color: $success;
    }
    
    TestReportSummaryWidget .stat-failed {
        color: $error;
    }
    """
    
    def __init__(self, report_data: Dict[str, Any], **kwargs):
        """Initialize summary widget.
        
        Args:
            report_data: Test report data
        """
        super().__init__(**kwargs)
        self.report_data = report_data
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        summary = self.report_data.get("summary", {})
        
        yield Static("📊 Test Summary", classes="summary-title")
        
        # Pass rate visualization
        total = summary.get("total_tests", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        pass_rate = summary.get("pass_rate", 0)
        
        # Progress bar
        bar_width = 30
        filled = int(pass_rate / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        bar_text = Text()
        bar_text.append(f"Pass Rate: [{bar[:filled]}", style="green")
        bar_text.append(f"{bar[filled:]}] ", style="dim")
        bar_text.append(f"{pass_rate:.1f}%", style="bold green" if pass_rate >= 80 else "bold yellow")
        
        yield Static(bar_text)
        
        # Stats
        with Horizontal(classes="stat-row"):
            yield Static("Total Tests:", classes="stat-label")
            yield Static(str(total), classes="stat-value")
        
        with Horizontal(classes="stat-row"):
            yield Static("Passed:", classes="stat-label")
            yield Static(str(passed), classes="stat-value stat-passed")
        
        with Horizontal(classes="stat-row"):
            yield Static("Failed:", classes="stat-label")
            yield Static(str(failed), classes="stat-value stat-failed")
        
        with Horizontal(classes="stat-row"):
            yield Static("Duration:", classes="stat-label")
            duration = self.report_data.get("duration_seconds", 0)
            yield Static(f"{duration:.2f}s", classes="stat-value")


class CategoryBreakdownWidget(Static):
    """Widget showing breakdown by category."""
    
    DEFAULT_CSS = """
    CategoryBreakdownWidget {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    CategoryBreakdownWidget .category-row {
        width: 100%;
        height: auto;
        padding: 0 0 1 0;
    }
    
    CategoryBreakdownWidget .category-name {
        width: 25;
        text-style: bold;
    }
    
    CategoryBreakdownWidget .category-bar {
        width: 1fr;
    }
    
    CategoryBreakdownWidget .category-stats {
        width: 15;
        text-align: right;
    }
    """
    
    def __init__(self, report_data: Dict[str, Any], **kwargs):
        """Initialize breakdown widget.
        
        Args:
            report_data: Test report data
        """
        super().__init__(**kwargs)
        self.report_data = report_data
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        categories = self.report_data.get("categories", {})
        
        yield Static("📁 Category Breakdown", classes="section-title")
        
        for name, data in categories.items():
            total = data.get("total", 0)
            passed = data.get("passed", 0)
            failed = data.get("failed", 0)
            
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            with Horizontal(classes="category-row"):
                # Name
                display_name = name.replace("_", " ").title()
                yield Static(display_name, classes="category-name")
                
                # Progress bar
                bar_width = 20
                filled = int(pass_rate / 100 * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                bar_text = Text()
                bar_text.append(bar[:filled], style="green")
                bar_text.append(bar[filled:], style="dim")
                
                yield Static(bar_text, classes="category-bar")
                
                # Stats
                yield Static(f"{passed}/{total}", classes="category-stats")


class FailedTestsWidget(Static):
    """Widget showing failed tests details."""
    
    DEFAULT_CSS = """
    FailedTestsWidget {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    FailedTestsWidget .failed-item {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $error 10%;
        border-left: thick $error;
    }
    
    FailedTestsWidget .failed-name {
        text-style: bold;
        color: $error;
    }
    
    FailedTestsWidget .failed-message {
        color: $text-muted;
        padding-left: 2;
    }
    
    FailedTestsWidget .no-failures {
        color: $success;
        text-style: italic;
        padding: 2;
        text-align: center;
    }
    """
    
    def __init__(self, report_data: Dict[str, Any], **kwargs):
        """Initialize failed tests widget.
        
        Args:
            report_data: Test report data
        """
        super().__init__(**kwargs)
        self.report_data = report_data
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("❌ Failed Tests", classes="section-title")
        
        all_results = self.report_data.get("all_results", [])
        failed_tests = [r for r in all_results if not r.get("passed", True)]
        
        if not failed_tests:
            yield Static("✅ No failed tests!", classes="no-failures")
            return
        
        for test in failed_tests:
            with Vertical(classes="failed-item"):
                name = test.get("name", "Unknown")
                category = test.get("category", "")
                
                yield Static(f"✗ {name} ({category})", classes="failed-name")
                
                error = test.get("error", "")
                message = test.get("message", "")
                
                if error:
                    yield Static(f"Error: {error}", classes="failed-message")
                elif message:
                    yield Static(f"Message: {message}", classes="failed-message")


class PerformanceMetricsWidget(Static):
    """Widget showing performance metrics."""
    
    DEFAULT_CSS = """
    PerformanceMetricsWidget {
        width: 100%;
        height: auto;
        padding: 1;
    }
    
    PerformanceMetricsWidget .metric-row {
        width: 100%;
        height: auto;
        padding: 0 0 1 0;
    }
    
    PerformanceMetricsWidget .metric-name {
        width: 30;
    }
    
    PerformanceMetricsWidget .metric-value {
        text-style: bold;
    }
    
    PerformanceMetricsWidget .metric-fast {
        color: $success;
    }
    
    PerformanceMetricsWidget .metric-slow {
        color: $warning;
    }
    """
    
    def __init__(self, report_data: Dict[str, Any], **kwargs):
        """Initialize metrics widget.
        
        Args:
            report_data: Test report data
        """
        super().__init__(**kwargs)
        self.report_data = report_data
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("⏱️ Performance Metrics", classes="section-title")
        
        all_results = self.report_data.get("all_results", [])
        
        if not all_results:
            yield Static("No performance data available")
            return
        
        # Calculate metrics
        durations = [r.get("duration_ms", 0) for r in all_results if r.get("duration_ms", 0) > 0]
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            total_duration = sum(durations)
            
            metrics = [
                ("Average test duration", f"{avg_duration:.1f}ms"),
                ("Fastest test", f"{min_duration:.1f}ms"),
                ("Slowest test", f"{max_duration:.1f}ms"),
                ("Total test time", f"{total_duration:.1f}ms"),
            ]
            
            for name, value in metrics:
                with Horizontal(classes="metric-row"):
                    yield Static(f"{name}:", classes="metric-name")
                    yield Static(value, classes="metric-value")
        else:
            yield Static("No timing data available")


class TestReportViewer(ModalScreen):
    """Full test report viewer with tabs.
    
    Displays comprehensive test results including:
    - Summary statistics
    - Category breakdown
    - Failed test details
    - Performance metrics
    - Export options
    """
    
    DEFAULT_CSS = """
    TestReportViewer {
        align: center middle;
    }
    
    TestReportViewer #report_container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $primary;
    }
    
    TestReportViewer .header {
        width: 100%;
        height: auto;
        padding: 1 2;
        background: $primary-darken-2;
    }
    
    TestReportViewer .header-title {
        text-style: bold;
        text-align: center;
    }
    
    TestReportViewer .content {
        height: 1fr;
        padding: 1;
    }
    
    TestReportViewer .footer {
        width: 100%;
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
        align: center middle;
    }
    
    TestReportViewer Button {
        margin: 0 1;
    }
    
    TestReportViewer .section-title {
        text-style: bold;
        color: $accent;
        padding: 1 0;
        border-bottom: solid $primary-darken-3;
        margin-bottom: 1;
    }
    """
    
    def __init__(self, report_data: Dict[str, Any], **kwargs):
        """Initialize report viewer.
        
        Args:
            report_data: Test report data
        """
        super().__init__(**kwargs)
        self.report_data = report_data
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container(id="report_container"):
            # Header
            with Horizontal(classes="header"):
                backend_name = self.report_data.get("backend_name", "Unknown")
                timestamp = self.report_data.get("timestamp", "")
                
                yield Static(
                    f"📋 Test Report: {backend_name}",
                    classes="header-title"
                )
            
            # Content with tabs
            with TabbedContent(classes="content"):
                with TabPane("Summary", id="summary_tab"):
                    with ScrollableContainer():
                        yield TestReportSummaryWidget(self.report_data)
                        yield CategoryBreakdownWidget(self.report_data)
                
                with TabPane("Failed Tests", id="failed_tab"):
                    with ScrollableContainer():
                        yield FailedTestsWidget(self.report_data)
                
                with TabPane("Performance", id="performance_tab"):
                    with ScrollableContainer():
                        yield PerformanceMetricsWidget(self.report_data)
                
                with TabPane("All Results", id="all_tab"):
                    yield self._create_results_table()
            
            # Footer
            with Horizontal(classes="footer"):
                yield Button("📥 Export JSON", id="export_json")
                yield Button("📄 Export HTML", id="export_html")
                yield Button("🖨️ Print Report", id="print")
                yield Button("Close", id="close", variant="primary")
    
    def _create_results_table(self) -> DataTable:
        """Create a data table with all results."""
        table = DataTable(id="results_table")
        
        table.add_columns("Status", "Test Name", "Category", "Duration", "Message")
        
        all_results = self.report_data.get("all_results", [])
        
        for result in all_results:
            status = "✓" if result.get("passed", False) else "✗"
            name = result.get("name", "Unknown")
            category = result.get("category", "")
            duration = f"{result.get('duration_ms', 0):.0f}ms"
            message = result.get("message", result.get("error", ""))[:50]
            
            table.add_row(status, name, category, duration, message)
        
        return table
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "export_json":
            self._export_json()
        
        elif button_id == "export_html":
            self._export_html()
        
        elif button_id == "print":
            self._print_report()
        
        elif button_id == "close":
            self.dismiss()
    
    def _export_json(self) -> None:
        """Export report as JSON."""
        backend_id = self.report_data.get("backend_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{backend_id}_test_report_{timestamp}.json"
        output_path = Path.home() / filename
        
        try:
            output_path.write_text(json.dumps(self.report_data, indent=2))
            self.notify(f"Report exported to: {output_path}")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def _export_html(self) -> None:
        """Export report as HTML."""
        backend_id = self.report_data.get("backend_id", "unknown")
        backend_name = self.report_data.get("backend_name", "Unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = self.report_data.get("summary", {})
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Report: {backend_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #333; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Test Report: {backend_name}</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {summary.get('total_tests', 0)}</p>
        <p><strong>Passed:</strong> <span class="passed">{summary.get('passed', 0)}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{summary.get('failed', 0)}</span></p>
        <p><strong>Pass Rate:</strong> {summary.get('pass_rate', 0):.1f}%</p>
    </div>
    
    <h2>All Results</h2>
    <table>
        <tr>
            <th>Status</th>
            <th>Test Name</th>
            <th>Category</th>
            <th>Duration</th>
            <th>Message</th>
        </tr>
"""
        
        for result in self.report_data.get("all_results", []):
            status = "✓" if result.get("passed", False) else "✗"
            status_class = "passed" if result.get("passed", False) else "failed"
            
            html_content += f"""        <tr>
            <td class="{status_class}">{status}</td>
            <td>{result.get('name', 'Unknown')}</td>
            <td>{result.get('category', '')}</td>
            <td>{result.get('duration_ms', 0):.0f}ms</td>
            <td>{result.get('message', result.get('error', ''))}</td>
        </tr>
"""
        
        html_content += """    </table>
</body>
</html>"""
        
        filename = f"{backend_id}_test_report_{timestamp}.html"
        output_path = Path.home() / filename
        
        try:
            output_path.write_text(html_content)
            self.notify(f"HTML report exported to: {output_path}")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def _print_report(self) -> None:
        """Print report to console."""
        console = Console()
        
        backend_name = self.report_data.get("backend_name", "Unknown")
        summary = self.report_data.get("summary", {})
        
        # Create summary table
        table = Table(title=f"Test Report: {backend_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Tests", str(summary.get("total_tests", 0)))
        table.add_row("Passed", str(summary.get("passed", 0)))
        table.add_row("Failed", str(summary.get("failed", 0)))
        table.add_row("Pass Rate", f"{summary.get('pass_rate', 0):.1f}%")
        
        console.print(table)
        
        self.notify("Report printed to console")


class TestResultDetailsDialog(ModalScreen):
    """Dialog showing detailed results for a single test."""
    
    DEFAULT_CSS = """
    TestResultDetailsDialog {
        align: center middle;
    }
    
    TestResultDetailsDialog #details_container {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 2;
    }
    
    TestResultDetailsDialog .detail-row {
        width: 100%;
        height: auto;
        padding: 0 0 1 0;
    }
    
    TestResultDetailsDialog .detail-label {
        width: 15;
        text-style: bold;
        color: $text-muted;
    }
    
    TestResultDetailsDialog .detail-value {
        width: 1fr;
    }
    
    TestResultDetailsDialog .error-text {
        color: $error;
        padding: 1;
        background: $error 10%;
        border: solid $error;
    }
    """
    
    def __init__(self, result: Dict[str, Any], **kwargs):
        """Initialize details dialog.
        
        Args:
            result: Single test result
        """
        super().__init__(**kwargs)
        self.result = result
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container(id="details_container"):
            name = self.result.get("name", "Unknown")
            passed = self.result.get("passed", False)
            
            status = "✓ PASSED" if passed else "✗ FAILED"
            status_style = "green" if passed else "red"
            
            yield Static(Text(f"{name}: {status}", style=f"bold {status_style}"))
            
            yield Static("─" * 50)
            
            # Details
            with Horizontal(classes="detail-row"):
                yield Static("Category:", classes="detail-label")
                yield Static(self.result.get("category", ""), classes="detail-value")
            
            with Horizontal(classes="detail-row"):
                yield Static("Duration:", classes="detail-label")
                yield Static(f"{self.result.get('duration_ms', 0):.0f}ms", classes="detail-value")
            
            with Horizontal(classes="detail-row"):
                yield Static("Status:", classes="detail-label")
                yield Static(self.result.get("status", "unknown"), classes="detail-value")
            
            if self.result.get("message"):
                with Horizontal(classes="detail-row"):
                    yield Static("Message:", classes="detail-label")
                    yield Static(self.result.get("message", ""), classes="detail-value")
            
            if self.result.get("error"):
                yield Static("Error:", classes="detail-label")
                yield Static(self.result.get("error", ""), classes="error-text")
            
            yield Static("─" * 50)
            
            yield Button("Close", id="close", variant="primary")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "close":
            self.dismiss()
