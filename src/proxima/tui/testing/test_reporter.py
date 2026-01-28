"""Test Reporter.

Provides comprehensive test result reporting and formatting.
Generates human-readable reports and machine-parseable summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from .test_executor import TestResult, TestStatus, TestCategory


@dataclass
class TestSummary:
    """Summary of test results."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_ms: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100
    
    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0
    
    @property
    def status_emoji(self) -> str:
        """Get status emoji."""
        if self.all_passed:
            return "✅"
        elif self.failed > 0 or self.errors > 0:
            return "❌"
        else:
            return "⚠️"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_ms": self.duration_ms,
            "pass_rate": self.pass_rate,
            "all_passed": self.all_passed,
        }


@dataclass
class CategorySummary:
    """Summary for a test category."""
    category: TestCategory
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def passed(self) -> int:
        """Count passed tests."""
        return sum(1 for t in self.tests if t.passed)
    
    @property
    def total(self) -> int:
        """Count total tests."""
        return len(self.tests)
    
    @property
    def all_passed(self) -> bool:
        """Check if all tests in category passed."""
        return all(t.passed or t.status == TestStatus.SKIPPED for t in self.tests)
    
    @property
    def status_emoji(self) -> str:
        """Get status emoji for category."""
        if self.all_passed:
            return "✅"
        elif any(t.status == TestStatus.ERROR for t in self.tests):
            return "💥"
        elif any(t.status == TestStatus.FAILED for t in self.tests):
            return "❌"
        else:
            return "⚠️"


@dataclass
class TestReport:
    """Complete test report."""
    backend_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    summary: TestSummary = field(default_factory=TestSummary)
    categories: Dict[TestCategory, CategorySummary] = field(default_factory=dict)
    results: List[TestResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend_name": self.backend_name,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary.to_dict(),
            "categories": {
                cat.value: {
                    "passed": summary.passed,
                    "total": summary.total,
                    "all_passed": summary.all_passed,
                }
                for cat, summary in self.categories.items()
            },
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Test Report: {self.backend_name}",
            "",
            f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tests | {self.summary.total_tests} |",
            f"| Passed | {self.summary.passed} |",
            f"| Failed | {self.summary.failed} |",
            f"| Errors | {self.summary.errors} |",
            f"| Skipped | {self.summary.skipped} |",
            f"| Pass Rate | {self.summary.pass_rate:.1f}% |",
            f"| Duration | {self.summary.duration_ms:.1f}ms |",
            "",
            f"**Overall Status:** {self.summary.status_emoji} {'PASSED' if self.summary.all_passed else 'FAILED'}",
            "",
            "## Results by Category",
            "",
        ]
        
        for category, cat_summary in self.categories.items():
            lines.extend([
                f"### {cat_summary.status_emoji} {category.value.replace('_', ' ').title()}",
                "",
                f"Passed: {cat_summary.passed}/{cat_summary.total}",
                "",
            ])
            
            for test in cat_summary.tests:
                status = "✅" if test.passed else "❌" if test.status == TestStatus.FAILED else "⚠️"
                lines.append(f"- {status} **{test.name}**: {test.message}")
                if test.details:
                    for key, value in test.details.items():
                        lines.append(f"  - {key}: {value}")
            
            lines.append("")
        
        # Failed tests detail
        failed_tests = [r for r in self.results if r.status in (TestStatus.FAILED, TestStatus.ERROR)]
        if failed_tests:
            lines.extend([
                "## Failed Tests Detail",
                "",
            ])
            for test in failed_tests:
                lines.extend([
                    f"### ❌ {test.name}",
                    "",
                    f"**Category:** {test.category.value}",
                    f"**Message:** {test.message}",
                    "",
                ])
                if test.error_traceback:
                    lines.extend([
                        "**Traceback:**",
                        "```",
                        test.error_traceback,
                        "```",
                        "",
                    ])
        
        return "\n".join(lines)


class TestReporter:
    """Generate and format test reports.
    
    Provides multiple output formats for test results including
    console display, markdown, JSON, and HTML.
    """
    
    def __init__(self, backend_name: str):
        """Initialize reporter.
        
        Args:
            backend_name: Name of the backend being tested
        """
        self.backend_name = backend_name
        self._results: List[TestResult] = []
        self._metadata: Dict[str, Any] = {}
    
    def add_result(self, result: TestResult) -> None:
        """Add a test result.
        
        Args:
            result: Test result to add
        """
        self._results.append(result)
    
    def add_results(self, results: List[TestResult]) -> None:
        """Add multiple test results.
        
        Args:
            results: List of test results
        """
        self._results.extend(results)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
    
    def generate_report(self) -> TestReport:
        """Generate a complete test report.
        
        Returns:
            TestReport with all results
        """
        # Calculate summary
        summary = TestSummary(
            total_tests=len(self._results),
            passed=sum(1 for r in self._results if r.passed),
            failed=sum(1 for r in self._results if r.status == TestStatus.FAILED),
            errors=sum(1 for r in self._results if r.status == TestStatus.ERROR),
            skipped=sum(1 for r in self._results if r.status == TestStatus.SKIPPED),
            duration_ms=sum(r.duration_ms for r in self._results),
        )
        
        # Group by category
        categories: Dict[TestCategory, CategorySummary] = {}
        for result in self._results:
            if result.category not in categories:
                categories[result.category] = CategorySummary(
                    category=result.category
                )
            categories[result.category].tests.append(result)
        
        return TestReport(
            backend_name=self.backend_name,
            summary=summary,
            categories=categories,
            results=self._results,
            metadata=self._metadata,
        )
    
    def format_console(self) -> str:
        """Format results for console display.
        
        Returns:
            Formatted string for console output
        """
        report = self.generate_report()
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append(f"  TEST RESULTS: {self.backend_name}")
        lines.append("=" * 60)
        lines.append("")
        
        # Category summaries
        for category, cat_summary in report.categories.items():
            status = cat_summary.status_emoji
            lines.append(f"{status} {category.value.upper()}: {cat_summary.passed}/{cat_summary.total} passed")
            
            for test in cat_summary.tests:
                if test.status == TestStatus.PASSED:
                    icon = "  ✓"
                elif test.status == TestStatus.FAILED:
                    icon = "  ✗"
                elif test.status == TestStatus.ERROR:
                    icon = "  💥"
                else:
                    icon = "  ○"
                
                duration = f"({test.duration_ms:.0f}ms)" if test.duration_ms > 0 else ""
                lines.append(f"{icon} {test.name} {duration}")
                
                if test.status in (TestStatus.FAILED, TestStatus.ERROR) and test.message:
                    lines.append(f"      └─ {test.message[:60]}")
            
            lines.append("")
        
        # Summary
        lines.append("-" * 60)
        s = report.summary
        lines.append(f"  Total: {s.total_tests} | Passed: {s.passed} | Failed: {s.failed} | Errors: {s.errors}")
        lines.append(f"  Pass Rate: {s.pass_rate:.1f}% | Duration: {s.duration_ms:.0f}ms")
        lines.append("-" * 60)
        lines.append("")
        
        if s.all_passed:
            lines.append("  ✅ ALL TESTS PASSED")
        else:
            lines.append("  ❌ SOME TESTS FAILED")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def format_json(self, indent: int = 2) -> str:
        """Format results as JSON.
        
        Args:
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        import json
        report = self.generate_report()
        return json.dumps(report.to_dict(), indent=indent)
    
    def format_summary_line(self) -> str:
        """Format a single-line summary.
        
        Returns:
            Single line summary string
        """
        report = self.generate_report()
        s = report.summary
        emoji = s.status_emoji
        return f"{emoji} {s.passed}/{s.total_tests} tests passed ({s.pass_rate:.0f}%) in {s.duration_ms:.0f}ms"
    
    def format_rich(self) -> str:
        """Format results with Rich markup for TUI display.
        
        Returns:
            Rich-formatted string
        """
        report = self.generate_report()
        lines = []
        
        # Summary box
        s = report.summary
        if s.all_passed:
            lines.append("[bold green]✓ All tests passed![/bold green]")
        else:
            lines.append("[bold red]✗ Some tests failed[/bold red]")
        
        lines.append("")
        lines.append(f"[dim]Tests: {s.total_tests} | Passed: {s.passed} | Failed: {s.failed}[/dim]")
        lines.append(f"[dim]Duration: {s.duration_ms:.0f}ms[/dim]")
        lines.append("")
        
        # Results by category
        for category, cat_summary in report.categories.items():
            cat_name = category.value.replace("_", " ").title()
            
            if cat_summary.all_passed:
                lines.append(f"[green]{cat_summary.status_emoji} {cat_name}[/green] ({cat_summary.passed}/{cat_summary.total})")
            else:
                lines.append(f"[red]{cat_summary.status_emoji} {cat_name}[/red] ({cat_summary.passed}/{cat_summary.total})")
            
            for test in cat_summary.tests:
                if test.passed:
                    lines.append(f"  [green]✓[/green] {test.name}")
                elif test.status == TestStatus.SKIPPED:
                    lines.append(f"  [yellow]○[/yellow] {test.name} [dim](skipped)[/dim]")
                else:
                    lines.append(f"  [red]✗[/red] {test.name}")
                    if test.message:
                        lines.append(f"    [dim]{test.message[:50]}[/dim]")
        
        return "\n".join(lines)
    
    def get_failures(self) -> List[TestResult]:
        """Get all failed tests.
        
        Returns:
            List of failed test results
        """
        return [
            r for r in self._results
            if r.status in (TestStatus.FAILED, TestStatus.ERROR)
        ]
    
    def get_summary(self) -> TestSummary:
        """Get test summary.
        
        Returns:
            TestSummary object
        """
        return TestSummary(
            total_tests=len(self._results),
            passed=sum(1 for r in self._results if r.passed),
            failed=sum(1 for r in self._results if r.status == TestStatus.FAILED),
            errors=sum(1 for r in self._results if r.status == TestStatus.ERROR),
            skipped=sum(1 for r in self._results if r.status == TestStatus.SKIPPED),
            duration_ms=sum(r.duration_ms for r in self._results),
        )
    
    def save_report(self, path: str, format: str = "markdown") -> None:
        """Save report to file.
        
        Args:
            path: Output file path
            format: Output format ('markdown', 'json', 'text')
        """
        from pathlib import Path
        
        if format == "markdown":
            content = self.generate_report().to_markdown()
        elif format == "json":
            content = self.format_json()
        else:
            content = self.format_console()
        
        Path(path).write_text(content)
