"""Test Results Display Widget.

Provides a rich display widget for test results in the TUI.
Shows real-time progress, category summaries, and detailed results.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, ProgressBar, Label
from textual.reactive import reactive
from textual.widget import Widget

from rich.text import Text
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table


class TestProgressBar(Static):
    """Progress bar for test execution."""
    
    progress = reactive(0.0)
    
    def __init__(
        self,
        label: str = "Progress",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.label = label
        self._width = 30
    
    def render(self) -> RenderableType:
        """Render the progress bar."""
        filled = int(self.progress / 100 * self._width)
        empty = self._width - filled
        
        bar = "█" * filled + "░" * empty
        percentage = f"{self.progress:.0f}%"
        
        return Text.assemble(
            (f"{self.label}: ", "bold"),
            (f"[{bar}]", "cyan"),
            (f" {percentage}", "dim"),
        )
    
    def set_progress(self, value: float) -> None:
        """Set progress value (0-100)."""
        self.progress = max(0, min(100, value))


class TestCategoryWidget(Static):
    """Widget displaying a test category status."""
    
    DEFAULT_CSS = """
    TestCategoryWidget {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """
    
    def __init__(
        self,
        category_name: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.category_name = category_name
        self._tests: List[Dict[str, Any]] = []
        self._status = "pending"  # pending, running, passed, failed
    
    def render(self) -> RenderableType:
        """Render the category status."""
        # Status icon
        icons = {
            "pending": "⚪",
            "running": "🔄",
            "passed": "✅",
            "failed": "❌",
            "partial": "⚠️",
        }
        icon = icons.get(self._status, "⚪")
        
        # Build display
        lines = [f"{icon} {self.category_name}"]
        
        # Show test results
        for test in self._tests[-5:]:  # Show last 5
            t_status = test.get("status", "pending")
            t_name = test.get("name", "Unknown")
            t_message = test.get("message", "")[:40]
            
            if t_status == "passed":
                lines.append(f"  [green]✓[/green] {t_name}")
            elif t_status in ("failed", "error"):
                lines.append(f"  [red]✗[/red] {t_name}")
                if t_message:
                    lines.append(f"    [dim]{t_message}[/dim]")
            elif t_status == "running":
                lines.append(f"  [yellow]▶[/yellow] {t_name}")
            elif t_status == "skipped":
                lines.append(f"  [dim]○[/dim] {t_name}")
        
        # Show count if more tests
        remaining = len(self._tests) - 5
        if remaining > 0:
            lines.append(f"  [dim]...and {remaining} more[/dim]")
        
        return Text.from_markup("\n".join(lines))
    
    def set_status(self, status: str) -> None:
        """Set category status."""
        self._status = status
        self.refresh()
    
    def add_test(self, test: Dict[str, Any]) -> None:
        """Add a test result."""
        self._tests.append(test)
        
        # Update status
        has_failed = any(t.get("status") in ("failed", "error") for t in self._tests)
        all_passed = all(t.get("status") in ("passed", "skipped") for t in self._tests)
        
        if has_failed:
            self._status = "failed" if all_passed else "partial"
        elif all_passed:
            self._status = "passed"
        else:
            self._status = "running"
        
        self.refresh()
    
    def clear_tests(self) -> None:
        """Clear all tests."""
        self._tests = []
        self._status = "pending"
        self.refresh()


class TestResultItem(Static):
    """Widget for a single test result."""
    
    DEFAULT_CSS = """
    TestResultItem {
        height: auto;
        padding: 0 1;
    }
    
    TestResultItem.passed {
        color: $success;
    }
    
    TestResultItem.failed {
        color: $error;
    }
    
    TestResultItem.running {
        color: $warning;
    }
    """
    
    def __init__(
        self,
        name: str,
        status: str = "pending",
        message: str = "",
        duration_ms: float = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.test_name = name
        self.test_status = status
        self.test_message = message
        self.duration_ms = duration_ms
    
    def render(self) -> RenderableType:
        """Render the test result."""
        icons = {
            "pending": "⚪",
            "running": "▶",
            "passed": "✓",
            "failed": "✗",
            "error": "💥",
            "skipped": "○",
        }
        icon = icons.get(self.test_status, "?")
        
        duration = f" ({self.duration_ms:.0f}ms)" if self.duration_ms > 0 else ""
        
        if self.test_status == "passed":
            return Text.from_markup(f"[green]{icon}[/green] {self.test_name}{duration}")
        elif self.test_status in ("failed", "error"):
            lines = [f"[red]{icon}[/red] {self.test_name}{duration}"]
            if self.test_message:
                lines.append(f"  [dim]{self.test_message}[/dim]")
            return Text.from_markup("\n".join(lines))
        elif self.test_status == "running":
            return Text.from_markup(f"[yellow]{icon}[/yellow] {self.test_name}...")
        else:
            return Text.from_markup(f"[dim]{icon} {self.test_name}[/dim]")
    
    def update_status(
        self,
        status: str,
        message: str = "",
        duration_ms: float = 0
    ) -> None:
        """Update the test status."""
        self.test_status = status
        self.test_message = message
        self.duration_ms = duration_ms
        self.set_class(self.test_status == "passed", "passed")
        self.set_class(self.test_status in ("failed", "error"), "failed")
        self.set_class(self.test_status == "running", "running")
        self.refresh()


class TestSummaryWidget(Static):
    """Widget displaying overall test summary."""
    
    DEFAULT_CSS = """
    TestSummaryWidget {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin: 1 0;
    }
    
    TestSummaryWidget.success {
        border: solid $success;
        background: $success-darken-3;
    }
    
    TestSummaryWidget.warning {
        border: solid $warning;
        background: $warning-darken-3;
    }
    
    TestSummaryWidget.error {
        border: solid $error;
        background: $error-darken-3;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._total = 0
        self._passed = 0
        self._failed = 0
        self._errors = 0
        self._skipped = 0
        self._duration = 0.0
        self._complete = False
    
    def render(self) -> RenderableType:
        """Render the summary."""
        if not self._complete:
            return Text.from_markup(
                "[bold]Running tests...[/bold]\n"
                f"Completed: {self._passed + self._failed + self._errors + self._skipped}/{self._total}"
            )
        
        pass_rate = (self._passed / self._total * 100) if self._total > 0 else 0
        
        if self._failed == 0 and self._errors == 0:
            icon = "✅"
            status = "All tests passed!"
        elif self._passed >= self._total * 0.7:
            icon = "⚠️"
            status = "Some tests failed"
        else:
            icon = "❌"
            status = "Tests failed"
        
        return Text.from_markup(
            f"[bold]{icon} {status}[/bold]\n\n"
            f"Total: {self._total} | "
            f"[green]Passed: {self._passed}[/green] | "
            f"[red]Failed: {self._failed}[/red] | "
            f"[yellow]Errors: {self._errors}[/yellow]\n"
            f"Pass Rate: {pass_rate:.1f}% | Duration: {self._duration:.0f}ms"
        )
    
    def update_stats(
        self,
        total: int = 0,
        passed: int = 0,
        failed: int = 0,
        errors: int = 0,
        skipped: int = 0,
        duration: float = 0.0,
        complete: bool = False
    ) -> None:
        """Update summary statistics."""
        self._total = total
        self._passed = passed
        self._failed = failed
        self._errors = errors
        self._skipped = skipped
        self._duration = duration
        self._complete = complete
        
        # Update styling
        self.remove_class("success", "warning", "error")
        if complete:
            if failed == 0 and errors == 0:
                self.add_class("success")
            elif passed >= total * 0.7:
                self.add_class("warning")
            else:
                self.add_class("error")
        
        self.refresh()


class TestResultsDisplay(Widget):
    """Complete test results display widget.
    
    Shows:
    - Progress bar
    - Current test status
    - Category summaries
    - Detailed results
    - Overall summary
    """
    
    DEFAULT_CSS = """
    TestResultsDisplay {
        height: auto;
        padding: 1;
    }
    
    TestResultsDisplay .header {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    TestResultsDisplay .results-container {
        height: auto;
        max-height: 30;
        overflow-y: auto;
        border: solid $primary-darken-2;
        padding: 1;
        margin: 1 0;
    }
    
    TestResultsDisplay .current-test {
        color: $warning;
        margin: 1 0;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._categories: Dict[str, TestCategoryWidget] = {}
        self._results: List[Dict[str, Any]] = []
        self._current_test: str = ""
        self._progress: float = 0.0
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("Test Results:", classes="header")
        
        yield TestProgressBar(id="progress_bar")
        
        yield Static("", id="current_test", classes="current-test")
        
        with ScrollableContainer(classes="results-container", id="results_scroll"):
            yield Vertical(id="categories_container")
        
        yield TestSummaryWidget(id="summary")
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        self.query_one("#summary", TestSummaryWidget).update_stats()
    
    def set_progress(self, progress: float, current_test: str = "") -> None:
        """Update progress.
        
        Args:
            progress: Progress percentage (0-100)
            current_test: Name of current test
        """
        self._progress = progress
        self._current_test = current_test
        
        self.query_one("#progress_bar", TestProgressBar).set_progress(progress)
        
        if current_test:
            status = "▶ Running: " + current_test
        else:
            status = ""
        self.query_one("#current_test", Static).update(status)
    
    def add_category(self, category_name: str) -> None:
        """Add a test category.
        
        Args:
            category_name: Name of the category
        """
        if category_name not in self._categories:
            widget = TestCategoryWidget(category_name, id=f"cat_{category_name}")
            self._categories[category_name] = widget
            
            container = self.query_one("#categories_container", Vertical)
            container.mount(widget)
    
    def add_result(
        self,
        category: str,
        name: str,
        status: str,
        message: str = "",
        duration_ms: float = 0,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a test result.
        
        Args:
            category: Test category name
            name: Test name
            status: Test status (passed, failed, error, skipped)
            message: Status message
            duration_ms: Test duration in milliseconds
            details: Optional additional details
        """
        result = {
            "category": category,
            "name": name,
            "status": status,
            "message": message,
            "duration_ms": duration_ms,
            "details": details or {},
        }
        self._results.append(result)
        
        # Add to category widget
        self.add_category(category)
        self._categories[category].add_test(result)
    
    def set_summary(
        self,
        total: int,
        passed: int,
        failed: int,
        errors: int = 0,
        skipped: int = 0,
        duration: float = 0.0,
        complete: bool = False
    ) -> None:
        """Set summary statistics.
        
        Args:
            total: Total number of tests
            passed: Number of passed tests
            failed: Number of failed tests
            errors: Number of error tests
            skipped: Number of skipped tests
            duration: Total duration in milliseconds
            complete: Whether testing is complete
        """
        self.query_one("#summary", TestSummaryWidget).update_stats(
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration=duration,
            complete=complete,
        )
    
    def clear(self) -> None:
        """Clear all results."""
        self._results = []
        
        # Clear categories
        container = self.query_one("#categories_container", Vertical)
        container.remove_children()
        self._categories = {}
        
        # Reset progress
        self.set_progress(0)
        
        # Reset summary
        self.query_one("#summary", TestSummaryWidget).update_stats()
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all test results.
        
        Returns:
            List of result dictionaries
        """
        return self._results.copy()
    
    def all_passed(self) -> bool:
        """Check if all tests passed.
        
        Returns:
            True if all tests passed
        """
        if not self._results:
            return False
        return all(
            r.get("status") in ("passed", "skipped")
            for r in self._results
        )


class MeasurementResultsWidget(Static):
    """Widget for displaying quantum measurement results."""
    
    DEFAULT_CSS = """
    MeasurementResultsWidget {
        height: auto;
        padding: 1;
        border: solid $accent;
        margin: 1 0;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._measurements: Dict[str, int] = {}
        self._total_shots: int = 0
    
    def render(self) -> RenderableType:
        """Render measurement results."""
        if not self._measurements:
            return Text("[dim]No measurement results[/dim]")
        
        lines = ["[bold]Measurement Results:[/bold]", ""]
        
        # Sort by count descending
        sorted_results = sorted(
            self._measurements.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for state, count in sorted_results:
            percentage = (count / self._total_shots * 100) if self._total_shots > 0 else 0
            bar_width = int(percentage / 5)  # 20 char max
            bar = "█" * bar_width + "░" * (20 - bar_width)
            
            lines.append(f"  |{state}⟩: [{bar}] {count} ({percentage:.1f}%)")
        
        lines.append("")
        lines.append(f"[dim]Total shots: {self._total_shots}[/dim]")
        
        return Text.from_markup("\n".join(lines))
    
    def set_measurements(
        self,
        measurements: Dict[str, int],
        total_shots: int
    ) -> None:
        """Set measurement results.
        
        Args:
            measurements: Dict of state -> count
            total_shots: Total number of shots
        """
        self._measurements = measurements
        self._total_shots = total_shots
        self.refresh()
    
    def clear(self) -> None:
        """Clear measurements."""
        self._measurements = {}
        self._total_shots = 0
        self.refresh()
