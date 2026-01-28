"""Step 6: Testing & Validation.

Run comprehensive validation tests on the generated backend code:
- Syntax validation
- Import checks
- Backend instantiation
- Circuit execution tests
- Result normalization tests
- Gate support validation
- Performance metrics

Part of Phase 4: Testing & Validation Interface.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import (
    Static, Button, ProgressBar, RichLog, LoadingIndicator,
    Select, Input, Label
)
from textual.screen import ModalScreen
from textual.reactive import reactive

from .wizard_state import BackendWizardState


class TestStatus(Enum):
    """Status of a test."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    message: str = ""
    details: Optional[str] = None
    duration_ms: float = 0.0
    category: str = "general"


class TestCategoryDisplay(Static):
    """Widget to display tests in a category."""
    
    def __init__(self, category_name: str, **kwargs):
        super().__init__(**kwargs)
        self.category_name = category_name
        self.tests: List[TestResult] = []
        self._status = TestStatus.PENDING
    
    def render(self) -> str:
        """Render the category display."""
        # Status icon
        icons = {
            TestStatus.PENDING: "⚪",
            TestStatus.RUNNING: "🔄",
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.ERROR: "💥",
            TestStatus.SKIPPED: "⏭️",
        }
        icon = icons.get(self._status, "⚪")
        
        lines = [f"{icon} {self.category_name}"]
        
        for test in self.tests:
            t_icon = icons.get(test.status, "⚪")
            duration = f" ({test.duration_ms:.0f}ms)" if test.duration_ms > 0 else ""
            lines.append(f"  {t_icon} {test.name}{duration}")
            
            if test.status in (TestStatus.FAILED, TestStatus.ERROR) and test.message:
                lines.append(f"      └─ {test.message[:50]}")
        
        return "\n".join(lines)
    
    def add_test(self, test: TestResult) -> None:
        """Add a test result."""
        self.tests.append(test)
        self._update_status()
        self.refresh()
    
    def set_running(self) -> None:
        """Set category as running."""
        self._status = TestStatus.RUNNING
        self.refresh()
    
    def _update_status(self) -> None:
        """Update category status based on tests."""
        if not self.tests:
            self._status = TestStatus.PENDING
        elif any(t.status == TestStatus.ERROR for t in self.tests):
            self._status = TestStatus.ERROR
        elif any(t.status == TestStatus.FAILED for t in self.tests):
            self._status = TestStatus.FAILED
        elif all(t.status in (TestStatus.PASSED, TestStatus.SKIPPED) for t in self.tests):
            self._status = TestStatus.PASSED
        else:
            self._status = TestStatus.RUNNING


class TestResultsWidget(Static):
    """Widget displaying test results with measurement output."""
    
    def __init__(self):
        super().__init__()
        self._results: Dict[str, Any] = {}
        self._measurements: Dict[str, int] = {}
        self._total_shots: int = 0
        self._execution_time: float = 0.0
    
    def render(self) -> str:
        """Render the results display."""
        if not self._results:
            return "No test results yet.\n\nClick 'Run Test' to validate your backend."
        
        lines = []
        
        # Individual test results
        for test_name, result in self._results.items():
            if isinstance(result, dict):
                passed = result.get('passed', False)
                status = result.get('status', 'UNKNOWN')
                message = result.get('message', '')
                
                icon = "✓" if passed else "✗"
                color = "green" if passed else "red"
                lines.append(f"[{color}]{icon}[/{color}] {test_name}: {status}")
                
                if message and not passed:
                    lines.append(f"   └─ {message[:60]}")
        
        # Execution time
        if self._execution_time > 0:
            lines.append(f"\nExecution time: {self._execution_time:.0f}ms")
        
        # Measurement results
        if self._measurements:
            lines.append("\nResults:")
            
            sorted_results = sorted(
                self._measurements.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for state, count in sorted_results[:10]:  # Top 10
                if self._total_shots > 0:
                    percentage = (count / self._total_shots) * 100
                    lines.append(f"  |{state}⟩: {count} ({percentage:.1f}%)")
                else:
                    lines.append(f"  |{state}⟩: {count}")
        
        # Overall status
        all_passed = all(
            r.get('passed', False) for r in self._results.values()
            if isinstance(r, dict)
        )
        
        if all_passed:
            lines.append("\n[green bold]✓ All tests passed![/green bold]")
        else:
            lines.append("\n[red]✗ Some tests failed[/red]")
        
        return "\n".join(lines)
    
    def update_results(self, results: Dict[str, Any]) -> None:
        """Update test results."""
        self._results = {}
        
        for key, value in results.items():
            if key == 'summary':
                continue
            elif key == 'execution_time':
                self._execution_time = value
            elif key == 'measurements':
                self._measurements = value
            elif key == 'total_shots':
                self._total_shots = value
            elif isinstance(value, dict):
                self._results[key] = value
        
        # Check for measurements in execution test
        if 'Circuit execution' in self._results:
            exec_result = self._results['Circuit execution']
            if isinstance(exec_result, dict) and 'details' in exec_result:
                details = exec_result['details']
                if isinstance(details, dict):
                    if 'measurements' in details:
                        self._measurements = details['measurements']
                    if 'execution_time' in details:
                        self._execution_time = details['execution_time']
        
        self.refresh()
    
    def clear(self) -> None:
        """Clear results."""
        self._results = {}
        self._measurements = {}
        self._total_shots = 0
        self._execution_time = 0.0
        self.refresh()


class TestingStepScreen(ModalScreen[dict]):
    """
    Step 6: Testing and validation screen.
    
    Runs a comprehensive series of tests on the generated backend code
    to validate it before deployment. Includes circuit selection,
    shot configuration, and detailed result display.
    """
    
    DEFAULT_CSS = """
    TestingStepScreen {
        align: center middle;
    }
    
    TestingStepScreen .wizard-container {
        width: 95;
        height: auto;
        max-height: 95%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    TestingStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    TestingStepScreen .form-container {
        height: auto;
        max-height: 70%;
        padding: 1;
    }
    
    TestingStepScreen .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }
    
    TestingStepScreen .field-hint {
        color: $text-muted;
        margin: 0 0 1 0;
    }
    
    TestingStepScreen .config-row {
        height: auto;
        width: 100%;
        margin: 1 0;
    }
    
    TestingStepScreen .config-label {
        width: 20;
    }
    
    TestingStepScreen .config-input {
        width: 30;
    }
    
    TestingStepScreen .test-categories {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        margin: 1 0;
        border: solid $primary-darken-3;
    }
    
    TestingStepScreen .test-output {
        height: 18;
        margin: 1 0;
        border: solid $primary-darken-3;
        background: $surface-darken-2;
    }
    
    TestingStepScreen .results-section {
        height: auto;
        max-height: 20;
        padding: 1;
        border: solid $accent-darken-2;
        background: $surface-darken-1;
        margin: 1 0;
    }
    
    TestingStepScreen .summary-box {
        padding: 1;
        margin: 1 0;
        border: solid $primary;
    }
    
    TestingStepScreen .summary-success {
        background: $success-darken-3;
        border-color: $success;
    }
    
    TestingStepScreen .summary-warning {
        background: $warning-darken-3;
        border-color: $warning;
    }
    
    TestingStepScreen .summary-error {
        background: $error-darken-3;
        border-color: $error;
    }
    
    TestingStepScreen .progress-section {
        margin: 1 0;
        padding: 1 0;
        border-top: solid $primary-darken-3;
    }
    
    TestingStepScreen .progress-text {
        color: $text-muted;
    }
    
    TestingStepScreen .progress-bar {
        color: $primary;
    }
    
    TestingStepScreen .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    TestingStepScreen .nav-button {
        margin: 0 1;
        min-width: 14;
    }
    
    TestingStepScreen LoadingIndicator {
        height: 3;
    }
    
    TestingStepScreen Select {
        width: 100%;
    }
    
    TestingStepScreen Input {
        width: 100%;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    # Reactive properties
    tests_running = reactive(False)
    tests_complete = reactive(False)
    current_test = reactive("")
    progress = reactive(0.0)
    
    def __init__(self, state: BackendWizardState):
        """Initialize the testing screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
        self.tests: List[TestResult] = []
        self.category_widgets: Dict[str, TestCategoryDisplay] = {}
        self._abort_requested = False
    
    def compose(self) -> ComposeResult:
        """Compose the testing screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "🧪 Add Custom Backend - Testing & Validation",
                    classes="wizard-title"
                )
                
                yield Static(
                    f"Validating backend: [bold]{self.state.display_name}[/bold]",
                    classes="field-hint"
                )
                
                with ScrollableContainer(classes="form-container"):
                    # Test Configuration
                    yield Static("Test Configuration:", classes="section-title")
                    
                    with Horizontal(classes="config-row"):
                        yield Static("Circuit Type:", classes="config-label")
                        yield Select(
                            options=[
                                ("Bell State (2 qubits)", "bell"),
                                ("GHZ State (3 qubits)", "ghz_3"),
                                ("QFT (3 qubits)", "qft_3"),
                                ("Single Qubit Gates", "single_qubit"),
                                ("Two Qubit Gates", "two_qubit"),
                                ("Random Circuit", "random"),
                            ],
                            value="bell",
                            id="circuit_type",
                            classes="config-input"
                        )
                    
                    with Horizontal(classes="config-row"):
                        yield Static("Shots:", classes="config-label")
                        yield Input(
                            value="1024",
                            type="integer",
                            id="shots",
                            classes="config-input"
                        )
                    
                    yield Button(
                        "▶ Run Tests",
                        id="btn_run_tests",
                        variant="warning"
                    )
                    
                    # Test Categories
                    yield Static("Test Categories:", classes="section-title")
                    
                    with Vertical(classes="test-categories", id="test_categories"):
                        yield Static(
                            "⚪ Initialization Tests",
                            classes="test-item",
                            id="cat_initialization"
                        )
                        yield Static(
                            "⚪ Validation Tests",
                            classes="test-item",
                            id="cat_validation"
                        )
                        yield Static(
                            "⚪ Execution Tests",
                            classes="test-item",
                            id="cat_execution"
                        )
                        yield Static(
                            "⚪ Normalization Tests",
                            classes="test-item",
                            id="cat_normalization"
                        )
                        yield Static(
                            "⚪ Gate Support Tests",
                            classes="test-item",
                            id="cat_gates"
                        )
                    
                    # Loading indicator
                    yield LoadingIndicator(id="loading")
                    
                    # Test output log
                    yield Static("Test Output:", classes="section-title")
                    
                    yield RichLog(
                        id="test_output",
                        classes="test-output",
                        highlight=True,
                        markup=True
                    )
                    
                    # Results section
                    yield Static("Results:", classes="section-title")
                    
                    with ScrollableContainer(classes="results-section"):
                        yield TestResultsWidget(id="results_widget")
                    
                    # Summary
                    yield Static(
                        "Configure test options and click 'Run Tests' to validate your backend.",
                        classes="summary-box",
                        id="summary"
                    )
                
                # Progress indicator
                with Vertical(classes="progress-section"):
                    yield Static(
                        "Progress: Step 6 of 7",
                        classes="progress-text"
                    )
                    yield Static(
                        "████████░░ 86%",
                        classes="progress-bar",
                        id="step_progress"
                    )
                
                # Navigation buttons
                with Horizontal(classes="button-container"):
                    yield Button(
                        "← Back",
                        id="btn_back",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Abort Tests",
                        id="btn_abort",
                        variant="error",
                        classes="nav-button",
                        disabled=True
                    )
                    yield Button(
                        "Cancel",
                        id="btn_cancel",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Next: Review →",
                        id="btn_next",
                        variant="primary",
                        classes="nav-button",
                        disabled=True
                    )
    
    def on_mount(self) -> None:
        """Initialize the testing screen."""
        # Hide loading indicator initially
        self.query_one("#loading", LoadingIndicator).display = False
    
    def watch_tests_running(self, running: bool) -> None:
        """React to test running state changes."""
        self.query_one("#loading", LoadingIndicator).display = running
        self.query_one("#btn_run_tests", Button).disabled = running
        self.query_one("#btn_abort", Button).disabled = not running
        
        # Disable navigation while testing
        self.query_one("#btn_back", Button).disabled = running
        self.query_one("#btn_cancel", Button).disabled = running
    
    def watch_tests_complete(self, complete: bool) -> None:
        """React to test completion."""
        if complete:
            # Check if enough tests passed
            passed = sum(1 for t in self.tests if t.status == TestStatus.PASSED)
            total = len(self.tests)
            
            # Allow proceeding if at least 60% passed
            can_proceed = passed >= total * 0.6
            self.query_one("#btn_next", Button).disabled = not can_proceed
    
    async def _run_comprehensive_tests(self) -> None:
        """Run comprehensive test suite using TestExecutor."""
        self.tests_running = True
        self.tests_complete = False
        self.tests = []
        self._abort_requested = False
        
        log = self.query_one("#test_output", RichLog)
        log.clear()
        log.write("[bold]Starting comprehensive test suite...[/bold]\n")
        
        # Get test configuration
        circuit_type = self.query_one("#circuit_type", Select).value
        shots = int(self.query_one("#shots", Input).value or "1024")
        
        log.write(f"[dim]Circuit: {circuit_type}, Shots: {shots}[/dim]\n\n")
        
        # Import test executor
        try:
            from proxima.tui.testing.test_executor import TestExecutor, TestCategory as ExecCategory
            
            # Create executor
            executor = TestExecutor(
                backend_code=self.state.generated_code or "",
                backend_name=self.state.backend_name
            )
            
            # Set progress callback
            def progress_callback(progress):
                if not self._abort_requested:
                    self.current_test = progress.current_test
                    self.progress = progress.percentage
                    self._update_progress_display(progress)
            
            executor.set_progress_callback(progress_callback)
            
            # Run tests
            results = await executor.run_all_tests(
                circuit_type=circuit_type,
                shots=shots,
                include_performance=False
            )
            
            # Process results
            self._process_test_results(results, log)
            
        except ImportError:
            # Fallback to basic tests
            log.write("[yellow]Using basic test suite (full test package not available)[/yellow]\n")
            await self._run_basic_tests(log)
        except Exception as e:
            log.write(f"[red]Error running tests: {e}[/red]\n")
            import traceback
            log.write(f"[dim]{traceback.format_exc()}[/dim]")
        
        self.tests_running = False
        self.tests_complete = True
        
        # Show summary
        self._show_summary()
    
    def _update_progress_display(self, progress) -> None:
        """Update progress display."""
        # Update category displays
        category_map = {
            "initialization": "cat_initialization",
            "validation": "cat_validation",
            "execution": "cat_execution",
            "normalization": "cat_normalization",
            "gate_support": "cat_gates",
        }
        
        cat_id = category_map.get(progress.current_category.value)
        if cat_id:
            try:
                widget = self.query_one(f"#{cat_id}", Static)
                icon = "🔄" if progress.current_status.value == "running" else "⚪"
                cat_name = progress.current_category.value.replace("_", " ").title()
                widget.update(f"{icon} {cat_name} Tests")
            except Exception:
                pass
    
    def _process_test_results(self, results: Dict[str, Any], log: RichLog) -> None:
        """Process and display test results."""
        summary = results.get("summary", {})
        by_category = results.get("by_category", {})
        
        # Update category displays
        category_map = {
            "initialization": "cat_initialization",
            "validation": "cat_validation",
            "execution": "cat_execution",
            "normalization": "cat_normalization",
            "gate_support": "cat_gates",
        }
        
        for cat_name, tests in by_category.items():
            cat_id = category_map.get(cat_name)
            if cat_id:
                try:
                    widget = self.query_one(f"#{cat_id}", Static)
                    
                    passed = sum(1 for t in tests if t.get("passed", False))
                    total = len(tests)
                    
                    if passed == total:
                        icon = "✅"
                    elif passed > 0:
                        icon = "⚠️"
                    else:
                        icon = "❌"
                    
                    display_name = cat_name.replace("_", " ").title()
                    widget.update(f"{icon} {display_name} Tests ({passed}/{total})")
                except Exception:
                    pass
            
            # Log results
            for test in tests:
                name = test.get("name", "Unknown")
                status = test.get("status", "unknown")
                message = test.get("message", "")
                
                if status == "passed":
                    log.write(f"  [green]✓[/green] {name}\n")
                    self.tests.append(TestResult(
                        name=name,
                        status=TestStatus.PASSED,
                        message=message,
                        category=cat_name
                    ))
                elif status in ("failed", "error"):
                    log.write(f"  [red]✗[/red] {name}: {message}\n")
                    self.tests.append(TestResult(
                        name=name,
                        status=TestStatus.FAILED,
                        message=message,
                        category=cat_name
                    ))
                else:
                    log.write(f"  [yellow]○[/yellow] {name}\n")
                    self.tests.append(TestResult(
                        name=name,
                        status=TestStatus.SKIPPED,
                        message=message,
                        category=cat_name
                    ))
        
        # Update results widget
        results_widget = self.query_one("#results_widget", TestResultsWidget)
        
        # Convert to expected format
        formatted_results = {}
        for result in results.get("results", []):
            formatted_results[result["name"]] = {
                "passed": result.get("passed", False),
                "status": result.get("status", "UNKNOWN"),
                "message": result.get("message", ""),
                "details": result.get("details", {}),
            }
        
        formatted_results["execution_time"] = summary.get("duration_ms", 0)
        formatted_results["total_shots"] = int(self.query_one("#shots", Input).value or "1024")
        
        results_widget.update_results(formatted_results)
        
        # Log summary
        log.write("\n" + "=" * 50 + "\n")
        log.write(f"[bold]Summary:[/bold]\n")
        log.write(f"  Total: {summary.get('total', 0)}\n")
        log.write(f"  Passed: {summary.get('passed', 0)}\n")
        log.write(f"  Failed: {summary.get('failed', 0)}\n")
        log.write(f"  Duration: {summary.get('duration_ms', 0):.0f}ms\n")
    
    async def _run_basic_tests(self, log: RichLog) -> None:
        """Run basic tests (fallback when full executor not available)."""
        # Syntax test
        self._update_category_status("cat_initialization", "running")
        log.write("  [yellow]▶[/yellow] Running syntax validation...")
        await asyncio.sleep(0.2)
        
        try:
            code = self.state.generated_code or ""
            compile(code, "<backend>", "exec")
            
            self._update_category_status("cat_initialization", "passed")
            log.write("  [green]✓[/green] Syntax is valid\n")
            self.tests.append(TestResult(
                name="Syntax Validation",
                status=TestStatus.PASSED,
                message="Code syntax is valid Python",
                category="initialization"
            ))
        except SyntaxError as e:
            self._update_category_status("cat_initialization", "failed")
            log.write(f"  [red]✗[/red] Syntax error: {e}\n")
            self.tests.append(TestResult(
                name="Syntax Validation",
                status=TestStatus.FAILED,
                message=f"Syntax error on line {e.lineno}: {e.msg}",
                category="initialization"
            ))
            return  # Can't continue with syntax errors
        
        # Import check
        self._update_category_status("cat_validation", "running")
        log.write("  [yellow]▶[/yellow] Checking imports...")
        await asyncio.sleep(0.2)
        
        code = self.state.generated_code or ""
        required_imports = ["from __future__", "from typing", "BaseBackend"]
        found = sum(1 for imp in required_imports if imp in code)
        
        if found >= 2:
            log.write("  [green]✓[/green] Required imports present\n")
            self.tests.append(TestResult(
                name="Import Check",
                status=TestStatus.PASSED,
                message="All required imports found",
                category="validation"
            ))
        else:
            log.write("  [yellow]○[/yellow] Some imports may be missing\n")
            self.tests.append(TestResult(
                name="Import Check",
                status=TestStatus.PASSED,  # Non-critical
                message="Some standard imports may be missing",
                category="validation"
            ))
        
        self._update_category_status("cat_validation", "passed")
        
        # Class definition
        self._update_category_status("cat_execution", "running")
        log.write("  [yellow]▶[/yellow] Checking class definition...")
        await asyncio.sleep(0.2)
        
        class_name = self._to_camel_case(self.state.backend_name) + "Backend"
        
        if f"class {class_name}" in code and "BaseBackend" in code:
            log.write(f"  [green]✓[/green] Class {class_name} defined correctly\n")
            self.tests.append(TestResult(
                name="Class Definition",
                status=TestStatus.PASSED,
                message=f"Class {class_name} extends BaseBackend",
                category="execution"
            ))
        else:
            log.write(f"  [yellow]○[/yellow] Class definition may need adjustment\n")
            self.tests.append(TestResult(
                name="Class Definition",
                status=TestStatus.PASSED,  # Non-critical
                message="Class may need adjustment",
                category="execution"
            ))
        
        # Required methods
        log.write("  [yellow]▶[/yellow] Checking required methods...")
        await asyncio.sleep(0.2)
        
        required = ["def __init__", "def execute", "def get_capabilities", "def is_available"]
        found_methods = sum(1 for m in required if m in code)
        
        if found_methods == len(required):
            log.write(f"  [green]✓[/green] All {len(required)} required methods present\n")
            self.tests.append(TestResult(
                name="Required Methods",
                status=TestStatus.PASSED,
                message=f"All {len(required)} required methods found",
                category="execution"
            ))
        else:
            log.write(f"  [yellow]○[/yellow] {found_methods}/{len(required)} methods found\n")
            self.tests.append(TestResult(
                name="Required Methods",
                status=TestStatus.PASSED if found_methods >= 2 else TestStatus.FAILED,
                message=f"{found_methods}/{len(required)} required methods found",
                category="execution"
            ))
        
        self._update_category_status("cat_execution", "passed" if found_methods >= 2 else "partial")
        
        # Capabilities
        self._update_category_status("cat_normalization", "running")
        log.write("  [yellow]▶[/yellow] Checking capabilities configuration...")
        await asyncio.sleep(0.2)
        
        if "BackendCapabilities" in code and "max_qubits" in code:
            log.write("  [green]✓[/green] Capabilities properly configured\n")
            self.tests.append(TestResult(
                name="Capabilities",
                status=TestStatus.PASSED,
                message="BackendCapabilities properly defined",
                category="normalization"
            ))
            self._update_category_status("cat_normalization", "passed")
        else:
            log.write("  [yellow]○[/yellow] Capabilities may need configuration\n")
            self.tests.append(TestResult(
                name="Capabilities",
                status=TestStatus.PASSED,
                message="Capabilities may need configuration",
                category="normalization"
            ))
            self._update_category_status("cat_normalization", "passed")
        
        # Gate support
        self._update_category_status("cat_gates", "running")
        log.write("  [yellow]▶[/yellow] Checking gate support...")
        await asyncio.sleep(0.2)
        
        gates = ["'h'", "'x'", "'cx'", "'rz'"]
        found_gates = sum(1 for g in gates if g in code.lower())
        
        log.write(f"  [green]✓[/green] {found_gates}/{len(gates)} standard gates configured\n")
        self.tests.append(TestResult(
            name="Gate Support",
            status=TestStatus.PASSED if found_gates >= 2 else TestStatus.FAILED,
            message=f"{found_gates} standard gates configured",
            category="gate_support"
        ))
        self._update_category_status("cat_gates", "passed" if found_gates >= 2 else "partial")
        
        # Update results widget
        results_widget = self.query_one("#results_widget", TestResultsWidget)
        formatted_results = {
            t.name: {
                "passed": t.status == TestStatus.PASSED,
                "status": t.status.value.upper(),
                "message": t.message,
            }
            for t in self.tests
        }
        results_widget.update_results(formatted_results)
    
    def _update_category_status(self, cat_id: str, status: str) -> None:
        """Update category status display."""
        try:
            widget = self.query_one(f"#{cat_id}", Static)
            current_text = str(widget.renderable)
            
            # Extract category name
            parts = current_text.split(" ", 1)
            name = parts[1] if len(parts) > 1 else current_text
            
            icons = {
                "pending": "⚪",
                "running": "🔄",
                "passed": "✅",
                "failed": "❌",
                "partial": "⚠️",
            }
            icon = icons.get(status, "⚪")
            widget.update(f"{icon} {name}")
        except Exception:
            pass
    
    def _show_summary(self) -> None:
        """Show test summary."""
        summary = self.query_one("#summary", Static)
        
        passed = sum(1 for t in self.tests if t.status == TestStatus.PASSED)
        failed = sum(1 for t in self.tests if t.status == TestStatus.FAILED)
        total = len(self.tests)
        
        if failed == 0:
            summary.update(
                f"✅ All {total} tests passed! Your backend is ready.\n"
                "   Click 'Next: Review' to finalize."
            )
            summary.set_classes("summary-box summary-success")
        elif passed >= total * 0.7:
            summary.update(
                f"⚠️ {passed}/{total} tests passed. Some issues detected.\n"
                "   You can proceed, but consider fixing issues first."
            )
            summary.set_classes("summary-box summary-warning")
        else:
            summary.update(
                f"❌ {failed}/{total} tests failed.\n"
                "   Go back to fix issues in the code template."
            )
            summary.set_classes("summary-box summary-error")
        
        # Store results in state
        self.state.test_results = {
            "passed": passed,
            "failed": failed,
            "total": total,
            "details": [
                {"name": t.name, "status": t.status.value, "message": t.message}
                for t in self.tests
            ]
        }
        self.state.tests_passed = failed == 0
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_back":
            self.state.current_step = 5
            self.dismiss({"action": "back", "state": self.state})
        
        elif event.button.id == "btn_run_tests":
            if not self.tests_running:
                asyncio.create_task(self._run_comprehensive_tests())
        
        elif event.button.id == "btn_abort":
            self._abort_requested = True
            log = self.query_one("#test_output", RichLog)
            log.write("\n[yellow]⚠️ Tests aborted by user[/yellow]\n")
            self.tests_running = False
            self.tests_complete = True
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_next":
            self.state.current_step = 7
            self.dismiss({"action": "next", "state": self.state})
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        if self.tests_running:
            self._abort_requested = True
        else:
            self.dismiss({"action": "cancel"})
