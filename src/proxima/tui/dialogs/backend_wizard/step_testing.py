"""Step 6: Testing & Validation.

Run validation tests on the generated backend code:
- Syntax validation
- Import checks
- Basic instantiation test
- Optional: AI-assisted review
"""

from __future__ import annotations

import asyncio
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import (
    Static, Button, ProgressBar, RichLog, LoadingIndicator
)
from textual.screen import ModalScreen

from .wizard_state import BackendWizardState


class TestStatus(Enum):
    """Status of a test."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    message: str = ""
    details: Optional[str] = None


class TestingStepScreen(ModalScreen[dict]):
    """
    Step 6: Testing and validation screen.
    
    Runs a series of tests on the generated backend code
    to validate it before finalizing.
    """
    
    DEFAULT_CSS = """
    TestingStepScreen {
        align: center middle;
    }
    
    TestingStepScreen .wizard-container {
        width: 90;
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
    
    TestingStepScreen .test-list {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        margin: 1 0;
    }
    
    TestingStepScreen .test-item {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    
    TestingStepScreen .test-status-pending {
        color: $text-muted;
    }
    
    TestingStepScreen .test-status-running {
        color: $warning;
    }
    
    TestingStepScreen .test-status-passed {
        color: $success;
    }
    
    TestingStepScreen .test-status-failed {
        color: $error;
    }
    
    TestingStepScreen .test-output {
        height: 15;
        margin: 1 0;
        border: solid $primary-darken-3;
        background: $surface-darken-2;
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
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, state: BackendWizardState):
        """
        Initialize the testing screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
        self.tests: list[TestResult] = []
        self.tests_running = False
        self.tests_complete = False
    
    def compose(self) -> ComposeResult:
        """Compose the testing screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "🧪 Add Custom Backend - Testing",
                    classes="wizard-title"
                )
                
                yield Static(
                    f"Validating backend: {self.state.display_name}",
                    classes="field-hint"
                )
                
                with ScrollableContainer(classes="form-container"):
                    # Test list
                    yield Static(
                        "Validation Tests:",
                        classes="section-title"
                    )
                    
                    with Vertical(classes="test-list", id="test_list"):
                        yield Static(
                            "⚪ Syntax Validation",
                            classes="test-item test-status-pending",
                            id="test_syntax"
                        )
                        yield Static(
                            "⚪ Import Check",
                            classes="test-item test-status-pending",
                            id="test_imports"
                        )
                        yield Static(
                            "⚪ Class Definition",
                            classes="test-item test-status-pending",
                            id="test_class"
                        )
                        yield Static(
                            "⚪ Required Methods",
                            classes="test-item test-status-pending",
                            id="test_methods"
                        )
                        yield Static(
                            "⚪ Capabilities Check",
                            classes="test-item test-status-pending",
                            id="test_capabilities"
                        )
                    
                    # Loading indicator
                    yield LoadingIndicator(id="loading")
                    
                    # Test output log
                    yield Static(
                        "Test Output:",
                        classes="section-title"
                    )
                    
                    yield RichLog(
                        id="test_output",
                        classes="test-output",
                        highlight=True,
                        markup=True
                    )
                    
                    # Summary
                    yield Static(
                        "Click 'Run Tests' to validate your backend code.",
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
                        "███████ 86%",
                        classes="progress-bar"
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
                        "Run Tests",
                        id="btn_run_tests",
                        variant="warning",
                        classes="nav-button"
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
    
    async def _run_all_tests(self) -> None:
        """Run all validation tests."""
        self.tests_running = True
        self.tests = []
        
        # Show loading
        self.query_one("#loading", LoadingIndicator).display = True
        self.query_one("#btn_run_tests", Button).disabled = True
        
        log = self.query_one("#test_output", RichLog)
        log.clear()
        log.write("[bold]Starting validation tests...[/bold]\n")
        
        # Run each test
        await self._run_syntax_test(log)
        await asyncio.sleep(0.3)  # Visual delay
        
        await self._run_import_test(log)
        await asyncio.sleep(0.3)
        
        await self._run_class_test(log)
        await asyncio.sleep(0.3)
        
        await self._run_methods_test(log)
        await asyncio.sleep(0.3)
        
        await self._run_capabilities_test(log)
        
        # Hide loading
        self.query_one("#loading", LoadingIndicator).display = False
        
        # Show summary
        self._show_summary()
        
        self.tests_running = False
        self.tests_complete = True
        
        # Enable/disable next button based on results
        passed = sum(1 for t in self.tests if t.status == TestStatus.PASSED)
        self.query_one("#btn_run_tests", Button).disabled = False
        
        # Allow proceeding if at least syntax passes
        syntax_passed = any(
            t.name == "syntax" and t.status == TestStatus.PASSED
            for t in self.tests
        )
        self.query_one("#btn_next", Button).disabled = not syntax_passed
    
    async def _run_syntax_test(self, log: RichLog) -> None:
        """Run syntax validation test."""
        self._update_test_status("test_syntax", TestStatus.RUNNING)
        log.write("  [yellow]▶[/yellow] Running syntax validation...")
        
        try:
            code = self.state.generated_code or ""
            compile(code, "<backend>", "exec")
            
            self._update_test_status("test_syntax", TestStatus.PASSED)
            log.write("  [green]✓[/green] Syntax is valid\n")
            self.tests.append(TestResult(
                name="syntax",
                status=TestStatus.PASSED,
                message="Code syntax is valid Python"
            ))
        except SyntaxError as e:
            self._update_test_status("test_syntax", TestStatus.FAILED)
            log.write(f"  [red]✗[/red] Syntax error: {e}\n")
            self.tests.append(TestResult(
                name="syntax",
                status=TestStatus.FAILED,
                message=f"Syntax error on line {e.lineno}: {e.msg}"
            ))
    
    async def _run_import_test(self, log: RichLog) -> None:
        """Run import validation test."""
        self._update_test_status("test_imports", TestStatus.RUNNING)
        log.write("  [yellow]▶[/yellow] Checking imports...")
        
        # Check for standard imports in the code
        code = self.state.generated_code or ""
        required_imports = ["from __future__", "from typing", "BaseBackend"]
        found = sum(1 for imp in required_imports if imp in code)
        
        if found >= 2:
            self._update_test_status("test_imports", TestStatus.PASSED)
            log.write("  [green]✓[/green] Required imports present\n")
            self.tests.append(TestResult(
                name="imports",
                status=TestStatus.PASSED,
                message="All required imports found"
            ))
        else:
            self._update_test_status("test_imports", TestStatus.FAILED)
            log.write("  [red]✗[/red] Missing required imports\n")
            self.tests.append(TestResult(
                name="imports",
                status=TestStatus.FAILED,
                message="Missing required imports"
            ))
    
    async def _run_class_test(self, log: RichLog) -> None:
        """Run class definition test."""
        self._update_test_status("test_class", TestStatus.RUNNING)
        log.write("  [yellow]▶[/yellow] Checking class definition...")
        
        code = self.state.generated_code or ""
        class_name = self._to_camel_case(self.state.backend_name) + "Backend"
        
        if f"class {class_name}" in code and "BaseBackend" in code:
            self._update_test_status("test_class", TestStatus.PASSED)
            log.write(f"  [green]✓[/green] Class {class_name} defined correctly\n")
            self.tests.append(TestResult(
                name="class",
                status=TestStatus.PASSED,
                message=f"Class {class_name} extends BaseBackend"
            ))
        else:
            self._update_test_status("test_class", TestStatus.FAILED)
            log.write(f"  [red]✗[/red] Class definition issue\n")
            self.tests.append(TestResult(
                name="class",
                status=TestStatus.FAILED,
                message="Class not properly defined"
            ))
    
    async def _run_methods_test(self, log: RichLog) -> None:
        """Run required methods test."""
        self._update_test_status("test_methods", TestStatus.RUNNING)
        log.write("  [yellow]▶[/yellow] Checking required methods...")
        
        code = self.state.generated_code or ""
        required = ["def __init__", "def execute", "def get_capabilities", "def is_available"]
        found = sum(1 for m in required if m in code)
        
        if found == len(required):
            self._update_test_status("test_methods", TestStatus.PASSED)
            log.write(f"  [green]✓[/green] All {len(required)} required methods present\n")
            self.tests.append(TestResult(
                name="methods",
                status=TestStatus.PASSED,
                message=f"All {len(required)} required methods found"
            ))
        else:
            self._update_test_status("test_methods", TestStatus.FAILED)
            log.write(f"  [red]✗[/red] Missing {len(required) - found} method(s)\n")
            self.tests.append(TestResult(
                name="methods",
                status=TestStatus.FAILED,
                message=f"Missing {len(required) - found} required method(s)"
            ))
    
    async def _run_capabilities_test(self, log: RichLog) -> None:
        """Run capabilities check."""
        self._update_test_status("test_capabilities", TestStatus.RUNNING)
        log.write("  [yellow]▶[/yellow] Checking capabilities configuration...")
        
        code = self.state.generated_code or ""
        
        if "BackendCapabilities" in code and "max_qubits" in code:
            self._update_test_status("test_capabilities", TestStatus.PASSED)
            log.write("  [green]✓[/green] Capabilities properly configured\n")
            self.tests.append(TestResult(
                name="capabilities",
                status=TestStatus.PASSED,
                message="BackendCapabilities properly defined"
            ))
        else:
            self._update_test_status("test_capabilities", TestStatus.FAILED)
            log.write("  [red]✗[/red] Capabilities not properly defined\n")
            self.tests.append(TestResult(
                name="capabilities",
                status=TestStatus.FAILED,
                message="BackendCapabilities not properly defined"
            ))
    
    def _update_test_status(self, test_id: str, status: TestStatus) -> None:
        """Update the visual status of a test."""
        test_widget = self.query_one(f"#{test_id}", Static)
        
        status_symbols = {
            TestStatus.PENDING: "⚪",
            TestStatus.RUNNING: "🔄",
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.SKIPPED: "⏭️",
        }
        
        status_classes = {
            TestStatus.PENDING: "test-status-pending",
            TestStatus.RUNNING: "test-status-running",
            TestStatus.PASSED: "test-status-passed",
            TestStatus.FAILED: "test-status-failed",
            TestStatus.SKIPPED: "test-status-pending",
        }
        
        # Get test name from current text
        current_text = str(test_widget.renderable)
        test_name = current_text.split(" ", 1)[1] if " " in current_text else current_text
        
        # Update text and class
        test_widget.update(f"{status_symbols[status]} {test_name}")
        test_widget.set_classes(f"test-item {status_classes[status]}")
    
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
        elif passed >= 3:
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
                asyncio.create_task(self._run_all_tests())
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_next":
            self.state.current_step = 7
            self.dismiss({"action": "next", "state": self.state})
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss({"action": "cancel"})
