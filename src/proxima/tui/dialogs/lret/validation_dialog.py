"""LRET Validation Checklist Dialog.

TUI dialog for running and displaying the LRET variants validation checklist.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, List, Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Button, Label, ProgressBar, DataTable
from textual.screen import ModalScreen
from textual.reactive import reactive
from rich.text import Text


class ValidationChecklistDialog(ModalScreen):
    """Dialog for running LRET validation checklist.
    
    Features:
    - Run all validation checks
    - View results by category
    - Export checklist to markdown
    - Generate validation report
    """
    
    DEFAULT_CSS = """
    ValidationChecklistDialog {
        align: center middle;
    }
    
    ValidationChecklistDialog > Container {
        width: 85%;
        height: 85%;
        border: thick $primary 50%;
        background: $surface;
    }
    
    ValidationChecklistDialog .dialog-header {
        height: 3;
        padding: 0 2;
        background: $primary-darken-2;
    }
    
    ValidationChecklistDialog .dialog-title {
        text-style: bold;
        padding: 1;
    }
    
    ValidationChecklistDialog .content-area {
        height: 1fr;
        padding: 1;
    }
    
    ValidationChecklistDialog .category-section {
        height: auto;
        min-height: 6;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary-darken-3;
        background: $surface-lighten-1;
    }
    
    ValidationChecklistDialog .category-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    ValidationChecklistDialog .check-item {
        height: 2;
        padding: 0 1;
    }
    
    ValidationChecklistDialog .check-passed {
        color: $success;
    }
    
    ValidationChecklistDialog .check-failed {
        color: $error;
    }
    
    ValidationChecklistDialog .check-warning {
        color: $warning;
    }
    
    ValidationChecklistDialog .check-skipped {
        color: $text-muted;
    }
    
    ValidationChecklistDialog .summary-section {
        height: 8;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
        margin-top: 1;
    }
    
    ValidationChecklistDialog .progress-section {
        height: 3;
        padding: 0 1;
    }
    
    ValidationChecklistDialog .dialog-footer {
        height: 3;
        layout: horizontal;
        padding: 0 2;
        background: $surface-darken-1;
    }
    
    ValidationChecklistDialog .action-btn {
        margin-right: 1;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
    ]
    
    is_running: reactive[bool] = reactive(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: Dict[str, List[Dict]] = {}
        self._summary: Dict[str, int] = {}
    
    def compose(self) -> ComposeResult:
        with Container():
            # Header
            with Container(classes="dialog-header"):
                yield Label("📋 LRET Validation Checklist", classes="dialog-title")
            
            # Progress section
            with Horizontal(classes="progress-section"):
                yield ProgressBar(id="progress-bar", total=100)
                yield Static("Ready to run validation", id="progress-status")
            
            # Content area with scrollable results
            with ScrollableContainer(classes="content-area"):
                yield Static(id="results-content")
            
            # Summary section
            with Container(classes="summary-section"):
                yield Static(id="summary-content")
            
            # Footer
            with Horizontal(classes="dialog-footer"):
                yield Button("Run Validation", id="btn-run", variant="primary", classes="action-btn")
                yield Button("Export Checklist", id="btn-export", classes="action-btn", disabled=True)
                yield Button("View Details", id="btn-details", classes="action-btn", disabled=True)
                yield Button("Close", id="btn-close", classes="action-btn")
    
    def on_mount(self) -> None:
        """Initialize the dialog."""
        self._update_initial_content()
    
    def _update_initial_content(self) -> None:
        """Show initial content before running validation."""
        content = self.query_one("#results-content", Static)
        
        text = Text()
        text.append("LRET Variants Validation Checklist\n\n", style="bold")
        text.append("This tool will verify that all LRET variant components\n")
        text.append("are properly installed and configured.\n\n")
        text.append("Categories to check:\n", style="bold")
        text.append("  • Installation - Dependencies and base modules\n")
        text.append("  • Cirq Scalability - Adapter and benchmarking\n")
        text.append("  • PennyLane Hybrid - Device and algorithms\n")
        text.append("  • Phase 7 Unified - Multi-framework support\n")
        text.append("  • TUI Integration - Dialogs and screens\n")
        text.append("  • Documentation - Required docs present\n\n")
        text.append("Click 'Run Validation' to start.\n")
        
        content.update(text)
        
        summary = self.query_one("#summary-content", Static)
        summary.update("No validation results yet.")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-run":
            self._run_validation()
        elif button_id == "btn-export":
            self._export_checklist()
        elif button_id == "btn-details":
            self._view_details()
        elif button_id == "btn-close":
            self.app.pop_screen()
    
    def _run_validation(self) -> None:
        """Run the validation checklist."""
        if self.is_running:
            return
        
        self.is_running = True
        self.notify("Running validation...", severity="information")
        
        asyncio.create_task(self._execute_validation())
    
    async def _execute_validation(self) -> None:
        """Execute validation asynchronously."""
        try:
            progress = self.query_one("#progress-bar", ProgressBar)
            status = self.query_one("#progress-status", Static)
            content = self.query_one("#results-content", Static)
            
            # Categories to check
            categories = [
                ("Installation", self._check_installation),
                ("Cirq Scalability", self._check_cirq_scalability),
                ("PennyLane Hybrid", self._check_pennylane_hybrid),
                ("Phase 7 Unified", self._check_phase7_unified),
                ("TUI Integration", self._check_tui_integration),
                ("Documentation", self._check_documentation),
            ]
            
            self._results = {}
            self._summary = {'PASSED': 0, 'FAILED': 0, 'WARNING': 0, 'SKIPPED': 0}
            
            result_text = Text()
            
            for i, (category_name, check_func) in enumerate(categories):
                # Update progress
                progress.update(progress=(i / len(categories)) * 100)
                status.update(f"Checking: {category_name}...")
                
                # Run checks
                results = check_func()
                self._results[category_name] = results
                
                # Update summary
                for result in results:
                    self._summary[result['status']] = self._summary.get(result['status'], 0) + 1
                
                # Add to display
                result_text.append(f"\n{category_name}\n", style="bold")
                result_text.append("─" * 40 + "\n")
                
                for result in results:
                    icon = self._get_status_icon(result['status'])
                    style = self._get_status_style(result['status'])
                    result_text.append(f"  {icon} {result['name']}\n", style=style)
                    if result.get('message'):
                        result_text.append(f"      {result['message']}\n", style="dim")
                
                content.update(result_text)
                await asyncio.sleep(0.1)
            
            # Complete
            progress.update(progress=100)
            status.update("Validation complete!")
            
            # Update summary
            self._update_summary()
            
            # Enable buttons
            try:
                self.query_one("#btn-export", Button).disabled = False
                self.query_one("#btn-details", Button).disabled = False
            except Exception:
                pass
            
            if self._summary.get('FAILED', 0) == 0:
                self.notify("✅ All checks passed!", severity="success")
            else:
                self.notify(f"❌ {self._summary['FAILED']} checks failed", severity="warning")
            
        except Exception as e:
            self.notify(f"Validation error: {e}", severity="error")
        finally:
            self.is_running = False
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for status."""
        icons = {
            'PASSED': '✅',
            'FAILED': '❌',
            'WARNING': '⚠️',
            'SKIPPED': '⏭️',
        }
        return icons.get(status, '•')
    
    def _get_status_style(self, status: str) -> str:
        """Get style for status."""
        styles = {
            'PASSED': 'green',
            'FAILED': 'red',
            'WARNING': 'yellow',
            'SKIPPED': 'dim',
        }
        return styles.get(status, '')
    
    def _check_installation(self) -> List[Dict]:
        """Check installation requirements."""
        results = []
        
        # Check base imports
        try:
            import proxima.backends.lret
            results.append({'name': 'LRET base module', 'status': 'PASSED', 'message': 'Imported successfully'})
        except ImportError as e:
            results.append({'name': 'LRET base module', 'status': 'FAILED', 'message': str(e)})
        
        # Check Cirq
        try:
            import cirq
            results.append({'name': 'Cirq', 'status': 'PASSED', 'message': f'Version {cirq.__version__}'})
        except ImportError:
            results.append({'name': 'Cirq', 'status': 'FAILED', 'message': 'Not installed'})
        
        # Check PennyLane
        try:
            import pennylane
            results.append({'name': 'PennyLane', 'status': 'PASSED', 'message': f'Version {pennylane.__version__}'})
        except ImportError:
            results.append({'name': 'PennyLane', 'status': 'FAILED', 'message': 'Not installed'})
        
        # Check Qiskit (optional)
        try:
            import qiskit
            results.append({'name': 'Qiskit (optional)', 'status': 'PASSED', 'message': f'Version {qiskit.__version__}'})
        except ImportError:
            results.append({'name': 'Qiskit (optional)', 'status': 'WARNING', 'message': 'Not installed'})
        
        return results
    
    def _check_cirq_scalability(self) -> List[Dict]:
        """Check Cirq Scalability variant."""
        results = []
        
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            results.append({'name': 'LRETCirqScalabilityAdapter', 'status': 'PASSED', 'message': 'Class available'})
            
            # Check adapter creation
            adapter = LRETCirqScalabilityAdapter()
            results.append({'name': 'Adapter creation', 'status': 'PASSED', 'message': 'Created successfully'})
        except ImportError as e:
            results.append({'name': 'LRETCirqScalabilityAdapter', 'status': 'FAILED', 'message': str(e)})
            results.append({'name': 'Adapter creation', 'status': 'SKIPPED', 'message': 'Module not available'})
        except Exception as e:
            results.append({'name': 'Adapter creation', 'status': 'FAILED', 'message': str(e)})
        
        # Check visualization
        try:
            from proxima.backends.lret.visualization import generate_benchmark_report
            results.append({'name': 'Visualization module', 'status': 'PASSED', 'message': 'Available'})
        except ImportError:
            results.append({'name': 'Visualization module', 'status': 'WARNING', 'message': 'Not available'})
        
        return results
    
    def _check_pennylane_hybrid(self) -> List[Dict]:
        """Check PennyLane Hybrid variant."""
        results = []
        
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            results.append({'name': 'QLRETDevice', 'status': 'PASSED', 'message': 'Class available'})
            
            # Check device creation
            dev = QLRETDevice(wires=2, shots=1024)
            results.append({'name': 'Device creation', 'status': 'PASSED', 'message': '2 wires, 1024 shots'})
        except ImportError as e:
            results.append({'name': 'QLRETDevice', 'status': 'FAILED', 'message': str(e)})
            results.append({'name': 'Device creation', 'status': 'SKIPPED', 'message': 'Module not available'})
        except Exception as e:
            results.append({'name': 'Device creation', 'status': 'FAILED', 'message': str(e)})
        
        # Check algorithms
        try:
            from proxima.backends.lret.algorithms import VQE, QAOA, QNN
            results.append({'name': 'VQE algorithm', 'status': 'PASSED', 'message': 'Available'})
            results.append({'name': 'QAOA algorithm', 'status': 'PASSED', 'message': 'Available'})
            results.append({'name': 'QNN classifier', 'status': 'PASSED', 'message': 'Available'})
        except ImportError:
            results.append({'name': 'Algorithms module', 'status': 'WARNING', 'message': 'Not available'})
        
        return results
    
    def _check_phase7_unified(self) -> List[Dict]:
        """Check Phase 7 Unified variant."""
        results = []
        
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter, Phase7Config
            results.append({'name': 'LRETPhase7UnifiedAdapter', 'status': 'PASSED', 'message': 'Class available'})
            results.append({'name': 'Phase7Config', 'status': 'PASSED', 'message': 'Class available'})
            
            # Check adapter creation
            adapter = LRETPhase7UnifiedAdapter()
            results.append({'name': 'Adapter creation', 'status': 'PASSED', 'message': 'Created successfully'})
        except ImportError as e:
            results.append({'name': 'Phase7UnifiedAdapter', 'status': 'FAILED', 'message': str(e)})
        except Exception as e:
            results.append({'name': 'Adapter creation', 'status': 'FAILED', 'message': str(e)})
        
        # Check gate fusion
        try:
            from proxima.backends.lret.phase7_unified import GateFusion
            results.append({'name': 'GateFusion', 'status': 'PASSED', 'message': 'Available'})
        except ImportError:
            results.append({'name': 'GateFusion', 'status': 'WARNING', 'message': 'Not available'})
        
        return results
    
    def _check_tui_integration(self) -> List[Dict]:
        """Check TUI integration."""
        results = []
        
        # Check dialogs
        dialogs = [
            'LRETInstallerDialog',
            'LRETConfigDialog',
            'Phase7Dialog',
            'VariantAnalysisDialog',
        ]
        
        for dialog_name in dialogs:
            try:
                from proxima.tui.dialogs import lret
                cls = getattr(lret, dialog_name, None)
                if cls:
                    results.append({'name': dialog_name, 'status': 'PASSED', 'message': 'Available'})
                else:
                    results.append({'name': dialog_name, 'status': 'FAILED', 'message': 'Not found'})
            except ImportError:
                results.append({'name': dialog_name, 'status': 'FAILED', 'message': 'Import error'})
        
        # Check screens
        try:
            from proxima.tui.screens import BenchmarkComparisonScreen
            results.append({'name': 'BenchmarkComparisonScreen', 'status': 'PASSED', 'message': 'Available'})
        except ImportError:
            results.append({'name': 'BenchmarkComparisonScreen', 'status': 'FAILED', 'message': 'Not available'})
        
        # Check wizard
        try:
            from proxima.tui.wizards import PennyLaneAlgorithmWizard
            results.append({'name': 'PennyLaneAlgorithmWizard', 'status': 'PASSED', 'message': 'Available'})
        except ImportError:
            results.append({'name': 'PennyLaneAlgorithmWizard', 'status': 'WARNING', 'message': 'Not available'})
        
        return results
    
    def _check_documentation(self) -> List[Dict]:
        """Check documentation."""
        results = []
        from pathlib import Path
        
        docs = [
            ('Integration Report', 'LRET_BACKEND_VARIANTS_INTEGRATION_REPORT.md'),
            ('README', 'README.md'),
            ('TUI Guide', 'TUI_GUIDE_FOR_PROXIMA.md'),
        ]
        
        for name, path in docs:
            if Path(path).exists():
                results.append({'name': name, 'status': 'PASSED', 'message': 'Present'})
            else:
                results.append({'name': name, 'status': 'WARNING', 'message': 'Not found'})
        
        return results
    
    def _update_summary(self) -> None:
        """Update summary display."""
        summary = self.query_one("#summary-content", Static)
        
        text = Text()
        text.append("SUMMARY\n", style="bold")
        text.append("─" * 40 + "\n")
        
        total = sum(self._summary.values())
        text.append(f"Total Checks: {total}\n")
        text.append(f"✅ Passed:  {self._summary.get('PASSED', 0)}\n", style="green")
        text.append(f"❌ Failed:  {self._summary.get('FAILED', 0)}\n", style="red")
        text.append(f"⚠️  Warnings: {self._summary.get('WARNING', 0)}\n", style="yellow")
        text.append(f"⏭️  Skipped: {self._summary.get('SKIPPED', 0)}\n", style="dim")
        
        summary.update(text)
    
    def _export_checklist(self) -> None:
        """Export checklist to markdown file."""
        try:
            from pathlib import Path
            from datetime import datetime
            
            lines = ["# LRET Validation Checklist\n"]
            lines.append(f"\n**Generated:** {datetime.now().isoformat()}\n")
            
            for category, results in self._results.items():
                lines.append(f"\n## {category}\n")
                
                for result in results:
                    checkbox = "[x]" if result['status'] == 'PASSED' else "[ ]"
                    lines.append(f"- {checkbox} {result['name']}")
                    if result.get('message'):
                        lines.append(f"  - {result['message']}")
                    lines.append("")
            
            # Summary
            lines.append("\n## Summary\n")
            lines.append(f"- ✅ Passed: {self._summary.get('PASSED', 0)}")
            lines.append(f"- ❌ Failed: {self._summary.get('FAILED', 0)}")
            lines.append(f"- ⚠️ Warnings: {self._summary.get('WARNING', 0)}")
            lines.append(f"- ⏭️ Skipped: {self._summary.get('SKIPPED', 0)}")
            
            # Write file
            filename = Path("LRET_VALIDATION_CHECKLIST.md")
            filename.write_text("\n".join(lines))
            
            self.notify(f"Checklist exported to {filename}", severity="success")
        except Exception as e:
            self.notify(f"Export error: {e}", severity="error")
    
    def _view_details(self) -> None:
        """View detailed results."""
        # Show detailed info
        failed_checks = []
        for category, results in self._results.items():
            for result in results:
                if result['status'] == 'FAILED':
                    failed_checks.append(f"{category}: {result['name']}")
        
        if failed_checks:
            self.notify("Failed checks:", severity="warning")
            for check in failed_checks[:5]:
                self.notify(f"  • {check}")
            if len(failed_checks) > 5:
                self.notify(f"  ... and {len(failed_checks) - 5} more")
        else:
            self.notify("No failed checks!", severity="success")
    
    def action_close(self) -> None:
        """Close the dialog."""
        self.app.pop_screen()


def show_validation_dialog(app) -> None:
    """Show the validation checklist dialog.
    
    Args:
        app: The Textual application instance.
    """
    app.push_screen(ValidationChecklistDialog())
