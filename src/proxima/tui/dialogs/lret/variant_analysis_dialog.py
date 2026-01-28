"""LRET Variant Analysis Dialog.

This dialog provides a TUI interface for analyzing and comparing
the three LRET backend variants:
1. Cirq Scalability - Benchmarking and comparison
2. PennyLane Hybrid - VQE, QAOA, QNN algorithms
3. Phase 7 Unified - Multi-framework execution

Features:
- View variant capabilities
- Run analysis tests
- Compare variants side-by-side
- View recommendations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import asyncio

from textual.app import ComposeResult
from textual.containers import (
    Container,
    Horizontal,
    Vertical,
    ScrollableContainer,
)
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Label,
    ProgressBar,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.screen import ModalScreen
from textual.reactive import reactive

try:
    from proxima.backends.lret.variant_analysis import (
        VariantAnalyzer,
        VariantInfo,
        VariantAnalysisResult,
        VariantComparisonResult,
        TaskType,
        VariantCapability,
        get_variant_analyzer,
        VARIANT_DEFINITIONS,
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False


DEFAULT_CSS = """
VariantAnalysisDialog {
    align: center middle;
}

VariantAnalysisDialog > Container {
    width: 95;
    height: 42;
    border: thick $primary;
    background: $surface;
    padding: 1 2;
}

VariantAnalysisDialog .dialog-title {
    text-align: center;
    text-style: bold;
    color: $text;
    padding-bottom: 1;
}

VariantAnalysisDialog .section-title {
    text-style: bold;
    color: $primary;
    padding: 1 0;
}

VariantAnalysisDialog .variant-card {
    height: auto;
    border: round $primary-darken-2;
    padding: 1;
    margin-bottom: 1;
}

VariantAnalysisDialog .variant-name {
    text-style: bold;
    color: $primary;
}

VariantAnalysisDialog .variant-description {
    color: $text-muted;
}

VariantAnalysisDialog .status-functional {
    color: $success;
}

VariantAnalysisDialog .status-installed {
    color: $warning;
}

VariantAnalysisDialog .status-error {
    color: $error;
}

VariantAnalysisDialog .capability-grid {
    height: auto;
    padding: 1;
}

VariantAnalysisDialog .comparison-table {
    height: 12;
}

VariantAnalysisDialog .recommendations-panel {
    height: 8;
    border: round $accent;
    padding: 1;
    overflow-y: auto;
}

VariantAnalysisDialog .button-row {
    height: 3;
    align: center middle;
    padding-top: 1;
}

VariantAnalysisDialog Button {
    margin: 0 1;
}

VariantAnalysisDialog DataTable {
    height: 100%;
}

VariantAnalysisDialog .progress-section {
    height: 4;
    padding: 1 0;
}
"""


class VariantAnalysisDialog(ModalScreen[Dict[str, Any]]):
    """Modal dialog for LRET variant analysis and comparison.
    
    Provides tabs for:
    - Overview: Summary of all variants
    - Capabilities: Detailed capability comparison
    - Analysis: Run tests and benchmarks
    - Recommendations: Task-based recommendations
    """
    
    CSS = DEFAULT_CSS
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("f5", "refresh", "Refresh"),
    ]
    
    # Reactive state
    is_analyzing: reactive[bool] = reactive(False)
    analysis_progress: reactive[float] = reactive(0.0)
    
    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize dialog."""
        super().__init__(name=name, id=id, classes=classes)
        self._analyzer: Optional[VariantAnalyzer] = None
        self._analysis_results: Dict[str, VariantAnalysisResult] = {}
        self._comparison_result: Optional[VariantComparisonResult] = None
    
    def compose(self) -> ComposeResult:
        """Compose dialog layout."""
        with Container():
            yield Label("📊 LRET Variant Analysis", classes="dialog-title")
            
            with TabbedContent():
                # Overview tab
                with TabPane("Overview", id="tab-overview"):
                    yield from self._compose_overview_tab()
                
                # Capabilities tab
                with TabPane("Capabilities", id="tab-capabilities"):
                    yield from self._compose_capabilities_tab()
                
                # Analysis tab
                with TabPane("Analysis", id="tab-analysis"):
                    yield from self._compose_analysis_tab()
                
                # Recommendations tab
                with TabPane("Recommendations", id="tab-recommendations"):
                    yield from self._compose_recommendations_tab()
            
            # Button row
            with Horizontal(classes="button-row"):
                yield Button("Refresh", variant="primary", id="btn-refresh")
                yield Button("Run Full Analysis", variant="success", id="btn-analyze")
                yield Button("Close", variant="default", id="btn-close")
    
    def _compose_overview_tab(self) -> ComposeResult:
        """Compose overview tab content."""
        yield Label("Registered Variants", classes="section-title")
        
        with ScrollableContainer(id="variants-container"):
            # Variant cards will be populated dynamically
            yield Static("Loading variants...", id="variants-loading")
    
    def _compose_capabilities_tab(self) -> ComposeResult:
        """Compose capabilities comparison tab."""
        yield Label("Capability Comparison Matrix", classes="section-title")
        
        with Container(classes="comparison-table"):
            yield DataTable(id="capability-table")
    
    def _compose_analysis_tab(self) -> ComposeResult:
        """Compose analysis tab."""
        yield Label("Variant Analysis", classes="section-title")
        
        with Horizontal(classes="config-row"):
            yield Checkbox("Run benchmarks", id="chk-benchmarks")
        
        with Vertical(classes="progress-section"):
            yield ProgressBar(total=100, id="analysis-progress")
            yield Static("Ready to analyze", id="analysis-status")
        
        yield Label("Analysis Results", classes="section-title")
        
        with Container(classes="comparison-table"):
            yield DataTable(id="analysis-table")
        
        with Horizontal(classes="button-row"):
            yield Button("Analyze All", variant="primary", id="btn-analyze-all")
    
    def _compose_recommendations_tab(self) -> ComposeResult:
        """Compose recommendations tab."""
        yield Label("Task-Based Recommendations", classes="section-title")
        
        with Container(classes="comparison-table"):
            yield DataTable(id="recommendations-table")
        
        yield Label("General Recommendations", classes="section-title")
        
        with ScrollableContainer(classes="recommendations-panel", id="recommendations-panel"):
            yield Static("Run analysis to see recommendations", id="recommendations-text")
    
    async def on_mount(self) -> None:
        """Handle mount event."""
        if ANALYSIS_AVAILABLE:
            self._analyzer = get_variant_analyzer()
            await self._populate_overview()
            await self._populate_capabilities()
            await self._populate_recommendations_table()
        else:
            self.notify("Analysis module not available", severity="error")
    
    async def _populate_overview(self) -> None:
        """Populate the overview tab with variant cards."""
        container = self.query_one("#variants-container", ScrollableContainer)
        loading = self.query_one("#variants-loading", Static)
        
        # Remove loading message
        loading.update("")
        
        # Create variant cards
        for name, info in VARIANT_DEFINITIONS.items():
            # Check status
            status = self._analyzer.check_variant_status(name) if self._analyzer else {}
            
            status_icon = "❓"
            status_class = ""
            if status.get('functional'):
                status_icon = "✅"
                status_class = "status-functional"
            elif status.get('installed'):
                status_icon = "⚠️"
                status_class = "status-installed"
            else:
                status_icon = "❌"
                status_class = "status-error"
            
            card_content = f"""[bold]{status_icon} {info.display_name}[/bold]
{info.description}

Branch: [cyan]{info.branch}[/cyan]
Max Qubits: [green]{info.max_qubits}[/green]
Priority: [yellow]{info.priority}[/yellow]
Dependencies: {', '.join(info.dependencies[:3])}{'...' if len(info.dependencies) > 3 else ''}"""
            
            # We'll use Static widgets for the cards
            await container.mount(
                Static(card_content, classes=f"variant-card {status_class}")
            )
    
    async def _populate_capabilities(self) -> None:
        """Populate the capabilities comparison table."""
        table = self.query_one("#capability-table", DataTable)
        
        # Add columns
        table.add_column("Capability", key="capability")
        for name in VARIANT_DEFINITIONS:
            table.add_column(name.replace('_', ' ').title(), key=name)
        
        # Capability rows
        capabilities = [
            ("Cirq Support", VariantCapability.CIRQ_SUPPORT),
            ("PennyLane Support", VariantCapability.PENNYLANE_SUPPORT),
            ("Qiskit Support", VariantCapability.QISKIT_SUPPORT),
            ("VQE", VariantCapability.VQE_SUPPORT),
            ("QAOA", VariantCapability.QAOA_SUPPORT),
            ("QNN", VariantCapability.QNN_SUPPORT),
            ("Gradients", VariantCapability.GRADIENT_COMPUTATION),
            ("GPU Acceleration", VariantCapability.GPU_ACCELERATION),
            ("Gate Fusion", VariantCapability.GATE_FUSION),
            ("Benchmarking", VariantCapability.BENCHMARKING),
            ("Noise Model", VariantCapability.NOISE_MODEL),
        ]
        
        for cap_name, cap_flag in capabilities:
            row = [cap_name]
            for var_name, var_info in VARIANT_DEFINITIONS.items():
                has_cap = bool(var_info.capabilities & cap_flag)
                row.append("✅" if has_cap else "❌")
            table.add_row(*row)
    
    async def _populate_recommendations_table(self) -> None:
        """Populate the task recommendations table."""
        table = self.query_one("#recommendations-table", DataTable)
        
        # Add columns
        table.add_column("Task Type", key="task")
        table.add_column("Recommended Variant", key="variant")
        table.add_column("Reason", key="reason")
        
        # Task recommendations
        tasks = [
            ("Benchmarking", "cirq_scalability", "Built-in LRET vs Cirq comparison"),
            ("Scalability Testing", "cirq_scalability", "Automated qubit scaling tests"),
            ("VQE Algorithm", "pennylane_hybrid", "Native VQE implementation"),
            ("QAOA Algorithm", "pennylane_hybrid", "Native QAOA with optimizer"),
            ("QNN Training", "pennylane_hybrid", "Gradient-based optimization"),
            ("Multi-Framework", "phase7_unified", "Supports Cirq, PennyLane, Qiskit"),
            ("GPU Execution", "phase7_unified", "cuQuantum integration"),
            ("Gate Optimization", "phase7_unified", "Automatic gate fusion"),
        ]
        
        for task, variant, reason in tasks:
            # Check if variant is functional
            if self._analyzer:
                status = self._analyzer.check_variant_status(variant)
                if status.get('functional'):
                    variant_display = f"✅ {variant}"
                else:
                    variant_display = f"⚠️ {variant} (not ready)"
            else:
                variant_display = variant
            
            table.add_row(task, variant_display, reason)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        
        if button_id == "btn-close":
            self.dismiss(None)
        
        elif button_id == "btn-refresh":
            await self._refresh_all()
        
        elif button_id == "btn-analyze" or button_id == "btn-analyze-all":
            await self._run_full_analysis()
    
    async def _refresh_all(self) -> None:
        """Refresh all data."""
        if self._analyzer:
            for name in self._analyzer.list_variants():
                self._analyzer.check_variant_status(name)
            
            self.notify("Status refreshed", severity="information")
    
    async def _run_full_analysis(self) -> None:
        """Run full analysis on all variants."""
        if not self._analyzer:
            self.notify("Analyzer not available", severity="error")
            return
        
        self.is_analyzing = True
        progress = self.query_one("#analysis-progress", ProgressBar)
        status = self.query_one("#analysis-status", Static)
        
        run_benchmarks = self.query_one("#chk-benchmarks", Checkbox).value
        
        variants = self._analyzer.list_variants()
        total = len(variants)
        
        try:
            for i, variant in enumerate(variants):
                status.update(f"Analyzing {variant}...")
                progress.update(progress=((i + 1) / total) * 100)
                
                result = self._analyzer.analyze_variant(
                    variant,
                    run_tests=True,
                    run_benchmarks=run_benchmarks
                )
                self._analysis_results[variant] = result
                
                await asyncio.sleep(0.1)  # Allow UI to update
            
            # Update analysis table
            await self._update_analysis_table()
            
            # Run comparison
            self._comparison_result = self._analyzer.compare_variants()
            
            # Update recommendations
            await self._update_recommendations()
            
            status.update("Analysis complete!")
            progress.update(progress=100)
            self.notify("Analysis complete", severity="success")
            
        except Exception as e:
            status.update(f"Error: {e}")
            self.notify(f"Analysis failed: {e}", severity="error")
        
        finally:
            self.is_analyzing = False
    
    async def _update_analysis_table(self) -> None:
        """Update the analysis results table."""
        table = self.query_one("#analysis-table", DataTable)
        table.clear(columns=True)
        
        # Add columns
        table.add_column("Variant", key="variant")
        table.add_column("Status", key="status")
        table.add_column("Tests Passed", key="passed")
        table.add_column("Tests Failed", key="failed")
        table.add_column("Time (ms)", key="time")
        
        for variant, result in self._analysis_results.items():
            status_icon = "✅" if result.tests_failed == 0 else "⚠️"
            if result.errors:
                status_icon = "❌"
            
            table.add_row(
                variant,
                status_icon,
                str(result.tests_passed),
                str(result.tests_failed),
                f"{result.analysis_time_ms:.1f}"
            )
    
    async def _update_recommendations(self) -> None:
        """Update recommendations panel."""
        panel = self.query_one("#recommendations-text", Static)
        
        if not self._comparison_result:
            panel.update("Run analysis to see recommendations")
            return
        
        recommendations = self._comparison_result.recommendations
        
        if recommendations:
            text = "\n".join(f"• {rec}" for rec in recommendations)
        else:
            text = "No specific recommendations at this time."
        
        # Add variant-specific recommendations
        for variant, result in self._analysis_results.items():
            if result.recommendations:
                text += f"\n\n[bold]{variant}:[/bold]"
                for rec in result.recommendations[:3]:
                    text += f"\n  • {rec}"
        
        panel.update(text)
    
    def action_cancel(self) -> None:
        """Cancel and close dialog."""
        self.dismiss(None)
    
    async def action_refresh(self) -> None:
        """Refresh data."""
        await self._refresh_all()


async def show_variant_analysis_dialog(app: Any) -> Optional[Dict[str, Any]]:
    """Show the variant analysis dialog.
    
    Args:
        app: Parent Textual app
        
    Returns:
        Analysis results or None if cancelled
    """
    dialog = VariantAnalysisDialog()
    return await app.push_screen(dialog)
