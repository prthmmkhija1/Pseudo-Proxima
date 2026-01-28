"""TUI dialog for LRET vs Cirq benchmark comparison.

This dialog provides an interactive interface for:
- Running new benchmarks
- Viewing benchmark results
- Generating visualization plots
- Exporting reports
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)


class LRETBenchmarkDialog(ModalScreen):
    """Dialog for LRET vs Cirq benchmark comparison.
    
    Provides interface to:
    - Configure and run scalability benchmarks
    - View results in table format
    - Generate comparison plots
    - Export reports and CSV data
    
    Keybindings:
        - ESC: Close dialog
        - r: Run new benchmark
        - p: Generate plots
        - e: Export report
    """
    
    CSS = """
    LRETBenchmarkDialog {
        align: center middle;
    }
    
    LRETBenchmarkDialog > Container {
        width: 90%;
        height: 85%;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        color: $text;
        padding: 1;
        background: $primary;
        margin-bottom: 1;
    }
    
    .benchmark-tabs {
        height: 100%;
    }
    
    .config-section {
        padding: 1;
        margin-bottom: 1;
    }
    
    .config-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .config-label {
        width: 20;
        padding-top: 1;
    }
    
    .config-input {
        width: 15;
    }
    
    .results-table {
        height: 100%;
    }
    
    .button-row {
        height: 3;
        align: center middle;
        margin-top: 1;
        margin-bottom: 1;
    }
    
    .button-row Button {
        margin: 0 1;
    }
    
    .action-primary {
        background: $success;
    }
    
    .action-secondary {
        background: $primary;
    }
    
    .progress-container {
        height: 5;
        padding: 1;
        margin-top: 1;
    }
    
    .log-container {
        height: 12;
        border: round $primary;
        margin-top: 1;
    }
    
    .summary-box {
        border: round $success;
        padding: 1;
        margin-top: 1;
        height: auto;
    }
    
    .summary-title {
        text-style: bold;
        color: $success;
    }
    
    .stat-value {
        color: $text;
        text-style: bold;
    }
    
    .status-running {
        color: $warning;
    }
    
    .status-complete {
        color: $success;
    }
    
    .status-error {
        color: $error;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("r", "run_benchmark", "Run Benchmark"),
        ("p", "generate_plots", "Generate Plots"),
        ("e", "export_report", "Export Report"),
    ]
    
    def __init__(self, name: str = None) -> None:
        """Initialize the benchmark dialog."""
        super().__init__(name=name)
        self._benchmark_result = None
        self._is_running = False
    
    def compose(self) -> ComposeResult:
        """Compose the dialog layout."""
        with Container():
            yield Static("LRET vs Cirq Benchmark Comparison", classes="dialog-title")
            
            with TabbedContent(classes="benchmark-tabs"):
                # Tab 1: Run Benchmark
                with TabPane("Run Benchmark", id="tab-run"):
                    yield from self._compose_run_tab()
                
                # Tab 2: View Results
                with TabPane("Results", id="tab-results"):
                    yield from self._compose_results_tab()
                
                # Tab 3: Visualization
                with TabPane("Visualize", id="tab-visualize"):
                    yield from self._compose_visualize_tab()
    
    def _compose_run_tab(self) -> ComposeResult:
        """Compose the benchmark configuration tab."""
        with ScrollableContainer():
            # Configuration section
            with Container(classes="config-section"):
                yield Static("Benchmark Configuration", classes="summary-title")
                
                # Qubit range
                with Horizontal(classes="config-row"):
                    yield Label("Min Qubits:", classes="config-label")
                    yield Input("4", id="input-min-qubits", classes="config-input")
                    yield Label("Max Qubits:", classes="config-label")
                    yield Input("12", id="input-max-qubits", classes="config-input")
                
                # Circuit depth
                with Horizontal(classes="config-row"):
                    yield Label("Circuit Depth:", classes="config-label")
                    yield Input("20", id="input-depth", classes="config-input")
                    yield Label("Shots:", classes="config-label")
                    yield Input("1024", id="input-shots", classes="config-input")
                
                # Circuit type selection
                with Horizontal(classes="config-row"):
                    yield Label("Circuit Type:", classes="config-label")
                    yield Input("random", id="input-circuit-type", classes="config-input")
                
                # Options
                with Horizontal(classes="config-row"):
                    yield Label("Export CSV:", classes="config-label")
                    yield Switch(value=True, id="switch-export-csv")
                    yield Label("Auto-Plot:", classes="config-label")
                    yield Switch(value=True, id="switch-auto-plot")
            
            # Progress section
            with Container(classes="progress-container"):
                yield Label("Progress:", id="label-progress")
                yield ProgressBar(id="progress-bar", show_eta=True)
            
            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("▶ Run Benchmark", id="btn-run", variant="success")
                yield Button("⏹ Stop", id="btn-stop", variant="error", disabled=True)
                yield Button("🗑 Clear Results", id="btn-clear", variant="default")
            
            # Log output
            with Container(classes="log-container"):
                yield RichLog(id="benchmark-log", highlight=True, markup=True)
    
    def _compose_results_tab(self) -> ComposeResult:
        """Compose the results viewing tab."""
        with ScrollableContainer():
            # Summary box
            with Container(classes="summary-box"):
                yield Static("Benchmark Summary", classes="summary-title")
                with Horizontal():
                    yield Static("Total Runs: ", id="summary-runs")
                    yield Static("Avg Speedup: ", id="summary-speedup")
                    yield Static("Avg Fidelity: ", id="summary-fidelity")
            
            # Results table
            yield DataTable(id="results-table", classes="results-table")
            
            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("📄 Load CSV", id="btn-load-csv", variant="primary")
                yield Button("💾 Export CSV", id="btn-export-csv", variant="primary")
                yield Button("🔄 Refresh", id="btn-refresh", variant="default")
    
    def _compose_visualize_tab(self) -> ComposeResult:
        """Compose the visualization tab."""
        with ScrollableContainer():
            yield Static("Generate Benchmark Visualizations", classes="summary-title")
            
            # Plot options
            with Container(classes="config-section"):
                with Horizontal(classes="config-row"):
                    yield Label("Output Format:", classes="config-label")
                    yield Input("png", id="input-plot-format", classes="config-input")
                
                with Horizontal(classes="config-row"):
                    yield Label("Output Directory:", classes="config-label")
                    yield Input("./benchmarks", id="input-output-dir")
            
            # Generate buttons
            with Horizontal(classes="button-row"):
                yield Button("📊 4-Panel Comparison", id="btn-plot-comparison", variant="success")
                yield Button("📈 Speedup Trend", id="btn-plot-speedup", variant="primary")
                yield Button("🎯 Fidelity Analysis", id="btn-plot-fidelity", variant="primary")
            
            with Horizontal(classes="button-row"):
                yield Button("📝 Generate Report", id="btn-generate-report", variant="success")
                yield Button("📂 Open Output Folder", id="btn-open-folder", variant="default")
            
            # Status
            yield Static("", id="plot-status")
    
    def on_mount(self) -> None:
        """Handle mount event."""
        # Initialize results table
        table = self.query_one("#results-table", DataTable)
        table.add_columns(
            "Qubits", "Depth", "LRET (ms)", "Cirq (ms)", 
            "Speedup", "Fidelity", "Rank"
        )
        
        # Log initial message
        log = self.query_one("#benchmark-log", RichLog)
        log.write("[bold blue]LRET vs Cirq Benchmark Tool Ready[/]")
        log.write("Configure parameters and click 'Run Benchmark' to start.")
    
    def action_close(self) -> None:
        """Close the dialog."""
        self.dismiss()
    
    async def action_run_benchmark(self) -> None:
        """Run benchmark action."""
        await self._run_benchmark()
    
    async def action_generate_plots(self) -> None:
        """Generate plots action."""
        await self._generate_comparison_plot()
    
    async def action_export_report(self) -> None:
        """Export report action."""
        await self._generate_report()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        
        if button_id == "btn-run":
            await self._run_benchmark()
        elif button_id == "btn-stop":
            self._stop_benchmark()
        elif button_id == "btn-clear":
            self._clear_results()
        elif button_id == "btn-load-csv":
            await self._load_csv()
        elif button_id == "btn-export-csv":
            await self._export_csv()
        elif button_id == "btn-refresh":
            self._refresh_results()
        elif button_id == "btn-plot-comparison":
            await self._generate_comparison_plot()
        elif button_id == "btn-plot-speedup":
            await self._generate_speedup_plot()
        elif button_id == "btn-plot-fidelity":
            await self._generate_fidelity_plot()
        elif button_id == "btn-generate-report":
            await self._generate_report()
        elif button_id == "btn-open-folder":
            self._open_output_folder()
    
    async def _run_benchmark(self) -> None:
        """Execute the benchmark run."""
        if self._is_running:
            return
        
        self._is_running = True
        log = self.query_one("#benchmark-log", RichLog)
        progress = self.query_one("#progress-bar", ProgressBar)
        btn_run = self.query_one("#btn-run", Button)
        btn_stop = self.query_one("#btn-stop", Button)
        
        btn_run.disabled = True
        btn_stop.disabled = False
        
        try:
            # Get configuration
            min_qubits = int(self.query_one("#input-min-qubits", Input).value)
            max_qubits = int(self.query_one("#input-max-qubits", Input).value)
            depth = int(self.query_one("#input-depth", Input).value)
            shots = int(self.query_one("#input-shots", Input).value)
            circuit_type = self.query_one("#input-circuit-type", Input).value
            export_csv = self.query_one("#switch-export-csv", Switch).value
            
            log.write(f"\n[bold green]Starting benchmark...[/]")
            log.write(f"  Qubits: {min_qubits} - {max_qubits}")
            log.write(f"  Depth: {depth}, Shots: {shots}")
            log.write(f"  Circuit Type: {circuit_type}")
            
            # Initialize progress
            total_steps = max_qubits - min_qubits + 1
            progress.update(total=total_steps, progress=0)
            
            # Try to run actual benchmark
            try:
                from proxima.backends.lret.cirq_scalability import (
                    LRETCirqScalabilityAdapter,
                    CirqScalabilityMetrics,
                    BenchmarkResult,
                )
                
                adapter = LRETCirqScalabilityAdapter()
                
                if adapter.is_available():
                    # Run the actual benchmark
                    log.write("[yellow]Running actual LRET benchmark...[/]")
                    self._benchmark_result = await asyncio.to_thread(
                        adapter.run_scalability_benchmark,
                        qubit_range=(min_qubits, max_qubits),
                        depth=depth,
                        shots=shots,
                        circuit_type=circuit_type,
                        export_csv=export_csv,
                    )
                else:
                    # Generate mock results
                    log.write("[yellow]LRET not installed, generating mock results...[/]")
                    self._benchmark_result = await self._generate_mock_results(
                        min_qubits, max_qubits, depth, progress, log
                    )
            except ImportError:
                log.write("[yellow]Backend not available, generating mock results...[/]")
                self._benchmark_result = await self._generate_mock_results(
                    min_qubits, max_qubits, depth, progress, log
                )
            
            # Update results table
            self._update_results_table()
            
            # Update summary
            self._update_summary()
            
            log.write(f"\n[bold green]Benchmark complete![/]")
            if self._benchmark_result.summary:
                summary = self._benchmark_result.summary
                log.write(f"  Average Speedup: {summary.get('avg_speedup', 0):.2f}x")
                log.write(f"  Average Fidelity: {summary.get('avg_fidelity', 0):.4f}")
            
            # Auto-plot if enabled
            if self.query_one("#switch-auto-plot", Switch).value:
                await self._generate_comparison_plot()
            
        except Exception as e:
            log.write(f"[bold red]Error: {e}[/]")
        finally:
            self._is_running = False
            btn_run.disabled = False
            btn_stop.disabled = True
    
    async def _generate_mock_results(
        self, 
        min_qubits: int, 
        max_qubits: int,
        depth: int,
        progress: ProgressBar,
        log: RichLog,
    ):
        """Generate mock benchmark results for demo."""
        import random
        
        try:
            from proxima.backends.lret.cirq_scalability import (
                CirqScalabilityMetrics,
                BenchmarkResult,
            )
        except ImportError:
            # Create simple mock classes if imports fail
            return None
        
        metrics = []
        total = max_qubits - min_qubits + 1
        
        for i, n_qubits in enumerate(range(min_qubits, max_qubits + 1)):
            await asyncio.sleep(0.3)  # Simulate work
            
            # Generate realistic-looking mock data
            lret_time = 0.5 + n_qubits * 0.2 + random.uniform(-0.1, 0.1)
            cirq_time = 1.0 + (2 ** (n_qubits / 4)) * 0.5 + random.uniform(-0.2, 0.2)
            speedup = cirq_time / lret_time
            rank = min(int(2 ** (n_qubits / 2.5)), 100)
            fidelity = 0.999 - (n_qubits - min_qubits) * 0.001 + random.uniform(-0.001, 0.001)
            
            metrics.append(CirqScalabilityMetrics(
                lret_time_ms=lret_time,
                cirq_fdm_time_ms=cirq_time,
                speedup_factor=speedup,
                lret_final_rank=rank,
                fidelity=max(0.99, fidelity),
                trace_distance=1.0 - max(0.99, fidelity),
                qubit_count=n_qubits,
                circuit_depth=depth,
            ))
            
            progress.update(progress=i + 1)
            log.write(f"  {n_qubits} qubits: {speedup:.2f}x speedup")
        
        result = BenchmarkResult(metrics=metrics)
        result.compute_summary()
        return result
    
    def _stop_benchmark(self) -> None:
        """Stop the running benchmark."""
        self._is_running = False
        log = self.query_one("#benchmark-log", RichLog)
        log.write("[bold yellow]Benchmark stopped by user[/]")
    
    def _clear_results(self) -> None:
        """Clear benchmark results."""
        self._benchmark_result = None
        table = self.query_one("#results-table", DataTable)
        table.clear()
        
        log = self.query_one("#benchmark-log", RichLog)
        log.clear()
        log.write("[bold blue]Results cleared.[/]")
        
        self._update_summary()
    
    def _update_results_table(self) -> None:
        """Update the results data table."""
        table = self.query_one("#results-table", DataTable)
        table.clear()
        
        if not self._benchmark_result or not self._benchmark_result.metrics:
            return
        
        for m in self._benchmark_result.metrics:
            table.add_row(
                str(m.qubit_count),
                str(m.circuit_depth),
                f"{m.lret_time_ms:.2f}",
                f"{m.cirq_fdm_time_ms:.2f}",
                f"{m.speedup_factor:.2f}x",
                f"{m.fidelity:.4f}",
                str(m.lret_final_rank),
            )
    
    def _update_summary(self) -> None:
        """Update the summary display."""
        runs_label = self.query_one("#summary-runs", Static)
        speedup_label = self.query_one("#summary-speedup", Static)
        fidelity_label = self.query_one("#summary-fidelity", Static)
        
        if not self._benchmark_result or not self._benchmark_result.summary:
            runs_label.update("Total Runs: 0")
            speedup_label.update("Avg Speedup: N/A")
            fidelity_label.update("Avg Fidelity: N/A")
            return
        
        summary = self._benchmark_result.summary
        runs_label.update(f"Total Runs: {summary.get('total_runs', 0)}")
        speedup_label.update(f"Avg Speedup: {summary.get('avg_speedup', 0):.2f}x")
        fidelity_label.update(f"Avg Fidelity: {summary.get('avg_fidelity', 0):.4f}")
    
    async def _load_csv(self) -> None:
        """Load benchmark results from CSV."""
        log = self.query_one("#benchmark-log", RichLog)
        
        try:
            from proxima.backends.lret.visualization import load_benchmark_csv
            
            # Try loading from default location
            csv_path = Path("./benchmarks/lret_cirq_comparison.csv")
            if csv_path.exists():
                self._benchmark_result = load_benchmark_csv(csv_path)
                self._update_results_table()
                self._update_summary()
                log.write(f"[green]Loaded results from {csv_path}[/]")
            else:
                log.write("[yellow]No benchmark CSV found. Run a benchmark first.[/]")
        except Exception as e:
            log.write(f"[red]Error loading CSV: {e}[/]")
    
    async def _export_csv(self) -> None:
        """Export results to CSV."""
        if not self._benchmark_result:
            log = self.query_one("#benchmark-log", RichLog)
            log.write("[yellow]No results to export. Run a benchmark first.[/]")
            return
        
        # Export is handled by the adapter
        log = self.query_one("#benchmark-log", RichLog)
        if self._benchmark_result.csv_path:
            log.write(f"[green]Results already exported to {self._benchmark_result.csv_path}[/]")
        else:
            log.write("[yellow]Export CSV during benchmark run or regenerate.[/]")
    
    def _refresh_results(self) -> None:
        """Refresh the results display."""
        self._update_results_table()
        self._update_summary()
    
    async def _generate_comparison_plot(self) -> None:
        """Generate 4-panel comparison plot."""
        status = self.query_one("#plot-status", Static)
        
        if not self._benchmark_result or not self._benchmark_result.metrics:
            status.update("[yellow]No benchmark data. Run a benchmark first.[/]")
            return
        
        try:
            from proxima.backends.lret.visualization import (
                plot_lret_cirq_comparison,
                PlotConfig,
            )
            
            output_dir = self.query_one("#input-output-dir", Input).value
            plot_format = self.query_one("#input-plot-format", Input).value
            
            config = PlotConfig(save_format=plot_format)
            
            status.update("[yellow]Generating 4-panel comparison plot...[/]")
            
            output_path = await asyncio.to_thread(
                plot_lret_cirq_comparison,
                self._benchmark_result,
                Path(output_dir) / f"comparison_plot.{plot_format}",
                config,
            )
            
            status.update(f"[green]Plot saved: {output_path}[/]")
            
        except ImportError as e:
            status.update(f"[red]Missing dependency: {e}[/]")
        except Exception as e:
            status.update(f"[red]Error: {e}[/]")
    
    async def _generate_speedup_plot(self) -> None:
        """Generate speedup trend plot."""
        status = self.query_one("#plot-status", Static)
        
        if not self._benchmark_result:
            status.update("[yellow]No benchmark data.[/]")
            return
        
        try:
            from proxima.backends.lret.visualization import plot_speedup_trend, PlotConfig
            
            output_dir = self.query_one("#input-output-dir", Input).value
            plot_format = self.query_one("#input-plot-format", Input).value
            
            config = PlotConfig(save_format=plot_format)
            
            status.update("[yellow]Generating speedup trend plot...[/]")
            
            output_path = await asyncio.to_thread(
                plot_speedup_trend,
                self._benchmark_result,
                Path(output_dir) / f"speedup_trend.{plot_format}",
                config,
            )
            
            status.update(f"[green]Plot saved: {output_path}[/]")
            
        except Exception as e:
            status.update(f"[red]Error: {e}[/]")
    
    async def _generate_fidelity_plot(self) -> None:
        """Generate fidelity analysis plot."""
        status = self.query_one("#plot-status", Static)
        
        if not self._benchmark_result:
            status.update("[yellow]No benchmark data.[/]")
            return
        
        try:
            from proxima.backends.lret.visualization import plot_fidelity_analysis, PlotConfig
            
            output_dir = self.query_one("#input-output-dir", Input).value
            plot_format = self.query_one("#input-plot-format", Input).value
            
            config = PlotConfig(save_format=plot_format)
            
            status.update("[yellow]Generating fidelity analysis plot...[/]")
            
            output_path = await asyncio.to_thread(
                plot_fidelity_analysis,
                self._benchmark_result,
                Path(output_dir) / f"fidelity_analysis.{plot_format}",
                config,
            )
            
            status.update(f"[green]Plot saved: {output_path}[/]")
            
        except Exception as e:
            status.update(f"[red]Error: {e}[/]")
    
    async def _generate_report(self) -> None:
        """Generate markdown benchmark report."""
        status = self.query_one("#plot-status", Static)
        
        if not self._benchmark_result:
            status.update("[yellow]No benchmark data.[/]")
            return
        
        try:
            from proxima.backends.lret.visualization import generate_benchmark_report
            
            output_dir = self.query_one("#input-output-dir", Input).value
            
            status.update("[yellow]Generating benchmark report...[/]")
            
            output_path = await asyncio.to_thread(
                generate_benchmark_report,
                self._benchmark_result,
                Path(output_dir) / "benchmark_report.md",
            )
            
            status.update(f"[green]Report saved: {output_path}[/]")
            
        except Exception as e:
            status.update(f"[red]Error: {e}[/]")
    
    def _open_output_folder(self) -> None:
        """Open the output folder in file explorer."""
        import subprocess
        import sys
        
        output_dir = Path(self.query_one("#input-output-dir", Input).value)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['explorer', str(output_dir)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', str(output_dir)])
            else:
                subprocess.Popen(['xdg-open', str(output_dir)])
        except Exception:
            pass
