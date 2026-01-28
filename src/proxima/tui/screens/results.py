"""Results screen for Proxima TUI.

Results browser with probability visualization.
"""

from typing import List, Dict
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, DataTable, ListView, ListItem, Label
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme
from ..styles.icons import PROGRESS_FILLED, PROGRESS_EMPTY
from pathlib import Path
from datetime import datetime
import json
import time

try:
    from proxima.data.export import (
        ExportEngine, ExportFormat, ReportData, ExportOptions
    )
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

# Singleton engine for exports
_export_engine = None

def get_export_engine():
    """Get or create the export engine singleton."""
    global _export_engine
    if _export_engine is None and EXPORT_AVAILABLE:
        _export_engine = ExportEngine()
    return _export_engine

try:
    from proxima.resources.session import SessionManager
    from proxima.data.results import ResultStore
    from pathlib import Path
    RESULTS_AVAILABLE = True
except ImportError:
    RESULTS_AVAILABLE = False

try:
    from ..dialogs.results import ResultStatsDialog, ResultCompareDialog
    RESULT_DIALOGS_AVAILABLE = True
except ImportError:
    RESULT_DIALOGS_AVAILABLE = False


class ResultsScreen(BaseScreen):
    """Results browser screen.
    
    Shows:
    - List of results
    - Result details with probability distribution
    - Export options
    """
    
    SCREEN_NAME = "results"
    SCREEN_TITLE = "Results Browser"
    
    DEFAULT_CSS = """
    ResultsScreen .results-list {
        width: 30;
        height: 100%;
        border-right: solid $primary-darken-2;
        background: $surface;
    }
    
    ResultsScreen .results-list-title {
        padding: 1;
        text-style: bold;
        border-bottom: solid $primary-darken-3;
    }
    
    ResultsScreen .selection-info {
        padding: 0 1;
        height: auto;
        color: $text-muted;
        border-bottom: solid $primary-darken-3;
    }
    
    ResultsScreen .result-detail {
        width: 1fr;
        height: 100%;
        padding: 1;
    }
    
    ResultsScreen .result-header {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    ResultsScreen .probability-section {
        height: 1fr;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
        overflow-y: auto;
    }
    
    ResultsScreen .actions-section {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    
    ResultsScreen .action-btn {
        margin-right: 1;
    }
    """
    
    BINDINGS = [
        ("space", "toggle_selection", "Toggle Select"),
        ("ctrl+a", "select_all", "Select All"),
        ("escape", "clear_selection", "Clear Selection"),
    ]
    
    def __init__(self, **kwargs):
        """Initialize results screen with multi-select support."""
        super().__init__(**kwargs)
        self._selected_result = None
        self._selected_results: List[Dict] = []  # For multi-select
    
    def compose_main(self):
        """Compose the results screen content."""
        with Horizontal(classes="main-content"):
            # Results list
            with Vertical(classes="results-list"):
                yield Static("Results", classes="results-list-title")
                yield Static("Select: Space | Compare: Select 2+", id="selection-info", classes="selection-info")
                yield ResultsListView()
            
            # Result detail
            with Vertical(classes="result-detail"):
                yield ResultHeaderPanel(classes="result-header")
                yield ProbabilityDistribution(classes="probability-section")
                
                with Horizontal(classes="actions-section"):
                    yield Button("View Full Stats", id="btn-stats", classes="action-btn")
                    yield Button("Export JSON", id="btn-json", classes="action-btn")
                    yield Button("Export HTML", id="btn-html", classes="action-btn")
                    yield Button("Compare", id="btn-compare", classes="action-btn")
    
    def action_toggle_selection(self) -> None:
        """Toggle selection of current result."""
        if not self._selected_result:
            self.notify("No result selected", severity="warning")
            return
        
        result = self._selected_result
        result_id = result.get('id', '')
        
        # Check if already selected
        existing = next((r for r in self._selected_results if r.get('id') == result_id), None)
        if existing:
            self._selected_results.remove(existing)
            self.notify(f"Deselected: {result.get('name', 'Unknown')}")
        else:
            self._selected_results.append(result)
            self.notify(f"Selected: {result.get('name', 'Unknown')} ({len(self._selected_results)} total)")
        
        self._update_selection_info()
    
    def action_select_all(self) -> None:
        """Select all results."""
        try:
            list_view = self.query_one(ResultsListView)
            for item in list_view.query("ListItem"):
                if hasattr(item, '_result_data') and item._result_data:
                    result_id = item._result_data.get('id', '')
                    if not any(r.get('id') == result_id for r in self._selected_results):
                        self._selected_results.append(item._result_data)
            self.notify(f"Selected all ({len(self._selected_results)} results)")
            self._update_selection_info()
        except Exception:
            pass
    
    def action_clear_selection(self) -> None:
        """Clear all selections."""
        self._selected_results = []
        self.notify("Selection cleared")
        self._update_selection_info()
    
    def _update_selection_info(self) -> None:
        """Update the selection info display."""
        try:
            info = self.query_one("#selection-info", Static)
            count = len(self._selected_results)
            if count == 0:
                info.update("Select: Space | Compare: Select 2+")
            elif count == 1:
                info.update(f"1 selected - Select more to compare")
            else:
                info.update(f"{count} selected - Ready to compare!")
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-stats":
            self._view_full_stats()
        elif button_id == "btn-json":
            self._export_json()
        elif button_id == "btn-html":
            self._export_html()
        elif button_id == "btn-compare":
            self._compare_results()



    def _view_full_stats(self) -> None:
        """View full statistics for selected result using stats dialog."""
        result = getattr(self, '_selected_result', None)
        
        if RESULT_DIALOGS_AVAILABLE:
            self.app.push_screen(ResultStatsDialog(result=result))
        else:
            # Fallback to notifications
            if not result:
                self.notify("Please select a result first", severity="warning")
                return
            
            stats_text = [
                f"=== Full Statistics for {result.get('name', 'Unknown')} ===",
                f"Execution ID: {result.get('id', 'N/A')}",
                f"Status: {result.get('status', 'Unknown')}",
                f"Backend: {result.get('backend', 'Unknown')}",
                f"Duration: {result.get('duration', 0):.2f}s",
                f"Total Shots: {result.get('total_shots', 0)}",
                f"Success Rate: {result.get('success_rate', 0):.1%}",
                f"Average Fidelity: {result.get('avg_fidelity', 0):.4f}",
            ]
            
            # Show detailed stats
            self.notify("\n".join(stats_text))
            self.notify("Install dialogs for full statistics view", severity="information")

    def _export_json(self) -> None:
        """Export results to JSON file using ExportEngine."""
        if not hasattr(self, '_selected_result') or not self._selected_result:
            self.notify("Please select a result first", severity="warning")
            return
        
        result = self._selected_result
        result_id = result.get('id', 'unknown')
        export_path = Path.home() / f"proxima_result_{result_id}.json"
        
        engine = get_export_engine()
        if engine and EXPORT_AVAILABLE:
            try:
                # Build ReportData from result
                report_data = ReportData(
                    title=f"Proxima Result: {result.get('name', 'Unknown')}",
                    generated_at=time.time(),
                    summary={
                        "id": result.get('id'),
                        "name": result.get('name'),
                        "status": result.get('status'),
                        "backend": result.get('backend'),
                        "duration": result.get('duration', 0),
                        "success_rate": result.get('success_rate', 0),
                        "avg_fidelity": result.get('avg_fidelity', 0),
                    },
                    raw_results=[result],
                    metadata=result.get('metadata', {}),
                )
                
                export_result = engine.export(
                    report_data,
                    format=ExportFormat.JSON,
                    output_path=export_path,
                    pretty_print=True
                )
                
                if export_result.success:
                    self.notify(f"[+] Exported to {export_path}", severity="success")
                else:
                    self.notify(f"[-] Export failed: {export_result.error}", severity="error")
            except Exception as e:
                self.notify(f"[-] Export failed: {e}", severity="error")
        else:
            # Fallback: direct JSON export
            try:
                with open(export_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                self.notify(f"[+] Exported to {export_path}", severity="success")
            except Exception as e:
                self.notify(f"[-] Export failed: {e}", severity="error")

    def _export_html(self) -> None:
        """Export results to HTML file using ExportEngine."""
        if not hasattr(self, '_selected_result') or not self._selected_result:
            self.notify("Please select a result first", severity="warning")
            return
        
        result = self._selected_result
        result_id = result.get('id', 'unknown')
        export_path = Path.home() / f"proxima_result_{result_id}.html"
        
        engine = get_export_engine()
        if engine and EXPORT_AVAILABLE:
            try:
                # Build ReportData from result
                report_data = ReportData(
                    title=f"Proxima Result: {result.get('name', 'Unknown')}",
                    generated_at=time.time(),
                    summary={
                        "id": result.get('id'),
                        "name": result.get('name'),
                        "status": result.get('status'),
                        "backend": result.get('backend'),
                        "duration": result.get('duration', 0),
                        "success_rate": result.get('success_rate', 0),
                        "avg_fidelity": result.get('avg_fidelity', 0),
                    },
                    raw_results=[result],
                    metadata=result.get('metadata', {}),
                    insights=result.get('insights', []),
                )
                
                export_result = engine.export(
                    report_data,
                    format=ExportFormat.HTML,
                    output_path=export_path,
                    html_inline_styles=True
                )
                
                if export_result.success:
                    self.notify(f"[+] Exported to {export_path}", severity="success")
                else:
                    self.notify(f"[-] Export failed: {export_result.error}", severity="error")
            except Exception as e:
                self.notify(f"[-] Export failed: {e}", severity="error")
        else:
            # Fallback: basic HTML export
            try:
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Proxima Result {result_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00ff9f; }}
        h2 {{ color: #00d9ff; }}
        .metric {{ margin: 10px 0; padding: 10px; background: #16213e; border-radius: 5px; }}
        .success {{ color: #00ff9f; }}
        .label {{ color: #888; }}
    </style>
</head>
<body>
<h1>Execution Result: {result.get('name', 'Unknown')}</h1>
<div class="metric"><span class="label">ID:</span> {result.get('id', 'N/A')}</div>
<div class="metric"><span class="label">Status:</span> <span class="success">{result.get('status', 'Unknown')}</span></div>
<div class="metric"><span class="label">Backend:</span> {result.get('backend', 'Unknown')}</div>
<div class="metric"><span class="label">Duration:</span> {result.get('duration', 0):.2f}s</div>
<div class="metric"><span class="label">Success Rate:</span> {result.get('success_rate', 0):.1%}</div>
<div class="metric"><span class="label">Average Fidelity:</span> {result.get('avg_fidelity', 0):.4f}</div>
<h2>Generated</h2>
<p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>"""
                with open(export_path, 'w') as f:
                    f.write(html_content)
                self.notify(f"[+] Exported to {export_path}", severity="success")
            except Exception as e:
                self.notify(f"[-] Export failed: {e}", severity="error")

    def _compare_results(self) -> None:
        """Compare selected results using comparison dialog."""
        # Use multi-selected results first, fall back to single selected
        results = self._selected_results if self._selected_results else []
        
        if not results and self._selected_result:
            results = [self._selected_result]
        
        if len(results) < 2:
            self.notify("Select at least 2 results to compare (use Space to toggle selection)", severity="warning")
            return
        
        if RESULT_DIALOGS_AVAILABLE:
            self.app.push_screen(ResultCompareDialog(results=results))
        else:
            # Fallback to detailed comparison notifications
            self.notify(f"Comparing {len(results)} results:", severity="information")
            
            # Build comparison table
            metrics = ['name', 'backend', 'duration', 'success_rate', 'shots']
            
            for i, result in enumerate(results[:4]):  # Limit to 4 for display
                name = result.get('name', 'Unknown')
                backend = result.get('backend', 'Unknown')
                duration = result.get('duration', 0)
                success_rate = result.get('success_rate', 0)
                shots = result.get('shots', result.get('total_shots', 1024))
                
                self.notify(
                    f"[{i+1}] {name}: {backend} | {duration:.2f}s | {success_rate:.1%} success | {shots} shots"
                )
            
            # Summary
            if len(results) >= 2:
                avg_duration = sum(r.get('duration', 0) for r in results) / len(results)
                avg_success = sum(r.get('success_rate', 0) for r in results) / len(results)
                self.notify(f"Average: {avg_duration:.2f}s | {avg_success:.1%} success rate")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle result selection from the list view."""
        item = event.item
        
        # Check if this item has real result data attached
        if hasattr(item, '_result_data') and item._result_data:
            self._selected_result = item._result_data
            self.notify(f"Selected: {self._selected_result.get('name', 'Unknown')}")
        else:
            # Fallback for sample data - create a minimal result dict
            try:
                label = item.query_one(Label)
                # Get the label text - Label content is in _content or we can use str()
                if hasattr(label, '_content'):
                    result_name = str(label._content)
                else:
                    # Alternative: get text from the label's render result
                    result_name = str(label.render()) if hasattr(label, 'render') else "Unknown"
            except Exception:
                result_name = "Unknown"
            
            self._selected_result = {
                'id': result_name.replace('.json', ''),
                'name': result_name,
                'status': 'completed',
                'backend': 'Cirq',
                'duration': 0.245,
                'success_rate': 0.95,
                'qubits': 4,
                'shots': 1024,
                'total_shots': 1024,
                'avg_fidelity': 0.992,
            }
            self.notify(f"Selected: {result_name} (sample data)")
        
        # Update the header panel and distribution display
        self._update_result_display()
    
    def _update_result_display(self) -> None:
        """Update the result header and distribution panels with selected result data."""
        if not hasattr(self, '_selected_result') or not self._selected_result:
            return
        
        result = self._selected_result
        
        # Update header panel
        try:
            header = self.query_one(ResultHeaderPanel)
            header._result = result
            header.refresh()
        except Exception:
            pass
        
        # Update probability distribution
        try:
            distribution = self.query_one(ProbabilityDistribution)
            distribution._result = result
            distribution.refresh()
        except Exception:
            pass



class ResultsListView(ListView):
    """List of available results."""

    def on_mount(self) -> None:
        """Populate the results list."""
        # Try to load real results
        if RESULTS_AVAILABLE:
            try:
                storage_dir = Path.home() / ".proxima" / "results"
                self._load_real_results(storage_dir)
                return
            except Exception:
                pass
        
        # Fallback to sample data
        results = [
            "result_001.json",
            "result_002.json",
            "comparison_001.json",
            "bell_state_run.json",
            "ghz_4qubit.json",
        ]

        for result in results:
            self.append(ListItem(Label(result)))

    def _load_real_results(self, storage_dir: Path) -> None:
        """Load real results from storage."""
        if not storage_dir.exists():
            raise Exception("Storage directory does not exist")
        
        import json
        results_loaded = False
        
        for result_file in sorted(storage_dir.glob("*.json"), reverse=True)[:10]:
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                result_id = result_data.get('id', result_file.stem)[:8]
                name = result_data.get('name', 'Unnamed')
                backend = result_data.get('backend', 'Unknown')
                status = result_data.get('status', 'Unknown')
                
                # Format status with icons
                status_display = {
                    'completed': '? Done',
                    'success': '? Done',
                    'failed': '? Failed',
                    'error': '? Error',
                    'running': '? Running',
                }.get(status.lower() if isinstance(status, str) else 'unknown', status)
                
                # Format time
                import time
                created = result_data.get('created_at', result_data.get('timestamp', 0))
                if isinstance(created, str):
                    try:
                        from datetime import datetime
                        created = datetime.fromisoformat(created).timestamp()
                    except Exception:
                        created = time.time()
                
                elapsed = time.time() - created
                if elapsed < 60:
                    time_str = f"{int(elapsed)}s ago"
                elif elapsed < 3600:
                    time_str = f"{int(elapsed / 60)}m ago"
                else:
                    time_str = f"{int(elapsed / 3600)}h ago"
                
                # Store result data for selection and append to list
                display_text = f"{name} ({status_display})"
                item = ListItem(Label(display_text))
                item._result_data = result_data
                self.append(item)
                results_loaded = True
            except Exception:
                continue
        
        if not results_loaded:
            raise Exception("No results loaded")


class ResultHeaderPanel(Static):
    """Header panel showing result metadata."""
    
    def __init__(self, **kwargs):
        """Initialize with optional result data."""
        super().__init__(**kwargs)
        self._result = None
    
    def render(self) -> Text:
        """Render the result header."""
        theme = get_theme()
        text = Text()
        
        # Get result data or use defaults
        result = self._result or {}
        name = result.get('name', 'Simulation Results')
        backend = result.get('backend', 'Cirq')
        simulator = result.get('simulator', 'StateVector')
        qubits = result.get('qubits', 4)
        shots = result.get('shots', result.get('total_shots', 1024))
        duration = result.get('duration', 0.245)
        
        # Format duration
        if isinstance(duration, (int, float)):
            if duration < 1:
                duration_str = f"{int(duration * 1000)}ms"
            else:
                duration_str = f"{duration:.2f}s"
        else:
            duration_str = str(duration)
        
        # Title line
        text.append(name, style=f"bold {theme.primary}")
        text.append("\n\n")
        
        # Metadata
        text.append("Backend: ", style=theme.fg_muted)
        text.append(f"{backend} ({simulator})", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Qubits: ", style=theme.fg_muted)
        text.append(str(qubits), style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Shots: ", style=theme.fg_muted)
        text.append(str(shots), style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Time: ", style=theme.fg_muted)
        text.append(duration_str, style=theme.fg_base)
        
        return text


class ProbabilityDistribution(Static):
    """Probability distribution visualization."""
    
    BAR_WIDTH = 40
    
    def __init__(self, **kwargs):
        """Initialize with optional result data."""
        super().__init__(**kwargs)
        self._result = None
    
    def render(self) -> Text:
        """Render the probability distribution."""
        theme = get_theme()
        text = Text()
        
        text.append("Probability Distribution:", style=f"bold {theme.fg_base}")
        text.append("\n\n")
        
        # Get probability data from result or use sample
        result = self._result or {}
        probabilities = result.get('probabilities', result.get('distribution', None))
        
        if probabilities and isinstance(probabilities, dict):
            # Convert dict to list of tuples, sorted by probability
            prob_list = [(state, prob * 100 if prob <= 1 else prob) 
                        for state, prob in probabilities.items()]
            prob_list.sort(key=lambda x: x[1], reverse=True)
            probabilities = prob_list[:5]  # Show top 5
        else:
            # Sample probability data
            probabilities = [
                ("|0000⟩", 48.2),
                ("|1111⟩", 47.1),
                ("|0011⟩", 2.4),
                ("|1100⟩", 1.8),
                ("|others⟩", 0.5),
            ]
        
        for state, prob in probabilities:
            # State label
            display_state = state if state.startswith('|') else f"|{state}⟩"
            text.append(f"{display_state:<10}", style=f"bold {theme.accent}")
            
            # Bar
            filled = int(self.BAR_WIDTH * prob / 100)
            empty = self.BAR_WIDTH - filled
            
            text.append(PROGRESS_FILLED * filled, style=f"bold {theme.primary}")
            text.append(PROGRESS_EMPTY * empty, style=theme.fg_subtle)
            
            # Percentage
            text.append(f" {prob:>5.1f}%", style=theme.fg_muted)
            text.append("\n")
        
        # Pattern detection from result
        pattern = result.get('pattern', 'GHZ State')
        confidence = result.get('pattern_confidence', 0.96)
        
        text.append("\n")
        text.append("Pattern: ", style=theme.fg_muted)
        text.append(str(pattern), style=f"bold {theme.success}")
        text.append(f" (confidence: {confidence:.0%})", style=theme.fg_muted)
        text.append("\n")
        
        # Statistics from result
        entropy = result.get('entropy', 1.02)
        fidelity = result.get('fidelity', result.get('avg_fidelity', 0.992))
        gini = result.get('gini', 0.03)
        
        text.append("Entropy: ", style=theme.fg_muted)
        text.append(f"{entropy:.2f}", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        fidelity_pct = fidelity * 100 if fidelity <= 1 else fidelity
        text.append("Fidelity: ", style=theme.fg_muted)
        text.append(f"{fidelity_pct:.1f}%", style=f"bold {theme.success}")
        text.append("  │  ", style=theme.border)
        
        text.append("Gini: ", style=theme.fg_muted)
        text.append(f"{gini:.2f}", style=theme.fg_base)
        
        return text
