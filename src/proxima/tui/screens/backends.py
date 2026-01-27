"""Backends screen for Proxima TUI.

Backend management and comparison.
"""

from textual.containers import Horizontal, Vertical, Container, Grid
from textual.widgets import Static, Button, DataTable
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme
from ..styles.icons import get_health_icon


try:
    from proxima.backends.registry import BackendRegistry
    from proxima.backends.base import BackendStatus
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

try:
    from ..dialogs.backends import BackendComparisonDialog, BackendMetricsDialog, BackendConfigDialog
    DIALOGS_AVAILABLE = True
except ImportError:
    DIALOGS_AVAILABLE = False


class BackendsScreen(BaseScreen):
    """Backend management screen.
    
    Shows:
    - List of backends with health status
    - Backend details
    - Performance comparison
    """
    
    SCREEN_NAME = "backends"
    SCREEN_TITLE = "Backend Management"
    
    DEFAULT_CSS = """
    BackendsScreen .backends-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
        padding: 1;
    }
    
    BackendsScreen .backend-card {
        height: 8;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    BackendsScreen .backend-card:hover {
        border: solid $primary;
    }
    
    BackendsScreen .backend-card.-selected {
        border: solid $primary;
        background: $surface-lighten-1;
    }
    
    BackendsScreen .backend-card:focus {
        border: double $primary;
    }
    
    BackendsScreen .backend-name {
        text-style: bold;
    }
    
    BackendsScreen .backend-status {
        margin-top: 1;
    }
    
    BackendsScreen .actions-section {
        height: auto;
        layout: horizontal;
        padding: 1;
        border-top: solid $primary-darken-3;
    }
    
    BackendsScreen .action-btn {
        margin-right: 1;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the backends screen."""
        super().__init__(**kwargs)
        self._selected_backend = None
    
    def compose_main(self):
        """Compose the backends screen content."""
        with Vertical(classes="main-content"):
            # Title
            yield Static(
                "Available Backends",
                classes="section-title",
            )
            
            # Backend cards grid
            with Grid(classes="backends-grid"):
                # Try to get real backends from registry
                backends_data = []
                
                if REGISTRY_AVAILABLE:
                    try:
                        registry = BackendRegistry()
                        registry.discover()
                        backend_names = registry.list_backends()
                        
                        for name in backend_names:
                            info = registry.get_backend_info(name)
                            health = registry.check_backend_health(name)
                            
                            backends_data.append({
                                'name': name,
                                'type': info.get('type', 'Simulator') if info else 'Simulator',
                                'status': 'healthy' if health and health.get('status') == 'healthy' else 'unknown',
                                'description': info.get('description', f'{name} backend') if info else f'{name} backend',
                                'capabilities': info.get('capabilities', []) if info else [],
                            })
                    except Exception:
                        pass
                
                # Fallback to sample data if no real backends
                if not backends_data:
                    backends_data = [
                        {'name': 'LRET', 'type': 'Simulator', 'status': 'healthy', 'description': 'LRET Quantum Simulator'},
                        {'name': 'Cirq', 'type': 'Simulator', 'status': 'healthy', 'description': 'Google Cirq Simulator'},
                        {'name': 'Qiskit Aer', 'type': 'Simulator', 'status': 'healthy', 'description': 'IBM Qiskit Aer'},
                        {'name': 'QuEST', 'type': 'Simulator', 'status': 'unknown', 'description': 'QuEST High-Performance'},
                        {'name': 'qsim', 'type': 'Simulator', 'status': 'unknown', 'description': 'Google qsim'},
                        {'name': 'cuQuantum', 'type': 'GPU', 'status': 'unknown', 'description': 'NVIDIA cuQuantum'},
                    ]
                
                # Create backend cards from data
                for backend in backends_data:
                    yield BackendCard(
                        backend['name'].lower().replace(' ', '_'),
                        backend['name'],
                        backend['description'],
                        backend['status']
                    )
            
            # Actions
            with Horizontal(classes="actions-section"):
                yield Button("Run Health Check", id="btn-health", classes="action-btn", variant="primary")
                yield Button("Compare Performance", id="btn-compare", classes="action-btn")
                yield Button("View Metrics", id="btn-metrics", classes="action-btn")
                yield Button("Configure", id="btn-configure", classes="action-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-health":
            self._run_health_check()
        elif button_id == "btn-compare":
            self._compare_performance()
        elif button_id == "btn-metrics":
            self._view_metrics()
        elif button_id == "btn-configure":
            self._configure_backend()


    def _run_health_check(self) -> None:
        """Run health check on selected backend."""
        backend = self._get_selected_backend()
        if not backend:
            self.notify("Please select a backend first", severity="warning")
            return
        
        backend_name = backend.get('name', 'Unknown')
        self.notify(f"Running health check on {backend_name}...")
        
        if REGISTRY_AVAILABLE:
            try:
                registry = BackendRegistry()
                health_result = registry.check_backend_health(backend_name)
                
                if health_result:
                    status = health_result.get('status', 'unknown')
                    response_time = health_result.get('response_time', 0)
                    
                    if status == 'healthy' or status == BackendStatus.HEALTHY:
                        self.notify(f"? {backend_name} is healthy ({response_time:.0f}ms)", severity="success")
                    elif status == 'degraded' or status == BackendStatus.DEGRADED:
                        self.notify(f"? {backend_name} is degraded ({response_time:.0f}ms)", severity="warning")
                    else:
                        self.notify(f"? {backend_name} is unavailable", severity="error")
                else:
                    self.notify(f"? Health check failed for {backend_name}", severity="error")
            except Exception as e:
                self.notify(f"? Health check error: {e}", severity="error")
        else:
            # Simulated health check when core not available
            import random
            response_time = random.randint(50, 200)
            self.notify(f"? {backend_name} appears healthy ({response_time}ms)", severity="success")
            self.notify("Note: Using simulated check (core not available)", severity="information")

    def _get_selected_backend(self):
        """Get the currently selected backend."""
        if hasattr(self, '_selected_backend') and self._selected_backend:
            return self._selected_backend
        
        # Try to get from UI selection
        try:
            table = self.query_one("#backends-table", DataTable)
            if table.cursor_row is not None:
                row = table.get_row_at(table.cursor_row)
                if row:
                    return {'name': str(row[0]), 'type': str(row[1]) if len(row) > 1 else 'Unknown'}
        except Exception:
            pass
        
        return None

    def _compare_performance(self) -> None:
        """Compare performance of backends using comparison dialog."""
        if DIALOGS_AVAILABLE:
            self.app.push_screen(BackendComparisonDialog())
        else:
            # Fallback to notifications
            self.notify("Performance Comparison", severity="information")
            
            if REGISTRY_AVAILABLE:
                try:
                    registry = BackendRegistry()
                    all_backends = registry.list_backends() if hasattr(registry, 'list_backends') else []
                    
                    comparison_data = []
                    for backend in all_backends[:5]:
                        health = registry.check_backend_health(backend)
                        if health:
                            comparison_data.append(f"{backend}: {health.get('response_time', 0):.0f}ms")
                    
                    if comparison_data:
                        for line in comparison_data:
                            self.notify(line)
                    else:
                        self.notify("No backends available for comparison", severity="warning")
                except Exception as e:
                    self.notify(f"Comparison failed: {e}", severity="error")
            else:
                self.notify("LRET: ~35ms | Cirq: ~55ms | Qiskit: ~70ms")
                self.notify("(Sample data - Install dialogs for full view)", severity="information")

    def _view_metrics(self) -> None:
        """View detailed metrics for selected backend using metrics dialog."""
        backend = self._get_selected_backend()
        backend_name = backend.get('name', None) if backend else None
        
        if DIALOGS_AVAILABLE:
            self.app.push_screen(BackendMetricsDialog(backend_name=backend_name))
        else:
            # Fallback to notifications
            if not backend:
                self.notify("Please select a backend first", severity="warning")
                return
            
            self.notify(f"Metrics for {backend_name}", severity="information")
            
            if REGISTRY_AVAILABLE:
                try:
                    registry = BackendRegistry()
                    metrics = registry.get_backend_metrics(backend_name) if hasattr(registry, 'get_backend_metrics') else None
                    
                    if metrics:
                        self.notify(f"Jobs Completed: {metrics.get('jobs_completed', 0)}")
                        self.notify(f"Success Rate: {metrics.get('success_rate', 0):.1%}")
                        self.notify(f"Avg Queue Time: {metrics.get('avg_queue_time', 0):.1f}s")
                        self.notify(f"Uptime: {metrics.get('uptime', 0):.1%}")
                    else:
                        self.notify("No detailed metrics available", severity="warning")
                except Exception as e:
                    self.notify(f"Error fetching metrics: {e}", severity="error")
            else:
                self.notify("Jobs: 150 | Success: 94.2% | Queue: 2.3s")
                self.notify("(Sample data - Install dialogs for full view)", severity="information")

    def _configure_backend(self) -> None:
        """Configure the selected backend using configuration dialog."""
        backend = self._get_selected_backend()
        backend_name = backend.get('name', None) if backend else None
        
        if DIALOGS_AVAILABLE:
            self.app.push_screen(BackendConfigDialog(backend_name=backend_name))
        else:
            # Fallback to notifications
            if not backend:
                self.notify("Please select a backend first", severity="warning")
                return
            
            backend_type = backend.get('type', 'Unknown')
            self.notify(f"Configuration for {backend_name}", severity="information")
            self.notify(f"Type: {backend_type}")
            
            # Show configuration options based on backend type
            if 'simulator' in backend_name.lower() or 'sim' in backend_type.lower():
                self.notify("Options: shots, seed, noise_model, coupling_map")
                self.notify("Use Settings screen to modify simulator options")
            elif 'ibm' in backend_name.lower():
                self.notify("Options: hub, group, project, optimization_level")
                self.notify("Configure IBM credentials in Settings > LLM Settings")
            elif 'braket' in backend_name.lower() or 'amazon' in backend_name.lower():
                self.notify("Options: region, s3_bucket, device_arn")
                self.notify("Configure AWS credentials in Settings")
            else:
                self.notify("Options: endpoint, timeout, max_retries")
                self.notify("Use Settings screen for advanced configuration")



class BackendCard(Static):
    """A card displaying backend information."""
    
    can_focus = True  # Make card focusable for selection
    
    def __init__(
        self,
        backend_id: str,
        name: str,
        description: str,
        status: str,
        **kwargs,
    ):
        """Initialize the backend card."""
        super().__init__(**kwargs)
        self.backend_id = backend_id
        self.backend_name = name
        self.description = description
        self.status = status
        self.classes = "backend-card"
        self._selected = False
    
    def on_click(self) -> None:
        """Handle click to select this backend."""
        # Find parent screen and update selection
        screen = self.screen
        if hasattr(screen, '_selected_backend'):
            # Deselect previous
            if screen._selected_backend:
                try:
                    for card in screen.query(BackendCard):
                        card.remove_class("-selected")
                        card._selected = False
                except Exception:
                    pass
        
        # Select this card
        self._selected = True
        self.add_class("-selected")
        screen._selected_backend = {
            'name': self.backend_name,
            'type': 'Simulator',
            'status': self.status,
        }
        screen.notify(f"Selected: {self.backend_name}")
    
    def render(self) -> Text:
        """Render the backend card."""
        theme = get_theme()
        text = Text()
        
        # Status icon and name
        icon = get_health_icon(self.status)
        color = theme.get_health_color(self.status)
        
        text.append(icon, style=f"bold {color}")
        text.append(" ")
        text.append(self.backend_name, style=f"bold {theme.fg_base}")
        text.append("\n")
        
        # Description
        text.append(self.description, style=theme.fg_muted)
        text.append("\n\n")
        
        # Status text
        status_text = self.status.capitalize()
        if self.status == "healthy":
            text.append(f"● {status_text}", style=f"bold {color}")
        elif self.status == "unavailable":
            text.append(f"○ Not Available", style=color)
        else:
            text.append(f"○ {status_text}", style=color)
        
        return text
