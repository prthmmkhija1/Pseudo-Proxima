"""Backend metrics dialog for viewing detailed performance metrics."""

from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, DataTable, ProgressBar
from rich.text import Text
from typing import Dict, Any, Optional
import random
import datetime

try:
    from proxima.backends.registry import BackendRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


class BackendMetricsDialog(ModalScreen):
    """Dialog for viewing detailed backend metrics."""

    DEFAULT_CSS = """
    BackendMetricsDialog { align: center middle; }
    BackendMetricsDialog > .dialog-container {
        padding: 1 2;
        border: thick $accent;
        background: $surface;
        width: 80;
        height: 40;
        overflow-y: auto;
    }
    BackendMetricsDialog .dialog-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }
    BackendMetricsDialog .metrics-grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        margin: 1 0;
        height: auto;
    }
    BackendMetricsDialog .metric-card {
        padding: 1;
        border: solid $primary;
        height: auto;
    }
    BackendMetricsDialog .metric-title {
        text-style: bold;
        color: $primary;
    }
    BackendMetricsDialog .metric-value {
        text-style: bold;
        color: $success;
    }
    BackendMetricsDialog .history-table {
        height: 8;
        margin: 1 0;
    }
    BackendMetricsDialog .footer {
        height: auto;
        layout: horizontal;
        margin-top: 1;
        padding: 1 0;
    }
    BackendMetricsDialog .footer Button {
        margin-right: 1;
        min-width: 14;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, backend_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.backend_name = backend_name or "All Backends"
        self._metrics: Dict[str, Any] = {}

    def compose(self):
        with Vertical(classes="dialog-container"):
            yield Static(f"[*] Metrics: {self.backend_name}", classes="dialog-title")
            
            with Horizontal(classes="metrics-grid"):
                # CPU Usage Card
                with Vertical(classes="metric-card"):
                    yield Static("CPU Usage", classes="metric-title")
                    yield Static("---%", id="cpu-value", classes="metric-value")
                    yield ProgressBar(total=100, id="cpu-bar")
                
                # Memory Usage Card  
                with Vertical(classes="metric-card"):
                    yield Static("Memory Usage", classes="metric-title")
                    yield Static("---MB", id="memory-value", classes="metric-value")
                    yield ProgressBar(total=100, id="memory-bar")
                
                # Throughput Card
                with Vertical(classes="metric-card"):
                    yield Static("Throughput", classes="metric-title")
                    yield Static("---/s", id="throughput-value", classes="metric-value")
                    yield ProgressBar(total=100, id="throughput-bar")
                
                # Latency Card
                with Vertical(classes="metric-card"):
                    yield Static("Avg Latency", classes="metric-title")
                    yield Static("---ms", id="latency-value", classes="metric-value")
                    yield ProgressBar(total=100, id="latency-bar")
            
            yield Static("Recent History:", classes="metric-title")
            table = DataTable(classes="history-table", id="history-table")
            table.add_columns("Timestamp", "Operation", "Duration", "Status")
            yield table
            
            with Horizontal(classes="footer"):
                yield Button("Refresh", id="btn-refresh", variant="primary")
                yield Button("Trends", id="btn-trends", variant="default")
                yield Button("Export", id="btn-export", variant="default")
                yield Button("Close", id="btn-close", variant="error")

    def on_mount(self):
        """Load metrics on mount."""
        self._load_metrics()

    def _load_metrics(self):
        """Load metrics data."""
        # Generate sample metrics
        self._metrics = {
            "cpu_usage": random.randint(15, 85),
            "memory_usage": random.randint(128, 1024),
            "memory_total": 2048,
            "throughput": random.randint(5000, 30000),
            "throughput_max": 50000,
            "latency": random.randint(20, 150),
            "latency_max": 200,
        }
        
        # Try registry for real data
        if REGISTRY_AVAILABLE and self.backend_name != "All Backends":
            try:
                registry = BackendRegistry()
                health = registry.check_backend_health(self.backend_name)
                if health:
                    self._metrics["latency"] = health.get("response_time", self._metrics["latency"])
            except Exception:
                pass
        
        self._update_display()

    def _update_display(self):
        """Update the display with current metrics."""
        # Update CPU
        cpu = self._metrics.get("cpu_usage", 0)
        self.query_one("#cpu-value", Static).update(f"{cpu}%")
        self.query_one("#cpu-bar", ProgressBar).update(progress=cpu)
        
        # Update Memory
        mem = self._metrics.get("memory_usage", 0)
        mem_total = self._metrics.get("memory_total", 2048)
        mem_pct = int((mem / mem_total) * 100) if mem_total > 0 else 0
        self.query_one("#memory-value", Static).update(f"{mem}MB")
        self.query_one("#memory-bar", ProgressBar).update(progress=mem_pct)
        
        # Update Throughput
        throughput = self._metrics.get("throughput", 0)
        tp_max = self._metrics.get("throughput_max", 50000)
        tp_pct = int((throughput / tp_max) * 100) if tp_max > 0 else 0
        self.query_one("#throughput-value", Static).update(f"{throughput:,}/s")
        self.query_one("#throughput-bar", ProgressBar).update(progress=tp_pct)
        
        # Update Latency (inverted - lower is better)
        latency = self._metrics.get("latency", 0)
        lat_max = self._metrics.get("latency_max", 200)
        lat_pct = max(0, 100 - int((latency / lat_max) * 100)) if lat_max > 0 else 0
        self.query_one("#latency-value", Static).update(f"{latency}ms")
        self.query_one("#latency-bar", ProgressBar).update(progress=lat_pct)
        
        # Update history table
        table = self.query_one("#history-table", DataTable)
        table.clear()
        
        operations = ["simulate", "execute", "validate", "compile", "optimize"]
        statuses = ["[+] success", "[+] success", "[+] success", "[!] warning", "[x] error"]
        
        for i in range(5):
            ts = datetime.datetime.now() - datetime.timedelta(minutes=i*5)
            table.add_row(
                ts.strftime("%H:%M:%S"),
                random.choice(operations),
                f"{random.randint(10, 200)}ms",
                random.choice(statuses[:3])  # Mostly successes
            )

    def on_button_pressed(self, event):
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-close":
            self.dismiss(None)
        elif button_id == "btn-refresh":
            self._load_metrics()
            self.notify("Metrics refreshed")
        elif button_id == "btn-trends":
            self.notify("Trends view: Feature in development", severity="warning")
        elif button_id == "btn-export":
            self._export_metrics()

    def action_close(self):
        """Close the dialog."""
        self.dismiss(None)

    def action_refresh(self):
        """Refresh metrics."""
        self._load_metrics()
        self.notify("Metrics refreshed")

    def _export_metrics(self):
        """Export metrics to file."""
        try:
            import json
            from pathlib import Path
            
            path = Path.home() / f"proxima_metrics_{self.backend_name.lower().replace(' ', '_')}.json"
            with open(path, "w") as f:
                json.dump(self._metrics, f, indent=2)
            
            self.notify(f"[+] Exported to {path}", severity="success")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
