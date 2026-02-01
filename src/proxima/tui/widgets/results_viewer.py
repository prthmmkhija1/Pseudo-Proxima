"""Real-Time Results Viewer Widget for Proxima TUI.

Provides a rich widget for viewing execution results in real-time
with streaming updates, SQLite persistence, and sortable data tables.

Features:
- Real-time result streaming via event bus
- SQLite persistence for result history
- Sortable DataTable with filtering
- Result comparison tools
- Export functionality
- Chart integration (placeholder for visualization)
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive, var
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    ProgressBar,
    RichLog,
    Rule,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widget import Widget
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from proxima.core.event_bus import (
    Event,
    EventBus,
    EventType,
    get_event_bus,
)


@dataclass
class ExecutionResult:
    """Represents a single execution result."""
    result_id: str
    execution_id: str
    backend: str
    algorithm: str
    timestamp: float
    duration_ms: float
    status: str  # success, error, partial
    output_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @property
    def timestamp_str(self) -> str:
        """Get formatted timestamp."""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    @property
    def duration_str(self) -> str:
        """Get formatted duration."""
        if self.duration_ms < 1000:
            return f"{self.duration_ms:.0f}ms"
        return f"{self.duration_ms / 1000:.2f}s"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "execution_id": self.execution_id,
            "backend": self.backend,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "output_data": self.output_data,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """Create from dictionary."""
        return cls(
            result_id=data["result_id"],
            execution_id=data["execution_id"],
            backend=data["backend"],
            algorithm=data["algorithm"],
            timestamp=data["timestamp"],
            duration_ms=data["duration_ms"],
            status=data["status"],
            output_data=data.get("output_data", {}),
            metrics=data.get("metrics", {}),
            error_message=data.get("error_message"),
        )


class ResultsDatabase:
    """SQLite-based persistence for execution results."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the database.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            # Default to user's data directory
            data_dir = Path.home() / ".proxima" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "results.db"
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    result_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    duration_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    output_data TEXT,
                    metrics TEXT,
                    error_message TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_timestamp
                ON results (timestamp DESC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_backend
                ON results (backend)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_algorithm
                ON results (algorithm)
            """)
    
    def save_result(self, result: ExecutionResult) -> None:
        """Save a result to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO results
                (result_id, execution_id, backend, algorithm, timestamp,
                 duration_ms, status, output_data, metrics, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.execution_id,
                result.backend,
                result.algorithm,
                result.timestamp,
                result.duration_ms,
                result.status,
                json.dumps(result.output_data),
                json.dumps(result.metrics),
                result.error_message,
            ))
    
    def get_result(self, result_id: str) -> Optional[ExecutionResult]:
        """Get a result by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            
            if row:
                return self._row_to_result(row)
        return None
    
    def get_results(
        self,
        backend: Optional[str] = None,
        algorithm: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ExecutionResult]:
        """Query results with optional filters."""
        conditions = []
        params = []
        
        if backend:
            conditions.append("backend = ?")
            params.append(backend)
        if algorithm:
            conditions.append("algorithm = ?")
            params.append(algorithm)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"""
                SELECT * FROM results
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset]).fetchall()
            
            return [self._row_to_result(row) for row in rows]
    
    def get_recent_results(self, limit: int = 50) -> List[ExecutionResult]:
        """Get most recent results."""
        return self.get_results(limit=limit)
    
    def get_backends(self) -> List[str]:
        """Get list of all backends with results."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT backend FROM results ORDER BY backend"
            ).fetchall()
            return [row[0] for row in rows]
    
    def get_algorithms(self) -> List[str]:
        """Get list of all algorithms with results."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT algorithm FROM results ORDER BY algorithm"
            ).fetchall()
            return [row[0] for row in rows]
    
    def delete_result(self, result_id: str) -> bool:
        """Delete a result."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM results WHERE result_id = ?",
                (result_id,),
            )
            return cursor.rowcount > 0
    
    def clear_old_results(self, days: int = 30) -> int:
        """Delete results older than specified days."""
        cutoff = time.time() - (days * 24 * 60 * 60)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM results WHERE timestamp < ?",
                (cutoff,),
            )
            return cursor.rowcount
    
    def _row_to_result(self, row: sqlite3.Row) -> ExecutionResult:
        """Convert database row to ExecutionResult."""
        return ExecutionResult(
            result_id=row["result_id"],
            execution_id=row["execution_id"],
            backend=row["backend"],
            algorithm=row["algorithm"],
            timestamp=row["timestamp"],
            duration_ms=row["duration_ms"],
            status=row["status"],
            output_data=json.loads(row["output_data"] or "{}"),
            metrics=json.loads(row["metrics"] or "{}"),
            error_message=row["error_message"],
        )


class ResultSummaryCard(Static):
    """Card showing summary of a single result."""
    
    DEFAULT_CSS = """
    ResultSummaryCard {
        height: auto;
        min-height: 5;
        padding: 1;
        margin: 0 0 1 0;
        border: solid $primary;
        background: $surface;
    }
    
    ResultSummaryCard:hover {
        border: double $accent;
    }
    
    ResultSummaryCard.selected {
        border: double $success;
        background: $surface-lighten-1;
    }
    
    ResultSummaryCard.status-success {
        border-left: thick $success;
    }
    
    ResultSummaryCard.status-error {
        border-left: thick $error;
    }
    """
    
    class Selected(Message):
        """Emitted when this card is selected."""
        def __init__(self, result: ExecutionResult) -> None:
            self.result = result
            super().__init__()
    
    def __init__(self, result: ExecutionResult, **kwargs):
        super().__init__(**kwargs)
        self.result = result
        self._update_classes()
    
    def _update_classes(self) -> None:
        """Update CSS classes based on status."""
        self.remove_class("status-success")
        self.remove_class("status-error")
        if self.result.status == "success":
            self.add_class("status-success")
        elif self.result.status == "error":
            self.add_class("status-error")
    
    def render(self) -> Text:
        """Render the card content."""
        status_icon = {
            "success": "✅",
            "error": "❌",
            "partial": "⚠️",
        }.get(self.result.status, "❓")
        
        return Text.assemble(
            (f"{status_icon} ", ""),
            (f"{self.result.algorithm}", "bold"),
            (" on ", "dim"),
            (f"{self.result.backend}", "bold cyan"),
            ("\n", ""),
            ("⏱️ ", ""),
            (f"{self.result.duration_str}", ""),
            (" | ", "dim"),
            ("📅 ", ""),
            (f"{self.result.timestamp_str}", "dim"),
        )
    
    def on_click(self) -> None:
        """Handle click to select this card."""
        self.post_message(self.Selected(self.result))


class ResultDetailsPanel(Container):
    """Panel showing detailed view of a result."""
    
    DEFAULT_CSS = """
    ResultDetailsPanel {
        height: 1fr;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    ResultDetailsPanel #details-header {
        height: 3;
        border-bottom: solid $primary;
    }
    
    ResultDetailsPanel #details-content {
        height: 1fr;
        overflow-y: auto;
    }
    
    ResultDetailsPanel #metrics-table {
        height: auto;
        max-height: 50%;
    }
    
    ResultDetailsPanel #output-log {
        height: 1fr;
    }
    """
    
    result: reactive[Optional[ExecutionResult]] = reactive(None)
    
    def compose(self) -> ComposeResult:
        """Compose the details panel."""
        with Vertical(id="details-header"):
            yield Label("Select a result to view details", id="header-label")
        
        with TabbedContent(id="details-tabs"):
            with TabPane("Summary", id="tab-summary"):
                yield Static("No result selected", id="summary-content")
            
            with TabPane("Metrics", id="tab-metrics"):
                yield DataTable(id="metrics-table")
            
            with TabPane("Output", id="tab-output"):
                yield RichLog(id="output-log", highlight=True, markup=True)
    
    def watch_result(self, result: Optional[ExecutionResult]) -> None:
        """Update display when result changes."""
        if result is None:
            return
        
        self._update_header(result)
        self._update_summary(result)
        self._update_metrics(result)
        self._update_output(result)
    
    def _update_header(self, result: ExecutionResult) -> None:
        """Update header label."""
        try:
            label = self.query_one("#header-label", Label)
            status_icon = "✅" if result.status == "success" else "❌"
            label.update(
                f"{status_icon} {result.algorithm} on {result.backend} "
                f"({result.duration_str})"
            )
        except NoMatches:
            pass
    
    def _update_summary(self, result: ExecutionResult) -> None:
        """Update summary tab."""
        try:
            content = self.query_one("#summary-content", Static)
            
            summary = Text()
            summary.append("Result ID: ", style="bold")
            summary.append(f"{result.result_id}\n")
            summary.append("Execution ID: ", style="bold")
            summary.append(f"{result.execution_id}\n")
            summary.append("Backend: ", style="bold")
            summary.append(f"{result.backend}\n")
            summary.append("Algorithm: ", style="bold")
            summary.append(f"{result.algorithm}\n")
            summary.append("Status: ", style="bold")
            summary.append(
                f"{result.status}\n",
                style="green" if result.status == "success" else "red",
            )
            summary.append("Duration: ", style="bold")
            summary.append(f"{result.duration_str}\n")
            summary.append("Timestamp: ", style="bold")
            summary.append(f"{result.timestamp_str}\n")
            
            if result.error_message:
                summary.append("\nError: ", style="bold red")
                summary.append(f"{result.error_message}\n", style="red")
            
            content.update(summary)
        except NoMatches:
            pass
    
    def _update_metrics(self, result: ExecutionResult) -> None:
        """Update metrics table."""
        try:
            table = self.query_one("#metrics-table", DataTable)
            table.clear(columns=True)
            table.add_columns("Metric", "Value")
            
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    value_str = f"{value:.6f}"
                else:
                    value_str = str(value)
                table.add_row(key, value_str)
        except NoMatches:
            pass
    
    def _update_output(self, result: ExecutionResult) -> None:
        """Update output log."""
        try:
            log = self.query_one("#output-log", RichLog)
            log.clear()
            
            output = result.output_data
            if isinstance(output, dict):
                # Pretty print JSON
                formatted = json.dumps(output, indent=2)
                log.write(Syntax(formatted, "json", theme="monokai"))
            elif isinstance(output, str):
                log.write(output)
            else:
                log.write(str(output))
        except NoMatches:
            pass


class ResultsTable(DataTable):
    """Sortable data table for results list."""
    
    DEFAULT_CSS = """
    ResultsTable {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("enter", "select_result", "Select"),
        Binding("d", "delete_result", "Delete"),
        Binding("e", "export_result", "Export"),
    ]
    
    class ResultSelected(Message):
        """Emitted when a result row is selected."""
        def __init__(self, result: ExecutionResult) -> None:
            self.result = result
            super().__init__()
    
    def __init__(self, **kwargs):
        super().__init__(cursor_type="row", **kwargs)
        self._results: List[ExecutionResult] = []
        self._sort_column = "timestamp"
        self._sort_reverse = True
    
    def on_mount(self) -> None:
        """Set up columns on mount."""
        self.add_columns(
            "Status",
            "Algorithm",
            "Backend",
            "Duration",
            "Timestamp",
        )
    
    def load_results(self, results: List[ExecutionResult]) -> None:
        """Load results into the table."""
        self._results = results
        self._refresh_table()
    
    def add_result(self, result: ExecutionResult) -> None:
        """Add a single result to the table."""
        self._results.insert(0, result)
        self._refresh_table()
    
    def _refresh_table(self) -> None:
        """Refresh the table display."""
        self.clear()
        
        # Sort results
        sorted_results = sorted(
            self._results,
            key=lambda r: getattr(r, self._sort_column, 0),
            reverse=self._sort_reverse,
        )
        
        for result in sorted_results:
            status_icon = {
                "success": "✅",
                "error": "❌",
                "partial": "⚠️",
            }.get(result.status, "❓")
            
            self.add_row(
                status_icon,
                result.algorithm,
                result.backend,
                result.duration_str,
                result.timestamp_str,
                key=result.result_id,
            )
    
    def action_select_result(self) -> None:
        """Select the current result."""
        row_key = self.cursor_row
        if 0 <= row_key < len(self._results):
            result = self._results[row_key]
            self.post_message(self.ResultSelected(result))
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        for result in self._results:
            if result.result_id == str(event.row_key.value):
                self.post_message(self.ResultSelected(result))
                break
    
    def action_delete_result(self) -> None:
        """Delete the current result (placeholder)."""
        self.notify("Delete: Not implemented")
    
    def action_export_result(self) -> None:
        """Export the current result (placeholder)."""
        self.notify("Export: Not implemented")


class RealTimeResultsViewer(Container):
    """Main results viewer with real-time updates.
    
    Combines results table, details panel, and event bus integration
    for live result streaming.
    """
    
    DEFAULT_CSS = """
    RealTimeResultsViewer {
        height: 1fr;
        width: 100%;
    }
    
    RealTimeResultsViewer #toolbar {
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
    }
    
    RealTimeResultsViewer #main-content {
        height: 1fr;
    }
    
    RealTimeResultsViewer #results-list {
        width: 40%;
        border-right: solid $primary;
    }
    
    RealTimeResultsViewer #results-details {
        width: 60%;
    }
    
    RealTimeResultsViewer #filter-row {
        height: 3;
        padding: 0 1;
    }
    
    RealTimeResultsViewer #filter-row Select {
        width: 20;
        margin-right: 1;
    }
    """
    
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("f", "toggle_filters", "Filters"),
        Binding("c", "clear_filters", "Clear Filters"),
    ]
    
    class NewResult(Message):
        """Emitted when a new result is received."""
        def __init__(self, result: ExecutionResult) -> None:
            self.result = result
            super().__init__()
    
    show_filters: reactive[bool] = reactive(True)
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        auto_subscribe: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._database = ResultsDatabase(db_path)
        self._auto_subscribe = auto_subscribe
        self._event_bus = get_event_bus()
        self._subscription_id: Optional[str] = None
        self._selected_result: Optional[ExecutionResult] = None
    
    def compose(self) -> ComposeResult:
        """Compose the results viewer."""
        with Horizontal(id="toolbar"):
            yield Button("🔄 Refresh", id="btn-refresh", variant="primary")
            yield Button("🗑 Clear Old", id="btn-clear-old", variant="warning")
            yield Button("📤 Export", id="btn-export", variant="default")
            yield Label("", id="result-count")
        
        if self.show_filters:
            with Horizontal(id="filter-row"):
                yield Select(
                    [("All Backends", "")],
                    id="filter-backend",
                    prompt="Backend",
                )
                yield Select(
                    [("All Algorithms", "")],
                    id="filter-algorithm",
                    prompt="Algorithm",
                )
                yield Select(
                    [
                        ("All Status", ""),
                        ("Success", "success"),
                        ("Error", "error"),
                        ("Partial", "partial"),
                    ],
                    id="filter-status",
                    prompt="Status",
                )
        
        with Horizontal(id="main-content"):
            with Vertical(id="results-list"):
                yield ResultsTable(id="results-table")
            
            with Vertical(id="results-details"):
                yield ResultDetailsPanel(id="details-panel")
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        self._load_initial_results()
        self._populate_filters()
        
        if self._auto_subscribe:
            self._subscribe_to_events()
    
    def on_unmount(self) -> None:
        """Clean up on unmount."""
        if self._subscription_id:
            self._event_bus.unsubscribe(self._subscription_id)
    
    def _load_initial_results(self) -> None:
        """Load initial results from database."""
        results = self._database.get_recent_results(limit=100)
        table = self.query_one("#results-table", ResultsTable)
        table.load_results(results)
        self._update_count(len(results))
    
    def _populate_filters(self) -> None:
        """Populate filter dropdowns."""
        try:
            backend_select = self.query_one("#filter-backend", Select)
            backends = self._database.get_backends()
            backend_select.set_options(
                [("All Backends", "")] + [(b, b) for b in backends]
            )
            
            algo_select = self.query_one("#filter-algorithm", Select)
            algorithms = self._database.get_algorithms()
            algo_select.set_options(
                [("All Algorithms", "")] + [(a, a) for a in algorithms]
            )
        except NoMatches:
            pass
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to result events."""
        self._subscription_id = self._event_bus.subscribe(
            self._handle_result_event,
            event_types={
                EventType.RESULT_AVAILABLE,
                EventType.RESULT_PARTIAL,
                EventType.RESULT_UPDATED,
            },
        )
    
    async def _handle_result_event(self, event: Event) -> None:
        """Handle incoming result events."""
        payload = event.payload
        
        result = ExecutionResult(
            result_id=payload.get("result_id", str(uuid.uuid4())[:8]),
            execution_id=event.source_id,
            backend=payload.get("backend", "unknown"),
            algorithm=payload.get("algorithm", "unknown"),
            timestamp=event.timestamp,
            duration_ms=payload.get("duration_ms", 0),
            status="partial" if event.event_type == EventType.RESULT_PARTIAL else "success",
            output_data=payload.get("output_data", {}),
            metrics=payload.get("metrics", {}),
            error_message=payload.get("error_message"),
        )
        
        # Save to database
        self._database.save_result(result)
        
        # Add to table
        table = self.query_one("#results-table", ResultsTable)
        table.add_result(result)
        
        # Update count
        self._update_count(len(table._results))
        
        # Notify
        self.post_message(self.NewResult(result))
    
    def _update_count(self, count: int) -> None:
        """Update the result count label."""
        try:
            label = self.query_one("#result-count", Label)
            label.update(f"📊 {count} results")
        except NoMatches:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-refresh":
            self.action_refresh()
        elif button_id == "btn-clear-old":
            self._clear_old_results()
        elif button_id == "btn-export":
            self._export_results()
    
    def on_results_table_result_selected(
        self,
        event: ResultsTable.ResultSelected,
    ) -> None:
        """Handle result selection from table."""
        self._selected_result = event.result
        details = self.query_one("#details-panel", ResultDetailsPanel)
        details.result = event.result
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle filter changes."""
        self._apply_filters()
    
    def _apply_filters(self) -> None:
        """Apply current filters and refresh table."""
        try:
            backend = self.query_one("#filter-backend", Select).value
            algorithm = self.query_one("#filter-algorithm", Select).value
            status = self.query_one("#filter-status", Select).value
            
            results = self._database.get_results(
                backend=backend if backend else None,
                algorithm=algorithm if algorithm else None,
                status=status if status else None,
                limit=100,
            )
            
            table = self.query_one("#results-table", ResultsTable)
            table.load_results(results)
            self._update_count(len(results))
        except NoMatches:
            pass
    
    def action_refresh(self) -> None:
        """Refresh results."""
        self._apply_filters()
        self._populate_filters()
        self.notify("Results refreshed")
    
    def action_toggle_filters(self) -> None:
        """Toggle filter visibility."""
        self.show_filters = not self.show_filters
    
    def action_clear_filters(self) -> None:
        """Clear all filters."""
        try:
            self.query_one("#filter-backend", Select).value = ""
            self.query_one("#filter-algorithm", Select).value = ""
            self.query_one("#filter-status", Select).value = ""
            self._apply_filters()
        except NoMatches:
            pass
    
    def _clear_old_results(self) -> None:
        """Clear results older than 30 days."""
        count = self._database.clear_old_results(days=30)
        self.notify(f"Cleared {count} old results")
        self.action_refresh()
    
    def _export_results(self) -> None:
        """Export current results (placeholder)."""
        self.notify("Export: Not implemented yet")
    
    def add_result(self, result: ExecutionResult) -> None:
        """Manually add a result."""
        self._database.save_result(result)
        table = self.query_one("#results-table", ResultsTable)
        table.add_result(result)
        self._update_count(len(table._results))
    
    @property
    def selected_result(self) -> Optional[ExecutionResult]:
        """Get the currently selected result."""
        return self._selected_result
