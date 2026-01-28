"""Phase 8: Deployment Complete Screen.

Final screen showing successful backend deployment with:
- Success confirmation
- Deployment summary
- Backend details
- Next steps
- Quick actions (test, docs, export)

Part of Phase 8: Final Deployment & Success Confirmation.
"""

from __future__ import annotations

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.widgets import Static, Button, Label
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from .wizard_state import BackendWizardState


class DeploymentSummaryWidget(Static):
    """Widget showing deployment summary statistics."""
    
    DEFAULT_CSS = """
    DeploymentSummaryWidget {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 1 0;
        background: $success 10%;
        border: solid $success;
    }
    
    DeploymentSummaryWidget .summary-title {
        text-style: bold;
        color: $success;
        margin-bottom: 1;
    }
    
    DeploymentSummaryWidget .summary-item {
        padding: 0 2;
    }
    
    DeploymentSummaryWidget .item-passed {
        color: $success;
    }
    
    DeploymentSummaryWidget .item-warning {
        color: $warning;
    }
    """
    
    def __init__(
        self,
        files_created: int,
        tests_passed: int,
        tests_total: int,
        registry_updated: bool,
        documentation_generated: bool,
        **kwargs
    ):
        """Initialize summary widget.
        
        Args:
            files_created: Number of files created
            tests_passed: Number of tests passed
            tests_total: Total number of tests
            registry_updated: Whether registry was updated
            documentation_generated: Whether docs were generated
        """
        super().__init__(**kwargs)
        self.files_created = files_created
        self.tests_passed = tests_passed
        self.tests_total = tests_total
        self.registry_updated = registry_updated
        self.documentation_generated = documentation_generated
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("Deployment Summary:", classes="summary-title")
        
        # Files created
        yield Static(f"  ✓ Files Created: {self.files_created}", classes="summary-item item-passed")
        
        # Tests passed
        pass_rate = (self.tests_passed / self.tests_total * 100) if self.tests_total > 0 else 0
        test_class = "item-passed" if pass_rate >= 80 else "item-warning"
        yield Static(
            f"  ✓ Tests Passed: {self.tests_passed}/{self.tests_total} ({pass_rate:.0f}%)",
            classes=f"summary-item {test_class}"
        )
        
        # Registry
        reg_status = "Yes" if self.registry_updated else "No"
        yield Static(f"  ✓ Registry Updated: {reg_status}", classes="summary-item item-passed")
        
        # Documentation
        doc_status = "Yes" if self.documentation_generated else "No"
        yield Static(
            f"  ✓ Documentation Generated: {doc_status}",
            classes="summary-item item-passed"
        )


class BackendDetailsWidget(Static):
    """Widget showing backend details."""
    
    DEFAULT_CSS = """
    BackendDetailsWidget {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 1 0;
        background: $primary 10%;
        border: solid $primary-darken-2;
    }
    
    BackendDetailsWidget .details-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    BackendDetailsWidget .details-item {
        padding: 0 2;
        color: $text;
    }
    
    BackendDetailsWidget .details-label {
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        backend_name: str,
        backend_id: str,
        version: str,
        backend_type: str,
        location: str,
        **kwargs
    ):
        """Initialize details widget.
        
        Args:
            backend_name: Display name of the backend
            backend_id: Internal ID
            version: Version string
            backend_type: Type of backend
            location: Installation location
        """
        super().__init__(**kwargs)
        self.backend_name = backend_name
        self.backend_id = backend_id
        self.version = version
        self.backend_type = backend_type
        self.location = location
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("Backend Details:", classes="details-title")
        
        yield Static(f"  Name: {self.backend_name}", classes="details-item")
        yield Static(f"  ID: {self.backend_id}", classes="details-item")
        yield Static(f"  Version: {self.version}", classes="details-item")
        yield Static(f"  Type: {self.backend_type}", classes="details-item")
        yield Static(f"  Location: {self.location}", classes="details-item")


class NextStepsWidget(Static):
    """Widget showing next steps after deployment."""
    
    DEFAULT_CSS = """
    NextStepsWidget {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 1 0;
        background: $boost;
        border: solid $accent;
    }
    
    NextStepsWidget .steps-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    NextStepsWidget .step-item {
        padding: 0 2;
        color: $text;
    }
    
    NextStepsWidget .command-hint {
        background: $surface-darken-2;
        padding: 0 1;
        margin: 0 2;
        font-family: monospace;
        color: $text;
    }
    """
    
    def __init__(self, backend_id: str, **kwargs):
        """Initialize next steps widget.
        
        Args:
            backend_id: Backend identifier
        """
        super().__init__(**kwargs)
        self.backend_id = backend_id
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("Next Steps:", classes="steps-title")
        
        yield Static(
            "  1. Backend is now available in backend selection menu",
            classes="step-item"
        )
        
        yield Static(
            "  2. Verify registration with:",
            classes="step-item"
        )
        yield Static(
            f"     proxima backends list",
            classes="command-hint"
        )
        
        yield Static(
            "  3. Test your backend with:",
            classes="step-item"
        )
        yield Static(
            f"     proxima run --backend {self.backend_id}",
            classes="command-hint"
        )
        
        yield Static(
            "  4. View documentation for advanced configuration",
            classes="step-item"
        )


class QuickActionsWidget(Static):
    """Widget with quick action buttons."""
    
    DEFAULT_CSS = """
    QuickActionsWidget {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 1 0;
    }
    
    QuickActionsWidget .actions-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    QuickActionsWidget .actions-row {
        width: 100%;
        height: auto;
        align: center middle;
    }
    
    QuickActionsWidget Button {
        margin: 0 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("Quick Actions:", classes="actions-title")
        
        with Horizontal(classes="actions-row"):
            yield Button("🧪 Test Backend", id="action_test", variant="default")
            yield Button("📖 View Documentation", id="action_docs", variant="default")
            yield Button("📥 Export Config", id="action_export", variant="default")


class DeploymentCompleteScreen(ModalScreen):
    """Final screen showing successful backend deployment.
    
    Phase 8: Final Deployment & Success Confirmation
    
    Features:
    - Success celebration animation
    - Deployment summary with statistics
    - Backend details
    - Next steps guide
    - Quick actions (test, docs, export)
    - Options to close or create another backend
    """
    
    DEFAULT_CSS = """
    DeploymentCompleteScreen {
        align: center middle;
    }
    
    DeploymentCompleteScreen #main_container {
        width: 80;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: thick $success;
        padding: 1 2;
    }
    
    DeploymentCompleteScreen .success-header {
        width: 100%;
        height: auto;
        padding: 1;
        background: $success 20%;
        text-align: center;
    }
    
    DeploymentCompleteScreen .success-icon {
        text-align: center;
        color: $success;
        text-style: bold;
        padding: 1;
    }
    
    DeploymentCompleteScreen .success-title {
        text-align: center;
        text-style: bold;
        color: $success;
        padding: 0 0 1 0;
    }
    
    DeploymentCompleteScreen .success-message {
        text-align: center;
        color: $text;
        padding: 1;
    }
    
    DeploymentCompleteScreen .content-scroll {
        height: auto;
        max-height: 50vh;
    }
    
    DeploymentCompleteScreen .divider {
        width: 100%;
        height: 1;
        background: $primary-darken-3;
        margin: 1 0;
    }
    
    DeploymentCompleteScreen .nav-buttons {
        width: 100%;
        height: auto;
        padding: 1;
        align: center middle;
        border-top: solid $primary-darken-3;
        margin-top: 1;
    }
    
    DeploymentCompleteScreen .nav-buttons Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("t", "test_backend", "Test Backend"),
        ("d", "view_docs", "View Documentation"),
        ("e", "export_config", "Export Config"),
        ("n", "create_another", "Create Another"),
    ]
    
    def __init__(
        self,
        wizard_state: BackendWizardState,
        deployment_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize deployment complete screen.
        
        Args:
            wizard_state: Completed wizard state
            deployment_results: Results from deployment process
        """
        super().__init__(**kwargs)
        self.wizard_state = wizard_state
        self.deployment_results = deployment_results or {}
        
        # Extract stats
        self.files_created = len(wizard_state.generated_files)
        
        test_results = wizard_state.test_results or {}
        summary = test_results.get("summary", {})
        self.tests_passed = summary.get("passed", 0)
        self.tests_total = summary.get("total_tests", 0)
        
        self.registry_updated = self.deployment_results.get("registry_updated", True)
        self.documentation_generated = self.deployment_results.get("documentation_generated", True)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with Container(id="main_container"):
            # Success header
            with Vertical(classes="success-header"):
                yield Static("🎉", classes="success-icon")
                yield Static("Backend Deployment Complete!", classes="success-title")
                yield Static("✓ SUCCESS", classes="success-icon")
            
            # Success message
            yield Static(
                f'Your backend "{self.wizard_state.backend_name}" has been successfully\n'
                "deployed and integrated into Proxima!",
                classes="success-message"
            )
            
            yield Static("", classes="divider")
            
            # Scrollable content
            with ScrollableContainer(classes="content-scroll"):
                # Deployment summary
                yield DeploymentSummaryWidget(
                    files_created=self.files_created,
                    tests_passed=self.tests_passed,
                    tests_total=self.tests_total,
                    registry_updated=self.registry_updated,
                    documentation_generated=self.documentation_generated,
                )
                
                # Backend details
                yield BackendDetailsWidget(
                    backend_name=self.wizard_state.backend_name,
                    backend_id=self.wizard_state.backend_id,
                    version=self.wizard_state.version,
                    backend_type=self.wizard_state.backend_type or "custom",
                    location=f"src/proxima/backends/{self.wizard_state.backend_id}/",
                )
                
                # Next steps
                yield NextStepsWidget(backend_id=self.wizard_state.backend_id)
                
                # Quick actions
                yield QuickActionsWidget()
            
            # Navigation buttons
            with Horizontal(classes="nav-buttons"):
                yield Button("Close", id="close", variant="primary")
                yield Button("Create Another Backend", id="create_another", variant="default")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "close":
            self.dismiss(True)
        
        elif button_id == "create_another":
            self._create_another_backend()
        
        elif button_id == "action_test":
            self._test_backend()
        
        elif button_id == "action_docs":
            self._view_documentation()
        
        elif button_id == "action_export":
            self._export_config()
    
    def _test_backend(self) -> None:
        """Run quick test on the backend."""
        from .quick_test_screen import QuickTestScreen
        
        self.app.push_screen(QuickTestScreen(
            backend_id=self.wizard_state.backend_id,
            backend_name=self.wizard_state.backend_name,
            wizard_state=self.wizard_state,
        ))
    
    def _view_documentation(self) -> None:
        """View generated documentation."""
        from .documentation_viewer import DocumentationViewer
        
        # Get documentation content
        docs_content = self._get_documentation_content()
        
        self.app.push_screen(DocumentationViewer(
            content=docs_content,
            backend_name=self.wizard_state.backend_name,
        ))
    
    def _get_documentation_content(self) -> str:
        """Get documentation content."""
        # Check if README was generated
        readme_key = f"backends/{self.wizard_state.backend_id}/README.md"
        
        if readme_key in self.wizard_state.generated_files:
            return self.wizard_state.generated_files[readme_key]
        
        # Generate default documentation
        return self._generate_default_docs()
    
    def _generate_default_docs(self) -> str:
        """Generate default documentation."""
        return f"""# {self.wizard_state.backend_name}

## Overview

{self.wizard_state.description or "A custom quantum simulator backend for Proxima."}

## Installation

This backend is automatically registered when placed in the `src/proxima/backends/` directory.

## Usage

```python
from proxima.backends import get_backend

# Get the backend
backend = get_backend("{self.wizard_state.backend_id}")

# Create a circuit
from proxima.core import QuantumCircuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Run the circuit
result = backend.run(circuit, shots=1000)
print(result.measurements)
```

## Configuration

- **Backend ID**: `{self.wizard_state.backend_id}`
- **Version**: `{self.wizard_state.version}`
- **Max Qubits**: `{self.wizard_state.capabilities.get("max_qubits", "N/A")}`

## Supported Gates

{self._format_gates_list()}

## API Reference

### `{self.wizard_state.backend_id.title()}Backend`

Main backend class.

#### Methods

- `run(circuit, shots=1024)`: Execute a quantum circuit
- `get_capabilities()`: Get backend capabilities
- `validate_circuit(circuit)`: Validate a circuit before execution

## License

This backend is part of the Proxima project.
"""
    
    def _format_gates_list(self) -> str:
        """Format gates list for documentation."""
        gates = self.wizard_state.gate_mappings.keys()
        if not gates:
            return "- No gates configured"
        
        return "\n".join(f"- `{gate}`" for gate in sorted(gates))
    
    def _export_config(self) -> None:
        """Export backend configuration."""
        config = self._build_export_config()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.wizard_state.backend_id}_config_{timestamp}.json"
        
        try:
            output_path = Path.home() / filename
            output_path.write_text(json.dumps(config, indent=2))
            
            self.notify(f"Configuration exported to: {output_path}")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def _build_export_config(self) -> Dict[str, Any]:
        """Build configuration for export."""
        return {
            "backend_name": self.wizard_state.backend_name,
            "backend_id": self.wizard_state.backend_id,
            "version": self.wizard_state.version,
            "backend_type": self.wizard_state.backend_type,
            "description": self.wizard_state.description,
            "capabilities": self.wizard_state.capabilities,
            "gate_mappings": self.wizard_state.gate_mappings,
            "created_at": datetime.now().isoformat(),
            "deployment_summary": {
                "files_created": self.files_created,
                "tests_passed": self.tests_passed,
                "tests_total": self.tests_total,
                "registry_updated": self.registry_updated,
                "documentation_generated": self.documentation_generated,
            },
            "files": list(self.wizard_state.generated_files.keys()),
        }
    
    def _create_another_backend(self) -> None:
        """Start wizard for another backend."""
        # Dismiss this screen
        self.dismiss(False)
        
        # The caller should handle restarting the wizard
        self.notify("Starting new backend wizard...")
    
    def action_close(self) -> None:
        """Handle close action."""
        self.dismiss(True)
    
    def action_test_backend(self) -> None:
        """Handle test backend action."""
        self._test_backend()
    
    def action_view_docs(self) -> None:
        """Handle view docs action."""
        self._view_documentation()
    
    def action_export_config(self) -> None:
        """Handle export config action."""
        self._export_config()
    
    def action_create_another(self) -> None:
        """Handle create another action."""
        self._create_another_backend()
