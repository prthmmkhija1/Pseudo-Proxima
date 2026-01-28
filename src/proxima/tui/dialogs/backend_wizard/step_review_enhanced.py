"""Enhanced Review & Deploy Step.

Final review and deployment step for the backend wizard.
Integrates with DeploymentManager for coordinated file writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import (
    Static, Button, Collapsible, TextArea, 
    ProgressBar, Label, DataTable
)
from textual.screen import ModalScreen
from textual.reactive import reactive

from .wizard_state import BackendWizardState
from .code_preview_dialog import CodePreviewDialog
from .deployment_success_dialog import DeploymentSuccessDialog, DeploymentFailureDialog


class DeploymentProgressWidget(Static):
    """Widget showing deployment progress."""
    
    DEFAULT_CSS = """
    DeploymentProgressWidget {
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary;
        margin: 1 0;
    }
    
    DeploymentProgressWidget .stage-label {
        color: $text;
        text-style: bold;
    }
    
    DeploymentProgressWidget .file-label {
        color: $text-muted;
        margin-left: 2;
    }
    
    DeploymentProgressWidget ProgressBar {
        margin: 1 0;
    }
    """
    
    stage = reactive("Initializing...")
    current_file = reactive("")
    progress = reactive(0.0)
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Label("", id="stage_label", classes="stage-label")
        yield Label("", id="file_label", classes="file-label")
        yield ProgressBar(total=100, show_eta=False, id="progress_bar")
    
    def watch_stage(self, stage: str) -> None:
        """Watch stage changes."""
        try:
            label = self.query_one("#stage_label", Label)
            label.update(f"📦 {stage}")
        except Exception:
            pass
    
    def watch_current_file(self, current_file: str) -> None:
        """Watch current file changes."""
        try:
            label = self.query_one("#file_label", Label)
            if current_file:
                label.update(f"  → {current_file}")
            else:
                label.update("")
        except Exception:
            pass
    
    def watch_progress(self, progress: float) -> None:
        """Watch progress changes."""
        try:
            bar = self.query_one("#progress_bar", ProgressBar)
            bar.update(progress=int(progress * 100))
        except Exception:
            pass


class FilesListWidget(Static):
    """Widget showing files to be created."""
    
    DEFAULT_CSS = """
    FilesListWidget {
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-3;
    }
    
    FilesListWidget .file-item {
        padding: 0 1;
    }
    
    FilesListWidget .file-icon {
        color: $success;
    }
    """
    
    def __init__(self, files: List[str], **kwargs):
        """Initialize files list widget.
        
        Args:
            files: List of file paths to display
        """
        super().__init__(**kwargs)
        self.files = files
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("Files to be created:", classes="section-title")
        
        for file_path in self.files:
            yield Static(f"  ✓ {file_path}", classes="file-item")
        
        if not self.files:
            yield Static("  (No files generated)", classes="file-item")


class BackendSummaryWidget(Static):
    """Widget showing backend summary."""
    
    DEFAULT_CSS = """
    BackendSummaryWidget {
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: solid $accent;
    }
    
    BackendSummaryWidget .summary-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    BackendSummaryWidget .summary-item {
        padding: 0 1;
    }
    
    BackendSummaryWidget .summary-label {
        color: $text-muted;
        width: 20;
    }
    
    BackendSummaryWidget .summary-value {
        color: $text;
    }
    """
    
    def __init__(self, state: BackendWizardState, **kwargs):
        """Initialize backend summary widget.
        
        Args:
            state: Wizard state with backend configuration
        """
        super().__init__(**kwargs)
        self.state = state
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("Backend Summary:", classes="summary-title")
        
        items = [
            ("Name", self.state.display_name or self.state.backend_name),
            ("Internal ID", self.state.backend_name),
            ("Version", self.state.version or "1.0.0"),
            ("Type", self.state.backend_type.value if self.state.backend_type else "N/A"),
            ("Library", self.state.library_name or "None"),
            ("Max Qubits", str(self.state.max_qubits)),
            ("Simulator Types", ", ".join(self.state.simulator_types) if self.state.simulator_types else "state_vector"),
        ]
        
        for label, value in items:
            yield Static(f"  {label}: {value}", classes="summary-item")
        
        # Features
        features = []
        if self.state.supports_noise:
            features.append("✓ Noise simulation")
        if self.state.supports_gpu:
            features.append("✓ GPU acceleration")
        if self.state.supports_batching:
            features.append("✓ Batch execution")
        
        if features:
            yield Static("")
            yield Static("  Features:", classes="summary-item")
            for feature in features:
                yield Static(f"    {feature}", classes="summary-item")


class EnhancedReviewStepScreen(ModalScreen[dict]):
    """Enhanced Step 7: Final review and deploy screen.
    
    Shows comprehensive summary of backend configuration,
    preview of generated code, and deployment with progress tracking.
    """
    
    DEFAULT_CSS = """
    EnhancedReviewStepScreen {
        align: center middle;
    }
    
    EnhancedReviewStepScreen .wizard-container {
        width: 95;
        height: auto;
        max-height: 95%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    EnhancedReviewStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    EnhancedReviewStepScreen .form-container {
        height: auto;
        max-height: 70%;
        padding: 1;
    }
    
    EnhancedReviewStepScreen .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }
    
    EnhancedReviewStepScreen .registry-info {
        padding: 1;
        margin: 1 0;
        background: $success-darken-3;
        border: solid $success-darken-2;
        color: $success;
    }
    
    EnhancedReviewStepScreen .progress-section {
        margin: 1 0;
        padding: 1 0;
        border-top: solid $primary-darken-3;
    }
    
    EnhancedReviewStepScreen .progress-text {
        color: $text-muted;
    }
    
    EnhancedReviewStepScreen .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    EnhancedReviewStepScreen .nav-button {
        margin: 0 1;
        min-width: 14;
    }
    
    EnhancedReviewStepScreen .deploy-button {
        min-width: 20;
    }
    
    EnhancedReviewStepScreen Collapsible {
        margin: 0 0 1 0;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    deploying = reactive(False)
    
    def __init__(self, state: BackendWizardState):
        """Initialize the review screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
        self._generated_files: List[str] = []
        self._update_generated_files()
    
    def _update_generated_files(self) -> None:
        """Update the list of files to be generated."""
        backend_name = self.state.backend_name or "my_backend"
        
        self._generated_files = [
            f"src/proxima/backends/contrib/{backend_name}/__init__.py",
            f"src/proxima/backends/contrib/{backend_name}/adapter.py",
            f"src/proxima/backends/contrib/{backend_name}/normalizer.py",
            f"src/proxima/backends/contrib/{backend_name}/README.md",
            f"tests/backends/test_{backend_name}.py",
        ]
    
    def compose(self) -> ComposeResult:
        """Compose the review screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "📋 Add Custom Backend - Review & Deploy",
                    classes="wizard-title"
                )
                
                yield Static(
                    "Review your backend configuration before deployment:",
                    classes="field-hint"
                )
                
                with ScrollableContainer(classes="form-container"):
                    # Backend Summary
                    yield BackendSummaryWidget(self.state)
                    
                    yield Static("")  # Spacer
                    
                    # Files to be created
                    yield FilesListWidget(self._generated_files)
                    
                    yield Static("")  # Spacer
                    
                    # Registry integration info
                    yield Static(
                        "Registry Integration:\n"
                        "  ✓ Backend will be auto-registered on next Proxima start\n"
                        "  ✓ Available in backend selection menus\n"
                        "  ✓ Accessible via: from proxima.backends.contrib import ...",
                        classes="registry-info"
                    )
                    
                    # View Code button
                    yield Button(
                        "👁 View Generated Code",
                        id="btn_view_code",
                        variant="default"
                    )
                    
                    # Deployment progress (hidden initially)
                    yield DeploymentProgressWidget(id="deployment_progress")
                
                # Progress indicator
                with Vertical(classes="progress-section"):
                    yield Static(
                        "Progress: Step 7 of 7",
                        classes="progress-text"
                    )
                    yield Static(
                        "[████████████████████] 100%",
                        classes="progress-text"
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
                        "Cancel",
                        id="btn_cancel",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "🚀 Deploy Backend",
                        id="btn_deploy",
                        variant="success",
                        classes="nav-button deploy-button"
                    )
    
    def on_mount(self) -> None:
        """Handle mount event."""
        # Hide progress widget initially
        try:
            progress = self.query_one("#deployment_progress", DeploymentProgressWidget)
            progress.display = False
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_back":
            self.state.current_step = 6
            self.dismiss({"action": "back", "state": self.state})
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_view_code":
            self._show_code_preview()
        
        elif event.button.id == "btn_deploy":
            if not self.deploying:
                asyncio.create_task(self._deploy_backend())
    
    def _show_code_preview(self) -> None:
        """Show code preview dialog."""
        # Get generated code from state
        generated_code = self._get_generated_code()
        
        if generated_code:
            self.app.push_screen(CodePreviewDialog(generated_code))
        else:
            self.notify("No code has been generated yet.", severity="warning")
    
    def _get_generated_code(self) -> Dict[str, str]:
        """Get generated code as dictionary."""
        backend_name = self.state.backend_name or "my_backend"
        display_name = self.state.display_name or backend_name.replace("_", " ").title()
        
        # If state has generated code, use it
        if hasattr(self.state, 'generated_files') and self.state.generated_files:
            return self.state.generated_files
        
        # Generate basic code structure
        return {
            "__init__.py": self._generate_init(backend_name, display_name),
            "adapter.py": self._generate_adapter(backend_name, display_name),
            "normalizer.py": self._generate_normalizer(backend_name),
            "README.md": self._generate_readme(backend_name, display_name),
        }
    
    def _generate_init(self, backend_name: str, display_name: str) -> str:
        """Generate __init__.py content."""
        class_name = self._to_class_name(backend_name)
        
        return f'''"""{display_name} Backend.

Auto-generated by Proxima Backend Addition Wizard.
"""

from .adapter import {class_name}Adapter

__all__ = ["{class_name}Adapter"]


def get_adapter(**kwargs):
    """Get the {display_name} adapter.
    
    Args:
        **kwargs: Adapter configuration
        
    Returns:
        {class_name}Adapter instance
    """
    return {class_name}Adapter(**kwargs)
'''
    
    def _generate_adapter(self, backend_name: str, display_name: str) -> str:
        """Generate adapter.py content."""
        class_name = self._to_class_name(backend_name)
        library = self.state.library_name or "# TODO: Import your library"
        max_qubits = self.state.max_qubits or 20
        
        return f'''"""{display_name} Adapter.

Adapts {display_name} to Proxima's unified interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

{library}


@dataclass
class {class_name}Config:
    """Configuration for {display_name}."""
    max_qubits: int = {max_qubits}
    shots: int = 1024
    seed: Optional[int] = None


class {class_name}Adapter:
    """Adapter for {display_name} quantum simulator."""
    
    name = "{backend_name}"
    display_name = "{display_name}"
    version = "{self.state.version or '1.0.0'}"
    
    def __init__(self, config: Optional[{class_name}Config] = None, **kwargs):
        """Initialize the adapter.
        
        Args:
            config: Adapter configuration
            **kwargs: Additional configuration
        """
        self.config = config or {class_name}Config(**kwargs)
        self._backend = None
    
    def initialize(self) -> None:
        """Initialize the backend."""
        # TODO: Initialize your backend here
        pass
    
    def run(self, circuit, shots: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Run a quantum circuit.
        
        Args:
            circuit: Quantum circuit to execute
            shots: Number of shots (uses config default if not specified)
            **kwargs: Additional execution options
            
        Returns:
            Execution results
        """
        shots = shots or self.config.shots
        
        # TODO: Implement circuit execution
        # 1. Convert circuit to backend format
        # 2. Execute on backend
        # 3. Return results
        
        raise NotImplementedError("TODO: Implement run method")
    
    def validate_circuit(self, circuit) -> bool:
        """Validate a circuit for this backend.
        
        Args:
            circuit: Circuit to validate
            
        Returns:
            True if circuit is valid
        """
        # TODO: Implement validation
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get backend capabilities.
        
        Returns:
            Dictionary of capabilities
        """
        return {{
            "max_qubits": self.config.max_qubits,
            "supports_noise": {str(self.state.supports_noise).lower()},
            "supports_gpu": {str(self.state.supports_gpu).lower()},
            "simulator_types": {self.state.simulator_types or ["state_vector"]},
        }}
    
    def shutdown(self) -> None:
        """Shutdown the backend."""
        self._backend = None
'''
    
    def _generate_normalizer(self, backend_name: str) -> str:
        """Generate normalizer.py content."""
        class_name = self._to_class_name(backend_name)
        
        return f'''"""Result Normalizer for {class_name}.

Normalizes backend-specific results to Proxima's unified format.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class NormalizedResult:
    """Normalized execution result."""
    measurements: Dict[str, int]
    probabilities: Optional[Dict[str, float]] = None
    state_vector: Optional[List[complex]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {{}}


class {class_name}Normalizer:
    """Normalizes {class_name} results to unified format."""
    
    @staticmethod
    def normalize(raw_result: Any, shots: int) -> NormalizedResult:
        """Normalize raw backend result.
        
        Args:
            raw_result: Raw result from backend
            shots: Number of shots used
            
        Returns:
            Normalized result
        """
        # TODO: Implement normalization for your backend
        # Extract measurements, probabilities, state vectors
        
        measurements = {{}}  # TODO: Extract from raw_result
        
        return NormalizedResult(
            measurements=measurements,
            metadata={{"shots": shots}}
        )
    
    @staticmethod
    def validate_counts(measurements: Dict[str, int], shots: int) -> bool:
        """Validate that measurement counts sum to shots.
        
        Args:
            measurements: Measurement count dictionary
            shots: Expected total shots
            
        Returns:
            True if counts are valid
        """
        total = sum(measurements.values())
        return total == shots
'''
    
    def _generate_readme(self, backend_name: str, display_name: str) -> str:
        """Generate README.md content."""
        return f'''# {display_name} Backend

Auto-generated by Proxima Backend Addition Wizard.

## Overview

This backend provides integration with {display_name} for Proxima.

## Configuration

```python
from proxima.backends.contrib.{backend_name} import get_adapter

adapter = get_adapter(
    max_qubits={self.state.max_qubits or 20},
    shots=1024
)
adapter.initialize()
```

## Features

- Max Qubits: {self.state.max_qubits or 20}
- Simulator Types: {", ".join(self.state.simulator_types) if self.state.simulator_types else "state_vector"}
- Noise Simulation: {"Yes" if self.state.supports_noise else "No"}
- GPU Acceleration: {"Yes" if self.state.supports_gpu else "No"}

## Usage

```python
# Run a circuit
result = adapter.run(circuit, shots=1024)

# Get capabilities
capabilities = adapter.get_capabilities()
```

## TODO

- [ ] Implement `run()` method
- [ ] Implement circuit conversion
- [ ] Add result normalization
- [ ] Add comprehensive tests
'''
    
    def _to_class_name(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
    
    async def _deploy_backend(self) -> None:
        """Deploy the backend with progress tracking."""
        self.deploying = True
        
        # Show progress widget
        try:
            progress = self.query_one("#deployment_progress", DeploymentProgressWidget)
            progress.display = True
        except Exception:
            pass
        
        # Disable buttons during deployment
        self._set_buttons_enabled(False)
        
        try:
            # Get generated code
            generated_code = self._get_generated_code()
            
            backend_name = self.state.backend_name or "my_backend"
            display_name = self.state.display_name or backend_name.replace("_", " ").title()
            
            # Import and use DeploymentManager
            try:
                from proxima.tui.controllers.deployment_manager import (
                    DeploymentManager,
                    DeploymentProgress
                )
                
                def on_progress(dp: DeploymentProgress):
                    try:
                        widget = self.query_one("#deployment_progress", DeploymentProgressWidget)
                        widget.stage = dp.stage.value.replace("_", " ").title()
                        widget.current_file = dp.current_file or ""
                        widget.progress = dp.progress
                    except Exception:
                        pass
                
                manager = DeploymentManager(on_progress=on_progress)
                
                result = await manager.deploy_backend(
                    backend_name=backend_name,
                    display_name=display_name,
                    generated_code=generated_code
                )
                
                if result.success:
                    # Show success dialog
                    await self.app.push_screen(DeploymentSuccessDialog(
                        backend_name=backend_name,
                        display_name=display_name,
                        created_files=result.created_files,
                        output_path=result.output_path
                    ))
                    
                    # Complete wizard
                    self.dismiss({
                        "action": "complete",
                        "state": self.state,
                        "output_path": result.output_path,
                        "created_files": result.created_files
                    })
                else:
                    # Show failure dialog
                    await self.app.push_screen(DeploymentFailureDialog(
                        error_message=result.error_message or "Unknown error",
                        backend_name=backend_name
                    ))
                    
            except ImportError:
                # Fallback to basic file writing
                await self._deploy_basic(backend_name, display_name, generated_code)
                
        except Exception as e:
            self.notify(f"Deployment failed: {e}", severity="error")
            
        finally:
            self.deploying = False
            self._set_buttons_enabled(True)
    
    async def _deploy_basic(
        self,
        backend_name: str,
        display_name: str,
        generated_code: Dict[str, str]
    ) -> None:
        """Basic deployment without DeploymentManager."""
        try:
            output_dir = Path("src/proxima/backends/contrib") / backend_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            created_files = []
            
            for file_name, content in generated_code.items():
                file_path = output_dir / file_name
                file_path.write_text(content, encoding='utf-8')
                created_files.append(str(file_path))
            
            self.notify(
                f"✅ Backend deployed: {output_dir}",
                severity="information",
                timeout=5
            )
            
            self.dismiss({
                "action": "complete",
                "state": self.state,
                "output_path": str(output_dir),
                "created_files": created_files
            })
            
        except Exception as e:
            self.notify(f"❌ Deployment failed: {e}", severity="error")
    
    def _set_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable navigation buttons."""
        button_ids = ["btn_back", "btn_cancel", "btn_deploy", "btn_view_code"]
        
        for btn_id in button_ids:
            try:
                button = self.query_one(f"#{btn_id}", Button)
                button.disabled = not enabled
            except Exception:
                pass
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        if not self.deploying:
            self.dismiss({"action": "cancel"})
