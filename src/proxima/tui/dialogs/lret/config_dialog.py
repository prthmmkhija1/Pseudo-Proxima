"""LRET Configuration Dialog for TUI.

Provides a modal dialog for configuring LRET backend variant settings:
- Enable/disable variants
- Set priorities for auto-selection
- Configure variant-specific options
- Set default variant

Navigation:
- Settings → Backend Management → Configure LRET Variants
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import (
    Static,
    Button,
    Input,
    Switch,
    Select,
    Label,
    Rule,
    TabbedContent,
    TabPane,
)
from textual.reactive import reactive

logger = logging.getLogger(__name__)


# Try to import LRET modules
try:
    from proxima.backends.lret.config import (
        LRETConfig,
        LRETVariantConfig,
        LRETVariantType,
        get_lret_config,
        save_lret_config,
    )
    from proxima.backends.lret.installer import (
        LRET_VARIANTS,
        check_variant_availability,
    )
    LRET_MODULE_AVAILABLE = True
except ImportError:
    LRET_MODULE_AVAILABLE = False
    LRET_VARIANTS = {}


class LRETConfigDialog(ModalScreen):
    """Dialog for configuring LRET variant settings.
    
    This dialog provides a tabbed interface for configuring each LRET
    variant's settings, including:
    - General settings (enable/disable, priority)
    - Cirq Scalability settings (FDM threshold, benchmark output)
    - PennyLane Hybrid settings (shots, diff method, optimizer)
    - Phase 7 Unified settings (framework preference, GPU, optimization)
    
    Bindings:
        escape: Cancel and close dialog
        ctrl+s: Save configuration
    """
    
    DEFAULT_CSS = """
    LRETConfigDialog {
        align: center middle;
    }
    
    LRETConfigDialog .dialog-container {
        width: 95;
        max-width: 110;
        height: 38;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    
    LRETConfigDialog .dialog-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
        width: 100%;
    }
    
    LRETConfigDialog .config-section {
        height: auto;
        margin: 1 0;
        padding: 1;
        border: solid $primary-darken-2;
    }
    
    LRETConfigDialog .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    LRETConfigDialog .config-row {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
    }
    
    LRETConfigDialog .config-label {
        width: 25;
        padding-top: 1;
    }
    
    LRETConfigDialog .config-input {
        width: 35;
    }
    
    LRETConfigDialog .config-help {
        color: $text-muted;
        margin-left: 2;
        width: 1fr;
    }
    
    LRETConfigDialog .dialog-footer {
        height: auto;
        margin-top: 1;
        padding-top: 1;
        border-top: solid $primary-darken-2;
    }
    
    LRETConfigDialog .dialog-footer Button {
        margin-right: 1;
    }
    
    LRETConfigDialog .status-indicator {
        padding: 0 1;
        margin-right: 1;
    }
    
    LRETConfigDialog .status-enabled {
        color: $success;
    }
    
    LRETConfigDialog .status-disabled {
        color: $text-muted;
    }
    
    LRETConfigDialog TabbedContent {
        height: 1fr;
    }
    
    LRETConfigDialog TabPane {
        padding: 1;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+s", "save", "Save"),
    ]
    
    has_changes: reactive[bool] = reactive(False)
    
    def __init__(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        super().__init__(name=name, id=id, classes=classes)
        self._config: Optional[LRETConfig] = None
        self._original_config: Optional[dict] = None
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog-container"):
            yield Static("⚙️ Configure LRET Backend Variants", classes="dialog-title")
            
            if not LRET_MODULE_AVAILABLE:
                yield Static(
                    "⚠️ LRET module not available. Please install it first.",
                    classes="not-available-message"
                )
                with Horizontal(classes="dialog-footer"):
                    yield Button("Close", variant="default", id="cancel-btn")
                return
            
            # Load configuration
            self._config = get_lret_config()
            self._original_config = self._config.to_dict()
            
            # Tabbed content for each variant
            with TabbedContent():
                # General settings tab
                with TabPane("General", id="tab-general"):
                    yield from self._compose_general_settings()
                
                # Cirq Scalability tab
                with TabPane("Cirq Scalability", id="tab-cirq"):
                    yield from self._compose_cirq_settings()
                
                # PennyLane Hybrid tab
                with TabPane("PennyLane", id="tab-pennylane"):
                    yield from self._compose_pennylane_settings()
                
                # Phase 7 Unified tab
                with TabPane("Phase 7", id="tab-phase7"):
                    yield from self._compose_phase7_settings()
            
            # Footer buttons
            with Horizontal(classes="dialog-footer"):
                yield Button("Save", variant="primary", id="save-btn")
                yield Button("Apply", variant="success", id="apply-btn")
                yield Button("Reset", variant="warning", id="reset-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")
    
    def _compose_general_settings(self) -> ComposeResult:
        """Compose general settings section."""
        with Vertical(classes="config-section"):
            yield Static("Default Settings", classes="section-title")
            
            with Horizontal(classes="config-row"):
                yield Label("Default Variant:", classes="config-label")
                yield Select(
                    [
                        ("Auto-select", "auto"),
                        ("Cirq Scalability", "cirq_scalability"),
                        ("PennyLane Hybrid", "pennylane_hybrid"),
                        ("Phase 7 Unified", "phase7_unified"),
                    ],
                    value=self._config.default_variant or "auto",
                    id="config-default-variant",
                    classes="config-input",
                )
            
            with Horizontal(classes="config-row"):
                yield Label("Install Directory:", classes="config-label")
                yield Input(
                    value=self._config.install_base_dir,
                    id="config-install-dir",
                    classes="config-input",
                )
        
        yield Rule()
        
        # Variant enable/disable section
        with Vertical(classes="config-section"):
            yield Static("Enable/Disable Variants", classes="section-title")
            
            for variant_name, variant_info in LRET_VARIANTS.items():
                variant_config = self._config.get_variant_config(variant_name)
                status = check_variant_availability(variant_name)
                
                with Horizontal(classes="config-row"):
                    yield Label(
                        variant_info.get("display_name", variant_name) + ":",
                        classes="config-label"
                    )
                    yield Switch(
                        value=variant_config.enabled,
                        id=f"config-{variant_name}-enabled",
                    )
                    
                    # Status indicator
                    if status.installed:
                        yield Static("✓ Installed", classes="status-indicator status-enabled")
                    else:
                        yield Static("Not installed", classes="status-indicator status-disabled")
    
    def _compose_cirq_settings(self) -> ComposeResult:
        """Compose Cirq Scalability settings."""
        cirq_config = self._config.cirq_scalability
        
        with Vertical(classes="config-section"):
            yield Static("Cirq Scalability Settings", classes="section-title")
            
            with Horizontal(classes="config-row"):
                yield Label("Priority:", classes="config-label")
                yield Input(
                    value=str(cirq_config.priority),
                    id="config-cirq-priority",
                    classes="config-input",
                    type="integer",
                )
                yield Static("Higher = preferred for auto-selection", classes="config-help")
            
            with Horizontal(classes="config-row"):
                yield Label("FDM Threshold:", classes="config-label")
                yield Input(
                    value=str(cirq_config.cirq_fdm_threshold),
                    id="config-cirq-fdm-threshold",
                    classes="config-input",
                    type="integer",
                )
                yield Static("Prefer Cirq FDM above this qubit count", classes="config-help")
            
            with Horizontal(classes="config-row"):
                yield Label("Benchmark Output:", classes="config-label")
                yield Input(
                    value=cirq_config.benchmark_output_dir,
                    id="config-cirq-benchmark-dir",
                    classes="config-input",
                )
            
            with Horizontal(classes="config-row"):
                yield Label("Comparison Mode:", classes="config-label")
                yield Switch(
                    value=cirq_config.enable_comparison_mode,
                    id="config-cirq-comparison",
                )
                yield Static("Auto-compare with Cirq FDM", classes="config-help")
            
            with Horizontal(classes="config-row"):
                yield Label("Auto-connect:", classes="config-label")
                yield Switch(
                    value=cirq_config.auto_connect,
                    id="config-cirq-autoconnect",
                )
    
    def _compose_pennylane_settings(self) -> ComposeResult:
        """Compose PennyLane Hybrid settings."""
        pl_config = self._config.pennylane_hybrid
        
        with Vertical(classes="config-section"):
            yield Static("PennyLane Hybrid Settings", classes="section-title")
            
            with Horizontal(classes="config-row"):
                yield Label("Priority:", classes="config-label")
                yield Input(
                    value=str(pl_config.priority),
                    id="config-pennylane-priority",
                    classes="config-input",
                    type="integer",
                )
            
            with Horizontal(classes="config-row"):
                yield Label("Default Shots:", classes="config-label")
                yield Input(
                    value=str(pl_config.pennylane_shots),
                    id="config-pennylane-shots",
                    classes="config-input",
                    type="integer",
                )
                yield Static("Number of measurement shots", classes="config-help")
            
            with Horizontal(classes="config-row"):
                yield Label("Diff Method:", classes="config-label")
                yield Select(
                    [
                        ("Parameter Shift", "parameter-shift"),
                        ("Adjoint", "adjoint"),
                        ("Backprop", "backprop"),
                    ],
                    value=pl_config.pennylane_diff_method,
                    id="config-pennylane-diff-method",
                    classes="config-input",
                )
                yield Static("Gradient computation method", classes="config-help")
            
            with Horizontal(classes="config-row"):
                yield Label("Default Optimizer:", classes="config-label")
                yield Select(
                    [
                        ("Adam", "adam"),
                        ("SGD", "sgd"),
                        ("Momentum", "momentum"),
                        ("RMSProp", "rmsprop"),
                    ],
                    value=pl_config.default_optimizer,
                    id="config-pennylane-optimizer",
                    classes="config-input",
                )
            
            with Horizontal(classes="config-row"):
                yield Label("Auto-connect:", classes="config-label")
                yield Switch(
                    value=pl_config.auto_connect,
                    id="config-pennylane-autoconnect",
                )
    
    def _compose_phase7_settings(self) -> ComposeResult:
        """Compose Phase 7 Unified settings."""
        p7_config = self._config.phase7_unified
        
        with Vertical(classes="config-section"):
            yield Static("Phase 7 Unified Settings", classes="section-title")
            
            with Horizontal(classes="config-row"):
                yield Label("Priority:", classes="config-label")
                yield Input(
                    value=str(p7_config.priority),
                    id="config-phase7-priority",
                    classes="config-input",
                    type="integer",
                )
            
            with Horizontal(classes="config-row"):
                yield Label("Optimization Level:", classes="config-label")
                yield Select(
                    [
                        ("None (0)", "0"),
                        ("Basic (1)", "1"),
                        ("Full (2)", "2"),
                    ],
                    value=str(p7_config.optimization_level),
                    id="config-phase7-optimization",
                    classes="config-input",
                )
                yield Static("Gate fusion & optimization", classes="config-help")
            
            with Horizontal(classes="config-row"):
                yield Label("Gate Fusion:", classes="config-label")
                yield Switch(
                    value=p7_config.gate_fusion_enabled,
                    id="config-phase7-fusion",
                )
            
            with Horizontal(classes="config-row"):
                yield Label("Fusion Mode:", classes="config-label")
                yield Select(
                    [
                        ("Row", "row"),
                        ("Column", "column"),
                        ("Hybrid", "hybrid"),
                    ],
                    value=p7_config.gate_fusion_mode,
                    id="config-phase7-fusion-mode",
                    classes="config-input",
                )
        
        yield Rule()
        
        with Vertical(classes="config-section"):
            yield Static("GPU Acceleration", classes="section-title")
            
            with Horizontal(classes="config-row"):
                yield Label("Enable GPU:", classes="config-label")
                yield Switch(
                    value=p7_config.gpu_enabled,
                    id="config-phase7-gpu",
                )
                yield Static("Requires cuQuantum", classes="config-help")
            
            with Horizontal(classes="config-row"):
                yield Label("GPU Device ID:", classes="config-label")
                yield Input(
                    value=str(p7_config.gpu_device_id),
                    id="config-phase7-gpu-id",
                    classes="config-input",
                    type="integer",
                )
            
            with Horizontal(classes="config-row"):
                yield Label("Auto-connect:", classes="config-label")
                yield Switch(
                    value=p7_config.auto_connect,
                    id="config-phase7-autoconnect",
                )
    
    def _collect_config_values(self) -> LRETConfig:
        """Collect current values from UI into config."""
        config = self._config
        
        # General settings
        default_var = self.query_one("#config-default-variant", Select).value
        config.default_variant = None if default_var == "auto" else default_var
        
        try:
            config.install_base_dir = self.query_one("#config-install-dir", Input).value
        except Exception:
            pass
        
        # Variant enabled states
        for variant_name in LRET_VARIANTS:
            try:
                enabled = self.query_one(f"#config-{variant_name}-enabled", Switch).value
                config.get_variant_config(variant_name).enabled = enabled
            except Exception:
                pass
        
        # Cirq settings
        try:
            config.cirq_scalability.priority = int(
                self.query_one("#config-cirq-priority", Input).value
            )
            config.cirq_scalability.cirq_fdm_threshold = int(
                self.query_one("#config-cirq-fdm-threshold", Input).value
            )
            config.cirq_scalability.benchmark_output_dir = self.query_one(
                "#config-cirq-benchmark-dir", Input
            ).value
            config.cirq_scalability.enable_comparison_mode = self.query_one(
                "#config-cirq-comparison", Switch
            ).value
            config.cirq_scalability.auto_connect = self.query_one(
                "#config-cirq-autoconnect", Switch
            ).value
        except Exception as e:
            logger.debug(f"Error collecting cirq config: {e}")
        
        # PennyLane settings
        try:
            config.pennylane_hybrid.priority = int(
                self.query_one("#config-pennylane-priority", Input).value
            )
            config.pennylane_hybrid.pennylane_shots = int(
                self.query_one("#config-pennylane-shots", Input).value
            )
            config.pennylane_hybrid.pennylane_diff_method = self.query_one(
                "#config-pennylane-diff-method", Select
            ).value
            config.pennylane_hybrid.default_optimizer = self.query_one(
                "#config-pennylane-optimizer", Select
            ).value
            config.pennylane_hybrid.auto_connect = self.query_one(
                "#config-pennylane-autoconnect", Switch
            ).value
        except Exception as e:
            logger.debug(f"Error collecting pennylane config: {e}")
        
        # Phase 7 settings
        try:
            config.phase7_unified.priority = int(
                self.query_one("#config-phase7-priority", Input).value
            )
            config.phase7_unified.optimization_level = int(
                self.query_one("#config-phase7-optimization", Select).value
            )
            config.phase7_unified.gate_fusion_enabled = self.query_one(
                "#config-phase7-fusion", Switch
            ).value
            config.phase7_unified.gate_fusion_mode = self.query_one(
                "#config-phase7-fusion-mode", Select
            ).value
            config.phase7_unified.gpu_enabled = self.query_one(
                "#config-phase7-gpu", Switch
            ).value
            config.phase7_unified.gpu_device_id = int(
                self.query_one("#config-phase7-gpu-id", Input).value
            )
            config.phase7_unified.auto_connect = self.query_one(
                "#config-phase7-autoconnect", Switch
            ).value
        except Exception as e:
            logger.debug(f"Error collecting phase7 config: {e}")
        
        return config
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            await self.action_cancel()
        elif event.button.id == "save-btn":
            await self.action_save()
            self.dismiss({"saved": True})
        elif event.button.id == "apply-btn":
            await self.action_save()
            self.notify("Configuration applied", severity="information")
        elif event.button.id == "reset-btn":
            await self._reset_to_defaults()
    
    async def action_cancel(self) -> None:
        """Cancel and close dialog."""
        self.dismiss({"saved": False})
    
    async def action_save(self) -> None:
        """Save configuration."""
        if not LRET_MODULE_AVAILABLE or not self._config:
            self.notify("Cannot save: LRET module not available", severity="error")
            return
        
        try:
            config = self._collect_config_values()
            
            # Validate
            is_valid, errors = config.validate()
            if not is_valid:
                for error in errors[:3]:
                    self.notify(f"Validation error: {error}", severity="error")
                return
            
            # Save
            path = save_lret_config(config)
            self.notify(f"Configuration saved to {path}", severity="information")
            self.has_changes = False
            
        except Exception as e:
            logger.exception("Failed to save configuration")
            self.notify(f"Save failed: {str(e)[:50]}", severity="error")
    
    async def _reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        if not LRET_MODULE_AVAILABLE:
            return
        
        from proxima.backends.lret.config import reset_lret_config
        
        self._config = reset_lret_config()
        self.notify("Configuration reset to defaults", severity="information")
        
        # Refresh dialog
        await self.recompose()
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Track changes to inputs."""
        self.has_changes = True
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Track changes to switches."""
        self.has_changes = True
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Track changes to selects."""
        self.has_changes = True
