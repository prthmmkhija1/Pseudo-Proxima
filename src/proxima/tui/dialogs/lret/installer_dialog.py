"""LRET Installer Dialog for TUI.

Provides a modal dialog for installing LRET backend variants:
1. Cirq Scalability - Performance comparison & benchmarking
2. PennyLane Hybrid - VQE, QAOA, gradient-based optimization
3. Phase 7 Unified - Multi-framework integration

Features:
- Checkbox selection for each variant
- Real-time progress bar during installation
- Status messages and error display
- Skip optional dependencies option
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Callable, Any

from textual import work
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import (
    Static,
    Button,
    Checkbox,
    Label,
    ProgressBar,
    Switch,
    Rule,
    LoadingIndicator,
)
from textual.reactive import reactive

logger = logging.getLogger(__name__)


# Try to import LRET modules - gracefully handle if not available
try:
    from proxima.backends.lret.installer import (
        LRET_VARIANTS,
        install_lret_variant,
        check_variant_availability,
        InstallationResult,
        InstallationStatus,
    )
    from proxima.backends.lret.config import get_lret_config
    LRET_MODULE_AVAILABLE = True
except ImportError:
    LRET_MODULE_AVAILABLE = False
    LRET_VARIANTS = {}


class VariantCheckbox(Horizontal):
    """Custom widget for variant selection with description."""
    
    DEFAULT_CSS = """
    VariantCheckbox {
        height: auto;
        margin: 0 0 1 0;
        padding: 1;
        background: $surface-lighten-1;
        border: solid $primary-darken-2;
    }
    
    VariantCheckbox:focus-within {
        border: solid $accent;
    }
    
    VariantCheckbox .variant-info {
        width: 1fr;
        padding-left: 1;
    }
    
    VariantCheckbox .variant-name {
        text-style: bold;
        color: $text;
    }
    
    VariantCheckbox .variant-desc {
        color: $text-muted;
    }
    
    VariantCheckbox .variant-features {
        color: $success;
        margin-top: 1;
    }
    
    VariantCheckbox .variant-status {
        width: 15;
        text-align: right;
        padding-right: 1;
    }
    
    VariantCheckbox .status-installed {
        color: $success;
    }
    
    VariantCheckbox .status-not-installed {
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        variant_name: str,
        variant_info: dict[str, Any],
        installed: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.variant_name = variant_name
        self.variant_info = variant_info
        self.installed = installed
    
    def compose(self) -> ComposeResult:
        yield Checkbox(
            "",
            id=f"check-{self.variant_name}",
            value=False,
        )
        
        with Vertical(classes="variant-info"):
            yield Static(
                f"[bold]{self.variant_info.get('display_name', self.variant_name)}[/bold]",
                classes="variant-name"
            )
            yield Static(
                self.variant_info.get("description", "No description"),
                classes="variant-desc"
            )
            features = self.variant_info.get("features", [])[:3]
            if features:
                yield Static(
                    "📦 " + " • ".join(features),
                    classes="variant-features"
                )
        
        status_class = "status-installed" if self.installed else "status-not-installed"
        status_text = "✓ Installed" if self.installed else "Not installed"
        yield Static(status_text, classes=f"variant-status {status_class}")


class LRETInstallerDialog(ModalScreen):
    """Dialog for installing LRET backend variants.
    
    This dialog allows users to select and install one or more LRET
    backend variants with real-time progress feedback.
    
    Bindings:
        escape: Cancel and close dialog
        enter: Install selected variants
    
    Messages:
        Dismissed with dict containing:
        - 'installed': list of successfully installed variants
        - 'failed': list of failed variants
        - 'cancelled': True if user cancelled
    """
    
    DEFAULT_CSS = """
    LRETInstallerDialog {
        align: center middle;
    }
    
    LRETInstallerDialog .dialog-container {
        width: 90;
        max-width: 100;
        height: auto;
        max-height: 40;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    
    LRETInstallerDialog .dialog-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
        width: 100%;
    }
    
    LRETInstallerDialog .section-header {
        text-style: bold;
        color: $primary;
        margin: 1 0;
    }
    
    LRETInstallerDialog .variants-container {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }
    
    LRETInstallerDialog .options-section {
        height: auto;
        margin: 1 0;
        padding: 1;
        background: $surface-darken-1;
    }
    
    LRETInstallerDialog .option-row {
        height: auto;
        margin: 0 0 1 0;
    }
    
    LRETInstallerDialog .option-label {
        width: 30;
    }
    
    LRETInstallerDialog .progress-section {
        height: auto;
        padding: 1;
        margin: 1 0;
        border: solid $primary-darken-2;
        display: none;
    }
    
    LRETInstallerDialog .progress-section.visible {
        display: block;
    }
    
    LRETInstallerDialog .progress-label {
        margin-bottom: 1;
    }
    
    LRETInstallerDialog .status-message {
        margin-top: 1;
        color: $text-muted;
    }
    
    LRETInstallerDialog .status-error {
        color: $error;
    }
    
    LRETInstallerDialog .status-success {
        color: $success;
    }
    
    LRETInstallerDialog .dialog-footer {
        height: auto;
        margin-top: 1;
        padding-top: 1;
        border-top: solid $primary-darken-2;
    }
    
    LRETInstallerDialog .dialog-footer Button {
        margin-right: 1;
    }
    
    LRETInstallerDialog .not-available-message {
        padding: 2;
        color: $warning;
        text-align: center;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "install", "Install"),
    ]
    
    is_installing: reactive[bool] = reactive(False)
    progress_value: reactive[float] = reactive(0.0)
    status_text: reactive[str] = reactive("")
    
    def __init__(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        super().__init__(name=name, id=id, classes=classes)
        self._installed_variants: list[str] = []
        self._failed_variants: list[str] = []
        self._variant_statuses: dict[str, bool] = {}
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog-container"):
            yield Static("📦 Install LRET Backend Variants", classes="dialog-title")
            
            if not LRET_MODULE_AVAILABLE:
                yield Static(
                    "⚠️ LRET module not available. Please check installation.",
                    classes="not-available-message"
                )
                with Horizontal(classes="dialog-footer"):
                    yield Button("Close", variant="default", id="cancel-btn")
                return
            
            yield Static("Select variants to install:", classes="section-header")
            
            # Load current installation status
            self._load_variant_statuses()
            
            with ScrollableContainer(classes="variants-container"):
                for variant_name, variant_info in LRET_VARIANTS.items():
                    installed = self._variant_statuses.get(variant_name, False)
                    yield VariantCheckbox(
                        variant_name=variant_name,
                        variant_info=variant_info,
                        installed=installed,
                        id=f"variant-{variant_name}",
                    )
            
            yield Rule()
            
            # Options section
            with Vertical(classes="options-section"):
                yield Static("Installation Options", classes="section-header")
                
                with Horizontal(classes="option-row"):
                    yield Label("Skip optional dependencies:", classes="option-label")
                    yield Switch(value=False, id="skip-optional")
                
                with Horizontal(classes="option-row"):
                    yield Label("Select all variants:", classes="option-label")
                    yield Switch(value=False, id="select-all")
            
            # Progress section (hidden initially)
            with Vertical(classes="progress-section", id="progress-section"):
                yield Label("Installation Progress:", classes="progress-label", id="progress-label")
                yield ProgressBar(total=100, id="install-progress", show_eta=False)
                yield Label("", classes="status-message", id="status-message")
            
            # Footer buttons
            with Horizontal(classes="dialog-footer"):
                yield Button("Install Selected", variant="primary", id="install-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")
    
    def _load_variant_statuses(self) -> None:
        """Load installation status for all variants."""
        if not LRET_MODULE_AVAILABLE:
            return
        
        for variant_name in LRET_VARIANTS:
            try:
                status = check_variant_availability(variant_name)
                self._variant_statuses[variant_name] = status.installed
            except Exception as e:
                logger.debug(f"Failed to check status for {variant_name}: {e}")
                self._variant_statuses[variant_name] = False
    
    def on_mount(self) -> None:
        """Handle dialog mount."""
        self.query_one("#install-btn", Button).focus()
    
    def watch_is_installing(self, installing: bool) -> None:
        """React to installation state changes."""
        install_btn = self.query_one("#install-btn", Button)
        cancel_btn = self.query_one("#cancel-btn", Button)
        progress_section = self.query_one("#progress-section")
        
        if installing:
            install_btn.disabled = True
            install_btn.label = "Installing..."
            cancel_btn.label = "Cancel Installation"
            progress_section.add_class("visible")
        else:
            install_btn.disabled = False
            install_btn.label = "Install Selected"
            cancel_btn.label = "Close"
    
    def watch_progress_value(self, value: float) -> None:
        """Update progress bar."""
        try:
            progress_bar = self.query_one("#install-progress", ProgressBar)
            progress_bar.update(progress=value)
        except Exception:
            pass
    
    def watch_status_text(self, text: str) -> None:
        """Update status message."""
        try:
            status_label = self.query_one("#status-message", Label)
            status_label.update(text)
        except Exception:
            pass
    
    async def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "select-all":
            # Toggle all checkboxes
            for variant_name in LRET_VARIANTS:
                try:
                    checkbox = self.query_one(f"#check-{variant_name}", Checkbox)
                    checkbox.value = event.value
                except Exception:
                    pass
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            await self.action_cancel()
        elif event.button.id == "install-btn":
            await self.action_install()
    
    async def action_cancel(self) -> None:
        """Cancel and close the dialog."""
        if self.is_installing:
            # TODO: Implement installation cancellation
            self.notify("Installation in progress...", severity="warning")
            return
        
        self.dismiss({
            "installed": self._installed_variants,
            "failed": self._failed_variants,
            "cancelled": True,
        })
    
    async def action_install(self) -> None:
        """Install selected variants."""
        if not LRET_MODULE_AVAILABLE:
            self.notify("LRET module not available", severity="error")
            return
        
        if self.is_installing:
            return
        
        # Get selected variants
        selected = []
        for variant_name in LRET_VARIANTS:
            try:
                checkbox = self.query_one(f"#check-{variant_name}", Checkbox)
                if checkbox.value:
                    selected.append(variant_name)
            except Exception:
                pass
        
        if not selected:
            self.notify("Please select at least one variant", severity="warning")
            return
        
        # Check if variants are already installed
        to_install = []
        for variant in selected:
            if self._variant_statuses.get(variant, False):
                self.notify(f"{variant} is already installed", severity="information")
            else:
                to_install.append(variant)
        
        if not to_install:
            self.notify("All selected variants are already installed", severity="information")
            return
        
        # Start installation
        skip_optional = self.query_one("#skip-optional", Switch).value
        self._run_installation(to_install, skip_optional)
    
    @work(exclusive=True, thread=True)
    def _run_installation(self, variants: list[str], skip_optional: bool) -> None:
        """Run installation in background thread."""
        self.is_installing = True
        self._installed_variants = []
        self._failed_variants = []
        
        total_variants = len(variants)
        
        for i, variant in enumerate(variants):
            base_progress = (i / total_variants) * 100
            
            def progress_callback(message: str, percent: float):
                overall = base_progress + (percent / total_variants)
                self.progress_value = overall
                self.status_text = f"[{variant}] {message}"
            
            self.status_text = f"Installing {variant}..."
            
            try:
                result = install_lret_variant(
                    variant,
                    progress_callback=progress_callback,
                    skip_optional_deps=skip_optional,
                )
                
                if result.success:
                    self._installed_variants.append(variant)
                    self.call_from_thread(
                        self.notify,
                        f"✓ {variant} installed successfully",
                        severity="information",
                    )
                else:
                    self._failed_variants.append(variant)
                    error_msg = result.errors[0] if result.errors else result.message
                    self.call_from_thread(
                        self.notify,
                        f"✗ {variant} failed: {error_msg}",
                        severity="error",
                    )
            except Exception as e:
                logger.exception(f"Installation failed for {variant}")
                self._failed_variants.append(variant)
                self.call_from_thread(
                    self.notify,
                    f"✗ {variant} error: {str(e)[:50]}",
                    severity="error",
                )
        
        # Complete
        self.progress_value = 100
        
        if self._installed_variants:
            self.status_text = f"✓ Installed: {', '.join(self._installed_variants)}"
            if self._failed_variants:
                self.status_text += f" | ✗ Failed: {', '.join(self._failed_variants)}"
        elif self._failed_variants:
            self.status_text = f"✗ All installations failed"
        else:
            self.status_text = "No variants installed"
        
        self.is_installing = False
        
        # Update variant statuses
        self.call_from_thread(self._refresh_statuses)
    
    def _refresh_statuses(self) -> None:
        """Refresh variant installation statuses after installation."""
        self._load_variant_statuses()
        
        # Update UI
        for variant_name in LRET_VARIANTS:
            installed = self._variant_statuses.get(variant_name, False)
            try:
                widget = self.query_one(f"#variant-{variant_name}", VariantCheckbox)
                status_widget = widget.query_one(".variant-status", Static)
                status_widget.update("✓ Installed" if installed else "Not installed")
                status_widget.remove_class("status-not-installed")
                status_widget.remove_class("status-installed")
                status_widget.add_class("status-installed" if installed else "status-not-installed")
            except Exception:
                pass
