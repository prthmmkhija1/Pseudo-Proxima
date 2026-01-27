"""Backend dialogs for enhanced TUI functionality."""

from .comparison import BackendComparisonDialog
from .metrics import BackendMetricsDialog  
from .config import BackendConfigDialog

# Alias for backward compatibility
BackendsDialog = BackendComparisonDialog

__all__ = [
    "BackendComparisonDialog",
    "BackendMetricsDialog",
    "BackendConfigDialog",
    "BackendsDialog",
]
