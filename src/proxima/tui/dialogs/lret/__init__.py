"""LRET Dialogs Package.

Contains dialogs for LRET variant management:
- LRETInstallerDialog: Install LRET backend variants
- LRETConfigDialog: Configure LRET variant settings
- LRETBenchmarkDialog: Run and view LRET vs Cirq benchmarks
- PennyLaneAlgorithmDialog: Run VQE, QAOA, QNN algorithms
- Phase7Dialog: Configure Phase 7 unified multi-framework execution
- VariantAnalysisDialog: Analyze and compare all LRET variants
- ValidationChecklistDialog: Run validation checklist for all LRET variants
"""

from proxima.tui.dialogs.lret.installer_dialog import LRETInstallerDialog
from proxima.tui.dialogs.lret.config_dialog import LRETConfigDialog
from proxima.tui.dialogs.lret.benchmark_dialog import LRETBenchmarkDialog
from proxima.tui.dialogs.lret.algorithm_dialog import PennyLaneAlgorithmDialog
from proxima.tui.dialogs.lret.phase7_dialog import Phase7Dialog
from proxima.tui.dialogs.lret.variant_analysis_dialog import VariantAnalysisDialog
from proxima.tui.dialogs.lret.validation_dialog import ValidationChecklistDialog

__all__ = [
    "LRETInstallerDialog",
    "LRETConfigDialog",
    "LRETBenchmarkDialog",
    "PennyLaneAlgorithmDialog",
    "Phase7Dialog",
    "VariantAnalysisDialog",
    "ValidationChecklistDialog",
]
