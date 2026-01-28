"""LRET Backend Variants Module.

This module provides support for three LRET backend variants:
1. cirq_scalability - Cirq FDM comparison and benchmarking
2. pennylane_hybrid - PennyLane device plugin for VQE/QAOA
3. phase7_unified - Multi-framework integration (Cirq, PennyLane, Qiskit)

Each variant can be installed and configured independently.
"""

from proxima.backends.lret.config import (
    LRETConfig,
    LRETVariantConfig,
    LRETVariantType,
)
from proxima.backends.lret.installer import (
    LRET_VARIANTS,
    check_variant_availability,
    get_variant_install_path,
    install_lret_variant,
    uninstall_lret_variant,
    list_installed_variants,
    verify_variant_installation,
)
from proxima.backends.lret.cirq_scalability import (
    LRETCirqScalabilityAdapter,
    CirqScalabilityMetrics,
    BenchmarkResult,
)
from proxima.backends.lret.visualization import (
    load_benchmark_csv,
    plot_lret_cirq_comparison,
    plot_speedup_trend,
    plot_fidelity_analysis,
    generate_benchmark_report,
    create_summary_table,
    PlotConfig,
)
from proxima.backends.lret.pennylane_device import (
    QLRETDevice,
    create_lret_device,
)
from proxima.backends.lret.algorithms import (
    VQE,
    VQEResult,
    QAOA,
    QAOAResult,
    QuantumNeuralNetwork,
)
from proxima.backends.lret.phase7_unified import (
    LRETPhase7UnifiedAdapter,
    Phase7Config,
    Framework,
    FusionMode,
    GateFusion,
    GateFusionStats,
    Phase7ExecutionMetrics,
    UnifiedExecutor,
    FrameworkConverter,
    create_phase7_adapter,
)
from proxima.backends.lret.variant_analysis import (
    VariantAnalyzer,
    VariantInfo,
    VariantAnalysisResult,
    VariantComparisonResult,
    VariantCapability,
    TaskType,
    VARIANT_DEFINITIONS,
    get_variant_analyzer,
    analyze_all_variants,
    get_best_variant_for_task,
)
from proxima.backends.lret.variant_registry import (
    VariantRegistry,
    VariantStatus,
    RegisteredVariant,
    PennyLaneHybridAdapter,
    get_registry,
    list_available_variants,
    get_variant_adapter,
    get_best_variant_adapter,
)

__all__ = [
    # Configuration
    "LRETConfig",
    "LRETVariantConfig",
    "LRETVariantType",
    # Installer
    "LRET_VARIANTS",
    "check_variant_availability",
    "get_variant_install_path",
    "install_lret_variant",
    "uninstall_lret_variant",
    "list_installed_variants",
    "verify_variant_installation",
    # Cirq Scalability Adapter
    "LRETCirqScalabilityAdapter",
    "CirqScalabilityMetrics",
    "BenchmarkResult",
    # Visualization
    "load_benchmark_csv",
    "plot_lret_cirq_comparison",
    "plot_speedup_trend",
    "plot_fidelity_analysis",
    "generate_benchmark_report",
    "create_summary_table",
    "PlotConfig",
    # PennyLane Device
    "QLRETDevice",
    "create_lret_device",
    # Algorithms
    "VQE",
    "VQEResult",
    "QAOA",
    "QAOAResult",
    "QuantumNeuralNetwork",
    # Phase 7 Unified
    "LRETPhase7UnifiedAdapter",
    "Phase7Config",
    "Framework",
    "FusionMode",
    "GateFusion",
    "GateFusionStats",
    "Phase7ExecutionMetrics",
    "UnifiedExecutor",
    "FrameworkConverter",
    "create_phase7_adapter",
    # Variant Analysis
    "VariantAnalyzer",
    "VariantInfo",
    "VariantAnalysisResult",
    "VariantComparisonResult",
    "VariantCapability",
    "TaskType",
    "VARIANT_DEFINITIONS",
    "get_variant_analyzer",
    "analyze_all_variants",
    "get_best_variant_for_task",
    # Variant Registry
    "VariantRegistry",
    "VariantStatus",
    "RegisteredVariant",
    "PennyLaneHybridAdapter",
    "get_registry",
    "list_available_variants",
    "get_variant_adapter",
    "get_best_variant_adapter",
    # Backward compatibility alias
    "LRETBackendAdapter",
]

# Backward compatibility alias - LRETBackendAdapter points to Phase7UnifiedAdapter
LRETBackendAdapter = LRETPhase7UnifiedAdapter
