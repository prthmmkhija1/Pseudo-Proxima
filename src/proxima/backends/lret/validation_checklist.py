"""LRET Variants Validation Checklist.

A comprehensive validation tool for verifying all LRET variant
components are properly installed and functioning.

Usage:
    python -m proxima.backends.lret.validation_checklist
    
Or from code:
    from proxima.backends.lret.validation_checklist import run_validation
    results = run_validation()
"""

from __future__ import annotations

import asyncio
import sys
import importlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from pathlib import Path


class CheckStatus(Enum):
    """Status of a validation check."""
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    SKIPPED = "⏭️ SKIPPED"
    WARNING = "⚠️ WARNING"


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    category: str
    status: CheckStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    results: List[CheckResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    
    def add_result(self, result: CheckResult) -> None:
        """Add a check result."""
        self.results.append(result)
        status_name = result.status.name
        self.summary[status_name] = self.summary.get(status_name, 0) + 1
    
    @property
    def passed(self) -> bool:
        """Check if all required checks passed."""
        return self.summary.get('FAILED', 0) == 0
    
    def get_category_results(self, category: str) -> List[CheckResult]:
        """Get results for a specific category."""
        return [r for r in self.results if r.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passed': self.passed,
            'summary': self.summary,
            'results': [
                {
                    'name': r.name,
                    'category': r.category,
                    'status': r.status.value,
                    'message': r.message,
                }
                for r in self.results
            ],
        }


class ValidationChecker:
    """Runs validation checks for LRET variants."""
    
    def __init__(self):
        self.report = ValidationReport()
    
    def check(
        self,
        name: str,
        category: str,
        condition: bool,
        success_msg: str = "",
        failure_msg: str = "",
        **details,
    ) -> CheckResult:
        """Run a simple boolean check."""
        status = CheckStatus.PASSED if condition else CheckStatus.FAILED
        message = success_msg if condition else failure_msg
        
        result = CheckResult(
            name=name,
            category=category,
            status=status,
            message=message,
            details=details,
        )
        self.report.add_result(result)
        return result
    
    def check_import(self, module_path: str, category: str) -> CheckResult:
        """Check if a module can be imported."""
        try:
            importlib.import_module(module_path)
            return self.check(
                name=f"Import {module_path}",
                category=category,
                condition=True,
                success_msg="Module imported successfully",
            )
        except ImportError as e:
            return self.check(
                name=f"Import {module_path}",
                category=category,
                condition=False,
                failure_msg=str(e),
            )
    
    def check_class(self, module_path: str, class_name: str, category: str) -> CheckResult:
        """Check if a class exists in a module."""
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            return self.check(
                name=f"Class {class_name}",
                category=category,
                condition=cls is not None,
                success_msg=f"{class_name} found",
                failure_msg=f"{class_name} not found in {module_path}",
            )
        except ImportError:
            return self.check(
                name=f"Class {class_name}",
                category=category,
                condition=False,
                failure_msg=f"Could not import {module_path}",
            )
    
    def skip(self, name: str, category: str, reason: str = "") -> CheckResult:
        """Mark a check as skipped."""
        result = CheckResult(
            name=name,
            category=category,
            status=CheckStatus.SKIPPED,
            message=reason,
        )
        self.report.add_result(result)
        return result
    
    def warn(self, name: str, category: str, message: str = "") -> CheckResult:
        """Add a warning."""
        result = CheckResult(
            name=name,
            category=category,
            status=CheckStatus.WARNING,
            message=message,
        )
        self.report.add_result(result)
        return result


def run_installation_checks(checker: ValidationChecker) -> None:
    """Run installation-related checks."""
    category = "Installation"
    
    # Check LRET base
    checker.check_import("proxima.backends.lret", category)
    checker.check_import("proxima.backends.lret.installer", category)
    
    # Check external dependencies
    checker.check_import("cirq", category)
    
    try:
        import pennylane
        checker.check(
            "PennyLane installed",
            category,
            True,
            f"PennyLane {pennylane.__version__}",
        )
    except ImportError:
        checker.check(
            "PennyLane installed",
            category,
            False,
            failure_msg="PennyLane not installed",
        )
    
    # Check Qiskit (optional)
    try:
        import qiskit
        checker.check(
            "Qiskit installed (optional)",
            category,
            True,
            f"Qiskit {qiskit.__version__}",
        )
    except ImportError:
        checker.warn(
            "Qiskit installed (optional)",
            category,
            "Qiskit not installed - some features may be limited",
        )


def run_cirq_scalability_checks(checker: ValidationChecker) -> None:
    """Run Cirq Scalability variant checks."""
    category = "Cirq Scalability"
    
    # Check module import
    result = checker.check_import("proxima.backends.lret.cirq_scalability", category)
    
    if result.status == CheckStatus.PASSED:
        # Check adapter class
        checker.check_class(
            "proxima.backends.lret.cirq_scalability",
            "LRETCirqScalabilityAdapter",
            category,
        )
        
        # Check benchmark result class
        checker.check_class(
            "proxima.backends.lret.cirq_scalability",
            "BenchmarkResult",
            category,
        )
        
        # Check visualization
        checker.check_import("proxima.backends.lret.visualization", category)
        
        # Test basic functionality
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            adapter = LRETCirqScalabilityAdapter()
            checker.check(
                "Adapter creation",
                category,
                True,
                "LRETCirqScalabilityAdapter created successfully",
            )
        except Exception as e:
            checker.check(
                "Adapter creation",
                category,
                False,
                failure_msg=str(e),
            )
    else:
        checker.skip("Adapter creation", category, "Module not available")


def run_pennylane_hybrid_checks(checker: ValidationChecker) -> None:
    """Run PennyLane Hybrid variant checks."""
    category = "PennyLane Hybrid"
    
    # Check device module
    result = checker.check_import("proxima.backends.lret.pennylane_device", category)
    
    if result.status == CheckStatus.PASSED:
        # Check QLRETDevice
        checker.check_class(
            "proxima.backends.lret.pennylane_device",
            "QLRETDevice",
            category,
        )
        
        # Check algorithms
        alg_result = checker.check_import("proxima.backends.lret.algorithms", category)
        
        if alg_result.status == CheckStatus.PASSED:
            checker.check_class("proxima.backends.lret.algorithms", "VQE", category)
            checker.check_class("proxima.backends.lret.algorithms", "QAOA", category)
            checker.check_class("proxima.backends.lret.algorithms", "QNN", category)
        
        # Test device creation
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            dev = QLRETDevice(wires=2, shots=1024)
            checker.check(
                "QLRETDevice creation",
                category,
                True,
                "Device created with 2 wires",
            )
        except Exception as e:
            checker.check(
                "QLRETDevice creation",
                category,
                False,
                failure_msg=str(e),
            )
        
        # Test gradient computation (if PennyLane available)
        try:
            import pennylane as qml
            from pennylane import numpy as np
            
            checker.check(
                "Gradient computation support",
                category,
                True,
                "PennyLane gradient support available",
            )
        except ImportError:
            checker.skip(
                "Gradient computation support",
                category,
                "PennyLane not available",
            )
    else:
        checker.skip("QLRETDevice creation", category, "Module not available")


def run_phase7_unified_checks(checker: ValidationChecker) -> None:
    """Run Phase 7 Unified variant checks."""
    category = "Phase 7 Unified"
    
    # Check module import
    result = checker.check_import("proxima.backends.lret.phase7_unified", category)
    
    if result.status == CheckStatus.PASSED:
        # Check classes
        checker.check_class(
            "proxima.backends.lret.phase7_unified",
            "LRETPhase7UnifiedAdapter",
            category,
        )
        checker.check_class(
            "proxima.backends.lret.phase7_unified",
            "Phase7Config",
            category,
        )
        checker.check_class(
            "proxima.backends.lret.phase7_unified",
            "GateFusion",
            category,
        )
        
        # Test adapter creation
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            adapter = LRETPhase7UnifiedAdapter()
            checker.check(
                "Phase7 adapter creation",
                category,
                True,
                "LRETPhase7UnifiedAdapter created",
            )
        except Exception as e:
            checker.check(
                "Phase7 adapter creation",
                category,
                False,
                failure_msg=str(e),
            )
        
        # Check GPU availability (optional)
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            adapter = LRETPhase7UnifiedAdapter()
            gpu_available = getattr(adapter, 'check_gpu_availability', lambda: False)()
            if gpu_available:
                checker.check(
                    "GPU acceleration",
                    category,
                    True,
                    "GPU acceleration available",
                )
            else:
                checker.warn(
                    "GPU acceleration",
                    category,
                    "GPU not available - using CPU",
                )
        except Exception:
            checker.warn(
                "GPU acceleration",
                category,
                "Could not check GPU availability",
            )
    else:
        checker.skip("Phase7 adapter creation", category, "Module not available")


def run_tui_integration_checks(checker: ValidationChecker) -> None:
    """Run TUI integration checks."""
    category = "TUI Integration"
    
    # Check dialogs
    checker.check_import("proxima.tui.dialogs.lret", category)
    
    # Check specific dialogs
    dialogs_to_check = [
        ("LRETInstallerDialog", "proxima.tui.dialogs.lret"),
        ("LRETConfigDialog", "proxima.tui.dialogs.lret"),
        ("LRETBenchmarkDialog", "proxima.tui.dialogs.lret"),
        ("Phase7Dialog", "proxima.tui.dialogs.lret"),
        ("VariantAnalysisDialog", "proxima.tui.dialogs.lret"),
    ]
    
    for class_name, module_path in dialogs_to_check:
        checker.check_class(module_path, class_name, category)
    
    # Check screens
    checker.check_import("proxima.tui.screens", category)
    checker.check_class("proxima.tui.screens", "BenchmarkComparisonScreen", category)
    checker.check_class("proxima.tui.screens", "BackendsScreen", category)
    
    # Check wizards
    wizard_result = checker.check_import("proxima.tui.wizards", category)
    if wizard_result.status == CheckStatus.PASSED:
        checker.check_class("proxima.tui.wizards", "PennyLaneAlgorithmWizard", category)


def run_documentation_checks(checker: ValidationChecker) -> None:
    """Run documentation checks."""
    category = "Documentation"
    
    docs_to_check = [
        "LRET_BACKEND_VARIANTS_INTEGRATION_REPORT.md",
        "README.md",
        "docs/backends/cuquantum-usage.md",
    ]
    
    for doc in docs_to_check:
        doc_path = Path(doc)
        checker.check(
            f"Documentation: {doc}",
            category,
            doc_path.exists(),
            success_msg="File exists",
            failure_msg="File not found",
        )


def run_validation(verbose: bool = True) -> ValidationReport:
    """Run all validation checks.
    
    Args:
        verbose: Whether to print results as they run.
    
    Returns:
        Complete validation report.
    """
    checker = ValidationChecker()
    
    # Run all check categories
    check_functions = [
        ("Installation", run_installation_checks),
        ("Cirq Scalability", run_cirq_scalability_checks),
        ("PennyLane Hybrid", run_pennylane_hybrid_checks),
        ("Phase 7 Unified", run_phase7_unified_checks),
        ("TUI Integration", run_tui_integration_checks),
        ("Documentation", run_documentation_checks),
    ]
    
    for category_name, check_func in check_functions:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {category_name}")
            print('='*60)
        
        initial_count = len(checker.report.results)
        check_func(checker)
        
        if verbose:
            # Print results for this category
            for result in checker.report.results[initial_count:]:
                print(f"  {result.status.value} {result.name}")
                if result.message:
                    print(f"      {result.message}")
    
    return checker.report


def print_summary(report: ValidationReport) -> None:
    """Print validation summary."""
    print("\n" + "="*60)
    print("  VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\n  Total Checks: {len(report.results)}")
    print(f"  ✅ Passed:  {report.summary.get('PASSED', 0)}")
    print(f"  ❌ Failed:  {report.summary.get('FAILED', 0)}")
    print(f"  ⚠️  Warnings: {report.summary.get('WARNING', 0)}")
    print(f"  ⏭️  Skipped: {report.summary.get('SKIPPED', 0)}")
    
    if report.passed:
        print("\n  ✅ ALL REQUIRED CHECKS PASSED")
    else:
        print("\n  ❌ SOME CHECKS FAILED")
        print("\n  Failed checks:")
        for result in report.results:
            if result.status == CheckStatus.FAILED:
                print(f"    - {result.category}: {result.name}")
                if result.message:
                    print(f"      {result.message}")
    
    print()


def generate_checklist_markdown(report: ValidationReport) -> str:
    """Generate markdown checklist from report."""
    lines = ["# LRET Variants Validation Checklist\n"]
    
    current_category = None
    for result in report.results:
        if result.category != current_category:
            current_category = result.category
            lines.append(f"\n## {current_category}\n")
        
        if result.status == CheckStatus.PASSED:
            checkbox = "[x]"
        elif result.status == CheckStatus.FAILED:
            checkbox = "[ ]"
        else:
            checkbox = "[-]"
        
        lines.append(f"- {checkbox} {result.name}")
        if result.message:
            lines.append(f"  - {result.message}")
    
    return "\n".join(lines)


def main():
    """Run validation and print results."""
    print("\n" + "="*60)
    print("  LRET VARIANTS VALIDATION CHECKLIST")
    print("="*60)
    
    report = run_validation(verbose=True)
    print_summary(report)
    
    # Save checklist to file
    checklist_path = Path("LRET_VALIDATION_CHECKLIST.md")
    checklist_md = generate_checklist_markdown(report)
    checklist_path.write_text(checklist_md)
    print(f"Checklist saved to: {checklist_path}")
    
    # Return exit code
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
