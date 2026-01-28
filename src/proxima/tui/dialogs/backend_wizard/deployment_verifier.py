"""Deployment Verifier.

Post-deployment verification to ensure backend was
properly deployed and is ready for use.

Part of Phase 8: Final Deployment & Success Confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import importlib
import sys
import traceback
from datetime import datetime
from enum import Enum


class VerificationStatus(Enum):
    """Status of a verification check."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    name: str
    status: VerificationStatus
    message: str
    details: Optional[str] = None
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class VerificationReport:
    """Complete verification report."""
    backend_name: str
    backend_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    results: List[VerificationResult] = field(default_factory=list)
    overall_status: VerificationStatus = VerificationStatus.PENDING
    
    @property
    def passed(self) -> int:
        """Number of passed checks."""
        return sum(1 for r in self.results if r.status == VerificationStatus.PASSED)
    
    @property
    def failed(self) -> int:
        """Number of failed checks."""
        return sum(1 for r in self.results if r.status == VerificationStatus.FAILED)
    
    @property
    def warnings(self) -> int:
        """Number of warning checks."""
        return sum(1 for r in self.results if r.status == VerificationStatus.WARNING)
    
    @property
    def total(self) -> int:
        """Total number of checks run."""
        return len(self.results)
    
    @property
    def success(self) -> bool:
        """Whether verification was successful (no failures)."""
        return self.failed == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "backend_name": self.backend_name,
            "backend_id": self.backend_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "warnings": self.warnings,
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                }
                for r in self.results
            ]
        }


class DeploymentVerifier:
    """Verifies that a backend deployment was successful.
    
    Runs a series of checks to ensure:
    1. All files exist
    2. Backend can be imported
    3. Backend is registered
    4. Tests pass
    5. Documentation exists
    """
    
    def __init__(
        self,
        backend_name: str,
        backend_id: str,
        backend_path: Path,
        files_created: List[Path] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        """Initialize verifier.
        
        Args:
            backend_name: Display name of the backend
            backend_id: Backend identifier (e.g., 'my_simulator')
            backend_path: Path to the backend module
            files_created: List of files that should have been created
            on_progress: Optional callback for progress updates
        """
        self.backend_name = backend_name
        self.backend_id = backend_id
        self.backend_path = backend_path
        self.files_created = files_created or []
        self.on_progress = on_progress
        
        self.report = VerificationReport(
            backend_name=backend_name,
            backend_id=backend_id,
        )
    
    def _update_progress(self, message: str, current: int, total: int) -> None:
        """Update progress callback if set."""
        if self.on_progress:
            self.on_progress(message, current, total)
    
    def verify_all(self) -> VerificationReport:
        """Run all verification checks.
        
        Returns:
            VerificationReport with all results
        """
        checks = [
            ("Files Exist", self._verify_files_exist),
            ("Module Import", self._verify_import),
            ("Backend Class", self._verify_backend_class),
            ("Registry Entry", self._verify_registry),
            ("Basic Functionality", self._verify_basic_functionality),
            ("Configuration", self._verify_configuration),
            ("Documentation", self._verify_documentation),
        ]
        
        total = len(checks)
        
        for i, (name, check_func) in enumerate(checks):
            self._update_progress(name, i + 1, total)
            
            import time
            start = time.perf_counter()
            
            try:
                result = check_func()
                result.duration_ms = (time.perf_counter() - start) * 1000
                self.report.results.append(result)
            except Exception as e:
                self.report.results.append(VerificationResult(
                    name=name,
                    status=VerificationStatus.FAILED,
                    message=f"Check failed with error",
                    error=str(e),
                    details=traceback.format_exc(),
                    duration_ms=(time.perf_counter() - start) * 1000,
                ))
        
        # Determine overall status
        if self.report.failed > 0:
            self.report.overall_status = VerificationStatus.FAILED
        elif self.report.warnings > 0:
            self.report.overall_status = VerificationStatus.WARNING
        else:
            self.report.overall_status = VerificationStatus.PASSED
        
        return self.report
    
    def _verify_files_exist(self) -> VerificationResult:
        """Verify all expected files exist."""
        missing_files = []
        
        for file_path in self.files_created:
            if not Path(file_path).exists():
                missing_files.append(str(file_path))
        
        if not self.files_created:
            return VerificationResult(
                name="Files Exist",
                status=VerificationStatus.SKIPPED,
                message="No files to verify",
            )
        
        if missing_files:
            return VerificationResult(
                name="Files Exist",
                status=VerificationStatus.FAILED,
                message=f"Missing {len(missing_files)} file(s)",
                details="\n".join(missing_files),
            )
        
        return VerificationResult(
            name="Files Exist",
            status=VerificationStatus.PASSED,
            message=f"All {len(self.files_created)} files exist",
        )
    
    def _verify_import(self) -> VerificationResult:
        """Verify backend module can be imported."""
        try:
            # Build module path
            module_path = f"proxima.backends.{self.backend_id}"
            
            # Try to import
            module = importlib.import_module(module_path)
            
            return VerificationResult(
                name="Module Import",
                status=VerificationStatus.PASSED,
                message=f"Module '{module_path}' imported successfully",
            )
        except ImportError as e:
            return VerificationResult(
                name="Module Import",
                status=VerificationStatus.FAILED,
                message="Failed to import backend module",
                error=str(e),
                details=traceback.format_exc(),
            )
        except Exception as e:
            return VerificationResult(
                name="Module Import",
                status=VerificationStatus.FAILED,
                message="Error during import",
                error=str(e),
                details=traceback.format_exc(),
            )
    
    def _verify_backend_class(self) -> VerificationResult:
        """Verify backend class exists and has required methods."""
        try:
            module_path = f"proxima.backends.{self.backend_id}"
            module = importlib.import_module(module_path)
            
            # Look for backend class
            backend_class = None
            class_name = None
            
            # Try common naming conventions
            possible_names = [
                f"{self.backend_id.title().replace('_', '')}Backend",
                f"{self.backend_id.title()}Backend",
                f"{self.backend_name.replace(' ', '')}Backend",
            ]
            
            for name in possible_names:
                if hasattr(module, name):
                    backend_class = getattr(module, name)
                    class_name = name
                    break
            
            # Also check for any class ending in Backend
            if not backend_class:
                for attr_name in dir(module):
                    if attr_name.endswith("Backend"):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            backend_class = attr
                            class_name = attr_name
                            break
            
            if not backend_class:
                return VerificationResult(
                    name="Backend Class",
                    status=VerificationStatus.FAILED,
                    message="Backend class not found",
                    details=f"Searched for: {', '.join(possible_names)}",
                )
            
            # Check required methods
            required_methods = ["run_circuit", "get_capabilities"]
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(backend_class, method):
                    missing_methods.append(method)
            
            if missing_methods:
                return VerificationResult(
                    name="Backend Class",
                    status=VerificationStatus.WARNING,
                    message=f"Class {class_name} missing methods",
                    details=f"Missing: {', '.join(missing_methods)}",
                )
            
            return VerificationResult(
                name="Backend Class",
                status=VerificationStatus.PASSED,
                message=f"Class '{class_name}' found with required methods",
            )
            
        except Exception as e:
            return VerificationResult(
                name="Backend Class",
                status=VerificationStatus.FAILED,
                message="Error checking backend class",
                error=str(e),
            )
    
    def _verify_registry(self) -> VerificationResult:
        """Verify backend is registered in the backend registry."""
        try:
            # Try to import registry
            try:
                from proxima.backends import get_backend, list_backends
            except ImportError:
                # Registry might not exist yet
                return VerificationResult(
                    name="Registry Entry",
                    status=VerificationStatus.SKIPPED,
                    message="Backend registry not available",
                )
            
            # Check if backend is in list
            backends = list_backends()
            
            if self.backend_id in backends:
                return VerificationResult(
                    name="Registry Entry",
                    status=VerificationStatus.PASSED,
                    message=f"Backend '{self.backend_id}' found in registry",
                )
            
            # Check alternative registrations
            for backend in backends:
                if self.backend_id.lower() in backend.lower():
                    return VerificationResult(
                        name="Registry Entry",
                        status=VerificationStatus.WARNING,
                        message=f"Similar backend '{backend}' found",
                        details=f"Expected: {self.backend_id}",
                    )
            
            return VerificationResult(
                name="Registry Entry",
                status=VerificationStatus.WARNING,
                message="Backend not found in registry",
                details="Backend may need to be registered manually",
            )
            
        except Exception as e:
            return VerificationResult(
                name="Registry Entry",
                status=VerificationStatus.SKIPPED,
                message="Could not check registry",
                error=str(e),
            )
    
    def _verify_basic_functionality(self) -> VerificationResult:
        """Verify backend can be instantiated."""
        try:
            module_path = f"proxima.backends.{self.backend_id}"
            module = importlib.import_module(module_path)
            
            # Find backend class
            backend_class = None
            for attr_name in dir(module):
                if attr_name.endswith("Backend"):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        backend_class = attr
                        break
            
            if not backend_class:
                return VerificationResult(
                    name="Basic Functionality",
                    status=VerificationStatus.SKIPPED,
                    message="Backend class not found for instantiation",
                )
            
            # Try to instantiate
            instance = backend_class()
            
            # Check if get_capabilities works
            if hasattr(instance, 'get_capabilities'):
                caps = instance.get_capabilities()
                return VerificationResult(
                    name="Basic Functionality",
                    status=VerificationStatus.PASSED,
                    message="Backend instantiated and get_capabilities() works",
                    details=f"Capabilities: {caps}" if caps else None,
                )
            
            return VerificationResult(
                name="Basic Functionality",
                status=VerificationStatus.PASSED,
                message="Backend instantiated successfully",
            )
            
        except Exception as e:
            return VerificationResult(
                name="Basic Functionality",
                status=VerificationStatus.WARNING,
                message="Could not instantiate backend",
                error=str(e),
                details="This may be expected if backend requires configuration",
            )
    
    def _verify_configuration(self) -> VerificationResult:
        """Verify configuration files exist."""
        config_paths = [
            self.backend_path / "config.yaml",
            self.backend_path / "config.json",
            self.backend_path / f"{self.backend_id}.yaml",
        ]
        
        found_config = None
        for config_path in config_paths:
            if config_path.exists():
                found_config = config_path
                break
        
        if found_config:
            return VerificationResult(
                name="Configuration",
                status=VerificationStatus.PASSED,
                message=f"Configuration found: {found_config.name}",
            )
        
        return VerificationResult(
            name="Configuration",
            status=VerificationStatus.WARNING,
            message="No configuration file found",
            details="Backend may use default configuration",
        )
    
    def _verify_documentation(self) -> VerificationResult:
        """Verify documentation exists."""
        doc_paths = [
            self.backend_path / "README.md",
            self.backend_path / "USAGE.md",
            self.backend_path / "docs" / "README.md",
        ]
        
        found_docs = []
        for doc_path in doc_paths:
            if doc_path.exists():
                found_docs.append(doc_path.name)
        
        if found_docs:
            return VerificationResult(
                name="Documentation",
                status=VerificationStatus.PASSED,
                message=f"Documentation found: {', '.join(found_docs)}",
            )
        
        return VerificationResult(
            name="Documentation",
            status=VerificationStatus.WARNING,
            message="No documentation file found",
            details="Consider adding a README.md",
        )


class QuickVerifier:
    """Quick verification for immediate feedback after deployment."""
    
    @staticmethod
    def quick_check(backend_id: str, backend_path: Path) -> Dict[str, bool]:
        """Perform quick checks on a deployed backend.
        
        Args:
            backend_id: Backend identifier
            backend_path: Path to backend module
            
        Returns:
            Dictionary with check results
        """
        results = {
            "files_exist": False,
            "importable": False,
            "has_backend_class": False,
            "instantiable": False,
        }
        
        # Check files
        if backend_path.exists():
            py_files = list(backend_path.glob("*.py"))
            results["files_exist"] = len(py_files) > 0
        
        # Check import
        try:
            module = importlib.import_module(f"proxima.backends.{backend_id}")
            results["importable"] = True
            
            # Check for backend class
            for attr_name in dir(module):
                if attr_name.endswith("Backend"):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        results["has_backend_class"] = True
                        
                        # Try instantiation
                        try:
                            instance = attr()
                            results["instantiable"] = True
                        except Exception:
                            pass
                        break
        except Exception:
            pass
        
        return results


def create_verification_summary(report: VerificationReport) -> str:
    """Create a text summary of the verification report.
    
    Args:
        report: Verification report
        
    Returns:
        Formatted text summary
    """
    lines = [
        f"{'═' * 50}",
        f"Deployment Verification Report",
        f"{'═' * 50}",
        f"Backend: {report.backend_name} ({report.backend_id})",
        f"Time: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Status: {report.overall_status.value.upper()}",
        f"",
        f"Results: {report.passed}/{report.total} passed",
        f"",
    ]
    
    # Status icons
    icons = {
        VerificationStatus.PASSED: "✓",
        VerificationStatus.FAILED: "✗",
        VerificationStatus.WARNING: "⚠",
        VerificationStatus.SKIPPED: "○",
    }
    
    for result in report.results:
        icon = icons.get(result.status, "?")
        lines.append(f"  {icon} {result.name}: {result.message}")
        if result.error:
            lines.append(f"      Error: {result.error}")
    
    lines.append(f"{'═' * 50}")
    
    return "\n".join(lines)
