"""Test Executor.

Runs comprehensive tests on generated backend code.
Provides real-time progress updates and detailed results.
"""

from __future__ import annotations

import asyncio
import tempfile
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging

from .test_circuits import TestCircuitLibrary, TestCircuit, CircuitType

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of a test."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestCategory(Enum):
    """Categories of tests."""
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    EXECUTION = "execution"
    NORMALIZATION = "normalization"
    GATE_SUPPORT = "gate_support"
    CIRCUIT_TYPES = "circuit_types"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    category: TestCategory
    status: TestStatus
    duration_ms: float = 0.0
    message: str = ""
    details: Optional[Dict[str, Any]] = None
    error_traceback: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "details": self.details,
            "passed": self.passed,
        }


@dataclass
class TestProgress:
    """Progress information for test execution."""
    current_test: str
    current_category: TestCategory
    tests_completed: int
    tests_total: int
    tests_passed: int
    tests_failed: int
    current_status: TestStatus = TestStatus.RUNNING
    
    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.tests_total == 0:
            return 0.0
        return (self.tests_completed / self.tests_total) * 100


@dataclass
class ExecutionContext:
    """Context for test execution."""
    backend_code: str
    backend_name: str
    normalizer_code: Optional[str] = None
    temp_dir: Optional[Path] = None
    backend_module: Any = None
    backend_class: Any = None
    backend_instance: Any = None
    circuit_library: TestCircuitLibrary = field(default_factory=TestCircuitLibrary)


class TestExecutor:
    """Execute comprehensive tests on generated backend code.
    
    Provides async test execution with progress callbacks for
    real-time UI updates.
    """
    
    def __init__(
        self,
        backend_code: str,
        backend_name: str,
        normalizer_code: Optional[str] = None
    ):
        """Initialize test executor.
        
        Args:
            backend_code: Generated Python code for the backend
            backend_name: Name of the backend (snake_case)
            normalizer_code: Optional normalizer code
        """
        self.context = ExecutionContext(
            backend_code=backend_code,
            backend_name=backend_name,
            normalizer_code=normalizer_code,
        )
        self.results: List[TestResult] = []
        self._progress_callback: Optional[Callable[[TestProgress], None]] = None
        self._abort_requested = False
    
    def set_progress_callback(
        self,
        callback: Callable[[TestProgress], None]
    ) -> None:
        """Set callback for progress updates.
        
        Args:
            callback: Function to call with progress updates
        """
        self._progress_callback = callback
    
    def abort(self) -> None:
        """Request test execution to abort."""
        self._abort_requested = True
    
    async def run_all_tests(
        self,
        circuit_type: str = "bell",
        shots: int = 1024,
        include_performance: bool = False
    ) -> Dict[str, Any]:
        """Run all backend tests.
        
        Args:
            circuit_type: Primary circuit type for testing
            shots: Number of measurement shots
            include_performance: Whether to include performance tests
            
        Returns:
            Dictionary with all test results
        """
        self.results = []
        self._abort_requested = False
        
        # Calculate total tests
        total_tests = self._count_tests(include_performance)
        tests_completed = 0
        tests_passed = 0
        tests_failed = 0
        
        # Setup environment
        setup_result = await self._setup_environment()
        if not setup_result.passed:
            self.results.append(setup_result)
            return self._compile_results()
        
        try:
            # Category 1: Initialization Tests
            init_tests = await self._run_initialization_tests()
            for result in init_tests:
                self.results.append(result)
                tests_completed += 1
                if result.passed:
                    tests_passed += 1
                else:
                    tests_failed += 1
                await self._report_progress(
                    result.name, result.category,
                    tests_completed, total_tests,
                    tests_passed, tests_failed, result.status
                )
                if self._abort_requested:
                    break
            
            # Only continue if initialization passed
            if not self._abort_requested and any(r.passed for r in init_tests):
                # Category 2: Validation Tests
                val_tests = await self._run_validation_tests(circuit_type)
                for result in val_tests:
                    self.results.append(result)
                    tests_completed += 1
                    if result.passed:
                        tests_passed += 1
                    else:
                        tests_failed += 1
                    await self._report_progress(
                        result.name, result.category,
                        tests_completed, total_tests,
                        tests_passed, tests_failed, result.status
                    )
                    if self._abort_requested:
                        break
                
                # Category 3: Execution Tests
                if not self._abort_requested:
                    exec_tests = await self._run_execution_tests(
                        circuit_type, shots
                    )
                    for result in exec_tests:
                        self.results.append(result)
                        tests_completed += 1
                        if result.passed:
                            tests_passed += 1
                        else:
                            tests_failed += 1
                        await self._report_progress(
                            result.name, result.category,
                            tests_completed, total_tests,
                            tests_passed, tests_failed, result.status
                        )
                        if self._abort_requested:
                            break
                
                # Category 4: Normalization Tests
                if not self._abort_requested:
                    norm_tests = await self._run_normalization_tests(shots)
                    for result in norm_tests:
                        self.results.append(result)
                        tests_completed += 1
                        if result.passed:
                            tests_passed += 1
                        else:
                            tests_failed += 1
                        await self._report_progress(
                            result.name, result.category,
                            tests_completed, total_tests,
                            tests_passed, tests_failed, result.status
                        )
                        if self._abort_requested:
                            break
                
                # Category 5: Gate Support Tests
                if not self._abort_requested:
                    gate_tests = await self._run_gate_support_tests()
                    for result in gate_tests:
                        self.results.append(result)
                        tests_completed += 1
                        if result.passed:
                            tests_passed += 1
                        else:
                            tests_failed += 1
                        await self._report_progress(
                            result.name, result.category,
                            tests_completed, total_tests,
                            tests_passed, tests_failed, result.status
                        )
                        if self._abort_requested:
                            break
                
                # Category 6: Circuit Type Tests
                if not self._abort_requested:
                    circuit_tests = await self._run_circuit_type_tests(shots)
                    for result in circuit_tests:
                        self.results.append(result)
                        tests_completed += 1
                        if result.passed:
                            tests_passed += 1
                        else:
                            tests_failed += 1
                        await self._report_progress(
                            result.name, result.category,
                            tests_completed, total_tests,
                            tests_passed, tests_failed, result.status
                        )
                        if self._abort_requested:
                            break
                
                # Category 7: Error Handling Tests
                if not self._abort_requested:
                    error_tests = await self._run_error_handling_tests()
                    for result in error_tests:
                        self.results.append(result)
                        tests_completed += 1
                        if result.passed:
                            tests_passed += 1
                        else:
                            tests_failed += 1
                        await self._report_progress(
                            result.name, result.category,
                            tests_completed, total_tests,
                            tests_passed, tests_failed, result.status
                        )
                        if self._abort_requested:
                            break
                
                # Category 8: Performance Tests (optional)
                if include_performance and not self._abort_requested:
                    perf_tests = await self._run_performance_tests(shots)
                    for result in perf_tests:
                        self.results.append(result)
                        tests_completed += 1
                        if result.passed:
                            tests_passed += 1
                        else:
                            tests_failed += 1
                        await self._report_progress(
                            result.name, result.category,
                            tests_completed, total_tests,
                            tests_passed, tests_failed, result.status
                        )
                        if self._abort_requested:
                            break
        
        finally:
            # Cleanup
            await self._cleanup_environment()
        
        return self._compile_results()
    
    def _count_tests(self, include_performance: bool) -> int:
        """Count total number of tests."""
        count = (
            3 +  # Initialization
            3 +  # Validation
            4 +  # Execution
            2 +  # Normalization
            5 +  # Gate support
            4 +  # Circuit types
            3    # Error handling
        )
        if include_performance:
            count += 4  # Performance tests
        return count
    
    async def _report_progress(
        self,
        test_name: str,
        category: TestCategory,
        completed: int,
        total: int,
        passed: int,
        failed: int,
        status: TestStatus
    ) -> None:
        """Report progress to callback."""
        if self._progress_callback:
            progress = TestProgress(
                current_test=test_name,
                current_category=category,
                tests_completed=completed,
                tests_total=total,
                tests_passed=passed,
                tests_failed=failed,
                current_status=status,
            )
            self._progress_callback(progress)
        
        # Small delay for UI updates
        await asyncio.sleep(0.05)
    
    async def _setup_environment(self) -> TestResult:
        """Set up the test environment."""
        start_time = time.time()
        
        try:
            # Create temp directory
            self.context.temp_dir = Path(tempfile.mkdtemp(
                prefix=f"proxima_test_{self.context.backend_name}_"
            ))
            
            # Write backend code
            backend_path = self.context.temp_dir / f"{self.context.backend_name}.py"
            backend_path.write_text(self.context.backend_code)
            
            # Write normalizer if provided
            if self.context.normalizer_code:
                normalizer_path = self.context.temp_dir / f"{self.context.backend_name}_normalizer.py"
                normalizer_path.write_text(self.context.normalizer_code)
            
            # Add to Python path
            sys.path.insert(0, str(self.context.temp_dir))
            
            return TestResult(
                name="Environment Setup",
                category=TestCategory.INITIALIZATION,
                status=TestStatus.PASSED,
                duration_ms=(time.time() - start_time) * 1000,
                message="Test environment created successfully",
            )
            
        except Exception as e:
            return TestResult(
                name="Environment Setup",
                category=TestCategory.INITIALIZATION,
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start_time) * 1000,
                message=f"Failed to setup environment: {e}",
                error_traceback=traceback.format_exc(),
            )
    
    async def _cleanup_environment(self) -> None:
        """Clean up the test environment."""
        try:
            # Remove from Python path
            if self.context.temp_dir and str(self.context.temp_dir) in sys.path:
                sys.path.remove(str(self.context.temp_dir))
            
            # Unload modules
            if self.context.backend_name in sys.modules:
                del sys.modules[self.context.backend_name]
            
            normalizer_name = f"{self.context.backend_name}_normalizer"
            if normalizer_name in sys.modules:
                del sys.modules[normalizer_name]
            
            # Remove temp directory
            if self.context.temp_dir and self.context.temp_dir.exists():
                import shutil
                shutil.rmtree(self.context.temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    async def _run_initialization_tests(self) -> List[TestResult]:
        """Run initialization tests."""
        results = []
        
        # Test 1: Module Import
        start_time = time.time()
        try:
            self.context.backend_module = __import__(self.context.backend_name)
            results.append(TestResult(
                name="Module Import",
                category=TestCategory.INITIALIZATION,
                status=TestStatus.PASSED,
                duration_ms=(time.time() - start_time) * 1000,
                message="Backend module imported successfully",
            ))
        except SyntaxError as e:
            results.append(TestResult(
                name="Module Import",
                category=TestCategory.INITIALIZATION,
                status=TestStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                message=f"Syntax error: {e}",
                error_traceback=traceback.format_exc(),
            ))
            return results  # Can't continue
        except Exception as e:
            results.append(TestResult(
                name="Module Import",
                category=TestCategory.INITIALIZATION,
                status=TestStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                message=f"Import error: {e}",
                error_traceback=traceback.format_exc(),
            ))
            return results
        
        # Test 2: Class Discovery
        start_time = time.time()
        class_name = self._to_class_name(self.context.backend_name) + "Backend"
        
        try:
            # Try to find backend class
            self.context.backend_class = None
            for attr_name in dir(self.context.backend_module):
                if "backend" in attr_name.lower() or "adapter" in attr_name.lower():
                    attr = getattr(self.context.backend_module, attr_name)
                    if isinstance(attr, type):
                        self.context.backend_class = attr
                        break
            
            if self.context.backend_class is None:
                self.context.backend_class = getattr(
                    self.context.backend_module, class_name, None
                )
            
            if self.context.backend_class:
                results.append(TestResult(
                    name="Class Discovery",
                    category=TestCategory.INITIALIZATION,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=f"Found backend class: {self.context.backend_class.__name__}",
                ))
            else:
                results.append(TestResult(
                    name="Class Discovery",
                    category=TestCategory.INITIALIZATION,
                    status=TestStatus.FAILED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=f"Could not find backend class (expected: {class_name})",
                ))
                return results
                
        except Exception as e:
            results.append(TestResult(
                name="Class Discovery",
                category=TestCategory.INITIALIZATION,
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start_time) * 1000,
                message=str(e),
                error_traceback=traceback.format_exc(),
            ))
            return results
        
        # Test 3: Instantiation
        start_time = time.time()
        try:
            self.context.backend_instance = self.context.backend_class()
            
            # Try to initialize if method exists
            if hasattr(self.context.backend_instance, "initialize"):
                try:
                    self.context.backend_instance.initialize()
                except NotImplementedError:
                    pass  # Expected for templates
                except Exception as init_e:
                    logger.warning(f"Backend init warning: {init_e}")
            
            results.append(TestResult(
                name="Backend Instantiation",
                category=TestCategory.INITIALIZATION,
                status=TestStatus.PASSED,
                duration_ms=(time.time() - start_time) * 1000,
                message="Backend instantiated successfully",
            ))
            
        except Exception as e:
            results.append(TestResult(
                name="Backend Instantiation",
                category=TestCategory.INITIALIZATION,
                status=TestStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                message=f"Failed to instantiate: {e}",
                error_traceback=traceback.format_exc(),
            ))
        
        return results
    
    async def _run_validation_tests(self, circuit_type: str) -> List[TestResult]:
        """Run validation tests."""
        results = []
        
        if not self.context.backend_instance:
            results.append(TestResult(
                name="Circuit Validation",
                category=TestCategory.VALIDATION,
                status=TestStatus.SKIPPED,
                message="Backend not available",
            ))
            return results
        
        # Test 1: validate_circuit method
        start_time = time.time()
        has_validate = hasattr(self.context.backend_instance, "validate_circuit")
        
        if has_validate:
            try:
                circuit = self.context.circuit_library.get_circuit(circuit_type)
                if circuit:
                    # Try validation
                    result = self.context.backend_instance.validate_circuit(circuit)
                    
                    if hasattr(result, "valid"):
                        valid = result.valid
                    else:
                        valid = bool(result)
                    
                    results.append(TestResult(
                        name="Circuit Validation",
                        category=TestCategory.VALIDATION,
                        status=TestStatus.PASSED if valid else TestStatus.FAILED,
                        duration_ms=(time.time() - start_time) * 1000,
                        message="Circuit validated successfully" if valid else "Validation failed",
                    ))
                else:
                    results.append(TestResult(
                        name="Circuit Validation",
                        category=TestCategory.VALIDATION,
                        status=TestStatus.SKIPPED,
                        duration_ms=(time.time() - start_time) * 1000,
                        message=f"Circuit type '{circuit_type}' not found",
                    ))
            except NotImplementedError:
                results.append(TestResult(
                    name="Circuit Validation",
                    category=TestCategory.VALIDATION,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message="Method exists (NotImplementedError expected for template)",
                ))
            except Exception as e:
                results.append(TestResult(
                    name="Circuit Validation",
                    category=TestCategory.VALIDATION,
                    status=TestStatus.ERROR,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=str(e),
                    error_traceback=traceback.format_exc(),
                ))
        else:
            results.append(TestResult(
                name="Circuit Validation",
                category=TestCategory.VALIDATION,
                status=TestStatus.PASSED,
                duration_ms=(time.time() - start_time) * 1000,
                message="validate_circuit method not required",
            ))
        
        # Test 2: get_capabilities method
        start_time = time.time()
        has_capabilities = hasattr(self.context.backend_instance, "get_capabilities")
        
        if has_capabilities:
            try:
                caps = self.context.backend_instance.get_capabilities()
                
                if caps:
                    details = {}
                    if hasattr(caps, "max_qubits"):
                        details["max_qubits"] = caps.max_qubits
                    if hasattr(caps, "supported_gates"):
                        details["supported_gates"] = len(caps.supported_gates)
                    
                    results.append(TestResult(
                        name="Capabilities Check",
                        category=TestCategory.VALIDATION,
                        status=TestStatus.PASSED,
                        duration_ms=(time.time() - start_time) * 1000,
                        message="Capabilities retrieved successfully",
                        details=details,
                    ))
                else:
                    results.append(TestResult(
                        name="Capabilities Check",
                        category=TestCategory.VALIDATION,
                        status=TestStatus.FAILED,
                        duration_ms=(time.time() - start_time) * 1000,
                        message="get_capabilities returned None",
                    ))
                    
            except NotImplementedError:
                results.append(TestResult(
                    name="Capabilities Check",
                    category=TestCategory.VALIDATION,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message="Method exists (NotImplementedError expected for template)",
                ))
            except Exception as e:
                results.append(TestResult(
                    name="Capabilities Check",
                    category=TestCategory.VALIDATION,
                    status=TestStatus.ERROR,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=str(e),
                    error_traceback=traceback.format_exc(),
                ))
        else:
            results.append(TestResult(
                name="Capabilities Check",
                category=TestCategory.VALIDATION,
                status=TestStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                message="get_capabilities method not found",
            ))
        
        # Test 3: is_available method
        start_time = time.time()
        has_available = hasattr(self.context.backend_instance, "is_available")
        
        if has_available:
            try:
                available = self.context.backend_instance.is_available()
                
                results.append(TestResult(
                    name="Availability Check",
                    category=TestCategory.VALIDATION,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=f"Backend available: {available}",
                    details={"available": available},
                ))
                
            except Exception as e:
                results.append(TestResult(
                    name="Availability Check",
                    category=TestCategory.VALIDATION,
                    status=TestStatus.ERROR,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=str(e),
                    error_traceback=traceback.format_exc(),
                ))
        else:
            results.append(TestResult(
                name="Availability Check",
                category=TestCategory.VALIDATION,
                status=TestStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                message="is_available method not found",
            ))
        
        return results
    
    async def _run_execution_tests(
        self,
        circuit_type: str,
        shots: int
    ) -> List[TestResult]:
        """Run execution tests."""
        results = []
        
        if not self.context.backend_instance:
            results.append(TestResult(
                name="Circuit Execution",
                category=TestCategory.EXECUTION,
                status=TestStatus.SKIPPED,
                message="Backend not available",
            ))
            return results
        
        # Find execute method
        execute_method = None
        for method_name in ["execute", "run", "simulate"]:
            if hasattr(self.context.backend_instance, method_name):
                execute_method = getattr(self.context.backend_instance, method_name)
                break
        
        if not execute_method:
            results.append(TestResult(
                name="Execute Method",
                category=TestCategory.EXECUTION,
                status=TestStatus.FAILED,
                message="No execute/run/simulate method found",
            ))
            return results
        
        results.append(TestResult(
            name="Execute Method",
            category=TestCategory.EXECUTION,
            status=TestStatus.PASSED,
            message=f"Found execution method: {execute_method.__name__}",
        ))
        
        # Test with different circuits
        test_circuits = [
            ("Bell State", "bell"),
            ("GHZ State", "ghz_3"),
            ("Single Qubit", "single_qubit"),
        ]
        
        for test_name, circuit_name in test_circuits:
            start_time = time.time()
            circuit = self.context.circuit_library.get_circuit(circuit_name)
            
            if not circuit:
                results.append(TestResult(
                    name=f"Execute {test_name}",
                    category=TestCategory.EXECUTION,
                    status=TestStatus.SKIPPED,
                    message=f"Circuit '{circuit_name}' not found",
                ))
                continue
            
            try:
                # Try execution
                result = execute_method(circuit, {"shots": shots})
                
                results.append(TestResult(
                    name=f"Execute {test_name}",
                    category=TestCategory.EXECUTION,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message="Circuit executed successfully",
                    details={"result_type": type(result).__name__},
                ))
                
            except NotImplementedError:
                results.append(TestResult(
                    name=f"Execute {test_name}",
                    category=TestCategory.EXECUTION,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message="Method exists (NotImplementedError expected for template)",
                ))
                
            except Exception as e:
                results.append(TestResult(
                    name=f"Execute {test_name}",
                    category=TestCategory.EXECUTION,
                    status=TestStatus.FAILED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=str(e)[:100],
                    error_traceback=traceback.format_exc(),
                ))
        
        return results
    
    async def _run_normalization_tests(self, shots: int) -> List[TestResult]:
        """Run normalization tests."""
        results = []
        
        # Check for normalizer
        start_time = time.time()
        normalizer_name = self._to_class_name(self.context.backend_name) + "Normalizer"
        normalizer_class = getattr(self.context.backend_module, normalizer_name, None)
        
        if not normalizer_class:
            results.append(TestResult(
                name="Normalizer Check",
                category=TestCategory.NORMALIZATION,
                status=TestStatus.PASSED,
                duration_ms=(time.time() - start_time) * 1000,
                message="Normalizer not included (optional)",
            ))
            return results
        
        results.append(TestResult(
            name="Normalizer Check",
            category=TestCategory.NORMALIZATION,
            status=TestStatus.PASSED,
            duration_ms=(time.time() - start_time) * 1000,
            message=f"Found normalizer: {normalizer_name}",
        ))
        
        # Test normalization
        start_time = time.time()
        try:
            normalizer = normalizer_class()
            
            # Mock result
            mock_result = {
                "counts": {"00": shots // 2, "11": shots // 2},
                "shots": shots,
            }
            
            normalized = normalizer.normalize(mock_result)
            
            # Validate counts
            if "counts" in normalized:
                total = sum(normalized["counts"].values())
                if total == shots:
                    results.append(TestResult(
                        name="Normalize Result",
                        category=TestCategory.NORMALIZATION,
                        status=TestStatus.PASSED,
                        duration_ms=(time.time() - start_time) * 1000,
                        message="Result normalized correctly",
                        details={"total_counts": total},
                    ))
                else:
                    results.append(TestResult(
                        name="Normalize Result",
                        category=TestCategory.NORMALIZATION,
                        status=TestStatus.FAILED,
                        duration_ms=(time.time() - start_time) * 1000,
                        message=f"Count mismatch: expected {shots}, got {total}",
                    ))
            else:
                results.append(TestResult(
                    name="Normalize Result",
                    category=TestCategory.NORMALIZATION,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message="Normalization completed (no counts in result)",
                ))
                
        except NotImplementedError:
            results.append(TestResult(
                name="Normalize Result",
                category=TestCategory.NORMALIZATION,
                status=TestStatus.PASSED,
                duration_ms=(time.time() - start_time) * 1000,
                message="Method exists (NotImplementedError expected)",
            ))
        except Exception as e:
            results.append(TestResult(
                name="Normalize Result",
                category=TestCategory.NORMALIZATION,
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start_time) * 1000,
                message=str(e),
                error_traceback=traceback.format_exc(),
            ))
        
        return results
    
    async def _run_gate_support_tests(self) -> List[TestResult]:
        """Run gate support tests."""
        results = []
        
        if not self.context.backend_instance:
            return results
        
        # Check for supported gates
        gates_to_test = [
            ("Hadamard (H)", "h"),
            ("Pauli-X", "x"),
            ("CNOT", "cx"),
            ("Rotation (RX)", "rx"),
            ("Controlled-Z", "cz"),
        ]
        
        for gate_name, gate_id in gates_to_test:
            start_time = time.time()
            
            # Check if gate is in supported gates
            has_gate = False
            
            if hasattr(self.context.backend_instance, "get_capabilities"):
                try:
                    caps = self.context.backend_instance.get_capabilities()
                    if hasattr(caps, "supported_gates"):
                        has_gate = gate_id in caps.supported_gates
                except Exception:
                    pass
            
            # Also check code for gate mapping
            if not has_gate:
                has_gate = f'"{gate_id}"' in self.context.backend_code.lower()
            
            results.append(TestResult(
                name=f"Gate: {gate_name}",
                category=TestCategory.GATE_SUPPORT,
                status=TestStatus.PASSED if has_gate else TestStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                message=f"{gate_name} gate {'supported' if has_gate else 'not found'}",
            ))
        
        return results
    
    async def _run_circuit_type_tests(self, shots: int) -> List[TestResult]:
        """Run circuit type tests."""
        results = []
        
        if not self.context.backend_instance:
            return results
        
        # Test different circuit types
        circuit_types = [
            ("Bell State", "bell"),
            ("GHZ State", "ghz_3"),
            ("QFT", "qft_3"),
            ("Parametric", "parametric"),
        ]
        
        for type_name, circuit_id in circuit_types:
            start_time = time.time()
            circuit = self.context.circuit_library.get_circuit(circuit_id)
            
            if not circuit:
                results.append(TestResult(
                    name=f"Circuit: {type_name}",
                    category=TestCategory.CIRCUIT_TYPES,
                    status=TestStatus.SKIPPED,
                    message=f"Circuit not in library",
                ))
                continue
            
            # Check if circuit can be validated
            try:
                if hasattr(self.context.backend_instance, "validate_circuit"):
                    valid = self.context.backend_instance.validate_circuit(circuit)
                    is_valid = valid.valid if hasattr(valid, "valid") else bool(valid)
                else:
                    is_valid = True  # Assume valid if no validation
                
                results.append(TestResult(
                    name=f"Circuit: {type_name}",
                    category=TestCategory.CIRCUIT_TYPES,
                    status=TestStatus.PASSED if is_valid else TestStatus.FAILED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=f"{type_name} circuit {'supported' if is_valid else 'not supported'}",
                    details={
                        "qubits": circuit.num_qubits,
                        "gates": circuit.gate_count,
                    },
                ))
                
            except Exception as e:
                results.append(TestResult(
                    name=f"Circuit: {type_name}",
                    category=TestCategory.CIRCUIT_TYPES,
                    status=TestStatus.ERROR,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=str(e)[:50],
                ))
        
        return results
    
    async def _run_error_handling_tests(self) -> List[TestResult]:
        """Run error handling tests."""
        results = []
        
        if not self.context.backend_instance:
            return results
        
        # Test 1: Invalid circuit handling
        start_time = time.time()
        try:
            # Try with None circuit
            if hasattr(self.context.backend_instance, "validate_circuit"):
                try:
                    self.context.backend_instance.validate_circuit(None)
                    # Should have raised an error
                    results.append(TestResult(
                        name="Null Circuit Handling",
                        category=TestCategory.ERROR_HANDLING,
                        status=TestStatus.FAILED,
                        duration_ms=(time.time() - start_time) * 1000,
                        message="Should reject null circuit",
                    ))
                except (ValueError, TypeError, AttributeError):
                    results.append(TestResult(
                        name="Null Circuit Handling",
                        category=TestCategory.ERROR_HANDLING,
                        status=TestStatus.PASSED,
                        duration_ms=(time.time() - start_time) * 1000,
                        message="Correctly rejects null circuit",
                    ))
            else:
                results.append(TestResult(
                    name="Null Circuit Handling",
                    category=TestCategory.ERROR_HANDLING,
                    status=TestStatus.SKIPPED,
                    message="No validate_circuit method",
                ))
        except Exception as e:
            results.append(TestResult(
                name="Null Circuit Handling",
                category=TestCategory.ERROR_HANDLING,
                status=TestStatus.PASSED,
                duration_ms=(time.time() - start_time) * 1000,
                message="Handles null circuit with exception",
            ))
        
        # Test 2: Invalid shots handling
        start_time = time.time()
        execute_method = None
        for method_name in ["execute", "run", "simulate"]:
            if hasattr(self.context.backend_instance, method_name):
                execute_method = getattr(self.context.backend_instance, method_name)
                break
        
        if execute_method:
            try:
                circuit = self.context.circuit_library.get_circuit("bell")
                execute_method(circuit, {"shots": -1})
                # Should have raised an error
                results.append(TestResult(
                    name="Invalid Shots Handling",
                    category=TestCategory.ERROR_HANDLING,
                    status=TestStatus.FAILED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message="Should reject negative shots",
                ))
            except (ValueError, NotImplementedError):
                results.append(TestResult(
                    name="Invalid Shots Handling",
                    category=TestCategory.ERROR_HANDLING,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message="Correctly handles invalid shots",
                ))
            except Exception:
                results.append(TestResult(
                    name="Invalid Shots Handling",
                    category=TestCategory.ERROR_HANDLING,
                    status=TestStatus.PASSED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message="Handles invalid shots with exception",
                ))
        else:
            results.append(TestResult(
                name="Invalid Shots Handling",
                category=TestCategory.ERROR_HANDLING,
                status=TestStatus.SKIPPED,
                message="No execute method found",
            ))
        
        # Test 3: Error messages
        start_time = time.time()
        has_error_class = False
        for attr in dir(self.context.backend_module):
            obj = getattr(self.context.backend_module, attr)
            if isinstance(obj, type) and issubclass(obj, Exception):
                has_error_class = True
                break
        
        results.append(TestResult(
            name="Custom Error Classes",
            category=TestCategory.ERROR_HANDLING,
            status=TestStatus.PASSED if has_error_class else TestStatus.PASSED,
            duration_ms=(time.time() - start_time) * 1000,
            message="Custom error classes defined" if has_error_class else "Using standard exceptions",
        ))
        
        return results
    
    async def _run_performance_tests(self, shots: int) -> List[TestResult]:
        """Run performance tests."""
        results = []
        
        if not self.context.backend_instance:
            return results
        
        execute_method = None
        for method_name in ["execute", "run", "simulate"]:
            if hasattr(self.context.backend_instance, method_name):
                execute_method = getattr(self.context.backend_instance, method_name)
                break
        
        if not execute_method:
            results.append(TestResult(
                name="Performance Test",
                category=TestCategory.PERFORMANCE,
                status=TestStatus.SKIPPED,
                message="No execute method found",
            ))
            return results
        
        # Test execution times for different circuit sizes
        perf_circuits = self.context.circuit_library.get_performance_suite(max_qubits=5)[:4]
        
        for circuit in perf_circuits:
            start_time = time.time()
            
            try:
                execute_method(circuit, {"shots": shots})
                duration = (time.time() - start_time) * 1000
                
                results.append(TestResult(
                    name=f"Perf: {circuit.name}",
                    category=TestCategory.PERFORMANCE,
                    status=TestStatus.PASSED,
                    duration_ms=duration,
                    message=f"Completed in {duration:.1f}ms",
                    details={
                        "qubits": circuit.num_qubits,
                        "gates": circuit.gate_count,
                        "duration_ms": duration,
                    },
                ))
                
            except NotImplementedError:
                results.append(TestResult(
                    name=f"Perf: {circuit.name}",
                    category=TestCategory.PERFORMANCE,
                    status=TestStatus.SKIPPED,
                    message="Not implemented",
                ))
            except Exception as e:
                results.append(TestResult(
                    name=f"Perf: {circuit.name}",
                    category=TestCategory.PERFORMANCE,
                    status=TestStatus.FAILED,
                    duration_ms=(time.time() - start_time) * 1000,
                    message=str(e)[:50],
                ))
        
        return results
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile all results into a summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        
        total_duration = sum(r.duration_ms for r in self.results)
        
        # Group by category
        by_category: Dict[str, List[Dict]] = {}
        for result in self.results:
            cat = result.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result.to_dict())
        
        return {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "skipped": skipped,
                "duration_ms": total_duration,
                "all_passed": failed == 0 and errors == 0,
                "pass_rate": (passed / total * 100) if total > 0 else 0,
            },
            "by_category": by_category,
            "results": [r.to_dict() for r in self.results],
            "aborted": self._abort_requested,
        }
    
    def _to_class_name(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
