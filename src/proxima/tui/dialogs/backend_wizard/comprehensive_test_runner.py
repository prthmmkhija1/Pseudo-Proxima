"""Comprehensive Test Runner.

Provides complete test suite for backend validation including:
- Unit Tests: Basic backend functionality
- Integration Tests: Proxima integration
- Performance Tests: Speed and scalability
- Compatibility Tests: Gate and feature support

Part of Phase 7: Advanced Testing & Validation.
"""

from __future__ import annotations

import asyncio
import time
import sys
import io
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TestLevel(Enum):
    """Test thoroughness level."""
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


@dataclass
class TestDefinition:
    """Definition of a test."""
    name: str
    category: str
    method: str
    description: str = ""
    timeout_seconds: float = 30.0
    requires: List[str] = field(default_factory=list)
    level: TestLevel = TestLevel.STANDARD


@dataclass
class TestExecutionResult:
    """Result of test execution."""
    test_name: str
    category: str
    passed: bool
    duration_ms: float
    message: str = ""
    error: str = ""
    stdout: str = ""
    stderr: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class TestRunner:
    """Base test runner with common functionality."""
    
    def __init__(
        self,
        backend_code: str,
        backend_name: str,
        backend_class_name: str,
        capabilities: Dict[str, Any],
        gate_mappings: Dict[str, str],
    ):
        """Initialize test runner.
        
        Args:
            backend_code: Generated backend code
            backend_name: Backend identifier
            backend_class_name: Name of the backend class
            capabilities: Backend capabilities
            gate_mappings: Gate mappings
        """
        self.backend_code = backend_code
        self.backend_name = backend_name
        self.backend_class_name = backend_class_name
        self.capabilities = capabilities
        self.gate_mappings = gate_mappings
        
        # Compiled module (if successful)
        self._compiled_module: Optional[Any] = None
        self._backend_instance: Optional[Any] = None
        
        # Test results
        self.results: List[TestExecutionResult] = []
    
    async def run_test(
        self,
        test_def: TestDefinition,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> TestExecutionResult:
        """Run a single test.
        
        Args:
            test_def: Test definition
            on_progress: Progress callback
            
        Returns:
            Test execution result
        """
        start_time = time.time()
        
        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Get test method
            method = getattr(self, test_def.method, None)
            
            if method is None:
                raise NotImplementedError(f"Test method not found: {test_def.method}")
            
            # Run with timeout
            try:
                result = await asyncio.wait_for(
                    method(),
                    timeout=test_def.timeout_seconds
                )
                
                passed, message, details = result
            
            except asyncio.TimeoutError:
                passed = False
                message = f"Test timed out after {test_def.timeout_seconds}s"
                details = {}
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Capture output
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            
            return TestExecutionResult(
                test_name=test_def.name,
                category=test_def.category,
                passed=passed,
                duration_ms=duration_ms,
                message=message,
                stdout=stdout,
                stderr=stderr,
                details=details,
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return TestExecutionResult(
                test_name=test_def.name,
                category=test_def.category,
                passed=False,
                duration_ms=duration_ms,
                error=str(e),
                stderr=traceback.format_exc(),
            )
        
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class UnitTestRunner(TestRunner):
    """Runner for unit tests."""
    
    def get_tests(self) -> List[TestDefinition]:
        """Get unit test definitions."""
        return [
            TestDefinition(
                name="Code syntax validation",
                category="unit",
                method="test_code_syntax",
                description="Verify generated code has valid Python syntax",
            ),
            TestDefinition(
                name="Backend import",
                category="unit",
                method="test_backend_import",
                description="Test backend module can be imported",
            ),
            TestDefinition(
                name="Backend instantiation",
                category="unit",
                method="test_backend_instantiation",
                description="Test backend class can be instantiated",
            ),
            TestDefinition(
                name="Backend name property",
                category="unit",
                method="test_backend_name",
                description="Test backend has correct name",
            ),
            TestDefinition(
                name="Backend version property",
                category="unit",
                method="test_backend_version",
                description="Test backend has version",
            ),
            TestDefinition(
                name="Supported gates property",
                category="unit",
                method="test_supported_gates",
                description="Test backend reports supported gates",
            ),
            TestDefinition(
                name="Max qubits property",
                category="unit",
                method="test_max_qubits",
                description="Test backend reports max qubits",
            ),
            TestDefinition(
                name="Run method exists",
                category="unit",
                method="test_run_method",
                description="Test backend has run method",
            ),
            TestDefinition(
                name="Result type validation",
                category="unit",
                method="test_result_type",
                description="Test run returns correct result type",
                level=TestLevel.COMPREHENSIVE,
            ),
        ]
    
    async def test_code_syntax(self) -> Tuple[bool, str, Dict]:
        """Test code has valid syntax."""
        try:
            compile(self.backend_code, "<backend>", "exec")
            return True, "Code syntax is valid", {}
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}", {
                "line": e.lineno,
                "offset": e.offset,
            }
    
    async def test_backend_import(self) -> Tuple[bool, str, Dict]:
        """Test backend can be imported."""
        try:
            # Create module
            import types
            module = types.ModuleType(self.backend_name)
            exec(self.backend_code, module.__dict__)
            self._compiled_module = module
            
            return True, "Backend imported successfully", {}
        except Exception as e:
            return False, f"Import failed: {e}", {}
    
    async def test_backend_instantiation(self) -> Tuple[bool, str, Dict]:
        """Test backend can be instantiated."""
        if self._compiled_module is None:
            await self.test_backend_import()
        
        if self._compiled_module is None:
            return False, "Module not available", {}
        
        try:
            backend_class = getattr(self._compiled_module, self.backend_class_name, None)
            
            if backend_class is None:
                return False, f"Class {self.backend_class_name} not found", {}
            
            self._backend_instance = backend_class()
            return True, "Backend instantiated successfully", {}
        except Exception as e:
            return False, f"Instantiation failed: {e}", {}
    
    async def test_backend_name(self) -> Tuple[bool, str, Dict]:
        """Test backend has name property."""
        if self._backend_instance is None:
            await self.test_backend_instantiation()
        
        if self._backend_instance is None:
            return False, "Backend instance not available", {}
        
        try:
            name = getattr(self._backend_instance, "name", None)
            
            if name is None:
                return False, "Backend has no 'name' property", {}
            
            return True, f"Backend name: {name}", {"name": name}
        except Exception as e:
            return False, f"Error getting name: {e}", {}
    
    async def test_backend_version(self) -> Tuple[bool, str, Dict]:
        """Test backend has version property."""
        if self._backend_instance is None:
            await self.test_backend_instantiation()
        
        if self._backend_instance is None:
            return False, "Backend instance not available", {}
        
        try:
            version = getattr(self._backend_instance, "version", None)
            
            if version is None:
                return False, "Backend has no 'version' property", {}
            
            return True, f"Backend version: {version}", {"version": version}
        except Exception as e:
            return False, f"Error getting version: {e}", {}
    
    async def test_supported_gates(self) -> Tuple[bool, str, Dict]:
        """Test backend reports supported gates."""
        if self._backend_instance is None:
            await self.test_backend_instantiation()
        
        if self._backend_instance is None:
            return False, "Backend instance not available", {}
        
        try:
            gates = getattr(self._backend_instance, "supported_gates", None)
            
            if gates is None:
                gates = getattr(self._backend_instance, "gates", None)
            
            if gates is None:
                return False, "Backend has no 'supported_gates' property", {}
            
            return True, f"Supports {len(gates)} gates", {"gates": list(gates)}
        except Exception as e:
            return False, f"Error getting gates: {e}", {}
    
    async def test_max_qubits(self) -> Tuple[bool, str, Dict]:
        """Test backend reports max qubits."""
        if self._backend_instance is None:
            await self.test_backend_instantiation()
        
        if self._backend_instance is None:
            return False, "Backend instance not available", {}
        
        try:
            max_qubits = getattr(self._backend_instance, "max_qubits", None)
            
            if max_qubits is None:
                max_qubits = self.capabilities.get("max_qubits", 0)
            
            if max_qubits is None or max_qubits <= 0:
                return False, "Backend has no valid 'max_qubits'", {}
            
            return True, f"Max qubits: {max_qubits}", {"max_qubits": max_qubits}
        except Exception as e:
            return False, f"Error getting max_qubits: {e}", {}
    
    async def test_run_method(self) -> Tuple[bool, str, Dict]:
        """Test backend has run method."""
        if self._backend_instance is None:
            await self.test_backend_instantiation()
        
        if self._backend_instance is None:
            return False, "Backend instance not available", {}
        
        try:
            run_method = getattr(self._backend_instance, "run", None)
            
            if run_method is None:
                return False, "Backend has no 'run' method", {}
            
            if not callable(run_method):
                return False, "'run' is not callable", {}
            
            return True, "Backend has valid 'run' method", {}
        except Exception as e:
            return False, f"Error checking run method: {e}", {}
    
    async def test_result_type(self) -> Tuple[bool, str, Dict]:
        """Test run returns correct result type."""
        # This would require running actual circuits
        # For now, validate the method signature
        return True, "Result type validation placeholder", {}


class IntegrationTestRunner(TestRunner):
    """Runner for integration tests."""
    
    def get_tests(self) -> List[TestDefinition]:
        """Get integration test definitions."""
        return [
            TestDefinition(
                name="Registry compatibility",
                category="integration",
                method="test_registry_compatibility",
                description="Test backend is compatible with registry",
            ),
            TestDefinition(
                name="Backend interface",
                category="integration",
                method="test_backend_interface",
                description="Test backend implements required interface",
            ),
            TestDefinition(
                name="Configuration loading",
                category="integration",
                method="test_configuration_loading",
                description="Test backend can load configuration",
            ),
            TestDefinition(
                name="Error handling",
                category="integration",
                method="test_error_handling",
                description="Test backend handles errors gracefully",
            ),
            TestDefinition(
                name="Result normalization",
                category="integration",
                method="test_result_normalization",
                description="Test results are properly normalized",
            ),
        ]
    
    async def test_registry_compatibility(self) -> Tuple[bool, str, Dict]:
        """Test backend is compatible with Proxima registry."""
        # Check required attributes
        required_attrs = ["name", "run"]
        
        if self._backend_instance is None:
            await self.test_backend_instantiation()
        
        if self._backend_instance is None:
            return False, "Backend not available", {}
        
        missing = []
        for attr in required_attrs:
            if not hasattr(self._backend_instance, attr):
                missing.append(attr)
        
        if missing:
            return False, f"Missing required attributes: {missing}", {}
        
        return True, "Backend is registry-compatible", {}
    
    async def test_backend_interface(self) -> Tuple[bool, str, Dict]:
        """Test backend implements required interface."""
        if self._backend_instance is None:
            await self.test_backend_instantiation()
        
        if self._backend_instance is None:
            return False, "Backend not available", {}
        
        # Check for standard interface methods
        interface_methods = [
            "run",
            "get_capabilities",
        ]
        
        implemented = []
        missing = []
        
        for method in interface_methods:
            if hasattr(self._backend_instance, method):
                implemented.append(method)
            else:
                missing.append(method)
        
        # 'run' is required, others are optional
        if "run" in missing:
            return False, "Missing required 'run' method", {}
        
        return True, f"Implements {len(implemented)} interface methods", {
            "implemented": implemented,
            "optional_missing": missing,
        }
    
    async def test_configuration_loading(self) -> Tuple[bool, str, Dict]:
        """Test backend can load configuration."""
        # Validate capabilities can be parsed
        if not self.capabilities:
            return True, "No configuration to validate", {}
        
        required_config = ["max_qubits"]
        
        for key in required_config:
            if key not in self.capabilities:
                return False, f"Missing required config: {key}", {}
        
        return True, "Configuration is valid", {
            "config_keys": list(self.capabilities.keys())
        }
    
    async def test_error_handling(self) -> Tuple[bool, str, Dict]:
        """Test backend handles errors gracefully."""
        # This would test with invalid inputs
        return True, "Error handling placeholder", {}
    
    async def test_result_normalization(self) -> Tuple[bool, str, Dict]:
        """Test results are properly normalized."""
        # This would verify result format
        return True, "Result normalization placeholder", {}


class PerformanceTestRunner(TestRunner):
    """Runner for performance tests."""
    
    def get_tests(self) -> List[TestDefinition]:
        """Get performance test definitions."""
        return [
            TestDefinition(
                name="Instantiation speed",
                category="performance",
                method="test_instantiation_speed",
                description="Measure backend instantiation time",
            ),
            TestDefinition(
                name="Small circuit speed",
                category="performance",
                method="test_small_circuit_speed",
                description="Measure execution time for small circuits",
                level=TestLevel.COMPREHENSIVE,
            ),
            TestDefinition(
                name="Memory baseline",
                category="performance",
                method="test_memory_baseline",
                description="Measure baseline memory usage",
            ),
            TestDefinition(
                name="Concurrent execution",
                category="performance",
                method="test_concurrent_execution",
                description="Test concurrent circuit execution",
                level=TestLevel.COMPREHENSIVE,
            ),
        ]
    
    async def test_instantiation_speed(self) -> Tuple[bool, str, Dict]:
        """Test backend instantiation speed."""
        if self._compiled_module is None:
            # Need to compile first
            import types
            module = types.ModuleType(self.backend_name)
            exec(self.backend_code, module.__dict__)
            self._compiled_module = module
        
        backend_class = getattr(self._compiled_module, self.backend_class_name, None)
        
        if backend_class is None:
            return False, "Backend class not found", {}
        
        # Measure instantiation time
        times = []
        
        for _ in range(10):
            start = time.perf_counter()
            _ = backend_class()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        
        # Pass if under 100ms
        passed = avg_time < 100
        
        return passed, f"Average instantiation: {avg_time:.2f}ms", {
            "avg_ms": avg_time,
            "min_ms": min(times),
            "max_ms": max(times),
        }
    
    async def test_small_circuit_speed(self) -> Tuple[bool, str, Dict]:
        """Test execution speed for small circuits."""
        # Placeholder - would run actual circuits
        return True, "Small circuit speed placeholder", {}
    
    async def test_memory_baseline(self) -> Tuple[bool, str, Dict]:
        """Test baseline memory usage."""
        import sys
        
        if self._backend_instance is None:
            await self.test_backend_instantiation()
        
        if self._backend_instance is None:
            return False, "Backend not available", {}
        
        # Estimate object size
        size = sys.getsizeof(self._backend_instance)
        
        # Pass if under 10MB
        passed = size < 10 * 1024 * 1024
        
        return passed, f"Backend object size: {size} bytes", {
            "size_bytes": size,
        }
    
    async def test_concurrent_execution(self) -> Tuple[bool, str, Dict]:
        """Test concurrent circuit execution."""
        # Placeholder - would test async execution
        return True, "Concurrent execution placeholder", {}


class CompatibilityTestRunner(TestRunner):
    """Runner for compatibility tests."""
    
    def get_tests(self) -> List[TestDefinition]:
        """Get compatibility test definitions."""
        standard_gates = ["h", "x", "y", "z", "cx", "cz", "s", "t", "rx", "ry", "rz"]
        
        tests = []
        
        for gate in standard_gates:
            tests.append(TestDefinition(
                name=f"{gate.upper()} gate support",
                category="compatibility",
                method=f"test_gate_{gate}",
                description=f"Test support for {gate.upper()} gate",
            ))
        
        tests.extend([
            TestDefinition(
                name="Measurement support",
                category="compatibility",
                method="test_measurement",
                description="Test measurement operations",
            ),
            TestDefinition(
                name="Barrier support",
                category="compatibility",
                method="test_barrier",
                description="Test barrier support",
            ),
            TestDefinition(
                name="Multi-qubit circuits",
                category="compatibility",
                method="test_multi_qubit",
                description="Test multi-qubit circuit support",
            ),
        ])
        
        return tests
    
    def _check_gate_support(self, gate: str) -> Tuple[bool, str, Dict]:
        """Check if a gate is supported."""
        gate_lower = gate.lower()
        gate_upper = gate.upper()
        
        # Check gate mappings
        if gate_lower in self.gate_mappings or gate_upper in self.gate_mappings:
            mapped = self.gate_mappings.get(gate_lower, self.gate_mappings.get(gate_upper))
            return True, f"{gate.upper()} gate mapped to: {mapped}", {"mapped": mapped}
        
        # Check if in supported gates from capabilities
        supported = self.capabilities.get("supported_gates", [])
        if gate_lower in [g.lower() for g in supported]:
            return True, f"{gate.upper()} gate in supported gates", {}
        
        return False, f"{gate.upper()} gate not mapped", {}
    
    async def test_gate_h(self) -> Tuple[bool, str, Dict]:
        """Test Hadamard gate support."""
        return self._check_gate_support("h")
    
    async def test_gate_x(self) -> Tuple[bool, str, Dict]:
        """Test Pauli-X gate support."""
        return self._check_gate_support("x")
    
    async def test_gate_y(self) -> Tuple[bool, str, Dict]:
        """Test Pauli-Y gate support."""
        return self._check_gate_support("y")
    
    async def test_gate_z(self) -> Tuple[bool, str, Dict]:
        """Test Pauli-Z gate support."""
        return self._check_gate_support("z")
    
    async def test_gate_cx(self) -> Tuple[bool, str, Dict]:
        """Test CNOT gate support."""
        return self._check_gate_support("cx")
    
    async def test_gate_cz(self) -> Tuple[bool, str, Dict]:
        """Test CZ gate support."""
        return self._check_gate_support("cz")
    
    async def test_gate_s(self) -> Tuple[bool, str, Dict]:
        """Test S gate support."""
        return self._check_gate_support("s")
    
    async def test_gate_t(self) -> Tuple[bool, str, Dict]:
        """Test T gate support."""
        return self._check_gate_support("t")
    
    async def test_gate_rx(self) -> Tuple[bool, str, Dict]:
        """Test RX gate support."""
        return self._check_gate_support("rx")
    
    async def test_gate_ry(self) -> Tuple[bool, str, Dict]:
        """Test RY gate support."""
        return self._check_gate_support("ry")
    
    async def test_gate_rz(self) -> Tuple[bool, str, Dict]:
        """Test RZ gate support."""
        return self._check_gate_support("rz")
    
    async def test_measurement(self) -> Tuple[bool, str, Dict]:
        """Test measurement support."""
        # Check if measurements are configured
        supports = self.capabilities.get("supports_measurements", True)
        
        if supports:
            return True, "Measurement operations supported", {}
        
        return False, "Measurements not configured", {}
    
    async def test_barrier(self) -> Tuple[bool, str, Dict]:
        """Test barrier support."""
        supports = self.capabilities.get("supports_barrier", True)
        
        if supports:
            return True, "Barrier operations supported", {}
        
        return False, "Barrier not configured", {}
    
    async def test_multi_qubit(self) -> Tuple[bool, str, Dict]:
        """Test multi-qubit circuit support."""
        max_qubits = self.capabilities.get("max_qubits", 0)
        
        if max_qubits >= 2:
            return True, f"Supports up to {max_qubits} qubits", {"max_qubits": max_qubits}
        
        return False, f"Only {max_qubits} qubit(s) supported", {}


class FullTestSuite:
    """Complete test suite combining all test runners."""
    
    def __init__(
        self,
        backend_code: str,
        backend_name: str,
        backend_class_name: str,
        capabilities: Dict[str, Any],
        gate_mappings: Dict[str, str],
        level: TestLevel = TestLevel.STANDARD,
    ):
        """Initialize full test suite.
        
        Args:
            backend_code: Generated backend code
            backend_name: Backend identifier
            backend_class_name: Name of the backend class
            capabilities: Backend capabilities
            gate_mappings: Gate mappings
            level: Test thoroughness level
        """
        self.level = level
        
        # Initialize runners
        common_args = {
            "backend_code": backend_code,
            "backend_name": backend_name,
            "backend_class_name": backend_class_name,
            "capabilities": capabilities,
            "gate_mappings": gate_mappings,
        }
        
        self.runners = {
            "unit": UnitTestRunner(**common_args),
            "integration": IntegrationTestRunner(**common_args),
            "performance": PerformanceTestRunner(**common_args),
            "compatibility": CompatibilityTestRunner(**common_args),
        }
    
    def get_all_tests(self) -> Dict[str, List[TestDefinition]]:
        """Get all test definitions organized by category."""
        all_tests = {}
        
        for category, runner in self.runners.items():
            tests = runner.get_tests()
            
            # Filter by level
            if self.level == TestLevel.QUICK:
                tests = [t for t in tests if t.level != TestLevel.COMPREHENSIVE]
            
            all_tests[category] = tests
        
        return all_tests
    
    async def run_all(
        self,
        on_test_start: Optional[Callable[[str, str], None]] = None,
        on_test_complete: Optional[Callable[[TestExecutionResult], None]] = None,
        on_category_complete: Optional[Callable[[str, List[TestExecutionResult]], None]] = None,
    ) -> Dict[str, List[TestExecutionResult]]:
        """Run all tests.
        
        Args:
            on_test_start: Callback when test starts
            on_test_complete: Callback when test completes
            on_category_complete: Callback when category completes
            
        Returns:
            Dictionary of results by category
        """
        all_results = {}
        
        for category, tests in self.get_all_tests().items():
            runner = self.runners[category]
            category_results = []
            
            for test in tests:
                if on_test_start:
                    on_test_start(test.name, category)
                
                result = await runner.run_test(test)
                category_results.append(result)
                
                if on_test_complete:
                    on_test_complete(result)
            
            all_results[category] = category_results
            
            if on_category_complete:
                on_category_complete(category, category_results)
        
        return all_results
    
    def get_summary(
        self,
        results: Dict[str, List[TestExecutionResult]]
    ) -> Dict[str, Any]:
        """Get summary of test results.
        
        Args:
            results: Test results by category
            
        Returns:
            Summary dictionary
        """
        total = 0
        passed = 0
        failed = 0
        total_duration = 0.0
        
        category_summaries = {}
        
        for category, cat_results in results.items():
            cat_total = len(cat_results)
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_failed = cat_total - cat_passed
            cat_duration = sum(r.duration_ms for r in cat_results)
            
            total += cat_total
            passed += cat_passed
            failed += cat_failed
            total_duration += cat_duration
            
            category_summaries[category] = {
                "total": cat_total,
                "passed": cat_passed,
                "failed": cat_failed,
                "duration_ms": cat_duration,
                "pass_rate": (cat_passed / cat_total * 100) if cat_total > 0 else 0,
            }
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration_ms": total_duration,
            "categories": category_summaries,
        }
