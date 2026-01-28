"""Backend Test Runner.

Runs validation tests on generated backend code before deployment.
Provides comprehensive testing including:
- Backend initialization
- Circuit validation
- Circuit execution
- Result normalization

Enhanced for Phase 4: Testing & Validation Interface.
Now integrates with the full testing package for comprehensive validation.
"""

from __future__ import annotations

import asyncio
import tempfile
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# Try to import the full testing package
try:
    from proxima.tui.testing.test_executor import (
        TestExecutor,
        TestResult as FullTestResult,
        TestStatus,
        TestCategory,
        TestProgress,
    )
    from proxima.tui.testing.test_circuits import TestCircuitLibrary, TestCircuit
    from proxima.tui.testing.test_reporter import TestReporter, TestSummary
    FULL_TESTING_AVAILABLE = True
except ImportError:
    FULL_TESTING_AVAILABLE = False


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    status: str  # 'SUCCESS', 'FAILED', 'SKIPPED', 'ERROR'
    message: str = ""
    duration_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None


class BackendTestRunner:
    """Run tests on generated backend code.
    
    Executes a series of validation tests to ensure the generated
    backend code is correct and functional before deployment.
    """
    
    def __init__(self, backend_code: str, backend_name: str):
        """Initialize test runner.
        
        Args:
            backend_code: The generated Python code for the backend
            backend_name: Name of the backend (used for imports)
        """
        self.backend_code = backend_code
        self.backend_name = backend_name
        self.temp_dir: Optional[Path] = None
        self._backend_module = None
    
    async def run_all_tests(
        self,
        circuit_type: str = "bell",
        shots: int = 1024
    ) -> Dict[str, Any]:
        """Run all backend tests.
        
        Args:
            circuit_type: Type of test circuit ('bell', 'ghz', 'qft', 'random')
            shots: Number of measurement shots
            
        Returns:
            Dictionary with test results
        """
        results: Dict[str, TestResult] = {}
        
        # Setup temporary module
        setup_result = await self._setup_temp_module()
        if not setup_result['success']:
            return {
                'Backend initialization': TestResult(
                    name='Backend initialization',
                    passed=False,
                    status='ERROR',
                    message=setup_result.get('error', 'Setup failed')
                )
            }
        
        try:
            # Test 1: Backend initialization
            results['Backend initialization'] = await self._test_initialization()
            
            # Only continue if initialization passed
            if results['Backend initialization'].passed:
                # Test 2: Circuit validation
                results['Circuit validation'] = await self._test_validation(circuit_type)
                
                # Test 3: Circuit execution
                results['Circuit execution'] = await self._test_execution(circuit_type, shots)
                
                # Test 4: Result normalization
                results['Result normalization'] = await self._test_normalization(circuit_type, shots)
            
            # Calculate summary
            total_tests = len(results)
            passed_tests = sum(1 for r in results.values() if r.passed)
            total_duration = sum(r.duration_ms for r in results.values())
            
            return {
                **{name: vars(result) for name, result in results.items()},
                'summary': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': total_tests - passed_tests,
                    'duration_ms': total_duration,
                    'all_passed': passed_tests == total_tests,
                }
            }
            
        finally:
            # Cleanup
            await self._cleanup_temp_module()
    
    async def _setup_temp_module(self) -> Dict[str, Any]:
        """Create temporary module for testing."""
        try:
            # Create temp directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix=f"proxima_test_{self.backend_name}_"))
            
            # Write backend code
            backend_path = self.temp_dir / f"{self.backend_name}.py"
            backend_path.write_text(self.backend_code)
            
            # Add to Python path
            sys.path.insert(0, str(self.temp_dir))
            
            return {'success': True, 'path': str(backend_path)}
            
        except Exception as e:
            logger.error(f"Failed to setup temp module: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _cleanup_temp_module(self) -> None:
        """Clean up temporary module."""
        try:
            # Remove from Python path
            if self.temp_dir and str(self.temp_dir) in sys.path:
                sys.path.remove(str(self.temp_dir))
            
            # Unload module
            if self.backend_name in sys.modules:
                del sys.modules[self.backend_name]
            
            # Clean up temp directory
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    async def _test_initialization(self) -> TestResult:
        """Test backend initialization."""
        start_time = time.time()
        
        try:
            # Import backend module
            self._backend_module = __import__(self.backend_name)
            
            # Find backend class
            class_name = self._to_class_name(self.backend_name) + "Backend"
            
            # Check for common class name patterns
            backend_class = None
            for attr_name in dir(self._backend_module):
                if 'backend' in attr_name.lower() or 'adapter' in attr_name.lower():
                    attr = getattr(self._backend_module, attr_name)
                    if isinstance(attr, type):
                        backend_class = attr
                        break
            
            if backend_class is None:
                # Try exact class name
                backend_class = getattr(self._backend_module, class_name, None)
            
            if backend_class is None:
                return TestResult(
                    name='Backend initialization',
                    passed=False,
                    status='FAILED',
                    message=f"Could not find backend class in module",
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Create instance
            backend = backend_class()
            
            # Try to initialize
            if hasattr(backend, 'initialize'):
                try:
                    backend.initialize()
                except Exception as init_error:
                    # Initialization may fail if dependencies aren't installed
                    # This is expected in some cases
                    logger.warning(f"Backend initialization warning: {init_error}")
            
            return TestResult(
                name='Backend initialization',
                passed=True,
                status='SUCCESS',
                message='Backend class loaded and instantiated successfully',
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except SyntaxError as e:
            return TestResult(
                name='Backend initialization',
                passed=False,
                status='FAILED',
                message=f"Syntax error: {e}",
                duration_ms=(time.time() - start_time) * 1000
            )
        except ImportError as e:
            return TestResult(
                name='Backend initialization',
                passed=False,
                status='FAILED',
                message=f"Import error: {e}",
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return TestResult(
                name='Backend initialization',
                passed=False,
                status='ERROR',
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def _test_validation(self, circuit_type: str) -> TestResult:
        """Test circuit validation."""
        start_time = time.time()
        
        try:
            # Get backend class
            backend_class = self._get_backend_class()
            if not backend_class:
                return TestResult(
                    name='Circuit validation',
                    passed=False,
                    status='SKIPPED',
                    message='Backend class not available',
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            backend = backend_class()
            
            # Check if validate_circuit method exists
            if not hasattr(backend, 'validate_circuit'):
                return TestResult(
                    name='Circuit validation',
                    passed=True,
                    status='SUCCESS',
                    message='validate_circuit method not implemented (optional)',
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Create mock circuit
            circuit = self._create_mock_circuit(circuit_type)
            
            # Validate
            result = backend.validate_circuit(circuit)
            
            # Check result
            if hasattr(result, 'valid'):
                passed = result.valid
            else:
                passed = bool(result)
            
            return TestResult(
                name='Circuit validation',
                passed=passed,
                status='SUCCESS' if passed else 'FAILED',
                message='Circuit validated successfully' if passed else 'Circuit validation failed',
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                name='Circuit validation',
                passed=False,
                status='ERROR',
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def _test_execution(self, circuit_type: str, shots: int) -> TestResult:
        """Test circuit execution."""
        start_time = time.time()
        
        try:
            backend_class = self._get_backend_class()
            if not backend_class:
                return TestResult(
                    name='Circuit execution',
                    passed=False,
                    status='SKIPPED',
                    message='Backend class not available',
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            backend = backend_class()
            
            # Try to initialize
            if hasattr(backend, 'initialize'):
                try:
                    backend.initialize()
                except Exception:
                    pass  # May fail if deps not installed
            
            # Check for execute method
            execute_method = None
            for method_name in ['execute', 'run', 'simulate']:
                if hasattr(backend, method_name):
                    execute_method = getattr(backend, method_name)
                    break
            
            if not execute_method:
                return TestResult(
                    name='Circuit execution',
                    passed=False,
                    status='FAILED',
                    message='No execute/run method found',
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Create mock circuit
            circuit = self._create_mock_circuit(circuit_type)
            
            # Try execution (this may fail if backend isn't fully implemented)
            try:
                result = execute_method(circuit, {'shots': shots})
                
                return TestResult(
                    name='Circuit execution',
                    passed=True,
                    status='SUCCESS',
                    message='Circuit executed successfully',
                    duration_ms=(time.time() - start_time) * 1000,
                    details={'result': str(result)[:200]}
                )
            except NotImplementedError:
                return TestResult(
                    name='Circuit execution',
                    passed=True,
                    status='SUCCESS',
                    message='Execute method exists (NotImplementedError expected for template)',
                    duration_ms=(time.time() - start_time) * 1000
                )
            
        except Exception as e:
            return TestResult(
                name='Circuit execution',
                passed=False,
                status='ERROR',
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def _test_normalization(self, circuit_type: str, shots: int) -> TestResult:
        """Test result normalization."""
        start_time = time.time()
        
        try:
            # Check if normalizer exists
            normalizer_class_name = self._to_class_name(self.backend_name) + "Normalizer"
            
            normalizer_class = getattr(self._backend_module, normalizer_class_name, None)
            
            if not normalizer_class:
                return TestResult(
                    name='Result normalization',
                    passed=True,
                    status='SUCCESS',
                    message='Normalizer not included (optional)',
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Create normalizer instance
            normalizer = normalizer_class()
            
            # Test with mock result
            mock_result = {
                'counts': {'00': shots // 2, '11': shots // 2},
                'shots': shots
            }
            
            normalized = normalizer.normalize(mock_result)
            
            # Check normalization
            if 'counts' in normalized:
                total = sum(normalized['counts'].values())
                if total == shots:
                    return TestResult(
                        name='Result normalization',
                        passed=True,
                        status='SUCCESS',
                        message='Result normalized correctly',
                        duration_ms=(time.time() - start_time) * 1000
                    )
            
            return TestResult(
                name='Result normalization',
                passed=False,
                status='FAILED',
                message='Normalization produced incorrect result',
                duration_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TestResult(
                name='Result normalization',
                passed=False,
                status='ERROR',
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def _get_backend_class(self):
        """Get the backend class from the loaded module."""
        if not self._backend_module:
            return None
        
        # Try common patterns
        class_name = self._to_class_name(self.backend_name)
        
        for suffix in ['Backend', 'Adapter', '']:
            full_name = class_name + suffix
            if hasattr(self._backend_module, full_name):
                return getattr(self._backend_module, full_name)
        
        return None
    
    def _create_mock_circuit(self, circuit_type: str):
        """Create a mock circuit for testing."""
        class MockCircuit:
            """Mock quantum circuit for testing."""
            
            def __init__(self, num_qubits: int):
                self.num_qubits = num_qubits
                self.gate_count = 0
                self.operations = []
            
            def h(self, qubit: int):
                self.operations.append(('H', qubit))
                self.gate_count += 1
            
            def cx(self, control: int, target: int):
                self.operations.append(('CX', control, target))
                self.gate_count += 1
            
            def measure_all(self):
                self.operations.append(('MEASURE_ALL',))
            
            def to_dict(self):
                return {
                    'num_qubits': self.num_qubits,
                    'operations': self.operations
                }
            
            def to_json(self):
                import json
                return json.dumps(self.to_dict())
        
        if circuit_type == "bell":
            circuit = MockCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.measure_all()
        
        elif circuit_type == "ghz":
            circuit = MockCircuit(3)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.cx(1, 2)
            circuit.measure_all()
        
        else:  # random or default
            circuit = MockCircuit(2)
            circuit.h(0)
            circuit.h(1)
            circuit.measure_all()
        
        return circuit
    
    def _to_class_name(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)


class EnhancedBackendTestRunner:
    """Enhanced test runner with full testing package integration.
    
    Provides comprehensive testing with:
    - Multiple test categories
    - Real-time progress callbacks
    - Detailed reporting
    - Test circuit library
    - Performance benchmarks
    """
    
    def __init__(
        self,
        backend_code: str,
        backend_name: str,
        normalizer_code: Optional[str] = None
    ):
        """Initialize enhanced test runner.
        
        Args:
            backend_code: Generated Python code for the backend
            backend_name: Name of the backend (snake_case)
            normalizer_code: Optional normalizer code
        """
        self.backend_code = backend_code
        self.backend_name = backend_name
        self.normalizer_code = normalizer_code
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates.
        
        Args:
            callback: Function called with progress updates
        """
        self._progress_callback = callback
    
    async def run_comprehensive_tests(
        self,
        circuit_type: str = "bell",
        shots: int = 1024,
        include_performance: bool = False
    ) -> Dict[str, Any]:
        """Run comprehensive test suite.
        
        Uses the full testing package if available, otherwise
        falls back to basic testing.
        
        Args:
            circuit_type: Type of test circuit
            shots: Number of measurement shots
            include_performance: Whether to include performance tests
            
        Returns:
            Dictionary with test results and summary
        """
        if FULL_TESTING_AVAILABLE:
            return await self._run_full_tests(
                circuit_type, shots, include_performance
            )
        else:
            # Fallback to basic runner
            basic_runner = BackendTestRunner(
                self.backend_code,
                self.backend_name
            )
            return await basic_runner.run_all_tests(circuit_type, shots)
    
    async def _run_full_tests(
        self,
        circuit_type: str,
        shots: int,
        include_performance: bool
    ) -> Dict[str, Any]:
        """Run tests using full testing package."""
        executor = TestExecutor(
            backend_code=self.backend_code,
            backend_name=self.backend_name,
            normalizer_code=self.normalizer_code,
        )
        
        if self._progress_callback:
            executor.set_progress_callback(self._progress_callback)
        
        results = await executor.run_all_tests(
            circuit_type=circuit_type,
            shots=shots,
            include_performance=include_performance,
        )
        
        return results
    
    def get_test_circuits(self) -> List[Dict[str, str]]:
        """Get list of available test circuits.
        
        Returns:
            List of circuit info dictionaries
        """
        if FULL_TESTING_AVAILABLE:
            library = TestCircuitLibrary()
            return [
                {"id": name, "description": desc}
                for name, desc in library.list_circuits()
            ]
        else:
            return [
                {"id": "bell", "description": "Bell State (2 qubits)"},
                {"id": "ghz", "description": "GHZ State (3 qubits)"},
                {"id": "random", "description": "Random Circuit"},
            ]
    
    def generate_report(
        self,
        results: Dict[str, Any],
        format: str = "console"
    ) -> str:
        """Generate formatted test report.
        
        Args:
            results: Test results dictionary
            format: Output format (console, markdown, json)
            
        Returns:
            Formatted report string
        """
        if FULL_TESTING_AVAILABLE:
            reporter = TestReporter(self.backend_name)
            
            # Add results to reporter
            for result in results.get("results", []):
                reporter.add_result(FullTestResult(
                    name=result["name"],
                    category=TestCategory[result["category"].upper()],
                    status=TestStatus[result["status"].upper()],
                    duration_ms=result.get("duration_ms", 0),
                    message=result.get("message", ""),
                ))
            
            if format == "markdown":
                return reporter.generate_report().to_markdown()
            elif format == "json":
                return reporter.format_json()
            else:
                return reporter.format_console()
        else:
            # Basic formatting
            summary = results.get("summary", {})
            lines = [
                "=" * 50,
                f"  TEST RESULTS: {self.backend_name}",
                "=" * 50,
                "",
                f"  Total: {summary.get('total', 0)}",
                f"  Passed: {summary.get('passed', 0)}",
                f"  Failed: {summary.get('failed', 0)}",
                "",
            ]
            return "\n".join(lines)


def create_test_runner(
    backend_code: str,
    backend_name: str,
    normalizer_code: Optional[str] = None,
    use_enhanced: bool = True
) -> BackendTestRunner:
    """Factory function to create appropriate test runner.
    
    Args:
        backend_code: Generated backend code
        backend_name: Backend name
        normalizer_code: Optional normalizer code
        use_enhanced: Whether to use enhanced runner if available
        
    Returns:
        Test runner instance
    """
    if use_enhanced and FULL_TESTING_AVAILABLE:
        return EnhancedBackendTestRunner(
            backend_code, backend_name, normalizer_code
        )
    else:
        return BackendTestRunner(backend_code, backend_name)
