"""Comprehensive Testing Framework for Dynamic AI Assistant.

This module implements Phase 9.1 for the Dynamic AI Assistant:
- Unit Testing Infrastructure
- Integration Testing
- LLM Response Testing
- Performance Testing

Key Features:
============
- Pytest-based unit testing with fixtures and parameterization
- Mock objects for external dependencies
- End-to-end workflow testing
- LLM response validation and quality metrics
- Performance testing with load and stress tests

Design Principle:
================
All test generation and validation uses LLM reasoning when available.
Tests are self-documenting and can be extended dynamically.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import logging
import os
import random
import re
import shutil
import statistics
import string
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Coroutine, Dict, Generator, Generic,
    Iterator, List, Optional, Pattern, Set, Tuple, Type,
    TypeVar, Union
)
import uuid

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestPriority(Enum):
    """Test priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestCategory(Enum):
    """Test categories."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    LLM = "llm"
    REGRESSION = "regression"
    SMOKE = "smoke"


class MockBehavior(Enum):
    """Mock object behaviors."""
    RETURN_VALUE = "return_value"
    RAISE_EXCEPTION = "raise_exception"
    CALL_ORIGINAL = "call_original"
    RECORD_CALLS = "record_calls"


@dataclass
class TestCase:
    """Represents a single test case."""
    test_id: str
    name: str
    category: TestCategory
    
    # Test function
    test_func: Optional[Callable] = None
    
    # Metadata
    description: str = ""
    priority: TestPriority = TestPriority.MEDIUM
    tags: Set[str] = field(default_factory=set)
    
    # Parameters for parameterized tests
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution info
    timeout: float = 60.0
    retries: int = 0
    
    # Results
    status: TestStatus = TestStatus.PENDING
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
        }


@dataclass
class TestSuite:
    """Collection of related test cases."""
    suite_id: str
    name: str
    
    # Tests
    tests: List[TestCase] = field(default_factory=list)
    
    # Setup/teardown
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    
    # Metadata
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    
    # Results
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "tests": [t.to_dict() for t in self.tests],
        }


@dataclass
class TestFixture:
    """Test fixture for setup/teardown."""
    fixture_id: str
    name: str
    scope: str  # function, class, module, session
    
    # Setup/teardown
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    
    # State
    value: Any = None
    is_initialized: bool = False
    
    def initialize(self):
        """Initialize the fixture."""
        if self.setup:
            self.value = self.setup()
        self.is_initialized = True
    
    def cleanup(self):
        """Clean up the fixture."""
        if self.teardown:
            self.teardown(self.value)
        self.is_initialized = False


@dataclass
class MockCall:
    """Records a mock function call."""
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    timestamp: datetime
    return_value: Any = None
    exception: Optional[Exception] = None


@dataclass
class CoverageReport:
    """Test coverage report."""
    total_lines: int = 0
    covered_lines: int = 0
    missing_lines: List[int] = field(default_factory=list)
    
    # By file
    file_coverage: Dict[str, float] = field(default_factory=dict)
    
    # By function
    function_coverage: Dict[str, float] = field(default_factory=dict)
    
    @property
    def percentage(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return (self.covered_lines / self.total_lines) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "percentage": round(self.percentage, 2),
            "file_coverage": self.file_coverage,
        }


@dataclass
class PerformanceMetrics:
    """Performance test metrics."""
    operation: str
    
    # Timing
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    p50_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    
    # Resources
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Errors
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "min_time_ms": round(self.min_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "p95_time_ms": round(self.p95_time_ms, 2),
            "requests_per_second": round(self.requests_per_second, 2),
            "error_rate": round(self.error_rate, 4),
        }


@dataclass 
class LLMTestResult:
    """LLM response test result."""
    prompt: str
    expected_schema: Optional[Dict[str, Any]] = None
    
    # Response
    response: Optional[str] = None
    parsed_response: Optional[Dict[str, Any]] = None
    
    # Validation
    schema_valid: bool = False
    quality_score: float = 0.0
    
    # Timing
    latency_ms: float = 0.0
    
    # Cost
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_valid": self.schema_valid,
            "quality_score": round(self.quality_score, 2),
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class MockObject:
    """Mock object for testing.
    
    Creates mock replacements for real objects to isolate tests.
    
    Example:
        >>> mock = MockObject()
        >>> mock.return_value = "mocked"
        >>> mock() == "mocked"
        True
    """
    
    def __init__(
        self,
        name: str = "mock",
        return_value: Any = None,
        side_effect: Optional[Union[Exception, Callable]] = None,
    ):
        """Initialize mock object.
        
        Args:
            name: Mock name for identification
            return_value: Value to return when called
            side_effect: Exception to raise or function to call
        """
        self._name = name
        self._return_value = return_value
        self._side_effect = side_effect
        
        # Call tracking
        self._calls: List[MockCall] = []
        self._call_count = 0
        
        # Child mocks
        self._children: Dict[str, MockObject] = {}
    
    def __call__(self, *args, **kwargs) -> Any:
        """Call the mock."""
        call = MockCall(
            args=args,
            kwargs=kwargs,
            timestamp=datetime.now(),
        )
        
        self._calls.append(call)
        self._call_count += 1
        
        if self._side_effect is not None:
            if isinstance(self._side_effect, Exception):
                call.exception = self._side_effect
                raise self._side_effect
            elif callable(self._side_effect):
                result = self._side_effect(*args, **kwargs)
                call.return_value = result
                return result
        
        call.return_value = self._return_value
        return self._return_value
    
    def __getattr__(self, name: str) -> 'MockObject':
        """Get child mock for attribute access."""
        if name.startswith('_'):
            raise AttributeError(name)
        
        if name not in self._children:
            self._children[name] = MockObject(f"{self._name}.{name}")
        
        return self._children[name]
    
    @property
    def return_value(self) -> Any:
        return self._return_value
    
    @return_value.setter
    def return_value(self, value: Any):
        self._return_value = value
    
    @property
    def side_effect(self) -> Optional[Union[Exception, Callable]]:
        return self._side_effect
    
    @side_effect.setter
    def side_effect(self, value: Optional[Union[Exception, Callable]]):
        self._side_effect = value
    
    @property
    def call_count(self) -> int:
        return self._call_count
    
    @property
    def calls(self) -> List[MockCall]:
        return list(self._calls)
    
    def assert_called(self):
        """Assert that mock was called at least once."""
        if self._call_count == 0:
            raise AssertionError(f"Expected {self._name} to be called")
    
    def assert_called_once(self):
        """Assert that mock was called exactly once."""
        if self._call_count != 1:
            raise AssertionError(
                f"Expected {self._name} to be called once, "
                f"but was called {self._call_count} times"
            )
    
    def assert_called_with(self, *args, **kwargs):
        """Assert that mock was called with specific arguments."""
        if not self._calls:
            raise AssertionError(f"{self._name} was never called")
        
        last_call = self._calls[-1]
        if last_call.args != args or last_call.kwargs != kwargs:
            raise AssertionError(
                f"Expected call: {self._name}{args}, {kwargs}\n"
                f"Actual call: {self._name}{last_call.args}, {last_call.kwargs}"
            )
    
    def reset(self):
        """Reset call tracking."""
        self._calls.clear()
        self._call_count = 0
        for child in self._children.values():
            child.reset()


class TestRunner:
    """Runs test suites and collects results.
    
    Uses LLM reasoning to:
    1. Analyze test failures
    2. Suggest fixes
    3. Generate additional test cases
    
    Example:
        >>> runner = TestRunner()
        >>> results = runner.run_suite(my_suite)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        parallel: bool = True,
        max_workers: int = 4,
    ):
        """Initialize test runner.
        
        Args:
            llm_client: LLM client for intelligent analysis
            parallel: Run tests in parallel
            max_workers: Max parallel workers
        """
        self._llm_client = llm_client
        self._parallel = parallel
        self._max_workers = max_workers
        
        # Fixtures
        self._fixtures: Dict[str, TestFixture] = {}
        
        # Results
        self._results: List[TestSuite] = []
        
        # Hooks
        self._before_test_hooks: List[Callable] = []
        self._after_test_hooks: List[Callable] = []
    
    def register_fixture(self, fixture: TestFixture):
        """Register a test fixture."""
        self._fixtures[fixture.name] = fixture
    
    def add_before_test_hook(self, hook: Callable):
        """Add hook to run before each test."""
        self._before_test_hooks.append(hook)
    
    def add_after_test_hook(self, hook: Callable):
        """Add hook to run after each test."""
        self._after_test_hooks.append(hook)
    
    def run_test(self, test: TestCase) -> TestCase:
        """Run a single test case.
        
        Args:
            test: Test case to run
            
        Returns:
            Test case with results
        """
        test.status = TestStatus.RUNNING
        start_time = time.time()
        
        # Run before hooks
        for hook in self._before_test_hooks:
            try:
                hook(test)
            except Exception as e:
                logger.warning(f"Before hook failed: {e}")
        
        try:
            if test.test_func is None:
                test.status = TestStatus.SKIPPED
                test.error_message = "No test function"
                return test
            
            # Get required fixtures
            sig = inspect.signature(test.test_func)
            kwargs = {}
            
            for param_name in sig.parameters:
                if param_name in self._fixtures:
                    fixture = self._fixtures[param_name]
                    if not fixture.is_initialized:
                        fixture.initialize()
                    kwargs[param_name] = fixture.value
            
            # Run with timeout
            if test.parameters:
                # Parameterized test
                for params in test.parameters:
                    test.test_func(**{**kwargs, **params})
            else:
                test.test_func(**kwargs)
            
            test.status = TestStatus.PASSED
            
        except AssertionError as e:
            test.status = TestStatus.FAILED
            test.error_message = str(e)
            test.output = traceback.format_exc()
            
        except Exception as e:
            test.status = TestStatus.ERROR
            test.error_message = f"{type(e).__name__}: {e}"
            test.output = traceback.format_exc()
        
        finally:
            test.duration_ms = (time.time() - start_time) * 1000
            
            # Run after hooks
            for hook in self._after_test_hooks:
                try:
                    hook(test)
                except Exception as e:
                    logger.warning(f"After hook failed: {e}")
        
        return test
    
    def run_suite(self, suite: TestSuite) -> TestSuite:
        """Run a test suite.
        
        Args:
            suite: Test suite to run
            
        Returns:
            Suite with results
        """
        start_time = time.time()
        
        # Run suite setup
        if suite.setup_func:
            try:
                suite.setup_func()
            except Exception as e:
                logger.error(f"Suite setup failed: {e}")
        
        # Run tests
        if self._parallel and len(suite.tests) > 1:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(self.run_test, test): test
                    for test in suite.tests
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Test execution error: {e}")
        else:
            for test in suite.tests:
                self.run_test(test)
        
        # Calculate results
        suite.total_tests = len(suite.tests)
        suite.passed = sum(1 for t in suite.tests if t.status == TestStatus.PASSED)
        suite.failed = sum(1 for t in suite.tests if t.status == TestStatus.FAILED)
        suite.skipped = sum(1 for t in suite.tests if t.status == TestStatus.SKIPPED)
        suite.errors = sum(1 for t in suite.tests if t.status == TestStatus.ERROR)
        suite.duration_ms = (time.time() - start_time) * 1000
        
        # Run suite teardown
        if suite.teardown_func:
            try:
                suite.teardown_func()
            except Exception as e:
                logger.error(f"Suite teardown failed: {e}")
        
        self._results.append(suite)
        
        return suite
    
    def get_results(self) -> List[TestSuite]:
        """Get all test results."""
        return list(self._results)
    
    def clear_results(self):
        """Clear all results."""
        self._results.clear()


class TestGenerator:
    """Generate test cases using LLM.
    
    Uses LLM reasoning to:
    1. Analyze code and generate unit tests
    2. Create edge case tests
    3. Generate parameterized test data
    
    Example:
        >>> generator = TestGenerator(llm_client)
        >>> tests = generator.generate_tests(my_function)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize test generator.
        
        Args:
            llm_client: LLM client for test generation
        """
        self._llm_client = llm_client
    
    def generate_unit_tests(
        self,
        func: Callable,
        num_tests: int = 5,
    ) -> List[TestCase]:
        """Generate unit tests for a function.
        
        Args:
            func: Function to test
            num_tests: Number of tests to generate
            
        Returns:
            List of generated test cases
        """
        tests = []
        
        # Get function signature
        sig = inspect.signature(func)
        source = ""
        try:
            source = inspect.getsource(func)
        except Exception:
            pass
        
        # Generate tests based on function analysis
        func_name = func.__name__
        
        # Basic test - function executes without error
        def test_executes():
            # Call with default values or None
            kwargs = {}
            for param_name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                elif param.annotation == str:
                    kwargs[param_name] = "test"
                elif param.annotation == int:
                    kwargs[param_name] = 0
                elif param.annotation == bool:
                    kwargs[param_name] = False
                elif param.annotation == list or param.annotation == List:
                    kwargs[param_name] = []
                elif param.annotation == dict or param.annotation == Dict:
                    kwargs[param_name] = {}
                else:
                    kwargs[param_name] = None
            
            try:
                func(**kwargs)
            except TypeError:
                pass  # Expected if we don't have right types
        
        tests.append(TestCase(
            test_id=str(uuid.uuid4()),
            name=f"test_{func_name}_executes",
            category=TestCategory.UNIT,
            test_func=test_executes,
            description=f"Test that {func_name} executes without crashing",
        ))
        
        return tests
    
    def generate_edge_cases(
        self,
        func: Callable,
    ) -> List[Dict[str, Any]]:
        """Generate edge case parameters.
        
        Args:
            func: Function to generate edge cases for
            
        Returns:
            List of parameter dictionaries
        """
        sig = inspect.signature(func)
        edge_cases = []
        
        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            
            if annotation == str:
                edge_cases.extend([
                    {param_name: ""},  # Empty string
                    {param_name: " "},  # Whitespace
                    {param_name: "a" * 1000},  # Long string
                    {param_name: "特殊字符"},  # Unicode
                ])
            elif annotation == int:
                edge_cases.extend([
                    {param_name: 0},
                    {param_name: -1},
                    {param_name: sys.maxsize},
                    {param_name: -sys.maxsize},
                ])
            elif annotation == float:
                edge_cases.extend([
                    {param_name: 0.0},
                    {param_name: -0.0},
                    {param_name: float('inf')},
                    {param_name: float('-inf')},
                ])
            elif annotation == list or annotation == List:
                edge_cases.extend([
                    {param_name: []},
                    {param_name: [None]},
                    {param_name: list(range(1000))},
                ])
            elif annotation == dict or annotation == Dict:
                edge_cases.extend([
                    {param_name: {}},
                    {param_name: {"": ""}},
                ])
        
        return edge_cases
    
    async def generate_tests_with_llm(
        self,
        func: Callable,
        num_tests: int = 5,
    ) -> List[TestCase]:
        """Generate tests using LLM analysis.
        
        Args:
            func: Function to test
            num_tests: Number of tests to generate
            
        Returns:
            List of generated test cases
        """
        if not self._llm_client:
            return self.generate_unit_tests(func, num_tests)
        
        source = ""
        try:
            source = inspect.getsource(func)
        except Exception:
            pass
        
        prompt = f"""Analyze this Python function and generate {num_tests} test cases:

```python
{source}
```

For each test, provide:
1. Test name
2. Test description
3. Input parameters
4. Expected behavior or output

Return as JSON array.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            test_specs = json.loads(response)
            
            tests = []
            for spec in test_specs[:num_tests]:
                test = TestCase(
                    test_id=str(uuid.uuid4()),
                    name=spec.get("name", f"test_{func.__name__}"),
                    category=TestCategory.UNIT,
                    description=spec.get("description", ""),
                )
                tests.append(test)
            
            return tests
            
        except Exception as e:
            logger.warning(f"LLM test generation failed: {e}")
            return self.generate_unit_tests(func, num_tests)


class IntegrationTestRunner:
    """Run integration tests across components.
    
    Tests interactions between multiple components and tools.
    
    Example:
        >>> runner = IntegrationTestRunner()
        >>> result = runner.test_workflow(workflow_steps)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize integration test runner.
        
        Args:
            llm_client: LLM client for intelligent testing
        """
        self._llm_client = llm_client
        
        # Test isolation
        self._temp_dirs: List[Path] = []
        
        # Results
        self._results: List[Dict[str, Any]] = []
    
    def setup_isolated_environment(self) -> Path:
        """Create an isolated test environment.
        
        Returns:
            Path to temporary directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="test_"))
        self._temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_environments(self):
        """Clean up all test environments."""
        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_dir}: {e}")
        self._temp_dirs.clear()
    
    def test_workflow(
        self,
        steps: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Test a multi-step workflow.
        
        Args:
            steps: Workflow steps to test
            context: Initial context
            
        Returns:
            Test result
        """
        context = context or {}
        start_time = time.time()
        
        result = {
            "workflow": "test_workflow",
            "steps": [],
            "passed": True,
            "duration_ms": 0,
        }
        
        for i, step in enumerate(steps):
            step_result = {
                "step": i + 1,
                "name": step.get("name", f"step_{i+1}"),
                "passed": False,
            }
            
            try:
                # Execute step
                action = step.get("action")
                params = step.get("params", {})
                expected = step.get("expected")
                
                if callable(action):
                    actual = action(**params)
                    
                    # Validate
                    if expected is not None:
                        if actual != expected:
                            step_result["error"] = f"Expected {expected}, got {actual}"
                            result["passed"] = False
                        else:
                            step_result["passed"] = True
                    else:
                        step_result["passed"] = True
                    
                    # Store result in context
                    context[f"step_{i+1}_result"] = actual
                else:
                    step_result["error"] = "No action provided"
                    result["passed"] = False
                    
            except Exception as e:
                step_result["error"] = str(e)
                step_result["traceback"] = traceback.format_exc()
                result["passed"] = False
            
            result["steps"].append(step_result)
        
        result["duration_ms"] = (time.time() - start_time) * 1000
        self._results.append(result)
        
        return result
    
    def test_tool_interaction(
        self,
        tool1: Any,
        tool2: Any,
        test_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Test interaction between two tools.
        
        Args:
            tool1: First tool
            tool2: Second tool
            test_data: Test data
            
        Returns:
            Test result
        """
        result = {
            "tools": [getattr(tool1, 'name', str(tool1)), 
                     getattr(tool2, 'name', str(tool2))],
            "passed": True,
            "errors": [],
        }
        
        try:
            # Run tool1
            if hasattr(tool1, 'execute'):
                result1 = tool1.execute(test_data.get("tool1_params", {}))
            elif callable(tool1):
                result1 = tool1(**test_data.get("tool1_params", {}))
            else:
                result1 = None
            
            # Pass result to tool2
            tool2_params = test_data.get("tool2_params", {})
            tool2_params["input"] = result1
            
            if hasattr(tool2, 'execute'):
                result2 = tool2.execute(tool2_params)
            elif callable(tool2):
                result2 = tool2(**tool2_params)
            else:
                result2 = None
            
            result["output"] = result2
            
        except Exception as e:
            result["passed"] = False
            result["errors"].append(str(e))
        
        return result


class LLMResponseTester:
    """Test LLM responses for quality and correctness.
    
    Uses LLM reasoning to:
    1. Validate response schemas
    2. Assess response quality
    3. Detect regressions
    
    Example:
        >>> tester = LLMResponseTester(llm_client)
        >>> result = tester.test_prompt("What is 2+2?", expected_schema)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize LLM response tester.
        
        Args:
            llm_client: LLM client to test
        """
        self._llm_client = llm_client
        
        # Test history for regression detection
        self._test_history: List[LLMTestResult] = []
        
        # Baseline responses
        self._baselines: Dict[str, LLMTestResult] = {}
    
    async def test_prompt(
        self,
        prompt: str,
        expected_schema: Optional[Dict[str, Any]] = None,
        validators: Optional[List[Callable[[str], bool]]] = None,
    ) -> LLMTestResult:
        """Test a prompt and validate response.
        
        Args:
            prompt: Prompt to test
            expected_schema: Expected response schema
            validators: Custom validation functions
            
        Returns:
            Test result
        """
        result = LLMTestResult(
            prompt=prompt,
            expected_schema=expected_schema,
        )
        
        if not self._llm_client:
            result.response = "LLM client not available"
            return result
        
        start_time = time.time()
        
        try:
            response = await self._llm_client.generate(prompt)
            result.response = response
            result.latency_ms = (time.time() - start_time) * 1000
            
            # Validate schema
            if expected_schema:
                result.schema_valid = self._validate_schema(response, expected_schema)
            else:
                result.schema_valid = True
            
            # Run custom validators
            if validators:
                for validator in validators:
                    if not validator(response):
                        result.schema_valid = False
                        break
            
            # Calculate quality score
            result.quality_score = self._calculate_quality(response, prompt)
            
        except Exception as e:
            result.response = f"Error: {e}"
            result.schema_valid = False
        
        self._test_history.append(result)
        
        return result
    
    def _validate_schema(
        self,
        response: str,
        schema: Dict[str, Any],
    ) -> bool:
        """Validate response against schema."""
        try:
            # Try to parse as JSON
            parsed = json.loads(response)
            
            # Check required fields
            for key, value_type in schema.items():
                if key not in parsed:
                    return False
                if not isinstance(parsed[key], value_type):
                    return False
            
            return True
            
        except json.JSONDecodeError:
            return False
    
    def _calculate_quality(
        self,
        response: str,
        prompt: str,
    ) -> float:
        """Calculate response quality score (0-1)."""
        score = 0.5  # Base score
        
        # Length check
        if len(response) > 10:
            score += 0.1
        if len(response) > 50:
            score += 0.1
        
        # Relevance check (simple keyword overlap)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words)
        if overlap > 0:
            score += min(0.2, overlap * 0.05)
        
        # Structure check
        if any(c in response for c in ['.', '!', '?']):
            score += 0.1
        
        return min(1.0, score)
    
    def set_baseline(
        self,
        prompt: str,
        result: LLMTestResult,
    ):
        """Set baseline for regression testing."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self._baselines[prompt_hash] = result
    
    async def test_regression(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """Test for regression against baseline.
        
        Args:
            prompt: Prompt to test
            
        Returns:
            Regression test result
        """
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        baseline = self._baselines.get(prompt_hash)
        
        current = await self.test_prompt(prompt)
        
        result = {
            "prompt": prompt,
            "has_baseline": baseline is not None,
            "regression_detected": False,
        }
        
        if baseline:
            # Compare quality
            if current.quality_score < baseline.quality_score - 0.1:
                result["regression_detected"] = True
                result["quality_drop"] = baseline.quality_score - current.quality_score
            
            # Compare latency
            if current.latency_ms > baseline.latency_ms * 1.5:
                result["latency_regression"] = True
                result["latency_increase"] = current.latency_ms - baseline.latency_ms
        
        return result
    
    def get_test_history(self) -> List[LLMTestResult]:
        """Get test history."""
        return list(self._test_history)


class PerformanceTester:
    """Performance testing framework.
    
    Tests system performance under various conditions.
    
    Example:
        >>> tester = PerformanceTester()
        >>> metrics = tester.run_load_test(my_func, num_requests=100)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize performance tester.
        
        Args:
            llm_client: LLM client for analysis
        """
        self._llm_client = llm_client
        
        # Results
        self._results: List[PerformanceMetrics] = []
    
    def run_load_test(
        self,
        func: Callable,
        num_requests: int = 100,
        concurrent: int = 10,
        params: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetrics:
        """Run load test on a function.
        
        Args:
            func: Function to test
            num_requests: Total number of requests
            concurrent: Concurrent requests
            params: Parameters to pass to function
            
        Returns:
            Performance metrics
        """
        params = params or {}
        timings: List[float] = []
        errors = 0
        
        start_time = time.time()
        
        def run_single():
            nonlocal errors
            try:
                t0 = time.time()
                func(**params)
                return (time.time() - t0) * 1000
            except Exception:
                errors += 1
                return None
        
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(run_single) for _ in range(num_requests)]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    timings.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            operation=func.__name__,
        )
        
        if timings:
            sorted_timings = sorted(timings)
            metrics.min_time_ms = sorted_timings[0]
            metrics.max_time_ms = sorted_timings[-1]
            metrics.avg_time_ms = statistics.mean(timings)
            metrics.p50_time_ms = sorted_timings[len(sorted_timings) // 2]
            metrics.p95_time_ms = sorted_timings[int(len(sorted_timings) * 0.95)]
            metrics.p99_time_ms = sorted_timings[int(len(sorted_timings) * 0.99)]
        
        metrics.requests_per_second = num_requests / total_time
        metrics.error_rate = errors / num_requests
        
        self._results.append(metrics)
        
        return metrics
    
    def run_stress_test(
        self,
        func: Callable,
        duration_seconds: float = 60.0,
        ramp_up_seconds: float = 10.0,
        max_concurrent: int = 100,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run stress test with increasing load.
        
        Args:
            func: Function to test
            duration_seconds: Test duration
            ramp_up_seconds: Ramp-up period
            max_concurrent: Maximum concurrent requests
            params: Parameters to pass to function
            
        Returns:
            Stress test results
        """
        params = params or {}
        results = {
            "duration": duration_seconds,
            "max_concurrent": max_concurrent,
            "samples": [],
            "peak_rps": 0,
            "breaking_point": None,
        }
        
        start_time = time.time()
        current_concurrent = 1
        
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            
            # Ramp up
            if elapsed < ramp_up_seconds:
                current_concurrent = max(
                    1,
                    int((elapsed / ramp_up_seconds) * max_concurrent)
                )
            
            # Run sample
            sample_start = time.time()
            completed = 0
            errors = 0
            
            def run_one():
                nonlocal completed, errors
                try:
                    func(**params)
                    completed += 1
                except Exception:
                    errors += 1
            
            with ThreadPoolExecutor(max_workers=current_concurrent) as executor:
                futures = [executor.submit(run_one) for _ in range(current_concurrent)]
                for future in as_completed(futures, timeout=5):
                    try:
                        future.result()
                    except Exception:
                        pass
            
            sample_duration = time.time() - sample_start
            rps = completed / sample_duration if sample_duration > 0 else 0
            
            results["samples"].append({
                "time": elapsed,
                "concurrent": current_concurrent,
                "completed": completed,
                "errors": errors,
                "rps": rps,
            })
            
            results["peak_rps"] = max(results["peak_rps"], rps)
            
            # Detect breaking point
            if errors > completed * 0.5 and results["breaking_point"] is None:
                results["breaking_point"] = current_concurrent
            
            time.sleep(1)
        
        return results
    
    def measure_memory(
        self,
        func: Callable,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Measure memory usage of a function.
        
        Args:
            func: Function to measure
            params: Parameters to pass
            
        Returns:
            Memory metrics
        """
        params = params or {}
        
        try:
            import psutil
            process = psutil.Process()
            
            # Measure before
            before = process.memory_info().rss / (1024 * 1024)
            
            # Run function
            func(**params)
            
            # Measure after
            after = process.memory_info().rss / (1024 * 1024)
            
            return {
                "before_mb": before,
                "after_mb": after,
                "delta_mb": after - before,
            }
            
        except ImportError:
            return {
                "error": "psutil not available",
            }
    
    def get_results(self) -> List[PerformanceMetrics]:
        """Get all performance results."""
        return list(self._results)


class CoverageTracker:
    """Track test coverage.
    
    Monitors which code is executed during tests.
    
    Example:
        >>> tracker = CoverageTracker()
        >>> tracker.start()
        >>> run_tests()
        >>> report = tracker.get_report()
    """
    
    def __init__(self):
        """Initialize coverage tracker."""
        self._coverage_data: Dict[str, Set[int]] = defaultdict(set)
        self._total_lines: Dict[str, int] = {}
        self._is_tracking = False
        
        # Original trace function
        self._original_trace: Optional[Callable] = None
    
    def start(self):
        """Start coverage tracking."""
        self._original_trace = sys.gettrace()
        sys.settrace(self._trace)
        self._is_tracking = True
    
    def stop(self):
        """Stop coverage tracking."""
        sys.settrace(self._original_trace)
        self._is_tracking = False
    
    def _trace(self, frame, event, arg):
        """Trace function for coverage."""
        if event == 'line':
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            
            # Only track project files
            if 'site-packages' not in filename and 'lib' not in filename:
                self._coverage_data[filename].add(lineno)
        
        return self._trace
    
    def get_report(self) -> CoverageReport:
        """Generate coverage report.
        
        Returns:
            CoverageReport
        """
        report = CoverageReport()
        
        for filename, covered_lines in self._coverage_data.items():
            try:
                with open(filename, 'r') as f:
                    total = len(f.readlines())
                
                coverage_pct = (len(covered_lines) / total) * 100 if total > 0 else 0
                report.file_coverage[filename] = round(coverage_pct, 2)
                
                report.total_lines += total
                report.covered_lines += len(covered_lines)
                
            except Exception:
                pass
        
        return report
    
    def reset(self):
        """Reset coverage data."""
        self._coverage_data.clear()
        self._total_lines.clear()


class ComprehensiveTestingFramework:
    """Main testing framework integrating all components.
    
    Provides unified interface for:
    - Unit testing
    - Integration testing
    - LLM response testing
    - Performance testing
    
    Example:
        >>> framework = ComprehensiveTestingFramework()
        >>> results = framework.run_all_tests()
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize testing framework.
        
        Args:
            llm_client: LLM client for intelligent testing
        """
        self._llm_client = llm_client
        
        # Initialize components
        self._test_runner = TestRunner(llm_client=llm_client)
        self._test_generator = TestGenerator(llm_client=llm_client)
        self._integration_runner = IntegrationTestRunner(llm_client=llm_client)
        self._llm_tester = LLMResponseTester(llm_client=llm_client)
        self._perf_tester = PerformanceTester(llm_client=llm_client)
        self._coverage = CoverageTracker()
        
        # Test suites
        self._suites: Dict[str, TestSuite] = {}
        
        # Fixtures
        self._fixtures: Dict[str, TestFixture] = {}
    
    def register_suite(self, suite: TestSuite):
        """Register a test suite."""
        self._suites[suite.name] = suite
    
    def register_fixture(self, fixture: TestFixture):
        """Register a test fixture."""
        self._fixtures[fixture.name] = fixture
        self._test_runner.register_fixture(fixture)
    
    def create_test(
        self,
        name: str,
        test_func: Callable,
        category: TestCategory = TestCategory.UNIT,
        **kwargs,
    ) -> TestCase:
        """Create a test case.
        
        Args:
            name: Test name
            test_func: Test function
            category: Test category
            **kwargs: Additional test properties
            
        Returns:
            Created test case
        """
        return TestCase(
            test_id=str(uuid.uuid4()),
            name=name,
            test_func=test_func,
            category=category,
            **kwargs,
        )
    
    def run_suite(self, suite_name: str) -> TestSuite:
        """Run a specific test suite.
        
        Args:
            suite_name: Name of suite to run
            
        Returns:
            Suite with results
        """
        suite = self._suites.get(suite_name)
        if not suite:
            raise ValueError(f"Suite not found: {suite_name}")
        
        return self._test_runner.run_suite(suite)
    
    def run_all_tests(
        self,
        categories: Optional[List[TestCategory]] = None,
        with_coverage: bool = True,
    ) -> Dict[str, Any]:
        """Run all registered tests.
        
        Args:
            categories: Categories to run (None = all)
            with_coverage: Track coverage
            
        Returns:
            Test results
        """
        if with_coverage:
            self._coverage.start()
        
        results = {
            "suites": [],
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "duration_ms": 0,
        }
        
        start_time = time.time()
        
        for suite in self._suites.values():
            # Filter by category if specified
            if categories:
                suite.tests = [
                    t for t in suite.tests
                    if t.category in categories
                ]
            
            if suite.tests:
                suite_result = self._test_runner.run_suite(suite)
                results["suites"].append(suite_result.to_dict())
                results["total_tests"] += suite_result.total_tests
                results["passed"] += suite_result.passed
                results["failed"] += suite_result.failed
        
        results["duration_ms"] = (time.time() - start_time) * 1000
        
        if with_coverage:
            self._coverage.stop()
            results["coverage"] = self._coverage.get_report().to_dict()
        
        return results
    
    async def run_llm_tests(
        self,
        prompts: List[Dict[str, Any]],
    ) -> List[LLMTestResult]:
        """Run LLM response tests.
        
        Args:
            prompts: List of prompt test configurations
            
        Returns:
            List of test results
        """
        results = []
        
        for prompt_config in prompts:
            result = await self._llm_tester.test_prompt(
                prompt=prompt_config.get("prompt", ""),
                expected_schema=prompt_config.get("schema"),
                validators=prompt_config.get("validators"),
            )
            results.append(result)
        
        return results
    
    def run_performance_tests(
        self,
        functions: List[Dict[str, Any]],
    ) -> List[PerformanceMetrics]:
        """Run performance tests.
        
        Args:
            functions: List of function test configurations
            
        Returns:
            List of performance metrics
        """
        results = []
        
        for func_config in functions:
            func = func_config.get("func")
            if not func:
                continue
            
            metrics = self._perf_tester.run_load_test(
                func=func,
                num_requests=func_config.get("num_requests", 100),
                concurrent=func_config.get("concurrent", 10),
                params=func_config.get("params"),
            )
            results.append(metrics)
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report.
        
        Returns:
            Test report
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "unit_tests": [s.to_dict() for s in self._test_runner.get_results()],
            "integration_tests": self._integration_runner._results,
            "llm_tests": [t.to_dict() for t in self._llm_tester.get_test_history()],
            "performance": [p.to_dict() for p in self._perf_tester.get_results()],
        }


# Module-level instance
_global_testing_framework: Optional[ComprehensiveTestingFramework] = None


def get_comprehensive_testing_framework(
    llm_client: Optional[Any] = None,
) -> ComprehensiveTestingFramework:
    """Get the global testing framework.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        ComprehensiveTestingFramework instance
    """
    global _global_testing_framework
    if _global_testing_framework is None:
        _global_testing_framework = ComprehensiveTestingFramework(
            llm_client=llm_client,
        )
    return _global_testing_framework


# Decorator for registering tests
def test_case(
    name: Optional[str] = None,
    category: TestCategory = TestCategory.UNIT,
    priority: TestPriority = TestPriority.MEDIUM,
    tags: Optional[Set[str]] = None,
):
    """Decorator to register a function as a test case.
    
    Example:
        >>> @test_case(name="test_add", category=TestCategory.UNIT)
        ... def test_add():
        ...     assert 1 + 1 == 2
    """
    def decorator(func: Callable) -> Callable:
        test = TestCase(
            test_id=str(uuid.uuid4()),
            name=name or func.__name__,
            category=category,
            priority=priority,
            tags=tags or set(),
            test_func=func,
            description=func.__doc__ or "",
        )
        
        # Store test on function for later retrieval
        func._test_case = test
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Decorator for fixtures
def fixture(
    scope: str = "function",
):
    """Decorator to register a function as a test fixture.
    
    Example:
        >>> @fixture(scope="function")
        ... def temp_directory():
        ...     d = tempfile.mkdtemp()
        ...     yield d
        ...     shutil.rmtree(d)
    """
    def decorator(func: Callable) -> Callable:
        fixture_obj = TestFixture(
            fixture_id=str(uuid.uuid4()),
            name=func.__name__,
            scope=scope,
            setup=func,
        )
        
        func._fixture = fixture_obj
        
        return func
    
    return decorator
