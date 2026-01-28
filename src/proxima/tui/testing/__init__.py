"""Proxima TUI Testing Package.

Provides comprehensive testing capabilities for backend validation.
Part of Phase 4: Testing & Validation Interface.

Components:
- TestExecutor: Runs tests on generated backend code
- TestCircuits: Library of test circuits (Bell, GHZ, QFT, etc.)
- TestReporter: Formats and displays test results
- TestSuite: Collection of test cases
"""

from .test_executor import (
    TestExecutor,
    TestResult,
    TestCategory,
    TestStatus,
)

from .test_circuits import (
    TestCircuitLibrary,
    TestCircuit,
    CircuitType,
)

from .test_reporter import (
    TestReporter,
    TestReport,
    TestSummary,
)

__all__ = [
    # Test Executor
    "TestExecutor",
    "TestResult",
    "TestCategory",
    "TestStatus",
    # Test Circuits
    "TestCircuitLibrary",
    "TestCircuit",
    "CircuitType",
    # Test Reporter
    "TestReporter",
    "TestReport",
    "TestSummary",
]
