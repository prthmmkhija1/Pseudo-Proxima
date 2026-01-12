"""Step 6.2: Testing Strategy - Pytest Configuration and Fixtures.

Test Pyramid:

           E2E       10%


        Integration  30%


          Unit           60%


Test Categories:
| Category    | Focus                 | Tools              |
| Unit        | Individual functions  | pytest, mock       |
| Integration | Component interaction | pytest, fixtures   |
| Backend     | Adapter functionality | Mock backends      |
| E2E         | Full workflows        | pytest, CLI runner |
| Performance | Resource usage        | pytest-benchmark   |
"""

import asyncio
import sys
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import comprehensive mock strategies
from tests.mocks import (
    MockBackendRegistry,
    MockConsentManager,
    MockDataFactory,
    MockFileSystem,
    MockLLMProvider,
    MockPipelineHandler,
    MockQuantumBackend,
    create_high_cpu_scenario,
    create_low_disk_scenario,
    create_low_memory_scenario,
    mock_psutil,
)

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (component interaction)"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests (full workflows)")
    config.addinivalue_line("markers", "backend: Backend adapter tests")
    config.addinivalue_line("markers", "performance: Performance/benchmark tests")
    config.addinivalue_line("markers", "slow: Slow tests (may take > 1s)")
    config.addinivalue_line(
        "markers", "requires_network: Tests requiring network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    run_all = config.getoption("--run-all", default=False)

    skip_slow = pytest.mark.skip(reason="Skipping slow tests (use --run-slow)")
    skip_network = pytest.mark.skip(reason="Skipping network tests (use --run-network)")

    for item in items:
        if not run_all:
            if "slow" in item.keywords and not config.getoption(
                "--run-slow", default=False
            ):
                item.add_marker(skip_slow)
            if "requires_network" in item.keywords and not config.getoption(
                "--run-network", default=False
            ):
                item.add_marker(skip_network)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--run-network", action="store_true", default=False, help="Run network tests"
    )
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all tests including slow/network",
    )


# =============================================================================
# COMMON FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def src_path(project_root) -> Path:
    """Get the src directory path."""
    return project_root / "src"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir) -> Generator[Path, None, None]:
    """Provide a temporary file path."""
    file_path = temp_dir / "test_file.txt"
    yield file_path


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Provide sample configuration for tests."""
    return {
        "version": "1.0.0",
        "log_level": "INFO",
        "output_dir": "./results",
        "backends": {
            "default": "local",
            "timeout_s": 300,
            "max_workers": 4,
        },
        "execution": {
            "parallel": False,
            "dry_run": False,
            "require_consent": True,
        },
        "export": {
            "format": "json",
            "include_metadata": True,
            "pretty_print": True,
        },
    }


@pytest.fixture
def sample_circuit_data() -> dict[str, Any]:
    """Provide sample quantum circuit data for tests."""
    return {
        "name": "test_circuit",
        "num_qubits": 2,
        "depth": 3,
        "gates": [
            {"name": "h", "qubits": [0]},
            {"name": "cx", "qubits": [0, 1]},
            {"name": "measure", "qubits": [0, 1]},
        ],
    }


@pytest.fixture
def sample_execution_result() -> dict[str, Any]:
    """Provide sample execution result for tests."""
    return {
        "id": "exec-001",
        "backend": "local",
        "status": "success",
        "duration_ms": 45.2,
        "timestamp": time.time(),
        "counts": {"00": 512, "11": 512},
        "metadata": {
            "shots": 1024,
            "seed": 42,
        },
    }


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_backend():
    """Provide a mock backend for testing."""
    backend = MagicMock()
    backend.name = "mock_backend"
    backend.backend_type = "simulator"
    backend.is_connected.return_value = True
    backend.execute = AsyncMock(
        return_value={
            "status": "success",
            "counts": {"00": 512, "11": 512},
            "duration_ms": 10.5,
        }
    )
    return backend


@pytest.fixture
def mock_executor():
    """Provide a mock executor for testing."""
    executor = MagicMock()
    executor.run = AsyncMock(
        return_value={
            "success": True,
            "results": [{"id": "1", "status": "completed"}],
        }
    )
    return executor


@pytest.fixture
def mock_config_manager():
    """Provide a mock configuration manager."""
    manager = MagicMock()
    manager.get.return_value = "test_value"
    manager.set.return_value = True
    manager.load.return_value = {"version": "1.0.0"}
    manager.save.return_value = True
    return manager


@pytest.fixture
def mock_consent_manager():
    """Provide a mock consent manager."""
    manager = MagicMock()
    manager.request_consent.return_value = True
    manager.is_sensitive_operation.return_value = False
    return manager


@pytest.fixture
def mock_file_system(temp_dir):
    """Provide mock file system utilities."""
    fs = MagicMock()
    fs.read_file.return_value = "file contents"
    fs.write_file.return_value = True
    fs.exists.return_value = True
    fs.temp_dir = temp_dir
    return fs


# =============================================================================
# ASYNC FIXTURES
# =============================================================================


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_mock_backend():
    """Provide an async mock backend."""
    backend = AsyncMock()
    backend.name = "async_mock"
    backend.connect = AsyncMock(return_value=True)
    backend.disconnect = AsyncMock(return_value=True)
    backend.execute = AsyncMock(return_value={"status": "success"})
    return backend


# =============================================================================
# DATA FIXTURES
# =============================================================================


@dataclass
class MockReportData:
    """Mock report data for testing exports."""

    title: str = "Test Report"
    summary: dict[str, Any] = field(default_factory=lambda: {"total": 10, "passed": 8})
    raw_results: list[dict] = field(default_factory=list)
    comparison: dict[str, Any] = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "raw_results": self.raw_results,
            "comparison": self.comparison,
            "insights": self.insights,
            "metadata": self.metadata,
        }


@pytest.fixture
def mock_report_data() -> MockReportData:
    """Provide mock report data."""
    return MockReportData(
        title="Test Execution Report",
        summary={"backends": 3, "passed": 2, "failed": 1},
        raw_results=[
            {"id": "1", "backend": "local", "status": "success", "duration_ms": 10},
            {"id": "2", "backend": "qiskit", "status": "success", "duration_ms": 20},
            {"id": "3", "backend": "ibm", "status": "failed", "duration_ms": 0},
        ],
        comparison={"winner": "local", "speedup": 2.0},
        insights=["Local backend is fastest", "IBM backend failed"],
        metadata={"version": "1.0.0", "timestamp": time.time()},
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


class TestHelper:
    """Helper utilities for tests."""

    @staticmethod
    def create_temp_config(temp_dir: Path, content: dict[str, Any]) -> Path:
        """Create a temporary config file."""
        import json

        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps(content, indent=2))
        return config_path

    @staticmethod
    def create_temp_agent_file(temp_dir: Path, content: str) -> Path:
        """Create a temporary agent.md file."""
        agent_path = temp_dir / "agent.md"
        agent_path.write_text(content)
        return agent_path

    @staticmethod
    def wait_for_condition(
        condition_fn, timeout: float = 5.0, interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true."""
        start = time.time()
        while time.time() - start < timeout:
            if condition_fn():
                return True
            time.sleep(interval)
        return False

    @staticmethod
    def assert_dict_subset(subset: dict, full: dict) -> None:
        """Assert that subset is contained in full dict."""
        for key, value in subset.items():
            assert key in full, f"Key '{key}' not found in dict"
            assert (
                full[key] == value
            ), f"Value mismatch for key '{key}': {full[key]} != {value}"


@pytest.fixture
def test_helper() -> TestHelper:
    """Provide test helper utilities."""
    return TestHelper()


# =============================================================================
# PERFORMANCE FIXTURES
# =============================================================================


@pytest.fixture
def benchmark_data():
    """Provide data for benchmark tests."""
    return {
        "small": list(range(100)),
        "medium": list(range(10000)),
        "large": list(range(100000)),
    }


@pytest.fixture
def timing():
    """Provide timing context manager."""

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed_ms = 0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed_ms = (self.end_time - self.start_time) * 1000

    return Timer


# =============================================================================
# CLI FIXTURES FOR E2E TESTS
# =============================================================================


@pytest.fixture
def cli_runner():
    """Provide a CLI test runner."""
    try:
        from click.testing import CliRunner

        return CliRunner()
    except ImportError:
        pytest.skip("click not installed")


# =============================================================================
# COMPREHENSIVE MOCK FIXTURES (from tests/mocks.py)
# =============================================================================


@pytest.fixture
def mock_quantum_backend() -> MockQuantumBackend:
    """Provide a comprehensive mock quantum backend."""
    return MockQuantumBackend()


@pytest.fixture
def mock_backend_registry() -> MockBackendRegistry:
    """Provide a mock backend registry with all backends."""
    return MockBackendRegistry()


@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    """Provide a mock LLM provider with canned responses."""
    return MockLLMProvider()


@pytest.fixture
def mock_data_factory() -> MockDataFactory:
    """Provide mock data factory for quantum results."""
    return MockDataFactory()


@pytest.fixture
def mock_psutil_normal():
    """Provide mock psutil with normal resource values."""
    with mock_psutil() as m:
        yield m


@pytest.fixture
def mock_psutil_low_memory():
    """Provide mock psutil simulating low memory scenario."""
    with mock_psutil(create_low_memory_scenario()) as m:
        yield m


@pytest.fixture
def mock_psutil_high_cpu():
    """Provide mock psutil simulating high CPU scenario."""
    with mock_psutil(create_high_cpu_scenario()) as m:
        yield m


@pytest.fixture
def mock_psutil_low_disk():
    """Provide mock psutil simulating low disk space scenario."""
    with mock_psutil(create_low_disk_scenario()) as m:
        yield m


@pytest.fixture
def mock_file_system_comprehensive() -> MockFileSystem:
    """Provide comprehensive mock file system."""
    return MockFileSystem()


@pytest.fixture
def mock_consent_auto_approve() -> MockConsentManager:
    """Provide mock consent manager that auto-approves."""
    return MockConsentManager(auto_approve=True)


@pytest.fixture
def mock_consent_deny() -> MockConsentManager:
    """Provide mock consent manager that denies consent."""
    return MockConsentManager(auto_approve=False)


@pytest.fixture
def mock_consent_with_sensitive_ops() -> MockConsentManager:
    """Provide mock consent manager with sensitive operations defined."""
    return MockConsentManager(
        auto_approve=True, require_consent_for=["send_to_remote_llm", "export_results"]
    )


@pytest.fixture
def mock_pipeline_handler() -> MockPipelineHandler:
    """Provide mock pipeline handler for testing stages."""
    return MockPipelineHandler("test_stage")


@pytest.fixture
def bell_state_counts() -> dict[str, int]:
    """Provide Bell state measurement counts."""
    return MockDataFactory.bell_state_counts()


@pytest.fixture
def ghz_state_counts() -> dict[str, int]:
    """Provide GHZ state measurement counts (3 qubits)."""
    return MockDataFactory.ghz_state_counts(num_qubits=3)


@pytest.fixture
def random_state_vector():
    """Provide a random normalized state vector."""
    return MockDataFactory.state_vector(num_qubits=2)


@pytest.fixture
def random_density_matrix():
    """Provide a random valid density matrix."""
    return MockDataFactory.density_matrix(num_qubits=2)
