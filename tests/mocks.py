"""
Comprehensive Mock Strategies for Testing
==========================================

Mock Strategies (from proper_implementation_steps.md):

| Component        | Mock Approach                         |
| ---------------- | ------------------------------------- |
| Quantum backends | Mock adapter returning canned results |
| LLM providers    | Mock HTTP responses                   |
| System resources | Inject fake psutil values             |
| File system      | Use tmpdir fixture                    |

This module provides reusable mock objects and factories for testing
all external integrations and system components.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# =============================================================================
# MOCK DATA FACTORIES
# =============================================================================


class MockDataFactory:
    """Factory for generating consistent mock data."""

    @staticmethod
    def quantum_counts(
        num_qubits: int = 2,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Generate realistic quantum measurement counts."""
        if seed is not None:
            np.random.seed(seed)

        # Generate all possible states
        states = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]

        # Generate random probabilities
        probs = np.random.dirichlet(np.ones(len(states)))

        # Convert to counts
        counts = {}
        remaining = shots
        for state, prob in zip(states[:-1], probs[:-1], strict=False):
            count = int(prob * shots)
            if count > 0:
                counts[state] = count
            remaining -= count

        if remaining > 0:
            counts[states[-1]] = remaining

        return counts

    @staticmethod
    def bell_state_counts(shots: int = 1024) -> dict[str, int]:
        """Generate Bell state measurement counts (|00⟩ + |11⟩)."""
        half = shots // 2
        return {"00": half, "11": shots - half}

    @staticmethod
    def ghz_state_counts(num_qubits: int = 3, shots: int = 1024) -> dict[str, int]:
        """Generate GHZ state measurement counts."""
        all_zeros = "0" * num_qubits
        all_ones = "1" * num_qubits
        half = shots // 2
        return {all_zeros: half, all_ones: shots - half}

    @staticmethod
    def state_vector(num_qubits: int = 2, normalized: bool = True) -> np.ndarray:
        """Generate a random state vector."""
        dim = 2**num_qubits
        real = np.random.randn(dim)
        imag = np.random.randn(dim)
        sv = real + 1j * imag

        if normalized:
            sv = sv / np.linalg.norm(sv)

        return sv

    @staticmethod
    def density_matrix(num_qubits: int = 2) -> np.ndarray:
        """Generate a random valid density matrix."""
        2**num_qubits
        # Create a random pure state and compute its density matrix
        sv = MockDataFactory.state_vector(num_qubits)
        rho = np.outer(sv, np.conj(sv))
        return rho

    @staticmethod
    def execution_result(
        backend: str = "mock",
        success: bool = True,
        num_qubits: int = 2,
        shots: int = 1024,
    ) -> dict[str, Any]:
        """Generate a mock execution result."""
        return {
            "backend": backend,
            "status": "success" if success else "failed",
            "counts": (
                MockDataFactory.quantum_counts(num_qubits, shots) if success else {}
            ),
            "duration_ms": np.random.uniform(10, 100),
            "num_qubits": num_qubits,
            "shots": shots,
            "timestamp": time.time(),
            "metadata": {
                "simulator_type": "state_vector",
                "seed": 42,
            },
        }


# =============================================================================
# QUANTUM BACKEND MOCKS
# =============================================================================


class MockQuantumBackend:
    """
    Mock quantum backend for testing.

    Provides canned results that can be configured for different test scenarios.
    """

    def __init__(
        self,
        name: str = "mock_backend",
        success_rate: float = 1.0,
        latency_ms: float = 10.0,
        num_qubits: int = 2,
    ):
        self.name = name
        self.success_rate = success_rate
        self.latency_ms = latency_ms
        self.num_qubits = num_qubits
        self._execute_count = 0
        self._connected = False

    async def connect(self) -> bool:
        """Simulate connection."""
        await asyncio.sleep(0.01)
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        """Simulate disconnection."""
        self._connected = False
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def execute(
        self,
        circuit: Any = None,
        shots: int = 1024,
        **options: Any,
    ) -> dict[str, Any]:
        """Execute a mock quantum circuit."""
        self._execute_count += 1

        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Simulate success/failure based on success_rate
        success = np.random.random() < self.success_rate

        if not success:
            return {
                "backend": self.name,
                "status": "failed",
                "error": "Simulated failure",
                "counts": {},
                "duration_ms": self.latency_ms,
            }

        return MockDataFactory.execution_result(
            backend=self.name,
            success=True,
            num_qubits=self.num_qubits,
            shots=shots,
        )

    def get_capabilities(self) -> dict[str, Any]:
        """Get mock backend capabilities."""
        return {
            "name": self.name,
            "max_qubits": 32,
            "simulator_types": ["state_vector", "density_matrix"],
            "supports_noise": True,
            "supports_gpu": False,
        }

    def reset(self) -> None:
        """Reset execution count."""
        self._execute_count = 0


class MockBackendRegistry:
    """Mock backend registry for testing."""

    def __init__(self):
        self._backends: dict[str, MockQuantumBackend] = {}
        self._setup_default_backends()

    def _setup_default_backends(self) -> None:
        """Set up default mock backends."""
        self._backends = {
            "cirq": MockQuantumBackend("cirq", latency_ms=15),
            "qiskit-aer": MockQuantumBackend("qiskit-aer", latency_ms=20),
            "lret": MockQuantumBackend("lret", latency_ms=10),
        }

    def get(self, name: str) -> MockQuantumBackend | None:
        return self._backends.get(name)

    def list_available(self) -> list[str]:
        return list(self._backends.keys())

    def register(self, backend: MockQuantumBackend) -> None:
        self._backends[backend.name] = backend


# =============================================================================
# LLM PROVIDER MOCKS
# =============================================================================


@dataclass
class MockLLMResponse:
    """Mock LLM response structure."""

    content: str
    model: str = "mock-model"
    tokens_used: int = 100
    finish_reason: str = "stop"
    latency_ms: float = 50.0


class MockLLMProvider:
    """
    Mock LLM provider for testing.

    Simulates HTTP responses from LLM providers.
    """

    # Canned responses for common queries
    CANNED_RESPONSES = {
        "explain": "This quantum circuit demonstrates entanglement between qubits.",
        "analyze": "The results show a Bell state with near-equal superposition.",
        "suggest": "Consider using a density matrix simulator for noise modeling.",
        "compare": "Backend A is 2x faster but Backend B has higher fidelity.",
        "default": "I analyzed your quantum simulation request.",
    }

    def __init__(
        self,
        name: str = "mock_llm",
        latency_ms: float = 50.0,
        success_rate: float = 1.0,
    ):
        self.name = name
        self.latency_ms = latency_ms
        self.success_rate = success_rate
        self._request_count = 0
        self._request_history: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        model: str = "mock-model",
        **options: Any,
    ) -> MockLLMResponse:
        """Generate a mock LLM response."""
        self._request_count += 1
        self._request_history.append(
            {
                "prompt": prompt,
                "model": model,
                "options": options,
                "timestamp": time.time(),
            }
        )

        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Simulate failure
        if np.random.random() > self.success_rate:
            raise Exception("Mock LLM provider error")

        # Select canned response based on prompt keywords
        response_key = "default"
        for key in self.CANNED_RESPONSES:
            if key in prompt.lower():
                response_key = key
                break

        return MockLLMResponse(
            content=self.CANNED_RESPONSES[response_key],
            model=model,
            tokens_used=len(prompt.split()) + 50,
            latency_ms=self.latency_ms,
        )

    def get_request_history(self) -> list[dict[str, Any]]:
        """Get history of requests for assertions."""
        return self._request_history

    def reset(self) -> None:
        """Reset request tracking."""
        self._request_count = 0
        self._request_history = []


def create_mock_httpx_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str = "",
) -> MagicMock:
    """Create a mock httpx response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data or {}
    response.text = text
    response.is_success = 200 <= status_code < 300
    return response


@contextmanager
def mock_llm_http_responses(responses: list[dict[str, Any]]) -> Generator:
    """
    Context manager to mock HTTP responses for LLM providers.

    Usage:
        responses = [{"content": "test response"}]
        with mock_llm_http_responses(responses):
            result = await llm.generate("test")
    """
    response_iter = iter(responses)

    async def mock_post(*args, **kwargs):
        try:
            resp_data = next(response_iter)
            return create_mock_httpx_response(json_data=resp_data)
        except StopIteration:
            return create_mock_httpx_response(status_code=500)

    with patch("httpx.AsyncClient.post", new=mock_post):
        yield


# =============================================================================
# SYSTEM RESOURCE MOCKS
# =============================================================================


@dataclass
class MockSystemResources:
    """Mock system resource values."""

    total_memory_mb: int = 16384
    available_memory_mb: int = 8192
    memory_percent: float = 50.0
    cpu_count: int = 8
    cpu_percent: float = 25.0
    disk_total_gb: float = 500.0
    disk_free_gb: float = 250.0


class MockPsutil:
    """
    Mock psutil module for testing resource monitoring.

    Allows injecting specific resource values for testing different scenarios.
    """

    def __init__(self, resources: MockSystemResources | None = None):
        self.resources = resources or MockSystemResources()

    def virtual_memory(self) -> MagicMock:
        """Return mock virtual memory stats."""
        mem = MagicMock()
        mem.total = self.resources.total_memory_mb * 1024 * 1024
        mem.available = self.resources.available_memory_mb * 1024 * 1024
        mem.percent = self.resources.memory_percent
        mem.used = mem.total - mem.available
        return mem

    def cpu_percent(self, interval: float = 0.1) -> float:
        """Return mock CPU percent."""
        return self.resources.cpu_percent

    def cpu_count(self) -> int:
        """Return mock CPU count."""
        return self.resources.cpu_count

    def disk_usage(self, path: str = "/") -> MagicMock:
        """Return mock disk usage."""
        disk = MagicMock()
        disk.total = int(self.resources.disk_total_gb * 1024**3)
        disk.free = int(self.resources.disk_free_gb * 1024**3)
        disk.used = disk.total - disk.free
        disk.percent = (disk.used / disk.total) * 100
        return disk


@contextmanager
def mock_psutil(
    resources: MockSystemResources | None = None,
) -> Generator[MockPsutil, None, None]:
    """
    Context manager to mock psutil with specific resource values.

    Usage:
        resources = MockSystemResources(memory_percent=95.0)
        with mock_psutil(resources) as mock:
            # Test low memory scenario
            result = check_resources()
    """
    mock = MockPsutil(resources)

    with patch.dict("sys.modules", {"psutil": mock}):
        with patch("psutil.virtual_memory", mock.virtual_memory):
            with patch("psutil.cpu_percent", mock.cpu_percent):
                with patch("psutil.cpu_count", mock.cpu_count):
                    with patch("psutil.disk_usage", mock.disk_usage):
                        yield mock


def create_low_memory_scenario() -> MockSystemResources:
    """Create a low memory test scenario."""
    return MockSystemResources(
        available_memory_mb=512,
        memory_percent=95.0,
    )


def create_high_cpu_scenario() -> MockSystemResources:
    """Create a high CPU usage test scenario."""
    return MockSystemResources(
        cpu_percent=95.0,
    )


def create_low_disk_scenario() -> MockSystemResources:
    """Create a low disk space test scenario."""
    return MockSystemResources(
        disk_free_gb=5.0,
    )


# =============================================================================
# FILE SYSTEM MOCKS
# =============================================================================


class MockFileSystem:
    """
    Mock file system for testing file operations.

    Uses in-memory storage instead of real filesystem.
    """

    def __init__(self):
        self._files: dict[str, str | bytes] = {}
        self._directories: set = {"/"}

    def write_text(self, path: str, content: str) -> None:
        """Write text content to mock file."""
        self._ensure_parent_dir(path)
        self._files[path] = content

    def write_bytes(self, path: str, content: bytes) -> None:
        """Write binary content to mock file."""
        self._ensure_parent_dir(path)
        self._files[path] = content

    def read_text(self, path: str) -> str:
        """Read text content from mock file."""
        if path not in self._files:
            raise FileNotFoundError(f"No such file: {path}")
        content = self._files[path]
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return content

    def read_bytes(self, path: str) -> bytes:
        """Read binary content from mock file."""
        if path not in self._files:
            raise FileNotFoundError(f"No such file: {path}")
        content = self._files[path]
        if isinstance(content, str):
            return content.encode("utf-8")
        return content

    def exists(self, path: str) -> bool:
        """Check if file or directory exists."""
        return path in self._files or path in self._directories

    def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        return path in self._files

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        return path in self._directories

    def mkdir(self, path: str, parents: bool = False) -> None:
        """Create a directory."""
        if parents:
            parts = path.split("/")
            current = ""
            for part in parts:
                if part:
                    current = f"{current}/{part}"
                    self._directories.add(current)
        else:
            self._directories.add(path)

    def _ensure_parent_dir(self, path: str) -> None:
        """Ensure parent directory exists."""
        parent = "/".join(path.split("/")[:-1])
        if parent:
            self.mkdir(parent, parents=True)

    def list_dir(self, path: str) -> list[str]:
        """List directory contents."""
        prefix = path.rstrip("/") + "/"
        items = set()

        for file_path in self._files:
            if file_path.startswith(prefix):
                relative = file_path[len(prefix) :]
                items.add(relative.split("/")[0])

        for dir_path in self._directories:
            if dir_path.startswith(prefix):
                relative = dir_path[len(prefix) :]
                if relative:
                    items.add(relative.split("/")[0])

        return sorted(items)

    def remove(self, path: str) -> None:
        """Remove a file."""
        if path in self._files:
            del self._files[path]

    def clear(self) -> None:
        """Clear all files and directories."""
        self._files.clear()
        self._directories = {"/"}


# =============================================================================
# CONSENT MOCKS
# =============================================================================


class MockConsentManager:
    """Mock consent manager for testing."""

    def __init__(
        self,
        auto_approve: bool = True,
        require_consent_for: list[str] | None = None,
    ):
        self.auto_approve = auto_approve
        self.require_consent_for = require_consent_for or []
        self._consent_log: list[dict[str, Any]] = []

    def request_consent(
        self,
        operation: str,
        details: str | None = None,
    ) -> bool:
        """Request consent for an operation."""
        self._consent_log.append(
            {
                "operation": operation,
                "details": details,
                "timestamp": time.time(),
                "granted": self.auto_approve,
            }
        )

        if operation in self.require_consent_for:
            # Would prompt user in real implementation
            return self.auto_approve

        return True

    def is_sensitive_operation(self, operation: str) -> bool:
        """Check if operation requires consent."""
        return operation in self.require_consent_for

    def get_consent_log(self) -> list[dict[str, Any]]:
        """Get log of consent requests."""
        return self._consent_log

    def reset(self) -> None:
        """Reset consent log."""
        self._consent_log = []


# =============================================================================
# PIPELINE MOCKS
# =============================================================================


class MockPipelineHandler:
    """Mock pipeline handler for testing stage execution."""

    def __init__(
        self,
        stage_name: str,
        success: bool = True,
        result_data: Any = None,
        delay_ms: float = 10.0,
    ):
        self.stage_name = stage_name
        self.success = success
        self.result_data = result_data
        self.delay_ms = delay_ms
        self._execute_count = 0

    async def execute(self, context: Any) -> dict[str, Any]:
        """Execute mock stage."""
        self._execute_count += 1
        await asyncio.sleep(self.delay_ms / 1000)

        return {
            "stage": self.stage_name,
            "success": self.success,
            "data": self.result_data,
            "error": None if self.success else "Mock stage failure",
        }


# =============================================================================
# PYTEST FIXTURES (to be imported in conftest.py)
# =============================================================================


def get_mock_fixtures():
    """
    Get dictionary of mock fixtures for use in conftest.py.

    Usage in conftest.py:
        from tests.mocks import get_mock_fixtures
        for name, fixture_func in get_mock_fixtures().items():
            globals()[name] = fixture_func
    """
    import pytest

    @pytest.fixture
    def mock_quantum_backend():
        return MockQuantumBackend()

    @pytest.fixture
    def mock_backend_registry():
        return MockBackendRegistry()

    @pytest.fixture
    def mock_llm_provider():
        return MockLLMProvider()

    @pytest.fixture
    def mock_psutil_normal():
        with mock_psutil() as m:
            yield m

    @pytest.fixture
    def mock_psutil_low_memory():
        with mock_psutil(create_low_memory_scenario()) as m:
            yield m

    @pytest.fixture
    def mock_file_system():
        return MockFileSystem()

    @pytest.fixture
    def mock_consent_auto_approve():
        return MockConsentManager(auto_approve=True)

    @pytest.fixture
    def mock_consent_deny():
        return MockConsentManager(auto_approve=False)

    @pytest.fixture
    def mock_data_factory():
        return MockDataFactory()

    return {
        "mock_quantum_backend": mock_quantum_backend,
        "mock_backend_registry": mock_backend_registry,
        "mock_llm_provider": mock_llm_provider,
        "mock_psutil_normal": mock_psutil_normal,
        "mock_psutil_low_memory": mock_psutil_low_memory,
        "mock_file_system": mock_file_system,
        "mock_consent_auto_approve": mock_consent_auto_approve,
        "mock_consent_deny": mock_consent_deny,
        "mock_data_factory": mock_data_factory,
    }


__all__ = [
    # Data factories
    "MockDataFactory",
    # Quantum mocks
    "MockQuantumBackend",
    "MockBackendRegistry",
    # LLM mocks
    "MockLLMResponse",
    "MockLLMProvider",
    "create_mock_httpx_response",
    "mock_llm_http_responses",
    # System resource mocks
    "MockSystemResources",
    "MockPsutil",
    "mock_psutil",
    "create_low_memory_scenario",
    "create_high_cpu_scenario",
    "create_low_disk_scenario",
    # File system mocks
    "MockFileSystem",
    # Consent mocks
    "MockConsentManager",
    # Pipeline mocks
    "MockPipelineHandler",
    # Fixture helpers
    "get_mock_fixtures",
]
