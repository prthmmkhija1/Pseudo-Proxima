"""
Test Utilities and Helpers

Common testing utilities, fixtures, and helper functions for Proxima tests.
This module provides reusable components for:
- Test data generation
- Assertion helpers
- Context managers for testing
- Performance measurement utilities
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import MagicMock

import numpy as np
import pytest


T = TypeVar("T")


# =============================================================================
# TIMING UTILITIES
# =============================================================================


@dataclass
class TimingResult:
    """Result of a timed operation."""
    
    elapsed_ms: float
    result: Any
    
    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ms / 1000.0


@contextlib.contextmanager
def timed_execution() -> Generator[list[float], None, None]:
    """Context manager for timing code execution.
    
    Usage:
        with timed_execution() as timing:
            # code to time
        print(f"Elapsed: {timing[0]:.2f}ms")
    """
    timing: list[float] = []
    start = time.perf_counter()
    try:
        yield timing
    finally:
        end = time.perf_counter()
        timing.append((end - start) * 1000)  # milliseconds


def time_function(func: Callable[..., T]) -> Callable[..., TimingResult]:
    """Decorator to time a function execution."""
    
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> TimingResult:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return TimingResult(
            elapsed_ms=(end - start) * 1000,
            result=result,
        )
    
    return wrapper


# =============================================================================
# QUANTUM DATA GENERATORS
# =============================================================================


def generate_random_counts(
    num_qubits: int,
    shots: int = 1024,
    seed: int | None = None,
) -> dict[str, int]:
    """Generate random quantum measurement counts.
    
    Args:
        num_qubits: Number of qubits in the system
        shots: Total number of measurements
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping bitstrings to counts
    """
    if seed is not None:
        np.random.seed(seed)
    
    states = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]
    probs = np.random.dirichlet(np.ones(len(states)))
    
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


def generate_bell_state_counts(shots: int = 1024) -> dict[str, int]:
    """Generate ideal Bell state measurement counts (|00⟩ + |11⟩)/√2."""
    half = shots // 2
    return {"00": half, "11": shots - half}


def generate_ghz_state_counts(num_qubits: int = 3, shots: int = 1024) -> dict[str, int]:
    """Generate ideal GHZ state measurement counts."""
    all_zeros = "0" * num_qubits
    all_ones = "1" * num_qubits
    half = shots // 2
    return {all_zeros: half, all_ones: shots - half}


def generate_random_statevector(num_qubits: int = 2, seed: int | None = None) -> np.ndarray:
    """Generate a random normalized statevector."""
    if seed is not None:
        np.random.seed(seed)
    
    dim = 2**num_qubits
    real = np.random.randn(dim)
    imag = np.random.randn(dim)
    sv = real + 1j * imag
    return sv / np.linalg.norm(sv)


def generate_random_density_matrix(num_qubits: int = 2, seed: int | None = None) -> np.ndarray:
    """Generate a random valid density matrix.
    
    Properties:
    - Hermitian (ρ = ρ†)
    - Positive semi-definite
    - Trace = 1
    """
    if seed is not None:
        np.random.seed(seed)
    
    sv = generate_random_statevector(num_qubits, seed)
    return np.outer(sv, sv.conj())


# =============================================================================
# ASSERTION HELPERS
# =============================================================================


def assert_counts_valid(counts: dict[str, int], shots: int, num_qubits: int) -> None:
    """Assert that measurement counts are valid.
    
    Checks:
    - Total counts equals shots
    - All bitstrings have correct length
    - All counts are non-negative
    """
    total = sum(counts.values())
    assert total == shots, f"Total counts {total} != shots {shots}"
    
    for bitstring, count in counts.items():
        assert len(bitstring) == num_qubits, f"Bitstring '{bitstring}' has wrong length"
        assert count >= 0, f"Negative count {count} for '{bitstring}'"


def assert_statevector_valid(sv: np.ndarray, num_qubits: int) -> None:
    """Assert that a statevector is valid.
    
    Checks:
    - Correct dimension
    - Normalized (|ψ|² = 1)
    """
    expected_dim = 2**num_qubits
    assert sv.shape == (expected_dim,), f"Dimension {sv.shape} != ({expected_dim},)"
    
    norm = np.linalg.norm(sv)
    assert abs(norm - 1.0) < 1e-10, f"Norm {norm} != 1.0"


def assert_density_matrix_valid(rho: np.ndarray, num_qubits: int) -> None:
    """Assert that a density matrix is valid.
    
    Checks:
    - Square matrix
    - Correct dimension
    - Hermitian
    - Trace = 1
    - Positive semi-definite
    """
    expected_dim = 2**num_qubits
    assert rho.shape == (expected_dim, expected_dim), f"Shape {rho.shape} invalid"
    
    # Hermitian
    assert np.allclose(rho, rho.conj().T), "Not Hermitian"
    
    # Trace = 1
    trace = np.trace(rho)
    assert abs(trace - 1.0) < 1e-10, f"Trace {trace} != 1.0"
    
    # Positive semi-definite (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(rho)
    assert all(ev >= -1e-10 for ev in eigenvalues), "Not positive semi-definite"


def assert_fidelity_close(
    counts1: dict[str, int],
    counts2: dict[str, int],
    tolerance: float = 0.1,
) -> None:
    """Assert that two count distributions have similar fidelity.
    
    Uses classical fidelity: F = (Σ √(p_i * q_i))²
    """
    # Get all states
    all_states = set(counts1.keys()) | set(counts2.keys())
    
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    
    # Calculate classical fidelity
    fidelity = 0.0
    for state in all_states:
        p = counts1.get(state, 0) / total1
        q = counts2.get(state, 0) / total2
        fidelity += np.sqrt(p * q)
    fidelity = fidelity**2
    
    assert fidelity >= 1 - tolerance, f"Fidelity {fidelity:.4f} < {1 - tolerance}"


# =============================================================================
# MOCK BUILDERS
# =============================================================================


class MockBuilder:
    """Builder for creating configured mock objects."""
    
    @staticmethod
    def backend(
        name: str = "mock",
        available: bool = True,
        max_qubits: int = 25,
        supports_noise: bool = False,
    ) -> MagicMock:
        """Build a mock backend."""
        mock = MagicMock()
        mock.name = name
        mock.is_available.return_value = available
        mock.get_capabilities.return_value = {
            "max_qubits": max_qubits,
            "supports_noise": supports_noise,
        }
        return mock
    
    @staticmethod
    def execution_result(
        backend: str = "mock",
        success: bool = True,
        counts: dict[str, int] | None = None,
        execution_time_ms: float = 100.0,
    ) -> MagicMock:
        """Build a mock execution result."""
        mock = MagicMock()
        mock.backend = backend
        mock.success = success
        mock.counts = counts or {"00": 512, "11": 512}
        mock.execution_time_ms = execution_time_ms
        return mock
    
    @staticmethod
    def llm_response(
        text: str = "Mock LLM response",
        model: str = "mock-model",
        tokens: int = 100,
    ) -> MagicMock:
        """Build a mock LLM response."""
        mock = MagicMock()
        mock.text = text
        mock.model = model
        mock.tokens_used = tokens
        return mock


# =============================================================================
# ASYNC TEST UTILITIES
# =============================================================================


def run_async(coro: Any) -> Any:
    """Run an async coroutine in a test context."""
    return asyncio.get_event_loop().run_until_complete(coro)


async def async_timeout(coro: Any, timeout: float = 5.0) -> Any:
    """Run an async coroutine with a timeout."""
    return await asyncio.wait_for(coro, timeout=timeout)


# =============================================================================
# FILE SYSTEM UTILITIES
# =============================================================================


@contextlib.contextmanager
def temp_yaml_config(config: dict[str, Any]) -> Generator[Path, None, None]:
    """Create a temporary YAML config file.
    
    Usage:
        with temp_yaml_config({"key": "value"}) as config_path:
            # use config_path
    """
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
    ) as f:
        yaml.safe_dump(config, f)
        f.flush()
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@contextlib.contextmanager
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory with standard structure."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as td:
        project = Path(td)
        (project / "circuits").mkdir()
        (project / "results").mkdir()
        (project / "configs").mkdir()
        yield project


# =============================================================================
# TEST MARKERS AND DECORATORS
# =============================================================================


def slow_test(func: Callable[..., T]) -> Callable[..., T]:
    """Mark a test as slow (may be skipped in quick runs)."""
    return pytest.mark.slow(func)


def requires_backend(backend_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Mark a test as requiring a specific backend."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check if backend is available
            try:
                from proxima.backends.registry import BackendRegistry
                registry = BackendRegistry()
                registry.discover()
                if backend_name not in registry._statuses:
                    pytest.skip(f"Backend {backend_name} not available")
                if not registry._statuses[backend_name].available:
                    pytest.skip(f"Backend {backend_name} not available")
            except Exception:
                pytest.skip(f"Could not check backend {backend_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================


def compare_execution_results(
    result1: dict[str, Any],
    result2: dict[str, Any],
    ignore_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two execution results and return differences.
    
    Returns:
        Dictionary with differences, empty if results are equivalent
    """
    ignore = set(ignore_keys or [])
    differences: dict[str, Any] = {}
    
    all_keys = set(result1.keys()) | set(result2.keys())
    
    for key in all_keys:
        if key in ignore:
            continue
        
        val1 = result1.get(key)
        val2 = result2.get(key)
        
        if val1 != val2:
            differences[key] = {
                "result1": val1,
                "result2": val2,
            }
    
    return differences


def calculate_speedup(time1: float, time2: float) -> float:
    """Calculate speedup ratio between two times.
    
    Returns:
        Speedup factor (>1 means time1 is faster)
    """
    if time1 == 0:
        return float("inf")
    return time2 / time1
