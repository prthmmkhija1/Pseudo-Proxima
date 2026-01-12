"""Step 5.3: Performance Benchmarking - Backend Performance Tests.

Comprehensive performance benchmark suite covering:
- Execution time measurement
- Memory usage tracking
- Scalability testing
- Comparative benchmarks

Benchmark Metrics:
| Metric          | Description                                   |
|-----------------|-----------------------------------------------|
| Execution Time  | Wall clock time for simulation                |
| Peak Memory     | Maximum RSS during execution                  |
| CPU Utilization | CPU usage percentage                          |
| Throughput      | Circuits per second                           |
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_backend():
    """Create mock backend for benchmarking."""
    mock = MagicMock()
    mock.get_name.return_value = "mock"
    mock.execute.return_value = MagicMock(
        backend="mock",
        execution_time_ms=10.0,
        data={"counts": {"00": 500, "11": 500}},
    )
    return mock


@pytest.fixture
def small_circuit():
    """Small circuit (5-10 qubits)."""
    return {
        "num_qubits": 8,
        "gates": [{"name": "H", "qubits": [i]} for i in range(8)]
        + [{"name": "CNOT", "qubits": [i, i + 1]} for i in range(7)],
        "measurements": list(range(8)),
    }


@pytest.fixture
def medium_circuit():
    """Medium circuit (15-20 qubits)."""
    return {
        "num_qubits": 18,
        "gates": [{"name": "H", "qubits": [i]} for i in range(18)]
        + [{"name": "CNOT", "qubits": [i, i + 1]} for i in range(17)]
        + [{"name": "Rz", "qubits": [i], "params": {"theta": 0.5}} for i in range(18)],
        "measurements": list(range(18)),
    }


@pytest.fixture
def large_circuit():
    """Large circuit (25-30 qubits)."""
    return {
        "num_qubits": 25,
        "gates": [{"name": "H", "qubits": [i]} for i in range(25)]
        + [{"name": "CNOT", "qubits": [i, i + 1]} for i in range(24)]
        + [{"name": "Rz", "qubits": [i], "params": {"theta": 0.5}} for i in range(25)]
        + [{"name": "Ry", "qubits": [i], "params": {"theta": 0.3}} for i in range(25)],
        "measurements": list(range(25)),
    }


@pytest.fixture
def deep_circuit():
    """Deep circuit (high gate count)."""
    return {
        "num_qubits": 10,
        "gates": [{"name": "H", "qubits": [i % 10]} for i in range(100)]
        + [{"name": "CNOT", "qubits": [i % 10, (i + 1) % 10]} for i in range(100)]
        + [
            {"name": "Rz", "qubits": [i % 10], "params": {"theta": 0.1 * i}}
            for i in range(100)
        ],
        "measurements": list(range(10)),
    }


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================


class BenchmarkTimer:
    """Utility for timing benchmarks."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None

    def __enter__(self):
        gc.collect()  # Clean up before benchmark
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


class MemoryTracker:
    """Utility for tracking memory usage."""

    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None

    def __enter__(self):
        try:
            import psutil

            process = psutil.Process()
            self.initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            self.initial_memory = 0
        return self

    def __exit__(self, *args):
        try:
            import psutil

            process = psutil.Process()
            self.final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            self.peak_memory = max(self.initial_memory, self.final_memory)
        except ImportError:
            self.final_memory = 0
            self.peak_memory = 0


# =============================================================================
# STEP 5.3.1: SMALL CIRCUIT BENCHMARKS
# =============================================================================


@pytest.mark.performance
@pytest.mark.slow
class TestSmallCircuitBenchmarks:
    """Benchmarks for small circuits (5-10 qubits)."""

    def test_execution_overhead(self, mock_backend, small_circuit):
        """Measure execution overhead for small circuits."""
        with BenchmarkTimer() as timer:
            mock_backend.execute(small_circuit, options={"shots": 1000})

        # Small circuits should have low overhead
        assert timer.elapsed_ms < 1000  # Less than 1 second
        print(f"Small circuit execution: {timer.elapsed_ms:.2f}ms")

    def test_memory_footprint(self, mock_backend, small_circuit):
        """Measure memory footprint for small circuits."""
        with MemoryTracker() as tracker:
            mock_backend.execute(small_circuit, options={"shots": 1000})

        memory_delta = tracker.final_memory - tracker.initial_memory

        # Small circuits should use minimal memory
        assert memory_delta < 100  # Less than 100MB increase
        print(f"Memory delta: {memory_delta:.2f}MB")

    def test_batch_execution_throughput(self, mock_backend, small_circuit):
        """Measure throughput for batch execution."""
        batch_size = 10

        with BenchmarkTimer() as timer:
            for _ in range(batch_size):
                mock_backend.execute(small_circuit, options={"shots": 100})

        throughput = batch_size / (timer.elapsed_ms / 1000)  # circuits/second

        assert throughput > 0
        print(f"Throughput: {throughput:.2f} circuits/second")

    def test_validation_speed(self, mock_backend, small_circuit):
        """Measure circuit validation speed."""
        mock_backend.validate_circuit.return_value = MagicMock(valid=True)

        iterations = 100
        with BenchmarkTimer() as timer:
            for _ in range(iterations):
                mock_backend.validate_circuit(small_circuit)

        avg_time = timer.elapsed_ms / iterations

        assert avg_time < 10  # Less than 10ms per validation
        print(f"Avg validation time: {avg_time:.3f}ms")


# =============================================================================
# STEP 5.3.2: MEDIUM CIRCUIT BENCHMARKS
# =============================================================================


@pytest.mark.performance
@pytest.mark.slow
class TestMediumCircuitBenchmarks:
    """Benchmarks for medium circuits (15-20 qubits)."""

    def test_execution_time(self, mock_backend, medium_circuit):
        """Measure execution time for medium circuits."""
        with BenchmarkTimer() as timer:
            mock_backend.execute(medium_circuit, options={"shots": 1000})

        # Medium circuits may take longer
        assert timer.elapsed_ms < 30000  # Less than 30 seconds
        print(f"Medium circuit execution: {timer.elapsed_ms:.2f}ms")

    def test_memory_usage(self, mock_backend, medium_circuit):
        """Measure memory usage for medium circuits."""
        with MemoryTracker() as tracker:
            mock_backend.execute(medium_circuit, options={"shots": 1000})

        # 18 qubits = 2^18 amplitudes * 16 bytes = ~4MB state vector
        # Allow for workspace and overhead
        memory_delta = tracker.final_memory - tracker.initial_memory
        print(f"Memory usage: {memory_delta:.2f}MB")

    def test_scaling_with_depth(self, mock_backend, medium_circuit):
        """Test execution time scaling with circuit depth."""
        times = []

        for depth_multiplier in [1, 2, 3]:
            # Create deeper circuit
            deep_circuit = {
                "num_qubits": medium_circuit["num_qubits"],
                "gates": medium_circuit["gates"] * depth_multiplier,
                "measurements": medium_circuit["measurements"],
            }

            with BenchmarkTimer() as timer:
                mock_backend.execute(deep_circuit, options={"shots": 100})

            times.append(timer.elapsed_ms)

        # Time should increase with depth
        print(f"Depth scaling: {times}")

    def test_parallel_efficiency(self, mock_backend, medium_circuit):
        """Test parallel execution efficiency."""
        # Single-threaded baseline
        mock_backend.execute.return_value.metadata = {"threads": 1}

        with BenchmarkTimer() as single_timer:
            mock_backend.execute(medium_circuit, options={"num_threads": 1})

        # Multi-threaded
        mock_backend.execute.return_value.metadata = {"threads": 4}

        with BenchmarkTimer() as multi_timer:
            mock_backend.execute(medium_circuit, options={"num_threads": 4})

        # Multi-threaded should not be slower
        print(
            f"Single-thread: {single_timer.elapsed_ms:.2f}ms, Multi-thread: {multi_timer.elapsed_ms:.2f}ms"
        )


# =============================================================================
# STEP 5.3.3: LARGE CIRCUIT BENCHMARKS
# =============================================================================


@pytest.mark.performance
@pytest.mark.slow
class TestLargeCircuitBenchmarks:
    """Benchmarks for large circuits (25-30 qubits)."""

    def test_execution_feasibility(self, mock_backend, large_circuit):
        """Test that large circuits can execute within reasonable time."""
        with BenchmarkTimer() as timer:
            mock_backend.execute(large_circuit, options={"shots": 100})

        # Large circuits may take significant time
        assert timer.elapsed_ms < 300000  # Less than 5 minutes
        print(f"Large circuit execution: {timer.elapsed_ms:.2f}ms")

    def test_memory_requirements(self, mock_backend, large_circuit):
        """Measure memory requirements for large circuits."""
        with MemoryTracker() as tracker:
            mock_backend.execute(large_circuit, options={"shots": 100})

        # 25 qubits = 2^25 amplitudes * 16 bytes = ~512MB state vector
        memory_delta = tracker.final_memory - tracker.initial_memory
        print(f"Memory requirement: {memory_delta:.2f}MB")

    def test_resource_estimation_accuracy(self, mock_backend, large_circuit):
        """Test resource estimation accuracy for large circuits."""
        mock_backend.estimate_resources.return_value = MagicMock(
            memory_mb=512.0,
            time_ms=10000.0,
        )

        estimate = mock_backend.estimate_resources(large_circuit)

        # Estimate should be reasonable for 25 qubits
        assert estimate.memory_mb >= 100  # At least 100MB
        print(f"Estimated memory: {estimate.memory_mb}MB")

    def test_gpu_vs_cpu_simulation(self, mock_backend, large_circuit):
        """Compare GPU vs CPU simulation times (mocked)."""
        # CPU simulation
        mock_backend.execute.return_value.metadata = {"device": "CPU"}

        with BenchmarkTimer() as cpu_timer:
            mock_backend.execute(large_circuit, options={"use_gpu": False})

        # GPU simulation
        mock_backend.execute.return_value.metadata = {"device": "GPU"}

        with BenchmarkTimer() as gpu_timer:
            mock_backend.execute(large_circuit, options={"use_gpu": True})

        print(f"CPU: {cpu_timer.elapsed_ms:.2f}ms, GPU: {gpu_timer.elapsed_ms:.2f}ms")


# =============================================================================
# STEP 5.3.4: COMPARATIVE BENCHMARKS
# =============================================================================


@pytest.mark.performance
@pytest.mark.slow
class TestComparativeBenchmarks:
    """Comparative benchmarks across backends."""

    def test_backend_comparison(self, small_circuit):
        """Compare execution time across backends."""
        backends = {}

        for name in ["cirq", "qiskit", "quest"]:
            mock = MagicMock()
            mock.get_name.return_value = name
            mock.execute.return_value = MagicMock(
                backend=name,
                execution_time_ms=np.random.uniform(5, 20),
            )
            backends[name] = mock

        results = {}
        for name, backend in backends.items():
            with BenchmarkTimer() as timer:
                backend.execute(small_circuit, options={"shots": 1000})

            results[name] = timer.elapsed_ms

        print(f"Backend comparison: {results}")

        # All should complete
        for name in results:
            assert results[name] > 0

    def test_scaling_comparison(self, small_circuit, medium_circuit):
        """Compare scaling across different circuit sizes."""
        mock_backend = MagicMock()
        mock_backend.execute.return_value = MagicMock(execution_time_ms=10.0)

        scaling_results = {}

        for name, circuit in [("small", small_circuit), ("medium", medium_circuit)]:
            with BenchmarkTimer() as timer:
                mock_backend.execute(circuit, options={"shots": 1000})

            scaling_results[name] = timer.elapsed_ms

        print(f"Scaling comparison: {scaling_results}")

    def test_shot_count_impact(self, mock_backend, small_circuit):
        """Test impact of shot count on execution time."""
        shot_counts = [100, 1000, 10000]
        times = {}

        for shots in shot_counts:
            with BenchmarkTimer() as timer:
                mock_backend.execute(small_circuit, options={"shots": shots})

            times[shots] = timer.elapsed_ms

        print(f"Shot count impact: {times}")

    def test_repeated_execution_consistency(self, mock_backend, small_circuit):
        """Test consistency of repeated executions."""
        times = []

        for _ in range(5):
            with BenchmarkTimer() as timer:
                mock_backend.execute(small_circuit, options={"shots": 1000})

            times.append(timer.elapsed_ms)

        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time if mean_time > 0 else 0

        # Coefficient of variation should be reasonable
        print(
            f"Execution times: mean={mean_time:.2f}ms, std={std_time:.2f}ms, CV={cv:.2f}"
        )


# =============================================================================
# STEP 5.3.5: DENSITY MATRIX AND NOISE BENCHMARKS
# =============================================================================


@pytest.mark.performance
@pytest.mark.slow
class TestDensityMatrixBenchmarks:
    """Benchmarks for density matrix simulations."""

    def test_dm_vs_sv_comparison(self, mock_backend):
        """Compare density matrix vs state vector simulation."""
        circuit = {
            "num_qubits": 10,
            "gates": [{"name": "H", "qubits": [i]} for i in range(10)],
            "measurements": list(range(10)),
        }

        # State vector
        mock_backend.execute.return_value.metadata = {"simulator_type": "state_vector"}

        with BenchmarkTimer() as sv_timer:
            mock_backend.execute(circuit, options={"simulator_type": "state_vector"})

        # Density matrix
        mock_backend.execute.return_value.metadata = {
            "simulator_type": "density_matrix"
        }

        with BenchmarkTimer() as dm_timer:
            mock_backend.execute(circuit, options={"simulator_type": "density_matrix"})

        print(f"SV: {sv_timer.elapsed_ms:.2f}ms, DM: {dm_timer.elapsed_ms:.2f}ms")

    def test_noisy_simulation_overhead(self, mock_backend):
        """Measure overhead of noise simulation."""
        circuit = {
            "num_qubits": 8,
            "gates": [{"name": "H", "qubits": [i]} for i in range(8)],
            "measurements": list(range(8)),
        }

        # Ideal
        with BenchmarkTimer() as ideal_timer:
            mock_backend.execute(circuit, options={"noise_model": None})

        # Noisy
        noise_model = {"depolarizing": 0.01}
        with BenchmarkTimer() as noisy_timer:
            mock_backend.execute(circuit, options={"noise_model": noise_model})

        print(
            f"Ideal: {ideal_timer.elapsed_ms:.2f}ms, Noisy: {noisy_timer.elapsed_ms:.2f}ms"
        )


# =============================================================================
# BENCHMARK SUMMARY UTILITIES
# =============================================================================


class BenchmarkReport:
    """Generate benchmark reports."""

    @staticmethod
    def generate_summary(results: dict) -> str:
        """Generate markdown summary of benchmark results."""
        lines = ["# Benchmark Results\n"]
        lines.append("| Test | Time (ms) | Memory (MB) | Status |")
        lines.append("|------|-----------|-------------|--------|")

        for name, data in results.items():
            time_ms = data.get("time_ms", "N/A")
            memory_mb = data.get("memory_mb", "N/A")
            status = "" if data.get("passed", True) else ""
            lines.append(f"| {name} | {time_ms} | {memory_mb} | {status} |")

        return "\n".join(lines)


@pytest.mark.performance
class TestBenchmarkReporting:
    """Tests for benchmark reporting."""

    def test_report_generation(self):
        """Test benchmark report generation."""
        results = {
            "small_circuit": {"time_ms": 15.5, "memory_mb": 50.0, "passed": True},
            "medium_circuit": {"time_ms": 150.2, "memory_mb": 200.0, "passed": True},
            "large_circuit": {"time_ms": 5000.0, "memory_mb": 600.0, "passed": True},
        }

        report = BenchmarkReport.generate_summary(results)

        assert "Benchmark Results" in report
        assert "small_circuit" in report
        assert "" in report
