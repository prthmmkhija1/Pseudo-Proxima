import time
from dataclasses import dataclass

from proxima.benchmarks.comparator import BackendComparator
from proxima.benchmarks.runner import BenchmarkRunner
from proxima.data.metrics import BenchmarkStatus


class TimedBackend:
    """Simulates a backend with configurable delay and deterministic output."""

    def __init__(self, name: str, delay_ms: float):
        self.name = name
        self.delay_ms = delay_ms

    def execute(self, circuit: object, shots: int = 1024):
        time.sleep(self.delay_ms / 1000.0)
        # Deterministic Bell state output for validation
        return {"counts": {"00": shots // 2, "11": shots - shots // 2}}


class SimpleRegistry:
    def __init__(self, backends: dict[str, TimedBackend]):
        self._backends = backends

    def get(self, name: str):
        return self._backends.get(name)


@dataclass
class DummyResourceMonitor:
    def start_monitoring(self):
        return None

    def stop_monitoring(self):
        return None

    def get_peak_memory_mb(self):
        return 0.0

    def get_memory_baseline_mb(self):
        return 0.0

    def get_average_cpu_percent(self):
        return 0.0

    def get_average_gpu_percent(self):
        return None


def test_backend_comparison_speedups(monkeypatch):
    """Test comparison between fast and slow backends shows correct speedup factors."""
    monkeypatch.setattr("proxima.benchmarks.runner.ResourceMonitor", lambda: DummyResourceMonitor())

    fast = TimedBackend("lret", delay_ms=3)
    slow = TimedBackend("cirq", delay_ms=12)
    registry = SimpleRegistry({"lret": fast, "cirq": slow})

    runner = BenchmarkRunner(registry)
    comparator = BackendComparator(runner=runner, backend_registry=registry)

    comparison = comparator.compare_backends(
        circuit={"name": "bell"}, backend_names=["lret", "cirq"], shots=50, num_runs=2
    )

    assert comparison.winner == "lret"
    assert comparison.results
    assert all(res.status == BenchmarkStatus.SUCCESS for res in comparison.results)
    assert comparison.speedup_factors.get("cirq", 0.0) >= 1.0
    assert comparator._validate_results(comparison.results)


def test_backend_comparison_validates_quantum_state(monkeypatch):
    """Test that comparison validates both backends produce same quantum state."""
    monkeypatch.setattr("proxima.benchmarks.runner.ResourceMonitor", lambda: DummyResourceMonitor())

    # Both backends produce identical Bell state output
    lret = TimedBackend("lret", delay_ms=5)
    cirq = TimedBackend("cirq", delay_ms=8)
    registry = SimpleRegistry({"lret": lret, "cirq": cirq})

    runner = BenchmarkRunner(registry)
    comparator = BackendComparator(runner=runner, backend_registry=registry)

    comparison = comparator.compare_backends(
        circuit={"name": "bell"}, backend_names=["lret", "cirq"], shots=100, num_runs=1
    )

    # Both should succeed
    assert len(comparison.results) == 2
    assert all(r.status == BenchmarkStatus.SUCCESS for r in comparison.results)

    # Validation should pass (same quantum state distribution)
    assert comparator._validate_results(comparison.results)

    # Report should be generated correctly
    report = comparison.generate_report()
    assert "lret" in report
    assert "cirq" in report
