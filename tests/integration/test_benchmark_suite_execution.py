from dataclasses import dataclass

from proxima.benchmarks.runner import BenchmarkRunner
from proxima.benchmarks.suite import BenchmarkSuite
from proxima.data.metrics import BenchmarkStatus


class StubBackend:
    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def execute(self, circuit: object, shots: int = 1024):
        self.calls += 1
        return {"counts": {"00": shots}}


class StubRegistry:
    def __init__(self, backends: dict[str, StubBackend]):
        self._backends = backends

    def get(self, name: str):
        return self._backends[name]


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


def test_benchmark_suite_executes_all(monkeypatch):
    """Test suite with 3 circuits × 2 backends × 2 runs = 12 executions as per spec."""
    monkeypatch.setattr("proxima.benchmarks.runner.ResourceMonitor", lambda: DummyResourceMonitor())

    backends = {"b1": StubBackend("b1"), "b2": StubBackend("b2")}
    registry = StubRegistry(backends)
    runner = BenchmarkRunner(registry)

    suite = BenchmarkSuite(
        name="smoke",
        circuits=[{"name": "c1"}, {"name": "c2"}, {"name": "c3"}],
        backends=["b1", "b2"],
        shots=32,
        runs=2,
    )

    results = suite.execute(runner, registry=None)

    # 3 circuits × 2 backends = 6 benchmark results
    assert results.summary["count"] == 6
    assert len(results.results) == 6
    assert all(res.status == BenchmarkStatus.SUCCESS for res in results.results)
    # Each backend called 3 circuits × 2 runs = 6 times
    assert backends["b1"].calls == 6
    assert backends["b2"].calls == 6


def test_suite_results_aggregate_correctly(monkeypatch):
    """Verify suite results aggregation with summary statistics."""
    monkeypatch.setattr("proxima.benchmarks.runner.ResourceMonitor", lambda: DummyResourceMonitor())

    backends = {"fast": StubBackend("fast"), "slow": StubBackend("slow")}
    registry = StubRegistry(backends)
    runner = BenchmarkRunner(registry)

    suite = BenchmarkSuite(
        name="aggregation_test",
        circuits=[{"name": "bell"}],
        backends=["fast", "slow"],
        shots=64,
        runs=1,
    )

    results = suite.execute(runner, registry=None)

    assert results.name == "aggregation_test"
    assert results.summary["count"] == 2
    assert results.summary["avg_time_ms"] >= 0
    assert results.summary["min_time_ms"] >= 0
    assert results.summary["max_time_ms"] >= results.summary["min_time_ms"]
