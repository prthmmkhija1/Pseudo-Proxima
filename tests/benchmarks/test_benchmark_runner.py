import time
from dataclasses import dataclass

import pytest

from proxima.benchmarks.runner import BenchmarkRunner, BenchmarkRunnerConfig
from proxima.data.metrics import BenchmarkStatus


class DummyBackend:
    def __init__(self, delay_ms: float = 5.0, should_fail: bool = False):
        self.delay_ms = delay_ms
        self.should_fail = should_fail
        self.calls = 0

    def execute(self, circuit: object, shots: int = 1024):
        self.calls += 1
        time.sleep(self.delay_ms / 1000.0)
        if self.should_fail:
            raise RuntimeError("boom")
        return {"counts": {"0": shots}}


class DummyRegistry:
    def __init__(self, backend: DummyBackend):
        self.backend = backend

    def get(self, name: str) -> DummyBackend:
        return self.backend


@dataclass
class DummyResourceMonitor:
    def start_monitoring(self) -> None:
        return None

    def stop_monitoring(self) -> None:
        return None

    def get_peak_memory_mb(self) -> float:
        return 10.0

    def get_memory_baseline_mb(self) -> float:
        return 5.0

    def get_average_cpu_percent(self) -> float:
        return 20.0

    def get_average_gpu_percent(self):
        return None


@pytest.fixture(autouse=True)
def patch_resource_monitor(monkeypatch):
    monkeypatch.setattr("proxima.benchmarks.runner.ResourceMonitor", lambda: DummyResourceMonitor())


def test_single_benchmark_success():
    backend = DummyBackend(delay_ms=2.0)
    registry = DummyRegistry(backend)
    runner = BenchmarkRunner(registry)

    result = runner.run_benchmark(circuit={"name": "bell"}, backend_name="mock", shots=50)

    assert result.status == BenchmarkStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.backend_name == "mock"
    assert result.metrics.execution_time_ms > 0
    assert backend.calls == 1


def test_multi_run_aggregates_stats():
    backend = DummyBackend(delay_ms=2.0)
    registry = DummyRegistry(backend)
    runner = BenchmarkRunner(registry)

    result = runner.run_benchmark_suite(circuit={"name": "bell"}, backend_name="mock", num_runs=3, shots=20)

    assert result.status == BenchmarkStatus.SUCCESS
    assert result.metrics is not None
    stats = result.metadata.get("statistics", {})
    assert stats.get("avg_time_ms", 0) > 0
    assert stats.get("success_rate_percent") == 100.0
    assert len(result.metadata.get("individual_runs", [])) == 3


def test_error_handling_marks_failure():
    backend = DummyBackend(delay_ms=1.0, should_fail=True)
    registry = DummyRegistry(backend)
    runner = BenchmarkRunner(registry)

    result = runner.run_benchmark(circuit={}, backend_name="mock")

    assert result.status == BenchmarkStatus.FAILED
    assert result.error_message is not None
    assert result.metrics is not None
    assert result.metrics.success_rate_percent == 0.0
