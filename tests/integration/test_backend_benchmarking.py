import time
from dataclasses import dataclass

from proxima.benchmarks.runner import BenchmarkRunner
from proxima.data.benchmark_registry import BenchmarkRegistry
from proxima.data.metrics import BenchmarkStatus


class LRETBackend:
    def __init__(self, delay_ms: float = 5.0):
        self.delay_ms = delay_ms
        self.calls = 0

    def execute(self, circuit: object, shots: int = 1024):
        self.calls += 1
        time.sleep(self.delay_ms / 1000.0)
        return {"counts": {"00": shots}}


class LRETRegistry:
    def __init__(self, backend: LRETBackend):
        self.backend = backend

    def get(self, name: str):
        return self.backend


@dataclass
class DummyResourceMonitor:
    def start_monitoring(self):
        return None

    def stop_monitoring(self):
        return None

    def get_peak_memory_mb(self):
        return 8.0

    def get_memory_baseline_mb(self):
        return 4.0

    def get_average_cpu_percent(self):
        return 10.0

    def get_average_gpu_percent(self):
        return None


def test_backend_benchmarking_with_lret(monkeypatch, tmp_path):
    monkeypatch.setattr("proxima.benchmarks.runner.ResourceMonitor", lambda: DummyResourceMonitor())

    backend = LRETBackend(delay_ms=5.0)
    registry = LRETRegistry(backend)
    result_store = BenchmarkRegistry(db_path=tmp_path / "bench.db")

    runner = BenchmarkRunner(registry, results_storage=result_store)
    result = runner.run_benchmark_suite(circuit={"name": "bell"}, backend_name="lret", num_runs=3, shots=100)

    assert result.status == BenchmarkStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.backend_name == "lret"
    assert len(result.metadata.get("individual_runs", [])) == 3

    stored = result_store.get_results_for_backend("lret")
    assert stored
    assert stored[0].metrics is not None
