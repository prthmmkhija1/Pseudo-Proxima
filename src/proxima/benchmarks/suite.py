"""Benchmark suites for automated multi-circuit benchmarking.

This module provides:
- BenchmarkSuite: Definition and execution of multi-circuit benchmark suites
- SuiteResults: Aggregated results container with summary statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

from proxima.benchmarks.runner import BenchmarkRunner
from proxima.data.benchmark_registry import BenchmarkRegistry
from proxima.data.metrics import BenchmarkResult

if TYPE_CHECKING:
    from collections.abc import Callable as CallableABC


@dataclass(slots=True)
class SuiteResults:
    """Aggregated results for a benchmark suite.

    Attributes:
        name: Suite identifier.
        results: List of individual benchmark results.
        summary: Aggregated statistics (count, avg, min, max times).
    """

    name: str
    results: list[BenchmarkResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }


@dataclass(slots=True)
class BenchmarkSuite:
    """Definition of a benchmark suite for automated multi-circuit benchmarking.

    Attributes:
        name: Suite identifier.
        circuits: List of circuit definitions or names to benchmark.
        backends: List of backend names to test against.
        shots: Number of shots per circuit execution.
        runs: Number of repeated runs per circuit-backend pair.
    """

    name: str
    circuits: list[Any]
    backends: list[str]
    shots: int = 1024
    runs: int = 3

    def execute(
        self,
        runner: BenchmarkRunner,
        *,
        registry: BenchmarkRegistry | None = None,
        progress_callback: Callable[[int, int, str, Any], None] | None = None,
    ) -> SuiteResults:
        """Execute the suite across all circuits and backends.

        Args:
            runner: BenchmarkRunner instance for executing individual benchmarks.
            registry: Optional registry for persisting results.
            progress_callback: Optional callback invoked after each benchmark.
                Signature: callback(completed, total, backend_name, circuit).

        Returns:
            SuiteResults containing all individual results and summary statistics.

        Example:
            >>> suite = BenchmarkSuite("my_suite", circuits, ["lret", "qsim"])
            >>> results = suite.execute(runner, registry=registry)
            >>> print(results.summary["avg_time_ms"])
        """
        all_results: list[BenchmarkResult] = []
        total_tasks = len(self.circuits) * len(self.backends)
        completed = 0

        for circuit in self.circuits:
            for backend in self.backends:
                res = runner.run_benchmark_suite(
                    circuit=circuit,
                    backend_name=backend,
                    num_runs=self.runs,
                    shots=self.shots,
                )
                all_results.append(res)
                if registry:
                    try:
                        registry.save_result(res)
                    except Exception:
                        pass
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_tasks, backend, circuit)

        summary = self._summarize(all_results)
        return SuiteResults(name=self.name, results=all_results, summary=summary)

    @staticmethod
    def _summarize(results: Iterable[BenchmarkResult]) -> dict[str, Any]:
        times = [r.metrics.execution_time_ms for r in results if r.metrics]
        return {
            "count": len(times),
            "avg_time_ms": sum(times) / len(times) if times else 0.0,
            "min_time_ms": min(times) if times else 0.0,
            "max_time_ms": max(times) if times else 0.0,
        }
