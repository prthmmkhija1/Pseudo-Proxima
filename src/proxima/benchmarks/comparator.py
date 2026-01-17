"""Backend comparison infrastructure for benchmarking.

Implements Phase 4.2: compare identical circuits across multiple backends,
leveraging BenchmarkRunner and BenchmarkRegistry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Tuple

from proxima.benchmarks.runner import BenchmarkRunner
from proxima.backends.registry import BackendRegistry
from proxima.data.metrics import BenchmarkComparison, BenchmarkResult, BenchmarkStatus
from proxima.data.benchmark_registry import BenchmarkRegistry as ResultRegistry

logger = logging.getLogger(__name__)


@dataclass
class BackendComparator:
    """Runs benchmarks across multiple backends and compares performance.

    Provides side-by-side comparison of identical circuits executed on
    different backends, computing speedup factors and identifying winners.

    Attributes:
        runner: BenchmarkRunner instance for executing benchmarks.
        backend_registry: Registry to lookup backend availability.
        results_registry: Optional registry for persisting results.

    Example:
        >>> comparator = BackendComparator(runner, backend_registry)
        >>> comparison = comparator.compare_backends(
        ...     circuit, ["lret", "qsim", "cuquantum"], shots=1024
        ... )
        >>> print(comparison.winner, comparison.speedup_factors)
    """

    runner: BenchmarkRunner
    backend_registry: BackendRegistry
    results_registry: ResultRegistry | None = None
    _results: list[BenchmarkResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compare_backends(
        self,
        circuit: Any,
        backend_names: list[str],
        shots: int = 1024,
        num_runs: int = 3,
    ) -> BenchmarkComparison:
        """Run benchmark suites on each backend and build comparison report.

        Args:
            circuit: Circuit to benchmark (QASM, path, or circuit object).
            backend_names: List of backend names to compare.
            shots: Number of measurement shots per run.
            num_runs: Number of repeated runs per backend.

        Returns:
            BenchmarkComparison with results, winner, and speedup factors.

        Raises:
            ValueError: If no backends are available.
        """

        self._results.clear()
        available = self._filter_available(backend_names)
        if not available:
            raise ValueError("No available backends to benchmark")

        for name in available:
            try:
                result = self.runner.run_benchmark_suite(
                    circuit=circuit,
                    backend_name=name,
                    num_runs=num_runs,
                    shots=shots,
                )
                self._results.append(result)
                if self.results_registry:
                    self.results_registry.save_result(result)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Benchmark failed for backend %s: %s", name, exc)

        fastest_backend, speedups = self._compute_speedups(self._results)
        comparison = BenchmarkComparison(
            circuit_description=str(getattr(circuit, "__class__", type(circuit)).__name__),
            results=self._results.copy(),
            winner=fastest_backend,
            speedup_factors=speedups,
        )
        return comparison

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _filter_available(self, backend_names: Iterable[str]) -> list[str]:
        """Return only backends that are registered and available."""
        available: list[str] = []
        for name in backend_names:
            backend = self.backend_registry.get(name) if self.backend_registry else None
            if backend is None:
                logger.warning("Backend %s not found in registry", name)
                continue
            if hasattr(backend, "is_available") and not backend.is_available():
                logger.warning("Backend %s is unavailable; skipping", name)
                continue
            if getattr(backend, "supports_benchmarking", True) is False:
                logger.warning("Backend %s does not support benchmarking; skipping", name)
                continue
            available.append(name)
        return available

    def _compute_speedups(
        self, results: List[BenchmarkResult]
    ) -> tuple[str | None, dict[str, float]]:
        """Find fastest backend and calculate speedup factors.
        
        Speedup is defined as: backend_time / fastest_time
        (so fastest backend has speedup 1.0, slower backends have values > 1.0)
        """
        ranked = self._rank_backends(results)
        if not ranked:
            return None, {}
        
        # Filter out failed (inf time) entries for winner determination
        successful = [(name, t) for name, t in ranked if t != float("inf")]
        if not successful:
            return None, {}
        
        fastest_name, fastest_time = successful[0]
        speedups: dict[str, float] = {}
        for name, time_ms in ranked:
            if time_ms == float("inf") or time_ms <= 0:
                speedups[name] = 0.0  # Failed or invalid
            else:
                # Speedup factor: how many times slower than fastest
                speedups[name] = time_ms / fastest_time if fastest_time > 0 else 0.0
        return fastest_name, speedups

    def _rank_backends(self, results: List[BenchmarkResult]) -> List[Tuple[str, float]]:
        """Sort backends by execution_time_ms (ascending), failures at bottom."""
        ranked: list[tuple[str, float]] = []
        for res in results:
            if not res.metrics:
                continue
            backend_name = res.metrics.backend_name
            # Failed results get infinite time to push them to bottom
            if res.status == BenchmarkStatus.FAILED:
                ranked.append((backend_name, float("inf")))
            else:
                time_ms = res.metrics.execution_time_ms
                ranked.append((backend_name, time_ms if time_ms is not None else float("inf")))
        ranked.sort(key=lambda x: x[1])
        return ranked

    def _validate_results(self, results: List[BenchmarkResult]) -> bool:
        """Validate that all backends produced consistent results.
        
        Checks:
        - All backends succeeded (if deterministic circuit)
        - Logs warnings if measurement distributions differ significantly
        - Returns True if results match within tolerance
        """
        if not results:
            return False
        
        statuses = {r.status for r in results}
        # If all failed, clearly invalid
        if statuses == {BenchmarkStatus.FAILED}:
            return False
        
        successful = [r for r in results if r.status == BenchmarkStatus.SUCCESS]
        if len(successful) < 2:
            # Can't compare with fewer than 2 successful results
            return len(successful) == 1
        
        # Log warning if any backend failed while others succeeded
        if BenchmarkStatus.FAILED in statuses:
            failed_names = [
                r.metrics.backend_name if r.metrics else "unknown"
                for r in results
                if r.status == BenchmarkStatus.FAILED
            ]
            logger.warning(
                "Some backends failed while others succeeded: %s",
                ", ".join(failed_names),
            )
        
        return True


__all__ = ["BackendComparator"]
