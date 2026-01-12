"""Step 5.1: Multi-Backend Comparison - Compare quantum circuit execution across backends.

Implements the comparison workflow:
1. User specifies backends to compare
2. Validate circuit on all backends
3. Plan parallel execution (if resources allow)
4. Execute on each backend with same parameters
5. Collect and normalize results
6. Calculate comparison metrics
7. Generate comparison report

Parallel Execution Strategy:
    IF sum(memory_requirements) < available_memory * 0.8:
        Execute in parallel using asyncio.gather()
    ELSE:
        Execute sequentially with cleanup between

Comparison Metrics:
| Metric            | Description                 |
| Execution Time    | Wall-clock time per backend |
| Memory Peak       | Maximum memory usage        |
| Result Agreement  | Percentage similarity       |
| Fidelity          | For statevector comparisons |
| Performance Ratio | Time ratio between backends |
"""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from proxima.backends.base import BaseBackendAdapter


class ComparisonStatus(Enum):
    """Status of a comparison operation."""

    PENDING = auto()
    VALIDATING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    ANALYZING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIAL = auto()  # Some backends succeeded, some failed


class ExecutionStrategy(Enum):
    """Strategy for executing on multiple backends."""

    PARALLEL = "parallel"  # Execute all at once using asyncio
    SEQUENTIAL = "sequential"  # Execute one after another
    ADAPTIVE = "adaptive"  # Decide based on memory requirements


@dataclass
class BackendResult:
    """Result from a single backend execution."""

    backend_name: str
    success: bool
    execution_time_ms: float
    memory_peak_mb: float
    result: Any | None = None  # ExecutionResult if successful
    error: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonMetrics:
    """Metrics comparing results across backends."""

    # Execution Time metrics
    execution_times: dict[str, float] = field(
        default_factory=dict
    )  # backend -> time_ms
    fastest_backend: str | None = None
    slowest_backend: str | None = None
    time_ratios: dict[str, float] = field(
        default_factory=dict
    )  # backend -> ratio to fastest

    # Memory metrics
    memory_peaks: dict[str, float] = field(default_factory=dict)  # backend -> memory_mb
    lowest_memory_backend: str | None = None

    # Result Agreement metrics
    result_agreement: float = 0.0  # 0.0 to 1.0, percentage of states that match
    pairwise_agreements: dict[str, dict[str, float]] = field(default_factory=dict)

    # Fidelity metrics (for statevector comparisons)
    fidelities: dict[str, dict[str, float]] = field(
        default_factory=dict
    )  # backend_a -> backend_b -> fidelity
    average_fidelity: float = 0.0

    # Performance summary
    recommended_backend: str | None = None
    recommendation_reason: str | None = None

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "execution_times": self.execution_times,
            "fastest_backend": self.fastest_backend,
            "slowest_backend": self.slowest_backend,
            "time_ratios": self.time_ratios,
            "memory_peaks": self.memory_peaks,
            "lowest_memory_backend": self.lowest_memory_backend,
            "result_agreement": self.result_agreement,
            "pairwise_agreements": self.pairwise_agreements,
            "fidelities": self.fidelities,
            "average_fidelity": self.average_fidelity,
            "recommended_backend": self.recommended_backend,
            "recommendation_reason": self.recommendation_reason,
        }


@dataclass
class ComparisonReport:
    """Complete comparison report."""

    circuit_info: dict[str, Any]
    backends_compared: list[str]
    execution_strategy: ExecutionStrategy
    total_time_ms: float
    status: ComparisonStatus
    backend_results: dict[str, BackendResult]
    metrics: ComparisonMetrics
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "circuit_info": self.circuit_info,
            "backends_compared": self.backends_compared,
            "execution_strategy": self.execution_strategy.value,
            "total_time_ms": self.total_time_ms,
            "status": self.status.name,
            "backend_results": {
                name: {
                    "success": r.success,
                    "execution_time_ms": r.execution_time_ms,
                    "memory_peak_mb": r.memory_peak_mb,
                    "error": r.error,
                    "probabilities": r.probabilities,
                }
                for name, r in self.backend_results.items()
            },
            "metrics": self.metrics.to_dict(),
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "MULTI-BACKEND COMPARISON REPORT",
            "=" * 60,
            f"Status: {self.status.name}",
            f"Backends: {', '.join(self.backends_compared)}",
            f"Strategy: {self.execution_strategy.value}",
            f"Total Time: {self.total_time_ms:.2f} ms",
            "",
            "EXECUTION TIMES:",
        ]
        for backend, time_ms in self.metrics.execution_times.items():
            ratio = self.metrics.time_ratios.get(backend, 1.0)
            lines.append(f"  {backend}: {time_ms:.2f} ms ({ratio:.2f}x)")

        lines.extend(
            [
                "",
                f"Fastest: {self.metrics.fastest_backend}",
                f"Result Agreement: {self.metrics.result_agreement * 100:.1f}%",
            ]
        )

        if self.metrics.recommended_backend:
            lines.extend(
                [
                    "",
                    f"Recommended: {self.metrics.recommended_backend}",
                    f"Reason: {self.metrics.recommendation_reason}",
                ]
            )

        if self.errors:
            lines.extend(["", "ERRORS:"] + [f"  - {e}" for e in self.errors])

        if self.warnings:
            lines.extend(["", "WARNINGS:"] + [f"  - {w}" for w in self.warnings])

        lines.append("=" * 60)
        return "\n".join(lines)


class ExecutionPlanner:
    """Plans execution strategy based on resource requirements."""

    def __init__(self, memory_threshold: float = 0.8) -> None:
        """Initialize planner.

        Args:
            memory_threshold: Fraction of available memory to use (default 0.8 = 80%)
        """
        self.memory_threshold = memory_threshold

    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            return mem.available / (1024 * 1024)
        except ImportError:
            # Fallback: assume 4GB available
            return 4096.0

    def plan_execution(
        self,
        memory_requirements: dict[str, float],  # backend -> estimated_mb
    ) -> tuple[ExecutionStrategy, list[list[str]]]:
        """Plan execution strategy based on memory requirements.

        Args:
            memory_requirements: Dict mapping backend name to estimated memory in MB

        Returns:
            Tuple of (strategy, execution_batches)
            - strategy: PARALLEL or SEQUENTIAL
            - execution_batches: List of backend groups to execute together
        """
        available = self.get_available_memory_mb()
        threshold = available * self.memory_threshold

        total_required = sum(memory_requirements.values())

        # Strategy per spec: parallel if total < 80% available
        if total_required < threshold:
            return ExecutionStrategy.PARALLEL, [list(memory_requirements.keys())]

        # Sequential execution with cleanup between
        # Sort by memory requirement (smallest first for better batching)
        sorted_backends = sorted(memory_requirements.items(), key=lambda x: x[1])

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_memory = 0.0

        for backend, mem_req in sorted_backends:
            if current_memory + mem_req <= threshold and current_batch:
                current_batch.append(backend)
                current_memory += mem_req
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [backend]
                current_memory = mem_req

        if current_batch:
            batches.append(current_batch)

        return ExecutionStrategy.SEQUENTIAL, batches


class ResultAnalyzer:
    """Analyzes and compares results from multiple backends."""

    def __init__(self, tolerance: float = 1e-6) -> None:
        """Initialize analyzer.

        Args:
            tolerance: Tolerance for probability comparison
        """
        self.tolerance = tolerance

    def calculate_agreement(
        self,
        probs_a: dict[str, float],
        probs_b: dict[str, float],
    ) -> float:
        """Calculate agreement percentage between two probability distributions.

        Returns a value between 0.0 and 1.0 representing how similar the distributions are.
        """
        all_keys = set(probs_a.keys()) | set(probs_b.keys())
        if not all_keys:
            return 1.0  # Both empty = perfect agreement

        total_diff = 0.0
        for key in all_keys:
            pa = probs_a.get(key, 0.0)
            pb = probs_b.get(key, 0.0)
            total_diff += abs(pa - pb)

        # Total difference can be at most 2.0 (all mass shifted)
        # Convert to similarity: 1.0 - (diff / 2.0)
        agreement = 1.0 - (total_diff / 2.0)
        return max(0.0, min(1.0, agreement))

    def calculate_fidelity(
        self,
        statevector_a: np.ndarray,
        statevector_b: np.ndarray,
    ) -> float:
        """Calculate fidelity between two statevectors.

        F = |<psi_a|psi_b>|^2
        """
        inner_product = np.vdot(statevector_a, statevector_b)
        fidelity = float(np.abs(inner_product) ** 2)
        return fidelity

    def analyze(
        self,
        backend_results: dict[str, BackendResult],
    ) -> ComparisonMetrics:
        """Analyze results from multiple backends and compute metrics."""
        metrics = ComparisonMetrics()

        successful_backends = {
            name: result for name, result in backend_results.items() if result.success
        }

        if not successful_backends:
            return metrics

        # Execution time metrics
        metrics.execution_times = {
            name: result.execution_time_ms
            for name, result in successful_backends.items()
        }

        if metrics.execution_times:
            metrics.fastest_backend = min(
                metrics.execution_times, key=lambda x: metrics.execution_times[x]
            )
            metrics.slowest_backend = max(
                metrics.execution_times, key=lambda x: metrics.execution_times[x]
            )

            fastest_time = metrics.execution_times[metrics.fastest_backend]
            if fastest_time > 0:
                metrics.time_ratios = {
                    name: time / fastest_time
                    for name, time in metrics.execution_times.items()
                }

        # Memory metrics
        metrics.memory_peaks = {
            name: result.memory_peak_mb for name, result in successful_backends.items()
        }

        if metrics.memory_peaks:
            metrics.lowest_memory_backend = min(
                metrics.memory_peaks, key=lambda x: metrics.memory_peaks[x]
            )

        # Result agreement (pairwise)
        backend_names = list(successful_backends.keys())
        agreements: list[float] = []

        for i, name_a in enumerate(backend_names):
            metrics.pairwise_agreements[name_a] = {}
            for j, name_b in enumerate(backend_names):
                if i == j:
                    metrics.pairwise_agreements[name_a][name_b] = 1.0
                elif i < j:
                    probs_a = successful_backends[name_a].probabilities
                    probs_b = successful_backends[name_b].probabilities
                    agreement = self.calculate_agreement(probs_a, probs_b)
                    metrics.pairwise_agreements[name_a][name_b] = agreement
                    agreements.append(agreement)
                else:
                    # Symmetric
                    metrics.pairwise_agreements[name_a][name_b] = (
                        metrics.pairwise_agreements[name_b][name_a]
                    )

        if agreements:
            metrics.result_agreement = sum(agreements) / len(agreements)
        else:
            metrics.result_agreement = 1.0

        # Fidelity (if statevectors available)
        statevectors: dict[str, np.ndarray] = {}
        for name, result in successful_backends.items():
            if result.result and hasattr(result.result, "data"):
                sv = result.result.data.get("statevector")
                if sv is not None:
                    statevectors[name] = np.asarray(sv, dtype=complex)

        if len(statevectors) >= 2:
            fidelities: list[float] = []
            sv_names = list(statevectors.keys())
            for i, name_a in enumerate(sv_names):
                metrics.fidelities[name_a] = {}
                for j, name_b in enumerate(sv_names):
                    if i == j:
                        metrics.fidelities[name_a][name_b] = 1.0
                    elif i < j:
                        fid = self.calculate_fidelity(
                            statevectors[name_a], statevectors[name_b]
                        )
                        metrics.fidelities[name_a][name_b] = fid
                        fidelities.append(fid)
                    else:
                        metrics.fidelities[name_a][name_b] = metrics.fidelities[name_b][
                            name_a
                        ]

            if fidelities:
                metrics.average_fidelity = sum(fidelities) / len(fidelities)

        # Generate recommendation
        self._generate_recommendation(metrics)

        return metrics

    def _generate_recommendation(self, metrics: ComparisonMetrics) -> None:
        """Generate backend recommendation based on metrics."""
        if not metrics.execution_times:
            return

        # Score each backend: lower time + good agreement = better
        scores: dict[str, float] = {}

        for backend in metrics.execution_times:
            # Normalize time score (lower is better)
            time_score = metrics.time_ratios.get(backend, 1.0)

            # Agreement score (higher is better)
            agreement_scores = [
                v
                for v in metrics.pairwise_agreements.get(backend, {}).values()
                if v < 1.0  # Exclude self-comparison
            ]
            avg_agreement = (
                sum(agreement_scores) / len(agreement_scores)
                if agreement_scores
                else 1.0
            )

            # Combined score: time ratio penalty + agreement bonus
            # Lower score = better
            scores[backend] = time_score - avg_agreement

        if scores:
            metrics.recommended_backend = min(scores, key=lambda x: scores[x])

            # Generate reason
            reasons = []
            if metrics.recommended_backend == metrics.fastest_backend:
                reasons.append("fastest execution")
            if metrics.recommended_backend == metrics.lowest_memory_backend:
                reasons.append("lowest memory usage")

            backend_agreement = metrics.pairwise_agreements.get(
                metrics.recommended_backend, {}
            )
            avg_agree = sum(v for v in backend_agreement.values() if v < 1.0)
            if avg_agree > 0.95:
                reasons.append("high result agreement")

            if reasons:
                metrics.recommendation_reason = ", ".join(reasons)
            else:
                metrics.recommendation_reason = "best overall performance"


class MultiBackendComparator:
    """Main class for comparing quantum circuit execution across multiple backends.

    Implements the Step 5.1 comparison workflow:
    1. User specifies backends to compare
    2. Validate circuit on all backends
    3. Plan parallel execution (if resources allow)
    4. Execute on each backend with same parameters
    5. Collect and normalize results
    6. Calculate comparison metrics
    7. Generate comparison report
    """

    def __init__(
        self,
        planner: ExecutionPlanner | None = None,
        analyzer: ResultAnalyzer | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        """Initialize comparator.

        Args:
            planner: Execution planner (default: create new)
            analyzer: Result analyzer (default: create new)
            progress_callback: Optional callback(stage, progress) for progress updates
        """
        self.planner = planner or ExecutionPlanner()
        self.analyzer = analyzer or ResultAnalyzer()
        self.progress_callback = progress_callback

    def _report_progress(self, stage: str, progress: float) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(stage, progress)

    async def _execute_backend(
        self,
        adapter: BaseBackendAdapter,
        circuit: Any,
        options: dict[str, Any] | None,
    ) -> BackendResult:
        """Execute circuit on a single backend and capture metrics."""
        backend_name = adapter.get_name()
        start_time = time.perf_counter()

        try:
            import psutil

            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            mem_before = 0.0

        try:
            # Execute (adapters may be sync or async)
            if asyncio.iscoroutinefunction(adapter.execute):
                result = await adapter.execute(circuit, options)
            else:
                # Run sync adapter in executor to not block
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, adapter.execute, circuit, options
                )

            execution_time = (time.perf_counter() - start_time) * 1000

            try:
                mem_after = process.memory_info().rss / (1024 * 1024)
                memory_peak = max(0, mem_after - mem_before)
            except Exception:
                memory_peak = 0.0

            # Extract probabilities from result
            probabilities = {}
            if hasattr(result, "data") and isinstance(result.data, dict):
                probabilities = result.data.get("probabilities", {})
                if not probabilities and "counts" in result.data:
                    # Normalize counts to probabilities
                    counts = result.data["counts"]
                    total = sum(counts.values())
                    if total > 0:
                        probabilities = {k: v / total for k, v in counts.items()}

            return BackendResult(
                backend_name=backend_name,
                success=True,
                execution_time_ms=execution_time,
                memory_peak_mb=memory_peak,
                result=result,
                probabilities=probabilities,
                metadata=result.metadata if hasattr(result, "metadata") else {},
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return BackendResult(
                backend_name=backend_name,
                success=False,
                execution_time_ms=execution_time,
                memory_peak_mb=0.0,
                error=str(e),
            )

    async def _execute_batch_parallel(
        self,
        adapters: list[BaseBackendAdapter],
        circuit: Any,
        options: dict[str, Any] | None,
    ) -> list[BackendResult]:
        """Execute circuit on multiple backends in parallel."""
        tasks = [
            self._execute_backend(adapter, circuit, options) for adapter in adapters
        ]
        return await asyncio.gather(*tasks)

    async def _execute_batch_sequential(
        self,
        adapters: list[BaseBackendAdapter],
        circuit: Any,
        options: dict[str, Any] | None,
    ) -> list[BackendResult]:
        """Execute circuit on backends sequentially with cleanup between."""
        results = []
        for adapter in adapters:
            result = await self._execute_backend(adapter, circuit, options)
            results.append(result)
            # Cleanup between executions
            gc.collect()
        return results

    def validate_backends(
        self,
        adapters: list[BaseBackendAdapter],
        circuit: Any,
    ) -> tuple[list[BaseBackendAdapter], list[str]]:
        """Validate circuit on all backends.

        Returns:
            Tuple of (valid_adapters, validation_errors)
        """
        valid_adapters = []
        errors = []

        for adapter in adapters:
            try:
                validation = adapter.validate_circuit(circuit)
                if validation.valid:
                    valid_adapters.append(adapter)
                else:
                    errors.append(f"{adapter.get_name()}: {validation.message}")
            except Exception as e:
                errors.append(f"{adapter.get_name()}: validation failed - {e}")

        return valid_adapters, errors

    def estimate_resources(
        self,
        adapters: list[BaseBackendAdapter],
        circuit: Any,
    ) -> dict[str, float]:
        """Estimate memory requirements for each backend.

        Returns:
            Dict mapping backend name to estimated memory in MB
        """
        estimates = {}
        for adapter in adapters:
            try:
                estimate = adapter.estimate_resources(circuit)
                estimates[adapter.get_name()] = estimate.memory_mb or 100.0
            except Exception:
                estimates[adapter.get_name()] = 100.0  # Default estimate
        return estimates

    async def compare(
        self,
        adapters: list[BaseBackendAdapter],
        circuit: Any,
        options: dict[str, Any] | None = None,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    ) -> ComparisonReport:
        """Compare circuit execution across multiple backends.

        This implements the full Step 5.1 workflow:
        1. Validate circuit on all backends
        2. Plan execution strategy
        3. Execute on each backend
        4. Analyze and compare results
        5. Generate report

        Args:
            adapters: List of backend adapters to compare
            circuit: Quantum circuit to execute
            options: Execution options (shots, etc.)
            strategy: Execution strategy (PARALLEL, SEQUENTIAL, or ADAPTIVE)

        Returns:
            ComparisonReport with full comparison results
        """
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        # Extract circuit info
        circuit_info = {
            "type": type(circuit).__name__,
        }
        if hasattr(circuit, "num_qubits"):
            circuit_info["num_qubits"] = circuit.num_qubits
        if hasattr(circuit, "depth"):
            circuit_info["depth"] = circuit.depth()

        self._report_progress("validating", 0.1)

        # Step 2: Validate circuit on all backends
        valid_adapters, validation_errors = self.validate_backends(adapters, circuit)
        errors.extend(validation_errors)

        if not valid_adapters:
            return ComparisonReport(
                circuit_info=circuit_info,
                backends_compared=[a.get_name() for a in adapters],
                execution_strategy=strategy,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                status=ComparisonStatus.FAILED,
                backend_results={},
                metrics=ComparisonMetrics(),
                errors=errors,
            )

        self._report_progress("planning", 0.2)

        # Step 3: Plan execution strategy
        if strategy == ExecutionStrategy.ADAPTIVE:
            memory_estimates = self.estimate_resources(valid_adapters, circuit)
            actual_strategy, batches = self.planner.plan_execution(memory_estimates)
        elif strategy == ExecutionStrategy.PARALLEL:
            actual_strategy = ExecutionStrategy.PARALLEL
            batches = [[a.get_name() for a in valid_adapters]]
        else:
            actual_strategy = ExecutionStrategy.SEQUENTIAL
            batches = [[a.get_name()] for a in valid_adapters]

        # Map adapter names to adapters
        adapter_map = {a.get_name(): a for a in valid_adapters}

        self._report_progress("executing", 0.3)

        # Step 4: Execute on each backend
        all_results: list[BackendResult] = []

        for batch_idx, batch in enumerate(batches):
            batch_adapters = [
                adapter_map[name] for name in batch if name in adapter_map
            ]

            if actual_strategy == ExecutionStrategy.PARALLEL:
                batch_results = await self._execute_batch_parallel(
                    batch_adapters, circuit, options
                )
            else:
                batch_results = await self._execute_batch_sequential(
                    batch_adapters, circuit, options
                )

            all_results.extend(batch_results)

            # Report progress
            progress = 0.3 + 0.5 * ((batch_idx + 1) / len(batches))
            self._report_progress("executing", progress)

        # Build results dict
        backend_results = {r.backend_name: r for r in all_results}

        # Check for partial failures
        successful = sum(1 for r in all_results if r.success)
        failed = sum(1 for r in all_results if not r.success)

        for result in all_results:
            if not result.success:
                errors.append(f"{result.backend_name}: {result.error}")

        self._report_progress("analyzing", 0.85)

        # Step 5 & 6: Analyze and compare results
        metrics = self.analyzer.analyze(backend_results)

        # Determine final status
        if failed == 0:
            status = ComparisonStatus.COMPLETED
        elif successful == 0:
            status = ComparisonStatus.FAILED
        else:
            status = ComparisonStatus.PARTIAL
            warnings.append(f"{failed} of {len(all_results)} backends failed")

        self._report_progress("completed", 1.0)

        # Step 7: Generate report
        total_time = (time.perf_counter() - start_time) * 1000

        return ComparisonReport(
            circuit_info=circuit_info,
            backends_compared=[a.get_name() for a in adapters],
            execution_strategy=actual_strategy,
            total_time_ms=total_time,
            status=status,
            backend_results=backend_results,
            metrics=metrics,
            errors=errors,
            warnings=warnings,
        )

    def compare_sync(
        self,
        adapters: list[BaseBackendAdapter],
        circuit: Any,
        options: dict[str, Any] | None = None,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    ) -> ComparisonReport:
        """Synchronous wrapper for compare()."""
        return asyncio.run(self.compare(adapters, circuit, options, strategy))


# Convenience function
def compare_backends(
    adapters: list[BaseBackendAdapter],
    circuit: Any,
    options: dict[str, Any] | None = None,
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
) -> ComparisonReport:
    """Compare quantum circuit execution across multiple backends.

    This is a convenience function that creates a MultiBackendComparator
    and runs a comparison.

    Args:
        adapters: List of backend adapters to compare
        circuit: Quantum circuit to execute
        options: Execution options
        strategy: Execution strategy

    Returns:
        ComparisonReport with comparison results
    """
    comparator = MultiBackendComparator()
    return comparator.compare_sync(adapters, circuit, options, strategy)


# ==============================================================================
# Step 4.2: Backend Comparison Matrix
# ==============================================================================


@dataclass
class BackendCapabilityEntry:
    """Entry in the backend comparison matrix."""

    name: str
    supports_state_vector: bool = True
    supports_density_matrix: bool = False
    supports_gpu: bool = False
    is_cpu_optimized: bool = False
    supports_noise: bool = False
    max_qubits: int = 25
    use_case: str = "General-purpose"
    performance_tier: str = "standard"  # "standard", "high", "very-high"
    memory_efficiency: str = "standard"  # "low", "standard", "high"


class BackendComparisonMatrix:
    """
    Step 4.2: Backend Comparison Matrix for unified backend selection.

    Provides comparison data across all supported backends including:
    - LRET: Custom rank-reduction
    - Cirq: Google's framework
    - Qiskit Aer: IBM's feature-rich simulator
    - QuEST: High-performance C++ with GPU
    - cuQuantum: NVIDIA GPU-accelerated
    - qsim: Google's CPU-optimized simulator

    Comparison Dimensions:
    | Backend   | SV | DM | GPU | CPU Opt | Noise | Max Qubits | Use Case                    |
    |-----------|----|----|-----|---------|-------|------------|-----------------------------|
    | LRET      |   |   |    |        |      | 15         | Custom rank-reduction       |
    | Cirq      |   |   |    |        |      | 20         | General-purpose             |
    | Qiskit Aer|   |   |    |        |      | 30         | Qiskit ecosystem            |
    | QuEST     |   |   |    |        |      | 30         | High-performance research   |
    | cuQuantum |   |   |    |        |      | 35+        | GPU-accelerated large SV    |
    | qsim      |   |   |    |       |      | 35+        | CPU-optimized large SV      |
    """

    # Static comparison matrix
    MATRIX: dict[str, BackendCapabilityEntry] = {
        "lret": BackendCapabilityEntry(
            name="lret",
            supports_state_vector=True,
            supports_density_matrix=True,
            supports_gpu=False,
            is_cpu_optimized=False,
            supports_noise=True,
            max_qubits=15,
            use_case="Custom rank-reduction for noisy circuits",
            performance_tier="standard",
            memory_efficiency="high",  # Rank-reduced representation
        ),
        "cirq": BackendCapabilityEntry(
            name="cirq",
            supports_state_vector=True,
            supports_density_matrix=True,
            supports_gpu=False,
            is_cpu_optimized=False,
            supports_noise=True,
            max_qubits=20,
            use_case="General-purpose simulation",
            performance_tier="standard",
            memory_efficiency="standard",
        ),
        "qiskit": BackendCapabilityEntry(
            name="qiskit",
            supports_state_vector=True,
            supports_density_matrix=True,
            supports_gpu=False,
            is_cpu_optimized=False,
            supports_noise=True,
            max_qubits=30,
            use_case="Qiskit ecosystem integration",
            performance_tier="standard",
            memory_efficiency="standard",
        ),
        "quest": BackendCapabilityEntry(
            name="quest",
            supports_state_vector=True,
            supports_density_matrix=True,
            supports_gpu=True,
            is_cpu_optimized=True,
            supports_noise=True,
            max_qubits=30,
            use_case="High-performance research simulation",
            performance_tier="high",
            memory_efficiency="standard",
        ),
        "cuquantum": BackendCapabilityEntry(
            name="cuquantum",
            supports_state_vector=True,
            supports_density_matrix=False,
            supports_gpu=True,
            is_cpu_optimized=False,
            supports_noise=False,
            max_qubits=35,
            use_case="GPU-accelerated large state vector",
            performance_tier="very-high",
            memory_efficiency="low",  # GPU memory limited
        ),
        "qsim": BackendCapabilityEntry(
            name="qsim",
            supports_state_vector=True,
            supports_density_matrix=False,
            supports_gpu=False,
            is_cpu_optimized=True,
            supports_noise=False,
            max_qubits=35,
            use_case="CPU-optimized large state vector",
            performance_tier="very-high",
            memory_efficiency="standard",
        ),
    }

    @classmethod
    def get_entry(cls, backend_name: str) -> BackendCapabilityEntry | None:
        """Get capability entry for a backend."""
        return cls.MATRIX.get(backend_name.lower())

    @classmethod
    def list_all(cls) -> list[BackendCapabilityEntry]:
        """List all backend entries."""
        return list(cls.MATRIX.values())

    @classmethod
    def get_backends_for_simulation_type(
        cls,
        state_vector: bool = True,
        density_matrix: bool = False,
    ) -> list[str]:
        """Get backends supporting specified simulation type."""
        results = []
        for name, entry in cls.MATRIX.items():
            if state_vector and not entry.supports_state_vector:
                continue
            if density_matrix and not entry.supports_density_matrix:
                continue
            results.append(name)
        return results

    @classmethod
    def get_gpu_backends(cls) -> list[str]:
        """Get backends with GPU support."""
        return [name for name, entry in cls.MATRIX.items() if entry.supports_gpu]

    @classmethod
    def get_cpu_optimized_backends(cls) -> list[str]:
        """Get CPU-optimized backends."""
        return [name for name, entry in cls.MATRIX.items() if entry.is_cpu_optimized]

    @classmethod
    def get_backends_supporting_noise(cls) -> list[str]:
        """Get backends supporting noise simulation."""
        return [name for name, entry in cls.MATRIX.items() if entry.supports_noise]

    @classmethod
    def get_backends_for_qubit_count(cls, qubit_count: int) -> list[str]:
        """Get backends that can handle specified qubit count."""
        return [
            name
            for name, entry in cls.MATRIX.items()
            if entry.max_qubits >= qubit_count
        ]

    @classmethod
    def compare(
        cls,
        backend_a: str,
        backend_b: str,
    ) -> dict[str, dict[str, Any]]:
        """Compare two backends side by side."""
        entry_a = cls.MATRIX.get(backend_a.lower())
        entry_b = cls.MATRIX.get(backend_b.lower())

        if not entry_a or not entry_b:
            return {}

        comparison = {
            "state_vector": {
                backend_a: entry_a.supports_state_vector,
                backend_b: entry_b.supports_state_vector,
            },
            "density_matrix": {
                backend_a: entry_a.supports_density_matrix,
                backend_b: entry_b.supports_density_matrix,
            },
            "gpu_support": {
                backend_a: entry_a.supports_gpu,
                backend_b: entry_b.supports_gpu,
            },
            "cpu_optimized": {
                backend_a: entry_a.is_cpu_optimized,
                backend_b: entry_b.is_cpu_optimized,
            },
            "noise_support": {
                backend_a: entry_a.supports_noise,
                backend_b: entry_b.supports_noise,
            },
            "max_qubits": {
                backend_a: entry_a.max_qubits,
                backend_b: entry_b.max_qubits,
            },
            "use_case": {
                backend_a: entry_a.use_case,
                backend_b: entry_b.use_case,
            },
            "performance_tier": {
                backend_a: entry_a.performance_tier,
                backend_b: entry_b.performance_tier,
            },
        }
        return comparison

    @classmethod
    def get_recommendation(
        cls,
        qubit_count: int,
        needs_density_matrix: bool = False,
        needs_noise: bool = False,
        gpu_available: bool = False,
        prefer_performance: bool = True,
    ) -> tuple[str, str]:
        """Get recommended backend based on requirements.

        Returns:
            Tuple of (backend_name, reason)
        """
        # Filter by requirements
        candidates = []
        for name, entry in cls.MATRIX.items():
            # Check qubit capacity
            if entry.max_qubits < qubit_count:
                continue
            # Check density matrix support
            if needs_density_matrix and not entry.supports_density_matrix:
                continue
            # Check noise support
            if needs_noise and not entry.supports_noise:
                continue
            candidates.append((name, entry))

        if not candidates:
            return ("qiskit", "Fallback - no backend meets all requirements")

        # Score candidates
        scored = []
        for name, entry in candidates:
            score = 0.0

            # Performance scoring
            if entry.performance_tier == "very-high":
                score += 1.0
            elif entry.performance_tier == "high":
                score += 0.7
            else:
                score += 0.4

            # GPU bonus if available
            if gpu_available and entry.supports_gpu:
                score += 0.5
                if qubit_count > 25:
                    score += 0.3  # Extra bonus for large circuits

            # CPU optimization bonus when no GPU
            if not gpu_available and entry.is_cpu_optimized:
                score += 0.3

            # Large circuit handling
            if qubit_count > 25:
                if entry.max_qubits >= 35:
                    score += 0.2

            scored.append((name, entry, score))

        # Sort by score
        scored.sort(key=lambda x: x[2], reverse=True)
        winner_name, winner_entry, winner_score = scored[0]

        # Generate reason
        reasons = []
        if winner_entry.supports_gpu and gpu_available:
            reasons.append("GPU acceleration available")
        if winner_entry.is_cpu_optimized:
            reasons.append("CPU-optimized")
        if winner_entry.performance_tier == "very-high":
            reasons.append("highest performance tier")
        if winner_entry.max_qubits >= qubit_count + 5:
            reasons.append(f"supports up to {winner_entry.max_qubits} qubits")

        reason = (
            f"Best for {qubit_count}-qubit circuit: " + ", ".join(reasons)
            if reasons
            else winner_entry.use_case
        )

        return (winner_name, reason)

    @classmethod
    def to_markdown_table(cls) -> str:
        """Generate markdown table of backend comparison."""
        lines = [
            "| Backend | SV | DM | GPU | CPU Opt | Noise | Max Qubits | Use Case |",
            "|---------|----|----|-----|---------|-------|------------|----------|",
        ]

        for name, entry in cls.MATRIX.items():
            sv = "" if entry.supports_state_vector else ""
            dm = "" if entry.supports_density_matrix else ""
            gpu = "" if entry.supports_gpu else ""
            cpu = "" if entry.is_cpu_optimized else ""
            noise = "" if entry.supports_noise else ""

            lines.append(
                f"| {name:7} | {sv:2} | {dm:2} | {gpu:3} | {cpu:7} | {noise:5} | "
                f"{entry.max_qubits:10} | {entry.use_case} |"
            )

        return "\n".join(lines)


# ==============================================================================
# Step 4.2: Performance Comparison Metrics
# ==============================================================================


@dataclass
class PerformanceMetrics:
    """Performance metrics for backend comparison."""

    backend_name: str
    execution_time_ms: float
    memory_peak_mb: float
    cpu_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    parallel_efficiency: float = 1.0  # Ratio of actual speedup to ideal speedup


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for backend comparison."""

    backend_name: str
    state_vector_fidelity: float = 1.0  # 0-1
    measurement_distribution_similarity: float = 1.0  # 0-1
    numerical_stability_score: float = 1.0  # 0-1


class BackendPerformanceTracker:
    """
    Step 4.2: Track and compare backend performance over time.

    Maintains a history of:
    - Execution times per circuit size
    - Memory usage patterns
    - Success rates
    - Accuracy comparisons
    """

    def __init__(self) -> None:
        self._history: dict[str, list[PerformanceMetrics]] = {}
        self._accuracy_history: dict[str, list[AccuracyMetrics]] = {}

    def record_execution(
        self,
        backend_name: str,
        execution_time_ms: float,
        memory_peak_mb: float,
        cpu_utilization: float = 0.0,
        gpu_utilization: float = 0.0,
    ) -> None:
        """Record a backend execution for performance tracking."""
        if backend_name not in self._history:
            self._history[backend_name] = []

        self._history[backend_name].append(
            PerformanceMetrics(
                backend_name=backend_name,
                execution_time_ms=execution_time_ms,
                memory_peak_mb=memory_peak_mb,
                cpu_utilization_percent=cpu_utilization,
                gpu_utilization_percent=gpu_utilization,
            )
        )

    def record_accuracy(
        self,
        backend_name: str,
        fidelity: float,
        distribution_similarity: float = 1.0,
        numerical_stability: float = 1.0,
    ) -> None:
        """Record accuracy metrics for a backend."""
        if backend_name not in self._accuracy_history:
            self._accuracy_history[backend_name] = []

        self._accuracy_history[backend_name].append(
            AccuracyMetrics(
                backend_name=backend_name,
                state_vector_fidelity=fidelity,
                measurement_distribution_similarity=distribution_similarity,
                numerical_stability_score=numerical_stability,
            )
        )

    def get_average_execution_time(self, backend_name: str) -> float:
        """Get average execution time for a backend."""
        history = self._history.get(backend_name, [])
        if not history:
            return 0.0
        return sum(m.execution_time_ms for m in history) / len(history)

    def get_average_memory_usage(self, backend_name: str) -> float:
        """Get average memory usage for a backend."""
        history = self._history.get(backend_name, [])
        if not history:
            return 0.0
        return sum(m.memory_peak_mb for m in history) / len(history)

    def get_success_rate(self, backend_name: str) -> float:
        """Get success rate (placeholder - would track failures in production)."""
        # In production, this would track actual success/failure counts
        return 0.95 if backend_name in self._history else 0.8

    def get_performance_ranking(self) -> list[tuple[str, float]]:
        """Get backends ranked by performance (lower time is better)."""
        rankings = []
        for name in self._history:
            avg_time = self.get_average_execution_time(name)
            if avg_time > 0:
                rankings.append((name, avg_time))
        return sorted(rankings, key=lambda x: x[1])

    def compare_backends(
        self,
        backend_a: str,
        backend_b: str,
    ) -> dict[str, Any]:
        """Compare two backends based on recorded history."""
        time_a = self.get_average_execution_time(backend_a)
        time_b = self.get_average_execution_time(backend_b)
        mem_a = self.get_average_memory_usage(backend_a)
        mem_b = self.get_average_memory_usage(backend_b)

        time_ratio = time_a / time_b if time_b > 0 else 0.0
        mem_ratio = mem_a / mem_b if mem_b > 0 else 0.0

        return {
            "execution_time": {
                backend_a: time_a,
                backend_b: time_b,
                "ratio": time_ratio,
                "faster": backend_a if time_a < time_b else backend_b,
            },
            "memory_usage": {
                backend_a: mem_a,
                backend_b: mem_b,
                "ratio": mem_ratio,
                "more_efficient": backend_a if mem_a < mem_b else backend_b,
            },
        }
