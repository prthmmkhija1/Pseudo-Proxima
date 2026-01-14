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

    async def _execute_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        adapter: BaseBackendAdapter,
        circuit: Any,
        options: dict[str, Any] | None,
        timeout: float | None = None,
    ) -> BackendResult:
        """Execute a backend with semaphore control and optional timeout.
        
        Args:
            semaphore: Semaphore for concurrency control
            adapter: Backend adapter to execute
            circuit: Quantum circuit
            options: Execution options
            timeout: Optional timeout in seconds
            
        Returns:
            BackendResult with execution outcome
        """
        async with semaphore:
            if timeout is not None:
                try:
                    return await asyncio.wait_for(
                        self._execute_backend(adapter, circuit, options),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    return BackendResult(
                        backend_name=adapter.get_name(),
                        success=False,
                        execution_time_ms=timeout * 1000,
                        memory_peak_mb=0.0,
                        error=f"Execution timed out after {timeout}s",
                    )
            else:
                return await self._execute_backend(adapter, circuit, options)

    async def _execute_batch_parallel(
        self,
        adapters: list[BaseBackendAdapter],
        circuit: Any,
        options: dict[str, Any] | None,
        max_concurrent: int | None = None,
        timeout_per_backend: float | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> list[BackendResult]:
        """Execute circuit on multiple backends in parallel with resource management.
        
        Enhanced parallel execution with:
        - Configurable concurrency limits via semaphore
        - Per-backend timeout handling
        - Cancellation support for graceful shutdown
        - Ordered result preservation
        - Automatic resource throttling based on system memory
        
        Args:
            adapters: List of backend adapters to execute
            circuit: Quantum circuit to run
            options: Execution options
            max_concurrent: Maximum concurrent executions (default: auto-detect)
            timeout_per_backend: Timeout per backend in seconds (default: None)
            cancellation_event: Event to signal cancellation (default: None)
            
        Returns:
            List of BackendResult in same order as adapters
        """
        if not adapters:
            return []
            
        # Auto-detect concurrency limit based on available memory
        if max_concurrent is None:
            try:
                import psutil
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                # Estimate ~2GB per backend execution, cap at number of adapters
                max_concurrent = max(1, min(len(adapters), int(available_memory_gb / 2)))
            except ImportError:
                # Default to min of 4 or number of adapters
                max_concurrent = min(4, len(adapters))
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks with semaphore control
        async def execute_with_check(
            adapter: BaseBackendAdapter,
        ) -> BackendResult:
            # Check for cancellation before starting
            if cancellation_event and cancellation_event.is_set():
                return BackendResult(
                    backend_name=adapter.get_name(),
                    success=False,
                    execution_time_ms=0.0,
                    memory_peak_mb=0.0,
                    error="Execution cancelled",
                )
            return await self._execute_with_semaphore(
                semaphore, adapter, circuit, options, timeout_per_backend
            )
        
        tasks = [execute_with_check(adapter) for adapter in adapters]
        
        # Use gather with return_exceptions to handle individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to BackendResult
        processed_results: list[BackendResult] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    BackendResult(
                        backend_name=adapters[idx].get_name(),
                        success=False,
                        execution_time_ms=0.0,
                        memory_peak_mb=0.0,
                        error=f"Unexpected error: {result}",
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results

    async def _execute_batch_sequential(
        self,
        adapters: list[BaseBackendAdapter],
        circuit: Any,
        options: dict[str, Any] | None,
        timeout_per_backend: float | None = None,
        cancellation_event: asyncio.Event | None = None,
        cleanup_between: bool = True,
    ) -> list[BackendResult]:
        """Execute circuit on backends sequentially with cleanup between.
        
        Enhanced sequential execution with:
        - Per-backend timeout handling
        - Cancellation support for graceful shutdown
        - Configurable cleanup between executions
        - Memory pressure monitoring
        
        Args:
            adapters: List of backend adapters to execute
            circuit: Quantum circuit to run
            options: Execution options
            timeout_per_backend: Timeout per backend in seconds (default: None)
            cancellation_event: Event to signal cancellation (default: None)
            cleanup_between: Whether to run gc.collect() between executions
            
        Returns:
            List of BackendResult in same order as adapters
        """
        results: list[BackendResult] = []
        
        for adapter in adapters:
            # Check for cancellation
            if cancellation_event and cancellation_event.is_set():
                results.append(
                    BackendResult(
                        backend_name=adapter.get_name(),
                        success=False,
                        execution_time_ms=0.0,
                        memory_peak_mb=0.0,
                        error="Execution cancelled",
                    )
                )
                continue
            
            # Execute with optional timeout
            if timeout_per_backend is not None:
                try:
                    result = await asyncio.wait_for(
                        self._execute_backend(adapter, circuit, options),
                        timeout=timeout_per_backend,
                    )
                except asyncio.TimeoutError:
                    result = BackendResult(
                        backend_name=adapter.get_name(),
                        success=False,
                        execution_time_ms=timeout_per_backend * 1000,
                        memory_peak_mb=0.0,
                        error=f"Execution timed out after {timeout_per_backend}s",
                    )
            else:
                result = await self._execute_backend(adapter, circuit, options)
            
            results.append(result)
            
            # Cleanup between executions to manage memory
            if cleanup_between:
                gc.collect()
                # Check memory pressure and wait if needed
                try:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 90:
                        # High memory pressure - force aggressive cleanup
                        gc.collect()
                        await asyncio.sleep(0.1)  # Brief pause for memory reclaim
                except ImportError:
                    pass
        
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
        max_concurrent: int | None = None,
        timeout_per_backend: float | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> ComparisonReport:
        """Compare circuit execution across multiple backends.

        This implements the full Step 5.1 workflow:
        1. Validate circuit on all backends
        2. Plan execution strategy
        3. Execute on each backend with resource management
        4. Analyze and compare results
        5. Generate report

        Args:
            adapters: List of backend adapters to compare
            circuit: Quantum circuit to execute
            options: Execution options (shots, etc.)
            strategy: Execution strategy (PARALLEL, SEQUENTIAL, or ADAPTIVE)
            max_concurrent: Maximum concurrent backend executions (default: auto-detect)
            timeout_per_backend: Timeout per backend in seconds (default: None - no timeout)
            cancellation_event: Event to signal cancellation for graceful shutdown

        Returns:
            ComparisonReport with full comparison results
        """
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        # Check for early cancellation
        if cancellation_event and cancellation_event.is_set():
            return ComparisonReport(
                circuit_info={"type": type(circuit).__name__},
                backends_compared=[a.get_name() for a in adapters],
                execution_strategy=strategy,
                total_time_ms=0.0,
                status=ComparisonStatus.FAILED,
                backend_results={},
                metrics=ComparisonMetrics(),
                errors=["Comparison cancelled before execution"],
            )

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

        # Step 4: Execute on each backend with enhanced resource management
        all_results: list[BackendResult] = []

        for batch_idx, batch in enumerate(batches):
            # Check for cancellation between batches
            if cancellation_event and cancellation_event.is_set():
                warnings.append(f"Cancelled after {batch_idx} of {len(batches)} batches")
                break
                
            batch_adapters = [
                adapter_map[name] for name in batch if name in adapter_map
            ]

            if actual_strategy == ExecutionStrategy.PARALLEL:
                batch_results = await self._execute_batch_parallel(
                    batch_adapters,
                    circuit,
                    options,
                    max_concurrent=max_concurrent,
                    timeout_per_backend=timeout_per_backend,
                    cancellation_event=cancellation_event,
                )
            else:
                batch_results = await self._execute_batch_sequential(
                    batch_adapters,
                    circuit,
                    options,
                    timeout_per_backend=timeout_per_backend,
                    cancellation_event=cancellation_event,
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


# =============================================================================
# Advanced Statistical Analysis (Feature - Comparison Aggregator)
# =============================================================================


@dataclass
class StatisticalSummary:
    """Statistical summary of a dataset."""
    
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # Interquartile range
    skewness: float
    kurtosis: float
    confidence_interval_95: tuple[float, float]


@dataclass
class HypothesisTestResult:
    """Result of a statistical hypothesis test."""
    
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # At alpha=0.05
    effect_size: float
    interpretation: str


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    
    variable_a: str
    variable_b: str
    correlation: float  # Pearson correlation coefficient
    p_value: float
    strength: str  # 'weak', 'moderate', 'strong'


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for comparison results.
    
    Provides:
    - Descriptive statistics
    - Hypothesis testing (t-test, Mann-Whitney U)
    - Effect size calculations (Cohen's d)
    - Correlation analysis
    - Confidence intervals
    - Outlier detection
    """
    
    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize analyzer with significance level."""
        self.alpha = alpha
    
    def compute_descriptive_stats(self, values: list[float]) -> StatisticalSummary:
        """Compute comprehensive descriptive statistics.
        
        Args:
            values: List of numeric values
            
        Returns:
            StatisticalSummary with all statistics
        """
        if not values:
            return StatisticalSummary(
                count=0, mean=0, std=0, min=0, max=0, median=0,
                q1=0, q3=0, iqr=0, skewness=0, kurtosis=0,
                confidence_interval_95=(0, 0),
            )
        
        arr = np.array(values)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        
        # Percentiles
        q1 = float(np.percentile(arr, 25))
        median = float(np.median(arr))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        
        # Skewness and kurtosis
        skewness = self._compute_skewness(arr, mean, std)
        kurtosis = self._compute_kurtosis(arr, mean, std)
        
        # 95% confidence interval for mean
        if n > 1 and std > 0:
            se = std / np.sqrt(n)
            # Using t-distribution critical value approximation
            t_crit = 1.96 if n > 30 else 2.0  # Simplified
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = ci_upper = mean
        
        return StatisticalSummary(
            count=n,
            mean=mean,
            std=std,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=median,
            q1=q1,
            q3=q3,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis,
            confidence_interval_95=(ci_lower, ci_upper),
        )
    
    def _compute_skewness(self, arr: np.ndarray, mean: float, std: float) -> float:
        """Compute skewness coefficient."""
        if std == 0 or len(arr) < 3:
            return 0.0
        n = len(arr)
        m3 = np.mean((arr - mean) ** 3)
        return float(m3 / (std ** 3) * np.sqrt(n * (n - 1)) / (n - 2)) if n > 2 else 0.0
    
    def _compute_kurtosis(self, arr: np.ndarray, mean: float, std: float) -> float:
        """Compute excess kurtosis."""
        if std == 0 or len(arr) < 4:
            return 0.0
        m4 = np.mean((arr - mean) ** 4)
        return float(m4 / (std ** 4) - 3)
    
    def welch_t_test(
        self,
        sample_a: list[float],
        sample_b: list[float],
        alternative: str = "two-sided",
    ) -> HypothesisTestResult:
        """Perform Welch's t-test for comparing two samples.
        
        This is more robust than Student's t-test when variances are unequal.
        
        Args:
            sample_a: First sample
            sample_b: Second sample
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            HypothesisTestResult with test outcome
        """
        if len(sample_a) < 2 or len(sample_b) < 2:
            return HypothesisTestResult(
                test_name="Welch's t-test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                interpretation="Insufficient data for test",
            )
        
        arr_a = np.array(sample_a)
        arr_b = np.array(sample_b)
        
        n_a, n_b = len(arr_a), len(arr_b)
        mean_a, mean_b = np.mean(arr_a), np.mean(arr_b)
        var_a, var_b = np.var(arr_a, ddof=1), np.var(arr_b, ddof=1)
        
        # Welch's t-statistic
        se = np.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return HypothesisTestResult(
                test_name="Welch's t-test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                interpretation="Zero variance in samples",
            )
        
        t_stat = float((mean_a - mean_b) / se)
        
        # Welch-Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else 1
        
        # Approximate p-value using normal distribution for large df
        p_value = self._t_distribution_pvalue(t_stat, df, alternative)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        cohens_d = float((mean_a - mean_b) / pooled_std) if pooled_std > 0 else 0.0
        
        # Interpretation
        effect_interpretation = self._interpret_effect_size(abs(cohens_d))
        significant = p_value < self.alpha
        
        if significant:
            direction = "greater than" if mean_a > mean_b else "less than"
            interpretation = f"Significant difference (p={p_value:.4f}). Sample A is {direction} Sample B with {effect_interpretation} effect."
        else:
            interpretation = f"No significant difference (p={p_value:.4f}). Effect size is {effect_interpretation}."
        
        return HypothesisTestResult(
            test_name="Welch's t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            effect_size=abs(cohens_d),
            interpretation=interpretation,
        )
    
    def _t_distribution_pvalue(self, t_stat: float, df: float, alternative: str) -> float:
        """Approximate p-value from t-distribution."""
        # Using normal approximation for simplicity
        # In production, use scipy.stats.t.sf
        from math import erf, sqrt
        
        # Standard normal CDF approximation
        def norm_cdf(x: float) -> float:
            return 0.5 * (1 + erf(x / sqrt(2)))
        
        if df > 30:
            # Use normal approximation
            if alternative == "two-sided":
                return 2 * (1 - norm_cdf(abs(t_stat)))
            elif alternative == "greater":
                return 1 - norm_cdf(t_stat)
            else:  # less
                return norm_cdf(t_stat)
        else:
            # Rough t-distribution approximation
            # Inflate p-value slightly for small df
            adjustment = 1 + 2 / df
            if alternative == "two-sided":
                return min(1.0, 2 * (1 - norm_cdf(abs(t_stat))) * adjustment)
            elif alternative == "greater":
                return min(1.0, (1 - norm_cdf(t_stat)) * adjustment)
            else:
                return min(1.0, norm_cdf(t_stat) * adjustment)
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def mann_whitney_u_test(
        self,
        sample_a: list[float],
        sample_b: list[float],
    ) -> HypothesisTestResult:
        """Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Useful when data is not normally distributed.
        
        Args:
            sample_a: First sample
            sample_b: Second sample
            
        Returns:
            HypothesisTestResult with test outcome
        """
        if len(sample_a) < 3 or len(sample_b) < 3:
            return HypothesisTestResult(
                test_name="Mann-Whitney U",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                interpretation="Insufficient data for test",
            )
        
        arr_a = np.array(sample_a)
        arr_b = np.array(sample_b)
        n_a, n_b = len(arr_a), len(arr_b)
        
        # Combine and rank
        combined = np.concatenate([arr_a, arr_b])
        ranks = self._rank_data(combined)
        
        # Calculate U statistics
        r_a = sum(ranks[:n_a])
        u_a = r_a - n_a * (n_a + 1) / 2
        u_b = n_a * n_b - u_a
        u = min(u_a, u_b)
        
        # Normal approximation for large samples
        mean_u = n_a * n_b / 2
        std_u = np.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
        
        if std_u > 0:
            z = (u - mean_u) / std_u
            from math import erf, sqrt
            p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        else:
            p_value = 1.0
            z = 0.0
        
        # Effect size (rank-biserial correlation)
        r = 1 - 2 * u / (n_a * n_b)
        
        significant = p_value < self.alpha
        
        if significant:
            interpretation = f"Significant difference (p={p_value:.4f}). Rank-biserial r={r:.3f}."
        else:
            interpretation = f"No significant difference (p={p_value:.4f})."
        
        return HypothesisTestResult(
            test_name="Mann-Whitney U",
            statistic=float(u),
            p_value=p_value,
            significant=significant,
            effect_size=abs(r),
            interpretation=interpretation,
        )
    
    def _rank_data(self, data: np.ndarray) -> np.ndarray:
        """Rank data with average rank for ties."""
        sorted_indices = np.argsort(data)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(1, len(data) + 1, dtype=float)
        
        # Handle ties by averaging ranks
        for val in np.unique(data):
            mask = data == val
            if np.sum(mask) > 1:
                ranks[mask] = np.mean(ranks[mask])
        
        return ranks
    
    def correlation_analysis(
        self,
        data_x: list[float],
        data_y: list[float],
        var_name_x: str = "X",
        var_name_y: str = "Y",
    ) -> CorrelationResult:
        """Compute Pearson correlation coefficient.
        
        Args:
            data_x: First variable values
            data_y: Second variable values
            var_name_x: Name of first variable
            var_name_y: Name of second variable
            
        Returns:
            CorrelationResult with correlation statistics
        """
        if len(data_x) != len(data_y) or len(data_x) < 3:
            return CorrelationResult(
                variable_a=var_name_x,
                variable_b=var_name_y,
                correlation=0.0,
                p_value=1.0,
                strength="none",
            )
        
        arr_x = np.array(data_x)
        arr_y = np.array(data_y)
        
        # Pearson correlation
        mean_x, mean_y = np.mean(arr_x), np.mean(arr_y)
        std_x, std_y = np.std(arr_x, ddof=1), np.std(arr_y, ddof=1)
        
        if std_x == 0 or std_y == 0:
            return CorrelationResult(
                variable_a=var_name_x,
                variable_b=var_name_y,
                correlation=0.0,
                p_value=1.0,
                strength="none",
            )
        
        cov = np.mean((arr_x - mean_x) * (arr_y - mean_y))
        r = float(cov / (std_x * std_y))
        
        # P-value approximation using t-distribution
        n = len(arr_x)
        if abs(r) < 1:
            t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
            p_value = self._t_distribution_pvalue(t_stat, n - 2, "two-sided")
        else:
            p_value = 0.0
        
        # Interpret strength
        abs_r = abs(r)
        if abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        return CorrelationResult(
            variable_a=var_name_x,
            variable_b=var_name_y,
            correlation=r,
            p_value=p_value,
            strength=strength,
        )
    
    def detect_outliers(
        self,
        values: list[float],
        method: str = "iqr",
    ) -> tuple[list[int], list[float]]:
        """Detect outliers in data.
        
        Args:
            values: Data values
            method: Detection method ('iqr' or 'zscore')
            
        Returns:
            Tuple of (outlier_indices, outlier_values)
        """
        if len(values) < 4:
            return [], []
        
        arr = np.array(values)
        outlier_mask = np.zeros(len(arr), dtype=bool)
        
        if method == "iqr":
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (arr < lower) | (arr > upper)
        elif method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            if std > 0:
                z_scores = np.abs((arr - mean) / std)
                outlier_mask = z_scores > 3
        
        indices = list(np.where(outlier_mask)[0])
        outlier_values = [float(arr[i]) for i in indices]
        
        return indices, outlier_values


# =============================================================================
# Visualization Generation (Feature - Comparison Aggregator)
# =============================================================================


@dataclass
class ComparisonVisualization:
    """Container for comparison visualizations."""
    
    execution_time_chart: str  # SVG or ASCII
    memory_chart: str
    agreement_heatmap: str
    performance_radar: str | None = None
    format: str = "svg"  # 'svg' or 'ascii'


class ComparisonVisualizer:
    """Generate visualizations for backend comparisons.
    
    Supports:
    - Bar charts for execution times
    - Memory usage comparisons
    - Agreement heatmaps
    - Performance radar charts
    - ASCII fallback for terminal output
    """
    
    COLORS = [
        "#3498db", "#2ecc71", "#e74c3c", "#f39c12",
        "#9b59b6", "#1abc9c", "#34495e", "#e67e22",
    ]
    
    def __init__(self, use_ascii: bool = False) -> None:
        """Initialize visualizer.
        
        Args:
            use_ascii: Use ASCII charts instead of SVG
        """
        self.use_ascii = use_ascii
    
    def generate_all(
        self,
        metrics: ComparisonMetrics,
    ) -> ComparisonVisualization:
        """Generate all visualizations for a comparison.
        
        Args:
            metrics: Comparison metrics to visualize
            
        Returns:
            ComparisonVisualization with all charts
        """
        if self.use_ascii:
            return ComparisonVisualization(
                execution_time_chart=self._ascii_bar_chart(
                    metrics.execution_times, "Execution Times (ms)"
                ),
                memory_chart=self._ascii_bar_chart(
                    metrics.memory_peaks, "Memory Usage (MB)"
                ),
                agreement_heatmap=self._ascii_agreement_matrix(
                    metrics.pairwise_agreements
                ),
                format="ascii",
            )
        else:
            return ComparisonVisualization(
                execution_time_chart=self._svg_bar_chart(
                    metrics.execution_times, "Execution Times (ms)"
                ),
                memory_chart=self._svg_bar_chart(
                    metrics.memory_peaks, "Memory Usage (MB)"
                ),
                agreement_heatmap=self._svg_heatmap(
                    metrics.pairwise_agreements, "Result Agreement"
                ),
                performance_radar=self._svg_radar_chart(metrics),
                format="svg",
            )
    
    def _ascii_bar_chart(
        self,
        data: dict[str, float],
        title: str,
        width: int = 40,
    ) -> str:
        """Generate ASCII bar chart."""
        if not data:
            return f"{title}\n  No data available"
        
        lines = [f"  {title}", "  " + "=" * len(title)]
        max_val = max(data.values()) if data.values() else 1
        max_label = max(len(str(k)) for k in data.keys())
        
        for label, value in sorted(data.items(), key=lambda x: -x[1]):
            bar_len = int((value / max_val) * width) if max_val > 0 else 0
            bar = "█" * bar_len
            label_str = str(label).ljust(max_label)
            lines.append(f"  {label_str} │{bar} {value:.2f}")
        
        return "\n".join(lines)
    
    def _ascii_agreement_matrix(
        self,
        agreements: dict[str, dict[str, float]],
    ) -> str:
        """Generate ASCII agreement matrix."""
        if not agreements:
            return "  Agreement Matrix\n  No data available"
        
        backends = list(agreements.keys())
        n = len(backends)
        
        # Header
        max_label = max(len(b) for b in backends)
        header = " " * (max_label + 3) + "  ".join(b[:6].ljust(6) for b in backends)
        
        lines = ["  Agreement Matrix (%)", "  " + "=" * 20, header]
        
        for b1 in backends:
            row_values = []
            for b2 in backends:
                val = agreements.get(b1, {}).get(b2, 0)
                row_values.append(f"{val*100:5.1f}%")
            lines.append(f"  {b1.ljust(max_label)} │ {' '.join(row_values)}")
        
        return "\n".join(lines)
    
    def _svg_bar_chart(
        self,
        data: dict[str, float],
        title: str,
        width: int = 500,
        height: int = 300,
    ) -> str:
        """Generate SVG bar chart."""
        if not data:
            return f'<svg width="{width}" height="{height}"><text x="10" y="20">No data</text></svg>'
        
        margin = 50
        bar_width = (width - 2 * margin) / len(data)
        max_val = max(data.values()) if data.values() else 1
        
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="20" text-anchor="middle" font-weight="bold">{title}</text>',
        ]
        
        for i, (label, value) in enumerate(data.items()):
            x = margin + i * bar_width + bar_width * 0.1
            bar_h = (value / max_val) * (height - 2 * margin - 30) if max_val > 0 else 0
            y = height - margin - bar_h
            color = self.COLORS[i % len(self.COLORS)]
            
            svg.append(
                f'<rect x="{x}" y="{y}" width="{bar_width * 0.8}" height="{bar_h}" '
                f'fill="{color}" rx="3"/>'
            )
            svg.append(
                f'<text x="{x + bar_width * 0.4}" y="{height - margin + 15}" '
                f'text-anchor="middle" font-size="10">{label[:8]}</text>'
            )
            svg.append(
                f'<text x="{x + bar_width * 0.4}" y="{y - 5}" '
                f'text-anchor="middle" font-size="9">{value:.1f}</text>'
            )
        
        svg.append('</svg>')
        return '\n'.join(svg)
    
    def _svg_heatmap(
        self,
        matrix: dict[str, dict[str, float]],
        title: str,
        width: int = 400,
        height: int = 400,
    ) -> str:
        """Generate SVG heatmap for agreement matrix."""
        if not matrix:
            return f'<svg width="{width}" height="{height}"><text x="10" y="20">No data</text></svg>'
        
        backends = list(matrix.keys())
        n = len(backends)
        margin = 80
        cell_size = (min(width, height) - 2 * margin) / n
        
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="20" text-anchor="middle" font-weight="bold">{title}</text>',
        ]
        
        # Draw cells
        for i, b1 in enumerate(backends):
            for j, b2 in enumerate(backends):
                value = matrix.get(b1, {}).get(b2, 0)
                x = margin + j * cell_size
                y = margin + i * cell_size
                
                # Color based on value (green gradient)
                intensity = int(value * 200)
                color = f"rgb({255 - intensity}, {155 + intensity // 2}, {155 - intensity // 2})"
                
                svg.append(
                    f'<rect x="{x}" y="{y}" width="{cell_size-1}" height="{cell_size-1}" '
                    f'fill="{color}" stroke="#ccc"/>'
                )
                
                # Value text
                svg.append(
                    f'<text x="{x + cell_size/2}" y="{y + cell_size/2 + 4}" '
                    f'text-anchor="middle" font-size="10">{value*100:.0f}%</text>'
                )
        
        # Row labels
        for i, b in enumerate(backends):
            y = margin + i * cell_size + cell_size / 2 + 4
            svg.append(
                f'<text x="{margin - 5}" y="{y}" text-anchor="end" font-size="10">{b[:10]}</text>'
            )
        
        # Column labels
        for j, b in enumerate(backends):
            x = margin + j * cell_size + cell_size / 2
            svg.append(
                f'<text x="{x}" y="{margin - 5}" text-anchor="middle" font-size="10" '
                f'transform="rotate(-45 {x} {margin - 5})">{b[:10]}</text>'
            )
        
        svg.append('</svg>')
        return '\n'.join(svg)
    
    def _svg_radar_chart(
        self,
        metrics: ComparisonMetrics,
        width: int = 400,
        height: int = 400,
    ) -> str:
        """Generate SVG radar chart for multi-dimensional comparison."""
        if not metrics.execution_times:
            return ""
        
        backends = list(metrics.execution_times.keys())
        if len(backends) < 2:
            return ""
        
        # Normalize metrics to 0-1 scale
        dimensions = ["Speed", "Memory", "Agreement"]
        
        # Calculate scores for each backend
        scores: dict[str, list[float]] = {}
        
        max_time = max(metrics.execution_times.values()) or 1
        max_mem = max(metrics.memory_peaks.values()) if metrics.memory_peaks else 1
        
        for backend in backends:
            time_score = 1 - (metrics.execution_times.get(backend, max_time) / max_time)
            mem_score = 1 - (metrics.memory_peaks.get(backend, max_mem) / max_mem) if max_mem > 0 else 0.5
            
            # Average agreement with other backends
            agreements = metrics.pairwise_agreements.get(backend, {})
            other_agreements = [v for k, v in agreements.items() if k != backend]
            agree_score = sum(other_agreements) / len(other_agreements) if other_agreements else 0.5
            
            scores[backend] = [time_score, mem_score, agree_score]
        
        # Draw radar chart
        center_x, center_y = width / 2, height / 2
        radius = min(width, height) / 2 - 60
        
        import math
        angle_step = 2 * math.pi / len(dimensions)
        
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="20" text-anchor="middle" font-weight="bold">Performance Radar</text>',
        ]
        
        # Draw grid
        for level in [0.25, 0.5, 0.75, 1.0]:
            points = []
            for i in range(len(dimensions)):
                angle = i * angle_step - math.pi / 2
                x = center_x + radius * level * math.cos(angle)
                y = center_y + radius * level * math.sin(angle)
                points.append(f"{x},{y}")
            svg.append(
                f'<polygon points="{" ".join(points)}" fill="none" stroke="#ddd"/>'
            )
        
        # Draw axes and labels
        for i, dim in enumerate(dimensions):
            angle = i * angle_step - math.pi / 2
            x = center_x + radius * 1.1 * math.cos(angle)
            y = center_y + radius * 1.1 * math.sin(angle)
            svg.append(
                f'<line x1="{center_x}" y1="{center_y}" x2="{center_x + radius * math.cos(angle)}" '
                f'y2="{center_y + radius * math.sin(angle)}" stroke="#ccc"/>'
            )
            svg.append(
                f'<text x="{x}" y="{y}" text-anchor="middle" font-size="10">{dim}</text>'
            )
        
        # Draw data for each backend
        for idx, (backend, score_list) in enumerate(scores.items()):
            color = self.COLORS[idx % len(self.COLORS)]
            points = []
            for i, score in enumerate(score_list):
                angle = i * angle_step - math.pi / 2
                x = center_x + radius * score * math.cos(angle)
                y = center_y + radius * score * math.sin(angle)
                points.append(f"{x},{y}")
            svg.append(
                f'<polygon points="{" ".join(points)}" fill="{color}" fill-opacity="0.3" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        
        # Legend
        for idx, backend in enumerate(backends):
            color = self.COLORS[idx % len(self.COLORS)]
            y = height - 40 + idx * 15
            svg.append(f'<rect x="10" y="{y}" width="10" height="10" fill="{color}"/>')
            svg.append(f'<text x="25" y="{y + 9}" font-size="10">{backend}</text>')
        
        svg.append('</svg>')
        return '\n'.join(svg)


# =============================================================================
# Result Significance Testing (Feature - Comparison Aggregator)
# =============================================================================


@dataclass
class SignificanceTestSuite:
    """Results from a suite of significance tests."""
    
    t_test: HypothesisTestResult | None = None
    mann_whitney: HypothesisTestResult | None = None
    effect_size: float = 0.0
    power_estimate: float = 0.0
    sample_size_recommendation: int = 0
    overall_conclusion: str = ""


class ResultSignificanceTester:
    """Test statistical significance of comparison results.
    
    Provides:
    - Multiple test battery
    - Effect size calculations
    - Power analysis
    - Sample size recommendations
    - Confidence assessments
    """
    
    def __init__(self, alpha: float = 0.05, power_target: float = 0.8) -> None:
        """Initialize tester.
        
        Args:
            alpha: Significance level
            power_target: Desired statistical power
        """
        self.alpha = alpha
        self.power_target = power_target
        self._analyzer = AdvancedStatisticalAnalyzer(alpha)
    
    def test_execution_time_difference(
        self,
        times_a: list[float],
        times_b: list[float],
        backend_a: str = "Backend A",
        backend_b: str = "Backend B",
    ) -> SignificanceTestSuite:
        """Test if execution time difference is statistically significant.
        
        Runs multiple tests to ensure robust conclusions.
        
        Args:
            times_a: Execution times for first backend
            times_b: Execution times for second backend
            backend_a: Name of first backend
            backend_b: Name of second backend
            
        Returns:
            SignificanceTestSuite with all test results
        """
        suite = SignificanceTestSuite()
        
        if len(times_a) < 2 or len(times_b) < 2:
            suite.overall_conclusion = "Insufficient data for significance testing"
            return suite
        
        # Parametric test (Welch's t-test)
        suite.t_test = self._analyzer.welch_t_test(times_a, times_b)
        
        # Non-parametric test (Mann-Whitney U)
        suite.mann_whitney = self._analyzer.mann_whitney_u_test(times_a, times_b)
        
        # Effect size (Cohen's d)
        suite.effect_size = suite.t_test.effect_size if suite.t_test else 0.0
        
        # Power estimate
        suite.power_estimate = self._estimate_power(
            len(times_a), len(times_b), suite.effect_size
        )
        
        # Sample size recommendation
        suite.sample_size_recommendation = self._recommend_sample_size(suite.effect_size)
        
        # Overall conclusion
        suite.overall_conclusion = self._generate_conclusion(
            suite, backend_a, backend_b, times_a, times_b
        )
        
        return suite
    
    def test_result_agreement_significance(
        self,
        agreements: list[float],
        threshold: float = 0.95,
    ) -> HypothesisTestResult:
        """Test if result agreement is significantly above a threshold.
        
        Uses one-sample t-test against threshold.
        
        Args:
            agreements: List of agreement scores
            threshold: Threshold to test against
            
        Returns:
            HypothesisTestResult
        """
        if len(agreements) < 3:
            return HypothesisTestResult(
                test_name="One-sample t-test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                interpretation="Insufficient data",
            )
        
        arr = np.array(agreements)
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        
        if std == 0:
            significant = mean > threshold
            return HypothesisTestResult(
                test_name="One-sample t-test",
                statistic=float('inf') if mean > threshold else float('-inf'),
                p_value=0.0 if mean > threshold else 1.0,
                significant=significant,
                effect_size=0.0,
                interpretation=f"All values equal to {mean:.3f}",
            )
        
        t_stat = (mean - threshold) / (std / np.sqrt(n))
        p_value = self._analyzer._t_distribution_pvalue(float(t_stat), n - 1, "greater")
        
        effect_size = (mean - threshold) / std
        significant = p_value < self.alpha
        
        if significant:
            interpretation = f"Agreement ({mean:.1%}) is significantly above {threshold:.0%} (p={p_value:.4f})"
        else:
            interpretation = f"Agreement ({mean:.1%}) is not significantly above {threshold:.0%} (p={p_value:.4f})"
        
        return HypothesisTestResult(
            test_name="One-sample t-test (vs threshold)",
            statistic=float(t_stat),
            p_value=p_value,
            significant=significant,
            effect_size=abs(effect_size),
            interpretation=interpretation,
        )
    
    def _estimate_power(self, n1: int, n2: int, effect_size: float) -> float:
        """Estimate statistical power given sample sizes and effect size."""
        if effect_size == 0:
            return self.alpha  # Power equals alpha when no effect
        
        # Simplified power calculation
        # In production, use scipy.stats.power
        n = 2 / (1/n1 + 1/n2)  # Harmonic mean
        ncp = effect_size * np.sqrt(n / 2)  # Non-centrality parameter
        
        # Rough approximation using normal distribution
        from math import erf, sqrt
        z_alpha = 1.96 if self.alpha == 0.05 else 1.645
        power = 0.5 * (1 + erf((ncp - z_alpha) / sqrt(2)))
        
        return min(0.99, max(0.01, power))
    
    def _recommend_sample_size(self, effect_size: float) -> int:
        """Recommend sample size for desired power."""
        if effect_size == 0:
            return 100  # Default for unknown effect
        
        # Sample size formula for two-sample t-test
        # n = 2 * ((z_alpha + z_beta) / d)^2
        z_alpha = 1.96  # For alpha = 0.05
        z_beta = 0.84   # For power = 0.8
        
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return max(10, int(np.ceil(n_per_group)))
    
    def _generate_conclusion(
        self,
        suite: SignificanceTestSuite,
        backend_a: str,
        backend_b: str,
        times_a: list[float],
        times_b: list[float],
    ) -> str:
        """Generate overall conclusion from test suite."""
        mean_a = np.mean(times_a)
        mean_b = np.mean(times_b)
        
        # Check agreement between tests
        t_sig = suite.t_test.significant if suite.t_test else False
        mw_sig = suite.mann_whitney.significant if suite.mann_whitney else False
        
        conclusions = []
        
        if t_sig and mw_sig:
            faster = backend_a if mean_a < mean_b else backend_b
            conclusions.append(
                f"Both parametric and non-parametric tests agree: "
                f"{faster} is significantly faster."
            )
        elif t_sig or mw_sig:
            conclusions.append(
                "Tests show mixed results. Consider collecting more data."
            )
        else:
            conclusions.append(
                f"No significant difference detected between {backend_a} and {backend_b}."
            )
        
        # Effect size interpretation
        effect_interp = self._analyzer._interpret_effect_size(suite.effect_size)
        conclusions.append(f"Effect size: {effect_interp} (d={suite.effect_size:.2f})")
        
        # Power assessment
        if suite.power_estimate < 0.5:
            conclusions.append(
                f"⚠️ Low statistical power ({suite.power_estimate:.0%}). "
                f"Recommend {suite.sample_size_recommendation} samples per backend."
            )
        elif suite.power_estimate < self.power_target:
            conclusions.append(
                f"Moderate power ({suite.power_estimate:.0%}). "
                f"Consider {suite.sample_size_recommendation} samples for more reliable results."
            )
        else:
            conclusions.append(f"Good statistical power ({suite.power_estimate:.0%}).")
        
        return " ".join(conclusions)
