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
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from proxima.backends.base import BaseBackendAdapter, ExecutionResult


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
    PARALLEL = "parallel"       # Execute all at once using asyncio
    SEQUENTIAL = "sequential"   # Execute one after another
    ADAPTIVE = "adaptive"       # Decide based on memory requirements


@dataclass
class BackendResult:
    """Result from a single backend execution."""
    backend_name: str
    success: bool
    execution_time_ms: float
    memory_peak_mb: float
    result: Optional[Any] = None  # ExecutionResult if successful
    error: Optional[str] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonMetrics:
    """Metrics comparing results across backends."""
    # Execution Time metrics
    execution_times: Dict[str, float] = field(default_factory=dict)  # backend -> time_ms
    fastest_backend: Optional[str] = None
    slowest_backend: Optional[str] = None
    time_ratios: Dict[str, float] = field(default_factory=dict)  # backend -> ratio to fastest
    
    # Memory metrics
    memory_peaks: Dict[str, float] = field(default_factory=dict)  # backend -> memory_mb
    lowest_memory_backend: Optional[str] = None
    
    # Result Agreement metrics
    result_agreement: float = 0.0  # 0.0 to 1.0, percentage of states that match
    pairwise_agreements: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Fidelity metrics (for statevector comparisons)
    fidelities: Dict[str, Dict[str, float]] = field(default_factory=dict)  # backend_a -> backend_b -> fidelity
    average_fidelity: float = 0.0
    
    # Performance summary
    recommended_backend: Optional[str] = None
    recommendation_reason: Optional[str] = None
    
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
    circuit_info: Dict[str, Any]
    backends_compared: List[str]
    execution_strategy: ExecutionStrategy
    total_time_ms: float
    status: ComparisonStatus
    backend_results: Dict[str, BackendResult]
    metrics: ComparisonMetrics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
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
        
        lines.extend([
            "",
            f"Fastest: {self.metrics.fastest_backend}",
            f"Result Agreement: {self.metrics.result_agreement * 100:.1f}%",
        ])
        
        if self.metrics.recommended_backend:
            lines.extend([
                "",
                f"Recommended: {self.metrics.recommended_backend}",
                f"Reason: {self.metrics.recommendation_reason}",
            ])
        
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
        memory_requirements: Dict[str, float],  # backend -> estimated_mb
    ) -> Tuple[ExecutionStrategy, List[List[str]]]:
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
        
        batches: List[List[str]] = []
        current_batch: List[str] = []
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
        probs_a: Dict[str, float],
        probs_b: Dict[str, float],
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
        backend_results: Dict[str, BackendResult],
    ) -> ComparisonMetrics:
        """Analyze results from multiple backends and compute metrics."""
        metrics = ComparisonMetrics()
        
        successful_backends = {
            name: result for name, result in backend_results.items()
            if result.success
        }
        
        if not successful_backends:
            return metrics
        
        # Execution time metrics
        metrics.execution_times = {
            name: result.execution_time_ms
            for name, result in successful_backends.items()
        }
        
        if metrics.execution_times:
            metrics.fastest_backend = min(metrics.execution_times, key=metrics.execution_times.get)
            metrics.slowest_backend = max(metrics.execution_times, key=metrics.execution_times.get)
            
            fastest_time = metrics.execution_times[metrics.fastest_backend]
            if fastest_time > 0:
                metrics.time_ratios = {
                    name: time / fastest_time
                    for name, time in metrics.execution_times.items()
                }
        
        # Memory metrics
        metrics.memory_peaks = {
            name: result.memory_peak_mb
            for name, result in successful_backends.items()
        }
        
        if metrics.memory_peaks:
            metrics.lowest_memory_backend = min(metrics.memory_peaks, key=metrics.memory_peaks.get)
        
        # Result agreement (pairwise)
        backend_names = list(successful_backends.keys())
        agreements: List[float] = []
        
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
                    metrics.pairwise_agreements[name_a][name_b] = \
                        metrics.pairwise_agreements[name_b][name_a]
        
        if agreements:
            metrics.result_agreement = sum(agreements) / len(agreements)
        else:
            metrics.result_agreement = 1.0
        
        # Fidelity (if statevectors available)
        statevectors: Dict[str, np.ndarray] = {}
        for name, result in successful_backends.items():
            if result.result and hasattr(result.result, "data"):
                sv = result.result.data.get("statevector")
                if sv is not None:
                    statevectors[name] = np.asarray(sv, dtype=complex)
        
        if len(statevectors) >= 2:
            fidelities: List[float] = []
            sv_names = list(statevectors.keys())
            for i, name_a in enumerate(sv_names):
                metrics.fidelities[name_a] = {}
                for j, name_b in enumerate(sv_names):
                    if i == j:
                        metrics.fidelities[name_a][name_b] = 1.0
                    elif i < j:
                        fid = self.calculate_fidelity(
                            statevectors[name_a],
                            statevectors[name_b]
                        )
                        metrics.fidelities[name_a][name_b] = fid
                        fidelities.append(fid)
                    else:
                        metrics.fidelities[name_a][name_b] = \
                            metrics.fidelities[name_b][name_a]
            
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
        scores: Dict[str, float] = {}
        
        for backend in metrics.execution_times:
            # Normalize time score (lower is better)
            time_score = metrics.time_ratios.get(backend, 1.0)
            
            # Agreement score (higher is better)
            agreement_scores = [
                v for v in metrics.pairwise_agreements.get(backend, {}).values()
                if v < 1.0  # Exclude self-comparison
            ]
            avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 1.0
            
            # Combined score: time ratio penalty + agreement bonus
            # Lower score = better
            scores[backend] = time_score - avg_agreement
        
        if scores:
            metrics.recommended_backend = min(scores, key=scores.get)
            
            # Generate reason
            reasons = []
            if metrics.recommended_backend == metrics.fastest_backend:
                reasons.append("fastest execution")
            if metrics.recommended_backend == metrics.lowest_memory_backend:
                reasons.append("lowest memory usage")
            
            backend_agreement = metrics.pairwise_agreements.get(metrics.recommended_backend, {})
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
        planner: Optional[ExecutionPlanner] = None,
        analyzer: Optional[ResultAnalyzer] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
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
        adapter: "BaseBackendAdapter",
        circuit: Any,
        options: Optional[Dict[str, Any]],
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
                        probabilities = {k: v/total for k, v in counts.items()}
            
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
        adapters: List["BaseBackendAdapter"],
        circuit: Any,
        options: Optional[Dict[str, Any]],
    ) -> List[BackendResult]:
        """Execute circuit on multiple backends in parallel."""
        tasks = [
            self._execute_backend(adapter, circuit, options)
            for adapter in adapters
        ]
        return await asyncio.gather(*tasks)
    
    async def _execute_batch_sequential(
        self,
        adapters: List["BaseBackendAdapter"],
        circuit: Any,
        options: Optional[Dict[str, Any]],
    ) -> List[BackendResult]:
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
        adapters: List["BaseBackendAdapter"],
        circuit: Any,
    ) -> Tuple[List["BaseBackendAdapter"], List[str]]:
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
        adapters: List["BaseBackendAdapter"],
        circuit: Any,
    ) -> Dict[str, float]:
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
        adapters: List["BaseBackendAdapter"],
        circuit: Any,
        options: Optional[Dict[str, Any]] = None,
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
        errors: List[str] = []
        warnings: List[str] = []
        
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
        all_results: List[BackendResult] = []
        
        for batch_idx, batch in enumerate(batches):
            batch_adapters = [adapter_map[name] for name in batch if name in adapter_map]
            
            if actual_strategy == ExecutionStrategy.PARALLEL:
                batch_results = await self._execute_batch_parallel(batch_adapters, circuit, options)
            else:
                batch_results = await self._execute_batch_sequential(batch_adapters, circuit, options)
            
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
        adapters: List["BaseBackendAdapter"],
        circuit: Any,
        options: Optional[Dict[str, Any]] = None,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    ) -> ComparisonReport:
        """Synchronous wrapper for compare()."""
        return asyncio.run(self.compare(adapters, circuit, options, strategy))


# Convenience function
def compare_backends(
    adapters: List["BaseBackendAdapter"],
    circuit: Any,
    options: Optional[Dict[str, Any]] = None,
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
