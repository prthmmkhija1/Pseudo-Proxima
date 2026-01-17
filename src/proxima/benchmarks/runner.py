"""Benchmark runner for Proxima.

Implements single-run and multi-run benchmarking with timing and resource monitoring.
Includes Phase 10 optimizations: validation, error handling, and adaptive sampling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

from proxima.backends.registry import BackendRegistry
from proxima.benchmarks.circuit_analyzer import CircuitAnalyzer
from proxima.data.metrics import (
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkStatus,
)
from proxima.resources.benchmark_timer import BenchmarkTimer
from proxima.resources.monitor import ResourceMonitor

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRunnerConfig:
    """Configuration for BenchmarkRunner.

    Attributes:
        default_shots: Default number of measurement shots per run.
        require_backend_available: If True, raise error when backend unavailable.
        store_results: If True, persist results to BenchmarkRegistry.

    Example:
        >>> config = BenchmarkRunnerConfig(default_shots=2048, store_results=False)
    """

    default_shots: int = 1024
    require_backend_available: bool = True
    store_results: bool = True


@dataclass
class BenchmarkRunner:
    """Orchestrates benchmark execution with timing and resource monitoring.

    Provides single-run and multi-run benchmarking capabilities with automatic
    collection of execution time, memory usage, CPU/GPU utilization, and throughput.

    Attributes:
        registry: BackendRegistry instance to resolve backend names.
        results_storage: Optional BenchmarkRegistry for persisting results.
        config: Runner configuration options.

    Example:
        >>> from proxima.backends.registry import BackendRegistry
        >>> backend_registry = BackendRegistry()
        >>> backend_registry.discover()
        >>> runner = BenchmarkRunner(backend_registry)
        >>> result = runner.run_benchmark(circuit, "lret", shots=1024)
        >>> print(result.metrics.execution_time_ms)
    """

    registry: BackendRegistry
    results_storage: Any | None = None
    config: BenchmarkRunnerConfig = field(default_factory=BenchmarkRunnerConfig)

    def run_benchmark(
        self,
        circuit: Any,
        backend_name: str,
        shots: int = 1024,
        metadata: Optional[dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Execute a single benchmark run.

        Args:
            circuit: Circuit to execute (QASM string, path, or circuit object).
            backend_name: Name of the backend to use.
            shots: Number of measurement shots.
            metadata: Optional metadata to attach to the result.

        Returns:
            BenchmarkResult containing metrics and status.

        Raises:
            KeyError: If backend_name is not found in registry.
            ValueError: If circuit validation fails and require_backend_available is True.
        """
        metadata = metadata or {}
        circuit_hash = self._hash_circuit(circuit)
        result = BenchmarkResult(circuit_hash=circuit_hash, metadata=metadata)

        # Phase 10.2: Validation layers
        from proxima.benchmarks.error_handling import (
            CircuitValidator,
            BackendValidator,
            ResultValidator,
            BenchmarkErrorCodes,
        )

        # Validate circuit before benchmarking
        circuit_validation = CircuitValidator.validate(circuit, backend_name)
        if not circuit_validation.valid:
            logger.warning(
                "Circuit validation failed: %s",
                json.dumps({"errors": circuit_validation.errors}),
            )
            if self.config.require_backend_available:
                raise ValueError(f"Circuit validation failed: {circuit_validation.errors}")

        for warning in circuit_validation.warnings:
            logger.info("Circuit validation warning: %s", warning)

        # Validate backend is available
        backend_validation = BackendValidator.validate(backend_name, self.registry)
        if not backend_validation.valid:
            logger.warning(
                "Backend validation failed: %s",
                json.dumps({"errors": backend_validation.errors}),
            )
            if self.config.require_backend_available:
                raise ValueError(f"Backend validation failed: {backend_validation.errors}")

        # Acquire backend
        backend = self.registry.get(backend_name)

        # Analyze circuit characteristics
        circuit_info = CircuitAnalyzer.analyze_circuit(circuit)

        monitor = ResourceMonitor()
        timer = BenchmarkTimer()

        # Phase 10.2: Comprehensive error handling with structured logging
        monitor.start_monitoring()
        try:
            timer.start()
            backend.execute(circuit, shots=shots)
            timer.stop()
            status = BenchmarkStatus.SUCCESS
            error_message = None
        except Exception as exc:  # noqa: BLE001
            if timer.is_running:
                timer.stop()
            status = BenchmarkStatus.FAILED
            error_message = str(exc)
            logger.error(
                "Backend execution failed: %s",
                json.dumps({
                    "backend": backend_name,
                    "error": str(exc),
                    "code": BenchmarkErrorCodes.BACKEND_EXECUTION_FAILED,
                }),
            )
        finally:
            try:
                monitor.stop_monitoring()
            except Exception as monitor_exc:
                logger.warning(
                    "Resource monitor stop failed: %s",
                    json.dumps({
                        "error": str(monitor_exc),
                        "code": BenchmarkErrorCodes.RESOURCE_MONITOR_FAILED,
                    }),
                )

        exec_ms = timer.elapsed_ms()
        throughput = shots / (exec_ms / 1000.0) if exec_ms > 0 else 0.0
        success_rate = 100.0 if status == BenchmarkStatus.SUCCESS else 0.0

        metrics = BenchmarkMetrics(
            execution_time_ms=exec_ms,
            memory_peak_mb=monitor.get_peak_memory_mb(),
            memory_baseline_mb=monitor.get_memory_baseline_mb(),
            throughput_shots_per_sec=throughput,
            success_rate_percent=success_rate,
            cpu_usage_percent=monitor.get_average_cpu_percent(),
            gpu_usage_percent=monitor.get_average_gpu_percent(),
            timestamp=datetime.utcnow(),
            backend_name=backend_name,
            circuit_info=circuit_info,
        )

        result.metrics = metrics
        result.status = status
        result.error_message = error_message

        # Phase 10.2: Validate result before storage
        result_validation = ResultValidator.validate(result)
        if not result_validation.valid:
            logger.warning(
                "Result validation failed: %s",
                json.dumps({"errors": result_validation.errors}),
            )

        for warning in result_validation.warnings:
            logger.info("Result validation warning: %s", warning)

        # Phase 10.2: Wrap storage in try/except with structured logging
        if self.results_storage and getattr(self.config, "store_results", True):
            try:
                self.results_storage.save_result(result)
            except Exception as storage_exc:
                # Storage failures should not break benchmark execution
                logger.warning(
                    "Result storage failed: %s",
                    json.dumps({
                        "error": str(storage_exc),
                        "code": BenchmarkErrorCodes.DATABASE_QUERY_FAILED,
                    }),
                )

        return result

    def run_benchmark_suite(
        self,
        circuit: Any,
        backend_name: str,
        num_runs: int = 5,
        shots: int = 1024,
        warmup_runs: int = 0,
    ) -> BenchmarkResult:
        # Warmup phase (no recording)
        backend = self.registry.get(backend_name)
        for _ in range(warmup_runs):
            try:
                backend.execute(circuit, shots=shots)
            except Exception:
                break

        # Measurement phase
        results: list[BenchmarkResult] = []
        for _ in range(num_runs):
            res = self.run_benchmark(circuit, backend_name, shots=shots)
            results.append(res)

        # Aggregate statistics
        exec_times = [r.metrics.execution_time_ms for r in results if r.metrics]
        success_runs = [r for r in results if r.status == BenchmarkStatus.SUCCESS]
        success_rate = (len(success_runs) / len(results) * 100.0) if results else 0.0

        if exec_times:
            avg_time = statistics.mean(exec_times)
            min_time = min(exec_times)
            max_time = max(exec_times)
            median_time = statistics.median(exec_times)
            stddev_time = statistics.stdev(exec_times) if len(exec_times) > 1 else 0.0
        else:
            avg_time = min_time = max_time = median_time = stddev_time = 0.0

        # Use first available metrics as base for non-time fields
        base_metrics = next((r.metrics for r in results if r.metrics), None)
        circuit_info = base_metrics.circuit_info if base_metrics else {}
        backend_name_final = base_metrics.backend_name if base_metrics else backend_name

        aggregated_metrics = BenchmarkMetrics(
            execution_time_ms=avg_time,
            memory_peak_mb=max(
                (r.metrics.memory_peak_mb for r in results if r.metrics),
                default=0.0,
            ),
            memory_baseline_mb=min(
                (r.metrics.memory_baseline_mb for r in results if r.metrics),
                default=0.0,
            ),
            throughput_shots_per_sec=(shots / (avg_time / 1000.0)) if avg_time > 0 else 0.0,
            success_rate_percent=success_rate,
            cpu_usage_percent=statistics.mean(
                [r.metrics.cpu_usage_percent for r in results if r.metrics]
            )
            if results
            else 0.0,
            gpu_usage_percent=self._safe_mean_gpu(
                [r.metrics.gpu_usage_percent for r in results if r.metrics and r.metrics.gpu_usage_percent is not None]
            ),
            timestamp=datetime.utcnow(),
            backend_name=backend_name_final,
            circuit_info=circuit_info,
        )

        aggregated_result = BenchmarkResult(
            metrics=aggregated_metrics,
            metadata={
                "individual_runs": [r.to_dict() for r in results],
                "statistics": {
                    "avg_time_ms": avg_time,
                    "min_time_ms": min_time,
                    "max_time_ms": max_time,
                    "median_time_ms": median_time,
                    "stddev_time_ms": stddev_time,
                    "success_rate_percent": success_rate,
                },
            },
            status=BenchmarkStatus.SUCCESS
            if success_rate == 100.0 and results
            else BenchmarkStatus.FAILED,
        )

        if self.results_storage and getattr(self.config, "store_results", True):
            try:
                self.results_storage.save_result(aggregated_result)
            except Exception:
                pass

        return aggregated_result

    @staticmethod
    def _safe_mean_gpu(values: list[float]) -> Optional[float]:
        """Safely compute mean for GPU values, returning None if empty."""
        if not values:
            return None
        try:
            return statistics.mean(values)
        except statistics.StatisticsError:
            return None

    @staticmethod
    def _hash_circuit(circuit: Any) -> str:
        """Generate SHA256 hash of circuit's JSON representation."""
        try:
            if hasattr(circuit, "to_json"):
                payload = circuit.to_json()
            elif hasattr(circuit, "dict"):
                payload = json.dumps(circuit.dict(), sort_keys=True)
            elif hasattr(circuit, "model_dump"):
                payload = json.dumps(circuit.model_dump(), sort_keys=True)
            else:
                payload = json.dumps(circuit, default=str, sort_keys=True)
        except Exception:
            payload = str(circuit)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def run_comparison(
        self,
        circuit: Any,
        backend_names: list[str],
        shots: int = 1024,
        num_runs: int = 3,
    ) -> dict[str, BenchmarkResult]:
        """Run benchmarks across multiple backends for comparison.

        Args:
            circuit: Circuit to execute.
            backend_names: List of backend names to compare.
            shots: Number of shots per run.
            num_runs: Number of runs per backend.

        Returns:
            Dictionary mapping backend names to their aggregated results.
        """
        results = {}
        for backend_name in backend_names:
            try:
                if num_runs > 1:
                    results[backend_name] = self.run_benchmark_suite(
                        circuit, backend_name, num_runs=num_runs, shots=shots
                    )
                else:
                    results[backend_name] = self.run_benchmark(
                        circuit, backend_name, shots=shots
                    )
            except Exception as e:
                logger.error(f"Failed to benchmark {backend_name}: {e}")
                results[backend_name] = BenchmarkResult(
                    status=BenchmarkStatus.FAILED,
                    error_message=str(e),
                )
        return results

    def run_scaling_benchmark(
        self,
        circuit_generator: Any,
        backend_name: str,
        qubit_range: Iterable[int],
        shots: int = 1024,
    ) -> list[BenchmarkResult]:
        """Run benchmark with scaling number of qubits.

        Args:
            circuit_generator: Callable that takes num_qubits and returns circuit.
            backend_name: Backend to use.
            qubit_range: Iterable of qubit counts to test.
            shots: Number of shots per run.

        Returns:
            List of BenchmarkResults for each qubit count.
        """
        results = []
        for num_qubits in qubit_range:
            try:
                circuit = circuit_generator(num_qubits)
                result = self.run_benchmark(
                    circuit, backend_name, shots=shots,
                    metadata={"num_qubits": num_qubits}
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Scaling benchmark failed at {num_qubits} qubits: {e}")
                results.append(BenchmarkResult(
                    status=BenchmarkStatus.FAILED,
                    error_message=str(e),
                    metadata={"num_qubits": num_qubits},
                ))
        return results

    def run_parallel_benchmarks(
        self,
        benchmark_configs: list[dict[str, Any]],
        max_workers: int = 4,
    ) -> list[BenchmarkResult]:
        """Run multiple benchmarks in parallel.

        Args:
            benchmark_configs: List of dicts with 'circuit', 'backend_name', 
                              and optional 'shots' keys.
            max_workers: Maximum number of parallel workers.

        Returns:
            List of BenchmarkResults in the same order as configs.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[BenchmarkResult | None] = [None] * len(benchmark_configs)

        def run_single(index: int, config: dict) -> tuple[int, BenchmarkResult]:
            try:
                result = self.run_benchmark(
                    circuit=config["circuit"],
                    backend_name=config["backend_name"],
                    shots=config.get("shots", self.config.default_shots),
                    metadata=config.get("metadata"),
                )
                return index, result
            except Exception as e:
                return index, BenchmarkResult(
                    status=BenchmarkStatus.FAILED,
                    error_message=str(e),
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single, i, cfg): i
                for i, cfg in enumerate(benchmark_configs)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return [r for r in results if r is not None]


# =============================================================================
# Standard Circuit Generators
# =============================================================================


class StandardCircuits:
    """Standard quantum circuit generators for benchmarking.

    Provides commonly used circuits for performance testing:
    - Bell state preparation
    - GHZ state preparation
    - Random circuits
    - QFT (Quantum Fourier Transform)
    - Grover's algorithm
    """

    @staticmethod
    def bell_state(backend_format: str = "qasm") -> str:
        """Generate a 2-qubit Bell state circuit.

        Args:
            backend_format: Output format ('qasm', 'cirq', 'qiskit').

        Returns:
            Circuit in the requested format.
        """
        qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
        if backend_format == "qasm":
            return qasm
        # For other formats, return QASM - adapter will convert
        return qasm

    @staticmethod
    def ghz_state(num_qubits: int = 3, backend_format: str = "qasm") -> str:
        """Generate a GHZ (Greenberger-Horne-Zeilinger) state circuit.

        Args:
            num_qubits: Number of qubits (minimum 2).
            backend_format: Output format.

        Returns:
            Circuit in the requested format.
        """
        num_qubits = max(2, num_qubits)
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
            'h q[0];',
        ]
        for i in range(num_qubits - 1):
            lines.append(f'cx q[{i}], q[{i + 1}];')
        lines.append('measure q -> c;')
        return '\n'.join(lines)

    @staticmethod
    def random_circuit(
        num_qubits: int,
        depth: int,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a random quantum circuit.

        Args:
            num_qubits: Number of qubits.
            depth: Circuit depth (layers of gates).
            seed: Random seed for reproducibility.

        Returns:
            QASM representation of the circuit.
        """
        import random
        if seed is not None:
            random.seed(seed)

        single_gates = ['h', 'x', 'y', 'z', 's', 't']
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]

        for _ in range(depth):
            # Random single-qubit gates
            for q in range(num_qubits):
                gate = random.choice(single_gates)
                lines.append(f'{gate} q[{q}];')
            # Random CNOT layer
            targets = list(range(num_qubits))
            random.shuffle(targets)
            for i in range(0, len(targets) - 1, 2):
                lines.append(f'cx q[{targets[i]}], q[{targets[i + 1]}];')

        lines.append('measure q -> c;')
        return '\n'.join(lines)

    @staticmethod
    def qft_circuit(num_qubits: int) -> str:
        """Generate a Quantum Fourier Transform circuit.

        Args:
            num_qubits: Number of qubits.

        Returns:
            QASM representation of the QFT circuit.
        """
        import math
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]

        for j in range(num_qubits):
            lines.append(f'h q[{j}];')
            for k in range(j + 1, num_qubits):
                angle = math.pi / (2 ** (k - j))
                lines.append(f'cu1({angle}) q[{k}], q[{j}];')

        # Swap qubits for proper output ordering
        for i in range(num_qubits // 2):
            lines.append(f'swap q[{i}], q[{num_qubits - 1 - i}];')

        lines.append('measure q -> c;')
        return '\n'.join(lines)

    @staticmethod
    def grover_iteration(num_qubits: int, target_state: int = 0) -> str:
        """Generate a single Grover iteration circuit.

        Args:
            num_qubits: Number of qubits (oracle + ancilla).
            target_state: Target state to search for.

        Returns:
            QASM representation of Grover iteration.
        """
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]

        # Initialize superposition
        for i in range(num_qubits):
            lines.append(f'h q[{i}];')

        # Oracle (mark target state with Z gate pattern)
        for i in range(num_qubits):
            if not (target_state & (1 << i)):
                lines.append(f'x q[{i}];')

        # Multi-controlled Z (simplified as CZ chain)
        if num_qubits >= 2:
            lines.append(f'h q[{num_qubits - 1}];')
            for i in range(num_qubits - 1):
                lines.append(f'cx q[{i}], q[{num_qubits - 1}];')
            lines.append(f'h q[{num_qubits - 1}];')

        for i in range(num_qubits):
            if not (target_state & (1 << i)):
                lines.append(f'x q[{i}];')

        # Diffusion operator
        for i in range(num_qubits):
            lines.append(f'h q[{i}];')
            lines.append(f'x q[{i}];')

        if num_qubits >= 2:
            lines.append(f'h q[{num_qubits - 1}];')
            for i in range(num_qubits - 1):
                lines.append(f'cx q[{i}], q[{num_qubits - 1}];')
            lines.append(f'h q[{num_qubits - 1}];')

        for i in range(num_qubits):
            lines.append(f'x q[{i}];')
            lines.append(f'h q[{i}];')

        lines.append('measure q -> c;')
        return '\n'.join(lines)

    @staticmethod
    def variational_circuit(
        num_qubits: int,
        num_layers: int = 2,
        params: Optional[list[float]] = None,
    ) -> str:
        """Generate a parameterized variational circuit.

        Args:
            num_qubits: Number of qubits.
            num_layers: Number of variational layers.
            params: Optional list of rotation angles.

        Returns:
            QASM representation with parameterized rotations.
        """
        import math
        if params is None:
            params = [0.5 * math.pi] * (num_qubits * num_layers * 3)

        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]

        param_idx = 0
        for layer in range(num_layers):
            # Rotation layer
            for q in range(num_qubits):
                if param_idx < len(params):
                    lines.append(f'rx({params[param_idx]}) q[{q}];')
                    param_idx += 1
                if param_idx < len(params):
                    lines.append(f'ry({params[param_idx]}) q[{q}];')
                    param_idx += 1
                if param_idx < len(params):
                    lines.append(f'rz({params[param_idx]}) q[{q}];')
                    param_idx += 1

            # Entangling layer
            for q in range(num_qubits - 1):
                lines.append(f'cx q[{q}], q[{q + 1}];')
            if num_qubits > 1:
                lines.append(f'cx q[{num_qubits - 1}], q[0];')

        lines.append('measure q -> c;')
        return '\n'.join(lines)


# =============================================================================
# Benchmark Orchestrator - Advanced Execution Control
# =============================================================================


@dataclass
class BenchmarkPlan:
    """A planned benchmark execution."""
    circuit: Any
    backend_name: str
    shots: int = 1024
    num_runs: int = 1
    warmup_runs: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority runs first


class BenchmarkOrchestrator:
    """Advanced benchmark orchestration with scheduling and resource management.

    Features:
    - Priority-based scheduling
    - Resource-aware execution
    - Automatic retries
    - Progress callbacks
    - Result aggregation
    """

    def __init__(
        self,
        runner: BenchmarkRunner,
        max_concurrent: int = 1,
        retry_count: int = 2,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            runner: BenchmarkRunner instance.
            max_concurrent: Maximum concurrent benchmarks.
            retry_count: Number of retries on failure.
        """
        self._runner = runner
        self._max_concurrent = max_concurrent
        self._retry_count = retry_count
        self._progress_callbacks: list[Any] = []
        self._plans: list[BenchmarkPlan] = []
        self._results: dict[str, BenchmarkResult] = {}

    def add_benchmark(self, plan: BenchmarkPlan) -> str:
        """Add a benchmark plan to the queue.

        Args:
            plan: Benchmark plan to add.

        Returns:
            Unique ID for the plan.
        """
        plan_id = f"plan_{len(self._plans)}_{hash(plan.backend_name)}"
        self._plans.append(plan)
        return plan_id

    def add_progress_callback(self, callback: Any) -> None:
        """Add a progress callback function.

        Args:
            callback: Callable that receives (completed, total, current_plan).
        """
        self._progress_callbacks.append(callback)

    def _notify_progress(self, completed: int, total: int, current: Optional[BenchmarkPlan]) -> None:
        """Notify all progress callbacks."""
        for cb in self._progress_callbacks:
            try:
                cb(completed, total, current)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def execute_all(self) -> dict[str, BenchmarkResult]:
        """Execute all planned benchmarks.

        Returns:
            Dictionary mapping plan IDs to results.
        """
        # Sort by priority (descending)
        sorted_plans = sorted(self._plans, key=lambda p: -p.priority)
        total = len(sorted_plans)

        for idx, plan in enumerate(sorted_plans):
            self._notify_progress(idx, total, plan)
            plan_id = f"plan_{idx}_{plan.backend_name}"

            result = None
            last_error = None

            for attempt in range(self._retry_count + 1):
                try:
                    if plan.num_runs > 1:
                        result = self._runner.run_benchmark_suite(
                            circuit=plan.circuit,
                            backend_name=plan.backend_name,
                            shots=plan.shots,
                            num_runs=plan.num_runs,
                            warmup_runs=plan.warmup_runs,
                        )
                    else:
                        result = self._runner.run_benchmark(
                            circuit=plan.circuit,
                            backend_name=plan.backend_name,
                            shots=plan.shots,
                            metadata=plan.metadata,
                        )

                    if result.status == BenchmarkStatus.SUCCESS:
                        break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Benchmark attempt {attempt + 1} failed: {e}")

            if result is None:
                result = BenchmarkResult(
                    status=BenchmarkStatus.FAILED,
                    error_message=str(last_error) if last_error else "Unknown error",
                )

            self._results[plan_id] = result

        self._notify_progress(total, total, None)
        return self._results

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary.

        Returns:
            Summary dictionary with statistics.
        """
        total = len(self._results)
        successful = sum(1 for r in self._results.values() if r.status == BenchmarkStatus.SUCCESS)
        failed = total - successful

        exec_times = [
            r.metrics.execution_time_ms
            for r in self._results.values()
            if r.metrics and r.status == BenchmarkStatus.SUCCESS
        ]

        return {
            "total_benchmarks": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "total_execution_time_ms": sum(exec_times),
            "avg_execution_time_ms": statistics.mean(exec_times) if exec_times else 0,
        }

    def clear(self) -> None:
        """Clear all plans and results."""
        self._plans.clear()
        self._results.clear()
