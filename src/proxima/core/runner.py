"""Quantum circuit runner implementation.

Converts plans to quantum circuits and executes them using backend adapters.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cirq

from proxima.backends.base import SimulatorType
from proxima.backends.registry import BackendRegistry
from proxima.utils.logging import get_logger

logger = get_logger("runner")


def create_bell_state_circuit() -> cirq.Circuit:
    """Create a 2-qubit Bell state circuit."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),  # Hadamard on qubit 0
        cirq.CNOT(q0, q1),  # CNOT with control=q0, target=q1
        cirq.measure(q0, q1, key="result"),  # Measure both qubits
    )
    return circuit


def create_ghz_state_circuit(num_qubits: int = 3) -> cirq.Circuit:
    """Create a GHZ state circuit."""
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),  # Hadamard on first qubit
    )
    # CNOT cascade
    for i in range(num_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit


def create_teleportation_circuit() -> cirq.Circuit:
    """Create a quantum teleportation circuit.

    Teleports the state of qubit 0 to qubit 2 using qubit 1 as an entangled resource.
    """
    q0, q1, q2 = cirq.LineQubit.range(3)

    circuit = cirq.Circuit(
        # Prepare state to teleport (arbitrary superposition)
        cirq.H(q0),
        cirq.T(q0),
        # Create Bell pair between q1 and q2
        cirq.H(q1),
        cirq.CNOT(q1, q2),
        # Bell measurement on q0 and q1
        cirq.CNOT(q0, q1),
        cirq.H(q0),
        cirq.measure(q0, q1, key="bell_measurement"),
        # Conditional corrections on q2 (simplified for simulation)
        # In real implementation, these would be conditional on measurements
        # For demonstration, we measure the final state
        cirq.measure(q2, key="result"),
    )
    return circuit


def parse_objective(objective: str) -> dict[str, Any]:
    """Parse natural language objective into circuit specification.

    Args:
        objective: Natural language description of the quantum circuit

    Returns:
        Dictionary with circuit_type and parameters
    """
    objective_lower = objective.lower()

    if "bell" in objective_lower:
        return {"circuit_type": "bell", "qubits": 2}
    elif "ghz" in objective_lower:
        # Try to extract qubit count
        import re

        match = re.search(r"(\d+)[-\s]*qubit", objective_lower)
        num_qubits = int(match.group(1)) if match else 3
        return {"circuit_type": "ghz", "qubits": num_qubits}
    elif "teleport" in objective_lower:
        return {"circuit_type": "teleportation", "qubits": 3}
    else:
        # Default to simple bell state
        return {"circuit_type": "bell", "qubits": 2}


def quantum_runner(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute a quantum circuit based on the plan.

    Args:
        plan: Execution plan with objective and configuration

    Returns:
        Execution results including counts and metadata
    """
    logger.info("runner.start", plan=plan)

    # Extract configuration from plan
    objective = plan.get("objective", "demo")
    backend_name = plan.get("backend", "cirq")
    shots = plan.get("shots", 1024)
    timeout_seconds = plan.get("timeout_seconds")

    # Handle auto backend selection
    if backend_name == "auto":
        registry = BackendRegistry()
        registry.discover()
        available = registry.list_available()
        if available:
            backend_name = available[0]  # Use first available backend
            logger.info(
                "runner.auto_backend_selected",
                backend=backend_name,
                available=available,
            )
        else:
            return {
                "status": "error",
                "error": "No backends available",
            }

    # Parse objective to determine circuit type
    circuit_spec = parse_objective(objective)
    logger.info("runner.parsed", circuit_spec=circuit_spec)

    # Create circuit based on type
    if circuit_spec["circuit_type"] == "bell":
        circuit = create_bell_state_circuit()
    elif circuit_spec["circuit_type"] == "ghz":
        circuit = create_ghz_state_circuit(circuit_spec["qubits"])
    elif circuit_spec["circuit_type"] == "teleportation":
        circuit = create_teleportation_circuit()
    else:
        # Default
        circuit = create_bell_state_circuit()

    logger.info("runner.circuit_created", qubits=len(circuit.all_qubits()))

    # Get backend adapter
    registry = BackendRegistry()
    registry.discover()  # Discover available backends
    adapter = registry.get(backend_name)

    if not adapter.is_available():
        return {
            "status": "error",
            "error": f"Backend {backend_name} not available",
        }

    # Execute circuit
    options = {
        "simulator_type": SimulatorType.STATE_VECTOR,
        "shots": shots,
        "repetitions": shots,
    }

    # Add timeout if specified
    if timeout_seconds is not None:
        options["timeout_seconds"] = timeout_seconds

    try:
        result = adapter.execute(circuit, options)
        logger.info(
            "runner.executed", backend=backend_name, time_ms=result.execution_time_ms
        )

        # Format results
        counts = result.data.get("counts", {})

        # Calculate percentages and format output
        total_shots = sum(counts.values())
        formatted_counts = {}
        for state, count in sorted(counts.items(), key=lambda x: -x[1]):
            percentage = (count / total_shots * 100) if total_shots > 0 else 0
            formatted_counts[state] = {
                "count": count,
                "percentage": round(percentage, 2),
            }

        return {
            "status": "success",
            "backend": backend_name,
            "circuit_type": circuit_spec["circuit_type"],
            "qubits": circuit_spec["qubits"],
            "shots": shots,
            "execution_time_ms": result.execution_time_ms,
            "counts": formatted_counts,
            "raw_counts": counts,
        }

    except Exception as e:
        logger.error("runner.failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


# ==============================================================================
# ERROR AGGREGATION (5% Gap Coverage)
# ==============================================================================


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    
    BACKEND_UNAVAILABLE = "backend_unavailable"
    CIRCUIT_INVALID = "circuit_invalid"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    QUANTUM_ERROR = "quantum_error"
    UNKNOWN = "unknown"


@dataclass
class AggregatedError:
    """Aggregated error information."""
    
    category: ErrorCategory
    message: str
    count: int = 1
    first_occurrence: float = field(default_factory=time.time)
    last_occurrence: float = field(default_factory=time.time)
    affected_tasks: list[str] = field(default_factory=list)
    sample_stacktrace: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "message": self.message,
            "count": self.count,
            "first_occurrence": self.first_occurrence,
            "last_occurrence": self.last_occurrence,
            "affected_tasks": self.affected_tasks[:10],  # Limit to first 10
            "total_affected": len(self.affected_tasks),
        }


@dataclass
class BatchExecutionStats:
    """Statistics for batch execution."""
    
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    retried_runs: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    success_rate: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "retried_runs": self.retried_runs,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms if self.min_duration_ms != float("inf") else 0,
            "max_duration_ms": self.max_duration_ms,
            "success_rate": self.success_rate,
        }


@dataclass
class ErrorPattern:
    """Pattern detected in errors."""
    
    pattern_id: str
    description: str
    category: ErrorCategory
    frequency: int
    time_window_s: float
    affected_backends: list[str] = field(default_factory=list)
    suggested_action: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "category": self.category.value,
            "frequency": self.frequency,
            "time_window_s": self.time_window_s,
            "affected_backends": self.affected_backends,
            "suggested_action": self.suggested_action,
        }


@dataclass
class RunResult:
    """Result of a single run in batch execution."""
    
    run_id: str
    plan: dict[str, Any]
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    error_category: ErrorCategory | None = None
    duration_ms: float = 0.0
    retries: int = 0
    backend: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "success": self.success,
            "error": self.error,
            "error_category": self.error_category.value if self.error_category else None,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
            "backend": self.backend,
            "timestamp": self.timestamp,
        }


class ErrorAggregator:
    """Aggregates and analyzes errors across multiple executions.
    
    Features:
    - Error collection and categorization
    - Pattern detection
    - Retry statistics
    - Trend analysis
    - Actionable insights
    """
    
    def __init__(
        self,
        pattern_detection_window_s: float = 60.0,
        pattern_threshold: int = 3,
    ) -> None:
        """Initialize error aggregator.
        
        Args:
            pattern_detection_window_s: Time window for pattern detection
            pattern_threshold: Minimum occurrences to detect pattern
        """
        self._pattern_window = pattern_detection_window_s
        self._pattern_threshold = pattern_threshold
        
        self._errors: list[dict[str, Any]] = []
        self._aggregated: dict[str, AggregatedError] = {}
        self._run_results: list[RunResult] = []
        self._patterns: list[ErrorPattern] = []
        self._lock = threading.RLock()
    
    def categorize_error(self, error_msg: str) -> ErrorCategory:
        """Categorize an error message.
        
        Args:
            error_msg: Error message to categorize
            
        Returns:
            Error category
        """
        msg_lower = error_msg.lower()
        
        if "not available" in msg_lower or "unavailable" in msg_lower:
            return ErrorCategory.BACKEND_UNAVAILABLE
        
        if "timeout" in msg_lower or "timed out" in msg_lower:
            return ErrorCategory.TIMEOUT
        
        if "circuit" in msg_lower and ("invalid" in msg_lower or "error" in msg_lower):
            return ErrorCategory.CIRCUIT_INVALID
        
        if "memory" in msg_lower or "resource" in msg_lower or "exhausted" in msg_lower:
            return ErrorCategory.RESOURCE_EXHAUSTED
        
        if "network" in msg_lower or "connection" in msg_lower:
            return ErrorCategory.NETWORK
        
        if "config" in msg_lower or "parameter" in msg_lower or "setting" in msg_lower:
            return ErrorCategory.CONFIGURATION
        
        if "qubit" in msg_lower or "gate" in msg_lower or "measurement" in msg_lower:
            return ErrorCategory.QUANTUM_ERROR
        
        return ErrorCategory.UNKNOWN
    
    def record_error(
        self,
        task_id: str,
        error_msg: str,
        backend: str = "",
        stacktrace: str | None = None,
    ) -> None:
        """Record an error occurrence.
        
        Args:
            task_id: ID of failed task
            error_msg: Error message
            backend: Backend that produced error
            stacktrace: Optional stack trace
        """
        with self._lock:
            category = self.categorize_error(error_msg)
            now = time.time()
            
            # Create aggregation key
            key = f"{category.value}:{error_msg[:100]}"
            
            if key in self._aggregated:
                agg = self._aggregated[key]
                agg.count += 1
                agg.last_occurrence = now
                agg.affected_tasks.append(task_id)
            else:
                self._aggregated[key] = AggregatedError(
                    category=category,
                    message=error_msg,
                    count=1,
                    first_occurrence=now,
                    last_occurrence=now,
                    affected_tasks=[task_id],
                    sample_stacktrace=stacktrace,
                )
            
            # Store raw error for pattern detection
            self._errors.append({
                "timestamp": now,
                "category": category,
                "message": error_msg,
                "backend": backend,
                "task_id": task_id,
            })
            
            # Cleanup old errors (keep last hour)
            cutoff = now - 3600
            self._errors = [e for e in self._errors if e["timestamp"] > cutoff]
    
    def record_run_result(self, result: RunResult) -> None:
        """Record a run result.
        
        Args:
            result: Run result to record
        """
        with self._lock:
            self._run_results.append(result)
            
            if not result.success and result.error:
                self.record_error(
                    result.run_id,
                    result.error,
                    result.backend,
                )
            
            # Keep only last 1000 results
            if len(self._run_results) > 1000:
                self._run_results = self._run_results[-1000:]
    
    def detect_patterns(self) -> list[ErrorPattern]:
        """Detect error patterns in recent errors.
        
        Returns:
            List of detected patterns
        """
        with self._lock:
            now = time.time()
            cutoff = now - self._pattern_window
            recent = [e for e in self._errors if e["timestamp"] > cutoff]
            
            if len(recent) < self._pattern_threshold:
                return []
            
            patterns: list[ErrorPattern] = []
            
            # Pattern 1: Category concentration
            category_counts = Counter(e["category"] for e in recent)
            for category, count in category_counts.items():
                if count >= self._pattern_threshold:
                    backends = list(set(
                        e["backend"] for e in recent 
                        if e["category"] == category and e["backend"]
                    ))
                    
                    pattern = ErrorPattern(
                        pattern_id=f"category_spike_{category.value}",
                        description=f"High frequency of {category.value} errors",
                        category=category,
                        frequency=count,
                        time_window_s=self._pattern_window,
                        affected_backends=backends,
                        suggested_action=self._suggest_action(category),
                    )
                    patterns.append(pattern)
            
            # Pattern 2: Backend-specific errors
            backend_counts = Counter(e["backend"] for e in recent if e["backend"])
            for backend, count in backend_counts.items():
                if count >= self._pattern_threshold:
                    backend_errors = [e for e in recent if e["backend"] == backend]
                    main_category = Counter(e["category"] for e in backend_errors).most_common(1)
                    
                    pattern = ErrorPattern(
                        pattern_id=f"backend_errors_{backend}",
                        description=f"Multiple errors from backend {backend}",
                        category=main_category[0][0] if main_category else ErrorCategory.UNKNOWN,
                        frequency=count,
                        time_window_s=self._pattern_window,
                        affected_backends=[backend],
                        suggested_action=f"Consider switching away from {backend} or investigating backend health",
                    )
                    patterns.append(pattern)
            
            # Pattern 3: Rapid failures
            if len(recent) >= 5:
                time_span = recent[-1]["timestamp"] - recent[0]["timestamp"]
                if time_span < 10:  # 5+ errors in 10 seconds
                    pattern = ErrorPattern(
                        pattern_id="rapid_failures",
                        description="Rapid succession of failures detected",
                        category=ErrorCategory.UNKNOWN,
                        frequency=len(recent),
                        time_window_s=time_span,
                        suggested_action="Consider pausing execution and investigating system state",
                    )
                    patterns.append(pattern)
            
            self._patterns = patterns
            return patterns
    
    def _suggest_action(self, category: ErrorCategory) -> str:
        """Suggest action for error category."""
        suggestions = {
            ErrorCategory.BACKEND_UNAVAILABLE: "Check backend availability or switch to alternative backend",
            ErrorCategory.CIRCUIT_INVALID: "Review circuit construction and validate gates",
            ErrorCategory.TIMEOUT: "Increase timeout or reduce circuit complexity",
            ErrorCategory.RESOURCE_EXHAUSTED: "Reduce parallelism or batch size",
            ErrorCategory.NETWORK: "Check network connectivity and retry with backoff",
            ErrorCategory.CONFIGURATION: "Review configuration parameters",
            ErrorCategory.QUANTUM_ERROR: "Check qubit count and gate compatibility",
            ErrorCategory.UNKNOWN: "Review error logs for more details",
        }
        return suggestions.get(category, "Investigate error details")
    
    def get_stats(self) -> BatchExecutionStats:
        """Get execution statistics.
        
        Returns:
            Aggregated statistics
        """
        with self._lock:
            stats = BatchExecutionStats()
            
            if not self._run_results:
                return stats
            
            stats.total_runs = len(self._run_results)
            stats.successful_runs = sum(1 for r in self._run_results if r.success)
            stats.failed_runs = stats.total_runs - stats.successful_runs
            stats.retried_runs = sum(1 for r in self._run_results if r.retries > 0)
            
            durations = [r.duration_ms for r in self._run_results if r.duration_ms > 0]
            if durations:
                stats.total_duration_ms = sum(durations)
                stats.avg_duration_ms = stats.total_duration_ms / len(durations)
                stats.min_duration_ms = min(durations)
                stats.max_duration_ms = max(durations)
            
            if stats.total_runs > 0:
                stats.success_rate = stats.successful_runs / stats.total_runs
            
            return stats
    
    def get_aggregated_errors(
        self,
        category: ErrorCategory | None = None,
        min_count: int = 1,
    ) -> list[dict[str, Any]]:
        """Get aggregated errors.
        
        Args:
            category: Filter by category
            min_count: Minimum occurrence count
            
        Returns:
            List of aggregated errors
        """
        with self._lock:
            result = []
            for agg in self._aggregated.values():
                if agg.count < min_count:
                    continue
                if category and agg.category != category:
                    continue
                result.append(agg.to_dict())
            
            return sorted(result, key=lambda x: x["count"], reverse=True)
    
    def get_error_summary(self) -> dict[str, Any]:
        """Get comprehensive error summary.
        
        Returns:
            Summary with stats, patterns, and recommendations
        """
        with self._lock:
            stats = self.get_stats()
            patterns = self.detect_patterns()
            
            # Get top errors by category
            by_category: dict[str, int] = {}
            for agg in self._aggregated.values():
                cat = agg.category.value
                by_category[cat] = by_category.get(cat, 0) + agg.count
            
            return {
                "stats": stats.to_dict(),
                "patterns": [p.to_dict() for p in patterns],
                "by_category": by_category,
                "top_errors": self.get_aggregated_errors(min_count=2)[:10],
                "total_unique_errors": len(self._aggregated),
                "total_error_occurrences": sum(a.count for a in self._aggregated.values()),
                "recommendations": self._generate_recommendations(patterns, by_category),
            }
    
    def _generate_recommendations(
        self,
        patterns: list[ErrorPattern],
        by_category: dict[str, int],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        total_errors = sum(by_category.values())
        
        if total_errors == 0:
            return ["No errors detected - system healthy"]
        
        # High error rate recommendation
        stats = self.get_stats()
        if stats.success_rate < 0.9:
            recommendations.append(
                f"Success rate is {stats.success_rate:.0%} - investigate common failures"
            )
        
        # Category-specific recommendations
        if by_category.get("backend_unavailable", 0) > 5:
            recommendations.append(
                "Multiple backend availability issues - check backend health or add fallbacks"
            )
        
        if by_category.get("timeout", 0) > 3:
            recommendations.append(
                "Frequent timeouts - consider increasing timeout limits or optimizing circuits"
            )
        
        if by_category.get("resource_exhausted", 0) > 2:
            recommendations.append(
                "Resource exhaustion detected - reduce parallelism or batch size"
            )
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.suggested_action and pattern.suggested_action not in recommendations:
                recommendations.append(pattern.suggested_action)
        
        return recommendations[:5]  # Limit to top 5
    
    def clear(self) -> None:
        """Clear all recorded errors and results."""
        with self._lock:
            self._errors.clear()
            self._aggregated.clear()
            self._run_results.clear()
            self._patterns.clear()


class BatchRunner:
    """Batch execution runner with error aggregation.
    
    Features:
    - Execute multiple plans
    - Aggregate errors across runs
    - Retry failed runs
    - Collect execution statistics
    """
    
    def __init__(
        self,
        max_retries: int = 2,
        retry_delay_base: float = 1.0,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> None:
        """Initialize batch runner.
        
        Args:
            max_retries: Maximum retry attempts per plan
            retry_delay_base: Base delay for exponential backoff
            parallel: Run in parallel
            max_workers: Maximum parallel workers
        """
        self._max_retries = max_retries
        self._retry_delay_base = retry_delay_base
        self._parallel = parallel
        self._max_workers = max_workers
        self._aggregator = ErrorAggregator()
        self._logger = get_logger("batch_runner")
    
    def run_batch(
        self,
        plans: list[dict[str, Any]],
        stop_on_error: bool = False,
    ) -> dict[str, Any]:
        """Run a batch of plans.
        
        Args:
            plans: List of execution plans
            stop_on_error: Stop on first error
            
        Returns:
            Batch results with aggregated errors
        """
        results: list[RunResult] = []
        start_time = time.time()
        
        for idx, plan in enumerate(plans):
            run_id = f"run_{idx}_{int(time.time())}"
            
            result = self._execute_with_retry(run_id, plan)
            results.append(result)
            self._aggregator.record_run_result(result)
            
            if stop_on_error and not result.success:
                self._logger.info("batch.stopped_on_error", run_id=run_id)
                break
        
        total_duration = (time.time() - start_time) * 1000
        
        return {
            "total_plans": len(plans),
            "executed": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "total_duration_ms": total_duration,
            "results": [r.to_dict() for r in results],
            "error_summary": self._aggregator.get_error_summary(),
        }
    
    def _execute_with_retry(
        self,
        run_id: str,
        plan: dict[str, Any],
    ) -> RunResult:
        """Execute a plan with retry logic.
        
        Args:
            run_id: Run identifier
            plan: Execution plan
            
        Returns:
            Run result
        """
        backend = plan.get("backend", "auto")
        last_error = None
        retries = 0
        
        for attempt in range(self._max_retries + 1):
            start = time.time()
            
            try:
                result = quantum_runner(plan)
                duration = (time.time() - start) * 1000
                
                if result.get("status") == "success":
                    return RunResult(
                        run_id=run_id,
                        plan=plan,
                        success=True,
                        result=result,
                        duration_ms=duration,
                        retries=retries,
                        backend=result.get("backend", backend),
                    )
                else:
                    last_error = result.get("error", "Unknown error")
                    error_category = self._aggregator.categorize_error(last_error)
                    
            except Exception as e:
                last_error = str(e)
                error_category = self._aggregator.categorize_error(last_error)
                duration = (time.time() - start) * 1000
            
            # Retry with backoff
            if attempt < self._max_retries:
                retries += 1
                delay = self._retry_delay_base * (2 ** attempt)
                time.sleep(delay)
        
        return RunResult(
            run_id=run_id,
            plan=plan,
            success=False,
            error=last_error,
            error_category=error_category,
            duration_ms=duration,
            retries=retries,
            backend=backend,
        )
    
    def get_error_aggregator(self) -> ErrorAggregator:
        """Get the error aggregator instance."""
        return self._aggregator
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._aggregator.clear()


# ==================== MODULE EXPORTS ====================

__all__ = [
    # Circuit creation
    "create_bell_state_circuit",
    "create_ghz_state_circuit",
    "create_teleportation_circuit",
    # Parsing
    "parse_objective",
    # Runner
    "quantum_runner",
    # Error Aggregation
    "ErrorCategory",
    "AggregatedError",
    "BatchExecutionStats",
    "ErrorPattern",
    "RunResult",
    "ErrorAggregator",
    "BatchRunner",
]
