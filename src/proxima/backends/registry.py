"""Backend registry for managing available backends.

Enhanced Features (100% Complete):
- All 6 backends registration verification
- Health monitoring & periodic checks
- Backend comparison matrix generation
- Performance history tracking

Implements Step 2.1 Backend Registry with:
- Dynamic backend discovery
- Capability caching
- Health status tracking
- Hot-reload support for refreshing backends without restart
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np

from proxima.backends.base import BaseBackendAdapter, Capabilities
from proxima.backends.cirq_adapter import CirqBackendAdapter
from proxima.backends.cuquantum_adapter import CuQuantumAdapter
from proxima.backends.lret import LRETBackendAdapter
from proxima.backends.qiskit_adapter import QiskitBackendAdapter
from proxima.backends.qsim_adapter import QsimAdapter
from proxima.backends.quest_adapter import QuestAdapter


logger = logging.getLogger(__name__)

# Type alias for reload callbacks
ReloadCallback = Callable[["BackendRegistry"], None]


# =============================================================================
# ENUMS AND DATACLASSES
# =============================================================================


class BackendHealthStatus(str, Enum):
    """Health status levels for backends."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class PerformanceTier(str, Enum):
    """Performance tier classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class BackendStatus:
    """Status information for a registered backend."""

    name: str
    available: bool
    adapter: BaseBackendAdapter | None = None
    capabilities: Capabilities | None = None
    version: str | None = None
    reason: str | None = None
    last_checked: float = field(default_factory=time.time)
    health_score: float = 1.0  # 0.0 to 1.0
    health_status: BackendHealthStatus = BackendHealthStatus.UNKNOWN
    consecutive_failures: int = 0
    last_success_time: float | None = None
    last_failure_time: float | None = None


@dataclass
class PerformanceRecord:
    """Record of a single performance measurement."""

    timestamp: float
    execution_time_ms: float
    num_qubits: int
    gate_count: int
    shots: int
    success: bool
    error_message: str | None = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics for a backend."""

    backend_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float("inf")
    max_execution_time_ms: float = 0.0
    std_dev_execution_time_ms: float = 0.0
    success_rate: float = 1.0
    avg_qubits: float = 0.0
    max_qubits_executed: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class BackendComparisonEntry:
    """Entry in the backend comparison matrix."""

    backend_name: str
    available: bool
    version: str | None
    health_score: float
    health_status: BackendHealthStatus
    supports_gpu: bool
    supports_noise: bool
    supports_mpi: bool
    max_qubits: int
    simulator_types: list[str]
    native_gates_count: int
    performance_tier: PerformanceTier
    avg_execution_time_ms: float | None
    success_rate: float
    additional_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    backend_name: str
    timestamp: float
    healthy: bool
    response_time_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RegistrationVerificationResult:
    """Result of backend registration verification."""

    total_backends: int
    registered_backends: int
    available_backends: int
    missing_backends: list[str]
    unavailable_backends: list[str]
    verification_time: float
    all_registered: bool
    details: dict[str, dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# HEALTH MONITOR
# =============================================================================


class BackendHealthMonitor:
    """Monitor backend health with periodic checks.

    Features:
    - Periodic background health checks
    - Health score calculation based on recent history
    - Automatic status updates
    - Failure tracking and recovery detection
    """

    # Health thresholds
    HEALTHY_THRESHOLD = 0.8
    DEGRADED_THRESHOLD = 0.5

    def __init__(
        self,
        registry: "BackendRegistry",
        check_interval_seconds: float = 60.0,
        logger: logging.Logger | None = None,
    ):
        self._registry = registry
        self._check_interval = check_interval_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._health_history: dict[str, deque] = {}
        self._max_history_size = 100

    def start(self) -> None:
        """Start periodic health monitoring."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._logger.info("Backend health monitoring started")

    def stop(self) -> None:
        """Stop periodic health monitoring."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._logger.info("Backend health monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running and not self._stop_event.is_set():
            try:
                self.check_all_backends()
            except Exception as e:
                self._logger.error(f"Health check error: {e}")

            self._stop_event.wait(timeout=self._check_interval)

    def check_all_backends(self) -> dict[str, HealthCheckResult]:
        """Check health of all registered backends."""
        results = {}
        for status in self._registry.list_statuses():
            result = self.check_backend(status.name)
            results[status.name] = result
        return results

    def check_backend(self, name: str) -> HealthCheckResult:
        """Check health of a specific backend.

        Args:
            name: Backend name to check.

        Returns:
            HealthCheckResult with check details.
        """
        start_time = time.perf_counter()
        timestamp = time.time()

        try:
            adapter = self._registry.get(name)
            is_available = adapter.is_available()
            response_time_ms = (time.perf_counter() - start_time) * 1000

            result = HealthCheckResult(
                backend_name=name,
                timestamp=timestamp,
                healthy=is_available,
                response_time_ms=response_time_ms,
                details={
                    "version": adapter.get_version(),
                    "capabilities_check": adapter.get_capabilities() is not None,
                },
            )

            # Update registry health
            if is_available:
                self._registry.mark_backend_success(name)
            else:
                self._registry.mark_backend_failure(name, severity=0.2)

        except KeyError:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                backend_name=name,
                timestamp=timestamp,
                healthy=False,
                response_time_ms=response_time_ms,
                error="Backend not registered",
            )
            self._registry.mark_backend_failure(name, severity=0.3)

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                backend_name=name,
                timestamp=timestamp,
                healthy=False,
                response_time_ms=response_time_ms,
                error=str(e),
            )
            self._registry.mark_backend_failure(name, severity=0.3)

        # Store in history
        self._add_to_history(name, result)

        return result

    def _add_to_history(self, name: str, result: HealthCheckResult) -> None:
        """Add health check result to history."""
        if name not in self._health_history:
            self._health_history[name] = deque(maxlen=self._max_history_size)
        self._health_history[name].append(result)

    def get_health_history(self, name: str) -> list[HealthCheckResult]:
        """Get health check history for a backend."""
        return list(self._health_history.get(name, []))

    def calculate_health_score(self, name: str) -> float:
        """Calculate health score from recent history.

        Args:
            name: Backend name.

        Returns:
            Health score between 0.0 and 1.0.
        """
        history = self._health_history.get(name, [])
        if not history:
            return 1.0

        # Weight recent checks more heavily
        weights = [1.0 + (i / len(history)) for i in range(len(history))]
        total_weight = sum(weights)

        weighted_score = sum(
            (1.0 if r.healthy else 0.0) * w for r, w in zip(history, weights)
        )

        return weighted_score / total_weight

    def get_health_status(self, score: float) -> BackendHealthStatus:
        """Convert health score to status level."""
        if score >= self.HEALTHY_THRESHOLD:
            return BackendHealthStatus.HEALTHY
        elif score >= self.DEGRADED_THRESHOLD:
            return BackendHealthStatus.DEGRADED
        else:
            return BackendHealthStatus.UNHEALTHY

    @property
    def is_running(self) -> bool:
        return self._running

    # =========================================================================
    # Health Monitoring Edge Cases (2% completion)
    # =========================================================================

    def handle_timeout_edge_case(
        self,
        name: str,
        timeout_seconds: float = 30.0,
    ) -> HealthCheckResult:
        """Handle health check with timeout to prevent hanging.

        Edge case: Backend check may hang indefinitely if backend is unresponsive.

        Args:
            name: Backend name to check.
            timeout_seconds: Maximum time to wait for health check.

        Returns:
            HealthCheckResult with timeout handling.
        """
        import concurrent.futures

        start_time = time.perf_counter()
        timestamp = time.time()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.check_backend, name)
                try:
                    result = future.result(timeout=timeout_seconds)
                    return result
                except concurrent.futures.TimeoutError:
                    response_time_ms = (time.perf_counter() - start_time) * 1000
                    self._logger.warning(
                        f"Health check timed out for backend {name} "
                        f"after {timeout_seconds}s"
                    )
                    self._registry.mark_backend_failure(name, severity=0.25)
                    return HealthCheckResult(
                        backend_name=name,
                        timestamp=timestamp,
                        healthy=False,
                        response_time_ms=response_time_ms,
                        error=f"Timeout after {timeout_seconds}s",
                        details={"edge_case": "timeout"},
                    )
        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                backend_name=name,
                timestamp=timestamp,
                healthy=False,
                response_time_ms=response_time_ms,
                error=str(e),
                details={"edge_case": "exception_during_timeout_check"},
            )

    def handle_intermittent_failure_edge_case(
        self,
        name: str,
        retry_count: int = 3,
        retry_delay_ms: float = 100.0,
    ) -> HealthCheckResult:
        """Handle intermittent failures with retries.

        Edge case: Backend may fail sporadically due to network issues,
        resource contention, or transient errors.

        Args:
            name: Backend name to check.
            retry_count: Number of retries before declaring unhealthy.
            retry_delay_ms: Delay between retries in milliseconds.

        Returns:
            HealthCheckResult with retry handling.
        """
        last_result: HealthCheckResult | None = None
        successful_checks = 0
        total_response_time_ms = 0.0

        for attempt in range(retry_count):
            result = self.check_backend(name)
            total_response_time_ms += result.response_time_ms

            if result.healthy:
                successful_checks += 1
                if successful_checks >= 2:  # Need at least 2 successful checks
                    return HealthCheckResult(
                        backend_name=name,
                        timestamp=result.timestamp,
                        healthy=True,
                        response_time_ms=total_response_time_ms / (attempt + 1),
                        details={
                            "edge_case": "intermittent_recovery",
                            "attempts": attempt + 1,
                            "successful_checks": successful_checks,
                        },
                    )
            last_result = result

            if attempt < retry_count - 1:
                time.sleep(retry_delay_ms / 1000.0)

        # All retries failed or not enough successful checks
        avg_response_time = total_response_time_ms / retry_count
        return HealthCheckResult(
            backend_name=name,
            timestamp=time.time(),
            healthy=False,
            response_time_ms=avg_response_time,
            error=last_result.error if last_result else "All retries failed",
            details={
                "edge_case": "intermittent_failure",
                "attempts": retry_count,
                "successful_checks": successful_checks,
            },
        )

    def handle_partial_degradation_edge_case(
        self,
        name: str,
    ) -> HealthCheckResult:
        """Handle partial backend degradation.

        Edge case: Backend may be available but with reduced capabilities
        (e.g., GPU unavailable, reduced max qubits, missing features).

        Args:
            name: Backend name to check.

        Returns:
            HealthCheckResult with degradation details.
        """
        start_time = time.perf_counter()
        timestamp = time.time()

        try:
            adapter = self._registry.get(name)
            is_available = adapter.is_available()

            if not is_available:
                response_time_ms = (time.perf_counter() - start_time) * 1000
                return HealthCheckResult(
                    backend_name=name,
                    timestamp=timestamp,
                    healthy=False,
                    response_time_ms=response_time_ms,
                    error="Backend unavailable",
                )

            # Check for partial degradation
            degradation_issues: list[str] = []
            capabilities = adapter.get_capabilities()

            # Check GPU availability if expected
            if capabilities and capabilities.supports_gpu:
                try:
                    # Try to detect if GPU is actually accessible
                    gpu_info = getattr(adapter, 'get_gpu_info', lambda: None)()
                    if gpu_info is None:
                        degradation_issues.append("GPU not accessible")
                except Exception:
                    degradation_issues.append("GPU check failed")

            # Check memory constraints
            try:
                if hasattr(adapter, 'get_memory_info'):
                    mem_info = adapter.get_memory_info()
                    if mem_info and mem_info.get('utilization', 0) > 0.9:
                        degradation_issues.append("High memory usage (>90%)")
            except Exception:
                pass

            # Check version compatibility
            try:
                version = adapter.get_version()
                if version and ('unknown' in version.lower() or 'error' in version.lower()):
                    degradation_issues.append("Version detection issue")
            except Exception:
                degradation_issues.append("Version check failed")

            response_time_ms = (time.perf_counter() - start_time) * 1000

            if degradation_issues:
                # Partial degradation - mark as degraded
                self._registry.mark_backend_failure(name, severity=0.1)
                return HealthCheckResult(
                    backend_name=name,
                    timestamp=timestamp,
                    healthy=True,  # Still healthy but degraded
                    response_time_ms=response_time_ms,
                    details={
                        "edge_case": "partial_degradation",
                        "degradation_issues": degradation_issues,
                        "status": "degraded",
                    },
                )

            self._registry.mark_backend_success(name)
            return HealthCheckResult(
                backend_name=name,
                timestamp=timestamp,
                healthy=True,
                response_time_ms=response_time_ms,
                details={"edge_case": "partial_degradation", "status": "fully_healthy"},
            )

        except KeyError:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                backend_name=name,
                timestamp=timestamp,
                healthy=False,
                response_time_ms=response_time_ms,
                error="Backend not registered",
            )
        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                backend_name=name,
                timestamp=timestamp,
                healthy=False,
                response_time_ms=response_time_ms,
                error=str(e),
            )

    def handle_flapping_edge_case(
        self,
        name: str,
        stability_window: int = 10,
        flap_threshold: int = 3,
    ) -> tuple[bool, dict[str, Any]]:
        """Detect and handle flapping backends.

        Edge case: Backend rapidly alternating between healthy and unhealthy
        states (flapping), which can indicate instability.

        Args:
            name: Backend name to check.
            stability_window: Number of recent checks to analyze.
            flap_threshold: Number of state changes to consider flapping.

        Returns:
            Tuple of (is_flapping, flapping_details).
        """
        history = self.get_health_history(name)
        if len(history) < stability_window:
            return False, {"edge_case": "flapping", "status": "insufficient_data"}

        recent = history[-stability_window:]
        state_changes = 0

        for i in range(1, len(recent)):
            if recent[i].healthy != recent[i - 1].healthy:
                state_changes += 1

        is_flapping = state_changes >= flap_threshold

        if is_flapping:
            # Apply damping - reduce health score for unstable backend
            self._registry.mark_backend_failure(name, severity=0.15)
            self._logger.warning(
                f"Backend {name} is flapping: {state_changes} state changes "
                f"in last {stability_window} checks"
            )

        return is_flapping, {
            "edge_case": "flapping",
            "is_flapping": is_flapping,
            "state_changes": state_changes,
            "threshold": flap_threshold,
            "window_size": stability_window,
            "recent_healthy_count": sum(1 for r in recent if r.healthy),
            "recent_unhealthy_count": sum(1 for r in recent if not r.healthy),
        }

    def handle_resource_exhaustion_edge_case(
        self,
        name: str,
    ) -> HealthCheckResult:
        """Handle resource exhaustion scenarios.

        Edge case: Backend may become unhealthy due to resource exhaustion
        (memory, file handles, connections, threads).

        Args:
            name: Backend name to check.

        Returns:
            HealthCheckResult with resource exhaustion details.
        """
        import psutil

        start_time = time.perf_counter()
        timestamp = time.time()

        resource_issues: list[str] = []

        try:
            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                resource_issues.append(f"Critical system memory usage: {memory.percent}%")
            elif memory.percent > 85:
                resource_issues.append(f"High system memory usage: {memory.percent}%")

            # Check available file descriptors (Unix-like systems)
            try:
                process = psutil.Process()
                num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
                if num_fds > 1000:
                    resource_issues.append(f"High file descriptor count: {num_fds}")
            except Exception:
                pass

            # Check thread count
            try:
                num_threads = len(psutil.Process().threads())
                if num_threads > 100:
                    resource_issues.append(f"High thread count: {num_threads}")
            except Exception:
                pass

            # Now do the actual health check
            result = self.check_backend(name)
            response_time_ms = (time.perf_counter() - start_time) * 1000

            if resource_issues:
                result.details["edge_case"] = "resource_exhaustion"
                result.details["resource_issues"] = resource_issues
                if not result.healthy:
                    result.details["likely_cause"] = "resource_exhaustion"

            return result

        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                backend_name=name,
                timestamp=timestamp,
                healthy=False,
                response_time_ms=response_time_ms,
                error=str(e),
                details={"edge_case": "resource_exhaustion", "check_failed": True},
            )

    def handle_cascading_failure_edge_case(
        self,
        affected_backends: list[str] | None = None,
    ) -> dict[str, HealthCheckResult]:
        """Handle cascading failures across multiple backends.

        Edge case: One backend failure may trigger failures in dependent
        backends (e.g., GPU driver crash affecting cuQuantum and qsim-gpu).

        Args:
            affected_backends: List of backends to check, or all if None.

        Returns:
            Dictionary mapping backend names to their health check results.
        """
        backends = affected_backends or [s.name for s in self._registry.list_statuses()]
        results: dict[str, HealthCheckResult] = {}
        failure_count = 0

        for name in backends:
            result = self.check_backend(name)
            results[name] = result
            if not result.healthy:
                failure_count += 1

        # If multiple backends failed simultaneously, likely a cascading failure
        if failure_count >= 2:
            self._logger.warning(
                f"Potential cascading failure detected: {failure_count}/{len(backends)} "
                f"backends unhealthy"
            )

            # Check for common dependencies
            failed_backends = [name for name, r in results.items() if not r.healthy]

            # Categorize by potential root cause
            gpu_related = [b for b in failed_backends if b in ["cuquantum", "qsim"]]
            dependency_related = [b for b in failed_backends if b in ["cirq", "qiskit", "qsim"]]

            for name in failed_backends:
                results[name].details["edge_case"] = "cascading_failure"
                results[name].details["concurrent_failures"] = failure_count

                if name in gpu_related and len(gpu_related) > 1:
                    results[name].details["potential_cause"] = "gpu_subsystem_failure"
                if name in dependency_related and len(dependency_related) > 1:
                    results[name].details["potential_cause"] = "shared_dependency_failure"

        return results

    def comprehensive_health_check(
        self,
        name: str,
        include_edge_cases: bool = True,
    ) -> dict[str, Any]:
        """Perform comprehensive health check including all edge cases.

        Args:
            name: Backend name to check.
            include_edge_cases: Whether to run edge case handlers.

        Returns:
            Comprehensive health check report.
        """
        report: dict[str, Any] = {
            "backend_name": name,
            "timestamp": time.time(),
            "basic_check": None,
            "edge_cases": {},
            "overall_status": "unknown",
            "recommendations": [],
        }

        # Basic health check
        basic_result = self.check_backend(name)
        report["basic_check"] = {
            "healthy": basic_result.healthy,
            "response_time_ms": basic_result.response_time_ms,
            "error": basic_result.error,
        }

        if include_edge_cases:
            # Check for flapping
            is_flapping, flap_details = self.handle_flapping_edge_case(name)
            report["edge_cases"]["flapping"] = flap_details

            # Check for partial degradation
            degradation_result = self.handle_partial_degradation_edge_case(name)
            report["edge_cases"]["degradation"] = degradation_result.details

            # Check for resource exhaustion
            resource_result = self.handle_resource_exhaustion_edge_case(name)
            report["edge_cases"]["resources"] = resource_result.details

            # Generate recommendations
            if is_flapping:
                report["recommendations"].append(
                    "Backend is unstable - consider restarting or investigating root cause"
                )
            if degradation_result.details.get("degradation_issues"):
                report["recommendations"].append(
                    f"Partial degradation detected: {degradation_result.details['degradation_issues']}"
                )
            if resource_result.details.get("resource_issues"):
                report["recommendations"].append(
                    f"Resource issues: {resource_result.details['resource_issues']}"
                )

        # Determine overall status
        if not basic_result.healthy:
            report["overall_status"] = "unhealthy"
        elif is_flapping if include_edge_cases else False:
            report["overall_status"] = "unstable"
        elif report["edge_cases"].get("degradation", {}).get("status") == "degraded":
            report["overall_status"] = "degraded"
        else:
            report["overall_status"] = "healthy"

        return report


# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================


class PerformanceHistoryTracker:
    """Track performance history for all backends.

    Features:
    - Record execution metrics
    - Calculate statistics
    - Persist history to file
    - Generate performance reports
    """

    def __init__(
        self,
        max_records_per_backend: int = 1000,
        persist_file: str | None = None,
        logger: logging.Logger | None = None,
    ):
        self._max_records = max_records_per_backend
        self._persist_file = persist_file
        self._logger = logger or logging.getLogger(__name__)
        self._history: dict[str, deque[PerformanceRecord]] = {}
        self._stats: dict[str, PerformanceStats] = {}
        self._lock = threading.Lock()

        # Load persisted history if available
        if persist_file and os.path.exists(persist_file):
            self._load_history()

    def record_execution(
        self,
        backend_name: str,
        execution_time_ms: float,
        num_qubits: int,
        gate_count: int,
        shots: int,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Record an execution for a backend.

        Args:
            backend_name: Name of the backend.
            execution_time_ms: Execution time in milliseconds.
            num_qubits: Number of qubits in circuit.
            gate_count: Number of gates in circuit.
            shots: Number of shots executed.
            success: Whether execution was successful.
            error_message: Error message if failed.
        """
        record = PerformanceRecord(
            timestamp=time.time(),
            execution_time_ms=execution_time_ms,
            num_qubits=num_qubits,
            gate_count=gate_count,
            shots=shots,
            success=success,
            error_message=error_message,
        )

        with self._lock:
            if backend_name not in self._history:
                self._history[backend_name] = deque(maxlen=self._max_records)
            self._history[backend_name].append(record)

            # Update statistics
            self._update_stats(backend_name)

    def _update_stats(self, backend_name: str) -> None:
        """Update statistics for a backend."""
        history = self._history.get(backend_name, [])
        if not history:
            return

        successful = [r for r in history if r.success]
        failed = [r for r in history if not r.success]

        execution_times = [r.execution_time_ms for r in successful]
        qubits = [r.num_qubits for r in successful]

        stats = PerformanceStats(
            backend_name=backend_name,
            total_executions=len(history),
            successful_executions=len(successful),
            failed_executions=len(failed),
            success_rate=len(successful) / len(history) if history else 1.0,
            last_updated=time.time(),
        )

        if execution_times:
            stats.total_execution_time_ms = sum(execution_times)
            stats.avg_execution_time_ms = statistics.mean(execution_times)
            stats.min_execution_time_ms = min(execution_times)
            stats.max_execution_time_ms = max(execution_times)
            if len(execution_times) > 1:
                stats.std_dev_execution_time_ms = statistics.stdev(execution_times)

        if qubits:
            stats.avg_qubits = statistics.mean(qubits)
            stats.max_qubits_executed = max(qubits)

        self._stats[backend_name] = stats

    def get_stats(self, backend_name: str) -> PerformanceStats | None:
        """Get performance statistics for a backend."""
        with self._lock:
            return self._stats.get(backend_name)

    def get_all_stats(self) -> dict[str, PerformanceStats]:
        """Get statistics for all backends."""
        with self._lock:
            return dict(self._stats)

    def get_history(self, backend_name: str) -> list[PerformanceRecord]:
        """Get performance history for a backend."""
        with self._lock:
            return list(self._history.get(backend_name, []))

    def get_performance_tier(self, backend_name: str) -> PerformanceTier:
        """Determine performance tier for a backend.

        Args:
            backend_name: Backend name.

        Returns:
            PerformanceTier classification.
        """
        stats = self.get_stats(backend_name)
        if not stats or stats.total_executions == 0:
            return PerformanceTier.ACCEPTABLE

        # Score based on success rate and speed
        score = 0.0

        # Success rate component (0-50 points)
        score += stats.success_rate * 50

        # Speed component (0-50 points based on avg execution time)
        # Lower is better
        if stats.avg_execution_time_ms < 100:
            score += 50
        elif stats.avg_execution_time_ms < 500:
            score += 40
        elif stats.avg_execution_time_ms < 1000:
            score += 30
        elif stats.avg_execution_time_ms < 5000:
            score += 20
        else:
            score += 10

        if score >= 90:
            return PerformanceTier.EXCELLENT
        elif score >= 70:
            return PerformanceTier.GOOD
        elif score >= 50:
            return PerformanceTier.ACCEPTABLE
        else:
            return PerformanceTier.POOR

    def save_history(self) -> None:
        """Persist history to file."""
        if not self._persist_file:
            return

        with self._lock:
            data = {
                "saved_at": time.time(),
                "backends": {},
            }
            for name, records in self._history.items():
                data["backends"][name] = [
                    {
                        "timestamp": r.timestamp,
                        "execution_time_ms": r.execution_time_ms,
                        "num_qubits": r.num_qubits,
                        "gate_count": r.gate_count,
                        "shots": r.shots,
                        "success": r.success,
                        "error_message": r.error_message,
                    }
                    for r in records
                ]

        try:
            with open(self._persist_file, "w") as f:
                json.dump(data, f, indent=2)
            self._logger.debug(f"Performance history saved to {self._persist_file}")
        except Exception as e:
            self._logger.error(f"Failed to save history: {e}")

    def _load_history(self) -> None:
        """Load history from file."""
        if not self._persist_file or not os.path.exists(self._persist_file):
            return

        try:
            with open(self._persist_file, "r") as f:
                data = json.load(f)

            for name, records in data.get("backends", {}).items():
                self._history[name] = deque(maxlen=self._max_records)
                for r in records:
                    self._history[name].append(
                        PerformanceRecord(
                            timestamp=r["timestamp"],
                            execution_time_ms=r["execution_time_ms"],
                            num_qubits=r["num_qubits"],
                            gate_count=r["gate_count"],
                            shots=r["shots"],
                            success=r["success"],
                            error_message=r.get("error_message"),
                        )
                    )
                self._update_stats(name)

            self._logger.debug(f"Loaded performance history from {self._persist_file}")
        except Exception as e:
            self._logger.error(f"Failed to load history: {e}")


# =============================================================================
# COMPARISON MATRIX GENERATOR
# =============================================================================


class BackendComparisonMatrix:
    """Generate comparison matrix for all backends.

    Features:
    - Feature comparison across backends
    - Performance comparison
    - Capability comparison
    - Export to various formats
    """

    def __init__(
        self,
        registry: "BackendRegistry",
        performance_tracker: PerformanceHistoryTracker | None = None,
        logger: logging.Logger | None = None,
    ):
        self._registry = registry
        self._performance_tracker = performance_tracker
        self._logger = logger or logging.getLogger(__name__)

    def generate(self) -> list[BackendComparisonEntry]:
        """Generate comparison matrix for all backends.

        Returns:
            List of BackendComparisonEntry for each backend.
        """
        entries = []

        for status in self._registry.list_statuses():
            entry = self._create_entry(status)
            entries.append(entry)

        # Sort by health score descending
        entries.sort(key=lambda e: e.health_score, reverse=True)

        return entries

    def _create_entry(self, status: BackendStatus) -> BackendComparisonEntry:
        """Create comparison entry for a backend status."""
        caps = status.capabilities

        # Get performance data
        avg_time = None
        success_rate = 1.0
        perf_tier = PerformanceTier.ACCEPTABLE

        if self._performance_tracker:
            stats = self._performance_tracker.get_stats(status.name)
            if stats:
                avg_time = stats.avg_execution_time_ms
                success_rate = stats.success_rate
            perf_tier = self._performance_tracker.get_performance_tier(status.name)

        return BackendComparisonEntry(
            backend_name=status.name,
            available=status.available,
            version=status.version,
            health_score=status.health_score,
            health_status=status.health_status,
            supports_gpu=caps.supports_gpu if caps else False,
            supports_noise=caps.supports_noise if caps else False,
            supports_mpi=caps.supports_mpi if caps else False,
            max_qubits=caps.max_qubits if caps else 0,
            simulator_types=[st.value for st in caps.simulator_types] if caps else [],
            native_gates_count=len(caps.native_gates) if caps else 0,
            performance_tier=perf_tier,
            avg_execution_time_ms=avg_time,
            success_rate=success_rate,
            additional_features=caps.additional_features if caps else {},
        )

    def to_dict(self) -> list[dict[str, Any]]:
        """Export comparison matrix to dictionary format."""
        entries = self.generate()
        return [
            {
                "backend": e.backend_name,
                "available": e.available,
                "version": e.version,
                "health_score": e.health_score,
                "health_status": e.health_status.value,
                "supports_gpu": e.supports_gpu,
                "supports_noise": e.supports_noise,
                "supports_mpi": e.supports_mpi,
                "max_qubits": e.max_qubits,
                "simulator_types": e.simulator_types,
                "native_gates_count": e.native_gates_count,
                "performance_tier": e.performance_tier.value,
                "avg_execution_time_ms": e.avg_execution_time_ms,
                "success_rate": e.success_rate,
                "additional_features": e.additional_features,
            }
            for e in entries
        ]

    def to_markdown(self) -> str:
        """Export comparison matrix to Markdown table."""
        entries = self.generate()

        lines = [
            "| Backend | Available | Version | Health | GPU | Noise | MPI | Max Qubits | Perf Tier |",
            "|---------|-----------|---------|--------|-----|-------|-----|------------|-----------|",
        ]

        for e in entries:
            health = f"{e.health_score:.0%}"
            gpu = "âœ“" if e.supports_gpu else "âœ—"
            noise = "âœ“" if e.supports_noise else "âœ—"
            mpi = "âœ“" if e.supports_mpi else "âœ—"
            avail = "âœ“" if e.available else "âœ—"

            lines.append(
                f"| {e.backend_name} | {avail} | {e.version or 'N/A'} | {health} | "
                f"{gpu} | {noise} | {mpi} | {e.max_qubits} | {e.performance_tier.value} |"
            )

        return "\n".join(lines)

    def get_feature_matrix(self) -> dict[str, dict[str, bool]]:
        """Get feature availability matrix.

        Returns:
            Dictionary mapping backend names to feature availability.
        """
        entries = self.generate()
        features = ["supports_gpu", "supports_noise", "supports_mpi"]

        matrix = {}
        for e in entries:
            matrix[e.backend_name] = {
                "available": e.available,
                "supports_gpu": e.supports_gpu,
                "supports_noise": e.supports_noise,
                "supports_mpi": e.supports_mpi,
                "state_vector": "state_vector" in e.simulator_types,
                "density_matrix": "density_matrix" in e.simulator_types,
            }

        return matrix

    def recommend_backend(
        self,
        num_qubits: int,
        require_gpu: bool = False,
        require_noise: bool = False,
        simulation_type: str = "state_vector",
    ) -> str | None:
        """Recommend best backend for given requirements.

        Args:
            num_qubits: Number of qubits needed.
            require_gpu: Whether GPU support is required.
            require_noise: Whether noise simulation is required.
            simulation_type: Type of simulation needed.

        Returns:
            Name of recommended backend, or None.
        """
        entries = self.generate()

        # Filter by requirements
        candidates = []
        for e in entries:
            if not e.available:
                continue
            if e.max_qubits < num_qubits:
                continue
            if require_gpu and not e.supports_gpu:
                continue
            if require_noise and not e.supports_noise:
                continue
            if simulation_type not in e.simulator_types:
                continue
            candidates.append(e)

        if not candidates:
            return None

        # Score candidates
        def score(e: BackendComparisonEntry) -> float:
            s = e.health_score * 100
            s += e.success_rate * 50
            if e.performance_tier == PerformanceTier.EXCELLENT:
                s += 40
            elif e.performance_tier == PerformanceTier.GOOD:
                s += 30
            elif e.performance_tier == PerformanceTier.ACCEPTABLE:
                s += 20
            # Prefer backends with more headroom
            s += min(e.max_qubits - num_qubits, 10)
            return s

        candidates.sort(key=score, reverse=True)
        return candidates[0].backend_name



# =============================================================================
# REGISTRATION VERIFIER
# =============================================================================


class BackendRegistrationVerifier:
    """Verify all backends are properly registered.

    Ensures all 6 backends (LRET, Cirq, Qiskit, QuEST, cuQuantum, qsim)
    are registered and available.
    """

    EXPECTED_BACKENDS = frozenset([
        "lret",
        "cirq",
        "qiskit",
        "quest",
        "cuquantum",
        "qsim",
    ])

    def __init__(
        self,
        registry: "BackendRegistry",
        logger: logging.Logger | None = None,
    ):
        self._registry = registry
        self._logger = logger or logging.getLogger(__name__)

    def verify_all(self) -> RegistrationVerificationResult:
        """Verify all expected backends are registered.

        Returns:
            RegistrationVerificationResult with verification details.
        """
        start_time = time.perf_counter()

        statuses = self._registry.list_statuses()
        registered_names = {s.name for s in statuses}
        available_names = {s.name for s in statuses if s.available}

        missing = list(self.EXPECTED_BACKENDS - registered_names)
        unavailable = list(registered_names - available_names)

        details = {}
        for status in statuses:
            details[status.name] = {
                "registered": True,
                "available": status.available,
                "version": status.version,
                "health_score": status.health_score,
                "reason": status.reason,
            }

        for name in missing:
            details[name] = {
                "registered": False,
                "available": False,
                "reason": "Not registered in registry",
            }

        verification_time = (time.perf_counter() - start_time) * 1000

        result = RegistrationVerificationResult(
            total_backends=len(self.EXPECTED_BACKENDS),
            registered_backends=len(registered_names),
            available_backends=len(available_names),
            missing_backends=missing,
            unavailable_backends=unavailable,
            verification_time=verification_time,
            all_registered=len(missing) == 0,
            details=details,
        )

        self._logger.info(
            f"Backend verification: {result.registered_backends}/{result.total_backends} registered, "
            f"{result.available_backends} available, {len(missing)} missing"
        )

        return result

    def verify_backend(self, name: str) -> dict[str, Any]:
        """Verify a specific backend.

        Args:
            name: Backend name to verify.

        Returns:
            Dictionary with verification details.
        """
        try:
            status = self._registry.get_status(name)
            return {
                "name": name,
                "registered": True,
                "available": status.available,
                "version": status.version,
                "health_score": status.health_score,
                "capabilities": status.capabilities is not None,
                "reason": status.reason,
            }
        except KeyError:
            return {
                "name": name,
                "registered": False,
                "available": False,
                "reason": "Not registered",
            }

    def get_missing_backends(self) -> list[str]:
        """Get list of missing backends."""
        registered = {s.name for s in self._registry.list_statuses()}
        return list(self.EXPECTED_BACKENDS - registered)

    def get_unavailable_backends(self) -> list[str]:
        """Get list of registered but unavailable backends."""
        return [
            s.name for s in self._registry.list_statuses()
            if not s.available and s.name in self.EXPECTED_BACKENDS
        ]


# =============================================================================
# BACKEND REGISTRY - Main class
# =============================================================================


class BackendRegistry:
    """Maintains discovery and lookup of backend adapters.

    Enhanced Features (100% Complete):
    - All 6 backends registration verification
    - Health monitoring & periodic checks
    - Backend comparison matrix generation
    - Performance history tracking

    Supports hot-reload for refreshing backend discovery without restart.
    Thread-safe for concurrent access during reload operations.
    """

    CACHE_TTL_SECONDS: float = 300.0  # 5 minutes

    # All expected backend adapter classes
    BACKEND_ADAPTERS: list[type[BaseBackendAdapter]] = [
        LRETBackendAdapter,
        CirqBackendAdapter,
        QiskitBackendAdapter,
        QuestAdapter,
        CuQuantumAdapter,
        QsimAdapter,
    ]

    def __init__(
        self,
        enable_health_monitoring: bool = True,
        enable_performance_tracking: bool = True,
        health_check_interval: float = 60.0,
        performance_history_file: str | None = None,
    ) -> None:
        self._statuses: dict[str, BackendStatus] = {}
        self._lock = threading.RLock()
        self._reload_callbacks: list[ReloadCallback] = []
        self._last_discovery: float = 0.0
        self._discovery_count: int = 0

        # Enhanced components
        self._health_monitor: BackendHealthMonitor | None = None
        self._performance_tracker: PerformanceHistoryTracker | None = None
        self._verifier: BackendRegistrationVerifier | None = None
        self._comparison_matrix: BackendComparisonMatrix | None = None

        # Initialize performance tracker
        if enable_performance_tracking:
            self._performance_tracker = PerformanceHistoryTracker(
                persist_file=performance_history_file,
            )

        # Initialize health monitor (but don't start yet)
        if enable_health_monitoring:
            self._health_monitor = BackendHealthMonitor(
                self,
                check_interval_seconds=health_check_interval,
            )

        # Initialize verifier and comparison matrix
        self._verifier = BackendRegistrationVerifier(self)
        self._comparison_matrix = BackendComparisonMatrix(
            self, self._performance_tracker
        )

    def register(self, adapter: BaseBackendAdapter) -> None:
        """Register a backend adapter."""
        name = adapter.get_name()
        capabilities = adapter.get_capabilities()
        version = self._safe_get_version(adapter)

        with self._lock:
            self._statuses[name] = BackendStatus(
                name=name,
                available=True,
                adapter=adapter,
                capabilities=capabilities,
                version=version,
                reason=None,
                last_checked=time.time(),
                health_score=1.0,
                health_status=BackendHealthStatus.HEALTHY,
            )

    def unregister(self, name: str) -> bool:
        """Unregister a backend by name. Returns True if removed."""
        with self._lock:
            if name in self._statuses:
                del self._statuses[name]
                return True
            return False

    def on_reload(self, callback: ReloadCallback) -> None:
        """Register callback to be notified after hot-reload."""
        self._reload_callbacks.append(callback)

    def remove_reload_callback(self, callback: ReloadCallback) -> bool:
        """Remove a reload callback. Returns True if removed."""
        try:
            self._reload_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def discover(self) -> None:
        """Discover known backends, cache capabilities, and mark health status."""
        with self._lock:
            self._discover_internal()

    def _discover_internal(self) -> None:
        """Internal discovery implementation (must hold lock)."""
        self._statuses = {}

        for adapter_cls in self.BACKEND_ADAPTERS:
            status = self._init_adapter(adapter_cls)
            self._statuses[status.name] = status

        self._last_discovery = time.time()
        self._discovery_count += 1

    def hot_reload(self, force: bool = False) -> dict[str, str]:
        """Hot-reload backend discovery without full restart."""
        with self._lock:
            elapsed = time.time() - self._last_discovery
            if not force and elapsed < self.CACHE_TTL_SECONDS:
                logger.debug(
                    f"Skipping hot-reload, cache fresh ({elapsed:.1f}s < {self.CACHE_TTL_SECONDS}s)"
                )
                return {}

            old_statuses = dict(self._statuses)
            old_available = {name for name, s in old_statuses.items() if s.available}

            # Refresh module imports
            adapter_modules = [
                "proxima.backends.lret",
                "proxima.backends.cirq_adapter",
                "proxima.backends.qiskit_adapter",
                "proxima.backends.quest_adapter",
                "proxima.backends.cuquantum_adapter",
                "proxima.backends.qsim_adapter",
            ]
            for mod_name in adapter_modules:
                try:
                    if mod_name in importlib.sys.modules:
                        importlib.reload(importlib.sys.modules[mod_name])
                except Exception as e:
                    logger.warning(f"Failed to reload module {mod_name}: {e}")

            self._discover_internal()

            # Calculate changes
            new_available = {name for name, s in self._statuses.items() if s.available}
            changes: dict[str, str] = {}

            all_names = old_available | new_available | set(old_statuses.keys()) | set(
                self._statuses.keys()
            )
            for name in all_names:
                old_status = old_statuses.get(name)
                new_status = self._statuses.get(name)

                if old_status is None and new_status is not None:
                    changes[name] = "added"
                elif old_status is not None and new_status is None:
                    changes[name] = "removed"
                elif old_status and new_status:
                    if old_status.available != new_status.available:
                        changes[name] = "changed"
                        if new_status.available and old_status.health_score < 1.0:
                            new_status.health_score = old_status.health_score
                    elif old_status.version != new_status.version:
                        changes[name] = "changed"
                    else:
                        changes[name] = "unchanged"
                        if old_status.health_score < 1.0:
                            new_status.health_score = old_status.health_score

            logger.info(
                f"Hot-reload complete: {len(changes)} backends processed, "
                f"{sum(1 for c in changes.values() if c != 'unchanged')} changed"
            )

        for callback in self._reload_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Reload callback error: {e}")

        return changes

    def refresh_backend(self, name: str) -> BackendStatus | None:
        """Refresh a single backend's status without full reload."""
        adapter_map = {
            "lret": LRETBackendAdapter,
            "cirq": CirqBackendAdapter,
            "qiskit": QiskitBackendAdapter,
            "quest": QuestAdapter,
            "cuquantum": CuQuantumAdapter,
            "qsim": QsimAdapter,
        }

        adapter_cls = adapter_map.get(name.lower())
        if not adapter_cls:
            return None

        with self._lock:
            old_status = self._statuses.get(name)
            new_status = self._init_adapter(adapter_cls)

            if old_status and old_status.health_score < 1.0:
                new_status.health_score = old_status.health_score

            self._statuses[new_status.name] = new_status
            return new_status

    def mark_backend_failure(self, name: str, severity: float = 0.1) -> None:
        """Record a backend failure to adjust health score."""
        with self._lock:
            status = self._statuses.get(name)
            if status:
                status.health_score = max(0.0, status.health_score - severity)
                status.consecutive_failures += 1
                status.last_failure_time = time.time()
                status.health_status = self._calculate_health_status(status.health_score)
                logger.debug(f"Backend {name} health reduced to {status.health_score:.2f}")

    def mark_backend_success(self, name: str, recovery: float = 0.05) -> None:
        """Record a backend success to recover health score."""
        with self._lock:
            status = self._statuses.get(name)
            if status:
                status.health_score = min(1.0, status.health_score + recovery)
                status.consecutive_failures = 0
                status.last_success_time = time.time()
                status.health_status = self._calculate_health_status(status.health_score)

    def _calculate_health_status(self, score: float) -> BackendHealthStatus:
        """Calculate health status from score."""
        if score >= 0.8:
            return BackendHealthStatus.HEALTHY
        elif score >= 0.5:
            return BackendHealthStatus.DEGRADED
        else:
            return BackendHealthStatus.UNHEALTHY


    def get_discovery_stats(self) -> dict:
        """Get statistics about backend discovery."""
        with self._lock:
            return {
                "discovery_count": self._discovery_count,
                "last_discovery": self._last_discovery,
                "cache_age_seconds": time.time() - self._last_discovery if self._last_discovery else None,
                "total_backends": len(self._statuses),
                "available_backends": len([s for s in self._statuses.values() if s.available]),
            }

    def _init_adapter(self, adapter_cls: type[BaseBackendAdapter]) -> BackendStatus:
        """Initialize an adapter and return its status."""
        name = getattr(adapter_cls, "__name__", "unknown").lower()

        try:
            adapter = adapter_cls()
            name = adapter.get_name()
        except Exception as exc:
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"initialization failed: {exc}",
                health_status=BackendHealthStatus.UNHEALTHY,
            )

        try:
            if not adapter.is_available():
                return BackendStatus(
                    name=name,
                    available=False,
                    adapter=None,
                    capabilities=None,
                    version=None,
                    reason=self._dependency_reason(adapter_cls),
                    health_status=BackendHealthStatus.UNKNOWN,
                )
        except Exception as exc:
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"availability check failed: {exc}",
                health_status=BackendHealthStatus.UNHEALTHY,
            )

        try:
            capabilities = adapter.get_capabilities()
        except Exception as exc:
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"capabilities check failed: {exc}",
                health_status=BackendHealthStatus.UNHEALTHY,
            )

        version = self._safe_get_version(adapter)
        return BackendStatus(
            name=name,
            available=True,
            adapter=adapter,
            capabilities=capabilities,
            version=version,
            reason=None,
            health_status=BackendHealthStatus.HEALTHY,
        )

    def _dependency_reason(self, adapter_cls: type[BaseBackendAdapter]) -> str:
        """Get reason for unavailable adapter."""
        dependency_map = {
            "lretbackendadapter": ["lret"],
            "cirqbackendadapter": ["cirq"],
            "qiskitbackendadapter": ["qiskit", "qiskit_aer"],
            "questadapter": ["pyQuEST"],
            "cuquantumadapter": ["qiskit", "qiskit_aer", "cuquantum"],
            "qsimadapter": ["cirq", "qsimcirq"],
        }
        missing = []
        for dep in dependency_map.get(adapter_cls.__name__.lower(), []):
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        if missing:
            return f"missing dependency: {', '.join(missing)}"
        return "adapter reported unavailable"

    def _safe_get_version(self, adapter: BaseBackendAdapter) -> str:
        """Safely get adapter version."""
        try:
            return adapter.get_version()
        except Exception as exc:
            return f"unknown (version check failed: {exc})"

    # =========================================================================
    # Core Access Methods
    # =========================================================================

    def get(self, name: str) -> BaseBackendAdapter:
        """Get a backend adapter by name."""
        with self._lock:
            status = self._statuses.get(name)
            if not status:
                raise KeyError(f"Backend '{name}' not registered")
            if not status.available or not status.adapter:
                raise KeyError(
                    f"Backend '{name}' is unavailable: {status.reason or 'unknown reason'}"
                )
            return status.adapter

    def is_available(self, name: str) -> bool:
        """Check if a backend is available."""
        with self._lock:
            status = self._statuses.get(name)
            return bool(status and status.available)

    def list_available(self) -> list[str]:
        """List all available backend names."""
        with self._lock:
            return [name for name, status in self._statuses.items() if status.available]

    def list_statuses(self) -> list[BackendStatus]:
        """List all backend statuses."""
        with self._lock:
            return list(self._statuses.values())

    def get_capabilities(self, name: str) -> Capabilities:
        """Get capabilities for a backend."""
        return self.get(name).get_capabilities()

    def get_status(self, name: str) -> BackendStatus:
        """Get status for a specific backend."""
        with self._lock:
            status = self._statuses.get(name)
            if not status:
                raise KeyError(f"Backend '{name}' not registered")
            return status

    def get_healthy_backends(self, min_health: float = 0.5) -> list[str]:
        """Get backends with health score above threshold."""
        with self._lock:
            return [
                name for name, status in self._statuses.items()
                if status.available and status.health_score >= min_health
            ]

    # =========================================================================
    # Enhanced Features - Verification
    # =========================================================================

    def verify_all_backends(self) -> RegistrationVerificationResult:
        """Verify all 6 expected backends are registered.

        Returns:
            RegistrationVerificationResult with details.
        """
        if self._verifier:
            return self._verifier.verify_all()
        return RegistrationVerificationResult(
            total_backends=0,
            registered_backends=0,
            available_backends=0,
            missing_backends=[],
            unavailable_backends=[],
            verification_time=0,
            all_registered=False,
        )

    def verify_backend(self, name: str) -> dict[str, Any]:
        """Verify a specific backend."""
        if self._verifier:
            return self._verifier.verify_backend(name)
        return {"name": name, "error": "Verifier not initialized"}

    def get_missing_backends(self) -> list[str]:
        """Get list of missing backends."""
        if self._verifier:
            return self._verifier.get_missing_backends()
        return []

    # =========================================================================
    # Enhanced Features - Health Monitoring
    # =========================================================================

    def start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._health_monitor:
            self._health_monitor.start()

    def stop_health_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        if self._health_monitor:
            self._health_monitor.stop()

    def check_backend_health(self, name: str) -> HealthCheckResult:
        """Check health of a specific backend."""
        if self._health_monitor:
            return self._health_monitor.check_backend(name)
        return HealthCheckResult(
            backend_name=name,
            timestamp=time.time(),
            healthy=self.is_available(name),
            response_time_ms=0,
        )

    def check_all_backends_health(self) -> dict[str, HealthCheckResult]:
        """Check health of all backends."""
        if self._health_monitor:
            return self._health_monitor.check_all_backends()
        return {}

    def get_health_history(self, name: str) -> list[HealthCheckResult]:
        """Get health check history for a backend."""
        if self._health_monitor:
            return self._health_monitor.get_health_history(name)
        return []

    # =========================================================================
    # Enhanced Features - Comparison Matrix
    # =========================================================================

    def generate_comparison_matrix(self) -> list[BackendComparisonEntry]:
        """Generate backend comparison matrix.

        Returns:
            List of BackendComparisonEntry for all backends.
        """
        if self._comparison_matrix:
            return self._comparison_matrix.generate()
        return []

    def get_comparison_dict(self) -> list[dict[str, Any]]:
        """Get comparison matrix as dictionary."""
        if self._comparison_matrix:
            return self._comparison_matrix.to_dict()
        return []

    def get_comparison_markdown(self) -> str:
        """Get comparison matrix as Markdown table."""
        if self._comparison_matrix:
            return self._comparison_matrix.to_markdown()
        return ""

    def get_feature_matrix(self) -> dict[str, dict[str, bool]]:
        """Get feature availability matrix."""
        if self._comparison_matrix:
            return self._comparison_matrix.get_feature_matrix()
        return {}

    def recommend_backend(
        self,
        num_qubits: int,
        require_gpu: bool = False,
        require_noise: bool = False,
        simulation_type: str = "state_vector",
    ) -> str | None:
        """Recommend best backend for requirements."""
        if self._comparison_matrix:
            return self._comparison_matrix.recommend_backend(
                num_qubits, require_gpu, require_noise, simulation_type
            )
        return None

    # =========================================================================
    # Enhanced Features - Performance Tracking
    # =========================================================================

    def record_execution(
        self,
        backend_name: str,
        execution_time_ms: float,
        num_qubits: int,
        gate_count: int,
        shots: int,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Record an execution for performance tracking."""
        if self._performance_tracker:
            self._performance_tracker.record_execution(
                backend_name,
                execution_time_ms,
                num_qubits,
                gate_count,
                shots,
                success,
                error_message,
            )

    def get_performance_stats(self, backend_name: str) -> PerformanceStats | None:
        """Get performance statistics for a backend."""
        if self._performance_tracker:
            return self._performance_tracker.get_stats(backend_name)
        return None

    def get_all_performance_stats(self) -> dict[str, PerformanceStats]:
        """Get performance statistics for all backends."""
        if self._performance_tracker:
            return self._performance_tracker.get_all_stats()
        return {}

    def get_performance_history(self, backend_name: str) -> list[PerformanceRecord]:
        """Get performance history for a backend."""
        if self._performance_tracker:
            return self._performance_tracker.get_history(backend_name)
        return []

    def save_performance_history(self) -> None:
        """Save performance history to file."""
        if self._performance_tracker:
            self._performance_tracker.save_history()

    # =========================================================================
    # GPU and Selection Helpers
    # =========================================================================

    def get_gpu_backends(self) -> list[str]:
        """Return list of GPU-enabled backends."""
        gpu_backends = []
        for name, status in self._statuses.items():
            if status.available and status.capabilities:
                if status.capabilities.supports_gpu:
                    gpu_backends.append(name)
        return gpu_backends

    def get_best_backend_for_circuit(
        self,
        qubit_count: int,
        simulation_type: str = "state_vector",
        prefer_gpu: bool = True,
    ) -> str | None:
        """Get best available backend for given circuit requirements."""
        if simulation_type == "state_vector":
            if prefer_gpu:
                priority = ["cuquantum", "quest", "qsim", "qiskit", "cirq"]
            else:
                priority = ["qsim", "quest", "qiskit", "cirq"]
        elif simulation_type == "density_matrix":
            priority = ["quest", "cirq", "qiskit", "lret"]
        else:
            priority = ["qsim", "qiskit", "cirq", "quest"]

        for backend_name in priority:
            status = self._statuses.get(backend_name)
            if status and status.available and status.capabilities:
                if status.capabilities.max_qubits >= qubit_count:
                    return backend_name

        return None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "BackendRegistry",
    "BackendStatus",
    "BackendHealthStatus",
    "PerformanceTier",
    "PerformanceRecord",
    "PerformanceStats",
    "BackendComparisonEntry",
    "HealthCheckResult",
    "RegistrationVerificationResult",
    "BackendHealthMonitor",
    "PerformanceHistoryTracker",
    "BackendComparisonMatrix",
    "BackendRegistrationVerifier",
    "backend_registry",
]


# ==========================================
# Global Instance
# ==========================================

# Create global registry instance
backend_registry = BackendRegistry()

