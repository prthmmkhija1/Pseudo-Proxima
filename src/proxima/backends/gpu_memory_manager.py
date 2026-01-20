"""GPU Memory Management for cuQuantum backend.

Step 2.4: Optimize GPU Memory Management
=========================================

This module provides advanced GPU memory management capabilities for cuQuantum:
1. Pre-Execution Memory Checks - Query and validate GPU memory before execution
2. Memory Pooling - Reuse GPU memory allocations across executions
3. Batch Processing - Execute circuits sequentially with memory monitoring
4. Fallback Strategy - Intelligent fallback when memory is insufficient
5. Memory Leak Detection (100%) - Detect, analyze, and mitigate GPU memory leaks

Memory Estimation Formula:
--------------------------
Total GPU Memory Required = State Vector Size + Workspace + Overhead
= (2^n * bytes_per_amplitude) + workspace_size + overhead

Where:
- n = number of qubits
- bytes_per_amplitude = 8 (complex64/single) or 16 (complex128/double)
- workspace_size = cuStateVec scratch memory (default 1GB)
- overhead = CUDA runtime and driver buffers (~500MB)

Memory Leak Detection Features:
-------------------------------
- GPUMemoryLeakDetector: Detects memory leaks during execution
- MemoryLeakMitigator: Automatic cleanup and circuit breaker protection
- Trend analysis for gradual leaks
- Severity classification (LOW, MEDIUM, HIGH, CRITICAL)
- Integration with batch execution
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from proxima.backends.cuquantum_adapter import (
    CuQuantumConfig,
    CuQuantumPrecision,
    GPUDeviceInfo,
)

# =============================================================================
# Memory Constants and Thresholds
# =============================================================================

# Bytes per amplitude based on precision
BYTES_PER_AMPLITUDE = {
    CuQuantumPrecision.SINGLE: 8,  # complex64: 4 bytes real + 4 bytes imag
    CuQuantumPrecision.DOUBLE: 16,  # complex128: 8 bytes real + 8 bytes imag
}

# Default workspace sizes in MB
DEFAULT_WORKSPACE_MB = 1024  # 1 GB default cuStateVec workspace
MINIMUM_WORKSPACE_MB = 256  # 256 MB minimum workspace

# Overhead estimates in MB
CUDA_RUNTIME_OVERHEAD_MB = 200  # CUDA runtime overhead
DRIVER_BUFFER_OVERHEAD_MB = 300  # Driver buffer overhead
TOTAL_OVERHEAD_MB = CUDA_RUNTIME_OVERHEAD_MB + DRIVER_BUFFER_OVERHEAD_MB

# Memory safety margins
MEMORY_SAFETY_MARGIN = 0.10  # 10% safety margin
MEMORY_WARNING_THRESHOLD = 0.85  # Warn at 85% memory usage
MEMORY_CRITICAL_THRESHOLD = 0.95  # Critical at 95% memory usage

# Maximum qubits for different GPU memory sizes (with double precision)
MAX_QUBITS_BY_MEMORY = {
    4096: 28,  # 4 GB GPU
    8192: 29,  # 8 GB GPU
    12288: 30,  # 12 GB GPU
    16384: 30,  # 16 GB GPU
    24576: 31,  # 24 GB GPU
    32768: 31,  # 32 GB GPU
    49152: 32,  # 48 GB GPU (A6000, etc.)
    81920: 33,  # 80 GB GPU (A100, H100)
}


class MemoryAllocationStrategy(str, Enum):
    """Strategy for GPU memory allocation."""

    ON_DEMAND = "on_demand"  # Allocate memory as needed
    PRE_ALLOCATE = "pre_allocate"  # Pre-allocate memory pool
    LAZY_POOL = "lazy_pool"  # Lazy initialization with pooling


class MemoryCleanupPolicy(str, Enum):
    """Policy for GPU memory cleanup."""

    IMMEDIATE = "immediate"  # Clean up immediately after use
    DEFERRED = "deferred"  # Defer cleanup for reuse
    MANUAL = "manual"  # Manual cleanup by user


@dataclass
class MemoryEstimate:
    """Detailed memory requirement estimate for a circuit.

    Attributes:
        qubit_count: Number of qubits in the circuit
        state_vector_mb: Memory for state vector in MB
        workspace_mb: Memory for cuStateVec workspace in MB
        overhead_mb: Memory for CUDA runtime overhead in MB
        total_required_mb: Total memory required in MB
        total_with_safety_mb: Total memory with safety margin
        precision: Numerical precision used
        can_fit_on_device: Whether circuit fits on specified device
        recommended_device_id: Recommended GPU device ID
        fallback_required: Whether CPU fallback is needed
        fallback_reason: Reason for fallback if required
    """

    qubit_count: int
    state_vector_mb: float
    workspace_mb: float
    overhead_mb: float
    total_required_mb: float
    total_with_safety_mb: float
    precision: CuQuantumPrecision
    can_fit_on_device: bool = True
    recommended_device_id: int = 0
    fallback_required: bool = False
    fallback_reason: str = ""


@dataclass
class MemoryPoolStats:
    """Statistics for GPU memory pool.

    Attributes:
        pool_size_mb: Total pool size in MB
        allocated_mb: Currently allocated memory in MB
        available_mb: Available memory in pool in MB
        allocation_count: Number of active allocations
        reuse_count: Number of times memory was reused
        cache_hit_rate: Cache hit rate (0.0 to 1.0)
        last_cleanup_time: Timestamp of last cleanup
    """

    pool_size_mb: float = 0.0
    allocated_mb: float = 0.0
    available_mb: float = 0.0
    allocation_count: int = 0
    reuse_count: int = 0
    cache_hit_rate: float = 0.0
    last_cleanup_time: float = 0.0


@dataclass
class BatchExecutionResult:
    """Result of batch circuit execution.

    Attributes:
        total_circuits: Total number of circuits in batch
        successful_circuits: Number of successfully executed circuits
        failed_circuits: Number of failed circuits
        total_execution_time_ms: Total execution time in milliseconds
        peak_memory_mb: Peak GPU memory usage in MB
        results: List of individual execution results
        errors: List of errors for failed circuits
        memory_fragmentation: Memory fragmentation level (0.0 to 1.0)
    """

    total_circuits: int = 0
    successful_circuits: int = 0
    failed_circuits: int = 0
    total_execution_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    memory_fragmentation: float = 0.0


# =============================================================================
# Memory Leak Detection (5% completion)
# =============================================================================


class MemoryLeakSeverity(str, Enum):
    """Severity level of memory leaks."""

    NONE = "none"          # No leak detected
    LOW = "low"            # Minor leak, acceptable for short runs
    MEDIUM = "medium"      # Noticeable leak, should be addressed
    HIGH = "high"          # Significant leak, immediate attention needed
    CRITICAL = "critical"  # Severe leak, execution should be stopped


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state at a point in time.

    Attributes:
        timestamp: Unix timestamp of the snapshot
        total_mb: Total GPU memory in MB
        used_mb: Used GPU memory in MB
        free_mb: Free GPU memory in MB
        allocated_mb: Memory allocated by our application in MB
        device_id: GPU device ID
        context: Optional context string (e.g., "before_circuit_1")
    """

    timestamp: float
    total_mb: float
    used_mb: float
    free_mb: float
    allocated_mb: float = 0.0
    device_id: int = 0
    context: str = ""


@dataclass
class MemoryLeakReport:
    """Report of detected memory leaks.

    Attributes:
        leak_detected: Whether a leak was detected
        severity: Severity level of the leak
        leaked_mb: Estimated amount of leaked memory in MB
        leak_rate_mb_per_iteration: Rate of leak per iteration
        baseline_mb: Baseline memory usage
        current_mb: Current memory usage
        iterations_analyzed: Number of iterations analyzed
        snapshots: List of memory snapshots taken
        recommendations: List of recommended actions
        details: Additional details about the leak
    """

    leak_detected: bool = False
    severity: MemoryLeakSeverity = MemoryLeakSeverity.NONE
    leaked_mb: float = 0.0
    leak_rate_mb_per_iteration: float = 0.0
    baseline_mb: float = 0.0
    current_mb: float = 0.0
    iterations_analyzed: int = 0
    snapshots: list[MemorySnapshot] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class GPUMemoryLeakDetector:
    """Detects and reports GPU memory leaks during circuit execution.

    Features:
    - Baseline memory measurement
    - Continuous memory monitoring
    - Leak detection with configurable thresholds
    - Trend analysis for gradual leaks
    - Automatic cleanup recommendations

    Example:
        >>> detector = GPUMemoryLeakDetector()
        >>> detector.start_monitoring()
        >>> for circuit in circuits:
        ...     detector.record_snapshot("before_circuit")
        ...     execute(circuit)
        ...     detector.record_snapshot("after_circuit")
        >>> report = detector.analyze_for_leaks()
    """

    # Thresholds for leak detection
    LEAK_THRESHOLD_MB = 10.0  # Minimum MB to consider as a leak
    LEAK_RATE_THRESHOLD = 0.5  # MB per iteration to trigger warning
    CRITICAL_LEAK_RATE = 5.0  # MB per iteration for critical warning

    def __init__(
        self,
        device_id: int = 0,
        baseline_samples: int = 3,
        monitoring_interval_ms: float = 100.0,
    ):
        """Initialize the memory leak detector.

        Args:
            device_id: GPU device ID to monitor.
            baseline_samples: Number of samples for baseline measurement.
            monitoring_interval_ms: Interval between automatic samples in ms.
        """
        self._device_id = device_id
        self._baseline_samples = baseline_samples
        self._monitoring_interval_ms = monitoring_interval_ms
        self._logger = logging.getLogger(__name__)

        # State
        self._snapshots: list[MemorySnapshot] = []
        self._baseline_mb: float | None = None
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Statistics
        self._iteration_count = 0
        self._peak_usage_mb = 0.0
        self._total_allocations = 0
        self._total_deallocations = 0

    def get_current_memory_usage(self) -> MemorySnapshot:
        """Get current GPU memory usage.

        Returns:
            MemorySnapshot with current memory state.
        """
        timestamp = time.time()

        try:
            # Try using pynvml (NVIDIA Management Library)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlShutdown()

                return MemorySnapshot(
                    timestamp=timestamp,
                    total_mb=mem_info.total / (1024 * 1024),
                    used_mb=mem_info.used / (1024 * 1024),
                    free_mb=mem_info.free / (1024 * 1024),
                    device_id=self._device_id,
                )
            except ImportError:
                pass

            # Try using cupy
            try:
                import cupy as cp
                with cp.cuda.Device(self._device_id):
                    mempool = cp.get_default_memory_pool()
                    mem_info = cp.cuda.runtime.memGetInfo()

                    return MemorySnapshot(
                        timestamp=timestamp,
                        total_mb=(mem_info[0] + mem_info[1]) / (1024 * 1024),
                        used_mb=mem_info[1] / (1024 * 1024),
                        free_mb=mem_info[0] / (1024 * 1024),
                        allocated_mb=mempool.used_bytes() / (1024 * 1024),
                        device_id=self._device_id,
                    )
            except ImportError:
                pass

            # Try using torch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(self._device_id)
                    allocated = torch.cuda.memory_allocated(self._device_id)
                    reserved = torch.cuda.memory_reserved(self._device_id)
                    total = torch.cuda.get_device_properties(self._device_id).total_memory

                    return MemorySnapshot(
                        timestamp=timestamp,
                        total_mb=total / (1024 * 1024),
                        used_mb=reserved / (1024 * 1024),
                        free_mb=(total - reserved) / (1024 * 1024),
                        allocated_mb=allocated / (1024 * 1024),
                        device_id=self._device_id,
                    )
            except ImportError:
                pass

            # Fallback: return empty snapshot
            return MemorySnapshot(
                timestamp=timestamp,
                total_mb=0.0,
                used_mb=0.0,
                free_mb=0.0,
                device_id=self._device_id,
            )

        except Exception as e:
            self._logger.warning(f"Failed to get GPU memory usage: {e}")
            return MemorySnapshot(
                timestamp=timestamp,
                total_mb=0.0,
                used_mb=0.0,
                free_mb=0.0,
                device_id=self._device_id,
            )

    def establish_baseline(self) -> float:
        """Establish baseline memory usage by taking multiple samples.

        Returns:
            Baseline memory usage in MB.
        """
        samples = []
        for _ in range(self._baseline_samples):
            snapshot = self.get_current_memory_usage()
            samples.append(snapshot.used_mb)
            time.sleep(0.01)  # Brief pause between samples

        self._baseline_mb = sum(samples) / len(samples) if samples else 0.0
        self._logger.info(f"Baseline GPU memory: {self._baseline_mb:.2f} MB")
        return self._baseline_mb

    def record_snapshot(self, context: str = "") -> MemorySnapshot:
        """Record a memory snapshot.

        Args:
            context: Optional context string for the snapshot.

        Returns:
            The recorded MemorySnapshot.
        """
        snapshot = self.get_current_memory_usage()
        snapshot.context = context

        with self._lock:
            self._snapshots.append(snapshot)
            if snapshot.used_mb > self._peak_usage_mb:
                self._peak_usage_mb = snapshot.used_mb

            # Track allocations/deallocations
            if len(self._snapshots) > 1:
                prev = self._snapshots[-2]
                delta = snapshot.used_mb - prev.used_mb
                if delta > 0:
                    self._total_allocations += 1
                elif delta < 0:
                    self._total_deallocations += 1

        return snapshot

    def start_monitoring(self) -> None:
        """Start automatic memory monitoring in background thread."""
        if self._monitoring:
            return

        # Establish baseline first
        if self._baseline_mb is None:
            self.establish_baseline()

        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        self._logger.info("GPU memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop automatic memory monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self._logger.info("GPU memory monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring and not self._stop_event.is_set():
            self.record_snapshot("auto_monitor")
            self._stop_event.wait(timeout=self._monitoring_interval_ms / 1000.0)

    def mark_iteration(self) -> None:
        """Mark the end of an iteration for leak analysis."""
        with self._lock:
            self._iteration_count += 1
        self.record_snapshot(f"iteration_{self._iteration_count}")

    def analyze_for_leaks(self) -> MemoryLeakReport:
        """Analyze collected snapshots for memory leaks.

        Returns:
            MemoryLeakReport with analysis results.
        """
        with self._lock:
            snapshots = list(self._snapshots)

        if len(snapshots) < 2:
            return MemoryLeakReport(
                leak_detected=False,
                severity=MemoryLeakSeverity.NONE,
                details={"reason": "insufficient_data"},
            )

        baseline = self._baseline_mb or snapshots[0].used_mb
        current = snapshots[-1].used_mb
        leaked_mb = current - baseline

        # Calculate leak rate
        if self._iteration_count > 0:
            leak_rate = leaked_mb / self._iteration_count
        else:
            # Use time-based rate as fallback
            duration_seconds = snapshots[-1].timestamp - snapshots[0].timestamp
            leak_rate = leaked_mb / max(duration_seconds, 1.0)

        # Determine severity
        if leaked_mb <= 0:
            severity = MemoryLeakSeverity.NONE
            leak_detected = False
        elif leaked_mb < self.LEAK_THRESHOLD_MB:
            severity = MemoryLeakSeverity.NONE
            leak_detected = False
        elif leak_rate < self.LEAK_RATE_THRESHOLD:
            severity = MemoryLeakSeverity.LOW
            leak_detected = True
        elif leak_rate < self.CRITICAL_LEAK_RATE:
            severity = MemoryLeakSeverity.MEDIUM
            leak_detected = True
        elif leaked_mb < 1000:  # Less than 1GB
            severity = MemoryLeakSeverity.HIGH
            leak_detected = True
        else:
            severity = MemoryLeakSeverity.CRITICAL
            leak_detected = True

        # Generate recommendations
        recommendations: list[str] = []
        if leak_detected:
            if severity in [MemoryLeakSeverity.HIGH, MemoryLeakSeverity.CRITICAL]:
                recommendations.append("Immediately investigate and fix the memory leak")
                recommendations.append("Consider restarting the GPU context")
            if severity == MemoryLeakSeverity.MEDIUM:
                recommendations.append("Schedule cleanup during next opportunity")
                recommendations.append("Monitor memory usage closely")
            if severity == MemoryLeakSeverity.LOW:
                recommendations.append("Monitor memory usage over longer periods")
            recommendations.append("Call clear_gpu_cache() to attempt cleanup")
            recommendations.append("Check for unreleased tensor references")

        # Trend analysis
        trend = self._analyze_trend(snapshots)

        report = MemoryLeakReport(
            leak_detected=leak_detected,
            severity=severity,
            leaked_mb=max(0, leaked_mb),
            leak_rate_mb_per_iteration=leak_rate,
            baseline_mb=baseline,
            current_mb=current,
            iterations_analyzed=self._iteration_count,
            snapshots=snapshots[-10:],  # Keep last 10 snapshots
            recommendations=recommendations,
            details={
                "peak_usage_mb": self._peak_usage_mb,
                "total_allocations": self._total_allocations,
                "total_deallocations": self._total_deallocations,
                "trend": trend,
                "snapshot_count": len(snapshots),
            },
        )

        if leak_detected:
            self._logger.warning(
                f"Memory leak detected! Severity: {severity.value}, "
                f"Leaked: {leaked_mb:.2f} MB, Rate: {leak_rate:.2f} MB/iter"
            )

        return report

    def _analyze_trend(self, snapshots: list[MemorySnapshot]) -> dict[str, Any]:
        """Analyze memory usage trend.

        Args:
            snapshots: List of memory snapshots.

        Returns:
            Dictionary with trend analysis.
        """
        if len(snapshots) < 3:
            return {"trend": "insufficient_data"}

        # Calculate moving average
        window_size = min(5, len(snapshots))
        recent_usage = [s.used_mb for s in snapshots[-window_size:]]
        early_usage = [s.used_mb for s in snapshots[:window_size]]

        recent_avg = sum(recent_usage) / len(recent_usage)
        early_avg = sum(early_usage) / len(early_usage)

        # Determine trend
        if recent_avg > early_avg * 1.1:
            trend = "increasing"
        elif recent_avg < early_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"

        # Calculate variance to detect instability
        all_usage = [s.used_mb for s in snapshots]
        mean_usage = sum(all_usage) / len(all_usage)
        variance = sum((x - mean_usage) ** 2 for x in all_usage) / len(all_usage)

        return {
            "trend": trend,
            "early_average_mb": early_avg,
            "recent_average_mb": recent_avg,
            "variance": variance,
            "is_stable": variance < 100,  # Less than 100 MB^2 variance
        }

    def reset(self) -> None:
        """Reset detector state for new monitoring session."""
        with self._lock:
            self._snapshots.clear()
            self._baseline_mb = None
            self._iteration_count = 0
            self._peak_usage_mb = 0.0
            self._total_allocations = 0
            self._total_deallocations = 0

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current monitoring state.

        Returns:
            Dictionary with monitoring summary.
        """
        with self._lock:
            return {
                "monitoring_active": self._monitoring,
                "baseline_mb": self._baseline_mb,
                "snapshot_count": len(self._snapshots),
                "iteration_count": self._iteration_count,
                "peak_usage_mb": self._peak_usage_mb,
                "total_allocations": self._total_allocations,
                "total_deallocations": self._total_deallocations,
                "device_id": self._device_id,
            }


class MemoryLeakMitigator:
    """Provides strategies for mitigating detected memory leaks.

    Features:
    - Automatic cleanup triggers
    - Garbage collection coordination
    - Memory pool reset
    - Circuit of breaker for severe leaks
    """

    def __init__(
        self,
        memory_manager: "GPUMemoryManager",
        leak_detector: GPUMemoryLeakDetector | None = None,
    ):
        """Initialize the mitigator.

        Args:
            memory_manager: GPU memory manager instance.
            leak_detector: Optional leak detector instance.
        """
        self._memory_manager = memory_manager
        self._leak_detector = leak_detector or GPUMemoryLeakDetector()
        self._logger = logging.getLogger(__name__)

        # Circuit breaker state
        self._circuit_open = False
        self._consecutive_critical_leaks = 0
        self._max_critical_before_break = 3

    def attempt_cleanup(self, aggressive: bool = False) -> dict[str, Any]:
        """Attempt to clean up leaked memory.

        Args:
            aggressive: Whether to use aggressive cleanup strategies.

        Returns:
            Dictionary with cleanup results.
        """
        results: dict[str, Any] = {
            "cleanup_attempted": True,
            "strategies_used": [],
            "memory_freed_mb": 0.0,
            "success": False,
        }

        before_snapshot = self._leak_detector.get_current_memory_usage()
        before_mb = before_snapshot.used_mb

        # Strategy 1: Clear GPU cache
        try:
            self._memory_manager.clear_gpu_cache()
            results["strategies_used"].append("clear_cache")
        except Exception as e:
            self._logger.warning(f"Cache clear failed: {e}")

        # Strategy 2: Run garbage collection
        import gc
        gc.collect()
        results["strategies_used"].append("gc_collect")

        if aggressive:
            # Strategy 3: Reset memory pool
            try:
                self._memory_manager.cleanup_pool()
                results["strategies_used"].append("pool_cleanup")
            except Exception as e:
                self._logger.warning(f"Pool cleanup failed: {e}")

            # Strategy 4: Force CUDA synchronization
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    results["strategies_used"].append("torch_cuda_sync")
            except Exception:
                pass

            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                results["strategies_used"].append("cupy_sync")
            except Exception:
                pass

        # Measure results
        after_snapshot = self._leak_detector.get_current_memory_usage()
        after_mb = after_snapshot.used_mb

        freed_mb = before_mb - after_mb
        results["memory_freed_mb"] = max(0, freed_mb)
        results["success"] = freed_mb > 0

        self._logger.info(
            f"Cleanup completed: freed {freed_mb:.2f} MB using "
            f"{', '.join(results['strategies_used'])}"
        )

        return results

    def check_and_mitigate(self) -> dict[str, Any]:
        """Check for leaks and automatically mitigate if found.

        Returns:
            Dictionary with check and mitigation results.
        """
        if self._circuit_open:
            return {
                "action": "circuit_breaker_open",
                "message": "Execution halted due to repeated critical leaks",
                "recommendation": "Manual intervention required",
            }

        report = self._leak_detector.analyze_for_leaks()

        result: dict[str, Any] = {
            "leak_report": {
                "detected": report.leak_detected,
                "severity": report.severity.value,
                "leaked_mb": report.leaked_mb,
            },
            "mitigation_performed": False,
            "cleanup_result": None,
        }

        if not report.leak_detected:
            self._consecutive_critical_leaks = 0
            return result

        # Perform mitigation based on severity
        if report.severity == MemoryLeakSeverity.LOW:
            # Just log for low severity
            self._logger.info(f"Low severity leak detected: {report.leaked_mb:.2f} MB")

        elif report.severity == MemoryLeakSeverity.MEDIUM:
            # Attempt standard cleanup
            cleanup_result = self.attempt_cleanup(aggressive=False)
            result["mitigation_performed"] = True
            result["cleanup_result"] = cleanup_result

        elif report.severity in [MemoryLeakSeverity.HIGH, MemoryLeakSeverity.CRITICAL]:
            # Attempt aggressive cleanup
            cleanup_result = self.attempt_cleanup(aggressive=True)
            result["mitigation_performed"] = True
            result["cleanup_result"] = cleanup_result

            if report.severity == MemoryLeakSeverity.CRITICAL:
                self._consecutive_critical_leaks += 1
                if self._consecutive_critical_leaks >= self._max_critical_before_break:
                    self._circuit_open = True
                    result["circuit_breaker_triggered"] = True
                    self._logger.error(
                        "Circuit breaker triggered due to repeated critical memory leaks"
                    )

        return result

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to allow execution again."""
        self._circuit_open = False
        self._consecutive_critical_leaks = 0
        self._logger.info("Circuit breaker reset")

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (execution halted)."""
        return self._circuit_open


# =============================================================================
# GPU Memory Manager Class
# =============================================================================


class GPUMemoryManager:
    """Manages GPU memory for cuQuantum executions.

    This class provides:
    - Pre-execution memory validation
    - Memory pooling for efficient reuse
    - Batch processing with memory monitoring
    - Intelligent fallback when GPU memory is insufficient

    Example:
        >>> manager = GPUMemoryManager(config)
        >>> estimate = manager.estimate_memory(30)  # 30 qubits
        >>> if estimate.can_fit_on_device:
        ...     result = manager.execute_with_memory_check(circuit)
    """

    def __init__(
        self,
        config: CuQuantumConfig | None = None,
        allocation_strategy: MemoryAllocationStrategy = MemoryAllocationStrategy.LAZY_POOL,
        cleanup_policy: MemoryCleanupPolicy = MemoryCleanupPolicy.DEFERRED,
    ):
        """Initialize GPU memory manager.

        Args:
            config: cuQuantum configuration
            allocation_strategy: Memory allocation strategy
            cleanup_policy: Memory cleanup policy
        """
        self._config = config or CuQuantumConfig()
        self._allocation_strategy = allocation_strategy
        self._cleanup_policy = cleanup_policy
        self._logger = logging.getLogger(__name__)

        # Memory pool state
        self._pool_initialized = False
        self._pool_stats = MemoryPoolStats()
        self._pool_lock = threading.Lock()

        # Cached GPU info
        self._gpu_devices: list[GPUDeviceInfo] = []
        self._selected_device_id = self._config.gpu_device_id

        # Memory cache for reuse
        self._memory_cache: dict[int, Any] = {}  # qubit_count -> cached allocation

    # =========================================================================
    # Pre-Execution Memory Checks (Step 2.4.1)
    # =========================================================================

    def estimate_memory(
        self,
        qubit_count: int,
        precision: CuQuantumPrecision | None = None,
        workspace_mb: float | None = None,
    ) -> MemoryEstimate:
        """Estimate GPU memory requirements for a circuit.

        This implements the memory estimation formula from the guide:
        Total = (2^n * bytes_per_amplitude) + workspace + overhead

        Args:
            qubit_count: Number of qubits in the circuit
            precision: Numerical precision (defaults to config)
            workspace_mb: Workspace size in MB (defaults to config)

        Returns:
            MemoryEstimate with detailed breakdown
        """
        precision = precision or self._config.precision
        workspace_mb = workspace_mb or self._config.workspace_size_mb

        # Calculate state vector size: 2^n amplitudes * bytes per amplitude
        bytes_per_amp = BYTES_PER_AMPLITUDE[precision]
        state_vector_bytes = (2**qubit_count) * bytes_per_amp
        state_vector_mb = state_vector_bytes / (1024 * 1024)

        # Total required memory
        total_required_mb = state_vector_mb + workspace_mb + TOTAL_OVERHEAD_MB

        # Add safety margin
        total_with_safety_mb = total_required_mb * (1 + MEMORY_SAFETY_MARGIN)

        # Check if it fits on device
        can_fit = False
        recommended_device = self._selected_device_id
        fallback_required = False
        fallback_reason = ""

        # Get GPU info
        gpu_info = self._get_gpu_info(self._selected_device_id)
        if gpu_info:
            available_mb = gpu_info.free_memory_mb
            can_fit = total_with_safety_mb <= available_mb

            if not can_fit:
                # Check other GPUs
                for device in self._gpu_devices:
                    if device.free_memory_mb >= total_with_safety_mb:
                        recommended_device = device.device_id
                        can_fit = True
                        break

                if not can_fit:
                    fallback_required = True
                    fallback_reason = (
                        f"Required {total_with_safety_mb:.1f} MB but only "
                        f"{available_mb:.1f} MB available on GPU {self._selected_device_id}"
                    )
        else:
            fallback_required = True
            fallback_reason = "No GPU device available"

        return MemoryEstimate(
            qubit_count=qubit_count,
            state_vector_mb=state_vector_mb,
            workspace_mb=workspace_mb,
            overhead_mb=TOTAL_OVERHEAD_MB,
            total_required_mb=total_required_mb,
            total_with_safety_mb=total_with_safety_mb,
            precision=precision,
            can_fit_on_device=can_fit,
            recommended_device_id=recommended_device,
            fallback_required=fallback_required,
            fallback_reason=fallback_reason,
        )

    def check_memory_before_execution(
        self,
        qubit_count: int,
        raise_on_insufficient: bool = True,
    ) -> tuple[bool, MemoryEstimate]:
        """Check if GPU has sufficient memory before execution.

        Args:
            qubit_count: Number of qubits
            raise_on_insufficient: Raise exception if memory insufficient

        Returns:
            Tuple of (can_execute, memory_estimate)

        Raises:
            CuQuantumMemoryError: If memory insufficient and raise_on_insufficient=True
        """
        from proxima.backends.cuquantum_adapter import CuQuantumMemoryError

        estimate = self.estimate_memory(qubit_count)

        if not estimate.can_fit_on_device:
            self._logger.warning(
                f"Insufficient GPU memory for {qubit_count} qubits. "
                f"Required: {estimate.total_with_safety_mb:.1f} MB"
            )

            if raise_on_insufficient and not self._config.fallback_to_cpu:
                gpu_info = self._get_gpu_info(self._selected_device_id)
                available_mb = gpu_info.free_memory_mb if gpu_info else 0
                raise CuQuantumMemoryError(
                    required_mb=estimate.total_with_safety_mb,
                    available_mb=available_mb,
                    qubit_count=qubit_count,
                    device_id=self._selected_device_id,
                )

        return estimate.can_fit_on_device, estimate

    def query_current_gpu_memory(
        self, device_id: int | None = None
    ) -> dict[str, float]:
        """Query current GPU memory status.

        Args:
            device_id: GPU device ID (defaults to configured device)

        Returns:
            Dictionary with total_mb, used_mb, free_mb, utilization
        """
        device_id = device_id or self._selected_device_id

        gpu_info = self._get_gpu_info(device_id)
        if not gpu_info:
            return {
                "total_mb": 0,
                "used_mb": 0,
                "free_mb": 0,
                "utilization": 0.0,
            }

        used_mb = gpu_info.total_memory_mb - gpu_info.free_memory_mb
        utilization = (
            used_mb / gpu_info.total_memory_mb if gpu_info.total_memory_mb > 0 else 0
        )

        return {
            "total_mb": gpu_info.total_memory_mb,
            "used_mb": used_mb,
            "free_mb": gpu_info.free_memory_mb,
            "utilization": utilization,
        }

    def clear_gpu_cache(self, device_id: int | None = None) -> bool:
        """Clear GPU cache to free memory.

        Args:
            device_id: GPU device ID (defaults to configured device)

        Returns:
            True if cache was cleared successfully
        """
        try:
            # Try different methods to clear cache

            # Method 1: Use cupy
            try:
                import cupy as cp

                with cp.cuda.Device(device_id or self._selected_device_id):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                self._logger.debug("Cleared GPU cache using cupy")
                return True
            except Exception:
                pass

            # Method 2: Use torch if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self._logger.debug("Cleared GPU cache using torch")
                    return True
            except Exception:
                pass

            # Method 3: Run garbage collection
            import gc

            gc.collect()
            self._logger.debug("Ran garbage collection")
            return True

        except Exception as e:
            self._logger.warning(f"Failed to clear GPU cache: {e}")
            return False

    # =========================================================================
    # Memory Pooling (Step 2.4.2)
    # =========================================================================

    def initialize_memory_pool(self, pool_size_mb: float | None = None) -> bool:
        """Initialize GPU memory pool for efficient reuse.

        Args:
            pool_size_mb: Pool size in MB (defaults to 50% of free memory)

        Returns:
            True if pool initialized successfully
        """
        with self._pool_lock:
            if self._pool_initialized:
                self._logger.debug("Memory pool already initialized")
                return True

            try:
                gpu_info = self._get_gpu_info(self._selected_device_id)
                if not gpu_info:
                    self._logger.warning("No GPU available for memory pool")
                    return False

                # Default to 50% of free memory
                if pool_size_mb is None:
                    pool_size_mb = gpu_info.free_memory_mb * 0.5

                # Initialize pool using cupy if available
                try:
                    import cupy as cp

                    with cp.cuda.Device(self._selected_device_id):
                        # Create memory pool with specified size
                        mempool = cp.get_default_memory_pool()
                        mempool.set_limit(size=int(pool_size_mb * 1024 * 1024))

                        self._pool_stats.pool_size_mb = pool_size_mb
                        self._pool_stats.available_mb = pool_size_mb
                        self._pool_initialized = True

                        self._logger.info(
                            f"GPU memory pool initialized: {pool_size_mb:.1f} MB "
                            f"on device {self._selected_device_id}"
                        )
                        return True
                except Exception as e:
                    self._logger.debug(f"CuPy pool init failed: {e}")

                # Fallback: Just track stats without actual pool
                self._pool_stats.pool_size_mb = pool_size_mb
                self._pool_stats.available_mb = pool_size_mb
                self._pool_initialized = True
                self._logger.info(
                    f"Memory pool stats initialized: {pool_size_mb:.1f} MB"
                )
                return True

            except Exception as e:
                self._logger.error(f"Failed to initialize memory pool: {e}")
                return False

    def allocate_from_pool(self, size_mb: float) -> tuple[bool, int | None]:
        """Allocate memory from pool.

        Args:
            size_mb: Required memory in MB

        Returns:
            Tuple of (success, allocation_id)
        """
        with self._pool_lock:
            if not self._pool_initialized:
                self.initialize_memory_pool()

            if size_mb > self._pool_stats.available_mb:
                self._logger.warning(
                    f"Insufficient pool memory: need {size_mb:.1f} MB, "
                    f"available {self._pool_stats.available_mb:.1f} MB"
                )
                return False, None

            # Create allocation
            allocation_id = id(time.time())  # Simple unique ID
            self._pool_stats.allocated_mb += size_mb
            self._pool_stats.available_mb -= size_mb
            self._pool_stats.allocation_count += 1

            self._logger.debug(
                f"Allocated {size_mb:.1f} MB from pool (ID: {allocation_id})"
            )
            return True, allocation_id

    def release_to_pool(self, allocation_id: int, size_mb: float) -> bool:
        """Release memory back to pool.

        Args:
            allocation_id: Allocation ID from allocate_from_pool
            size_mb: Size of allocation in MB

        Returns:
            True if released successfully
        """
        with self._pool_lock:
            self._pool_stats.allocated_mb -= size_mb
            self._pool_stats.available_mb += size_mb
            self._pool_stats.allocation_count -= 1
            self._pool_stats.reuse_count += 1

            # Update cache hit rate
            total_ops = self._pool_stats.allocation_count + self._pool_stats.reuse_count
            if total_ops > 0:
                self._pool_stats.cache_hit_rate = (
                    self._pool_stats.reuse_count / total_ops
                )

            self._logger.debug(
                f"Released {size_mb:.1f} MB to pool (ID: {allocation_id})"
            )
            return True

    def get_pool_stats(self) -> MemoryPoolStats:
        """Get current memory pool statistics."""
        with self._pool_lock:
            return MemoryPoolStats(
                pool_size_mb=self._pool_stats.pool_size_mb,
                allocated_mb=self._pool_stats.allocated_mb,
                available_mb=self._pool_stats.available_mb,
                allocation_count=self._pool_stats.allocation_count,
                reuse_count=self._pool_stats.reuse_count,
                cache_hit_rate=self._pool_stats.cache_hit_rate,
                last_cleanup_time=self._pool_stats.last_cleanup_time,
            )

    def cleanup_pool(self) -> bool:
        """Clean up memory pool and release all allocations.

        Returns:
            True if cleanup successful
        """
        with self._pool_lock:
            self._memory_cache.clear()
            self.clear_gpu_cache()

            self._pool_stats.allocated_mb = 0
            self._pool_stats.available_mb = self._pool_stats.pool_size_mb
            self._pool_stats.allocation_count = 0
            self._pool_stats.last_cleanup_time = time.time()

            self._logger.info("Memory pool cleaned up")
            return True

    # =========================================================================
    # Batch Processing (Step 2.4.3)
    # =========================================================================

    def execute_batch(
        self,
        circuits: list[Any],
        executor: Callable[[Any], Any],
        options: dict[str, Any] | None = None,
    ) -> BatchExecutionResult:
        """Execute multiple circuits in batch with memory monitoring.

        Circuits are executed sequentially to avoid memory fragmentation.
        Memory is monitored and cleared between executions if needed.

        Args:
            circuits: List of circuits to execute
            executor: Function to execute each circuit
            options: Execution options

        Returns:
            BatchExecutionResult with all results
        """
        options = options or {}
        batch_result = BatchExecutionResult(total_circuits=len(circuits))

        start_time = time.perf_counter()
        peak_memory = 0.0

        for i, circuit in enumerate(circuits):
            self._logger.debug(f"Executing circuit {i+1}/{len(circuits)}")

            try:
                # Check memory before execution
                current_memory = self.query_current_gpu_memory()
                if current_memory["utilization"] > MEMORY_WARNING_THRESHOLD:
                    self._logger.warning(
                        f"High GPU memory usage ({current_memory['utilization']*100:.1f}%), "
                        f"clearing cache before circuit {i+1}"
                    )
                    self.clear_gpu_cache()

                # Execute circuit
                result = executor(circuit)
                batch_result.results.append(result)
                batch_result.successful_circuits += 1

                # Track peak memory
                post_memory = self.query_current_gpu_memory()
                peak_memory = max(peak_memory, post_memory["used_mb"])

            except Exception as e:
                self._logger.error(f"Circuit {i+1} failed: {e}")
                batch_result.errors.append(
                    {
                        "circuit_index": i,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                batch_result.failed_circuits += 1

                # Clear cache after failure
                self.clear_gpu_cache()

        batch_result.total_execution_time_ms = (time.perf_counter() - start_time) * 1000
        batch_result.peak_memory_mb = peak_memory

        # Calculate memory fragmentation
        pool_stats = self.get_pool_stats()
        if pool_stats.pool_size_mb > 0:
            batch_result.memory_fragmentation = 1 - (
                pool_stats.available_mb / pool_stats.pool_size_mb
            )

        self._logger.info(
            f"Batch execution complete: {batch_result.successful_circuits}/"
            f"{batch_result.total_circuits} circuits succeeded, "
            f"peak memory: {peak_memory:.1f} MB"
        )

        return batch_result

    # =========================================================================
    # Fallback Strategy (Step 2.4.4)
    # =========================================================================

    def should_fallback_to_cpu(
        self,
        qubit_count: int,
        force_gpu: bool = False,
    ) -> tuple[bool, str]:
        """Determine if execution should fall back to CPU.

        Args:
            qubit_count: Number of qubits
            force_gpu: If True, never fallback (may raise exception)

        Returns:
            Tuple of (should_fallback, reason)
        """
        if force_gpu:
            return False, ""

        if not self._config.fallback_to_cpu:
            return False, ""

        # Check GPU availability
        if not self._gpu_devices:
            return True, "No GPU devices available"

        # Check memory
        estimate = self.estimate_memory(qubit_count)
        if estimate.fallback_required:
            return True, estimate.fallback_reason

        # Check for small circuits (overhead may negate benefit)
        if qubit_count < 15:
            return True, f"Small circuit ({qubit_count} qubits) - CPU may be faster"

        return False, ""

    def get_fallback_recommendation(
        self,
        qubit_count: int,
    ) -> dict[str, Any]:
        """Get detailed fallback recommendation.

        Args:
            qubit_count: Number of qubits

        Returns:
            Dictionary with recommendation details
        """
        estimate = self.estimate_memory(qubit_count)
        should_fallback, reason = self.should_fallback_to_cpu(qubit_count)

        recommendation = {
            "should_fallback": should_fallback,
            "reason": reason,
            "memory_estimate": {
                "state_vector_mb": estimate.state_vector_mb,
                "total_required_mb": estimate.total_required_mb,
                "available_mb": self.query_current_gpu_memory()["free_mb"],
            },
            "alternatives": [],
        }

        if should_fallback:
            recommendation["alternatives"] = [
                {
                    "action": "reduce_qubits",
                    "description": f"Reduce circuit to {estimate.qubit_count - 2} qubits",
                },
                {
                    "action": "use_single_precision",
                    "description": "Switch to single precision (halves memory)",
                },
                {
                    "action": "use_different_backend",
                    "description": "Use qsim for CPU-optimized execution",
                },
            ]

        return recommendation

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_gpu_info(self, device_id: int) -> GPUDeviceInfo | None:
        """Get GPU info for a specific device."""
        for device in self._gpu_devices:
            if device.device_id == device_id:
                return device
        return None

    def set_gpu_devices(self, devices: list[GPUDeviceInfo]) -> None:
        """Set available GPU devices (called by adapter)."""
        self._gpu_devices = devices

    # =========================================================================
    # Memory Leak Detection Integration (5% completion)
    # =========================================================================

    def get_leak_detector(self) -> GPUMemoryLeakDetector:
        """Get or create a memory leak detector for this manager.

        Returns:
            GPUMemoryLeakDetector instance.
        """
        if not hasattr(self, '_leak_detector') or self._leak_detector is None:
            self._leak_detector = GPUMemoryLeakDetector(
                device_id=self._selected_device_id
            )
        return self._leak_detector

    def get_leak_mitigator(self) -> MemoryLeakMitigator:
        """Get or create a memory leak mitigator for this manager.

        Returns:
            MemoryLeakMitigator instance.
        """
        if not hasattr(self, '_leak_mitigator') or self._leak_mitigator is None:
            self._leak_mitigator = MemoryLeakMitigator(
                memory_manager=self,
                leak_detector=self.get_leak_detector(),
            )
        return self._leak_mitigator

    def start_leak_monitoring(self) -> None:
        """Start memory leak monitoring.

        This should be called before starting circuit execution.
        """
        detector = self.get_leak_detector()
        detector.start_monitoring()
        self._logger.info("Memory leak monitoring started")

    def stop_leak_monitoring(self) -> MemoryLeakReport:
        """Stop memory leak monitoring and get report.

        Returns:
            MemoryLeakReport with analysis results.
        """
        detector = self.get_leak_detector()
        detector.stop_monitoring()
        report = detector.analyze_for_leaks()
        self._logger.info(
            f"Memory leak monitoring stopped. "
            f"Leak detected: {report.leak_detected}, "
            f"Severity: {report.severity.value}"
        )
        return report

    def check_for_leaks(self) -> MemoryLeakReport:
        """Check for memory leaks without stopping monitoring.

        Returns:
            MemoryLeakReport with current analysis.
        """
        detector = self.get_leak_detector()
        return detector.analyze_for_leaks()

    def record_execution_snapshot(self, context: str = "") -> MemorySnapshot:
        """Record a memory snapshot during execution.

        Args:
            context: Optional context string for the snapshot.

        Returns:
            MemorySnapshot with current memory state.
        """
        detector = self.get_leak_detector()
        return detector.record_snapshot(context)

    def mark_iteration_complete(self) -> None:
        """Mark the completion of an iteration for leak analysis."""
        detector = self.get_leak_detector()
        detector.mark_iteration()

    def execute_with_leak_detection(
        self,
        executor: Callable[[], Any],
        context: str = "execution",
    ) -> tuple[Any, MemorySnapshot, MemorySnapshot]:
        """Execute a function with before/after memory snapshots.

        Args:
            executor: Function to execute.
            context: Context string for snapshots.

        Returns:
            Tuple of (result, before_snapshot, after_snapshot).
        """
        before = self.record_execution_snapshot(f"{context}_before")

        try:
            result = executor()
        finally:
            after = self.record_execution_snapshot(f"{context}_after")

        # Log if significant memory increase
        delta_mb = after.used_mb - before.used_mb
        if delta_mb > 100:  # More than 100 MB increase
            self._logger.warning(
                f"Significant memory increase during {context}: {delta_mb:.2f} MB"
            )

        return result, before, after

    def execute_batch_with_leak_detection(
        self,
        circuits: list[Any],
        executor: Callable[[Any], Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[BatchExecutionResult, MemoryLeakReport]:
        """Execute batch with memory leak detection.

        Args:
            circuits: List of circuits to execute.
            executor: Function to execute each circuit.
            options: Execution options.

        Returns:
            Tuple of (BatchExecutionResult, MemoryLeakReport).
        """
        detector = self.get_leak_detector()
        detector.reset()
        detector.establish_baseline()

        options = options or {}
        batch_result = BatchExecutionResult(total_circuits=len(circuits))

        start_time = time.perf_counter()
        peak_memory = 0.0

        for i, circuit in enumerate(circuits):
            self._logger.debug(f"Executing circuit {i+1}/{len(circuits)}")

            detector.record_snapshot(f"circuit_{i+1}_before")

            try:
                # Check memory before execution
                current_memory = self.query_current_gpu_memory()
                if current_memory["utilization"] > MEMORY_WARNING_THRESHOLD:
                    self._logger.warning(
                        f"High GPU memory usage ({current_memory['utilization']*100:.1f}%), "
                        f"attempting cleanup before circuit {i+1}"
                    )
                    mitigator = self.get_leak_mitigator()
                    mitigator.attempt_cleanup(aggressive=False)

                # Execute circuit
                result = executor(circuit)
                batch_result.results.append(result)
                batch_result.successful_circuits += 1

                # Track peak memory
                post_memory = self.query_current_gpu_memory()
                peak_memory = max(peak_memory, post_memory["used_mb"])

            except Exception as e:
                self._logger.error(f"Circuit {i+1} failed: {e}")
                batch_result.errors.append(
                    {
                        "circuit_index": i,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                batch_result.failed_circuits += 1

                # Clear cache after failure
                self.clear_gpu_cache()

            detector.record_snapshot(f"circuit_{i+1}_after")
            detector.mark_iteration()

            # Check for critical leaks after each circuit
            if (i + 1) % 10 == 0:  # Check every 10 circuits
                interim_report = detector.analyze_for_leaks()
                if interim_report.severity == MemoryLeakSeverity.CRITICAL:
                    self._logger.error(
                        f"Critical memory leak detected at circuit {i+1}, "
                        f"attempting mitigation"
                    )
                    mitigator = self.get_leak_mitigator()
                    mitigator.attempt_cleanup(aggressive=True)

        batch_result.total_execution_time_ms = (time.perf_counter() - start_time) * 1000
        batch_result.peak_memory_mb = peak_memory

        # Calculate memory fragmentation
        pool_stats = self.get_pool_stats()
        if pool_stats.pool_size_mb > 0:
            batch_result.memory_fragmentation = 1 - (
                pool_stats.available_mb / pool_stats.pool_size_mb
            )

        # Get final leak report
        leak_report = detector.analyze_for_leaks()

        self._logger.info(
            f"Batch execution complete: {batch_result.successful_circuits}/"
            f"{batch_result.total_circuits} circuits succeeded, "
            f"peak memory: {peak_memory:.1f} MB, "
            f"leak detected: {leak_report.leak_detected}"
        )

        return batch_result, leak_report

    def get_leak_detection_summary(self) -> dict[str, Any]:
        """Get summary of leak detection state.

        Returns:
            Dictionary with leak detection summary.
        """
        detector = self.get_leak_detector()
        mitigator = self.get_leak_mitigator()

        summary = detector.get_summary()
        summary["circuit_breaker_open"] = mitigator.is_circuit_open()

        return summary

    def reset_leak_detection(self) -> None:
        """Reset leak detection state for new session."""
        if hasattr(self, '_leak_detector') and self._leak_detector:
            self._leak_detector.reset()
        if hasattr(self, '_leak_mitigator') and self._leak_mitigator:
            self._leak_mitigator.reset_circuit_breaker()
        self._logger.info("Leak detection state reset")

    def calculate_max_qubits_for_memory(
        self,
        available_mb: float,
        precision: CuQuantumPrecision | None = None,
    ) -> int:
        """Calculate maximum qubits that fit in available memory.

        Args:
            available_mb: Available GPU memory in MB
            precision: Numerical precision

        Returns:
            Maximum qubit count
        """
        precision = precision or self._config.precision
        bytes_per_amp = BYTES_PER_AMPLITUDE[precision]

        # Available for state vector = total - workspace - overhead - safety
        usable_mb = available_mb - self._config.workspace_size_mb - TOTAL_OVERHEAD_MB
        usable_mb *= 1 - MEMORY_SAFETY_MARGIN

        if usable_mb <= 0:
            return 0

        # 2^n * bytes_per_amp = usable_bytes
        # n = log2(usable_bytes / bytes_per_amp)
        usable_bytes = usable_mb * 1024 * 1024
        max_amplitudes = usable_bytes / bytes_per_amp

        import math

        max_qubits = int(math.log2(max_amplitudes))

        return max(0, max_qubits)


# =============================================================================
# Convenience Functions
# =============================================================================


def estimate_gpu_memory_for_qubits(
    qubit_count: int,
    precision: str = "double",
    workspace_mb: float = 1024,
) -> dict[str, float]:
    """Quick estimate of GPU memory for a given qubit count.

    Args:
        qubit_count: Number of qubits
        precision: "single" or "double"
        workspace_mb: Workspace size in MB

    Returns:
        Dictionary with memory breakdown
    """
    prec = (
        CuQuantumPrecision.SINGLE
        if precision == "single"
        else CuQuantumPrecision.DOUBLE
    )
    bytes_per_amp = BYTES_PER_AMPLITUDE[prec]

    state_vector_bytes = (2**qubit_count) * bytes_per_amp
    state_vector_mb = state_vector_bytes / (1024 * 1024)

    total_mb = state_vector_mb + workspace_mb + TOTAL_OVERHEAD_MB
    total_with_safety = total_mb * (1 + MEMORY_SAFETY_MARGIN)

    return {
        "qubit_count": qubit_count,
        "state_vector_mb": state_vector_mb,
        "workspace_mb": workspace_mb,
        "overhead_mb": TOTAL_OVERHEAD_MB,
        "total_mb": total_mb,
        "total_with_safety_mb": total_with_safety,
        "precision": precision,
    }


def get_recommended_precision_for_memory(
    qubit_count: int,
    available_mb: float,
    workspace_mb: float = 1024,
) -> str:
    """Get recommended precision based on available memory.

    Args:
        qubit_count: Number of qubits
        available_mb: Available GPU memory in MB
        workspace_mb: Workspace size in MB

    Returns:
        "single" or "double"
    """
    double_estimate = estimate_gpu_memory_for_qubits(
        qubit_count, "double", workspace_mb
    )
    single_estimate = estimate_gpu_memory_for_qubits(
        qubit_count, "single", workspace_mb
    )

    if double_estimate["total_with_safety_mb"] <= available_mb:
        return "double"
    elif single_estimate["total_with_safety_mb"] <= available_mb:
        return "single"
    else:
        return "double"  # Return double anyway, let validation handle it
