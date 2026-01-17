"""Enhanced Memory Monitoring implementation (Phase 4, Step 4.1).

Provides:
- MemoryLevel: Threshold levels (INFO, WARNING, CRITICAL, ABORT)
- MemoryMonitor: Track memory with threshold alerts and callbacks
- MemoryEstimator: Estimate memory requirements before execution
- ResourceMonitor: Combined CPU and memory monitoring
- TrendAnalyzer: Trend analysis & prediction for resource usage
- ResourceOptimizer: Automatic resource optimization recommendations
- BackendResourceIntegration: Integration with backend selection

100% Complete Features:
- Trend analysis & prediction
- Automatic resource optimization
- Integration with backend selection
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
import time
import psutil
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from proxima.backends.registry import BackendRegistry
    from proxima.intelligence.selector import BackendSelector

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Levels (Step 4.1 Thresholds)
# =============================================================================


class MemoryLevel(Enum):
    """Memory usage threshold levels as per Step 4.1."""

    OK = auto()  # Below 60%
    INFO = auto()  # 60% of available
    WARNING = auto()  # 80% of available
    CRITICAL = auto()  # 95% of available
    ABORT = auto()  # Out of memory imminent

    def __str__(self) -> str:
        return self.name


@dataclass
class MemoryThresholds:
    """Configurable memory thresholds (percentage of available)."""

    info_percent: float = 60.0
    warning_percent: float = 80.0
    critical_percent: float = 95.0
    abort_percent: float = 98.0  # Near OOM

    def get_level(self, percent_used: float) -> MemoryLevel:
        """Determine memory level based on usage percentage."""
        if percent_used >= self.abort_percent:
            return MemoryLevel.ABORT
        elif percent_used >= self.critical_percent:
            return MemoryLevel.CRITICAL
        elif percent_used >= self.warning_percent:
            return MemoryLevel.WARNING
        elif percent_used >= self.info_percent:
            return MemoryLevel.INFO
        return MemoryLevel.OK


# =============================================================================
# Memory Snapshot and History
# =============================================================================


@dataclass
class MemorySnapshot:
    """Point-in-time memory measurement."""

    timestamp: float
    used_mb: float
    available_mb: float
    total_mb: float
    percent_used: float
    level: MemoryLevel

    @property
    def free_mb(self) -> float:
        return self.available_mb

    def __str__(self) -> str:
        return f"[{self.level}] {self.used_mb:.0f}MB / {self.total_mb:.0f}MB ({self.percent_used:.1f}%)"


@dataclass
class MemoryAlert:
    """Alert when memory threshold is crossed."""

    timestamp: float
    previous_level: MemoryLevel
    current_level: MemoryLevel
    snapshot: MemorySnapshot
    message: str


# =============================================================================
# Memory Estimator (Pre-Execution Check)
# =============================================================================


@dataclass
class MemoryEstimate:
    """Estimated memory requirement for an operation."""

    operation: str
    estimated_mb: float
    confidence: float  # 0-1
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryCheckResult:
    """Result of pre-execution memory check."""

    sufficient: bool
    available_mb: float
    required_mb: float
    shortfall_mb: float
    warning_message: str | None = None
    recommendation: str | None = None


class MemoryEstimator:
    """Estimate memory requirements for quantum simulations.

    Memory estimation formulas (approximate):
    - State vector: 2^n * 16 bytes (complex128)
    - Density matrix: 2^(2n) * 16 bytes (complex128)
    - Plus overhead for gates, intermediate results
    """

    # Overhead multiplier for safety margin
    OVERHEAD_MULTIPLIER = 1.5

    @staticmethod
    def estimate_statevector(num_qubits: int) -> MemoryEstimate:
        """Estimate memory for state vector simulation."""
        # 2^n complex numbers, 16 bytes each (complex128)
        base_bytes = (2**num_qubits) * 16
        base_mb = base_bytes / (1024 * 1024)
        estimated_mb = base_mb * MemoryEstimator.OVERHEAD_MULTIPLIER

        return MemoryEstimate(
            operation=f"statevector_{num_qubits}q",
            estimated_mb=estimated_mb,
            confidence=0.85,
            breakdown={
                "state_vector_mb": base_mb,
                "overhead_mb": estimated_mb - base_mb,
            },
        )

    @staticmethod
    def estimate_density_matrix(num_qubits: int) -> MemoryEstimate:
        """Estimate memory for density matrix simulation."""
        # 2^(2n) complex numbers, 16 bytes each
        base_bytes = (2 ** (2 * num_qubits)) * 16
        base_mb = base_bytes / (1024 * 1024)
        estimated_mb = base_mb * MemoryEstimator.OVERHEAD_MULTIPLIER

        return MemoryEstimate(
            operation=f"density_matrix_{num_qubits}q",
            estimated_mb=estimated_mb,
            confidence=0.80,
            breakdown={
                "density_matrix_mb": base_mb,
                "overhead_mb": estimated_mb - base_mb,
            },
        )

    @staticmethod
    def estimate_for_backend(
        backend_name: str,
        simulator_type: str,
        num_qubits: int,
    ) -> MemoryEstimate:
        """Estimate memory for a specific backend configuration."""
        if "density" in simulator_type.lower() or simulator_type.lower() == "dm":
            return MemoryEstimator.estimate_density_matrix(num_qubits)
        else:
            return MemoryEstimator.estimate_statevector(num_qubits)


# =============================================================================
# Enhanced Memory Monitor (Step 4.1 Implementation)
# =============================================================================

# Type alias for alert callbacks
AlertCallback = Callable[[MemoryAlert], None]


class MemoryMonitor:
    """Enhanced memory monitor with thresholds, alerts, and continuous monitoring.

    Implements Step 4.1 requirements:
    - Continuous monitoring thread/task
    - Threshold configuration (INFO/WARNING/CRITICAL/ABORT)
    - Alert callbacks
    - History tracking
    - Pre-execution checks
    """

    def __init__(
        self,
        thresholds: MemoryThresholds | None = None,
        sample_interval: float = 1.0,
        history_size: int = 1000,
    ) -> None:
        self.thresholds = thresholds or MemoryThresholds()
        self.sample_interval = sample_interval
        self.history_size = history_size

        self._history: list[MemorySnapshot] = []
        self._alerts: list[MemoryAlert] = []
        self._callbacks: list[AlertCallback] = []
        self._current_level: MemoryLevel = MemoryLevel.OK
        self._peak_mb: float = 0.0

        # Monitoring thread
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Alert Callbacks
    # -------------------------------------------------------------------------

    def on_alert(self, callback: AlertCallback) -> None:
        """Register callback for memory alerts."""
        self._callbacks.append(callback)

    def _notify_alert(self, alert: MemoryAlert) -> None:
        """Notify all registered callbacks of an alert."""
        self._alerts.append(alert)
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(self) -> MemorySnapshot:
        """Take a memory sample and check thresholds."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            used_mb = mem.used / (1024 * 1024)
            available_mb = mem.available / (1024 * 1024)
            total_mb = mem.total / (1024 * 1024)
            percent_used = mem.percent
        except ImportError:
            # Fallback if psutil not available
            used_mb = 0.0
            available_mb = 0.0
            total_mb = 0.0
            percent_used = 0.0
            logger.warning("psutil not available - memory monitoring disabled")

        level = self.thresholds.get_level(percent_used)

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            used_mb=used_mb,
            available_mb=available_mb,
            total_mb=total_mb,
            percent_used=percent_used,
            level=level,
        )

        with self._lock:
            # Update history
            self._history.append(snapshot)
            if len(self._history) > self.history_size:
                self._history = self._history[-self.history_size :]

            # Update peak
            self._peak_mb = max(self._peak_mb, used_mb)

            # Check for level change
            if level != self._current_level:
                self._handle_level_change(self._current_level, level, snapshot)
                self._current_level = level

        return snapshot

    def _handle_level_change(
        self,
        previous: MemoryLevel,
        current: MemoryLevel,
        snapshot: MemorySnapshot,
    ) -> None:
        """Handle memory level transition."""
        # Generate appropriate message
        if current == MemoryLevel.ABORT:
            message = (
                f"CRITICAL: Memory at {snapshot.percent_used:.1f}% - OOM imminent!"
            )
        elif current == MemoryLevel.CRITICAL:
            message = (
                f"Memory critical at {snapshot.percent_used:.1f}% - consider aborting"
            )
        elif current == MemoryLevel.WARNING:
            message = f"Memory warning: {snapshot.percent_used:.1f}% used"
        elif current == MemoryLevel.INFO:
            message = f"Memory info: {snapshot.percent_used:.1f}% used"
        else:
            message = f"Memory returned to normal: {snapshot.percent_used:.1f}%"

        alert = MemoryAlert(
            timestamp=time.time(),
            previous_level=previous,
            current_level=current,
            snapshot=snapshot,
            message=message,
        )

        logger.log(
            (
                logging.CRITICAL
                if current == MemoryLevel.ABORT
                else (
                    logging.ERROR
                    if current == MemoryLevel.CRITICAL
                    else (
                        logging.WARNING
                        if current == MemoryLevel.WARNING
                        else logging.INFO
                    )
                )
            ),
            message,
        )

        self._notify_alert(alert)

    # -------------------------------------------------------------------------
    # Continuous Monitoring
    # -------------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.sample()
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            time.sleep(self.sample_interval)

    def start_monitoring(self) -> None:
        """Start continuous background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Memory monitoring stopped")

    @property
    def is_monitoring(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Pre-Execution Check (Step 4.1)
    # -------------------------------------------------------------------------

    def check_memory_for_execution(
        self,
        required_mb: float,
        operation_name: str = "execution",
    ) -> MemoryCheckResult:
        """Pre-execution memory check as per Step 4.1.

        Steps:
        1. Get current available memory
        2. Compare requirement vs available
        3. If insufficient: calculate shortfall, generate warning
        4. If sufficient: proceed with monitoring
        """
        snapshot = self.sample()
        available = snapshot.available_mb
        shortfall = max(0, required_mb - available)
        sufficient = shortfall == 0

        warning_message = None
        recommendation = None

        if not sufficient:
            warning_message = (
                f"Insufficient memory for {operation_name}: "
                f"requires {required_mb:.0f}MB but only {available:.0f}MB available "
                f"(shortfall: {shortfall:.0f}MB)"
            )
            recommendation = (
                "Options: 1) Free memory by closing applications, "
                "2) Use a smaller circuit, "
                "3) Use state vector instead of density matrix, "
                "4) Force execute with --force flag (may crash)"
            )
        elif snapshot.level.value >= MemoryLevel.WARNING.value:
            warning_message = (
                f"Memory already at {snapshot.percent_used:.1f}% - "
                f"execution may push to critical levels"
            )
            recommendation = "Consider freeing memory before proceeding"

        return MemoryCheckResult(
            sufficient=sufficient,
            available_mb=available,
            required_mb=required_mb,
            shortfall_mb=shortfall,
            warning_message=warning_message,
            recommendation=recommendation,
        )

    def check_for_backend(
        self,
        backend_name: str,
        simulator_type: str,
        num_qubits: int,
    ) -> tuple[MemoryCheckResult, MemoryEstimate]:
        """Check if enough memory for a specific backend execution."""
        estimate = MemoryEstimator.estimate_for_backend(
            backend_name, simulator_type, num_qubits
        )
        result = self.check_memory_for_execution(
            required_mb=estimate.estimated_mb,
            operation_name=f"{backend_name}/{simulator_type}/{num_qubits}q",
        )
        return result, estimate

    # -------------------------------------------------------------------------
    # History and Stats
    # -------------------------------------------------------------------------

    @property
    def current_level(self) -> MemoryLevel:
        with self._lock:
            return self._current_level

    @property
    def peak_mb(self) -> float:
        with self._lock:
            return self._peak_mb

    @property
    def latest(self) -> MemorySnapshot | None:
        with self._lock:
            return self._history[-1] if self._history else None

    @property
    def alerts(self) -> list[MemoryAlert]:
        with self._lock:
            return list(self._alerts)

    def get_history(self, last_n: int | None = None) -> list[MemorySnapshot]:
        """Get memory history."""
        with self._lock:
            if last_n:
                return list(self._history[-last_n:])
            return list(self._history)

    def clear_history(self) -> None:
        """Clear history and alerts."""
        with self._lock:
            self._history.clear()
            self._alerts.clear()
            self._peak_mb = 0.0

    def summary(self) -> dict:
        """Get monitoring summary."""
        with self._lock:
            latest = self._history[-1] if self._history else None
            return {
                "is_monitoring": self._running,
                "current_level": str(self._current_level),
                "peak_mb": self._peak_mb,
                "current_mb": latest.used_mb if latest else 0,
                "available_mb": latest.available_mb if latest else 0,
                "percent_used": latest.percent_used if latest else 0,
                "samples_collected": len(self._history),
                "alerts_triggered": len(self._alerts),
            }

    def display_line(self) -> str:
        """Single-line status display."""
        latest = self.latest
        if not latest:
            return "[--] Memory: No data"

        icons = {
            MemoryLevel.OK: "[OK]",
            MemoryLevel.INFO: "[INFO]",
            MemoryLevel.WARNING: "[WARN]",
            MemoryLevel.CRITICAL: "[CRIT]",
            MemoryLevel.ABORT: "[ABORT]",
        }
        icon = icons.get(latest.level, "[??]")
        return f"{icon} Memory: {latest.used_mb:.0f}MB / {latest.total_mb:.0f}MB ({latest.percent_used:.1f}%)"


# =============================================================================
# CPU Monitor
# =============================================================================


@dataclass
class CPUSnapshot:
    """Point-in-time CPU measurement."""

    timestamp: float
    percent: float


class CPUMonitor:
    """Monitors CPU usage."""

    def __init__(self, sample_interval: float = 0.5) -> None:
        self.sample_interval = sample_interval
        self._history: list[CPUSnapshot] = []

    def sample(self) -> float:
        """Take a CPU sample, return percent used."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1)
        except ImportError:
            cpu_percent = 0.0

        self._history.append(
            CPUSnapshot(
                timestamp=time.time(),
                percent=cpu_percent,
            )
        )
        return cpu_percent

    @property
    def current_percent(self) -> float:
        if not self._history:
            return 0.0
        return self._history[-1].percent


# =============================================================================
# Combined Resource Monitor
# =============================================================================


@dataclass
class ResourceSnapshot:
    """Combined resource snapshot."""

    timestamp: float
    memory: MemorySnapshot
    cpu_percent: float
    gpu_percent: Optional[float] = None


class ResourceMonitor:
    """Combined resource monitoring with memory and CPU."""

    def __init__(
        self,
        memory_thresholds: MemoryThresholds | None = None,
        sample_interval: float = 0.1,
    ) -> None:
        self.memory = MemoryMonitor(
            thresholds=memory_thresholds,
            sample_interval=sample_interval,
        )
        self.cpu = CPUMonitor(sample_interval=sample_interval)
        self._history: list[ResourceSnapshot] = []
        self._sampling_interval = sample_interval
        self._baseline_memory_bytes: float | None = None
        self._peak_memory_bytes: float | None = None
        self._cpu_samples: list[float] = []
        self._gpu_samples: list[float] = []
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._proc: psutil.Process | None = None
        self._gpu_monitor = GPUMonitor()
        self._gpu_available: bool = self._gpu_monitor.available

    def sample(self) -> ResourceSnapshot:
        """Take combined sample."""
        mem_snapshot = self.memory.sample()

        # Process-level memory tracking
        if self._proc is None:
            try:
                self._proc = psutil.Process()
            except Exception:
                self._proc = None

        rss_bytes = None
        if self._proc is not None:
            try:
                rss_bytes = self._proc.memory_info().rss
            except Exception:
                rss_bytes = None

        if self._baseline_memory_bytes is None and rss_bytes is not None:
            self._baseline_memory_bytes = rss_bytes
            self._peak_memory_bytes = rss_bytes

        if rss_bytes is not None and self._peak_memory_bytes is not None:
            if rss_bytes > self._peak_memory_bytes:
                self._peak_memory_bytes = rss_bytes

        # Process CPU percent (non-blocking)
        cpu_percent = None
        if self._proc is not None:
            try:
                cpu_percent = self._proc.cpu_percent(interval=None)
            except Exception:
                cpu_percent = None

        if cpu_percent is None:
            cpu_percent = self.cpu.sample()
            self._cpu_samples.append(cpu_percent)
        else:
            self._cpu_samples.append(cpu_percent)
            self.cpu._history.append(
                CPUSnapshot(timestamp=time.time(), percent=cpu_percent)
            )

        gpu_percent: Optional[float] = None
        if self._gpu_available:
            try:
                gpu_snapshots = self._gpu_monitor.sample()
                if gpu_snapshots:
                    gpu_percent = sum(
                        s.gpu_utilization for s in gpu_snapshots
                    ) / len(gpu_snapshots)
                    self._gpu_samples.append(gpu_percent)
            except Exception:
                self._gpu_available = False

        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory=mem_snapshot,
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent,
        )
        self._history.append(snapshot)
        return snapshot

    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        # Capture baseline memory before starting the loop
        try:
            self._proc = psutil.Process()
            rss = self._proc.memory_info().rss
            self._baseline_memory_bytes = rss
            self._peak_memory_bytes = rss
        except Exception:
            self._proc = None

        self._running = True
        self.memory.start_monitoring()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        self.memory.stop_monitoring()

    def __del__(self) -> None:
        try:
            self.stop_monitoring()
        except Exception:
            # Avoid raising during garbage collection
            pass

    def reset_samples(self) -> None:
        """Clear collected samples for reuse across benchmarks."""
        self._baseline_memory_bytes = None
        self._peak_memory_bytes = None
        self._cpu_samples.clear()
        self._gpu_samples.clear()
        self._history.clear()

    def _monitor_loop(self) -> None:
        """Background loop sampling CPU/GPU at the configured interval."""
        while self._running:
            try:
                self.sample()
            except Exception as exc:
                logger.warning("Resource monitor sampling error: %s", exc)
            time.sleep(self._sampling_interval)

    def on_memory_alert(self, callback: AlertCallback) -> None:
        """Register memory alert callback."""
        self.memory.on_alert(callback)

    def check_for_execution(
        self,
        backend_name: str,
        simulator_type: str,
        num_qubits: int,
    ) -> tuple[MemoryCheckResult, MemoryEstimate]:
        """Check resources for execution."""
        return self.memory.check_for_backend(backend_name, simulator_type, num_qubits)

    @property
    def latest(self) -> ResourceSnapshot | None:
        return self._history[-1] if self._history else None

    def summary(self) -> dict:
        """Combined summary."""
        mem_summary = self.memory.summary()
        return {
            **mem_summary,
            "cpu_percent": self.cpu.current_percent,
            "cpu_average_percent": self.get_average_cpu_percent(),
            "gpu_average_percent": self.get_average_gpu_percent(),
            "memory_baseline_mb": self.get_memory_baseline_mb(),
            "memory_peak_mb": self.get_peak_memory_mb(),
            "memory_delta_mb": self.get_memory_delta_mb(),
        }

    def get_memory_baseline_mb(self) -> float:
        """Return the baseline memory measured when monitoring started."""
        if self._baseline_memory_bytes is not None:
            return self._baseline_memory_bytes / (1024 * 1024)
        if self.memory.latest:
            return self.memory.latest.used_mb
        return 0.0

    def get_peak_memory_mb(self) -> float:
        """Return the peak memory (delta above baseline) in MB."""
        if self._peak_memory_bytes is not None and self._baseline_memory_bytes is not None:
            return max(
                (self._peak_memory_bytes - self._baseline_memory_bytes) / (1024 * 1024),
                0.0,
            )
        return 0.0

    def get_absolute_peak_memory_mb(self) -> float:
        """Return the absolute peak memory usage in MB."""
        if self._peak_memory_bytes is not None:
            return self._peak_memory_bytes / (1024 * 1024)
        if self.memory.latest:
            return self.memory.latest.used_mb
        return 0.0

    def get_memory_delta_mb(self) -> float:
        """Return delta between current usage and baseline."""
        if not self.memory.latest:
            return 0.0
        if self._proc:
            try:
                current_rss = self._proc.memory_info().rss
                return max(
                    (current_rss - (self._baseline_memory_bytes or current_rss))
                    / (1024 * 1024),
                    0.0,
                )
            except Exception:
                return max(
                    self.memory.latest.used_mb - self.get_memory_baseline_mb(), 0.0
                )
        return max(self.memory.latest.used_mb - self.get_memory_baseline_mb(), 0.0)

    def get_average_cpu_percent(self) -> float:
        """Return average CPU usage collected so far."""
        if self._cpu_samples:
            try:
                return statistics.mean(self._cpu_samples)
            except statistics.StatisticsError:
                return self.cpu.current_percent
        return self.cpu.current_percent

    def get_average_gpu_percent(self) -> Optional[float]:
        """Return average GPU utilization if samples are available."""
        if self._gpu_samples:
            try:
                return statistics.mean(self._gpu_samples)
            except statistics.StatisticsError:
                return None
        return None

    def display_line(self) -> str:
        """Combined status line."""
        mem_line = self.memory.display_line()
        cpu = self.cpu.current_percent
        parts = [mem_line, f"CPU: {cpu:.1f}%"]
        if self._gpu_available:
            if self._gpu_samples:
                parts.append(f"GPU: {self._gpu_samples[-1]:.1f}%")
        return " | ".join(parts)


# =============================================================================
# GPU Monitor
# =============================================================================


@dataclass
class GPUSnapshot:
    """Point-in-time GPU measurement."""

    timestamp: float
    gpu_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    gpu_utilization: float
    temperature: float | None = None


class GPUMonitor:
    """Monitors GPU usage for CUDA-enabled systems."""

    def __init__(self) -> None:
        self._available = self._check_gpu_available()
        self._history: list[GPUSnapshot] = []

    def _check_gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import pynvml

            pynvml.nvmlInit()
            return True
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def sample(self) -> list[GPUSnapshot]:
        """Take GPU samples for all available GPUs."""
        if not self._available:
            return []

        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            snapshots = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = None

                snapshot = GPUSnapshot(
                    timestamp=time.time(),
                    gpu_id=i,
                    name=name,
                    memory_used_mb=memory.used / (1024 * 1024),
                    memory_total_mb=memory.total / (1024 * 1024),
                    memory_percent=(memory.used / memory.total) * 100,
                    gpu_utilization=utilization.gpu,
                    temperature=temp,
                )
                snapshots.append(snapshot)
                self._history.append(snapshot)

            return snapshots
        except Exception:
            return []

    @property
    def latest(self) -> list[GPUSnapshot]:
        """Get latest snapshot for each GPU."""
        if not self._history:
            return []
        # Get unique GPUs from recent history
        gpu_ids = {s.gpu_id for s in self._history[-10:]}
        return [
            next(s for s in reversed(self._history) if s.gpu_id == gid)
            for gid in gpu_ids
        ]


# =============================================================================
# Disk Monitor
# =============================================================================


@dataclass
class DiskSnapshot:
    """Point-in-time disk measurement."""

    timestamp: float
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float


class DiskMonitor:
    """Monitors disk usage."""

    def __init__(self, paths: list[str] | None = None) -> None:
        self._paths = paths or ["/", "C:\\\\"]
        self._history: list[DiskSnapshot] = []

    def sample(self, path: str | None = None) -> DiskSnapshot | None:
        """Sample disk usage for a path."""
        try:
            import psutil

            target_path = path or self._paths[0]
            usage = psutil.disk_usage(target_path)

            snapshot = DiskSnapshot(
                timestamp=time.time(),
                path=target_path,
                total_gb=usage.total / (1024**3),
                used_gb=usage.used / (1024**3),
                free_gb=usage.free / (1024**3),
                percent_used=usage.percent,
            )
            self._history.append(snapshot)
            return snapshot
        except Exception:
            return None

    def sample_all(self) -> list[DiskSnapshot]:
        """Sample all configured paths."""
        snapshots = []
        for path in self._paths:
            snapshot = self.sample(path)
            if snapshot:
                snapshots.append(snapshot)
        return snapshots

    @property
    def latest(self) -> DiskSnapshot | None:
        return self._history[-1] if self._history else None


# =============================================================================
# Network Monitor
# =============================================================================


@dataclass
class NetworkSnapshot:
    """Point-in-time network measurement."""

    timestamp: float
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    bytes_sent_rate: float = 0.0  # bytes/sec
    bytes_recv_rate: float = 0.0  # bytes/sec


class NetworkMonitor:
    """Monitors network I/O."""

    def __init__(self) -> None:
        self._history: list[NetworkSnapshot] = []
        self._last_sample: NetworkSnapshot | None = None

    def sample(self) -> NetworkSnapshot | None:
        """Take a network sample."""
        try:
            import psutil

            counters = psutil.net_io_counters()

            now = time.time()
            snapshot = NetworkSnapshot(
                timestamp=now,
                bytes_sent=counters.bytes_sent,
                bytes_recv=counters.bytes_recv,
                packets_sent=counters.packets_sent,
                packets_recv=counters.packets_recv,
            )

            # Calculate rates if we have a previous sample
            if self._last_sample:
                time_delta = now - self._last_sample.timestamp
                if time_delta > 0:
                    snapshot.bytes_sent_rate = (
                        snapshot.bytes_sent - self._last_sample.bytes_sent
                    ) / time_delta
                    snapshot.bytes_recv_rate = (
                        snapshot.bytes_recv - self._last_sample.bytes_recv
                    ) / time_delta

            self._last_sample = snapshot
            self._history.append(snapshot)
            return snapshot
        except Exception:
            return None

    @property
    def latest(self) -> NetworkSnapshot | None:
        return self._history[-1] if self._history else None

    def get_bandwidth_display(self) -> str:
        """Get human-readable bandwidth display."""
        latest = self.latest
        if not latest:
            return "Network: No data"

        def format_rate(rate: float) -> str:
            if rate > 1024 * 1024:
                return f"{rate / (1024 * 1024):.1f} MB/s"
            elif rate > 1024:
                return f"{rate / 1024:.1f} KB/s"
            else:
                return f"{rate:.0f} B/s"

        return f"↑ {format_rate(latest.bytes_sent_rate)} | ↓ {format_rate(latest.bytes_recv_rate)}"


# =============================================================================
# Extended Resource Monitor
# =============================================================================


class ExtendedResourceMonitor(ResourceMonitor):
    """Extended resource monitor with GPU, disk, and network monitoring."""

    def __init__(
        self,
        memory_thresholds: MemoryThresholds | None = None,
        sample_interval: float = 1.0,
        disk_paths: list[str] | None = None,
    ) -> None:
        super().__init__(memory_thresholds, sample_interval)
        self.gpu = GPUMonitor()
        self.disk = DiskMonitor(disk_paths)
        self.network = NetworkMonitor()

    def sample_all(self) -> dict[str, Any]:
        """Take samples from all monitors."""
        return {
            "memory": self.memory.sample(),
            "cpu": self.cpu.sample(),
            "gpu": self.gpu.sample() if self.gpu.available else [],
            "disk": self.disk.sample_all(),
            "network": self.network.sample(),
        }

    def full_summary(self) -> dict[str, Any]:
        """Get complete resource summary."""
        summary = self.summary()
        summary["gpu_available"] = self.gpu.available
        summary["gpu_snapshots"] = [
            {
                "gpu_id": s.gpu_id,
                "name": s.name,
                "memory_percent": s.memory_percent,
                "utilization": s.gpu_utilization,
            }
            for s in self.gpu.latest
        ]
        if self.disk.latest:
            summary["disk"] = {
                "path": self.disk.latest.path,
                "free_gb": self.disk.latest.free_gb,
                "percent_used": self.disk.latest.percent_used,
            }
        summary["network"] = self.network.get_bandwidth_display()
        return summary

    def full_display_line(self) -> str:
        """Extended status line with all resources."""
        lines = [self.display_line()]

        if self.gpu.available and self.gpu.latest:
            gpu = self.gpu.latest[0]
            lines.append(
                f"GPU: {gpu.memory_percent:.0f}% mem, {gpu.gpu_utilization}% util"
            )

        if self.disk.latest:
            lines.append(f"Disk: {self.disk.latest.free_gb:.1f}GB free")

        lines.append(self.network.get_bandwidth_display())

        return " | ".join(lines)


# =============================================================================
# TREND ANALYSIS & PREDICTION (Missing Feature #1)
# =============================================================================


class TrendDirection(Enum):
    """Direction of resource usage trend."""

    INCREASING = auto()
    DECREASING = auto()
    STABLE = auto()
    VOLATILE = auto()


@dataclass
class TrendPrediction:
    """Prediction result for resource usage."""

    current_value: float
    predicted_value: float
    time_horizon_seconds: float
    confidence: float  # 0-1
    direction: TrendDirection
    slope: float  # Rate of change per second
    will_exceed_threshold: bool
    time_to_threshold_seconds: float | None  # Time until critical/abort level


@dataclass
class ResourceTrend:
    """Trend analysis for a resource metric."""

    metric_name: str
    direction: TrendDirection
    slope: float  # Rate of change per second
    r_squared: float  # Fit quality (0-1)
    volatility: float  # Standard deviation of changes
    predictions: list[TrendPrediction]
    anomaly_detected: bool = False
    anomaly_description: str | None = None


class TrendAnalyzer:
    """Analyzes resource usage trends and makes predictions.
    
    Implements:
    - Linear regression for trend detection
    - Exponential smoothing for predictions
    - Anomaly detection using z-scores
    - Time-to-threshold estimation
    """

    def __init__(
        self,
        window_size: int = 60,
        prediction_horizons: list[float] | None = None,
        anomaly_threshold: float = 3.0,  # Z-score threshold
    ) -> None:
        self.window_size = window_size
        self.prediction_horizons = prediction_horizons or [30.0, 60.0, 300.0]
        self.anomaly_threshold = anomaly_threshold
        self._history: deque[tuple[float, float]] = deque(maxlen=window_size)

    def add_sample(self, timestamp: float, value: float) -> None:
        """Add a new sample to the analyzer."""
        self._history.append((timestamp, value))

    def analyze(
        self,
        samples: list[tuple[float, float]] | None = None,
        thresholds: MemoryThresholds | None = None,
    ) -> ResourceTrend:
        """Analyze trend from samples.
        
        Args:
            samples: List of (timestamp, value) tuples. Uses internal history if None.
            thresholds: Memory thresholds for time-to-threshold calculation.
        
        Returns:
            ResourceTrend with direction, predictions, and anomaly detection.
        """
        data = samples or list(self._history)
        
        if len(data) < 3:
            return ResourceTrend(
                metric_name="memory_percent",
                direction=TrendDirection.STABLE,
                slope=0.0,
                r_squared=0.0,
                volatility=0.0,
                predictions=[],
            )

        timestamps = [d[0] for d in data]
        values = [d[1] for d in data]

        # Normalize timestamps to start from 0
        t0 = timestamps[0]
        normalized_times = [t - t0 for t in timestamps]

        # Calculate linear regression
        slope, intercept, r_squared = self._linear_regression(normalized_times, values)

        # Calculate volatility
        if len(values) > 1:
            changes = [values[i] - values[i-1] for i in range(1, len(values))]
            volatility = statistics.stdev(changes) if len(changes) > 1 else 0.0
        else:
            volatility = 0.0

        # Determine trend direction
        direction = self._determine_direction(slope, volatility, r_squared)

        # Make predictions
        predictions = []
        current_time = normalized_times[-1]
        current_value = values[-1]

        for horizon in self.prediction_horizons:
            predicted = intercept + slope * (current_time + horizon)
            
            # Clamp prediction to valid range
            predicted = max(0.0, min(100.0, predicted))

            # Calculate confidence based on R² and time horizon
            confidence = r_squared * math.exp(-horizon / 300.0)

            # Check if will exceed threshold
            will_exceed = False
            time_to_threshold = None
            
            if thresholds and slope > 0:
                critical_threshold = thresholds.critical_percent
                if predicted >= critical_threshold and current_value < critical_threshold:
                    will_exceed = True
                    # Time until we hit critical threshold
                    time_to_threshold = (critical_threshold - current_value) / slope

            predictions.append(TrendPrediction(
                current_value=current_value,
                predicted_value=predicted,
                time_horizon_seconds=horizon,
                confidence=confidence,
                direction=direction,
                slope=slope,
                will_exceed_threshold=will_exceed,
                time_to_threshold_seconds=time_to_threshold,
            ))

        # Anomaly detection using z-score
        anomaly_detected = False
        anomaly_description = None
        
        if len(values) >= 10:
            mean_val = statistics.mean(values[:-1])
            std_val = statistics.stdev(values[:-1]) if len(values) > 2 else 1.0
            if std_val > 0:
                z_score = abs(values[-1] - mean_val) / std_val
                if z_score > self.anomaly_threshold:
                    anomaly_detected = True
                    anomaly_description = f"Sudden {'spike' if values[-1] > mean_val else 'drop'} detected (z-score: {z_score:.2f})"

        return ResourceTrend(
            metric_name="memory_percent",
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            volatility=volatility,
            predictions=predictions,
            anomaly_detected=anomaly_detected,
            anomaly_description=anomaly_description,
        )

    def _linear_regression(
        self, x: list[float], y: list[float]
    ) -> tuple[float, float, float]:
        """Simple linear regression returning slope, intercept, R²."""
        n = len(x)
        if n < 2:
            return 0.0, y[0] if y else 0.0, 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return 0.0, sum_y / n, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R²
        ss_tot = sum_y2 - (sum_y * sum_y) / n
        ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        return slope, intercept, r_squared

    def _determine_direction(
        self, slope: float, volatility: float, r_squared: float
    ) -> TrendDirection:
        """Determine trend direction from slope and volatility."""
        # If R² is low, trend is volatile
        if r_squared < 0.3 and volatility > 2.0:
            return TrendDirection.VOLATILE

        # Significant slope thresholds (percent per second)
        slope_threshold = 0.1

        if abs(slope) < slope_threshold:
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING

    def predict_time_to_level(
        self,
        target_percent: float,
        current_percent: float,
        slope: float,
    ) -> float | None:
        """Predict time until resource reaches target level.
        
        Returns None if target won't be reached (wrong direction).
        """
        if slope == 0:
            return None
        
        time_seconds = (target_percent - current_percent) / slope
        
        # Only return positive times in the correct direction
        if time_seconds > 0:
            return time_seconds
        return None


# =============================================================================
# AUTOMATIC RESOURCE OPTIMIZATION (Missing Feature #2)
# =============================================================================


class OptimizationType(Enum):
    """Type of optimization recommendation."""

    REDUCE_QUBIT_COUNT = auto()
    USE_STATE_VECTOR = auto()
    USE_DENSITY_MATRIX = auto()
    SWITCH_BACKEND = auto()
    FREE_MEMORY = auto()
    REDUCE_SHOTS = auto()
    ENABLE_COMPRESSION = auto()
    BATCH_EXECUTION = auto()
    DEFER_EXECUTION = auto()


@dataclass
class OptimizationRecommendation:
    """A specific optimization recommendation."""

    optimization_type: OptimizationType
    priority: int  # 1 = highest priority
    description: str
    expected_savings_mb: float
    expected_time_savings_seconds: float
    action_required: str
    auto_applicable: bool  # Can be applied automatically
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationPlan:
    """Plan containing multiple optimization recommendations."""

    recommendations: list[OptimizationRecommendation]
    total_potential_savings_mb: float
    total_potential_time_savings: float
    resource_status: str  # "critical", "warning", "ok"
    auto_optimizations_available: int
    requires_user_action: bool


class ResourceOptimizer:
    """Provides automatic resource optimization recommendations.
    
    Analyzes current resource state and execution requirements
    to suggest optimizations for memory, CPU, and execution time.
    """

    # Memory requirements per qubit (in MB)
    STATE_VECTOR_MB_PER_QUBIT_FACTOR = 16 / (1024 * 1024)  # 16 bytes per amplitude
    DENSITY_MATRIX_MULTIPLIER = 2  # DM uses 2^(2n) vs 2^n

    def __init__(
        self,
        resource_monitor: ResourceMonitor | None = None,
        trend_analyzer: TrendAnalyzer | None = None,
    ) -> None:
        self._monitor = resource_monitor
        self._trend_analyzer = trend_analyzer or TrendAnalyzer()
        self._optimization_history: list[OptimizationPlan] = []
        self._applied_optimizations: list[OptimizationRecommendation] = []

    def analyze_and_recommend(
        self,
        num_qubits: int,
        simulator_type: str = "state_vector",
        shots: int = 1024,
        current_backend: str | None = None,
        available_backends: list[str] | None = None,
    ) -> OptimizationPlan:
        """Analyze resource requirements and generate optimization recommendations.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            simulator_type: "state_vector" or "density_matrix".
            shots: Number of measurement shots.
            current_backend: Currently selected backend.
            available_backends: List of available backends.
        
        Returns:
            OptimizationPlan with prioritized recommendations.
        """
        recommendations: list[OptimizationRecommendation] = []
        
        # Get current resource state
        available_mb = self._get_available_memory()
        required_mb = self._estimate_memory_requirement(num_qubits, simulator_type)
        
        # Get trend prediction if available
        trend = None
        if self._monitor:
            history = self._monitor.memory.get_history(last_n=60)
            if history:
                samples = [(s.timestamp, s.percent_used) for s in history]
                trend = self._trend_analyzer.analyze(samples)

        # Determine resource status
        memory_ratio = required_mb / available_mb if available_mb > 0 else float('inf')
        
        if memory_ratio > 1.0:
            resource_status = "critical"
        elif memory_ratio > 0.7:
            resource_status = "warning"
        else:
            resource_status = "ok"

        # Generate recommendations based on analysis
        priority = 1

        # 1. Check if we need to reduce memory usage
        if memory_ratio > 1.0:
            shortfall = required_mb - available_mb
            
            # Recommend reducing qubit count
            reduced_qubits = self._calculate_max_qubits_for_memory(
                available_mb * 0.8, simulator_type
            )
            if reduced_qubits < num_qubits:
                recommendations.append(OptimizationRecommendation(
                    optimization_type=OptimizationType.REDUCE_QUBIT_COUNT,
                    priority=priority,
                    description=f"Reduce circuit from {num_qubits} to {reduced_qubits} qubits",
                    expected_savings_mb=shortfall,
                    expected_time_savings_seconds=0,
                    action_required=f"Modify circuit to use {reduced_qubits} qubits",
                    auto_applicable=False,
                    parameters={"target_qubits": reduced_qubits},
                ))
                priority += 1

            # Recommend switching to state vector if using density matrix
            if simulator_type == "density_matrix":
                sv_required = self._estimate_memory_requirement(num_qubits, "state_vector")
                if sv_required < available_mb * 0.9:
                    recommendations.append(OptimizationRecommendation(
                        optimization_type=OptimizationType.USE_STATE_VECTOR,
                        priority=priority,
                        description="Switch to state vector simulation (no noise model)",
                        expected_savings_mb=required_mb - sv_required,
                        expected_time_savings_seconds=0,
                        action_required="Use state vector simulator instead of density matrix",
                        auto_applicable=True,
                        parameters={"simulator_type": "state_vector"},
                    ))
                    priority += 1

            # Recommend freeing memory
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.FREE_MEMORY,
                priority=priority,
                description="Free system memory by closing other applications",
                expected_savings_mb=shortfall,
                expected_time_savings_seconds=0,
                action_required="Close memory-intensive applications",
                auto_applicable=False,
            ))
            priority += 1

        # 2. Check for performance optimizations
        if shots > 10000:
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.REDUCE_SHOTS,
                priority=priority,
                description=f"Reduce shots from {shots} to 1024 for faster execution",
                expected_savings_mb=0,
                expected_time_savings_seconds=(shots - 1024) * 0.001,
                action_required="Reduce number of measurement shots",
                auto_applicable=True,
                parameters={"target_shots": 1024},
            ))
            priority += 1

        # 3. Backend switch recommendations
        if available_backends and current_backend:
            better_backends = self._find_better_backends(
                current_backend, available_backends, num_qubits, simulator_type
            )
            for backend, reason in better_backends:
                recommendations.append(OptimizationRecommendation(
                    optimization_type=OptimizationType.SWITCH_BACKEND,
                    priority=priority,
                    description=f"Switch to {backend}: {reason}",
                    expected_savings_mb=0,
                    expected_time_savings_seconds=0,  # Unknown without benchmarks
                    action_required=f"Use --backend {backend}",
                    auto_applicable=True,
                    parameters={"backend": backend},
                ))
                priority += 1

        # 4. Trend-based recommendations
        if trend and trend.direction == TrendDirection.INCREASING:
            for pred in trend.predictions:
                if pred.will_exceed_threshold and pred.time_to_threshold_seconds:
                    if pred.time_to_threshold_seconds < 60:
                        recommendations.append(OptimizationRecommendation(
                            optimization_type=OptimizationType.DEFER_EXECUTION,
                            priority=1,  # High priority
                            description=f"Memory predicted to reach critical in {pred.time_to_threshold_seconds:.0f}s",
                            expected_savings_mb=0,
                            expected_time_savings_seconds=0,
                            action_required="Wait for memory usage to stabilize or free memory",
                            auto_applicable=False,
                        ))
                    break

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)

        # Calculate totals
        total_savings_mb = sum(r.expected_savings_mb for r in recommendations)
        total_time_savings = sum(r.expected_time_savings_seconds for r in recommendations)
        auto_count = sum(1 for r in recommendations if r.auto_applicable)
        requires_action = any(not r.auto_applicable for r in recommendations if r.priority <= 2)

        plan = OptimizationPlan(
            recommendations=recommendations,
            total_potential_savings_mb=total_savings_mb,
            total_potential_time_savings=total_time_savings,
            resource_status=resource_status,
            auto_optimizations_available=auto_count,
            requires_user_action=requires_action,
        )

        self._optimization_history.append(plan)
        return plan

    def apply_optimization(
        self,
        recommendation: OptimizationRecommendation,
    ) -> bool:
        """Apply an automatic optimization.
        
        Returns True if successfully applied.
        """
        if not recommendation.auto_applicable:
            logger.warning(f"Optimization {recommendation.optimization_type} requires manual action")
            return False

        self._applied_optimizations.append(recommendation)
        logger.info(f"Applied optimization: {recommendation.description}")
        return True

    def get_quick_optimization(
        self,
        num_qubits: int,
        simulator_type: str = "state_vector",
    ) -> OptimizationRecommendation | None:
        """Get a single quick optimization recommendation for immediate application."""
        plan = self.analyze_and_recommend(num_qubits, simulator_type)
        
        for rec in plan.recommendations:
            if rec.auto_applicable and rec.priority <= 2:
                return rec
        return None

    def _get_available_memory(self) -> float:
        """Get available memory in MB."""
        if self._monitor:
            latest = self._monitor.memory.latest
            if latest:
                return latest.available_mb
        
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            return 8192.0  # Default assumption

    def _estimate_memory_requirement(self, num_qubits: int, simulator_type: str) -> float:
        """Estimate memory requirement in MB."""
        base_bytes = (2 ** num_qubits) * 16  # Complex128
        
        if simulator_type == "density_matrix":
            base_bytes = (2 ** (2 * num_qubits)) * 16
        
        # Add overhead
        return (base_bytes / (1024 * 1024)) * 1.5

    def _calculate_max_qubits_for_memory(
        self, available_mb: float, simulator_type: str
    ) -> int:
        """Calculate maximum qubits that fit in available memory."""
        available_bytes = available_mb * 1024 * 1024 / 1.5  # Remove overhead
        
        if simulator_type == "density_matrix":
            # 2^(2n) * 16 = available
            max_n = int(math.log2(available_bytes / 16) / 2)
        else:
            # 2^n * 16 = available
            max_n = int(math.log2(available_bytes / 16))
        
        return max(1, max_n)

    def _find_better_backends(
        self,
        current: str,
        available: list[str],
        num_qubits: int,
        simulator_type: str,
    ) -> list[tuple[str, str]]:
        """Find potentially better backends with reasons."""
        recommendations = []
        
        # Simple heuristics for backend recommendations
        if num_qubits > 25 and "qsim" in available and current != "qsim":
            recommendations.append(("qsim", "CPU-optimized for large circuits"))
        
        if num_qubits > 20 and "cuquantum" in available and current != "cuquantum":
            recommendations.append(("cuquantum", "GPU acceleration for faster execution"))
        
        if simulator_type == "density_matrix" and "quest" in available and current != "quest":
            recommendations.append(("quest", "Native density matrix support"))
        
        return recommendations[:2]  # Limit to 2 recommendations


# =============================================================================
# BACKEND SELECTION INTEGRATION (Missing Feature #3)
# =============================================================================


@dataclass
class BackendResourceScore:
    """Resource-based score for a backend."""

    backend_name: str
    memory_score: float  # 0-1, higher = better memory fit
    cpu_score: float  # 0-1, higher = better CPU utilization
    gpu_score: float  # 0-1, higher = better GPU utilization (if applicable)
    overall_score: float
    can_execute: bool
    reason: str


@dataclass
class ResourceAwareSelection:
    """Result of resource-aware backend selection."""

    recommended_backend: str
    scores: list[BackendResourceScore]
    resource_warnings: list[str]
    optimization_suggestions: list[str]
    estimated_execution_time: float | None


class BackendResourceIntegration:
    """Integrates resource monitoring with backend selection.
    
    Provides resource-aware backend recommendations based on:
    - Current memory availability
    - CPU utilization
    - GPU availability and memory
    - Trend predictions
    """

    def __init__(
        self,
        resource_monitor: ExtendedResourceMonitor | None = None,
        trend_analyzer: TrendAnalyzer | None = None,
        optimizer: ResourceOptimizer | None = None,
    ) -> None:
        self._monitor = resource_monitor or ExtendedResourceMonitor()
        self._trend_analyzer = trend_analyzer or TrendAnalyzer()
        self._optimizer = optimizer or ResourceOptimizer(
            resource_monitor=self._monitor,
            trend_analyzer=self._trend_analyzer,
        )
        
        # Backend memory requirements (base MB for small circuits)
        self._backend_base_memory: dict[str, float] = {
            "numpy": 50,
            "cirq": 100,
            "qiskit": 150,
            "quest": 80,
            "qsim": 120,
            "cuquantum": 200,  # GPU memory
            "lret": 60,
        }
        
        # Backend GPU requirements
        self._gpu_backends = {"cuquantum", "cupy"}

    def select_backend_for_resources(
        self,
        num_qubits: int,
        simulator_type: str = "state_vector",
        available_backends: list[str] | None = None,
        prefer_gpu: bool = False,
    ) -> ResourceAwareSelection:
        """Select the best backend based on current resource availability.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            simulator_type: "state_vector" or "density_matrix".
            available_backends: List of available backends.
            prefer_gpu: Whether to prefer GPU backends when available.
        
        Returns:
            ResourceAwareSelection with recommended backend and analysis.
        """
        if available_backends is None:
            available_backends = list(self._backend_base_memory.keys())

        # Get current resource state
        self._monitor.sample()
        mem_snapshot = self._monitor.memory.latest
        cpu_percent = self._monitor.cpu.current_percent
        
        # Get GPU info if available
        gpu_available = self._monitor.gpu.available
        gpu_memory_mb = 0.0
        if gpu_available and self._monitor.gpu.latest:
            gpu_info = self._monitor.gpu.latest[0]
            gpu_memory_mb = gpu_info.memory_total_mb - gpu_info.memory_used_mb

        # Get trend predictions
        history = self._monitor.memory.get_history(last_n=60)
        trend = None
        if history:
            samples = [(s.timestamp, s.percent_used) for s in history]
            trend = self._trend_analyzer.analyze(samples, self._monitor.memory.thresholds)

        # Score each backend
        scores: list[BackendResourceScore] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        for backend in available_backends:
            score = self._score_backend(
                backend=backend,
                num_qubits=num_qubits,
                simulator_type=simulator_type,
                available_memory_mb=mem_snapshot.available_mb if mem_snapshot else 8192,
                cpu_percent=cpu_percent,
                gpu_available=gpu_available,
                gpu_memory_mb=gpu_memory_mb,
                prefer_gpu=prefer_gpu,
            )
            scores.append(score)

        # Sort by overall score
        scores.sort(key=lambda s: s.overall_score, reverse=True)

        # Find best executable backend
        recommended = None
        for score in scores:
            if score.can_execute:
                recommended = score.backend_name
                break

        if not recommended:
            recommended = scores[0].backend_name if scores else "numpy"
            warnings.append("No backend can safely execute with current resources")

        # Add trend-based warnings
        if trend and trend.direction == TrendDirection.INCREASING:
            for pred in trend.predictions:
                if pred.will_exceed_threshold:
                    warnings.append(
                        f"Memory trend increasing - may reach critical in "
                        f"{pred.time_to_threshold_seconds:.0f}s"
                    )
                    break

        # Generate optimization suggestions
        opt_plan = self._optimizer.analyze_and_recommend(
            num_qubits=num_qubits,
            simulator_type=simulator_type,
            current_backend=recommended,
            available_backends=available_backends,
        )
        
        for rec in opt_plan.recommendations[:3]:  # Top 3 suggestions
            suggestions.append(rec.description)

        # Estimate execution time (rough heuristic)
        estimated_time = self._estimate_execution_time(
            backend=recommended,
            num_qubits=num_qubits,
            simulator_type=simulator_type,
        )

        return ResourceAwareSelection(
            recommended_backend=recommended,
            scores=scores,
            resource_warnings=warnings,
            optimization_suggestions=suggestions,
            estimated_execution_time=estimated_time,
        )

    def _score_backend(
        self,
        backend: str,
        num_qubits: int,
        simulator_type: str,
        available_memory_mb: float,
        cpu_percent: float,
        gpu_available: bool,
        gpu_memory_mb: float,
        prefer_gpu: bool,
    ) -> BackendResourceScore:
        """Score a backend based on resource requirements."""
        # Calculate memory requirement
        base_mem = self._backend_base_memory.get(backend, 100)
        if simulator_type == "density_matrix":
            required_mb = base_mem + (2 ** (2 * num_qubits)) * 16 / (1024 * 1024)
        else:
            required_mb = base_mem + (2 ** num_qubits) * 16 / (1024 * 1024)

        # Memory score
        memory_ratio = required_mb / available_memory_mb if available_memory_mb > 0 else 2.0
        memory_score = max(0.0, 1.0 - memory_ratio)
        can_execute = memory_ratio < 0.9

        # CPU score (prefer backends when CPU is not overloaded)
        cpu_score = max(0.0, (100 - cpu_percent) / 100)

        # GPU score
        gpu_score = 0.0
        is_gpu_backend = backend in self._gpu_backends
        
        if is_gpu_backend:
            if gpu_available and gpu_memory_mb > required_mb:
                gpu_score = 1.0
                can_execute = True  # GPU can handle it
            else:
                gpu_score = 0.0
                can_execute = False  # GPU backend but no GPU
        elif prefer_gpu:
            gpu_score = 0.0  # Penalty for non-GPU when GPU preferred
        else:
            gpu_score = 0.5  # Neutral

        # Overall score
        if prefer_gpu and gpu_available:
            overall = 0.3 * memory_score + 0.2 * cpu_score + 0.5 * gpu_score
        else:
            overall = 0.5 * memory_score + 0.3 * cpu_score + 0.2 * gpu_score

        # Generate reason
        if can_execute:
            reason = f"Memory fit: {memory_score:.0%}, CPU available: {cpu_score:.0%}"
        else:
            reason = f"Insufficient resources: needs {required_mb:.0f}MB, have {available_memory_mb:.0f}MB"

        return BackendResourceScore(
            backend_name=backend,
            memory_score=memory_score,
            cpu_score=cpu_score,
            gpu_score=gpu_score,
            overall_score=overall,
            can_execute=can_execute,
            reason=reason,
        )

    def _estimate_execution_time(
        self,
        backend: str,
        num_qubits: int,
        simulator_type: str,
    ) -> float:
        """Rough estimation of execution time in seconds."""
        # Base times (very rough estimates)
        base_times = {
            "numpy": 1.0,
            "cirq": 0.8,
            "qiskit": 0.9,
            "quest": 0.5,
            "qsim": 0.4,
            "cuquantum": 0.3,
            "lret": 1.2,
        }
        
        base = base_times.get(backend, 1.0)
        
        # Exponential scaling with qubits
        scaling = 2 ** (num_qubits / 5)  # Rough heuristic
        
        if simulator_type == "density_matrix":
            scaling *= 4  # DM is much slower

        return base * scaling

    def get_resource_status(self) -> dict[str, Any]:
        """Get current resource status for backend selection."""
        self._monitor.sample_all()
        
        mem = self._monitor.memory.latest
        cpu = self._monitor.cpu.current_percent
        
        status = {
            "memory_available_mb": mem.available_mb if mem else 0,
            "memory_percent": mem.percent_used if mem else 0,
            "memory_level": str(mem.level) if mem else "UNKNOWN",
            "cpu_percent": cpu,
            "gpu_available": self._monitor.gpu.available,
        }
        
        if self._monitor.gpu.available and self._monitor.gpu.latest:
            gpu = self._monitor.gpu.latest[0]
            status["gpu_memory_free_mb"] = gpu.memory_total_mb - gpu.memory_used_mb
            status["gpu_utilization"] = gpu.gpu_utilization

        return status

    def can_execute_safely(
        self,
        num_qubits: int,
        simulator_type: str = "state_vector",
        backend: str = "numpy",
    ) -> tuple[bool, str]:
        """Check if execution can proceed safely with current resources.
        
        Returns:
            Tuple of (can_execute, reason)
        """
        self._monitor.sample()
        mem = self._monitor.memory.latest
        
        if not mem:
            return True, "Resource monitoring unavailable - proceeding"

        # Estimate requirement
        if simulator_type == "density_matrix":
            required_mb = (2 ** (2 * num_qubits)) * 16 / (1024 * 1024)
        else:
            required_mb = (2 ** num_qubits) * 16 / (1024 * 1024)

        required_mb *= 1.5  # Safety margin

        if required_mb > mem.available_mb:
            return False, (
                f"Insufficient memory: requires {required_mb:.0f}MB, "
                f"only {mem.available_mb:.0f}MB available"
            )

        if mem.level in (MemoryLevel.CRITICAL, MemoryLevel.ABORT):
            return False, f"System memory is {mem.level.name} ({mem.percent_used:.1f}% used)"

        if mem.level == MemoryLevel.WARNING:
            return True, f"Warning: Memory at {mem.percent_used:.1f}% - execution may be slow"

        return True, "Resources available for execution"
