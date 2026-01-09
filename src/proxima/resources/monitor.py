"""Enhanced Memory Monitoring implementation (Phase 4, Step 4.1).

Provides:
- MemoryLevel: Threshold levels (INFO, WARNING, CRITICAL, ABORT)
- MemoryMonitor: Track memory with threshold alerts and callbacks
- MemoryEstimator: Estimate memory requirements before execution
- ResourceMonitor: Combined CPU and memory monitoring
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

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
            message = f"CRITICAL: Memory at {snapshot.percent_used:.1f}% - OOM imminent!"
        elif current == MemoryLevel.CRITICAL:
            message = f"Memory critical at {snapshot.percent_used:.1f}% - consider aborting"
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
                    else logging.WARNING if current == MemoryLevel.WARNING else logging.INFO
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
        estimate = MemoryEstimator.estimate_for_backend(backend_name, simulator_type, num_qubits)
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


class ResourceMonitor:
    """Combined resource monitoring with memory and CPU."""

    def __init__(
        self,
        memory_thresholds: MemoryThresholds | None = None,
        sample_interval: float = 1.0,
    ) -> None:
        self.memory = MemoryMonitor(
            thresholds=memory_thresholds,
            sample_interval=sample_interval,
        )
        self.cpu = CPUMonitor(sample_interval=sample_interval)
        self._history: list[ResourceSnapshot] = []

    def sample(self) -> ResourceSnapshot:
        """Take combined sample."""
        mem_snapshot = self.memory.sample()
        cpu_percent = self.cpu.sample()

        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory=mem_snapshot,
            cpu_percent=cpu_percent,
        )
        self._history.append(snapshot)
        return snapshot

    def start_monitoring(self) -> None:
        """Start background monitoring."""
        self.memory.start_monitoring()

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.memory.stop_monitoring()

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
        }

    def display_line(self) -> str:
        """Combined status line."""
        mem_line = self.memory.display_line()
        cpu = self.cpu.current_percent
        return f"{mem_line} | CPU: {cpu:.1f}%"
