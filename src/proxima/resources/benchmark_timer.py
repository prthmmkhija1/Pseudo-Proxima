"""High-precision timer utilities for benchmarking.

Provides pause/resume support, checkpoints, and context manager usage for
accurate measurement of benchmarked code sections.
"""

from __future__ import annotations

import time
from typing import Dict


class BenchmarkTimer:
    """High-resolution timer with pause/resume and checkpoints."""

    __slots__ = (
        "_start_time",
        "_stop_time",
        "_is_running",
        "_paused_duration",
        "_pause_start",
        "_checkpoints",
    )

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._is_running: bool = False
        self._paused_duration: float = 0.0
        self._pause_start: float | None = None
        self._checkpoints: Dict[str, float] = {}

    def start(self) -> None:
        """Start the timer."""
        if self._is_running:
            raise RuntimeError("Timer is already running")
        self._start_time = time.perf_counter()
        self._stop_time = None
        self._paused_duration = 0.0
        self._pause_start = None
        self._is_running = True

    def stop(self) -> None:
        """Stop the timer."""
        if not self._is_running:
            raise RuntimeError("Timer is not running")
        self._stop_time = time.perf_counter()
        self._is_running = False

    def pause(self) -> None:
        """Pause the timer without resetting elapsed time."""
        if not self._is_running:
            raise RuntimeError("Cannot pause a stopped timer")
        if self._pause_start is not None:
            return
        self._pause_start = time.perf_counter()

    def resume(self) -> None:
        """Resume a paused timer."""
        if self._pause_start is None:
            return
        pause_time = time.perf_counter() - self._pause_start
        self._paused_duration += pause_time
        self._pause_start = None

    def elapsed_ms(self) -> float:
        """Return the elapsed time in milliseconds."""
        if self._start_time is None:
            return 0.0
        end_time = self._stop_time if self._stop_time is not None else time.perf_counter()
        effective_elapsed = end_time - self._start_time - self._paused_duration
        return max(effective_elapsed, 0.0) * 1000.0

    def reset(self) -> None:
        """Reset the timer to its initial state."""
        self._start_time = None
        self._stop_time = None
        self._is_running = False
        self._paused_duration = 0.0
        self._pause_start = None
        self._checkpoints.clear()

    def checkpoint(self, name: str) -> None:
        """Record a named checkpoint with the current elapsed time."""
        self._checkpoints[name] = self.elapsed_ms()

    def get_checkpoint_duration(self, name: str) -> float:
        """Return time elapsed since a checkpoint was recorded."""
        if name not in self._checkpoints:
            raise KeyError(f"Checkpoint '{name}' not found")
        return max(self.elapsed_ms() - self._checkpoints[name], 0.0)

    def list_checkpoints(self) -> Dict[str, float]:
        """Return all checkpoints with their recorded elapsed times."""
        return dict(self._checkpoints)

    @property
    def is_running(self) -> bool:
        """Return True if timer is currently running."""
        return self._is_running

    def elapsed_seconds(self) -> float:
        """Return the elapsed time in seconds."""
        return self.elapsed_ms() / 1000.0

    def __enter__(self) -> BenchmarkTimer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# =============================================================================
# Timer Statistics
# =============================================================================


class TimerStatistics:
    """Statistical analysis for timer measurements.

    Collects multiple timing samples and computes statistical measures.
    """

    def __init__(self, name: str = "default") -> None:
        """Initialize timer statistics.

        Args:
            name: Name for this timing collection.
        """
        self._name = name
        self._samples: list[float] = []

    @property
    def name(self) -> str:
        """Get the name of this statistics collection."""
        return self._name

    def add_sample(self, elapsed_ms: float) -> None:
        """Add a timing sample in milliseconds.

        Args:
            elapsed_ms: Elapsed time in milliseconds.
        """
        self._samples.append(elapsed_ms)

    def add_from_timer(self, timer: BenchmarkTimer) -> None:
        """Add sample from a stopped timer.

        Args:
            timer: A BenchmarkTimer that has been stopped.
        """
        self.add_sample(timer.elapsed_ms())

    @property
    def count(self) -> int:
        """Get number of samples."""
        return len(self._samples)

    @property
    def samples(self) -> list[float]:
        """Get all samples."""
        return list(self._samples)

    def clear(self) -> None:
        """Clear all samples."""
        self._samples.clear()

    def mean(self) -> float:
        """Calculate mean elapsed time."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def median(self) -> float:
        """Calculate median elapsed time."""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_samples[mid - 1] + sorted_samples[mid]) / 2
        return sorted_samples[mid]

    def min(self) -> float:
        """Get minimum elapsed time."""
        return min(self._samples) if self._samples else 0.0

    def max(self) -> float:
        """Get maximum elapsed time."""
        return max(self._samples) if self._samples else 0.0

    def stdev(self) -> float:
        """Calculate standard deviation."""
        if len(self._samples) < 2:
            return 0.0
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self._samples) / (len(self._samples) - 1)
        return variance ** 0.5

    def percentile(self, p: float) -> float:
        """Calculate percentile value.

        Args:
            p: Percentile (0-100).

        Returns:
            Value at the given percentile.
        """
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)
        k = (n - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, n - 1)
        return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])

    def p50(self) -> float:
        """Get 50th percentile (median)."""
        return self.percentile(50)

    def p90(self) -> float:
        """Get 90th percentile."""
        return self.percentile(90)

    def p95(self) -> float:
        """Get 95th percentile."""
        return self.percentile(95)

    def p99(self) -> float:
        """Get 99th percentile."""
        return self.percentile(99)

    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (CV).

        Returns ratio of standard deviation to mean, useful for
        comparing variability across different timing scales.
        """
        mean = self.mean()
        if mean == 0:
            return 0.0
        return (self.stdev() / mean) * 100

    def remove_outliers(self, z_threshold: float = 2.0) -> int:
        """Remove outliers using Z-score method.

        Args:
            z_threshold: Z-score threshold for outlier detection.

        Returns:
            Number of outliers removed.
        """
        if len(self._samples) < 3:
            return 0

        mean = self.mean()
        stdev = self.stdev()
        if stdev == 0:
            return 0

        original_count = len(self._samples)
        self._samples = [
            x for x in self._samples
            if abs(x - mean) / stdev <= z_threshold
        ]
        return original_count - len(self._samples)

    def to_dict(self) -> Dict[str, float]:
        """Convert statistics to dictionary.

        Returns:
            Dictionary with all statistical measures.
        """
        return {
            "name": self._name,
            "count": self.count,
            "mean_ms": self.mean(),
            "median_ms": self.median(),
            "min_ms": self.min(),
            "max_ms": self.max(),
            "stdev_ms": self.stdev(),
            "p50_ms": self.p50(),
            "p90_ms": self.p90(),
            "p95_ms": self.p95(),
            "p99_ms": self.p99(),
            "cv_percent": self.coefficient_of_variation(),
        }

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Formatted summary string.
        """
        return (
            f"Timer '{self._name}': {self.count} samples\n"
            f"  Mean: {self.mean():.2f} ms\n"
            f"  Median: {self.median():.2f} ms\n"
            f"  Min: {self.min():.2f} ms, Max: {self.max():.2f} ms\n"
            f"  Stdev: {self.stdev():.2f} ms (CV: {self.coefficient_of_variation():.1f}%)\n"
            f"  P95: {self.p95():.2f} ms, P99: {self.p99():.2f} ms"
        )


# =============================================================================
# Lap Timer
# =============================================================================


class LapTimer:
    """Timer with lap (split) time support.

    Records intermediate lap times while maintaining overall elapsed time.
    Useful for benchmarking multi-phase operations.
    """

    def __init__(self) -> None:
        """Initialize lap timer."""
        self._timer = BenchmarkTimer()
        self._laps: list[tuple[str, float, float]] = []  # (name, lap_time, cumulative)
        self._last_lap_time: float = 0.0

    def start(self) -> None:
        """Start the lap timer."""
        self._timer.start()
        self._laps.clear()
        self._last_lap_time = 0.0

    def lap(self, name: str = "") -> float:
        """Record a lap time.

        Args:
            name: Optional name for the lap.

        Returns:
            Time since last lap (or start) in milliseconds.
        """
        current_time = self._timer.elapsed_ms()
        lap_time = current_time - self._last_lap_time
        lap_name = name or f"lap_{len(self._laps) + 1}"
        self._laps.append((lap_name, lap_time, current_time))
        self._last_lap_time = current_time
        return lap_time

    def stop(self) -> float:
        """Stop the timer and return total elapsed time.

        Returns:
            Total elapsed time in milliseconds.
        """
        self._timer.stop()
        return self._timer.elapsed_ms()

    @property
    def laps(self) -> list[tuple[str, float, float]]:
        """Get all laps as (name, lap_time_ms, cumulative_time_ms) tuples."""
        return list(self._laps)

    def get_lap(self, name: str) -> tuple[float, float] | None:
        """Get a specific lap by name.

        Args:
            name: Lap name.

        Returns:
            Tuple of (lap_time_ms, cumulative_time_ms) or None if not found.
        """
        for lap_name, lap_time, cumulative in self._laps:
            if lap_name == name:
                return lap_time, cumulative
        return None

    def elapsed_ms(self) -> float:
        """Get current elapsed time in milliseconds."""
        return self._timer.elapsed_ms()

    def lap_count(self) -> int:
        """Get number of recorded laps."""
        return len(self._laps)

    def fastest_lap(self) -> tuple[str, float] | None:
        """Get the fastest lap.

        Returns:
            Tuple of (name, time_ms) for fastest lap.
        """
        if not self._laps:
            return None
        fastest = min(self._laps, key=lambda x: x[1])
        return fastest[0], fastest[1]

    def slowest_lap(self) -> tuple[str, float] | None:
        """Get the slowest lap.

        Returns:
            Tuple of (name, time_ms) for slowest lap.
        """
        if not self._laps:
            return None
        slowest = max(self._laps, key=lambda x: x[1])
        return slowest[0], slowest[1]

    def average_lap_time(self) -> float:
        """Get average lap time in milliseconds."""
        if not self._laps:
            return 0.0
        return sum(lap[1] for lap in self._laps) / len(self._laps)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary.

        Returns:
            Dictionary with all lap data.
        """
        return {
            "total_time_ms": self._timer.elapsed_ms(),
            "lap_count": len(self._laps),
            "average_lap_ms": self.average_lap_time(),
            "laps": [
                {"name": name, "lap_time_ms": lap, "cumulative_ms": cumulative}
                for name, lap, cumulative in self._laps
            ],
        }

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Formatted summary string.
        """
        lines = [f"LapTimer: {len(self._laps)} laps, Total: {self._timer.elapsed_ms():.2f} ms"]
        for name, lap_time, cumulative in self._laps:
            lines.append(f"  {name}: {lap_time:.2f} ms (cumulative: {cumulative:.2f} ms)")
        return "\n".join(lines)


# =============================================================================
# Multi-Timer Manager
# =============================================================================


class MultiTimer:
    """Manages multiple named timers for concurrent timing.

    Useful for timing multiple parallel or interleaved operations.
    """

    def __init__(self) -> None:
        """Initialize multi-timer."""
        self._timers: Dict[str, BenchmarkTimer] = {}

    def create(self, name: str) -> BenchmarkTimer:
        """Create and return a new named timer.

        Args:
            name: Timer name.

        Returns:
            The created BenchmarkTimer.
        """
        if name in self._timers:
            raise ValueError(f"Timer '{name}' already exists")
        timer = BenchmarkTimer()
        self._timers[name] = timer
        return timer

    def get(self, name: str) -> BenchmarkTimer | None:
        """Get a timer by name.

        Args:
            name: Timer name.

        Returns:
            The timer or None if not found.
        """
        return self._timers.get(name)

    def start(self, name: str) -> BenchmarkTimer:
        """Create and start a named timer.

        Args:
            name: Timer name.

        Returns:
            The started timer.
        """
        if name in self._timers:
            timer = self._timers[name]
            timer.reset()
        else:
            timer = BenchmarkTimer()
            self._timers[name] = timer
        timer.start()
        return timer

    def stop(self, name: str) -> float:
        """Stop a timer and return elapsed time.

        Args:
            name: Timer name.

        Returns:
            Elapsed time in milliseconds.

        Raises:
            KeyError: If timer not found.
        """
        if name not in self._timers:
            raise KeyError(f"Timer '{name}' not found")
        timer = self._timers[name]
        timer.stop()
        return timer.elapsed_ms()

    def elapsed(self, name: str) -> float:
        """Get elapsed time for a timer.

        Args:
            name: Timer name.

        Returns:
            Elapsed time in milliseconds.
        """
        if name not in self._timers:
            return 0.0
        return self._timers[name].elapsed_ms()

    def stop_all(self) -> Dict[str, float]:
        """Stop all timers and return their elapsed times.

        Returns:
            Dictionary mapping names to elapsed times.
        """
        results = {}
        for name, timer in self._timers.items():
            if timer.is_running:
                timer.stop()
            results[name] = timer.elapsed_ms()
        return results

    def reset_all(self) -> None:
        """Reset all timers."""
        for timer in self._timers.values():
            timer.reset()

    def clear(self) -> None:
        """Remove all timers."""
        self._timers.clear()

    @property
    def names(self) -> list[str]:
        """Get list of all timer names."""
        return list(self._timers.keys())

    def to_dict(self) -> Dict[str, float]:
        """Get all timer elapsed times as dictionary.

        Returns:
            Dictionary mapping names to elapsed times in ms.
        """
        return {name: timer.elapsed_ms() for name, timer in self._timers.items()}

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Formatted summary string.
        """
        lines = [f"MultiTimer: {len(self._timers)} timers"]
        for name, timer in sorted(self._timers.items()):
            status = "running" if timer.is_running else "stopped"
            lines.append(f"  {name}: {timer.elapsed_ms():.2f} ms ({status})")
        return "\n".join(lines)


# =============================================================================
# Timing Decorators
# =============================================================================


def timed(func):
    """Decorator to time function execution.

    Adds timing_ms attribute to function result if it's a dict,
    or prints timing info otherwise.

    Args:
        func: Function to decorate.

    Returns:
        Wrapped function with timing.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = BenchmarkTimer()
        timer.start()
        try:
            result = func(*args, **kwargs)
        finally:
            timer.stop()

        elapsed = timer.elapsed_ms()

        # Try to attach timing to result
        if isinstance(result, dict):
            result["_timing_ms"] = elapsed
        else:
            # Store on function for later retrieval
            wrapper._last_timing_ms = elapsed

        return result

    wrapper._last_timing_ms = 0.0
    return wrapper


class TimingContext:
    """Context manager for timing code blocks with named sections.

    Example:
        with TimingContext() as ctx:
            ctx.section("init")
            # initialization code
            ctx.section("process")
            # processing code
        print(ctx.summary())
    """

    def __init__(self, name: str = "timing") -> None:
        """Initialize timing context.

        Args:
            name: Name for this timing context.
        """
        self._name = name
        self._timer = BenchmarkTimer()
        self._sections: list[tuple[str, float]] = []
        self._current_section: str | None = None
        self._section_start: float = 0.0

    def __enter__(self) -> TimingContext:
        """Start timing."""
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and finalize current section."""
        if self._current_section:
            elapsed = self._timer.elapsed_ms() - self._section_start
            self._sections.append((self._current_section, elapsed))
        self._timer.stop()

    def section(self, name: str) -> None:
        """Start a new timing section.

        Args:
            name: Section name.
        """
        current_time = self._timer.elapsed_ms()

        # Close previous section
        if self._current_section:
            elapsed = current_time - self._section_start
            self._sections.append((self._current_section, elapsed))

        self._current_section = name
        self._section_start = current_time

    @property
    def sections(self) -> list[tuple[str, float]]:
        """Get all sections as (name, duration_ms) tuples."""
        return list(self._sections)

    @property
    def total_time_ms(self) -> float:
        """Get total elapsed time."""
        return self._timer.elapsed_ms()

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "name": self._name,
            "total_time_ms": self._timer.elapsed_ms(),
            "sections": [
                {"name": name, "duration_ms": duration}
                for name, duration in self._sections
            ],
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        total = self._timer.elapsed_ms()
        lines = [f"TimingContext '{self._name}': {total:.2f} ms total"]
        for name, duration in self._sections:
            pct = (duration / total * 100) if total > 0 else 0
            lines.append(f"  {name}: {duration:.2f} ms ({pct:.1f}%)")
        return "\n".join(lines)


__all__ = [
    "BenchmarkTimer",
    "TimerStatistics",
    "LapTimer",
    "MultiTimer",
    "timed",
    "TimingContext",
]
