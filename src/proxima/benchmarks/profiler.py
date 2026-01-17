"""Backend performance profiling based on benchmark history."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, TypeVar

from proxima.benchmarks.statistics import StatisticsCalculator
from proxima.data.benchmark_registry import BenchmarkRegistry
from proxima.data.metrics import BenchmarkResult


T = TypeVar("T")


class ProfileCategory(str, Enum):
    """Categories for profiling analysis."""
    
    SPEED = "speed"
    MEMORY = "memory"
    SCALING = "scaling"
    STABILITY = "stability"
    EFFICIENCY = "efficiency"


@dataclass
class ProfileScore:
    """Score for a specific profiling category."""
    
    category: ProfileCategory
    score: float  # 0.0 to 1.0
    grade: str  # A, B, C, D, F
    details: str = ""
    
    @classmethod
    def from_score(cls, category: ProfileCategory, score: float, details: str = "") -> "ProfileScore":
        """Create score with automatic grade calculation."""
        if score >= 0.9:
            grade = "A"
        elif score >= 0.8:
            grade = "B"
        elif score >= 0.7:
            grade = "C"
        elif score >= 0.6:
            grade = "D"
        else:
            grade = "F"
        return cls(category=category, score=score, grade=grade, details=details)


@dataclass(slots=True)
class BackendProfile:
    """Performance profile for a quantum backend."""

    backend_name: str
    total_benchmarks: int
    average_execution_time_ms: float
    performance_by_qubit_count: dict[int, float] = field(default_factory=dict)
    performance_by_depth: dict[int, float] = field(default_factory=dict)
    memory_usage_trend: dict[int, float] = field(default_factory=dict)
    optimal_circuit_sizes: list[int] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    # Extended profile data
    scores: list[ProfileScore] = field(default_factory=list)
    variance_by_qubit: dict[int, float] = field(default_factory=dict)
    percentile_95_ms: float = 0.0
    percentile_99_ms: float = 0.0
    throughput_per_second: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_report(self) -> str:
        lines = [f"Backend: {self.backend_name}", f"Total benchmarks: {self.total_benchmarks}"]
        lines.append(f"Average execution time: {self.average_execution_time_ms:.2f} ms")
        if self.percentile_95_ms > 0:
            lines.append(f"P95 execution time: {self.percentile_95_ms:.2f} ms")
        if self.percentile_99_ms > 0:
            lines.append(f"P99 execution time: {self.percentile_99_ms:.2f} ms")
        if self.throughput_per_second > 0:
            lines.append(f"Throughput: {self.throughput_per_second:.2f} circuits/sec")
        if self.performance_by_qubit_count:
            lines.append("Avg time by qubit count:")
            for q, t in sorted(self.performance_by_qubit_count.items()):
                lines.append(f"  q={q}: {t:.2f} ms")
        if self.scores:
            lines.append("Scores:")
            for score in self.scores:
                lines.append(f"  {score.category.value}: {score.grade} ({score.score:.2f})")
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "backend_name": self.backend_name,
            "total_benchmarks": self.total_benchmarks,
            "average_execution_time_ms": self.average_execution_time_ms,
            "percentile_95_ms": self.percentile_95_ms,
            "percentile_99_ms": self.percentile_99_ms,
            "throughput_per_second": self.throughput_per_second,
            "performance_by_qubit_count": self.performance_by_qubit_count,
            "performance_by_depth": self.performance_by_depth,
            "memory_usage_trend": self.memory_usage_trend,
            "optimal_circuit_sizes": self.optimal_circuit_sizes,
            "recommendations": self.recommendations,
            "scores": [
                {"category": s.category.value, "score": s.score, "grade": s.grade}
                for s in self.scores
            ],
            "created_at": self.created_at.isoformat(),
        }
    
    @property
    def overall_grade(self) -> str:
        """Calculate overall grade from individual scores."""
        if not self.scores:
            return "N/A"
        avg_score = sum(s.score for s in self.scores) / len(self.scores)
        if avg_score >= 0.9:
            return "A"
        elif avg_score >= 0.8:
            return "B"
        elif avg_score >= 0.7:
            return "C"
        elif avg_score >= 0.6:
            return "D"
        return "F"


class BackendProfiler:
    """Generates performance profiles for a backend using historical results."""

    def __init__(self, registry: BenchmarkRegistry, stats: StatisticsCalculator | None = None) -> None:
        self.registry = registry
        self.stats = stats or StatisticsCalculator()

    def generate_profile(self, backend_name: str) -> BackendProfile:
        results = self.registry.get_results_for_backend(backend_name, limit=None)
        if not results:
            return BackendProfile(
                backend_name=backend_name,
                total_benchmarks=0,
                average_execution_time_ms=0.0,
            )

        exec_times = [r.metrics.execution_time_ms for r in results if r.metrics]
        avg_time = float(statistics.mean(exec_times)) if exec_times else 0.0

        perf_by_qubits = self._aggregate_by(results, key="qubit_count")
        perf_by_depth = self._aggregate_by(results, key="depth")
        mem_usage = self._aggregate_memory(results)
        optimal_sizes = self._find_optimal_qubits(perf_by_qubits)

        profile = BackendProfile(
            backend_name=backend_name,
            total_benchmarks=len(results),
            average_execution_time_ms=avg_time,
            performance_by_qubit_count=perf_by_qubits,
            performance_by_depth=perf_by_depth,
            memory_usage_trend=mem_usage,
            optimal_circuit_sizes=optimal_sizes,
            recommendations=[],
        )
        profile.recommendations = self._generate_recommendations(profile)
        return profile

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _aggregate_by(self, results: List[BenchmarkResult], key: str) -> dict[int, float]:
        buckets: dict[int, list[float]] = {}
        for r in results:
            if not r.metrics:
                continue
            value = r.metrics.circuit_info.get(key) if r.metrics.circuit_info else None
            if value is None:
                continue
            try:
                idx = int(value)
            except Exception:
                continue
            buckets.setdefault(idx, []).append(r.metrics.execution_time_ms)
        return {k: float(statistics.mean(v)) for k, v in buckets.items() if v}

    def _aggregate_memory(self, results: List[BenchmarkResult]) -> dict[int, float]:
        buckets: dict[int, list[float]] = {}
        for r in results:
            if not r.metrics:
                continue
            q = r.metrics.circuit_info.get("qubit_count") if r.metrics.circuit_info else None
            if q is None:
                continue
            try:
                q_int = int(q)
            except Exception:
                continue
            buckets.setdefault(q_int, []).append(r.metrics.memory_peak_mb)
        return {k: float(statistics.mean(v)) for k, v in buckets.items() if v}

    def _find_optimal_qubits(self, perf_by_qubits: dict[int, float], top_k: int = 3) -> list[int]:
        if not perf_by_qubits:
            return []
        return [q for q, _ in sorted(perf_by_qubits.items(), key=lambda x: x[1])[:top_k]]

    def _generate_recommendations(self, profile: BackendProfile) -> list[str]:
        recs: list[str] = []
        avg = profile.average_execution_time_ms
        if avg and avg < 50:
            recs.append("Recommended for quick simulations (low average execution time).")
        if profile.memory_usage_trend:
            avg_mem = statistics.mean(profile.memory_usage_trend.values())
            if avg_mem < 256:
                recs.append("Low memory usage; suitable for constrained environments.")
        if profile.optimal_circuit_sizes:
            recs.append(
                f"Performs best around qubit counts: {', '.join(map(str, profile.optimal_circuit_sizes))}."
            )
        # Detect cliffs: sudden slowdowns between adjacent qubit counts (>1.5x)
        cliffs: list[int] = []
        sorted_perf = sorted(profile.performance_by_qubit_count.items())
        for (q1, t1), (q2, t2) in zip(sorted_perf, sorted_perf[1:]):
            if t1 and t2 and t2 / t1 > 1.5:
                cliffs.append(q2)
        if cliffs:
            recs.append(
                f"Performance cliffs observed near qubit counts: {', '.join(map(str, cliffs))}."
            )
        # GPU backends with high throughput are suited for batch jobs
        if "gpu" in profile.backend_name.lower() or "cuda" in profile.backend_name.lower():
            if avg and avg < 100:
                recs.append("GPU backend with high throughput; recommended for batch jobs.")
        return recs


# =============================================================================
# Extended Profiling (Feature - Benchmarks)
# =============================================================================


class DetailedProfiler(BackendProfiler):
    """Extended profiler with detailed analysis capabilities."""
    
    def generate_detailed_profile(self, backend_name: str) -> BackendProfile:
        """Generate comprehensive profile with scores.
        
        Args:
            backend_name: Backend to profile.
            
        Returns:
            BackendProfile with all metrics and scores.
        """
        # Get base profile
        profile = self.generate_profile(backend_name)
        
        if profile.total_benchmarks == 0:
            return profile
        
        # Get results for detailed analysis
        results = self.registry.get_results_for_backend(backend_name, limit=None)
        exec_times = [r.metrics.execution_time_ms for r in results if r.metrics]
        
        # Calculate percentiles
        if exec_times:
            sorted_times = sorted(exec_times)
            profile.percentile_95_ms = self._percentile(sorted_times, 95)
            profile.percentile_99_ms = self._percentile(sorted_times, 99)
            
            # Throughput estimation
            total_time_sec = sum(exec_times) / 1000
            if total_time_sec > 0:
                profile.throughput_per_second = len(exec_times) / total_time_sec
        
        # Calculate variance by qubit count
        profile.variance_by_qubit = self._calculate_variance_by_qubit(results)
        
        # Generate scores
        profile.scores = self._calculate_scores(profile, results)
        
        return profile
    
    def _percentile(self, sorted_data: list[float], p: int) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def _calculate_variance_by_qubit(
        self, 
        results: List[BenchmarkResult],
    ) -> dict[int, float]:
        """Calculate execution time variance by qubit count."""
        buckets: dict[int, list[float]] = {}
        for r in results:
            if not r.metrics or not r.metrics.circuit_info:
                continue
            q = r.metrics.circuit_info.get("qubit_count")
            if q is None:
                continue
            try:
                buckets.setdefault(int(q), []).append(r.metrics.execution_time_ms)
            except (TypeError, ValueError):
                continue
        
        return {
            q: statistics.variance(times) if len(times) > 1 else 0.0
            for q, times in buckets.items()
        }
    
    def _calculate_scores(
        self,
        profile: BackendProfile,
        results: List[BenchmarkResult],
    ) -> list[ProfileScore]:
        """Calculate performance scores for each category."""
        scores: list[ProfileScore] = []
        
        # Speed score (based on average time)
        # Lower is better, normalize to 0-1 (assuming 1000ms is poor)
        speed_score = max(0, 1 - profile.average_execution_time_ms / 1000)
        scores.append(ProfileScore.from_score(
            ProfileCategory.SPEED,
            speed_score,
            f"Avg: {profile.average_execution_time_ms:.2f}ms",
        ))
        
        # Memory score
        if profile.memory_usage_trend:
            avg_mem = statistics.mean(profile.memory_usage_trend.values())
            # Lower memory is better, normalize (assuming 4GB is poor)
            mem_score = max(0, 1 - avg_mem / 4096)
            scores.append(ProfileScore.from_score(
                ProfileCategory.MEMORY,
                mem_score,
                f"Avg: {avg_mem:.2f}MB",
            ))
        
        # Scaling score (how well it scales with qubits)
        if profile.performance_by_qubit_count:
            perf = profile.performance_by_qubit_count
            sorted_qubits = sorted(perf.keys())
            if len(sorted_qubits) >= 2:
                # Calculate scaling factor (lower growth rate is better)
                growth_rates = []
                for i in range(1, len(sorted_qubits)):
                    q1, q2 = sorted_qubits[i-1], sorted_qubits[i]
                    t1, t2 = perf[q1], perf[q2]
                    if t1 > 0:
                        growth_rates.append((t2 - t1) / t1)
                
                if growth_rates:
                    avg_growth = statistics.mean(growth_rates)
                    # Lower growth is better
                    scaling_score = max(0, 1 - min(avg_growth, 2) / 2)
                    scores.append(ProfileScore.from_score(
                        ProfileCategory.SCALING,
                        scaling_score,
                        f"Avg growth: {avg_growth * 100:.1f}%",
                    ))
        
        # Stability score (based on variance)
        if profile.variance_by_qubit:
            variances = list(profile.variance_by_qubit.values())
            if variances:
                avg_variance = statistics.mean(variances)
                # Lower variance is better
                stability_score = max(0, 1 - min(avg_variance, 1000) / 1000)
                scores.append(ProfileScore.from_score(
                    ProfileCategory.STABILITY,
                    stability_score,
                    f"Avg variance: {avg_variance:.2f}",
                ))
        
        # Efficiency score (throughput vs memory)
        if profile.throughput_per_second > 0 and profile.memory_usage_trend:
            avg_mem = statistics.mean(profile.memory_usage_trend.values())
            # Higher throughput per MB is better
            efficiency = profile.throughput_per_second / max(avg_mem, 1)
            eff_score = min(1.0, efficiency / 0.1)  # 0.1 circuit/s/MB is good
            scores.append(ProfileScore.from_score(
                ProfileCategory.EFFICIENCY,
                eff_score,
                f"{profile.throughput_per_second:.2f} circuits/s",
            ))
        
        return scores
    
    def compare_profiles(
        self,
        backend_names: list[str],
    ) -> dict[str, BackendProfile]:
        """Generate and compare profiles for multiple backends.
        
        Args:
            backend_names: List of backends to profile.
            
        Returns:
            Dict mapping backend names to profiles.
        """
        profiles = {}
        for name in backend_names:
            profiles[name] = self.generate_detailed_profile(name)
        return profiles
    
    def get_best_backend_for(
        self,
        category: ProfileCategory,
        backend_names: list[str] | None = None,
    ) -> str | None:
        """Find the best backend for a specific category.
        
        Args:
            category: Performance category to optimize.
            backend_names: Backends to consider (all if None).
            
        Returns:
            Name of best backend or None.
        """
        if backend_names is None:
            backend_names = self.registry.list_backends()
        
        best_backend = None
        best_score = -1.0
        
        for name in backend_names:
            profile = self.generate_detailed_profile(name)
            for score in profile.scores:
                if score.category == category and score.score > best_score:
                    best_score = score.score
                    best_backend = name
        
        return best_backend


class RealTimeProfiler:
    """Profile execution in real-time with callback support."""
    
    def __init__(self) -> None:
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._metrics: list[dict[str, Any]] = []
        self._start_time: float | None = None
    
    def add_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add callback for real-time metric updates."""
        self._callbacks.append(callback)
    
    def start_profiling(self) -> None:
        """Start profiling session."""
        self._start_time = time.perf_counter()
        self._metrics = []
    
    def record_execution(
        self,
        backend_name: str,
        execution_time_ms: float,
        memory_mb: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record an execution metric.
        
        Args:
            backend_name: Backend that executed.
            execution_time_ms: Execution time in milliseconds.
            memory_mb: Peak memory in MB.
            extra: Additional metadata.
        """
        elapsed = time.perf_counter() - (self._start_time or 0)
        
        metric = {
            "backend": backend_name,
            "execution_time_ms": execution_time_ms,
            "memory_mb": memory_mb,
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(),
            **(extra or {}),
        }
        
        self._metrics.append(metric)
        
        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(metric)
            except Exception:
                pass  # Don't let callback errors stop profiling
    
    def stop_profiling(self) -> dict[str, Any]:
        """Stop profiling and return summary.
        
        Returns:
            Summary of profiling session.
        """
        total_time = time.perf_counter() - (self._start_time or 0)
        
        summary = {
            "total_executions": len(self._metrics),
            "total_time_s": total_time,
            "metrics": self._metrics,
        }
        
        if self._metrics:
            exec_times = [m["execution_time_ms"] for m in self._metrics]
            summary["avg_execution_ms"] = statistics.mean(exec_times)
            summary["min_execution_ms"] = min(exec_times)
            summary["max_execution_ms"] = max(exec_times)
            if len(exec_times) > 1:
                summary["std_dev_ms"] = statistics.stdev(exec_times)
        
        self._start_time = None
        return summary
    
    def get_current_stats(self) -> dict[str, Any]:
        """Get current profiling statistics without stopping."""
        if not self._metrics:
            return {"executions": 0}
        
        exec_times = [m["execution_time_ms"] for m in self._metrics]
        return {
            "executions": len(self._metrics),
            "avg_ms": statistics.mean(exec_times),
            "min_ms": min(exec_times),
            "max_ms": max(exec_times),
        }


__all__ = [
    "BackendProfiler", 
    "BackendProfile",
    "DetailedProfiler",
    "RealTimeProfiler",
    "ProfileCategory",
    "ProfileScore",
]
