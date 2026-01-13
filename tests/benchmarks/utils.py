"""
Benchmark Utilities

Utility functions for benchmark execution, reporting, and result management.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import contextmanager


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    
    platform: str = ""
    python_version: str = ""
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @classmethod
    def collect(cls) -> "SystemInfo":
        """Collect current system information."""
        import platform
        import sys
        
        info = cls(
            platform=platform.platform(),
            python_version=sys.version,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        # CPU count
        try:
            info.cpu_count = os.cpu_count() or 0
        except Exception:
            pass
        
        # Memory
        try:
            import psutil
            info.memory_total_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
        
        # GPU
        try:
            import cupy
            info.gpu_available = True
            info.gpu_name = "CUDA GPU Available"
        except ImportError:
            pass
        
        return info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "memory_total_gb": round(self.memory_total_gb, 2),
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "timestamp": self.timestamp,
        }


@contextmanager
def timing_context(name: str = "Operation"):
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.4f}s")


class BenchmarkTimer:
    """Timer for tracking benchmark operations."""
    
    def __init__(self):
        self._times: Dict[str, List[float]] = {}
        self._start_time: Optional[float] = None
        self._current_name: Optional[str] = None
    
    def start(self, name: str = "default"):
        """Start timing an operation."""
        self._current_name = name
        self._start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and record the elapsed time."""
        if self._start_time is None:
            return 0.0
        
        elapsed = time.perf_counter() - self._start_time
        
        name = self._current_name or "default"
        if name not in self._times:
            self._times[name] = []
        self._times[name].append(elapsed)
        
        self._start_time = None
        self._current_name = None
        
        return elapsed
    
    def get_stats(self, name: str = "default") -> Dict[str, float]:
        """Get statistics for a timed operation."""
        import statistics
        
        times = self._times.get(name, [])
        if not times:
            return {"count": 0}
        
        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "total": sum(times),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all timed operations."""
        return {name: self.get_stats(name) for name in self._times}
    
    def reset(self):
        """Reset all recorded times."""
        self._times = {}
        self._start_time = None
        self._current_name = None


def generate_benchmark_report(
    results: List[Any],
    output_format: str = "markdown",
    title: str = "Benchmark Report"
) -> str:
    """
    Generate a formatted benchmark report.
    
    Args:
        results: List of benchmark result objects
        output_format: Output format ("markdown", "json", "text")
        title: Report title
    
    Returns:
        Formatted report string
    """
    if output_format == "json":
        return _generate_json_report(results)
    elif output_format == "text":
        return _generate_text_report(results, title)
    else:
        return _generate_markdown_report(results, title)


def _generate_json_report(results: List[Any]) -> str:
    """Generate JSON report."""
    data = []
    for result in results:
        if hasattr(result, 'to_dict'):
            data.append(result.to_dict())
        else:
            data.append(str(result))
    
    return json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "system_info": SystemInfo.collect().to_dict(),
        "results": data
    }, indent=2)


def _generate_text_report(results: List[Any], title: str) -> str:
    """Generate plain text report."""
    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    
    for i, result in enumerate(results, 1):
        lines.append(f"--- Result {i} ---")
        if hasattr(result, 'to_dict'):
            for key, value in result.to_dict().items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {result}")
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def _generate_markdown_report(results: List[Any], title: str) -> str:
    """Generate Markdown report."""
    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## System Information",
        "",
    ]
    
    system_info = SystemInfo.collect()
    lines.extend([
        f"- **Platform:** {system_info.platform}",
        f"- **Python:** {system_info.python_version.split()[0]}",
        f"- **CPUs:** {system_info.cpu_count}",
        f"- **Memory:** {system_info.memory_total_gb:.1f} GB",
        f"- **GPU:** {'Yes' if system_info.gpu_available else 'No'}",
        "",
        "## Results",
        "",
    ])
    
    # Group results by type
    backend_results = []
    circuit_results = []
    comparison_results = []
    other_results = []
    
    for result in results:
        result_type = type(result).__name__
        if "Backend" in result_type:
            backend_results.append(result)
        elif "Circuit" in result_type:
            circuit_results.append(result)
        elif "Comparison" in result_type:
            comparison_results.append(result)
        else:
            other_results.append(result)
    
    # Backend Results
    if backend_results:
        lines.extend(_format_backend_results_md(backend_results))
    
    # Circuit Results
    if circuit_results:
        lines.extend(_format_circuit_results_md(circuit_results))
    
    # Comparison Results
    if comparison_results:
        lines.extend(_format_comparison_results_md(comparison_results))
    
    # Other Results
    if other_results:
        lines.append("### Other Results")
        lines.append("")
        for result in other_results:
            if hasattr(result, 'to_dict'):
                lines.append(f"```json\n{json.dumps(result.to_dict(), indent=2)}\n```")
            else:
                lines.append(f"- {result}")
        lines.append("")
    
    return "\n".join(lines)


def _format_backend_results_md(results: List[Any]) -> List[str]:
    """Format backend results as Markdown."""
    lines = [
        "### Backend Benchmarks",
        "",
        "| Backend | Qubits | Circuit | Mean Time (s) | Throughput |",
        "|---------|--------|---------|--------------|------------|",
    ]
    
    for r in results:
        if hasattr(r, 'successful') and r.successful:
            d = r.to_dict() if hasattr(r, 'to_dict') else {}
            lines.append(
                f"| {d.get('backend_name', 'N/A')} | "
                f"{d.get('num_qubits', 'N/A')} | "
                f"{d.get('circuit_type', 'N/A')} | "
                f"{d.get('mean_time', 0):.4f} | "
                f"{d.get('throughput', 0):.1f} shots/s |"
            )
    
    lines.append("")
    return lines


def _format_circuit_results_md(results: List[Any]) -> List[str]:
    """Format circuit results as Markdown."""
    lines = [
        "### Circuit Benchmarks",
        "",
        "| Operation | Qubits | Depth | Mean Time (s) |",
        "|-----------|--------|-------|--------------|",
    ]
    
    for r in results:
        if hasattr(r, 'successful') and r.successful:
            d = r.to_dict() if hasattr(r, 'to_dict') else {}
            lines.append(
                f"| {d.get('operation', 'N/A')} | "
                f"{d.get('num_qubits', 'N/A')} | "
                f"{d.get('circuit_depth', 'N/A')} | "
                f"{d.get('mean_time', 0):.4f} |"
            )
    
    lines.append("")
    return lines


def _format_comparison_results_md(results: List[Any]) -> List[str]:
    """Format comparison results as Markdown."""
    lines = [
        "### Backend Comparisons",
        "",
        "| Circuit | Qubits | Fastest | Most Accurate |",
        "|---------|--------|---------|---------------|",
    ]
    
    for r in results:
        if hasattr(r, 'successful') and r.successful:
            d = r.to_dict() if hasattr(r, 'to_dict') else {}
            lines.append(
                f"| {d.get('circuit_type', 'N/A')} | "
                f"{d.get('num_qubits', 'N/A')} | "
                f"{d.get('fastest_backend', 'N/A')} | "
                f"{d.get('most_accurate_backend', 'N/A')} |"
            )
    
    lines.append("")
    return lines


def save_benchmark_results(
    results: List[Any],
    filepath: Union[str, Path],
    include_system_info: bool = True
) -> None:
    """
    Save benchmark results to a JSON file.
    
    Args:
        results: List of benchmark result objects
        filepath: Path to save results
        include_system_info: Whether to include system information
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "version": "1.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "results": []
    }
    
    if include_system_info:
        data["system_info"] = SystemInfo.collect().to_dict()
    
    for result in results:
        if hasattr(result, 'to_dict'):
            data["results"].append(result.to_dict())
        else:
            data["results"].append({"raw": str(result)})
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_benchmark_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load benchmark results from a JSON file.
    
    Args:
        filepath: Path to load results from
    
    Returns:
        Dictionary with loaded results
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_benchmark_runs(
    run1: Union[str, Path, Dict],
    run2: Union[str, Path, Dict]
) -> Dict[str, Any]:
    """
    Compare two benchmark runs.
    
    Args:
        run1: First benchmark run (filepath or dict)
        run2: Second benchmark run (filepath or dict)
    
    Returns:
        Comparison results
    """
    if isinstance(run1, (str, Path)):
        run1 = load_benchmark_results(run1)
    if isinstance(run2, (str, Path)):
        run2 = load_benchmark_results(run2)
    
    comparison = {
        "run1_timestamp": run1.get("generated", "unknown"),
        "run2_timestamp": run2.get("generated", "unknown"),
        "differences": [],
    }
    
    results1 = {_result_key(r): r for r in run1.get("results", [])}
    results2 = {_result_key(r): r for r in run2.get("results", [])}
    
    all_keys = set(results1.keys()) | set(results2.keys())
    
    for key in all_keys:
        r1 = results1.get(key)
        r2 = results2.get(key)
        
        if r1 and r2:
            time1 = r1.get("mean_time", 0)
            time2 = r2.get("mean_time", 0)
            
            if time1 and time2:
                speedup = time1 / time2 if time2 > 0 else 0
                comparison["differences"].append({
                    "key": key,
                    "run1_time": time1,
                    "run2_time": time2,
                    "speedup": speedup,
                    "improvement": f"{(speedup - 1) * 100:.1f}%"
                })
    
    return comparison


def _result_key(result: Dict) -> str:
    """Generate a unique key for a result."""
    parts = []
    for field in ["backend_name", "circuit_type", "num_qubits", "operation"]:
        if field in result:
            parts.append(str(result[field]))
    return "/".join(parts) if parts else str(id(result))


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_throughput(value: float, unit: str = "ops/s") -> str:
    """Format throughput in human-readable format."""
    if value < 1000:
        return f"{value:.1f} {unit}"
    elif value < 1_000_000:
        return f"{value/1000:.1f}K {unit}"
    else:
        return f"{value/1_000_000:.2f}M {unit}"


def estimate_memory_usage(num_qubits: int) -> float:
    """
    Estimate memory usage for a quantum simulation.
    
    Args:
        num_qubits: Number of qubits
    
    Returns:
        Estimated memory in GB
    """
    # State vector: 2^n complex numbers, 16 bytes each
    state_vector_size = (2 ** num_qubits) * 16
    
    # Additional overhead (approximately 2x)
    total_bytes = state_vector_size * 2
    
    return total_bytes / (1024 ** 3)


def can_simulate_locally(
    num_qubits: int,
    safety_factor: float = 0.5
) -> bool:
    """
    Check if simulation can run locally.
    
    Args:
        num_qubits: Number of qubits
        safety_factor: Fraction of available memory to use
    
    Returns:
        True if simulation fits in memory
    """
    required = estimate_memory_usage(num_qubits)
    
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 ** 3)
        return required < available * safety_factor
    except ImportError:
        # Assume 8GB available if psutil not installed
        return required < 8 * safety_factor


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "Progress",
    length: int = 50
) -> None:
    """Print a progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    print(f"\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})", end="")
    if current >= total:
        print()
