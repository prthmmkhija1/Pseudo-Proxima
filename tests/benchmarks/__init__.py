"""
Proxima Performance Benchmarks

This package contains performance benchmarks for quantum simulation
backends and core functionality.
"""

from .backend_benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    BackendBenchmarkSuite,
    run_backend_benchmarks,
)
from .circuit_benchmarks import (
    CircuitBenchmarkSuite,
    run_circuit_benchmarks,
)
from .comparison_benchmarks import (
    ComparisonBenchmarkSuite,
    run_comparison_benchmarks,
)
from .utils import (
    generate_benchmark_report,
    save_benchmark_results,
    load_benchmark_results,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BackendBenchmarkSuite",
    "CircuitBenchmarkSuite",
    "ComparisonBenchmarkSuite",
    "run_backend_benchmarks",
    "run_circuit_benchmarks",
    "run_comparison_benchmarks",
    "generate_benchmark_report",
    "save_benchmark_results",
    "load_benchmark_results",
]
