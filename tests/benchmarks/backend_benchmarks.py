"""
Backend Performance Benchmarks

Benchmarks for comparing performance across different quantum simulation backends.
"""

import time
import statistics
import gc
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
import json


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Qubit configurations to test
    qubit_counts: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24])
    
    # Number of shots per execution
    shots: int = 1000
    
    # Number of repetitions for timing
    repetitions: int = 5
    
    # Warmup runs before measurement
    warmup_runs: int = 2
    
    # Circuit types to benchmark
    circuit_types: List[str] = field(default_factory=lambda: [
        "bell",
        "ghz",
        "qft",
        "random",
        "variational"
    ])
    
    # Backends to benchmark
    backends: List[str] = field(default_factory=lambda: [
        "cirq",
        "qiskit_aer",
        "quest",
        "cuquantum",
        "qsim"
    ])
    
    # Timeout per benchmark (seconds)
    timeout: float = 300.0
    
    # Whether to include memory measurements
    measure_memory: bool = True
    
    # Whether to skip unavailable backends
    skip_unavailable: bool = True


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    backend: str
    circuit_type: str
    num_qubits: int
    shots: int
    
    # Timing statistics (seconds)
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    median_time: float
    
    # Additional metrics
    throughput: float  # shots/second
    memory_mb: Optional[float] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    successful: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "circuit_type": self.circuit_type,
            "num_qubits": self.num_qubits,
            "shots": self.shots,
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "median_time": self.median_time,
            "throughput": self.throughput,
            "memory_mb": self.memory_mb,
            "timestamp": self.timestamp,
            "successful": self.successful,
            "error_message": self.error_message,
        }


class BackendBenchmarkSuite:
    """
    Suite of benchmarks for quantum simulation backends.
    
    Measures:
    - Execution time for various circuit types and sizes
    - Memory usage
    - Throughput (shots per second)
    - Scaling behavior
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark suite.
        
        Args:
            config: Benchmark configuration. Uses defaults if not provided.
        """
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self._backends: Dict[str, Any] = {}
    
    def _get_backend(self, name: str) -> Any:
        """Get or create backend instance."""
        if name not in self._backends:
            try:
                # Import here to avoid circular imports
                from proxima.backends import get_backend
                self._backends[name] = get_backend(name)
            except Exception as e:
                if not self.config.skip_unavailable:
                    raise
                return None
        return self._backends[name]
    
    def _create_circuit(self, circuit_type: str, num_qubits: int) -> str:
        """Create benchmark circuit in OpenQASM format."""
        if circuit_type == "bell":
            return self._create_bell_circuit(num_qubits)
        elif circuit_type == "ghz":
            return self._create_ghz_circuit(num_qubits)
        elif circuit_type == "qft":
            return self._create_qft_circuit(num_qubits)
        elif circuit_type == "random":
            return self._create_random_circuit(num_qubits)
        elif circuit_type == "variational":
            return self._create_variational_circuit(num_qubits)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    def _create_bell_circuit(self, num_qubits: int) -> str:
        """Create Bell-like entanglement circuit."""
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
            'h q[0];',
        ]
        for i in range(num_qubits - 1):
            lines.append(f'cx q[{i}], q[{i+1}];')
        lines.append('measure q -> c;')
        return '\n'.join(lines)
    
    def _create_ghz_circuit(self, num_qubits: int) -> str:
        """Create GHZ state preparation circuit."""
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
            'h q[0];',
        ]
        for i in range(1, num_qubits):
            lines.append(f'cx q[0], q[{i}];')
        lines.append('measure q -> c;')
        return '\n'.join(lines)
    
    def _create_qft_circuit(self, num_qubits: int) -> str:
        """Create Quantum Fourier Transform circuit."""
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]
        
        import math
        
        for i in range(num_qubits):
            lines.append(f'h q[{i}];')
            for j in range(i + 1, num_qubits):
                angle = math.pi / (2 ** (j - i))
                lines.append(f'cp({angle}) q[{j}], q[{i}];')
        
        # Swap qubits for proper ordering
        for i in range(num_qubits // 2):
            lines.append(f'swap q[{i}], q[{num_qubits - 1 - i}];')
        
        lines.append('measure q -> c;')
        return '\n'.join(lines)
    
    def _create_random_circuit(self, num_qubits: int, depth: int = None) -> str:
        """Create random circuit with various gates."""
        import random
        
        if depth is None:
            depth = num_qubits * 2
        
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]
        
        single_gates = ['h', 'x', 'y', 'z', 's', 't']
        
        random.seed(42)  # Reproducible
        
        for _ in range(depth):
            # Single qubit gates
            for q in range(num_qubits):
                if random.random() < 0.5:
                    gate = random.choice(single_gates)
                    lines.append(f'{gate} q[{q}];')
            
            # Two qubit gates
            for q in range(num_qubits - 1):
                if random.random() < 0.3:
                    lines.append(f'cx q[{q}], q[{q+1}];')
        
        lines.append('measure q -> c;')
        return '\n'.join(lines)
    
    def _create_variational_circuit(self, num_qubits: int, layers: int = 3) -> str:
        """Create variational ansatz circuit."""
        import math
        
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]
        
        for layer in range(layers):
            # Rotation layer
            for q in range(num_qubits):
                theta = 0.1 * (layer + 1) * (q + 1)
                lines.append(f'rx({theta}) q[{q}];')
                lines.append(f'ry({theta * 2}) q[{q}];')
            
            # Entanglement layer
            for q in range(num_qubits - 1):
                lines.append(f'cx q[{q}], q[{q+1}];')
        
        lines.append('measure q -> c;')
        return '\n'.join(lines)
    
    def _measure_memory(self) -> Optional[float]:
        """Measure current memory usage in MB."""
        if not self.config.measure_memory:
            return None
        
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return None
    
    def run_single_benchmark(
        self,
        backend_name: str,
        circuit_type: str,
        num_qubits: int
    ) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            backend_name: Name of the backend to benchmark
            circuit_type: Type of circuit to use
            num_qubits: Number of qubits
        
        Returns:
            BenchmarkResult with timing statistics
        """
        backend = self._get_backend(backend_name)
        
        if backend is None:
            return BenchmarkResult(
                backend=backend_name,
                circuit_type=circuit_type,
                num_qubits=num_qubits,
                shots=self.config.shots,
                mean_time=0,
                std_time=0,
                min_time=0,
                max_time=0,
                median_time=0,
                throughput=0,
                successful=False,
                error_message="Backend not available"
            )
        
        try:
            circuit = self._create_circuit(circuit_type, num_qubits)
            
            # Warmup runs
            for _ in range(self.config.warmup_runs):
                backend.run_qasm(circuit, shots=self.config.shots)
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory before
            memory_before = self._measure_memory()
            
            # Timed runs
            times = []
            for _ in range(self.config.repetitions):
                start = time.perf_counter()
                backend.run_qasm(circuit, shots=self.config.shots)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            # Measure memory after
            memory_after = self._measure_memory()
            memory_used = None
            if memory_before and memory_after:
                memory_used = memory_after - memory_before
            
            mean_time = statistics.mean(times)
            
            return BenchmarkResult(
                backend=backend_name,
                circuit_type=circuit_type,
                num_qubits=num_qubits,
                shots=self.config.shots,
                mean_time=mean_time,
                std_time=statistics.stdev(times) if len(times) > 1 else 0,
                min_time=min(times),
                max_time=max(times),
                median_time=statistics.median(times),
                throughput=self.config.shots / mean_time if mean_time > 0 else 0,
                memory_mb=memory_used,
                successful=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                backend=backend_name,
                circuit_type=circuit_type,
                num_qubits=num_qubits,
                shots=self.config.shots,
                mean_time=0,
                std_time=0,
                min_time=0,
                max_time=0,
                median_time=0,
                throughput=0,
                successful=False,
                error_message=str(e)
            )
    
    def run_all(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[BenchmarkResult]:
        """
        Run all configured benchmarks.
        
        Args:
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of all benchmark results
        """
        self.results = []
        
        total = (
            len(self.config.backends) *
            len(self.config.circuit_types) *
            len(self.config.qubit_counts)
        )
        current = 0
        
        for backend in self.config.backends:
            for circuit_type in self.config.circuit_types:
                for num_qubits in self.config.qubit_counts:
                    current += 1
                    
                    if progress_callback:
                        progress_callback(
                            f"[{current}/{total}] {backend} / {circuit_type} / {num_qubits}q"
                        )
                    
                    result = self.run_single_benchmark(
                        backend,
                        circuit_type,
                        num_qubits
                    )
                    self.results.append(result)
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        if not self.results:
            return {"error": "No benchmark results"}
        
        # Group by backend
        by_backend = {}
        for result in self.results:
            if result.backend not in by_backend:
                by_backend[result.backend] = []
            by_backend[result.backend].append(result)
        
        summary = {
            "total_benchmarks": len(self.results),
            "successful": sum(1 for r in self.results if r.successful),
            "failed": sum(1 for r in self.results if not r.successful),
            "backends": {}
        }
        
        for backend, results in by_backend.items():
            successful = [r for r in results if r.successful]
            if successful:
                summary["backends"][backend] = {
                    "benchmarks_run": len(results),
                    "successful": len(successful),
                    "mean_time": statistics.mean(r.mean_time for r in successful),
                    "mean_throughput": statistics.mean(r.throughput for r in successful),
                }
            else:
                summary["backends"][backend] = {
                    "benchmarks_run": len(results),
                    "successful": 0,
                    "error": results[0].error_message if results else "Unknown"
                }
        
        return summary


def run_backend_benchmarks(
    config: Optional[BenchmarkConfig] = None,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """
    Convenience function to run backend benchmarks.
    
    Args:
        config: Benchmark configuration
        verbose: Whether to print progress
    
    Returns:
        List of benchmark results
    """
    suite = BackendBenchmarkSuite(config)
    
    progress_fn = None
    if verbose:
        progress_fn = lambda msg: print(msg)
    
    results = suite.run_all(progress_callback=progress_fn)
    
    if verbose:
        print("\n=== Benchmark Summary ===")
        summary = suite.get_summary()
        print(json.dumps(summary, indent=2))
    
    return results
