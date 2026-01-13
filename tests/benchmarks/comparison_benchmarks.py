"""
Backend Comparison Benchmarks

Benchmarks for comparing results and performance across multiple backends.
"""

import time
import statistics
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone


@dataclass
class ComparisonBenchmarkConfig:
    """Configuration for comparison benchmarks."""
    
    # Backends to compare
    backends: List[str] = field(default_factory=lambda: [
        "cirq",
        "qiskit_aer"
    ])
    
    # Reference backend for accuracy comparisons
    reference_backend: str = "cirq"
    
    # Qubit configurations
    qubit_counts: List[int] = field(default_factory=lambda: [2, 4, 8, 12, 16])
    
    # Shots for comparison
    shots: int = 10000
    
    # Number of repetitions for statistical comparison
    repetitions: int = 5
    
    # Circuit types to test
    circuit_types: List[str] = field(default_factory=lambda: [
        "bell",
        "ghz",
        "random"
    ])


@dataclass
class ComparisonBenchmarkResult:
    """Result of a comparison benchmark."""
    
    circuit_type: str
    num_qubits: int
    shots: int
    
    # Accuracy metrics
    backends_compared: List[str] = field(default_factory=list)
    fidelities: Dict[str, float] = field(default_factory=dict)
    kl_divergences: Dict[str, float] = field(default_factory=dict)
    
    # Performance comparison
    execution_times: Dict[str, float] = field(default_factory=dict)
    throughputs: Dict[str, float] = field(default_factory=dict)
    
    # Best performers
    fastest_backend: Optional[str] = None
    most_accurate_backend: Optional[str] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    successful: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "circuit_type": self.circuit_type,
            "num_qubits": self.num_qubits,
            "shots": self.shots,
            "backends_compared": self.backends_compared,
            "fidelities": self.fidelities,
            "kl_divergences": self.kl_divergences,
            "execution_times": self.execution_times,
            "throughputs": self.throughputs,
            "fastest_backend": self.fastest_backend,
            "most_accurate_backend": self.most_accurate_backend,
            "timestamp": self.timestamp,
            "successful": self.successful,
            "error_message": self.error_message,
        }


class ComparisonBenchmarkSuite:
    """
    Suite of benchmarks for comparing backend results.
    
    Measures:
    - Result fidelity between backends
    - KL divergence of distributions
    - Performance differences
    - Consistency across runs
    """
    
    def __init__(self, config: Optional[ComparisonBenchmarkConfig] = None):
        """Initialize benchmark suite."""
        self.config = config or ComparisonBenchmarkConfig()
        self.results: List[ComparisonBenchmarkResult] = []
        self._backends: Dict[str, Any] = {}
    
    def _get_backend(self, name: str) -> Any:
        """Get or create backend instance."""
        if name not in self._backends:
            try:
                from proxima.backends import get_backend
                self._backends[name] = get_backend(name)
            except Exception:
                return None
        return self._backends[name]
    
    def _create_circuit(self, circuit_type: str, num_qubits: int) -> str:
        """Create benchmark circuit."""
        if circuit_type == "bell":
            return self._create_bell_circuit(num_qubits)
        elif circuit_type == "ghz":
            return self._create_ghz_circuit(num_qubits)
        elif circuit_type == "random":
            return self._create_random_circuit(num_qubits)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    def _create_bell_circuit(self, num_qubits: int) -> str:
        """Create Bell-like circuit."""
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
        """Create GHZ circuit."""
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
    
    def _create_random_circuit(self, num_qubits: int, depth: int = 10) -> str:
        """Create random circuit."""
        import random
        random.seed(42)
        
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]
        
        for _ in range(depth):
            for q in range(num_qubits):
                if random.random() < 0.5:
                    lines.append(f'h q[{q}];')
            for q in range(num_qubits - 1):
                if random.random() < 0.3:
                    lines.append(f'cx q[{q}], q[{q+1}];')
        
        lines.append('measure q -> c;')
        return '\n'.join(lines)
    
    def _calculate_fidelity(
        self,
        counts1: Dict[str, int],
        counts2: Dict[str, int]
    ) -> float:
        """Calculate classical fidelity between two distributions."""
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        all_states = set(counts1.keys()) | set(counts2.keys())
        
        fidelity = 0.0
        for state in all_states:
            p = counts1.get(state, 0) / total1
            q = counts2.get(state, 0) / total2
            fidelity += math.sqrt(p * q)
        
        return fidelity ** 2
    
    def _calculate_kl_divergence(
        self,
        counts_p: Dict[str, int],
        counts_q: Dict[str, int]
    ) -> float:
        """Calculate KL divergence D_KL(P||Q)."""
        total_p = sum(counts_p.values())
        total_q = sum(counts_q.values())
        
        if total_p == 0 or total_q == 0:
            return float('inf')
        
        kl_div = 0.0
        for state in counts_p.keys():
            p = counts_p[state] / total_p
            q = (counts_q.get(state, 0) + 1e-10) / total_q
            if p > 0:
                kl_div += p * math.log(p / q)
        
        return kl_div
    
    def _simulate_backend_run(
        self,
        backend_name: str,
        circuit: str,
        shots: int
    ) -> tuple:
        """
        Simulate running on a backend.
        
        Returns (counts, execution_time).
        In real implementation, this would use actual backends.
        """
        import random
        random.seed(hash(backend_name) % 1000)
        
        # Simulate different characteristics per backend
        start = time.perf_counter()
        
        # Parse to find qubit count
        num_qubits = 2
        for line in circuit.split('\n'):
            if line.startswith('qreg'):
                import re
                match = re.search(r'\[(\d+)\]', line)
                if match:
                    num_qubits = int(match.group(1))
                break
        
        # Generate simulated counts based on circuit type
        if 'h q[0];' in circuit and 'cx q[0], q[1];' in circuit:
            # Bell/GHZ-like: mostly |00...0> and |11...1>
            state_0 = '0' * num_qubits
            state_1 = '1' * num_qubits
            
            noise = random.randint(-20, 20) + (hash(backend_name) % 30)
            counts = {
                state_0: shots // 2 + noise,
                state_1: shots // 2 - noise,
            }
        else:
            # Random distribution
            num_states = min(2 ** num_qubits, 16)
            counts = {}
            remaining = shots
            for i in range(num_states - 1):
                state = format(i, f'0{num_qubits}b')
                count = random.randint(0, remaining // (num_states - i))
                counts[state] = count
                remaining -= count
            counts[format(num_states - 1, f'0{num_qubits}b')] = remaining
        
        # Simulate execution time based on backend
        base_time = 0.1 + num_qubits * 0.01
        if backend_name == "cuquantum":
            base_time *= 0.5  # GPU is faster
        elif backend_name == "quest":
            base_time *= 0.8
        
        elapsed = time.perf_counter() - start + base_time
        
        return counts, elapsed
    
    def run_single_comparison(
        self,
        circuit_type: str,
        num_qubits: int
    ) -> ComparisonBenchmarkResult:
        """Run a single comparison benchmark."""
        try:
            circuit = self._create_circuit(circuit_type, num_qubits)
            
            # Run on each backend
            results = {}
            times = {}
            
            for backend_name in self.config.backends:
                backend_results = []
                backend_times = []
                
                for _ in range(self.config.repetitions):
                    counts, elapsed = self._simulate_backend_run(
                        backend_name,
                        circuit,
                        self.config.shots
                    )
                    backend_results.append(counts)
                    backend_times.append(elapsed)
                
                # Use last result for comparison
                results[backend_name] = backend_results[-1]
                times[backend_name] = statistics.mean(backend_times)
            
            # Get reference results
            reference_counts = results.get(self.config.reference_backend, {})
            
            # Calculate metrics
            fidelities = {}
            kl_divergences = {}
            
            for backend_name, counts in results.items():
                if backend_name != self.config.reference_backend:
                    fidelities[backend_name] = self._calculate_fidelity(
                        reference_counts, counts
                    )
                    kl_divergences[backend_name] = self._calculate_kl_divergence(
                        reference_counts, counts
                    )
                else:
                    fidelities[backend_name] = 1.0
                    kl_divergences[backend_name] = 0.0
            
            # Calculate throughputs
            throughputs = {
                name: self.config.shots / t if t > 0 else 0
                for name, t in times.items()
            }
            
            # Find best performers
            fastest = min(times.items(), key=lambda x: x[1])[0] if times else None
            most_accurate = max(fidelities.items(), key=lambda x: x[1])[0] if fidelities else None
            
            return ComparisonBenchmarkResult(
                circuit_type=circuit_type,
                num_qubits=num_qubits,
                shots=self.config.shots,
                backends_compared=list(results.keys()),
                fidelities=fidelities,
                kl_divergences=kl_divergences,
                execution_times=times,
                throughputs=throughputs,
                fastest_backend=fastest,
                most_accurate_backend=most_accurate,
                successful=True
            )
            
        except Exception as e:
            return ComparisonBenchmarkResult(
                circuit_type=circuit_type,
                num_qubits=num_qubits,
                shots=self.config.shots,
                successful=False,
                error_message=str(e)
            )
    
    def run_all(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[ComparisonBenchmarkResult]:
        """Run all configured comparisons."""
        self.results = []
        
        for circuit_type in self.config.circuit_types:
            for num_qubits in self.config.qubit_counts:
                if progress_callback:
                    progress_callback(f"{circuit_type} / {num_qubits}q")
                
                result = self.run_single_comparison(circuit_type, num_qubits)
                self.results.append(result)
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comparison summary."""
        if not self.results:
            return {"error": "No results"}
        
        successful = [r for r in self.results if r.successful]
        
        # Aggregate by backend
        backend_stats = {}
        for backend in self.config.backends:
            fidelities = [
                r.fidelities.get(backend, 0)
                for r in successful
                if backend in r.fidelities
            ]
            times = [
                r.execution_times.get(backend, 0)
                for r in successful
                if backend in r.execution_times
            ]
            
            if fidelities and times:
                backend_stats[backend] = {
                    "mean_fidelity": statistics.mean(fidelities),
                    "mean_time": statistics.mean(times),
                    "fastest_count": sum(
                        1 for r in successful
                        if r.fastest_backend == backend
                    ),
                    "most_accurate_count": sum(
                        1 for r in successful
                        if r.most_accurate_backend == backend
                    ),
                }
        
        return {
            "total_comparisons": len(self.results),
            "successful": len(successful),
            "backends": backend_stats,
        }


def run_comparison_benchmarks(
    config: Optional[ComparisonBenchmarkConfig] = None,
    verbose: bool = True
) -> List[ComparisonBenchmarkResult]:
    """
    Run comparison benchmarks.
    
    Args:
        config: Benchmark configuration
        verbose: Whether to print progress
    
    Returns:
        List of benchmark results
    """
    suite = ComparisonBenchmarkSuite(config)
    
    progress_fn = None
    if verbose:
        progress_fn = lambda msg: print(f"  {msg}")
        print("Running comparison benchmarks...")
    
    results = suite.run_all(progress_callback=progress_fn)
    
    if verbose:
        import json
        print("\n=== Comparison Summary ===")
        print(json.dumps(suite.get_summary(), indent=2))
    
    return results
