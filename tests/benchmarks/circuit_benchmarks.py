"""
Circuit Performance Benchmarks

Benchmarks for circuit compilation, optimization, and transformation.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone


@dataclass
class CircuitBenchmarkConfig:
    """Configuration for circuit benchmarks."""
    
    # Qubit configurations
    qubit_counts: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20])
    
    # Circuit depths to test
    circuit_depths: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    
    # Number of repetitions
    repetitions: int = 5
    
    # Operations to benchmark
    operations: List[str] = field(default_factory=lambda: [
        "parse",
        "validate",
        "optimize",
        "compile",
        "transpile"
    ])
    
    # Target backends for compilation
    target_backends: List[str] = field(default_factory=lambda: [
        "cirq",
        "qiskit_aer"
    ])


@dataclass
class CircuitBenchmarkResult:
    """Result of a circuit benchmark."""
    
    operation: str
    num_qubits: int
    circuit_depth: int
    target_backend: Optional[str]
    
    # Timing
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    
    # Circuit metrics (for optimization benchmarks)
    original_gate_count: Optional[int] = None
    optimized_gate_count: Optional[int] = None
    gate_reduction_percent: Optional[float] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    successful: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "num_qubits": self.num_qubits,
            "circuit_depth": self.circuit_depth,
            "target_backend": self.target_backend,
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "original_gate_count": self.original_gate_count,
            "optimized_gate_count": self.optimized_gate_count,
            "gate_reduction_percent": self.gate_reduction_percent,
            "timestamp": self.timestamp,
            "successful": self.successful,
            "error_message": self.error_message,
        }


class CircuitBenchmarkSuite:
    """
    Suite of benchmarks for circuit operations.
    
    Measures:
    - Circuit parsing time
    - Validation time
    - Optimization effectiveness and time
    - Compilation time to different backends
    - Transpilation time
    """
    
    def __init__(self, config: Optional[CircuitBenchmarkConfig] = None):
        """Initialize benchmark suite."""
        self.config = config or CircuitBenchmarkConfig()
        self.results: List[CircuitBenchmarkResult] = []
    
    def _generate_circuit(self, num_qubits: int, depth: int) -> str:
        """Generate a random circuit with specified parameters."""
        import random
        
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
        ]
        
        single_gates = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg']
        
        random.seed(42)  # Reproducible
        
        for d in range(depth):
            # Add random single-qubit gates
            for q in range(num_qubits):
                if random.random() < 0.6:
                    gate = random.choice(single_gates)
                    lines.append(f'{gate} q[{q}];')
            
            # Add some two-qubit gates
            for q in range(num_qubits - 1):
                if random.random() < 0.4:
                    lines.append(f'cx q[{q}], q[{q+1}];')
        
        lines.append('measure q -> c;')
        return '\n'.join(lines)
    
    def _count_gates(self, circuit: str) -> int:
        """Count gates in a circuit."""
        gate_count = 0
        for line in circuit.split('\n'):
            line = line.strip()
            if line and not line.startswith(('OPENQASM', 'include', 'qreg', 'creg', '//')):
                if any(line.startswith(g) for g in ['h ', 'x ', 'y ', 'z ', 's ', 't ', 'cx ', 'cz ', 'swap ', 'measure', 'rx', 'ry', 'rz', 'cp']):
                    gate_count += 1
        return gate_count
    
    def _benchmark_parse(self, circuit: str) -> List[float]:
        """Benchmark circuit parsing."""
        times = []
        for _ in range(self.config.repetitions):
            start = time.perf_counter()
            # Simulate parsing
            lines = circuit.split('\n')
            parsed = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('//'):
                    parsed.append(line)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        return times
    
    def _benchmark_validate(self, circuit: str, num_qubits: int) -> List[float]:
        """Benchmark circuit validation."""
        times = []
        for _ in range(self.config.repetitions):
            start = time.perf_counter()
            # Simulate validation
            errors = []
            for line in circuit.split('\n'):
                line = line.strip()
                if line.startswith('cx '):
                    # Check qubit indices
                    pass
                elif line.startswith('qreg'):
                    # Validate qubit register
                    pass
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        return times
    
    def _benchmark_optimize(self, circuit: str) -> tuple:
        """Benchmark circuit optimization."""
        times = []
        original_count = self._count_gates(circuit)
        optimized_count = original_count  # Placeholder
        
        for _ in range(self.config.repetitions):
            start = time.perf_counter()
            # Simulate optimization passes
            lines = circuit.split('\n')
            optimized = []
            prev_gate = None
            
            for line in lines:
                # Simple optimization: remove consecutive inverse gates
                if prev_gate and self._are_inverse(prev_gate, line):
                    optimized.pop()
                    prev_gate = None
                else:
                    optimized.append(line)
                    prev_gate = line
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            optimized_count = self._count_gates('\n'.join(optimized))
        
        return times, original_count, optimized_count
    
    def _are_inverse(self, gate1: str, gate2: str) -> bool:
        """Check if two gates are inverses."""
        inverse_pairs = [
            ('h ', 'h '),
            ('x ', 'x '),
            ('y ', 'y '),
            ('z ', 'z '),
            ('s ', 'sdg '),
            ('t ', 'tdg '),
        ]
        for g1, g2 in inverse_pairs:
            if gate1.startswith(g1) and gate2.startswith(g2):
                # Check same qubit
                q1 = gate1.split()[1] if len(gate1.split()) > 1 else ''
                q2 = gate2.split()[1] if len(gate2.split()) > 1 else ''
                if q1 == q2:
                    return True
        return False
    
    def _benchmark_compile(self, circuit: str, target: str) -> List[float]:
        """Benchmark circuit compilation."""
        times = []
        for _ in range(self.config.repetitions):
            start = time.perf_counter()
            # Simulate compilation to target backend
            # In real implementation, this would use actual compilation
            compiled = circuit  # Placeholder
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        return times
    
    def _benchmark_transpile(self, circuit: str, target: str) -> List[float]:
        """Benchmark circuit transpilation."""
        times = []
        for _ in range(self.config.repetitions):
            start = time.perf_counter()
            # Simulate transpilation
            transpiled = circuit  # Placeholder
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        return times
    
    def run_single_benchmark(
        self,
        operation: str,
        num_qubits: int,
        circuit_depth: int,
        target_backend: Optional[str] = None
    ) -> CircuitBenchmarkResult:
        """Run a single benchmark."""
        try:
            circuit = self._generate_circuit(num_qubits, circuit_depth)
            
            if operation == "parse":
                times = self._benchmark_parse(circuit)
            elif operation == "validate":
                times = self._benchmark_validate(circuit, num_qubits)
            elif operation == "optimize":
                times, orig_count, opt_count = self._benchmark_optimize(circuit)
                reduction = ((orig_count - opt_count) / orig_count * 100) if orig_count > 0 else 0
                
                return CircuitBenchmarkResult(
                    operation=operation,
                    num_qubits=num_qubits,
                    circuit_depth=circuit_depth,
                    target_backend=target_backend,
                    mean_time=statistics.mean(times),
                    std_time=statistics.stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    original_gate_count=orig_count,
                    optimized_gate_count=opt_count,
                    gate_reduction_percent=reduction,
                    successful=True
                )
            elif operation == "compile":
                times = self._benchmark_compile(circuit, target_backend or "cirq")
            elif operation == "transpile":
                times = self._benchmark_transpile(circuit, target_backend or "cirq")
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return CircuitBenchmarkResult(
                operation=operation,
                num_qubits=num_qubits,
                circuit_depth=circuit_depth,
                target_backend=target_backend,
                mean_time=statistics.mean(times),
                std_time=statistics.stdev(times) if len(times) > 1 else 0,
                min_time=min(times),
                max_time=max(times),
                successful=True
            )
            
        except Exception as e:
            return CircuitBenchmarkResult(
                operation=operation,
                num_qubits=num_qubits,
                circuit_depth=circuit_depth,
                target_backend=target_backend,
                mean_time=0,
                std_time=0,
                min_time=0,
                max_time=0,
                successful=False,
                error_message=str(e)
            )
    
    def run_all(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[CircuitBenchmarkResult]:
        """Run all configured benchmarks."""
        self.results = []
        
        for operation in self.config.operations:
            for num_qubits in self.config.qubit_counts:
                for depth in self.config.circuit_depths:
                    if operation in ["compile", "transpile"]:
                        for target in self.config.target_backends:
                            if progress_callback:
                                progress_callback(
                                    f"{operation} / {num_qubits}q / depth={depth} / {target}"
                                )
                            result = self.run_single_benchmark(
                                operation, num_qubits, depth, target
                            )
                            self.results.append(result)
                    else:
                        if progress_callback:
                            progress_callback(
                                f"{operation} / {num_qubits}q / depth={depth}"
                            )
                        result = self.run_single_benchmark(
                            operation, num_qubits, depth
                        )
                        self.results.append(result)
        
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if not self.results:
            return {"error": "No results"}
        
        by_operation = {}
        for result in self.results:
            if result.operation not in by_operation:
                by_operation[result.operation] = []
            by_operation[result.operation].append(result)
        
        summary = {
            "total_benchmarks": len(self.results),
            "operations": {}
        }
        
        for op, results in by_operation.items():
            successful = [r for r in results if r.successful]
            if successful:
                summary["operations"][op] = {
                    "count": len(results),
                    "successful": len(successful),
                    "mean_time": statistics.mean(r.mean_time for r in successful),
                }
                if op == "optimize" and any(r.gate_reduction_percent for r in successful):
                    summary["operations"][op]["mean_reduction"] = statistics.mean(
                        r.gate_reduction_percent for r in successful 
                        if r.gate_reduction_percent is not None
                    )
        
        return summary


def run_circuit_benchmarks(
    config: Optional[CircuitBenchmarkConfig] = None,
    verbose: bool = True
) -> List[CircuitBenchmarkResult]:
    """
    Run circuit benchmarks.
    
    Args:
        config: Benchmark configuration
        verbose: Whether to print progress
    
    Returns:
        List of benchmark results
    """
    suite = CircuitBenchmarkSuite(config)
    
    progress_fn = None
    if verbose:
        progress_fn = lambda msg: print(f"  {msg}")
        print("Running circuit benchmarks...")
    
    results = suite.run_all(progress_callback=progress_fn)
    
    if verbose:
        import json
        print("\n=== Circuit Benchmark Summary ===")
        print(json.dumps(suite.get_summary(), indent=2))
    
    return results
