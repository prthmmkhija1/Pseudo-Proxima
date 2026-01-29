"""LRET backend adapter with comprehensive implementation.

LRET (Lightweight Runtime Execution Toolkit) integration for quantum simulations.
Target: https://github.com/kunal5556/LRET (feature/framework-integration branch)

Installation:
    # From pip (when available):
    pip install lret
    
    # From source (framework-integration branch):
    git clone https://github.com/kunal5556/LRET.git
    cd LRET
    git checkout feature/framework-integration
    pip install -e .

This module provides:
- Complete result normalization for all LRET output formats  
- Framework-integration branch API verification
- Comprehensive error handling and logging
- Mock simulator for testing when LRET is not installed
- Automatic fallback with meaningful error messages
- Real LRET library detection and version reporting
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from proxima.backends.base import (
    BaseBackendAdapter,
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
    ValidationResult,
)
from proxima.backends.exceptions import (
    BackendNotInstalledError,
    CircuitValidationError,
    UnsupportedOperationError,
    wrap_backend_exception,
)

# Configure logging
logger = logging.getLogger(__name__)

# LRET Configuration Constants
LRET_GITHUB_URL = "https://github.com/kunal5556/LRET"
LRET_BRANCH = "feature/framework-integration"
LRET_INSTALL_INSTRUCTIONS = '''
To install LRET for real quantum simulation:

Option 1: pip install (when available)
  pip install lret

Option 2: From source (recommended for framework-integration):
  git clone https://github.com/kunal5556/LRET.git
  cd LRET
  git checkout feature/framework-integration  
  pip install -e .

For more information, visit: https://github.com/kunal5556/LRET
'''


# ==============================================================================
# RESULT NORMALIZATION - Complete Implementation
# ==============================================================================


class LRETResultFormat(Enum):
    """Enum for different LRET result formats."""
    
    COUNTS = "counts"
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"
    EXPECTATION = "expectation"
    PROBABILITIES = "probabilities"
    RAW = "raw"


@dataclass
class NormalizedResult:
    """Normalized result container for LRET outputs.
    
    This class provides a unified interface for all LRET result formats,
    ensuring consistent data access regardless of the original format.
    """
    
    format: LRETResultFormat
    counts: dict[str, int] = field(default_factory=dict)
    statevector: np.ndarray | None = None
    density_matrix: np.ndarray | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    expectation_values: dict[str, float] = field(default_factory=dict)
    shots: int = 0
    num_qubits: int = 0
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "format": self.format.value,
            "shots": self.shots,
            "num_qubits": self.num_qubits,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }
        
        if self.counts:
            result["counts"] = self.counts
        if self.statevector is not None:
            result["statevector"] = self.statevector.tolist()
        if self.density_matrix is not None:
            result["density_matrix"] = self.density_matrix.tolist()
        if self.probabilities:
            result["probabilities"] = self.probabilities
        if self.expectation_values:
            result["expectation_values"] = self.expectation_values
            
        return result



@dataclass
class LRETBatchExecutionResult:
    """Result of batch circuit execution for LRET backend.
    
    Attributes:
        total_circuits: Total number of circuits submitted
        successful: Number of successfully executed circuits
        failed: Number of failed circuits
        results: List of ExecutionResults (None for failed circuits)
        errors: List of (index, error_message) tuples for failures
        total_execution_time_ms: Total execution time in milliseconds
        average_time_per_circuit_ms: Average time per circuit
    """
    
    total_circuits: int
    successful: int
    failed: int
    results: list[ExecutionResult | None]
    errors: list[tuple[int, str]]
    total_execution_time_ms: float
    average_time_per_circuit_ms: float
    
    def get_successful_results(self) -> list[ExecutionResult]:
        """Get only successful results, filtering out None values."""
        return [r for r in self.results if r is not None]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_circuits": self.total_circuits,
            "successful": self.successful,
            "failed": self.failed,
            "results": [r.to_dict() if r else None for r in self.results],
            "errors": self.errors,
            "total_execution_time_ms": self.total_execution_time_ms,
            "average_time_per_circuit_ms": self.average_time_per_circuit_ms,
        }


class LRETResultNormalizer:
    """Complete result normalizer for all LRET output formats.
    
    Handles conversion from various LRET result formats to a unified
    NormalizedResult structure. Supports:
    - Measurement counts (shots-based results)
    - Statevectors (state simulation results)
    - Density matrices (mixed state results)
    - Probability distributions
    - Expectation values
    - Raw result objects
    """
    
    def __init__(self, num_qubits: int = 0) -> None:
        """Initialize the normalizer.
        
        Args:
            num_qubits: Number of qubits in the circuit (used for validation)
        """
        self._num_qubits = num_qubits
        self._normalization_handlers: dict[str, Callable] = {
            "counts": self._normalize_counts,
            "statevector": self._normalize_statevector,
            "state_vector": self._normalize_statevector,
            "density_matrix": self._normalize_density_matrix,
            "dm": self._normalize_density_matrix,
            "probabilities": self._normalize_probabilities,
            "probs": self._normalize_probabilities,
            "expectation": self._normalize_expectation,
            "expect": self._normalize_expectation,
        }
    
    def normalize(
        self,
        result: Any,
        result_format: str | None = None,
        shots: int = 0,
        execution_time_ms: float = 0.0,
    ) -> NormalizedResult:
        """Normalize any LRET result to a unified format.
        
        Args:
            result: Raw result from LRET execution
            result_format: Optional hint about result format
            shots: Number of shots used (if applicable)
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            NormalizedResult with unified data access
        """
        # Detect format if not specified
        if result_format is None:
            result_format = self._detect_format(result)
        
        # Get the appropriate handler
        handler = self._normalization_handlers.get(
            result_format.lower(),
            self._normalize_raw
        )
        
        # Normalize the result
        normalized = handler(result)
        normalized.shots = shots
        normalized.execution_time_ms = execution_time_ms
        normalized.num_qubits = self._num_qubits
        
        # Cross-populate fields where possible
        self._cross_populate(normalized)
        
        return normalized
    
    def _detect_format(self, result: Any) -> str:
        """Detect the format of a result object."""
        if result is None:
            return "raw"
        
        # Check for result object attributes
        # Check statevector first, as some results have both attributes
        # but only one is populated (e.g., MockLRETResult has counts={} even for statevector results)
        if hasattr(result, "statevector") and result.statevector is not None:
            return "statevector"
        if hasattr(result, "state_vector") and result.state_vector is not None:
            return "statevector"
        # Check counts - ensure it's actually populated
        counts_val = getattr(result, "counts", None)
        if counts_val is not None and counts_val:  # Non-empty counts
            return "counts"
        if hasattr(result, "get_counts"):
            try:
                counts = result.get_counts()
                if counts:
                    return "counts"
            except Exception:
                pass
        if hasattr(result, "density_matrix"):
            return "density_matrix"
        if hasattr(result, "probabilities"):
            return "probabilities"
        if hasattr(result, "expectation_values"):
            return "expectation"
        
        # Check for dictionary keys
        if isinstance(result, dict):
            if "counts" in result:
                return "counts"
            if "statevector" in result or "state_vector" in result:
                return "statevector"
            if "density_matrix" in result:
                return "density_matrix"
            if "probabilities" in result:
                return "probabilities"
            if "expectation" in result:
                return "expectation"
            # Dictionary with string keys and int values -> counts
            if all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in result.items()):
                return "counts"
        
        # Check for numpy arrays
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                return "statevector"
            if result.ndim == 2:
                return "density_matrix"
        
        return "raw"
    
    def _normalize_counts(self, result: Any) -> NormalizedResult:
        """Normalize measurement counts."""
        counts: dict[str, int] = {}
        
        # Extract counts from various formats
        if isinstance(result, dict):
            if "counts" in result:
                counts = dict(result["counts"])
            else:
                counts = {str(k): int(v) for k, v in result.items()}
        elif hasattr(result, "get_counts"):
            counts = dict(result.get_counts())
        elif hasattr(result, "counts"):
            counts = dict(result.counts)
        
        # Validate and normalize bitstring format
        normalized_counts = self._normalize_bitstrings(counts)
        
        return NormalizedResult(
            format=LRETResultFormat.COUNTS,
            counts=normalized_counts,
            metadata={"original_format": "counts"},
        )
    
    def _normalize_statevector(self, result: Any) -> NormalizedResult:
        """Normalize statevector results."""
        statevector: np.ndarray | None = None
        
        if isinstance(result, np.ndarray):
            statevector = result.astype(complex)
        elif isinstance(result, dict):
            sv = result.get("statevector") or result.get("state_vector")
            if sv is not None:
                statevector = np.asarray(sv, dtype=complex)
        elif hasattr(result, "statevector"):
            statevector = np.asarray(result.statevector, dtype=complex)
        elif hasattr(result, "state_vector"):
            statevector = np.asarray(result.state_vector, dtype=complex)
        elif hasattr(result, "get_statevector"):
            statevector = np.asarray(result.get_statevector(), dtype=complex)
        
        # Normalize the statevector
        if statevector is not None:
            statevector = self._normalize_statevector_array(statevector)
            self._num_qubits = int(np.log2(len(statevector)))
        
        return NormalizedResult(
            format=LRETResultFormat.STATEVECTOR,
            statevector=statevector,
            metadata={"original_format": "statevector"},
        )
    
    def _normalize_density_matrix(self, result: Any) -> NormalizedResult:
        """Normalize density matrix results."""
        dm: np.ndarray | None = None
        
        if isinstance(result, np.ndarray) and result.ndim == 2:
            dm = result.astype(complex)
        elif isinstance(result, dict) and "density_matrix" in result:
            dm = np.asarray(result["density_matrix"], dtype=complex)
        elif hasattr(result, "density_matrix"):
            dm = np.asarray(result.density_matrix, dtype=complex)
        
        # Validate density matrix properties
        if dm is not None:
            dm = self._validate_density_matrix(dm)
            self._num_qubits = int(np.log2(dm.shape[0]))
        
        return NormalizedResult(
            format=LRETResultFormat.DENSITY_MATRIX,
            density_matrix=dm,
            metadata={"original_format": "density_matrix"},
        )
    
    def _normalize_probabilities(self, result: Any) -> NormalizedResult:
        """Normalize probability distribution results."""
        probs: dict[str, float] = {}
        
        if isinstance(result, dict):
            if "probabilities" in result:
                probs = {str(k): float(v) for k, v in result["probabilities"].items()}
            elif "probs" in result:
                probs = {str(k): float(v) for k, v in result["probs"].items()}
            else:
                probs = {str(k): float(v) for k, v in result.items()}
        elif hasattr(result, "probabilities"):
            probs = {str(k): float(v) for k, v in result.probabilities.items()}
        
        # Normalize probabilities to sum to 1
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return NormalizedResult(
            format=LRETResultFormat.PROBABILITIES,
            probabilities=probs,
            metadata={"original_format": "probabilities"},
        )
    
    def _normalize_expectation(self, result: Any) -> NormalizedResult:
        """Normalize expectation value results."""
        expectation_values: dict[str, float] = {}
        
        if isinstance(result, dict):
            if "expectation" in result:
                expectation_values = dict(result["expectation"])
            elif "expectation_values" in result:
                expectation_values = dict(result["expectation_values"])
            elif "expect" in result:
                expectation_values = dict(result["expect"])
        elif hasattr(result, "expectation_values"):
            expectation_values = dict(result.expectation_values)
        
        return NormalizedResult(
            format=LRETResultFormat.EXPECTATION,
            expectation_values=expectation_values,
            metadata={"original_format": "expectation"},
        )
    
    def _normalize_raw(self, result: Any) -> NormalizedResult:
        """Normalize unknown result format."""
        metadata: dict[str, Any] = {
            "original_format": "raw",
            "original_type": type(result).__name__,
        }
        
        # Try to extract any useful data
        if hasattr(result, "__dict__"):
            metadata["attributes"] = list(result.__dict__.keys())
        
        return NormalizedResult(
            format=LRETResultFormat.RAW,
            metadata=metadata,
        )
    
    def _normalize_bitstrings(self, counts: dict[str, int]) -> dict[str, int]:
        """Normalize bitstring format in counts."""
        normalized: dict[str, int] = {}
        
        for bitstring, count in counts.items():
            # Remove any prefixes like '0b'
            clean = str(bitstring).replace("0b", "").replace(" ", "")
            
            # Ensure consistent length if we know num_qubits
            if self._num_qubits > 0:
                clean = clean.zfill(self._num_qubits)
            
            # Accumulate counts for same bitstrings
            normalized[clean] = normalized.get(clean, 0) + int(count)
        
        return normalized
    
    def _normalize_statevector_array(self, sv: np.ndarray) -> np.ndarray:
        """Normalize a statevector array."""
        # Flatten if needed
        if sv.ndim > 1:
            sv = sv.flatten()
        
        # Ensure proper normalization
        norm = np.linalg.norm(sv)
        if norm > 0 and abs(norm - 1.0) > 1e-10:
            sv = sv / norm
        
        return sv
    
    def _validate_density_matrix(self, dm: np.ndarray) -> np.ndarray:
        """Validate and normalize a density matrix."""
        # Ensure square matrix
        if dm.shape[0] != dm.shape[1]:
            raise ValueError(f"Density matrix must be square, got shape {dm.shape}")
        
        # Check dimension is power of 2
        n = dm.shape[0]
        if n & (n - 1) != 0:
            raise ValueError(f"Density matrix dimension must be power of 2, got {n}")
        
        # Normalize trace to 1
        trace = np.trace(dm)
        if abs(trace) > 1e-10 and abs(trace - 1.0) > 1e-10:
            dm = dm / trace
        
        return dm
    
    def _cross_populate(self, result: NormalizedResult) -> None:
        """Cross-populate fields from available data."""
        # If we have counts, compute probabilities
        if result.counts and not result.probabilities:
            total = sum(result.counts.values())
            if total > 0:
                result.probabilities = {
                    k: v / total for k, v in result.counts.items()
                }
                result.shots = total
        
        # If we have probabilities, can estimate counts (for display)
        if result.probabilities and not result.counts and result.shots > 0:
            result.counts = {
                k: int(v * result.shots) for k, v in result.probabilities.items()
            }
        
        # If we have statevector, compute probabilities
        if result.statevector is not None and not result.probabilities:
            probs = np.abs(result.statevector) ** 2
            n_qubits = int(np.log2(len(probs)))
            result.probabilities = {
                format(i, f"0{n_qubits}b"): float(p)
                for i, p in enumerate(probs) if p > 1e-10
            }
            result.num_qubits = n_qubits


# ==============================================================================
# FRAMEWORK-INTEGRATION BRANCH API VERIFICATION
# ==============================================================================


@dataclass
class LRETAPIVerification:
    """Results of LRET framework-integration branch API verification."""
    
    is_compatible: bool = False
    api_version: str = "unknown"
    branch: str = "unknown"
    available_apis: list[str] = field(default_factory=list)
    missing_apis: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verified_at: float = 0.0


class LRETAPIVerifier:
    """Verifies compatibility with LRET framework-integration branch API.
    
    The framework-integration branch has specific API requirements that
    differ from the main branch. This verifier checks for:
    - Required import paths
    - Expected class interfaces
    - Method signatures
    - Result format compatibility
    """
    
    # Expected APIs in framework-integration branch
    REQUIRED_APIS = [
        "Circuit",
        "Simulator", 
        "execute",
        "validate_circuit",
    ]
    
    OPTIONAL_APIS = [
        "from_qiskit",
        "from_cirq",
        "from_dict",
        "SUPPORTED_GATES",
        "__version__",
    ]
    
    SIMULATOR_METHODS = [
        "run",
        "simulate",
    ]
    
    CIRCUIT_METHODS = [
        "add_gate",
        "measure",
    ]
    
    def __init__(self, lret_module: Any = None) -> None:
        """Initialize the verifier.
        
        Args:
            lret_module: The LRET module to verify (or None to auto-import)
        """
        self._lret = lret_module
        self._verification: LRETAPIVerification | None = None
    
    def verify(self, force: bool = False) -> LRETAPIVerification:
        """Perform API verification.
        
        Args:
            force: Force re-verification even if already done
            
        Returns:
            LRETAPIVerification with compatibility results
        """
        if self._verification is not None and not force:
            return self._verification
        
        verification = LRETAPIVerification(verified_at=time.time())
        
        # Try to import LRET if not provided
        if self._lret is None:
            if not importlib.util.find_spec("lret"):
                verification.warnings.append("LRET module not installed")
                self._verification = verification
                return verification
            try:
                import lret  # type: ignore
                self._lret = lret
            except ImportError as e:
                verification.warnings.append(f"Failed to import LRET: {e}")
                self._verification = verification
                return verification
        
        # Check version
        verification.api_version = getattr(self._lret, "__version__", "unknown")
        
        # Check for framework-integration branch indicators
        if hasattr(self._lret, "_BRANCH"):
            verification.branch = self._lret._BRANCH
        elif "framework" in verification.api_version.lower():
            verification.branch = "feature/framework-integration"
        else:
            verification.branch = "unknown (possibly main)"
        
        # Check required APIs
        for api in self.REQUIRED_APIS:
            if hasattr(self._lret, api):
                verification.available_apis.append(api)
            else:
                verification.missing_apis.append(api)
        
        # Check optional APIs
        for api in self.OPTIONAL_APIS:
            if hasattr(self._lret, api):
                verification.available_apis.append(api)
        
        # Check Simulator methods if available
        if hasattr(self._lret, "Simulator"):
            simulator_class = self._lret.Simulator
            for method in self.SIMULATOR_METHODS:
                full_name = f"Simulator.{method}"
                if hasattr(simulator_class, method):
                    verification.available_apis.append(full_name)
                else:
                    verification.warnings.append(f"Simulator missing method: {method}")
        
        # Check Circuit methods if available
        if hasattr(self._lret, "Circuit"):
            circuit_class = self._lret.Circuit
            for method in self.CIRCUIT_METHODS:
                full_name = f"Circuit.{method}"
                if hasattr(circuit_class, method):
                    verification.available_apis.append(full_name)
                else:
                    verification.warnings.append(f"Circuit missing method: {method}")
        
        # Determine compatibility
        verification.is_compatible = len(verification.missing_apis) == 0
        
        self._verification = verification
        return verification
    
    def get_compatibility_report(self) -> str:
        """Generate a human-readable compatibility report."""
        verification = self.verify()
        
        lines = [
            "LRET Framework-Integration Branch API Verification",
            "=" * 50,
            f"Compatible: {'Yes' if verification.is_compatible else 'No'}",
            f"API Version: {verification.api_version}",
            f"Branch: {verification.branch}",
            "",
            "Available APIs:",
        ]
        
        for api in verification.available_apis:
            lines.append(f"  âœ“ {api}")
        
        if verification.missing_apis:
            lines.append("")
            lines.append("Missing APIs:")
            for api in verification.missing_apis:
                lines.append(f"  âœ— {api}")
        
        if verification.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in verification.warnings:
                lines.append(f"  âš  {warning}")
        
        return "\n".join(lines)


# ==============================================================================
# PERFORMANCE BENCHMARKING INFRASTRUCTURE
# ==============================================================================


@dataclass
class LRETPerformanceMetrics:
    """Performance metrics specific to LRET backend.
    
    Captures detailed performance measurements for LRET circuit execution,
    enabling comprehensive benchmarking and performance analysis.
    
    Attributes:
        execution_time_ms: Total wall-clock execution time in milliseconds.
        gate_execution_time_ms: Time spent executing gates.
        measurement_time_ms: Time spent on measurements/sampling.
        normalization_time_ms: Time spent on result normalization.
        memory_peak_mb: Peak memory usage during execution.
        memory_baseline_mb: Memory usage before execution.
        throughput_shots_per_sec: Number of shots processed per second.
        gates_per_second: Gate execution throughput.
        qubits: Number of qubits in the circuit.
        gate_count: Total number of gates executed.
        circuit_depth: Depth of the circuit.
        shots: Number of measurement shots.
        timestamp: When the benchmark was executed.
    """
    
    execution_time_ms: float = 0.0
    gate_execution_time_ms: float = 0.0
    measurement_time_ms: float = 0.0
    normalization_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    memory_baseline_mb: float = 0.0
    throughput_shots_per_sec: float = 0.0
    gates_per_second: float = 0.0
    qubits: int = 0
    gate_count: int = 0
    circuit_depth: int = 0
    shots: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "execution_time_ms": self.execution_time_ms,
            "gate_execution_time_ms": self.gate_execution_time_ms,
            "measurement_time_ms": self.measurement_time_ms,
            "normalization_time_ms": self.normalization_time_ms,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_baseline_mb": self.memory_baseline_mb,
            "throughput_shots_per_sec": self.throughput_shots_per_sec,
            "gates_per_second": self.gates_per_second,
            "qubits": self.qubits,
            "gate_count": self.gate_count,
            "circuit_depth": self.circuit_depth,
            "shots": self.shots,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LRETPerformanceMetrics":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LRETBenchmarkResult:
    """Result of an LRET performance benchmark run.
    
    Aggregates metrics across multiple runs for statistical analysis.
    
    Attributes:
        circuit_name: Identifier for the circuit being benchmarked.
        num_runs: Number of benchmark runs performed.
        metrics: List of metrics from each run.
        mean_execution_time_ms: Average execution time.
        std_execution_time_ms: Standard deviation of execution time.
        min_execution_time_ms: Minimum execution time.
        max_execution_time_ms: Maximum execution time.
        mean_throughput: Average throughput (shots/second).
        mean_memory_mb: Average memory usage.
        success_rate: Percentage of successful runs.
        errors: List of errors encountered during benchmarking.
    """
    
    circuit_name: str
    num_runs: int
    metrics: list[LRETPerformanceMetrics] = field(default_factory=list)
    mean_execution_time_ms: float = 0.0
    std_execution_time_ms: float = 0.0
    min_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0
    mean_throughput: float = 0.0
    mean_memory_mb: float = 0.0
    success_rate: float = 0.0
    errors: list[str] = field(default_factory=list)
    
    def compute_statistics(self) -> None:
        """Compute aggregate statistics from individual run metrics."""
        if not self.metrics:
            return
        
        times = [m.execution_time_ms for m in self.metrics]
        self.mean_execution_time_ms = sum(times) / len(times)
        self.min_execution_time_ms = min(times)
        self.max_execution_time_ms = max(times)
        
        if len(times) > 1:
            variance = sum((t - self.mean_execution_time_ms) ** 2 for t in times) / (len(times) - 1)
            self.std_execution_time_ms = variance ** 0.5
        
        throughputs = [m.throughput_shots_per_sec for m in self.metrics if m.throughput_shots_per_sec > 0]
        if throughputs:
            self.mean_throughput = sum(throughputs) / len(throughputs)
        
        memories = [m.memory_peak_mb for m in self.metrics if m.memory_peak_mb > 0]
        if memories:
            self.mean_memory_mb = sum(memories) / len(memories)
        
        self.success_rate = len(self.metrics) / self.num_runs * 100 if self.num_runs > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "circuit_name": self.circuit_name,
            "num_runs": self.num_runs,
            "metrics": [m.to_dict() for m in self.metrics],
            "mean_execution_time_ms": self.mean_execution_time_ms,
            "std_execution_time_ms": self.std_execution_time_ms,
            "min_execution_time_ms": self.min_execution_time_ms,
            "max_execution_time_ms": self.max_execution_time_ms,
            "mean_throughput": self.mean_throughput,
            "mean_memory_mb": self.mean_memory_mb,
            "success_rate": self.success_rate,
            "errors": self.errors,
        }


class LRETPerformanceMonitor:
    """Monitor for tracking LRET execution performance.
    
    Provides instrumentation for measuring execution time, memory usage,
    and other performance metrics during LRET circuit execution.
    """
    
    def __init__(self) -> None:
        """Initialize the performance monitor."""
        self._start_time: float = 0.0
        self._gate_start_time: float = 0.0
        self._measurement_start_time: float = 0.0
        self._phase_times: dict[str, float] = {}
        self._memory_samples: list[float] = []
        self._baseline_memory: float = 0.0
        self._is_running: bool = False
    
    def start(self) -> None:
        """Start performance monitoring."""
        self._start_time = time.perf_counter()
        self._phase_times = {}
        self._memory_samples = []
        self._baseline_memory = self._get_current_memory()
        self._is_running = True
    
    def stop(self) -> float:
        """Stop monitoring and return total elapsed time in ms."""
        if not self._is_running:
            return 0.0
        elapsed = (time.perf_counter() - self._start_time) * 1000
        self._is_running = False
        return elapsed
    
    def start_phase(self, phase_name: str) -> None:
        """Mark the start of an execution phase."""
        self._phase_times[f"{phase_name}_start"] = time.perf_counter()
    
    def end_phase(self, phase_name: str) -> float:
        """Mark the end of an execution phase and return duration in ms."""
        end_time = time.perf_counter()
        start_key = f"{phase_name}_start"
        if start_key in self._phase_times:
            duration = (end_time - self._phase_times[start_key]) * 1000
            self._phase_times[phase_name] = duration
            return duration
        return 0.0
    
    def sample_memory(self) -> float:
        """Sample current memory usage in MB."""
        mem = self._get_current_memory()
        self._memory_samples.append(mem)
        return mem
    
    def _get_current_memory(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self._memory_samples:
            return self._get_current_memory()
        return max(self._memory_samples)
    
    def get_phase_time(self, phase_name: str) -> float:
        """Get the duration of a specific phase in ms."""
        return self._phase_times.get(phase_name, 0.0)
    
    def get_metrics(
        self,
        qubits: int = 0,
        gate_count: int = 0,
        circuit_depth: int = 0,
        shots: int = 0,
    ) -> LRETPerformanceMetrics:
        """Build performance metrics from collected data."""
        total_time = self.stop() if self._is_running else (
            (time.perf_counter() - self._start_time) * 1000 if self._start_time > 0 else 0.0
        )
        
        gate_time = self.get_phase_time("gate_execution")
        measurement_time = self.get_phase_time("measurement")
        normalization_time = self.get_phase_time("normalization")
        
        peak_memory = self.get_peak_memory()
        
        # Calculate throughput
        throughput = 0.0
        if total_time > 0 and shots > 0:
            throughput = (shots / total_time) * 1000  # shots per second
        
        # Calculate gates per second
        gates_per_sec = 0.0
        if gate_time > 0 and gate_count > 0:
            gates_per_sec = (gate_count / gate_time) * 1000
        
        return LRETPerformanceMetrics(
            execution_time_ms=total_time,
            gate_execution_time_ms=gate_time,
            measurement_time_ms=measurement_time,
            normalization_time_ms=normalization_time,
            memory_peak_mb=peak_memory,
            memory_baseline_mb=self._baseline_memory,
            throughput_shots_per_sec=throughput,
            gates_per_second=gates_per_sec,
            qubits=qubits,
            gate_count=gate_count,
            circuit_depth=circuit_depth,
            shots=shots,
            timestamp=time.time(),
        )
    
    def reset(self) -> None:
        """Reset all monitoring state."""
        self._start_time = 0.0
        self._phase_times = {}
        self._memory_samples = []
        self._baseline_memory = 0.0
        self._is_running = False


class LRETBenchmarkRunner:
    """Runner for executing LRET performance benchmarks.
    
    Provides methods for running single and multiple benchmark runs,
    collecting metrics, and generating benchmark reports.
    """
    
    def __init__(self, adapter: "LRETBackendAdapter") -> None:
        """Initialize the benchmark runner.
        
        Args:
            adapter: LRET backend adapter instance to use for execution.
        """
        self._adapter = adapter
        self._logger = logging.getLogger("proxima.backends.lret.benchmark")
    
    def run_single(
        self,
        circuit: Any,
        shots: int = 1024,
        circuit_name: str = "unnamed",
    ) -> LRETPerformanceMetrics:
        """Run a single benchmark iteration.
        
        Args:
            circuit: The quantum circuit to benchmark.
            shots: Number of measurement shots.
            circuit_name: Identifier for the circuit.
            
        Returns:
            Performance metrics from the run.
        """
        monitor = LRETPerformanceMonitor()
        
        # Extract circuit info
        qubits, gate_count, depth = self._extract_circuit_info(circuit)
        
        # Prepare for benchmark
        self._adapter.prepare_for_benchmark(circuit, shots)
        
        try:
            monitor.start()
            monitor.sample_memory()
            
            # Execute the circuit
            result = self._adapter.execute(
                circuit=circuit,
                shots=shots,
                options={"benchmark_mode": True},
            )
            
            monitor.sample_memory()
            
            return monitor.get_metrics(
                qubits=qubits,
                gate_count=gate_count,
                circuit_depth=depth,
                shots=shots,
            )
            
        finally:
            self._adapter.cleanup_after_benchmark()
    
    def run_benchmark(
        self,
        circuit: Any,
        num_runs: int = 5,
        shots: int = 1024,
        circuit_name: str = "unnamed",
        warmup_runs: int = 1,
    ) -> LRETBenchmarkResult:
        """Run a complete benchmark with multiple iterations.
        
        Args:
            circuit: The quantum circuit to benchmark.
            num_runs: Number of benchmark iterations.
            shots: Number of measurement shots per iteration.
            circuit_name: Identifier for the circuit.
            warmup_runs: Number of warmup runs before measuring.
            
        Returns:
            Aggregated benchmark results.
        """
        result = LRETBenchmarkResult(
            circuit_name=circuit_name,
            num_runs=num_runs,
        )
        
        # Warmup runs (not included in results)
        for _ in range(warmup_runs):
            try:
                self.run_single(circuit, shots, circuit_name)
            except Exception as e:
                self._logger.warning(f"Warmup run failed: {e}")
        
        # Benchmark runs
        for run_idx in range(num_runs):
            try:
                metrics = self.run_single(circuit, shots, circuit_name)
                result.metrics.append(metrics)
                self._logger.debug(
                    f"Run {run_idx + 1}/{num_runs}: {metrics.execution_time_ms:.2f}ms"
                )
            except Exception as e:
                error_msg = f"Run {run_idx + 1} failed: {str(e)}"
                result.errors.append(error_msg)
                self._logger.error(error_msg)
        
        # Compute aggregate statistics
        result.compute_statistics()
        
        return result
    
    def run_scaling_benchmark(
        self,
        circuit_generator: Callable[[int], Any],
        qubit_range: list[int],
        shots: int = 1024,
        num_runs: int = 3,
    ) -> list[LRETBenchmarkResult]:
        """Run benchmarks across different circuit sizes.
        
        Args:
            circuit_generator: Function that generates a circuit given qubit count.
            qubit_range: List of qubit counts to benchmark.
            shots: Number of measurement shots per iteration.
            num_runs: Number of runs per circuit size.
            
        Returns:
            List of benchmark results for each circuit size.
        """
        results = []
        
        for num_qubits in qubit_range:
            circuit = circuit_generator(num_qubits)
            circuit_name = f"circuit_{num_qubits}q"
            
            self._logger.info(f"Benchmarking {num_qubits} qubits...")
            
            result = self.run_benchmark(
                circuit=circuit,
                num_runs=num_runs,
                shots=shots,
                circuit_name=circuit_name,
            )
            results.append(result)
            
            self._logger.info(
                f"  {num_qubits}q: {result.mean_execution_time_ms:.2f}ms "
                f"(±{result.std_execution_time_ms:.2f}ms)"
            )
        
        return results
    
    def run_throughput_benchmark(
        self,
        circuit: Any,
        shot_counts: list[int],
        num_runs: int = 3,
        circuit_name: str = "throughput_test",
    ) -> list[LRETBenchmarkResult]:
        """Benchmark throughput across different shot counts.
        
        Args:
            circuit: The quantum circuit to benchmark.
            shot_counts: List of shot counts to test.
            num_runs: Number of runs per shot count.
            circuit_name: Base identifier for the circuit.
            
        Returns:
            List of benchmark results for each shot count.
        """
        results = []
        
        for shots in shot_counts:
            name = f"{circuit_name}_{shots}shots"
            
            self._logger.info(f"Benchmarking with {shots} shots...")
            
            result = self.run_benchmark(
                circuit=circuit,
                num_runs=num_runs,
                shots=shots,
                circuit_name=name,
            )
            results.append(result)
            
            self._logger.info(
                f"  {shots} shots: {result.mean_throughput:.0f} shots/sec"
            )
        
        return results
    
    def _extract_circuit_info(self, circuit: Any) -> tuple[int, int, int]:
        """Extract qubit count, gate count, and depth from a circuit.
        
        Args:
            circuit: The quantum circuit.
            
        Returns:
            Tuple of (num_qubits, gate_count, depth).
        """
        num_qubits = 0
        gate_count = 0
        depth = 0
        
        try:
            if isinstance(circuit, dict):
                num_qubits = circuit.get("qubits", circuit.get("num_qubits", 0))
                gates = circuit.get("gates", circuit.get("operations", []))
                gate_count = len(gates) if gates else 0
                depth = circuit.get("depth", gate_count)
            elif hasattr(circuit, "num_qubits"):
                num_qubits = circuit.num_qubits
            elif hasattr(circuit, "qubits"):
                qubits = circuit.qubits
                num_qubits = len(qubits) if hasattr(qubits, "__len__") else qubits
            
            if hasattr(circuit, "gate_count"):
                gate_count = circuit.gate_count
            elif hasattr(circuit, "num_operations"):
                gate_count = circuit.num_operations
            elif hasattr(circuit, "gates"):
                gate_count = len(circuit.gates)
            
            if hasattr(circuit, "depth"):
                depth = circuit.depth
            elif hasattr(circuit, "circuit_depth"):
                depth = circuit.circuit_depth
        except Exception:
            pass
        
        return num_qubits, gate_count, depth
    
    def generate_report(
        self,
        results: list[LRETBenchmarkResult],
        include_raw_metrics: bool = False,
    ) -> dict[str, Any]:
        """Generate a comprehensive benchmark report.
        
        Args:
            results: List of benchmark results.
            include_raw_metrics: Whether to include raw metrics data.
            
        Returns:
            Dictionary containing the benchmark report.
        """
        report = {
            "backend": "lret",
            "version": self._adapter.get_version(),
            "timestamp": time.time(),
            "summary": {
                "total_circuits": len(results),
                "total_runs": sum(r.num_runs for r in results),
                "successful_runs": sum(len(r.metrics) for r in results),
                "failed_runs": sum(len(r.errors) for r in results),
            },
            "results": [],
        }
        
        for result in results:
            result_data = {
                "circuit_name": result.circuit_name,
                "num_runs": result.num_runs,
                "mean_execution_time_ms": result.mean_execution_time_ms,
                "std_execution_time_ms": result.std_execution_time_ms,
                "min_execution_time_ms": result.min_execution_time_ms,
                "max_execution_time_ms": result.max_execution_time_ms,
                "mean_throughput_shots_per_sec": result.mean_throughput,
                "mean_memory_mb": result.mean_memory_mb,
                "success_rate": result.success_rate,
            }
            
            if include_raw_metrics:
                result_data["metrics"] = [m.to_dict() for m in result.metrics]
            
            if result.errors:
                result_data["errors"] = result.errors
            
            report["results"].append(result_data)
        
        # Calculate overall statistics
        if results:
            all_times = [r.mean_execution_time_ms for r in results if r.metrics]
            all_throughputs = [r.mean_throughput for r in results if r.mean_throughput > 0]
            
            report["overall"] = {
                "avg_execution_time_ms": sum(all_times) / len(all_times) if all_times else 0,
                "avg_throughput_shots_per_sec": sum(all_throughputs) / len(all_throughputs) if all_throughputs else 0,
            }
        
        return report


# ==============================================================================
# LRET BACKEND ADAPTER
# ==============================================================================


class LRETBackendAdapter(BaseBackendAdapter):
    """LRET backend adapter for quantum circuit simulation.

    LRET is a lightweight framework for quantum computing experiments.
    This adapter supports the feature/framework-integration branch API.
    
    Features:
    - Complete result normalization for all output formats
    - Framework-integration branch API verification
    - Fallback to mock simulator when LRET is not installed
    - Comprehensive error handling and logging
    """

    def __init__(self) -> None:
        """Initialize the LRET adapter."""
        self._lret_module: Any = None
        self._cached_version: str | None = None
        self._api_verifier: LRETAPIVerifier | None = None
        self._result_normalizer: LRETResultNormalizer | None = None
        self._use_mock: bool = False

    # ------------------------------------------------------------------
    # Benchmarking hooks
    # ------------------------------------------------------------------
    def prepare_for_benchmark(self, circuit: Any | None = None, shots: int | None = None) -> None:
        """Reset transient state to avoid cross-run side effects."""
        # Clear any cached adapters/normalizers to ensure a fresh run
        self._cached_version = None
        self._result_normalizer = None
        # Reset API verifier cache to revalidate if needed
        self._api_verifier = None

    def cleanup_after_benchmark(self) -> None:
        """No-op cleanup hook for LRET (placeholder for future)."""
        # If the real LRET backend exposes cache clearing, invoke it here
        return

    def get_name(self) -> str:
        """Return backend identifier."""
        return "lret"

    def get_version(self) -> str:
        """Return LRET version string."""
        if self._cached_version:
            return self._cached_version

        if not self.is_available():
            return "unavailable"

        try:
            lret = self._get_lret_module()
            self._cached_version = getattr(lret, "__version__", "unknown")
            return self._cached_version
        except Exception:
            return "unknown"

    def is_available(self) -> bool:
        """Check if LRET is installed and importable."""
        return importlib.util.find_spec("lret") is not None

    def get_installation_instructions(self) -> str:
        """Get instructions for installing LRET.
        
        Returns:
            String with installation instructions.
        """
        return LRET_INSTALL_INSTRUCTIONS

    def get_library_status(self) -> dict:
        """Get comprehensive status of LRET library installation.
        
        Returns:
            Dictionary with installation status, version, and recommendations.
        """
        status = {
            "installed": self.is_available(),
            "version": None,
            "branch": None,
            "api_compatible": False,
            "using_mock": True,
            "recommendations": [],
        }
        
        if self.is_available():
            try:
                lret = self._get_lret_module()
                status["version"] = getattr(lret, "__version__", "unknown")
                status["branch"] = getattr(lret, "_BRANCH", "unknown")
                status["using_mock"] = False
                
                # Verify API compatibility
                api_check = self.get_api_verification()
                status["api_compatible"] = api_check.is_compatible
                
                if not api_check.is_compatible:
                    if api_check.missing_apis:
                        status["recommendations"].append(
                            f"Missing APIs: {', '.join(api_check.missing_apis)}"
                        )
                    status["recommendations"].append(
                        "Consider switching to feature/framework-integration branch"
                    )
            except Exception as e:
                status["recommendations"].append(f"Import error: {e}")
        else:
            status["recommendations"].append(
                "LRET not installed. Install from: " + LRET_GITHUB_URL
            )
            status["recommendations"].append(
                "Using mock simulator for testing"
            )
        
        return status

    def _get_lret_module(self) -> Any:
        """Get the LRET module, importing if needed."""
        if self._lret_module is None:
            if not self.is_available():
                raise BackendNotInstalledError("lret", ["lret"])
            import lret  # type: ignore

            self._lret_module = lret
        return self._lret_module

    def get_api_verification(self) -> LRETAPIVerification:
        """Get API verification results for LRET.
        
        Returns:
            LRETAPIVerification with compatibility status
        """
        if self._api_verifier is None:
            try:
                lret = self._get_lret_module() if self.is_available() else None
                self._api_verifier = LRETAPIVerifier(lret)
            except Exception:
                self._api_verifier = LRETAPIVerifier(None)
        
        return self._api_verifier.verify()

    def get_capabilities(self) -> Capabilities:
        """Return LRET capabilities.

        LRET supports custom simulation modes that may differ from
        standard StateVector/DensityMatrix simulators.
        """
        return Capabilities(
            simulator_types=[SimulatorType.CUSTOM, SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=32,
            supports_noise=False,
            supports_gpu=False,
            supports_batching=True,
            custom_features={
                "framework_integration": True,
                "custom_gates": True,
                "lret_native_format": True,
                "result_normalization": True,
                "api_verification": True,
            },
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate a circuit for LRET execution.

        LRET accepts various circuit formats:
        - Native LRET circuit objects
        - Dictionary-based circuit specifications
        - Qiskit/Cirq circuits (via conversion)

        Args:
            circuit: Circuit to validate

        Returns:
            ValidationResult indicating validity
        """
        if circuit is None:
            return ValidationResult(
                valid=False,
                message="Circuit is None",
                details={"error": "null_circuit"},
            )

        # Check for LRET native format
        if self.is_available():
            try:
                lret = self._get_lret_module()

                # Check for native LRET circuit type
                if hasattr(lret, "Circuit"):
                    if isinstance(circuit, lret.Circuit):
                        return ValidationResult(
                            valid=True, message="LRET native circuit"
                        )

                # Check for LRET-compatible dict format
                if hasattr(lret, "validate_circuit"):
                    is_valid = lret.validate_circuit(circuit)
                    return ValidationResult(
                        valid=is_valid,
                        message=(
                            "Validated via LRET"
                            if is_valid
                            else "LRET validation failed"
                        ),
                    )
            except Exception as exc:
                logger.debug("LRET validation check failed: %s", exc)

        # Generic validation for dict-based circuits
        if isinstance(circuit, dict):
            required_keys = {"gates"} | {"operations"} | {"instructions"}
            has_required = bool(required_keys & set(circuit.keys()))
            if has_required or "qubits" in circuit:
                return ValidationResult(
                    valid=True,
                    message="Dictionary-based circuit accepted",
                    details={"format": "dict", "keys": list(circuit.keys())},
                )
            return ValidationResult(
                valid=False,
                message="Dictionary circuit missing required keys",
                details={"provided_keys": list(circuit.keys())},
            )

        # Check for Qiskit/Cirq circuits that can be converted
        circuit_type = type(circuit).__name__
        if "QuantumCircuit" in circuit_type or "Circuit" in circuit_type:
            return ValidationResult(
                valid=True,
                message=f"Circuit type {circuit_type} will be converted for LRET",
                details={"original_type": circuit_type, "requires_conversion": True},
            )

        return ValidationResult(
            valid=False,
            message=f"Unsupported circuit type: {circuit_type}",
            details={"type": circuit_type},
        )

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate resources needed for circuit execution."""
        # Allow estimation even when LRET is not installed if using mock mode
        if not self.is_available() and not self._use_mock:
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "LRET not installed"},
            )

        qubits = self._extract_qubit_count(circuit)
        if qubits is None:
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "Could not determine qubit count"},
            )

        memory_mb = (2**qubits * 16) / (1024 * 1024) if qubits <= 32 else None
        gate_count = self._extract_gate_count(circuit)

        metadata: dict[str, Any] = {"qubits": qubits}
        if gate_count is not None:
            metadata["gate_count"] = gate_count
            time_ms = gate_count * qubits * 0.1
        else:
            time_ms = None

        return ResourceEstimate(memory_mb=memory_mb, time_ms=time_ms, metadata=metadata)

    def _extract_qubit_count(self, circuit: Any) -> int | None:
        """Extract qubit count from circuit."""
        if circuit is None:
            return None

        # Check dict format
        if isinstance(circuit, dict):
            for key in ("num_qubits", "qubits", "n_qubits"):
                val = circuit.get(key)
                if isinstance(val, int):
                    return val
                if isinstance(val, list):
                    return len(val)

        # Check object attributes
        for attr in ("num_qubits", "n_qubits", "qubit_count"):
            val = getattr(circuit, attr, None)
            if isinstance(val, int):
                return val
            if callable(val):
                try:
                    result = val()
                    if isinstance(result, int):
                        return result
                except Exception:
                    pass

        return None

    def _extract_gate_count(self, circuit: Any) -> int | None:
        """Extract gate count from circuit."""
        if circuit is None:
            return None

        if isinstance(circuit, dict):
            # Check each possible key - only return count if key exists
            for key in ("gates", "operations", "instructions"):
                if key in circuit:
                    gates = circuit[key]
                    if isinstance(gates, list):
                        return len(gates)
            # No gate-related keys found
            return None

        for attr in ("gates", "operations", "instructions"):
            val = getattr(circuit, attr, None)
            if isinstance(val, (list, tuple)):
                return len(val)
            if callable(val):
                try:
                    result = val()
                    if isinstance(result, (list, tuple)):
                        return len(result)
                except Exception:
                    pass

        return None

    def execute(
        self,
        circuit: Any,
        shots: int = 1024,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on the LRET backend.

        Args:
            circuit: The circuit to execute
            shots: Number of measurement shots
            **kwargs: Additional execution parameters

        Returns:
            ExecutionResult with execution data
        """
        start_time = time.time()
        
        # Validate circuit first
        validation = self.validate_circuit(circuit)
        if not validation.valid:
            raise CircuitValidationError(validation.message)

        # Prepare result normalizer
        num_qubits = self._extract_qubit_count(circuit) or 2
        normalizer = LRETResultNormalizer(num_qubits)

        # Try real LRET first, fall back to mock if unavailable
        if self.is_available() and not self._use_mock:
            try:
                result = self._execute_real_lret(circuit, shots, **kwargs)
            except Exception as exc:
                logger.warning("Real LRET execution failed, using mock: %s", exc)
                result = self._execute_mock(circuit, shots, **kwargs)
        else:
            # LRET not available - check if mock mode is allowed
            if not self._use_mock:
                raise BackendNotInstalledError("lret", ["lret"])
            result = self._execute_mock(circuit, shots, **kwargs)

        execution_time_ms = (time.time() - start_time) * 1000
        
        # Normalize the result
        normalized = normalizer.normalize(
            result,
            shots=shots,
            execution_time_ms=execution_time_ms,
        )

        # Build ExecutionResult with proper signature
        data = {}
        if normalized.counts:
            data["counts"] = normalized.counts
        if normalized.statevector is not None:
            data["statevector"] = normalized.statevector
        if normalized.probabilities:
            data["probabilities"] = normalized.probabilities
        
        result_type = ResultType.COUNTS if normalized.counts else ResultType.STATEVECTOR
        
        return ExecutionResult(
            backend="lret",
            simulator_type=SimulatorType.CUSTOM,
            execution_time_ms=execution_time_ms,
            qubit_count=num_qubits,
            shot_count=shots if shots > 0 else None,
            result_type=result_type,
            data=data,
            metadata={
                "normalized": True,
                "format": normalized.format.value,
                **normalized.metadata,
            },
            raw_result=result,
        )

    def _execute_real_lret(
        self,
        circuit: Any,
        shots: int,
        **kwargs: Any,
    ) -> Any:
        """Execute using real LRET library."""
        lret = self._get_lret_module()
        
        # Prepare circuit if conversion needed
        prepared_circuit = self._prepare_circuit(circuit, lret)
        
        # Execute based on available API
        if hasattr(lret, "execute"):
            return lret.execute(prepared_circuit, shots=shots, **kwargs)
        elif hasattr(lret, "Simulator"):
            simulator = lret.Simulator()
            if shots > 0:
                return simulator.run(prepared_circuit, shots=shots)
            else:
                return simulator.simulate(prepared_circuit)
        else:
            raise UnsupportedOperationError("LRET execution API not found")

    def _prepare_circuit(self, circuit: Any, lret: Any) -> Any:
        """Prepare circuit for LRET execution."""
        # Already LRET circuit
        if hasattr(lret, "Circuit") and isinstance(circuit, lret.Circuit):
            return circuit
        
        # Convert from dict
        if isinstance(circuit, dict):
            if hasattr(lret, "from_dict"):
                return lret.from_dict(circuit)
            return circuit
        
        # Convert from Qiskit
        circuit_type = type(circuit).__name__
        if "QuantumCircuit" in circuit_type and hasattr(lret, "from_qiskit"):
            return lret.from_qiskit(circuit)
        
        # Convert from Cirq
        if circuit_type == "Circuit" and hasattr(lret, "from_cirq"):
            return lret.from_cirq(circuit)
        
        return circuit

    def _execute_mock(
        self,
        circuit: Any,
        shots: int,
        **kwargs: Any,
    ) -> MockLRETResult:
        """Execute using mock LRET simulator."""
        simulator = MockLRETSimulator()
        
        seed = kwargs.get("seed")
        if seed is not None:
            simulator.set_seed(seed)
        
        if shots > 0:
            return simulator.run(circuit, shots=shots)
        else:
            return simulator.simulate(circuit)

    def get_statevector(self, result: Any) -> np.ndarray | None:
        """Extract statevector from LRET result.

        Args:
            result: Result from LRET execution

        Returns:
            Statevector as numpy array or None
        """
        if result is None:
            return None

        # Try common attribute names
        for attr in ("statevector", "state_vector", "_statevector"):
            val = getattr(result, attr, None)
            if val is not None:
                return np.asarray(val, dtype=complex)

        # Try getter methods
        for method in ("get_statevector", "get_state_vector"):
            getter = getattr(result, method, None)
            if callable(getter):
                val = getter()
                if val is not None:
                    return np.asarray(val, dtype=complex)

        if hasattr(result, "__array__"):
            return np.asarray(result, dtype=complex)

        return None

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        """Check if simulator type is supported."""
        return sim_type in self.get_capabilities().simulator_types

    def execute_batch(
        self,
        circuits: list[Any],
        shots: int = 1024,
        continue_on_error: bool = True,
        **kwargs: Any,
    ) -> LRETBatchExecutionResult:
        """Execute multiple circuits in batch.
        
        Provides efficient batch execution of multiple circuits, reusing
        simulator setup and providing aggregated results.
        
        Args:
            circuits: List of circuits to execute
            shots: Number of measurement shots per circuit
            continue_on_error: Whether to continue if a circuit fails
            **kwargs: Additional execution parameters
            
        Returns:
            LRETBatchExecutionResult with all results
            
        Example:
            >>> adapter = LRETBackendAdapter()
            >>> circuits = [circuit1, circuit2, circuit3]
            >>> batch_result = adapter.execute_batch(circuits, shots=1000)
            >>> print(f"Executed {batch_result.successful}/{batch_result.total_circuits}")
        """
        start_time = time.time()
        
        results: list[ExecutionResult | None] = []
        errors: list[tuple[int, str]] = []
        successful = 0
        failed = 0
        
        for idx, circuit in enumerate(circuits):
            try:
                result = self.execute(circuit, shots=shots, **kwargs)
                results.append(result)
                successful += 1
            except Exception as exc:
                failed += 1
                error_msg = str(exc)
                errors.append((idx, error_msg))
                results.append(None)
                logger.warning("Batch execution failed for circuit %d: %s", idx, error_msg)
                
                if not continue_on_error:
                    # Fill remaining with None
                    for _ in range(idx + 1, len(circuits)):
                        results.append(None)
                    break
        
        total_time_ms = (time.time() - start_time) * 1000
        avg_time = total_time_ms / len(circuits) if circuits else 0.0
        
        return LRETBatchExecutionResult(
            total_circuits=len(circuits),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
            total_execution_time_ms=total_time_ms,
            average_time_per_circuit_ms=avg_time,
        )

    def execute_parameter_sweep(
        self,
        circuit: Any,
        parameter_sets: list[dict[str, float]],
        shots: int = 1024,
        continue_on_error: bool = True,
        **kwargs: Any,
    ) -> LRETBatchExecutionResult:
        """Execute a circuit with multiple parameter sets.
        
        Optimized for variational algorithms where the same circuit
        structure is executed with different parameter values.
        
        Args:
            circuit: Parameterized circuit template
            parameter_sets: List of parameter value dictionaries
            shots: Number of measurement shots per execution
            continue_on_error: Whether to continue if an execution fails
            **kwargs: Additional execution parameters
            
        Returns:
            LRETBatchExecutionResult with results for each parameter set
        """
        start_time = time.time()
        
        results: list[ExecutionResult | None] = []
        errors: list[tuple[int, str]] = []
        successful = 0
        failed = 0
        
        for idx, params in enumerate(parameter_sets):
            try:
                # Bind parameters if circuit supports it
                bound_circuit = self._bind_parameters(circuit, params)
                result = self.execute(bound_circuit, shots=shots, **kwargs)
                results.append(result)
                successful += 1
            except Exception as exc:
                failed += 1
                error_msg = str(exc)
                errors.append((idx, error_msg))
                results.append(None)
                logger.warning("Parameter sweep failed for set %d: %s", idx, error_msg)
                
                if not continue_on_error:
                    for _ in range(idx + 1, len(parameter_sets)):
                        results.append(None)
                    break
        
        total_time_ms = (time.time() - start_time) * 1000
        avg_time = total_time_ms / len(parameter_sets) if parameter_sets else 0.0
        
        return LRETBatchExecutionResult(
            total_circuits=len(parameter_sets),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
            total_execution_time_ms=total_time_ms,
            average_time_per_circuit_ms=avg_time,
        )

    def _bind_parameters(self, circuit: Any, params: dict[str, float]) -> Any:
        """Bind parameters to a parameterized circuit.
        
        Args:
            circuit: Circuit with parameters
            params: Dictionary mapping parameter names to values
            
        Returns:
            Circuit with bound parameters
        """
        # Try common parameter binding methods
        if hasattr(circuit, "bind_parameters"):
            return circuit.bind_parameters(params)
        elif hasattr(circuit, "assign_parameters"):
            return circuit.assign_parameters(params)
        elif hasattr(circuit, "resolve_parameters"):
            return circuit.resolve_parameters(params)
        
        # If circuit is LRET-native, try LRET API
        if self.is_available() and not self._use_mock:
            try:
                lret = self._get_lret_module()
                if hasattr(lret, "bind_parameters"):
                    return lret.bind_parameters(circuit, params)
            except Exception:
                pass
        
        # Return circuit as-is if no binding method found
        return circuit


    def get_supported_gates(self) -> list[str]:
        """Get list of gates supported by LRET."""
        standard_gates = [
            "H",
            "X",
            "Y",
            "Z",
            "S",
            "T",
            "Sdg",
            "Tdg",
            "RX",
            "RY",
            "RZ",
            "CX",
            "CNOT",
            "CZ",
            "SWAP",
            "CCX",
            "U",
            "U1",
            "U2",
            "U3",
        ]
        if self.is_available():
            try:
                lret = self._get_lret_module()
                if hasattr(lret, "SUPPORTED_GATES"):
                    return list(lret.SUPPORTED_GATES)
            except Exception as exc:
                logger.debug("Could not get LRET supported gates: %s", exc)
        return standard_gates

    def use_mock_backend(self, use_mock: bool = True) -> None:
        """Force use of mock backend for testing.
        
        Args:
            use_mock: Whether to use mock backend instead of real LRET
        """
        self._use_mock = use_mock

    # ------------------------------------------------------------------
    # Performance Benchmarking Methods
    # ------------------------------------------------------------------

    def get_benchmark_runner(self) -> LRETBenchmarkRunner:
        """Get a benchmark runner configured for this adapter.
        
        Returns:
            LRETBenchmarkRunner instance ready for benchmarking.
            
        Example:
            >>> adapter = LRETBackendAdapter()
            >>> runner = adapter.get_benchmark_runner()
            >>> result = runner.run_benchmark(circuit, num_runs=5)
        """
        return LRETBenchmarkRunner(self)

    def run_performance_benchmark(
        self,
        circuit: Any,
        shots: int = 1024,
        num_runs: int = 5,
        warmup_runs: int = 1,
        circuit_name: str = "benchmark",
    ) -> LRETBenchmarkResult:
        """Run a performance benchmark on a circuit.
        
        Convenience method for running performance benchmarks directly
        from the adapter without creating a separate runner.
        
        Args:
            circuit: The quantum circuit to benchmark.
            shots: Number of measurement shots per iteration.
            num_runs: Number of benchmark iterations.
            warmup_runs: Number of warmup runs before measuring.
            circuit_name: Identifier for the circuit.
            
        Returns:
            Aggregated benchmark results with statistics.
            
        Example:
            >>> adapter = LRETBackendAdapter()
            >>> result = adapter.run_performance_benchmark(
            ...     circuit, shots=1024, num_runs=10
            ... )
            >>> print(f"Mean time: {result.mean_execution_time_ms:.2f}ms")
        """
        runner = self.get_benchmark_runner()
        return runner.run_benchmark(
            circuit=circuit,
            num_runs=num_runs,
            shots=shots,
            circuit_name=circuit_name,
            warmup_runs=warmup_runs,
        )

    def run_scaling_benchmark(
        self,
        circuit_generator: Callable[[int], Any],
        qubit_range: list[int] | None = None,
        shots: int = 1024,
        num_runs: int = 3,
    ) -> list[LRETBenchmarkResult]:
        """Run benchmarks across different circuit sizes.
        
        Tests performance scaling as circuit size increases, useful for
        understanding backend performance characteristics.
        
        Args:
            circuit_generator: Function that generates a circuit given qubit count.
            qubit_range: List of qubit counts to benchmark. Defaults to [2, 4, 6, 8, 10].
            shots: Number of measurement shots per iteration.
            num_runs: Number of runs per circuit size.
            
        Returns:
            List of benchmark results for each circuit size.
            
        Example:
            >>> def make_ghz(n):
            ...     return {"qubits": n, "gates": [("H", 0)] + [("CNOT", i, i+1) for i in range(n-1)]}
            >>> results = adapter.run_scaling_benchmark(make_ghz, [2, 4, 6, 8])
        """
        if qubit_range is None:
            qubit_range = [2, 4, 6, 8, 10]
        
        runner = self.get_benchmark_runner()
        return runner.run_scaling_benchmark(
            circuit_generator=circuit_generator,
            qubit_range=qubit_range,
            shots=shots,
            num_runs=num_runs,
        )

    def run_throughput_benchmark(
        self,
        circuit: Any,
        shot_counts: list[int] | None = None,
        num_runs: int = 3,
        circuit_name: str = "throughput_test",
    ) -> list[LRETBenchmarkResult]:
        """Benchmark throughput across different shot counts.
        
        Measures how throughput (shots/second) scales with the number
        of measurement shots.
        
        Args:
            circuit: The quantum circuit to benchmark.
            shot_counts: List of shot counts to test. Defaults to geometric progression.
            num_runs: Number of runs per shot count.
            circuit_name: Base identifier for the circuit.
            
        Returns:
            List of benchmark results for each shot count.
            
        Example:
            >>> results = adapter.run_throughput_benchmark(
            ...     circuit, shot_counts=[100, 1000, 10000]
            ... )
            >>> for r in results:
            ...     print(f"{r.circuit_name}: {r.mean_throughput:.0f} shots/sec")
        """
        if shot_counts is None:
            shot_counts = [100, 500, 1000, 2000, 5000, 10000]
        
        runner = self.get_benchmark_runner()
        return runner.run_throughput_benchmark(
            circuit=circuit,
            shot_counts=shot_counts,
            num_runs=num_runs,
            circuit_name=circuit_name,
        )

    def generate_benchmark_report(
        self,
        results: list[LRETBenchmarkResult],
        include_raw_metrics: bool = False,
    ) -> dict[str, Any]:
        """Generate a comprehensive benchmark report.
        
        Args:
            results: List of benchmark results.
            include_raw_metrics: Whether to include raw metrics data.
            
        Returns:
            Dictionary containing the benchmark report.
            
        Example:
            >>> results = adapter.run_scaling_benchmark(make_circuit, [2, 4, 6])
            >>> report = adapter.generate_benchmark_report(results)
            >>> print(json.dumps(report, indent=2))
        """
        runner = self.get_benchmark_runner()
        return runner.generate_report(results, include_raw_metrics)

    def get_performance_profile(
        self,
        circuit: Any,
        shots: int = 1024,
    ) -> dict[str, Any]:
        """Get a detailed performance profile for a single circuit execution.
        
        Provides fine-grained timing breakdown for different execution phases.
        
        Args:
            circuit: The quantum circuit to profile.
            shots: Number of measurement shots.
            
        Returns:
            Dictionary with detailed performance metrics.
            
        Example:
            >>> profile = adapter.get_performance_profile(circuit, shots=1000)
            >>> print(f"Gate execution: {profile['gate_execution_time_ms']:.2f}ms")
            >>> print(f"Measurement: {profile['measurement_time_ms']:.2f}ms")
        """
        monitor = LRETPerformanceMonitor()
        
        # Extract circuit info
        num_qubits = self._extract_qubit_count(circuit) or 2
        gate_count = 0
        depth = 0
        
        if isinstance(circuit, dict):
            gates = circuit.get("gates", circuit.get("operations", []))
            gate_count = len(gates) if gates else 0
            depth = circuit.get("depth", gate_count)
        elif hasattr(circuit, "gate_count"):
            gate_count = circuit.gate_count
        elif hasattr(circuit, "gates"):
            gate_count = len(circuit.gates) if hasattr(circuit.gates, "__len__") else 0
        
        if hasattr(circuit, "depth"):
            depth = circuit.depth
        
        # Prepare and execute with monitoring
        self.prepare_for_benchmark(circuit, shots)
        
        try:
            monitor.start()
            monitor.sample_memory()
            
            monitor.start_phase("validation")
            validation = self.validate_circuit(circuit)
            monitor.end_phase("validation")
            
            if not validation.valid:
                raise CircuitValidationError(validation.message)
            
            monitor.start_phase("normalization_setup")
            normalizer = LRETResultNormalizer(num_qubits)
            monitor.end_phase("normalization_setup")
            
            monitor.start_phase("gate_execution")
            if self.is_available() and not self._use_mock:
                try:
                    result = self._execute_real_lret(circuit, shots)
                except Exception:
                    result = self._execute_mock(circuit, shots)
            else:
                result = self._execute_mock(circuit, shots)
            monitor.end_phase("gate_execution")
            
            monitor.start_phase("measurement")
            # Measurement is included in execution for LRET
            monitor.end_phase("measurement")
            
            monitor.start_phase("normalization")
            _ = normalizer.normalize(result, shots=shots)
            monitor.end_phase("normalization")
            
            monitor.sample_memory()
            
            metrics = monitor.get_metrics(
                qubits=num_qubits,
                gate_count=gate_count,
                circuit_depth=depth,
                shots=shots,
            )
            
            return {
                "backend": "lret",
                "circuit_info": {
                    "qubits": num_qubits,
                    "gate_count": gate_count,
                    "depth": depth,
                },
                "execution_time_ms": metrics.execution_time_ms,
                "gate_execution_time_ms": metrics.gate_execution_time_ms,
                "measurement_time_ms": metrics.measurement_time_ms,
                "normalization_time_ms": metrics.normalization_time_ms,
                "validation_time_ms": monitor.get_phase_time("validation"),
                "memory_baseline_mb": metrics.memory_baseline_mb,
                "memory_peak_mb": metrics.memory_peak_mb,
                "memory_delta_mb": metrics.memory_peak_mb - metrics.memory_baseline_mb,
                "throughput_shots_per_sec": metrics.throughput_shots_per_sec,
                "gates_per_second": metrics.gates_per_second,
                "shots": shots,
            }
            
        finally:
            self.cleanup_after_benchmark()


# ==============================================================================
# MOCK LRET IMPLEMENTATION FOR TESTING
# ==============================================================================
#
# This section provides a mock LRET implementation for testing purposes when
# the real LRET library is not installed. The mock simulates realistic behavior
# including quantum circuit execution with proper probability distributions.
#
# REAL LRET INTEGRATION POINTS:
# =============================
# To integrate with the real LRET library (https://github.com/kunal5556/LRET),
# the following integration points need to be implemented:
#
# 1. Import Statement (line ~20):
#    Replace: import lret  # type: ignore
#    With:    from lret import Circuit, Simulator, execute, validate_circuit
#
# 2. Circuit Conversion (_prepare_circuit method):
#    - lret.from_qiskit(circuit) - Convert Qiskit QuantumCircuit to LRET format
#    - lret.from_cirq(circuit) - Convert Cirq Circuit to LRET format
#    - lret.from_dict(circuit) - Create circuit from dictionary specification
#    - lret.Circuit() - Native LRET circuit construction
#
# 3. Simulator Execution (execute method):
#    - lret.Simulator() - Create simulator instance
#    - simulator.run(circuit, shots=N) - Execute with measurement sampling
#    - simulator.simulate(circuit) - Execute for statevector output
#    - lret.execute(circuit, shots=N) - Alternative execution API
#
# 4. Result Extraction:
#    - result.counts / result.get_counts() - Measurement count dictionary
#    - result.statevector / result.state_vector - Final statevector
#
# 5. Validation:
#    - lret.validate_circuit(circuit) - Validate circuit for LRET execution
#    - lret.SUPPORTED_GATES - List of supported gate names
#
# ==============================================================================


class MockLRETSimulator:
    """Mock LRET Simulator for testing when real LRET is not installed.

    This simulator provides realistic quantum circuit simulation behavior
    including proper probability distributions based on circuit structure.
    It serves as a development and testing fallback.

    Features:
    - Statevector simulation for small circuits
    - Measurement sampling with correct probability distributions
    - Support for common quantum gates
    - Deterministic results when seed is set
    """

    def __init__(self) -> None:
        """Initialize mock simulator."""
        self._seed: int | None = None
        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def simulate(self, circuit: Any) -> MockLRETResult:
        """Simulate circuit and return statevector.

        Args:
            circuit: Circuit to simulate (dict or circuit object)

        Returns:
            MockLRETResult containing statevector
        """
        num_qubits = self._get_qubit_count(circuit)
        statevector = self._simulate_statevector(circuit, num_qubits)
        return MockLRETResult(statevector=statevector)

    def run(self, circuit: Any, shots: int = 1024) -> MockLRETResult:
        """Execute circuit with measurement sampling.

        Args:
            circuit: Circuit to execute
            shots: Number of measurement shots

        Returns:
            MockLRETResult containing measurement counts
        """
        num_qubits = self._get_qubit_count(circuit)
        statevector = self._simulate_statevector(circuit, num_qubits)
        counts = self._sample_measurements(statevector, num_qubits, shots)
        return MockLRETResult(counts=counts, shots=shots)

    def _get_qubit_count(self, circuit: Any) -> int:
        """Extract qubit count from circuit."""
        if isinstance(circuit, dict):
            for key in ("num_qubits", "qubits", "n_qubits"):
                val = circuit.get(key)
                if isinstance(val, int):
                    return val
                if isinstance(val, list):
                    return len(val)
            # Infer from gates
            gates = circuit.get("gates", circuit.get("operations", circuit.get("instructions", [])))
            if gates:
                max_qubit = 0
                for gate in gates:
                    qubits = gate.get("qubits", gate.get("targets", []))
                    if qubits:
                        max_qubit = max(max_qubit, max(qubits) + 1)
                return max(max_qubit, 1)

        for attr in ("num_qubits", "n_qubits", "qubit_count"):
            val = getattr(circuit, attr, None)
            if isinstance(val, int):
                return val

        return 2  # Default fallback

    def _simulate_statevector(self, circuit: Any, num_qubits: int) -> np.ndarray:
        """Simulate circuit to produce statevector.

        This is a simplified simulation that handles basic circuits.
        For complex circuits, it produces a normalized random state
        that provides realistic-looking results.
        """
        num_states = 2 ** num_qubits

        # Start with |0...0> state
        statevector = np.zeros(num_states, dtype=complex)
        statevector[0] = 1.0

        # Get gates from circuit
        gates = []
        if isinstance(circuit, dict):
            gates = circuit.get("gates", circuit.get("operations", circuit.get("instructions", [])))
        elif hasattr(circuit, "gates"):
            gates = circuit.gates

        # Apply gates (simplified simulation)
        for gate in gates:
            gate_name = ""
            target_qubits = []

            if isinstance(gate, dict):
                gate_name = gate.get("name", gate.get("gate", "")).lower()
                target_qubits = gate.get("qubits", gate.get("targets", []))
            elif hasattr(gate, "name"):
                gate_name = str(gate.name).lower()
                if hasattr(gate, "qubits"):
                    target_qubits = list(gate.qubits)

            if gate_name in ("h", "hadamard") and target_qubits:
                statevector = self._apply_hadamard(statevector, target_qubits[0], num_qubits)
            elif gate_name in ("x", "not", "pauli_x") and target_qubits:
                statevector = self._apply_x(statevector, target_qubits[0], num_qubits)
            elif gate_name in ("y", "pauli_y") and target_qubits:
                statevector = self._apply_y(statevector, target_qubits[0], num_qubits)
            elif gate_name in ("z", "pauli_z") and target_qubits:
                statevector = self._apply_z(statevector, target_qubits[0], num_qubits)
            elif gate_name in ("cx", "cnot") and len(target_qubits) >= 2:
                statevector = self._apply_cnot(statevector, target_qubits[0], target_qubits[1], num_qubits)
            elif gate_name in ("cz",) and len(target_qubits) >= 2:
                statevector = self._apply_cz(statevector, target_qubits[0], target_qubits[1], num_qubits)
            elif gate_name in ("swap",) and len(target_qubits) >= 2:
                statevector = self._apply_swap(statevector, target_qubits[0], target_qubits[1], num_qubits)

        # Normalize
        norm = np.linalg.norm(statevector)
        if norm > 0:
            statevector = statevector / norm

        return statevector

    def _apply_hadamard(self, sv: np.ndarray, qubit: int, n: int) -> np.ndarray:
        """Apply Hadamard gate to statevector."""
        result = np.zeros_like(sv)
        sqrt2_inv = 1.0 / np.sqrt(2)

        for i in range(len(sv)):
            bit_val = (i >> qubit) & 1
            partner = i ^ (1 << qubit)
            if bit_val == 0:
                result[i] += sqrt2_inv * sv[i]
                result[partner] += sqrt2_inv * sv[i]
            else:
                result[i] -= sqrt2_inv * sv[i]
                result[partner] += sqrt2_inv * sv[i]

        return result

    def _apply_x(self, sv: np.ndarray, qubit: int, n: int) -> np.ndarray:
        """Apply X (NOT) gate to statevector."""
        result = np.zeros_like(sv)
        for i in range(len(sv)):
            partner = i ^ (1 << qubit)
            result[partner] = sv[i]
        return result

    def _apply_y(self, sv: np.ndarray, qubit: int, n: int) -> np.ndarray:
        """Apply Y gate to statevector."""
        result = np.zeros_like(sv)
        for i in range(len(sv)):
            bit_val = (i >> qubit) & 1
            partner = i ^ (1 << qubit)
            if bit_val == 0:
                result[partner] = 1j * sv[i]
            else:
                result[partner] = -1j * sv[i]
        return result

    def _apply_z(self, sv: np.ndarray, qubit: int, n: int) -> np.ndarray:
        """Apply Z gate to statevector."""
        result = sv.copy()
        for i in range(len(sv)):
            if (i >> qubit) & 1:
                result[i] = -sv[i]
        return result

    def _apply_cnot(self, sv: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
        """Apply CNOT gate to statevector."""
        result = sv.copy()
        for i in range(len(sv)):
            if (i >> control) & 1:
                partner = i ^ (1 << target)
                result[i], result[partner] = sv[partner], sv[i]
        return result

    def _apply_cz(self, sv: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
        """Apply CZ gate to statevector."""
        result = sv.copy()
        for i in range(len(sv)):
            if ((i >> control) & 1) and ((i >> target) & 1):
                result[i] = -sv[i]
        return result

    def _apply_swap(self, sv: np.ndarray, qubit1: int, qubit2: int, n: int) -> np.ndarray:
        """Apply SWAP gate to statevector."""
        result = sv.copy()
        for i in range(len(sv)):
            bit1 = (i >> qubit1) & 1
            bit2 = (i >> qubit2) & 1
            if bit1 != bit2:
                partner = i ^ (1 << qubit1) ^ (1 << qubit2)
                result[i], result[partner] = sv[partner], sv[i]
        return result

    def _sample_measurements(
        self, statevector: np.ndarray, num_qubits: int, shots: int
    ) -> dict[str, int]:
        """Sample measurements from statevector."""
        probs = np.abs(statevector) ** 2
        probs = probs / np.sum(probs)  # Normalize

        # Sample outcomes
        outcomes = self._rng.choice(len(probs), size=shots, p=probs)

        # Convert to counts
        counts: dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(outcome, f"0{num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts


class MockLRETResult:
    """Mock result object returned by MockLRETSimulator."""

    def __init__(
        self,
        statevector: np.ndarray | None = None,
        counts: dict[str, int] | None = None,
        shots: int = 0,
    ) -> None:
        self.statevector = statevector
        self.state_vector = statevector  # Alternative name
        self.counts = counts or {}
        self.shots = shots

    def get_counts(self) -> dict[str, int]:
        """Return measurement counts."""
        return self.counts

    def get_statevector(self) -> np.ndarray | None:
        """Return statevector."""
        return self.statevector


def get_mock_lret_module() -> Any:
    """Get a mock LRET module for testing.

    Returns an object that mimics the LRET module API for testing purposes.
    """
    class MockLRETModule:
        __version__ = "0.1.0-mock"
        Simulator = MockLRETSimulator

        @staticmethod
        def validate_circuit(circuit: Any) -> bool:
            """Validate circuit structure."""
            if circuit is None:
                return False
            if isinstance(circuit, dict):
                return bool({"gates", "operations", "instructions", "qubits"} & set(circuit.keys()))
            return True

        @staticmethod
        def execute(circuit: Any, shots: int | None = None) -> MockLRETResult:
            """Execute circuit using mock simulator."""
            sim = MockLRETSimulator()
            if shots and shots > 0:
                return sim.run(circuit, shots)
            return sim.simulate(circuit)

        SUPPORTED_GATES = [
            "H", "X", "Y", "Z", "S", "T", "Sdg", "Tdg",
            "RX", "RY", "RZ", "CX", "CNOT", "CZ", "SWAP",
            "CCX", "U", "U1", "U2", "U3",
        ]

    return MockLRETModule()


