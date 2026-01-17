"""Qiskit backend adapter for StateVector and DensityMatrix simulation.

Implements Step 1.1.3b: Qiskit Aer adapter with:
- StateVector simulation
- DensityMatrix simulation
- GPU support integration (prerequisite for cuQuantum)
- Advanced transpilation options
- Comprehensive snapshot modes for intermediate state inspection
- Comprehensive noise model support
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

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
    QubitLimitExceededError,
    wrap_backend_exception,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# GPU SUPPORT INTEGRATION
# ==============================================================================


class GPUDeviceType(Enum):
    """Supported GPU device types."""
    
    CUDA = "cuda"
    CUSTATEVEC = "custatevec"
    CUTENSORNET = "cutensornet"
    AUTO = "auto"


@dataclass
class GPUConfiguration:
    """Configuration for GPU-accelerated simulation."""
    
    # Whether to enable GPU acceleration
    enabled: bool = False
    
    # Device type to use
    device_type: GPUDeviceType = GPUDeviceType.AUTO
    
    # Specific GPU device ID
    device_id: int = 0
    
    # Maximum GPU memory to use (in MB, None for auto)
    max_memory_mb: int | None = None
    
    # Whether to fall back to CPU if GPU unavailable
    fallback_to_cpu: bool = True
    
    # Blocking behavior for GPU operations
    blocking_enable: bool = True
    
    # Batch size for GPU operations
    batched_shots_gpu: int = 1
    
    # Maximum parallel experiments
    max_parallel_experiments: int = 1


@dataclass 
class GPUStatus:
    """Status of GPU availability and configuration."""
    
    is_available: bool = False
    device_name: str = "N/A"
    device_count: int = 0
    current_device: int = 0
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    cuda_version: str = "N/A"
    custatevec_available: bool = False
    cutensornet_available: bool = False
    error_message: str = ""


class GPUManager:
    """Manages GPU resources for Qiskit Aer simulations.
    
    This provides the foundation for cuQuantum integration by:
    - Detecting available GPU devices
    - Configuring Aer for GPU acceleration
    - Managing GPU memory
    - Providing fallback strategies
    """
    
    def __init__(self) -> None:
        """Initialize GPU manager."""
        self._cached_status: GPUStatus | None = None
    
    def detect_gpu(self) -> GPUStatus:
        """Detect available GPU devices and capabilities.
        
        Returns:
            GPUStatus with detected configuration
        """
        if self._cached_status is not None:
            return self._cached_status
        
        status = GPUStatus()
        
        try:
            # Try to detect CUDA via cupy or pycuda
            cuda_available = self._check_cuda()
            status.is_available = cuda_available
            
            if cuda_available:
                status = self._get_cuda_info(status)
            
            # Check for cuStateVec (cuQuantum state vector)
            status.custatevec_available = self._check_custatevec()
            
            # Check for cuTensorNet (cuQuantum tensor network)
            status.cutensornet_available = self._check_cutensornet()
            
        except Exception as e:
            status.error_message = str(e)
        
        self._cached_status = status
        return status
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        # Try multiple detection methods
        try:
            import cupy
            return True
        except ImportError:
            pass
        
        try:
            # Check via qiskit-aer
            from qiskit_aer import AerSimulator
            sim = AerSimulator()
            return 'GPU' in sim.available_devices()
        except Exception:
            pass
        
        return False
    
    def _check_custatevec(self) -> bool:
        """Check if cuStateVec (cuQuantum) is available."""
        try:
            import cuquantum
            return hasattr(cuquantum, 'custatevec')
        except ImportError:
            return False
    
    def _check_cutensornet(self) -> bool:
        """Check if cuTensorNet (cuQuantum) is available."""
        try:
            import cuquantum
            return hasattr(cuquantum, 'cutensornet')
        except ImportError:
            return False
    
    def _get_cuda_info(self, status: GPUStatus) -> GPUStatus:
        """Get detailed CUDA device information."""
        try:
            import cupy as cp
            
            status.device_count = cp.cuda.runtime.getDeviceCount()
            status.current_device = cp.cuda.runtime.getDevice()
            
            props = cp.cuda.runtime.getDeviceProperties(status.current_device)
            status.device_name = props['name'].decode()
            status.memory_total_mb = props['totalGlobalMem'] / (1024 * 1024)
            
            meminfo = cp.cuda.runtime.memGetInfo()
            status.memory_used_mb = (status.memory_total_mb * 1024 * 1024 - meminfo[0]) / (1024 * 1024)
            
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            status.cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            
        except Exception as e:
            status.error_message = str(e)
        
        return status
    
    def configure_simulator(
        self,
        config: GPUConfiguration,
    ) -> dict[str, Any]:
        """Generate simulator options for GPU configuration.
        
        Args:
            config: GPU configuration
            
        Returns:
            Dictionary of simulator options
        """
        options: dict[str, Any] = {}
        
        if not config.enabled:
            options["device"] = "CPU"
            return options
        
        status = self.detect_gpu()
        
        if not status.is_available:
            if config.fallback_to_cpu:
                logger.warning("GPU not available, falling back to CPU")
                options["device"] = "CPU"
            else:
                raise RuntimeError("GPU requested but not available")
            return options
        
        # Configure GPU device
        if config.device_type == GPUDeviceType.CUSTATEVEC and status.custatevec_available:
            options["device"] = "GPU"
            options["cuStateVec_enable"] = True
        elif config.device_type == GPUDeviceType.CUDA:
            options["device"] = "GPU"
        elif config.device_type == GPUDeviceType.AUTO:
            options["device"] = "GPU"
            if status.custatevec_available:
                options["cuStateVec_enable"] = True
        
        # Additional GPU options
        options["blocking_enable"] = config.blocking_enable
        options["batched_shots_gpu"] = config.batched_shots_gpu
        options["max_parallel_experiments"] = config.max_parallel_experiments
        
        if config.max_memory_mb:
            options["max_memory_mb"] = config.max_memory_mb
        
        return options
    
    def clear_cache(self) -> None:
        """Clear cached GPU status."""
        self._cached_status = None


# ==============================================================================
# ADVANCED TRANSPILATION OPTIONS
# ==============================================================================


class TranspilationLevel(Enum):
    """Transpilation optimization levels with descriptions."""
    
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3


@dataclass
class TranspilationConfig:
    """Advanced transpilation configuration."""
    
    # Base optimization level (0-3)
    optimization_level: int = 1
    
    # Basis gates to target
    basis_gates: list[str] | None = None
    
    # Coupling map for connectivity constraints
    coupling_map: list[tuple[int, int]] | None = None
    
    # Backend target for transpilation
    backend_target: str | None = None
    
    # Layout method: trivial, dense, noise_adaptive, sabre
    layout_method: str = "sabre"
    
    # Routing method: basic, stochastic, lookahead, sabre
    routing_method: str = "sabre"
    
    # Scheduling method: asap, alap, as_late_as_possible
    scheduling_method: str | None = None
    
    # Instruction durations for scheduling
    instruction_durations: list[tuple[str, Any, float]] | None = None
    
    # Translation method
    translation_method: str | None = None
    
    # Approximation degree (0 to 1)
    approximation_degree: float = 1.0
    
    # Seed for reproducibility
    seed_transpiler: int | None = None
    
    # Unroll custom definitions
    unroll_custom_definitions: bool = True
    
    # Initial layout
    initial_layout: list[int] | dict[int, int] | None = None


@dataclass
class TranspilationResult:
    """Result of circuit transpilation."""
    
    circuit: Any
    original_depth: int
    transpiled_depth: int
    original_gates: int
    transpiled_gates: int
    original_two_qubit_gates: int
    transpiled_two_qubit_gates: int
    layout: dict[int, int] | None = None
    optimization_time_ms: float = 0.0
    
    @property
    def depth_reduction(self) -> float:
        """Calculate depth reduction percentage."""
        if self.original_depth == 0:
            return 0.0
        return (1 - self.transpiled_depth / self.original_depth) * 100
    
    @property
    def gate_reduction(self) -> float:
        """Calculate gate count reduction percentage."""
        if self.original_gates == 0:
            return 0.0
        return (1 - self.transpiled_gates / self.original_gates) * 100


class AdvancedTranspiler:
    """Advanced transpilation manager for Qiskit circuits.
    
    Provides fine-grained control over circuit optimization
    including layout, routing, and scheduling.
    """
    
    # Default basis gates for various targets
    BASIS_GATE_SETS = {
        "universal": ["u1", "u2", "u3", "cx"],
        "ibmq": ["id", "rz", "sx", "x", "cx"],
        "ionq": ["rxx", "ryy", "rzz", "rx", "ry", "rz"],
        "rigetti": ["rx", "rz", "cz"],
        "aer_simulator": None,  # Use all available
    }
    
    def __init__(self) -> None:
        """Initialize transpiler."""
        self._qiskit = None
    
    def _get_qiskit(self) -> Any:
        """Lazy load Qiskit."""
        if self._qiskit is None:
            try:
                import qiskit
                self._qiskit = qiskit
            except ImportError:
                raise BackendNotInstalledError("qiskit", ["qiskit"])
        return self._qiskit
    
    def transpile(
        self,
        circuit: Any,
        config: TranspilationConfig | None = None,
        backend: Any = None,
    ) -> TranspilationResult:
        """Transpile a circuit with advanced options.
        
        Args:
            circuit: Qiskit QuantumCircuit
            config: Transpilation configuration
            backend: Optional backend target
            
        Returns:
            TranspilationResult with optimized circuit
        """
        qiskit = self._get_qiskit()
        config = config or TranspilationConfig()
        
        # Capture original metrics
        original_depth = circuit.depth()
        original_gates = circuit.size()
        original_2q = self._count_two_qubit_gates(circuit)
        
        start = time.perf_counter()
        
        # Build transpile kwargs
        transpile_kwargs: dict[str, Any] = {
            "optimization_level": config.optimization_level,
        }
        
        if config.basis_gates:
            transpile_kwargs["basis_gates"] = config.basis_gates
        elif config.backend_target and config.backend_target in self.BASIS_GATE_SETS:
            bg = self.BASIS_GATE_SETS[config.backend_target]
            if bg:
                transpile_kwargs["basis_gates"] = bg
        
        if config.coupling_map:
            from qiskit.transpiler import CouplingMap
            transpile_kwargs["coupling_map"] = CouplingMap(config.coupling_map)
        
        if config.layout_method:
            transpile_kwargs["layout_method"] = config.layout_method
        
        if config.routing_method:
            transpile_kwargs["routing_method"] = config.routing_method
        
        if config.scheduling_method:
            transpile_kwargs["scheduling_method"] = config.scheduling_method
        
        if config.instruction_durations:
            from qiskit.transpiler import InstructionDurations
            transpile_kwargs["instruction_durations"] = InstructionDurations(
                config.instruction_durations
            )
        
        if config.approximation_degree < 1.0:
            transpile_kwargs["approximation_degree"] = config.approximation_degree
        
        if config.seed_transpiler is not None:
            transpile_kwargs["seed_transpiler"] = config.seed_transpiler
        
        if config.initial_layout:
            transpile_kwargs["initial_layout"] = config.initial_layout
        
        if backend:
            transpile_kwargs["backend"] = backend
        
        # Perform transpilation
        transpiled = qiskit.transpile(circuit, **transpile_kwargs)
        
        opt_time_ms = (time.perf_counter() - start) * 1000
        
        # Get layout if available
        layout = None
        if hasattr(transpiled, "_layout") and transpiled._layout:
            try:
                layout = dict(transpiled._layout.get_physical_bits())
            except Exception:
                pass
        
        return TranspilationResult(
            circuit=transpiled,
            original_depth=original_depth,
            transpiled_depth=transpiled.depth(),
            original_gates=original_gates,
            transpiled_gates=transpiled.size(),
            original_two_qubit_gates=original_2q,
            transpiled_two_qubit_gates=self._count_two_qubit_gates(transpiled),
            layout=layout,
            optimization_time_ms=opt_time_ms,
        )
    
    def _count_two_qubit_gates(self, circuit: Any) -> int:
        """Count two-qubit gates in circuit."""
        count = 0
        for instr, qargs, _ in circuit.data:
            if len(qargs) >= 2:
                count += 1
        return count
    
    def estimate_transpilation_benefit(
        self,
        circuit: Any,
        levels: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Estimate the benefit of different optimization levels.
        
        Args:
            circuit: Circuit to analyze
            levels: Optimization levels to test (default [0,1,2,3])
            
        Returns:
            Dictionary mapping level to metrics
        """
        levels = levels or [0, 1, 2, 3]
        results = {}
        
        for level in levels:
            config = TranspilationConfig(optimization_level=level)
            result = self.transpile(circuit, config)
            
            results[level] = {
                "depth": result.transpiled_depth,
                "gates": result.transpiled_gates,
                "two_qubit_gates": result.transpiled_two_qubit_gates,
                "depth_reduction": result.depth_reduction,
                "gate_reduction": result.gate_reduction,
                "time_ms": result.optimization_time_ms,
            }
        
        return results


# ==============================================================================
# SNAPSHOT-BASED EXECUTION
# ==============================================================================


class SnapshotType(Enum):
    """Types of snapshots for intermediate state capture."""
    
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"
    PROBABILITIES = "probabilities"
    EXPECTATION_VALUE = "expectation_value"
    STABILIZER = "stabilizer"
    MATRIX_PRODUCT_STATE = "matrix_product_state"


@dataclass
class SnapshotConfig:
    """Configuration for a snapshot instruction."""
    
    snapshot_type: SnapshotType
    label: str
    qubits: list[int] | None = None
    observable: Any = None  # For expectation_value snapshots
    variance: bool = False  # Whether to also compute variance


@dataclass
class SnapshotResult:
    """Result from a snapshot during execution."""
    
    label: str
    snapshot_type: SnapshotType
    data: Any
    qubits: list[int] | None = None
    variance: float | None = None


class SnapshotManager:
    """Manages snapshot-based circuit execution.
    
    Snapshots allow capturing intermediate quantum states
    during simulation, useful for debugging and analysis.
    """
    
    def __init__(self) -> None:
        """Initialize snapshot manager."""
        self._qiskit = None
    
    def _get_qiskit(self) -> Any:
        """Lazy load Qiskit."""
        if self._qiskit is None:
            try:
                import qiskit
                self._qiskit = qiskit
            except ImportError:
                raise BackendNotInstalledError("qiskit", ["qiskit"])
        return self._qiskit
    
    def add_snapshot(
        self,
        circuit: Any,
        config: SnapshotConfig,
    ) -> Any:
        """Add a snapshot instruction to a circuit.
        
        Args:
            circuit: Qiskit QuantumCircuit
            config: Snapshot configuration
            
        Returns:
            Modified circuit with snapshot
        """
        self._get_qiskit()
        
        modified = circuit.copy()
        
        if config.snapshot_type == SnapshotType.STATEVECTOR:
            modified.save_statevector(label=config.label)
        
        elif config.snapshot_type == SnapshotType.DENSITY_MATRIX:
            modified.save_density_matrix(label=config.label)
        
        elif config.snapshot_type == SnapshotType.PROBABILITIES:
            if config.qubits:
                modified.save_probabilities(config.qubits, label=config.label)
            else:
                modified.save_probabilities(label=config.label)
        
        elif config.snapshot_type == SnapshotType.EXPECTATION_VALUE:
            if config.observable is None:
                raise ValueError("Observable required for expectation_value snapshot")
            modified.save_expectation_value(
                config.observable,
                config.qubits or list(range(circuit.num_qubits)),
                label=config.label,
                variance=config.variance,
            )
        
        elif config.snapshot_type == SnapshotType.STABILIZER:
            modified.save_stabilizer(label=config.label)
        
        elif config.snapshot_type == SnapshotType.MATRIX_PRODUCT_STATE:
            modified.save_matrix_product_state(label=config.label)
        
        return modified
    
    def add_multiple_snapshots(
        self,
        circuit: Any,
        configs: list[SnapshotConfig],
    ) -> Any:
        """Add multiple snapshots to a circuit.
        
        Args:
            circuit: Qiskit QuantumCircuit
            configs: List of snapshot configurations
            
        Returns:
            Modified circuit with all snapshots
        """
        result = circuit
        for config in configs:
            result = self.add_snapshot(result, config)
        return result
    
    def add_snapshots_at_layers(
        self,
        circuit: Any,
        snapshot_type: SnapshotType = SnapshotType.STATEVECTOR,
        layer_indices: list[int] | None = None,
    ) -> Any:
        """Add snapshots at specific circuit layers.
        
        Args:
            circuit: Qiskit QuantumCircuit
            snapshot_type: Type of snapshot to add
            layer_indices: Which layers to snapshot (None for all)
            
        Returns:
            New circuit with snapshots
        """
        qiskit = self._get_qiskit()
        from qiskit import QuantumCircuit
        
        # Decompose circuit to layers
        num_qubits = circuit.num_qubits
        num_clbits = circuit.num_clbits
        new_circuit = QuantumCircuit(num_qubits, num_clbits)
        
        current_layer = 0
        layers_to_snapshot = set(layer_indices) if layer_indices else None
        
        for instr, qargs, cargs in circuit.data:
            new_circuit.append(instr, qargs, cargs)
            current_layer += 1
            
            should_snapshot = (
                layers_to_snapshot is None or current_layer in layers_to_snapshot
            )
            
            if should_snapshot:
                config = SnapshotConfig(
                    snapshot_type=snapshot_type,
                    label=f"layer_{current_layer}",
                )
                new_circuit = self.add_snapshot(new_circuit, config)
        
        return new_circuit
    
    def extract_snapshots(
        self,
        result: Any,
        circuit: Any,
    ) -> list[SnapshotResult]:
        """Extract snapshot data from execution result.
        
        Args:
            result: Qiskit Result object
            circuit: The executed circuit
            
        Returns:
            List of SnapshotResult objects
        """
        snapshots = []
        
        try:
            result_data = result.data(circuit)
            
            for key, value in result_data.items():
                if key.startswith("snapshot_") or any(
                    st.value in key for st in SnapshotType
                ):
                    # Determine snapshot type from key
                    snap_type = SnapshotType.STATEVECTOR
                    for st in SnapshotType:
                        if st.value in key:
                            snap_type = st
                            break
                    
                    snapshots.append(SnapshotResult(
                        label=key,
                        snapshot_type=snap_type,
                        data=value,
                    ))
        except Exception as e:
            logger.warning(f"Failed to extract snapshots: {e}")
        
        return snapshots


# ==============================================================================
# COMPREHENSIVE NOISE MODEL SUPPORT
# ==============================================================================


class NoiseModelType(Enum):
    """Types of noise models."""
    
    DEPOLARIZING = "depolarizing"
    THERMAL = "thermal"
    READOUT = "readout"
    AMPLITUDE_DAMPING = "amplitude_damping"
    PHASE_DAMPING = "phase_damping"
    BITFLIP = "bitflip"
    PHASEFLIP = "phaseflip"
    PAULI = "pauli"
    KRAUS = "kraus"
    CUSTOM = "custom"


@dataclass
class NoiseParameters:
    """Parameters for noise model configuration."""
    
    # Single-qubit gate error rate
    single_qubit_error: float = 0.001
    
    # Two-qubit gate error rate
    two_qubit_error: float = 0.01
    
    # Readout error rate
    readout_error: float = 0.01
    
    # T1 relaxation time (microseconds)
    t1: float = 50.0
    
    # T2 dephasing time (microseconds)
    t2: float = 70.0
    
    # Single-qubit gate time (nanoseconds)
    gate_time_1q: float = 50.0
    
    # Two-qubit gate time (nanoseconds)
    gate_time_2q: float = 300.0
    
    # Measurement time (nanoseconds)
    measurement_time: float = 1000.0


@dataclass
class NoiseModelConfig:
    """Configuration for comprehensive noise modeling."""
    
    # Primary noise type
    noise_type: NoiseModelType = NoiseModelType.DEPOLARIZING
    
    # Noise parameters
    parameters: NoiseParameters = field(default_factory=NoiseParameters)
    
    # Gates to apply noise to
    target_gates: list[str] | None = None
    
    # Specific qubits to apply noise to (None for all)
    target_qubits: list[int] | None = None
    
    # Whether to include readout errors
    include_readout_errors: bool = True
    
    # Whether to include thermal relaxation
    include_thermal_relaxation: bool = False
    
    # Custom Kraus operators
    custom_kraus: list[np.ndarray] | None = None


@dataclass
class NoiseAnalysisResult:
    """Result of noise model analysis."""
    
    noise_type: str
    total_error_rate: float
    single_qubit_fidelity: float
    two_qubit_fidelity: float
    readout_fidelity: float
    coherent_error_fraction: float
    incoherent_error_fraction: float
    error_channels: list[str] = field(default_factory=list)


class NoiseModelManager:
    """Comprehensive noise model management for Qiskit Aer.
    
    Provides:
    - Multiple noise model types
    - Fine-grained qubit/gate targeting
    - Noise model composition
    - Analysis and validation
    """
    
    # Standard gate sets for noise application
    SINGLE_QUBIT_GATES = ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 't', 'id']
    TWO_QUBIT_GATES = ['cx', 'cy', 'cz', 'swap', 'iswap', 'ecr']
    
    def __init__(self) -> None:
        """Initialize noise model manager."""
        self._cached_models: dict[str, Any] = {}
    
    def create_noise_model(
        self,
        config: NoiseModelConfig,
    ) -> Any:
        """Create a noise model from configuration.
        
        Args:
            config: Noise model configuration
            
        Returns:
            Qiskit Aer NoiseModel
        """
        try:
            from qiskit_aer.noise import (
                NoiseModel,
                depolarizing_error,
                thermal_relaxation_error,
                amplitude_damping_error,
                phase_damping_error,
                pauli_error,
                ReadoutError,
                kraus_error,
            )
        except ImportError:
            raise BackendNotInstalledError("qiskit", ["qiskit", "qiskit-aer"])
        
        noise_model = NoiseModel()
        params = config.parameters
        
        target_1q_gates = config.target_gates or self.SINGLE_QUBIT_GATES
        target_2q_gates = [g for g in (config.target_gates or self.TWO_QUBIT_GATES) 
                          if g in self.TWO_QUBIT_GATES]
        
        # Add primary noise type
        if config.noise_type == NoiseModelType.DEPOLARIZING:
            error_1q = depolarizing_error(params.single_qubit_error, 1)
            error_2q = depolarizing_error(params.two_qubit_error, 2)
            
            if config.target_qubits:
                for qubit in config.target_qubits:
                    noise_model.add_quantum_error(error_1q, target_1q_gates, [qubit])
            else:
                noise_model.add_all_qubit_quantum_error(error_1q, target_1q_gates)
            
            noise_model.add_all_qubit_quantum_error(error_2q, target_2q_gates)
        
        elif config.noise_type == NoiseModelType.AMPLITUDE_DAMPING:
            error_1q = amplitude_damping_error(params.single_qubit_error)
            noise_model.add_all_qubit_quantum_error(error_1q, target_1q_gates)
        
        elif config.noise_type == NoiseModelType.PHASE_DAMPING:
            error_1q = phase_damping_error(params.single_qubit_error)
            noise_model.add_all_qubit_quantum_error(error_1q, target_1q_gates)
        
        elif config.noise_type == NoiseModelType.BITFLIP:
            error_1q = pauli_error([('X', params.single_qubit_error), 
                                     ('I', 1 - params.single_qubit_error)])
            noise_model.add_all_qubit_quantum_error(error_1q, target_1q_gates)
        
        elif config.noise_type == NoiseModelType.PHASEFLIP:
            error_1q = pauli_error([('Z', params.single_qubit_error), 
                                     ('I', 1 - params.single_qubit_error)])
            noise_model.add_all_qubit_quantum_error(error_1q, target_1q_gates)
        
        elif config.noise_type == NoiseModelType.PAULI:
            # General Pauli channel
            px = params.single_qubit_error / 3
            py = params.single_qubit_error / 3
            pz = params.single_qubit_error / 3
            pi = 1 - px - py - pz
            error_1q = pauli_error([('X', px), ('Y', py), ('Z', pz), ('I', pi)])
            noise_model.add_all_qubit_quantum_error(error_1q, target_1q_gates)
        
        elif config.noise_type == NoiseModelType.KRAUS:
            if config.custom_kraus is None:
                raise ValueError("Kraus operators required for KRAUS noise type")
            error = kraus_error(config.custom_kraus)
            noise_model.add_all_qubit_quantum_error(error, target_1q_gates)
        
        # Add thermal relaxation if requested
        if config.include_thermal_relaxation or config.noise_type == NoiseModelType.THERMAL:
            t1_ns = params.t1 * 1000  # Convert to ns
            t2_ns = params.t2 * 1000
            
            error_1q = thermal_relaxation_error(t1_ns, t2_ns, params.gate_time_1q)
            noise_model.add_all_qubit_quantum_error(error_1q, target_1q_gates)
            
            # For 2-qubit gates, tensor the thermal errors
            error_2q = thermal_relaxation_error(t1_ns, t2_ns, params.gate_time_2q).tensor(
                thermal_relaxation_error(t1_ns, t2_ns, params.gate_time_2q)
            )
            noise_model.add_all_qubit_quantum_error(error_2q, target_2q_gates)
        
        # Add readout errors if requested
        if config.include_readout_errors:
            p0_1 = params.readout_error  # P(1|0)
            p1_0 = params.readout_error  # P(0|1)
            readout_err = ReadoutError([[1 - p0_1, p0_1], [p1_0, 1 - p1_0]])
            noise_model.add_all_qubit_readout_error(readout_err)
        
        return noise_model
    
    def compose_noise_models(
        self,
        models: list[Any],
    ) -> Any:
        """Compose multiple noise models into one.
        
        Args:
            models: List of NoiseModel objects
            
        Returns:
            Combined NoiseModel
        """
        if not models:
            from qiskit_aer.noise import NoiseModel
            return NoiseModel()
        
        combined = models[0]
        for model in models[1:]:
            # Merge error channels
            for qerror in model._local_quantum_errors:
                combined.add_quantum_error(qerror[0], qerror[1], qerror[2])
            for rerror in model._local_readout_errors:
                combined.add_readout_error(rerror[0], rerror[1])
        
        return combined
    
    def analyze_noise_model(
        self,
        noise_model: Any,
        num_qubits: int = 1,
    ) -> NoiseAnalysisResult:
        """Analyze a noise model's characteristics.
        
        Args:
            noise_model: Qiskit Aer NoiseModel
            num_qubits: Number of qubits in the system
            
        Returns:
            NoiseAnalysisResult with analysis
        """
        # Calculate approximate fidelities
        quantum_errors = []
        total_1q_error = 0.0
        total_2q_error = 0.0
        readout_fidelity = 1.0
        
        try:
            # Analyze quantum errors
            if hasattr(noise_model, '_local_quantum_errors'):
                for error_data in noise_model._local_quantum_errors:
                    error = error_data[0]
                    gates = error_data[1]
                    
                    # Approximate error rate
                    probabilities = error.probabilities
                    error_rate = 1 - max(probabilities)
                    
                    if any(g in self.TWO_QUBIT_GATES for g in gates):
                        total_2q_error += error_rate
                    else:
                        total_1q_error += error_rate
                    
                    quantum_errors.append(type(error).__name__)
            
            # Analyze readout errors
            if hasattr(noise_model, '_local_readout_errors'):
                for error_data in noise_model._local_readout_errors:
                    matrix = error_data[0].probabilities
                    # Approximate readout fidelity as average of diagonal
                    readout_fidelity *= (matrix[0][0] + matrix[1][1]) / 2
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
        
        return NoiseAnalysisResult(
            noise_type=str(noise_model),
            total_error_rate=total_1q_error + total_2q_error,
            single_qubit_fidelity=1 - total_1q_error,
            two_qubit_fidelity=1 - total_2q_error,
            readout_fidelity=readout_fidelity,
            coherent_error_fraction=0.0,  # Would need more analysis
            incoherent_error_fraction=1.0,
            error_channels=list(set(quantum_errors)),
        )
    
    def create_device_noise_model(
        self,
        backend: Any,
    ) -> Any:
        """Create noise model from a real device backend.
        
        Args:
            backend: Qiskit Backend object
            
        Returns:
            NoiseModel matching the device
        """
        from qiskit_aer.noise import NoiseModel
        return NoiseModel.from_backend(backend)


# ==============================================================================
# MAIN ADAPTER CLASS
# ==============================================================================


class QiskitBackendAdapter(BaseBackendAdapter):
    """Qiskit Aer backend adapter with comprehensive simulation support.

    Supports:
    - State vector and density matrix simulation
    - GPU acceleration (prerequisite for cuQuantum)
    - Advanced transpilation options
    - Comprehensive snapshot modes
    - Full noise model support
    - Parameter binding for variational circuits
    """

    def __init__(self) -> None:
        """Initialize the Qiskit adapter."""
        self._cached_version: str | None = None
        self._gpu_manager = GPUManager()
        self._transpiler = AdvancedTranspiler()
        self._snapshot_manager = SnapshotManager()
        self._noise_manager = NoiseModelManager()
        self._gpu_config = GPUConfiguration()

    # ------------------------------------------------------------------
    # Benchmarking hooks
    # ------------------------------------------------------------------
    def prepare_for_benchmark(self, circuit: Any | None = None, shots: int | None = None) -> None:
        """Reset transient managers to avoid cache effects during benchmarks."""
        # Fresh GPU detection/transpiler state per benchmark run
        self._cached_version = None
        self._transpiler = AdvancedTranspiler()
        self._snapshot_manager = SnapshotManager()
        self._noise_manager = NoiseModelManager()
        # Ensure GPU status is refreshed if benchmarks toggle GPU usage
        self._gpu_manager = GPUManager()

    def cleanup_after_benchmark(self) -> None:
        """No-op cleanup hook for Qiskit adapter (placeholder)."""
        return

    def get_name(self) -> str:
        return "qiskit"

    def get_version(self) -> str:
        if self._cached_version:
            return self._cached_version

        spec = importlib.util.find_spec("qiskit")
        if spec and spec.loader:
            try:
                import qiskit

                self._cached_version = getattr(qiskit, "__version__", "unknown")
                return self._cached_version
            except Exception:
                return "unknown"
        return "unavailable"

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("qiskit") is not None
            and importlib.util.find_spec("qiskit_aer") is not None
        )

    def get_capabilities(self) -> Capabilities:
        gpu_status = self._gpu_manager.detect_gpu()
        
        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=28,
            supports_noise=True,
            supports_gpu=gpu_status.is_available,
            supports_batching=True,
            custom_features={
                "transpilation": True,
                "optimization_levels": [0, 1, 2, 3],
                "snapshot_modes": [st.value for st in SnapshotType],
                "noise_models": [nt.value for nt in NoiseModelType],
                "parameter_binding": True,
                "advanced_transpilation": True,
                "gpu_acceleration": gpu_status.is_available,
                "custatevec_available": gpu_status.custatevec_available,
                "cutensornet_available": gpu_status.cutensornet_available,
            },
        )

    # ==========================================================================
    # GPU SUPPORT
    # ==========================================================================

    def get_gpu_status(self) -> GPUStatus:
        """Get current GPU status."""
        return self._gpu_manager.detect_gpu()

    def configure_gpu(self, config: GPUConfiguration) -> None:
        """Configure GPU settings for future executions."""
        self._gpu_config = config

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_manager.detect_gpu().is_available

    # ==========================================================================
    # ADVANCED TRANSPILATION
    # ==========================================================================

    def get_transpiler(self) -> AdvancedTranspiler:
        """Get the advanced transpiler."""
        return self._transpiler

    def transpile_circuit(
        self,
        circuit: Any,
        config: TranspilationConfig | None = None,
    ) -> TranspilationResult:
        """Transpile a circuit with advanced options."""
        return self._transpiler.transpile(circuit, config)

    # ==========================================================================
    # SNAPSHOT MANAGEMENT
    # ==========================================================================

    def get_snapshot_manager(self) -> SnapshotManager:
        """Get the snapshot manager."""
        return self._snapshot_manager

    def add_snapshot(
        self,
        circuit: Any,
        snapshot_type: str = "statevector",
        label: str = "snapshot",
        qubits: list[int] | None = None,
    ) -> Any:
        """Add a snapshot instruction to the circuit."""
        config = SnapshotConfig(
            snapshot_type=SnapshotType(snapshot_type),
            label=label,
            qubits=qubits,
        )
        return self._snapshot_manager.add_snapshot(circuit, config)

    # ==========================================================================
    # NOISE MODEL MANAGEMENT
    # ==========================================================================

    def get_noise_manager(self) -> NoiseModelManager:
        """Get the noise model manager."""
        return self._noise_manager

    def create_noise_model(
        self,
        error_type: str = "depolarizing",
        single_qubit_error: float = 0.001,
        two_qubit_error: float = 0.01,
        **kwargs: Any,
    ) -> Any:
        """Create a noise model for noisy simulation."""
        config = NoiseModelConfig(
            noise_type=NoiseModelType(error_type),
            parameters=NoiseParameters(
                single_qubit_error=single_qubit_error,
                two_qubit_error=two_qubit_error,
                **{k: v for k, v in kwargs.items() if k in NoiseParameters.__dataclass_fields__},
            ),
            include_readout_errors=kwargs.get("include_readout_errors", True),
            include_thermal_relaxation=kwargs.get("include_thermal_relaxation", False),
        )
        return self._noise_manager.create_noise_model(config)

    def analyze_noise_model(self, noise_model: Any) -> NoiseAnalysisResult:
        """Analyze a noise model's characteristics."""
        return self._noise_manager.analyze_noise_model(noise_model)

    # ==========================================================================
    # CIRCUIT VALIDATION
    # ==========================================================================

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        if not self.is_available():
            return ValidationResult(
                valid=False, message="qiskit/qiskit-aer not installed"
            )
        try:
            from qiskit import QuantumCircuit
        except Exception as exc:
            return ValidationResult(valid=False, message=f"qiskit import failed: {exc}")

        if not isinstance(circuit, QuantumCircuit):
            return ValidationResult(
                valid=False, message="input is not a qiskit.QuantumCircuit"
            )

        if circuit.parameters:
            return ValidationResult(
                valid=True,
                message="Circuit has unbound parameters - requires parameter binding",
                details={
                    "requires_params": True,
                    "parameters": [str(p) for p in circuit.parameters],
                },
            )

        return ValidationResult(valid=True, message="ok")

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "qiskit/qiskit-aer not installed"},
            )
        try:
            from qiskit import QuantumCircuit
        except Exception:
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "qiskit import failed"},
            )

        if not isinstance(circuit, QuantumCircuit):
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "not a QuantumCircuit"},
            )

        qubits = circuit.num_qubits
        depth = circuit.depth() or 0
        gate_count = circuit.size()
        memory_mb = (2**qubits * 16) / (1024 * 1024) if qubits <= 28 else None

        time_ms = gate_count * 0.01 + depth * 0.1 if gate_count > 0 else None

        two_qubit_gates = sum(
            1 for instr, _, _ in circuit.data
            if hasattr(instr, 'num_qubits') and instr.num_qubits >= 2
        )

        metadata = {
            "qubits": qubits,
            "depth": depth,
            "gate_count": gate_count,
            "two_qubit_gates": two_qubit_gates,
        }
        return ResourceEstimate(memory_mb=memory_mb, time_ms=time_ms, metadata=metadata)

    def bind_parameters(
        self,
        circuit: Any,
        params: dict[str, float] | list[float],
    ) -> Any:
        """Bind parameters to a variational circuit.
        
        Note: Uses assign_parameters for Qiskit 1.0+ compatibility.
        """
        if not self.is_available():
            raise BackendNotInstalledError("qiskit", ["qiskit", "qiskit-aer"])

        from qiskit import QuantumCircuit

        if not isinstance(circuit, QuantumCircuit):
            raise CircuitValidationError(
                backend_name="qiskit",
                reason="Expected qiskit.QuantumCircuit",
            )

        if isinstance(params, list):
            if len(params) != len(circuit.parameters):
                raise ValueError(
                    f"Parameter count mismatch: {len(params)} provided, "
                    f"{len(circuit.parameters)} required"
                )
            param_dict = dict(zip(circuit.parameters, params))
        else:
            param_dict = {
                p: params[str(p)]
                for p in circuit.parameters
                if str(p) in params
            }

        # Use assign_parameters for Qiskit 1.0+ compatibility
        # (bind_parameters was deprecated in favor of assign_parameters)
        return circuit.assign_parameters(param_dict)

    # ==========================================================================
    # EXECUTION
    # ==========================================================================

    def execute(
        self, circuit: Any, options: dict[str, Any] | None = None
    ) -> ExecutionResult:
        if not self.is_available():
            raise BackendNotInstalledError("qiskit", ["qiskit", "qiskit-aer"])

        validation = self.validate_circuit(circuit)
        if not validation.valid:
            raise CircuitValidationError(
                backend_name="qiskit",
                reason=validation.message or "Invalid circuit",
            )

        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise BackendNotInstalledError(
                "qiskit", ["qiskit", "qiskit-aer"], original_exception=exc
            )

        if not isinstance(circuit, QuantumCircuit):
            raise CircuitValidationError(
                backend_name="qiskit",
                reason="Expected qiskit.QuantumCircuit",
                circuit_info={"type": type(circuit).__name__},
            )

        qubit_count = circuit.num_qubits
        max_qubits = self.get_capabilities().max_qubits
        if qubit_count > max_qubits:
            raise QubitLimitExceededError(
                backend_name="qiskit",
                requested_qubits=qubit_count,
                max_qubits=max_qubits,
            )

        options = options or {}
        sim_type = options.get("simulator_type", SimulatorType.STATE_VECTOR)
        shots = int(options.get("shots", options.get("repetitions", 0)))
        density_mode = sim_type == SimulatorType.DENSITY_MATRIX
        noise_model = options.get("noise_model")
        optimization_level = int(options.get("optimization_level", 1))
        params = options.get("params", options.get("parameters"))
        snapshots = options.get("snapshots", [])
        
        # GPU configuration
        gpu_config = options.get("gpu_config", self._gpu_config)
        
        # Advanced transpilation config
        transpilation_config = options.get("transpilation_config")

        if sim_type not in (SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX):
            raise ValueError(f"Unsupported simulator type: {sim_type}")

        try:
            # Bind parameters if provided
            exec_circuit = circuit
            if params:
                exec_circuit = self.bind_parameters(circuit, params)

            # Configure simulator
            method = "density_matrix" if density_mode else "statevector"
            
            sim_options = {"method": method}
            if noise_model:
                sim_options["noise_model"] = noise_model
            
            # Add GPU options
            if gpu_config.enabled:
                gpu_opts = self._gpu_manager.configure_simulator(gpu_config)
                sim_options.update(gpu_opts)
            
            simulator = AerSimulator(**sim_options)
            
            # Transpile with advanced options if provided
            if transpilation_config:
                transpile_result = self._transpiler.transpile(
                    exec_circuit, transpilation_config, simulator
                )
                t_circuit = transpile_result.circuit
            else:
                t_circuit = transpile(
                    exec_circuit,
                    simulator,
                    optimization_level=optimization_level,
                )

            final_circuit = t_circuit.copy()

            # Add snapshots if requested
            for snap in snapshots:
                if isinstance(snap, str):
                    final_circuit = self.add_snapshot(final_circuit, snap)
                elif isinstance(snap, dict):
                    final_circuit = self.add_snapshot(
                        final_circuit,
                        snap.get("type", "statevector"),
                        snap.get("label", "snapshot"),
                        snap.get("qubits"),
                    )
                elif isinstance(snap, SnapshotConfig):
                    final_circuit = self._snapshot_manager.add_snapshot(final_circuit, snap)

            if shots == 0:
                if density_mode:
                    final_circuit.save_density_matrix()
                else:
                    final_circuit.save_statevector()

            start = time.perf_counter()
            result = simulator.run(
                final_circuit, shots=shots if shots > 0 else None
            ).result()
            execution_time_ms = (time.perf_counter() - start) * 1000.0

            if shots > 0:
                counts = result.get_counts(final_circuit)
                data = {"counts": counts, "shots": shots}
                result_type = ResultType.COUNTS
                raw_result = result
                result_data = result.data(final_circuit)  # Get data for potential snapshots
            else:
                result_data = result.data(final_circuit)
                if density_mode:
                    density_matrix = result_data.get("density_matrix")
                    result_type = ResultType.DENSITY_MATRIX
                    data = {"density_matrix": density_matrix}
                    raw_result = result
                else:
                    statevector = result_data.get("statevector")
                    result_type = ResultType.STATEVECTOR
                    data = {"statevector": statevector}
                    raw_result = result

            # Add any snapshot data
            for key in result_data.keys():
                if key.startswith("snapshot_") or key not in ("statevector", "density_matrix", "counts"):
                    if key not in data:
                        data[key] = result_data[key]

            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=sim_type,
                execution_time_ms=execution_time_ms,
                qubit_count=qubit_count,
                shot_count=shots if shots > 0 else None,
                result_type=result_type,
                data=data,
                metadata={
                    "qiskit_version": self.get_version(),
                    "optimization_level": optimization_level,
                    "noisy": noise_model is not None,
                    "parameterized": params is not None,
                    "gpu_enabled": gpu_config.enabled,
                },
                raw_result=raw_result,
            )

        except (
            BackendNotInstalledError,
            CircuitValidationError,
            QubitLimitExceededError,
        ):
            raise
        except Exception as exc:
            raise wrap_backend_exception(exc, "qiskit", "execution")

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        return sim_type in self.get_capabilities().simulator_types

    def get_supported_gates(self) -> list[str]:
        """Get list of gates supported by Qiskit."""
        return [
            "h", "x", "y", "z", "s", "sdg", "t", "tdg",
            "rx", "ry", "rz", "u", "u1", "u2", "u3",
            "cx", "cy", "cz", "swap", "iswap",
            "ccx", "cswap",
            "reset", "measure", "barrier",
        ]
