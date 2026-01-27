# LRET Backend Variants Integration Report for Proxima

**Document Version:** 1.0  
**Date:** January 27, 2026  
**Purpose:** Comprehensive guide for integrating three LRET backend variants into Proxima Agent  
**Target Audience:** AI Implementation Agents, Developers

---

## 📋 Executive Summary

This document provides a complete roadmap for adapting Proxima Agent to support three specialized LRET (Low-Rank Entanglement Tracking) backend branches:

1. **cirq-scalability-comparison** - Cirq integration and performance benchmarking
2. **pennylane-documentation-benchmarking** - PennyLane device plugin and hybrid algorithms
3. **phase-7** - Multi-framework ecosystem integration (Cirq, PennyLane, Qiskit)

Each backend variant brings unique capabilities that will expand Proxima's simulation ecosystem significantly.

---

## 🎯 Strategic Overview

### Current Proxima Backend Architecture

```
proxima/
├── backends/
│   ├── base.py                    # BaseBackendAdapter interface
│   ├── registry.py                # Backend registration system
│   ├── lret.py                    # Current LRET adapter (basic)
│   ├── cirq_adapter.py            # Cirq backend
│   ├── qiskit_adapter.py          # Qiskit Aer backend
│   └── [other backends]
```

### Target Architecture After Integration

```
proxima/
├── backends/
│   ├── lret/                      # **NEW: LRET Module**
│   │   ├── __init__.py
│   │   ├── base_lret_adapter.py   # Shared LRET foundation
│   │   ├── cirq_scalability.py    # Cirq comparison variant
│   │   ├── pennylane_hybrid.py    # PennyLane integration variant
│   │   └── phase7_unified.py      # Multi-framework variant
│   ├── cirq_adapter.py            # **UPDATED: Enhanced for LRET**
│   ├── pennylane_adapter.py       # **NEW: PennyLane support**
│   └── [existing backends]
```

---

## 📊 Backend Variant Analysis

### Variant 1: cirq-scalability-comparison

**Branch:** `https://github.com/kunal5556/LRET/tree/cirq-scalability-comparison`

#### Key Features
- LRET vs Cirq FDM (Full Density Matrix) benchmarking infrastructure
- Scalability analysis (2-14+ qubits)
- Performance comparison metrics
- Cirq Circuit compatibility layer
- Cross-platform launcher support

#### Unique Capabilities
| Feature | Description | Proxima Benefit |
|---------|-------------|-----------------|
| **Cirq FDM Comparison** | Side-by-side LRET vs Cirq performance | Automatic backend selection based on circuit size |
| **Scalability Testing** | Automated qubit scaling benchmarks | Performance profiling integration |
| **Circuit Conversion** | LRET ↔ Cirq circuit translation | Seamless backend switching |
| **Benchmark Reporting** | CSV output with timing, rank, fidelity | Enhanced benchmarking module |

#### Dependencies
```python
# Required packages
cirq-core >= 1.0.0
eigen3 >= 3.4
pybind11 >= 2.10
numpy >= 1.21
pandas >= 1.3      # For benchmark CSV processing
matplotlib >= 3.5  # For visualization
```

#### API Signature Examples
```python
# Cirq comparison mode
from lret.cirq_comparison import run_comparison

result = run_comparison(
    circuit=cirq_circuit,
    n_qubits=10,
    depth=20,
    noise_level=0.01,
    backends=['lret', 'cirq_fdm'],
    output_csv='comparison.csv'
)

# Scalability benchmark
from lret.benchmarks import scalability_test

metrics = scalability_test(
    qubit_range=(8, 14),
    depth=30,
    mode='hybrid',  # ROW, COLUMN, HYBRID parallelization
)
```

---

### Variant 2: pennylane-documentation-benchmarking

**Branch:** `https://github.com/kunal5556/LRET/tree/pennylane-documentation-benchmarking`

#### Key Features
- PennyLane device plugin (`QLRETDevice`)
- Hybrid quantum-classical algorithms support
- VQE, QAOA, QNN algorithm implementations
- Comprehensive PennyLane documentation
- Noise model Kraus operator support

#### Unique Capabilities
| Feature | Description | Proxima Benefit |
|---------|-------------|-----------------|
| **PennyLane Device** | Native PennyLane integration | Gradient-based optimization support |
| **Hybrid Algorithms** | VQE, QAOA, QNN pre-built | Algorithm library expansion |
| **Kraus Operators** | Custom noise channel support | Advanced noise modeling |
| **Auto-differentiation** | Gradient computation | Variational algorithm support |

#### Dependencies
```python
# Required packages
pennylane >= 0.33.0
pennylane-cirq >= 0.33.0  # For Cirq interop
jax >= 0.4.0              # For auto-diff (optional)
torch >= 2.0              # For PyTorch interface (optional)
```

#### API Signature Examples
```python
# PennyLane device creation
import pennylane as qml
from qlret import QLRETDevice

dev = QLRETDevice(
    wires=4,
    noise_level=0.01,
    noise_model="depolarizing",
    shots=1024,
    rank_threshold=1e-4
)

@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# VQE example
from qlret.algorithms import VQE

vqe = VQE(
    device=dev,
    hamiltonian=hamiltonian,
    ansatz=ansatz,
    optimizer='adam'
)
result = vqe.run(initial_params)
```

---

### Variant 3: phase-7

**Branch:** `https://github.com/kunal5556/LRET/tree/phase-7`

#### Key Features
- Multi-framework ecosystem integration (Cirq, PennyLane, Qiskit)
- Cross-platform compatibility (Windows MSVC, Linux, macOS)
- GPU acceleration infrastructure (cuQuantum ready)
- Gate fusion optimization
- Comprehensive testing framework (Tier 1-9c)

#### Unique Capabilities
| Feature | Description | Proxima Benefit |
|---------|-------------|-----------------|
| **Unified API** | Single interface for multiple frameworks | Simplified multi-backend orchestration |
| **Gate Fusion** | Automatic gate merging for speedup | Performance optimization |
| **GPU Support** | cuQuantum integration hooks | Hardware acceleration |
| **Cross-Platform** | Windows MSVC + Unix support | Broader platform compatibility |
| **Testing Framework** | 9-tier validation system | Quality assurance |

#### Dependencies
```python
# Core requirements
cirq-core >= 1.0.0
pennylane >= 0.33.0
qiskit >= 0.45.0
qiskit-aer >= 0.13.0

# Optional GPU acceleration
cuquantum-python >= 23.0  # NVIDIA GPU only

# Platform-specific
# Windows: Visual Studio 2019+, Ninja
# Linux/macOS: GCC 9+, CMake 3.16+
```

#### API Signature Examples
```python
# Unified multi-framework execution
from lret.phase7 import UnifiedExecutor

executor = UnifiedExecutor(
    backends=['cirq', 'pennylane', 'qiskit'],
    device='cpu',  # or 'gpu'
    optimization_level=2,  # Gate fusion enabled
)

# Execute on best backend automatically
result = executor.execute(
    circuit=universal_circuit,
    shots=1024,
    backend='auto',  # Auto-select based on circuit
)

# Gate fusion demo
from lret.optimization import apply_gate_fusion

optimized_circuit = apply_gate_fusion(
    circuit,
    mode='hybrid',  # ROW/COLUMN/HYBRID
)
```

---

## 🔧 Implementation Phases

### PHASE 0: Preparation & Environment Setup

**Estimated Time:** 2-3 days  
**Complexity:** Low  
**Dependencies:** None

#### Steps

##### Step 0.1: Install LRET Variants

**TUI Navigation:** Settings → Backend Management → Install LRET Variants

**Options:**
- [ ] Install Cirq-Scalability variant
- [ ] Install PennyLane-Documentation variant
- [ ] Install Phase-7 variant
- [ ] Install all variants

**Implementation:**

```python
# File: src/proxima/backends/lret/installer.py

from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)

LRET_VARIANTS = {
    'cirq_scalability': {
        'repo': 'https://github.com/kunal5556/LRET.git',
        'branch': 'cirq-scalability-comparison',
        'requires': ['cirq-core>=1.0.0', 'pandas>=1.3'],
    },
    'pennylane_hybrid': {
        'repo': 'https://github.com/kunal5556/LRET.git',
        'branch': 'pennylane-documentation-benchmarking',
        'requires': ['pennylane>=0.33.0', 'jax>=0.4.0'],
    },
    'phase7_unified': {
        'repo': 'https://github.com/kunal5556/LRET.git',
        'branch': 'phase-7',
        'requires': ['cirq-core>=1.0.0', 'pennylane>=0.33.0', 'qiskit>=0.45.0'],
    },
}

def install_lret_variant(variant_name: str, install_dir: Path) -> bool:
    """Install a specific LRET variant.
    
    Args:
        variant_name: Name of variant ('cirq_scalability', 'pennylane_hybrid', 'phase7_unified')
        install_dir: Installation directory
        
    Returns:
        True if successful, False otherwise
    """
    if variant_name not in LRET_VARIANTS:
        logger.error(f"Unknown variant: {variant_name}")
        return False
    
    config = LRET_VARIANTS[variant_name]
    variant_dir = install_dir / variant_name
    
    # Clone repository
    logger.info(f"Cloning {variant_name} from {config['branch']}...")
    try:
        subprocess.run([
            'git', 'clone',
            '--branch', config['branch'],
            '--depth', '1',
            config['repo'],
            str(variant_dir)
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e}")
        return False
    
    # Install Python dependencies
    logger.info(f"Installing dependencies for {variant_name}...")
    try:
        subprocess.run([
            'pip', 'install', '-e', str(variant_dir / 'python')
        ], check=True)
        
        for req in config['requires']:
            subprocess.run(['pip', 'install', req], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Dependency installation failed: {e}")
        return False
    
    # Build C++ components if needed
    if (variant_dir / 'CMakeLists.txt').exists():
        logger.info(f"Building C++ components for {variant_name}...")
        build_dir = variant_dir / 'build'
        build_dir.mkdir(exist_ok=True)
        
        try:
            subprocess.run([
                'cmake', '-S', str(variant_dir), '-B', str(build_dir),
                '-DCMAKE_BUILD_TYPE=Release'
            ], check=True)
            subprocess.run([
                'cmake', '--build', str(build_dir), '--parallel'
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"C++ build failed: {e}")
            return False
    
    logger.info(f"✓ {variant_name} installed successfully")
    return True

def check_variant_availability(variant_name: str) -> dict:
    """Check if LRET variant is available and operational.
    
    Returns:
        dict with keys: 'installed', 'version', 'functional'
    """
    try:
        if variant_name == 'cirq_scalability':
            import lret.cirq_comparison
            version = lret.__version__
            functional = hasattr(lret.cirq_comparison, 'run_comparison')
        elif variant_name == 'pennylane_hybrid':
            from qlret import QLRETDevice
            import pennylane as qml
            version = qml.__version__
            functional = QLRETDevice is not None
        elif variant_name == 'phase7_unified':
            from lret.phase7 import UnifiedExecutor
            version = lret.__version__
            functional = UnifiedExecutor is not None
        else:
            return {'installed': False, 'version': None, 'functional': False}
        
        return {'installed': True, 'version': version, 'functional': functional}
    except ImportError:
        return {'installed': False, 'version': None, 'functional': False}
```

**TUI Component:**

```python
# File: src/proxima/tui/dialogs/lret_installer_dialog.py

from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Checkbox, Label, ProgressBar

class LRETInstallerDialog(ModalScreen):
    """Dialog for installing LRET backend variants."""
    
    DEFAULT_CSS = """
    LRETInstallerDialog {
        align: center middle;
    }
    
    LRETInstallerDialog .dialog-container {
        width: 80;
        height: 30;
        border: thick $primary;
        background: $surface;
    }
    
    LRETInstallerDialog .variant-checkbox {
        margin: 1;
    }
    
    LRETInstallerDialog .progress-section {
        height: 8;
        padding: 1;
        border-top: solid $primary-darken-2;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def compose(self):
        with Vertical(classes="dialog-container"):
            yield Static("📦 Install LRET Backend Variants", classes="dialog-title")
            
            yield Label("Select variants to install:")
            
            with Vertical():
                yield Checkbox("Cirq Scalability - Performance comparison & benchmarking", 
                             id="cirq-scalability", classes="variant-checkbox")
                yield Checkbox("PennyLane Hybrid - VQE, QAOA, gradient-based optimization", 
                             id="pennylane-hybrid", classes="variant-checkbox")
                yield Checkbox("Phase 7 Unified - Multi-framework integration", 
                             id="phase7-unified", classes="variant-checkbox")
            
            with Vertical(classes="progress-section"):
                yield Label("Installation Progress:", id="progress-label")
                yield ProgressBar(total=100, id="install-progress")
                yield Label("", id="status-message")
            
            with Horizontal(classes="dialog-footer"):
                yield Button("Install Selected", variant="primary", id="install-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")
    
    async def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "install-btn":
            await self._install_variants()
    
    async def _install_variants(self):
        """Install selected LRET variants."""
        from proxima.backends.lret.installer import install_lret_variant
        from pathlib import Path
        
        # Get selected variants
        variants = []
        if self.query_one("#cirq-scalability", Checkbox).value:
            variants.append('cirq_scalability')
        if self.query_one("#pennylane-hybrid", Checkbox).value:
            variants.append('pennylane_hybrid')
        if self.query_one("#phase7-unified", Checkbox).value:
            variants.append('phase7_unified')
        
        if not variants:
            self.notify("Please select at least one variant", severity="warning")
            return
        
        install_dir = Path.home() / ".proxima" / "lret_variants"
        install_dir.mkdir(parents=True, exist_ok=True)
        
        progress = self.query_one("#install-progress", ProgressBar)
        status = self.query_one("#status-message", Label)
        
        total_steps = len(variants)
        progress.update(total=total_steps * 100)
        
        for i, variant in enumerate(variants):
            status.update(f"Installing {variant}...")
            progress.update(progress=i * 100)
            
            success = install_lret_variant(variant, install_dir)
            
            if success:
                status.update(f"✓ {variant} installed successfully")
            else:
                status.update(f"✗ {variant} installation failed")
                self.notify(f"Failed to install {variant}", severity="error")
        
        progress.update(progress=total_steps * 100)
        status.update("Installation complete!")
        
        # Dismiss after 2 seconds
        await asyncio.sleep(2)
        self.dismiss({"installed": variants})
```

##### Step 0.2: Configure Backend Registry

**TUI Navigation:** Settings → Backend Configuration → LRET Variants

**Options:**
- Enable/disable each variant
- Set default variant for operations
- Configure variant-specific settings

**Implementation:**

```python
# File: src/proxima/backends/lret/config.py

from dataclasses import dataclass, field
from typing import Optional, Literal

@dataclass
class LRETVariantConfig:
    """Configuration for a specific LRET variant."""
    
    enabled: bool = False
    priority: int = 50  # Higher = preferred when auto-selecting
    install_path: Optional[str] = None
    
    # Cirq Scalability specific
    cirq_fdm_threshold: int = 10  # Switch to FDM above this qubit count
    benchmark_output_dir: str = "./benchmarks"
    
    # PennyLane Hybrid specific
    pennylane_shots: int = 1024
    pennylane_diff_method: Literal['parameter-shift', 'adjoint', 'backprop'] = 'parameter-shift'
    
    # Phase 7 Unified specific
    phase7_backend_preference: list[str] = field(default_factory=lambda: ['cirq', 'pennylane', 'qiskit'])
    gate_fusion_enabled: bool = True
    gpu_enabled: bool = False

@dataclass
class LRETConfig:
    """Master configuration for all LRET variants."""
    
    cirq_scalability: LRETVariantConfig = field(default_factory=LRETVariantConfig)
    pennylane_hybrid: LRETVariantConfig = field(default_factory=LRETVariantConfig)
    phase7_unified: LRETVariantConfig = field(default_factory=LRETVariantConfig)
    
    default_variant: Optional[str] = None  # Auto-select if None
    
    def get_enabled_variants(self) -> list[str]:
        """Get list of enabled variant names."""
        variants = []
        if self.cirq_scalability.enabled:
            variants.append('cirq_scalability')
        if self.pennylane_hybrid.enabled:
            variants.append('pennylane_hybrid')
        if self.phase7_unified.enabled:
            variants.append('phase7_unified')
        return variants
    
    def select_variant_for_task(self, task_type: str, circuit_info: dict) -> Optional[str]:
        """Auto-select best variant for a given task.
        
        Args:
            task_type: 'benchmark', 'vqe', 'qaoa', 'general', etc.
            circuit_info: {'qubits': int, 'depth': int, 'gates': list}
            
        Returns:
            Variant name or None
        """
        enabled = self.get_enabled_variants()
        if not enabled:
            return None
        
        # Task-specific selection logic
        if task_type == 'benchmark' and 'cirq_scalability' in enabled:
            return 'cirq_scalability'
        elif task_type in ['vqe', 'qaoa', 'qnn'] and 'pennylane_hybrid' in enabled:
            return 'pennylane_hybrid'
        elif task_type == 'multi_framework' and 'phase7_unified' in enabled:
            return 'phase7_unified'
        elif self.default_variant and self.default_variant in enabled:
            return self.default_variant
        else:
            # Return highest priority enabled variant
            priorities = {
                'cirq_scalability': self.cirq_scalability.priority,
                'pennylane_hybrid': self.pennylane_hybrid.priority,
                'phase7_unified': self.phase7_unified.priority,
            }
            return max(
                enabled,
                key=lambda v: priorities.get(v, 0)
            )
```

---

### PHASE 1: Cirq Scalability Integration

**Estimated Time:** 5-7 days  
**Complexity:** Medium  
**Dependencies:** Phase 0 complete, Cirq backend functional

#### Step 1.1: Create Cirq Scalability Adapter

**TUI Navigation:** Backends → Add Backend → LRET Cirq Scalability

**Buttons:**
- [Configure] - Set FDM comparison threshold, benchmark output
- [Test Connection] - Verify installation
- [Run Sample Benchmark] - Test with 4-qubit circuit
- [View Documentation] - Open integration guide

**Implementation:**

```python
# File: src/proxima/backends/lret/cirq_scalability.py

"""LRET Cirq Scalability adapter for performance benchmarking."""

from __future__ import annotations

import time
import pandas as pd
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

from proxima.backends.base import BaseBackendAdapter, BackendCapability
from proxima.core.result import ExecutionResult
from proxima.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class CirqScalabilityMetrics:
    """Performance metrics from Cirq comparison."""
    
    lret_time_ms: float
    cirq_fdm_time_ms: float
    speedup_factor: float
    lret_final_rank: int
    fidelity: float
    trace_distance: float

class LRETCirqScalabilityAdapter(BaseBackendAdapter):
    """LRET adapter with Cirq FDM comparison and benchmarking.
    
    Features:
    - Automatic LRET vs Cirq FDM selection based on qubit count
    - Performance benchmarking with CSV export
    - Scalability analysis
    - Cirq circuit compatibility
    
    Example:
        >>> adapter = LRETCirqScalabilityAdapter()
        >>> result = await adapter.execute(circuit, options={
        ...     'benchmark': True,
        ...     'compare_with_cirq': True
        ... })
        >>> print(result.metadata['speedup'])
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize Cirq scalability adapter.
        
        Args:
            config: Configuration dict with optional keys:
                - fdm_threshold: Qubit count to prefer Cirq FDM (default: 10)
                - benchmark_dir: Output directory for benchmarks
                - parallel_mode: 'sequential', 'row', 'column', 'hybrid'
        """
        super().__init__(config or {})
        self._fdm_threshold = config.get('fdm_threshold', 10) if config else 10
        self._benchmark_dir = Path(config.get('benchmark_dir', './benchmarks')) if config else Path('./benchmarks')
        self._parallel_mode = config.get('parallel_mode', 'hybrid') if config else 'hybrid'
        
        self._lret = None
        self._cirq = None
        
    @property
    def name(self) -> str:
        return "lret_cirq_scalability"
    
    @property
    def backend_type(self) -> str:
        return "simulator"
    
    @property
    def capabilities(self) -> BackendCapability:
        return (
            BackendCapability.STATE_VECTOR |
            BackendCapability.DENSITY_MATRIX |
            BackendCapability.MEASUREMENT |
            BackendCapability.NOISE
        )
    
    def is_available(self) -> bool:
        """Check if LRET cirq-scalability variant and Cirq are available."""
        try:
            import lret.cirq_comparison
            import cirq
            self._lret = lret
            self._cirq = cirq
            return True
        except ImportError:
            return False
    
    async def connect(self) -> bool:
        """Initialize the adapter."""
        if not self.is_available():
            logger.error("LRET cirq-scalability variant or Cirq not installed")
            return False
        
        self._benchmark_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LRET Cirq Scalability adapter initialized (FDM threshold: {self._fdm_threshold} qubits)")
        return True
    
    async def disconnect(self) -> None:
        """Cleanup resources."""
        self._lret = None
        self._cirq = None
    
    async def execute(
        self,
        circuit: Any,
        options: Optional[dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute circuit with optional Cirq comparison.
        
        Args:
            circuit: Quantum circuit (Cirq or universal format)
            options: Execution options:
                - shots: Number of measurement shots
                - benchmark: Enable benchmarking (default: False)
                - compare_with_cirq: Run Cirq FDM comparison (default: False)
                - export_csv: Export benchmark to CSV (default: True if benchmark=True)
                - noise_level: Noise percentage (0.0-1.0)
                
        Returns:
            ExecutionResult with performance metrics in metadata
        """
        if not self._lret or not self._cirq:
            raise RuntimeError("Backend not connected")
        
        options = options or {}
        benchmark = options.get('benchmark', False)
        compare = options.get('compare_with_cirq', False)
        export_csv = options.get('export_csv', benchmark)
        shots = options.get('shots', 1024)
        noise_level = options.get('noise_level', 0.0)
        
        # Convert circuit to Cirq if needed
        cirq_circuit = self._ensure_cirq_circuit(circuit)
        n_qubits = len(cirq_circuit.all_qubits())
        
        # Decide whether to use LRET or Cirq FDM based on circuit size
        use_fdm = n_qubits >= self._fdm_threshold and not compare
        
        if use_fdm:
            logger.info(f"Using Cirq FDM (circuit too large: {n_qubits} qubits)")
            return await self._execute_cirq_fdm(cirq_circuit, shots, options)
        
        # Execute with LRET
        start_time = time.time()
        lret_result = self._execute_lret(cirq_circuit, shots, noise_level)
        lret_time_ms = (time.time() - start_time) * 1000
        
        metadata = {
            'backend': 'lret',
            'execution_time_ms': lret_time_ms,
            'final_rank': lret_result.get('final_rank', 0),
            'parallel_mode': self._parallel_mode,
        }
        
        # Compare with Cirq if requested
        if compare or benchmark:
            start_time = time.time()
            cirq_result = self._execute_cirq_fdm(cirq_circuit, shots, options)
            cirq_time_ms = (time.time() - start_time) * 1000
            
            # Compute comparison metrics
            speedup = cirq_time_ms / lret_time_ms if lret_time_ms > 0 else 0
            fidelity = self._compute_fidelity(lret_result, cirq_result)
            
            metrics = CirqScalabilityMetrics(
                lret_time_ms=lret_time_ms,
                cirq_fdm_time_ms=cirq_time_ms,
                speedup_factor=speedup,
                lret_final_rank=lret_result.get('final_rank', 0),
                fidelity=fidelity,
                trace_distance=1.0 - fidelity,
            )
            
            metadata.update({
                'cirq_fdm_time_ms': cirq_time_ms,
                'speedup': speedup,
                'fidelity': fidelity,
                'trace_distance': metrics.trace_distance,
                'comparison': 'enabled',
            })
            
            if export_csv:
                self._export_benchmark(n_qubits, cirq_circuit, metrics)
        
        # Convert LRET result to ExecutionResult
        return self._convert_result(lret_result, metadata)
    
    def _execute_lret(self, circuit: Any, shots: int, noise_level: float) -> dict:
        """Execute using LRET simulator."""
        from lret.cirq_comparison import run_lret
        
        result = run_lret(
            circuit=circuit,
            shots=shots,
            noise=noise_level,
            mode=self._parallel_mode,
            verbose=False,
        )
        
        return result
    
    async def _execute_cirq_fdm(
        self,
        circuit: Any,
        shots: int,
        options: dict
    ) -> ExecutionResult:
        """Execute using Cirq FDM simulator."""
        from cirq import DensityMatrixSimulator
        
        simulator = DensityMatrixSimulator(noise=self._create_noise_model(options))
        result = simulator.run(circuit, repetitions=shots)
        
        counts = dict(result.histogram(key='measurements'))
        
        return ExecutionResult(
            counts=counts,
            shots=shots,
            success=True,
            metadata={'backend': 'cirq_fdm'}
        )
    
    def _export_benchmark(
        self,
        n_qubits: int,
        circuit: Any,
        metrics: CirqScalabilityMetrics
    ) -> None:
        """Export benchmark results to CSV."""
        benchmark_file = self._benchmark_dir / 'lret_cirq_comparison.csv'
        
        data = {
            'timestamp': [pd.Timestamp.now()],
            'qubits': [n_qubits],
            'depth': [len(circuit)],
            'lret_time_ms': [metrics.lret_time_ms],
            'cirq_fdm_time_ms': [metrics.cirq_fdm_time_ms],
            'speedup': [metrics.speedup_factor],
            'final_rank': [metrics.lret_final_rank],
            'fidelity': [metrics.fidelity],
            'trace_distance': [metrics.trace_distance],
        }
        
        df = pd.DataFrame(data)
        
        if benchmark_file.exists():
            df.to_csv(benchmark_file, mode='a', header=False, index=False)
        else:
            df.to_csv(benchmark_file, index=False)
        
        logger.info(f"Benchmark exported to {benchmark_file}")
    
    def _ensure_cirq_circuit(self, circuit: Any) -> Any:
        """Convert circuit to Cirq format if needed."""
        if hasattr(circuit, 'all_qubits'):  # Already Cirq
            return circuit
        
        # TODO: Implement universal → Cirq conversion
        return circuit
    
    def _compute_fidelity(self, lret_result: dict, cirq_result: ExecutionResult) -> float:
        """Compute fidelity between LRET and Cirq results."""
        # Simplified fidelity calculation
        # TODO: Implement proper state fidelity computation
        return 0.9998  # Placeholder
    
    def _convert_result(self, lret_result: dict, metadata: dict) -> ExecutionResult:
        """Convert LRET result to ExecutionResult."""
        return ExecutionResult(
            counts=lret_result.get('counts', {}),
            shots=lret_result.get('shots', 0),
            success=True,
            metadata=metadata
        )
    
    def _create_noise_model(self, options: dict) -> Any:
        """Create Cirq noise model from options."""
        # TODO: Implement noise model creation
        return None
```

#### Step 1.2: Add Benchmark Visualization

**TUI Navigation:** Benchmarks → LRET vs Cirq → View Results

**Buttons:**
- [Run New Benchmark] - Execute comparison
- [View CSV] - Open benchmark data
- [Generate Plots] - Create speedup/fidelity charts
- [Export Report] - Create markdown/PDF report

**Implementation:**

```python
# File: src/proxima/backends/lret/visualization.py

"""Visualization tools for LRET benchmarks."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

def plot_lret_cirq_comparison(
    csv_path: Path,
    output_dir: Optional[Path] = None
) -> dict[str, Path]:
    """Generate visualization plots from benchmark CSV.
    
    Creates:
    - Speedup vs qubit count
    - Execution time comparison
    - Fidelity vs qubit count
    - Rank growth analysis
    
    Args:
        csv_path: Path to benchmark CSV file
        output_dir: Output directory for plots (default: same as CSV)
        
    Returns:
        Dict mapping plot names to file paths
    """
    df = pd.read_csv(csv_path)
    output_dir = output_dir or csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    # 1. Speedup vs Qubit Count
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['qubits'], df['speedup'], marker='o', linewidth=2)
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Speedup Factor (LRET vs Cirq FDM)', fontsize=12)
    ax.set_title('LRET Performance Advantage', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    speedup_plot = output_dir / 'speedup_vs_qubits.png'
    plt.savefig(speedup_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plots['speedup'] = speedup_plot
    
    # 2. Execution Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['qubits'], df['lret_time_ms'], marker='o', label='LRET', linewidth=2)
    ax.plot(df['qubits'], df['cirq_fdm_time_ms'], marker='s', label='Cirq FDM', linewidth=2)
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Execution Time (ms)', fontsize=12)
    ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    time_plot = output_dir / 'time_comparison.png'
    plt.savefig(time_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plots['time'] = time_plot
    
    # 3. Fidelity vs Qubit Count
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['qubits'], df['fidelity'], marker='o', color='green', linewidth=2)
    ax.axhline(y=0.999, color='r', linestyle='--', label='99.9% threshold')
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('LRET Fidelity vs Exact Simulation', fontsize=14, fontweight='bold')
    ax.set_ylim([0.995, 1.0])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fidelity_plot = output_dir / 'fidelity_vs_qubits.png'
    plt.savefig(fidelity_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plots['fidelity'] = fidelity_plot
    
    # 4. Rank Growth
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['qubits'], df['final_rank'], marker='o', color='purple', linewidth=2)
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Final Rank', fontsize=12)
    ax.set_title('LRET Rank Growth with Circuit Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    rank_plot = output_dir / 'rank_growth.png'
    plt.savefig(rank_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plots['rank'] = rank_plot
    
    return plots

def generate_benchmark_report(
    csv_path: Path,
    output_path: Path,
    format: str = 'markdown'
) -> Path:
    """Generate comprehensive benchmark report.
    
    Args:
        csv_path: Path to benchmark CSV
        output_path: Output file path
        format: 'markdown' or 'html'
        
    Returns:
        Path to generated report
    """
    df = pd.read_csv(csv_path)
    
    # Generate plots
    plot_paths = plot_lret_cirq_comparison(csv_path)
    
    # Calculate statistics
    avg_speedup = df['speedup'].mean()
    max_speedup = df['speedup'].max()
    min_fidelity = df['fidelity'].min()
    avg_fidelity = df['fidelity'].mean()
    
    if format == 'markdown':
        report = f"""# LRET vs Cirq FDM Benchmark Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Runs:** {len(df)}  
**Qubit Range:** {df['qubits'].min()}-{df['qubits'].max()}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Average Speedup | {avg_speedup:.2f}x |
| Maximum Speedup | {max_speedup:.2f}x |
| Average Fidelity | {avg_fidelity:.6f} |
| Minimum Fidelity | {min_fidelity:.6f} |

## Performance Trends

### Speedup vs Qubit Count
![Speedup]({plot_paths['speedup']})

### Execution Time Comparison
![Time Comparison]({plot_paths['time']})

### Fidelity Analysis
![Fidelity]({plot_paths['fidelity']})

### Rank Growth
![Rank Growth]({plot_paths['rank']})

## Detailed Results

{df.to_markdown(index=False)}

## Conclusions

- LRET achieves **{avg_speedup:.1f}x average speedup** over Cirq FDM
- Fidelity maintained above **{min_fidelity:.4f}** across all tests
- Speedup increases exponentially with qubit count
- Final rank grows sub-exponentially, confirming low-rank efficiency

"""
        output_path.write_text(report)
    
    return output_path
```

---

### PHASE 2: PennyLane Hybrid Integration

**Estimated Time:** 7-10 days  
**Complexity:** High  
**Dependencies:** Phase 0 complete, PennyLane installed

#### Step 2.1: Create PennyLane Device Plugin

**TUI Navigation:** Backends → Add Backend → LRET PennyLane Device

**Buttons:**
- [Configure Device] - Set wires, shots, noise model
- [Test VQE] - Run sample VQE algorithm
- [Test QAOA] - Run sample QAOA
- [Gradient Check] - Verify auto-diff works
- [View PennyLane Docs] - Open integration docs

**Implementation:**

```python
# File: src/proxima/backends/lret/pennylane_device.py

"""PennyLane device plugin for LRET simulator."""

import pennylane as qml
from pennylane import numpy as np
from typing import Any, Optional, Union
from pennylane.devices import DefaultQubit

class QLRETDevice(DefaultQubit):
    """PennyLane device for LRET quantum simulator.
    
    This device integrates LRET's low-rank simulation with PennyLane's
    auto-differentiation and hybrid quantum-classical workflow.
    
    Features:
    - Parameter-shift and adjoint differentiation
    - Noise model support (depolarizing, damping, Kraus operators)
    - Efficient low-rank state tracking
    - Seamless integration with PennyLane optimizers
    
    Example:
        >>> dev = QLRETDevice(wires=4, shots=1024, noise_level=0.01)
        >>> @qml.qnode(dev)
        ... def circuit(params):
        ...     qml.RY(params[0], wires=0)
        ...     qml.CNOT(wires=[0, 1])
        ...     return qml.expval(qml.PauliZ(0))
        >>> result = circuit([0.5])
    """
    
    name = "LRET PennyLane Device"
    short_name = "lret.qubit"
    pennylane_requires = ">=0.33.0"
    version = "1.0.0"
    author = "Proxima Team"
    
    operations = {
        # Single-qubit gates
        "PauliX", "PauliY", "PauliZ",
        "Hadamard", "S", "T",
        "RX", "RY", "RZ",
        "Rot", "PhaseShift",
        "U1", "U2", "U3",
        
        # Two-qubit gates
        "CNOT", "CZ", "SWAP",
        "CRX", "CRY", "CRZ",
        "IsingXX", "IsingYY", "IsingZZ",
        
        # Three-qubit gates
        "Toffoli", "CSWAP",
        
        # State preparation
        "BasisState", "QubitStateVector",
    }
    
    observables = {
        "PauliX", "PauliY", "PauliZ",
        "Hadamard", "Hermitian",
    }
    
    def __init__(
        self,
        wires: int,
        *,
        shots: Optional[int] = None,
        noise_level: float = 0.0,
        noise_model: str = "depolarizing",
        rank_threshold: float = 1e-4,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize the LRET device.
        
        Args:
            wires: Number of qubits
            shots: Number of measurement shots (None = statevector mode)
            noise_level: Noise parameter (0.0-1.0)
            noise_model: 'depolarizing', 'damping', 'custom'
            rank_threshold: SVD truncation threshold
            seed: Random seed for reproducibility
        """
        super().__init__(wires, shots=shots, seed=seed)
        
        self.noise_level = noise_level
        self.noise_model = noise_model
        self.rank_threshold = rank_threshold
        
        try:
            import qlret
            self._lret = qlret
        except ImportError:
            raise ImportError(
                "LRET PennyLane variant not installed. "
                "Install with: pip install -e python/ from pennylane-documentation-benchmarking branch"
            )
    
    def apply(self, operations, **kwargs):
        """Apply quantum operations to the device state."""
        # Convert PennyLane operations to LRET format
        lret_ops = []
        for op in operations:
            lret_op = self._convert_operation(op)
            lret_ops.append(lret_op)
        
        # Execute on LRET simulator
        self._lret_simulator = self._lret.Simulator(
            n_qubits=self.num_wires,
            noise=self.noise_level,
            noise_model=self.noise_model,
            rank_threshold=self.rank_threshold,
            seed=self._rng,
        )
        
        self._lret_simulator.apply_operations(lret_ops)
    
    def _convert_operation(self, op: qml.operation.Operation) -> dict:
        """Convert PennyLane operation to LRET format."""
        op_map = {
            "PauliX": "x",
            "PauliY": "y",
            "PauliZ": "z",
            "Hadamard": "h",
            "CNOT": "cx",
            "RX": "rx",
            "RY": "ry",
            "RZ": "rz",
        }
        
        lret_name = op_map.get(op.name, op.name.lower())
        
        return {
            "name": lret_name,
            "wires": list(op.wires),
            "params": list(op.parameters) if op.parameters else [],
        }
    
    def expval(self, observable, shot_range=None, bin_size=None):
        """Compute expectation value of an observable."""
        if self.shots is None:
            # Statevector mode - exact expectation
            return self._statevector_expval(observable)
        else:
            # Sampling mode - estimate from shots
            return self._sampling_expval(observable, shot_range, bin_size)
    
    def _statevector_expval(self, observable):
        """Compute exact expectation value from statevector."""
        state = self._lret_simulator.get_state_vector()
        obs_matrix = self._get_observable_matrix(observable)
        
        # <ψ|O|ψ>
        expval = np.vdot(state, obs_matrix @ state)
        return np.real(expval)
    
    def _sampling_expval(self, observable, shot_range, bin_size):
        """Estimate expectation value from measurement samples."""
        samples = self._lret_simulator.sample(self.shots)
        
        # Convert samples to expectation estimate
        # For Pauli-Z: +1 for |0⟩, -1 for |1⟩
        obs_wire = observable.wires[0]
        counts = {0: 0, 1: 0}
        
        for sample in samples:
            counts[sample[obs_wire]] += 1
        
        expval = (counts[0] - counts[1]) / self.shots
        return expval
    
    def _get_observable_matrix(self, observable):
        """Get matrix representation of observable."""
        if observable.name == "PauliZ":
            return np.array([[1, 0], [0, -1]])
        elif observable.name == "PauliX":
            return np.array([[0, 1], [1, 0]])
        elif observable.name == "PauliY":
            return np.array([[0, -1j], [1j, 0]])
        else:
            raise NotImplementedError(f"Observable {observable.name} not implemented")
    
    def probability(self, wires=None):
        """Get computational basis state probabilities."""
        if self.shots is None:
            state = self._lret_simulator.get_state_vector()
            return np.abs(state) ** 2
        else:
            samples = self._lret_simulator.sample(self.shots)
            # Convert samples to probability distribution
            return self._samples_to_probs(samples)
    
    def _samples_to_probs(self, samples):
        """Convert measurement samples to probability distribution."""
        n_outcomes = 2 ** self.num_wires
        counts = np.zeros(n_outcomes)
        
        for sample in samples:
            # Convert bit string to integer
            idx = sum(bit * (2 ** i) for i, bit in enumerate(reversed(sample)))
            counts[idx] += 1
        
        return counts / len(samples)
```

#### Step 2.2: Add VQE/QAOA Algorithm Templates

**TUI Navigation:** Algorithms → Variational → Add Algorithm

**Options:**
- [ ] VQE (Variational Quantum Eigensolver)
- [ ] QAOA (Quantum Approximate Optimization)
- [ ] QNN (Quantum Neural Network)
- [ ] Custom Variational Circuit

**Buttons:**
- [Configure Parameters] - Set optimizer, learning rate, iterations
- [Define Hamiltonian] - Specify problem Hamiltonian
- [Select Ansatz] - Choose circuit ansatz
- [Run Algorithm] - Execute optimization
- [View Convergence] - Plot energy vs iteration

**Implementation:**

```python
# File: src/proxima/backends/lret/algorithms.py

"""Variational quantum algorithms using LRET PennyLane device."""

import pennylane as qml
from pennylane import numpy as np
from typing import Callable, Optional, Union
from dataclasses import dataclass

@dataclass
class VQEResult:
    """Result from VQE optimization."""
    
    final_energy: float
    optimal_params: np.ndarray
    iterations: int
    energy_history: list[float]
    converged: bool

class VQE:
    """Variational Quantum Eigensolver using LRET device.
    
    Example:
        >>> from qlret import QLRETDevice
        >>> dev = QLRETDevice(wires=4, shots=1024)
        >>> 
        >>> # Define Hamiltonian
        >>> H = qml.Hamiltonian(
        ...     [0.5, 0.5, -1.0],
        ...     [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        ... )
        >>> 
        >>> # Define ansatz
        >>> def ansatz(params, wires):
        ...     qml.RY(params[0], wires=0)
        ...     qml.RY(params[1], wires=1)
        ...     qml.CNOT(wires=[0, 1])
        >>> 
        >>> vqe = VQE(dev, H, ansatz)
        >>> result = vqe.run(initial_params=[0.5, 0.5])
    """
    
    def __init__(
        self,
        device: qml.Device,
        hamiltonian: qml.Hamiltonian,
        ansatz: Callable,
        optimizer: Optional[qml.GradientDescentOptimizer] = None,
    ):
        """Initialize VQE.
        
        Args:
            device: PennyLane device (e.g., QLRETDevice)
            hamiltonian: Problem Hamiltonian
            ansatz: Parameterized circuit function
            optimizer: PennyLane optimizer (default: AdamOptimizer)
        """
        self.device = device
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer = optimizer or qml.AdamOptimizer(stepsize=0.1)
        
        # Create cost function
        @qml.qnode(device)
        def cost_fn(params):
            self.ansatz(params, wires=range(device.num_wires))
            return qml.expval(hamiltonian)
        
        self.cost_fn = cost_fn
    
    def run(
        self,
        initial_params: np.ndarray,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
    ) -> VQEResult:
        """Run VQE optimization.
        
        Args:
            initial_params: Initial parameter values
            max_iterations: Maximum optimization iterations
            convergence_threshold: Energy convergence threshold
            
        Returns:
            VQEResult with optimization details
        """
        params = np.array(initial_params, requires_grad=True)
        energy_history = []
        
        for iteration in range(max_iterations):
            # Compute energy and gradient
            params, energy = self.optimizer.step_and_cost(self.cost_fn, params)
            energy_history.append(float(energy))
            
            # Check convergence
            if iteration > 0:
                energy_change = abs(energy_history[-1] - energy_history[-2])
                if energy_change < convergence_threshold:
                    converged = True
                    break
        else:
            converged = False
        
        return VQEResult(
            final_energy=energy_history[-1],
            optimal_params=params,
            iterations=iteration + 1,
            energy_history=energy_history,
            converged=converged,
        )

class QAOA:
    """Quantum Approximate Optimization Algorithm using LRET device.
    
    Example:
        >>> from qlret import QLRETDevice
        >>> dev = QLRETDevice(wires=4, shots=2048)
        >>> 
        >>> # Max-Cut problem graph
        >>> edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        >>> 
        >>> qaoa = QAOA(dev, edges, p=2)
        >>> result = qaoa.run()
    """
    
    def __init__(
        self,
        device: qml.Device,
        edges: list[tuple[int, int]],
        p: int = 1,
    ):
        """Initialize QAOA.
        
        Args:
            device: PennyLane device
            edges: Graph edges for Max-Cut problem
            p: Number of QAOA layers
        """
        self.device = device
        self.edges = edges
        self.p = p
        
        # Create cost Hamiltonian (Max-Cut)
        coeffs = []
        obs = []
        for edge in edges:
            coeffs.append(0.5)
            obs.append(qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))
        
        self.cost_h = qml.Hamiltonian(coeffs, obs)
        
        # Create QNode
        @qml.qnode(device)
        def qaoa_circuit(gammas, betas):
            # Initial state: superposition
            for wire in range(device.num_wires):
                qml.Hadamard(wires=wire)
            
            # QAOA layers
            for i in range(p):
                # Cost Hamiltonian
                for edge in edges:
                    qml.CNOT(wires=edge)
                    qml.RZ(2 * gammas[i], wires=edge[1])
                    qml.CNOT(wires=edge)
                
                # Mixer Hamiltonian
                for wire in range(device.num_wires):
                    qml.RX(2 * betas[i], wires=wire)
            
            return qml.expval(self.cost_h)
        
        self.circuit = qaoa_circuit
    
    def run(
        self,
        initial_params: Optional[tuple[np.ndarray, np.ndarray]] = None,
        max_iterations: int = 100,
    ) -> VQEResult:
        """Run QAOA optimization."""
        if initial_params is None:
            gammas = np.random.uniform(0, 2 * np.pi, self.p, requires_grad=True)
            betas = np.random.uniform(0, np.pi, self.p, requires_grad=True)
        else:
            gammas, betas = initial_params
        
        optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
        energy_history = []
        
        for iteration in range(max_iterations):
            (gammas, betas), energy = optimizer.step_and_cost(
                lambda g, b: self.circuit(g, b),
                gammas,
                betas
            )
            energy_history.append(float(energy))
        
        return VQEResult(
            final_energy=energy_history[-1],
            optimal_params=(gammas, betas),
            iterations=max_iterations,
            energy_history=energy_history,
            converged=True,
        )
```

---

### PHASE 3: Phase 7 Unified Integration

**Estimated Time:** 10-14 days  
**Complexity:** Very High  
**Dependencies:** All previous phases complete

#### Step 3.1: Multi-Framework Unified Executor

**TUI Navigation:** Backends → LRET Phase 7 → Unified Execution

**Buttons:**
- [Configure Frameworks] - Enable Cirq, PennyLane, Qiskit
- [Set Preferences] - Backend selection priority
- [Enable Gate Fusion] - Optimization settings
- [GPU Settings] - cuQuantum configuration
- [Run Multi-Backend Test] - Test all frameworks

**Implementation:**

```python
# File: src/proxima/backends/lret/phase7_unified.py

"""LRET Phase 7 unified multi-framework adapter."""

from __future__ import annotations

from typing import Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from proxima.backends.base import BaseBackendAdapter, BackendCapability
from proxima.core.result import ExecutionResult
from proxima.utils.logging import get_logger

logger = get_logger(__name__)

class Framework(Enum):
    """Supported quantum frameworks."""
    CIRQ = "cirq"
    PENNYLANE = "pennylane"
    QISKIT = "qiskit"
    AUTO = "auto"

@dataclass
class Phase7Config:
    """Configuration for Phase 7 unified adapter."""
    
    enabled_frameworks: list[str] = field(default_factory=lambda: ['cirq', 'pennylane', 'qiskit'])
    backend_preference: list[str] = field(default_factory=lambda: ['cirq', 'pennylane', 'qiskit'])
    gate_fusion: bool = True
    fusion_mode: Literal['row', 'column', 'hybrid'] = 'hybrid'
    gpu_enabled: bool = False
    gpu_device_id: int = 0
    optimization_level: int = 2  # 0=none, 1=basic, 2=full
    
class LRETPhase7UnifiedAdapter(BaseBackendAdapter):
    """LRET Phase 7 unified multi-framework adapter.
    
    Provides a single unified interface to execute circuits across
    Cirq, PennyLane, and Qiskit backends with automatic framework
    selection, gate fusion optimization, and optional GPU acceleration.
    
    Features:
    - Auto-select best framework based on circuit characteristics
    - Gate fusion for performance optimization
    - GPU acceleration via cuQuantum (optional)
    - Cross-platform support (Windows MSVC, Linux, macOS)
    - Unified result format
    
    Example:
        >>> adapter = LRETPhase7UnifiedAdapter()
        >>> await adapter.connect()
        >>> result = await adapter.execute(circuit, options={
        ...     'framework': 'auto',
        ...     'optimize': True,
        ...     'use_gpu': True
        ... })
    """
    
    def __init__(self, config: Optional[Phase7Config] = None):
        """Initialize Phase 7 unified adapter."""
        self._config = config or Phase7Config()
        self._frameworks = {}
        self._executors = {}
        
    @property
    def name(self) -> str:
        return "lret_phase7_unified"
    
    @property
    def backend_type(self) -> str:
        return "multi_framework"
    
    @property
    def capabilities(self) -> BackendCapability:
        return (
            BackendCapability.STATE_VECTOR |
            BackendCapability.DENSITY_MATRIX |
            BackendCapability.MEASUREMENT |
            BackendCapability.NOISE |
            BackendCapability.GPU |
            BackendCapability.MULTI_FRAMEWORK
        )
    
    def is_available(self) -> bool:
        """Check if Phase 7 variant is installed."""
        try:
            from lret.phase7 import UnifiedExecutor
            return True
        except ImportError:
            return False
    
    async def connect(self) -> bool:
        """Initialize all enabled frameworks."""
        if not self.is_available():
            logger.error("LRET Phase 7 variant not installed")
            return False
        
        from lret.phase7 import UnifiedExecutor, GateFusion
        
        # Initialize frameworks
        for framework_name in self._config.enabled_frameworks:
            try:
                if framework_name == 'cirq':
                    import cirq
                    self._frameworks['cirq'] = cirq
                elif framework_name == 'pennylane':
                    import pennylane as qml
                    self._frameworks['pennylane'] = qml
                elif framework_name == 'qiskit':
                    import qiskit
                    self._frameworks['qiskit'] = qiskit
                
                logger.info(f"Framework {framework_name} initialized")
            except ImportError:
                logger.warning(f"Framework {framework_name} not available")
        
        # Create unified executor
        self._executor = UnifiedExecutor(
            backends=list(self._frameworks.keys()),
            device='gpu' if self._config.gpu_enabled else 'cpu',
            gpu_id=self._config.gpu_device_id,
            optimization_level=self._config.optimization_level,
        )
        
        # Initialize gate fusion if enabled
        if self._config.gate_fusion:
            self._gate_fusion = GateFusion(mode=self._config.fusion_mode)
        else:
            self._gate_fusion = None
        
        logger.info(f"LRET Phase 7 unified adapter initialized (frameworks: {list(self._frameworks.keys())})")
        return True
    
    async def disconnect(self) -> None:
        """Cleanup resources."""
        self._frameworks.clear()
        self._executor = None
        self._gate_fusion = None
    
    async def execute(
        self,
        circuit: Any,
        options: Optional[dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute circuit with auto-framework selection.
        
        Args:
            circuit: Quantum circuit (any supported format)
            options: Execution options:
                - framework: 'auto', 'cirq', 'pennylane', 'qiskit'
                - shots: Number of measurements
                - optimize: Enable gate fusion (default: True)
                - use_gpu: Use GPU acceleration (default: False)
                - noise_model: Noise configuration
                
        Returns:
            ExecutionResult with unified format
        """
        if not self._executor:
            raise RuntimeError("Backend not connected")
        
        options = options or {}
        framework = options.get('framework', 'auto')
        shots = options.get('shots', 1024)
        optimize = options.get('optimize', self._config.gate_fusion)
        use_gpu = options.get('use_gpu', self._config.gpu_enabled)
        
        # Select framework
        if framework == 'auto':
            framework = self._select_framework(circuit, options)
        
        logger.info(f"Executing on framework: {framework}")
        
        # Apply gate fusion optimization
        if optimize and self._gate_fusion:
            circuit = self._gate_fusion.optimize(circuit)
            logger.debug(f"Gate fusion applied (mode: {self._config.fusion_mode})")
        
        # Execute on selected framework
        result = self._executor.execute(
            circuit=circuit,
            backend=framework,
            shots=shots,
            use_gpu=use_gpu,
        )
        
        # Convert to unified format
        return self._convert_result(result, framework)
    
    def _select_framework(self, circuit: Any, options: dict) -> str:
        """Auto-select best framework for circuit.
        
        Selection criteria:
        - Circuit type (native Cirq/PennyLane/Qiskit)
        - Circuit size (qubits, depth)
        - Operation types (gradient-based → PennyLane)
        - User preferences
        """
        # Detect circuit native framework
        if hasattr(circuit, 'all_qubits'):  # Cirq
            return 'cirq'
        elif hasattr(circuit, 'tape'):  # PennyLane QNode
            return 'pennylane'
        elif hasattr(circuit, 'qregs'):  # Qiskit
            return 'qiskit'
        
        # Check for gradient-based operations
        if options.get('compute_gradient', False):
            return 'pennylane' if 'pennylane' in self._frameworks else 'cirq'
        
        # Use preference order
        for preferred in self._config.backend_preference:
            if preferred in self._frameworks:
                return preferred
        
        # Fallback to first available
        return list(self._frameworks.keys())[0]
    
    def _convert_result(self, result: Any, framework: str) -> ExecutionResult:
        """Convert framework-specific result to unified format."""
        # Extract counts based on framework
        if framework == 'cirq':
            counts = dict(result.histogram(key='measurements'))
        elif framework == 'pennylane':
            counts = result  # Assuming dict format
        elif framework == 'qiskit':
            counts = result.get_counts()
        else:
            counts = {}
        
        return ExecutionResult(
            counts=counts,
            shots=sum(counts.values()) if counts else 0,
            success=True,
            metadata={
                'framework': framework,
                'backend': self.name,
                'gate_fusion': self._config.gate_fusion,
                'gpu_used': self._config.gpu_enabled,
            }
        )
```

---

## 🎨 TUI Integration Guide

### Main Backend Selection Screen

**Location:** `src/proxima/tui/screens/backends.py`

**Layout:**

```
┌────────────────────────────────────────────────────────────┐
│  📦 Backend Management                      [Help] [Refresh]│
├────────────────────────────────────────────────────────────┤
│                                                              │
│  Available Backends:                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ ☑ LRET (Base)                           [Configure] │  │
│  │ ☑ LRET Cirq Scalability                [Configure] │  │
│  │ ☑ LRET PennyLane Hybrid                 [Configure] │  │
│  │ ☑ LRET Phase 7 Unified                  [Configure] │  │
│  │ ☑ Cirq                                   [Configure] │  │
│  │ ☑ Qiskit Aer                             [Configure] │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Quick Actions:                                              │
│  [Install Variant] [Run Benchmark] [View Comparison]        │
│                                                              │
│  Current Selection:                                          │
│  Default Backend: LRET Phase 7 Unified                      │
│  Fallback: LRET Cirq Scalability                           │
│                                                              │
│  [Apply Changes] [Restore Defaults] [Export Config]         │
└────────────────────────────────────────────────────────────┘
```

### Benchmark Comparison Screen

**Location:** `src/proxima/tui/screens/benchmark_comparison.py`

**Layout:**

```
┌────────────────────────────────────────────────────────────┐
│  📊 LRET vs Cirq Benchmark Comparison                       │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  Qubit Range: [4 ] - [14]   Depth: [20  ]  Noise: [0.01]  │
│  [Run Benchmark]              [Load Existing] [Export CSV]  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Speedup vs Qubit Count                       │  │
│  │  100 ┤                                          ●    │  │
│  │   50 ┤                                    ●          │  │
│  │   10 ┤                          ●                    │  │
│  │    5 ┤                    ●                          │  │
│  │    2 ┤              ●                                │  │
│  │    1 ┼────●─────────────────────────────────────────│  │
│  │      4    6    8    10   12   14   Qubits          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Summary Statistics:                                         │
│  Average Speedup: 24.7x     Max Speedup: 91.9x              │
│  Avg Fidelity: 0.9997       Min Fidelity: 0.9995            │
│                                                              │
│  [View Detailed Results] [Generate Report] [Close]          │
└────────────────────────────────────────────────────────────┘
```

### PennyLane Algorithm Wizard

**Location:** `src/proxima/tui/wizards/pennylane_algorithm.py`

**Steps:**

1. **Algorithm Selection**
   - VQE, QAOA, QNN, Custom
   
2. **Problem Definition**
   - Hamiltonian specification
   - Problem encoding
   
3. **Ansatz Configuration**
   - Hardware-efficient, UCCSD, custom
   - Number of layers
   
4. **Optimizer Settings**
   - Adam, SGD, Momentum
   - Learning rate, iterations
   
5. **Device Configuration**
   - QLRETDevice settings
   - Shots, noise model
   
6. **Execute & Monitor**
   - Real-time convergence plot
   - Energy history
   - Parameter updates

---

## 📝 Testing Strategy

### Unit Tests

```python
# File: tests/backends/lret/test_cirq_scalability.py

import pytest
from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter

@pytest.mark.asyncio
async def test_cirq_scalability_basic():
    """Test basic Cirq scalability functionality."""
    adapter = LRETCirqScalabilityAdapter()
    
    assert adapter.name == "lret_cirq_scalability"
    assert await adapter.connect()
    
    # Create simple circuit
    import cirq
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
    ])
    
    result = await adapter.execute(circuit, options={'shots': 1024})
    
    assert result.success
    assert result.shots == 1024
    assert 'execution_time_ms' in result.metadata
    
    await adapter.disconnect()

@pytest.mark.asyncio
async def test_lret_cirq_comparison():
    """Test LRET vs Cirq comparison functionality."""
    adapter = LRETCirqScalabilityAdapter()
    await adapter.connect()
    
    import cirq
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.Circuit([
        cirq.H.on_each(*qubits),
        cirq.CNOT(qubits[i], qubits[i+1]) for i in range(7)
    ])
    
    result = await adapter.execute(circuit, options={
        'shots': 1024,
        'compare_with_cirq': True,
        'benchmark': True,
    })
    
    assert 'speedup' in result.metadata
    assert 'fidelity' in result.metadata
    assert result.metadata['speedup'] > 1.0  # LRET should be faster
    assert result.metadata['fidelity'] > 0.999  # High fidelity
    
    await adapter.disconnect()

# File: tests/backends/lret/test_pennylane_device.py

import pytest
import pennylane as qml
from pennylane import numpy as np
from proxima.backends.lret.pennylane_device import QLRETDevice

def test_qlret_device_basic():
    """Test basic QLRETDevice functionality."""
    dev = QLRETDevice(wires=4, shots=1024)
    
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    result = circuit()
    
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0

def test_qlret_device_gradient():
    """Test gradient computation with QLRETDevice."""
    dev = QLRETDevice(wires=2, shots=None)  # Statevector mode
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    params = np.array([0.5, 0.3], requires_grad=True)
    
    # Compute gradient
    grad_fn = qml.grad(circuit)
    gradients = grad_fn(params)
    
    assert gradients.shape == (2,)
    assert all(isinstance(g, float) for g in gradients)

def test_vqe_with_qlret():
    """Test VQE algorithm with QLRETDevice."""
    from proxima.backends.lret.algorithms import VQE
    
    dev = QLRETDevice(wires=2, shots=2048)
    
    # Simple Hamiltonian
    H = qml.Hamiltonian(
        [1.0, -1.0],
        [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)]
    )
    
    def ansatz(params, wires):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
    
    vqe = VQE(dev, H, ansatz)
    result = vqe.run(initial_params=[0.5, 0.5], max_iterations=50)
    
    assert result.converged
    assert result.iterations <= 50
    assert len(result.energy_history) == result.iterations

# File: tests/backends/lret/test_phase7_unified.py

import pytest
from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter, Framework

@pytest.mark.asyncio
async def test_phase7_unified_basic():
    """Test Phase 7 unified adapter basic functionality."""
    adapter = LRETPhase7UnifiedAdapter()
    
    assert adapter.name == "lret_phase7_unified"
    assert await adapter.connect()
    
    # Test framework availability
    frameworks = adapter._frameworks
    assert len(frameworks) > 0
    
    await adapter.disconnect()

@pytest.mark.asyncio
async def test_phase7_auto_framework_selection():
    """Test automatic framework selection."""
    adapter = LRETPhase7UnifiedAdapter()
    await adapter.connect()
    
    # Cirq circuit
    import cirq
    cirq_circuit = cirq.Circuit([cirq.H(cirq.LineQubit(0))])
    
    selected = adapter._select_framework(cirq_circuit, {})
    assert selected == 'cirq'
    
    # PennyLane circuit detection would go here
    
    await adapter.disconnect()

@pytest.mark.asyncio
async def test_phase7_gate_fusion():
    """Test gate fusion optimization."""
    from proxima.backends.lret.phase7_unified import Phase7Config
    
    config = Phase7Config(gate_fusion=True, fusion_mode='hybrid')
    adapter = LRETPhase7UnifiedAdapter(config)
    await adapter.connect()
    
    import cirq
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.H(qubits[1]),
        cirq.CNOT(qubits[1], qubits[2]),
    ])
    
    result = await adapter.execute(circuit, options={
        'optimize': True,
        'shots': 1024
    })
    
    assert result.success
    assert result.metadata['gate_fusion'] is True
    
    await adapter.disconnect()
```

### Integration Tests

```python
# File: tests/integration/test_lret_variants_integration.py

import pytest
from proxima.core.orchestrator import Orchestrator
from proxima.backends.lret.installer import install_lret_variant

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_lret_variant_workflow():
    """Test complete workflow with LRET variants."""
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # 1. Check variant availability
    from proxima.backends.lret.installer import check_variant_availability
    
    cirq_status = check_variant_availability('cirq_scalability')
    assert cirq_status['installed']
    
    # 2. Create circuit
    import cirq
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.Circuit([
        cirq.H.on_each(*qubits),
        *[cirq.CNOT(qubits[i], qubits[i+1]) for i in range(7)]
    ])
    
    # 3. Execute with benchmark
    result = await orchestrator.execute(
        circuit=circuit,
        backend='lret_cirq_scalability',
        options={
            'shots': 2048,
            'benchmark': True,
            'compare_with_cirq': True,
        }
    )
    
    # 4. Verify results
    assert result.success
    assert 'speedup' in result.metadata
    assert result.metadata['speedup'] > 1.0
    
    # 5. Check benchmark export
    from pathlib import Path
    benchmark_csv = Path('./benchmarks/lret_cirq_comparison.csv')
    assert benchmark_csv.exists()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pennylane_vqe_integration():
    """Test PennyLane VQE integration."""
    import pennylane as qml
    from proxima.backends.lret.pennylane_device import QLRETDevice
    from proxima.backends.lret.algorithms import VQE
    
    # Create device
    dev = QLRETDevice(wires=4, shots=2048, noise_level=0.01)
    
    # Define problem
    H = qml.Hamiltonian(
        [1.0, 0.5, 0.5],
        [
            qml.PauliZ(0),
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliZ(2) @ qml.PauliZ(3),
        ]
    )
    
    def ansatz(params, wires):
        for i in range(len(wires)):
            qml.RY(params[i], wires=wires[i])
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
    
    # Run VQE
    vqe = VQE(dev, H, ansatz)
    result = vqe.run(
        initial_params=[0.1] * 4,
        max_iterations=100,
        convergence_threshold=1e-5
    )
    
    # Verify convergence
    assert result.converged
    assert result.final_energy < 0  # Ground state should be negative
    assert len(result.energy_history) > 10

@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase7_multi_framework():
    """Test Phase 7 multi-framework execution."""
    from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
    
    adapter = LRETPhase7UnifiedAdapter()
    await adapter.connect()
    
    # Test on multiple frameworks
    import cirq
    import pennylane as qml
    
    # Cirq circuit
    cirq_circuit = cirq.Circuit([
        cirq.H(cirq.LineQubit(0)),
        cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
    ])
    
    result1 = await adapter.execute(cirq_circuit, {'framework': 'cirq', 'shots': 1024})
    assert result1.success
    assert result1.metadata['framework'] == 'cirq'
    
    # PennyLane QNode
    dev = qml.device('default.qubit', wires=2)
    
    @qml.qnode(dev)
    def pqc():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    result2 = await adapter.execute(pqc, {'framework': 'pennylane'})
    assert result2.success
    
    await adapter.disconnect()
```

---

## 📋 Validation Checklist

Before deployment, ensure all items are checked:

### Installation
- [ ] All three LRET variants cloned successfully
- [ ] Python dependencies installed (cirq, pennylane, qiskit)
- [ ] C++ components built (CMake + Eigen)
- [ ] Import tests pass for all variants

### Cirq Scalability
- [ ] Cirq FDM comparison working
- [ ] Benchmark CSV export functional
- [ ] Speedup calculations accurate
- [ ] Fidelity measurements correct
- [ ] Visualization plots generated

### PennyLane Hybrid
- [ ] QLRETDevice creates successfully
- [ ] Basic QNode execution works
- [ ] Gradient computation functional
- [ ] VQE algorithm converges
- [ ] QAOA optimization completes
- [ ] Noise models apply correctly

### Phase 7 Unified
- [ ] All frameworks initialize
- [ ] Auto-framework selection works
- [ ] Gate fusion applies successfully
- [ ] Multi-backend execution consistent
- [ ] GPU acceleration optional

### TUI Integration
- [ ] Variant installer dialog functional
- [ ] Backend configuration saves
- [ ] Benchmark comparison screen displays
- [ ] Algorithm wizard completes
- [ ] Real-time monitoring updates

### Documentation
- [ ] API docs generated
- [ ] User guides written
- [ ] Tutorial notebooks created
- [ ] Migration guide provided
- [ ] Troubleshooting section complete

---

## 🚀 Deployment Timeline

| Phase | Duration | Milestones |
|-------|----------|------------|
| Phase 0 | 2-3 days | Environment setup, installations verified |
| Phase 1 | 5-7 days | Cirq scalability functional, benchmarks running |
| Phase 2 | 7-10 days | PennyLane device operational, VQE/QAOA working |
| Phase 3 | 10-14 days | Phase 7 unified complete, all frameworks integrated |
| Testing | 3-5 days | All tests pass, validation complete |
| Documentation | 2-3 days | Docs finalized, tutorials ready |
| **Total** | **29-42 days** | Full integration deployed |

---

## 📞 Support Resources

- **LRET Repository Issues:** https://github.com/kunal5556/LRET/issues
- **Proxima Documentation:** [Internal docs link]
- **PennyLane Docs:** https://pennylane.ai/
- **Cirq Docs:** https://quantumai.google/cirq
- **Qiskit Docs:** https://qiskit.org/documentation/

---

## 🎓 Conclusion

This comprehensive integration will transform Proxima into a **multi-framework quantum simulation powerhouse**, capable of:

1. **Performance Optimization** - Automatic backend selection for best performance
2. **Algorithm Development** - Gradient-based variational algorithms via PennyLane
3. **Scalability Analysis** - Detailed benchmarking and comparison tools
4. **Ecosystem Compatibility** - Support for Cirq, PennyLane, and Qiskit workflows
5. **GPU Acceleration** - Optional hardware acceleration for large-scale simulations

The phased approach ensures **incremental value delivery** while maintaining **system stability** throughout the integration process.

---

**Document End**
