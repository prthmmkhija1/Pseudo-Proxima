# qsim Backend Installation Guide

> **Version:** 1.0  
> **Last Updated:** January 13, 2026  
> **Backend:** qsim (Google's High-Performance Quantum Simulator)

---

## Overview

qsim is Google's high-performance quantum circuit simulator, optimized for modern CPUs with AVX2/AVX512 instruction sets. It provides state vector simulation with excellent performance through automatic gate fusion and OpenMP parallelization, making it ideal for large-scale quantum circuit simulations on CPU.

### Key Features

- **AVX2/AVX512 Vectorization**: Leverages modern CPU SIMD instructions
- **OpenMP Parallelization**: Multi-threaded execution across CPU cores
- **Automatic Gate Fusion**: Combines gates for optimal performance
- **Cirq Integration**: Seamless integration with Google's Cirq framework
- **High Qubit Count**: Simulate up to 35+ qubits with sufficient RAM

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | x86_64 with AVX2 | Intel Xeon / AMD EPYC with AVX512 |
| **RAM** | 8 GB | 64 GB+ (for 25+ qubits) |
| **CPU Cores** | 4 | 16+ |
| **Storage** | 1 GB free | 10 GB+ (for large simulations) |

### CPU Feature Requirements

qsim requires **AVX2** instruction support at minimum. Most modern CPUs (2015+) support AVX2.

**Supported CPUs:**

| CPU Family | AVX2 | AVX512 | Status |
|------------|------|--------|--------|
| Intel Haswell (4th gen+) | ✅ | ❌ | Supported |
| Intel Skylake-X | ✅ | ✅ | Recommended |
| Intel Ice Lake / Sapphire Rapids | ✅ | ✅ | Recommended |
| AMD Zen / Zen 2 / Zen 3 / Zen 4 | ✅ | ❌/✅ | Supported |
| Apple M1/M2 (via Rosetta) | ❌ | ❌ | Not Recommended |
| Pre-2015 CPUs | ❌ | ❌ | Not Supported |

### Software Requirements

| Component | Version | Required |
|-----------|---------|----------|
| **Python** | 3.9 - 3.12 | ✅ Yes |
| **Cirq** | 1.0+ | ✅ Yes |
| **qsimcirq** | 0.20+ | ✅ Yes |
| **NumPy** | 1.20+ | ✅ Yes |
| **C++ Compiler** | GCC 7+ / Clang 8+ | For source builds |

---

## Installation Methods

### Method 1: pip Installation (Recommended)

The simplest approach using pre-built wheels:

```bash
# Install qsimcirq (includes qsim and Cirq)
pip install qsimcirq

# Verify installation
python -c "import qsimcirq; print(qsimcirq.__version__)"
```

### Method 2: Conda Installation

Using conda for environment isolation:

```bash
# Create conda environment
conda create -n proxima-qsim python=3.11
conda activate proxima-qsim

# Install qsimcirq
pip install qsimcirq

# Install Proxima
pip install proxima-agent

# Verify
python -c "import qsimcirq; import cirq; print('OK')"
```

### Method 3: Source Installation (Advanced)

For custom builds with specific optimizations:

```bash
# 1. Clone qsim repository
git clone https://github.com/quantumlib/qsim.git
cd qsim

# 2. Install build dependencies
pip install pybind11 cmake

# 3. Build with optimizations
mkdir build && cd build

# For AVX2 systems
cmake .. -DCMAKE_BUILD_TYPE=Release

# For AVX512 systems (if supported)
cmake .. -DCMAKE_BUILD_TYPE=Release -DQSIM_ENABLE_AVX512=ON

# Build
make -j$(nproc)

# 4. Install Python bindings
cd ../qsimcirq_tests
pip install -e .
```

### Method 4: Docker Installation

Using Docker for isolated environment:

```bash
# Pull qsim Docker image
docker pull gcr.io/quantum-builds/qsim:latest

# Run with mounted workspace
docker run -it -v $(pwd):/workspace gcr.io/quantum-builds/qsim:latest

# Inside container
pip install proxima-agent
```

---

## Verification Steps

### Step 1: Check CPU Features

```python
def check_cpu_features():
    """Check if CPU supports required features."""
    import subprocess
    
    try:
        # Linux/macOS
        result = subprocess.run(['cat', '/proc/cpuinfo'], 
                              capture_output=True, text=True)
        cpuinfo = result.stdout
    except:
        # Windows
        import platform
        cpuinfo = platform.processor()
    
    avx2 = 'avx2' in cpuinfo.lower()
    avx512 = 'avx512' in cpuinfo.lower() or 'avx-512' in cpuinfo.lower()
    
    print(f"AVX2 support: {'✅' if avx2 else '❌'}")
    print(f"AVX512 support: {'✅' if avx512 else '❌'}")
    
    if not avx2:
        print("⚠️ WARNING: qsim requires AVX2 support")
    
    return avx2

check_cpu_features()
```

### Step 2: Verify qsimcirq Installation

```python
try:
    import qsimcirq
    print(f"qsimcirq version: {qsimcirq.__version__}")
    print("qsimcirq installed successfully!")
except ImportError as e:
    print(f"qsimcirq not installed: {e}")
    print("Install with: pip install qsimcirq")
```

### Step 3: Verify Cirq Installation

```python
try:
    import cirq
    print(f"Cirq version: {cirq.__version__}")
    
    # Test basic Cirq functionality
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1])
    )
    print(f"Test circuit created: {len(circuit)} moments")
except ImportError as e:
    print(f"Cirq not installed: {e}")
```

### Step 4: Verify qsim Simulator

```python
import cirq
import qsimcirq

# Create a simple circuit
qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.measure(*qubits, key='result')
)

# Create qsim simulator
simulator = qsimcirq.QSimSimulator()

# Run simulation
result = simulator.run(circuit, repetitions=1000)
counts = result.histogram(key='result')

print(f"qsim simulator working!")
print(f"Counts: {dict(counts)}")
```

### Step 5: Verify Proxima qsim Backend

```python
from proxima.backends import get_backend

try:
    qsim_backend = get_backend('qsim')
    print(f"Backend: {qsim_backend.get_name()}")
    print(f"Version: {qsim_backend.get_version()}")
    print(f"Available: {qsim_backend.is_available()}")
    
    caps = qsim_backend.get_capabilities()
    print(f"Max qubits: {caps.max_qubits}")
    print(f"AVX2 enabled: {caps.avx2_enabled}")
    print(f"AVX512 enabled: {caps.avx512_enabled}")
    print(f"Max threads: {caps.max_threads}")
except Exception as e:
    print(f"qsim backend not available: {e}")
```

---

## Troubleshooting

### Common Issues

#### 1. AVX2 Not Available

**Error:**
```
Illegal instruction (core dumped)
```

**Cause:** CPU does not support AVX2 instructions.

**Solution:**
```bash
# Check CPU features
cat /proc/cpuinfo | grep avx2

# If AVX2 not available, use alternative backends
# The Cirq backend will work without AVX2
```

#### 2. qsimcirq Import Error

**Error:**
```
ImportError: libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

**Solution:**
```bash
# Update libstdc++
conda install -c conda-forge libstdcxx-ng

# Or on Ubuntu
sudo apt-get install libstdc++6
```

#### 3. OpenMP Not Found

**Error:**
```
OpenMP not found, parallel execution disabled
```

**Solution:**
```bash
# Install OpenMP
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS
brew install libomp

# Set library path
export OMP_NUM_THREADS=16
```

#### 4. Slow Execution

**Issue:** qsim runs slower than expected.

**Solution:**
```python
# Check thread configuration
import os
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")

# Set optimal thread count
import multiprocessing
optimal_threads = multiprocessing.cpu_count() - 2  # Leave some for system
os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
```

#### 5. Memory Allocation Failure

**Error:**
```
std::bad_alloc: failed to allocate memory
```

**Solution:**
```bash
# Check available memory
free -h

# Reduce circuit size or use memory-efficient options
# For 30 qubits: 2^30 * 16 bytes = 16 GB RAM needed
```

---

## OpenMP Configuration

### Setting Thread Count

```bash
# Set before running Python
export OMP_NUM_THREADS=16

# Or in Python
import os
os.environ['OMP_NUM_THREADS'] = '16'
```

### Thread Affinity

```bash
# Spread threads across cores
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Or bind threads to sockets for NUMA systems
export OMP_PROC_BIND=close
export OMP_PLACES=sockets
```

### Optimal Settings by System

```bash
# Desktop (8 cores)
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=close

# Workstation (32 cores)
export OMP_NUM_THREADS=28
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Server (64+ cores)
export OMP_NUM_THREADS=60
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

---

## RAM Requirements

| Qubits | State Vector Size | Required RAM |
|--------|------------------|--------------|
| 20 | 16 MB | 1 GB |
| 25 | 512 MB | 2 GB |
| 28 | 4 GB | 8 GB |
| 30 | 16 GB | 32 GB |
| 32 | 64 GB | 128 GB |
| 35 | 512 GB | 1 TB |

> **Formula:** RAM = 2^n × 16 bytes (complex128) + overhead

---

## Performance Verification

### Basic Performance Test

```python
import time
import cirq
import qsimcirq

def benchmark_qsim(num_qubits, num_gates=100):
    """Benchmark qsim performance."""
    qubits = cirq.LineQubit.range(num_qubits)
    
    # Create random circuit
    circuit = cirq.Circuit()
    for _ in range(num_gates):
        circuit.append(cirq.H(qubits[0]))
        for i in range(num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    
    simulator = qsimcirq.QSimSimulator()
    
    # Warmup
    simulator.simulate(circuit)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        simulator.simulate(circuit)
    elapsed = (time.perf_counter() - start) / 10 * 1000
    
    print(f"{num_qubits} qubits: {elapsed:.2f} ms per execution")
    return elapsed

# Test different qubit counts
for n in [10, 15, 20, 25]:
    benchmark_qsim(n)
```

### Thread Scaling Test

```python
import os
import time
import cirq
import qsimcirq

def test_thread_scaling(num_qubits=20):
    """Test performance scaling with thread count."""
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit(
        [cirq.H(q) for q in qubits],
        [cirq.CNOT(qubits[i], qubits[i+1]) for i in range(num_qubits-1)]
    )
    
    results = []
    for threads in [1, 2, 4, 8, 16]:
        os.environ['OMP_NUM_THREADS'] = str(threads)
        
        # Need to recreate simulator for new thread count
        simulator = qsimcirq.QSimSimulator()
        
        start = time.perf_counter()
        for _ in range(5):
            simulator.simulate(circuit)
        elapsed = (time.perf_counter() - start) / 5 * 1000
        
        speedup = results[0]['time'] / elapsed if results else 1.0
        results.append({'threads': threads, 'time': elapsed, 'speedup': speedup})
        
        print(f"{threads} threads: {elapsed:.2f} ms (speedup: {speedup:.2f}x)")
    
    return results

test_thread_scaling()
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OMP_NUM_THREADS` | Number of OpenMP threads | All cores |
| `OMP_PROC_BIND` | Thread binding policy | none |
| `OMP_PLACES` | Thread placement | threads |
| `QSIM_VERBOSITY` | Logging verbosity | 0 |
| `PROXIMA_QSIM_FUSION` | Gate fusion mode | auto |

Example configuration:
```bash
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export QSIM_VERBOSITY=1
```

---

## Platform-Specific Notes

### Linux

Best supported platform. Ensure OpenMP is installed:
```bash
sudo apt-get install libomp-dev libgomp1
```

### macOS

Works with Intel Macs. For Apple Silicon:
```bash
# Run under Rosetta 2
arch -x86_64 python your_script.py
```

Note: Native ARM support is limited.

### Windows

Install via pip (pre-built wheels available):
```powershell
pip install qsimcirq
```

For source builds, use Visual Studio 2019+ with C++ workload.

---

## Integration with Proxima

After installation, qsim is automatically detected by Proxima:

```python
from proxima.backends import get_available_backends

# List available backends
backends = get_available_backends()
for name, available in backends.items():
    status = "✅" if available else "❌"
    print(f"{status} {name}")

# qsim should show as available
```

---

## See Also

- [qsim Usage Guide](qsim-usage.md) - Detailed usage examples
- [Backend Selection Guide](backend-selection.md) - Choosing the right backend
- [Google qsim GitHub](https://github.com/quantumlib/qsim)
- [Cirq Documentation](https://quantumai.google/cirq)
