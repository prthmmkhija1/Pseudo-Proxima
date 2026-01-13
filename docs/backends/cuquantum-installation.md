# cuQuantum Backend Installation Guide

> **Version:** 1.0  
> **Last Updated:** January 13, 2026  
> **Backend:** cuQuantum (NVIDIA GPU-Accelerated Quantum Simulation)

---

## Overview

cuQuantum is NVIDIA's SDK for GPU-accelerated quantum circuit simulation. In Proxima, cuQuantum provides high-performance state vector simulation through integration with Qiskit Aer's GPU backend. This enables simulation of larger quantum circuits with significant speedups over CPU-only approaches.

### Key Features

- **GPU Acceleration**: Leverages NVIDIA CUDA for massive parallelism
- **Large Circuit Support**: Simulate up to 35+ qubits with sufficient GPU memory
- **cuStateVec Integration**: Direct access to NVIDIA's optimized state vector operations
- **Multi-GPU Support**: Distribute simulations across multiple GPUs
- **Automatic Fallback**: Falls back to CPU execution when GPU is unavailable

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with Compute Capability 7.0+ (Volta) | NVIDIA Ampere/Hopper (A100, H100) |
| **GPU Memory** | 8 GB | 24 GB+ (for 25+ qubits) |
| **System RAM** | 16 GB | 64 GB+ |
| **CPU** | Any x86_64 | Multi-core with high memory bandwidth |

### Supported NVIDIA GPUs

| GPU Series | Compute Capability | Status |
|------------|-------------------|--------|
| V100 (Volta) | 7.0 | ✅ Supported |
| T4 (Turing) | 7.5 | ✅ Supported |
| A100 (Ampere) | 8.0 | ✅ Recommended |
| RTX 30xx | 8.6 | ✅ Supported |
| RTX 40xx | 8.9 | ✅ Supported |
| H100 (Hopper) | 9.0 | ✅ Recommended |
| Older GPUs | < 7.0 | ❌ Not Supported |

### Software Requirements

| Component | Version | Required |
|-----------|---------|----------|
| **CUDA Toolkit** | 11.8+ or 12.x | ✅ Yes |
| **NVIDIA Driver** | 520.0+ | ✅ Yes |
| **cuQuantum SDK** | 24.03.0+ | ✅ Yes |
| **Python** | 3.9 - 3.12 | ✅ Yes |
| **Qiskit** | 1.0+ | ✅ Yes |
| **Qiskit Aer GPU** | 0.13+ | ✅ Yes |

---

## Installation Methods

### Method 1: pip Installation (Recommended)

The simplest approach using pre-built packages:

```bash
# 1. Install CUDA-enabled Qiskit Aer
pip install qiskit-aer-gpu

# 2. Install cuQuantum Python bindings (optional, for direct API access)
pip install cuquantum-python-cu12  # For CUDA 12.x
# or
pip install cuquantum-python-cu11  # For CUDA 11.x

# 3. Verify installation
python -c "from qiskit_aer import AerSimulator; print(AerSimulator.available_devices())"
```

### Method 2: Conda Installation

Using conda for better dependency management:

```bash
# Create conda environment
conda create -n proxima-gpu python=3.11
conda activate proxima-gpu

# Install CUDA toolkit
conda install -c conda-forge cudatoolkit=12.0

# Install cuQuantum
conda install -c conda-forge cuquantum

# Install Qiskit Aer with GPU support
pip install qiskit-aer-gpu

# Verify
python -c "import cuquantum; print(cuquantum.__version__)"
```

### Method 3: Docker Container

Using NVIDIA's container for guaranteed compatibility:

```bash
# Pull cuQuantum container
docker pull nvcr.io/nvidia/cuquantum-appliance:24.03

# Run with GPU access
docker run --gpus all -it nvcr.io/nvidia/cuquantum-appliance:24.03

# Inside container, install Proxima
pip install proxima-agent
```

### Method 4: Source Installation

For advanced users needing custom builds:

```bash
# 1. Clone cuQuantum SDK
git clone https://github.com/NVIDIA/cuQuantum.git
cd cuQuantum

# 2. Install cuStateVec Python bindings
cd python
pip install .

# 3. Install Qiskit Aer from source with GPU support
git clone https://github.com/Qiskit/qiskit-aer.git
cd qiskit-aer
pip install . --config-settings="--build-option=--with-cuda"
```

---

## CUDA Toolkit Installation

### Linux (Ubuntu/Debian)

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-12-3

# Set environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

### Windows

1. Download CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and select "Custom Installation"
3. Ensure "CUDA Toolkit" and "Development" components are selected
4. Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin`
5. Restart terminal and verify:
   ```powershell
   nvcc --version
   nvidia-smi
   ```

### macOS

> ⚠️ **Note**: CUDA is not supported on macOS with Apple Silicon. cuQuantum requires NVIDIA GPUs.

For Intel Macs with NVIDIA GPUs (legacy systems):
```bash
# CUDA support on macOS was deprecated
# Consider using cloud GPU instances or Linux systems
```

---

## Verification Steps

### Step 1: Verify GPU Detection

```python
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

### Step 2: Verify CUDA Installation

```python
import subprocess
result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
print(result.stdout)
```

### Step 3: Verify cuQuantum Installation

```python
try:
    import cuquantum
    print(f"cuQuantum version: {cuquantum.__version__}")
    print("cuQuantum installed successfully!")
except ImportError as e:
    print(f"cuQuantum not installed: {e}")
```

### Step 4: Verify Qiskit Aer GPU

```python
from qiskit_aer import AerSimulator

# Check available devices
devices = AerSimulator.available_devices()
print(f"Available devices: {devices}")

if 'GPU' in devices:
    print("GPU backend available!")
    
    # Create GPU simulator
    gpu_sim = AerSimulator(method='statevector', device='GPU')
    print(f"GPU simulator created: {gpu_sim}")
else:
    print("WARNING: GPU backend not available")
```

### Step 5: Verify Proxima cuQuantum Backend

```python
from proxima.backends import get_backend

try:
    cuquantum_backend = get_backend('cuquantum')
    print(f"Backend: {cuquantum_backend.get_name()}")
    print(f"Version: {cuquantum_backend.get_version()}")
    print(f"Available: {cuquantum_backend.is_available()}")
    
    caps = cuquantum_backend.get_capabilities()
    print(f"Max qubits: {caps.max_qubits}")
    print(f"GPU Memory: {caps.gpu_memory_mb} MB")
except Exception as e:
    print(f"cuQuantum backend not available: {e}")
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Driver Version Mismatch

**Error:**
```
CUDA driver version is insufficient for CUDA runtime version
```

**Solution:**
```bash
# Check current driver version
nvidia-smi

# Update NVIDIA driver
# Linux:
sudo apt-get install nvidia-driver-535

# Windows:
# Download latest driver from nvidia.com/drivers
```

#### 2. cuQuantum Library Not Found

**Error:**
```
ImportError: libcustatevec.so.1: cannot open shared object file
```

**Solution:**
```bash
# Set library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Or install cuQuantum
conda install -c conda-forge cuquantum
```

#### 3. GPU Out of Memory

**Error:**
```
CUDA out of memory. Tried to allocate X GB
```

**Solution:**
```python
# Reduce qubit count
# Or use memory-efficient options
result = backend.execute(
    circuit,
    options={
        'precision': 'single',  # Use float32 instead of float64
        'blocking': True,       # Wait for completion
    }
)
```

#### 4. Compute Capability Too Low

**Error:**
```
GPU compute capability 6.1 is below minimum required 7.0
```

**Solution:**
- Upgrade to a newer GPU (Volta or later)
- Use CPU-only backend instead:
  ```python
  from proxima.backends import get_backend
  backend = get_backend('qiskit')  # CPU fallback
  ```

#### 5. Multiple GPU Selection

**Issue:** Wrong GPU being used

**Solution:**
```python
# Specify GPU device ID
result = backend.execute(
    circuit,
    options={
        'device_id': 1,  # Use second GPU
    }
)

# Or set environment variable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
```

---

## GPU Memory Requirements

| Qubits | State Vector Size | Required GPU Memory |
|--------|------------------|---------------------|
| 20 | 16 MB | ~1 GB (with workspace) |
| 25 | 512 MB | ~2 GB |
| 28 | 4 GB | ~6 GB |
| 30 | 16 GB | ~20 GB |
| 32 | 64 GB | ~80 GB (multi-GPU) |
| 35 | 512 GB | Multi-GPU cluster |

> **Formula:** Memory = 2^n × 16 bytes (complex128) + ~1GB workspace

---

## Multi-GPU Configuration

### Enabling Multi-GPU

```python
from proxima.backends import get_backend

backend = get_backend('cuquantum')

# Configure multi-GPU
config = backend.get_multi_gpu_config()
print(f"Available GPUs: {config['gpu_ids']}")
print(f"NVLink available: {config['nvlink_available']}")

# Execute with multi-GPU
result = backend.execute(
    large_circuit,
    options={
        'multi_gpu': True,
        'gpu_ids': [0, 1, 2, 3],  # Use 4 GPUs
    }
)
```

### NVLink Optimization

For systems with NVLink interconnect:

```python
# Check NVLink topology
config = backend.get_multi_gpu_config()
if config['nvlink_available']:
    print("NVLink detected - using optimized inter-GPU communication")
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | All GPUs |
| `CUSTATEVEC_LOG_LEVEL` | cuStateVec logging level | 0 (off) |
| `PROXIMA_GPU_MEMORY_LIMIT` | Max GPU memory (MB) | Auto |
| `PROXIMA_CUQUANTUM_PRECISION` | Precision mode | double |

Example:
```bash
export CUDA_VISIBLE_DEVICES=0,1
export PROXIMA_GPU_MEMORY_LIMIT=20000
export PROXIMA_CUQUANTUM_PRECISION=single
```

---

## Performance Tips

1. **Use Single Precision** for circuits where numerical precision is less critical:
   ```python
   options={'precision': 'single'}  # 2x memory savings
   ```

2. **Enable Kernel Caching** for repeated circuits:
   ```python
   options={'cache_kernels': True}
   ```

3. **Warm Up GPU** before benchmarking:
   ```python
   backend.warm_up_gpu(num_qubits=10)
   ```

4. **Use Memory Pools** for batch execution:
   ```python
   backend.create_memory_pool(size_mb=8000)
   ```

---

## See Also

- [cuQuantum Usage Guide](cuquantum-usage.md) - Detailed usage examples
- [Backend Selection Guide](backend-selection.md) - Choosing the right backend
- [NVIDIA cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/)
- [Qiskit Aer GPU Documentation](https://qiskit.github.io/qiskit-aer/)
