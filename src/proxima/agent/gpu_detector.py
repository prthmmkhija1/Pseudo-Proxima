"""GPU Detection Module for Proxima Agent.

Phase 4: Backend Building & Compilation System

Provides GPU detection capabilities for:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm)
- Apple Silicon (Metal)

Used to determine which GPU-accelerated backends can be built.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.gpu_detector")


class GPUVendor(Enum):
    """GPU vendor types."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class GPUCapability(Enum):
    """GPU computational capabilities."""
    CUDA = "cuda"           # NVIDIA CUDA
    ROCM = "rocm"           # AMD ROCm
    METAL = "metal"         # Apple Metal
    OPENCL = "opencl"       # OpenCL (generic)
    NONE = "none"


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    name: str
    vendor: GPUVendor
    memory_mb: int = 0
    compute_capability: str = ""  # e.g., "8.6" for NVIDIA
    driver_version: str = ""
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    device_id: int = 0
    is_available: bool = True
    capabilities: List[GPUCapability] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "vendor": self.vendor.value,
            "memory_mb": self.memory_mb,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "rocm_version": self.rocm_version,
            "device_id": self.device_id,
            "is_available": self.is_available,
            "capabilities": [c.value for c in self.capabilities],
        }
    
    @property
    def memory_gb(self) -> float:
        """Get memory in GB."""
        return self.memory_mb / 1024
    
    @property
    def supports_cuda(self) -> bool:
        """Check if GPU supports CUDA."""
        return GPUCapability.CUDA in self.capabilities
    
    @property
    def supports_rocm(self) -> bool:
        """Check if GPU supports ROCm."""
        return GPUCapability.ROCM in self.capabilities


@dataclass
class GPUEnvironment:
    """GPU environment information."""
    gpus: List[GPUInfo] = field(default_factory=list)
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    cuda_home: Optional[str] = None
    rocm_available: bool = False
    rocm_version: Optional[str] = None
    rocm_home: Optional[str] = None
    metal_available: bool = False
    
    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return len(self.gpus) > 0
    
    @property
    def has_nvidia(self) -> bool:
        """Check for NVIDIA GPU."""
        return any(g.vendor == GPUVendor.NVIDIA for g in self.gpus)
    
    @property
    def has_amd(self) -> bool:
        """Check for AMD GPU."""
        return any(g.vendor == GPUVendor.AMD for g in self.gpus)
    
    @property
    def total_memory_mb(self) -> int:
        """Get total GPU memory across all GPUs."""
        return sum(g.memory_mb for g in self.gpus)
    
    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        """Get the GPU with most memory."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.memory_mb)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpus": [g.to_dict() for g in self.gpus],
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cuda_home": self.cuda_home,
            "rocm_available": self.rocm_available,
            "rocm_version": self.rocm_version,
            "rocm_home": self.rocm_home,
            "metal_available": self.metal_available,
            "has_gpu": self.has_gpu,
            "total_memory_mb": self.total_memory_mb,
        }


class GPUDetector:
    """Detect GPUs and their capabilities.
    
    Supports:
    - NVIDIA GPUs via nvidia-smi
    - AMD GPUs via rocm-smi
    - Apple Silicon via system_profiler
    
    Example:
        >>> detector = GPUDetector()
        >>> env = detector.detect()
        >>> 
        >>> if env.cuda_available:
        ...     print(f"CUDA {env.cuda_version} available")
        ...     for gpu in env.gpus:
        ...         print(f"  {gpu.name}: {gpu.memory_gb:.1f} GB")
    """
    
    def __init__(self, timeout: int = 10):
        """Initialize the detector.
        
        Args:
            timeout: Command timeout in seconds
        """
        self.timeout = timeout
        self._cache: Optional[GPUEnvironment] = None
    
    def detect(self, use_cache: bool = True) -> GPUEnvironment:
        """Detect GPU environment.
        
        Args:
            use_cache: Use cached result if available
            
        Returns:
            GPUEnvironment with detected GPUs
        """
        if use_cache and self._cache is not None:
            return self._cache
        
        env = GPUEnvironment()
        
        # Detect NVIDIA GPUs
        nvidia_gpus = self._detect_nvidia()
        if nvidia_gpus:
            env.gpus.extend(nvidia_gpus)
            env.cuda_available = True
            env.cuda_version = self._get_cuda_version()
            env.cuda_home = self._get_cuda_home()
        
        # Detect AMD GPUs
        amd_gpus = self._detect_amd()
        if amd_gpus:
            env.gpus.extend(amd_gpus)
            env.rocm_available = True
            env.rocm_version = self._get_rocm_version()
            env.rocm_home = self._get_rocm_home()
        
        # Detect Apple Silicon
        if platform.system() == "Darwin":
            apple_gpu = self._detect_apple_silicon()
            if apple_gpu:
                env.gpus.append(apple_gpu)
                env.metal_available = True
        
        self._cache = env
        logger.info(f"GPU detection complete: {len(env.gpus)} GPU(s) found")
        
        return env
    
    def _run_command(self, command: List[str]) -> Tuple[str, bool]:
        """Run a command and return output.
        
        Args:
            command: Command and arguments
            
        Returns:
            Tuple of (output, success)
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.stdout.strip(), result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Command failed: {command[0]} - {e}")
            return "", False
    
    def _detect_nvidia(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-smi."""
        if not shutil.which("nvidia-smi"):
            return []
        
        gpus = []
        
        # Query GPU info
        output, success = self._run_command([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits"
        ])
        
        if not success or not output:
            return []
        
        for line in output.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                try:
                    gpu = GPUInfo(
                        name=parts[1],
                        vendor=GPUVendor.NVIDIA,
                        memory_mb=int(float(parts[2])),
                        driver_version=parts[3],
                        compute_capability=parts[4],
                        device_id=int(parts[0]),
                        capabilities=[GPUCapability.CUDA],
                    )
                    gpus.append(gpu)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse NVIDIA GPU info: {e}")
        
        return gpus
    
    def _detect_amd(self) -> List[GPUInfo]:
        """Detect AMD GPUs using rocm-smi."""
        if not shutil.which("rocm-smi"):
            return []
        
        gpus = []
        
        # Query GPU info
        output, success = self._run_command([
            "rocm-smi",
            "--showproductname",
            "--showmeminfo", "vram",
            "--csv"
        ])
        
        if not success or not output:
            # Try alternative command
            output, success = self._run_command(["rocm-smi", "--showallinfo"])
            if not success:
                return []
        
        # Parse ROCm output (format varies by version)
        lines = output.strip().split("\n")
        device_id = 0
        
        for line in lines:
            # Look for device names
            if "GPU" in line and ":" in line:
                name_match = re.search(r"GPU\[\d+\]\s*:\s*(.+)", line)
                if name_match:
                    gpu = GPUInfo(
                        name=name_match.group(1).strip(),
                        vendor=GPUVendor.AMD,
                        device_id=device_id,
                        capabilities=[GPUCapability.ROCM, GPUCapability.OPENCL],
                    )
                    gpus.append(gpu)
                    device_id += 1
            
            # Look for memory info
            if "Total Memory" in line or "vram" in line.lower():
                mem_match = re.search(r"(\d+)\s*(?:MB|MiB)", line)
                if mem_match and gpus:
                    gpus[-1].memory_mb = int(mem_match.group(1))
        
        return gpus
    
    def _detect_apple_silicon(self) -> Optional[GPUInfo]:
        """Detect Apple Silicon GPU."""
        if platform.system() != "Darwin":
            return None
        
        # Check if Apple Silicon
        output, success = self._run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if not success:
            return None
        
        if "Apple" not in output:
            return None
        
        # Get GPU info via system_profiler
        output, success = self._run_command([
            "system_profiler", "SPDisplaysDataType"
        ])
        
        if not success:
            return None
        
        # Parse system_profiler output
        name = "Apple GPU"
        memory_mb = 0
        
        for line in output.split("\n"):
            line = line.strip()
            if "Chipset Model:" in line:
                name = line.split(":")[1].strip()
            elif "VRAM" in line or "Memory" in line:
                mem_match = re.search(r"(\d+)\s*(?:MB|GB)", line)
                if mem_match:
                    val = int(mem_match.group(1))
                    if "GB" in line:
                        memory_mb = val * 1024
                    else:
                        memory_mb = val
        
        return GPUInfo(
            name=name,
            vendor=GPUVendor.APPLE,
            memory_mb=memory_mb,
            capabilities=[GPUCapability.METAL],
        )
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version."""
        # Try nvcc first
        output, success = self._run_command(["nvcc", "--version"])
        if success:
            match = re.search(r"release (\d+\.\d+)", output)
            if match:
                return match.group(1)
        
        # Try nvidia-smi
        output, success = self._run_command(["nvidia-smi"])
        if success:
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
            if match:
                return match.group(1)
        
        return None
    
    def _get_cuda_home(self) -> Optional[str]:
        """Get CUDA installation path."""
        # Check environment variable
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home and Path(cuda_home).exists():
            return cuda_home
        
        # Common paths
        common_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def _get_rocm_version(self) -> Optional[str]:
        """Get ROCm version."""
        output, success = self._run_command(["rocm-smi", "--version"])
        if success:
            match = re.search(r"(\d+\.\d+\.\d+)", output)
            if match:
                return match.group(1)
        
        # Check rocm-core package
        output, success = self._run_command(["apt", "show", "rocm-core"])
        if success:
            match = re.search(r"Version:\s*(\d+\.\d+)", output)
            if match:
                return match.group(1)
        
        return None
    
    def _get_rocm_home(self) -> Optional[str]:
        """Get ROCm installation path."""
        rocm_home = os.environ.get("ROCM_PATH")
        if rocm_home and Path(rocm_home).exists():
            return rocm_home
        
        common_paths = [
            "/opt/rocm",
            "/usr/local/rocm",
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def clear_cache(self) -> None:
        """Clear cached GPU detection results."""
        self._cache = None
    
    def check_cuda_compatibility(self, min_version: str) -> bool:
        """Check if CUDA version meets minimum requirement.
        
        Args:
            min_version: Minimum CUDA version (e.g., "11.8")
            
        Returns:
            True if requirement met
        """
        env = self.detect()
        if not env.cuda_available or not env.cuda_version:
            return False
        
        try:
            current = tuple(map(int, env.cuda_version.split(".")))
            required = tuple(map(int, min_version.split(".")))
            return current >= required
        except ValueError:
            return False
    
    def get_recommended_backends(self) -> List[str]:
        """Get list of recommended backends based on GPU availability.
        
        Returns:
            List of backend names
        """
        env = self.detect()
        backends = ["cirq", "qiskit"]  # Always available
        
        if env.cuda_available:
            backends.extend(["cuquantum", "qsim_cuda"])
        
        if env.rocm_available:
            backends.append("qsim_rocm")
        
        if env.metal_available:
            backends.append("pennylane_lightning_gpu")
        
        return backends


# Global detector instance
_detector: Optional[GPUDetector] = None


def get_gpu_detector() -> GPUDetector:
    """Get the global GPU detector instance."""
    global _detector
    if _detector is None:
        _detector = GPUDetector()
    return _detector


def detect_gpus() -> GPUEnvironment:
    """Convenience function to detect GPUs."""
    return get_gpu_detector().detect()


def has_cuda() -> bool:
    """Check if CUDA is available."""
    return get_gpu_detector().detect().cuda_available


def has_gpu() -> bool:
    """Check if any GPU is available."""
    return get_gpu_detector().detect().has_gpu
