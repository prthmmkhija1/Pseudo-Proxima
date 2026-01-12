"""GPU Memory Management for cuQuantum backend.

Step 2.4: Optimize GPU Memory Management
=========================================

This module provides advanced GPU memory management capabilities for cuQuantum:
1. Pre-Execution Memory Checks - Query and validate GPU memory before execution
2. Memory Pooling - Reuse GPU memory allocations across executions
3. Batch Processing - Execute circuits sequentially with memory monitoring
4. Fallback Strategy - Intelligent fallback when memory is insufficient

Memory Estimation Formula:
--------------------------
Total GPU Memory Required = State Vector Size + Workspace + Overhead
= (2^n * bytes_per_amplitude) + workspace_size + overhead

Where:
- n = number of qubits
- bytes_per_amplitude = 8 (complex64/single) or 16 (complex128/double)
- workspace_size = cuStateVec scratch memory (default 1GB)
- overhead = CUDA runtime and driver buffers (~500MB)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from proxima.backends.cuquantum_adapter import (
    CuQuantumConfig,
    CuQuantumPrecision,
    GPUDeviceInfo,
)

# =============================================================================
# Memory Constants and Thresholds
# =============================================================================

# Bytes per amplitude based on precision
BYTES_PER_AMPLITUDE = {
    CuQuantumPrecision.SINGLE: 8,  # complex64: 4 bytes real + 4 bytes imag
    CuQuantumPrecision.DOUBLE: 16,  # complex128: 8 bytes real + 8 bytes imag
}

# Default workspace sizes in MB
DEFAULT_WORKSPACE_MB = 1024  # 1 GB default cuStateVec workspace
MINIMUM_WORKSPACE_MB = 256  # 256 MB minimum workspace

# Overhead estimates in MB
CUDA_RUNTIME_OVERHEAD_MB = 200  # CUDA runtime overhead
DRIVER_BUFFER_OVERHEAD_MB = 300  # Driver buffer overhead
TOTAL_OVERHEAD_MB = CUDA_RUNTIME_OVERHEAD_MB + DRIVER_BUFFER_OVERHEAD_MB

# Memory safety margins
MEMORY_SAFETY_MARGIN = 0.10  # 10% safety margin
MEMORY_WARNING_THRESHOLD = 0.85  # Warn at 85% memory usage
MEMORY_CRITICAL_THRESHOLD = 0.95  # Critical at 95% memory usage

# Maximum qubits for different GPU memory sizes (with double precision)
MAX_QUBITS_BY_MEMORY = {
    4096: 28,  # 4 GB GPU
    8192: 29,  # 8 GB GPU
    12288: 30,  # 12 GB GPU
    16384: 30,  # 16 GB GPU
    24576: 31,  # 24 GB GPU
    32768: 31,  # 32 GB GPU
    49152: 32,  # 48 GB GPU (A6000, etc.)
    81920: 33,  # 80 GB GPU (A100, H100)
}


class MemoryAllocationStrategy(str, Enum):
    """Strategy for GPU memory allocation."""

    ON_DEMAND = "on_demand"  # Allocate memory as needed
    PRE_ALLOCATE = "pre_allocate"  # Pre-allocate memory pool
    LAZY_POOL = "lazy_pool"  # Lazy initialization with pooling


class MemoryCleanupPolicy(str, Enum):
    """Policy for GPU memory cleanup."""

    IMMEDIATE = "immediate"  # Clean up immediately after use
    DEFERRED = "deferred"  # Defer cleanup for reuse
    MANUAL = "manual"  # Manual cleanup by user


@dataclass
class MemoryEstimate:
    """Detailed memory requirement estimate for a circuit.

    Attributes:
        qubit_count: Number of qubits in the circuit
        state_vector_mb: Memory for state vector in MB
        workspace_mb: Memory for cuStateVec workspace in MB
        overhead_mb: Memory for CUDA runtime overhead in MB
        total_required_mb: Total memory required in MB
        total_with_safety_mb: Total memory with safety margin
        precision: Numerical precision used
        can_fit_on_device: Whether circuit fits on specified device
        recommended_device_id: Recommended GPU device ID
        fallback_required: Whether CPU fallback is needed
        fallback_reason: Reason for fallback if required
    """

    qubit_count: int
    state_vector_mb: float
    workspace_mb: float
    overhead_mb: float
    total_required_mb: float
    total_with_safety_mb: float
    precision: CuQuantumPrecision
    can_fit_on_device: bool = True
    recommended_device_id: int = 0
    fallback_required: bool = False
    fallback_reason: str = ""


@dataclass
class MemoryPoolStats:
    """Statistics for GPU memory pool.

    Attributes:
        pool_size_mb: Total pool size in MB
        allocated_mb: Currently allocated memory in MB
        available_mb: Available memory in pool in MB
        allocation_count: Number of active allocations
        reuse_count: Number of times memory was reused
        cache_hit_rate: Cache hit rate (0.0 to 1.0)
        last_cleanup_time: Timestamp of last cleanup
    """

    pool_size_mb: float = 0.0
    allocated_mb: float = 0.0
    available_mb: float = 0.0
    allocation_count: int = 0
    reuse_count: int = 0
    cache_hit_rate: float = 0.0
    last_cleanup_time: float = 0.0


@dataclass
class BatchExecutionResult:
    """Result of batch circuit execution.

    Attributes:
        total_circuits: Total number of circuits in batch
        successful_circuits: Number of successfully executed circuits
        failed_circuits: Number of failed circuits
        total_execution_time_ms: Total execution time in milliseconds
        peak_memory_mb: Peak GPU memory usage in MB
        results: List of individual execution results
        errors: List of errors for failed circuits
        memory_fragmentation: Memory fragmentation level (0.0 to 1.0)
    """

    total_circuits: int = 0
    successful_circuits: int = 0
    failed_circuits: int = 0
    total_execution_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    memory_fragmentation: float = 0.0


# =============================================================================
# GPU Memory Manager Class
# =============================================================================


class GPUMemoryManager:
    """Manages GPU memory for cuQuantum executions.

    This class provides:
    - Pre-execution memory validation
    - Memory pooling for efficient reuse
    - Batch processing with memory monitoring
    - Intelligent fallback when GPU memory is insufficient

    Example:
        >>> manager = GPUMemoryManager(config)
        >>> estimate = manager.estimate_memory(30)  # 30 qubits
        >>> if estimate.can_fit_on_device:
        ...     result = manager.execute_with_memory_check(circuit)
    """

    def __init__(
        self,
        config: CuQuantumConfig | None = None,
        allocation_strategy: MemoryAllocationStrategy = MemoryAllocationStrategy.LAZY_POOL,
        cleanup_policy: MemoryCleanupPolicy = MemoryCleanupPolicy.DEFERRED,
    ):
        """Initialize GPU memory manager.

        Args:
            config: cuQuantum configuration
            allocation_strategy: Memory allocation strategy
            cleanup_policy: Memory cleanup policy
        """
        self._config = config or CuQuantumConfig()
        self._allocation_strategy = allocation_strategy
        self._cleanup_policy = cleanup_policy
        self._logger = logging.getLogger(__name__)

        # Memory pool state
        self._pool_initialized = False
        self._pool_stats = MemoryPoolStats()
        self._pool_lock = threading.Lock()

        # Cached GPU info
        self._gpu_devices: list[GPUDeviceInfo] = []
        self._selected_device_id = self._config.gpu_device_id

        # Memory cache for reuse
        self._memory_cache: dict[int, Any] = {}  # qubit_count -> cached allocation

    # =========================================================================
    # Pre-Execution Memory Checks (Step 2.4.1)
    # =========================================================================

    def estimate_memory(
        self,
        qubit_count: int,
        precision: CuQuantumPrecision | None = None,
        workspace_mb: float | None = None,
    ) -> MemoryEstimate:
        """Estimate GPU memory requirements for a circuit.

        This implements the memory estimation formula from the guide:
        Total = (2^n * bytes_per_amplitude) + workspace + overhead

        Args:
            qubit_count: Number of qubits in the circuit
            precision: Numerical precision (defaults to config)
            workspace_mb: Workspace size in MB (defaults to config)

        Returns:
            MemoryEstimate with detailed breakdown
        """
        precision = precision or self._config.precision
        workspace_mb = workspace_mb or self._config.workspace_size_mb

        # Calculate state vector size: 2^n amplitudes * bytes per amplitude
        bytes_per_amp = BYTES_PER_AMPLITUDE[precision]
        state_vector_bytes = (2**qubit_count) * bytes_per_amp
        state_vector_mb = state_vector_bytes / (1024 * 1024)

        # Total required memory
        total_required_mb = state_vector_mb + workspace_mb + TOTAL_OVERHEAD_MB

        # Add safety margin
        total_with_safety_mb = total_required_mb * (1 + MEMORY_SAFETY_MARGIN)

        # Check if it fits on device
        can_fit = False
        recommended_device = self._selected_device_id
        fallback_required = False
        fallback_reason = ""

        # Get GPU info
        gpu_info = self._get_gpu_info(self._selected_device_id)
        if gpu_info:
            available_mb = gpu_info.free_memory_mb
            can_fit = total_with_safety_mb <= available_mb

            if not can_fit:
                # Check other GPUs
                for device in self._gpu_devices:
                    if device.free_memory_mb >= total_with_safety_mb:
                        recommended_device = device.device_id
                        can_fit = True
                        break

                if not can_fit:
                    fallback_required = True
                    fallback_reason = (
                        f"Required {total_with_safety_mb:.1f} MB but only "
                        f"{available_mb:.1f} MB available on GPU {self._selected_device_id}"
                    )
        else:
            fallback_required = True
            fallback_reason = "No GPU device available"

        return MemoryEstimate(
            qubit_count=qubit_count,
            state_vector_mb=state_vector_mb,
            workspace_mb=workspace_mb,
            overhead_mb=TOTAL_OVERHEAD_MB,
            total_required_mb=total_required_mb,
            total_with_safety_mb=total_with_safety_mb,
            precision=precision,
            can_fit_on_device=can_fit,
            recommended_device_id=recommended_device,
            fallback_required=fallback_required,
            fallback_reason=fallback_reason,
        )

    def check_memory_before_execution(
        self,
        qubit_count: int,
        raise_on_insufficient: bool = True,
    ) -> tuple[bool, MemoryEstimate]:
        """Check if GPU has sufficient memory before execution.

        Args:
            qubit_count: Number of qubits
            raise_on_insufficient: Raise exception if memory insufficient

        Returns:
            Tuple of (can_execute, memory_estimate)

        Raises:
            CuQuantumMemoryError: If memory insufficient and raise_on_insufficient=True
        """
        from proxima.backends.cuquantum_adapter import CuQuantumMemoryError

        estimate = self.estimate_memory(qubit_count)

        if not estimate.can_fit_on_device:
            self._logger.warning(
                f"Insufficient GPU memory for {qubit_count} qubits. "
                f"Required: {estimate.total_with_safety_mb:.1f} MB"
            )

            if raise_on_insufficient and not self._config.fallback_to_cpu:
                gpu_info = self._get_gpu_info(self._selected_device_id)
                available_mb = gpu_info.free_memory_mb if gpu_info else 0
                raise CuQuantumMemoryError(
                    required_mb=estimate.total_with_safety_mb,
                    available_mb=available_mb,
                    qubit_count=qubit_count,
                    device_id=self._selected_device_id,
                )

        return estimate.can_fit_on_device, estimate

    def query_current_gpu_memory(
        self, device_id: int | None = None
    ) -> dict[str, float]:
        """Query current GPU memory status.

        Args:
            device_id: GPU device ID (defaults to configured device)

        Returns:
            Dictionary with total_mb, used_mb, free_mb, utilization
        """
        device_id = device_id or self._selected_device_id

        gpu_info = self._get_gpu_info(device_id)
        if not gpu_info:
            return {
                "total_mb": 0,
                "used_mb": 0,
                "free_mb": 0,
                "utilization": 0.0,
            }

        used_mb = gpu_info.total_memory_mb - gpu_info.free_memory_mb
        utilization = (
            used_mb / gpu_info.total_memory_mb if gpu_info.total_memory_mb > 0 else 0
        )

        return {
            "total_mb": gpu_info.total_memory_mb,
            "used_mb": used_mb,
            "free_mb": gpu_info.free_memory_mb,
            "utilization": utilization,
        }

    def clear_gpu_cache(self, device_id: int | None = None) -> bool:
        """Clear GPU cache to free memory.

        Args:
            device_id: GPU device ID (defaults to configured device)

        Returns:
            True if cache was cleared successfully
        """
        try:
            # Try different methods to clear cache

            # Method 1: Use cupy
            try:
                import cupy as cp

                with cp.cuda.Device(device_id or self._selected_device_id):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                self._logger.debug("Cleared GPU cache using cupy")
                return True
            except Exception:
                pass

            # Method 2: Use torch if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self._logger.debug("Cleared GPU cache using torch")
                    return True
            except Exception:
                pass

            # Method 3: Run garbage collection
            import gc

            gc.collect()
            self._logger.debug("Ran garbage collection")
            return True

        except Exception as e:
            self._logger.warning(f"Failed to clear GPU cache: {e}")
            return False

    # =========================================================================
    # Memory Pooling (Step 2.4.2)
    # =========================================================================

    def initialize_memory_pool(self, pool_size_mb: float | None = None) -> bool:
        """Initialize GPU memory pool for efficient reuse.

        Args:
            pool_size_mb: Pool size in MB (defaults to 50% of free memory)

        Returns:
            True if pool initialized successfully
        """
        with self._pool_lock:
            if self._pool_initialized:
                self._logger.debug("Memory pool already initialized")
                return True

            try:
                gpu_info = self._get_gpu_info(self._selected_device_id)
                if not gpu_info:
                    self._logger.warning("No GPU available for memory pool")
                    return False

                # Default to 50% of free memory
                if pool_size_mb is None:
                    pool_size_mb = gpu_info.free_memory_mb * 0.5

                # Initialize pool using cupy if available
                try:
                    import cupy as cp

                    with cp.cuda.Device(self._selected_device_id):
                        # Create memory pool with specified size
                        mempool = cp.get_default_memory_pool()
                        mempool.set_limit(size=int(pool_size_mb * 1024 * 1024))

                        self._pool_stats.pool_size_mb = pool_size_mb
                        self._pool_stats.available_mb = pool_size_mb
                        self._pool_initialized = True

                        self._logger.info(
                            f"GPU memory pool initialized: {pool_size_mb:.1f} MB "
                            f"on device {self._selected_device_id}"
                        )
                        return True
                except Exception as e:
                    self._logger.debug(f"CuPy pool init failed: {e}")

                # Fallback: Just track stats without actual pool
                self._pool_stats.pool_size_mb = pool_size_mb
                self._pool_stats.available_mb = pool_size_mb
                self._pool_initialized = True
                self._logger.info(
                    f"Memory pool stats initialized: {pool_size_mb:.1f} MB"
                )
                return True

            except Exception as e:
                self._logger.error(f"Failed to initialize memory pool: {e}")
                return False

    def allocate_from_pool(self, size_mb: float) -> tuple[bool, int | None]:
        """Allocate memory from pool.

        Args:
            size_mb: Required memory in MB

        Returns:
            Tuple of (success, allocation_id)
        """
        with self._pool_lock:
            if not self._pool_initialized:
                self.initialize_memory_pool()

            if size_mb > self._pool_stats.available_mb:
                self._logger.warning(
                    f"Insufficient pool memory: need {size_mb:.1f} MB, "
                    f"available {self._pool_stats.available_mb:.1f} MB"
                )
                return False, None

            # Create allocation
            allocation_id = id(time.time())  # Simple unique ID
            self._pool_stats.allocated_mb += size_mb
            self._pool_stats.available_mb -= size_mb
            self._pool_stats.allocation_count += 1

            self._logger.debug(
                f"Allocated {size_mb:.1f} MB from pool (ID: {allocation_id})"
            )
            return True, allocation_id

    def release_to_pool(self, allocation_id: int, size_mb: float) -> bool:
        """Release memory back to pool.

        Args:
            allocation_id: Allocation ID from allocate_from_pool
            size_mb: Size of allocation in MB

        Returns:
            True if released successfully
        """
        with self._pool_lock:
            self._pool_stats.allocated_mb -= size_mb
            self._pool_stats.available_mb += size_mb
            self._pool_stats.allocation_count -= 1
            self._pool_stats.reuse_count += 1

            # Update cache hit rate
            total_ops = self._pool_stats.allocation_count + self._pool_stats.reuse_count
            if total_ops > 0:
                self._pool_stats.cache_hit_rate = (
                    self._pool_stats.reuse_count / total_ops
                )

            self._logger.debug(
                f"Released {size_mb:.1f} MB to pool (ID: {allocation_id})"
            )
            return True

    def get_pool_stats(self) -> MemoryPoolStats:
        """Get current memory pool statistics."""
        with self._pool_lock:
            return MemoryPoolStats(
                pool_size_mb=self._pool_stats.pool_size_mb,
                allocated_mb=self._pool_stats.allocated_mb,
                available_mb=self._pool_stats.available_mb,
                allocation_count=self._pool_stats.allocation_count,
                reuse_count=self._pool_stats.reuse_count,
                cache_hit_rate=self._pool_stats.cache_hit_rate,
                last_cleanup_time=self._pool_stats.last_cleanup_time,
            )

    def cleanup_pool(self) -> bool:
        """Clean up memory pool and release all allocations.

        Returns:
            True if cleanup successful
        """
        with self._pool_lock:
            self._memory_cache.clear()
            self.clear_gpu_cache()

            self._pool_stats.allocated_mb = 0
            self._pool_stats.available_mb = self._pool_stats.pool_size_mb
            self._pool_stats.allocation_count = 0
            self._pool_stats.last_cleanup_time = time.time()

            self._logger.info("Memory pool cleaned up")
            return True

    # =========================================================================
    # Batch Processing (Step 2.4.3)
    # =========================================================================

    def execute_batch(
        self,
        circuits: list[Any],
        executor: Callable[[Any], Any],
        options: dict[str, Any] | None = None,
    ) -> BatchExecutionResult:
        """Execute multiple circuits in batch with memory monitoring.

        Circuits are executed sequentially to avoid memory fragmentation.
        Memory is monitored and cleared between executions if needed.

        Args:
            circuits: List of circuits to execute
            executor: Function to execute each circuit
            options: Execution options

        Returns:
            BatchExecutionResult with all results
        """
        options = options or {}
        batch_result = BatchExecutionResult(total_circuits=len(circuits))

        start_time = time.perf_counter()
        peak_memory = 0.0

        for i, circuit in enumerate(circuits):
            self._logger.debug(f"Executing circuit {i+1}/{len(circuits)}")

            try:
                # Check memory before execution
                current_memory = self.query_current_gpu_memory()
                if current_memory["utilization"] > MEMORY_WARNING_THRESHOLD:
                    self._logger.warning(
                        f"High GPU memory usage ({current_memory['utilization']*100:.1f}%), "
                        f"clearing cache before circuit {i+1}"
                    )
                    self.clear_gpu_cache()

                # Execute circuit
                result = executor(circuit)
                batch_result.results.append(result)
                batch_result.successful_circuits += 1

                # Track peak memory
                post_memory = self.query_current_gpu_memory()
                peak_memory = max(peak_memory, post_memory["used_mb"])

            except Exception as e:
                self._logger.error(f"Circuit {i+1} failed: {e}")
                batch_result.errors.append(
                    {
                        "circuit_index": i,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                batch_result.failed_circuits += 1

                # Clear cache after failure
                self.clear_gpu_cache()

        batch_result.total_execution_time_ms = (time.perf_counter() - start_time) * 1000
        batch_result.peak_memory_mb = peak_memory

        # Calculate memory fragmentation
        pool_stats = self.get_pool_stats()
        if pool_stats.pool_size_mb > 0:
            batch_result.memory_fragmentation = 1 - (
                pool_stats.available_mb / pool_stats.pool_size_mb
            )

        self._logger.info(
            f"Batch execution complete: {batch_result.successful_circuits}/"
            f"{batch_result.total_circuits} circuits succeeded, "
            f"peak memory: {peak_memory:.1f} MB"
        )

        return batch_result

    # =========================================================================
    # Fallback Strategy (Step 2.4.4)
    # =========================================================================

    def should_fallback_to_cpu(
        self,
        qubit_count: int,
        force_gpu: bool = False,
    ) -> tuple[bool, str]:
        """Determine if execution should fall back to CPU.

        Args:
            qubit_count: Number of qubits
            force_gpu: If True, never fallback (may raise exception)

        Returns:
            Tuple of (should_fallback, reason)
        """
        if force_gpu:
            return False, ""

        if not self._config.fallback_to_cpu:
            return False, ""

        # Check GPU availability
        if not self._gpu_devices:
            return True, "No GPU devices available"

        # Check memory
        estimate = self.estimate_memory(qubit_count)
        if estimate.fallback_required:
            return True, estimate.fallback_reason

        # Check for small circuits (overhead may negate benefit)
        if qubit_count < 15:
            return True, f"Small circuit ({qubit_count} qubits) - CPU may be faster"

        return False, ""

    def get_fallback_recommendation(
        self,
        qubit_count: int,
    ) -> dict[str, Any]:
        """Get detailed fallback recommendation.

        Args:
            qubit_count: Number of qubits

        Returns:
            Dictionary with recommendation details
        """
        estimate = self.estimate_memory(qubit_count)
        should_fallback, reason = self.should_fallback_to_cpu(qubit_count)

        recommendation = {
            "should_fallback": should_fallback,
            "reason": reason,
            "memory_estimate": {
                "state_vector_mb": estimate.state_vector_mb,
                "total_required_mb": estimate.total_required_mb,
                "available_mb": self.query_current_gpu_memory()["free_mb"],
            },
            "alternatives": [],
        }

        if should_fallback:
            recommendation["alternatives"] = [
                {
                    "action": "reduce_qubits",
                    "description": f"Reduce circuit to {estimate.qubit_count - 2} qubits",
                },
                {
                    "action": "use_single_precision",
                    "description": "Switch to single precision (halves memory)",
                },
                {
                    "action": "use_different_backend",
                    "description": "Use qsim for CPU-optimized execution",
                },
            ]

        return recommendation

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_gpu_info(self, device_id: int) -> GPUDeviceInfo | None:
        """Get GPU info for a specific device."""
        for device in self._gpu_devices:
            if device.device_id == device_id:
                return device
        return None

    def set_gpu_devices(self, devices: list[GPUDeviceInfo]) -> None:
        """Set available GPU devices (called by adapter)."""
        self._gpu_devices = devices

    def calculate_max_qubits_for_memory(
        self,
        available_mb: float,
        precision: CuQuantumPrecision | None = None,
    ) -> int:
        """Calculate maximum qubits that fit in available memory.

        Args:
            available_mb: Available GPU memory in MB
            precision: Numerical precision

        Returns:
            Maximum qubit count
        """
        precision = precision or self._config.precision
        bytes_per_amp = BYTES_PER_AMPLITUDE[precision]

        # Available for state vector = total - workspace - overhead - safety
        usable_mb = available_mb - self._config.workspace_size_mb - TOTAL_OVERHEAD_MB
        usable_mb *= 1 - MEMORY_SAFETY_MARGIN

        if usable_mb <= 0:
            return 0

        # 2^n * bytes_per_amp = usable_bytes
        # n = log2(usable_bytes / bytes_per_amp)
        usable_bytes = usable_mb * 1024 * 1024
        max_amplitudes = usable_bytes / bytes_per_amp

        import math

        max_qubits = int(math.log2(max_amplitudes))

        return max(0, max_qubits)


# =============================================================================
# Convenience Functions
# =============================================================================


def estimate_gpu_memory_for_qubits(
    qubit_count: int,
    precision: str = "double",
    workspace_mb: float = 1024,
) -> dict[str, float]:
    """Quick estimate of GPU memory for a given qubit count.

    Args:
        qubit_count: Number of qubits
        precision: "single" or "double"
        workspace_mb: Workspace size in MB

    Returns:
        Dictionary with memory breakdown
    """
    prec = (
        CuQuantumPrecision.SINGLE
        if precision == "single"
        else CuQuantumPrecision.DOUBLE
    )
    bytes_per_amp = BYTES_PER_AMPLITUDE[prec]

    state_vector_bytes = (2**qubit_count) * bytes_per_amp
    state_vector_mb = state_vector_bytes / (1024 * 1024)

    total_mb = state_vector_mb + workspace_mb + TOTAL_OVERHEAD_MB
    total_with_safety = total_mb * (1 + MEMORY_SAFETY_MARGIN)

    return {
        "qubit_count": qubit_count,
        "state_vector_mb": state_vector_mb,
        "workspace_mb": workspace_mb,
        "overhead_mb": TOTAL_OVERHEAD_MB,
        "total_mb": total_mb,
        "total_with_safety_mb": total_with_safety,
        "precision": precision,
    }


def get_recommended_precision_for_memory(
    qubit_count: int,
    available_mb: float,
    workspace_mb: float = 1024,
) -> str:
    """Get recommended precision based on available memory.

    Args:
        qubit_count: Number of qubits
        available_mb: Available GPU memory in MB
        workspace_mb: Workspace size in MB

    Returns:
        "single" or "double"
    """
    double_estimate = estimate_gpu_memory_for_qubits(
        qubit_count, "double", workspace_mb
    )
    single_estimate = estimate_gpu_memory_for_qubits(
        qubit_count, "single", workspace_mb
    )

    if double_estimate["total_with_safety_mb"] <= available_mb:
        return "double"
    elif single_estimate["total_with_safety_mb"] <= available_mb:
        return "single"
    else:
        return "double"  # Return double anyway, let validation handle it
