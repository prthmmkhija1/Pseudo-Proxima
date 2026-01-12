"""Default configuration values and constants for configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Default configuration tree used when no files are present.
DEFAULT_CONFIG: dict[str, Any] = {
    "general": {
        "verbosity": "info",
        "output_format": "text",
        "color_enabled": True,
    },
    "backends": {
        "default_backend": "auto",
        "parallel_execution": False,
        "timeout_seconds": 300,
    },
    "llm": {
        "provider": "none",
        "model": "",
        "local_endpoint": "",
        "api_key_env_var": "",
        "require_consent": True,
    },
    "resources": {
        "memory_warn_threshold_mb": 4096,
        "memory_critical_threshold_mb": 8192,
        "max_execution_time_seconds": 3600,
    },
    "consent": {
        "auto_approve_local_llm": False,
        "auto_approve_remote_llm": False,
        "remember_decisions": False,
    },
    # ==========================================================================
    # Step 1.5: QuEST Backend Configuration
    # ==========================================================================
    "quest": {
        # Enable/disable QuEST backend
        "enabled": True,
        # Numerical precision: "single", "double", or "quad"
        "precision": "double",
        # GPU acceleration: "auto" (detect), True (force), False (disable)
        "gpu_enabled": "auto",
        # GPU device ID to use (0-indexed)
        "gpu_device_id": 0,
        # Number of OpenMP threads: 0 = auto-detect based on CPU cores
        "openmp_threads": 0,
        # Truncation threshold for density matrix rank reduction
        "truncation_threshold": 1e-10,
        # Maximum rank for density matrices: 0 = unlimited
        "max_rank": 0,
        # Maximum qubits for statevector simulation
        "max_qubits_sv": 30,
        # Maximum qubits for density matrix simulation
        "max_qubits_dm": 15,
        # Memory limit in MB: 0 = unlimited
        "memory_limit_mb": 0,
        # Validate state normalization after each operation (debug mode)
        "validate_normalization": False,
    },
    # ==========================================================================
    # Step 2.3: cuQuantum Backend Configuration
    # ==========================================================================
    "cuquantum": {
        # Enable/disable cuQuantum backend
        "enabled": True,
        # Execution mode: "gpu_only", "gpu_preferred", "auto"
        "execution_mode": "gpu_preferred",
        # GPU device ID to use (0-indexed)
        "gpu_device_id": 0,
        # Numerical precision: "single" or "double"
        "precision": "double",
        # GPU memory limit in MB: 0 = unlimited
        "memory_limit_mb": 0,
        # cuStateVec workspace size in MB
        "workspace_size_mb": 1024,
        # Wait for GPU completion before returning
        "blocking": True,
        # Enable gate fusion optimization
        "fusion_enabled": True,
        # Maximum qubits for GPU simulation
        "max_qubits": 35,
        # Allow fallback to CPU if GPU fails
        "fallback_to_cpu": True,
    },
    # ==========================================================================
    # Step 3.5: qsim Backend Configuration
    # ==========================================================================
    "qsim": {
        # Enable/disable qsim backend
        "enabled": True,
        # Number of threads: 0 = auto-detect based on CPU cores
        "num_threads": 0,
        # Gate fusion level: "off", "low", "medium", "high"
        "gate_fusion": "medium",
        # Maximum fused gate size (1-6 qubits)
        "max_fused_gate_size": 4,
        # Verbosity level (0 = quiet, 1 = info, 2 = debug)
        "verbosity": 0,
        # Use GPU acceleration (requires qsim GPU build)
        "use_gpu": False,
        # Maximum qubits for simulation (based on available memory)
        "max_qubits": 32,
        # Random seed for reproducible simulations (None = random)
        "seed": None,
    },
    # ==========================================================================
    # Backend Priority Configuration (for auto-selection)
    # ==========================================================================
    "backend_priorities": {
        # Priority order for statevector simulation with GPU available
        "state_vector_gpu": ["cuquantum", "quest", "qsim", "cirq", "qiskit"],
        # Priority order for statevector simulation CPU-only
        "state_vector_cpu": ["qsim", "quest", "cirq", "qiskit"],
        # Priority order for density matrix simulation
        "density_matrix": ["quest", "cirq", "qiskit", "lret"],
        # Priority order for noisy circuit simulation
        "noisy_circuit": ["quest", "qiskit", "cirq"],
    },
}


ENV_PREFIX = "PROXIMA"
USER_CONFIG_PATH = Path.home() / ".proxima" / "config.yaml"
PROJECT_CONFIG_FILENAME = "proxima.yaml"
DEFAULT_CONFIG_RELATIVE_PATH = Path("configs") / "default.yaml"


# ==============================================================================
# Step 1.5: QuEST Configuration Helper Functions
# ==============================================================================


def get_quest_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get QuEST-specific configuration with defaults.

    Args:
        config: Optional configuration dictionary to merge with defaults

    Returns:
        Complete QuEST configuration dictionary
    """
    quest_defaults = DEFAULT_CONFIG.get("quest", {})
    if config is None:
        return quest_defaults.copy()

    # Merge with user config
    result = quest_defaults.copy()
    user_quest = config.get("quest", {})
    result.update(user_quest)
    return result


def validate_quest_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate QuEST configuration values.

    Args:
        config: QuEST configuration dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Validate precision
    precision = config.get("precision", "double")
    if precision not in ("single", "double", "quad"):
        errors.append(
            f"Invalid precision '{precision}'. Must be 'single', 'double', or 'quad'"
        )

    # Validate gpu_enabled
    gpu_enabled = config.get("gpu_enabled", "auto")
    if gpu_enabled not in ("auto", True, False, "true", "false"):
        errors.append(
            f"Invalid gpu_enabled '{gpu_enabled}'. Must be 'auto', true, or false"
        )

    # Validate numeric values
    numeric_fields = [
        ("gpu_device_id", 0, 16),
        ("openmp_threads", 0, 512),
        ("max_qubits_sv", 1, 50),
        ("max_qubits_dm", 1, 25),
        ("memory_limit_mb", 0, 1024 * 1024),  # Max 1TB
    ]

    for field, min_val, max_val in numeric_fields:
        value = config.get(field)
        if value is not None:
            try:
                int_val = int(value)
                if int_val < min_val or int_val > max_val:
                    errors.append(
                        f"{field} must be between {min_val} and {max_val}, got {int_val}"
                    )
            except (TypeError, ValueError):
                errors.append(f"{field} must be an integer, got {type(value).__name__}")

    # Validate truncation_threshold
    threshold = config.get("truncation_threshold")
    if threshold is not None:
        try:
            float_val = float(threshold)
            if float_val < 0 or float_val > 1:
                errors.append(
                    f"truncation_threshold must be between 0 and 1, got {float_val}"
                )
        except (TypeError, ValueError):
            errors.append(
                f"truncation_threshold must be a float, got {type(threshold).__name__}"
            )

    return (len(errors) == 0, errors)


# ==============================================================================
# Step 2.3: cuQuantum Configuration Helper Functions
# ==============================================================================


def get_cuquantum_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get cuQuantum-specific configuration with defaults.

    Args:
        config: Optional configuration dictionary to merge with defaults

    Returns:
        Complete cuQuantum configuration dictionary
    """
    cuquantum_defaults = DEFAULT_CONFIG.get("cuquantum", {})
    if config is None:
        return cuquantum_defaults.copy()

    # Merge with user config
    result = cuquantum_defaults.copy()
    user_cuquantum = config.get("cuquantum", {})
    result.update(user_cuquantum)
    return result


def validate_cuquantum_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate cuQuantum configuration values.

    Args:
        config: cuQuantum configuration dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Validate execution_mode
    execution_mode = config.get("execution_mode", "gpu_preferred")
    if execution_mode not in ("gpu_only", "gpu_preferred", "auto"):
        errors.append(
            f"Invalid execution_mode '{execution_mode}'. Must be 'gpu_only', 'gpu_preferred', or 'auto'"
        )

    # Validate precision
    precision = config.get("precision", "double")
    if precision not in ("single", "double"):
        errors.append(f"Invalid precision '{precision}'. Must be 'single' or 'double'")

    # Validate numeric values
    numeric_fields = [
        ("gpu_device_id", 0, 16),
        ("memory_limit_mb", 0, 1024 * 1024),  # Max 1TB
        ("workspace_size_mb", 64, 64 * 1024),  # 64MB to 64GB
        ("max_qubits", 1, 50),
    ]

    for field, min_val, max_val in numeric_fields:
        value = config.get(field)
        if value is not None:
            try:
                int_val = int(value)
                if int_val < min_val or int_val > max_val:
                    errors.append(
                        f"{field} must be between {min_val} and {max_val}, got {int_val}"
                    )
            except (TypeError, ValueError):
                errors.append(f"{field} must be an integer, got {type(value).__name__}")

    # Validate boolean values
    bool_fields = ["blocking", "fusion_enabled", "fallback_to_cpu"]
    for field in bool_fields:
        value = config.get(field)
        if value is not None and not isinstance(value, bool):
            if value not in ("true", "false", True, False):
                errors.append(f"{field} must be a boolean, got {type(value).__name__}")

    return (len(errors) == 0, errors)


def estimate_gpu_memory_required(num_qubits: int, precision: str = "double") -> float:
    """Estimate GPU memory required for state vector simulation.

    Formula: (2^n * bytes_per_amplitude + workspace) / (1024^2)

    Args:
        num_qubits: Number of qubits in circuit
        precision: "single" (8 bytes) or "double" (16 bytes)

    Returns:
        Estimated GPU memory required in MB
    """
    bytes_per_amplitude = 16 if precision == "double" else 8

    # State vector size
    sv_size = (2**num_qubits) * bytes_per_amplitude

    # Add workspace (default 1GB)
    workspace = 1024 * 1024 * 1024

    # Add overhead (10%)
    total = (sv_size + workspace) * 1.1

    return total / (1024 * 1024)


# ==============================================================================
# Step 3.5: qsim Configuration Helper Functions
# ==============================================================================


def get_qsim_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get qsim-specific configuration with defaults.

    Args:
        config: Optional configuration dictionary to merge with defaults

    Returns:
        Complete qsim configuration dictionary
    """
    qsim_defaults = DEFAULT_CONFIG.get("qsim", {})
    if config is None:
        return qsim_defaults.copy()

    # Merge with user config
    result = qsim_defaults.copy()
    user_qsim = config.get("qsim", {})
    result.update(user_qsim)
    return result


def validate_qsim_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate qsim configuration values.

    Args:
        config: qsim configuration dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Validate gate_fusion level
    gate_fusion = config.get("gate_fusion", "medium")
    if gate_fusion not in ("off", "low", "medium", "high"):
        errors.append(
            f"Invalid gate_fusion '{gate_fusion}'. Must be 'off', 'low', 'medium', or 'high'"
        )

    # Validate numeric values
    numeric_fields = [
        ("num_threads", 0, 512),
        ("max_fused_gate_size", 1, 6),
        ("verbosity", 0, 2),
        ("max_qubits", 1, 40),
    ]

    for field, min_val, max_val in numeric_fields:
        value = config.get(field)
        if value is not None:
            try:
                int_val = int(value)
                if int_val < min_val or int_val > max_val:
                    errors.append(
                        f"{field} must be between {min_val} and {max_val}, got {int_val}"
                    )
            except (TypeError, ValueError):
                errors.append(f"{field} must be an integer, got {type(value).__name__}")

    # Validate boolean values
    use_gpu = config.get("use_gpu")
    if use_gpu is not None and not isinstance(use_gpu, bool):
        if use_gpu not in ("true", "false", True, False):
            errors.append(f"use_gpu must be a boolean, got {type(use_gpu).__name__}")

    # Validate seed (can be None or integer)
    seed = config.get("seed")
    if seed is not None:
        try:
            int(seed)
        except (TypeError, ValueError):
            errors.append(f"seed must be an integer or None, got {type(seed).__name__}")

    return (len(errors) == 0, errors)


def estimate_qsim_memory_required(num_qubits: int) -> float:
    """Estimate memory required for qsim state vector simulation.

    Formula: 2^n * 16 bytes (complex128) / (1024^2)

    Args:
        num_qubits: Number of qubits in circuit

    Returns:
        Estimated memory required in MB
    """
    bytes_per_amplitude = 16  # complex128

    # State vector size
    sv_size = (2**num_qubits) * bytes_per_amplitude

    # Add overhead (20% for workspace)
    total = sv_size * 1.2

    return total / (1024 * 1024)


def get_backend_priority(
    simulation_type: str,
    gpu_available: bool = False,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Get backend priority list for auto-selection.

    Args:
        simulation_type: "state_vector", "density_matrix", or "noisy"
        gpu_available: Whether GPU is available
        config: Optional configuration to override defaults

    Returns:
        Ordered list of backend names to try
    """
    priorities = DEFAULT_CONFIG.get("backend_priorities", {})
    if config:
        priorities = config.get("backend_priorities", priorities)

    if simulation_type == "state_vector":
        if gpu_available:
            return priorities.get(
                "state_vector_gpu", ["cuquantum", "quest", "qsim", "cirq", "qiskit"]
            )
        return priorities.get("state_vector_cpu", ["qsim", "quest", "cirq", "qiskit"])
    elif simulation_type == "density_matrix":
        return priorities.get("density_matrix", ["quest", "cirq", "qiskit"])
    elif simulation_type in ("noisy", "noisy_circuit"):
        return priorities.get("noisy_circuit", ["quest", "qiskit", "cirq"])

    # Default fallback
    return ["cuquantum", "quest", "qsim", "cirq", "qiskit", "lret"]


# ==============================================================================
# Step 4.3: Backend Auto-Selection Configuration
# ==============================================================================

# Auto-selection settings for intelligent backend selection
AUTO_SELECTION_CONFIG: dict[str, Any] = {
    # Enable/disable auto-selection
    "enabled": True,
    # Default selection strategy: "balanced", "performance", "memory", "accuracy", "gpu_preferred", "cpu_optimized"
    "default_strategy": "balanced",
    # Whether to prefer GPU backends when available
    "prefer_gpu": True,
    # Memory threshold for parallel execution (fraction of available memory)
    "memory_threshold": 0.8,
    # Enable LLM-assisted selection for complex decisions
    "llm_assisted": False,
    # Cache selection decisions for repeated circuits
    "cache_selections": True,
    # Cache TTL in seconds
    "cache_ttl_seconds": 3600,
    # Log selection decisions
    "log_decisions": True,
    # Fallback backend if auto-selection fails
    "fallback_backend": "numpy",
}

# GPU-specific auto-selection settings
GPU_SELECTION_CONFIG: dict[str, Any] = {
    # GPU memory reservation percentage (don't use more than this)
    "memory_reservation_percent": 80,
    # Minimum qubits to consider GPU
    "min_qubits_for_gpu": 15,
    # Preferred GPU backend order
    "gpu_backend_priority": ["cuquantum", "quest", "cupy"],
    # Fallback to CPU if GPU execution fails
    "fallback_to_cpu": True,
    # GPU warmup enabled (first run may be slower)
    "warmup_enabled": True,
}

# CPU-specific auto-selection settings
CPU_SELECTION_CONFIG: dict[str, Any] = {
    # Preferred CPU backend order
    "cpu_backend_priority": ["qsim", "quest", "cirq", "qiskit", "numpy"],
    # Thread count for multi-threaded backends (0 = auto-detect)
    "thread_count": 0,
    # Enable vectorization detection (AVX2/AVX512)
    "detect_vectorization": True,
    # Preferred backend for small circuits (< 15 qubits)
    "small_circuit_backend": "numpy",
    # Preferred backend for medium circuits (15-25 qubits)
    "medium_circuit_backend": "qsim",
    # Preferred backend for large circuits (> 25 qubits)
    "large_circuit_backend": "qsim",
}

# Simulation type specific configurations
SIMULATION_TYPE_CONFIG: dict[str, Any] = {
    "state_vector": {
        "gpu_priority": ["cuquantum", "quest", "qsim", "cirq", "qiskit"],
        "cpu_priority": ["qsim", "quest", "cirq", "qiskit", "numpy"],
    },
    "density_matrix": {
        "priority": ["quest", "cirq", "qiskit", "lret"],
    },
    "noisy_circuit": {
        "priority": ["quest", "qiskit", "cirq"],
    },
}


def get_auto_selection_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get auto-selection configuration with defaults.

    Args:
        config: Optional configuration dictionary to merge with defaults

    Returns:
        Complete auto-selection configuration dictionary
    """
    result = AUTO_SELECTION_CONFIG.copy()
    if config:
        user_config = config.get("auto_selection", {})
        result.update(user_config)
    return result


def get_gpu_selection_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get GPU selection configuration with defaults.

    Args:
        config: Optional configuration dictionary to merge with defaults

    Returns:
        Complete GPU selection configuration dictionary
    """
    result = GPU_SELECTION_CONFIG.copy()
    if config:
        user_config = config.get("gpu_selection", {})
        result.update(user_config)
    return result


def get_cpu_selection_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get CPU selection configuration with defaults.

    Args:
        config: Optional configuration dictionary to merge with defaults

    Returns:
        Complete CPU selection configuration dictionary
    """
    result = CPU_SELECTION_CONFIG.copy()
    if config:
        user_config = config.get("cpu_selection", {})
        result.update(user_config)
    return result


def get_simulation_type_priority(
    simulation_type: str,
    gpu_available: bool = False,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Get backend priority list for specific simulation type.

    Args:
        simulation_type: "state_vector", "density_matrix", or "noisy_circuit"
        gpu_available: Whether GPU is available
        config: Optional configuration to override defaults

    Returns:
        Ordered list of backend names to try
    """
    sim_config = SIMULATION_TYPE_CONFIG.copy()
    if config:
        sim_config.update(config.get("simulation_type", {}))

    type_config = sim_config.get(simulation_type, {})

    if simulation_type == "state_vector":
        if gpu_available:
            return type_config.get(
                "gpu_priority", ["cuquantum", "quest", "qsim", "cirq", "qiskit"]
            )
        return type_config.get(
            "cpu_priority", ["qsim", "quest", "cirq", "qiskit", "numpy"]
        )
    elif simulation_type == "density_matrix":
        return type_config.get("priority", ["quest", "cirq", "qiskit", "lret"])
    elif simulation_type == "noisy_circuit":
        return type_config.get("priority", ["quest", "qiskit", "cirq"])

    # Default fallback
    return ["qsim", "quest", "cirq", "qiskit", "numpy"]


def get_recommended_backend_for_circuit(
    qubit_count: int,
    simulation_type: str = "state_vector",
    gpu_available: bool = False,
    needs_noise: bool = False,
    config: dict[str, Any] | None = None,
) -> str:
    """Get recommended backend for given circuit requirements.

    Args:
        qubit_count: Number of qubits in the circuit
        simulation_type: Type of simulation
        gpu_available: Whether GPU is available
        needs_noise: Whether noise simulation is required
        config: Optional configuration to override defaults

    Returns:
        Recommended backend name
    """
    cpu_config = get_cpu_selection_config(config)

    # Handle noisy circuits
    if needs_noise:
        priorities = get_simulation_type_priority(
            "noisy_circuit", gpu_available, config
        )
        return priorities[0] if priorities else "qiskit"

    # Handle density matrix
    if simulation_type == "density_matrix":
        priorities = get_simulation_type_priority(
            "density_matrix", gpu_available, config
        )
        return priorities[0] if priorities else "quest"

    # Handle state vector based on circuit size
    if qubit_count < 15:
        return cpu_config.get("small_circuit_backend", "numpy")
    elif qubit_count <= 25:
        if gpu_available:
            return "cuquantum"
        return cpu_config.get("medium_circuit_backend", "qsim")
    else:
        # Large circuits
        if gpu_available:
            return "cuquantum"
        return cpu_config.get("large_circuit_backend", "qsim")
