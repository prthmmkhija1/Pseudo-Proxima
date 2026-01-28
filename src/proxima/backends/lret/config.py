"""LRET Variant Configuration Module.

Provides configuration classes for all LRET backend variants:
- cirq_scalability: Cirq FDM comparison and benchmarking
- pennylane_hybrid: PennyLane device plugin for variational algorithms
- phase7_unified: Multi-framework unified execution

Configuration is stored in ~/.proxima/lret_config.yaml by default.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

logger = logging.getLogger(__name__)


class LRETVariantType(str, Enum):
    """Enum for LRET variant types."""
    
    CIRQ_SCALABILITY = "cirq_scalability"
    PENNYLANE_HYBRID = "pennylane_hybrid"
    PHASE7_UNIFIED = "phase7_unified"
    
    @classmethod
    def from_string(cls, value: str) -> "LRETVariantType":
        """Convert string to variant type."""
        mapping = {
            "cirq_scalability": cls.CIRQ_SCALABILITY,
            "cirq-scalability": cls.CIRQ_SCALABILITY,
            "cirq-scalability-comparison": cls.CIRQ_SCALABILITY,
            "pennylane_hybrid": cls.PENNYLANE_HYBRID,
            "pennylane-hybrid": cls.PENNYLANE_HYBRID,
            "pennylane-documentation-benchmarking": cls.PENNYLANE_HYBRID,
            "phase7_unified": cls.PHASE7_UNIFIED,
            "phase7-unified": cls.PHASE7_UNIFIED,
            "phase-7": cls.PHASE7_UNIFIED,
        }
        normalized = value.lower().strip()
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(f"Unknown LRET variant: {value}")


@dataclass
class LRETVariantConfig:
    """Configuration for a specific LRET variant.
    
    Attributes:
        enabled: Whether this variant is enabled for use
        priority: Selection priority (higher = preferred when auto-selecting)
        install_path: Path where the variant is installed
        auto_connect: Whether to auto-connect on application start
        
        # Cirq Scalability specific
        cirq_fdm_threshold: Qubit count to prefer Cirq FDM (default: 10)
        benchmark_output_dir: Directory for benchmark CSV outputs
        enable_comparison_mode: Auto-compare with Cirq FDM
        
        # PennyLane Hybrid specific
        pennylane_shots: Default shots for PennyLane device
        pennylane_diff_method: Differentiation method for gradients
        default_optimizer: Default optimizer for variational algorithms
        
        # Phase 7 Unified specific
        phase7_backend_preference: Priority order for framework selection
        gate_fusion_enabled: Enable gate fusion optimization
        gate_fusion_mode: Fusion mode ('row', 'column', 'hybrid')
        gpu_enabled: Enable GPU acceleration (requires cuQuantum)
        gpu_device_id: GPU device ID to use
        optimization_level: Optimization level (0=none, 1=basic, 2=full)
    """
    
    # Common settings
    enabled: bool = False
    priority: int = 50
    install_path: Optional[str] = None
    auto_connect: bool = True
    
    # Cirq Scalability specific
    cirq_fdm_threshold: int = 10
    benchmark_output_dir: str = "./benchmarks"
    enable_comparison_mode: bool = True
    
    # PennyLane Hybrid specific
    pennylane_shots: int = 1024
    pennylane_diff_method: Literal["parameter-shift", "adjoint", "backprop"] = "parameter-shift"
    default_optimizer: str = "adam"
    
    # Phase 7 Unified specific
    phase7_backend_preference: list[str] = field(default_factory=lambda: ["cirq", "pennylane", "qiskit"])
    gate_fusion_enabled: bool = True
    gate_fusion_mode: Literal["row", "column", "hybrid"] = "hybrid"
    gpu_enabled: bool = False
    gpu_device_id: int = 0
    optimization_level: int = 2
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LRETVariantConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LRETConfig:
    """Master configuration for all LRET variants.
    
    This class manages configuration for all three LRET variants and provides
    methods for variant selection, persistence, and validation.
    
    Attributes:
        cirq_scalability: Configuration for Cirq scalability variant
        pennylane_hybrid: Configuration for PennyLane hybrid variant
        phase7_unified: Configuration for Phase 7 unified variant
        default_variant: Default variant to use (None for auto-select)
        config_version: Configuration schema version
        install_base_dir: Base directory for variant installations
    """
    
    cirq_scalability: LRETVariantConfig = field(default_factory=LRETVariantConfig)
    pennylane_hybrid: LRETVariantConfig = field(default_factory=LRETVariantConfig)
    phase7_unified: LRETVariantConfig = field(default_factory=LRETVariantConfig)
    
    default_variant: Optional[str] = None
    config_version: str = "1.0.0"
    install_base_dir: str = field(default_factory=lambda: str(Path.home() / ".proxima" / "lret_variants"))
    
    def __post_init__(self):
        """Ensure install paths are set if not provided."""
        base = Path(self.install_base_dir)
        
        if not self.cirq_scalability.install_path:
            self.cirq_scalability.install_path = str(base / "cirq_scalability")
        if not self.pennylane_hybrid.install_path:
            self.pennylane_hybrid.install_path = str(base / "pennylane_hybrid")
        if not self.phase7_unified.install_path:
            self.phase7_unified.install_path = str(base / "phase7_unified")
    
    def get_variant_config(self, variant: str | LRETVariantType) -> LRETVariantConfig:
        """Get configuration for a specific variant.
        
        Args:
            variant: Variant name or type
            
        Returns:
            LRETVariantConfig for the specified variant
            
        Raises:
            ValueError: If variant is unknown
        """
        if isinstance(variant, str):
            variant = LRETVariantType.from_string(variant)
        
        mapping = {
            LRETVariantType.CIRQ_SCALABILITY: self.cirq_scalability,
            LRETVariantType.PENNYLANE_HYBRID: self.pennylane_hybrid,
            LRETVariantType.PHASE7_UNIFIED: self.phase7_unified,
        }
        
        return mapping[variant]
    
    def set_variant_enabled(self, variant: str | LRETVariantType, enabled: bool) -> None:
        """Enable or disable a specific variant.
        
        Args:
            variant: Variant name or type
            enabled: Whether to enable the variant
        """
        config = self.get_variant_config(variant)
        config.enabled = enabled
        logger.info(f"Variant {variant} {'enabled' if enabled else 'disabled'}")
    
    def get_enabled_variants(self) -> list[str]:
        """Get list of enabled variant names.
        
        Returns:
            List of enabled variant names
        """
        variants = []
        if self.cirq_scalability.enabled:
            variants.append(LRETVariantType.CIRQ_SCALABILITY.value)
        if self.pennylane_hybrid.enabled:
            variants.append(LRETVariantType.PENNYLANE_HYBRID.value)
        if self.phase7_unified.enabled:
            variants.append(LRETVariantType.PHASE7_UNIFIED.value)
        return variants
    
    def select_variant_for_task(
        self,
        task_type: str,
        circuit_info: Optional[dict[str, Any]] = None
    ) -> Optional[str]:
        """Auto-select best variant for a given task.
        
        Args:
            task_type: Type of task ('benchmark', 'vqe', 'qaoa', 'qnn', 
                      'general', 'multi_framework')
            circuit_info: Optional circuit info dict with keys:
                         'qubits', 'depth', 'gates', 'framework'
                         
        Returns:
            Variant name or None if no suitable variant available
        """
        enabled = self.get_enabled_variants()
        if not enabled:
            logger.warning("No LRET variants enabled")
            return None
        
        circuit_info = circuit_info or {}
        
        # Task-specific selection logic
        task_mapping = {
            "benchmark": LRETVariantType.CIRQ_SCALABILITY.value,
            "comparison": LRETVariantType.CIRQ_SCALABILITY.value,
            "scalability": LRETVariantType.CIRQ_SCALABILITY.value,
            "vqe": LRETVariantType.PENNYLANE_HYBRID.value,
            "qaoa": LRETVariantType.PENNYLANE_HYBRID.value,
            "qnn": LRETVariantType.PENNYLANE_HYBRID.value,
            "variational": LRETVariantType.PENNYLANE_HYBRID.value,
            "gradient": LRETVariantType.PENNYLANE_HYBRID.value,
            "multi_framework": LRETVariantType.PHASE7_UNIFIED.value,
            "unified": LRETVariantType.PHASE7_UNIFIED.value,
            "cross_platform": LRETVariantType.PHASE7_UNIFIED.value,
        }
        
        # Check if task has a preferred variant that is enabled
        preferred = task_mapping.get(task_type.lower())
        if preferred and preferred in enabled:
            logger.debug(f"Selected {preferred} for task type {task_type}")
            return preferred
        
        # Check if user set a default variant
        if self.default_variant and self.default_variant in enabled:
            logger.debug(f"Using default variant: {self.default_variant}")
            return self.default_variant
        
        # Auto-select based on circuit info if available
        if circuit_info:
            framework = circuit_info.get("framework", "").lower()
            if framework == "pennylane" and LRETVariantType.PENNYLANE_HYBRID.value in enabled:
                return LRETVariantType.PENNYLANE_HYBRID.value
            elif framework in ["cirq", "qiskit"] and LRETVariantType.PHASE7_UNIFIED.value in enabled:
                return LRETVariantType.PHASE7_UNIFIED.value
        
        # Return highest priority enabled variant
        priorities = {
            LRETVariantType.CIRQ_SCALABILITY.value: self.cirq_scalability.priority,
            LRETVariantType.PENNYLANE_HYBRID.value: self.pennylane_hybrid.priority,
            LRETVariantType.PHASE7_UNIFIED.value: self.phase7_unified.priority,
        }
        
        selected = max(enabled, key=lambda v: priorities.get(v, 0))
        logger.debug(f"Auto-selected variant {selected} based on priority")
        return selected
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "config_version": self.config_version,
            "install_base_dir": self.install_base_dir,
            "default_variant": self.default_variant,
            "cirq_scalability": self.cirq_scalability.to_dict(),
            "pennylane_hybrid": self.pennylane_hybrid.to_dict(),
            "phase7_unified": self.phase7_unified.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LRETConfig":
        """Create configuration from dictionary."""
        config = cls(
            config_version=data.get("config_version", "1.0.0"),
            install_base_dir=data.get("install_base_dir", str(Path.home() / ".proxima" / "lret_variants")),
            default_variant=data.get("default_variant"),
        )
        
        if "cirq_scalability" in data:
            config.cirq_scalability = LRETVariantConfig.from_dict(data["cirq_scalability"])
        if "pennylane_hybrid" in data:
            config.pennylane_hybrid = LRETVariantConfig.from_dict(data["pennylane_hybrid"])
        if "phase7_unified" in data:
            config.phase7_unified = LRETVariantConfig.from_dict(data["phase7_unified"])
        
        return config
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save configuration to YAML file.
        
        Args:
            path: Optional path to save to. Defaults to ~/.proxima/lret_config.yaml
            
        Returns:
            Path where configuration was saved
        """
        if path is None:
            path = Path.home() / ".proxima" / "lret_config.yaml"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"LRET configuration saved to {path}")
        return path
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LRETConfig":
        """Load configuration from YAML file.
        
        Args:
            path: Optional path to load from. Defaults to ~/.proxima/lret_config.yaml
            
        Returns:
            Loaded LRETConfig or default configuration if file not found
        """
        if path is None:
            path = Path.home() / ".proxima" / "lret_config.yaml"
        
        if not path.exists():
            logger.debug(f"No configuration file found at {path}, using defaults")
            return cls()
        
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            
            config = cls.from_dict(data)
            logger.info(f"LRET configuration loaded from {path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            return cls()
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate the configuration.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate install base directory
        base_dir = Path(self.install_base_dir)
        if not base_dir.parent.exists():
            errors.append(f"Parent directory of install_base_dir does not exist: {base_dir.parent}")
        
        # Validate variant-specific settings
        if self.cirq_scalability.cirq_fdm_threshold < 2:
            errors.append("cirq_fdm_threshold must be at least 2")
        
        if self.pennylane_hybrid.pennylane_shots < 1:
            errors.append("pennylane_shots must be at least 1")
        
        if self.phase7_unified.optimization_level not in [0, 1, 2]:
            errors.append("optimization_level must be 0, 1, or 2")
        
        if self.phase7_unified.gate_fusion_mode not in ["row", "column", "hybrid"]:
            errors.append("gate_fusion_mode must be 'row', 'column', or 'hybrid'")
        
        # Validate default variant if set
        if self.default_variant:
            try:
                LRETVariantType.from_string(self.default_variant)
            except ValueError:
                errors.append(f"Invalid default_variant: {self.default_variant}")
        
        return len(errors) == 0, errors


# Singleton configuration instance
_global_config: Optional[LRETConfig] = None


def get_lret_config() -> LRETConfig:
    """Get the global LRET configuration.
    
    Loads from disk on first call, returns cached instance afterwards.
    
    Returns:
        Global LRETConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = LRETConfig.load()
    return _global_config


def save_lret_config(config: Optional[LRETConfig] = None) -> Path:
    """Save the LRET configuration to disk.
    
    Args:
        config: Optional config to save. Uses global config if not provided.
        
    Returns:
        Path where configuration was saved
    """
    global _global_config
    if config is not None:
        _global_config = config
    elif _global_config is None:
        _global_config = LRETConfig()
    
    return _global_config.save()


def reset_lret_config() -> LRETConfig:
    """Reset the global configuration to defaults.
    
    Returns:
        New default LRETConfig instance
    """
    global _global_config
    _global_config = LRETConfig()
    return _global_config
