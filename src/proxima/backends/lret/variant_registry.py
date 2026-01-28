"""LRET Variant Registry.

This module provides a centralized registry for managing LRET backend variants.
It handles variant registration, discovery, and lifecycle management.

The registry supports:
- Dynamic variant registration
- Variant capability querying
- Adapter factory pattern
- Health checking and status monitoring
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from proxima.backends.base import BaseBackendAdapter

logger = logging.getLogger(__name__)


class VariantStatus(Enum):
    """Status of a registered variant."""
    
    UNKNOWN = "unknown"
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    FUNCTIONAL = "functional"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class RegisteredVariant:
    """Information about a registered variant."""
    
    name: str
    display_name: str
    adapter_class: Type[BaseBackendAdapter]
    factory: Optional[Callable[..., BaseBackendAdapter]] = None
    status: VariantStatus = VariantStatus.UNKNOWN
    version: Optional[str] = None
    description: str = ""
    enabled: bool = True
    priority: int = 50
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VariantRegistry:
    """Central registry for LRET backend variants.
    
    Provides a singleton registry for managing all LRET variant adapters.
    Handles registration, lookup, factory creation, and status monitoring.
    
    Example:
        >>> registry = VariantRegistry.get_instance()
        >>> registry.register(
        ...     name='cirq_scalability',
        ...     adapter_class=LRETCirqScalabilityAdapter,
        ...     display_name='Cirq Scalability'
        ... )
        >>> adapter = registry.create_adapter('cirq_scalability')
    """
    
    _instance: Optional['VariantRegistry'] = None
    
    def __init__(self):
        """Initialize the variant registry."""
        self._variants: Dict[str, RegisteredVariant] = {}
        self._adapters: Dict[str, BaseBackendAdapter] = {}
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> 'VariantRegistry':
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._auto_register()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
    
    def _auto_register(self) -> None:
        """Auto-register known LRET variants."""
        if self._initialized:
            return
        
        # Register Cirq Scalability variant
        try:
            from proxima.backends.lret.cirq_scalability import LRETCirqScalabilityAdapter
            self.register(
                name='cirq_scalability',
                adapter_class=LRETCirqScalabilityAdapter,
                display_name='LRET Cirq Scalability',
                description='LRET vs Cirq FDM benchmarking and scalability analysis',
                priority=70,
                metadata={
                    'branch': 'cirq-scalability-comparison',
                    'features': ['benchmarking', 'scalability', 'cirq_comparison'],
                }
            )
            logger.info("Registered cirq_scalability variant")
        except ImportError as e:
            logger.debug(f"Could not register cirq_scalability: {e}")
        
        # Register PennyLane Hybrid variant
        try:
            # PennyLane variant uses device pattern, create a wrapper
            from proxima.backends.lret.pennylane_device import QLRETDevice
            from proxima.backends.lret.algorithms import VQE, QAOA, QuantumNeuralNetwork
            
            self.register(
                name='pennylane_hybrid',
                adapter_class=PennyLaneHybridAdapter,
                display_name='LRET PennyLane Hybrid',
                description='PennyLane device plugin with VQE, QAOA, QNN support',
                priority=80,
                metadata={
                    'branch': 'pennylane-documentation-benchmarking',
                    'features': ['vqe', 'qaoa', 'qnn', 'gradients'],
                    'device_class': QLRETDevice,
                    'algorithms': {
                        'vqe': VQE,
                        'qaoa': QAOA,
                        'qnn': QuantumNeuralNetwork,
                    }
                }
            )
            logger.info("Registered pennylane_hybrid variant")
        except ImportError as e:
            logger.debug(f"Could not register pennylane_hybrid: {e}")
        
        # Register Phase 7 Unified variant
        try:
            from proxima.backends.lret.phase7_unified import LRETPhase7UnifiedAdapter
            self.register(
                name='phase7_unified',
                adapter_class=LRETPhase7UnifiedAdapter,
                display_name='LRET Phase 7 Unified',
                description='Multi-framework integration (Cirq, PennyLane, Qiskit)',
                priority=90,
                metadata={
                    'branch': 'phase-7',
                    'features': ['multi_framework', 'gate_fusion', 'gpu_support'],
                }
            )
            logger.info("Registered phase7_unified variant")
        except ImportError as e:
            logger.debug(f"Could not register phase7_unified: {e}")
        
        self._initialized = True
    
    def register(
        self,
        name: str,
        adapter_class: Type[BaseBackendAdapter],
        display_name: Optional[str] = None,
        description: str = "",
        factory: Optional[Callable[..., BaseBackendAdapter]] = None,
        priority: int = 50,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a variant with the registry.
        
        Args:
            name: Unique identifier for the variant
            adapter_class: Class that implements BaseBackendAdapter
            display_name: Human-readable name
            description: Variant description
            factory: Optional factory function for creating adapters
            priority: Priority for auto-selection (higher = preferred)
            enabled: Whether variant is enabled
            metadata: Additional metadata
        """
        if name in self._variants:
            logger.warning(f"Variant {name} already registered, updating")
        
        variant = RegisteredVariant(
            name=name,
            display_name=display_name or name,
            adapter_class=adapter_class,
            factory=factory,
            description=description,
            priority=priority,
            enabled=enabled,
            metadata=metadata or {},
        )
        
        # Check if variant is functional
        variant.status = self._check_variant_status(variant)
        
        self._variants[name] = variant
        logger.info(f"Registered variant: {name} (status: {variant.status.value})")
    
    def unregister(self, name: str) -> bool:
        """Unregister a variant.
        
        Args:
            name: Variant name to unregister
            
        Returns:
            True if variant was unregistered
        """
        if name in self._variants:
            del self._variants[name]
            if name in self._adapters:
                del self._adapters[name]
            logger.info(f"Unregistered variant: {name}")
            return True
        return False
    
    def _check_variant_status(self, variant: RegisteredVariant) -> VariantStatus:
        """Check the status of a variant."""
        if not variant.enabled:
            return VariantStatus.DISABLED
        
        try:
            # Try to create an instance
            adapter = variant.adapter_class()
            
            # Check if it's available
            if hasattr(adapter, 'is_available'):
                if adapter.is_available():
                    variant.version = getattr(adapter, 'get_version', lambda: None)()
                    return VariantStatus.FUNCTIONAL
                else:
                    return VariantStatus.INSTALLED
            else:
                return VariantStatus.FUNCTIONAL
                
        except ImportError:
            return VariantStatus.NOT_INSTALLED
        except Exception as e:
            variant.last_error = str(e)
            return VariantStatus.ERROR
    
    def get_variant(self, name: str) -> Optional[RegisteredVariant]:
        """Get a registered variant by name."""
        return self._variants.get(name)
    
    def list_variants(self) -> List[str]:
        """Get list of all registered variant names."""
        return list(self._variants.keys())
    
    def list_enabled_variants(self) -> List[str]:
        """Get list of enabled variant names."""
        return [
            name for name, var in self._variants.items() 
            if var.enabled
        ]
    
    def list_functional_variants(self) -> List[str]:
        """Get list of functional variant names."""
        return [
            name for name, var in self._variants.items()
            if var.status == VariantStatus.FUNCTIONAL
        ]
    
    def create_adapter(
        self,
        name: str,
        **kwargs
    ) -> Optional[BaseBackendAdapter]:
        """Create an adapter instance for a variant.
        
        Args:
            name: Variant name
            **kwargs: Arguments to pass to adapter constructor
            
        Returns:
            Adapter instance or None if creation fails
        """
        variant = self._variants.get(name)
        if not variant:
            logger.error(f"Unknown variant: {name}")
            return None
        
        if not variant.enabled:
            logger.error(f"Variant {name} is disabled")
            return None
        
        try:
            if variant.factory:
                adapter = variant.factory(**kwargs)
            else:
                adapter = variant.adapter_class(**kwargs)
            
            self._adapters[name] = adapter
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to create adapter for {name}: {e}")
            variant.last_error = str(e)
            variant.status = VariantStatus.ERROR
            return None
    
    def get_adapter(self, name: str) -> Optional[BaseBackendAdapter]:
        """Get an existing adapter instance or create one.
        
        Args:
            name: Variant name
            
        Returns:
            Adapter instance or None
        """
        if name in self._adapters:
            return self._adapters[name]
        return self.create_adapter(name)
    
    def get_best_variant(
        self,
        features: Optional[List[str]] = None,
        frameworks: Optional[List[str]] = None
    ) -> Optional[str]:
        """Get the best variant based on criteria.
        
        Args:
            features: Required features (e.g., ['vqe', 'gpu'])
            frameworks: Required frameworks (e.g., ['cirq', 'qiskit'])
            
        Returns:
            Best matching variant name or None
        """
        candidates = []
        
        for name, variant in self._variants.items():
            if variant.status != VariantStatus.FUNCTIONAL:
                continue
            
            if not variant.enabled:
                continue
            
            # Check features
            if features:
                variant_features = variant.metadata.get('features', [])
                if not all(f in variant_features for f in features):
                    continue
            
            # Check frameworks
            if frameworks:
                variant_features = variant.metadata.get('features', [])
                framework_map = {
                    'cirq': 'cirq_comparison',
                    'pennylane': 'vqe',  # PennyLane variant has VQE
                    'qiskit': 'multi_framework',
                }
                required = [framework_map.get(fw, fw) for fw in frameworks]
                if not any(f in variant_features for f in required):
                    continue
            
            candidates.append((name, variant.priority))
        
        if not candidates:
            return None
        
        # Return highest priority
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def get_status_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get status summary of all variants.
        
        Returns:
            Dictionary with variant status information
        """
        summary = {}
        
        for name, variant in self._variants.items():
            summary[name] = {
                'display_name': variant.display_name,
                'status': variant.status.value,
                'enabled': variant.enabled,
                'priority': variant.priority,
                'version': variant.version,
                'description': variant.description,
                'features': variant.metadata.get('features', []),
                'last_error': variant.last_error,
            }
        
        return summary
    
    def enable_variant(self, name: str) -> bool:
        """Enable a variant.
        
        Args:
            name: Variant name
            
        Returns:
            True if successful
        """
        variant = self._variants.get(name)
        if variant:
            variant.enabled = True
            variant.status = self._check_variant_status(variant)
            return True
        return False
    
    def disable_variant(self, name: str) -> bool:
        """Disable a variant.
        
        Args:
            name: Variant name
            
        Returns:
            True if successful
        """
        variant = self._variants.get(name)
        if variant:
            variant.enabled = False
            variant.status = VariantStatus.DISABLED
            return True
        return False
    
    def refresh_status(self, name: Optional[str] = None) -> None:
        """Refresh status of variant(s).
        
        Args:
            name: Specific variant to refresh, or all if None
        """
        if name:
            variant = self._variants.get(name)
            if variant:
                variant.status = self._check_variant_status(variant)
        else:
            for variant in self._variants.values():
                variant.status = self._check_variant_status(variant)


class PennyLaneHybridAdapter(BaseBackendAdapter):
    """Adapter wrapper for PennyLane Hybrid variant.
    
    Wraps the PennyLane device and algorithms to conform to
    the BaseBackendAdapter interface.
    """
    
    def __init__(self, wires: int = 4, shots: int = 1024, **kwargs):
        """Initialize PennyLane hybrid adapter.
        
        Args:
            wires: Number of qubits
            shots: Number of measurement shots
            **kwargs: Additional device arguments
        """
        from proxima.backends.lret.pennylane_device import QLRETDevice
        from proxima.backends.base import Capabilities, SimulatorType
        
        self._wires = wires
        self._shots = shots
        self._kwargs = kwargs
        self._device = QLRETDevice(wires=wires, shots=shots, **kwargs)
        
        self._capabilities = Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR],
            max_qubits=24,
            supports_noise=True,
            supports_gpu=False,
            supports_batching=False,
            custom_features={
                'vqe': True,
                'qaoa': True,
                'qnn': True,
                'gradients': True,
            }
        )
    
    @property
    def device(self):
        """Get the underlying PennyLane device."""
        return self._device
    
    def get_name(self) -> str:
        return "lret_pennylane_hybrid"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_capabilities(self):
        return self._capabilities
    
    def validate_circuit(self, circuit: Any):
        from proxima.backends.base import ValidationResult
        # PennyLane circuits are QNodes
        if callable(circuit):
            return ValidationResult(valid=True)
        return ValidationResult(
            valid=False,
            message="Expected a PennyLane QNode"
        )
    
    def estimate_resources(self, circuit: Any):
        from proxima.backends.base import ResourceEstimate
        return ResourceEstimate(
            memory_mb=(2 ** self._wires * 16) / (1024 * 1024),
            time_ms=self._wires * 10,
        )
    
    def execute(
        self,
        circuit: Any,
        options: Optional[Dict[str, Any]] = None
    ):
        from proxima.backends.base import (
            ExecutionResult, SimulatorType, ResultType
        )
        import time
        
        start = time.time()
        
        try:
            if callable(circuit):
                result = circuit()
            else:
                result = None
            
            elapsed = (time.time() - start) * 1000
            
            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=SimulatorType.STATE_VECTOR,
                execution_time_ms=elapsed,
                qubit_count=self._wires,
                shot_count=self._shots,
                result_type=ResultType.COUNTS,
                data={'result': result},
                metadata={'device': 'qlret'},
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=SimulatorType.STATE_VECTOR,
                execution_time_ms=elapsed,
                qubit_count=self._wires,
                shot_count=0,
                result_type=ResultType.COUNTS,
                data={'error': str(e)},
                metadata={},
            )
    
    def supports_simulator(self, sim_type) -> bool:
        from proxima.backends.base import SimulatorType
        return sim_type == SimulatorType.STATE_VECTOR
    
    def is_available(self) -> bool:
        try:
            from proxima.backends.lret.pennylane_device import QLRETDevice
            return True
        except ImportError:
            return False


# Convenience functions

def get_registry() -> VariantRegistry:
    """Get the global variant registry instance."""
    return VariantRegistry.get_instance()


def list_available_variants() -> List[str]:
    """List all available (functional) variants."""
    return get_registry().list_functional_variants()


def get_variant_adapter(name: str) -> Optional[BaseBackendAdapter]:
    """Get an adapter for a specific variant."""
    return get_registry().get_adapter(name)


def get_best_variant_adapter(
    features: Optional[List[str]] = None,
    frameworks: Optional[List[str]] = None
) -> Optional[BaseBackendAdapter]:
    """Get the best variant adapter based on criteria."""
    registry = get_registry()
    best = registry.get_best_variant(features=features, frameworks=frameworks)
    if best:
        return registry.get_adapter(best)
    return None
