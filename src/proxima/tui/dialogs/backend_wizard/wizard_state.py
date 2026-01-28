"""Backend Wizard State Management.

Centralized state management for the backend addition wizard.
Tracks all user inputs across wizard steps and provides validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class BackendType(Enum):
    """Types of backends that can be added."""
    PYTHON_LIBRARY = "python_library"
    COMMAND_LINE = "command_line"
    API_SERVER = "api_server"
    CUSTOM = "custom"


class GateMappingMode(Enum):
    """Gate mapping configuration modes."""
    AUTOMATIC = "automatic"
    TEMPLATE = "template"
    MANUAL = "manual"


@dataclass
class GateMapping:
    """Represents a single gate mapping."""
    proxima_gate: str
    backend_gate: str
    parameters: List[str] = field(default_factory=list)
    custom_code: Optional[str] = None


@dataclass
class BackendWizardState:
    """
    State container for the backend addition wizard.
    
    Holds all configuration data as the user progresses through
    the wizard steps. This state is passed between steps and
    ultimately used to generate the backend code.
    
    Attributes:
        current_step: Current wizard step (1-7)
        can_proceed: Whether user can proceed to next step
        backend_type: Type of backend (python_library, command_line, etc.)
        backend_name: Internal identifier for the backend
        display_name: User-facing name shown in UI
        version: Semantic version string
        description: Brief description of the backend
        library_name: Python package/module name to import
        author: Backend author/maintainer
        simulator_types: List of supported simulation types
        max_qubits: Maximum number of qubits supported
        supports_noise: Whether backend supports noise models
        supports_gpu: Whether backend supports GPU acceleration
        supports_batching: Whether backend supports batch execution
        gate_mapping_mode: How gates are mapped
        gate_mappings: Custom gate mappings if using manual mode
        generated_code: Generated code files
        test_results: Results from backend tests
    """
    
    # Wizard progress
    current_step: int = 1
    can_proceed: bool = False
    
    # Step 1: Backend Type
    backend_type: Optional[str] = None
    
    # Step 2: Basic Information
    backend_name: str = ""
    display_name: str = ""
    version: str = "1.0.0"
    description: str = ""
    library_name: str = ""
    author: str = ""
    
    # Step 3: Capabilities
    simulator_types: List[str] = field(default_factory=list)
    max_qubits: int = 20
    supports_noise: bool = False
    supports_gpu: bool = False
    supports_batching: bool = False
    supports_parameter_binding: bool = False
    supports_custom_gates: bool = False
    
    # Step 4: Gate Mapping
    gate_mapping_mode: str = "automatic"
    gate_template: Optional[str] = None
    gate_mappings: Dict[str, GateMapping] = field(default_factory=dict)
    
    # Step 5: Code Template
    code_template: str = "standard"
    custom_code: Dict[str, str] = field(default_factory=dict)
    
    # Step 6: Testing
    test_results: Dict[str, Any] = field(default_factory=dict)
    tests_passed: bool = False
    
    # Step 7: Review
    confirmed: bool = False
    
    # Generated outputs
    generated_code: Dict[str, str] = field(default_factory=dict)
    output_directory: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if not self.simulator_types:
            self.simulator_types = ["state_vector"]
    
    def validate_step(self, step: int) -> tuple[bool, str]:
        """
        Validate data for a specific step.
        
        Args:
            step: Step number to validate (1-7)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if step == 1:
            if not self.backend_type:
                return False, "Please select a backend type"
            return True, ""
        
        elif step == 2:
            if not self.backend_name:
                return False, "Backend name is required"
            if not self.display_name:
                return False, "Display name is required"
            if not self._is_valid_identifier(self.backend_name):
                return False, "Backend name must be a valid Python identifier"
            return True, ""
        
        elif step == 3:
            if not self.simulator_types:
                return False, "At least one simulator type must be selected"
            if self.max_qubits < 1:
                return False, "Maximum qubits must be at least 1"
            return True, ""
        
        elif step == 4:
            if self.gate_mapping_mode == "manual" and not self.gate_mappings:
                return False, "Manual mode requires at least one gate mapping"
            return True, ""
        
        elif step == 5:
            # Code template is optional
            return True, ""
        
        elif step == 6:
            if not self.tests_passed:
                return False, "All tests must pass before proceeding"
            return True, ""
        
        elif step == 7:
            if not self.confirmed:
                return False, "Please confirm the configuration"
            return True, ""
        
        return False, f"Unknown step: {step}"
    
    def _is_valid_identifier(self, name: str) -> bool:
        """Check if name is a valid Python identifier."""
        import re
        return bool(re.match(r'^[a-z][a-z0-9_]*$', name))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "current_step": self.current_step,
            "backend_type": self.backend_type,
            "backend_name": self.backend_name,
            "display_name": self.display_name,
            "version": self.version,
            "description": self.description,
            "library_name": self.library_name,
            "author": self.author,
            "simulator_types": self.simulator_types,
            "max_qubits": self.max_qubits,
            "supports_noise": self.supports_noise,
            "supports_gpu": self.supports_gpu,
            "supports_batching": self.supports_batching,
            "gate_mapping_mode": self.gate_mapping_mode,
            "gate_template": self.gate_template,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendWizardState":
        """Create state from dictionary."""
        state = cls()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state
    
    def reset(self) -> None:
        """Reset state to initial values."""
        self.current_step = 1
        self.can_proceed = False
        self.backend_type = None
        self.backend_name = ""
        self.display_name = ""
        self.version = "1.0.0"
        self.description = ""
        self.library_name = ""
        self.author = ""
        self.simulator_types = ["state_vector"]
        self.max_qubits = 20
        self.supports_noise = False
        self.supports_gpu = False
        self.supports_batching = False
        self.gate_mapping_mode = "automatic"
        self.gate_template = None
        self.gate_mappings = {}
        self.code_template = "standard"
        self.custom_code = {}
        self.test_results = {}
        self.tests_passed = False
        self.confirmed = False
        self.generated_code = {}
        self.output_directory = None
