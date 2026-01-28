"""Backend Code Generator.

Generates the complete backend implementation files based on
wizard configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import textwrap

from .wizard_state import BackendWizardState, BackendType


@dataclass
class GeneratedFile:
    """Represents a generated file."""
    path: Path
    content: str
    description: str


class BackendCodeGenerator:
    """
    Generates backend implementation code from wizard state.
    
    Creates all necessary files for a complete backend implementation:
    - Main backend module
    - Test file
    - Configuration file
    - Documentation
    """
    
    def __init__(self, state: BackendWizardState, output_base: Optional[Path] = None):
        """
        Initialize the code generator.
        
        Args:
            state: The completed wizard state
            output_base: Base directory for output (defaults to src/proxima/backends/contrib)
        """
        self.state = state
        self.output_base = output_base or Path("src/proxima/backends/contrib")
        self.output_dir = self.output_base / self.state.backend_name
    
    def generate_all_files(self) -> Tuple[bool, List[Path], Dict[str, str]]:
        """
        Generate all backend files.
        
        Returns:
            Tuple of (success, list of created file paths, dict of file contents)
        """
        try:
            files: List[GeneratedFile] = []
            
            # Main backend file
            files.append(self._generate_backend_module())
            
            # __init__.py
            files.append(self._generate_init_file())
            
            # Test file
            files.append(self._generate_test_file())
            
            # README
            files.append(self._generate_readme())
            
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write all files
            created_paths = []
            contents = {}
            
            for gen_file in files:
                file_path = self.output_dir / gen_file.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(gen_file.content)
                
                created_paths.append(file_path)
                contents[str(gen_file.path)] = gen_file.content
            
            return True, created_paths, contents
        
        except Exception as e:
            return False, [], {"error": str(e)}
    
    def _generate_backend_module(self) -> GeneratedFile:
        """Generate the main backend module file."""
        # Use the code from wizard if available
        if self.state.generated_code:
            return GeneratedFile(
                path=Path("backend.py"),
                content=self.state.generated_code,
                description="Main backend implementation"
            )
        
        # Otherwise generate from scratch
        class_name = self._to_camel_case(self.state.backend_name) + "Backend"
        
        content = self._generate_backend_code(class_name)
        
        return GeneratedFile(
            path=Path("backend.py"),
            content=content,
            description="Main backend implementation"
        )
    
    def _generate_init_file(self) -> GeneratedFile:
        """Generate the __init__.py file."""
        class_name = self._to_camel_case(self.state.backend_name) + "Backend"
        
        content = f'''"""
{self.state.display_name} Backend Package.

This package provides integration with {self.state.library_name or "a custom simulator"}.
"""

from .backend import {class_name}, get_backend

__all__ = ["{class_name}", "get_backend"]
'''
        
        return GeneratedFile(
            path=Path("__init__.py"),
            content=content,
            description="Package initialization"
        )
    
    def _generate_test_file(self) -> GeneratedFile:
        """Generate the test file."""
        class_name = self._to_camel_case(self.state.backend_name) + "Backend"
        
        content = f'''"""Tests for {self.state.display_name} Backend."""

import pytest
from unittest.mock import MagicMock, patch

# Import the backend
from .backend import {class_name}, get_backend


class Test{class_name}:
    """Test suite for {class_name}."""
    
    @pytest.fixture
    def backend(self):
        """Create a backend instance for testing."""
        return {class_name}()
    
    def test_backend_name(self, backend):
        """Test backend has correct name."""
        assert backend.name == "{self.state.backend_name}"
    
    def test_backend_version(self, backend):
        """Test backend has correct version."""
        assert backend.version == "{self.state.version}"
    
    def test_get_capabilities(self, backend):
        """Test capabilities are properly defined."""
        caps = backend.get_capabilities()
        assert caps.max_qubits == {self.state.max_qubits}
        assert caps.supports_noise == {self.state.supports_noise}
        assert caps.supports_gpu == {self.state.supports_gpu}
    
    def test_get_backend_factory(self):
        """Test factory function returns backend."""
        backend = get_backend()
        assert isinstance(backend, {class_name})
    
    @pytest.mark.skipif(
        not {class_name}.is_available(),
        reason="{self.state.library_name or 'Backend library'} not installed"
    )
    def test_execute_basic_circuit(self, backend):
        """Test basic circuit execution."""
        # TODO: Add test with actual circuit
        pass
    
    def test_gate_mapping(self, backend):
        """Test gate mapping is defined."""
        mapping = backend._get_gate_mapping()
        assert "H" in mapping
        assert "X" in mapping
        assert "CNOT" in mapping


def test_is_available():
    """Test availability check."""
    # This should not raise
    result = {class_name}.is_available()
    assert isinstance(result, bool)
'''
        
        return GeneratedFile(
            path=Path("test_backend.py"),
            content=content,
            description="Unit tests"
        )
    
    def _generate_readme(self) -> GeneratedFile:
        """Generate README documentation."""
        sim_types = ", ".join(self.state.simulator_types) or "state_vector"
        
        features = []
        if self.state.supports_noise:
            features.append("- ✅ Noise simulation support")
        if self.state.supports_gpu:
            features.append("- ✅ GPU acceleration")
        if self.state.supports_batching:
            features.append("- ✅ Batch execution")
        if self.state.supports_parameter_binding:
            features.append("- ✅ Parameter binding")
        if self.state.supports_custom_gates:
            features.append("- ✅ Custom gate definitions")
        
        features_str = "\n".join(features) if features else "- Basic simulation"
        
        content = f'''# {self.state.display_name}

{self.state.description or "A custom quantum computing backend for Proxima."}

## Overview

- **Backend Name**: `{self.state.backend_name}`
- **Version**: {self.state.version}
- **Author**: {self.state.author or "Unknown"}
- **Library**: {self.state.library_name or "Custom"}

## Capabilities

- **Max Qubits**: {self.state.max_qubits}
- **Simulator Types**: {sim_types}

### Features

{features_str}

## Installation

```bash
# Install the required library
pip install {self.state.library_name or self.state.backend_name}
```

## Usage

```python
from proxima.backends.contrib.{self.state.backend_name} import get_backend

# Create backend instance
backend = get_backend()

# Execute a circuit
result = backend.execute(circuit, shots=1024)
```

## Configuration

Register the backend in Proxima:

```yaml
# In proxima config
backends:
  custom:
    - name: {self.state.backend_name}
      module: proxima.backends.contrib.{self.state.backend_name}
```

## Testing

```bash
pytest src/proxima/backends/contrib/{self.state.backend_name}/test_backend.py
```

## License

MIT License
'''
        
        return GeneratedFile(
            path=Path("README.md"),
            content=content,
            description="Documentation"
        )
    
    def _generate_backend_code(self, class_name: str) -> str:
        """Generate the complete backend code."""
        sim_types = self.state.simulator_types or ["state_vector"]
        
        # Build gate mapping
        gate_map_items = []
        for proxima_gate, mapping in self.state.gate_mappings.items():
            gate_map_items.append(f'"{proxima_gate}": "{mapping.backend_gate}"')
        
        if not gate_map_items:
            gate_map_items = [
                '"H": "H"', '"X": "X"', '"Y": "Y"', '"Z": "Z"',
                '"S": "S"', '"T": "T"', '"CNOT": "CNOT"', '"CZ": "CZ"'
            ]
        
        gate_map_str = ",\n            ".join(gate_map_items)
        
        return f'''"""
{self.state.display_name} Backend for Proxima.

{self.state.description or "A custom quantum computing backend implementation."}

Author: {self.state.author or "Unknown"}
Version: {self.state.version}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass, field
import logging

# Proxima imports
from proxima.backends.base import BaseBackend, BackendCapabilities
from proxima.core.circuit import QuantumCircuit
from proxima.core.result import ExecutionResult

{"# Backend library import" if self.state.library_name else ""}
{"import " + self.state.library_name if self.state.library_name else "# TODO: Import your backend library"}

logger = logging.getLogger(__name__)


class {class_name}(BaseBackend):
    """
    {self.state.display_name} backend implementation.
    
    This backend wraps {self.state.library_name or "a custom simulator"} to provide
    quantum circuit execution capabilities.
    
    Supported Features:
    - Max qubits: {self.state.max_qubits}
    - Simulator types: {", ".join(sim_types)}
    - Noise support: {"Yes" if self.state.supports_noise else "No"}
    - GPU support: {"Yes" if self.state.supports_gpu else "No"}
    """
    
    name = "{self.state.backend_name}"
    display_name = "{self.state.display_name}"
    version = "{self.state.version}"
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the backend.
        
        Args:
            **kwargs: Additional backend configuration options
        """
        super().__init__(**kwargs)
        self._simulator = None
        self._options = kwargs
        
        # TODO: Initialize your backend library here
        # Example:
        # self._simulator = your_library.Simulator()
        
        logger.info(f"Initialized {{self.name}} backend")
    
    def get_capabilities(self) -> BackendCapabilities:
        """
        Return the capabilities of this backend.
        
        Returns:
            BackendCapabilities object describing what this backend supports
        """
        return BackendCapabilities(
            max_qubits={self.state.max_qubits},
            simulator_types={sim_types},
            supports_noise={self.state.supports_noise},
            supports_gpu={self.state.supports_gpu},
            supports_batching={self.state.supports_batching},
            supports_parameter_binding={self.state.supports_parameter_binding},
            supports_custom_gates={self.state.supports_custom_gates},
        )
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if this backend is available.
        
        Returns:
            True if the backend can be used, False otherwise
        """
        try:
            {"import " + self.state.library_name if self.state.library_name else "pass  # TODO: Check library availability"}
            return True
        except ImportError:
            logger.warning("{self.state.library_name or 'Backend library'} not installed")
            return False
    
    def execute(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        **kwargs: Any
    ) -> ExecutionResult:
        """
        Execute a quantum circuit.
        
        Args:
            circuit: The quantum circuit to execute
            shots: Number of measurement shots
            **kwargs: Additional execution options
            
        Returns:
            ExecutionResult containing measurement outcomes
        """
        # TODO: Implement circuit execution
        # 1. Convert Proxima circuit to your backend format
        # 2. Execute on your simulator
        # 3. Convert results back to Proxima format
        
        # Example structure:
        # converted = self._convert_circuit(circuit)
        # raw_result = self._simulator.run(converted, shots=shots)
        # return self._convert_result(raw_result)
        
        raise NotImplementedError("Execute method not yet implemented")
    
    def _convert_circuit(self, circuit: QuantumCircuit) -> Any:
        """Convert Proxima circuit to backend format."""
        # TODO: Implement circuit conversion
        gate_mapping = self._get_gate_mapping()
        
        # Example conversion logic:
        # backend_circuit = self._library.Circuit(circuit.num_qubits)
        # for op in circuit.operations:
        #     backend_gate = gate_mapping.get(op.name, op.name)
        #     backend_circuit.add_gate(backend_gate, op.qubits, op.params)
        # return backend_circuit
        pass
    
    def _convert_result(self, raw_result: Any) -> ExecutionResult:
        """Convert backend result to Proxima format."""
        # TODO: Implement result conversion
        # return ExecutionResult(
        #     counts=raw_result.get_counts(),
        #     statevector=raw_result.get_statevector(),
        #     ...
        # )
        pass
    
    def _get_gate_mapping(self) -> Dict[str, str]:
        """
        Get the mapping from Proxima gates to backend gates.
        
        Returns:
            Dictionary mapping Proxima gate names to backend gate names
        """
        return {{
            {gate_map_str}
        }}


def get_backend(**kwargs: Any) -> {class_name}:
    """
    Factory function to create the backend instance.
    
    Args:
        **kwargs: Backend configuration options
        
    Returns:
        Configured backend instance
    """
    return {class_name}(**kwargs)
'''
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
