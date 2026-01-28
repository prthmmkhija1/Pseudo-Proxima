"""Backend Package Generator Integration.

High-level interface for generating complete backend packages.
Integrates template library, renderer, code generator, and validator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from .template_library import TemplateLibrary, TemplateType
from .template_renderer import (
    TemplateRenderer,
    BackendVariableBuilder,
    MultiFileRenderer,
    RenderResult,
)
from .code_generator import FullBackendGenerator, GateMappingGenerator
from .code_validator import CodeValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class GeneratedFile:
    """Represents a generated file."""
    path: str
    content: str
    validation: Optional[ValidationResult] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if file passed validation."""
        return self.validation is None or self.validation.valid


@dataclass
class GenerationResult:
    """Result of backend package generation."""
    success: bool
    files: Dict[str, GeneratedFile]
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]
    
    @property
    def file_count(self) -> int:
        """Get number of files generated."""
        return len(self.files)
    
    @property
    def valid_count(self) -> int:
        """Get number of valid files."""
        return sum(1 for f in self.files.values() if f.is_valid)
    
    def get_file_contents(self) -> Dict[str, str]:
        """Get dictionary of path -> content."""
        return {path: f.content for path, f in self.files.items()}


class BackendPackageGenerator:
    """Generate complete backend packages.
    
    This is the main entry point for Phase 3 code generation.
    It coordinates template rendering, code generation, and validation.
    
    Usage:
        generator = BackendPackageGenerator()
        result = generator.generate(
            backend_name="my_backend",
            display_name="My Backend",
            backend_type="python_library",
            ...
        )
        
        if result.success:
            for path, file in result.files.items():
                print(f"Generated: {path}")
    """
    
    def __init__(self, validate: bool = True, strict: bool = False):
        """Initialize generator.
        
        Args:
            validate: Whether to validate generated code
            strict: Whether to treat warnings as errors
        """
        self.template_library = TemplateLibrary()
        self.renderer = MultiFileRenderer()
        self.code_generator = FullBackendGenerator()
        self.validator = CodeValidator(strict=strict)
        self.validate = validate
    
    def generate(
        self,
        backend_name: str,
        display_name: str = None,
        backend_type: str = "custom",
        version: str = "1.0.0",
        author: str = "",
        description: str = "",
        library_name: str = "",
        max_qubits: int = 20,
        gate_mappings: Dict[str, str] = None,
        supports_noise: bool = False,
        supports_gpu: bool = False,
        supports_batching: bool = False,
        output_dir: str = None,
        **extra
    ) -> GenerationResult:
        """Generate complete backend package.
        
        Args:
            backend_name: Internal name (snake_case)
            display_name: Display name for UI
            backend_type: Backend type (python_library, command_line, api_server, custom)
            version: Version string
            author: Author name
            description: Backend description
            library_name: Python library to import
            max_qubits: Maximum qubit count
            gate_mappings: Gate name mappings
            supports_noise: Noise simulation support
            supports_gpu: GPU acceleration support
            supports_batching: Batching support
            output_dir: Optional output directory
            **extra: Additional template variables
            
        Returns:
            GenerationResult with generated files
        """
        files = {}
        errors = []
        warnings = []
        stats = {
            "backend_name": backend_name,
            "backend_type": backend_type,
            "files_generated": 0,
            "files_valid": 0,
            "total_lines": 0,
        }
        
        try:
            # Generate files using code generator
            generated = self.code_generator.generate_complete_backend(
                backend_name=backend_name,
                display_name=display_name or backend_name.replace("_", " ").title(),
                backend_type=backend_type,
                version=version,
                author=author,
                description=description,
                library_name=library_name,
                max_qubits=max_qubits,
                gate_mappings=gate_mappings,
                supports_noise=supports_noise,
                supports_gpu=supports_gpu,
            )
            
            # Also generate from template library for additional files
            additional_templates = self._get_additional_templates(backend_type)
            additional_vars = self._build_template_vars(
                backend_name=backend_name,
                display_name=display_name,
                backend_type=backend_type,
                version=version,
                author=author,
                description=description,
                library_name=library_name,
                max_qubits=max_qubits,
                supports_noise=supports_noise,
                supports_gpu=supports_gpu,
                **extra
            )
            
            # Merge generated files
            for filename, content in generated.items():
                validation = None
                
                if self.validate and filename.endswith('.py'):
                    validation = self.validator.validate(content, filename=filename)
                    
                    if not validation.valid:
                        for issue in validation.errors:
                            errors.append(f"{filename}: {issue.message}")
                    
                    for issue in validation.warnings:
                        warnings.append(f"{filename}: {issue.message}")
                
                files[filename] = GeneratedFile(
                    path=filename,
                    content=content,
                    validation=validation
                )
                
                stats["files_generated"] += 1
                stats["total_lines"] += content.count("\n") + 1
                
                if validation is None or validation.valid:
                    stats["files_valid"] += 1
            
            # Generate README
            readme_content = self._generate_readme(
                backend_name=backend_name,
                display_name=display_name or backend_name.replace("_", " ").title(),
                backend_type=backend_type,
                version=version,
                author=author,
                description=description,
                library_name=library_name,
                max_qubits=max_qubits,
                supports_noise=supports_noise,
                supports_gpu=supports_gpu,
            )
            
            files[f"{backend_name}/README.md"] = GeneratedFile(
                path=f"{backend_name}/README.md",
                content=readme_content,
                validation=None
            )
            stats["files_generated"] += 1
            stats["files_valid"] += 1
            
            # Generate test file
            test_content = self._generate_tests(
                backend_name=backend_name,
                display_name=display_name or backend_name.replace("_", " ").title(),
                version=version,
                max_qubits=max_qubits,
            )
            
            test_path = f"tests/backends/test_{backend_name}.py"
            test_validation = None
            
            if self.validate:
                test_validation = self.validator.validate(test_content, filename=test_path)
            
            files[test_path] = GeneratedFile(
                path=test_path,
                content=test_content,
                validation=test_validation
            )
            stats["files_generated"] += 1
            if test_validation is None or test_validation.valid:
                stats["files_valid"] += 1
            
            success = len(errors) == 0
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            errors.append(f"Generation error: {str(e)}")
            success = False
        
        return GenerationResult(
            success=success,
            files=files,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def _get_additional_templates(self, backend_type: str) -> Dict[str, str]:
        """Get additional templates based on backend type."""
        templates = {}
        
        # Config template
        templates["config.yaml"] = self.template_library.get_template(
            TemplateType.CONFIG, backend_type
        )
        
        return templates
    
    def _build_template_vars(self, **kwargs) -> Dict[str, Any]:
        """Build template variables from kwargs."""
        builder = BackendVariableBuilder()
        return builder.build(**kwargs)
    
    def _generate_readme(self, **kwargs) -> str:
        """Generate README.md content."""
        class_name = self._to_class_name(kwargs["backend_name"])
        
        features = []
        if kwargs.get("supports_noise"):
            features.append("- ✅ Noise simulation support")
        if kwargs.get("supports_gpu"):
            features.append("- ✅ GPU acceleration support")
        if not features:
            features.append("- State vector simulation")
        
        install_cmd = f"pip install {kwargs['library_name']}" if kwargs.get("library_name") else "# See installation instructions"
        
        return f'''# {kwargs["display_name"]}

{kwargs.get("description", "Custom quantum computing backend.")}

## Overview

| Property | Value |
|----------|-------|
| **Backend Name** | `{kwargs["backend_name"]}` |
| **Version** | {kwargs["version"]} |
| **Type** | {kwargs["backend_type"].replace("_", " ").title()} |
| **Max Qubits** | {kwargs["max_qubits"]} |

## Features

{chr(10).join(features)}

## Installation

```bash
{install_cmd}
```

## Quick Start

```python
from proxima.backends.{kwargs["backend_name"]} import {class_name}Adapter

# Create and initialize adapter
adapter = {class_name}Adapter()
adapter.initialize()

# Check capabilities
caps = adapter.get_capabilities()
print(f"Max qubits: {{caps.max_qubits}}")

# Execute a circuit
result = adapter.execute(circuit, options={{'shots': 1024}})
print(f"Counts: {{result.data['counts']}}")

# Clean up
adapter.cleanup()
```

## Configuration

```python
adapter = {class_name}Adapter(config={{
    'shots': 1024,
    # Add backend-specific options
}})
```

## API Reference

### {class_name}Adapter

Main adapter class for {kwargs["display_name"]}.

**Methods:**

| Method | Description |
|--------|-------------|
| `initialize()` | Initialize the backend |
| `execute(circuit, options)` | Execute a quantum circuit |
| `validate_circuit(circuit)` | Validate circuit compatibility |
| `get_capabilities()` | Get backend capabilities |
| `estimate_resources(circuit)` | Estimate execution resources |
| `cleanup()` | Release resources |

## Metadata

- **Generated by:** Proxima Backend Wizard (Phase 3)
- **Version:** {kwargs["version"]}
{f"- **Author:** {kwargs['author']}" if kwargs.get("author") else ""}

## License

Same as Proxima project.
'''
    
    def _generate_tests(self, **kwargs) -> str:
        """Generate test file content."""
        class_name = self._to_class_name(kwargs["backend_name"])
        
        return f'''"""Tests for {kwargs["display_name"]} backend.

Auto-generated by Proxima Backend Wizard (Phase 3).
"""

import pytest
from unittest.mock import MagicMock, patch

from proxima.backends.{kwargs["backend_name"]} import {class_name}Adapter
from proxima.backends.{kwargs["backend_name"]}.errors import (
    BackendError,
    InitializationError,
    ExecutionError,
    ValidationError,
)


class Test{class_name}Adapter:
    """Test suite for {class_name}Adapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        adapter = {class_name}Adapter()
        yield adapter
        adapter.cleanup()
    
    def test_get_name(self, adapter):
        """Test backend name."""
        assert adapter.get_name() == "{kwargs["backend_name"]}"
    
    def test_get_version(self, adapter):
        """Test backend version."""
        assert adapter.get_version() == "{kwargs["version"]}"
    
    def test_get_capabilities(self, adapter):
        """Test capabilities."""
        caps = adapter.get_capabilities()
        assert caps.max_qubits == {kwargs["max_qubits"]}
        assert len(caps.simulator_types) > 0
    
    def test_validate_empty_circuit(self, adapter):
        """Test validation rejects empty circuit."""
        result = adapter.validate_circuit(None)
        assert not result.valid
        assert "empty" in result.message.lower() or "none" in result.message.lower()
    
    def test_validate_valid_circuit(self, adapter):
        """Test validation accepts valid circuit."""
        circuit = MagicMock()
        circuit.num_qubits = 2
        circuit.gates = []
        
        result = adapter.validate_circuit(circuit)
        assert result.valid
    
    def test_validate_too_many_qubits(self, adapter):
        """Test validation rejects too many qubits."""
        circuit = MagicMock()
        circuit.num_qubits = {kwargs["max_qubits"]} + 10
        circuit.gates = []
        
        result = adapter.validate_circuit(circuit)
        assert not result.valid
        assert "qubit" in result.message.lower()
    
    def test_estimate_resources(self, adapter):
        """Test resource estimation."""
        circuit = MagicMock()
        circuit.num_qubits = 4
        circuit.gates = [MagicMock() for _ in range(10)]
        
        estimate = adapter.estimate_resources(circuit)
        assert estimate.memory_mb > 0
        assert estimate.time_ms >= 0
    
    def test_is_available(self):
        """Test availability check."""
        adapter = {class_name}Adapter()
        result = adapter.is_available()
        assert isinstance(result, bool)
    
    def test_cleanup(self, adapter):
        """Test cleanup doesn't raise."""
        adapter.cleanup()
        # Should be able to call multiple times
        adapter.cleanup()


class Test{class_name}Normalizer:
    """Test result normalizer."""
    
    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance."""
        from proxima.backends.{kwargs["backend_name"]} import {class_name}Normalizer
        return {class_name}Normalizer()
    
    def test_normalize_dict(self, normalizer):
        """Test normalizing dict result."""
        result = normalizer.normalize({{'counts': {{'00': 50, '11': 50}}}})
        
        assert 'counts' in result
        assert result['shots'] == 100
        assert result['success'] is True
    
    def test_normalize_empty(self, normalizer):
        """Test normalizing empty result."""
        result = normalizer.normalize({{}})
        
        assert 'counts' in result
        assert result['shots'] == 0


class TestCircuitConverter:
    """Test circuit converter."""
    
    @pytest.fixture
    def converter(self):
        """Create converter instance."""
        from proxima.backends.{kwargs["backend_name"]} import CircuitConverter
        return CircuitConverter()
    
    def test_to_json(self, converter):
        """Test JSON conversion."""
        circuit = MagicMock()
        circuit.num_qubits = 2
        circuit.gates = []
        
        result = converter.to_json(circuit)
        
        assert result['num_qubits'] == 2
        assert 'gates' in result
    
    def test_to_native(self, converter):
        """Test native format conversion."""
        circuit = MagicMock()
        circuit.num_qubits = 3
        circuit.gates = []
        
        result = converter.to_native(circuit)
        
        assert result['num_qubits'] == 3
'''
    
    def _to_class_name(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
    
    def write_to_disk(
        self,
        result: GenerationResult,
        base_path: str = "src/proxima/backends"
    ) -> Tuple[bool, List[str]]:
        """Write generated files to disk.
        
        Args:
            result: GenerationResult to write
            base_path: Base directory for backend files
            
        Returns:
            Tuple of (success, written_files)
        """
        written = []
        
        try:
            base = Path(base_path)
            
            for path, file in result.files.items():
                # Determine full path
                if path.startswith("tests/"):
                    full_path = Path(path)
                else:
                    full_path = base / path
                
                # Create directories
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file
                full_path.write_text(file.content, encoding='utf-8')
                written.append(str(full_path))
                
                logger.info(f"Written: {full_path}")
            
            return True, written
            
        except Exception as e:
            logger.error(f"Failed to write files: {e}", exc_info=True)
            return False, written


def generate_backend(
    backend_name: str,
    backend_type: str = "custom",
    **kwargs
) -> GenerationResult:
    """Convenience function to generate a backend.
    
    Args:
        backend_name: Internal name (snake_case)
        backend_type: Backend type
        **kwargs: Additional options
        
    Returns:
        GenerationResult
    """
    generator = BackendPackageGenerator()
    return generator.generate(
        backend_name=backend_name,
        backend_type=backend_type,
        **kwargs
    )
