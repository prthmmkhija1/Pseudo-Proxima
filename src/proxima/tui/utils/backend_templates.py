"""Backend Template Engine.

Provides Jinja2-based templates for generating backend code files.
Supports multiple backend types: Python Library, Command Line, API Server, Custom.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from string import Template
import textwrap


class SimpleTemplate:
    """Simple string template with variable substitution."""
    
    def __init__(self, template: str):
        self.template = template
    
    def render(self, **kwargs: Any) -> str:
        """Render the template with given variables."""
        result = self.template
        
        # Handle list rendering
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = value
        
        # Simple variable substitution
        for key, value in kwargs.items():
            if isinstance(value, bool):
                value = str(value)
            elif isinstance(value, list):
                # Handle list items in templates
                list_placeholder = f"{{{{ {key} }}}}"
                if list_placeholder in result:
                    result = result.replace(list_placeholder, str(value))
                continue
            
            # Replace {{ key }} style placeholders
            result = result.replace(f"{{{{ {key} }}}}", str(value) if value else "")
            # Replace {% if key %}...{% endif %} blocks
            # (simplified - just keep content if value is truthy)
        
        return result


class BackendTemplateEngine:
    """Template engine for generating backend code files.
    
    Provides templates for:
    - Python Library backends
    - Command Line backends
    - API Server backends
    - Custom backends
    - Normalizers
    - Init files
    - README files
    - Test files
    """
    
    def get_adapter_template(self, backend_type: str) -> SimpleTemplate:
        """Get adapter template based on backend type.
        
        Args:
            backend_type: One of 'python_library', 'command_line', 'api_server', 'custom'
            
        Returns:
            Template for the specified backend type
        """
        templates = {
            "python_library": PYTHON_LIBRARY_ADAPTER_TEMPLATE,
            "command_line": COMMAND_LINE_ADAPTER_TEMPLATE,
            "api_server": API_SERVER_ADAPTER_TEMPLATE,
            "custom": CUSTOM_ADAPTER_TEMPLATE,
        }
        
        template_str = templates.get(backend_type, CUSTOM_ADAPTER_TEMPLATE)
        return SimpleTemplate(template_str)
    
    def get_normalizer_template(self) -> SimpleTemplate:
        """Get normalizer template."""
        return SimpleTemplate(NORMALIZER_TEMPLATE)
    
    def get_init_template(self) -> SimpleTemplate:
        """Get __init__.py template."""
        return SimpleTemplate(INIT_TEMPLATE)
    
    def get_readme_template(self) -> SimpleTemplate:
        """Get README.md template."""
        return SimpleTemplate(README_TEMPLATE)
    
    def get_test_template(self) -> SimpleTemplate:
        """Get test file template."""
        return SimpleTemplate(TEST_TEMPLATE)
    
    def render_adapter(
        self,
        backend_type: str,
        backend_name: str,
        display_name: str,
        version: str,
        description: str = "",
        author: str = "",
        library_name: str = "",
        simulator_types: list = None,
        max_qubits: int = 20,
        supports_noise: bool = False,
        supports_gpu: bool = False,
        supports_batching: bool = False,
        custom_init: str = "",
        custom_execute: str = "",
        gate_mappings: dict = None,
    ) -> str:
        """Render a complete adapter file.
        
        Args:
            backend_type: Type of backend
            backend_name: Internal backend name (snake_case)
            display_name: Display name for UI
            version: Version string
            description: Backend description
            author: Author name
            library_name: Python library to import
            simulator_types: List of supported simulator types
            max_qubits: Maximum qubit count
            supports_noise: Whether noise is supported
            supports_gpu: Whether GPU is supported
            supports_batching: Whether batching is supported
            custom_init: Custom initialization code
            custom_execute: Custom execution code
            gate_mappings: Gate name mappings
            
        Returns:
            Rendered adapter code
        """
        simulator_types = simulator_types or ["state_vector"]
        gate_mappings = gate_mappings or {}
        
        # Generate class name
        class_name = self._to_class_name(backend_name) + "Adapter"
        
        # Generate simulator type imports
        sim_type_lines = []
        for sim_type in simulator_types:
            sim_type_lines.append(f"                SimulatorType.{sim_type.upper()},")
        sim_types_str = "\n".join(sim_type_lines)
        
        # Generate gate mappings
        gate_map_lines = []
        for proxima_gate, backend_gate in gate_mappings.items():
            gate_map_lines.append(f'            "{proxima_gate}": "{backend_gate}",')
        gate_map_str = "\n".join(gate_map_lines) if gate_map_lines else '            # Default mappings'
        
        # Build the adapter
        code = f'''"""{{ display_name }} Backend Adapter.

Auto-generated by Proxima Backend Wizard.
Backend Type: {backend_type.replace("_", " ").title()}
Version: {version}
{f"Author: {author}" if author else ""}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from proxima.backends.base import (
    BaseBackendAdapter,
    Capabilities,
    SimulatorType,
    ValidationResult,
    ResourceEstimate,
    ExecutionResult,
    ResultType,
)

logger = logging.getLogger(__name__)


class {class_name}(BaseBackendAdapter):
    """Adapter for {display_name}.
    
    {description or "Custom quantum computing backend."}
    """
    
    name = "{backend_name}"
    version = "{version}"
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the {display_name} adapter."""
        self.config = config or {{}}
        self._simulator = None
        self._initialized = False
        logger.info(f"Created {{self.name}} adapter")
    
    def get_name(self) -> str:
        """Return backend identifier."""
        return self.name
    
    def get_version(self) -> str:
        """Return backend version string."""
        return self.version
    
    def get_capabilities(self) -> Capabilities:
        """Return supported capabilities."""
        return Capabilities(
            simulator_types=[
{sim_types_str}
            ],
            max_qubits={max_qubits},
            supports_noise={supports_noise},
            supports_gpu={supports_gpu},
            supports_batching={supports_batching},
        )
    
    def initialize(self) -> None:
        """Initialize the backend."""
        if self._initialized:
            return
        
        {self._generate_init_code(backend_type, library_name, custom_init)}
        
        self._initialized = True
        logger.info(f"Initialized {{self.name}} backend")
    
    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate circuit compatibility with the backend."""
        if not circuit:
            return ValidationResult(
                valid=False,
                message="Circuit is None or empty"
            )
        
        if hasattr(circuit, 'num_qubits'):
            if circuit.num_qubits > {max_qubits}:
                return ValidationResult(
                    valid=False,
                    message=f"Circuit has {{circuit.num_qubits}} qubits, maximum is {max_qubits}"
                )
        
        return ValidationResult(valid=True)
    
    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate resources for execution."""
        qubit_count = getattr(circuit, 'num_qubits', 0)
        
        # Memory estimate: 2^n * 16 bytes for state vector
        memory_mb = (2 ** qubit_count * 16) / (1024 * 1024)
        
        # Time estimate
        gate_count = getattr(circuit, 'gate_count', 0)
        time_ms = gate_count * 0.1
        
        return ResourceEstimate(
            memory_mb=memory_mb,
            time_ms=time_ms,
        )
    
    def execute(
        self,
        circuit: Any,
        options: Dict[str, Any] = None
    ) -> ExecutionResult:
        """Execute a circuit and return results."""
        if not self._initialized:
            self.initialize()
        
        options = options or {{}}
        shots = options.get('shots', 1024)
        
        import time
        start_time = time.time()
        
        {self._generate_execute_code(backend_type, library_name, custom_execute)}
        
        execution_time = (time.time() - start_time) * 1000
        
        return ExecutionResult(
            backend=self.name,
            simulator_type=SimulatorType.STATE_VECTOR,
            execution_time_ms=execution_time,
            qubit_count=getattr(circuit, 'num_qubits', 0),
            shot_count=shots,
            result_type=ResultType.COUNTS,
            data={{"counts": counts}},
            raw_result=raw_result,
        )
    
    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        """Return whether the simulator type is supported."""
        supported = [
{sim_types_str}
        ]
        return sim_type in supported
    
    def is_available(self) -> bool:
        """Return whether the backend is available on this system."""
        {self._generate_availability_check(backend_type, library_name)}
    
    def _get_gate_mapping(self) -> Dict[str, str]:
        """Get gate name mappings from Proxima to backend format."""
        return {{
{gate_map_str}
        }}
    
    def _convert_circuit(self, circuit: Any) -> Any:
        """Convert Proxima circuit to backend format."""
        # TODO: Implement circuit conversion using gate mappings
        return circuit
    
    def cleanup(self) -> None:
        """Clean up backend resources."""
        if self._simulator:
            if hasattr(self._simulator, 'close'):
                self._simulator.close()
            self._simulator = None
        self._initialized = False
        logger.info(f"Cleaned up {{self.name}} backend")
'''
        return code
    
    def _to_class_name(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
    
    def _generate_init_code(self, backend_type: str, library_name: str, custom_init: str) -> str:
        """Generate initialization code based on backend type."""
        if custom_init:
            return custom_init
        
        if backend_type == "python_library" and library_name:
            return f'''try:
            import {library_name}
            self._simulator = {library_name}.Simulator()
        except ImportError as e:
            raise RuntimeError(
                f"{library_name} not installed. Install with: pip install {library_name}"
            ) from e'''
        
        elif backend_type == "command_line":
            return '''import subprocess
        # Verify command-line tool is available
        try:
            subprocess.run(["your_tool", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("Command-line tool not available") from e'''
        
        elif backend_type == "api_server":
            return '''import httpx
        # Verify API server is reachable
        self._client = httpx.Client(base_url=self.config.get("api_url", "http://localhost:8000"))
        try:
            self._client.get("/health")
        except httpx.RequestError as e:
            raise RuntimeError("API server not reachable") from e'''
        
        else:
            return "# Custom initialization - implement as needed\n        pass"
    
    def _generate_execute_code(self, backend_type: str, library_name: str, custom_execute: str) -> str:
        """Generate execution code based on backend type."""
        if custom_execute:
            return custom_execute
        
        if backend_type == "python_library" and library_name:
            return '''# Convert circuit to backend format
        native_circuit = self._convert_circuit(circuit)
        
        # Execute
        raw_result = self._simulator.run(native_circuit, shots=shots)
        counts = raw_result.get('counts', {})'''
        
        elif backend_type == "command_line":
            return '''import subprocess
        import json
        import tempfile
        
        # Save circuit to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(circuit.to_json())
            circuit_file = f.name
        
        # Execute command-line tool
        result = subprocess.run(
            ["your_tool", "run", circuit_file, "--shots", str(shots)],
            capture_output=True,
            text=True
        )
        
        raw_result = json.loads(result.stdout)
        counts = raw_result.get('counts', {})'''
        
        elif backend_type == "api_server":
            return '''# Send circuit to API
        response = self._client.post(
            "/execute",
            json={
                "circuit": circuit.to_dict(),
                "shots": shots
            }
        )
        response.raise_for_status()
        
        raw_result = response.json()
        counts = raw_result.get('counts', {})'''
        
        else:
            return '''# Custom execution - implement as needed
        raw_result = {}
        counts = {}'''
    
    def _generate_availability_check(self, backend_type: str, library_name: str) -> str:
        """Generate availability check code."""
        if backend_type == "python_library" and library_name:
            return f'''try:
            import {library_name}
            return True
        except ImportError:
            return False'''
        
        elif backend_type == "command_line":
            return '''try:
            import subprocess
            subprocess.run(["your_tool", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False'''
        
        elif backend_type == "api_server":
            return '''try:
            import httpx
            response = httpx.get(self.config.get("api_url", "http://localhost:8000") + "/health")
            return response.status_code == 200
        except Exception:
            return False'''
        
        else:
            return "return True"


# ============================================================================
# TEMPLATE DEFINITIONS
# ============================================================================

PYTHON_LIBRARY_ADAPTER_TEMPLATE = '''"""{{ display_name }} Backend Adapter.

Auto-generated by Proxima Backend Wizard.
Backend Type: Python Library
Version: {{ version }}
"""

from typing import Any, Dict
from proxima.backends.base import BaseBackendAdapter


class {{ adapter_class }}(BaseBackendAdapter):
    """Adapter for {{ display_name }}."""
    
    name = "{{ backend_name }}"
    version = "{{ version }}"
    
    # Implementation generated by wizard
'''


COMMAND_LINE_ADAPTER_TEMPLATE = '''"""{{ display_name }} Backend Adapter.

Auto-generated by Proxima Backend Wizard.
Backend Type: Command Line Tool
Version: {{ version }}
"""

import subprocess
from typing import Any, Dict
from proxima.backends.base import BaseBackendAdapter


class {{ adapter_class }}(BaseBackendAdapter):
    """Command-line adapter for {{ display_name }}."""
    
    name = "{{ backend_name }}"
    version = "{{ version }}"
    
    # Uses subprocess to call external tool
'''


API_SERVER_ADAPTER_TEMPLATE = '''"""{{ display_name }} Backend Adapter.

Auto-generated by Proxima Backend Wizard.
Backend Type: API Server
Version: {{ version }}
"""

import httpx
from typing import Any, Dict
from proxima.backends.base import BaseBackendAdapter


class {{ adapter_class }}(BaseBackendAdapter):
    """API adapter for {{ display_name }}."""
    
    name = "{{ backend_name }}"
    version = "{{ version }}"
    
    # Uses HTTP client for API calls
'''


CUSTOM_ADAPTER_TEMPLATE = '''"""{{ display_name }} Backend Adapter.

Auto-generated by Proxima Backend Wizard.
Backend Type: Custom
Version: {{ version }}
"""

from typing import Any, Dict
from proxima.backends.base import BaseBackendAdapter


class {{ adapter_class }}(BaseBackendAdapter):
    """Custom adapter for {{ display_name }}."""
    
    name = "{{ backend_name }}"
    version = "{{ version }}"
    
    # Customize implementation as needed
'''


NORMALIZER_TEMPLATE = '''"""Result normalizer for {{ display_name }}.

Auto-generated by Proxima Backend Wizard.
"""

from typing import Dict, Any
from proxima.core.result import ExecutionResult


class {{ normalizer_class }}:
    """Normalize results from {{ display_name }}."""
    
    def normalize(self, raw_result: Any) -> ExecutionResult:
        """
        Convert backend-specific result to Proxima format.
        
        Args:
            raw_result: Raw result from {{ display_name }}
        
        Returns:
            Normalized ExecutionResult
        """
        counts = {}
        
        if isinstance(raw_result, dict):
            counts = raw_result.get('counts', {})
        elif hasattr(raw_result, 'measurements'):
            counts = raw_result.measurements
        
        # Normalize state strings
        normalized_counts = {}
        for state, count in counts.items():
            normalized_state = self._normalize_state(state)
            normalized_counts[normalized_state] = count
        
        return {
            'counts': normalized_counts,
            'shots': sum(normalized_counts.values()),
        }
    
    def _normalize_state(self, state: str) -> str:
        """Normalize state string representation."""
        state = str(state).strip("|<> ")
        
        if state.isdigit() and set(state).issubset({'0', '1'}):
            return state
        
        try:
            return format(int(state, 2), 'b')
        except ValueError:
            return state
'''


INIT_TEMPLATE = '''"""{{ backend_name }} backend module.

Auto-generated by Proxima Backend Wizard.
"""

from .adapter import {{ adapter_class }}
from .normalizer import {{ normalizer_class }}

__all__ = ["{{ adapter_class }}", "{{ normalizer_class }}"]
'''


README_TEMPLATE = '''# {{ display_name }}

{{ description }}

## Installation

```bash
pip install {{ library_name }}
```

## Usage

```python
from proxima.backends.{{ backend_name }} import {{ adapter_class }}

adapter = {{ adapter_class }}()
adapter.initialize()

result = adapter.execute(circuit, options={'shots': 1024})
```

## Configuration

- `shots`: Number of measurement shots (default: 1024)

## Metadata

- **Version**: {{ version }}
- **Author**: {{ author }}
- **Auto-generated**: Yes
- **Generator**: Proxima Backend Wizard
'''


TEST_TEMPLATE = '''"""Tests for {{ display_name }} backend.

Auto-generated by Proxima Backend Wizard.
"""

import pytest
from proxima.backends.{{ backend_name }} import {{ adapter_class }}


@pytest.fixture
def adapter():
    """Create {{ display_name }} adapter instance."""
    adapter = {{ adapter_class }}()
    adapter.initialize()
    yield adapter
    adapter.cleanup()


def test_adapter_initialization(adapter):
    """Test adapter initializes correctly."""
    assert adapter.get_name() == "{{ backend_name }}"


def test_get_capabilities(adapter):
    """Test capabilities reporting."""
    caps = adapter.get_capabilities()
    assert caps.max_qubits > 0


def test_is_available():
    """Test availability check."""
    result = {{ adapter_class }}.is_available()
    assert isinstance(result, bool)
'''
