"""Step 5: Code Template Generation.

Generate the initial backend code template based on
collected information and let user review/edit.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, Center, ScrollableContainer
from textual.widgets import Static, Button, TextArea, RadioSet, RadioButton, LoadingIndicator
from textual.screen import ModalScreen

from .wizard_state import BackendWizardState


class CodeTemplateStepScreen(ModalScreen[dict]):
    """
    Step 5: Code template generation and preview screen.
    
    Generates a Python class template for the custom backend
    based on collected configuration and allows user to review.
    """
    
    DEFAULT_CSS = """
    CodeTemplateStepScreen {
        align: center middle;
    }
    
    CodeTemplateStepScreen .wizard-container {
        width: 95;
        height: auto;
        max-height: 95%;
        border: double $primary;
        background: $surface;
        padding: 1 2;
    }
    
    CodeTemplateStepScreen .wizard-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 0;
        border-bottom: solid $primary-darken-2;
    }
    
    CodeTemplateStepScreen .form-container {
        height: auto;
        max-height: 70%;
        padding: 1;
    }
    
    CodeTemplateStepScreen .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }
    
    CodeTemplateStepScreen .field-hint {
        color: $text-muted;
        margin: 0 0 1 0;
    }
    
    CodeTemplateStepScreen .code-area {
        height: 25;
        margin: 1 0;
        border: solid $primary-darken-3;
    }
    
    CodeTemplateStepScreen .template-options {
        margin: 1 0;
        padding: 1;
        background: $surface-darken-1;
    }
    
    CodeTemplateStepScreen .option-row {
        height: auto;
        margin: 0 0 1 0;
    }
    
    CodeTemplateStepScreen .info-box {
        background: $primary-darken-3;
        padding: 1;
        margin: 1 0;
        border: solid $primary;
    }
    
    CodeTemplateStepScreen .success-box {
        background: $success-darken-3;
        padding: 1;
        margin: 1 0;
        border: solid $success;
        color: $success;
    }
    
    CodeTemplateStepScreen .progress-section {
        margin: 1 0;
        padding: 1 0;
        border-top: solid $primary-darken-3;
    }
    
    CodeTemplateStepScreen .progress-text {
        color: $text-muted;
    }
    
    CodeTemplateStepScreen .progress-bar {
        color: $primary;
    }
    
    CodeTemplateStepScreen .button-container {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
        padding: 1 0;
        border-top: solid $primary-darken-2;
    }
    
    CodeTemplateStepScreen .nav-button {
        margin: 0 1;
        min-width: 14;
    }
    
    CodeTemplateStepScreen .loading-container {
        width: 100%;
        height: 10;
        align: center middle;
    }
    
    CodeTemplateStepScreen .loading-text {
        text-align: center;
        color: $primary;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+s", "save_template", "Save Template"),
    ]
    
    def __init__(self, state: BackendWizardState):
        """
        Initialize the code template screen.
        
        Args:
            state: The shared wizard state
        """
        super().__init__()
        self.state = state
        self.generated = False
    
    def compose(self) -> ComposeResult:
        """Compose the code template screen layout."""
        with Center():
            with Vertical(classes="wizard-container"):
                # Title
                yield Static(
                    "📝 Add Custom Backend - Code Template",
                    classes="wizard-title"
                )
                
                yield Static(
                    f"Generating code template for: {self.state.display_name}",
                    classes="field-hint"
                )
                
                with ScrollableContainer(classes="form-container"):
                    # Template options
                    yield Static(
                        "Template Options:",
                        classes="section-title"
                    )
                    
                    with Vertical(classes="template-options"):
                        with Horizontal(classes="option-row"):
                            with RadioSet(id="template_style"):
                                yield RadioButton(
                                    "Full Template (recommended)",
                                    value=True,
                                    id="style_full"
                                )
                                yield RadioButton(
                                    "Minimal Template",
                                    id="style_minimal"
                                )
                                yield RadioButton(
                                    "Advanced Template",
                                    id="style_advanced"
                                )
                    
                    # Code preview
                    yield Static(
                        "Generated Code Preview:",
                        classes="section-title"
                    )
                    
                    yield Static(
                        "You can edit this code directly. Changes will be saved.",
                        classes="field-hint"
                    )
                    
                    yield TextArea(
                        id="code_area",
                        classes="code-area",
                        language="python",
                        show_line_numbers=True
                    )
                    
                    # Success message
                    yield Static(
                        "✅ Template generated! Review and modify as needed.\n"
                        "   The code includes all required methods based on your configuration.",
                        classes="success-box",
                        id="success_msg"
                    )
                    
                    # Info about next steps
                    yield Static(
                        "ℹ️ Next Steps:\n"
                        "  • Review the generated code for accuracy\n"
                        "  • Add your implementation in the TODO sections\n"
                        "  • The testing step will validate your backend\n"
                        "  • You can always edit the code after creation",
                        classes="info-box"
                    )
                
                # Progress indicator
                with Vertical(classes="progress-section"):
                    yield Static(
                        "Progress: Step 5 of 7",
                        classes="progress-text"
                    )
                    yield Static(
                        "██████░ 71%",
                        classes="progress-bar"
                    )
                
                # Navigation buttons
                with Horizontal(classes="button-container"):
                    yield Button(
                        "← Back",
                        id="btn_back",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Regenerate",
                        id="btn_regenerate",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Cancel",
                        id="btn_cancel",
                        variant="default",
                        classes="nav-button"
                    )
                    yield Button(
                        "Next: Testing →",
                        id="btn_next",
                        variant="primary",
                        classes="nav-button"
                    )
    
    def on_mount(self) -> None:
        """Generate the code template on mount."""
        # Hide success message initially
        self.query_one("#success_msg", Static).display = False
        
        # Generate initial code
        self._generate_code()
    
    def _generate_code(self) -> None:
        """Generate the backend code template."""
        code = self._build_code_template()
        
        # Update the text area
        code_area = self.query_one("#code_area", TextArea)
        code_area.text = code
        
        # Store in state
        self.state.generated_code = code
        
        # Show success message
        self.query_one("#success_msg", Static).display = True
        self.generated = True
    
    def _build_code_template(self) -> str:
        """Build the Python code template based on state."""
        # Backend class name (CamelCase)
        class_name = self._to_camel_case(self.state.backend_name) + "Backend"
        
        # Build imports
        imports = self._build_imports()
        
        # Build class definition
        class_def = self._build_class_definition(class_name)
        
        # Build methods
        methods = self._build_methods(class_name)
        
        # Combine all parts
        code = f'''"""
{self.state.display_name} Backend for Proxima.

{self.state.description or "A custom quantum computing backend implementation."}

Author: {self.state.author or "Unknown"}
Version: {self.state.version}
"""

{imports}

{class_def}

{methods}


# Backend registration
def get_backend() -> {class_name}:
    """Factory function to create the backend instance."""
    return {class_name}()
'''
        return code
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
    
    def _build_imports(self) -> str:
        """Build import statements."""
        imports = [
            "from __future__ import annotations",
            "",
            "from typing import Any, Dict, List, Optional, Sequence",
            "from dataclasses import dataclass, field",
            "import logging",
            "",
            "# Proxima imports",
            "from proxima.backends.base import BaseBackend, BackendCapabilities",
            "from proxima.core.circuit import QuantumCircuit",
            "from proxima.core.result import ExecutionResult",
        ]
        
        if self.state.library_name:
            imports.append(f"")
            imports.append(f"# Backend library import")
            imports.append(f"import {self.state.library_name}")
        
        return "\n".join(imports)
    
    def _build_class_definition(self, class_name: str) -> str:
        """Build the class definition."""
        return f'''
logger = logging.getLogger(__name__)


class {class_name}(BaseBackend):
    """
    {self.state.display_name} backend implementation.
    
    This backend wraps {self.state.library_name or "a custom simulator"} to provide
    quantum circuit execution capabilities.
    
    Supported Features:
    - Max qubits: {self.state.max_qubits}
    - Simulator types: {", ".join(self.state.simulator_types) or "state_vector"}
    - Noise support: {"Yes" if self.state.supports_noise else "No"}
    - GPU support: {"Yes" if self.state.supports_gpu else "No"}
    """
    
    name = "{self.state.backend_name}"
    display_name = "{self.state.display_name}"
    version = "{self.state.version}"
'''
    
    def _build_methods(self, class_name: str) -> str:
        """Build the class methods."""
        methods = []
        
        # __init__ method
        methods.append(self._build_init_method())
        
        # get_capabilities method
        methods.append(self._build_capabilities_method())
        
        # is_available method
        methods.append(self._build_is_available_method())
        
        # execute method
        methods.append(self._build_execute_method())
        
        # Gate mapping methods
        methods.append(self._build_gate_methods())
        
        return "\n".join(methods)
    
    def _build_init_method(self) -> str:
        """Build the __init__ method."""
        return '''    def __init__(self, **kwargs: Any) -> None:
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
        
        logger.info(f"Initialized {self.name} backend")
'''
    
    def _build_capabilities_method(self) -> str:
        """Build the get_capabilities method."""
        sim_types = self.state.simulator_types or ["state_vector"]
        
        return f'''    def get_capabilities(self) -> BackendCapabilities:
        """
        Return the capabilities of this backend.
        
        Returns:
            BackendCapabilities object describing what this backend supports
        """
        return BackendCapabilities(
            max_qubits={self.state.max_qubits},
            simulator_types={sim_types},
            supports_noise={str(self.state.supports_noise)},
            supports_gpu={str(self.state.supports_gpu)},
            supports_batching={str(self.state.supports_batching)},
            supports_parameter_binding={str(self.state.supports_parameter_binding)},
            supports_custom_gates={str(self.state.supports_custom_gates)},
        )
'''
    
    def _build_is_available_method(self) -> str:
        """Build the is_available method."""
        library = self.state.library_name or "your_library"
        return f'''    @classmethod
    def is_available(cls) -> bool:
        """
        Check if this backend is available.
        
        Returns:
            True if the backend can be used, False otherwise
        """
        try:
            import {library}
            return True
        except ImportError:
            logger.warning("{library} not installed")
            return False
'''
    
    def _build_execute_method(self) -> str:
        """Build the execute method."""
        return '''    def execute(
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
        pass
    
    def _convert_result(self, raw_result: Any) -> ExecutionResult:
        """Convert backend result to Proxima format."""
        # TODO: Implement result conversion
        pass
'''
    
    def _build_gate_methods(self) -> str:
        """Build gate mapping methods."""
        gate_map = {}
        for proxima_gate, mapping in self.state.gate_mappings.items():
            gate_map[proxima_gate] = mapping.backend_gate
        
        if not gate_map:
            gate_map = {
                "H": "H", "X": "X", "Y": "Y", "Z": "Z",
                "CNOT": "CNOT", "CZ": "CZ"
            }
        
        gate_map_str = ",\n            ".join(
            f'"{k}": "{v}"' for k, v in gate_map.items()
        )
        
        return f'''    def _get_gate_mapping(self) -> Dict[str, str]:
        """
        Get the mapping from Proxima gates to backend gates.
        
        Returns:
            Dictionary mapping Proxima gate names to backend gate names
        """
        return {{
            {gate_map_str}
        }}
'''
    
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle template style change."""
        if event.radio_set.id == "template_style":
            # Regenerate code with new style
            self._generate_code()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "btn_back":
            self.state.current_step = 4
            self.dismiss({"action": "back", "state": self.state})
        
        elif event.button.id == "btn_regenerate":
            self._generate_code()
            self.notify("Code regenerated!", severity="information")
        
        elif event.button.id == "btn_cancel":
            self.dismiss({"action": "cancel"})
        
        elif event.button.id == "btn_next":
            # Save any edits from the text area
            code_area = self.query_one("#code_area", TextArea)
            self.state.generated_code = code_area.text
            
            self.state.current_step = 6
            self.dismiss({"action": "next", "state": self.state})
    
    def action_cancel(self) -> None:
        """Handle escape key."""
        self.dismiss({"action": "cancel"})
    
    def action_save_template(self) -> None:
        """Save the template to state."""
        code_area = self.query_one("#code_area", TextArea)
        self.state.generated_code = code_area.text
        self.notify("Template saved!", severity="information")
