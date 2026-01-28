"""Template Renderer for Backend Code Generation.

Renders templates with variable substitution and conditional blocks.
Supports ${variable} and ${if:condition}...${endif} syntax.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class RenderError(Exception):
    """Error during template rendering."""
    pass


@dataclass
class RenderResult:
    """Result of template rendering."""
    content: str
    variables_used: List[str]
    warnings: List[str]


class TemplateRenderer:
    """Render templates with variable substitution.
    
    Supports:
    - ${variable} - Simple variable substitution
    - ${variable|default} - Variable with default value
    - ${if:condition}...${endif} - Conditional blocks
    - ${for:item:list}...${endfor} - Loop blocks
    - ${include:template_name} - Template inclusion
    """
    
    # Pattern for ${variable} or ${variable|default}
    VAR_PATTERN = re.compile(r'\$\{(\w+)(?:\|([^}]*))?\}')
    
    # Pattern for ${if:condition}...${endif}
    IF_PATTERN = re.compile(
        r'\$\{if:(\w+)\}(.*?)\$\{endif\}',
        re.DOTALL
    )
    
    # Pattern for ${for:item:list}...${endfor}
    FOR_PATTERN = re.compile(
        r'\$\{for:(\w+):(\w+)\}(.*?)\$\{endfor\}',
        re.DOTALL
    )
    
    def __init__(self, strict: bool = False):
        """Initialize renderer.
        
        Args:
            strict: If True, raise error for missing variables
        """
        self.strict = strict
        self._variables_used: List[str] = []
        self._warnings: List[str] = []
    
    def render(self, template: str, variables: Dict[str, Any]) -> RenderResult:
        """Render a template with given variables.
        
        Args:
            template: Template string
            variables: Dictionary of variable values
            
        Returns:
            RenderResult with rendered content
            
        Raises:
            RenderError: If strict mode and variable missing
        """
        self._variables_used = []
        self._warnings = []
        
        # Process conditionals first
        result = self._process_conditionals(template, variables)
        
        # Process loops
        result = self._process_loops(result, variables)
        
        # Process variable substitution
        result = self._process_variables(result, variables)
        
        return RenderResult(
            content=result,
            variables_used=self._variables_used,
            warnings=self._warnings
        )
    
    def _process_conditionals(self, template: str, variables: Dict[str, Any]) -> str:
        """Process ${if:condition}...${endif} blocks."""
        def replace_if(match):
            condition = match.group(1)
            content = match.group(2)
            
            # Check if condition variable is truthy
            value = variables.get(condition)
            if value:
                return content
            return ""
        
        return self.IF_PATTERN.sub(replace_if, template)
    
    def _process_loops(self, template: str, variables: Dict[str, Any]) -> str:
        """Process ${for:item:list}...${endfor} blocks."""
        def replace_for(match):
            item_name = match.group(1)
            list_name = match.group(2)
            content = match.group(3)
            
            items = variables.get(list_name, [])
            if not isinstance(items, (list, tuple)):
                items = [items]
            
            result = []
            for item in items:
                # Create loop context
                loop_vars = variables.copy()
                loop_vars[item_name] = item
                
                # Render content with loop variable
                rendered = self._process_variables(content, loop_vars)
                result.append(rendered)
            
            return "".join(result)
        
        return self.FOR_PATTERN.sub(replace_for, template)
    
    def _process_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Process ${variable} substitutions."""
        def replace_var(match):
            var_name = match.group(1)
            default = match.group(2)
            
            self._variables_used.append(var_name)
            
            if var_name in variables:
                value = variables[var_name]
                
                # Handle different types
                if isinstance(value, bool):
                    return str(value)
                elif isinstance(value, (list, tuple)):
                    return ", ".join(str(v) for v in value)
                elif value is None:
                    return default if default else ""
                else:
                    return str(value)
            
            # Variable not found
            if default is not None:
                return default
            
            if self.strict:
                raise RenderError(f"Missing variable: {var_name}")
            
            self._warnings.append(f"Missing variable: {var_name}")
            return ""
        
        return self.VAR_PATTERN.sub(replace_var, template)
    
    def render_file(
        self,
        template_path: str,
        variables: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> RenderResult:
        """Render a template file.
        
        Args:
            template_path: Path to template file
            variables: Variables for rendering
            output_path: Optional output file path
            
        Returns:
            RenderResult with rendered content
        """
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        result = self.render(template, variables)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.content)
        
        return result


class VariableProcessor:
    """Process and transform template variables."""
    
    @staticmethod
    def to_class_name(snake_str: str) -> str:
        """Convert snake_case to CamelCase."""
        components = snake_str.replace("-", "_").split("_")
        return "".join(x.title() for x in components)
    
    @staticmethod
    def to_snake_case(camel_str: str) -> str:
        """Convert CamelCase to snake_case."""
        result = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', result).lower()
    
    @staticmethod
    def to_display_name(identifier: str) -> str:
        """Convert identifier to display name."""
        # Replace underscores with spaces
        name = identifier.replace("_", " ").replace("-", " ")
        # Title case
        return name.title()
    
    @staticmethod
    def escape_string(s: str) -> str:
        """Escape string for use in Python code."""
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    
    @staticmethod
    def indent(text: str, spaces: int = 4) -> str:
        """Indent text by specified spaces."""
        indent_str = " " * spaces
        lines = text.split("\n")
        return "\n".join(indent_str + line if line.strip() else line for line in lines)


class BackendVariableBuilder:
    """Build variable dictionary for backend templates."""
    
    def __init__(self):
        self.processor = VariableProcessor()
    
    def build(
        self,
        backend_name: str,
        display_name: str = None,
        version: str = "1.0.0",
        author: str = "",
        description: str = "",
        backend_type: str = "custom",
        library_name: str = "",
        max_qubits: int = 20,
        supports_noise: bool = False,
        supports_gpu: bool = False,
        supports_batching: bool = False,
        gate_mappings: Dict[str, str] = None,
        **extra
    ) -> Dict[str, Any]:
        """Build complete variable dictionary.
        
        Args:
            backend_name: Internal name (snake_case)
            display_name: Display name (defaults from backend_name)
            version: Version string
            author: Author name
            description: Backend description
            backend_type: Type of backend
            library_name: Python library name
            max_qubits: Maximum qubit count
            supports_noise: Noise simulation support
            supports_gpu: GPU acceleration support
            supports_batching: Batching support
            gate_mappings: Gate name mappings
            **extra: Additional variables
            
        Returns:
            Complete variable dictionary
        """
        # Generate derived values
        class_name = self.processor.to_class_name(backend_name)
        display_name = display_name or self.processor.to_display_name(backend_name)
        
        # Author line for templates
        author_line = f"Author: {author}" if author else ""
        
        # Install command
        if library_name:
            install_command = f"pip install {library_name}"
        else:
            install_command = "# Follow installation instructions for your backend"
        
        # Build gate mapping lines
        gate_mapping_lines = []
        if gate_mappings:
            for proxima_gate, backend_gate in gate_mappings.items():
                gate_mapping_lines.append(
                    f'        "{proxima_gate}": "{backend_gate}",'
                )
        
        # Additional simulator types
        additional_sim_types = ""
        if supports_noise:
            additional_sim_types += "SimulatorType.DENSITY_MATRIX,"
        
        variables = {
            # Names
            "backend_name": backend_name,
            "display_name": display_name,
            "class_name": class_name,
            
            # Metadata
            "version": version,
            "author": author,
            "author_line": author_line,
            "description": description or f"Custom quantum computing backend: {display_name}",
            
            # Type and library
            "backend_type": backend_type,
            "library_name": library_name,
            "install_command": install_command,
            
            # Capabilities
            "max_qubits": max_qubits,
            "supports_noise": str(supports_noise),
            "supports_gpu": str(supports_gpu),
            "supports_batching": str(supports_batching),
            "additional_sim_types": additional_sim_types,
            
            # Gate mappings
            "gate_mappings": gate_mappings or {},
            "gate_mapping_lines": "\n".join(gate_mapping_lines),
            
            # Command-line specific
            "tool_name": extra.get("tool_name", backend_name),
            "tool_command": extra.get("tool_command", backend_name),
            
            # API specific
            "api_url": extra.get("api_url", "http://localhost:8000"),
        }
        
        # Add any extra variables
        variables.update(extra)
        
        return variables


class MultiFileRenderer:
    """Render multiple template files for a backend."""
    
    def __init__(self):
        self.renderer = TemplateRenderer()
        self.var_builder = BackendVariableBuilder()
    
    def render_backend_package(
        self,
        templates: Dict[str, str],
        **kwargs
    ) -> Dict[str, RenderResult]:
        """Render all templates for a backend package.
        
        Args:
            templates: Dict of filename -> template_content
            **kwargs: Variables for rendering
            
        Returns:
            Dict of filename -> RenderResult
        """
        # Build variables
        variables = self.var_builder.build(**kwargs)
        
        # Render each template
        results = {}
        for filename, template in templates.items():
            # Substitute variables in filename too
            rendered_filename = self._render_filename(filename, variables)
            
            try:
                result = self.renderer.render(template, variables)
                results[rendered_filename] = result
            except RenderError as e:
                results[rendered_filename] = RenderResult(
                    content=f"# Error rendering template: {e}",
                    variables_used=[],
                    warnings=[str(e)]
                )
        
        return results
    
    def _render_filename(self, filename: str, variables: Dict[str, Any]) -> str:
        """Render variables in filename."""
        result = filename
        for var, value in variables.items():
            result = result.replace(f"${{{var}}}", str(value))
        return result
    
    def get_file_contents(self, results: Dict[str, RenderResult]) -> Dict[str, str]:
        """Extract just the file contents from results.
        
        Args:
            results: Dict of filename -> RenderResult
            
        Returns:
            Dict of filename -> content
        """
        return {
            filename: result.content
            for filename, result in results.items()
        }
