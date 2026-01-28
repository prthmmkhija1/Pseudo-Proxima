"""Proxima TUI Code Generation Package.

Provides advanced code generation capabilities for backend creation.
Phase 3 implementation of the Backend Addition Wizard.

Components:
- ASTCodeGenerator: AST-based Python code generation
- GateMappingGenerator: Gate mapping and conversion code
- CircuitConverterGenerator: Circuit format conversion
- ValidationCodeGenerator: Validation logic generation
- ErrorHandlingGenerator: Error handling patterns
- FullBackendGenerator: Complete backend package generation
- TemplateLibrary: Comprehensive template storage
- TemplateRenderer: Template variable substitution
- CodeValidator: Generated code validation
- BackendPackageGenerator: High-level package generation
"""

from .code_generator import (
    # Core classes
    ASTCodeGenerator,
    GateMappingGenerator,
    CircuitConverterGenerator,
    ValidationCodeGenerator,
    ErrorHandlingGenerator,
    FullBackendGenerator,
    # Supporting types
    CodeBlock,
    CodeBlockType,
)

from .template_library import (
    TemplateLibrary,
    TemplateType,
    TemplateMetadata,
)

from .template_renderer import (
    TemplateRenderer,
    VariableProcessor,
    BackendVariableBuilder,
    MultiFileRenderer,
    RenderResult,
    RenderError,
)

from .code_validator import (
    CodeValidator,
    SyntaxValidator,
    ImportValidator,
    StructureValidator,
    StyleValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)

from .package_generator import (
    BackendPackageGenerator,
    GeneratedFile,
    GenerationResult,
    generate_backend,
)

__all__ = [
    # Code Generator
    "ASTCodeGenerator",
    "GateMappingGenerator",
    "CircuitConverterGenerator",
    "ValidationCodeGenerator",
    "ErrorHandlingGenerator",
    "FullBackendGenerator",
    "CodeBlock",
    "CodeBlockType",
    # Template Library
    "TemplateLibrary",
    "TemplateType",
    "TemplateMetadata",
    # Template Renderer
    "TemplateRenderer",
    "VariableProcessor",
    "BackendVariableBuilder",
    "MultiFileRenderer",
    "RenderResult",
    "RenderError",
    # Code Validator
    "CodeValidator",
    "SyntaxValidator",
    "ImportValidator",
    "StructureValidator",
    "StyleValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    # Package Generator
    "BackendPackageGenerator",
    "GeneratedFile",
    "GenerationResult",
    "generate_backend",
]
