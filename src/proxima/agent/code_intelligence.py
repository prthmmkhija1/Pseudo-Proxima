"""Code Intelligence Module for AST-based Code Modification.

Phase 8: Backend Code Modification with Safety

Provides intelligent code analysis and modification:
- AST-based Python code parsing
- Symbol location (functions, classes, methods)
- Smart indentation detection
- Code insertion with proper formatting
- Import management
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.code_intelligence")


class SymbolType(Enum):
    """Types of code symbols."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    MODULE = "module"
    DECORATOR = "decorator"
    PARAMETER = "parameter"


@dataclass
class CodeLocation:
    """Location in code."""
    file: str
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0
    
    @property
    def line_range(self) -> Tuple[int, int]:
        """Get line range."""
        return (self.start_line, self.end_line)
    
    def contains_line(self, line: int) -> bool:
        """Check if location contains a line."""
        return self.start_line <= line <= self.end_line


@dataclass
class Symbol:
    """A code symbol (function, class, variable, etc.)."""
    name: str
    symbol_type: SymbolType
    location: CodeLocation
    parent: Optional[str] = None  # Parent class/function name
    signature: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified name."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.symbol_type.value,
            "location": {
                "start_line": self.location.start_line,
                "end_line": self.location.end_line,
            },
            "parent": self.parent,
            "signature": self.signature,
            "decorators": self.decorators,
            "parameters": self.parameters,
        }


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    names: List[str]  # Imported names (empty for module import)
    alias: Optional[str] = None
    is_from_import: bool = False
    location: Optional[CodeLocation] = None
    
    def to_statement(self) -> str:
        """Convert to import statement."""
        if self.is_from_import:
            names_str = ", ".join(self.names)
            return f"from {self.module} import {names_str}"
        elif self.alias:
            return f"import {self.module} as {self.alias}"
        else:
            return f"import {self.module}"


@dataclass
class ModificationTemplate:
    """Template for code modification."""
    name: str
    description: str
    template: str
    variables: List[str]  # Variables to replace in template
    
    def apply(self, **kwargs: str) -> str:
        """Apply variables to template."""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"${{{var}}}", kwargs[var])
        return result


@dataclass
class CodeAnalysisResult:
    """Result of code analysis."""
    file: str
    symbols: List[Symbol]
    imports: List[ImportInfo]
    errors: List[str]
    indentation: str  # Detected indentation (spaces/tabs)
    line_count: int
    
    @property
    def classes(self) -> List[Symbol]:
        """Get all class symbols."""
        return [s for s in self.symbols if s.symbol_type == SymbolType.CLASS]
    
    @property
    def functions(self) -> List[Symbol]:
        """Get all function symbols."""
        return [s for s in self.symbols if s.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD)]


class CodeIntelligence:
    """Intelligent code analysis and modification.
    
    Features:
    - Parse Python code using AST
    - Locate functions, classes, methods
    - Detect indentation style
    - Generate code modifications
    - Manage imports
    
    Example:
        >>> intel = CodeIntelligence()
        >>> result = intel.analyze_file("backend.py")
        >>> 
        >>> # Find a function
        >>> func = intel.find_symbol(result, "process_request")
        >>> if func:
        ...     print(f"Found at line {func.location.start_line}")
        >>> 
        >>> # Add import
        >>> new_code = intel.add_import(
        ...     code=original_code,
        ...     module="typing",
        ...     names=["Optional", "List"]
        ... )
    """
    
    # Common modification templates
    DEFAULT_TEMPLATES = {
        "add_method": ModificationTemplate(
            name="add_method",
            description="Add method to class",
            template='''
    def ${method_name}(self${parameters}) -> ${return_type}:
        """${docstring}"""
        ${body}
''',
            variables=["method_name", "parameters", "return_type", "docstring", "body"],
        ),
        "add_function": ModificationTemplate(
            name="add_function",
            description="Add standalone function",
            template='''
def ${function_name}(${parameters}) -> ${return_type}:
    """${docstring}"""
    ${body}
''',
            variables=["function_name", "parameters", "return_type", "docstring", "body"],
        ),
        "add_class": ModificationTemplate(
            name="add_class",
            description="Add class definition",
            template='''
class ${class_name}${bases}:
    """${docstring}"""
    
    def __init__(self${init_params}):
        ${init_body}
''',
            variables=["class_name", "bases", "docstring", "init_params", "init_body"],
        ),
        "add_property": ModificationTemplate(
            name="add_property",
            description="Add property to class",
            template='''
    @property
    def ${property_name}(self) -> ${return_type}:
        """${docstring}"""
        return self._${property_name}
    
    @${property_name}.setter
    def ${property_name}(self, value: ${return_type}) -> None:
        self._${property_name} = value
''',
            variables=["property_name", "return_type", "docstring"],
        ),
    }
    
    def __init__(self):
        """Initialize CodeIntelligence."""
        self.templates = dict(self.DEFAULT_TEMPLATES)
        self._symbol_cache: Dict[str, CodeAnalysisResult] = {}
    
    def analyze_code(self, code: str, filename: str = "<code>") -> CodeAnalysisResult:
        """Analyze Python code.
        
        Args:
            code: Python source code
            filename: Optional filename for error messages
            
        Returns:
            CodeAnalysisResult
        """
        symbols: List[Symbol] = []
        imports: List[ImportInfo] = []
        errors: List[str] = []
        
        # Detect indentation
        indentation = self._detect_indentation(code)
        
        # Parse AST
        try:
            tree = ast.parse(code, filename=filename)
            
            # Extract symbols and imports
            self._extract_symbols(tree, filename, symbols)
            self._extract_imports(tree, filename, imports)
            
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
        
        return CodeAnalysisResult(
            file=filename,
            symbols=symbols,
            imports=imports,
            errors=errors,
            indentation=indentation,
            line_count=code.count('\n') + 1,
        )
    
    def analyze_file(self, file_path: str, use_cache: bool = True) -> CodeAnalysisResult:
        """Analyze a Python file.
        
        Args:
            file_path: Path to Python file
            use_cache: Whether to use cached results
            
        Returns:
            CodeAnalysisResult
        """
        path = Path(file_path)
        
        # Check cache
        if use_cache and file_path in self._symbol_cache:
            return self._symbol_cache[file_path]
        
        try:
            code = path.read_text(encoding="utf-8")
            result = self.analyze_code(code, str(path))
            
            if use_cache:
                self._symbol_cache[file_path] = result
            
            return result
            
        except Exception as e:
            return CodeAnalysisResult(
                file=str(path),
                symbols=[],
                imports=[],
                errors=[f"Failed to read file: {e}"],
                indentation="    ",
                line_count=0,
            )
    
    def _detect_indentation(self, code: str) -> str:
        """Detect indentation style."""
        # Count leading whitespace patterns
        space_counts: Dict[int, int] = {}
        tab_count = 0
        
        for line in code.split('\n'):
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue
            
            leading = line[:len(line) - len(stripped)]
            
            if leading.startswith('\t'):
                tab_count += 1
            elif leading.startswith(' '):
                space_count = len(leading)
                if space_count > 0:
                    space_counts[space_count] = space_counts.get(space_count, 0) + 1
        
        # Prefer tabs if they're used more
        if tab_count > sum(space_counts.values()):
            return '\t'
        
        # Find most common space indent
        if space_counts:
            # Find smallest common indentation
            for size in [2, 4, 8]:
                if any(count % size == 0 for count in space_counts.keys()):
                    return ' ' * size
        
        return '    '  # Default 4 spaces
    
    def _extract_symbols(
        self,
        tree: ast.AST,
        filename: str,
        symbols: List[Symbol],
        parent: Optional[str] = None,
    ) -> None:
        """Extract symbols from AST."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                location = CodeLocation(
                    file=filename,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    start_col=node.col_offset,
                )
                
                symbol = Symbol(
                    name=node.name,
                    symbol_type=SymbolType.CLASS,
                    location=location,
                    parent=parent,
                    docstring=ast.get_docstring(node),
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                )
                symbols.append(symbol)
                
                # Recurse into class
                self._extract_symbols(node, filename, symbols, parent=node.name)
                
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                location = CodeLocation(
                    file=filename,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    start_col=node.col_offset,
                )
                
                # Determine if method or function
                symbol_type = SymbolType.METHOD if parent else SymbolType.FUNCTION
                
                # Extract parameters
                params = self._extract_parameters(node)
                
                # Extract return type
                return_type = None
                if node.returns:
                    return_type = self._get_annotation_str(node.returns)
                
                # Build signature
                signature = self._build_signature(node)
                
                symbol = Symbol(
                    name=node.name,
                    symbol_type=symbol_type,
                    location=location,
                    parent=parent,
                    signature=signature,
                    docstring=ast.get_docstring(node),
                    decorators=[self._get_decorator_name(d) for d in node.decorator_list],
                    parameters=params,
                    return_type=return_type,
                )
                symbols.append(symbol)
            
            elif isinstance(node, (ast.Module,)):
                self._extract_symbols(node, filename, symbols, parent)
    
    def _extract_imports(
        self,
        tree: ast.AST,
        filename: str,
        imports: List[ImportInfo],
    ) -> None:
        """Extract import statements."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        is_from_import=False,
                        location=CodeLocation(
                            file=filename,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                        ),
                    ))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(ImportInfo(
                        module=node.module,
                        names=[alias.name for alias in node.names],
                        is_from_import=True,
                        location=CodeLocation(
                            file=filename,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                        ),
                    ))
    
    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_decorator_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return "<unknown>"
    
    def _get_annotation_str(self, node: ast.expr) -> str:
        """Get string representation of annotation."""
        try:
            return ast.unparse(node)
        except Exception:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Constant):
                return str(node.value)
            return "<unknown>"
    
    def _extract_parameters(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract parameter names."""
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        return params
    
    def _build_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Build function signature."""
        try:
            params = []
            for arg in node.args.args:
                param = arg.arg
                if arg.annotation:
                    param += f": {self._get_annotation_str(arg.annotation)}"
                params.append(param)
            
            sig = f"def {node.name}({', '.join(params)})"
            if node.returns:
                sig += f" -> {self._get_annotation_str(node.returns)}"
            
            return sig
        except Exception:
            return f"def {node.name}(...)"
    
    def find_symbol(
        self,
        analysis: CodeAnalysisResult,
        name: str,
        symbol_type: Optional[SymbolType] = None,
    ) -> Optional[Symbol]:
        """Find a symbol by name.
        
        Args:
            analysis: Analysis result
            name: Symbol name (can be qualified like "Class.method")
            symbol_type: Optional type filter
            
        Returns:
            Symbol or None
        """
        # Try qualified name first
        for symbol in analysis.symbols:
            if symbol.qualified_name == name:
                if symbol_type is None or symbol.symbol_type == symbol_type:
                    return symbol
        
        # Try simple name
        for symbol in analysis.symbols:
            if symbol.name == name:
                if symbol_type is None or symbol.symbol_type == symbol_type:
                    return symbol
        
        return None
    
    def find_symbols_by_type(
        self,
        analysis: CodeAnalysisResult,
        symbol_type: SymbolType,
    ) -> List[Symbol]:
        """Find all symbols of a type."""
        return [s for s in analysis.symbols if s.symbol_type == symbol_type]
    
    def get_class_methods(
        self,
        analysis: CodeAnalysisResult,
        class_name: str,
    ) -> List[Symbol]:
        """Get all methods of a class."""
        return [
            s for s in analysis.symbols
            if s.symbol_type == SymbolType.METHOD and s.parent == class_name
        ]
    
    def add_import(
        self,
        code: str,
        module: str,
        names: Optional[List[str]] = None,
        alias: Optional[str] = None,
    ) -> str:
        """Add import statement to code.
        
        Args:
            code: Source code
            module: Module to import
            names: Names to import (for from import)
            alias: Alias for module
            
        Returns:
            Modified code
        """
        lines = code.split('\n')
        
        # Find import section
        import_end = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                import_end = i + 1
            elif stripped and not stripped.startswith('#') and import_end > 0:
                break
        
        # Check if import already exists
        analysis = self.analyze_code(code)
        for imp in analysis.imports:
            if imp.module == module:
                if names is None:
                    return code  # Already imported
                if imp.is_from_import and set(names).issubset(set(imp.names)):
                    return code  # Names already imported
        
        # Build import statement
        if names:
            new_import = f"from {module} import {', '.join(names)}"
        elif alias:
            new_import = f"import {module} as {alias}"
        else:
            new_import = f"import {module}"
        
        # Insert import
        if import_end == 0:
            # No imports yet, add after docstring if present
            insert_at = 0
            if lines and (lines[0].startswith('"""') or lines[0].startswith("'''")):
                # Find end of docstring
                for i in range(1, len(lines)):
                    if '"""' in lines[i] or "'''" in lines[i]:
                        insert_at = i + 1
                        break
            lines.insert(insert_at, new_import)
        else:
            lines.insert(import_end, new_import)
        
        return '\n'.join(lines)
    
    def insert_after_symbol(
        self,
        code: str,
        symbol_name: str,
        new_code: str,
    ) -> str:
        """Insert code after a symbol definition.
        
        Args:
            code: Source code
            symbol_name: Symbol to insert after
            new_code: Code to insert
            
        Returns:
            Modified code
        """
        analysis = self.analyze_code(code)
        symbol = self.find_symbol(analysis, symbol_name)
        
        if not symbol:
            raise ValueError(f"Symbol not found: {symbol_name}")
        
        lines = code.split('\n')
        insert_line = symbol.location.end_line
        
        # Ensure proper indentation
        if symbol.symbol_type == SymbolType.METHOD:
            # Match class indentation
            base_indent = analysis.indentation
        else:
            base_indent = ""
        
        # Indent new code
        indented_code = self._indent_code(new_code, base_indent)
        
        lines.insert(insert_line, '')
        lines.insert(insert_line + 1, indented_code)
        
        return '\n'.join(lines)
    
    def insert_in_class(
        self,
        code: str,
        class_name: str,
        new_code: str,
        after_method: Optional[str] = None,
    ) -> str:
        """Insert code inside a class.
        
        Args:
            code: Source code
            class_name: Target class name
            new_code: Code to insert
            after_method: Optional method to insert after
            
        Returns:
            Modified code
        """
        analysis = self.analyze_code(code)
        class_symbol = self.find_symbol(analysis, class_name, SymbolType.CLASS)
        
        if not class_symbol:
            raise ValueError(f"Class not found: {class_name}")
        
        lines = code.split('\n')
        
        if after_method:
            method = self.find_symbol(analysis, f"{class_name}.{after_method}")
            if method:
                insert_line = method.location.end_line
            else:
                insert_line = class_symbol.location.end_line
        else:
            # Insert at end of class
            insert_line = class_symbol.location.end_line
        
        # Indent with class indentation
        indented_code = self._indent_code(new_code, analysis.indentation)
        
        lines.insert(insert_line, '')
        lines.insert(insert_line + 1, indented_code)
        
        return '\n'.join(lines)
    
    def _indent_code(self, code: str, base_indent: str) -> str:
        """Apply base indentation to code."""
        lines = code.split('\n')
        indented = []
        
        for line in lines:
            if line.strip():
                indented.append(base_indent + line)
            else:
                indented.append('')
        
        return '\n'.join(indented)
    
    def apply_template(
        self,
        template_name: str,
        **kwargs: str,
    ) -> str:
        """Apply a modification template.
        
        Args:
            template_name: Name of template
            **kwargs: Template variables
            
        Returns:
            Generated code
        """
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
        
        return template.apply(**kwargs)
    
    def get_symbol_at_line(
        self,
        analysis: CodeAnalysisResult,
        line: int,
    ) -> Optional[Symbol]:
        """Get symbol at a specific line."""
        for symbol in analysis.symbols:
            if symbol.location.contains_line(line):
                return symbol
        return None
    
    def suggest_imports(
        self,
        code: str,
        undefined_name: str,
    ) -> List[str]:
        """Suggest imports for undefined name.
        
        Args:
            code: Source code
            undefined_name: Undefined name
            
        Returns:
            List of suggested import statements
        """
        # Common stdlib imports
        common_imports = {
            "Path": "from pathlib import Path",
            "Optional": "from typing import Optional",
            "List": "from typing import List",
            "Dict": "from typing import Dict",
            "Tuple": "from typing import Tuple",
            "Any": "from typing import Any",
            "Union": "from typing import Union",
            "Callable": "from typing import Callable",
            "dataclass": "from dataclasses import dataclass",
            "field": "from dataclasses import field",
            "Enum": "from enum import Enum",
            "datetime": "from datetime import datetime",
            "json": "import json",
            "os": "import os",
            "re": "import re",
            "sys": "import sys",
            "logging": "import logging",
            "asyncio": "import asyncio",
            "threading": "import threading",
        }
        
        suggestions = []
        
        if undefined_name in common_imports:
            suggestions.append(common_imports[undefined_name])
        
        return suggestions
    
    def clear_cache(self) -> None:
        """Clear symbol cache."""
        self._symbol_cache.clear()


# Global instance
_code_intelligence: Optional[CodeIntelligence] = None


def get_code_intelligence() -> CodeIntelligence:
    """Get the global CodeIntelligence instance."""
    global _code_intelligence
    if _code_intelligence is None:
        _code_intelligence = CodeIntelligence()
    return _code_intelligence
