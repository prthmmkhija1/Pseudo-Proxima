"""Code Validator for Generated Backend Code.

Validates generated Python code for syntax, imports, and structure.
Provides detailed error messages and suggestions for fixes.
"""

from __future__ import annotations

import ast
import re
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        location = f"line {self.line}" if self.line else ""
        if self.column:
            location += f", col {self.column}"
        
        parts = [f"[{self.severity.value.upper()}]"]
        if location:
            parts.append(f"({location})")
        parts.append(self.message)
        
        if self.suggestion:
            parts.append(f"\n  Suggestion: {self.suggestion}")
        
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0


class SyntaxValidator:
    """Validate Python syntax."""
    
    def validate(self, code: str, filename: str = "<code>") -> ValidationResult:
        """Validate Python syntax.
        
        Args:
            code: Python source code
            filename: Filename for error messages
            
        Returns:
            ValidationResult with any syntax errors
        """
        issues = []
        
        try:
            ast.parse(code, filename=filename)
            return ValidationResult(valid=True)
        
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Syntax error: {e.msg}",
                line=e.lineno,
                column=e.offset,
                code="E001",
                suggestion=self._get_syntax_suggestion(e)
            ))
            return ValidationResult(valid=False, issues=issues)
    
    def _get_syntax_suggestion(self, error: SyntaxError) -> str:
        """Get suggestion for fixing syntax error."""
        msg = error.msg.lower()
        
        if "unexpected indent" in msg:
            return "Check indentation - use 4 spaces consistently"
        elif "expected ':'" in msg:
            return "Add colon after if/for/def/class statement"
        elif "invalid syntax" in msg and error.text:
            if "=" in error.text:
                return "Check for typos or missing operators"
        elif "unmatched" in msg:
            return "Check for matching parentheses, brackets, or braces"
        
        return "Review the line for Python syntax issues"


class ImportValidator:
    """Validate import statements."""
    
    # Standard library modules
    STDLIB_MODULES = {
        'os', 'sys', 'time', 'json', 're', 'logging', 'typing',
        'pathlib', 'tempfile', 'subprocess', 'functools', 'itertools',
        'collections', 'dataclasses', 'enum', 'abc', 'copy', 'math',
        'random', 'datetime', 'asyncio', 'threading', 'concurrent',
    }
    
    # Known third-party modules
    KNOWN_THIRD_PARTY = {
        'httpx', 'requests', 'numpy', 'scipy', 'qiskit', 'cirq',
        'pennylane', 'pyquil', 'braket', 'textual', 'rich', 'pytest',
    }
    
    def validate(self, code: str) -> ValidationResult:
        """Validate import statements.
        
        Args:
            code: Python source code
            
        Returns:
            ValidationResult with import issues
        """
        issues = []
        stats = {
            "stdlib_imports": 0,
            "third_party_imports": 0,
            "local_imports": 0,
        }
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ValidationResult(valid=True, issues=[], stats=stats)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    issues.extend(self._check_import(alias.name, node.lineno))
                    self._categorize_import(alias.name, stats)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    issues.extend(self._check_import(node.module, node.lineno))
                    self._categorize_import(node.module, stats)
        
        # Check for issues that would prevent execution
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        
        return ValidationResult(
            valid=not has_errors,
            issues=issues,
            stats=stats
        )
    
    def _check_import(self, module: str, line: int) -> List[ValidationIssue]:
        """Check a single import."""
        issues = []
        base_module = module.split('.')[0]
        
        # Check if module exists (for stdlib/known)
        if base_module in self.STDLIB_MODULES:
            pass  # OK
        elif base_module in self.KNOWN_THIRD_PARTY:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Third-party import: {module}",
                line=line,
                code="I001",
                suggestion=f"Ensure {base_module} is installed: pip install {base_module}"
            ))
        elif base_module.startswith('proxima'):
            pass  # Local import, OK
        
        return issues
    
    def _categorize_import(self, module: str, stats: Dict[str, int]) -> None:
        """Categorize an import for statistics."""
        base_module = module.split('.')[0]
        
        if base_module in self.STDLIB_MODULES:
            stats["stdlib_imports"] += 1
        elif base_module in self.KNOWN_THIRD_PARTY:
            stats["third_party_imports"] += 1
        else:
            stats["local_imports"] += 1


class StructureValidator:
    """Validate code structure for backend adapters."""
    
    # Required methods for backend adapters
    REQUIRED_METHODS = {
        'get_name', 'get_version', 'get_capabilities',
        'initialize', 'validate_circuit', 'execute',
        'supports_simulator', 'is_available', 'cleanup',
    }
    
    # Optional but recommended methods
    RECOMMENDED_METHODS = {
        'estimate_resources', '__enter__', '__exit__',
    }
    
    def validate(self, code: str, class_name: str = None) -> ValidationResult:
        """Validate code structure.
        
        Args:
            code: Python source code
            class_name: Expected class name to validate
            
        Returns:
            ValidationResult with structure issues
        """
        issues = []
        stats = {
            "classes": 0,
            "methods": 0,
            "functions": 0,
            "missing_methods": [],
        }
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ValidationResult(valid=False, issues=[], stats=stats)
        
        # Find all classes
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        stats["classes"] = len(classes)
        
        # Find adapter class
        adapter_class = None
        for cls in classes:
            if cls.name.endswith('Adapter'):
                adapter_class = cls
                break
            if class_name and cls.name == class_name:
                adapter_class = cls
                break
        
        if not adapter_class and class_name:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Expected class '{class_name}' not found",
                code="S001"
            ))
            return ValidationResult(valid=False, issues=issues, stats=stats)
        
        if adapter_class:
            issues.extend(self._validate_adapter_class(adapter_class, stats))
        
        # Count functions
        stats["functions"] = len([
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and
            not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree))
        ])
        
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        return ValidationResult(valid=not has_errors, issues=issues, stats=stats)
    
    def _validate_adapter_class(
        self,
        cls: ast.ClassDef,
        stats: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate adapter class structure."""
        issues = []
        
        # Get all method names
        methods = {
            node.name for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        stats["methods"] = len(methods)
        
        # Check for required methods
        missing = self.REQUIRED_METHODS - methods
        stats["missing_methods"] = list(missing)
        
        for method in missing:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Missing required method: {method}",
                code="S002",
                suggestion=f"Implement the {method}() method in the adapter class"
            ))
        
        # Check for recommended methods
        for method in self.RECOMMENDED_METHODS - methods:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Missing recommended method: {method}",
                code="S003",
                suggestion=f"Consider implementing {method}() for better functionality"
            ))
        
        # Check for docstrings
        if not ast.get_docstring(cls):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Class missing docstring",
                line=cls.lineno,
                code="S004",
                suggestion="Add a docstring describing the backend adapter"
            ))
        
        # Check class attributes
        required_attrs = {'name', 'version'}
        defined_attrs = set()
        
        for node in cls.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_attrs.add(target.id)
        
        for attr in required_attrs - defined_attrs:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Missing class attribute: {attr}",
                code="S005",
                suggestion=f"Add '{attr} = \"value\"' as a class attribute"
            ))
        
        return issues


class StyleValidator:
    """Validate code style and best practices."""
    
    MAX_LINE_LENGTH = 100
    MAX_FUNCTION_LENGTH = 50
    
    def validate(self, code: str) -> ValidationResult:
        """Validate code style.
        
        Args:
            code: Python source code
            
        Returns:
            ValidationResult with style issues
        """
        issues = []
        stats = {
            "lines": 0,
            "long_lines": 0,
            "blank_lines": 0,
        }
        
        lines = code.split('\n')
        stats["lines"] = len(lines)
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.MAX_LINE_LENGTH:
                stats["long_lines"] += 1
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HINT,
                    message=f"Line too long ({len(line)} > {self.MAX_LINE_LENGTH})",
                    line=i,
                    code="W001"
                ))
            
            # Count blank lines
            if not line.strip():
                stats["blank_lines"] += 1
            
            # Check for trailing whitespace
            if line.rstrip() != line:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HINT,
                    message="Trailing whitespace",
                    line=i,
                    code="W002"
                ))
            
            # Check for tabs
            if '\t' in line:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Tab character found (use spaces)",
                    line=i,
                    code="W003"
                ))
        
        # Check function lengths
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = node.end_lineno - node.lineno + 1
                    if func_lines > self.MAX_FUNCTION_LENGTH:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Function '{node.name}' too long ({func_lines} lines)",
                            line=node.lineno,
                            code="W004",
                            suggestion="Consider breaking into smaller functions"
                        ))
        except SyntaxError:
            pass
        
        return ValidationResult(valid=True, issues=issues, stats=stats)


class CodeValidator:
    """Main validator combining all validation checks."""
    
    def __init__(self, strict: bool = False):
        """Initialize validator.
        
        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict
        self.syntax_validator = SyntaxValidator()
        self.import_validator = ImportValidator()
        self.structure_validator = StructureValidator()
        self.style_validator = StyleValidator()
    
    def validate(
        self,
        code: str,
        filename: str = "<code>",
        class_name: str = None,
        check_style: bool = True
    ) -> ValidationResult:
        """Perform all validation checks.
        
        Args:
            code: Python source code
            filename: Filename for error messages
            class_name: Expected class name
            check_style: Whether to check style
            
        Returns:
            Combined ValidationResult
        """
        all_issues = []
        all_stats = {}
        
        # Syntax check (required)
        syntax_result = self.syntax_validator.validate(code, filename)
        all_issues.extend(syntax_result.issues)
        
        if not syntax_result.valid:
            # Can't continue if syntax is invalid
            return ValidationResult(
                valid=False,
                issues=all_issues,
                stats=all_stats
            )
        
        # Import check
        import_result = self.import_validator.validate(code)
        all_issues.extend(import_result.issues)
        all_stats.update(import_result.stats)
        
        # Structure check
        structure_result = self.structure_validator.validate(code, class_name)
        all_issues.extend(structure_result.issues)
        all_stats.update(structure_result.stats)
        
        # Style check (optional)
        if check_style:
            style_result = self.style_validator.validate(code)
            all_issues.extend(style_result.issues)
            all_stats.update(style_result.stats)
        
        # Determine overall validity
        if self.strict:
            valid = not any(
                i.severity in (ValidationSeverity.ERROR, ValidationSeverity.WARNING)
                for i in all_issues
            )
        else:
            valid = not any(
                i.severity == ValidationSeverity.ERROR
                for i in all_issues
            )
        
        return ValidationResult(
            valid=valid,
            issues=all_issues,
            stats=all_stats
        )
    
    def validate_file(self, filepath: str, **kwargs) -> ValidationResult:
        """Validate a Python file.
        
        Args:
            filepath: Path to Python file
            **kwargs: Additional options for validate()
            
        Returns:
            ValidationResult
        """
        path = Path(filepath)
        
        if not path.exists():
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"File not found: {filepath}",
                    code="F001"
                )]
            )
        
        code = path.read_text(encoding='utf-8')
        return self.validate(code, filename=str(path), **kwargs)
    
    def validate_package(
        self,
        files: Dict[str, str],
        expected_structure: Dict[str, str] = None
    ) -> Dict[str, ValidationResult]:
        """Validate a package of files.
        
        Args:
            files: Dict of filename -> content
            expected_structure: Expected file structure
            
        Returns:
            Dict of filename -> ValidationResult
        """
        results = {}
        
        for filename, content in files.items():
            # Determine expected class name from filename
            class_name = None
            if 'adapter' in filename:
                # Extract class name from content
                for line in content.split('\n'):
                    if 'class ' in line and 'Adapter' in line:
                        match = re.search(r'class\s+(\w+Adapter)', line)
                        if match:
                            class_name = match.group(1)
                            break
            
            results[filename] = self.validate(
                content,
                filename=filename,
                class_name=class_name
            )
        
        return results
