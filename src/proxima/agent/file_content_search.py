"""File Content Search and Analysis for Proxima Agent.

Phase 5: File System Operations & Administrative Access

Provides code search and analysis including:
- Fast text search using ripgrep when available
- Regex and literal string search
- Python code analysis using AST
- Code metrics calculation
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.file_content_search")


class SearchEngine(Enum):
    """Search engine type."""
    RIPGREP = "ripgrep"
    PYTHON = "python"
    AUTO = "auto"


class CodeElementType(Enum):
    """Types of code elements."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    VARIABLE = "variable"
    IMPORT = "import"
    CONSTANT = "constant"
    DECORATOR = "decorator"


@dataclass
class SearchResult:
    """A search result from content search."""
    file_path: Path
    line_number: int
    column: int
    line_content: str
    match_text: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "column": self.column,
            "line_content": self.line_content,
            "match_text": self.match_text,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


@dataclass
class CodeElement:
    """A code element found during analysis."""
    name: str
    element_type: CodeElementType
    file_path: Path
    line_number: int
    end_line: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "element_type": self.element_type.value,
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "end_line": self.end_line,
            "docstring": self.docstring,
            "signature": self.signature,
            "parent": self.parent,
            "decorators": self.decorators,
        }


@dataclass
class CodeMetrics:
    """Code metrics for a file."""
    file_path: Path
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    functions: int
    classes: int
    imports: int
    complexity: int  # McCabe complexity approximation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "complexity": self.complexity,
        }


@dataclass
class SearchSummary:
    """Summary of search results."""
    query: str
    total_matches: int
    files_searched: int
    files_with_matches: int
    search_time_ms: float
    results: List[SearchResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "total_matches": self.total_matches,
            "files_searched": self.files_searched,
            "files_with_matches": self.files_with_matches,
            "search_time_ms": self.search_time_ms,
            "results": [r.to_dict() for r in self.results],
        }


class FileContentSearch:
    """Search file contents and analyze code.
    
    Features:
    - Fast search using ripgrep when available
    - Regex and literal search
    - Context lines around matches
    - File type filtering
    
    Example:
        >>> search = FileContentSearch(project_root=Path("."))
        >>> 
        >>> # Search for a pattern
        >>> results = search.search("def main", file_pattern="*.py")
        >>> for result in results.results:
        ...     print(f"{result.file_path}:{result.line_number}")
        >>> 
        >>> # Search with regex
        >>> results = search.search(r"class \w+Handler", is_regex=True)
    """
    
    # File types for filtering
    FILE_TYPE_EXTENSIONS = {
        "python": [".py", ".pyx", ".pxd", ".pyi"],
        "cpp": [".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"],
        "javascript": [".js", ".jsx", ".ts", ".tsx"],
        "rust": [".rs"],
        "go": [".go"],
        "java": [".java"],
        "yaml": [".yaml", ".yml"],
        "json": [".json"],
        "markdown": [".md", ".markdown"],
        "text": [".txt", ".text"],
    }
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        use_ripgrep: bool = True,
        max_results: int = 1000,
        context_lines: int = 3,
    ):
        """Initialize the search.
        
        Args:
            project_root: Root directory for search
            use_ripgrep: Use ripgrep if available
            max_results: Maximum number of results
            context_lines: Lines of context before/after
        """
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.max_results = max_results
        self.context_lines = context_lines
        
        # Check for ripgrep
        self.ripgrep_available = False
        if use_ripgrep:
            self.ripgrep_available = shutil.which("rg") is not None
            if self.ripgrep_available:
                logger.debug("Using ripgrep for search")
    
    def search(
        self,
        pattern: str,
        path: Optional[Path] = None,
        file_pattern: Optional[str] = None,
        file_type: Optional[str] = None,
        is_regex: bool = False,
        case_sensitive: bool = True,
        whole_word: bool = False,
        include_hidden: bool = False,
    ) -> SearchSummary:
        """Search for content in files.
        
        Args:
            pattern: Search pattern
            path: Directory to search (default: project root)
            file_pattern: Glob pattern for files (e.g., "*.py")
            file_type: File type name (e.g., "python", "cpp")
            is_regex: Treat pattern as regex
            case_sensitive: Case-sensitive search
            whole_word: Match whole words only
            include_hidden: Include hidden files
            
        Returns:
            SearchSummary with results
        """
        import time
        start_time = time.time()
        
        search_path = Path(path) if path else self.project_root
        
        # Determine file extensions
        extensions = None
        if file_type and file_type in self.FILE_TYPE_EXTENSIONS:
            extensions = self.FILE_TYPE_EXTENSIONS[file_type]
        
        # Choose search method
        if self.ripgrep_available:
            results = self._search_ripgrep(
                pattern, search_path, file_pattern, extensions,
                is_regex, case_sensitive, whole_word, include_hidden
            )
        else:
            results = self._search_python(
                pattern, search_path, file_pattern, extensions,
                is_regex, case_sensitive, whole_word, include_hidden
            )
        
        # Count files with matches
        files_with_matches = len(set(r.file_path for r in results))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SearchSummary(
            query=pattern,
            total_matches=len(results),
            files_searched=0,  # Unknown with ripgrep
            files_with_matches=files_with_matches,
            search_time_ms=elapsed_ms,
            results=results[:self.max_results],
        )
    
    def _search_ripgrep(
        self,
        pattern: str,
        search_path: Path,
        file_pattern: Optional[str],
        extensions: Optional[List[str]],
        is_regex: bool,
        case_sensitive: bool,
        whole_word: bool,
        include_hidden: bool,
    ) -> List[SearchResult]:
        """Search using ripgrep."""
        cmd = ["rg", "--json"]
        
        # Context
        cmd.extend(["-B", str(self.context_lines), "-A", str(self.context_lines)])
        
        # Options
        if not is_regex:
            cmd.append("-F")  # Fixed string
        if not case_sensitive:
            cmd.append("-i")
        if whole_word:
            cmd.append("-w")
        if include_hidden:
            cmd.append("--hidden")
        
        # Max results
        cmd.extend(["-m", str(self.max_results)])
        
        # File type filtering
        if file_pattern:
            cmd.extend(["-g", file_pattern])
        if extensions:
            for ext in extensions:
                cmd.extend(["-g", f"*{ext}"])
        
        # Pattern and path
        cmd.append(pattern)
        cmd.append(str(search_path))
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            logger.warning("ripgrep search timed out")
            return []
        except Exception as e:
            logger.warning(f"ripgrep error: {e}")
            return []
        
        # Parse JSON output
        return self._parse_ripgrep_output(result.stdout)
    
    def _parse_ripgrep_output(self, output: str) -> List[SearchResult]:
        """Parse ripgrep JSON output."""
        import json
        
        results = []
        context_buffer: Dict[str, List[str]] = {}
        current_path: Optional[str] = None
        
        for line in output.strip().split("\n"):
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            msg_type = data.get("type")
            
            if msg_type == "begin":
                current_path = data.get("data", {}).get("path", {}).get("text")
                context_buffer[current_path] = []
            
            elif msg_type == "context":
                if current_path:
                    text = data.get("data", {}).get("lines", {}).get("text", "")
                    context_buffer.get(current_path, []).append(text.rstrip("\n\r"))
            
            elif msg_type == "match":
                match_data = data.get("data", {})
                path_text = match_data.get("path", {}).get("text", "")
                line_number = match_data.get("line_number", 0)
                lines = match_data.get("lines", {}).get("text", "")
                
                # Get match positions
                submatches = match_data.get("submatches", [])
                match_text = ""
                column = 0
                if submatches:
                    match_text = submatches[0].get("match", {}).get("text", "")
                    column = submatches[0].get("start", 0)
                
                result = SearchResult(
                    file_path=Path(path_text),
                    line_number=line_number,
                    column=column,
                    line_content=lines.rstrip("\n\r"),
                    match_text=match_text,
                    context_before=context_buffer.get(path_text, [])[-self.context_lines:],
                )
                results.append(result)
                context_buffer[path_text] = []
            
            elif msg_type == "end":
                current_path = None
        
        return results
    
    def _search_python(
        self,
        pattern: str,
        search_path: Path,
        file_pattern: Optional[str],
        extensions: Optional[List[str]],
        is_regex: bool,
        case_sensitive: bool,
        whole_word: bool,
        include_hidden: bool,
    ) -> List[SearchResult]:
        """Search using Python (fallback)."""
        results = []
        
        # Build regex
        if is_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                logger.warning(f"Invalid regex: {e}")
                return []
        else:
            escaped = re.escape(pattern)
            if whole_word:
                escaped = rf"\b{escaped}\b"
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(escaped, flags)
        
        # Find files
        for file_path in self._find_files(
            search_path, file_pattern, extensions, include_hidden
        ):
            if len(results) >= self.max_results:
                break
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except (OSError, IOError):
                continue
            
            for i, line in enumerate(lines):
                if len(results) >= self.max_results:
                    break
                
                match = regex.search(line)
                if match:
                    result = SearchResult(
                        file_path=file_path,
                        line_number=i + 1,
                        column=match.start(),
                        line_content=line.rstrip("\n\r"),
                        match_text=match.group(),
                        context_before=[
                            lines[j].rstrip("\n\r")
                            for j in range(max(0, i - self.context_lines), i)
                        ],
                        context_after=[
                            lines[j].rstrip("\n\r")
                            for j in range(i + 1, min(len(lines), i + 1 + self.context_lines))
                        ],
                    )
                    results.append(result)
        
        return results
    
    def _find_files(
        self,
        search_path: Path,
        file_pattern: Optional[str],
        extensions: Optional[List[str]],
        include_hidden: bool,
    ) -> Generator[Path, None, None]:
        """Find files to search."""
        for root, dirs, files in os.walk(search_path):
            # Filter hidden directories
            if not include_hidden:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
            
            # Skip common ignore directories
            dirs[:] = [
                d for d in dirs
                if d not in {"__pycache__", "node_modules", ".git", ".hg", "venv", ".venv"}
            ]
            
            for name in files:
                # Skip hidden files
                if not include_hidden and name.startswith("."):
                    continue
                
                path = Path(root) / name
                
                # Check pattern
                if file_pattern:
                    import fnmatch
                    if not fnmatch.fnmatch(name, file_pattern):
                        continue
                
                # Check extension
                if extensions:
                    if path.suffix.lower() not in extensions:
                        continue
                
                yield path


class PythonCodeAnalyzer:
    """Analyze Python code using AST.
    
    Features:
    - Extract functions, classes, methods
    - Find imports and dependencies
    - Calculate code metrics
    - Detect code patterns
    
    Example:
        >>> analyzer = PythonCodeAnalyzer()
        >>> 
        >>> # Analyze a file
        >>> elements = analyzer.analyze_file(Path("main.py"))
        >>> for elem in elements:
        ...     print(f"{elem.element_type}: {elem.name}")
        >>> 
        >>> # Get metrics
        >>> metrics = analyzer.get_metrics(Path("main.py"))
        >>> print(f"Lines: {metrics.total_lines}")
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze_file(self, file_path: Path) -> List[CodeElement]:
        """Analyze a Python file and extract code elements.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of CodeElement objects
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except (OSError, IOError) as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []
        
        elements = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                element = self._extract_function(node, file_path)
                elements.append(element)
            
            elif isinstance(node, ast.AsyncFunctionDef):
                element = self._extract_function(node, file_path, is_async=True)
                elements.append(element)
            
            elif isinstance(node, ast.ClassDef):
                element = self._extract_class(node, file_path)
                elements.append(element)
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method = self._extract_function(
                            item, file_path, parent=node.name
                        )
                        method.element_type = CodeElementType.METHOD
                        elements.append(method)
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    elements.append(CodeElement(
                        name=alias.name,
                        element_type=CodeElementType.IMPORT,
                        file_path=file_path,
                        line_number=node.lineno,
                        end_line=node.lineno,
                    ))
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    elements.append(CodeElement(
                        name=f"{module}.{alias.name}",
                        element_type=CodeElementType.IMPORT,
                        file_path=file_path,
                        line_number=node.lineno,
                        end_line=node.lineno,
                    ))
        
        return elements
    
    def _extract_function(
        self,
        node: ast.FunctionDef,
        file_path: Path,
        parent: Optional[str] = None,
        is_async: bool = False,
    ) -> CodeElement:
        """Extract function information."""
        # Get signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"
        
        prefix = "async " if is_async else ""
        signature = f"{prefix}def {node.name}({', '.join(args)}){returns}"
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass
        
        return CodeElement(
            name=node.name,
            element_type=CodeElementType.FUNCTION,
            file_path=file_path,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=signature,
            parent=parent,
            decorators=decorators,
        )
    
    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
    ) -> CodeElement:
        """Extract class information."""
        # Get bases
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                pass
        
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass
        
        return CodeElement(
            name=node.name,
            element_type=CodeElementType.CLASS,
            file_path=file_path,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=signature,
            decorators=decorators,
        )
    
    def get_metrics(self, file_path: Path) -> CodeMetrics:
        """Calculate code metrics for a file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            CodeMetrics object
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (OSError, IOError) as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return CodeMetrics(
                file_path=file_path,
                total_lines=0,
                code_lines=0,
                comment_lines=0,
                blank_lines=0,
                functions=0,
                classes=0,
                imports=0,
                complexity=0,
            )
        
        total_lines = len(lines)
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        in_multiline_string = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                blank_lines += 1
            elif stripped.startswith("#"):
                comment_lines += 1
            elif '"""' in stripped or "'''" in stripped:
                # Simplified multiline string detection
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_multiline_string = not in_multiline_string
                comment_lines += 1
            elif in_multiline_string:
                comment_lines += 1
            else:
                code_lines += 1
        
        # Get elements for counts
        elements = self.analyze_file(file_path)
        
        functions = sum(
            1 for e in elements
            if e.element_type in (CodeElementType.FUNCTION, CodeElementType.METHOD)
        )
        classes = sum(
            1 for e in elements
            if e.element_type == CodeElementType.CLASS
        )
        imports = sum(
            1 for e in elements
            if e.element_type == CodeElementType.IMPORT
        )
        
        # Simplified complexity (count of branches)
        complexity = self._calculate_complexity(file_path)
        
        return CodeMetrics(
            file_path=file_path,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            functions=functions,
            classes=classes,
            imports=imports,
            complexity=complexity,
        )
    
    def _calculate_complexity(self, file_path: Path) -> int:
        """Calculate McCabe complexity approximation."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
        except Exception:
            return 0
        
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Count decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
            elif isinstance(node, ast.Assert):
                complexity += 1
        
        return complexity
    
    def find_dependencies(self, file_path: Path) -> List[str]:
        """Find import dependencies of a file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of module names
        """
        elements = self.analyze_file(file_path)
        
        dependencies = []
        for elem in elements:
            if elem.element_type == CodeElementType.IMPORT:
                # Get root module
                module = elem.name.split(".")[0]
                if module not in dependencies:
                    dependencies.append(module)
        
        return dependencies


def get_file_content_search(
    project_root: Optional[Path] = None,
) -> FileContentSearch:
    """Get a FileContentSearch instance."""
    return FileContentSearch(project_root)


def get_python_analyzer() -> PythonCodeAnalyzer:
    """Get a PythonCodeAnalyzer instance."""
    return PythonCodeAnalyzer()
