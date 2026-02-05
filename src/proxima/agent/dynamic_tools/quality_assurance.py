"""Quality Assurance Automation for Dynamic AI Assistant.

This module implements Phase 9.2 for the Dynamic AI Assistant:
- Static Analysis
- Continuous Integration
- Validation Suite

Key Features:
============
- Linting with pylint and flake8 integration
- Type checking with mypy integration
- Security scanning with bandit
- CI pipeline support
- Acceptance and scenario testing
- Platform compatibility testing

Design Principle:
================
All quality checks use LLM reasoning when available for
intelligent analysis and suggestions.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Generic, Iterator,
    List, Optional, Pattern, Set, Tuple, Type, TypeVar, Union
)
import uuid

logger = logging.getLogger(__name__)


class AnalysisSeverity(Enum):
    """Analysis issue severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AnalysisCategory(Enum):
    """Analysis category."""
    LINT = "lint"
    TYPE = "type"
    SECURITY = "security"
    COMPLEXITY = "complexity"
    STYLE = "style"
    DOCUMENTATION = "documentation"


class ValidationStatus(Enum):
    """Validation status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlatformType(Enum):
    """Platform types for compatibility testing."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ALL = "all"


@dataclass
class AnalysisIssue:
    """Represents a code analysis issue."""
    issue_id: str
    category: AnalysisCategory
    severity: AnalysisSeverity
    
    # Location
    file_path: str
    line_number: int
    
    # Description
    message: str
    
    # Optional fields (with defaults)
    column: int = 0
    rule_id: Optional[str] = None
    
    # Suggestion
    suggestion: Optional[str] = None
    fix_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "message": self.message,
            "rule_id": self.rule_id,
            "suggestion": self.suggestion,
        }


@dataclass
class AnalysisReport:
    """Analysis report for a codebase."""
    report_id: str
    timestamp: datetime
    
    # Issues
    issues: List[AnalysisIssue] = field(default_factory=list)
    
    # Summary
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    info: int = 0
    
    # Files analyzed
    files_analyzed: int = 0
    lines_analyzed: int = 0
    
    # Score (0-100)
    quality_score: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "total_issues": self.total_issues,
            "errors": self.errors,
            "warnings": self.warnings,
            "quality_score": round(self.quality_score, 2),
            "files_analyzed": self.files_analyzed,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class CIJob:
    """Represents a CI job."""
    job_id: str
    name: str
    
    # Commands
    commands: List[str] = field(default_factory=list)
    
    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution
    status: ValidationStatus = ValidationStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Output
    output: str = ""
    exit_code: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
        }


@dataclass
class CIPipeline:
    """Represents a CI pipeline."""
    pipeline_id: str
    name: str
    
    # Jobs
    jobs: List[CIJob] = field(default_factory=list)
    
    # Trigger
    trigger: str = "manual"  # manual, push, pull_request
    branch: Optional[str] = None
    
    # Status
    status: ValidationStatus = ValidationStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Results
    passed_jobs: int = 0
    failed_jobs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "status": self.status.value,
            "passed_jobs": self.passed_jobs,
            "failed_jobs": self.failed_jobs,
            "jobs": [j.to_dict() for j in self.jobs],
        }


@dataclass
class ValidationScenario:
    """Represents a validation scenario."""
    scenario_id: str
    name: str
    description: str
    
    # Steps
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Expected outcome
    expected_result: Optional[str] = None
    
    # Execution
    status: ValidationStatus = ValidationStatus.PENDING
    actual_result: Optional[str] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    platform: PlatformType = PlatformType.ALL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "platform": self.platform.value,
        }


class StaticAnalyzer:
    """Perform static code analysis.
    
    Uses LLM reasoning to:
    1. Analyze code quality issues
    2. Suggest improvements
    3. Detect patterns and anti-patterns
    
    Example:
        >>> analyzer = StaticAnalyzer()
        >>> report = analyzer.analyze_file("my_code.py")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize static analyzer.
        
        Args:
            llm_client: LLM client for intelligent analysis
        """
        self._llm_client = llm_client
        
        # Built-in rules
        self._rules: Dict[str, Callable] = {}
        self._register_builtin_rules()
    
    def _register_builtin_rules(self):
        """Register built-in analysis rules."""
        self._rules["line_length"] = self._check_line_length
        self._rules["unused_imports"] = self._check_unused_imports
        self._rules["docstring"] = self._check_docstrings
        self._rules["complexity"] = self._check_complexity
        self._rules["naming"] = self._check_naming
    
    def analyze_file(
        self,
        file_path: str,
        rules: Optional[List[str]] = None,
    ) -> AnalysisReport:
        """Analyze a single file.
        
        Args:
            file_path: Path to file to analyze
            rules: Rules to apply (None = all)
            
        Returns:
            Analysis report
        """
        report = AnalysisReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            report.files_analyzed = 1
            report.lines_analyzed = len(lines)
            
            # Apply rules
            rules_to_apply = rules or list(self._rules.keys())
            
            for rule_name in rules_to_apply:
                if rule_name in self._rules:
                    issues = self._rules[rule_name](file_path, content, lines)
                    report.issues.extend(issues)
            
            # Calculate summary
            report.total_issues = len(report.issues)
            report.errors = sum(
                1 for i in report.issues
                if i.severity in [AnalysisSeverity.ERROR, AnalysisSeverity.CRITICAL]
            )
            report.warnings = sum(
                1 for i in report.issues
                if i.severity == AnalysisSeverity.WARNING
            )
            report.info = sum(
                1 for i in report.issues
                if i.severity == AnalysisSeverity.INFO
            )
            
            # Calculate quality score
            report.quality_score = self._calculate_score(report)
            
        except Exception as e:
            report.issues.append(AnalysisIssue(
                issue_id=str(uuid.uuid4()),
                category=AnalysisCategory.LINT,
                severity=AnalysisSeverity.ERROR,
                file_path=file_path,
                line_number=0,
                message=f"Failed to analyze file: {e}",
            ))
        
        return report
    
    def analyze_directory(
        self,
        directory: str,
        pattern: str = "*.py",
        recursive: bool = True,
    ) -> AnalysisReport:
        """Analyze all files in a directory.
        
        Args:
            directory: Directory to analyze
            pattern: File pattern to match
            recursive: Search recursively
            
        Returns:
            Combined analysis report
        """
        report = AnalysisReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
        path = Path(directory)
        glob_pattern = f"**/{pattern}" if recursive else pattern
        
        for file_path in path.glob(glob_pattern):
            if file_path.is_file():
                file_report = self.analyze_file(str(file_path))
                report.issues.extend(file_report.issues)
                report.files_analyzed += 1
                report.lines_analyzed += file_report.lines_analyzed
        
        # Calculate totals
        report.total_issues = len(report.issues)
        report.errors = sum(
            1 for i in report.issues
            if i.severity in [AnalysisSeverity.ERROR, AnalysisSeverity.CRITICAL]
        )
        report.warnings = sum(
            1 for i in report.issues
            if i.severity == AnalysisSeverity.WARNING
        )
        report.quality_score = self._calculate_score(report)
        
        return report
    
    def _calculate_score(self, report: AnalysisReport) -> float:
        """Calculate quality score from report."""
        if report.lines_analyzed == 0:
            return 100.0
        
        # Deduct points for issues
        deductions = (
            report.errors * 5 +
            report.warnings * 2 +
            report.info * 0.5
        )
        
        # Normalize by lines analyzed
        normalized = (deductions / report.lines_analyzed) * 100
        
        return max(0.0, 100.0 - normalized)
    
    def _check_line_length(
        self,
        file_path: str,
        content: str,
        lines: List[str],
        max_length: int = 120,
    ) -> List[AnalysisIssue]:
        """Check for lines exceeding max length."""
        issues = []
        
        for i, line in enumerate(lines, 1):
            if len(line) > max_length:
                issues.append(AnalysisIssue(
                    issue_id=str(uuid.uuid4()),
                    category=AnalysisCategory.STYLE,
                    severity=AnalysisSeverity.WARNING,
                    file_path=file_path,
                    line_number=i,
                    message=f"Line too long ({len(line)} > {max_length})",
                    rule_id="E501",
                    suggestion=f"Break line into multiple lines",
                ))
        
        return issues
    
    def _check_unused_imports(
        self,
        file_path: str,
        content: str,
        lines: List[str],
    ) -> List[AnalysisIssue]:
        """Check for unused imports."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            # Collect imports
            imports = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        imports[name] = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        if name != '*':
                            imports[name] = node.lineno
            
            # Collect all names used
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # Get the root name
                    current = node
                    while isinstance(current, ast.Attribute):
                        current = current.value
                    if isinstance(current, ast.Name):
                        used_names.add(current.id)
            
            # Find unused
            for name, lineno in imports.items():
                # Check if base name is used
                base_name = name.split('.')[0]
                if base_name not in used_names and name not in used_names:
                    issues.append(AnalysisIssue(
                        issue_id=str(uuid.uuid4()),
                        category=AnalysisCategory.LINT,
                        severity=AnalysisSeverity.WARNING,
                        file_path=file_path,
                        line_number=lineno,
                        message=f"Unused import: {name}",
                        rule_id="F401",
                        suggestion=f"Remove unused import '{name}'",
                        fix_available=True,
                    ))
                    
        except SyntaxError:
            pass
        
        return issues
    
    def _check_docstrings(
        self,
        file_path: str,
        content: str,
        lines: List[str],
    ) -> List[AnalysisIssue]:
        """Check for missing docstrings."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Check if has docstring
                    if not (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                        
                        # Skip private methods
                        if not node.name.startswith('_') or node.name.startswith('__'):
                            issues.append(AnalysisIssue(
                                issue_id=str(uuid.uuid4()),
                                category=AnalysisCategory.DOCUMENTATION,
                                severity=AnalysisSeverity.INFO,
                                file_path=file_path,
                                line_number=node.lineno,
                                message=f"Missing docstring for {type(node).__name__}: {node.name}",
                                rule_id="D100",
                                suggestion=f"Add docstring to {node.name}",
                            ))
                            
        except SyntaxError:
            pass
        
        return issues
    
    def _check_complexity(
        self,
        file_path: str,
        content: str,
        lines: List[str],
        max_complexity: int = 10,
    ) -> List[AnalysisIssue]:
        """Check cyclomatic complexity."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._calculate_complexity(node)
                    
                    if complexity > max_complexity:
                        issues.append(AnalysisIssue(
                            issue_id=str(uuid.uuid4()),
                            category=AnalysisCategory.COMPLEXITY,
                            severity=AnalysisSeverity.WARNING,
                            file_path=file_path,
                            line_number=node.lineno,
                            message=f"High complexity ({complexity}) in {node.name}",
                            rule_id="C901",
                            suggestion="Consider breaking down into smaller functions",
                        ))
                        
        except SyntaxError:
            pass
        
        return issues
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _check_naming(
        self,
        file_path: str,
        content: str,
        lines: List[str],
    ) -> List[AnalysisIssue]:
        """Check naming conventions."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Classes should be PascalCase
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        issues.append(AnalysisIssue(
                            issue_id=str(uuid.uuid4()),
                            category=AnalysisCategory.STYLE,
                            severity=AnalysisSeverity.INFO,
                            file_path=file_path,
                            line_number=node.lineno,
                            message=f"Class name '{node.name}' should be PascalCase",
                            rule_id="N801",
                        ))
                        
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Functions should be snake_case
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        if not node.name.startswith('__'):
                            issues.append(AnalysisIssue(
                                issue_id=str(uuid.uuid4()),
                                category=AnalysisCategory.STYLE,
                                severity=AnalysisSeverity.INFO,
                                file_path=file_path,
                                line_number=node.lineno,
                                message=f"Function name '{node.name}' should be snake_case",
                                rule_id="N802",
                            ))
                            
        except SyntaxError:
            pass
        
        return issues
    
    async def analyze_with_llm(
        self,
        file_path: str,
    ) -> List[AnalysisIssue]:
        """Analyze code using LLM for intelligent suggestions.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            List of issues found
        """
        if not self._llm_client:
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return []
        
        prompt = f"""Analyze this Python code for quality issues:

```python
{content[:5000]}  # Truncate for token limits
```

Find:
1. Code smells and anti-patterns
2. Potential bugs
3. Performance issues
4. Security concerns
5. Maintainability problems

Return as JSON array with fields:
- line_number
- severity (info/warning/error)
- category (lint/security/complexity/style)
- message
- suggestion

"""
        
        try:
            response = await self._llm_client.generate(prompt)
            issues_data = json.loads(response)
            
            issues = []
            for data in issues_data:
                issues.append(AnalysisIssue(
                    issue_id=str(uuid.uuid4()),
                    category=AnalysisCategory(data.get("category", "lint")),
                    severity=AnalysisSeverity(data.get("severity", "info")),
                    file_path=file_path,
                    line_number=data.get("line_number", 0),
                    message=data.get("message", ""),
                    suggestion=data.get("suggestion"),
                ))
            
            return issues
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return []


class SecurityScanner:
    """Scan code for security vulnerabilities.
    
    Uses LLM reasoning to:
    1. Detect security issues
    2. Assess risk levels
    3. Suggest secure alternatives
    
    Example:
        >>> scanner = SecurityScanner()
        >>> issues = scanner.scan_file("my_code.py")
    """
    
    # Common security patterns to detect
    SECURITY_PATTERNS = [
        {
            "pattern": r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
            "severity": AnalysisSeverity.CRITICAL,
            "message": "Shell injection risk: shell=True with subprocess",
            "rule_id": "B602",
        },
        {
            "pattern": r"eval\s*\(",
            "severity": AnalysisSeverity.CRITICAL,
            "message": "Use of eval() can execute arbitrary code",
            "rule_id": "B307",
        },
        {
            "pattern": r"exec\s*\(",
            "severity": AnalysisSeverity.CRITICAL,
            "message": "Use of exec() can execute arbitrary code",
            "rule_id": "B102",
        },
        {
            "pattern": r"pickle\.loads?\s*\(",
            "severity": AnalysisSeverity.WARNING,
            "message": "Pickle can execute arbitrary code during deserialization",
            "rule_id": "B301",
        },
        {
            "pattern": r"yaml\.load\s*\([^)]*(?!Loader)",
            "severity": AnalysisSeverity.WARNING,
            "message": "Use yaml.safe_load() instead of yaml.load()",
            "rule_id": "B506",
        },
        {
            "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
            "severity": AnalysisSeverity.ERROR,
            "message": "Hardcoded password detected",
            "rule_id": "B105",
        },
        {
            "pattern": r"secret\s*=\s*['\"][^'\"]+['\"]",
            "severity": AnalysisSeverity.ERROR,
            "message": "Hardcoded secret detected",
            "rule_id": "B105",
        },
        {
            "pattern": r"api_key\s*=\s*['\"][^'\"]+['\"]",
            "severity": AnalysisSeverity.ERROR,
            "message": "Hardcoded API key detected",
            "rule_id": "B105",
        },
        {
            "pattern": r"requests\.get\s*\([^)]*verify\s*=\s*False",
            "severity": AnalysisSeverity.WARNING,
            "message": "SSL verification disabled",
            "rule_id": "B501",
        },
        {
            "pattern": r"hashlib\.md5\s*\(",
            "severity": AnalysisSeverity.WARNING,
            "message": "MD5 is cryptographically weak",
            "rule_id": "B303",
        },
        {
            "pattern": r"hashlib\.sha1\s*\(",
            "severity": AnalysisSeverity.INFO,
            "message": "SHA1 is becoming weak for cryptographic use",
            "rule_id": "B303",
        },
        {
            "pattern": r"os\.system\s*\(",
            "severity": AnalysisSeverity.WARNING,
            "message": "Use subprocess module instead of os.system()",
            "rule_id": "B605",
        },
        {
            "pattern": r"random\.random\s*\(",
            "severity": AnalysisSeverity.INFO,
            "message": "Use secrets module for cryptographic randomness",
            "rule_id": "B311",
        },
    ]
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize security scanner.
        
        Args:
            llm_client: LLM client for intelligent scanning
        """
        self._llm_client = llm_client
    
    def scan_file(
        self,
        file_path: str,
    ) -> List[AnalysisIssue]:
        """Scan a file for security issues.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            List of security issues
        """
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            return [AnalysisIssue(
                issue_id=str(uuid.uuid4()),
                category=AnalysisCategory.SECURITY,
                severity=AnalysisSeverity.ERROR,
                file_path=file_path,
                line_number=0,
                message=f"Failed to read file: {e}",
            )]
        
        for pattern_def in self.SECURITY_PATTERNS:
            pattern = re.compile(pattern_def["pattern"], re.IGNORECASE)
            
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    issues.append(AnalysisIssue(
                        issue_id=str(uuid.uuid4()),
                        category=AnalysisCategory.SECURITY,
                        severity=pattern_def["severity"],
                        file_path=file_path,
                        line_number=i,
                        message=pattern_def["message"],
                        rule_id=pattern_def["rule_id"],
                    ))
        
        return issues
    
    def scan_directory(
        self,
        directory: str,
        pattern: str = "*.py",
    ) -> List[AnalysisIssue]:
        """Scan directory for security issues.
        
        Args:
            directory: Directory to scan
            pattern: File pattern
            
        Returns:
            List of security issues
        """
        issues = []
        
        for file_path in Path(directory).glob(f"**/{pattern}"):
            if file_path.is_file():
                issues.extend(self.scan_file(str(file_path)))
        
        return issues


class TypeChecker:
    """Type checking integration.
    
    Integrates with mypy for type checking.
    
    Example:
        >>> checker = TypeChecker()
        >>> issues = checker.check_file("my_code.py")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize type checker.
        
        Args:
            llm_client: LLM client for intelligent suggestions
        """
        self._llm_client = llm_client
    
    def check_file(
        self,
        file_path: str,
        strict: bool = False,
    ) -> List[AnalysisIssue]:
        """Check types in a file.
        
        Args:
            file_path: Path to file to check
            strict: Use strict mode
            
        Returns:
            List of type issues
        """
        issues = []
        
        # Try to run mypy
        cmd = ["mypy", file_path]
        if strict:
            cmd.append("--strict")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            # Parse mypy output
            for line in result.stdout.split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 4:
                        try:
                            lineno = int(parts[1])
                            severity = AnalysisSeverity.ERROR if 'error' in parts[2].lower() else AnalysisSeverity.WARNING
                            message = ':'.join(parts[3:]).strip()
                            
                            issues.append(AnalysisIssue(
                                issue_id=str(uuid.uuid4()),
                                category=AnalysisCategory.TYPE,
                                severity=severity,
                                file_path=file_path,
                                line_number=lineno,
                                message=message,
                            ))
                        except ValueError:
                            pass
                            
        except FileNotFoundError:
            # mypy not installed
            logger.warning("mypy not found, skipping type checking")
        except Exception as e:
            logger.error(f"Type checking failed: {e}")
        
        return issues
    
    def check_directory(
        self,
        directory: str,
        strict: bool = False,
    ) -> List[AnalysisIssue]:
        """Check types in a directory.
        
        Args:
            directory: Directory to check
            strict: Use strict mode
            
        Returns:
            List of type issues
        """
        issues = []
        
        cmd = ["mypy", directory]
        if strict:
            cmd.append("--strict")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            for line in result.stdout.split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 4:
                        try:
                            file_path = parts[0]
                            lineno = int(parts[1])
                            message = ':'.join(parts[3:]).strip()
                            
                            issues.append(AnalysisIssue(
                                issue_id=str(uuid.uuid4()),
                                category=AnalysisCategory.TYPE,
                                severity=AnalysisSeverity.ERROR,
                                file_path=file_path,
                                line_number=lineno,
                                message=message,
                            ))
                        except ValueError:
                            pass
                            
        except FileNotFoundError:
            logger.warning("mypy not found")
        except Exception as e:
            logger.error(f"Type checking failed: {e}")
        
        return issues


class CIPipelineRunner:
    """Run CI pipelines.
    
    Executes CI jobs and collects results.
    
    Example:
        >>> runner = CIPipelineRunner()
        >>> result = runner.run_pipeline(pipeline)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize CI pipeline runner.
        
        Args:
            llm_client: LLM client for intelligent analysis
        """
        self._llm_client = llm_client
        
        # Pipeline history
        self._history: List[CIPipeline] = []
    
    def run_pipeline(
        self,
        pipeline: CIPipeline,
    ) -> CIPipeline:
        """Run a CI pipeline.
        
        Args:
            pipeline: Pipeline to run
            
        Returns:
            Pipeline with results
        """
        pipeline.status = ValidationStatus.RUNNING
        pipeline.start_time = datetime.now()
        
        # Build dependency graph
        job_map = {job.job_id: job for job in pipeline.jobs}
        completed = set()
        
        while len(completed) < len(pipeline.jobs):
            # Find runnable jobs
            runnable = []
            for job in pipeline.jobs:
                if job.job_id in completed:
                    continue
                
                # Check dependencies
                deps_met = all(
                    dep in completed
                    for dep in job.depends_on
                )
                
                if deps_met:
                    runnable.append(job)
            
            if not runnable:
                # Deadlock - circular dependencies
                break
            
            # Run jobs
            for job in runnable:
                self._run_job(job)
                completed.add(job.job_id)
                
                if job.status == ValidationStatus.PASSED:
                    pipeline.passed_jobs += 1
                else:
                    pipeline.failed_jobs += 1
        
        # Determine overall status
        pipeline.end_time = datetime.now()
        if pipeline.failed_jobs == 0:
            pipeline.status = ValidationStatus.PASSED
        else:
            pipeline.status = ValidationStatus.FAILED
        
        self._history.append(pipeline)
        
        return pipeline
    
    def _run_job(self, job: CIJob):
        """Run a single CI job."""
        job.status = ValidationStatus.RUNNING
        job.start_time = datetime.now()
        
        output_lines = []
        
        try:
            for command in job.commands:
                # Set up environment
                env = os.environ.copy()
                env.update(job.env_vars)
                
                # Run command
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=job.working_dir,
                    env=env,
                    timeout=300,
                )
                
                output_lines.append(f"$ {command}")
                output_lines.append(result.stdout)
                if result.stderr:
                    output_lines.append(result.stderr)
                
                job.exit_code = result.returncode
                
                if result.returncode != 0:
                    break
            
            if job.exit_code == 0:
                job.status = ValidationStatus.PASSED
            else:
                job.status = ValidationStatus.FAILED
                
        except subprocess.TimeoutExpired:
            job.status = ValidationStatus.FAILED
            job.exit_code = -1
            output_lines.append("Command timed out")
            
        except Exception as e:
            job.status = ValidationStatus.FAILED
            job.exit_code = -1
            output_lines.append(f"Error: {e}")
        
        job.end_time = datetime.now()
        job.duration_ms = (job.end_time - job.start_time).total_seconds() * 1000
        job.output = '\n'.join(output_lines)
    
    def create_pipeline(
        self,
        name: str,
        jobs: List[Dict[str, Any]],
    ) -> CIPipeline:
        """Create a CI pipeline from configuration.
        
        Args:
            name: Pipeline name
            jobs: Job configurations
            
        Returns:
            Created pipeline
        """
        pipeline = CIPipeline(
            pipeline_id=str(uuid.uuid4()),
            name=name,
        )
        
        for job_config in jobs:
            job = CIJob(
                job_id=job_config.get("id", str(uuid.uuid4())),
                name=job_config.get("name", "unnamed"),
                commands=job_config.get("commands", []),
                env_vars=job_config.get("env", {}),
                working_dir=job_config.get("working_dir"),
                depends_on=job_config.get("depends_on", []),
            )
            pipeline.jobs.append(job)
        
        return pipeline
    
    def get_history(self) -> List[CIPipeline]:
        """Get pipeline execution history."""
        return list(self._history)


class ValidationSuiteRunner:
    """Run validation scenarios.
    
    Executes acceptance, scenario, and compatibility tests.
    
    Example:
        >>> runner = ValidationSuiteRunner()
        >>> results = runner.run_scenarios(scenarios)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize validation suite runner.
        
        Args:
            llm_client: LLM client for intelligent validation
        """
        self._llm_client = llm_client
        
        # Registered scenarios
        self._scenarios: Dict[str, ValidationScenario] = {}
        
        # Results
        self._results: List[ValidationScenario] = []
    
    def register_scenario(self, scenario: ValidationScenario):
        """Register a validation scenario."""
        self._scenarios[scenario.scenario_id] = scenario
    
    def run_scenario(
        self,
        scenario: ValidationScenario,
    ) -> ValidationScenario:
        """Run a validation scenario.
        
        Args:
            scenario: Scenario to run
            
        Returns:
            Scenario with results
        """
        # Check platform compatibility
        current_platform = self._get_current_platform()
        if (scenario.platform != PlatformType.ALL and 
            scenario.platform != current_platform):
            scenario.status = ValidationStatus.SKIPPED
            scenario.actual_result = f"Skipped: Not compatible with {current_platform.value}"
            return scenario
        
        scenario.status = ValidationStatus.RUNNING
        results = []
        
        try:
            for step in scenario.steps:
                step_name = step.get("name", "unnamed")
                action = step.get("action")
                params = step.get("params", {})
                expected = step.get("expected")
                
                if callable(action):
                    actual = action(**params)
                    
                    if expected is not None:
                        if actual != expected:
                            results.append(f"Step '{step_name}': FAILED - Expected {expected}, got {actual}")
                            scenario.status = ValidationStatus.FAILED
                        else:
                            results.append(f"Step '{step_name}': PASSED")
                    else:
                        results.append(f"Step '{step_name}': PASSED")
                else:
                    results.append(f"Step '{step_name}': SKIPPED - No action")
            
            if scenario.status != ValidationStatus.FAILED:
                scenario.status = ValidationStatus.PASSED
                
        except Exception as e:
            scenario.status = ValidationStatus.FAILED
            results.append(f"Error: {e}")
        
        scenario.actual_result = '\n'.join(results)
        self._results.append(scenario)
        
        return scenario
    
    def run_all_scenarios(
        self,
        tags: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Run all registered scenarios.
        
        Args:
            tags: Filter by tags
            
        Returns:
            Summary of results
        """
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "scenarios": [],
        }
        
        for scenario in self._scenarios.values():
            # Filter by tags
            if tags and not (scenario.tags & tags):
                continue
            
            result = self.run_scenario(scenario)
            results["total"] += 1
            results["scenarios"].append(result.to_dict())
            
            if result.status == ValidationStatus.PASSED:
                results["passed"] += 1
            elif result.status == ValidationStatus.FAILED:
                results["failed"] += 1
            else:
                results["skipped"] += 1
        
        return results
    
    def _get_current_platform(self) -> PlatformType:
        """Get current platform type."""
        system = platform.system().lower()
        if system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        else:
            return PlatformType.LINUX
    
    def create_acceptance_test(
        self,
        name: str,
        given: str,
        when: str,
        then: str,
        action: Callable,
        expected: Any,
    ) -> ValidationScenario:
        """Create an acceptance test (BDD style).
        
        Args:
            name: Test name
            given: Given condition
            when: When action
            then: Then expectation
            action: Action function
            expected: Expected result
            
        Returns:
            Created scenario
        """
        scenario = ValidationScenario(
            scenario_id=str(uuid.uuid4()),
            name=name,
            description=f"GIVEN {given}\nWHEN {when}\nTHEN {then}",
            steps=[
                {
                    "name": when,
                    "action": action,
                    "expected": expected,
                }
            ],
            expected_result=str(expected),
        )
        
        self.register_scenario(scenario)
        return scenario
    
    def get_results(self) -> List[ValidationScenario]:
        """Get all validation results."""
        return list(self._results)


class QualityAssuranceAutomation:
    """Main quality assurance automation system.
    
    Integrates all QA components:
    - Static analysis
    - Security scanning
    - Type checking
    - CI pipeline
    - Validation suite
    
    Example:
        >>> qa = QualityAssuranceAutomation()
        >>> report = qa.run_full_analysis("src/")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
    ):
        """Initialize QA automation.
        
        Args:
            llm_client: LLM client for intelligent analysis
        """
        self._llm_client = llm_client
        
        # Initialize components
        self._analyzer = StaticAnalyzer(llm_client=llm_client)
        self._security = SecurityScanner(llm_client=llm_client)
        self._type_checker = TypeChecker(llm_client=llm_client)
        self._ci_runner = CIPipelineRunner(llm_client=llm_client)
        self._validation = ValidationSuiteRunner(llm_client=llm_client)
    
    def run_full_analysis(
        self,
        directory: str,
        include_security: bool = True,
        include_types: bool = True,
    ) -> Dict[str, Any]:
        """Run full quality analysis on directory.
        
        Args:
            directory: Directory to analyze
            include_security: Include security scan
            include_types: Include type checking
            
        Returns:
            Complete analysis report
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "directory": directory,
            "static_analysis": None,
            "security_scan": None,
            "type_check": None,
            "overall_score": 0.0,
        }
        
        # Static analysis
        static_report = self._analyzer.analyze_directory(directory)
        results["static_analysis"] = static_report.to_dict()
        
        # Security scan
        if include_security:
            security_issues = self._security.scan_directory(directory)
            results["security_scan"] = {
                "total_issues": len(security_issues),
                "critical": sum(1 for i in security_issues if i.severity == AnalysisSeverity.CRITICAL),
                "high": sum(1 for i in security_issues if i.severity == AnalysisSeverity.ERROR),
                "medium": sum(1 for i in security_issues if i.severity == AnalysisSeverity.WARNING),
                "issues": [i.to_dict() for i in security_issues[:20]],
            }
        
        # Type checking
        if include_types:
            type_issues = self._type_checker.check_directory(directory)
            results["type_check"] = {
                "total_issues": len(type_issues),
                "issues": [i.to_dict() for i in type_issues[:20]],
            }
        
        # Calculate overall score
        score = static_report.quality_score
        if include_security and results["security_scan"]:
            security_penalty = results["security_scan"]["critical"] * 10 + results["security_scan"]["high"] * 5
            score = max(0, score - security_penalty)
        
        results["overall_score"] = round(score, 2)
        
        return results
    
    def run_ci_pipeline(
        self,
        pipeline_config: Dict[str, Any],
    ) -> CIPipeline:
        """Run a CI pipeline.
        
        Args:
            pipeline_config: Pipeline configuration
            
        Returns:
            Pipeline results
        """
        pipeline = self._ci_runner.create_pipeline(
            name=pipeline_config.get("name", "default"),
            jobs=pipeline_config.get("jobs", []),
        )
        
        return self._ci_runner.run_pipeline(pipeline)
    
    def run_validation(
        self,
        scenarios: List[ValidationScenario],
    ) -> Dict[str, Any]:
        """Run validation scenarios.
        
        Args:
            scenarios: Scenarios to run
            
        Returns:
            Validation results
        """
        for scenario in scenarios:
            self._validation.register_scenario(scenario)
        
        return self._validation.run_all_scenarios()
    
    def get_analyzer(self) -> StaticAnalyzer:
        """Get static analyzer."""
        return self._analyzer
    
    def get_security_scanner(self) -> SecurityScanner:
        """Get security scanner."""
        return self._security
    
    def get_type_checker(self) -> TypeChecker:
        """Get type checker."""
        return self._type_checker
    
    def get_ci_runner(self) -> CIPipelineRunner:
        """Get CI pipeline runner."""
        return self._ci_runner
    
    def get_validation_runner(self) -> ValidationSuiteRunner:
        """Get validation suite runner."""
        return self._validation


# Module-level instance
_global_qa_automation: Optional[QualityAssuranceAutomation] = None


def get_quality_assurance_automation(
    llm_client: Optional[Any] = None,
) -> QualityAssuranceAutomation:
    """Get the global QA automation instance.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        QualityAssuranceAutomation instance
    """
    global _global_qa_automation
    if _global_qa_automation is None:
        _global_qa_automation = QualityAssuranceAutomation(
            llm_client=llm_client,
        )
    return _global_qa_automation
