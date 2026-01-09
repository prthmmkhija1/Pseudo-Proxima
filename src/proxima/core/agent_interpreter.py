"""Step 5.2: Agent.md Interpreter - Parse and execute agent.md files.

File Parser:
1. Read file content
2. Parse as Markdown
3. Extract metadata section
4. Extract configuration section
5. Parse task definitions
6. Validate task parameters
7. Build execution plan

Task Execution:
FOR each task in agent_file.tasks:
    1. Display task description
    2. Request consent for sensitive operations
    3. Create task execution plan
    4. Execute using standard pipeline
    5. Collect results
    6. Continue to next task or stop on error

FINALLY:
    Generate combined report
"""

from __future__ import annotations

import re
import time
import yaml
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from proxima.resources.consent import ConsentManager


class TaskType(Enum):
    """Types of tasks that can be defined in an agent.md file."""
    CIRCUIT_EXECUTION = "circuit_execution"
    BACKEND_COMPARISON = "backend_comparison"
    RESULT_ANALYSIS = "result_analysis"
    EXPORT = "export"
    CUSTOM = "custom"


class TaskStatus(Enum):
    """Status of a task execution."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A validation issue found in the agent.md file."""
    severity: ValidationSeverity
    message: str
    line: Optional[int] = None
    section: Optional[str] = None

    def __str__(self) -> str:
        loc = f"line {self.line}" if self.line else self.section or "unknown"
        return f"[{self.severity.value.upper()}] {loc}: {self.message}"


@dataclass
class AgentMetadata:
    """Metadata extracted from the agent.md file."""
    name: str = "Unnamed Agent"
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    created: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    requires_consent: bool = True
    trusted: bool = False
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "created": self.created,
            "tags": self.tags,
            "requires_consent": self.requires_consent,
            "trusted": self.trusted,
        }


@dataclass
class AgentConfiguration:
    """Configuration section from the agent.md file."""
    default_backend: Optional[str] = None
    backends: List[str] = field(default_factory=list)
    shots: int = 1024
    timeout_seconds: int = 300
    continue_on_error: bool = False
    parallel_execution: bool = False
    output_dir: Optional[str] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "default_backend": self.default_backend,
            "backends": self.backends,
            "shots": self.shots,
            "timeout_seconds": self.timeout_seconds,
            "continue_on_error": self.continue_on_error,
            "parallel_execution": self.parallel_execution,
            "output_dir": self.output_dir,
            "custom_settings": self.custom_settings,
        }


@dataclass
class TaskDefinition:
    """A task definition extracted from the agent.md file."""
    id: str
    name: str
    type: TaskType
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    requires_consent: bool = False
    consent_reason: Optional[str] = None
    timeout_seconds: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "requires_consent": self.requires_consent,
            "consent_reason": self.consent_reason,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class TaskResult:
    """Result of executing a task."""
    task_id: str
    status: TaskStatus
    start_time: float
    end_time: float
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AgentFile:
    """Parsed agent.md file."""
    source_path: Optional[Path]
    raw_content: str
    metadata: AgentMetadata
    configuration: AgentConfiguration
    tasks: List[TaskDefinition]
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return not any(
            issue.severity == ValidationSeverity.ERROR
            for issue in self.validation_issues
        )
    
    @property
    def task_count(self) -> int:
        return len(self.tasks)
    
    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def to_dict(self) -> dict:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "metadata": self.metadata.to_dict(),
            "configuration": self.configuration.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
            "is_valid": self.is_valid,
            "validation_issues": [str(i) for i in self.validation_issues],
        }


@dataclass
class ExecutionReport:
    """Combined report from executing an agent.md file."""
    agent_file: AgentFile
    start_time: float
    end_time: float
    task_results: List[TaskResult]
    status: str  # "completed", "partial", "failed"
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def successful_tasks(self) -> int:
        return sum(1 for r in self.task_results if r.status == TaskStatus.COMPLETED)
    
    @property
    def failed_tasks(self) -> int:
        return sum(1 for r in self.task_results if r.status == TaskStatus.FAILED)
    
    def to_dict(self) -> dict:
        return {
            "agent": self.agent_file.metadata.name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "task_results": [r.to_dict() for r in self.task_results],
            "errors": self.errors,
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"AGENT EXECUTION REPORT: {self.agent_file.metadata.name}",
            "=" * 60,
            f"Status: {self.status.upper()}",
            f"Duration: {self.duration_ms:.2f} ms",
            f"Tasks: {self.successful_tasks} succeeded, {self.failed_tasks} failed",
            "",
            "TASK RESULTS:",
        ]
        for result in self.task_results:
            status_icon = "" if result.status == TaskStatus.COMPLETED else ""
            lines.append(f"  {status_icon} {result.task_id}: {result.status.name} ({result.duration_ms:.2f} ms)")
            if result.error:
                lines.append(f"      Error: {result.error}")
        
        if self.errors:
            lines.extend(["", "ERRORS:"] + [f"  - {e}" for e in self.errors])
        
        lines.append("=" * 60)
        return "\n".join(lines)


class AgentFileParser:
    """Parses agent.md files into structured AgentFile objects.
    
    File Parser Steps:
    1. Read file content
    2. Parse as Markdown
    3. Extract metadata section
    4. Extract configuration section
    5. Parse task definitions
    6. Validate task parameters
    7. Build execution plan
    """
    
    # Regex patterns for parsing
    METADATA_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n',
        re.DOTALL | re.MULTILINE
    )
    SECTION_PATTERN = re.compile(
        r'^##\s+(.+?)\s*$',
        re.MULTILINE
    )
    TASK_PATTERN = re.compile(
        r'^###\s+Task:\s*(.+?)\s*$',
        re.MULTILINE
    )
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w*)\n(.*?)```',
        re.DOTALL
    )
    
    def __init__(self) -> None:
        self.validation_issues: List[ValidationIssue] = []
    
    def parse_file(self, file_path: Path) -> AgentFile:
        """Parse an agent.md file from disk."""
        if not file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {file_path}")
        
        content = file_path.read_text(encoding="utf-8")
        return self.parse_content(content, source_path=file_path)
    
    def parse_content(
        self,
        content: str,
        source_path: Optional[Path] = None,
    ) -> AgentFile:
        """Parse agent.md content string.
        
        Steps:
        1. Read file content (already done)
        2. Parse as Markdown
        3. Extract metadata section
        4. Extract configuration section
        5. Parse task definitions
        6. Validate task parameters
        7. Build execution plan (return AgentFile)
        """
        self.validation_issues = []
        
        # Step 3: Extract metadata section (YAML frontmatter)
        metadata = self._extract_metadata(content)
        
        # Step 4: Extract configuration section
        configuration = self._extract_configuration(content)
        
        # Step 5: Parse task definitions
        tasks = self._extract_tasks(content)
        
        # Step 6: Validate task parameters
        self._validate_tasks(tasks, configuration)
        
        # Step 7: Build and return AgentFile
        return AgentFile(
            source_path=source_path,
            raw_content=content,
            metadata=metadata,
            configuration=configuration,
            tasks=tasks,
            validation_issues=self.validation_issues,
        )
    
    def _extract_metadata(self, content: str) -> AgentMetadata:
        """Extract YAML frontmatter metadata."""
        match = self.METADATA_PATTERN.search(content)
        if not match:
            self.validation_issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No metadata section found (YAML frontmatter)",
                section="metadata"
            ))
            return AgentMetadata()
        
        try:
            yaml_content = match.group(1)
            data = yaml.safe_load(yaml_content) or {}
            
            return AgentMetadata(
                name=data.get("name", "Unnamed Agent"),
                version=str(data.get("version", "1.0.0")),
                description=data.get("description", ""),
                author=data.get("author", ""),
                created=data.get("created"),
                tags=data.get("tags", []),
                requires_consent=data.get("requires_consent", True),
                trusted=data.get("trusted", False),
            )
        except yaml.YAMLError as e:
            self.validation_issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Invalid YAML in metadata: {e}",
                section="metadata"
            ))
            return AgentMetadata()
    
    def _extract_configuration(self, content: str) -> AgentConfiguration:
        """Extract configuration section."""
        # Look for ## Configuration section
        config_match = re.search(
            r'##\s+Configuration\s*\n(.*?)(?=\n##\s|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )
        
        if not config_match:
            return AgentConfiguration()
        
        section_content = config_match.group(1)
        
        # Look for YAML or JSON code block
        code_match = self.CODE_BLOCK_PATTERN.search(section_content)
        if code_match:
            lang = code_match.group(1).lower()
            code_content = code_match.group(2)
            
            try:
                if lang in ("yaml", "yml", ""):
                    data = yaml.safe_load(code_content) or {}
                elif lang == "json":
                    import json
                    data = json.loads(code_content)
                else:
                    data = yaml.safe_load(code_content) or {}
                
                return AgentConfiguration(
                    default_backend=data.get("default_backend"),
                    backends=data.get("backends", []),
                    shots=data.get("shots", 1024),
                    timeout_seconds=data.get("timeout_seconds", 300),
                    continue_on_error=data.get("continue_on_error", False),
                    parallel_execution=data.get("parallel_execution", False),
                    output_dir=data.get("output_dir"),
                    custom_settings=data.get("custom_settings", {}),
                )
            except Exception as e:
                self.validation_issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid configuration: {e}",
                    section="configuration"
                ))
        
        return AgentConfiguration()
    
    def _extract_tasks(self, content: str) -> List[TaskDefinition]:
        """Extract task definitions from ## Tasks section."""
        tasks: List[TaskDefinition] = []
        
        # Look for ## Tasks section
        tasks_match = re.search(
            r'##\s+Tasks\s*\n(.*?)(?=\n##\s|\Z)',
            content,
            re.DOTALL | re.IGNORECASE
        )
        
        if not tasks_match:
            self.validation_issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No Tasks section found",
                section="tasks"
            ))
            return tasks
        
        tasks_content = tasks_match.group(1)
        
        # Find all ### Task: definitions
        task_matches = list(self.TASK_PATTERN.finditer(tasks_content))
        
        for i, match in enumerate(task_matches):
            task_name = match.group(1).strip()
            
            # Get content until next task or end
            start_pos = match.end()
            end_pos = task_matches[i + 1].start() if i + 1 < len(task_matches) else len(tasks_content)
            task_content = tasks_content[start_pos:end_pos]
            
            task = self._parse_task(task_name, task_content, i + 1)
            if task:
                tasks.append(task)
        
        return tasks
    
    def _parse_task(
        self,
        name: str,
        content: str,
        index: int,
    ) -> Optional[TaskDefinition]:
        """Parse a single task definition."""
        task_id = f"task_{index}"
        task_type = TaskType.CUSTOM
        description = ""
        parameters: Dict[str, Any] = {}
        depends_on: List[str] = []
        requires_consent = False
        consent_reason = None
        timeout_seconds = None
        
        # Extract description (first paragraph)
        lines = content.strip().split("\n")
        desc_lines = []
        for line in lines:
            if line.strip() and not line.startswith("```") and not line.startswith("-"):
                desc_lines.append(line.strip())
            else:
                break
        description = " ".join(desc_lines)
        
        # Look for code block with parameters
        code_match = self.CODE_BLOCK_PATTERN.search(content)
        if code_match:
            try:
                params = yaml.safe_load(code_match.group(2)) or {}
                
                # Extract special fields
                task_id = params.pop("id", task_id)
                task_type_str = params.pop("type", "custom")
                try:
                    task_type = TaskType(task_type_str)
                except ValueError:
                    task_type = TaskType.CUSTOM
                
                depends_on = params.pop("depends_on", [])
                if isinstance(depends_on, str):
                    depends_on = [depends_on]
                
                requires_consent = params.pop("requires_consent", False)
                consent_reason = params.pop("consent_reason", None)
                timeout_seconds = params.pop("timeout_seconds", None)
                
                # Remaining are task-specific parameters
                parameters = params
                
            except Exception as e:
                self.validation_issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Could not parse task parameters: {e}",
                    section=f"task:{name}"
                ))
        
        return TaskDefinition(
            id=task_id,
            name=name,
            type=task_type,
            description=description,
            parameters=parameters,
            depends_on=depends_on,
            requires_consent=requires_consent,
            consent_reason=consent_reason,
            timeout_seconds=timeout_seconds,
        )
    
    def _validate_tasks(
        self,
        tasks: List[TaskDefinition],
        config: AgentConfiguration,
    ) -> None:
        """Validate task definitions."""
        task_ids = {t.id for t in tasks}
        
        for task in tasks:
            # Check dependencies exist
            for dep in task.depends_on:
                if dep not in task_ids:
                    self.validation_issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Task '{task.id}' depends on unknown task '{dep}'",
                        section=f"task:{task.name}"
                    ))
            
            # Check circular dependencies (simple check)
            if task.id in task.depends_on:
                self.validation_issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Task '{task.id}' has circular dependency on itself",
                    section=f"task:{task.name}"
                ))
            
            # Validate based on task type
            if task.type == TaskType.CIRCUIT_EXECUTION:
                if "circuit" not in task.parameters and "circuit_file" not in task.parameters:
                    self.validation_issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Circuit execution task '{task.id}' missing circuit parameter",
                        section=f"task:{task.name}"
                    ))
            
            elif task.type == TaskType.BACKEND_COMPARISON:
                if not task.parameters.get("backends") and not config.backends:
                    self.validation_issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Backend comparison task '{task.id}' has no backends specified",
                        section=f"task:{task.name}"
                    ))


class TaskExecutor(Protocol):
    """Protocol for task executors."""
    
    def execute(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute a task and return the result."""
        ...


class DefaultTaskExecutor:
    """Default task executor that handles common task types."""
    
    def __init__(self) -> None:
        self._handlers: Dict[TaskType, Callable] = {
            TaskType.CIRCUIT_EXECUTION: self._execute_circuit,
            TaskType.BACKEND_COMPARISON: self._execute_comparison,
            TaskType.RESULT_ANALYSIS: self._execute_analysis,
            TaskType.EXPORT: self._execute_export,
            TaskType.CUSTOM: self._execute_custom,
        }
    
    def execute(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute a task based on its type."""
        handler = self._handlers.get(task.type, self._execute_custom)
        return handler(task, context)
    
    def _execute_circuit(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute a circuit execution task."""
        # This would integrate with the backend system
        return {
            "task_type": "circuit_execution",
            "parameters": task.parameters,
            "message": "Circuit execution placeholder - integrate with backend adapters",
        }
    
    def _execute_comparison(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute a backend comparison task."""
        return {
            "task_type": "backend_comparison",
            "parameters": task.parameters,
            "message": "Comparison placeholder - integrate with MultiBackendComparator",
        }
    
    def _execute_analysis(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute a result analysis task."""
        return {
            "task_type": "result_analysis",
            "parameters": task.parameters,
            "message": "Analysis placeholder",
        }
    
    def _execute_export(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute an export task."""
        return {
            "task_type": "export",
            "parameters": task.parameters,
            "message": "Export placeholder",
        }
    
    def _execute_custom(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute a custom task."""
        return {
            "task_type": "custom",
            "parameters": task.parameters,
            "message": "Custom task executed",
        }


class AgentInterpreter:
    """Interprets and executes agent.md files.
    
    Task Execution Flow:
    FOR each task in agent_file.tasks:
        1. Display task description
        2. Request consent for sensitive operations
        3. Create task execution plan
        4. Execute using standard pipeline
        5. Collect results
        6. Continue to next task or stop on error
    
    FINALLY:
        Generate combined report
    """
    
    def __init__(
        self,
        parser: Optional[AgentFileParser] = None,
        executor: Optional[TaskExecutor] = None,
        consent_manager: Optional["ConsentManager"] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        display_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize interpreter.
        
        Args:
            parser: File parser (default: create new)
            executor: Task executor (default: DefaultTaskExecutor)
            consent_manager: For requesting consent
            progress_callback: Callback(stage, progress) for progress updates
            display_callback: Callback(message) for displaying task info
        """
        self.parser = parser or AgentFileParser()
        self.executor = executor or DefaultTaskExecutor()
        self.consent_manager = consent_manager
        self.progress_callback = progress_callback
        self.display_callback = display_callback
    
    def _display(self, message: str) -> None:
        """Display a message to the user."""
        if self.display_callback:
            self.display_callback(message)
        else:
            print(message)
    
    def _report_progress(self, stage: str, progress: float) -> None:
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(stage, progress)
    
    def load_file(self, file_path: Path) -> AgentFile:
        """Load and parse an agent.md file."""
        return self.parser.parse_file(file_path)
    
    def load_content(self, content: str) -> AgentFile:
        """Load and parse agent.md content string."""
        return self.parser.parse_content(content)
    
    def _request_consent(
        self,
        task: TaskDefinition,
        agent_file: AgentFile,
    ) -> bool:
        """Request consent for a task if needed."""
        if not task.requires_consent:
            return True
        
        if self.consent_manager is None:
            # No consent manager - ask via display
            self._display(f"\n Task '{task.name}' requires consent")
            if task.consent_reason:
                self._display(f"   Reason: {task.consent_reason}")
            # Default to True if no consent manager
            return True
        
        # Use consent manager
        from proxima.resources.consent import ConsentCategory
        
        topic = f"agent:{agent_file.metadata.name}:task:{task.id}"
        description = task.consent_reason or f"Execute task: {task.name}"
        
        # Untrusted agent files always require consent
        if not agent_file.metadata.trusted:
            return self.consent_manager.request_consent(
                topic=topic,
                category=ConsentCategory.UNTRUSTED_AGENT_MD,
                description=description,
                force_prompt=True,  # Always ask for untrusted
            )
        
        return self.consent_manager.request_consent(
            topic=topic,
            description=description,
        )
    
    def _build_execution_order(
        self,
        tasks: List[TaskDefinition],
    ) -> List[TaskDefinition]:
        """Build execution order respecting dependencies (topological sort)."""
        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: len(t.depends_on) for t in tasks}
        dependents: Dict[str, List[str]] = {t.id: [] for t in tasks}
        
        for task in tasks:
            for dep in task.depends_on:
                if dep in dependents:
                    dependents[dep].append(task.id)
        
        # Kahn's algorithm
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result: List[TaskDefinition] = []
        
        while queue:
            tid = queue.pop(0)
            result.append(task_map[tid])
            
            for dependent in dependents[tid]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(result) != len(tasks):
            # Cycle detected - return original order
            return tasks
        
        return result
    
    def execute(
        self,
        agent_file: AgentFile,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionReport:
        """Execute an agent.md file.
        
        Task Execution Flow:
        1. Display task description
        2. Request consent for sensitive operations
        3. Create task execution plan
        4. Execute using standard pipeline
        5. Collect results
        6. Continue to next task or stop on error
        
        Returns:
            ExecutionReport with all task results
        """
        start_time = time.time()
        context = context or {}
        task_results: List[TaskResult] = []
        errors: List[str] = []
        
        # Validate agent file
        if not agent_file.is_valid:
            return ExecutionReport(
                agent_file=agent_file,
                start_time=start_time,
                end_time=time.time(),
                task_results=[],
                status="failed",
                errors=[str(i) for i in agent_file.validation_issues if i.severity == ValidationSeverity.ERROR],
            )
        
        # Check if consent required for untrusted file
        if agent_file.metadata.requires_consent and not agent_file.metadata.trusted:
            self._display(f"\n Executing untrusted agent: {agent_file.metadata.name}")
            if self.consent_manager:
                from proxima.resources.consent import ConsentCategory
                if not self.consent_manager.request_consent(
                    topic=f"agent:{agent_file.metadata.name}",
                    category=ConsentCategory.UNTRUSTED_AGENT_MD,
                    description=f"Execute agent file: {agent_file.metadata.name}",
                    force_prompt=True,
                ):
                    return ExecutionReport(
                        agent_file=agent_file,
                        start_time=start_time,
                        end_time=time.time(),
                        task_results=[],
                        status="failed",
                        errors=["Consent denied for untrusted agent file"],
                    )
        
        # Build execution order
        ordered_tasks = self._build_execution_order(agent_file.tasks)
        total_tasks = len(ordered_tasks)
        
        self._display(f"\nExecuting agent: {agent_file.metadata.name}")
        self._display(f"Tasks: {total_tasks}")
        self._display("-" * 40)
        
        # Track completed task IDs for dependency checking
        completed_tasks: set = set()
        
        # Execute tasks
        for i, task in enumerate(ordered_tasks):
            task_start = time.time()
            self._report_progress("executing", (i / total_tasks))
            
            # Step 1: Display task description
            self._display(f"\n[{i+1}/{total_tasks}] Task: {task.name}")
            if task.description:
                self._display(f"    {task.description}")
            
            # Check dependencies are completed
            missing_deps = [d for d in task.depends_on if d not in completed_tasks]
            if missing_deps:
                result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.SKIPPED,
                    start_time=task_start,
                    end_time=time.time(),
                    error=f"Dependencies not completed: {missing_deps}",
                )
                task_results.append(result)
                errors.append(f"Task '{task.id}' skipped: missing dependencies")
                
                if not agent_file.configuration.continue_on_error:
                    break
                continue
            
            # Step 2: Request consent for sensitive operations
            if task.requires_consent:
                if not self._request_consent(task, agent_file):
                    result = TaskResult(
                        task_id=task.id,
                        status=TaskStatus.CANCELLED,
                        start_time=task_start,
                        end_time=time.time(),
                        error="Consent denied",
                    )
                    task_results.append(result)
                    errors.append(f"Task '{task.id}' cancelled: consent denied")
                    
                    if not agent_file.configuration.continue_on_error:
                        break
                    continue
            
            # Step 3 & 4: Create execution plan and execute
            try:
                task_context = {
                    **context,
                    "configuration": agent_file.configuration,
                    "previous_results": {r.task_id: r for r in task_results},
                }
                
                task_result_data = self.executor.execute(task, task_context)
                
                result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    start_time=task_start,
                    end_time=time.time(),
                    result=task_result_data,
                )
                task_results.append(result)
                completed_tasks.add(task.id)
                
                self._display(f"     Completed in {result.duration_ms:.2f} ms")
                
            except Exception as e:
                result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    start_time=task_start,
                    end_time=time.time(),
                    error=str(e),
                )
                task_results.append(result)
                errors.append(f"Task '{task.id}' failed: {e}")
                
                self._display(f"     Failed: {e}")
                
                # Step 6: Continue or stop on error
                if not agent_file.configuration.continue_on_error:
                    break
        
        self._report_progress("completed", 1.0)
        
        # Determine final status
        if not task_results:
            status = "failed"
        elif all(r.status == TaskStatus.COMPLETED for r in task_results):
            status = "completed"
        elif any(r.status == TaskStatus.COMPLETED for r in task_results):
            status = "partial"
        else:
            status = "failed"
        
        # Generate combined report
        return ExecutionReport(
            agent_file=agent_file,
            start_time=start_time,
            end_time=time.time(),
            task_results=task_results,
            status=status,
            errors=errors,
        )
    
    def execute_file(
        self,
        file_path: Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionReport:
        """Load and execute an agent.md file."""
        agent_file = self.load_file(file_path)
        return self.execute(agent_file, context)


# Convenience function
def run_agent_file(
    file_path: Path,
    context: Optional[Dict[str, Any]] = None,
    consent_manager: Optional["ConsentManager"] = None,
) -> ExecutionReport:
    """Load and execute an agent.md file.
    
    Args:
        file_path: Path to the agent.md file
        context: Optional context dict for task execution
        consent_manager: Optional consent manager for sensitive operations
        
    Returns:
        ExecutionReport with execution results
    """
    interpreter = AgentInterpreter(consent_manager=consent_manager)
    return interpreter.execute_file(file_path, context)
