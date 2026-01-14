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
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

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
    line: int | None = None
    section: str | None = None

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
    created: str | None = None
    tags: list[str] = field(default_factory=list)
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

    default_backend: str | None = None
    backends: list[str] = field(default_factory=list)
    shots: int = 1024
    timeout_seconds: int = 300
    continue_on_error: bool = False
    parallel_execution: bool = False
    output_dir: str | None = None
    custom_settings: dict[str, Any] = field(default_factory=dict)

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
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    requires_consent: bool = False
    consent_reason: str | None = None
    timeout_seconds: int | None = None
    # Enhanced: Complex task features
    condition: str | None = None  # Condition expression for conditional execution
    retry_count: int = 0  # Number of retries on failure
    retry_delay_seconds: float = 1.0  # Delay between retries
    loop_over: str | None = None  # Variable to iterate over (e.g., "backends")
    loop_variable: str = "item"  # Name of loop iteration variable
    parallel: bool = False  # Execute in parallel with other parallel tasks
    group_id: str | None = None  # Group ID for task grouping
    subtasks: list["TaskDefinition"] = field(default_factory=list)  # Nested subtasks
    on_error: str | None = None  # Task ID to run on error
    on_success: str | None = None  # Task ID to run on success
    priority: int = 0  # Execution priority (higher = earlier)
    tags: list[str] = field(default_factory=list)  # Tags for filtering

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
            "condition": self.condition,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "loop_over": self.loop_over,
            "loop_variable": self.loop_variable,
            "parallel": self.parallel,
            "group_id": self.group_id,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "on_error": self.on_error,
            "on_success": self.on_success,
            "priority": self.priority,
            "tags": self.tags,
        }


@dataclass
class TaskResult:
    """Result of executing a task."""

    task_id: str
    status: TaskStatus
    start_time: float
    end_time: float
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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

    source_path: Path | None
    raw_content: str
    metadata: AgentMetadata
    configuration: AgentConfiguration
    tasks: list[TaskDefinition]
    validation_issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(
            issue.severity == ValidationSeverity.ERROR
            for issue in self.validation_issues
        )

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    def get_task(self, task_id: str) -> TaskDefinition | None:
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
    task_results: list[TaskResult]
    status: str  # "completed", "partial", "failed"
    errors: list[str] = field(default_factory=list)

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
            lines.append(
                f"  {status_icon} {result.task_id}: {result.status.name} ({result.duration_ms:.2f} ms)"
            )
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
    METADATA_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)
    SECTION_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
    TASK_PATTERN = re.compile(r"^###\s+Task:\s*(.+?)\s*$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

    def __init__(self) -> None:
        self.validation_issues: list[ValidationIssue] = []

    def parse_file(self, file_path: Path) -> AgentFile:
        """Parse an agent.md file from disk."""
        if not file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        return self.parse_content(content, source_path=file_path)

    def parse_content(
        self,
        content: str,
        source_path: Path | None = None,
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
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="No metadata section found (YAML frontmatter)",
                    section="metadata",
                )
            )
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
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid YAML in metadata: {e}",
                    section="metadata",
                )
            )
            return AgentMetadata()

    def _extract_configuration(self, content: str) -> AgentConfiguration:
        """Extract configuration section."""
        # Look for ## Configuration section
        config_match = re.search(
            r"##\s+Configuration\s*\n(.*?)(?=\n##\s|\Z)",
            content,
            re.DOTALL | re.IGNORECASE,
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
                self.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid configuration: {e}",
                        section="configuration",
                    )
                )

        return AgentConfiguration()

    def _extract_tasks(self, content: str) -> list[TaskDefinition]:
        """Extract task definitions from ## Tasks section."""
        tasks: list[TaskDefinition] = []

        # Look for ## Tasks section
        tasks_match = re.search(
            r"##\s+Tasks\s*\n(.*?)(?=\n##\s|\Z)", content, re.DOTALL | re.IGNORECASE
        )

        if not tasks_match:
            self.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="No Tasks section found",
                    section="tasks",
                )
            )
            return tasks

        tasks_content = tasks_match.group(1)

        # Find all ### Task: definitions
        task_matches = list(self.TASK_PATTERN.finditer(tasks_content))

        for i, match in enumerate(task_matches):
            task_name = match.group(1).strip()

            # Get content until next task or end
            start_pos = match.end()
            end_pos = (
                task_matches[i + 1].start()
                if i + 1 < len(task_matches)
                else len(tasks_content)
            )
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
    ) -> TaskDefinition | None:
        """Parse a single task definition."""
        task_id = f"task_{index}"
        task_type = TaskType.CUSTOM
        description = ""
        parameters: dict[str, Any] = {}
        depends_on: list[str] = []
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
                
                # Extract complex task fields
                condition = params.pop("condition", None)
                retry_count = params.pop("retry_count", 0)
                retry_delay_seconds = params.pop("retry_delay_seconds", 1.0)
                loop_over = params.pop("loop_over", None)
                loop_variable = params.pop("loop_variable", "item")
                parallel = params.pop("parallel", False)
                group_id = params.pop("group_id", None)
                on_error = params.pop("on_error", None)
                on_success = params.pop("on_success", None)
                priority = params.pop("priority", 0)
                tags = params.pop("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                
                # Parse subtasks if present
                subtasks_data = params.pop("subtasks", [])
                subtasks = []
                for si, sub_data in enumerate(subtasks_data):
                    if isinstance(sub_data, dict):
                        sub_task = TaskDefinition(
                            id=sub_data.get("id", f"{task_id}_sub_{si+1}"),
                            name=sub_data.get("name", f"Subtask {si+1}"),
                            type=TaskType(sub_data.get("type", "custom")),
                            description=sub_data.get("description", ""),
                            parameters=sub_data.get("parameters", {}),
                            depends_on=sub_data.get("depends_on", []),
                            timeout_seconds=sub_data.get("timeout_seconds"),
                        )
                        subtasks.append(sub_task)

                # Remaining are task-specific parameters
                parameters = params

            except Exception as e:
                self.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Could not parse task parameters: {e}",
                        section=f"task:{name}",
                    )
                )
                # Set defaults for complex fields when parsing fails
                condition = None
                retry_count = 0
                retry_delay_seconds = 1.0
                loop_over = None
                loop_variable = "item"
                parallel = False
                group_id = None
                on_error = None
                on_success = None
                priority = 0
                tags = []
                subtasks = []

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
            condition=condition,
            retry_count=retry_count,
            retry_delay_seconds=retry_delay_seconds,
            loop_over=loop_over,
            loop_variable=loop_variable,
            parallel=parallel,
            group_id=group_id,
            subtasks=subtasks,
            on_error=on_error,
            on_success=on_success,
            priority=priority,
            tags=tags,
        )

    def _validate_tasks(
        self,
        tasks: list[TaskDefinition],
        config: AgentConfiguration,
    ) -> None:
        """Validate task definitions."""
        task_ids = {t.id for t in tasks}

        for task in tasks:
            # Check dependencies exist
            for dep in task.depends_on:
                if dep not in task_ids:
                    self.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Task '{task.id}' depends on unknown task '{dep}'",
                            section=f"task:{task.name}",
                        )
                    )

            # Check circular dependencies (simple check)
            if task.id in task.depends_on:
                self.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Task '{task.id}' has circular dependency on itself",
                        section=f"task:{task.name}",
                    )
                )

            # Validate based on task type
            if task.type == TaskType.CIRCUIT_EXECUTION:
                if (
                    "circuit" not in task.parameters
                    and "circuit_file" not in task.parameters
                ):
                    self.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Circuit execution task '{task.id}' missing circuit parameter",
                            section=f"task:{task.name}",
                        )
                    )

            elif task.type == TaskType.BACKEND_COMPARISON:
                if not task.parameters.get("backends") and not config.backends:
                    self.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Backend comparison task '{task.id}' has no backends specified",
                            section=f"task:{task.name}",
                        )
                    )


class TaskExecutor(Protocol):
    """Protocol for task executors."""

    def execute(self, task: TaskDefinition, context: dict[str, Any]) -> Any:
        """Execute a task and return the result."""
        ...


class DefaultTaskExecutor:
    """Default task executor that handles common task types.

    Integrates with the backend system, comparison engine, and export facilities
    to provide fully functional task execution.

    Supports configurable defaults for execution parameters and retry logic
    for improved error recovery.
    """

    # Default configuration - can be overridden via set_defaults()
    DEFAULT_SHOTS = 1024
    DEFAULT_TIMEOUT_SECONDS = 300
    DEFAULT_RETRY_COUNT = 3
    DEFAULT_RETRY_DELAY_SECONDS = 1.0
    DEFAULT_BACKEND = "auto"
    DEFAULT_EXPORT_FORMAT = "json"
    MAX_SCRIPT_SIZE_BYTES = 10240  # 10KB max script size
    SCRIPT_EXECUTION_TIMEOUT = 30  # 30 seconds max script execution

    def __init__(
        self,
        default_shots: int | None = None,
        default_backend: str | None = None,
        default_timeout: int | None = None,
        retry_count: int | None = None,
        retry_delay: float | None = None,
        enable_script_execution: bool = True,
    ) -> None:
        """Initialize task executor with configurable defaults.

        Args:
            default_shots: Default number of shots for circuit execution.
            default_backend: Default backend to use (or 'auto').
            default_timeout: Default timeout in seconds.
            retry_count: Number of retries on failure.
            retry_delay: Delay between retries in seconds.
            enable_script_execution: Whether to allow script execution in custom tasks.
        """
        self._default_shots = default_shots or self.DEFAULT_SHOTS
        self._default_backend = default_backend or self.DEFAULT_BACKEND
        self._default_timeout = default_timeout or self.DEFAULT_TIMEOUT_SECONDS
        self._retry_count = (
            retry_count if retry_count is not None else self.DEFAULT_RETRY_COUNT
        )
        self._retry_delay = (
            retry_delay if retry_delay is not None else self.DEFAULT_RETRY_DELAY_SECONDS
        )
        self._enable_script_execution = enable_script_execution

        self._handlers: dict[TaskType, Callable] = {
            TaskType.CIRCUIT_EXECUTION: self._execute_circuit,
            TaskType.BACKEND_COMPARISON: self._execute_comparison,
            TaskType.RESULT_ANALYSIS: self._execute_analysis,
            TaskType.EXPORT: self._execute_export,
            TaskType.CUSTOM: self._execute_custom,
        }

    def set_defaults(
        self,
        shots: int | None = None,
        backend: str | None = None,
        timeout: int | None = None,
        retry_count: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        """Update default configuration.

        Args:
            shots: Default number of shots.
            backend: Default backend name.
            timeout: Default timeout in seconds.
            retry_count: Number of retries on failure.
            retry_delay: Delay between retries in seconds.
        """
        if shots is not None:
            self._default_shots = shots
        if backend is not None:
            self._default_backend = backend
        if timeout is not None:
            self._default_timeout = timeout
        if retry_count is not None:
            self._retry_count = retry_count
        if retry_delay is not None:
            self._retry_delay = retry_delay

    def _with_retry(
        self,
        func: Callable,
        task: TaskDefinition,
        context: dict[str, Any],
        task_type: str,
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            func: Function to execute.
            task: Task definition.
            context: Execution context.
            task_type: Type name for error reporting.

        Returns:
            Execution result.
        """
        last_error = None
        for attempt in range(self._retry_count):
            try:
                return func(task, context)
            except Exception as e:
                last_error = e
                if attempt < self._retry_count - 1:
                    import time

                    time.sleep(self._retry_delay)
                    continue

        # All retries failed
        return {
            "task_type": task_type,
            "success": False,
            "error": f"Failed after {self._retry_count} attempts: {last_error}",
            "last_exception": str(last_error),
            "attempts": self._retry_count,
        }

    def execute(self, task: TaskDefinition, context: dict[str, Any]) -> Any:
        """Execute a task based on its type."""
        handler = self._handlers.get(task.type, self._execute_custom)
        return handler(task, context)

    def _execute_circuit(self, task: TaskDefinition, context: dict[str, Any]) -> Any:
        """Execute a circuit execution task using the backend system."""
        from proxima.backends.registry import backend_registry
        from proxima.intelligence.selector import (
            BackendSelector,
            CircuitCharacteristics,
        )

        params = task.parameters
        circuit = params.get("circuit")
        backend_name = params.get("backend", self._default_backend)
        shots = params.get("shots", self._default_shots)
        options = params.get("options", {})
        options["shots"] = shots

        # Auto-select backend if needed
        if backend_name == "auto" or not backend_name:
            selector = BackendSelector()
            qubit_count = getattr(circuit, "num_qubits", 4)
            selection = selector.select_from_characteristics(
                CircuitCharacteristics(qubit_count=qubit_count, gate_count=0, depth=0)
            )
            backend_name = selection.backend

        # Get adapter and execute
        try:
            adapter = backend_registry.get(backend_name)
        except KeyError:
            return {
                "task_type": "circuit_execution",
                "success": False,
                "error": f"Backend '{backend_name}' not available",
            }

        # Validate circuit
        validation = adapter.validate_circuit(circuit)
        if not validation.valid:
            return {
                "task_type": "circuit_execution",
                "success": False,
                "error": f"Circuit validation failed: {validation.message}",
            }

        # Execute
        try:
            result = adapter.execute(circuit, options)
            return {
                "task_type": "circuit_execution",
                "success": True,
                "backend": backend_name,
                "result": result,
                "shots": shots,
            }
        except Exception as e:
            return {
                "task_type": "circuit_execution",
                "success": False,
                "backend": backend_name,
                "error": str(e),
            }

    def _execute_comparison(self, task: TaskDefinition, context: dict[str, Any]) -> Any:
        """Execute a backend comparison task using MultiBackendComparator."""
        import asyncio

        from proxima.backends.registry import backend_registry
        from proxima.data.compare import MultiBackendComparator

        params = task.parameters
        circuit = params.get("circuit")
        backend_names = params.get("backends", ["cirq", "qiskit"])
        shots = params.get("shots", 1024)
        options = {"shots": shots}

        # Get adapters for each backend name
        adapters = []
        for name in backend_names:
            try:
                adapters.append(backend_registry.get(name))
            except KeyError:
                pass  # Skip unavailable backends

        if not adapters:
            return {
                "task_type": "backend_comparison",
                "success": False,
                "backends": backend_names,
                "error": "No valid backends available for comparison",
            }

        comparator = MultiBackendComparator()

        # Run comparison (async)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            report = loop.run_until_complete(
                comparator.compare(adapters, circuit, options)
            )
            return {
                "task_type": "backend_comparison",
                "success": True,
                "report": report.to_dict(),
                "backends": backend_names,
                "recommended": report.metrics.recommended_backend,
            }
        except Exception as e:
            return {
                "task_type": "backend_comparison",
                "success": False,
                "backends": backend_names,
                "error": str(e),
            }

    def _execute_analysis(self, task: TaskDefinition, context: dict[str, Any]) -> Any:
        """Execute a result analysis task using the insights engine."""
        from proxima.intelligence.insights import InsightEngine

        params = task.parameters
        result_data = params.get("result", context.get("last_result"))

        if not result_data:
            return {
                "task_type": "result_analysis",
                "success": False,
                "error": "No result data provided for analysis",
            }

        engine = InsightEngine()
        try:
            insights = engine.analyze(result_data)
            return {
                "task_type": "result_analysis",
                "success": True,
                "insights": insights,
            }
        except Exception as e:
            return {
                "task_type": "result_analysis",
                "success": False,
                "error": str(e),
            }

    def _execute_export(self, task: TaskDefinition, context: dict[str, Any]) -> Any:
        """Execute an export task using the export engine."""
        from pathlib import Path

        from proxima.data.export import ExportEngine, ExportFormat, ReportData

        params = task.parameters
        data = params.get("data", context.get("last_result"))
        format_str = params.get("format", "json").lower()
        output_path = params.get("output")

        if not data:
            return {
                "task_type": "export",
                "success": False,
                "error": "No data provided for export",
            }

        # Map format string to ExportFormat enum
        format_map = {
            "json": ExportFormat.JSON,
            "csv": ExportFormat.CSV,
            "xlsx": ExportFormat.XLSX,
            "html": ExportFormat.HTML,
        }
        export_format = format_map.get(format_str, ExportFormat.JSON)

        # Convert data to ReportData if needed
        if isinstance(data, dict):
            report_data = ReportData(
                title=data.get("title", "Proxima Export"),
                summary=data.get("summary", data),
                raw_results=data.get("raw_results", [data]),
                metadata=data.get("metadata", {}),
            )
        elif isinstance(data, ReportData):
            report_data = data
        else:
            report_data = ReportData(summary={"data": str(data)})

        engine = ExportEngine()
        try:
            result = engine.export(
                report_data,
                export_format,
                Path(output_path) if output_path else None,
            )
            return {
                "task_type": "export",
                "success": result.success,
                "format": format_str,
                "output": str(result.output_path) if result.output_path else "stdout",
                "file_size_bytes": result.file_size_bytes,
                "error": result.error,
            }
        except Exception as e:
            return {
                "task_type": "export",
                "success": False,
                "format": format_str,
                "error": str(e),
            }

    def _execute_custom(self, task: TaskDefinition, context: dict[str, Any]) -> Any:
        """Execute a custom task.

        Custom tasks support flexible execution through parameters:
        - 'script': Python code to execute (sandboxed)
        - 'command': Shell command to run
        - 'function': Name of a registered custom function
        - 'plugin': Plugin name to invoke
        """
        params = task.parameters
        result: dict[str, Any] = {
            "task_type": "custom",
            "task_id": task.id,
            "parameters": params,
        }

        # Handle different custom task modes
        if "function" in params:
            # Call a registered custom function
            func_name = params["function"]
            func_args = params.get("args", {})
            try:
                from proxima.plugins.manager import get_plugin_manager

                plugin_mgr = get_plugin_manager()
                if plugin_mgr.has_function(func_name):
                    func_result = plugin_mgr.call_function(func_name, **func_args)
                    result["success"] = True
                    result["result"] = func_result
                else:
                    result["success"] = False
                    result["error"] = f"Function not found: {func_name}"
            except ImportError:
                result["success"] = False
                result["error"] = "Plugin manager not available"
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)

        elif "plugin" in params:
            # Invoke a plugin
            plugin_name = params["plugin"]
            plugin_action = params.get("action", "execute")
            plugin_args = params.get("args", {})
            try:
                from proxima.plugins.manager import get_plugin_manager

                plugin_mgr = get_plugin_manager()
                plugin_result = plugin_mgr.invoke(
                    plugin_name, plugin_action, **plugin_args
                )
                result["success"] = True
                result["result"] = plugin_result
            except ImportError:
                result["success"] = False
                result["error"] = "Plugin manager not available"
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)

        elif "script" in params:
            # Execute a simple script (sandboxed - only allow safe operations)
            # Security: Validate and limit script execution
            script = params["script"]

            # Security check 1: Script execution must be enabled
            if not self._enable_script_execution:
                result["success"] = False
                result["error"] = "Script execution is disabled for security"
                return result

            # Security check 2: Size limit
            if len(script.encode("utf-8")) > self.MAX_SCRIPT_SIZE_BYTES:
                result["success"] = False
                result["error"] = (
                    f"Script exceeds maximum size of {self.MAX_SCRIPT_SIZE_BYTES} bytes"
                )
                return result

            # Security check 3: Dangerous pattern detection
            dangerous_patterns = [
                "__import__",
                "eval(",
                "exec(",
                "compile(",
                "open(",
                "file(",
                "input(",
                "os.",
                "sys.",
                "subprocess",
                "shutil",
                "pathlib",
                "importlib",
                "__class__",
                "__base__",
                "__subclasses__",
                "__globals__",
                "__code__",
                "__builtins__",
            ]
            script_lower = script.lower()
            for pattern in dangerous_patterns:
                if pattern.lower() in script_lower:
                    result["success"] = False
                    result["error"] = f"Script contains prohibited pattern: {pattern}"
                    result["security_violation"] = True
                    return result

            try:
                import threading

                # Create a restricted namespace for execution
                safe_globals: dict[str, Any] = {
                    "__builtins__": {
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                        "tuple": tuple,
                        "set": set,
                        "sum": sum,
                        "min": min,
                        "max": max,
                        "range": range,
                        "enumerate": enumerate,
                        "zip": zip,
                        "sorted": sorted,
                        "reversed": reversed,
                        "abs": abs,
                        "round": round,
                        "pow": pow,
                        "all": all,
                        "any": any,
                        "map": map,
                        "filter": filter,
                        "isinstance": isinstance,
                        "type": type,
                        "print": print,  # Allow print for debugging
                        "True": True,
                        "False": False,
                        "None": None,
                    }
                }
                local_vars: dict[str, Any] = {"context": context}

                # Execute with timeout using threading
                exec_exception: list[Exception] = []
                exec_result: list[Any] = [None]

                def execute_script() -> None:
                    try:
                        exec(script, safe_globals, local_vars)  # noqa: S102
                        exec_result[0] = local_vars.get("result", "Script executed")
                    except Exception as e:
                        exec_exception.append(e)

                thread = threading.Thread(target=execute_script)
                thread.daemon = True
                thread.start()
                thread.join(timeout=self.SCRIPT_EXECUTION_TIMEOUT)

                if thread.is_alive():
                    result["success"] = False
                    result["error"] = (
                        f"Script execution timed out after {self.SCRIPT_EXECUTION_TIMEOUT} seconds"
                    )
                    result["timeout"] = True
                elif exec_exception:
                    result["success"] = False
                    result["error"] = f"Script execution failed: {exec_exception[0]}"
                else:
                    result["success"] = True
                    result["result"] = exec_result[0]

            except Exception as e:
                result["success"] = False
                result["error"] = f"Script execution failed: {e}"

        else:
            # Default: just acknowledge the custom task
            result["success"] = True
            result["message"] = "Custom task acknowledged"
            result["description"] = task.description

        return result


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
        parser: AgentFileParser | None = None,
        executor: TaskExecutor | None = None,
        consent_manager: ConsentManager | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
        display_callback: Callable[[str], None] | None = None,
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

    def _execute_with_timeout(
        self,
        task: TaskDefinition,
        context: dict[str, Any],
        timeout_seconds: int | None,
    ) -> Any:
        """Execute a task with optional timeout enforcement.

        Args:
            task: The task to execute
            context: Execution context
            timeout_seconds: Maximum execution time in seconds

        Returns:
            Task execution result

        Raises:
            TimeoutError: If task exceeds timeout
        """
        import concurrent.futures

        if timeout_seconds is None or timeout_seconds <= 0:
            # No timeout - execute directly
            return self.executor.execute(task, context)

        # Use ThreadPoolExecutor for timeout enforcement
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.executor.execute, task, context)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError as err:
                raise TimeoutError(
                    f"Task '{task.id}' exceeded timeout of {timeout_seconds} seconds"
                ) from err

    def _evaluate_condition(
        self,
        condition: str,
        context: dict[str, Any],
    ) -> bool:
        """Evaluate a condition expression for conditional task execution.
        
        Supports expressions like:
        - "previous_task.success == True"
        - "backend_count > 1"
        - "results.task_1.status == 'completed'"
        
        Args:
            condition: Condition expression string
            context: Execution context with variables
            
        Returns:
            Boolean result of condition evaluation
        """
        if not condition:
            return True
            
        # Build safe evaluation namespace
        safe_namespace = {
            "True": True,
            "False": False,
            "None": None,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "any": any,
            "all": all,
            # Add context variables
            **context,
        }
        
        try:
            # Simple expression evaluation
            result = eval(condition, {"__builtins__": {}}, safe_namespace)  # noqa: S307
            return bool(result)
        except Exception:
            # If evaluation fails, treat as false
            return False

    def _interpolate_parameters(
        self,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Interpolate variables in task parameters.
        
        Supports syntax like:
        - "${previous_result.value}"
        - "${config.shots}"
        - "${loop_item.name}"
        
        Args:
            parameters: Task parameters dict
            context: Execution context with variables
            
        Returns:
            Parameters with interpolated values
        """
        import re
        
        def resolve_path(path: str, ctx: dict) -> Any:
            """Resolve a dot-separated path in context."""
            parts = path.split(".")
            value = ctx
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
                if value is None:
                    return None
            return value
        
        def interpolate_value(val: Any) -> Any:
            if isinstance(val, str):
                # Find all ${...} patterns
                pattern = r"\$\{([^}]+)\}"
                matches = re.findall(pattern, val)
                if matches:
                    for match in matches:
                        resolved = resolve_path(match.strip(), context)
                        if resolved is not None:
                            if val == f"${{{match}}}":
                                # Full replacement - return resolved type
                                return resolved
                            else:
                                # Partial replacement - convert to string
                                val = val.replace(f"${{{match}}}", str(resolved))
                return val
            elif isinstance(val, dict):
                return {k: interpolate_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [interpolate_value(v) for v in val]
            return val
        
        return interpolate_value(parameters)

    def _execute_with_retry(
        self,
        task: TaskDefinition,
        context: dict[str, Any],
        timeout_seconds: int | None,
    ) -> tuple[Any, int]:
        """Execute a task with retry logic.
        
        Args:
            task: Task to execute
            context: Execution context
            timeout_seconds: Timeout per attempt
            
        Returns:
            Tuple of (result, attempts_made)
        """
        max_attempts = max(1, task.retry_count + 1)
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                result = self._execute_with_timeout(task, context, timeout_seconds)
                return result, attempt + 1
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    # Wait before retry with exponential backoff
                    delay = task.retry_delay_seconds * (2 ** attempt)
                    time.sleep(delay)
        
        # All attempts failed
        raise last_error or RuntimeError(f"Task {task.id} failed after {max_attempts} attempts")

    def _expand_loop_task(
        self,
        task: TaskDefinition,
        context: dict[str, Any],
    ) -> list[TaskDefinition]:
        """Expand a loop task into multiple task instances.
        
        Args:
            task: Task with loop_over defined
            context: Execution context containing the iterable
            
        Returns:
            List of expanded task definitions
        """
        if not task.loop_over:
            return [task]
        
        # Resolve the iterable from context
        iterable = None
        if task.loop_over in context:
            iterable = context[task.loop_over]
        elif "." in task.loop_over:
            parts = task.loop_over.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    break
            iterable = value
        
        if not iterable or not hasattr(iterable, "__iter__"):
            return [task]
        
        expanded_tasks = []
        for idx, item in enumerate(iterable):
            # Create a new task instance for each iteration
            expanded_task = TaskDefinition(
                id=f"{task.id}_iter_{idx}",
                name=f"{task.name} [{idx}]",
                type=task.type,
                description=task.description,
                parameters={
                    **task.parameters,
                    task.loop_variable: item,
                    "_loop_index": idx,
                    "_loop_item": item,
                },
                depends_on=task.depends_on if idx == 0 else [f"{task.id}_iter_{idx-1}"],
                requires_consent=task.requires_consent,
                consent_reason=task.consent_reason,
                timeout_seconds=task.timeout_seconds,
                condition=task.condition,
                retry_count=task.retry_count,
                retry_delay_seconds=task.retry_delay_seconds,
                parallel=task.parallel,
                group_id=task.group_id or task.id,
                on_error=task.on_error,
                on_success=task.on_success,
                priority=task.priority,
                tags=[*task.tags, f"loop:{task.id}"],
            )
            expanded_tasks.append(expanded_task)
        
        return expanded_tasks

    async def _execute_parallel_tasks(
        self,
        tasks: list[TaskDefinition],
        context: dict[str, Any],
        agent_file: AgentFile,
    ) -> list[TaskResult]:
        """Execute a group of tasks in parallel.
        
        Args:
            tasks: List of tasks to execute in parallel
            context: Shared execution context
            agent_file: Agent file for configuration
            
        Returns:
            List of task results
        """
        import asyncio
        
        async def execute_task(task: TaskDefinition) -> TaskResult:
            task_start = time.time()
            try:
                # Check condition
                if task.condition and not self._evaluate_condition(task.condition, context):
                    return TaskResult(
                        task_id=task.id,
                        status=TaskStatus.SKIPPED,
                        start_time=task_start,
                        end_time=time.time(),
                        error="Condition not met",
                        metadata={"condition": task.condition},
                    )
                
                # Interpolate parameters
                interpolated_params = self._interpolate_parameters(
                    task.parameters, context
                )
                task_with_interpolated = TaskDefinition(
                    id=task.id,
                    name=task.name,
                    type=task.type,
                    description=task.description,
                    parameters=interpolated_params,
                    depends_on=task.depends_on,
                    timeout_seconds=task.timeout_seconds,
                )
                
                task_timeout = (
                    task.timeout_seconds or agent_file.configuration.timeout_seconds
                )
                
                # Execute with retry in thread pool
                loop = asyncio.get_event_loop()
                result_data, attempts = await loop.run_in_executor(
                    None,
                    self._execute_with_retry,
                    task_with_interpolated,
                    context,
                    task_timeout,
                )
                
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    start_time=task_start,
                    end_time=time.time(),
                    result=result_data,
                    metadata={"attempts": attempts},
                )
            except Exception as e:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    start_time=task_start,
                    end_time=time.time(),
                    error=str(e),
                )
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*[execute_task(t) for t in tasks])
        return list(results)

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
        tasks: list[TaskDefinition],
    ) -> list[TaskDefinition]:
        """Build execution order respecting dependencies (topological sort)."""
        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        in_degree = {t.id: len(t.depends_on) for t in tasks}
        dependents: dict[str, list[str]] = {t.id: [] for t in tasks}

        for task in tasks:
            for dep in task.depends_on:
                if dep in dependents:
                    dependents[dep].append(task.id)

        # Kahn's algorithm
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result: list[TaskDefinition] = []

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
        context: dict[str, Any] | None = None,
    ) -> ExecutionReport:
        """Execute an agent.md file with full complex task support.

        Enhanced Task Execution Flow:
        1. Expand loop tasks into multiple instances
        2. Group parallel tasks together
        3. Sort by priority and dependencies
        4. For each task or parallel group:
           a. Check condition expression
           b. Display task description
           c. Request consent for sensitive operations
           d. Interpolate variables in parameters
           e. Execute with retry logic
           f. Handle on_success/on_error callbacks
           g. Execute subtasks if present
        5. Collect results and continue/stop based on config

        Returns:
            ExecutionReport with all task results
        """
        import asyncio
        
        start_time = time.time()
        context = context or {}
        task_results: list[TaskResult] = []
        errors: list[str] = []

        # Validate agent file
        if not agent_file.is_valid:
            return ExecutionReport(
                agent_file=agent_file,
                start_time=start_time,
                end_time=time.time(),
                task_results=[],
                status="failed",
                errors=[
                    str(i)
                    for i in agent_file.validation_issues
                    if i.severity == ValidationSeverity.ERROR
                ],
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

        # Step 1: Expand loop tasks
        expanded_tasks: list[TaskDefinition] = []
        for task in agent_file.tasks:
            if task.loop_over:
                expanded = self._expand_loop_task(task, context)
                expanded_tasks.extend(expanded)
            else:
                expanded_tasks.append(task)

        # Step 2: Sort by priority (higher first), then build execution order
        priority_sorted = sorted(expanded_tasks, key=lambda t: -t.priority)
        ordered_tasks = self._build_execution_order(priority_sorted)
        
        # Step 3: Group parallel tasks
        parallel_groups: dict[str, list[TaskDefinition]] = {}
        sequential_tasks: list[TaskDefinition] = []
        
        for task in ordered_tasks:
            if task.parallel and agent_file.configuration.parallel_execution:
                group_id = task.group_id or "default_parallel"
                if group_id not in parallel_groups:
                    parallel_groups[group_id] = []
                parallel_groups[group_id].append(task)
            else:
                sequential_tasks.append(task)

        total_tasks = len(ordered_tasks)
        self._display(f"\nExecuting agent: {agent_file.metadata.name}")
        self._display(f"Tasks: {total_tasks} ({len(parallel_groups)} parallel groups)")
        self._display("-" * 40)

        # Track completed task IDs for dependency checking
        completed_tasks: set[str] = set()
        task_results_map: dict[str, TaskResult] = {}
        
        # Helper to execute a single task
        def execute_single_task(
            task: TaskDefinition,
            task_index: int,
        ) -> TaskResult:
            task_start = time.time()

            # Check condition
            if task.condition:
                condition_context = {
                    **context,
                    "configuration": agent_file.configuration,
                    "previous_results": task_results_map,
                    "completed_tasks": list(completed_tasks),
                }
                if not self._evaluate_condition(task.condition, condition_context):
                    self._display(f"\n[{task_index}/{total_tasks}] Task: {task.name} [SKIPPED - condition not met]")
                    return TaskResult(
                        task_id=task.id,
                        status=TaskStatus.SKIPPED,
                        start_time=task_start,
                        end_time=time.time(),
                        error="Condition not met",
                        metadata={"condition": task.condition},
                    )

            # Display task description
            self._display(f"\n[{task_index}/{total_tasks}] Task: {task.name}")
            if task.description:
                self._display(f"    {task.description}")

            # Check dependencies are completed
            missing_deps = [d for d in task.depends_on if d not in completed_tasks]
            if missing_deps:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.SKIPPED,
                    start_time=task_start,
                    end_time=time.time(),
                    error=f"Dependencies not completed: {missing_deps}",
                )

            # Request consent for sensitive operations
            if task.requires_consent:
                if not self._request_consent(task, agent_file):
                    return TaskResult(
                        task_id=task.id,
                        status=TaskStatus.CANCELLED,
                        start_time=task_start,
                        end_time=time.time(),
                        error="Consent denied",
                    )

            # Build task context with interpolation
            task_context = {
                **context,
                "configuration": agent_file.configuration,
                "previous_results": task_results_map,
                "completed_tasks": list(completed_tasks),
            }

            # Interpolate parameters
            interpolated_params = self._interpolate_parameters(
                task.parameters, task_context
            )
            
            # Create task with interpolated parameters
            interpolated_task = TaskDefinition(
                id=task.id,
                name=task.name,
                type=task.type,
                description=task.description,
                parameters=interpolated_params,
                depends_on=task.depends_on,
                timeout_seconds=task.timeout_seconds,
                retry_count=task.retry_count,
                retry_delay_seconds=task.retry_delay_seconds,
            )

            # Get timeout for this task
            task_timeout = (
                task.timeout_seconds or agent_file.configuration.timeout_seconds
            )

            try:
                # Execute with retry logic
                task_result_data, attempts = self._execute_with_retry(
                    interpolated_task, task_context, task_timeout
                )

                result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    start_time=task_start,
                    end_time=time.time(),
                    result=task_result_data,
                    metadata={"attempts": attempts, "retry_count": task.retry_count},
                )

                self._display(f"     Completed in {result.duration_ms:.2f} ms (attempts: {attempts})")
                
                return result

            except Exception as e:
                result = TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    start_time=task_start,
                    end_time=time.time(),
                    error=str(e),
                    metadata={"retry_count": task.retry_count},
                )

                self._display(f"     Failed: {e}")
                return result

        # Execute parallel groups first (if parallel execution is enabled)
        if agent_file.configuration.parallel_execution and parallel_groups:
            for group_id, group_tasks in parallel_groups.items():
                self._display(f"\n[Parallel Group: {group_id}]")
                
                try:
                    # Run parallel execution
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        parallel_results = loop.run_until_complete(
                            self._execute_parallel_tasks(group_tasks, context, agent_file)
                        )
                    finally:
                        loop.close()
                    
                    for result in parallel_results:
                        task_results.append(result)
                        task_results_map[result.task_id] = result
                        if result.status == TaskStatus.COMPLETED:
                            completed_tasks.add(result.task_id)
                        elif result.status == TaskStatus.FAILED:
                            errors.append(f"Task '{result.task_id}' failed: {result.error}")
                            
                except Exception as e:
                    errors.append(f"Parallel group '{group_id}' failed: {e}")

        # Execute sequential tasks
        task_index = len(task_results) + 1
        for task in sequential_tasks:
            self._report_progress("executing", (task_index / total_tasks))

            result = execute_single_task(task, task_index)
            task_results.append(result)
            task_results_map[result.task_id] = result

            if result.status == TaskStatus.COMPLETED:
                completed_tasks.add(task.id)
                
                # Handle on_success callback
                if task.on_success:
                    on_success_task = agent_file.get_task(task.on_success)
                    if on_success_task:
                        self._display(f"    Running on_success: {task.on_success}")
                        callback_result = execute_single_task(
                            on_success_task, task_index
                        )
                        task_results.append(callback_result)
                        task_results_map[callback_result.task_id] = callback_result
                        
                # Execute subtasks
                for subtask in task.subtasks:
                    sub_result = execute_single_task(subtask, task_index)
                    task_results.append(sub_result)
                    task_results_map[sub_result.task_id] = sub_result
                    if sub_result.status == TaskStatus.COMPLETED:
                        completed_tasks.add(subtask.id)

            elif result.status == TaskStatus.FAILED:
                errors.append(f"Task '{task.id}' failed: {result.error}")
                
                # Handle on_error callback
                if task.on_error:
                    on_error_task = agent_file.get_task(task.on_error)
                    if on_error_task:
                        self._display(f"    Running on_error: {task.on_error}")
                        callback_result = execute_single_task(on_error_task, task_index)
                        task_results.append(callback_result)
                        task_results_map[callback_result.task_id] = callback_result

                if not agent_file.configuration.continue_on_error:
                    break

            elif result.status in (TaskStatus.SKIPPED, TaskStatus.CANCELLED):
                if result.error:
                    errors.append(f"Task '{task.id}': {result.error}")

                if not agent_file.configuration.continue_on_error:
                    break

            task_index += 1

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
        context: dict[str, Any] | None = None,
    ) -> ExecutionReport:
        """Load and execute an agent.md file."""
        agent_file = self.load_file(file_path)
        return self.execute(agent_file, context)


# Convenience function
def run_agent_file(
    file_path: Path,
    context: dict[str, Any] | None = None,
    consent_manager: ConsentManager | None = None,
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


# ==================== ENHANCED AGENT.MD PARSING ====================


class AgentMDParseError(Exception):
    """Error during agent.md parsing."""

    def __init__(
        self,
        message: str,
        line_number: int | None = None,
        section: str | None = None,
        original_error: Exception | None = None,
    ):
        self.message = message
        self.line_number = line_number
        self.section = section
        self.original_error = original_error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.line_number:
            parts.append(f" (line {self.line_number})")
        if self.section:
            parts.append(f" in section '{self.section}'")
        return "".join(parts)


@dataclass
class MarkdownSection:
    """A parsed section from a Markdown file."""

    title: str
    level: int
    content: str
    start_line: int
    end_line: int
    subsections: list["MarkdownSection"] = field(default_factory=list)


@dataclass
class FrontmatterData:
    """Parsed YAML frontmatter from agent.md."""

    raw: str
    data: dict[str, Any]
    start_line: int
    end_line: int


class EnhancedMarkdownParser:
    """Enhanced parser for agent.md files with full Markdown support.

    Features:
    - YAML frontmatter parsing
    - Nested section extraction
    - Code block handling
    - Variable interpolation
    - Include directive support
    """

    FRONTMATTER_PATTERN = re.compile(r"^---\s*$", re.MULTILINE)
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```(\w+)?\s*\n(.*?)\n```", re.DOTALL)
    VARIABLE_PATTERN = re.compile(r"\{\{\s*(\w+(?:\.\w+)*)\s*\}\}")
    INCLUDE_PATTERN = re.compile(r"^<!--\s*include:\s*(.+?)\s*-->", re.MULTILINE)

    def __init__(self, base_path: Path | None = None):
        """Initialize the parser.

        Args:
            base_path: Base path for resolving include directives
        """
        self.base_path = base_path or Path.cwd()

    def parse(self, content: str) -> tuple[FrontmatterData | None, list[MarkdownSection]]:
        """Parse markdown content into frontmatter and sections.

        Args:
            content: Raw markdown content

        Returns:
            Tuple of (frontmatter, sections)
        """
        lines = content.split("\n")

        # Extract frontmatter
        frontmatter = self._extract_frontmatter(lines)

        # Skip frontmatter lines for section parsing
        start_line = frontmatter.end_line + 1 if frontmatter else 0
        remaining_lines = lines[start_line:]

        # Parse sections
        sections = self._extract_sections(remaining_lines, start_line)

        return frontmatter, sections

    def _extract_frontmatter(self, lines: list[str]) -> FrontmatterData | None:
        """Extract YAML frontmatter from the beginning of the file."""
        if not lines or lines[0].strip() != "---":
            return None

        end_idx = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = idx
                break

        if end_idx is None:
            return None

        frontmatter_lines = lines[1:end_idx]
        raw_yaml = "\n".join(frontmatter_lines)

        try:
            data = yaml.safe_load(raw_yaml) or {}
        except yaml.YAMLError as exc:
            raise AgentMDParseError(
                f"Invalid YAML frontmatter: {exc}",
                line_number=2,
                section="frontmatter",
                original_error=exc,
            )

        return FrontmatterData(
            raw=raw_yaml,
            data=data,
            start_line=0,
            end_line=end_idx,
        )

    def _extract_sections(
        self,
        lines: list[str],
        offset: int = 0,
    ) -> list[MarkdownSection]:
        """Extract sections from markdown lines."""
        sections: list[MarkdownSection] = []
        current_section: MarkdownSection | None = None
        content_lines: list[str] = []

        for idx, line in enumerate(lines):
            actual_line = idx + offset

            match = self.HEADING_PATTERN.match(line)
            if match:
                # Save previous section
                if current_section:
                    current_section.content = "\n".join(content_lines).strip()
                    current_section.end_line = actual_line - 1
                    sections.append(current_section)
                    content_lines = []

                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = MarkdownSection(
                    title=title,
                    level=level,
                    content="",
                    start_line=actual_line,
                    end_line=actual_line,
                )
            elif current_section:
                content_lines.append(line)

        # Don't forget the last section
        if current_section:
            current_section.content = "\n".join(content_lines).strip()
            current_section.end_line = len(lines) + offset - 1
            sections.append(current_section)

        # Organize into hierarchy
        return self._organize_sections(sections)

    def _organize_sections(
        self,
        flat_sections: list[MarkdownSection],
    ) -> list[MarkdownSection]:
        """Organize flat sections into a hierarchy based on heading levels."""
        if not flat_sections:
            return []

        root_sections: list[MarkdownSection] = []
        section_stack: list[MarkdownSection] = []

        for section in flat_sections:
            # Pop sections from stack until we find a parent
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()

            if section_stack:
                # Add as subsection
                section_stack[-1].subsections.append(section)
            else:
                # Top-level section
                root_sections.append(section)

            section_stack.append(section)

        return root_sections

    def extract_code_blocks(self, content: str) -> list[tuple[str, str]]:
        """Extract code blocks from markdown content.

        Returns:
            List of (language, code) tuples
        """
        blocks = []
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            lang = match.group(1) or ""
            code = match.group(2)
            blocks.append((lang, code))
        return blocks

    def interpolate_variables(
        self,
        content: str,
        variables: dict[str, Any],
    ) -> str:
        """Replace variable placeholders with values.

        Args:
            content: Content with {{ variable }} placeholders
            variables: Variable values

        Returns:
            Interpolated content
        """
        def replace_var(match: re.Match) -> str:
            var_path = match.group(1)
            value = self._get_nested_value(variables, var_path)
            return str(value) if value is not None else match.group(0)

        return self.VARIABLE_PATTERN.sub(replace_var, content)

    def _get_nested_value(
        self,
        data: dict[str, Any],
        path: str,
    ) -> Any:
        """Get a nested value from a dict using dot notation."""
        parts = path.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

    def process_includes(self, content: str) -> str:
        """Process include directives.

        Args:
            content: Content with <!-- include: path --> directives

        Returns:
            Content with includes expanded
        """
        def replace_include(match: re.Match) -> str:
            include_path = match.group(1).strip()
            full_path = self.base_path / include_path

            if not full_path.exists():
                return f"<!-- ERROR: Include not found: {include_path} -->"

            try:
                return full_path.read_text(encoding="utf-8")
            except Exception as exc:
                return f"<!-- ERROR: Could not read include: {exc} -->"

        return self.INCLUDE_PATTERN.sub(replace_include, content)



# ==================== COMMAND VALIDATION ====================


class CommandValidationError(Exception):
    """Error during command validation."""

    def __init__(
        self,
        message: str,
        command: str | None = None,
        parameter: str | None = None,
        suggestions: list[str] | None = None,
    ):
        self.message = message
        self.command = command
        self.parameter = parameter
        self.suggestions = suggestions or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]
        if self.command:
            parts.append(f" (command: {self.command})")
        if self.parameter:
            parts.append(f" (parameter: {self.parameter})")
        if self.suggestions:
            parts.append(f" Suggestions: {', '.join(self.suggestions)}")
        return "".join(parts)


@dataclass
class ValidationResult:
    """Result of command validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_command: dict[str, Any] | None = None


@dataclass
class ParameterSpec:
    """Specification for a command parameter."""

    name: str
    type: type | tuple[type, ...]
    required: bool = False
    default: Any = None
    choices: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    validator: Callable[[Any], bool] | None = None
    description: str = ""


@dataclass
class CommandSpec:
    """Specification for a command."""

    name: str
    parameters: list[ParameterSpec] = field(default_factory=list)
    required_params: list[str] = field(default_factory=list)
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    requires_consent: bool = False


class CommandValidator:
    """Validates commands before execution.

    Features:
    - Parameter type checking
    - Required parameter validation
    - Range and choice validation
    - Pattern matching
    - Custom validators
    - Command normalization
    """

    def __init__(self):
        """Initialize the validator with built-in command specs."""
        self._command_specs: dict[str, CommandSpec] = {}
        self._alias_map: dict[str, str] = {}
        self._register_builtin_commands()

    def _register_builtin_commands(self) -> None:
        """Register built-in command specifications."""
        # Circuit execution command
        self.register_command(CommandSpec(
            name="execute_circuit",
            parameters=[
                ParameterSpec("qubits", int, required=True, min_value=1, max_value=50),
                ParameterSpec("shots", int, required=False, default=1024, min_value=1, max_value=1000000),
                ParameterSpec("backend", str, required=False, default="cirq"),
                ParameterSpec("gates", list, required=False, default=[]),
                ParameterSpec("measurements", bool, required=False, default=True),
            ],
            aliases=["run_circuit", "exec"],
            description="Execute a quantum circuit",
        ))

        # Backend comparison command
        self.register_command(CommandSpec(
            name="compare_backends",
            parameters=[
                ParameterSpec("backends", list, required=True),
                ParameterSpec("qubits", int, required=True, min_value=1, max_value=50),
                ParameterSpec("shots", int, required=False, default=1024),
                ParameterSpec("metrics", list, required=False, default=["fidelity", "time"]),
            ],
            aliases=["backend_comparison", "compare"],
            description="Compare results across backends",
        ))

        # Export command
        self.register_command(CommandSpec(
            name="export_results",
            parameters=[
                ParameterSpec("format", str, required=False, default="json", choices=["json", "csv", "qasm", "pdf"]),
                ParameterSpec("path", str, required=False),
                ParameterSpec("include_analysis", bool, required=False, default=True),
            ],
            aliases=["export", "save"],
            description="Export execution results",
        ))

        # Analysis command
        self.register_command(CommandSpec(
            name="analyze_results",
            parameters=[
                ParameterSpec("type", str, required=False, default="summary", choices=["summary", "detailed", "comparison"]),
                ParameterSpec("metrics", list, required=False),
                ParameterSpec("visualize", bool, required=False, default=False),
            ],
            aliases=["analyze", "analysis"],
            description="Analyze execution results",
        ))

    def register_command(self, spec: CommandSpec) -> None:
        """Register a command specification."""
        self._command_specs[spec.name] = spec
        for alias in spec.aliases:
            self._alias_map[alias] = spec.name

    def get_command_spec(self, command_name: str) -> CommandSpec | None:
        """Get the specification for a command."""
        # Check direct name
        if command_name in self._command_specs:
            return self._command_specs[command_name]
        # Check aliases
        if command_name in self._alias_map:
            return self._command_specs[self._alias_map[command_name]]
        return None

    def validate(
        self,
        command_name: str,
        parameters: dict[str, Any],
    ) -> ValidationResult:
        """Validate a command and its parameters.

        Args:
            command_name: Name of the command
            parameters: Command parameters

        Returns:
            ValidationResult with validation status
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Resolve command name
        spec = self.get_command_spec(command_name)
        if not spec:
            similar = self._find_similar_commands(command_name)
            return ValidationResult(
                valid=False,
                errors=[f"Unknown command: {command_name}"],
                warnings=[f"Did you mean: {', '.join(similar)}"] if similar else [],
            )

        normalized = {}

        # Validate parameters
        param_specs = {p.name: p for p in spec.parameters}

        # Check required parameters
        for param_name in spec.required_params:
            if param_name not in parameters:
                errors.append(f"Missing required parameter: {param_name}")

        # Check for required params defined in ParameterSpec
        for param_spec in spec.parameters:
            if param_spec.required and param_spec.name not in parameters:
                errors.append(f"Missing required parameter: {param_spec.name}")

        # Validate each provided parameter
        for param_name, value in parameters.items():
            if param_name not in param_specs:
                similar = self._find_similar_params(param_name, spec)
                if similar:
                    warnings.append(f"Unknown parameter '{param_name}'. Did you mean: {', '.join(similar)}?")
                else:
                    warnings.append(f"Unknown parameter: {param_name}")
                continue

            param_spec = param_specs[param_name]
            validated_value, param_errors = self._validate_parameter(value, param_spec)

            if param_errors:
                errors.extend(param_errors)
            else:
                normalized[param_name] = validated_value

        # Apply defaults for missing optional parameters
        for param_spec in spec.parameters:
            if param_spec.name not in normalized and not param_spec.required:
                normalized[param_spec.name] = param_spec.default

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_command={
                "name": spec.name,
                "parameters": normalized,
            } if len(errors) == 0 else None,
        )

    def _validate_parameter(
        self,
        value: Any,
        spec: ParameterSpec,
    ) -> tuple[Any, list[str]]:
        """Validate a single parameter."""
        errors = []

        # Type check
        if not isinstance(value, spec.type):
            try:
                # Try to coerce
                if spec.type == int:
                    value = int(value)
                elif spec.type == float:
                    value = float(value)
                elif spec.type == bool:
                    value = str(value).lower() in ("true", "1", "yes")
                elif spec.type == str:
                    value = str(value)
            except (ValueError, TypeError):
                errors.append(f"Parameter '{spec.name}' expected {spec.type.__name__}, got {type(value).__name__}")
                return value, errors

        # Range check for numbers
        if isinstance(value, (int, float)):
            if spec.min_value is not None and value < spec.min_value:
                errors.append(f"Parameter '{spec.name}' must be >= {spec.min_value}")
            if spec.max_value is not None and value > spec.max_value:
                errors.append(f"Parameter '{spec.name}' must be <= {spec.max_value}")

        # Choice check
        if spec.choices and value not in spec.choices:
            errors.append(f"Parameter '{spec.name}' must be one of: {', '.join(str(c) for c in spec.choices)}")

        # Pattern check
        if spec.pattern and isinstance(value, str):
            if not re.match(spec.pattern, value):
                errors.append(f"Parameter '{spec.name}' does not match required pattern")

        # Custom validator
        if spec.validator:
            try:
                if not spec.validator(value):
                    errors.append(f"Parameter '{spec.name}' failed custom validation")
            except Exception as exc:
                errors.append(f"Parameter '{spec.name}' validation error: {exc}")

        return value, errors

    def _find_similar_commands(self, name: str) -> list[str]:
        """Find commands similar to the given name."""
        all_names = list(self._command_specs.keys()) + list(self._alias_map.keys())
        return self._find_similar(name, all_names)

    def _find_similar_params(self, name: str, spec: CommandSpec) -> list[str]:
        """Find parameters similar to the given name."""
        param_names = [p.name for p in spec.parameters]
        return self._find_similar(name, param_names)

    def _find_similar(self, target: str, candidates: list[str], max_distance: int = 3) -> list[str]:
        """Find similar strings using Levenshtein distance."""
        similar = []
        target_lower = target.lower()

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Simple substring check
            if target_lower in candidate_lower or candidate_lower in target_lower:
                similar.append(candidate)
                continue

            # Calculate edit distance (simplified)
            distance = self._edit_distance(target_lower, candidate_lower)
            if distance <= max_distance:
                similar.append(candidate)

        return similar[:3]  # Return top 3

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate simple edit distance between two strings."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]



# ==================== ERROR RECOVERY STRATEGIES ====================


class RecoveryAction(Enum):
    """Actions that can be taken to recover from errors."""

    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"
    ROLLBACK = "rollback"
    PROMPT_USER = "prompt_user"
    USE_CACHED = "use_cached"
    REDUCE_LOAD = "reduce_load"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    success: bool
    action_taken: RecoveryAction
    message: str
    recovered_data: Any = None
    retry_count: int = 0


@dataclass
class ErrorContext:
    """Context information about an error."""

    error: Exception
    error_type: str
    task_id: str | None = None
    stage: str | None = None
    attempt: int = 1
    max_attempts: int = 3
    start_time: float = 0.0
    additional_info: dict[str, Any] = field(default_factory=dict)


class RecoveryStrategy(ABC):
    """Base class for error recovery strategies."""

    @abstractmethod
    def can_recover(self, context: ErrorContext) -> bool:
        """Check if this strategy can handle the error."""
        pass

    @abstractmethod
    def recover(self, context: ErrorContext) -> RecoveryResult:
        """Attempt to recover from the error."""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority of this strategy (lower = higher priority)."""
        pass


class RetryStrategy(RecoveryStrategy):
    """Strategy that retries failed operations with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential: bool = True,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential = exponential
        self.jitter = jitter

    @property
    def priority(self) -> int:
        return 10

    def can_recover(self, context: ErrorContext) -> bool:
        """Can retry if we haven't exceeded max attempts."""
        return context.attempt <= self.max_retries

    def recover(self, context: ErrorContext) -> RecoveryResult:
        """Schedule a retry with appropriate delay."""
        if context.attempt > self.max_retries:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.RETRY,
                message=f"Max retries ({self.max_retries}) exceeded",
                retry_count=context.attempt - 1,
            )

        delay = self._calculate_delay(context.attempt)
        time.sleep(delay)

        return RecoveryResult(
            success=True,
            action_taken=RecoveryAction.RETRY,
            message=f"Retrying (attempt {context.attempt}/{self.max_retries})",
            retry_count=context.attempt,
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with optional exponential backoff and jitter."""
        if self.exponential:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())

        return delay


class FallbackStrategy(RecoveryStrategy):
    """Strategy that uses fallback options when primary fails."""

    def __init__(
        self,
        fallback_backends: list[str] | None = None,
        fallback_values: dict[str, Any] | None = None,
    ):
        self.fallback_backends = fallback_backends or ["cirq", "qiskit", "numpy"]
        self.fallback_values = fallback_values or {}
        self._current_fallback_idx = 0

    @property
    def priority(self) -> int:
        return 20

    def can_recover(self, context: ErrorContext) -> bool:
        """Can use fallback if there are alternatives available."""
        # Check if we have fallback options
        if "backend" in context.additional_info:
            current = context.additional_info["backend"]
            return any(b != current for b in self.fallback_backends)
        return bool(self.fallback_values)

    def recover(self, context: ErrorContext) -> RecoveryResult:
        """Switch to a fallback option."""
        if "backend" in context.additional_info:
            current = context.additional_info["backend"]
            for backend in self.fallback_backends:
                if backend != current:
                    return RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.FALLBACK,
                        message=f"Switching from {current} to fallback backend {backend}",
                        recovered_data={"backend": backend},
                    )

        if self.fallback_values:
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.FALLBACK,
                message="Using fallback values",
                recovered_data=self.fallback_values,
            )

        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.FALLBACK,
            message="No fallback options available",
        )


class CacheStrategy(RecoveryStrategy):
    """Strategy that uses cached results when live execution fails."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".proxima" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def priority(self) -> int:
        return 30

    def can_recover(self, context: ErrorContext) -> bool:
        """Check if we have cached data for this context."""
        cache_key = self._get_cache_key(context)
        cache_path = self.cache_dir / f"{cache_key}.json"
        return cache_path.exists()

    def recover(self, context: ErrorContext) -> RecoveryResult:
        """Load cached data."""
        cache_key = self._get_cache_key(context)
        cache_path = self.cache_dir / f"{cache_key}.json"

        try:
            import json
            with open(cache_path, encoding="utf-8") as f:
                cached_data = json.load(f)

            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.USE_CACHED,
                message=f"Using cached result from {cached_data.get('timestamp', 'unknown')}",
                recovered_data=cached_data.get("data"),
            )
        except Exception as exc:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.USE_CACHED,
                message=f"Failed to load cache: {exc}",
            )

    def _get_cache_key(self, context: ErrorContext) -> str:
        """Generate a cache key from the context."""
        import hashlib
        key_parts = [
            context.task_id or "unknown",
            context.stage or "unknown",
            str(context.additional_info),
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def cache_result(
        self,
        context: ErrorContext,
        data: Any,
    ) -> None:
        """Cache a successful result for future recovery."""
        import json
        from datetime import datetime

        cache_key = self._get_cache_key(context)
        cache_path = self.cache_dir / f"{cache_key}.json"

        cache_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": context.task_id,
            "stage": context.stage,
            "data": data,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, default=str)


class ReduceLoadStrategy(RecoveryStrategy):
    """Strategy that reduces workload to recover from resource exhaustion."""

    def __init__(
        self,
        reduction_factor: float = 0.5,
        min_value: int = 1,
    ):
        self.reduction_factor = reduction_factor
        self.min_value = min_value

    @property
    def priority(self) -> int:
        return 25

    def can_recover(self, context: ErrorContext) -> bool:
        """Check if this is a resource-related error."""
        error_str = str(context.error).lower()
        resource_keywords = ["memory", "oom", "resource", "timeout", "limit", "exhausted"]
        return any(kw in error_str for kw in resource_keywords)

    def recover(self, context: ErrorContext) -> RecoveryResult:
        """Reduce workload parameters."""
        reduced_params = {}

        # Reduce shots
        if "shots" in context.additional_info:
            current = context.additional_info["shots"]
            reduced = max(int(current * self.reduction_factor), self.min_value)
            reduced_params["shots"] = reduced

        # Reduce qubits if possible
        if "qubits" in context.additional_info:
            current = context.additional_info["qubits"]
            reduced = max(int(current * self.reduction_factor), self.min_value)
            reduced_params["qubits"] = reduced

        # Reduce batch size
        if "batch_size" in context.additional_info:
            current = context.additional_info["batch_size"]
            reduced = max(int(current * self.reduction_factor), self.min_value)
            reduced_params["batch_size"] = reduced

        if reduced_params:
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.REDUCE_LOAD,
                message=f"Reduced parameters: {reduced_params}",
                recovered_data=reduced_params,
            )

        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.REDUCE_LOAD,
            message="No parameters available to reduce",
        )


class SkipStrategy(RecoveryStrategy):
    """Strategy that skips failed tasks and continues."""

    def __init__(self, skippable_stages: list[str] | None = None):
        self.skippable_stages = skippable_stages or [
            "analysis",
            "export",
            "visualization",
        ]

    @property
    def priority(self) -> int:
        return 40

    def can_recover(self, context: ErrorContext) -> bool:
        """Check if the current stage is skippable."""
        if not context.stage:
            return False
        return context.stage.lower() in [s.lower() for s in self.skippable_stages]

    def recover(self, context: ErrorContext) -> RecoveryResult:
        """Skip the current task."""
        return RecoveryResult(
            success=True,
            action_taken=RecoveryAction.SKIP,
            message=f"Skipping stage '{context.stage}' and continuing",
        )


class ErrorRecoveryManager:
    """Manages error recovery strategies.

    Features:
    - Multiple recovery strategies
    - Priority-based strategy selection
    - Automatic retry with backoff
    - Fallback mechanisms
    - Caching of successful results
    - Workload reduction
    """

    def __init__(self):
        self.strategies: list[RecoveryStrategy] = []
        self._register_default_strategies()
        import structlog
        self.logger = structlog.get_logger("recovery")

    def _register_default_strategies(self) -> None:
        """Register default recovery strategies."""
        self.register_strategy(RetryStrategy())
        self.register_strategy(FallbackStrategy())
        self.register_strategy(CacheStrategy())
        self.register_strategy(ReduceLoadStrategy())
        self.register_strategy(SkipStrategy())

    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """Register a recovery strategy."""
        self.strategies.append(strategy)
        # Sort by priority
        self.strategies.sort(key=lambda s: s.priority)

    def attempt_recovery(
        self,
        error: Exception,
        task_id: str | None = None,
        stage: str | None = None,
        attempt: int = 1,
        max_attempts: int = 3,
        **additional_info,
    ) -> RecoveryResult:
        """Attempt to recover from an error.

        Args:
            error: The exception that occurred
            task_id: ID of the failed task
            stage: Current pipeline stage
            attempt: Current attempt number
            max_attempts: Maximum retry attempts
            **additional_info: Additional context

        Returns:
            RecoveryResult with recovery outcome
        """
        context = ErrorContext(
            error=error,
            error_type=type(error).__name__,
            task_id=task_id,
            stage=stage,
            attempt=attempt,
            max_attempts=max_attempts,
            start_time=time.time(),
            additional_info=additional_info,
        )

        self.logger.info(
            "recovery.attempting",
            error_type=context.error_type,
            task_id=task_id,
            stage=stage,
            attempt=attempt,
        )

        # Try each strategy in priority order
        for strategy in self.strategies:
            if strategy.can_recover(context):
                result = strategy.recover(context)

                self.logger.info(
                    "recovery.result",
                    strategy=type(strategy).__name__,
                    action=result.action_taken.value,
                    success=result.success,
                )

                if result.success:
                    return result

        # All strategies failed
        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.ABORT,
            message="All recovery strategies exhausted",
        )

    def with_recovery(
        self,
        func: Callable[..., Any],
        task_id: str | None = None,
        stage: str | None = None,
        max_attempts: int = 3,
        **additional_info,
    ) -> tuple[bool, Any, RecoveryResult | None]:
        """Execute a function with automatic error recovery.

        Args:
            func: Function to execute
            task_id: Task identifier
            stage: Current stage
            max_attempts: Maximum retry attempts
            **additional_info: Additional context

        Returns:
            Tuple of (success, result, last_recovery_result)
        """
        attempt = 1
        last_recovery: RecoveryResult | None = None
        current_params = additional_info.copy()

        while attempt <= max_attempts:
            try:
                result = func(**current_params)
                return True, result, last_recovery
            except Exception as exc:
                self.logger.warning(
                    "recovery.error",
                    error=str(exc),
                    attempt=attempt,
                )

                recovery_result = self.attempt_recovery(
                    error=exc,
                    task_id=task_id,
                    stage=stage,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    **current_params,
                )
                last_recovery = recovery_result

                if not recovery_result.success:
                    return False, None, recovery_result

                # Apply recovered parameters
                if recovery_result.recovered_data:
                    if isinstance(recovery_result.recovered_data, dict):
                        current_params.update(recovery_result.recovered_data)

                if recovery_result.action_taken == RecoveryAction.SKIP:
                    return True, None, recovery_result

                attempt += 1

        return False, None, last_recovery


# ==================== UPDATED EXPORTS ====================


__all__ = [
    # Existing exports (keep original)
    "TaskType",
    "TaskStatus",
    "ValidationSeverity",
    "ValidationIssue",
    "AgentMetadata",
    "AgentConfiguration",
    "TaskDefinition",
    "AgentFile",
    "TaskResult",
    "ExecutionReport",
    "AgentInterpreter",
    "run_agent_file",
    # Enhanced parsing
    "AgentMDParseError",
    "MarkdownSection",
    "FrontmatterData",
    "EnhancedMarkdownParser",
    # Command validation
    "CommandValidationError",
    "ValidationResult",
    "ParameterSpec",
    "CommandSpec",
    "CommandValidator",
    # Error recovery
    "RecoveryAction",
    "RecoveryResult",
    "ErrorContext",
    "RecoveryStrategy",
    "RetryStrategy",
    "FallbackStrategy",
    "CacheStrategy",
    "ReduceLoadStrategy",
    "SkipStrategy",
    "ErrorRecoveryManager",
]
