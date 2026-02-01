"""Task Planner for Natural Language Planning.

Phase 6: Natural Language Planning & Execution

Provides task planning capabilities including:
- LLM-based plan generation from natural language
- Structured execution plan creation
- Dependency analysis and ordering
- Plan validation and feasibility checking
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.task_planner")


class TaskCategory(Enum):
    """Categories of tasks the planner can handle."""
    BUILD = "build"
    ANALYZE = "analyze"
    MODIFY = "modify"
    EXECUTE = "execute"
    QUERY = "query"
    GIT = "git"
    FILE = "file"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PlanStatus(Enum):
    """Status of an execution plan."""
    DRAFT = "draft"
    VALIDATED = "validated"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    step_id: int
    tool: str
    arguments: Dict[str, Any]
    description: str
    depends_on: List[int] = field(default_factory=list)
    estimated_duration: int = 30  # seconds
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "description": self.description,
            "depends_on": self.depends_on,
            "estimated_duration": self.estimated_duration,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """Create from dictionary."""
        step = cls(
            step_id=data["step_id"],
            tool=data["tool"],
            arguments=data.get("arguments", {}),
            description=data.get("description", ""),
            depends_on=data.get("depends_on", []),
            estimated_duration=data.get("estimated_duration", 30),
        )
        if "status" in data:
            step.status = StepStatus(data["status"])
        return step
    
    @property
    def can_retry(self) -> bool:
        """Check if step can be retried."""
        return self.retries < self.max_retries


@dataclass
class ExecutionPlan:
    """A complete execution plan."""
    plan_id: str
    description: str
    steps: List[PlanStep]
    category: TaskCategory
    created_at: datetime = field(default_factory=datetime.now)
    status: PlanStatus = PlanStatus.DRAFT
    parallel_groups: List[List[int]] = field(default_factory=list)
    estimated_total_duration: int = 0
    user_request: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived values after init."""
        if not self.parallel_groups:
            self._compute_parallel_groups()
        if not self.estimated_total_duration:
            self._compute_total_duration()
    
    def _compute_parallel_groups(self) -> None:
        """Compute groups of steps that can run in parallel."""
        if not self.steps:
            return
        
        # Topological sort with level assignment
        completed: Set[int] = set()
        groups: List[List[int]] = []
        remaining = {s.step_id for s in self.steps}
        
        while remaining:
            # Find steps whose dependencies are all complete
            ready = []
            for step in self.steps:
                if step.step_id in remaining:
                    deps_met = all(d in completed for d in step.depends_on)
                    if deps_met:
                        ready.append(step.step_id)
            
            if not ready:
                # Circular dependency detected, break with remaining
                ready = list(remaining)
            
            groups.append(ready)
            completed.update(ready)
            remaining -= set(ready)
        
        self.parallel_groups = groups
    
    def _compute_total_duration(self) -> None:
        """Compute estimated total duration."""
        if not self.parallel_groups:
            self.estimated_total_duration = sum(s.estimated_duration for s in self.steps)
            return
        
        # Duration is max of each parallel group
        total = 0
        step_map = {s.step_id: s for s in self.steps}
        
        for group in self.parallel_groups:
            group_duration = max(
                step_map[sid].estimated_duration 
                for sid in group 
                if sid in step_map
            ) if group else 0
            total += group_duration
        
        self.estimated_total_duration = total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "category": self.category.value,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "parallel_groups": self.parallel_groups,
            "estimated_total_duration": self.estimated_total_duration,
            "user_request": self.user_request,
            "context": self.context,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        """Create from dictionary."""
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            plan_id=data["plan_id"],
            description=data["description"],
            steps=steps,
            category=TaskCategory(data.get("category", "unknown")),
            parallel_groups=data.get("parallel_groups", []),
            estimated_total_duration=data.get("estimated_total_duration", 0),
            user_request=data.get("user_request", ""),
            context=data.get("context", {}),
        )
    
    def get_step(self, step_id: int) -> Optional[PlanStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps that are ready to execute."""
        ready = []
        completed_ids = {
            s.step_id for s in self.steps 
            if s.status == StepStatus.COMPLETED
        }
        
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                deps_met = all(d in completed_ids for d in step.depends_on)
                if deps_met:
                    ready.append(step)
        
        return ready
    
    @property
    def progress(self) -> float:
        """Calculate progress as percentage."""
        if not self.steps:
            return 0.0
        
        completed = sum(
            s.estimated_duration 
            for s in self.steps 
            if s.status == StepStatus.COMPLETED
        )
        total = sum(s.estimated_duration for s in self.steps)
        
        return (completed / total * 100) if total > 0 else 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            for s in self.steps
        )
    
    @property
    def has_failed(self) -> bool:
        """Check if any step has failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)


@dataclass
class IntentRecognitionResult:
    """Result of intent recognition."""
    category: TaskCategory
    confidence: float
    entities: Dict[str, Any]
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None


class TaskPlanner:
    """Plan tasks from natural language requests.
    
    Features:
    - Intent recognition and classification
    - Dependency analysis
    - Tool selection
    - Plan validation
    
    Example:
        >>> planner = TaskPlanner()
        >>> 
        >>> # Generate a plan
        >>> plan = await planner.create_plan(
        ...     "Build LRET Cirq backend and run tests"
        ... )
        >>> 
        >>> # Validate the plan
        >>> is_valid, errors = planner.validate_plan(plan)
    """
    
    # Known tools and their categories
    KNOWN_TOOLS = {
        "execute_command": TaskCategory.EXECUTE,
        "build_backend": TaskCategory.BUILD,
        "run_script": TaskCategory.EXECUTE,
        "read_file": TaskCategory.FILE,
        "write_file": TaskCategory.FILE,
        "list_directory": TaskCategory.FILE,
        "git_clone": TaskCategory.GIT,
        "git_pull": TaskCategory.GIT,
        "git_push": TaskCategory.GIT,
        "git_commit": TaskCategory.GIT,
        "git_status": TaskCategory.GIT,
        "analyze_code": TaskCategory.ANALYZE,
        "search_files": TaskCategory.FILE,
        "modify_backend": TaskCategory.MODIFY,
        "get_system_info": TaskCategory.SYSTEM,
        "install_package": TaskCategory.EXECUTE,
    }
    
    # Pattern-based intent recognition
    INTENT_PATTERNS = [
        (r"build\s+(?:the\s+)?(\w+)(?:\s+backend)?", TaskCategory.BUILD, "backend_name"),
        (r"run\s+(?:the\s+)?tests?(?:\s+for\s+(\w+))?", TaskCategory.EXECUTE, "test_target"),
        (r"install\s+(\w+)", TaskCategory.EXECUTE, "package_name"),
        (r"show\s+(?:the\s+)?(?:git\s+)?status", TaskCategory.GIT, None),
        (r"commit\s+(?:the\s+)?changes?(?:\s+with\s+message\s+['\"](.+)['\"])?", TaskCategory.GIT, "commit_message"),
        (r"push\s+(?:to\s+)?(?:the\s+)?(?:remote|origin)?", TaskCategory.GIT, None),
        (r"pull\s+(?:from\s+)?(?:the\s+)?(?:remote|origin)?", TaskCategory.GIT, None),
        (r"clone\s+(.+)", TaskCategory.GIT, "repo_url"),
        (r"read\s+(?:the\s+)?file\s+(.+)", TaskCategory.FILE, "file_path"),
        (r"list\s+(?:the\s+)?(?:files\s+in\s+)?(.+)", TaskCategory.FILE, "directory"),
        (r"search\s+(?:for\s+)?['\"](.+)['\"]", TaskCategory.FILE, "search_query"),
        (r"analyze\s+(?:the\s+)?(?:code|backend)\s*(.+)?", TaskCategory.ANALYZE, "target"),
        (r"what\s+(?:files?\s+)?changed", TaskCategory.GIT, None),
        (r"check\s+(?:the\s+)?compilation|warnings?|errors?", TaskCategory.ANALYZE, None),
    ]
    
    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        available_tools: Optional[List[str]] = None,
    ):
        """Initialize the task planner.
        
        Args:
            llm_provider: Optional LLM provider for advanced planning
            available_tools: List of available tool names
        """
        self.llm_provider = llm_provider
        self.available_tools = available_tools or list(self.KNOWN_TOOLS.keys())
        
        # Plan templates for common tasks
        self._plan_templates: Dict[str, List[Dict[str, Any]]] = {}
        self._load_default_templates()
        
        logger.info("TaskPlanner initialized")
    
    def _load_default_templates(self) -> None:
        """Load default plan templates."""
        self._plan_templates = {
            "build_and_test": [
                {
                    "tool": "build_backend",
                    "arguments": {"backend_name": "{backend_name}"},
                    "description": "Build {backend_name} backend",
                    "depends_on": [],
                },
                {
                    "tool": "execute_command",
                    "arguments": {"command": "python -m pytest tests/backends/test_{backend_name}.py -v"},
                    "description": "Run tests for {backend_name}",
                    "depends_on": [1],
                },
            ],
            "git_commit_push": [
                {
                    "tool": "git_status",
                    "arguments": {},
                    "description": "Check git status",
                    "depends_on": [],
                },
                {
                    "tool": "execute_command",
                    "arguments": {"command": "git add -A"},
                    "description": "Stage all changes",
                    "depends_on": [1],
                },
                {
                    "tool": "git_commit",
                    "arguments": {"message": "{commit_message}"},
                    "description": "Commit changes",
                    "depends_on": [2],
                },
                {
                    "tool": "git_push",
                    "arguments": {},
                    "description": "Push to remote",
                    "depends_on": [3],
                },
            ],
            "install_and_build": [
                {
                    "tool": "install_package",
                    "arguments": {"package": "{package_name}"},
                    "description": "Install {package_name}",
                    "depends_on": [],
                },
                {
                    "tool": "build_backend",
                    "arguments": {"backend_name": "{backend_name}"},
                    "description": "Build {backend_name} backend",
                    "depends_on": [1],
                },
            ],
        }
    
    def recognize_intent(self, request: str) -> IntentRecognitionResult:
        """Recognize intent from natural language request.
        
        Args:
            request: User's natural language request
            
        Returns:
            IntentRecognitionResult with category and entities
        """
        request_lower = request.lower().strip()
        entities: Dict[str, Any] = {}
        
        # Try pattern matching first
        for pattern, category, entity_name in self.INTENT_PATTERNS:
            match = re.search(pattern, request_lower)
            if match:
                if entity_name and match.groups():
                    entities[entity_name] = match.group(1)
                
                return IntentRecognitionResult(
                    category=category,
                    confidence=0.9,
                    entities=entities,
                )
        
        # Fallback to keyword-based classification
        category, confidence = self._classify_by_keywords(request_lower)
        
        if confidence < 0.5:
            return IntentRecognitionResult(
                category=TaskCategory.UNKNOWN,
                confidence=confidence,
                entities=entities,
                requires_clarification=True,
                clarification_prompt="I'm not sure what you want to do. Could you please clarify?",
            )
        
        return IntentRecognitionResult(
            category=category,
            confidence=confidence,
            entities=entities,
        )
    
    def _classify_by_keywords(self, request: str) -> Tuple[TaskCategory, float]:
        """Classify request by keyword presence."""
        keyword_scores: Dict[TaskCategory, float] = {cat: 0.0 for cat in TaskCategory}
        
        keywords = {
            TaskCategory.BUILD: ["build", "compile", "make", "create"],
            TaskCategory.EXECUTE: ["run", "execute", "test", "install", "pip"],
            TaskCategory.GIT: ["git", "commit", "push", "pull", "clone", "branch", "merge"],
            TaskCategory.FILE: ["file", "read", "write", "list", "directory", "folder", "search"],
            TaskCategory.ANALYZE: ["analyze", "check", "inspect", "review", "warnings", "errors"],
            TaskCategory.MODIFY: ["modify", "change", "edit", "update", "fix"],
            TaskCategory.QUERY: ["what", "show", "display", "get", "status", "info"],
            TaskCategory.SYSTEM: ["system", "os", "platform", "environment"],
        }
        
        for category, words in keywords.items():
            for word in words:
                if word in request:
                    keyword_scores[category] += 1.0 / len(words)
        
        best_category = max(keyword_scores, key=keyword_scores.get)
        best_score = keyword_scores[best_category]
        
        return best_category, min(best_score, 1.0)
    
    def generate_plan_id(self) -> str:
        """Generate a unique plan ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = uuid.uuid4().hex[:6]
        return f"plan_{timestamp}_{random_suffix}"
    
    async def create_plan(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Create an execution plan from natural language request.
        
        Args:
            request: User's natural language request
            context: Additional context information
            
        Returns:
            ExecutionPlan with steps to execute
        """
        context = context or {}
        
        # Recognize intent
        intent = self.recognize_intent(request)
        
        # Try to use LLM if available and intent is unclear
        if self.llm_provider and intent.confidence < 0.7:
            plan = await self._create_plan_with_llm(request, context)
            if plan:
                return plan
        
        # Generate plan based on intent
        plan = self._create_plan_from_intent(request, intent, context)
        
        return plan
    
    def _create_plan_from_intent(
        self,
        request: str,
        intent: IntentRecognitionResult,
        context: Dict[str, Any],
    ) -> ExecutionPlan:
        """Create a plan based on recognized intent."""
        steps: List[PlanStep] = []
        
        if intent.category == TaskCategory.BUILD:
            backend_name = intent.entities.get("backend_name", "unknown")
            # Check if it includes testing
            if "test" in request.lower():
                steps = self._create_build_and_test_steps(backend_name)
            else:
                steps = [
                    PlanStep(
                        step_id=1,
                        tool="build_backend",
                        arguments={"backend_name": backend_name},
                        description=f"Build {backend_name} backend",
                        estimated_duration=120,
                    )
                ]
        
        elif intent.category == TaskCategory.EXECUTE:
            if "test" in request.lower():
                target = intent.entities.get("test_target", "")
                steps = self._create_test_steps(target)
            elif "install" in request.lower():
                package = intent.entities.get("package_name", "")
                steps = [
                    PlanStep(
                        step_id=1,
                        tool="install_package",
                        arguments={"package": package},
                        description=f"Install {package}",
                        estimated_duration=60,
                    )
                ]
            else:
                # Generic command execution
                steps = [
                    PlanStep(
                        step_id=1,
                        tool="execute_command",
                        arguments={"command": self._extract_command(request)},
                        description="Execute command",
                        estimated_duration=30,
                    )
                ]
        
        elif intent.category == TaskCategory.GIT:
            steps = self._create_git_steps(request, intent)
        
        elif intent.category == TaskCategory.FILE:
            steps = self._create_file_steps(request, intent)
        
        elif intent.category == TaskCategory.ANALYZE:
            target = intent.entities.get("target", "")
            steps = [
                PlanStep(
                    step_id=1,
                    tool="analyze_code",
                    arguments={"target": target},
                    description=f"Analyze {target or 'code'}",
                    estimated_duration=60,
                )
            ]
        
        else:
            # Unknown or query - create a simple info step
            steps = [
                PlanStep(
                    step_id=1,
                    tool="get_system_info",
                    arguments={},
                    description="Get system information",
                    estimated_duration=5,
                )
            ]
        
        return ExecutionPlan(
            plan_id=self.generate_plan_id(),
            description=self._generate_plan_description(request, intent),
            steps=steps,
            category=intent.category,
            user_request=request,
            context=context,
        )
    
    def _create_build_and_test_steps(self, backend_name: str) -> List[PlanStep]:
        """Create steps for build and test."""
        return [
            PlanStep(
                step_id=1,
                tool="build_backend",
                arguments={"backend_name": backend_name},
                description=f"Build {backend_name} backend",
                estimated_duration=120,
            ),
            PlanStep(
                step_id=2,
                tool="execute_command",
                arguments={"command": f"python -m pytest tests/backends/test_{backend_name}.py -v"},
                description=f"Run tests for {backend_name}",
                depends_on=[1],
                estimated_duration=60,
            ),
        ]
    
    def _create_test_steps(self, target: str) -> List[PlanStep]:
        """Create steps for running tests."""
        if target:
            test_path = f"tests/backends/test_{target}.py"
            cmd = f"python -m pytest {test_path} -v"
        else:
            cmd = "python -m pytest tests/ -v"
        
        return [
            PlanStep(
                step_id=1,
                tool="execute_command",
                arguments={"command": cmd},
                description=f"Run tests{' for ' + target if target else ''}",
                estimated_duration=120,
            )
        ]
    
    def _create_git_steps(
        self,
        request: str,
        intent: IntentRecognitionResult,
    ) -> List[PlanStep]:
        """Create steps for git operations."""
        request_lower = request.lower()
        steps = []
        
        if "status" in request_lower or "changed" in request_lower:
            steps = [
                PlanStep(
                    step_id=1,
                    tool="git_status",
                    arguments={},
                    description="Check git status",
                    estimated_duration=5,
                )
            ]
        
        elif "commit" in request_lower and "push" in request_lower:
            message = intent.entities.get("commit_message", "Update")
            steps = [
                PlanStep(
                    step_id=1,
                    tool="execute_command",
                    arguments={"command": "git add -A"},
                    description="Stage all changes",
                    estimated_duration=5,
                ),
                PlanStep(
                    step_id=2,
                    tool="git_commit",
                    arguments={"message": message},
                    description="Commit changes",
                    depends_on=[1],
                    estimated_duration=5,
                ),
                PlanStep(
                    step_id=3,
                    tool="git_push",
                    arguments={},
                    description="Push to remote",
                    depends_on=[2],
                    estimated_duration=30,
                ),
            ]
        
        elif "commit" in request_lower:
            message = intent.entities.get("commit_message", "Update")
            steps = [
                PlanStep(
                    step_id=1,
                    tool="execute_command",
                    arguments={"command": "git add -A"},
                    description="Stage all changes",
                    estimated_duration=5,
                ),
                PlanStep(
                    step_id=2,
                    tool="git_commit",
                    arguments={"message": message},
                    description="Commit changes",
                    depends_on=[1],
                    estimated_duration=5,
                ),
            ]
        
        elif "push" in request_lower:
            steps = [
                PlanStep(
                    step_id=1,
                    tool="git_push",
                    arguments={},
                    description="Push to remote",
                    estimated_duration=30,
                )
            ]
        
        elif "pull" in request_lower:
            steps = [
                PlanStep(
                    step_id=1,
                    tool="git_pull",
                    arguments={},
                    description="Pull from remote",
                    estimated_duration=30,
                )
            ]
        
        elif "clone" in request_lower:
            url = intent.entities.get("repo_url", "")
            steps = [
                PlanStep(
                    step_id=1,
                    tool="git_clone",
                    arguments={"url": url},
                    description=f"Clone repository",
                    estimated_duration=60,
                )
            ]
        
        return steps or [
            PlanStep(
                step_id=1,
                tool="git_status",
                arguments={},
                description="Check git status",
                estimated_duration=5,
            )
        ]
    
    def _create_file_steps(
        self,
        request: str,
        intent: IntentRecognitionResult,
    ) -> List[PlanStep]:
        """Create steps for file operations."""
        request_lower = request.lower()
        
        if "read" in request_lower:
            path = intent.entities.get("file_path", "")
            return [
                PlanStep(
                    step_id=1,
                    tool="read_file",
                    arguments={"path": path},
                    description=f"Read file {path}",
                    estimated_duration=5,
                )
            ]
        
        elif "list" in request_lower:
            directory = intent.entities.get("directory", ".")
            return [
                PlanStep(
                    step_id=1,
                    tool="list_directory",
                    arguments={"path": directory},
                    description=f"List directory {directory}",
                    estimated_duration=5,
                )
            ]
        
        elif "search" in request_lower:
            query = intent.entities.get("search_query", "")
            return [
                PlanStep(
                    step_id=1,
                    tool="search_files",
                    arguments={"query": query},
                    description=f"Search for '{query}'",
                    estimated_duration=30,
                )
            ]
        
        return [
            PlanStep(
                step_id=1,
                tool="list_directory",
                arguments={"path": "."},
                description="List current directory",
                estimated_duration=5,
            )
        ]
    
    def _extract_command(self, request: str) -> str:
        """Extract command from request."""
        # Look for quoted command
        quoted = re.search(r'["\'](.+?)["\']', request)
        if quoted:
            return quoted.group(1)
        
        # Look for command after "run" or "execute"
        match = re.search(r'(?:run|execute)\s+(.+)', request, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return request
    
    def _generate_plan_description(
        self,
        request: str,
        intent: IntentRecognitionResult,
    ) -> str:
        """Generate a description for the plan."""
        category_descriptions = {
            TaskCategory.BUILD: "Build",
            TaskCategory.EXECUTE: "Execute",
            TaskCategory.GIT: "Git operations",
            TaskCategory.FILE: "File operations",
            TaskCategory.ANALYZE: "Analyze",
            TaskCategory.MODIFY: "Modify",
            TaskCategory.QUERY: "Query",
            TaskCategory.SYSTEM: "System operations",
        }
        
        prefix = category_descriptions.get(intent.category, "Execute")
        
        # Use first 50 chars of request if short enough
        if len(request) <= 60:
            return request
        
        return f"{prefix}: {request[:50]}..."
    
    async def _create_plan_with_llm(
        self,
        request: str,
        context: Dict[str, Any],
    ) -> Optional[ExecutionPlan]:
        """Create a plan using LLM."""
        if not self.llm_provider:
            return None
        
        # Build prompt for LLM
        tools_description = self._get_tools_description()
        
        prompt = f"""You are a task planner for Proxima, a quantum computing TUI application.
Given the user's request, generate an execution plan as JSON.

Available tools:
{tools_description}

User request: {request}

Generate a JSON plan with this structure:
{{
  "description": "Brief description of the plan",
  "steps": [
    {{
      "step_id": 1,
      "tool": "tool_name",
      "arguments": {{"arg1": "value1"}},
      "description": "What this step does",
      "depends_on": [],
      "estimated_duration": 30
    }}
  ]
}}

Rules:
1. Only use tools from the available list
2. Set depends_on to list of step_ids that must complete first
3. Estimate duration in seconds
4. Keep steps minimal and focused

JSON plan:"""
        
        try:
            response = await self.llm_provider.complete(prompt)
            plan_data = json.loads(response)
            
            steps = [
                PlanStep.from_dict({**s, "step_id": i + 1})
                for i, s in enumerate(plan_data.get("steps", []))
            ]
            
            return ExecutionPlan(
                plan_id=self.generate_plan_id(),
                description=plan_data.get("description", request),
                steps=steps,
                category=TaskCategory.UNKNOWN,
                user_request=request,
                context=context,
            )
        except Exception as e:
            logger.warning(f"LLM plan generation failed: {e}")
            return None
    
    def _get_tools_description(self) -> str:
        """Get description of available tools."""
        descriptions = {
            "execute_command": "Execute a shell command",
            "build_backend": "Build a quantum backend (args: backend_name)",
            "run_script": "Run a script file (args: path, arguments)",
            "read_file": "Read file contents (args: path)",
            "write_file": "Write to file (args: path, content)",
            "list_directory": "List directory contents (args: path)",
            "git_clone": "Clone git repository (args: url, destination)",
            "git_pull": "Pull from remote",
            "git_push": "Push to remote",
            "git_commit": "Commit changes (args: message)",
            "git_status": "Get git status",
            "analyze_code": "Analyze code (args: target)",
            "search_files": "Search in files (args: query, path)",
            "install_package": "Install package (args: package)",
            "get_system_info": "Get system information",
        }
        
        lines = []
        for tool in self.available_tools:
            desc = descriptions.get(tool, "No description")
            lines.append(f"- {tool}: {desc}")
        
        return "\n".join(lines)
    
    def validate_plan(self, plan: ExecutionPlan) -> Tuple[bool, List[str]]:
        """Validate an execution plan.
        
        Args:
            plan: Plan to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check for empty plan
        if not plan.steps:
            errors.append("Plan has no steps")
            return False, errors
        
        # Check step IDs are unique
        step_ids = [s.step_id for s in plan.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")
        
        # Check dependencies reference valid steps
        for step in plan.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id} depends on non-existent step {dep}")
                if dep >= step.step_id:
                    errors.append(f"Step {step.step_id} depends on later step {dep}")
        
        # Check tools are available
        for step in plan.steps:
            if step.tool not in self.available_tools:
                errors.append(f"Step {step.step_id} uses unknown tool: {step.tool}")
        
        # Check for circular dependencies
        if self._has_circular_dependency(plan):
            errors.append("Plan has circular dependencies")
        
        return len(errors) == 0, errors
    
    def _has_circular_dependency(self, plan: ExecutionPlan) -> bool:
        """Check for circular dependencies in plan."""
        # Build adjacency list
        graph: Dict[int, Set[int]] = {s.step_id: set(s.depends_on) for s in plan.steps}
        
        # DFS to detect cycle
        visited: Set[int] = set()
        rec_stack: Set[int] = set()
        
        def has_cycle(node: int) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return True
        
        return False
    
    def add_step(
        self,
        plan: ExecutionPlan,
        step: PlanStep,
        after_step_id: Optional[int] = None,
    ) -> ExecutionPlan:
        """Add a step to an existing plan.
        
        Args:
            plan: Plan to modify
            step: Step to add
            after_step_id: Insert after this step (None = append)
            
        Returns:
            Modified plan
        """
        if after_step_id is None:
            plan.steps.append(step)
        else:
            # Find position and insert
            for i, s in enumerate(plan.steps):
                if s.step_id == after_step_id:
                    plan.steps.insert(i + 1, step)
                    break
        
        # Renumber steps
        for i, s in enumerate(plan.steps):
            s.step_id = i + 1
        
        # Recompute parallel groups
        plan._compute_parallel_groups()
        plan._compute_total_duration()
        
        return plan
    
    def remove_step(self, plan: ExecutionPlan, step_id: int) -> ExecutionPlan:
        """Remove a step from a plan.
        
        Args:
            plan: Plan to modify
            step_id: Step ID to remove
            
        Returns:
            Modified plan
        """
        # Remove step
        plan.steps = [s for s in plan.steps if s.step_id != step_id]
        
        # Update dependencies
        for step in plan.steps:
            step.depends_on = [d for d in step.depends_on if d != step_id]
        
        # Renumber steps
        old_to_new = {}
        for i, s in enumerate(plan.steps):
            old_to_new[s.step_id] = i + 1
            s.step_id = i + 1
        
        # Update dependency references
        for step in plan.steps:
            step.depends_on = [old_to_new.get(d, d) for d in step.depends_on]
        
        # Recompute
        plan._compute_parallel_groups()
        plan._compute_total_duration()
        
        return plan


# Global instance
_planner: Optional[TaskPlanner] = None


def get_task_planner() -> TaskPlanner:
    """Get the global TaskPlanner instance."""
    global _planner
    if _planner is None:
        _planner = TaskPlanner()
    return _planner
