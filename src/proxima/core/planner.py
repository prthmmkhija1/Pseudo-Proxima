"""Execution planner implementation with DAG-based task planning.

Planner delegates the reasoning step to an injected callable (LLM or local
model). It drives the execution state machine through planning states and
produces a Directed Acyclic Graph (DAG) of tasks for optimal parallel execution.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from proxima.core.state import ExecutionStateMachine
from proxima.utils.logging import get_logger

PlanFunction = Callable[[str], dict[str, Any]]


class TaskStatus(Enum):
    """Status of a task in the execution DAG."""
    
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class TaskNode:
    """A node in the execution DAG representing a single task.
    
    Attributes:
        task_id: Unique identifier for the task.
        action: The action to perform (e.g., 'create_circuit', 'execute').
        description: Human-readable description.
        parameters: Task-specific parameters.
        dependencies: List of task IDs this task depends on.
        dependents: List of task IDs that depend on this task.
        status: Current execution status.
        result: Task result after execution.
        priority: Execution priority (higher = earlier).
        estimated_duration_ms: Estimated execution time.
        retry_count: Number of retries allowed.
        timeout_s: Timeout in seconds.
        tags: Optional tags for categorization.
    """
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    priority: int = 0
    estimated_duration_ms: float = 100.0
    retry_count: int = 0
    timeout_s: float | None = None
    tags: list[str] = field(default_factory=list)

    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies completed)."""
        return self.status == TaskStatus.READY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "action": self.action,
            "description": self.description,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "status": self.status.name,
            "priority": self.priority,
            "estimated_duration_ms": self.estimated_duration_ms,
            "tags": self.tags,
        }


@dataclass
class ExecutionDAG:
    """Directed Acyclic Graph for task execution.
    
    Manages task dependencies and determines execution order for
    optimal parallel execution.
    """
    
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    root_tasks: list[str] = field(default_factory=list)
    _completed_tasks: set[str] = field(default_factory=set)

    def add_task(self, task: TaskNode) -> None:
        """Add a task to the DAG.
        
        Args:
            task: The task node to add.
        """
        self.nodes[task.task_id] = task
        
        # Track root tasks (no dependencies)
        if not task.dependencies:
            self.root_tasks.append(task.task_id)
        
        # Update dependents of dependencies
        for dep_id in task.dependencies:
            if dep_id in self.nodes:
                if task.task_id not in self.nodes[dep_id].dependents:
                    self.nodes[dep_id].dependents.append(task.task_id)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the DAG for cycles and missing dependencies.
        
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors: list[str] = []
        
        # Check for missing dependencies
        for task_id, task in self.nodes.items():
            for dep_id in task.dependencies:
                if dep_id not in self.nodes:
                    errors.append(
                        f"Task {task_id} depends on unknown task {dep_id}"
                    )
        
        # Check for cycles using DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self.nodes.get(node_id)
            if node:
                for dependent_id in node.dependents:
                    if dependent_id not in visited:
                        if has_cycle(dependent_id):
                            return True
                    elif dependent_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for task_id in self.nodes:
            if task_id not in visited:
                if has_cycle(task_id):
                    errors.append(f"Cycle detected involving task {task_id}")
                    break
        
        return len(errors) == 0, errors

    def get_ready_tasks(self) -> list[TaskNode]:
        """Get all tasks that are ready to execute.
        
        Returns:
            List of tasks with all dependencies completed.
        """
        ready: list[TaskNode] = []
        
        for task in self.nodes.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            all_deps_done = all(
                self.nodes.get(dep_id, TaskNode()).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if all_deps_done:
                ready.append(task)
        
        # Sort by priority (higher first)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def mark_ready(self, task_id: str) -> None:
        """Mark a task as ready for execution."""
        if task_id in self.nodes:
            self.nodes[task_id].status = TaskStatus.READY

    def mark_running(self, task_id: str) -> None:
        """Mark a task as currently running."""
        if task_id in self.nodes:
            self.nodes[task_id].status = TaskStatus.RUNNING

    def mark_completed(self, task_id: str, result: Any = None) -> list[str]:
        """Mark a task as completed and return newly ready tasks.
        
        Args:
            task_id: The completed task ID.
            result: The task result.
            
        Returns:
            List of task IDs that are now ready.
        """
        if task_id not in self.nodes:
            return []
        
        task = self.nodes[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result
        self._completed_tasks.add(task_id)
        
        # Find newly ready tasks
        newly_ready: list[str] = []
        for dependent_id in task.dependents:
            dependent = self.nodes.get(dependent_id)
            if dependent and dependent.status == TaskStatus.PENDING:
                # Check if all dependencies are now completed
                all_deps_done = all(
                    self.nodes.get(dep_id, TaskNode()).status == TaskStatus.COMPLETED
                    for dep_id in dependent.dependencies
                )
                if all_deps_done:
                    newly_ready.append(dependent_id)
        
        return newly_ready

    def mark_failed(self, task_id: str, error: str) -> list[str]:
        """Mark a task as failed and return tasks to skip.
        
        Args:
            task_id: The failed task ID.
            error: Error message.
            
        Returns:
            List of dependent task IDs that should be skipped.
        """
        if task_id not in self.nodes:
            return []
        
        task = self.nodes[task_id]
        task.status = TaskStatus.FAILED
        task.error = error
        
        # Find all downstream tasks to skip
        to_skip: list[str] = []
        queue = deque(task.dependents)
        
        while queue:
            dep_id = queue.popleft()
            if dep_id in to_skip:
                continue
            
            dep_task = self.nodes.get(dep_id)
            if dep_task:
                dep_task.status = TaskStatus.SKIPPED
                to_skip.append(dep_id)
                queue.extend(dep_task.dependents)
        
        return to_skip

    def get_execution_order(self) -> list[list[str]]:
        """Get tasks grouped by execution level (for parallel execution).
        
        Returns:
            List of task ID lists, where each inner list can run in parallel.
        """
        levels: list[list[str]] = []
        remaining = set(self.nodes.keys())
        completed: set[str] = set()
        
        while remaining:
            # Find tasks with all dependencies in completed set
            current_level: list[str] = []
            for task_id in remaining:
                task = self.nodes[task_id]
                if all(dep_id in completed for dep_id in task.dependencies):
                    current_level.append(task_id)
            
            if not current_level:
                # Cycle detected or error
                break
            
            # Sort by priority
            current_level.sort(
                key=lambda tid: self.nodes[tid].priority, reverse=True
            )
            levels.append(current_level)
            
            # Move to completed
            for task_id in current_level:
                completed.add(task_id)
                remaining.discard(task_id)
        
        return levels

    def get_critical_path(self) -> list[str]:
        """Calculate the critical path (longest duration path).
        
        Returns:
            List of task IDs on the critical path.
        """
        if not self.nodes:
            return []
        
        # Calculate earliest completion time for each task
        earliest: dict[str, float] = {}
        
        # Process in topological order
        for level in self.get_execution_order():
            for task_id in level:
                task = self.nodes[task_id]
                # Earliest start is max of all dependency completions
                earliest_start = 0.0
                for dep_id in task.dependencies:
                    if dep_id in earliest:
                        earliest_start = max(earliest_start, earliest[dep_id])
                earliest[task_id] = earliest_start + task.estimated_duration_ms
        
        # Find the task with latest completion
        if not earliest:
            return []
        
        end_task = max(earliest.keys(), key=lambda k: earliest[k])
        
        # Trace back the critical path
        path: list[str] = [end_task]
        current = end_task
        
        while True:
            task = self.nodes[current]
            if not task.dependencies:
                break
            
            # Find the dependency on the critical path
            max_time = 0.0
            critical_dep = None
            for dep_id in task.dependencies:
                if dep_id in earliest and earliest[dep_id] > max_time:
                    max_time = earliest[dep_id]
                    critical_dep = dep_id
            
            if critical_dep:
                path.insert(0, critical_dep)
                current = critical_dep
            else:
                break
        
        return path

    def estimate_total_duration(self, max_parallel: int = 1) -> float:
        """Estimate total execution duration with parallelism.
        
        Args:
            max_parallel: Maximum parallel task count.
            
        Returns:
            Estimated duration in milliseconds.
        """
        total_duration = 0.0
        
        for level in self.get_execution_order():
            # Split level into batches based on max_parallel
            level_tasks = [self.nodes[tid] for tid in level]
            for i in range(0, len(level_tasks), max_parallel):
                batch = level_tasks[i:i + max_parallel]
                # Batch duration is max of all tasks in batch
                batch_duration = max(
                    t.estimated_duration_ms for t in batch
                ) if batch else 0
                total_duration += batch_duration
        
        return total_duration

    def to_dict(self) -> dict[str, Any]:
        """Convert DAG to dictionary representation."""
        return {
            "nodes": {tid: t.to_dict() for tid, t in self.nodes.items()},
            "root_tasks": self.root_tasks,
            "execution_order": self.get_execution_order(),
            "critical_path": self.get_critical_path(),
            "estimated_duration_ms": self.estimate_total_duration(max_parallel=4),
        }


class Planner:
    """LLM-assisted planner that produces a DAG execution plan from an objective."""

    def __init__(
        self, state_machine: ExecutionStateMachine, plan_fn: PlanFunction | None = None
    ):
        self.state_machine = state_machine
        self.plan_fn = plan_fn
        self.logger = get_logger("planner")

    def plan(self, objective: str) -> dict[str, Any]:
        """Generate a plan for the given objective using the configured model."""

        self.state_machine.start()
        try:
            if not self.plan_fn:
                # Generate a meaningful plan when no model is provided
                plan = self._generate_default_plan(objective)
            else:
                plan = self.plan_fn(objective)

            self.state_machine.plan_complete()
            self.logger.info("planning.complete", plan_summary=list(plan.keys()))
            return plan
        except Exception as exc:  # noqa: BLE001
            self.state_machine.plan_failed()
            self.logger.error("planning.failed", error=str(exc))
            raise

    def plan_as_dag(self, objective: str) -> ExecutionDAG:
        """Generate a DAG-based execution plan for optimal parallelization.
        
        Args:
            objective: The high-level objective to plan for.
            
        Returns:
            ExecutionDAG with all tasks and dependencies.
        """
        self.state_machine.start()
        try:
            # Get the basic plan
            if self.plan_fn:
                basic_plan = self.plan_fn(objective)
            else:
                basic_plan = self._generate_default_plan(objective)
            
            # Convert to DAG
            dag = self._convert_plan_to_dag(basic_plan)
            
            # Validate DAG
            is_valid, errors = dag.validate()
            if not is_valid:
                self.logger.warning("dag.validation_warnings", errors=errors)
            
            self.state_machine.plan_complete()
            self.logger.info(
                "dag_planning.complete",
                tasks=len(dag.nodes),
                levels=len(dag.get_execution_order()),
                critical_path_length=len(dag.get_critical_path()),
            )
            return dag
            
        except Exception as exc:
            self.state_machine.plan_failed()
            self.logger.error("dag_planning.failed", error=str(exc))
            raise

    def _convert_plan_to_dag(self, plan: dict[str, Any]) -> ExecutionDAG:
        """Convert a basic plan to an ExecutionDAG.
        
        Args:
            plan: Basic plan dictionary with steps.
            
        Returns:
            ExecutionDAG with proper dependencies.
        """
        dag = ExecutionDAG()
        steps = plan.get("steps", [])
        backends = plan.get("backends", [])
        execution_mode = plan.get("execution_mode", "single")
        
        task_id_map: dict[int, str] = {}
        
        for step in steps:
            step_num = step.get("step", 0)
            action = step.get("action", "")
            
            if action == "execute" and execution_mode == "comparison" and backends:
                # Create parallel execution tasks for each backend
                execute_task_ids: list[str] = []
                
                for backend in backends:
                    task = TaskNode(
                        action="execute",
                        description=f"Execute on {backend}",
                        parameters={
                            **step.get("parameters", {}),
                            "backend": backend,
                        },
                        dependencies=self._get_dependencies(step_num, task_id_map),
                        priority=100 - step_num,
                        estimated_duration_ms=500.0,  # Backend execution typically slower
                        tags=["execution", backend],
                    )
                    dag.add_task(task)
                    execute_task_ids.append(task.task_id)
                
                # Store all execution tasks for dependency resolution
                task_id_map[step_num] = execute_task_ids[0]  # Primary reference
                
                # Create aggregation point for collect_results
                for tid in execute_task_ids[1:]:
                    # Mark as same step for dependency resolution
                    pass
                
            else:
                # Single task
                task = TaskNode(
                    action=action,
                    description=step.get("description", action),
                    parameters=step.get("parameters", {}),
                    dependencies=self._get_dependencies(step_num, task_id_map),
                    priority=100 - step_num,
                    estimated_duration_ms=self._estimate_task_duration(action),
                    tags=self._get_task_tags(action),
                )
                dag.add_task(task)
                task_id_map[step_num] = task.task_id
        
        return dag

    def _get_dependencies(
        self, step_num: int, task_id_map: dict[int, str]
    ) -> list[str]:
        """Get dependencies for a step based on step number."""
        if step_num <= 1:
            return []
        
        # Depend on previous step
        prev_step = step_num - 1
        if prev_step in task_id_map:
            return [task_id_map[prev_step]]
        
        return []

    def _estimate_task_duration(self, action: str) -> float:
        """Estimate task duration based on action type."""
        durations = {
            "create_circuit": 10.0,
            "execute": 500.0,
            "collect_results": 50.0,
            "compare": 100.0,
            "analyze": 200.0,
            "export": 100.0,
        }
        return durations.get(action, 100.0)

    def _get_task_tags(self, action: str) -> list[str]:
        """Get tags for a task based on action type."""
        tag_map = {
            "create_circuit": ["circuit", "initialization"],
            "execute": ["execution", "backend"],
            "collect_results": ["results", "aggregation"],
            "compare": ["analysis", "comparison"],
            "analyze": ["analysis", "insights"],
            "export": ["output", "export"],
        }
        return tag_map.get(action, [action])

    def _generate_default_plan(self, objective: str) -> dict[str, Any]:
        """Generate a default plan based on objective keywords.

        Analyzes the objective to determine:
        - Circuit type (bell, ghz, teleportation, etc.)
        - Execution mode (single, comparison)
        - Required backends
        - Number of shots
        """
        objective_lower = objective.lower()

        # Determine circuit type from objective
        circuit_type = "bell"  # default
        qubits = 2
        if "ghz" in objective_lower:
            circuit_type = "ghz"
            qubits = 3
            # Try to extract qubit count
            import re

            match = re.search(r"(\d+)[-\s]*qubit", objective_lower)
            if match:
                qubits = int(match.group(1))
        elif "teleport" in objective_lower:
            circuit_type = "teleportation"
            qubits = 3
        elif "superposition" in objective_lower or "hadamard" in objective_lower:
            circuit_type = "superposition"
            qubits = 1
        elif "entangle" in objective_lower:
            circuit_type = "bell"
            qubits = 2

        # Determine execution mode
        execution_mode = "single"
        backends = []
        if "compare" in objective_lower or "comparison" in objective_lower:
            execution_mode = "comparison"
            backends = ["cirq", "qiskit"]
        elif "all backend" in objective_lower:
            execution_mode = "comparison"
            backends = ["cirq", "qiskit", "lret"]

        # Extract shots if mentioned
        shots = 1024
        import re

        shots_match = re.search(r"(\d+)\s*shots?", objective_lower)
        if shots_match:
            shots = int(shots_match.group(1))

        # Build plan steps
        steps = [
            {
                "step": 1,
                "action": "create_circuit",
                "description": f"Create {circuit_type} circuit with {qubits} qubits",
                "parameters": {"circuit_type": circuit_type, "qubits": qubits},
            },
            {
                "step": 2,
                "action": "execute",
                "description": f"Execute circuit with {shots} shots",
                "parameters": {"shots": shots, "backends": backends or ["auto"]},
            },
            {
                "step": 3,
                "action": "collect_results",
                "description": "Collect and normalize results",
                "parameters": {},
            },
        ]

        if execution_mode == "comparison":
            steps.append(
                {
                    "step": 4,
                    "action": "compare",
                    "description": "Compare results across backends",
                    "parameters": {"backends": backends},
                }
            )

        return {
            "objective": objective,
            "circuit_type": circuit_type,
            "qubits": qubits,
            "shots": shots,
            "execution_mode": execution_mode,
            "backends": backends,
            "steps": steps,
            "generated_by": "default_planner",
        }


__all__ = [
    "TaskStatus",
    "TaskNode",
    "ExecutionDAG",
    "Planner",
    "PlanFunction",
    "PlanTemplate",
    "PlanValidator",
    "PlanOptimizer",
]


# ==============================================================================
# PLAN TEMPLATES
# ==============================================================================


class PlanTemplate:
    """Pre-defined plan templates for common scenarios.
    
    Provides ready-to-use plans for standard quantum computing tasks.
    """
    
    @staticmethod
    def single_circuit_execution(
        circuit_type: str = "bell",
        backend: str = "auto",
        shots: int = 1024,
        qubits: int = 2,
    ) -> dict[str, Any]:
        """Create a single circuit execution plan.
        
        Args:
            circuit_type: Type of circuit to create
            backend: Backend to use
            shots: Number of shots
            qubits: Number of qubits
            
        Returns:
            Plan dictionary
        """
        return {
            "objective": f"Execute {circuit_type} circuit",
            "circuit_type": circuit_type,
            "qubits": qubits,
            "shots": shots,
            "execution_mode": "single",
            "backends": [backend],
            "steps": [
                {
                    "step": 1,
                    "action": "create_circuit",
                    "description": f"Create {circuit_type} circuit with {qubits} qubits",
                    "parameters": {"circuit_type": circuit_type, "qubits": qubits},
                },
                {
                    "step": 2,
                    "action": "execute",
                    "description": f"Execute circuit with {shots} shots",
                    "parameters": {"shots": shots, "backends": [backend]},
                },
                {
                    "step": 3,
                    "action": "collect_results",
                    "description": "Collect and normalize results",
                    "parameters": {},
                },
            ],
        }
    
    @staticmethod
    def backend_comparison(
        backends: list[str],
        circuit_type: str = "bell",
        shots: int = 1024,
        qubits: int = 2,
    ) -> dict[str, Any]:
        """Create a multi-backend comparison plan.
        
        Args:
            backends: List of backends to compare
            circuit_type: Type of circuit
            shots: Number of shots per backend
            qubits: Number of qubits
            
        Returns:
            Plan dictionary
        """
        return {
            "objective": f"Compare {circuit_type} across backends",
            "circuit_type": circuit_type,
            "qubits": qubits,
            "shots": shots,
            "execution_mode": "comparison",
            "backends": backends,
            "steps": [
                {
                    "step": 1,
                    "action": "create_circuit",
                    "parameters": {"circuit_type": circuit_type, "qubits": qubits},
                },
                {
                    "step": 2,
                    "action": "execute",
                    "parameters": {"shots": shots, "backends": backends},
                },
                {
                    "step": 3,
                    "action": "collect_results",
                    "parameters": {},
                },
                {
                    "step": 4,
                    "action": "compare",
                    "parameters": {"backends": backends},
                },
            ],
        }
    
    @staticmethod
    def benchmark_suite(
        backend: str,
        circuit_types: list[str] | None = None,
        shots: int = 1024,
        runs_per_circuit: int = 3,
    ) -> dict[str, Any]:
        """Create a benchmark suite plan.
        
        Args:
            backend: Backend to benchmark
            circuit_types: Circuits to test (default: common set)
            shots: Shots per run
            runs_per_circuit: Number of runs per circuit
            
        Returns:
            Plan dictionary
        """
        if circuit_types is None:
            circuit_types = ["bell", "ghz", "superposition"]
        
        steps = []
        step_num = 1
        
        for circuit_type in circuit_types:
            steps.append({
                "step": step_num,
                "action": "create_circuit",
                "parameters": {"circuit_type": circuit_type},
            })
            step_num += 1
            
            steps.append({
                "step": step_num,
                "action": "benchmark",
                "parameters": {
                    "backend": backend,
                    "shots": shots,
                    "runs": runs_per_circuit,
                },
            })
            step_num += 1
        
        steps.append({
            "step": step_num,
            "action": "analyze",
            "parameters": {"type": "benchmark_summary"},
        })
        
        return {
            "objective": f"Benchmark {backend} with {len(circuit_types)} circuits",
            "execution_mode": "benchmark",
            "backends": [backend],
            "steps": steps,
        }
    
    @staticmethod
    def scaling_analysis(
        backend: str,
        qubit_range: tuple[int, int] = (2, 10),
        circuit_type: str = "ghz",
        shots: int = 1024,
    ) -> dict[str, Any]:
        """Create a scaling analysis plan.
        
        Args:
            backend: Backend to test
            qubit_range: (min_qubits, max_qubits) range
            circuit_type: Circuit type to scale
            shots: Shots per run
            
        Returns:
            Plan dictionary
        """
        min_q, max_q = qubit_range
        steps = []
        step_num = 1
        
        for qubits in range(min_q, max_q + 1):
            steps.append({
                "step": step_num,
                "action": "create_circuit",
                "parameters": {"circuit_type": circuit_type, "qubits": qubits},
            })
            step_num += 1
            
            steps.append({
                "step": step_num,
                "action": "execute",
                "parameters": {"backend": backend, "shots": shots},
            })
            step_num += 1
        
        steps.append({
            "step": step_num,
            "action": "analyze",
            "parameters": {"type": "scaling_analysis"},
        })
        
        return {
            "objective": f"Scaling analysis for {backend} ({min_q}-{max_q} qubits)",
            "execution_mode": "scaling",
            "backends": [backend],
            "steps": steps,
        }


# ==============================================================================
# PLAN VALIDATOR
# ==============================================================================


@dataclass
class ValidationResult:
    """Result of plan validation."""
    
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class PlanValidator:
    """Validates execution plans for correctness.
    
    Checks:
    - Required fields are present
    - Step dependencies are valid
    - Actions are recognized
    - Parameters are valid
    """
    
    REQUIRED_FIELDS = ["steps"]
    VALID_ACTIONS = [
        "create_circuit", "execute", "collect_results",
        "compare", "analyze", "export", "benchmark",
    ]
    
    def validate(self, plan: dict[str, Any]) -> ValidationResult:
        """Validate a plan.
        
        Args:
            plan: Plan to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []
        
        # Check required fields
        for field_name in self.REQUIRED_FIELDS:
            if field_name not in plan:
                errors.append(f"Missing required field: {field_name}")
        
        if "steps" not in plan:
            return ValidationResult(False, errors, warnings)
        
        steps = plan["steps"]
        if not isinstance(steps, list):
            errors.append("'steps' must be a list")
            return ValidationResult(False, errors, warnings)
        
        if len(steps) == 0:
            errors.append("Plan must have at least one step")
            return ValidationResult(False, errors, warnings)
        
        # Validate each step
        step_numbers = set()
        for i, step in enumerate(steps):
            step_num = step.get("step", i + 1)
            
            if step_num in step_numbers:
                warnings.append(f"Duplicate step number: {step_num}")
            step_numbers.add(step_num)
            
            action = step.get("action")
            if not action:
                errors.append(f"Step {step_num}: missing 'action' field")
            elif action not in self.VALID_ACTIONS:
                warnings.append(
                    f"Step {step_num}: unknown action '{action}' "
                    f"(known: {', '.join(self.VALID_ACTIONS)})"
                )
            
            # Validate action-specific parameters
            params = step.get("parameters", {})
            self._validate_action_params(action, params, step_num, errors, warnings)
        
        # Check for logical issues
        if plan.get("execution_mode") == "comparison":
            backends = plan.get("backends", [])
            if len(backends) < 2:
                warnings.append("Comparison mode requires at least 2 backends")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _validate_action_params(
        self,
        action: str | None,
        params: dict,
        step_num: int,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Validate parameters for specific actions."""
        if action == "create_circuit":
            if not params.get("circuit_type") and not params.get("qasm"):
                warnings.append(
                    f"Step {step_num}: create_circuit should have "
                    "'circuit_type' or 'qasm' parameter"
                )
        
        elif action == "execute":
            shots = params.get("shots")
            if shots is not None and (not isinstance(shots, int) or shots < 1):
                errors.append(f"Step {step_num}: 'shots' must be a positive integer")
        
        elif action == "benchmark":
            runs = params.get("runs")
            if runs is not None and (not isinstance(runs, int) or runs < 1):
                errors.append(f"Step {step_num}: 'runs' must be a positive integer")


# ==============================================================================
# PLAN OPTIMIZER
# ==============================================================================


class PlanOptimizer:
    """Optimizes execution plans for better performance.
    
    Optimizations:
    - Merge compatible steps
    - Reorder for parallelism
    - Remove redundant steps
    - Add caching hints
    """
    
    def optimize(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Optimize a plan.
        
        Args:
            plan: Plan to optimize
            
        Returns:
            Optimized plan
        """
        optimized = dict(plan)
        steps = list(plan.get("steps", []))
        
        # Apply optimizations
        steps = self._merge_compatible_creates(steps)
        steps = self._add_parallelism_hints(steps)
        steps = self._add_caching_hints(steps, plan)
        
        optimized["steps"] = steps
        optimized["_optimized"] = True
        
        return optimized
    
    def _merge_compatible_creates(
        self, steps: list[dict]
    ) -> list[dict]:
        """Merge consecutive create_circuit steps if compatible."""
        # For now, keep them separate for simplicity
        return steps
    
    def _add_parallelism_hints(self, steps: list[dict]) -> list[dict]:
        """Add hints about which steps can run in parallel."""
        # Find independent execution steps
        execute_indices = [
            i for i, s in enumerate(steps)
            if s.get("action") == "execute"
        ]
        
        if len(execute_indices) > 1:
            # Mark as parallelizable
            for idx in execute_indices:
                steps[idx]["_parallel_hint"] = True
        
        return steps
    
    def _add_caching_hints(
        self,
        steps: list[dict],
        plan: dict,
    ) -> list[dict]:
        """Add caching hints for reusable results."""
        # Mark circuit creation as cacheable
        for step in steps:
            if step.get("action") == "create_circuit":
                params = step.get("parameters", {})
                cache_key = f"{params.get('circuit_type', 'unknown')}_{params.get('qubits', 0)}"
                step["_cache_key"] = cache_key
        
        return steps
    
    def estimate_duration(self, plan: dict[str, Any]) -> float:
        """Estimate total plan execution duration in milliseconds.
        
        Args:
            plan: Plan to estimate
            
        Returns:
            Estimated duration in ms
        """
        total_ms = 0.0
        
        duration_estimates = {
            "create_circuit": 10.0,
            "execute": 500.0,
            "collect_results": 50.0,
            "compare": 100.0,
            "analyze": 200.0,
            "export": 100.0,
            "benchmark": 2000.0,
        }
        
        for step in plan.get("steps", []):
            action = step.get("action", "")
            base_duration = duration_estimates.get(action, 100.0)
            
            # Adjust for parameters
            params = step.get("parameters", {})
            if action == "execute":
                shots = params.get("shots", 1024)
                base_duration *= max(1, shots / 1024)
            elif action == "benchmark":
                runs = params.get("runs", 3)
                base_duration *= runs
            
            total_ms += base_duration
        
        return total_ms
