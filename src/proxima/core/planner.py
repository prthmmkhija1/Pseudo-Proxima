"""Execution planner implementation with DAG-based task planning.

Planner delegates the reasoning step to an injected callable (LLM or local
model). It drives the execution state machine through planning states and
produces a Directed Acyclic Graph (DAG) of tasks for optimal parallel execution.

Enhanced with:
- LLM-assisted planning with multiple strategies
- Objective decomposition for complex tasks
- Plan explanation and risk assessment
- Iterative refinement based on feedback
"""

from __future__ import annotations

import json
import time
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


# ==============================================================================
# COMPLEX DAG VALIDATION (2% Gap Coverage)
# ==============================================================================


@dataclass
class ResourceConstraint:
    """Defines a resource constraint for task validation."""
    
    resource_id: str
    resource_type: str  # "gpu", "cpu", "memory", "quantum_simulator", etc.
    amount: float
    exclusive: bool = False  # If True, no other task can use this resource concurrently


@dataclass
class ParallelConstraint:
    """Defines parallel execution constraints between tasks."""
    
    task_ids: list[str]
    constraint_type: str  # "mutex", "same_group", "different_group", "ordered"
    reason: str = ""


@dataclass
class SemanticRule:
    """Defines semantic validation rules."""
    
    rule_id: str
    description: str
    validator: Callable[["ExecutionDAG", TaskNode], tuple[bool, str | None]]


@dataclass 
class ValidationResult:
    """Comprehensive DAG validation result."""
    
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    resource_conflicts: list[dict[str, Any]] = field(default_factory=list)
    parallel_constraint_violations: list[dict[str, Any]] = field(default_factory=list)
    semantic_issues: list[dict[str, Any]] = field(default_factory=list)
    validation_time_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "resource_conflicts": self.resource_conflicts,
            "parallel_constraint_violations": self.parallel_constraint_violations,
            "semantic_issues": self.semantic_issues,
            "validation_time_ms": self.validation_time_ms,
        }
    
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            suggestions=self.suggestions + other.suggestions,
            resource_conflicts=self.resource_conflicts + other.resource_conflicts,
            parallel_constraint_violations=(
                self.parallel_constraint_violations + other.parallel_constraint_violations
            ),
            semantic_issues=self.semantic_issues + other.semantic_issues,
            validation_time_ms=self.validation_time_ms + other.validation_time_ms,
        )


class ComplexDAGValidator:
    """Advanced DAG validator with comprehensive validation capabilities.
    
    Features:
    - Parallel execution constraint validation
    - Resource conflict detection
    - Semantic validation rules
    - Dependency strength typing
    - Execution path analysis
    - Circular dependency detection with path reporting
    """
    
    # Built-in semantic rules
    BUILTIN_RULES: list[SemanticRule] = []
    
    def __init__(
        self,
        resource_pool: dict[str, float] | None = None,
        custom_rules: list[SemanticRule] | None = None,
        logger: Any = None,
    ) -> None:
        """Initialize validator.
        
        Args:
            resource_pool: Available resources {resource_id: amount}
            custom_rules: Additional semantic rules
            logger: Logger instance
        """
        self._resource_pool = resource_pool or {}
        self._custom_rules = custom_rules or []
        self._parallel_constraints: list[ParallelConstraint] = []
        self._logger = logger or get_logger("dag_validator")
    
    def add_parallel_constraint(self, constraint: ParallelConstraint) -> None:
        """Add a parallel execution constraint."""
        self._parallel_constraints.append(constraint)
    
    def set_resource_pool(self, pool: dict[str, float]) -> None:
        """Set available resource pool."""
        self._resource_pool = pool
    
    def validate_comprehensive(
        self,
        dag: ExecutionDAG,
        task_resources: dict[str, list[ResourceConstraint]] | None = None,
    ) -> ValidationResult:
        """Perform comprehensive DAG validation.
        
        Args:
            dag: DAG to validate
            task_resources: Resource requirements per task
            
        Returns:
            Comprehensive validation result
        """
        start_time = time.perf_counter()
        result = ValidationResult(is_valid=True)
        
        # 1. Basic validation (cycles, missing deps)
        basic_result = self._validate_basic(dag)
        result = result.merge(basic_result)
        
        # 2. Parallel constraint validation
        parallel_result = self._validate_parallel_constraints(dag)
        result = result.merge(parallel_result)
        
        # 3. Resource conflict detection
        if task_resources:
            resource_result = self._validate_resources(dag, task_resources)
            result = result.merge(resource_result)
        
        # 4. Semantic validation
        semantic_result = self._validate_semantics(dag)
        result = result.merge(semantic_result)
        
        # 5. Dependency analysis
        dep_result = self._analyze_dependencies(dag)
        result = result.merge(dep_result)
        
        result.validation_time_ms = (time.perf_counter() - start_time) * 1000
        result.is_valid = len(result.errors) == 0
        
        return result
    
    def _validate_basic(self, dag: ExecutionDAG) -> ValidationResult:
        """Validate basic DAG structure."""
        result = ValidationResult(is_valid=True)
        
        # Check for missing dependencies
        for task_id, task in dag.nodes.items():
            for dep_id in task.dependencies:
                if dep_id not in dag.nodes:
                    result.errors.append(
                        f"Task '{task_id}' depends on unknown task '{dep_id}'"
                    )
        
        # Enhanced cycle detection with path reporting
        cycle_path = self._find_cycle_path(dag)
        if cycle_path:
            result.errors.append(
                f"Cycle detected: {' -> '.join(cycle_path)}"
            )
        
        # Check for orphaned tasks (no path from root)
        reachable = self._find_reachable_tasks(dag)
        orphaned = set(dag.nodes.keys()) - reachable - set(dag.root_tasks)
        for task_id in orphaned:
            result.warnings.append(
                f"Task '{task_id}' may be unreachable from root tasks"
            )
        
        return result
    
    def _find_cycle_path(self, dag: ExecutionDAG) -> list[str] | None:
        """Find cycle and return the path."""
        visited: set[str] = set()
        rec_stack: dict[str, int] = {}  # Maps to position in path
        path: list[str] = []
        
        def dfs(node_id: str) -> list[str] | None:
            visited.add(node_id)
            path.append(node_id)
            rec_stack[node_id] = len(path) - 1
            
            node = dag.nodes.get(node_id)
            if node:
                for dependent_id in node.dependents:
                    if dependent_id not in visited:
                        result = dfs(dependent_id)
                        if result:
                            return result
                    elif dependent_id in rec_stack:
                        # Cycle found - extract cycle path
                        cycle_start = rec_stack[dependent_id]
                        return path[cycle_start:] + [dependent_id]
            
            path.pop()
            del rec_stack[node_id]
            return None
        
        for task_id in dag.nodes:
            if task_id not in visited:
                cycle = dfs(task_id)
                if cycle:
                    return cycle
        
        return None
    
    def _find_reachable_tasks(self, dag: ExecutionDAG) -> set[str]:
        """Find all tasks reachable from root tasks."""
        reachable: set[str] = set()
        queue = deque(dag.root_tasks)
        
        while queue:
            task_id = queue.popleft()
            if task_id in reachable:
                continue
            reachable.add(task_id)
            
            task = dag.nodes.get(task_id)
            if task:
                queue.extend(task.dependents)
        
        return reachable
    
    def _validate_parallel_constraints(
        self, dag: ExecutionDAG
    ) -> ValidationResult:
        """Validate parallel execution constraints."""
        result = ValidationResult(is_valid=True)
        
        # Get execution levels (tasks that could run in parallel)
        levels = dag.get_execution_order()
        
        for constraint in self._parallel_constraints:
            if constraint.constraint_type == "mutex":
                # Mutex tasks must not be in the same level
                for level in levels:
                    mutex_in_level = [
                        tid for tid in constraint.task_ids if tid in level
                    ]
                    if len(mutex_in_level) > 1:
                        result.parallel_constraint_violations.append({
                            "type": "mutex",
                            "tasks": mutex_in_level,
                            "level": levels.index(level),
                            "reason": constraint.reason,
                        })
                        result.errors.append(
                            f"Mutex tasks {mutex_in_level} scheduled in same level"
                        )
            
            elif constraint.constraint_type == "same_group":
                # Same group tasks must be in the same level
                task_levels: dict[str, int] = {}
                for level_idx, level in enumerate(levels):
                    for tid in level:
                        task_levels[tid] = level_idx
                
                group_levels = {
                    task_levels.get(tid, -1) 
                    for tid in constraint.task_ids 
                    if tid in dag.nodes
                }
                group_levels.discard(-1)
                
                if len(group_levels) > 1:
                    result.parallel_constraint_violations.append({
                        "type": "same_group",
                        "tasks": constraint.task_ids,
                        "levels": list(group_levels),
                        "reason": constraint.reason,
                    })
                    result.warnings.append(
                        f"Same-group tasks {constraint.task_ids} in different levels"
                    )
            
            elif constraint.constraint_type == "ordered":
                # Tasks must execute in specified order
                task_levels: dict[str, int] = {}
                for level_idx, level in enumerate(levels):
                    for tid in level:
                        task_levels[tid] = level_idx
                
                for i in range(len(constraint.task_ids) - 1):
                    t1, t2 = constraint.task_ids[i], constraint.task_ids[i + 1]
                    if t1 in task_levels and t2 in task_levels:
                        if task_levels[t1] >= task_levels[t2]:
                            result.parallel_constraint_violations.append({
                                "type": "ordered",
                                "tasks": [t1, t2],
                                "expected": f"{t1} before {t2}",
                                "actual_levels": [task_levels[t1], task_levels[t2]],
                            })
                            result.errors.append(
                                f"Order constraint violated: {t1} must execute before {t2}"
                            )
        
        return result
    
    def _validate_resources(
        self,
        dag: ExecutionDAG,
        task_resources: dict[str, list[ResourceConstraint]],
    ) -> ValidationResult:
        """Validate resource constraints."""
        result = ValidationResult(is_valid=True)
        
        levels = dag.get_execution_order()
        
        for level_idx, level in enumerate(levels):
            # Aggregate resource requirements for this level
            level_resources: dict[str, float] = {}
            exclusive_resources: dict[str, list[str]] = {}
            
            for task_id in level:
                constraints = task_resources.get(task_id, [])
                for rc in constraints:
                    if rc.exclusive:
                        if rc.resource_id not in exclusive_resources:
                            exclusive_resources[rc.resource_id] = []
                        exclusive_resources[rc.resource_id].append(task_id)
                    
                    level_resources[rc.resource_id] = (
                        level_resources.get(rc.resource_id, 0) + rc.amount
                    )
            
            # Check exclusive resource conflicts
            for res_id, tasks in exclusive_resources.items():
                if len(tasks) > 1:
                    result.resource_conflicts.append({
                        "level": level_idx,
                        "resource": res_id,
                        "conflict_type": "exclusive",
                        "tasks": tasks,
                    })
                    result.errors.append(
                        f"Level {level_idx}: Exclusive resource '{res_id}' required by multiple tasks: {tasks}"
                    )
            
            # Check resource pool limits
            for res_id, required in level_resources.items():
                available = self._resource_pool.get(res_id, float("inf"))
                if required > available:
                    result.resource_conflicts.append({
                        "level": level_idx,
                        "resource": res_id,
                        "conflict_type": "exceeded",
                        "required": required,
                        "available": available,
                    })
                    result.warnings.append(
                        f"Level {level_idx}: Resource '{res_id}' may exceed limit "
                        f"({required} required, {available} available)"
                    )
                    result.suggestions.append(
                        f"Consider splitting level {level_idx} or adding dependencies "
                        f"to reduce concurrent resource usage"
                    )
        
        return result
    
    def _validate_semantics(self, dag: ExecutionDAG) -> ValidationResult:
        """Validate semantic rules."""
        result = ValidationResult(is_valid=True)
        
        all_rules = self.BUILTIN_RULES + self._custom_rules
        
        for rule in all_rules:
            for task_id, task in dag.nodes.items():
                try:
                    is_valid, error_msg = rule.validator(dag, task)
                    if not is_valid:
                        result.semantic_issues.append({
                            "rule_id": rule.rule_id,
                            "task_id": task_id,
                            "description": rule.description,
                            "error": error_msg,
                        })
                        result.errors.append(
                            f"Semantic rule '{rule.rule_id}' failed for task '{task_id}': {error_msg}"
                        )
                except Exception as e:
                    result.warnings.append(
                        f"Rule '{rule.rule_id}' evaluation failed for '{task_id}': {e}"
                    )
        
        return result
    
    def _analyze_dependencies(self, dag: ExecutionDAG) -> ValidationResult:
        """Analyze dependency patterns for potential issues."""
        result = ValidationResult(is_valid=True)
        
        # Check for deep dependency chains
        max_depth = 0
        for task_id in dag.nodes:
            depth = self._get_dependency_depth(dag, task_id)
            max_depth = max(max_depth, depth)
        
        if max_depth > 10:
            result.warnings.append(
                f"Deep dependency chain detected (depth: {max_depth}). "
                "Consider restructuring for better parallelism."
            )
        
        # Check for wide fan-out
        for task_id, task in dag.nodes.items():
            if len(task.dependents) > 20:
                result.warnings.append(
                    f"Task '{task_id}' has {len(task.dependents)} dependents. "
                    "High fan-out may cause bottlenecks."
                )
        
        # Check for wide fan-in
        for task_id, task in dag.nodes.items():
            if len(task.dependencies) > 10:
                result.warnings.append(
                    f"Task '{task_id}' has {len(task.dependencies)} dependencies. "
                    "High fan-in may delay execution."
                )
                result.suggestions.append(
                    f"Consider adding intermediate aggregation task before '{task_id}'"
                )
        
        # Check for redundant dependencies
        redundant = self._find_redundant_dependencies(dag)
        for task_id, redundant_deps in redundant.items():
            result.warnings.append(
                f"Task '{task_id}' has redundant dependencies: {redundant_deps}"
            )
            result.suggestions.append(
                f"Dependencies {redundant_deps} of '{task_id}' are already implied by other dependencies"
            )
        
        return result
    
    def _get_dependency_depth(
        self, dag: ExecutionDAG, task_id: str, memo: dict[str, int] | None = None
    ) -> int:
        """Calculate dependency depth for a task."""
        if memo is None:
            memo = {}
        
        if task_id in memo:
            return memo[task_id]
        
        task = dag.nodes.get(task_id)
        if not task or not task.dependencies:
            memo[task_id] = 0
            return 0
        
        max_dep_depth = max(
            self._get_dependency_depth(dag, dep_id, memo)
            for dep_id in task.dependencies
        )
        memo[task_id] = max_dep_depth + 1
        return memo[task_id]
    
    def _find_redundant_dependencies(
        self, dag: ExecutionDAG
    ) -> dict[str, list[str]]:
        """Find redundant (transitively implied) dependencies."""
        redundant: dict[str, list[str]] = {}
        
        for task_id, task in dag.nodes.items():
            if len(task.dependencies) < 2:
                continue
            
            # Get all transitive dependencies
            transitive: set[str] = set()
            for dep_id in task.dependencies:
                self._collect_transitive_deps(dag, dep_id, transitive)
            
            # Check if any direct dep is in transitive deps of another direct dep
            task_redundant = []
            for dep_id in task.dependencies:
                if dep_id in transitive:
                    task_redundant.append(dep_id)
            
            if task_redundant:
                redundant[task_id] = task_redundant
        
        return redundant
    
    def _collect_transitive_deps(
        self, dag: ExecutionDAG, task_id: str, collected: set[str]
    ) -> None:
        """Collect all transitive dependencies of a task."""
        task = dag.nodes.get(task_id)
        if not task:
            return
        
        for dep_id in task.dependencies:
            if dep_id not in collected:
                collected.add(dep_id)
                self._collect_transitive_deps(dag, dep_id, collected)


# Semantic rule factory functions
def create_task_type_rule(
    allowed_types: list[str],
) -> SemanticRule:
    """Create a rule that validates task types."""
    def validator(dag: ExecutionDAG, task: TaskNode) -> tuple[bool, str | None]:
        task_type = task.metadata.get("task_type", "unknown")
        if task_type not in allowed_types:
            return False, f"Invalid task type: {task_type}"
        return True, None
    
    return SemanticRule(
        rule_id="task_type_check",
        description=f"Task type must be one of: {allowed_types}",
        validator=validator,
    )


def create_duration_limit_rule(
    max_duration_ms: float,
) -> SemanticRule:
    """Create a rule that validates task duration limits."""
    def validator(dag: ExecutionDAG, task: TaskNode) -> tuple[bool, str | None]:
        if task.estimated_duration_ms > max_duration_ms:
            return False, (
                f"Estimated duration {task.estimated_duration_ms}ms "
                f"exceeds limit {max_duration_ms}ms"
            )
        return True, None
    
    return SemanticRule(
        rule_id="duration_limit",
        description=f"Task duration must not exceed {max_duration_ms}ms",
        validator=validator,
    )


def create_dependency_depth_rule(
    max_depth: int,
) -> SemanticRule:
    """Create a rule that validates dependency depth."""
    def validator(dag: ExecutionDAG, task: TaskNode) -> tuple[bool, str | None]:
        depth = _calculate_depth(dag, task.task_id, set())
        if depth > max_depth:
            return False, f"Dependency depth {depth} exceeds limit {max_depth}"
        return True, None
    
    def _calculate_depth(
        dag: ExecutionDAG, task_id: str, visited: set[str]
    ) -> int:
        if task_id in visited:
            return 0
        visited.add(task_id)
        
        task = dag.nodes.get(task_id)
        if not task or not task.dependencies:
            return 0
        
        return 1 + max(
            _calculate_depth(dag, dep, visited) for dep in task.dependencies
        )
    
    return SemanticRule(
        rule_id="dependency_depth",
        description=f"Dependency depth must not exceed {max_depth}",
        validator=validator,
    )


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
    "LLMAssistedPlanner",
    "PlanningContext",
    "PlanningStrategy",
    # Complex DAG Validation
    "ResourceConstraint",
    "ParallelConstraint",
    "SemanticRule",
    "ValidationResult",
    "ComplexDAGValidator",
    "create_task_type_rule",
    "create_duration_limit_rule",
    "create_dependency_depth_rule",
]


# ==============================================================================
# ENHANCED LLM-ASSISTED PLANNING
# ==============================================================================


from dataclasses import dataclass, field as dataclass_field
from enum import Enum, auto as enum_auto
from typing import Protocol


class PlanningStrategy(Enum):
    """Strategy for planning approach."""
    
    DIRECT = enum_auto()      # Direct translation of objective to plan
    ITERATIVE = enum_auto()   # Refine plan through multiple LLM calls
    DECOMPOSE = enum_auto()   # Break down complex objectives into sub-goals
    TEMPLATE = enum_auto()    # Use templates with LLM customization
    ADAPTIVE = enum_auto()    # Adapt strategy based on objective complexity


@dataclass
class PlanningContext:
    """Context for LLM-assisted planning with history and constraints."""
    
    objective: str
    constraints: list[str] = dataclass_field(default_factory=list)
    preferences: dict[str, Any] = dataclass_field(default_factory=dict)
    available_backends: list[str] = dataclass_field(default_factory=list)
    resource_limits: dict[str, float] = dataclass_field(default_factory=dict)
    previous_plans: list[dict[str, Any]] = dataclass_field(default_factory=list)
    feedback_history: list[dict[str, str]] = dataclass_field(default_factory=list)
    domain_knowledge: dict[str, Any] = dataclass_field(default_factory=dict)
    
    def add_constraint(self, constraint: str) -> None:
        """Add a planning constraint."""
        self.constraints.append(constraint)
    
    def add_feedback(self, plan_id: str, feedback: str, rating: int) -> None:
        """Add feedback about a previous plan."""
        self.feedback_history.append({
            "plan_id": plan_id,
            "feedback": feedback,
            "rating": rating,
            "timestamp": time.time() if 'time' in dir() else 0,
        })


class LLMAssistedPlanner:
    """Enhanced LLM-assisted planner with deep integration capabilities.
    
    Features:
    - Multi-turn planning with refinement
    - Constraint-aware planning
    - Learning from feedback
    - Objective decomposition for complex tasks
    - Template-based planning with LLM customization
    - Plan explanation generation
    - Alternative plan generation
    - Risk assessment
    """
    
    # Specialized prompts for different planning aspects
    PLANNING_PROMPTS = {
        "decompose_objective": '''Analyze this quantum computing objective and break it into sub-goals:

Objective: {objective}

Available Backends: {backends}
Constraints: {constraints}

Break this down into:
1. Required sub-tasks (in order of execution)
2. Dependencies between sub-tasks
3. Estimated complexity of each sub-task (low/medium/high)
4. Potential parallelization opportunities

Respond in structured JSON format with fields: subtasks, dependencies, complexity, parallel_groups''',

        "generate_plan": '''Create a detailed execution plan for this quantum computing task:

Objective: {objective}
Sub-goals: {subgoals}
Available Backends: {backends}
Constraints: {constraints}
Resource Limits: {resource_limits}

Previous successful patterns:
{previous_patterns}

Generate a plan with:
1. Ordered steps with action, description, parameters
2. Backend assignments for each step
3. Estimated duration for each step
4. Risk assessment for each step

Respond in JSON format matching this schema:
{{
    "steps": [...],
    "execution_mode": "single|comparison|distributed",
    "estimated_duration_ms": number,
    "risk_level": "low|medium|high",
    "notes": [...]
}}''',

        "refine_plan": '''Review and refine this execution plan based on feedback:

Current Plan:
{current_plan}

Feedback: {feedback}
Constraints: {constraints}

Improve the plan by:
1. Addressing the feedback
2. Optimizing for the constraints
3. Reducing unnecessary steps
4. Improving parallelism where possible

Provide the refined plan in the same JSON format.''',

        "explain_plan": '''Explain this quantum computing execution plan in clear terms:

Plan:
{plan}

Objective: {objective}

Provide:
1. A plain-language summary of what will happen
2. Why each step is necessary
3. What results to expect
4. Potential issues to watch for

Make the explanation accessible to someone new to quantum computing.''',

        "assess_risk": '''Assess the risks in this execution plan:

Plan:
{plan}

Available Resources:
- Memory: {memory_limit}
- Time: {time_limit}
- Backends: {backends}

Evaluate:
1. Likelihood of failure for each step
2. Resource consumption risks
3. Backend-specific concerns
4. Mitigation strategies

Provide risk assessment in JSON format with risk_level, concerns, and mitigations.''',

        "generate_alternatives": '''Generate alternative approaches for this objective:

Objective: {objective}
Current Plan: {current_plan}
Constraints: {constraints}

Provide 2-3 alternative plans with:
1. Different backend strategies
2. Different circuit approaches
3. Trade-offs (speed vs accuracy, resources vs quality)

Each alternative should be a complete plan in JSON format.''',
    }
    
    def __init__(
        self,
        llm_callback: Callable[[str], str] | None = None,
        default_strategy: PlanningStrategy = PlanningStrategy.ADAPTIVE,
    ):
        """Initialize the LLM-assisted planner.
        
        Args:
            llm_callback: Function that takes prompt and returns LLM response
            default_strategy: Default planning strategy to use
        """
        self._llm_callback = llm_callback
        self._default_strategy = default_strategy
        self._plan_cache: dict[str, dict[str, Any]] = {}
        self._feedback_db: list[dict[str, Any]] = []
        self.logger = get_logger("llm_planner")
    
    @property
    def available(self) -> bool:
        """Check if LLM is available for planning."""
        return self._llm_callback is not None
    
    def plan_with_context(
        self,
        context: PlanningContext,
        strategy: PlanningStrategy | None = None,
    ) -> dict[str, Any]:
        """Generate a plan using the specified strategy and context.
        
        Args:
            context: Planning context with objective and constraints
            strategy: Planning strategy to use (or default)
            
        Returns:
            Execution plan dictionary
        """
        strategy = strategy or self._default_strategy
        
        # Choose strategy based on objective complexity
        if strategy == PlanningStrategy.ADAPTIVE:
            strategy = self._select_strategy(context)
        
        self.logger.info(
            "planning.start",
            objective=context.objective[:50],
            strategy=strategy.name,
        )
        
        if strategy == PlanningStrategy.DIRECT:
            return self._plan_direct(context)
        elif strategy == PlanningStrategy.ITERATIVE:
            return self._plan_iterative(context)
        elif strategy == PlanningStrategy.DECOMPOSE:
            return self._plan_decompose(context)
        elif strategy == PlanningStrategy.TEMPLATE:
            return self._plan_template(context)
        else:
            return self._plan_direct(context)
    
    def _select_strategy(self, context: PlanningContext) -> PlanningStrategy:
        """Select best strategy based on objective analysis."""
        objective = context.objective.lower()
        
        # Complex objectives benefit from decomposition
        complex_indicators = [
            "multiple", "compare", "benchmark", "analyze",
            "optimize", "all backends", "comprehensive",
        ]
        if any(ind in objective for ind in complex_indicators):
            return PlanningStrategy.DECOMPOSE
        
        # Simple well-known patterns use templates
        template_patterns = [
            "bell state", "ghz", "teleportation",
            "superposition", "entangle",
        ]
        if any(pat in objective for pat in template_patterns):
            return PlanningStrategy.TEMPLATE
        
        # Has feedback history - use iterative refinement
        if context.feedback_history:
            return PlanningStrategy.ITERATIVE
        
        # Default to direct
        return PlanningStrategy.DIRECT
    
    def _plan_direct(self, context: PlanningContext) -> dict[str, Any]:
        """Generate plan directly from objective."""
        if not self.available:
            return self._fallback_plan(context)
        
        prompt = self.PLANNING_PROMPTS["generate_plan"].format(
            objective=context.objective,
            subgoals="(direct planning - no decomposition)",
            backends=", ".join(context.available_backends) or "auto-select",
            constraints="; ".join(context.constraints) or "none",
            resource_limits=json.dumps(context.resource_limits) if context.resource_limits else "unlimited",
            previous_patterns=self._get_relevant_patterns(context),
        )
        
        try:
            response = self._llm_callback(prompt)
            plan = self._parse_plan_response(response)
            plan["_strategy"] = "direct"
            return plan
        except Exception as e:
            self.logger.warning("planning.llm_failed", error=str(e))
            return self._fallback_plan(context)
    
    def _plan_iterative(
        self,
        context: PlanningContext,
        max_iterations: int = 3,
    ) -> dict[str, Any]:
        """Generate plan through iterative refinement."""
        # Start with direct plan
        current_plan = self._plan_direct(context)
        
        if not self.available:
            return current_plan
        
        # Apply feedback from history
        for i, feedback_entry in enumerate(context.feedback_history[-max_iterations:]):
            prompt = self.PLANNING_PROMPTS["refine_plan"].format(
                current_plan=json.dumps(current_plan, indent=2),
                feedback=feedback_entry.get("feedback", ""),
                constraints="; ".join(context.constraints),
            )
            
            try:
                response = self._llm_callback(prompt)
                refined_plan = self._parse_plan_response(response)
                if refined_plan.get("steps"):
                    current_plan = refined_plan
                    current_plan["_iteration"] = i + 1
            except Exception:
                continue
        
        current_plan["_strategy"] = "iterative"
        return current_plan
    
    def _plan_decompose(self, context: PlanningContext) -> dict[str, Any]:
        """Generate plan by decomposing objective into sub-goals."""
        if not self.available:
            return self._fallback_plan(context)
        
        # Step 1: Decompose objective
        decompose_prompt = self.PLANNING_PROMPTS["decompose_objective"].format(
            objective=context.objective,
            backends=", ".join(context.available_backends) or "auto-select",
            constraints="; ".join(context.constraints) or "none",
        )
        
        try:
            decompose_response = self._llm_callback(decompose_prompt)
            decomposition = self._parse_json_response(decompose_response)
        except Exception:
            decomposition = {"subtasks": [context.objective]}
        
        # Step 2: Generate plan for decomposed goals
        subgoals = decomposition.get("subtasks", [context.objective])
        
        plan_prompt = self.PLANNING_PROMPTS["generate_plan"].format(
            objective=context.objective,
            subgoals=json.dumps(subgoals),
            backends=", ".join(context.available_backends) or "auto-select",
            constraints="; ".join(context.constraints) or "none",
            resource_limits=json.dumps(context.resource_limits) if context.resource_limits else "unlimited",
            previous_patterns=self._get_relevant_patterns(context),
        )
        
        try:
            response = self._llm_callback(plan_prompt)
            plan = self._parse_plan_response(response)
            plan["_decomposition"] = decomposition
            plan["_strategy"] = "decompose"
            return plan
        except Exception:
            return self._fallback_plan(context)
    
    def _plan_template(self, context: PlanningContext) -> dict[str, Any]:
        """Generate plan using templates with LLM customization."""
        objective = context.objective.lower()
        
        # Select template based on objective
        if "bell" in objective or "entangle" in objective:
            base_plan = PlanTemplate.single_circuit_execution(
                circuit_type="bell",
                backend=context.available_backends[0] if context.available_backends else "auto",
            )
        elif "ghz" in objective:
            qubits = self._extract_qubit_count(context.objective) or 3
            base_plan = PlanTemplate.single_circuit_execution(
                circuit_type="ghz",
                qubits=qubits,
            )
        elif "compare" in objective:
            base_plan = PlanTemplate.backend_comparison(
                backends=context.available_backends or ["cirq", "qiskit"],
            )
        else:
            base_plan = PlanTemplate.single_circuit_execution()
        
        # Customize with LLM if available
        if self.available and context.constraints:
            try:
                customize_prompt = f"""Customize this plan based on user constraints:

Plan: {json.dumps(base_plan, indent=2)}

Constraints: {'; '.join(context.constraints)}

Return the modified plan in the same JSON format, adjusting parameters as needed."""
                
                response = self._llm_callback(customize_prompt)
                customized = self._parse_plan_response(response)
                if customized.get("steps"):
                    base_plan = customized
            except Exception:
                pass
        
        base_plan["_strategy"] = "template"
        return base_plan
    
    def explain_plan(self, plan: dict[str, Any], objective: str) -> str:
        """Generate a human-readable explanation of the plan.
        
        Args:
            plan: The execution plan
            objective: Original objective
            
        Returns:
            Plain-language explanation
        """
        if not self.available:
            return self._fallback_explanation(plan, objective)
        
        prompt = self.PLANNING_PROMPTS["explain_plan"].format(
            plan=json.dumps(plan, indent=2),
            objective=objective,
        )
        
        try:
            return self._llm_callback(prompt)
        except Exception:
            return self._fallback_explanation(plan, objective)
    
    def assess_risk(
        self,
        plan: dict[str, Any],
        resource_limits: dict[str, float] | None = None,
        backends: list[str] | None = None,
    ) -> dict[str, Any]:
        """Assess risks in the execution plan.
        
        Args:
            plan: The execution plan
            resource_limits: Resource constraints
            backends: Available backends
            
        Returns:
            Risk assessment dictionary
        """
        if not self.available:
            return self._fallback_risk_assessment(plan)
        
        prompt = self.PLANNING_PROMPTS["assess_risk"].format(
            plan=json.dumps(plan, indent=2),
            memory_limit=resource_limits.get("memory_mb", "unlimited") if resource_limits else "unlimited",
            time_limit=resource_limits.get("timeout_s", "unlimited") if resource_limits else "unlimited",
            backends=", ".join(backends) if backends else "all available",
        )
        
        try:
            response = self._llm_callback(prompt)
            return self._parse_json_response(response)
        except Exception:
            return self._fallback_risk_assessment(plan)
    
    def generate_alternatives(
        self,
        plan: dict[str, Any],
        objective: str,
        constraints: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate alternative plans.
        
        Args:
            plan: Current plan
            objective: Original objective
            constraints: Planning constraints
            
        Returns:
            List of alternative plans
        """
        if not self.available:
            return []
        
        prompt = self.PLANNING_PROMPTS["generate_alternatives"].format(
            objective=objective,
            current_plan=json.dumps(plan, indent=2),
            constraints="; ".join(constraints) if constraints else "none",
        )
        
        try:
            response = self._llm_callback(prompt)
            alternatives = self._parse_json_response(response)
            
            if isinstance(alternatives, list):
                return alternatives
            elif isinstance(alternatives, dict) and "alternatives" in alternatives:
                return alternatives["alternatives"]
            else:
                return []
        except Exception:
            return []
    
    def record_feedback(
        self,
        plan: dict[str, Any],
        feedback: str,
        rating: int,
        outcome: str,
    ) -> None:
        """Record feedback about a plan for future learning.
        
        Args:
            plan: The executed plan
            feedback: User feedback
            rating: Rating 1-5
            outcome: Execution outcome (success/failure/partial)
        """
        self._feedback_db.append({
            "plan_hash": hash(json.dumps(plan.get("steps", []), sort_keys=True)),
            "objective": plan.get("objective", ""),
            "circuit_type": plan.get("circuit_type", ""),
            "feedback": feedback,
            "rating": rating,
            "outcome": outcome,
            "timestamp": time.time() if 'time' in dir() else 0,
        })
    
    def _get_relevant_patterns(self, context: PlanningContext) -> str:
        """Get relevant patterns from feedback history."""
        if not self._feedback_db:
            return "No previous patterns available"
        
        # Find high-rated plans with similar objectives
        relevant = [
            f for f in self._feedback_db
            if f["rating"] >= 4 and f["outcome"] == "success"
        ][:3]
        
        if not relevant:
            return "No successful patterns found"
        
        patterns = [
            f"- {f['circuit_type']}: {f['feedback']}" for f in relevant
        ]
        return "\n".join(patterns)
    
    def _parse_plan_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response into plan dictionary."""
        plan = self._parse_json_response(response)
        
        # Ensure required fields
        if "steps" not in plan:
            plan["steps"] = []
        
        return plan
    
    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response."""
        import re
        
        # Try to find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array
        array_match = re.search(r'\[[\s\S]*\]', response)
        if array_match:
            try:
                return {"items": json.loads(array_match.group())}
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def _extract_qubit_count(self, text: str) -> int | None:
        """Extract qubit count from text."""
        import re
        match = re.search(r'(\d+)[-\s]*qubit', text.lower())
        if match:
            return int(match.group(1))
        return None
    
    def _fallback_plan(self, context: PlanningContext) -> dict[str, Any]:
        """Generate fallback plan when LLM is unavailable."""
        objective = context.objective.lower()
        
        circuit_type = "bell"
        qubits = 2
        
        if "ghz" in objective:
            circuit_type = "ghz"
            qubits = self._extract_qubit_count(context.objective) or 3
        elif "teleport" in objective:
            circuit_type = "teleportation"
            qubits = 3
        
        return {
            "objective": context.objective,
            "circuit_type": circuit_type,
            "qubits": qubits,
            "shots": 1024,
            "execution_mode": "single",
            "backends": context.available_backends[:1] if context.available_backends else ["auto"],
            "steps": [
                {"step": 1, "action": "create_circuit", "parameters": {"circuit_type": circuit_type, "qubits": qubits}},
                {"step": 2, "action": "execute", "parameters": {"shots": 1024}},
                {"step": 3, "action": "collect_results", "parameters": {}},
            ],
            "_strategy": "fallback",
        }
    
    def _fallback_explanation(self, plan: dict[str, Any], objective: str) -> str:
        """Generate fallback explanation without LLM."""
        steps = plan.get("steps", [])
        circuit_type = plan.get("circuit_type", "quantum")
        
        lines = [
            f"## Plan Explanation for: {objective}",
            "",
            f"This plan will execute a {circuit_type} circuit through {len(steps)} steps:",
            "",
        ]
        
        for i, step in enumerate(steps, 1):
            action = step.get("action", "unknown")
            desc = step.get("description", action)
            lines.append(f"{i}. **{action.replace('_', ' ').title()}**: {desc}")
        
        return "\n".join(lines)
    
    def _fallback_risk_assessment(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Generate fallback risk assessment without LLM."""
        steps = plan.get("steps", [])
        
        concerns = []
        risk_level = "low"
        
        if len(steps) > 5:
            concerns.append("Plan has many steps - higher chance of failure")
            risk_level = "medium"
        
        for step in steps:
            params = step.get("parameters", {})
            if params.get("shots", 0) > 10000:
                concerns.append("High shot count may be slow")
            if params.get("qubits", 0) > 10:
                concerns.append("Many qubits may require significant memory")
                risk_level = "high"
        
        return {
            "risk_level": risk_level,
            "concerns": concerns,
            "mitigations": [
                "Monitor memory usage during execution",
                "Set appropriate timeouts",
                "Use checkpoints for long-running plans",
            ],
        }


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


# ==============================================================================
# DEEP LLM INTEGRATION (2% Gap Coverage)
# ==============================================================================


class LLMPlanningMode(Enum):
    """LLM planning modes."""
    
    DIRECT = "direct"           # Single LLM call for plan
    ITERATIVE = "iterative"     # Multiple refinement rounds
    HYBRID = "hybrid"           # Template + LLM enhancement
    EXPLAINED = "explained"     # Plan with explanations


@dataclass
class PlanRefinementContext:
    """Context for iterative plan refinement."""
    
    original_query: str
    current_plan: dict[str, Any]
    iteration: int = 0
    max_iterations: int = 3
    feedback_history: list[dict[str, Any]] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)
    
    def add_feedback(self, feedback: str, quality_score: float) -> None:
        """Add refinement feedback."""
        self.feedback_history.append({
            "iteration": self.iteration,
            "feedback": feedback,
            "score": quality_score,
        })
        self.quality_scores.append(quality_score)
        self.iteration += 1
    
    def should_continue(self) -> bool:
        """Check if refinement should continue."""
        if self.iteration >= self.max_iterations:
            return False
        if len(self.quality_scores) >= 2:
            # Stop if quality isn't improving
            if self.quality_scores[-1] <= self.quality_scores[-2] + 0.05:
                return False
        return True


@dataclass
class PlanExplanation:
    """Detailed explanation of a plan step."""
    
    step_index: int
    action: str
    purpose: str
    dependencies: list[int]
    estimated_time: float
    potential_issues: list[str]
    alternatives: list[str]


@dataclass
class EnhancedPlanResult:
    """Result from enhanced LLM planning."""
    
    plan: dict[str, Any]
    explanations: list[PlanExplanation]
    confidence: float
    suggestions: list[str]
    risk_assessment: dict[str, Any]
    refinement_history: list[dict[str, Any]]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan": self.plan,
            "explanations": [
                {
                    "step": e.step_index,
                    "action": e.action,
                    "purpose": e.purpose,
                    "dependencies": e.dependencies,
                    "estimated_time_ms": e.estimated_time,
                    "potential_issues": e.potential_issues,
                    "alternatives": e.alternatives,
                }
                for e in self.explanations
            ],
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "risk_assessment": self.risk_assessment,
            "refinement_iterations": len(self.refinement_history),
        }


class DeepLLMPlanner:
    """Advanced LLM-integrated planner with iterative refinement.
    
    Features:
    - Iterative plan refinement with feedback loops
    - Plan explanation generation
    - Risk assessment
    - Alternative suggestions
    - Quality scoring
    - Streaming plan generation
    """
    
    def __init__(
        self,
        llm_callback: Callable[[str], str] | None = None,
        streaming_callback: Callable[[str], None] | None = None,
        mode: LLMPlanningMode = LLMPlanningMode.HYBRID,
    ) -> None:
        """Initialize deep LLM planner.
        
        Args:
            llm_callback: Callback to invoke LLM
            streaming_callback: Callback for streaming output
            mode: Planning mode
        """
        self._llm_callback = llm_callback
        self._streaming_callback = streaming_callback
        self._mode = mode
        self._validator = PlanValidator()
        self._optimizer = PlanOptimizer()
        self._base_planner = LLMAssistedPlanner(llm_callback)
    
    async def plan_with_explanation(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> EnhancedPlanResult:
        """Generate a plan with detailed explanations.
        
        Args:
            query: User query/intent
            context: Additional context
            
        Returns:
            EnhancedPlanResult with plan and explanations
        """
        # Get base plan
        base_plan = self._base_planner.create_plan(query, context)
        
        # Generate explanations for each step
        explanations = await self._generate_explanations(base_plan)
        
        # Assess risks
        risk_assessment = await self._assess_risks(base_plan, context)
        
        # Generate suggestions
        suggestions = await self._generate_suggestions(base_plan, query)
        
        # Calculate confidence
        confidence = self._calculate_confidence(base_plan, explanations)
        
        return EnhancedPlanResult(
            plan=base_plan,
            explanations=explanations,
            confidence=confidence,
            suggestions=suggestions,
            risk_assessment=risk_assessment,
            refinement_history=[],
        )
    
    async def iterative_refinement(
        self,
        query: str,
        initial_plan: dict[str, Any] | None = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.9,
    ) -> EnhancedPlanResult:
        """Refine a plan iteratively using LLM feedback.
        
        Args:
            query: Original query
            initial_plan: Starting plan (generated if None)
            max_iterations: Maximum refinement iterations
            quality_threshold: Stop when quality reaches this
            
        Returns:
            EnhancedPlanResult with refined plan
        """
        # Get or create initial plan
        current_plan = initial_plan or self._base_planner.create_plan(query)
        
        refinement_ctx = PlanRefinementContext(
            original_query=query,
            current_plan=current_plan,
            max_iterations=max_iterations,
        )
        
        refinement_history: list[dict[str, Any]] = []
        
        while refinement_ctx.should_continue():
            # Evaluate current plan quality
            quality_score = await self._evaluate_plan_quality(
                current_plan, query
            )
            
            if quality_score >= quality_threshold:
                break
            
            # Get LLM feedback for improvement
            feedback = await self._get_refinement_feedback(
                current_plan, query, refinement_ctx
            )
            
            # Apply refinements
            refined_plan = await self._apply_refinements(
                current_plan, feedback
            )
            
            refinement_history.append({
                "iteration": refinement_ctx.iteration,
                "quality_before": quality_score,
                "feedback": feedback,
                "changes_made": self._diff_plans(current_plan, refined_plan),
            })
            
            refinement_ctx.add_feedback(feedback, quality_score)
            current_plan = refined_plan
        
        # Final optimization
        optimized_plan = self._optimizer.optimize(current_plan)
        
        # Generate final explanations
        explanations = await self._generate_explanations(optimized_plan)
        
        return EnhancedPlanResult(
            plan=optimized_plan,
            explanations=explanations,
            confidence=quality_score if 'quality_score' in dir() else 0.8,
            suggestions=[],
            risk_assessment=await self._assess_risks(optimized_plan),
            refinement_history=refinement_history,
        )
    
    async def stream_plan_generation(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> EnhancedPlanResult:
        """Generate plan with streaming updates.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            EnhancedPlanResult
        """
        steps: list[dict[str, Any]] = []
        
        if self._streaming_callback:
            self._streaming_callback("🎯 Analyzing query...\n")
        
        # Analyze query intent
        intent = self._analyze_intent(query)
        
        if self._streaming_callback:
            self._streaming_callback(f"📋 Intent: {intent.get('action', 'unknown')}\n")
            self._streaming_callback("🔧 Generating steps...\n")
        
        # Generate steps one by one with streaming
        step_generators = {
            "compare": self._generate_compare_steps,
            "benchmark": self._generate_benchmark_steps,
            "analyze": self._generate_analyze_steps,
            "execute": self._generate_execute_steps,
        }
        
        action = intent.get("action", "execute")
        generator = step_generators.get(action, self._generate_execute_steps)
        
        for step in generator(intent):
            steps.append(step)
            if self._streaming_callback:
                self._streaming_callback(f"  Step {len(steps)}: {step.get('action', 'unknown')}\n")
        
        plan = {
            "steps": steps,
            "metadata": {
                "query": query,
                "intent": intent,
                "streaming": True,
            },
        }
        
        if self._streaming_callback:
            self._streaming_callback("✅ Plan generation complete!\n")
        
        return await self.plan_with_explanation(query, context)
    
    async def _generate_explanations(
        self, plan: dict[str, Any]
    ) -> list[PlanExplanation]:
        """Generate explanations for each plan step."""
        explanations = []
        steps = plan.get("steps", [])
        
        for i, step in enumerate(steps):
            action = step.get("action", "unknown")
            params = step.get("parameters", {})
            
            # Determine purpose based on action
            purpose_map = {
                "create_circuit": "Creates a quantum circuit for simulation",
                "execute": "Runs the quantum circuit on the specified backend",
                "collect_results": "Gathers measurement outcomes and statistics",
                "compare": "Compares results across multiple backends",
                "analyze": "Performs detailed analysis of quantum states",
                "benchmark": "Measures performance metrics across multiple runs",
                "export": "Saves results to the specified format",
            }
            
            purpose = purpose_map.get(action, f"Performs {action} operation")
            
            # Add parameter-specific details
            if action == "create_circuit" and "circuit_type" in params:
                purpose += f" of type '{params['circuit_type']}'"
            elif action == "execute" and "backend" in params:
                purpose += f" using {params['backend']}"
            
            # Identify dependencies
            dependencies = []
            if action in ("execute", "benchmark"):
                # Depends on circuit creation
                for j in range(i - 1, -1, -1):
                    if steps[j].get("action") == "create_circuit":
                        dependencies.append(j)
                        break
            elif action in ("collect_results", "compare", "analyze"):
                # Depends on execution
                for j in range(i - 1, -1, -1):
                    if steps[j].get("action") in ("execute", "benchmark"):
                        dependencies.append(j)
                        break
            
            # Estimate time
            time_estimates = {
                "create_circuit": 10.0,
                "execute": 500.0,
                "collect_results": 50.0,
                "compare": 100.0,
                "analyze": 200.0,
                "benchmark": 2000.0,
                "export": 100.0,
            }
            estimated_time = time_estimates.get(action, 100.0)
            
            # Identify potential issues
            issues = []
            if action == "execute" and params.get("shots", 0) > 10000:
                issues.append("High shot count may increase execution time")
            if action == "benchmark" and params.get("runs", 0) > 10:
                issues.append("Many benchmark runs will extend total duration")
            
            # Suggest alternatives
            alternatives = []
            if action == "execute" and not params.get("backend"):
                alternatives.append("Consider specifying multiple backends for comparison")
            
            explanations.append(PlanExplanation(
                step_index=i,
                action=action,
                purpose=purpose,
                dependencies=dependencies,
                estimated_time=estimated_time,
                potential_issues=issues,
                alternatives=alternatives,
            ))
        
        return explanations
    
    async def _assess_risks(
        self,
        plan: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assess risks in the execution plan."""
        risks = {
            "level": "low",
            "factors": [],
            "mitigations": [],
        }
        
        steps = plan.get("steps", [])
        
        # Check for high-risk patterns
        total_qubits = 0
        has_noise = False
        high_shots = False
        
        for step in steps:
            params = step.get("parameters", {})
            
            if "qubits" in params:
                total_qubits = max(total_qubits, params["qubits"])
            if params.get("noise_model"):
                has_noise = True
            if params.get("shots", 0) > 10000:
                high_shots = True
        
        if total_qubits > 20:
            risks["factors"].append({
                "type": "high_qubit_count",
                "severity": "high",
                "description": f"Circuit uses {total_qubits} qubits which may cause memory issues",
            })
            risks["mitigations"].append("Consider using tensor network methods")
            risks["level"] = "high"
        
        if has_noise and total_qubits > 15:
            risks["factors"].append({
                "type": "noise_with_large_circuit",
                "severity": "medium",
                "description": "Noise simulation on large circuits is computationally expensive",
            })
            risks["mitigations"].append("Use approximate noise models")
            if risks["level"] != "high":
                risks["level"] = "medium"
        
        if high_shots:
            risks["factors"].append({
                "type": "high_shot_count",
                "severity": "low",
                "description": "High shot count increases execution time",
            })
        
        return risks
    
    async def _generate_suggestions(
        self,
        plan: dict[str, Any],
        query: str,
    ) -> list[str]:
        """Generate improvement suggestions for the plan."""
        suggestions = []
        steps = plan.get("steps", [])
        
        # Check if plan could benefit from parallelization
        execute_steps = [s for s in steps if s.get("action") == "execute"]
        if len(execute_steps) > 1:
            suggestions.append(
                "Multiple execution steps detected - consider parallel execution for faster results"
            )
        
        # Check for missing optimization opportunities
        has_benchmark = any(s.get("action") == "benchmark" for s in steps)
        has_compare = any(s.get("action") == "compare" for s in steps)
        
        if has_benchmark and not has_compare:
            suggestions.append(
                "Add a comparison step to analyze benchmark results across backends"
            )
        
        # Suggest caching for repeated patterns
        circuit_types = [
            s.get("parameters", {}).get("circuit_type")
            for s in steps if s.get("action") == "create_circuit"
        ]
        if len(circuit_types) > len(set(circuit_types)):
            suggestions.append(
                "Identical circuit types detected - consider caching to avoid redundant creation"
            )
        
        return suggestions
    
    def _calculate_confidence(
        self,
        plan: dict[str, Any],
        explanations: list[PlanExplanation],
    ) -> float:
        """Calculate confidence score for the plan."""
        # Validate plan
        errors = self._validator.validate(plan)
        if errors:
            return max(0.3, 1.0 - len(errors) * 0.1)
        
        # Check coverage
        steps = plan.get("steps", [])
        if not steps:
            return 0.5
        
        # Higher confidence for complete workflows
        has_create = any(s.get("action") == "create_circuit" for s in steps)
        has_execute = any(s.get("action") in ("execute", "benchmark") for s in steps)
        has_output = any(s.get("action") in ("export", "collect_results", "analyze") for s in steps)
        
        coverage_score = sum([has_create, has_execute, has_output]) / 3
        
        # Check for potential issues
        issue_count = sum(len(e.potential_issues) for e in explanations)
        issue_penalty = min(0.3, issue_count * 0.05)
        
        return min(1.0, coverage_score * 0.7 + 0.5 - issue_penalty)
    
    async def _evaluate_plan_quality(
        self, plan: dict[str, Any], query: str
    ) -> float:
        """Evaluate the quality of a plan."""
        # Validation check
        errors = self._validator.validate(plan)
        if errors:
            return max(0.2, 0.7 - len(errors) * 0.1)
        
        # Completeness check
        steps = plan.get("steps", [])
        actions = {s.get("action") for s in steps}
        
        expected_actions = {"create_circuit", "execute"}
        coverage = len(actions & expected_actions) / len(expected_actions)
        
        # Query alignment
        query_lower = query.lower()
        alignment = 1.0
        if "compare" in query_lower and "compare" not in actions:
            alignment *= 0.8
        if "benchmark" in query_lower and "benchmark" not in actions:
            alignment *= 0.8
        
        return coverage * 0.5 + alignment * 0.5
    
    async def _get_refinement_feedback(
        self,
        plan: dict[str, Any],
        query: str,
        context: PlanRefinementContext,
    ) -> str:
        """Get LLM feedback for plan refinement."""
        if not self._llm_callback:
            # Default feedback without LLM
            steps = plan.get("steps", [])
            if not any(s.get("action") == "create_circuit" for s in steps):
                return "ADD_CIRCUIT_CREATION"
            if not any(s.get("action") in ("execute", "benchmark") for s in steps):
                return "ADD_EXECUTION"
            return "OPTIMIZE_PARAMETERS"
        
        # Use LLM for feedback
        prompt = f"""Analyze this quantum computing plan and suggest improvements:

Query: {query}

Current Plan:
{plan}

Previous Feedback: {context.feedback_history}

Provide specific, actionable feedback to improve the plan."""
        
        return self._llm_callback(prompt)
    
    async def _apply_refinements(
        self, plan: dict[str, Any], feedback: str
    ) -> dict[str, Any]:
        """Apply refinements based on feedback."""
        refined = dict(plan)
        steps = list(plan.get("steps", []))
        
        feedback_upper = feedback.upper()
        
        if "ADD_CIRCUIT_CREATION" in feedback_upper:
            steps.insert(0, {
                "action": "create_circuit",
                "parameters": {"circuit_type": "ghz", "qubits": 3},
            })
        
        if "ADD_EXECUTION" in feedback_upper:
            steps.append({
                "action": "execute",
                "parameters": {"shots": 1024},
            })
        
        if "OPTIMIZE" in feedback_upper:
            # Apply optimization
            refined = self._optimizer.optimize({"steps": steps})
            return refined
        
        refined["steps"] = steps
        return refined
    
    def _diff_plans(
        self, old: dict[str, Any], new: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate diff between two plans."""
        old_steps = old.get("steps", [])
        new_steps = new.get("steps", [])
        
        return {
            "steps_added": len(new_steps) - len(old_steps),
            "actions_changed": [
                s.get("action") for s in new_steps
            ] != [s.get("action") for s in old_steps],
        }
    
    def _analyze_intent(self, query: str) -> dict[str, Any]:
        """Analyze query intent."""
        query_lower = query.lower()
        
        if "compare" in query_lower:
            return {"action": "compare", "multi_backend": True}
        elif "benchmark" in query_lower:
            return {"action": "benchmark", "performance": True}
        elif "analyze" in query_lower:
            return {"action": "analyze", "detailed": True}
        else:
            return {"action": "execute", "simple": True}
    
    def _generate_compare_steps(
        self, intent: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate steps for comparison."""
        return [
            {"action": "create_circuit", "parameters": {"circuit_type": "ghz", "qubits": 3}},
            {"action": "execute", "parameters": {"backend": "cirq", "shots": 1024}},
            {"action": "execute", "parameters": {"backend": "qiskit", "shots": 1024}},
            {"action": "compare", "parameters": {"metrics": ["fidelity", "time"]}},
        ]
    
    def _generate_benchmark_steps(
        self, intent: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate steps for benchmarking."""
        return [
            {"action": "create_circuit", "parameters": {"circuit_type": "ghz", "qubits": 3}},
            {"action": "benchmark", "parameters": {"runs": 5, "warmup": 2}},
            {"action": "analyze", "parameters": {"type": "statistics"}},
        ]
    
    def _generate_analyze_steps(
        self, intent: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate steps for analysis."""
        return [
            {"action": "create_circuit", "parameters": {"circuit_type": "ghz", "qubits": 3}},
            {"action": "execute", "parameters": {"shots": 4096}},
            {"action": "analyze", "parameters": {"type": "full", "include_tomography": True}},
        ]
    
    def _generate_execute_steps(
        self, intent: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate steps for simple execution."""
        return [
            {"action": "create_circuit", "parameters": {"circuit_type": "ghz", "qubits": 3}},
            {"action": "execute", "parameters": {"shots": 1024}},
            {"action": "collect_results", "parameters": {}},
        ]

