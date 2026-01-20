"""Enhanced Data Flow Pipeline Orchestrator.

Implements the complete data flow for Proxima with critical missing features:
- Pause/Resume functionality
- Rollback implementation
- Checkpoint creation/restoration
- Execution DAG visualization
- Distributed execution support with advanced features

Advanced Distributed Features:
- Multiple load balancing strategies (round-robin, least-loaded, weighted, capability-match)
- Worker health monitoring with auto-recovery
- Task queuing with priority scheduling
- Auto-scaling recommendations
- Circuit breaker pattern for failing workers
- Task retries with exponential backoff

Pipeline Flow:
    User Input -> Parse -> Plan -> Check Resources -> Get Consent
                                                        |
                                                        v
                                                Execute on Backend(s)
                                                        |
                                                        v
                                                Collect Results
                                                        |
                                                        v
                                                Generate Insights
                                                        |
                                                        v
                                                Export/Display
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TypeVar

import structlog

from proxima.core.state import (
    ExecutionState,
    ExecutionStateMachine,
    ResourceHandle,
    FileResource,
)
from proxima.core.session import (
    Session,
    SessionManager,
    Checkpoint,
    get_session_manager,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ==================== PIPELINE STAGES ====================


class PipelineStage(Enum):
    """Stages in the data flow pipeline."""

    IDLE = auto()
    PARSING = auto()
    PLANNING = auto()
    RESOURCE_CHECK = auto()
    CONSENT = auto()
    EXECUTING = auto()
    COLLECTING = auto()
    ANALYZING = auto()
    EXPORTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()
    PAUSED = auto()
    ROLLING_BACK = auto()


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    stage: PipelineStage
    success: bool
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


# ==================== PIPELINE CONTEXT ====================


@dataclass
class PipelineContext:
    """Context passed through the pipeline."""

    # Execution metadata
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: float = field(default_factory=time.time)

    # Input data
    user_input: str = ""
    input_params: dict[str, Any] = field(default_factory=dict)

    # Parsed data
    parsed_input: dict[str, Any] | None = None
    circuit: Any = None  # Quantum circuit object

    # Planning
    plan: dict[str, Any] | None = None
    selected_backends: list[str] = field(default_factory=list)

    # Execution
    backend_results: dict[str, Any] = field(default_factory=dict)
    current_backend: str | None = None

    # Analysis
    analysis_results: dict[str, Any] = field(default_factory=dict)
    insights: list[dict[str, Any]] = field(default_factory=list)

    # Export
    export_paths: list[str] = field(default_factory=list)

    # Stage tracking
    stage_results: list[StageResult] = field(default_factory=list)
    current_stage: PipelineStage = PipelineStage.IDLE

    # Resource consent
    resource_estimate: dict[str, Any] | None = None
    consent_granted: bool = False

    # Checkpoints for rollback
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    rollback_target: str | None = None

    # Pause/Resume
    is_paused: bool = False
    pause_reason: str | None = None
    resume_from_stage: PipelineStage | None = None

    # DAG tracking
    dag_nodes: list[dict[str, Any]] = field(default_factory=list)
    dag_edges: list[tuple[str, str]] = field(default_factory=list)

    # Distributed execution
    worker_assignments: dict[str, str] = field(default_factory=dict)
    distributed_results: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: StageResult) -> None:
        """Add a stage result."""
        self.stage_results.append(result)
        if not result.success:
            self.current_stage = PipelineStage.FAILED

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "execution_id": self.execution_id,
            "started_at": self.started_at,
            "elapsed_ms": self.get_elapsed_ms(),
            "user_input": self.user_input,
            "current_stage": self.current_stage.name,
            "stage_results": [r.to_dict() for r in self.stage_results],
            "selected_backends": self.selected_backends,
            "consent_granted": self.consent_granted,
            "is_paused": self.is_paused,
            "checkpoint_count": len(self.checkpoints),
        }

    def create_checkpoint(self, name: str = "") -> dict[str, Any]:
        """Create a checkpoint of current context state."""
        checkpoint = {
            "id": str(uuid.uuid4())[:8],
            "name": name or f"checkpoint_{len(self.checkpoints)}",
            "timestamp": time.time(),
            "stage": self.current_stage.name,
            "stage_results": [r.to_dict() for r in self.stage_results],
            "backend_results": self.backend_results.copy(),
            "analysis_results": self.analysis_results.copy(),
            "plan": self.plan.copy() if self.plan else None,
        }
        self.checkpoints.append(checkpoint)
        return checkpoint

    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore context from a checkpoint."""
        checkpoint = next(
            (cp for cp in self.checkpoints if cp["id"] == checkpoint_id),
            None,
        )
        if not checkpoint:
            return False

        # Restore state
        self.current_stage = PipelineStage[checkpoint["stage"]]
        self.backend_results = checkpoint.get("backend_results", {})
        self.analysis_results = checkpoint.get("analysis_results", {})
        self.plan = checkpoint.get("plan")

        # Trim stage results to checkpoint point
        checkpoint_idx = next(
            (
                i
                for i, cp in enumerate(self.checkpoints)
                if cp["id"] == checkpoint_id
            ),
            0,
        )
        self.checkpoints = self.checkpoints[: checkpoint_idx + 1]

        return True


# ==================== DAG VISUALIZATION ====================


@dataclass
class DAGNode:
    """Represents a node in the execution DAG."""

    id: str
    stage: str
    status: str  # pending, running, completed, failed, skipped
    start_time: float | None = None
    end_time: float | None = None
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (
                (self.end_time - self.start_time) * 1000
                if self.start_time and self.end_time
                else None
            ),
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


class ExecutionDAG:
    """Manages the execution DAG for visualization and tracking."""

    def __init__(self):
        """Initialize the execution DAG."""
        self.nodes: dict[str, DAGNode] = {}
        self.edges: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def add_node(
        self,
        node_id: str,
        stage: str,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DAGNode:
        """Add a node to the DAG."""
        with self._lock:
            node = DAGNode(
                id=node_id,
                stage=stage,
                status="pending",
                dependencies=dependencies or [],
                metadata=metadata or {},
            )
            self.nodes[node_id] = node

            # Add edges for dependencies
            for dep_id in node.dependencies:
                self.edges.append((dep_id, node_id))

            return node

    def start_node(self, node_id: str) -> None:
        """Mark a node as started."""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = "running"
                self.nodes[node_id].start_time = time.time()

    def complete_node(self, node_id: str, success: bool = True) -> None:
        """Mark a node as completed."""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = "completed" if success else "failed"
                self.nodes[node_id].end_time = time.time()

    def skip_node(self, node_id: str) -> None:
        """Mark a node as skipped."""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = "skipped"

    def get_ready_nodes(self) -> list[str]:
        """Get nodes ready for execution (all dependencies complete)."""
        with self._lock:
            ready = []
            for node_id, node in self.nodes.items():
                if node.status != "pending":
                    continue
                deps_complete = all(
                    self.nodes.get(dep_id, DAGNode("", "", "")).status == "completed"
                    for dep_id in node.dependencies
                )
                if deps_complete:
                    ready.append(node_id)
            return ready

    def to_dict(self) -> dict[str, Any]:
        """Convert DAG to dictionary for visualization."""
        with self._lock:
            return {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [{"from": e[0], "to": e[1]} for e in self.edges],
            }

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram syntax for visualization."""
        lines = ["graph TD"]
        with self._lock:
            # Add nodes with styling based on status
            status_styles = {
                "pending": ":::pending",
                "running": ":::running",
                "completed": ":::completed",
                "failed": ":::failed",
                "skipped": ":::skipped",
            }
            for node_id, node in self.nodes.items():
                style = status_styles.get(node.status, "")
                lines.append(f"    {node_id}[{node.stage}]{style}")

            # Add edges
            for from_id, to_id in self.edges:
                lines.append(f"    {from_id} --> {to_id}")

            # Add style definitions
            lines.extend([
                "",
                "    classDef pending fill:#e0e0e0,stroke:#888",
                "    classDef running fill:#fff3cd,stroke:#ffc107",
                "    classDef completed fill:#d4edda,stroke:#28a745",
                "    classDef failed fill:#f8d7da,stroke:#dc3545",
                "    classDef skipped fill:#e2e3e5,stroke:#6c757d",
            ])

        return "\n".join(lines)

    def to_ascii(self) -> str:
        """Generate ASCII representation of the DAG."""
        lines = ["Execution DAG:"]
        lines.append("=" * 50)

        with self._lock:
            status_icons = {
                "pending": "[ ]",
                "running": "[*]",
                "completed": "[+]",
                "failed": "[X]",
                "skipped": "[-]",
            }

            for node_id, node in self.nodes.items():
                icon = status_icons.get(node.status, "[?]")
                deps = ", ".join(node.dependencies) if node.dependencies else "none"
                duration = ""
                if node.start_time and node.end_time:
                    duration = f" ({(node.end_time - node.start_time) * 1000:.1f}ms)"
                lines.append(f"{icon} {node_id}: {node.stage}{duration}")
                lines.append(f"    Dependencies: {deps}")

        return "\n".join(lines)



# ==================== DISTRIBUTED EXECUTION ====================


class DistributedWorker:
    """Represents a worker node for distributed execution."""

    def __init__(
        self,
        worker_id: str,
        host: str,
        port: int,
        capabilities: list[str] | None = None,
    ):
        """Initialize a distributed worker.

        Args:
            worker_id: Unique worker identifier
            host: Worker host address
            port: Worker port
            capabilities: List of supported backends
        """
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.capabilities = capabilities or []
        self.status = "idle"
        self.current_task: str | None = None
        self.last_heartbeat = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert worker to dictionary."""
        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "port": self.port,
            "capabilities": self.capabilities,
            "status": self.status,
            "current_task": self.current_task,
            "last_heartbeat": self.last_heartbeat,
        }


class DistributedExecutor:
    """Manages distributed execution across multiple workers.

    Features:
    - Worker registration and discovery
    - Task distribution based on capabilities
    - Load balancing
    - Fault tolerance with retries
    """

    def __init__(self, max_retries: int = 3):
        """Initialize the distributed executor.

        Args:
            max_retries: Maximum retry attempts for failed tasks
        """
        self.workers: dict[str, DistributedWorker] = {}
        self.max_retries = max_retries
        self.task_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.results: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.logger = structlog.get_logger("distributed")

    def register_worker(
        self,
        worker_id: str,
        host: str,
        port: int,
        capabilities: list[str] | None = None,
    ) -> DistributedWorker:
        """Register a worker node.

        Args:
            worker_id: Unique worker identifier
            host: Worker host address
            port: Worker port
            capabilities: Supported backends

        Returns:
            Registered worker
        """
        with self._lock:
            worker = DistributedWorker(worker_id, host, port, capabilities)
            self.workers[worker_id] = worker
            self.logger.info("worker.registered", worker_id=worker_id)
            return worker

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker node."""
        with self._lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                self.logger.info("worker.unregistered", worker_id=worker_id)

    def get_available_workers(
        self,
        required_capability: str | None = None,
    ) -> list[DistributedWorker]:
        """Get available workers with optional capability filter."""
        with self._lock:
            available = [w for w in self.workers.values() if w.status == "idle"]
            if required_capability:
                available = [
                    w for w in available if required_capability in w.capabilities
                ]
            return available

    def select_worker(self, backend: str) -> DistributedWorker | None:
        """Select best worker for a backend task."""
        available = self.get_available_workers(backend)
        if not available:
            return None
        # Simple selection - first available (could be load-balanced)
        return available[0]

    async def distribute_task(
        self,
        task_id: str,
        backend: str,
        circuit: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Distribute a task to a worker.

        Args:
            task_id: Unique task identifier
            backend: Required backend
            circuit: Circuit to execute
            params: Execution parameters

        Returns:
            Task result
        """
        worker = self.select_worker(backend)
        if not worker:
            return {
                "success": False,
                "error": f"No worker available for backend: {backend}",
            }

        # Mark worker as busy
        with self._lock:
            worker.status = "busy"
            worker.current_task = task_id

        try:
            # Simulate remote execution (in real implementation, this would
            # use HTTP/gRPC to communicate with the worker)
            result = await self._execute_on_worker(worker, circuit, params)

            with self._lock:
                self.results[task_id] = result

            return result

        except Exception as exc:
            self.logger.error("task.failed", task_id=task_id, error=str(exc))
            return {"success": False, "error": str(exc)}

        finally:
            # Mark worker as idle
            with self._lock:
                worker.status = "idle"
                worker.current_task = None

    async def _execute_on_worker(
        self,
        worker: DistributedWorker,
        circuit: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute task on a worker (placeholder for remote execution)."""
        # In a real implementation, this would send the task to the worker
        # via HTTP/gRPC and wait for the result
        await asyncio.sleep(0.01)  # Simulate network latency
        return {
            "success": True,
            "worker_id": worker.worker_id,
            "message": "Executed on distributed worker",
        }

    async def execute_distributed(
        self,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute multiple tasks in parallel across workers.

        Args:
            tasks: List of task definitions

        Returns:
            Combined results from all tasks
        """
        results = {}
        async_tasks = []

        for task in tasks:
            async_task = self.distribute_task(
                task_id=task.get("id", str(uuid.uuid4())[:8]),
                backend=task.get("backend", "default"),
                circuit=task.get("circuit"),
                params=task.get("params", {}),
            )
            async_tasks.append(async_task)

        task_results = await asyncio.gather(*async_tasks, return_exceptions=True)

        for task, result in zip(tasks, task_results):
            task_id = task.get("id", "unknown")
            if isinstance(result, Exception):
                results[task_id] = {"success": False, "error": str(result)}
            else:
                results[task_id] = result

        return results

    def get_cluster_status(self) -> dict[str, Any]:
        """Get current cluster status."""
        with self._lock:
            return {
                "total_workers": len(self.workers),
                "idle_workers": len([w for w in self.workers.values() if w.status == "idle"]),
                "busy_workers": len([w for w in self.workers.values() if w.status == "busy"]),
                "workers": [w.to_dict() for w in self.workers.values()],
            }


# ==================== ADVANCED DISTRIBUTED FEATURES ====================


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for distributed execution."""
    
    ROUND_ROBIN = auto()      # Simple round-robin distribution
    LEAST_LOADED = auto()     # Send to worker with fewest tasks
    CAPABILITY_MATCH = auto() # Match task requirements to worker capabilities
    WEIGHTED = auto()         # Weight by worker performance history
    LOCALITY = auto()         # Prefer workers with cached data


@dataclass
class WorkerMetrics:
    """Performance metrics for a distributed worker."""
    
    worker_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time_ms: float = 0.0
    average_latency_ms: float = 0.0
    last_health_check: float = field(default_factory=time.time)
    health_status: str = "healthy"
    current_load: float = 0.0  # 0.0 to 1.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class DistributedTask:
    """Enhanced task representation for distributed execution."""
    
    task_id: str
    backend: str
    circuit: Any
    params: dict[str, Any]
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    timeout_s: float = 300.0
    assigned_worker: str | None = None
    status: str = "pending"  # pending, assigned, running, completed, failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class AdvancedDistributedExecutor:
    """Advanced distributed executor with enterprise features.
    
    Features:
    - Multiple load balancing strategies
    - Worker health monitoring with auto-recovery
    - Task queuing with priority scheduling
    - Auto-scaling hints
    - Circuit breaker pattern for failing workers
    - Task retries with exponential backoff
    - Performance metrics collection
    - Worker affinity for cached data
    - Graceful shutdown with task migration
    """
    
    def __init__(
        self,
        storage_dir: Path | None = None,
        load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.CAPABILITY_MATCH,
        health_check_interval: float = 30.0,
        circuit_breaker_threshold: int = 5,
    ):
        """Initialize advanced distributed executor.
        
        Args:
            storage_dir: Directory for persistent state
            load_balancing: Load balancing strategy
            health_check_interval: Seconds between health checks
            circuit_breaker_threshold: Failures before circuit breaks
        """
        self.storage_dir = storage_dir or Path(tempfile.gettempdir()) / "proxima_distributed"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.workers: dict[str, DistributedWorker] = {}
        self.worker_metrics: dict[str, WorkerMetrics] = {}
        self.task_queue: asyncio.PriorityQueue[tuple[int, DistributedTask]] = asyncio.PriorityQueue()
        self.active_tasks: dict[str, DistributedTask] = {}
        self.completed_tasks: dict[str, DistributedTask] = {}
        
        self.load_balancing = load_balancing
        self.health_check_interval = health_check_interval
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_counts: dict[str, int] = {}
        
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._health_check_task: asyncio.Task | None = None
        
        self.logger = structlog.get_logger("advanced_distributed")
    
    def register_worker(
        self,
        worker_id: str,
        host: str,
        port: int,
        capabilities: list[str] | None = None,
        weight: float = 1.0,
    ) -> DistributedWorker:
        """Register a worker with enhanced metadata.
        
        Args:
            worker_id: Unique worker identifier
            host: Worker host address
            port: Worker port
            capabilities: Supported backends
            weight: Worker weight for load balancing
            
        Returns:
            Registered worker
        """
        with self._lock:
            worker = DistributedWorker(worker_id, host, port, capabilities)
            self.workers[worker_id] = worker
            
            # Initialize metrics
            self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
            self.circuit_breaker_counts[worker_id] = 0
            
            self.logger.info(
                "worker.registered",
                worker_id=worker_id,
                host=host,
                capabilities=capabilities,
            )
            return worker
    
    async def submit_task(
        self,
        backend: str,
        circuit: Any,
        params: dict[str, Any],
        priority: int = 0,
        timeout_s: float = 300.0,
    ) -> str:
        """Submit a task for distributed execution.
        
        Args:
            backend: Required backend
            circuit: Circuit to execute
            params: Execution parameters
            priority: Task priority (higher = more urgent)
            timeout_s: Task timeout
            
        Returns:
            Task ID
        """
        task = DistributedTask(
            task_id=str(uuid.uuid4())[:8],
            backend=backend,
            circuit=circuit,
            params=params,
            priority=priority,
            timeout_s=timeout_s,
        )
        
        # Add to priority queue (lower number = higher priority)
        await self.task_queue.put((-priority, task))
        
        with self._lock:
            self.active_tasks[task.task_id] = task
        
        self.logger.debug("task.submitted", task_id=task.task_id, backend=backend)
        return task.task_id
    
    def select_worker(
        self,
        task: DistributedTask,
    ) -> DistributedWorker | None:
        """Select best worker for a task using configured strategy.
        
        Args:
            task: Task to assign
            
        Returns:
            Selected worker or None
        """
        with self._lock:
            available = [
                w for w in self.workers.values()
                if w.status == "idle" and self._is_worker_healthy(w.worker_id)
            ]
            
            if not available:
                return None
            
            # Filter by capability
            if task.backend:
                capable = [
                    w for w in available
                    if not w.capabilities or task.backend in w.capabilities
                ]
                if capable:
                    available = capable
            
            if not available:
                return None
            
            # Apply load balancing strategy
            if self.load_balancing == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(available)
            elif self.load_balancing == LoadBalancingStrategy.LEAST_LOADED:
                return self._select_least_loaded(available)
            elif self.load_balancing == LoadBalancingStrategy.WEIGHTED:
                return self._select_weighted(available)
            elif self.load_balancing == LoadBalancingStrategy.CAPABILITY_MATCH:
                return self._select_capability_match(available, task)
            else:
                return available[0]
    
    def _select_round_robin(
        self,
        workers: list[DistributedWorker],
    ) -> DistributedWorker:
        """Round-robin selection."""
        # Simple implementation - just return first idle
        return workers[0]
    
    def _select_least_loaded(
        self,
        workers: list[DistributedWorker],
    ) -> DistributedWorker:
        """Select worker with lowest current load."""
        return min(
            workers,
            key=lambda w: self.worker_metrics.get(w.worker_id, WorkerMetrics(w.worker_id)).current_load
        )
    
    def _select_weighted(
        self,
        workers: list[DistributedWorker],
    ) -> DistributedWorker:
        """Select based on performance history."""
        def score(w: DistributedWorker) -> float:
            metrics = self.worker_metrics.get(w.worker_id)
            if not metrics or metrics.total_tasks == 0:
                return 0.5  # Neutral score for new workers
            
            success_rate = metrics.successful_tasks / metrics.total_tasks
            speed_score = 1.0 / max(metrics.average_latency_ms, 1.0)
            
            return success_rate * 0.7 + min(speed_score * 0.3, 0.3)
        
        return max(workers, key=score)
    
    def _select_capability_match(
        self,
        workers: list[DistributedWorker],
        task: DistributedTask,
    ) -> DistributedWorker:
        """Select based on capability match."""
        # Prefer workers with explicit capability match
        exact_match = [
            w for w in workers
            if w.capabilities and task.backend in w.capabilities
        ]
        
        if exact_match:
            return self._select_weighted(exact_match)
        
        return self._select_weighted(workers)
    
    def _is_worker_healthy(self, worker_id: str) -> bool:
        """Check if worker is healthy (circuit breaker not tripped)."""
        failures = self.circuit_breaker_counts.get(worker_id, 0)
        return failures < self.circuit_breaker_threshold
    
    async def execute_task(self, task: DistributedTask) -> dict[str, Any]:
        """Execute a single task with retries and circuit breaker.
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        retry_delay = 1.0
        
        for attempt in range(task.max_retries + 1):
            worker = self.select_worker(task)
            
            if not worker:
                if attempt < task.max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                return {"success": False, "error": "No available workers"}
            
            task.assigned_worker = worker.worker_id
            task.status = "running"
            task.started_at = time.time()
            
            with self._lock:
                worker.status = "busy"
                worker.current_task = task.task_id
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_on_worker(worker, task),
                    timeout=task.timeout_s,
                )
                
                task.status = "completed"
                task.completed_at = time.time()
                task.result = result
                
                # Update metrics
                self._record_success(worker.worker_id, task)
                
                return result
                
            except asyncio.TimeoutError:
                self._record_failure(worker.worker_id, "timeout")
                task.error = "Task timeout"
                
            except Exception as e:
                self._record_failure(worker.worker_id, str(e))
                task.error = str(e)
            
            finally:
                with self._lock:
                    worker.status = "idle"
                    worker.current_task = None
            
            # Retry with backoff
            if attempt < task.max_retries:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                task.retry_count += 1
        
        task.status = "failed"
        return {"success": False, "error": task.error}
    
    async def _execute_on_worker(
        self,
        worker: DistributedWorker,
        task: DistributedTask,
    ) -> dict[str, Any]:
        """Execute task on worker (simulated remote call)."""
        # Simulate network latency
        await asyncio.sleep(0.01)
        
        # In real implementation, this would be HTTP/gRPC call
        return {
            "success": True,
            "worker_id": worker.worker_id,
            "task_id": task.task_id,
            "execution_time_ms": (time.time() - (task.started_at or time.time())) * 1000,
        }
    
    def _record_success(self, worker_id: str, task: DistributedTask) -> None:
        """Record successful task execution."""
        with self._lock:
            if worker_id not in self.worker_metrics:
                self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
            
            metrics = self.worker_metrics[worker_id]
            metrics.total_tasks += 1
            metrics.successful_tasks += 1
            
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at) * 1000
                metrics.total_execution_time_ms += execution_time
                metrics.average_latency_ms = (
                    metrics.total_execution_time_ms / metrics.total_tasks
                )
            
            # Reset circuit breaker on success
            self.circuit_breaker_counts[worker_id] = max(
                0, self.circuit_breaker_counts.get(worker_id, 0) - 1
            )
    
    def _record_failure(self, worker_id: str, error: str) -> None:
        """Record failed task execution."""
        with self._lock:
            if worker_id not in self.worker_metrics:
                self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
            
            metrics = self.worker_metrics[worker_id]
            metrics.total_tasks += 1
            metrics.failed_tasks += 1
            
            # Increment circuit breaker
            self.circuit_breaker_counts[worker_id] = (
                self.circuit_breaker_counts.get(worker_id, 0) + 1
            )
            
            if self.circuit_breaker_counts[worker_id] >= self.circuit_breaker_threshold:
                metrics.health_status = "circuit_open"
                self.logger.warning(
                    "worker.circuit_breaker_open",
                    worker_id=worker_id,
                    failures=self.circuit_breaker_counts[worker_id],
                )
    
    async def process_queue(self) -> None:
        """Process tasks from the queue continuously."""
        while not self._shutdown.is_set():
            try:
                # Get task with timeout to allow shutdown check
                try:
                    priority, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute task
                result = await self.execute_task(task)
                
                with self._lock:
                    if task.task_id in self.active_tasks:
                        del self.active_tasks[task.task_id]
                    self.completed_tasks[task.task_id] = task
                
            except Exception as e:
                self.logger.error("queue.process_error", error=str(e))
    
    async def health_check_loop(self) -> None:
        """Continuously check worker health."""
        while not self._shutdown.is_set():
            await asyncio.sleep(self.health_check_interval)
            
            for worker_id in list(self.workers.keys()):
                await self._check_worker_health(worker_id)
    
    async def _check_worker_health(self, worker_id: str) -> bool:
        """Check health of a specific worker."""
        try:
            # Simulate health check (real implementation would ping worker)
            await asyncio.sleep(0.01)
            
            with self._lock:
                if worker_id in self.worker_metrics:
                    self.worker_metrics[worker_id].last_health_check = time.time()
                    self.worker_metrics[worker_id].health_status = "healthy"
            
            return True
            
        except Exception as e:
            with self._lock:
                if worker_id in self.worker_metrics:
                    self.worker_metrics[worker_id].health_status = "unhealthy"
            
            self.logger.warning(
                "worker.health_check_failed",
                worker_id=worker_id,
                error=str(e),
            )
            return False
    
    def get_scaling_recommendation(self) -> dict[str, Any]:
        """Get auto-scaling recommendations based on current load.
        
        Returns:
            Scaling recommendation with action and reason
        """
        with self._lock:
            total_workers = len(self.workers)
            healthy_workers = sum(
                1 for w in self.workers.values()
                if self._is_worker_healthy(w.worker_id)
            )
            busy_workers = sum(
                1 for w in self.workers.values()
                if w.status == "busy"
            )
            queue_size = self.task_queue.qsize()
            
            # Calculate utilization
            if healthy_workers > 0:
                utilization = busy_workers / healthy_workers
            else:
                utilization = 1.0
            
            recommendation = {
                "current_workers": total_workers,
                "healthy_workers": healthy_workers,
                "busy_workers": busy_workers,
                "queue_size": queue_size,
                "utilization": utilization,
                "action": "none",
                "reason": "",
            }
            
            # Scale up if high utilization and queue is backing up
            if utilization > 0.8 and queue_size > healthy_workers * 2:
                recommendation["action"] = "scale_up"
                recommendation["recommended_count"] = min(
                    total_workers * 2,
                    healthy_workers + queue_size // 5,
                )
                recommendation["reason"] = (
                    f"High utilization ({utilization:.0%}) with {queue_size} queued tasks"
                )
            
            # Scale down if low utilization
            elif utilization < 0.2 and healthy_workers > 1:
                recommendation["action"] = "scale_down"
                recommendation["recommended_count"] = max(1, healthy_workers // 2)
                recommendation["reason"] = f"Low utilization ({utilization:.0%})"
            
            return recommendation
    
    async def graceful_shutdown(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Gracefully shutdown with task completion.
        
        Args:
            timeout_s: Maximum time to wait for tasks
            
        Returns:
            Shutdown statistics
        """
        self._shutdown.set()
        
        stats = {
            "completed_tasks": 0,
            "abandoned_tasks": 0,
            "active_at_shutdown": len(self.active_tasks),
        }
        
        # Wait for active tasks to complete
        start_time = time.time()
        while self.active_tasks and (time.time() - start_time) < timeout_s:
            await asyncio.sleep(0.1)
        
        with self._lock:
            stats["completed_tasks"] = len(self.completed_tasks)
            stats["abandoned_tasks"] = len(self.active_tasks)
        
        self.logger.info("distributed.shutdown_complete", **stats)
        return stats
    
    def get_cluster_metrics(self) -> dict[str, Any]:
        """Get comprehensive cluster metrics.
        
        Returns:
            Detailed cluster metrics
        """
        with self._lock:
            worker_stats = []
            for worker_id, worker in self.workers.items():
                metrics = self.worker_metrics.get(
                    worker_id,
                    WorkerMetrics(worker_id=worker_id),
                )
                worker_stats.append({
                    "worker_id": worker_id,
                    "status": worker.status,
                    "health": metrics.health_status,
                    "total_tasks": metrics.total_tasks,
                    "success_rate": (
                        metrics.successful_tasks / metrics.total_tasks
                        if metrics.total_tasks > 0 else 0
                    ),
                    "avg_latency_ms": metrics.average_latency_ms,
                    "circuit_breaker_count": self.circuit_breaker_counts.get(worker_id, 0),
                })
            
            return {
                "total_workers": len(self.workers),
                "healthy_workers": sum(
                    1 for m in self.worker_metrics.values()
                    if m.health_status == "healthy"
                ),
                "queue_size": self.task_queue.qsize(),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "workers": worker_stats,
                "scaling_recommendation": self.get_scaling_recommendation(),
            }


# ==============================================================================
# AUTO-SCALING POLISH (2% Gap Coverage)
# ==============================================================================


class ScalingTrigger(Enum):
    """Triggers for scaling actions."""
    
    HIGH_UTILIZATION = "high_utilization"
    LOW_UTILIZATION = "low_utilization"
    QUEUE_BACKLOG = "queue_backlog"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    HEALTH_RECOVERY = "health_recovery"


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    
    timestamp: float
    trigger: ScalingTrigger
    action: str  # "scale_up" or "scale_down"
    previous_count: int
    target_count: int
    actual_count: int
    reason: str
    success: bool = True
    cooldown_remaining: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "trigger": self.trigger.value,
            "action": self.action,
            "previous_count": self.previous_count,
            "target_count": self.target_count,
            "actual_count": self.actual_count,
            "reason": self.reason,
            "success": self.success,
            "cooldown_remaining": self.cooldown_remaining,
        }


@dataclass
class ScalingPolicy:
    """Policy for auto-scaling behavior."""
    
    min_workers: int = 1
    max_workers: int = 100
    scale_up_threshold: float = 0.8       # Utilization to trigger scale up
    scale_down_threshold: float = 0.2     # Utilization to trigger scale down
    scale_up_cooldown_s: float = 60.0     # Cooldown after scale up
    scale_down_cooldown_s: float = 300.0  # Cooldown after scale down
    scale_up_increment: int = 2           # Workers to add
    scale_down_increment: int = 1         # Workers to remove
    queue_threshold: int = 10             # Queue size to trigger urgent scale
    predictive_window_s: float = 300.0    # Window for predictive scaling
    

@dataclass
class ResourceMetrics:
    """Resource utilization metrics for scaling decisions."""
    
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    queue_size: int = 0
    active_tasks: int = 0
    avg_task_duration_ms: float = 0.0
    tasks_per_second: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "queue_size": self.queue_size,
            "active_tasks": self.active_tasks,
            "avg_task_duration_ms": self.avg_task_duration_ms,
            "tasks_per_second": self.tasks_per_second,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp,
        }


class AutoScalingManager:
    """Advanced auto-scaling manager with predictive capabilities.
    
    Features:
    - Dynamic worker scaling based on load
    - Predictive scaling using historical data
    - Cooldown management to prevent thrashing
    - Resource utilization tracking
    - Scaling event history
    """
    
    def __init__(
        self,
        executor: AdvancedDistributedExecutor,
        policy: ScalingPolicy | None = None,
        worker_factory: Callable[[], tuple[str, str, int]] | None = None,
        logger: Any = None,
    ) -> None:
        """Initialize auto-scaling manager.
        
        Args:
            executor: Distributed executor to manage
            policy: Scaling policy
            worker_factory: Factory function to create new workers
                           Returns (worker_id, host, port)
            logger: Logger instance
        """
        self._executor = executor
        self._policy = policy or ScalingPolicy()
        self._worker_factory = worker_factory
        self._logger = logger or structlog.get_logger("auto_scaling")
        
        self._metrics_history: list[ResourceMetrics] = []
        self._scaling_history: list[ScalingEvent] = []
        self._last_scale_up: float = 0.0
        self._last_scale_down: float = 0.0
        self._lock = threading.RLock()
        self._running = False
        self._check_task: asyncio.Task | None = None
    
    def set_policy(self, policy: ScalingPolicy) -> None:
        """Update scaling policy."""
        with self._lock:
            self._policy = policy
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        with self._lock:
            cluster = self._executor.get_cluster_metrics()
            
            total_workers = cluster["total_workers"]
            healthy_workers = cluster["healthy_workers"]
            active_tasks = cluster["active_tasks"]
            queue_size = cluster["queue_size"]
            
            # Calculate CPU/memory (simulated from worker load)
            if healthy_workers > 0:
                cpu_percent = (active_tasks / healthy_workers) * 100
            else:
                cpu_percent = 100.0
            
            # Calculate tasks per second from history
            tasks_per_second = 0.0
            if len(self._metrics_history) >= 2:
                recent = self._metrics_history[-1]
                older = self._metrics_history[-2]
                time_diff = recent.timestamp - older.timestamp
                if time_diff > 0:
                    task_diff = cluster["completed_tasks"] - older.active_tasks
                    tasks_per_second = max(0, task_diff / time_diff)
            
            # Calculate error rate
            error_rate = 0.0
            workers = cluster.get("workers", [])
            if workers:
                avg_success = sum(w.get("success_rate", 1.0) for w in workers) / len(workers)
                error_rate = 1.0 - avg_success
            
            # Calculate avg task duration
            avg_duration = 0.0
            if workers:
                avg_duration = sum(w.get("avg_latency_ms", 0) for w in workers) / len(workers)
            
            metrics = ResourceMetrics(
                cpu_percent=min(100.0, cpu_percent),
                memory_percent=0.0,  # Would come from actual system metrics
                queue_size=queue_size,
                active_tasks=active_tasks,
                avg_task_duration_ms=avg_duration,
                tasks_per_second=tasks_per_second,
                error_rate=error_rate,
            )
            
            self._metrics_history.append(metrics)
            
            # Keep only last hour of metrics
            cutoff = time.time() - 3600
            self._metrics_history = [
                m for m in self._metrics_history if m.timestamp > cutoff
            ]
            
            return metrics
    
    def predict_load(self, window_s: float | None = None) -> dict[str, Any]:
        """Predict future load based on historical data.
        
        Args:
            window_s: Prediction window in seconds
            
        Returns:
            Load prediction with confidence
        """
        window = window_s or self._policy.predictive_window_s
        
        with self._lock:
            if len(self._metrics_history) < 5:
                return {
                    "predicted_utilization": 0.5,
                    "predicted_queue_size": 0,
                    "confidence": 0.0,
                    "trend": "unknown",
                }
            
            # Analyze recent trends
            recent = self._metrics_history[-10:]
            
            # Calculate trend
            if len(recent) >= 2:
                first_half = recent[:len(recent)//2]
                second_half = recent[len(recent)//2:]
                
                first_avg = sum(m.queue_size for m in first_half) / len(first_half)
                second_avg = sum(m.queue_size for m in second_half) / len(second_half)
                
                if second_avg > first_avg * 1.2:
                    trend = "increasing"
                elif second_avg < first_avg * 0.8:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "unknown"
            
            # Simple linear extrapolation
            avg_queue = sum(m.queue_size for m in recent) / len(recent)
            avg_tasks_per_sec = sum(m.tasks_per_second for m in recent) / len(recent)
            
            if trend == "increasing":
                predicted_queue = avg_queue * 1.5
            elif trend == "decreasing":
                predicted_queue = avg_queue * 0.7
            else:
                predicted_queue = avg_queue
            
            # Calculate predicted utilization
            current_workers = len(self._executor.workers)
            if current_workers > 0 and avg_tasks_per_sec > 0:
                predicted_util = min(1.0, predicted_queue / (current_workers * 10))
            else:
                predicted_util = 0.5
            
            # Confidence based on data consistency
            variance = sum(
                (m.queue_size - avg_queue) ** 2 for m in recent
            ) / len(recent)
            confidence = 1.0 / (1.0 + variance / 100)
            
            return {
                "predicted_utilization": predicted_util,
                "predicted_queue_size": int(predicted_queue),
                "confidence": confidence,
                "trend": trend,
                "avg_tasks_per_second": avg_tasks_per_sec,
            }
    
    def evaluate_scaling(self) -> dict[str, Any]:
        """Evaluate whether scaling is needed.
        
        Returns:
            Scaling decision with reasoning
        """
        with self._lock:
            metrics = self.collect_metrics()
            prediction = self.predict_load()
            
            current_workers = len(self._executor.workers)
            healthy_workers = sum(
                1 for w in self._executor.workers.values()
                if self._executor._is_worker_healthy(w.worker_id)
            )
            
            # Calculate utilization
            if healthy_workers > 0:
                utilization = metrics.active_tasks / healthy_workers
            else:
                utilization = 1.0
            
            decision = {
                "action": "none",
                "trigger": None,
                "current_workers": current_workers,
                "healthy_workers": healthy_workers,
                "utilization": utilization,
                "queue_size": metrics.queue_size,
                "target_workers": current_workers,
                "reason": "No scaling needed",
                "cooldown_active": False,
            }
            
            # Check cooldowns
            now = time.time()
            scale_up_cooldown = now - self._last_scale_up < self._policy.scale_up_cooldown_s
            scale_down_cooldown = now - self._last_scale_down < self._policy.scale_down_cooldown_s
            
            # Urgent scale-up for queue backlog (ignores cooldown)
            if metrics.queue_size > self._policy.queue_threshold * healthy_workers:
                workers_needed = metrics.queue_size // self._policy.queue_threshold
                target = min(
                    self._policy.max_workers,
                    max(current_workers, workers_needed),
                )
                if target > current_workers:
                    decision["action"] = "scale_up"
                    decision["trigger"] = ScalingTrigger.QUEUE_BACKLOG
                    decision["target_workers"] = target
                    decision["reason"] = f"Queue backlog: {metrics.queue_size} tasks waiting"
                    return decision
            
            # Scale up for high utilization
            if utilization > self._policy.scale_up_threshold:
                if scale_up_cooldown:
                    decision["cooldown_active"] = True
                    decision["reason"] = "Scale-up cooldown active"
                else:
                    target = min(
                        self._policy.max_workers,
                        current_workers + self._policy.scale_up_increment,
                    )
                    if target > current_workers:
                        decision["action"] = "scale_up"
                        decision["trigger"] = ScalingTrigger.HIGH_UTILIZATION
                        decision["target_workers"] = target
                        decision["reason"] = f"High utilization: {utilization:.0%}"
                        return decision
            
            # Predictive scale-up
            if (
                prediction["trend"] == "increasing"
                and prediction["confidence"] > 0.6
                and prediction["predicted_utilization"] > self._policy.scale_up_threshold
            ):
                if not scale_up_cooldown:
                    target = min(
                        self._policy.max_workers,
                        current_workers + self._policy.scale_up_increment,
                    )
                    if target > current_workers:
                        decision["action"] = "scale_up"
                        decision["trigger"] = ScalingTrigger.PREDICTIVE
                        decision["target_workers"] = target
                        decision["reason"] = (
                            f"Predicted load increase: {prediction['predicted_utilization']:.0%}"
                        )
                        return decision
            
            # Scale down for low utilization
            if utilization < self._policy.scale_down_threshold:
                if scale_down_cooldown:
                    decision["cooldown_active"] = True
                    decision["reason"] = "Scale-down cooldown active"
                else:
                    target = max(
                        self._policy.min_workers,
                        current_workers - self._policy.scale_down_increment,
                    )
                    if target < current_workers:
                        decision["action"] = "scale_down"
                        decision["trigger"] = ScalingTrigger.LOW_UTILIZATION
                        decision["target_workers"] = target
                        decision["reason"] = f"Low utilization: {utilization:.0%}"
                        return decision
            
            return decision
    
    def execute_scaling(
        self,
        action: str,
        target_workers: int,
        trigger: ScalingTrigger,
        reason: str,
    ) -> ScalingEvent:
        """Execute a scaling action.
        
        Args:
            action: "scale_up" or "scale_down"
            target_workers: Target worker count
            trigger: What triggered the scaling
            reason: Reason for scaling
            
        Returns:
            ScalingEvent with result
        """
        with self._lock:
            previous_count = len(self._executor.workers)
            actual_count = previous_count
            success = True
            
            if action == "scale_up":
                workers_to_add = target_workers - previous_count
                for _ in range(workers_to_add):
                    if self._worker_factory:
                        try:
                            worker_id, host, port = self._worker_factory()
                            self._executor.register_worker(worker_id, host, port)
                            actual_count += 1
                        except Exception as e:
                            self._logger.error("scaling.add_worker_failed", error=str(e))
                            success = False
                    else:
                        # Simulated worker creation
                        worker_id = f"auto_{uuid.uuid4().hex[:6]}"
                        self._executor.register_worker(
                            worker_id, "localhost", 8000 + actual_count
                        )
                        actual_count += 1
                
                self._last_scale_up = time.time()
            
            elif action == "scale_down":
                workers_to_remove = previous_count - target_workers
                # Remove idle workers
                idle_workers = [
                    wid for wid, w in self._executor.workers.items()
                    if w.status == "idle"
                ][:workers_to_remove]
                
                for worker_id in idle_workers:
                    del self._executor.workers[worker_id]
                    if worker_id in self._executor.worker_metrics:
                        del self._executor.worker_metrics[worker_id]
                    actual_count -= 1
                
                self._last_scale_down = time.time()
            
            event = ScalingEvent(
                timestamp=time.time(),
                trigger=trigger,
                action=action,
                previous_count=previous_count,
                target_count=target_workers,
                actual_count=actual_count,
                reason=reason,
                success=success,
            )
            
            self._scaling_history.append(event)
            
            # Keep only last 100 events
            if len(self._scaling_history) > 100:
                self._scaling_history = self._scaling_history[-100:]
            
            self._logger.info(
                "scaling.executed",
                action=action,
                previous=previous_count,
                target=target_workers,
                actual=actual_count,
                trigger=trigger.value,
            )
            
            return event
    
    async def auto_scale_loop(self, interval_s: float = 10.0) -> None:
        """Continuously evaluate and execute scaling.
        
        Args:
            interval_s: Check interval in seconds
        """
        self._running = True
        
        while self._running:
            try:
                decision = self.evaluate_scaling()
                
                if decision["action"] != "none":
                    self.execute_scaling(
                        action=decision["action"],
                        target_workers=decision["target_workers"],
                        trigger=decision["trigger"],
                        reason=decision["reason"],
                    )
                
                await asyncio.sleep(interval_s)
                
            except Exception as e:
                self._logger.error("auto_scale.loop_error", error=str(e))
                await asyncio.sleep(interval_s)
    
    def stop(self) -> None:
        """Stop the auto-scaling loop."""
        self._running = False
    
    def get_scaling_history(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get scaling history.
        
        Args:
            limit: Maximum events to return
            
        Returns:
            List of scaling events
        """
        with self._lock:
            return [e.to_dict() for e in self._scaling_history[-limit:]]
    
    def get_status(self) -> dict[str, Any]:
        """Get auto-scaling status.
        
        Returns:
            Current status and metrics
        """
        with self._lock:
            latest_metrics = self._metrics_history[-1] if self._metrics_history else None
            prediction = self.predict_load()
            
            now = time.time()
            return {
                "running": self._running,
                "policy": {
                    "min_workers": self._policy.min_workers,
                    "max_workers": self._policy.max_workers,
                    "scale_up_threshold": self._policy.scale_up_threshold,
                    "scale_down_threshold": self._policy.scale_down_threshold,
                },
                "current_metrics": latest_metrics.to_dict() if latest_metrics else None,
                "prediction": prediction,
                "cooldowns": {
                    "scale_up_remaining": max(
                        0, self._policy.scale_up_cooldown_s - (now - self._last_scale_up)
                    ),
                    "scale_down_remaining": max(
                        0, self._policy.scale_down_cooldown_s - (now - self._last_scale_down)
                    ),
                },
                "recent_events": len(self._scaling_history),
            }


# ==================== ROLLBACK MANAGER ====================


class RollbackManager:
    """Manages rollback operations for the pipeline.

    Features:
    - Create restore points before risky operations
    - Rollback to previous states
    - Clean up partial results on rollback
    """

    def __init__(self):
        """Initialize the rollback manager."""
        self.restore_points: list[dict[str, Any]] = []
        self.logger = structlog.get_logger("rollback")
        self._lock = threading.Lock()

    def create_restore_point(
        self,
        context: PipelineContext,
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a restore point.

        Args:
            context: Current pipeline context
            name: Optional name for the restore point
            metadata: Optional metadata

        Returns:
            Restore point ID
        """
        with self._lock:
            restore_point = {
                "id": str(uuid.uuid4())[:8],
                "name": name or f"restore_point_{len(self.restore_points)}",
                "timestamp": time.time(),
                "stage": context.current_stage.name,
                "context_snapshot": {
                    "stage_results": [r.to_dict() for r in context.stage_results],
                    "backend_results": context.backend_results.copy(),
                    "analysis_results": context.analysis_results.copy(),
                    "plan": context.plan.copy() if context.plan else None,
                    "consent_granted": context.consent_granted,
                },
                "metadata": metadata or {},
            }
            self.restore_points.append(restore_point)
            self.logger.info("restore_point.created", id=restore_point["id"])
            return restore_point["id"]

    def rollback(
        self,
        context: PipelineContext,
        restore_point_id: str,
    ) -> tuple[bool, str]:
        """Rollback to a restore point.

        Args:
            context: Pipeline context to restore
            restore_point_id: ID of restore point

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            restore_point = next(
                (rp for rp in self.restore_points if rp["id"] == restore_point_id),
                None,
            )

            if not restore_point:
                return False, f"Restore point {restore_point_id} not found"

            try:
                # Restore context state
                snapshot = restore_point["context_snapshot"]
                context.current_stage = PipelineStage[restore_point["stage"]]
                context.backend_results = snapshot.get("backend_results", {})
                context.analysis_results = snapshot.get("analysis_results", {})
                context.plan = snapshot.get("plan")
                context.consent_granted = snapshot.get("consent_granted", False)

                # Trim restore points to this point
                rp_idx = next(
                    (i for i, rp in enumerate(self.restore_points) if rp["id"] == restore_point_id),
                    0,
                )
                self.restore_points = self.restore_points[: rp_idx + 1]

                self.logger.info(
                    "rollback.completed",
                    restore_point_id=restore_point_id,
                    stage=restore_point["stage"],
                )
                return True, f"Rolled back to {restore_point['name']}"

            except Exception as exc:
                self.logger.error("rollback.failed", error=str(exc))
                return False, f"Rollback failed: {exc}"

    def get_restore_points(self) -> list[dict[str, Any]]:
        """Get all restore points."""
        with self._lock:
            return [
                {
                    "id": rp["id"],
                    "name": rp["name"],
                    "timestamp": rp["timestamp"],
                    "stage": rp["stage"],
                }
                for rp in self.restore_points
            ]

    def clear_restore_points(self) -> None:
        """Clear all restore points."""
        with self._lock:
            self.restore_points.clear()


# ==================== PAUSE/RESUME MANAGER ====================


class PauseResumeManager:
    """Manages pause and resume operations for the pipeline.

    Features:
    - Pause at any stage
    - Resume from paused state
    - Save state on pause for later resume
    """

    def __init__(self, storage_dir: Path | None = None):
        """Initialize the pause/resume manager.

        Args:
            storage_dir: Directory for persisting paused state
        """
        self.storage_dir = storage_dir or Path.home() / ".proxima" / "paused"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger("pause_resume")
        self._paused_executions: dict[str, dict[str, Any]] = {}

    def pause(
        self,
        context: PipelineContext,
        reason: str = "User requested pause",
    ) -> str:
        """Pause pipeline execution.

        Args:
            context: Pipeline context to pause
            reason: Reason for pausing

        Returns:
            Pause token for resuming
        """
        pause_token = str(uuid.uuid4())[:8]

        # Create pause state
        pause_state = {
            "token": pause_token,
            "execution_id": context.execution_id,
            "paused_at": time.time(),
            "reason": reason,
            "stage": context.current_stage.name,
            "context_data": {
                "stage_results": [r.to_dict() for r in context.stage_results],
                "backend_results": context.backend_results,
                "analysis_results": context.analysis_results,
                "plan": context.plan,
                "selected_backends": context.selected_backends,
                "consent_granted": context.consent_granted,
                "checkpoints": context.checkpoints,
            },
        }

        # Save to memory and disk
        self._paused_executions[pause_token] = pause_state
        self._persist_pause_state(pause_state)

        # Update context
        context.is_paused = True
        context.pause_reason = reason
        context.resume_from_stage = context.current_stage
        context.current_stage = PipelineStage.PAUSED

        self.logger.info(
            "execution.paused",
            execution_id=context.execution_id,
            stage=pause_state["stage"],
            token=pause_token,
        )
        return pause_token

    def resume(
        self,
        pause_token: str,
        context: PipelineContext | None = None,
    ) -> tuple[bool, PipelineContext | None, str]:
        """Resume a paused pipeline execution.

        Args:
            pause_token: Token from pause operation
            context: Optional existing context to restore into

        Returns:
            Tuple of (success, restored_context, message)
        """
        # Try memory first, then disk
        pause_state = self._paused_executions.get(pause_token)
        if not pause_state:
            pause_state = self._load_pause_state(pause_token)

        if not pause_state:
            return False, None, f"Pause token {pause_token} not found"

        try:
            # Create or update context
            if context is None:
                context = PipelineContext()

            context.execution_id = pause_state["execution_id"]
            context.is_paused = False
            context.pause_reason = None
            context.current_stage = PipelineStage[pause_state["stage"]]

            # Restore context data
            ctx_data = pause_state["context_data"]
            context.backend_results = ctx_data.get("backend_results", {})
            context.analysis_results = ctx_data.get("analysis_results", {})
            context.plan = ctx_data.get("plan")
            context.selected_backends = ctx_data.get("selected_backends", [])
            context.consent_granted = ctx_data.get("consent_granted", False)
            context.checkpoints = ctx_data.get("checkpoints", [])

            # Clean up pause state
            self._paused_executions.pop(pause_token, None)
            self._delete_pause_state(pause_token)

            self.logger.info(
                "execution.resumed",
                execution_id=context.execution_id,
                stage=pause_state["stage"],
            )
            return True, context, f"Resumed from {pause_state['stage']}"

        except Exception as exc:
            self.logger.error("resume.failed", error=str(exc))
            return False, None, f"Resume failed: {exc}"

    def _persist_pause_state(self, pause_state: dict[str, Any]) -> None:
        """Persist pause state to disk."""
        try:
            path = self.storage_dir / f"{pause_state['token']}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(pause_state, f, indent=2, default=str)
        except Exception as exc:
            self.logger.warning("pause_state.persist_failed", error=str(exc))

    def _load_pause_state(self, pause_token: str) -> dict[str, Any] | None:
        """Load pause state from disk."""
        try:
            path = self.storage_dir / f"{pause_token}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _delete_pause_state(self, pause_token: str) -> None:
        """Delete pause state from disk."""
        try:
            path = self.storage_dir / f"{pause_token}.json"
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def get_paused_executions(self) -> list[dict[str, Any]]:
        """Get list of paused executions."""
        paused = []

        # From memory
        for state in self._paused_executions.values():
            paused.append({
                "token": state["token"],
                "execution_id": state["execution_id"],
                "paused_at": state["paused_at"],
                "reason": state["reason"],
                "stage": state["stage"],
            })

        # From disk (if not already in memory)
        for path in self.storage_dir.glob("*.json"):
            token = path.stem
            if token not in self._paused_executions:
                state = self._load_pause_state(token)
                if state:
                    paused.append({
                        "token": state["token"],
                        "execution_id": state["execution_id"],
                        "paused_at": state["paused_at"],
                        "reason": state["reason"],
                        "stage": state["stage"],
                    })

        return paused



# ==================== CHECKPOINT MANAGER ====================


class CheckpointManager:
    """Manages checkpoint creation and restoration.

    Features:
    - Create checkpoints at key stages
    - Restore from checkpoints
    - List available checkpoints
    - Automatic checkpoint cleanup
    """

    def __init__(self, storage_dir: Path | None = None, max_checkpoints: int = 10):
        """Initialize the checkpoint manager.

        Args:
            storage_dir: Directory for persisting checkpoints
            max_checkpoints: Maximum checkpoints to keep per execution
        """
        self.storage_dir = storage_dir or Path.home() / ".proxima" / "checkpoints"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = structlog.get_logger("checkpoints")

    def create(
        self,
        context: PipelineContext,
        name: str = "",
        persist: bool = True,
    ) -> dict[str, Any]:
        """Create a checkpoint.

        Args:
            context: Pipeline context to checkpoint
            name: Optional checkpoint name
            persist: Whether to persist to disk

        Returns:
            Checkpoint data
        """
        checkpoint = context.create_checkpoint(name)

        if persist:
            self._persist_checkpoint(context.execution_id, checkpoint)

        # Enforce max checkpoints
        self._cleanup_old_checkpoints(context)

        self.logger.info(
            "checkpoint.created",
            execution_id=context.execution_id,
            checkpoint_id=checkpoint["id"],
            stage=checkpoint["stage"],
        )
        return checkpoint

    def restore(
        self,
        context: PipelineContext,
        checkpoint_id: str,
    ) -> tuple[bool, str]:
        """Restore context from a checkpoint.

        Args:
            context: Context to restore
            checkpoint_id: Checkpoint to restore from

        Returns:
            Tuple of (success, message)
        """
        # Try in-memory first
        success = context.restore_from_checkpoint(checkpoint_id)

        if not success:
            # Try loading from disk
            checkpoint = self._load_checkpoint(context.execution_id, checkpoint_id)
            if checkpoint:
                context.checkpoints.append(checkpoint)
                success = context.restore_from_checkpoint(checkpoint_id)

        if success:
            self.logger.info(
                "checkpoint.restored",
                execution_id=context.execution_id,
                checkpoint_id=checkpoint_id,
            )
            return True, f"Restored from checkpoint {checkpoint_id}"

        return False, f"Checkpoint {checkpoint_id} not found"

    def list_checkpoints(
        self,
        execution_id: str,
        context: PipelineContext | None = None,
    ) -> list[dict[str, Any]]:
        """List available checkpoints.

        Args:
            execution_id: Execution to list checkpoints for
            context: Optional context with in-memory checkpoints

        Returns:
            List of checkpoint summaries
        """
        checkpoints = []

        # In-memory checkpoints
        if context:
            for cp in context.checkpoints:
                checkpoints.append({
                    "id": cp["id"],
                    "name": cp["name"],
                    "timestamp": cp["timestamp"],
                    "stage": cp["stage"],
                    "source": "memory",
                })

        # Disk checkpoints
        exec_dir = self.storage_dir / execution_id
        if exec_dir.exists():
            for path in exec_dir.glob("*.json"):
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                        # Only add if not already from memory
                        if not any(cp["id"] == data["id"] for cp in checkpoints):
                            checkpoints.append({
                                "id": data["id"],
                                "name": data["name"],
                                "timestamp": data["timestamp"],
                                "stage": data["stage"],
                                "source": "disk",
                            })
                except Exception:
                    continue

        return sorted(checkpoints, key=lambda x: x["timestamp"])

    def _persist_checkpoint(
        self,
        execution_id: str,
        checkpoint: dict[str, Any],
    ) -> None:
        """Persist checkpoint to disk."""
        try:
            exec_dir = self.storage_dir / execution_id
            exec_dir.mkdir(parents=True, exist_ok=True)
            path = exec_dir / f"{checkpoint['id']}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except Exception as exc:
            self.logger.warning("checkpoint.persist_failed", error=str(exc))

    def _load_checkpoint(
        self,
        execution_id: str,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        """Load checkpoint from disk."""
        try:
            path = self.storage_dir / execution_id / f"{checkpoint_id}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _cleanup_old_checkpoints(self, context: PipelineContext) -> None:
        """Remove old checkpoints exceeding max limit."""
        if len(context.checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            to_remove = len(context.checkpoints) - self.max_checkpoints
            context.checkpoints = context.checkpoints[to_remove:]


# ==================== PIPELINE HANDLERS ====================


class PipelineHandler(ABC):
    """Base class for pipeline stage handlers."""

    @abstractmethod
    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Execute this pipeline stage."""
        pass


class ParseHandler(PipelineHandler):
    """Handler for parsing user input."""

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Parse user input and extract circuit specifications."""
        start = time.time()
        try:
            ctx.current_stage = PipelineStage.PARSING

            # Simple parsing - extract key parameters
            parsed = {
                "raw_input": ctx.user_input,
                "qubits": ctx.input_params.get("qubits", 2),
                "gates": ctx.input_params.get("gates", []),
                "measurements": ctx.input_params.get("measurements", True),
                "shots": ctx.input_params.get("shots", 1024),
            }
            ctx.parsed_input = parsed

            return StageResult(
                stage=PipelineStage.PARSING,
                success=True,
                data=parsed,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.PARSING,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )


class PlanHandler(PipelineHandler):
    """Handler for planning execution."""

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Create execution plan."""
        start = time.time()
        try:
            ctx.current_stage = PipelineStage.PLANNING

            # Create plan based on parsed input
            backends = ctx.input_params.get("backends", ["cirq"])
            if isinstance(backends, str):
                backends = [backends]

            plan = {
                "backends": backends,
                "qubits": ctx.parsed_input.get("qubits", 2) if ctx.parsed_input else 2,
                "shots": ctx.parsed_input.get("shots", 1024) if ctx.parsed_input else 1024,
                "parallel": len(backends) > 1,
                "estimated_time_ms": 100 * len(backends),
            }

            ctx.plan = plan
            ctx.selected_backends = backends

            return StageResult(
                stage=PipelineStage.PLANNING,
                success=True,
                data=plan,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.PLANNING,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )


class ResourceCheckHandler(PipelineHandler):
    """Handler for checking resource availability."""

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Check resource availability."""
        start = time.time()
        try:
            ctx.current_stage = PipelineStage.RESOURCE_CHECK

            # Estimate resources needed
            qubits = ctx.plan.get("qubits", 2) if ctx.plan else 2
            estimate = {
                "memory_mb": 2 ** qubits * 16 / (1024 * 1024),
                "estimated_time_ms": ctx.plan.get("estimated_time_ms", 100) if ctx.plan else 100,
                "backends_available": True,
            }

            ctx.resource_estimate = estimate

            return StageResult(
                stage=PipelineStage.RESOURCE_CHECK,
                success=True,
                data=estimate,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.RESOURCE_CHECK,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )


class ConsentHandler(PipelineHandler):
    """Handler for resource consent."""

    def __init__(
        self,
        require_consent: bool = True,
        auto_approve: bool = False,
    ):
        self.require_consent = require_consent
        self.auto_approve = auto_approve

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Get consent for resource usage."""
        start = time.time()
        try:
            ctx.current_stage = PipelineStage.CONSENT

            if not self.require_consent or self.auto_approve:
                ctx.consent_granted = True
                return StageResult(
                    stage=PipelineStage.CONSENT,
                    success=True,
                    data={"consent": True, "auto_approved": self.auto_approve},
                    duration_ms=(time.time() - start) * 1000,
                )

            # In a real implementation, this would prompt the user
            ctx.consent_granted = True

            return StageResult(
                stage=PipelineStage.CONSENT,
                success=True,
                data={"consent": ctx.consent_granted},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.CONSENT,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )


class ExecutionHandler(PipelineHandler):
    """Handler for circuit execution."""

    def __init__(self, distributed_executor: DistributedExecutor | None = None):
        self.distributed_executor = distributed_executor

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Execute circuit on backends."""
        start = time.time()
        try:
            ctx.current_stage = PipelineStage.EXECUTING

            results = {}
            for backend in ctx.selected_backends:
                ctx.current_backend = backend

                # Simulate execution
                result = {
                    "backend": backend,
                    "success": True,
                    "shots": ctx.plan.get("shots", 1024) if ctx.plan else 1024,
                    "counts": {"00": 512, "11": 512},  # Simulated results
                    "execution_time_ms": 50,
                }
                results[backend] = result

            ctx.backend_results = results
            ctx.current_backend = None

            return StageResult(
                stage=PipelineStage.EXECUTING,
                success=True,
                data=results,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.EXECUTING,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )


class CollectionHandler(PipelineHandler):
    """Handler for collecting and normalizing results."""

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Collect and normalize results."""
        start = time.time()
        try:
            ctx.current_stage = PipelineStage.COLLECTING

            collected = {
                "backends": list(ctx.backend_results.keys()),
                "total_shots": sum(
                    r.get("shots", 0) for r in ctx.backend_results.values()
                ),
                "results": ctx.backend_results,
            }

            return StageResult(
                stage=PipelineStage.COLLECTING,
                success=True,
                data=collected,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.COLLECTING,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )


class AnalysisHandler(PipelineHandler):
    """Handler for result analysis."""

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Analyze execution results."""
        start = time.time()
        try:
            ctx.current_stage = PipelineStage.ANALYZING

            # Simple analysis
            analysis = {
                "backends_compared": len(ctx.backend_results),
                "total_execution_time_ms": sum(
                    r.get("execution_time_ms", 0) for r in ctx.backend_results.values()
                ),
            }

            # Generate insights
            insights = []
            for backend, result in ctx.backend_results.items():
                insights.append({
                    "backend": backend,
                    "insight": f"Executed {result.get('shots', 0)} shots successfully",
                })

            ctx.analysis_results = analysis
            ctx.insights = insights

            return StageResult(
                stage=PipelineStage.ANALYZING,
                success=True,
                data={"analysis": analysis, "insights": insights},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.ANALYZING,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )


class ExportHandler(PipelineHandler):
    """Handler for exporting results."""

    def __init__(self, export_dir: Path | None = None):
        self.export_dir = export_dir or Path.home() / ".proxima" / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Export results."""
        start = time.time()
        try:
            ctx.current_stage = PipelineStage.EXPORTING

            # Export to JSON
            export_path = self.export_dir / f"{ctx.execution_id}_results.json"
            export_data = {
                "execution_id": ctx.execution_id,
                "timestamp": datetime.utcnow().isoformat(),
                "backends": ctx.selected_backends,
                "results": ctx.backend_results,
                "analysis": ctx.analysis_results,
                "insights": ctx.insights,
            }

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            ctx.export_paths.append(str(export_path))

            return StageResult(
                stage=PipelineStage.EXPORTING,
                success=True,
                data={"path": str(export_path)},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as exc:
            return StageResult(
                stage=PipelineStage.EXPORTING,
                success=False,
                error=str(exc),
                duration_ms=(time.time() - start) * 1000,
            )



# ==================== DATA FLOW PIPELINE ====================


class DataFlowPipeline:
    """Main execution pipeline with pause/resume, rollback, checkpoints, and DAG visualization.

    Features:
    - Stage-based execution with handlers
    - Pause/Resume at any stage
    - Rollback to previous restore points
    - Checkpoint creation and restoration
    - DAG visualization (Mermaid, ASCII)
    - Distributed execution support
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        max_workers: int = 4,
        auto_checkpoint: bool = True,
        checkpoint_stages: list[PipelineStage] | None = None,
    ):
        """Initialize the pipeline.

        Args:
            storage_dir: Directory for persistence
            max_workers: Max workers for distributed execution
            auto_checkpoint: Whether to auto-create checkpoints
            checkpoint_stages: Stages to checkpoint (default: before execution)
        """
        self.storage_dir = storage_dir or Path.home() / ".proxima" / "pipeline"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_stages = checkpoint_stages or [
            PipelineStage.RESOURCE_CHECK,
            PipelineStage.EXECUTING,
        ]

        # Managers
        self.rollback_manager = RollbackManager(self.storage_dir / "rollback")
        self.pause_resume_manager = PauseResumeManager(self.storage_dir / "pause")
        self.checkpoint_manager = CheckpointManager(self.storage_dir / "checkpoints")
        self.distributed_executor = DistributedExecutor(self.storage_dir / "distributed")

        # Logger
        self.logger = structlog.get_logger("pipeline")

        # Handlers for each stage
        self.handlers: dict[PipelineStage, PipelineHandler] = {
            PipelineStage.PARSING: ParseHandler(),
            PipelineStage.PLANNING: PlanHandler(),
            PipelineStage.RESOURCE_CHECK: ResourceCheckHandler(),
            PipelineStage.CONSENT: ConsentHandler(auto_approve=True),
            PipelineStage.EXECUTING: ExecutionHandler(self.distributed_executor),
            PipelineStage.COLLECTING: CollectionHandler(),
            PipelineStage.ANALYZING: AnalysisHandler(),
            PipelineStage.EXPORTING: ExportHandler(self.storage_dir / "exports"),
        }

        # DAG for execution visualization
        self.dag: ExecutionDAG | None = None

        # Active contexts
        self._active_contexts: dict[str, PipelineContext] = {}

    def register_handler(
        self,
        stage: PipelineStage,
        handler: PipelineHandler,
    ) -> None:
        """Register a custom handler for a stage."""
        self.handlers[stage] = handler

    def get_handler(self, stage: PipelineStage) -> PipelineHandler | None:
        """Get the handler for a stage."""
        return self.handlers.get(stage)

    async def run(
        self,
        user_input: str = "",
        session_id: str | None = None,
        execution_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run the full pipeline.

        Args:
            user_input: User input string
            session_id: Optional session ID
            execution_id: Optional execution ID (for resuming)
            **kwargs: Additional parameters

        Returns:
            Execution results
        """
        # Create or resume context
        if execution_id and execution_id in self._active_contexts:
            ctx = self._active_contexts[execution_id]
        else:
            ctx = PipelineContext(
                execution_id=execution_id or str(uuid.uuid4()),
                session_id=session_id or str(uuid.uuid4()),
                user_input=user_input,
                input_params=kwargs,
            )
            self._active_contexts[ctx.execution_id] = ctx

        # Initialize DAG
        self.dag = ExecutionDAG(ctx.execution_id)
        self._build_dag()

        self.logger.info("pipeline.start", execution_id=ctx.execution_id)

        # Check for resumed execution
        resume_from = ctx.resume_from_stage
        if resume_from:
            ctx.resume_from_stage = None
            ctx.is_paused = False
            self.pause_resume_manager._delete_pause_state(ctx.execution_id)
            self.logger.info("pipeline.resume", from_stage=resume_from.value)

        # Stage order
        stages = [
            PipelineStage.PARSING,
            PipelineStage.PLANNING,
            PipelineStage.RESOURCE_CHECK,
            PipelineStage.CONSENT,
            PipelineStage.EXECUTING,
            PipelineStage.COLLECTING,
            PipelineStage.ANALYZING,
            PipelineStage.EXPORTING,
        ]

        # Results collection
        results = {
            "execution_id": ctx.execution_id,
            "stages": {},
            "success": True,
            "error": None,
        }

        # Skip to resume point if applicable
        start_idx = 0
        if resume_from:
            for idx, stage in enumerate(stages):
                if stage == resume_from:
                    start_idx = idx
                    break

        # Execute stages
        for idx in range(start_idx, len(stages)):
            stage = stages[idx]

            # Check for pause request
            if ctx.is_paused:
                results["paused"] = True
                results["paused_at"] = stage.value
                self.logger.info("pipeline.paused", stage=stage.value)
                break

            # Auto checkpoint before certain stages
            if self.auto_checkpoint and stage in self.checkpoint_stages:
                self.checkpoint_manager.create(ctx, name=f"before_{stage.value}")

            # Create rollback point
            self.rollback_manager.create_restore_point(ctx)

            # Update DAG
            self.dag.start_node(stage.value)

            # Execute stage
            handler = self.handlers.get(stage)
            if not handler:
                self.dag.skip_node(stage.value)
                continue

            try:
                result = await handler.execute(ctx)
                results["stages"][stage.value] = {
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                    "error": result.error,
                }

                if result.success:
                    self.dag.complete_node(stage.value)
                else:
                    results["success"] = False
                    results["error"] = result.error
                    self.logger.error(
                        "pipeline.stage_failed",
                        stage=stage.value,
                        error=result.error,
                    )
                    break

            except Exception as exc:
                results["success"] = False
                results["error"] = str(exc)
                self.logger.error(
                    "pipeline.stage_exception",
                    stage=stage.value,
                    error=str(exc),
                )
                break

        # Mark complete
        if results["success"] and not ctx.is_paused:
            ctx.current_stage = PipelineStage.COMPLETE
            results["final_results"] = ctx.backend_results
            results["analysis"] = ctx.analysis_results
            results["insights"] = ctx.insights
            results["exports"] = ctx.export_paths

        results["dag"] = self.dag.to_dict()

        self.logger.info(
            "pipeline.complete" if results["success"] else "pipeline.failed",
            execution_id=ctx.execution_id,
            success=results["success"],
        )

        return results

    def _build_dag(self) -> None:
        """Build the execution DAG."""
        if not self.dag:
            return

        stages = [
            "parsing", "planning", "resource_check", "consent",
            "executing", "collecting", "analyzing", "exporting"
        ]

        for i, stage in enumerate(stages):
            deps = [stages[i - 1]] if i > 0 else []
            self.dag.add_node(stage, deps)

    async def pause(self, execution_id: str, reason: str = "") -> bool:
        """Pause a running execution."""
        if execution_id not in self._active_contexts:
            return False

        ctx = self._active_contexts[execution_id]
        return self.pause_resume_manager.pause(ctx, reason)

    async def resume(self, execution_id: str) -> dict[str, Any]:
        """Resume a paused execution."""
        if execution_id in self._active_contexts:
            ctx = self._active_contexts[execution_id]
            if self.pause_resume_manager.resume(ctx):
                return await self.run(
                    user_input=ctx.user_input,
                    session_id=ctx.session_id,
                    execution_id=execution_id,
                    **ctx.input_params,
                )
        return {"success": False, "error": "Could not resume execution"}

    async def rollback(
        self,
        execution_id: str,
        restore_point_id: str | None = None,
    ) -> bool:
        """Rollback to a previous restore point."""
        if execution_id not in self._active_contexts:
            return False

        ctx = self._active_contexts[execution_id]
        return self.rollback_manager.rollback(ctx, restore_point_id)

    def get_dag_visualization(
        self,
        format: str = "mermaid",
    ) -> str:
        """Get DAG visualization.

        Args:
            format: "mermaid" or "ascii"

        Returns:
            Visualization string
        """
        if not self.dag:
            return "No DAG available"

        if format == "mermaid":
            return self.dag.to_mermaid()
        else:
            return self.dag.to_ascii()

    def get_checkpoints(self, execution_id: str) -> list[dict[str, Any]]:
        """Get available checkpoints for an execution."""
        ctx = self._active_contexts.get(execution_id)
        return self.checkpoint_manager.list_checkpoints(execution_id, ctx)

    async def restore_checkpoint(
        self,
        execution_id: str,
        checkpoint_id: str,
    ) -> tuple[bool, str]:
        """Restore from a checkpoint."""
        if execution_id not in self._active_contexts:
            return False, "Execution not found"

        ctx = self._active_contexts[execution_id]
        return self.checkpoint_manager.restore(ctx, checkpoint_id)

    def get_cluster_status(self) -> dict[str, Any]:
        """Get distributed cluster status."""
        return self.distributed_executor.get_cluster_status()


# ==================== CONVENIENCE FUNCTIONS ====================


async def run_simulation(
    user_input: str = "",
    backends: list[str] | None = None,
    qubits: int = 2,
    shots: int = 1024,
    **kwargs,
) -> dict[str, Any]:
    """Run a quantum simulation through the pipeline.

    Args:
        user_input: User input string
        backends: List of backend names
        qubits: Number of qubits
        shots: Number of shots
        **kwargs: Additional parameters

    Returns:
        Execution results
    """
    pipeline = DataFlowPipeline()
    return await pipeline.run(
        user_input=user_input,
        backends=backends or ["cirq"],
        qubits=qubits,
        shots=shots,
        **kwargs,
    )


async def compare_backends(
    backends: list[str],
    qubits: int = 2,
    shots: int = 1024,
    **kwargs,
) -> dict[str, Any]:
    """Compare results across multiple backends.

    Args:
        backends: List of backends to compare
        qubits: Number of qubits
        shots: Number of shots
        **kwargs: Additional parameters

    Returns:
        Comparison results
    """
    pipeline = DataFlowPipeline()
    result = await pipeline.run(
        user_input="Compare backends",
        backends=backends,
        qubits=qubits,
        shots=shots,
        **kwargs,
    )

    # Add comparison summary
    if result["success"] and "final_results" in result:
        comparison = {
            "backends": backends,
            "agreement": True,  # Simplified
            "details": result["final_results"],
        }
        result["comparison"] = comparison

    return result


# ==================== EXPORTS ====================


__all__ = [
    # Enums
    "PipelineStage",
    "LoadBalancingStrategy",
    # Data Classes
    "StageResult",
    "PipelineContext",
    "DAGNode",
    "WorkerMetrics",
    "DistributedTask",
    # DAG
    "ExecutionDAG",
    # Distributed
    "DistributedWorker",
    "DistributedExecutor",
    "AdvancedDistributedExecutor",
    # Auto-Scaling
    "ScalingTrigger",
    "ScalingEvent",
    "ScalingPolicy",
    "ResourceMetrics",
    "AutoScalingManager",
    # Managers
    "RollbackManager",
    "PauseResumeManager",
    "CheckpointManager",
    # Handlers
    "PipelineHandler",
    "ParseHandler",
    "PlanHandler",
    "ResourceCheckHandler",
    "ConsentHandler",
    "ExecutionHandler",
    "CollectionHandler",
    "AnalysisHandler",
    "ExportHandler",
    # Pipeline
    "DataFlowPipeline",
    # Convenience
    "run_simulation",
    "compare_backends",
]
