"""Plan Execution Engine for Natural Language Planning.

Phase 6: Natural Language Planning & Execution

Provides plan execution capabilities including:
- Dependency-ordered execution
- Parallel step execution
- Progress tracking and reporting
- Error handling and recovery
- Plan adaptation on failure
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from proxima.utils.logging import get_logger

from .task_planner import (
    ExecutionPlan,
    PlanStep,
    PlanStatus,
    StepStatus,
    TaskCategory,
)

logger = get_logger("agent.plan_executor")


class ExecutionMode(Enum):
    """Modes of plan execution."""
    SEQUENTIAL = "sequential"  # One step at a time
    PARALLEL = "parallel"      # Independent steps in parallel
    DRY_RUN = "dry_run"        # Preview only, no actual execution


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_id: int
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    output: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "output": self.output,
        }


@dataclass
class ExecutionResult:
    """Result of executing an entire plan."""
    plan_id: str
    success: bool
    steps_completed: int
    steps_failed: int
    steps_skipped: int
    total_duration_seconds: float
    step_results: List[StepResult] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "steps_skipped": self.steps_skipped,
            "total_duration_seconds": self.total_duration_seconds,
            "step_results": [r.to_dict() for r in self.step_results],
            "error": self.error,
        }
    
    @property
    def summary(self) -> str:
        """Get a summary of the execution."""
        status = "✅ Success" if self.success else "❌ Failed"
        return (
            f"{status} | "
            f"Completed: {self.steps_completed} | "
            f"Failed: {self.steps_failed} | "
            f"Skipped: {self.steps_skipped} | "
            f"Duration: {self.total_duration_seconds:.1f}s"
        )


# Callback types
ProgressCallback = Callable[[ExecutionPlan, PlanStep, float], None]
StepStartCallback = Callable[[PlanStep], None]
StepCompleteCallback = Callable[[PlanStep, StepResult], None]
PlanCompleteCallback = Callable[[ExecutionPlan, ExecutionResult], None]


@dataclass
class ExecutorCallbacks:
    """Callbacks for plan execution events."""
    on_progress: Optional[ProgressCallback] = None
    on_step_start: Optional[StepStartCallback] = None
    on_step_complete: Optional[StepCompleteCallback] = None
    on_plan_complete: Optional[PlanCompleteCallback] = None
    on_error: Optional[Callable[[PlanStep, Exception], None]] = None


class ToolExecutor:
    """Execute individual tools.
    
    This is the bridge between the plan executor and actual tool implementations.
    Subclass this to connect to your tool registry.
    """
    
    def __init__(self, tools: Optional[Dict[str, Callable]] = None):
        """Initialize with optional tool mapping."""
        self._tools = tools or {}
    
    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool function."""
        self._tools[name] = func
    
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Any, str]:
        """Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tuple of (success, result, output_or_error)
        """
        if tool_name not in self._tools:
            return False, None, f"Unknown tool: {tool_name}"
        
        try:
            func = self._tools[tool_name]
            
            # Check if async
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)
            
            # Normalize result
            if isinstance(result, dict):
                success = result.get("success", True)
                output = result.get("output", str(result))
                return success, result, output
            
            return True, result, str(result) if result else "Success"
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return False, None, str(e)
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools


class PlanExecutor:
    """Execute task plans.
    
    Features:
    - Dependency-ordered execution
    - Parallel execution of independent steps
    - Progress tracking
    - Error handling and retry
    - Plan adaptation
    
    Example:
        >>> executor = PlanExecutor(tool_executor)
        >>> 
        >>> # Execute a plan
        >>> result = await executor.execute(plan)
        >>> 
        >>> # Execute with callbacks
        >>> result = await executor.execute(
        ...     plan,
        ...     callbacks=ExecutorCallbacks(
        ...         on_progress=lambda p, s, pct: print(f"{pct:.0f}%"),
        ...     ),
        ... )
    """
    
    def __init__(
        self,
        tool_executor: Optional[ToolExecutor] = None,
        mode: ExecutionMode = ExecutionMode.PARALLEL,
        max_parallel: int = 4,
        retry_failed: bool = True,
        stop_on_failure: bool = False,
    ):
        """Initialize the plan executor.
        
        Args:
            tool_executor: Executor for individual tools
            mode: Execution mode (sequential, parallel, dry_run)
            max_parallel: Maximum concurrent steps
            retry_failed: Whether to retry failed steps
            stop_on_failure: Stop execution on first failure
        """
        self.tool_executor = tool_executor or ToolExecutor()
        self.mode = mode
        self.max_parallel = max_parallel
        self.retry_failed = retry_failed
        self.stop_on_failure = stop_on_failure
        
        # Execution state
        self._current_plan: Optional[ExecutionPlan] = None
        self._running = False
        self._paused = False
        self._cancelled = False
        
        # Semaphore for parallel execution limit
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        logger.info(f"PlanExecutor initialized with mode={mode.value}")
    
    async def execute(
        self,
        plan: ExecutionPlan,
        callbacks: Optional[ExecutorCallbacks] = None,
    ) -> ExecutionResult:
        """Execute a plan.
        
        Args:
            plan: The plan to execute
            callbacks: Optional callbacks for progress reporting
            
        Returns:
            ExecutionResult with details of the execution
        """
        self._current_plan = plan
        self._running = True
        self._paused = False
        self._cancelled = False
        self._semaphore = asyncio.Semaphore(self.max_parallel)
        
        callbacks = callbacks or ExecutorCallbacks()
        start_time = time.time()
        
        # Update plan status
        plan.status = PlanStatus.EXECUTING
        
        step_results: List[StepResult] = []
        
        try:
            if self.mode == ExecutionMode.DRY_RUN:
                step_results = await self._dry_run(plan, callbacks)
            elif self.mode == ExecutionMode.SEQUENTIAL:
                step_results = await self._execute_sequential(plan, callbacks)
            else:
                step_results = await self._execute_parallel(plan, callbacks)
            
        except asyncio.CancelledError:
            logger.info("Plan execution cancelled")
            plan.status = PlanStatus.CANCELLED
        except Exception as e:
            logger.error(f"Plan execution error: {e}")
            plan.status = PlanStatus.FAILED
        finally:
            self._running = False
        
        # Calculate statistics
        total_duration = time.time() - start_time
        completed = sum(1 for r in step_results if r.success)
        failed = sum(1 for r in step_results if not r.success)
        skipped = len(plan.steps) - len(step_results)
        
        # Determine overall success
        success = failed == 0 and not self._cancelled
        
        # Update plan status
        if self._cancelled:
            plan.status = PlanStatus.CANCELLED
        elif failed > 0:
            plan.status = PlanStatus.FAILED
        else:
            plan.status = PlanStatus.COMPLETED
        
        result = ExecutionResult(
            plan_id=plan.plan_id,
            success=success,
            steps_completed=completed,
            steps_failed=failed,
            steps_skipped=skipped,
            total_duration_seconds=total_duration,
            step_results=step_results,
        )
        
        # Notify completion
        if callbacks.on_plan_complete:
            callbacks.on_plan_complete(plan, result)
        
        logger.info(f"Plan execution complete: {result.summary}")
        
        return result
    
    async def _execute_sequential(
        self,
        plan: ExecutionPlan,
        callbacks: ExecutorCallbacks,
    ) -> List[StepResult]:
        """Execute steps sequentially."""
        results: List[StepResult] = []
        completed_ids: Set[int] = set()
        
        for group in plan.parallel_groups:
            for step_id in group:
                if self._cancelled:
                    break
                
                while self._paused:
                    await asyncio.sleep(0.1)
                
                step = plan.get_step(step_id)
                if not step:
                    continue
                
                result = await self._execute_step(step, callbacks)
                results.append(result)
                
                if result.success:
                    completed_ids.add(step_id)
                elif self.stop_on_failure:
                    break
        
        return results
    
    async def _execute_parallel(
        self,
        plan: ExecutionPlan,
        callbacks: ExecutorCallbacks,
    ) -> List[StepResult]:
        """Execute independent steps in parallel."""
        results: List[StepResult] = []
        completed_ids: Set[int] = set()
        
        for group in plan.parallel_groups:
            if self._cancelled:
                break
            
            while self._paused:
                await asyncio.sleep(0.1)
            
            # Execute group in parallel
            group_tasks = []
            for step_id in group:
                step = plan.get_step(step_id)
                if step:
                    task = self._execute_step_with_semaphore(step, callbacks)
                    group_tasks.append(task)
            
            # Wait for all in group to complete
            group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
            
            for result in group_results:
                if isinstance(result, Exception):
                    # Handle exception as failed step
                    results.append(StepResult(
                        step_id=-1,
                        success=False,
                        error=str(result),
                    ))
                else:
                    results.append(result)
                    if result.success:
                        completed_ids.add(result.step_id)
            
            # Check if we should stop
            if self.stop_on_failure:
                if any(not r.success for r in group_results if isinstance(r, StepResult)):
                    break
        
        return results
    
    async def _execute_step_with_semaphore(
        self,
        step: PlanStep,
        callbacks: ExecutorCallbacks,
    ) -> StepResult:
        """Execute a step with semaphore for concurrency limit."""
        async with self._semaphore:
            return await self._execute_step(step, callbacks)
    
    async def _execute_step(
        self,
        step: PlanStep,
        callbacks: ExecutorCallbacks,
    ) -> StepResult:
        """Execute a single step."""
        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.now()
        
        # Notify step start
        if callbacks.on_step_start:
            callbacks.on_step_start(step)
        
        start_time = time.time()
        
        try:
            # Execute the tool
            success, result, output = await self.tool_executor.execute(
                step.tool,
                step.arguments,
            )
            
            duration = time.time() - start_time
            
            step_result = StepResult(
                step_id=step.step_id,
                success=success,
                result=result,
                error=None if success else output,
                duration_seconds=duration,
                output=output if success else "",
            )
            
            # Update step status
            step.status = StepStatus.COMPLETED if success else StepStatus.FAILED
            step.result = result
            step.error = None if success else output
            
        except Exception as e:
            duration = time.time() - start_time
            
            step_result = StepResult(
                step_id=step.step_id,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            
            step.status = StepStatus.FAILED
            step.error = str(e)
            
            # Notify error
            if callbacks.on_error:
                callbacks.on_error(step, e)
        
        step.completed_at = datetime.now()
        
        # Retry if enabled and failed
        if not step_result.success and self.retry_failed and step.can_retry:
            step.retries += 1
            logger.info(f"Retrying step {step.step_id} (attempt {step.retries})")
            return await self._execute_step(step, callbacks)
        
        # Notify step complete
        if callbacks.on_step_complete:
            callbacks.on_step_complete(step, step_result)
        
        # Update progress
        if callbacks.on_progress:
            progress = self._calculate_progress()
            callbacks.on_progress(self._current_plan, step, progress)
        
        return step_result
    
    async def _dry_run(
        self,
        plan: ExecutionPlan,
        callbacks: ExecutorCallbacks,
    ) -> List[StepResult]:
        """Simulate execution without actually running tools."""
        results: List[StepResult] = []
        
        for step in plan.steps:
            # Notify step start
            if callbacks.on_step_start:
                callbacks.on_step_start(step)
            
            # Simulate execution time
            await asyncio.sleep(0.1)
            
            result = StepResult(
                step_id=step.step_id,
                success=True,
                result={"dry_run": True},
                output=f"[DRY RUN] Would execute: {step.tool}({step.arguments})",
                duration_seconds=0.1,
            )
            
            results.append(result)
            
            step.status = StepStatus.COMPLETED
            step.result = result.result
            
            # Notify step complete
            if callbacks.on_step_complete:
                callbacks.on_step_complete(step, result)
            
            # Update progress
            if callbacks.on_progress:
                progress = (len(results) / len(plan.steps)) * 100
                callbacks.on_progress(plan, step, progress)
        
        return results
    
    def _calculate_progress(self) -> float:
        """Calculate current execution progress."""
        if not self._current_plan:
            return 0.0
        return self._current_plan.progress
    
    def pause(self) -> None:
        """Pause execution."""
        self._paused = True
        if self._current_plan:
            self._current_plan.status = PlanStatus.PAUSED
        logger.info("Plan execution paused")
    
    def resume(self) -> None:
        """Resume execution."""
        self._paused = False
        if self._current_plan:
            self._current_plan.status = PlanStatus.EXECUTING
        logger.info("Plan execution resumed")
    
    def cancel(self) -> None:
        """Cancel execution."""
        self._cancelled = True
        if self._current_plan:
            self._current_plan.status = PlanStatus.CANCELLED
        logger.info("Plan execution cancelled")
    
    @property
    def is_running(self) -> bool:
        """Check if execution is in progress."""
        return self._running
    
    @property
    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self._paused
    
    @property
    def current_plan(self) -> Optional[ExecutionPlan]:
        """Get the currently executing plan."""
        return self._current_plan


class AdaptivePlanExecutor(PlanExecutor):
    """Plan executor with adaptive capabilities.
    
    Can modify plans based on execution results and ask LLM
    for revised plans when steps fail.
    """
    
    def __init__(
        self,
        tool_executor: Optional[ToolExecutor] = None,
        llm_provider: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize with optional LLM for plan adaptation."""
        super().__init__(tool_executor, **kwargs)
        self.llm_provider = llm_provider
    
    async def _execute_step(
        self,
        step: PlanStep,
        callbacks: ExecutorCallbacks,
    ) -> StepResult:
        """Execute step with adaptation on failure."""
        result = await super()._execute_step(step, callbacks)
        
        # If failed and we have LLM, try to adapt
        if not result.success and self.llm_provider and not step.can_retry:
            adapted_step = await self._adapt_step(step, result)
            if adapted_step:
                logger.info(f"Adapted step {step.step_id}, retrying with new approach")
                # Replace step and retry
                step.tool = adapted_step.tool
                step.arguments = adapted_step.arguments
                step.retries = 0
                return await super()._execute_step(step, callbacks)
        
        return result
    
    async def _adapt_step(
        self,
        step: PlanStep,
        result: StepResult,
    ) -> Optional[PlanStep]:
        """Use LLM to adapt a failed step."""
        if not self.llm_provider:
            return None
        
        prompt = f"""A step in an execution plan failed. Suggest an alternative approach.

Failed step:
- Tool: {step.tool}
- Arguments: {step.arguments}
- Error: {result.error}

Suggest an alternative step that could achieve the same goal.
Respond with JSON:
{{
  "tool": "alternative_tool_name",
  "arguments": {{"arg1": "value1"}},
  "description": "What this alternative does"
}}

If no alternative is possible, respond with: {{"no_alternative": true}}
"""
        
        try:
            import json
            response = await self.llm_provider.complete(prompt)
            data = json.loads(response)
            
            if data.get("no_alternative"):
                return None
            
            return PlanStep(
                step_id=step.step_id,
                tool=data["tool"],
                arguments=data.get("arguments", {}),
                description=data.get("description", "Adapted step"),
            )
        except Exception as e:
            logger.warning(f"Step adaptation failed: {e}")
            return None


# Global instance
_executor: Optional[PlanExecutor] = None


def get_plan_executor() -> PlanExecutor:
    """Get the global PlanExecutor instance."""
    global _executor
    if _executor is None:
        _executor = PlanExecutor()
    return _executor


async def execute_plan(
    plan: ExecutionPlan,
    callbacks: Optional[ExecutorCallbacks] = None,
) -> ExecutionResult:
    """Convenience function to execute a plan."""
    return await get_plan_executor().execute(plan, callbacks)
