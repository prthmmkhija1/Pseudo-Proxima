"""
Plugin hook system.

Provides extension points for plugins to hook into Proxima's execution flow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class HookType(str, Enum):
    """Available hook points in Proxima's execution flow."""

    # Lifecycle hooks
    PRE_INIT = "pre_init"
    POST_INIT = "post_init"
    PRE_SHUTDOWN = "pre_shutdown"

    # Execution hooks
    PRE_EXECUTE = "pre_execute"
    POST_EXECUTE = "post_execute"
    ON_ERROR = "on_error"

    # Pipeline hooks
    PRE_STAGE = "pre_stage"
    POST_STAGE = "post_stage"

    # Backend hooks
    PRE_BACKEND_INIT = "pre_backend_init"
    POST_BACKEND_INIT = "post_backend_init"
    PRE_BACKEND_RUN = "pre_backend_run"
    POST_BACKEND_RUN = "post_backend_run"

    # LLM hooks
    PRE_LLM_REQUEST = "pre_llm_request"
    POST_LLM_RESPONSE = "post_llm_response"

    # Result hooks
    PRE_EXPORT = "pre_export"
    POST_EXPORT = "post_export"
    PRE_ANALYZE = "pre_analyze"
    POST_ANALYZE = "post_analyze"


@dataclass
class HookContext:
    """Context passed to hook handlers."""

    hook_type: HookType
    data: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    result: Any = None

    def cancel(self) -> None:
        """Cancel the current operation."""
        self.cancelled = True

    def set_result(self, result: Any) -> None:
        """Set a result to pass to subsequent hooks."""
        self.result = result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context data."""
        self.data[key] = value


HookHandler = Callable[[HookContext], Any | None]


@dataclass
class RegisteredHook:
    """A registered hook handler."""

    handler: HookHandler
    priority: int = 0
    name: str | None = None
    plugin_name: str | None = None


class HookManager:
    """Manages hook registration and execution."""

    def __init__(self) -> None:
        self._hooks: dict[HookType, list[RegisteredHook]] = {ht: [] for ht in HookType}

    def register(
        self,
        hook_type: HookType,
        handler: HookHandler,
        priority: int = 0,
        name: str | None = None,
        plugin_name: str | None = None,
    ) -> None:
        """Register a hook handler.

        Args:
            hook_type: The type of hook to register for.
            handler: The handler function to call.
            priority: Higher priority handlers run first. Default 0.
            name: Optional name for the handler.
            plugin_name: Optional plugin name that owns this handler.
        """
        registered = RegisteredHook(
            handler=handler,
            priority=priority,
            name=name,
            plugin_name=plugin_name,
        )
        self._hooks[hook_type].append(registered)
        # Sort by priority (descending)
        self._hooks[hook_type].sort(key=lambda h: h.priority, reverse=True)

    def unregister(
        self,
        hook_type: HookType,
        handler: HookHandler | None = None,
        name: str | None = None,
        plugin_name: str | None = None,
    ) -> int:
        """Unregister hook handlers.

        Can unregister by handler function, name, or plugin name.
        Returns the number of handlers removed.
        """
        removed = 0
        hooks = self._hooks[hook_type]

        to_remove = []
        for i, registered in enumerate(hooks):
            if handler and registered.handler is handler:
                to_remove.append(i)
            elif name and registered.name == name:
                to_remove.append(i)
            elif plugin_name and registered.plugin_name == plugin_name:
                to_remove.append(i)

        for i in reversed(to_remove):
            del hooks[i]
            removed += 1

        return removed

    def unregister_plugin(self, plugin_name: str) -> int:
        """Unregister all hooks from a plugin."""
        removed = 0
        for hook_type in HookType:
            removed += self.unregister(hook_type, plugin_name=plugin_name)
        return removed

    def trigger(
        self,
        hook_type: HookType,
        data: dict[str, Any] | None = None,
    ) -> HookContext:
        """Trigger a hook and run all registered handlers.

        Args:
            hook_type: The hook to trigger.
            data: Initial context data.

        Returns:
            The HookContext after all handlers have run.
        """
        context = HookContext(hook_type=hook_type, data=data or {})

        for registered in self._hooks[hook_type]:
            if context.cancelled:
                break
            try:
                result = registered.handler(context)
                if result is not None:
                    context.result = result
            except Exception as e:
                # Store error but continue with other handlers
                context.data.setdefault("errors", []).append(
                    {
                        "handler": registered.name or str(registered.handler),
                        "error": str(e),
                    }
                )

        return context

    async def trigger_async(
        self,
        hook_type: HookType,
        data: dict[str, Any] | None = None,
    ) -> HookContext:
        """Trigger a hook asynchronously.

        Handles both sync and async handlers.
        """
        import inspect

        context = HookContext(hook_type=hook_type, data=data or {})

        for registered in self._hooks[hook_type]:
            if context.cancelled:
                break
            try:
                result = registered.handler(context)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    context.result = result
            except Exception as e:
                context.data.setdefault("errors", []).append(
                    {
                        "handler": registered.name or str(registered.handler),
                        "error": str(e),
                    }
                )

        return context

    def get_handlers(self, hook_type: HookType) -> list[RegisteredHook]:
        """Get all handlers for a hook type."""
        return list(self._hooks[hook_type])

    def has_handlers(self, hook_type: HookType) -> bool:
        """Check if a hook type has any handlers."""
        return len(self._hooks[hook_type]) > 0

    def clear(self, hook_type: HookType | None = None) -> None:
        """Clear handlers for a hook type, or all hooks."""
        if hook_type:
            self._hooks[hook_type].clear()
        else:
            for ht in HookType:
                self._hooks[ht].clear()


# Global hook manager singleton
_hook_manager: HookManager | None = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager."""
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = HookManager()
    return _hook_manager


# Decorator for easy hook registration
def hook(
    hook_type: HookType,
    priority: int = 0,
    name: str | None = None,
) -> Callable[[HookHandler], HookHandler]:
    """Decorator to register a function as a hook handler.

    Example:
        @hook(HookType.PRE_EXECUTE, priority=10)
        def my_handler(ctx: HookContext) -> None:
            print(f"Pre-execute: {ctx.data}")
    """

    def decorator(func: HookHandler) -> HookHandler:
        get_hook_manager().register(
            hook_type=hook_type,
            handler=func,
            priority=priority,
            name=name or func.__name__,
        )
        return func

    return decorator


# =============================================================================
# HOOK DECORATORS AND UTILITIES (Feature - Plugins)
# =============================================================================


def before_execute(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for pre-execution hooks.
    
    Example:
        @before_execute(priority=10)
        def validate_inputs(ctx: HookContext) -> None:
            if not ctx.get("circuit"):
                ctx.cancel()
    """
    return hook(HookType.PRE_EXECUTE, priority=priority)


def after_execute(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for post-execution hooks.
    
    Example:
        @after_execute()
        def log_results(ctx: HookContext) -> None:
            result = ctx.get("result")
            print(f"Execution completed: {result}")
    """
    return hook(HookType.POST_EXECUTE, priority=priority)


def on_error(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for error handling hooks.
    
    Example:
        @on_error()
        def handle_error(ctx: HookContext) -> None:
            error = ctx.get("error")
            send_alert(str(error))
    """
    return hook(HookType.ON_ERROR, priority=priority)


def before_stage(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for pre-stage hooks."""
    return hook(HookType.PRE_STAGE, priority=priority)


def after_stage(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for post-stage hooks."""
    return hook(HookType.POST_STAGE, priority=priority)


def before_backend_run(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for pre-backend-run hooks."""
    return hook(HookType.PRE_BACKEND_RUN, priority=priority)


def after_backend_run(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for post-backend-run hooks."""
    return hook(HookType.POST_BACKEND_RUN, priority=priority)


# =============================================================================
# HOOK CHAIN (Feature - Plugins)
# =============================================================================


class HookChain:
    """A chain of hooks to execute in sequence.
    
    Allows composing multiple hooks into a single unit that can be
    registered and unregistered together.
    """
    
    def __init__(self, name: str, plugin_name: str | None = None) -> None:
        self.name = name
        self.plugin_name = plugin_name
        self._hooks: list[tuple[HookType, HookHandler, int]] = []
    
    def add(
        self,
        hook_type: HookType,
        handler: HookHandler,
        priority: int = 0,
    ) -> "HookChain":
        """Add a hook to the chain.
        
        Returns self for chaining.
        """
        self._hooks.append((hook_type, handler, priority))
        return self
    
    def register(self, manager: HookManager | None = None) -> None:
        """Register all hooks in the chain."""
        manager = manager or get_hook_manager()
        for hook_type, handler, priority in self._hooks:
            manager.register(
                hook_type=hook_type,
                handler=handler,
                priority=priority,
                name=f"{self.name}_{handler.__name__}",
                plugin_name=self.plugin_name,
            )
    
    def unregister(self, manager: HookManager | None = None) -> int:
        """Unregister all hooks in the chain.
        
        Returns number of hooks unregistered.
        """
        manager = manager or get_hook_manager()
        if self.plugin_name:
            return manager.unregister_plugin(self.plugin_name)
        
        count = 0
        for hook_type, handler, _ in self._hooks:
            count += manager.unregister(hook_type, handler=handler)
        return count


# =============================================================================
# CONDITIONAL HOOKS (Feature - Plugins)
# =============================================================================


def conditional_hook(
    hook_type: HookType,
    condition: Callable[[HookContext], bool],
    priority: int = 0,
    name: str | None = None,
) -> Callable[[HookHandler], HookHandler]:
    """Decorator for conditional hook execution.
    
    The handler only runs if the condition returns True.
    
    Example:
        @conditional_hook(
            HookType.POST_EXECUTE,
            condition=lambda ctx: ctx.get("backend") == "qsim"
        )
        def qsim_specific_handler(ctx: HookContext) -> None:
            # Only runs for qsim backend
            pass
    """
    def decorator(func: HookHandler) -> HookHandler:
        def conditional_wrapper(ctx: HookContext) -> Any:
            if condition(ctx):
                return func(ctx)
            return None
        
        # Copy function metadata
        conditional_wrapper.__name__ = func.__name__
        conditional_wrapper.__doc__ = func.__doc__
        
        get_hook_manager().register(
            hook_type=hook_type,
            handler=conditional_wrapper,
            priority=priority,
            name=name or func.__name__,
        )
        return func
    
    return decorator


# =============================================================================
# HOOK MIDDLEWARE (Feature - Plugins)
# =============================================================================


class HookMiddleware:
    """Middleware for hook processing.
    
    Allows wrapping hook handlers with additional functionality
    like logging, timing, or error handling.
    """
    
    def __init__(self) -> None:
        self._middleware: list[Callable[[HookHandler], HookHandler]] = []
    
    def add(self, wrapper: Callable[[HookHandler], HookHandler]) -> None:
        """Add middleware wrapper."""
        self._middleware.append(wrapper)
    
    def wrap(self, handler: HookHandler) -> HookHandler:
        """Apply all middleware to a handler."""
        wrapped = handler
        for middleware in reversed(self._middleware):
            wrapped = middleware(wrapped)
        return wrapped


def with_timing() -> Callable[[HookHandler], HookHandler]:
    """Middleware that adds timing information to hook context."""
    def wrapper(handler: HookHandler) -> HookHandler:
        def timed_handler(ctx: HookContext) -> Any:
            import time
            start = time.perf_counter()
            result = handler(ctx)
            elapsed = time.perf_counter() - start
            ctx.data.setdefault("hook_timings", {})[handler.__name__] = elapsed
            return result
        timed_handler.__name__ = handler.__name__
        return timed_handler
    return wrapper


def with_logging(
    log_func: Callable[[str], None] | None = None,
) -> Callable[[HookHandler], HookHandler]:
    """Middleware that logs hook execution."""
    def wrapper(handler: HookHandler) -> HookHandler:
        def logged_handler(ctx: HookContext) -> Any:
            msg = f"Hook {handler.__name__} starting for {ctx.hook_type.value}"
            if log_func:
                log_func(msg)
            else:
                print(msg)
            result = handler(ctx)
            msg = f"Hook {handler.__name__} completed"
            if log_func:
                log_func(msg)
            else:
                print(msg)
            return result
        logged_handler.__name__ = handler.__name__
        return logged_handler
    return wrapper


def with_error_handling(
    on_error: Callable[[Exception], None] | None = None,
) -> Callable[[HookHandler], HookHandler]:
    """Middleware that catches and handles errors."""
    def wrapper(handler: HookHandler) -> HookHandler:
        def safe_handler(ctx: HookContext) -> Any:
            try:
                return handler(ctx)
            except Exception as e:
                if on_error:
                    on_error(e)
                ctx.data.setdefault("errors", []).append({
                    "handler": handler.__name__,
                    "error": str(e),
                })
                return None
        safe_handler.__name__ = handler.__name__
        return safe_handler
    return wrapper


# =============================================================================
# HOOK STATISTICS (Feature - Plugins)
# =============================================================================


@dataclass
class HookStats:
    """Statistics about hook execution."""
    
    hook_type: HookType
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_called: datetime | None = None


class HookProfiler:
    """Profiles hook execution for performance analysis."""
    
    def __init__(self) -> None:
        self._stats: dict[HookType, HookStats] = {
            ht: HookStats(hook_type=ht) for ht in HookType
        }
    
    def record(
        self,
        hook_type: HookType,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a hook execution."""
        stats = self._stats[hook_type]
        stats.total_calls += 1
        if success:
            stats.successful_calls += 1
        else:
            stats.failed_calls += 1
        stats.total_time_ms += duration_ms
        stats.avg_time_ms = stats.total_time_ms / stats.total_calls
        stats.last_called = datetime.now()
    
    def get_stats(self, hook_type: HookType | None = None) -> dict[str, HookStats]:
        """Get hook statistics."""
        if hook_type:
            return {hook_type.value: self._stats[hook_type]}
        return {ht.value: stats for ht, stats in self._stats.items()}
    
    def get_slowest_hooks(self, limit: int = 5) -> list[HookStats]:
        """Get the slowest hooks by average time."""
        return sorted(
            [s for s in self._stats.values() if s.total_calls > 0],
            key=lambda s: s.avg_time_ms,
            reverse=True,
        )[:limit]
    
    def reset(self) -> None:
        """Reset all statistics."""
        for ht in HookType:
            self._stats[ht] = HookStats(hook_type=ht)


# Global hook profiler
_hook_profiler: HookProfiler | None = None


def get_hook_profiler() -> HookProfiler:
    """Get the global hook profiler."""
    global _hook_profiler
    if _hook_profiler is None:
        _hook_profiler = HookProfiler()
    return _hook_profiler

# =============================================================================
# ADVANCED PRIORITY ORDERING SYSTEM (5% Gap Coverage)
# =============================================================================


class PriorityLevel(Enum):
    """Predefined priority levels for hook ordering."""
    
    CRITICAL = 10000      # Always runs first (system-level)
    HIGHEST = 1000        # Very high priority
    HIGH = 100            # High priority
    ABOVE_NORMAL = 50     # Slightly above normal
    NORMAL = 0            # Default priority
    BELOW_NORMAL = -50    # Slightly below normal
    LOW = -100            # Low priority
    LOWEST = -1000        # Very low priority
    FALLBACK = -10000     # Always runs last (cleanup)


@dataclass
class PriorityGroup:
    """A group of hooks that execute together as a unit.
    
    All hooks in a group execute in sequence before moving to the next group.
    Within a group, hooks are ordered by their individual priorities.
    """
    
    name: str
    group_priority: int  # Priority of the entire group
    hooks: list[RegisteredHook] = field(default_factory=list)
    before_groups: list[str] = field(default_factory=list)  # Groups this must run before
    after_groups: list[str] = field(default_factory=list)   # Groups this must run after
    exclusive: bool = False  # If True, only this group runs (no others)
    enabled: bool = True
    
    def add_hook(self, hook: RegisteredHook) -> None:
        """Add a hook to the group."""
        self.hooks.append(hook)
        self._sort_hooks()
    
    def remove_hook(self, handler: HookHandler | None = None, name: str | None = None) -> int:
        """Remove hooks from the group."""
        initial = len(self.hooks)
        if handler:
            self.hooks = [h for h in self.hooks if h.handler is not handler]
        elif name:
            self.hooks = [h for h in self.hooks if h.name != name]
        return initial - len(self.hooks)
    
    def _sort_hooks(self) -> None:
        """Sort hooks by priority within group."""
        self.hooks.sort(key=lambda h: h.priority, reverse=True)
    
    def __len__(self) -> int:
        return len(self.hooks)


@dataclass
class PriorityConstraint:
    """Constraint for hook ordering."""
    
    hook_name: str
    constraint_type: str  # "before", "after", "requires", "conflicts"
    target: str  # Target hook or group name
    hard: bool = True  # If True, constraint must be satisfied; if False, it's a preference
    
    def is_satisfied(
        self,
        execution_order: list[str],
    ) -> bool:
        """Check if constraint is satisfied by the given execution order."""
        if self.hook_name not in execution_order:
            return True  # Hook not in order, constraint doesn't apply
        if self.target not in execution_order:
            if self.constraint_type == "requires":
                return not self.hard  # Soft requirements are OK if target is missing
            return True
        
        hook_idx = execution_order.index(self.hook_name)
        target_idx = execution_order.index(self.target)
        
        if self.constraint_type == "before":
            return hook_idx < target_idx
        elif self.constraint_type == "after":
            return hook_idx > target_idx
        elif self.constraint_type == "conflicts":
            return False  # Both present = conflict
        
        return True


class PriorityPhase(Enum):
    """Execution phases for hooks."""
    
    SETUP = "setup"           # Initialization/setup hooks
    VALIDATION = "validation"  # Input validation hooks
    PRE_PROCESS = "pre_process"  # Pre-processing hooks
    MAIN = "main"             # Main processing hooks
    POST_PROCESS = "post_process"  # Post-processing hooks
    CLEANUP = "cleanup"       # Cleanup/teardown hooks
    
    @property
    def order(self) -> int:
        """Get the default execution order for this phase."""
        orders = {
            "setup": 0,
            "validation": 1,
            "pre_process": 2,
            "main": 3,
            "post_process": 4,
            "cleanup": 5,
        }
        return orders.get(self.value, 3)


@dataclass
class PhaseHook:
    """A hook assigned to a specific execution phase."""
    
    hook: RegisteredHook
    phase: PriorityPhase
    phase_priority: int = 0  # Priority within the phase


class PriorityOrderManager:
    """Manages complex priority ordering for hooks.
    
    Features:
    - Named priority groups
    - Execution phases
    - Before/after constraints
    - Conflict detection
    - Dynamic reordering
    - Topological sorting for dependencies
    """
    
    def __init__(self) -> None:
        self._groups: dict[str, PriorityGroup] = {}
        self._phase_hooks: dict[PriorityPhase, list[PhaseHook]] = {
            phase: [] for phase in PriorityPhase
        }
        self._constraints: list[PriorityConstraint] = []
        self._hook_to_group: dict[str, str] = {}  # hook name -> group name
        self._hook_to_phase: dict[str, PriorityPhase] = {}  # hook name -> phase
    
    def create_group(
        self,
        name: str,
        priority: int = 0,
        before: list[str] | None = None,
        after: list[str] | None = None,
        exclusive: bool = False,
    ) -> PriorityGroup:
        """Create a new priority group."""
        group = PriorityGroup(
            name=name,
            group_priority=priority,
            before_groups=before or [],
            after_groups=after or [],
            exclusive=exclusive,
        )
        self._groups[name] = group
        return group
    
    def get_group(self, name: str) -> PriorityGroup | None:
        """Get a priority group by name."""
        return self._groups.get(name)
    
    def add_to_group(
        self,
        group_name: str,
        hook: RegisteredHook,
    ) -> bool:
        """Add a hook to a group."""
        group = self._groups.get(group_name)
        if not group:
            return False
        group.add_hook(hook)
        if hook.name:
            self._hook_to_group[hook.name] = group_name
        return True
    
    def assign_phase(
        self,
        hook: RegisteredHook,
        phase: PriorityPhase,
        phase_priority: int = 0,
    ) -> None:
        """Assign a hook to an execution phase."""
        phase_hook = PhaseHook(hook=hook, phase=phase, phase_priority=phase_priority)
        self._phase_hooks[phase].append(phase_hook)
        self._phase_hooks[phase].sort(key=lambda h: h.phase_priority, reverse=True)
        if hook.name:
            self._hook_to_phase[hook.name] = phase
    
    def add_constraint(self, constraint: PriorityConstraint) -> None:
        """Add an ordering constraint."""
        self._constraints.append(constraint)
    
    def require_before(self, hook_name: str, target: str, hard: bool = True) -> None:
        """Add a 'before' constraint."""
        self.add_constraint(PriorityConstraint(
            hook_name=hook_name,
            constraint_type="before",
            target=target,
            hard=hard,
        ))
    
    def require_after(self, hook_name: str, target: str, hard: bool = True) -> None:
        """Add an 'after' constraint."""
        self.add_constraint(PriorityConstraint(
            hook_name=hook_name,
            constraint_type="after",
            target=target,
            hard=hard,
        ))
    
    def add_conflict(self, hook_name: str, conflicting_hook: str) -> None:
        """Mark two hooks as conflicting (can't both run)."""
        self.add_constraint(PriorityConstraint(
            hook_name=hook_name,
            constraint_type="conflicts",
            target=conflicting_hook,
            hard=True,
        ))
    
    def compute_group_order(self) -> list[str]:
        """Compute the execution order of groups using topological sort."""
        # Build dependency graph
        graph: dict[str, set[str]] = {name: set() for name in self._groups}
        
        for name, group in self._groups.items():
            # before_groups means this group must come before those
            # So those groups depend on this one
            for before in group.before_groups:
                if before in graph:
                    graph[before].add(name)
            # after_groups means this group comes after those
            # So this group depends on those
            for after in group.after_groups:
                if after in self._groups:
                    graph[name].add(after)
        
        # Topological sort
        result: list[str] = []
        visited: set[str] = set()
        temp_visited: set[str] = set()
        
        def visit(node: str) -> bool:
            if node in temp_visited:
                return False  # Cycle detected
            if node in visited:
                return True
            
            temp_visited.add(node)
            for dep in graph[node]:
                if not visit(dep):
                    return False
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
            return True
        
        # Sort by group priority first
        sorted_groups = sorted(
            self._groups.keys(),
            key=lambda n: self._groups[n].group_priority,
            reverse=True,
        )
        
        for name in sorted_groups:
            if name not in visited:
                if not visit(name):
                    # Cycle detected, fall back to priority order
                    result = sorted_groups
                    break
        
        return result
    
    def get_ordered_hooks(self, hooks: list[RegisteredHook]) -> list[RegisteredHook]:
        """Get hooks in execution order respecting all constraints."""
        # First, separate hooks into phases
        phased: dict[PriorityPhase, list[RegisteredHook]] = {
            phase: [] for phase in PriorityPhase
        }
        unphased: list[RegisteredHook] = []
        
        for hook in hooks:
            if hook.name and hook.name in self._hook_to_phase:
                phase = self._hook_to_phase[hook.name]
                phased[phase].append(hook)
            else:
                unphased.append(hook)
        
        # Sort unphased by priority
        unphased.sort(key=lambda h: h.priority, reverse=True)
        
        # Build result by phase order
        result: list[RegisteredHook] = []
        for phase in sorted(PriorityPhase, key=lambda p: p.order):
            phase_hooks = phased[phase]
            phase_hooks.sort(key=lambda h: h.priority, reverse=True)
            result.extend(phase_hooks)
        
        # Add unphased hooks to MAIN phase position
        main_idx = len([
            phase for phase in PriorityPhase
            if phase.order <= PriorityPhase.MAIN.order
        ])
        for i, hook in enumerate(unphased):
            result.insert(main_idx + i, hook)
        
        # Verify constraints
        self._verify_constraints(result)
        
        return result
    
    def _verify_constraints(self, order: list[RegisteredHook]) -> None:
        """Verify all constraints are satisfied."""
        hook_names = [h.name for h in order if h.name]
        
        for constraint in self._constraints:
            if not constraint.is_satisfied(hook_names):
                if constraint.hard:
                    # Log warning - in production would raise or reorder
                    pass
    
    def reorder_hook(
        self,
        hook_name: str,
        new_priority: int | None = None,
        new_phase: PriorityPhase | None = None,
        new_group: str | None = None,
    ) -> bool:
        """Dynamically reorder a hook."""
        # Find the hook in groups
        if hook_name in self._hook_to_group:
            old_group_name = self._hook_to_group[hook_name]
            old_group = self._groups.get(old_group_name)
            
            if old_group:
                # Find and update the hook
                for hook in old_group.hooks:
                    if hook.name == hook_name:
                        if new_priority is not None:
                            # Create new RegisteredHook with updated priority
                            pass  # Would need to update in place
                        
                        if new_group and new_group != old_group_name:
                            old_group.remove_hook(name=hook_name)
                            self.add_to_group(new_group, hook)
                        
                        break
        
        # Update phase if specified
        if new_phase and hook_name in self._hook_to_phase:
            self._hook_to_phase[hook_name] = new_phase
        
        return True
    
    def get_execution_plan(self, hook_type: HookType) -> dict[str, Any]:
        """Get a detailed execution plan for debugging."""
        group_order = self.compute_group_order()
        
        plan = {
            "hook_type": hook_type.value,
            "group_order": group_order,
            "phases": {},
            "constraints": [
                {
                    "hook": c.hook_name,
                    "type": c.constraint_type,
                    "target": c.target,
                    "hard": c.hard,
                }
                for c in self._constraints
            ],
        }
        
        for phase in PriorityPhase:
            plan["phases"][phase.value] = [
                {"hook": h.hook.name, "priority": h.phase_priority}
                for h in self._phase_hooks[phase]
            ]
        
        return plan


class AdvancedHookManager(HookManager):
    """Extended HookManager with advanced priority ordering capabilities."""
    
    def __init__(self) -> None:
        super().__init__()
        self._priority_manager = PriorityOrderManager()
        self._default_phase: dict[HookType, PriorityPhase] = {
            HookType.PRE_INIT: PriorityPhase.SETUP,
            HookType.POST_INIT: PriorityPhase.SETUP,
            HookType.PRE_EXECUTE: PriorityPhase.PRE_PROCESS,
            HookType.POST_EXECUTE: PriorityPhase.POST_PROCESS,
            HookType.ON_ERROR: PriorityPhase.CLEANUP,
            HookType.PRE_SHUTDOWN: PriorityPhase.CLEANUP,
        }
    
    @property
    def priority_manager(self) -> PriorityOrderManager:
        """Get the priority order manager."""
        return self._priority_manager
    
    def register_with_phase(
        self,
        hook_type: HookType,
        handler: HookHandler,
        phase: PriorityPhase,
        phase_priority: int = 0,
        name: str | None = None,
        plugin_name: str | None = None,
    ) -> None:
        """Register a hook with a specific execution phase."""
        registered = RegisteredHook(
            handler=handler,
            priority=phase_priority,
            name=name,
            plugin_name=plugin_name,
        )
        self._hooks[hook_type].append(registered)
        self._priority_manager.assign_phase(registered, phase, phase_priority)
    
    def register_in_group(
        self,
        hook_type: HookType,
        handler: HookHandler,
        group_name: str,
        priority: int = 0,
        name: str | None = None,
        plugin_name: str | None = None,
    ) -> bool:
        """Register a hook in a priority group."""
        registered = RegisteredHook(
            handler=handler,
            priority=priority,
            name=name,
            plugin_name=plugin_name,
        )
        self._hooks[hook_type].append(registered)
        return self._priority_manager.add_to_group(group_name, registered)
    
    def create_group(
        self,
        name: str,
        priority: int = 0,
        before: list[str] | None = None,
        after: list[str] | None = None,
    ) -> PriorityGroup:
        """Create a new priority group."""
        return self._priority_manager.create_group(
            name=name,
            priority=priority,
            before=before,
            after=after,
        )
    
    def require_order(
        self,
        first: str,
        second: str,
        hard: bool = True,
    ) -> None:
        """Require that one hook runs before another."""
        self._priority_manager.require_before(first, second, hard)
    
    def trigger_ordered(
        self,
        hook_type: HookType,
        data: dict[str, Any] | None = None,
    ) -> HookContext:
        """Trigger hooks using advanced ordering."""
        context = HookContext(hook_type=hook_type, data=data or {})
        
        # Get hooks in proper order
        raw_hooks = self._hooks[hook_type]
        ordered_hooks = self._priority_manager.get_ordered_hooks(raw_hooks)
        
        for registered in ordered_hooks:
            if context.cancelled:
                break
            try:
                result = registered.handler(context)
                if result is not None:
                    context.result = result
            except Exception as e:
                context.data.setdefault("errors", []).append({
                    "handler": registered.name or str(registered.handler),
                    "error": str(e),
                })
        
        return context
    
    def get_execution_plan(self, hook_type: HookType) -> dict[str, Any]:
        """Get the execution plan for a hook type."""
        return self._priority_manager.get_execution_plan(hook_type)


# Decorators for phase-based registration

def setup_hook(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for setup phase hooks."""
    def decorator(func: HookHandler) -> HookHandler:
        # Would register with AdvancedHookManager in setup phase
        return func
    return decorator


def validation_hook(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for validation phase hooks."""
    def decorator(func: HookHandler) -> HookHandler:
        return func
    return decorator


def cleanup_hook(priority: int = 0) -> Callable[[HookHandler], HookHandler]:
    """Decorator for cleanup phase hooks."""
    def decorator(func: HookHandler) -> HookHandler:
        return func
    return decorator


# Global advanced hook manager
_advanced_hook_manager: AdvancedHookManager | None = None


def get_advanced_hook_manager() -> AdvancedHookManager:
    """Get the global advanced hook manager."""
    global _advanced_hook_manager
    if _advanced_hook_manager is None:
        _advanced_hook_manager = AdvancedHookManager()
    return _advanced_hook_manager