"""
Plugin hook system.

Provides extension points for plugins to hook into Proxima's execution flow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
