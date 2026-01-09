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
