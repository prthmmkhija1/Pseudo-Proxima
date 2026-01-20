# Plugin Hooks Reference

Complete documentation for Proxima's hook system, enabling plugins to intercept and extend execution lifecycle events.

## Overview

The hook system provides a powerful mechanism for plugins to:

- **Monitor** execution events (logging, metrics)
- **Modify** data at key points (preprocessing, postprocessing)
- **React** to events (notifications, cleanup)
- **Extend** functionality without modifying core code

---

## Hook Types

### HookType Enum

Available hook points in the execution lifecycle.

```python
from proxima.plugins.hooks import HookType

class HookType(str, Enum):
    """Types of hooks available in Proxima."""
    
    # Execution lifecycle
    PRE_EXECUTE = "pre_execute"
    """Before circuit execution starts."""
    
    POST_EXECUTE = "post_execute"
    """After circuit execution completes."""
    
    # Comparison lifecycle
    PRE_COMPARE = "pre_compare"
    """Before multi-backend comparison."""
    
    POST_COMPARE = "post_compare"
    """After comparison completes."""
    
    # Error handling
    ON_ERROR = "on_error"
    """When an error occurs during execution."""
    
    # Backend events
    ON_BACKEND_CHANGE = "on_backend_change"
    """When the active backend changes."""
    
    ON_BACKEND_INIT = "on_backend_init"
    """When a backend is initialized."""
    
    # Session events
    ON_SESSION_START = "on_session_start"
    """When a new session begins."""
    
    ON_SESSION_END = "on_session_end"
    """When a session ends."""
    
    ON_CHECKPOINT = "on_checkpoint"
    """When a checkpoint is created."""
    
    # Result events
    ON_RESULT_STORE = "on_result_store"
    """When results are stored."""
    
    ON_RESULT_EXPORT = "on_result_export"
    """When results are exported."""
    
    # LLM events
    PRE_LLM_REQUEST = "pre_llm_request"
    """Before an LLM request is sent."""
    
    POST_LLM_RESPONSE = "post_llm_response"
    """After an LLM response is received."""
```

---

## Hook Context

### HookContext

Context object passed to hook callbacks.

```python
from proxima.plugins.hooks import HookContext

@dataclass
class HookContext:
    """Context provided to hook callbacks."""
    
    hook_type: HookType
    """Type of hook being triggered."""
    
    data: dict[str, Any]
    """Hook-specific data payload."""
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    """When the hook was triggered."""
    
    session_id: str | None = None
    """Current session ID, if any."""
    
    backend_name: str | None = None
    """Current backend name, if applicable."""
    
    execution_id: str | None = None
    """Current execution ID, if applicable."""
    
    cancellable: bool = False
    """Whether this hook can cancel the operation."""
    
    cancelled: bool = False
    """Set to True to cancel the operation (if cancellable)."""
    
    modified_data: dict[str, Any] | None = None
    """Modified data to use instead of original."""
    
    def cancel(self) -> None:
        """Cancel the current operation (if cancellable)."""
        if self.cancellable:
            self.cancelled = True
        else:
            raise HookError(f"Hook {self.hook_type} is not cancellable")
    
    def modify(self, **updates) -> None:
        """Modify the data for subsequent processing."""
        if self.modified_data is None:
            self.modified_data = dict(self.data)
        self.modified_data.update(updates)
```

---

## Hook Manager

### HookManager

Central manager for hook registration and triggering.

```python
from proxima.plugins.hooks import HookManager, get_hook_manager

class HookManager:
    """Manages hook registration and triggering."""
    
    def register(
        self,
        hook_type: HookType,
        callback: Callable[[HookContext], None],
        priority: int = 0,
        name: str | None = None,
    ) -> str:
        """
        Register a hook callback.
        
        Args:
            hook_type: Type of hook to register for
            callback: Function to call when hook triggers
            priority: Execution priority (higher = earlier)
            name: Optional name for identification
            
        Returns:
            Registration ID for later unregistration
        """
    
    def unregister(
        self,
        hook_type: HookType,
        callback_or_id: Callable | str,
    ) -> bool:
        """
        Unregister a hook callback.
        
        Args:
            hook_type: Hook type
            callback_or_id: Callback function or registration ID
            
        Returns:
            True if callback was unregistered
        """
    
    def trigger(
        self,
        hook_type: HookType,
        context: HookContext,
    ) -> HookContext:
        """
        Trigger all callbacks for a hook type.
        
        Args:
            hook_type: Hook type to trigger
            context: Context to pass to callbacks
            
        Returns:
            Modified context after all callbacks
        """
    
    async def trigger_async(
        self,
        hook_type: HookType,
        context: HookContext,
    ) -> HookContext:
        """Trigger hooks asynchronously."""
    
    def list_hooks(
        self,
        hook_type: HookType | None = None,
    ) -> list[dict[str, Any]]:
        """
        List registered hooks.
        
        Args:
            hook_type: Filter by type (None = all)
            
        Returns:
            List of hook info dictionaries
        """
    
    def clear(self, hook_type: HookType | None = None) -> int:
        """
        Clear registered hooks.
        
        Args:
            hook_type: Type to clear (None = all)
            
        Returns:
            Number of hooks cleared
        """


# Global hook manager instance
def get_hook_manager() -> HookManager:
    """Get the global hook manager instance."""
    ...
```

---

## Creating Hook Plugins

### HookPlugin Base Class

```python
from proxima.plugins.base import Plugin, PluginMetadata, PluginType
from proxima.plugins.hooks import HookType, HookContext, get_hook_manager

class HookPlugin(Plugin):
    """Base class for hook plugins."""
    
    # Subclasses should define which hooks they use
    hooks: list[HookType] = []
    
    def initialize(self, context: PluginContext) -> None:
        """Register hooks on initialization."""
        self._context = context
        self._register_hooks()
    
    def shutdown(self) -> None:
        """Unregister hooks on shutdown."""
        self._unregister_hooks()
    
    def _register_hooks(self) -> None:
        """Register all hook handlers."""
        manager = get_hook_manager()
        self._registrations = []
        
        for hook_type in self.hooks:
            handler = getattr(self, f"_on_{hook_type.value}", None)
            if handler:
                reg_id = manager.register(hook_type, handler, name=self.name)
                self._registrations.append((hook_type, reg_id))
    
    def _unregister_hooks(self) -> None:
        """Unregister all hook handlers."""
        manager = get_hook_manager()
        for hook_type, reg_id in self._registrations:
            manager.unregister(hook_type, reg_id)
```

### Example: Timing Hook Plugin

```python
from proxima.plugins.base import PluginMetadata, PluginType
from proxima.plugins.hooks import HookType, HookContext
import time

class TimingHookPlugin(HookPlugin):
    """Plugin that tracks execution timing."""
    
    METADATA = PluginMetadata(
        name="timing_hook",
        version="1.0.0",
        plugin_type=PluginType.HOOK,
        description="Track execution timing for all backends",
    )
    
    hooks = [HookType.PRE_EXECUTE, HookType.POST_EXECUTE]
    
    def __init__(self):
        super().__init__()
        self._timings: dict[str, float] = {}
        self._start_times: dict[str, float] = {}
    
    def _on_pre_execute(self, context: HookContext) -> None:
        """Record start time."""
        execution_id = context.execution_id or "default"
        self._start_times[execution_id] = time.perf_counter()
    
    def _on_post_execute(self, context: HookContext) -> None:
        """Calculate and store execution time."""
        execution_id = context.execution_id or "default"
        if execution_id in self._start_times:
            elapsed = time.perf_counter() - self._start_times[execution_id]
            self._timings[execution_id] = elapsed
            del self._start_times[execution_id]
    
    def get_timing(self, execution_id: str) -> float | None:
        """Get timing for a specific execution."""
        return self._timings.get(execution_id)
    
    def get_all_timings(self) -> dict[str, float]:
        """Get all recorded timings."""
        return dict(self._timings)
```

### Example: Validation Hook Plugin

```python
class ValidationHookPlugin(HookPlugin):
    """Plugin that validates circuits before execution."""
    
    METADATA = PluginMetadata(
        name="validation_hook",
        version="1.0.0",
        plugin_type=PluginType.HOOK,
        description="Validate circuits before execution",
    )
    
    hooks = [HookType.PRE_EXECUTE]
    
    def __init__(self, max_qubits: int = 20, max_depth: int = 100):
        super().__init__()
        self.max_qubits = max_qubits
        self.max_depth = max_depth
    
    def _on_pre_execute(self, context: HookContext) -> None:
        """Validate circuit before execution."""
        circuit_info = context.data.get("circuit_info", {})
        
        qubits = circuit_info.get("num_qubits", 0)
        depth = circuit_info.get("depth", 0)
        
        errors = []
        
        if qubits > self.max_qubits:
            errors.append(f"Circuit has {qubits} qubits, max is {self.max_qubits}")
        
        if depth > self.max_depth:
            errors.append(f"Circuit depth {depth} exceeds max {self.max_depth}")
        
        if errors and context.cancellable:
            context.data["validation_errors"] = errors
            context.cancel()
```

---

## Hook Data Reference

### PRE_EXECUTE Data

```python
# context.data for PRE_EXECUTE hook
{
    "circuit": Any,              # Circuit object or QASM
    "circuit_info": {
        "num_qubits": int,
        "depth": int,
        "gate_count": int,
    },
    "backend": str,              # Target backend name
    "shots": int,                # Number of shots
    "options": dict,             # Execution options
}
```

### POST_EXECUTE Data

```python
# context.data for POST_EXECUTE hook
{
    "result": ExecutionResult,   # Execution result
    "circuit": Any,              # Original circuit
    "backend": str,              # Backend used
    "shots": int,                # Shots executed
    "execution_time": float,     # Time in seconds
    "success": bool,             # Whether execution succeeded
    "error": str | None,         # Error message if failed
}
```

### PRE_COMPARE Data

```python
# context.data for PRE_COMPARE hook
{
    "circuit": Any,              # Circuit to compare
    "backends": list[str],       # Backends to use
    "shots": int,                # Shots per backend
    "metrics": list[str],        # Metrics to calculate
}
```

### POST_COMPARE Data

```python
# context.data for POST_COMPARE hook
{
    "comparison_result": ComparisonResult,
    "circuit": Any,
    "backends": list[str],
    "metrics": dict[str, float],
    "total_time": float,
}
```

### ON_ERROR Data

```python
# context.data for ON_ERROR hook
{
    "error": Exception,          # The exception
    "error_type": str,           # Exception class name
    "error_message": str,        # Error message
    "traceback": str,            # Full traceback
    "context": str,              # Context where error occurred
    "recoverable": bool,         # Whether error is recoverable
}
```

---

## Advanced Hook Patterns

### Async Hooks

```python
class AsyncHookPlugin(HookPlugin):
    """Plugin with async hook handlers."""
    
    async def _on_post_execute(self, context: HookContext) -> None:
        """Async post-execute handler."""
        # Async operations (API calls, file I/O, etc.)
        await self.send_notification(context.data)
    
    async def send_notification(self, data: dict) -> None:
        """Send async notification."""
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=data)
```

### Conditional Hooks

```python
class ConditionalHookPlugin(HookPlugin):
    """Plugin that conditionally processes hooks."""
    
    def __init__(self, backends: list[str] | None = None):
        super().__init__()
        self.target_backends = backends
    
    def _on_pre_execute(self, context: HookContext) -> None:
        """Only process for specific backends."""
        backend = context.data.get("backend")
        
        if self.target_backends and backend not in self.target_backends:
            return  # Skip this backend
        
        # Process hook
        self.process_execution(context)
```

### Chain of Responsibility

```python
class ChainHookPlugin(HookPlugin):
    """Plugin that chains multiple processors."""
    
    def __init__(self):
        super().__init__()
        self.processors = []
    
    def add_processor(self, processor: Callable) -> None:
        """Add a processor to the chain."""
        self.processors.append(processor)
    
    def _on_post_execute(self, context: HookContext) -> None:
        """Run all processors in chain."""
        data = context.data
        
        for processor in self.processors:
            data = processor(data)
            if data is None:
                break  # Processor terminated chain
        
        if data:
            context.modify(**data)
```

---

## Best Practices

### Do's

1. **Keep hooks lightweight** - Avoid heavy computation in hooks
2. **Handle errors gracefully** - Don't let hook errors crash execution
3. **Use appropriate priority** - Higher priority for critical hooks
4. **Clean up on shutdown** - Always unregister hooks
5. **Document hook behavior** - Clearly describe what your hooks do

### Don'ts

1. **Don't block in sync hooks** - Use async hooks for I/O
2. **Don't modify immutable data** - Use `context.modify()` instead
3. **Don't store sensitive data** - Be careful with passwords/keys
4. **Don't assume hook order** - Use priority if order matters
5. **Don't cancel unnecessarily** - Only cancel when truly needed

### Error Handling

```python
class SafeHookPlugin(HookPlugin):
    """Plugin with proper error handling."""
    
    def _on_post_execute(self, context: HookContext) -> None:
        """Safely process hook with error handling."""
        try:
            self.process_result(context.data)
        except Exception as e:
            # Log error but don't crash
            self.logger.error(f"Hook processing failed: {e}")
            # Optionally add error to context
            context.data.setdefault("hook_errors", []).append({
                "plugin": self.name,
                "error": str(e),
            })
```

---

## Hook Testing

### Testing Hooks

```python
import pytest
from proxima.plugins.hooks import HookManager, HookType, HookContext

def test_hook_registration():
    """Test hook registration."""
    manager = HookManager()
    callback = Mock()
    
    reg_id = manager.register(HookType.PRE_EXECUTE, callback)
    
    assert reg_id is not None
    assert len(manager.list_hooks(HookType.PRE_EXECUTE)) == 1

def test_hook_triggering():
    """Test hook triggering."""
    manager = HookManager()
    callback = Mock()
    
    manager.register(HookType.POST_EXECUTE, callback)
    
    context = HookContext(
        hook_type=HookType.POST_EXECUTE,
        data={"result": "test"},
    )
    
    manager.trigger(HookType.POST_EXECUTE, context)
    
    callback.assert_called_once()
    assert callback.call_args[0][0].data == {"result": "test"}

def test_hook_priority():
    """Test hook priority ordering."""
    manager = HookManager()
    call_order = []
    
    manager.register(
        HookType.PRE_EXECUTE,
        lambda ctx: call_order.append("low"),
        priority=1,
    )
    manager.register(
        HookType.PRE_EXECUTE,
        lambda ctx: call_order.append("high"),
        priority=10,
    )
    
    context = HookContext(hook_type=HookType.PRE_EXECUTE, data={})
    manager.trigger(HookType.PRE_EXECUTE, context)
    
    assert call_order == ["high", "low"]
```
