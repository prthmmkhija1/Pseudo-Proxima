"""
Example Hook Plugins.

Demonstrates how to create hook plugins for extending Proxima's execution flow.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, ClassVar

from proxima.plugins.base import Plugin, PluginMetadata, PluginType
from proxima.plugins.hooks import HookContext, HookType, get_hook_manager

logger = logging.getLogger(__name__)


class LoggingHookPlugin(Plugin):
    """Log execution events.
    
    Provides detailed logging of:
    - Execution lifecycle events
    - Backend operations
    - LLM requests
    - Errors and warnings
    
    Configuration:
        log_level: Logging level (default: 'INFO')
        include_timing: Include timing information (default: True)
        log_data: Include data in logs (default: False)
    
    Example:
        plugin = LoggingHookPlugin({"log_level": "DEBUG"})
    """
    
    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="logging_hook",
        version="1.0.0",
        plugin_type=PluginType.HOOK,
        description="Log execution events for debugging and monitoring",
        author="Proxima Team",
        provides=["logging", "debugging"],
        config_schema={
            "type": "object",
            "properties": {
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    "default": "INFO",
                },
                "include_timing": {"type": "boolean", "default": True},
                "log_data": {"type": "boolean", "default": False},
            },
        },
    )
    
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the logging hook plugin."""
        super().__init__(config)
        self._start_times: dict[str, float] = {}
        self._logger = logging.getLogger("proxima.hooks.logging")
    
    def initialize(self, context=None) -> None:
        """Register hooks with the hook manager."""
        manager = get_hook_manager()
        
        # Register for all hook types
        hooks_to_register = [
            (HookType.PRE_EXECUTE, self._on_pre_execute),
            (HookType.POST_EXECUTE, self._on_post_execute),
            (HookType.ON_ERROR, self._on_error),
            (HookType.PRE_BACKEND_RUN, self._on_pre_backend_run),
            (HookType.POST_BACKEND_RUN, self._on_post_backend_run),
            (HookType.PRE_LLM_REQUEST, self._on_pre_llm_request),
            (HookType.POST_LLM_RESPONSE, self._on_post_llm_response),
        ]
        
        for hook_type, handler in hooks_to_register:
            manager.register(
                hook_type,
                handler,
                name=f"{self.name}_{hook_type.value}",
                plugin_name=self.name,
            )
        
        self._log("INFO", "Logging hooks registered")
    
    def shutdown(self) -> None:
        """Unregister hooks."""
        manager = get_hook_manager()
        manager.unregister_by_plugin(self.name)
        self._log("INFO", "Logging hooks unregistered")
    
    def _on_pre_execute(self, context: HookContext) -> None:
        """Handle pre-execute event."""
        self._start_times["execute"] = time.perf_counter()
        self._log("INFO", "Execution started", context.data)
    
    def _on_post_execute(self, context: HookContext) -> None:
        """Handle post-execute event."""
        elapsed = self._get_elapsed("execute")
        self._log(
            "INFO",
            f"Execution completed in {elapsed:.2f}ms",
            context.data if self.get_config("log_data") else None,
        )
    
    def _on_error(self, context: HookContext) -> None:
        """Handle error event."""
        error = context.get("error", "Unknown error")
        self._log("ERROR", f"Execution error: {error}", context.data)
    
    def _on_pre_backend_run(self, context: HookContext) -> None:
        """Handle pre-backend-run event."""
        backend = context.get("backend", "unknown")
        self._start_times[f"backend_{backend}"] = time.perf_counter()
        self._log("DEBUG", f"Backend run started: {backend}")
    
    def _on_post_backend_run(self, context: HookContext) -> None:
        """Handle post-backend-run event."""
        backend = context.get("backend", "unknown")
        elapsed = self._get_elapsed(f"backend_{backend}")
        self._log("DEBUG", f"Backend run completed: {backend} ({elapsed:.2f}ms)")
    
    def _on_pre_llm_request(self, context: HookContext) -> None:
        """Handle pre-LLM-request event."""
        provider = context.get("provider", "unknown")
        self._start_times[f"llm_{provider}"] = time.perf_counter()
        self._log("DEBUG", f"LLM request started: {provider}")
    
    def _on_post_llm_response(self, context: HookContext) -> None:
        """Handle post-LLM-response event."""
        provider = context.get("provider", "unknown")
        elapsed = self._get_elapsed(f"llm_{provider}")
        self._log("DEBUG", f"LLM response received: {provider} ({elapsed:.2f}ms)")
    
    def _get_elapsed(self, key: str) -> float:
        """Get elapsed time in milliseconds."""
        if key in self._start_times:
            elapsed = (time.perf_counter() - self._start_times.pop(key)) * 1000
            return elapsed
        return 0.0
    
    def _log(
        self,
        level: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Log a message at the specified level."""
        if not self._enabled:
            return
        
        config_level = self.get_config("log_level", "INFO")
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        
        if levels.get(level, 20) < levels.get(config_level, 20):
            return
        
        include_timing = self.get_config("include_timing", True)
        timestamp = datetime.now(timezone.utc).isoformat() if include_timing else ""
        
        full_message = f"[{timestamp}] {message}" if timestamp else message
        
        if data and self.get_config("log_data", False):
            full_message += f" | data={data}"
        
        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn(full_message)


class MetricsHookPlugin(Plugin):
    """Collect execution metrics.
    
    Tracks:
    - Execution counts
    - Timing statistics
    - Error rates
    - Backend usage
    
    Metrics can be retrieved for monitoring and dashboards.
    
    Example:
        plugin = MetricsHookPlugin()
        plugin.initialize()
        # ... run executions ...
        metrics = plugin.get_metrics()
    """
    
    METADATA: ClassVar[PluginMetadata] = PluginMetadata(
        name="metrics_hook",
        version="1.0.0",
        plugin_type=PluginType.HOOK,
        description="Collect and expose execution metrics",
        author="Proxima Team",
        provides=["metrics", "monitoring"],
    )
    
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the metrics hook plugin."""
        super().__init__(config)
        self._reset_metrics()
    
    def _reset_metrics(self) -> None:
        """Reset all metrics."""
        self._execution_count = 0
        self._error_count = 0
        self._backend_usage: dict[str, int] = defaultdict(int)
        self._timing_samples: list[float] = []
        self._start_times: dict[str, float] = {}
        self._started_at = datetime.now(timezone.utc)
    
    def initialize(self, context=None) -> None:
        """Register hooks with the hook manager."""
        manager = get_hook_manager()
        
        manager.register(
            HookType.PRE_EXECUTE,
            self._on_pre_execute,
            name=f"{self.name}_pre_execute",
            plugin_name=self.name,
        )
        manager.register(
            HookType.POST_EXECUTE,
            self._on_post_execute,
            name=f"{self.name}_post_execute",
            plugin_name=self.name,
        )
        manager.register(
            HookType.ON_ERROR,
            self._on_error,
            name=f"{self.name}_on_error",
            plugin_name=self.name,
        )
        manager.register(
            HookType.POST_BACKEND_RUN,
            self._on_post_backend_run,
            name=f"{self.name}_post_backend",
            plugin_name=self.name,
        )
    
    def shutdown(self) -> None:
        """Unregister hooks."""
        manager = get_hook_manager()
        manager.unregister_by_plugin(self.name)
    
    def _on_pre_execute(self, context: HookContext) -> None:
        """Track execution start."""
        exec_id = context.get("execution_id", str(self._execution_count))
        self._start_times[exec_id] = time.perf_counter()
        self._execution_count += 1
    
    def _on_post_execute(self, context: HookContext) -> None:
        """Track execution completion."""
        exec_id = context.get("execution_id", str(self._execution_count - 1))
        if exec_id in self._start_times:
            elapsed = (time.perf_counter() - self._start_times.pop(exec_id)) * 1000
            self._timing_samples.append(elapsed)
            
            # Keep only last 1000 samples
            if len(self._timing_samples) > 1000:
                self._timing_samples = self._timing_samples[-1000:]
    
    def _on_error(self, context: HookContext) -> None:
        """Track errors."""
        self._error_count += 1
    
    def _on_post_backend_run(self, context: HookContext) -> None:
        """Track backend usage."""
        backend = context.get("backend", "unknown")
        self._backend_usage[backend] += 1
    
    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of collected metrics.
        """
        uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        
        # Calculate timing statistics
        timing_stats = {}
        if self._timing_samples:
            samples = self._timing_samples
            timing_stats = {
                "count": len(samples),
                "mean_ms": sum(samples) / len(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "p50_ms": self._percentile(samples, 50),
                "p95_ms": self._percentile(samples, 95),
                "p99_ms": self._percentile(samples, 99),
            }
        
        error_rate = self._error_count / self._execution_count if self._execution_count > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "error_rate": error_rate,
            "executions_per_minute": (self._execution_count / uptime * 60) if uptime > 0 else 0,
            "backend_usage": dict(self._backend_usage),
            "timing": timing_stats,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        self._reset_metrics()
    
    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of sorted data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]
