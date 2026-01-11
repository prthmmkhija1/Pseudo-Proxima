"""CLI Workflow Runners - Full workflow execution with progress, prompts, and output.

This module provides:
- WorkflowContext: Shared context for CLI workflows
- WorkflowRunner: Base class for workflow execution
- RunWorkflow: Complete run command workflow
- CompareWorkflow: Multi-backend comparison workflow
- ExportWorkflow: Export execution results
- ValidationWorkflow: Validate configuration and circuits
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Generic, TypeVar

import typer

from proxima.backends.registry import backend_registry
from proxima.config.settings import Settings
from proxima.core.executor import Executor
from proxima.core.planner import Planner
from proxima.core.state import ExecutionStateMachine
from proxima.resources.consent import ConsentCategory, ConsentManager
from proxima.resources.control import ExecutionController
from proxima.resources.monitor import ResourceMonitor
from proxima.resources.timer import ExecutionTimer, ProgressTracker
from proxima.utils.logging import get_logger

# ========== Workflow Status ==========


class WorkflowStatus(Enum):
    """Status of a workflow execution."""

    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    status: WorkflowStatus
    output: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED

    def __str__(self) -> str:
        if self.success:
            return f"Workflow completed in {self.duration_seconds:.2f}s"
        else:
            return f"Workflow {self.status.name}: {self.error or 'unknown error'}"


# ========== Workflow Context ==========


@dataclass
class WorkflowContext:
    """Shared context for CLI workflows."""

    settings: Settings
    dry_run: bool = False
    force: bool = False
    verbose: int = 0
    quiet: bool = False
    output_format: str = "text"
    no_progress: bool = False

    # Runtime components (initialized lazily)
    _consent_manager: ConsentManager | None = None
    _resource_monitor: ResourceMonitor | None = None
    _execution_controller: ExecutionController | None = None
    _logger: Any = None

    @property
    def consent_manager(self) -> ConsentManager:
        if self._consent_manager is None:
            self._consent_manager = ConsentManager()
            if self.force:
                self._consent_manager.enable_force_override()
        return self._consent_manager

    @property
    def resource_monitor(self) -> ResourceMonitor:
        if self._resource_monitor is None:
            self._resource_monitor = ResourceMonitor()
        return self._resource_monitor

    @property
    def execution_controller(self) -> ExecutionController:
        if self._execution_controller is None:
            self._execution_controller = ExecutionController()
        return self._execution_controller

    @property
    def logger(self):
        if self._logger is None:
            self._logger = get_logger("cli.workflow")
        return self._logger

    @classmethod
    def from_typer_context(cls, ctx: typer.Context) -> WorkflowContext:
        """Create context from Typer context object."""
        obj = ctx.obj or {}
        settings = obj.get("settings")
        if settings is None:
            from proxima.config.settings import config_service

            settings = config_service.load()

        return cls(
            settings=settings,
            dry_run=obj.get("dry_run", False),
            force=obj.get("force", False),
            verbose=obj.get("verbose", 0),
            quiet=obj.get("quiet", False),
            output_format=obj.get("output_format", "text"),
        )


# ========== Workflow Runner Base ==========


T = TypeVar("T")


class WorkflowRunner(ABC, Generic[T]):
    """Base class for workflow runners."""

    def __init__(self, context: WorkflowContext) -> None:
        self.context = context
        self._timer: ExecutionTimer | None = None
        self._progress: ProgressTracker | None = None

    @property
    def name(self) -> str:
        """Workflow name for logging/display."""
        return self.__class__.__name__

    def run(self, **kwargs) -> WorkflowResult:
        """Execute the workflow with full lifecycle management."""
        start_time = time.time()
        self._timer = ExecutionTimer()
        self._timer.start()

        try:
            # Pre-execution checks
            self._log_start(**kwargs)
            if not self._check_preconditions(**kwargs):
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    error="Precondition check failed",
                    duration_seconds=time.time() - start_time,
                )

            # Request consent if needed
            if not self._request_consent(**kwargs):
                return WorkflowResult(
                    status=WorkflowStatus.CANCELLED,
                    error="Consent denied",
                    duration_seconds=time.time() - start_time,
                )

            # Check resources
            if not self._check_resources(**kwargs):
                return WorkflowResult(
                    status=WorkflowStatus.FAILED,
                    error="Insufficient resources",
                    duration_seconds=time.time() - start_time,
                )

            # Dry run check
            if self.context.dry_run:
                plan = self._plan(**kwargs)
                return WorkflowResult(
                    status=WorkflowStatus.COMPLETED,
                    output={"plan": plan, "dry_run": True},
                    duration_seconds=time.time() - start_time,
                    metadata={"dry_run": True},
                )

            # Execute workflow
            output = self._execute(**kwargs)

            self._timer.stop()
            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                output=output,
                duration_seconds=self._timer.total_elapsed_seconds,
            )

        except KeyboardInterrupt:
            return WorkflowResult(
                status=WorkflowStatus.CANCELLED,
                error="Interrupted by user",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            self.context.logger.error(f"{self.name} failed", error=str(e))
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
        finally:
            if self._timer:
                self._timer.stop()
            self._cleanup()

    def _log_start(self, **kwargs) -> None:
        """Log workflow start."""
        self.context.logger.info(f"{self.name}.start", **kwargs)

    def _check_preconditions(self, **kwargs) -> bool:
        """Check if workflow can proceed. Override in subclasses."""
        return True

    def _request_consent(self, **kwargs) -> bool:
        """Request user consent if needed. Override in subclasses."""
        if self.context.force:
            return True
        return True

    def _check_resources(self, **kwargs) -> bool:
        """Check resource availability. Override in subclasses."""
        return True

    @abstractmethod
    def _plan(self, **kwargs) -> dict[str, Any]:
        """Create execution plan (for dry-run)."""
        pass

    @abstractmethod
    def _execute(self, **kwargs) -> T:
        """Execute the workflow."""
        pass

    def _cleanup(self) -> None:
        """Cleanup after workflow execution."""
        pass


# ========== Run Workflow ==========


@dataclass
class RunOptions:
    """Options for run workflow."""

    objective: str
    backend: str | None = None
    shots: int | None = None
    simulator_type: str | None = None
    timeout_seconds: float | None = None
    save_results: bool = True


class RunWorkflow(WorkflowRunner[dict[str, Any]]):
    """Complete run command workflow."""

    def __init__(self, context: WorkflowContext, options: RunOptions) -> None:
        super().__init__(context)
        self.options = options
        self._fsm: ExecutionStateMachine | None = None
        self._planner: Planner | None = None
        self._executor: Executor | None = None

    @property
    def name(self) -> str:
        return "RunWorkflow"

    def _check_preconditions(self, **kwargs) -> bool:
        """Verify backend is available."""
        backend_name = self.options.backend or self.context.settings.backends.default_backend

        try:
            status = backend_registry.get_status(backend_name)
            if not status.available:
                typer.echo(f"Backend '{backend_name}' is not available: {status.reason}")
                return False
        except KeyError:
            typer.echo(f"Backend '{backend_name}' is not registered")
            return False

        return True

    def _request_consent(self, **kwargs) -> bool:
        """Request consent for execution."""
        if self.context.force:
            return True

        # Check if using remote LLM
        if self.context.settings.llm.provider.lower() in ("openai", "anthropic", "remote"):
            if not self.context.consent_manager.request_consent(
                "remote_llm_usage",
                category=ConsentCategory.REMOTE_LLM,
                description="This operation may send data to a remote LLM API.",
            ):
                return False

        return True

    def _check_resources(self, **kwargs) -> bool:
        """Check memory and resources."""
        # Simple resource check
        memory_check = self.context.resource_monitor.memory.sample()
        if memory_check.level.name in ("CRITICAL",):
            typer.echo(f"Critical memory level: {memory_check.percent_used:.1f}% used")
            if not self.context.force:
                return False
        return True

    def _plan(self, **kwargs) -> dict[str, Any]:
        """Create execution plan."""
        backend_name = self.options.backend or self.context.settings.backends.default_backend

        return {
            "objective": self.options.objective,
            "backend": backend_name,
            "shots": self.options.shots,
            "simulator_type": self.options.simulator_type,
            "timeout": self.options.timeout_seconds,
            "settings": {
                "verbosity": self.context.settings.general.verbosity,
                "output_format": self.context.settings.general.output_format,
            },
        }

    def _execute(self, **kwargs) -> dict[str, Any]:
        """Execute the run workflow."""
        self._fsm = ExecutionStateMachine()
        self._planner = Planner(self._fsm)
        self._executor = Executor(self._fsm)

        # Plan phase
        plan = self._planner.plan(self.options.objective)

        # Execute phase
        result = self._executor.run(plan)

        return {
            "state": self._fsm.state,
            "plan": plan,
            "result": result,
            "execution_time": self._timer.total_elapsed_seconds if self._timer else 0,
        }


# ========== Compare Workflow ==========


@dataclass
class CompareOptions:
    """Options for compare workflow."""

    objective: str
    backends: list[str]
    parallel: bool = False
    save_results: bool = True


class CompareWorkflow(WorkflowRunner[dict[str, Any]]):
    """Multi-backend comparison workflow."""

    def __init__(self, context: WorkflowContext, options: CompareOptions) -> None:
        super().__init__(context)
        self.options = options

    @property
    def name(self) -> str:
        return "CompareWorkflow"

    def _check_preconditions(self, **kwargs) -> bool:
        """Verify all backends are available."""
        available_backends = []
        unavailable = []

        for backend_name in self.options.backends:
            try:
                status = backend_registry.get_status(backend_name)
                if status.available:
                    available_backends.append(backend_name)
                else:
                    unavailable.append(f"{backend_name}: {status.reason}")
            except KeyError:
                unavailable.append(f"{backend_name}: not registered")

        if unavailable:
            typer.echo("Unavailable backends:")
            for msg in unavailable:
                typer.echo(f"  - {msg}")

        if not available_backends:
            typer.echo("No available backends for comparison")
            return False

        # Update to only available backends
        self.options.backends = available_backends
        return True

    def _plan(self, **kwargs) -> dict[str, Any]:
        """Create comparison plan."""
        return {
            "objective": self.options.objective,
            "backends": self.options.backends,
            "parallel": self.options.parallel,
            "comparison_type": "execution_time_and_results",
        }

    def _execute(self, **kwargs) -> dict[str, Any]:
        """Execute comparison across backends."""
        if self.options.parallel and len(self.options.backends) > 1:
            results = self._execute_parallel()
        else:
            results = self._execute_sequential()

        return {
            "objective": self.options.objective,
            "backends": self.options.backends,
            "results": results,
            "comparison": self._generate_comparison(results),
        }

    def _execute_sequential(self) -> dict[str, Any]:
        """Execute backends sequentially."""
        results = {}

        for backend_name in self.options.backends:
            typer.echo(f"Running on {backend_name}...")

            fsm = ExecutionStateMachine()
            planner = Planner(fsm)
            executor = Executor(fsm)

            start_time = time.time()
            plan = planner.plan(self.options.objective)
            result = executor.run(plan)
            elapsed = time.time() - start_time

            results[backend_name] = {
                "result": result,
                "execution_time": elapsed,
                "state": fsm.state,
            }

        return results

    def _execute_parallel(self) -> dict[str, Any]:
        """Execute backends in parallel using ThreadPoolExecutor."""
        import concurrent.futures

        results: dict[str, Any] = {}

        def run_backend(backend_name: str) -> tuple[str, dict[str, Any]]:
            """Run a single backend and return results."""
            fsm = ExecutionStateMachine()
            planner = Planner(fsm)
            executor = Executor(fsm)

            start_time = time.time()
            plan = planner.plan(self.options.objective)
            result = executor.run(plan)
            elapsed = time.time() - start_time

            return backend_name, {
                "result": result,
                "execution_time": elapsed,
                "state": fsm.state,
            }

        typer.echo(f"Running {len(self.options.backends)} backends in parallel...")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.options.backends)
        ) as executor:
            futures = {
                executor.submit(run_backend, backend): backend for backend in self.options.backends
            }

            for future in concurrent.futures.as_completed(futures):
                backend_name = futures[future]
                try:
                    name, data = future.result()
                    results[name] = data
                    typer.echo(f"  ✓ {name} completed in {data['execution_time']:.3f}s")
                except Exception as exc:
                    results[backend_name] = {
                        "result": None,
                        "execution_time": 0,
                        "state": "error",
                        "error": str(exc),
                    }
                    typer.echo(f"  ✗ {backend_name} failed: {exc}")

        return results

    def _generate_comparison(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate comparison summary."""
        times = {name: data["execution_time"] for name, data in results.items()}
        fastest = min(times, key=times.get) if times else None

        return {
            "fastest_backend": fastest,
            "execution_times": times,
            "speedup": (
                {name: times[fastest] / t if t > 0 else 0 for name, t in times.items()}
                if fastest
                else {}
            ),
        }


# ========== Validation Workflow ==========


@dataclass
class ValidateOptions:
    """Options for validation workflow."""

    config_path: Path | None = None
    circuit_path: Path | None = None
    backend: str | None = None
    strict: bool = False


class ValidationWorkflow(WorkflowRunner[dict[str, Any]]):
    """Configuration and circuit validation workflow."""

    def __init__(self, context: WorkflowContext, options: ValidateOptions) -> None:
        super().__init__(context)
        self.options = options

    @property
    def name(self) -> str:
        return "ValidationWorkflow"

    def _plan(self, **kwargs) -> dict[str, Any]:
        """Create validation plan."""
        return {
            "config_path": str(self.options.config_path) if self.options.config_path else None,
            "circuit_path": str(self.options.circuit_path) if self.options.circuit_path else None,
            "backend": self.options.backend,
            "strict": self.options.strict,
        }

    def _execute(self, **kwargs) -> dict[str, Any]:
        """Execute validation."""
        issues: list[dict[str, Any]] = []
        warnings: list[str] = []

        # Validate configuration
        if self.options.config_path:
            config_issues = self._validate_config()
            issues.extend(config_issues)

        # Validate current settings
        settings_issues = self._validate_settings()
        issues.extend(settings_issues)

        # Validate backend availability
        backend_issues = self._validate_backend()
        issues.extend(backend_issues)

        has_errors = any(i.get("severity") == "error" for i in issues)

        return {
            "valid": not has_errors,
            "issues": issues,
            "warnings": warnings,
            "checked": {
                "config": self.options.config_path is not None,
                "settings": True,
                "backend": self.options.backend is not None,
            },
        }

    def _validate_config(self) -> list[dict[str, Any]]:
        """Validate configuration file."""
        issues = []

        if self.options.config_path and not self.options.config_path.exists():
            issues.append(
                {
                    "severity": "error",
                    "path": str(self.options.config_path),
                    "message": "Configuration file not found",
                }
            )

        return issues

    def _validate_settings(self) -> list[dict[str, Any]]:
        """Validate current settings."""
        from proxima.config.validation import validate_settings

        result = validate_settings(self.context.settings.model_dump())
        issues = []

        for issue in result.issues:
            issues.append(
                {
                    "severity": issue.severity.value,
                    "path": issue.path,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                }
            )

        return issues

    def _validate_backend(self) -> list[dict[str, Any]]:
        """Validate backend availability."""
        issues = []
        backend_name = self.options.backend or self.context.settings.backends.default_backend

        try:
            status = backend_registry.get_status(backend_name)
            if not status.available:
                issues.append(
                    {
                        "severity": "warning",
                        "path": "backends.default_backend",
                        "message": f"Backend '{backend_name}' is not available: {status.reason}",
                    }
                )
        except KeyError:
            issues.append(
                {
                    "severity": "error",
                    "path": "backends.default_backend",
                    "message": f"Backend '{backend_name}' is not registered",
                }
            )

        return issues


# ========== Export Workflow ==========


@dataclass
class ExportOptions:
    """Options for export workflow."""

    output_path: Path
    format: str = "json"
    include_metadata: bool = True
    source: str = "history"  # history, session, or results


class ExportWorkflow(WorkflowRunner[dict[str, Any]]):
    """Export execution results workflow."""

    def __init__(self, context: WorkflowContext, options: ExportOptions) -> None:
        super().__init__(context)
        self.options = options

    @property
    def name(self) -> str:
        return "ExportWorkflow"

    def _plan(self, **kwargs) -> dict[str, Any]:
        """Create export plan."""
        return {
            "output_path": str(self.options.output_path),
            "format": self.options.format,
            "source": self.options.source,
            "include_metadata": self.options.include_metadata,
        }

    def _execute(self, **kwargs) -> dict[str, Any]:
        """Execute export."""
        import json

        # Gather data based on source
        if self.options.source == "history":
            try:
                from proxima.data.history import execution_history

                data = [r.to_dict() for r in execution_history.list_results()]
            except ImportError:
                # History module not available, export empty
                data = []
        elif self.options.source == "session":
            try:
                from proxima.resources.session import SessionManager

                manager = SessionManager()
                sessions = manager.list_sessions()
                data = [{"id": s.id, "status": s.status.value} for s in sessions]
            except ImportError:
                data = []
        else:
            data = {"message": "No data to export"}

        # Add metadata
        if self.options.include_metadata:
            export_data = {
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": self.options.source,
                "format": self.options.format,
                "data": data,
            }
        else:
            export_data = data

        # Write output
        self.options.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.options.format == "json":
            self.options.output_path.write_text(
                json.dumps(export_data, indent=2, default=str),
                encoding="utf-8",
            )
        elif self.options.format == "yaml":
            import yaml

            self.options.output_path.write_text(
                yaml.safe_dump(export_data, sort_keys=False),
                encoding="utf-8",
            )
        else:
            # Plain text
            self.options.output_path.write_text(str(export_data), encoding="utf-8")

        return {
            "path": str(self.options.output_path),
            "format": self.options.format,
            "records": len(data) if isinstance(data, list) else 1,
        }


# ========== Convenience Functions ==========


def run_workflow(
    ctx: typer.Context,
    objective: str,
    backend: str | None = None,
    **kwargs,
) -> WorkflowResult:
    """Convenience function to run a workflow from CLI."""
    context = WorkflowContext.from_typer_context(ctx)
    options = RunOptions(objective=objective, backend=backend, **kwargs)
    workflow = RunWorkflow(context, options)
    return workflow.run()


def compare_backends(
    ctx: typer.Context,
    objective: str,
    backends: list[str],
    **kwargs,
) -> WorkflowResult:
    """Convenience function to compare backends from CLI."""
    context = WorkflowContext.from_typer_context(ctx)
    options = CompareOptions(objective=objective, backends=backends, **kwargs)
    workflow = CompareWorkflow(context, options)
    return workflow.run()


def validate_config(
    ctx: typer.Context,
    config_path: Path | None = None,
    **kwargs,
) -> WorkflowResult:
    """Convenience function to validate configuration from CLI."""
    context = WorkflowContext.from_typer_context(ctx)
    options = ValidateOptions(config_path=config_path, **kwargs)
    workflow = ValidationWorkflow(context, options)
    return workflow.run()
