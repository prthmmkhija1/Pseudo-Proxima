"""Benchmark scheduling using APScheduler (Phase 7.3).

Provides scheduled/continuous benchmarking capabilities:
- Start/stop scheduler daemon
- Add cron-based benchmark jobs
- Query scheduled job status

Requires APScheduler as an optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

try:  # Optional dependency
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
except Exception:  # pragma: no cover
    BackgroundScheduler = None  # type: ignore
    CronTrigger = None  # type: ignore
    IntervalTrigger = None  # type: ignore


class JobStatus(str, Enum):
    """Scheduled job status."""
    
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScheduledJob:
    """Information about a scheduled benchmark job."""
    
    job_id: str
    name: str
    trigger: str
    next_run_time: datetime | None
    status: JobStatus
    run_count: int = 0
    last_run_time: datetime | None = None
    last_run_success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Configuration for the benchmark scheduler."""
    
    timezone: str = "UTC"
    max_concurrent_jobs: int = 3
    job_defaults: dict[str, Any] = field(default_factory=lambda: {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 60,
    })
    executor_type: str = "threadpool"
    executor_max_workers: int = 4


@dataclass(slots=True)
class BenchmarkScheduler:
    """Lightweight wrapper over APScheduler for benchmark jobs.

    Attributes:
        schedule_config: Configuration dict for scheduler settings.
        scheduler: The underlying APScheduler BackgroundScheduler instance.
    """

    schedule_config: dict[str, Any] = field(default_factory=dict)
    scheduler: Any | None = field(default=None)
    _job_info: dict[str, ScheduledJob] = field(default_factory=dict)
    _listeners: list[Callable[[str, Any], None]] = field(default_factory=list)

    def start(self) -> None:
        if BackgroundScheduler is None:
            raise RuntimeError("APScheduler is not installed")
        if self.scheduler is None:
            self.scheduler = BackgroundScheduler()
        if not self.scheduler.running:
            self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()

    def add_job(self, func: Callable[..., Any], trigger: str = "cron", **trigger_args: Any) -> str:
        if self.scheduler is None:
            self.start()
        job = self.scheduler.add_job(func, trigger, **trigger_args)
        
        # Track job info
        self._job_info[job.id] = ScheduledJob(
            job_id=job.id,
            name=func.__name__ if hasattr(func, '__name__') else str(func),
            trigger=trigger,
            next_run_time=job.next_run_time,
            status=JobStatus.PENDING,
        )
        
        return job.id

    def status(self) -> list[str]:
        if self.scheduler is None:
            return []
        return [f"{job.id}: {job.next_run_time}" for job in self.scheduler.get_jobs()]
    
    # =========================================================================
    # Extended Scheduler API (Feature - Benchmarks)
    # =========================================================================
    
    def add_interval_job(
        self,
        func: Callable[..., Any],
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Add a job that runs at regular intervals.
        
        Args:
            func: Function to execute.
            hours: Interval in hours.
            minutes: Interval in minutes.
            seconds: Interval in seconds.
            name: Optional job name.
            **kwargs: Additional arguments for the job.
            
        Returns:
            Job ID.
        """
        if IntervalTrigger is None:
            raise RuntimeError("APScheduler is not installed")
        
        if self.scheduler is None:
            self.start()
        
        trigger = IntervalTrigger(hours=hours, minutes=minutes, seconds=seconds)
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            name=name or func.__name__,
            **kwargs,
        )
        
        self._job_info[job.id] = ScheduledJob(
            job_id=job.id,
            name=name or func.__name__,
            trigger=f"interval({hours}h {minutes}m {seconds}s)",
            next_run_time=job.next_run_time,
            status=JobStatus.PENDING,
        )
        
        return job.id
    
    def add_cron_job(
        self,
        func: Callable[..., Any],
        cron_expression: str | None = None,
        hour: int | str = "*",
        minute: int | str = 0,
        day_of_week: str = "*",
        name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Add a job with cron-style scheduling.
        
        Args:
            func: Function to execute.
            cron_expression: Optional cron expression string.
            hour: Hour(s) to run (0-23 or '*').
            minute: Minute(s) to run (0-59 or '*').
            day_of_week: Day(s) of week (mon-sun or '*').
            name: Optional job name.
            **kwargs: Additional arguments.
            
        Returns:
            Job ID.
        """
        if CronTrigger is None:
            raise RuntimeError("APScheduler is not installed")
        
        if self.scheduler is None:
            self.start()
        
        if cron_expression:
            trigger = CronTrigger.from_crontab(cron_expression)
        else:
            trigger = CronTrigger(hour=hour, minute=minute, day_of_week=day_of_week)
        
        job = self.scheduler.add_job(
            func,
            trigger=trigger,
            name=name or func.__name__,
            **kwargs,
        )
        
        cron_str = cron_expression or f"{minute} {hour} * * {day_of_week}"
        self._job_info[job.id] = ScheduledJob(
            job_id=job.id,
            name=name or func.__name__,
            trigger=f"cron({cron_str})",
            next_run_time=job.next_run_time,
            status=JobStatus.PENDING,
        )
        
        return job.id
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job.
        
        Args:
            job_id: Job ID to remove.
            
        Returns:
            True if job was removed.
        """
        if self.scheduler is None:
            return False
        
        try:
            self.scheduler.remove_job(job_id)
            self._job_info.pop(job_id, None)
            return True
        except Exception:
            return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job.
        
        Args:
            job_id: Job ID to pause.
            
        Returns:
            True if job was paused.
        """
        if self.scheduler is None:
            return False
        
        try:
            self.scheduler.pause_job(job_id)
            if job_id in self._job_info:
                self._job_info[job_id].status = JobStatus.PAUSED
            return True
        except Exception:
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job.
        
        Args:
            job_id: Job ID to resume.
            
        Returns:
            True if job was resumed.
        """
        if self.scheduler is None:
            return False
        
        try:
            self.scheduler.resume_job(job_id)
            if job_id in self._job_info:
                self._job_info[job_id].status = JobStatus.PENDING
            return True
        except Exception:
            return False
    
    def get_job(self, job_id: str) -> ScheduledJob | None:
        """Get job information.
        
        Args:
            job_id: Job ID to look up.
            
        Returns:
            ScheduledJob or None.
        """
        return self._job_info.get(job_id)
    
    def list_jobs(self) -> list[ScheduledJob]:
        """List all scheduled jobs.
        
        Returns:
            List of ScheduledJob objects.
        """
        # Update next_run_time from actual scheduler
        if self.scheduler:
            for job in self.scheduler.get_jobs():
                if job.id in self._job_info:
                    self._job_info[job.id].next_run_time = job.next_run_time
        
        return list(self._job_info.values())
    
    def run_job_now(self, job_id: str) -> bool:
        """Run a job immediately (in addition to its schedule).
        
        Args:
            job_id: Job ID to run.
            
        Returns:
            True if job was triggered.
        """
        if self.scheduler is None:
            return False
        
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.now())
                return True
        except Exception:
            pass
        return False
    
    def add_listener(self, callback: Callable[[str, Any], None]) -> None:
        """Add a listener for job events.
        
        Args:
            callback: Function(event_type, job_id) to call on events.
        """
        self._listeners.append(callback)
    
    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics.
        
        Returns:
            Dictionary with scheduler stats.
        """
        total = len(self._job_info)
        paused = sum(1 for j in self._job_info.values() if j.status == JobStatus.PAUSED)
        
        return {
            "total_jobs": total,
            "active_jobs": total - paused,
            "paused_jobs": paused,
            "scheduler_running": self.scheduler.running if self.scheduler else False,
        }


def run_scheduled_benchmark(job_func: Callable[[], Any]) -> Any:
    """Execute a scheduled benchmark job (wrapper for clarity)."""
    return job_func()


# =============================================================================
# Convenience Factories (Feature - Benchmarks)
# =============================================================================


def create_daily_benchmark_job(
    scheduler: BenchmarkScheduler,
    benchmark_func: Callable[..., Any],
    hour: int = 2,
    minute: int = 0,
    name: str = "daily_benchmark",
) -> str:
    """Create a job that runs daily at specified time.
    
    Args:
        scheduler: BenchmarkScheduler instance.
        benchmark_func: Benchmark function to run.
        hour: Hour to run (0-23, default 2 AM).
        minute: Minute to run (0-59).
        name: Job name.
        
    Returns:
        Job ID.
    """
    return scheduler.add_cron_job(
        benchmark_func,
        hour=hour,
        minute=minute,
        name=name,
    )


def create_hourly_benchmark_job(
    scheduler: BenchmarkScheduler,
    benchmark_func: Callable[..., Any],
    minute: int = 0,
    name: str = "hourly_benchmark",
) -> str:
    """Create a job that runs every hour.
    
    Args:
        scheduler: BenchmarkScheduler instance.
        benchmark_func: Benchmark function to run.
        minute: Minute of each hour to run.
        name: Job name.
        
    Returns:
        Job ID.
    """
    return scheduler.add_cron_job(
        benchmark_func,
        hour="*",
        minute=minute,
        name=name,
    )


def create_continuous_benchmark_job(
    scheduler: BenchmarkScheduler,
    benchmark_func: Callable[..., Any],
    interval_minutes: int = 15,
    name: str = "continuous_benchmark",
) -> str:
    """Create a job that runs continuously at interval.
    
    Args:
        scheduler: BenchmarkScheduler instance.
        benchmark_func: Benchmark function to run.
        interval_minutes: Interval between runs.
        name: Job name.
        
    Returns:
        Job ID.
    """
    return scheduler.add_interval_job(
        benchmark_func,
        minutes=interval_minutes,
        name=name,
    )

