"""CLI Progress Display - Progress bars, spinners, and status indicators.

This module provides:
- ProgressDisplay: Base class for progress displays
- SpinnerProgress: Simple spinner for indeterminate operations
- BarProgress: Progress bar for determinate operations
- MultiProgress: Multiple concurrent progress bars
- StepProgress: Step-by-step progress for workflows
- progress_context: Context manager for progress display
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeVar

try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.status import Status
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ========== Progress Status ==========


class ProgressStatus(Enum):
    """Status of a progress operation."""

    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


# ========== Base Progress Display ==========


class ProgressDisplay(ABC):
    """Base class for progress displays."""

    def __init__(
        self,
        message: str = "",
        total: int | None = None,
        no_progress: bool = False,
    ) -> None:
        self.message = message
        self.total = total
        self.current = 0
        self.no_progress = no_progress
        self._status = ProgressStatus.PENDING
        self._start_time: float | None = None
        self._end_time: float | None = None

    @property
    def status(self) -> ProgressStatus:
        return self._status

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.time()
        return end - self._start_time

    @property
    def percentage(self) -> float:
        """Completion percentage (0-100)."""
        if self.total is None or self.total == 0:
            return 0.0
        return (self.current / self.total) * 100

    def start(self) -> None:
        """Start the progress display."""
        self._status = ProgressStatus.RUNNING
        self._start_time = time.time()
        if not self.no_progress:
            self._start()

    def update(self, advance: int = 1, message: str | None = None) -> None:
        """Update progress."""
        self.current = min(self.current + advance, self.total or self.current + advance)
        if message:
            self.message = message
        if not self.no_progress:
            self._update(advance, message)

    def set(self, current: int, message: str | None = None) -> None:
        """Set current progress value."""
        self.current = current
        if message:
            self.message = message
        if not self.no_progress:
            self._set(current, message)

    def complete(self, message: str | None = None) -> None:
        """Mark progress as complete."""
        self._status = ProgressStatus.COMPLETED
        self._end_time = time.time()
        if self.total:
            self.current = self.total
        if message:
            self.message = message
        if not self.no_progress:
            self._complete(message)

    def fail(self, error: str | None = None) -> None:
        """Mark progress as failed."""
        self._status = ProgressStatus.FAILED
        self._end_time = time.time()
        if not self.no_progress:
            self._fail(error)

    def cancel(self) -> None:
        """Cancel the progress."""
        self._status = ProgressStatus.CANCELLED
        self._end_time = time.time()
        if not self.no_progress:
            self._cancel()

    @abstractmethod
    def _start(self) -> None:
        pass

    @abstractmethod
    def _update(self, advance: int, message: str | None) -> None:
        pass

    @abstractmethod
    def _set(self, current: int, message: str | None) -> None:
        pass

    @abstractmethod
    def _complete(self, message: str | None) -> None:
        pass

    @abstractmethod
    def _fail(self, error: str | None) -> None:
        pass

    @abstractmethod
    def _cancel(self) -> None:
        pass


# ========== Simple Progress (Console-based) ==========


class SimpleProgress(ProgressDisplay):
    """Simple console-based progress (no rich)."""

    def __init__(
        self,
        message: str = "",
        total: int | None = None,
        no_progress: bool = False,
        width: int = 40,
    ) -> None:
        super().__init__(message, total, no_progress)
        self.width = width
        self._last_line_length = 0

    def _start(self) -> None:
        self._print_progress()

    def _update(self, advance: int, message: str | None) -> None:
        self._print_progress()

    def _set(self, current: int, message: str | None) -> None:
        self._print_progress()

    def _complete(self, message: str | None) -> None:
        self._clear_line()
        msg = message or self.message or "Done"
        sys.stdout.write(f"✓ {msg}\n")
        sys.stdout.flush()

    def _fail(self, error: str | None) -> None:
        self._clear_line()
        msg = error or "Failed"
        sys.stdout.write(f"✗ {msg}\n")
        sys.stdout.flush()

    def _cancel(self) -> None:
        self._clear_line()
        sys.stdout.write("⊘ Cancelled\n")
        sys.stdout.flush()

    def _print_progress(self) -> None:
        """Print progress bar to console."""
        self._clear_line()

        if self.total:
            filled = int(self.width * self.current / self.total)
            bar = "█" * filled + "░" * (self.width - filled)
            pct = self.percentage
            line = f"\r{self.message} [{bar}] {pct:5.1f}% ({self.current}/{self.total})"
        else:
            # Indeterminate progress
            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            idx = int(self.elapsed * 10) % len(spinner)
            line = f"\r{spinner[idx]} {self.message}"

        self._last_line_length = len(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def _clear_line(self) -> None:
        """Clear the current line."""
        sys.stdout.write("\r" + " " * self._last_line_length + "\r")
        sys.stdout.flush()


# ========== Rich Progress (if available) ==========


if RICH_AVAILABLE:

    class RichSpinner(ProgressDisplay):
        """Rich spinner for indeterminate operations."""

        def __init__(
            self,
            message: str = "",
            no_progress: bool = False,
            console: Console | None = None,
            spinner: str = "dots",
        ) -> None:
            super().__init__(message, total=None, no_progress=no_progress)
            self.console = console or Console()
            self.spinner_type = spinner
            self._status_display: Status | None = None

        def _start(self) -> None:
            self._status_display = self.console.status(
                self.message,
                spinner=self.spinner_type,
            )
            self._status_display.start()

        def _update(self, advance: int, message: str | None) -> None:
            if self._status_display and message:
                self._status_display.update(message)

        def _set(self, current: int, message: str | None) -> None:
            if self._status_display and message:
                self._status_display.update(message)

        def _complete(self, message: str | None) -> None:
            if self._status_display:
                self._status_display.stop()
            msg = message or self.message or "Done"
            self.console.print(f"[green]✓[/green] {msg}")

        def _fail(self, error: str | None) -> None:
            if self._status_display:
                self._status_display.stop()
            msg = error or "Failed"
            self.console.print(f"[red]✗[/red] {msg}")

        def _cancel(self) -> None:
            if self._status_display:
                self._status_display.stop()
            self.console.print("[yellow]⊘[/yellow] Cancelled")

    class RichProgress(ProgressDisplay):
        """Rich progress bar for determinate operations."""

        def __init__(
            self,
            message: str = "",
            total: int | None = None,
            no_progress: bool = False,
            console: Console | None = None,
            transient: bool = False,
        ) -> None:
            super().__init__(message, total, no_progress)
            self.console = console or Console()
            self.transient = transient
            self._progress: Progress | None = None
            self._task_id: TaskID | None = None

        def _start(self) -> None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=self.transient,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.message,
                total=self.total,
            )

        def _update(self, advance: int, message: str | None) -> None:
            if self._progress and self._task_id is not None:
                updates: dict[str, Any] = {"advance": advance}
                if message:
                    updates["description"] = message
                self._progress.update(self._task_id, **updates)

        def _set(self, current: int, message: str | None) -> None:
            if self._progress and self._task_id is not None:
                updates: dict[str, Any] = {"completed": current}
                if message:
                    updates["description"] = message
                self._progress.update(self._task_id, **updates)

        def _complete(self, message: str | None) -> None:
            if self._progress:
                if self._task_id is not None and self.total:
                    self._progress.update(self._task_id, completed=self.total)
                self._progress.stop()
            msg = message or self.message or "Done"
            self.console.print(f"[green]✓[/green] {msg}")

        def _fail(self, error: str | None) -> None:
            if self._progress:
                self._progress.stop()
            msg = error or "Failed"
            self.console.print(f"[red]✗[/red] {msg}")

        def _cancel(self) -> None:
            if self._progress:
                self._progress.stop()
            self.console.print("[yellow]⊘[/yellow] Cancelled")


# ========== Step Progress ==========


@dataclass
class Step:
    """A step in a multi-step workflow."""

    name: str
    description: str = ""
    status: ProgressStatus = ProgressStatus.PENDING
    duration: float = 0.0
    error: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.status in (ProgressStatus.COMPLETED, ProgressStatus.FAILED)

    @property
    def icon(self) -> str:
        icons = {
            ProgressStatus.PENDING: "○",
            ProgressStatus.RUNNING: "◐",
            ProgressStatus.PAUSED: "◑",
            ProgressStatus.COMPLETED: "●",
            ProgressStatus.FAILED: "✗",
            ProgressStatus.CANCELLED: "⊘",
        }
        return icons.get(self.status, "○")


class StepProgress:
    """Step-by-step progress for multi-step workflows."""

    def __init__(
        self,
        steps: list[str] | list[Step],
        title: str = "Progress",
        no_progress: bool = False,
    ) -> None:
        self.title = title
        self.no_progress = no_progress
        self._current_step = 0
        self._start_time: float | None = None

        # Convert string steps to Step objects
        self.steps: list[Step] = []
        for s in steps:
            if isinstance(s, str):
                self.steps.append(Step(name=s))
            else:
                self.steps.append(s)

        if RICH_AVAILABLE:
            self.console = Console()
            self._live: Live | None = None
        else:
            self.console = None
            self._live = None

    def start(self) -> None:
        """Start the step progress display."""
        self._start_time = time.time()
        if self.steps:
            self.steps[0].status = ProgressStatus.RUNNING

        if not self.no_progress and RICH_AVAILABLE:
            self._live = Live(self._render(), console=self.console, refresh_per_second=4)
            self._live.start()
        elif not self.no_progress:
            self._print_steps()

    def advance(self, error: str | None = None) -> None:
        """Advance to next step."""
        if self._current_step < len(self.steps):
            step = self.steps[self._current_step]
            if error:
                step.status = ProgressStatus.FAILED
                step.error = error
            else:
                step.status = ProgressStatus.COMPLETED

            self._current_step += 1

            if self._current_step < len(self.steps):
                self.steps[self._current_step].status = ProgressStatus.RUNNING

        if not self.no_progress:
            self._refresh()

    def skip(self) -> None:
        """Skip current step."""
        if self._current_step < len(self.steps):
            self.steps[self._current_step].status = ProgressStatus.CANCELLED
            self._current_step += 1

            if self._current_step < len(self.steps):
                self.steps[self._current_step].status = ProgressStatus.RUNNING

        if not self.no_progress:
            self._refresh()

    def complete(self) -> None:
        """Complete all remaining steps."""
        for i in range(self._current_step, len(self.steps)):
            if self.steps[i].status == ProgressStatus.RUNNING:
                self.steps[i].status = ProgressStatus.COMPLETED

        if not self.no_progress:
            self._refresh()
            if self._live:
                self._live.stop()
            self._print_summary()

    def fail(self, error: str | None = None) -> None:
        """Mark current step as failed and stop."""
        if self._current_step < len(self.steps):
            step = self.steps[self._current_step]
            step.status = ProgressStatus.FAILED
            step.error = error

        if not self.no_progress:
            self._refresh()
            if self._live:
                self._live.stop()

    def _refresh(self) -> None:
        """Refresh the display."""
        if RICH_AVAILABLE and self._live:
            self._live.update(self._render())
        elif not self.no_progress:
            self._print_steps()

    def _render(self) -> Table:
        """Render step table (Rich)."""
        table = Table(title=self.title, show_header=False, box=None)
        table.add_column("icon", width=3)
        table.add_column("step")

        for step in self.steps:
            color = {
                ProgressStatus.PENDING: "dim",
                ProgressStatus.RUNNING: "yellow",
                ProgressStatus.COMPLETED: "green",
                ProgressStatus.FAILED: "red",
                ProgressStatus.CANCELLED: "dim",
            }.get(step.status, "dim")

            text = f"[{color}]{step.icon} {step.name}[/{color}]"
            if step.error:
                text += f" [red dim]({step.error})[/red dim]"

            table.add_row("", text)

        return table

    def _print_steps(self) -> None:
        """Print steps to console (fallback)."""
        print(f"\n{self.title}")
        for step in self.steps:
            print(f"  {step.icon} {step.name}")

    def _print_summary(self) -> None:
        """Print completion summary."""
        completed = sum(1 for s in self.steps if s.status == ProgressStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == ProgressStatus.FAILED)
        total = len(self.steps)

        elapsed = time.time() - (self._start_time or 0)

        if RICH_AVAILABLE and self.console:
            if failed == 0:
                self.console.print(
                    f"\n[green]✓[/green] Completed {completed}/{total} steps in {elapsed:.1f}s"
                )
            else:
                self.console.print(
                    f"\n[red]✗[/red] {failed} step(s) failed, {completed}/{total} completed"
                )
        else:
            print(f"\nCompleted {completed}/{total} steps in {elapsed:.1f}s")


# ========== Context Managers ==========


@contextmanager
def spinner_context(
    message: str,
    no_progress: bool = False,
) -> Generator[ProgressDisplay, None, None]:
    """Context manager for spinner progress."""
    if RICH_AVAILABLE and not no_progress:
        progress = RichSpinner(message, no_progress=no_progress)
    else:
        progress = SimpleProgress(message, no_progress=no_progress)

    try:
        progress.start()
        yield progress
        progress.complete()
    except Exception as e:
        progress.fail(str(e))
        raise


@contextmanager
def progress_context(
    message: str,
    total: int | None = None,
    no_progress: bool = False,
) -> Generator[ProgressDisplay, None, None]:
    """Context manager for progress display."""
    if total is not None:
        if RICH_AVAILABLE and not no_progress:
            progress = RichProgress(message, total=total, no_progress=no_progress)
        else:
            progress = SimpleProgress(message, total=total, no_progress=no_progress)
    else:
        if RICH_AVAILABLE and not no_progress:
            progress = RichSpinner(message, no_progress=no_progress)
        else:
            progress = SimpleProgress(message, no_progress=no_progress)

    try:
        progress.start()
        yield progress
        progress.complete()
    except Exception as e:
        progress.fail(str(e))
        raise


@contextmanager
def step_context(
    steps: list[str],
    title: str = "Progress",
    no_progress: bool = False,
) -> Generator[StepProgress, None, None]:
    """Context manager for step progress."""
    progress = StepProgress(steps, title=title, no_progress=no_progress)

    try:
        progress.start()
        yield progress
        progress.complete()
    except Exception as e:
        progress.fail(str(e))
        raise


# ========== Iterable Progress ==========


T = TypeVar("T")


def track(
    iterable: Iterator[T],
    description: str = "Processing...",
    total: int | None = None,
    no_progress: bool = False,
) -> Iterator[T]:
    """Track progress while iterating."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            pass

    with progress_context(description, total=total, no_progress=no_progress) as progress:
        for item in iterable:
            yield item
            progress.update(1)


# ========== Callback-based Progress ==========


class ProgressCallback:
    """Callback class for progress updates."""

    def __init__(
        self,
        message: str = "",
        total: int | None = None,
        no_progress: bool = False,
    ) -> None:
        self.no_progress = no_progress
        if RICH_AVAILABLE and not no_progress:
            self._progress = RichProgress(message, total=total, no_progress=no_progress)
        else:
            self._progress = SimpleProgress(message, total=total, no_progress=no_progress)

    def __call__(self, current: int, total: int | None = None, message: str | None = None) -> None:
        """Update progress."""
        if total and self._progress.total != total:
            self._progress.total = total
        self._progress.set(current, message)

    def start(self) -> None:
        self._progress.start()

    def complete(self, message: str | None = None) -> None:
        self._progress.complete(message)

    def fail(self, error: str | None = None) -> None:
        self._progress.fail(error)
