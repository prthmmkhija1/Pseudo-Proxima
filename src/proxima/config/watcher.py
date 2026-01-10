"""Configuration file watcher for auto-reload.

Provides file system monitoring to detect configuration changes
and trigger automatic reloads or notifications.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from weakref import WeakSet


class WatchEvent(Enum):
    """Types of file watch events."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChange:
    """Information about a file change."""

    path: Path
    event: WatchEvent
    timestamp: datetime = field(default_factory=datetime.now)
    old_path: Path | None = None  # For MOVED events
    file_hash: str | None = None  # Hash of new content

    def __str__(self) -> str:
        if self.event == WatchEvent.MOVED and self.old_path:
            return f"{self.event.value}: {self.old_path} → {self.path}"
        return f"{self.event.value}: {self.path}"


# Type for change callback functions
ChangeCallback = Callable[[FileChange], None]
AsyncChangeCallback = Callable[[FileChange], Any]  # Coroutine


# =============================================================================
# ABSTRACT WATCHER
# =============================================================================


class FileWatcher(ABC):
    """Abstract base class for file watchers."""

    def __init__(self) -> None:
        self._callbacks: list[ChangeCallback] = []
        self._async_callbacks: list[AsyncChangeCallback] = []
        self._running = False

    @abstractmethod
    def start(self) -> None:
        """Start watching for changes."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop watching for changes."""
        pass

    @abstractmethod
    def add_path(self, path: Path) -> None:
        """Add a path to watch."""
        pass

    @abstractmethod
    def remove_path(self, path: Path) -> None:
        """Remove a path from watching."""
        pass

    def on_change(self, callback: ChangeCallback) -> None:
        """Register a callback for file changes."""
        self._callbacks.append(callback)

    def on_change_async(self, callback: AsyncChangeCallback) -> None:
        """Register an async callback for file changes."""
        self._async_callbacks.append(callback)

    def _notify(self, change: FileChange) -> None:
        """Notify all registered callbacks of a change."""
        for callback in self._callbacks:
            try:
                callback(change)
            except Exception:
                pass  # Don't let callback errors stop notifications

        # Handle async callbacks
        if self._async_callbacks:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    for callback in self._async_callbacks:
                        asyncio.create_task(callback(change))
                else:
                    for callback in self._async_callbacks:
                        loop.run_until_complete(callback(change))
            except RuntimeError:
                pass  # No event loop available

    @property
    def is_running(self) -> bool:
        return self._running


# =============================================================================
# POLLING WATCHER (Cross-platform fallback)
# =============================================================================


class PollingWatcher(FileWatcher):
    """File watcher using polling (works on all platforms).

    Less efficient than native watchers but universally compatible.
    """

    def __init__(self, poll_interval: float = 1.0) -> None:
        super().__init__()
        self._poll_interval = poll_interval
        self._watched_paths: dict[Path, dict[str, Any]] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def add_path(self, path: Path) -> None:
        """Add a path to watch."""
        path = path.resolve()
        self._watched_paths[path] = self._get_file_info(path)

    def remove_path(self, path: Path) -> None:
        """Remove a path from watching."""
        path = path.resolve()
        self._watched_paths.pop(path, None)

    def start(self) -> None:
        """Start the polling thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        """Stop the polling thread."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            for path, old_info in list(self._watched_paths.items()):
                new_info = self._get_file_info(path)

                change = self._detect_change(path, old_info, new_info)
                if change:
                    self._watched_paths[path] = new_info
                    self._notify(change)

            self._stop_event.wait(self._poll_interval)

    def _get_file_info(self, path: Path) -> dict[str, Any]:
        """Get file information for comparison."""
        if not path.exists():
            return {"exists": False}

        try:
            stat = path.stat()
            return {
                "exists": True,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "hash": self._compute_hash(path),
            }
        except Exception:
            return {"exists": False}

    def _compute_hash(self, path: Path) -> str | None:
        """Compute hash of file contents."""
        try:
            content = path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return None

    def _detect_change(
        self,
        path: Path,
        old_info: dict[str, Any],
        new_info: dict[str, Any],
    ) -> FileChange | None:
        """Detect if a change occurred."""
        old_exists = old_info.get("exists", False)
        new_exists = new_info.get("exists", False)

        if not old_exists and new_exists:
            return FileChange(path, WatchEvent.CREATED, file_hash=new_info.get("hash"))

        if old_exists and not new_exists:
            return FileChange(path, WatchEvent.DELETED)

        if old_exists and new_exists:
            # Check if modified
            if (
                old_info.get("mtime") != new_info.get("mtime")
                or old_info.get("size") != new_info.get("size")
                or old_info.get("hash") != new_info.get("hash")
            ):
                return FileChange(path, WatchEvent.MODIFIED, file_hash=new_info.get("hash"))

        return None


# =============================================================================
# WATCHDOG WATCHER (More efficient, requires watchdog package)
# =============================================================================


class WatchdogWatcher(FileWatcher):
    """File watcher using the watchdog library.

    More efficient than polling, uses native OS file system events.
    Requires: pip install watchdog
    """

    def __init__(self) -> None:
        super().__init__()
        self._observer: Any = None
        self._handlers: dict[Path, Any] = {}
        self._watchdog_available = self._check_watchdog()

    def _check_watchdog(self) -> bool:
        """Check if watchdog is available."""
        try:
            import watchdog  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        return self._watchdog_available

    def add_path(self, path: Path) -> None:
        """Add a path to watch."""
        if not self._watchdog_available:
            raise RuntimeError("watchdog package not installed")

        from watchdog.events import FileSystemEvent, FileSystemEventHandler

        path = path.resolve()

        # Create handler for this path
        watcher = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event: FileSystemEvent) -> None:
                if not event.is_directory:
                    change = FileChange(Path(event.src_path), WatchEvent.CREATED)
                    watcher._notify(change)

            def on_modified(self, event: FileSystemEvent) -> None:
                if not event.is_directory:
                    change = FileChange(Path(event.src_path), WatchEvent.MODIFIED)
                    watcher._notify(change)

            def on_deleted(self, event: FileSystemEvent) -> None:
                if not event.is_directory:
                    change = FileChange(Path(event.src_path), WatchEvent.DELETED)
                    watcher._notify(change)

            def on_moved(self, event: FileSystemEvent) -> None:
                if not event.is_directory:
                    change = FileChange(
                        Path(event.dest_path),
                        WatchEvent.MOVED,
                        old_path=Path(event.src_path),
                    )
                    watcher._notify(change)

        handler = Handler()
        self._handlers[path] = handler

        if self._observer is not None:
            # Watch the parent directory to catch file changes
            watch_path = path.parent if path.is_file() else path
            self._observer.schedule(handler, str(watch_path), recursive=False)

    def remove_path(self, path: Path) -> None:
        """Remove a path from watching."""
        path = path.resolve()
        self._handlers.pop(path, None)

    def start(self) -> None:
        """Start the watchdog observer."""
        if not self._watchdog_available:
            raise RuntimeError("watchdog package not installed")

        if self._running:
            return

        from watchdog.observers import Observer

        self._observer = Observer()

        # Schedule all registered paths
        for path, handler in self._handlers.items():
            watch_path = path.parent if path.is_file() else path
            self._observer.schedule(handler, str(watch_path), recursive=False)

        self._observer.start()
        self._running = True

    def stop(self) -> None:
        """Stop the watchdog observer."""
        if not self._running or self._observer is None:
            return

        self._observer.stop()
        self._observer.join(timeout=2.0)
        self._observer = None
        self._running = False


# =============================================================================
# CONFIG WATCHER
# =============================================================================


class ConfigWatcher:
    """Specialized watcher for configuration files with auto-reload.

    Watches configuration files and automatically reloads settings
    when changes are detected.
    """

    def __init__(
        self,
        config_paths: list[Path] | None = None,
        auto_reload: bool = True,
        debounce_seconds: float = 0.5,
    ) -> None:
        self._config_paths = config_paths or []
        self._auto_reload = auto_reload
        self._debounce_seconds = debounce_seconds

        # Try watchdog first, fall back to polling
        self._watcher = self._create_watcher()

        # Debounce tracking
        self._last_change_time: float = 0
        self._pending_reload = False

        # Callbacks
        self._reload_callbacks: list[Callable[[Path], None]] = []
        self._error_callbacks: list[Callable[[Exception], None]] = []

        # Set up internal change handler
        self._watcher.on_change(self._handle_change)

    def _create_watcher(self) -> FileWatcher:
        """Create the best available watcher."""
        watchdog_watcher = WatchdogWatcher()
        if watchdog_watcher.is_available:
            return watchdog_watcher
        return PollingWatcher()

    def add_config(self, path: Path) -> None:
        """Add a configuration file to watch."""
        path = path.resolve()
        if path not in self._config_paths:
            self._config_paths.append(path)
            self._watcher.add_path(path)

    def remove_config(self, path: Path) -> None:
        """Remove a configuration file from watching."""
        path = path.resolve()
        if path in self._config_paths:
            self._config_paths.remove(path)
            self._watcher.remove_path(path)

    def start(self) -> None:
        """Start watching configuration files."""
        for path in self._config_paths:
            self._watcher.add_path(path)
        self._watcher.start()

    def stop(self) -> None:
        """Stop watching configuration files."""
        self._watcher.stop()

    def on_reload(self, callback: Callable[[Path], None]) -> None:
        """Register a callback for configuration reloads."""
        self._reload_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register a callback for reload errors."""
        self._error_callbacks.append(callback)

    def _handle_change(self, change: FileChange) -> None:
        """Handle a file change event."""
        # Only handle our config files
        if change.path not in self._config_paths:
            return

        # Ignore deletions (don't reload if file is deleted)
        if change.event == WatchEvent.DELETED:
            return

        # Debounce rapid changes
        now = time.time()
        if now - self._last_change_time < self._debounce_seconds:
            self._pending_reload = True
            return

        self._last_change_time = now
        self._pending_reload = False

        if self._auto_reload:
            self._trigger_reload(change.path)

    def _trigger_reload(self, path: Path) -> None:
        """Trigger a configuration reload."""
        try:
            # Notify callbacks
            for callback in self._reload_callbacks:
                callback(path)
        except Exception as e:
            for error_callback in self._error_callbacks:
                error_callback(e)

    @property
    def is_running(self) -> bool:
        return self._watcher.is_running

    @property
    def watched_paths(self) -> list[Path]:
        return list(self._config_paths)


# =============================================================================
# INTEGRATION WITH CONFIG SERVICE
# =============================================================================


class WatchedConfigService:
    """Config service wrapper with automatic file watching and reload.

    Wraps ConfigService to add automatic reload when files change.
    """

    def __init__(self, config_service: Any) -> None:
        self._config_service = config_service
        self._watcher: ConfigWatcher | None = None
        self._settings_cache: Any = None
        self._change_listeners: WeakSet = WeakSet()

    def enable_watch(self) -> None:
        """Enable configuration file watching."""
        if self._watcher is not None:
            return

        # Get paths from config service
        paths = [
            self._config_service.default_config_path,
            self._config_service.project_config_path,
            self._config_service.user_config_path,
        ]

        self._watcher = ConfigWatcher(
            config_paths=[p for p in paths if p.exists()],
            auto_reload=True,
        )

        self._watcher.on_reload(self._on_config_reload)
        self._watcher.start()

    def disable_watch(self) -> None:
        """Disable configuration file watching."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    def _on_config_reload(self, path: Path) -> None:
        """Handle configuration file reload."""
        # Clear cache
        self._settings_cache = None

        # Reload settings
        try:
            self._settings_cache = self._config_service.load()
        except Exception:
            pass

    def load(self, **kwargs: Any) -> Any:
        """Load settings (with caching)."""
        if self._settings_cache is None:
            self._settings_cache = self._config_service.load(**kwargs)
        return self._settings_cache

    def __getattr__(self, name: str) -> Any:
        """Proxy to underlying config service."""
        return getattr(self._config_service, name)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_config_watcher(
    paths: list[Path],
    on_change: ChangeCallback | None = None,
) -> ConfigWatcher:
    """Create and configure a config watcher.

    Args:
        paths: Configuration file paths to watch
        on_change: Optional callback for changes

    Returns:
        Configured ConfigWatcher instance
    """
    watcher = ConfigWatcher(config_paths=paths)

    if on_change:
        watcher._watcher.on_change(on_change)

    return watcher


def watch_config_file(
    path: Path,
    callback: Callable[[Path], None],
) -> ConfigWatcher:
    """Watch a single configuration file for changes.

    Args:
        path: Configuration file path
        callback: Function to call when file changes

    Returns:
        Running ConfigWatcher instance
    """
    watcher = ConfigWatcher(config_paths=[path])
    watcher.on_reload(callback)
    watcher.start()
    return watcher
