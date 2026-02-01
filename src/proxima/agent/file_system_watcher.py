"""File System Watcher for Proxima Agent.

Phase 5: File System Operations & Administrative Access

Provides file system monitoring including:
- Cross-platform file watching (using watchdog when available)
- Event filtering by file patterns
- Rate limiting for event bursts
- Automatic rebuild triggers
"""

from __future__ import annotations

import fnmatch
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Set

from proxima.utils.logging import get_logger

logger = get_logger("agent.file_system_watcher")


class FileEventType(Enum):
    """Types of file system events."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    RENAMED = "renamed"


@dataclass
class FileEvent:
    """A file system event."""
    event_type: FileEventType
    path: Path
    is_directory: bool
    timestamp: datetime = field(default_factory=datetime.now)
    old_path: Optional[Path] = None  # For move/rename events
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "path": str(self.path),
            "is_directory": self.is_directory,
            "timestamp": self.timestamp.isoformat(),
            "old_path": str(self.old_path) if self.old_path else None,
        }


@dataclass
class WatchConfig:
    """Configuration for a watched directory."""
    path: Path
    recursive: bool = True
    patterns: List[str] = field(default_factory=lambda: ["*"])
    ignore_patterns: List[str] = field(default_factory=list)
    ignore_directories: bool = False
    case_sensitive: bool = True


# Type alias for event callback
EventCallback = Callable[[FileEvent], None]


class EventDebouncer:
    """Debounce rapid file system events.
    
    Groups events within a time window to prevent flooding.
    """
    
    def __init__(
        self,
        delay_seconds: float = 0.5,
        max_events: int = 100,
    ):
        """Initialize debouncer.
        
        Args:
            delay_seconds: Time to wait before emitting events
            max_events: Maximum events to buffer
        """
        self.delay_seconds = delay_seconds
        self.max_events = max_events
        
        self._pending: Dict[str, FileEvent] = {}
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._callbacks: List[Callable[[List[FileEvent]], None]] = []
    
    def add_event(self, event: FileEvent) -> None:
        """Add an event to be debounced.
        
        Args:
            event: File event
        """
        with self._lock:
            # Use path as key, later events override earlier
            key = str(event.path)
            self._pending[key] = event
            
            # Limit buffer size
            if len(self._pending) > self.max_events:
                oldest_key = next(iter(self._pending))
                del self._pending[oldest_key]
            
            # Reset timer
            if self._timer:
                self._timer.cancel()
            
            self._timer = threading.Timer(self.delay_seconds, self._emit)
            self._timer.daemon = True
            self._timer.start()
    
    def _emit(self) -> None:
        """Emit buffered events."""
        with self._lock:
            events = list(self._pending.values())
            self._pending.clear()
            self._timer = None
        
        for callback in self._callbacks:
            try:
                callback(events)
            except Exception as e:
                logger.warning(f"Event callback error: {e}")
    
    def on_events(self, callback: Callable[[List[FileEvent]], None]) -> None:
        """Register callback for debounced events.
        
        Args:
            callback: Function called with list of events
        """
        self._callbacks.append(callback)
    
    def flush(self) -> List[FileEvent]:
        """Flush and return pending events immediately."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            events = list(self._pending.values())
            self._pending.clear()
        return events


class FileSystemWatcher:
    """Watch file system for changes.
    
    Features:
    - Cross-platform support (watchdog or polling fallback)
    - Pattern-based filtering
    - Event debouncing
    - Automatic rebuild triggers
    
    Example:
        >>> watcher = FileSystemWatcher()
        >>> 
        >>> # Watch a directory
        >>> watcher.watch(
        ...     Path("src"),
        ...     patterns=["*.py", "*.pyx"],
        ...     ignore_patterns=["__pycache__/*"],
        ... )
        >>> 
        >>> # Register callback
        >>> watcher.on_event(lambda e: print(f"{e.event_type}: {e.path}"))
        >>> 
        >>> # Start watching
        >>> watcher.start()
    """
    
    # Default patterns to ignore
    DEFAULT_IGNORE_PATTERNS = [
        "*.pyc",
        "*.pyo",
        "__pycache__/*",
        ".git/*",
        ".hg/*",
        ".svn/*",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        "*.egg-info/*",
        ".tox/*",
        ".venv/*",
        "venv/*",
        "node_modules/*",
        "build/*",
        "dist/*",
    ]
    
    def __init__(
        self,
        use_polling: bool = False,
        polling_interval: float = 1.0,
        debounce_delay: float = 0.5,
    ):
        """Initialize the watcher.
        
        Args:
            use_polling: Force polling mode instead of watchdog
            polling_interval: Polling interval in seconds
            debounce_delay: Event debounce delay in seconds
        """
        self.use_polling = use_polling
        self.polling_interval = polling_interval
        
        self._watches: Dict[str, WatchConfig] = {}
        self._callbacks: List[EventCallback] = []
        self._debouncer = EventDebouncer(debounce_delay)
        
        self._running = False
        self._observer = None
        self._poll_thread: Optional[threading.Thread] = None
        self._file_states: Dict[str, float] = {}  # path -> mtime for polling
        
        # Try to import watchdog
        self._watchdog_available = self._check_watchdog()
        
        # Connect debouncer to callbacks
        self._debouncer.on_events(self._handle_debounced_events)
    
    def _check_watchdog(self) -> bool:
        """Check if watchdog is available."""
        if self.use_polling:
            return False
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            return True
        except ImportError:
            logger.info("watchdog not available, using polling fallback")
            return False
    
    def watch(
        self,
        path: Path,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        ignore_directories: bool = False,
    ) -> None:
        """Add a directory to watch.
        
        Args:
            path: Directory to watch
            patterns: Glob patterns to include
            ignore_patterns: Glob patterns to exclude
            recursive: Watch subdirectories
            ignore_directories: Ignore directory events
        """
        path = Path(path).resolve()
        
        if not path.exists():
            raise ValueError(f"Watch path does not exist: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Watch path is not a directory: {path}")
        
        config = WatchConfig(
            path=path,
            recursive=recursive,
            patterns=patterns or ["*"],
            ignore_patterns=(ignore_patterns or []) + self.DEFAULT_IGNORE_PATTERNS,
            ignore_directories=ignore_directories,
        )
        
        self._watches[str(path)] = config
        
        # If already running, add to observer
        if self._running and self._observer and self._watchdog_available:
            self._add_watchdog_watch(config)
        
        logger.info(f"Watching: {path} (patterns: {config.patterns})")
    
    def unwatch(self, path: Path) -> None:
        """Remove a directory from watching.
        
        Args:
            path: Directory to stop watching
        """
        path = Path(path).resolve()
        key = str(path)
        
        if key in self._watches:
            del self._watches[key]
            logger.info(f"Stopped watching: {path}")
    
    def on_event(self, callback: EventCallback) -> None:
        """Register an event callback.
        
        Args:
            callback: Function called for each event
        """
        self._callbacks.append(callback)
    
    def start(self) -> None:
        """Start watching."""
        if self._running:
            return
        
        self._running = True
        
        if self._watchdog_available:
            self._start_watchdog()
        else:
            self._start_polling()
        
        logger.info("File watcher started")
    
    def stop(self) -> None:
        """Stop watching."""
        if not self._running:
            return
        
        self._running = False
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5)
            self._poll_thread = None
        
        # Flush remaining events
        self._debouncer.flush()
        
        logger.info("File watcher stopped")
    
    def _start_watchdog(self) -> None:
        """Start using watchdog."""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileSystemEvent
        
        class WatchdogHandler(FileSystemEventHandler):
            def __init__(handler_self, watcher: FileSystemWatcher):
                handler_self.watcher = watcher
            
            def on_created(handler_self, event: FileSystemEvent):
                handler_self.watcher._on_watchdog_event(
                    FileEventType.CREATED, event
                )
            
            def on_modified(handler_self, event: FileSystemEvent):
                handler_self.watcher._on_watchdog_event(
                    FileEventType.MODIFIED, event
                )
            
            def on_deleted(handler_self, event: FileSystemEvent):
                handler_self.watcher._on_watchdog_event(
                    FileEventType.DELETED, event
                )
            
            def on_moved(handler_self, event: FileSystemEvent):
                handler_self.watcher._on_watchdog_event(
                    FileEventType.MOVED, event
                )
        
        self._observer = Observer()
        handler = WatchdogHandler(self)
        
        for config in self._watches.values():
            self._observer.schedule(
                handler,
                str(config.path),
                recursive=config.recursive,
            )
        
        self._observer.start()
    
    def _add_watchdog_watch(self, config: WatchConfig) -> None:
        """Add a watch to running watchdog observer."""
        if not self._observer:
            return
        
        from watchdog.events import FileSystemEventHandler
        
        # Create handler (simplified, reuses existing logic)
        class Handler(FileSystemEventHandler):
            def __init__(handler_self, watcher):
                handler_self.watcher = watcher
            
            def on_any_event(handler_self, event):
                event_type = {
                    "created": FileEventType.CREATED,
                    "modified": FileEventType.MODIFIED,
                    "deleted": FileEventType.DELETED,
                    "moved": FileEventType.MOVED,
                }.get(event.event_type)
                if event_type:
                    handler_self.watcher._on_watchdog_event(event_type, event)
        
        self._observer.schedule(
            Handler(self),
            str(config.path),
            recursive=config.recursive,
        )
    
    def _on_watchdog_event(
        self,
        event_type: FileEventType,
        watchdog_event: Any,
    ) -> None:
        """Handle a watchdog event."""
        path = Path(watchdog_event.src_path)
        
        # Check if path matches any watch config
        if not self._should_process(path, watchdog_event.is_directory):
            return
        
        event = FileEvent(
            event_type=event_type,
            path=path,
            is_directory=watchdog_event.is_directory,
        )
        
        # Handle move events
        if event_type == FileEventType.MOVED and hasattr(watchdog_event, "dest_path"):
            event.old_path = path
            event.path = Path(watchdog_event.dest_path)
        
        self._debouncer.add_event(event)
    
    def _start_polling(self) -> None:
        """Start using polling fallback."""
        # Build initial file state
        self._file_states.clear()
        for config in self._watches.values():
            self._scan_directory(config)
        
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
    
    def _poll_loop(self) -> None:
        """Polling loop for file changes."""
        while self._running:
            try:
                for config in list(self._watches.values()):
                    self._check_changes(config)
            except Exception as e:
                logger.warning(f"Poll error: {e}")
            
            time.sleep(self.polling_interval)
    
    def _scan_directory(self, config: WatchConfig) -> None:
        """Scan directory and record file states."""
        try:
            if config.recursive:
                paths = config.path.rglob("*")
            else:
                paths = config.path.glob("*")
            
            for path in paths:
                if self._should_process(path, path.is_dir()):
                    try:
                        self._file_states[str(path)] = path.stat().st_mtime
                    except (OSError, FileNotFoundError):
                        pass
        except Exception as e:
            logger.warning(f"Scan error for {config.path}: {e}")
    
    def _check_changes(self, config: WatchConfig) -> None:
        """Check for file changes in a watched directory."""
        current_files: Set[str] = set()
        
        try:
            if config.recursive:
                paths = list(config.path.rglob("*"))
            else:
                paths = list(config.path.glob("*"))
        except Exception:
            return
        
        for path in paths:
            if not self._should_process(path, path.is_dir()):
                continue
            
            path_str = str(path)
            current_files.add(path_str)
            
            try:
                mtime = path.stat().st_mtime
            except (OSError, FileNotFoundError):
                continue
            
            if path_str not in self._file_states:
                # New file
                self._file_states[path_str] = mtime
                self._debouncer.add_event(FileEvent(
                    event_type=FileEventType.CREATED,
                    path=path,
                    is_directory=path.is_dir(),
                ))
            elif self._file_states[path_str] != mtime:
                # Modified file
                self._file_states[path_str] = mtime
                self._debouncer.add_event(FileEvent(
                    event_type=FileEventType.MODIFIED,
                    path=path,
                    is_directory=path.is_dir(),
                ))
        
        # Check for deleted files
        deleted = set(self._file_states.keys()) - current_files
        for path_str in deleted:
            if any(path_str.startswith(str(w.path)) for w in self._watches.values()):
                del self._file_states[path_str]
                self._debouncer.add_event(FileEvent(
                    event_type=FileEventType.DELETED,
                    path=Path(path_str),
                    is_directory=False,  # Can't know for deleted
                ))
    
    def _should_process(self, path: Path, is_directory: bool) -> bool:
        """Check if a path should be processed based on filters."""
        # Find matching watch config
        config = None
        for watch_config in self._watches.values():
            try:
                path.relative_to(watch_config.path)
                config = watch_config
                break
            except ValueError:
                continue
        
        if config is None:
            return False
        
        # Skip directories if configured
        if is_directory and config.ignore_directories:
            return False
        
        name = path.name
        rel_path = str(path.relative_to(config.path))
        
        # Check ignore patterns
        for pattern in config.ignore_patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
                return False
        
        # Check include patterns
        for pattern in config.patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
                return True
        
        return False
    
    def _handle_debounced_events(self, events: List[FileEvent]) -> None:
        """Handle debounced events."""
        for event in events:
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Event callback error: {e}")
    
    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running
    
    @property
    def watched_paths(self) -> List[Path]:
        """Get list of watched paths."""
        return [config.path for config in self._watches.values()]


class RebuildTrigger:
    """Trigger rebuilds based on file changes.
    
    Example:
        >>> trigger = RebuildTrigger(watcher)
        >>> 
        >>> # Register backend to watch
        >>> trigger.watch_backend(
        ...     "lret_cirq",
        ...     source_dir=Path("src/lret_cirq"),
        ...     patterns=["*.py", "*.pyx", "*.cpp"],
        ... )
        >>> 
        >>> # Register rebuild callback
        >>> trigger.on_rebuild(lambda backend: print(f"Rebuild: {backend}"))
    """
    
    def __init__(
        self,
        watcher: FileSystemWatcher,
        cooldown_seconds: float = 5.0,
    ):
        """Initialize rebuild trigger.
        
        Args:
            watcher: File system watcher
            cooldown_seconds: Minimum time between rebuilds
        """
        self.watcher = watcher
        self.cooldown_seconds = cooldown_seconds
        
        self._backends: Dict[str, WatchConfig] = {}
        self._last_rebuild: Dict[str, float] = {}
        self._rebuild_callbacks: List[Callable[[str, List[FileEvent]], None]] = []
        
        # Subscribe to watcher events
        self.watcher.on_event(self._on_file_event)
    
    def watch_backend(
        self,
        backend_name: str,
        source_dir: Path,
        patterns: Optional[List[str]] = None,
    ) -> None:
        """Watch a backend for changes.
        
        Args:
            backend_name: Name of the backend
            source_dir: Source directory
            patterns: File patterns to watch
        """
        source_dir = Path(source_dir).resolve()
        
        config = WatchConfig(
            path=source_dir,
            patterns=patterns or ["*.py", "*.pyx", "*.cpp", "*.cu", "*.h"],
            recursive=True,
        )
        
        self._backends[backend_name] = config
        self._last_rebuild[backend_name] = 0
        
        # Add to watcher
        self.watcher.watch(
            source_dir,
            patterns=config.patterns,
            recursive=config.recursive,
        )
        
        logger.info(f"Watching backend {backend_name} at {source_dir}")
    
    def on_rebuild(
        self,
        callback: Callable[[str, List[FileEvent]], None],
    ) -> None:
        """Register rebuild callback.
        
        Args:
            callback: Function called with (backend_name, events)
        """
        self._rebuild_callbacks.append(callback)
    
    def _on_file_event(self, event: FileEvent) -> None:
        """Handle a file event."""
        # Find matching backend
        for backend_name, config in self._backends.items():
            try:
                event.path.relative_to(config.path)
            except ValueError:
                continue
            
            # Check cooldown
            now = time.time()
            if now - self._last_rebuild.get(backend_name, 0) < self.cooldown_seconds:
                return
            
            self._last_rebuild[backend_name] = now
            
            # Trigger rebuild
            for callback in self._rebuild_callbacks:
                try:
                    callback(backend_name, [event])
                except Exception as e:
                    logger.warning(f"Rebuild callback error: {e}")
            
            break


def get_file_system_watcher(
    use_polling: bool = False,
) -> FileSystemWatcher:
    """Get a FileSystemWatcher instance."""
    return FileSystemWatcher(use_polling=use_polling)
