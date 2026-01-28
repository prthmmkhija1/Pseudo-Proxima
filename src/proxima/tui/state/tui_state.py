"""Central TUI State - Reactive state management for Proxima TUI.

Holds all state information for the TUI, enabling reactive updates.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class StageInfo:
    """Information about an execution stage."""
    
    name: str
    index: int
    status: str = "pending"  # pending, current, done, error
    duration_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class BackendStatus:
    """Status information for a backend."""
    
    name: str
    status: str = "unknown"  # healthy, degraded, unhealthy, unknown
    response_time_ms: Optional[float] = None
    simulator: Optional[str] = None
    available: bool = False
    error_message: Optional[str] = None


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    
    id: str
    stage_index: int
    timestamp: datetime
    size_bytes: int = 0


@dataclass
class ResultInfo:
    """Information about an execution result."""
    
    id: str
    task: str
    backend: str
    status: str
    timestamp: datetime
    duration_ms: float = 0.0
    file_path: Optional[str] = None
    preview: Optional[Dict[str, Any]] = None


@dataclass
class TUIState:
    """Central state for the Proxima TUI.
    
    This dataclass holds all state information that the TUI needs to render.
    Changes to this state trigger reactive updates in the UI.
    """
    
    # ==================== Execution State ====================
    execution_status: str = "IDLE"  # ExecutionState enum value
    current_task: Optional[str] = None
    current_task_id: Optional[str] = None
    current_backend: Optional[str] = None
    current_simulator: Optional[str] = None
    qubits: int = 0
    shots: int = 0
    
    # ==================== Progress State ====================
    progress_percent: float = 0.0
    current_stage: str = ""
    stage_index: int = 0
    total_stages: int = 0
    elapsed_ms: float = 0.0
    eta_ms: Optional[float] = None
    
    # Stage History
    completed_stages: List[StageInfo] = field(default_factory=list)
    all_stages: List[StageInfo] = field(default_factory=list)
    
    # ==================== Session State ====================
    active_session_id: Optional[str] = None
    session_title: Optional[str] = None
    session_status: str = "active"  # active, paused, completed, error
    working_directory: Optional[str] = None
    
    # ==================== Backend State ====================
    backend_statuses: Dict[str, BackendStatus] = field(default_factory=dict)
    active_backend_name: Optional[str] = None
    default_backend: str = "cirq"
    
    # ==================== Memory State ====================
    memory_percent: float = 0.0
    memory_level: str = "OK"  # OK, INFO, WARNING, CRITICAL, ABORT
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    
    # ==================== Checkpoint State ====================
    latest_checkpoint: Optional[CheckpointInfo] = None
    checkpoint_count: int = 0
    last_checkpoint_time: Optional[datetime] = None
    rollback_available: bool = False
    
    # ==================== LLM State ====================
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_connected: bool = False
    thinking_enabled: bool = False
    thinking_panel_visible: bool = False  # Whether AI thinking panel is shown
    
    # Token/Cost Tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    
    # AI Thinking State
    current_thinking_content: str = ""
    thinking_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # ==================== Results State ====================
    latest_result: Optional[ResultInfo] = None
    result_history: List[ResultInfo] = field(default_factory=list)
    
    # ==================== UI State ====================
    current_screen: str = "dashboard"
    sidebar_compact: bool = False
    log_visible: bool = True
    focus_area: str = "main"  # main, sidebar, input, dialog
    
    # Dialog state
    active_dialog: Optional[str] = None
    dialog_data: Optional[Dict[str, Any]] = None
    
    # ==================== Log State ====================
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    max_log_entries: int = 1000
    
    def update_execution_status(self, status: str) -> None:
        """Update execution status."""
        self.execution_status = status
    
    def update_progress(
        self,
        percent: float,
        stage: str = "",
        stage_index: int = 0,
        total_stages: int = 0,
        elapsed_ms: float = 0.0,
        eta_ms: Optional[float] = None,
    ) -> None:
        """Update progress information."""
        self.progress_percent = percent
        self.current_stage = stage
        self.stage_index = stage_index
        self.total_stages = total_stages
        self.elapsed_ms = elapsed_ms
        self.eta_ms = eta_ms
    
    def update_memory(
        self,
        percent: float,
        level: str,
        used_mb: float,
        available_mb: float,
    ) -> None:
        """Update memory information."""
        self.memory_percent = percent
        self.memory_level = level
        self.memory_used_mb = used_mb
        self.memory_available_mb = available_mb
    
    def add_log_entry(
        self,
        level: str,
        message: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add a log entry."""
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
        }
        
        self.log_entries.append(entry)
        
        # Trim log if too long
        if len(self.log_entries) > self.max_log_entries:
            self.log_entries = self.log_entries[-self.max_log_entries:]
    
    def clear_execution(self) -> None:
        """Clear execution-related state."""
        self.execution_status = "IDLE"
        self.current_task = None
        self.current_task_id = None
        self.progress_percent = 0.0
        self.current_stage = ""
        self.stage_index = 0
        self.total_stages = 0
        self.elapsed_ms = 0.0
        self.eta_ms = None
        self.completed_stages = []
        self.all_stages = []
        self.latest_checkpoint = None
        self.checkpoint_count = 0
        self.rollback_available = False
    
    def set_backend_status(self, name: str, status: BackendStatus) -> None:
        """Set status for a backend."""
        self.backend_statuses[name] = status
    
    def get_backend_status(self, name: str) -> Optional[BackendStatus]:
        """Get status for a backend."""
        return self.backend_statuses.get(name)
    
    def get_healthy_backends(self) -> List[str]:
        """Get list of healthy backend names."""
        return [
            name for name, status in self.backend_statuses.items()
            if status.status == "healthy"
        ]
    
    def get_formatted_elapsed(self) -> str:
        """Get formatted elapsed time string."""
        seconds = self.elapsed_ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    
    def get_formatted_eta(self) -> str:
        """Get formatted ETA string."""
        if self.eta_ms is None:
            return "--"
        seconds = self.eta_ms / 1000
        if seconds < 60:
            return f"~{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"~{minutes}m {secs:.0f}s"
    
    def get_formatted_memory(self) -> str:
        """Get formatted memory usage string."""
        return f"{self.memory_used_mb:.1f} GB / {self.memory_available_mb:.1f} GB"
    
    def get_token_summary(self) -> str:
        """Get token usage summary string."""
        total = self.prompt_tokens + self.completion_tokens
        return f"{self.progress_percent:.0f}% ({total/1000:.1f}K) ${self.total_cost:.2f}"
