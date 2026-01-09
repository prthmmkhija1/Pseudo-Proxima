"""Resource management modules."""

from .timer import (
    ExecutionTimer,
    ProgressTracker,
    ETACalculator,
    StageInfo,
    DisplayController,
    DisplayUpdate,
    UpdateReason,
)
from .monitor import (
    ResourceMonitor,
    MemoryMonitor,
    CPUMonitor,
    MemoryLevel,
    MemoryThresholds,
    MemorySnapshot,
    MemoryAlert,
    MemoryEstimator,
    MemoryEstimate,
    MemoryCheckResult,
)
from .control import (
    ExecutionController,
    ControlState,
    ControlSignal,
    ControlEvent,
    CheckpointData,
    CheckpointManager,
    AbortException,
    PauseException,
)
from .consent import (
    ConsentManager,
    ConsentLevel,
    ConsentRecord,
    ConsentCategory,
    ConsentCategoryConfig,
    ConsentResponse,
    ConsentRequest,
    ConsentCheckResult,
    ConsentPrompt,
    DefaultConsentPrompt,
    ConsentDeniedException,
    requires_consent,
    CATEGORY_CONFIGS,
)
from .session import Session, SessionManager, SessionStatus

__all__ = [
    # Timer (Step 4.2)
    "ExecutionTimer",
    "ProgressTracker",
    "ETACalculator",
    "StageInfo",
    "DisplayController",
    "DisplayUpdate",
    "UpdateReason",
    # Monitor (Step 4.1)
    "ResourceMonitor",
    "MemoryMonitor",
    "CPUMonitor",
    "MemoryLevel",
    "MemoryThresholds",
    "MemorySnapshot",
    "MemoryAlert",
    "MemoryEstimator",
    "MemoryEstimate",
    "MemoryCheckResult",
    # Control (Step 4.3)
    "ExecutionController",
    "ControlState",
    "ControlSignal",
    "ControlEvent",
    "CheckpointData",
    "CheckpointManager",
    "AbortException",
    "PauseException",
    # Consent (Step 4.4)
    "ConsentManager",
    "ConsentLevel",
    "ConsentRecord",
    "ConsentCategory",
    "ConsentCategoryConfig",
    "ConsentResponse",
    "ConsentRequest",
    "ConsentCheckResult",
    "ConsentPrompt",
    "DefaultConsentPrompt",
    "ConsentDeniedException",
    "requires_consent",
    "CATEGORY_CONFIGS",
    # Session
    "Session",
    "SessionManager",
    "SessionStatus",
]
