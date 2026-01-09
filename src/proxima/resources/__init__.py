"""Resource management modules."""

from .consent import (
    CATEGORY_CONFIGS,
    ConsentCategory,
    ConsentCategoryConfig,
    ConsentCheckResult,
    ConsentDeniedException,
    ConsentLevel,
    ConsentManager,
    ConsentPrompt,
    ConsentRecord,
    ConsentRequest,
    ConsentResponse,
    DefaultConsentPrompt,
    requires_consent,
)
from .control import (
    AbortException,
    CheckpointData,
    CheckpointManager,
    ControlEvent,
    ControlSignal,
    ControlState,
    ExecutionController,
    PauseException,
)
from .monitor import (
    CPUMonitor,
    MemoryAlert,
    MemoryCheckResult,
    MemoryEstimate,
    MemoryEstimator,
    MemoryLevel,
    MemoryMonitor,
    MemorySnapshot,
    MemoryThresholds,
    ResourceMonitor,
)
from .session import Session, SessionManager, SessionStatus
from .timer import (
    DisplayController,
    DisplayUpdate,
    ETACalculator,
    ExecutionTimer,
    ProgressTracker,
    StageInfo,
    UpdateReason,
)

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
