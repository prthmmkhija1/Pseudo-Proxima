"""Safety and Consent System for Proxima Agent.

Provides comprehensive safety features for the AI agent:
- Consent management for sensitive operations
- Operation checkpoints for undo/redo/rollback
- Safe execution boundaries
- Audit logging for all operations

This module ensures the agent operates within user-defined boundaries
and allows full control over what operations are permitted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.safety")


class ConsentType(Enum):
    """Types of consent requests."""
    COMMAND_EXECUTION = "command_execution"
    FILE_MODIFICATION = "file_modification"
    GIT_OPERATION = "git_operation"
    ADMIN_ACCESS = "admin_access"
    NETWORK_ACCESS = "network_access"
    BACKEND_MODIFICATION = "backend_modification"
    SYSTEM_CHANGE = "system_change"
    BULK_OPERATION = "bulk_operation"


class ConsentDecision(Enum):
    """User consent decisions."""
    APPROVED = auto()
    DENIED = auto()
    APPROVED_ONCE = auto()
    APPROVED_SESSION = auto()
    APPROVED_ALWAYS = auto()


@dataclass
class ConsentRequest:
    """A request for user consent."""
    id: str
    consent_type: ConsentType
    operation: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_name: Optional[str] = None
    risk_level: str = "medium"  # low, medium, high, critical
    reversible: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "consent_type": self.consent_type.value,
            "operation": self.operation,
            "description": self.description,
            "details": self.details,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "risk_level": self.risk_level,
            "reversible": self.reversible,
        }
    
    def get_display_message(self) -> str:
        """Get human-readable consent message."""
        risk_emoji = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "⚠️",
            "critical": "🚨",
        }
        emoji = risk_emoji.get(self.risk_level, "⚠️")
        
        lines = [
            f"{emoji} **{self.operation}**",
            "",
            self.description,
        ]
        
        if self.details:
            lines.append("")
            lines.append("Details:")
            for key, value in self.details.items():
                lines.append(f"  - {key}: {value}")
        
        if not self.reversible:
            lines.append("")
            lines.append("⚠️ This operation cannot be undone.")
        
        return "\n".join(lines)


@dataclass
class ConsentResponse:
    """Response to a consent request."""
    request_id: str
    decision: ConsentDecision
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_message: Optional[str] = None
    
    @property
    def approved(self) -> bool:
        """Check if consent was granted."""
        return self.decision in (
            ConsentDecision.APPROVED,
            ConsentDecision.APPROVED_ONCE,
            ConsentDecision.APPROVED_SESSION,
            ConsentDecision.APPROVED_ALWAYS,
        )


@dataclass
class OperationCheckpoint:
    """Checkpoint for rollback/undo support."""
    id: str
    operation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Optional[Dict[str, Any]] = None
    file_backups: Dict[str, str] = field(default_factory=dict)  # path -> backup_path
    metadata: Dict[str, Any] = field(default_factory=dict)
    can_rollback: bool = True
    rolled_back: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "file_backups": self.file_backups,
            "metadata": self.metadata,
            "can_rollback": self.can_rollback,
            "rolled_back": self.rolled_back,
        }


class RollbackManager:
    """Manages checkpoints and rollback operations.
    
    Provides undo/redo/rollback capabilities for agent operations:
    - File modifications
    - Configuration changes
    - Backend code changes
    
    Example:
        >>> manager = RollbackManager()
        >>> checkpoint = manager.create_checkpoint("modify_file", files=["config.py"])
        >>> # ... perform modification ...
        >>> manager.complete_checkpoint(checkpoint.id)
        >>> # Later, if needed:
        >>> manager.rollback(checkpoint.id)
    """
    
    def __init__(
        self,
        backup_dir: Optional[str] = None,
        max_checkpoints: int = 100,
        max_backup_age_hours: int = 24,
    ):
        """Initialize rollback manager.
        
        Args:
            backup_dir: Directory for storing backups
            max_checkpoints: Maximum number of checkpoints to retain
            max_backup_age_hours: Maximum age of backups before cleanup
        """
        self.backup_dir = Path(backup_dir) if backup_dir else Path.home() / ".proxima" / "backups"
        self.max_checkpoints = max_checkpoints
        self.max_backup_age_hours = max_backup_age_hours
        
        self._checkpoints: Dict[str, OperationCheckpoint] = {}
        self._checkpoint_order: List[str] = []  # For undo order
        self._redo_stack: List[str] = []
        self._lock = threading.Lock()
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RollbackManager initialized with backup dir: {self.backup_dir}")
    
    def _generate_id(self) -> str:
        """Generate a unique checkpoint ID."""
        timestamp = int(time.time() * 1000)
        random_part = os.urandom(4).hex()
        return f"ckpt_{timestamp}_{random_part}"
    
    def _backup_file(self, file_path: str) -> Optional[str]:
        """Create a backup of a file.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path to backup file, or None if failed
        """
        try:
            source = Path(file_path)
            if not source.exists():
                return None
            
            # Create hash-based backup name
            file_hash = hashlib.md5(str(source).encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source.stem}_{timestamp}_{file_hash}{source.suffix}.bak"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(source, backup_path)
            logger.debug(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to backup file {file_path}: {e}")
            return None
    
    def create_checkpoint(
        self,
        operation: str,
        files: Optional[List[str]] = None,
        state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OperationCheckpoint:
        """Create a checkpoint before an operation.
        
        Args:
            operation: Description of the operation
            files: Files to backup
            state: Additional state to preserve
            metadata: Checkpoint metadata
            
        Returns:
            Created checkpoint
        """
        with self._lock:
            # Cleanup old checkpoints if needed
            if len(self._checkpoints) >= self.max_checkpoints:
                self._cleanup_old_checkpoints()
            
            checkpoint_id = self._generate_id()
            file_backups = {}
            
            # Backup files
            if files:
                for file_path in files:
                    backup_path = self._backup_file(file_path)
                    if backup_path:
                        file_backups[file_path] = backup_path
            
            checkpoint = OperationCheckpoint(
                id=checkpoint_id,
                operation=operation,
                state_before=state or {},
                file_backups=file_backups,
                metadata=metadata or {},
            )
            
            self._checkpoints[checkpoint_id] = checkpoint
            self._checkpoint_order.append(checkpoint_id)
            self._redo_stack.clear()  # Clear redo stack on new operation
            
            logger.info(f"Created checkpoint {checkpoint_id} for: {operation}")
            return checkpoint
    
    def complete_checkpoint(
        self,
        checkpoint_id: str,
        state_after: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a checkpoint as complete with the resulting state.
        
        Args:
            checkpoint_id: ID of checkpoint to complete
            state_after: State after the operation
            
        Returns:
            True if completed successfully
        """
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint:
                checkpoint.state_after = state_after
                return True
            return False
    
    def rollback(self, checkpoint_id: str) -> Tuple[bool, str]:
        """Rollback to a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return False, f"Checkpoint not found: {checkpoint_id}"
            
            if not checkpoint.can_rollback:
                return False, f"Checkpoint cannot be rolled back: {checkpoint.operation}"
            
            if checkpoint.rolled_back:
                return False, "Checkpoint already rolled back"
            
            # Restore files from backups
            errors = []
            for original_path, backup_path in checkpoint.file_backups.items():
                try:
                    if os.path.exists(backup_path):
                        shutil.copy2(backup_path, original_path)
                        logger.info(f"Restored: {original_path}")
                    else:
                        errors.append(f"Backup not found: {backup_path}")
                except Exception as e:
                    errors.append(f"Failed to restore {original_path}: {e}")
            
            checkpoint.rolled_back = True
            
            if errors:
                return False, "Partial rollback: " + "; ".join(errors)
            
            return True, f"Successfully rolled back: {checkpoint.operation}"
    
    def undo(self) -> Tuple[bool, str]:
        """Undo the last operation.
        
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if not self._checkpoint_order:
                return False, "Nothing to undo"
            
            # Find last non-rolled-back checkpoint
            for i in range(len(self._checkpoint_order) - 1, -1, -1):
                checkpoint_id = self._checkpoint_order[i]
                checkpoint = self._checkpoints.get(checkpoint_id)
                if checkpoint and not checkpoint.rolled_back:
                    success, message = self.rollback(checkpoint_id)
                    if success:
                        self._redo_stack.append(checkpoint_id)
                    return success, message
            
            return False, "Nothing to undo"
    
    def redo(self) -> Tuple[bool, str]:
        """Redo the last undone operation.
        
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if not self._redo_stack:
                return False, "Nothing to redo"
            
            checkpoint_id = self._redo_stack.pop()
            checkpoint = self._checkpoints.get(checkpoint_id)
            
            if not checkpoint:
                return False, "Checkpoint not found for redo"
            
            # Re-apply state_after if available
            if checkpoint.state_after:
                for original_path in checkpoint.file_backups.keys():
                    # The state_after should contain the modified content
                    if original_path in checkpoint.state_after:
                        try:
                            with open(original_path, "w") as f:
                                f.write(checkpoint.state_after[original_path])
                        except Exception as e:
                            return False, f"Failed to redo: {e}"
            
            checkpoint.rolled_back = False
            return True, f"Redone: {checkpoint.operation}"
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to stay under limit."""
        while len(self._checkpoints) >= self.max_checkpoints:
            if self._checkpoint_order:
                oldest_id = self._checkpoint_order.pop(0)
                checkpoint = self._checkpoints.pop(oldest_id, None)
                if checkpoint:
                    # Clean up backup files
                    for backup_path in checkpoint.file_backups.values():
                        try:
                            if os.path.exists(backup_path):
                                os.remove(backup_path)
                        except Exception:
                            pass
            else:
                break
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[OperationCheckpoint]:
        """Get a checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self, limit: int = 20) -> List[OperationCheckpoint]:
        """List recent checkpoints.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of checkpoints (newest first)
        """
        checkpoints = []
        for checkpoint_id in reversed(self._checkpoint_order[-limit:]):
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint:
                checkpoints.append(checkpoint)
        return checkpoints
    
    def cleanup_old_backups(self) -> int:
        """Clean up backup files older than max_backup_age_hours.
        
        Returns:
            Number of files cleaned up
        """
        cutoff_time = time.time() - (self.max_backup_age_hours * 3600)
        cleaned = 0
        
        try:
            for backup_file in self.backup_dir.glob("*.bak"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    cleaned += 1
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
        
        return cleaned


class SafetyManager:
    """Manages safety and consent for agent operations.
    
    Provides:
    - Consent request/response handling
    - Operation boundaries enforcement
    - Audit logging
    - Dangerous operation prevention
    
    Example:
        >>> safety = SafetyManager()
        >>> # Request consent for an operation
        >>> request = safety.request_consent(
        ...     ConsentType.COMMAND_EXECUTION,
        ...     "pip install qiskit",
        ...     "Install Qiskit quantum computing library"
        ... )
        >>> # In TUI, display request and get user response
        >>> if await safety.wait_for_consent(request.id):
        ...     # Proceed with operation
        ...     pass
    """
    
    # Operations that are always allowed without consent
    SAFE_OPERATIONS: Set[str] = {
        "read_file",
        "list_directory",
        "get_system_info",
        "git_status",
        "navigate_to",
        "find_files",
    }
    
    # Operations that always require consent
    DANGEROUS_OPERATIONS: Set[str] = {
        "request_admin",
        "git_push",
        "modify_backend_code",
        "write_file",
        "execute_command",
    }
    
    # Blocked commands (never allowed)
    BLOCKED_PATTERNS: List[str] = [
        "rm -rf /",
        "del /f /s /q C:\\",
        "format",
        ":(){:|:&};:",  # Fork bomb
        "shutdown",
        "reboot",
    ]
    
    def __init__(
        self,
        rollback_manager: Optional[RollbackManager] = None,
        consent_callback: Optional[Callable[[ConsentRequest], ConsentResponse]] = None,
        auto_approve_safe: bool = True,
        audit_log_path: Optional[str] = None,
    ):
        """Initialize safety manager.
        
        Args:
            rollback_manager: Rollback manager for checkpoints
            consent_callback: Callback for getting user consent
            auto_approve_safe: Auto-approve safe operations
            audit_log_path: Path for audit log file
        """
        self.rollback_manager = rollback_manager or RollbackManager()
        self.consent_callback = consent_callback
        self.auto_approve_safe = auto_approve_safe
        self.audit_log_path = audit_log_path
        
        self._pending_consents: Dict[str, ConsentRequest] = {}
        self._consent_events: Dict[str, threading.Event] = {}
        self._consent_responses: Dict[str, ConsentResponse] = {}
        self._session_approvals: Set[str] = set()  # tool_name for session-wide approval
        self._always_approvals: Set[str] = set()  # tool_name for permanent approval
        self._lock = threading.Lock()
        
        logger.info("SafetyManager initialized")
    
    def _generate_consent_id(self) -> str:
        """Generate unique consent request ID."""
        timestamp = int(time.time() * 1000)
        random_part = os.urandom(4).hex()
        return f"consent_{timestamp}_{random_part}"
    
    def _log_audit(
        self,
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        """Log an audit event.
        
        Args:
            event_type: Type of event
            details: Event details
        """
        if not self.audit_log_path:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
        }
        
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def is_blocked(self, operation: str) -> bool:
        """Check if an operation is blocked.
        
        Args:
            operation: Operation string (command, etc.)
            
        Returns:
            True if blocked
        """
        op_lower = operation.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in op_lower:
                logger.warning(f"Blocked dangerous operation: {operation}")
                self._log_audit("blocked_operation", {"operation": operation})
                return True
        return False
    
    def is_safe_operation(self, tool_name: str) -> bool:
        """Check if a tool operation is considered safe.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if safe
        """
        return tool_name in self.SAFE_OPERATIONS
    
    def requires_consent(self, tool_name: str) -> bool:
        """Check if a tool requires consent.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if consent required
        """
        if self.auto_approve_safe and self.is_safe_operation(tool_name):
            return False
        if tool_name in self._always_approvals:
            return False
        if tool_name in self._session_approvals:
            return False
        return True
    
    def request_consent(
        self,
        consent_type: ConsentType,
        operation: str,
        description: str,
        tool_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "medium",
        reversible: bool = True,
    ) -> ConsentRequest:
        """Create a consent request.
        
        Args:
            consent_type: Type of consent needed
            operation: Operation name/description
            description: Detailed description
            tool_name: Name of the tool
            details: Additional details
            risk_level: Risk level (low, medium, high, critical)
            reversible: Whether operation can be undone
            
        Returns:
            Consent request object
        """
        with self._lock:
            request_id = self._generate_consent_id()
            request = ConsentRequest(
                id=request_id,
                consent_type=consent_type,
                operation=operation,
                description=description,
                tool_name=tool_name,
                details=details or {},
                risk_level=risk_level,
                reversible=reversible,
            )
            
            self._pending_consents[request_id] = request
            self._consent_events[request_id] = threading.Event()
            
            self._log_audit("consent_requested", request.to_dict())
            logger.info(f"Consent requested: {request_id} for {operation}")
            
            return request
    
    def respond_to_consent(
        self,
        request_id: str,
        decision: ConsentDecision,
        user_message: Optional[str] = None,
    ) -> bool:
        """Respond to a consent request.
        
        Args:
            request_id: ID of the consent request
            decision: User's decision
            user_message: Optional user message
            
        Returns:
            True if response recorded
        """
        with self._lock:
            request = self._pending_consents.get(request_id)
            if not request:
                return False
            
            response = ConsentResponse(
                request_id=request_id,
                decision=decision,
                user_message=user_message,
            )
            
            self._consent_responses[request_id] = response
            
            # Handle approval levels
            if response.approved and request.tool_name:
                if decision == ConsentDecision.APPROVED_SESSION:
                    self._session_approvals.add(request.tool_name)
                elif decision == ConsentDecision.APPROVED_ALWAYS:
                    self._always_approvals.add(request.tool_name)
            
            # Signal waiting threads
            if request_id in self._consent_events:
                self._consent_events[request_id].set()
            
            self._log_audit("consent_responded", {
                "request_id": request_id,
                "decision": decision.name,
                "approved": response.approved,
            })
            
            logger.info(f"Consent {request_id}: {decision.name}")
            return True
    
    def wait_for_consent(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[ConsentResponse]:
        """Wait for a consent response.
        
        Args:
            request_id: ID of the consent request
            timeout: Maximum time to wait (None = forever)
            
        Returns:
            Consent response, or None if timeout
        """
        event = self._consent_events.get(request_id)
        if not event:
            return None
        
        # Use callback if available
        if self.consent_callback:
            request = self._pending_consents.get(request_id)
            if request:
                try:
                    response = self.consent_callback(request)
                    self.respond_to_consent(
                        request_id,
                        response.decision,
                        response.user_message,
                    )
                except Exception as e:
                    logger.error(f"Consent callback error: {e}")
        
        # Wait for response
        if event.wait(timeout=timeout):
            return self._consent_responses.get(request_id)
        return None
    
    async def wait_for_consent_async(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[ConsentResponse]:
        """Async version of wait_for_consent.
        
        Args:
            request_id: ID of the consent request
            timeout: Maximum time to wait
            
        Returns:
            Consent response, or None if timeout
        """
        import asyncio
        
        event = self._consent_events.get(request_id)
        if not event:
            return None
        
        start_time = time.time()
        while not event.is_set():
            await asyncio.sleep(0.1)
            if timeout and (time.time() - start_time) > timeout:
                return None
        
        return self._consent_responses.get(request_id)
    
    def get_pending_consents(self) -> List[ConsentRequest]:
        """Get all pending consent requests.
        
        Returns:
            List of pending requests
        """
        with self._lock:
            # Filter to only those without responses
            pending = []
            for request_id, request in self._pending_consents.items():
                if request_id not in self._consent_responses:
                    pending.append(request)
            return pending
    
    def cancel_consent(self, request_id: str) -> bool:
        """Cancel a pending consent request.
        
        Args:
            request_id: ID of request to cancel
            
        Returns:
            True if cancelled
        """
        with self._lock:
            if request_id in self._pending_consents:
                del self._pending_consents[request_id]
                # Respond with DENIED to unblock any waiters
                self.respond_to_consent(request_id, ConsentDecision.DENIED, "Cancelled")
                return True
            return False
    
    def clear_session_approvals(self) -> None:
        """Clear all session-level approvals."""
        with self._lock:
            self._session_approvals.clear()
            logger.info("Cleared session approvals")
    
    def clear_always_approvals(self) -> None:
        """Clear all permanent approvals."""
        with self._lock:
            self._always_approvals.clear()
            logger.info("Cleared permanent approvals")
    
    def create_checkpoint(
        self,
        operation: str,
        files: Optional[List[str]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> OperationCheckpoint:
        """Create a checkpoint before a risky operation.
        
        Args:
            operation: Operation description
            files: Files to backup
            state: Additional state to preserve
            
        Returns:
            Checkpoint object
        """
        return self.rollback_manager.create_checkpoint(
            operation=operation,
            files=files,
            state=state,
        )
    
    def rollback(self, checkpoint_id: str) -> Tuple[bool, str]:
        """Rollback to a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Tuple of (success, message)
        """
        return self.rollback_manager.rollback(checkpoint_id)
    
    def undo(self) -> Tuple[bool, str]:
        """Undo the last operation."""
        return self.rollback_manager.undo()
    
    def redo(self) -> Tuple[bool, str]:
        """Redo the last undone operation."""
        return self.rollback_manager.redo()
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of audit entries
        """
        if not self.audit_log_path or not os.path.exists(self.audit_log_path):
            return []
        
        entries = []
        try:
            with open(self.audit_log_path, "r") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
        
        return entries
