"""Approval Workflow Manager.

Manages the approval workflow for AI-generated changes.
Provides structured approval process with validation and audit trail.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

if TYPE_CHECKING:
    from .change_tracker import ChangeTracker, FileChange

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"
    AUTO_APPROVED = "auto_approved"


class ApprovalCategory(Enum):
    """Category of changes for approval."""
    CODE_GENERATION = "code_generation"
    FILE_CREATION = "file_creation"
    FILE_MODIFICATION = "file_modification"
    FILE_DELETION = "file_deletion"
    REGISTRY_UPDATE = "registry_update"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"


@dataclass
class ApprovalRequest:
    """Represents a request for approval."""
    
    id: str
    category: ApprovalCategory
    title: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewer_notes: str = ""
    auto_approve_eligible: bool = False
    risk_level: str = "low"  # low, medium, high
    dependencies: List[str] = field(default_factory=list)
    
    def approve(self, notes: str = "") -> None:
        """Approve this request."""
        self.status = ApprovalStatus.APPROVED
        self.reviewed_at = datetime.now()
        self.reviewer_notes = notes
        logger.info(f"Approved: {self.id} - {self.title}")
    
    def reject(self, notes: str = "") -> None:
        """Reject this request."""
        self.status = ApprovalStatus.REJECTED
        self.reviewed_at = datetime.now()
        self.reviewer_notes = notes
        logger.info(f"Rejected: {self.id} - {self.title}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer_notes": self.reviewer_notes,
            "risk_level": self.risk_level,
        }


@dataclass
class ApprovalPolicy:
    """Policy for automatic approvals."""
    
    name: str
    enabled: bool = True
    auto_approve_low_risk: bool = True
    auto_approve_documentation: bool = True
    auto_approve_tests: bool = False
    require_review_for_deletions: bool = True
    require_review_for_registry: bool = True
    max_auto_approve_lines: int = 100
    
    def can_auto_approve(self, request: ApprovalRequest) -> bool:
        """Check if a request can be auto-approved.
        
        Args:
            request: The approval request
            
        Returns:
            True if can be auto-approved
        """
        if not self.enabled:
            return False
        
        # Check category-specific rules
        if request.category == ApprovalCategory.FILE_DELETION:
            if self.require_review_for_deletions:
                return False
        
        if request.category == ApprovalCategory.REGISTRY_UPDATE:
            if self.require_review_for_registry:
                return False
        
        if request.category == ApprovalCategory.DOCUMENTATION:
            if self.auto_approve_documentation:
                return True
        
        if request.category == ApprovalCategory.TEST_GENERATION:
            if self.auto_approve_tests:
                return True
        
        # Check risk level
        if request.risk_level == "low" and self.auto_approve_low_risk:
            # Check line count
            lines = request.details.get("lines_added", 0)
            if lines <= self.max_auto_approve_lines:
                return True
        
        return False


class ApprovalWorkflowManager:
    """Manages the approval workflow for AI-generated changes.
    
    Provides:
    - Structured approval requests
    - Policy-based auto-approval
    - Audit trail
    - Dependency tracking
    - Batch operations
    """
    
    def __init__(self, policy: Optional[ApprovalPolicy] = None):
        """Initialize approval workflow manager.
        
        Args:
            policy: Approval policy (uses default if None)
        """
        self.policy = policy or ApprovalPolicy(name="default")
        self._requests: Dict[str, ApprovalRequest] = {}
        self._approval_order: List[str] = []
        self._callbacks: Dict[str, List[Callable]] = {
            "on_approved": [],
            "on_rejected": [],
            "on_status_changed": [],
        }
        self._next_id = 1
    
    def create_request(
        self,
        category: ApprovalCategory,
        title: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "low",
        dependencies: Optional[List[str]] = None,
        auto_check: bool = True
    ) -> ApprovalRequest:
        """Create a new approval request.
        
        Args:
            category: Category of the change
            title: Short title for the request
            description: Detailed description
            details: Additional details (lines added, etc.)
            risk_level: Risk level (low, medium, high)
            dependencies: IDs of dependent requests
            auto_check: Whether to check for auto-approval
            
        Returns:
            The created ApprovalRequest
        """
        request_id = f"REQ-{self._next_id:04d}"
        self._next_id += 1
        
        request = ApprovalRequest(
            id=request_id,
            category=category,
            title=title,
            description=description,
            details=details or {},
            risk_level=risk_level,
            dependencies=dependencies or [],
        )
        
        # Check for auto-approval
        if auto_check:
            request.auto_approve_eligible = self.policy.can_auto_approve(request)
            if request.auto_approve_eligible:
                request.status = ApprovalStatus.AUTO_APPROVED
                request.reviewed_at = datetime.now()
                request.reviewer_notes = "Auto-approved by policy"
        
        self._requests[request_id] = request
        self._approval_order.append(request_id)
        
        logger.debug(f"Created approval request: {request_id} - {title}")
        
        return request
    
    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a request by ID."""
        return self._requests.get(request_id)
    
    def get_all_requests(self) -> List[ApprovalRequest]:
        """Get all requests in order."""
        return [self._requests[rid] for rid in self._approval_order if rid in self._requests]
    
    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending requests."""
        return [r for r in self.get_all_requests() if r.status == ApprovalStatus.PENDING]
    
    def get_approved_requests(self) -> List[ApprovalRequest]:
        """Get all approved requests."""
        return [
            r for r in self.get_all_requests()
            if r.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
        ]
    
    def get_rejected_requests(self) -> List[ApprovalRequest]:
        """Get all rejected requests."""
        return [r for r in self.get_all_requests() if r.status == ApprovalStatus.REJECTED]
    
    def approve(self, request_id: str, notes: str = "") -> bool:
        """Approve a request.
        
        Args:
            request_id: ID of the request
            notes: Reviewer notes
            
        Returns:
            True if approved successfully
        """
        request = self._requests.get(request_id)
        if not request:
            return False
        
        # Check dependencies
        if not self._check_dependencies(request):
            logger.warning(f"Cannot approve {request_id}: dependencies not met")
            return False
        
        request.approve(notes)
        self._trigger_callbacks("on_approved", request)
        self._trigger_callbacks("on_status_changed", request)
        
        return True
    
    def reject(self, request_id: str, notes: str = "") -> bool:
        """Reject a request.
        
        Args:
            request_id: ID of the request
            notes: Reviewer notes
            
        Returns:
            True if rejected successfully
        """
        request = self._requests.get(request_id)
        if not request:
            return False
        
        request.reject(notes)
        
        # Reject dependent requests
        self._reject_dependents(request_id)
        
        self._trigger_callbacks("on_rejected", request)
        self._trigger_callbacks("on_status_changed", request)
        
        return True
    
    def approve_all_pending(self, notes: str = "") -> int:
        """Approve all pending requests.
        
        Args:
            notes: Reviewer notes
            
        Returns:
            Number of requests approved
        """
        count = 0
        for request in self.get_pending_requests():
            if self.approve(request.id, notes):
                count += 1
        return count
    
    def reject_all_pending(self, notes: str = "") -> int:
        """Reject all pending requests.
        
        Args:
            notes: Reviewer notes
            
        Returns:
            Number of requests rejected
        """
        count = 0
        for request in self.get_pending_requests():
            if self.reject(request.id, notes):
                count += 1
        return count
    
    def _check_dependencies(self, request: ApprovalRequest) -> bool:
        """Check if all dependencies are approved."""
        for dep_id in request.dependencies:
            dep = self._requests.get(dep_id)
            if not dep:
                continue
            if dep.status not in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED):
                return False
        return True
    
    def _reject_dependents(self, request_id: str) -> None:
        """Reject all requests that depend on this one."""
        for request in self.get_all_requests():
            if request_id in request.dependencies:
                if request.status == ApprovalStatus.PENDING:
                    request.reject("Dependency rejected")
                    self._reject_dependents(request.id)
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event.
        
        Args:
            event: Event name (on_approved, on_rejected, on_status_changed)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, request: ApprovalRequest) -> None:
        """Trigger callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get approval statistics."""
        total = len(self._requests)
        pending = len(self.get_pending_requests())
        approved = len([
            r for r in self.get_all_requests()
            if r.status == ApprovalStatus.APPROVED
        ])
        auto_approved = len([
            r for r in self.get_all_requests()
            if r.status == ApprovalStatus.AUTO_APPROVED
        ])
        rejected = len(self.get_rejected_requests())
        
        return {
            "total": total,
            "pending": pending,
            "approved": approved,
            "auto_approved": auto_approved,
            "rejected": rejected,
            "approval_rate": (approved + auto_approved) / total if total > 0 else 0,
        }
    
    def can_proceed(self) -> bool:
        """Check if all required approvals are complete.
        
        Returns:
            True if there are no pending required requests
        """
        return len(self.get_pending_requests()) == 0
    
    def export_audit_log(self, format: str = "json") -> str:
        """Export audit log of all approvals.
        
        Args:
            format: Export format (json)
            
        Returns:
            Audit log as string
        """
        log_entries = []
        
        for request in self.get_all_requests():
            entry = request.to_dict()
            log_entries.append(entry)
        
        if format == "json":
            return json.dumps({
                "audit_log": log_entries,
                "stats": self.get_stats(),
                "exported_at": datetime.now().isoformat(),
            }, indent=2)
        
        raise ValueError(f"Unsupported format: {format}")
    
    def clear(self) -> None:
        """Clear all requests."""
        self._requests.clear()
        self._approval_order.clear()
        self._next_id = 1


class ChangeApprovalIntegrator:
    """Integrates ChangeTracker with ApprovalWorkflowManager.
    
    Creates approval requests from file changes and
    applies approved changes to the tracker.
    """
    
    def __init__(
        self,
        workflow: ApprovalWorkflowManager,
        tracker: Optional['ChangeTracker'] = None
    ):
        """Initialize integrator.
        
        Args:
            workflow: Approval workflow manager
            tracker: Change tracker instance
        """
        self.workflow = workflow
        self.tracker = tracker
        self._change_to_request: Dict[str, str] = {}  # change hash -> request id
    
    def create_requests_from_changes(self) -> List[ApprovalRequest]:
        """Create approval requests from all tracked changes.
        
        Returns:
            List of created approval requests
        """
        if not self.tracker:
            return []
        
        requests = []
        
        for change in self.tracker.changes:
            # Determine category
            if change.change_type.value == "create":
                if "test" in change.file_path.lower():
                    category = ApprovalCategory.TEST_GENERATION
                elif change.file_path.endswith(".md"):
                    category = ApprovalCategory.DOCUMENTATION
                else:
                    category = ApprovalCategory.FILE_CREATION
            elif change.change_type.value == "modify":
                category = ApprovalCategory.FILE_MODIFICATION
            else:
                category = ApprovalCategory.FILE_DELETION
            
            # Determine risk level
            risk_level = "low"
            if change.change_type.value == "delete":
                risk_level = "high"
            elif change.lines_added > 200:
                risk_level = "medium"
            
            # Create request
            request = self.workflow.create_request(
                category=category,
                title=f"{change.change_type.value.title()}: {change.file_path.split('/')[-1]}",
                description=change.description or f"AI-generated {change.change_type.value} operation",
                details={
                    "file_path": change.file_path,
                    "lines_added": change.lines_added,
                    "lines_removed": change.lines_removed,
                },
                risk_level=risk_level,
            )
            
            # Link change to request
            change_hash = f"{change.file_path}:{change.timestamp.isoformat()}"
            self._change_to_request[change_hash] = request.id
            
            requests.append(request)
        
        return requests
    
    def sync_approvals_to_changes(self) -> None:
        """Sync approval status back to changes."""
        if not self.tracker:
            return
        
        for change in self.tracker.changes:
            change_hash = f"{change.file_path}:{change.timestamp.isoformat()}"
            request_id = self._change_to_request.get(change_hash)
            
            if request_id:
                request = self.workflow.get_request(request_id)
                if request:
                    change.approved = request.status in (
                        ApprovalStatus.APPROVED,
                        ApprovalStatus.AUTO_APPROVED
                    )
    
    def get_approved_changes(self) -> List['FileChange']:
        """Get all approved changes.
        
        Returns:
            List of approved file changes
        """
        self.sync_approvals_to_changes()
        
        if not self.tracker:
            return []
        
        return [c for c in self.tracker.changes if c.approved]
