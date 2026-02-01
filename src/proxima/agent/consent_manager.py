"""Consent Manager for Sensitive Operations.

Phase 8: Backend Code Modification with Safety

Provides comprehensive consent management including:
- Consent request handling
- Multiple consent scopes (once, session, always)
- Consent rule storage and revocation
- Audit trail of consent decisions
- Timeout-based auto-deny
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from proxima.utils.logging import get_logger

logger = get_logger("agent.consent_manager")


class ConsentScope(Enum):
    """Scope of consent approval."""
    ONCE = "once"           # This operation only
    SESSION = "session"     # All similar operations in session
    ALWAYS = "always"       # Permanently approved
    DENY = "deny"           # Rejected


class RiskLevel(Enum):
    """Risk level of an operation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    @property
    def color(self) -> str:
        """Get color for display."""
        colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "orange",
            RiskLevel.CRITICAL: "red",
        }
        return colors.get(self, "white")
    
    @property
    def emoji(self) -> str:
        """Get emoji for display."""
        emojis = {
            RiskLevel.LOW: "ℹ️",
            RiskLevel.MEDIUM: "⚠️",
            RiskLevel.HIGH: "🔶",
            RiskLevel.CRITICAL: "🚨",
        }
        return emojis.get(self, "❓")


@dataclass
class ConsentRequest:
    """A request for user consent."""
    request_id: str
    operation: str
    description: str
    risk_level: RiskLevel
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_name: Optional[str] = None
    reversible: bool = True
    timeout_seconds: int = 300  # 5 minutes default
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "details": self.details,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "reversible": self.reversible,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsentRequest":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            operation=data["operation"],
            description=data["description"],
            risk_level=RiskLevel(data.get("risk_level", "medium")),
            details=data.get("details", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            tool_name=data.get("tool_name"),
            reversible=data.get("reversible", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            metadata=data.get("metadata", {}),
        )
    
    def get_display_message(self) -> str:
        """Get formatted message for display."""
        emoji = self.risk_level.emoji
        lines = [
            f"{emoji} **{self.operation}**",
            "",
            self.description,
        ]
        
        if self.details:
            lines.append("")
            lines.append("Details:")
            for key, value in self.details.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                lines.append(f"  • {key}: {value}")
        
        if not self.reversible:
            lines.append("")
            lines.append("⚠️ **This operation cannot be undone.**")
        
        if self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            lines.append("")
            lines.append(f"🔒 Risk Level: {self.risk_level.value.upper()}")
        
        return "\n".join(lines)


@dataclass
class ConsentResponse:
    """Response to a consent request."""
    request_id: str
    scope: ConsentScope
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_message: Optional[str] = None
    
    @property
    def approved(self) -> bool:
        """Check if consent was granted."""
        return self.scope != ConsentScope.DENY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "scope": self.scope.value,
            "timestamp": self.timestamp,
            "user_message": self.user_message,
            "approved": self.approved,
        }


@dataclass
class ConsentRule:
    """A stored consent rule for automatic approval."""
    rule_id: str
    operation_pattern: str  # Pattern to match operations
    tool_name: Optional[str] = None
    granted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    granted_by: str = "user"
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, operation: str, tool_name: Optional[str] = None) -> bool:
        """Check if rule matches an operation.
        
        Args:
            operation: Operation string
            tool_name: Tool name
            
        Returns:
            True if matches
        """
        # Exact match
        if self.operation_pattern == operation:
            return True
        
        # Wildcard match
        if self.operation_pattern.endswith("*"):
            prefix = self.operation_pattern[:-1]
            if operation.startswith(prefix):
                return True
        
        # Tool name match
        if self.tool_name and tool_name:
            if self.tool_name == tool_name:
                return True
        
        return False
    
    def is_expired(self) -> bool:
        """Check if rule has expired."""
        if not self.expires_at:
            return False
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expires
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "operation_pattern": self.operation_pattern,
            "tool_name": self.tool_name,
            "granted_at": self.granted_at,
            "granted_by": self.granted_by,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsentRule":
        """Create from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            operation_pattern=data["operation_pattern"],
            tool_name=data.get("tool_name"),
            granted_at=data.get("granted_at", datetime.now().isoformat()),
            granted_by=data.get("granted_by", "user"),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConsentAuditEntry:
    """Audit log entry for consent decisions."""
    timestamp: str
    request: ConsentRequest
    response: ConsentResponse
    auto_approved: bool = False
    rule_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "request": self.request.to_dict(),
            "response": self.response.to_dict(),
            "auto_approved": self.auto_approved,
            "rule_id": self.rule_id,
        }


class ConsentManager:
    """Manages user consent for sensitive operations.
    
    Features:
    - Interactive consent requests with timeout
    - Session-based consent caching
    - Persistent "always" rules
    - Audit logging
    - Rule revocation
    
    Example:
        >>> manager = ConsentManager()
        >>> 
        >>> # Request consent
        >>> request = manager.create_request(
        ...     operation="modify_backend_code",
        ...     description="Modify simulator.py",
        ...     risk_level=RiskLevel.MEDIUM,
        ... )
        >>> 
        >>> # Wait for response (with UI callback)
        >>> response = await manager.wait_for_consent(request)
        >>> 
        >>> if response.approved:
        ...     # Proceed with operation
        ...     pass
    """
    
    def __init__(
        self,
        rules_path: Optional[str] = None,
        audit_path: Optional[str] = None,
        default_timeout: int = 300,
    ):
        """Initialize consent manager.
        
        Args:
            rules_path: Path to consent rules file
            audit_path: Path to audit log file
            default_timeout: Default timeout for consent requests
        """
        self.rules_path = Path(rules_path) if rules_path else Path.home() / ".proxima" / "consent_rules.json"
        self.audit_path = Path(audit_path) if audit_path else Path.home() / ".proxima" / "consent_audit.jsonl"
        self.default_timeout = default_timeout
        
        self._pending_requests: Dict[str, ConsentRequest] = {}
        self._request_events: Dict[str, asyncio.Event] = {}
        self._responses: Dict[str, ConsentResponse] = {}
        self._session_approvals: Set[str] = set()  # operation patterns approved for session
        self._rules: Dict[str, ConsentRule] = {}
        self._audit_entries: List[ConsentAuditEntry] = []
        self._lock = threading.Lock()
        self._request_counter = 0
        
        # Consent callback for UI
        self._consent_callback: Optional[Callable[[ConsentRequest], None]] = None
        
        # Load persisted rules
        self._load_rules()
        
        # Ensure directories exist
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("ConsentManager initialized")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        timestamp = int(time.time() * 1000)
        return f"consent_{timestamp}_{self._request_counter}"
    
    def _generate_rule_id(self) -> str:
        """Generate unique rule ID."""
        timestamp = int(time.time() * 1000)
        random_part = os.urandom(4).hex()
        return f"rule_{timestamp}_{random_part}"
    
    def set_consent_callback(
        self,
        callback: Callable[[ConsentRequest], None],
    ) -> None:
        """Set callback for consent requests.
        
        Args:
            callback: Function to call when consent is needed
        """
        self._consent_callback = callback
    
    def create_request(
        self,
        operation: str,
        description: str,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        reversible: bool = True,
        timeout: Optional[int] = None,
    ) -> ConsentRequest:
        """Create a consent request.
        
        Args:
            operation: Operation identifier
            description: Human-readable description
            risk_level: Risk level of operation
            details: Additional details
            tool_name: Name of tool requesting consent
            reversible: Whether operation can be undone
            timeout: Timeout in seconds
            
        Returns:
            ConsentRequest
        """
        request = ConsentRequest(
            request_id=self._generate_request_id(),
            operation=operation,
            description=description,
            risk_level=risk_level,
            details=details or {},
            tool_name=tool_name,
            reversible=reversible,
            timeout_seconds=timeout or self.default_timeout,
        )
        
        with self._lock:
            self._pending_requests[request.request_id] = request
            self._request_events[request.request_id] = asyncio.Event()
        
        return request
    
    def check_auto_approval(
        self,
        operation: str,
        tool_name: Optional[str] = None,
    ) -> Optional[ConsentResponse]:
        """Check if operation is auto-approved.
        
        Args:
            operation: Operation identifier
            tool_name: Tool name
            
        Returns:
            ConsentResponse if auto-approved, None otherwise
        """
        # Check session approvals
        if operation in self._session_approvals:
            return ConsentResponse(
                request_id="auto",
                scope=ConsentScope.SESSION,
            )
        
        # Check persistent rules
        for rule in self._rules.values():
            if rule.is_expired():
                continue
            if rule.matches(operation, tool_name):
                return ConsentResponse(
                    request_id="auto",
                    scope=ConsentScope.ALWAYS,
                )
        
        return None
    
    async def wait_for_consent(
        self,
        request: ConsentRequest,
        check_auto: bool = True,
    ) -> ConsentResponse:
        """Wait for user consent.
        
        Args:
            request: Consent request
            check_auto: Check for auto-approval first
            
        Returns:
            ConsentResponse
        """
        # Check auto-approval
        if check_auto:
            auto_response = self.check_auto_approval(
                request.operation,
                request.tool_name,
            )
            if auto_response:
                auto_response.request_id = request.request_id
                self._log_audit(request, auto_response, auto_approved=True)
                return auto_response
        
        # Trigger consent callback
        if self._consent_callback:
            self._consent_callback(request)
        
        # Wait for response with timeout
        event = self._request_events.get(request.request_id)
        if not event:
            return ConsentResponse(
                request_id=request.request_id,
                scope=ConsentScope.DENY,
                user_message="Request not found",
            )
        
        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=request.timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Auto-deny on timeout
            response = ConsentResponse(
                request_id=request.request_id,
                scope=ConsentScope.DENY,
                user_message="Request timed out",
            )
            self._cleanup_request(request.request_id)
            self._log_audit(request, response)
            return response
        
        # Get response
        response = self._responses.get(request.request_id)
        if not response:
            response = ConsentResponse(
                request_id=request.request_id,
                scope=ConsentScope.DENY,
                user_message="No response received",
            )
        
        self._log_audit(request, response)
        self._cleanup_request(request.request_id)
        
        return response
    
    def respond_to_request(
        self,
        request_id: str,
        scope: ConsentScope,
        user_message: Optional[str] = None,
    ) -> bool:
        """Respond to a consent request.
        
        Args:
            request_id: Request ID
            scope: Consent scope
            user_message: Optional message from user
            
        Returns:
            True if response was recorded
        """
        with self._lock:
            request = self._pending_requests.get(request_id)
            if not request:
                logger.warning(f"Request not found: {request_id}")
                return False
            
            response = ConsentResponse(
                request_id=request_id,
                scope=scope,
                user_message=user_message,
            )
            
            self._responses[request_id] = response
            
            # Handle scope
            if scope == ConsentScope.SESSION:
                self._session_approvals.add(request.operation)
            elif scope == ConsentScope.ALWAYS:
                self._add_rule(request.operation, request.tool_name)
            
            # Signal waiting task
            event = self._request_events.get(request_id)
            if event:
                event.set()
            
            return True
    
    def _cleanup_request(self, request_id: str) -> None:
        """Clean up a completed request."""
        with self._lock:
            self._pending_requests.pop(request_id, None)
            self._request_events.pop(request_id, None)
            # Keep response for reference
    
    def _add_rule(
        self,
        operation_pattern: str,
        tool_name: Optional[str] = None,
    ) -> ConsentRule:
        """Add a persistent consent rule."""
        rule = ConsentRule(
            rule_id=self._generate_rule_id(),
            operation_pattern=operation_pattern,
            tool_name=tool_name,
        )
        
        self._rules[rule.rule_id] = rule
        self._save_rules()
        
        logger.info(f"Added consent rule: {operation_pattern}")
        return rule
    
    def revoke_rule(self, rule_id: str) -> bool:
        """Revoke a consent rule.
        
        Args:
            rule_id: Rule ID to revoke
            
        Returns:
            True if revoked
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            self._save_rules()
            logger.info(f"Revoked consent rule: {rule_id}")
            return True
        return False
    
    def clear_session_approvals(self) -> None:
        """Clear all session approvals."""
        self._session_approvals.clear()
        logger.info("Cleared session approvals")
    
    def list_rules(self) -> List[ConsentRule]:
        """List all consent rules."""
        return list(self._rules.values())
    
    def list_session_approvals(self) -> List[str]:
        """List session-approved operations."""
        return list(self._session_approvals)
    
    def get_pending_requests(self) -> List[ConsentRequest]:
        """Get all pending consent requests."""
        return list(self._pending_requests.values())
    
    def _load_rules(self) -> None:
        """Load rules from disk."""
        if not self.rules_path.exists():
            return
        
        try:
            with open(self.rules_path, "r") as f:
                data = json.load(f)
            
            for rule_data in data.get("rules", []):
                rule = ConsentRule.from_dict(rule_data)
                if not rule.is_expired():
                    self._rules[rule.rule_id] = rule
            
            logger.info(f"Loaded {len(self._rules)} consent rules")
        except Exception as e:
            logger.error(f"Failed to load consent rules: {e}")
    
    def _save_rules(self) -> None:
        """Save rules to disk."""
        try:
            data = {
                "version": 1,
                "updated_at": datetime.now().isoformat(),
                "rules": [r.to_dict() for r in self._rules.values()],
            }
            
            with open(self.rules_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save consent rules: {e}")
    
    def _log_audit(
        self,
        request: ConsentRequest,
        response: ConsentResponse,
        auto_approved: bool = False,
        rule_id: Optional[str] = None,
    ) -> None:
        """Log a consent decision to audit trail."""
        entry = ConsentAuditEntry(
            timestamp=datetime.now().isoformat(),
            request=request,
            response=response,
            auto_approved=auto_approved,
            rule_id=rule_id,
        )
        
        self._audit_entries.append(entry)
        
        # Write to file
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_audit_log(self, limit: int = 100) -> List[ConsentAuditEntry]:
        """Get recent audit entries.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of audit entries
        """
        return self._audit_entries[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consent statistics."""
        total_requests = len(self._audit_entries)
        approved = len([e for e in self._audit_entries if e.response.approved])
        denied = total_requests - approved
        auto_approved = len([e for e in self._audit_entries if e.auto_approved])
        
        return {
            "total_requests": total_requests,
            "approved": approved,
            "denied": denied,
            "auto_approved": auto_approved,
            "approval_rate": round(approved / max(total_requests, 1) * 100, 1),
            "active_rules": len(self._rules),
            "session_approvals": len(self._session_approvals),
            "pending_requests": len(self._pending_requests),
        }


# Global instance
_consent_manager: Optional[ConsentManager] = None


def get_consent_manager() -> ConsentManager:
    """Get the global ConsentManager instance."""
    global _consent_manager
    if _consent_manager is None:
        _consent_manager = ConsentManager()
    return _consent_manager


async def request_consent(
    operation: str,
    description: str,
    risk_level: str = "medium",
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """Convenience function to request consent.
    
    Returns:
        True if approved
    """
    manager = get_consent_manager()
    request = manager.create_request(
        operation=operation,
        description=description,
        risk_level=RiskLevel(risk_level),
        details=details,
    )
    response = await manager.wait_for_consent(request)
    return response.approved
