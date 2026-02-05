"""
Phase 10: Deployment and Monitoring

This module implements production deployment, observability, and continuous improvement
capabilities for the dynamic AI assistant. It provides:

- Production Deployment (Phase 10.1): Staged rollouts, configuration management, dependencies
- Observability and Monitoring (Phase 10.2): Logging, metrics, tracing, health checks
- Continuous Improvement (Phase 10.3): Usage analytics, feedback loop, model monitoring

Architecture Principles:
- Stable infrastructure that supports dynamic model operation
- No hardcoding - all configurations and rules are dynamic
- Works with any integrated LLM (Ollama, Gemini, GPT, Claude, etc.)
- Self-describing components for LLM understanding
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import psutil
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Enums for Deployment and Monitoring
# ============================================================================

class DeploymentStrategy(Enum):
    """Deployment strategies for staged rollouts."""
    DIRECT = "direct"  # Direct deployment (all at once)
    CANARY = "canary"  # Canary deployment (small percentage first)
    BLUE_GREEN = "blue_green"  # Blue-green deployment
    ROLLING = "rolling"  # Rolling update
    FEATURE_FLAG = "feature_flag"  # Feature flag based


class DeploymentStatus(Enum):
    """Status of a deployment."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FeedbackType(Enum):
    """Types of user feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    SUGGESTION = "suggestion"


# ============================================================================
# Data Classes for Deployment and Monitoring
# ============================================================================

@dataclass
class DeploymentConfig:
    """Configuration for a deployment."""
    deployment_id: str
    version: str
    strategy: DeploymentStrategy
    
    # Target environment
    environment: str  # development, staging, production
    
    # Rollout configuration
    canary_percentage: float = 5.0
    rollout_steps: List[float] = field(default_factory=lambda: [5, 25, 50, 100])
    
    # Verification
    verification_timeout_seconds: int = 300
    health_check_interval_seconds: int = 10
    
    # Rollback
    auto_rollback_on_failure: bool = True
    rollback_threshold_error_rate: float = 0.05
    
    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "version": self.version,
            "strategy": self.strategy.value,
            "environment": self.environment,
            "canary_percentage": self.canary_percentage,
            "rollout_steps": self.rollout_steps,
            "auto_rollback_on_failure": self.auto_rollback_on_failure,
            "feature_flags": self.feature_flags,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    deployment_id: str
    status: DeploymentStatus
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Results
    current_step: int = 0
    total_steps: int = 1
    current_percentage: float = 0.0
    
    # Verification results
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    verification_errors: List[str] = field(default_factory=list)
    
    # Rollback info
    rolled_back: bool = False
    rollback_reason: Optional[str] = None
    previous_version: Optional[str] = None
    
    # Audit
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "current_percentage": self.current_percentage,
            "health_checks_passed": self.health_checks_passed,
            "health_checks_failed": self.health_checks_failed,
            "rolled_back": self.rolled_back,
        }


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    config_id: str
    environment: str
    
    # Configuration values
    values: Dict[str, Any] = field(default_factory=dict)
    
    # Secrets (encrypted references, not actual values)
    secret_refs: Dict[str, str] = field(default_factory=dict)
    
    # Versioning
    version: int = 1
    previous_version: Optional[int] = None
    
    # Validation
    schema: Optional[Dict[str, Any]] = None
    validated: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "environment": self.environment,
            "version": self.version,
            "validated": self.validated,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    version: str
    
    # Security
    has_vulnerability: bool = False
    vulnerability_severity: Optional[str] = None
    vulnerability_id: Optional[str] = None
    
    # License
    license: Optional[str] = None
    license_compliant: bool = True
    
    # Compatibility
    compatible: bool = True
    compatibility_issues: List[str] = field(default_factory=list)
    
    # Update info
    latest_version: Optional[str] = None
    update_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "has_vulnerability": self.has_vulnerability,
            "vulnerability_severity": self.vulnerability_severity,
            "license": self.license,
            "license_compliant": self.license_compliant,
            "compatible": self.compatible,
            "update_available": self.update_available,
        }


@dataclass
class LogEntry:
    """Structured log entry."""
    log_id: str
    timestamp: datetime
    level: LogLevel
    message: str
    
    # Context
    logger_name: str = ""
    module: str = ""
    function: str = ""
    line_number: int = 0
    
    # Structured data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Exception info
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Redaction
    redacted_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger_name": self.logger_name,
            "module": self.module,
            "data": self.data,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }


@dataclass
class Metric:
    """A metric data point."""
    metric_id: str
    name: str
    metric_type: MetricType
    value: float
    
    # Labels/tags
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Aggregation
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
        }


@dataclass
class TraceSpan:
    """A span in a distributed trace."""
    trace_id: str
    span_id: str
    operation_name: str
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Hierarchy
    parent_span_id: Optional[str] = None
    
    # Context
    service_name: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: str = "ok"  # ok, error
    error_message: Optional[str] = None
    
    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "duration_ms": self.duration_ms,
            "parent_span_id": self.parent_span_id,
            "status": self.status,
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_id: str
    check_name: str
    status: HealthStatus
    
    # Details
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    
    # Dependencies
    dependency_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
        }


@dataclass
class Alert:
    """An alert notification."""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    
    # Context
    source: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    fired_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    # State
    is_active: bool = True
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    
    # Deduplication
    fingerprint: str = ""
    occurrence_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "is_active": self.is_active,
            "fired_at": self.fired_at.isoformat(),
            "occurrence_count": self.occurrence_count,
        }


@dataclass
class UsageEvent:
    """A usage analytics event."""
    event_id: str
    event_type: str
    
    # Context (anonymized)
    session_id: str = ""
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Feature tracking
    feature_name: Optional[str] = None
    operation_name: Optional[str] = None
    
    # Outcome
    success: bool = True
    error_type: Optional[str] = None
    
    # Performance
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "feature_name": self.feature_name,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }


@dataclass
class UserFeedback:
    """User feedback on AI responses."""
    feedback_id: str
    feedback_type: FeedbackType
    
    # Rating (1-5 stars or thumbs up/down)
    rating: Optional[int] = None
    
    # Content
    comment: str = ""
    
    # Context
    session_id: str = ""
    request_id: Optional[str] = None
    
    # Categorization
    category: str = ""
    tags: Set[str] = field(default_factory=set)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Status
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "rating": self.rating,
            "comment": self.comment,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModelPerformanceMetric:
    """Metrics for LLM model performance."""
    metric_id: str
    model_name: str
    
    # Quality metrics
    tool_selection_accuracy: float = 0.0
    parameter_extraction_accuracy: float = 0.0
    response_quality_score: float = 0.0
    
    # Hallucination detection
    hallucination_detected: bool = False
    hallucination_type: Optional[str] = None
    
    # Cost metrics
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    
    # Performance
    latency_ms: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "model_name": self.model_name,
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "parameter_extraction_accuracy": self.parameter_extraction_accuracy,
            "response_quality_score": self.response_quality_score,
            "hallucination_detected": self.hallucination_detected,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
        }


# ============================================================================
# Phase 10.1: Production Deployment
# ============================================================================

class DeploymentPipeline:
    """
    Manages staged rollout deployments with verification and rollback.
    
    Supports multiple deployment strategies:
    - Direct: Deploy to all instances at once
    - Canary: Deploy to a small percentage first
    - Blue-Green: Deploy to a parallel environment and switch
    - Rolling: Gradually replace instances
    - Feature Flag: Enable features gradually via flags
    """
    
    def __init__(self):
        self._deployments: Dict[str, DeploymentResult] = {}
        self._current_deployment: Optional[str] = None
        self._rollback_history: List[Dict[str, Any]] = []
        self._feature_flags: Dict[str, bool] = {}
        self._verification_checks: List[Callable[[], bool]] = []
        self._lock = threading.Lock()
    
    async def create_deployment(
        self,
        version: str,
        strategy: DeploymentStrategy,
        environment: str,
        config: Optional[Dict[str, Any]] = None
    ) -> DeploymentConfig:
        """Create a new deployment configuration."""
        deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"
        
        deployment_config = DeploymentConfig(
            deployment_id=deployment_id,
            version=version,
            strategy=strategy,
            environment=environment,
            canary_percentage=config.get("canary_percentage", 5.0) if config else 5.0,
            rollout_steps=config.get("rollout_steps", [5, 25, 50, 100]) if config else [5, 25, 50, 100],
            auto_rollback_on_failure=config.get("auto_rollback", True) if config else True,
            feature_flags=config.get("feature_flags", {}) if config else {},
            description=config.get("description", "") if config else "",
        )
        
        return deployment_config
    
    async def execute_deployment(
        self,
        config: DeploymentConfig,
        health_checker: Optional[Callable[[], HealthCheckResult]] = None
    ) -> DeploymentResult:
        """Execute a deployment with the configured strategy."""
        result = DeploymentResult(
            deployment_id=config.deployment_id,
            status=DeploymentStatus.IN_PROGRESS,
            start_time=datetime.now(),
            total_steps=len(config.rollout_steps),
        )
        
        with self._lock:
            self._deployments[config.deployment_id] = result
            self._current_deployment = config.deployment_id
        
        try:
            # Execute based on strategy
            if config.strategy == DeploymentStrategy.DIRECT:
                await self._execute_direct(config, result, health_checker)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary(config, result, health_checker)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green(config, result, health_checker)
            elif config.strategy == DeploymentStrategy.ROLLING:
                await self._execute_rolling(config, result, health_checker)
            elif config.strategy == DeploymentStrategy.FEATURE_FLAG:
                await self._execute_feature_flag(config, result, health_checker)
            
            # Verify deployment
            result.status = DeploymentStatus.VERIFYING
            await self._verify_deployment(config, result, health_checker)
            
            if result.health_checks_failed > 0 and config.auto_rollback_on_failure:
                await self._rollback(config, result)
            else:
                result.status = DeploymentStatus.COMPLETED
                result.current_percentage = 100.0
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.verification_errors.append(str(e))
            
            if config.auto_rollback_on_failure:
                await self._rollback(config, result)
        
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        # Log deployment
        result.audit_log.append({
            "event": "deployment_completed",
            "status": result.status.value,
            "timestamp": datetime.now().isoformat(),
        })
        
        return result
    
    async def _execute_direct(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        health_checker: Optional[Callable]
    ):
        """Execute direct deployment (all at once)."""
        result.audit_log.append({
            "event": "direct_deployment_started",
            "version": config.version,
            "timestamp": datetime.now().isoformat(),
        })
        result.current_step = 1
        result.current_percentage = 100.0
    
    async def _execute_canary(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        health_checker: Optional[Callable]
    ):
        """Execute canary deployment with gradual rollout."""
        for step, percentage in enumerate(config.rollout_steps):
            result.current_step = step + 1
            result.current_percentage = percentage
            
            result.audit_log.append({
                "event": "canary_step",
                "step": step + 1,
                "percentage": percentage,
                "timestamp": datetime.now().isoformat(),
            })
            
            # Verify at each step
            if health_checker:
                check_result = health_checker()
                if check_result.status == HealthStatus.HEALTHY:
                    result.health_checks_passed += 1
                else:
                    result.health_checks_failed += 1
                    if config.auto_rollback_on_failure:
                        break
            
            # Wait before next step
            await asyncio.sleep(config.health_check_interval_seconds)
    
    async def _execute_blue_green(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        health_checker: Optional[Callable]
    ):
        """Execute blue-green deployment."""
        # Deploy to green (standby) environment
        result.audit_log.append({
            "event": "blue_green_deploy_to_green",
            "timestamp": datetime.now().isoformat(),
        })
        result.current_step = 1
        result.current_percentage = 50.0
        
        # Verify green environment
        if health_checker:
            check_result = health_checker()
            if check_result.status != HealthStatus.HEALTHY:
                result.health_checks_failed += 1
                return
            result.health_checks_passed += 1
        
        # Switch traffic to green
        result.audit_log.append({
            "event": "blue_green_switch_traffic",
            "timestamp": datetime.now().isoformat(),
        })
        result.current_step = 2
        result.current_percentage = 100.0
    
    async def _execute_rolling(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        health_checker: Optional[Callable]
    ):
        """Execute rolling update deployment."""
        total_instances = 10  # Example
        batch_size = 2
        
        for batch in range(0, total_instances, batch_size):
            batch_num = batch // batch_size + 1
            result.current_step = batch_num
            result.current_percentage = min(100, (batch + batch_size) / total_instances * 100)
            
            result.audit_log.append({
                "event": "rolling_batch_update",
                "batch": batch_num,
                "instances": f"{batch}-{batch + batch_size}",
                "timestamp": datetime.now().isoformat(),
            })
            
            # Verify batch
            if health_checker:
                check_result = health_checker()
                if check_result.status == HealthStatus.HEALTHY:
                    result.health_checks_passed += 1
                else:
                    result.health_checks_failed += 1
                    break
    
    async def _execute_feature_flag(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        health_checker: Optional[Callable]
    ):
        """Execute feature flag based deployment."""
        # Enable feature flags gradually
        for flag_name, enabled in config.feature_flags.items():
            self._feature_flags[flag_name] = enabled
            result.audit_log.append({
                "event": "feature_flag_updated",
                "flag": flag_name,
                "enabled": enabled,
                "timestamp": datetime.now().isoformat(),
            })
        
        result.current_percentage = 100.0
    
    async def _verify_deployment(
        self,
        config: DeploymentConfig,
        result: DeploymentResult,
        health_checker: Optional[Callable]
    ):
        """Verify deployment with health checks."""
        if not health_checker:
            return
        
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < config.verification_timeout_seconds:
            check_result = health_checker()
            
            if check_result.status == HealthStatus.HEALTHY:
                result.health_checks_passed += 1
            else:
                result.health_checks_failed += 1
                result.verification_errors.append(check_result.message)
            
            await asyncio.sleep(config.health_check_interval_seconds)
            
            # Stop if too many failures
            if result.health_checks_failed > 3:
                break
    
    async def _rollback(
        self,
        config: DeploymentConfig,
        result: DeploymentResult
    ):
        """Rollback a deployment."""
        result.rolled_back = True
        result.rollback_reason = "Health check failures exceeded threshold"
        result.status = DeploymentStatus.ROLLED_BACK
        
        result.audit_log.append({
            "event": "rollback_initiated",
            "reason": result.rollback_reason,
            "timestamp": datetime.now().isoformat(),
        })
        
        self._rollback_history.append({
            "deployment_id": config.deployment_id,
            "version": config.version,
            "timestamp": datetime.now().isoformat(),
            "reason": result.rollback_reason,
        })
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled."""
        return self._feature_flags.get(feature_name, False)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get the status of a deployment."""
        return self._deployments.get(deployment_id)
    
    def get_rollback_history(self) -> List[Dict[str, Any]]:
        """Get rollback history."""
        return self._rollback_history.copy()


class ConfigurationManager:
    """
    Manages environment-specific configurations with versioning and validation.
    
    Features:
    - Environment-specific configs (dev, staging, production)
    - Secret management with encryption references
    - Configuration validation against schemas
    - Version history with rollback support
    - Audit trail for all changes
    """
    
    def __init__(self):
        self._configs: Dict[str, EnvironmentConfig] = {}
        self._version_history: Dict[str, List[EnvironmentConfig]] = defaultdict(list)
        self._validators: Dict[str, Callable[[Dict], bool]] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._secret_store: Dict[str, str] = {}  # In production, use proper secret management
    
    def set_config(
        self,
        environment: str,
        values: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None
    ) -> EnvironmentConfig:
        """Set configuration for an environment."""
        config_id = f"config_{environment}_{uuid.uuid4().hex[:8]}"
        
        # Get previous version
        previous = self._configs.get(environment)
        previous_version = previous.version if previous else None
        
        config = EnvironmentConfig(
            config_id=config_id,
            environment=environment,
            values=values,
            version=(previous.version + 1) if previous else 1,
            previous_version=previous_version,
            schema=schema,
        )
        
        # Validate if schema provided
        if schema:
            config.validated = self._validate_config(values, schema)
        
        # Store version history
        if previous:
            self._version_history[environment].append(previous)
        
        self._configs[environment] = config
        
        # Audit log
        self._audit_log.append({
            "event": "config_updated",
            "environment": environment,
            "version": config.version,
            "timestamp": datetime.now().isoformat(),
        })
        
        return config
    
    def get_config(self, environment: str) -> Optional[EnvironmentConfig]:
        """Get configuration for an environment."""
        return self._configs.get(environment)
    
    def get_value(self, environment: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        config = self._configs.get(environment)
        if not config:
            return default
        return config.values.get(key, default)
    
    def _validate_config(self, values: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate configuration against schema."""
        try:
            # Simple validation - check required fields
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in values:
                    return False
            
            # Check field types
            properties = schema.get("properties", {})
            for field, spec in properties.items():
                if field in values:
                    expected_type = spec.get("type")
                    if expected_type == "string" and not isinstance(values[field], str):
                        return False
                    elif expected_type == "integer" and not isinstance(values[field], int):
                        return False
                    elif expected_type == "boolean" and not isinstance(values[field], bool):
                        return False
            
            return True
        except Exception:
            return False
    
    def rollback_config(self, environment: str, target_version: Optional[int] = None) -> bool:
        """Rollback configuration to a previous version."""
        history = self._version_history.get(environment, [])
        if not history:
            return False
        
        if target_version:
            # Find specific version
            for config in history:
                if config.version == target_version:
                    self._configs[environment] = config
                    self._audit_log.append({
                        "event": "config_rollback",
                        "environment": environment,
                        "to_version": target_version,
                        "timestamp": datetime.now().isoformat(),
                    })
                    return True
            return False
        else:
            # Rollback to previous version
            previous = history.pop()
            self._configs[environment] = previous
            self._audit_log.append({
                "event": "config_rollback",
                "environment": environment,
                "to_version": previous.version,
                "timestamp": datetime.now().isoformat(),
            })
            return True
    
    def set_secret_ref(self, environment: str, key: str, secret_ref: str):
        """Set a secret reference for an environment."""
        config = self._configs.get(environment)
        if config:
            config.secret_refs[key] = secret_ref
            config.updated_at = datetime.now()
    
    def get_secret(self, secret_ref: str) -> Optional[str]:
        """Get a secret value by reference (in production, use proper secret management)."""
        return self._secret_store.get(secret_ref)
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get configuration audit log."""
        return self._audit_log.copy()


class DependencyManager:
    """
    Manages dependencies with vulnerability scanning and license checking.
    
    Features:
    - Dependency pinning for stability
    - Vulnerability scanning (safety/pip-audit integration)
    - License compliance checking
    - Update strategy management
    - Compatibility testing
    """
    
    def __init__(self):
        self._dependencies: Dict[str, DependencyInfo] = {}
        self._pinned_versions: Dict[str, str] = {}
        self._vulnerability_cache: Dict[str, List[Dict]] = {}
        self._allowed_licenses: Set[str] = {
            "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
            "ISC", "Python-2.0", "PSF-2.0"
        }
    
    async def scan_dependencies(
        self,
        requirements_file: Optional[str] = None
    ) -> List[DependencyInfo]:
        """Scan and analyze project dependencies."""
        dependencies = []
        
        try:
            # Get installed packages
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                for pkg in packages:
                    dep_info = DependencyInfo(
                        name=pkg["name"],
                        version=pkg["version"],
                    )
                    
                    # Check for vulnerabilities (simplified)
                    dep_info = await self._check_vulnerabilities(dep_info)
                    
                    # Check license
                    dep_info = await self._check_license(dep_info)
                    
                    # Check for updates
                    dep_info = await self._check_updates(dep_info)
                    
                    dependencies.append(dep_info)
                    self._dependencies[dep_info.name] = dep_info
        
        except Exception as e:
            logger.error(f"Error scanning dependencies: {e}")
        
        return dependencies
    
    async def _check_vulnerabilities(self, dep: DependencyInfo) -> DependencyInfo:
        """Check for known vulnerabilities in a dependency."""
        try:
            # In production, integrate with safety or pip-audit
            # This is a simplified check
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", dep.name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Placeholder for actual vulnerability checking
            # In production, use: safety check --json or pip-audit
            dep.has_vulnerability = False
            
        except Exception:
            pass
        
        return dep
    
    async def _check_license(self, dep: DependencyInfo) -> DependencyInfo:
        """Check license compliance for a dependency."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", dep.name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("License:"):
                        dep.license = line.split(":", 1)[1].strip()
                        dep.license_compliant = any(
                            allowed in (dep.license or "")
                            for allowed in self._allowed_licenses
                        )
                        break
        
        except Exception:
            pass
        
        return dep
    
    async def _check_updates(self, dep: DependencyInfo) -> DependencyInfo:
        """Check if updates are available for a dependency."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "index", "versions", dep.name],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                # Parse latest version
                match = re.search(r"LATEST:\s+(\S+)", result.stdout)
                if match:
                    dep.latest_version = match.group(1)
                    dep.update_available = dep.latest_version != dep.version
        
        except Exception:
            pass
        
        return dep
    
    def pin_dependency(self, name: str, version: str):
        """Pin a dependency to a specific version."""
        self._pinned_versions[name] = version
    
    def get_pinned_version(self, name: str) -> Optional[str]:
        """Get pinned version for a dependency."""
        return self._pinned_versions.get(name)
    
    def add_allowed_license(self, license_name: str):
        """Add a license to the allowed list."""
        self._allowed_licenses.add(license_name)
    
    def get_vulnerable_dependencies(self) -> List[DependencyInfo]:
        """Get list of dependencies with known vulnerabilities."""
        return [
            dep for dep in self._dependencies.values()
            if dep.has_vulnerability
        ]
    
    def get_non_compliant_licenses(self) -> List[DependencyInfo]:
        """Get list of dependencies with non-compliant licenses."""
        return [
            dep for dep in self._dependencies.values()
            if not dep.license_compliant
        ]
    
    def get_outdated_dependencies(self) -> List[DependencyInfo]:
        """Get list of dependencies with available updates."""
        return [
            dep for dep in self._dependencies.values()
            if dep.update_available
        ]


# ============================================================================
# Phase 10.2: Observability and Monitoring
# ============================================================================

class StructuredLogger:
    """
    Structured logging with JSON output, rotation, and sensitive data redaction.
    
    Features:
    - Structured JSON logging
    - Log level management
    - Log rotation and retention
    - Sensitive data redaction
    - Trace context propagation
    """
    
    # Patterns for sensitive data redaction
    SENSITIVE_PATTERNS = [
        (r'password["\s:=]+["\']?[\w\-@#$%^&*!]+["\']?', 'password=***REDACTED***'),
        (r'api[_-]?key["\s:=]+["\']?[\w\-]+["\']?', 'api_key=***REDACTED***'),
        (r'token["\s:=]+["\']?[\w\-\.]+["\']?', 'token=***REDACTED***'),
        (r'secret["\s:=]+["\']?[\w\-]+["\']?', 'secret=***REDACTED***'),
        (r'bearer\s+[\w\-\.]+', 'bearer ***REDACTED***'),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***EMAIL***'),
    ]
    
    def __init__(
        self,
        name: str = "proxima",
        level: LogLevel = LogLevel.INFO,
        output_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_redaction: bool = True
    ):
        self._name = name
        self._level = level
        self._output_file = output_file
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._enable_redaction = enable_redaction
        self._log_buffer: List[LogEntry] = []
        self._trace_context: Dict[str, str] = {}
        self._lock = threading.Lock()
        
        # Configure Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self._level_to_python_level(level))
    
    def _level_to_python_level(self, level: LogLevel) -> int:
        """Convert LogLevel to Python logging level."""
        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping.get(level, logging.INFO)
    
    def _redact_sensitive(self, message: str) -> Tuple[str, List[str]]:
        """Redact sensitive information from log message."""
        redacted_fields = []
        
        if not self._enable_redaction:
            return message, redacted_fields
        
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
                redacted_fields.append(pattern)
        
        return message, redacted_fields
    
    def set_trace_context(self, trace_id: str, span_id: Optional[str] = None):
        """Set trace context for correlation."""
        self._trace_context = {
            "trace_id": trace_id,
            "span_id": span_id,
        }
    
    def clear_trace_context(self):
        """Clear trace context."""
        self._trace_context = {}
    
    def log(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ) -> LogEntry:
        """Log a structured message."""
        # Redact sensitive data
        redacted_message, redacted_fields = self._redact_sensitive(message)
        
        # Get caller info
        frame = sys._getframe(2) if hasattr(sys, '_getframe') else None
        
        entry = LogEntry(
            log_id=f"log_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            level=level,
            message=redacted_message,
            logger_name=self._name,
            module=frame.f_code.co_filename if frame else "",
            function=frame.f_code.co_name if frame else "",
            line_number=frame.f_lineno if frame else 0,
            data=data or {},
            trace_id=self._trace_context.get("trace_id"),
            span_id=self._trace_context.get("span_id"),
            redacted_fields=redacted_fields,
        )
        
        # Add exception info
        if exc_info:
            entry.exception_type = type(exc_info).__name__
            entry.exception_message = str(exc_info)
            entry.stack_trace = traceback.format_exc()
        
        # Buffer entry
        with self._lock:
            self._log_buffer.append(entry)
            
            # Keep buffer size manageable
            if len(self._log_buffer) > 10000:
                self._log_buffer = self._log_buffer[-5000:]
        
        # Log to Python logger
        python_level = self._level_to_python_level(level)
        self._logger.log(python_level, json.dumps(entry.to_dict()))
        
        return entry
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Log debug message."""
        return self.log(LogLevel.DEBUG, message, data)
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Log info message."""
        return self.log(LogLevel.INFO, message, data)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Log warning message."""
        return self.log(LogLevel.WARNING, message, data)
    
    def error(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ) -> LogEntry:
        """Log error message."""
        return self.log(LogLevel.ERROR, message, data, exc_info)
    
    def critical(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ) -> LogEntry:
        """Log critical message."""
        return self.log(LogLevel.CRITICAL, message, data, exc_info)
    
    def get_recent_logs(
        self,
        count: int = 100,
        level: Optional[LogLevel] = None
    ) -> List[LogEntry]:
        """Get recent log entries."""
        with self._lock:
            logs = self._log_buffer.copy()
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        return logs[-count:]
    
    def search_logs(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[LogEntry]:
        """Search logs by message content."""
        with self._lock:
            logs = self._log_buffer.copy()
        
        results = []
        for log in logs:
            # Time filter
            if start_time and log.timestamp < start_time:
                continue
            if end_time and log.timestamp > end_time:
                continue
            
            # Content filter
            if query.lower() in log.message.lower():
                results.append(log)
        
        return results


class MetricsCollector:
    """
    Collects and manages application metrics.
    
    Features:
    - Counter, gauge, histogram, summary metric types
    - Custom labels/tags for dimensions
    - Metric aggregation and rollup
    - Dashboard visualization support
    - Alerting integration
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._retention_hours: int = 24
        self._lock = threading.Lock()
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        
        with self._lock:
            self._counters[key] += value
            current_value = self._counters[key]
        
        metric = Metric(
            metric_id=f"metric_{uuid.uuid4().hex[:8]}",
            name=name,
            metric_type=MetricType.COUNTER,
            value=current_value,
            labels=labels or {},
            description=f"Counter: {name}",
        )
        
        self._store_metric(name, metric)
        return metric
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        
        with self._lock:
            self._gauges[key] = value
        
        metric = Metric(
            metric_id=f"metric_{uuid.uuid4().hex[:8]}",
            name=name,
            metric_type=MetricType.GAUGE,
            value=value,
            labels=labels or {},
            description=f"Gauge: {name}",
        )
        
        self._store_metric(name, metric)
        return metric
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Observe a value for a histogram metric."""
        key = self._make_key(name, labels)
        
        with self._lock:
            self._histograms[key].append(value)
            # Keep last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-500:]
        
        metric = Metric(
            metric_id=f"metric_{uuid.uuid4().hex[:8]}",
            name=name,
            metric_type=MetricType.HISTOGRAM,
            value=value,
            labels=labels or {},
            description=f"Histogram: {name}",
        )
        
        self._store_metric(name, metric)
        return metric
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _store_metric(self, name: str, metric: Metric):
        """Store a metric data point."""
        with self._lock:
            self._metrics[name].append(metric)
            
            # Cleanup old metrics
            cutoff = datetime.now() - timedelta(hours=self._retention_hours)
            self._metrics[name] = [
                m for m in self._metrics[name]
                if m.timestamp > cutoff
            ]
    
    def get_counter_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)
    
    def get_gauge_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key)
    
    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "sum": sum(sorted_values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "p50": sorted_values[count // 2],
            "p90": sorted_values[int(count * 0.9)],
            "p99": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1],
        }
    
    def get_metric_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Metric]:
        """Get time series data for a metric."""
        with self._lock:
            metrics = self._metrics.get(name, []).copy()
        
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        return metrics
    
    def get_all_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        return list(self._metrics.keys())
    
    # Pre-defined application metrics
    def record_llm_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool
    ):
        """Record LLM request metrics."""
        labels = {"model": model}
        
        self.increment_counter("llm_requests_total", labels=labels)
        if not success:
            self.increment_counter("llm_requests_failed", labels=labels)
        
        self.increment_counter("llm_tokens_input_total", value=input_tokens, labels=labels)
        self.increment_counter("llm_tokens_output_total", value=output_tokens, labels=labels)
        self.observe_histogram("llm_latency_ms", latency_ms, labels=labels)
    
    def record_tool_execution(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool
    ):
        """Record tool execution metrics."""
        labels = {"tool": tool_name}
        
        self.increment_counter("tool_executions_total", labels=labels)
        if not success:
            self.increment_counter("tool_executions_failed", labels=labels)
        
        self.observe_histogram("tool_duration_ms", duration_ms, labels=labels)


class DistributedTracer:
    """
    Distributed tracing for request flow tracking.
    
    Features:
    - Trace and span management
    - Parent-child span relationships
    - Trace context propagation
    - Performance profiling
    - Trace visualization support
    """
    
    def __init__(self, service_name: str = "proxima"):
        self._service_name = service_name
        self._active_traces: Dict[str, TraceSpan] = {}
        self._completed_traces: List[TraceSpan] = []
        self._current_trace_id: Optional[str] = None
        self._span_stack: List[str] = []
        self._lock = threading.Lock()
    
    def start_trace(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Start a new trace."""
        trace_id = f"trace_{uuid.uuid4().hex[:16]}"
        span_id = f"span_{uuid.uuid4().hex[:8]}"
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            service_name=self._service_name,
            attributes=attributes or {},
        )
        
        with self._lock:
            self._active_traces[span_id] = span
            self._current_trace_id = trace_id
            self._span_stack.append(span_id)
        
        return span
    
    def start_span(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Start a new span within the current trace."""
        with self._lock:
            trace_id = self._current_trace_id or f"trace_{uuid.uuid4().hex[:16]}"
            parent_span_id = self._span_stack[-1] if self._span_stack else None
        
        span_id = f"span_{uuid.uuid4().hex[:8]}"
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            parent_span_id=parent_span_id,
            service_name=self._service_name,
            attributes=attributes or {},
        )
        
        with self._lock:
            self._active_traces[span_id] = span
            self._span_stack.append(span_id)
        
        return span
    
    def end_span(
        self,
        span: TraceSpan,
        status: str = "ok",
        error_message: Optional[str] = None
    ):
        """End a span."""
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        span.error_message = error_message
        
        with self._lock:
            if span.span_id in self._active_traces:
                del self._active_traces[span.span_id]
            
            if self._span_stack and self._span_stack[-1] == span.span_id:
                self._span_stack.pop()
            
            self._completed_traces.append(span)
            
            # Keep limited history
            if len(self._completed_traces) > 1000:
                self._completed_traces = self._completed_traces[-500:]
    
    def add_event(self, span: TraceSpan, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to a span."""
        event = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        }
        span.events.append(event)
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            spans = [
                span for span in self._completed_traces
                if span.trace_id == trace_id
            ]
            # Include active spans too
            for span in self._active_traces.values():
                if span.trace_id == trace_id:
                    spans.append(span)
        
        return sorted(spans, key=lambda s: s.start_time)
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID for context propagation."""
        return self._current_trace_id
    
    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID."""
        with self._lock:
            return self._span_stack[-1] if self._span_stack else None
    
    def get_slow_spans(self, threshold_ms: float = 1000.0) -> List[TraceSpan]:
        """Get spans that exceeded duration threshold."""
        with self._lock:
            return [
                span for span in self._completed_traces
                if span.duration_ms > threshold_ms
            ]


class HealthChecker:
    """
    Health check system with liveness and readiness probes.
    
    Features:
    - Comprehensive health endpoints
    - Liveness and readiness probes
    - Dependency health checks
    - Alert integration
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._check_interval_seconds: int = 30
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
        critical: bool = False
    ):
        """Register a health check."""
        self._checks[name] = check_func
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheckResult(
                check_id=f"check_{uuid.uuid4().hex[:8]}",
                check_name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found",
            )
        
        start_time = time.time()
        try:
            result = self._checks[name]()
            result.response_time_ms = (time.time() - start_time) * 1000
            self._last_results[name] = result
            return result
        except Exception as e:
            result = HealthCheckResult(
                check_id=f"check_{uuid.uuid4().hex[:8]}",
                check_name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
            )
            self._last_results[name] = result
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name in self._checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in self._last_results.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED
    
    def liveness_probe(self) -> HealthCheckResult:
        """Kubernetes liveness probe - is the application alive?"""
        return HealthCheckResult(
            check_id=f"liveness_{uuid.uuid4().hex[:8]}",
            check_name="liveness",
            status=HealthStatus.HEALTHY,
            message="Application is alive",
        )
    
    def readiness_probe(self) -> HealthCheckResult:
        """Kubernetes readiness probe - is the application ready to serve traffic?"""
        overall = self.get_overall_status()
        
        if overall == HealthStatus.HEALTHY:
            return HealthCheckResult(
                check_id=f"readiness_{uuid.uuid4().hex[:8]}",
                check_name="readiness",
                status=HealthStatus.HEALTHY,
                message="Application is ready",
            )
        else:
            return HealthCheckResult(
                check_id=f"readiness_{uuid.uuid4().hex[:8]}",
                check_name="readiness",
                status=overall,
                message="Application not fully ready",
            )
    
    # Built-in health checks
    def check_disk_space(self, min_free_gb: float = 1.0) -> HealthCheckResult:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage("/")
            free_gb = disk.free / (1024 ** 3)
            
            if free_gb >= min_free_gb:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_gb:.2f} GB free"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Low disk space: {free_gb:.2f} GB free"
            
            return HealthCheckResult(
                check_id=f"disk_{uuid.uuid4().hex[:8]}",
                check_name="disk_space",
                status=status,
                message=message,
                details={"free_gb": free_gb, "total_gb": disk.total / (1024 ** 3)},
            )
        except Exception as e:
            return HealthCheckResult(
                check_id=f"disk_{uuid.uuid4().hex[:8]}",
                check_name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)}",
            )
    
    def check_memory(self, max_usage_percent: float = 90.0) -> HealthCheckResult:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent <= max_usage_percent:
                status = HealthStatus.HEALTHY
                message = f"Memory OK: {usage_percent:.1f}% used"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"High memory: {usage_percent:.1f}% used"
            
            return HealthCheckResult(
                check_id=f"memory_{uuid.uuid4().hex[:8]}",
                check_name="memory",
                status=status,
                message=message,
                details={"usage_percent": usage_percent, "available_mb": memory.available / (1024 ** 2)},
            )
        except Exception as e:
            return HealthCheckResult(
                check_id=f"memory_{uuid.uuid4().hex[:8]}",
                check_name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)}",
            )
    
    def check_cpu(self, max_usage_percent: float = 95.0) -> HealthCheckResult:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent <= max_usage_percent:
                status = HealthStatus.HEALTHY
                message = f"CPU OK: {cpu_percent:.1f}% used"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"High CPU: {cpu_percent:.1f}% used"
            
            return HealthCheckResult(
                check_id=f"cpu_{uuid.uuid4().hex[:8]}",
                check_name="cpu",
                status=status,
                message=message,
                details={"usage_percent": cpu_percent},
            )
        except Exception as e:
            return HealthCheckResult(
                check_id=f"cpu_{uuid.uuid4().hex[:8]}",
                check_name="cpu",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)}",
            )


class AlertManager:
    """
    Alert management with severity levels and deduplication.
    
    Features:
    - Alert creation with severity levels
    - Alert deduplication
    - Alert aggregation
    - Notification integration
    - Alert history
    """
    
    def __init__(self):
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._notification_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
    
    def fire_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        source: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Alert:
        """Fire an alert."""
        # Create fingerprint for deduplication
        fingerprint = hashlib.md5(
            f"{name}:{message}:{json.dumps(labels or {})}".encode()
        ).hexdigest()[:16]
        
        with self._lock:
            # Check for existing alert
            if fingerprint in self._active_alerts:
                existing = self._active_alerts[fingerprint]
                existing.occurrence_count += 1
                return existing
        
        alert = Alert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            name=name,
            severity=severity,
            message=message,
            source=source,
            labels=labels or {},
            fingerprint=fingerprint,
        )
        
        with self._lock:
            self._active_alerts[fingerprint] = alert
            self._alert_history.append(alert)
        
        # Notify handlers
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception:
                pass
        
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        with self._lock:
            for fingerprint, alert in self._active_alerts.items():
                if alert.alert_id == alert_id:
                    alert.is_active = False
                    alert.resolved_at = datetime.now()
                    del self._active_alerts[fingerprint]
                    break
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._active_alerts.values():
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    break
    
    def add_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        severity: AlertSeverity,
        message: str
    ):
        """Add an alerting rule."""
        self._rules[name] = {
            "condition": condition,
            "severity": severity,
            "message": message,
        }
    
    def evaluate_rules(self):
        """Evaluate all alerting rules."""
        for name, rule in self._rules.items():
            try:
                if rule["condition"]():
                    self.fire_alert(
                        name=name,
                        severity=rule["severity"],
                        message=rule["message"],
                    )
            except Exception:
                pass
    
    def register_notification_handler(self, handler: Callable[[Alert], None]):
        """Register a notification handler."""
        self._notification_handlers.append(handler)
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active alerts."""
        with self._lock:
            alerts = list(self._active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.fired_at, reverse=True)
    
    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """Get alert history."""
        with self._lock:
            alerts = self._alert_history.copy()
        
        if start_time:
            alerts = [a for a in alerts if a.fired_at >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.fired_at <= end_time]
        
        return alerts


# ============================================================================
# Phase 10.3: Continuous Improvement
# ============================================================================

class UsageAnalytics:
    """
    Anonymous usage tracking and analytics.
    
    Features:
    - Anonymous event tracking
    - Feature adoption metrics
    - Success rate tracking
    - Error rate monitoring
    - Performance benchmarking
    """
    
    def __init__(self):
        self._events: List[UsageEvent] = []
        self._feature_counts: Dict[str, int] = defaultdict(int)
        self._operation_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total": 0, "success": 0, "duration_sum": 0}
        )
        self._session_id = f"session_{uuid.uuid4().hex[:8]}"
        self._lock = threading.Lock()
    
    def track_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        feature_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        success: bool = True,
        duration_ms: float = 0.0
    ) -> UsageEvent:
        """Track a usage event."""
        event = UsageEvent(
            event_id=f"event_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            session_id=self._session_id,
            data=data or {},
            feature_name=feature_name,
            operation_name=operation_name,
            success=success,
            duration_ms=duration_ms,
        )
        
        with self._lock:
            self._events.append(event)
            
            # Update feature counts
            if feature_name:
                self._feature_counts[feature_name] += 1
            
            # Update operation stats
            if operation_name:
                self._operation_stats[operation_name]["total"] += 1
                if success:
                    self._operation_stats[operation_name]["success"] += 1
                self._operation_stats[operation_name]["duration_sum"] += duration_ms
            
            # Keep limited history
            if len(self._events) > 10000:
                self._events = self._events[-5000:]
        
        return event
    
    def get_feature_adoption(self) -> Dict[str, int]:
        """Get feature adoption counts."""
        with self._lock:
            return dict(self._feature_counts)
    
    def get_operation_success_rate(self, operation_name: str) -> float:
        """Get success rate for an operation."""
        with self._lock:
            stats = self._operation_stats.get(operation_name)
        
        if not stats or stats["total"] == 0:
            return 0.0
        
        return stats["success"] / stats["total"]
    
    def get_operation_avg_duration(self, operation_name: str) -> float:
        """Get average duration for an operation."""
        with self._lock:
            stats = self._operation_stats.get(operation_name)
        
        if not stats or stats["total"] == 0:
            return 0.0
        
        return stats["duration_sum"] / stats["total"]
    
    def get_error_rate(self, time_window_minutes: int = 60) -> float:
        """Get error rate within a time window."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            recent_events = [e for e in self._events if e.timestamp > cutoff]
        
        if not recent_events:
            return 0.0
        
        errors = sum(1 for e in recent_events if not e.success)
        return errors / len(recent_events)
    
    def get_event_count_by_type(
        self,
        time_window_minutes: int = 60
    ) -> Dict[str, int]:
        """Get event counts by type within a time window."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            recent_events = [e for e in self._events if e.timestamp > cutoff]
        
        counts: Dict[str, int] = defaultdict(int)
        for event in recent_events:
            counts[event.event_type] += 1
        
        return dict(counts)


class FeedbackManager:
    """
    User feedback collection and analysis.
    
    Features:
    - Feedback collection
    - Rating system
    - Feedback categorization
    - Feedback analytics
    """
    
    def __init__(self):
        self._feedback: List[UserFeedback] = []
        self._categories: Set[str] = set()
        self._lock = threading.Lock()
    
    def submit_feedback(
        self,
        feedback_type: FeedbackType,
        rating: Optional[int] = None,
        comment: str = "",
        category: str = "",
        request_id: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> UserFeedback:
        """Submit user feedback."""
        feedback = UserFeedback(
            feedback_id=f"feedback_{uuid.uuid4().hex[:8]}",
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            request_id=request_id,
            category=category,
            tags=tags or set(),
        )
        
        with self._lock:
            self._feedback.append(feedback)
            if category:
                self._categories.add(category)
        
        return feedback
    
    def get_average_rating(
        self,
        time_window_days: int = 7,
        category: Optional[str] = None
    ) -> float:
        """Get average rating within a time window."""
        cutoff = datetime.now() - timedelta(days=time_window_days)
        
        with self._lock:
            feedback = [
                f for f in self._feedback
                if f.timestamp > cutoff and f.rating is not None
            ]
            
            if category:
                feedback = [f for f in feedback if f.category == category]
        
        if not feedback:
            return 0.0
        
        return sum(f.rating for f in feedback) / len(feedback)
    
    def get_feedback_by_type(
        self,
        feedback_type: FeedbackType,
        limit: int = 100
    ) -> List[UserFeedback]:
        """Get feedback by type."""
        with self._lock:
            feedback = [
                f for f in self._feedback
                if f.feedback_type == feedback_type
            ]
        
        return feedback[-limit:]
    
    def get_feedback_summary(
        self,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """Get feedback summary statistics."""
        cutoff = datetime.now() - timedelta(days=time_window_days)
        
        with self._lock:
            recent = [f for f in self._feedback if f.timestamp > cutoff]
        
        if not recent:
            return {"total": 0}
        
        type_counts = defaultdict(int)
        ratings = []
        
        for f in recent:
            type_counts[f.feedback_type.value] += 1
            if f.rating is not None:
                ratings.append(f.rating)
        
        return {
            "total": len(recent),
            "by_type": dict(type_counts),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "positive_rate": type_counts.get("positive", 0) / len(recent),
        }


class ModelPerformanceMonitor:
    """
    LLM model performance monitoring.
    
    Features:
    - Response quality tracking
    - Hallucination detection
    - Tool selection accuracy
    - Cost vs quality analysis
    - Model drift detection
    """
    
    def __init__(self):
        self._metrics: List[ModelPerformanceMetric] = []
        self._baseline_metrics: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    def record_performance(
        self,
        model_name: str,
        tool_selection_accuracy: float,
        parameter_extraction_accuracy: float,
        response_quality_score: float,
        hallucination_detected: bool = False,
        hallucination_type: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0
    ) -> ModelPerformanceMetric:
        """Record model performance metrics."""
        metric = ModelPerformanceMetric(
            metric_id=f"perf_{uuid.uuid4().hex[:8]}",
            model_name=model_name,
            tool_selection_accuracy=tool_selection_accuracy,
            parameter_extraction_accuracy=parameter_extraction_accuracy,
            response_quality_score=response_quality_score,
            hallucination_detected=hallucination_detected,
            hallucination_type=hallucination_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        
        with self._lock:
            self._metrics.append(metric)
            
            # Keep limited history
            if len(self._metrics) > 10000:
                self._metrics = self._metrics[-5000:]
        
        return metric
    
    def set_baseline(
        self,
        model_name: str,
        tool_selection_accuracy: float,
        parameter_extraction_accuracy: float,
        response_quality_score: float,
        latency_ms: float
    ):
        """Set baseline metrics for drift detection."""
        self._baseline_metrics[model_name] = {
            "tool_selection_accuracy": tool_selection_accuracy,
            "parameter_extraction_accuracy": parameter_extraction_accuracy,
            "response_quality_score": response_quality_score,
            "latency_ms": latency_ms,
        }
    
    def detect_drift(
        self,
        model_name: str,
        threshold_percent: float = 10.0
    ) -> Dict[str, bool]:
        """Detect model drift from baseline."""
        baseline = self._baseline_metrics.get(model_name)
        if not baseline:
            return {"drift_detected": False, "reason": "no_baseline"}
        
        # Get recent metrics
        with self._lock:
            recent = [
                m for m in self._metrics[-100:]
                if m.model_name == model_name
            ]
        
        if not recent:
            return {"drift_detected": False, "reason": "no_recent_data"}
        
        # Calculate current averages
        current = {
            "tool_selection_accuracy": sum(m.tool_selection_accuracy for m in recent) / len(recent),
            "parameter_extraction_accuracy": sum(m.parameter_extraction_accuracy for m in recent) / len(recent),
            "response_quality_score": sum(m.response_quality_score for m in recent) / len(recent),
            "latency_ms": sum(m.latency_ms for m in recent) / len(recent),
        }
        
        # Check for drift
        drift_metrics = {}
        for metric_name in baseline:
            baseline_value = baseline[metric_name]
            current_value = current[metric_name]
            
            if baseline_value > 0:
                change_percent = abs(current_value - baseline_value) / baseline_value * 100
                drift_metrics[metric_name] = change_percent > threshold_percent
        
        return {
            "drift_detected": any(drift_metrics.values()),
            "drifted_metrics": drift_metrics,
            "current": current,
            "baseline": baseline,
        }
    
    def get_hallucination_rate(
        self,
        model_name: str,
        time_window_hours: int = 24
    ) -> float:
        """Get hallucination rate for a model."""
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        
        with self._lock:
            recent = [
                m for m in self._metrics
                if m.model_name == model_name and m.timestamp > cutoff
            ]
        
        if not recent:
            return 0.0
        
        hallucinations = sum(1 for m in recent if m.hallucination_detected)
        return hallucinations / len(recent)
    
    def get_cost_efficiency(
        self,
        model_name: str,
        time_window_hours: int = 24
    ) -> Dict[str, float]:
        """Get cost efficiency metrics for a model."""
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        
        with self._lock:
            recent = [
                m for m in self._metrics
                if m.model_name == model_name and m.timestamp > cutoff
            ]
        
        if not recent:
            return {}
        
        total_cost = sum(m.cost_usd for m in recent)
        total_requests = len(recent)
        avg_quality = sum(m.response_quality_score for m in recent) / len(recent)
        
        return {
            "total_cost": total_cost,
            "cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "avg_quality_score": avg_quality,
            "quality_per_dollar": avg_quality / total_cost if total_cost > 0 else 0,
        }
    
    def get_model_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across models."""
        with self._lock:
            recent = self._metrics[-1000:]
        
        models: Dict[str, List[ModelPerformanceMetric]] = defaultdict(list)
        for m in recent:
            models[m.model_name].append(m)
        
        comparison = {}
        for model_name, metrics in models.items():
            comparison[model_name] = {
                "avg_tool_accuracy": sum(m.tool_selection_accuracy for m in metrics) / len(metrics),
                "avg_param_accuracy": sum(m.parameter_extraction_accuracy for m in metrics) / len(metrics),
                "avg_quality": sum(m.response_quality_score for m in metrics) / len(metrics),
                "avg_latency_ms": sum(m.latency_ms for m in metrics) / len(metrics),
                "hallucination_rate": sum(1 for m in metrics if m.hallucination_detected) / len(metrics),
                "total_cost": sum(m.cost_usd for m in metrics),
            }
        
        return comparison


class ExperimentFramework:
    """
    Experiment framework for A/B testing and feature rollouts.
    
    Features:
    - Experiment creation and management
    - A/B test variant assignment
    - Gradual rollout mechanisms
    - Success metrics tracking
    """
    
    def __init__(self):
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def create_experiment(
        self,
        name: str,
        variants: List[str],
        traffic_allocation: Optional[Dict[str, float]] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create a new experiment."""
        if not variants:
            variants = ["control", "treatment"]
        
        if not traffic_allocation:
            # Equal split
            allocation = {v: 1.0 / len(variants) for v in variants}
        else:
            allocation = traffic_allocation
        
        experiment = {
            "name": name,
            "variants": variants,
            "traffic_allocation": allocation,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "participant_count": 0,
        }
        
        self._experiments[name] = experiment
        return experiment
    
    def get_variant(self, experiment_name: str, user_id: str) -> Optional[str]:
        """Get variant assignment for a user."""
        experiment = self._experiments.get(experiment_name)
        if not experiment or not experiment["active"]:
            return None
        
        # Check existing assignment
        with self._lock:
            if experiment_name in self._assignments.get(user_id, {}):
                return self._assignments[user_id][experiment_name]
        
        # Assign variant based on traffic allocation
        allocation = experiment["traffic_allocation"]
        
        # Use hash for consistent assignment
        hash_value = int(hashlib.md5(
            f"{experiment_name}:{user_id}".encode()
        ).hexdigest(), 16) % 1000 / 1000.0
        
        cumulative = 0.0
        assigned_variant = experiment["variants"][-1]  # Default to last
        
        for variant, percentage in allocation.items():
            cumulative += percentage
            if hash_value < cumulative:
                assigned_variant = variant
                break
        
        # Store assignment
        with self._lock:
            self._assignments[user_id][experiment_name] = assigned_variant
            experiment["participant_count"] += 1
        
        return assigned_variant
    
    def record_result(
        self,
        experiment_name: str,
        user_id: str,
        metric_name: str,
        value: float
    ):
        """Record a result for an experiment."""
        variant = self._assignments.get(user_id, {}).get(experiment_name)
        if not variant:
            return
        
        result = {
            "user_id": user_id,
            "variant": variant,
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
        
        with self._lock:
            self._results[experiment_name].append(result)
    
    def get_experiment_results(
        self,
        experiment_name: str
    ) -> Dict[str, Dict[str, float]]:
        """Get experiment results by variant."""
        with self._lock:
            results = self._results.get(experiment_name, []).copy()
        
        if not results:
            return {}
        
        # Group by variant and metric
        variant_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        for r in results:
            variant_metrics[r["variant"]][r["metric_name"]].append(r["value"])
        
        # Calculate averages
        summary = {}
        for variant, metrics in variant_metrics.items():
            summary[variant] = {
                metric: sum(values) / len(values)
                for metric, values in metrics.items()
            }
        
        return summary
    
    def stop_experiment(self, experiment_name: str):
        """Stop an experiment."""
        if experiment_name in self._experiments:
            self._experiments[experiment_name]["active"] = False
            self._experiments[experiment_name]["stopped_at"] = datetime.now().isoformat()


# ============================================================================
# Main Integration Classes
# ============================================================================

class DeploymentAndMonitoring:
    """
    Main integration class for deployment and monitoring capabilities.
    
    Integrates all Phase 10 components:
    - Production Deployment (10.1)
    - Observability and Monitoring (10.2)
    - Continuous Improvement (10.3)
    """
    
    def __init__(self):
        # Phase 10.1: Production Deployment
        self._deployment_pipeline = DeploymentPipeline()
        self._configuration_manager = ConfigurationManager()
        self._dependency_manager = DependencyManager()
        
        # Phase 10.2: Observability and Monitoring
        self._logger = StructuredLogger()
        self._metrics = MetricsCollector()
        self._tracer = DistributedTracer()
        self._health_checker = HealthChecker()
        self._alert_manager = AlertManager()
        
        # Phase 10.3: Continuous Improvement
        self._usage_analytics = UsageAnalytics()
        self._feedback_manager = FeedbackManager()
        self._model_monitor = ModelPerformanceMonitor()
        self._experiment_framework = ExperimentFramework()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self._health_checker.register_check(
            "disk_space",
            lambda: self._health_checker.check_disk_space()
        )
        self._health_checker.register_check(
            "memory",
            lambda: self._health_checker.check_memory()
        )
        self._health_checker.register_check(
            "cpu",
            lambda: self._health_checker.check_cpu()
        )
    
    # Deployment operations
    async def deploy(
        self,
        version: str,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        environment: str = "production",
        config: Optional[Dict[str, Any]] = None
    ) -> DeploymentResult:
        """Deploy a new version."""
        self._logger.info(
            f"Starting deployment of version {version}",
            data={"strategy": strategy.value, "environment": environment}
        )
        
        deployment_config = await self._deployment_pipeline.create_deployment(
            version=version,
            strategy=strategy,
            environment=environment,
            config=config,
        )
        
        # Health checker for verification
        def health_checker():
            return self._health_checker.readiness_probe()
        
        result = await self._deployment_pipeline.execute_deployment(
            config=deployment_config,
            health_checker=health_checker,
        )
        
        # Track deployment metrics
        self._metrics.increment_counter(
            "deployments_total",
            labels={"environment": environment, "status": result.status.value}
        )
        
        return result
    
    # Configuration operations
    def set_environment_config(
        self,
        environment: str,
        values: Dict[str, Any]
    ) -> EnvironmentConfig:
        """Set environment configuration."""
        return self._configuration_manager.set_config(environment, values)
    
    def get_config_value(self, environment: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._configuration_manager.get_value(environment, key, default)
    
    # Dependency operations
    async def scan_dependencies(self) -> List[DependencyInfo]:
        """Scan project dependencies."""
        return await self._dependency_manager.scan_dependencies()
    
    # Logging operations
    def log(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> LogEntry:
        """Log a message."""
        return self._logger.log(level, message, data)
    
    # Metrics operations
    def increment_metric(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Increment a counter metric."""
        return self._metrics.increment_counter(name, value, labels)
    
    def set_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Set a gauge metric."""
        return self._metrics.set_gauge(name, value, labels)
    
    # Tracing operations
    def start_trace(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Start a new trace."""
        return self._tracer.start_trace(operation_name, attributes)
    
    def start_span(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Start a new span."""
        return self._tracer.start_span(operation_name, attributes)
    
    def end_span(self, span: TraceSpan, status: str = "ok"):
        """End a span."""
        self._tracer.end_span(span, status)
    
    # Health check operations
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        return self._health_checker.run_all_checks()
    
    def get_health_status(self) -> HealthStatus:
        """Get overall health status."""
        return self._health_checker.get_overall_status()
    
    # Alert operations
    def fire_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str
    ) -> Alert:
        """Fire an alert."""
        return self._alert_manager.fire_alert(name, severity, message)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self._alert_manager.get_active_alerts()
    
    # Usage analytics operations
    def track_usage(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        feature_name: Optional[str] = None,
        success: bool = True,
        duration_ms: float = 0.0
    ) -> UsageEvent:
        """Track a usage event."""
        return self._usage_analytics.track_event(
            event_type=event_type,
            data=data,
            feature_name=feature_name,
            success=success,
            duration_ms=duration_ms,
        )
    
    # Feedback operations
    def submit_feedback(
        self,
        feedback_type: FeedbackType,
        rating: Optional[int] = None,
        comment: str = ""
    ) -> UserFeedback:
        """Submit user feedback."""
        return self._feedback_manager.submit_feedback(
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
        )
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary."""
        return self._feedback_manager.get_feedback_summary()
    
    # Model monitoring operations
    def record_model_performance(
        self,
        model_name: str,
        tool_selection_accuracy: float,
        parameter_extraction_accuracy: float,
        response_quality_score: float,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> ModelPerformanceMetric:
        """Record model performance."""
        return self._model_monitor.record_performance(
            model_name=model_name,
            tool_selection_accuracy=tool_selection_accuracy,
            parameter_extraction_accuracy=parameter_extraction_accuracy,
            response_quality_score=response_quality_score,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    
    def detect_model_drift(self, model_name: str) -> Dict[str, bool]:
        """Detect model drift."""
        return self._model_monitor.detect_drift(model_name)
    
    # Experiment operations
    def create_experiment(
        self,
        name: str,
        variants: List[str],
        description: str = ""
    ) -> Dict[str, Any]:
        """Create an experiment."""
        return self._experiment_framework.create_experiment(
            name=name,
            variants=variants,
            description=description,
        )
    
    def get_experiment_variant(self, experiment_name: str, user_id: str) -> Optional[str]:
        """Get experiment variant for a user."""
        return self._experiment_framework.get_variant(experiment_name, user_id)
    
    def get_experiment_results(self, experiment_name: str) -> Dict[str, Dict[str, float]]:
        """Get experiment results."""
        return self._experiment_framework.get_experiment_results(experiment_name)
    
    # Property accessors for sub-components
    @property
    def deployment_pipeline(self) -> DeploymentPipeline:
        return self._deployment_pipeline
    
    @property
    def configuration_manager(self) -> ConfigurationManager:
        return self._configuration_manager
    
    @property
    def dependency_manager(self) -> DependencyManager:
        return self._dependency_manager
    
    @property
    def logger(self) -> StructuredLogger:
        return self._logger
    
    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics
    
    @property
    def tracer(self) -> DistributedTracer:
        return self._tracer
    
    @property
    def health_checker(self) -> HealthChecker:
        return self._health_checker
    
    @property
    def alert_manager(self) -> AlertManager:
        return self._alert_manager
    
    @property
    def usage_analytics(self) -> UsageAnalytics:
        return self._usage_analytics
    
    @property
    def feedback_manager(self) -> FeedbackManager:
        return self._feedback_manager
    
    @property
    def model_monitor(self) -> ModelPerformanceMonitor:
        return self._model_monitor
    
    @property
    def experiment_framework(self) -> ExperimentFramework:
        return self._experiment_framework


# ============================================================================
# Global Instance and Accessor
# ============================================================================

_deployment_monitoring_instance: Optional[DeploymentAndMonitoring] = None
_instance_lock = threading.Lock()


def get_deployment_and_monitoring() -> DeploymentAndMonitoring:
    """Get the global DeploymentAndMonitoring instance."""
    global _deployment_monitoring_instance
    
    if _deployment_monitoring_instance is None:
        with _instance_lock:
            if _deployment_monitoring_instance is None:
                _deployment_monitoring_instance = DeploymentAndMonitoring()
    
    return _deployment_monitoring_instance


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "DeploymentStrategy",
    "DeploymentStatus",
    "HealthStatus",
    "AlertSeverity",
    "MetricType",
    "LogLevel",
    "FeedbackType",
    
    # Data Classes
    "DeploymentConfig",
    "DeploymentResult",
    "EnvironmentConfig",
    "DependencyInfo",
    "LogEntry",
    "Metric",
    "TraceSpan",
    "HealthCheckResult",
    "Alert",
    "UsageEvent",
    "UserFeedback",
    "ModelPerformanceMetric",
    
    # Phase 10.1: Production Deployment
    "DeploymentPipeline",
    "ConfigurationManager",
    "DependencyManager",
    
    # Phase 10.2: Observability and Monitoring
    "StructuredLogger",
    "MetricsCollector",
    "DistributedTracer",
    "HealthChecker",
    "AlertManager",
    
    # Phase 10.3: Continuous Improvement
    "UsageAnalytics",
    "FeedbackManager",
    "ModelPerformanceMonitor",
    "ExperimentFramework",
    
    # Main Integration Class
    "DeploymentAndMonitoring",
    
    # Global Accessor
    "get_deployment_and_monitoring",
]
