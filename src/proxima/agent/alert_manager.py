"""Alert Manager for Performance Monitoring.

Phase 9: Agent Statistics & Telemetry System

Provides anomaly detection and alerting:
- Threshold-based alerts
- Anomaly detection (statistical)
- Alert history and management
- Notification callbacks
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Deque
from collections import deque

from proxima.utils.logging import get_logger

logger = get_logger("agent.alert_manager")


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    HIGH_LATENCY = "high_latency"
    ERROR_SPIKE = "error_spike"
    TOKEN_LIMIT = "token_limit"
    MEMORY_USAGE = "memory_usage"
    DISK_SPACE = "disk_space"
    BUILD_FAILURE = "build_failure"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    CUSTOM = "custom"


class AlertState(Enum):
    """States of an alert."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass
class AlertThreshold:
    """Threshold configuration for an alert type."""
    alert_type: AlertType
    metric_name: str
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    comparison: str = "gt"  # gt, lt, eq
    window_seconds: int = 60
    cooldown_seconds: int = 300
    enabled: bool = True
    
    def check(self, value: float) -> Optional[AlertSeverity]:
        """Check if value triggers threshold.
        
        Args:
            value: Value to check
            
        Returns:
            AlertSeverity or None
        """
        if not self.enabled:
            return None
        
        if self.comparison == "gt":
            if value >= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= self.error_threshold:
                return AlertSeverity.ERROR
            elif value >= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison == "lt":
            if value <= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= self.error_threshold:
                return AlertSeverity.ERROR
            elif value <= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison == "eq":
            if value == self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value == self.error_threshold:
                return AlertSeverity.ERROR
            elif value == self.warning_threshold:
                return AlertSeverity.WARNING
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "metric_name": self.metric_name,
            "warning_threshold": self.warning_threshold,
            "error_threshold": self.error_threshold,
            "critical_threshold": self.critical_threshold,
            "comparison": self.comparison,
            "window_seconds": self.window_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "enabled": self.enabled,
        }


@dataclass
class Alert:
    """An alert instance."""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float
    state: AlertState = AlertState.ACTIVE
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    
    @property
    def is_active(self) -> bool:
        """Check if alert is active."""
        return self.state == AlertState.ACTIVE
    
    @property
    def datetime(self) -> datetime:
        """Get datetime from timestamp."""
        return datetime.fromtimestamp(self.timestamp)
    
    @property
    def age_seconds(self) -> float:
        """Get alert age in seconds."""
        return time.time() - self.timestamp
    
    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.state = AlertState.ACKNOWLEDGED
        self.acknowledged_at = time.time()
    
    def resolve(self) -> None:
        """Resolve the alert."""
        self.state = AlertState.RESOLVED
        self.resolved_at = time.time()
    
    def dismiss(self) -> None:
        """Dismiss the alert."""
        self.state = AlertState.DISMISSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "state": self.state.value,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "metadata": self.metadata,
            "age_seconds": self.age_seconds,
        }


@dataclass
class AlertStats:
    """Statistics about alerts."""
    total_alerts: int = 0
    active_alerts: int = 0
    acknowledged_alerts: int = 0
    resolved_alerts: int = 0
    alerts_by_type: Dict[str, int] = field(default_factory=dict)
    alerts_by_severity: Dict[str, int] = field(default_factory=dict)


class AnomalyDetector:
    """Statistical anomaly detection.
    
    Uses rolling window statistics to detect outliers.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        threshold_stddev: float = 3.0,
    ):
        """Initialize detector.
        
        Args:
            window_size: Size of rolling window
            threshold_stddev: Standard deviations for anomaly
        """
        self._window_size = window_size
        self._threshold_stddev = threshold_stddev
        self._values: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
    
    def _get_buffer(self, metric_name: str) -> Deque[float]:
        """Get or create buffer for metric."""
        if metric_name not in self._values:
            self._values[metric_name] = deque(maxlen=self._window_size)
        return self._values[metric_name]
    
    def record(self, metric_name: str, value: float) -> Optional[float]:
        """Record value and check for anomaly.
        
        Args:
            metric_name: Metric name
            value: Value to record
            
        Returns:
            Anomaly score if anomaly detected, None otherwise
        """
        with self._lock:
            buffer = self._get_buffer(metric_name)
            
            # Need minimum values for detection
            if len(buffer) < 10:
                buffer.append(value)
                return None
            
            # Calculate statistics
            values = list(buffer)
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            stddev = variance ** 0.5
            
            # Check for anomaly
            if stddev > 0:
                z_score = abs(value - mean) / stddev
                if z_score > self._threshold_stddev:
                    buffer.append(value)
                    return z_score
            
            buffer.append(value)
            return None
    
    def get_baseline(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Get baseline statistics for a metric.
        
        Returns:
            Dict with mean, stddev, min, max or None
        """
        with self._lock:
            if metric_name not in self._values:
                return None
            
            values = list(self._values[metric_name])
            if len(values) < 2:
                return None
            
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            
            return {
                "mean": mean,
                "stddev": variance ** 0.5,
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }


class AlertManager:
    """Manages alerts and notifications.
    
    Features:
    - Threshold-based alerting
    - Anomaly detection
    - Alert history
    - Notification callbacks
    - Cooldown management
    
    Example:
        >>> manager = AlertManager()
        >>> 
        >>> # Add callback
        >>> manager.on_alert(lambda alert: print(f"Alert: {alert.title}"))
        >>> 
        >>> # Check metric
        >>> alert = manager.check_metric("response_time", 5500)
        >>> if alert:
        ...     print(f"Alert triggered: {alert.severity}")
    """
    
    # Default thresholds
    DEFAULT_THRESHOLDS = [
        AlertThreshold(
            alert_type=AlertType.HIGH_LATENCY,
            metric_name="response_time_ms",
            warning_threshold=3000,
            error_threshold=5000,
            critical_threshold=10000,
            comparison="gt",
        ),
        AlertThreshold(
            alert_type=AlertType.ERROR_SPIKE,
            metric_name="error_rate",
            warning_threshold=0.03,
            error_threshold=0.05,
            critical_threshold=0.10,
            comparison="gt",
        ),
        AlertThreshold(
            alert_type=AlertType.TOKEN_LIMIT,
            metric_name="token_usage_percent",
            warning_threshold=70,
            error_threshold=80,
            critical_threshold=90,
            comparison="gt",
        ),
        AlertThreshold(
            alert_type=AlertType.MEMORY_USAGE,
            metric_name="memory_mb",
            warning_threshold=512,
            error_threshold=768,
            critical_threshold=1024,
            comparison="gt",
        ),
        AlertThreshold(
            alert_type=AlertType.DISK_SPACE,
            metric_name="disk_free_gb",
            warning_threshold=5,
            error_threshold=2,
            critical_threshold=1,
            comparison="lt",
        ),
    ]
    
    # Maximum alerts to keep
    MAX_HISTORY = 1000
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_anomaly_detection: bool = True,
    ):
        """Initialize alert manager.
        
        Args:
            config_path: Path to config file
            enable_anomaly_detection: Enable statistical anomaly detection
        """
        self._config_path = Path(config_path) if config_path else None
        self._lock = threading.Lock()
        
        # Thresholds
        self._thresholds: Dict[str, AlertThreshold] = {}
        for threshold in self.DEFAULT_THRESHOLDS:
            self._thresholds[threshold.metric_name] = threshold
        
        # Alert storage
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: Deque[str] = deque(maxlen=self.MAX_HISTORY)
        self._alert_counter = 0
        
        # Cooldown tracking
        self._last_alert_time: Dict[str, float] = {}
        
        # Anomaly detection
        self._anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None
        
        # Callbacks
        self._on_alert: List[Callable[[Alert], None]] = []
        self._on_resolve: List[Callable[[Alert], None]] = []
        
        # Load config
        if self._config_path and self._config_path.exists():
            self._load_config()
        
        logger.info("AlertManager initialized")
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"alert_{int(time.time())}_{self._alert_counter}"
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self._config_path and self._config_path.exists():
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                for threshold_data in data.get("thresholds", []):
                    threshold = AlertThreshold(
                        alert_type=AlertType(threshold_data["alert_type"]),
                        metric_name=threshold_data["metric_name"],
                        warning_threshold=threshold_data["warning_threshold"],
                        error_threshold=threshold_data["error_threshold"],
                        critical_threshold=threshold_data["critical_threshold"],
                        comparison=threshold_data.get("comparison", "gt"),
                        window_seconds=threshold_data.get("window_seconds", 60),
                        cooldown_seconds=threshold_data.get("cooldown_seconds", 300),
                        enabled=threshold_data.get("enabled", True),
                    )
                    self._thresholds[threshold.metric_name] = threshold
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        if not self._config_path:
            return
        
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "thresholds": [t.to_dict() for t in self._thresholds.values()],
            }
            self._config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _check_cooldown(self, metric_name: str, cooldown_seconds: int) -> bool:
        """Check if metric is in cooldown.
        
        Returns:
            True if in cooldown
        """
        last_time = self._last_alert_time.get(metric_name, 0)
        return (time.time() - last_time) < cooldown_seconds
    
    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and register an alert."""
        alert = Alert(
            id=self._generate_alert_id(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=time.time(),
            metric_name=metric_name,
            metric_value=metric_value,
            threshold_value=threshold_value,
            metadata=metadata or {},
        )
        
        self._alerts[alert.id] = alert
        self._alert_history.append(alert.id)
        
        if metric_name:
            self._last_alert_time[metric_name] = time.time()
        
        # Notify callbacks
        for callback in self._on_alert:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"Alert created: [{severity.value}] {title}")
        
        return alert
    
    def check_metric(
        self,
        metric_name: str,
        value: float,
    ) -> Optional[Alert]:
        """Check metric against thresholds.
        
        Args:
            metric_name: Metric name
            value: Current value
            
        Returns:
            Alert if triggered, None otherwise
        """
        with self._lock:
            threshold = self._thresholds.get(metric_name)
            
            if threshold:
                # Check cooldown
                if self._check_cooldown(metric_name, threshold.cooldown_seconds):
                    return None
                
                severity = threshold.check(value)
                if severity:
                    return self._create_alert(
                        alert_type=threshold.alert_type,
                        severity=severity,
                        title=f"{threshold.alert_type.value.replace('_', ' ').title()} Detected",
                        message=f"{metric_name} is {value} (threshold: {threshold.error_threshold})",
                        metric_name=metric_name,
                        metric_value=value,
                        threshold_value=threshold.error_threshold,
                    )
            
            # Check anomaly detection
            if self._anomaly_detector:
                z_score = self._anomaly_detector.record(metric_name, value)
                if z_score and not self._check_cooldown(f"anomaly_{metric_name}", 300):
                    return self._create_alert(
                        alert_type=AlertType.CUSTOM,
                        severity=AlertSeverity.WARNING,
                        title=f"Anomaly Detected: {metric_name}",
                        message=f"Value {value} is {z_score:.1f} standard deviations from mean",
                        metric_name=metric_name,
                        metric_value=value,
                        metadata={"z_score": z_score},
                    )
            
            return None
    
    def create_custom_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create a custom alert.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            metadata: Additional metadata
            
        Returns:
            Created alert
        """
        with self._lock:
            return self._create_alert(
                alert_type=AlertType.CUSTOM,
                severity=severity,
                title=title,
                message=message,
                metadata=metadata,
            )
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if acknowledged
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.is_active:
                alert.acknowledge()
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if resolved
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.state in (AlertState.ACTIVE, AlertState.ACKNOWLEDGED):
                alert.resolve()
                
                for callback in self._on_resolve:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Resolve callback error: {e}")
                
                return True
            return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            True if dismissed
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.dismiss()
                return True
            return False
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [
                alert for alert in self._alerts.values()
                if alert.is_active
            ]
    
    def get_alerts(
        self,
        state: Optional[AlertState] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alerts with optional filtering.
        
        Args:
            state: Filter by state
            severity: Filter by severity
            limit: Maximum alerts
            
        Returns:
            List of alerts
        """
        with self._lock:
            alerts = list(self._alerts.values())
            
            if state:
                alerts = [a for a in alerts if a.state == state]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            # Sort by timestamp descending
            alerts.sort(key=lambda a: a.timestamp, reverse=True)
            
            return alerts[:limit]
    
    def get_stats(self) -> AlertStats:
        """Get alert statistics."""
        with self._lock:
            stats = AlertStats()
            stats.total_alerts = len(self._alerts)
            
            for alert in self._alerts.values():
                # Count by state
                if alert.state == AlertState.ACTIVE:
                    stats.active_alerts += 1
                elif alert.state == AlertState.ACKNOWLEDGED:
                    stats.acknowledged_alerts += 1
                elif alert.state == AlertState.RESOLVED:
                    stats.resolved_alerts += 1
                
                # Count by type
                type_key = alert.alert_type.value
                stats.alerts_by_type[type_key] = stats.alerts_by_type.get(type_key, 0) + 1
                
                # Count by severity
                sev_key = alert.severity.value
                stats.alerts_by_severity[sev_key] = stats.alerts_by_severity.get(sev_key, 0) + 1
            
            return stats
    
    def set_threshold(self, threshold: AlertThreshold) -> None:
        """Set or update a threshold.
        
        Args:
            threshold: Threshold configuration
        """
        with self._lock:
            self._thresholds[threshold.metric_name] = threshold
            self._save_config()
    
    def get_threshold(self, metric_name: str) -> Optional[AlertThreshold]:
        """Get threshold for a metric."""
        return self._thresholds.get(metric_name)
    
    def enable_threshold(self, metric_name: str, enabled: bool = True) -> bool:
        """Enable or disable a threshold.
        
        Args:
            metric_name: Metric name
            enabled: Enable state
            
        Returns:
            True if successful
        """
        with self._lock:
            threshold = self._thresholds.get(metric_name)
            if threshold:
                threshold.enabled = enabled
                self._save_config()
                return True
            return False
    
    def on_alert(self, callback: Callable[[Alert], None]) -> None:
        """Register alert callback."""
        self._on_alert.append(callback)
    
    def on_resolve(self, callback: Callable[[Alert], None]) -> None:
        """Register resolve callback."""
        self._on_resolve.append(callback)
    
    def clear_history(self, keep_active: bool = True) -> int:
        """Clear alert history.
        
        Args:
            keep_active: Keep active alerts
            
        Returns:
            Number of alerts cleared
        """
        with self._lock:
            if keep_active:
                to_remove = [
                    alert_id for alert_id, alert in self._alerts.items()
                    if not alert.is_active
                ]
            else:
                to_remove = list(self._alerts.keys())
            
            for alert_id in to_remove:
                del self._alerts[alert_id]
            
            return len(to_remove)


# ========== Global Instance ==========

_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def check_metric(metric_name: str, value: float) -> Optional[Alert]:
    """Convenience function to check a metric."""
    return get_alert_manager().check_metric(metric_name, value)


def create_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> Alert:
    """Convenience function to create an alert."""
    return get_alert_manager().create_custom_alert(title, message, severity)
