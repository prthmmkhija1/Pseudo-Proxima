"""Administrative Privilege Handler for Proxima Agent.

Phase 5: File System Operations & Administrative Access

Provides administrative privilege handling including:
- Privilege level detection (admin/root)
- Elevation mechanisms (UAC, sudo)
- Privilege checking before operations
- Audit logging for privileged operations
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.admin_privilege_handler")


class PrivilegeLevel(Enum):
    """Privilege levels."""
    STANDARD = "standard"       # Regular user
    ELEVATED = "elevated"       # Administrator/root
    SYSTEM = "system"           # System level (Windows only)
    UNKNOWN = "unknown"


class ElevationMethod(Enum):
    """Methods for privilege elevation."""
    UAC = "uac"                 # Windows UAC
    SUDO = "sudo"               # Unix sudo
    PKEXEC = "pkexec"           # Linux PolicyKit
    RUNAS = "runas"             # Windows runas
    NONE = "none"               # No elevation possible


class OperationCategory(Enum):
    """Categories of privileged operations."""
    FILE_SYSTEM = "file_system"         # System file access
    PACKAGE_INSTALL = "package_install" # Installing packages
    SERVICE_CONTROL = "service_control" # Starting/stopping services
    NETWORK = "network"                 # Network configuration
    REGISTRY = "registry"               # Windows registry (Windows only)
    PERMISSION = "permission"           # Changing file permissions
    SYSTEM_CONFIG = "system_config"     # System configuration changes


@dataclass
class PrivilegeInfo:
    """Information about current privileges."""
    level: PrivilegeLevel
    is_admin: bool
    username: str
    user_id: Optional[int]
    groups: List[str]
    elevation_available: bool
    elevation_method: ElevationMethod
    platform: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "is_admin": self.is_admin,
            "username": self.username,
            "user_id": self.user_id,
            "groups": self.groups,
            "elevation_available": self.elevation_available,
            "elevation_method": self.elevation_method.value,
            "platform": self.platform,
        }


@dataclass
class PrivilegedOperation:
    """A privileged operation request."""
    operation_id: str
    category: OperationCategory
    description: str
    command: Optional[str]
    requires_elevation: bool
    risk_level: int  # 1-5, 5 being highest
    timestamp: datetime = field(default_factory=datetime.now)
    approved: bool = False
    executed: bool = False
    result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "category": self.category.value,
            "description": self.description,
            "command": self.command,
            "requires_elevation": self.requires_elevation,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp.isoformat(),
            "approved": self.approved,
            "executed": self.executed,
            "result": self.result,
        }


@dataclass
class ElevationResult:
    """Result of an elevation attempt."""
    success: bool
    method: ElevationMethod
    error: Optional[str] = None
    output: Optional[str] = None
    return_code: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "method": self.method.value,
            "error": self.error,
            "output": self.output,
            "return_code": self.return_code,
        }


# Type alias for consent callback
ConsentCallback = Callable[[PrivilegedOperation], bool]


class AdminPrivilegeHandler:
    """Handle administrative privileges for the agent.
    
    Features:
    - Detect current privilege level
    - Request elevation when needed
    - Audit log privileged operations
    - Consent prompts for risky operations
    
    Example:
        >>> handler = AdminPrivilegeHandler()
        >>> 
        >>> # Check privileges
        >>> info = handler.get_privilege_info()
        >>> print(f"Is admin: {info.is_admin}")
        >>> 
        >>> # Execute with elevation
        >>> if handler.requires_elevation("install numpy"):
        ...     result = handler.execute_elevated("pip install numpy")
    """
    
    # Operations that typically require elevation
    PRIVILEGED_OPERATIONS = {
        # Windows
        "netsh": OperationCategory.NETWORK,
        "reg": OperationCategory.REGISTRY,
        "sc": OperationCategory.SERVICE_CONTROL,
        "bcdedit": OperationCategory.SYSTEM_CONFIG,
        "diskpart": OperationCategory.SYSTEM_CONFIG,
        # Unix
        "apt": OperationCategory.PACKAGE_INSTALL,
        "apt-get": OperationCategory.PACKAGE_INSTALL,
        "yum": OperationCategory.PACKAGE_INSTALL,
        "dnf": OperationCategory.PACKAGE_INSTALL,
        "pacman": OperationCategory.PACKAGE_INSTALL,
        "brew": OperationCategory.PACKAGE_INSTALL,
        "systemctl": OperationCategory.SERVICE_CONTROL,
        "service": OperationCategory.SERVICE_CONTROL,
        "iptables": OperationCategory.NETWORK,
        "mount": OperationCategory.FILE_SYSTEM,
        "umount": OperationCategory.FILE_SYSTEM,
        "chown": OperationCategory.PERMISSION,
        "chmod": OperationCategory.PERMISSION,
    }
    
    # Paths that require elevated access
    PRIVILEGED_PATHS_WINDOWS = [
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData",
    ]
    
    PRIVILEGED_PATHS_UNIX = [
        "/etc",
        "/usr",
        "/var",
        "/opt",
        "/root",
        "/boot",
    ]
    
    def __init__(
        self,
        audit_log_path: Optional[Path] = None,
        consent_callback: Optional[ConsentCallback] = None,
    ):
        """Initialize the privilege handler.
        
        Args:
            audit_log_path: Path for audit log file
            consent_callback: Callback for consent prompts
        """
        self.audit_log_path = audit_log_path or Path(tempfile.gettempdir()) / "proxima_audit.log"
        self.consent_callback = consent_callback
        
        self._privilege_info: Optional[PrivilegeInfo] = None
        self._operation_history: List[PrivilegedOperation] = []
    
    def get_privilege_info(self, refresh: bool = False) -> PrivilegeInfo:
        """Get current privilege information.
        
        Args:
            refresh: Force refresh of cached info
            
        Returns:
            PrivilegeInfo object
        """
        if self._privilege_info is not None and not refresh:
            return self._privilege_info
        
        system = platform.system()
        
        if system == "Windows":
            info = self._detect_windows_privileges()
        elif system in ("Linux", "Darwin"):
            info = self._detect_unix_privileges()
        else:
            info = PrivilegeInfo(
                level=PrivilegeLevel.UNKNOWN,
                is_admin=False,
                username=os.environ.get("USER", "unknown"),
                user_id=None,
                groups=[],
                elevation_available=False,
                elevation_method=ElevationMethod.NONE,
                platform=system,
            )
        
        self._privilege_info = info
        return info
    
    def _detect_windows_privileges(self) -> PrivilegeInfo:
        """Detect privileges on Windows."""
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            is_admin = False
        
        username = os.environ.get("USERNAME", "unknown")
        
        # Get groups
        groups = []
        try:
            result = subprocess.run(
                ["whoami", "/groups"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "S-1-5-32-544" in line:  # Administrators
                        groups.append("Administrators")
                    elif "S-1-5-32-545" in line:  # Users
                        groups.append("Users")
        except Exception:
            pass
        
        # Check elevation availability
        elevation_available = True  # UAC is usually available
        
        return PrivilegeInfo(
            level=PrivilegeLevel.ELEVATED if is_admin else PrivilegeLevel.STANDARD,
            is_admin=is_admin,
            username=username,
            user_id=None,
            groups=groups,
            elevation_available=elevation_available,
            elevation_method=ElevationMethod.UAC,
            platform="Windows",
        )
    
    def _detect_unix_privileges(self) -> PrivilegeInfo:
        """Detect privileges on Unix/Linux/macOS."""
        uid = os.getuid()
        euid = os.geteuid()
        is_root = euid == 0
        
        username = os.environ.get("USER", "unknown")
        
        # Get groups
        groups = []
        try:
            result = subprocess.run(
                ["groups"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                groups = result.stdout.strip().split()
        except Exception:
            pass
        
        # Check elevation availability
        elevation_method = ElevationMethod.NONE
        elevation_available = False
        
        # Check for sudo
        if subprocess.run(
            ["which", "sudo"],
            capture_output=True,
        ).returncode == 0:
            elevation_method = ElevationMethod.SUDO
            elevation_available = True
        # Check for pkexec
        elif subprocess.run(
            ["which", "pkexec"],
            capture_output=True,
        ).returncode == 0:
            elevation_method = ElevationMethod.PKEXEC
            elevation_available = True
        
        return PrivilegeInfo(
            level=PrivilegeLevel.ELEVATED if is_root else PrivilegeLevel.STANDARD,
            is_admin=is_root,
            username=username,
            user_id=uid,
            groups=groups,
            elevation_available=elevation_available,
            elevation_method=elevation_method,
            platform=platform.system(),
        )
    
    def is_admin(self) -> bool:
        """Check if running with admin/root privileges."""
        return self.get_privilege_info().is_admin
    
    def requires_elevation(self, command: str) -> bool:
        """Check if a command requires elevation.
        
        Args:
            command: Command to check
            
        Returns:
            True if elevation is likely required
        """
        if self.is_admin():
            return False
        
        # Extract base command
        parts = command.strip().split()
        if not parts:
            return False
        
        base_cmd = parts[0].lower()
        
        # Check against known privileged commands
        if base_cmd in self.PRIVILEGED_OPERATIONS:
            return True
        
        # Check for privileged paths in command
        system = platform.system()
        privileged_paths = (
            self.PRIVILEGED_PATHS_WINDOWS if system == "Windows"
            else self.PRIVILEGED_PATHS_UNIX
        )
        
        for path in privileged_paths:
            if path.lower() in command.lower():
                return True
        
        return False
    
    def get_operation_category(self, command: str) -> Optional[OperationCategory]:
        """Get the category of an operation.
        
        Args:
            command: Command to categorize
            
        Returns:
            OperationCategory or None
        """
        parts = command.strip().split()
        if not parts:
            return None
        
        base_cmd = parts[0].lower()
        return self.PRIVILEGED_OPERATIONS.get(base_cmd)
    
    def request_consent(
        self,
        operation: PrivilegedOperation,
    ) -> bool:
        """Request user consent for a privileged operation.
        
        Args:
            operation: Operation requiring consent
            
        Returns:
            True if consent granted
        """
        if self.consent_callback:
            try:
                return self.consent_callback(operation)
            except Exception as e:
                logger.warning(f"Consent callback error: {e}")
                return False
        
        # Default: deny if no callback
        logger.warning(f"No consent callback for operation: {operation.description}")
        return False
    
    def execute_elevated(
        self,
        command: str,
        working_dir: Optional[Path] = None,
        timeout: int = 300,
        require_consent: bool = True,
    ) -> ElevationResult:
        """Execute a command with elevated privileges.
        
        Args:
            command: Command to execute
            working_dir: Working directory
            timeout: Command timeout in seconds
            require_consent: Require user consent
            
        Returns:
            ElevationResult
        """
        info = self.get_privilege_info()
        
        # Create operation record
        operation = PrivilegedOperation(
            operation_id=f"op_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            category=self.get_operation_category(command) or OperationCategory.SYSTEM_CONFIG,
            description=f"Execute elevated: {command[:50]}...",
            command=command,
            requires_elevation=not info.is_admin,
            risk_level=3,
        )
        
        # Request consent if needed
        if require_consent:
            operation.approved = self.request_consent(operation)
            if not operation.approved:
                self._log_operation(operation)
                return ElevationResult(
                    success=False,
                    method=info.elevation_method,
                    error="User consent denied",
                )
        else:
            operation.approved = True
        
        # Execute based on platform
        if info.is_admin:
            result = self._execute_direct(command, working_dir, timeout)
        elif info.platform == "Windows":
            result = self._execute_elevated_windows(command, working_dir, timeout)
        else:
            result = self._execute_elevated_unix(command, working_dir, timeout, info.elevation_method)
        
        operation.executed = True
        operation.result = "success" if result.success else result.error
        self._log_operation(operation)
        self._operation_history.append(operation)
        
        return result
    
    def _execute_direct(
        self,
        command: str,
        working_dir: Optional[Path],
        timeout: int,
    ) -> ElevationResult:
        """Execute command directly (already elevated)."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(working_dir) if working_dir else None,
            )
            
            return ElevationResult(
                success=result.returncode == 0,
                method=ElevationMethod.NONE,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                return_code=result.returncode,
            )
            
        except subprocess.TimeoutExpired:
            return ElevationResult(
                success=False,
                method=ElevationMethod.NONE,
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            return ElevationResult(
                success=False,
                method=ElevationMethod.NONE,
                error=str(e),
            )
    
    def _execute_elevated_windows(
        self,
        command: str,
        working_dir: Optional[Path],
        timeout: int,
    ) -> ElevationResult:
        """Execute command with elevation on Windows using UAC."""
        try:
            # Create a batch file with the command
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".bat",
                delete=False,
            ) as f:
                if working_dir:
                    f.write(f'cd /d "{working_dir}"\n')
                f.write(f"{command}\n")
                batch_path = f.name
            
            try:
                # Use ShellExecute with runas verb
                import ctypes
                
                # This will trigger UAC prompt
                ret = ctypes.windll.shell32.ShellExecuteW(
                    None,           # hwnd
                    "runas",        # operation (elevate)
                    "cmd.exe",      # file
                    f'/c "{batch_path}"',  # parameters
                    str(working_dir) if working_dir else None,  # directory
                    1,              # show command (SW_SHOWNORMAL)
                )
                
                # ShellExecuteW returns > 32 on success
                if ret > 32:
                    return ElevationResult(
                        success=True,
                        method=ElevationMethod.UAC,
                        output="Command executed with elevation (UAC)",
                    )
                else:
                    return ElevationResult(
                        success=False,
                        method=ElevationMethod.UAC,
                        error=f"UAC elevation failed with code {ret}",
                    )
                    
            finally:
                # Clean up batch file
                try:
                    os.unlink(batch_path)
                except Exception:
                    pass
                    
        except Exception as e:
            return ElevationResult(
                success=False,
                method=ElevationMethod.UAC,
                error=str(e),
            )
    
    def _execute_elevated_unix(
        self,
        command: str,
        working_dir: Optional[Path],
        timeout: int,
        method: ElevationMethod,
    ) -> ElevationResult:
        """Execute command with elevation on Unix using sudo/pkexec."""
        try:
            if method == ElevationMethod.SUDO:
                elevated_cmd = f"sudo {command}"
            elif method == ElevationMethod.PKEXEC:
                elevated_cmd = f"pkexec {command}"
            else:
                return ElevationResult(
                    success=False,
                    method=method,
                    error="No elevation method available",
                )
            
            result = subprocess.run(
                elevated_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(working_dir) if working_dir else None,
            )
            
            return ElevationResult(
                success=result.returncode == 0,
                method=method,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                return_code=result.returncode,
            )
            
        except subprocess.TimeoutExpired:
            return ElevationResult(
                success=False,
                method=method,
                error=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            return ElevationResult(
                success=False,
                method=method,
                error=str(e),
            )
    
    def _log_operation(self, operation: PrivilegedOperation) -> None:
        """Log a privileged operation to audit log.
        
        Args:
            operation: Operation to log
        """
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            log_entry = {
                **operation.to_dict(),
                "log_time": datetime.now().isoformat(),
            }
            
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")
    
    def get_operation_history(self) -> List[PrivilegedOperation]:
        """Get history of privileged operations."""
        return list(self._operation_history)
    
    def check_path_requires_elevation(self, path: Path) -> bool:
        """Check if accessing a path requires elevation.
        
        Args:
            path: Path to check
            
        Returns:
            True if elevation likely required
        """
        if self.is_admin():
            return False
        
        path = Path(path).resolve()
        system = platform.system()
        
        privileged_paths = (
            self.PRIVILEGED_PATHS_WINDOWS if system == "Windows"
            else self.PRIVILEGED_PATHS_UNIX
        )
        
        for priv_path in privileged_paths:
            try:
                path.relative_to(priv_path)
                return True
            except ValueError:
                continue
        
        return False
    
    def get_non_elevated_alternative(self, command: str) -> Optional[str]:
        """Get a non-elevated alternative for a command if available.
        
        Args:
            command: Original command
            
        Returns:
            Alternative command or None
        """
        parts = command.strip().split()
        if not parts:
            return None
        
        base_cmd = parts[0].lower()
        
        # pip can use --user flag
        if base_cmd == "pip" or base_cmd == "pip3":
            if "install" in parts and "--user" not in parts:
                return command + " --user"
        
        # npm can use local install
        if base_cmd == "npm" and "install" in parts:
            if "-g" in parts or "--global" in parts:
                return command.replace("-g", "").replace("--global", "")
        
        return None


def get_admin_privilege_handler(
    consent_callback: Optional[ConsentCallback] = None,
) -> AdminPrivilegeHandler:
    """Get an AdminPrivilegeHandler instance."""
    return AdminPrivilegeHandler(consent_callback=consent_callback)


def is_admin() -> bool:
    """Check if running with admin privileges."""
    return AdminPrivilegeHandler().is_admin()


def requires_elevation(command: str) -> bool:
    """Check if a command requires elevation."""
    return AdminPrivilegeHandler().requires_elevation(command)
