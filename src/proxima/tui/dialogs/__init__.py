"""Proxima TUI Dialogs Package.

Contains all dialog definitions for the TUI.
"""

from .base import BaseDialog
from .commands import CommandPalette, Command, CommandItem, DEFAULT_COMMANDS
from .permissions import PermissionsDialog, PermissionRequest
from .confirmation import ConfirmationDialog
from .input import InputDialog
from .common import DialogButton, DialogHeader, DialogFooter, FilterableList, create_button_row
from .models import ModelsDialog, ModelItem
from .backends import BackendsDialog
from .sessions import SessionsDialog, SessionItem, SessionInfo
from .error import ErrorDialog
from .simulation import SimulationDialog, SimulationConfig

__all__ = [
    # Base
    "BaseDialog",
    # Common utilities
    "DialogButton",
    "DialogHeader",
    "DialogFooter",
    "FilterableList",
    "create_button_row",
    # Command Palette
    "CommandPalette",
    "Command",
    "CommandItem",
    "DEFAULT_COMMANDS",
    # Permissions
    "PermissionsDialog",
    "PermissionRequest",
    # Confirmation
    "ConfirmationDialog",
    # Input
    "InputDialog",
    # Models
    "ModelsDialog",
    "ModelItem",
    # Backends
    "BackendsDialog",
    # Sessions
    "SessionsDialog",
    "SessionItem",
    "SessionInfo",
    # Error
    "ErrorDialog",
    # Simulation
    "SimulationDialog",
    "SimulationConfig",
]
