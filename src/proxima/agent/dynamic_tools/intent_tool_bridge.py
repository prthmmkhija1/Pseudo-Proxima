"""Intent-to-Tool Bridge for the Natural-Language Pipeline.

Phases 3–8 — Maps every ``IntentType`` to the appropriate
registered tool, builds tool-specific argument dictionaries from
``Intent.entities``, and dispatches execution through the
``ToolRegistry`` / ``ToolInterface`` layer.

Phase 4 additions:
* Enhanced file-system argument builders (line ranges, context fallbacks)
* ``ScriptExecutor`` / ``InterpreterRegistry`` integration for ``RUN_SCRIPT``
* Virtual-environment auto-detection before script execution
* Terminal output capture → ``SessionContext.last_operation_result``
* Safety consent flow for destructive file/directory operations
  (``ConsentRequest`` from ``safety.py``)

Phase 5 additions:
* ``ProjectDependencyManager`` integration for ``INSTALL_DEPENDENCY``,
  ``CHECK_DEPENDENCY``, and ``CONFIGURE_ENVIRONMENT`` intents.
* Smart project dependency detection (requirements.txt, setup.py,
  pyproject.toml, package.json, Pipfile, environment.yml, setup.cfg).
* Automatic error-detection-and-fix loop after failed ``RUN_COMMAND`` /
  ``RUN_SCRIPT`` / ``INSTALL_DEPENDENCY``.
* Backend dependency pre-check via ``BuildProfileLoader``.
* ``CONFIGURE_ENVIRONMENT`` supports create/activate venv and set env var.

Phase 7 additions:
* ``AdminPrivilegeHandler`` integration for ``ADMIN_ELEVATE`` intent.
* Auto-detection of commands requiring elevation (package installs,
  service control, protected directory writes, network config,
  registry access, permission changes, CUDA/GPU operations).
* Safe escalation with explicit user consent (``ConsentType.ADMIN_ACCESS``).
* Category-specific handlers: PACKAGE_INSTALL (venv suggestion),
  SERVICE_CONTROL (before/after status), PERMISSION (current/new display),
  CUDA/GPU (post-install nvidia-smi/nvcc verification).
* Hard security blocks (boot records, firmware, remote-pipe execution).
* Per-session escalation limit (max 3).
* Append-only audit logging to ``~/.proxima/logs/admin_audit.log``.

Phase 8 additions:
* Enhanced ``GIT_CLONE`` handling: robust URL normalisation (HTTPS,
  SSH, short-form ``owner/repo``), optional ``--branch``, default
  clone path under ``~/.proxima/backends/<repo>``.
* Enhanced ``GIT_PULL`` / ``GIT_PUSH``: git-repo verification,
  auto ``--set-upstream`` on first push, consent for push.
* Enhanced ``GIT_CHECKOUT`` / ``GIT_BRANCH`` / ``GIT_MERGE``:
  branch-name validation, sub-operation routing, conflict reporting.
* Enhanced ``GIT_COMMIT`` / ``GIT_ADD``: LLM-generated commit
  messages from staged diff, ``commit all`` support, pattern-based
  ``git add``.
* All ``GIT_*`` intents registered in ``INTENT_TO_TOOL`` and wired
  to Phase 8 dispatch (``_dispatch_phase8``).

Architecture Notes
~~~~~~~~~~~~~~~~~~
* **No hardcoded model references.** Any integrated model (local or
  remote) works via the existing ``LLMRouter`` / ``LLMRequest`` API.
* **Two dispatch paths:**
  1. *Direct dispatch* — single intent → single tool execution.
  2. *Plan dispatch* — ``MULTI_STEP`` / ``PLAN_EXECUTION`` intents
     are translated into ``ExecutionPlan`` objects and handed to the
     ``PlanExecutor``.
* **Safety layer** — mirrors ``SafetyManager`` constants so the bridge
  can pre-screen commands before delegation.
"""

from __future__ import annotations

import getpass
import logging
import difflib
import hashlib
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from proxima.agent.dynamic_tools.robust_nl_processor import (
    ExtractedEntity,
    Intent,
    IntentType,
    SessionContext,
)
from proxima.agent.dynamic_tools.tool_interface import ToolResult, ExecutionStatus

# Phase 4 — optional imports for script execution and safety consent
try:
    from proxima.agent.script_executor import (
        ScriptExecutor,
        ScriptLanguage,
        InterpreterRegistry,
    )
    _SCRIPT_EXECUTOR_AVAILABLE = True
except ImportError:
    _SCRIPT_EXECUTOR_AVAILABLE = False

try:
    from proxima.agent.safety import ConsentType, ConsentRequest
    _SAFETY_AVAILABLE = True
except ImportError:
    _SAFETY_AVAILABLE = False

# Phase 5 — Project-level dependency management
try:
    from proxima.agent.dependency_manager import ProjectDependencyManager
    _DEP_MANAGER_AVAILABLE = True
except ImportError:
    _DEP_MANAGER_AVAILABLE = False

# Phase 5 — Backend build profile loader
try:
    from proxima.agent.backend_builder import BuildProfileLoader
    _BUILD_PROFILE_AVAILABLE = True
except ImportError:
    _BUILD_PROFILE_AVAILABLE = False

# Phase 6 — Backend modification, checkpoint management, and preview
try:
    from proxima.agent.backend_modifier import (
        BackendModifier,
        ModificationType,
        CodeChange,
    )
    _BACKEND_MODIFIER_AVAILABLE = True
except ImportError:
    _BACKEND_MODIFIER_AVAILABLE = False

try:
    from proxima.agent.checkpoint_manager import CheckpointManager
    _CHECKPOINT_MANAGER_AVAILABLE = True
except ImportError:
    _CHECKPOINT_MANAGER_AVAILABLE = False

try:
    from proxima.agent.modification_preview import (
        ModificationPreviewGenerator,
    )
    _MOD_PREVIEW_AVAILABLE = True
except ImportError:
    _MOD_PREVIEW_AVAILABLE = False

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Phase 7 — Administrative privilege handling
try:
    from proxima.agent.admin_privilege_handler import (
        AdminPrivilegeHandler,
        ElevationMethod,
        ElevationResult,
        OperationCategory,
        PrivilegeLevel,
        PrivilegedOperation,
    )
    _ADMIN_HANDLER_AVAILABLE = True
except ImportError:
    _ADMIN_HANDLER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Module-level constants — extracted from hot paths for performance
# ═══════════════════════════════════════════════════════════════════════

# Extension → interpreter command (fallback when ScriptExecutor unavailable)
_SCRIPT_RUNNER_MAP: Dict[str, str] = {
    ".py":  "python",
    ".js":  "node",
    ".ts":  "npx ts-node",
    ".sh":  "bash",
    ".ps1": "powershell -NoProfile -File",
    ".rb":  "ruby",
    ".lua": "lua",
}

# Risk severity ordering for get_command_risk_level()
_RISK_ORDER: Dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# Destructive intent → (ConsentType name, risk_level, description, reversible)
_DESTRUCTIVE_INTENTS: Dict = {}  # populated after IntentType import below


# ═══════════════════════════════════════════════════════════════════════
# Static mapping: IntentType → registered tool name(s)
# ═══════════════════════════════════════════════════════════════════════
# The values are either a single tool name ``str`` or a list of tool
# names when the intent may span multiple tools (first match wins).
# ``None`` means the intent is handled in-processor or has no direct
# tool equivalent yet.

INTENT_TO_TOOL: Dict[IntentType, Optional[str]] = {
    # ── Navigation ────────────────────────────────────────────────
    IntentType.NAVIGATE_DIRECTORY:      "change_directory",
    IntentType.LIST_DIRECTORY:          "list_directory",
    IntentType.SHOW_CURRENT_DIR:        "get_working_directory",
    # ── File Operations ───────────────────────────────────────────
    IntentType.CREATE_FILE:             "write_file",
    IntentType.READ_FILE:               "read_file",
    IntentType.WRITE_FILE:              "write_file",
    IntentType.DELETE_FILE:             "delete_file",
    IntentType.COPY_FILE:              "run_command",  # no dedicated copy tool; uses cp/copy
    IntentType.MOVE_FILE:              "move_file",
    IntentType.SEARCH_FILE:            "search_files",
    IntentType.CREATE_DIRECTORY:       "run_command",  # mkdir via shell
    IntentType.DELETE_DIRECTORY:       "run_command",  # rmdir via shell
    IntentType.COPY_DIRECTORY:         "run_command",  # xcopy/cp -r via shell
    # ── Git Basic ─────────────────────────────────────────────────
    IntentType.GIT_CHECKOUT:           "git_branch",
    IntentType.GIT_CLONE:              "run_command",
    IntentType.GIT_PULL:               "run_command",
    IntentType.GIT_PUSH:               "run_command",
    IntentType.GIT_STATUS:             "git_status",
    IntentType.GIT_COMMIT:             "git_commit",
    IntentType.GIT_ADD:                "git_add",
    IntentType.GIT_BRANCH:            "git_branch",
    IntentType.GIT_FETCH:             "run_command",
    # ── Git Extended ──────────────────────────────────────────────
    IntentType.GIT_MERGE:             "run_command",
    IntentType.GIT_REBASE:            "run_command",
    IntentType.GIT_STASH:             "run_command",
    IntentType.GIT_LOG:               "git_log",
    IntentType.GIT_DIFF:              "git_diff",
    IntentType.GIT_CONFLICT_RESOLVE:  "run_command",
    # ── Terminal & Script ─────────────────────────────────────────
    IntentType.RUN_COMMAND:            "run_command",
    IntentType.RUN_SCRIPT:             "run_command",
    IntentType.TERMINAL_MONITOR:       None,          # handled by TUI layer
    IntentType.TERMINAL_KILL:          None,
    IntentType.TERMINAL_OUTPUT:        None,
    IntentType.TERMINAL_LIST:          None,
    # ── Queries ───────────────────────────────────────────────────
    IntentType.QUERY_LOCATION:        "search_files",  # TODO: custom handler (Phase 3 spec)
    IntentType.QUERY_STATUS:          None,             # custom handler — checks SessionContext
    # ── Dependencies & Environment ────────────────────────────────
    IntentType.INSTALL_DEPENDENCY:     "run_command",
    IntentType.CONFIGURE_ENVIRONMENT:  "run_command",
    IntentType.CHECK_DEPENDENCY:       "run_command",  # Phase 5: custom handler via ProjectDependencyManager
    # ── Analysis & Export ─────────────────────────────────────────
    IntentType.ANALYZE_RESULTS:        None,          # in-process
    IntentType.EXPORT_RESULTS:         "write_file",
    # ── Planning ──────────────────────────────────────────────────
    IntentType.PLAN_EXECUTION:         None,          # creates ExecutionPlan
    IntentType.MULTI_STEP:             None,          # plan dispatch
    # ── Undo / Redo ───────────────────────────────────────────────
    IntentType.UNDO_OPERATION:         None,          # Phase 6: custom handler → CheckpointManager
    IntentType.REDO_OPERATION:         None,          # Phase 6: custom handler → CheckpointManager
    # ── Backend Operations ────────────────────────────────────────
    IntentType.BACKEND_BUILD:          None,          # Phase 6: custom handler → build pipeline
    IntentType.BACKEND_CONFIGURE:      None,          # Phase 6: custom handler → configs/default.yaml
    IntentType.BACKEND_TEST:           None,          # Phase 6: custom handler → YAML verification
    IntentType.BACKEND_MODIFY:         None,          # Phase 6: custom handler → BackendModifier + consent
    IntentType.BACKEND_LIST:           None,          # Phase 6: custom handler → read YAML profiles
    # ── System ────────────────────────────────────────────────────
    IntentType.SYSTEM_INFO:            "run_command",
    IntentType.ADMIN_ELEVATE:          None,          # requires TUI consent
    IntentType.WEB_SEARCH:             None,          # in-process
    # ── Unknown ───────────────────────────────────────────────────
    IntentType.UNKNOWN:                None,
}

# Now that IntentType is available, populate the destructive-intents map.
_DESTRUCTIVE_INTENTS.update({
    IntentType.DELETE_FILE:      ("FILE_MODIFICATION", "medium",   "delete a file",                     False),
    IntentType.DELETE_DIRECTORY:  ("FILE_MODIFICATION", "high",     "delete a directory and its contents", False),
    IntentType.MOVE_FILE:        ("FILE_MODIFICATION", "low",      "move/rename a file",                 True),
})


# ═══════════════════════════════════════════════════════════════════════
# Safety constants
# ═══════════════════════════════════════════════════════════════════════

SAFE_COMMANDS: Set[str] = {
    "git status", "git log", "git diff", "git branch",
    "ls", "dir", "pwd", "cd", "cat", "type", "echo",
    "pip list", "pip show", "npm list", "conda list",
    "python --version", "node --version", "git --version",
    "whoami", "hostname", "date",
    # Phase 4 additions
    "head", "tail", "which", "where", "wc", "file", "stat",
    "get-childitem", "get-content", "get-location", "get-item",
    "test-path", "select-string",
}

DANGEROUS_PATTERNS: List[str] = [
    r"rm\s+-rf\s+/",
    r"del\s+/s\s+/q",
    r"format\s+[a-zA-Z]:",
    r"mkfs\.",
    r"dd\s+if=",
    r"chmod\s+777",
    r":(){ :|:& };:",     # fork bomb
    r">\s*/dev/sd",
    r"shutdown",
    r"reboot",
    r"init\s+0",
    # Phase 4 additions — lower-risk destructive patterns
    r"\brm\b",
    r"\bdel\b",
    r"\brmdir\b",
    r"\bsudo\b",
    r"\brunas\b",
    r"chmod\s+[0-7]{3,4}",
]

BLOCKED_PATTERNS: List[str] = [
    r"rm\s+-rf\s+/$",
    r"rm\s+-rf\s+/\*",
    r"del\s+/s\s+/q\s+C:\\$",
    r"format\s+C:",
    r":(){ :|:& };:",
]

# Phase 4 — risk level for each dangerous pattern (keyed by regex string)
_DANGEROUS_RISK_MAP: Dict[str, str] = {
    r"rm\s+-rf\s+/":       "critical",
    r"del\s+/s\s+/q":      "critical",
    r"format\s+[a-zA-Z]:": "critical",
    r"mkfs\.":             "critical",
    r"dd\s+if=":           "critical",
    r":(){ :|:& };:":      "critical",
    r">\s*/dev/sd":         "critical",
    r"shutdown":            "high",
    r"reboot":              "high",
    r"init\s+0":            "high",
    r"chmod\s+777":         "medium",
    r"\brm\b":             "high",
    r"\bdel\b":            "high",
    r"\brmdir\b":          "high",
    r"\bsudo\b":           "high",
    r"\brunas\b":          "high",
    r"chmod\s+[0-7]{3,4}": "medium",
}

# Pre-compile for performance
_DANGEROUS_RE = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]
_BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════
# Phase 6 — Backend name normalisation and intent routing
# ═══════════════════════════════════════════════════════════════════════

_BACKEND_NAME_MAP: Dict[str, str] = {
    "lret cirq": "lret_cirq_scalability",
    "lret cirq scalability": "lret_cirq_scalability",
    "cirq scalability": "lret_cirq_scalability",
    "lret pennylane": "lret_pennylane_hybrid",
    "lret pennylane hybrid": "lret_pennylane_hybrid",
    "pennylane hybrid": "lret_pennylane_hybrid",
    "lret phase 7": "lret_phase_7_unified",
    "lret phase7": "lret_phase_7_unified",
    "lret unified": "lret_phase_7_unified",
    "phase 7 unified": "lret_phase_7_unified",
    "cirq": "cirq",
    "google cirq": "cirq",
    "qiskit": "qiskit",
    "qiskit aer": "qiskit",
    "ibm qiskit": "qiskit",
    "quest": "quest",
    "qsim": "qsim",
    "google qsim": "qsim",
    "qsim cuda": "qsim_cuda",
    "qsim gpu": "qsim_cuda",
    "cuquantum": "cuquantum",
    "cuquantum sim": "cuquantum",
    "nvidia cuquantum": "cuquantum",
}

_PHASE6_INTENTS = frozenset({
    IntentType.BACKEND_BUILD,
    IntentType.BACKEND_MODIFY,
    IntentType.BACKEND_TEST,
    IntentType.BACKEND_CONFIGURE,
    IntentType.BACKEND_LIST,
    IntentType.UNDO_OPERATION,
    IntentType.REDO_OPERATION,
})

# ═══════════════════════════════════════════════════════════════════════
# Phase 7 — Admin detection patterns and constants
# ═══════════════════════════════════════════════════════════════════════

_PHASE7_INTENTS = frozenset({
    IntentType.ADMIN_ELEVATE,
})

# Patterns that indicate a command requires administrative privileges.
# Each entry maps a compiled regex to an (OperationCategory_name, reason) tuple.
_ADMIN_DETECTION_PATTERNS: List[Tuple[re.Pattern, str, str]] = []


def _build_admin_patterns() -> List[Tuple["re.Pattern[str]", str, str]]:
    """Build the admin-detection regex list once at import time."""
    raw: List[Tuple[str, str, str]] = [
        # Package installation to system-level paths
        (r"\bpip\s+install\b(?!.*--user)", "PACKAGE_INSTALL",
         "pip install without --user flag may require admin for system Python"),
        (r"\bapt(?:-get)?\s+install\b", "PACKAGE_INSTALL",
         "apt package installation requires admin"),
        (r"\byum\s+install\b", "PACKAGE_INSTALL",
         "yum package installation requires admin"),
        (r"\bdnf\s+install\b", "PACKAGE_INSTALL",
         "dnf package installation requires admin"),
        (r"\bpacman\s+-S\b", "PACKAGE_INSTALL",
         "pacman package installation requires admin"),
        (r"\bbrew\s+install\b", "PACKAGE_INSTALL",
         "brew installation may require admin"),
        (r"\bchoco\s+install\b", "PACKAGE_INSTALL",
         "chocolatey package installation requires admin"),

        # System service control
        (r"\bsc\s+(start|stop|config|create|delete)\b", "SERVICE_CONTROL",
         "Windows service control requires admin"),
        (r"\bsystemctl\s+(start|stop|restart|enable|disable)\b",
         "SERVICE_CONTROL", "systemd service control requires admin"),
        (r"\bservice\s+\S+\s+(start|stop|restart)\b", "SERVICE_CONTROL",
         "Service control requires admin"),
        (r"\bnet\s+(start|stop)\b", "SERVICE_CONTROL",
         "Windows net start/stop requires admin"),

        # Writing to protected directories
        (r"(?:>|>>|copy|xcopy|move|mv|cp|write)\s+.*(?:C:\\Windows|C:\\Program\s*Files)",
         "FILE_SYSTEM",
         "Writing to Windows system directory requires admin"),
        (r"(?:>|>>|cp|mv|install|write)\s+.*/(?:usr|etc|opt)/",
         "FILE_SYSTEM",
         "Writing to Unix system directory requires admin"),

        # Network configuration
        (r"\bnetsh\b", "NETWORK", "Windows network configuration requires admin"),
        (r"\biptables\b", "NETWORK", "Firewall configuration requires admin"),
        (r"\bufw\b", "NETWORK", "UFW firewall configuration requires admin"),
        (r"\bfirewall-cmd\b", "NETWORK",
         "Firewall configuration requires admin"),

        # Registry access (Windows)
        (r"\breg\s+(add|delete)\b", "REGISTRY",
         "Windows registry modification requires admin"),
        (r"\bregedit\b", "REGISTRY",
         "Registry editor requires admin"),

        # Permission changes
        (r"\bchmod\b", "PERMISSION", "Changing file permissions may require admin"),
        (r"\bchown\b", "PERMISSION",
         "Changing file ownership requires admin"),
        (r"\bicacls\b", "PERMISSION",
         "Changing Windows ACLs requires admin"),
        (r"\battrib\b", "PERMISSION",
         "Changing file attributes may require admin"),
        (r"\btakeown\b", "PERMISSION",
         "Taking file ownership requires admin"),

        # CUDA / GPU operations
        (r"\bcuda.*(install|setup)\b", "SYSTEM_CONFIG",
         "CUDA toolkit installation requires admin"),
        (r"\bnvidia-smi\s+(-pm|-pl|-ac|-rgc)", "SYSTEM_CONFIG",
         "nvidia-smi GPU configuration requires admin"),
        (r"gpu\s*driver.*(install|update)", "SYSTEM_CONFIG",
         "GPU driver installation requires admin"),
    ]
    return [
        (re.compile(pat, re.IGNORECASE), cat, reason)
        for pat, cat, reason in raw
    ]


_ADMIN_DETECTION_PATTERNS = _build_admin_patterns()

# Protected system directories
_PROTECTED_DIRS_WINDOWS = (
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "C:\\ProgramData",
)
_PROTECTED_DIRS_UNIX = (
    "/usr/", "/etc/", "/opt/", "/var/", "/root/", "/boot/", "/sbin/",
)

# Hard security restrictions — patterns that MUST be blocked at critical
# consent level even if user explicitly requests them.
_SECURITY_BLOCK_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"\b(?:mbr|boot\s*record|partition\s*table|firmware)\b.*"
                r"(?:modify|write|delete|format)", re.IGNORECASE),
     "Modification of boot records, partition tables, or firmware is prohibited"),
    (re.compile(r"(?:curl|wget|Invoke-WebRequest).*\|\s*(?:bash|sh|powershell)",
                re.IGNORECASE),
     "Executing remote scripts via pipe is blocked for security"),
]

# Security-sensitive operations that require critical consent + warning
_SECURITY_SENSITIVE_RE: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"\b(?:disable|stop|turn\s*off)\b.*"
                r"(?:Windows\s*Defender|SELinux|firewall|antivirus)",
                re.IGNORECASE),
     "Disabling system security features"),
]

# Maximum consecutive admin escalations per session (Step 7.4)
_MAX_ADMIN_ESCALATIONS_PER_SESSION = 3


# ═══════════════════════════════════════════════════════════════════════
# Phase 8 — GitHub / Git operation constants
# ═══════════════════════════════════════════════════════════════════════

_PHASE8_INTENTS = frozenset({
    IntentType.GIT_CLONE,
    IntentType.GIT_PULL,
    IntentType.GIT_PUSH,
    IntentType.GIT_CHECKOUT,
    IntentType.GIT_BRANCH,
    IntentType.GIT_MERGE,
    IntentType.GIT_COMMIT,
    IntentType.GIT_ADD,
})

# URL normalisation regex for GitHub clone operations (Step 8.1)
_GIT_URL_HTTPS_RE = re.compile(
    r"https?://(?:www\.)?github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?$",
    re.IGNORECASE,
)
_GIT_URL_SSH_RE = re.compile(
    r"git@github\.com:[\w\-\.]+/[\w\-\.]+(?:\.git)?$",
    re.IGNORECASE,
)
_GIT_URL_SHORT_DOMAIN_RE = re.compile(
    r"^github\.com/[\w\-\.]+/[\w\-\.]+$",
    re.IGNORECASE,
)
_GIT_URL_OWNER_REPO_RE = re.compile(
    r"^[\w\-\.]+/[\w\-\.]+$",
)

# Words that must NOT be treated as branch names (Step 8.3)
_BRANCH_NAME_BLACKLIST = frozenset({
    "the", "and", "for", "with", "this", "that", "from",
    "clone", "build", "configure", "into", "after", "before",
    "next", "then", "all", "everything", "nothing", "none",
    "repo", "repository", "branch", "commit", "push", "pull",
    "merge", "rebase", "stash", "file", "directory",
})


def _normalise_git_url(raw: str) -> str:
    """Normalise a Git URL to a full clone-able form.

    Handles:
    - Full HTTPS: ``https://github.com/user/repo`` (passthrough)
    - Full HTTPS .git: ``https://github.com/user/repo.git``
    - SSH: ``git@github.com:user/repo.git``
    - Short domain: ``github.com/user/repo``
    - Owner/repo: ``user/repo`` → ``https://github.com/user/repo``
    """
    url = raw.strip().rstrip("/")
    if _GIT_URL_HTTPS_RE.match(url) or _GIT_URL_SSH_RE.match(url):
        return url
    if url.startswith("https://") or url.startswith("http://"):
        return url  # non-GitHub HTTPS URL
    if url.startswith("git@"):
        return url  # non-GitHub SSH URL
    if _GIT_URL_SHORT_DOMAIN_RE.match(url):
        return f"https://{url}"
    if _GIT_URL_OWNER_REPO_RE.match(url):
        return f"https://github.com/{url}"
    return url  # unknown format — use as-is


def _repo_name_from_url(url: str) -> str:
    """Extract the repository name from a git URL."""
    name = url.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    # Handle SSH format  git@github.com:user/repo.git
    if ":" in name:
        name = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
    return name or "repo"


def _validate_branch_name(name: str) -> bool:
    """Validate a branch name (Step 8.3).

    Must be >= 2 chars, start with a letter, not be a common
    English word, and not contain shell-unsafe characters.
    """
    if not name or len(name) < 2:
        return False
    if not name[0].isalpha():
        return False
    if name.lower() in _BRANCH_NAME_BLACKLIST:
        return False
    # Reject shell metacharacters to prevent injection
    if re.search(r'[;|&$`!(){}\[\]<>\\\s\x00-\x1f~^:?*\[\]]', name):
        return False
    return True


def _sanitize_git_ref(name: str) -> str:
    """Return *name* with any shell-unsafe characters removed.

    This is a defence-in-depth measure on top of ``_validate_branch_name``.
    Git ref names that pass validation should already be safe, but this
    provides an extra layer before string interpolation into shell commands.
    """
    # Keep only alphanumeric, hyphens, underscores, dots, slashes
    return re.sub(r'[^\w./-]', '', name)


def _is_git_repo(directory: str) -> bool:
    """Return ``True`` if *directory* is inside a git repository."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0 and "true" in (proc.stdout or "").lower()
    except Exception:
        return False


def _git_current_branch(directory: str) -> Optional[str]:
    """Return the current branch name, or ``None``."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.strip()
    except Exception:
        pass
    return None


def _resolve_git_work_dir(
    intent: Intent,
    cwd: str,
    session_context: Optional[SessionContext] = None,
) -> str:
    """Resolve the working directory for a git operation.

    Priority: explicit path entity > SessionContext.current_directory > *cwd*.
    """
    work_dir = cwd
    if session_context and session_context.current_directory:
        work_dir = session_context.current_directory
    path_entity = _entity_value(intent, "path", "dirname", "directory")
    if path_entity:
        work_dir = resolve_path(path_entity, cwd)
    return work_dir


def _run_git_subprocess(
    command: str,
    cwd: str,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run a git command in a subprocess.

    Thin wrapper that centralises ``shell=True, capture_output=True,
    text=True`` so callers don't repeat them.
    """
    return subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ═══════════════════════════════════════════════════════════════════════
# Dependency file detection
# ═══════════════════════════════════════════════════════════════════════

# Map of dependency file → install command template
_DEPENDENCY_FILES: Dict[str, str] = {
    "requirements.txt":  "pip install -r requirements.txt",
    "setup.py":          "pip install -e .",
    "pyproject.toml":    "pip install -e .",
    "Pipfile":           "pipenv install",
    "package.json":      "npm install",
    "yarn.lock":         "yarn install",
    "Gemfile":           "bundle install",
    "Cargo.toml":        "cargo build",
    "go.mod":            "go mod download",
    "CMakeLists.txt":    "cmake -B build && cmake --build build",
    "Makefile":          "make",
    "pom.xml":           "mvn install",
    "build.gradle":      "gradle build",
}


def _detect_dependency_file(repo_path: str) -> Optional[Tuple[str, str]]:
    """Detect the dependency file in a repository.

    Returns ``(filename, install_command)`` or ``None``.
    """
    if not repo_path or not os.path.isdir(repo_path):
        return None
    for fname, cmd in _DEPENDENCY_FILES.items():
        if os.path.isfile(os.path.join(repo_path, fname)):
            return fname, cmd
    return None


# ═══════════════════════════════════════════════════════════════════════
# Path resolution helper
# ═══════════════════════════════════════════════════════════════════════

def resolve_path(raw_path: Optional[str], cwd: str) -> str:
    """Resolve a possibly-relative path against *cwd*.

    Handles ``~``, environment variables, and ``..`` components.
    Returns an absolute path string.
    """
    if not raw_path:
        return cwd
    expanded = os.path.expanduser(os.path.expandvars(raw_path))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(cwd, expanded))


# ═══════════════════════════════════════════════════════════════════════
# Argument builders — one per intent family
# ═══════════════════════════════════════════════════════════════════════

def _entity_value(intent: Intent, *names: str) -> Optional[str]:
    """Return the first non-``None`` entity value matching *names*."""
    for n in names:
        v = intent.get_entity(n)
        if v:
            return v
    return None


def _build_args_navigate(intent: Intent, cwd: str) -> Dict[str, Any]:
    path = _entity_value(intent, "path", "dirname", "directory")
    return {"directory": resolve_path(path, cwd)}


def _build_args_list_directory(intent: Intent, cwd: str) -> Dict[str, Any]:
    path = _entity_value(intent, "path", "dirname", "directory")
    return {"path": resolve_path(path, cwd)}


def _build_args_get_cwd(intent: Intent, cwd: str) -> Dict[str, Any]:
    return {}


def _build_args_read_file(intent: Intent, cwd: str) -> Dict[str, Any]:
    path = _entity_value(intent, "path", "filename", "file")
    args: Dict[str, Any] = {"path": resolve_path(path, cwd)}
    # Phase 4: line range support
    start = _entity_value(intent, "start_line", "line_start", "from_line")
    end = _entity_value(intent, "end_line", "line_end", "to_line")
    line_range = _entity_value(intent, "line_range", "lines")
    if line_range and not start:
        # Parse "10-20" or "10:20" format
        match = re.match(r'(\d+)\s*[-:]\s*(\d+)', str(line_range))
        if match:
            start, end = match.group(1), match.group(2)
    if start and str(start).isdigit():
        args["start_line"] = int(start)
    if end and str(end).isdigit():
        args["end_line"] = int(end)
    return args


def _build_args_write_file(intent: Intent, cwd: str) -> Dict[str, Any]:
    path = _entity_value(intent, "path", "filename", "file")
    content = _entity_value(intent, "content", "text", "data") or ""
    return {"path": resolve_path(path, cwd), "content": content}


def _build_args_delete_file(intent: Intent, cwd: str) -> Dict[str, Any]:
    path = _entity_value(intent, "path", "filename", "file")
    return {"path": resolve_path(path, cwd)}


def _build_args_move_file(intent: Intent, cwd: str) -> Dict[str, Any]:
    src = _entity_value(intent, "source", "path", "filename", "file")
    dst = _entity_value(intent, "destination", "target")
    return {
        "source": resolve_path(src, cwd),
        "destination": resolve_path(dst, cwd),
    }


def _build_args_search_files(intent: Intent, cwd: str) -> Dict[str, Any]:
    pattern = _entity_value(intent, "pattern", "query", "search_term", "filename")
    content = _entity_value(intent, "content_pattern", "content", "text")
    path = _entity_value(intent, "path", "directory")
    args: Dict[str, Any] = {"path": resolve_path(path, cwd)}
    # SearchFilesTool expects 'name_pattern' (glob) and/or 'content_pattern' (text/regex)
    if content:
        args["content_pattern"] = content
    if pattern:
        args["name_pattern"] = pattern
    # If neither was extracted, default name_pattern to wildcard
    if "name_pattern" not in args and "content_pattern" not in args:
        args["name_pattern"] = "*"
    return args


def _build_args_file_info(intent: Intent, cwd: str) -> Dict[str, Any]:
    path = _entity_value(intent, "path", "filename", "file")
    return {"path": resolve_path(path, cwd)}


# ── Configure environment builder (Phase 5, Step 5.2) ────────────

def _build_args_configure_environment(
    intent: Intent, cwd: str
) -> Dict[str, Any]:
    """Build arguments for ``CONFIGURE_ENVIRONMENT`` intents.

    Supports three sub-operations:
    1. **Create venv** — ``python -m venv <name>``
    2. **Activate venv** — OS-appropriate activation command
    3. **Set env var** — ``$env:VAR=value`` (Win) / ``export VAR=value`` (Unix)
    """
    raw = (intent.raw_message or "").lower()
    env_name = _entity_value(intent, "env_name", "venv_name", "name") or ".venv"
    var_name = _entity_value(intent, "variable", "var_name", "key")
    var_value = _entity_value(intent, "value", "var_value")

    # Sub-operation: set environment variable
    if var_name and var_value:
        if os.name == "nt":
            cmd = f'$env:{var_name} = "{var_value}"'
        else:
            cmd = f'export {var_name}="{var_value}"'
        return {
            "command": cmd,
            "working_directory": cwd,
            "_env_op": "set_var",
            "_var_name": var_name,
            "_var_value": var_value,
        }

    # Sub-operation: activate existing venv
    if "activate" in raw:
        # Detect existing venvs
        for vname in (env_name, ".venv", "venv", "env"):
            vpath = os.path.join(cwd, vname)
            if os.path.isdir(vpath):
                if os.name == "nt":
                    activate = os.path.join(vpath, "Scripts", "Activate.ps1")
                    cmd = f'& "{activate}"'
                else:
                    activate = os.path.join(vpath, "bin", "activate")
                    cmd = f'source "{activate}"'
                return {
                    "command": cmd,
                    "working_directory": cwd,
                    "_env_op": "activate",
                    "_env_name": vname,
                    "_env_path": vpath,
                }
        # No venv found — create then activate
        if os.name == "nt":
            cmd = f'python -m venv {env_name}; & "{os.path.join(cwd, env_name, "Scripts", "Activate.ps1")}"'
        else:
            cmd = f'python -m venv {env_name} && source "{os.path.join(cwd, env_name, "bin", "activate")}"'
        return {
            "command": cmd,
            "working_directory": cwd,
            "_env_op": "create_activate",
            "_env_name": env_name,
            "_env_path": os.path.join(cwd, env_name),
        }

    # Default sub-operation: create venv
    return {
        "command": f"python -m venv {env_name}",
        "working_directory": cwd,
        "_env_op": "create",
        "_env_name": env_name,
        "_env_path": os.path.join(cwd, env_name),
    }


# ── Git argument builders ─────────────────────────────────────────

def _build_args_git_status(intent: Intent, cwd: str) -> Dict[str, Any]:
    return {}


def _build_args_git_commit(intent: Intent, cwd: str) -> Dict[str, Any]:
    msg = _entity_value(intent, "message", "commit_message") or "Auto commit"
    return {"message": msg}


def _build_args_git_add(intent: Intent, cwd: str) -> Dict[str, Any]:
    files = _entity_value(intent, "files", "path", "filename")
    return {"files": files or "."}


def _build_args_git_log(intent: Intent, cwd: str) -> Dict[str, Any]:
    n = _entity_value(intent, "count", "number", "limit")
    return {"limit": int(n) if n and n.isdigit() else 10}


def _build_args_git_diff(intent: Intent, cwd: str) -> Dict[str, Any]:
    path = _entity_value(intent, "path", "file")
    return {"path": path} if path else {}


def _build_args_git_branch(intent: Intent, cwd: str) -> Dict[str, Any]:
    branch = _entity_value(intent, "branch", "branch_name")
    # GIT_CHECKOUT needs action="switch"; GIT_BRANCH defaults to "list"
    if intent.intent_type == IntentType.GIT_CHECKOUT:
        args: Dict[str, Any] = {"action": "switch"}
        if branch:
            args["name"] = branch
        return args
    # For GIT_BRANCH: list/create/delete
    if branch:
        return {"name": branch, "action": "list"}
    return {"action": "list"}


# ── Run command / git shell builders ──────────────────────────────

def _build_args_run_command(intent: Intent, cwd: str) -> Dict[str, Any]:
    """Build arguments for ``run_command`` tool.

    This handles many intent types that use the shell: git clone,
    git pull, mkdir, scripts, dependency installation, etc.

    All branches return ``"working_directory"`` (matching
    ``RunCommandTool``'s parameter name) instead of ``"cwd"``.
    """
    # Prefer an explicit 'command' entity
    cmd = _entity_value(intent, "command")
    if cmd:
        return {"command": cmd, "working_directory": cwd}

    it = intent.intent_type

    # ── Git clone (Phase 8, Step 8.1) ───────────────────────────
    if it == IntentType.GIT_CLONE:
        url_raw = _entity_value(intent, "url", "repo_url", "repository")
        dest = _entity_value(intent, "destination", "path", "dirname")
        branch = _entity_value(intent, "branch", "branch_name")

        if url_raw:
            url = _normalise_git_url(url_raw)
        else:
            url = ""

        # Default target directory if none provided
        if not dest and url:
            repo_name = _repo_name_from_url(url)
            dest = os.path.join(
                os.path.expanduser("~"), ".proxima", "backends", repo_name,
            )

        parts = ["git", "clone"]
        if branch and _validate_branch_name(branch):
            parts.extend(["--branch", branch])
        if url:
            parts.append(url)
        if dest:
            parts.append(resolve_path(dest, cwd))
        return {"command": " ".join(parts), "working_directory": cwd}

    # ── Git pull (Phase 8, Step 8.2) ────────────────────────────
    # NOTE: Normally intercepted by _dispatch_phase8() before reaching here.
    # Kept as fallback if build_tool_arguments() is called directly.
    if it == IntentType.GIT_PULL:
        remote = _entity_value(intent, "remote") or ""
        branch = _entity_value(intent, "branch", "branch_name") or ""
        parts = ["git", "pull"]
        if remote:
            parts.append(remote)
            if branch:
                parts.append(branch)
        return {"command": " ".join(parts), "working_directory": cwd}

    # ── Git push (Phase 8, Step 8.2) — handled by Phase 8 dispatch
    # NOTE: Normally intercepted by _dispatch_phase8() before reaching here.
    if it == IntentType.GIT_PUSH:
        remote = _entity_value(intent, "remote") or ""
        branch = _entity_value(intent, "branch", "branch_name") or ""
        parts = ["git", "push"]
        if remote:
            parts.append(remote)
            if branch:
                parts.append(branch)
        return {"command": " ".join(parts), "working_directory": cwd}

    if it == IntentType.GIT_FETCH:
        return {"command": "git fetch --all", "working_directory": cwd}

    if it == IntentType.GIT_MERGE:
        branch = _entity_value(intent, "branch", "branch_name")
        if branch and _validate_branch_name(branch):
            return {"command": f"git merge {branch}", "working_directory": cwd}
        return {"command": "git merge", "working_directory": cwd}

    if it == IntentType.GIT_REBASE:
        branch = _entity_value(intent, "branch", "branch_name")
        return {"command": f"git rebase {branch}" if branch else "git rebase", "working_directory": cwd}

    if it == IntentType.GIT_STASH:
        action = _entity_value(intent, "action") or "push"
        return {"command": f"git stash {action}".strip(), "working_directory": cwd}

    if it == IntentType.GIT_CONFLICT_RESOLVE:
        return {"command": "git mergetool", "working_directory": cwd}

    # ── Checkout (Phase 8, Step 8.3) ────────────────────────────
    # NOTE: Normally intercepted by _dispatch_phase8() before reaching here.
    if it == IntentType.GIT_CHECKOUT:
        branch = _entity_value(intent, "branch", "branch_name")
        if branch and _validate_branch_name(branch):
            return {"command": f"git checkout {branch}", "working_directory": cwd}
        return {"command": "git branch -a", "working_directory": cwd}

    # ── Directory create/delete/copy ──────────────────────────
    if it == IntentType.CREATE_DIRECTORY:
        path = _entity_value(intent, "path", "dirname", "directory")
        target = resolve_path(path, cwd)
        if os.name == "nt":
            return {"command": f'New-Item -ItemType Directory -Path "{target}" -Force', "working_directory": cwd}
        return {"command": f'mkdir -p "{target}"', "working_directory": cwd}

    if it == IntentType.DELETE_DIRECTORY:
        path = _entity_value(intent, "path", "dirname", "directory")
        target = resolve_path(path, cwd)
        if os.name == "nt":
            return {"command": f'rmdir /s /q "{target}"', "working_directory": cwd}
        return {"command": f'rm -rf "{target}"', "working_directory": cwd}

    if it == IntentType.COPY_DIRECTORY:
        src = _entity_value(intent, "source", "path")
        dst = _entity_value(intent, "destination", "target")
        if os.name != "nt":
            return {"command": f'cp -r "{resolve_path(src, cwd)}" "{resolve_path(dst, cwd)}"', "working_directory": cwd}
        return {"command": f'xcopy /E /I "{resolve_path(src, cwd)}" "{resolve_path(dst, cwd)}"', "working_directory": cwd}

    # ── File copy (no dedicated copy tool) ────────────────────
    if it == IntentType.COPY_FILE:
        src = _entity_value(intent, "source", "path", "filename", "file")
        dst = _entity_value(intent, "destination", "target")
        src_resolved = resolve_path(src, cwd)
        dst_resolved = resolve_path(dst, cwd)
        if os.name != "nt":
            return {"command": f'cp "{src_resolved}" "{dst_resolved}"', "working_directory": cwd}
        return {"command": f'copy "{src_resolved}" "{dst_resolved}"', "working_directory": cwd}

    # ── Script execution (Phase 4 — ScriptExecutor integration) ─
    if it == IntentType.RUN_SCRIPT:
        script = _entity_value(intent, "script_path", "path", "filename", "file")
        if script:
            script_path = resolve_path(script, cwd)
            script_dir = os.path.dirname(script_path) or cwd

            # Determine interpreter via ScriptExecutor if available
            runner: Optional[str] = None
            if _SCRIPT_EXECUTOR_AVAILABLE:
                try:
                    from pathlib import Path as _Path
                    executor = ScriptExecutor()
                    lang = executor.detect_language(_Path(script_path))
                    if lang and lang != ScriptLanguage.UNKNOWN:
                        interp = executor.registry.get_interpreter(lang)
                        if interp and interp.available:
                            runner = interp.path
                            if interp.args:
                                runner += " " + " ".join(interp.args)
                except Exception:
                    pass  # fall through to extension-based lookup

            # Fallback: extension-based runner map
            if not runner:
                ext = os.path.splitext(script_path)[1].lower()
                runner = _SCRIPT_RUNNER_MAP.get(ext, "python")

            cmd_parts: List[str] = []

            # Check for virtual environment in script's directory
            for venv_name in ("venv", ".venv", "env"):
                venv_path = os.path.join(script_dir, venv_name)
                if os.path.isdir(venv_path):
                    if os.name == "nt":
                        activate = os.path.join(venv_path, "Scripts", "Activate.ps1")
                        if os.path.isfile(activate):
                            cmd_parts.append(f'& "{activate}";')
                    else:
                        activate = os.path.join(venv_path, "bin", "activate")
                        if os.path.isfile(activate):
                            cmd_parts.append(f'source "{activate}" &&')
                    break

            cmd_parts.append(f'{runner} "{script_path}"')
            return {
                "command": " ".join(cmd_parts),
                "working_directory": script_dir,
            }
        return {"command": "echo 'No script path provided'", "working_directory": cwd}

    # ── Dependency install (Phase 5, Step 5.2) ────────────────
    if it == IntentType.INSTALL_DEPENDENCY:
        pkg = _entity_value(intent, "package", "dependency", "name")
        # Check for an explicit install command in entities
        explicit_cmd = _entity_value(intent, "command")
        if explicit_cmd:
            return {"command": explicit_cmd, "working_directory": cwd}

        # Phase 5: use ProjectDependencyManager for smart detection
        if _DEP_MANAGER_AVAILABLE:
            if pkg:
                # Named packages — collect all package entities
                all_pkgs = intent.get_all_entities("package")
                if not all_pkgs:
                    all_pkgs = [pkg]
                elif pkg not in all_pkgs:
                    all_pkgs.insert(0, pkg)
                return {
                    "command": f"pip install {' '.join(all_pkgs)}",
                    "working_directory": cwd,
                    "_dep_manager_meta": {"packages": all_pkgs},
                }
            # No explicit packages — auto-detect from project
            mgr = ProjectDependencyManager()
            info = mgr.detect_project_dependencies(cwd)
            mgr_name = info.get("detected_manager")
            source = info.get("source_file")
            if mgr_name == "pipenv":
                return {"command": "pipenv install", "working_directory": cwd}
            if mgr_name == "conda" and source == "environment.yml":
                return {"command": "conda env create -f environment.yml", "working_directory": cwd}
            if mgr_name in ("npm", "yarn"):
                return {"command": f"{mgr_name} install", "working_directory": cwd}
            if source == "requirements.txt":
                return {"command": "pip install -r requirements.txt", "working_directory": cwd}
            if source in ("setup.py", "pyproject.toml"):
                return {"command": "pip install -e .", "working_directory": cwd}
            if info["python_packages"]:
                return {
                    "command": "pip install " + " ".join(info["python_packages"]),
                    "working_directory": cwd,
                }
            # No dependency file found — give an informative fallback
            return {
                "command": "echo 'No dependency file detected in the project directory'",
                "working_directory": cwd,
            }
        else:
            # Fallback: legacy detection
            detected = _detect_dependency_file(cwd)
            if detected:
                _, install_cmd = detected
                return {"command": install_cmd, "working_directory": cwd}
            if pkg:
                return {"command": f"pip install {pkg}", "working_directory": cwd}
        # No dependency file found and no packages specified
        return {
            "command": "echo 'No dependency file detected in the project directory'",
            "working_directory": cwd,
        }

    # ── Check dependency (Phase 5, Step 5.2) ──────────────────
    if it == IntentType.CHECK_DEPENDENCY:
        pkg = _entity_value(intent, "package", "dependency", "name")
        if pkg:
            # Phase 5: collect all unique package entities (deduplicated)
            all_pkgs = list(dict.fromkeys(intent.get_all_entities("package") or []))
            if pkg not in all_pkgs:
                all_pkgs.insert(0, pkg)
            return {
                "command": f"pip show {pkg}",
                "working_directory": cwd,
                "_check_packages": all_pkgs,
            }
        return {"command": "pip list", "working_directory": cwd}

    # ── Configure environment (Phase 5, Step 5.2) ─────────────
    if it == IntentType.CONFIGURE_ENVIRONMENT:
        return _build_args_configure_environment(intent, cwd)

    # ── Backend operations are handled by Phase 6 custom handlers ─
    # (BACKEND_BUILD, BACKEND_CONFIGURE, BACKEND_TEST, BACKEND_MODIFY,
    # BACKEND_LIST, UNDO_OPERATION, REDO_OPERATION map to None and are
    # intercepted in IntentToolBridge._dispatch_phase6 before reaching
    # this function.)

    # ── System info ───────────────────────────────────────────
    if it == IntentType.SYSTEM_INFO:
        if os.name == "nt":
            return {"command": "systeminfo", "working_directory": cwd}
        return {"command": "uname -a && df -h && free -m", "working_directory": cwd}

    # Fallback — echo the raw message
    return {"command": f"echo '{intent.raw_message}'", "working_directory": cwd}


# ═══════════════════════════════════════════════════════════════════════
# Argument builder dispatch table
# ═══════════════════════════════════════════════════════════════════════

_ARG_BUILDERS: Dict[str, Any] = {
    "change_directory":       _build_args_navigate,
    "list_directory":         _build_args_list_directory,
    "get_working_directory":  _build_args_get_cwd,
    "read_file":              _build_args_read_file,
    "write_file":             _build_args_write_file,
    "delete_file":            _build_args_delete_file,
    "move_file":              _build_args_move_file,
    "search_files":           _build_args_search_files,
    "file_info":              _build_args_file_info,
    "git_status":             _build_args_git_status,
    "git_commit":             _build_args_git_commit,
    "git_add":                _build_args_git_add,
    "git_log":                _build_args_git_log,
    "git_diff":               _build_args_git_diff,
    "git_branch":             _build_args_git_branch,
    "run_command":            _build_args_run_command,
}


# ═══════════════════════════════════════════════════════════════════════
# Build tool arguments — public API
# ═══════════════════════════════════════════════════════════════════════

def build_tool_arguments(
    intent: Intent,
    cwd: str,
    context: Optional[SessionContext] = None,
) -> Dict[str, Any]:
    """Build the argument dictionary for the tool mapped to *intent*.

    Falls back to ``_build_args_run_command`` for intents that route
    through the ``run_command`` tool.

    Parameters
    ----------
    context : SessionContext, optional
        Session context for path fallbacks and contextual resolution
        (Phase 4, Step 4.1).
    """
    tool_name = INTENT_TO_TOOL.get(intent.intent_type)
    if not tool_name:
        return {}
    builder = _ARG_BUILDERS.get(tool_name, _build_args_run_command)
    args = builder(intent, cwd)

    # Phase 4: Apply context-dependent fallbacks
    if context is not None:
        _apply_context_fallbacks(intent, args, cwd, context)

    return args


def _apply_context_fallbacks(
    intent: Intent,
    args: Dict[str, Any],
    cwd: str,
    context: SessionContext,
) -> None:
    """Apply SessionContext fallbacks to tool arguments (Phase 4, Step 4.1).

    Mutates *args* in place when the builder produced a generic fallback
    (e.g. resolved path == cwd because no path entity was found).
    """
    it = intent.intent_type

    # Fallback path from SessionContext for READ_FILE / WRITE_FILE / CREATE_FILE
    if it in (IntentType.READ_FILE, IntentType.WRITE_FILE, IntentType.CREATE_FILE):
        path = args.get("path", "")
        # If resolved path is just cwd (no file specified), try context
        if path == cwd and context.last_mentioned_paths:
            args["path"] = resolve_path(context.last_mentioned_paths[0], cwd)

    # Fallback for LIST_DIRECTORY — use context.current_directory
    if it == IntentType.LIST_DIRECTORY:
        path = args.get("path", "")
        if not path or path == cwd:
            args["path"] = context.current_directory


def _truncate_output(text: str, max_len: int = 5000) -> str:
    """Truncate *text* to *max_len* with middle elision (Phase 4, Step 4.4)."""
    if not text or len(text) <= max_len:
        return text or ""
    head_len = max_len // 2 - 20
    tail_len = max_len - head_len - 40
    return text[:head_len] + "\n[...truncated...]\n" + text[-tail_len:]


# ═══════════════════════════════════════════════════════════════════════
# Safety checks
# ═══════════════════════════════════════════════════════════════════════

def is_blocked(command: str) -> bool:
    """Return ``True`` if *command* matches a blocked pattern."""
    for rx in _BLOCKED_RE:
        if rx.search(command):
            return True
    return False


def is_dangerous(command: str) -> bool:
    """Return ``True`` if *command* matches a dangerous pattern."""
    for rx in _DANGEROUS_RE:
        if rx.search(command):
            return True
    return False


def is_safe(command: str) -> bool:
    """Return ``True`` if *command* is a known safe read-only operation."""
    cmd_lower = command.strip().lower()
    return cmd_lower in SAFE_COMMANDS or any(
        cmd_lower.startswith(s) for s in SAFE_COMMANDS
    )


def get_command_risk_level(command: str) -> Optional[str]:
    """Return the highest risk level matching *command*, or ``None`` if safe.

    Checks *command* against ``_DANGEROUS_RISK_MAP``.  When multiple
    patterns match, the most severe level wins (critical > high > medium > low).
    """
    highest: Optional[str] = None
    for pattern, level in _DANGEROUS_RISK_MAP.items():
        if re.search(pattern, command, re.IGNORECASE):
            if highest is None or _RISK_ORDER.get(level, 0) > _RISK_ORDER.get(highest, 0):
                highest = level
    return highest


# ═══════════════════════════════════════════════════════════════════════
# IntentToolBridge class
# ═══════════════════════════════════════════════════════════════════════

class IntentToolBridge:
    """Bridge between recognised ``Intent`` objects and the tool system.

    Usage::

        bridge = IntentToolBridge()
        result = bridge.dispatch(intent, cwd="/home/user/project")
    """

    def __init__(
        self,
        tool_registry=None,
        safety_manager=None,
        consent_callback=None,
    ):
        """Initialise the bridge.

        Parameters
        ----------
        tool_registry : optional
            A ``ToolRegistry`` instance.  If ``None``, the global
            singleton is obtained via ``get_tool_registry()``.
        safety_manager : optional
            A ``SafetyManager`` instance for consent flow.
        consent_callback : optional
            An async or sync callable ``(consent_request) -> bool``
            used when a dangerous operation needs user approval.
        """
        self._tool_registry = tool_registry
        self._safety_manager = safety_manager
        self._consent_callback = consent_callback
        self._execution_history: List[Dict[str, Any]] = []
        # Phase 5 — cached dependency manager (one per bridge lifetime)
        self._dep_mgr: Optional[ProjectDependencyManager] = None
        # Phase 6 — cached checkpoint/modifier managers
        self._checkpoint_mgr: Optional[Any] = None
        self._backend_modifier_inst: Optional[Any] = None
        self._profile_loader: Optional[Any] = None
        # Phase 7 — admin privilege handler + session escalation counter
        self._admin_handler: Optional[Any] = None
        self._admin_escalation_count: int = 0
        # Phase 8 — LLM reference for auto-commit-message generation
        self._llm_provider: Optional[Any] = None

    def _get_dep_manager(self) -> Optional[ProjectDependencyManager]:
        """Return a cached ``ProjectDependencyManager`` instance."""
        if not _DEP_MANAGER_AVAILABLE:
            return None
        if self._dep_mgr is None:
            self._dep_mgr = ProjectDependencyManager()
        return self._dep_mgr

    # ── lazy init ─────────────────────────────────────────────────

    def _get_registry(self):
        if self._tool_registry is None:
            try:
                from proxima.agent.dynamic_tools.tool_registry import get_tool_registry
                self._tool_registry = get_tool_registry()
            except Exception:
                logger.warning("Could not obtain ToolRegistry singleton")
        return self._tool_registry

    # ── public API ────────────────────────────────────────────────

    def dispatch(
        self,
        intent: Intent,
        cwd: str | None = None,
        execution_context=None,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Dispatch *intent* to the appropriate tool and return a ``ToolResult``.

        Parameters
        ----------
        intent : Intent
            The recognised intent (from ``RobustNLProcessor.recognize_intent``).
        cwd : str, optional
            Current working directory.  Falls back to ``os.getcwd()``.
        execution_context : ExecutionContext, optional
            Full execution context object (passed through to tools).
        session_context : SessionContext, optional
            Session context for output capture, path fallbacks, and
            directory stack management (Phase 4).

        Returns
        -------
        ToolResult
            The result of tool execution.
        """
        cwd = cwd or os.getcwd()
        start = time.monotonic()

        tool_name = INTENT_TO_TOOL.get(intent.intent_type)

        # Phase 6: Custom handlers for backend operations and undo/redo
        phase6_result = self._dispatch_phase6(intent, cwd, session_context)
        if phase6_result is not None:
            elapsed = (time.monotonic() - start) * 1000
            phase6_result.execution_time_ms = elapsed
            self._capture_output(intent, phase6_result, session_context)
            self._record(
                intent,
                phase6_result.tool_name or intent.intent_type.name,
                {},
                phase6_result,
            )
            return phase6_result

        # Phase 7: Administrative access and privilege escalation
        phase7_result = self._dispatch_phase7(intent, cwd, session_context)
        if phase7_result is not None:
            elapsed = (time.monotonic() - start) * 1000
            phase7_result.execution_time_ms = elapsed
            self._capture_output(intent, phase7_result, session_context)
            self._record(
                intent,
                phase7_result.tool_name or intent.intent_type.name,
                {},
                phase7_result,
            )
            return phase7_result

        # Phase 8: GitHub / Git operations (clone, pull, push, etc.)
        phase8_result = self._dispatch_phase8(intent, cwd, session_context)
        if phase8_result is not None:
            elapsed = (time.monotonic() - start) * 1000
            phase8_result.execution_time_ms = elapsed
            self._capture_output(intent, phase8_result, session_context)
            self._record(
                intent,
                phase8_result.tool_name or intent.intent_type.name,
                {},
                phase8_result,
            )
            return phase8_result

        # Intents with no direct tool mapping
        if tool_name is None:
            return self._handle_no_tool(intent)

        # Phase 5, Step 5.2: CHECK_DEPENDENCY — structured check via manager
        if (
            intent.intent_type == IntentType.CHECK_DEPENDENCY
            and _DEP_MANAGER_AVAILABLE
        ):
            return self._handle_check_dependency(intent, cwd, session_context)

        # Build arguments (with context fallbacks)
        args = build_tool_arguments(intent, cwd, context=session_context)

        # Phase 4: Safety — consent for destructive file operations
        consent_result = self._check_destructive_consent(intent, args, cwd)
        if consent_result is not None:
            return consent_result  # denied or blocked

        # Safety pre-screening for commands
        if tool_name == "run_command" and "command" in args:
            cmd_str = args["command"]
            if is_blocked(cmd_str):
                return ToolResult(
                    success=False,
                    error=f"Command blocked by safety policy: {cmd_str}",
                    tool_name=tool_name,
                    message="🚫 This command is blocked for safety reasons.",
                )
            if is_dangerous(cmd_str):
                # Attempt consent flow
                if not self._request_consent(intent, cmd_str):
                    return ToolResult(
                        success=False,
                        error="User declined dangerous command",
                        tool_name=tool_name,
                        message="⚠️ Dangerous command declined by user.",
                    )

        # Phase 7: Intercept commands that require admin privileges
        if (
            tool_name in ("run_command", "run_script")
            and "command" in args
            and _ADMIN_HANDLER_AVAILABLE
        ):
            admin_intercept = self._intercept_admin_for_command(
                intent, args["command"], cwd, session_context,
            )
            if admin_intercept is not None:
                admin_intercept.execution_time_ms = (
                    (time.monotonic() - start) * 1000
                )
                self._capture_output(intent, admin_intercept, session_context)
                self._record(
                    intent,
                    admin_intercept.tool_name or tool_name,
                    args,
                    admin_intercept,
                )
                return admin_intercept

        # Obtain tool instance
        registry = self._get_registry()
        if registry is None:
            result = self._fallback_execute(intent, args, cwd)
            self._capture_output(intent, result, session_context)
            return result

        tool_inst = registry.get_tool_instance(tool_name)
        if tool_inst is None:
            result = self._fallback_execute(intent, args, cwd)
            self._capture_output(intent, result, session_context)
            return result

        # Execute through the tool interface
        try:
            if execution_context is not None:
                result = tool_inst.execute(args, execution_context)
            else:
                # Build a minimal context
                from proxima.agent.dynamic_tools.execution_context import ExecutionContext
                ctx = ExecutionContext(current_directory=cwd)
                result = tool_inst.execute(args, ctx)

            elapsed = (time.monotonic() - start) * 1000
            result.execution_time_ms = elapsed
            result.tool_name = tool_name

            # Phase 4: Capture output to SessionContext
            self._capture_output(intent, result, session_context)

            # Phase 5, Step 5.3: Error detection loop for failed commands
            if (
                not result.success
                and intent.intent_type in (
                    IntentType.RUN_COMMAND,
                    IntentType.RUN_SCRIPT,
                    IntentType.INSTALL_DEPENDENCY,
                )
                and _DEP_MANAGER_AVAILABLE
            ):
                result = self._attempt_error_fix(
                    intent, result, args, cwd, tool_name,
                    execution_context, session_context,
                    tool_inst=tool_inst,
                )

            # Phase 5, Step 5.2: Update SessionContext after CONFIGURE_ENVIRONMENT
            if (
                result.success
                and intent.intent_type == IntentType.CONFIGURE_ENVIRONMENT
                and session_context is not None
            ):
                env_name = args.get("_env_name", ".venv")
                env_path = args.get("_env_path") or os.path.join(cwd, env_name)
                session_context.active_environments[env_name] = env_path

            # Phase 5, Step 5.2: Update installed_packages after INSTALL_DEPENDENCY
            if (
                result.success
                and intent.intent_type == IntentType.INSTALL_DEPENDENCY
                and session_context is not None
            ):
                meta = args.get("_dep_manager_meta", {})
                for pkg in meta.get("packages", []):
                    if pkg not in session_context.installed_packages:
                        session_context.installed_packages.append(pkg)

            # Record history
            self._record(intent, tool_name, args, result)
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Tool %s failed: %s", tool_name, exc, exc_info=True)
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name=tool_name,
                execution_time_ms=elapsed,
                message=f"❌ {tool_name} failed: {exc}",
            )

    def dispatch_multi_step(
        self,
        intent: Intent,
        cwd: str | None = None,
        execution_context=None,
        session_context: Optional[SessionContext] = None,
    ) -> List[ToolResult]:
        """Dispatch each sub-intent of a ``MULTI_STEP`` intent sequentially.

        Context (cwd) is updated between steps: if a step changes the
        working directory, subsequent steps use the new directory.
        The SessionContext directory stack is also updated (Phase 4).

        Returns a list of ``ToolResult`` — one per sub-intent.
        """
        cwd = cwd or os.getcwd()
        results: List[ToolResult] = []

        for sub in intent.sub_intents:
            result = self.dispatch(
                sub, cwd=cwd,
                execution_context=execution_context,
                session_context=session_context,
            )
            results.append(result)

            # Update cwd if the step navigated or cloned a repo
            if result.success:
                if (
                    sub.intent_type == IntentType.NAVIGATE_DIRECTORY
                    and result.result
                ):
                    new_dir = str(result.result)
                    if os.path.isdir(new_dir):
                        cwd = new_dir
                        # Phase 4: update current_directory for step chaining.
                        # NOTE: push_directory is handled by TUI _update_nl_context
                        # to avoid double-push.
                        if session_context is not None:
                            session_context.current_directory = new_dir
                elif sub.intent_type == IntentType.GIT_CLONE:
                    # Phase 8: After cloning, cd into the cloned directory.
                    # The Phase 8 handler already updates session_context,
                    # but we still need to update the local cwd for chaining.
                    clone_dir = None
                    if isinstance(result.result, dict):
                        clone_dir = result.result.get("destination")
                    if not clone_dir:
                        dest = _entity_value(sub, "destination", "path", "dirname")
                        url_raw = _entity_value(sub, "url", "repo_url", "repository")
                        if dest:
                            clone_dir = resolve_path(dest, cwd)
                        elif url_raw:
                            repo_name = _repo_name_from_url(
                                _normalise_git_url(url_raw),
                            )
                            clone_dir = os.path.join(cwd, repo_name)
                    if clone_dir and os.path.isdir(clone_dir):
                        cwd = clone_dir

            # Stop on failure (configurable in future)
            if not result.success:
                logger.warning(
                    "Multi-step halted at step %d: %s",
                    len(results),
                    result.error,
                )
                break

        return results

    # ── helpers ───────────────────────────────────────────────────

    def _handle_no_tool(self, intent: Intent) -> ToolResult:
        """Handle intents that have no direct tool mapping.

        Provides custom logic for some intent types (e.g. QUERY_STATUS)
        and returns an informational acknowledgement for the rest.
        """
        it = intent.intent_type

        # ── QUERY_STATUS: report last operation from execution history ──
        if it == IntentType.QUERY_STATUS:
            if self._execution_history:
                last = self._execution_history[-1]
                status = "succeeded ✅" if last["success"] else "failed ❌"
                return ToolResult(
                    success=True,
                    result=last,
                    tool_name="query_status",
                    message=(
                        f"Last operation: **{last['intent_type']}** "
                        f"via *{last['tool_name']}* — {status}"
                    ),
                )
            return ToolResult(
                success=True,
                result=None,
                tool_name="query_status",
                message="No operations have been executed yet in this session.",
            )

        # ── Default: informational acknowledgement ──
        label = it.name.replace("_", " ").title()
        return ToolResult(
            success=True,
            result=None,
            tool_name="",
            message=(
                f"✅ Intent recognised: **{label}**\n"
                f"   Confidence: {intent.confidence:.0%}\n"
                f"   Entities: {', '.join(e.value for e in intent.entities) or 'none'}\n"
                f"   ℹ️ This intent is handled by the agent orchestration layer."
            ),
        )

    def _request_consent(self, intent: Intent, payload: Any) -> bool:
        """Ask the user for consent before running a dangerous command.

        *payload* may be a plain string description **or** a
        ``ConsentRequest`` object (Phase 4, Step 4.5).  If the callback
        rejects the ``ConsentRequest`` (e.g. old callback that only
        accepts strings), a fallback to the string representation is
        attempted automatically.
        """
        if self._consent_callback is None:
            # No callback — deny by default
            return False
        try:
            return bool(self._consent_callback(payload))
        except TypeError:
            # Callback may not accept ConsentRequest — retry with string
            try:
                desc = (
                    payload.description
                    if hasattr(payload, "description")
                    else str(payload)
                )
                return bool(self._consent_callback(desc))
            except Exception:
                return False
        except Exception:
            return False

    def _check_destructive_consent(
        self,
        intent: Intent,
        args: Dict[str, Any],
        cwd: str,
    ) -> Optional[ToolResult]:
        """Check consent for destructive file/directory operations (Phase 4, Step 4.5).

        Returns a ``ToolResult`` (denied/blocked) if the operation should
        NOT proceed, or ``None`` if execution may continue.

        When the ``safety`` module is available, builds a proper
        ``ConsentRequest`` and passes it to the callback.
        """
        it = intent.intent_type
        tool_name = INTENT_TO_TOOL.get(it, "") or ""

        # Check destructive map (uses module-level _DESTRUCTIVE_INTENTS)
        if it in _DESTRUCTIVE_INTENTS:
            _consent_type_name, risk, desc, reversible = _DESTRUCTIVE_INTENTS[it]
            target = args.get("path") or args.get("source") or "unknown"
            full_desc = f"{desc}: {target}"

            # Build a ConsentRequest when the safety module is available
            consent_payload: Any = full_desc
            if _SAFETY_AVAILABLE:
                try:
                    consent_payload = ConsentRequest(
                        id=f"{it.name}_{int(time.time())}",
                        consent_type=ConsentType.FILE_MODIFICATION,
                        operation=it.name.lower(),
                        description=full_desc,
                        details={"target": target, "cwd": cwd, "args": args},
                        tool_name=tool_name,
                        risk_level=risk,
                        reversible=reversible,
                    )
                except Exception:
                    pass  # fall back to string

            if not self._request_consent(intent, consent_payload):
                return ToolResult(
                    success=False,
                    error=f"User declined destructive operation: {full_desc}",
                    tool_name=tool_name,
                    message=f"⚠️ Operation declined: {full_desc}",
                )

        # WRITE_FILE overwrite check
        if it in (IntentType.WRITE_FILE, IntentType.CREATE_FILE):
            target_path = args.get("path", "")
            if target_path and os.path.isfile(target_path):
                # File exists — request overwrite consent
                if not self._request_consent(
                    intent, f"overwrite existing file: {target_path}"
                ):
                    return ToolResult(
                        success=False,
                        error=f"User declined file overwrite: {target_path}",
                        tool_name=tool_name,
                        message=f"⚠️ File overwrite declined: {target_path}",
                    )

        # ADMIN_ELEVATE — Phase 7 handles the full escalation flow.
        # The ``_dispatch_phase7`` / ``_handle_admin_elevate`` methods
        # perform consent, security checks, and execution.
        # The old stub is kept only as a guard when Phase 7 is not
        # available (e.g. missing admin_privilege_handler module).
        if it == IntentType.ADMIN_ELEVATE:
            if _ADMIN_HANDLER_AVAILABLE:
                # Phase 7 will handle — do not block here.
                return None
            # Fallback: simple consent when handler is unavailable
            if not self._request_consent(
                intent, "elevate to administrator/root privileges"
            ):
                return ToolResult(
                    success=False,
                    error="User declined admin elevation",
                    tool_name="",
                    message="⚠️ Admin elevation declined.",
                )

        return None  # OK to proceed

    def _capture_output(
        self,
        intent: Intent,
        result: ToolResult,
        session_context: Optional[SessionContext] = None,
    ) -> None:
        """Capture tool output to SessionContext (Phase 4, Step 4.4).

        Stores truncated output in ``session_context.last_operation_result``
        and updates ``session_context.last_script_executed`` for script
        executions.
        """
        if session_context is None:
            return

        it = intent.intent_type

        # Capture output for RUN_COMMAND and RUN_SCRIPT
        if it in (IntentType.RUN_COMMAND, IntentType.RUN_SCRIPT):
            output = str(result.result) if result.result else (result.message or "")
            # Phase 4 spec: keep full output available via a reference
            if hasattr(session_context, 'variables') and isinstance(
                getattr(session_context, 'variables', None), dict
            ):
                session_context.variables["_full_last_output"] = output
            session_context.last_operation_result = _truncate_output(output, 5000)

            # Track script execution
            if it == IntentType.RUN_SCRIPT:
                script = _entity_value(
                    intent, "script_path", "path", "filename", "file"
                )
                if script:
                    session_context.last_script_executed = script

        # Store last operation result for all other intents (for "it"/"that" resolution)
        elif result.result:
            full_output = str(result.result)
            if hasattr(session_context, 'variables') and isinstance(
                getattr(session_context, 'variables', None), dict
            ):
                session_context.variables["_full_last_output"] = full_output
            session_context.last_operation_result = _truncate_output(
                full_output, 5000
            )

    # ── Phase 5 helpers ──────────────────────────────────────────

    def _handle_check_dependency(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``CHECK_DEPENDENCY`` via ``ProjectDependencyManager`` (Phase 5, Step 5.2).

        Produces a formatted table of package statuses instead of raw
        shell output.
        """
        start = time.monotonic()
        mgr = self._get_dep_manager() or ProjectDependencyManager()

        # Gather requested packages
        pkg = _entity_value(intent, "package", "dependency", "name")
        all_pkgs = intent.get_all_entities("package") if pkg else []
        if pkg and pkg not in all_pkgs:
            all_pkgs.insert(0, pkg)

        if not all_pkgs:
            # No specific package — list all installed
            try:
                proc = subprocess.run(
                    [mgr.python_executable, "-m", "pip", "list", "--format=columns"],
                    capture_output=True, text=True, timeout=30,
                )
                output = proc.stdout or proc.stderr or "No packages found."
                elapsed = (time.monotonic() - start) * 1000
                result = ToolResult(
                    success=proc.returncode == 0,
                    result=output,
                    tool_name="check_dependency",
                    execution_time_ms=elapsed,
                    message=output[:3000],
                )
                self._capture_output(intent, result, session_context)
                self._record(intent, "check_dependency", {}, result)
                return result
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                return ToolResult(
                    success=False, error=str(exc),
                    tool_name="check_dependency",
                    execution_time_ms=elapsed,
                    message=f"❌ Failed to list packages: {exc}",
                )

        # Check each package (batch for performance)
        batch = mgr.check_installed_batch(all_pkgs)
        rows: List[str] = ["| Package | Installed | Version |", "| --- | --- | --- |"]
        all_ok = True
        for p in all_pkgs:
            installed, version = batch.get(p, (False, None))
            status = "✅ Yes" if installed else "❌ No"
            ver = version or "—"
            rows.append(f"| {p} | {status} | {ver} |")
            if not installed:
                all_ok = False

        table = "\n".join(rows)
        elapsed = (time.monotonic() - start) * 1000
        result = ToolResult(
            success=True,
            result=table,
            tool_name="check_dependency",
            execution_time_ms=elapsed,
            message=table,
        )
        self._capture_output(intent, result, session_context)
        self._record(intent, "check_dependency", {"packages": all_pkgs}, result)
        return result

    def _attempt_error_fix(
        self,
        intent: Intent,
        result: ToolResult,
        args: Dict[str, Any],
        cwd: str,
        tool_name: str,
        execution_context,
        session_context: Optional[SessionContext],
        *,
        tool_inst=None,
    ) -> ToolResult:
        """Attempt automatic error detection and fix (Phase 5, Step 5.3).

        If the ``ProjectDependencyManager`` identifies a fixable error
        pattern in the failed output, this method:
        1. Asks for user consent to apply the fix.
        2. Executes the fix command.
        3. Retries the original command.
        4. Returns either the retry result or the original failure
           (annotated with the fix suggestion).
        """
        error_output = str(result.result or "") + str(result.error or "")
        if not error_output:
            return result

        mgr = self._get_dep_manager() or ProjectDependencyManager()
        fix_cmd = mgr.detect_and_fix_errors(error_output, cwd)
        if fix_cmd is None:
            # No auto-fix — annotate with any recognised descriptions
            descriptions = mgr.describe_errors(error_output)
            if descriptions:
                extra = "\n".join(
                    f"• {d['description']}" + (f" — fix: `{d['fix']}`" if d.get("fix") else "")
                    for d in descriptions
                )
                result.message = (
                    (result.message or "")
                    + f"\n\n🔍 **Detected issues:**\n{extra}"
                )
            return result

        # Ask user for consent to apply the fix
        consent_desc = f"Auto-fix: {fix_cmd}"
        if not self._request_consent(intent, consent_desc):
            result.message = (
                (result.message or "")
                + f"\n\n💡 **Suggested fix:** `{fix_cmd}`\n"
                + "Run the fix command manually or approve when prompted."
            )
            return result

        # Execute the fix
        logger.info("Phase 5: applying auto-fix: %s", fix_cmd)
        fix_args = {"command": fix_cmd, "working_directory": cwd}
        if tool_inst is not None:
            try:
                from proxima.agent.dynamic_tools.execution_context import ExecutionContext
                ctx = execution_context or ExecutionContext(current_directory=cwd)
                fix_result = tool_inst.execute(fix_args, ctx)
            except Exception as exc:
                logger.warning("Auto-fix execution failed: %s", exc)
                result.message = (
                    (result.message or "")
                    + f"\n\n⚠️ Fix `{fix_cmd}` failed: {exc}"
                )
                return result
        else:
            fix_result = self._fallback_execute(intent, fix_args, cwd)

        if not fix_result.success:
            result.message = (
                (result.message or "")
                + f"\n\n⚠️ Attempted fix `{fix_cmd}` but it failed."
            )
            return result

        # Retry the original command
        logger.info("Phase 5: retrying original command after fix")
        if tool_inst is not None:
            try:
                from proxima.agent.dynamic_tools.execution_context import ExecutionContext
                ctx = execution_context or ExecutionContext(current_directory=cwd)
                retry_result = tool_inst.execute(args, ctx)
            except Exception as exc:
                result.message = (
                    (result.message or "")
                    + f"\n\n✅ Fix applied (`{fix_cmd}`), but retry failed: {exc}"
                )
                return result
        else:
            retry_result = self._fallback_execute(intent, args, cwd)

        if retry_result.success:
            retry_result.message = (
                f"✅ Auto-fixed and retried successfully.\n"
                f"Fix applied: `{fix_cmd}`\n\n"
                + (retry_result.message or str(retry_result.result or ""))
            )
            # Update capture with the successful retry
            self._capture_output(intent, retry_result, session_context)
            return retry_result
        else:
            result.message = (
                (result.message or "")
                + f"\n\n⚠️ Fix `{fix_cmd}` was applied, but the command still fails.\n"
                + f"Original error: {result.error}\n"
                + f"Retry error: {retry_result.error}"
            )
            return result

    def check_backend_dependencies(
        self,
        backend_name: str,
        cwd: str,
    ) -> ToolResult:
        """Pre-check backend dependencies before build (Phase 5, Step 5.4).

        Loads the backend's build profile, checks each required Python
        package via ``ProjectDependencyManager``, and returns a report.
        Missing packages are listed so the caller (TUI) can offer to
        install them.
        """
        start = time.monotonic()

        loader = self._get_profile_loader()
        if loader is None:
            return ToolResult(
                success=False,
                error="BuildProfileLoader not available",
                tool_name="check_backend_deps",
                message="⚠️ Cannot load backend profiles — module unavailable.",
            )
        if not _DEP_MANAGER_AVAILABLE:
            return ToolResult(
                success=False,
                error="ProjectDependencyManager not available",
                tool_name="check_backend_deps",
                message="⚠️ Dependency manager unavailable.",
            )

        profile = loader.get_backend_config(backend_name)
        if profile is None:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                success=False,
                error=f"No build profile found for '{backend_name}'",
                tool_name="check_backend_deps",
                execution_time_ms=elapsed,
                message=f"⚠️ No profile for backend '{backend_name}'.",
            )

        deps_section = profile.get("dependencies", {})
        packages = deps_section.get("packages", [])
        system_packages = deps_section.get("system_packages", [])

        mgr = self._get_dep_manager() or ProjectDependencyManager()
        checks = mgr.check_backend_dependencies(packages)
        missing = [c for c in checks if not c["installed"]]

        # Check system packages via 'where' (Win) / 'which' (Unix)
        sys_missing: List[str] = []
        for sp in system_packages:
            which_cmd = "where" if os.name == "nt" else "which"
            try:
                proc = subprocess.run(
                    [which_cmd, sp], capture_output=True, text=True, timeout=10,
                )
                if proc.returncode != 0:
                    sys_missing.append(sp)
            except Exception:
                sys_missing.append(sp)

        # Build formatted result
        lines: List[str] = [f"**Dependency check for {backend_name}:**\n"]
        for c in checks:
            icon = "✅" if c["installed"] else "❌"
            ver = f"(v{c['version']})" if c["version"] else ""
            lines.append(f"  {icon} {c['required']} {ver}")

        if sys_missing:
            lines.append("\n**Missing system packages:**")
            for sp in sys_missing:
                lines.append(f"  ❌ {sp}")

        if missing:
            install_cmd = "pip install " + " ".join(c["required"] for c in missing)
            lines.append(f"\n💡 Install missing: `{install_cmd}`")

        report = "\n".join(lines)
        elapsed = (time.monotonic() - start) * 1000
        return ToolResult(
            success=len(missing) == 0 and len(sys_missing) == 0,
            result={
                "checks": checks,
                "missing_python": [c["required"] for c in missing],
                "missing_system": sys_missing,
            },
            tool_name="check_backend_deps",
            execution_time_ms=elapsed,
            message=report,
        )

    # ══════════════════════════════════════════════════════════════
    # Phase 6 — Backend Build, Compilation, and Code Modification
    # ══════════════════════════════════════════════════════════════

    def _get_checkpoint_manager(self) -> Optional[Any]:
        """Return a cached ``CheckpointManager`` instance (Phase 6)."""
        if not _CHECKPOINT_MANAGER_AVAILABLE:
            return None
        if self._checkpoint_mgr is None:
            self._checkpoint_mgr = CheckpointManager()
        return self._checkpoint_mgr

    def _get_backend_modifier(self) -> Optional[Any]:
        """Return a cached ``BackendModifier`` instance (Phase 6)."""
        if not _BACKEND_MODIFIER_AVAILABLE:
            return None
        if self._backend_modifier_inst is None:
            try:
                from proxima.agent.safety import RollbackManager
                rm = RollbackManager()
            except ImportError:
                rm = None
            self._backend_modifier_inst = BackendModifier(
                rollback_manager=rm,
                project_root=os.getcwd(),
            )
        return self._backend_modifier_inst

    def _get_profile_loader(self):
        """Return a cached ``BuildProfileLoader`` instance (Phase 6)."""
        if not _BUILD_PROFILE_AVAILABLE:
            return None
        if self._profile_loader is None:
            self._profile_loader = BuildProfileLoader()
        return self._profile_loader

    def _resolve_backend_name(self, raw_name: str) -> Optional[str]:
        """Normalise a user-supplied backend name to a YAML profile key.

        Checks the static ``_BACKEND_NAME_MAP`` first, then tries a
        fuzzy substring match, and finally queries the ``BuildProfileLoader``
        for exact profile keys.  Returns ``None`` if no match is found.
        """
        if not raw_name:
            return None
        normalised = raw_name.strip().lower().replace("-", " ").replace("_", " ")
        # Direct match
        if normalised in _BACKEND_NAME_MAP:
            return _BACKEND_NAME_MAP[normalised]
        # Fuzzy substring match
        for key, value in _BACKEND_NAME_MAP.items():
            if key in normalised or normalised in key:
                return value
        # Try as-is (may already be a valid profile key)
        profile_key = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
        loader = self._get_profile_loader()
        if loader and loader.get_backend_config(profile_key):
            return profile_key
        return None

    def _dispatch_phase6(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> Optional[ToolResult]:
        """Route Phase 6 intents to their custom handlers.

        Returns ``None`` if the intent is not a Phase 6 intent so that
        ``dispatch()`` continues with the standard pipeline.
        """
        it = intent.intent_type
        if it not in _PHASE6_INTENTS:
            return None
        if it == IntentType.BACKEND_BUILD:
            return self._handle_backend_build(intent, cwd, session_context)
        if it == IntentType.BACKEND_MODIFY:
            return self._handle_backend_modify(intent, cwd, session_context)
        if it == IntentType.BACKEND_TEST:
            return self._handle_backend_test(intent, cwd, session_context)
        if it == IntentType.BACKEND_CONFIGURE:
            return self._handle_backend_configure(intent, cwd, session_context)
        if it == IntentType.BACKEND_LIST:
            return self._handle_backend_list(intent, cwd)
        if it == IntentType.UNDO_OPERATION:
            return self._handle_undo(intent, session_context)
        if it == IntentType.REDO_OPERATION:
            return self._handle_redo(intent, session_context)
        return None

    # ── Step 6.1: Backend Build via YAML Profiles ─────────────────

    def _handle_backend_build(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Build/compile a quantum simulator backend (Phase 6, Step 6.1).

        Pipeline:
        1. Identify backend and resolve to a YAML profile key.
        2. Load the profile from ``configs/backend_build_profiles.yaml``.
        3. Pre-check dependencies (Python packages + system tools).
        4. Execute build steps sequentially via ``subprocess``.
        5. Run verification (import test, expected files).
        6. Return a formatted report.
        """
        start = time.monotonic()
        backend_raw = _entity_value(intent, "backend", "name") or ""

        # Resolve backend name
        profile_key = self._resolve_backend_name(backend_raw)

        # Fallback to session context
        if not profile_key and session_context:
            if session_context.last_built_backend:
                profile_key = self._resolve_backend_name(
                    session_context.last_built_backend
                )

        if not profile_key and not backend_raw:
            return ToolResult(
                success=False,
                error="No backend specified",
                tool_name="backend_build",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    "❌ Please specify which backend to build.\n"
                    "Examples: 'build cirq', 'build lret pennylane', 'build qiskit'"
                ),
            )

        # Load YAML profile
        profile = None
        if profile_key:
            loader = self._get_profile_loader()
            profile = loader.get_backend_config(profile_key) if loader else None

        if profile:
            return self._build_from_profile(
                profile_key, profile, cwd, session_context, start, intent
            )
        return self._build_generic(
            backend_raw or profile_key or "", cwd, session_context, start
        )

    def _build_from_profile(
        self,
        profile_key: str,
        profile: Dict[str, Any],
        cwd: str,
        session_context: Optional[SessionContext],
        start: float,
        intent: Optional[Intent] = None,
    ) -> ToolResult:
        """Execute a full build pipeline from a YAML profile (Step 6.1)."""
        backend_name = profile.get("name", profile_key)
        backend_dir_rel = profile.get("directory", ".")
        backend_dir = (
            os.path.join(cwd, backend_dir_rel)
            if not os.path.isabs(backend_dir_rel)
            else backend_dir_rel
        )
        parts: List[str] = [f"## 🔨 Building **{backend_name}**\n"]

        # ── Dependency pre-check ─────────────────────────────────
        dep_section = profile.get("dependencies", {})
        packages = dep_section.get("packages", [])
        system_packages = dep_section.get("system_packages", [])

        if packages and _DEP_MANAGER_AVAILABLE:
            mgr = self._get_dep_manager() or ProjectDependencyManager()
            checks = mgr.check_backend_dependencies(packages)
            missing = [c for c in checks if not c["installed"]]
            if missing:
                missing_list = ", ".join(c["required"] for c in missing)
                parts.append(f"⚠️ Missing dependencies: {missing_list}")

                # Request user consent before installing (spec requirement)
                install_consent_msg = (
                    f"Install {len(missing)} missing package(s) for "
                    f"{backend_name}?\n  {missing_list}"
                )
                consent_payload: Any = install_consent_msg
                if _SAFETY_AVAILABLE:
                    try:
                        consent_payload = ConsentRequest(
                            id=f"dep_install_{int(time.time())}",
                            consent_type=ConsentType.BACKEND_MODIFICATION,
                            operation="dependency_install",
                            description=install_consent_msg,
                            details={
                                "packages": [c["required"] for c in missing],
                            },
                            tool_name="backend_build",
                            risk_level="medium",
                            reversible=False,
                        )
                    except Exception:
                        pass

                if intent and not self._request_consent(intent, consent_payload):
                    parts.append(
                        "⚠️ Dependency installation skipped (user declined)."
                    )
                else:
                    install_cmd = "pip install " + " ".join(
                        c["required"] for c in missing
                    )
                    parts.append(f"📦 Installing: `{install_cmd}`")
                    try:
                        proc = subprocess.run(
                            install_cmd,
                            shell=True,
                            capture_output=True,
                            text=True,
                            cwd=backend_dir if os.path.isdir(backend_dir) else cwd,
                            timeout=300,
                        )
                        if proc.returncode == 0:
                            parts.append("✅ Dependencies installed successfully")
                        else:
                            parts.append(
                                f"❌ Dependency installation failed:\n```\n"
                                f"{(proc.stderr or proc.stdout or '')[:500]}\n```"
                            )
                            return ToolResult(
                                success=False,
                                result="\n".join(parts),
                                tool_name="backend_build",
                                execution_time_ms=(time.monotonic() - start) * 1000,
                                message="\n".join(parts),
                            )
                    except subprocess.TimeoutExpired:
                        parts.append("❌ Dependency installation timed out (300s)")
                        return ToolResult(
                            success=False,
                            result="\n".join(parts),
                            tool_name="backend_build",
                            execution_time_ms=(time.monotonic() - start) * 1000,
                            message="\n".join(parts),
                        )
            else:
                parts.append("✅ All Python dependencies satisfied")

        # Check system packages
        if system_packages:
            which_cmd = "where" if os.name == "nt" else "which"
            sys_missing: List[str] = []
            for sp in system_packages:
                try:
                    proc = subprocess.run(
                        [which_cmd, sp],
                        capture_output=True, text=True, timeout=10,
                    )
                    if proc.returncode != 0:
                        sys_missing.append(sp)
                except Exception:
                    sys_missing.append(sp)
            if sys_missing:
                parts.append(
                    f"⚠️ Missing system packages: {', '.join(sys_missing)}\n"
                    "Please install them manually before building."
                )

        # ── Build execution ──────────────────────────────────────
        build_steps = profile.get("build_steps", [])
        total = len(build_steps)
        step_results: List[Dict[str, Any]] = []
        working_dir = backend_dir if os.path.isdir(backend_dir) else cwd

        # Prepare environment variables from profile
        env = dict(os.environ)
        for k, v in profile.get("environment_variables", {}).items():
            env[k] = str(v).replace("${PROJECT_ROOT}", cwd)

        for idx, step in enumerate(build_steps, 1):
            step_id = step.get("step_id", idx)
            command = step.get("command", "")
            description = step.get("description", command)
            timeout = step.get("timeout", 120)
            retry = step.get("retry", False)
            optional = step.get("optional", False)
            step_cwd = step.get("working_dir")
            if step_cwd and step_cwd != ".":
                step_cwd = (
                    os.path.join(working_dir, step_cwd)
                    if not os.path.isabs(step_cwd)
                    else step_cwd
                )
            else:
                step_cwd = working_dir

            parts.append(f"\n⚙️ **Step {idx}/{total}:** {description}")

            max_attempts = int(retry) if isinstance(retry, int) and retry > 1 else (2 if retry else 1)
            success = False
            output = ""
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    parts.append(f"  🔄 Retry attempt {attempt}/{max_attempts}")
                    time.sleep(5)
                try:
                    proc = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=step_cwd,
                        timeout=timeout,
                        env=env,
                    )
                    output = proc.stdout or proc.stderr or ""
                    if proc.returncode == 0:
                        success = True
                        break
                    output = proc.stderr or proc.stdout or ""
                except subprocess.TimeoutExpired:
                    output = f"Timed out after {timeout}s"
                except Exception as exc:
                    output = str(exc)

            step_results.append({
                "step_id": step_id,
                "success": success,
                "output": output[:500],
            })

            if success:
                parts.append("  ✅ Completed")
            elif optional:
                parts.append("  ⏭️ Skipped (optional step)")
            else:
                parts.append(
                    f"  ❌ Failed\n```\n{output[:300]}\n```"
                )
                parts.append(f"\n❌ Build halted at step {idx}/{total}.")
                return ToolResult(
                    success=False,
                    result={"step_results": step_results},
                    tool_name="backend_build",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message="\n".join(parts),
                )

        # ── Verification ─────────────────────────────────────────
        verification = profile.get("verification", {})
        verify_ok = True

        if verification:
            parts.append("\n🧪 **Verification:**")
            test_import = verification.get("test_import")
            expected_files = verification.get("expected_files", [])

            if test_import:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-c", test_import],
                        capture_output=True, text=True, timeout=30,
                    )
                    if proc.returncode == 0:
                        parts.append("  ✅ Import test passed")
                    else:
                        parts.append(
                            f"  ❌ Import test failed: {proc.stderr[:200]}"
                        )
                        verify_ok = False
                except Exception as exc:
                    parts.append(f"  ❌ Import test error: {exc}")
                    verify_ok = False

            if expected_files:
                all_present = True
                for ef in expected_files:
                    ef_path = os.path.join(working_dir, ef)
                    if not os.path.exists(ef_path):
                        parts.append(f"  ❌ Missing expected file: {ef}")
                        all_present = False
                        verify_ok = False
                if all_present:
                    parts.append("  ✅ All expected files present")

            # Run test_command if specified (Step 6.1 Step 4)
            test_command = verification.get("test_command")
            if test_command:
                try:
                    proc = subprocess.run(
                        test_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=working_dir,
                        timeout=120,
                    )
                    if proc.returncode == 0:
                        parts.append("  ✅ Test command passed")
                    else:
                        parts.append(
                            f"  ❌ Test command failed:\n```\n"
                            f"{(proc.stderr or proc.stdout or '')[:300]}\n```"
                        )
                        verify_ok = False
                except subprocess.TimeoutExpired:
                    parts.append("  ❌ Test command timed out (120s)")
                    verify_ok = False
                except Exception as exc:
                    parts.append(f"  ❌ Test command error: {exc}")
                    verify_ok = False

        elapsed = time.monotonic() - start
        success_count = sum(1 for sr in step_results if sr["success"])
        if verify_ok:
            parts.append(
                f"\n✅ **Backend {backend_name} built successfully**\n"
                f"Build steps completed: {success_count}/{total}\n"
                f"Total time: {elapsed:.1f}s"
            )
        else:
            parts.append(
                f"\n⚠️ **Build completed but verification failed**\n"
                f"Build steps: {success_count}/{total}\n"
                f"Total time: {elapsed:.1f}s"
            )

        # Update session context
        if session_context is not None:
            session_context.last_built_backend = profile_key

        return ToolResult(
            success=verify_ok,
            result={"step_results": step_results, "backend": profile_key},
            tool_name="backend_build",
            execution_time_ms=elapsed * 1000,
            message="\n".join(parts),
        )

    def _build_generic(
        self,
        backend_name: str,
        cwd: str,
        session_context: Optional[SessionContext],
        start: float,
    ) -> ToolResult:
        """Build a backend without a YAML profile (generic detection).

        Checks for ``setup.py``, ``pyproject.toml``, ``Makefile``, or
        ``CMakeLists.txt`` and runs the corresponding build command.
        """
        parts: List[str] = [f"## 🔨 Building **{backend_name}** (generic)\n"]

        # Detect build system
        build_cmd = None
        if os.path.isfile(os.path.join(cwd, "setup.py")):
            build_cmd = "pip install -e ."
        elif os.path.isfile(os.path.join(cwd, "pyproject.toml")):
            build_cmd = "pip install -e ."
        elif os.path.isfile(os.path.join(cwd, "Makefile")):
            build_cmd = "make"
        elif os.path.isfile(os.path.join(cwd, "CMakeLists.txt")):
            if os.name == "nt":
                build_cmd = (
                    "if not exist build mkdir build "
                    "&& cd build && cmake .. && cmake --build ."
                )
            else:
                build_cmd = (
                    "mkdir -p build && cd build && cmake .. && cmake --build ."
                )

        if not build_cmd:
            return ToolResult(
                success=False,
                error="No build system detected",
                tool_name="backend_build",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    f"❌ No build system detected for **{backend_name}** in `{cwd}`.\n"
                    "Looked for: setup.py, pyproject.toml, Makefile, CMakeLists.txt.\n"
                    "Please specify the build command explicitly."
                ),
            )

        parts.append(f"⚙️ Detected build command: `{build_cmd}`")
        try:
            proc = subprocess.run(
                build_cmd, shell=True,
                capture_output=True, text=True,
                cwd=cwd, timeout=600,
            )
            output = proc.stdout or proc.stderr or ""
            if proc.returncode == 0:
                parts.append("✅ Build completed successfully")
                parts.append(f"```\n{output[:500]}\n```")
                if session_context:
                    session_context.last_built_backend = backend_name
                return ToolResult(
                    success=True, result=output,
                    tool_name="backend_build",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message="\n".join(parts),
                )
            else:
                parts.append(
                    f"❌ Build failed\n```\n"
                    f"{(proc.stderr or proc.stdout or '')[:500]}\n```"
                )
                return ToolResult(
                    success=False, result=output,
                    tool_name="backend_build",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message="\n".join(parts),
                )
        except subprocess.TimeoutExpired:
            parts.append("❌ Build timed out (600s)")
            return ToolResult(
                success=False,
                result="\n".join(parts),
                tool_name="backend_build",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="\n".join(parts),
            )

    # ── Step 6.2: Backend Code Modification with Safety ───────────

    def _handle_backend_modify(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Modify backend source code with safety checks (Phase 6, Step 6.2).

        Flow: consent → checkpoint → preview diff → apply → verify.
        All modifications are reversible via ``undo``.
        """
        start = time.monotonic()

        # Extract entities
        backend_raw = _entity_value(intent, "backend", "name")
        file_path_raw = _entity_value(intent, "path", "filename", "file")
        old_content = _entity_value(intent, "old_content", "search", "find")
        new_content = _entity_value(
            intent, "new_content", "replace", "content", "text"
        )
        line_num_raw = _entity_value(intent, "line_number", "line")
        mod_desc = (
            _entity_value(intent, "description")
            or intent.raw_message
            or "Backend modification"
        )

        # Resolve file path
        file_path: Optional[str] = None
        if file_path_raw:
            file_path = resolve_path(file_path_raw, cwd)
        elif backend_raw:
            profile_key = self._resolve_backend_name(backend_raw)
            if profile_key:
                loader = self._get_profile_loader()
                profile = loader.get_backend_config(profile_key) if loader else None
                if profile and profile.get("directory"):
                    file_path = os.path.join(cwd, profile["directory"])

        if not file_path or not os.path.exists(file_path):
            return ToolResult(
                success=False,
                error="File not found or not specified",
                tool_name="backend_modify",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    "❌ Please specify the file to modify.\n"
                    "Example: 'modify backend.py in cirq backend — change X to Y'"
                ),
            )

        # If file_path is a directory, list contents for guidance
        if os.path.isdir(file_path):
            files = [
                f for f in os.listdir(file_path)[:20]
                if os.path.isfile(os.path.join(file_path, f))
            ]
            return ToolResult(
                success=False,
                error="Specified path is a directory, not a file",
                tool_name="backend_modify",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    f"📁 `{file_path}` is a directory. Please specify the exact file.\n"
                    f"Files:\n" + "\n".join(f"  • {f}" for f in files)
                ),
            )

        # Ensure we have modification content
        if not old_content and not new_content:
            return ToolResult(
                success=False,
                error="No modification content specified",
                tool_name="backend_modify",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    "❌ Please specify what to change.\n"
                    "Examples:\n"
                    "  • 'replace X with Y in backend.py'\n"
                    "  • 'add logging to line 42 of backend.py'\n"
                    "  • 'append a function to backend.py'"
                ),
            )

        # ── Step 1: Consent ──────────────────────────────────────
        consent_desc = f"Modify {os.path.basename(file_path)}: {mod_desc}"
        consent_payload: Any = consent_desc
        if _SAFETY_AVAILABLE:
            try:
                consent_payload = ConsentRequest(
                    id=f"backend_modify_{int(time.time())}",
                    consent_type=ConsentType.BACKEND_MODIFICATION,
                    operation="backend_modify",
                    description=consent_desc,
                    details={
                        "file": file_path,
                        "backend": backend_raw or "",
                        "modification": mod_desc,
                    },
                    tool_name="backend_modify",
                    risk_level="high",
                    reversible=True,
                )
            except Exception:
                pass

        if not self._request_consent(intent, consent_payload):
            return ToolResult(
                success=False,
                error="User declined backend modification",
                tool_name="backend_modify",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"⚠️ Backend modification declined: {consent_desc}",
            )

        # ── Step 2: Read current file ────────────────────────────
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                current_content = f.read()
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Cannot read file: {exc}",
                tool_name="backend_modify",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Cannot read `{file_path}`: {exc}",
            )

        # Compute new content based on available entities
        mod_type: str
        if old_content and new_content:
            # REPLACE
            if old_content not in current_content:
                return ToolResult(
                    success=False,
                    error="Old content not found in file",
                    tool_name="backend_modify",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message=(
                        f"❌ The specified text was not found in "
                        f"`{os.path.basename(file_path)}`.\n"
                        f"Searched for:\n```\n{old_content[:200]}\n```"
                    ),
                )
            modified_content = current_content.replace(old_content, new_content, 1)
            mod_type = "replace"
        elif new_content and not old_content:
            # INSERT at line or APPEND
            line_num = (
                int(line_num_raw)
                if line_num_raw and line_num_raw.isdigit()
                else None
            )
            if line_num is not None:
                lines = current_content.splitlines(True)
                insert_idx = min(line_num - 1, len(lines))
                lines.insert(insert_idx, new_content + "\n")
                modified_content = "".join(lines)
                mod_type = "insert"
            else:
                modified_content = (
                    current_content.rstrip() + "\n\n" + new_content + "\n"
                )
                mod_type = "append"
        else:
            # DELETE (old_content specified, no new_content)
            if old_content and old_content not in current_content:
                return ToolResult(
                    success=False,
                    error="Content to delete not found",
                    tool_name="backend_modify",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message=(
                        f"❌ Content to delete not found in "
                        f"`{os.path.basename(file_path)}`."
                    ),
                )
            modified_content = current_content.replace(old_content or "", "", 1)
            mod_type = "delete"

        # ── Step 3: Checkpoint ───────────────────────────────────
        cp_mgr = self._get_checkpoint_manager()
        checkpoint_id: Optional[str] = None
        if cp_mgr is not None:
            try:
                cp = cp_mgr.create_checkpoint(
                    operation="backend_modify",
                    description=f"{backend_raw or 'backend'}: {mod_desc}",
                    files=[file_path],
                    metadata={
                        "backend": backend_raw,
                        "mod_type": mod_type,
                    },
                )
                checkpoint_id = cp.id
            except Exception as exc:
                logger.warning("Checkpoint creation failed: %s", exc)

        # ── Step 4: Preview (diff) ───────────────────────────────
        diff_text = ""
        if _MOD_PREVIEW_AVAILABLE:
            try:
                gen = ModificationPreviewGenerator(context_lines=3)
                preview = gen.generate_preview(
                    file_path, current_content, modified_content,
                    modification_type=mod_type, description=mod_desc,
                )
                diff_text = gen.format_preview_text(preview, max_lines=60)
            except Exception:
                pass
        if not diff_text:
            # Fallback: simple unified diff
            diff_lines = list(difflib.unified_diff(
                current_content.splitlines(True),
                modified_content.splitlines(True),
                fromfile=f"a/{os.path.basename(file_path)}",
                tofile=f"b/{os.path.basename(file_path)}",
                lineterm="",
            ))
            diff_text = "\n".join(diff_lines[:60])

        # ── Step 5: Apply modification ───────────────────────────
        expected_hash = hashlib.sha256(modified_content.encode("utf-8")).hexdigest()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
        except Exception as exc:
            # Attempt rollback on write failure
            if cp_mgr and checkpoint_id:
                try:
                    cp_mgr.undo()
                except Exception:
                    pass
            return ToolResult(
                success=False,
                error=f"Write failed: {exc}",
                tool_name="backend_modify",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Failed to write `{file_path}`: {exc}",
            )

        # ── Step 6: Verify write (checksum comparison) ───────────
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                written_content = f.read()
            actual_hash = hashlib.sha256(written_content.encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                logger.warning(
                    "Checksum mismatch after write: expected %s, got %s",
                    expected_hash, actual_hash,
                )
                # Attempt rollback on verification failure
                if cp_mgr and checkpoint_id:
                    try:
                        cp_mgr.undo()
                    except Exception:
                        pass
                return ToolResult(
                    success=False,
                    error="Post-write verification failed (checksum mismatch)",
                    tool_name="backend_modify",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message=(
                        f"❌ Verification failed: written content does not match "
                        f"expected content for `{file_path}`."
                    ),
                )
        except Exception as exc:
            logger.warning("Post-write verification read failed: %s", exc)

        # Complete checkpoint
        if cp_mgr and checkpoint_id:
            try:
                cp_mgr.complete_checkpoint(
                    checkpoint_id, state_after={"modified": True}
                )
            except Exception:
                pass

        # Update session context
        if session_context is not None:
            if backend_raw and checkpoint_id:
                session_context.backend_checkpoints[backend_raw] = checkpoint_id
            session_context.last_modified_files = [file_path]

        elapsed = (time.monotonic() - start) * 1000
        message = (
            f"✅ **Backend code modified** ({mod_type})\n\n"
            f"**File:** `{file_path}`\n"
            f"**Change:** {mod_desc}\n"
        )
        if checkpoint_id:
            message += (
                f"**Checkpoint:** `{checkpoint_id}` (use 'undo' to revert)\n"
            )
        if diff_text:
            message += f"\n**Diff:**\n```diff\n{diff_text}\n```"

        return ToolResult(
            success=True,
            result={
                "file": file_path,
                "mod_type": mod_type,
                "checkpoint_id": checkpoint_id,
                "diff": diff_text[:2000],
            },
            tool_name="backend_modify",
            execution_time_ms=elapsed,
            message=message,
        )

    # ── Step 6.3: Undo / Redo / Rollback ─────────────────────────

    def _handle_undo(
        self,
        intent: Intent,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Undo the last operation via ``CheckpointManager`` (Phase 6, Step 6.3)."""
        start = time.monotonic()
        cp_mgr = self._get_checkpoint_manager()
        if cp_mgr is None:
            return ToolResult(
                success=False,
                error="CheckpointManager unavailable",
                tool_name="undo",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Undo system is not available.",
            )

        state = cp_mgr.get_undo_redo_state()
        if not state.can_undo:
            return ToolResult(
                success=True,
                result=None,
                tool_name="undo",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="ℹ️ Nothing to undo. No operations have been recorded.",
            )

        desc = state.undo_description or "last operation"

        # Request consent before undoing (spec requirement)
        consent_msg = f"Undo {desc}?"
        consent_payload: Any = consent_msg
        if _SAFETY_AVAILABLE:
            try:
                consent_payload = ConsentRequest(
                    id=f"undo_{int(time.time())}",
                    consent_type=ConsentType.BACKEND_MODIFICATION,
                    operation="undo",
                    description=consent_msg,
                    details={"operation": desc},
                    tool_name="undo",
                    risk_level="medium",
                    reversible=True,
                )
            except Exception:
                pass

        if not self._request_consent(intent, consent_payload):
            return ToolResult(
                success=False,
                error="User declined undo",
                tool_name="undo",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"⚠️ Undo declined: {desc}",
            )

        result = cp_mgr.undo()
        elapsed = (time.monotonic() - start) * 1000

        if result.success:
            restored = (
                ", ".join(result.restored_files)
                if result.restored_files else "none"
            )
            return ToolResult(
                success=True,
                result={
                    "checkpoint_id": result.checkpoint_id,
                    "restored_files": result.restored_files,
                },
                tool_name="undo",
                execution_time_ms=elapsed,
                message=(
                    f"↩️ **Undo complete:** {desc}\n"
                    f"Restored files: {restored}\n"
                    f"{result.message}"
                ),
            )
        else:
            failed = (
                ", ".join(f"{f[0]}: {f[1]}" for f in result.failed_files)
                if result.failed_files else ""
            )
            return ToolResult(
                success=False,
                error=result.message,
                tool_name="undo",
                execution_time_ms=elapsed,
                message=(
                    f"❌ Undo failed: {result.message}\n"
                    + (f"Failed files: {failed}" if failed else "")
                ),
            )

    def _handle_redo(
        self,
        intent: Intent,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Redo the last undone operation (Phase 6, Step 6.3)."""
        start = time.monotonic()
        cp_mgr = self._get_checkpoint_manager()
        if cp_mgr is None:
            return ToolResult(
                success=False,
                error="CheckpointManager unavailable",
                tool_name="redo",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Redo system is not available.",
            )

        state = cp_mgr.get_undo_redo_state()
        if not state.can_redo:
            return ToolResult(
                success=True,
                result=None,
                tool_name="redo",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="ℹ️ Nothing to redo.",
            )

        desc = state.redo_description or "last undone operation"

        # Request consent before redoing (spec requirement)
        redo_consent_msg = f"Redo {desc}?"
        redo_consent_payload: Any = redo_consent_msg
        if _SAFETY_AVAILABLE:
            try:
                redo_consent_payload = ConsentRequest(
                    id=f"redo_{int(time.time())}",
                    consent_type=ConsentType.BACKEND_MODIFICATION,
                    operation="redo",
                    description=redo_consent_msg,
                    details={"operation": desc},
                    tool_name="redo",
                    risk_level="medium",
                    reversible=True,
                )
            except Exception:
                pass

        if not self._request_consent(intent, redo_consent_payload):
            return ToolResult(
                success=False,
                error="User declined redo",
                tool_name="redo",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"⚠️ Redo declined: {desc}",
            )

        result = cp_mgr.redo()
        elapsed = (time.monotonic() - start) * 1000

        if result.success:
            restored = (
                ", ".join(result.restored_files)
                if result.restored_files else "none"
            )
            return ToolResult(
                success=True,
                result={
                    "checkpoint_id": result.checkpoint_id,
                    "restored_files": result.restored_files,
                },
                tool_name="redo",
                execution_time_ms=elapsed,
                message=(
                    f"↪️ **Redo complete:** {desc}\n"
                    f"Restored files: {restored}\n"
                    f"{result.message}"
                ),
            )
        else:
            return ToolResult(
                success=False,
                error=result.message,
                tool_name="redo",
                execution_time_ms=elapsed,
                message=f"❌ Redo failed: {result.message}",
            )

    # ── Step 6.4: Backend Testing and Verification ────────────────

    def _handle_backend_test(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Test/verify a built backend (Phase 6, Step 6.4).

        Runs verification steps from the YAML profile:
        1. Import test (``python -c "{test_import}"``)
        2. Test command (test suite)
        3. Expected files existence check
        """
        start = time.monotonic()
        backend_raw = _entity_value(intent, "backend", "name") or ""
        profile_key = self._resolve_backend_name(backend_raw)

        # Fallback to last built backend
        if not profile_key and session_context:
            if session_context.last_built_backend:
                profile_key = self._resolve_backend_name(
                    session_context.last_built_backend
                )

        if not profile_key:
            return ToolResult(
                success=False,
                error="No backend specified for testing",
                tool_name="backend_test",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    "❌ Please specify which backend to test.\n"
                    "Example: 'test cirq backend', 'verify qiskit build'"
                ),
            )

        loader = self._get_profile_loader()
        if loader is None:
            return ToolResult(
                success=False,
                error="Build profiles unavailable",
                tool_name="backend_test",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Build profile loader is not available.",
            )

        profile = loader.get_backend_config(profile_key)
        if not profile:
            return ToolResult(
                success=False,
                error=f"No profile for '{profile_key}'",
                tool_name="backend_test",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ No build profile found for **{profile_key}**.",
            )

        backend_name = profile.get("name", profile_key)
        backend_dir = profile.get("directory", ".")
        if not os.path.isabs(backend_dir):
            backend_dir = os.path.join(cwd, backend_dir)

        verification = profile.get("verification", {})
        if not verification:
            return ToolResult(
                success=False,
                error="No verification section in profile",
                tool_name="backend_test",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"⚠️ No verification steps defined for **{backend_name}**.",
            )

        parts: List[str] = [
            f"## 🧪 Backend Test Results: **{backend_name}**\n"
        ]
        all_passed = True

        # ── Import test ──────────────────────────────────────────
        test_import = verification.get("test_import")
        if test_import:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", test_import],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.returncode == 0:
                    parts.append("✅ **Import test:** Passed")
                else:
                    parts.append(
                        f"❌ **Import test:** Failed — {proc.stderr[:200]}"
                    )
                    all_passed = False
            except Exception as exc:
                parts.append(f"❌ **Import test:** Error — {exc}")
                all_passed = False
        else:
            parts.append("⏭️ **Import test:** Not configured")

        # ── Test command ─────────────────────────────────────────
        test_cmd = verification.get("test_command")
        if test_cmd:
            try:
                proc = subprocess.run(
                    test_cmd, shell=True,
                    capture_output=True, text=True,
                    cwd=backend_dir if os.path.isdir(backend_dir) else cwd,
                    timeout=300,
                )
                output = proc.stdout or proc.stderr or ""
                if proc.returncode == 0:
                    passed_match = re.search(r'(\d+)\s+passed', output)
                    failed_match = re.search(r'(\d+)\s+failed', output)
                    if passed_match:
                        p = passed_match.group(1)
                        f_count = (
                            failed_match.group(1) if failed_match else "0"
                        )
                        parts.append(
                            f"✅ **Test suite:** {p} passed, {f_count} failed"
                        )
                    else:
                        parts.append("✅ **Test suite:** Passed")
                else:
                    parts.append(
                        f"❌ **Test suite:** Failed\n```\n{output[:300]}\n```"
                    )
                    all_passed = False
            except subprocess.TimeoutExpired:
                parts.append("❌ **Test suite:** Timed out (300s)")
                all_passed = False
            except Exception as exc:
                parts.append(f"❌ **Test suite:** Error — {exc}")
                all_passed = False
        else:
            parts.append("⏭️ **Test suite:** Not configured")

        # ── Expected files ───────────────────────────────────────
        expected_files = verification.get("expected_files", [])
        if expected_files:
            missing: List[str] = []
            for ef in expected_files:
                ef_path = (
                    os.path.join(backend_dir, ef)
                    if not os.path.isabs(ef) else ef
                )
                if not os.path.exists(ef_path):
                    missing.append(ef)
            if missing:
                parts.append(
                    f"❌ **Expected files:** Missing: {', '.join(missing)}"
                )
                all_passed = False
            else:
                parts.append(
                    f"✅ **Expected files:** All {len(expected_files)} present"
                )
        else:
            parts.append("⏭️ **Expected files:** Not configured")

        # Overall
        parts.append(
            f"\n**Overall:** {'✅ PASS' if all_passed else '❌ FAIL'}"
        )

        elapsed = (time.monotonic() - start) * 1000
        return ToolResult(
            success=all_passed,
            result={"backend": profile_key, "passed": all_passed},
            tool_name="backend_test",
            execution_time_ms=elapsed,
            message="\n".join(parts),
        )

    # ── Step 6.5: Backend Configuration ───────────────────────────

    def _handle_backend_configure(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Configure Proxima to use a specific backend (Phase 6, Step 6.5).

        Updates ``configs/default.yaml`` with the chosen backend,
        creates a checkpoint, and verifies the backend import.
        """
        start = time.monotonic()
        backend_raw = _entity_value(intent, "backend", "name") or ""
        profile_key = self._resolve_backend_name(backend_raw)

        # Fallback to last built backend
        if not profile_key and session_context:
            if session_context.last_built_backend:
                profile_key = self._resolve_backend_name(
                    session_context.last_built_backend
                )
                backend_raw = session_context.last_built_backend

        if not profile_key:
            return ToolResult(
                success=False,
                error="No backend specified",
                tool_name="backend_configure",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    "❌ Please specify which backend to configure.\n"
                    "Example: 'configure proxima to use cirq', "
                    "'set qiskit as default backend'"
                ),
            )

        # Load profile for metadata
        backend_name = profile_key
        backend_path = ""
        profile: Optional[Dict[str, Any]] = None
        loader = self._get_profile_loader()
        if loader:
            profile = loader.get_backend_config(profile_key)
            if profile:
                backend_name = profile.get("name", profile_key)
                backend_path = profile.get("directory", "")

        # Locate configs/default.yaml
        config_path = os.path.join(cwd, "configs", "default.yaml")
        if not os.path.isfile(config_path):
            for candidate in [
                os.path.join(cwd, "..", "configs", "default.yaml"),
                os.path.join(
                    os.path.dirname(os.path.dirname(cwd)),
                    "configs", "default.yaml",
                ),
            ]:
                if os.path.isfile(candidate):
                    config_path = os.path.normpath(candidate)
                    break

        # Read current config
        config_data: Dict[str, Any] = {}
        raw_fallback = False
        if os.path.isfile(config_path):
            try:
                if _YAML_AVAILABLE:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = _yaml.safe_load(f) or {}
                else:
                    raw_fallback = True
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = {"_raw": f.read()}
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Cannot read config: {exc}",
                    tool_name="backend_configure",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message=f"❌ Cannot read `{config_path}`: {exc}",
                )

        # ── Checkpoint before writing ────────────────────────────
        cp_mgr = self._get_checkpoint_manager()
        checkpoint_id: Optional[str] = None
        if cp_mgr and os.path.isfile(config_path):
            try:
                cp = cp_mgr.create_checkpoint(
                    operation="backend_configure",
                    description=f"Configure Proxima to use {backend_name}",
                    files=[config_path],
                )
                checkpoint_id = cp.id
            except Exception:
                pass

        # ── Update configuration ─────────────────────────────────
        if _YAML_AVAILABLE and not raw_fallback:
            backends_section = config_data.setdefault("backends", {})
            backends_section["default_backend"] = profile_key
            if backend_path:
                backends_section["backend_module_path"] = backend_path
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    _yaml.safe_dump(
                        config_data, f,
                        default_flow_style=False, sort_keys=False,
                    )
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Failed to write config: {exc}",
                    tool_name="backend_configure",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message=f"❌ Failed to update `{config_path}`: {exc}",
                )
        else:
            # Fallback: regex-based text replacement
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    text = f.read()
                text = re.sub(
                    r'(default_backend:\s*).*',
                    rf'\g<1>{profile_key}',
                    text,
                )
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Config update failed: {exc}",
                    tool_name="backend_configure",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message=f"❌ Failed to update config: {exc}",
                )

        # Complete checkpoint
        if cp_mgr and checkpoint_id:
            try:
                cp_mgr.complete_checkpoint(
                    checkpoint_id, state_after={"backend": profile_key}
                )
            except Exception:
                pass

        # ── Verify the configuration ─────────────────────────────
        verify_msg = ""
        if profile:
            test_import = (
                (profile.get("verification") or {}).get("test_import")
            )
            if test_import:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-c", test_import],
                        capture_output=True, text=True, timeout=30,
                    )
                    if proc.returncode == 0:
                        verify_msg = "\n✅ Backend import verification passed"
                    else:
                        verify_msg = (
                            "\n⚠️ Backend import verification failed: "
                            f"{proc.stderr[:200]}"
                        )
                except Exception:
                    verify_msg = "\n⚠️ Could not verify backend import"

        elapsed = (time.monotonic() - start) * 1000
        return ToolResult(
            success=True,
            result={
                "backend": profile_key,
                "config_path": config_path,
                "checkpoint_id": checkpoint_id,
            },
            tool_name="backend_configure",
            execution_time_ms=elapsed,
            message=(
                f"✅ **Proxima configured to use {backend_name}**\n"
                f"Configuration file: `{config_path}`\n"
                f"Default backend: `{profile_key}`"
                + (f"\nBackend path: `{backend_path}`" if backend_path else "")
                + verify_msg
            ),
        )

    # ── Step 6.6: Backend Listing ─────────────────────────────────

    def _handle_backend_list(
        self,
        intent: Intent,
        cwd: str,
    ) -> ToolResult:
        """List available backend profiles (Phase 6, Step 6.6).

        Reads ``configs/backend_build_profiles.yaml`` and formats
        a table of all backends and build profiles.
        """
        start = time.monotonic()

        loader = self._get_profile_loader()
        if loader is None:
            return ToolResult(
                success=False,
                error="Build profiles unavailable",
                tool_name="backend_list",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Backend build profile loader is not available.",
            )

        config = loader.load()
        backends = config.get("backends", {})

        if not backends:
            return ToolResult(
                success=True,
                result=[],
                tool_name="backend_list",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="ℹ️ No backend profiles are configured.",
            )

        # Build table
        rows: List[str] = [
            "## 📋 Available Backend Profiles\n",
            "| # | Key | Name | Description | GPU |",
            "| --- | --- | --- | --- | --- |",
        ]
        for idx, (key, cfg) in enumerate(backends.items(), 1):
            name = cfg.get("name", key)
            desc = cfg.get("description", "—")
            gpu = "Yes" if cfg.get("gpu_required", False) else "No"
            rows.append(f"| {idx} | {key} | {name} | {desc} | {gpu} |")

        # Build profiles (preset selections of backends)
        profiles = config.get("build_profiles", {})
        if profiles:
            rows.append("\n**Build Profiles:**")
            for pname, pval in profiles.items():
                blist = ", ".join(pval.get("backends", []))
                rows.append(f"  • **{pname}**: {blist}")

        table = "\n".join(rows)
        elapsed = (time.monotonic() - start) * 1000
        return ToolResult(
            success=True,
            result={
                "backends": list(backends.keys()),
                "profiles": list(profiles.keys()),
            },
            tool_name="backend_list",
            execution_time_ms=elapsed,
            message=table,
        )

    def _fallback_execute(
        self, intent: Intent, args: Dict[str, Any], cwd: str
    ) -> ToolResult:
        """Fallback when the tool registry is unavailable.

        Attempts basic shell execution for ``run_command`` targets.
        Safety checks are applied before execution.
        """
        cmd = args.get("command")
        if not cmd:
            return ToolResult(
                success=False,
                error="No command to execute and tool registry unavailable",
                message="❌ Cannot execute: tool registry unavailable.",
            )

        # Safety pre-screening (mirrors the dispatch() path)
        if is_blocked(cmd):
            return ToolResult(
                success=False,
                error=f"Command blocked by safety policy: {cmd}",
                message="🚫 This command is blocked for safety reasons.",
            )
        if is_dangerous(cmd):
            if not self._request_consent(intent, cmd):
                return ToolResult(
                    success=False,
                    error="User declined dangerous command",
                    message="⚠️ Dangerous command declined by user.",
                )

        try:
            working_dir = args.get("working_directory", cwd)
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=args.get("timeout", 60),
            )
            success = result.returncode == 0
            output = result.stdout or result.stderr
            return ToolResult(
                success=success,
                result=output,
                tool_name="run_command (fallback)",
                message=output[:500] if output else ("✅ Done" if success else "❌ Failed"),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                message=f"❌ Fallback execution failed: {exc}",
            )

    # ═══════════════════════════════════════════════════════════════
    # Phase 7: Administrative Access and Safe Privilege Escalation
    # ═══════════════════════════════════════════════════════════════

    def _get_admin_handler(self) -> Optional["AdminPrivilegeHandler"]:
        """Return a cached ``AdminPrivilegeHandler`` instance.

        The handler is created lazily on first access.  If the
        ``admin_privilege_handler`` module is not importable, returns
        ``None``.
        """
        if not _ADMIN_HANDLER_AVAILABLE:
            return None
        if self._admin_handler is None:
            audit_path = os.path.join(
                os.path.expanduser("~"), ".proxima", "logs",
            )
            os.makedirs(audit_path, exist_ok=True)
            self._admin_handler = AdminPrivilegeHandler(
                audit_log_path=Path(os.path.join(audit_path, "admin_operations.log")),
                consent_callback=self._admin_consent_bridge,
            )
        return self._admin_handler

    # ── Consent-building helper (eliminates duplication) ──────────

    @staticmethod
    def _build_admin_consent_request(
        request_id: str,
        operation: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        risk_level: str = "critical",
        reversible: bool = False,
    ) -> Any:
        """Build a ``ConsentRequest`` for admin operations.

        Returns the ``ConsentRequest`` if ``safety`` is available,
        otherwise returns the plain *description* string.
        """
        if _SAFETY_AVAILABLE:
            try:
                return ConsentRequest(
                    id=request_id,
                    consent_type=ConsentType.ADMIN_ACCESS,
                    operation=operation,
                    description=description,
                    details=details or {},
                    tool_name="admin_elevate",
                    risk_level=risk_level,
                    reversible=reversible,
                )
            except Exception:
                pass
        return description

    def _admin_consent_bridge(self, operation: "PrivilegedOperation") -> bool:
        """Translate ``AdminPrivilegeHandler`` consent into bridge consent.

        The ``AdminPrivilegeHandler`` calls this callback with a
        ``PrivilegedOperation`` dataclass.  We build a ``ConsentRequest``
        and forward to our normal consent pipeline.
        """
        consent_desc = (
            f"🔐 Administrative Access Required\n\n"
            f"Operation: {operation.description}\n"
            f"Category: {operation.category.name}\n"
            f"Risk: {'CRITICAL' if operation.risk_level >= 4 else 'HIGH'}\n\n"
            f"This operation requires elevated privileges."
        )
        if os.name == "nt":
            consent_desc += " On Windows, a UAC prompt will appear."
        else:
            consent_desc += (
                " On Linux/macOS, you will be prompted for your password."
            )
        consent_desc += "\n\nProceed with privilege escalation? (yes/no)"

        consent_payload = self._build_admin_consent_request(
            request_id=f"admin_{int(time.time())}",
            operation="admin_elevate",
            description=consent_desc,
            details={
                "category": operation.category.name,
                "command": operation.command or "",
                "risk_level": operation.risk_level,
            },
        )

        if self._consent_callback is None:
            return False
        try:
            return bool(self._consent_callback(consent_payload))
        except Exception:
            # Fallback: try with plain string if ConsentRequest is rejected
            try:
                return bool(self._consent_callback(consent_desc))
            except Exception:
                return False

    # ── Step 7.1: Admin detection ─────────────────────────────────

    @staticmethod
    def _detect_admin_required(
        command: str,
    ) -> Optional[Tuple[str, str]]:
        """Check if *command* requires administrative privileges.

        Returns ``(category_name, reason)`` if admin is needed, else
        ``None``.  The check is purely pattern-based and independent of
        the actual OS privilege level.
        """
        if not command:
            return None
        for pattern, category_name, reason in _ADMIN_DETECTION_PATTERNS:
            if pattern.search(command):
                return (category_name, reason)
        return None

    @staticmethod
    def _is_venv_active() -> bool:
        """Return ``True`` if the current Python interpreter is in a venv."""
        return (
            hasattr(sys, "real_prefix")
            or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        )

    @staticmethod
    def _check_path_is_protected(file_path: str) -> bool:
        """Return ``True`` if *file_path* is inside a protected system directory."""
        norm = os.path.normpath(file_path).replace("\\", "/")
        if os.name == "nt":
            for pd in _PROTECTED_DIRS_WINDOWS:
                if norm.lower().startswith(pd.lower().replace("\\", "/")):
                    return True
        else:
            for pd in _PROTECTED_DIRS_UNIX:
                if norm.startswith(pd):
                    return True
        return False

    def _check_security_hard_blocks(
        self, command: str,
    ) -> Optional[ToolResult]:
        """Enforce hard security restrictions (Step 7.4).

        Returns a ``ToolResult`` (blocked) if the command violates a
        non-overridable security constraint, else ``None``.
        """
        for pattern, message in _SECURITY_BLOCK_PATTERNS:
            if pattern.search(command):
                return ToolResult(
                    success=False,
                    error=f"Security violation: {message}",
                    tool_name="admin_elevate",
                    message=f"🚫 **Blocked:** {message}",
                )
        return None

    def _check_security_sensitive(
        self,
        intent: Intent,
        command: str,
    ) -> Optional[ToolResult]:
        """Check for security-sensitive operations requiring critical consent.

        These operations (e.g. disabling firewalls) are allowed but only
        with explicit critical-level consent AND a clear warning.
        Returns a ``ToolResult`` (denied) if user declines.
        """
        for pattern, description in _SECURITY_SENSITIVE_RE:
            if pattern.search(command):
                warning_msg = (
                    f"⚠️ **CRITICAL SECURITY WARNING**\n\n"
                    f"You are about to: **{description}**\n"
                    f"Command: `{command}`\n\n"
                    f"This will reduce your system's security.\n"
                    f"Are you absolutely sure? (yes/no)"
                )
                consent_payload = self._build_admin_consent_request(
                    request_id=f"security_sensitive_{int(time.time())}",
                    operation="disable_security",
                    description=warning_msg,
                    details={"command": command},
                )
                if not self._request_consent(intent, consent_payload):
                    return ToolResult(
                        success=False,
                        error="User declined security-sensitive operation",
                        tool_name="admin_elevate",
                        message=f"⚠️ Security-sensitive operation declined: {description}",
                    )
        return None

    # ── Step 7.4: Audit logging ───────────────────────────────────

    @staticmethod
    def _log_admin_audit(
        command: str,
        description: str,
        consent_decision: str,
        result_status: str,
        files_affected: Optional[List[str]] = None,
    ) -> None:
        """Append an entry to ``~/.proxima/logs/admin_audit.log``.

        The log is append-only (ISO 8601 timestamps).
        """
        log_dir = os.path.join(os.path.expanduser("~"), ".proxima", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "admin_audit.log")

        entry = (
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"user={getpass.getuser()} "
            f"operation=\"{description}\" "
            f"command=\"{command}\" "
            f"consent={consent_decision} "
            f"result={result_status}"
        )
        if files_affected:
            entry += f" files_affected={','.join(files_affected)}"
        entry += "\n"

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as exc:
            logger.warning("Failed to write admin audit log: %s", exc)

    # ── Step 7.2 & 7.3: Escalation and execution ─────────────────

    def _dispatch_phase7(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> Optional[ToolResult]:
        """Route Phase 7 intents (``ADMIN_ELEVATE``) to admin handlers.

        Returns ``None`` if the intent is not Phase 7.
        """
        if intent.intent_type not in _PHASE7_INTENTS:
            return None
        return self._handle_admin_elevate(intent, cwd, session_context)

    def _handle_admin_elevate(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle explicit admin elevation requests (Phase 7, Step 7.2).

        Detects the command, checks security constraints, requests
        consent, and executes via ``AdminPrivilegeHandler``.
        Routes to category-specific handlers where appropriate.
        """
        start = time.monotonic()

        # Extract the command to elevate
        command = (
            _entity_value(intent, "command", "cmd", "text")
            or intent.raw_message
            or ""
        )
        # Strip common preamble ("run as admin: ...")
        for prefix in ("run as admin", "sudo", "elevate", "admin"):
            if command.lower().startswith(prefix):
                command = command[len(prefix):].lstrip(": ").strip()
                break

        if not command:
            return ToolResult(
                success=False,
                error="No command to elevate",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    "❌ No command specified for admin elevation.\n"
                    "Please specify what operation needs admin privileges."
                ),
            )

        # ── Common pre-checks (shared with _intercept) ───────────
        pre = self._admin_pre_checks(command, intent)
        if pre is not None:
            return pre

        # ── Get current privilege level ──────────────────────────
        handler = self._get_admin_handler()
        if handler is None:
            return ToolResult(
                success=False,
                error="AdminPrivilegeHandler not available",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Admin privilege handler module is not available.",
            )

        priv_info = handler.get_privilege_info()

        # If already elevated or system, execute directly
        if priv_info.level in (PrivilegeLevel.ELEVATED, PrivilegeLevel.SYSTEM):
            return self._execute_admin_command(
                intent, command, cwd, handler, priv_info,
                session_context, start, already_elevated=True,
            )

        # ── Detect category & reason ─────────────────────────────
        detection = self._detect_admin_required(command)
        category_name = detection[0] if detection else "SYSTEM_CONFIG"
        reason = (
            detection[1] if detection
            else "This command may require elevated privileges"
        )

        # ── Route to category-specific handlers (Step 7.3) ───────
        if category_name == "SERVICE_CONTROL":
            return self._handle_service_control(intent, cwd, session_context)
        if category_name == "PERMISSION":
            file_path = _entity_value(intent, "path", "file", "target") or ""
            return self._handle_permission_change(
                intent, file_path, command, cwd, session_context,
            )
        if category_name == "SYSTEM_CONFIG" and re.search(
            r"\bcuda|nvidia|gpu\b", command, re.IGNORECASE,
        ):
            return self._handle_cuda_gpu_setup(
                intent, command, cwd, session_context,
            )

        # ── Request consent (Step 7.2, Step 1) ───────────────────
        consent_desc = self._format_admin_consent_desc(
            command, category_name, reason,
        )
        consent_payload = self._build_admin_consent_request(
            request_id=f"admin_elevate_{int(time.time())}",
            operation="admin_elevate",
            description=consent_desc,
            details={
                "command": command,
                "category": category_name,
                "reason": reason,
                "current_level": priv_info.level.value,
                "elevation_method": priv_info.elevation_method.value,
            },
        )

        if not self._request_consent(intent, consent_payload):
            self._log_admin_audit(command, reason, "DENIED", "USER_DECLINED")
            return ToolResult(
                success=False,
                error="User declined admin elevation",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"⚠️ Admin elevation declined for: `{command}`",
            )

        # ── Execute elevated ─────────────────────────────────────
        return self._execute_admin_command(
            intent, command, cwd, handler, priv_info,
            session_context, start, already_elevated=False,
        )

    def _execute_admin_command(
        self,
        intent: Intent,
        command: str,
        cwd: str,
        handler: "AdminPrivilegeHandler",
        priv_info: Any,
        session_context: Optional[SessionContext],
        start: float,
        already_elevated: bool = False,
    ) -> ToolResult:
        """Execute a command with elevated privileges via ``AdminPrivilegeHandler``.

        Handles both the already-elevated and needs-escalation paths.
        """
        parts: List[str] = []
        if already_elevated:
            parts.append(
                f"✅ Already running with **{priv_info.level.value}** "
                f"privileges — executing directly."
            )
        else:
            parts.append(
                f"🔐 Escalating to **{priv_info.elevation_method.value.upper()}** "
                f"on **{priv_info.platform}**…"
            )

        # Increment escalation counter (only when actually escalating)
        if not already_elevated:
            self._admin_escalation_count += 1

        try:
            elevation_result: ElevationResult = handler.execute_elevated(
                command,
                working_dir=Path(cwd) if cwd else None,
                timeout=300,
                require_consent=False,  # consent already obtained above
            )
        except Exception as exc:
            self._log_admin_audit(
                command, "admin_elevate", "APPROVED", "ERROR",
            )
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Admin execution error: {exc}",
            )

        elapsed = (time.monotonic() - start) * 1000

        if elevation_result.success:
            output = elevation_result.output or ""
            parts.append(f"✅ Command completed successfully")
            if output:
                parts.append(f"\n**Output:**\n```\n{output[:1000]}\n```")

            self._log_admin_audit(
                command, "admin_elevate", "APPROVED", "SUCCESS",
            )

            # Update session context
            if session_context is not None:
                session_context.last_operation_result = output[:500]

            return ToolResult(
                success=True,
                result={
                    "command": command,
                    "output": output,
                    "method": elevation_result.method.value,
                    "return_code": elevation_result.return_code,
                },
                tool_name="admin_elevate",
                execution_time_ms=elapsed,
                message="\n".join(parts),
            )
        else:
            error_msg = elevation_result.error or "Unknown error"
            parts.append(f"❌ **Elevated command failed:**\n{error_msg}")
            if elevation_result.output:
                parts.append(
                    f"\n**Output:**\n```\n{elevation_result.output[:500]}\n```"
                )

            self._log_admin_audit(
                command, "admin_elevate", "APPROVED", "FAILURE",
            )

            return ToolResult(
                success=False,
                error=error_msg,
                tool_name="admin_elevate",
                execution_time_ms=elapsed,
                message="\n".join(parts),
            )

    # ── Shared pre-checks and helpers ───────────────────────────────

    def _admin_pre_checks(
        self,
        command: str,
        intent: Intent,
    ) -> Optional[ToolResult]:
        """Run hard-block, escalation-limit, and security-sensitive checks.

        Returns a ``ToolResult`` if the command should be rejected,
        otherwise ``None`` (safe to proceed).
        """
        # Hard security blocks (Step 7.4)
        block_result = self._check_security_hard_blocks(command)
        if block_result is not None:
            self._log_admin_audit(command, "admin_elevate", "BLOCKED", "BLOCKED")
            return block_result

        # Escalation limit (Step 7.4)
        if self._admin_escalation_count >= _MAX_ADMIN_ESCALATIONS_PER_SESSION:
            self._log_admin_audit(
                command, "admin_elevate", "BLOCKED", "LIMIT_EXCEEDED",
            )
            return ToolResult(
                success=False,
                error="Maximum admin escalations exceeded",
                tool_name="admin_elevate",
                message=(
                    f"🚫 Maximum admin escalations reached "
                    f"({_MAX_ADMIN_ESCALATIONS_PER_SESSION} per session).\n"
                    f"Please restart the session to continue with "
                    f"elevated operations."
                ),
            )

        # Security-sensitive check
        sens_result = self._check_security_sensitive(intent, command)
        if sens_result is not None:
            self._log_admin_audit(
                command, "admin_elevate", "DENIED", "SECURITY_SENSITIVE",
            )
            return sens_result

        return None

    @staticmethod
    def _format_admin_consent_desc(
        command: str,
        category_name: str,
        reason: str,
    ) -> str:
        """Build a standard admin-consent description string."""
        desc = (
            f"🔐 **Administrative Access Required**\n\n"
            f"**Operation:** `{command}`\n"
            f"**Category:** {category_name}\n"
            f"**Reason:** {reason}\n"
            f"**Risk:** HIGH\n\n"
            f"This operation requires elevated privileges."
        )
        if os.name == "nt":
            desc += " On Windows, a UAC prompt will appear."
        else:
            desc += (
                " On Linux/macOS, you will be prompted for your password."
            )
        desc += "\n\nProceed with privilege escalation? (yes/no)"
        return desc

    # ── Step 7.1: Pre-dispatch admin interception ─────────────────

    def _intercept_admin_for_command(
        self,
        intent: Intent,
        command: str,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> Optional[ToolResult]:
        """Intercept commands that require admin privileges.

        Called from ``dispatch()`` for RUN_COMMAND/INSTALL_DEPENDENCY
        intents before normal execution.

        If the command requires elevation and the user is not already
        elevated, routes through the Phase 7 escalation flow.
        Returns ``None`` if no admin is needed (proceed normally).
        """
        # ── Pattern-based detection ──────────────────────────────
        detection = self._detect_admin_required(command)

        # ── Path-based detection (protected system dirs) ─────────
        if detection is None:
            path_entity = _entity_value(intent, "path", "file", "target")
            if path_entity and self._check_path_is_protected(path_entity):
                detection = (
                    "FILE_SYSTEM",
                    f"Target path '{path_entity}' is inside a protected "
                    f"system directory",
                )

        if detection is None:
            return None

        category_name, reason = detection

        # Check current privileges
        handler = self._get_admin_handler()
        if handler is None:
            return None

        priv_info = handler.get_privilege_info()
        if priv_info.level in (PrivilegeLevel.ELEVATED, PrivilegeLevel.SYSTEM):
            return None

        # PACKAGE_INSTALL inside a venv — no admin needed
        if category_name == "PACKAGE_INSTALL" and self._is_venv_active():
            return None

        # Common pre-checks (hard blocks, escalation limit, sensitive)
        pre = self._admin_pre_checks(command, intent)
        if pre is not None:
            return pre

        # ── PACKAGE_INSTALL: suggest venv first (Step 7.3) ───────
        # This consent replaces the general escalation consent — if the
        # user approves, we proceed directly to elevation (no double ask).
        if category_name == "PACKAGE_INSTALL" and not self._is_venv_active():
            venv_suggest = (
                f"⚠️ **System-level package installation detected**\n\n"
                f"Command: `{command}`\n\n"
                f"Your Python environment is system-wide (not a virtual "
                f"environment).\nInstalling packages system-wide requires "
                f"admin privileges and may affect other applications.\n\n"
                f"**Recommendation:** Create a virtual environment first:\n"
                f"```\npython -m venv .venv\n```\n\n"
                f"Would you like to proceed with admin-level installation "
                f"anyway? (yes/no)"
            )
            consent_payload = self._build_admin_consent_request(
                request_id=f"pkg_admin_{int(time.time())}",
                operation="system_package_install",
                description=venv_suggest,
                details={"command": command, "reason": reason},
                risk_level="high",
            )
            if not self._request_consent(intent, consent_payload):
                self._log_admin_audit(
                    command, "system_package_install",
                    "DENIED", "USER_DECLINED",
                )
                return ToolResult(
                    success=False,
                    error="User declined system-level package installation",
                    tool_name="admin_elevate",
                    message="⚠️ System-level package install declined.",
                )
            # User approved — skip the generic consent and elevate now
            start = time.monotonic()
            return self._execute_admin_command(
                intent, command, cwd, handler, priv_info,
                session_context, start, already_elevated=False,
            )

        # ── SERVICE_CONTROL: route to tailored handler (Step 7.3) ─
        if category_name == "SERVICE_CONTROL":
            return self._handle_service_control(intent, cwd, session_context)

        # ── PERMISSION: route to tailored handler (Step 7.3) ─────
        if category_name == "PERMISSION":
            file_path = _entity_value(intent, "path", "file", "target") or ""
            return self._handle_permission_change(
                intent, file_path, command, cwd, session_context,
            )

        # ── CUDA/GPU: route to tailored handler (Step 7.3) ───────
        if category_name == "SYSTEM_CONFIG" and re.search(
            r"\bcuda|nvidia|gpu\b", command, re.IGNORECASE,
        ):
            return self._handle_cuda_gpu_setup(
                intent, command, cwd, session_context,
            )

        # ── General admin escalation (all other categories) ──────
        consent_desc = self._format_admin_consent_desc(
            command, category_name, reason,
        )
        consent_payload_gen = self._build_admin_consent_request(
            request_id=f"admin_intercept_{int(time.time())}",
            operation="admin_elevate",
            description=consent_desc,
            details={
                "command": command,
                "category": category_name,
                "reason": reason,
            },
        )
        if not self._request_consent(intent, consent_payload_gen):
            self._log_admin_audit(command, reason, "DENIED", "USER_DECLINED")
            return ToolResult(
                success=False,
                error="User declined admin elevation",
                tool_name="admin_elevate",
                message=f"⚠️ Admin elevation declined for: `{command}`",
            )

        start = time.monotonic()
        return self._execute_admin_command(
            intent, command, cwd, handler, priv_info,
            session_context, start, already_elevated=False,
        )

    # ── Step 7.3: Tailored admin operation handlers ───────────────

    def _handle_service_control(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle service start/stop with status before/after (Step 7.3).

        This is invoked when a RUN_COMMAND is detected as SERVICE_CONTROL.
        """
        start = time.monotonic()
        command = (
            _entity_value(intent, "command", "cmd", "text")
            or intent.raw_message
            or ""
        )
        service_name = ""

        # Try to extract service name
        try:
            parts_list = shlex.split(command)
        except ValueError:
            parts_list = command.split()

        for i, part in enumerate(parts_list):
            if part in ("start", "stop", "restart", "status") and i + 1 < len(parts_list):
                service_name = parts_list[i + 1]
                break
            if part in ("systemctl",) and i + 2 < len(parts_list):
                service_name = parts_list[i + 2]
                break

        # Show current status before operation
        status_output = ""
        if service_name:
            if os.name == "nt":
                status_cmd = f"sc query {service_name}"
            else:
                status_cmd = f"systemctl status {service_name}"
            try:
                proc = subprocess.run(
                    status_cmd, shell=True, capture_output=True,
                    text=True, timeout=15,
                )
                status_output = proc.stdout or proc.stderr or ""
            except Exception:
                pass

        msg_parts: List[str] = []
        if status_output:
            msg_parts.append(
                f"**Current status of {service_name}:**\n```\n{status_output[:500]}\n```"
            )

        # ── Consent before elevation ─────────────────────────────
        consent_desc = (
            f"🔧 **Service Control**\n\n"
            f"**Command:** `{command}`\n"
        )
        if service_name:
            consent_desc += f"**Service:** `{service_name}`\n"
        if status_output:
            consent_desc += (
                f"**Current status:**\n```\n{status_output[:300]}\n```\n"
            )
        consent_desc += "\nProceed with service control? (yes/no)"

        consent_payload = self._build_admin_consent_request(
            request_id=f"svc_ctrl_{int(time.time())}",
            operation="service_control",
            description=consent_desc,
            details={
                "command": command,
                "service": service_name,
                "current_status": status_output[:200],
            },
            risk_level="high",
            reversible=True,
        )

        if not self._request_consent(intent, consent_payload):
            self._log_admin_audit(
                command, "service_control", "DENIED", "USER_DECLINED",
            )
            return ToolResult(
                success=False,
                error="User declined service control",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="⚠️ Service control operation declined.",
            )

        # Execute the service command with elevation
        handler = self._get_admin_handler()
        if handler is None:
            return ToolResult(
                success=False,
                error="AdminPrivilegeHandler not available",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Admin handler not available for service control.",
            )

        priv_info = handler.get_privilege_info()
        result = self._execute_admin_command(
            intent, command, cwd, handler, priv_info,
            session_context, start,
            already_elevated=(priv_info.level != PrivilegeLevel.STANDARD),
        )

        # Show status after operation
        if service_name and result.success:
            try:
                if os.name == "nt":
                    after_cmd = f"sc query {service_name}"
                else:
                    after_cmd = f"systemctl status {service_name}"
                proc = subprocess.run(
                    after_cmd, shell=True, capture_output=True,
                    text=True, timeout=15,
                )
                after_status = proc.stdout or proc.stderr or ""
                if after_status:
                    result.message = (
                        (result.message or "")
                        + f"\n\n**Status after operation:**\n```\n{after_status[:500]}\n```"
                    )
            except Exception:
                pass

        return result

    def _handle_permission_change(
        self,
        intent: Intent,
        file_path: str,
        command: str,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle permission changes with before/after display (Step 7.3).

        Shows current permissions, the planned change, requests consent,
        then executes with elevation.
        """
        start = time.monotonic()

        # Get current permissions
        current_perms = ""
        try:
            if os.name == "nt":
                proc = subprocess.run(
                    f"icacls \"{file_path}\"",
                    shell=True, capture_output=True, text=True, timeout=10,
                )
            else:
                proc = subprocess.run(
                    f"ls -la \"{file_path}\"",
                    shell=True, capture_output=True, text=True, timeout=10,
                )
            current_perms = proc.stdout or proc.stderr or ""
        except Exception:
            pass

        consent_desc = (
            f"🔐 **Permission Change**\n\n"
            f"**Command:** `{command}`\n"
            f"**Target:** `{file_path}`\n"
        )
        if current_perms:
            consent_desc += (
                f"**Current permissions:**\n```\n{current_perms[:300]}\n```\n"
            )
        consent_desc += "\nProceed with permission change? (yes/no)"

        consent_payload = self._build_admin_consent_request(
            request_id=f"perm_change_{int(time.time())}",
            operation="permission_change",
            description=consent_desc,
            details={
                "command": command,
                "target": file_path,
                "current_permissions": current_perms[:200],
            },
            risk_level="high",
            reversible=True,
        )

        if not self._request_consent(intent, consent_payload):
            self._log_admin_audit(
                command, "permission_change", "DENIED", "USER_DECLINED",
                files_affected=[file_path],
            )
            return ToolResult(
                success=False,
                error="User declined permission change",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="⚠️ Permission change declined.",
            )

        handler = self._get_admin_handler()
        if handler is None:
            return ToolResult(
                success=False,
                error="AdminPrivilegeHandler not available",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Admin handler not available.",
            )

        priv_info = handler.get_privilege_info()
        result = self._execute_admin_command(
            intent, command, cwd, handler, priv_info,
            session_context, start,
            already_elevated=(priv_info.level != PrivilegeLevel.STANDARD),
        )

        if result.success:
            self._log_admin_audit(
                command, "permission_change", "APPROVED", "SUCCESS",
                files_affected=[file_path],
            )
        return result

    def _handle_cuda_gpu_setup(
        self,
        intent: Intent,
        command: str,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle CUDA/GPU setup with post-install verification (Step 7.3)."""
        start = time.monotonic()

        # ── Consent before elevation ─────────────────────────────
        consent_desc = (
            f"🖥️ **CUDA / GPU Setup**\n\n"
            f"**Command:** `{command}`\n\n"
            f"This will install or configure GPU drivers/toolkits which "
            f"requires elevated privileges.\n\n"
            f"Proceed with CUDA/GPU setup? (yes/no)"
        )
        consent_payload = self._build_admin_consent_request(
            request_id=f"cuda_gpu_{int(time.time())}",
            operation="cuda_gpu_setup",
            description=consent_desc,
            details={"command": command},
            risk_level="high",
            reversible=False,
        )

        if not self._request_consent(intent, consent_payload):
            self._log_admin_audit(
                command, "cuda_gpu_setup", "DENIED", "USER_DECLINED",
            )
            return ToolResult(
                success=False,
                error="User declined CUDA/GPU setup",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="⚠️ CUDA/GPU setup declined.",
            )

        handler = self._get_admin_handler()
        if handler is None:
            return ToolResult(
                success=False,
                error="AdminPrivilegeHandler not available",
                tool_name="admin_elevate",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Admin handler not available for CUDA/GPU setup.",
            )

        priv_info = handler.get_privilege_info()
        result = self._execute_admin_command(
            intent, command, cwd, handler, priv_info,
            session_context, start,
            already_elevated=(priv_info.level != PrivilegeLevel.STANDARD),
        )

        if not result.success:
            return result

        # Post-install verification
        verify_parts: List[str] = [result.message or ""]
        verify_parts.append("\n🧪 **Post-installation verification:**")

        # Check nvidia-smi
        try:
            proc = subprocess.run(
                "nvidia-smi",
                shell=True, capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                verify_parts.append(f"  ✅ `nvidia-smi`: {(proc.stdout or '').splitlines()[0] if proc.stdout else 'OK'}")
            else:
                verify_parts.append(f"  ❌ `nvidia-smi` failed: {proc.stderr[:100] if proc.stderr else 'not found'}")
        except Exception:
            verify_parts.append("  ❌ `nvidia-smi` not available")

        # Check nvcc
        try:
            proc = subprocess.run(
                "nvcc --version",
                shell=True, capture_output=True, text=True, timeout=10,
            )
            if proc.returncode == 0:
                version_line = ""
                for line in (proc.stdout or "").splitlines():
                    if "release" in line.lower():
                        version_line = line.strip()
                        break
                verify_parts.append(f"  ✅ `nvcc`: {version_line or 'available'}")
            else:
                verify_parts.append("  ❌ `nvcc` not available")
        except Exception:
            verify_parts.append("  ❌ `nvcc` not available")

        result.message = "\n".join(verify_parts)
        return result

    # ═════════════════════════════════════════════════════════════════
    # Phase 8 — GitHub / Git Operations
    # ═════════════════════════════════════════════════════════════════

    def _dispatch_phase8(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> Optional[ToolResult]:
        """Route Phase 8 intents (all ``GIT_*`` operations) to handlers.

        Returns ``None`` if the intent is not Phase 8.
        """
        if intent.intent_type not in _PHASE8_INTENTS:
            return None

        it = intent.intent_type

        if it == IntentType.GIT_CLONE:
            return self._handle_git_clone(intent, cwd, session_context)
        if it == IntentType.GIT_PULL:
            return self._handle_git_pull(intent, cwd, session_context)
        if it == IntentType.GIT_PUSH:
            return self._handle_git_push(intent, cwd, session_context)
        if it == IntentType.GIT_CHECKOUT:
            return self._handle_git_checkout(intent, cwd, session_context)
        if it == IntentType.GIT_BRANCH:
            return self._handle_git_branch(intent, cwd, session_context)
        if it == IntentType.GIT_MERGE:
            return self._handle_git_merge(intent, cwd, session_context)
        if it == IntentType.GIT_COMMIT:
            return self._handle_git_commit(intent, cwd, session_context)
        if it == IntentType.GIT_ADD:
            return self._handle_git_add(intent, cwd, session_context)

        return None

    # ── Step 8.1: Git Clone ───────────────────────────────────────

    def _handle_git_clone(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``GIT_CLONE`` with URL normalisation and session updates.

        - Normalises URL (HTTPS, SSH, short-form owner/repo)
        - Extracts optional branch
        - Defaults clone path to ``~/.proxima/backends/<repo>``
        - Updates SessionContext after success
        """
        start = time.monotonic()
        url_raw = (
            _entity_value(intent, "url", "repo_url", "repository")
            or intent.raw_message.strip()
        )
        dest = _entity_value(intent, "destination", "path", "dirname")
        branch = _entity_value(intent, "branch", "branch_name")

        # Try to extract a URL from raw message if not in entities
        if not url_raw or not any(c in url_raw for c in (".", "/", "@")):
            url_raw = ""
        url = _normalise_git_url(url_raw) if url_raw else ""

        if not url:
            return ToolResult(
                success=False,
                error="No repository URL specified",
                tool_name="git_clone",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    "❌ No repository URL found.\n"
                    "Please specify a URL, e.g. "
                    "`clone https://github.com/user/repo` or "
                    "`clone user/repo`."
                ),
            )

        repo_name = _repo_name_from_url(url)

        # Default target path
        if not dest:
            dest = os.path.join(
                os.path.expanduser("~"), ".proxima", "backends", repo_name,
            )
        else:
            dest = resolve_path(dest, cwd)

        # Build the command
        parts: List[str] = ["git", "clone"]
        if branch and _validate_branch_name(branch):
            parts.extend(["--branch", branch])
        parts.append(url)
        parts.append(dest)
        command = " ".join(parts)

        # Execute
        try:
            proc = _run_git_subprocess(command, cwd, timeout=300)
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error="Git clone timed out (300s)",
                tool_name="git_clone",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"⏱️ `git clone` timed out for `{url}`.",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="git_clone",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Clone failed: {exc}",
            )

        elapsed = (time.monotonic() - start) * 1000
        output = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode == 0:
            # Update session context (Step 8.1, item 5)
            if session_context is not None:
                session_context.cloned_repos[url] = dest
                session_context.last_cloned_repo = dest
                session_context.last_cloned_url = url
                # push_directory saves the OLD cwd then sets current_directory
                session_context.push_directory(dest)

            msg_parts = [
                f"✅ Successfully cloned `{repo_name}` into `{dest}`",
            ]
            if branch:
                msg_parts.append(f"   Branch: `{branch}`")
            if output.strip():
                msg_parts.append(f"\n```\n{output.strip()[:800]}\n```")

            return ToolResult(
                success=True,
                result={
                    "url": url,
                    "destination": dest,
                    "branch": branch or "",
                    "repo_name": repo_name,
                    "output": output[:1000],
                },
                tool_name="git_clone",
                execution_time_ms=elapsed,
                message="\n".join(msg_parts),
            )
        else:
            return ToolResult(
                success=False,
                error=output.strip()[:500],
                tool_name="git_clone",
                execution_time_ms=elapsed,
                message=f"❌ Clone failed:\n```\n{output.strip()[:800]}\n```",
            )

    # ── Step 8.2: Git Pull ────────────────────────────────────────

    def _handle_git_pull(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``GIT_PULL`` with git-repo verification."""
        start = time.monotonic()
        work_dir = _resolve_git_work_dir(intent, cwd, session_context)

        if not _is_git_repo(work_dir):
            return ToolResult(
                success=False,
                error=f"Not a git repository: {work_dir}",
                tool_name="git_pull",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ `{work_dir}` is not inside a git repository.",
            )

        remote = _entity_value(intent, "remote") or ""
        branch = _entity_value(intent, "branch", "branch_name") or ""
        parts = ["git", "pull"]
        if remote:
            parts.append(_sanitize_git_ref(remote))
            if branch:
                parts.append(_sanitize_git_ref(branch))
        command = " ".join(parts)

        try:
            proc = _run_git_subprocess(command, work_dir, timeout=120)
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error="git pull timed out",
                tool_name="git_pull",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="⏱️ `git pull` timed out.",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="git_pull",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Pull failed: {exc}",
            )

        elapsed = (time.monotonic() - start) * 1000
        output = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode == 0:
            return ToolResult(
                success=True,
                result={"output": output[:1000], "directory": work_dir},
                tool_name="git_pull",
                execution_time_ms=elapsed,
                message=f"✅ Pull completed in `{work_dir}`\n```\n{output.strip()[:800]}\n```",
            )
        return ToolResult(
            success=False,
            error=output.strip()[:500],
            tool_name="git_pull",
            execution_time_ms=elapsed,
            message=f"❌ Pull failed:\n```\n{output.strip()[:800]}\n```",
        )

    # ── Step 8.2: Git Push ────────────────────────────────────────

    def _handle_git_push(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``GIT_PUSH`` with consent, auto-upstream, and error recovery."""
        start = time.monotonic()
        work_dir = _resolve_git_work_dir(intent, cwd, session_context)

        if not _is_git_repo(work_dir):
            return ToolResult(
                success=False,
                error=f"Not a git repository: {work_dir}",
                tool_name="git_push",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ `{work_dir}` is not inside a git repository.",
            )

        # Consent required for push (Step 8.2)
        consent_desc = (
            "📤 **Git Push**\n\n"
            f"Directory: `{work_dir}`\n\n"
            "This will push local commits to the remote repository.\n"
            "Proceed? (yes/no)"
        )
        if _SAFETY_AVAILABLE:
            try:
                consent_payload: Any = ConsentRequest(
                    id=f"git_push_{int(time.time())}",
                    consent_type=ConsentType.GIT_OPERATION,
                    operation="git_push",
                    description=consent_desc,
                    details={"directory": work_dir},
                    tool_name="git_push",
                    risk_level="medium",
                    reversible=False,
                )
            except Exception:
                consent_payload = consent_desc
        else:
            consent_payload = consent_desc

        if not self._request_consent(intent, consent_payload):
            return ToolResult(
                success=False,
                error="User declined git push",
                tool_name="git_push",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="⚠️ Git push declined.",
            )

        remote = _entity_value(intent, "remote") or ""
        branch = _entity_value(intent, "branch", "branch_name") or ""
        parts = ["git", "push"]
        if remote:
            parts.append(_sanitize_git_ref(remote))
            if branch:
                parts.append(_sanitize_git_ref(branch))
        command = " ".join(parts)

        try:
            proc = _run_git_subprocess(command, work_dir, timeout=120)
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error="git push timed out",
                tool_name="git_push",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="⏱️ `git push` timed out.",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="git_push",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Push failed: {exc}",
            )

        elapsed_first = (time.monotonic() - start) * 1000
        output = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode == 0:
            return ToolResult(
                success=True,
                result={"output": output[:1000], "directory": work_dir},
                tool_name="git_push",
                execution_time_ms=elapsed_first,
                message=f"✅ Push completed\n```\n{output.strip()[:800]}\n```",
            )

        # Auto-recovery: no upstream branch (Step 8.2)
        if "no upstream branch" in output.lower() or "set-upstream" in output.lower():
            cur_branch = _sanitize_git_ref(
                _git_current_branch(work_dir) or "main"
            )
            upstream_cmd = f"git push --set-upstream origin {cur_branch}"
            try:
                proc2 = _run_git_subprocess(upstream_cmd, work_dir, timeout=120)
                output2 = (proc2.stdout or "") + (proc2.stderr or "")
                elapsed = (time.monotonic() - start) * 1000
                if proc2.returncode == 0:
                    return ToolResult(
                        success=True,
                        result={"output": output2[:1000], "directory": work_dir},
                        tool_name="git_push",
                        execution_time_ms=elapsed,
                        message=(
                            f"✅ Push completed (auto set-upstream for "
                            f"`{cur_branch}`)\n```\n{output2.strip()[:800]}\n```"
                        ),
                    )
                output = output2  # use the second attempt's output for error
            except Exception:
                pass

        # Authentication error suggestion
        suggestions: List[str] = []
        if "authentication" in output.lower() or "403" in output or "401" in output:
            suggestions.append(
                "💡 Set up credentials: `git config credential.helper store` "
                "or use SSH keys."
            )

        elapsed = (time.monotonic() - start) * 1000
        msg = f"❌ Push failed:\n```\n{output.strip()[:800]}\n```"
        if suggestions:
            msg += "\n\n" + "\n".join(suggestions)

        return ToolResult(
            success=False,
            error=output.strip()[:500],
            tool_name="git_push",
            execution_time_ms=elapsed,
            message=msg,
            suggestions=suggestions or None,
        )

    # ── Step 8.3: Git Checkout ────────────────────────────────────

    def _handle_git_checkout(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``GIT_CHECKOUT`` with branch validation and fallback."""
        start = time.monotonic()
        branch = _entity_value(intent, "branch", "branch_name")
        work_dir = _resolve_git_work_dir(intent, cwd, session_context)

        if not _is_git_repo(work_dir):
            return ToolResult(
                success=False,
                error="Not a git repository",
                tool_name="git_checkout",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ `{work_dir}` is not inside a git repository.",
            )

        if not branch or not _validate_branch_name(branch):
            return ToolResult(
                success=False,
                error=f"Invalid or missing branch name: {branch!r}",
                tool_name="git_checkout",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    f"❌ Invalid branch name `{branch}`.\n"
                    "Branch names must be at least 2 characters, "
                    "start with a letter, and not be a common word."
                ),
            )

        # Try normal checkout first
        safe_branch = _sanitize_git_ref(branch)
        command = f"git checkout {safe_branch}"
        try:
            proc = _run_git_subprocess(command, work_dir, timeout=30)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="git_checkout",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Checkout failed: {exc}",
            )

        output = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode == 0:
            # Update session context (Step 8.3)
            if session_context is not None:
                session_context.last_mentioned_branches.insert(0, branch)
                if len(session_context.last_mentioned_branches) > 10:
                    session_context.last_mentioned_branches = (
                        session_context.last_mentioned_branches[:10]
                    )

            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                success=True,
                result={"branch": branch, "output": output[:500]},
                tool_name="git_checkout",
                execution_time_ms=elapsed,
                message=f"✅ Switched to branch `{branch}`\n```\n{output.strip()[:400]}\n```",
            )

        # Fallback: try creating from remote tracking branch (Step 8.3)
        # Broaden detection to cover various git error messages
        output_lower = output.lower()
        if (
            "did not match" in output_lower
            or "not found" in output_lower
            or "pathspec" in output_lower
            or "invalid reference" in output_lower
        ):
            fallback_cmd = f"git checkout -b {safe_branch} origin/{safe_branch}"
            try:
                proc2 = _run_git_subprocess(fallback_cmd, work_dir, timeout=30)
                output2 = (proc2.stdout or "") + (proc2.stderr or "")
                if proc2.returncode == 0:
                    if session_context is not None:
                        session_context.last_mentioned_branches.insert(0, branch)
                    elapsed = (time.monotonic() - start) * 1000
                    return ToolResult(
                        success=True,
                        result={"branch": branch, "output": output2[:500]},
                        tool_name="git_checkout",
                        execution_time_ms=elapsed,
                        message=(
                            f"✅ Created and switched to branch `{branch}` "
                            f"(from `origin/{branch}`)\n"
                            f"```\n{output2.strip()[:400]}\n```"
                        ),
                    )
                output = output2
            except Exception:
                pass

        elapsed = (time.monotonic() - start) * 1000
        return ToolResult(
            success=False,
            error=output.strip()[:500],
            tool_name="git_checkout",
            execution_time_ms=elapsed,
            message=f"❌ Checkout failed:\n```\n{output.strip()[:600]}\n```",
        )

    # ── Step 8.3: Git Branch ──────────────────────────────────────

    def _handle_git_branch(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``GIT_BRANCH`` with sub-operation routing.

        Detects create/delete/list/rename from the user's message.
        """
        start = time.monotonic()
        raw = (intent.raw_message or "").lower()
        branch = _entity_value(intent, "branch", "branch_name", "name")
        work_dir = _resolve_git_work_dir(intent, cwd, session_context)

        if not _is_git_repo(work_dir):
            return ToolResult(
                success=False,
                error="Not a git repository",
                tool_name="git_branch",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ `{work_dir}` is not inside a git repository.",
            )

        # ── Detect sub-operation from message ────────────────────
        if re.search(r"\b(?:create|new)\b", raw) and branch and _validate_branch_name(branch):
            command = f"git checkout -b {_sanitize_git_ref(branch)}"
        elif re.search(r"\bdelete\b", raw) and branch:
            # Consent for delete
            consent_desc = (
                f"⚠️ **Delete Branch**\n\n"
                f"Branch: `{branch}`\n"
                f"This will delete the local branch.\n"
                f"Proceed? (yes/no)"
            )
            if not self._request_consent(intent, consent_desc):
                return ToolResult(
                    success=False,
                    error="User declined branch delete",
                    tool_name="git_branch",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message=f"⚠️ Branch delete declined for `{branch}`.",
                )
            command = f"git branch -d {_sanitize_git_ref(branch)}"
        elif re.search(r"\brename\b", raw):
            old_name = branch
            # Try to find second branch name
            all_branches = intent.get_all_entities("branch")
            new_name = all_branches[1] if len(all_branches) > 1 else None
            if old_name and new_name:
                command = (
                    f"git branch -m {_sanitize_git_ref(old_name)} "
                    f"{_sanitize_git_ref(new_name)}"
                )
            else:
                return ToolResult(
                    success=False,
                    error="Rename requires old and new branch names",
                    tool_name="git_branch",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                    message="❌ Please specify both old and new branch names.",
                )
        else:
            # Default: list all branches
            command = "git branch -a"

        try:
            proc = _run_git_subprocess(command, work_dir, timeout=30)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="git_branch",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Branch operation failed: {exc}",
            )

        elapsed = (time.monotonic() - start) * 1000
        output = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode == 0:
            return ToolResult(
                success=True,
                result={"output": output[:1000], "command": command},
                tool_name="git_branch",
                execution_time_ms=elapsed,
                message=f"✅ `{command}`\n```\n{output.strip()[:800]}\n```",
            )
        return ToolResult(
            success=False,
            error=output.strip()[:500],
            tool_name="git_branch",
            execution_time_ms=elapsed,
            message=f"❌ Branch operation failed:\n```\n{output.strip()[:600]}\n```",
        )

    # ── Step 8.3: Git Merge ───────────────────────────────────────

    def _handle_git_merge(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``GIT_MERGE`` with consent and conflict detection."""
        start = time.monotonic()
        branch = _entity_value(intent, "branch", "branch_name")
        work_dir = _resolve_git_work_dir(intent, cwd, session_context)

        if not _is_git_repo(work_dir):
            return ToolResult(
                success=False,
                error="Not a git repository",
                tool_name="git_merge",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ `{work_dir}` is not inside a git repository.",
            )

        if not branch or not _validate_branch_name(branch):
            return ToolResult(
                success=False,
                error=f"Invalid or missing branch name: {branch!r}",
                tool_name="git_merge",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message="❌ Please specify a valid branch to merge from.",
            )

        # Consent required for merge (Step 8.3)
        consent_desc = (
            f"🔀 **Git Merge**\n\n"
            f"Merge `{branch}` into current branch.\n"
            f"Directory: `{work_dir}`\n\n"
            f"Proceed? (yes/no)"
        )
        if _SAFETY_AVAILABLE:
            try:
                consent_payload: Any = ConsentRequest(
                    id=f"git_merge_{int(time.time())}",
                    consent_type=ConsentType.GIT_OPERATION,
                    operation="git_merge",
                    description=consent_desc,
                    details={"branch": branch, "directory": work_dir},
                    tool_name="git_merge",
                    risk_level="medium",
                    reversible=True,
                )
            except Exception:
                consent_payload = consent_desc
        else:
            consent_payload = consent_desc

        if not self._request_consent(intent, consent_payload):
            return ToolResult(
                success=False,
                error="User declined git merge",
                tool_name="git_merge",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"⚠️ Merge of `{branch}` declined.",
            )

        command = f"git merge {_sanitize_git_ref(branch)}"
        try:
            proc = _run_git_subprocess(command, work_dir, timeout=60)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="git_merge",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Merge failed: {exc}",
            )

        elapsed = (time.monotonic() - start) * 1000
        output = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode == 0:
            return ToolResult(
                success=True,
                result={"branch": branch, "output": output[:1000]},
                tool_name="git_merge",
                execution_time_ms=elapsed,
                message=f"✅ Merged `{branch}` successfully\n```\n{output.strip()[:600]}\n```",
            )

        # Conflict detection (Step 8.3)
        if "conflict" in output.lower():
            # List conflicting files
            conflict_files: List[str] = []
            try:
                diff_proc = _run_git_subprocess(
                    "git diff --name-only --diff-filter=U",
                    work_dir,
                    timeout=10,
                )
                if diff_proc.returncode == 0 and diff_proc.stdout:
                    conflict_files = [
                        f.strip() for f in diff_proc.stdout.strip().splitlines()
                        if f.strip()
                    ]
            except Exception:
                pass

            conflict_msg = f"⚠️ **Merge conflict** merging `{branch}`\n\n"
            if conflict_files:
                conflict_msg += "**Conflicting files:**\n"
                for cf in conflict_files[:20]:
                    conflict_msg += f"  - `{cf}`\n"
            conflict_msg += (
                "\nResolve conflicts manually, then run:\n"
                "```\ngit add . && git commit\n```\n"
                "Or abort with `git merge --abort`."
            )

            return ToolResult(
                success=False,
                error="Merge conflict",
                result={"conflict_files": conflict_files},
                tool_name="git_merge",
                execution_time_ms=elapsed,
                message=conflict_msg,
            )

        return ToolResult(
            success=False,
            error=output.strip()[:500],
            tool_name="git_merge",
            execution_time_ms=elapsed,
            message=f"❌ Merge failed:\n```\n{output.strip()[:600]}\n```",
        )

    # ── Step 8.4: Git Commit ──────────────────────────────────────

    def _handle_git_commit(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``GIT_COMMIT`` with auto-message and commit-all support.

        - Extracts commit message from entities or quoted text
        - Falls back to LLM-generated message from staged diff
        - Supports "commit all" / "commit everything"
        """
        start = time.monotonic()
        raw = intent.raw_message or ""

        work_dir = _resolve_git_work_dir(intent, cwd, session_context)

        if not _is_git_repo(work_dir):
            return ToolResult(
                success=False,
                error="Not a git repository",
                tool_name="git_commit",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ `{work_dir}` is not inside a git repository.",
            )

        # Check for "commit all" / "commit everything"
        commit_all = bool(re.search(
            r"\b(?:commit\s+(?:all|everything)|all\s+changes)\b",
            raw, re.IGNORECASE,
        ))

        if commit_all:
            try:
                _run_git_subprocess("git add -A", work_dir, timeout=30)
            except Exception:
                pass

        # Check if anything is staged
        try:
            status_proc = _run_git_subprocess(
                "git diff --cached --stat", work_dir, timeout=15,
            )
            staged = (status_proc.stdout or "").strip()
        except Exception:
            staged = ""

        if not staged:
            return ToolResult(
                success=False,
                error="Nothing staged for commit",
                tool_name="git_commit",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=(
                    "❌ Nothing staged for commit.\n"
                    "Stage changes first with `git add` or say "
                    "\"commit all\" to auto-stage everything."
                ),
            )

        # Extract commit message
        message = _entity_value(intent, "message", "commit_message", "text")

        # Try quoted text in the raw message
        if not message:
            quoted = re.findall(r'["\']([^"\']+)["\']', raw)
            if quoted:
                message = quoted[0]

        # Try "with message X" / "message X" / "saying X"
        if not message:
            m = re.search(
                r'(?:with\s+message|message|saying)\s+["\']?(.+?)(?:["\']?\s*$)',
                raw, re.IGNORECASE,
            )
            if m:
                message = m.group(1).strip().rstrip('"\'')

        # LLM fallback: generate from diff (Step 8.4)
        if not message and self._llm_provider is not None:
            try:
                diff_proc = _run_git_subprocess(
                    "git diff --staged", work_dir, timeout=30,
                )
                diff_text = (diff_proc.stdout or "")[:4000]
                if diff_text:
                    llm_prompt = (
                        "Generate a concise commit message for "
                        "these changes:\n\n" + diff_text
                    )
                    # Use the provider's generate method
                    llm_resp = self._llm_provider.generate(llm_prompt)
                    if llm_resp and isinstance(llm_resp, str):
                        message = llm_resp.strip().strip('"\'')[:100]
            except Exception:
                pass

        # Final fallback message
        if not message:
            message = "Update files"

        # Platform-aware quoting for commit message
        if os.name == "nt":
            # cmd.exe uses double quotes; escape internal double-quotes
            safe_msg = message.replace('"', '\\"')
            command = f'git commit -m "{safe_msg}"'
        else:
            # POSIX shells: single-quote with escaping
            safe_msg = message.replace("'", "'\\''")
            command = f"git commit -m '{safe_msg}'"

        try:
            proc = _run_git_subprocess(command, work_dir, timeout=30)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="git_commit",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Commit failed: {exc}",
            )

        elapsed = (time.monotonic() - start) * 1000
        output = (proc.stdout or "") + (proc.stderr or "")

        if proc.returncode == 0:
            # Extract commit hash
            commit_hash = ""
            hash_match = re.search(r'\[[\w\-/]+\s+([a-f0-9]+)\]', output)
            if hash_match:
                commit_hash = hash_match.group(1)

            return ToolResult(
                success=True,
                result={
                    "message": message,
                    "commit_hash": commit_hash,
                    "output": output[:500],
                },
                tool_name="git_commit",
                execution_time_ms=elapsed,
                message=(
                    f"✅ Committed: `{message}`"
                    + (f" (`{commit_hash}`)" if commit_hash else "")
                    + f"\n```\n{output.strip()[:400]}\n```"
                ),
            )

        return ToolResult(
            success=False,
            error=output.strip()[:500],
            tool_name="git_commit",
            execution_time_ms=elapsed,
            message=f"❌ Commit failed:\n```\n{output.strip()[:600]}\n```",
        )

    # ── Step 8.4: Git Add ─────────────────────────────────────────

    def _handle_git_add(
        self,
        intent: Intent,
        cwd: str,
        session_context: Optional[SessionContext] = None,
    ) -> ToolResult:
        """Handle ``GIT_ADD`` with pattern and "add all" support."""
        start = time.monotonic()
        raw = (intent.raw_message or "").lower()
        work_dir = _resolve_git_work_dir(intent, cwd, session_context)

        if not _is_git_repo(work_dir):
            return ToolResult(
                success=False,
                error="Not a git repository",
                tool_name="git_add",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ `{work_dir}` is not inside a git repository.",
            )

        # Determine what to add
        add_all = bool(re.search(
            r"\b(?:add\s+(?:all|everything)|stage\s+(?:all|everything))\b",
            raw,
        ))

        if add_all:
            command = "git add -A"
        else:
            files = intent.get_all_entities("path") or intent.get_all_entities("filename")
            if files:
                quoted = [f'"{f}"' for f in files]
                command = f"git add {' '.join(quoted)}"
            else:
                # Pattern detection: "add all python files" → *.py
                pattern_match = re.search(
                    r"add\s+(?:all\s+)?(\w+)\s+files?",
                    raw,
                )
                if pattern_match:
                    ext_word = pattern_match.group(1).lower()
                    ext_map = {
                        "python": "*.py", "py": "*.py",
                        "javascript": "*.js", "js": "*.js",
                        "typescript": "*.ts", "ts": "*.ts",
                        "rust": "*.rs", "rs": "*.rs",
                        "java": "*.java", "c": "*.c",
                        "cpp": "*.cpp", "go": "*.go",
                        "html": "*.html", "css": "*.css",
                        "json": "*.json", "yaml": "*.yaml",
                        "yml": "*.yml", "md": "*.md",
                        "markdown": "*.md", "text": "*.txt",
                        "txt": "*.txt", "shell": "*.sh",
                        "sh": "*.sh", "bash": "*.sh",
                    }
                    pattern = ext_map.get(ext_word, f"*.{ext_word}")
                    command = f'git add "{pattern}"'
                else:
                    # Default: stage everything
                    command = "git add -A"

        try:
            proc = _run_git_subprocess(command, work_dir, timeout=30)
        except Exception as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                tool_name="git_add",
                execution_time_ms=(time.monotonic() - start) * 1000,
                message=f"❌ Git add failed: {exc}",
            )

        elapsed_add = (time.monotonic() - start) * 1000

        if proc.returncode != 0:
            output = (proc.stdout or "") + (proc.stderr or "")
            return ToolResult(
                success=False,
                error=output.strip()[:500],
                tool_name="git_add",
                execution_time_ms=elapsed_add,
                message=f"❌ Git add failed:\n```\n{output.strip()[:600]}\n```",
            )

        # Show status after add (Step 8.4)
        status_out = ""
        try:
            status_proc = _run_git_subprocess(
                "git status --short", work_dir, timeout=10,
            )
            status_out = (status_proc.stdout or "").strip()
        except Exception:
            pass

        elapsed = (time.monotonic() - start) * 1000
        msg = f"✅ `{command}` completed"
        if status_out:
            msg += f"\n\n**Staged files:**\n```\n{status_out[:800]}\n```"

        return ToolResult(
            success=True,
            result={"command": command, "status": status_out[:500]},
            tool_name="git_add",
            execution_time_ms=elapsed,
            message=msg,
        )

    # ── Phase 8 helper: set LLM provider ──────────────────────────

    def set_llm_provider(self, provider: Any) -> None:
        """Register an LLM provider for commit message generation.

        The provider must have a ``generate(prompt: str) -> str`` method.
        """
        self._llm_provider = provider

    def _record(
        self,
        intent: Intent,
        tool_name: str,
        args: Dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Record an execution in the history list."""
        self._execution_history.append({
            "intent_type": intent.intent_type.name,
            "tool_name": tool_name,
            "args": args,
            "success": result.success,
            "timestamp": time.time(),
        })
        # Keep last 100 entries
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]

    # ── introspection ─────────────────────────────────────────────

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return a copy of the execution history."""
        return list(self._execution_history)

    @staticmethod
    def get_tool_for_intent(intent_type: IntentType) -> Optional[str]:
        """Return the registered tool name for an intent type."""
        return INTENT_TO_TOOL.get(intent_type)

    @staticmethod
    def detect_dependency_file(repo_path: str) -> Optional[Tuple[str, str]]:
        """Public wrapper around ``_detect_dependency_file``."""
        return _detect_dependency_file(repo_path)
