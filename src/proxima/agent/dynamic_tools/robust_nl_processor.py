"""Robust Natural Language Processor for Dynamic Intent Recognition.

This module provides model-agnostic natural language understanding that works
with any integrated LLM model, including smaller models like llama2-uncensored.

Key features:
1. Hybrid intent recognition (rule-based + LLM-assisted)
2. Context tracking across multiple messages
3. Robust fallback mechanisms when LLM doesn't return expected format
4. Multi-step operation chaining
5. Session state management

Architecture Principles:
- The assistant's architecture remains stable
- The integrated model operates dynamically through NL understanding
- Intent-driven execution regardless of phrasing
"""

import re
import os
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import ClassVar, List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum, auto
import threading


class IntentType(Enum):
    """Canonical intent taxonomy for all agent capabilities.

    This is the single source of truth for all intent types across the system.
    Every capability the agent supports maps to one or more intents defined here.

    Grouped by domain:
    - Navigation & Directory
    - Git operations (basic + extended)
    - File operations
    - Directory operations
    - Terminal & Script execution
    - Query operations
    - Dependency management
    - Backend build & modification
    - Search & Analysis
    - Plan & Execution control
    - System & Admin
    - Web & Research
    - Complex / Meta
    """

    # ── Navigation ────────────────────────────────────────────────────
    NAVIGATE_DIRECTORY = auto()
    LIST_DIRECTORY = auto()
    SHOW_CURRENT_DIR = auto()

    # ── Git operations (basic) ────────────────────────────────────────
    GIT_CHECKOUT = auto()
    GIT_CLONE = auto()
    GIT_PULL = auto()
    GIT_PUSH = auto()
    GIT_STATUS = auto()
    GIT_COMMIT = auto()
    GIT_ADD = auto()
    GIT_BRANCH = auto()
    GIT_FETCH = auto()

    # ── Git operations (extended — Phase 1 additions) ─────────────────
    GIT_MERGE = auto()
    GIT_REBASE = auto()
    GIT_STASH = auto()
    GIT_LOG = auto()
    GIT_DIFF = auto()
    GIT_CONFLICT_RESOLVE = auto()

    # ── File operations ───────────────────────────────────────────────
    CREATE_FILE = auto()
    READ_FILE = auto()
    WRITE_FILE = auto()
    DELETE_FILE = auto()
    COPY_FILE = auto()
    MOVE_FILE = auto()

    # ── Directory operations ──────────────────────────────────────────
    CREATE_DIRECTORY = auto()
    DELETE_DIRECTORY = auto()
    COPY_DIRECTORY = auto()

    # ── Terminal & Script execution ───────────────────────────────────
    RUN_COMMAND = auto()
    RUN_SCRIPT = auto()
    TERMINAL_MONITOR = auto()
    TERMINAL_KILL = auto()
    TERMINAL_OUTPUT = auto()
    TERMINAL_LIST = auto()

    # ── Query operations ──────────────────────────────────────────────
    QUERY_LOCATION = auto()   # "where is X", "where did you clone"
    QUERY_STATUS = auto()     # "what happened", "did it work"

    # ── Dependency management ─────────────────────────────────────────
    INSTALL_DEPENDENCY = auto()
    CONFIGURE_ENVIRONMENT = auto()
    CHECK_DEPENDENCY = auto()

    # ── Search & Analysis ─────────────────────────────────────────────
    SEARCH_FILE = auto()
    ANALYZE_RESULTS = auto()
    EXPORT_RESULTS = auto()

    # ── Plan & Execution control ──────────────────────────────────────
    PLAN_EXECUTION = auto()
    UNDO_OPERATION = auto()
    REDO_OPERATION = auto()

    # ── Backend build & modification ──────────────────────────────────
    BACKEND_BUILD = auto()
    BACKEND_CONFIGURE = auto()
    BACKEND_TEST = auto()
    BACKEND_MODIFY = auto()
    BACKEND_LIST = auto()

    # ── System & Admin ────────────────────────────────────────────────
    SYSTEM_INFO = auto()
    ADMIN_ELEVATE = auto()

    # ── Web & Research ────────────────────────────────────────────────
    WEB_SEARCH = auto()

    # ── Complex / Meta ────────────────────────────────────────────────
    MULTI_STEP = auto()
    UNKNOWN = auto()

@dataclass
class ExtractedEntity:
    """An entity extracted from natural language."""
    entity_type: str  # 'path', 'branch', 'url', 'filename', 'command'
    value: str
    confidence: float = 1.0
    source: str = "regex"  # 'regex', 'llm', 'context'


@dataclass
class Intent:
    """A recognized user intent."""
    intent_type: IntentType
    entities: List[ExtractedEntity] = field(default_factory=list)
    confidence: float = 0.0
    raw_message: str = ""
    explanation: str = ""
    sub_intents: List['Intent'] = field(default_factory=list)
    
    def get_entity(self, entity_type: str) -> Optional[str]:
        """Get the first entity of a given type."""
        for entity in self.entities:
            if entity.entity_type == entity_type:
                return entity.value
        return None
    
    def get_all_entities(self, entity_type: str) -> List[str]:
        """Get all entities of a given type."""
        return [e.value for e in self.entities if e.entity_type == entity_type]


@dataclass
class SessionContext:
    """Tracks context across multiple messages in a session.

    Maintains stateful information that persists across conversation turns,
    enabling pronoun resolution ("it", "that", "the repo"), directory
    stack navigation ("go back"), and contextual intent inference.
    """

    # ── Core state ────────────────────────────────────────────────────
    current_directory: str = field(default_factory=os.getcwd)
    last_mentioned_paths: List[str] = field(default_factory=list)
    last_mentioned_branches: List[str] = field(default_factory=list)
    last_mentioned_urls: List[str] = field(default_factory=list)
    last_operation: Optional[Intent] = None
    operation_history: List[Intent] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)

    # ── Clone tracking ────────────────────────────────────────────────
    cloned_repos: Dict[str, str] = field(default_factory=dict)
    last_cloned_repo: Optional[str] = None
    last_cloned_url: Optional[str] = None

    # ── Phase 2 additions ─────────────────────────────────────────────
    # Terminal tracking: terminal_id → {command, state, last_output, pid}
    active_terminals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Packages installed during this session
    installed_packages: List[str] = field(default_factory=list)
    # Active virtual environments: name → path
    active_environments: Dict[str, str] = field(default_factory=dict)
    # Text result of the last executed operation (for "it" / "that" resolution)
    last_operation_result: Optional[str] = None
    # Path of the last script that was run
    last_script_executed: Optional[str] = None
    # Recently referenced package names (max 10)
    last_mentioned_packages: List[str] = field(default_factory=list)
    # Last 20 (user_message, intent_type_name) pairs for context inference
    conversation_history: List[Tuple[str, str]] = field(default_factory=list)
    # Directory stack for pushd / popd semantics
    working_directory_stack: List[str] = field(default_factory=list)
    # Name or path of the last backend that was built
    last_built_backend: Optional[str] = None
    # Backend name → most recent checkpoint ID
    backend_checkpoints: Dict[str, str] = field(default_factory=dict)
    # Files modified in the last backend code-modification operation
    last_modified_files: List[str] = field(default_factory=list)

    # ── Helper constants (ClassVar — excluded from __init__/__repr__) ──
    _ACTION_VERBS: ClassVar[Tuple[str, ...]] = (
        'run', 'execute', 'build', 'test', 'compile', 'start', 'launch',
        'install', 'deploy', 'use',
    )
    _QUERY_VERBS: ClassVar[Tuple[str, ...]] = (
        'show', 'what', 'print', 'display', 'view', 'see', 'read', 'get',
        'tell', 'describe',
    )

    # ── Existing helpers ──────────────────────────────────────────────

    def add_path(self, path: str):
        """Add a path to context."""
        if path and path not in self.last_mentioned_paths:
            self.last_mentioned_paths.insert(0, path)
            if len(self.last_mentioned_paths) > 10:
                self.last_mentioned_paths.pop()

    def add_branch(self, branch: str):
        """Add a branch to context."""
        if branch and branch not in self.last_mentioned_branches:
            self.last_mentioned_branches.insert(0, branch)
            if len(self.last_mentioned_branches) > 10:
                self.last_mentioned_branches.pop()

    def add_url(self, url: str):
        """Add a URL to context."""
        if url and url not in self.last_mentioned_urls:
            self.last_mentioned_urls.insert(0, url)
            if len(self.last_mentioned_urls) > 10:
                self.last_mentioned_urls.pop()

    def record_clone(self, url: str, cloned_path: str):
        """Record a cloned repository."""
        self.cloned_repos[url] = cloned_path
        self.last_cloned_repo = cloned_path
        self.last_cloned_url = url
        self.add_url(url)
        self.add_path(cloned_path)

    def update_from_intent(self, intent: Intent):
        """Update context from a processed intent."""
        self.last_operation = intent
        self.operation_history.append(intent)

        for entity in intent.entities:
            if entity.entity_type == 'path':
                self.add_path(entity.value)
            elif entity.entity_type == 'branch':
                self.add_branch(entity.value)
            elif entity.entity_type == 'url':
                self.add_url(entity.value)
            elif entity.entity_type == 'package':
                self.add_package(entity.value)
            elif entity.entity_type in ('script_path', 'script'):
                self.last_script_executed = entity.value

    # ── Phase 2 helpers ───────────────────────────────────────────────

    def add_package(self, package: str):
        """Record a recently-mentioned package name (max 10)."""
        if package and package not in self.last_mentioned_packages:
            self.last_mentioned_packages.insert(0, package)
            if len(self.last_mentioned_packages) > 10:
                self.last_mentioned_packages.pop()

    def add_conversation_entry(self, message: str, intent_type: str):
        """Append a (user_message, intent_type_name) pair and trim to 20."""
        self.conversation_history.append((message, intent_type))
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def push_directory(self, path: str):
        """Push the current directory onto the stack and update cwd."""
        self.working_directory_stack.append(self.current_directory)
        self.current_directory = path
        self.add_path(path)

    def pop_directory(self) -> Optional[str]:
        """Pop the previous directory from the stack, update cwd, return it."""
        if self.working_directory_stack:
            prev = self.working_directory_stack.pop()
            self.current_directory = prev
            return prev
        return None

    # ── Pronoun / reference resolution (Phase 2, Step 2.3) ───────────

    def resolve_reference(
        self,
        text: str,
        verb_context: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a contextual pronoun or reference to a concrete value.

        Resolution rules are applied in order; the first match wins.

        Args:
            text: The pronoun or reference phrase (e.g. "it", "the repo").
            verb_context: Optional surrounding verb for disambiguation.

        Returns:
            The resolved concrete value, or ``None`` if unresolvable.
        """
        t = text.strip().lower()

        # ── Rule 1: "it" / "that" — disambiguate via verb context ────
        if t in ('it', 'that'):
            if verb_context:
                vc = verb_context.strip().lower()
                # Action-oriented verb → return an actionable entity
                if any(vc.startswith(v) for v in self._ACTION_VERBS):
                    return (
                        self.last_script_executed
                        or self.last_cloned_repo
                        or self.last_operation_result
                    )
                # Query-oriented verb → return the last result
                if any(vc.startswith(v) for v in self._QUERY_VERBS):
                    return self.last_operation_result
            # Fallback when no verb context is available
            return self.last_operation_result

        # ── Rule 2: "the result" / "the output" ──────────────────────
        if t in ('the result', 'the output', 'result', 'output'):
            return self.last_operation_result

        # ── Rule 3: "the repo" / "the repository" / "that repo" ──────
        if t in ('the repo', 'the repository', 'that repo', 'repo'):
            return self.last_cloned_repo or self.last_cloned_url

        # ── Rule 4: "there" / "that directory" / "that folder" ────────
        if t in ('there', 'that directory', 'that folder', 'the directory', 'the folder'):
            return self.last_mentioned_paths[0] if self.last_mentioned_paths else None

        # ── Rule 5: "that branch" / "the branch" ─────────────────────
        if t in ('that branch', 'the branch'):
            return self.last_mentioned_branches[0] if self.last_mentioned_branches else None

        # ── Rule 6: "the script" / "that script" ─────────────────────
        if t in ('the script', 'that script'):
            return self.last_script_executed

        # ── Rule 7: "that backend" / "the backend" ───────────────────
        if t in ('that backend', 'the backend'):
            return self.last_built_backend

        # ── Rule 8: "back" / "previous directory" ─────────────────────
        if t in ('back', 'previous directory', 'previous folder'):
            return self.pop_directory()

        return None


# ---------------------------------------------------------------------------
# Set of intent types added in Phase 1 that are **recognised** by the
# processor's keyword engine but whose *execution* is deferred to the
# upper orchestration layer (IntentToolBridge / agent_ai_assistant).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Intent types whose execution is deferred to the upper orchestration layer
# (IntentToolBridge / agent_ai_assistant).  This includes Phase-1 additions
# as well as original intents that never had a direct execute branch.
# ---------------------------------------------------------------------------
_DEFERRED_INTENTS: frozenset = frozenset({
    # Git basic (no direct execute branch)
    IntentType.GIT_PUSH,
    IntentType.GIT_COMMIT,
    IntentType.GIT_ADD,
    IntentType.GIT_BRANCH,
    IntentType.GIT_FETCH,
    # Git extended (Phase 1)
    IntentType.GIT_MERGE,
    IntentType.GIT_REBASE,
    IntentType.GIT_STASH,
    IntentType.GIT_LOG,
    IntentType.GIT_DIFF,
    IntentType.GIT_CONFLICT_RESOLVE,
    # Terminal monitoring (Phase 1)
    IntentType.TERMINAL_MONITOR,
    IntentType.TERMINAL_KILL,
    IntentType.TERMINAL_OUTPUT,
    IntentType.TERMINAL_LIST,
    # Dependency management (Phase 1)
    IntentType.INSTALL_DEPENDENCY,
    IntentType.CONFIGURE_ENVIRONMENT,
    IntentType.CHECK_DEPENDENCY,
    # Search & Analysis (Phase 1)
    IntentType.SEARCH_FILE,
    IntentType.ANALYZE_RESULTS,
    IntentType.EXPORT_RESULTS,
    # Plan & Execution control (Phase 1)
    IntentType.PLAN_EXECUTION,
    IntentType.UNDO_OPERATION,
    IntentType.REDO_OPERATION,
    # Backend (Phase 1)
    IntentType.BACKEND_BUILD,
    IntentType.BACKEND_CONFIGURE,
    IntentType.BACKEND_TEST,
    IntentType.BACKEND_MODIFY,
    IntentType.BACKEND_LIST,
    # System & Admin (Phase 1)
    IntentType.SYSTEM_INFO,
    IntentType.ADMIN_ELEVATE,
    # Web (Phase 1)
    IntentType.WEB_SEARCH,
    # File ops (no direct execute branch)
    IntentType.CREATE_FILE,
    IntentType.READ_FILE,
    IntentType.WRITE_FILE,
    IntentType.DELETE_FILE,
    IntentType.COPY_FILE,
    IntentType.MOVE_FILE,
    # Directory ops (no direct execute branch)
    IntentType.CREATE_DIRECTORY,
    IntentType.DELETE_DIRECTORY,
    IntentType.COPY_DIRECTORY,
    # Query (no direct execute branch)
    IntentType.QUERY_STATUS,
})


class RobustNLProcessor:
    """Robust Natural Language Processor that works with any LLM.
    
    This processor uses a hybrid approach:
    1. First attempts rule-based pattern matching (always works)
    2. Uses LLM for disambiguation when needed
    3. Falls back gracefully when LLM doesn't return expected format
    4. Maintains context across multiple messages
    """

    # Stopwords used during branch-name validation (avoid per-call list alloc)
    _BRANCH_STOPWORDS = frozenset({
        'the', 'and', 'for', 'with', 'this', 'that',
        'from', 'clone', 'build', 'run', 'to', 'a', 'an',
        'of', 'it', 'in', 'on', 'at', 'be', 'is', 'are',
        'repo', 'repository', 'directory', 'folder', 'file',
        'branch', 'all', 'please', 'want', 'use', 'updated',
    })

    # Stopwords used during dirname validation
    _DIRNAME_STOPWORDS = frozenset({
        'the', 'a', 'an', 'this', 'that', 'and', 'or',
        'directory', 'folder', 'file', 'repo', 'branch',
    })

    # Pre-compiled query-detection patterns (avoid per-call re.compile)
    _QUERY_PATTERNS = [
        re.compile(r'where\s+is\s+'),
        re.compile(r'where\s+did\s+'),
        re.compile(r'where\s+was\s+'),
        re.compile(r'where\s+is\s+that'),
        re.compile(r'where\s+is\s+the'),
        re.compile(r'location\s+of'),
        re.compile(r'find\s+the\s+.*(?:repo|clone|path)'),
        re.compile(r'what\s+path'),
        re.compile(r'where.*(?:repo|clone|put|save)'),
    ]
    
    def __init__(self, llm_router=None):
        """Initialize the processor.
        
        Args:
            llm_router: Optional LLM router for LLM-assisted parsing
        """
        self._llm_router = llm_router
        self._context = SessionContext()
        self._lock = threading.Lock()
        
        # Compile regex patterns for entity extraction
        self._patterns = self._compile_patterns()
        
        # Intent patterns - maps keywords to intents
        self._intent_keywords = self._build_intent_keywords()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for entity extraction."""
        return {
            # Path patterns - including quoted paths (for paths with spaces)
            'windows_path': [
                # Quoted Windows paths (with spaces) - PRIORITY
                re.compile(r'"([A-Za-z]:[\\\/][^"]+)"', re.IGNORECASE),
                re.compile(r"'([A-Za-z]:[\\\/][^']+)'", re.IGNORECASE),
                # Unquoted Windows paths (no spaces)
                re.compile(r'([A-Za-z]:[\\\/][^\s,\'"]+)', re.IGNORECASE),
                re.compile(r'(?:at|in|from|to|inside|into)\s+([A-Za-z]:[\\\/][^\s,\'"]+)', re.IGNORECASE),
            ],
            'unix_path': [
                # Quoted Unix paths
                re.compile(r'"(\/[^"]+)"', re.IGNORECASE),
                re.compile(r"'(\/[^']+)'", re.IGNORECASE),
                # Unquoted Unix paths
                re.compile(r'(?:at|in|from|to|inside|into)\s+(\/[^\s,\'"]+)', re.IGNORECASE),
                re.compile(r'(?:at|in|from|to|inside|into)\s+(~[^\s,\'"]*)', re.IGNORECASE),
            ],
            'relative_path': [
                re.compile(r'(?:at|in|from|to|inside|into)\s+(\.\/?[^\s,\'"]+)', re.IGNORECASE),
                re.compile(r'(?:folder|directory|dir)\s+([a-zA-Z_][a-zA-Z0-9_\-\.]*)', re.IGNORECASE),
            ],
            
            # Branch patterns - FIXED: Don't match list numbers or common words
            'branch': [
                # "branch X" or "X branch" patterns
                re.compile(r'(?:branch|the)\s+([a-zA-Z][a-zA-Z0-9_\-]+(?:[\-\/][a-zA-Z0-9_\-]+)*)\s+branch', re.IGNORECASE),
                re.compile(r'branch\s+([a-zA-Z][a-zA-Z0-9_\-]+(?:[\-\/][a-zA-Z0-9_\-]+)*)', re.IGNORECASE),
                # "switch/checkout to X branch" patterns
                re.compile(r'(?:switch|checkout)\s+(?:to\s+)?([a-zA-Z][a-zA-Z0-9_\-]+(?:[\-\/][a-zA-Z0-9_\-]+)*)\s+branch', re.IGNORECASE),
                # "switch to X" where X looks like a branch name (has hyphens)
                re.compile(r'(?:switch|checkout)\s+to\s+([a-zA-Z][a-zA-Z0-9]*(?:\-[a-zA-Z0-9]+)+)', re.IGNORECASE),
            ],
            
            # URL patterns - FIXED: Capture full https:// URL correctly
            'github_url': [
                # Full URL with https:// - capture the whole thing
                re.compile(r'(https://github\.com/[^\s\'"<>]+)', re.IGNORECASE),
                re.compile(r'(http://github\.com/[^\s\'"<>]+)', re.IGNORECASE),
                # URL without protocol (github.com/...)
                re.compile(r'(?<![:/])(github\.com/[^\s\'"<>]+)', re.IGNORECASE),
                # SSH URL   git@github.com:user/repo.git
                re.compile(r'(git@github\.com:[\w\-\.]+/[\w\-\.]+(?:\.git)?)', re.IGNORECASE),
                # Short-form owner/repo after clone-related context words
                # e.g. "clone kunal5556/LRET" or "pull from user/repo"
                re.compile(
                    r'(?:clone|pull|fork|fetch)\s+'
                    r'([\w\-\.]+/[\w\-\.]+)',
                    re.IGNORECASE,
                ),
            ],
            'git_url': [
                re.compile(r'(https?://[^\s\'"<>]+\.git)', re.IGNORECASE),
                re.compile(r'(git@[^\s\'"<>]+)', re.IGNORECASE),
            ],
            'any_url': [
                # Generic URL pattern to catch any https:// URL
                re.compile(r'(https?://[^\s\'"<>]+)', re.IGNORECASE),
            ],
            
            # Filename/script patterns
            'python_script': [
                re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*\.py)', re.IGNORECASE),
                re.compile(r'(?:run|execute)\s+([^\s]+\.py)', re.IGNORECASE),
            ],
            'script': [
                re.compile(r'(?:run|execute)\s+(?:the\s+)?(?:script\s+)?([^\s]+\.(py|sh|bash|ps1|bat|cmd))', re.IGNORECASE),
            ],
            
            # Command patterns
            'quoted_command': [
                re.compile(r'["\']([^"\']+)["\']', re.IGNORECASE),
                re.compile(r'`([^`]+)`', re.IGNORECASE),
            ],

            # ── Phase 2 additions ─────────────────────────────────────

            # Package names — after dependency-related keywords
            'package': [
                # "pip install numpy>=1.21.0 pandas torch"
                re.compile(
                    r'(?:pip|pip3)\s+install\s+(?:-[^\s]+\s+)*'        # pip install [-r|-e|...]
                    r'([a-zA-Z0-9_][a-zA-Z0-9_\-]*'                   # first pkg
                    r'(?:[><=!~]+[^\s,]+)?'                            # optional version spec
                    r'(?:\s+[a-zA-Z0-9_][a-zA-Z0-9_\-]*(?:[><=!~]+[^\s,]+)?)*)',  # extra pkgs
                    re.IGNORECASE,
                ),
                # "install <pkg>" / "add package <pkg>"
                re.compile(
                    r'(?:install|add\s+package)\s+'
                    r'([a-zA-Z0-9_][a-zA-Z0-9_\-]*(?:[><=!~]+[^\s,]+)?)',
                    re.IGNORECASE,
                ),
                # "conda install pkg1 pkg2"
                re.compile(
                    r'conda\s+install\s+(?:-[^\s]+\s+)*'
                    r'([a-zA-Z0-9_][a-zA-Z0-9_\-]*'
                    r'(?:\s+[a-zA-Z0-9_][a-zA-Z0-9_\-]*)*)',
                    re.IGNORECASE,
                ),
                # "npm install pkg"
                re.compile(
                    r'npm\s+install\s+(?:-[^\s]+\s+)*'
                    r'([a-zA-Z0-9@_][a-zA-Z0-9@_/\-]*)',
                    re.IGNORECASE,
                ),
            ],

            # Script file paths (more specific than generic 'path')
            'script_path': [
                # Quoted or unquoted paths ending in common script extensions
                re.compile(
                    r'(?:run|execute|start|launch)?\s*'
                    r'(?:the\s+)?(?:script\s+)?'
                    r'["\']?([^\s"\']+\.(?:py|sh|ps1|bat|js|lua))["\']?',
                    re.IGNORECASE,
                ),
                # Standalone script filename
                re.compile(
                    r'([a-zA-Z_][a-zA-Z0-9_/\\:\-\.]*\.(?:py|sh|ps1|bat|js|lua))',
                    re.IGNORECASE,
                ),
            ],

            # Environment names (venv, .venv, etc.)
            'environment': [
                re.compile(
                    r'(?:activate|create(?:\s+venv)?(?:\s+named)?|'
                    r'use\s+(?:virtual\s+)?env(?:ironment)?)\s+'
                    r'([a-zA-Z_][a-zA-Z0-9_\-\.]*)',
                    re.IGNORECASE,
                ),
                # Bare venv-like names preceded by "environment" / "venv"
                re.compile(
                    r'\b(\.?venv|env|myenv|\.env)\b',
                    re.IGNORECASE,
                ),
            ],

            # Line ranges ("lines 10-50", "line 23", "from line 10 to 20")
            'line_range': [
                re.compile(r'lines?\s+(\d+)\s*[-–]\s*(\d+)', re.IGNORECASE),
                re.compile(r'from\s+line\s+(\d+)\s+to\s+(?:line\s+)?(\d+)', re.IGNORECASE),
                re.compile(r'line\s+(\d+)', re.IGNORECASE),
            ],

            # Process / terminal identifiers
            'process_id': [
                re.compile(r'(?:terminal|process|session)\s+#?(\d+)', re.IGNORECASE),
                re.compile(r'PID\s+(\d+)', re.IGNORECASE),
                re.compile(
                    r'(?:the\s+)?(?:build|first|second|third|last|main)\s+'
                    r'(?:terminal|process)',
                    re.IGNORECASE,
                ),
            ],

            # Context-aware quoted content (classified by preceding preposition)
            'quoted_context': [
                # After spatial / directional prepositions → path
                re.compile(
                    r'(?:in|to|at|from|into|under)\s+["\']([^"\']+)["\']',
                    re.IGNORECASE,
                ),
                # After naming prepositions → name
                re.compile(
                    r'(?:named|called|with\s+name)\s+["\']([^"\']+)["\']',
                    re.IGNORECASE,
                ),
                # After execution verbs → command
                re.compile(
                    r'(?:run|execute|with\s+command)\s+["\']([^"\']+)["\']',
                    re.IGNORECASE,
                ),
                # After install verbs → package
                re.compile(
                    r'(?:install|add)\s+["\']([^"\']+)["\']',
                    re.IGNORECASE,
                ),
            ],

            # Directory names from navigation context (pre-compiled)
            'dirname': [
                re.compile(
                    r'(?:go\s+)?(?:to|into|inside)\s+(?:the\s+)?'
                    r'([A-Za-z][A-Za-z0-9_\-\.]*)\s+(?:directory|folder|dir)',
                    re.IGNORECASE,
                ),
                re.compile(
                    r'(?:go\s+)?(?:to|into|inside)\s+([A-Za-z][A-Za-z0-9_\-\.]*)',
                    re.IGNORECASE,
                ),
                re.compile(
                    r'inside\s+([A-Za-z][A-Za-z0-9_\-\.\/]*)',
                    re.IGNORECASE,
                ),
            ],
        }
    
    def _build_intent_keywords(self) -> Dict[IntentType, List[str]]:
        """Build keyword mappings for intent recognition.

        Each ``IntentType`` maps to a list of trigger phrases.  During
        recognition the message is scanned for these phrases and the
        intent with the highest cumulative keyword-length score wins.

        **Collision-prone keywords** (e.g. *build*, *search*, *log*,
        *test*, *install*, *export*, *configure*) are intentionally kept
        under the most common intent they indicate; the scoring algorithm
        in ``recognize_intent()`` naturally favours longer / more-specific
        matches (e.g. "build backend" beats standalone "build").
        """
        return {
            # ── Navigation ────────────────────────────────────────────
            IntentType.NAVIGATE_DIRECTORY: [
                'go to', 'go into', 'go inside', 'cd ', 'navigate to',
                'change directory', 'change to', 'enter', 'open folder',
                'open directory', 'switch to folder', 'switch to directory',
                'move to', 'go in', 'inside'
            ],
            IntentType.LIST_DIRECTORY: [
                'list', 'ls', 'dir', 'show files', 'show folder',
                'what files', 'files in', 'contents of', 'list files'
            ],
            IntentType.SHOW_CURRENT_DIR: [
                'pwd', 'where am i', 'current directory', 'current folder',
                'which directory', 'what directory', 'cwd'
            ],

            # ── Git (basic) ───────────────────────────────────────────
            IntentType.GIT_CHECKOUT: [
                'checkout', 'switch to branch', 'switch branch', 'git checkout',
                'git switch', 'change branch', 'use branch'
            ],
            IntentType.GIT_CLONE: [
                'clone', 'git clone', 'download repo', 'fetch repo'
            ],
            IntentType.GIT_PULL: [
                'git pull', 'pull', 'update repo', 'sync', 'fetch changes'
            ],
            IntentType.GIT_PUSH: [
                'git push', 'push', 'upload changes', 'push changes'
            ],
            IntentType.GIT_STATUS: [
                'git status', 'status', 'repo status'
            ],
            IntentType.GIT_COMMIT: [
                'git commit', 'commit', 'save changes', 'commit changes'
            ],
            IntentType.GIT_ADD: [
                'git add', 'stage', 'add to staging', 'stage files'
            ],
            IntentType.GIT_BRANCH: [
                'git branch', 'branches', 'list branches', 'show branches',
                'create branch', 'new branch', 'delete branch'
            ],
            IntentType.GIT_FETCH: [
                'git fetch', 'fetch', 'fetch remote'
            ],

            # ── Git (extended) ────────────────────────────────────────
            IntentType.GIT_MERGE: [
                'merge', 'git merge', 'merge branch', 'merge into', 'merge from'
            ],
            IntentType.GIT_REBASE: [
                'rebase', 'git rebase', 'rebase on', 'rebase onto', 'rebase from'
            ],
            IntentType.GIT_STASH: [
                'stash', 'git stash', 'stash changes', 'pop stash',
                'stash pop', 'stash list', 'stash drop'
            ],
            IntentType.GIT_LOG: [
                'git log', 'show commits', 'commit history',
                'log', 'show log', 'recent commits', 'last commits',
                'show history'
            ],
            IntentType.GIT_DIFF: [
                'git diff', 'show diff', 'what changed', 'show changes',
                'diff with', 'compare', 'see changes', 'see diff'
            ],
            IntentType.GIT_CONFLICT_RESOLVE: [
                'resolve conflict', 'fix conflict', 'merge conflict',
                'conflict resolution', 'resolve merge', 'accept theirs',
                'accept ours', 'abort merge', 'resolve git conflict'
            ],

            # ── Terminal & Script execution ───────────────────────────
            IntentType.RUN_COMMAND: [
                'run', 'execute', 'run command', 'execute command',
                'terminal', 'shell', 'cmd', 'powershell'
            ],
            IntentType.RUN_SCRIPT: [
                'run script', 'execute script', 'run python', 'python',
                '.py', 'run the script', 'execute the script'
            ],
            IntentType.TERMINAL_MONITOR: [
                'monitor terminals', 'show terminals', 'terminal status',
                'what is running', 'active processes', 'background jobs',
                'running terminals', 'watch terminals', 'show processes'
            ],
            IntentType.TERMINAL_OUTPUT: [
                'show output', 'terminal output', 'what did it print',
                'show log', 'show terminal', 'output of',
                'see output', 'print output', 'display output'
            ],
            IntentType.TERMINAL_KILL: [
                'kill terminal', 'stop terminal', 'kill process',
                'stop process', 'cancel process', 'terminate',
                'abort process', 'end process', 'stop running'
            ],
            IntentType.TERMINAL_LIST: [
                'list terminals', 'all terminals', 'terminal list',
                'show all terminals', 'how many terminals'
            ],

            # ── File operations ───────────────────────────────────────
            IntentType.CREATE_FILE: [
                'create file', 'make file', 'new file', 'touch',
                'write file', 'save file'
            ],
            IntentType.READ_FILE: [
                'read file', 'show file', 'cat', 'display file',
                'view file', 'open file', 'file content'
            ],
            IntentType.DELETE_FILE: [
                'delete file', 'remove file', 'rm', 'del'
            ],
            IntentType.COPY_FILE: [
                'copy file', 'cp', 'duplicate file'
            ],
            IntentType.MOVE_FILE: [
                'move file', 'mv', 'rename file'
            ],
            IntentType.WRITE_FILE: [
                'write to file', 'update file content', 'modify file',
                'edit file content', 'overwrite file', 'write content'
            ],

            # ── Directory operations ──────────────────────────────────
            IntentType.CREATE_DIRECTORY: [
                'create folder', 'mkdir', 'make directory', 'new folder',
                'create directory'
            ],
            IntentType.DELETE_DIRECTORY: [
                'delete folder', 'rmdir', 'remove directory', 'delete directory'
            ],
            IntentType.COPY_DIRECTORY: [
                'copy folder', 'copy directory', 'duplicate folder',
                'duplicate directory', 'cp -r'
            ],

            # ── Dependency management ─────────────────────────────────
            IntentType.INSTALL_DEPENDENCY: [
                'install', 'pip install', 'install package', 'install dependency',
                'install dependencies', 'install requirements', 'pip install -r',
                'install module', 'add package', 'install lib', 'install library',
                'npm install', 'conda install', 'apt install', 'brew install'
            ],
            IntentType.CONFIGURE_ENVIRONMENT: [
                'configure environment', 'setup environment', 'create venv',
                'create virtual environment', 'activate venv', 'set env var',
                'set environment variable', 'setup python', 'configure path',
                'set path'
            ],
            IntentType.CHECK_DEPENDENCY: [
                'check dependency', 'check dependencies', 'is installed',
                'verify package', 'check if', 'dependency check', 'pip show',
                'pip list', 'check version', 'verify installation'
            ],

            # ── Search & Analysis ─────────────────────────────────────
            IntentType.SEARCH_FILE: [
                'search', 'find in file', 'grep', 'search for', 'look for',
                'find text', 'search content', 'search files'
            ],
            IntentType.ANALYZE_RESULTS: [
                'analyze', 'analyze results', 'analysis', 'evaluate',
                'assess', 'examine results', 'interpret results',
                'summarize results'
            ],
            IntentType.EXPORT_RESULTS: [
                'export', 'export results', 'save results',
                'download results', 'export to', 'save to file',
                'write results'
            ],

            # ── Plan & Execution control ──────────────────────────────
            IntentType.PLAN_EXECUTION: [
                'plan', 'create plan', 'make a plan', 'execution plan',
                'step by step', 'steps to', 'plan to', 'plan how',
                'plan the', 'what steps', 'how should i'
            ],
            IntentType.UNDO_OPERATION: [
                'undo', 'undo that', 'revert', 'revert that', 'take back',
                'undo last', 'undo change', 'reverse that'
            ],
            IntentType.REDO_OPERATION: [
                'redo', 'redo that', 'redo last', 'do again',
                'redo change', 'apply again'
            ],

            # ── Backend build & modification ──────────────────────────
            IntentType.BACKEND_BUILD: [
                'build backend', 'compile backend', 'build and compile',
                'setup build', 'build lret', 'build cirq', 'build qiskit',
                'build quest', 'build qsim', 'build cuquantum',
                'build pennylane', 'compile lret', 'compile cirq',
                'compile qiskit', 'compile quest', 'compile qsim'
            ],
            IntentType.BACKEND_CONFIGURE: [
                'configure backend', 'configure proxima', 'use backend',
                'set backend', 'switch backend', 'set as default',
                'activate backend', 'configure proxima to use',
                'use it as backend', 'set it as backend'
            ],
            IntentType.BACKEND_TEST: [
                'test backend', 'verify backend', 'test build',
                'run backend tests', 'check backend', 'validate backend',
                'verify build', 'test the build'
            ],
            IntentType.BACKEND_MODIFY: [
                'modify backend', 'change backend code', 'edit backend',
                'modify code', 'change source', 'edit source',
                'patch backend', 'update backend code',
                'modify the backend', 'change the code', 'edit the source'
            ],
            IntentType.BACKEND_LIST: [
                'list backends', 'available backends', 'show backends',
                'what backends', 'backend list', 'which backends',
                'supported backends', 'show build profiles'
            ],

            # ── System & Admin ────────────────────────────────────────
            IntentType.SYSTEM_INFO: [
                'system info', 'system information', 'python version',
                'os info', 'gpu info', 'disk space', 'memory',
                'cpu info', 'what system'
            ],
            IntentType.ADMIN_ELEVATE: [
                'admin', 'administrator', 'sudo', 'run as admin',
                'elevate', 'root', 'admin access', 'elevated',
                'admin privileges'
            ],

            # ── Web & Research ────────────────────────────────────────
            IntentType.WEB_SEARCH: [
                'search the web', 'web search', 'google',
                'look up online', 'search online', 'fetch url',
                'fetch page', 'open url', 'browse',
                'search internet', 'find online'
            ],

            # ── Query operations ──────────────────────────────────────
            # NOTE: These are handled with PRIORITY in recognize_intent
            IntentType.QUERY_LOCATION: [
                'where is', 'where did', 'where was', 'location of',
                'where is that', 'where is the', 'find the', 'path of',
                'where did you clone', 'where did you put', 'cloned to',
                'where is it', 'show me where', 'what path', 'where'
            ],
            IntentType.QUERY_STATUS: [
                'did it work', 'was it successful', 'what happened',
                'is it done', 'did you finish', 'status of'
            ],
        }
    
    def _is_query_intent(self, msg_lower: str) -> bool:
        """Check if message is a query about location/status."""
        return any(p.search(msg_lower) for p in self._QUERY_PATTERNS)
    
    def _create_query_intent(self, message: str, msg_lower: str) -> Intent:
        """Create a query intent with proper type."""
        # Determine if asking about location or status
        status_patterns = ['did it work', 'successful', 'what happened', 'is it done', 'did you finish']
        is_status = any(p in msg_lower for p in status_patterns)
        
        intent_type = IntentType.QUERY_STATUS if is_status else IntentType.QUERY_LOCATION
        
        # Phase 2: Resolve pronouns (e.g. "where is it?")
        entities = self.extract_entities(message)
        entities = self._resolve_entity_references(entities, msg_lower)
        
        intent = Intent(
            intent_type=intent_type,
            entities=entities,
            confidence=0.9,
            raw_message=message
        )
        intent.explanation = f"Query: {intent_type.name.replace('_', ' ').lower()}"
        return intent
    
    def _is_clone_intent(self, msg_lower: str, message: str) -> bool:
        """Check if message is a clone operation."""
        has_clone_keyword = 'clone' in msg_lower or 'git clone' in msg_lower
        has_url = bool(re.search(r'https?://|github\.com|gitlab\.com|bitbucket\.', message, re.IGNORECASE))
        return has_clone_keyword or has_url
    
    def _create_clone_intent(self, message: str) -> Intent:
        """Create a clone intent with proper entity extraction."""
        entities = self.extract_entities(message)
        # Phase 2: Resolve pronoun references (e.g. "clone that repo")
        entities = self._resolve_entity_references(entities, message.lower())
        
        # Ensure URL is extracted
        url_match = re.search(r'(https?://[^\s\'"<>]+)', message, re.IGNORECASE)
        if url_match:
            url = url_match.group(1).rstrip('.,;:')
            # Check if URL already in entities
            has_url = any(e.entity_type == 'url' for e in entities)
            if not has_url:
                entities.append(ExtractedEntity('url', url, 0.95, 'priority'))
        
        intent = Intent(
            intent_type=IntentType.GIT_CLONE,
            entities=entities,
            confidence=0.9,
            raw_message=message
        )
        intent.explanation = self._generate_explanation(intent)
        return intent
    
    def _is_dependency_intent(self, msg_lower: str) -> bool:
        """Check if message is a dependency installation command."""
        dep_patterns = [
            'pip install', 'pip3 install', 'npm install', 'conda install',
            'apt install', 'apt-get install', 'brew install', 'yarn add',
            'install dependency', 'install dependencies', 'install requirements',
            'install packages', 'install the dependencies',
        ]
        return any(p in msg_lower for p in dep_patterns)

    def _is_script_intent(self, msg_lower: str, message: str) -> bool:
        """Check if message is a script execution request."""
        has_run_keyword = any(kw in msg_lower for kw in [
            'run ', 'execute ', 'launch ', 'start ',
        ])
        has_script_ext = bool(re.search(
            r'\S+\.(?:py|sh|ps1|bat|js|lua)\b', message, re.IGNORECASE
        ))
        return has_run_keyword and has_script_ext

    def _is_direct_command_intent(self, msg_lower: str) -> bool:
        """Check if message starts with a direct shell command."""
        direct_prefixes = [
            'cd ', 'ls ', 'dir ', 'pwd', 'mkdir ', 'rmdir ',
            'cat ', 'echo ', 'type ', 'head ', 'tail ',
        ]
        return any(msg_lower.startswith(p) for p in direct_prefixes)

    def _infer_install_command(self, part: str) -> str:
        """Infer the install command from the message part."""
        part_lower = part.lower()
        
        # Check for specific package managers mentioned
        if 'npm' in part_lower:
            return 'npm install'
        elif 'yarn' in part_lower:
            return 'yarn install'
        elif 'pip' in part_lower:
            return 'pip install -r requirements.txt'
        elif 'poetry' in part_lower:
            return 'poetry install'
        elif 'cargo' in part_lower:
            return 'cargo build'
        else:
            # Default: try pip for Python projects
            return 'pip install -r requirements.txt'
    
    def _infer_build_command(self, part: str) -> str:
        """Infer the build command from the message part."""
        part_lower = part.lower()
        
        if 'make' in part_lower:
            return 'make'
        elif 'cmake' in part_lower:
            return 'cmake . && make'
        elif 'npm' in part_lower:
            return 'npm run build'
        elif 'cargo' in part_lower:
            return 'cargo build'
        elif 'gradle' in part_lower:
            return 'gradle build'
        elif 'maven' in part_lower or 'mvn' in part_lower:
            return 'mvn package'
        else:
            # Default for Python projects
            return 'python setup.py build'
    
    def _infer_test_command(self, part: str) -> str:
        """Infer the test command from the message part."""
        part_lower = part.lower()
        
        if 'pytest' in part_lower:
            return 'pytest'
        elif 'unittest' in part_lower:
            return 'python -m unittest discover'
        elif 'npm' in part_lower:
            return 'npm test'
        elif 'cargo' in part_lower:
            return 'cargo test'
        else:
            # Default: pytest
            return 'pytest'

    def set_llm_router(self, router):
        """Set the LLM router for LLM-assisted parsing."""
        self._llm_router = router
    
    def get_context(self) -> SessionContext:
        """Get the current session context."""
        return self._context
    
    def set_current_directory(self, path: str):
        """Update the current directory in context."""
        with self._lock:
            self._context.current_directory = path
            self._context.add_path(path)
    
    def extract_entities(self, message: str) -> List[ExtractedEntity]:
        """Extract entities from a message using regex patterns.

        Extracts paths, branches, URLs, scripts, commands, packages,
        script_paths, environments, line ranges, process IDs, and
        context-aware quoted content (Phase 2 additions).
        """
        entities: List[ExtractedEntity] = []
        seen_values: set = set()  # de-duplicate identical values

        def _add(entity_type: str, value: str, confidence: float, source: str = 'regex'):
            key = (entity_type, value)
            if key not in seen_values:
                seen_values.add(key)
                entities.append(ExtractedEntity(entity_type, value, confidence, source))

        # ── Paths ─────────────────────────────────────────────────────
        for pattern in self._patterns['windows_path']:
            for match in pattern.finditer(message):
                _add('path', match.group(1).strip().rstrip('.,;:'), 0.9)

        for pattern in self._patterns['unix_path']:
            for match in pattern.finditer(message):
                _add('path', match.group(1).strip().rstrip('.,;:'), 0.9)

        # ── Branches ──────────────────────────────────────────────────
        for pattern in self._patterns['branch']:
            for match in pattern.finditer(message):
                branch = match.group(1).strip()
                if (len(branch) >= 3 and
                    branch[0].isalpha() and
                    not re.match(r'^\d+[\.]?$', branch) and
                    not re.match(r'^[\d\.\-]+$', branch) and
                    sum(c.isdigit() for c in branch) < len(branch) // 2 and
                    branch.lower() not in self._BRANCH_STOPWORDS):
                    _add('branch', branch, 0.85)

        # ── URLs ──────────────────────────────────────────────────────
        for pattern in self._patterns['github_url']:
            for match in pattern.finditer(message):
                _add('url', match.group(1).strip(), 0.95)

        for pattern in self._patterns['git_url']:
            for match in pattern.finditer(message):
                _add('url', match.group(1).strip(), 0.95)

        # ── Scripts (legacy) ──────────────────────────────────────────
        for pattern in self._patterns['python_script']:
            for match in pattern.finditer(message):
                _add('script', match.group(1).strip(), 0.9)

        for pattern in self._patterns['script']:
            for match in pattern.finditer(message):
                _add('script', match.group(1).strip(), 0.9)

        # ── Quoted commands ───────────────────────────────────────────
        for pattern in self._patterns['quoted_command']:
            for match in pattern.finditer(message):
                cmd = match.group(1).strip()
                if len(cmd) > 2:
                    _add('command', cmd, 0.8)

        # ── Directory names from navigation context ───────────────────
        for pattern in self._patterns['dirname']:
            m = pattern.search(message)
            if m:
                dirname = m.group(1).strip()
                if dirname.lower() not in self._DIRNAME_STOPWORDS:
                    _add('dirname', dirname, 0.7)

        # ════════════════════════════════════════════════════════════════
        # Phase 2 entity extractors
        # ════════════════════════════════════════════════════════════════

        # ── Package names ─────────────────────────────────────────────
        for pattern in self._patterns['package']:
            for match in pattern.finditer(message):
                raw = match.group(1).strip()
                # Split on whitespace — each token is a separate package
                for pkg in raw.split():
                    pkg = pkg.strip().rstrip('.,;:')
                    if pkg and len(pkg) >= 2:
                        _add('package', pkg, 0.9)

        # ── Script file paths (more specific than generic 'script') ───
        for pattern in self._patterns['script_path']:
            for match in pattern.finditer(message):
                sp = match.group(1).strip().rstrip('.,;:')
                if sp and len(sp) >= 4:  # at least "a.py"
                    _add('script_path', sp, 0.9)

        # ── Environment names ─────────────────────────────────────────
        for pattern in self._patterns['environment']:
            for match in pattern.finditer(message):
                env_name = (
                    match.group(1).strip()
                    if match.lastindex and match.lastindex >= 1
                    else match.group(0).strip()
                )
                if env_name:
                    _add('environment', env_name, 0.85)

        # ── Line ranges ───────────────────────────────────────────────
        # Process range patterns (2-group) first, then single-line (1-group).
        # Track matched spans to avoid the single-line pattern re-matching
        # digits that are already part of a range.
        _lr_spans: list = []  # (start, end) character spans already consumed
        for pattern in self._patterns['line_range']:
            for match in pattern.finditer(message):
                # Skip if this match overlaps an already-consumed span
                ms, me = match.span()
                if any(s <= ms < e or s < me <= e for s, e in _lr_spans):
                    continue
                _lr_spans.append((ms, me))
                groups = match.groups()
                if len(groups) == 2 and groups[1] is not None:
                    _add('line_range', f"{groups[0]}-{groups[1]}", 0.9)
                elif len(groups) >= 1:
                    _add('line_range', groups[0], 0.9)

        # ── Process / terminal identifiers ────────────────────────────
        for pattern in self._patterns['process_id']:
            for match in pattern.finditer(message):
                if match.lastindex:
                    _add('process_id', match.group(1).strip(), 0.85)
                else:
                    # Ordinal reference like "the build terminal"
                    _add('process_id', match.group(0).strip(), 0.7)

        # ── Context-aware quoted content ──────────────────────────────
        # Classify based on which sub-pattern matched
        qc_patterns = self._patterns['quoted_context']
        # idx 0 → path, idx 1 → name, idx 2 → command, idx 3 → package
        qc_types = ['path', 'name', 'command', 'package']
        for idx, pattern in enumerate(qc_patterns):
            for match in pattern.finditer(message):
                val = match.group(1).strip()
                if val and len(val) > 1:
                    _add(qc_types[idx], val, 0.85)

        return entities

    # ── Phase 2, Step 2.4: Pronoun / reference resolution ─────────────

    # Pronoun patterns that trigger context resolution
    _PRONOUN_PATTERNS = frozenset({
        'it', 'that', 'there',
        'the repo', 'the repository', 'that repo', 'repo',
        'the script', 'that script',
        'the result', 'the output', 'result', 'output',
        'the branch', 'that branch',
        'the backend', 'that backend',
        'the directory', 'that directory', 'the folder', 'that folder',
        'back', 'previous directory', 'previous folder',
    })

    def _resolve_entity_references(
        self,
        entities: List[ExtractedEntity],
        msg_lower: str,
    ) -> List[ExtractedEntity]:
        """Post-process entities to resolve pronouns to concrete values.

        For every entity whose ``value`` matches a known pronoun pattern,
        attempt to resolve it via ``SessionContext.resolve_reference()``.

        * If resolution succeeds → replace ``value``, set ``source='context'``
        * If resolution fails   → keep original, lower confidence to 0.3
        """
        resolved: List[ExtractedEntity] = []
        for entity in entities:
            val_lower = entity.value.strip().lower()
            if val_lower not in self._PRONOUN_PATTERNS:
                resolved.append(entity)
                continue

            # Extract the verb immediately preceding the pronoun
            verb_context = self._extract_verb_context(msg_lower, val_lower)

            concrete = self._context.resolve_reference(val_lower, verb_context)
            if concrete:
                resolved.append(ExtractedEntity(
                    entity_type=entity.entity_type,
                    value=concrete,
                    confidence=entity.confidence,
                    source='context',
                ))
            else:
                # Keep the entity but lower confidence
                resolved.append(ExtractedEntity(
                    entity_type=entity.entity_type,
                    value=entity.value,
                    confidence=0.3,
                    source=entity.source,
                ))
        return resolved

    @staticmethod
    def _extract_verb_context(msg_lower: str, pronoun: str) -> Optional[str]:
        """Extract the verb that immediately precedes *pronoun* in the message.

        Returns the word(s) before the pronoun that look like a verb phrase,
        or ``None`` if no clear verb is found.  For example::

            "run it"       → "run"
            "execute that" → "execute"
            "show it"      → "show"
        """
        idx = msg_lower.find(pronoun)
        if idx <= 0:
            return None
        preceding = msg_lower[:idx].rstrip()
        if not preceding:
            return None
        # Take the last 1-2 words before the pronoun as verb context
        words = preceding.split()
        verb = ' '.join(words[-2:]) if len(words) >= 2 else words[-1]
        return verb

    def recognize_intent(self, message: str) -> Intent:
        """Recognize the user's intent from natural language.

        Uses a 5-layer pipeline where each layer is a fallback if the
        previous one fails or produces low confidence:

        Layer 1 — High-Priority Pattern Matching (no LLM needed)
        Layer 2 — Multi-Step Detection
        Layer 3 — Keyword Scoring
        Layer 4 — LLM-Assisted Classification (optional, model-agnostic)
        Layer 5 — Context-Based Inference
        """
        msg_lower = message.lower()

        # ══════════════════════════════════════════════════════════════
        # Layer 1: High-Priority Pattern Matching (no LLM needed)
        # Unambiguous intents determined from text patterns alone.
        # ══════════════════════════════════════════════════════════════
        layer1 = self._layer1_pattern_match(message, msg_lower)
        if layer1 is not None:
            return layer1

        # ══════════════════════════════════════════════════════════════
        # Layer 2: Multi-Step Detection
        # ══════════════════════════════════════════════════════════════
        layer2 = self._layer2_multi_step(message, msg_lower)
        if layer2 is not None:
            return layer2

        # ══════════════════════════════════════════════════════════════
        # Layer 3: Keyword Scoring
        # ══════════════════════════════════════════════════════════════
        entities = self.extract_entities(message)
        entities = self._resolve_entity_references(entities, msg_lower)

        layer3_intent, layer3_confidence, scored_candidates = self._layer3_keyword_scoring(
            msg_lower, entities
        )

        if layer3_confidence >= 0.5:
            intent = Intent(
                intent_type=layer3_intent,
                entities=entities,
                confidence=layer3_confidence,
                raw_message=message,
            )
            intent.explanation = self._generate_explanation(intent)
            self._enhance_with_context(intent)
            return intent

        # ══════════════════════════════════════════════════════════════
        # Layer 4: LLM-Assisted Classification (optional)
        # Sends a multiple-choice question to the integrated model.
        # ══════════════════════════════════════════════════════════════
        layer4 = self._layer4_llm_classification(
            message, msg_lower, entities, scored_candidates
        )
        if layer4 is not None:
            return layer4

        # ══════════════════════════════════════════════════════════════
        # Layer 5: Context-Based Inference
        # Uses SessionContext to infer from conversation flow.
        # ══════════════════════════════════════════════════════════════
        layer5 = self._layer5_context_inference(message, msg_lower, entities)
        if layer5 is not None:
            return layer5

        # All layers failed — return UNKNOWN
        intent = Intent(
            intent_type=IntentType.UNKNOWN,
            entities=entities,
            confidence=0.0,
            raw_message=message,
        )
        intent.explanation = "Could not determine intent"
        return intent

    # ── Layer 1: High-Priority Pattern Matching ───────────────────────

    def _layer1_pattern_match(
        self, message: str, msg_lower: str
    ) -> Optional[Intent]:
        """Layer 1 — intents unambiguously determined from text patterns.

        If the message contains multi-step separators, we skip all
        non-query patterns so Layer 2 can handle the full sequence.
        """

        # Queries — ALWAYS first (never part of multi-step)
        if self._is_query_intent(msg_lower):
            return self._create_query_intent(message, msg_lower)

        # Guard: if the message looks like a multi-step operation,
        # skip the remaining Layer 1 checks and let Layer 2 handle it.
        _multi_seps = (' then ', ' and then ', ' after that ', ' next ', ' finally ')
        if any(sep in msg_lower for sep in _multi_seps):
            return None
        if re.search(r'(?:^|\n)\s*\d+[\.\)]\s+', message):
            return None
        if ';' in message and len(message) > 20:
            return None

        # Clone operations — URL is distinctive
        if self._is_clone_intent(msg_lower, message):
            return self._create_clone_intent(message)

        # Script execution — run/execute + file extension
        if self._is_script_intent(msg_lower, message):
            entities = self.extract_entities(message)
            entities = self._resolve_entity_references(entities, msg_lower)
            intent = Intent(
                intent_type=IntentType.RUN_SCRIPT,
                entities=entities,
                confidence=0.9,
                raw_message=message,
            )
            intent.explanation = self._generate_explanation(intent)
            return intent

        # Dependency installation — pip/npm/conda install
        if self._is_dependency_intent(msg_lower):
            entities = self.extract_entities(message)
            entities = self._resolve_entity_references(entities, msg_lower)
            intent = Intent(
                intent_type=IntentType.INSTALL_DEPENDENCY,
                entities=entities,
                confidence=0.9,
                raw_message=message,
            )
            intent.explanation = self._generate_explanation(intent)
            return intent

        # Direct shell commands (cd, ls, pwd, mkdir, ...)
        if self._is_direct_command_intent(msg_lower):
            entities = self.extract_entities(message)
            entities = self._resolve_entity_references(entities, msg_lower)
            # Determine specific intent from the prefix
            if msg_lower.startswith('cd '):
                it = IntentType.NAVIGATE_DIRECTORY
            elif msg_lower.startswith(('ls ', 'dir ')):
                it = IntentType.LIST_DIRECTORY
            elif msg_lower.startswith('pwd'):
                it = IntentType.SHOW_CURRENT_DIR
            elif msg_lower.startswith('mkdir '):
                it = IntentType.CREATE_DIRECTORY
            elif msg_lower.startswith('rmdir '):
                it = IntentType.DELETE_DIRECTORY
            else:
                it = IntentType.RUN_COMMAND
            intent = Intent(
                intent_type=it,
                entities=entities,
                confidence=0.9,
                raw_message=message,
            )
            intent.explanation = self._generate_explanation(intent)
            return intent

        return None

    # ── Layer 2: Multi-Step Detection ─────────────────────────────────

    def _layer2_multi_step(
        self, message: str, msg_lower: str
    ) -> Optional[Intent]:
        """Layer 2 — detect multi-step operations."""
        multi_step_separators = [
            ' then ', ' and then ', ' after that ', ' next ', ' finally ',
        ]
        is_multi_step = any(sep in msg_lower for sep in multi_step_separators)
        has_numbered_list = bool(
            re.search(r'(?:^|\n)\s*\d+[\.\)]\s+', message)
        )
        # Also detect semicolons separating distinct commands
        has_semicolons = ';' in message and len(message) > 20

        if is_multi_step or has_numbered_list or has_semicolons:
            return self._parse_multi_step_intent(message)
        return None

    # ── Layer 3: Keyword Scoring ──────────────────────────────────────

    def _layer3_keyword_scoring(
        self,
        msg_lower: str,
        entities: List[ExtractedEntity],
    ) -> Tuple[IntentType, float, List[Tuple[IntentType, float]]]:
        """Layer 3 — score every intent type against the message.

        Returns (best_intent, best_confidence, sorted_candidates).
        ``sorted_candidates`` is a list of (IntentType, confidence) pairs
        sorted descending by confidence for use in Layer 4.
        """
        scored: List[Tuple[IntentType, float]] = []

        for intent_type, keywords in self._intent_keywords.items():
            score = 0.0
            matches = 0
            for keyword in keywords:
                if keyword in msg_lower:
                    matches += 1
                    score += len(keyword) / 20.0
            if matches > 0:
                confidence = min(0.5 + (score * 0.5), 0.95)
                scored.append((intent_type, confidence))

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored:
            return scored[0][0], scored[0][1], scored
        return IntentType.UNKNOWN, 0.0, scored

    # ── Layer 4: LLM-Assisted Classification ──────────────────────────

    _INTENT_DESCRIPTIONS: Dict[IntentType, str] = {
        IntentType.NAVIGATE_DIRECTORY: "Navigate to a directory",
        IntentType.LIST_DIRECTORY: "List directory contents",
        IntentType.SHOW_CURRENT_DIR: "Show current working directory",
        IntentType.GIT_CHECKOUT: "Switch git branch",
        IntentType.GIT_CLONE: "Clone a git repository",
        IntentType.GIT_PULL: "Pull latest changes",
        IntentType.GIT_PUSH: "Push commits to remote",
        IntentType.GIT_STATUS: "Check git status",
        IntentType.GIT_COMMIT: "Commit staged changes",
        IntentType.GIT_ADD: "Stage files for commit",
        IntentType.GIT_BRANCH: "List or create branches",
        IntentType.GIT_FETCH: "Fetch from remote",
        IntentType.GIT_MERGE: "Merge branches",
        IntentType.GIT_REBASE: "Rebase branch",
        IntentType.GIT_STASH: "Stash or pop changes",
        IntentType.GIT_LOG: "View commit history",
        IntentType.GIT_DIFF: "Show diffs / changes",
        IntentType.GIT_CONFLICT_RESOLVE: "Resolve merge conflicts",
        IntentType.CREATE_FILE: "Create a new file",
        IntentType.READ_FILE: "Read file contents",
        IntentType.WRITE_FILE: "Write / edit a file",
        IntentType.DELETE_FILE: "Delete a file",
        IntentType.COPY_FILE: "Copy a file",
        IntentType.MOVE_FILE: "Move / rename a file",
        IntentType.CREATE_DIRECTORY: "Create a directory",
        IntentType.DELETE_DIRECTORY: "Delete a directory",
        IntentType.COPY_DIRECTORY: "Copy a directory",
        IntentType.RUN_COMMAND: "Run a shell command",
        IntentType.RUN_SCRIPT: "Execute a script file",
        IntentType.TERMINAL_MONITOR: "Monitor terminal sessions",
        IntentType.TERMINAL_KILL: "Kill a terminal process",
        IntentType.TERMINAL_OUTPUT: "Show terminal output",
        IntentType.TERMINAL_LIST: "List terminals",
        IntentType.QUERY_LOCATION: "Find where something is",
        IntentType.QUERY_STATUS: "Check operation status",
        IntentType.INSTALL_DEPENDENCY: "Install packages / dependencies",
        IntentType.CONFIGURE_ENVIRONMENT: "Configure environment / venv",
        IntentType.CHECK_DEPENDENCY: "Check if dependency is installed",
        IntentType.SEARCH_FILE: "Search file contents",
        IntentType.ANALYZE_RESULTS: "Analyze execution results",
        IntentType.EXPORT_RESULTS: "Export results to file",
        IntentType.PLAN_EXECUTION: "Create an execution plan",
        IntentType.UNDO_OPERATION: "Undo last operation",
        IntentType.REDO_OPERATION: "Redo undone operation",
        IntentType.BACKEND_BUILD: "Build a quantum backend",
        IntentType.BACKEND_CONFIGURE: "Configure a backend",
        IntentType.BACKEND_TEST: "Test a backend",
        IntentType.BACKEND_MODIFY: "Modify backend source code",
        IntentType.BACKEND_LIST: "List available backends",
        IntentType.SYSTEM_INFO: "Get system information",
        IntentType.ADMIN_ELEVATE: "Request admin privileges",
        IntentType.WEB_SEARCH: "Search the web / fetch URL",
        IntentType.MULTI_STEP: "Execute multiple steps",
    }

    def _layer4_llm_classification(
        self,
        message: str,
        msg_lower: str,
        entities: List[ExtractedEntity],
        scored_candidates: List[Tuple[IntentType, float]],
    ) -> Optional[Intent]:
        """Layer 4 — ask the integrated model via a multiple-choice prompt.

        This works with ANY model (including small local models) because
        it uses a simple numbered-choice format instead of JSON extraction.
        """
        if not self._llm_router:
            return None

        # Take top 4 candidates from Layer 3 (or fill with common intents)
        candidates: List[IntentType] = []
        for it, _ in scored_candidates[:4]:
            candidates.append(it)

        # Pad with common intents if fewer than 4 candidates
        common_fallbacks = [
            IntentType.RUN_COMMAND, IntentType.NAVIGATE_DIRECTORY,
            IntentType.RUN_SCRIPT, IntentType.SEARCH_FILE,
        ]
        for fb in common_fallbacks:
            if len(candidates) >= 4:
                break
            if fb not in candidates:
                candidates.append(fb)

        # Build the multiple-choice prompt
        options = []
        for i, it in enumerate(candidates[:4], 1):
            desc = self._INTENT_DESCRIPTIONS.get(it, it.name.replace('_', ' ').lower())
            options.append(f"{i}. {it.name}: {desc}")
        options.append("5. None of the above")

        prompt = (
            f"The user said: '{message}'\n"
            f"Which action best matches? Pick ONE number:\n"
            + "\n".join(options) + "\n"
            "Answer with just the number:"
        )

        try:
            # Use the LLM router to get a response
            from proxima.intelligence.llm_router import LLMRequest
            request = LLMRequest(
                prompt=prompt,
                system_prompt="You are a classifier. Reply with a single digit (1-5) only.",
                temperature=0.0,
                max_tokens=8,
            )
            response = self._llm_router.route(request)
            if response and hasattr(response, 'text') and response.text:
                text = response.text.strip()
                # Parse the first digit found
                digit_match = re.search(r'[1-5]', text)
                if digit_match:
                    choice = int(digit_match.group())
                    if 1 <= choice <= len(candidates):
                        chosen = candidates[choice - 1]
                        intent = Intent(
                            intent_type=chosen,
                            entities=entities,
                            confidence=0.7,
                            raw_message=message,
                        )
                        intent.explanation = self._generate_explanation(intent)
                        self._enhance_with_context(intent)
                        return intent
                    # choice == 5 or out of range → fall through
        except Exception:
            pass  # LLM unavailable or error — fall through

        return None

    # ── Layer 5: Context-Based Inference ──────────────────────────────

    def _layer5_context_inference(
        self,
        message: str,
        msg_lower: str,
        entities: List[ExtractedEntity],
    ) -> Optional[Intent]:
        """Layer 5 — infer intent from SessionContext conversation flow."""
        ctx = self._context

        # "build it" / "test it" after a clone → BACKEND_BUILD / BACKEND_TEST
        if 'build' in msg_lower and ('it' in msg_lower or 'that' in msg_lower):
            # Check if last operation was a clone of a known backend
            if ctx.last_cloned_repo:
                _known_backends = {
                    'lret', 'cirq', 'qiskit', 'quest', 'qsim',
                    'cuquantum', 'pennylane',
                }
                repo_lower = ctx.last_cloned_repo.lower()
                if any(b in repo_lower for b in _known_backends):
                    intent = Intent(
                        intent_type=IntentType.BACKEND_BUILD,
                        entities=entities,
                        confidence=0.6,
                        raw_message=message,
                    )
                    intent.explanation = "Build the recently cloned backend"
                    return intent
                # Not a known backend → generic build command
                intent = Intent(
                    intent_type=IntentType.RUN_COMMAND,
                    entities=entities,
                    confidence=0.55,
                    raw_message=message,
                )
                intent.explanation = "Run build command on cloned repository"
                return intent

        if 'test' in msg_lower and ('it' in msg_lower or 'that' in msg_lower):
            if ctx.last_built_backend or ctx.last_cloned_repo:
                intent = Intent(
                    intent_type=IntentType.BACKEND_TEST,
                    entities=entities,
                    confidence=0.55,
                    raw_message=message,
                )
                intent.explanation = "Test the recently built backend"
                return intent

        # "run it" → RUN_SCRIPT using last_script_executed
        if 'run' in msg_lower and ('it' in msg_lower or 'that' in msg_lower):
            if ctx.last_script_executed:
                entities_copy = list(entities)
                entities_copy.append(ExtractedEntity(
                    'script_path', ctx.last_script_executed, 0.7, 'context'
                ))
                intent = Intent(
                    intent_type=IntentType.RUN_SCRIPT,
                    entities=entities_copy,
                    confidence=0.6,
                    raw_message=message,
                )
                intent.explanation = f"Re-run {ctx.last_script_executed}"
                return intent

        # "go back" / "previous directory" → NAVIGATE_DIRECTORY with pop
        if 'go back' in msg_lower or 'previous directory' in msg_lower:
            prev = ctx.pop_directory()
            if prev:
                entities_copy = list(entities)
                entities_copy.append(ExtractedEntity(
                    'path', prev, 0.8, 'context'
                ))
                intent = Intent(
                    intent_type=IntentType.NAVIGATE_DIRECTORY,
                    entities=entities_copy,
                    confidence=0.7,
                    raw_message=message,
                )
                intent.explanation = f"Navigate back to {prev}"
                return intent

        # "install the dependencies" / "install deps" after cloning
        if 'install' in msg_lower and ('dependencies' in msg_lower or 'deps' in msg_lower or 'requirements' in msg_lower):
            if ctx.last_cloned_repo:
                entities_copy = list(entities)
                entities_copy.append(ExtractedEntity(
                    'path', ctx.last_cloned_repo, 0.7, 'context'
                ))
                intent = Intent(
                    intent_type=IntentType.INSTALL_DEPENDENCY,
                    entities=entities_copy,
                    confidence=0.6,
                    raw_message=message,
                )
                intent.explanation = "Install dependencies for cloned repo"
                return intent

        return None
    
    def _parse_multi_step_intent(self, message: str) -> Intent:
        """Parse a multi-step operation from natural language.
        
        Handles messages like:
        - "switch to X branch then go inside Y directory and run Z script"
        - "clone repo to folder then build it"
        - "1. Clone repo 2. Switch branch 3. Build"
        """
        parts = []
        
        # First, check if this is a numbered list format (1. ... 2. ... or 1) ... 2) ...)
        numbered_pattern = r'(?:^|\n)\s*\d+[\.):]\s+'
        if re.search(numbered_pattern, message):
            # Split by numbered list items and filter out preamble
            raw_parts = re.split(r'(?:^|\n)\s*\d+[\.):]\s+', message)
            for p in raw_parts:
                p = p.strip()
                # Skip empty parts and parts that look like preamble (no action keywords)
                if p and len(p) > 3:
                    # Clean any trailing numbers for next list item
                    p = re.sub(r'\s*\d+[\.):]?\s*$', '', p).strip()
                    if p:
                        parts.append(p)
        else:
            # Split by separators like "then", "after that", etc.
            separators = r'\s+(?:then|and then|after that|next|finally)\s+'
            raw_parts = re.split(separators, message, flags=re.IGNORECASE)
            parts = [p.strip() for p in raw_parts if p.strip()]
        
        # Parse each part as a sub-intent
        sub_intents = []
        all_entities = []
        
        for part in parts:
            if not part:
                continue
            
            part_lower = part.lower()
            
            # Extract entities from this part
            entities = self.extract_entities(part)
            # Phase 2: Resolve pronouns in each sub-step (e.g. "build it")
            entities = self._resolve_entity_references(entities, part_lower)
            all_entities.extend(entities)
            
            # Check if this part contains a URL (strong indicator of clone)
            url_match = re.search(r'(https?://[^\s\'"<>]+)', part, re.IGNORECASE)
            has_url = url_match is not None
            has_clone_word = 'clone' in part_lower
            
            # PRIORITY 1: Clone operation (has URL OR 'clone' keyword)
            if has_url or has_clone_word:
                if url_match:
                    url = url_match.group(1).rstrip('.,;:')
                    # Ensure URL entity is present
                    if not any(e.entity_type == 'url' for e in entities):
                        entities.append(ExtractedEntity('url', url, 0.95, 'priority'))
                sub_intents.append({
                    'type': IntentType.GIT_CLONE,
                    'part': part,
                    'entities': entities
                })
                continue
            
            # PRIORITY 2: Git checkout (branch switching)
            # Look for explicit branch-related keywords
            if ('switch' in part_lower or 'checkout' in part_lower) and 'branch' in part_lower:
                sub_intents.append({
                    'type': IntentType.GIT_CHECKOUT,
                    'part': part,
                    'entities': entities
                })
                continue
            
            # Also check if "switch to X" where X looks like a branch name
            if 'switch to' in part_lower or 'checkout' in part_lower:
                # Check if we have a branch entity or a hyphenated name (branch-like)
                branch_entity = any(e.entity_type == 'branch' for e in entities)
                has_hyphenated = bool(re.search(r'(?:switch|checkout)\s+(?:to\s+)?([a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)+)', part_lower))
                if branch_entity or has_hyphenated:
                    sub_intents.append({
                        'type': IntentType.GIT_CHECKOUT,
                        'part': part,
                        'entities': entities
                    })
                    continue
            
            # PRIORITY 3: Navigation (go inside, go to directory)
            if any(kw in part_lower for kw in ['go inside', 'go into', 'go to', 'inside', 'enter', 'cd ']):
                # Make sure this isn't actually a URL reference
                if not has_url:
                    sub_intents.append({
                        'type': IntentType.NAVIGATE_DIRECTORY,
                        'part': part,
                        'entities': entities
                    })
                    continue
            
            # PRIORITY 4: Script execution
            if 'run' in part_lower or 'execute' in part_lower or '.py' in part_lower:
                sub_intents.append({
                    'type': IntentType.RUN_SCRIPT,
                    'part': part,
                    'entities': entities
                })
                continue
            
            # PRIORITY 5: Install/compile/test/configure commands
            if any(kw in part_lower for kw in ['install', 'dependencies', 'pip', 'npm', 'requirements']):
                entities.append(ExtractedEntity('command', self._infer_install_command(part), 0.8, 'inferred'))
                sub_intents.append({
                    'type': IntentType.RUN_COMMAND,
                    'part': part,
                    'entities': entities
                })
                continue
            
            if any(kw in part_lower for kw in ['compile', 'build', 'make']):
                entities.append(ExtractedEntity('command', self._infer_build_command(part), 0.8, 'inferred'))
                sub_intents.append({
                    'type': IntentType.RUN_COMMAND,
                    'part': part,
                    'entities': entities
                })
                continue
            
            if any(kw in part_lower for kw in ['test', 'pytest', 'unittest']):
                entities.append(ExtractedEntity('command', self._infer_test_command(part), 0.8, 'inferred'))
                sub_intents.append({
                    'type': IntentType.RUN_COMMAND,
                    'part': part,
                    'entities': entities
                })
                continue
            
            if any(kw in part_lower for kw in ['configure', 'setup', 'config']):
                sub_intents.append({
                    'type': IntentType.RUN_COMMAND,
                    'part': part,
                    'entities': entities
                })
                continue
            
            # Fallback: keyword-based detection
            sub_intent_type = IntentType.UNKNOWN
            best_score = 0.0
            
            for intent_type, keywords in self._intent_keywords.items():
                for keyword in keywords:
                    if keyword in part_lower:
                        score = len(keyword) / 20.0
                        if score > best_score:
                            best_score = score
                            sub_intent_type = intent_type
            
            if sub_intent_type != IntentType.UNKNOWN:
                sub_intents.append({
                    'type': sub_intent_type,
                    'part': part,
                    'entities': entities
                })
        
        # Convert raw sub-intent dicts to proper Intent objects
        parsed_sub_intents: List[Intent] = []
        for sub in sub_intents:
            sub_intent = Intent(
                intent_type=sub['type'],
                entities=sub['entities'],
                confidence=0.8,
                raw_message=sub['part'],
            )
            sub_intent.explanation = self._generate_explanation(sub_intent)
            parsed_sub_intents.append(sub_intent)

        # Create multi-step intent with sub_intents field
        intent = Intent(
            intent_type=IntentType.MULTI_STEP,
            entities=all_entities,
            confidence=0.8 if parsed_sub_intents else 0.3,
            raw_message=message,
            sub_intents=parsed_sub_intents,
        )
        
        # Generate explanation
        step_descriptions = []
        for i, sub in enumerate(parsed_sub_intents, 1):
            step_descriptions.append(f"Step {i}: {sub.intent_type.name.replace('_', ' ').title()}")
        
        intent.explanation = f"Multi-step operation: {' → '.join(step_descriptions)}"
        
        return intent
    
    # Explanations for each intent type — built once as a class constant.
    _INTENT_EXPLANATIONS: Dict[IntentType, str] = {
        # Navigation
        IntentType.NAVIGATE_DIRECTORY: "Navigate to directory",
        IntentType.LIST_DIRECTORY: "List directory contents",
        IntentType.SHOW_CURRENT_DIR: "Show current directory",
        # Git basic
        IntentType.GIT_CHECKOUT: "Switch to git branch",
        IntentType.GIT_CLONE: "Clone repository",
        IntentType.GIT_PULL: "Pull changes from remote",
        IntentType.GIT_PUSH: "Push changes to remote",
        IntentType.GIT_STATUS: "Show git status",
        IntentType.GIT_COMMIT: "Commit changes",
        IntentType.GIT_ADD: "Stage files for commit",
        IntentType.GIT_BRANCH: "Git branch operation",
        IntentType.GIT_FETCH: "Fetch from remote",
        # Git extended
        IntentType.GIT_MERGE: "Merge branches",
        IntentType.GIT_REBASE: "Rebase branch",
        IntentType.GIT_STASH: "Stash changes",
        IntentType.GIT_LOG: "Show commit history",
        IntentType.GIT_DIFF: "Show diff",
        IntentType.GIT_CONFLICT_RESOLVE: "Resolve merge conflict",
        # Terminal & Script
        IntentType.RUN_COMMAND: "Run terminal command",
        IntentType.RUN_SCRIPT: "Run script",
        IntentType.TERMINAL_MONITOR: "Monitor terminals",
        IntentType.TERMINAL_KILL: "Kill terminal process",
        IntentType.TERMINAL_OUTPUT: "Show terminal output",
        IntentType.TERMINAL_LIST: "List active terminals",
        # File operations
        IntentType.CREATE_FILE: "Create file",
        IntentType.READ_FILE: "Read file",
        IntentType.WRITE_FILE: "Write file",
        IntentType.DELETE_FILE: "Delete file",
        IntentType.COPY_FILE: "Copy file",
        IntentType.MOVE_FILE: "Move file",
        # Directory operations
        IntentType.CREATE_DIRECTORY: "Create directory",
        IntentType.DELETE_DIRECTORY: "Delete directory",
        IntentType.COPY_DIRECTORY: "Copy directory",
        # Dependency management
        IntentType.INSTALL_DEPENDENCY: "Install dependency",
        IntentType.CONFIGURE_ENVIRONMENT: "Configure environment",
        IntentType.CHECK_DEPENDENCY: "Check dependency",
        # Search & Analysis
        IntentType.SEARCH_FILE: "Search files",
        IntentType.ANALYZE_RESULTS: "Analyze results",
        IntentType.EXPORT_RESULTS: "Export results",
        # Plan & Execution control
        IntentType.PLAN_EXECUTION: "Create execution plan",
        IntentType.UNDO_OPERATION: "Undo last operation",
        IntentType.REDO_OPERATION: "Redo operation",
        # Backend
        IntentType.BACKEND_BUILD: "Build backend",
        IntentType.BACKEND_CONFIGURE: "Configure backend",
        IntentType.BACKEND_TEST: "Test backend",
        IntentType.BACKEND_MODIFY: "Modify backend code",
        IntentType.BACKEND_LIST: "List available backends",
        # System & Admin
        IntentType.SYSTEM_INFO: "Show system information",
        IntentType.ADMIN_ELEVATE: "Elevate privileges",
        # Web
        IntentType.WEB_SEARCH: "Search the web",
        # Query
        IntentType.QUERY_LOCATION: "Query location",
        IntentType.QUERY_STATUS: "Query status",
        # Meta
        IntentType.MULTI_STEP: "Multi-step operation",
        IntentType.UNKNOWN: "Unknown operation",
    }

    def _generate_explanation(self, intent: Intent) -> str:
        """Generate a human-readable explanation of the intent."""
        base = self._INTENT_EXPLANATIONS.get(intent.intent_type, "Operation")
        
        # Add entity details
        details = []
        path = intent.get_entity('path') or intent.get_entity('dirname')
        branch = intent.get_entity('branch')
        script = intent.get_entity('script')
        
        if path:
            details.append(f"path: {path}")
        if branch:
            details.append(f"branch: {branch}")
        if script:
            details.append(f"script: {script}")
        
        if details:
            return f"{base} ({', '.join(details)})"
        return base
    
    def _enhance_with_context(self, intent: Intent):
        """Enhance intent with context from previous messages."""
        # If we need a path but don't have one, try to get from context
        if intent.intent_type in [IntentType.NAVIGATE_DIRECTORY, IntentType.GIT_CHECKOUT,
                                   IntentType.LIST_DIRECTORY, IntentType.RUN_SCRIPT]:
            if not intent.get_entity('path'):
                # Check if we have a dirname that could be resolved
                dirname = intent.get_entity('dirname')
                if dirname and self._context.last_mentioned_paths:
                    # Try to find a matching path in context
                    for ctx_path in self._context.last_mentioned_paths:
                        if dirname.lower() in ctx_path.lower():
                            intent.entities.append(
                                ExtractedEntity('path', ctx_path, 0.6, 'context')
                            )
                            break
    
    def resolve_path(self, path_or_name: str) -> str:
        """Resolve a path or directory name to an absolute path.
        
        Handles:
        - Absolute paths
        - Relative paths
        - Directory names (resolved against context)
        """
        if not path_or_name:
            return self._context.current_directory
        
        # Expand user home
        path_or_name = os.path.expanduser(path_or_name)
        path_or_name = os.path.expandvars(path_or_name)
        
        # Check if it's already absolute
        if os.path.isabs(path_or_name):
            return path_or_name
        
        # Try to resolve against current directory
        resolved = os.path.join(self._context.current_directory, path_or_name)
        if os.path.exists(resolved):
            return os.path.abspath(resolved)
        
        # Try to find in context paths
        for ctx_path in self._context.last_mentioned_paths:
            if path_or_name.lower() in os.path.basename(ctx_path).lower():
                return ctx_path
            # Also check subdirectories
            potential = os.path.join(ctx_path, path_or_name)
            if os.path.exists(potential):
                return os.path.abspath(potential)
        
        # Return as-is with current directory
        return os.path.abspath(os.path.join(self._context.current_directory, path_or_name))
    
    def execute_intent(self, intent: Intent) -> Tuple[bool, str]:
        """Execute a recognized intent.

        For the original core intents (navigation, git basics, scripts,
        commands, multi-step, and queries) this method carries out the
        action directly.  Newly-added intents from Phase 1 (dependency
        management, backend operations, terminal monitoring, etc.) are
        **recognized and confirmed** here but their full execution is
        delegated to the ``IntentToolBridge`` (Phase 3) or the
        ``agent_ai_assistant`` orchestration layer when available.

        Returns:
            Tuple of (success: bool, result_message: str)
        """
        try:
            if intent.intent_type == IntentType.NAVIGATE_DIRECTORY:
                return self._execute_navigate(intent)

            elif intent.intent_type == IntentType.LIST_DIRECTORY:
                return self._execute_list_directory(intent)

            elif intent.intent_type == IntentType.SHOW_CURRENT_DIR:
                return self._execute_pwd()

            elif intent.intent_type == IntentType.GIT_CHECKOUT:
                return self._execute_git_checkout(intent)

            elif intent.intent_type == IntentType.GIT_STATUS:
                return self._execute_git_status()

            elif intent.intent_type == IntentType.GIT_PULL:
                return self._execute_git_pull()

            elif intent.intent_type == IntentType.GIT_CLONE:
                return self._execute_git_clone(intent)

            elif intent.intent_type == IntentType.RUN_SCRIPT:
                return self._execute_run_script(intent)

            elif intent.intent_type == IntentType.RUN_COMMAND:
                return self._execute_run_command(intent)

            elif intent.intent_type == IntentType.MULTI_STEP:
                return self._execute_multi_step(intent)

            elif intent.intent_type == IntentType.QUERY_LOCATION:
                return self._execute_query_location(intent)

            # ------------------------------------------------------------------
            # Phase-1 recognised intents: return a structured acknowledgement
            # that includes the intent name so the upper orchestration layer
            # (agent_ai_assistant / IntentToolBridge) can dispatch them.
            # ------------------------------------------------------------------
            elif intent.intent_type in _DEFERRED_INTENTS:
                label = intent.intent_type.name.replace('_', ' ').title()
                return True, (
                    f"✅ Intent recognised: **{label}**\n"
                    f"   Confidence: {intent.confidence:.0%}\n"
                    f"   Entities: {', '.join(e.value for e in intent.entities) or 'none'}\n"
                    f"   ℹ️ Execution is handled by the agent orchestration layer."
                )

            else:
                return False, f"⚠️ Intent '{intent.intent_type.name}' recognised but not yet implemented"

        except Exception as e:
            return False, f"❌ Error executing {intent.intent_type.name}: {str(e)}"
        
        finally:
            # Update context with this operation
            self._context.update_from_intent(intent)
    
    def _execute_multi_step(self, intent: Intent) -> Tuple[bool, str]:
        """Execute a multi-step operation."""
        sub_intents = intent.sub_intents
        
        if not sub_intents:
            return False, "❌ No steps found in multi-step operation"
        
        results = []
        success_count = 0
        
        for i, sub_intent in enumerate(sub_intents, 1):
            step_type = sub_intent.intent_type
            
            results.append(f"\n**Step {i}: {step_type.name.replace('_', ' ').title()}**")
            
            try:
                # Execute based on type
                if step_type == IntentType.GIT_CLONE:
                    success, result = self._execute_git_clone(sub_intent)
                elif step_type == IntentType.GIT_CHECKOUT:
                    success, result = self._execute_git_checkout(sub_intent)
                elif step_type == IntentType.NAVIGATE_DIRECTORY:
                    success, result = self._execute_navigate(sub_intent)
                elif step_type == IntentType.RUN_SCRIPT:
                    success, result = self._execute_run_script(sub_intent)
                elif step_type == IntentType.RUN_COMMAND:
                    success, result = self._execute_run_command(sub_intent)
                elif step_type == IntentType.LIST_DIRECTORY:
                    success, result = self._execute_list_directory(sub_intent)
                elif step_type == IntentType.GIT_PULL:
                    success, result = self._execute_git_pull()
                elif step_type == IntentType.GIT_STATUS:
                    success, result = self._execute_git_status()
                else:
                    success = False
                    result = f"⚠️ Step type {step_type.name} not yet supported"
                
                if success:
                    success_count += 1
                
                results.append(result)
                
                # Update context after each step
                self._context.update_from_intent(sub_intent)
                
            except Exception as e:
                results.append(f"❌ Step failed: {str(e)}")
        
        # Summary
        summary = f"\n✨ **Completed {success_count}/{len(sub_intents)} steps successfully**"
        results.append(summary)
        
        return success_count == len(sub_intents), "\n".join(results)
    
    def _execute_navigate(self, intent: Intent) -> Tuple[bool, str]:
        """Execute directory navigation."""
        path = intent.get_entity('path') or intent.get_entity('dirname')
        
        if not path:
            # Try to find directory name in raw message
            match = re.search(r'(?:inside|into|to)\s+([A-Za-z][A-Za-z0-9_\-\.\/\\]*)', 
                            intent.raw_message, re.IGNORECASE)
            if match:
                path = match.group(1).strip()
        
        if not path:
            return False, "❌ Could not determine directory to navigate to"
        
        # IMPORTANT: Check if this looks like a URL (not a path)
        if path.startswith('http') or '://' in path or 'github.com' in path.lower():
            return False, f"❌ '{path}' looks like a URL, not a directory path"
        
        # Resolve the path
        resolved_path = self.resolve_path(path)
        
        if not os.path.exists(resolved_path):
            # Try searching in common locations
            search_locations = [
                self._context.current_directory,
                os.path.expanduser('~'),
                os.path.expanduser('~/Documents'),
                os.path.expanduser('~/Desktop'),
            ]
            
            for loc in search_locations:
                potential = os.path.join(loc, path)
                if os.path.exists(potential):
                    resolved_path = os.path.abspath(potential)
                    break
            else:
                # Still not found
                return False, f"❌ Directory not found: `{path}`\n" \
                             f"   Searched in: `{self._context.current_directory}`"
        
        if not os.path.isdir(resolved_path):
            return False, f"❌ Not a directory: `{resolved_path}`"
        
        # Change directory
        os.chdir(resolved_path)
        self._context.current_directory = resolved_path
        
        return True, f"✅ Changed directory to: `{resolved_path}`"
    
    def _execute_list_directory(self, intent: Intent) -> Tuple[bool, str]:
        """Execute directory listing."""
        path = intent.get_entity('path') or self._context.current_directory
        path = self.resolve_path(path)
        
        if not os.path.exists(path):
            return False, f"❌ Directory not found: `{path}`"
        
        if not os.path.isdir(path):
            return False, f"❌ Not a directory: `{path}`"
        
        entries = os.listdir(path)
        dirs = []
        files = []
        
        for entry in sorted(entries)[:50]:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                dirs.append(f"📁 {entry}/")
            else:
                files.append(f"📄 {entry}")
        
        result_list = dirs + files
        output = "\n".join(result_list[:50])
        if len(entries) > 50:
            output += f"\n... and {len(entries) - 50} more"
        
        return True, f"📂 **Contents of `{path}`** ({len(entries)} items):\n```\n{output}\n```"
    
    def _execute_pwd(self) -> Tuple[bool, str]:
        """Execute pwd command."""
        return True, f"📂 Current directory: `{self._context.current_directory}`"
    
    def _execute_git_checkout(self, intent: Intent) -> Tuple[bool, str]:
        """Execute git checkout/switch."""
        branch = intent.get_entity('branch')
        
        if not branch:
            # Try to extract from raw message
            match = re.search(r'to\s+([a-zA-Z0-9_\-]+(?:[\-\/][a-zA-Z0-9_\-]+)*)', 
                            intent.raw_message, re.IGNORECASE)
            if match:
                branch = match.group(1).strip()
        
        if not branch:
            return False, "❌ Please specify a branch name"
        
        # Get path if specified
        path = intent.get_entity('path')
        
        original_dir = os.getcwd()
        try:
            if path and os.path.isdir(path):
                os.chdir(path)
            
            # First fetch to make sure we have latest
            subprocess.run(['git', 'fetch', '--all'], 
                         capture_output=True, text=True, timeout=30)
            
            # Try checkout
            result = subprocess.run(['git', 'checkout', branch], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return True, f"✅ Switched to branch: `{branch}`\n```\n{result.stdout}\n```"
            else:
                # Maybe it's a remote branch?
                result2 = subprocess.run(['git', 'checkout', '-b', branch, f'origin/{branch}'],
                                        capture_output=True, text=True, timeout=30)
                if result2.returncode == 0:
                    return True, f"✅ Checked out remote branch: `{branch}`\n```\n{result2.stdout}\n```"
                
                return False, f"❌ Checkout failed:\n```\n{result.stderr}\n```"
        
        finally:
            if path and os.path.isdir(path):
                os.chdir(original_dir)
    
    def _execute_git_status(self) -> Tuple[bool, str]:
        """Execute git status."""
        result = subprocess.run(['git', 'status'], 
                              capture_output=True, text=True, timeout=30)
        return True, f"📊 **Git Status:**\n```\n{result.stdout or result.stderr}\n```"
    
    def _execute_git_pull(self) -> Tuple[bool, str]:
        """Execute git pull."""
        result = subprocess.run(['git', 'pull'], 
                              capture_output=True, text=True, timeout=60)
        status = "✅" if result.returncode == 0 else "❌"
        return result.returncode == 0, f"{status} **Git Pull:**\n```\n{result.stdout or result.stderr}\n```"
    
    def _execute_run_script(self, intent: Intent) -> Tuple[bool, str]:
        """Execute a script."""
        script = intent.get_entity('script')
        
        if not script:
            # Try to find script in message
            match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*\.py)', intent.raw_message, re.IGNORECASE)
            if match:
                script = match.group(1)
        
        if not script:
            return False, "❌ No script specified"
        
        # Resolve script path
        script_path = self.resolve_path(script)
        
        if not os.path.exists(script_path):
            # Search in current directory
            potential = os.path.join(self._context.current_directory, script)
            if os.path.exists(potential):
                script_path = potential
            else:
                return False, f"❌ Script not found: `{script}`\n   Searched in: `{self._context.current_directory}`"
        
        # Determine interpreter
        if script.endswith('.py'):
            cmd = ['python', script_path]
        elif script.endswith('.sh'):
            cmd = ['bash', script_path]
        elif script.endswith('.ps1'):
            cmd = ['powershell', '-File', script_path]
        else:
            cmd = ['python', script_path]  # Default to python
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=300, cwd=self._context.current_directory)
            
            output = result.stdout if result.stdout else result.stderr
            status = "✅" if result.returncode == 0 else "❌"
            
            return result.returncode == 0, \
                   f"{status} **Script Executed:** `{script}`\n```\n{output[:3000]}\n```"
        
        except subprocess.TimeoutExpired:
            return False, f"❌ Script timed out (300s limit): `{script}`"
    
    def _execute_run_command(self, intent: Intent) -> Tuple[bool, str]:
        """Execute a terminal command."""
        command = intent.get_entity('command')
        
        if not command:
            # Try to extract from message
            patterns = [
                r'(?:run|execute)\s+["\']([^"\']+)["\']',
                r'(?:run|execute)\s+`([^`]+)`',
                r'(?:run|execute)\s+command\s+(.+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, intent.raw_message, re.IGNORECASE)
                if match:
                    command = match.group(1).strip()
                    break
        
        if not command:
            return False, "❌ No command specified"
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, 
                                  text=True, timeout=120, cwd=self._context.current_directory)
            
            output = result.stdout if result.stdout else result.stderr
            status = "✅" if result.returncode == 0 else "❌"
            
            return result.returncode == 0, \
                   f"{status} **Executed:** `{command}`\n```\n{output[:3000]}\n```"
        
        except subprocess.TimeoutExpired:
            return False, f"❌ Command timed out (120s limit)"
    
    def _execute_git_clone(self, intent: Intent) -> Tuple[bool, str]:
        """Execute git clone with support for destination directory."""
        url = intent.get_entity('url')
        destination = intent.get_entity('path')
        raw_msg = intent.raw_message
        
        # Try to extract URL from raw message if not in entities
        if not url:
            url_patterns = [
                r'(https://github\.com/[^\s\'"<>]+)',
                r'(http://github\.com/[^\s\'"<>]+)',
                r'(https?://[^\s\'"<>]+\.git)',
                r'(https?://[^\s\'"<>]+)',
            ]
            for pattern in url_patterns:
                match = re.search(pattern, raw_msg, re.IGNORECASE)
                if match:
                    url = match.group(1).strip().rstrip('.,;:')
                    break
        
        if not url:
            return False, "❌ No repository URL specified"
        
        # Ensure URL has protocol
        if url.startswith('github.com'):
            url = 'https://' + url
        
        # Try to extract destination directory from raw message if not in entities
        if not destination:
            # Pattern: in "path" directory, in 'path' directory, to "path", into "path"
            dest_patterns = [
                r'(?:in|to|into)\s+"([^"]+)"',  # in "C:\path with spaces"
                r"(?:in|to|into)\s+'([^']+)'",  # in 'C:\path with spaces'
                r'(?:in|to|into)\s+([A-Za-z]:[\\\/][^\s]+)',  # in C:\path (no spaces)
                r'directory\s+"([^"]+)"',  # directory "path"
                r"directory\s+'([^']+)'",  # directory 'path'
            ]
            for pattern in dest_patterns:
                match = re.search(pattern, raw_msg, re.IGNORECASE)
                if match:
                    potential_dest = match.group(1).strip()
                    # Make sure it's not the URL
                    if not potential_dest.startswith('http') and 'github.com' not in potential_dest:
                        destination = potential_dest
                        break
        
        # Extract repo name from URL
        repo_name = url.rstrip('/').rstrip('.git').split('/')[-1]
        
        # Determine clone location
        if destination:
            destination = os.path.expanduser(os.path.expandvars(destination))
            if not os.path.isabs(destination):
                destination = os.path.join(self._context.current_directory, destination)
            clone_path = os.path.join(destination, repo_name)
        else:
            clone_path = os.path.join(self._context.current_directory, repo_name)
        
        try:
            # Create destination directory if specified and doesn't exist
            if destination and not os.path.exists(destination):
                os.makedirs(destination, exist_ok=True)
            
            # Determine working directory for clone
            clone_dir = destination if destination else self._context.current_directory
            
            # Execute git clone in the target directory
            result = subprocess.run(['git', 'clone', url], 
                                  capture_output=True, text=True, timeout=300,
                                  cwd=clone_dir)
            
            if result.returncode == 0:
                # Record in context
                self._context.record_clone(url, clone_path)
                
                return True, f"✅ **Successfully cloned repository:**\n" \
                            f"   📦 URL: `{url}`\n" \
                            f"   📁 Cloned to: `{clone_path}`"
            else:
                return False, f"❌ **Clone failed:**\n```\n{result.stderr}\n```"
        
        except subprocess.TimeoutExpired:
            return False, f"❌ Clone timed out (300s limit)"
        except Exception as e:
            return False, f"❌ Clone error: {str(e)}"
    
    def _execute_query_location(self, intent: Intent) -> Tuple[bool, str]:
        """Handle questions about where something is located."""
        msg_lower = intent.raw_message.lower()
        
        # Try to extract specific name being asked about
        name_patterns = [
            r'where\s+is\s+(?:the\s+)?([A-Za-z0-9_\-]+)\s+(?:repo|repository|folder|directory)',
            r'where\s+is\s+([A-Za-z0-9_\-]+)',
            r'location\s+of\s+([A-Za-z0-9_\-]+)',
            r'find\s+(?:the\s+)?([A-Za-z0-9_\-]+)',
        ]
        
        queried_name = None
        for pattern in name_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                queried_name = match.group(1).lower()
                break
        
        # Check if asking about a specific cloned repo by name
        if queried_name and self._context.cloned_repos:
            for url, path in self._context.cloned_repos.items():
                repo_name = url.rstrip('/').rstrip('.git').split('/')[-1].lower()
                if queried_name == repo_name or queried_name in repo_name:
                    return True, f"📁 **Repository '{repo_name}' location:**\n" \
                                f"   📦 URL: `{url}`\n" \
                                f"   📂 Path: `{path}`\n" \
                                f"   💡 Use: `cd {path}` to navigate there"
        
        # Check if asking about last cloned repo in general
        if 'repo' in msg_lower or 'clone' in msg_lower or 'that' in msg_lower:
            if self._context.last_cloned_repo:
                return True, f"📁 **Last cloned repository location:**\n" \
                            f"   📦 URL: `{self._context.last_cloned_url}`\n" \
                            f"   📂 Path: `{self._context.last_cloned_repo}`\n" \
                            f"   💡 Use: `cd {self._context.last_cloned_repo}` to navigate there"
            else:
                return False, "❌ No repository has been cloned in this session yet."
        
        # Check if asking about a specific path from context
        for path in self._context.last_mentioned_paths:
            if os.path.exists(path):
                return True, f"📁 **Path in context:**\n   `{path}`"
        
        # Default: show current directory and any cloned repos
        response = f"📁 **Current directory:** `{self._context.current_directory}`"
        if self._context.cloned_repos:
            response += "\n\n**Cloned repositories in this session:**"
            for url, path in self._context.cloned_repos.items():
                repo_name = url.rstrip('/').rstrip('.git').split('/')[-1]
                response += f"\n   • `{repo_name}`: `{path}`"
        return True, response


# Global instance
_robust_nl_processor: Optional[RobustNLProcessor] = None
_processor_lock = threading.Lock()


def get_robust_nl_processor(llm_router=None) -> RobustNLProcessor:
    """Get or create the global robust NL processor instance."""
    global _robust_nl_processor
    
    with _processor_lock:
        if _robust_nl_processor is None:
            _robust_nl_processor = RobustNLProcessor(llm_router)
        elif llm_router is not None:
            _robust_nl_processor.set_llm_router(llm_router)
        
        return _robust_nl_processor
