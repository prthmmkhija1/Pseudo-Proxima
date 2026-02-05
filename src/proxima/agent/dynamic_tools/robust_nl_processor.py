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
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum, auto
import threading


class IntentType(Enum):
    """Types of user intents."""
    # Navigation
    NAVIGATE_DIRECTORY = auto()
    LIST_DIRECTORY = auto()
    SHOW_CURRENT_DIR = auto()
    
    # Git operations
    GIT_CHECKOUT = auto()
    GIT_CLONE = auto()
    GIT_PULL = auto()
    GIT_PUSH = auto()
    GIT_STATUS = auto()
    GIT_COMMIT = auto()
    GIT_ADD = auto()
    GIT_BRANCH = auto()
    GIT_FETCH = auto()
    
    # File operations
    CREATE_FILE = auto()
    READ_FILE = auto()
    WRITE_FILE = auto()
    DELETE_FILE = auto()
    COPY_FILE = auto()
    MOVE_FILE = auto()
    
    # Directory operations
    CREATE_DIRECTORY = auto()
    DELETE_DIRECTORY = auto()
    COPY_DIRECTORY = auto()
    
    # Terminal operations
    RUN_COMMAND = auto()
    RUN_SCRIPT = auto()
    
    # Query operations
    QUERY_LOCATION = auto()  # "where is X", "where did you clone"
    QUERY_STATUS = auto()    # "what happened", "did it work"
    
    # Complex operations
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
    """Tracks context across multiple messages in a session."""
    current_directory: str = field(default_factory=os.getcwd)
    last_mentioned_paths: List[str] = field(default_factory=list)
    last_mentioned_branches: List[str] = field(default_factory=list)
    last_mentioned_urls: List[str] = field(default_factory=list)
    last_operation: Optional[Intent] = None
    operation_history: List[Intent] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Track cloned repositories: {url: cloned_path}
    cloned_repos: Dict[str, str] = field(default_factory=dict)
    last_cloned_repo: Optional[str] = None  # Path to last cloned repo
    last_cloned_url: Optional[str] = None   # URL of last cloned repo
    
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
        
        # Extract paths, branches, urls
        for entity in intent.entities:
            if entity.entity_type == 'path':
                self.add_path(entity.value)
            elif entity.entity_type == 'branch':
                self.add_branch(entity.value)
            elif entity.entity_type == 'url':
                self.add_url(entity.value)


class RobustNLProcessor:
    """Robust Natural Language Processor that works with any LLM.
    
    This processor uses a hybrid approach:
    1. First attempts rule-based pattern matching (always works)
    2. Uses LLM for disambiguation when needed
    3. Falls back gracefully when LLM doesn't return expected format
    4. Maintains context across multiple messages
    """
    
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
        }
    
    def _build_intent_keywords(self) -> Dict[IntentType, List[str]]:
        """Build keyword mappings for intent recognition."""
        return {
            # Navigation
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
            
            # Git
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
                'git status', 'status', 'repo status', 'what changed'
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
            
            # Terminal
            IntentType.RUN_COMMAND: [
                'run', 'execute', 'run command', 'execute command',
                'terminal', 'shell', 'cmd', 'powershell'
            ],
            IntentType.RUN_SCRIPT: [
                'run script', 'execute script', 'run python', 'python',
                '.py', 'run the script', 'execute the script'
            ],
            
            # File operations
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
            
            # Directory operations
            IntentType.CREATE_DIRECTORY: [
                'create folder', 'mkdir', 'make directory', 'new folder',
                'create directory'
            ],
            IntentType.DELETE_DIRECTORY: [
                'delete folder', 'rmdir', 'remove directory', 'delete directory'
            ],
            
            # Query operations - for questions about previous operations
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
        query_patterns = [
            r'where\s+is\s+',
            r'where\s+did\s+',
            r'where\s+was\s+',
            r'where\s+is\s+that',
            r'where\s+is\s+the',
            r'location\s+of',
            r'find\s+the\s+.*(?:repo|clone|path)',
            r'what\s+path',
            r'where.*(?:repo|clone|put|save)',
        ]
        return any(re.search(p, msg_lower) for p in query_patterns)
    
    def _create_query_intent(self, message: str, msg_lower: str) -> Intent:
        """Create a query intent with proper type."""
        # Determine if asking about location or status
        status_patterns = ['did it work', 'successful', 'what happened', 'is it done', 'did you finish']
        is_status = any(p in msg_lower for p in status_patterns)
        
        intent_type = IntentType.QUERY_STATUS if is_status else IntentType.QUERY_LOCATION
        
        intent = Intent(
            intent_type=intent_type,
            entities=self.extract_entities(message),
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
        """Extract entities from a message using regex patterns."""
        entities = []
        
        # Extract paths
        for pattern in self._patterns['windows_path']:
            for match in pattern.finditer(message):
                path = match.group(1).strip().rstrip('.,;:')
                entities.append(ExtractedEntity('path', path, 0.9, 'regex'))
        
        for pattern in self._patterns['unix_path']:
            for match in pattern.finditer(message):
                path = match.group(1).strip().rstrip('.,;:')
                entities.append(ExtractedEntity('path', path, 0.9, 'regex'))
        
        # Extract branches - with strict filtering
        for pattern in self._patterns['branch']:
            for match in pattern.finditer(message):
                branch = match.group(1).strip()
                # Strict validation for branch names:
                # 1. Must be at least 3 characters
                # 2. Must start with a letter
                # 3. Cannot be all digits or mostly digits
                # 4. Cannot be a common word
                # 5. Cannot contain only punctuation
                if (len(branch) >= 3 and 
                    branch[0].isalpha() and 
                    not re.match(r'^\d+[\.]?$', branch) and  # Not just digits
                    not re.match(r'^[\d\.\-]+$', branch) and  # Not digits with punctuation
                    sum(c.isdigit() for c in branch) < len(branch) // 2 and  # Less than half digits
                    branch.lower() not in ['the', 'and', 'for', 'with', 'this', 'that', 
                                          'from', 'clone', 'build', 'run', 'to', 'a', 'an',
                                          'of', 'it', 'in', 'on', 'at', 'be', 'is', 'are',
                                          'repo', 'repository', 'directory', 'folder', 'file',
                                          'branch', 'all', 'please', 'want', 'use', 'updated']):
                    entities.append(ExtractedEntity('branch', branch, 0.85, 'regex'))
        
        # Extract URLs
        for pattern in self._patterns['github_url']:
            for match in pattern.finditer(message):
                url = match.group(1).strip()
                entities.append(ExtractedEntity('url', url, 0.95, 'regex'))
        
        for pattern in self._patterns['git_url']:
            for match in pattern.finditer(message):
                url = match.group(1).strip()
                entities.append(ExtractedEntity('url', url, 0.95, 'regex'))
        
        # Extract scripts
        for pattern in self._patterns['python_script']:
            for match in pattern.finditer(message):
                script = match.group(1).strip()
                entities.append(ExtractedEntity('script', script, 0.9, 'regex'))
        
        for pattern in self._patterns['script']:
            for match in pattern.finditer(message):
                script = match.group(1).strip()
                entities.append(ExtractedEntity('script', script, 0.9, 'regex'))
        
        # Extract quoted commands
        for pattern in self._patterns['quoted_command']:
            for match in pattern.finditer(message):
                cmd = match.group(1).strip()
                if len(cmd) > 2:  # Avoid single chars
                    entities.append(ExtractedEntity('command', cmd, 0.8, 'regex'))
        
        # Try to extract directory names from context
        dir_patterns = [
            r'(?:go\s+)?(?:to|into|inside)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9_\-\.]*)\s+(?:directory|folder|dir)',
            r'(?:go\s+)?(?:to|into|inside)\s+([A-Za-z][A-Za-z0-9_\-\.]*)',
            r'inside\s+([A-Za-z][A-Za-z0-9_\-\.\/]*)',
        ]
        for pattern in dir_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                dirname = match.group(1).strip()
                # Filter out common words
                if dirname.lower() not in ['the', 'a', 'an', 'this', 'that', 'and', 'or', 
                                           'directory', 'folder', 'file', 'repo', 'branch']:
                    entities.append(ExtractedEntity('dirname', dirname, 0.7, 'regex'))
        
        return entities
    
    def recognize_intent(self, message: str) -> Intent:
        """Recognize the user's intent from natural language.
        
        Uses hybrid approach:
        1. Priority checks for specific intents (queries, clones)
        2. Multi-step detection (then, and, after, numbered lists)
        3. Rule-based pattern matching
        4. Context from previous messages
        """
        msg_lower = message.lower()
        
        # PRIORITY 1: Check for location/status queries FIRST
        # These should NEVER be confused with other operations
        if self._is_query_intent(msg_lower):
            return self._create_query_intent(message, msg_lower)
        
        # Check for multi-step operations BEFORE clone check
        # (because clone might be part of multi-step)
        multi_step_separators = [' then ', ' and then ', ' after that ', ' next ', ' finally ']
        is_multi_step = any(sep in msg_lower for sep in multi_step_separators)
        
        # Also check for numbered lists: "1. ... 2. ... 3. ..."
        has_numbered_list = bool(re.search(r'(?:^|\n)\s*\d+[\.\)]\s+', message))
        
        if is_multi_step or has_numbered_list:
            return self._parse_multi_step_intent(message)
        
        # PRIORITY 2: Check for clone operations (contains URL)
        # This takes priority because URLs are distinctive
        if self._is_clone_intent(msg_lower, message):
            return self._create_clone_intent(message)
        
        # Single operation processing
        entities = self.extract_entities(message)
        
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        
        # Score each intent type based on keyword matches
        for intent_type, keywords in self._intent_keywords.items():
            score = 0.0
            matches = 0
            for keyword in keywords:
                if keyword in msg_lower:
                    matches += 1
                    # Longer keyword matches are more specific
                    score += len(keyword) / 20.0
            
            if matches > 0:
                # Normalize score
                confidence = min(0.5 + (score * 0.5), 0.95)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent_type
        
        # Build the intent object
        intent = Intent(
            intent_type=best_intent,
            entities=entities,
            confidence=best_confidence,
            raw_message=message
        )
        
        # Generate explanation
        intent.explanation = self._generate_explanation(intent)
        
        # Enhance with context
        self._enhance_with_context(intent)
        
        return intent
    
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
        
        # Create multi-step intent
        intent = Intent(
            intent_type=IntentType.MULTI_STEP,
            entities=all_entities,
            confidence=0.8 if sub_intents else 0.3,
            raw_message=message
        )
        
        # Store sub-intents in a custom field (using entities as workaround)
        intent._sub_intents = sub_intents  # type: ignore
        
        # Generate explanation
        step_descriptions = []
        for i, sub in enumerate(sub_intents, 1):
            step_descriptions.append(f"Step {i}: {sub['type'].name.replace('_', ' ').title()}")
        
        intent.explanation = f"Multi-step operation: {' → '.join(step_descriptions)}"
        
        return intent
    
    def _generate_explanation(self, intent: Intent) -> str:
        """Generate a human-readable explanation of the intent."""
        explanations = {
            IntentType.NAVIGATE_DIRECTORY: "Navigate to directory",
            IntentType.LIST_DIRECTORY: "List directory contents",
            IntentType.SHOW_CURRENT_DIR: "Show current directory",
            IntentType.GIT_CHECKOUT: "Switch to git branch",
            IntentType.GIT_CLONE: "Clone repository",
            IntentType.GIT_PULL: "Pull changes from remote",
            IntentType.GIT_PUSH: "Push changes to remote",
            IntentType.GIT_STATUS: "Show git status",
            IntentType.GIT_COMMIT: "Commit changes",
            IntentType.GIT_ADD: "Stage files for commit",
            IntentType.GIT_BRANCH: "Git branch operation",
            IntentType.GIT_FETCH: "Fetch from remote",
            IntentType.GIT_CLONE: "Clone repository",
            IntentType.RUN_COMMAND: "Run terminal command",
            IntentType.RUN_SCRIPT: "Run script",
            IntentType.CREATE_FILE: "Create file",
            IntentType.READ_FILE: "Read file",
            IntentType.DELETE_FILE: "Delete file",
            IntentType.COPY_FILE: "Copy file",
            IntentType.MOVE_FILE: "Move file",
            IntentType.CREATE_DIRECTORY: "Create directory",
            IntentType.DELETE_DIRECTORY: "Delete directory",
            IntentType.QUERY_LOCATION: "Query location",
            IntentType.QUERY_STATUS: "Query status",
            IntentType.UNKNOWN: "Unknown operation",
        }
        
        base = explanations.get(intent.intent_type, "Operation")
        
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
            if not intent.get_entity('path') and not intent.get_entity('dirname'):
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
            
            else:
                return False, f"⚠️ Intent '{intent.intent_type.name}' recognized but not yet implemented"
        
        except Exception as e:
            return False, f"❌ Error executing {intent.intent_type.name}: {str(e)}"
        
        finally:
            # Update context with this operation
            self._context.update_from_intent(intent)
    
    def _execute_multi_step(self, intent: Intent) -> Tuple[bool, str]:
        """Execute a multi-step operation."""
        sub_intents = getattr(intent, '_sub_intents', [])
        
        if not sub_intents:
            return False, "❌ No steps found in multi-step operation"
        
        results = []
        success_count = 0
        
        for i, sub in enumerate(sub_intents, 1):
            step_type = sub['type']
            entities = sub['entities']
            part = sub['part']
            
            # Create a sub-intent object
            sub_intent = Intent(
                intent_type=step_type,
                entities=entities,
                confidence=0.8,
                raw_message=part
            )
            sub_intent.explanation = self._generate_explanation(sub_intent)
            
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
