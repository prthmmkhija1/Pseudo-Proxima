"""Natural Language Command Parser.

Phase 6: Natural Language Planning & Execution

Provides command parsing capabilities including:
- Pattern-based command recognition
- LLM fallback for ambiguous requests
- Entity extraction from natural language
- Multi-turn dialogue for clarification
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from proxima.utils.logging import get_logger

logger = get_logger("agent.nl_command_parser")


class CommandType(Enum):
    """Types of parsed commands."""
    TOOL_EXECUTION = "tool_execution"
    QUESTION = "question"
    CLARIFICATION_NEEDED = "clarification_needed"
    CONFIRMATION = "confirmation"
    CANCELLATION = "cancellation"
    HELP = "help"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for parsing."""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.5 - 0.8
    LOW = "low"         # < 0.5


@dataclass
class ExtractedEntity:
    """An entity extracted from the command."""
    name: str
    value: Any
    type: str  # "backend_name", "file_path", "command", etc.
    confidence: float
    source_text: str  # Original text that matched
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class ParsedCommand:
    """Result of parsing a natural language command."""
    command_type: CommandType
    tool_name: Optional[str]
    arguments: Dict[str, Any]
    entities: List[ExtractedEntity]
    confidence: float
    original_text: str
    normalized_text: str
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    clarification_options: List[str] = field(default_factory=list)
    suggested_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command_type": self.command_type.value,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "entities": [
                {
                    "name": e.name,
                    "value": e.value,
                    "type": e.type,
                    "confidence": e.confidence,
                }
                for e in self.entities
            ],
            "confidence": self.confidence,
            "original_text": self.original_text,
            "requires_confirmation": self.requires_confirmation,
            "clarification_options": self.clarification_options,
        }
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category."""
        if self.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    
    @property
    def is_actionable(self) -> bool:
        """Check if command can be executed."""
        return (
            self.command_type == CommandType.TOOL_EXECUTION
            and self.tool_name is not None
            and self.confidence > 0.5
        )


@dataclass
class ConversationContext:
    """Context for multi-turn conversations."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_command: Optional[ParsedCommand] = None
    pending_clarification: Optional[str] = None
    entities_in_scope: Dict[str, Any] = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """Get recent messages."""
        return self.messages[-count:]
    
    def update_entities(self, entities: List[ExtractedEntity]) -> None:
        """Update entities in scope."""
        for entity in entities:
            self.entities_in_scope[entity.name] = entity.value
    
    def clear(self) -> None:
        """Clear conversation context."""
        self.messages.clear()
        self.last_command = None
        self.pending_clarification = None
        self.entities_in_scope.clear()


@dataclass
class CommandPattern:
    """A pattern for matching commands."""
    pattern: str
    tool_name: str
    entity_mapping: Dict[str, int]  # entity_name -> capture group number
    description: str
    examples: List[str] = field(default_factory=list)
    priority: int = 0  # Higher priority patterns checked first


class NLCommandParser:
    """Parse natural language commands.
    
    Features:
    - Pattern-based recognition
    - Entity extraction
    - LLM fallback for ambiguity
    - Multi-turn dialogue support
    
    Example:
        >>> parser = NLCommandParser()
        >>> 
        >>> # Parse a command
        >>> result = parser.parse("build the lret cirq backend")
        >>> print(result.tool_name)  # "build_backend"
        >>> print(result.arguments)  # {"backend_name": "lret_cirq"}
        >>> 
        >>> # Handle clarification
        >>> if result.command_type == CommandType.CLARIFICATION_NEEDED:
        ...     print(result.clarification_options)
    """
    
    # Known backends for entity validation
    KNOWN_BACKENDS = [
        "lret_cirq", "lret_qiskit", "lret_pennylane", "lret_braket",
        "qiskit", "cirq", "pennylane", "braket", "pyquil",
    ]
    
    # Default command patterns
    DEFAULT_PATTERNS = [
        # Build commands
        CommandPattern(
            pattern=r"^(?:please\s+)?build\s+(?:the\s+)?(\w+(?:[\s_-]\w+)?)\s*(?:backend)?$",
            tool_name="build_backend",
            entity_mapping={"backend_name": 1},
            description="Build a quantum backend",
            examples=["build lret cirq", "build the qiskit backend"],
            priority=10,
        ),
        CommandPattern(
            pattern=r"^(?:please\s+)?compile\s+(?:the\s+)?(\w+(?:[\s_-]\w+)?)$",
            tool_name="build_backend",
            entity_mapping={"backend_name": 1},
            description="Compile a backend",
            examples=["compile lret_cirq"],
            priority=9,
        ),
        
        # Test commands
        CommandPattern(
            pattern=r"^(?:please\s+)?run\s+(?:the\s+)?tests?\s*(?:for\s+)?(\w+)?$",
            tool_name="execute_command",
            entity_mapping={"test_target": 1},
            description="Run tests",
            examples=["run tests", "run tests for cirq"],
            priority=10,
        ),
        CommandPattern(
            pattern=r"^(?:please\s+)?test\s+(?:the\s+)?(\w+(?:[\s_-]\w+)?)$",
            tool_name="execute_command",
            entity_mapping={"test_target": 1},
            description="Test a component",
            examples=["test lret_cirq"],
            priority=9,
        ),
        
        # Git commands
        CommandPattern(
            pattern=r"^(?:show\s+)?(?:the\s+)?git\s+status$",
            tool_name="git_status",
            entity_mapping={},
            description="Show git status",
            examples=["git status", "show the git status"],
            priority=10,
        ),
        CommandPattern(
            pattern=r"^(?:what\s+)?(?:files?\s+)?(?:have\s+)?changed\??$",
            tool_name="git_status",
            entity_mapping={},
            description="Show changed files",
            examples=["what changed?", "what files changed?"],
            priority=9,
        ),
        CommandPattern(
            pattern=r"^(?:please\s+)?commit\s+(?:the\s+)?(?:changes?\s+)?(?:with\s+(?:message\s+)?)?['\"]?(.+?)['\"]?$",
            tool_name="git_commit",
            entity_mapping={"message": 1},
            description="Commit changes",
            examples=["commit with message 'fix bug'", "commit changes"],
            priority=10,
        ),
        CommandPattern(
            pattern=r"^(?:please\s+)?push(?:\s+(?:to\s+)?(?:the\s+)?(?:remote|origin))?$",
            tool_name="git_push",
            entity_mapping={},
            description="Push to remote",
            examples=["push", "push to origin"],
            priority=10,
        ),
        CommandPattern(
            pattern=r"^(?:please\s+)?pull(?:\s+(?:from\s+)?(?:the\s+)?(?:remote|origin))?$",
            tool_name="git_pull",
            entity_mapping={},
            description="Pull from remote",
            examples=["pull", "pull from origin"],
            priority=10,
        ),
        CommandPattern(
            pattern=r"^(?:please\s+)?clone\s+(.+)$",
            tool_name="git_clone",
            entity_mapping={"url": 1},
            description="Clone repository",
            examples=["clone https://github.com/user/repo"],
            priority=10,
        ),
        
        # File commands
        CommandPattern(
            pattern=r"^(?:please\s+)?read\s+(?:the\s+)?(?:file\s+)?(.+)$",
            tool_name="read_file",
            entity_mapping={"path": 1},
            description="Read file contents",
            examples=["read file.py", "read the config.yaml"],
            priority=10,
        ),
        CommandPattern(
            pattern=r"^(?:please\s+)?(?:show|list)\s+(?:the\s+)?(?:files\s+in\s+)?(.+)$",
            tool_name="list_directory",
            entity_mapping={"path": 1},
            description="List directory",
            examples=["list src/", "show files in backends/"],
            priority=9,
        ),
        CommandPattern(
            pattern=r"^(?:please\s+)?search\s+(?:for\s+)?['\"]?(.+?)['\"]?(?:\s+in\s+(.+))?$",
            tool_name="search_files",
            entity_mapping={"query": 1, "path": 2},
            description="Search in files",
            examples=["search for 'def main'", "search 'import' in src/"],
            priority=10,
        ),
        
        # Install commands
        CommandPattern(
            pattern=r"^(?:please\s+)?(?:pip\s+)?install\s+(.+)$",
            tool_name="install_package",
            entity_mapping={"package": 1},
            description="Install package",
            examples=["install numpy", "pip install qiskit"],
            priority=10,
        ),
        
        # Execute commands
        CommandPattern(
            pattern=r"^(?:please\s+)?(?:run|execute)\s+['\"](.+)['\"]$",
            tool_name="execute_command",
            entity_mapping={"command": 1},
            description="Execute command",
            examples=["run 'python script.py'", "execute 'make build'"],
            priority=10,
        ),
        
        # Help commands
        CommandPattern(
            pattern=r"^(?:show\s+)?help(?:\s+(?:for|on|about)\s+(.+))?$",
            tool_name="help",
            entity_mapping={"topic": 1},
            description="Show help",
            examples=["help", "help on git commands"],
            priority=5,
        ),
    ]
    
    # Confirmation patterns
    CONFIRMATION_PATTERNS = [
        (r"^(?:yes|y|yeah|yep|sure|ok|okay|confirm|do it|proceed)$", True),
        (r"^(?:no|n|nope|cancel|abort|stop|don't|nevermind)$", False),
    ]
    
    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        patterns: Optional[List[CommandPattern]] = None,
        known_backends: Optional[List[str]] = None,
    ):
        """Initialize the command parser.
        
        Args:
            llm_provider: Optional LLM for fallback parsing
            patterns: Custom command patterns
            known_backends: List of known backend names
        """
        self.llm_provider = llm_provider
        self.patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self.known_backends = known_backends or self.KNOWN_BACKENDS.copy()
        
        # Sort patterns by priority
        self.patterns.sort(key=lambda p: -p.priority)
        
        # Conversation context
        self._context = ConversationContext()
        
        logger.info("NLCommandParser initialized")
    
    def parse(
        self,
        text: str,
        use_context: bool = True,
    ) -> ParsedCommand:
        """Parse a natural language command.
        
        Args:
            text: The text to parse
            use_context: Whether to use conversation context
            
        Returns:
            ParsedCommand with parsing results
        """
        original_text = text
        text = self._normalize_text(text)
        
        # Check for confirmation/cancellation first
        confirmation = self._check_confirmation(text)
        if confirmation is not None:
            return self._create_confirmation_result(confirmation, original_text)
        
        # Check for help request
        if self._is_help_request(text):
            return self._create_help_result(text, original_text)
        
        # Try pattern matching
        result = self._match_patterns(text, original_text)
        if result and result.confidence > 0.5:
            # Validate and enhance with context
            if use_context:
                result = self._enhance_with_context(result)
            
            # Update context
            self._context.last_command = result
            self._context.update_entities(result.entities)
            
            return result
        
        # Try LLM fallback if available
        if self.llm_provider:
            result = self._parse_with_llm(text, original_text)
            if result:
                return result
        
        # Return unclear result
        return ParsedCommand(
            command_type=CommandType.CLARIFICATION_NEEDED,
            tool_name=None,
            arguments={},
            entities=[],
            confidence=0.3,
            original_text=original_text,
            normalized_text=text,
            clarification_options=self._generate_suggestions(text),
            suggested_response="I'm not sure what you want to do. Could you clarify?",
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        text = text.strip().lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove trailing punctuation
        text = text.rstrip('?!.')
        return text
    
    def _check_confirmation(self, text: str) -> Optional[bool]:
        """Check if text is a confirmation or cancellation."""
        for pattern, is_confirm in self.CONFIRMATION_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return is_confirm
        return None
    
    def _create_confirmation_result(
        self,
        is_confirmed: bool,
        original_text: str,
    ) -> ParsedCommand:
        """Create result for confirmation/cancellation."""
        cmd_type = CommandType.CONFIRMATION if is_confirmed else CommandType.CANCELLATION
        return ParsedCommand(
            command_type=cmd_type,
            tool_name=None,
            arguments={"confirmed": is_confirmed},
            entities=[],
            confidence=1.0,
            original_text=original_text,
            normalized_text=original_text.lower(),
        )
    
    def _is_help_request(self, text: str) -> bool:
        """Check if text is a help request."""
        help_patterns = [
            r"^help",
            r"^what can you do",
            r"^how do i",
            r"^show me how",
            r"^explain",
        ]
        return any(re.match(p, text) for p in help_patterns)
    
    def _create_help_result(
        self,
        text: str,
        original_text: str,
    ) -> ParsedCommand:
        """Create result for help request."""
        # Extract topic if present
        topic = None
        match = re.search(r"(?:help|how|explain)\s+(?:on|about|with|for)?\s*(.+)", text)
        if match:
            topic = match.group(1).strip()
        
        return ParsedCommand(
            command_type=CommandType.HELP,
            tool_name="help",
            arguments={"topic": topic},
            entities=[],
            confidence=1.0,
            original_text=original_text,
            normalized_text=text,
            suggested_response=self._generate_help_response(topic),
        )
    
    def _generate_help_response(self, topic: Optional[str]) -> str:
        """Generate help response."""
        if topic:
            return f"Here's help about {topic}..."
        
        return """I can help you with:
• **Build** - Build quantum backends (e.g., "build lret cirq")
• **Test** - Run tests (e.g., "run tests for cirq")
• **Git** - Git operations (e.g., "git status", "commit changes", "push")
• **Files** - File operations (e.g., "read file.py", "list src/")
• **Search** - Search code (e.g., "search for 'def main'")
• **Install** - Install packages (e.g., "install numpy")

Just type what you want to do in natural language!"""
    
    def _match_patterns(
        self,
        text: str,
        original_text: str,
    ) -> Optional[ParsedCommand]:
        """Match text against command patterns."""
        for pattern in self.patterns:
            match = re.match(pattern.pattern, text, re.IGNORECASE)
            if match:
                # Extract entities
                entities = []
                arguments = {}
                
                for entity_name, group_num in pattern.entity_mapping.items():
                    try:
                        value = match.group(group_num)
                        if value:
                            value = self._normalize_entity_value(entity_name, value)
                            entities.append(ExtractedEntity(
                                name=entity_name,
                                value=value,
                                type=entity_name,
                                confidence=0.9,
                                source_text=match.group(group_num),
                            ))
                            arguments[entity_name] = value
                    except IndexError:
                        pass
                
                # Calculate confidence based on match quality
                confidence = self._calculate_match_confidence(pattern, match, entities)
                
                # Check if confirmation needed
                needs_confirm, confirm_prompt = self._check_needs_confirmation(
                    pattern.tool_name, arguments
                )
                
                return ParsedCommand(
                    command_type=CommandType.TOOL_EXECUTION,
                    tool_name=pattern.tool_name,
                    arguments=arguments,
                    entities=entities,
                    confidence=confidence,
                    original_text=original_text,
                    normalized_text=text,
                    requires_confirmation=needs_confirm,
                    confirmation_prompt=confirm_prompt,
                )
        
        return None
    
    def _normalize_entity_value(self, entity_name: str, value: str) -> str:
        """Normalize an entity value."""
        value = value.strip()
        
        if entity_name == "backend_name":
            # Normalize backend name
            value = value.lower().replace(" ", "_").replace("-", "_")
            
            # Try to match known backends
            for backend in self.known_backends:
                if value in backend or backend.startswith(value):
                    return backend
        
        elif entity_name == "path":
            # Clean up file path
            value = value.strip("'\"")
        
        elif entity_name == "command":
            # Clean up command
            value = value.strip("'\"")
        
        return value
    
    def _calculate_match_confidence(
        self,
        pattern: CommandPattern,
        match: re.Match,
        entities: List[ExtractedEntity],
    ) -> float:
        """Calculate confidence score for a match."""
        confidence = 0.8  # Base confidence for pattern match
        
        # Boost for exact matches
        if match.group(0) == match.string:
            confidence += 0.1
        
        # Boost for validated entities
        for entity in entities:
            if entity.type == "backend_name" and entity.value in self.known_backends:
                confidence += 0.05
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _check_needs_confirmation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Check if command needs user confirmation."""
        # Dangerous operations need confirmation
        dangerous_tools = {
            "git_push": "Push changes to remote repository?",
            "git_reset": "Reset changes? This may lose work.",
            "delete_file": "Delete this file permanently?",
            "execute_command": None,  # Only confirm certain commands
        }
        
        if tool_name in dangerous_tools:
            prompt = dangerous_tools[tool_name]
            if tool_name == "execute_command":
                cmd = arguments.get("command", "")
                if any(kw in cmd.lower() for kw in ["rm", "del", "drop", "reset"]):
                    return True, f"Execute potentially destructive command: {cmd}?"
                return False, None
            return True, prompt
        
        return False, None
    
    def _enhance_with_context(self, result: ParsedCommand) -> ParsedCommand:
        """Enhance result with conversation context."""
        # Fill in missing entities from context
        for entity_name, value in self._context.entities_in_scope.items():
            if entity_name not in result.arguments:
                # Only add if relevant to this tool
                if self._is_entity_relevant(entity_name, result.tool_name):
                    result.arguments[entity_name] = value
        
        return result
    
    def _is_entity_relevant(self, entity_name: str, tool_name: str) -> bool:
        """Check if an entity is relevant to a tool."""
        relevance_map = {
            "backend_name": ["build_backend", "test_backend", "analyze_code"],
            "path": ["read_file", "write_file", "list_directory", "search_files"],
            "test_target": ["execute_command"],
        }
        
        for entity, tools in relevance_map.items():
            if entity == entity_name and tool_name in tools:
                return True
        
        return False
    
    def _parse_with_llm(
        self,
        text: str,
        original_text: str,
    ) -> Optional[ParsedCommand]:
        """Use LLM to parse command."""
        if not self.llm_provider:
            return None
        
        # This would call the LLM - simplified for now
        logger.debug(f"LLM fallback for: {text}")
        return None
    
    def _generate_suggestions(self, text: str) -> List[str]:
        """Generate command suggestions for unclear input."""
        suggestions = []
        
        # Look for partial matches
        keywords = text.split()
        
        for pattern in self.patterns:
            for keyword in keywords:
                if keyword in pattern.description.lower():
                    for example in pattern.examples[:1]:
                        if example not in suggestions:
                            suggestions.append(example)
        
        # Add generic suggestions if nothing found
        if not suggestions:
            suggestions = [
                "build <backend_name>",
                "run tests",
                "git status",
                "help",
            ]
        
        return suggestions[:4]
    
    def add_pattern(self, pattern: CommandPattern) -> None:
        """Add a custom command pattern."""
        self.patterns.append(pattern)
        self.patterns.sort(key=lambda p: -p.priority)
    
    def add_backend(self, backend_name: str) -> None:
        """Add a known backend name."""
        if backend_name not in self.known_backends:
            self.known_backends.append(backend_name)
    
    def get_context(self) -> ConversationContext:
        """Get the conversation context."""
        return self._context
    
    def clear_context(self) -> None:
        """Clear conversation context."""
        self._context.clear()
    
    def get_available_commands(self) -> List[Dict[str, Any]]:
        """Get list of available commands."""
        commands = []
        seen_tools = set()
        
        for pattern in self.patterns:
            if pattern.tool_name not in seen_tools:
                commands.append({
                    "tool": pattern.tool_name,
                    "description": pattern.description,
                    "examples": pattern.examples,
                })
                seen_tools.add(pattern.tool_name)
        
        return commands


# Global instance
_parser: Optional[NLCommandParser] = None


def get_nl_command_parser() -> NLCommandParser:
    """Get the global NLCommandParser instance."""
    global _parser
    if _parser is None:
        _parser = NLCommandParser()
    return _parser


def parse_command(text: str) -> ParsedCommand:
    """Convenience function to parse a command."""
    return get_nl_command_parser().parse(text)
