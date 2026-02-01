"""Context-Aware Command Suggestions.

Phase 6: Natural Language Planning & Execution

Provides intelligent command suggestions including:
- Context-based next step suggestions
- Learning from user command patterns
- Relevance ranking of suggestions
- Integration with plan execution history
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from proxima.utils.logging import get_logger

logger = get_logger("agent.command_suggestions")


@dataclass
class CommandSequence:
    """A sequence of related commands."""
    commands: List[str]
    count: int = 1
    last_used: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "commands": self.commands,
            "count": self.count,
            "last_used": self.last_used.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommandSequence":
        """Create from dictionary."""
        seq = cls(commands=data["commands"], count=data.get("count", 1))
        if "last_used" in data:
            seq.last_used = datetime.fromisoformat(data["last_used"])
        return seq


@dataclass
class Suggestion:
    """A command suggestion."""
    command: str
    description: str
    relevance: float  # 0.0 to 1.0
    category: str
    source: str  # "context", "pattern", "history", "default"
    action: Optional[str] = None  # Tool name to execute
    arguments: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command": self.command,
            "description": self.description,
            "relevance": self.relevance,
            "category": self.category,
            "source": self.source,
            "action": self.action,
            "arguments": self.arguments,
        }


@dataclass
class ExecutionContext:
    """Context about current execution state."""
    last_command: Optional[str] = None
    last_result: Optional[str] = None
    last_success: bool = True
    active_backend: Optional[str] = None
    git_status: Optional[Dict[str, Any]] = None
    open_files: List[str] = field(default_factory=list)
    recent_commands: List[str] = field(default_factory=list)
    working_directory: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "last_command": self.last_command,
            "last_result": self.last_result,
            "last_success": self.last_success,
            "active_backend": self.active_backend,
            "git_status": self.git_status,
            "open_files": self.open_files,
            "recent_commands": self.recent_commands,
            "working_directory": self.working_directory,
        }


class PatternLearner:
    """Learn command patterns from user behavior."""
    
    def __init__(self, max_sequences: int = 100):
        """Initialize the pattern learner.
        
        Args:
            max_sequences: Maximum sequences to store per pattern length
        """
        self.max_sequences = max_sequences
        
        # Store sequences by length (2-command, 3-command, etc.)
        self._sequences: Dict[int, Dict[str, CommandSequence]] = defaultdict(dict)
        
        # Recent command history for pattern detection
        self._history: List[str] = []
        self._history_max = 50
    
    def record_command(self, command: str) -> None:
        """Record a command execution.
        
        Args:
            command: The command that was executed
        """
        self._history.append(command)
        
        # Trim history if needed
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]
        
        # Update sequences
        self._update_sequences()
    
    def _update_sequences(self) -> None:
        """Update command sequences based on history."""
        if len(self._history) < 2:
            return
        
        # Extract sequences of length 2 and 3
        for length in [2, 3]:
            if len(self._history) >= length:
                seq = tuple(self._history[-length:])
                key = "->".join(seq)
                
                if key in self._sequences[length]:
                    self._sequences[length][key].count += 1
                    self._sequences[length][key].last_used = datetime.now()
                else:
                    self._sequences[length][key] = CommandSequence(
                        commands=list(seq),
                        count=1,
                    )
                
                # Prune if too many
                if len(self._sequences[length]) > self.max_sequences:
                    self._prune_sequences(length)
    
    def _prune_sequences(self, length: int) -> None:
        """Remove least used/oldest sequences."""
        sequences = self._sequences[length]
        
        # Sort by count and recency
        sorted_keys = sorted(
            sequences.keys(),
            key=lambda k: (sequences[k].count, sequences[k].last_used),
            reverse=True,
        )
        
        # Keep only top sequences
        to_remove = sorted_keys[self.max_sequences:]
        for key in to_remove:
            del sequences[key]
    
    def predict_next(self, recent: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next commands based on recent history.
        
        Args:
            recent: Recent commands
            top_k: Number of predictions to return
            
        Returns:
            List of (command, confidence) tuples
        """
        predictions: Dict[str, float] = defaultdict(float)
        
        # Check for matching sequences
        for length in [3, 2]:
            if len(recent) >= length - 1:
                prefix = tuple(recent[-(length - 1):])
                
                for key, seq in self._sequences[length].items():
                    if tuple(seq.commands[:-1]) == prefix:
                        next_cmd = seq.commands[-1]
                        # Score based on count and recency
                        age_days = (datetime.now() - seq.last_used).days
                        recency_factor = 1.0 / (1 + age_days * 0.1)
                        score = seq.count * recency_factor
                        predictions[next_cmd] = max(predictions[next_cmd], score)
        
        # Normalize scores
        if predictions:
            max_score = max(predictions.values())
            predictions = {k: v / max_score for k, v in predictions.items()}
        
        # Sort and return top k
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_predictions[:top_k]
    
    def get_common_sequences(self, min_count: int = 2) -> List[CommandSequence]:
        """Get common command sequences.
        
        Args:
            min_count: Minimum occurrence count
            
        Returns:
            List of common sequences
        """
        common = []
        for sequences in self._sequences.values():
            for seq in sequences.values():
                if seq.count >= min_count:
                    common.append(seq)
        
        return sorted(common, key=lambda s: s.count, reverse=True)
    
    def save(self, path: Path) -> None:
        """Save patterns to file."""
        data = {
            str(length): {k: v.to_dict() for k, v in seqs.items()}
            for length, seqs in self._sequences.items()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path) -> None:
        """Load patterns from file."""
        if not path.exists():
            return
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            for length_str, seqs in data.items():
                length = int(length_str)
                self._sequences[length] = {
                    k: CommandSequence.from_dict(v)
                    for k, v in seqs.items()
                }
        except Exception as e:
            logger.warning(f"Failed to load patterns: {e}")


class CommandSuggestions:
    """Generate context-aware command suggestions.
    
    Features:
    - Contextual suggestions based on last action
    - Pattern-based suggestions from history
    - Smart defaults for common workflows
    - Personalized learning
    
    Example:
        >>> suggestions = CommandSuggestions()
        >>> 
        >>> # Update context
        >>> suggestions.update_context(
        ...     last_command="build lret_cirq",
        ...     last_success=True,
        ... )
        >>> 
        >>> # Get suggestions
        >>> suggs = suggestions.get_suggestions()
        >>> for s in suggs:
        ...     print(f"{s.command}: {s.description}")
    """
    
    # Contextual suggestion rules: after X, suggest Y
    CONTEXTUAL_RULES = {
        "build": [
            ("Run tests", "run tests for {backend}", "execute_command", 0.9),
            ("Show build logs", "show build logs", "read_file", 0.7),
            ("Check warnings", "check compilation warnings", "analyze_code", 0.6),
        ],
        "test": [
            ("Run again", "run tests again", "execute_command", 0.5),
            ("Fix failures", "analyze test failures", "analyze_code", 0.7),
            ("Commit changes", "commit changes", "git_commit", 0.6),
        ],
        "git_status": [
            ("Commit changes", "commit all changes", "git_commit", 0.8),
            ("Show diff", "show diff", "execute_command", 0.7),
            ("Discard changes", "discard changes", "execute_command", 0.4),
        ],
        "git_commit": [
            ("Push", "push to remote", "git_push", 0.9),
            ("Check status", "git status", "git_status", 0.5),
        ],
        "git_push": [
            ("Check status", "git status", "git_status", 0.6),
            ("View remote", "show remote branches", "execute_command", 0.4),
        ],
        "git_pull": [
            ("Check status", "git status", "git_status", 0.7),
            ("Run tests", "run tests", "execute_command", 0.6),
            ("Build", "rebuild backend", "build_backend", 0.5),
        ],
        "read_file": [
            ("Edit file", "edit this file", "write_file", 0.5),
            ("Search in file", "search in file", "search_files", 0.6),
        ],
        "search_files": [
            ("Read matched file", "read matched file", "read_file", 0.7),
            ("Search again", "search for something else", "search_files", 0.5),
        ],
        "install": [
            ("Build", "build backend", "build_backend", 0.8),
            ("Run tests", "run tests", "execute_command", 0.6),
        ],
    }
    
    # Default suggestions when no context
    DEFAULT_SUGGESTIONS = [
        Suggestion(
            command="git status",
            description="Check repository status",
            relevance=0.7,
            category="git",
            source="default",
            action="git_status",
        ),
        Suggestion(
            command="run tests",
            description="Run the test suite",
            relevance=0.6,
            category="testing",
            source="default",
            action="execute_command",
            arguments={"command": "python -m pytest tests/ -v"},
        ),
        Suggestion(
            command="build",
            description="Build a backend",
            relevance=0.5,
            category="build",
            source="default",
            action="build_backend",
        ),
        Suggestion(
            command="help",
            description="Show available commands",
            relevance=0.4,
            category="help",
            source="default",
            action="help",
        ),
    ]
    
    def __init__(
        self,
        enable_learning: bool = True,
        patterns_path: Optional[Path] = None,
    ):
        """Initialize the suggestion system.
        
        Args:
            enable_learning: Whether to learn from user patterns
            patterns_path: Path to save/load learned patterns
        """
        self.enable_learning = enable_learning
        self.patterns_path = patterns_path
        
        # Current context
        self._context = ExecutionContext()
        
        # Pattern learner
        self._learner = PatternLearner() if enable_learning else None
        
        # Load saved patterns
        if self._learner and patterns_path and patterns_path.exists():
            self._learner.load(patterns_path)
        
        logger.info("CommandSuggestions initialized")
    
    def update_context(
        self,
        last_command: Optional[str] = None,
        last_result: Optional[str] = None,
        last_success: bool = True,
        active_backend: Optional[str] = None,
        git_status: Optional[Dict[str, Any]] = None,
        open_files: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
    ) -> None:
        """Update the execution context.
        
        Args:
            last_command: The last executed command
            last_result: Result of the last command
            last_success: Whether last command succeeded
            active_backend: Currently active backend
            git_status: Current git status
            open_files: List of open files
            working_directory: Current working directory
        """
        if last_command:
            self._context.last_command = last_command
            self._context.recent_commands.append(last_command)
            
            # Keep recent commands limited
            if len(self._context.recent_commands) > 10:
                self._context.recent_commands = self._context.recent_commands[-10:]
            
            # Record for learning
            if self._learner:
                self._learner.record_command(last_command)
        
        if last_result is not None:
            self._context.last_result = last_result
        
        self._context.last_success = last_success
        
        if active_backend:
            self._context.active_backend = active_backend
        
        if git_status:
            self._context.git_status = git_status
        
        if open_files:
            self._context.open_files = open_files
        
        if working_directory:
            self._context.working_directory = working_directory
    
    def get_suggestions(
        self,
        max_suggestions: int = 5,
        include_defaults: bool = True,
    ) -> List[Suggestion]:
        """Get command suggestions based on current context.
        
        Args:
            max_suggestions: Maximum number of suggestions
            include_defaults: Include default suggestions if needed
            
        Returns:
            List of Suggestion objects sorted by relevance
        """
        suggestions: List[Suggestion] = []
        
        # Get contextual suggestions
        contextual = self._get_contextual_suggestions()
        suggestions.extend(contextual)
        
        # Get pattern-based suggestions
        if self._learner:
            pattern_suggs = self._get_pattern_suggestions()
            suggestions.extend(pattern_suggs)
        
        # Get git-aware suggestions
        if self._context.git_status:
            git_suggs = self._get_git_suggestions()
            suggestions.extend(git_suggs)
        
        # Get failure recovery suggestions
        if not self._context.last_success:
            recovery_suggs = self._get_recovery_suggestions()
            suggestions.extend(recovery_suggs)
        
        # Add defaults if needed
        if include_defaults and len(suggestions) < max_suggestions:
            defaults = self._get_default_suggestions()
            suggestions.extend(defaults)
        
        # Deduplicate by command
        seen: Set[str] = set()
        unique: List[Suggestion] = []
        for s in suggestions:
            if s.command not in seen:
                seen.add(s.command)
                unique.append(s)
        
        # Sort by relevance
        unique.sort(key=lambda s: s.relevance, reverse=True)
        
        return unique[:max_suggestions]
    
    def _get_contextual_suggestions(self) -> List[Suggestion]:
        """Get suggestions based on last command."""
        suggestions = []
        
        if not self._context.last_command:
            return suggestions
        
        # Find matching rule
        last_cmd = self._context.last_command.lower()
        
        for trigger, rules in self.CONTEXTUAL_RULES.items():
            if trigger in last_cmd:
                for name, description, action, relevance in rules:
                    # Replace placeholders
                    desc = description
                    if "{backend}" in desc and self._context.active_backend:
                        desc = desc.replace("{backend}", self._context.active_backend)
                    
                    suggestions.append(Suggestion(
                        command=name.lower(),
                        description=desc,
                        relevance=relevance,
                        category=action.split("_")[0] if "_" in action else action,
                        source="context",
                        action=action,
                    ))
                break
        
        return suggestions
    
    def _get_pattern_suggestions(self) -> List[Suggestion]:
        """Get suggestions based on learned patterns."""
        suggestions = []
        
        if not self._learner or not self._context.recent_commands:
            return suggestions
        
        predictions = self._learner.predict_next(
            self._context.recent_commands,
            top_k=3,
        )
        
        for cmd, confidence in predictions:
            suggestions.append(Suggestion(
                command=cmd,
                description=f"Based on your history",
                relevance=confidence * 0.8,  # Slightly lower than context
                category="history",
                source="pattern",
            ))
        
        return suggestions
    
    def _get_git_suggestions(self) -> List[Suggestion]:
        """Get suggestions based on git status."""
        suggestions = []
        git_status = self._context.git_status
        
        if not git_status:
            return suggestions
        
        # If there are uncommitted changes
        if git_status.get("has_changes"):
            suggestions.append(Suggestion(
                command="commit changes",
                description=f"Commit {git_status.get('changed_count', 'your')} changed files",
                relevance=0.8,
                category="git",
                source="context",
                action="git_commit",
            ))
        
        # If there are untracked files
        if git_status.get("untracked_count", 0) > 0:
            suggestions.append(Suggestion(
                command="add files",
                description=f"Stage {git_status['untracked_count']} untracked files",
                relevance=0.7,
                category="git",
                source="context",
                action="execute_command",
                arguments={"command": "git add -A"},
            ))
        
        # If ahead of remote
        if git_status.get("ahead", 0) > 0:
            suggestions.append(Suggestion(
                command="push",
                description=f"Push {git_status['ahead']} commits to remote",
                relevance=0.9,
                category="git",
                source="context",
                action="git_push",
            ))
        
        # If behind remote
        if git_status.get("behind", 0) > 0:
            suggestions.append(Suggestion(
                command="pull",
                description=f"Pull {git_status['behind']} commits from remote",
                relevance=0.9,
                category="git",
                source="context",
                action="git_pull",
            ))
        
        return suggestions
    
    def _get_recovery_suggestions(self) -> List[Suggestion]:
        """Get suggestions for recovering from failures."""
        suggestions = []
        
        last_cmd = self._context.last_command or ""
        last_result = self._context.last_result or ""
        
        # Build failure
        if "build" in last_cmd.lower():
            suggestions.append(Suggestion(
                command="check dependencies",
                description="Verify all dependencies are installed",
                relevance=0.8,
                category="recovery",
                source="context",
                action="execute_command",
                arguments={"command": "pip check"},
            ))
            suggestions.append(Suggestion(
                command="view error logs",
                description="Show detailed build logs",
                relevance=0.7,
                category="recovery",
                source="context",
                action="read_file",
            ))
        
        # Test failure
        if "test" in last_cmd.lower():
            suggestions.append(Suggestion(
                command="run failed tests only",
                description="Re-run only the failed tests",
                relevance=0.8,
                category="recovery",
                source="context",
                action="execute_command",
                arguments={"command": "python -m pytest --lf -v"},
            ))
        
        # Git failure
        if "git" in last_cmd.lower() or "push" in last_cmd.lower():
            if "conflict" in last_result.lower():
                suggestions.append(Suggestion(
                    command="show conflicts",
                    description="List files with merge conflicts",
                    relevance=0.9,
                    category="recovery",
                    source="context",
                    action="git_status",
                ))
            elif "reject" in last_result.lower() or "non-fast-forward" in last_result.lower():
                suggestions.append(Suggestion(
                    command="pull first",
                    description="Pull remote changes before pushing",
                    relevance=0.9,
                    category="recovery",
                    source="context",
                    action="git_pull",
                ))
        
        return suggestions
    
    def _get_default_suggestions(self) -> List[Suggestion]:
        """Get default suggestions."""
        # Filter out suggestions that match recent commands
        recent = set(self._context.recent_commands[-3:])
        
        return [
            s for s in self.DEFAULT_SUGGESTIONS
            if s.command not in recent
        ]
    
    def save_patterns(self) -> None:
        """Save learned patterns to file."""
        if self._learner and self.patterns_path:
            self._learner.save(self.patterns_path)
    
    def clear_context(self) -> None:
        """Clear the current context."""
        self._context = ExecutionContext()
    
    def get_context(self) -> ExecutionContext:
        """Get the current context."""
        return self._context


# Global instance
_suggestions: Optional[CommandSuggestions] = None


def get_command_suggestions() -> CommandSuggestions:
    """Get the global CommandSuggestions instance."""
    global _suggestions
    if _suggestions is None:
        _suggestions = CommandSuggestions()
    return _suggestions


def suggest_next(
    last_command: Optional[str] = None,
    max_suggestions: int = 5,
) -> List[Suggestion]:
    """Convenience function to get suggestions.
    
    Args:
        last_command: The last command that was executed
        max_suggestions: Maximum suggestions to return
        
    Returns:
        List of suggestions
    """
    suggestions = get_command_suggestions()
    if last_command:
        suggestions.update_context(last_command=last_command)
    return suggestions.get_suggestions(max_suggestions=max_suggestions)
