"""Intent Classification System for Dynamic Tool System.

This module provides sophisticated intent recognition without keyword matching.
It uses LLM reasoning to understand user requests in any natural language form.

Phase 2.2.1: Intent Classification Pipeline
============================================
- Multi-stage intent classification using semantic understanding
- Primary intent detection (file operation, git, terminal, analysis, question)
- Sub-intent classification for specific operation types
- Confidence scoring with threshold tuning
- Ambiguity detection for unclear requests
- Clarification dialogue for ambiguous intents

Key Design Principle:
--------------------
NO HARDCODED KEYWORDS - All understanding happens through:
1. LLM reasoning about user intent
2. Semantic similarity to tool descriptions
3. Context-aware interpretation
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from .tool_interface import ToolCategory
from .tool_registry import ToolRegistry, get_tool_registry, ToolSearchResult
from .execution_context import ExecutionContext, get_current_context
from .robust_nl_processor import IntentType as CanonicalIntentType

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """Broad intent categories used by the LLM-based classifier.

    These represent high-level domains (file, git, terminal, etc.).
    Each category maps to one or more specific *CanonicalIntentType* values
    defined in ``robust_nl_processor.IntentType`` via the
    ``CATEGORY_TO_INTENTS`` mapping below.

    .. deprecated::
        Prefer importing ``IntentType`` directly from
        ``robust_nl_processor`` for specific intent matching.
    """
    FILE_OPERATION = "file_operation"
    DIRECTORY_OPERATION = "directory_operation"
    GIT_OPERATION = "git_operation"
    TERMINAL_OPERATION = "terminal_operation"
    SEARCH_OPERATION = "search_operation"
    ANALYSIS_OPERATION = "analysis_operation"
    CONFIGURATION = "configuration"
    INFORMATION_QUERY = "information_query"
    CONVERSATION = "conversation"
    MULTI_STEP = "multi_step"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Mapping from broad IntentCategory to specific CanonicalIntentType members.
# This allows the LLM-based classifier to narrow from a broad category
# (e.g. FILE_OPERATION) to a concrete intent (e.g. READ_FILE).
# ---------------------------------------------------------------------------
CATEGORY_TO_INTENTS: Dict[IntentCategory, List[CanonicalIntentType]] = {
    IntentCategory.FILE_OPERATION: [
        CanonicalIntentType.CREATE_FILE,
        CanonicalIntentType.READ_FILE,
        CanonicalIntentType.WRITE_FILE,
        CanonicalIntentType.DELETE_FILE,
        CanonicalIntentType.COPY_FILE,
        CanonicalIntentType.MOVE_FILE,
        CanonicalIntentType.SEARCH_FILE,
    ],
    IntentCategory.DIRECTORY_OPERATION: [
        CanonicalIntentType.NAVIGATE_DIRECTORY,
        CanonicalIntentType.LIST_DIRECTORY,
        CanonicalIntentType.CREATE_DIRECTORY,
        CanonicalIntentType.DELETE_DIRECTORY,
        CanonicalIntentType.COPY_DIRECTORY,
        CanonicalIntentType.SHOW_CURRENT_DIR,
    ],
    IntentCategory.GIT_OPERATION: [
        CanonicalIntentType.GIT_CLONE,
        CanonicalIntentType.GIT_PULL,
        CanonicalIntentType.GIT_PUSH,
        CanonicalIntentType.GIT_COMMIT,
        CanonicalIntentType.GIT_ADD,
        CanonicalIntentType.GIT_BRANCH,
        CanonicalIntentType.GIT_CHECKOUT,
        CanonicalIntentType.GIT_FETCH,
        CanonicalIntentType.GIT_STATUS,
        CanonicalIntentType.GIT_MERGE,
        CanonicalIntentType.GIT_REBASE,
        CanonicalIntentType.GIT_STASH,
        CanonicalIntentType.GIT_LOG,
        CanonicalIntentType.GIT_DIFF,
        CanonicalIntentType.GIT_CONFLICT_RESOLVE,
    ],
    IntentCategory.TERMINAL_OPERATION: [
        CanonicalIntentType.RUN_COMMAND,
        CanonicalIntentType.RUN_SCRIPT,
        CanonicalIntentType.TERMINAL_MONITOR,
        CanonicalIntentType.TERMINAL_KILL,
        CanonicalIntentType.TERMINAL_OUTPUT,
        CanonicalIntentType.TERMINAL_LIST,
    ],
    IntentCategory.SEARCH_OPERATION: [
        CanonicalIntentType.SEARCH_FILE,
        CanonicalIntentType.WEB_SEARCH,
    ],
    IntentCategory.ANALYSIS_OPERATION: [
        CanonicalIntentType.ANALYZE_RESULTS,
        CanonicalIntentType.EXPORT_RESULTS,
    ],
    IntentCategory.CONFIGURATION: [
        CanonicalIntentType.BACKEND_BUILD,
        CanonicalIntentType.BACKEND_CONFIGURE,
        CanonicalIntentType.BACKEND_TEST,
        CanonicalIntentType.BACKEND_MODIFY,
        CanonicalIntentType.BACKEND_LIST,
        CanonicalIntentType.INSTALL_DEPENDENCY,
        CanonicalIntentType.CONFIGURE_ENVIRONMENT,
        CanonicalIntentType.CHECK_DEPENDENCY,
        CanonicalIntentType.ADMIN_ELEVATE,
    ],
    IntentCategory.INFORMATION_QUERY: [
        CanonicalIntentType.QUERY_LOCATION,
        CanonicalIntentType.QUERY_STATUS,
        CanonicalIntentType.SYSTEM_INFO,
    ],
    IntentCategory.CONVERSATION: [
        # Conversation does not map to a specific canonical intent;
        # it means the user is chatting, not requesting an action.
    ],
    IntentCategory.MULTI_STEP: [
        CanonicalIntentType.MULTI_STEP,
        CanonicalIntentType.PLAN_EXECUTION,
        CanonicalIntentType.UNDO_OPERATION,
        CanonicalIntentType.REDO_OPERATION,
    ],
    IntentCategory.UNKNOWN: [
        CanonicalIntentType.UNKNOWN,
    ],
}


class IntentConfidence(Enum):
    """Confidence levels for intent classification."""
    HIGH = "high"          # > 0.85 - Execute without clarification
    MEDIUM = "medium"      # 0.6 - 0.85 - May need confirmation
    LOW = "low"            # 0.3 - 0.6 - Likely needs clarification
    VERY_LOW = "very_low"  # < 0.3 - Definitely needs clarification


@dataclass
class ClassifiedIntent:
    """Result of intent classification."""
    # Primary classification
    primary_intent: IntentCategory
    sub_intent: Optional[str] = None
    
    # Confidence metrics
    confidence_score: float = 0.0
    confidence_level: IntentConfidence = IntentConfidence.LOW
    
    # Extracted information
    target_tools: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    inferred_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    requires_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    ambiguity_reasons: List[str] = field(default_factory=list)
    
    # Multi-step support
    is_multi_step: bool = False
    sub_intents: List['ClassifiedIntent'] = field(default_factory=list)
    
    # Raw data
    raw_user_input: str = ""
    classification_reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def canonical_intents(self) -> List[CanonicalIntentType]:
        """Return the list of specific canonical intents for this category."""
        return CATEGORY_TO_INTENTS.get(self.primary_intent, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_intent": self.primary_intent.value,
            "sub_intent": self.sub_intent,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "target_tools": self.target_tools,
            "extracted_entities": self.extracted_entities,
            "inferred_parameters": self.inferred_parameters,
            "requires_clarification": self.requires_clarification,
            "clarification_questions": self.clarification_questions,
            "ambiguity_reasons": self.ambiguity_reasons,
            "is_multi_step": self.is_multi_step,
            "sub_intents": [s.to_dict() for s in self.sub_intents],
            "raw_user_input": self.raw_user_input,
            "classification_reasoning": self.classification_reasoning,
            "timestamp": self.timestamp,
        }
    
    @property
    def is_actionable(self) -> bool:
        """Check if intent is clear enough to act on."""
        return (
            self.confidence_level in (IntentConfidence.HIGH, IntentConfidence.MEDIUM)
            and len(self.target_tools) > 0
            and not self.requires_clarification
        )
    
    @property
    def needs_user_confirmation(self) -> bool:
        """Check if user confirmation is needed before execution."""
        return self.confidence_level == IntentConfidence.MEDIUM


class LLMBackend(Protocol):
    """Protocol for LLM backends used by the classifier."""
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response from the LLM."""
        ...


@dataclass
class ClassifierConfig:
    """Configuration for the intent classifier."""
    # Confidence thresholds
    high_confidence_threshold: float = 0.85
    medium_confidence_threshold: float = 0.6
    low_confidence_threshold: float = 0.3
    
    # Classification settings
    max_tools_to_consider: int = 5
    enable_multi_step_detection: bool = True
    enable_entity_extraction: bool = True
    
    # LLM settings for classification
    classification_temperature: float = 0.1
    classification_max_tokens: int = 1024
    
    # Clarification settings
    auto_clarify_below_confidence: float = 0.5
    max_clarification_questions: int = 3


class IntentClassifier:
    """Classifies user intent through LLM reasoning.
    
    This classifier does NOT use keyword matching. Instead, it:
    1. Presents the user's request to an LLM
    2. Provides context about available tools
    3. Asks the LLM to reason about intent
    4. Extracts structured classification from the response
    
    The classification is dynamic and works with any phrasing.
    """
    
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        config: Optional[ClassifierConfig] = None,
        llm_backend: Optional[LLMBackend] = None,
    ):
        """Initialize the classifier.
        
        Args:
            registry: Tool registry for tool information
            config: Classifier configuration
            llm_backend: LLM backend for reasoning
        """
        self._registry = registry or get_tool_registry()
        self._config = config or ClassifierConfig()
        self._llm_backend = llm_backend
        
        # Classification prompt template (dynamic, not hardcoded responses)
        self._classification_prompt = self._build_classification_prompt()
    
    def _build_classification_prompt(self) -> str:
        """Build the classification system prompt dynamically from registry.
        
        This prompt enables the LLM to understand and classify any
        user request without hardcoded keyword matching.
        """
        # Dynamically build tool categories from registry
        categories = {}
        for registered in self._registry.get_all_tools():
            # Handle both enum and string category values
            cat = registered.definition.category
            if hasattr(cat, 'value'):
                cat = cat.value
            elif not isinstance(cat, str):
                cat = str(cat)
            
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "name": registered.definition.name,
                "description": registered.definition.description,
            })
        
        category_descriptions = []
        for cat, tools in categories.items():
            tool_list = ", ".join(t["name"] for t in tools)
            category_descriptions.append(f"- **{cat}**: {tool_list}")
        
        return f"""You are an intent classification system for an AI assistant.
Your task is to understand what the user wants to accomplish and map it to available tools.

## Available Tool Categories:
{chr(10).join(category_descriptions)}

## Your Task:
1. Understand the user's intent from their natural language request
2. Determine which tool category best matches their intent
3. Identify specific tools that could accomplish the task
4. Extract any parameters or entities from the request
5. Assess your confidence in the classification
6. Identify if the request requires multiple steps

## Classification Guidelines:
- Focus on WHAT the user wants to accomplish, not HOW they phrase it
- Consider context from conversation history if provided
- If the request is ambiguous, list possible interpretations
- If multiple tools might apply, rank them by relevance
- For multi-step requests, break down into individual intents

## Response Format:
Respond with a JSON object:
{{
    "primary_intent": "category_name",
    "sub_intent": "specific_action or null",
    "confidence_score": 0.0-1.0,
    "reasoning": "explanation of classification",
    "target_tools": ["tool1", "tool2"],
    "extracted_entities": {{"entity_type": "value"}},
    "inferred_parameters": {{"param_name": "value"}},
    "is_ambiguous": true/false,
    "ambiguity_reasons": ["reason1", "reason2"],
    "clarification_questions": ["question1"] or [],
    "is_multi_step": true/false,
    "sub_steps": [
        {{"intent": "...", "tools": ["..."]}}
    ] or []
}}"""
    
    def classify(
        self,
        user_input: str,
        context: Optional[ExecutionContext] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ClassifiedIntent:
        """Classify user intent through LLM reasoning.
        
        Args:
            user_input: The user's natural language request
            context: Execution context for additional information
            conversation_history: Recent conversation for context
            
        Returns:
            Classified intent with confidence and extracted information
        """
        context = context or get_current_context()
        
        # Build the classification request
        classification_request = self._build_classification_request(
            user_input, context, conversation_history
        )
        
        # If we have an LLM backend, use it for classification
        if self._llm_backend:
            try:
                response = self._llm_backend.generate(
                    prompt=classification_request,
                    system_prompt=self._classification_prompt,
                    temperature=self._config.classification_temperature,
                    max_tokens=self._config.classification_max_tokens,
                )
                return self._parse_classification_response(response, user_input)
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, falling back to semantic search")
        
        # Fallback: Use semantic search for classification
        return self._classify_with_semantic_search(user_input, context)
    
    def _build_classification_request(
        self,
        user_input: str,
        context: ExecutionContext,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Build the prompt for classification."""
        parts = [f"User Request: {user_input}"]
        
        # Add context
        parts.append(f"\nCurrent Directory: {context.working_directory}")
        
        if context.git_state and context.git_state.repository_root:
            parts.append(f"Git Repository: {context.git_state.repository_root}")
            if context.git_state.current_branch:
                parts.append(f"Current Branch: {context.git_state.current_branch}")
        
        # Add conversation history for context
        if conversation_history:
            recent = conversation_history[-5:]  # Last 5 messages
            history_text = "\n".join([
                f"{m['role']}: {m['content'][:100]}..."
                for m in recent
            ])
            parts.append(f"\nRecent Conversation:\n{history_text}")
        
        parts.append("\nClassify this request and respond with JSON:")
        
        return "\n".join(parts)
    
    def _parse_classification_response(
        self,
        response: str,
        user_input: str,
    ) -> ClassifiedIntent:
        """Parse the LLM's classification response."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            # Map primary intent to IntentCategory
            primary_intent_str = data.get("primary_intent", "unknown")
            primary_intent = self._map_intent_category(primary_intent_str)
            
            # Determine confidence level
            confidence_score = data.get("confidence_score", 0.5)
            confidence_level = self._get_confidence_level(confidence_score)
            
            # Build classified intent
            classified = ClassifiedIntent(
                primary_intent=primary_intent,
                sub_intent=data.get("sub_intent"),
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                target_tools=data.get("target_tools", []),
                extracted_entities=data.get("extracted_entities", {}),
                inferred_parameters=data.get("inferred_parameters", {}),
                requires_clarification=data.get("is_ambiguous", False),
                clarification_questions=data.get("clarification_questions", []),
                ambiguity_reasons=data.get("ambiguity_reasons", []),
                is_multi_step=data.get("is_multi_step", False),
                raw_user_input=user_input,
                classification_reasoning=data.get("reasoning", ""),
            )
            
            # Parse sub-steps for multi-step requests
            if classified.is_multi_step and "sub_steps" in data:
                classified.sub_intents = [
                    ClassifiedIntent(
                        primary_intent=self._map_intent_category(step.get("intent", "unknown")),
                        target_tools=step.get("tools", []),
                        raw_user_input=user_input,
                    )
                    for step in data["sub_steps"]
                ]
            
            return classified
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse classification response: {e}")
            # Return low-confidence classification
            return ClassifiedIntent(
                primary_intent=IntentCategory.UNKNOWN,
                confidence_score=0.2,
                confidence_level=IntentConfidence.VERY_LOW,
                requires_clarification=True,
                clarification_questions=[
                    "I'm not sure I understand. Could you rephrase your request?",
                    "What would you like me to do?",
                ],
                raw_user_input=user_input,
            )
    
    def _map_intent_category(self, intent_str: str) -> IntentCategory:
        """Map a string intent to IntentCategory enum."""
        intent_mapping = {
            "file_system": IntentCategory.FILE_OPERATION,
            "file_operation": IntentCategory.FILE_OPERATION,
            "file": IntentCategory.FILE_OPERATION,
            "directory": IntentCategory.DIRECTORY_OPERATION,
            "directory_operation": IntentCategory.DIRECTORY_OPERATION,
            "folder": IntentCategory.DIRECTORY_OPERATION,
            "git": IntentCategory.GIT_OPERATION,
            "git_operation": IntentCategory.GIT_OPERATION,
            "version_control": IntentCategory.GIT_OPERATION,
            "terminal": IntentCategory.TERMINAL_OPERATION,
            "terminal_operation": IntentCategory.TERMINAL_OPERATION,
            "command": IntentCategory.TERMINAL_OPERATION,
            "shell": IntentCategory.TERMINAL_OPERATION,
            "search": IntentCategory.SEARCH_OPERATION,
            "search_operation": IntentCategory.SEARCH_OPERATION,
            "find": IntentCategory.SEARCH_OPERATION,
            "analysis": IntentCategory.ANALYSIS_OPERATION,
            "analysis_operation": IntentCategory.ANALYSIS_OPERATION,
            "config": IntentCategory.CONFIGURATION,
            "configuration": IntentCategory.CONFIGURATION,
            "settings": IntentCategory.CONFIGURATION,
            "question": IntentCategory.INFORMATION_QUERY,
            "information_query": IntentCategory.INFORMATION_QUERY,
            "info": IntentCategory.INFORMATION_QUERY,
            "conversation": IntentCategory.CONVERSATION,
            "chat": IntentCategory.CONVERSATION,
            "greeting": IntentCategory.CONVERSATION,
            "multi_step": IntentCategory.MULTI_STEP,
            "multiple": IntentCategory.MULTI_STEP,
        }
        
        intent_lower = intent_str.lower().replace(" ", "_")
        return intent_mapping.get(intent_lower, IntentCategory.UNKNOWN)
    
    def _get_confidence_level(self, score: float) -> IntentConfidence:
        """Convert confidence score to level."""
        if score >= self._config.high_confidence_threshold:
            return IntentConfidence.HIGH
        elif score >= self._config.medium_confidence_threshold:
            return IntentConfidence.MEDIUM
        elif score >= self._config.low_confidence_threshold:
            return IntentConfidence.LOW
        else:
            return IntentConfidence.VERY_LOW
    
    def _classify_with_semantic_search(
        self,
        user_input: str,
        context: ExecutionContext,
    ) -> ClassifiedIntent:
        """Classify using semantic search when LLM is not available.
        
        This fallback uses the tool registry's semantic search to find
        relevant tools based on the user's input.
        """
        # Search for relevant tools
        search_results = self._registry.search_tools(
            user_input,
            max_results=self._config.max_tools_to_consider,
        )
        
        if not search_results:
            return ClassifiedIntent(
                primary_intent=IntentCategory.UNKNOWN,
                confidence_score=0.1,
                confidence_level=IntentConfidence.VERY_LOW,
                requires_clarification=True,
                clarification_questions=["What would you like me to help you with?"],
                raw_user_input=user_input,
            )
        
        # Determine primary intent from top result's category
        top_result = search_results[0]
        category = top_result.tool.definition.category
        primary_intent = self._category_to_intent(category)
        
        # Calculate confidence from search scores
        top_score = top_result.relevance_score
        # Normalize to 0-1 range (assuming max score around 10)
        confidence_score = min(top_score / 10.0, 1.0)
        confidence_level = self._get_confidence_level(confidence_score)
        
        # Check for ambiguity (multiple tools with similar scores)
        is_ambiguous = False
        if len(search_results) >= 2:
            score_diff = search_results[0].relevance_score - search_results[1].relevance_score
            if score_diff < 1.0:  # Very close scores indicate ambiguity
                is_ambiguous = True
        
        return ClassifiedIntent(
            primary_intent=primary_intent,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            target_tools=[r.tool.definition.name for r in search_results[:3]],
            requires_clarification=is_ambiguous and confidence_level != IntentConfidence.HIGH,
            ambiguity_reasons=[r.match_reason for r in search_results[:3]] if is_ambiguous else [],
            raw_user_input=user_input,
            classification_reasoning=f"Matched via semantic search: {top_result.match_reason}",
        )
    
    def _category_to_intent(self, category: ToolCategory) -> IntentCategory:
        """Map tool category to intent category."""
        mapping = {
            ToolCategory.FILE_SYSTEM: IntentCategory.FILE_OPERATION,
            ToolCategory.GIT: IntentCategory.GIT_OPERATION,
            ToolCategory.TERMINAL: IntentCategory.TERMINAL_OPERATION,
            ToolCategory.SEARCH: IntentCategory.SEARCH_OPERATION,
            ToolCategory.ANALYSIS: IntentCategory.ANALYSIS_OPERATION,
            ToolCategory.BACKEND: IntentCategory.CONFIGURATION,
            ToolCategory.GITHUB: IntentCategory.GIT_OPERATION,
            ToolCategory.NETWORK: IntentCategory.INFORMATION_QUERY,
        }
        return mapping.get(category, IntentCategory.UNKNOWN)
    
    def detect_multi_step(
        self,
        user_input: str,
        context: Optional[ExecutionContext] = None,
    ) -> List[str]:
        """Detect if a request contains multiple steps.
        
        Uses LLM reasoning to identify sequential or compound requests.
        
        Args:
            user_input: User's request
            context: Execution context
            
        Returns:
            List of individual step descriptions
        """
        if not self._llm_backend:
            return [user_input]  # Can't detect multi-step without LLM
        
        prompt = f"""Analyze if this request contains multiple steps or is a single action:

Request: {user_input}

If it's multiple steps, list each step. If it's a single step, return just the one step.

Respond with JSON:
{{
    "is_multi_step": true/false,
    "steps": ["step 1 description", "step 2 description", ...]
}}"""
        
        try:
            response = self._llm_backend.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512,
            )
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return data.get("steps", [user_input])
        except Exception as e:
            logger.warning(f"Multi-step detection failed: {e}")
        
        return [user_input]
    
    def generate_clarification(
        self,
        intent: ClassifiedIntent,
    ) -> str:
        """Generate a clarification request for ambiguous intents.
        
        Args:
            intent: The classified intent requiring clarification
            
        Returns:
            Natural language clarification request
        """
        if intent.clarification_questions:
            return intent.clarification_questions[0]
        
        if intent.ambiguity_reasons:
            options = intent.ambiguity_reasons[:3]
            return f"I found multiple possible actions. Did you mean: {', '.join(options)}?"
        
        if not intent.target_tools:
            return "I couldn't find a matching action. Could you describe what you'd like to do?"
        
        return "Could you provide more details about what you'd like me to do?"


# Module-level classifier instance
_global_classifier: Optional[IntentClassifier] = None


def get_intent_classifier(
    registry: Optional[ToolRegistry] = None,
    config: Optional[ClassifierConfig] = None,
) -> IntentClassifier:
    """Get the global intent classifier instance.
    
    Args:
        registry: Optional tool registry
        config: Optional classifier configuration
        
    Returns:
        The intent classifier instance
    """
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = IntentClassifier(registry, config)
    return _global_classifier


def configure_intent_classifier(
    config: ClassifierConfig,
    llm_backend: Optional[LLMBackend] = None,
):
    """Configure the global intent classifier.
    
    Args:
        config: Classifier configuration
        llm_backend: Optional LLM backend for reasoning
    """
    global _global_classifier
    _global_classifier = IntentClassifier(
        config=config,
        llm_backend=llm_backend,
    )
