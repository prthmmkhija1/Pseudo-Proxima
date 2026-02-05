"""
Phase 12: Documentation and Knowledge Management

This module implements comprehensive documentation and knowledge management
capabilities for the dynamic AI assistant. It provides:

- Comprehensive Documentation (Phase 12.1): User docs, developer docs, system docs
- Knowledge Base Integration (Phase 12.2): Documentation embedding, semantic search,
  in-context help

Architecture Principles:
- Stable infrastructure that supports dynamic model operation
- No hardcoding - all documentation and knowledge are discoverable
- Works with any integrated LLM (Ollama, Gemini, GPT, Claude, etc.)
- Self-describing components for LLM understanding
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
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
# Enums for Documentation and Knowledge Management
# ============================================================================

class DocumentationType(Enum):
    """Types of documentation."""
    USER_GUIDE = "user_guide"
    TUTORIAL = "tutorial"
    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"
    API_REFERENCE = "api_reference"
    ARCHITECTURE = "architecture"
    DEVELOPER_GUIDE = "developer_guide"
    CONTRIBUTION = "contribution"
    DEPLOYMENT = "deployment"
    CONFIGURATION = "configuration"
    RUNBOOK = "runbook"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class DocumentationFormat(Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    RST = "rst"
    JSON = "json"
    PLAIN_TEXT = "plain_text"


class HelpLevel(Enum):
    """Help detail levels."""
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    EXPERT = "expert"


class SearchRelevance(Enum):
    """Search result relevance levels."""
    EXACT_MATCH = "exact_match"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DocumentationStatus(Enum):
    """Documentation status."""
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class KnowledgeCategory(Enum):
    """Knowledge base categories."""
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    REFERENCE = "reference"
    EXAMPLE = "example"
    BEST_PRACTICE = "best_practice"
    TROUBLESHOOTING = "troubleshooting"
    FAQ = "faq"


class TutorialDifficulty(Enum):
    """Tutorial difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# ============================================================================
# Data Classes for Documentation and Knowledge Management
# ============================================================================

@dataclass
class DocumentationEntry:
    """Represents a documentation entry."""
    doc_id: str
    title: str
    content: str
    doc_type: DocumentationType
    
    # Metadata
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    category: str = ""
    
    # Status
    status: DocumentationStatus = DocumentationStatus.PUBLISHED
    version: str = "1.0.0"
    
    # Authorship
    author: str = ""
    contributors: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Relations
    related_docs: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Embedding for semantic search
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type.value,
            "description": self.description,
            "keywords": self.keywords,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Tutorial:
    """Represents an interactive tutorial."""
    tutorial_id: str
    title: str
    description: str
    difficulty: TutorialDifficulty
    
    # Content
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_time_minutes: int = 30
    
    # Prerequisites
    prerequisites: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    
    # Metadata
    category: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Status
    status: DocumentationStatus = DocumentationStatus.PUBLISHED
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tutorial_id": self.tutorial_id,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "estimated_time_minutes": self.estimated_time_minutes,
            "steps_count": len(self.steps),
            "status": self.status.value,
        }


@dataclass
class FAQEntry:
    """Represents a FAQ entry."""
    faq_id: str
    question: str
    answer: str
    
    # Categorization
    category: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Related
    related_faqs: List[str] = field(default_factory=list)
    related_docs: List[str] = field(default_factory=list)
    
    # Metrics
    view_count: int = 0
    helpful_count: int = 0
    not_helpful_count: int = 0
    
    # Status
    status: DocumentationStatus = DocumentationStatus.PUBLISHED
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Embedding
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "faq_id": self.faq_id,
            "question": self.question,
            "category": self.category,
            "helpful_count": self.helpful_count,
            "view_count": self.view_count,
            "status": self.status.value,
        }


@dataclass
class TroubleshootingGuide:
    """Represents a troubleshooting guide."""
    guide_id: str
    title: str
    problem_description: str
    
    # Symptoms
    symptoms: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    
    # Diagnostic steps
    diagnostic_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Solutions
    solutions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Prevention
    prevention_tips: List[str] = field(default_factory=list)
    
    # Related
    related_issues: List[str] = field(default_factory=list)
    
    # Metadata
    category: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Status
    status: DocumentationStatus = DocumentationStatus.PUBLISHED
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Embedding
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "guide_id": self.guide_id,
            "title": self.title,
            "symptoms_count": len(self.symptoms),
            "solutions_count": len(self.solutions),
            "category": self.category,
            "status": self.status.value,
        }


@dataclass
class APIDocumentation:
    """Represents API documentation."""
    api_id: str
    name: str
    description: str
    
    # API details
    module_path: str = ""
    class_name: str = ""
    method_name: str = ""
    
    # Signature
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = ""
    return_description: str = ""
    
    # Examples
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Exceptions
    exceptions: List[Dict[str, str]] = field(default_factory=list)
    
    # Related
    related_apis: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)
    
    # Deprecation
    deprecated: bool = False
    deprecation_message: str = ""
    
    # Version
    since_version: str = ""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_id": self.api_id,
            "name": self.name,
            "module_path": self.module_path,
            "deprecated": self.deprecated,
            "parameters_count": len(self.parameters),
        }


@dataclass
class KnowledgeArticle:
    """Represents a knowledge base article."""
    article_id: str
    title: str
    content: str
    category: KnowledgeCategory
    
    # Summary
    summary: str = ""
    
    # Keywords
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Related
    related_articles: List[str] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    version: str = "1.0"
    
    # Status
    status: DocumentationStatus = DocumentationStatus.PUBLISHED
    
    # Metrics
    view_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Embedding for semantic search
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "category": self.category.value,
            "summary": self.summary,
            "view_count": self.view_count,
            "rating": self.rating,
            "status": self.status.value,
        }


@dataclass
class SearchResult:
    """Represents a search result."""
    result_id: str
    title: str
    content_preview: str
    source_type: str
    source_id: str
    
    # Relevance
    relevance: SearchRelevance
    relevance_score: float
    
    # Highlights
    highlights: List[str] = field(default_factory=list)
    
    # Metadata
    category: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "title": self.title,
            "content_preview": self.content_preview,
            "source_type": self.source_type,
            "relevance": self.relevance.value,
            "relevance_score": self.relevance_score,
        }


@dataclass
class ContextualHelp:
    """Represents contextual help."""
    help_id: str
    context: str
    help_content: str
    help_level: HelpLevel
    
    # Related
    related_commands: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    # Links
    documentation_links: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "help_id": self.help_id,
            "context": self.context,
            "help_level": self.help_level.value,
            "examples_count": len(self.examples),
        }


@dataclass
class HelpHistoryEntry:
    """Represents a help history entry."""
    entry_id: str
    query: str
    result_type: str
    result_id: str
    
    # Context
    context: str = ""
    
    # Feedback
    was_helpful: Optional[bool] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "query": self.query,
            "result_type": self.result_type,
            "was_helpful": self.was_helpful,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Bookmark:
    """Represents a documentation bookmark."""
    bookmark_id: str
    user_id: str
    doc_type: str
    doc_id: str
    title: str
    
    # Notes
    notes: str = ""
    
    # Tags
    tags: List[str] = field(default_factory=list)
    
    # Timestamp
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bookmark_id": self.bookmark_id,
            "doc_type": self.doc_type,
            "doc_id": self.doc_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DocumentationMetrics:
    """Documentation quality metrics."""
    doc_id: str
    
    # Quality scores
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    clarity_score: float = 0.0
    relevance_score: float = 0.0
    
    # Usage metrics
    view_count: int = 0
    search_hits: int = 0
    time_on_page_avg_seconds: float = 0.0
    
    # Feedback
    helpful_count: int = 0
    not_helpful_count: int = 0
    
    # Issues
    reported_issues: int = 0
    resolved_issues: int = 0
    
    # Last updated
    last_review: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score,
            "clarity_score": self.clarity_score,
            "view_count": self.view_count,
            "helpful_count": self.helpful_count,
        }


# ============================================================================
# Phase 12.1: Comprehensive Documentation
# ============================================================================

class UserDocumentationManager:
    """
    Manages user-facing documentation.
    
    Features:
    - Interactive tutorials and walkthroughs
    - Contextual help system
    - Searchable knowledge base
    - FAQ system
    - Troubleshooting guides
    """
    
    def __init__(self):
        self._tutorials: Dict[str, Tutorial] = {}
        self._faqs: Dict[str, FAQEntry] = {}
        self._troubleshooting_guides: Dict[str, TroubleshootingGuide] = {}
        self._user_docs: Dict[str, DocumentationEntry] = {}
        self._lock = threading.Lock()
        
        # Initialize default documentation
        self._initialize_default_documentation()
    
    def _initialize_default_documentation(self):
        """Initialize default user documentation."""
        # Add getting started tutorial
        self.add_tutorial(Tutorial(
            tutorial_id="tutorial_getting_started",
            title="Getting Started with Proxima",
            description="Learn the basics of using the Proxima AI Assistant",
            difficulty=TutorialDifficulty.BEGINNER,
            steps=[
                {"step": 1, "title": "Open the Assistant", "content": "Launch Proxima and navigate to the AI Assistant (press 6)"},
                {"step": 2, "title": "Your First Command", "content": "Try asking 'list files' to see files in the current directory"},
                {"step": 3, "title": "Natural Language", "content": "Use natural language - 'show me files', 'what's here?', etc."},
                {"step": 4, "title": "Git Operations", "content": "Try 'git status' or 'show git history'"},
                {"step": 5, "title": "Getting Help", "content": "Ask 'help' or 'what can you do?' for capabilities"},
            ],
            estimated_time_minutes=10,
            category="basics",
            tags=["beginner", "introduction", "getting-started"],
        ))
        
        # Add common FAQs
        self.add_faq(FAQEntry(
            faq_id="faq_natural_language",
            question="How do I phrase my requests?",
            answer="You can use any natural language! Say things like 'show files', 'list directory', "
                   "'what's in this folder?', or 'display contents'. The AI understands various phrasings.",
            category="usage",
            tags=["natural-language", "commands", "basics"],
        ))
        
        self.add_faq(FAQEntry(
            faq_id="faq_supported_operations",
            question="What operations are supported?",
            answer="The assistant supports file operations (read, write, create, delete), "
                   "git operations (clone, commit, push, pull), terminal commands, "
                   "and various analysis tasks. Ask 'what can you do?' for details.",
            category="capabilities",
            tags=["features", "operations", "capabilities"],
        ))
    
    def add_tutorial(self, tutorial: Tutorial):
        """Add a tutorial."""
        with self._lock:
            self._tutorials[tutorial.tutorial_id] = tutorial
    
    def get_tutorial(self, tutorial_id: str) -> Optional[Tutorial]:
        """Get a tutorial by ID."""
        with self._lock:
            return self._tutorials.get(tutorial_id)
    
    def list_tutorials(
        self,
        difficulty: Optional[TutorialDifficulty] = None,
        category: Optional[str] = None
    ) -> List[Tutorial]:
        """List tutorials with optional filtering."""
        with self._lock:
            tutorials = list(self._tutorials.values())
        
        if difficulty:
            tutorials = [t for t in tutorials if t.difficulty == difficulty]
        
        if category:
            tutorials = [t for t in tutorials if t.category == category]
        
        return tutorials
    
    def add_faq(self, faq: FAQEntry):
        """Add a FAQ entry."""
        with self._lock:
            self._faqs[faq.faq_id] = faq
    
    def get_faq(self, faq_id: str) -> Optional[FAQEntry]:
        """Get a FAQ by ID."""
        with self._lock:
            faq = self._faqs.get(faq_id)
            if faq:
                faq.view_count += 1
            return faq
    
    def search_faqs(self, query: str) -> List[FAQEntry]:
        """Search FAQs by query."""
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for faq in self._faqs.values():
                # Check question and answer
                if (query_lower in faq.question.lower() or
                    query_lower in faq.answer.lower() or
                    any(query_lower in tag.lower() for tag in faq.tags)):
                    results.append(faq)
        
        return results
    
    def add_troubleshooting_guide(self, guide: TroubleshootingGuide):
        """Add a troubleshooting guide."""
        with self._lock:
            self._troubleshooting_guides[guide.guide_id] = guide
    
    def find_troubleshooting_guide(
        self,
        error_message: Optional[str] = None,
        symptom: Optional[str] = None
    ) -> List[TroubleshootingGuide]:
        """Find troubleshooting guides by error or symptom."""
        results = []
        
        with self._lock:
            for guide in self._troubleshooting_guides.values():
                match = False
                
                if error_message:
                    for err in guide.error_messages:
                        if error_message.lower() in err.lower():
                            match = True
                            break
                
                if symptom and not match:
                    for sym in guide.symptoms:
                        if symptom.lower() in sym.lower():
                            match = True
                            break
                
                if match:
                    results.append(guide)
        
        return results
    
    def add_user_documentation(self, doc: DocumentationEntry):
        """Add user documentation."""
        with self._lock:
            self._user_docs[doc.doc_id] = doc
    
    def get_user_documentation(
        self,
        doc_type: Optional[DocumentationType] = None
    ) -> List[DocumentationEntry]:
        """Get user documentation."""
        with self._lock:
            docs = list(self._user_docs.values())
        
        if doc_type:
            docs = [d for d in docs if d.doc_type == doc_type]
        
        return docs
    
    def mark_faq_helpful(self, faq_id: str, helpful: bool):
        """Mark FAQ as helpful or not."""
        with self._lock:
            faq = self._faqs.get(faq_id)
            if faq:
                if helpful:
                    faq.helpful_count += 1
                else:
                    faq.not_helpful_count += 1


class DeveloperDocumentationManager:
    """
    Manages developer documentation.
    
    Features:
    - API documentation
    - Code documentation with docstrings
    - Architecture documentation
    - Contribution guidelines
    - Design decision records
    """
    
    def __init__(self):
        self._api_docs: Dict[str, APIDocumentation] = {}
        self._architecture_docs: Dict[str, DocumentationEntry] = {}
        self._design_records: Dict[str, DocumentationEntry] = {}
        self._contribution_guides: Dict[str, DocumentationEntry] = {}
        self._lock = threading.Lock()
    
    def add_api_documentation(self, api_doc: APIDocumentation):
        """Add API documentation."""
        with self._lock:
            self._api_docs[api_doc.api_id] = api_doc
    
    def get_api_documentation(self, api_id: str) -> Optional[APIDocumentation]:
        """Get API documentation by ID."""
        with self._lock:
            return self._api_docs.get(api_id)
    
    def search_api_documentation(self, query: str) -> List[APIDocumentation]:
        """Search API documentation."""
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for api_doc in self._api_docs.values():
                if (query_lower in api_doc.name.lower() or
                    query_lower in api_doc.description.lower() or
                    query_lower in api_doc.module_path.lower()):
                    results.append(api_doc)
        
        return results
    
    def generate_api_doc_from_function(
        self,
        func: Callable,
        module_path: str
    ) -> APIDocumentation:
        """Generate API documentation from a function."""
        import inspect
        
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        
        # Parse parameters
        parameters = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            
            param_info = {
                "name": name,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                "required": param.default == inspect.Parameter.empty,
            }
            parameters.append(param_info)
        
        # Parse return type
        return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "None"
        
        api_doc = APIDocumentation(
            api_id=f"api_{func.__name__}_{uuid.uuid4().hex[:8]}",
            name=func.__name__,
            description=docstring.split("\n")[0] if docstring else "",
            module_path=module_path,
            method_name=func.__name__,
            parameters=parameters,
            return_type=return_type,
        )
        
        self.add_api_documentation(api_doc)
        return api_doc
    
    def add_architecture_documentation(self, doc: DocumentationEntry):
        """Add architecture documentation."""
        with self._lock:
            self._architecture_docs[doc.doc_id] = doc
    
    def get_architecture_documentation(self) -> List[DocumentationEntry]:
        """Get all architecture documentation."""
        with self._lock:
            return list(self._architecture_docs.values())
    
    def add_design_record(self, record: DocumentationEntry):
        """Add a design decision record."""
        with self._lock:
            self._design_records[record.doc_id] = record
    
    def get_design_records(self) -> List[DocumentationEntry]:
        """Get all design decision records."""
        with self._lock:
            return list(self._design_records.values())
    
    def add_contribution_guide(self, guide: DocumentationEntry):
        """Add a contribution guide."""
        with self._lock:
            self._contribution_guides[guide.doc_id] = guide
    
    def get_contribution_guides(self) -> List[DocumentationEntry]:
        """Get all contribution guides."""
        with self._lock:
            return list(self._contribution_guides.values())


class SystemDocumentationManager:
    """
    Manages system documentation.
    
    Features:
    - Deployment documentation
    - Configuration reference
    - Operational runbooks
    - Disaster recovery procedures
    - Security documentation
    """
    
    def __init__(self):
        self._deployment_docs: Dict[str, DocumentationEntry] = {}
        self._config_docs: Dict[str, DocumentationEntry] = {}
        self._runbooks: Dict[str, DocumentationEntry] = {}
        self._recovery_procedures: Dict[str, DocumentationEntry] = {}
        self._security_docs: Dict[str, DocumentationEntry] = {}
        self._lock = threading.Lock()
        
        # Initialize default documentation
        self._initialize_default_documentation()
    
    def _initialize_default_documentation(self):
        """Initialize default system documentation."""
        # Add deployment documentation
        self.add_deployment_documentation(DocumentationEntry(
            doc_id="deploy_local",
            title="Local Deployment Guide",
            content="""# Local Deployment Guide

## Prerequisites
- Python 3.10+
- Git
- Virtual environment (recommended)

## Steps
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv/Scripts/activate` (Windows) or `source venv/bin/activate` (Unix)
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python run_tui.py`

## Configuration
- Edit `configs/default.yaml` for default settings
- Environment variables can override configuration
""",
            doc_type=DocumentationType.DEPLOYMENT,
            description="Guide for deploying Proxima locally",
            keywords=["deployment", "local", "setup"],
        ))
        
        # Add configuration reference
        self.add_configuration_documentation(DocumentationEntry(
            doc_id="config_reference",
            title="Configuration Reference",
            content="""# Configuration Reference

## Environment Variables
- `PROXIMA_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `PROXIMA_CONFIG_PATH`: Path to configuration file
- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)

## Configuration File (YAML)
```yaml
llm:
  provider: ollama
  model: gemini2:latest
  temperature: 0.7

tools:
  timeout_seconds: 60
  max_retries: 3

logging:
  level: INFO
  format: structured
```
""",
            doc_type=DocumentationType.CONFIGURATION,
            description="Reference for all configuration options",
            keywords=["configuration", "settings", "environment"],
        ))
    
    def add_deployment_documentation(self, doc: DocumentationEntry):
        """Add deployment documentation."""
        with self._lock:
            self._deployment_docs[doc.doc_id] = doc
    
    def get_deployment_documentation(self) -> List[DocumentationEntry]:
        """Get all deployment documentation."""
        with self._lock:
            return list(self._deployment_docs.values())
    
    def add_configuration_documentation(self, doc: DocumentationEntry):
        """Add configuration documentation."""
        with self._lock:
            self._config_docs[doc.doc_id] = doc
    
    def get_configuration_documentation(self) -> List[DocumentationEntry]:
        """Get all configuration documentation."""
        with self._lock:
            return list(self._config_docs.values())
    
    def add_runbook(self, runbook: DocumentationEntry):
        """Add an operational runbook."""
        with self._lock:
            self._runbooks[runbook.doc_id] = runbook
    
    def get_runbooks(self) -> List[DocumentationEntry]:
        """Get all runbooks."""
        with self._lock:
            return list(self._runbooks.values())
    
    def add_recovery_procedure(self, procedure: DocumentationEntry):
        """Add a disaster recovery procedure."""
        with self._lock:
            self._recovery_procedures[procedure.doc_id] = procedure
    
    def get_recovery_procedures(self) -> List[DocumentationEntry]:
        """Get all recovery procedures."""
        with self._lock:
            return list(self._recovery_procedures.values())
    
    def add_security_documentation(self, doc: DocumentationEntry):
        """Add security documentation."""
        with self._lock:
            self._security_docs[doc.doc_id] = doc
    
    def get_security_documentation(self) -> List[DocumentationEntry]:
        """Get all security documentation."""
        with self._lock:
            return list(self._security_docs.values())


# ============================================================================
# Phase 12.2: Knowledge Base Integration
# ============================================================================

class DocumentationIndexer:
    """
    Indexes documentation for semantic search.
    
    Features:
    - Documentation indexing with embeddings
    - Semantic search across documentation
    - Documentation versioning
    - Quality metrics
    """
    
    def __init__(self):
        self._index: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._metrics: Dict[str, DocumentationMetrics] = {}
        self._lock = threading.Lock()
    
    def _compute_simple_embedding(self, text: str) -> List[float]:
        """
        Compute a simple embedding for text.
        
        In production, this would use a proper embedding model like
        sentence-transformers, but we use a simple hash-based approach
        for demonstration that works without external dependencies.
        """
        # Normalize text
        text = text.lower()
        words = re.findall(r'\w+', text)
        
        # Create a simple bag-of-words style embedding
        # This is a simplified approach - real implementation would use ML models
        embedding = [0.0] * 128
        
        for i, word in enumerate(words[:128]):
            # Use hash to map word to embedding dimension
            h = hash(word) % 128
            embedding[h] += 1.0 / (i + 1)  # Decay by position
        
        # Normalize
        magnitude = sum(x*x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def index_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Index a document for search."""
        # Compute embedding
        full_text = f"{title} {content}"
        embedding = self._compute_simple_embedding(full_text)
        
        with self._lock:
            self._index[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "content": content,
                "doc_type": doc_type,
                "metadata": metadata or {},
                "indexed_at": datetime.now().isoformat(),
            }
            self._embeddings[doc_id] = embedding
            
            # Initialize metrics
            if doc_id not in self._metrics:
                self._metrics[doc_id] = DocumentationMetrics(doc_id=doc_id)
    
    def search(
        self,
        query: str,
        doc_type: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """Search documents by query."""
        query_embedding = self._compute_simple_embedding(query)
        query_lower = query.lower()
        
        results = []
        
        with self._lock:
            for doc_id, doc_info in self._index.items():
                if doc_type and doc_info["doc_type"] != doc_type:
                    continue
                
                # Compute similarity
                doc_embedding = self._embeddings.get(doc_id, [])
                if doc_embedding:
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                else:
                    similarity = 0.0
                
                # Boost for exact matches
                title_lower = doc_info["title"].lower()
                content_lower = doc_info["content"].lower()
                
                if query_lower in title_lower:
                    similarity += 0.5
                
                if query_lower in content_lower:
                    similarity += 0.2
                
                # Determine relevance level
                if similarity > 0.8:
                    relevance = SearchRelevance.EXACT_MATCH
                elif similarity > 0.5:
                    relevance = SearchRelevance.HIGH
                elif similarity > 0.3:
                    relevance = SearchRelevance.MEDIUM
                else:
                    relevance = SearchRelevance.LOW
                
                # Extract highlights
                highlights = self._extract_highlights(doc_info["content"], query)
                
                # Create content preview
                content_preview = doc_info["content"][:200] + "..." if len(doc_info["content"]) > 200 else doc_info["content"]
                
                results.append(SearchResult(
                    result_id=f"search_{doc_id}",
                    title=doc_info["title"],
                    content_preview=content_preview,
                    source_type=doc_info["doc_type"],
                    source_id=doc_id,
                    relevance=relevance,
                    relevance_score=similarity,
                    highlights=highlights,
                ))
                
                # Update metrics
                if doc_id in self._metrics:
                    self._metrics[doc_id].search_hits += 1
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:limit]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def _extract_highlights(self, content: str, query: str, max_highlights: int = 3) -> List[str]:
        """Extract highlighted snippets from content."""
        highlights = []
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find occurrences
        start = 0
        while len(highlights) < max_highlights:
            idx = content_lower.find(query_lower, start)
            if idx == -1:
                break
            
            # Extract context around match
            snippet_start = max(0, idx - 50)
            snippet_end = min(len(content), idx + len(query) + 50)
            snippet = content[snippet_start:snippet_end]
            
            if snippet_start > 0:
                snippet = "..." + snippet
            if snippet_end < len(content):
                snippet = snippet + "..."
            
            highlights.append(snippet)
            start = idx + len(query)
        
        return highlights
    
    def add_version(
        self,
        doc_id: str,
        version: str,
        content: str,
        author: str = ""
    ):
        """Add a new version of a document."""
        with self._lock:
            self._versions[doc_id].append({
                "version": version,
                "content": content,
                "author": author,
                "created_at": datetime.now().isoformat(),
            })
    
    def get_versions(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a document."""
        with self._lock:
            return self._versions.get(doc_id, []).copy()
    
    def get_metrics(self, doc_id: str) -> Optional[DocumentationMetrics]:
        """Get metrics for a document."""
        with self._lock:
            return self._metrics.get(doc_id)
    
    def update_metrics(
        self,
        doc_id: str,
        view: bool = False,
        helpful: Optional[bool] = None
    ):
        """Update document metrics."""
        with self._lock:
            if doc_id not in self._metrics:
                self._metrics[doc_id] = DocumentationMetrics(doc_id=doc_id)
            
            metrics = self._metrics[doc_id]
            
            if view:
                metrics.view_count += 1
            
            if helpful is not None:
                if helpful:
                    metrics.helpful_count += 1
                else:
                    metrics.not_helpful_count += 1


class ContextualHelpSystem:
    """
    Provides in-context help.
    
    Features:
    - Contextual help tooltips
    - Inline documentation
    - Command examples on demand
    - Error-specific help
    - Progressive disclosure
    - Help history and bookmarks
    """
    
    def __init__(self, indexer: DocumentationIndexer):
        self._indexer = indexer
        self._contextual_help: Dict[str, ContextualHelp] = {}
        self._error_help: Dict[str, Dict[str, str]] = {}
        self._command_examples: Dict[str, List[str]] = {}
        self._help_history: List[HelpHistoryEntry] = []
        self._bookmarks: Dict[str, List[Bookmark]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Initialize default contextual help
        self._initialize_default_help()
    
    def _initialize_default_help(self):
        """Initialize default contextual help."""
        # File operations help
        self.add_contextual_help(ContextualHelp(
            help_id="help_file_ops",
            context="file_operations",
            help_content="File operations allow you to read, write, create, delete, copy, and move files.",
            help_level=HelpLevel.STANDARD,
            related_commands=["read_file", "write_file", "create_file", "delete_file"],
            examples=[
                "read file config.yaml",
                "create file notes.txt with content 'Hello'",
                "delete file old_notes.txt",
            ],
        ))
        
        # Git operations help
        self.add_contextual_help(ContextualHelp(
            help_id="help_git_ops",
            context="git_operations",
            help_content="Git operations allow you to manage version control including clone, commit, push, pull, and branch management.",
            help_level=HelpLevel.STANDARD,
            related_commands=["git_clone", "git_commit", "git_push", "git_pull", "git_branch"],
            examples=[
                "git status",
                "git commit -m 'Update readme'",
                "switch to branch main",
                "clone https://github.com/user/repo",
            ],
        ))
        
        # Add error-specific help
        self.add_error_help("FileNotFoundError", {
            "description": "The specified file does not exist",
            "common_causes": ["Typo in file path", "File was deleted", "Wrong directory"],
            "solutions": ["Check the file path", "Use 'list files' to see available files", "Check current directory with 'pwd'"],
        })
        
        self.add_error_help("PermissionError", {
            "description": "You don't have permission to access this file",
            "common_causes": ["File is read-only", "Insufficient user permissions", "File locked by another process"],
            "solutions": ["Check file permissions", "Run with elevated privileges if needed", "Close other programs using the file"],
        })
        
        # Add command examples
        self.add_command_examples("list_directory", [
            "list files",
            "show directory contents",
            "what's in this folder?",
            "display files here",
            "ls",
        ])
        
        self.add_command_examples("read_file", [
            "read file README.md",
            "show me config.yaml",
            "display contents of script.py",
            "what's in the package.json?",
            "cat requirements.txt",
        ])
    
    def add_contextual_help(self, help_entry: ContextualHelp):
        """Add contextual help."""
        with self._lock:
            self._contextual_help[help_entry.context] = help_entry
    
    def get_contextual_help(
        self,
        context: str,
        level: HelpLevel = HelpLevel.STANDARD
    ) -> Optional[ContextualHelp]:
        """Get contextual help for a context."""
        with self._lock:
            help_entry = self._contextual_help.get(context)
            if help_entry:
                # Record in history
                self._help_history.append(HelpHistoryEntry(
                    entry_id=f"history_{uuid.uuid4().hex[:8]}",
                    query=context,
                    result_type="contextual_help",
                    result_id=help_entry.help_id,
                ))
            return help_entry
    
    def add_error_help(self, error_type: str, help_info: Dict[str, Any]):
        """Add error-specific help."""
        with self._lock:
            self._error_help[error_type] = help_info
    
    def get_error_help(self, error_message: str) -> Optional[Dict[str, Any]]:
        """Get help for an error message."""
        # Try to identify error type
        for error_type, help_info in self._error_help.items():
            if error_type.lower() in error_message.lower():
                return {
                    "error_type": error_type,
                    **help_info,
                }
        
        # Search documentation for error-related help
        results = self._indexer.search(error_message, limit=3)
        if results:
            return {
                "error_type": "Unknown",
                "related_documentation": [r.to_dict() for r in results],
            }
        
        return None
    
    def add_command_examples(self, command: str, examples: List[str]):
        """Add examples for a command."""
        with self._lock:
            self._command_examples[command] = examples
    
    def get_command_examples(self, command: str) -> List[str]:
        """Get examples for a command."""
        with self._lock:
            return self._command_examples.get(command, []).copy()
    
    def get_examples_for_query(self, query: str) -> List[str]:
        """Get relevant examples for a query."""
        query_lower = query.lower()
        examples = []
        
        with self._lock:
            for command, cmd_examples in self._command_examples.items():
                if query_lower in command.lower():
                    examples.extend(cmd_examples)
        
        return examples[:5]  # Limit to 5 examples
    
    def get_progressive_help(
        self,
        topic: str,
        current_level: HelpLevel
    ) -> Tuple[str, HelpLevel]:
        """Get progressively more detailed help."""
        # Map to next level
        level_progression = {
            HelpLevel.BRIEF: HelpLevel.STANDARD,
            HelpLevel.STANDARD: HelpLevel.DETAILED,
            HelpLevel.DETAILED: HelpLevel.EXPERT,
            HelpLevel.EXPERT: HelpLevel.EXPERT,
        }
        
        next_level = level_progression[current_level]
        
        # Search for topic with more detail
        results = self._indexer.search(topic, limit=1)
        
        if results:
            content = results[0].content_preview
            
            # Simulate different detail levels
            if next_level == HelpLevel.BRIEF:
                content = content[:100] + "..."
            elif next_level == HelpLevel.STANDARD:
                content = content[:300] + "..."
            elif next_level == HelpLevel.DETAILED:
                content = content[:500] + "..."
            # EXPERT returns full content
            
            return content, next_level
        
        return f"No detailed help available for '{topic}'", current_level
    
    def add_bookmark(
        self,
        user_id: str,
        doc_type: str,
        doc_id: str,
        title: str,
        notes: str = ""
    ) -> Bookmark:
        """Add a bookmark."""
        bookmark = Bookmark(
            bookmark_id=f"bookmark_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            doc_type=doc_type,
            doc_id=doc_id,
            title=title,
            notes=notes,
        )
        
        with self._lock:
            self._bookmarks[user_id].append(bookmark)
        
        return bookmark
    
    def get_bookmarks(self, user_id: str) -> List[Bookmark]:
        """Get bookmarks for a user."""
        with self._lock:
            return self._bookmarks.get(user_id, []).copy()
    
    def remove_bookmark(self, user_id: str, bookmark_id: str) -> bool:
        """Remove a bookmark."""
        with self._lock:
            bookmarks = self._bookmarks.get(user_id, [])
            for i, bm in enumerate(bookmarks):
                if bm.bookmark_id == bookmark_id:
                    bookmarks.pop(i)
                    return True
        return False
    
    def get_help_history(self, limit: int = 20) -> List[HelpHistoryEntry]:
        """Get recent help history."""
        with self._lock:
            return self._help_history[-limit:].copy()
    
    def mark_help_helpful(self, entry_id: str, helpful: bool):
        """Mark a help entry as helpful or not."""
        with self._lock:
            for entry in self._help_history:
                if entry.entry_id == entry_id:
                    entry.was_helpful = helpful
                    break


class KnowledgeBaseManager:
    """
    Manages the knowledge base.
    
    Features:
    - Knowledge article management
    - Category organization
    - Cross-referencing
    - Rating and feedback
    """
    
    def __init__(self, indexer: DocumentationIndexer):
        self._indexer = indexer
        self._articles: Dict[str, KnowledgeArticle] = {}
        self._by_category: Dict[KnowledgeCategory, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Initialize default knowledge
        self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """Initialize default knowledge articles."""
        # Add concept article about natural language processing
        self.add_article(KnowledgeArticle(
            article_id="kb_natural_language",
            title="Natural Language Understanding",
            content="""# Natural Language Understanding

The Proxima AI Assistant uses natural language understanding to interpret your requests.
This means you can phrase commands in many different ways.

## How It Works
1. Your input is analyzed to understand the intent
2. Relevant tools are discovered through semantic search
3. Parameters are extracted from your request
4. The appropriate tool is executed

## Examples
- "show files" → list_directory
- "what's here?" → list_directory
- "read config.yaml" → read_file

## Tips
- Be as natural as you want
- Specify file paths when needed
- You can chain commands: "go to src and list files"
""",
            category=KnowledgeCategory.CONCEPT,
            summary="Understanding how the AI interprets your requests",
            keywords=["natural language", "understanding", "intent", "commands"],
            tags=["basics", "nlp", "understanding"],
        ))
        
        # Add procedure article
        self.add_article(KnowledgeArticle(
            article_id="kb_file_operations",
            title="File Operations Guide",
            content="""# File Operations Guide

## Reading Files
- "read file.txt" - Read entire file
- "show first 10 lines of file.txt" - Read partial
- "what's in config.yaml?" - Natural language

## Creating Files
- "create file notes.txt with content 'Hello'"
- "make a new file called test.py"

## Deleting Files
- "delete old_file.txt"
- "remove backup.zip"

## Copying and Moving
- "copy file.txt to backup.txt"
- "move old.txt to archive/old.txt"
""",
            category=KnowledgeCategory.PROCEDURE,
            summary="How to perform file operations",
            keywords=["files", "read", "write", "create", "delete", "copy", "move"],
            tags=["files", "operations", "guide"],
        ))
    
    def add_article(self, article: KnowledgeArticle):
        """Add a knowledge article."""
        with self._lock:
            self._articles[article.article_id] = article
            self._by_category[article.category].append(article.article_id)
        
        # Index the article
        self._indexer.index_document(
            doc_id=article.article_id,
            title=article.title,
            content=article.content,
            doc_type="knowledge_article",
            metadata={
                "category": article.category.value,
                "tags": article.tags,
            },
        )
    
    def get_article(self, article_id: str) -> Optional[KnowledgeArticle]:
        """Get an article by ID."""
        with self._lock:
            article = self._articles.get(article_id)
            if article:
                article.view_count += 1
            return article
    
    def search_articles(self, query: str, limit: int = 10) -> List[KnowledgeArticle]:
        """Search knowledge articles."""
        results = self._indexer.search(query, doc_type="knowledge_article", limit=limit)
        
        articles = []
        with self._lock:
            for result in results:
                article = self._articles.get(result.source_id)
                if article:
                    articles.append(article)
        
        return articles
    
    def get_articles_by_category(
        self,
        category: KnowledgeCategory
    ) -> List[KnowledgeArticle]:
        """Get articles by category."""
        with self._lock:
            article_ids = self._by_category.get(category, [])
            return [self._articles[aid] for aid in article_ids if aid in self._articles]
    
    def rate_article(self, article_id: str, rating: float):
        """Rate an article (1-5 scale)."""
        with self._lock:
            article = self._articles.get(article_id)
            if article:
                # Update running average
                total = article.rating * article.rating_count + rating
                article.rating_count += 1
                article.rating = total / article.rating_count
    
    def get_popular_articles(self, limit: int = 10) -> List[KnowledgeArticle]:
        """Get most popular articles by view count."""
        with self._lock:
            articles = sorted(
                self._articles.values(),
                key=lambda a: a.view_count,
                reverse=True
            )
            return articles[:limit]
    
    def get_top_rated_articles(self, limit: int = 10) -> List[KnowledgeArticle]:
        """Get top-rated articles."""
        with self._lock:
            # Filter articles with at least 5 ratings
            rated = [a for a in self._articles.values() if a.rating_count >= 5]
            rated.sort(key=lambda a: a.rating, reverse=True)
            return rated[:limit]


# ============================================================================
# Main Integration Class
# ============================================================================

class DocumentationAndKnowledge:
    """
    Main integration class for documentation and knowledge management.
    
    Integrates all Phase 12 components:
    - Comprehensive Documentation (12.1)
    - Knowledge Base Integration (12.2)
    """
    
    def __init__(self):
        # Initialize indexer first (used by other components)
        self._indexer = DocumentationIndexer()
        
        # Phase 12.1: Comprehensive Documentation
        self._user_docs = UserDocumentationManager()
        self._developer_docs = DeveloperDocumentationManager()
        self._system_docs = SystemDocumentationManager()
        
        # Phase 12.2: Knowledge Base Integration
        self._contextual_help = ContextualHelpSystem(self._indexer)
        self._knowledge_base = KnowledgeBaseManager(self._indexer)
        
        # Index existing documentation
        self._index_all_documentation()
    
    def _index_all_documentation(self):
        """Index all documentation for search."""
        # Index tutorials
        for tutorial in self._user_docs.list_tutorials():
            self._indexer.index_document(
                doc_id=tutorial.tutorial_id,
                title=tutorial.title,
                content=tutorial.description + " " + " ".join(
                    step.get("content", "") for step in tutorial.steps
                ),
                doc_type="tutorial",
                metadata={"difficulty": tutorial.difficulty.value},
            )
        
        # Index FAQs
        for faq in self._user_docs._faqs.values():
            self._indexer.index_document(
                doc_id=faq.faq_id,
                title=faq.question,
                content=faq.answer,
                doc_type="faq",
                metadata={"category": faq.category},
            )
        
        # Index system docs
        for doc in self._system_docs.get_deployment_documentation():
            self._indexer.index_document(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                doc_type="deployment",
            )
        
        for doc in self._system_docs.get_configuration_documentation():
            self._indexer.index_document(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                doc_type="configuration",
            )
    
    # =========================================================================
    # Unified Search
    # =========================================================================
    
    def search(
        self,
        query: str,
        doc_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """Search across all documentation."""
        if doc_types:
            results = []
            for doc_type in doc_types:
                results.extend(self._indexer.search(query, doc_type=doc_type, limit=limit))
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:limit]
        
        return self._indexer.search(query, limit=limit)
    
    def search_and_summarize(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search and provide a summarized response."""
        results = self.search(query, limit=limit)
        
        return {
            "query": query,
            "total_results": len(results),
            "results": [r.to_dict() for r in results],
            "summary": self._generate_summary(results),
        }
    
    def _generate_summary(self, results: List[SearchResult]) -> str:
        """Generate a summary from search results."""
        if not results:
            return "No relevant documentation found."
        
        summaries = []
        for result in results[:3]:
            summaries.append(f"- {result.title}: {result.content_preview[:100]}...")
        
        return f"Found {len(results)} relevant documents:\n" + "\n".join(summaries)
    
    # =========================================================================
    # Tutorial Operations
    # =========================================================================
    
    def get_tutorial(self, tutorial_id: str) -> Optional[Tutorial]:
        """Get a tutorial by ID."""
        return self._user_docs.get_tutorial(tutorial_id)
    
    def list_tutorials(
        self,
        difficulty: Optional[TutorialDifficulty] = None
    ) -> List[Tutorial]:
        """List available tutorials."""
        return self._user_docs.list_tutorials(difficulty=difficulty)
    
    def get_tutorial_step(
        self,
        tutorial_id: str,
        step_number: int
    ) -> Optional[Dict[str, Any]]:
        """Get a specific tutorial step."""
        tutorial = self._user_docs.get_tutorial(tutorial_id)
        if tutorial and 0 < step_number <= len(tutorial.steps):
            return tutorial.steps[step_number - 1]
        return None
    
    # =========================================================================
    # FAQ Operations
    # =========================================================================
    
    def search_faqs(self, query: str) -> List[FAQEntry]:
        """Search FAQs."""
        return self._user_docs.search_faqs(query)
    
    def get_faq(self, faq_id: str) -> Optional[FAQEntry]:
        """Get a FAQ entry."""
        return self._user_docs.get_faq(faq_id)
    
    def mark_faq_helpful(self, faq_id: str, helpful: bool):
        """Mark FAQ as helpful."""
        self._user_docs.mark_faq_helpful(faq_id, helpful)
    
    # =========================================================================
    # Troubleshooting Operations
    # =========================================================================
    
    def find_troubleshooting(
        self,
        error_message: Optional[str] = None,
        symptom: Optional[str] = None
    ) -> List[TroubleshootingGuide]:
        """Find troubleshooting guides."""
        return self._user_docs.find_troubleshooting_guide(
            error_message=error_message,
            symptom=symptom
        )
    
    # =========================================================================
    # Help Operations
    # =========================================================================
    
    def get_contextual_help(
        self,
        context: str,
        level: HelpLevel = HelpLevel.STANDARD
    ) -> Optional[ContextualHelp]:
        """Get contextual help."""
        return self._contextual_help.get_contextual_help(context, level)
    
    def get_error_help(self, error_message: str) -> Optional[Dict[str, Any]]:
        """Get help for an error."""
        return self._contextual_help.get_error_help(error_message)
    
    def get_command_examples(self, command: str) -> List[str]:
        """Get examples for a command."""
        return self._contextual_help.get_command_examples(command)
    
    def get_examples_for_query(self, query: str) -> List[str]:
        """Get relevant examples for a query."""
        return self._contextual_help.get_examples_for_query(query)
    
    def get_progressive_help(
        self,
        topic: str,
        current_level: HelpLevel
    ) -> Tuple[str, HelpLevel]:
        """Get progressively more detailed help."""
        return self._contextual_help.get_progressive_help(topic, current_level)
    
    # =========================================================================
    # Bookmark Operations
    # =========================================================================
    
    def add_bookmark(
        self,
        user_id: str,
        doc_type: str,
        doc_id: str,
        title: str,
        notes: str = ""
    ) -> Bookmark:
        """Add a bookmark."""
        return self._contextual_help.add_bookmark(
            user_id, doc_type, doc_id, title, notes
        )
    
    def get_bookmarks(self, user_id: str) -> List[Bookmark]:
        """Get user bookmarks."""
        return self._contextual_help.get_bookmarks(user_id)
    
    def remove_bookmark(self, user_id: str, bookmark_id: str) -> bool:
        """Remove a bookmark."""
        return self._contextual_help.remove_bookmark(user_id, bookmark_id)
    
    # =========================================================================
    # Knowledge Base Operations
    # =========================================================================
    
    def search_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeArticle]:
        """Search knowledge base."""
        return self._knowledge_base.search_articles(query, limit)
    
    def get_knowledge_article(self, article_id: str) -> Optional[KnowledgeArticle]:
        """Get a knowledge article."""
        return self._knowledge_base.get_article(article_id)
    
    def get_articles_by_category(
        self,
        category: KnowledgeCategory
    ) -> List[KnowledgeArticle]:
        """Get articles by category."""
        return self._knowledge_base.get_articles_by_category(category)
    
    def rate_article(self, article_id: str, rating: float):
        """Rate a knowledge article."""
        self._knowledge_base.rate_article(article_id, rating)
    
    def get_popular_articles(self, limit: int = 10) -> List[KnowledgeArticle]:
        """Get popular articles."""
        return self._knowledge_base.get_popular_articles(limit)
    
    # =========================================================================
    # Documentation Management
    # =========================================================================
    
    def add_documentation(
        self,
        title: str,
        content: str,
        doc_type: DocumentationType,
        **kwargs
    ) -> DocumentationEntry:
        """Add new documentation."""
        doc = DocumentationEntry(
            doc_id=f"doc_{uuid.uuid4().hex[:8]}",
            title=title,
            content=content,
            doc_type=doc_type,
            **kwargs
        )
        
        # Add to appropriate manager
        if doc_type in [DocumentationType.USER_GUIDE, DocumentationType.TUTORIAL]:
            self._user_docs.add_user_documentation(doc)
        elif doc_type in [DocumentationType.API_REFERENCE, DocumentationType.DEVELOPER_GUIDE]:
            self._developer_docs.add_architecture_documentation(doc)
        elif doc_type in [DocumentationType.DEPLOYMENT, DocumentationType.CONFIGURATION]:
            self._system_docs.add_deployment_documentation(doc)
        
        # Index the document
        self._indexer.index_document(
            doc_id=doc.doc_id,
            title=doc.title,
            content=doc.content,
            doc_type=doc_type.value,
        )
        
        return doc
    
    def add_knowledge_article(
        self,
        title: str,
        content: str,
        category: KnowledgeCategory,
        **kwargs
    ) -> KnowledgeArticle:
        """Add a knowledge article."""
        article = KnowledgeArticle(
            article_id=f"kb_{uuid.uuid4().hex[:8]}",
            title=title,
            content=content,
            category=category,
            **kwargs
        )
        
        self._knowledge_base.add_article(article)
        return article
    
    # =========================================================================
    # Property Accessors
    # =========================================================================
    
    @property
    def indexer(self) -> DocumentationIndexer:
        return self._indexer
    
    @property
    def user_docs(self) -> UserDocumentationManager:
        return self._user_docs
    
    @property
    def developer_docs(self) -> DeveloperDocumentationManager:
        return self._developer_docs
    
    @property
    def system_docs(self) -> SystemDocumentationManager:
        return self._system_docs
    
    @property
    def contextual_help(self) -> ContextualHelpSystem:
        return self._contextual_help
    
    @property
    def knowledge_base(self) -> KnowledgeBaseManager:
        return self._knowledge_base


# ============================================================================
# Global Instance and Accessor
# ============================================================================

_documentation_knowledge_instance: Optional[DocumentationAndKnowledge] = None
_instance_lock = threading.Lock()


def get_documentation_and_knowledge() -> DocumentationAndKnowledge:
    """Get the global DocumentationAndKnowledge instance."""
    global _documentation_knowledge_instance
    
    if _documentation_knowledge_instance is None:
        with _instance_lock:
            if _documentation_knowledge_instance is None:
                _documentation_knowledge_instance = DocumentationAndKnowledge()
    
    return _documentation_knowledge_instance


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "DocumentationType",
    "DocumentationFormat",
    "HelpLevel",
    "SearchRelevance",
    "DocumentationStatus",
    "KnowledgeCategory",
    "TutorialDifficulty",
    
    # Data Classes
    "DocumentationEntry",
    "Tutorial",
    "FAQEntry",
    "TroubleshootingGuide",
    "APIDocumentation",
    "KnowledgeArticle",
    "SearchResult",
    "ContextualHelp",
    "HelpHistoryEntry",
    "Bookmark",
    "DocumentationMetrics",
    
    # Phase 12.1: Comprehensive Documentation
    "UserDocumentationManager",
    "DeveloperDocumentationManager",
    "SystemDocumentationManager",
    
    # Phase 12.2: Knowledge Base Integration
    "DocumentationIndexer",
    "ContextualHelpSystem",
    "KnowledgeBaseManager",
    
    # Main Integration Class
    "DocumentationAndKnowledge",
    
    # Global Accessor
    "get_documentation_and_knowledge",
]
